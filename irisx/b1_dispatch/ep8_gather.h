// ep8_gather.h
// ================================================================================================
// EP8 multi-source remote A-tile gather + fp8 dequant for fused MoE expert-GEMM.
//
// PROBLEM (vs V4).  V4 gathers each BM x BK A-tile from a SINGLE remote rank (g.src_rank): every
// row of the tile lives on the same source GPU.  In real EP8 MoE the rows packed into ONE expert's
// M-region were routed there from up to 8 DIFFERENT source ranks (top-k=8, EP_SIZE=8).  So a single
// BM-tall A-tile can straddle a rank boundary: rows [r0..ra) come from rank 2, rows [ra..rb) from
// rank 5, etc.  This header generalizes V4's gather to a MULTI-SOURCE gather driven by the shared
// `route_segment[]` ABI, while preserving V4's exact fp8-e4m3 uint4 load + per-128 fp32 scale +
// __HIP_E4M3 dequant into the swizzled bf16 ST_A tile.
//
// TWO PATHS (selected per M-tile, once, before the K loop):
//
//   PATH 2 (FAST PATH) — tile_is_single_source(): true when exactly ONE segment covers all valid
//     rows of the tile.  Then the whole tile maps to one (src_rank, src_row_begin) with a constant
//     row offset.  We run a V4-IDENTICAL gather loop (one src_rank, one base row) plus a local
//     short-circuit when src_rank == cur_rank (direct HBM, no translate / no XGMI).  Zero per-row
//     branching; this is the common case (most expert tiles are dominated by one source).
//
//   PATH 1 (SEGMENT ITERATOR) — general case: the tile straddles >=2 source ranks.  We build a
//     compact per-tile map `row_seg[BM]` ONCE in shared memory: for each dst row in the tile,
//     row_seg[i] = index of the route_segment that owns it, or SEG_NONE (-1) for tail / unrouted
//     rows.  Segments are disjoint + sorted (ABI guarantee) so this map is filled with no races
//     (each segment writes a disjoint contiguous span).  The per-element gather then does ONE
//     lookup per row -> (src_rank, src_row) and reads the segment's source metadata once; tail rows
//     load the zero-sentinel uint4 so unrouted rows dequant to exactly 0 (provable remote gather).
//
// MEMORY ORDER.  The gather is READ-ONLY of remote activations.  The happens-before that makes the
// remote A bytes visible is the HOST-SIDE iris.barrier() between the producer ranks finishing their
// activation write and the consumer launching this kernel (hipDeviceSynchronize + MPI_Barrier).
// That barrier is a full system fence on every rank, so inside the kernel a plain ctx.load (relaxed
// deref) is correct — no per-load acquire needed.  For callers that instead want IN-LAUNCH
// producer/consumer handoff (no host barrier between write and read), we provide optional
// system-scope release/acquire helpers (release_segment_ready / acquire_segment_ready) built on
// IRIS atomic_store/atomic_load with memory_scope_system.
//
// REUSE.  gather_dequant_A_tile_multisource() has the SAME signature shape as V4's
// gather_dequant_A_tile<VEC> plus a `seg` view, so V5 can drop it into V4's producer prologue/loop
// unchanged.  build_row_seg_map() is called once per M-tile by the consumer/producer warps.
// ================================================================================================
#pragma once

#include <iris/iris.hpp>
#include <hip/hip_fp8.h>

namespace ep8_gather {

using fp8_t = __hip_fp8_storage_t;        // unsigned char, 1 byte
static constexpr int   QGROUP   = 128;    // per-128-group fp8 block-scale quant group
static constexpr int SEG_NONE = -1;   // row_seg sentinel: dst row has no source segment
                                      // (int, NOT signed char: absolute seg index can exceed 127 with
                                      //  many experts/segments -> signed-char overflow corrupted rows)

// ------------------------------------------------------------------------------------------------
// Shared metadata ABI (from AGENT_COMMON.md §3).  Consumed verbatim; do NOT redefine in callers.
// One contiguous run of rows for one expert coming from one source rank.
// ------------------------------------------------------------------------------------------------
struct route_segment {
    int expert_id;       // local expert index [0,32)
    int src_rank;        // owning rank of the activation rows [0,8)
    int src_row_begin;   // first row in the source rank's activation buffer
    int dst_row_begin;   // first row in this expert's packed output region
    int row_count;       // number of contiguous rows
};

// ------------------------------------------------------------------------------------------------
// A lightweight, kernel-side view of one M-tile's slice of route_segment[].
//   segs        : pointer to the (LOCAL, replicated) route_segment array on this rank's heap.
//   seg_begin   : first segment index relevant to this tile (expert_task.segment_begin).
//   seg_count   : number of segments for this tile (expert_task.segment_count).
//   tile_dst0   : dst_row_begin of this tile = expert_offsets[e] + m_tile_begin (absolute packed
//                 row of tile row 0).  Segment dst_row_begin are in the SAME absolute packed space.
//   valid_rows  : number of real rows in this tile (<= BM); rows [valid_rows,BM) are tail.
// ------------------------------------------------------------------------------------------------
struct seg_tile_view {
    const route_segment* segs;
    int seg_begin;
    int seg_count;
    int tile_dst0;
    int valid_rows;
};

// fp8 e4m3 (OCP, gfx950) -> float.  Matches V4's __HIP_E4M3 path exactly.
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// PATH 2 predicate.  True iff exactly one segment covers EVERY valid row of the tile, i.e. the tile
// is single-source and we can use the V4-identical fast path.  Sets *out_src_rank / *out_row_off so
// the fast path can map dst-row r (0..valid_rows) -> src_row = (tile_dst0 + r) + row_off, on
// out_src_rank.  row_off = src_row_begin - dst_row_begin (constant across the whole tile).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool tile_is_single_source(
        const seg_tile_view& v, int* out_src_rank, int* out_row_off) {
    if (v.seg_count != 1) return false;
    const route_segment s = v.segs[v.seg_begin];
    const int seg_lo = s.dst_row_begin - v.tile_dst0;          // first tile-local row of segment
    const int seg_hi = seg_lo + s.row_count;                   // one-past-last
    // Single segment must cover [0, valid_rows) of this tile.
    if (seg_lo > 0 || seg_hi < v.valid_rows) return false;
    *out_src_rank = s.src_rank;
    *out_row_off  = s.src_row_begin - s.dst_row_begin;         // src_row = dst_row + row_off
    return true;
}

// ------------------------------------------------------------------------------------------------
// PATH 1 setup.  Fill row_seg[BM]: per tile-local dst row, the segment INDEX (absolute into segs[])
// owning it, or SEG_NONE.  Called once per M-tile by ALL threads of the block; each segment writes
// a disjoint contiguous span [seg_lo, seg_hi) so there are no write races even with all threads
// participating.  tid/nthreads = caller's flat thread id / thread count (block-wide).
// BM is a compile-time tile height.
// ------------------------------------------------------------------------------------------------
template<int BM>
__device__ __forceinline__ void build_row_seg_map(
        int (&row_seg)[BM], const seg_tile_view& v, int tid, int nthreads) {
    // 1) default everything to SEG_NONE (tail / unrouted -> zero sentinel).
    for (int i = tid; i < BM; i += nthreads) row_seg[i] = SEG_NONE;
    __syncthreads();
    // 2) stamp each segment's disjoint span with its absolute segment index.
    //    Segments are disjoint + sorted per ABI, so spans never overlap.
    for (int si = 0; si < v.seg_count; ++si) {
        const route_segment s = v.segs[v.seg_begin + si];
        int lo = s.dst_row_begin - v.tile_dst0;               // clamp to this tile
        int hi = lo + s.row_count;
        if (lo < 0) lo = 0;
        if (hi > BM) hi = BM;
        for (int r = lo + tid; r < hi; r += nthreads)
            row_seg[r] = v.seg_begin + si;     // absolute index into segs[] (int: no 127 overflow)
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------
// Core: gather + dequant one BM x BK A-tile into the swizzled shared bf16 tile `dst`.
// ST_A is the HK shared-tile type (st_bf<BM,BK,...>); ELEM is bf16.  The swizzle / sub-tile math is
// IDENTICAL to V4's gather_dequant_A_tile so the produced LDS layout is bit-compatible.
//
// `row_seg` is the Path-1 map (ignored on the fast path).  `fast_single` selects Path 2: when true,
// fast_src_rank / fast_row_off describe the single-source mapping and row_seg is not consulted.
//
// VEC = elements per vectorized fp8 load (16 -> uint4, as in V4).  block_row is the tile's absolute
// dst row 0 == v.tile_dst0.  a_base / sc_base are the LOCAL pointers to this rank's activation +
// scale heap buffers (symmetric-heap: same offset on every rank); ctx.load(ptr, src_rank) reads the
// SAME offset on src_rank.  When src_rank == ctx.cur_rank() we deref locally (no translate, no XGMI).
// ------------------------------------------------------------------------------------------------
template<typename ST_A, typename ELEM, int BM, int BK, int VEC>
__device__ __forceinline__ void gather_dequant_A_tile_multisource(
        ST_A& dst,
        const int (&row_seg)[BM],
        const seg_tile_view& v,
        const fp8_t* a_base,        // local activation buffer base (heap, M_src x K)
        const float* sc_base,       // local scale buffer base       (heap, M_src x K/128)
        int K,                      // activation K dim
        int Msrc,                   // rows in the source activation buffer (per rank)
        int k0,                     // K offset of this K-tile (= tile * BK)
        iris::iris_device_view ctx,
        int tid, int nthreads,
        bool fast_single, int fast_src_rank, int fast_row_off) {

    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    const int NG = K / QGROUP;

    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = BM * CHUNKS_PER_ROW;
    const int cur_rank = ctx.cur_rank();

    for (int ci = tid; ci < total_chunks; ci += nthreads) {
        const int r  = ci / CHUNKS_PER_ROW;          // tile-local dst row [0,BM)
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;  // K offset within the tile
        const int gk = k0 + kc;

        // ---- resolve (src_rank, src_row) for this dst row ----
        int src_rank, src_row;
        bool valid;
        if (fast_single) {
            // PATH 2: constant mapping, V4-style. Valid iff within valid_rows.
            valid    = (r < v.valid_rows);
            src_rank = fast_src_rank;
            src_row  = (v.tile_dst0 + r) + fast_row_off;   // == src_row_begin + (r - seg_lo)
        } else {
            // PATH 1: one lookup. SEG_NONE -> tail/unrouted -> zero sentinel.
            const int sidx = row_seg[r];
            if (sidx == SEG_NONE) {
                valid = false; src_rank = cur_rank; src_row = 0;
            } else {
                const route_segment s = v.segs[(int)sidx];   // metadata read once per (row,chunk)
                const int local_dst = (v.tile_dst0 + r) - s.dst_row_begin;
                src_rank = s.src_rank;
                src_row  = s.src_row_begin + local_dst;
                valid    = true;
            }
        }

        // ---- V4-identical fp8 uint4 load (local deref if src_rank==cur_rank, else XGMI) ----
        uint4 packed;
        if (valid && src_row < Msrc && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)src_row * K + gk;
            const uint4* vptr = reinterpret_cast<const uint4*>(aptr);
            packed = (src_rank == cur_rank) ? *vptr : ctx.load(vptr, src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);             // ZERO SENTINEL
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        // ---- per-128-group fp32 scale (same source mapping) ----
        const int grp = gk / QGROUP;
        float scale = 1.0f;
        if (valid && src_row < Msrc && grp < NG) {
            const float* sptr = sc_base + (size_t)src_row * NG + grp;
            scale = (src_rank == cur_rank) ? *sptr : ctx.load(sptr, src_rank);
        }

        // ---- dequant into swizzled ST_A (bit-identical to V4) ----
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            ELEM val = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_A::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_A::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(ELEM);
            *reinterpret_cast<ELEM*>(base + intra_byte) = val;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// OPTIONAL in-launch producer/consumer handoff (only when there is NO host iris.barrier() between
// the remote write and this read).  System-scope so the flag and the data it guards are coherent
// across GPUs over XGMI.
//
//   release_segment_ready: producer rank, AFTER writing its activation rows + a system fence, sets
//     a per-(src_rank) "ready" flag on the CONSUMER's heap with release ordering.
//   acquire_segment_ready: consumer spins until the flag for src_rank is observed with acquire
//     ordering, establishing happens-before with the producer's writes before any ctx.load.
// `flags` is a local pointer (symmetric heap) to a world_size-sized int array.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void release_segment_ready(
        iris::iris_device_view ctx, int* flags, int consumer_rank, int my_src_rank) {
    ctx.fence<iris::memory_scope_system>(iris::memory_order_release);
    ctx.atomic_store<int, iris::memory_scope_system>(
        flags + my_src_rank, 1, consumer_rank, iris::memory_order_release);
}

__device__ __forceinline__ void acquire_segment_ready(
        iris::iris_device_view ctx, int* flags, int src_rank) {
    const int me = ctx.cur_rank();
    while (ctx.atomic_load<int, iris::memory_scope_system>(
               flags + src_rank, me, iris::memory_order_acquire) == 0) { /* spin */ }
}

}  // namespace ep8_gather
