// b1_dispatch / kernel.cpp
// ================================================================================================
// B1-DISPATCH V0 — the production-shaped main-line MoE expert kernel, by COMPOSITION.
//
// B1-dispatch is the route-aware EP8 analog of B1-copy: real top-k routing, 32 local experts, and
// MULTI-SOURCE gather (each expert's rows were routed there from up to 8 different ranks), but kept
// SERIAL (no comm/compute overlap yet — overlap is a later experiment only if it beats this).
//
// It is the back-to-back composition of TWO ALREADY-VERIFIED components (the #1 rule of this task is
// to REUSE them, NOT rewrite them — three prior candidates P1/P2/P3 all failed because they rewrote
// the gather/GEMM from scratch and produced RMS~1.0 garbage + 10-25x slow):
//
//   PHASE 1  dispatch_gather_pack(...)  -> kernel `gather_pack_kernel`
//     Multi-source EP8 gather of each expert's rows from their SOURCE ranks (route_segments) ONCE
//     into a LOCAL expert-major packed fp8 + scale buffer (the v5_grouped layout: BM-padded per-
//     expert regions; padding rows stay zero). A crosses XGMI EXACTLY HERE, exactly once.
//       - ROW RESOLUTION (which (src_rank, src_row) feeds each packed dst row) is REUSED VERBATIM
//         from ep8_gather.h: seg_tile_view, tile_is_single_source(), build_row_seg_map() — the
//         VERIFIED multi-source routing brain (Agent 03: np=8, RMS 0.00167, PASSED).
//       - DATA MOVEMENT (the actual byte copy) is REUSED VERBATIM from harness_kernels.cpp's B1
//         gather_once_kernel body: vectorized uint4 fp8-byte remote load + scalar fp32 scale load,
//         writing a PLAIN ROW-MAJOR local fp8 buffer (no dequant, no swizzle here — exactly like
//         B1-copy; dequant happens inside the phase-2 GEMM, identical to B0/v5).
//     The ONLY new glue is the loop that walks packed rows, calls the verified resolver per row, and
//     copies raw fp8+scale into the packed buffer (plus the zero-sentinel for unrouted/tail rows).
//
//   PHASE 2  grouped_gemm(...)  -> kernel `micro_tk_baseline` (COPIED VERBATIM FROM v5_grouped)
//     The VERIFIED grouped 32-expert GEMM (Agent 02 V5: all 5 routes PASSED, RMS 0.00331). We use
//     the SERIAL/baseline grouped path (`micro_tk_baseline`, the clean two-phase grouped GEMM — NOT
//     the V4-style direct-pull overlap `micro_tk`). Its internal A gather reads via
//     ctx.load(..., g.src_rank); B1-dispatch sets src_rank = CONSUMER (the rank that just packed A
//     locally) so that load is a LOCAL HBM deref — A does NOT re-cross XGMI. This is the whole point:
//     gather/pack once (phase 1), then a local grouped GEMM (phase 2).
//
// HANDOFF (phase1 output == phase2 input, byte-exact same layout):
//   phase1 writes  A_packed_fp8[Mpacked, K]  (fp8 bytes via bf16 view) + A_packed_sc[Mpacked, NG]
//   phase2 reads   the same two buffers as its g.a / g.sc. Mpacked = sum_e padded(M_e) (BM-padded
//   per-expert prefix from build_tasks.build_packed_layout). Token-major scales [Mpacked,NG] — the
//   SAME order v5_grouped expects (production's group-major scale transpose is a later concern,
//   documented in FMOE_LAYOUT.md; V0 keeps phase1 and phase2 in the SAME order so they compose).
//
// NOTHING below the two phase wrappers is new GEMM/gather logic: the gather resolver lives in the
// (untouched) ep8_gather.h, the GEMM body is a verbatim copy of v5_grouped/kernel.cpp, and the byte
// copy is a verbatim copy of harness_kernels.cpp's gather_once body. The new code is exactly: the
// gather_pack_kernel driver loop + the two pybind wrappers.
// ================================================================================================
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include "ep8_gather.h"     // VERIFIED multi-source row resolver (Agent 03) — included, not edited.
#include <cstdio>
using namespace kittens;

// ep8_gather.h's namespace holds the VERIFIED resolver + the route_segment ABI. Pull the pieces we
// reuse verbatim into this TU.
using ep8_gather::route_segment;
using ep8_gather::seg_tile_view;
using ep8_gather::tile_is_single_source;
using ep8_gather::build_row_seg_map;
using ep8_gather::SEG_NONE;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte (same as both components)
static constexpr int QGROUP = 128;   // per-128-group fp8 block-scale quant group

// ================================================================================================
// PHASE 1 — multi-source gather/pack/quant ONCE into a LOCAL expert-major packed fp8 + scale buffer.
//
// Grid: one block per BM-tile (blockIdx.x), matching the tile granularity that the verified resolver
// (seg_tile_view / build_row_seg_map / tile_is_single_source) operates on. Each block:
//   1. loads its tile's seg_tile_view from tilemeta (same fields the ep8_gather probe loads);
//   2. picks Path 2 (single-source fast) vs Path 1 (segment iterator) with the VERIFIED predicate;
//   3. for each valid packed row in the tile, resolves (src_rank, src_row) with the VERIFIED logic
//      and copies the raw fp8 K bytes (uint4) + NG fp32 scales from that source into the local
//      packed buffer — the EXACT B1-copy gather_once movement, but multi-source + zero-sentinel.
// Unrouted / tail (SEG_NONE) rows are written as zeros (fp8 0x00 dequants to 0; scale 0) so the
// padding/zero-sentinel rows produce exactly-zero GEMM output (the no-contamination guarantee).
// ================================================================================================
#ifndef GP_BM
#define GP_BM 64                       // packed-row tile height (matches build_tasks BM / ep8 probe)
#endif
#ifndef GP_THREADS
#define GP_THREADS 256
#endif

struct gatherpack_globals {
    gl<bf16,  -1, -1, -1, -1> a_src;    // [Msrc, K/2] fp8-as-bf16 on EACH source rank (IRIS heap)
    gl<float, -1, -1, -1, -1> sc_src;   // [Msrc, K/128] fp32 scales on each source rank (IRIS heap)
    gl<bf16,  -1, -1, -1, -1> a_dst;    // [Mpacked, K/2] LOCAL packed fp8-as-bf16 (IRIS heap)
    gl<float, -1, -1, -1, -1> sc_dst;   // [Mpacked, K/128] LOCAL packed scales (IRIS heap)
    gl<int,   -1, -1, -1, -1> seg;      // route_segment[] flattened [Nseg,5] (local)
    gl<int,   -1, -1, -1, -1> tilemeta; // per-tile [Ntile,4]: seg_begin, seg_count, tile_dst0, valid_rows
    iris::iris_device_view iris_ctx;
    int Msrc, Mpacked, K, Nseg, Ntile;
    hipStream_t stream;
    dim3 grid()  { return dim3(Ntile > 0 ? Ntile : 1); }   // one block per packed BM-tile
    dim3 block() { return dim3(GP_THREADS); }
};

__global__ __launch_bounds__(GP_THREADS, 1)
void gather_pack_kernel(gatherpack_globals g) {
    __shared__ int row_seg[GP_BM];          // Path-1 per-row segment-index map (int: absolute seg
                                            // index exceeds 127 with many experts -> signed-char overflow)

    const int m_tile   = blockIdx.x;
    const int tid      = threadIdx.x;
    const int nthreads = GP_THREADS;
    const int K        = g.K;
    const int NG       = K / QGROUP;
    const int cur_rank = g.iris_ctx.cur_rank();
    iris::iris_device_view ctx = g.iris_ctx;

    if (m_tile >= g.Ntile) return;

    // ---- load this tile's seg_tile_view from tilemeta (IDENTICAL to ep8_gather/kernel.cpp) ----
    seg_tile_view v;
    v.segs       = reinterpret_cast<const route_segment*>(&g.seg[{0, 0, 0, 0}]);
    v.seg_begin  = g.tilemeta[{0, 0, m_tile, 0}];
    v.seg_count  = g.tilemeta[{0, 0, m_tile, 1}];
    v.tile_dst0  = g.tilemeta[{0, 0, m_tile, 2}];
    v.valid_rows = g.tilemeta[{0, 0, m_tile, 3}];

    const fp8_t* a_src_base = reinterpret_cast<const fp8_t*>(&g.a_src[{0, 0, 0, 0}]);
    const float* sc_src_base = &g.sc_src[{0, 0, 0, 0}];
    fp8_t* a_dst_base  = reinterpret_cast<fp8_t*>(&g.a_dst[{0, 0, 0, 0}]);
    float* sc_dst_base = &g.sc_dst[{0, 0, 0, 0}];

    // ---- choose path with the VERIFIED predicate (ep8_gather.h) ----
    int fast_src_rank = 0, fast_row_off = 0;
    const bool fast = tile_is_single_source(v, &fast_src_rank, &fast_row_off);
    if (!fast) build_row_seg_map<GP_BM>(row_seg, v, tid, nthreads);   // VERIFIED Path-1 map builder
    else       __syncthreads();

    // ---- per row in this tile: resolve (src_rank, src_row) [VERIFIED logic, lifted from
    //      ep8_gather.h::gather_dequant_A_tile_multisource], then COPY RAW fp8+scale
    //      [VERIFIED movement, lifted from harness gather_once_kernel]. ----
    const int chunks_per_row = K / 16;                 // 16 fp8 bytes (uint4) per thread, as in B1
    const long total_chunks  = (long)GP_BM * chunks_per_row;

    for (long ci = tid; ci < total_chunks; ci += nthreads) {
        const int r  = (int)(ci / chunks_per_row);     // tile-local packed row [0,GP_BM)
        const int kc = (int)(ci % chunks_per_row) * 16;// K byte offset within the row
        const int dst_row = v.tile_dst0 + r;           // GLOBAL packed dst row
        if (dst_row >= g.Mpacked) continue;

        // ---- resolve source for this row (VERBATIM resolution from ep8_gather multisource) ----
        int src_rank, src_row;
        bool valid;
        if (fast) {                                    // PATH 2: single-source constant mapping
            valid    = (r < v.valid_rows);
            src_rank = fast_src_rank;
            src_row  = (v.tile_dst0 + r) + fast_row_off;
        } else {                                       // PATH 1: one lookup; SEG_NONE -> zero
            const int sidx = row_seg[r];
            if (sidx == SEG_NONE) { valid = false; src_rank = cur_rank; src_row = 0; }
            else {
                const route_segment s = v.segs[(int)sidx];
                const int local_dst = (v.tile_dst0 + r) - s.dst_row_begin;
                src_rank = s.src_rank;
                src_row  = s.src_row_begin + local_dst;
                valid    = true;
            }
        }

        // ---- COPY the raw fp8 16-byte chunk (B1 gather_once movement: uint4 remote load -> store).
        //      Local short-circuit when the source is THIS rank (no XGMI), as in ep8_gather. ----
        uint4 vbytes;
        if (valid && src_row < g.Msrc && (kc + 16) <= K) {
            const uint4* sp = reinterpret_cast<const uint4*>(a_src_base + (size_t)src_row * K + kc);
            vbytes = (src_rank == cur_rank) ? *sp : ctx.load(sp, src_rank);
        } else {
            vbytes = make_uint4(0u, 0u, 0u, 0u);       // ZERO SENTINEL: unrouted/tail/padding -> 0
        }
        *reinterpret_cast<uint4*>(a_dst_base + (size_t)dst_row * K + kc) = vbytes;

        // ---- COPY the scales for the groups this chunk spans (1 group per 128 bytes). One scalar
        //      fp32 load per group, same source mapping (B1 movement). 16-byte chunk touches the
        //      single group kc/QGROUP. ----
        const int grp = kc / QGROUP;
        if (grp < NG) {
            float scale = 0.0f;
            if (valid && src_row < g.Msrc) {
                const float* spc = sc_src_base + (size_t)src_row * NG + grp;
                scale = (src_rank == cur_rank) ? *spc : ctx.load(spc, src_rank);
            }
            // many chunks map to the same group; the write is idempotent (same value) so racing
            // writes of identical data are safe and avoid a reduction/barrier.
            sc_dst_base[(size_t)dst_row * NG + grp] = scale;
        }
    }
}

void dispatch_gather_pack(gatherpack_globals g) {
    if (g.Ntile <= 0) return;
    gather_pack_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
// PHASE 2 — grouped 32-expert GEMM over the LOCAL packed buffer.
//
// EVERYTHING from here to the pybind block is COPIED VERBATIM FROM irisx/v5_grouped/kernel.cpp
// (Agent 02 V5, PASSED all 5 routes, RMS 0.00331). The only behavioral change is at the CALL SITE
// (example.py): grouped_gemm is invoked with src_rank = CONSUMER so the kernel's ctx.load reads the
// LOCAL packed A (no XGMI). The kernel SOURCE is unchanged. We expose ONLY the SERIAL baseline path
// (micro_tk_baseline); the fused/direct-pull micro_tk is intentionally not wired for V0.
// ================================================================================================

// flat task layout (keep in sync with build_tasks.py TASK_W / column order). [VERBATIM v5]
static constexpr int TASK_W = 6;
enum { T_EXPERT = 0, T_MBEGIN = 1, T_VALID = 2, T_NSUPER = 3, T_NSUB = 4, T_EROWBEG = 5 };

// ----------------------------- tile / block configuration --------------------------------------- [VERBATIM v5]
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef NSUB
#define NSUB 8                  // COMPILE-TIME max NSUB (host picks runtime nsub <= this)
#endif
#ifndef NUM_PRODUCER_WORKERS
#define NUM_PRODUCER_WORKERS 4
#endif
#ifndef NUM_CONSUMER_WORKERS
#define NUM_CONSUMER_WORKERS 4
#endif
#ifndef NSTAGE
#define NSTAGE 2            // shared-tile buffering depth (2 = double-buffer)
#endif

#define N_PER_BLOCK (NSUB * BN)
#define NUM_WARPS (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS)
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using PG = kittens::group<NUM_PRODUCER_WORKERS>;   // producer warp group (fast local B load)

// Shared tile types (bf16 — A is dequantized into bf16 before MFMA). [VERBATIM v5]
using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

struct micro_globals {
    gl<bf16,  -1, -1, -1, -1> a;       // [Mpacked, K/2]  (reinterpreted fp8 e4m3 [Mpacked,K])
    gl<float, -1, -1, -1, -1> sc;      // [Mpacked, K/128]  fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[E*N, K] local (expert-major), C[Mpacked, N] local
    gl<int,   -1, -1, -1, -1> tasks;   // [num_tasks, TASK_W] int32 flat work list
    iris::iris_device_view iris_ctx;
    int Mpacked, N, K, src_rank;
    int num_tasks;
    int nsub;                           // runtime NSUB chosen by host (<= compile-time NSUB)
    int fused;                          // 1 = fused grouped path, 0 = per-tile baseline path
    hipStream_t stream;
    dim3 grid()  { return dim3(num_tasks > 0 ? num_tasks : 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * (sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
    }
};

// fp8 -> float using OCP e4m3 (matches V1's __HIP_E4M3). [VERBATIM v5]
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// Remote fp8 gather + dequant of a BM x BK A-tile into the swizzled shared bf16 tile. [VERBATIM v5]
// (For B1-dispatch the caller passes src_rank == cur_rank so ctx.load is a LOCAL deref.)
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_tile(
        ST_A &dst, int tile, int block_row, int row_limit, int src_rank,
        const micro_globals &g, int warp_id, int laneid) {
    const int k0 = tile * BK;
    const int tid = (warp_id * kittens::WARP_THREADS) + laneid;
    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    const int K = g.K;
    const int NG = K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;

    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = BM * CHUNKS_PER_ROW;

    for (int ci = tid; ci < total_chunks; ci += NUM_PRODUCER_THREADS) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gr = block_row + r;                 // GLOBAL packed-A row
        const int gk = k0 + kc;

        uint4 packed;
        if (gr < row_limit && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        const int grp = (gk) / QGROUP;
        float scale = 1.0f;
        if (gr < row_limit && grp < NG) {
            const float* sptr = &g.sc[{0, 0, gr, grp}];
            scale = ctx.load(sptr, src_rank);
        }

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            bf16 val = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_A::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_A::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    }
}

// ----------------------------------------------------------------------------------------------- [VERBATIM v5]
// FUSED kernel (A-stationary, NSTAGE double-buffer) — B1-dispatch V1's phase 2. Gathers A[BM,BK]
// ONCE per K-tile and reuses it across all nsub N-subtiles (NSUB register accumulators), so A is read
// from the local packed buffer 1x per K-tile instead of nsub x (the baseline's redundant re-read).
// COPIED VERBATIM from v5_grouped/kernel.cpp::micro_tk (Agent 02, PASSED all 5 routes). For
// B1-dispatch the caller passes src_rank=CONSUMER so gather_dequant_A_tile's ctx.load is a LOCAL deref.
// -----------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    const int task = blockIdx.x;
    if (task >= g.num_tasks) return;

    const int* tk = &g.tasks[{0, 0, task, 0}];
    const int local_expert    = tk[T_EXPERT];
    const int m_tile_begin    = tk[T_MBEGIN];
    const int valid_rows      = tk[T_VALID];
    const int n_superblock    = tk[T_NSUPER];
    const int nsub            = tk[T_NSUB];
    const int expert_row_begin= tk[T_EROWBEG];

    const int block_row = expert_row_begin + m_tile_begin;
    const int row_limit = block_row + valid_rows;
    const int block_n0  = n_superblock * (nsub * BN);
    const int b_row0    = local_expert * g.N;
    const int n_tile0   = block_n0 / BN;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE]       = al.allocate<ST_A, NSTAGE>();
    ST_B (&Bs)[NSTAGE][NSUB] = al.allocate<ST_B, NSTAGE, NSUB>();

    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);
    const int b_tile_row0 = b_row0 / BN;

    constexpr int PREFETCH = NSTAGE - 1;

    if (is_producer) {
        #pragma unroll
        for (int s = 0; s < PREFETCH; ++s) {
            if (s < num_tiles) {
                gather_dequant_A_tile<16>(As[s], s, block_row, row_limit, src_rank, g, warp_id, laneid);
                for (int sub = 0; sub < nsub; ++sub)
                    PG::load<2, false>(Bs[s][sub], g.b, {0, 0, b_tile_row0 + n_tile0 + sub, s}, swizzled_offsets_B);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum[NSUB];
    if (is_consumer) {
        for (int sub = 0; sub < nsub; ++sub) zero(C_accum[sub]);
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile % NSTAGE;
        const int fetch = tile + PREFETCH;
        if (is_producer && fetch < num_tiles) {
            const int slot = fetch % NSTAGE;
            gather_dequant_A_tile<16>(As[slot], fetch, block_row, row_limit, src_rank, g, warp_id, laneid);
            for (int sub = 0; sub < nsub; ++sub)
                PG::load<2, false>(Bs[slot][sub], g.b, {0, 0, b_tile_row0 + n_tile0 + sub, fetch}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            load(a_frag, As[cur]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            for (int sub = 0; sub < nsub; ++sub) {
                rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
                auto b_sub = subtile_inplace<CONS_N, BK>(Bs[cur][sub], {cons_id, 0});
                load(b_frag, b_sub);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum[sub], a_frag, b_frag, C_accum[sub]);
                __builtin_amdgcn_s_setprio(0);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        for (int sub = 0; sub < nsub; ++sub) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            store(g.c, C_accum[sub], {0, 0, block_row / BM, out_col0 / CONS_N});
        }
    }
}

// ----------------------------------------------------------------------------------------------- [VERBATIM v5]
// BASELINE kernel (two-phase, NO overlap) — the SERIAL grouped GEMM (re-gathers A per N-subtile).
// Kept as the head-to-head reference; B1-dispatch V1 uses micro_tk (fused) above.
// -----------------------------------------------------------------------------------------------
using ST_A_b = st_bf<BM, BK, st_16x32_s>;
using ST_B_b = st_bf<BN, BK, st_16x32_s>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk_baseline(micro_globals g) {
    const int task = blockIdx.x;
    if (task >= g.num_tasks) return;

    const int* tk = &g.tasks[{0, 0, task, 0}];
    const int local_expert    = tk[T_EXPERT];
    const int m_tile_begin    = tk[T_MBEGIN];
    const int valid_rows      = tk[T_VALID];
    const int n_superblock    = tk[T_NSUPER];
    const int nsub            = tk[T_NSUB];
    const int expert_row_begin= tk[T_EROWBEG];

    const int block_row = expert_row_begin + m_tile_begin;
    const int row_limit = block_row + valid_rows;
    const int block_n0  = n_superblock * (nsub * BN);
    const int b_row0    = local_expert * g.N;
    const int n_tile0   = block_n0 / BN;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_b (&As)[1] = al.allocate<ST_A_b, 1>();
    ST_B_b (&Bs)[1] = al.allocate<ST_B_b, 1>();

    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);
    const int b_tile_row0 = b_row0 / BN;

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;

    for (int sub = 0; sub < nsub; ++sub) {
        rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum;
        zero(C_accum);
        for (int tile = 0; tile < num_tiles; ++tile) {
            if (is_producer) {
                gather_dequant_A_tile<16>(As[0], tile, block_row, row_limit, src_rank, g, warp_id, laneid);
                PG::load<2, false>(Bs[0], g.b, {0, 0, b_tile_row0 + n_tile0 + sub, tile}, swizzled_offsets_B);
                __builtin_amdgcn_s_waitcnt(0);
            }
            __syncthreads();
            if (is_consumer) {
                rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
                rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
                load(a_frag, As[0]);
                auto b_sub = subtile_inplace<CONS_N, BK>(Bs[0], {cons_id, 0});
                load(b_frag, b_sub);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum, a_frag, b_frag, C_accum);
                __builtin_amdgcn_s_setprio(0);
            }
            __builtin_amdgcn_sched_barrier(0);
            __syncthreads();
        }
        if (is_consumer) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N});
        }
        __syncthreads();
    }
}

void dispatch_grouped_gemm(micro_globals g) {
    if (g.num_tasks <= 0) return;   // empty route -> nothing to launch.
    if (g.fused) {                  // V1: A-stationary fused (gather A once per K-tile, reuse over NSUB)
        const unsigned long mem_size = g.dynamic_shared_memory();
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    } else {                        // V0: serial baseline (re-gathers A per N-subtile)
        const unsigned long mem_size = (unsigned long)(sizeof(ST_A_b) + sizeof(ST_B_b)) + 1024;
        hipFuncSetAttribute((void*)micro_tk_baseline, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk_baseline<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    }
}

// ================================================================================================
// PHASE 2 (ALT) — B0-class 8-wave ping-pong GROUPED GEMM  [Stage B wiring of grouped_b0]
//
// ADDITIVE: nothing above is changed. This is a SECOND phase-2 path the host can select by calling
// `grouped_gemm_b0(...)` instead of `grouped_gemm(...)`. It replaces micro_tk's 64x64 producer/
// consumer body (~32-69 TFLOP/s, only 4/8 waves MFMA) with the proven B0 256x256x64 8-wave ping-pong
// body (all 8 waves MFMA, ~183 TFLOP/s ceiling) from reference/v2_hk_expert_gemm/fmoe_expert_v2.cu.
//
// The GEMM body is that B0 kernel VERBATIM, with ONLY the per-block tile-base indices remapped to a
// per-task (expert, m_tile, n_tile, expert_row_begin) decode (the grouping idea from v5). It is a
// straight copy of irisx/grouped_b0/grouped_b0.cu::grouped_expert_gemm. All new names are B0_-prefixed
// so they do not collide with the v5 micro_tk macros/types above.
//
// Dataflow vs micro_tk: micro_tk dequants fp8->bf16 inside the producer warps. The B0 body wants bf16
// A in HBM, so this path runs a DEQUANT PREAMBLE (packed fp8 -> bf16 scratch) first, then the bf16
// 8-wave GEMM (== exactly how B0/B1-copy work). Both kernels are LOCAL (no IRIS) — the packed buffer
// is already on this rank from phase 1.
//
// REQUIREMENT: each expert's packed region must be padded to a multiple of B0_BM=256 so a 256-row
// block stays inside one expert (no cross-expert contamination; zero padding rows MFMA to 0). The
// host builds the matching task list with b1_dispatch/b0_tasks.py::build_b0_tasks(..., BM=256), and
// phase 1 must pack with BM=256 (vs the current GP_BM=64) — see BENCHMARKING_HANDOFF.md §5.
// ================================================================================================
static constexpr int B0_TASK_W = 4;
enum { B0_T_EXPERT = 0, B0_T_MTILE = 1, B0_T_NTILE = 2, B0_T_EROWBEG = 3 };
static constexpr int B0_BM = 256;        // packed-row padding granularity for the B0 path

// dequant preamble: packed fp8 [Mpacked,K] (+ sc [Mpacked,NG]) -> bf16 [Mpacked,K]. Padding rows are
// already 0 in the packed buffer (phase-1 zero-sentinel) so they dequant to 0 with no special-casing.
__global__ void dequant_packed_dense(const fp8_t* __restrict__ a_fp8,
                                     const float* __restrict__ a_sc,
                                     bf16* __restrict__ a_bf16, int Mpacked, int Kdim) {
    const int row = blockIdx.x;
    if (row >= Mpacked) return;
    const int NG = Kdim / QGROUP;
    const fp8_t* frow = a_fp8 + (size_t)row * Kdim;
    const float* srow = a_sc  + (size_t)row * NG;
    bf16* orow = a_bf16 + (size_t)row * Kdim;
    for (int h = threadIdx.x; h < Kdim; h += blockDim.x)
        orow[h] = (bf16)(fp8_to_f32(frow[h]) * srow[h / QGROUP]);
}

using B0G = kittens::group<8>;

template <int NN, int KK>
__global__ __launch_bounds__(512, 2)
void grouped_b0_gemm(const gl<bf16, -1, -1, -1, -1> A,   // [Mpacked, K] bf16 (dequanted)
                     const gl<bf16, -1, -1, -1, -1> B,   // [E*N,     K] bf16 expert-major
                     const gl<bf16, -1, -1, -1, -1> C,   // [Mpacked, N] bf16
                     const int* __restrict__ tasks, int num_tasks) {
    constexpr int WARPS_COL = 4, WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256, BLOCK_SIZE_COL = 256, BLOCK_K = 64;
    constexpr int k_iters = KK / BLOCK_K;
    constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2;
    constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2;
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW / 2;
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL / 2;

    using B0_ST_A = st_bf<HALF_BLOCK_SIZE_ROW, BLOCK_K, st_16x32_s>;
    using B0_ST_B = st_bf<HALF_BLOCK_SIZE_COL, BLOCK_K, st_16x32_s>;
    __shared__ B0_ST_A As[2][2];
    __shared__ B0_ST_B Bs[2][2];

    using B0_RT_A = rt_bf<REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s>;
    using B0_RT_B = rt_bf<REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s>;
    using B0_RT_C = rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s>;
    B0_RT_A a;
    B0_RT_B b0, b1;
    B0_RT_C cA, cB, cC, cD;

    const int task = blockIdx.x;
    if (task >= num_tasks) return;
    const int* tk = tasks + (size_t)task * B0_TASK_W;
    const int e   = tk[B0_T_EXPERT];
    const int mt  = tk[B0_T_MTILE];
    const int nt  = tk[B0_T_NTILE];
    const int ERB = tk[B0_T_EROWBEG];                 // multiple of B0_BM=256

    const int a_row_tile = ERB / 128 + mt * 2;        // 128-row A half-tile base
    const int b_row_tile = (e * NN) / 128 + nt * 2;   // 128-row B half-tile base (expert-major)
    const int c_row_tile = ERB / 64 + mt * 4;         // 64-row  C reg-tile base
    const int c_col_tile = nt * 8;                    // 32-col  C reg-tile base

    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);
    int tic = 0, toc = 1;

    uint32_t swizzled_offsets_A[64];
    uint32_t swizzled_offsets_B[64];
    B0G::prefill_swizzled_offsets(As[tic][0], A, swizzled_offsets_A);
    B0G::prefill_swizzled_offsets(Bs[tic][0], B, swizzled_offsets_B);

    zero(cA); zero(cB); zero(cC); zero(cD);

    B0G::load(Bs[tic][0], B, {0, 0, b_row_tile,     0}, swizzled_offsets_B);
    B0G::load(As[tic][0], A, {0, 0, a_row_tile,     0}, swizzled_offsets_A);
    B0G::load(Bs[tic][1], B, {0, 0, b_row_tile + 1, 0}, swizzled_offsets_B);
    B0G::load(As[tic][1], A, {0, 0, a_row_tile + 1, 0}, swizzled_offsets_A);

    if (warp_m == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    B0G::load(Bs[toc][0], B, {0, 0, b_row_tile,     1}, swizzled_offsets_B);
    B0G::load(As[toc][0], A, {0, 0, a_row_tile,     1}, swizzled_offsets_A);
    B0G::load(Bs[toc][1], B, {0, 0, b_row_tile + 1, 1}, swizzled_offsets_B);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        auto bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        B0G::load(As[toc][1], A, {0, 0, a_row_tile + 1, k + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        B0G::load(Bs[tic][0], B, {0, 0, b_row_tile, k + 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        B0G::load(As[tic][0], A, {0, 0, a_row_tile, k + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        B0G::load(Bs[tic][1], B, {0, 0, b_row_tile + 1, k + 2}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        constexpr int k = k_iters - 2;
        auto bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        B0G::load(As[toc][1], A, {0, 0, a_row_tile + 1, k + 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
        load(b0, bs0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1, toc ^= 1;
    }

    {
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_m == 0) { __builtin_amdgcn_s_barrier(); }

    store(C, cA, {0, 0, c_row_tile + warp_m,             c_col_tile + warp_n});
    store(C, cB, {0, 0, c_row_tile + warp_m,             c_col_tile + WARPS_COL + warp_n});
    store(C, cC, {0, 0, c_row_tile + WARPS_ROW + warp_m, c_col_tile + warp_n});
    store(C, cD, {0, 0, c_row_tile + WARPS_ROW + warp_m, c_col_tile + WARPS_COL + warp_n});
}

struct b0_globals {
    gl<bf16,  -1, -1, -1, -1> a;       // [Mpacked, K/2] packed fp8-as-bf16 (phase-1 output, LOCAL)
    gl<float, -1, -1, -1, -1> sc;      // [Mpacked, K/128] fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[E*N, K] bf16 expert-major, C[Mpacked, N] bf16
    gl<int,   -1, -1, -1, -1> tasks;   // [num_tasks, B0_TASK_W] B0 task list (BM=256)
    int Mpacked, N, K, num_tasks;
    hipStream_t stream;
};

// dequant preamble (packed fp8 -> bf16 scratch) then the B0 8-wave grouped GEMM. Both LOCAL.
void dispatch_grouped_gemm_b0(b0_globals g) {
    if (g.num_tasks <= 0) return;
    // bf16 scratch for the dequanted A, cached across calls (== harness local_gemm pattern).
    static bf16* d_a_bf16 = nullptr;
    static size_t cap = 0;
    const size_t need = (size_t)g.Mpacked * g.K;
    if (need > cap) {
        if (d_a_bf16) hipFree(d_a_bf16);
        hipMalloc(&d_a_bf16, need * sizeof(bf16));
        cap = need;
    }
    const fp8_t* a_fp8 = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    const float* a_sc  = &g.sc[{0, 0, 0, 0}];
    dequant_packed_dense<<<g.Mpacked, 256, 0, g.stream>>>(a_fp8, a_sc, d_a_bf16, g.Mpacked, g.K);

    gl<bf16, -1, -1, -1, -1> A(d_a_bf16, 1, 1, g.Mpacked, g.K);
    const int* tasks = &g.tasks[{0, 0, 0, 0}];
    const int threads = 8 * 64;
    // compile-time (N,K) so the 8-wave schedule's k_iters is constant; production shapes only.
    if      (g.N == 2048 && g.K == 7168) grouped_b0_gemm<2048, 7168><<<g.num_tasks, threads, 0, g.stream>>>(A, g.b, g.c, tasks, g.num_tasks);
    else if (g.N == 4096 && g.K == 7168) grouped_b0_gemm<4096, 7168><<<g.num_tasks, threads, 0, g.stream>>>(A, g.b, g.c, tasks, g.num_tasks);
    else if (g.N == 7168 && g.K == 2048) grouped_b0_gemm<7168, 2048><<<g.num_tasks, threads, 0, g.stream>>>(A, g.b, g.c, tasks, g.num_tasks);
    else printf("grouped_gemm_b0: unsupported (N=%d,K=%d) — add an instantiation in dispatch_grouped_gemm_b0\n", g.N, g.K);
}

// ================================================================================================
PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "b1_dispatch V0: EP8 multi-source gather/pack ONCE (phase1) + local grouped GEMM (phase2)";
    // PHASE 1: multi-source gather/pack/quant once -> local expert-major packed fp8 + scales.
    py::bind_function<dispatch_gather_pack>(m, "dispatch_gather_pack",
        &gatherpack_globals::a_src,
        &gatherpack_globals::sc_src,
        &gatherpack_globals::a_dst,
        &gatherpack_globals::sc_dst,
        &gatherpack_globals::seg,
        &gatherpack_globals::tilemeta,
        &gatherpack_globals::iris_ctx,
        &gatherpack_globals::Msrc,
        &gatherpack_globals::Mpacked,
        &gatherpack_globals::K,
        &gatherpack_globals::Nseg,
        &gatherpack_globals::Ntile
    );
    // PHASE 2: local grouped 32-expert GEMM (serial baseline path) over the packed buffer.
    py::bind_function<dispatch_grouped_gemm>(m, "grouped_gemm",
        &micro_globals::a,
        &micro_globals::sc,
        &micro_globals::b,
        &micro_globals::c,
        &micro_globals::tasks,
        &micro_globals::iris_ctx,
        &micro_globals::Mpacked,
        &micro_globals::N,
        &micro_globals::K,
        &micro_globals::src_rank,
        &micro_globals::num_tasks,
        &micro_globals::nsub,
        &micro_globals::fused
    );
    // PHASE 2 (ALT): B0-class 8-wave grouped GEMM (dequant preamble + 256x256 ping-pong). LOCAL only
    // (no iris_ctx). Consumes the SAME packed a/sc as grouped_gemm + a B0 task list (b0_tasks.py).
    py::bind_function<dispatch_grouped_gemm_b0>(m, "grouped_gemm_b0",
        &b0_globals::a,
        &b0_globals::sc,
        &b0_globals::b,
        &b0_globals::c,
        &b0_globals::tasks,
        &b0_globals::Mpacked,
        &b0_globals::N,
        &b0_globals::K,
        &b0_globals::num_tasks
    );
}
