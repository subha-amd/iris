// ================================================================================================
// gather_pack_dropin.cpp  —  the fused MoE dispatch, ISOLATED for analysis.
//
// This file is a READING / ANALYSIS extract. Every kernel below is copied VERBATIM from the
// production `kernel.cpp` (the file the build actually compiles); this copy just pulls the drop-in
// path out of the 2000-line file, away from the GEMM / decode variants / combine / tilecomm / the old
// SEG-based gather, and annotates it. To change behaviour, edit `kernel.cpp`, not this file.
//
// WHAT THIS REPLACES
// ------------------
// The unfused production MoE dispatch region is three separate library kernels:
//     MORI EpDispatch   (all-to-all: push each token to the GPU that owns its expert)
//   + aiter moe_sorting (×2 internal passes: group the received tokens by expert -> sorted indices)
//   + aiter dynamic_quant (bf16 -> fp8 of the activations)
// These three produce "fc1's A operand": the activations, quantized to fp8, grouped expert-major.
//
// This fused path produces the SAME logical result in ONE pull-based pass. Its input is EXACTLY the
// router's output — per rank, `tokens` bf16 [T, 7168] and `topk_ids` [T, 8] — so it is a valid DROP-IN
// (the "tier-2" comparison; see fairbench/FAIRNESS_AUDIT.md §5.7). Measured on 8× MI350X, warm, all 8
// ranks, MAX over ranks, real top-8 route:
//     drop-in fused dispatch : 309 µs prefill (T=1024) / 103 µs decode (T=64)
//     production bf16-dispatch: 392 µs        / 131 µs      ->  1.27× at both
// (Of that 1.27×, ~1.1× is the fusion itself and the rest is moving fp8 not bf16 over XGMI.)
//
// THE PIPELINE (what the Python driver, fairbench/bench_dropin.py, chains — all on-clock):
//
//   input per rank:  tokens bf16 [T,7168]   topk_ids [T,8]   (the router's output; nothing else)
//        │
//        ├─(0) aiter per_1x128 quant: tokens bf16 -> A_src fp8 [T,7168] + scales, placed on the IRIS
//        │      symmetric heap so peer GPUs can pull it.  (aiter op, not in this file.)
//        │
//        ├─(1) plan_allgather_ids   : every rank pulls every rank's topk_ids over XGMI
//        │      -> all_ids [world*T*8]      (a pull consumer must know the GLOBAL routing to know what
//        │                                   to pull; MORI's PUSH never needs this — see note in §1)
//        │
//        ├─(2) build_plan (count->scan->scatter): from all_ids, find the tokens routed to THIS rank's
//        │      experts and assign each a packed-row slot
//        │      -> erb [E+1]  (expert_row_begin, 256-padded prefix; erb[E] = Mpacked)
//        │      -> rowmap [Mpacked, 2] = (src_rank, src_row) per packed row; (-1,-1) = padding
//        │
//        └─(3) gather_pack_rowmap   : for each packed row, pull its fp8 activation from
//               (src_rank, src_row) over XGMI into the expert-major packed buffer
//               -> A_pk fp8 [Mpacked, 7168] + scales   ==  fc1's A operand.
//
// The all-to-all data movement is stage (3) — the pull gather — and it is ~73% of the region and the
// XGMI-bandwidth-bound bottleneck (XGMI ≈ 47 GiB/s/link, ~20× slower than HBM). Stages (1)+(2) are the
// on-device routing (what MORI does inside its dispatch kernel); (0) is the quant.
//
// Shapes (DeepSeek-R1, EP8): world=8, E=32 local experts/rank (256 global), topk=8, K=H=7168, fp8 e4m3
// with per-128 fp32 block scales (QGROUP=128, NG=56). Each rank holds T tokens (data-parallel).
// ================================================================================================

// ---- prerequisites (same as kernel.cpp) --------------------------------------------------------
#include "kittens.cuh"                    // HipKittens: gl<> global-tensor view, bf16, tile types
#include "pyutils/pyutils.cuh"            // py::bind_function (the Python entry points)
#include <iris/iris.hpp>                  // IRIS: iris_device_view, ctx.load = XGMI remote read
#include <hip/hip_fp8.h>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;        // unsigned char, 1 byte
static constexpr int QGROUP = 128;        // per-128-group fp8 block-scale quant group

// gather tiling knobs. GP_SPLIT>1 decouples the grid from the tile count so more XGMI loads are
// in-flight at once (grid = Ntile alone is only ~50% of a 256-CU chip and exposes pull latency).
#define GP_BM       64                    // packed-row tile height (one block owns a 64-row tile)
#define GP_THREADS  256
#define GP_SPLIT    8                     // blocks per BM-tile; each takes a disjoint slice of the copy

#define PLAN_PAD    256                   // per-expert packed-region padding (the GEMM tiles at 256/16)
#define PLAN_MAX_E  1024                  // max experts/rank the scan's LDS array holds

// ================================================================================================
// STAGE 1 — plan_allgather_ids : IRIS all-gather of every rank's topk_ids.
//
// A PULL gather forces the consumer to know the GLOBAL routing (which token on which rank goes to
// which expert) so it knows what to pull. MORI's PUSH does not — each rank pushes on its own topk_ids.
// So this all-gather is an intrinsic, extra cost of choosing pull over push. It is cheap (topk_ids is
// [T,8] ints, tiny vs the 7168-wide activations) — ~14 µs prefill / ~10 µs decode — but it is real,
// and it lives in the drop-in (tier-2) number, not hidden off-clock.
//
// my_ids lives on the IRIS symmetric heap (identical offset on every rank) so ctx.load(src+i, r)
// reaches peer r's copy over XGMI. Output all_ids is local, laid out rank-major: [rank0 T*8][rank1 ...].
// ================================================================================================
struct plan_ag_globals {
    gl<int, -1, -1, -1, -1> my_ids;    // [T*TOPK, 1]  this rank's topk_ids (IRIS heap, symmetric)
    gl<int, -1, -1, -1, -1> all_ids;   // [world*T*TOPK, 1] out (local)
    iris::iris_device_view iris_ctx;
    int world, n_per_rank;             // n_per_rank = T*TOPK
    hipStream_t stream;
    dim3 grid()  { return dim3((n_per_rank + 255) / 256); }
    dim3 block() { return dim3(256); }
};

__global__ void plan_allgather_ids_kernel(plan_ag_globals g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n_per_rank) return;
    const int cur = g.iris_ctx.cur_rank();
    iris::iris_device_view ctx = g.iris_ctx;
    const int* src = &g.my_ids[{0, 0, 0, 0}];
    int* dst = &g.all_ids[{0, 0, 0, 0}];
    // pull lane i from every peer r (local deref when r == this rank, else one XGMI read).
    for (int r = 0; r < g.world; ++r)
        dst[r * g.n_per_rank + i] = (r == cur) ? src[i] : ctx.load(src + i, r);
}

void dispatch_plan_allgather_ids(plan_ag_globals g) {
    if (g.n_per_rank <= 0 || g.world <= 0) return;   // a 0-block grid is hipErrorInvalidConfiguration
    plan_allgather_ids_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
// STAGE 2 — build_plan : turn the all-gathered topk_ids into a packing plan, ON DEVICE.
//
// This is the on-device routing MORI folds inside its dispatch kernel (atomicAdd slot assignment).
// It replaces aiter's moe_sorting: instead of producing a permutation the GEMM applies later, it
// directly assigns each (token routed to a local expert) a contiguous packed-row slot, expert-major.
//
//   plan_count   : histogram — how many (token,expert) pairs land on each of this rank's E experts.
//   plan_scan    : 256-padded exclusive prefix over the counts -> erb[e] (each expert's base row) and
//                  cursor[e] (a running write-pointer, seeded to erb[e]).  erb[E] = Mpacked (total).
//   plan_scatter : for each pair routed here, atomicAdd(cursor[e]) to claim the next slot and write
//                  (src_rank, src_row) into rowmap[slot].  Order within an expert is atomics-
//                  nondeterministic — fine, any permutation of an expert's rows is a valid packing.
//
// NOTE (an optimization target): plan_scatter uses one global atomicAdd per (token,expert) pair, and
// AMD global atomics are slow. A warp-aggregated / block-privatized histogram would cut this. This is
// stage (2) of the pipeline, ~53 µs at prefill — the 2nd-largest piece after the gather.
// ================================================================================================
struct plan_globals {
    gl<int, -1, -1, -1, -1> all_ids;   // [world*T*TOPK, 1] global expert ids (from stage 1)
    gl<int, -1, -1, -1, -1> counts;    // [E,1] scratch
    gl<int, -1, -1, -1, -1> cursor;    // [E,1] scratch (write pointer per expert)
    gl<int, -1, -1, -1, -1> erb;       // [E+1,1] out: expert_row_begin; erb[E] = Mpacked
    gl<int, -1, -1, -1, -1> rowmap;    // [Mpacked_max, 2] out: (src_rank, src_row); (-1,-1) = padding
    int world, T, TOPK, E, dst_rank, Mpacked_max;
    hipStream_t stream;
    dim3 grid()  { return dim3(1); }
    dim3 block() { return dim3(256); }
};

// zero the histogram and pre-fill the whole packed rowmap with the (-1,-1) padding sentinel.
__global__ void plan_reset_kernel(plan_globals g) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g.E; i += stride)
        g.counts[{0, 0, i, 0}] = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g.Mpacked_max; i += stride) {
        g.rowmap[{0, 0, i, 0}] = -1;    // unrouted / padding -> zero-sentinel in the gather (stage 3)
        g.rowmap[{0, 0, i, 1}] = -1;
    }
}

// histogram: count the (token,expert) pairs whose GLOBAL expert id is owned by THIS rank ([lo,hi)).
__global__ void plan_count_kernel(plan_globals g) {
    const int n = g.world * g.T * g.TOPK;
    const int lo = g.dst_rank * g.E, hi = lo + g.E;
    int* counts = &g.counts[{0, 0, 0, 0}];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int gid = g.all_ids[{0, 0, i, 0}];
        if (gid >= lo && gid < hi) atomicAdd(counts + (gid - lo), 1);
    }
}

// single-block 256-padded exclusive prefix sum -> expert_row_begin + cursor. erb[E] = Mpacked.
__global__ void plan_scan_kernel(plan_globals g) {
    __shared__ int s[PLAN_MAX_E];
    const int tid = threadIdx.x;
    if (g.E > PLAN_MAX_E) { if (tid == 0) g.erb[{0, 0, 0, 0}] = -1; return; }   // host checks erb[0] >= 0
    for (int e = tid; e < g.E; e += blockDim.x) s[e] = g.counts[{0, 0, e, 0}];
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < g.E; ++e) {
            g.erb[{0, 0, e, 0}]    = acc;
            g.cursor[{0, 0, e, 0}] = acc;
            acc += ((s[e] + PLAN_PAD - 1) / PLAN_PAD) * PLAN_PAD;   // pad each expert up to PLAN_PAD
        }
        g.erb[{0, 0, g.E, 0}] = acc;                               // = Mpacked (total padded rows)
    }
}

// scatter: each pair routed here claims the next slot in its expert's region and records its source.
__global__ void plan_scatter_kernel(plan_globals g) {
    const int n = g.world * g.T * g.TOPK;
    const int lo = g.dst_rank * g.E, hi = lo + g.E;
    const int per_rank = g.T * g.TOPK;
    int* cursor = &g.cursor[{0, 0, 0, 0}];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int gid = g.all_ids[{0, 0, i, 0}];
        if (gid < lo || gid >= hi) continue;             // not our expert
        const int src_rank = i / per_rank;               // which rank this (token,expert) pair came from
        const int src_row  = (i % per_rank) / g.TOPK;    // the token index on that rank
        const int dst = atomicAdd(cursor + (gid - lo), 1);   // <- the slow global atomic
        if (dst < g.Mpacked_max) {
            g.rowmap[{0, 0, dst, 0}] = src_rank;
            g.rowmap[{0, 0, dst, 1}] = src_row;
        }
    }
}

// the four kernels run in order on one stream (each <<<>>> serializes after the previous).
void dispatch_build_plan(plan_globals g) {
    const int n = g.world * g.T * g.TOPK;
    if (n <= 0 || g.E <= 0 || g.Mpacked_max <= 0) return;
    const int reset_n = (n > g.Mpacked_max) ? n : g.Mpacked_max;
    const int blocks  = (reset_n + 255) / 256;
    const int nblk    = (n + 255) / 256;
    plan_reset_kernel  <<<blocks, 256,        0, g.stream>>>(g);
    plan_count_kernel  <<<nblk,   256,        0, g.stream>>>(g);
    plan_scan_kernel   <<<1,      PLAN_MAX_E, 0, g.stream>>>(g);
    plan_scatter_kernel<<<nblk,   256,        0, g.stream>>>(g);
}

// ================================================================================================
// STAGE 3 — gather_pack_rowmap : THE ALL-TO-ALL. The bottleneck (~226 µs prefill, ~73% of the region).
//
// For each packed row `dst_row`, look up (src_rank, src_row) in the rowmap and PULL that token's fp8
// activation from peer `src_rank` over XGMI into the local expert-major packed buffer. This is the
// data movement that replaces EpDispatch's push — same tokens crossing the same fabric, but pulled
// (fp8) instead of pushed (bf16), and landing expert-major directly (so no separate sort pass, and no
// scattered read at GEMM time).
//
// Grid: (Ntile, GP_SPLIT). One block owns a GP_BM(=64)-row tile; GP_SPLIT blocks cooperate on it to
// keep more XGMI loads in flight. Per lane, per iteration: one uint4 = 16 fp8 bytes (the widest single
// load) + one fp32 scale per 128-group. Local short-circuit when src_rank == this rank (HBM, no XGMI).
// Unrouted/padding rows (src_rank == -1) get the fp8 zero-sentinel (0x00 dequantizes to exactly 0).
//
// OPTIMIZATION SURFACE (why this is the target the auto-gpu-kernel optimizer attacks):
//   - each lane moves 16 B (uint4) per transaction; wider/fewer transactions raise effective XGMI BW.
//   - a pull re-reads each distinct token ~1.5× (once per expert it routes to); pulling each distinct
//     token ONCE then replicating locally (HBM, ~20× faster) could cut XGMI bytes ~1.5×.
//   - GP_SPLIT / block count sets how many remote loads are in flight — tune to saturate 8 links.
// ================================================================================================
struct gatherpack_rowmap_globals {
    gl<bf16,  -1, -1, -1, -1> a_src;    // [Msrc, K/2] fp8-as-bf16 on EACH source rank (IRIS heap)
    gl<float, -1, -1, -1, -1> sc_src;   // [Msrc, K/128] fp32 scales on each source rank (IRIS heap)
    gl<bf16,  -1, -1, -1, -1> a_dst;    // [Mpacked, K/2] LOCAL packed fp8-as-bf16 (IRIS heap)
    gl<float, -1, -1, -1, -1> sc_dst;   // [Mpacked, K/128] LOCAL packed scales (IRIS heap)
    gl<int,   -1, -1, -1, -1> rowmap;   // [Mpacked, 2] = (src_rank, src_row); (-1,*) = unrouted/padding
    iris::iris_device_view iris_ctx;
    int Msrc, Mpacked, K;
    hipStream_t stream;
    dim3 grid()  { int nt = (Mpacked + GP_BM - 1) / GP_BM; return dim3(nt > 0 ? nt : 1, GP_SPLIT); }
    dim3 block() { return dim3(GP_THREADS); }
};

__global__ __launch_bounds__(GP_THREADS, 1)
void gather_pack_rowmap_kernel(gatherpack_rowmap_globals g) {
    __shared__ int rm_rank[GP_BM];        // this tile's (src_rank, src_row) staged in LDS: one int2/row
    __shared__ int rm_row[GP_BM];

    const int m_tile   = blockIdx.x;
    const int split    = blockIdx.y;      // GP_SPLIT: this block's slice of the copy work
    const int tid      = threadIdx.x;
    const int nthreads = GP_THREADS;
    const int K        = g.K;
    const int NG       = K / QGROUP;
    const int cur_rank = g.iris_ctx.cur_rank();
    iris::iris_device_view ctx = g.iris_ctx;

    const int tile_dst0 = m_tile * GP_BM;
    if (tile_dst0 >= g.Mpacked) return;

    // load this tile's rowmap rows into LDS once (a single int2 load per row — the whole reason the
    // flat rowmap ABI beats the run-encoded route_segment ABI: no serial per-tile segment scan).
    for (int i = tid; i < GP_BM; i += nthreads) {
        const int r = tile_dst0 + i;
        if (r < g.Mpacked) { rm_rank[i] = g.rowmap[{0, 0, r, 0}]; rm_row[i] = g.rowmap[{0, 0, r, 1}]; }
        else               { rm_rank[i] = -1;                     rm_row[i] = -1; }
    }
    __syncthreads();

    const fp8_t* a_src_base  = reinterpret_cast<const fp8_t*>(&g.a_src[{0, 0, 0, 0}]);
    const float* sc_src_base = &g.sc_src[{0, 0, 0, 0}];
    fp8_t* a_dst_base  = reinterpret_cast<fp8_t*>(&g.a_dst[{0, 0, 0, 0}]);
    float* sc_dst_base = &g.sc_dst[{0, 0, 0, 0}];

    const int chunks_per_row = K / 16;                 // 16 fp8 bytes (uint4) per lane per chunk
    const long total_chunks  = (long)GP_BM * chunks_per_row;

    // GP_SPLIT blocks stride through this tile's (row × K/16) chunks; each lane copies one uint4.
    for (long ci = (long)split * nthreads + tid; ci < total_chunks; ci += (long)GP_SPLIT * nthreads) {
        const int r  = (int)(ci / chunks_per_row);     // tile-local packed row
        const int kc = (int)(ci % chunks_per_row) * 16;// byte offset within the row
        const int dst_row = tile_dst0 + r;             // global packed dst row
        if (dst_row >= g.Mpacked) continue;

        const int src_rank = rm_rank[r];
        const int src_row  = rm_row[r];
        const bool valid   = (src_rank >= 0);

        // ---- THE XGMI PULL: 16 fp8 bytes from (src_rank, src_row); local deref if it's our own rank.
        uint4 vbytes;
        if (valid && src_row < g.Msrc && (kc + 16) <= K) {
            const uint4* sp = reinterpret_cast<const uint4*>(a_src_base + (size_t)src_row * K + kc);
            vbytes = (src_rank == cur_rank) ? *sp : ctx.load(sp, src_rank);   // <- ctx.load == XGMI read
        } else {
            vbytes = make_uint4(0u, 0u, 0u, 0u);       // ZERO SENTINEL for unrouted/padding rows
        }
        *reinterpret_cast<uint4*>(a_dst_base + (size_t)dst_row * K + kc) = vbytes;

        // ---- the per-128-group fp32 scale that goes with this chunk (idempotent: same value/group).
        const int grp = kc / QGROUP;
        if (grp < NG) {
            float scale = 0.0f;
            if (valid && src_row < g.Msrc) {
                const float* spc = sc_src_base + (size_t)src_row * NG + grp;
                scale = (src_rank == cur_rank) ? *spc : ctx.load(spc, src_rank);
            }
            sc_dst_base[(size_t)dst_row * NG + grp] = scale;
        }
    }
}

void dispatch_gather_pack_rowmap(gatherpack_rowmap_globals g) {
    if (g.Mpacked <= 0) return;
    gather_pack_rowmap_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
// Python entry points (the three the drop-in region calls, in order). In the real build these live in
// kernel.cpp's PYBIND11_MODULE(tk_kernel, ...) block; shown here so the ABI is visible in one place.
//
// From fairbench/bench_dropin.py, per timed region:
//     aq(tokens) -> A_src fp8 (+ copy to heap)                       # stage 0 (aiter, external)
//     tk_kernel.plan_allgather_ids(MY_IDS, ALL_IDS, ctx, world, T*TOPK)                 # stage 1
//     tk_kernel.build_plan(ALL_IDS, COUNTS, CURSOR, ERB, ROWMAP, world, T, TOPK, E, rank, Mpk_max)  # 2
//     tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, ROWMAP, ctx,
//                                           T, Mpk_max, K)                              # stage 3
//   -> A_pk (fp8, expert-major) is fc1's A operand.
// ================================================================================================
#ifdef GATHER_PACK_DROPIN_PYBIND        // (not defined in this reading copy; real bindings are in kernel.cpp)
PYBIND11_MODULE(tk_kernel_dropin, m) {
    py::bind_function<dispatch_plan_allgather_ids>(m, "plan_allgather_ids",
        &plan_ag_globals::my_ids, &plan_ag_globals::all_ids, &plan_ag_globals::iris_ctx,
        &plan_ag_globals::world, &plan_ag_globals::n_per_rank);

    py::bind_function<dispatch_build_plan>(m, "build_plan",
        &plan_globals::all_ids, &plan_globals::counts, &plan_globals::cursor, &plan_globals::erb,
        &plan_globals::rowmap, &plan_globals::world, &plan_globals::T, &plan_globals::TOPK,
        &plan_globals::E, &plan_globals::dst_rank, &plan_globals::Mpacked_max);

    py::bind_function<dispatch_gather_pack_rowmap>(m, "dispatch_gather_pack_rowmap",
        &gatherpack_rowmap_globals::a_src, &gatherpack_rowmap_globals::sc_src,
        &gatherpack_rowmap_globals::a_dst, &gatherpack_rowmap_globals::sc_dst,
        &gatherpack_rowmap_globals::rowmap, &gatherpack_rowmap_globals::iris_ctx,
        &gatherpack_rowmap_globals::Msrc, &gatherpack_rowmap_globals::Mpacked,
        &gatherpack_rowmap_globals::K);
}
#endif
