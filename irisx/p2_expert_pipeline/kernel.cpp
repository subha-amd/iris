// p2_expert_pipeline / kernel.cpp
// ================================================================================================
// CANDIDATE P2 — EXPERT-GRANULAR COPY-ONCE DOUBLE-BUFFER (gather expert e+1 while GEMM computes e).
//
// WHY (Gate-1, EXPERIMENT_LEDGER.md): the same three requirements as P1 — (1) ~1x remote A traffic,
//   (2) B0-class GEMM efficiency, (3) comm/compute overlap — but at EXPERT granularity instead of
//   per-K-tile. Coarser sync = far fewer flags, no per-K-tile producer/consumer ping-pong inside the
//   GEMM, and the GEMM stays the EXACT B0 kernel. Likely the fastest path to a real EP (grouped)
//   result. Build on v5_grouped's packed-expert layout + ep8_gather's multi-source gather.
//
// DATAFLOW:
//   Mpacked activation space packs E experts back-to-back; each expert e occupies padded rows
//   [expert_row_begin[e], expert_row_begin[e] + padded_rows[e]) (padded UP to a multiple of BM so a
//   full B0 tile never spills across experts — v5_grouped's no-contamination layout). The remote fp8
//   A + scales live on src_rank(s); B is local expert-major [E*N, K]; C is local [Mpacked, N].
//
//   Instead of one giant inbox we keep TWO rotating LOCAL bf16 expert slots, each sized for the
//   LARGEST expert region (slot_rows x K). slot = e % 2.
//
//   PRODUCER kernel (modest grid, own stream, launched first):
//     for e = 0..E-1 (claimed via fetch_add cursor; one block-group per expert):
//       - WAIT (acquire) until done[e-2] is set, i.e. the consumer has finished the PREVIOUS user of
//         this slot (slot = e%2). For e<2 the slot is free (done initialized to "ready-to-fill").
//       - gather+dequant expert e's padded region (multi-source via ep8_gather route_segments) ONCE
//         from src rank(s) into slot e%2 (row-major bf16). Copy amplification ~= 1.0x.
//       - fence_system; release ready[e] = (gen<<1)|1.
//
//   CONSUMER kernel = B0 GROUPED GEMM, one block per (expert, m_tile, n_tile) flat task:
//       - ACQUIRE-wait ready[e] for this task's expert; read A from slot e%2 (local); run EXACT B0
//         inner loop; store C.
//       - the LAST block of expert e (atomic arrival counter == tiles_of_e) release-sets done[e],
//         freeing slot e%2 for the producer to refill with expert e+2.
//
//   So at steady state the producer is filling expert e+1 (slot (e+1)%2) while the consumer GEMMs
//   expert e (slot e%2) — comm/compute overlap at expert granularity, A moved exactly once, GEMM at
//   B0 efficiency.
//
// VARIABLE M_e / EMPTY EXPERTS / TAIL ROWS:
//   - padded_rows[e] is a multiple of BM; valid_rows[e] (<= padded) bounds the real rows. Gather
//     masks at valid_rows -> padding rows read the ZERO sentinel (dequant to 0). C of padding rows is
//     dead space no one reads (host packs disjoint regions).
//   - EMPTY expert (valid_rows==0, padded_rows==0): producer still sets ready[e] immediately (no
//     gather) and the host emits ZERO consumer tasks for it, so the consumer's arrival counter for e
//     is 0 and done[e] is pre-set by the host. The producer's done-wait for e+2 thus never stalls.
//   - MULTIPLE SOURCE RANKS per expert: ep8_gather route_segments drive a multi-source gather
//     (fast single-source path + segment iterator for straddling tiles).
//
// DEADLOCK AVOIDANCE (full argument in P2_DESIGN.md):
//   - TWO SEPARATE LAUNCHES, TWO non-blocking streams; producer first reserves CUs.
//   - The slot recycle handshake is a STRICT 2-deep pipeline: producer waits done[e-2] before
//     filling slot e%2; consumer waits ready[e] before reading; consumer sets done[e] when its last
//     tile of e retires. This is a bounded producer/consumer queue of depth 2 — classic deadlock-free
//     (producer can always make progress on at most 2 outstanding experts; consumer always has its
//     awaited expert eventually produced). Host PRE-INITIALIZES done[-2],done[-1] (encoded as the
//     two priming slots) to "free" so the first two fills proceed; empty experts' done pre-set.
//   - No consumer-to-consumer dependency; the per-expert arrival counter only gates done[e], never a
//     GEMM. A single resident producer block-group drains all experts in order.
// ================================================================================================
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
#include "expert_pipeline_abi.h"
#include "ep8_gather.h"          // multi-source gather + route_segment ABI (Agent 03)
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;
static constexpr int QGROUP = 128;

__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

#ifndef NUM_PRODUCER_THREADS_P2
#define NUM_PRODUCER_THREADS_P2 256
#endif

// Consumer GEMM == B0 tiling (verbatim occupancy).
constexpr int B0_WARPS = 8;
using B0G = kittens::group<B0_WARPS>;
constexpr int B0_BM = 256, B0_BN = 256, B0_BK = 64;
using B0_ST_A = st_bf<B0_BM / 2, B0_BK, st_16x32_s>;
using B0_ST_B = st_bf<B0_BN / 2, B0_BK, st_16x32_s>;

// Flat consumer task layout (host-built). One block per task.
static constexpr int P2_TASK_W = 5;
enum { PT_EXPERT = 0, PT_MTILE = 1, PT_NTILE = 2, PT_SLOTROWS = 3, PT_EROWBEG = 4 };
// PT_EXPERT: local expert id; PT_MTILE: B0-row-tile within the expert's slot (each = B0_BM rows);
// PT_NTILE: B0 N-tile; PT_SLOTROWS: rows allocated per slot (slot stride); PT_EROWBEG: expert's
// global packed row begin (for the C store).

struct p2_globals {
    // remote fp8 source(s): activation buffer (fp8 bytes as bf16[Msrc,K/2]) + scales, on src ranks.
    gl<bf16,  -1, -1, -1, -1> a_src;     // [Msrc, K/2]
    gl<float, -1, -1, -1, -1> sc_src;    // [Msrc, K/128]
    gl<bf16,  -1, -1, -1, -1> b, c;      // local B[E*N, K], C[Mpacked, N]
    // TWO rotating LOCAL bf16 expert slots: [2, slot_rows, K] flattened.
    gl<bf16,  -1, -1, -1, -1> slots;     // [2*slot_rows, K]
    // route metadata (replicated local copy): route_segment[] as int [Nseg,5], plus per-expert info.
    gl<int,   -1, -1, -1, -1> segs;      // [Nseg, 5]  (expert_id,src_rank,src_row_begin,dst_row_begin,row_count)
    gl<int,   -1, -1, -1, -1> expert_meta; // [E, 4] (valid_rows, seg_begin, seg_count, padded_rows)
    gl<int,   -1, -1, -1, -1> tasks;     // [num_tasks, P2_TASK_W]
    // handshake flags (local, consumer-rank heap).
    gl<int,   -1, -1, -1, -1> ready_gl;  // [E]   producer -> consumer
    gl<int,   -1, -1, -1, -1> done_gl;   // [E+2] consumer -> producer (slot recycle; +2 priming)
    gl<int,   -1, -1, -1, -1> arrive_gl; // [E]   per-expert consumer tile arrival counter
    gl<int,   -1, -1, -1, -1> ecursor_gl;// [1]   producer expert dispenser
    iris::iris_device_view iris_ctx;
    int E, N, K, Mpacked, Msrc;
    int slot_rows;            // rows per slot (>= max padded_rows over experts)
    int generation;
    int num_tasks;
    int num_producer_blocks;

    dim3 producer_grid()  { return dim3(num_producer_blocks > 0 ? num_producer_blocks : 1); }
    dim3 producer_block() { return dim3(NUM_PRODUCER_THREADS_P2); }
    dim3 consumer_grid()  { return dim3(num_tasks > 0 ? num_tasks : 1); }
    dim3 consumer_block() { return dim3(B0_WARPS * 64); }
    size_t consumer_shared() { return 2 * (sizeof(B0_ST_A) + sizeof(B0_ST_B)) + 1024; }
};

// ------------------------------------------------------------------------------------------------
// PRODUCER: per-expert copy-once gather into rotating slot, with 2-deep slot-recycle handshake.
// One block-group claims expert e via fetch_add(ecursor). Whole block cooperates on the gather.
// Multi-source via ep8_gather route_segments (per BM tile within the expert region).
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_PRODUCER_THREADS_P2, 1)
void p2_producer(p2_globals g) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    iris::iris_device_view ctx = g.iris_ctx;
    const int local_rank = ctx.cur_rank();
    const int E = g.E, K = g.K, NG = K / QGROUP;
    const int ready_val = ep_ready_flag(g.generation);

    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a_src[{0, 0, 0, 0}]);
    const float* sc_base = &g.sc_src[{0, 0, 0, 0}];
    bf16* slots = &g.slots[{0, 0, 0, 0}];
    const int* segs_raw = &g.segs[{0, 0, 0, 0}];
    const int* emeta = &g.expert_meta[{0, 0, 0, 0}];
    int* ready  = &g.ready_gl[{0, 0, 0, 0}];
    int* done   = &g.done_gl[{0, 0, 0, 0}];   // index [e+2]; done[0],done[1] = priming "free"
    int* ecur   = &g.ecursor_gl[{0, 0, 0, 0}];
    const ep8_gather::route_segment* segs =
        reinterpret_cast<const ep8_gather::route_segment*>(segs_raw);

    __shared__ int s_e;
    while (true) {
        if (tid == 0)
            s_e = ctx.fetch_add<int, iris::memory_scope_device>(
                      ecur, 1, local_rank, iris::memory_order_relaxed);
        __syncthreads();
        const int e = s_e;
        if (e >= E) break;

        const int valid_rows = emeta[e * 4 + 0];
        const int seg_begin  = emeta[e * 4 + 1];
        const int seg_count  = emeta[e * 4 + 2];
        const int slot = e & 1;
        bf16* slot_base = slots + (size_t)slot * (size_t)g.slot_rows * K;

        // --- slot recycle: wait until the previous user of this slot (expert e-2) is done. ---
        // done is offset by +2 so done[e] here means "slot for (e-2) freed"; host primes done[0..1].
        if (tid == 0) {
            int f = ctx.atomic_load<int, iris::memory_scope_system>(
                        &done[e], local_rank, iris::memory_order_acquire);
            while (f != ready_val) {
                f = ctx.atomic_load<int, iris::memory_scope_system>(
                        &done[e], local_rank, iris::memory_order_acquire);
            }
        }
        __syncthreads();

        // --- copy-once multi-source gather+dequant of this expert's valid rows into slot ---
        // Iterate BM tiles within [0, valid_rows); each tile resolves its source segment(s).
        const int padded = emeta[e * 4 + 3];
        for (int tile_row = 0; tile_row < padded; tile_row += ep8_gather_BM) {
            ep8_gather::seg_tile_view v;
            v.segs       = segs;
            v.seg_begin  = seg_begin;
            v.seg_count  = seg_count;
            v.tile_dst0  = tile_row;            // dst rows are SLOT-LOCAL (expert region starts at 0)
            int vr_here = valid_rows - tile_row;
            if (vr_here < 0) vr_here = 0;
            if (vr_here > ep8_gather_BM) vr_here = ep8_gather_BM;
            v.valid_rows = vr_here;

            // Resolve fast single-source vs multi-source. dst rows here are slot-local; segments'
            // dst_row_begin are also expressed slot-local in expert_meta-built segs (host responsibility).
            int fast_src = 0, fast_off = 0;
            bool fast = ep8_gather::tile_is_single_source(v, &fast_src, &fast_off);

            // gather this BM x K tile (all K) row by row into slot_base[tile_row + r].
            const int chunks_per_row = K / 16;
            for (int r = 0; r < ep8_gather_BM; ++r) {
                const int dst_row = tile_row + r;
                if (dst_row >= padded) break;
                bf16* o_row = slot_base + (size_t)dst_row * K;
                bool valid = (r < v.valid_rows);
                int src_rank = local_rank, src_row = 0;
                if (valid) {
                    if (fast) { src_rank = fast_src; src_row = (v.tile_dst0 + r) + fast_off; }
                    else {
                        // segment lookup for this slot-local dst row.
                        const int abs_dst = v.tile_dst0 + r;
                        int sidx = -1;
                        for (int si = 0; si < seg_count; ++si) {
                            const ep8_gather::route_segment s = segs[seg_begin + si];
                            const int lo = s.dst_row_begin, hi = s.dst_row_begin + s.row_count;
                            if (abs_dst >= lo && abs_dst < hi) { sidx = seg_begin + si; break; }
                        }
                        if (sidx < 0) valid = false;
                        else {
                            const ep8_gather::route_segment s = segs[sidx];
                            src_rank = s.src_rank;
                            src_row  = s.src_row_begin + (abs_dst - s.dst_row_begin);
                        }
                    }
                }
                for (int c = tid; c < chunks_per_row; c += nthreads) {
                    const int k = c * 16;
                    uint4 packed;
                    if (valid && src_row < g.Msrc) {
                        const fp8_t* aptr = a_base + (size_t)src_row * K + k;
                        const uint4* vptr = reinterpret_cast<const uint4*>(aptr);
                        packed = (src_rank == local_rank) ? *vptr : ctx.load(vptr, src_rank);
                    } else {
                        packed = make_uint4(0u, 0u, 0u, 0u);   // ZERO SENTINEL (pad/unrouted)
                    }
                    const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);
                    const float* sc_row = sc_base + (size_t)src_row * NG;
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int kk = k + j;
                        float scale = 1.0f;
                        if (valid && src_row < g.Msrc) {
                            const float* sp = sc_row + (kk / QGROUP);
                            scale = (src_rank == local_rank) ? *sp : ctx.load(sp, src_rank);
                        }
                        o_row[kk] = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            ctx.fence<iris::memory_scope_system>(iris::memory_order_release);
            ctx.atomic_store<int, iris::memory_scope_system>(
                &ready[e], ready_val, local_rank, iris::memory_order_release);
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------
// CONSUMER == B0 grouped GEMM, one block per flat task. Waits ready[e]; reads A from slot e%2; runs
// the EXACT B0 inner loop; stores C. The last-arriving block of expert e release-sets done[e+2] to
// free the slot for expert e+2.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(B0_WARPS * 64, 2)
void p2_consumer(p2_globals g) {
    const int task = blockIdx.x;
    if (task >= g.num_tasks) return;

    const int* tk = &g.tasks[{0, 0, task, 0}];
    const int e          = tk[PT_EXPERT];
    const int m_tile     = tk[PT_MTILE];       // B0_BM-row tile within the expert's slot
    const int n_tile     = tk[PT_NTILE];       // B0_BN N tile
    const int slot_rows  = tk[PT_SLOTROWS];
    const int erow_begin = tk[PT_EROWBEG];     // expert's global packed C row begin

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    B0_ST_A (&As)[2] = al.allocate<B0_ST_A, 2>();
    B0_ST_B (&Bs)[2] = al.allocate<B0_ST_B, 2>();

    const int N = g.N, K = g.K;
    constexpr int WARPS_COL = 4, WARPS_ROW = 2;
    constexpr int REG_M = B0_BM / WARPS_ROW / 2;
    constexpr int REG_N = B0_BN / WARPS_COL / 2;
    const int k_iters = K / B0_BK;
    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;

    iris::iris_device_view ctx = g.iris_ctx;
    const int local_rank = ctx.cur_rank();
    const int ready_val = ep_ready_flag(g.generation);
    int* ready  = &g.ready_gl[{0, 0, 0, 0}];
    int* done   = &g.done_gl[{0, 0, 0, 0}];
    int* arrive = &g.arrive_gl[{0, 0, 0, 0}];

    // ---- acquire-wait this expert's slot ready ----
    if (threadIdx.x == 0) {
        int f = ctx.atomic_load<int, iris::memory_scope_system>(
                    &ready[e], local_rank, iris::memory_order_acquire);
        while (f != ready_val) {
            f = ctx.atomic_load<int, iris::memory_scope_system>(
                    &ready[e], local_rank, iris::memory_order_acquire);
        }
    }
    __syncthreads();

    // ---- view slot e%2 as a [slot_rows, K] gl so B0G::load addresses tiles within it ----
    const int slot = e & 1;
    bf16* slot_base = &g.slots[{0, 0, 0, 0}] + (size_t)slot * (size_t)slot_rows * K;
    gl<bf16, -1, -1, -1, -1> A_slot(slot_base, 1, 1, slot_rows, K);

    rt_bf<REG_M, B0_BK, row_l, rt_16x32_s> a;
    rt_bf<REG_N, B0_BK, row_l, rt_16x32_s> b0;
    rt_fl<REG_M, REG_N, col_l, rt_16x16_s> cacc;
    zero(cacc);

    uint32_t soA[64], soB[64];
    B0G::prefill_swizzled_offsets(As[0], A_slot, soA);
    B0G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    const int b_tile_row0 = (e * N) / B0_BN;   // B is expert-major [E*N, K]

    int tic = 0;
    for (int k = 0; k < k_iters; ++k, tic ^= 1) {
        B0G::load(As[tic], A_slot, {0, 0, m_tile * 2 + warp_m, k}, soA);
        B0G::load(Bs[tic], g.b,    {0, 0, b_tile_row0 + n_tile * 2 + warp_n, k}, soB);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        auto as = kittens::subtile_inplace<REG_M, B0_BK>(As[tic], {0, 0});
        auto bs = kittens::subtile_inplace<REG_N, B0_BK>(Bs[tic], {0, 0});
        load(a, as);
        load(b0, bs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cacc, a, b0, cacc);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }
    // store into the expert's global packed C region.
    const int crow_tile0 = (erow_begin / B0_BM);
    store(g.c, cacc, {0, 0, (crow_tile0 + m_tile) * WARPS_ROW * 2 + warp_m,
                      n_tile * WARPS_COL * 2 + warp_n});

    // ---- last block of expert e frees the slot for expert e+2 ----
    // arrive[e] is initialized by the host to tiles_of_e and counts DOWN; the block that observes the
    // pre-decrement value == 1 was the last tile of expert e, and it release-sets done[e+2].
    __syncthreads();
    if (threadIdx.x == 0) {
        int rem = ctx.fetch_sub<int, iris::memory_scope_system>(
                      &arrive[e], 1, local_rank, iris::memory_order_acq_rel);
        if (rem == 1) {   // this was the last tile of expert e
            ctx.fence<iris::memory_scope_system>(iris::memory_order_release);
            // done has E+2 slots; done[e+2] frees this slot for expert e+2 (e+2 <= E+1 always valid).
            ctx.atomic_store<int, iris::memory_scope_system>(
                &done[e + 2], ready_val, local_rank, iris::memory_order_release);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// HOST: two non-blocking streams; producer first (reserves CUs), consumer immediately after, no sync
// between. Host resets ready/arrive/ecursor, PRIMES done[0],done[1] to ready_val (slots free for the
// first two experts), pre-sets done[e+2] for empty experts, and sets arrive[e] = tiles_of_e. gen
// bumped each call. (done.size == E+2; arrive[e] counts DOWN.)
// ------------------------------------------------------------------------------------------------
void dispatch_p2(p2_globals g) {
    const size_t sc = g.consumer_shared();
    hipFuncSetAttribute((void*)p2_consumer, hipFuncAttributeMaxDynamicSharedMemorySize, sc);

    hipStream_t prod_stream, cons_stream;
    hipStreamCreateWithFlags(&prod_stream, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&cons_stream, hipStreamNonBlocking);

    p2_producer<<<g.producer_grid(), g.producer_block(), 0, prod_stream>>>(g);
    p2_consumer<<<g.consumer_grid(), g.consumer_block(), sc, cons_stream>>>(g);

    hipStreamSynchronize(prod_stream);
    hipStreamSynchronize(cons_stream);
    hipStreamDestroy(prod_stream);
    hipStreamDestroy(cons_stream);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "p2_expert_pipeline tk_kernel: expert-granular copy-once double-buffer (overlap)";
    py::bind_function<dispatch_p2>(m, "dispatch_p2",
        &p2_globals::a_src,
        &p2_globals::sc_src,
        &p2_globals::b,
        &p2_globals::c,
        &p2_globals::slots,
        &p2_globals::segs,
        &p2_globals::expert_meta,
        &p2_globals::tasks,
        &p2_globals::ready_gl,
        &p2_globals::done_gl,
        &p2_globals::arrive_gl,
        &p2_globals::ecursor_gl,
        &p2_globals::iris_ctx,
        &p2_globals::E,
        &p2_globals::N,
        &p2_globals::K,
        &p2_globals::Mpacked,
        &p2_globals::Msrc,
        &p2_globals::slot_rows,
        &p2_globals::generation,
        &p2_globals::num_tasks,
        &p2_globals::num_producer_blocks
    );
}
