// sched_8wave / kernel.cpp
// ------------------------------------------------------------------------------------------------
// TRUE HipKittens 8-wave PING-PONG schedule, transplanted onto the IRIS remote-FP8 MoE problem.
//
// What this is, vs V4 (irisx/v4_astationary_kernel/kernel.cpp):
//   V4 uses a PERMANENT producer/consumer split: 4 warps ONLY gather+dequant A (+load B), 4 warps
//   ONLY MFMA.  The two halves never swap jobs; overlap is producer-feeds-consumer across a
//   double-buffer.  Half the waves never issue an MMA; half never issue a load.
//
//   The canonical HK FP8_8wave/8_wave.cu does the OPPOSITE: ONE group<8>, split by
//   warp_m = warpid()/WARPS_COL in {0,1} into TWO symmetric wavegroups.  EVERY wave issues BOTH
//   G::load (feed) AND mma_ABt (compute).  The two wavegroups run a ping-pong: while wavegroup A
//   computes between two s_barrier()s, wavegroup B feeds the next stage; at the next barrier they
//   swap roles.  This is "role-swap", not "role-split".  It is seeded by a conditional prologue
//   barrier (only warp_m==1 hits it, putting the two groups a half-phase out of step), maintained
//   by the dense per-MMA s_barrier() lattice in the K loop, and rebalanced by a conditional
//   epilogue barrier (only warp_m==0) so both groups finish their barrier accounting even.
//
// Output ownership (NO partial-K reduction):
//   Both wavegroups walk the FULL K dimension.  They do NOT each compute a partial sum over half
//   of K that must later be reduced.  Instead warp_m splits the BM output ROWS into two DISJOINT
//   halves: wavegroup 0 owns rows [0, HALF_BM), wavegroup 1 owns rows [HALF_BM, BM).  Each
//   wavegroup gathers its OWN A row-half (HALF_BM x BK) and MFMAs it against the full-K B, so each
//   accumulator is already a COMPLETE dot product over all of K for its rows.  Stores are to
//   disjoint row ranges => no cross-wavegroup add, no atomics, no second reduction kernel.
//   (Contrast V4, where all 8 accumulators cover the same BM rows but different N subtiles.)
//
// Remote-MoE adaptation of the HK feed/compute primitives:
//   * A feed  = IRIS remote FP8 gather + per-128-group dequant of a HALF_BM x BK row-half into a
//               swizzled shared bf16 tile (reuses V4's gather_dequant math verbatim; each
//               wavegroup gathers only its own row-half).
//   * B feed  = local-HBM bf16 load (cheap, ~7 TB/s) — both wavegroups cooperatively load B.
//   * compute = shared->reg (load_st_to_rt) + mma_ABt, NSUB disjoint accumulators per wave.
//
// Parameterized: BM,BN,BK in {32,64}; NSUB in {2,4,8}; NSTAGE in {2,3}.  A V4-style two-phase
// baseline (micro_tk_baseline, no role-swap, no overlap) is kept in-file for apples-to-apples.
// Single-expert only (transplant into Agent 02's grouped scheduler later).
//
// NO GPU was touched producing this file.  Resource numbers are [NEEDS-NODE] (SSH gated: no active
// Conductor reservation at authoring time) — main agent to compile under flock/build_04.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte
static constexpr int QGROUP = 128;   // FP8 block-scale quant group (DeepSeek style)

// ----------------------------- tile / block configuration ---------------------------------------
#ifndef BM
#define BM 64                 // output rows per block (split into two disjoint row-halves)
#endif
#ifndef BN
#define BN 64                 // output cols per N-subtile
#endif
#ifndef BK
#define BK 64                 // K-step
#endif
#ifndef NSUB
#define NSUB 4                // N-subtiles per block (A-stationary reuse, like V4)
#endif
#ifndef NSTAGE
#define NSTAGE 2              // shared-tile buffering depth (2=double, 3=triple)
#endif

// 8 warps, ONE group, split into two wavegroups by warp_m = warpid()/WARPS_COL.
#ifndef WARPS_COL
#define WARPS_COL 4           // warps per wavegroup along N
#endif
#define WARPS_ROW 2           // number of wavegroups (warp_m in {0,1})
#define NUM_WARPS (WARPS_ROW * WARPS_COL)             // = 8
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

#define N_PER_BLOCK (NSUB * BN)
#define HALF_BM (BM / WARPS_ROW)                       // rows owned by ONE wavegroup
#define CONS_N (BN / WARPS_COL)                        // N-cols owned by one wave within a subtile

using G  = kittens::group<NUM_WARPS>;                  // all 8 warps cooperate on B load
// Per-wavegroup group for the A row-half gather (4 warps each). We instead gather with explicit
// thread maths below (mirrors V4) because the source rank / dequant path is custom.

// Shared tiles. A is dequantized to bf16 before MFMA. We keep a SEPARATE A tile per wavegroup
// (its own row-half) and the NSUB B subtiles are shared by both wavegroups.
using ST_A = st_bf<HALF_BM, BK, st_16x32_s>;          // one wavegroup's A row-half
using ST_B = st_bf<BN,      BK, st_16x32_s>;          // full BN; each wave subtiles CONS_N out

struct micro_globals {
    gl<bf16,  -1, -1, -1, -1> a;       // [M, K/2]  (reinterpreted fp8 e4m3 [M,K])
    gl<float, -1, -1, -1, -1> sc;      // [M, K/128]  fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[N,K] local, C[M,N] local
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int fused;                          // 1 = ping-pong role-swap path, 0 = two-phase baseline
    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(N, (int)N_PER_BLOCK), ceil_div(M, (int)BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    // shared: NSTAGE * (2 A row-halves) + NSTAGE * NSUB B subtiles.
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * (WARPS_ROW * sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
    }
};

// fp8 -> float using OCP e4m3 (gfx950 OCP, NOT fnuz).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of a HALF_BM x BK A row-half into the swizzled shared bf16 tile.
// `row_base` is the GLOBAL first row of THIS wavegroup's row-half. All NUM_THREADS cooperate but
// only the calling wavegroup's threads are routed here (we pass the wavegroup's thread span).
// (Math mirrors V4's gather_dequant_A_tile; only the row extent is HALF_BM and the thread pool is
//  one wavegroup = WARPS_COL warps.)
// ------------------------------------------------------------------------------------------------
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_half(
        ST_A &dst, int tile, int row_base, int src_rank,
        const micro_globals &g, int group_tid, int group_threads) {
    const int k0 = tile * BK;
    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    const int K = g.K;
    const int NG = K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;

    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = HALF_BM * CHUNKS_PER_ROW;

    for (int ci = group_tid; ci < total_chunks; ci += group_threads) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gr = row_base + r;
        const int gk = k0 + kc;

        uint4 packed;
        if (gr < g.M && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        const int grp = gk / QGROUP;
        float scale = 1.0f;
        if (gr < g.M && grp < NG) {
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

// ------------------------------------------------------------------------------------------------
// PING-PONG 8-WAVE kernel (role-SWAP). Both wavegroups feed AND compute; they swap each barrier.
//
// Barrier state machine (mirrors FP8_8wave/8_wave.cu):
//   - prologue: prefetch stage 0 (both halves' A + NSUB B); then `if (warp_m==1) s_barrier();`
//     seeds the half-phase offset so the two wavegroups are interleaved.
//   - K loop: per K-tile, per N-subtile, the pattern is
//         feed next stage (G::load B / gather A row-half)  ->  s_barrier
//         s_waitcnt lgkmcnt(0); setprio(1); mma_ABt; setprio(0)  ->  s_barrier
//     so at any instant one wavegroup is inside the MMA region and the other is inside the feed
//     region; the barriers hand the LDS/compute units back and forth (ping-pong).
//   - epilogue: `if (warp_m==0) s_barrier();` rebalances the barrier count so both wavegroups
//     have hit the same number of s_barrier()s before the disjoint-row stores.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE][WARPS_ROW] = al.allocate<ST_A, NSTAGE, WARPS_ROW>();
    ST_B (&Bs)[NSTAGE][NSUB]      = al.allocate<ST_B, NSTAGE, NSUB>();

    const int block_row = blockIdx.y * BM;
    const int block_n0  = blockIdx.x * N_PER_BLOCK;
    const int warp_id   = kittens::warpid();
    const int warp_m    = warp_id / WARPS_COL;          // {0,1} = which wavegroup / row-half
    const int warp_n    = warp_id % WARPS_COL;          // {0..WARPS_COL-1} = N within subtile
    const int laneid    = kittens::laneid();
    const int src_rank  = g.src_rank;
    const int num_tiles = g.K / BK;
    const int n_tile0   = blockIdx.x * NSUB;            // first B N-tile index this block owns

    // This wavegroup's GLOBAL first output row (disjoint halves => no partial-K reduction).
    const int row_base  = block_row + warp_m * HALF_BM;

    // Thread coordinates WITHIN this wavegroup (WARPS_COL warps) for the A row-half gather.
    const int group_tid     = warp_n * kittens::WARP_THREADS + laneid;
    const int group_threads = WARPS_COL * kittens::WARP_THREADS;

    // B swizzle offsets (all 8 warps cooperatively load B).
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    constexpr int PREFETCH = NSTAGE - 1;

    // ---- prologue: prefetch the first PREFETCH K-tiles (both A row-halves + NSUB B subtiles) ----
    #pragma unroll
    for (int s = 0; s < PREFETCH; ++s) {
        if (s < num_tiles) {
            // each wavegroup gathers ITS OWN A row-half
            gather_dequant_A_half<16>(As[s][warp_m], s, row_base, src_rank, g,
                                      group_tid, group_threads);
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub)
                G::load<2, false>(Bs[s][sub], g.b, {0, 0, n_tile0 + sub, s}, swizzled_offsets_B);
        }
    }
    __builtin_amdgcn_s_waitcnt(0);

    // SEED the ping-pong: only the second wavegroup hits this barrier, putting the two groups a
    // half-phase out of step (canonical HK `if (warp_m==1) s_barrier();`).
    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_barrier();

    // NSUB disjoint fp32 accumulators per wave (one per N-subtile it co-owns). Each is a COMPLETE
    // dot product over all of K for THIS wavegroup's row-half => no partial-K reduction.
    rt_fl<HALF_BM, CONS_N, col_l, rt_16x16_s> C_accum[NSUB];
    #pragma unroll
    for (int sub = 0; sub < NSUB; ++sub) zero(C_accum[sub]);

    // ---- main K loop: dense per-step s_barrier lattice maintains the role-swap ping-pong ----
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur   = tile % NSTAGE;
        const int fetch = tile + PREFETCH;
        const int slot  = fetch % NSTAGE;

        // Load THIS wavegroup's A row-half once, reuse across all NSUB N-subtiles (A-stationary).
        rt_bf<HALF_BM, BK, row_l, rt_16x32_s> a_frag;
        load(a_frag, As[cur][warp_m]);
        asm volatile("s_waitcnt lgkmcnt(0)");

        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
            // FEED phase: prefetch next stage for this subtile (B local-HBM) while the OTHER
            // wavegroup is in its MMA phase. A row-half is prefetched on sub==0 only.
            if (fetch < num_tiles) {
                if (sub == 0)
                    gather_dequant_A_half<16>(As[slot][warp_m], fetch, row_base, src_rank, g,
                                              group_tid, group_threads);
                G::load<2, false>(Bs[slot][sub], g.b, {0, 0, n_tile0 + sub, fetch},
                                  swizzled_offsets_B);
            }
            __builtin_amdgcn_s_barrier();   // hand off: feed done -> other group may compute

            // COMPUTE phase: MFMA this subtile against the (already-loaded) A row-half.
            rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
            auto b_sub = subtile_inplace<CONS_N, BK>(Bs[cur][sub], {warp_n, 0});
            load(b_frag, b_sub);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[sub], a_frag, b_frag, C_accum[sub]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();   // hand off: compute done -> other group may feed
        }
    }

    // ---- epilogue: rebalance the barrier count (canonical HK `if (warp_m==0) s_barrier();`) ----
    if (warp_m == 0) __builtin_amdgcn_s_barrier();

    // ---- disjoint-row stores: each wavegroup writes ONLY its own HALF_BM rows => no reduction ----
    #pragma unroll
    for (int sub = 0; sub < NSUB; ++sub) {
        const int out_col0 = block_n0 + sub * BN + warp_n * CONS_N;
        // C tile coords: row tile index uses row_base (disjoint per wavegroup).
        store(g.c, C_accum[sub], {0, 0, row_base / HALF_BM, out_col0 / CONS_N});
    }
}

// ------------------------------------------------------------------------------------------------
// BASELINE (two-phase, NO role-swap, NO overlap) — apples-to-apples reference, V4/V3-style.
// grid = (N/BN, M/BM), one output tile per block, full producer/consumer barrier per K-tile.
// 4 warps gather A (full BM), 4 warps MFMA. This is the unfused reference the ping-pong must beat.
// ------------------------------------------------------------------------------------------------
using ST_A_b = st_bf<BM, BK, st_16x32_s>;
using ST_B_b = st_bf<BN, BK, st_16x32_s>;
#define NUM_PRODUCER_WORKERS_B (NUM_WARPS / 2)
#define NUM_PRODUCER_THREADS_B (NUM_PRODUCER_WORKERS_B * kittens::WARP_THREADS)
using PG_b = kittens::group<NUM_PRODUCER_WORKERS_B>;

// full-BM gather (baseline) reusing the same dequant math.
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_full(
        ST_A_b &dst, int tile, int block_row, int src_rank,
        const micro_globals &g, int warp_id, int laneid) {
    const int k0 = tile * BK;
    const int tid = warp_id * kittens::WARP_THREADS + laneid;
    constexpr int SUBR = ST_A_b::underlying_subtile_rows;
    constexpr int SUBC = ST_A_b::underlying_subtile_cols;
    constexpr int SUBN = ST_A_b::underlying_subtile_elements;
    const int K = g.K;
    const int NG = K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;
    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = BM * CHUNKS_PER_ROW;
    for (int ci = tid; ci < total_chunks; ci += NUM_PRODUCER_THREADS_B) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gr = block_row + r;
        const int gk = k0 + kc;
        uint4 packed;
        if (gr < g.M && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else packed = make_uint4(0u, 0u, 0u, 0u);
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);
        const int grp = gk / QGROUP;
        float scale = 1.0f;
        if (gr < g.M && grp < NG) { const float* sptr = &g.sc[{0,0,gr,grp}]; scale = ctx.load(sptr, src_rank); }
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            bf16 val = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_A_b::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_A_b::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk_baseline(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_b (&As)[1] = al.allocate<ST_A_b, 1>();
    ST_B_b (&Bs)[1] = al.allocate<ST_B_b, 1>();

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS_B);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS_B);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS_B) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS_B;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG_b::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);

    constexpr int CONS_N_B = BN / (NUM_WARPS - NUM_PRODUCER_WORKERS_B);
    rt_fl<BM, CONS_N_B, col_l, rt_16x16_s> C_accum;
    zero(C_accum);

    for (int tile = 0; tile < num_tiles; ++tile) {
        if (is_producer) {
            gather_dequant_A_full<16>(As[0], tile, block_row, src_rank, g, warp_id, laneid);
            PG_b::load<2, false>(Bs[0], g.b, {0, 0, (int)blockIdx.x, tile}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        }
        __syncthreads();
        if (is_consumer) {
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            rt_bf<CONS_N_B, BK, row_l, rt_16x32_s> b_frag;
            load(a_frag, As[0]);
            auto b_sub = subtile_inplace<CONS_N_B, BK>(Bs[0], {cons_id, 0});
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
        const int out_col0 = block_col + cons_id * CONS_N_B;
        store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N_B});
    }
}

void dispatch_micro(micro_globals g) {
    if (g.fused) {
        const unsigned long mem_size = g.dynamic_shared_memory();
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    } else {
        const unsigned long mem_size = (unsigned long)(sizeof(ST_A_b) + sizeof(ST_B_b)) + 1024;
        hipFuncSetAttribute((void*)micro_tk_baseline, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        dim3 bgrid(ceil_div(g.N, (int)BN), ceil_div(g.M, (int)BM));
        micro_tk_baseline<<<bgrid, g.block(), mem_size, g.stream>>>(g);
    }
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "sched_8wave (HK 8-wave ping-pong role-swap) tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a,
        &micro_globals::sc,
        &micro_globals::b,
        &micro_globals::c,
        &micro_globals::iris_ctx,
        &micro_globals::M,
        &micro_globals::N,
        &micro_globals::K,
        &micro_globals::src_rank,
        &micro_globals::fused
    );
}
