// fmoe_sched_4wave / kernel.cpp
// ------------------------------------------------------------------------------------------------
// SCHED-4WAVE — symmetric 4-wave LATENCY kernel for the small-M MoE expert-GEMM path.
//
// WHY THIS EXISTS (the V4 small-M loss it fixes):
//   V4 (irisx/v4_astationary_kernel) is a producer/consumer kernel: 4 permanent producer waves
//   gather+dequant the remote fp8 A-tile over IRIS, 4 permanent consumer waves MFMA.  The two
//   wave-sets hand off across an s_barrier every K-tile.  At LARGE M this overlaps gather and MFMA
//   well (V4 wins M>=512).  But at SMALL M (M<=256) there is too little MFMA work to hide:
//     - half the waves (the consumers) sit idle during the gather, half (producers) sit idle during
//       the MFMA — the s_barrier serializes the two halves at the K-tile granularity;
//     - with M<=256 each block has only BM/BN rows of real work, so MFMA can't cover gather latency
//       and the barrier hand-off cost dominates.  Measured: V4 = 0.85-0.90x vs baseline at M<=256.
//
// THE FIX (symmetric, no producer/consumer split):
//   ALL 4 waves (one per SIMD on a CU) are IDENTICAL.  Each wave:
//     - owns its OWN strip of CONS_M = BM/4 rows of the output tile;
//     - issues BOTH the IRIS remote fp8 gather+dequant of ITS rows AND the MFMA on ITS rows;
//     - carries its own in-flight cross-GPU load.  Because every wave both loads and computes,
//       the gather latency of wave w on K-tile (t+PREFETCH) is hidden by wave w's OWN MFMA on
//       K-tile t — within the SAME wave, NOT across a wasted producer/consumer s_barrier.
//   There is no inter-wave hand-off barrier in the steady loop; each wave is a self-contained
//   software pipeline.  This is the HipKittens FP8_4wave latency-path idea (4_wave.cu:
//   do_interleaved_cluster fine-grains load_one against mma_one), applied to the IRIS remote-gather
//   MoE setting.
//
// PER-WAVE SOFTWARE PIPELINE (NOT all-load-then-compute):
//   prologue: each wave prefetches K-tile 0 (gather+dequant A strip into LDS, load B subtiles);
//   steady loop tile t:  issue gather+B-load for tile (t+PREFETCH) into the other LDS buffer, THEN
//     MFMA tile t out of resident LDS, interleaving per-subtile B-fragment loads with the MFMA
//     under s_setprio so the in-flight VMEM gather overlaps the compute of the previous tile.
//
// LOW REGISTER PRESSURE (=> 4 waves/SIMD occupancy target):
//   Waves SPLIT the M dimension: CONS_M = BM/4.  Each wave keeps only NSUB (1-2) live fp32
//   accumulator tiles of shape rt_fl<CONS_M, BN_sub> — vs V4's 8 live accumulators per consumer
//   wave.  With BM=32, NSUB=2, CONS_M=8 the accumulator footprint is tiny, freeing VGPRs so 4 waves
//   can be resident per SIMD (latency hiding via occupancy ON TOP OF the per-wave pipeline).  A is
//   gathered ONCE per K-tile into LDS and reused across NSUB (keeps V4's cross-GPU-traffic win:
//   A crosses the interconnect N/(NSUB*BN) times, not N/BN times).
//
// ABI / numerics: IDENTICAL to V4 — fp8 e4m3 (OCP) remote gather, per-128-group fp32 dequant,
//   B/C bf16, zero-sentinel on the non-source rank, RMS-rel correctness.  A matching two-phase
//   baseline (micro_tk_baseline, V4-identical) is included in-file for a head-to-head.
//
// STORE CONVENTION (confirmed against V4): store(g.c, accum, {0,0,row_tile,col_tile}) indexes in
//   ACCUMULATOR-TILE units, i.e. row_tile = out_row0 / CONS_M, col_tile = out_col0 / BN_sub.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte
static constexpr int QGROUP = 128;   // FP8 block-scale quant group

// ----------------------------- tile / block configuration ---------------------------------------
// Defaults tuned for the small-M latency path. Sweep: M={8,16,32,64,128,256}, BM={16,32},
// BN={32,64}, BK={32,64}, NSUB={1,2,4}.
#ifndef BM
#define BM 32
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef NSUB
#define NSUB 2
#endif
// Symmetric: one wave per SIMD, all identical (load + compute). No producer/consumer split.
#ifndef NUM_WORKERS
#define NUM_WORKERS 4
#endif
#ifndef NSTAGE
#define NSTAGE 2            // per-wave LDS double-buffer depth
#endif

#define N_PER_BLOCK (NSUB * BN)
#define NUM_WARPS NUM_WORKERS
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
// Each wave owns CONS_M rows of the BM-row tile.
#define CONS_M (BM / NUM_WORKERS)

// Shared tile types (bf16 — A is dequantized into bf16 before MFMA).
// A LDS tile holds the full BM rows (all waves' strips); each wave writes/reads its CONS_M slice.
using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

struct micro_globals {
    gl<bf16,  -1, -1, -1, -1> a;       // [M, K/2]  (reinterpreted fp8 e4m3 [M,K])
    gl<float, -1, -1, -1, -1> sc;      // [M, K/128]  fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[N,K] local, C[M,N] local
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int fused;                          // 1 = fused 4-wave path, 0 = two-phase baseline path
    hipStream_t stream;
    // grid x walks N in N_PER_BLOCK steps (each block owns NSUB N-subtiles); grid y walks M in BM.
    dim3 grid()  { return dim3(ceil_div(N, (int)N_PER_BLOCK), ceil_div(M, (int)BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    // shared: NSTAGE A tiles (1 each, reused across N) + NSTAGE * NSUB B tiles.
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * (sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
    }
};

// fp8 -> float using OCP e4m3 (matches V1/V4's __HIP_E4M3).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of THIS WAVE'S CONS_M x BK strip of the A-tile into the swizzled
// shared bf16 tile `dst`.  Only this wave's lanes participate (symmetric: every wave gathers its
// own rows).  `wave_row0` = first row this wave owns within the BM tile (= cons_id * CONS_M).
// Layout/dequant identical to V4's gather_dequant_A_tile; the difference is the row range + that
// every wave (not just producers) runs it.
// ------------------------------------------------------------------------------------------------
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_strip(
        ST_A &dst, int tile, int block_row, int wave_row0, int src_rank,
        const micro_globals &g, int laneid) {
    const int k0 = tile * BK;
    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    const int K = g.K;
    const int NG = K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;

    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = CONS_M * CHUNKS_PER_ROW;   // only this wave's rows

    // Each wave uses ITS OWN 64 lanes (one warp) to gather its CONS_M-row strip.
    for (int ci = laneid; ci < total_chunks; ci += kittens::WARP_THREADS) {
        const int r  = ci / CHUNKS_PER_ROW;             // 0..CONS_M-1 (wave-local row)
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int tile_row = wave_row0 + r;             // 0..BM-1 within the LDS tile
        const int gr = block_row + tile_row;            // global A row
        const int gk = k0 + kc;

        uint4 packed;
        if (gr < g.M && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        const int grp = (gk) / QGROUP;
        float scale = 1.0f;
        if (gr < g.M && grp < NG) {
            const float* sptr = &g.sc[{0, 0, gr, grp}];
            scale = ctx.load(sptr, src_rank);
        }

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            bf16 val = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
            // write into the FULL-BM swizzled tile at row=tile_row
            const int sub_row = tile_row / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_A::underlying_subtiles_per_row + sub_col;
            const int rr = tile_row % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_A::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    }
}

// Per-wave local B-subtile load: each wave loads the CONS_M-irrelevant full BN x BK B subtile it
// MFMAs against (B is local HBM, cheap). Single-warp cooperative load into the swizzled LDS tile.
template<int VEC>
__device__ __forceinline__ void load_B_subtile(
        ST_B &dst, int n_tile, int tile, const micro_globals &g, int laneid) {
    constexpr int SUBR = ST_B::underlying_subtile_rows;
    constexpr int SUBC = ST_B::underlying_subtile_cols;
    constexpr int SUBN = ST_B::underlying_subtile_elements;
    const int K = g.K;
    const int k0 = tile * BK;
    const int n0 = n_tile * BN;
    const bf16* b_base = reinterpret_cast<const bf16*>(&g.b[{0, 0, 0, 0}]);
    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = BN * CHUNKS_PER_ROW;
    for (int ci = laneid; ci < total_chunks; ci += kittens::WARP_THREADS) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gn = n0 + r;
        const int gk = k0 + kc;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            bf16 val = (gn < g.N && (gk + j) < K) ? b_base[(size_t)gn * K + gk + j] : (bf16)0;
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_B::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_B::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// SYMMETRIC 4-WAVE FUSED kernel.  Every wave is identical: gather its own A strip + load B subtiles
// + MFMA its own CONS_M rows.  Per-wave software pipeline hides gather latency behind the SAME
// wave's MFMA of the previous K-tile.  No producer/consumer s_barrier in the steady loop.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE]       = al.allocate<ST_A, NSTAGE>();
    ST_B (&Bs)[NSTAGE][NSUB] = al.allocate<ST_B, NSTAGE, NSUB>();

    const int block_row = blockIdx.y * BM;
    const int block_n0  = blockIdx.x * N_PER_BLOCK;     // first N column this block owns
    const int warp_id   = kittens::warpid();            // 0..NUM_WORKERS-1, one per SIMD
    const int laneid    = kittens::laneid();
    const int src_rank  = g.src_rank;
    const int num_tiles = g.K / BK;
    const int n_tile0   = blockIdx.x * NSUB;            // first B N-tile index this block owns
    const int wave_row0 = warp_id * CONS_M;            // this wave's row offset within the BM tile

    constexpr int PREFETCH = NSTAGE - 1;

    // -------- Prologue: every wave prefetches the first PREFETCH K-tiles (A strip + NSUB B) -------
    #pragma unroll
    for (int s = 0; s < PREFETCH; ++s) {
        if (s < num_tiles) {
            gather_dequant_A_strip<16>(As[s], s, block_row, wave_row0, src_rank, g, laneid);
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub)
                load_B_subtile<16>(Bs[s][sub], n_tile0 + sub, s, g, laneid);
        }
    }
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();   // make prologue LDS visible (A strips written by sibling waves)

    // Each wave keeps only NSUB live fp32 accumulators of its CONS_M-row strip => low VGPR pressure.
    rt_fl<CONS_M, BN, col_l, rt_16x16_s> C_accum[NSUB];
    #pragma unroll
    for (int sub = 0; sub < NSUB; ++sub) zero(C_accum[sub]);

    // -------- Steady-state per-wave software pipeline --------
    // tile t: issue the gather+B-load for (t+PREFETCH) into the next LDS slot, THEN MFMA tile t out
    // of the resident LDS slot.  The in-flight VMEM (remote gather) of (t+PREFETCH) overlaps the
    // MFMA of t WITHIN THIS WAVE — no producer/consumer hand-off.
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur   = tile % NSTAGE;
        const int fetch = tile + PREFETCH;

        // Issue next K-tile's loads (non-blocking VMEM) — this is the latency we hide.
        if (fetch < num_tiles) {
            const int slot = fetch % NSTAGE;
            gather_dequant_A_strip<16>(As[slot], fetch, block_row, wave_row0, src_rank, g, laneid);
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub)
                load_B_subtile<16>(Bs[slot][sub], n_tile0 + sub, fetch, g, laneid);
        }

        // Load THIS wave's A fragment (its CONS_M rows) ONCE, reuse across NSUB N-subtiles.
        rt_bf<CONS_M, BK, row_l, rt_16x32_s> a_frag;
        auto a_sub = subtile_inplace<CONS_M, BK>(As[cur], {warp_id, 0});
        load(a_frag, a_sub);
        asm volatile("s_waitcnt lgkmcnt(0)");

        // Interleave per-subtile B-fragment load with the MFMA under s_setprio so the in-flight
        // remote gather (issued above) overlaps compute.
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
            rt_bf<BN, BK, row_l, rt_16x32_s> b_frag;
            load(b_frag, Bs[cur][sub]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[sub], a_frag, b_frag, C_accum[sub]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);
        }
        // Single lightweight barrier so the NEXT iteration's A-strip writes (by sibling waves) are
        // visible before this wave loads them. NOT a producer/consumer hand-off — all waves both
        // produced and consumed this iteration.
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue: each wave stores its NSUB CONS_M-row accumulator tiles.
    // store indexes in accumulator-tile units: row_tile = out_row0/CONS_M, col_tile = out_col0/BN.
    #pragma unroll
    for (int sub = 0; sub < NSUB; ++sub) {
        const int out_row0 = block_row + wave_row0;
        const int out_col0 = block_n0 + sub * BN;
        store(g.c, C_accum[sub], {0, 0, out_row0 / CONS_M, out_col0 / BN});
    }
}

// ------------------------------------------------------------------------------------------------
// BASELINE kernel (two-phase, NO overlap) — symmetric-4-wave analogue of V4's baseline.  Grid =
// (N/BN, M/BM), one output tile per block.  All 4 waves gather (split by rows) then s_barrier then
// all 4 waves MFMA.  This is the unfused reference the fused symmetric kernel must beat at small M.
// Kept structurally matched to the fused path so the head-to-head isolates the per-wave pipeline.
// ------------------------------------------------------------------------------------------------
using ST_A_b = st_bf<BM, BK, st_16x32_s>;
using ST_B_b = st_bf<BN, BK, st_16x32_s>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk_baseline(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_b (&As)[1] = al.allocate<ST_A_b, 1>();
    ST_B_b (&Bs)[1] = al.allocate<ST_B_b, 1>();

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int warp_id   = kittens::warpid();
    const int laneid    = kittens::laneid();
    const int src_rank  = g.src_rank;
    const int num_tiles = g.K / BK;
    const int wave_row0 = warp_id * CONS_M;

    rt_fl<CONS_M, BN, col_l, rt_16x16_s> C_accum;
    zero(C_accum);

    for (int tile = 0; tile < num_tiles; ++tile) {
        // Phase 1: every wave gathers its own A strip + (wave 0..NSUB) loads B.
        gather_dequant_A_strip<16>(As[0], tile, block_row, wave_row0, src_rank, g, laneid);
        if (warp_id == 0) load_B_subtile<16>(Bs[0], (int)blockIdx.x, tile, g, laneid);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        // Phase 2: every wave MFMAs its own CONS_M rows.
        rt_bf<CONS_M, BK, row_l, rt_16x32_s> a_frag;
        rt_bf<BN, BK, row_l, rt_16x32_s> b_frag;
        auto a_sub = subtile_inplace<CONS_M, BK>(As[0], {warp_id, 0});
        load(a_frag, a_sub);
        load(b_frag, Bs[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a_frag, b_frag, C_accum);
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    const int out_row0 = block_row + wave_row0;
    const int out_col0 = block_col;
    store(g.c, C_accum, {0, 0, out_row0 / CONS_M, out_col0 / BN});
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
    m.doc() = "fmoe_sched_4wave tk_kernel python module";
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
