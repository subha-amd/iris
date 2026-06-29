// SPDX-License-Identifier: MIT
// ================================================================================================
// grouped_b0 — GROUPED 32-expert MoE gate/up GEMM using the *B0-class* HipKittens schedule.
//
// WHY THIS FILE EXISTS
// --------------------
// The b1_dispatch phase-2 grouped GEMM (reference/v5_grouped/kernel.cpp::micro_tk) reuses the V3/V4
// "fused direct-pull" body: a 64x64x64 tile with a permanent 4-producer/4-consumer wave split at
// occupancy 1. On AMD that schedule is slow (only 4 of 8 waves ever issue an MFMA; tiny tile =>
// poor arithmetic intensity). Measured: ~32-69 TFLOP/s (EXPERIMENT_LEDGER, B1-dispatch V0/V1).
//
// The project ALSO has the *correct* HK GEMM: the 256x256x64, 8-wave ping-pong body in
// reference/v2_hk_expert_gemm/fmoe_expert_v2.cu::expert_gemm_bf16 (a faithful port of upstream
// HipKittens kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu). As B0 it hits ~183 TFLOP/s (the compute
// ceiling). ALL 8 waves issue MFMA; memory/compute overlap comes from double-buffering + the
// ping-pong half-barriers, NOT from sacrificing waves to be pure loaders.
//
// grouped_b0 = that proven B0 body, byte-identical schedule, with ONLY the per-block tile-base
// indices remapped from a flat (block_row, block_col) decode to a per-task (expert, m_tile, n_tile)
// decode (the grouping idea from reference/v5_grouped). Nothing in the delicate K-loop / waitcnt /
// barrier discipline is touched. This is the "fix the wrong tile + wrong schedule" kernel.
//
// SCOPE (first cut, deliberately narrow so it is auditable + buildable without MPI/IRIS):
//   - Phase-2 GEMM ONLY. A is assumed already gathered+packed LOCALLY (expert-major, BM-padded),
//     exactly the buffer b1_dispatch phase-1 produces. No remote gather here (that stays a separate
//     kernel; see B1_DISPATCH_STATUS.md / the "copy-once then local GEMM" dataflow).
//   - Dequant preamble fp8 e4m3 (+per-128 fp32 scale) -> bf16, then bf16 MMA (same as B0/B1-copy).
//     Native-FP8 MMA is a SEPARATE later track (HK's in-MMA scaled path is MX-e8m0/32 only; our
//     scale is fp32/128 — see V2_HK_ANALYSIS.md sec 4).
//   - Single GPU, self-contained main(): host-synthesizes a ragged grouped packed buffer, builds the
//     task list, runs dequant + grouped GEMM, checks correctness vs a CPU fp32 reference, and times.
//
// TILE GEOMETRY (inherited verbatim from the B0 body):
//   BLOCK 256x256 output, 8 warps as WARPS_ROW=2 x WARPS_COL=4, each warp owns 4 fp32 accumulator
//   quadrants (cA,cB,cC,cD = REG 64x32 each). A/B half-tiles are 128 rows; BLOCK_K=64.
//   => the host MUST pad each expert's packed-row count up to a multiple of 256 (BM), so a 256-row
//      block stays inside ONE expert (no cross-expert contamination) and padded rows (which are 0 in
//      the packed buffer) produce 0 outputs into dead C rows no consumer reads.
//
// BUILD (on the node, inside the HK checkout; single source, no MPI/IRIS — like the V2 probe):
//   cd <HK_ROOT>/kernels   # or any dir where THUNDERKITTENS_ROOT resolves
//   /opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
//       -I<HK_ROOT>/include -I/opt/rocm/include/hip grouped_b0.cu -o grouped_b0
//   ./grouped_b0            # runs a ragged-correctness case then a perf case; prints TFLOP/s
//
// COMPARE THE PRINTED TFLOP/s TO:  micro_tk ~69 (the body we are replacing) and B0 ~183 (ceiling).
// ================================================================================================
#include "kittens.cuh"
#include <hip/hip_fp8.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace kittens;

// ---- shapes: DeepSeek-R1 W13 gate/up slice (matches the project's grouped case) ----
static constexpr int N        = 2048;            // gate/up output cols per expert
static constexpr int K        = 7168;            // hidden / reduction dim
static constexpr int GROUP    = 128;             // per-K-group fp8 block-scale span
static constexpr int N_GROUPS = K / GROUP;       // 56
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr int BM = 256;                   // packed-row padding granularity (== BLOCK_SIZE_ROW)

using fp8_t = __hip_fp8_storage_t;               // raw byte, matches b1_dispatch packed_fp8

// flat task tuple (one per 256x256 output tile of one expert)
static constexpr int TASK_W = 4;
enum { T_EXPERT = 0, T_MTILE = 1, T_NTILE = 2, T_EROWBEG = 3 };

// ================================================================================================
// Dequant preamble: packed_fp8[Mpacked,K] (+ packed_sc[Mpacked,N_GROUPS] fp32) -> A_bf16[Mpacked,K].
// One block per packed row; threads stride over K. Padding rows are already 0 in packed_fp8 (the
// gather masks them), so they dequant to 0 with no special-casing. Identical numerics to B0/B1-copy.
// ================================================================================================
__global__ void dequant_a_dense(const fp8_t* __restrict__ a_fp8,
                                const float* __restrict__ a_sc,
                                bf16* __restrict__ a_bf16,
                                int Mpacked) {
    const int row = blockIdx.x;
    if (row >= Mpacked) return;
    const fp8_t* frow = a_fp8 + (size_t)row * K;
    const float* srow = a_sc  + (size_t)row * N_GROUPS;
    bf16* orow = a_bf16 + (size_t)row * K;
    for (int h = threadIdx.x; h < K; h += blockDim.x) {
        __hip_fp8_e4m3 f; f.__x = frow[h];
        orow[h] = (bf16)(static_cast<float>(f) * srow[h / GROUP]);
    }
}

// ================================================================================================
// GROUPED B0 GEMM:  C[m, :] = A[m, :] . B[expert(m)*N + :, :]^T   for every active packed row m.
//
// Body is reference/v2_hk_expert_gemm/fmoe_expert_v2.cu::expert_gemm_bf16 VERBATIM, except the four
// tile-base indices, which are now derived from this block's task tuple instead of a flat blockIdx
// decode.  The remap (all integer because expert_row_begin is a multiple of BM=256):
//     a_row_tile = ERB/128 + mt*2        (A half-tiles are 128 rows; replaces block_row*2)
//     b_row_tile = (e*N)/128 + nt*2      (B is expert-major [E*N,K]; replaces block_col*2)
//     c_row_tile = ERB/64  + mt*4        (C reg-tiles are 64 rows;  replaces block_row*WARPS_ROW*2)
//     c_col_tile = nt*8                  (C reg-tiles are 32 cols;  replaces block_col*WARPS_COL*2)
// Everything else — the prologue, the #pragma unroll 2 steady state, the drain epilogue, every
// s_waitcnt / s_setprio / s_barrier / sched_barrier — is the proven schedule, unchanged.
// ================================================================================================
constexpr int NUM_WARPS = 8;
using G = kittens::group<NUM_WARPS>;

template <int NN, int KK>
__global__ __launch_bounds__(512, 2)
void grouped_expert_gemm(const kittens::gl<bf16, -1, -1, -1, -1> A,   // [Mpacked, K]
                         const kittens::gl<bf16, -1, -1, -1, -1> B,   // [E*N,     K]  expert-major
                         const kittens::gl<bf16, -1, -1, -1, -1> C,   // [Mpacked, N]
                         const int* __restrict__ tasks, int num_tasks) {
    constexpr int WARPS_COL = 4;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256;
    constexpr int BLOCK_SIZE_COL = 256;
    constexpr int BLOCK_K = 64;
    constexpr int k_iters = KK / BLOCK_K;        // 112
    constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2;
    constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2;
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW / 2;
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL / 2;

    using ST_A = st_bf<HALF_BLOCK_SIZE_ROW, BLOCK_K, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE_COL, BLOCK_K, st_16x32_s>;
    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    using RT_A = rt_bf<REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s>;
    using RT_B = rt_bf<REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s>;
    using RT_C = rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s>;

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;

    // ---- task decode (the ONLY structural change vs the B0 body) ----
    const int task = blockIdx.x;
    if (task >= num_tasks) return;
    const int* tk = tasks + (size_t)task * TASK_W;
    const int e   = tk[T_EXPERT];
    const int mt  = tk[T_MTILE];
    const int nt  = tk[T_NTILE];
    const int ERB = tk[T_EROWBEG];               // expert_row_begin, multiple of BM=256

    const int a_row_tile = ERB / 128 + mt * 2;   // 128-row A half-tile base
    const int b_row_tile = (e * NN) / 128 + nt * 2; // 128-row B half-tile base (expert-major)
    const int c_row_tile = ERB / 64 + mt * 4;    // 64-row  C reg-tile base
    const int c_col_tile = nt * 8;               // 32-col  C reg-tile base

    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);

    int tic = 0, toc = 1;

    uint32_t swizzled_offsets_A[64];
    uint32_t swizzled_offsets_B[64];
    G::prefill_swizzled_offsets(As[tic][0], A, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[tic][0], B, swizzled_offsets_B);

    zero(cA); zero(cB); zero(cC); zero(cD);

    G::load(Bs[tic][0], B, {0, 0, b_row_tile,     0}, swizzled_offsets_B);
    G::load(As[tic][0], A, {0, 0, a_row_tile,     0}, swizzled_offsets_A);
    G::load(Bs[tic][1], B, {0, 0, b_row_tile + 1, 0}, swizzled_offsets_B);
    G::load(As[tic][1], A, {0, 0, a_row_tile + 1, 0}, swizzled_offsets_A);

    if (warp_m == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], B, {0, 0, b_row_tile,     1}, swizzled_offsets_B);
    G::load(As[toc][0], A, {0, 0, a_row_tile,     1}, swizzled_offsets_A);
    G::load(Bs[toc][1], B, {0, 0, b_row_tile + 1, 1}, swizzled_offsets_B);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        auto bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        G::load(As[toc][1], A, {0, 0, a_row_tile + 1, k + 1}, swizzled_offsets_A);
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
        G::load(Bs[tic][0], B, {0, 0, b_row_tile, k + 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        G::load(As[tic][0], A, {0, 0, a_row_tile, k + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, b_row_tile + 1, k + 2}, swizzled_offsets_B);
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
        G::load(As[toc][1], A, {0, 0, a_row_tile + 1, k + 1}, swizzled_offsets_A);
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

// ================================================================================================
//                                       host driver
// ================================================================================================
static inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) { fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e)); exit(1); }
}

// Run one grouped case. `Me` = per-expert real row counts (length E). Pads each expert up to a
// multiple of BM, builds the packed buffer / task list, runs dequant + grouped GEMM, optionally
// checks correctness vs a CPU fp32 reference, and times.
static bool run_case(const char* label, const std::vector<int>& Me, bool full_check) {
    const int E = (int)Me.size();
    std::vector<int> padded(E), erb(E);
    int Mpacked = 0;
    for (int e = 0; e < E; e++) {
        padded[e] = ((Me[e] + BM - 1) / BM) * BM;
        erb[e] = Mpacked;
        Mpacked += padded[e];
    }
    int real_rows = 0; for (int v : Me) real_rows += v;

    // ---- task list: one 256x256 output tile per (expert, m_tile, n_tile) ----
    std::vector<int> tasks;
    for (int e = 0; e < E; e++)
        for (int mt = 0; mt < padded[e] / BM; mt++)
            for (int nt = 0; nt < N / 256; nt++) {
                tasks.push_back(e); tasks.push_back(mt); tasks.push_back(nt); tasks.push_back(erb[e]);
            }
    const int num_tasks = (int)tasks.size() / TASK_W;

    printf("\n=== CASE %s: E=%d, real_rows=%d, Mpacked=%d (pad waste %.1f%%), tasks=%d ===\n",
           label, E, real_rows, Mpacked, 100.0 * (Mpacked - real_rows) / std::max(1, Mpacked), num_tasks);

    const size_t a_elems  = (size_t)Mpacked * K;
    const size_t sc_elems = (size_t)Mpacked * N_GROUPS;
    const size_t b_elems  = (size_t)E * N * K;
    const size_t c_elems  = (size_t)Mpacked * N;

    // ---- host synth: packed_fp8 / packed_sc (active rows random, padding rows 0) + true-A ref ----
    std::vector<fp8_t> h_fp8(a_elems, 0);
    std::vector<float> h_sc(sc_elems, 0.0f);
    std::vector<float> h_aref(a_elems, 0.0f);
    std::vector<int>   row_expert(Mpacked, -1);
    std::vector<char>  row_active(Mpacked, 0);
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int e = 0; e < E; e++)
        for (int s = 0; s < padded[e]; s++) {
            int row = erb[e] + s;
            row_expert[row] = e;
            bool active = s < Me[e];
            row_active[row] = active ? 1 : 0;
            if (!active) continue;                     // padding row -> stays 0
            for (int g = 0; g < N_GROUPS; g++) {
                float vals[GROUP]; float amax = 0.0f;
                for (int j = 0; j < GROUP; j++) { float v = dist(gen); vals[j] = v; amax = std::max(amax, std::fabs(v)); }
                float scale = (amax > 0.0f) ? (amax / FP8_E4M3_MAX) : 1.0f;
                float inv = 1.0f / scale;
                h_sc[(size_t)row * N_GROUPS + g] = scale;
                for (int j = 0; j < GROUP; j++) {
                    int h = g * GROUP + j;
                    fp8_t q = __hip_cvt_float_to_fp8(vals[j] * inv, __HIP_SATFINITE, __HIP_E4M3);
                    h_fp8[(size_t)row * K + h] = q;
                    __hip_fp8_e4m3 f; f.__x = q;
                    h_aref[(size_t)row * K + h] = static_cast<float>(f) * scale;  // matches device dequant
                }
            }
        }

    // ---- weight B as plain bf16 per expert (bring-up: real R1 weights not needed for the schedule) ----
    std::vector<bf16> h_b(b_elems);
    std::vector<float> h_bref(b_elems);
    for (size_t i = 0; i < b_elems; i++) { float v = dist(gen) * 0.1f; h_b[i] = (bf16)v; h_bref[i] = (float)(bf16)v; }

    // ---- device buffers ----
    fp8_t* d_fp8; float* d_sc; bf16 *d_a, *d_b, *d_c; int* d_tasks;
    hip_check(hipMalloc(&d_fp8, a_elems * sizeof(fp8_t)), "malloc fp8");
    hip_check(hipMalloc(&d_sc,  sc_elems * sizeof(float)), "malloc sc");
    hip_check(hipMalloc(&d_a,   a_elems * sizeof(bf16)), "malloc a");
    hip_check(hipMalloc(&d_b,   b_elems * sizeof(bf16)), "malloc b");
    hip_check(hipMalloc(&d_c,   c_elems * sizeof(bf16)), "malloc c");
    hip_check(hipMalloc(&d_tasks, tasks.size() * sizeof(int)), "malloc tasks");
    hip_check(hipMemcpy(d_fp8, h_fp8.data(), a_elems * sizeof(fp8_t), hipMemcpyHostToDevice), "cp fp8");
    hip_check(hipMemcpy(d_sc,  h_sc.data(),  sc_elems * sizeof(float), hipMemcpyHostToDevice), "cp sc");
    hip_check(hipMemcpy(d_b,   h_b.data(),   b_elems * sizeof(bf16), hipMemcpyHostToDevice), "cp b");
    hip_check(hipMemcpy(d_tasks, tasks.data(), tasks.size() * sizeof(int), hipMemcpyHostToDevice), "cp tasks");
    hip_check(hipMemset(d_c, 0, c_elems * sizeof(bf16)), "memset c");

    // ---- dequant preamble + grouped GEMM ----
    kittens::gl<bf16, -1, -1, -1, -1> A(d_a, 1, 1, Mpacked, K);
    kittens::gl<bf16, -1, -1, -1, -1> Bg(d_b, 1, 1, E * N, K);
    kittens::gl<bf16, -1, -1, -1, -1> Cg(d_c, 1, 1, Mpacked, N);
    const int threads = NUM_WARPS * 64;

    auto run_once = [&]() {
        dequant_a_dense<<<Mpacked, 256>>>(d_fp8, d_sc, d_a, Mpacked);
        grouped_expert_gemm<N, K><<<num_tasks, threads>>>(A, Bg, Cg, d_tasks, num_tasks);
    };

    for (int i = 0; i < 5; i++) run_once();             // warmup
    hip_check(hipDeviceSynchronize(), "warmup sync");
    hip_check(hipGetLastError(), "warmup err");

    // time the GEMM only (dequant preamble is part of B0/B1 too; time both as the phase-2 unit)
    hipEvent_t s0, s1; hipEventCreate(&s0); hipEventCreate(&s1);
    const int iters = 50;
    hipEventRecord(s0);
    for (int i = 0; i < iters; i++) grouped_expert_gemm<N, K><<<num_tasks, threads>>>(A, Bg, Cg, d_tasks, num_tasks);
    hipEventRecord(s1); hipEventSynchronize(s1);
    float ms = 0; hipEventElapsedTime(&ms, s0, s1); ms /= iters;
    double gflop_real   = 2.0 * real_rows * N * K / 1e9;
    double gflop_padded = 2.0 * Mpacked   * N * K / 1e9;
    printf("  GEMM: %.4f ms/iter | %.1f TFLOP/s (real rows) | %.1f TFLOP/s (padded) "
           "[compare micro_tk ~69, B0 ceiling ~183]\n",
           ms, gflop_real / (ms * 1e-3) / 1e3, gflop_padded / (ms * 1e-3) / 1e3);

    // ---- correctness vs CPU fp32 reference (sample n every 37, like fmoe_expert_v2) ----
    bool pass = true;
    {
        std::vector<bf16> h_c(c_elems);
        hip_check(hipMemcpy(h_c.data(), d_c, c_elems * sizeof(bf16), hipMemcpyDeviceToHost), "cp c");
        double num_sq = 0.0, den_sq = 0.0, max_abs = 0.0;
        long padded_nonzero = 0, checked = 0;
        const int n_stride = full_check ? 37 : 257;     // coarser spot-check for the big perf case
        for (int m = 0; m < Mpacked; m++) {
            int e = row_expert[m];
            bool active = row_active[m];
            for (int n = 0; n < N; n += n_stride) {
                double acc = 0.0;
                const float* arow = &h_aref[(size_t)m * K];
                const float* brow = &h_bref[((size_t)e * N + n) * K];
                for (int k = 0; k < K; k++) acc += (double)arow[k] * (double)brow[k];
                float got = (float)h_c[(size_t)m * N + n];
                if (active) {
                    double err = std::fabs(acc - (double)got);
                    num_sq += err * err; den_sq += acc * acc; max_abs = std::max(max_abs, err); checked++;
                } else if (std::fabs(got) > 1e-3f) {
                    padded_nonzero++;                    // padding/contamination guard
                }
            }
        }
        double rms_rel = std::sqrt(num_sq / std::max(1e-30, den_sq));
        pass = (rms_rel < 0.05) && (padded_nonzero == 0);
        printf("  CORRECTNESS: RMS-rel=%.5f (tol 0.05), max_abs=%.4f, samples=%ld, "
               "padded/contaminated nonzero=%ld (expect 0) -> %s\n",
               rms_rel, max_abs, checked, padded_nonzero, pass ? "PASS" : "FAIL");
    }

    hipFree(d_fp8); hipFree(d_sc); hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_tasks);
    return pass;
}

int main(int argc, char** argv) {
    printf("grouped_b0 — B0-class 8-wave ping-pong, GROUPED over experts (N=%d, K=%d)\n", N, K);

    // Case 1 — correctness: ragged M_e (exercises BM-padding, masking, no cross-expert contamination).
    std::vector<int> me_ragged = {40, 256, 130, 300, 512, 7, 256, 99};
    bool ok1 = run_case("ragged-correctness", me_ragged, /*full_check=*/true);

    // Case 2 — perf: 8 experts x 1024 rows = 8192 packed rows, 256 tiles -> fills the GPU; the headline
    // TFLOP/s directly comparable to the EXPERIMENT_LEDGER grouped case (micro_tk ~69, B0 ceiling ~183).
    std::vector<int> me_perf(8, 1024);
    bool ok2 = run_case("perf-8x1024", me_perf, /*full_check=*/false);

    printf("\nRESULT: %s\n", (ok1 && ok2) ? "PASSED" : "FAILED");
    return (ok1 && ok2) ? 0 : 1;
}
