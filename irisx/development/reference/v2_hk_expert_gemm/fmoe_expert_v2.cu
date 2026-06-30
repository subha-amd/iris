// SPDX-License-Identifier: MIT
// V2 bring-up: single-GPU fused MoE expert gate/up GEMM on HipKittens (gfx950 / CDNA4).
//
// Consumes the IRISX MoE-dispatch FP8 buffer directly (expert-major packed layout):
//   packed_fp8 [local_e][src_rank][slot][H]   fp8 OCP e4m3 (max 448), H=7168
//   packed_sc  [local_e][src_rank][slot][N_GROUPS] fp32 per-128-K-group scale (=max(|g|)/448), N_GROUPS=56
//
// Design (per V2_HK_ANALYSIS.md):
//   - Option (b): PREAMBLE-dequant fp8->bf16 (apply per-128 fp32 group scale), then bf16 MMA core.
//   - bf16 MMA core is a direct bf16 port of the build-verified
//       kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu  (256x256x128 tiles, 8-wave ping-pong).
//   - Weight B synthesized as plain bf16, scale=1 (documented; real R1 weights not needed for bring-up).
//   - Variable-M: M_padded = world*PER_SRC_CAPACITY fixed; rows >= send_counts[e] are zeroed in the
//       dequant preamble (zero-padded rows stay zero) -> masked out, no garbage.
//   - Launch-per-expert (option i): host loop, pointer offset per expert. Here we run expert 0.
//   - OUT OF SCOPE: down-proj, IRIS remote gather, comm/compute overlap, in-MMA scaling.

#include "kittens.cuh"
#include <hip/hip_fp8.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <chrono>

using namespace kittens;

// ---- our buffer constants (match irisx test_moe_dispatch_pack_quant.hip) ----
static constexpr int H = 7168;                 // hidden / reduction dim K
static constexpr int GROUP = 128;              // per-K-group scale span
static constexpr int N_GROUPS = H / GROUP;     // 56
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr int WORLD = 8;
static constexpr int PER_SRC_CAPACITY = 64;
static constexpr int M_PADDED = WORLD * PER_SRC_CAPACITY; // 512
static constexpr int N = 2048;                 // gate/up output cols
static constexpr int K = H;                    // 7168

using fp8_t = __hip_fp8_storage_t;             // raw byte, matches irisx packed_fp8

// ============================================================================
// Dequant preamble: packed_fp8[M_PADDED,H] + packed_sc[M_PADDED,N_GROUPS] -> A_bf16[M_PADDED,H]
// Applies the per-128 group fp32 scale. Rows with (slot >= send_count[src]) are zeroed (masking).
// One block per row; threads stride over H.
// ============================================================================
__global__ void dequant_a_kernel(const fp8_t* __restrict__ packed_fp8,
                                 const float* __restrict__ packed_sc,
                                 const int*   __restrict__ send_counts, // [WORLD], per src_rank token count for this expert
                                 bf16* __restrict__ A_bf16) {
    const int row = blockIdx.x;                 // 0..M_PADDED-1
    if (row >= M_PADDED) return;
    const int src_rank = row / PER_SRC_CAPACITY;
    const int slot     = row % PER_SRC_CAPACITY;
    const bool active  = slot < send_counts[src_rank];

    const fp8_t* frow = packed_fp8 + (size_t)row * H;
    const float* srow = packed_sc  + (size_t)row * N_GROUPS;
    bf16* orow = A_bf16 + (size_t)row * H;

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float v = 0.0f;
        if (active) {
            __hip_fp8_e4m3 f; f.__x = frow[h];
            v = static_cast<float>(f) * srow[h / GROUP];   // dequant
        }
        orow[h] = (bf16)v;
    }
}

// ============================================================================
// bf16 GEMM core: C[M,N] = A[M,K] . B[N,K]^T  (mma_ABt contract). bf16 inputs, bf16 out.
// Direct bf16 port of FP8_8wave/8_wave.cu (256x256x128, 8-wave ping-pong).
// ============================================================================
constexpr int NUM_WARPS = 8;
using G = kittens::group<NUM_WARPS>;

template <int MM, int NN, int KK>
__global__ __launch_bounds__(512, 2)
void expert_gemm_bf16(const kittens::gl<bf16, 1, 1, MM, KK> A,
                      const kittens::gl<bf16, 1, 1, NN, KK> B,
                      const kittens::gl<bf16, 1, 1, MM, NN> C) {
    constexpr int WARPS_COL = 4;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256;
    constexpr int BLOCK_SIZE_COL = 256;
    constexpr int BLOCK_K = 64;                  // bf16 LDS is 2x fp8; use 64 to fit 160KB LDS.
                                                 // (scale already applied in the dequant preamble, so
                                                 //  BLOCK_K need not equal the 128 scale group here.)
    constexpr int blocks_per_col = NN / BLOCK_SIZE_COL;
    constexpr int k_iters = KK / BLOCK_K;        // 56
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

    int global_block_id = blockIdx.x;
    int block_row = global_block_id / blocks_per_col;
    int block_col = global_block_id % blocks_per_col;

    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);

    int tic = 0, toc = 1;

    constexpr int memcpy_per_tile_A = 1; // unused size guard placeholder
    uint32_t swizzled_offsets_A[64];
    uint32_t swizzled_offsets_B[64];
    G::prefill_swizzled_offsets(As[tic][0], A, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[tic][0], B, swizzled_offsets_B);

    zero(cA); zero(cB); zero(cC); zero(cD);

    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, swizzled_offsets_B);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, swizzled_offsets_A);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, swizzled_offsets_B);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, swizzled_offsets_A);

    if (warp_m == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, swizzled_offsets_B);
    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, swizzled_offsets_A);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, swizzled_offsets_B);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic^=1, toc^=1) {
        auto bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, swizzled_offsets_A);
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
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, swizzled_offsets_B);
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
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, swizzled_offsets_A);
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

        tic^=1, toc^=1;
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

    store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m,             block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m,             block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}

// ----------------------------------------------------------------------------
static inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) { fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    // send_counts per src_rank for THIS expert: default a ragged set < PER_SRC_CAPACITY to exercise masking.
    // src 0..7 -> {40,17,64,0,55,33,9,50}; total real rows < M_PADDED so masking matters.
    std::vector<int> send_counts = {40, 17, 64, 0, 55, 33, 9, 50};

    printf("V2 fmoe expert gate/up GEMM (bring-up)\n");
    printf("  M_PADDED=%d (world=%d x PER_SRC_CAPACITY=%d), N=%d, K=H=%d, N_GROUPS=%d\n",
           M_PADDED, WORLD, PER_SRC_CAPACITY, N, K, N_GROUPS);

    const size_t a_elems  = (size_t)M_PADDED * H;
    const size_t sc_elems = (size_t)M_PADDED * N_GROUPS;
    const size_t b_elems  = (size_t)N * K;
    const size_t c_elems  = (size_t)M_PADDED * N;

    // ---- Host-side synth of the packed_fp8 / packed_sc buffer (mimics dispatch quant) ----
    std::vector<fp8_t> h_packed_fp8(a_elems, 0);
    std::vector<float> h_packed_sc(sc_elems, 0.0f);
    std::vector<float> h_a_ref(a_elems, 0.0f);   // true dequantized A (for CPU reference)
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int row = 0; row < M_PADDED; row++) {
        int src = row / PER_SRC_CAPACITY, slot = row % PER_SRC_CAPACITY;
        bool active = slot < send_counts[src];
        if (!active) continue; // padded -> stays zero
        for (int g = 0; g < N_GROUPS; g++) {
            float vals[GROUP]; float amax = 0.0f;
            for (int j = 0; j < GROUP; j++) { float v = dist(gen); vals[j] = v; amax = std::max(amax, std::fabs(v)); }
            float scale = (amax > 0.0f) ? (amax / FP8_E4M3_MAX) : 1.0f;
            float inv = 1.0f / scale;
            h_packed_sc[(size_t)row * N_GROUPS + g] = scale;
            for (int j = 0; j < GROUP; j++) {
                int h = g * GROUP + j;
                fp8_t q = __hip_cvt_float_to_fp8(vals[j] * inv, __HIP_SATFINITE, __HIP_E4M3);
                h_packed_fp8[(size_t)row * H + h] = q;
                __hip_fp8_e4m3 f; f.__x = q;
                h_a_ref[(size_t)row * H + h] = static_cast<float>(f) * scale; // matches device dequant exactly
            }
        }
    }

    // ---- Synth weight B as plain bf16, scale=1 (documented bring-up choice) ----
    std::vector<bf16> h_b(b_elems);
    std::vector<float> h_b_ref(b_elems);
    for (size_t i = 0; i < b_elems; i++) { float v = dist(gen) * 0.1f; h_b[i] = (bf16)v; h_b_ref[i] = (float)(bf16)v; }

    // ---- Device buffers ----
    fp8_t* d_packed_fp8; float* d_packed_sc; int* d_send_counts;
    bf16* d_a; bf16* d_b; bf16* d_c;
    hip_check(hipMalloc(&d_packed_fp8, a_elems * sizeof(fp8_t)), "malloc fp8");
    hip_check(hipMalloc(&d_packed_sc, sc_elems * sizeof(float)), "malloc sc");
    hip_check(hipMalloc(&d_send_counts, WORLD * sizeof(int)), "malloc sc cnt");
    hip_check(hipMalloc(&d_a, a_elems * sizeof(bf16)), "malloc a");
    hip_check(hipMalloc(&d_b, b_elems * sizeof(bf16)), "malloc b");
    hip_check(hipMalloc(&d_c, c_elems * sizeof(bf16)), "malloc c");

    hip_check(hipMemcpy(d_packed_fp8, h_packed_fp8.data(), a_elems * sizeof(fp8_t), hipMemcpyHostToDevice), "cp fp8");
    hip_check(hipMemcpy(d_packed_sc, h_packed_sc.data(), sc_elems * sizeof(float), hipMemcpyHostToDevice), "cp sc");
    hip_check(hipMemcpy(d_send_counts, send_counts.data(), WORLD * sizeof(int), hipMemcpyHostToDevice), "cp cnt");
    hip_check(hipMemcpy(d_b, h_b.data(), b_elems * sizeof(bf16), hipMemcpyHostToDevice), "cp b");
    hip_check(hipMemset(d_c, 0, c_elems * sizeof(bf16)), "memset c");

    // ---- Dequant preamble ----
    dequant_a_kernel<<<M_PADDED, 256>>>(d_packed_fp8, d_packed_sc, d_send_counts, d_a);
    hip_check(hipGetLastError(), "dequant launch");
    hip_check(hipDeviceSynchronize(), "dequant sync");

    // ---- GEMM ----
    kittens::gl<bf16, 1, 1, M_PADDED, K> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<bf16, 1, 1, N, K>        Bg(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<bf16, 1, 1, M_PADDED, N> Cg(d_c, nullptr, nullptr, nullptr, nullptr);

    constexpr int grid = (M_PADDED * N) / (256 * 256);
    constexpr int threads = NUM_WARPS * 64;

    // warmup
    for (int i = 0; i < 5; i++) {
        expert_gemm_bf16<M_PADDED, N, K><<<grid, threads>>>(A, Bg, Cg);
    }
    hip_check(hipDeviceSynchronize(), "warmup sync");
    hip_check(hipGetLastError(), "warmup err");

    // timing
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    const int iters = 50;
    hipEventRecord(s);
    for (int i = 0; i < iters; i++) expert_gemm_bf16<M_PADDED, N, K><<<grid, threads>>>(A, Bg, Cg);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e); ms /= iters;
    double tflops = (2.0 * M_PADDED * N * K) / (ms * 1e-3) / 1e12;
    printf("  GEMM: %.4f ms/iter, %.2f TFLOP/s (over padded M=%d)\n", ms, tflops, M_PADDED);

    // ---- Copy back & CPU reference (dequant ref already in h_a_ref) ----
    std::vector<bf16> h_c(c_elems);
    hip_check(hipMemcpy(h_c.data(), d_c, c_elems * sizeof(bf16), hipMemcpyDeviceToHost), "cp c back");

    // CPU ref: C[m,n] = sum_k a_ref[m,k]*b_ref[n,k]  (fp32 accum, same dequant as device).
    // Correctness metric for a bf16-input GEMM with random sign-crossing outputs:
    //   - global RMS-relative error ||C - ref|| / ||ref||  (the standard GEMM measure), and
    //   - max element error normalized by the output RMS scale (NOT per-element, which blows up
    //     near sign-crossing zeros and is meaningless for bf16). bf16 has 8 mantissa bits, so a
    //     7168-deep accum gives ~1-2% RMS error -> tolerance 0.05.
    long active_rows_checked = 0, padded_rows_nonzero = 0, checked = 0;
    double num_sq = 0.0, den_sq = 0.0;             // for ||C-ref||/||ref||
    double max_norm_err = 0.0, max_abs = 0.0;
    // first pass: output RMS scale over active samples
    double sumsq_ref = 0.0; long cnt_ref = 0;
    for (int m = 0; m < M_PADDED; m++) {
        int src = m / PER_SRC_CAPACITY, slot = m % PER_SRC_CAPACITY;
        if (!(slot < send_counts[src])) continue;
        for (int n = 0; n < N; n += 37) {
            double acc = 0.0;
            const float* arow = &h_a_ref[(size_t)m * H];
            const float* brow = &h_b_ref[(size_t)n * K];
            for (int k = 0; k < K; k++) acc += (double)arow[k] * (double)brow[k];
            sumsq_ref += acc * acc; cnt_ref++;
        }
    }
    double out_rms = std::sqrt(sumsq_ref / std::max(1L, cnt_ref));

    for (int m = 0; m < M_PADDED; m++) {
        int src = m / PER_SRC_CAPACITY, slot = m % PER_SRC_CAPACITY;
        bool active = slot < send_counts[src];
        for (int n = 0; n < N; n += 37) {
            double acc = 0.0;
            const float* arow = &h_a_ref[(size_t)m * H];
            const float* brow = &h_b_ref[(size_t)n * K];
            for (int k = 0; k < K; k++) acc += (double)arow[k] * (double)brow[k];
            float ref = (float)acc;
            float got = (float)h_c[(size_t)m * N + n];
            double err = std::fabs((double)ref - (double)got);
            if (active) {
                num_sq += err * err; den_sq += (double)ref * (double)ref;
                max_abs = std::max(max_abs, err);
                max_norm_err = std::max(max_norm_err, err / out_rms);  // normalized by output scale
                checked++;
            } else {
                if (std::fabs(got) > 1e-2f * out_rms) padded_rows_nonzero++;
            }
        }
        if (active) active_rows_checked++;
    }
    double rms_rel = std::sqrt(num_sq / std::max(1e-30, den_sq));

    printf("\n=== CORRECTNESS (vs CPU fp32 dequant reference) ===\n");
    printf("  active rows checked: %ld, samples: %ld, output RMS scale: %.4f\n",
           active_rows_checked, checked, out_rms);
    printf("  RMS-relative error ||C-ref||/||ref|| = %.5f\n", rms_rel);
    printf("  max_abs_err = %.5f, max_err/out_rms = %.5f\n", max_abs, max_norm_err);
    printf("  MASKING: padded(masked) rows with nonzero output = %ld (expect 0)\n", padded_rows_nonzero);

    bool pass = (padded_rows_nonzero == 0) && (rms_rel < 0.05) && (max_norm_err < 0.1);
    printf("\n  RESULT: %s\n", pass ? "PASSED" : "FAILED");

    hipFree(d_packed_fp8); hipFree(d_packed_sc); hipFree(d_send_counts);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return pass ? 0 : 1;
}
