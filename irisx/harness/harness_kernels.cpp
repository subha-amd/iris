// irisx/harness/harness_kernels.cpp
// ================================================================================================
// Harness pybind kernels for B0 (local compute ceiling) and B1 (strong unfused IRISX baseline).
//
// These two ops are the ones the EXISTING candidate modules do NOT already expose as callable
// Python functions, so the unified runner (run_harness.py) needs them:
//
//   local_gemm(A_fp8, A_sc, B, C, M, N, K)
//       dequant fp8 e4m3 [M,K] + per-128 fp32 scales -> bf16, then bf16 GEMM C = A . B^T.
//       NO comm. == B0 compute ceiling.  Same numerics as v2_hk_expert_gemm/fmoe_expert_v2.cu.
//
//   dispatch_pack_quant_once(A_fp8_bf16, A_sc, A_local_fp8, A_local_sc, iris_ctx, M, K, src_rank)
//       IRIS-gather A's fp8 bytes + scales from `src_rank` EXACTLY ONCE into a LOCAL fp8 buffer
//       (+ copy scales).  This is the unfused "dispatch/pack/quant once" step that B1 pairs with a
//       single local_gemm.  The gather logic mirrors v3_fused_kernel/kernel.cpp::gather (vectorized
//       uint4 remote loads) but writes a PLAIN ROW-MAJOR local fp8 buffer (no swizzle, no dequant
//       here — dequant happens in local_gemm's preamble, identical to B0).
//
// B0 vs B1 are then apples-to-apples: same dequant, same GEMM, the ONLY difference is whether A
// arrived for free (B0, already local) or via a single IRIS gather (B1, the honest comm cost).
//
// Reuses HipKittens tile headers + IRIS exactly like v3/v4.  Build inside a HK distributed-kernels
// checkout as module name `harness_kernel`.  COMPILE-CHECKABLE on the node; subagents compile-only.
// ================================================================================================
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;     // unsigned char, 1 byte
static constexpr int QGROUP = 128;

// fp8 -> float, OCP e4m3 (matches V1/V2/V3 dequant exactly).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ================================================================================================
// Dequant preamble (== fmoe_expert_v2.cu::dequant_a_kernel, minus the send_counts masking — the
// harness packs dense M rows, no padding, so every row is active).  fp8[M,K] + sc[M,K/128] -> bf16.
// ================================================================================================
__global__ void dequant_a_dense(const fp8_t* __restrict__ a_fp8,
                                const float* __restrict__ a_sc,
                                bf16* __restrict__ a_bf16,
                                int M, int K) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int NG = K / QGROUP;
    const fp8_t* frow = a_fp8 + (size_t)row * K;
    const float* srow = a_sc  + (size_t)row * NG;
    bf16* orow = a_bf16 + (size_t)row * K;
    for (int h = threadIdx.x; h < K; h += blockDim.x) {
        __hip_fp8_e4m3 f; f.__x = frow[h];
        orow[h] = (bf16)(static_cast<float>(f) * srow[h / QGROUP]);
    }
}

// ================================================================================================
// B0 local bf16 GEMM core.  Direct reuse of the v2_hk_expert_gemm 8-wave ping-pong layout, but
// templated on runtime M,N,K via kittens::gl with -1 dims (the v2 file hardcoded constants; here
// the harness sweeps shapes, so we use dynamic gl + a tiled grid).  256x256x64 tiles, 8 warps.
// For brevity and to avoid duplicating the entire 200-line v2 ping-pong schedule, this uses the
// SAME tile/MMA primitives in a compact correct form; the main agent may swap in the exact v2
// 8_wave schedule for peak TFLOP/s (see BENCHMARK_METHODOLOGY.md note B0-schedule).
// ================================================================================================
constexpr int B0_WARPS = 8;
using B0G = kittens::group<B0_WARPS>;
constexpr int B0_BM = 256, B0_BN = 256, B0_BK = 64;

using B0_ST_A = st_bf<B0_BM / 2, B0_BK, st_16x32_s>;
using B0_ST_B = st_bf<B0_BN / 2, B0_BK, st_16x32_s>;

__global__ __launch_bounds__(B0_WARPS * 64, 2)
void b0_gemm(const gl<bf16, -1, -1, -1, -1> A,
             const gl<bf16, -1, -1, -1, -1> B,
             const gl<bf16, -1, -1, -1, -1> C,
             int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    B0_ST_A (&As)[2] = al.allocate<B0_ST_A, 2>();
    B0_ST_B (&Bs)[2] = al.allocate<B0_ST_B, 2>();

    constexpr int WARPS_COL = 4, WARPS_ROW = 2;
    constexpr int REG_M = B0_BM / WARPS_ROW / 2;
    constexpr int REG_N = B0_BN / WARPS_COL / 2;
    const int k_iters = K / B0_BK;
    const int blocks_per_col = (N + B0_BN - 1) / B0_BN;
    const int block_row = blockIdx.x / blocks_per_col;
    const int block_col = blockIdx.x % blocks_per_col;
    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;

    rt_bf<REG_M, B0_BK, row_l, rt_16x32_s> a;
    rt_bf<REG_N, B0_BK, row_l, rt_16x32_s> b0;
    rt_fl<REG_M, REG_N, col_l, rt_16x16_s> cacc;
    zero(cacc);

    uint32_t soA[64], soB[64];
    B0G::prefill_swizzled_offsets(As[0], A, soA);
    B0G::prefill_swizzled_offsets(Bs[0], B, soB);

    int tic = 0;
    for (int k = 0; k < k_iters; ++k, tic ^= 1) {
        B0G::load(As[tic], A, {0, 0, block_row * 2 + warp_m, k}, soA);
        B0G::load(Bs[tic], B, {0, 0, block_col * 2 + warp_n, k}, soB);
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
    store(C, cacc, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
}

// host wrapper: dequant preamble + GEMM.  A_fp8 = fp8 view [M,K], A_sc = fp32 [M,K/128].
struct b0_globals {
    gl<bf16, -1, -1, -1, -1> a_fp8;   // fp8 storage reinterpreted as bf16[M,K/2]
    gl<float, -1, -1, -1, -1> sc;     // [M,K/128]
    gl<bf16, -1, -1, -1, -1> b, c;    // B[N,K], C[M,N]
    int M, N, K;
    hipStream_t stream;
};

void local_gemm(b0_globals g) {
    // scratch bf16 A buffer (M*K bf16).  Allocated once per call; main agent may cache it.
    static bf16* d_a_bf16 = nullptr;
    static size_t cap = 0;
    size_t need = (size_t)g.M * g.K;
    if (need > cap) {
        if (d_a_bf16) hipFree(d_a_bf16);
        hipMalloc(&d_a_bf16, need * sizeof(bf16));
        cap = need;
    }
    const fp8_t* a_fp8 = reinterpret_cast<const fp8_t*>(&g.a_fp8[{0, 0, 0, 0}]);
    const float* a_sc = &g.sc[{0, 0, 0, 0}];
    dequant_a_dense<<<g.M, 256, 0, g.stream>>>(a_fp8, a_sc, d_a_bf16, g.M, g.K);

    gl<bf16, -1, -1, -1, -1> A(d_a_bf16, 1, 1, g.M, g.K);
    const int grid = ((g.M + B0_BM - 1) / B0_BM) * ((g.N + B0_BN - 1) / B0_BN);
    const size_t smem = 2 * (sizeof(B0_ST_A) + sizeof(B0_ST_B)) + 1024;
    hipFuncSetAttribute((void*)b0_gemm, hipFuncAttributeMaxDynamicSharedMemorySize, smem);
    b0_gemm<<<grid, B0_WARPS * 64, smem, g.stream>>>(A, g.b, g.c, g.M, g.N, g.K);
}

// ================================================================================================
// B1: dispatch_pack_quant_once — IRIS-gather fp8 bytes + scales from src_rank ONCE into a LOCAL
// row-major fp8 buffer.  One block per row-chunk; vectorized uint4 (16 fp8 bytes/thread) loads.
// ================================================================================================
struct gather_globals {
    gl<bf16, -1, -1, -1, -1> a_src;      // remote fp8 storage as bf16[M,K/2] on src_rank
    gl<float, -1, -1, -1, -1> sc_src;    // remote scales [M,K/128] on src_rank
    gl<bf16, -1, -1, -1, -1> a_dst;      // LOCAL fp8 storage as bf16[M,K/2]
    gl<float, -1, -1, -1, -1> sc_dst;    // LOCAL scales [M,K/128]
    iris::iris_device_view iris_ctx;
    int M, K, src_rank;
    hipStream_t stream;
};

__global__ void gather_once_kernel(gather_globals g) {
    const int K = g.K, M = g.M, NG = K / QGROUP;
    const fp8_t* src_base = reinterpret_cast<const fp8_t*>(&g.a_src[{0, 0, 0, 0}]);
    fp8_t* dst_base = reinterpret_cast<fp8_t*>(&g.a_dst[{0, 0, 0, 0}]);
    const float* sc_src = &g.sc_src[{0, 0, 0, 0}];
    float* sc_dst = &g.sc_dst[{0, 0, 0, 0}];
    iris::iris_device_view ctx = g.iris_ctx;

    const int row = blockIdx.x;
    if (row >= M) return;
    // gather K fp8 bytes per row, 16 bytes (uint4) per thread.
    const int chunks = K / 16;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        const int k = c * 16;
        const fp8_t* sp = src_base + (size_t)row * K + k;
        uint4 v = ctx.load(reinterpret_cast<const uint4*>(sp), g.src_rank);
        *reinterpret_cast<uint4*>(dst_base + (size_t)row * K + k) = v;
    }
    // gather the NG scales for this row.
    for (int gi = threadIdx.x; gi < NG; gi += blockDim.x) {
        sc_dst[(size_t)row * NG + gi] = ctx.load(sc_src + (size_t)row * NG + gi, g.src_rank);
    }
}

void dispatch_pack_quant_once(gather_globals g) {
    gather_once_kernel<<<g.M, 256, 0, g.stream>>>(g);
}

// ================================================================================================
PYBIND11_MODULE(harness_kernel, m) {
    m.doc() = "irisx harness kernels: B0 local_gemm + B1 dispatch_pack_quant_once";
    py::bind_function<local_gemm>(m, "local_gemm",
        &b0_globals::a_fp8, &b0_globals::sc, &b0_globals::b, &b0_globals::c,
        &b0_globals::M, &b0_globals::N, &b0_globals::K);
    py::bind_function<dispatch_pack_quant_once>(m, "dispatch_pack_quant_once",
        &gather_globals::a_src, &gather_globals::sc_src,
        &gather_globals::a_dst, &gather_globals::sc_dst,
        &gather_globals::iris_ctx,
        &gather_globals::M, &gather_globals::K, &gather_globals::src_rank);
}
