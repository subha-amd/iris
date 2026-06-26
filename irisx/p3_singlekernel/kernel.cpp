// irisx/p3_singlekernel/kernel.cpp
// ================================================================================================
// P3 — SINGLE-KERNEL copy-once + in-block overlap MoE expert-GEMM.
//
// One launch, ONE kernel, NO cross-stream, NO inter-block flag handshake.  This is the main line
// after P1/P2 (two-kernel producer/consumer-flag) FAILED (RMS=inf, 15-100x slow — see ledger
// "P1/P2 ... BOTH FAIL").  The P1/P2 root cause was the two-kernel cross-stream flag handshake:
// the producer kernel made few resident blocks while consumer-kernel blocks spin-waited (no real
// overlap, huge spin waste) and the flag/visibility map mis-published tiles (RMS=inf).  P3 has
// NEITHER hazard — there is exactly ONE kernel, one grid, one block per M-tile; ALL overlap is
// IN-BLOCK between warps of the same block sharing LDS, the SAME mechanism that already WORKS in
// V4/B5 (ledger: B3->B5's 1.82x is "ALL overlap").  No cross-block / cross-stream visibility is
// ever relied on, so the inf-RMS publish hazard cannot occur.
//
// What P3 fixes vs V4/B5 (which ran at only 44 TFLOP/s):
//   (1) COPY-ONCE.  Grid is 1-D over M-tiles ONLY (M/BM blocks).  Each block owns one BM=256
//       row-strip and ALL N columns, so no two blocks ever gather the same A.  Within a block, A is
//       gathered from REMOTE exactly once per K-tile while computing the FIRST N-panel, and cached
//       (already dequantized to bf16) into a block-private LOCAL-HBM A-strip; the remaining N-panels
//       read A from LOCAL HBM (~7 TB/s) and never re-cross XGMI.  => each remote A element crosses
//       the interconnect EXACTLY ONCE (B1-class A traffic), unlike V4's N/N_PER_BLOCK re-gathers.
//   (2) NEAR-B0 COMPUTE.  NO permanent producer/consumer warp split (V4 wasted 4/8 warps gathering).
//       ALL 8 warps MFMA, exactly B0's 256x256x64 8-warp ping-pong (harness b0_gemm = 183 TFLOP/s).
//       The remote gather is issued by all 8 warps too, ONE K-tile AHEAD (double-buffer, tic^=1), so
//       the cross-GPU latency of A[t+1] is hidden under the MFMA of A[t].  The MFMA math/occupancy
//       is byte-for-byte B0; only the *source* of the A LDS tile differs (remote on panel 0, local
//       HBM after).  This is the prompt's KEY SIMPLIFICATION: "B0's GEMM, but the A it reads is
//       gathered from remote into LDS one K-tile ahead instead of read from local HBM."
//
// ABI: fp8 e4m3 (OCP gfx950) A + per-128 fp32 scales (remote on src_rank) -> bf16 dequant in-kernel
//      -> bf16 GEMM C = A . B^T.  Identical numerics to B0/B1/V4 (RMS-rel ~0.0033 vs bf16 ref).
//      Zero-sentinel: A real only on src_rank, zeros on the consumer rank => nonzero C proves the
//      remote gather actually happened.
//
// Module name `tk_kernel` (build auto-discovers */kernel.cpp).  COMPILE-only for subagents; the
// MAIN AGENT runs all on-device tests under the GPU lock.
// ================================================================================================

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;     // unsigned char, 1 byte
static constexpr int QGROUP = 128;

// ----------------------------- tile / block configuration ---------------------------------------
// Mirrors B0's peak 8-warp schedule (harness_kernels.cpp::b0_gemm): 256x256x64 tiles, 8 warps,
// WARPS_COL=4 x WARPS_ROW=2.  We do NOT change the MFMA layout — only the SOURCE of the A LDS tile.
constexpr int P3_WARPS    = 8;
using P3G = kittens::group<P3_WARPS>;
constexpr int P3_BM = 256, P3_BN = 256, P3_BK = 64;
constexpr int WARPS_COL = 4, WARPS_ROW = 2;
constexpr int REG_M = P3_BM / WARPS_ROW / 2;   // 64
constexpr int REG_N = P3_BN / WARPS_COL / 2;   // 32

// A and B shared tiles are 128-row half-tiles (== B0_ST_A / B0_ST_B).
using P3_ST_A = st_bf<P3_BM / 2, P3_BK, st_16x32_s>;
using P3_ST_B = st_bf<P3_BN / 2, P3_BK, st_16x32_s>;

// fp8 -> float using OCP e4m3 (matches B0/V4 dequant exactly).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

struct p3_globals {
    gl<bf16,  -1, -1, -1, -1> a;     // [M, K/2]  remote fp8 e4m3 [M,K] on src_rank, viewed bf16
    gl<float, -1, -1, -1, -1> sc;    // [M, K/128] fp32 per-group scales on src_rank
    gl<bf16,  -1, -1, -1, -1> b, c;  // B[N,K] local, C[M,N] local
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int fused;                        // 1 = P3 overlap path; 0 = serial copy-once-then-GEMM baseline
    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(M, (int)P3_BM)); }   // M-tiles only => copy-once
    dim3 block() { return dim3(P3_WARPS * kittens::WARP_THREADS); }
    // shared: double-buffered A tile (128-row half) + double-buffered B tile (128-row half).
    size_t dynamic_shared_memory() {
        return (size_t)2 * (sizeof(P3_ST_A) + sizeof(P3_ST_B)) + 1024;
    }
};

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + per-128 dequant of ONE 128-row half of the A K-tile into swizzled shared
// tile `dst`.  Issued by ALL 8 warps cooperatively (no permanent producer warps).  `half` (=warp_m,
// 0..1) selects which 128-row half of the 256-row strip — matching B0's {block_row*2+warp_m, k}
// A indexing.  Each fp8 vector (uint4 = 16 bytes) crosses XGMI exactly ONCE per (m_tile,k_tile).
// ------------------------------------------------------------------------------------------------
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_half(
        P3_ST_A &dst, int k_tile, int strip_row0, int half, const p3_globals &g) {
    const int k0   = k_tile * P3_BK;
    const int row0 = strip_row0 + half * (P3_BM / 2);
    const int tid  = threadIdx.x;                       // 0..511 (all 8 warps)
    const int K    = g.K;
    const int NG   = K / QGROUP;
    constexpr int SUBR = P3_ST_A::underlying_subtile_rows;
    constexpr int SUBC = P3_ST_A::underlying_subtile_cols;
    constexpr int SUBN = P3_ST_A::underlying_subtile_elements;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;

    constexpr int ROWS = P3_BM / 2;                     // 128
    constexpr int CHUNKS_PER_ROW = P3_BK / VEC;         // 4
    const int total_chunks = ROWS * CHUNKS_PER_ROW;     // 512

    for (int ci = tid; ci < total_chunks; ci += P3_WARPS * kittens::WARP_THREADS) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gr = row0 + r;
        const int gk = k0 + kc;

        uint4 packed;
        if (gr < g.M && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), g.src_rank);   // ONE XGMI read
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        const int grp = gk / QGROUP;
        float scale = 1.0f;
        if (gr < g.M && grp < NG) {
            scale = ctx.load(&g.sc[{0, 0, gr, grp}], g.src_rank);
        }

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            bf16 val = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * P3_ST_A::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = P3_ST_A::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Fill the LDS A half-tile `dst` for (k_tile, half) from the LOCAL bf16 A-cache (no XGMI).  Used by
// N-panels > 0 (copy-once: A already gathered+cached during panel 0).  Cache layout is plain
// row-major [P3_BM, K] per block; we slice [half*(BM/2)..][k0..k0+BK].
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void load_A_half_from_cache(
        P3_ST_A &dst, int k_tile, int strip_row0, int half,
        const bf16* __restrict__ cache_strip, int K) {
    const int k0   = k_tile * P3_BK;
    const int row0 = half * (P3_BM / 2);                // local row offset inside the cache strip
    const int tid  = threadIdx.x;
    constexpr int SUBR = P3_ST_A::underlying_subtile_rows;
    constexpr int SUBC = P3_ST_A::underlying_subtile_cols;
    constexpr int SUBN = P3_ST_A::underlying_subtile_elements;
    constexpr int ROWS = P3_BM / 2;
    for (int idx = tid; idx < ROWS * P3_BK; idx += P3_WARPS * kittens::WARP_THREADS) {
        const int r = idx / P3_BK;
        const int k = idx % P3_BK;
        const bf16 val = cache_strip[(size_t)(row0 + r) * K + (k0 + k)];
        const int sub_row = r / SUBR, sub_col = k / SUBC;
        const int sub_id  = sub_row * P3_ST_A::underlying_subtiles_per_row + sub_col;
        const int rr = r % SUBR, cc = k % SUBC;
        const uint32_t intra_byte = P3_ST_A::swizzle({rr, cc});
        char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
        *reinterpret_cast<bf16*>(base + intra_byte) = val;
    }
}

// Persist the LDS A half-tile `src` to the block-private LOCAL bf16 A-cache (panel 0 only).
__device__ __forceinline__ void store_A_half_to_cache(
        const P3_ST_A &src, int k_tile, int half,
        bf16* __restrict__ cache_strip, int K) {
    const int k0   = k_tile * P3_BK;
    const int row0 = half * (P3_BM / 2);
    const int tid  = threadIdx.x;
    constexpr int SUBR = P3_ST_A::underlying_subtile_rows;
    constexpr int SUBC = P3_ST_A::underlying_subtile_cols;
    constexpr int SUBN = P3_ST_A::underlying_subtile_elements;
    constexpr int ROWS = P3_BM / 2;
    for (int idx = tid; idx < ROWS * P3_BK; idx += P3_WARPS * kittens::WARP_THREADS) {
        const int r = idx / P3_BK;
        const int k = idx % P3_BK;
        const int sub_row = r / SUBR, sub_col = k / SUBC;
        const int sub_id  = sub_row * P3_ST_A::underlying_subtiles_per_row + sub_col;
        const int rr = r % SUBR, cc = k % SUBC;
        const uint32_t intra_byte = P3_ST_A::swizzle({rr, cc});
        const char* base = reinterpret_cast<const char*>(&src.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
        cache_strip[(size_t)(row0 + r) * K + (k0 + k)] = *reinterpret_cast<const bf16*>(base + intra_byte);
    }
}

// ================================================================================================
// P3 FUSED kernel.  ONE block per M-tile.  Outer loop over N-panels; inner K-walk is B0's 8-warp
// ping-pong with A double-buffered ONE K-tile ahead.  On panel 0 the A prefetch is a REMOTE gather
// (overlaps MFMA) AND is cached to local HBM; on panels >0 the A prefetch reads local HBM.  So A
// crosses XGMI exactly once and the GEMM is B0-class on every panel.
// ================================================================================================
__global__ __launch_bounds__(P3_WARPS * 64, 2)
void p3_gemm(p3_globals g, bf16* __restrict__ a_cache) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    P3_ST_A (&As)[2] = al.allocate<P3_ST_A, 2>();
    P3_ST_B (&Bs)[2] = al.allocate<P3_ST_B, 2>();

    const int k_iters    = g.K / P3_BK;
    const int n_panels   = (g.N + P3_BN - 1) / P3_BN;
    const int strip_row0 = blockIdx.x * P3_BM;
    const int warp_m     = warpid() / WARPS_COL;        // 0..1 (which 128-row half)
    const int warp_n     = warpid() % WARPS_COL;        // 0..3
    const int K          = g.K;
    bf16* cache_strip    = a_cache + (size_t)blockIdx.x * P3_BM * (size_t)K;

    uint32_t soB[64];
    P3G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    for (int np = 0; np < n_panels; ++np) {
        const bool panel0 = (np == 0);
        rt_fl<REG_M, REG_N, col_l, rt_16x16_s> cacc;
        zero(cacc);

        // ---- prologue: prefetch K-tile 0 (A half + B half) into stage 0 ----
        if (panel0) {
            gather_dequant_A_half<16>(As[0], 0, strip_row0, warp_m, g);
        } else {
            load_A_half_from_cache(As[0], 0, strip_row0, warp_m, cache_strip, K);
        }
        P3G::load(Bs[0], g.b, {0, 0, np * 2 + warp_n, 0}, soB);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        if (panel0) store_A_half_to_cache(As[0], 0, warp_m, cache_strip, K);

        int tic = 0;
        for (int k = 0; k < k_iters; ++k, tic ^= 1) {
            const int nxt = tic ^ 1;
            const int kn  = k + 1;

            // ---- PREFETCH A[k+1] + B[k+1] into the other stage (overlaps this iter's MFMA) ----
            if (kn < k_iters) {
                if (panel0) {
                    gather_dequant_A_half<16>(As[nxt], kn, strip_row0, warp_m, g);  // REMOTE, hidden
                } else {
                    load_A_half_from_cache(As[nxt], kn, strip_row0, warp_m, cache_strip, K);
                }
                P3G::load(Bs[nxt], g.b, {0, 0, np * 2 + warp_n, kn}, soB);
            }

            // ---- MFMA the CURRENT stage (B0-identical) ----
            rt_bf<REG_M, P3_BK, row_l, rt_16x32_s> a;
            rt_bf<REG_N, P3_BK, row_l, rt_16x32_s> b0;
            auto as = kittens::subtile_inplace<REG_M, P3_BK>(As[tic], {0, 0});
            auto bs = kittens::subtile_inplace<REG_N, P3_BK>(Bs[tic], {0, 0});
            load(a, as);
            load(b0, bs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cacc, a, b0, cacc);
            __builtin_amdgcn_s_setprio(0);

            // wait for the prefetch (remote gather / cache read + B load) before the swap.
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();

            // cache the just-fetched A[k+1] on panel 0 so later panels can reuse it.
            if (panel0 && kn < k_iters) store_A_half_to_cache(As[nxt], kn, warp_m, cache_strip, K);
        }

        store(g.c, cacc,
              {0, 0, (int)blockIdx.x * WARPS_ROW * 2 + warp_m, np * WARPS_COL * 2 + warp_n});
    }
}

// ================================================================================================
// In-module SERIAL baseline (fused=0) — convenience reproduction of B1-copy: gather+dequant ALL of
// A to local HBM (NO MFMA overlap), then run the pure-LOCAL B0 GEMM over the cache.  example.py
// reports this so P3-fused has an in-module B1 reference; the AUTHORITATIVE B1 is still the harness
// (dispatch_pack_quant_once + local_gemm).  This baseline does NOT overlap (copy then compute).
// ================================================================================================
__global__ void p3_gather_all(p3_globals g, bf16* __restrict__ a_cache) {
    const int K = g.K, NG = K / QGROUP;
    const int strip_row0 = blockIdx.x * P3_BM;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;
    bf16* cache_strip = a_cache + (size_t)blockIdx.x * P3_BM * (size_t)K;
    const int chunks_per_row = K / 16;
    for (int idx = threadIdx.x; idx < P3_BM * chunks_per_row; idx += blockDim.x) {
        const int r  = idx / chunks_per_row;
        const int kc = (idx % chunks_per_row) * 16;
        const int gr = strip_row0 + r;
        if (gr >= g.M) continue;
        uint4 packed = ctx.load(reinterpret_cast<const uint4*>(a_base + (size_t)gr * K + kc), g.src_rank);
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);
        const int grp = kc / QGROUP;
        const float scale = ctx.load(&g.sc[{0, 0, gr, grp}], g.src_rank);
        bf16* out = cache_strip + (size_t)r * K + kc;
        #pragma unroll
        for (int j = 0; j < 16; ++j) out[j] = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
    }
}

// Pure-local B0 GEMM that reads the bf16 A-cache (no remote).  Same MFMA core as p3_gemm but A is
// always sourced from the local cache (panel-agnostic).  Used only by the serial baseline.
__global__ __launch_bounds__(P3_WARPS * 64, 2)
void p3_gemm_local(p3_globals g, const bf16* __restrict__ a_cache) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    P3_ST_A (&As)[2] = al.allocate<P3_ST_A, 2>();
    P3_ST_B (&Bs)[2] = al.allocate<P3_ST_B, 2>();

    const int k_iters    = g.K / P3_BK;
    const int n_panels   = (g.N + P3_BN - 1) / P3_BN;
    const int strip_row0 = blockIdx.x * P3_BM;
    const int warp_m     = warpid() / WARPS_COL;
    const int warp_n     = warpid() % WARPS_COL;
    const int K          = g.K;
    const bf16* cache_strip = a_cache + (size_t)blockIdx.x * P3_BM * (size_t)K;

    uint32_t soB[64];
    P3G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    for (int np = 0; np < n_panels; ++np) {
        rt_fl<REG_M, REG_N, col_l, rt_16x16_s> cacc;
        zero(cacc);
        int tic = 0;
        for (int k = 0; k < k_iters; ++k, tic ^= 1) {
            load_A_half_from_cache(As[tic], k, strip_row0, warp_m, cache_strip, K);
            P3G::load(Bs[tic], g.b, {0, 0, np * 2 + warp_n, k}, soB);
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)");
            rt_bf<REG_M, P3_BK, row_l, rt_16x32_s> a;
            rt_bf<REG_N, P3_BK, row_l, rt_16x32_s> b0;
            auto as = kittens::subtile_inplace<REG_M, P3_BK>(As[tic], {0, 0});
            auto bs = kittens::subtile_inplace<REG_N, P3_BK>(Bs[tic], {0, 0});
            load(a, as);
            load(b0, bs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(cacc, a, b0, cacc);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
        }
        store(g.c, cacc,
              {0, 0, (int)blockIdx.x * WARPS_ROW * 2 + warp_m, np * WARPS_COL * 2 + warp_n});
    }
}

void dispatch_p3(p3_globals g) {
    static bf16* d_a_cache = nullptr;
    static size_t cap = 0;
    const int grid_blocks = ceil_div(g.M, (int)P3_BM);
    size_t need = (size_t)grid_blocks * P3_BM * (size_t)g.K;
    if (need > cap) {
        if (d_a_cache) hipFree(d_a_cache);
        hipMalloc(&d_a_cache, need * sizeof(bf16));
        cap = need;
    }
    const size_t smem = g.dynamic_shared_memory();
    if (g.fused) {
        hipFuncSetAttribute((void*)p3_gemm, hipFuncAttributeMaxDynamicSharedMemorySize, smem);
        p3_gemm<<<g.grid(), g.block(), smem, g.stream>>>(g, d_a_cache);
    } else {
        // SERIAL B1-style: gather-all to local HBM, then pure-local B0 GEMM (no overlap).
        p3_gather_all<<<grid_blocks, 256, 0, g.stream>>>(g, d_a_cache);
        hipFuncSetAttribute((void*)p3_gemm_local, hipFuncAttributeMaxDynamicSharedMemorySize, smem);
        p3_gemm_local<<<g.grid(), g.block(), smem, g.stream>>>(g, d_a_cache);
    }
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "p3_singlekernel tk_kernel: single-kernel copy-once + in-block overlap MoE GEMM";
    py::bind_function<dispatch_p3>(m, "dispatch_micro",
        &p3_globals::a, &p3_globals::sc, &p3_globals::b, &p3_globals::c,
        &p3_globals::iris_ctx,
        &p3_globals::M, &p3_globals::N, &p3_globals::K, &p3_globals::src_rank,
        &p3_globals::fused);
}
