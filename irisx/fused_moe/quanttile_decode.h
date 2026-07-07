// ================================================================================================
// QuantTile v0 — ONE grouped decode-GEMM body, weight FORMAT chosen by a compile-time tile
// descriptor.  Unifies the two hand-written per-format decode kernels
//     grouped_expert_gemm_decode_fp8_sat    (sat_decode.cu     — fp8-e4m3, per-128-K-block f32 scale)
//     grouped_expert_gemm_decode_mxfp4_sat  (sat_decode_fp4.cu — mxfp4-e2m1, per-32-K-block E8M0 scale)
// into a SINGLE body `grouped_expert_gemm_decode_qt<qt::Fmt, N, K>`.  fp8 and mxfp4 are now two
// instantiations of the same source — NO per-format kernel fork.  This is the falsifiable v0
// milestone from irisx/tilecomm/RESEARCH_PLAN.md (Actions A1):
//     G1 (perf, load-bearing): unified@fp8   within ±5% of the standalone fp8   (3.97 TB/s)  -> zero-cost
//     G2 (perf):               unified@mxfp4 keeps the ~1.6x GEMM win over fp8
//     G3b (bit-stability):     unified output == the pre-unification kernel's output (RMS <= 1e-4)
//     G3a (correctness):       RMS vs format-faithful dequant unchanged (fp8 0.0037 / mxfp4 0.0033)
//
// The two bodies are byte-for-byte identical (skinny BM=16 x BN=256 tile, 8 warps, bf16 MFMA, same
// task-list indexing, same load/store) EXCEPT three spots, all carried by the descriptor `qt::Tile<F>`:
//   (1) packed-B register-tile WIDTH : K fp8 -> K/2 bf16 words (PACK_DIV=2) ; K fp4 -> K/4 (PACK_DIV=4)
//   (2) weight SCALE layout+type     : f32 per-128-K block (fp8) vs E8M0-uint8 per-32-K block (mxfp4)
//   (3) in-register UNPACK            : reinterpret+convert+fmul (fp8) vs hw cvt_scalef32_pk_bf16_fp4 (mxfp4)
// The scale POINTER TYPE differs, so the body is templated on `qt::Tile<F>::ScaleT` — this is exactly
// the "a tile carries its own format" idea: format+scale-layout are a compile-time type property.
//
// USAGE: #include AFTER an HK translation unit that provides `kittens`/`rt_bf`/`mma_ABt`/`bf16`/
// `bf16_2`/`fp8e4m3_4`/`base_types::convertor` and the TASK_W / T_EXPERT... task-tuple macros
// (i.e. after grouped_b0.cu, or after kernel.cpp's typedefs — there define QT_TASK_W etc. to B0_*).
// Self-contained for its own unpack helpers (qt::e8m0_dec, qt::fp4_pair).
// ================================================================================================
#pragma once

// Allow the includer to remap the task-tuple macro names (grouped_b0.cu uses TASK_W/T_*, kernel.cpp
// uses B0_TASK_W/B0_T_*). Default to the standalone (grouped_b0.cu) names.
#ifndef QT_TASK_W
#define QT_TASK_W    TASK_W
#define QT_T_EXPERT  T_EXPERT
#define QT_T_MTILE   T_MTILE
#define QT_T_NTILE   T_NTILE
#define QT_T_EROWBEG T_EROWBEG
#endif

namespace qt {

enum class Fmt { FP8_E4M3, MXFP4_E2M1 };

// --- self-contained unpack helpers -------------------------------------------------------------
// E8M0 (8-bit power-of-two exponent) -> f32 :  2^(e-127), with e==0 -> 0.
__device__ __forceinline__ float e8m0_dec(unsigned char e) { return __uint_as_float((unsigned)e << 23); }
// hardware fp4(e2m1) pair -> bf16 pair with the MX scale folded in (gfx950 v_cvt_scalef32_pk_bf16_fp4).
// SEL selects the byte of `reg`; each 32-bit reg holds 8 fp4 -> 4 SELs -> 4 bf16 pairs.
template <int SEL>
__device__ __forceinline__ bf16_2 fp4_pair(unsigned reg, float s) {
    auto v = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(reg, s, SEL);
    bf16_2 out; __builtin_memcpy(&out, &v, 4); return out;
}

// --- the tile descriptor: everything the body needs about B's representation, at compile time -----
template <Fmt F> struct Tile;
template <> struct Tile<Fmt::FP8_E4M3> {
    using ScaleT = float;                        // per-block f32 scale, applied in float
    static constexpr int PACK_DIV  = 2;          // K fp8 elems packed into K/2 bf16 words
    static constexpr int SCALE_BLK = 128;        // one scale per 128-K block
    __device__ static float dec(float s) { return s; }
};
template <> struct Tile<Fmt::MXFP4_E2M1> {
    using ScaleT = unsigned char;                // E8M0 exponent, decoded in-kernel
    static constexpr int PACK_DIV  = 4;          // K fp4 elems packed into K/4 bf16 words
    static constexpr int SCALE_BLK = 32;         // E8M0 scale per 32-K block
    __device__ static float dec(unsigned char e) { return e8m0_dec(e); }
};

} // namespace qt

// --- the unified body ---------------------------------------------------------------------------
template <qt::Fmt FMT, int NN, int KK>
__global__ __launch_bounds__(512, 2)
void grouped_expert_gemm_decode_qt(
        const kittens::gl<bf16, -1, -1, -1, -1> A,     // [Mpacked, K]   bf16 activations (format-independent)
        const kittens::gl<bf16, -1, -1, -1, -1> Bpk,   // [E*N, K/PACK_DIV]  PRE-SWIZZLED packed weights as bf16
        const kittens::gl<bf16, -1, -1, -1, -1> C,     // [Mpacked, N]   final scaled bf16 output
        const typename qt::Tile<FMT>::ScaleT* __restrict__ sB,  // fp8: [E*N,K/128] f32 ; mxfp4: [E*N,K/32] E8M0
        const int* __restrict__ tasks, int num_tasks) {
    using QT = qt::Tile<FMT>;
    constexpr int BM_DEC = 16, BN = 256, BLOCK_K = 128;
    constexpr int WARPS_COL = 8;
    constexpr int REG_BLOCK_N = BN / WARPS_COL;               // 32
    constexpr int k_iters = KK / BLOCK_K;                     // 56 for K=7168
    constexpr int NROWBLK = KK / QT::SCALE_BLK;               // scales per N-row (fp8: K/128, fp4: K/32)
    constexpr int SPB     = BLOCK_K / QT::SCALE_BLK;          // scales per 128-K iter (fp8:1, fp4:4)
    using RT_A   = rt_bf<BM_DEC,      BLOCK_K,               row_l, rt_16x32_s>;   // 16x128 bf16 A
    using RT_BPK = rt_bf<REG_BLOCK_N, BLOCK_K / QT::PACK_DIV, row_l, rt_16x32_s>;  // 32x64 (fp8) | 32x32 (fp4)
    using RT_B   = rt_bf<REG_BLOCK_N, BLOCK_K,               row_l, rt_16x32_s>;   // 32x128 bf16 (unpacked)
    using RT_C   = rt_fl<BM_DEC, REG_BLOCK_N, col_l, rt_16x16_s>;                  // 16x32 fp32 accum
    RT_A a; RT_BPK bpk; RT_B b; RT_C c;

    const int task = blockIdx.x;
    if (task >= num_tasks) return;
    const int* tk = tasks + (size_t)task * QT_TASK_W;
    const int e = tk[QT_T_EXPERT], mt = tk[QT_T_MTILE], nt = tk[QT_T_NTILE], ERB = tk[QT_T_EROWBEG];
    const int warp_n = warpid();

    const int a_row16 = ERB / BM_DEC + mt;
    const int b_row32 = (e * NN) / REG_BLOCK_N + nt * WARPS_COL + warp_n;
    const int c_row16 = ERB / BM_DEC + mt;
    const int c_col32 = nt * WARPS_COL + warp_n;

    const int    r16   = kittens::laneid() % 16;
    const int    n0    = e * NN + nt * BN + warp_n * REG_BLOCK_N;
    const size_t srow0 = (size_t)(n0 + r16)      * NROWBLK;   // this lane's 2 N-rows' scale bases
    const size_t srow1 = (size_t)(n0 + 16 + r16) * NROWBLK;

    zero(c);
    #pragma unroll 4
    for (int k = 0; k < k_iters; k++) {
        kittens::load<2>(a,   A,   {0, 0, a_row16, k});   // 16x128 bf16 A slab (broadcast)
        kittens::load<2>(bpk, Bpk, {0, 0, b_row32, k});   // this warp's 32x128 packed-weight strip

        if constexpr (FMT == qt::Fmt::FP8_E4M3) {         // ---- FP8: reinterpret + convert + per-128-block scale
            const float scale_i[2] = { QT::dec(sB[srow0 + SPB * k]), QT::dec(sB[srow1 + SPB * k]) };
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const float sc = scale_i[i];
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const fp8e4m3_4& q = reinterpret_cast<const fp8e4m3_4&>(bpk.tiles[i][j].data[d]);
                        float4 f = base_types::convertor<float4, fp8e4m3_4>::convert(q);
                        const int J  = 2 * j + (d >> 1);
                        const int Dp = 2 * (d & 1);
                        b.tiles[i][J].data[Dp]     = base_types::convertor<bf16_2, float2>::convert(make_float2(f.x * sc, f.y * sc));
                        b.tiles[i][J].data[Dp + 1] = base_types::convertor<bf16_2, float2>::convert(make_float2(f.z * sc, f.w * sc));
                    }
                }
            }
        } else {                                          // ---- MXFP4: hw cvt + per-32-block E8M0 scale
            float sc[2][4];
            #pragma unroll
            for (int d = 0; d < 4; d++) { sc[0][d] = QT::dec(sB[srow0 + SPB * k + d]); sc[1][d] = QT::dec(sB[srow1 + SPB * k + d]); }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    const float s = sc[i][d];
                    unsigned reg; __builtin_memcpy(&reg, &bpk.tiles[i][0].data[d], 4);
                    b.tiles[i][d].data[0] = qt::fp4_pair<0>(reg, s);
                    b.tiles[i][d].data[1] = qt::fp4_pair<1>(reg, s);
                    b.tiles[i][d].data[2] = qt::fp4_pair<2>(reg, s);
                    b.tiles[i][d].data[3] = qt::fp4_pair<3>(reg, s);
                }
            }
        }
        mma_ABt(c, a, b, c);
    }
    store(C, c, {0, 0, c_row16, c_col32});
}
