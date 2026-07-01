// ================================================================================================
// sat_decode_fp4.cu — SATURATING MXFP4 (OCP W4, E8M0 per-32-block) BM=16 decode GEMM.
//
// Route 1: store B as fp4 (e2m1, 1/4 the bf16 bytes) PRE-SWIZZLED so it loads through the FAST bf16
// global->register path (a rt_bf<32,32> raw byte copy = 32x128 packed fp4 per warp), UNPACK fp4->bf16
// in-register applying the per-32-K-block E8M0 weight scale, then bf16xbf16 mma. A stays bf16.
// Same 16x256 8-warp tile / same mma schedule as the fp8 _sat body; only HALF the fp8 bytes.
//
// Swizzle kDecSatPermFp4[128] derived by algebraic composition of the verified fp8 PERM128
// (sim_frag.py: FRAG2K extracted from the fp8 _sat path, J=d placement) -> reconstructs identity.
// Each loaded register d (d in 0..3) holds true-K 32-block d of the current 128-block => all 8 fp4 in
// register d share ONE E8M0 scale sBe[n_row, 4*k + d]. Scales are NOT swizzled.
//
// BUILD (node, single GPU):
//   /opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 -ffast-math \
//       -DGB0_N=2048 -DGB0_K=7168 -I<HK>/include -I/opt/rocm/include/hip sat_decode_fp4.cu -o sat_decode_fp4
// ================================================================================================
#define GB0_SKIP_MAIN
#include "grouped_b0.cu"
#include <bit>
#include <hip/hip_fp4.h>

// ---- verified fp4 swizzle (per 128-fp4-block): B_hbm[:, p] = B_true[:, kDecSatPermFp4[p]] ----
static const int kDecSatPermFp4[128] = {
      0,  1,  2,  3,  4,  5,  6,  7,  32, 33, 34, 35, 36, 37, 38, 39,  64, 65, 66, 67, 68, 69, 70, 71,  96, 97, 98, 99,100,101,102,103,
      8,  9, 10, 11, 12, 13, 14, 15,  40, 41, 42, 43, 44, 45, 46, 47,  72, 73, 74, 75, 76, 77, 78, 79, 104,105,106,107,108,109,110,111,
     16, 17, 18, 19, 20, 21, 22, 23,  48, 49, 50, 51, 52, 53, 54, 55,  80, 81, 82, 83, 84, 85, 86, 87, 112,113,114,115,116,117,118,119,
     24, 25, 26, 27, 28, 29, 30, 31,  56, 57, 58, 59, 60, 61, 62, 63,  88, 89, 90, 91, 92, 93, 94, 95, 120,121,122,123,124,125,126,127
};

// e2m1 code (0..7) -> magnitude; code 8..15 = negatives of 0..7.
static const float E2M1_VAL[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
// host: nearest-even encode of x into a 4-bit e2m1 code.
static inline uint8_t e2m1_encode(float x) {
    uint8_t sign = (x < 0.0f) ? 0x8 : 0x0;
    float a = std::fabs(x);
    int best = 0; float bestd = 1e30f;
    for (int c = 0; c < 8; c++) { float d = std::fabs(a - E2M1_VAL[c]); if (d < bestd - 1e-9f) { bestd = d; best = c; } }
    if (best == 0) sign = 0x0;   // canonical +0
    return (uint8_t)(sign | best);
}
static inline float e2m1_decode(uint8_t code) { float m = E2M1_VAL[code & 0x7]; return (code & 0x8) ? -m : m; }

__device__ __forceinline__ float e8m0_to_f32(uint8_t e) { return __uint_as_float((uint32_t)e << 23); } // 2^(e-127), e=0 -> 0
// hardware fp4->bf16 pair with MX scale folded in (v_cvt_scalef32_pk_bf16_fp4). sel picks byte of `reg`.
template<int SEL>
__device__ __forceinline__ bf16_2 cvt_fp4_pair(uint32_t reg, float s) {
    uint32_t bits = std::bit_cast<uint32_t>(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(reg, s, SEL));
    bf16_2 out; __builtin_memcpy(&out, &bits, 4); return out;
}

// ================================================================================================
template <int NN, int KK>
__global__ __launch_bounds__(512, 2)
void grouped_expert_gemm_decode_mxfp4_sat(
        const kittens::gl<bf16, -1, -1, -1, -1> A,      // [Mpacked, K]   bf16 activations
        const kittens::gl<bf16, -1, -1, -1, -1> Bpk,    // [E*N, K/4]     fp4 weights as bf16 (PRE-SWIZZLED)
        const kittens::gl<bf16, -1, -1, -1, -1> C,      // [Mpacked, N]   final (scaled) output
        const uint8_t* __restrict__ sBe,                // [E*N, K/32]    E8M0 per-32-block weight scale
        const int* __restrict__ tasks, int num_tasks) {
    constexpr int BM_DEC = 16, BN = 256, BLOCK_K = 128;
    constexpr int WARPS_COL = 8;
    constexpr int REG_BLOCK_N = BN / WARPS_COL;        // 32
    constexpr int k_iters = KK / BLOCK_K;              // 56
    constexpr int NBLK32  = KK / 32;                   // scales per N-row

    using RT_A   = rt_bf<BM_DEC,      BLOCK_K,   row_l, rt_16x32_s>;   // 16x128 bf16 A
    using RT_BPK = rt_bf<REG_BLOCK_N, BLOCK_K/4, row_l, rt_16x32_s>;  // 32x32  bf16 = 32x128 packed fp4
    using RT_B   = rt_bf<REG_BLOCK_N, BLOCK_K,   row_l, rt_16x32_s>;  // 32x128 bf16 (unpacked B)
    using RT_C   = rt_fl<BM_DEC, REG_BLOCK_N, col_l, rt_16x16_s>;     // 16x32 fp32 accumulator
    RT_A a; RT_BPK bpk; RT_B b; RT_C c;

    const int task = blockIdx.x;
    if (task >= num_tasks) return;
    const int* tk = tasks + (size_t)task * TASK_W;
    const int e = tk[T_EXPERT], mt = tk[T_MTILE], nt = tk[T_NTILE], ERB = tk[T_EROWBEG];
    const int warp_n = warpid();                        // 0..7 (warp_m == 0)

    const int a_row16 = ERB / BM_DEC + mt;
    const int b_row32 = (e * NN) / REG_BLOCK_N + nt * WARPS_COL + warp_n;
    const int c_row16 = ERB / BM_DEC + mt;
    const int c_col32 = nt * WARPS_COL + warp_n;

    const int    r16 = kittens::laneid() % 16;
    const int    n0  = e * NN + nt * BN + warp_n * REG_BLOCK_N;
    const size_t se0 = (size_t)(n0 + r16)      * NBLK32;   // row-block i=0 scale base
    const size_t se1 = (size_t)(n0 + 16 + r16) * NBLK32;   // row-block i=1 scale base

    zero(c);
    #pragma unroll 4
    for (int k = 0; k < k_iters; k++) {
        kittens::load<2>(a,   A,   {0, 0, a_row16, k});   // 16x128 bf16 A (broadcast across warps)
        kittens::load<2>(bpk, Bpk, {0, 0, b_row32, k});   // 32x32  bf16 (= this warp's 32x128 fp4 strip)

        // 4 E8M0 scales for this 128-block (true-block 4k+d), per row-block i.
        float sc[2][4];
        #pragma unroll
        for (int d = 0; d < 4; d++) { sc[0][d] = e8m0_to_f32(sBe[se0 + 4*k + d]); sc[1][d] = e8m0_to_f32(sBe[se1 + 4*k + d]); }

        // unpack fp4->bf16 via the gfx950 HARDWARE scaled-convert: register d -> col-subtile J=d;
        // byte-sel B of the 32-bit reg -> (byteB low nibble -> bf16.x, high nibble -> bf16.y) * scale,
        // exactly matching data[B]=(byteB_lo,byteB_hi). One v_cvt_scalef32_pk_bf16_fp4 per data[] slot.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                const float s = sc[i][d];
                uint32_t reg; __builtin_memcpy(&reg, &bpk.tiles[i][0].data[d], 4);
#ifdef FP4_LOADFLOOR
                // ablation: skip the hw convert (garbage output) -> measures load+mma floor (no convert).
                b.tiles[i][d].data[0] = bpk.tiles[i][0].data[d]; b.tiles[i][d].data[1] = bpk.tiles[i][0].data[d];
                b.tiles[i][d].data[2] = bpk.tiles[i][0].data[d]; b.tiles[i][d].data[3] = bpk.tiles[i][0].data[d];
                (void)reg; (void)s;
#else
                b.tiles[i][d].data[0] = cvt_fp4_pair<0>(reg, s);
                b.tiles[i][d].data[1] = cvt_fp4_pair<1>(reg, s);
                b.tiles[i][d].data[2] = cvt_fp4_pair<2>(reg, s);
                b.tiles[i][d].data[3] = cvt_fp4_pair<3>(reg, s);
#endif
            }
        }
        mma_ABt(c, a, b, c);
    }
    store(C, c, {0, 0, c_row16, c_col32});
}

// ================================================================================================
static bool run_case_fp4_sat(const char* label, const std::vector<int>& Me, int check) {
    const bool do_check = (check >= 1);
    const int E = (int)Me.size();
    const int bm = 16;
    std::vector<int> padded(E), erb(E);
    int Mpacked = 0;
    for (int e = 0; e < E; e++) { padded[e] = ((Me[e] + bm - 1) / bm) * bm; erb[e] = Mpacked; Mpacked += padded[e]; }
    int real_rows = 0; for (int v : Me) real_rows += v;

    std::vector<int> tasks;
    for (int e = 0; e < E; e++)
        for (int mt = 0; mt < padded[e] / bm; mt++)
            for (int nt = 0; nt < N / 256; nt++) { tasks.push_back(e); tasks.push_back(mt); tasks.push_back(nt); tasks.push_back(erb[e]); }
    const int num_tasks = (int)tasks.size() / TASK_W;

    printf("\n=== CASE(MXFP4-SAT,BM16) %s: E=%d, real_rows=%d, Mpacked=%d (pad %.1f%%), tasks=%d ===\n",
           label, E, real_rows, Mpacked, 100.0 * (Mpacked - real_rows) / std::max(1, Mpacked), num_tasks);

    const size_t a_elems = (size_t)Mpacked * K, b_elems = (size_t)E * N * K;
    const size_t b_bytes = b_elems / 2;                 // fp4 packed
    const size_t c_elems = (size_t)Mpacked * N;
    const int    nblk128 = K / 128, nblk32 = K / 32;

    std::vector<bf16>    h_a(a_elems, (bf16)0.0f);
    std::vector<float>   h_aref(do_check ? a_elems : 0, 0.0f);
    std::vector<uint8_t> h_b4(b_elems, 0);              // e2m1 codes (0..15), true-K order
    std::vector<uint8_t> h_b4_swz_packed(b_bytes, 0);   // swizzled + packed 2/byte
    std::vector<uint8_t> h_sBe((size_t)E * N * nblk32, 127);
    std::vector<float>   h_bref(do_check ? b_elems : 0, 0.0f);   // dequant-fp4 (kernel-correctness ref)
    std::vector<float>   h_btrue(do_check ? b_elems : 0, 0.0f);  // TRUE B (precision-class ref)
    std::vector<int>     row_expert(Mpacked, -1);
    std::mt19937 gen(7); std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int e = 0; e < E; e++)
        for (int s = 0; s < padded[e]; s++) {
            int row = erb[e] + s; row_expert[row] = e;
            if (s >= Me[e]) continue;
            for (int h = 0; h < K; h++) { float v = dist(gen); h_a[(size_t)row*K + h] = (bf16)v; if (do_check) h_aref[(size_t)row*K + h] = (float)(bf16)v; }
        }

    // per-32-block MXFP4 quant (E8M0 scale + e2m1 codes) over TRUE-K columns.
    for (int e = 0; e < E; e++)
        for (int n = 0; n < N; n++) {
            const size_t nrow = (size_t)e * N + n;
            std::vector<float> brow(K);
            for (int k = 0; k < K; k++) brow[k] = dist(gen) * 0.1f;
            for (int blk = 0; blk < nblk32; blk++) {
                float amax = 0.0f;
                for (int k = blk*32; k < blk*32 + 32; k++) amax = std::max(amax, std::fabs(brow[k]));
                uint8_t e8; float scale;
                if (amax == 0.0f) { e8 = 127; scale = 1.0f; }
                else { int ex; std::frexp(amax, &ex); int be = ex - 3 + 127; if (be < 1) be = 1; if (be > 254) be = 254; e8 = (uint8_t)be; scale = std::ldexp(1.0f, be - 127); }
                h_sBe[nrow * nblk32 + blk] = e8;
                float invs = 1.0f / scale;
                for (int k = blk*32; k < blk*32 + 32; k++) {
                    uint8_t code = e2m1_encode(brow[k] * invs);
                    h_b4[nrow * K + k] = code;
                    if (do_check) { h_bref[nrow*K + k] = e2m1_decode(code) * scale; h_btrue[nrow*K + k] = brow[k]; }
                }
            }
        }

    // offline swizzle (per 128-fp4-block) + pack 2 codes/byte: byte b holds fp4 cols 2b(low),2b+1(high).
    for (size_t r = 0; r < (size_t)E * N; r++) {
        const uint8_t* src = &h_b4[r * K];
        uint8_t* dst = &h_b4_swz_packed[r * (K/2)];
        for (int blk = 0; blk < nblk128; blk++) {
            const uint8_t* sblk = src + blk*128;
            for (int b = 0; b < 64; b++) {   // 128 fp4 -> 64 bytes
                uint8_t lo = sblk[kDecSatPermFp4[2*b + 0]] & 0xF;
                uint8_t hi = sblk[kDecSatPermFp4[2*b + 1]] & 0xF;
                dst[blk*64 + b] = (uint8_t)(lo | (hi << 4));
            }
        }
    }

    bf16 *d_a, *d_c; uint8_t *d_b4, *d_sBe; int *d_tasks;
    hip_check(hipMalloc(&d_a, a_elems * sizeof(bf16)), "m a");
    hip_check(hipMalloc(&d_b4, b_bytes), "m b4");
    hip_check(hipMalloc(&d_c, c_elems * sizeof(bf16)), "m c");
    hip_check(hipMalloc(&d_sBe, (size_t)E * N * nblk32), "m sBe");
    hip_check(hipMalloc(&d_tasks, tasks.size() * sizeof(int)), "m tasks");
    hip_check(hipMemcpy(d_a, h_a.data(), a_elems * sizeof(bf16), hipMemcpyHostToDevice), "c a");
    hip_check(hipMemcpy(d_b4, h_b4_swz_packed.data(), b_bytes, hipMemcpyHostToDevice), "c b4");
    hip_check(hipMemcpy(d_sBe, h_sBe.data(), (size_t)E * N * nblk32, hipMemcpyHostToDevice), "c sBe");
    hip_check(hipMemcpy(d_tasks, tasks.data(), tasks.size() * sizeof(int), hipMemcpyHostToDevice), "c tasks");
    hip_check(hipMemset(d_c, 0, c_elems * sizeof(bf16)), "ms c");

    kittens::gl<bf16, -1, -1, -1, -1> A(d_a, 1, 1, Mpacked, K);
    kittens::gl<bf16, -1, -1, -1, -1> Bpk((bf16*)d_b4, 1, 1, E * N, K / 4);   // fp4 bytes as quarter-width bf16
    kittens::gl<bf16, -1, -1, -1, -1> Cg(d_c, 1, 1, Mpacked, N);
    const int threads = NUM_WARPS * 64;

    auto launch = [&]() { grouped_expert_gemm_decode_mxfp4_sat<N, K><<<num_tasks, threads>>>(A, Bpk, Cg, d_sBe, d_tasks, num_tasks); };
    for (int i = 0; i < 5; i++) launch();
    hip_check(hipDeviceSynchronize(), "warm"); hip_check(hipGetLastError(), "warm err");

    hipEvent_t s0, s1; hipEventCreate(&s0); hipEventCreate(&s1);
    const int iters = 50;
    hipEventRecord(s0);
    for (int i = 0; i < iters; i++) launch();
    hipEventRecord(s1); hipEventSynchronize(s1);
    float ms = 0; hipEventElapsedTime(&ms, s0, s1); ms /= iters;
    double gflop_real = 2.0 * real_rows * N * K / 1e9, gflop_pad = 2.0 * Mpacked * N * K / 1e9;
    double b_gb = (double)num_tasks * 256.0 * K * 0.5 / 1e9;   // fp4 = 0.5 byte/elem
    printf("  GEMM(MXFP4-SAT): %.4f ms/iter | %.1f TFLOP/s real | %.1f padded | B-stream %.3f TB/s\n",
           ms, gflop_real / (ms*1e-3) / 1e3, gflop_pad / (ms*1e-3) / 1e3, b_gb / (ms*1e-3) / 1e3);

    if (std::getenv("SAT_VS_BF16")) {
        bf16 *d_bbf, *d_cbf;
        hip_check(hipMalloc(&d_bbf, b_elems * sizeof(bf16)), "m bbf");
        hip_check(hipMalloc(&d_cbf, c_elems * sizeof(bf16)), "m cbf");
        hip_check(hipMemset(d_bbf, 0, b_elems * sizeof(bf16)), "ms bbf");
        kittens::gl<bf16, -1, -1, -1, -1> Bbf16(d_bbf, 1, 1, E * N, K);
        kittens::gl<bf16, -1, -1, -1, -1> Cbf(d_cbf, 1, 1, Mpacked, N);
        auto lbf = [&]() { grouped_expert_gemm_decode<N, K><<<num_tasks, threads>>>(A, Bbf16, Cbf, d_tasks, num_tasks); };
        for (int i = 0; i < 5; i++) lbf(); hip_check(hipDeviceSynchronize(), "bf16 warm");
        hipEventRecord(s0); for (int i = 0; i < iters; i++) lbf(); hipEventRecord(s1); hipEventSynchronize(s1);
        float ms_bf = 0; hipEventElapsedTime(&ms_bf, s0, s1); ms_bf /= iters;
        printf("  GEMM(bf16-ref): %.4f ms/iter  => MXFP4 SPEEDUP %.2fx (bf16 streams 4x the bytes)\n", ms_bf, ms_bf / ms);
        hipFree(d_bbf); hipFree(d_cbf);
    }

    bool pass = true;
    if (do_check) {
        hip_check(hipDeviceSynchronize(), "sync");
        std::vector<bf16> h_c(c_elems);
        hip_check(hipMemcpy(h_c.data(), d_c, c_elems * sizeof(bf16), hipMemcpyDeviceToHost), "c->h");
        double nq = 0, dq = 0, nq_t = 0, dq_t = 0, maxa = 0; long pad_nz = 0, checked = 0;
        const int n_stride = (check >= 2) ? 37 : 257;
        std::vector<char> active(Mpacked, 0);
        for (int e = 0; e < E; e++) for (int s = 0; s < Me[e]; s++) active[erb[e] + s] = 1;
        for (int m = 0; m < Mpacked; m++) {
            int e = row_expert[m];
            for (int n = 0; n < N; n += n_stride) {
                double acc = 0, acc_t = 0; const float* ar = &h_aref[(size_t)m*K];
                const float* br = &h_bref[((size_t)e*N + n)*K]; const float* bt = &h_btrue[((size_t)e*N + n)*K];
                for (int k = 0; k < K; k++) { acc += (double)ar[k]*br[k]; acc_t += (double)ar[k]*bt[k]; }
                float got = (float)h_c[(size_t)m*N + n];
                if (active[m]) { double er = std::fabs(acc - got); nq += er*er; dq += acc*acc; maxa = std::max(maxa, er);
                                 double et = std::fabs(acc_t - got); nq_t += et*et; dq_t += acc_t*acc_t; checked++; }
                else if (std::fabs(got) > 1e-3f) pad_nz++;
            }
        }
        double rms_k = std::sqrt(nq / std::max(1e-30, dq));        // kernel correctness (vs dequant-fp4)
        double rms_p = std::sqrt(nq_t / std::max(1e-30, dq_t));    // precision class (vs TRUE bf16 B)
        pass = (rms_k < 0.02) && (pad_nz == 0);
        printf("  CORRECTNESS: RMS-vs-dequant-fp4=%.5f (gate<0.02) | RMS-vs-TRUE-B=%.5f (fp4 precision class) | max_abs=%.4f pad_nz=%ld -> %s\n",
               rms_k, rms_p, maxa, pad_nz, pass ? "PASS" : "FAIL");
    }
    hipFree(d_a); hipFree(d_b4); hipFree(d_c); hipFree(d_sBe); hipFree(d_tasks);
    return pass;
}

int main() {
    printf("sat_decode_fp4 — SATURATING MXFP4 BM16 decode (N=%d K=%d)\n", N, K);
    std::vector<int> me_ragged = {4,16,1,9,15,2,17,7,16,3, 8,12,5,16,1,16,9,4,13,7, 16,2,11,6,15,8,3,16,10,5, 16,4};
    bool h1 = run_case_fp4_sat("decode-ragged-correctness", me_ragged, /*check=*/2);
    bool h2 = run_case_fp4_sat("E32-decode-tiny  (aiter 49.7)", ME_DECODE_TINY, /*check=*/0);
    bool h3 = run_case_fp4_sat("E32-decode       (aiter 171.6)", ME_DECODE,      /*check=*/0);
    printf("\nMXFP4 SAT RESULT: %s\n", (h1 && h2 && h3) ? "PASSED" : "FAILED");
    return (h1 && h2 && h3) ? 0 : 1;
}
