// ================================================================================================
// sat_decode.cu — SATURATING fp8 BM=16 decode GEMM (the DECODE-WIN path).
//
// Store B (expert weights) as fp8 (HALF the HBM bytes of bf16) but PRE-SWIZZLE them offline so they
// can be loaded through the FAST bf16 global->register path (a half-width rt_bf<32,64> raw byte copy),
// then UNPACK fp8->bf16 in-register (per-N-row scale, in float, pre-rounding), then bf16xbf16 mma.
// A stays bf16. No LDS, no barriers => the load uses the SAME saturating buffer_load_b128 schedule as
// the proven bf16 decode kernel, on HALF the B bytes => ~2x the bf16 weight-stream rate ~= the fp8
// weight floor. This is the only HK-compatible way to realize fp8's weight-halving on decode (HK
// forbids fp8 global->register, forcing the slow ~2.6 TB/s LDS-staged path otherwise).
//
// Offline swizzle (per 128-K block, per N-row):  B_hbm[n, blk*128 + p] = B_fp8_true[n, blk*128 + PERM[p]]
// PERM derived + lane-by-lane round-trip-verified in roundtrip_layout.py (reconstructs identity for all
// 128 K columns).  This file embeds the same PERM and re-checks it at runtime against the kernel result.
//
// BUILD (node B, single GPU):
//   /opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
//       -I<HK_ROOT>/include -I/opt/rocm/include/hip sat_decode.cu -o sat_decode
// ================================================================================================
#define GB0_SKIP_MAIN
#include "grouped_b0.cu"
#include <bit>

// offline column permutation (per 128-K block): B_hbm[:, p] = B_fp8_true[:, PERM128[p]]
static const int PERM128[128] = {
    0,1,2,3,4,5,6,7, 32,33,34,35,36,37,38,39, 8,9,10,11,12,13,14,15, 40,41,42,43,44,45,46,47,
    16,17,18,19,20,21,22,23, 48,49,50,51,52,53,54,55, 24,25,26,27,28,29,30,31, 56,57,58,59,60,61,62,63,
    64,65,66,67,68,69,70,71, 96,97,98,99,100,101,102,103, 72,73,74,75,76,77,78,79, 104,105,106,107,108,109,110,111,
    80,81,82,83,84,85,86,87, 112,113,114,115,116,117,118,119, 88,89,90,91,92,93,94,95, 120,121,122,123,124,125,126,127
};

// ================================================================================================
// The saturating fp8 decode kernel.  Geometry identical to grouped_expert_gemm_decode (bf16 decode):
// 16(M) x 256(N) output, 8 warps (WARPS_ROW=1 x WARPS_COL=8), each warp owns a disjoint 32-col N-strip.
// Only difference vs the bf16 decode: B is loaded as a half-width bf16 tile (the packed fp8 bytes) and
// unpacked to a full bf16 tile before the (unchanged) bf16 mma.
// ================================================================================================
template <int NN, int KK>
__global__ __launch_bounds__(512, 2)
void grouped_expert_gemm_decode_fp8_sat(
        const kittens::gl<bf16, -1, -1, -1, -1> A,     // [Mpacked, K]   bf16 activations
        const kittens::gl<bf16, -1, -1, -1, -1> Bpk,   // [E*N, K/2]     fp8 weights reinterpreted as bf16 (PRE-SWIZZLED)
        const kittens::gl<bf16, -1, -1, -1, -1> C,     // [Mpacked, N]   final (scaled) output
        const float* __restrict__ sBn,                 // [E*N, K/128]   PER-128-K-BLOCK fp8 scale
        const int* __restrict__ tasks, int num_tasks) {
    constexpr int BM_DEC = 16, BN = 256, BLOCK_K = 128;
    constexpr int WARPS_COL = 8;
    constexpr int REG_BLOCK_N = BN / WARPS_COL;        // 32
    constexpr int k_iters = KK / BLOCK_K;              // 56
    constexpr int NBLK = k_iters;

    using RT_A   = rt_bf<BM_DEC,      BLOCK_K,   row_l, rt_16x32_s>;   // 16x128 bf16 A
    using RT_BPK = rt_bf<REG_BLOCK_N, BLOCK_K/2, row_l, rt_16x32_s>;  // 32x64  bf16 = 32x128 packed fp8
    using RT_B   = rt_bf<REG_BLOCK_N, BLOCK_K,   row_l, rt_16x32_s>;  // 32x128 bf16 (unpacked B)
    using RT_C   = rt_fl<BM_DEC, REG_BLOCK_N, col_l, rt_16x16_s>;     // 16x32 fp32 accumulator
    RT_A a; RT_BPK bpk; RT_B b; RT_C c;

    const int task = blockIdx.x;
    if (task >= num_tasks) return;
    const int* tk = tasks + (size_t)task * TASK_W;
    const int e = tk[T_EXPERT], mt = tk[T_MTILE], nt = tk[T_NTILE], ERB = tk[T_EROWBEG];
    const int warp_n = warpid();                       // 0..7  (warp_m == 0)

    const int a_row16 = ERB / BM_DEC + mt;                                  // A: 16-row units
    const int b_row32 = (e * NN) / REG_BLOCK_N + nt * WARPS_COL + warp_n;   // B: 32-row units
    const int c_row16 = ERB / BM_DEC + mt;                                  // C: 16-row units
    const int c_col32 = nt * WARPS_COL + warp_n;                            // C: 32-col units

    // per-block scales: lane L owns N-rows (n0 + 16*i + L%16), i in {0,1}; one scale per 128-K block.
    const int    r16 = kittens::laneid() % 16;
    const int    n0  = e * NN + nt * BN + warp_n * REG_BLOCK_N;
    const size_t srow0 = (size_t)(n0 + r16)      * NBLK;
    const size_t srow1 = (size_t)(n0 + 16 + r16) * NBLK;

    zero(c);

    #pragma unroll 4
    for (int k = 0; k < k_iters; k++) {
        kittens::load<2>(a,   A,   {0, 0, a_row16, k});   // 16x128 bf16 A slab (broadcast across warps)
        kittens::load<2>(bpk, Bpk, {0, 0, b_row32, k});   // 32x64  bf16 (= this warp's 32x128 fp8 strip)
        const float scale_i[2] = { sBn[srow0 + k], sBn[srow1 + k] };   // per-128-block B scale

        // unpack fp8->bf16 in-register, applying the per-block scale in float (pre-rounding).
        // (i,j,d) loaded -> target (I=i, J=2j+(d>>1), data[2*(d&1) + (b>>1)], subindex b&1). See PERM128.
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
        mma_ABt(c, a, b, c);
    }
    store(C, c, {0, 0, c_row16, c_col32});
}

// ================================================================================================
// host driver — self-contained: bf16 A, per-N-row fp8 B (PRE-SWIZZLED), in-kernel scale => final C.
// Also times the proven bf16 decode kernel on the SAME case for a direct head-to-head (sat fp8 should
// be ~2x faster: half the B bytes at the same saturating bandwidth).
// ================================================================================================
static bool run_case_fp8_sat(const char* label, const std::vector<int>& Me, int check) {
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

    printf("\n=== CASE(fp8-SAT,BM16) %s: E=%d, real_rows=%d, Mpacked=%d (pad waste %.1f%%), tasks=%d ===\n",
           label, E, real_rows, Mpacked, 100.0 * (Mpacked - real_rows) / std::max(1, Mpacked), num_tasks);

    const size_t a_elems = (size_t)Mpacked * K, b_elems = (size_t)E * N * K, c_elems = (size_t)Mpacked * N;
    const int    nblk = K / 128;
    const bool   ref_true = !std::getenv("SAT_REF_FP8");   // default: compare vs TRUE (unquantized) B = region-equiv

    // ---- host: bf16 A; per-128-block fp8 B (logical h_b8), per-block scale sBn, then offline-swizzle ----
    std::vector<bf16>  h_a(a_elems, (bf16)0.0f);
    std::vector<float> h_aref(do_check ? a_elems : 0, 0.0f);
    std::vector<fp8_t> h_b8(b_elems, 0), h_b8_swz(b_elems, 0);
    std::vector<float> h_sBn((size_t)E * N * nblk, 1.0f);
    std::vector<float> h_bref(do_check ? b_elems : 0, 0.0f);
    std::vector<int>   row_expert(Mpacked, -1);
    std::mt19937 gen(7); std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int e = 0; e < E; e++)
        for (int s = 0; s < padded[e]; s++) {
            int row = erb[e] + s; row_expert[row] = e;
            bool active = (s < Me[e]);
            if (!active) continue;                                  // padding rows stay 0
            for (int h = 0; h < K; h++) {
                float v = dist(gen);
                h_a[(size_t)row * K + h] = (bf16)v;
                if (do_check) h_aref[(size_t)row * K + h] = (float)(bf16)v;   // bf16-rounded A for the ref
            }
        }

    for (int e = 0; e < E; e++)
        for (int n = 0; n < N; n++) {
            std::vector<float> brow(K);
            for (int k = 0; k < K; k++) brow[k] = dist(gen) * 0.1f;
            const size_t nrow = (size_t)e * N + n;
            for (int blk = 0; blk < nblk; blk++) {                       // PER-128-K-BLOCK fp8 quant
                float amax = 0.0f;
                for (int k = blk * 128; k < blk * 128 + 128; k++) amax = std::max(amax, std::fabs(brow[k]));
                float sB = (amax > 0.0f) ? (amax / FP8_E4M3_MAX) : 1.0f;
                h_sBn[nrow * nblk + blk] = sB;
                float invB = 1.0f / sB;
                for (int k = blk * 128; k < blk * 128 + 128; k++) {
                    fp8_t q = __hip_cvt_float_to_fp8(brow[k] * invB, __HIP_SATFINITE, __HIP_E4M3);
                    h_b8[nrow * K + k] = q;
                    if (do_check) {
                        if (ref_true) h_bref[nrow * K + k] = brow[k];     // TRUE B (region-equivalent error)
                        else { __hip_fp8_e4m3 f; f.__x = q; h_bref[nrow * K + k] = static_cast<float>(f) * sB; }
                    }
                }
            }
        }

    // offline swizzle: B_hbm[n, blk*128 + p] = h_b8[n, blk*128 + PERM128[p]]
    const int NBLK = K / 128;
    for (size_t r = 0; r < (size_t)E * N; r++) {
        const fp8_t* src = &h_b8[r * K];
        fp8_t* dst = &h_b8_swz[r * K];
        for (int blk = 0; blk < NBLK; blk++)
            for (int p = 0; p < 128; p++)
                dst[blk * 128 + p] = src[blk * 128 + PERM128[p]];
    }

    bf16 *d_a, *d_c; fp8_t *d_b8_swz; float *d_sBn; int *d_tasks;
    hip_check(hipMalloc(&d_a, a_elems * sizeof(bf16)), "m a");
    hip_check(hipMalloc(&d_b8_swz, b_elems), "m b8swz");
    hip_check(hipMalloc(&d_c, c_elems * sizeof(bf16)), "m c");
    hip_check(hipMalloc(&d_sBn, (size_t)E * N * nblk * sizeof(float)), "m sBn");
    hip_check(hipMalloc(&d_tasks, tasks.size() * sizeof(int)), "m tasks");
    hip_check(hipMemcpy(d_a, h_a.data(), a_elems * sizeof(bf16), hipMemcpyHostToDevice), "c a");
    hip_check(hipMemcpy(d_b8_swz, h_b8_swz.data(), b_elems, hipMemcpyHostToDevice), "c b8swz");
    hip_check(hipMemcpy(d_sBn, h_sBn.data(), (size_t)E * N * nblk * sizeof(float), hipMemcpyHostToDevice), "c sBn");
    hip_check(hipMemcpy(d_tasks, tasks.data(), tasks.size() * sizeof(int), hipMemcpyHostToDevice), "c tasks");
    hip_check(hipMemset(d_c, 0, c_elems * sizeof(bf16)), "ms c");

    kittens::gl<bf16, -1, -1, -1, -1> A(d_a, 1, 1, Mpacked, K);
    kittens::gl<bf16, -1, -1, -1, -1> Bpk((bf16*)d_b8_swz, 1, 1, E * N, K / 2);   // fp8 bytes as half-width bf16
    kittens::gl<bf16, -1, -1, -1, -1> Cg(d_c, 1, 1, Mpacked, N);
    const int threads = NUM_WARPS * 64;

    auto launch_sat = [&]() {
        grouped_expert_gemm_decode_fp8_sat<N, K><<<num_tasks, threads>>>(A, Bpk, Cg, d_sBn, d_tasks, num_tasks);
    };
    for (int i = 0; i < 5; i++) launch_sat();
    hip_check(hipDeviceSynchronize(), "warm sync"); hip_check(hipGetLastError(), "warm err");

    hipEvent_t s0, s1; hipEventCreate(&s0); hipEventCreate(&s1);
    const int iters = 50;
    hipEventRecord(s0);
    for (int i = 0; i < iters; i++) launch_sat();
    hipEventRecord(s1); hipEventSynchronize(s1);
    float ms = 0; hipEventElapsedTime(&ms, s0, s1); ms /= iters;
    double gflop_real = 2.0 * real_rows * N * K / 1e9, gflop_pad = 2.0 * Mpacked * N * K / 1e9;
    // B weight bytes streamed (fp8 = 1 byte): one 256xK strip per task.
    double b_gb = (double)num_tasks * 256.0 * K * 1.0 / 1e9;
    printf("  GEMM(fp8-SAT): %.4f ms/iter | %.1f TFLOP/s real | %.1f TFLOP/s padded | B-stream %.3f TB/s\n",
           ms, gflop_real / (ms * 1e-3) / 1e3, gflop_pad / (ms * 1e-3) / 1e3, b_gb / (ms * 1e-3) / 1e3);

    // ---- head-to-head: the proven bf16 BM=16 decode on the SAME task list (2x the B bytes) ----
    if (std::getenv("SAT_VS_BF16")) {
        bf16 *d_bbf, *d_cbf;
        hip_check(hipMalloc(&d_bbf, b_elems * sizeof(bf16)), "m bbf");
        hip_check(hipMalloc(&d_cbf, c_elems * sizeof(bf16)), "m cbf");
        hip_check(hipMemset(d_bbf, 0, b_elems * sizeof(bf16)), "ms bbf");   // values irrelevant for bandwidth timing
        kittens::gl<bf16, -1, -1, -1, -1> Bbf16(d_bbf, 1, 1, E * N, K);
        kittens::gl<bf16, -1, -1, -1, -1> Cbf(d_cbf, 1, 1, Mpacked, N);
        auto launch_bf16 = [&]() { grouped_expert_gemm_decode<N, K><<<num_tasks, threads>>>(A, Bbf16, Cbf, d_tasks, num_tasks); };
        for (int i = 0; i < 5; i++) launch_bf16();
        hip_check(hipDeviceSynchronize(), "bf16 warm");
        hipEventRecord(s0);
        for (int i = 0; i < iters; i++) launch_bf16();
        hipEventRecord(s1); hipEventSynchronize(s1);
        float ms_bf = 0; hipEventElapsedTime(&ms_bf, s0, s1); ms_bf /= iters;
        printf("  GEMM(bf16-ref): %.4f ms/iter | B-stream %.3f TB/s   => sat fp8 SPEEDUP %.2fx\n",
               ms_bf, 2.0 * b_gb / (ms_bf * 1e-3) / 1e3, ms_bf / ms);
        hipFree(d_bbf); hipFree(d_cbf);
    }

    bool pass = true;
    if (do_check) {
        hip_check(hipDeviceSynchronize(), "sync");
        std::vector<bf16> h_c(c_elems);
        hip_check(hipMemcpy(h_c.data(), d_c, c_elems * sizeof(bf16), hipMemcpyDeviceToHost), "c->h");
        double num_sq = 0.0, den_sq = 0.0, max_abs = 0.0; long padded_nz = 0, checked = 0;
        const int n_stride = (check >= 2) ? 37 : 257;
        std::vector<char> is_active(Mpacked, 0);
        for (int e = 0; e < E; e++) for (int s = 0; s < Me[e]; s++) is_active[erb[e] + s] = 1;
        for (int m = 0; m < Mpacked; m++) {
            int e = row_expert[m];
            for (int n = 0; n < N; n += n_stride) {
                double acc = 0.0; const float* arow = &h_aref[(size_t)m * K]; const float* brow = &h_bref[((size_t)e * N + n) * K];
                for (int k = 0; k < K; k++) acc += (double)arow[k] * (double)brow[k];
                float got = (float)h_c[(size_t)m * N + n];
                if (is_active[m]) { double err = std::fabs(acc - (double)got); num_sq += err * err; den_sq += acc * acc; max_abs = std::max(max_abs, err); checked++; }
                else if (std::fabs(got) > 1e-3f) padded_nz++;
            }
        }
        double rms_rel = std::sqrt(num_sq / std::max(1e-30, den_sq));
        pass = (rms_rel < 0.05) && (padded_nz == 0);
        printf("  CORRECTNESS(fp8-SAT): RMS-rel=%.5f (tol 0.05), max_abs=%.4f, samples=%ld, padded nz=%ld -> %s\n",
               rms_rel, max_abs, checked, padded_nz, pass ? "PASS" : "FAIL");
    }

    hipFree(d_a); hipFree(d_b8_swz); hipFree(d_c); hipFree(d_sBn); hipFree(d_tasks);
    return pass;
}

// per-token row counts (same as focus_decode_fp8 / grouped_b0 main)
int main() {
    printf("sat_decode — SATURATING fp8 BM16 decode (N=%d K=%d)\n", N, K);
    std::vector<int> me_ragged = {4,16,1,9,15,2,17,7,16,3, 8,12,5,16,1,16,9,4,13,7, 16,2,11,6,15,8,3,16,10,5, 16,4};
    bool h1 = run_case_fp8_sat("decode-ragged-correctness", me_ragged, /*check=*/2);
    bool h2 = run_case_fp8_sat("E32-decode-tiny  (aiter 49.7)", ME_DECODE_TINY, /*check=*/0);
    bool h3 = run_case_fp8_sat("E32-decode       (aiter 171.6)", ME_DECODE,      /*check=*/0);
    printf("\nSAT RESULT: %s\n", (h1 && h2 && h3) ? "PASSED" : "FAILED");
    return (h1 && h2 && h3) ? 0 : 1;
}
