// Focused driver: fp8 BM16 decode path (exp_12 double-buffer).
#define GB0_SKIP_MAIN
#include "grouped_b0.cu"
int main() {
    printf("focus_decode_fp8 — fp8 BM16 decode (N=%d K=%d)\n", N, K);
    std::vector<int> me_decode_ragged = {4,16,1,9,15,2,17,7,16,3};
    bool h1 = run_case_fp8("decode-ragged-correctness",      me_decode_ragged, /*check=*/2, /*tile_mode=*/1);
    bool h2 = run_case_fp8("E32-decode-tiny  (aiter 49.7)",  ME_DECODE_TINY,   /*check=*/0, /*tile_mode=*/1);
    bool h3 = run_case_fp8("E32-decode       (aiter 171.6)", ME_DECODE,        /*check=*/0, /*tile_mode=*/1);
    printf("\nFOCUS RESULT: %s\n", (h1 && h2 && h3) ? "PASSED" : "FAILED");
    return (h1 && h2 && h3) ? 0 : 1;
}
