// v4_bm64_nsub6 — keep BM=64, NSUB=6 (midpoint of the accumulator-count sweep).
//   C_accum[NSUB] VGPR = 6 * 4 * (64/16)*(16/16) = 6*4*(4*1) = 96 (was 128 at NSUB=8).
//   a_frag = 32, b_frag = 8.  Predicted total VGPR ~190, occ likely still 2 (just under the
//   256-VGPR/occ-3 cliff -> may or may not clear 170-VGPR threshold for occ 3; borderline).
//   Same N_PER_BLOCK=384 non-divisor-of-2048 tail issue as v4_bm32_nsub6.
#define BM 64
#define BN 64
#define BK 64
#define NSUB 6
#define NUM_PRODUCER_WORKERS 4
#define NUM_CONSUMER_WORKERS 4
#define NSTAGE 2
#include "../v4_variant_base.cuh"
