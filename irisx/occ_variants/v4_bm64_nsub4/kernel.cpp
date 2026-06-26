// v4_bm64_nsub4 — halve NSUB (8->4): half as many live accumulators.
//   C_accum[NSUB] VGPR = NSUB * 4 * (BM/16)*(CONS_N/16) = 4*4*(4*1) = 64 (was 128).
//   a_frag/b_frag unchanged (32/8).  Predicted total VGPR ~158, occ ~3.
//   Grid N-dim doubles (N_PER_BLOCK = NSUB*BN halved): less A reuse per block (A crosses
//   interconnect 2x more than NSUB=8) -> register relief traded against gather amortization.
#define BM 64
#define BN 64
#define BK 64
#define NSUB 4
#define NUM_PRODUCER_WORKERS 4
#define NUM_CONSUMER_WORKERS 4
#define NSTAGE 2
#include "../v4_variant_base.cuh"
