// v4_bm32_nsub8 — halve BM (64->32) to halve every accumulator/a_frag base-tile height.
//   C_accum[NSUB] VGPR = NSUB * 4 * (BM/16)*(CONS_N/16) = 8*4*(2*1) = 64 (was 128).
//   a_frag VGPR        = 4 * (BM/16)*(BK/32)            = 4*(2*2)   = 16 (was 32).
//   Predicted total VGPR ~142, occ ~3 waves/SIMD.  Grid M-dim doubles (BM halved).
// Pure -D reparameterization of v4_variant_base.cuh.
#define BM 32
#define BN 64
#define BK 64
#define NSUB 8
#define NUM_PRODUCER_WORKERS 4
#define NUM_CONSUMER_WORKERS 4
#define NSTAGE 2
#include "../v4_variant_base.cuh"
