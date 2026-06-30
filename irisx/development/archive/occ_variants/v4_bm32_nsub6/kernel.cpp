// v4_bm32_nsub6 — combine BM=32 with NSUB=6 (between 4 and 8).
//   C_accum[NSUB] VGPR = 6 * 4 * (32/16)*(64/16... CONS_N=16 -> /16=1) = 6*4*(2*1) = 48.
//   a_frag = 16, b_frag = 8.  Predicted total VGPR ~120, occ ~3-4.
//   NOTE: N=2048,BN=64 -> N/BN=32 N-tiles; N_PER_BLOCK=6*64=384 does NOT divide 2048 evenly
//   (2048/384 = 5.33). ceil_div gives 6 N-blocks, last block has only 2 valid N-subtiles ->
//   wasted MFMAs on 4 padding subtiles unless tail-masked. Documented as a partial-tile risk.
#define BM 32
#define BN 64
#define BK 64
#define NSUB 6
#define NUM_PRODUCER_WORKERS 4
#define NUM_CONSUMER_WORKERS 4
#define NSTAGE 2
#include "../v4_variant_base.cuh"
