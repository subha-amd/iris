// v4_cons8_nsub8 — REQUESTED: CONS_N=8 (NUM_CONSUMER_WORKERS=8 on BN=64) to shrink b_frag and
// accumulator columns.  NEGATIVE RESULT (compile-time): INFEASIBLE with the current rt_16x16
// accumulator.
//
//   CONS_N = BN / NUM_CONSUMER_WORKERS = 64 / 8 = 8.
//   C_accum = rt_fl<BM, CONS_N=8, col_l, rt_16x16_s>.
//   rt.cuh static_assert:  cols % base_tile_cols == 0  ->  8 % 16 != 0  ->  COMPILE ERROR.
//   (b_frag = rt_bf<8,64,...,rt_16x32_s> would also fail: 8 % 16 != 0.)
//
// To actually realize CONS_N=8 you would need a 16x8 (or 8x8) accumulator base shape, which does
// not exist in HipKittens cdna4 rt_shape.cuh (smallest col tiling is 16).  Reducing accumulator
// COLUMN footprint below 16 is therefore not expressible without new HK base shapes + matching
// mma/store paths.  The register win we CAN get by touching the consumer dimension instead comes
// from the staged-consumer variant (fewer NSUB accumulators per consumer), see
// ../v4_staged_consumer/.
//
// This file is intentionally left as a guarded stub so the build tree documents the dead end
// without breaking the whole-tree build.  Define ALLOW_INFEASIBLE_CONS8 to force the (failing)
// instantiation if you want to see the compiler diagnostic yourself.
#ifdef ALLOW_INFEASIBLE_CONS8
#define BM 64
#define BN 64
#define BK 64
#define NSUB 8
#define NUM_PRODUCER_WORKERS 4
#define NUM_CONSUMER_WORKERS 8   // -> CONS_N=8 -> static_assert fires (intended).
#define NSTAGE 2
#include "../v4_variant_base.cuh"
#else
// Empty TU: variant is documented-infeasible. See header comment + REGISTER_OCCUPANCY.md.
int v4_cons8_nsub8_infeasible_placeholder = 0;
#endif
