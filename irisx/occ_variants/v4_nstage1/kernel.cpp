// v4_nstage1 — NSTAGE=1 (single-buffer the shared tiles).  This is an LDS-relief variant, NOT a
// direct VGPR variant: VGPR is ~unchanged (~222) because the accumulator/frag set is identical,
// but dynamic LDS drops from NSTAGE=2 to NSTAGE=1:
//   LDS = NSTAGE*(sizeof(ST_A) + NSUB*sizeof(ST_B)) + 1024
//   ST_A = st_bf<64,64> = 64*64*2 = 8192 B ; ST_B = st_bf<64,64> = 8192 B.
//   NSTAGE=2: 2*(8192 + 8*8192)+1024 = 2*73728+1024 = 148480 B  (~145 KB, near the gfx950 cap).
//   NSTAGE=1: 1*(8192 + 8*8192)+1024 =   73728+1024 =  74752 B  (~73 KB).
// gfx950 LDS = 64 KB/CU shared by resident blocks. At ~145 KB the FUSED kernel can NOT co-resident
// two blocks on LDS grounds alone (occupancy already LDS-capped to 1 block, independent of VGPR!).
// Dropping to ~73 KB lets a 2nd block co-reside *if* VGPR also permits — so NSTAGE=1 is a
// PREREQUISITE for any of the VGPR variants above to actually reach occ>1.  COST: loses
// producer/consumer double-buffer overlap (PREFETCH = NSTAGE-1 = 0 -> prologue prefetches nothing,
// every K-tile stalls on its own gather). Likely a latency regression unless paired with enough
// co-resident blocks to hide it. Include in the sweep precisely to measure that trade.
#define BM 64
#define BN 64
#define BK 64
#define NSUB 8
#define NUM_PRODUCER_WORKERS 4
#define NUM_CONSUMER_WORKERS 4
#define NSTAGE 1
#include "../v4_variant_base.cuh"
