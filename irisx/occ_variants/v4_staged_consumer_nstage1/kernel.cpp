// v4_staged_consumer_nstage1 — the COMBINATION candidate (Agent 07's top pick).
//   staged-consumer (4 producers + 8 consumers, each consumer owns 1 of NSUB=8 subtiles) gives
//   low VGPR (~122, NO scratch spill), and NSTAGE=1 halves LDS (145 KB -> 73 KB) so 2 blocks can
//   co-reside per CU. This is the only single design predicted to hit BOTH low register pressure
//   (spill removed) AND occ=2 blocks/CU on the fused path.
//   COST: NSTAGE=1 => PREFETCH=0 => loses producer/consumer double-buffer overlap (each K-tile
//   stalls on its own gather). The bet is that 2x resident blocks hide that stall. MUST be
//   GPU-measured against v4_staged_consumer (NSTAGE=2) to confirm the trade is net-positive.
#define NSTAGE 1
// All other knobs default inside v4_staged_consumer's source (BM64/BN64/BK64/NSUB8/4P+8C).
#include "../v4_staged_consumer/kernel.cpp"
