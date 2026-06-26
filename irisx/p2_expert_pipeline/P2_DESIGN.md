# P2 — Expert-Granular Copy-Once Double-Buffer (B0-GEMM consumer)

## Goal
Same three requirements as P1 — (1) ~1x remote A traffic, (2) B0-class GEMM, (3) overlap — but at
EXPERT granularity, which is the natural unit for a real EP (grouped) MoE and the likely fastest path
to a real EP result. Build on v5_grouped's packed-expert layout + ep8_gather's route_segment ABI.

## Scheme: two rotating local expert slots
Keep TWO rotating LOCAL bf16 expert slots, each sized for the largest expert region
(`slot_rows x K`), `slot = e % 2`. At steady state the producer fills expert e+1 into slot
`(e+1)%2` while the consumer GEMMs expert e from slot `e%2` — gather/compute overlap at expert
granularity, A moved exactly once, GEMM at B0 efficiency.

## Producer/consumer protocol (depth-2 pipeline)
- **Flags:** `ready[e]` (producer->consumer), `done[E+2]` (consumer->producer slot recycle),
  `arrive[e]` (per-expert tile counter, consumer counts DOWN). `ready=(gen<<1)|1` anti-stale.
- **Producer** (own stream, first): `fetch_add(ecursor)` claims expert e; **waits `done[e]`** (slot
  for expert e-2 freed; `done` is +2-offset so `done[0],done[1]` are primed free by the host);
  copy-once multi-source gather+dequant of expert e's padded region into slot `e%2` (fp8 uint4 +
  per-128 scales, local short-circuit when `src_rank==cur_rank`); `fence_system(release)` then
  `atomic_store(release) ready[e]`.
- **Consumer** = B0 grouped GEMM, one block per flat task `(expert, m_tile, n_tile, slot_rows,
  erow_begin)`: acquire-wait `ready[e]`; view slot `e%2` as a `[slot_rows,K]` gl; run the EXACT B0
  inner loop; store into the expert's global packed C region. The LAST block of expert e
  (`fetch_sub(arrive[e])==1`) `fence_system(release)` then sets `done[e+2]`, freeing slot `e%2` for
  expert e+2.

## Variable M_e / empty experts / tail rows / multi-source
- `padded_rows[e]` is a multiple of B0_BM; `valid_rows[e]` bounds real rows. The gather masks at
  `valid_rows` so padding rows read the ZERO sentinel (dequant to 0); padded C rows are dead space no
  one reads (disjoint packed regions, v5_grouped no-contamination layout).
- **Empty expert** (valid_rows==0): producer sets `ready[e]` immediately (no gather); host emits ZERO
  consumer tasks so `arrive[e]==0` and the HOST PRE-SETS `done[e+2]` -> the recycle chain never
  stalls on an empty expert.
- **Multiple source ranks** (np=8): `route_segment[]` (ep8_gather ABI) drive the gather; the producer
  uses the single-source fast path when one segment covers a tile, else a per-row segment lookup.
  Segment `dst_row_begin` are expressed SLOT-LOCAL (host builds segs/expert_meta that way).
- **np=2 single-source** bring-up: ONE segment per non-empty expert, `src_rank=0`,
  `src_row_begin = global packed begin`, `dst_row_begin = 0` (slot-local).

## Deadlock-avoidance argument
1. **CU reservation:** two SEPARATE launches / two non-blocking streams; producer first reserves CUs.
2. **Bounded queue:** this is a classic producer/consumer queue of DEPTH 2. The producer can have at
   most 2 experts outstanding; it makes progress whenever a slot frees. The consumer's awaited expert
   is always eventually produced (producer processes experts in cursor order 0..E-1). The slot
   recycle: producer waits `done[e]`; consumer waits `ready[e]`; consumer frees `done[e+2]` when
   expert e retires. Host primes `done[0],done[1]` so the first two fills proceed and pre-sets
   `done[e+2]` for empty experts. No cycle -> deadlock-free.
3. **No consumer-to-consumer dependency:** `arrive[e]` only gates `done[e+2]`, never a GEMM. A single
   resident producer block-group drains all experts in order.
4. **Anti-stale + reset:** host bumps gen and resets ready/arrive/ecursor + re-primes done each step.

## P1 vs P2 (conceptual)
- **P1** syncs at per-row-band (per-K-band) granularity with a full A[M,K] inbox: simplest, single
  expert / single source, finest overlap, but the inbox is the whole A. Best for the M1024 single-
  GEMM head-to-head vs B1-copy.
- **P2** syncs at per-EXPERT granularity with two rotating slots: bounded memory regardless of E,
  handles variable M_e / empty experts / multi-source, and is the real EP dataflow. Coarser sync =
  fewer flags and a cleaner B0 consumer, but overlap is limited to one-expert-ahead (a very small or
  very large single expert reduces overlap). Best for the grouped 32-expert head-to-head vs
  B1-dispatch.
- Both keep the consumer = verbatim B0, so both should retain B0-class compute efficiency; the
  difference is the comm-overlap structure and the EP realism.

## Metrics
- Primary `T_pipeline`; `compute_retention`; `overlap_efficiency` (vs a same-route serial
  B1-dispatch). Correctness: per-expert RMS-rel vs `dequant(A_e) @ B_e^T`; zero-sentinel.

## Risks
- `slot_rows = max padded_rows`; with a hot/one-hot route slot_rows can be large -> 2 big slots. For
  one_hot (slot_rows = Mpacked) the 2-slot memory ~= 2x a full inbox; acceptable, heap sized 2 GB.
- One-expert-ahead overlap: if the route is one_hot or has one dominant expert, overlap collapses to
  ~serial (that expert's gather can't hide behind a tiny neighbor). Expected; report it.
- `arrive[e]` counts DOWN from tiles_of_e; host MUST reset it each generation (done in example.py).
- `ep8_gather_BM` (=64) must match between kernel.cpp and example.py (GBM).
