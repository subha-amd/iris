# Speaker script — TileComm research update (2026-07-07)

Deck: `index.html` (open in a browser; ↓/→ or click the right-rail dots; ◐ toggles theme).
Audience: Simran, Muhammad Awad, Muhammad Osama. ~10–12 min + discussion.

The arc: *I built a fused MoE region; the biggest comm win was a hand-rolled schedule; that
schedule should be a library abstraction; here's the abstraction, a calibrated cost model, and
how it drops into what I already have. Second, MXFP4 status + the memory question is settled.*

Lead with the abstraction — that's what Awad and Osama asked for. Keep the recap to 60 seconds.

---

**Slide 1 — Title.** "Last time I showed the fused MoE region. Today I want to go a level up:
make the *communication schedule* itself a first-class, demand-aware library concern — the
tile-level communication abstraction you both pointed at. I'll also close out the MXFP4 and
memory questions."

**Slide 2 — Recap (fast).** "Quick recap so we're grounded: the fused region is 1.56× on
prefill, we collapse 7 kernels to 3, and decode is a weight wall so fusion caps there. The
part I want to build on: our biggest single communication win didn't come from a better
algorithm — it came from changing the *order* blocks issue their remote writes."

**Slide 3 — The observation (the hook).** "The combine collective went from 934 to 386 µs —
2.4× — from one line: round-robin the destination cells across the 8 XGMI links instead of
grouping by rank. Sorted order lets a window of concurrent blocks hammer one link; round-robin
spreads it. A human noticed the imbalance and hand-wrote the fix. That instinct is worth 2.4×.
The question is why a human had to have it."

**Slide 4 — The problem.** "That round-robin is right by luck. Three problems: it balances cell
*count*, not *bytes* — so it breaks under the expert imbalance Simran raised; it's re-derived
per collective — gather has its own, reduce_scatter will need a third; and the default you get
for free is the *pathological* 934 µs order, so the good schedule is opt-in and easy to miss.
The schedule should be *computed by the library* from demand + topology."

**Slide 5 — The abstraction.** "So: TileComm. Three layers. You DECLARE a set of tile
transfers — which tile, to which rank, how big. The library SCHEDULES it — link-balances the
order from the declared demand and the probed XGMI topology; that's the traffic-shaper. Then it
EXECUTEs — lowers to IRIS load/stores inside a HipKittens kernel. Four intents — gather,
scatter, reduce_scatter, all_reduce — cover the whole MoE region and the TP collectives. The
key: you declare once, and bulk-synchronous and tile-fused are two lowerings of the same
thing." (Point at Osama:) "The 'how many tiles before you reduce' permutation space you called
unexplored becomes a schedule the library *searches*, not a constant a human tunes."

**Slide 6 — The quadrant.** "Why is this unoccupied? NCCL is whole-tensor, fixed algorithms.
MSCCL lets you program a collective but at buffer granularity, hand-authored — you still write
the schedule. aiter/MORI bake it in. IRIS gives point primitives plus one all_reduce.
HipKittens describes tiles of *compute* and has no comm concept at all. Nobody describes *tiles
of communication* that a scheduler can balance and a GEMM can overlap. That's the missing verb."

**Slide 7 — The live viz.** "Here's the whole mechanism in one picture. Each tick is a remote
write, colored by destination link. The box is the concurrency window. Watch the bar underneath
— that's the busiest link in the window. Sorted keeps one link saturated and seven idle;
round-robin and proportional keep all eight busy. Same bytes, 2.4× less wall-clock."

**Slide 8 — Cost model.** "I built a wave-based link-contention cost model — the one Simran
asked for — calibrated so uniform combine reproduces the measured 934 and 386. Then I ran three
schedulers across expert skew. Two honest findings. One: the scheduling *decision* is worth
2.4× to 6× over the naive order — that's the big lever, and the abstraction makes it impossible
to fall into the bad order. Two: my demand-aware 'proportional' schedule tracks the link lower
bound at every skew, where round-robin drifts 1–4% and grows with imbalance. I want to be
straight: reorder-only is capped by the hottest link — the *unbounded* win is comm/compute
overlap, which is the tile-fused mode and the next real target."

**Slide 9 — Grounded.** "This isn't vaporware — every intent already exists hand-rolled in my
kernel. combine is a reduce_scatter whose schedule is literally my round-robin; gather is a
tile_gather; the future GEMM+all-reduce is a reduce_scatter with a schedule to search. The
concrete first step is a drop-in: replace the interleave argument with a
`tilecomm.schedule()` call, regression-gate at 386 µs, then beat round-robin on a real
imbalanced trace. Same kernel, measured against a number we already trust."

**Slide 10 — MXFP4 + memory.** "Two things on the second direction. First, the memory question:
I checked — R1 is a non-issue. MI350X is 309 GB each, 2.47 TB across 8; R1 in MXFP4 is 403 GB,
16% of HBM. The old wall was disk, not memory, and the node already has it cached. Second, the
fp4 kernel: Route-1 weight-only fp4 decode is done and shipped — 1.6× on the GEMM — and
weight-only is the *right* decode choice because decode is weight-bound. The honest blocker is
the baseline: my fp4 comparison replayed a buggy, untuned aiter kernel, so beating it means
nothing. The plan is to measure against the real SGLang R1-FP4 stack — which is already on this
node — profiling the MoE region under TP4×DP2 + DP-attention + EP, the realistic config, not
TP8." (To Simran:) "Your offer to run the e2e is the gold denominator — I'd take you up on it."

**Slide 11 — Roadmap + asks.** "Roadmap: an on-node XGMI probe for a second measured point
(running now); the combine drop-in; reduce_scatter as a real IRIS collective; then the
tile-fused version inside the GEMM — the overlap prize; and multi-path routing under skew.
Asks: Simran, the per-expert token-count trace and the e2e run; Osama, a gut check on the
tile-fused reduce_scatter as the schedule-search target; Awad, is declare/schedule/execute the
right shape and is the drop-in the right first milestone?"

---

## Anticipated questions (have answers ready)

**"Is 1–4% over round-robin worth a whole abstraction?"** The 1–4% isn't the pitch — the pitch
is 2.4–6× over the naive order made automatic and correct-by-construction, robustness to
imbalance for free, reuse across gather/combine/reduce_scatter, and the substrate for the
tile-fused overlap win, which *is* large. Reorder-only is the floor of the value, not the ceiling.

**"Why not just always use round-robin then?"** It balances count, not bytes — under expert
imbalance it drifts, and it can't express the tile-fused schedule at all. And "always
round-robin" is itself a per-kernel human decision that the naive path doesn't give you.

**"How is this different from MSCCL / MSCCL++?"** MSCCL programs collectives at buffer
granularity with hand-authored schedules, standalone. TileComm is tile-granular,
*demand-adaptive* (the schedule is computed, not written), and designed to fuse into the
compute kernel's tile loop. The fusion + auto-scheduling is the delta.

**"Your cost model is calibrated to one point — is it real?"** Be upfront: it's a fit to one
measured region number (combine, 934/386), and the mechanism isn't validated yet. I ran an on-node
XGMI probe tonight to test the link-contention story — it was *issue-bound* (IRIS stores are
fire-and-forget, so it timed issue rate not link bandwidth, implying ~4.7 TB/s, >10× the fabric)
and showed only ~1.05×. So the probe is inconclusive, not a refutation — but I won't claim the model
is mechanistically proven. It's a design-space tool calibrated to a real number; fixing the probe's
completion fence is the immediate next step. (This is on slide 6 and slide 11 — lead with it, don't
let someone catch it.)

**"So does scheduling even matter, if the probe shows ~1×?"** The probe couldn't test it (issue-bound).
The 2.4× is a real region measurement; what's open is *why* — pure link-spreading, or the combine's
read/reduction structure. Either way the abstraction's case holds: scheduling should be empirical and
library-owned precisely because the mechanism behind a hand-tuned win isn't obvious. And the big prize
(tile-fused comm/compute overlap) doesn't depend on this at all.

**"On an all-to-all fabric, can ordering even beat the hottest link?"** No — reorder-only is
capped by the hottest link's bytes (I say this on slide 8). Multi-path routing (roadmap step 5)
is what moves that floor, and only when the demand matrix is skewed onto specific links. The
unbounded win before that is hiding comm under compute (tile-fusion).

**"Does the fp4 GEMM help decode?"** (Simran already flagged this.) The tensor cores get faster
but you still stream all the experts' weights, so decode barely moves — weight-only fp4 halves
the *weight bytes*, which is the actual decode bottleneck, so that's the lever, and it's the
Route-1 choice.

**"Did you get a new end-to-end MXFP4 speedup?"** No — deliberately. Route-1 is shipped; the
missing piece is a legitimate external baseline, which needs the real SGLang serve. I didn't
brute-force that on a shared node tonight; the design + cost model was the higher-value use of
the time, and the e2e is the clean way to get the real number.
