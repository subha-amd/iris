# Research plan — the tile-level communication+compute abstraction: what to build first, and why

> **Format:** auto-gpu-kernel `research.md` plan (Diagnosis → Strategy → Recommendation → Actions →
> Do-not-try → Coordination). **Audience:** Muhammad Awad, Muhammad Osama, Simran Arora.
> **Source of truth for every number:** `scratchpad/MEASURED_FINDINGS.md` (on-node TP4 profiling,
> 8× MI350X gfx950, TP4, bf16, median-of-30, MAX-over-ranks), `MASTER_HANDOFF.md §5/§6/§10`,
> and the four design docs under `irisx/tilecomm/design/`. Ceilings are labelled as ceilings; every
> unmeasured input is flagged. Status: **plan, not a result** — the two lowerings are designed and
> ceiling-sized, not yet built.

---

## Diagnosis

Serving R1 splits into two regimes with **different bottlenecks and different levers**, and the
measurement pins both. **Decode (the latency path) is a weight-memory wall:** the expert GEMM streams
all 32 local experts' ~1.4 GB fp8 weights every step (~176 µs HBM floor at 8 TB/s), decode ≈ 80% weight
floor + 20% boundaries, so *any* comm/overlap fusion caps decode at ~1.1–1.25× and the only high-ceiling
lever is **fewer weight bytes** — fp4, already shipped at MXFP4 **~1.6× over the fp8 `_sat` decode GEMM**
and **1.11× / 1.69× on the fused decode region (520.6 vs 578.9 fp8 vs 881.3 bf16 µs, TOTAL_M=512)**
(`MEASURED_FINDINGS §C`, `MASTER_HANDOFF §5.1/§10.1`). **Prefill (the throughput path) is
communication-dominated:** the TP4 all-reduce after the GEMM is **59–85% of the serial GEMM+AR time**
(1.2–4.5× the GEMM itself), so fusing it has a real **~1.2–1.7× ceiling** (down-proj 1.43×, attn out-proj
1.18×, example-default 1.69×, small-prefill 1.49×) — but the shipped IRIS fused-collective examples that
would realize it are **4.7× (ex.09) to 280× (ex.08) slower than unfused torch+RCCL**, because ex.09's
Triton streamK GEMM is 471 vs torch's 1149 TFLOP/s (2.4× off) *and* its one-shot AR has zero overlap
(`MEASURED_FINDINGS §A/§B`). So decode and prefill want **two different kernels attacking two different
walls**, and the traffic-shaping lever we first reached for is worth only **1–4% over a hand-rolled
round-robin** (`DESIGN §4/§7`) — a guardrail, not a contribution.

## Strategy

**Refactor** — adopt one notation (`stage`/`retire` typed-tile edges: residency + format as first-class
type axes) and ship its **format-axis lowering first** (QuantTile-v0, decode weight wall) as the
falsifiable proof the abstraction is real and zero-cost, then its **residency-axis lowering second**
(tile-fused reduce-scatter, TP4-prefill AR); this reuses every verified-good component instead of
re-deriving comm from scratch, and it makes the highest-serving-impact wall the *first* proof-point.

## Recommendation (lead here)

**Build QuantTile-v0 FIRST. Position tile-fused reduce-scatter (Abstraction A) as the SECOND lowering of
the same `stage`/`retire` notation. Demote traffic-shaping (Abstraction C) to a correctness guardrail.**

The unifying idea is a **type system for tile edges** (`design/STAGE_RETIRE_MODEL.md`): a tile's
**residency** (it may live on another rank → IRIS RMA) and its **format** (it may be a compressed,
block-scaled encoding → CDNA4 scaled-MFMA / in-register dequant) are first-class type parameters, and the
author writes two verbs — `stage(dst)` (inbound: fill an HK tile from a declared residency+format) and
`retire(c)` (outbound: dispose of a produced tile via a declared collective+overlap) — instead of
hand-coding `load/store/get/put`/dequant. This is the genuine **tile-level COMMUNICATION+compute
abstraction** Awad asked for, and — critically — its **first proof-point attacks the real serving
bottleneck**, not a toy.

**Why QuantTile-v0 leads (highest serving impact + cleanest novelty + falsifiable now):**

1. **It attacks the decode weight wall — the actual serving denominator.** Decode latency is what
   PD-disaggregated serving optimizes, decode is ~80% weight floor, and fewer weight bytes is the *only*
   lever above the ~1.25× comm-fusion cap (`MASTER_HANDOFF §5.1`). The lever is already proven: MXFP4
   **~1.6×** standalone, **1.11×/1.69×** in-region.
2. **Cleanest novelty, in the format/lowering axis.** HipKittens hard-codes a **separate GEMM per quant
   format** (`kernels/gemm/fp8fp32`, `kernels/gemm/mxfp8`), and our own MoE fork hard-codes
   `_fp8_sat` vs `_mxfp4_sat` — two kernels that are **~90% identical** (same BM=16/BN=256/BK=128 8-warp
   skeleton, same bf16 `mma_ABt`, differing only in packed-B width, in-register unpack, scale layout, and
   the offline column permutation; `design/QUANTTILE_V0.md §0`). **A format-carrying tile that lowers to
   one body** turns "write a kernel per quant format" into "declare a tile's format" — the empty quadrant
   is *representation as a first-class tile property* on AMD, which the adversarial prior-art sweep found
   unoccupied in the tile-RMA world (IRIS/HipKittens carry no format; Tilus carries no transport;
   `design/NOVELTY.md`).
3. **Falsifiable against numbers we already trust, this week.** The v0 milestone is one templated
   `grouped_b0_gemm_decode_qt<N,K,F>` body + one `qt::traits<F>` descriptor set replacing the two forked
   decode kernels + the bf16 reference, with **no `#ifdef`/copy-paste per-format kernel remaining**, gated
   against the *self-referential* trusted numbers (fp8 within ±5% of the standalone GEMM, MXFP4 keeping
   the win, bit-stable vs the pre-unification kernels). The contribution is **the abstraction + a
   demonstration that it is zero-cost**, not a new speed number (the speedups already exist in the two
   shipped kernels).

**Why Abstraction A is SECOND, not first (real ceiling, honest build cost, prefill-only):**

- A's ceiling is **real and measured-sized: 1.18–1.7×, shape-dependent** — the AR *dominates* the prefill
  GEMM+AR sequence, so this validates both Osama's overlap prize ("how many tiles you produce before you
  reduce… that permutation space is insane and unexplored") and Simran's target ("no AMD engine fuses the
  all-reduce"). The measurement actually **refines** the earlier assumption that comm was small vs the
  GEMM — it is not; it is 59–85% of the sequence.
- But A is **prefill-throughput-ONLY.** Under the realistic **TP4×DP2 + EP + DP-attention** config, the TP
  all-reduce is **essentially gone at decode** (attention is data-parallel), so **A contributes 0× to the
  decode-latency path** (`design/OVERLAP_A.md §4`). It touches the prefill half of PD-disagg serving, not
  the denominator QuantTile attacks.
- And A is **high build cost with an unproven half.** Realizing its ceiling needs *both* (a) a HipKittens
  producer/consumer-warp GEMM body with an async tile-reduce epilogue (the naive in-GEMM path stalls 13×:
  36 vs 466 TFLOP/s, `MASTER_HANDOFF §6`) *and* (b) an in-kernel reduce-scatter matching RCCL bandwidth
  (**unproven on IRIS** — the XGMI probe was issue-bound). A 2.4×-slow GEMM body alone already makes the
  fused kernel >1× *slower* than unfused (471 TFLOP/s → 1.48 ms GEMM stage > the entire 1.269 ms unfused
  serial), so **you cannot fuse your way out of a slow GEMM** — the competitive body is a hard prerequisite
  (`design/OVERLAP_A.md §1a`).

**Why C (traffic-shaping) is demoted to a guardrail:** reorder-only scheduling on a fully-connected fabric
is bounded by the hottest link; it buys 2.4–6× over the *naive* CSR order (the 934→386 µs combine) but
only **1–4% over the hand-rolled round-robin** already shipped (`DESIGN §4`). Its value is
correctness-by-construction (the author can never fall into the 934 µs cliff) + robustness to expert
imbalance — a substrate for A's schedule, not a headline.

**The honest boundary we hold (do not oversell the unification):** `stage`/`retire` is an **API-level
unifier — one declaration surface, two independently-lowered backends — NOT one coherent cost model.** A
is a dependency/pipeline scheduler; B is a representation/MFMA lowering. Pitching them as one optimization
problem invites the fatal reviewer question — *"which single cost model arbitrates XGMI-overlap vs
HBM-bytes vs scale-layout vs MFMA-occupancy vs accuracy?"* — a model we do **not** have and must not claim
(`design/STAGE_RETIRE_MODEL.md §6`). The two axes are orthogonal by construction (residency doesn't change
numerics, format doesn't change topology), so each is *entitled* to its own validated cost model; they
interact in exactly one cell (fp4-over-XGMI combine), which is scoped as future work, explicitly unmeasured.

---

## Actions (priority-ordered; each names the exact kernel/file/milestone and the number it is gated against)

**A0 — Re-verify the fp8 `_sat` 3.97 TB/s decode-GEMM baseline on the fresh HK build (pre-req, cheap).**
Codex flagged this: QuantTile-v0's load-bearing gate G1 rests on the 3.97 TB/s anchor, and it is a
same-node standalone number that needs the current HK build to re-confirm before the delta rests on it
(`design/STAGE_RETIRE_MODEL.md §7`, `MEASURED_FINDINGS §C`). Run the trusted standalone GEMM harness at
N=4096,K=7168,BM=16, same node, rotating buffers, best-of-N CUDA-event.
**Gate:** reproduce 3.97 TB/s within noise; if it drifts, re-anchor G1 to the freshly measured value.

**A1 — Build QuantTile-v0 (the lead deliverable).** Collapse three bodies into one templated body +
descriptor, no per-format fork:
- `grouped_b0_gemm_decode_fp8_sat<N,K>` (`kernel.cpp:1272`) → `grouped_b0_gemm_decode_qt<N,K,fp8e4m3>`
- `grouped_b0_gemm_decode_mxfp4_sat<N,K>` (`kernel.cpp:1438`) → `grouped_b0_gemm_decode_qt<N,K,mxfp4>`
- `grouped_b0_gemm_decode<N,K>` (`kernel.cpp:1014`) → `grouped_b0_gemm_decode_qt<N,K,bf16>` (reference)
- fp8/mxfp4 unpack inner loops (`kernel.cpp:1310-1325` / `1478-1492`) → `qt::unpack<F>` (the ONLY
  format-specific line inside the loop)
- `kDecSatPerm`/`kDecSatPermFp4` → `qt::traits<F>::preperm` (unified: on-device, pointer-cached, both)
- `DECODE_SAT`/`DECODE_MXFP4` env fork → one `dispatch_grouped_gemm_b0_decode_qt` switching on `format`

**Gates (falsify v0 if any misses; `design/QUANTTILE_V0.md §3`):**
| gate | threshold | anchor |
|---|---|---|
| **G1** fp8 no-regression (unified@fp8 vs trusted standalone) | within ±5% → ≥ 3.77 TB/s | 3.97 TB/s (A0-reconfirmed) |
| **G2** mxfp4 win preserved (unified@mxfp4 vs @fp8, same shape) | mxfp4 ≤ fp8_time / 1.5 | 1.6× |
| **G3a** correctness vs format-faithful dequant | fp8 RMS ≤ 5e-3, mxfp4 RMS ≤ 5e-3 | 0.0037 / 0.0033 |
| **G3b** unification bit-stable vs pre-unification kernels | RMS ≤ 1e-4 (fp-reassoc noise) | regression guard |
| **G4** fused decode region still wins (unified@mxfp4 vs @fp8, TOTAL_M=512) | ≥ 1.10× | 520.6/578.9 = 1.11× |

G1 is load-bearing: it proves the descriptor is a `constexpr` policy (zero-cost), not a dynamic-dispatch
tax. **Scope discipline:** A stays bf16 (W4A16); no native fp4×fp4 MFMA; no prefill; no combine/gather
changes; **do NOT start by reproducing the 386 µs combine**; do NOT gate against the captured aiter a4w4.

**A2 — Fix the XGMI probe + ship `reduce_scatter` as a bulk-synchronous IRIS collective (unblocks A;
Osama's named stepping stone).** The probe (`irisx/tilecomm/xgmi_probe.py`) is issue-bound (IRIS stores
fire-and-forget; `do_bench` timed issue rate not link BW, implying ~4.7 TB/s > 10× the fabric). Fix =
store-completion fence + enough bytes/link to back-pressure + MAX-over-ranks timing. Then implement
reduce-scatter as bulk-sync (all-reduce = reduce-scatter + all-gather): partition M×N into W owner-chunks,
each rank `iris.store`s its W−1 non-owned chunks (link-balanced via the `tilesched.py` Layer-2 schedule),
each owner sums arrivals **locally in fp32** (not ex.08's per-element fabric atomics), per-owner-chunk
arrival counter, then all-gather.
**Gate:** in-kernel RS bandwidth **matches RCCL ~150 GB/s MAX-over-ranks** moving 2(W−1)/W·output_bytes
(example shape: 113 MB/rank → ~0.755 ms ≈ RCCL 0.750 ms). Until this is measured with completion fences,
"matches RCCL" is a *target*, not a result.

**A3 — Port a competitive HK GEMM body into the IRIS fused-collective slot (A's compute half).** Replace
ex.09's Triton streamK body (471 TFLOP/s) with the HK 8-wave ping-pong body that already exists for
bf16/fp8/mxfp8 (`HipKittens/kernels/gemm/mxfp8/MXFP8_8wave` = 421 TFLOPS reference; `fp8fp32/FP8_8wave`) —
the body our shipped `grouped_b0_gemm` derives from, so a known ~torch-rate quantity.
**Gate (first, before any overlap):** the fused kernel must reach **≤ 1.0× the unfused 1.269 ms** (i.e.
merely *match* torch+RCCL) — which already beats the shipped examples by 4.7–280×. A body below ~torch
rate can never close (§1a), so this gate is a hard filter before A4.

**A4 — Tile-fused `retire` epilogue + the `G` sweep (the full Overlap-A; Osama's prize).**
Warp-specialize the epilogue on the A3 body: producer warps run the MFMA pipeline uninterrupted, a
consumer/DMA warp drains each *completed* output tile into the A2 reduce-scatter, so tile-i's comm
overlaps tile-(i+1)'s compute. The schedule reduces to primarily **one scalar knob** `G` = tiles produced
before firing a reduce batch, with a modeled optimum `G* ≈ sqrt(P·L/(t_c+t_m))` (example shape: P=576
tiles, t_c≈1.05 µs, t_m≈1.30 µs, **L UNMEASURED** → G*≈27, i.e. ~one M-block-row, neither 1 nor "all";
`design/OVERLAP_A.md §3`).
**Gate:** correctness (RMS vs a torch AR reference) first; then claw from ≤1.0× toward the shape-dependent
**1.18–1.7× prefill ceiling** (realistic *delivered* ~1.1–1.4× on comm-bound shapes, eroded by the
all-gather tail + imperfect overlap). **Quote it as a prefill number; never mix denominators.** `L` must
be measured on-node before `G*` is anything but a design estimate.

---

## Do-not-try (with reasons + evidence)

- **Reorder-only traffic-shaping as the headline contribution.** Capped at **1–4% over the hand-rolled
  round-robin** (`DESIGN §4`: proportional vs round_robin `RR/prop` ≤ 1.03 across the Zipf sweep). The
  2.4× (934→386 µs combine) is 2.4× over the *naive CSR order*, not over a good schedule. Keep C as a
  **correctness guardrail** (impossible-to-fall-into-the-cliff) + robustness-to-imbalance substrate for
  A's `G` schedule; do not re-invest in it as a lever.
- **The shipped IRIS fused examples (ex.08/09) as A's substrate as-is.** Measured **4.7× (ex.09) to 280×
  (ex.08) slower than unfused torch+RCCL** at the example shape (`MEASURED_FINDINGS §B`). ex.09's Triton
  GEMM alone (471 TFLOP/s → 1.48 ms) is already 1.16× slower than the *entire* unfused 1.269 ms serial, so
  no amount of AR fusion closes it; ex.08's per-element cross-rank atomic AR is 7.76 TFLOP/s. They are
  pedagogical. A needs a **net-new** competitive body (A3) + a real in-kernel RS (A2).
- **The grand-unified-cost-model pitch ("transport + format as one optimization problem").** There is **no
  scalar objective** over the 5 incommensurable currencies (µs-hidden-comm, HBM-bytes, scale-layout,
  MFMA-occupancy, RMS-accuracy); a reviewer finds the seam in 30 seconds
  (`design/STAGE_RETIRE_MODEL.md §6`). Frame the contribution as an **interface** unifier with two
  **separately-validated** cost models, decoupling justified by axis-orthogonality. "Unify" applies to the
  declaration surface, never the objective function.
- **Native fp4×fp4 `mma_ABt_scaled<cbsz=4,blgp=4>` in v0.** Two unresolved on-device unknowns: (1) the fp4
  operand byte→K layout with K doubling to 256 is undocumented (the same scramble the fp8 `_sat` swizzle
  had to be reverse-engineered for) and can silently transpose K; (2) whether `pack_scales` accepts 8
  per-32 E8M0 blocks for K=256 or only 4 (⇒ per-64, lossy vs MXFP4's per-32) is unverified
  (`MASTER_HANDOFF §10.1 Route-2`, `design/QUANTTILE_V0.md §7.2`). v0's mxfp4 uses in-register
  `cvt_scalef32_pk_bf16_fp4` + standard bf16 MFMA (K=128, verified, RMS 0.0033) — neither unknown applies.
  Route-2's value is **prefill**; it is v1, gated by a standalone `mxfp8`-body probe with cbsz flipped.
- **Gating any headline against the captured aiter `a4w4` region.** It is **untuned AND buggy** for our EP
  shape — every `block_size_M` sweep falls back to `2stage default`, EP fmoe 691 µs is slower than aiter's
  own fp8 EP (503), ROCm/aiter #3632/#2343 confirm the CK-a4w4+EP path is immature (`MASTER_HANDOFF §10.1`).
  Beating a baseline we generated means nothing. Use the reproducible SGLang stack or Simran's e2e run.
- **Any decode all-reduce fusion.** Under DP-attention there is **no TP all-reduce at decode** to fuse
  (`MASTER_HANDOFF §10 opportunity map`); the decode collective is the MoE all-to-all (already fused), and
  decode is weight-bound. A is prefill-only — do not size it from a decode trace or a TP8 trace.

---

## Coordination notes (a concrete ask for each advisor)

- **Muhammad Awad (the abstraction is the contribution).** The deliverable he asked for is the
  `stage`/`retire` **typed-tile-edge programming model** (residency + format first-class), and its **first
  falsifiable lowering is QuantTile-v0** (format axis, decode weight wall) — a genuine tile-level
  communication+compute abstraction whose first proof-point hits the real serving bottleneck, not a slide.
  **Ask:** bless (1) the **"interface unifier, not one cost model"** framing (two orthogonal axes, two
  separately-validated backends, unification only at the declaration surface) as the defensible thesis
  shape, and (2) the **bulk-synchronous-first → concurrency** path he already advocated (A2 bulk-sync
  reduce-scatter before A4 tile-fusion). Confirm he agrees QuantTile-v0 leading (vs opening with a comm
  collective) still satisfies "the contribution is the abstraction, not raw speed" — it does, because v0
  proves `stage(...).format(...)` is a real compiler dispatch, zero-cost, against a trusted number.
- **Muhammad Osama (the overlap prize + reduce_scatter).** His "how many tiles before you reduce"
  permutation space is concretized in A as the single knob `G` with modeled optimum
  `G* ≈ sqrt(P·L/(t_c+t_m)) ≈ 27` tiles (`design/OVERLAP_A.md §3`), and his suggested next collective
  (reduce_scatter, since all-reduce = reduce_scatter + all-gather) is A2. **Ask:** gut-check the
  **tile-fused reduce-scatter schedule and the producer/consumer warp split** — specifically (a) does the
  async-epilogue design (producer warps uninterrupted, one consumer/DMA warp draining completed tiles)
  match what he'd expect to avoid the 13× stall we measured on the naive in-GEMM path; (b) is `G` the right
  single degree of freedom or does the permutation space have structure we're flattening; (c) confirm `L`
  (per-batch transfer latency) is the one unmeasured input that makes `G*` a design estimate until probed.
- **Simran Arora (realistic config + the AR-fusion target + e2e).** Three asks: **(1) take her up on the
  e2e serve** — our external fp4 baseline is *fake* (captured aiter a4w4, untuned+buggy), and she offered
  to run the full reproducible SGLang DeepSeek-R1 stack; that trace is the **gold denominator** for both
  the QuantTile region win and A's prefill AR, and it replaces the 403 GB brute-force serve we couldn't
  land on a shared node (`MASTER_HANDOFF §10`). **(2) Confirm the R1 checkpoint scale format** — is the
  production fp8 checkpoint she targets **per-128-K fp32** (aiter/DeepSeek native) or **genuinely MX**
  (`amd/DeepSeek-R1-MXFP4`, powers-of-two)? v0 does **not** depend on the answer (fp8 uses a fp32-fold
  route, checkpoint-agnostic), but the **v1 `native_scaled_mfma` path is only correct for MX checkpoints**
  — feeding fp32-per-128 into E8M0 corrupts the result (`design/QUANTTILE_V0.md §7.1`). **(3) A directly
  serves her "no AMD engine fuses the all-reduce"** — but be explicit with her that **A is prefill-only**
  (the TP AR is gone at decode under DP-attention); the **decode-latency path she cares about most is
  served by QuantTile (B)**, and A is the prefill-throughput complement. Do not let the AR-fusion
  excitement pull the decode roadmap onto a lever that contributes 0× to decode.
