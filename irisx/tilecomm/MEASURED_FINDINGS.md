# Measured findings — on-node profiling of the abstraction candidates (8× MI350X, gfx950, 2026-07-07)

All measured on the fresh dedicated node `smci350-rck-g03-f16-03` (10.190.162.60), inside container
`subha_probe` (rocm/pytorch:latest, torch 2.10+rocm7.2, triton 3.6, IRIS editable). TP4, bf16,
median-of-30 CUDA-event timing, MAX over ranks, `torch.distributed` NCCL(RCCL) reference.

## A. The TP4 all-reduce is the DOMINANT cost of the prefill GEMM→AR sequence (Simran was right)
Unfused reference = full-256-CU torch GEMM (per-rank K-slice) THEN a separate RCCL all-reduce — exactly
what production does today. Across R1-realistic TP4 shapes:

| GEMM (M,N,K) — role                       | GEMM 256CU | RCCL all-reduce | serial GEMM+AR | AR share |
|-------------------------------------------|-----------:|----------------:|---------------:|---------:|
| 8192,4608,36864 (example default)         |   0.606 ms |   0.750 ms      |   1.269 ms     | **59%**  |
| 8192,7168,18432 (R1 down-proj, prefill)   |   0.565 ms |   1.148 ms      |   1.641 ms     | **70%**  |
| 8192,7168,7168  (R1 attn out-proj, prefill)| 0.252 ms |   1.147 ms      |   1.357 ms     | **85%**  |
| 1024,7168,18432 (small prefill)           |   0.093 ms |   0.176 ms      |   0.262 ms     | **67%**  |
| 256,7168,18432  (decode-ish M)            |   0.053 ms |   0.073 ms      |   0.121 ms     | **60%**  |

- **The all-reduce is 59–85% of the serial GEMM+AR time** — it is 1.2–4.5× the GEMM. Comm DOMINATES the
  sequence, it is NOT cheap-relative-to-GEMM. RCCL all-reduce bus BW ≈ 150 GB/s here (modest — either
  RCCL isn't fully tuned for these sizes on gfx950 or per-iter barrier overhead inflates it; the *ratio*
  is the robust takeaway, not the absolute BW).
- **Overlap ceiling for abstraction A (tile-fused reduce-scatter/all-reduce) ≈ serial/AR:**
  down-proj 1.641/1.148 = **1.43×**, attn out-proj 1.357/1.147 = **1.18×**, example 1.269/0.750 = **1.69×**,
  small-prefill 0.262/0.176 = **1.49×**. (Rationale: if the GEMM's output tiles are reduced as they are
  produced, the short GEMM hides under the long AR, so fused ≈ max(GEMM,AR) = AR.) **So A has a real
  ~1.2–1.7× PREFILL ceiling** — better than the traffic-shaping lever — validating Simran's "no AMD engine
  fuses the all-reduce" as a genuine opportunity.

## B. But the current IRIS fused-collective substrate is NOT competitive — A needs real new kernel work
Ran the two shipped IRIS GEMM+all_reduce examples at the 8192,4608,36864 TP4 shape (--gemm_sms 128):

| substrate                                  | fused GEMM+AR total | vs unfused 1.269 ms |
|--------------------------------------------|--------------------:|--------------------:|
| unfused torch GEMM + RCCL all-reduce (ref) |            1.269 ms | 1.0×                |
| ex.09 one-shot all-reduce (IRIS)           |            5.898 ms | **4.7× SLOWER**     |
| ex.08 atomics all-reduce (IRIS)            |          358.8   ms | **280× SLOWER**     |

- Both examples are pedagogical: ex.09 is bottlenecked by a Triton streamK GEMM at **471 TFLOP/s** (vs
  torch's **1149 TFLOP/s**, 2.4× slower) AND an unoptimized one-shot AR; ex.08's per-element cross-rank
  atomic-add AR is catastrophic (7.76 TFLOP/s).
- **Conclusion:** realizing A's ~1.4–1.7× ceiling requires (1) a competitive GEMM body (HipKittens 8-wave,
  NOT the Triton example) with a producer/consumer warp split for async tile-reduce, AND (2) an in-kernel
  reduce-scatter that matches RCCL bandwidth. Both are substantial and (2) is unproven on IRIS. A is a
  high-effort, medium-ceiling, PREFILL-ONLY path. Env note: MI350X = 256 CUs, so the examples' auto
  `gemm_sms = 2^floor(log2(256)) = 256` trips the `gemm_sms >= total_sms` guard — must pass `--gemm_sms`.

## C. Decode (the latency-critical path) has NO TP all-reduce and is weight-bound → B is the real lever
Under DP-attention the TP all-reduce is gone at decode; the expert GEMM streams all 32 local experts'
~1.4 GB fp8 weights/step (~176 µs HBM floor). Gather+combine ≈ 18% of the decode region. So comm fusion
(A or C) caps decode at ~1.1–1.25×; the only high-ceiling decode lever is **fewer weight bytes (fp4)** —
abstraction B (QuantTile). Route-1 W4A16 MXFP4 already shipped ~1.6× on the standalone decode GEMM
(ledger; codex flagged verifying the 3.97 TB/s fp8 baseline — needs the HK build, deferred this session).

## D. Codex adversarial critique (2nd set of eyes) — the sharpest points
- **B (QuantTile) = highest production ceiling** (attacks the dominant decode weight wall, the real serving
  denominator); **A (tile-fused collective) = cleanest comm-thesis** but prefill-only + crowded prior art.
- **Refuses the brief's grand unification:** "transport + format" is an API-level unifier only, NOT one
  coherent optimization problem — A is a dependency/pipeline scheduler, B is a representation/MFMA lowering;
  pitching them as one invites "which cost model arbitrates XGMI-overlap vs HBM-bytes vs scale-layout vs
  MFMA-occupancy vs accuracy?" — a model we don't have.
- **Build QuantTile v0 FIRST** — falsifiable milestone: one HK grouped decode GEMM body serving both
  fp8-sat AND MXFP4 via a tile descriptor, fp8 within ~5% of standalone 3.97 TB/s, MXFP4 keeping ~1.6×,
  NO per-format kernel fork. Explicitly do NOT start by reproducing the 386 µs combine.
- **Empty quadrant = representation + residency, not overlap:** a tile whose *remote residency AND quantized
  representation* are both first-class, lowering to IRIS RMA + CDNA4 scaled-MFMA, "preserving compression
  until the last responsible moment." Overlap (Flux/CoCoNet/MSCCL++/TRT-LLM fused AR+RMSNorm+quant) is taken.
- **Verbs:** `stage` (inbound tile: residency+format+consumer) + `retire` (produced tile: collective+overlap);
  hide load/store/get/put. C→substrate/guardrail; D(sparse)→a policy on top of B.

## Synthesis (my read, reconciling measurement + codex + the mentors' asks)
- **The measurement REFINES codex on one point:** the TP4 AR is NOT cheap — it's 60–85% of the prefill
  GEMM+AR — so A's prefill ceiling (~1.4–1.7×) is real and Awad's "comm abstraction" ask is well-founded
  for PREFILL THROUGHPUT. Codex under-weighted this because it assumed comm was small vs GEMM.
- **But the decode LATENCY path (what serving optimizes) is weight-bound with no AR — so B is the only
  high-ceiling lever there.** A and B attack different regimes (A=prefill throughput, B=decode latency).
- **The needle-threading recommendation:** adopt codex's `stage`/`retire` typed-tile-edge NOTATION (a tile
  whose residency + format are first-class), make **QuantTile-v0 the falsifiable first milestone** (highest
  impact, decode weight wall, clean novelty in the format/lowering axis), and position **tile-fused
  reduce-scatter (A) as the second lowering of the same notation** for the TP4-prefill AR (satisfies Osama's
  overlap prize + Simran's AR-fusion target, with the measured 1.4–1.7× ceiling and the honest build cost).
  Demote C (traffic-shaping) to a guardrail. This gives Awad a genuine tile-level comm+compute abstraction
  whose FIRST proof-point attacks the real serving bottleneck.
