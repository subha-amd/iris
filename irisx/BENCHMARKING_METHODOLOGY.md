# Benchmarking a fused MoE kernel against the unfused production path

This is the "how do people normally benchmark multi-GPU fused vs unfused kernels"
reference. It complements `UNFUSED_FUSED_BASELINE_FINDINGS.md` (which records the
specific R1 baseline-hunt) with the general method and the concrete scripts.

## The one rule: compare REGIONS, not kernels

A fusion benchmark is only meaningful if both sides have the **same I/O boundary**:
identical input tensors in, identical output tensors out (within an RMS tolerance).
You define the region by its dataflow, *not* by how many kernels each side launches —
that's the whole point of fusion (fewer kernels, same I/O).

For the C4 decode MoE expert region:

```
ENTER: routed bf16 tokens, present locally, in arbitrary (unsorted) order, + top-k decision
EXIT:  bf16 expert outputs, combined over the top-k, in original token order

production covers it as:  EpDispatch | moe_sorting | dynamic_quant | fmoe_fp8 | EpCombine
the fused path must cover: ep8_gather (=dispatch+sort+quant) | grouped_b0 GEMM | combine
```

If your fused path skips a stage (e.g. b1_dispatch is "born sorted", skips `moe_sorting`),
that is a legitimate *win* — but only if the measured fused region still reaches the same
EXIT. b1_dispatch today has **no combine**, so it is a strict subset of the region and its
714 µs is not yet comparable to the full production region. Add the combine, then compare.

## The standard recipe (what every fused-kernel paper/PR does)

1. **Operationally define the region** by input/output tensors. Write down ENTER/EXIT.
2. **Capture realistic inputs from a real run** — here, the per-expert row counts `M_e`,
   the shapes (K=7168, N=4096 fc1 / 7168 fc2), dtype (fp8 e4m3, fp32 per-128 scales, bf16 out).
   Do **not** use a round-number synthetic and call it production.
3. **Baseline = the stock framework ops, composed over the region, run in sequence**, timed
   with device events, `warmup + N iters + median` (p50; also report p90/p99). Lock GPU
   clocks (`rocm-smi --setperfdeterminism`) so numbers are stable. Tools: `aiter.run_perftest`,
   `torch.cuda.Event`, `triton.testing.do_bench`, rocprofv3/omnitrace for per-kernel.
4. **Candidate = your fused kernel over the identical region**, same inputs, same timing.
5. **Multi-GPU regions**: run on the same #GPUs with the real collective backend (MORI/RCCL).
   The region's production latency is **MAX over ranks** (the decode step waits for the
   slowest rank), averaged over many steps. A single-rank trace *under-counts*. If you must
   measure single-GPU, state that you're measuring the local-compute component and add the
   collective term from a trace or a standalone all-to-all microbench.
6. **Correctness gate first.** RMS-rel vs a high-precision (bf16/fp32) reference must pass
   (~0.003–0.03) before any timing is trusted. A fast wrong kernel is worthless.
7. **Report at multiple operating points** — decode (small `M_e`) and prefill (large `M_e`).
   Fusion wins are operating-point-dependent (the cost model shows the regime flips).
8. **Cross-validate the harness against the trace.** Your standalone baseline's per-kernel
   times should match the production trace's per-kernel times at the same `M_e`. If they
   don't, the harness isn't reproducing production (wrong shapes/tuning/clocks).
9. **Quote the denominator honestly.** Speedup over the *full region*, region stated. Never
   headline a speedup over a weak baseline (this repo already learned that with B3-vs-B1).

## Three tiers of baseline for THIS project (no R1 re-download needed)

| Tier | Baseline | How measured | Status |
|---|---|---|---|
| 1. Trace-grounded | EpDispatch+sort+quant+fmoe+EpCombine | read per-kernel µs off the C4 trace at real `M_e` | data in hand (`c4-highthroughput-query2`: EpDispatch 30.7µs, EpCombine 23.2µs) |
| 2. Replay harness | same chain on one GPU | `b2_production/b2_unfused_region.py` — stock aiter sort+quant+fmoe in sequence + the two XGMI terms; sweep `M_e` | runs today |
| 3. In-framework drop-in | real TP4/DP2 vLLM MoE region | HIP-event timers at the MoE boundary, MAX-over-ranks, end-to-end TPOT; swap impl, same workload | gold standard (Phase H) |

Tier 2 is the answer to "run the same aiter kernels in sequence and measure my version
against the unfused baseline." Because `aiter.fused_moe` is a Python orchestrator that
launches `moe_sorting → dynamic_quant → fmoe → moe_sum` as *separate GPU kernels* (exactly
the names in the C4 trace), timing it **is** running the unfused chain in sequence; the
harness just adds the two MORI cross-GPU kernels it can't run single-GPU as explicit,
overridable terms taken from the trace.

## Scripts

- `b2_production/b2_unfused_region.py` — Tier-2 fair baseline (LATENCY, region accounting).
- `b2_production/b2_aiter.py` — production GEMM efficiency (TFLOP/s, FLOP-normalized; Level-1).
- `b2_production/moe_cost_model.py` — analytic roofline; predicts the regime and the fusion ceiling.
- `grouped_b0/grouped_b0.cu`, `b1_dispatch/example.py` — the fused candidate (Level-1 / Level-2).

## The honesty checklist before any slide says "Nx faster"

- [ ] same region I/O boundary on both sides (combine included on the fused side)
- [ ] same `M_e` (the real decode distribution, captured — not 8192-row prefill)
- [ ] same dtype accounting (where does quant happen on each side?)
- [ ] correctness RMS passes on the fused side at that `M_e`
- [ ] multi-GPU term is MAX-over-ranks (or trace-sourced + labeled)
- [ ] clocks locked, warmup done, median reported, operating point stated
