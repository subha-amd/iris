# EXPERIMENT LEDGER — IRISX/HK MoE on 8×MI355X

Durable, committed record of every on-device measurement. The main agent is the ONLY writer here,
and the only actor that runs GPU jobs (serialized under `flock /tmp/mi355x_project_gpu.lock`).
Raw rows also go to `results/results.csv` (gitignored; regenerable — `git add -f` if you want it
tracked). **Record negative results too — never keep only the best number.**

## Columns (also the CSV header)
`candidate, commit, date, ranks, route_dist, M_label(per_expert|aggregate), M, N, K, dtype,
BM, BN, BK, NSUB, schedule, grid_blocks, VGPR, AGPR, SGPR, LDS, scratch, lat_us, p50, p95, p99,
TFLOPs, rms_rel, zero_sentinel, spd_vs_B1, spd_vs_B2, spd_vs_B3, notes`

## Baseline definitions (Agent 01)
- **B0** local V2 HK GEMM, no comm — compute ceiling.
- **B1** IRISX dispatch-pack-quant A ONCE into local fp8+scale buffer, then local V2 GEMM — the
  STRONG unfused baseline. Headline speedups must cite this.
- **B2** production MORI dispatch/pack/quant + AITER/CK fmoe — production baseline.
- **B3** V3 direct-pull, no overlap — historic weak baseline (refetches A per N tile).
- **B4** A-stationary remote pull, no overlap — isolates reuse from overlap.
- **B5** V4 A-stationary + overlap.

## Decision gates (no claim crosses a gate unmet)
- G1 baseline validity: V4 compared to B1 + B2 before any headline.
- G2 production shape: W13/W2 shapes+layouts source-verified (Agent 00) before "R1 expert kernel".
- G3 decode realism: tested on captured EP8 per-expert distributions before "decode win".
- G4 real multi-rank: no production claim from fixed-source np=2.
- G5 end-to-end: no model-level claim until ATOM-integrated under C4/C6.

---

## Phase A — rebaseline canonical V4
_status: DONE 2026-06-25 (main agent, node cv350 / r1_c4, np=2, --mca pml ob1 --mca btl self,vader)_

Build clean (gfx950). Baseline here = V4's in-file `micro_tk_baseline` (= B3-family direct-pull,
no overlap). All RMS_rel=0.00331, local_A_zero=True, C_zero=False (remote gather real).

| candidate | M | N | K | baseline_us | fused_us | spd_vs_B3 | notes |
|---|---|---|---|---|---|---|---|
| V4 astationary | 1024 | 2048 | 7168 | 1240.6 | 679.9 | 1.825x | run1/3 |
| V4 astationary | 1024 | 2048 | 7168 | 1222.3 | 675.6 | 1.809x | run2/3 |
| V4 astationary | 1024 | 2048 | 7168 | 1243.2 | 682.6 | 1.821x | run3/3 (stable, ~44.2 TFLOP/s) |
| V4 astationary | 512  | 2048 | 7168 | 642.5  | 563.0 | 1.141x | crossover into win |
| V4 astationary | 256  | 2048 | 7168 | 490.4  | 546.6 | 0.897x | LOSS (grid-starved) |
| V4 astationary | 128  | 2048 | 7168 | 464.7  | 549.2 | 0.846x | LOSS (grid-starved) |

CONFIRMED: 1.80-1.83x stable at M=1024; wins M>=512; loses M<=256. Matches V4_ASTATIONARY_RESULTS.
CAVEAT (Gate 1 unmet): this baseline refetches A per N-tile (weak). Honest speedup needs B1
(gather-once + local GEMM) from Agent 01's harness — see Phase B below.

## Phase B — strong baselines B0/B1 (GATE 1) — THE HONEST COMPARISON
_status: DONE 2026-06-26 (main agent, cv350/r1_c4, np=2; B0 np=2 consumer-rank). Stable 2 runs._

Harness had 3 bugs found+fixed on the way (host gl-indexing, pybind module name, B0 GEMM OOB
tiling -> replaced with the proven V2 kernel) PLUS a barrier-asymmetry DEADLOCK in run_harness
(only the consumer rank ran the timing loop, so it waited forever on iris.barrier() the producer
never reached; fixed so all ranks run the timing loops symmetrically).

| case | what | M | N | K | lat_us | TFLOPs | rms_rel | sentinel |
|---|---|---|---|---|---|---|---|---|
| B0 | local GEMM, NO comm (compute ceiling) | 1024 | 2048 | 7168 | 164.2 | 183.1 | 0.00331 | n/a |
| B1 | gather-once + local GEMM (STRONG baseline) | 1024 | 2048 | 7168 | 291.4 | 103.2 | 0.00331 | True |
| B1 split | transfer=143.5us + compute=162.1us (serial two-phase) | | | | ~291 | | | |
| V4 | fused (from Phase A) | 1024 | 2048 | 7168 | 678.0 | 44.2 | 0.00331 | True |

*** GATE 1 RESULT (reverses the headline) ***
- vs the WEAK V3 baseline (refetch A 32x): V4 = 1.82x FASTER.
- vs the STRONG B1 baseline (gather A once, then local GEMM): V4 = 291/678 = **0.43x — i.e. V4 is
  ~2.3x SLOWER than gather-once+local-GEMM at M=1024.**
WHY: V4's A-stationary still refetches A 4x (NSUB=8 -> N/(NSUB*BN)=2 superblocks... measured win was
vs 32x). B1 moves A exactly ONCE (143us transfer) then runs a clean 162us local GEMM at 103 TFLOP/s.
V4's fused kernel runs at only 44 TFLOP/s (occupancy 1-2, LDS-bound) AND still re-crosses XGMI.
So the real lesson: the fusion/overlap does NOT beat simply not-refetching. B1 is the bar to clear.
B0 shows the compute ceiling is 164us/183 TFLOP/s, so B1's 162us compute phase is near-optimal;
the only thing fusion can attack is B1's 143us transfer (overlap it under compute). Upper bound if
transfer were FULLY hidden under the 162us compute: ~164us => best case ~1.78x over B1, NOT more.

IMPLICATION: V4 as-is is NOT the design to carry forward. The target is "B1 + overlap transfer under
compute" at high occupancy (=Agent 10 B-single-buffer to lift TFLOP/s, + cache-once so A crosses 1x).
TODO: B1 small-M sweep (M={128,256,512}); B1 at N=4096; then re-rank all schedule candidates vs B1.

## Phase B — strong baselines (B0/B1/B2) + recomputed V4 speedup
_status: PENDING_

## Phase C — single-expert schedule ablations (4P4C / 8-wave / 4-wave / occupancy / XCD / cache)
_status: PENDING_

## Phase D — grouped 32-expert, np=2 (uniform/Zipf/hot/empty)
_status: PENDING_

## Phase E — grouped EP8 multi-source (np=8)
_status: PENDING_

## Phase F — captured production distributions (C4/C6 replay)
_status: PENDING_

## Phase G — full production shapes (W13 / W2 / full FFN)
_status: PENDING_

## Phase H — ATOM integration (TPOT/TTFT/throughput)
_status: PENDING_
