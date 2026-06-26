# EXPERIMENT LEDGER — IRISX/HK MoE on 8×MI355X

Durable, committed record of every on-device measurement. The main agent is the ONLY writer here,
and the only actor that runs GPU jobs (serialized under `flock /tmp/mi355x_project_gpu.lock`).
Raw rows also go to `results/results.csv` (gitignored; regenerable — `git add -f` if you want it
tracked). **Record negative results too — never keep only the best number.**

## Columns (also the CSV header)
`candidate, commit, date, ranks, route_dist, M_label(per_expert|aggregate), M, N, K, dtype,
BM, BN, BK, NSUB, schedule, grid_blocks, VGPR, AGPR, SGPR, LDS, scratch, lat_us, p50, p95, p99,
TFLOPs, rms_rel, zero_sentinel, spd_vs_B1, spd_vs_B2, spd_vs_B3, notes`

## Baseline definitions (canonical names — enforced 2026-06-26 after Gate 1)
- **B0** local V2 HK GEMM, no comm — compute ceiling (~164us / 183 TFLOP/s @ M1024).
- **B1-copy** one remote copy of A (+scales) into a local buffer, then exact local B0 GEMM — the
  strong copy-once baseline. **Headline speedups must cite B1-copy (or B1-dispatch), never B3.**
- **B1-dispatch** route-aware dispatch/pack/quant + local GROUPED GEMM (the real EP baseline once
  grouped lands). No candidate is a production winner until it beats B1-dispatch.
- **B2** MORI/ATOM dispatch + AITER/CK fmoe — production baseline (must be compared, not just beaten).
- **B3** repeated-direct-pull serial baseline (refetches A per N-tile, ~32x). NOT "production/strong".
- **B4** A-stationary remote pull, SERIAL (no overlap) — isolates reuse from overlap.
- **B5** A-stationary remote pull + overlap = the V4 "direct-pull A-stationary mechanism proof".
- **P1** copy-once tile-inbox overlap (producer copies A 1x + ready flag; consumer = B0 GEMM).
- **P2** expert-granular double-buffer overlap (gather expert e+1 while GEMM computes e).
Decision: P1/P2 < B1 -> continue tile-overlap research; ~= B1 -> abstraction useful, perf weak;
> B1 -> bulk dispatch+local GEMM is the preferred dataflow (optimize dispatch, not a fused GEMM).

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
| B1-copy | one remote copy + exact local B0 GEMM | 1024 | 2048 | 7168 | 291.4 | 103.2 (end-to-end) | 0.00331 | True |
| B5 (V4) | direct-pull A-stationary mechanism proof (fused) | 1024 | 2048 | 7168 | 678.0 | 44.2 (end-to-end) | 0.00331 | True |

CORRECTIONS (per redirection 2026-06-26):
- TFLOP/s columns for B1-copy (103) and B5 (44) are END-TO-END EFFECTIVE throughput, NOT GEMM
  efficiency. B1-copy's COMPUTE PHASE (~162us) is ~185 TFLOP/s = matches B0's 183. So B1's GEMM is
  B0-class; the 103 number is dragged down by the serial 143us transfer. Do NOT say "B1 GEMM = 103".
- 291 (T_total) vs 143+162=305 (separate timed() medians): these are NOT a verified same-iteration
  serial decomposition — they are medians from SEPARATE timed() loops, each of which re-runs the
  full run_fn (the split timers call phase_T and phase_C as independent run_fns, so each carries its
  own warmup/launch/sync overhead -> their sum overcounts vs the single combined loop). Marked
  STILL-UNKNOWN until same-iteration cuda-event instrumentation (task #44) confirms the split.
- NAMING (enforced from here): B3 = "repeated-direct-pull serial baseline" (NOT production/strong).
  V4 = "direct-pull A-stationary mechanism proof" = B5 (NOT production fused winner).

*** GATE 1 RESULT (reverses the headline) ***
- vs the repeated-direct-pull serial baseline B3 (refetch A 32x): V4/B5 = 1.82x FASTER.
- vs the strong copy-once baseline B1-copy: V4/B5 = 291/678 = **0.43x (V4 is ~2.33x SLOWER).**
WHY: B5 still re-crosses XGMI for A (~4x at NSUB=8) AND runs at only 44 TFLOP/s end-to-end (LDS-bound,
occ 1-2). B1-copy moves A once then runs a B0-class GEMM. Fusion/overlap does NOT beat not-refetching.
DECISION RULE (M1024/N2048/K7168): perfect-overlap floor = max(T_copy,T_gemm)=max(143,162)=162us;
B1 total 291us => theoretical MAX speedup over B1 ~= 291/162 ~= 1.80x (a CEILING, not a target).
NEW DIRECTION: a competitive design needs simultaneously (1) ~1x remote A traffic, (2) B0-class GEMM
efficiency, (3) comm/compute overlap. That is the copy-once tile-inbox pipeline (P1) or expert-
granular double-buffer (P2), NOT V4-as-written. V4 is now a schedule ablation, not the main line.
B1 transfer = 7,569,408 B / 143us ~= 53 GB/s (payload-only) -- check if B1 transfer itself can go
faster; a faster B1 raises the bar and shrinks the overlap headroom.

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
