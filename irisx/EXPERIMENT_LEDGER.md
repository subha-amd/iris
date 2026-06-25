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

## Phase A — rebaseline canonical V3/V4
_status: PENDING (main agent)_

| candidate | M | N | K | lat_us | spd_vs_B3 | rms_rel | zero_sentinel | notes |
|---|---|---|---|---|---|---|---|---|
| _to be filled_ | | | | | | | | |

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
