# V5 — Grouped (all-experts-in-one-grid) scheduler

Generalizes **V4 A-stationary** (one expert per launch) to all `E=32` local experts in **one**
kernel launch. A real EP-MoE has wildly uneven rows-per-expert `M_e`; launching one V4 grid per
expert either serializes 32 tiny launches or pads every expert to a common `M` and burns work on
empty experts. V5 fuses all experts into one flat task grid.

Do NOT edit `v3_fused_kernel/` or `v4_astationary_kernel/`. This is a new candidate dir.

---

## 1. One grid over all experts

The host (`build_tasks.py`) flattens per-expert tiles into a flat int32 array
`tasks[num_tasks][6]` and launches `grid.x = num_tasks` (one block per task). Each task tuple:

```
tasks[i] = (local_expert, m_tile_begin, valid_rows, n_superblock, nsub, expert_row_begin)
```

| field | meaning |
|-------|---------|
| `local_expert`     | expert `e` in `[0,E)` → selects B base row `e*N` |
| `m_tile_begin`     | row offset WITHIN expert `e`'s padded region, multiple of `BM` |
| `valid_rows`       | `<= BM` real rows (tail mask; padding rows read zero in the gather) |
| `n_superblock`     | which `NSUB`-wide N panel (`0 .. ceil(N/(NSUB*BN))-1`) |
| `nsub`             | runtime NSUB for THIS launch (same for all tasks; carried for ABI clarity) |
| `expert_row_begin` | expert `e`'s first row in the GLOBAL padded packed-A/C space |

Tile order is expert-major → M-tile → N-superblock. **Empty experts emit zero tasks** (no wasted
blocks, no contamination).

A block reconstructs exactly one V4 block:
`block_row = expert_row_begin + m_tile_begin`, `b_row0 = local_expert*N`,
`block_n0 = n_superblock*(nsub*BN)`.

## 2. No cross-expert contamination (host BM-padding, not masked store)

**Chosen scheme:** the host pads each expert's packed A/C region UP to a multiple of `BM`
(`build_packed_layout`), so the regions are disjoint and BM-aligned. Then a block's
`[block_row, block_row+BM)` always lies inside ONE expert's padded rows — a **full-tile store can
never spill** into the next expert's rows. Correctness comes from:

1. **Disjoint padded regions** → the store target is always in-bounds for this expert.
2. **Gather masks at the TRUE `valid_rows`** (`gr < block_row + valid_rows`) → the
   `BM - valid_rows` padding rows read ZERO and contribute zero to the MFMA. The padding C rows
   they produce are written but live in dead padding space no consumer ever reads.

This is preferred over an element-wise masked store: the store stays a single fast full-tile store;
the only cost is `< BM` padding rows per expert in the packed buffer (≤ `E*(BM-1)` extra rows
total, i.e. ≤ 2016 rows for E=32, BM=64 — negligible vs `Mpacked`).

## 3. Per-expert B

B is packed expert-major `[E*N, K]`; expert `e`'s weights start at row `e*N`. The kernel's B-tile
row index is `b_row0/BN + n_tile0 + sub`.

## 4. Adaptive NSUB (host)

`choose_nsub` picks the **largest** NSUB in `{8,4,2,1}` that still leaves
`>= 256` runnable blocks (prefer `>= 512`), where

```
aggregate_blocks(NSUB) = sum_e ceil(M_e / BM) * ceil(N / (NSUB*BN))
```

Larger NSUB = more A-reuse (less redundant cross-GPU gather) but FEWER blocks; small/skewed routes
need a smaller NSUB to keep the 256-CU grid full and hide IRIS gather latency. Falls back to the
smallest candidate (most blocks) if even NSUB=1 can't reach 256 (a genuinely tiny route).
Compile-time `NSUB` is the MAX the host will request (default 8) so shared/register buffers are
always large enough; the kernel iterates only the first `nsub` of them.

## 5. Inner K-loop + 4P/4C wave schedule — preserved VERBATIM from V4

The producer/consumer split (4 permanent producer + 4 permanent consumer warps), NSTAGE
double-buffering, the inner K-loop, the single-A-load-reused-NSUB-times consumer body, and the
`s_waitcnt`/`s_barrier`/`s_setprio`/`sched_barrier` scheduling are byte-for-byte V4. V5 only
changes WHICH `(expert, m, n)` tile a block computes and the B base row. Schedule redesign
(8-wave ping-pong, 4-wave interleave, occupancy) is owned by Agents 04/05/07.

---

## Proposed serial test matrix (np=2 first)

All on the SAME `E=32, K=7168, N=2048, BM=BN=BK=64`. Vary the route distribution and TOTAL_M.
Run serially under the GPU lock (MAIN AGENT only). Pass criteria per run: FUSED RMS-rel `< 0.01`
vs the numpy-CPU grouped reference, `local_A_zero=True` (zero-sentinel proves remote gather),
FUSED == BASELINE within RMS-rel, and FUSED `C_zero=False`.

| # | route | description | what it stresses |
|---|-------|-------------|------------------|
| 1 | `uniform`     | every expert gets `TOTAL_M/E` rows | balanced grid, baseline correctness |
| 2 | `zipf`        | `M_e ∝ 1/e^1.2` (heavy head) | realistic decode skew; adaptive NSUB |
| 3 | `one_hot`     | ALL rows → expert 0 | worst skew; one expert dominates; tail mask |
| 4 | `several_hot` | 4 hot experts, rest empty | mixed empty + hot; empty-expert zero-task path |
| 5 | `many_empty`  | 6 active of 32 | many empty experts emit zero tasks; grid stays full via NSUB |

Sweep TOTAL_M ∈ {2048, 8192, 32768} for each (small/medium/large) to exercise the adaptive-NSUB
ladder (small TOTAL_M → NSUB drops from 8→4→2→1 to keep ≥256 blocks).

Run command (per cell), MAIN AGENT only, serialized under the GPU lock:

```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/fmoe_fused_v5_grouped
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    ROUTE=zipf TOTAL_M=8192 E=32 K=7168 N=2048 \
      mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"'
```

## Host-only self-test (no GPU)

`build_tasks.py` has a pure-CPU `_selftest()` (`python3 build_tasks.py`) that checks all five
invariants for every route: empty experts → zero tasks; every tile lands inside its padded expert
region; `num_tasks == aggregate_blocks(nsub)`; padded regions disjoint + monotonic. Safe to run
anywhere (numpy only, no torch/GPU).
