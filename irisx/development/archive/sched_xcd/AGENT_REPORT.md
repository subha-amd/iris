# AGENT 08 — XCD-aware grid scheduling + hardware-cache reuse — REPORT

Candidate dir: `irisx/sched_xcd/` (built on `irisx/v4_astationary_kernel/`; V4 untouched).
Branch: `agent/08-xcd-scheduling`. Target: MI355X / gfx950 (`NUM_XCDS=8`, `CUS_PER_XCD=32`).

This agent does NOT touch the GPU. All on-device runs below are PROPOSED for the main agent.

---

## Exact build command (node, compile-only OK for subagent; main agent for the A/B builds)

Mirror this dir under `<HK_ROOT>/distributed-kernels/sched_xcd/` (build auto-discovers
`*/kernel.cpp`), then, on `<NODE>` in container `r1_c4`:

```bash
export HIP_VISIBLE_DEVICES="" ; export ROCR_VISIBLE_DEVICES=""    # subagent compile-only safety
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_08 -DDK_BUILD=sched_xcd
  flock /tmp/mi355x_compile.lock -c "cmake --build build_08 -j8 --target sched_xcd"'
```

The A/B test needs **two builds** of the same source, differing only in the compile-time switch:

```bash
# build A — XCD remap ON (default; XCD_W=8, XCD_C=auto=num_n)
flock /tmp/mi355x_compile.lock -c "cmake --build build_08 -j8 --target sched_xcd"
#   (default XCD_REMAP=1 baked in kernel.cpp)

# build B — exact stock V4 control. Add -DXCD_REMAP=0 to the kernel TU. Easiest: a sibling dir
#   sched_xcd_stock/ whose kernel.cpp is `#define XCD_REMAP 0` then `#include "../sched_xcd/kernel.cpp"`,
#   OR pass -DEXTRA_HIPCC_FLAGS="-DXCD_REMAP=0" if the cmake exposes it.
```

Static resource report (subagent-allowed, recommended to confirm remap added no spills):
```bash
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ; export ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels/sched_xcd
  hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage -fsyntax-only kernel.cpp'
# expected: VGPR/AGPR/LDS/scratch ~ identical to V4 (~222 VGPR, ~160B scratch); remap is pure
# integer index math (a few SGPR), no new shared/register tiles.
```

> COMPILE STATUS: **[NEEDS-NODE]** — node SSH was not reachable from this subagent run, so the
> build was not executed here. Source is a minimal diff over V4 (only `xcd_map_block` + the 3 lines
> deriving block_row/block_n0/n_tile0 from it), expected to compile cleanly. `chiplet_transform_
> chunked`, `NUM_XCDS`, `CUS_PER_XCD` are already in the included `common/util.cuh`.

---

## Proposed GPU test command (MAIN AGENT ONLY, serialized)

Canonical A/B point: **M=1024 N=2048 K=7168**.

```bash
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/sched_xcd
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    M=1024 K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"'
```
Run once per build (A = remap, B = stock). `--mca pml ob1 --mca btl self,vader` is MANDATORY on this
node (R1 server holds VRAM; IB registration would otherwise fail).

### rocprof-compute counters to confirm/refute the L2-reuse hypothesis
Wrap the SAME mpirun under rocprof-compute, rank 1 (the GEMM rank), for BOTH builds:

```bash
flock /tmp/mi355x_project_gpu.lock -c "
  M=1024 K=7168 N=2048 ITERS=20 WARMUP=5 \
  rocprof-compute profile -n xcd_remap_on  -- \
    mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 example.py"
# repeat with build B and -n xcd_remap_off
```

Counters to collect and compare (remap=1 vs remap=0), kernel `micro_tk`:
| signal | counter (rocprof-compute / rocprofv3 metric) | hypothesis predicts (remap=1 vs =0) |
|---|---|---|
| XGMI read bytes | `xgmi_read_bytes` / fabric "Remote Read" data (GPU↔GPU link) | **lower** if remote A is cached & reused on one XCD |
| XGMI read packets / transaction count | XGMI read request/packet count | **lower** (fewer fabric transactions for shared A) |
| L2 (TCC) hit % | `TCC_HIT / (TCC_HIT+TCC_MISS)` (L2 cache hit ratio) | **higher** if remote A lines are reusable in L2 |
| LLC / MALL hit % | Infinity-Cache / last-level (MALL) hit ratio | **higher** if reuse lands in LLC |
| (sanity) total kernel time | `micro_tk` duration | secondary — bytes/hit% are the primary signal |

Collect these via `rocprof-compute analyze` or directly:
```bash
rocprofv3 --pmc TCC_HIT TCC_MISS  -- <mpirun ...>     # L2 hit ratio
# XGMI/MALL metrics: use rocprof-compute's memory-chart / `--block TCC TCP L2 fabric` view.
```

---

## Expected output (example.py)
```
[FUSED-XCD] M=1024 K=7168 N=2048 ... RMS_rel=0.00331 local_A_zero=True C_zero=False -> PASSED
[BASELINE ] M=1024 K=7168 N=2048 ... RMS_rel=0.00331 local_A_zero=True C_zero=False -> PASSED
  BASELINE ...  ;  FUSED-XCD ...  ;  SPEEDUP ...
```
RMS-rel and the PASSED/zero-sentinel lines MUST be **identical for build A and build B** (the remap
is numerics-neutral). Wall-clock may differ slightly; the real verdict is in the counters.

## Correctness criteria
1. `RMS_rel == 0.00331` (same as V4) for the fused path in BOTH builds. **Any drift ⇒ the remap is
   not a clean permutation — STOP** (likely an `XCD_W` that doesn't divide `num_m`; see constraint).
2. `local_A_zero=True` and `C_zero=False` (zero-sentinel: result came from the remote gather).
3. `xcd_map_viz.py` bijection check prints `OK` for the tested `(M, XCD_W)` (CPU pre-check).

## Assumptions
- HW round-robins blocks to XCDs by `linear_block_id % 8` on gfx950 (the premise `chiplet_transform_
  chunked` inverts). If the HW assignment differs (firmware/driver scheduler), the XCD co-location
  is not guaranteed — counters will reveal it.
- `chunk = num_n` makes one M-tile's A-sharing superblocks one contiguous chunk → one XCD.
- V4's grid, gather, dequant, MFMA pipeline are correct and unchanged (verified by RMS-rel parity).

## Known risks / negative-result paths
- **Hypothesis may be FALSE:** remote P2P/XGMI loads may bypass or not reusably populate L2/LLC on
  the consumer XCD; then remap moves XGMI bytes without reducing them (a valid negative result).
- **Ragged-window task loss:** `XCD_W` must divide `num_m` (or `num_m ≤ XCD_W`). Canonical shapes
  (num_m ∈ {8,16,32}) are safe; other M needs `XCD_W` set to a divisor of `num_m`.
- Co-locating an M-tile's superblocks on one XCD could create XCD load imbalance / L2 pressure that
  offsets reuse gains — watch per-XCD occupancy and TCC miss latency.
- Comparison is vs V4 (a weak A-refetching baseline); the honest baseline is Agent 01's B1/B2. Do
  not over-claim any XCD win.

## Files changed (this branch, candidate dir only)
- `irisx/sched_xcd/kernel.cpp`     — V4 + `xcd_map_block` (XCD_REMAP/XCD_W/XCD_C); fused path remapped.
- `irisx/sched_xcd/example.py`     — np=2 head-to-head driver scaffold (for main agent), M/K/N defaults = canonical point.
- `irisx/sched_xcd/xcd_map_viz.py` — CPU-only predictor + bijection verifier of the XCD/CU assignment.
- `irisx/sched_xcd/XCD_SCHEDULING.md` — remap math, predicted mapping, numerics proof, L2-reuse hypothesis.
- `irisx/sched_xcd/AGENT_REPORT.md` — this file.
