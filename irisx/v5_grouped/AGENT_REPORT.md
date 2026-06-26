# AGENT 02 — Grouped 32-expert scheduler (V5) — REPORT

Generalizes V4 A-stationary to ALL E=32 local experts in ONE grid via a host-built flat task list.
NO GPU was touched by this agent (compile was attempted but the node command channel was gated —
see "Static resource info"). All numbers below the kernel changes are **[PREDICTED]**.

## Files changed (new candidate dir `irisx/v5_grouped/`)
- `kernel.cpp`            — grouped fused kernel `micro_tk` + grouped baseline `micro_tk_baseline`
- `example.py`           — grouped np=2 driver (padded packed A/C, B[E*N,K], numpy-CPU grouped ref)
- `build_tasks.py`       — task-list builder + adaptive-NSUB selector + pure-CPU self-test
- `GROUPED_SCHEDULER.md` — design + serial test matrix (5 route distributions, np=2 first)
- `AGENT_REPORT.md`      — this file

## Exact build command (MAIN AGENT, on node)
Mirror this dir to `<HK_ROOT>/distributed-kernels/fmoe_fused_v5_grouped/` (identical contents), then:
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_02 -DDK_BUILD=fmoe_fused_v5_grouped
  flock /tmp/mi355x_compile.lock -c "cmake --build build_02 -j8 --target fmoe_fused_v5_grouped"'
```
Compile-only resource report (no GPU needed):
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES=""; export ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels/fmoe_fused_v5_grouped
  flock /tmp/mi355x_compile.lock -c "hipcc --offload-arch=gfx950 \
    -Rpass-analysis=kernel-resource-usage -c kernel.cpp -I.. <HK include flags> 2>&1 | tee resource.txt"'
```

## Exact PROPOSED GPU test command (MAIN AGENT only, serialized under the GPU lock)
np=2 grouped correctness — START HERE (uniform route, medium size):
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/fmoe_fused_v5_grouped
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    ROUTE=uniform TOTAL_M=8192 E=32 K=7168 N=2048 \
      mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"'
```
Then sweep the matrix (one at a time, same flock):
`ROUTE` in {uniform, zipf, one_hot, several_hot, many_empty} × `TOTAL_M` in {2048, 8192, 32768}.
Pure-CPU host self-test (safe anywhere, no GPU): `python3 build_tasks.py`.

## Expected output (per run)
```
[grouped] route=uniform E=32 TOTAL_M=8192 -> Mpacked=8192 NSUB=8 num_tasks=... n_blocks=...
==========================================================================================
[FUSED   ] route=uniform Mpacked=8192 N=2048 K=7168 NSUB=8 tasks=...  max_abs=... max_rel=... RMS_rel=0.00... local_A_zero=True C_zero=False -> PASSED
[BASELINE] route=uniform ...                                                              RMS_rel=0.00... local_A_zero=True C_zero=False -> PASSED
------------------------------------------------------------------------------------------
  BASELINE (two-phase, no overlap) : ...... us/iter   ..... TFLOP/s
  FUSED    (grouped A-stationary)  : ...... us/iter   ..... TFLOP/s
  SPEEDUP (baseline/fused)         : .....x   (FUSED WINS | baseline wins)
==========================================================================================
```
Also appends a row to `results.csv` with the canonical columns (candidate=`v5_grouped`).

## Correctness criteria
- FUSED RMS-rel `< 0.01` vs the numpy-CPU grouped reference (V4 measured 0.00331; same dequant).
- `local_A_zero=True` — rank-1's local A buffer is all zeros, so a correct C proves the kernel
  gathered A from rank 0 over IRIS (zero-sentinel). This is the key remote-gather proof.
- `C_zero=False` (the kernel actually wrote output) and FUSED == BASELINE within RMS-rel.
- Padding rows are EXCLUDED from the comparison (real-row mask); they are dead space.
- Host self-test (`build_tasks.py`) must print `ALL INVARIANTS PASSED` (no GPU).

## Assumptions
- ABI from AGENT_COMMON §3 (expert_task / expert_offsets) is the contract; V5's 6-int task tuple
  matches the documented `expert_task` fields (with `expert_row_begin` carrying the packed prefix).
- B is expert-major `[E*N, K]` (consistent with rows_per_expert prefix layout). If Agent 00
  finalizes a different B packing, only `b_row0`/`b_tile_row0` in the kernel change.
- fp8 e4m3 OCP on gfx950, fp32 per-128-group scales, bf16 weights/output (FMOE_LAYOUT.md).
- Compile-time `NSUB`=8 is the max the host will request; runtime `nsub` ∈ {8,4,2,1} ≤ that.
- The HK `gl<bf16>` for B reshapes `[E*N, K]` into BN×BK tiles so tile-row index `b_row0/BN + ...`
  is valid (same indexing convention V4 used for its single-expert B). **VERIFY on first compile**:
  if the gl tiling rejects a non-power-of-two leading dim, pass B as `[1,1,E*N,K]` explicitly.

## Known risks
1. **Runtime nsub loops are not `#pragma unroll`** (unlike V4's compile-time NSUB). Slightly more
   loop overhead / possibly higher register pressure from the `C_accum[NSUB]` array still being
   compile-time sized. If occupancy regresses, specialize the kernel per-nsub (template) — V4 had
   it unrolled at 222 VGPR / occupancy 2.
2. **B tile indexing with expert-major B** is the most likely first-compile failure (see assumption
   above). Mitigation noted inline in kernel.cpp.
3. **one_hot route** makes one expert hold ALL rows → very tall M for expert 0; ensure Mpacked
   heap fits (driver uses heap_size_mb=512; bump if TOTAL_M=32768 one_hot overflows).
4. Padding overhead is ≤ E*(BM-1) rows (≤ 2016 for E=32,BM=64) — negligible but real; FLOPs in the
   CSV count only real rows so TFLOP/s is honest.

## Static resource info — [PREDICTED]
Compile on the node was **not completed**: the SSH command channel (`ssh ... 'docker exec ...'`) was
denied by publickey gating even though the interactive banner connects (Conductor SSH-key/reservation
gate). Per the project GPU rule the agent did not block. Predicted from V4 (this kernel is a near
-verbatim generalization — same producer/consumer body, same shared-tile types, +1 small task-decode
prologue and a runtime `nsub` loop bound):
- VGPR ~222 [PREDICTED, =V4]; occupancy ~2 waves/SIMD [PREDICTED]; scratch ~160 B spill [PREDICTED].
- LDS = `NSTAGE*(sizeof(ST_A) + NSUB*sizeof(ST_B)) + 1024` (compile-time, = V4 at NSUB=8).
- Runtime-`nsub` loops may add a few VGPRs vs V4's unrolled loops — MAIN AGENT to measure with
  `-Rpass-analysis=kernel-resource-usage`.

## What the MAIN AGENT runs to validate np=2 grouped correctness (TL;DR)
1. `python3 build_tasks.py` → expect `ALL INVARIANTS PASSED` (no GPU).
2. Build with `build_02` (command above).
3. `ROUTE=uniform TOTAL_M=8192 E=32 K=7168 N=2048` np=2 run (command above) → expect FUSED &
   BASELINE both `PASSED` (RMS_rel<0.01, local_A_zero=True, C_zero=False).
4. Sweep the 5 routes × 3 TOTAL_M; one_hot/many_empty exercise skew + empty-expert zero-task path.
