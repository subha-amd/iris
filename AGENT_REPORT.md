# AGENT 01 REPORT — strong baselines + unified B-case harness

Deliverable: a single compile-ready harness that benchmarks every candidate (B0..B5) on identical
tensors/layouts/precision/correctness and emits rows in the exact `irisx/results/results.csv`
schema. NO GPU was touched by this agent (GPU rule). Items needing the node are marked `[NEEDS-NODE]`.

## Files changed (all under `irisx/harness/`, plus this report)
- `irisx/harness/harness_common.py` — shared tensor / quant / dequant / correctness / timing /
  CSV-schema / grouped-distribution utilities (single source of truth for "identical").
- `irisx/harness/run_harness.py` — the unified B-case runner (B0..B5 dispatch, CSV emit, sentinel,
  split timers for B1, `--list` / `--dry-run` GPU-free modes).
- `irisx/harness/harness_kernels.cpp` — pybind module `harness_kernel`: `local_gemm` (B0) and
  `dispatch_pack_quant_once` (B1). Reuses HK tile primitives + IRIS exactly like v3/v4.
- `irisx/harness/CMakeLists.txt` — build recipe for `harness_kernel` (mirrors v3/v4 pattern).
- `irisx/harness/BENCHMARK_METHODOLOGY.md` — what each B-case is, timing, per_expert-vs-aggregate
  rule, B2 wiring gap.
- `AGENT_REPORT.md` — this file.

## Build command (node, inside container `r1_c4`; subagent compile-only is allowed)
B0/B1 (the new harness_kernel):
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""   # compile-only, no device
  cp -r <repo>/irisx/harness <HK_ROOT>/distributed-kernels/harness
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_01 -DDK_BUILD=harness -DIRIS_HIP_ARCHITECTURES=gfx950
  flock /tmp/mi355x_compile.lock -c "cmake --build build_01 -j8 --target harness_kernel"'
```
B3/B4/B5 reuse the EXISTING candidate modules — build them as already documented:
```
# V3 (for B3):  cmake -B build_01 -DDK_BUILD=fmoe_fused_v3 ... --target fmoe_fused_v3
# V4 (for B4/B5): cmake -B build_01 -DDK_BUILD=fmoe_fused_v4_astationary ... --target fmoe_fused_v4_astationary
```
(`<HK_ROOT>`=`/home/...`, `<repo>` = the iris checkout. Placeholders per AGENT_COMMON.)

## Proposed GPU run commands (MAIN AGENT ONLY, under flock, np=2 first)
All run from the dir holding the relevant built module (so the `tk_kernel`/`harness_kernel` `.so`
imports), with the mandatory MPI flags. `CASE`, `M`, `N`, `K`, `M_LABEL`, `ROUTE` select the row.

Compute ceiling + strong baseline (B0,B1) — run in the harness build dir (np=2; B0 ignores comm):
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/harness
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  for M in 8 16 32 64 128 256 512 1024; do
    flock /tmp/mi355x_project_gpu.lock -c "
      CASE=B0 M=$M N=2048 K=7168 M_LABEL=per_expert ROUTE=single \
        mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 run_harness.py
      CASE=B1 M=$M N=2048 K=7168 M_LABEL=per_expert ROUTE=single \
        mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 run_harness.py"
  done'
```
Historic + isolation + fused (B3 in V3 dir; B4/B5 in V4 dir):
```
# B3 (V3 build dir):
CASE=B3 KERNEL_VARIANT=v3 M=$M N=2048 K=7168 M_LABEL=per_expert ROUTE=single \
  mpirun ... -np 2 python3 <repo>/irisx/harness/run_harness.py
# B4 + B5 (V4 build dir):
CASE=B4 KERNEL_VARIANT=v4 NSUB=8 M=$M N=2048 K=7168 ... -np 2 python3 .../run_harness.py
CASE=B5 KERNEL_VARIANT=v4 NSUB=8 M=$M N=2048 K=7168 ... -np 2 python3 .../run_harness.py
```
Shapes to sweep: W13 gate/up `K=7168, N∈{2048,4096}`; W2/down `K=2048, N=7168`.
Grouped 32-expert (aggregate M): `ROUTE∈{uniform,zipf,onehot,captured} M_TOTAL=8192 N_EXPERTS=32 M_LABEL=aggregate`.
np=8 later: same commands with `-np 8` once multi-source routing (Agent 03) lands.

## Expected output (per run)
A line per callable case, e.g.:
```
[B1] M=256 N=2048 K=7168 M_label=per_expert route=single  lat=...us  TFLOPs=...  rms_rel=0.0033  zero_sentinel=True  OK
```
and one appended row in `irisx/results/results.csv` with the full schema. B2 prints
`[B2] SKIPPED: ... [NEEDS-NODE]` and emits no row until wired.

## Correctness criteria (identical for all cases)
- `rms_rel < 0.10` (bf16 GEMM tolerance; v2/v3 observed ~0.0033–0.0037).
- `C` not all-zero.
- comm cases: `zero_sentinel == True` (consumer's local A was the zero sentinel → C came over IRIS).
- B0: `zero_sentinel == n/a` (no remote read by design).

## What is compile-checkable here vs main-agent-only
- **Compile-checkable (node, no GPU):** `harness_kernels.cpp` (the B0/B1 pybind module). It uses the
  same HK tile types + IRIS device-view + pybind binding pattern as the building-verified v3/v4
  kernels, so it should compile under gfx950 in `r1_c4`. Static VGPR/AGPR/SGPR/LDS/scratch can be
  extracted compile-only and pasted into the CSV's resource columns (currently `[NEEDS-NODE]`):
  `hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage -c harness_kernels.cpp` then
  `llvm-objdump -d --mcpu=gfx950` / `roc-obj`.
- **Main-agent-only (on device):** every latency/TFLOPs/rms_rel/zero_sentinel number, i.e. running
  `run_harness.py` under mpirun. Subagents must not.
- **Python syntax:** `python -c "import ast; ast.parse(open('run_harness.py').read())"` — could not be
  executed by this agent (local Bash was unavailable this session); the commit step ran it (see
  Verification). No torch/iris import happens at module import, so `--list`/`--dry-run` are GPU-free.

## Assumptions
- B3/B4/B5 reuse the existing `tk_kernel.dispatch_micro(a, sc, b, c, ctx, M,N,K, src_rank, fused)`
  ABI verbatim (confirmed from v3/v4 kernel.cpp pybind). B4 = V4 with `fused=0`, B5 = V4 `fused=1`,
  B3 = V3 `fused=0`.
- `harness_kernel.local_gemm` and `dispatch_pack_quant_once` take the bf16-reinterpreted fp8 view
  (`gl<bf16>` over M*K bytes), exactly as v3/v4 pass `A_fp8_bf16` — the runner builds these views.
- IRIS owns the MPI lifecycle (mpi4py auto-init disabled), per v3/v4 example.py.
- B0's GEMM uses a compact 256x256x64 8-wave tile (correct, reuses HK primitives). For *peak* B0
  TFLOP/s the main agent may swap in the exact v2 8_wave ping-pong schedule from
  `v2_hk_expert_gemm/fmoe_expert_v2.cu::expert_gemm_bf16` (noted in methodology B0-schedule); the
  compute-ceiling *correctness* is unaffected.

## Known risks
- `harness_kernels.cpp` has NOT been compiled by this agent (no node access this session) →
  `[NEEDS-NODE]` compile verification. The MMA tile-fragment shapes follow v2/v3 conventions but the
  main agent should run the compile step before the first GPU run.
- B0 grid assumes M,N multiples of 256 for full tiles; tail M (e.g. M=8) is padded by the GEMM's tile
  coverage. For very small M (<256) B0/B1 the row count rounds up to one 256-tile — latency at tiny M
  is tile-quantized (documented; expected, the interesting regime is M≥256 anyway).
- B2 is a stub by design (see methodology "B2 wiring gap"): production MORI + AITER/CK are not
  importable from the repo; the main agent must wire them on the node.
- One `tk_kernel` per process → B3 and B4/B5 must run in their respective build dirs (handled by the
  proposed commands; `KERNEL_VARIANT` only labels the CSV, it does not pick the .so).

## Static info / [NEEDS-NODE]
All resource columns and all timings: `[NEEDS-NODE]` (require node compile / GPU run).
No secrets in any committed file (placeholders only); secret-scanned before commit.
