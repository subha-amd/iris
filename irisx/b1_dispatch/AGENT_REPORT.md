# AGENT 13 — B1-dispatch V0 — REPORT

B1-dispatch V0 = the production-shaped main-line MoE expert kernel, built by COMPOSITION of two
already-verified components (NOT a rewrite). Serial, route-aware EP8: PHASE 1 multi-source
gather/pack/quant ONCE -> LOCAL expert-major packed fp8+scale buffer; PHASE 2 the verified
v5_grouped serial grouped GEMM over that local buffer. A crosses XGMI exactly once (phase 1).

NO GPU was touched by this agent (per AGENT_COMMON §0). All on-device numbers below are the
EXPECTED criteria; the main agent runs + records.

## Files (new candidate dir `irisx/b1_dispatch/`)
- `kernel.cpp`            — phase-1 `gather_pack_kernel` (NEW glue) + phase-2 `micro_tk_baseline`
                            (VERBATIM from v5_grouped) + 2 pybind fns (`dispatch_gather_pack`,
                            `grouped_gemm`). Module name `tk_kernel`.
- `ep8_gather.h`          — VERIFIED multi-source row resolver (COPY of ep8_gather/ep8_gather.h,
                            unchanged), so the TU is self-contained on the node.
- `b1_dispatch_route.py`  — NEW glue: 32-expert MULTI-SOURCE routing over the v5 BM-padded packed
                            layout + per-BM-tile metadata + quantize_v1 (e4m3fn). Pure CPU.
- `build_tasks.py`        — VERBATIM COPY of v5_grouped/build_tasks.py (task list + adaptive NSUB).
- `example.py`            — np=8 driver: phase1 -> phase2, CPU grouped reference, zero-sentinel,
                            cuda-event T_gather/T_gemm/T_total split, results.csv row.
- `B1_DISPATCH.md`        — design + composition map.
- `AGENT_REPORT.md`       — this file.

## EXACTLY what is copied VERBATIM vs new (the #1 rule)
COPIED VERBATIM (do NOT re-derive):
- Phase-2 GEMM: the ENTIRE block in kernel.cpp from `static constexpr int TASK_W = 6;` through
  `void dispatch_grouped_gemm(...)` is a verbatim copy of `v5_grouped/kernel.cpp` — `micro_globals`,
  `fp8_to_f32`, `gather_dequant_A_tile`, `micro_tk_baseline`, tile/warp config. (v5's fused
  `micro_tk` is intentionally NOT copied; V0 uses only the serial baseline path.)
- Phase-1 row resolution: `seg_tile_view`, `tile_is_single_source`, `build_row_seg_map`,
  `route_segment`, `SEG_NONE` — included from the unchanged `ep8_gather.h`; the per-row
  (src_rank,src_row) resolution inside `gather_pack_kernel` is lifted verbatim from
  `ep8_gather.h::gather_dequant_A_tile_multisource` (the fast/Path-1 branch logic).
- Phase-1 data movement: the uint4 fp8-byte remote-load-then-local-store + scalar fp32 scale load is
  the `harness_kernels.cpp::gather_once_kernel` body (B1-copy), generalized to multi-source.
- `build_tasks.py` and `quantize_v1` (e4m3fn) are verbatim from v5_grouped / ep8 ref.

NEW glue (the only new logic):
- `gather_pack_kernel` driver loop (walk packed BM-tiles -> resolve -> raw copy -> zero-sentinel).
- `b1_dispatch_route.build_multisource_route` (32-expert multi-source segments over the packed layout).
- the 2 pybind wrappers + example.py driver + the cuda-event timing split.

## Exact build command (MAIN AGENT, on node)
Mirror this dir to `<HK_ROOT>/distributed-kernels/b1_dispatch/` (identical contents — includes
`ep8_gather.h` and `build_tasks.py` so the TU + driver are self-contained), then:
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_13 -DDK_BUILD=b1_dispatch
  flock /tmp/mi355x_compile.lock -c "cmake --build build_13 -j8 --target b1_dispatch"'
```
Compile-only resource report (no GPU):
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES=""; export ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels/b1_dispatch
  flock /tmp/mi355x_compile.lock -c "hipcc --offload-arch=gfx950 \
    -Rpass-analysis=kernel-resource-usage -c kernel.cpp -I.. <HK include flags> 2>&1 | tee resource.txt"'
```

## Pure-CPU host self-tests (safe anywhere, NO GPU)
```
python3 build_tasks.py          # -> "ALL INVARIANTS PASSED"
python3 b1_dispatch_route.py    # -> per-route OK lines + "ALL INVARIANTS PASSED"
```
Run these FIRST — they validate the routing glue (disjoint+sorted segments, runs inside each
expert's real rows, every real row routed once, both gather paths exercised) with no GPU.

## Exact PROPOSED np=8 GPU test command (MAIN AGENT only, under the GPU lock)
START HERE (uniform route, medium size):
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/b1_dispatch
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    ROUTE=uniform TOTAL_M=8192 E=32 K=7168 N=2048 MSRC=4096 \
      mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 8 python3 example.py"'
```
Then sweep (one at a time, same flock): `ROUTE` in {uniform, zipf, one_hot, several_hot, many_empty}.
For larger `TOTAL_M` (e.g. 32768) raise `MSRC` so no source rank is exhausted (need
`MSRC * 8 >= TOTAL_M` with headroom; the driver raises a clear error if a rank runs out).

## Expected output (per run)
```
[b1-dispatch] route=uniform E=32 TOTAL_M=8192 -> Mpacked=8192 NSUB=8 num_tasks=... segs=... tiles=128 single_src_tiles=... multi_src_tiles=... rows[0:6]=[...]
============================================================================================
[B1-dispatch V0] route=uniform Mpacked=8192 N=2048 K=7168 NSUB=8 tasks=... segs=...
  remote (XGMI) gathered rows = <large, >0>
  RMS_rel=0.00...  max_rel=...
  packed_A_nonzero=True  C_zero=False  remote_path=True
  -> PASSED
--------------------------------------------------------------------------------------------
  T_gather (phase1, multi-source gather/pack ONCE) : ...... us
  T_gemm   (phase2, local grouped GEMM)            : ...... us   ..... TFLOP/s
  T_total  (serial phase1 + phase2)                : ...... us   ..... TFLOP/s (e2e)
  (compare to B1-copy: copy 135us + gemm 150us = 285us @ M1024 single-source)
============================================================================================
```
Appends a `B1-dispatch` row to results.csv (canonical columns; T_gather/T_gemm in notes).

## Correctness criteria
- `RMS_rel < 0.01` vs the numpy-CPU grouped reference (v5 measured 0.00331; same dequant + GEMM).
- `packed_A_nonzero=True` and `C_zero=False` — phase 1 moved real bytes; phase 2 wrote output.
- zero-sentinel: unrouted packed rows give EXACTLY-zero C rows (no cross-expert contamination).
- `remote (XGMI) gathered rows > 0` — the real multi-source path was exercised (not all-local).
- Both CPU self-tests print PASSED.

## Assumptions
- ABI = AGENT_COMMON §3 / ep8_gather.h (`route_segment`, `seg_tile_view`); v5's 6-int task tuple.
- B is expert-major `[E*N, K]`, bf16; output bf16; fp8 e4m3fn (OCP, gfx950); fp32 per-128 scales.
- Phase 1 packs token-major scales `[Mpacked,NG]` to match phase 2's reader; production's group-major
  scale transpose (FMOE_LAYOUT.md §5) is a LATER concern, not V0.
- CONSUMER rank (default 7) packs locally + computes; src_rank passed to phase 2 == CONSUMER so the
  GEMM's ctx.load is a local deref (A does not re-cross XGMI). Any rank may be the consumer.
- Compile-time NSUB=8 is the max the host requests; runtime nsub in {8,4,2,1} <= that.

## Known risks
1. **Phase-1 grid granularity**: one block per BM-tile (Ntile~128 blocks for Mpacked=8192), 256
   threads each — coarser than B1-copy's one-block-per-row. If phase-1 latency is high, the obvious
   tune is more blocks (split a tile across blocks by K) — but the gather is correct regardless.
2. **Scale idempotent race**: many 16-byte chunks in a 128-group write that group's scale with the
   SAME value; benign (no barrier needed), but noted.
3. **MSRC sizing**: each source rank needs enough rows for the runs assigned to it. Default MSRC=4096
   is ample for TOTAL_M<=8192 (avg ~1024/rank); for TOTAL_M=32768 raise MSRC. The driver raises a
   clear error if a rank is exhausted (not a silent wrong answer).
4. **B tile indexing with expert-major B** is v5's known first-compile watch item (AGENT_REPORT v5
   risk #2); inherited verbatim, so if v5 compiled+passed, this does too.
5. **iris.empty int32**: avoided (SEG/TILE are float32-alloc + int32 view); TASKS is a LOCAL torch
   int32 tensor (not on the heap).
6. **Barrier symmetry**: every rank runs the timing/loop bodies and calls iris.barrier() the same
   number of times; only recording/printing is gated to the consumer (avoids the documented deadlock).

## Static resource info
NOT compiled by this agent (no GPU/compile per the rule; design-by-composition). Phase 2 is a verbatim
copy of v5's `micro_tk_baseline`, so its resources == v5 baseline. Phase 1 (`gather_pack_kernel`) is a
small copy kernel (uint4 loads + scalar scale stores, one __shared__ char[64]) — expected low VGPR,
no scratch spill. MAIN AGENT to confirm with `-Rpass-analysis=kernel-resource-usage`.

## Secret scan
No tokens, hostnames, or real paths in any committed file (placeholders `<NODE>`/`<HK_ROOT>`/`r1_c4`
only). `NODE_ACCESS.local.md` (gitignored) is not referenced.

## TL;DR for the MAIN AGENT
1. `python3 build_tasks.py` and `python3 b1_dispatch_route.py` -> both PASSED (no GPU).
2. Build with `build_13` (command above).
3. `ROUTE=uniform TOTAL_M=8192 E=32 K=7168 N=2048 MSRC=4096` np=8 run -> expect PASSED
   (RMS_rel<0.01, packed_A_nonzero, C not zero, remote rows>0) and the T_gather/T_gemm/T_total split.
4. Sweep the 5 routes; compare T_total to B1-copy (285us @ M1024) and to v5_grouped's fused/baseline.
