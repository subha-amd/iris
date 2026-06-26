# Agent 11 — COPY-ONCE communication/compute overlap (P1, P2)

New main research direction after Gate-1 refuted the fused V4 (V4 = 0.43x of B1-copy = 2.33x SLOWER,
because it refetches A ~4x and runs the GEMM at ~44 TFLOP/s / occ 1-2). Both candidates here move
each A tile EXACTLY ONCE and overlap that movement with the VERBATIM B0 GEMM (256x256x64, 8 warps,
`__launch_bounds__(512,2)`), so they keep B0-class compute efficiency while hiding the copy.

Bar (EXPERIMENT_LEDGER, M1024/N2048/K7168): B0 ~164us, B1-copy ~291us, perfect-overlap floor ~162us,
ceiling speedup over B1 ~1.80x. A candidate is interesting only if T_pipeline is meaningfully < 291us
toward ~162-200us.

I did NOT touch the GPU (AGENT_COMMON rule). All files are compile-ready; the MAIN AGENT compiles,
runs np=2 under `flock /tmp/mi355x_project_gpu.lock`, and commits.

---

## Files written

```
irisx/p1_tile_inbox/kernel.cpp            P1 producer (copy-once) + consumer (= B0 GEMM)
irisx/p1_tile_inbox/tile_inbox_abi.h      flag/claim encoding (self-contained subset of Agent 06)
irisx/p1_tile_inbox/example.py            np=2 driver: T_pipeline + RMS-rel + zero-sentinel + CSV
irisx/p1_tile_inbox/P1_DESIGN.md          protocol + deadlock argument + metrics
irisx/p2_expert_pipeline/kernel.cpp       P2 producer (expert double-buffer) + consumer (= B0 grouped)
irisx/p2_expert_pipeline/expert_pipeline_abi.h   flag encoding + ep8_gather_BM
irisx/p2_expert_pipeline/ep8_gather.h     trimmed self-contained route_segment ABI (subset of Agent 03)
irisx/p2_expert_pipeline/example.py       np=2 grouped driver: route build + T_pipeline + per-expert check
irisx/p2_expert_pipeline/P2_DESIGN.md     scheme + deadlock argument + P1-vs-P2 + metrics
AGENT_REPORT.md                           this file
```

Both `PYBIND11_MODULE` are named `tk_kernel` and both `example.py` `import tk_kernel`, per the build
contract (auto-discovers `*/kernel.cpp`, forces module name `tk_kernel`).

---

## P1 — build

Mirror `irisx/p1_tile_inbox/` to `<HK_ROOT>/distributed-kernels/p1_tile_inbox/` (keep node + repo
copies identical), then on `<NODE>` inside the ATOM container `r1_c4`:

```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""   # compile-only safety
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_11 -DDK_BUILD=p1_tile_inbox -DIRIS_HIP_ARCHITECTURES=gfx950
  flock /tmp/mi355x_compile.lock -c "cmake --build build_11 -j8 --target tk_kernel"'
```

## P1 — proposed np=2 GPU run (MAIN AGENT only, under the GPU lock)

```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/p1_tile_inbox
  flock /tmp/mi355x_project_gpu.lock -c "
    M=1024 N=2048 K=7168 ITERS=50 WARMUP=10 PROD_BLOCKS=16 \
    mpirun -np 2 --mca pml ob1 --mca btl self,vader python example.py"'
```
Sweep `PROD_BLOCKS` in {8,16,32} (producer CU reservation vs throughput). `BM_PROD` defaults to 64
and MUST match kernel.cpp.

## P1 — expected output / correctness
```
[P1      ] M=1024 K=7168 N=2048  max_rel=...  RMS_rel~0.00331  local_A_zero=True  C_zero=False -> PASSED
  P1 COPY-ONCE TILE INBOX (B0 consumer, overlap):  XXX.XX us/iter   YYY.YY TFLOP/s
  T_pipeline=XXX.Xus  vs B0=164.2us vs B1=291.4us  spd_vs_B1=...x  overlap_vs_floor(162us)=...
CSV,P1,...
```
Correctness criteria (all must hold): `RMS_rel < 0.10` (expect ~0.00331, same as B0/B1),
`local_A_zero=True` (consumer rank has no local A -> proves remote gather), `C_zero=False`.
Interesting if `T_pipeline << 291us` (toward 162-200us); record the number regardless (negative
results count).

---

## P2 — build

Mirror `irisx/p2_expert_pipeline/` to `<HK_ROOT>/distributed-kernels/p2_expert_pipeline/`, then:
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_11 -DDK_BUILD=p2_expert_pipeline -DIRIS_HIP_ARCHITECTURES=gfx950
  flock /tmp/mi355x_compile.lock -c "cmake --build build_11 -j8 --target tk_kernel"'
```

## P2 — proposed np=2 GPU run
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/p2_expert_pipeline
  flock /tmp/mi355x_project_gpu.lock -c "
    E=32 N=2048 K=7168 TOTAL_M=8192 ROUTE=uniform ITERS=50 WARMUP=10 PROD_BLOCKS=8 \
    mpirun -np 2 --mca pml ob1 --mca btl self,vader python example.py"'
```
Then sweep `ROUTE` in {uniform, zipf, one_hot, many_empty} to exercise variable M_e / empty experts.
`GBM` (ep8_gather_BM) defaults 64 and MUST match kernel.cpp.

## P2 — expected output / correctness
```
[P2      ] E=32 TOTAL_M=8192 K=7168 N=2048  max_rel=...  RMS_rel~0.003  local_A_zero=True  C_zero=False -> PASSED
  P2 EXPERT DOUBLE-BUFFER (B0 consumer, overlap):  XXX.XX us/iter   YYY.YY TFLOP/s (routed-rows)
CSV,P2,...
```
Correctness: per-expert `C[rows_e] = dequant(A_e) @ B_e^T`, aggregate `RMS_rel < 0.10`,
`local_A_zero=True`, `C_zero=False`. Empty experts contribute no rows (skipped in the check).

---

## Assumptions
- HK pybind (`pyutils from_object`) accepts int32 `gl` tensors for the flag/cursor arrays (proven by
  Agent 06's cache_first_touch and v5_grouped, which bind the same way).
- The proven B0 GEMM in `harness/harness_kernels.cpp::b0_gemm` is the compute ceiling; I copied its
  tiling/occupancy verbatim into both consumers. If the main agent swaps in the exact v2 8-wave
  ping-pong for peak TFLOP/s, replace the consumer inner loop identically in both files.
- IRIS symmetric heap: identical allocation ORDER on every rank -> identical offsets (the drivers do
  this). np=2 single-source for P2 bring-up; np=8 multi-source reuses the SAME kernel via
  `route_segment[]` spanning ranks.
- `INBOX[M,K]` bf16 (~14.7 MB at M1024) fits the 1024 MB P1 heap; P2's two `slot_rows x K` slots fit
  the 2048 MB heap.

## Known risks
- **Producer dequant throughput (P1/P2):** the producer's scalar fp8->bf16 loop may be slower than
  the parallel `dequant_a_dense`. If T_copy >> 162us the producer is the bottleneck -> raise
  PROD_BLOCKS / vectorize the dequant. Gather traffic itself is the same uint4 as B1.
- **PROD_BLOCKS balance:** too large competes with consumer CUs; too small starves the consumer.
- **One-expert-ahead overlap (P2):** one_hot / dominant-expert routes collapse overlap toward serial
  (a big expert's gather can't hide behind a tiny neighbor). Expected; report it.
- **slot_rows blow-up (P2):** one_hot makes slot_rows = Mpacked -> 2 large slots. Acceptable at the
  configured heap size.
- **ABI matching:** `BM_PROD` (P1) and `ep8_gather_BM`/`GBM` (P2) must match between kernel.cpp and
  example.py. `arrive[e]`/`done[]` MUST be reset+primed each generation (drivers do this).

## Deadlock-avoidance (summary; full args in P1_DESIGN.md / P2_DESIGN.md)
- BOTH: two SEPARATE launches on two non-blocking streams, producer FIRST (reserves CUs); NO
  hipStreamSynchronize between the launches; symmetric `iris.barrier()` counts on ALL ranks (only the
  consumer rank launches kernels) so an asymmetric barrier count can't deadlock (the bug that bit
  Agent 01).
- P1: cursor hands out every band once + CAS makes exactly one producer materialize/signal each;
  consumers wait only on bands, never on each other -> no cycle.
- P2: strict depth-2 slot queue; producer waits done[e], consumer waits ready[e], last consumer tile
  of e frees done[e+2]; host primes done[0],done[1] and pre-sets done[e+2] for empty experts ->
  bounded, cycle-free.

## Did NOT compile
Per AGENT_COMMON, design/codegen agents may compile-only but I have no Bash/ssh in this environment
(Read/Write/Edit/Glob/Grep only). The MAIN AGENT should compile (build_11, -j8, under the compile
lock) and run. Files are written to be drop-in for the auto-discovered `tk_kernel` build.
```
