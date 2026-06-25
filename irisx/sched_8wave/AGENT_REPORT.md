# AGENT 04 REPORT — sched_8wave (true HK 8-wave ping-pong role-swap)

## Summary
Transplants the canonical HipKittens 8-wave ping-pong schedule
(`<HK_ROOT>/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu`) onto the IRIS remote-FP8 MoE expert-GEMM.
TRUE role-swap (both wavegroups feed AND compute, swapping every `s_barrier()`), NOT V4's
permanent producer/consumer split. Disjoint output-row ownership avoids any partial-K reduction.
Single-expert; meant to be transplanted into Agent 02's grouped scheduler later.

## Files changed (all under `irisx/sched_8wave/`)
- `kernel.cpp` — ping-pong `micro_tk` + V4-style two-phase `micro_tk_baseline` (apples-to-apples),
  same metadata ABI / pybind entry `dispatch_micro` as V4.
- `example.py` — np=2 driver scaffold (rank0 holds fp8 A + scales; rank1 runs the kernel),
  RMS-rel correctness + zero-sentinel + head-to-head timing. Default K=7168.
- `PINGPONG_8WAVE.md` — schedule diagram, barrier state machine, output-ownership proof.
- `AGENT_REPORT.md` — this file.

## Exact build command (main agent, on node)
Mirror the candidate dir into the node HK tree, then build with the COMPILE lock and low -j:
```
# 1. copy repo dir -> node build tree (build system auto-discovers */kernel.cpp)
#    <repo>/irisx/sched_8wave  ->  <HK_ROOT>/distributed-kernels/sched_8wave
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""   # compile-only, no device
  cd <HK_ROOT>/distributed-kernels
  flock /tmp/mi355x_compile.lock -c "
    cmake -B build_04 -DDK_BUILD=sched_8wave -DIRIS_HIP_ARCHITECTURES=gfx950 &&
    cmake --build build_04 -j8 --target sched_8wave"'
```
Static resource report (compile-only, also under the lock):
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels/sched_8wave
  flock /tmp/mi355x_compile.lock -c "hipcc --offload-arch=gfx950 \
    -Rpass-analysis=kernel-resource-usage -c kernel.cpp -o /tmp/sched8_04.o 2>&1 | tee /tmp/sched8_04.res"
  # then: llvm-objdump -d --mcpu=gfx950 /tmp/sched8_04.o | head, roc-obj/readelf for vgpr/sgpr/lds/scratch'
```

## Exact PROPOSED GPU test command (MAIN AGENT ONLY — never the subagent)
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/sched_8wave
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    M=1024 K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"'
```
`--mca pml ob1 --mca btl self,vader` is MANDATORY (R1 server holds VRAM; IB reg fails otherwise).

### Sweep plan
- M in {8,16,32,64,128,256,512,1024}  (per-expert rows M_e)
- N in {2048, 4096}                    (W13 gate/up panel; 4096 = both fused)
- K = 7168 (fixed, R1 W13 input)
- For each shape run BOTH fused=1 (ping-pong) and fused=0 (baseline) — the harness does both.
- Config sweep (recompile per config via -DBM/-DBN/-DBK/-DNSUB/-DNSTAGE): default BM=BN=BK=64,
  NSUB=4, NSTAGE=2 first; then NSUB in {2,8}, NSTAGE=3, to find the occupancy sweet spot.

## Expected output
Per shape, two lines like:
```
[PINGPONG] M=1024 K=7168 N=2048 ... RMS_rel=0.003xx local_A_zero=True C_zero=False -> PASSED
[BASELINE] M=1024 K=7168 N=2048 ... RMS_rel=0.003xx local_A_zero=True C_zero=False -> PASSED
------------------------------------------------------------------------------
  BASELINE (two-phase, no overlap) :  ~1200 us/iter   ~xx TFLOP/s
  PINGPONG (8-wave role-swap)      :   <baseline       >xx TFLOP/s
  SPEEDUP (baseline/pingpong)      :   >1.0x  (PINGPONG WINS)   [expected for M>=512]
```

## Correctness criteria
- `RMS_rel < 0.10` (expect ~0.003, matching V4's 0.00331 — same gather/dequant path).
- `local_A_zero == True` AND `C_zero == False`: zero-sentinel proves C was produced from the
  REMOTE rank-0 A (rank-1's local A is all zeros), i.e. the IRIS gather really ran.
- PINGPONG and BASELINE must agree (both PASSED) on every shape.

## Static resource table (VGPR/AGPR/SGPR/LDS/scratch)
**[NEEDS-NODE — main agent to compile]**. SSH to `<NODE>` was gated at authoring time
(Conductor "Permission denied (publickey)" / no active reservation), so no static extraction was
possible. Fill from `/tmp/sched8_04.res` (`-Rpass-analysis=kernel-resource-usage`) once built:

| config (BM/BN/BK/NSUB/NSTAGE) | VGPR | AGPR | SGPR | LDS bytes | scratch | occ (waves/SIMD) |
|---|---|---|---|---|---|---|
| 64/64/64/4/2 (default) | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | ~`NSTAGE*(2*sizeof(ST_A)+NSUB*sizeof(ST_B))+1024` | [NEEDS-NODE] | [NEEDS-NODE] |
| 64/64/64/8/2 | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] |
| 64/64/64/2/3 | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] |
| 32/64/64/4/2 | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] |

LDS estimate (computable statically): default 64/64/64/4/2 with bf16 ST_A(32x64) and ST_B(64x64):
`NSTAGE*(WARPS_ROW*sizeof(ST_A) + NSUB*sizeof(ST_B)) = 2*(2*32*64*2 + 4*64*64*2) = 2*(8192+32768)
= 81920 B (+1024)` ~= 82.9 KB/block. With `__launch_bounds__(512,2)` requesting occupancy 2,
this must fit twice in the 160 KB LDS budget (82.9*2 = 165.8 KB > 160 KB) — so occupancy 2 may NOT
be reachable at NSUB=4/NSTAGE=2; the kernel requests it but the compiler/runtime may clamp to 1.
**This is a key thing to verify on node** and a reason to also try NSUB=2.

## Assumptions
- HK `kittens.cuh` on the node exposes `st_bf`, `rt_fl`, `rt_bf`, `group<N>::load`/
  `prefill_swizzled_offsets`, `subtile_inplace`, `load`, `mma_ABt`, `st_16x32_s`,
  `rt_16x16_s`/`rt_16x32_s`, `__hip_cvt_fp8_to_halfraw(__HIP_E4M3)` — same API V3/V4 use.
- `iris::iris_device_view::load(const T*, int rank)` performs the remote gather (as in V4).
- B (weights) is local-HBM bf16; A is remote fp8 e4m3 OCP (gfx950) with per-128 fp32 scales.
- Same symmetric-heap allocation-order contract as V4's example.py.
- bf16xbf16 MFMA path (A dequantized in the gather). An fp8xfp8 variant (skip dequant, scale in
  epilogue) is a future optimization, NOT in this candidate.

## Known risks
1. **Occupancy 2 may be unreachable** at NSUB=4/NSTAGE=2 due to ~83 KB LDS/block (see table note).
   Mitigation: NSUB=2, or NSTAGE=2 with smaller BN. Verify on node before claiming a win.
2. **Barrier balance**: relies on the prologue (WG1-only) + epilogue (WG0-only) conditional
   barriers exactly mirroring; and on EVERY loop `s_barrier()` being unconditional. A future edit
   that conditionally skips a loop barrier would hang the workgroup.
3. **Half-phase seed correctness**: the ping-pong assumes the HK barrier-offset trick behaves the
   same under this NSUB-subtiled inner loop as in the canonical single-accumulator-set source.
   The first on-node run must confirm no deadlock and correct numerics (RMS-rel ~0.003).
4. **A-half gather underutilization** at BM=32 (HALF_BM=16). Default BM=64 avoids it.
5. **Small M**: like V4, expect ping-pong to LOSE at M<=256 (grid starves the 256-CU array); the
   win is in the batched regime M>=512. Do not over-claim; honest comparison is vs Agent 01's B1
   (gather-once-then-local-GEMM), not just this in-file two-phase baseline.

## Compile status
**[NEEDS-NODE]** — not compiled. SSH gated (no active Conductor reservation: `Permission denied
(publickey)`). Per AGENT_COMMON rule, did not block. No GPU touched. Main agent: build with the
command above under `flock /tmp/mi355x_compile.lock`, build dir `build_04`, `-j8`,
`HIP_VISIBLE_DEVICES=""`, then fill the resource table from `-Rpass-analysis=kernel-resource-usage`.
```
