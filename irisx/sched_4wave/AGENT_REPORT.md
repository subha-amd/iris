# AGENT_REPORT — Agent 05 (sched_4wave, symmetric 4-wave latency path)

Candidate: `irisx/sched_4wave/` (kernel.cpp + example.py + INTERLEAVE_4WAVE.md).
Goal: fix V4's small-M (M<=256) loss with a symmetric, no-producer/consumer 4-wave latency kernel.

## Exact build command (node, gfx950 — MAIN AGENT or a node-enabled run)

Mirror this dir under the node's auto-discovered build tree, then build with the compile lock:
```
# on node, inside container r1_c4
export HIP_VISIBLE_DEVICES="" ; export ROCR_VISIBLE_DEVICES=""
mkdir -p <HK_ROOT>/distributed-kernels/sched_4wave
cp irisx/sched_4wave/kernel.cpp   <HK_ROOT>/distributed-kernels/sched_4wave/
cp irisx/sched_4wave/example.py   <HK_ROOT>/distributed-kernels/sched_4wave/
cd <HK_ROOT>/distributed-kernels
cmake -B build_05 -DIRIS_HIP_ARCHITECTURES=gfx950 -DDK_BUILD=sched_4wave
flock /tmp/mi355x_compile.lock -c "cmake --build build_05 -j8 --target sched_4wave"
```
Static resource report (compile-only, allowed for subagents):
```
flock /tmp/mi355x_compile.lock -c "hipcc --offload-arch=gfx950 \
  -Rpass-analysis=kernel-resource-usage -c kernel.cpp -o /tmp/sched_4wave.o \
  -I<HK include paths from the cmake build>"
# then: roc-obj / llvm-objdump -d --mcpu=gfx950 /tmp/sched_4wave.o  for VGPR/AGPR/LDS/scratch
```

## Compile status

**[NEEDS-NODE]** — node SSH was gated at run time (Conductor `Permission denied (publickey)` =
no active reservation in the window). One probe connected; the container exec was denied. Could not
compile or extract static VGPR/AGPR/scratch/LDS. Build command above is ready for the main agent.

A single-TU `hipcc -fsyntax-only` is the cheapest syntax gate if the node frees up.

## Proposed GPU test commands (MAIN AGENT ONLY, serialized under flock)

Small-M sweep is the priority (the regime V4 loses). Defaults BM=32 BN=64 BK=64 NSUB=2.
```
cd <HK_ROOT>/distributed-kernels/sched_4wave
source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64

# (a) Correctness + head-to-head at the priority small-M points (default tile config):
for MM in 8 16 32 64 128 256; do
  flock /tmp/mi355x_project_gpu.lock -c "
    M=$MM K=2048 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"
done

# (b) Tile/NSUB sweep at a representative small M (recompile per config; -D overrides):
#     BM={16,32} BN={32,64} BK={32,64} NSUB={1,2,4}
#   e.g. rebuild with: cmake --build build_05 -j8 --target sched_4wave \
#        then run with M in {32,64,128}. (Pass -DBM=.. etc. via the kernel's #ifndef macros at
#        compile time; one rebuild per config under the compile lock.)

# (c) Production K (R1 W13): K=7168, N=2048, small M:
for MM in 16 64 256; do
  flock /tmp/mi355x_project_gpu.lock -c "
    M=$MM K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"
done
```

## Expected output (per run)

```
==============================================================================
[FUSED   ] M=64 K=2048 N=2048  max_abs=..  max_rel=..  RMS_rel=0.0033x  local_A_zero=True  C_zero=False  -> PASSED
[BASELINE] M=64 K=2048 N=2048  ...  RMS_rel=0.0033x  local_A_zero=True  C_zero=False  -> PASSED
------------------------------------------------------------------------------
  BASELINE (two-phase, no overlap) :   xxx.xx us/iter   yy.yy TFLOP/s
  FUSED    (4-wave interleave)     :   xxx.xx us/iter   yy.yy TFLOP/s
  SPEEDUP (baseline/fused)         :   1.xxx   (FUSED WINS)
==============================================================================
```

## Correctness criteria

- `RMS_rel < 0.10` for BOTH fused and baseline (expect ~0.003, matching V4).
- `local_A_zero=True` (non-source rank's local A is zero) AND `C_zero=False` -> zero-sentinel proves
  the result came from the cross-GPU IRIS gather, not stale local data.
- Fused and baseline RMS-rel must agree (same numerics, only schedule differs).

## Success criteria for THIS kernel (the point of the agent)

- At M<=256 the FUSED 4-wave should be **>= 1.0x** vs its baseline (V4 was 0.85-0.90x here).
- Ideally fused TFLOP/s at small M improves over re-running V4's fused path at the same M.
- Compare honestly against Agent 01's B1 (gather-once-then-local-GEMM), not just this in-file baseline.

## Assumptions

- HK `gl` rejects 1-byte element types, so fp8 A is carried as `bf16 [M,K/2]` and reinterpreted as
  fp8 bytes in-kernel (same trick as V3/V4 example.py).
- `BM % NUM_WORKERS == 0` (CONS_M integer). Defaults satisfy this (32/4=8). Sweep keeps BM in
  {16,32} so CONS_M in {4,8}.
- `subtile_inplace<CONS_M,BK>(As[cur], {warp_id,0})` slices the wave's CONS_M-row strip — assumes
  CONS_M is a valid HK subtile row count for `st_16x32_s` (CONS_M>=... ; if CONS_M=4 hits an HK
  subtile-granularity limit, fall back to BM=32/NSUB or use a per-wave A LDS tile of CONS_M rows).
- Store convention `{0,0,out_row0/CONS_M, out_col0/BN}` (accumulator-tile units) — confirmed against
  V4's `store(g.c, accum, {0,0,row_tile,col_tile})`.
- `load_B_subtile` is a straightforward per-lane swizzled copy; if the HK `PG::load` fast path is
  preferred for B, it can replace it (B is local HBM, not the bottleneck).

## Known risks

1. **HK subtile granularity at CONS_M=4/8.** `subtile_inplace<CONS_M,BK>` and `rt_bf<CONS_M,..>`
   may require CONS_M to be a multiple of the 16x32 subtile rows. If CONS_M=8 is too small for the
   MFMA fragment shape, options: (i) keep BM=32, NUM_WORKERS=4 but have each wave MFMA a 16-row
   fragment covering 2 of its logical rows via masking; (ii) use BM=64 with CONS_M=16. NEEDS-NODE
   compile to confirm — flagged as the top risk.
2. **NEEDS-NODE: no static resource numbers.** The waves/SIMD=4 occupancy claim is a design target,
   unverified. Must extract VGPR/AGPR/LDS/scratch once the node frees up.
3. At very small M (8,16) the grid may be a single block-row; ensure tail masking (gr<M) is correct
   — it is handled in `gather_dequant_A_strip` (zero-fill out-of-range).
4. `load_B_subtile` per-lane scalar path may be slower than the HK vectorized loader; B is local so
   this is latency not bandwidth-critical, but worth swapping to `PG::load` if B-load shows up hot.

## Files changed (this branch / candidate dir)

```
irisx/sched_4wave/kernel.cpp           # symmetric 4-wave fused + matched two-phase baseline
irisx/sched_4wave/example.py           # np=2 head-to-head driver (small-M default)
irisx/sched_4wave/INTERLEAVE_4WAVE.md  # per-wave pipeline + register-pressure explanation
irisx/sched_4wave/AGENT_REPORT.md      # this file
```

## Static resource table

| candidate            | VGPR | AGPR | scratch | LDS                                   | waves/SIMD (target) | output tile        | remote loads / output elt |
|----------------------|------|------|---------|---------------------------------------|---------------------|--------------------|---------------------------|
| sched_4wave (default) | [NEEDS-NODE] | [NEEDS-NODE] | [NEEDS-NODE] | NSTAGE*(BM*BK + NSUB*BN*BK)*2B + 1KB = 2*(32*64 + 2*64*64)*2 + 1024 ≈ 49 KB | 4 (design target) | BM x N_PER_BLOCK = 32 x 128 (per wave: CONS_M=8 x NSUB*BN=128) | A gathered once per K-tile, reused NSUB; => K/BK * (1/NSUB) remote A-elt loads per output elt |

(LDS figure is the static `dynamic_shared_memory()` for defaults; VGPR/AGPR/scratch require the
node compile.)
