# V2 HipKittens fused-MoE expert GEMM — bring-up results (MI355X, gfx950, 2026-06-25)

V2 bring-up = the simplest **correct** single-GPU fused-MoE expert **gate/up GEMM** written in
HipKittens that consumes our IRISX MoE-dispatch FP8 buffer **directly** (expert-major packed
layout — our design, not production's row-major+sort contract). This is the bring-up step only:
single-GPU, single-process, local buffers. No IRIS remote gather, no comm/compute overlap, no
down-projection. Follows `V2_HK_ANALYSIS.md` (option (b) preamble-dequant + bf16 MMA core;
option (i) launch-per-expert; padded-M + send_counts masking).

## What it computes
One expert's gate/up projection:
```
A = packed_fp8[local_e]  viewed as [M_padded, H]   fp8 e4m3 + packed_sc[M_padded, N_GROUPS] fp32 per-128 scale
B = synthesized weight [N=2048, H=7168] bf16
C = A_dequant . B^T  -> [M_padded, 2048] bf16     (mma_ABt contract: C[M,N] += A[M,K]·B[N,K]^T)
M_padded = world*PER_SRC_CAPACITY = 8*64 = 512 ;  K = H = 7168 ;  N_GROUPS = 56
```

## Files created (all under kernels/, plus this doc)
- `<HK_ROOT>/kernels/fmoe_expert_v2/fmoe_expert_v2.cu` — dequant preamble kernel
  + bf16 GEMM core + host harness (synth buffer, CPU fp32 reference, timing, masking check).
- `<HK_ROOT>/kernels/fmoe_expert_v2/Makefile` — standalone build (mirrors FP8_8wave).
- `<HK_ROOT>/V2_RESULTS.md` — this file.

## Build (verified, gfx950 / CDNA4 default)
```
cd <HK_ROOT>/kernels/fmoe_expert_v2
THUNDERKITTENS_ROOT=<HK_ROOT> make
# -> /opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
#       -I<HK_ROOT>/include -I/opt/rocm/include/hip -c fmoe_expert_v2.cu ...
# -> BUILD=0 ; produced ELF "fmoe_v2"
```
(The `No module named pybind11` line is benign — standalone build, not pyext.)

## Run (verified, single-GPU)
```
cd <HK_ROOT>/kernels/fmoe_expert_v2
./fmoe_v2     # RUN_EXIT=0
```
Output:
```
V2 fmoe expert gate/up GEMM (bring-up)
  M_PADDED=512 (world=8 x PER_SRC_CAPACITY=64), N=2048, K=H=7168, N_GROUPS=56
  GEMM: 0.1194 ms/iter, 125.88 TFLOP/s (over padded M=512)

=== CORRECTNESS (vs CPU fp32 dequant reference) ===
  active rows checked: 268, samples: 15008, output RMS scale: 8.6405
  RMS-relative error ||C-ref||/||ref|| = 0.00368
  max_abs_err = 0.19414, max_err/out_rms = 0.02247
  MASKING: padded(masked) rows with nonzero output = 0 (expect 0)

  RESULT: PASSED
```

## Correctness
Reference: CPU, fp32 accumulation, dequantizing the **same** fp8+per-128-fp32-scale buffer the
device dequant uses (bit-identical dequant: `float(e4m3) * scale`), times the same bf16 weight.
- **RMS-relative error ||C−ref||/||ref|| = 0.00368 (0.37%)** — the standard GEMM correctness measure.
- **max element error = 2.25% of the output RMS scale** (max_abs 0.194 on outputs of RMS ~8.64).
This is exactly the expected bf16 rounding over a K=7168 reduction (bf16 = 8 mantissa bits).
Note: a naive *per-element* relative error is meaningless here — random A·B outputs cross zero, so
near-zero refs blow up the ratio; we therefore report RMS-relative + error-normalized-by-output-scale.

## Masking (variable-M)
`send_counts = {40,17,64,0,55,33,9,50}` per src_rank (deliberately ragged, including a 0-token src
and a full-64 src). The dequant preamble zeroes every row with `slot >= send_counts[src]` (and the
zero-padded buffer rows stay zero), so padded rows contribute nothing. **0 padded(masked) rows had
nonzero output** — masking is correct for M < M_padded.

## Performance
**0.1194 ms/iter, 125.9 TFLOP/s** for one expert's gate/up GEMM at the *padded* shape
512×2048×7168 (256 CUs, 50 timed iters after warmup). This counts FLOPs over the full padded M;
real-token efficiency is lower because most padded rows are masked-zero. The number is a bring-up
sanity figure, not an optimized target — the bf16 8-wave core was ported structurally verbatim and
not retuned for this small-M shape.

## Design options taken (per V2_HK_ANALYSIS.md)
- **Dequant: option (b)** — preamble kernel converts the fp8 A buffer to a bf16 A buffer, applying
  each 128-K-group's fp32 scale, then runs the bf16 MMA core. Simplest correct path; one extra
  convert + one extra bf16 A buffer (M_padded×H×2 B).
- **MMA core**: a **bf16 port of the build-verified `kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu`**
  (256×256 output tile, 8-wave ping-pong via `s_waitcnt`/`s_barrier`/`s_setprio`/`sched_barrier`;
  no NVIDIA warp specialization). Used the generic `load(rt,st)` / `mma_ABt` (bf16.bf16→fp32) path.
- **Weight B: plain bf16, scale = 1** (documented bring-up choice; real R1 weights not needed —
  only correct shapes/dtype). When B becomes fp8 block-scaled, dequant it the same way in the preamble.
- **Variable-M: padded fixed M = world×PER_SRC_CAPACITY**, masked by `send_counts` in the preamble.
- **Launch-per-expert (option i)**: the harness runs expert 0; the kernel is templated on M/N/K and
  pointer-offsetting per `local_e` is a trivial host loop (the dispatch buffer is `[local_e][...]`).

## Deviations from the design doc
1. **BLOCK_K = 64, not 128.** The doc suggested BLOCK_K=128 (= one scale group). bf16 LDS tiles are
   2× the bytes of fp8, so 256×256×128 bf16 double-buffered tiles need 256 KB LDS > the 160 KB
   gfx950 limit (the compiler rejected it: "local memory (262144) exceeds limit (163840)"). Since the
   per-128 scale is **already applied in the preamble**, BLOCK_K no longer needs to equal the scale
   group, so dropping to 64 (128 KB LDS) is harmless and correct. 7168/64 = 112 K-iters.
2. **Standalone (no IRIS/CMake).** Per the task scope, used the FP8_8wave standalone Makefile pattern;
   did not touch the distributed-kernels CMake/CPM/IRIS path.

## Deferred (later steps, explicitly out of scope here)
- **Down-projection** ([.,2048]→[.,7168]) and the SiLU(gate)*up activation fusion.
- **IRIS remote gather**: replace the local A load with `iris_ctx.load(&packed_fp8[...], src_rank)`
  (the buffer's `src_rank` axis = IRIS rank) to gather A-tiles from the producing rank's heap.
- **Comm/compute overlap**: prefetch the next rank/expert's remote tile across the MFMA.
- **In-MMA / per-K-group accumulator scaling (option a)**: fold the two fp32 group scales per K-step
  instead of a preamble convert — saves the extra bf16 A buffer + pass, for final perf.
- **One-grid expert-major tile map (option ii)** instead of launch-per-expert, for occupancy/overlap.
- **Perf tuning** of the bf16 core for this small-M MoE shape.

## Recommended next step
Add the **per-expert host loop** (trivial pointer offset over `local_e` into the `[local_e][...]`
buffer) and a real per-expert weight slab, then fuse **SiLU(gate)*up + the down-projection GEMM** as
a second GEMM on the same buffer — that gives a full single-GPU expert FFN. Defer the IRIS remote
gather + overlap to the step after, building on this verified single-GPU core.
```
status: builds + runs + numerically correct + masking correct on gfx950, single-GPU. DONE for bring-up.
```
