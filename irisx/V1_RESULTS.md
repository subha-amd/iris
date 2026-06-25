# V1 MoE dispatch+pack+FP8-quant — results (8x MI355X, gfx950, 2026-06-25)

V1 = V0b (precomputed-offset, sender-owned slice, no remote atomics) + folded per-128-group
FP8 e4m3 quantization in the remote-store path. Writes the `fmoe_bf16_blockscaleFp8` input
layout (fp8 values expert-major + per-128-group fp32 scales), collapsing the production
pre-GEMM chain `EpDispatch -> opus_moe_sorting x2 -> dynamic_per_group_scaled_quant`.

Payload/token: bf16 `7168*2 = 14,336 B` -> fp8+scales `7168 + 56*4 = 7,392 B` (**51.6%**).

## Build (verified)
```
source /usr/share/Modules/init/bash && module load mpi/openmpi-x86_64
cd <repo>/irisx   # node path: see ../NODE_ACCESS.local.md
cmake -B build -DIRIS_BUILD_BENCHMARKS=ON -DIRIS_BUILD_TESTS=ON -DIRIS_HIP_ARCHITECTURES=gfx950
cmake --build build --parallel 8 --target moe_dispatch_pack_quant test_moe_dispatch_pack_quant
```

## Run (force shared-mem MPI transport; node IB can't register memory)
```
mpirun --mca pml ob1 --mca btl self,vader -np 8 ./build/tests/test_moe_dispatch_pack_quant
mpirun --mca pml ob1 --mca btl self,vader -np 8 ./build/benchmarks/moe_dispatch_pack_quant      # add "1" for in-kernel profile
```

## Correctness
test_moe_dispatch_pack_quant: ALL TESTS PASSED on 8 ranks (9 assertions/rank).
  [V1] checked 496 assignments, 0 multiset errors, 0 bad route_slot
  [V1] FP8 e4m3 dequant vs bf16: **max_rel_err = 0.0476**, mean_rel_err = 0.0199 (n=55,776 samples)
The receiver dequantizes packed fp8+scales and compares against the bf16 reference (rebuilt
from MPI-gathered hidden) within e4m3 tolerance (asserted bound 0.13; measured 0.0476). The
per-(src,expert) token multiset matches; route_slot[token][topk] is in range.

## Performance — T_local sweep (H=7168, topk=8, 256 experts, world=8; grid=256 CUs)
Each (token,k) assignment moves its payload. Production-comparable decode batch buckets.

| T_local | V1 fp8 us/inst | V1 GB/s | V0b bf16 us/inst | V0b GB/s | remote% | V1 vs V0b |
|--------:|---------------:|--------:|-----------------:|---------:|--------:|----------:|
| 8       | 16.9 | 28.0  | 16.3 | 56.2  | 88.3 | +3.7% (tie) |
| 16      | 18.7 | 50.5  | 19.5 | 93.9  | 87.6 | -4.1%       |
| 32      | 22.6 | 83.7  | 25.7 | 142.6 | 87.0 | -12.1%      |
| 64      | 32.3 | 117.1 | 38.3 | 191.9 | 87.8 | -15.7%      |
| 128     | **46.0** | 164.7 | 59.9 | 245.1 | 87.8 | **-23.2%**  |

- **GB/s is lower for V1 even though it is faster** because it moves ~half the bytes; the
  GB/s column reflects the smaller payload, not slower transport. Compare the **us/instance**.
- V1 ties V0b at the smallest batch (latency-bound) and pulls ahead as batch grows, peaking at
  **~23% faster at T=128** while sending half the bytes.

## vs the production pre-GEMM envelope (the RIGHT baseline)
Production `EpDispatch + opus_moe_sorting x2 + MoE-input dynamic_quant` measured at
**~46 us/instance** (KERNEL_PLAN SQL1/SQL2, C4-HT rank0, 85,318 instances; p50 43, min 32).

V1 at T=128 lands at **46.0 us/instance** — i.e. it matches the full production three-kernel
pre-GEMM envelope in a single device kernel, while also removing the two bf16 HBM round-trips
(dispatch staging reread by sort; bf16 expert-major reread by quant) and the extra graph nodes.
At decode-typical small batches (T=8..32) V1 is **16.9..22.6 us**, i.e. well under the ~46 us
envelope. This is the first apples-closer V1 number; caveats below.

## The FIVE validations (KERNEL_PLAN §4)
1. **Quant semantics — CONFIRMED equivalent.** scale = max(|tile|)/448 over each 128-element
   group of a single token's hidden vector. It depends ONLY on that token's own hidden groups,
   never on the expert-packed destination layout. Therefore quantizing pre-dispatch (in the
   sender's store path, as V1 does) is numerically identical to the production post-pack
   `dynamic_per_group_scaled_quant`. Pre-dispatch quant is valid.
2. **Exact fmoe layout — produced, FLAGGED for ATOM confirmation.** No `fmoe_bf16_blockscaleFp8`
   source is on this node, so the byte layout could not be verified against the real kernel.
   V1 produces: `packed_fp8[local_e][src_rank][slot][H]` (fp8 e4m3, expert-major, contiguous
   H) + `packed_sc[local_e][src_rank][slot][N_GROUPS]` (fp32, 56 per-128-group scales,
   group g = elements [128g,128g+128)). FP8 = OCP e4m3 (`__HIP_E4M3`, HIP_FP8_TYPE_OCP=1 on
   this ROCm 7.2.4), saturate-to-finite, max=448. **MUST confirm against ATOM:** (a) scale
   dtype (fp32 vs e8m0/bf16) and whether scales are interleaved vs separate tensor, (b) the
   exact [expert][token] index ordering fmoe expects, (c) e4m3 OCP vs fnuz.
3. **Accuracy — PASS within e4m3 tolerance.** max relative error 0.0476, mean 0.0199 over
   55,776 element samples across 8 ranks (e4m3 has 3 mantissa bits -> worst-case per-element
   rel step ~2^-3 = 0.125; measured well inside). Note: not the ATOM gsm8k lm_eval smoke test
   (no model on node) — that end-to-end accuracy check remains TODO.
4. **Remote fraction — measured 87.0–88.3%** (MPI_Allreduce over actual routing), matching the
   EP8 expectation (~87.5% = 7/8). The FP8 payload cut applies to that ~88% of assignments;
   the ~12% intra-rank assignments still write fp8 (local store, no XGMI).
5. **Bandwidth vs latency/quant bound — QUANT-BOUND, not byte-bound.** This is the key finding.
   Halving the bytes did NOT halve the time: V1 is ~23% faster at T=128, not ~50%. The
   in-kernel profile (PROFILE=1) consistently shows the quant region (load + group-max reduce +
   fp8 convert) costs far more cycles than the remote-store region. (The PROFILE numbers are
   themselves inflated by lane-0 clock64/atomicAdd serialization, so treat the ratio as
   directional, not absolute.) At small batch (T=8) V1 ties V0b -> latency-bound. Conclusion:
   the remaining win lever is faster on-chip quant (or moving quant off the critical store
   path), NOT further payload reduction. FP4 would cut bytes more but not time at this bound.

## Kernel design (what made it competitive)
First V1 draft (1 block/assignment, 56 groups serial, per-group __syncthreads, single-byte
fp8 stores) was **3.5x slower than V0b** — pure quant/serialization overhead. Final design:
- 256-thread block = 8 independent 32-lane tiles; each tile owns one 128-elem quant group and
  strides over the 56 groups (8 groups in flight, no inter-group barrier).
- Group max = 32-lane sub-warp `__shfl_xor` reduction (NO `__syncthreads`).
- Each lane quantizes 4 contiguous elements and packs them into one 32-bit word -> a single
  4-byte remote store (32 stores/group instead of 128). One fp32 scale store per group by lane 0.
- Kept V0b's count->offset machinery, sender-owned slice (no remote atomics), bulk completion
  signal, device-queried grid (256 CUs), route_slot output, compile-time PROFILE mode.

## Honest remaining non-comparability
- **No CUDA-graph capture / no compute-comm overlap.** Production runs in graph mode (~1 ns
  stage gaps); this is a standalone microbench. The ~46 us match is launch-overhead-inclusive.
- **Intra-node IPC only** (IRISX is single-node symmetric heap). No inter-node XGMI/RDMA path.
- **fmoe layout not byte-verified** (validation #2) — could be off vs the real kernel.
- **No end-to-end accuracy** (no gsm8k/lm_eval; no R1 weights on node) — only numeric e4m3 bound.
- **Synthetic uniform routing** (skew would change per-expert load and the remote fraction tail).
- **MoRI comparison not run** — ATOM/MoRI quantized-dispatch flag not exposed on this node;
  documented as the strong external bar (PRIOR_ART.md) but not measured here.

## Files (under irisx/)
- benchmarks/moe_dispatch_pack_quant.hip   (V1 kernel + V0b bf16 reference + count + signal + PROFILE + sweep)
- tests/test_moe_dispatch_pack_quant.hip   (Catch2, 8-rank: dequant vs bf16 + multiset + route_slot)
- benchmarks/CMakeLists.txt, tests/CMakeLists.txt   (registered moe_dispatch_pack_quant / test_*)
- V1_RESULTS.md   (this file)

## Recommended next step
Quant is the bound, not bytes (validation #5). Options, in order of payoff:
1. Move the fp8 conversion off the critical store path (overlap quant of group g+1 with the
   remote store of group g) or use the packed `__hip_cvt_float2_to_fp8x2` path to halve convert ops.
2. Confirm the real `fmoe_bf16_blockscaleFp8` layout against ATOM (validation #2) and align bytes.
3. Run the ATOM gsm8k smoke test for true end-to-end accuracy (validation #3).
4. V2: feed these fp8 tiles into a HipKittens expert GEMM (the actual novel tile-comm fusion).
