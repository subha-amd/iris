# B1-copy explained — what it is, what it is NOT

B1-copy is the **strong baseline** that refuted V4 at Gate 1 (B1=291µs vs V4=678µs at
M1024/N2048/K7168). This doc states exactly what it does, so it is not mistaken for a production
or fused kernel. Source: `irisx/harness/harness_kernels.cpp` (built as module `tk_kernel`),
driven by `irisx/harness/run_harness.py` case B1.

## Exact kernels/functions B1-copy calls
B1-copy = TWO separate kernel launches, back-to-back, SERIAL, on the consumer rank:
1. `dispatch_pack_quant_once(...)` -> launches `gather_once_kernel<<<M, 256>>>`
   - grid = one block per row (M blocks), 256 threads/block.
   - each block copies one row's K fp8 bytes from the remote src_rank into a LOCAL row-major fp8
     buffer, **16 bytes per thread (uint4)**, then copies that row's NG=K/128 fp32 scales.
   - NO dequant here, NO swizzle — plain row-major local fp8 + scale buffers.
2. `local_gemm(...)` -> the EXACT B0 path: a dequant preamble (`dequant_a_dense<<<M,256>>>`,
   fp8->bf16 using per-128 group scales) then `b0_gemm` = 256x256x64, 8-warp ping-pong HK MFMA.
   - This is byte-identical to B0 (the no-comm compute ceiling). B0 vs B1 differ ONLY in whether A
     arrived for free (B0, already local) or via the one gather (B1).

So B1 = (IRIS gather A once) + (B0 GEMM). Measured: copy 135µs + gemm 150µs = 285µs same-iteration.

## Copy mechanism: IRIS ctx.load (NOT hipMemcpyPeer, NOT SDMA)
The copy is **GPU-issued remote loads via the IRIS device view**:
```
uint4 v = ctx.load(reinterpret_cast<const uint4*>(sp), g.src_rank);   // remote read over symmetric heap
*reinterpret_cast<uint4*>(dst_base + ...) = v;                        // local store
```
- This is a **pure load/store over the IRIS symmetric heap** (XGMI), issued by GPU threads —
  the same mechanism V2.1/V3/V4 use. It is NOT `hipMemcpyPeer` and NOT a DMA/SDMA engine copy.
- Vector width: 16 bytes/thread (uint4) for fp8 bytes. Scales copied as scalar fp32 (1 thread/group).
- Direction: remote LOAD (consumer pulls from producer's heap), then local store. ~56 GB/s measured
  for the M1024 payload (7.57 MB / 135µs) — well under the ~128 GB/s/link, so there is headroom.

## Which production ATOM/AITER stages B1-copy does NOT implement
Production decode MoE path:
`grouped_topk -> EpDispatch -> opus_moe_sorting x2 -> dynamic_quant -> fmoe -> EpCombine -> wv_splitk`
B1-copy implements NONE of the routing/packing/combine. It models only an IDEALIZED:
"A is already gathered/packed/quantized contiguously on the consumer" -> local dense GEMM. Specifically it does NOT do:
- grouped_topk (router) — B1 uses a fixed single source, no top-k.
- real EpDispatch — B1 copies a single dense M×K matrix from ONE src_rank, not route-driven
  scatter from 8 ranks to expert-owning GPUs.
- opus_moe_sorting — no expert-major reordering / sort indices.
- dynamic_quant — B1's A is PRE-quantized on the source; it just copies the fp8 bytes.
- real fmoe layout — B1's local buffer is plain row-major dense [M,K], NOT the AITER fmoe
  sorted-id + per-(expert,128x128) scale layout.
- EpCombine / wv_splitk — no scatter-back, no route_reverse, no shared-expert.
- grouped 32-expert execution — B1 is ONE dense expert matrix, not 32 variable-M_e experts.

## Why B1-copy is a strong baseline but not production-shaped
- STRONG baseline: it isolates the honest minimum cost of "move A once + compute" with a B0-class
  GEMM. Any fused/overlap design must beat it, and its same-iteration split (copy 135 < gemm 150)
  sets a hard overlap ceiling of 1.90x at M1024 (1.28x at M256). [VERIFIED]
- NOT production: it skips every routing/packing/combine stage above. The real next target is
  **B1-dispatch** = route-aware EP8 gather/pack/quant ONCE into the expert-major fmoe layout +
  route_reverse, then a local GROUPED GEMM over 32 experts. B1-dispatch is the production-shaped
  kernel; B1-copy is only its idealized lower bound on the gather+compute cost.

## Numbers (M1024/N2048/K7168, np=2, cv350/r1_c4) — all VERIFIED
B0 164µs/183 TFLOP/s · B1-copy 291µs (copy135+gemm150) · V4/B5 678µs/44 TFLOP/s.
B3/B4/B5 ablation: A-stationary reuse alone = 1.00x; the entire V4 win was OVERLAP (1.82x over B3).
