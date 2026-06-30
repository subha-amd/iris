# MoE Expert Region Presentation Outline

## Slide 1: What The Expert Region Actually Does

The MoE expert region starts after routing has decided which experts each token should visit. With `topk=8`, one original token can become up to eight routed token-expert rows. Each routed row is sent to the GPU that owns the selected expert, runs that expert's FFN, and then contributes back to the original token.

The expert FFN itself is:

```text
x[H=7168]
  -> fc1 / W13_e: H -> 2 * INTER
  -> split into gate[INTER=2048] and up[INTER=2048]
  -> SiLU(gate) * up
  -> fc2 / W2_e: INTER -> H
  -> expert output[H=7168]
```

So `fc1` and `fc2` are not two different MoE layers. They are the two matrix multiplications inside one expert FFN. `fc1` is also called `W13` because it emits both the gate and up projections in one GEMM. `fc2` is the down projection back to hidden size.

`fp8 requant` means converting a freshly computed activation back into fp8 bytes plus scale factors. In this codebase the intermediate after `SiLU(gate) * up` is quantized in groups of 128 elements: compute an `amax`, set `scale = amax / 448`, store fp8 values, and store the scale.

## Slide 2: Why The Unfused Path Is Inefficient

The unfused baseline has three high-level library calls:

```text
MORI EpDispatch -> aiter.fused_moe -> MORI EpCombine
```

At the GPU-kernel-stage level, the C4-faithful bf16-dispatch path is:

```text
1. MORI EpDispatch
2. moe_sorting pass 1
3. moe_sorting pass 2
4. dynamic_quant
5. fmoe_fp8_blockscale_g1u1_silu
6. moe_sum
7. MORI EpCombine
```

The main inefficiency is not that aiter's expert FFN kernel is bad. The `fmoe_fp8_blockscale_g1u1_silu` kernel is already internally fused for the expert math: it contains fc1/W13, SiLU(gate)*up, and fc2/W2. The inefficiency is around that kernel: dispatch, sorting, quantization, and combine are separate stages with materialized HBM handoffs.

The HBM handoffs look like this:

```text
EpDispatch
  writes dispatched bf16 token rows to local HBM on expert GPUs

moe_sorting pass 1
  reads the dispatched token/routing metadata from HBM
  writes counts/prefix metadata for expert ordering

moe_sorting pass 2
  reads dispatched bf16 token rows from HBM
  writes sorted/expert-ordered bf16 token rows back to HBM

dynamic_quant
  reads sorted bf16 token rows from HBM
  writes fp8 activation bytes plus per-group scales to HBM

fmoe_fp8_blockscale_g1u1_silu
  reads fp8 activations/scales and fp8 expert weights from HBM
  writes expert output rows to HBM

moe_sum
  reads expert output rows and route weights from HBM
  writes locally accumulated/combined token output to HBM

EpCombine
  reads combined/expert output from HBM
  communicates results back to origin ranks over XGMI
  writes final token-ordered output on origin ranks
```

So the core inefficiencies are mostly memory and communication related:

The dispatch-to-fmoe boundary materializes tokens in HBM before sorting and quantization. The fmoe kernel then reads a different representation: sorted, fp8, scaled activations. That means the region pays for layout conversion as a real memory pass.

The sorting stage exists because after all-to-all dispatch, tokens arrive in an order convenient for communication, not necessarily in the expert-major order required by the expert GEMM. Sorting pass 1 builds counts/offsets; sorting pass 2 scatters rows into expert order. This is not GEMM work. It is memory traffic and synchronization around layout.

The dynamic quant stage exists because the C4-faithful MORI dispatch moves bf16 tokens. Before fp8 fmoe can run, those bf16 activations must be converted into fp8 plus scales. That is another full read/write pass over activation data.

The combine side has another materialized boundary: fmoe writes expert outputs, then `moe_sum` / `EpCombine` reads them to accumulate weighted top-k outputs and send results back to origin ranks.

There is also kernel-launch and synchronization overhead. This matters most in decode, where row counts are small and fixed launch/latency costs become visible. In prefill, the dominant cost is mostly memory traffic plus GEMM/weight streaming.

The slide takeaway should be:

```text
Unfused inefficiency = good expert kernel surrounded by expensive layout/materialization boundaries.
```

## Slide 3: The Unfused Baseline, Drawn Precisely

Use this as the exact chain on the slide:

```text
MORI EpDispatch
  -> moe_sorting x2
  -> dynamic_quant
  -> fmoe_fp8_blockscale_g1u1_silu
       [fc1/W13 + SiLU(gate)*up + fc2/W2]
  -> moe_sum
  -> MORI EpCombine
```

The important distinction is:

```text
aiter.fused_moe is one Python/library call,
but it launches multiple GPU kernels.
```

Inside `aiter.fused_moe`, the local GPU stages are:

```text
moe_sorting pass 1
moe_sorting pass 2
dynamic_quant
fmoe_fp8_blockscale_g1u1_silu
moe_sum
```

So the full unfused region is seven stages when including MORI dispatch and MORI combine.

## Slide 4: What Our Optimized Path Fuses

Our optimized path is not one monolithic kernel. It is a region-level fused design where the intermediate layouts are controlled by our code.

Use this side-by-side mapping:

```text
Unfused:
EpDispatch -> sort x2 -> dynamic_quant -> aiter fmoe[fc1+SiLU+fc2] -> moe_sum -> EpCombine

Ours:
gather_pack[dispatch+sort+pack]
  -> grouped_b0 fc1
  -> silu_quant
  -> grouped_b0 fc2
  -> combine_pull[sum+combine]
```

More detailed mapping:

```text
EpDispatch + moe_sorting x2
  -> replaced by gather_pack_kernel
     The gather writes rows directly into expert-major packed layout.

dynamic_quant
  -> removed as a standalone stage in our b1 path.
     In the b1 harness, source A is already fp8+scale before the timed region.
     The GEMM side then uses a vectorized dequant preamble.
     This removes the consumer-side HBM materialization, not the mathematical need for quantization.

aiter fmoe expert FFN
  -> replaced by our explicit fc1 grouped_b0_gemm,
     silu_quant_kernel, and fc2 grouped_b0_gemm.

moe_sum + EpCombine
  -> replaced by combine_pull_kernel.
     It groups rows by destination token, reduces locally in fp32,
     then writes one bf16 output row back to the origin rank.
```

The slide wording should be:

```text
We fuse the handoffs around the expert FFN, not merely the FFN math itself.
```

## Slide 5: Gather Pack Kernel

`gather_pack_kernel` is the replacement for `EpDispatch + moe_sorting` at the region level.

The unfused path first communicates token rows, then sorts them into expert order. Our path uses the routing metadata to gather each source row directly into its final expert-major packed row. That means the output of gather is already the layout consumed by grouped GEMM.

Key implementation details:

```text
Grid:
  dim3(Ntile, GP_SPLIT)

Per tile:
  load tile metadata
  check single-source fast path
  otherwise build row -> route segment map

Per row/chunk:
  resolve src_rank and src_row
  load 16 fp8 bytes as uint4
  use local load if src_rank == current rank
  otherwise use IRIS ctx.load over XGMI
  write fp8 bytes to local packed A
  copy per-128 scale
```

Padding and unrouted rows become zero sentinels. This guarantees padded expert rows do not contaminate GEMM output.

The reason this removes sorting is that the destination row is already the expert-major packed row. We do not need a separate `moe_sorting` pass to create the GEMM layout.

## Slide 6: Dynamic Quant And Dequant Handling

In the unfused bf16-dispatch baseline, `dynamic_quant` is required because MORI dispatch moves bf16 activations. The fp8 fmoe kernel needs fp8 activation bytes and scales, so the sorted bf16 activation buffer must be read and converted.

Our current b1 gather path copies fp8 bytes and fp32 scales into the packed buffer. It does not dequantize inside `gather_pack_kernel`. The B0 GEMM path runs a vectorized dequant preamble:

```text
packed fp8 A + per-128 scale
  -> vectorized fp8_to_bf16 dequant
  -> local bf16 scratch A
  -> B0 grouped GEMM
```

The important optimization is that this dequant path was vectorized using packed fp8 conversion and coalesced stores. In the overlap experiments, vectorized gather+dequant changed:

```text
gather-only:              ~168 us
gather+dequant before:    321 us
gather+dequant after:     162 us
effective dequant cost:   ~0 us
```

That is a fusion advantage: dequant work can be made cheap enough to sit under the memory/communication latency. The unfused path cannot hide that same conversion across a library boundary because dispatch and fmoe materialize their handoff.

Important caveat for questions: this does not mean quantization disappears from the model. In the current b1 harness, source activations are already fp8+scale before the measured gather. If the production entry boundary is bf16 tokens, then quantization has to be accounted for either before gather or folded into the gather-side path. The design claim is that it is no longer a separate consumer-side `dynamic_quant` kernel between sorting and fmoe.

## Slide 7: Expert FFN Math

Each routed row runs the expert FFN:

```text
fc1 / W13:
  [M_e, H=7168] x [2*INTER=4096, H=7168]^T
  -> [M_e, 4096]

split:
  gate = first 2048 columns
  up   = second 2048 columns

activation:
  hidden_mid = SiLU(gate) * up

fc2 / W2:
  [M_e, INTER=2048] x [H=7168, INTER=2048]^T
  -> [M_e, H=7168]
```

In the unfused baseline, this math is inside `fmoe_fp8_blockscale_g1u1_silu`. In our path, we expose the steps as:

```text
grouped_b0_gemm for fc1
silu_quant_kernel for activation + fp8 requant
grouped_b0_gemm for fc2
```

This means aiter is more fused inside the expert FFN, while our approach is more fused across the whole dispatch-to-combine region.

## Slide 8: Fused SiLU + FP8 Requant

The activation stage became important because once we stopped using aiter's full fmoe kernel, we had to implement the middle of the FFN ourselves.

The naive PyTorch activation path materialized several large intermediate tensors:

```text
gate bf16 -> fp32
up bf16 -> fp32
SiLU(gate)
multiply by up
amax per 128
scale
fp8 quantized output
```

Our `silu_quant_kernel` does the activation and requant in one kernel:

```text
read C1 = gate || up
compute h = SiLU(gate) * up
compute per-128 amax
write fp8 A2
write scale A2_sc
```

Measured progression:

```text
PyTorch activation:          471 us
first fused kernel:          127 us
contiguous/local-amax path:   38.6 us
```

The key technical reason for the second jump was reducing atomic contention in the amax computation. Each thread owns a contiguous set of elements within one quant group, computes one local amax, and performs one atomic update.

## Slide 9: Combine Pull

The unfused path uses MORI EpCombine. Our first combine implementation was a scatter-style equivalent: each expert-output row pushed contributions back to the origin token/rank.

The diagnostic result was:

```text
scatter fp32 atomic:       788 us
scatter fp32 plain store:  755 us
scatter fp32 uint4 store:  778 us
```

So the problem was not primarily atomics, and not primarily transaction count. It was remote write volume:

```text
8192 rows * 7168 hidden * 4 bytes fp32 ~= 234 MB remote writes
```

The pull combine inverts ownership:

```text
for each destination cell (dst_rank, dst_token):
  gather contributing local expert-output rows
  reduce weighted sum locally in fp32
  write one bf16 row to the origin rank
```

The decisive scheduling detail is destination-rank interleaving:

```text
pull bf16 sorted by dst_rank:      934 us
pull bf16 interleaved by dst_rank: 386 us
MORI EpCombine:                   398 us
```

Sorted-by-rank hammers one XGMI link at a time. Interleaving keeps concurrent blocks spread across all destination links.

## Slide 10: Prefill Result

The prefill result is the strongest claim.

Current SPX-node result:

```text
b1 fused prefill:   1259 us
b3 unfused prefill: 1941 us
speedup:            1.54x
```

Earlier irisx1 exp13 decomposition:

```text
b3 unfused:
  dispatch 290 + fmoe 2622 + combine 398 = 3246 us

b1 fused:
  gather 147 + fc1 921 + act 39 + fc2 481 + combine 387 = 1974 us

speedup:
  1.64x
```

The exact absolute numbers depend on the node/config, so do not mix denominators across machines. The stable story is that prefill wins after removing the layout handoffs, fusing activation/requant, and replacing scatter combine with pull combine.

## Slide 11: Decode Status

Decode is not solved by the same prefill path because the bottleneck changes.

With BM=256, small expert row counts waste most GEMM work. For decode-like shapes, many experts have only a few real rows, but the BM=256 path still computes large padded tiles.

The BM16 decode path fixes the padding issue structurally while keeping the same packed layout:

```text
same 256-padded expert regions
but GEMM task list emits 16-row tiles
```

Current integrated decode result:

```text
b3 unfused decode:       533 us
b1 BM256 decode:         916 us
b1 BM16 bf16 decode:     874 us
```

The reason bf16 BM16 still loses is the weight-byte floor. It still streams bf16 expert weights. The ledger estimates the bf16 decode region floor around 677 us, already above the b3 denominator.

The standalone fp8 BM16 work reached aiter parity after fixing the scheduling fence:

```text
decode-tiny: 49.0 vs aiter 49.7 TFLOP/s
decode:      171.0 vs aiter 171.6 TFLOP/s
```

But the integrated region still needs the saturating fp8/pre-swizzled weight path to become the all-regime win.

## Slide 12: Alternatives We Tested And Rejected

Host-stream overlap was tested and rejected. Even after fixing chunked gather with `GP_SPLIT`, concurrent gather and GEMM streams did not beat the serial/bulk path because the B0 GEMM saturates CUs and streams time-slice.

Measured examples:

```text
concurrent_full vs serial_sum: -12 us to -153 us
pipeline total >= bulk/serial
```

In-kernel gather-under-GEMM was also tested through the only existing body that did remote gather under MFMA:

```text
microtk gather-under-MFMA: 6724 us, 35.8 TFLOP/s
B0 local-A GEMM:            516 us, 466 TFLOP/s
```

The fast B0 body has no producer/consumer warp split and no A reuse across N-subtiles. Blocking remote loads stall the waves instead of hiding communication. So this is not a small patch to the current B0 kernel; it would require a different GEMM body that exposes async A-fill and reuse.

Native fp8 at BM=256 was also rejected as a decode lever:

```text
full occupancy: only about +8%
decode/prefill: neutral or negative
```

At BM=256, decode is compute-bound on padded rows, so halving weight bytes does not help. FP8 only becomes useful once small-BM makes decode weight-memory-bound.

## Slide 13: Summary Mapping

Use this as the final bracket slide:

```text
UNFUSED

EpDispatch
  -> moe_sorting x2
  -> dynamic_quant
  -> fmoe_fp8_blockscale_g1u1_silu
       [fc1/W13 + SiLU(gate)*up + fc2/W2]
  -> moe_sum
  -> EpCombine
```

```text
OURS

gather_pack_kernel
  [EpDispatch + sorting + direct expert-major pack]

vectorized dequant preamble + grouped_b0_gemm
  [input quant/dequant handoff + fc1]

silu_quant_kernel
  [SiLU(gate)*up + fp8 requant]

vectorized dequant preamble + grouped_b0_gemm
  [fc2]

combine_pull_kernel
  [moe_sum + EpCombine-style return path]
```

Final thesis:

```text
aiter fuses the expert math.
our path fuses the expert region.
```

The win comes from making communication produce the compute layout, avoiding standalone sorting, avoiding standalone activation materialization, reducing quant/dequant handoff cost, and replacing scatter-style combine with byte-efficient pull reduce.
