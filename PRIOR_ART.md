# Prior Art & Positioning — MoE dispatch/combine on AMD

> Existing work on the exact kernel family we're building (MoE all-to-all dispatch/combine).
> Read this so we (a) don't claim novelty we don't have, (b) steal the proven design ideas,
> and (c) benchmark against the right baselines. Source: prior-art review (2026-06), not
> independently re-verified here — confirm specifics against each repo before quoting numbers.

## TL;DR — what's novel and what isn't
- **NOT novel:** a standalone device-side MoE all-to-all dispatch/combine kernel using P2P/IPC
  remote memory, per-expert atomic counters, vectorized remote stores, and a saved
  token→slot route map. RadeonFlow, Gau Nernst, DeepEP, and Perplexity have all done this on AMD.
- **NOT novel:** quantized EP communication. MoRI (production, SGLang on MI355X) already does
  **FP4 dispatch + FP8 combine** for MXFP4 R1, ~2.56× round-trip BW reduction (28,672 → 11,200
  B/token).
- **The actual contribution:** an **IRISX + HipKittens tile-level C++/HIP communication
  abstraction** for AMD that (1) replaces the *specific* ATOM R1 path
  `EpDispatch → opus_moe_sorting ×2 → dynamic_quant` with one device-side `dispatch_pack_quant`
  kernel producing the *exact* `fmoe_bf16_blockscaleFp8` input layout, and (2) exposes the
  128-wide remote tile move as a primitive that can later live *inside* a HipKittens expert GEMM.
  The differentiation is the reusable tile abstraction + production-layout fusion, not "we wrote
  an all-to-all."

## Positioning — say this, not that
- ❌ "Nobody has written MoE all-to-all dispatch on AMD." (RadeonFlow/Gau/DeepEP/PPLX did.)
- ❌ "Quantizing dispatch is new." (MoRI ships FP4-dispatch/FP8-combine.)
- ❌ "The win is filling idle graph bubbles." (SQL7: ~1 ns gaps; win is HBM-layout/launch/payload.)
- ❌ "Fuse gather into the GEMM first." (Trace + Gau: gather's consumer is the pack/sort, not GEMM.)
- ✅ "Prior work proves this path matters and gives a strong baseline. My target is the IRISX/HK
  tile primitive that replaces `EpDispatch→sort→quant` with a shape-specialized
  `dispatch_pack_quant` producing the real FP8 expert-GEMM layout — a stepping stone to
  AMD-native tile-level compute/comm fusion, benchmarked against MoRI/RadeonFlow/DeepEP."

## The references (study, then beat/benchmark)
| source | what it is | what to take |
|---|---|---|
| **RadeonFlow** `dist-infer/all2all/all2all.cpp` | AMD Challenge grand-prize kernels; hand-written all2all + fake FFN + pull-combine. Hard-codes `MAX_TOPK=8, MAX_HIDDEN_DIM=7168, MAX_NUM_EXPERTS=256, BLOCK_SIZE=256` (our exact shapes). Raw HIP IPC. | data-structure layout (`recv_x[local_expert][slot][hidden]`, `dst_idxs[token][topk]`), atomic slot-claim, vectorized non-temporal copy, bulk signal sync. **Algorithmic reference, NOT a drop-in MI355X baseline.** |
| **Gau Nernst blog** (AMD MI300X a2a) | detailed write-up of building all2all: symmetric heap + translate, acquire/release, atomicAdd slot claim, send/recv split, in-kernel timestamp profiling. | the lessons below; especially that atomics/spin-locks become the bottleneck and torch profiler is unreliable for multi-GPU (use in-kernel timestamps). |
| **MoRI** (SGLang/LMSYS, MI355X) | production quantized EP all-to-all: FP4 dispatch + FP8 combine; cites R1 H=7168/top-8; 2.56× BW cut. | the strong V1 baseline. Check whether ATOM exposes a MoRI quantized-dispatch flag to compare against. |
| **DeepEP** | mature EP comm lib; high-throughput + low-latency a2a, FP8, minimal-SM designs. | low-SM-occupation design ideas; a benchmark bar. |
| **Perplexity pplx-kernels** | dispatch/combine, CUDA-graph capture, send/recv split, **symmetric buffers with sender-owned slices that avoid sender-side sync**. | strong support for V0b (precomputed offsets) over remote atomics. |
| **Flux / COMET / FlashMoE / Triton-distributed** | research on fused GEMM+comm, fine-grained overlap, persistent fused MoE megakernel, OpenSHMEM-in-Triton. | the V2/V3 north star (tile-level overlap), but staged — do NOT start with a persistent megakernel. |
| **HipKittens paper** | tile abstractions generalize to AMD but **schedules must change**: wave specialization underperforms (static reg alloc → producer waves waste registers); use 8-wave ping-pong / 4-wave interleave. | the V2 scheduling constraint — don't copy NVIDIA producer/consumer wave specialization. |

## Concrete design lessons (bake these into V0/V1)
1. **Two slot-claim variants (already in plan, now strongly supported):** V0a remote-atomic
   `fetch_add` (correctness); V0b precomputed offsets / sender-owned slices
   `packed[local_expert][src_rank][slot][hidden]` (perf — PPLX-style, no remote atomics, no
   sender sync). Both RadeonFlow and Gau pay one atomic/assignment and hit contention; PPLX
   avoids it. Build both, benchmark both.
2. **`route_map[token][topk] = slot` is a FIRST-CLASS OUTPUT, not a detail.** RadeonFlow's
   `nvl_dst_idxs[token][topk]` is what makes combine possible later. V0 must emit
   `route_slot[token][topk]` (+ derive dst_rank/local_expert). Without it, V3 combine can't find
   where each expert output landed.
3. **Don't hardcode grid size.** RadeonFlow uses `NUM_SMS=304` (MI300X-ish). MI355X has **256 CUs**
   (8 XCDs). Query it: `grid = device_props.multiProcessorCount`, or pass as a launch arg.
4. **Hidden-vector chunking.** RadeonFlow itself has `TODO: split token to chunks`. One wave per
   full H=7168 vector under-utilizes the GPU at small decode batch (token×topk may not fill CUs).
   Test: one-wave-per-(token,k) vs one-wave-per-(token,k,128-chunk). For V1 the natural chunk is
   the **128-element quant group** (H=7168 = 56 groups).
5. **Cross-rank completion signal is required for correctness.** Local stream order ≠ "all peer
   GPUs finished writing into my memory." The dispatch kernel must end only after a bulk
   completion signal from all ranks (RadeonFlow's lightweight `nvl_signal` path ≈ 1 µs, vs the
   full ping-pong barrier 40–90 µs). For V0/V1 use **bulk completion**; per-expert/per-tile
   signaling is a V2 overlap concern.
6. **In-kernel timestamp profiling early.** torch/rocprof is unreliable for multi-GPU kernels
   (Gau's finding, and matches our own rocprofv3-attach failure). Add a compile-time profiling
   mode with `s_memrealtime`-style timestamps around: slot-claim, load, quant, remote-store, signal.
7. **Buffer-zeroing overhead.** Gau folded buffer resets into a later kernel. Relates to our
   `fillBufferAligned` (which our SQL4 showed is GEMM-workspace, not MoE-path — so likely not our
   concern, but watch for zeroing of the recv counters/buffers).
8. **Vectorized non-temporal loads/stores** for the hidden copy (e.g. 16B `vec_t`), as RadeonFlow does.

## Benchmark baselines (V0/V1 must compare against these, not torch)
```
production:  ATOM/AITER EpDispatch + opus_moe_sorting(P0/P23) [+ MoE-input dynamic_quant for V1]
             MoRI quantized dispatch (FP4/FP8) if ATOM exposes the flag        ← V1 strong bar
references:  RadeonFlow all2all (adapted to EP=8/MI355X), Gau a2a, DeepEP, PPLX patterns
ceiling:     irisx all_put GB/s (raw remote-store bandwidth)
sanity only: torch.distributed.all_to_all (NOT the bar)
```
