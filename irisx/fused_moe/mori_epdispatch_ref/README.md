# MORI EpDispatch/EpCombine — reference source (for analysis vs our gather_pack/combine_pull)

Pulled from the aiter/MORI container (`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`) at
`/opt/venv/lib/python3.12/site-packages/mori/_jit-sources/` (MORI ships its JIT sources, so these
are the *real* kernels the C4 trace's `EpDispatchIntraNodeKernel` / `EpCombineIntraNodeKernel` run —
not a proxy). Copied here so we can compare, kernel-for-kernel, against `../kernel.cpp`
(`gather_pack_kernel`, `combine_pull_kernel`).

## Files
| file | what it is |
|---|---|
| `dispatch_combine.hpp` | the config (`EpDispatchCombineConfig`) + the kernel args struct (`EpDispatchCombineArgs<T>` / `...Raw`), `PrepareInference`, the cached/replay **routing** struct (`EpDispatchCombineRoutingPtrs`), `LocalExpertCountArgs`, and the standard-MoE convert args. **The clearest file for "what EpDispatch consumes/produces."** |
| `ep_local_expert_count.hpp` | `LocalExpertCountKernel_body` — the per-local-expert **histogram** (atomicAdd of each received `(token,expert)` → `localExpertCount[e]`). This is the sort's count pass. |
| `ep_intranode.hip` | the `extern "C" __global__` **instantiations** (`EpDispatchIntraNodeKernel`, `_stdmoe`, LL, `EpCombineIntraNodeKernel_*`, the fp8/fp4 combine variants, the `ConvertDispatchOutput`/`ConvertCombineInput` kernels). |
| `ep_common.hip` | shared macros (`WRAP_*`) + the include chain. Note it `#include`s `src/ops/dispatch_combine/intranode.hpp` — where the actual dispatch **body** lives (see "still to pull"). |
| `dispatch_combine.cpp` | host-side C++ orchestration (`GetEpDispatchCombineArgsRaw`, handle/buffer setup). 27 KB, not yet fully read. |
| `dispatch_combine.py` | the Python API (`EpDispatchCombineOp` / `EpDispatchCombineConfig`) that `b3_ep8_unfused.py` calls. 62 KB. |

## The dispatch BODY (now present)
`intranode.hpp` = the `EpDispatchIntraNodeKernel_body` + `EpCombineIntraNodeKernel_body` (the actual a2a
algorithm). `common.hpp` / `convert.hpp` are the shared device helpers it uses; `internode.hpp` /
`internode_v1.cpp` / `low_latency_async.cpp` are the multi-node + low-latency variants (not used for our
single-node C4 path). `ep_common.hip:20` is where these get included into the JIT compile.

The aiter kernels EpDispatch is paired with (moe_sorting, dynamic_quant, fmoe, moe_sum) are in the sibling
`../aiter_ref/` — see that README for the full unfused-pipeline → source map.

## Key findings so far (from the files above)
1. **EpDispatch starts from top-k, exactly as we assumed.** `PrepareInference` (hpp:222) takes
   `inpTokenBuf` (activations) + `tokenIndices` (= `topk_ids`) + `weightsBuf` (= `topk_weights`).
   It is *given* router output; it does not compute the router.
2. **It builds the placement ON-DEVICE, from top-k, via atomic counters.** The args carry
   `destPeTokenCounter` / `localPeTokenCounter` / `dispDestTokIdMap` / `dispTokOffset*` (hpp:372–386):
   the kernel histograms tokens per destination PE with atomics, derives offsets, then sends.
   `LocalExpertCountKernel` (the per-expert histogram) is a *separate* kernel.
   → This is the piece **our host-side `build_multisource_route` (SEG/TILE) replaces**: we precompute
   the placement on the host; MORI computes it on-device each dispatch.
3. **MORI ALSO has a cached-routing / replay path** (`EpDispatchCombineRoutingPtrs`, hpp:94–111;
   `GetEpDispatchCombineArgsRaw(..., routing, replayMode)`, hpp:552–557): compute the routing maps
   once, replay them on later dispatches. **`replayMode` is the direct analog of our precomputed
   SEG/TILE.** So the honest fairness split is two-tier:
   - MORI **cache pass** (build maps + `LocalExpertCount`)  ≈  our **host SEG/TILE build**.
   - MORI **replay** dispatch (maps cached)                 ≈  our **`gather_pack`** (maps precomputed).
4. **`_stdmoe` + `ConvertDispatchOutput`** produce the expert-major "standard MoE" packed layout
   directly (`packedRecvX`, `packedRecvLayoutRange`) — the analog of `gather_pack`'s expert-major
   output buffer. So MORI can also fold the sort into the dispatch output.

## The comparison this sets up (see also `../../tilecomm/` writeups)
`gather_pack` (host SEG/TILE + **pull**) is most directly comparable to **MORI replay-mode dispatch**
(cached routing + **push**). MORI's on-device routing math (cache pass + `LocalExpertCount` + aiter
`moe_sorting` + `dynamic_quant`) is what our host adapter + fp8-pre-quant subsume. The a2a itself
won't execute standalone on this MI350X node (verified MORI/RCCL hang), so its µs comes from the real
8-GPU SGLang run; the sort/quant/count run single-GPU per rank.
