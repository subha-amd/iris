# AITER unfused-pipeline kernels — reference source

The actual aiter kernel sources for the **unfused production MoE region** (the 7-stage baseline in the
june-30 talk), pulled from `/app/aiter-test/{aiter,csrc}` in the aiter/MORI container
(`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`). These are the real kernels `b3_ep8_unfused.py` /
SGLang run — grabbed so we can reference our fused kernels (`../kernel.cpp`) against them exactly.
The MORI EpDispatch/EpCombine sources are the sibling dir `../mori_epdispatch_ref/`.

## The unfused pipeline → source-file map (june-30 §"unfused baseline")
| stage (unfused) | what it does | source here |
|---|---|---|
| **EpDispatch** | a2a: push tokens to expert-owner GPUs | `../mori_epdispatch_ref/` (`intranode.hpp` body, `dispatch_combine.*`) |
| **moe_sorting ×2** | count per-expert + scatter into expert-major order | `csrc/moe_sorting/moe_sorting_kernels.cu` (+ `moe_sorting_opus_kernels.cu`), `csrc/include/{moe_sorting.h,warp_sort.h}`, `moe_sorting.py` |
| **dynamic_quant** | bf16 → fp8 (per-1x128) before the GEMM | `csrc/quant_act/{quant_kernels.cu,quant_mxfp4.cu,quant_utils.cuh,mx_quant_utils.h}`, `quant.py` |
| **fmoe (fc1 + SiLU·up + fc2)** | the fused expert GEMM (the "good kernel") | `csrc/fmoe_2stages/gemm_moe_ck2stages_common*.cuh` (the CK 2-stage **algorithm**: `_common.cuh`, `_common_blockscale.cuh` = the fp8 `fmoe_..blockscaleFp8`, `_common_mxfp4.cuh`/`_bns.cuh` = the MXFP4 path), `gemm_moe_ck2stages.{h,cu}`, `moe_ck_gemm.hpp`, one fp8 `gemm1/gemm2` instance, `fused_moe.py`; activation in `csrc/quant_act/activation_kernels.cu` |
| **moe_sum** | weighted top-k reduce of expert outputs | `csrc/moe_op_sum/{topk_softmax_kernels.cu,moe_align_block_size_kernels.cu,moe_op.h}`, `moe_op.py`, `topk.py` |
| **EpCombine** | a2a: scatter expert outputs back to origin tokens | `../mori_epdispatch_ref/` (the `EpCombineIntraNodeKernel_*` in `ep_intranode.hip` + `intranode.hpp`) |

## The two low-precision GEMM paths (what our decode QuantTile competes with)
- `csrc/ck_a8w8_fp8/gemm_a8w8_blockscale_bpreshuffle{.h,_common.cuh}` — the **fp8 block-scale** GEMM (per-128-K fp32 scales, B pre-shuffled).
- `csrc/ck_a4w4_fp4/gemm_a4w4_blockscale{.h,_common.cuh}` — the **MXFP4** (`a4w4`) GEMM (the untuned/buggy-for-EP path from §10; the one we must NOT gate against).

## Notes
- The fmoe expert GEMM is **CK (Composable Kernel) 2-stage**: `gemm1` does fc1 + SiLU·up (fused), `gemm2` does fc2, both fused into one `fused_moe` call — this is why the june-30 deck has no standalone unfused activation to race. The `_common*.cuh` files are the templates (the algorithm); the ~50 `moe_ck_gemm{1,2}_instance_*.cu` are generated tiling variants (we kept one representative fp8 pair).
- Only the *representative* instances + the *template* headers were pulled (not every generated tiling / `.o` / `.hsaco`). To pull more, see the paths in the container under `/app/aiter-test/csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/`.
- License: aiter is MIT (see the headers); MORI is MIT. These are third-party reference copies for analysis, not part of our build.
