# V2 — HipKittens fused MoE expert GEMM (bring-up)

These files are **mirrored here for tracking** but are built inside a HipKittens checkout,
not standalone. They depend on HipKittens (HazyResearch/HipKittens) tile headers.

- `fmoe_expert_v2.cu` — single-GPU bring-up of the MoE expert gate/up GEMM that consumes the
  IRISX dispatch FP8 buffer (`packed_fp8 [local_e][src_rank][slot][H]` + `packed_sc` per-128
  fp32 scales). Option (b) preamble-dequant fp8→bf16, bf16 8-wave MMA core, send_counts masking.
- `Makefile` — mirrors the HipKittens `kernels/gemm/fp8fp32/FP8_8wave` build pattern.

## To build/run
Place under a HipKittens checkout (e.g. `HipKittens/kernels/fmoe_expert_v2/`), then:
```
THUNDERKITTENS_ROOT=<HK_ROOT> make    # GPU_TARGET defaults to CDNA4 = gfx950
./fmoe_v2
```
Verified on 8×MI355X (gfx950): RMS-rel error 0.00368, masking correct, ~126 TFLOP/s
(padded M=512, N=2048, K=7168). See `../V2_RESULTS.md` for full results, and
`../V2_HK_ANALYSIS.md` for the design rationale.

## Deferred (next steps)
down-projection + SiLU, per-expert host loop with real weights, IRIS remote-gather of A-tiles
(the comm/compute overlap thesis), in-MMA per-K-group scaling, expert-major one-grid map.
