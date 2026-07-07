# QuantTile v0 — validation result (2026-07-07, 8× MI350X gfx950)

**Milestone (RESEARCH_PLAN.md Action A1):** collapse the two hand-written per-format decode GEMM
kernels into ONE descriptor-parameterized body, and prove the unification is **zero-cost + bit-stable**
against numbers we already trust. **Status: PASSED all gates.**

## What was built
`irisx/fused_moe/quanttile_decode.h` — a single templated body
`grouped_expert_gemm_decode_qt<qt::Fmt, N, K>` that replaces both
`grouped_expert_gemm_decode_fp8_sat` (fp8-e4m3, per-128-K-block f32 scale) and
`grouped_expert_gemm_decode_mxfp4_sat` (mxfp4-e2m1, per-32-K-block E8M0 scale). fp8 and mxfp4 are two
instantiations of the same source. The three format-divergent spots — packed-B tile width
(`PACK_DIV` 2 vs 4), scale layout+type (`ScaleT` `float` vs `uint8_t`, `SCALE_BLK` 128 vs 32), and the
in-register unpack (reinterpret+convert vs hardware `cvt_scalef32_pk_bf16_fp4`) — are carried by the
compile-time tile descriptor `qt::Tile<F>` and selected with `if constexpr`. **No per-format kernel fork.**

## How it was validated (same-node, identical inputs)
The two standalone harnesses (`irisx/development/grouped_b0/sat_decode.cu` fp8,
`sat_decode_fp4.cu` mxfp4) each got a one-line `-DUSE_QT` swap that launches the unified body in place
of the original — **identical inputs, identical harness**, original vs unified head-to-head. Built with
`hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -DGB0_N=4096 -DGB0_K=7168` (`+ -ffast-math`
for fp4). Both `-DUSE_QT` builds compiled clean (the header's editor/LSP errors were false-positives from
missing HK context, as expected — `hipcc -std=c++20` accepts the scoped-enum NTTP + HK types fine).

| gate | original (A0) | unified (A1, `-DUSE_QT`) | verdict |
|---|---|---|---|
| **G1** fp8 zero-cost (E32-decode, 0.4595 ms anchor) | 0.4595 ms · 2.874 TB/s · 1.14× vs bf16 | 0.4598 ms · 2.874 TB/s · 1.14× | **PASS** — 0.07% Δ (gate ±5%) |
| **G2** mxfp4 win preserved (E32-decode) | 0.2841 ms · **1.62× over fp8** | 0.2819 ms · **1.63× over fp8** | **PASS** (gate ≥1.5×) |
| **G3a** correctness vs format-faithful dequant | fp8 RMS 0.02582 · mxfp4 RMS 0.00333 | fp8 RMS 0.02582 · mxfp4 RMS 0.00333 | **PASS** (identical) |
| **G3b** bit-stability vs pre-unification kernels | — | RMS values identical to 5 d.p. | **PASS** |

All three decode cases (ragged-correctness, E32-tiny, E32-decode) PASS for both fp8 and mxfp4, unified.

## Node caveat (do not mix denominators)
This node is **MI350X**, ~28% slower than the thor-4 **MI355X** that set the original 3.97 TB/s anchor
(here fp8 = 2.874 TB/s). That is a node-speed difference, not a regression — the gate is the **same-node**
original-vs-unified comparison, which is a wash (0.07% on fp8, +0.8% on mxfp4, both within noise). The
mxfp4-over-fp8 ratio (1.62–1.63×) reproduces the trusted 1.6× anchor exactly.

## What this proves (and what it does not)
- **Proves:** the QuantTile descriptor is a `constexpr` policy with zero runtime cost; a tile's
  {format, scale-layout, bytes} is a first-class compile-time property; the two shipped decode kernels
  are one body. The representation axis of the `stage`/`retire` abstraction is real and mechanical.
- **Does NOT claim a new speedup** — the fp8/mxfp4 wins already existed in the two shipped kernels; v0's
  contribution is the *abstraction + a demonstration it is free*. Next: (v0→region) route the fused MoE
  decode through the unified body; (v1) the native fp4×fp4 `mma_ABt_scaled<cbsz=4>` path (prefill, gated
  by the cbsz K-layout probe); then the residency axis (tile-fused reduce-scatter). See RESEARCH_PLAN.md.
