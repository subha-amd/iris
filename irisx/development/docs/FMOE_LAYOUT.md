# FMOE input layout — source-verified spec for `fmoe_fp8_blockscale_g1u1` (R1-0528 EP, gfx950/MI355X)

Source investigation of the production stack in docker image
`rocm/atom-dev:vllm-v0.22.0-nightly_20260610` (read-only, no GPU). Validation #2 of
V1_RESULTS.md. The op our V1 reproduces the input for is **`fmoe_fp8_blockscale_g1u1`**
(AITER), reached from ATOM via `aiter.fused_moe.fused_moe` with `QuantType.per_1x128`.

> TL;DR: scale dtype/granularity (fp32, per-(1 token x 128 along K=H)) is **correct**, and
> on gfx950 the fp8 format (**OCP e4m3 / float8_e4m3fn**) is **correct**. The two real
> mismatches are: (1) the **scale array must be transposed/shuffled** (group-major, token in
> the contiguous dim) — not our plain `[...][N_GROUPS]` token-major order; and (2) the value
> buffer the kernel reads is **plain row-major `[token][H]` indexed via `sorted_ids`**, not an
> expert-major `[expert][rank][slot][H]` pack. A is **NOT** preshuffled (only weights B are).

---

## 0. Which code path is production on this node

- ATOM `/app/ATOM/atom/model_ops/moe.py`: imports `from aiter.fused_moe import fused_moe`
  (line 13) and calls `fused_moe(...)` (lines 609/696/1120). `QuantType.per_1x128` is
  selected for fp8 block-quant (line 378, 749). Weights are preshuffled via
  `shuffle_weight`/`shuffle_scale` and tagged `is_shuffled=True` (lines 942-968) — **B-side only**.
- `aiter/fused_moe.py` line 769-770: dispatch table
  `(Silu/Gelu, per_1x128, bf16 in, fp8 w1, fp8 w2, ...) -> aiter.fmoe_fp8_blockscale_g1u1`.
- The legacy `asm_moe` block_scale path (`fused_moe_bf16_asm.py`) is the gfx942 ASM variant;
  it hardcodes fnuz and is **not** the gfx950 production path, but it documents the same
  contract (transposed scale, row-major A) and is quoted below as corroborating evidence.

---

## 1. Scale: dtype, granularity, axis  (CONFIRMED from source)

- **dtype = fp32.** `pertoken_quant` defaults `scale_dtype=dtypes.fp32`
  (quant.py:56-63); `per_group_quant_hip` allocates
  `scale = torch.empty((*shape[:-1], shape[-1]//group_size), dtype=dtypes.fp32)`
  (quant.py:~420). Docstring of `dynamic_per_group_scaled_quant` (quant.py:722):
  "`dtypes.fp8` -> MXFP8 (**fp32 per-group scale today**)". NOT e8m0 for this fp8 path
  (e8m0 is only the fp4x2/MXFP4 and the optional MXFP8 `scale_type=fp8_e8m0` path, which
  per_1x128 does not use).
- **group = 128 along K (= the hidden/contraction axis H=7168).** `QuantType.per_1x128`
  = `per_block_quant_wrapper((1,128))(pertoken_quant)` (quant.py:323); wrapper asserts
  `blk_m==1` and reduces over the last (K) dim in 128-chunks (quant.py:~300-310);
  `block_shape==(128,128)` and `scale_blk_k=128` (fused_moe_bf16_asm.py:141-152). So one
  scale per (single token row, 128 contiguous hidden elements) = 56 groups for H=7168.
- **Per-(token,group), NOT per-(expert,group).** Quant is over each token's own hidden
  vector, independent of expert packing (matches V1_RESULTS validation #1).

## 2. Scale layout / interleave  (CONFIRMED separate + transposed; exact pad = needs-runtime-check)

- **Separate contiguous tensor**, passed as its own arg `input_scale`/`a1_scale` to
  `fmoe_fp8_blockscale_g1u1(out, input, gate, down, sorted_ids, sorted_weights,
  sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale, ...,
  fc_scale_blkn=128, fc_scale_blkk=128, ...)` (moe_op.py:173). NOT interleaved with values.
- **Transposed / shuffled (group-major, token in the contiguous last dim).** Two independent
  call sites do this:
  - CK path (production): `fused_moe.py:615-616` —
    `if quant_type==per_1x128: quant_func = partial(quant_func, transpose_scale=True)`.
    `per_group_quant_hip(..., transpose_scale=True)` forwards `shuffle_scale=transpose_scale`
    into the device kernel `dynamic_per_token_scaled_quant(y, x.view(-1,group), scale,
    shuffle_scale=transpose_scale, ...)` (quant.py:~430-440). When the input is already
    quantized it instead calls `aiter.partial_transpose(scale_t, a1_scale, ...)`
    (fused_moe.py:628-633) — same effect: transpose the scale.
  - ASM path (corroborating): `a1_scale = a1_scale.squeeze(-1).t().contiguous()`
    (fused_moe_bf16_asm.py:159). Logical scale is `[M_tokens, H/128]`; the kernel wants it
    **transposed to `[H/128, M_tokens]` contiguous** (group index outer, token index inner).
- Net: the GEMM reads scale as **group-major** — for a fixed 128-group g, all tokens'
  scales are contiguous. (The exact zero-padding of the token dim to the GEMM tile —
  e.g. pad to block_size_M=32 — is **needs-runtime-check**; the transpose order itself is confirmed.)

## 3. FP8 format: OCP e4m3 vs fnuz  (CONFIRMED — arch-dependent)

`aiter/utility/dtypes.py:10-26`:
```
"gfx942": {"fp8": torch.float8_e4m3fnuz},   # MI300  -> FNUZ, max 240
"gfx950": {"fp8": torch.float8_e4m3fn},     # MI355X -> OCP,  max 448
fp8 = get_dtype_fp8()   # resolves per running arch
```
- **This node is gfx950 (MI355X) -> `float8_e4m3fn` = OCP e4m3, max 448.** This matches
  our V1 (`__HIP_E4M3`, max 448).
- Footgun caveat: the same kernel family on **gfx942 uses fnuz (max 240)**. The legacy
  `asm_moe` block path even hardcodes `w1.dtype == torch.float8_e4m3fnuz`
  (fused_moe_bf16_asm.py:147). ATOM has `normalize_e4m3fn_to_e4m3fnuz` /
  `need_normalize_e4m3fn_to_e4m3fnuz = (quant_dtype==float8_e4m3fnuz)` (moe.py:48,1174).
  So V1's OCP e4m3 is correct **only** for gfx950; on MI300 the bytes would need fnuz.

## 4. Value (A-matrix) layout: row-major vs preshuffled  (CONFIRMED plain row-major)

- A (token activations) is **plain row-major `[token, H]` fp8**, NOT preshuffled:
  - CK path: `a1 = hidden_states` (kept as-is) / quantized in place to
    `[token_num, model_dim]`; only the **scale** is transposed (fused_moe.py:607-633).
  - ASM path: `a1_q = a1_q.view(-1, model_dim)` — flat row-major `[M, H]`
    (fused_moe_bf16_asm.py:157).
- **Preshuffle is B-side (weights) only.** Every CK kernel/header is named
  `...blockscale_b_preshuffle...` (e.g.
  `blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_gufusion_v3.hpp`); the example
  is `gemm_multiply_multiply_xdl_fp8_blockscale_bpreshuffle.cpp`. ATOM shuffles only
  w13/w2 and sets `is_shuffled=True` on the weights (moe.py:942-968). The `is_shuffled`
  flag is read from `w1/w2` (fused_moe.py:378,1681) — never from the activation. The
  `ps`/`psx`/`pf2` tokens in the `.co` filenames are **weight-preshuffle / prefetch**
  variants, not A-preshuffle.
- A is consumed via the MoE sort indirection: the kernel takes `sorted_ids`,
  `sorted_expert_ids`, `num_valid_ids` and gathers the row-major `[token,H]` A by sorted id.
  There is **no expert-major `[expert][rank][slot][H]` A buffer** in the production contract.

## 5. Exact index math the production kernel expects

Let M = number of token rows (`hidden_states.shape[0]`), H = 7168, G = 128, NG = H/G = 56.

- **Values** `a1_q` (fp8 e4m3 OCP), shape `[M, H]` row-major:
  `offset_bytes(token m, hidden h) = m*H + h`  (1 byte/elem). Token order is the **input
  token order**; the GEMM remaps to experts at runtime via `sorted_ids` (it does NOT expect a
  pre-packed expert-major buffer).
- **Scales** `a1_scale` (fp32), logically `[M, NG]` but stored **transposed** `[NG, M]`
  contiguous (group-major):
  `offset_floats(token m, group g) = g*M + m`  (i.e. `a1_scale.t().contiguous()`;
  M may be padded to the GEMM M-tile — exact pad needs-runtime-check).
- Weights (out of scope for our producer) are preshuffled + `fc1_scale`/`fc2_scale` are
  per-(expert, 128x128) block scales — we do not produce these.

---

## 6. Point-by-point DIFF: V1 assumption vs production reality

Our current V1 (moe_dispatch_pack_quant.hip):
```
packed_fp8 [local_e][src_rank][slot][H]        fp8 e4m3 OCP, expert-major, contiguous H
packed_sc  [local_e][src_rank][slot][N_GROUPS] fp32 per-128-group scale, token-major
```

| # | Aspect | V1 produces | Production expects | Verdict / fix |
|---|--------|-------------|--------------------|---------------|
| 1 | Scale dtype | fp32 | fp32 | **CORRECT** (confirmed) |
| 2 | Scale granularity | per-(token,128-group along H) | per-(1 token x 128 along K=H), 56 groups | **CORRECT** (confirmed) |
| 3 | FP8 format | OCP e4m3 (`__HIP_E4M3`, max 448) | gfx950: OCP e4m3 (float8_e4m3fn, 448) | **CORRECT on gfx950** (confirmed). NOTE: would be WRONG (need fnuz/240) on gfx942. |
| 4 | A preshuffle | plain row-major (per token row `[H]`) | plain row-major, NOT preshuffled | **CORRECT** — only B is preshuffled (confirmed) |
| 5 | Scale memory order | token-major `[...][g]` (g contiguous) | **transposed: group-major `[g][token]`** (token contiguous) | **MISMATCH** — must transpose the scale array (store `scale[g*M_pad + token]`, not `[base*NG + g]`). Confirmed by `transpose_scale=True` (fused_moe.py:616) and `.t().contiguous()` (asm:159). |
| 6 | Value buffer org | expert-major `[local_e][rank][slot][H]` | row-major `[token][H]` + `sorted_ids` indirection | **MISMATCH in framing** — production GEMM does NOT take an expert-major A pack; it takes row-major A and a sort map. If V1 keeps an expert-major pack, it must ALSO emit a matching `sorted_ids/sorted_expert_ids/num_valid_ids` (the `moe_sorting` output) OR re-layout A to `[token,H]` so fmoe can consume it. Confirmed: `a1=hidden_states` row-major + sorted_ids args (fused_moe.py / moe_op.py:173). |
| 7 | Scale token padding | none (NG per slot) | token dim likely padded to GEMM M-tile (block_size_M=32) | **needs-runtime-check** — transpose order confirmed; exact pad not provable from source. |

### Concrete changes V1 needs
1. **Transpose the scale buffer** to group-major: index `scale[g * M_pad + token_row]`
   (fp32), where `token_row` is the row index fmoe will read (post-sort). This is the single
   most important byte-layout fix.
2. **Reconcile the value-buffer model.** Production fmoe consumes row-major `[token,H]` A plus
   the `moe_sorting` index tensors (`sorted_ids`, `sorted_expert_ids`, `num_valid_ids`,
   `sorted_weights`). V1's expert-major `[local_e][rank][slot][H]` pack is a *different*
   contract; to feed real fmoe, either (a) write A as `[token,H]` and produce the sort map,
   or (b) treat V1's pack as a custom GEMM input and write our own GEMM (the V2 plan).
3. **Keep OCP e4m3** for gfx950 (no change). Add an arch guard: if ever run on gfx942, switch
   to fnuz (max 240).
4. Scales stay fp32 (no e8m0 change) and 128-group along H (no change).

---

## 7. Confidence per finding

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Scale = fp32 | **CONFIRMED (source)** | quant.py:56-63, ~420, 722 docstring |
| group=128 along K, per-(1 token) | **CONFIRMED (source)** | quant.py:323, ~300-310; asm:141-152; fused_moe.py:769 |
| fp8 = OCP e4m3 on gfx950 (fnuz on gfx942) | **CONFIRMED (source)** | dtypes.py:10-26; asm:147; moe.py:1174 |
| Scale is separate tensor, transposed (group-major) | **CONFIRMED (source)** | fused_moe.py:615-633; asm:159; moe_op.py:173 |
| A is plain row-major, NOT preshuffled (B-side preshuffle only) | **CONFIRMED (source)** | asm:157; fused_moe.py:378,607,1681; CK b_preshuffle header/example names; moe.py:942-968 |
| A consumed via sorted_ids (no expert-major A pack) | **CONFIRMED (source)** | moe_op.py:173; fused_moe.py:607-690 |
| Exact token-dim padding of transposed scale | **needs-runtime-check** | not provable from source; transpose order is, pad isn't |

## 8. Key source files (evidence)
- /app/aiter-test/aiter/ops/moe_op.py:173  — fmoe_fp8_blockscale_g1u1 signature (args/order)
- /app/aiter-test/aiter/fused_moe.py:378,607-690,769-770,1190-1270  — CK path, transpose_scale, dispatch, is_shuffled
- /app/aiter-test/aiter/fused_moe_bf16_asm.py:140-180  — block_shape(128,128), fnuz weight, a1_q row-major, a1_scale.t().contiguous()
- /app/aiter-test/aiter/ops/quant.py:56-130,300-470,722  — pertoken_quant, per_block_quant_wrapper, per_group_quant_hip, scale dtype, shuffle_scale
- /app/aiter-test/aiter/utility/dtypes.py:10-27  — gfx942 fnuz vs gfx950 fn(OCP)
- /app/ATOM/atom/model_ops/moe.py:13,378,609,942-968,1174  — production wiring, per_1x128, weight-only shuffle, e4m3fn->fnuz normalize
- /app/aiter-test/3rdparty/composable_kernel/.../blockwise_gemm_pipeline_xdlops_moe_blockscale_b_preshuffle_gufusion_v3.hpp  — B-side preshuffle naming
