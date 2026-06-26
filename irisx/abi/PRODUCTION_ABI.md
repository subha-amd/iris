# PRODUCTION_ABI — finalized MoE dispatch/route ABI (R1-0528, EP8, gfx950/MI355X)

Agent 00 owns this file. It is the single binding ABI all IRISX MoE kernels speak. It
**extends** the structs in `irisx/AGENT_COMMON.md` §3; where this file and AGENT_COMMON
disagree, **this file wins** (AGENT_COMMON §3 explicitly delegates the reverse-route /
combine metadata to Agent 00).

Companion files: `route_abi.h` (the C header — authoritative struct bytes),
`ROUTE_CAPTURE_SCHEMA.md` (instrumentation), `route_generators.py` (synthetic buffers).
Layout evidence for the production fmoe input lives in `irisx/FMOE_LAYOUT.md` (source-cited).

Placeholders: `<NODE>`,`<USER>`,`<HK_ROOT>`,`r1_c4`. Every item that cannot be proven from
the local repo source is tagged **[NEEDS-NODE-MAIN-AGENT]** with the exact grep to run on the
node (image `rocm/atom-dev:vllm-v0.22.0-nightly_20260610`, container `r1_c4`, GPU disabled).

---

## 0. Problem constants (binding)

```
EP_SIZE = 8 ranks                 256 global routed experts
32 local experts / rank           top-k = 8
H = 7168  (W13 input / contraction K, and W2 output N)
GROUP = 128  ->  N_GROUPS = 56 fp8 block-scale groups per token
FP8 = OCP e4m3 (float8_e4m3fn, max 448)  on gfx950   [fnuz/240 on gfx942 — guard]
scales = fp32                     output = bf16
```
`M` always = **per-expert rows (M_e)** unless a field name says aggregate. Never mix.

---

## 1. The op we feed (re-derived, source-cited)

Production op: **`aiter.fmoe_fp8_blockscale_g1u1`**, reached from ATOM
`moe.py` -> `aiter.fused_moe.fused_moe(..., QuantType.per_1x128)`.

- Dispatch table `(Silu, per_1x128, bf16 in, fp8 w1, fp8 w2) -> fmoe_fp8_blockscale_g1u1`
  — `aiter/fused_moe.py:769-770`.
- Signature/arg order — `aiter/ops/moe_op.py:173`:
  `fmoe_fp8_blockscale_g1u1(out, input, gate, down, sorted_ids, sorted_weights,
  sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale,
  fc_scale_blkn=128, fc_scale_blkk=128, ...)`.
- `per_1x128` selected — `moe.py:378,749`; weights preshuffled, `is_shuffled=True` —
  `moe.py:942-968` (**B-side only**).

### 1.1 W13 (fc1) — fused gate||up, g1u1
- "g1u1" = **gate (1) and up (1) fused** into one weight `w13`; epilogue is
  `SiLU(gate) * up`. Effective fc1 output width **N = 4096** (= 2 x 2048).
- Per-rank weight tensor logical shape (B side, preshuffled — we do NOT produce it):
  `w13[local_e, N=4096, K=H=7168]` fp8 + per-(expert,128x128) `fc1_scale`.
- **[NEEDS-NODE-MAIN-AGENT]** exact N-split order inside the fused 4096 (is it
  `[gate(0:2048) | up(2048:4096)]` or interleaved per-128-block?). Grep on node:
  ```
  docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
    grep -rniE "g1u1|gate.*up|N//2|n//2|n/2|silu" \
    /app/aiter-test/aiter/fused_moe.py /app/aiter-test/aiter/ops/moe_op.py'
  # and the CK kernel epilogue:
  grep -rniE "gufusion|gate|up|silu|swiglu" \
    /app/aiter-test/3rdparty/composable_kernel/ | grep -i moe | head
  ```

### 1.2 W2 (fc2) — down projection
- Logical (B side): `w2[local_e, N=7168, K=2048]` fp8 + per-(expert,128x128) `fc2_scale`.
  K=2048 = fc1 output width / 2 after SiLU gating (4096 -> 2048 active -> down to H).
- High-confidence from R1 model dims; **[NEEDS-NODE-MAIN-AGENT]** fresh source-cite:
  ```
  docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
    grep -rniE "w2|down|intermediate|2048|7168" /app/ATOM/atom/model_ops/moe.py | head'
  ```

---

## 2. The A-side (activation) input contract — what our dispatch must emit

From `irisx/FMOE_LAYOUT.md` (all CONFIRMED-source unless noted):

- **Values** `input` = fp8 e4m3 OCP, **plain row-major `[M, H]`**, 1 byte/elem,
  `offset(m,h) = m*H + h`. Token order = input token order; the GEMM remaps to experts via
  `sorted_ids`. A is **NOT preshuffled** (only B/weights are). `asm:157`, `fused_moe.py:607`.
- **Scales** `input_scale` = fp32, logically `[M, NG=56]` but stored **TRANSPOSED**
  group-major `[NG, M]` contiguous: `offset_floats(m,g) = g*M_pad + m`
  (`a1_scale.t().contiguous()` asm:159; `transpose_scale=True` fused_moe.py:615).
- **Sort indirection** the GEMM also consumes: `sorted_ids`, `sorted_expert_ids`,
  `num_valid_ids`, `sorted_weights` (the `moe_sorting` output). moe_op.py:173.

### 2.1 THE SCALE TRAP (token-major vs group-major) — most important byte-layout fact
IRISX dispatch (`moe_dispatch_pack_quant.hip:129,164`) writes scales **token-major**:
`packed_sc[base*N_GROUPS + g]` — for a fixed token, the 56 group-scales are contiguous.
Production fmoe reads **group-major**: for a fixed group `g`, all tokens' scales are
contiguous (`scale[g*M_pad + m]`). **These are transposes of each other.** Any kernel that
reuses the IRISX pack MUST transpose, or it will silently read wrong scales (no crash —
numerically garbage). FMOE_LAYOUT §2,§5,§6 row #5. Confirmed source; exact `M_pad` not.
- **[NEEDS-NODE-MAIN-AGENT]** exact token-dim padding `M_pad` of the transposed scale
  (suspected pad to GEMM `block_size_M`=32):
  ```
  docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
    grep -rniE "block_size_m|blk_m|pad|partial_transpose|shuffle_scale|m_pad" \
    /app/aiter-test/aiter/ops/quant.py /app/aiter-test/aiter/fused_moe.py'
  ```

### 2.2 Value-buffer framing mismatch
IRISX keeps an **expert-major** pack `[local_e][src_rank][slot][H]`
(`moe_dispatch_pack_quant.hip:126-128`). Production fmoe wants **row-major `[token,H]` +
sort map**. To feed real fmoe either (a) emit A as `[token,H]` and produce
`sorted_ids/sorted_expert_ids/num_valid_ids`, or (b) treat the expert-major pack as input to
**our own** GEMM (the V2+ plan). FMOE_LAYOUT §6 row #6. This ABI standardizes the metadata
for path (b) and gives `route_reverse` to bridge back to (a) for combine.

---

## 3. Finalized structs (authoritative bytes in `route_abi.h`)

### 3.1 Inherited verbatim from AGENT_COMMON §3 (unchanged)
```c
struct route_segment {           // one contiguous run: one expert, one source rank
    int expert_id;               // local expert index [0,32)
    int src_rank;                // owning rank of the rows [0,8)
    int src_row_begin;           // first row in src rank's activation buffer
    int dst_row_begin;           // first row in this expert's packed region
    int row_count;
};
int expert_offsets[33];          // prefix sum: offsets[e]..offsets[e+1] = expert e's rows
int rows_per_expert[32];
struct expert_task {             // flattened GEMM work unit (Agent 02)
    int local_expert;
    int m_tile_begin;            // row offset within expert region (multiple of BM)
    int valid_rows;              // <= BM, tail mask
    int n_superblock;            // NSUB-wide N panel
    int segment_begin;           // index into route_segment[]
    int segment_count;
};
```

### 3.2 Agent-00 ADDITIONS (new — the deltas vs AGENT_COMMON)

**(a) `route_reverse` — packed-row -> original token (combine/EpCombine needs this).**
One entry per packed row, in packed-row order (so consumers index it by the GEMM output row).
Generalizes the V1 `route_slot[token][topk]` (which is the inverse direction) into the
forward map the combine kernel actually walks.
```c
struct route_reverse {
    int src_rank;        // rank that owns the original activation row [0,8)
    int src_token;       // original token index on src_rank [0, T_local)
    int topk_slot;       // which of the token's top-k picks this row is [0,8)
    float route_weight;  // softmax gate weight for (src_token, topk_slot); combine scales by this
};
```
Relationship to V1 `route_slot`: V1 emits `route_slot[t*TOPK + k] = slot`
(`moe_dispatch_pack_quant.hip:124`) — i.e. token->slot. `route_reverse[packed_row]`
is the inverse: slot/packed_row -> (src_rank, src_token, topk_slot) + weight. Producers that
already write `route_slot` can fill `route_reverse` in the same dispatch pass (one extra
indexed store per assignment).

**(b) `route_params` — the header that makes a dump self-describing.**
Every ABI buffer set is prefixed by exactly one of these (magic-checked).
```c
#define ROUTE_ABI_MAGIC   0x52415831u   /* "RAX1" */
#define ROUTE_ABI_VERSION 1u
struct route_params {
    unsigned magic;          // ROUTE_ABI_MAGIC
    unsigned version;        // ROUTE_ABI_VERSION
    int ep_size;             // 8
    int my_rank;             // [0,8)
    int n_local_experts;     // 32
    int top_k;               // 8
    int hidden;              // 7168
    int group;               // 128
    int n_groups;            // 56
    int fp8_is_ocp;          // 1 = e4m3fn/448 (gfx950); 0 = e4m3fnuz/240 (gfx942)
    int scale_layout;        // 0 = token-major (IRISX native); 1 = group-major (production fmoe)
    int m_pad;               // token-dim pad of group-major scale; 0 if unknown [NEEDS-NODE]
    int total_packed_rows;   // = expert_offsets[32]
    int t_local;             // tokens this rank dispatched
};
```
`scale_layout` makes the §2.1 trap explicit in-band: a consumer asserts the layout it needs
and transposes if `scale_layout==0` but it wants `1`.

---

## 4. W13 / W2 shapes summary (binding for the GEMM agents)

| GEMM | A (we emit)                | B (preshuffled, not ours)         | K    | N    | epilogue          |
|------|----------------------------|-----------------------------------|------|------|-------------------|
| fc1  | fp8 `[M_e,7168]` row-major | `w13[e,4096,7168]` fp8 +128x128 sc | 7168 | 4096 | split 4096->2x2048, `SiLU(gate)*up` -> 2048 |
| fc2  | fp8 `[M_e,2048]` row-major | `w2[e,7168,2048]` fp8 +128x128 sc  | 2048 | 7168 | none -> bf16 out  |

fc1 N-split order and fc2 fresh cite are **[NEEDS-NODE]** (§1.1, §1.2).

---

## 5. What EpCombine needs (reverse path)

Combine = scatter each expert's GEMM output row back to its origin token and accumulate the
top-k contributions, scaled by the gate weight. Required inputs, all in this ABI:
1. `route_reverse[total_packed_rows]` — for each packed/output row: `(src_rank, src_token,
   topk_slot, route_weight)`. This is the ONLY mapping that lets combine find the destination.
2. `expert_offsets[33]` — to know each expert's output-row span (combine can stream per expert).
3. The bf16 GEMM output `[total_packed_rows, H]` (fc2 result), packed-row order.
4. Accumulator `[T_local, H]` bf16/fp32 on the **origin** rank; combine does a remote
   accumulate `acc[src_token] += route_weight * out[packed_row]` over the row's top-k slots.

Notes / gaps:
- The forward dispatch already has the weights at routing time (softmax gates). Carrying
  `route_weight` into `route_reverse` avoids a second gather in combine.
- **[NEEDS-NODE-MAIN-AGENT]** does production EpCombine expect a specific reverse index tensor
  name/dtype (so we can match it exactly rather than roll our own)?
  ```
  docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
    grep -rniE "combine|unpermute|scatter|moe_sum|topk_weight|sorted_weights" \
    /app/aiter-test/aiter/fused_moe.py /app/ATOM/atom/model_ops/moe.py | head -40'
  ```
- **[NEEDS-NODE-MAIN-AGENT]** confirm `sorted_weights` (the value passed at moe_op.py:173) is
  exactly the per-(sorted-row) gate weight we put in `route_reverse.route_weight` (so our
  combine matches production semantics):
  ```
  docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
    grep -rniE "sorted_weight|topk_weight|moe_sorting|sort" \
    /app/aiter-test/aiter/fused_moe.py | head'
  ```

---

## 6. Graph-capture / pointer-stability notes

The production server runs R1 under CUDA/HIP graphs (decode is graph-captured). For our
kernels and any instrumentation to be graph-safe:
- **All ABI buffers must be allocated once and pointer-stable across steps.** During capture,
  pointers baked into the graph are reused every replay; reallocating per step or using a
  bump-allocator that returns new addresses breaks capture. Allocate the symmetric-heap
  buffers (`packed_fp8`, `packed_sc`, `route_*`, `expert_offsets`, ...) up front for `T_MAX`
  and reuse (this is what the V1 bench already does — `moe_dispatch_pack_quant.hip:241-254`).
- **No host-visible control flow inside the captured region** (no `hipMemcpy` D2H of counts to
  decide grid size). Grid is `device_props.multiProcessorCount`, fixed (V1:219). Counts
  (`send_counts`, `assign_slot`) must be device-resident and zeroed with `hipMemsetAsync` on
  the captured stream, not host-synchronized.
- **Instrumentation must be no-op-when-disabled at compile time** (template param, like V1's
  `PROFILE`), so the enabled path never changes the captured graph's kernel set. See
  ROUTE_CAPTURE_SCHEMA.md — capture writes to a pre-allocated ring buffer, never allocates.
- **[NEEDS-NODE-MAIN-AGENT]** confirm the R1 server actually graph-captures the MoE region
  (vs eager) and the capture batch buckets, so our T sweep matches:
  ```
  docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
    grep -rniE "graph|capture|cudagraph|hipgraph|capture_size|batch.*bucket" \
    /app/ATOM/atom/ | grep -iE "moe|graph" | head'
  ```

---

## 7. Open [NEEDS-NODE-MAIN-AGENT] items (consolidated)

| # | Gap | Grep / check (node, GPU disabled) | §ref |
|---|-----|-----------------------------------|------|
| Q1 | fc1 fused-N split order (gate\|up vs interleaved) | §1.1 grep | §1.1 |
| Q2 | W2 K=2048,N=7168 fresh source cite | §1.2 grep | §1.2 |
| Q3 | transposed-scale `M_pad` (pad to block_size_M=32?) | §2.1 grep | §2.1 |
| Q4 | EpCombine reverse-index tensor name/dtype | §5 grep | §5 |
| Q5 | `sorted_weights` == our `route_weight`? | §5 grep | §5 |
| Q6 | R1 actually graph-captures MoE + batch buckets | §6 grep | §6 |
| Q7 | per-(expert,src_rank) **capacity** policy (drop vs pad; V1 uses PER_SRC_CAPACITY=64) | `grep -rniE "capacity\|drop\|num_valid\|max.*token" /app/aiter-test/aiter/fused_moe.py /app/ATOM/atom/model_ops/moe.py` | — |
| Q8 | EpCombine metadata exact shapes (does it need expert_offsets or its own?) | `grep -rniE "expert.*offset\|cu_seqlen\|num_tokens_post\|expert_ids" /app/aiter-test/aiter/fused_moe.py` | §5 |
| Q9 | graph-capture pointer stability of the production heap (do they reuse one allocation?) | `grep -rniE "register\|symm\|heap\|persistent\|static.*buffer" /app/ATOM/atom/ | grep -i moe` | §6 |
| Q10 | source-rank pre-grouping (does production pre-group A by src_rank before GEMM, or rely purely on sorted_ids?) | `grep -rniE "src_rank\|source.*rank\|per.*rank\|group.*rank\|all2all\|dispatch" /app/aiter-test/aiter/fused_moe.py /app/ATOM/atom/model_ops/moe.py | head` | §2.2 |

Q1-Q6 block exact production-byte fidelity; Q7-Q10 block the combine + multisource (Agent 03)
generalization. None block the IRISX-internal GEMM agents (01,02,04-08), which consume the
structs in §3 directly.
