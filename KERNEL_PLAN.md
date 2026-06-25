# IRISX MoE Gather Kernel — Plan & Profiling Basis

> What to build, why, and the exact shapes/boundaries it must hit. Grounded in the
> C4/C6 DeepSeek-R1-0528 decode profiling (8×MI355X, ATOM) and the June 23 Simran/Osama
> meeting. Companion data: `kernels-testing/EP_PROFILING_RESULTS.md`,
> `kernels-testing/C4_RESULTS.md`, `kernels-testing/C6_RESULTS.md`, and the Perfetto
> traces in `amd-general/perfetto_traces/`.

## TL;DR — what to write first

Write an **IRISX device-side kernel that collapses the pre-GEMM MoE dispatch-prep chain**.
Today that chain is, in order:
```
EpDispatch → opus_moe_sorting(P0) → opus_moe_sorting(P23) → dynamic_quant → [fmoe GEMM]
```
The framing is **NOT** "fuse 6 kernels into 1." It is: replace the **pre-GEMM data-prep
slice** with one IRISX kernel that routes top-k token activations to expert-owning ranks
and writes them **directly in the expert-major (FP8) layout** the `fmoe` GEMM expects.

- **V0**: `ep_dispatch_pack_bf16` — collapse `EpDispatch → opus_moe_sorting(P0/P23)` into
  one dispatch + expert-major-pack kernel. Two sub-variants (atomic vs precomputed-offset).
- **V1**: `ep_dispatch_pack_quant_fp8` — V0 + fold in the MoE-input quant. Removes the
  sort+quant HBM round-trips **and** ~halves the XGMI payload (bf16→fp8). First real perf target.
- **V2** (research / HipKittens): tile-level dispatch feeding a HipKittens expert GEMM (overlap).
- **V3** (later, harder): fuse expert-GEMM epilogue + `EpCombine` (scatter).

Two framing rules (so we don't overclaim):
1. The win is **NOT** filling idle bubbles — SQL7 shows every stage transition is ~1 ns
   (graph mode already packs the pipeline). The win is **eliminating materialized
   intermediate HBM layouts, reducing graph nodes, and (V1) cutting XGMI bytes**.
2. V1 attacks the **pre-GEMM slice** of the movement/prep envelope, not the whole ~24–25%.
   Use the three-surface breakdown in §1.

Do **not** start by fusing gather into the GEMM — the trace proves the gather's immediate
consumer is the sorter, not the GEMM.

---

## 1. The profiling basis (what the data says)

### Decode MoE loop, per layer (from C4-HT Query 2, invariant over all 300 sampled rows)
```
... grouped_topk ──► EpDispatch(bf16) ──► opus_moe_sorting P0_v2 ──► opus_moe_sorting P23
        ▲                                                                    │
        │                                                                    ▼
   (next layer)                                              dynamic_per_group_scaled_quant
        ▲                                                                    │
        │                                                                    ▼
   wv_splitk ◄── EpCombine ◄── fmoe_bf16_blockscaleFp8 (expert GEMM) ◄───────┘
```
- **EpDispatch (GATHER)** is *always* preceded by `grouped_topk` (router) and *always*
  followed by **two phases** of `opus_moe_sorting` — NOT the GEMM. This is the key
  placement fact: V0 should subsume the sort/pack.
- **EpCombine (SCATTER)** is *always* immediately after `fmoe` (expert GEMM) and before
  `wv_splitk` (the always-on shared expert) → next router.
- `wv_splitk_small_fp16_bf16` is the **shared-expert** path (`n_shared_experts=1`): it runs
  for every token regardless of routing, in parallel with the routed MoE. Not ignorable —
  the combine epilogue ultimately sits alongside it.

### Cost — THREE SURFACES (don't conflate them)
Present these as three distinct numbers so the V1 claim is rigorous:

| surface | what it includes | C4 | C6 | use for |
|---|---|--:|--:|---|
| **Movement** | EpDispatch + EpCombine | **14.2%** | **15.6%** | the comm bottleneck headline |
| **V1 pre-GEMM target** | EpDispatch + sort×2 + MoE-input quant | (see below) | — | what V1 actually replaces |
| **Full MoE fusion envelope** | movement + sort + quant (+ maybe combine) | ~24.5% | ~24% | long-term project surface |

8-rank aggregated category share (graph mode, production-faithful):
| | C4 (TP4×DP2) | C6 (DP8) |
|---|--:|--:|
| EP_dispatch (gather) | 6.18% | 7.30% |
| EP_combine (scatter) | 8.05% | 8.32% |
| TP collective (allreduce/RS) | ~0 | **0.00%** |

- C6 has **zero** TP collectives → gather/scatter is the *only* inter-GPU comm, attribution
  is clean. C4 and C6 agree → the gather is intrinsic to EP decode.
- **V1 must NOT claim the full ~24%.** It does not replace EpCombine, and it must NOT count
  the dense/attention quant kernels (SQL3 below shows only ~19% of all `dynamic_quant` calls
  are MoE-input). Slide phrasing: *"the full EP movement/prep envelope is ~24–25%, but V1
  specifically targets the pre-GEMM slice — dispatch, expert-major pack, and MoE-input quant."*

### Measured V1 pre-GEMM envelope (C4-HT rank0, SQL1/SQL2 — the real V1 target)
Window = [EpDispatch → first following fmoe). Per-instance, averaged over 85,318 instances:

| stage | avg µs/instance | % of pre-GEMM envelope |
|---|--:|--:|
| EpDispatch (gather) | 29.2 | 63.4% |
| opus_moe_sorting (×2 phases) | 11.0 | 23.9% |
| dynamic_quant (MoE-input) | 5.8 | 12.7% |
| **pre-GEMM TOTAL** | **~46.0** | 100% |

So **V1's per-call target is ~46 µs, not ~29 µs.** Distribution is tight (p50 43 µs, min 32 µs;
the 1097 µs max is an inter-decode-step artifact, not a real tail). For context, the `fmoe`
GEMM that follows is ~129 µs — so the pre-GEMM prep is ~26% of (prep + GEMM) per layer.

**SQL3 — quant context (only MoE-input quant is in scope):**
| quant context | calls | avg µs | note |
|---|--:|--:|---|
| dense_or_attention_quant | 269,864 | 6.04 | NOT V1 scope |
| other_quant | 89,304 | 5.95 | NOT V1 scope |
| **moe_input_quant (→fmoe)** | **85,318** | **5.83** | **V1 scope only** |

**SQL4 — `fillBuffer` is NOT in the MoE path.** It appears only `before_dense_gemm` (GEMM
workspace init), never between MoE quant and fmoe. → **V1 must not claim to remove fillBuffer.**

- Per-call: dispatch ~29 µs (p99 ~61 µs), combine ~22 µs (p99 ~44 µs). All-rank report:
  combine is bigger and tail-heavier than dispatch.
- Ranks balanced to 0.1% → **optimize average movement, not worst-rank.**

### Where the win comes from (CONFIRMED via SQL 7 — transition gaps, C4-HT rank0)
The full MoE chain runs **back-to-back with ~1 ns gaps** in graph mode — there are NO idle
bubbles to reclaim. Measured consecutive-stage gaps (85,318 transitions each):

| transition | avg gap | avg prev µs | avg cur µs |
|---|--:|--:|--:|
| grouped_topk → EpDispatch | 0.001 µs | 5.71 | 29.16 |
| EpDispatch → opus_moe_sorting | 0.001 µs | 29.16 | 5.27 |
| opus_moe_sorting → opus_moe_sorting | 0.001 µs | 5.26 | 5.66 |
| opus_moe_sorting → dynamic_quant | 0.001 µs | 5.67 | 5.83 |
| dynamic_quant → fmoe | 0.001 µs | 5.83 | 128.80 |
| fmoe → EpCombine | 0.001 µs | 128.80 | 33.46 |

**Conclusion: the win is NOT bubble-filling.** The cost model is:
```
saved ≈ removed intermediate HBM traffic (staging layout written by dispatch, reread by sort,
                                          bf16 packed reread by quant)
      + removed sort/quant kernel launches / graph nodes
      + reduced XGMI payload if FP8 is sent instead of BF16   (V1 only)
      − extra routing / quant / atomic overhead inside the fused kernel
```
This matches Osama's meeting note: the kernels run "in a very serial way, no fusion" — each
is a separate launch doing a full HBM round-trip, which fusion collapses. Concretely the
pre-GEMM HBM round-trips removed are:
```
current:  EpDispatch → (bf16 staging) → sort×2 → (bf16 expert-major) → quant → (fp8+scales) → fmoe
V1:       ep_dispatch_pack_quant_fp8 → (fp8+scales written once, directly) → fmoe
```

Caveats: `EpCombine→wv_splitk` (avg 1116 µs, max 35 ms) and `dynamic_quant→dynamic_quant`
(catch-all across all model layers) are NOT real fusion gaps — they're inter-decode-step
idle and cross-layer aggregation artifacts (the "ms-scale" artifacts Simran warned about).
Ignore them. Also note `fmoe` avg ≈ 129 µs is the single most expensive stage (the expert
GEMM), and the gather (EpDispatch ≈ 29 µs) is ~5× the sort/quant kernels (≈ 5–6 µs each).

---

## 2. Model shapes (from R1-0528 config.json — the trace can't give these)

| Quantity | Value | Kernel role |
|---|--:|---|
| hidden size H | **7168** | per-token vector length / tile width |
| top-k | **8** | routed expert assignments per token |
| routed experts | **256** | global expert count |
| EP_SIZE (TP4×DP2 or DP8) | **8** | → **32 experts/GPU** (`256/8`) |
| MoE intermediate | **2048** | expert GEMM N |
| MoE layers | **58** | (61 − 3 dense); ~85k dispatch calls/decode burst |
| shared experts | **1** | the `wv_splitk` always-on path |
| n_group / topk_group | 8 / 4 | grouped top-k routing |
| quant | **FP8 e4m3, block [128,128]** | group size 128 → 56 scales per 7168 token |

**Message sizes (per token, per top-k slot):**
- BF16 token: `7168 × 2 = 14,336 B`
- FP8 token + scales: `7168 + (7168/128)×4 = 7,168 + 224 = 7,392 B` (~2× smaller)
- Each token issues up to **8** such sends (top-k=8) to its experts' owner GPUs.

---

## 3. IRISX primitives available (from `irisx/include/iris/iris.hpp`)

Header-only, intra-node (IPC over a symmetric heap), C++20, ≤8 GPUs. Device-view API:
- `iris_view.store<T>(ptr, value, remote_rank)` — remote store (the coalesced XGMI write)
- `iris_view.load<T>(ptr, remote_rank)` — remote load
- `fetch_add<T>(ptr, val, remote_rank, order)` — atomic, for slot claiming (collisions)
- `translate(ptr, rank)` = `heap_bases_[rank] + (ptr − heap_bases_[cur_rank_])` — symmetric
  heap address translation (the basis of all cross-GPU access)
- memory orders/scopes (acquire/release, device/system) for ordering the writes

Template to copy: `irisx/benchmarks/all_put.hip` — grid-stride loop of `iris_view.store(...)`
to remote ranks + a GB/s timer. V0 is a smarter all_put: route each token to its
router-chosen destination instead of broadcasting a constant.

**Gap to flag:** IRISX is intra-node IPC only and is NOT yet integrated with HipKittens.
V0/V1 are pure IRISX (tractable). V2 (HipKittens tile loaders reading IRISX symmetric-heap
pointers) is the novel, hard part — that's the contribution, not a quick win.

---

## 4. Kernel specs

### V0 — `ep_dispatch_pack_bf16` (collapse EpDispatch → opus_moe_sorting ×2)
Two sub-variants — build both; the second is the likely performance path.

Inputs (per rank): local token hidden `[T_local, H]` bf16; routing `topk_ids[T_local, 8]`,
`topk_weights[T_local, 8]`; `expert_to_rank[256]`; destination packed buffers on the
symmetric heap in expert-major layout.

**V0a — remote atomic slot-claim (correctness baseline):**
```
expert = topk_ids[t,k];  dst_rank = expert/32;  local_e = expert%32
slot   = iris_view.fetch_add(&count[dst_rank][local_e], 1, dst_rank)   // remote atomic
dst    = packed_base(local_e) + slot*H
for (h = lane; h < H; h += warp) iris_view.store(&dst[h], hidden[t][h], dst_rank)
```
**RISK — this may be the main bottleneck.** Remote `fetch_add` on a hot expert serializes
across XGMI; MoE routing is skewed, so popular experts become contention points. Use V0a
only to establish correctness, not to judge performance.

**V0b — precomputed-offset layout (no remote atomics, likely faster):**
Lay out the packed buffer as `packed[local_expert][source_rank][local_slot][H]` so each
source rank owns a deterministic slice of every destination expert's buffer:
```
dst = packed_base + local_expert_offset + source_rank_slice_offset + local_pos_for_expert
```
Needs send-count / prefix-offset metadata, which means **one small counting kernel before
the data-movement kernel**. That's fine — SQL7 shows graph-mode gaps are ~1 ns, so "one
kernel at all costs" is NOT the goal; a two-kernel design with no remote-atomic bottleneck
can beat a one-kernel atomic design.

Validates (both): symmetric-heap addressing, remote store at scale, the expert-major layout,
static-shape graph compatibility. Benchmark V0a and V0b vs `all_put` GB/s ceiling and vs the
production `EpDispatch + opus_moe_sorting×2` (~40 µs/instance combined).

**Design requirements from prior art (see `PRIOR_ART.md` — RadeonFlow/Gau/PPLX):**
- **`route_slot[token][topk]` is a first-class OUTPUT** (RadeonFlow's `nvl_dst_idxs`). Emit it
  in V0 even though combine is V3 — without it V3 can't find where each expert output landed.
- **Don't hardcode grid size.** RadeonFlow's `NUM_SMS=304` is MI300X; MI355X has **256 CUs**.
  Use `grid = device_props.multiProcessorCount` or a launch arg.
- **Cross-rank completion signal required.** Local stream order ≠ all peers done writing into
  my memory. End the dispatch with a **bulk completion signal** (RadeonFlow's lightweight
  `nvl_signal` ≈ 1 µs, not the full 40–90 µs barrier). Per-expert/tile signaling is V2.
- **Hidden chunking** (RadeonFlow's own `TODO: split token to chunks`): one-wave-per-full-H=7168
  under-utilizes at small decode batch. Test one-wave-per-(token,k) vs per-(token,k,128-chunk).
- **In-kernel timestamp profiling early** (Gau: torch/rocprof unreliable for multi-GPU). Add a
  compile-time profile mode timing slot-claim / load / remote-store / signal.
- **V0b layout matches PPLX** (sender-owned slices avoid sender-side sync) — the likely perf path.

### V1 — `ep_dispatch_pack_quant_fp8` (V0b + fold in MoE-input quant)
On the better V0 layout, quantize each 128-element group to FP8 e4m3 *before* the remote
store, writing the exact `fmoe_bf16_blockscaleFp8` input layout (fp8 + per-128 scales):
```
for each 128-elem group g in token t:
    bf16 tile = hk_load_tile<128>(hidden[t,g])   // HK-style tile load
    scale     = max(|tile|) / FP8_E4M3_MAX
    fp8 tile  = quantize(tile, scale)
    irisx::remote_store_tile(fp8 tile, dst_rank, dst_fp8_ptr)   // HK tile + IRIS store
    iris_view.store(scale, dst_rank, dst_scale_ptr)
```
**Payload win:** bf16 `7168×2 = 14,336 B` → fp8+scales `7168 + 56×4 = 7,392 B` ≈ **51.6%**.

**Positioning (prior art):** quantized EP comm is NOT novel — **MoRI** (production SGLang/MI355X)
already does FP4-dispatch + FP8-combine for MXFP4 R1 (~2.56× round-trip BW cut). So V1's claim is
NOT "quantized dispatch is new"; it's "fold quant into the IRISX device-side tile path and produce
the exact `fmoe_bf16_blockscaleFp8` layout, replacing `EpDispatch→sort→quant`." **Benchmark V1
against MoRI quantized dispatch if ATOM exposes the flag** (see `PRIOR_ART.md`), not just the bf16 path.

**FIVE validations before claiming the full V1 win** (do NOT assert these yet):
| validation | why it matters |
|---|---|
| quant semantics | confirm `dynamic_per_group_scaled_quant` scales depend only on each token's hidden groups (per-128, per-token) — if so, pre-dispatch quant is equivalent. If scale depends on the expert-packed layout, it is not. |
| exact fmoe input layout | `fmoe_bf16_blockscaleFp8` expects a specific fp8+scale memory layout — V1 must reproduce it byte-for-byte. |
| accuracy | run the ATOM gsm8k lm_eval smoke test; compare against bf16-dispatch path. |
| remote fraction | payload cut only applies to *cross-rank* assignments. EP8 expectation ~87.5% remote, but actual routing may be skewed — measure it. |
| bandwidth-boundness | if EpDispatch is latency/atomic-bound rather than byte-bound, halving bytes won't halve time. Confirm via the transport roofline (below). |

### V2 — tile-level dispatch → HipKittens expert GEMM (the HK contribution)
This is where HipKittens actually matters (V0/V1 are pure IRIS, no MFMA). HK GEMM
threadblocks consume expert-major fp8 tiles produced by the dispatch — ideally pulling
remote token tiles into LDS/registers as they compute. Design options:
| design | overlap | difficulty |
|---|---|---|
| sequential fused kernel (dispatch/pack/quant then GEMM over completed buffer) | low | low |
| pipeline across graph nodes (V1 kernel → HK GEMM) | none in-kernel | low — easiest integration |
| producer/consumer workgroups (some move/quant tiles, others MFMA) | real tile overlap | high |
**AMD caveat (HK paper):** do NOT use NVIDIA-style wave specialization inside one workgroup
— AMD's static register allocation means producer waves consume registers without computing.
Prefer workgroup-level specialization or HK's 8-wave ping-pong / 4-wave interleave schedules.
The IRIS/HK bridge primitive to build first: `irisx::remote_store_tile<TileT, TILE_H=128>(...)`.

### V3 — expert-GEMM epilogue + `EpCombine` (later, harder)
Combine is globally bigger/tailier, but fusing into the GEMM epilogue is resource-constrained
(Osama's warning). Needs the post-combine expansion (SQL5/§6) understood first; combine also
must accumulate routed + shared-expert (`wv_splitk`) outputs in the right token order.

---

## 5. Experiments before declaring V1 "the" kernel
**A. IRISX transport roofline** (microbenchmarks, extends `all_put.hip`):
- remote bf16 store (contiguous) vs fp8 store + scale store
- store chunk-size sweep: 128B / 256B / 512B / 1KB / 4KB → GB/s & latency vs message size
- remote `fetch_add` contention: 1 / 32 / 256 counters, varying hot-expert skew
- The 4 cases to separate byte-reduction from time-reduction: (1) bf16 store only,
  (2) fp8 store + scale, (3) bf16 dispatch+pack, (4) fp8 dispatch+pack+quant.

**B. Routing-distribution logging** (Perfetto can't give this — instrument top-k/dispatch):
remote_fraction, tokens-per-expert, hot-expert skew, per source→dest send counts.

**C. Correctness golden path:** V0 packed bf16 vs production `EpDispatch+sorting` output;
V1 fp8 values + scales vs production `dispatch+sort+quant`; then the gsm8k smoke test.

## 6. Open questions for Simran / Osama
1. Order: V0 dispatch+pack → V1 +fp8 quant → V2 HK tile feed. Match the production kernel
   boundary you want, or prioritize the (larger, tailier) combine side first?
2. Is pre-dispatch FP8 quant (V1) numerically acceptable for R1, or must dispatch stay bf16?
3. For V2, target intra-node IRISX (current) only, or plan for the inter-node path too?
4. V0: is the precomputed-offset (V0b) layout acceptable to ATOM/vLLM's graph capture, or do
   they need a single-kernel (V0a-style) node?

## 7. Verification still needed before the final shape-specialized kernel
- ✅ SQL7 (transition gaps) — done: ~1 ns gaps → win is layout/launch removal, not bubbles.
- ✅ SQL1–4 (V1 envelope, quant context, fillBuffer) — done: V1 target ≈ 46 µs/instance;
  fillBuffer is GEMM-workspace (out of scope); only 85,318 quant calls are MoE-input.
- ☐ SQL5 (post-combine expansion) — before proposing any combine-side (V3) fusion.
- ☐ `rocprof-compute` counters on `EpDispatchIntraNodeKernel` (VGPR/LDS/occupancy/XGMI BW)
  — the bandwidth ceiling V0 must approach.
- ☐ Confirm dispatched dtype is bf16 (kernel name implies it) + per-rank token counts.

---

## One-sentence summary
Write a shape-specialized IRISX `dispatch_pack_quant` kernel for R1 EP decode: route top-k
token activations to expert-owning ranks and write them directly into the expert-major
FP8+scale layout `fmoe` expects — the first C++/HIP tile-level communication primitive that
can later feed a HipKittens expert GEMM. The win is removing materialized intermediate HBM
layouts + graph nodes (not idle-bubble removal), plus a ~2× XGMI payload cut from sending FP8.
