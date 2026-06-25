# ROUTE_CAPTURE_SCHEMA — opt-in route instrumentation (no-op when disabled)

Captures the real routing decisions of the production MoE so the IRISX kernels can be driven
and validated against **actual** R1 traffic (vs synthetic). Design goals: (1) zero cost and
zero graph-shape change when disabled; (2) graph-capture-safe when enabled; (3) a compact,
self-describing dump that `route_generators.py` and the GEMM agents can replay.

Companion: `route_abi.h` (the structs), `PRODUCTION_ABI.md` (the ABI), `instrumentation.patch`
(the actual opt-in patch). GPU rule: capture is written by Agent 00; **only the main agent**
runs it on device.

---

## 1. What is captured (per rank, per layer, per step)

One **capture record** per (rank, moe_layer, decode/prefill step). Fields:

| Field | Type | Meaning |
|-------|------|---------|
| `magic` / `version` | u32 | `ROUTE_ABI_MAGIC` / `ROUTE_ABI_VERSION` (RAX1 / 1) |
| `rank` | i32 | EP rank [0,8) |
| `layer` | i32 | MoE layer index in the model |
| `step` | i64 | monotonic forward-step counter (decode token / prefill chunk) |
| `t_local` | i32 | tokens this rank routed this step |
| `top_k` | i32 | 8 |
| `n_local_experts` | i32 | 32 |
| `total_assignments` | i32 | `t_local * top_k` |
| `rows_per_expert[32]` | i32 | how many incoming rows each local expert got (post-route) |
| `expert_offsets[33]` | i32 | prefix sum of rows_per_expert |
| `remote_assignments` | i32 | assignments whose dst_rank != rank (XGMI traffic proxy) |
| `dropped_assignments` | i32 | assignments past capacity (0 if uncapped) |
| `max_expert_load` | i32 | max rows_per_expert (hot-expert / imbalance signal) |

Plus, gated separately (larger, opt-in `ROUTE_CAPTURE_FULL`):
- `topk_ids[t_local*top_k]` i32 — full per-token expert picks (the raw routing matrix).
- `topk_weights[t_local*top_k]` f32 — the gate weights (feeds `route_reverse.route_weight`).
- `route_reverse[total_packed_rows]` — the forward map (see route_abi.h); lets a replay
  reconstruct combine exactly.

The summary record is small (~150 B + 2*132 B arrays); the FULL arrays scale with traffic and
are off by default.

---

## 2. No-op-when-disabled design

Three compile-time levels via macro `ROUTE_CAPTURE` (default 0):
```
ROUTE_CAPTURE = 0   // OFF: every capture call compiles to nothing (empty inline).
ROUTE_CAPTURE = 1   // SUMMARY: per-step summary record into a pre-allocated ring buffer.
ROUTE_CAPTURE = 2   // FULL: SUMMARY + topk_ids/topk_weights/route_reverse arrays.
```
- At `=0` the capture entry points are `static inline` empty bodies, so the optimizer removes
  them and **the captured HIP graph's kernel set is byte-identical** to the uninstrumented
  build. No branch, no buffer, no allocation. This mirrors V1's `template<int PROFILE>` pattern
  (`moe_dispatch_pack_quant.hip:99,137,152,156,165`) which already proved zero-overhead gating.
- At `>=1` capture writes ONLY into a buffer allocated **once at init** (the ring, §3). No
  device-side allocation, no host sync inside the captured region -> graph-safe
  (PRODUCTION_ABI §6).
- A runtime env `ROUTE_CAPTURE_ENABLE=0/1` can additionally gate writes at level>=1 by a single
  predicated store (one cheap `if (g_capture_on)`), so a graph captured with capture compiled
  in can still be replayed with capture effectively off without recapture.

---

## 3. Dump format (compact, self-describing)

### 3.1 On-device ring buffer (capture target)
Pre-allocated `[N_SLOTS]` of fixed-size summary records + a separate FULL arena for level 2.
A device atomic `head` indexes the next slot (`head % N_SLOTS`). Pointer-stable for graphs.
The host drains the ring **outside** the captured region (between graph replays) into the dump
file below.

### 3.2 Binary dump (`route_capture.bin`) — preferred for FULL
Little-endian. File = global header, then a sequence of records.
```
[file header]
  u32 magic         = ROUTE_ABI_MAGIC ("RAX1")
  u32 version       = ROUTE_ABI_VERSION
  u32 record_count
  u32 flags         // bit0: FULL arrays present
[then record_count x]
  struct route_capture_record   (see route_abi.h — fixed 8B-aligned)
[then, if FULL, an arena of variable arrays, each prefixed by (u32 record_index, u32 kind,
  u32 len) where kind in {0=topk_ids,1=topk_weights,2=route_reverse}]
```
Self-describing: a reader validates `magic`/`version`, reads `record_count`, and knows array
layout from `flags` + each array's `(index,kind,len)` prefix. No external schema needed.

### 3.3 JSON sidecar (`route_capture.json`) — for the summary, human/CI-friendly
One object per record (FULL arrays omitted or referenced by offset into the .bin):
```json
{ "magic":"RAX1","version":1,
  "records":[
    {"rank":0,"layer":3,"step":1024,"t_local":32,"top_k":8,
     "total_assignments":256,"remote_assignments":201,"dropped_assignments":0,
     "max_expert_load":19,
     "rows_per_expert":[7,3,...32 ints...],
     "expert_offsets":[0,7,10,...33 ints...]} ]}
```
`route_generators.py` reads either form; CI diffs the JSON summary, replay harnesses read the
.bin (it carries the FULL arrays needed to reconstruct exact routing + combine).

---

## 4. Where the hooks go (for the main agent to wire on node)

The capture points are at the routing boundary, AFTER top-k selection and BEFORE the GEMM:
- one summary emit after `moe_dispatch_count` produces `send_counts`/`assign_slot`
  (counts -> `rows_per_expert`, `expert_offsets`, `remote_assignments`);
- the FULL `route_reverse`/weights emit folded into the dispatch pass that already writes
  `route_slot` (`moe_dispatch_pack_quant.hip:124`) — one extra indexed store per assignment.

For the **production** R1 path the equivalent hook is at the `fused_moe` call (ATOM
`moe.py:609`) right after `moe_sorting` yields `sorted_ids/sorted_expert_ids/num_valid_ids`;
those tensors already contain `rows_per_expert`-equivalent info. **[NEEDS-NODE-MAIN-AGENT]**
confirm the exact tensor to read and add the drain there:
```
docker exec r1_c4 bash -lc 'export HIP_VISIBLE_DEVICES=""; \
  grep -rniE "moe_sorting|num_tokens_post|sorted_expert_ids|num_valid_ids|cu_seqlen" \
  /app/aiter-test/aiter/fused_moe.py /app/ATOM/atom/model_ops/moe.py | head'
```

---

## 5. Correctness checks on a capture

A valid dump satisfies (the replay harness asserts these):
- `sum(rows_per_expert) == total_assignments` (no rows lost) when `dropped_assignments==0`.
- `expert_offsets[0]==0`, `expert_offsets[32]==sum(rows_per_expert)`, monotonic.
- `0 <= max_expert_load <= total_assignments`.
- every `route_reverse[r].topk_slot in [0,top_k)`, `src_rank in [0,ep_size)`.
- `sum_over_ranks(remote_assignments)` matches the known XGMI traffic envelope for the batch.
