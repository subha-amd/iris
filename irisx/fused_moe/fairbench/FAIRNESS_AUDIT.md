# Fairness audit — `gather_pack` vs `EpDispatch + moe_sorting`

**Status: IN PROGRESS (2026-07-08).** Source findings below are verified against the code and are
final. **Timing numbers are not yet collected** — every `TBD` is a number this document must not
assert until it has been measured on the 8× MI350 node (`smci350-rck-g03-f16-03`).

The june-30 deck claims the three fused kernels give **1.56× over the unfused AITER/MORI baseline**
(prefill, `1247 µs` vs `1941 µs`). This document establishes what that number actually compares, and
rebuilds the comparison so that **the inputs, the outputs, the routing, and the hardware occupancy are
the same on both sides.**

---

## 0. The headline problem the user raised

`gather_pack_kernel` (`kernel.cpp:108`) and MORI's `EpDispatchIntraNodeKernel`
(`mori_epdispatch_ref/intranode.hpp:84`) **do not start at the same place.**

| | `gather_pack` | `EpDispatch` |
|---|---|---|
| activations in | `A_src[Msrc,7168]` **fp8 e4m3** + `A_src_sc[Msrc,56]` fp32 | `tokens[T,7168]` **bf16** |
| routing in | `SEG[Nseg,5]` + `TILE[Ntile,4]`, **host-precomputed** | `topk_ids[T,8]`, `topk_weights[T,8]` |
| direction | **pull** (`ctx.load` from peers) | **push** (`WarpCopy` to peers) |
| dedup | none — one packed row per `(token,expert)` pair | **yes** — one send per `(token, dest_PE)` |
| out | `A_pk[Mpacked,7168]` fp8, expert-major, BM=256-padded | `dispatchOut[R,7168]` bf16, arrival order |
| | | + `outIndices[R,8]`, `outWeights[R,8]`, `totalRecvTokenNum` |

So the fused kernel is handed (a) already-quantized activations and (b) an already-computed routing
plan, and it produces a different artifact. That is three separate unfairnesses, and there are more.

---

## 1. The five fairness breaks (all source-verified)

### 1.1 ⚠️ The fused region is timed with **only one rank executing**

`irisx/fused_moe/example.py:643-649`:

```python
def timed():
    for _ in range(WARMUP):
        if rank == CONSUMER:          # CONSUMER = 7
            _region_once()
    ...
    for _ in range(ITERS):
        if rank == CONSUMER:
            _region_once()
            ...
        else:
            torch.cuda.synchronize()  # ranks 0..6 do NOTHING
        iris.barrier()
```

Rank 7 gathers from seven **idle** peers. No all-to-all contention, no straggler, no cross-rank
serialization. `baselines/b3_ep8_unfused.py:124-145` runs the real 8-way collective on all ranks and
reduces with `MAX` over ranks (the production denominator — the step waits for the slowest rank).

This is the single largest threat to the `1.56×`, and it is independent of the ABI question.

### 1.2 The fused path never pays the input quantization

`example.py:352-355` quantizes the source activations **once, on the host, before the timed loop**:

```python
A_real = (gnp.standard_normal((MSRC, K)).astype(np.float32) / 8.0)
q_u8, sc_f32, deq_f32 = RT.quantize_v1(A_real, K)     # numpy, off the clock
A_src_fp8.copy_(...)
```

The baseline dispatches **bf16** and pays `dynamic_quant` *inside* `fused_moe`
(`aiter_ref/fused_moe.py:1766-1773`, `quant_func(hidden_states, ..., num_rows=num_local_tokens)`).
So the fused region gets fp8 activations for free — which also **halves the bytes it moves over XGMI**.

### 1.3 The routing plan is built on the host, off the clock

`b1_dispatch_route.build_multisource_route()` runs in numpy on the host. A real router cannot do this:
`topk_ids` exists only on the GPU and changes every token, every layer, every decode step.

MORI builds the placement **on device, inside the dispatch kernel** — `intranode.hpp:145-170`: a warp
per `(token, top-k slot)`, dedup by destination PE via `__any()`, then `atomicAdd` slot assignment into
`dispTokOffsetMemObj`. `moe_sorting` likewise builds its index lists on device.

MORI *does* expose a cached/replay path (`EpDispatchCombineRoutingPtrs`, `dispatch_combine.hpp:96-111`;
`dispatch(..., routing=...)`), which is the honest analog of a precomputed `SEG`/`TILE`. Hence two tiers:

- **tier 1 (plan amortized):** MORI replay dispatch vs `gather_pack` with a precomputed plan.
- **tier 2 (full cost):** MORI cache-mode dispatch vs `gather_pack` + an **on-device** plan builder.

Tier 2 is what production pays. A tier-2 plan builder (`plan_count/scan/scatter` +
`plan_allgather_ids`) has been added to `kernel.cpp` for this audit.

### 1.4 The synthetic route flatters the gather

`build_multisource_route` walks a **monotone per-rank cursor** and emits contiguous runs of 1..40 rows.
Two consequences a real top-8 router does not reproduce. Measured by `fairbench/real_route.py` against
actual `fused_topk` output (`world=8`, `T=1024`, `E=256`, `topk=8`):

| property | synthetic route | **real top-8 route** |
|---|---|---|
| mean `route_segment` run length | ~20.5 | **1.03** |
| single-source tiles (Path-2 fast path fires) | many | **3 / 200** |
| distinct source tokens per routed row (`dup`) | **1.000** (⇒ top-1, not top-8) | **1.506** |
| BM=256 padding (`ROUTE=uniform`, `TOTAL_M=8192`) | **0.0 %** | **35.4 %** |

Three things follow:

- **`tile_is_single_source()` (`ep8_gather.h:95`) is dead code under real routing.** Every tile takes
  Path 1, whose `build_row_seg_map()` (`ep8_gather.h:116-133`) is a **serial loop over `seg_count`**
  segments — 64 iterations per 64-row tile, one active thread each.
- **A pull gather re-reads each distinct token 1.5×.** MORI's push dedups (`intranode.hpp:141-152`);
  a pull cannot. The synthetic route hides this by never routing a token to two local experts.
- **`ROUTE=uniform` with `TOTAL_M=8192` and `E=32` puts exactly 256 rows in every expert** — precisely
  one BM=256 tile, zero padding. Real routing gives `rows_per_expert ~ Binomial(8192, 1/32)`, mean 258,
  σ≈15.8, so **most experts land just over 256 and pad to 512.** `Mpacked` goes 8192 → 12800.
  The prefill GEMM (`build_b0_tasks`, `n_m_tiles = ceil(m_e/256)`) then does **1.56× more MFMA work.**
  That is the same size as the claimed win. (Decode is unaffected: `build_b0_tasks_decode` tiles at
  BM=16 over the same 256-padded buffer, so it computes `ceil(m_e/16)*16` rows, not the padded region.)

### 1.5 The two sides do not produce the same artifact

`moe_sorting` writes **index arrays only** (`sorted_token_ids`, `sorted_expert_ids`, `num_valid_ids`)
plus a zeroed `moe_buf`. The unfused fmoe GEMM applies the permutation **for free, in its A-load**.
Nothing is materialized.

`gather_pack` physically writes `Mpacked × 7168` fp8 bytes into a new expert-major buffer, which the
downstream GEMM then re-reads from HBM.

So "`gather_pack` replaces `EpDispatch` + `moe_sorting`×2" is not a kernel-for-kernel identity. The
only defensible comparison boundary is **"everything between the router and the fc1 MFMA"**, with the
bytes each side moves reported alongside the µs.

> Note: MORI already ships a **fused dispatch+sort** that *does* emit an expert-major packed buffer —
> `dispatch_standard_moe()` → `EpDispatchIntraNodeKernel_<T>_stdmoe` → `packedRecvX`
> (`dispatch_combine.py:1192`, `intranode.hpp:83` `EnableStdMoE`). It is **not compiled** in this
> container (`set_standard_moe_output_buffers` resolves to `None`, needs `ENABLE_STANDARD_MOE_ADAPT=ON`).
> That is the true "already-fused" competitor and the honest thing to race against.

---

## 1.6 ⚠️⚠️ `combine_pull` silently drops contributions under real EP routing

This is the most serious finding in this document, and it is about correctness, not fairness.

`combine_pull_kernel` (`kernel.cpp:1971`) delegates to `tilecomm::tile_reduce_scatter`
(`irisx/tilecomm/tilecomm_device.h`). One block owns one **destination cell** `(dst_rank, dst_token)`,
reduces that cell's **local** packed rows in fp32, and then:

```c++
// tilecomm_device.h:151   (and :123, :135 for the bf16 / bf16x2 store granularities)
if (local) *d = out.v; else ctx.store<uint4>(d, out.v, dst_rank);
```

A **plain store**, not an atomic accumulate — the header says so explicitly (`:77`):
*"No atomics — a private accumulator per (tile, element)."*

That is correct **only if every contribution to a destination token lives on a single producer rank.**

Measured against a real top-8 router over 256 experts, 32 per rank (`real_route.combine_fanout`):

```
distinct expert-owner ranks per origin token:  mean = 5.327,  min = 2,  max = 8
fraction of tokens with fanout > 1:            100.0 %
histogram (fanout 0..8): [0, 0, 5, 127, 1223, 3363, 2801, 649, 24]
```

**Every token** has its 8 experts spread across ≥2 ranks. So ≥2 producer ranks each compute a *partial*
sum for the same `accb[dst_token]` and each `ctx.store`s it — last writer wins, and on average **4.3 of
the 5.3 partial sums are discarded.**

Two accidents of the benchmark hide this completely:
1. `example.py` runs the region on **one rank** (§1.1), so no cross-rank store ever collides.
2. `build_multisource_route` gives every token **fanout == 1** (§1.4), so even multi-rank execution
   would not collide.

`COMBINE_TLOCAL=N` (`b1_dispatch_route.build_route_reverse`) does force top-k collisions — but only
*within* one rank's packed rows, which the local fp32 reduce handles correctly. It does not test the
cross-rank case at all.

**Consequence for the claim "our pull-combine beats AMD's own MORI EpCombine (386 vs 398 µs)":**
MORI's `EpCombineIntraNodeKernel` runs on the *origin* rank and **pulls and accumulates** each token's
contributions from every peer's staging buffer (`intranode.hpp:674-698`, `srcPtrs[j]` over `destPe` +
`core::WarpAccum`). Our kernel runs on the *producer* rank and pushes one pre-reduced row. It is not
doing MORI's job. The `combine_scatter` variant (`kernel.cpp`, IRIS `ctx.fetch_add`, **788 µs**) *is*
cross-rank-correct — and it loses to MORI's 398 µs.

### ✅ CONFIRMED ON DEVICE (8× MI350, 2026-07-08)

`fairbench/probe_combine_fanout.py` — 8 ranks, no aiter, no MORI, just IRIS + `tk_kernel.combine_pull`.
Every rank contributes one row of value `r+1`, weight 1.0, to the **same** cell `(dst_rank=0, dst_token=0)`.

```
combine_pull cross-rank accumulation probe   (store_gran=8, H=512, world=8)
  case A  fanout=8  expected accb[0,0] = 36.0   got = 8.0   -> FAIL — contributions DROPPED
  case B  fanout=1  expected accb[0,0] =  1.0   got =  1.0   -> PASS
```

Correct at fanout 1, silently wrong at fanout 8 — it kept exactly one rank's partial sum. Under real
top-8 routing that is **every token**.

**Implication for the deck:** the `386 µs` pull-combine is not a valid `EpCombine` replacement. The
cross-rank-correct kernel you already have — `combine_scatter` (`ctx.fetch_add`, **788 µs**) — is the one
that does MORI's job, and it *loses* to MORI's 398 µs. The combine result must be withdrawn or re-derived.

---

## 2. Two bugs found in the baseline harness (these *inflate* the baseline)

Both make the unfused side look slower than it is, i.e. they inflate the reported `1.56×`.

### 2.1 `b3` over-sizes MORI's recv buffer, which over-sizes `moe_sorting`

`baselines/b3_ep8_unfused.py:111`:

```python
max_num_inp_token_per_rank=max(8192, tokens_per_rank * 4),
```

At decode (`TOKENS_PER_RANK=64`) that is **8192**, so `MaxNumTokensToRecv() = 8 × 8192 = 65536` while
only ~336 tokens actually arrive. `fused_moe` then calls `moe_sorting(topk_ids=di, ...)` with
`di.shape[0] = 65536`, and in `aiter_ref/csrc/include/moe_sorting_opus.h` the **static** token count
`h.tokens` (not the runtime `p_local_tokens[0]`) drives:

- `moe_sorting_is_oneshot(tokens_, num_experts_)` — **which kernel is selected at all** (`:1382`);
- `moe_sorting_get_workspace_size(tokens_, ...)` → `moe_sorting_mp_mesh_stride(tokens)` = `pad32(65536)`,
  so the `[num_experts × 65536]` mesh workspace is **allocated** every call (`:1132-1138`, `:1378`);
- `k.mesh_stride`, `k.smem_rows`, `k.tokens_per_thread`, and the launch grid (`:536-541`).

What *does* honour the real count: the `moe_buf` zeroing (`tokens_ = p_local_tokens[0]`, `:1087-1103`)
and — via the `is_local_token` branch of `maybe_clear_workspace` (`:3528-3536`) — the workspace clear.

So the *allocation, kernel selection, LDS footprint and grid* are sized for 65536 tokens on every decode
step, while the actual data movement is not. **Whether that costs measurable µs is an empirical
question** — `bench_dispatch_prefix.py` runs `MAX_INP=real` (=`T`) and `MAX_INP=b3` (=`max(8192,4T)`)
so the delta is measured, not assumed. A real serving stack sets `max_num_inp_token_per_rank` to the
actual max batch.

### 2.2 ⚠️ MORI's `totalRecvTokenNum` accumulates across dispatches — so `b3`'s `fmoe` time grows every iteration

- `intranode.hpp:236`: `atomicAdd(args.totalRecvTokenNum, recvTokenNum)` — every dispatch **adds**.
- `dispatch_combine.cpp:328-329`: `hipMemset(totalRecvTokenNum, 0, ...)` — **once, at handle construction.**
- `dispatch_combine.cpp:460`: `void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {}` — **an empty stub.**
- `dispatch_combine.py`: `dispatch()` never calls `_reset_func`; only `combine(call_reset=True)` does,
  and `call_reset` **defaults to `False`** (`:864`).

`b3`'s `timed_iter()` (`b3_ep8_unfused.py:136-145`) calls `op.dispatch(...)` each iteration and hands the
returned `drn_` **view** straight to `fused_moe(..., num_local_tokens=drn_)`. Since nothing zeroes the
counter, `drn_` reads `iteration × R`. Over 10 warmup + 50 timed iterations, `moe_sorting`'s
`num_local_tokens` climbs to ~60×R, so its `moe_buf` zeroing and `moe_align` work grow linearly and the
**median is taken over a monotonically increasing sequence.**

**STATUS: source-verified, awaiting empirical confirmation** (`probe_drn.py` — print `drn` after
successive dispatches). Do not put this in the deck until measured.

---

## 3. The corrected benchmark

`fairbench/bench_dispatch_prefix.py`. One process per rank under `mpirun -np 8`, hosting IRIS +
`tk_kernel` + MORI + aiter together, so both paths race on the **same tensors, same GPUs, same iteration**.

- **Inputs (both):** `tokens[T,7168]` bf16, `topk_ids[T,8]`, `topk_weights[T,8]` from real `fused_topk`.
- **Boundary (both):** fc1's A operand is ready.
- **Every rank runs every stage, every iteration. Median over `ITERS`, then MAX over the 8 ranks.**
- **Correctness gate** (timings are meaningless without it):
  1. `A_pk[dst]` bytes == `tokens[src_rank][src_token]` fp8 bytes, for every routed row;
  2. `A_pk_sc[dst]` == the source scale;
  3. every padding row of `A_pk` is exactly zero (the zero-sentinel guarantee);
  4. the `expert → {(src_rank, src_token)}` map is **identical** to MORI's, recovered from
     `get_dispatch_src_token_pos()` (= `FlatTokenIndex(pe, tok)`, `common.hpp:31`).

Stages timed:

| stage | side |
|---|---|
| `quant[T]` — aiter `per_1x128` of `tokens[T,H]` | fused (and unfused-fp8) |
| `EpDispatch bf16` (routing on device) | unfused, tier 2 |
| `EpDispatch bf16 REPLAY` (plan cached) | unfused, tier 1 |
| `quant[R]` — aiter `per_1x128` of `dispatch_out[R,H]` | unfused-bf16 |
| `EpDispatch fp8` + scales | unfused-fp8, tier 2 |
| `EpDispatch fp8 REPLAY` | unfused-fp8, tier 1 |
| `moe_sorting` | unfused |
| `gather_pack` (SEG/TILE, plan precomputed) | fused, tier 1 |
| `gather_pack_rowmap` (flat plan) | fused — new, see §4 |
| `plan_allgather_ids` + `build_plan` (on device) | fused, tier 2 — new, see §4 |

### Environment traps discovered while bringing this up

1. **`shmem_torch_process_group_init()` deadlocks under `mpirun`** — its uid broadcast goes through a
   torch.distributed NCCL collective (`mori/shmem/api.py:113-121`). Use `shmem_get_unique_id()` + an
   mpi4py `bcast` + `shmem_init_attr(MORI_SHMEM_INIT_WITH_UNIQUEID, ...)`.
2. **`shmem_mpi_init()` is not compiled** into this container's `mori_cpp` (`AttributeError`).
3. **Orphaned processes wedge MORI.** A `b3` run that crashed 18 h earlier
   (`~/b3_t64_retry.log`: `HIP failure: 'invalid argument'`) left 7 workers reparented to PID 1; every
   subsequent MORI bootstrap on the node spun forever. `rocm-smi --showmemuse` also showed 80% VRAM held
   by defunct `sglang::schedul` processes whose container PID 1 (`sleep infinity`) never reaped them.
   Both cleared; `probe_boot.py` then bootstraps MORI repeatedly, back-to-back, without issue.
4. **`shmem_init_attr()` still hangs inside `bench_dispatch_prefix.py` — ROOT CAUSE NOT YET FOUND.**
   All 8 ranks spin at 100% CPU, 0% GPU, stack pinned at `mori/shmem/api.py:157`. `probe_boot.py`
   performs the *same* call sequence and succeeds. Ruled out so far: orphan processes (node is clean);
   IRIS heap size (hangs at both 256 MB and 2048 MB); **`import aiter` ordering** (moving every aiter
   import after the bootstrap did *not* fix it — an earlier hypothesis, now disproven). Under
   investigation; the remaining candidate is `import torch` vs `import iris_py`/`tk_kernel` ordering
   (`example.py` and `probe_boot.py` both import torch first, and both work).

---

### ⚠️ A build-system trap worth knowing

The shared `HipKittens/distributed-kernels/build/` cache was configured with
`DK_BUILD:STRING=b1_tilecomm` (left over from the 07-07 TileComm session). `add_dk_kernel()` is only
called for the *selected* `DK_BUILD`, so `~/do_build.sh` (`cmake --build build`) returned **rc=0 in zero
seconds and rebuilt nothing** — `b1_dispatch` is not even a target under that cache. The stale Jul-7
`tk_kernel.so` sat there looking like a fresh successful build.

**Always run `~/cfg_build.sh` (which passes `-DDK_BUILD=b1_dispatch`) before `~/do_build.sh`.** Any
"rebuild" of `b1_dispatch` done during the TileComm session may have been a silent no-op.

---

## 4. Kernel additions made for this audit (`kernel.cpp`)

- **`gather_pack_rowmap_kernel`** — identical data movement, but resolves each packed row through a flat
  `rowmap[Mpacked,2] = (src_rank, src_row)` in LDS instead of the `route_segment` run encoding. Under a
  real router the run encoding buys nothing (mean run 1.03) and costs `build_row_seg_map`'s serial scan.
  This is also the only ABI an on-device plan builder can emit — a counting sort with atomics has no
  notion of "runs".
- **`plan_count / plan_scan / plan_scatter` + `plan_allgather_ids`** — build the pull gather's routing
  plan on device (256-padded prefix + atomic slot assignment), mirroring MORI's on-device placement.
  Closes the tier-2 gap.

---

## 5. Results

**TBD — nothing measured yet.** This section must contain, per `T ∈ {64, 1024}`:

- the per-stage MAX-over-ranks table,
- tier-1 and tier-2 prefix totals for `{unfused bf16, unfused fp8, fused SEG, fused rowmap}`,
- XGMI bytes moved and local HBM bytes written by each side,
- the corrected full-region prefill/decode numbers with **all 8 ranks active**,
- a same-node `b3` baseline (the canonical `1941 / 533 µs` came from **thor-4 on Rainier**, a different
  machine; MASTER_HANDOFF §11 warns the cluster varies ~1.8× node-to-node, and `b3` has **never
  completed on this node** — `b3_sweep.log` stops at the RCCL banner, `b3_t64_retry.log` crashed).

Until then, **the 1.56× should not be quoted.**
