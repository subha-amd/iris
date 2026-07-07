# Abstraction A — tile-fused reduce-scatter / all-reduce for the TP4 PREFILL path

> **One-line thesis.** On the TP4 prefill path the all-reduce that follows a GEMM is **59–85% of
> the serial GEMM+AR time** (measured, §0), so fusing it — reducing output tiles *as the GEMM
> produces them* so the short GEMM hides under the long AR — has a real **~1.2–1.7× ceiling**. But
> the two IRIS examples that already fuse a GEMM with a collective (08/09) are **4.7×–280× slower
> than the unfused torch+RCCL baseline** because *both* halves of the substrate are broken: a slow
> Triton streamK GEMM (471 vs 1149 TFLOP/s) **and** a naive one-shot / per-element-atomic AR with a
> per-tile global barrier. Realizing the ceiling requires replacing **both** with (1) a HipKittens
> producer/consumer-warp GEMM body whose epilogue emits tile transfers, and (2) an in-kernel
> reduce-scatter that matches RCCL bandwidth. This is a **high-effort, medium-ceiling, PREFILL-ONLY**
> path. Decode gets nothing (§4).
>
> Status: design only. Every number below is either measured on-node (cited to
> `MEASURED_FINDINGS.md`, 8× MI350X gfx950, TP4, bf16, median-of-30, MAX-over-ranks) or an explicitly
> labelled **cost sketch / model** with its unmeasured inputs flagged.

---

## 0. The measured ceiling this abstraction is designed against (do not inflate past this)

Unfused reference = full-256-CU torch GEMM (per-rank K-slice) **then** a separate RCCL all-reduce —
exactly what production does today. `MEASURED_FINDINGS.md §A`:

| GEMM (M,N,K) — role                        | GEMM 256CU | RCCL AR  | serial GEMM+AR | AR share | **A ceiling = serial/AR** |
|--------------------------------------------|-----------:|---------:|---------------:|---------:|--------------------------:|
| 8192,4608,36864 (example default)          |   0.606 ms | 0.750 ms |     1.269 ms   |  59%     | **1.69×**                 |
| 8192,7168,18432 (R1 down-proj, prefill)    |   0.565 ms | 1.148 ms |     1.641 ms   |  70%     | **1.43×**                 |
| 8192,7168,7168 (R1 attn out-proj, prefill) |   0.252 ms | 1.147 ms |     1.357 ms   |  85%     | **1.18×**                 |
| 1024,7168,18432 (small prefill)            |   0.093 ms | 0.176 ms |     0.262 ms   |  67%     | **1.49×**                 |

The ceiling logic: if each output tile is reduced as it is produced, the reduce-scatter runs
concurrently with the GEMM and the fused op floors at **max(T_gemm, T_ar) ≈ T_ar** (the AR is the
longer of the two on every prefill shape here). So `fused_ideal ≈ T_ar` and the speedup ceiling is
`serial / T_ar` — the last column. **The attn out-proj (1.18×) is the floor and the example shape
(1.69×) the top; the honest headline range is 1.2–1.7×, shape-dependent, and it shrinks toward 1.0×
as the GEMM gets larger relative to the AR.** RCCL bus BW here is ≈150 GB/s (modest — see §5 caveat);
the robust takeaway is the *ratio*, not the absolute BW.

---

## 1. Root cause — why the shipped IRIS substrate (examples 08/09) is uncompetitive

Measured at the 8192,4608,36864 TP4 shape, `--gemm_sms 128` (`MEASURED_FINDINGS.md §B`):

| substrate                                   | fused GEMM+AR | vs unfused 1.269 ms |
|---------------------------------------------|--------------:|--------------------:|
| unfused torch GEMM + RCCL AR (reference)    |     1.269 ms  | 1.0×                |
| ex.09 one-shot all-reduce (IRIS)            |     5.898 ms  | **4.7× SLOWER**     |
| ex.08 atomics all-reduce (IRIS)             |   358.8   ms  | **280× SLOWER**     |

There are **two independent defects**, and either one alone already disqualifies the substrate.

### 1a. The GEMM body is a compiler-pipelined Triton streamK, not a hand-scheduled MFMA loop — 471 vs 1149 TFLOP/s (2.4× off)

All three examples (`gemm_one_shot_all_reduce.py`, `gemm_all_reduce_atomics.py`,
`gemm_all_scatter.py`) share the identical `persistent_gemm_*` body: a persistent streamK loop
(`for tile_id in range(pid, total_tiles, NUM_SMS)`) with `acc += tl.dot(a, b)` over
`BLOCK_M,N,K = 128,128,32`, `num_warps = 4` (ex09) / `8` (ex07/08), `num_stages = 3`. It leans
entirely on Triton's software pipeliner to overlap global loads with the MFMA. On CDNA4 that yields
**471 TFLOP/s vs torch/rocBLAS 1149** — it captures only 41% of the achievable rate. There is **no
LDS ping-pong double-buffer, no A-reuse across N-tiles, and no warp specialization** — precisely the
structure HipKittens' 8-wave body provides and this one lacks.

**Why this alone is fatal, in numbers.** The per-rank K-slice GEMM (K=9216) is
`2·8192·4608·9216 = 6.95e11` FLOP. At 471 TFLOP/s that is **1.48 ms for the GEMM stage alone** —
already **1.16× slower than the entire unfused 1.269 ms serial GEMM+AR.** So even with a *perfect,
zero-cost, fully-overlapped* all-reduce, ex09's GEMM body can never beat unfused. **You cannot fuse
your way out of a 2.4×-slow GEMM.** A competitive fused kernel must first have a GEMM at ~torch rate;
otherwise the fusion is arithmetic that can't close.

### 1b. The all-reduce is a per-tile global barrier + read-from-all-ranks one-shot (09), or a per-element cross-rank atomic RMW (08) — neither is a bandwidth-optimal reduce-scatter, and neither overlaps comm with the next tile's compute

**ex.09 (one-shot, 5.898 ms).** After computing each tile, the kernel:
1. signals **every** remote rank with `iris.atomic_add(tile_completed+tile_id, 1, ..., sem="release", scope="sys")` (lines 208–218), then
2. **spin-waits** on `iris.atomic_cas(tile_completed+tile_id, W-1, 0, ..., sem="acquire", scope="sys")` until all ranks report done (lines 221–234), then
3. **reads the tile back from ALL W ranks** and sums: `for remote_rank in range(world_size): acc += iris.load(C + sub_offset, cur_rank, remote_rank, ...)` and stores to `c_global` (lines 267–271).

Three structural problems: **(a)** a *per-tile, all-ranks, sys-scope barrier* — the GEMM cannot run
ahead of the collective, so there is **zero compute/comm overlap** (the opposite of the abstraction's
goal); **(b)** each rank moves **W× the tile** (read-all) instead of the `(W-1)/W` a reduce-scatter
moves — the reduction is done *redundantly on every rank* (a full all-reduce per rank) rather than
decomposed into reduce-scatter + all-gather; **(c)** the reduce is gated on the slowest rank *every
tile*. Accounting: the slow GEMM is ~1.48 ms, but the measured total is 5.898 ms — **the AR structure
adds ~4.4 ms of pure barrier + read-all-traffic overhead.**

**ex.08 (atomics, 358.8 ms).** Each output *element* becomes a remote read-modify-write:
`iris.atomic_add(c_global + global_offset, c, cur_rank, remote_rank, ...)` to every remote rank
(lines 135–147). Element-granularity fabric atomics → **7.76 TFLOP/s, 280× slower.** Pathological;
included only as the lower bound of what "naive fusion" costs.

**ex.07 (all-scatter)** is the closest *shape* to what we want — each rank `iris.store`s its
N-partitioned output tile to the right offset on every remote (lines 138–150, fire-and-forget bulk
stores, no per-element atomics) — but it does **no reduction** (it is an all-*gather* of
N-shards, not an all-*reduce*), and it still has no producer/consumer split. It is the right
*primitive* (bulk `iris.store`, link-spreadable) attached to the wrong *algebra*.

### 1c. The deeper reason the naive in-GEMM path stalls (the §6 finding, restated)

Even if you bolt a bulk reduce-scatter onto the fast HK body naively, it stalls: `MASTER_HANDOFF §6`
measured **in-kernel gather-under-MFMA at 13× slower (36 vs 466 TFLOP/s)** because the fast 8-wave
body has no producer/consumer warp split and no A-reuse, so a blocking remote transfer stalls the
whole wave. **Overlap is not free; it requires a different GEMM body.** This is the crux of A.

---

## 2. What a competitive version actually needs (the two builds)

### 2a. A HipKittens producer/consumer-warp GEMM body with an async tile-reduce epilogue

Replace the Triton streamK body with the HK 8-wave ping-pong GEMM (register/shared tiles,
double-buffered shared→register, `mma_ABt` MFMA, fp32 accum) that already exists in HK for bf16 and
for fp8/mxfp8 (`kernels/gemm/mxfp8/MXFP8_8wave` = 421 TFLOPS reference, `fp8fp32/FP8_8wave`) — this is
the body our shipped `grouped_b0_gemm` is derived from, so it is a known quantity at ~torch rate. Then
**warp-specialize the epilogue**: producer warps run the MFMA pipeline uninterrupted; a
consumer/DMA warp drains each *completed* output tile into the reduce-scatter (an `iris.store` to the
owner rank + an arrival-counter increment) so **tile i's comm overlaps tile i+1's compute**. HK has
**no communication concept at all**, so this epilogue — an IRIS `store`/counter emitted from inside
an HK kernel's tile loop — is net-new code and the paradigm contribution. The unbuilt piece from
`MASTER_HANDOFF §6` ("a different GEMM body with async A-fill + reuse") is exactly this.

### 2b. An in-kernel reduce-scatter that matches RCCL bandwidth

All-reduce = **reduce-scatter + all-gather**. Partition the M×N output into W owner-chunks; each rank
`iris.store`s its `W-1` non-owned chunks to their owners (bulk, fire-and-forget, **link-balanced** by
round-robining destination chunks across XGMI links — reuse the TileComm Layer-2 schedule that already
earned the combine's 2.4×), each owner sums its arrivals **locally in fp32, no fabric atomics** (unlike
ex08), gated by a **per-owner-chunk arrival counter** (not a per-tile all-ranks barrier — so compute
runs ahead), then all-gathers the reduced chunks. This moves `2(W-1)/W × output_bytes` per rank — for
the example shape, output = 8192×4608 bf16 = 75.5 MB, so `2·(3/4)·75.5 = 113 MB/rank`; at the measured
150 GB/s that is 0.755 ms, matching the RCCL 0.750 ms. **The target is: match RCCL's `(W-1)/W`
byte-movement and its ~150 GB/s — nothing more exotic.** This half is **unproven on IRIS** (the store
path is fire-and-forget and the on-node XGMI probe was issue-bound — `xgmi_probe_results.md`; a valid
in-kernel RS needs store-completion fences + enough bytes/link to back-pressure + MAX-over-ranks
timing before its bandwidth can even be claimed).

**Neither half exists today. (2a) is a known body needing a net-new comm epilogue + warp split;
(2b) is unproven on IRIS. Both are substantial.**

---

## 3. The schedule — "how many tiles to produce before you reduce" as a concrete tunable `G`

Osama's "permutation space" reduces, for A, to primarily **one scalar knob** `G` = *the number of
output tiles the GEMM produces before the consumer warp fires a reduce batch* (plus the destination
ordering within a batch = the TileComm Layer-2 link-balancing). `G` sets the granularity of the fused
reduce-scatter.

**The trade-off `G` controls:**
- **Small `G` (→1, ex09's regime):** minimal drain tail, but tiny messages — per-transfer issue +
  fence + counter latency `L` dominates, XGMI DMA efficiency is poor, and you approach ex09's
  per-tile-barrier disease.
- **Large `G` (→ all tiles, i.e. bulk-synchronous):** full DMA efficiency per message, but the first
  `G` tiles are produced before any comm starts (fill) **and** the last batch's comm can only run
  *after* the GEMM finishes (drain) — so the tail stops hiding under compute and you regress toward
  the serial `T_gemm + T_ar`.

**Cost sketch (model, not measured — inputs flagged).** Let the competitive GEMM take `T_gemm` and
produce `P` output tiles, so per-tile compute `t_c = T_gemm/P`; let the reduce-scatter move a per-tile
comm cost `t_m = T_ar/P` plus a fixed per-*batch* latency `L`. With `P/G` batches:

```
fused(G) ≈  G·t_c                      (fill: first batch produced before comm starts)
          + max(T_gemm, T_ar)          (the overlapped steady state — the ceiling term)
          + G·t_m + (P/G)·L + T_AGtail (drain: last batch's comm + per-batch latency + all-gather tail)

overhead(G) = fused(G) − max(T_gemm, T_ar)  ≈  G·(t_c + t_m) + (P/G)·L + T_AGtail
minimized at   G* ≈ sqrt( P·L / (t_c + t_m) )
```

**Concrete instantiation (example shape, competitive body).** With HK 256×256 tiles the grid is
`(8192/256)·(4608/256) = 32·18 = 576` tiles; a competitive `T_gemm ≈ 0.606 ms` gives
`t_c ≈ 1.05 µs/tile`, and `T_ar ≈ 0.750 ms` gives `t_m ≈ 1.30 µs/tile` — **roughly balanced, which is
exactly the regime where overlap pays.** Taking `L ≈ 3 µs` (**UNMEASURED** — the XGMI probe was
issue-bound, so `L` is a placeholder to be measured on-node), `G* ≈ sqrt(576·3 / 2.35) ≈ 27 tiles` —
i.e. produce **~24–32 tiles (≈ one M-block-row of N-tiles, ~4–5% of the grid)** before firing a
reduce batch. The point is not the exact 27; it is that **`G*` is a small handful of tiles, neither 1
(ex09) nor "all" (bulk-sync)**, and that the optimum is a calibratable `sqrt(P·L/(t_c+t_m))` the
library can pick from the declared demand + a once-measured `L` — turning Osama's "insane permutation
space" into a one-parameter sweep. `T_AGtail` is the residual all-gather that cannot overlap the GEMM
(the last chunk is only fully reduced after the GEMM's final tile) and is the main reason the delivered
win sits **below** the `serial/T_ar` ceiling.

---

## 4. Why this is PREFILL-throughput-ONLY (decode gets nothing)

Under the realistic config **TP4 × DP2 + Expert-Parallel + DP-attention** (`RESEARCH_BRIEF §2`):

- **Prefill (TP4):** a TP all-reduce fires **2×/layer**, and the op immediately before it is a GEMM —
  A's target. Token-rich, GEMM is compute/BW-rich, the AR is 59–85% of the sequence → **A's 1.2–1.7×
  ceiling lives here and nowhere else.**
- **Decode:** attention is **data-parallel (DP-attention), so the TP all-reduce is essentially GONE**
  — there is no all-reduce for A to fuse. The dominant decode collective is the MoE all-to-all
  (gather/combine), a *different* op already fused by the shipped `fused_moe`. Worse, decode is a
  **weight-memory wall**: the expert GEMM streams all 32 local experts' ~1.4 GB fp8 weights/step
  (~176 µs HBM floor) and comm is only ~18% of the region, so *any* comm fusion caps decode at
  ~1.1–1.25× (`MEASURED_FINDINGS §C`). **A contributes 0× to decode.**

So A is a **prefill-throughput** lever. That matters for the prefill half of PD-disaggregated serving,
but it does **not** touch the decode-latency path that dominates the serving denominator — which is why
the synthesis (`MEASURED_FINDINGS §D/Synthesis`) positions **QuantTile (B) first** (attacks the decode
weight wall) and **A as the second lowering of the same `stage`/`retire` tile-edge notation** for the
TP4-prefill AR. A satisfies Osama's overlap prize + Simran's "no AMD engine fuses the all-reduce"
target; it is not the highest-serving-impact lever.

---

## 5. Honest build cost and realistic expected win

**Build cost (high).**
1. **HK producer/consumer GEMM body + comm epilogue (2a):** the 8-wave body exists (bf16/fp8/mxfp8),
   but the **warp-specialized epilogue that emits IRIS tile transfers from inside the HK tile loop is
   net-new**, and the §6 evidence (13× stall on the naive path) says a careless split fails. Estimate:
   the hardest single piece; multi-week; this *is* the open research item.
2. **In-kernel reduce-scatter matching RCCL (2b):** **unproven on IRIS.** Requires store-completion
   fences, link-balanced destination scheduling (reuse TileComm Layer-2), local fp32 reduction with
   per-chunk arrival counters, an all-gather phase, and a *validated* bandwidth (the probe that was to
   confirm the mechanism was issue-bound). Estimate: substantial; carries the most execution risk.
3. **The `G` sweep + correctness gate** (RMS vs a torch AR reference; regression-gate that the fused
   kernel first *matches* unfused before chasing the ceiling).

**Realistic expected win (be disciplined).**
- **Ceiling: 1.18–1.7×, shape-dependent** (§0), and it *requires simultaneously* (a) a ~torch-rate
  GEMM body, (b) an in-kernel RS at ~RCCL BW, and (c) near-perfect overlap with a negligible
  `T_AGtail`. Miss any one and the win collapses (§1a: a 2.4×-slow body alone makes it >1× *slower*).
- **Honest v1 target: first get to ≤1.0× (i.e. *match* unfused)** — merely *not regressing* beats the
  shipped examples by 4.7–280×. Then claw toward the ceiling.
- **Realistic *delivered* win, if (a)+(b) land:** on the comm-bound prefill shapes,
  **~1.1–1.4×** (down-proj ceiling 1.43×, attn-out 1.18×), eroded below the ceiling by the all-gather
  tail, imperfect overlap, and any GEMM-body or RS-BW shortfall. **A real but modest single-collective
  prefill-throughput gain — not a decode or end-to-end win.** Quote it as a *prefill* number and never
  mix denominators.

**Known unknowns (flag every one):**
- **`L` (per-batch transfer latency) is unmeasured** → `G*` in §3 is a design estimate, not a tuned
  value; the whole `G` cost sketch is a model calibrated to *ratios*, not a validated predictor.
- **RCCL's 150 GB/s bus BW is modest** — either RCCL is under-tuned for these sizes on gfx950 or
  per-iter barrier overhead inflates it. If the true fabric ceiling is higher, `T_ar` (and thus A's
  ceiling) shrinks; if RCCL is genuinely this slow, an in-kernel RS that *beats* it would *raise* A's
  ceiling above the table. **We do not yet know which** — this must be pinned before over-claiming.
- **The in-kernel RS bandwidth on IRIS is unvalidated** (issue-bound probe). Until §2b is measured
  MAX-over-ranks with completion fences, "matches RCCL" is a target, not a result.
- **Prior art is crowded** (`MEASURED_FINDINGS §D`): Flux / CoCoNet / MSCCL++ / TRT-LLM already fuse
  AR+RMSNorm+quant. A's novelty is the *tile-granular, demand-scheduled, HK-body-native* lowering on
  the AMD/IRIS substrate + the `G` schedule — not the idea of fusing an AR.
