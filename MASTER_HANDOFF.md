# MASTER HANDOFF — Fused DeepSeek-R1 MoE expert region on AMD MI355X

> **You are a Claude agent resuming this project. Read this whole file first.** It encapsulates the goal, the current state, where everything lives, how to run on the cluster, and how to use the `auto-gpu-kernel` optimizer + spawn subagents. Last updated 2026-07-08.
>
> **🛑 NEW AGENT: read §0.4 (2026-07-08 FAIRNESS AUDIT) FIRST.** The headline `1.56×` and the
> `combine 386 vs 398 µs` numbers in §0/§3/§4 are **under audit and must not be quoted**. One kernel
> (`combine_pull`) is **outright incorrect** under real EP routing — confirmed on device.
>
> Then read §0.5 (2026-07-07) — it supersedes the "tiled communication abstraction" framing with a
> measured finding (traffic-shaping is a weak lever) and a ranked fork of higher-ceiling directions.

---

## 0. TL;DR (the 60-second version)

- **Goal:** take the production DeepSeek-R1 MoE expert region and make it faster by **fusing it** with **HipKittens** (on-GPU tile compute) + **IRIS** (cross-GPU XGMI comm), beating the unfused **MORI-dispatch → aiter-fmoe → MORI-combine** baseline. Thesis: *"aiter fuses the expert math; we fuse the expert region."*
- 🛑 **The results below are UNDER AUDIT (§0.4, 2026-07-08). Do not quote them.** They were measured with
  **one rank executing** (`example.py:644`), pre-quantized activations, a host-built routing plan, and a
  synthetic route that gives **zero** BM=256 padding where real routing gives 35.4%. The combine number is
  **void**: `combine_pull` drops cross-rank contributions (confirmed on device, expected 36 got 8).
- **Result as previously reported (8 full MI355X, correctness-gated *against the synthetic route*):**
  - **PREFILL: 1.56× faster** (fused 1247 µs vs unfused 1941 µs) — ⚠️ see §0.4 breaks 1–5. ~~Our pull-combine even **beats AMD's own MORI EpCombine** (386 vs 398 µs).~~ **← retracted, §0.4.**
  - **DECODE: ~2–3% faster** (516.5 vs 527 µs) at **matched fp8 precision** — marginal, honest. Decode is a **weight-memory wall** (see §5) that caps fusion there. ⚠️ same concurrency/quant/plan caveats.
  - ⚠️ Both denominators are from **thor-4 on Rainier**, not the MI350 node — and `b3` has never completed on the MI350 node. §11 warns the cluster varies ~1.8×.
- **The final kernel lives in** `irisx/fused_moe/` (formerly `b1_dispatch`). **Baselines** in `irisx/baselines/`. **Everything else** is archived in `irisx/development/`.
- **The authoritative results log is `irisx/EXPERIMENT_LEDGER.md`** — read it for every measured number, every dead end, and the build recipe.
- **The cluster (Rainier SLURM, MI355X) is in §7.** The login IP **changes** — if it's unreachable, ask the user for the current one.

---

## 0.4 — 2026-07-08 FAIRNESS AUDIT 🛑 the headline numbers do not survive contact ★ READ THIS FIRST

**Full writeup: `irisx/fused_moe/fairbench/FAIRNESS_AUDIT.md`.** User-initiated after noticing that
`gather_pack` and MORI's `EpDispatch` "start at different points". They do — and that turned out to be
the *smallest* of six problems. Nothing below is a guess; each is a source line or an on-device result.

### ✅ CONFIRMED ON DEVICE — `combine_pull` is WRONG under real EP routing (correctness, not fairness)
`combine_pull_kernel` → `tilecomm::tile_reduce_scatter` reduces a destination cell's **local** rows in
fp32 and then does a **plain `ctx.store`** (`irisx/tilecomm/tilecomm_device.h:123/135/151`; the header
says it: *"No atomics — a private accumulator per (tile, element)"*). That is correct only if every
contribution to a token lives on ONE producer rank.

Real top-8 routing over 256 experts / 32 per rank: **100% of tokens** have their experts on ≥2 ranks
(mean **5.33**, `real_route.combine_fanout`). So ≥2 ranks each store a partial sum to the same
`accb[token]` — last writer wins.

`fairbench/probe_combine_fanout.py` (8 ranks, no aiter/MORI): 8 ranks each contribute value `r+1` to one
cell. **Expected 36, got 8.** fanout=1 passes; fanout=8 drops 7 of 8 contributions.

⇒ **The "our pull-combine beats AMD's own MORI EpCombine (386 vs 398 µs)" claim is void.** MORI's
`EpCombineIntraNodeKernel` pulls-and-accumulates from every peer (`intranode.hpp:674-698`, `WarpAccum`).
The cross-rank-correct kernel we already have is `combine_scatter` (`ctx.fetch_add`) at **788 µs** — which
*loses* to MORI's 398 µs.

### Source-verified fairness breaks in the 1.56× prefill number
1. **`example.py:643-649` runs the region on `rank == CONSUMER` ONLY.** Ranks 0–6 idle at a barrier. The
   gather pulls from seven *idle* peers: no all-to-all contention, no straggler. `b3` runs the real 8-way
   collective and reduces with MAX over ranks. → new `ALL_RANKS=1` env in `example.py` fixes this.
2. **The fused path never quantizes.** `A_src` is fp8 + scales, produced on the host off the clock
   (`example.py:352-355`). `b3` dispatches bf16 and pays `dynamic_quant` inside `fused_moe`
   (`aiter_ref/fused_moe.py:1766-1773`). Free fp8 also halves the fused side's XGMI bytes.
3. **The routing plan (`SEG`/`TILE`) is host-built, off the clock.** MORI builds placement on device every
   dispatch (`intranode.hpp:145-170`, atomicAdd slot alloc). MORI *does* expose a cached/replay path
   (`dispatch(..., routing=...)`) — the honest tier-1 analog. Tier-2 needs an on-device plan builder
   (now written: `plan_count/scan/scatter` + `plan_allgather_ids` in `kernel.cpp`).
4. **The synthetic route flatters the gather.** `build_multisource_route` walks monotone per-rank cursors:
   | | synthetic | **real top-8** |
   |---|---|---|
   | mean `route_segment` run | ~20.5 | **1.03** (so `tile_is_single_source` Path-2 is dead code) |
   | source-token reuse (`dup`) | **1.000** (⇒ top-1!) | **1.506** (a pull re-reads; a push dedups) |
   | BM=256 padding | **0.0%** | **35.4%** |

   The padding one is the sharpest: `ROUTE=uniform` + `TOTAL_M=8192` + `E=32` puts **exactly 256 rows in
   every expert** — one perfect tile, **zero waste**. Real routing gives `rows_per_expert ~ Bin(8192, 1/32)`
   (mean 258, σ≈15.8), so most experts pad to 512. Measured over all 8 ranks (mean rows the GEMM computes,
   per rank, for 8192 real rows): **fused BM=256 → 12128 (1.480×)**; aiter `BLOCK_SIZE_M=32` → 8700
   (1.062×); BM=16 → 8446 (1.031×). So under real routing the fused prefill GEMM does **1.394× the MFMA the
   baseline pays**, and under the synthetic route it pays **1.000×**. Comparable in size to the whole 1.56×
   claim. **Fixable** — tile prefill at BM=16 like decode already does — but the published number never
   paid it. (Decode is immune: `build_b0_tasks_decode` already tiles at BM=16.)
5. **The two sides don't produce the same artifact.** `moe_sorting` emits **index arrays only**; the
   unfused fmoe GEMM applies the permutation for free in its A-load. `gather_pack` materializes
   `Mpacked × 7168` fp8 bytes. "3 kernels fused into 1" is not a kernel-for-kernel identity — the only
   defensible boundary is "router → fc1 MFMA". *(MORI already ships a fused dispatch+sort emitting an
   expert-major buffer: `dispatch_standard_moe` → `packedRecvX`. Not compiled in our container.)*

### Two bugs that INFLATE the b3 baseline (they cut the other way)
- `b3:111` `max_num_inp_token_per_rank=max(8192, 4*T)` → `MaxNumTokensToRecv()=65536`, and aiter's
  `moe_sorting` takes its **static** token count from `topk_ids.size(0)`, which picks the kernel
  (`moe_sorting_is_oneshot`), sizes the `[E × pad32(65536)]` mesh workspace and the grid/LDS
  (`moe_sorting_opus.h:1382, :1132-1138, :536-541`). → new `MAX_INP=real` env.
- MORI's `totalRecvTokenNum` **accumulates across dispatches**: `atomicAdd` every call
  (`intranode.hpp:236`), `hipMemset` once at construction (`dispatch_combine.cpp:328`), `LaunchReset` is
  an **empty stub** (`:460`), and `dispatch()` never resets. b3 hands the growing `drn` view straight to
  `fused_moe(num_local_tokens=drn_)` each iteration. → new `FIX_DRN=1` env. *(source-verified; confirm
  with `fairbench/probe_drn.py` before quoting)*

### Also: the canonical baseline is from a DIFFERENT MACHINE
`b3` has **never completed on this node** (`~/b3_sweep.log` stops at the RCCL banner; `~/b3_t64_retry.log`
dies with `HIP failure: 'invalid argument'`). The canonical `1941 / 533 µs` came from **thor-4 on
Rainier**. §11 of this file warns the cluster varies ~1.8× node-to-node. A same-node baseline is required.

### Where the audit lives / how to run it
- `irisx/fused_moe/fairbench/FAIRNESS_AUDIT.md` — the full writeup, with every source line.
- `fairbench/real_route.py` — real top-8 plan builder + the distortion stats above (`python3 real_route.py`).
- `fairbench/probe_combine_fanout.py` — the 8-rank combine correctness demo (**run this first**).
- `fairbench/bench_dispatch_prefix.py` — the fair `gather_pack` vs `EpDispatch+quant+moe_sorting` race,
  same inputs, all 8 ranks, MAX over ranks, correctness-gated, two routing tiers.
- `fairbench/probe_drn.py` — confirms the `totalRecvTokenNum` accumulation.
- New kernels in `kernel.cpp`: `gather_pack_rowmap` (flat `(src_rank,src_row)` ABI — the run-encoded
  `route_segment` buys nothing at mean run 1.03) and the on-device plan builder.
- **Build trap:** the `distributed-kernels/build/` cmake cache had `DK_BUILD=b1_tilecomm`, so
  `do_build.sh` returned rc=0 and compiled **nothing**. Always run `cfg_build.sh` first, then check the
  `.so` mtime.

### Status — DISPATCH-PREFIX TIMINGS NOW MEASURED (2026-07-08, fresh 8× MI350X `smci350-odcdh2-a08-2`)
Fair, warm (30 iter / 10 warm-up), all 8 ranks, MAX over ranks, real top-8 route, `T=1024` prefill.
Two-process design (MORI+aiter under `mp.Pool`; IRIS gather under `mpirun`) — the single-process
`bench_dispatch_prefix.py` is still blocked on the MORI `shmem_init_attr` node-state hang (§3.4 of the
audit; the fix is exclusive/clean GPUs, not code). Prefix = "router → fc1 A operand ready":

| prefix | tier-1 (routing cached) | tier-2 (routing on device) |
|---|---|---|
| **FUSED** `quant[T]` + `gather_pack_rowmap` | **252 µs** | 320 µs |
| UNFUSED **fp8-dispatch** (tight) | 295 µs | 332 µs |
| UNFUSED **bf16-dispatch** (b3/C4 default) | 368 µs | 392 µs |

**The 1.56× decomposes.** At the dispatch boundary the *isolated fusion* win (no separate sort pass —
placement IS the gather) is **1.17× cached, ~1.04× when the plan is built on-device** (our on-device
`build_plan` = 68 µs costs MORE than MORI's on-device routing ≈ 37 µs, eating most of the gain). The
larger 1.46× vs the bf16 baseline is **~half fp8-vs-bf16 movement**, which a production stack gets for
free with `DISPATCH=fp8`. Full table + decomposition in `fairbench/FAIRNESS_AUDIT.md §5.3`.

Also measured: **`gather_pack_rowmap` beats the `route_segment` ABI 1.106×** (ship rowmap); **`moe_sorting`
≈ 72–73 µs at BOTH `MAX_INP` settings → §2.1 "sort inflation" is REFUTED** (the 174 µs first reading was
cold JIT; use ≥10 warm-up on this stack). 13/13 correctness gates pass on all 8 ranks.

**Still pending:** the prefill *region* (with the expert GEMM + combine) under `ALL_RANKS=1` to correct
the deck's `1247 vs 1941 µs`; decode `T=64`. And the combine is still §0.4-incorrect, so any region
number that includes it is measuring a broken kernel until `combine_pull` is fixed.

**Build recipe for the fresh node** (the old node's prebuilt `tk_kernel` died with it): clone
`HazyResearch/HipKittens` (tip `60fd1dd`) + `subha-amd/iris@subha/moe-dispatch-v0`; `cp -a
iris/irisx/fused_moe/. HipKittens/distributed-kernels/b1_dispatch/` and `iris/irisx/tilecomm/.
HipKittens/distributed-kernels/tilecomm/`; `cmake -B build -DGPU_TARGET=CDNA4 -DDK_BUILD=b1_dispatch`
(CPM-fetches the IRIS lib from `ROCm/iris:muhaawd/irisx`); `cmake --build build --target b1_dispatch
--target iris_py`; `pip install mpi4py`. `.so`s land in `b1_dispatch/` (tk_kernel) + `distributed-kernels/`
(iris_py) → run with `PYTHONPATH=…/distributed-kernels`. Container `subha_fair` from
`rocm/atom-dev:nightly_202607061543`. Shared node (tenants `okachur`, `hangy`).

---

## 0.5 — 2026-07-07 session: the TileComm comm abstraction (explored) + the abstraction fork

**What this session did.** Pivoted from the fused MoE *kernel* to the *abstraction* Awad/Osama asked
for (the "tile-level communication abstraction"). Designed and prototyped **TileComm** — a kernel
**declares** tile transfers → the library **schedules** them link-balanced → **executes** on IRIS
primitives inside a HipKittens kernel. Built a calibrated cost model, a design doc, a talk, and an
on-node validation probe. Paper working title: *"Tile-Level Communication for Fused Multi-GPU
Inference."* **Everything lives in `irisx/tilecomm/` and `july-07-presentation/`** (map at the end of
this section).

### The honest result — comm *scheduling* (traffic-shaping) is a WEAK lever. Do not re-invest in it.
The seed was the combine kernel's **2.4× win (934→386 µs)** from one hand-rolled schedule
(`build_combine_pull(..., interleave=True)` round-robining cells across XGMI links). We tried to turn
that into a general demand-aware scheduler. Two structural reasons it under-delivers:
1. **On a fully-connected 8-GPU XGMI fabric, a scatter/gather's bytes-per-link are fixed by the demand
   matrix.** Reordering *when* blocks fire can only avoid the pathological *sorted* order — it can
   never beat the hottest-link floor. The cost model (`tilesched.py`, calibrated to the real 934/386)
   shows the schedule is worth **2.4–6× over the naive order but only 1–4% over the good hand-rolled
   round-robin.** The 1–4% is the actual ceiling, not a tuning gap.
2. **Decode — the serving-latency-critical path — is weight-memory-bound; comm is only ~18% of it.**
   So ANY comm-centric abstraction caps decode at ~1.22×. Comm is the wrong target for the path that
   matters most for serving R1.

### The on-node probe — INCONCLUSIVE, and exactly why (`irisx/tilecomm/xgmi_probe_results.md`)
Ran `xgmi_probe.py` on the 8× MI350X. Two takeaways:
- **A real `iris.store` gotcha (worth knowing):** it faults ("write access to read-only page") on a
  **data-dependent, memory-loaded `to_rank`**. You MUST select the destination through a constexpr
  `tl.static_range(WORLD)` loop, exactly like `examples/07_gemm_all_scatter`. After that fix it runs.
- **The probe is issue-bound, so it neither confirms nor refutes the 2.4×.** IRIS stores are
  fire-and-forget and `iris.do_bench` timed *issue rate*, not link bandwidth (it implied ~4.7 TB/s,
  >10× the fabric), so all three schedules looked identical (~1.05×). The `examples/01_store` control
  gave a real **47 GiB/s/link**. **To make the probe a valid 2nd data point:** add a store-completion
  fence + push enough bytes/link to back-pressure the issue queue + time MAX-over-ranks, then re-run.
  Until then the combine 2.4× stays a real *region* number whose mechanism (link-spread vs the
  combine's read/reduction structure) is **not yet isolated**; the cost model is a calibrated design
  tool, **not** a mechanistically-validated predictor.

### Memory feasibility for DeepSeek-R1 — SETTLED. It fits. Stop worrying about it.
MI350X = **309 GB HBM each → 2.47 TB across 8** (verified via `rocm-smi`). R1 MXFP4 (403 GB) = **16% of
HBM**; fp8 (~671 GB) = 27%. The historical wall was DISK (403 GB download), not HBM.

### The strategic fork — what the NEW AGENT should pick from (ranked by R1's real bottlenecks)
Traffic-shaping is out. The higher-ceiling tile-level abstractions:
1. **Quant-tile** — a tile that carries its own format + scale (fp4 / fp8 / MX), unpacked in-register
   at MFMA time. Attacks the **dominant decode weight wall** (the *only* high-ceiling decode lever is
   fewer weight bytes). Builds directly on the shipped Route-1 MXFP4 work (§10.1). HK currently
   hard-codes per-format kernels (`fp8fp32`, `mxfp8`), so a unified quant-tile is a genuine HK-paradigm
   contribution. **Highest R1-serving impact + still novel.**
2. **Comm/compute OVERLAP** — the *fusion* half of TileComm (NOT the scheduling half): tile-fused
   all-reduce + GEMM on the **TP4 prefill** path. This is Osama's "fuse inside the GEMM" prize and
   Simran's "no AMD engine fuses the all-reduce with anything — either fusion has huge production
   impact." Keeps Awad's comm-abstraction framing, re-pointed to the layer that actually has headroom.
   Depends on the producer/consumer-warp GEMM body (the open item in §6).
3. **Sparse-expert / conditional-tile** — skip streaming weights for local experts that received zero
   tokens at decode. Real at small batch; bounded ceiling.
4. ~~Comm traffic-shaping / demand-aware scheduling~~ — **weakest, capped by the fabric. Done
   exploring — don't repeat it.**

### The decision rule (the discipline we skipped — do NOT pick from priors again)
We picked comm-scheduling from a prior and it was weak. **Get the realistic-config profile FIRST**
(decode: quantify the weight wall vs the ~18% comm; prefill: quantify the TP4 all-reduce share) under
**TP4×DP2 + DP-attention + EP**, *then* commit to an abstraction. Predicted shape: decode → attack
weights (quant-tile / sparse-expert); prefill → fuse the all-reduce. This realistic-config profile is
the SAME §10 open item — do it once and it decides both the abstraction AND the fp4 baseline.

### Where it lives / how to run
- `irisx/tilecomm/DESIGN.md` — the abstraction spec + paper framing (declare/schedule/execute, the 4
  intents, the quadrant vs NCCL/MSCCL/aiter/IRIS/HK, mapping onto `fused_moe`, roadmap, honest limits).
- `irisx/tilecomm/tilesched.py` — `python3 tilesched.py` — the cost model + 3 schedulers + skew sweep (laptop-ok, numpy).
- `irisx/tilecomm/xgmi_probe.py` + `xgmi_probe_results.md` — the on-node probe + the honest read (issue-bound).
- `irisx/tilecomm/README.md` — the directory guide.
- `july-07-presentation/index.html` + `SCRIPT.md` — the talk (abstraction-first) with prepped Q&A incl. the probe caveat.
- **Node (NEW, different from §7 Rainier and the older Node A):** `ssh -i ~/.ssh/muhammad-gpu subvadla@10.190.161.47`
  — host `cv350-rck-g03-f03-18`, 8× MI350X gfx950, **shared** (do NOT disturb other tenants'
  containers `rkarhila_*`, `pedaniel-*`), **already GPU-ready** (no modprobe/docker-start needed).
  Container `subha_tilecomm_probe` is left running with `iris` pip-installed from `~/iris_lib` (torch
  2.9.1+rocm7.2, triton 3.6). DeepSeek models cached in `/data2/huggingface/hub`. See Claude memory
  `mi350x-node-access` (Node B) + `tilecomm-abstraction`.
- Commits pushed to `subha/moe-dispatch-v0` (through `5953fec`).

---

## 1. The goal, in context (what Simran / Muhammad Osama / Muhammad Awad asked for)

From the planning meeting (transcript: `/Users/subha/repos/amd-general/HipKittens + Iris.docx`):
- **Config (Simran):** *"Data-parallel TP4 is the better throughput/interactivity point."* The MoE all-to-all (gather/scatter) only appears in **prefill-decode disaggregation with DP-attention + expert-parallel**. → We use **C4 = TP4 × DP2 + EP + DP-attention** as the baseline config.
- **The kernel (Osama):** *"Do the MoE decode **gather** — build the model with **HipKittens kernels + a load/store (IRIS) on the MoE side**. That kernel looks simple but has the essence of everything we want from the abstraction perspective."* → That is exactly our design.
- **The research prize (Osama):** *"The cooler thing is **tile-level fusion inside the GEMM** — there's a lot more to hide behind there — but it's harder because you're GEMM-resource-constrained."* → We **demonstrated** tile-level comm/compute overlap for the dequant (free, hidden under XGMI), and **characterized** why the full gather-under-MFMA needs a different GEMM body (see §6). This is the open direction.
- **The abstraction (Awad):** *"The contribution is the **tile-level communication abstraction** (HK compute + IRIS comm), not just raw speed."*
- **Methodology (Simran):** *"Long all-reduce times are usually **profiling artifacts** — use in-kernel / MAX-over-ranks timing."* → We do (see §8).
- **A cost model (Simran/Osama):** *"Write a cost model."* → We did; it's what revealed the weight wall (§5).

A presentation summarizing all of this is in **`june-30-presentation/`** (`moe_fused_talk.pdf` + `SCRIPT.md` per-slide speaking notes).

---

## 2. Repo map — where everything lives

**This repo** = `github.com/subha-amd/iris`, branch **`subha/moe-dispatch-v0`**. It is the IRIS library; **our MoE work is under `irisx/`.**

```
iris/
  MASTER_HANDOFF.md          <- THIS FILE (start here)
  irisx/
    EXPERIMENT_LEDGER.md     <- ★ authoritative results log (read this)
    README.md
    fused_moe/               <- ★ THE final, usable fused kernel (was "b1_dispatch")
      kernel.cpp             <-   all 5 device kernels (gather/GEMM/act/combine + decode GEMM)
      example.py             <-   the driver (env-gated: FFN, COMBINE, TOTAL_M, DECODE_SAT, ...)
      b0_tasks.py            <-   builds the flat per-expert GEMM tile task-list (BM padding)
      build_tasks.py         <-   packed-layout / ERB padding helper
      b1_dispatch_route.py   <-   multi-source route (which rank each routed row came from)
      ep8_gather.h           <-   verified multi-source row resolver (used by the gather)
      B1_DISPATCH.md         <-   kernel-level notes
    baselines/               <- ★ the unfused production baseline + cost model
      b3_ep8_unfused.py      <-   THE baseline: MORI EpDispatch(bf16) + aiter fused_moe + MORI EpCombine
      b2_aiter.py            <-   aiter GEMM-throughput bar (TFLOP/s)
      b2_unfused_region.py   <-   single-GPU stock region sweep over M_e (the cost-model data)
      moe_cost_model.py      <-   analytic roofline incl. the weight-wall term
      probe_ep.py probe_mori.py  ...
    development/             <- everything else (dev kernels, infra, old docs) — NOT the production path
      b1_overlap/            <-   the comm/compute-overlap experiments (free dequant, gather-under-MFMA)
      grouped_b0/            <-   standalone GEMM dev (grouped_b0.cu, sat_decode.cu) — GEMM was copied into fused_moe
      reference/             <-   v5_grouped, v2_hk_expert_gemm (the GEMM lineage)
      ep8_gather/ abi/ harness/ benchmarks/ tests/ scripts/ cmake/ include/ archive/
      docs/                 <-   older docs (PROJECT_SUMMARY, FMOE_LAYOUT, BENCHMARKING_METHODOLOGY, C4_RESULTS_DECK.html, ...)
  june-30-presentation/      <- the talk (Beamer): moe_fused_talk.tex/.pdf, SCRIPT.md, img/, source_slides/
```

> Note: the EXPERIMENT_LEDGER and many docs still say **"b1_dispatch"** — that is the same thing as **`fused_moe/`** (just renamed for clarity). The cluster build target is still named `b1_dispatch` (§7).

**The optimizer** (`auto-gpu-kernel`) is a **separate repo** at `/Users/subha/repos/auto-gpu-kernel` — see §9.

---

## 3. The final fused kernel — what it does (precise)

All in `irisx/fused_moe/kernel.cpp`. The shipped path is `FFN=full COMBINE=1` (prefill) / `+ DECODE_SAT=1` (decode). **HipKittens [HK] = on-GPU compute; IRIS = cross-GPU comm; no aiter kernels are used in our path.**

| stage | kernel (kernel.cpp) | what it does | grid → block |
|---|---|---|---|
| gather | `gather_pack_kernel` [IRIS] | for each routed row, IRIS `ctx.load(ptr, src_rank)` = one XGMI read of 16 fp8 bytes + scale into the **expert-major packed buffer**; local rows skip XGMI; padding/unrouted rows → fp8 `0x00` "zero sentinels" (multiply to exactly 0). Replaces EpDispatch + 2 sort passes. | `(N_tiles, GP_SPLIT)`; thread = 1 row |
| dequant | `dequant_packed_dense` / `_mtile` [HK] | vectorized fp8→bf16 of the packed A right before the GEMM (the GEMM consumes bf16). | `Mpacked` (or task-driven), 256 thr |
| fc1/fc2 | `grouped_b0_gemm` [HK] | the expert matmul: **8-wave, 256×256×64, double-buffered shared→reg ping-pong, `mma_ABt` (16×16×32 MFMA), fp32 accum.** **One block per per-expert tile**, flat task list → all 32 local experts in one launch. Prefill is **bf16-precision** (RMS 0.018). | `num_tasks`, 512 thr (2×4 warps) |
| activation | `silu_quant_kernel` / `_mtile` [HK] | reads fc1 out (gate‖up), computes SiLU(gate)·up, **requant to fp8** (per-128 amax/448) for fc2. The 127→38.6 µs win was local-amax (16-way vs 128-way `atomicMax`). **Not a matmul.** | `Mpacked` (decode: real m-tiles), 256 thr |
| decode GEMM | `grouped_b0_gemm_decode` / `_fp8_sat` [HK] | **a *different* GEMM for decode: skinny BM=16 tile** (vs prefill's 256) so small experts don't grind padding; **fp8-weight-streamed** — store weights fp8 (½ the HBM bytes), load via the *fast bf16 path* + offline pre-swizzle + in-register unpack → **bf16 MFMA** (math is bf16, memory is fp8; see §5.6). 3.97 TB/s standalone. A true native-fp8 MFMA path exists behind `DECODE_SAT=0` but coarsens scales (RMS 0.073). Critical: a `s_waitcnt lgkmcnt(0)` + `sched_barrier(0)` before the mma (compiler was hoisting the mma above the data wait). | `num_tasks`, 512 thr |
| combine | `combine_pull_kernel` [IRIS] 🛑 **INCORRECT under real EP routing — see §0.4** | each *destination cell* reduces the ≤8 fc2 rows **that are on THIS rank** in fp32, then **plain-stores** one bf16 row via `ctx.store` (no atomics). ⚠️ **A token's experts live on ~5.33 different ranks under real top-8 routing (100% of tokens have fanout>1), so every other rank's partial sum is overwritten and lost.** Only correct at fanout=1, which is all the synthetic route and the single-rank timing loop ever produce. The round-robin dst-rank interleave (934→386 µs) is real, but the kernel is doing less work than MORI's. The "rejected" `combine_scatter_kernel` (IRIS `ctx.fetch_add`, 788 µs) is the **cross-rank-correct** one. | `num_cells`, 256 thr |

**Per-stage time** (decode, thor-4, the fair gate ≈516 µs): gather 41.5 · fc1 274 · act 7.2 · fc2 150 · combine 47 → **fc1+fc2 = 82% (the weight wall)**.

---

## 4. Current results (precise — don't mix denominators across nodes)

| regime | fused (b1, `fused_moe`) | unfused (b3, `baselines/b3_ep8_unfused.py`) | verdict |
|---|---|---|---|
| **prefill** (TOTAL_M=8192) | **1247 µs** (RMS 0.018) | 1941 µs | **1.56× WIN** ✅ |
| **decode** (TOTAL_M=512, warm) | **516.5 µs** (RMS 0.057) | 527 µs (533 canonical) | ~2–3% win |
| combine | ~~386 µs (pull)~~ **VOID — drops cross-rank sums (§0.4)** | 398 µs (MORI) | ~~we win~~ the correct variant (`combine_scatter`, fetch_add) is **788 µs** → **we lose** |
| decode GEMM (standalone) | 3.97 TB/s, 1.46× over bf16, RMS 0.0037 | aiter parity (171.0 vs 171.6 TFLOP/s) | parity |

Latest commits on origin: `0d0abf7` (decode round-1) ← `de876b5` (task-driven silu_quant) ← `aaa678b2` (sat integration) ← `53a454f` (fair gate) ← `78a8a185` (decode fp8 GEMM). **All measured numbers + dead ends are in `irisx/EXPERIMENT_LEDGER.md`.**

---

## 5. The key technical findings (the "why")

1. **Decode is a WEIGHT-MEMORY WALL — this is the central insight.** At decode each expert sees only ~4–16 rows (`M_e`), but the GEMM must stream **all 32 local experts' ~1.4 GB of fp8 weights from HBM every step** — a fixed ~176 µs floor (8 TB/s) independent of token count. So the decode region ≈ **80% weight floor + 20% boundaries** (gather/sort/quant/combine). That 80% floor is **identical for us and the baseline** (same weights), so **even if fusion made every boundary free, decode caps at ~1.25×** — we get ~1.03×. **Prefill has no such cap** (the GEMM does token-scaled compute), which is why it wins 1.56×. *The win lives where the dominant cost is improvable.*
2. **The padding tax.** The grouped GEMM pads each expert up to a multiple of `BM` (constant per expert; imbalance is absorbed by the *number* of tiles, `ceil(M_e/BM)`). BM=256 at decode → ~94% padding/expert → compute-bound on zeros. **BM=16 (the MFMA min) fixes it.** Production aiter pads too (~`block_m`=32), so it's a self-inflicted, fixable handicap — *not* an unfair benchmark.
3. **Comm/compute overlap, demonstrated.** In `b1_overlap`, folding the fp8→bf16 dequant *into* the gather (so it runs in the shadow of the XGMI loads) made it free: gather+dequant 162 µs == gather-only 168 µs. The shipped `fused_moe` keeps them as separate vectorized kernels; the *in-GEMM* version is the open wall (§6).
4. **The fp8 RMS floor.** Decode RMS 0.057 > 0.05 is the inherent e4m3 *weight* quant error (prefill bf16-weights = 0.018). It's scale-invariant. But the aiter baseline *also* runs fp8 weights, so 0.057 is production's own precision class — a strict <0.05 needs a bf16 matmul (+135 µs = a loss). State it as a µs-vs-precision tradeoff.
5. **fp4 is the next decode lever (discussed, not yet built).** fp4 weights halve the 1.4 GB stream → halve the weight floor → and *raise* the decode fusion ceiling (boundaries become a bigger share). The CDNA4 MFMA we use (`mfma_scale_f32_16x16x128_f8f6f4`) already supports fp4. The design changes: a finer block-scale (MXFP4 32-block e8m0, or NVFP4 16-block fp8), likely W4A8/A16, and an fp4-faithful reference. **Caveat: only a fair comparison if aiter has an fp4 fmoe path — verify first.**
6. **Our expert GEMMs run bf16 MFMA, NOT fp8 like aiter — we dequantize before the matmul.** This is a real format difference worth knowing (and a question for the HK maintainers, below). `gather_pack` ships fp8 over XGMI, but every shipped GEMM dequantizes to **bf16** before the MFMA:
   - **Prefill** (`grouped_b0_gemm`): activations dequant fp8→bf16 (`dequant_packed_dense`), **weights are stored bf16** (`example.py:270`) → **bf16×bf16 MFMA (16×16×32)**, RMS **0.018**. This is *more accurate* than aiter's fp8 (we don't quantize the weights at all), but it **assumes bf16 weights in HBM** — production R1 ships **fp8** weights, so dropping this into vLLM needs a one-time fp8→bf16 weight dequant (≈2× the weight footprint) **or** an fp8-GEMM adaptation. ⚠️ flag for Simran.
   - **Decode shipped** (`grouped_b0_gemm_decode_fp8_sat`, `DECODE_SAT=1`): weights **stored fp8 (1 byte — the byte-saving that matters for the weight-bound regime)**, loaded through the *fast bf16 path* (2 fp8 bytes reinterpreted as 1 bf16 via an offline per-128-block column **pre-swizzle**), unpacked fp8→bf16 **in-register**, then **bf16 MFMA**. So **memory = fp8, math = bf16**. RMS 0.057 is set by the fp8 *weight* quant (matches aiter's precision class) — the activations stay bf16.
   - **Decode native-fp8** (`grouped_b0_gemm_decode_fp8`, `DECODE_SAT=0`, **NOT shipped**): a true fp8×fp8 MFMA (16×16×128) — but it needs A requantized to a single **per-row** scale (our scales are **per-128-block**), and that coarsening pushed RMS to **0.073**, so it's behind the flag.
   - **Why bf16 and not fp8 (checked against the HK source at `/Users/subha/repos/HipKittens`, HEAD `840c967` "MXFP8 Optimizations"):** HipKittens has **no native fp8 global→register load — it's a hard `static_assert("Unsupported type for load")`** (`include/cdna4/ops/warp/memory/tile/global_to_register.cuh:30,139`); the fast `buffer_load_b64/b128` path is bf16/float only. (Precise: global→**shared** *does* accept fp8 — `global_to_shared.cuh:266` allows 1-byte dtypes; only global→**register** forbids it, which is why the intended fp8 path stages via shared.) **This is by design, not a gap** — HK's *own* fp8 kernels (`kernels/gemm/fp8fp32/FP8_8wave`, `kernels/gemm/mxfp8/MXFP8_8wave`) stage fp8 through **LDS**: `st_fp8e4m3` shared tiles via `G::load` (global→shared) then `load_st_to_rt` (shared→register, `ds_read_b128`). Our non-`_sat` `_fp8` path matches that exactly; we measured the LDS-staged path **~2.6 TB/s** for the skinny BM=16 tile, vs our `_sat` reinterpret-as-bf16 trick **3.97 TB/s** — i.e. **`_sat` already beats HK's intended fp8 path** for that tile. So the unpack is NOT a real bottleneck and a "native fp8 load" is neither available nor the lever — **fewer bytes (fp4) is.**
   - **The scaled MFMA is ALREADY exposed in HK** (this resolves the earlier "open question"): `mma_ABt_scaled(d,a,b,c,scale_a,scale_b)` (`include/cdna4/ops/warp/register/tile/mma.cuh:529` → `mfma1616128_scaled` → `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`), with `pack_scales` and a full reference GEMM in `kernels/gemm/mxfp8/`. It consumes **E8M0 (power-of-two), per-32-K-block** scales = the **MX** format. "requant to per-row" = the cost of applying scales *outside* an unscaled MFMA (which sums all of K, so only one scale/row fits); the scaled MFMA applies per-block scales *inside* the accumulate → no requant, no accuracy loss.
   - **The real remaining gap is a quantization FORMAT mismatch, not an HK feature:** aiter/DeepSeek fp8 = **per-128-K, fp32** scales; HK/hardware scaled MFMA = **per-32-K, E8M0** scales. per-128→per-32 is lossless replication, but **fp32→E8M0 is lossy** unless the model is genuinely MX-quantized. ⇒ **If we use the AMD-quantized MXFP8/MXFP4 model (almost certainly what Simran sent), HK already gives a full scaled, no-requant fp8/fp4 GEMM — reuse the `mxfp8` 8-wave body + wrap our expert-grouping + IRIS gather/combine around it.** This makes §5.5 / §10.1 (the fp4 decode lever) much closer than previously framed: the GEMM exists in HK; the work is the MX wiring + the MX-format reference, not a new MFMA path.
   - **Implication:** the **1.56× prefill win is purely from fusion** — achieved while our GEMM does *more* work per weight byte than aiter's fp8 (bf16 MFMA is ~½ the fp8 MFMA rate; bf16 weights are 2× the bytes). So a true fp8/fp4 GEMM is **headroom, not a regression**.

---

## 6. What did NOT work (and the architectural reason — don't re-try blindly)

- **Host-stream overlap** (gather ∥ GEMM on 2 streams): concurrent = −12 to −150 µs vs serial. The GEMM saturates the CUs; streams just time-slice.
- **In-kernel gather-under-MFMA** (the "fuse inside the GEMM" prize): 13× slower (36 vs 466 TFLOP/s). The fast 8-wave body has no producer/consumer warp split and no A-reuse, so blocking remote loads stall the wave. **Needs a *different* GEMM body with async A-fill + reuse** — the real open research direction.
- **Native fp8 at BM=256:** neutral — decode is compute-bound on padding there, so fewer weight bytes don't help until BM=16 makes it memory-bound.
- **Register double-buffer for the decode tile:** 3× slower — blew VGPR → occupancy collapse. At BM=16 latency-hiding = occupancy, not a wider reg pipe (use LDS double-buffer).
- **Compacting the GEMM padding (Mpacked 8192→512):** measured high-risk/low-reward; padding mostly hides under the weight stream, the only exposed cost was the *act* tax (already taken), and it breaks ERB tile-alignment.
- **A strict RMS<0.05 decode win:** fundamental — requires bf16 = a loss.

---

## 7. The cluster (Rainier SLURM, 8× MI355X / gfx950 / CDNA4) — how to run

**Login (passwordless; the IP CHANGES — if unreachable, ask the user for the current one):**
```
ssh -p 2425 -o ControlPath=none subvadla@172.19.164.255    # was 10.0.0.228 earlier; user-provided
```
Also see Claude memory `gpu-node-access.md`, `rainier-push-auth.md`, `rainier-bench-gotchas.md`.

**Allocations (each = one whole 8-GPU node, 4h wall limit):**
```
salloc --no-shell --immediate=60 --gres=gpu:8 --time=4:00:00 --job-name=irisx-main    # node A (8-GPU region)
salloc --no-shell --immediate=60 --gres=gpu:8 --time=4:00:00 --job-name=irisx-main2   # node B (single-GPU GEMM dev)
squeue -u $USER -o "%.7i %.13j %.9L %N"      # look up live job ids BY NAME (they change on re-grab)
srun --jobid=<ID> --overlap bash -lc "<cmd>"  # run on a held node
```
- **Re-grab discipline:** `salloc` with an existing same-name job adds a SECOND job on a new node; `head -1` returns the OLD one → after the new container is up, **`scancel` the old job** so name-lookup + the keeper converge. A keeper script `~/rainier_keeper.sh` (runs on the login node, `setsid`-detached) re-grabs when <1h left and restarts containers; if login is down the keeper is down too (allocations may lapse — that's OK, committed work persists in NFS).
- **SPX vs DPX:** the current nodes are **SPX = 8 logical = 8 physical GPUs**; use the **natural** `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. (Earlier DPX nodes had 16 logical = 8 physical × 2; there you had to pin `0,2,4,...,14`. If a node gives "invalid device ordinal" on 0–7, it's DPX — switch to the even pin.)

**Containers** (`irisx1` on node A, `irisx2` on node B), image `rocm/atom-dev:vllm-v0.22.0-nightly_20260610` (has aiter + MORI):
```
srun --jobid=<ID> --overlap bash -lc "docker ps --format '{{.Names}}' | grep -qx irisx1 || \
  docker run -d --name irisx1 --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v \$HOME:\$HOME -w \$HOME \
  rocm/atom-dev:vllm-v0.22.0-nightly_20260610 sleep infinity"
docker exec -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e MORI_GPU_ARCHS=gfx950 -e HSA_XNACK=1 irisx1 <cmd>
```
`$HOME` (= `/home/subvadla`) is **shared NFS** across all nodes — clone/build there.

**Build the production kernel** (build mirror, separate from the repo): `~/HipKittens/distributed-kernels/b1_dispatch/` — sync `irisx/fused_moe/*` into it, then `cmake -B build -DDK_BUILD=b1_dispatch -DGPU_TARGET=CDNA4 && cmake --build build -j16` (flock `/tmp/mi355x_compile.lock`). It builds `tk_kernel*.so` + needs `iris_py.so` (build via the dk build too). The build target stays named **`b1_dispatch`**.

**Run discipline:** `docker exec -d` (detached, survives SSH drops); `docker exec irisx1 pkill -9 python3` + a **fresh `MASTER_PORT`** before each 8-GPU `mpirun -np 8`.

---

## 8. How to run the gate (the benchmark) — and the methodology traps

**The fair comparison = same region, SAME node, both pre-warmed, MAX over 8 ranks, correctness-gated.**
- **Node speed varies ~1.8×** across the cluster — *always* run b1 and b3 back-to-back on one node. The canonical baseline (b3 decode=533, prefill=1941) was set on **thor-4** (`irisx-main2`).
- **Pre-warm aiter:** it JIT-compiles `fused_moe` per (shape,dtype) into a **container-local** cache (`/app/aiter-test/aiter/jit`, ~10 min cold) — a "hang" that's really a cold cache. Run b3 once to warm it.
- **Fused (b1):** in `~/iris/irisx/fused_moe`, `mpirun --allow-run-as-root -np 8 python3 example.py` with `-e FFN=full -e COMBINE=1 -e DECODE_SAT=1 -e TOTAL_M=512` (decode) / `TOTAL_M=8192` (prefill), `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, fresh `MASTER_PORT`. (A wrapper `run_region.sh` may exist in the build mirror.)
- **Unfused (b3):** `python3 baselines/b3_ep8_unfused.py` with `DISPATCH=bf16 QUANT=per_1x128 TOKENS_PER_RANK=64` (decode) / larger (prefill).
- **Pushing commits:** the cluster containers have **no GitHub creds**. To push, bundle from the cluster and push from the local authenticated clone (memory `rainier-push-auth.md`): `ssh ... 'cd ~/iris && git bundle create ~/x.bundle <origin-HEAD>..subha/moe-dispatch-v0'`, scp it, `git fetch /tmp/x.bundle ...:tmp && git merge --ff-only tmp && git push`.

---

## 9. The optimizer — `auto-gpu-kernel` (how to drive iteration + spawn subagents)

Located at **`/Users/subha/repos/auto-gpu-kernel/`**, and **backed up to your private fork `github.com/subha-v/auto-gpu-kernel`** (git remote **`mine`**, branch `main`) — clone that to resume the optimizer. (It's a fork of Dogacel/auto-gpu-kernel; `origin` points at Dogacel and is read-only for you — **push to `mine`**, not `origin`.) It is a self-improving kernel-optimization harness we retooled for this project.

```
auto-gpu-kernel/
  iris_moe_gemm/        <- Task A: the standalone grouped GEMM (decode/prefill, fp8/bf16)
  iris_moe_overlap/     <- Task B: the fused region + combine + overlap
  iris_template/        <- the task template
  fable/                <- the SOTA NVIDIA MoE blueprint (algorithm reference; regime branching, BM=16 decode)
  RAINIER_SLURM.md ALLOCATION_KEEPER.md current_allocations.env rainier_keeper.sh
  each task has:
    .claude/agents/{research,profiler,workload_inspector}.md   <- specialist SUBAGENTS
    .claude/commands/{optimize,benchmark,log-experiment}.md     <- the loop
    experiments/exp_N/{plan,result}.md, summary.md, LESSONS.md  <- the running log
    solution/hip/<kernel>.cu, CLAUDE.md
```

**How to use it (the `/optimize` loop):** read the task's `CLAUDE.md` + `experiments/{summary,LESSONS}.md`, then iterate: **assess → diagnose → profile → ONE change → build+bench (fair, same-node) → log `experiments/exp_N/` → decide → repeat.**

**Spawning subagents (do this — it's the design):** use the **Agent tool** with these clean-context specialists:
- **`research`** (`iris_moe_gemm/.claude/agents/research.md`, model opus, effort max): a *clean-context diagnosis* agent. It reads `irisx/EXPERIMENT_LEDGER.md` + the on-disk experiments + the `fable` blueprint + HipKittens, names the plateau's root cause, and writes the next `exp_(N+1)/plan.md`. **Spawn it before optimizing**, especially when stuck or unsure of the bottleneck.
- **`profiler`**: profile before optimizing (don't tune the wrong bottleneck — e.g. compute-tuning a weight-bound region).
- **`workload_inspector`**: captures the real workload profile (M_e distribution, shapes).
Run independent subagents concurrently (one Agent message, multiple tool calls). For background agents, prefer the **cluster-side file mtime + git commits + the `kernel.cpp` diagnostics** as liveness signals — **NOT** output-file size or proc-count alone (those misfired and caused working agents to be wrongly killed; when unsure, do nothing).

---

## 10. Open directions / next steps (in rough priority)

> ⚠️ **DO THIS BEFORE more fp4 work — establish a LEGITIMATE EXTERNAL baseline (lesson learned 2026-07-01).**
> Our fp4 comparison so far replays a *captured* aiter `a4w4` CK kernel that turned out **untuned + buggy** for
> our EP shape (see §10.1) — effectively a "fake baseline" we generated. Beating it means nothing. Use a baseline
> **we did NOT generate**, in this priority:
> 1. **AMD's official reproducible SGLang DeepSeek-R1-FP4 benchmark** — Docker
>    `rocm7.0_preview_ubuntu_22.04_sgl-dev-v0.5.2rc2_mi35x_rc1`, model **`amd/DeepSeek-R1-0528-MXFP4-Preview`**,
>    TP8, full server+bench commands (1024 in/out, conc 64, 128 prompts). Runs the REAL production stack (SGLang +
>    **FlyDSL** MoE + MoRI) on our exact MI350X. Run it, **profile the MoE region** + the realistic-config collective
>    breakdown, and iterate our fused region against THAT trace.
>    (`rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-sglang-deepseek-r1-fp4.html`)
>    **STATUS 2026-07-06 — ATTEMPTED, INCOMPLETE (no MoE baseline captured yet).** Infra learnings from the attempt:
>    the model is **403 GB** (82 safetensors, NOT ~190) → needs a node with **~500 GB genuinely-free** disk (put
>    `HF_HOME` on the big mount; use an HF token — unauth HF is ~5 MB/s). The aiter **EP MoE memory-faults in *eager*
>    mode too** (not just graph capture) — root cause is very likely **NUMA balancing enabled** (aiter warns; run
>    `echo 0 > /proc/sys/kernel/numa_balancing` before serving). SGLang capture = `/start_profile` with `num_steps` +
>    `profile_by_stage=true` → N prefill + N decode passes, one gzipped chrome trace per TP rank. **Profile the
>    REALISTIC config (C4 / DP8), NOT TP8** (see the opportunity map at the end of §10). Repeated node/SSH churn + an
>    API overload stopped the run before a trace landed. **Fallback: Simran offered to run the full e2e** — take her up
>    on it for the region baseline rather than brute-forcing the 403 GB serve on a shared node.
> 2. **Kernel-level:** aiter's *tuned single-GPU* a4w4 bar (FlyDSL `afp4_wfp4_bf16`, **169 µs @ M_e16**) — real,
>    external, and **2.4× ahead of our Route-1** (410 µs). Route 2 chases this.
> 3. **Reference points (published/third-party):** LMSYS/SGLang MoRI (MI355X 2,436 tok/s/GPU, MXFP4 FP4-dispatch +
>    FP8-combine, 2.56× round-trip BW cut); SemiAnalysis **InferenceX / InferenceMAX** (open-source live bench);
>    MLPerf Inference v6.0 (audited, Llama-heavy). Simran offered to run the full e2e SGLang test — gold denominator.
>
> **The rule: never iterate against a kernel you captured/generated; iterate against the reproducible production
> stack or an audited third-party number.**

1. **fp4 weights for decode** (§5.5/§5.6) — **✅ DONE 2026-06-30 (Route 1, shipped in `kernel.cpp`).**
   - **Model (amd/DeepSeek-R1-MXFP4):** OCP **MXFP4 W4A4** — weights (static) AND activations (dynamic) fp4
     e2m1, group_size 32, E8M0 per-block scale along K (HF config.json + card). Our Route-1 decode keeps A
     bf16 (weight-only fp4; more accurate than the model's true A4).
   - **Kernel:** `grouped_b0_gemm_decode_mxfp4_sat` — store B fp4 (¼ bf16 bytes), PRE-SWIZZLE (`kDecSatPermFp4`
     = algebraic compose of the fp8 `PERM128`), load via the fast bf16 path as `rt_bf<32,32>`, unpack fp4→bf16
     with the **gfx950 hardware `__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4`** (MX scale folded in — the software
     `float4(fp4e2m1_4)` path is 4× slower, the key perf lever), bf16 MFMA. Env `DECODE_MXFP4=1`.
   - **Standalone GEMM (same node):** MXFP4 **1.6× over our fp8 `_sat` decode**, ~2× over bf16. Gate: RMS vs
     dequant-fp4 = 0.0033 (correct); fp4 precision class vs true bf16 B = 0.117 (vs fp8 0.057).
   - **Fused decode REGION (8× MI350, same node, TOTAL_M=512):** MXFP4 **520.6 µs** (FFN PASS RMS 0.018) vs fp8
     578.9 vs bf16 881.3 → **1.11× over fp8, 1.69× over bf16** (GEMM-only 1.16×/1.96×; region diluted by the
     shared A-dequant + gather/act/combine boundaries).
   - **Baseline status (Phase A, 2026-07-01):** the captured aiter `a4w4` EP fmoe is **untuned AND buggy for our
     shape** — every `block_size_M` / `max_num_inp_token` sweep falls back to `2stage default` (no tuned config for
     E_local=32 / per_1x32), so the unfused a4w4 region stays **791 µs**, and its EP fmoe (691) is even *slower*
     than aiter's fp8 EP fmoe (503). ROCm/aiter #3632 (not HIP-graph-safe, gfx950) + #2343 (EP memory-fault,
     MI355X) confirm the CK-a4w4+EP path is immature. **So our 1.52× over that region is over an untuned/buggy
     kernel — do NOT headline it** (see the ⚠️ callout above for the real external baseline plan). *Update
     2026-07-06:* the EP memory-fault reproduced in **eager** mode (not just graph capture) and is very likely
     **NUMA balancing** (`echo 0 > /proc/sys/kernel/numa_balancing` before serving) — a config issue, so the a4w4 EP
     path may become *runnable* once NUMA is off, but it is still **untuned** for our shape.
   - **Route 2 (true fp4×fp4 scaled-MFMA) — feasibility validated (Phase B, exp_17), NOT yet built:** needs **NO
     new HK tile types** (the earlier premise was wrong) — `mma_ABt_scaled<cbsz,blgp>` accepts `fp8e4m3` A/B and
     `cbsz/blgp=4` selects fp4 on the existing `rt_fp8e4m3` tiles; the fp4-cbsz variant compiles clean. Stock HK
     `mxfp8/MXFP8_8wave` (fp8 scaled-MFMA) builds+passes at **421 TFLOPS** — the ready vehicle. Two on-device
     unknowns remain (multi-iteration; stopped rather than flail): (a) the fp4 operand byte→K layout with cbsz=4
     (K doubles to 256/tile, undocumented like the fp8 `_sat` swizzle), (b) the scale granularity (mxfp8 packs 4
     E8M0 for K=128; fp4's K=256 needs 8 per-32 blocks — per-64 would be lossy vs MXFP4's per-32). Route 2's value
     is **prefill** (decode is weight-bound where Route-1 W4A16 is already the recommended variant).
2. **The gather-under-MFMA GEMM body** (§6) — a producer/consumer-warp GEMM with async A-fill + reuse, to make true in-kernel comm/compute overlap work. This is Osama's "fuse inside the GEMM" prize.
3. **`reduce_scatter`** as the second IRIS collective (the meeting's other target).
4. **End-to-end TPOT** integration — measure the region win at the model level, not just region-level (the gold-standard denominator).
5. **SPX full-GPU re-confirm** of all numbers (the canonical set is on thor-4; keep same-node discipline).

### The end-to-end opportunity map (beyond MoE) — READ THE CONFIG CAVEAT FIRST (added 2026-07-06)

**⚠️ Collective costs are CONFIG-DEPENDENT — the all-reduce is NOT a universal lever.** The meeting-1 profiling
("collectives ~30% of total, all-reduce is the top kernel, fires 2×/layer") was measured at **TP8, which is
UNREALISTIC** for DeepSeek on AMD (Simran: "data-parallel TP4 is the better throughput/interactivity point; TP8
isn't used much for this model size on AMD — enough HBM"). By config:

| setting | TP all-reduce / reduce-scatter | MoE all-to-all | realistic? |
|---|---|---|---|
| TP8 (meeting-1) | **dominates (~30%)** | present | **NO — artifact** |
| TP4 × DP2 (C4) | small (within the 4-GPU group only) | present | yes (prefill) |
| DP8 / TP1 (C6) | **removed entirely** | present (only cross-GPU traffic) | ~decode |
| DP-attention + EP | **~gone** (attention is data-parallel) | **dominant** | yes (decode) |

So in the realistic **decode** path the **MoE all-to-all is the dominant collective** — the thing we already fused,
so our work is on the decode critical path. The all-reduce fusion is a **TP4-PREFILL** lever (real, but TP4-sized,
not the TP8 ~30%). **Never size a collective opportunity from a TP8 trace.**

**The levers, from Meeting 1 (`amd-general/HipKittens + Iris.docx`), ranked by the notes:**
- **MoE gather/combine** — ✅ already built (Osama: "the easy starting point… but has the essence of everything").
  It IS the dominant decode collective.
- **all-reduce + RMSnorm + quant fusion** (TP4 prefill) — Simran: "all AMD engines do NOT fuse the all-reduce with
  anything right now, so either one will have a huge production impact." aiter has one but "it's not very fast." The
  easier NVIDIA-style fusion; high impact **for the TP prefill path**.
- **GEMM + all-reduce tile-level fusion** — Osama's research prize: "how many tiles you produce from the GEMM side
  before you reduce… that permutation space is absolutely insane and really unexplored." Harder (GEMM-resource-bound),
  most differentiated.
- **reduce_scatter as a new IRIS device-side primitive** — the TP-decomposition stepping stone toward the fused
  GEMM+all-reduce (all-reduce = reduce_scatter + all-gather). Osama's suggested next collective.
- **attention / MLA** — the other pre-all-reduce region; profile its share.

**Decision rule:** get the **realistic-config** profile FIRST (the §10 ⚠️ callout), then pick between Route 2 (MoE)
and the TP4 collective fusions by **measured %**, not by the TP8 prior.

**External-benchmark reality (web, 2026-07-01):** fp4 is the validated production direction — AMD merged MXFP4+AITER
into vLLM/SGLang; NVIDIA's DeepSeek-R1 record uses **NVFP4 MoE + PD-disagg EP** (= our C4 config). BUT the literature
is explicit: at **small-batch / decode, 4-bit *activation* quant gives ~no speedup (weight-memory-bound) — weight-only
fp4 is preferred** — which independently **validates our Route-1 W4A16 decode choice**. fp4's throughput win is a
**prefill** effect (~1.41× over fp8 at batch 128). NVFP4 (per-16 fp8-scale) beats MXFP4 (per-32 E8M0 power-of-two) on
accuracy; MXFP4's power-of-two scale needs good calibration (our fp4 RMS class = 0.117). Reference bars: LMSYS/SGLang
MoRI (MI355X **2,436 tok/s/GPU**, MXFP4 FP4-dispatch+FP8-combine); SemiAnalysis InferenceX/InferenceMAX (open-source);
MLPerf v6.0 (audited).

---

## 11. Conventions & gotchas (read before you touch anything)

- **Commit from the user's account; do NOT add Claude as co-author.** After meaningful changes, update the relevant README and commit + push (push from the local clone — see §8).
- **Never mix µs denominators across nodes** (1.8× speed variance). Quote 1.56× (thor-4) for prefill.
- **Liveness for background agents:** cluster file mtime + commits + the `kernel.cpp` diagnostics — never output-file size / proc-count alone.
- **The kernel build is on the cluster** (HipKittens/distributed-kernels mirror), not the repo's CMake; the repo `irisx/fused_moe/` is the *source of truth* — sync it into the mirror to build.
- **The cluster `~/iris` may have stale uncommitted changes** (redundant with origin); `git fetch && git reset --hard origin/subha/moe-dispatch-v0` to sync after this reorg (the old `b1_dispatch/` path is now `fused_moe/`).
- **Honesty:** the decode result is a marginal win at the fp8 floor — report it precisely, don't oversell.
