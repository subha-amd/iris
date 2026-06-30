# MASTER HANDOFF — Fused DeepSeek-R1 MoE expert region on AMD MI355X

> **You are a Claude agent resuming this project. Read this whole file first.** It encapsulates the goal, the current state, where everything lives, how to run on the cluster, and how to use the `auto-gpu-kernel` optimizer + spawn subagents. Last updated 2026-06-30.

---

## 0. TL;DR (the 60-second version)

- **Goal:** take the production DeepSeek-R1 MoE expert region and make it faster by **fusing it** with **HipKittens** (on-GPU tile compute) + **IRIS** (cross-GPU XGMI comm), beating the unfused **MORI-dispatch → aiter-fmoe → MORI-combine** baseline. Thesis: *"aiter fuses the expert math; we fuse the expert region."*
- **Current result (8 full MI355X, same-node, correctness-gated):**
  - **PREFILL: 1.56× faster** (fused 1247 µs vs unfused 1941 µs) — the headline, solid. Our pull-combine even **beats AMD's own MORI EpCombine** (386 vs 398 µs).
  - **DECODE: ~2–3% faster** (516.5 vs 527 µs) at **matched fp8 precision** — marginal, honest. Decode is a **weight-memory wall** (see §5) that caps fusion there.
- **The final kernel lives in** `irisx/fused_moe/` (formerly `b1_dispatch`). **Baselines** in `irisx/baselines/`. **Everything else** is archived in `irisx/development/`.
- **The authoritative results log is `irisx/EXPERIMENT_LEDGER.md`** — read it for every measured number, every dead end, and the build recipe.
- **The cluster (Rainier SLURM, MI355X) is in §7.** The login IP **changes** — if it's unreachable, ask the user for the current one.

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
| combine | `combine_pull_kernel` [IRIS] | the **reverse** of gather: each *destination token* gathers its ≤8 fc2 rows from local memory, **sums in fp32 locally (no atomics)**, writes **one bf16** row home via `ctx.store`. Beats MORI by **round-robining cells across dst-ranks** (spreads XGMI links: 934→386 µs). The rejected scatter version (`combine_scatter_kernel`) used IRIS `ctx.fetch_add` and was 788 µs (XGMI write-bandwidth bound, 234 MB). | `num_cells`, 256 thr |

**Per-stage time** (decode, thor-4, the fair gate ≈516 µs): gather 41.5 · fc1 274 · act 7.2 · fc2 150 · combine 47 → **fc1+fc2 = 82% (the weight wall)**.

---

## 4. Current results (precise — don't mix denominators across nodes)

| regime | fused (b1, `fused_moe`) | unfused (b3, `baselines/b3_ep8_unfused.py`) | verdict |
|---|---|---|---|
| **prefill** (TOTAL_M=8192) | **1247 µs** (RMS 0.018) | 1941 µs | **1.56× WIN** ✅ |
| **decode** (TOTAL_M=512, warm) | **516.5 µs** (RMS 0.057) | 527 µs (533 canonical) | ~2–3% win |
| combine | 386 µs (pull) | 398 µs (MORI) | we win |
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
   - **Why bf16 and not fp8:** HipKittens (on CDNA4, as of this work) has **no *fast* fp8 global→register tile load** — only bf16 has the vectorized, swizzle-matched load. fp8 can only be (a) **staged through LDS** via a hand-written `ds_read_b128` (`dec_load_st_to_rt`, kernel.cpp:1090), which we measured **~2.6 TB/s** (slow), or (b) **smuggled through the bf16 load** (`_sat` trick). So we dequant to bf16 to hit the fast path. The *second* blocker for a true block-scaled fp8 GEMM is matching aiter's **per-128-block** scale at the MFMA without a per-row requant accuracy hit.
   - **Implication:** the **1.56× prefill win is purely from fusion** — achieved while our GEMM does *more* work per weight byte than aiter's fp8 (bf16 MFMA is ~½ the fp8 MFMA rate; bf16 weights are 2× the bytes). So a true fp8/fp4 GEMM is **headroom, not a regression** — the same lever as §5.5 and §10.1. **Open question raised with HK maintainers:** (1) is there a supported fast fp8/fp6/fp4 HBM→register tile load, or is reinterpret-as-bf16 the intended pattern? (2) does HK expose the CDNA4 scaled MFMA `mfma_scale_f32_16x16x128_f8f6f4` with per-block scale operands (and at what block granularity), so we can do block-scaled fp8/fp4 natively?

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

1. **fp4 weights for decode** (§5.5) — the highest-leverage decode experiment; halves the weight wall + raises the fusion ceiling. First verify aiter has an fp4 `fmoe` for a fair baseline.
2. **The gather-under-MFMA GEMM body** (§6) — a producer/consumer-warp GEMM with async A-fill + reuse, to make true in-kernel comm/compute overlap work. This is Osama's "fuse inside the GEMM" prize.
3. **`reduce_scatter`** as the second IRIS collective (the meeting's other target).
4. **End-to-end TPOT** integration — measure the region win at the model level, not just region-level (the gold-standard denominator).
5. **SPX full-GPU re-confirm** of all numbers (the canonical set is on thor-4; keep same-node discipline).

---

## 11. Conventions & gotchas (read before you touch anything)

- **Commit from the user's account; do NOT add Claude as co-author.** After meaningful changes, update the relevant README and commit + push (push from the local clone — see §8).
- **Never mix µs denominators across nodes** (1.8× speed variance). Quote 1.56× (thor-4) for prefill.
- **Liveness for background agents:** cluster file mtime + commits + the `kernel.cpp` diagnostics — never output-file size / proc-count alone.
- **The kernel build is on the cluster** (HipKittens/distributed-kernels mirror), not the repo's CMake; the repo `irisx/fused_moe/` is the *source of truth* — sync it into the mirror to build.
- **The cluster `~/iris` may have stale uncommitted changes** (redundant with origin); `git fetch && git reset --hard origin/subha/moe-dispatch-v0` to sync after this reorg (the old `b1_dispatch/` path is now `fused_moe/`).
- **Honesty:** the decode result is a marginal win at the fp8 floor — report it precisely, don't oversell.
