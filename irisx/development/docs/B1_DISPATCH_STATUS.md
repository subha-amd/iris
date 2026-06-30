# B1-dispatch — status, code locations, roadblock, and benchmarking plan

_Last updated: 2026-06-26. Author: lead-engineer session. All numbers below were measured on the
cv350 MI355X node (container `r1_c4`), np=8 unless noted, and independently re-run by the main agent._

---

## 1. What B1-dispatch IS (the new kernel)

B1-dispatch is the **production-shaped expert-parallel (EP8) MoE pipeline** for DeepSeek-R1 on one
8×MI355X node. It is the design the project re-centered on after the earlier fused V4 kernel was
shown to be uncompetitive (see "How we got here" below). It runs as TWO phases, back-to-back, on
each of 8 ranks:

```
PHASE 1  gather/pack/quant ONCE
  route_segments (who-owns-which-rows) ->
  multi-source remote FP8 activation + per-128 scale gather over IRIS (XGMI) from all 8 ranks ->
  local expert-major BM-padded packed buffer (+ route_reverse metadata for combine)

PHASE 2  local grouped GEMM
  grouped 32-expert HipKittens GEMM over the local packed buffer
  (A-stationary fused variant: gather each A tile once per K-tile, reuse across NSUB N-subtiles)
```

The key property vs the old V4: **each activation tile crosses the interconnect exactly ONCE**
(phase 1), then phase 2 is a purely local grouped GEMM. V4 re-crossed XGMI repeatedly and ran the
GEMM at low occupancy.

### Built by COMPOSITION, not rewrite
A hard-won lesson (three earlier from-scratch attempts P1/P2/P3 all failed by rewriting the gather):
B1-dispatch reuses VERIFIED components verbatim —
- the multi-source gather resolver from the EP8 gather component (Agent 03),
- the grouped 32-expert GEMM from the V5 grouped scheduler (Agent 02),
- the per-128 fp8 quant reference (e4m3fn).
The only new code is the glue (the gather-pack driver loop + the route builder + pybind + driver).

---

## 2. WHERE THE CODE IS

### The kernel itself (main artifact)
Git branch: **`agent/13-b1dispatch`** (in the iris fork; worktree at
`C:/Users/subvadla/repos/iris/.agents/13-b1dispatch/`).
Directory: **`irisx/b1_dispatch/`**
| file | what |
|---|---|
| `kernel.cpp` | both phases. Phase-1 `gather_pack_kernel` (multi-source gather→local packed fp8+scale). Phase-2 `micro_tk` (A-stationary fused grouped GEMM, the V1 default) + `micro_tk_baseline` (serial, the V0 reference). pybind module `tk_kernel` exposing `dispatch_gather_pack` + `grouped_gemm`. |
| `example.py` | np=8 driver: builds routing, fills per-rank activations, runs phase1+phase2, checks RMS-rel + zero-sentinel + a phase-1 isolation probe, times T_gather/T_gemm/T_total via cuda events. `FUSED=1` default. |
| `ep8_gather.h` | the VERIFIED multi-source row resolver (copied from Agent 03), `route_segment` ABI. |
| `build_tasks.py` | grouped task-list builder + adaptive-NSUB selector (CPU, copied from Agent 02). |
| `b1_dispatch_route.py` | NEW glue: builds 32-expert multi-source `route_segment`s over the BM-padded packed layout (ties Agent 02 packing to Agent 03 gather ABI). |
| `B1_DISPATCH.md`, `AGENT_REPORT.md` | design notes + build/run instructions. |

### The production baseline B2 (in progress, NOT yet working — see roadblock)
Same branch/worktree, directory **`irisx/b2_production/`**
| file | what |
|---|---|
| `b2_aiter.py` | calls `aiter.fused_moe` (production sort+quant+fmoe+combine). Currently SINGLE-GPU only and OOMing — this is the roadblock in §4. |

### Supporting / context docs
- `irisx/EXPERIMENT_LEDGER.md` — the authoritative committed record of every measurement (Phase A,
  Gate 1, B0/B1 sweep, B3/B4/B5 ablation, B1-dispatch V0/V1, etc.). **Read this for all numbers.**
- `irisx/B1_EXPLAINED.md` — what the B1-copy baseline is (and is NOT); IRIS-vs-memcpy mechanism.
- `irisx/abi/PRODUCTION_ABI.md` + `irisx/abi/route_abi.h` — the route metadata ABI.
- The verified components B1-dispatch composes: `irisx/ep8_gather/` (branch `agent/03-ep8-multisource`),
  `irisx/v5_grouped/` (branch `agent/02-grouped-scheduler`), `irisx/harness/` (branch
  `agent/01-strong-baseline`, has B0 `local_gemm` + B1 `dispatch_pack_quant_once`).

### On the node
Mirrored under `<HK_ROOT>/distributed-kernels/b1_dispatch/` (built as module `tk_kernel` via
`cmake -B build -DDK_BUILD=b1_dispatch`). Connection/build/run details: `NODE_ACCESS.local.md`
(gitignored).

---

## 3. WHAT WORKS (verified on device)

### Correctness — B1-dispatch V0 and V1 both PASS all 5 route distributions (np=8)
E=32 local experts, TOTAL_M=8192, N=2048, K=7168, MSRC=4096/rank, fp8 e4m3 per-128 block-scale.
Routes: uniform / zipf / one_hot / several_hot / many_empty.
- Phase-1 packed-A vs reference: **RMS = 0.000000** (0 rows mismatched) on every route.
- End-to-end output: **RMS_rel = 0.0037**, zero-sentinel exact, ~7111 rows gathered remotely over
  XGMI from multiple source ranks. **PASSED** on every route.

A real bug was found and fixed by the main agent during V0 bring-up: `build_row_seg_map` stored the
absolute segment index in a `signed char` (overflow at 127); with 32 experts (~420 segments) this
corrupted ~70% of packed rows (RMS 0.95). Fixed `signed char -> int`. (This was exactly the risk
Agent 03 had flagged.)

### Performance — V1 (fused phase2) is 2.1× faster than V0, all routes correct
M=8192 uniform: T_gather ≈ 216 µs, T_gemm 7428 µs (V0) → **3462 µs (V1)**, T_total 7641 → **3684 µs**.
GEMM throughput 32 → **69 TFLOP/s**. Across routes: 52–73 TFLOP/s end-to-end.

---

## 4. THE CURRENT ROADBLOCK — we cannot yet measure the *honest* end-to-end speedup

### The problem in one line
**We have a correct, working EP8 pipeline (B1-dispatch) but no valid baseline to divide it by yet.**

### Why
A speedup number is only meaningful against the right baseline. Two baselines we used earlier are
NOT the right bar for an EP8 pipeline:
- **B0** (local GEMM, no comm) — this is the compute *ceiling*, a diagnostic ("could our GEMM go
  faster?"). It has no gather, so it is not a pipeline. Comparing B1-dispatch to B0 is meaningless
  for end-to-end.
- **single-GPU `aiter.fused_moe`** — this is only the *local-compute slice* of production (sort +
  quant + fmoe + combine on ONE GPU's experts). It has NO cross-GPU EpDispatch/EpCombine, so it is
  not an EP8 pipeline either.

The correct baseline is the **production EP8 path on 8 GPUs**:
```
grouped_topk -> MORI EpDispatch (all-to-all, 8 GPUs) -> moe_sorting -> dynamic_quant
             -> aiter fmoe (per-GPU) -> MORI EpCombine (all-to-all back)
```
measured np=8 with the same routing/shapes, the same way B1-dispatch is measured. Call this **B2**.

### The concrete blocker hit while wiring B2
Both `aiter` and `mori` ARE importable in the container (good — the production ops exist). But:
- The **R1 vLLM server is holding ~96% of every GPU's VRAM** (~3 GB free per GPU).
- Production `aiter.fused_moe` allocates the full R1 expert weights (~3.5 GB for E=32, K=7168,
  inter=2048) → **HIP out-of-memory**.
- Our IRIS kernels coexist with the server only because they use tiny 256–512 MB symmetric heaps;
  the production aiter path needs real weight memory that the live server currently occupies.

So: the production EP8 baseline B2 needs VRAM (on all 8 GPUs) that the running R1 server is using.
This is an environment/resource constraint, not a code bug.

---

## 5. HOW WE PLAN TO BENCHMARK (the methodology, once B2 is unblocked)

### Apples-to-apples rules
- Same routing distribution, same shapes (E=32 local, K=7168, W13 g1u1 N=4096, W2 N=7168, top-k=8),
  same fp8 e4m3 per-128 block-scale, same np=8, on the same node.
- Time end-to-end per rank (the slowest rank governs), warmup + median over many iters.
- Correctness gate first (RMS-rel ~0.0033 vs a bf16 reference, zero-sentinel proving real remote
  movement) BEFORE trusting any timing.

### The comparisons that matter
1. **B1-dispatch T_total  vs  B2 (MORI EpDispatch + aiter fmoe + EpCombine) T_total**, np=8 — the
   real production question: does copy-once-grouped beat the production dispatch+fmoe path?
2. **B1-dispatch-serial  vs  B1-dispatch-overlapped** — internal: does overlapping the gather under
   the GEMM help, holding the dataflow fixed. (Overlap is a later step; V1 is still serial phase1→2.)
3. Sub-measurements for diagnosis only (NOT headline): T_gather vs MORI EpDispatch+EpCombine cost
   (the comm half); T_gemm vs aiter fmoe (the compute half); both vs B0 (the compute ceiling).

### Decision rule
B1-dispatch is a production candidate ONLY if its np=8 T_total beats B2's np=8 T_total on realistic
(captured C4/C6) route distributions. If it cannot, the conclusion is that bulk-synchronous
dispatch + production fmoe is the better dataflow, and effort should move to optimizing dispatch
rather than a custom grouped GEMM.

### Options on the table to unblock B2 (pending a decision)
- (a) free/standalone VRAM on all 8 GPUs (pause or shrink the R1 server's memory fraction) so the
  full-shape production EP8 path fits — the true baseline;
- (b) run the EP8 production path at reduced experts/inter_dim so weights fit in the ~3 GB free —
  real EP8 comm + real production kernels, scaled-down shapes;
- (c) measure only the MORI EpDispatch/EpCombine all-to-all cost (tiny tensors, fits) vs our IRIS
  gather cost — isolates the comm question without the weight-VRAM problem.

---

## 6. How we got here (one paragraph of context)
The original fused kernel V4 showed 1.82× — but only vs a deliberately weak baseline (B3, which
refetches A 32×). Against the strong copy-once baseline (B1-copy: copy A once + local GEMM, 291 µs),
V4 was actually 2.3× SLOWER. A B3/B4/B5 ablation then showed V4's entire win was comm/compute
*overlap*, not its A-stationary reuse (reuse alone = 1.00×). Three attempts to build a better
overlap kernel from scratch (P1/P2/P3) all failed by rewriting the gather. The project then
re-centered on the production-shaped B1-dispatch (this doc), built by composing verified parts —
which now works correctly and is being prepared for the honest B2 comparison described above.
```
```
