# BENCHMARKING HANDOFF — grouped_b0 vs the unfused production MoE path

**Audience:** a future agent (likely with node + perfetto-trace access) who needs to benchmark the
new `grouped_b0` GEMM and the b1_dispatch pipeline against the production "unfused" MoE expert path.
**Author context:** written 2026-06-29 off-node (no GPU, no trace on this machine). Everything here is
derived from the source-verified ABI docs in this repo; the trace-extraction steps are instructions
for whoever has the trace.

---

## 0. The two comparisons (the whole job)

```
LEVEL 1 — GEMM only (do first; needs NO production VRAM):
    grouped_b0 (dequant + B0 8-wave GEMM)   vs   production fmoe_fp8_blockscale_g1u1
    same M_e distribution, same K/N, same dtype. Isolates "is my expert GEMM competitive."

LEVEL 2 — full expert phase (after grouped_b0 is wired into b1_dispatch):
    b1_dispatch  phase1 ep8_gather  +  phase2 grouped_b0          (ours)
        vs
    EpDispatch + moe_sorting + dynamic_quant + fmoe               (production)
    Isolates "does copy-once-then-fast-local-GEMM beat the production dispatch+fmoe pipeline."
```

The headline the mentors want is **Level 2**. Level 1 is the prerequisite: if the GEMM isn't fast,
Level 2 can't win, and you won't be able to tell whether a Level-2 loss is the GEMM or the dataflow.

---

## 1. The production "unfused" path — exact kernels + shapes (source-verified)

From `abi/PRODUCTION_ABI.md` (§1, §4) and `FMOE_LAYOUT.md` — all CONFIRMED from the ATOM/aiter source
in image `rocm/atom-dev:vllm-v0.22.0-nightly_20260610`:

**Kernel chain (decode MoE expert region):**
```
grouped_topk → EpDispatch → moe_sorting (×2) → dynamic_quant → aiter.fmoe_fp8_blockscale_g1u1 → EpCombine → wv_splitk
└ router ┘   └─ cross-GPU ─┘  └──── pre-GEMM data prep ────┘   └──── the expert GEMM ────┘   └ cross-GPU ┘
```
- Production op: **`aiter.fmoe_fp8_blockscale_g1u1`**, reached from ATOM `moe.py` →
  `aiter.fused_moe.fused_moe(..., QuantType.per_1x128)` (`fused_moe.py:769-770`).

**Shapes (binding — `PRODUCTION_ABI.md` §4):**

| GEMM | A (activations) | B (weights, preshuffled) | K | N | epilogue |
|------|-----------------|--------------------------|---|---|----------|
| **fc1 (W13, gate+up fused "g1u1")** | fp8 `[M_e, 7168]` | `w13[e, 4096, 7168]` | 7168 | **4096** | split 4096→2×2048, `SiLU(gate)·up` → 2048 |
| **fc2 (W2, down)** | fp8 `[M_e, 2048]` | `w2[e, 7168, 2048]`  | 2048 | **7168** | none → bf16 |

Constants: EP=8 ranks, **32 local experts/GPU**, top-k=8, H=7168, GROUP=128 → 56 fp8 block-scale
groups, **fp8 = OCP e4m3 (float8_e4m3fn, max 448) on gfx950**, scales = fp32, output = bf16.

---

## 2. The fairness contract (identical inputs — this is what makes a ratio meaningful)

Both sides MUST see the same:
- **Shapes:** the fc1 (K=7168,N=4096) and/or fc2 (K=2048,N=7168) above. ⚠️ **grouped_b0.cu currently
  defaults N=2048** — that is only ONE gate/up projection, i.e. HALF of fc1. For an apples-to-apples
  fc1 number, run grouped_b0 at **N=4096**; for fc2, at N=7168,K=2048. (The b1_dispatch wiring in §5
  instantiates the B0 GEMM for all three (N,K) combos so the host just picks.)
- **Per-expert row counts `M_e`:** use the REAL decode distribution from the trace (see §3), not a
  made-up uniform. Same `M_e` on both sides.
- **dtype + quant:** fp8 OCP e4m3, fp32 per-128 scales, bf16 out. Same on both sides.
- **Correctness gate FIRST:** RMS-rel vs a bf16 reference must pass (~0.003–0.03) before any timing is
  trusted. grouped_b0.cu already self-checks; the production side is correct by construction.

Fairness comes from same-inputs, **not** from running inside vLLM.

---

## 3. The perfetto trace is your BASELINE SOURCE, not your denominator

Do NOT divide your kernel time by the whole-model trace. The trace's two jobs:

1. **Production baseline numbers, for free.** The trace already recorded `fmoe_fp8_blockscale_g1u1`
   (and `EpDispatch` / `moe_sorting` / `dynamic_quant` / `EpCombine`) **durations** under real 8-GPU
   decode. Read those off directly — you do NOT need to re-run AITER (which is VRAM-blocked, see §6).
2. **The realistic `M_e` distribution.** The per-call grid dims / token counts give the actual
   per-expert row counts during decode. Feed that same distribution into your microbenchmark.

**What to extract from the trace (MoE decode region, steady-state decode step):**
- kernel **names** present (confirm the §1 chain),
- per-call **duration** of `fmoe...g1u1` (→ Level-1 baseline), and of the dispatch+sort+quant chain +
  EpDispatch/EpCombine (→ Level-2 baseline),
- per-call **grid/block dims** or token counts (→ the `M_e` distribution and total routed rows),
- the **decode batch size** the trace ran at (so the `M_e` you feed matches that operating point).

If the trace only gives durations without dims, correlate `M_e` with the decode batch size / known R1
routing (top-k=8 over 256 experts, 32 local/GPU). Note in results what was assumed.

---

## 4. LEVEL 1 recipe — grouped_b0 standalone (no production VRAM needed)

`grouped_b0/grouped_b0.cu` is self-contained (single-GPU, no MPI/IRIS). Build + run:
```bash
/opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
    -I<HK_ROOT>/include -I/opt/rocm/include/hip grouped_b0/grouped_b0.cu -o grouped_b0
./grouped_b0     # prints TFLOP/s (real + padded) and RMS/contamination per case
```
Steps:
1. **First green it** (correctness + a TFLOP/s number) at the default shapes. Confirm the headline
   beats micro_tk's ~69 TFLOP/s and approaches B0's ~183 (see `EXPERIMENT_LEDGER.md`).
2. **Then make it match production fc1/fc2:** edit the `N`/`K` constants (or add cases) to
   `N=4096,K=7168` (fc1) and `N=7168,K=2048` (fc2), and set the `M_e` vector to the trace distribution.
3. **Compare** grouped_b0's per-call time to the trace's `fmoe...g1u1` duration at the same `M_e`.
   Record both real-row and padded TFLOP/s (padding waste depends on BM vs M_e — see §7).

Deliverable: a table of {shape, M_e, grouped_b0 µs, trace fmoe µs, ratio, RMS}.

---

## 5. LEVEL 2 recipe — b1_dispatch with the B0 phase-2 (the pipeline comparison)

The B0 GEMM is now wired into `b1_dispatch/kernel.cpp` as an ADDITIVE phase-2 path (the old `micro_tk`
is untouched). New pybind function:
```
tk_kernel.grouped_gemm_b0(a, sc, b, c, tasks_b0, Mpacked, N, K, num_tasks)
```
- `a`,`sc` = the packed fp8 buffer + fp32 scales (phase-1 output) — SAME buffers `grouped_gemm` reads.
- `b` = bf16 weights `[E*N, K]`, `c` = bf16 out `[Mpacked, N]`.
- `tasks_b0` = the **B0 task list** (TASK_W=4: `expert, m_tile, n_tile, expert_row_begin`), BM=256.
  Build it with the helper `b1_dispatch/b0_tasks.py::build_b0_tasks(rows_per_expert, N, BM=256)`.
- The dispatch internally does the **dequant preamble** (packed fp8 → bf16 scratch) then the B0 GEMM,
  switching on `(N,K)` ∈ {(2048,7168),(4096,7168),(7168,2048)}.

**⚠️ Phase-1 padding must match BM=256.** The B0 GEMM needs each expert's packed region 256-aligned,
but the current phase-1 gather pads to `GP_BM=64`. So for Level 2 you must build the packed buffer
+ tilemeta with **BM=256** (re-pad the route layout), so `expert_row_begin` is a multiple of 256.
This is the one remaining integration step (the gather body itself doesn't change — only the host
route/packing granularity). Until then, run the B0 path on a 256-padded buffer built directly.

Then time, per rank (slowest governs), with the same cuda-event method as `b1_dispatch/example.py`:
- **ours:** `dispatch_gather_pack` (phase 1) + `grouped_gemm_b0` (phase 2).
- **production:** the §1 chain (or its trace durations).
Same routing, same shapes, correctness-gated.

---

## 6. The VRAM blocker (why Level 2's live baseline is hard) + workarounds

`B1_DISPATCH_STATUS.md` §4: the live R1 vLLM server holds ~96% of every GPU's VRAM (~3 GB free), so a
fresh `aiter.fused_moe` allocating full R1 weights (~3.5 GB) OOMs. Options:
- **(a) Use the trace durations** for the production side (no live AITER run needed) — preferred for a
  first pass; see §3.
- **(b)** Free/shrink the server's memory fraction to run the real EP8 production path at full shape.
- **(c)** Run the production path at reduced experts/inter-dim so weights fit in the free VRAM (real
  kernels, scaled-down shapes).
- grouped_b0 standalone (Level 1) needs almost no VRAM (B is `[E*N,K]`; E=8 → ~235 MB), so it runs
  alongside the server fine.

---

## 7. Known gotchas (read before trusting a number)

- **N=2048 ≠ fc1.** Real fc1 is the fused g1u1 at N=4096 (§2). Don't quote a 2048 number as "fc1."
- **Scale layout trap.** IRIS/our buffer is **token-major** `[M,56]`; production fmoe reads
  **group-major (transposed)** `[56,M]` (`FMOE_LAYOUT.md` §2.1). For OUR own GEMM (grouped_b0) we keep
  token-major end-to-end, so it's internally consistent — but if you ever feed AITER fmoe OUR buffer
  directly, you MUST transpose the scales or you get silent garbage (no crash).
- **BM=256 padding waste.** grouped_b0 pads each expert to 256 rows. For small `M_e` (decode-light
  experts) this wastes MFMA on zero rows — report BOTH real-row and padded TFLOP/s. If padding waste
  is large at the trace's `M_e`, that motivates the BM sweep (a re-derived schedule for BM<256).
- **Graph capture.** Production decode is HIP-graph-captured (`PRODUCTION_ABI.md` §6): buffers must be
  pointer-stable, no host-sync inside the captured region, grid fixed. Only matters for a true
  in-server Level-2 integration, not for the microbenchmark.
- **fc1 N-split order** (gate|up vs interleaved) and exact scale `M_pad` are still `[NEEDS-NODE]`
  (`PRODUCTION_ABI.md` §7 Q1/Q3) — needed only for byte-feeding production fmoe, not for our own GEMM.

---

## 8. Status checklist (update as you go)

- [ ] grouped_b0.cu builds on node (gfx950) — see §4 step 1
- [ ] grouped_b0 correctness PASS (RMS<0.05, contamination 0) on ragged case
- [ ] grouped_b0 TFLOP/s recorded vs micro_tk ~69 / B0 ~183
- [ ] grouped_b0 re-run at fc1 (N=4096) and fc2 (N=7168,K=2048) shapes
- [ ] trace extraction: fmoe + dispatch/sort/quant durations + M_e distribution captured
- [ ] LEVEL 1 table {shape, M_e, ours µs, trace fmoe µs, ratio} produced
- [ ] b1_dispatch B0 path built (`grouped_gemm_b0`) + b0_tasks.py task list
- [ ] phase-1 gather re-padded to BM=256 (Level-2 prerequisite)
- [ ] LEVEL 2 table {ours phase1+phase2 vs production chain} produced
- [ ] EXPERIMENT_LEDGER.md updated with all of the above
