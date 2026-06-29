# DeepSeek-R1 MoE Fusion on 8×MI355X — Project Summary

> **CURRENT STATE (updated 2026-06-29) — read this first.**
> The "1.82× headline" in the historical section below was **refuted at Gate 1**: V4 beat only a
> deliberately-weak baseline (B3, refetch-A-32×); against the strong copy-once baseline (B1-copy) V4
> was ~2.3× *slower*. The whole V4 win was comm/compute *overlap*, not its A-stationary reuse. The
> project re-centered on the **production-shaped B1-dispatch pipeline** (gather/pack/quant ONCE →
> local grouped GEMM). See `EXPERIMENT_LEDGER.md` + `B1_DISPATCH_STATUS.md` for the authoritative,
> corrected record. The historical V0→V4 narrative is kept below for provenance, not as live guidance.
>
> **Active focus:** the b1_dispatch **phase-2 grouped GEMM** was slow (~32–69 TFLOP/s) because it
> reused the V4 64×64 producer/consumer body, NOT the project's proven B0-class GEMM (256×256 8-wave
> ping-pong, ~183 TFLOP/s). `grouped_b0/` is the fix — see its README.

## Repository layout (reorganized 2026-06-29)
```
irisx/
  grouped_b0/        ← ACTIVE: B0-class 8-wave grouped GEMM (the tile+schedule fix)   [NEW]
  b1_dispatch/       ← production-shaped EP8 pipeline: gather/pack/quant ONCE → grouped GEMM
  ep8_gather/        ← verified multi-source EP8 gather (phase-1 component)
  harness/           ← B0 (compute ceiling) + B1 (strong copy-once baseline) + benchmark methodology
  b2_production/     ← MORI EpDispatch + AITER/CK fmoe production baseline (wiring, VRAM-blocked)
  abi/               ← route / packed-layout ABI
  reference/         ← the two GEMM bodies grouped_b0 builds on (kept for reference)
      v2_hk_expert_gemm/   B0 256×256 8-wave ping-pong body (fmoe_expert_v2.cu)
      v5_grouped/          grouped task-list scheduler (micro_tk)
  archive/           ← superseded experiments (the v1–v4 line + parked schedule ablations)
      v2_1_iris_gather_gemm/ v3_fused_kernel/ v4_astationary_kernel/
      p1_tile_inbox/ p2_expert_pipeline/ p3_singlekernel/
      sched_4wave/ sched_8wave/ sched_xcd/ occ_variants/ lds_analysis/ cache_first_touch/
      results/             old V0..V4 *_RESULTS.md writeups
  examples/ benchmarks/ tests/ include/ cmake/   ← IRIS library (built by CMakeLists.txt; untouched)
```
Top-level docs kept here: `EXPERIMENT_LEDGER.md` (authoritative measurements), `B1_DISPATCH_STATUS.md`,
`B1_EXPLAINED.md`, `FMOE_LAYOUT.md`, `V2_HK_ANALYSIS.md` (grouped_b0 design basis), `AGENT_COMMON.md`.

---

## Historical narrative (V0 → V4) — kept for provenance
> One-page map of the early effort: from profiling the production decode path to a fused
> tile-level communication+compute kernel. **NOTE:** the "1.82×" framing here is superseded — see the
> CURRENT STATE banner above. Paths moved in the 2026-06-29 reorg: `v2_hk_expert_gemm/` →
> `reference/`, and `v2_1_*`, `v3_*`, `v4_*`, plus the `*_RESULTS.md` files → `archive/`.

## The thesis
In MoE decode with expert parallelism, tokens must be moved across GPUs to their experts
(the "gather/dispatch"), then the expert GEMM runs. Production does this in **two serial
phases** — move+quantize (interconnect busy, matrix cores idle), then GEMM (vice-versa).
The contribution: a **single fused kernel** where the expert GEMM's producer warps gather
token tiles from remote GPUs over IRIS *while* consumer warps do the matmul — overlapping
communication with compute. On MI355X this matters because the inter-GPU link (~128 GB/s)
is ~16× slower than HBM (~7.2 TB/s), so exposed comm is expensive and worth hiding.

## The production decode path we're attacking
```
grouped_topk → EpDispatch → opus_moe_sorting ×2 → dynamic_quant → fmoe (expert GEMM)
              └──────────── pre-GEMM data prep (~46 µs/inst) ─────────┘   └─ compute ─┘
```
Profiling (C4/C6 configs): EP gather+scatter is ~14–16% of decode GPU time; the
gather+pack+quant envelope ~24%. Full profiling write-up is in the `kernels-testing` repo
(`EP_PROFILING_RESULTS.md`); design rationale in `KERNEL_PLAN.md` + `PRIOR_ART.md`.

## The stages — what each did, the result, and where to look

| stage | what it does | result (8×MI355X, verified) | primary files |
|---|---|---|---|
| **V0** | device-side dispatch+pack: route top-k tokens to expert-owning GPUs, write expert-major. Two variants: V0a remote-atomic slot claim, V0b precomputed-offset (no atomics). | correct; V0b **300 GB/s** | `benchmarks/moe_dispatch_pack.hip`, `tests/test_moe_dispatch_pack.hip`, `V0_RESULTS.md` |
| **V1** | V0 + fold per-128-group FP8 e4m3 quant into the remote store (halves wire bytes). | correct; ~46 µs/inst. **Finding: quant-bound** (FP8 convert > store), so byte-halving only gave ~23% | `benchmarks/moe_dispatch_pack_quant.hip`, `tests/test_moe_dispatch_pack_quant.hip`, `V1_RESULTS.md` |
| **V1.1** | tried to fix V1: software-pipeline quant/store + packed fp8x2 convert. | **negative result** (~2%): store cycles −66% but V1 was 99% quant; fp8x2 is a software wrapper (no HW packed convert in ROCm 7.2.4) | `benchmarks/moe_dispatch_pack_quant_pipelined.hip`, `V1_1_RESULTS.md` |
| **fmoe layout** | source-verified what the production `fmoe_bf16_blockscaleFp8` GEMM actually expects. | confirmed our fp32/128-group/OCP-e4m3 choices; found production uses row-major A + sort-index, not our expert-major pack (informs V2 direction) | `FMOE_LAYOUT.md` |
| **V2 analysis** | feasibility: does HipKittens build for gfx950? where's the IRIS bridge? | HK builds+runs for gfx950; `distributed-kernels/bf16_gemm` already carries an IRIS device view | `V2_HK_ANALYSIS.md` |
| **V2** | our OWN HipKittens expert GEMM (single-GPU) consuming the V1 FP8 buffer: dequant preamble → bf16 8-wave MMA core. | correct (RMS-rel 0.0037), ~126 TFLOP/s | `v2_hk_expert_gemm/fmoe_expert_v2.cu`, `v2_hk_expert_gemm/README.md`, `V2_RESULTS.md` |
| **V2.1** | the gather mechanism: a producer/consumer GEMM whose producer warps pull A-tiles from a **remote** GPU's IRIS heap during compute. | correct (RMS-rel 0.0033); zero-sentinel proves cross-GPU gather. (mechanism proof, no perf claim) | `v2_1_iris_gather_gemm/kernel.cpp`, `.../example.py`, `.../README.md`, `V2_1_RESULTS.md` |
| **V3** | the FULL fusion: merge V1 (FP8 dequant) + V2.1 (remote gather) into one kernel — gather fp8 tiles from remote GPU, dequant in producer, MFMA in consumer, all overlapped. Fair same-code overlap-removed baseline. | correct; **1.02× over unfused.** Finding: gather-bound — A re-fetched 32× (once per N-block) | `v3_fused_kernel/kernel.cpp`, `.../example.py`, `.../README.md`, `V3_FUSED_RESULTS.md` |
| **V4** | fix V3's redundant gather: **A-stationary, wide-N** — each block gathers A[BM,BK] once per K-tile, reuses across NSUB=8 N-subtiles (cross-GPU crossings 32×→4×). | **1.80–1.83× over unfused** at M=1024,N=2048,K=7168 (correct, re-run 3×, stable). Wins for M≥512; loses at M≤256 | `v4_astationary_kernel/kernel.cpp`, `.../example.py`, `.../README.md`, `V4_ASTATIONARY_RESULTS.md` |

## How to read the kernels (suggested order)
1. **`benchmarks/moe_dispatch_pack.hip`** (V0) — start here: the IRIS remote-store + expert-major
   layout + slot-claim. Simplest use of the symmetric-heap `iris_view.store/fetch_add`.
2. **`benchmarks/moe_dispatch_pack_quant.hip`** (V1) — adds the per-128 FP8 quant in the store path.
3. **`v2_hk_expert_gemm/fmoe_expert_v2.cu`** (V2) — the HipKittens GEMM: tile types, dequant preamble,
   bf16 8-wave MMA core, send_counts masking. Single-GPU, no comm.
4. **`v2_1_iris_gather_gemm/kernel.cpp`** (V2.1) — the key new idea: `gather_A_tile` pulls A from a
   remote rank via `iris_ctx.load(ptr, src_rank)`; producer/consumer warp split.
5. **`v3_fused_kernel/kernel.cpp`** (V3) — V1+V2.1 combined; note `micro_tk` (fused) vs
   `micro_tk_baseline` (same code, overlap removed) for the head-to-head.
6. **`v4_astationary_kernel/kernel.cpp`** (V4) — the winner; see the NSUB N-subtile reuse loop that
   makes each remote A-tile cross the interconnect once per K-tile instead of once per N-block.

Each kernel dir / RESULTS.md has its own build+run commands and the verified numbers.

## Headline result
**Fused A-stationary kernel = 1.82× faster than the unfused two-phase baseline** at M=1024,
N=2048, K=7168, identical numerics (RMS-rel 0.00331). The fused kernel hides the cross-GPU
token gather under the expert matmul; the baseline pays them serially.

## Honest limits / where it wins
- **Wins in the batched-decode regime (M ≥ 512 tokens/expert):** 1.05–1.83×.
- **Loses at small M (≤256):** 0.85–0.90× — A-stationary removes the redundant blocks that were
  accidentally hiding interconnect latency, so a too-small grid starves the 256 CUs. Small-M needs
  a different tactic (cache-on-first-touch).
- At the winning shape it's now occupancy-capped (2 waves/SIMD from 8 fp32 accumulators) — more
  headroom likely with accumulator/occupancy tuning.

## Deferred / next steps
restore occupancy at NSUB=8; adaptive NSUB by shape; cache-on-first-touch (win at small M too);
real multi-rank expert routing (per-tile src_rank, not fixed rank 0); full expert FFN
(gate/up → SiLU → down-proj); fp8 weights; scale 2-rank → 8-rank; SDMA offload for bulk movement.

## Build/run + node access
All kernels build inside a HipKittens checkout (`distributed-kernels/`) and run inside the ATOM
container via mpirun with `--mca pml ob1 --mca btl self,vader` (the node's IB can't register
memory). gfx950 = `GPU_TARGET=CDNA4`. Node connection details are in `NODE_ACCESS.local.md`
(gitignored — not in this fork). Per-stage commands are in each RESULTS.md.
