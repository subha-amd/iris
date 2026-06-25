# Agent 07 — register pressure / scratch spill / occupancy — REPORT

## Status of resource numbers
**All VGPR/AGPR/SGPR/LDS/scratch/occupancy figures are ANALYTICAL PREDICTIONS.** Node SSH was not
reachable on this run (the harness blocked the ssh command), so no static resource report was
extracted. On-node compile + static extraction is **[NEEDS-NODE — main agent to confirm]**.
Predictions are derived directly from the HipKittens cdna4 register-tile headers (`rt_base.cuh`,
`rt_shape.cuh`, `rt.cuh`) — see `irisx/occ_variants/REGISTER_OCCUPANCY.md` §1 for the math, and
`RESOURCE_TABLE.csv` for the numeric grid.

## Root cause (one line, evidence-based)
`rt_fl<64,16,col_l,rt_16x16> C_accum[NSUB=8]` = 8 × 16 = **128 live VGPR** of float accumulator per
consumer warp (NSUB is the multiplier); + a_frag 32 + b_frag 8 + ~54 producer overhead ⇒ ~222 VGPR,
~32 over the 256-VGPR/occ-2 budget ⇒ ~160 B scratch spill. Calibration: NSUB8→222, NSUB4→158
(Δ64 = 4 accumulators × 16 VGPR), BM32→142 — fixed overhead ≈ 54 throughout.

## CRITICAL reframing finding
On the fused path, occupancy is **LDS-bound, not VGPR-bound**. `LDS_per_block` at NSTAGE=2/NSUB=8 =
**~145 KB**, so only **1 block/CU** fits (assuming CDNA4 LDS_CAP=160 KB [NEEDS-NODE]). Lowering VGPR
alone (e.g. bm32_nsub8: 222→142) does NOT raise occupancy — you must also cut **LDS** (NSTAGE=1 or
smaller NSUB). VGPR cuts still pay off by removing the scratch spill (a real per-iter cycle cost) and
by *unlocking* higher occupancy once LDS is relieved.

## Files changed / added (all under irisx/occ_variants/, none touch canonical v3/v4)
- `v4_variant_base.cuh` — single parameterized source (config via -D); shared by the -D variants.
- `v4_bm32_nsub8/kernel.cpp`, `v4_bm64_nsub4/kernel.cpp`, `v4_bm32_nsub6/kernel.cpp`,
  `v4_bm64_nsub6/kernel.cpp`, `v4_nstage1/kernel.cpp` — thin -D wrappers over the base header.
- `v4_cons8_nsub8/kernel.cpp` — documented INFEASIBLE (CONS_N=8 ∤ 16; guarded stub).
- `v4_staged_consumer/kernel.cpp` — STRUCTURAL variant (own full source): 4 producers + 8 consumers,
  each consumer owns 1 of NSUB subtiles (full BN cols) → acc 128→64 VGPR, no spill.
- `v4_staged_consumer_nstage1/kernel.cpp` — staged-consumer + NSTAGE=1 (LDS 145→73 KB → 2 blocks/CU).
- `example.py` — shared head-to-head runner (config via env M/K/N; MAIN AGENT runs it on-device).
- `RESOURCE_TABLE.csv`, `REGISTER_OCCUPANCY.md` — the analysis deliverables.

## Build commands (on-node; subagent compile is OPTIONAL and was not possible this run)
The distributed-kernels build auto-discovers `*/kernel.cpp`. Mirror `irisx/occ_variants/` under
`<HK_ROOT>/distributed-kernels/` (keep node copy == repo copy). For each variant <dir>:
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""   # subagent/static-only safety
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_07 -DDK_BUILD=<dir> -DIRIS_HIP_ARCHITECTURES=gfx950
  flock /tmp/mi355x_compile.lock -c "cmake --build build_07 -j8 --target <dir>"'
```
Static resource extraction (the deliverable to turn PREDICTED → MEASURED):
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels/<dir>
  hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage -c kernel.cpp -o /tmp/<dir>.o \
    -I<HK_ROOT>/include -I<iris include paths>   # grabs VGPR/SGPR/LDS/scratch from the diagnostic
  # or, on the built .hsaco: roc-obj / readelf -> .note metadata (vgpr/agpr/sgpr/lds/scratch)'
```

## PROPOSED GPU test command (MAIN AGENT ONLY, serialized under flock)
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/<dir>
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    for M in 8 32 64 128 256 512 1024; do
      M=$M K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
        -np 2 python3 example.py
    done"'
```
Also sweep N=4096. (`--mca pml ob1 --mca btl self,vader` is MANDATORY on this node.)

## Expected output / correctness criteria
example.py prints FUSED + BASELINE: RMS_rel (must be < 0.10; canonical V4 ≈ 0.0033), `local_A_zero`
must be True and `C_zero` False (zero-sentinel proves the remote gather actually happened), and
us/iter + TFLOP/s + speedup. A variant is CORRECT iff both lines PASS at every M. Then compare
fused us/iter across variants at each M to pick the occupancy winner.

## PRIORITIZED candidates for main-agent GPU testing (with rationale)
1. **v4_staged_consumer_nstage1** — predicted ~122 VGPR (no spill) AND 2 blocks/CU (LDS 73 KB). The
   only single design that fixes BOTH register pressure and the LDS occupancy cap. Highest upside;
   risk = lost double-buffer overlap (measure vs #4).
2. **v4_bm64_nsub4** — ~158 VGPR, LDS 73 KB → **2 blocks/CU** without a structural rewrite (pure -D).
   Safest occupancy win; cost is 2× A interconnect traffic (smaller N_PER_BLOCK). Good A/B vs canonical.
3. **v4_nstage1** — isolates the LDS lever (canonical tiling, 2 blocks/CU) so you can measure the
   double-buffer-overlap-vs-occupancy trade cleanly before trusting #1.
4. **v4_staged_consumer** (NSTAGE=2) — isolates the register lever (spill removed, occ still 1) so the
   #1 result can be attributed to LDS vs VGPR. Also the cleanest "fixes spill, keeps overlap" point.
5. **v4_bm32_nsub8** — removes the spill, halves a_frag; occupancy unchanged (LDS-bound) but useful to
   confirm the spill-removal cycle savings in isolation; grid M-dim ×2 may help mid-M fill.
6. (low) **v4_bm32_nsub6 / v4_bm64_nsub6** — sweep completeness only; 384∤2048 causes partial-N-block
   wasted MFMAs (no tail mask) — expect a correctness-preserving but efficiency-losing result.

## Assumptions
- CDNA4/MI355X LDS_CAP = 160 KB/CU and VGPR budget = 512/SIMD (occ = floor(512/VGPR)). [NEEDS-NODE]
- ~54 VGPR fixed producer overhead (from canonical V4 calibration). AGPR ≈ 0 (VGPR-accumulate MFMA).
- The build auto-discovers `*/kernel.cpp`; example.py imports a per-dir `tk_kernel` module.

## Known risks
- If LDS_CAP is actually 64 KB, the NSTAGE=2 fused path does not fit at all → BK/NSUB must shrink
  (larger redesign, out of scope). Canonical V4 reportedly running is evidence LDS_CAP ≥ 145 KB.
- nstage1 variants lose producer/consumer overlap (PREFETCH=0) — may regress latency despite occ=2.
- *_nsub6 variants: N_PER_BLOCK=384 does not divide 2048/4096 → partial last N-block does wasted
  MFMAs on padding subtiles (correct output, lower efficiency; not tail-masked).
- Small M (≤64) is GRID-occupancy-starved (only N-block-count blocks); no per-block register/LDS
  change fixes that — needs split-K or smaller BM (deferred; split-K excluded per task since it
  needs partial-C storage + reduction).
- Numbers are PREDICTED; do not pick a winner from this doc — GPU measurement required (per task).

## Did NOT do (by rule)
No GPU runs, no mpirun, no rocprof, no HIP execution. No split-K. Did not pick a winner.
```
