# V4 A-stationary — register pressure / scratch spill / occupancy analysis (Agent 07)

> ALL resource numbers in this document are **ANALYTICAL PREDICTIONS**. Node SSH was not reachable
> this run, so no static `-Rpass-analysis=kernel-resource-usage` report was extracted.
> Every "VGPR/AGPR/SGPR/LDS/scratch/occupancy" figure is marked **PREDICTED** and must be
> confirmed on-node by the main agent. On-node compile is **[NEEDS-NODE]**.

## 1. Root cause of 222 VGPR / 160 B scratch spill (with evidence)

**One line:** the `rt_fl<BM=64,CONS_N=16,col_l,rt_16x16>` accumulator array `C_accum[NSUB=8]` puts
**128 live VGPR** of float accumulator in every consumer warp (NSUB is the multiplier), and with
a_frag (32) + b_frag (8) + producer overhead (~54) the single `__launch_bounds__(256,1)` function
needs ~222 VGPR, ~32 over the occ=2 budget of 256, so the compiler spills the coldest accumulator
base-tiles to ~160 B of scratch.

**Evidence (register math read from HipKittens cdna4 headers, not guessed):**
- `rt_base.cuh`: `registers_per_thread = packed_per_thread * sizeof(dtype)/4`.
  - `rt_16x16` (`rt_shape.cuh`: 16x16, stride 4, 256 elem, 4 elem/thread): float → packed float2,
    packed_per_thread=2, **4 VGPR per base tile**.
  - `rt_16x32` (16x32, stride 8, 512 elem, 8 elem/thread): bf16 → bf16_2, packed_per_thread=4,
    **4 VGPR per base tile**.
- `rt.cuh`: a tile is `tiles[height][width]`, `height=rows/16`, `width=cols/base_cols`.
  - `C_accum = rt_fl<64,16,col_l,rt_16x16>`: height=64/16=4, width=16/16=1 → 4 base × 4 = **16 VGPR**;
    `C_accum[NSUB=8]` → **128 VGPR**. ← dominant, the spill driver.
  - `a_frag = rt_bf<64,64,row_l,rt_16x32>`: height=4, width=2 → 8 base × 4 = **32 VGPR** (live across
    the NSUB inner loop).
  - `b_frag = rt_bf<16,64,row_l,rt_16x32>`: height=1, width=2 → 2 base × 4 = **8 VGPR** (short-lived).
- `__launch_bounds__(NUM_THREADS,1)` ⇒ one function; producer (uint4 packed gather + fp32 scale +
  address math) needs ~50–54 VGPR; consumer needs 128+32+8+~10 = ~178; the function takes the max
  but the *spill* is driven by the 128-VGPR accumulator that must stay live across all K-tiles.
- **Calibration (the ~54 fixed-overhead constant):** canonical NSUB=8 → 222; NSUB=4 → 158
  (Δ = 64 = exactly the 4 dropped accumulators × 16 VGPR); BM=32/NSUB=8 → 142
  (accumulators halved 128→64, Δ=80, residual ≈ a_frag halving). Fixed remainder ≈ 54 in all three.
- 160 B scratch = compiler over the 256-VGPR/occ-2 budget spilling the coldest accumulator base
  tiles (4 B/elem × a couple of base tiles' worth of cold lanes).

## 2. THE constraint that actually limits occupancy here: **LDS, not VGPR**

This is the most important finding and it reframes every variant.

`LDS_per_block = NSTAGE*(sizeof(ST_A) + NSUB*sizeof(ST_B)) + 1024`, with `st_bf<64,64> = 8192 B`:
- NSTAGE=2, NSUB=8 → `2*(8192 + 8*8192)+1024 = 148480 B ≈ 145 KB`.

CDNA4/MI355X LDS is **assumed 160 KB/CU** (up from 64 KB on CDNA3) — **[NEEDS-NODE to confirm]**.
- At 145 KB/block only **1 block fits per CU** (2 blocks = 290 KB > 160 KB), regardless of VGPR.
- ⇒ For every NSTAGE=2 / NSUB=8 variant, occupancy is **LDS-capped at 1 block/CU**. Lowering VGPR
  from 222→142 (e.g. `v4_bm32_nsub8`) raises the *VGPR-permitted* waves but the kernel still gets
  **1 block/CU** because LDS is the binding constraint. Register relief alone does NOT buy occupancy
  on the fused path.
- If LDS_CAP is actually 64 KB, the fused kernel does not fit at all at NSTAGE=2 and BK or NSUB
  must shrink — a separate, larger redesign. (The canonical V4 reportedly runs, which is itself
  evidence that LDS_CAP ≥ 145 KB on this node.)

**Implication:** the real occupancy levers on the fused path are the ones that cut **LDS**:
`NSTAGE=1` (halves LDS to ~73 KB → 2 blocks/CU) and smaller `NSUB` (fewer B buffers). VGPR variants
matter only once LDS is relieved, or to remove the scratch spill (which costs cycles even at occ 1).

## 3. Per-variant analysis (all PREDICTED — see RESOURCE_TABLE.csv for the numeric grid)

| variant | VGPR | LDS | occ (blocks/CU) | what it tests |
|---|---|---|---|---|
| canonical (ref) | 222 (+160 B spill) | 145 KB | 1 (LDS-bound) | baseline |
| **v4_bm32_nsub8** | ~142 | 145 KB | **1 (LDS-bound)** | kills the spill, halves acc & a_frag; grid M-dim ×2. Occupancy unchanged unless LDS also cut. Best *spill-removal* candidate. |
| **v4_bm64_nsub4** | ~158 | **73 KB** | **2** | halves accumulators AND LDS (NSUB 8→4) → genuinely reaches 2 blocks/CU. Cost: A crosses interconnect 2× more (N_PER_BLOCK halved) — trades gather amortization for occupancy. |
| v4_bm32_nsub6 | ~120 | 112 KB | 1 (LDS-bound) | lowest VGPR, but N_PER_BLOCK=384 ∤ 2048 → 6th N-block 2/6 valid subtiles (wasted MFMAs unless tail-masked). |
| v4_bm64_nsub6 | ~190 | 112 KB | 1 (LDS-bound) | borderline VGPR, same 384∤2048 tail waste; weakest of the set. |
| v4_cons8_nsub8 | — | — | — | **INFEASIBLE**: CONS_N=8 not divisible by 16 → `rt.cuh` `cols % base_cols == 0` static_assert fires. Cannot express sub-16 accumulator columns with HK cdna4 base shapes. |
| **v4_nstage1** | ~222 (+spill) | **73 KB** | **2** | pure LDS relief → 2 blocks/CU at canonical tiling. Cost: PREFETCH=0, loses producer/consumer double-buffer overlap (every K-tile stalls on its gather). Tests whether 2× occupancy hides the lost overlap. |
| **v4_staged_consumer** (4P+8C) | ~122 | 145 KB | 1 (LDS-bound) | partitions the NSUB axis across 8 consumer warps (each owns 1 subtile, full BN cols) → acc 128→64 VGPR, no spill, the LEGAL way to get the cons8 register win. Occupancy still LDS-bound at NSTAGE=2. |
| **v4_staged_consumer + nstage1** | ~122 | **73 KB** | **2** | the combination: low VGPR (no spill) AND 2 blocks/CU. Strongest single design if the lost double-buffer overlap is tolerable. **[needs the wrapper define NSTAGE=1]** |

### Output tile & grid (K=7168, num_k_tiles=K/BK=112)
- Block output tile = `BM × N_PER_BLOCK` = `BM × (NSUB*BN)`.
  - canonical/nstage1/staged: 64×512. bm32_nsub8: 32×512. bm64_nsub4: 64×256. *_nsub6: BM×384.
- Grid = `ceil_div(N, NSUB*BN) × ceil_div(M, BM)`.
  - N=2048: N-blocks = 4 (NSUB8), 8 (NSUB4), 6 (NSUB6, last partial). N=4096: 8 / 16 / 11.
  - M-blocks = ceil(M/BM); for M∈{8,16,32,64,128,256,512,1024} and BM=64 → {1,1,1,1,2,4,8,16};
    BM=32 → {1,1,1,2,4,8,16,32}. Small M (≤64) → 1 M-block; the grid is then N-block-count wide
    only (4–16 blocks) → severely under-fills the CU array — small-M is occupancy-starved at the
    GRID level, which no per-block register change can fix (needs split-K or smaller BM; split-K
    deferred — would require partial-C storage + a reduction pass, out of scope here).

## 4. Scratch / AGPR notes
- AGPR: HK MFMA path uses VGPR accumulators (`rt_fl` in VGPRs), so **AGPR predicted ~0** for all
  variants (no `__builtin_amdgcn_mfma` AGPR-accumulate form here). [NEEDS-NODE to confirm via objdump.]
- Scratch: canonical spills ~160 B. Every variant whose total_vgpr ≤ ~200 (rounded) is predicted to
  fit the 256-VGPR/occ-2 budget with **0 scratch** (bm32_nsub8, bm64_nsub4, *_nsub6, staged). nstage1
  keeps the 222-VGPR spill (~160 B) — it relieves LDS, not registers.
- SGPR: ~64 predicted for all (kernel arg block + loop/index scalars); not a binding constraint.

## 5. What WOULD move occupancy (ranked)
1. **Cut LDS** (NSTAGE=1 and/or smaller NSUB): the only lever that changes blocks/CU on the fused
   path while LDS=145 KB.
2. **Cut VGPR below the occ-cliff** (staged-consumer, bm32): removes the 160 B spill (a real cycle
   cost even at occ 1) and *unlocks* higher occupancy *once LDS is also cut*.
3. Grid-level fill for small M (split-K / smaller BM) — separate work, deferred.
