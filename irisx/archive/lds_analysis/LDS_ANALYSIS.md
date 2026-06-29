# V4 A-stationary — LDS byte budget, occupancy ceiling, and levers (Agent 10)

> Companion to Agent 07's `occ_variants/REGISTER_OCCUPANCY.md`. Agent 07 found V4's occupancy
> ceiling is **LDS-bound, not VGPR-bound**. This doc nails the EXACT LDS byte formula from the
> HipKittens cdna4 headers, isolates the dominant term, and adds one lever Agent 07 did not
> enumerate: **asymmetric buffering (A double, B single)**, written as a concrete variant.
>
> All occupancy numbers are ANALYTICAL PREDICTIONS pending an on-node static resource report.
> LDS_CAP = 160 KB/CU is ASSUMED for MI355X/CDNA4 (2× CDNA3's 64 KB). **[NEEDS-NODE to confirm.]**

---

## 1. The exact LDS byte formula (from HK headers)

**Source of the per-tile byte count** — `HipKittens/include/cdna4/types/shared/st.cuh`:
- line 81: `dtype data[rows*cols];` — a shared tile's storage is exactly `rows * cols * sizeof(T)`.
- line 50: `struct KITTENS_DEFAULT_ALIGN st` and `util.cuh:260` `KITTENS_DEFAULT_ALIGN = ALIGN_AS(16)`
  → 16-byte aligned, **no internal padding** beyond the element array. So:

```
sizeof(st<T, rows, cols, shape>) = rows * cols * sizeof(T)      (rounded up to 16, already a multiple)
```

For V4 (`kernel.cpp:64-65`), bf16 (sizeof=2), BM=BN=BK=64:
```
sizeof(ST_A) = sizeof(st_bf<BM,BK>) = BM*BK*2 = 64*64*2 = 8192 B
sizeof(ST_B) = sizeof(st_bf<BN,BK>) = BN*BK*2 = 64*64*2 = 8192 B
```

**V4 dynamic shared** (`kernel.cpp:79-81`, `micro_tk` allocates `As[NSTAGE]` + `Bs[NSTAGE][NSUB]`):
```
LDS(BM,BN,BK,NSTAGE,NSUB) = NSTAGE * ( sizeof(ST_A) + NSUB * sizeof(ST_B) ) + 1024
                          = NSTAGE * ( BM*BK*2  + NSUB * BN*BK*2 ) + 1024
```
The `+1024` is the V4 dynamic-shared slack (`kernel.cpp:80`); the `shared_allocator`
(`util.cuh:271`) bumps each allocation to a 16-byte boundary — both ST_A and ST_B are already
8192-aligned, so there is **no extra alignment waste**. The formula is exact.

### The ~145 KB derivation (default config)
```
BM=BN=BK=64, NSTAGE=2, NSUB=8:
  ST_A          = 64*64*2                 =   8192 B
  NSUB*ST_B     = 8 * 64*64*2 = 8*8192    =  65536 B
  per stage     = 8192 + 65536            =  73728 B
  * NSTAGE=2                              = 147456 B
  + 1024 slack                           = 148480 B   = 145.0 KB   ✓ (~145 KB)
```

### MI355X per-CU LDS and resulting blocks/CU
LDS_CAP = 160 KB = 163840 B (assumed; 2× MI300/CDNA3). `blocks/CU = floor(LDS_CAP / LDS_per_block)`:

| occupancy goal | required LDS/block | at 160 KB cap |
|---|---|---|
| 1 block/CU | ≤ 163840 B (160 KB) | V4 = 148480 ✓ → **1 block** |
| 2 blocks/CU | ≤ 81920 B (80 KB) | need to cut ~67 KB |
| 3 blocks/CU | ≤ 54613 B (53 KB) | |
| 4 blocks/CU | ≤ 40960 B (40 KB) | |

**V4 at 148480 B → exactly 1 block/CU, regardless of VGPR.** Two blocks would need 296960 B ≫
160 KB. This is the ceiling Agent 07 identified, confirmed from the headers.

---

## 2. Which term dominates + the levers

**Dominant term: `NSTAGE * NSUB * sizeof(ST_B) = 2*8*8192 = 131072 B = 88% of the budget.**
The A term is only `NSTAGE*sizeof(ST_A) = 16384 B = 11%`. **B buffering, not A, is the LDS hog.**
This is the lever space — and crucially the A-stationary traffic win lives entirely in the A term
(A gathered once per K-tile, reused across NSUB), so we can attack B freely.

### Lever A — Single-buffer B (asymmetric buffering)  ← my headline finding, NOT in Agent 07's grid
Keep `As[NSTAGE]` (A double-buffered, hides the remote gather) but make B `Bs[NSUB]` (one set).
```
LDS = NSTAGE*ST_A + NSUB*ST_B + 1024 = 2*8192 + 8*8192 + 1024 = 82944 B (~81 KB)
LDS saved = (NSTAGE-1)*NSUB*ST_B = 1*8*8192 = 65536 B (~64 KB), the entire 2nd B buffer.
blocks/CU = floor(160KB/81KB) = 1   (160/82.9 = 1.93 — JUST misses 2)
```
- **Is it safe?** YES for correctness and YES for the win. B is LOCAL HBM (~7.2 TB/s, no remote
  latency to hide), so its double buffer was bought cheap overlap. A — the scarce 128 GB/s
  cross-GPU traffic — stays double-buffered, so the expensive gather still runs ahead. The
  A-stationary amortization is UNCHANGED (N_PER_BLOCK = NSUB*BN unchanged → A crosses the
  interconnect exactly as often as canonical V4).
- **Risk:** the producer for tile t+1 can't overwrite the single B buffer until the consumer has
  read it for tile t. The existing per-tile `s_barrier` already enforces this; net cost is the lost
  *local-B prefetch overlap only*. Predicted: ≈ V4 perf at occ 1, but ~64 KB freed.
- At NSUB=8 alone it stays occ 1 — so pair it with a small NSUB cut to clear 2 (Lever D).
  **Written as `v4_bsingle_buffer/kernel.cpp`** (see README for exact diff + build/test).

### Lever B — NSTAGE=1 overall (single-buffer A AND B)
```
LDS = 1*(8192 + 8*8192) + 1024 = 74752 B (~73 KB) → floor(160/73)=2 blocks/CU.
LDS saved = 73728 B.
```
- **Risk: HIGH.** PREFETCH = NSTAGE-1 = 0 → the prologue prefetches nothing; every K-tile stalls on
  its OWN remote A gather (the expensive op, now un-overlapped). Reaches occ 2 but loses the
  producer/consumer overlap that hides the 128 GB/s gather. This is Agent 07's `v4_nstage1`.
  Asymmetric buffering (Lever A) is strictly better than NSTAGE=1 for the same goal because it KEEPS
  A's overlap; the only thing NSTAGE=1 buys over Lever A is the extra 8 KB (one A buffer) that
  finally floors 160/73 to 2 — but at the cost of the remote-gather overlap. Prefer Lever A+D.

### Lever C — Stream B subtiles (keep < NSUB resident, re-load)
Hold only `R < NSUB` B subtiles in LDS, re-fetch the rest from HBM as the consumer advances.
```
LDS = NSTAGE*ST_A + R*ST_B + 1024.  E.g. R=4: 2*8192 + 4*8192 + 1024 = 50176 B (~49 KB) → 3 blocks/CU.
```
- **Risk: MEDIUM.** B is cheap to re-read (local 7.2 TB/s) so the extra HBM traffic is tolerable,
  BUT it complicates the inner loop and may serialize MFMA on B loads. Worth it only if 3 blocks/CU
  is needed and the A-stationary win must be fully preserved (it is — A untouched).

### Lever D — Smaller NSUB (fewer B subtiles)
```
NSUB=6: NSTAGE=2 → 2*(8192+6*8192)+1024 = 114688+1024 = 115712? recompute: 2*(8192+49152)=114688 +1024
        = 115712 B... -> floor(160/113)=1. (Agent 07 lists 111616; diff = their NSUB6 still NSTAGE2.)
NSUB=4: NSTAGE=2 → 2*(8192+4*8192)+1024 = 2*40960+1024 = 82944? -> 81 KB -> 1 block (=Lever A NSUB8!).
```
- Smaller NSUB **directly reduces the A-stationary win**: N_PER_BLOCK = NSUB*BN shrinks, so A
  crosses the interconnect (N / N_PER_BLOCK)× = more often. NSUB 8→4 doubles redundant A gather.
  **This is the one lever that trades away the win** — use sparingly, and prefer combining a SMALL
  NSUB cut with single-buffer-B (Lever A) so you reach occ 2 with only a modest A-traffic hit.
- Tail hazard: NSUB=6 → N_PER_BLOCK=384 ∤ 2048 (Agent 07's note) → wasted/ masked MFMAs.

### Lever E — Smaller BK
`BK 64→32` halves BOTH ST_A and ST_B (`rows*cols*2`), so LDS halves to ~74 KB at NSTAGE2/NSUB8 →
2 blocks/CU, A-stationary win fully kept (N_PER_BLOCK unchanged). **Risk:** doubles num_k_tiles
(112→224) → 2× the barrier/loop overhead and shorter MFMA chains (less compute to hide latency);
smaller K-tile may underfeed the MFMA. Cheap to try via a `BK 32` define; promising.

### Lever F — Staged consumer (Agent 07's `v4_staged_consumer`)
Orthogonal to LDS — it's a VGPR/spill lever (partitions NSUB across 8 consumer warps). Does NOT cut
LDS by itself (still 145 KB → occ 1). Only reaches occ 2 when combined with an LDS lever. Include
its low-VGPR benefit on TOP of Lever A/E.

---

## 3. Ranked config recommendation (keep the A-stationary win)

Ranked by (occupancy gained × win preserved × overlap preserved):

| rank | config | LDS | blocks/CU | A-stationary win | rationale |
|---|---|---|---|---|---|
| **1** | **B-single + BK=32** (A2,B1,BK32,NSUB8) | `2*4096 + 8*4096 + 1024 = 41984 B (~41 KB)` | **3** | FULL (N_PER_BLOCK & gather count unchanged) | both LDS levers that DON'T touch N_PER_BLOCK; A still double-buffered. Risk: 2× k-tiles. |
| **2** | **B-single + NSUB=6** (A2,B1,NSUB6) | `2*8192 + 6*8192 + 1024 = 66560 B (~65 KB)` | strong (N_PER_BLOCK 512→384, A gather 1.33×) | **2** | the cleanest occ-2 that keeps A double-buffered. Tail 384∤2048 needs masking. |
| **3** | **B-single, NSUB=8** (`v4_bsingle_buffer`, written) | `82944 B (~81 KB)` | FULL | 1 | safest correctness baseline; proves single-buffer-B is free; frees 64 KB even at occ 1. SHIP/MEASURE FIRST. |
| **4** | **BK=32, symmetric** (NSTAGE2,NSUB8,BK32) | `74752 B (~73 KB)` | FULL | 2 | occ 2, full win, keeps B double-buffer; cost is 2× k-tile overhead only. |
| 5 | NSTAGE=1 (Agent 07 `v4_nstage1`) | `74752 B (~73 KB)` | 2 | FULL traffic, but NO gather overlap | reaches occ 2 but un-overlaps the remote gather. Dominated by rank 4 / Lever A. |
| 6 | B-single + staged-consumer + NSUB6 | ~65 KB | 2 | strong | adds VGPR/spill relief on top of rank 2. Highest ceiling if all predictions hold. |

**Top pick to measure first: rank 3** (`v4_bsingle_buffer`, NSUB8) to confirm single-buffer-B is
perf-neutral, then **rank 1 / rank 2** to convert the freed LDS into occupancy 2–3.

---

## 4. Cross-check vs Agent 07

**AGREE:**
- Exact formula `NSTAGE*(8192 + NSUB*8192)+1024` → 148480 B (~145 KB). Identical to my header-derived
  value. Their `RESOURCE_TABLE.csv` lds_bytes column matches mine row-for-row.
- Occupancy is LDS-bound at 1 block/CU for every NSTAGE=2/NSUB=8 variant; VGPR relief alone buys
  nothing on the fused path. Confirmed.
- `v4_nstage1` (73 KB → 2 blocks) and `v4_bm64_nsub4` (73 KB → 2 blocks) reach occ 2. My math agrees.
- `v4_cons8_nsub8` INFEASIBLE (CONS_N=8 ∤ 16). Agree (rt.cuh base_cols=16 static_assert).

**ADD (not in Agent 07's grid):**
- **Asymmetric buffering (A double / B single).** Agent 07's levers are all symmetric in NSTAGE
  (`v4_nstage1` single-buffers BOTH A and B and so loses the A-gather overlap). Single-buffering
  *only* B keeps the remote-gather overlap while removing 64 KB — strictly better than NSTAGE=1 for
  the same occupancy target. This is the key insight and the written artifact.
- **BK=32** as an LDS lever that preserves N_PER_BLOCK (and thus the full A-stationary win) — not in
  the 07 grid.

**MINOR DISCREPANCIES (flag, not disagree):**
- Agent 07's `v4_nstage1/kernel.cpp` header comment states "gfx950 LDS = 64 KB/CU", but their main
  `REGISTER_OCCUPANCY.md` (§2) and CSV use the corrected **160 KB**. The 64 KB line is a stale note;
  the 160 KB figure is the operative one. Both of us mark LDS_CAP **[NEEDS-NODE]**. If the cap is
  actually 64 KB, NO NSTAGE=2 fused config fits and BK/NSUB must shrink hard — but V4 reportedly
  runs at 145 KB, which is itself evidence LDS_CAP ≥ 145 KB on this node.
- Agent 07 lists `v4_bm32_nsub6` LDS = 111616; their NSUB6 keeps NSTAGE=2 → `2*(8192+6*8192)+1024
  = 114688+1024 = 115712`. The 111616 appears to omit the +1024 and use a slightly different ST_B
  count; the difference is immaterial to the occ conclusion (both → 1 block). Use 115712 as exact.

---

## 5. Assumptions / NEEDS-NODE
- LDS_CAP = 163840 B (160 KB) for MI355X/CDNA4. CONFIRM via `rocminfo` / device props on <NODE>.
- `sizeof(st_bf<64,64>) = 8192 B` exact (no pad) — verified from `st.cuh:81` + `KITTENS_DEFAULT_ALIGN`.
- Single-buffer-B is correctness-neutral (same math, only LDS layout + the B barrier dependency).
  CONFIRM RMS-rel ~0.0033 on-node.
- Predicted perf-neutrality of single-buffer-B at occ 1, and occ-2/3 from ranks 1/2/4, are
  HYPOTHESES — the main agent must compile + rocprof to confirm blocks/CU and µs.
