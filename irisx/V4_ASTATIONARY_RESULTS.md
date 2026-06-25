# V4 A-Stationary Fused MoE Expert-GEMM — Results

**Kernel:** `distributed-kernels/fmoe_fused_v4_astationary/kernel.cpp`
**Hardware:** AMD MI355X (gfx950, CDNA4), np=2, rank0=fp8 A source over IRIS, rank1=consumer GEMM.
**Built on V3** (`fmoe_fused_v3`) — V3 untouched. `example.py` is byte-identical to V3 (fair head-to-head harness: same fp8 gather, same dequant, same RMS-rel check, same zero-sentinel).
**Locked config:** `BM=64 BN=64 BK=64 NSUB=8 NSTAGE=2`, 4 producer / 4 consumer warps.

---

## 1. The problem (V3)
V3 grid = `(N/BN, M/BM)`: one block per output tile, each walks all K re-gathering its own A strip
from the **remote** rank over the 128 GB/s interconnect. Every N-block re-gathers the **same** A rows,
so A crosses the interconnect `N/BN` times (= **32x** at N=2048). ~459us of V3's ~478us was this
redundant gather. V3 fused beat baseline only **1.02x**.

## 2. The strategy that worked: A-stationary, wide-N per block
Give each block ONE M-block (`BM` rows) and a WIDE N-range of `NSUB` sub-tiles
(`N_PER_BLOCK = NSUB*BN` columns). For each K-tile the producers gather `A[BM,BK]` **once** into the
shared tile and the consumers **reuse it across all NSUB N-subtiles** (NSUB accumulators in registers,
NSUB local B-subtiles in LDS). B is local HBM (~7.2 TB/s, cheap); only A is the scarce cross-GPU
traffic, and that is what we amortize. A now crosses the interconnect `N/N_PER_BLOCK` times instead of
`N/BN` — an **NSUB-fold reduction** in redundant gather.

### Redundant-gather metric (bytes of A over IRIS per output element)
A is fp8 (1 byte). Minimal possible = `K/N` bytes/elem (each A-row crosses once, feeds N outputs).
Redundancy multiplier = (times each A row crosses):

| Config | N=2048 | N=4096 |
|---|---|---|
| V3 (`N/BN`) | **32x** | 64x |
| **V4 NSUB=8 (`N/N_PER_BLOCK`)** | **4x** | 8x |

=> 8x less cross-GPU A traffic. (True 1x needs `N_PER_BLOCK=N`, i.e. all-N accumulators live across the
K loop = 512 KB regs/block — impossible; NSUB=8 is the register/occupancy-feasible sweet spot.)

## 3. Iteration log (what was tried, measured effect)
All shapes K=7168. "blocks" = grid size; MI355X has 256 CUs so block count drives latency-hiding.

| # | Config | Shape | VGPR | Occ (w/SIMD) | blocks | fused us | base us | speedup | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | V3 baseline (ref) | M256 N2048 | — | 4 | 128 | 478 | 489 | 1.02x | starting point |
| 1 | NSUB=4 | M256 N2048 | 158 | 3 (spill) | 32 | 512 | 488 | **0.95x** | grid starved → regression |
| 2 | NSUB=2 | M1024 N2048 | 126 | 4 | 256 | 1085 | 1222 | 1.13x | occ ok, little reuse |
| 3 | NSUB=4 | M1024 N4096 | 158 | 3 | 256 | 1101 | 1364 | 1.24x | reuse beats occ loss |
| 4 | NSUB=8 | M1024 N4096 | 222 | 2 | 128 | 1137 | 1361 | 1.20x | occ=2 caps it |
| 5 | **NSUB=8** | **M1024 N2048** | 222 | 2 | 64 | **676** | 1232 | **1.82x** | **winner** |
| 6 | BM=32 NSUB=8 | M1024 N2048 | 142 | 3 | 128 | 638 | 1245* | 1.95x* | *baseline also BM=32 (unfair); fair vs BM64 base = 1.91x |
| 7 | BM=128 NSUB=4 | M1024 N2048 | 220 | 2 | 64 | 1181 | 1149 | 0.97x | wide BM → occ=2, loses |

**Key learning:** the win is set by **block count (latency hiding) × gather-reuse**, NOT raw occupancy.
The V3 redundant gather was not pure waste — its many blocks hid the remote-load latency. A-stationary
removes blocks, so it only wins once M is large enough to keep the grid full. NSUB=8 + large M is the
regime where redundant-gather reduction finally converts to wall-clock.

## 4. Final head-to-head (locked BM=64 NSUB=8, baseline = V3-identical two-phase)

| M | N | fused blocks | redundancy | BASELINE us | FUSED us | **speedup** |
|---|---|---|---|---|---|---|
| 128 | 2048 | 8 | 4x | 467 | 550 | 0.85x |
| 128 | 4096 | 16 | 8x | 467 | 548 | 0.85x |
| 256 | 2048 | 16 | 4x | 489 | 547 | 0.89x |
| 256 | 4096 | 32 | 8x | 491 | 547 | 0.90x |
| 512 | 2048 | 32 | 4x | 641 | 562 | 1.14x |
| 512 | 4096 | 64 | 8x | 690 | 660 | 1.05x |
| 1024 | 2048 | 64 | 4x | 1232 | **676** | **1.82x** ← best |
| 1024 | 4096 | 128 | 8x | 1371 | 1140 | 1.20x |

All shapes PASS correctness: **RMS_rel = 0.00331** (matches V3/bf16 ref), `local_A_zero=True`
(zero-sentinel proves the gather is real cross-GPU), `C_zero=False`.

Fused TFLOP/s at the winner: **44.4 TFLOP/s** vs baseline 24.4 (M1024 N2048).

## 5. Honest bottleneck analysis
- **Small M (128/256): comm-latency-bound, loses.** With NSUB=8 there are only 8–32 blocks on 256 CUs
  at occupancy 2 — the grid can't issue enough concurrent remote loads to hide IRIS latency. Fewer A
  bytes don't help when you're latency-, not bandwidth-, limited. (V3's redundant blocks were
  accidentally hiding latency.) For these shapes NSUB=4 (occ 3, 2× the blocks) is less bad but still
  near-parity — these are fundamentally too small to fill the GPU.
- **Large M, N=2048: the win regime, up to 1.82x.** Enough M-blocks keep the grid full, AND the 32x→4x
  redundancy cut removes the dominant cross-GPU cost. Fused us drops 1232→676. This is now
  **compute/occupancy-bound**, not comm-bound: the 4x-reduced gather fully overlaps behind the MFMA, so
  the remaining lever is occupancy (stuck at 2 due to 8 accumulators @ VGPR 222 + 160B scratch spill).
- **N=4096 underperforms N=2048** at fixed NSUB: redundancy is 8x (vs 4x) so more A still crosses, and
  the bigger working set pushes occupancy down — the reuse is half as effective per byte.

## 6. Deferred / next steps
1. **Restore occupancy at NSUB=8.** VGPR=222, occ=2, 160B scratch spill from 8 live fp32 accumulators.
   Splitting the K loop so fewer accumulators are live, or using `BM=32` for the fused kernel *while
   pinning the baseline at BM=64* (fair), already showed occ=3 and ~1.9x — the cleanest next win.
2. **Adaptive NSUB by shape** (runtime dispatch: NSUB=8 for large M/N=2048, NSUB=4 otherwise, NSUB=2 or
   plain V3 for small M) — one kernel can't be optimal across the whole sweep; the loser rows above are
   purely the wrong-NSUB-for-shape penalty.
3. **Cache-on-first-touch (strategy 2):** keep V3's full block count (latency hiding) but have only the
   first block touching each A tile gather it remotely to local HBM; later N-blocks read locally. This
   decouples redundancy reduction from block-count loss — the path to >2x on small M too. Needs a
   cross-block ready-flag in the symmetric heap (the hard part); deferred.

## 7. Files
- `distributed-kernels/fmoe_fused_v4_astationary/kernel.cpp` — the A-stationary fused kernel + V3-identical baseline.
- `distributed-kernels/fmoe_fused_v4_astationary/example.py` — head-to-head driver (identical to V3).
- `distributed-kernels/fmoe_fused_v4_astationary/sweep_v4.log`, `sweep_v4b.log`, `sweep_v4_final.log` — iteration + final data.
- This doc: `HipKittens/V4_ASTATIONARY_RESULTS.md`.
