# PINGPONG_8WAVE — true HipKittens 8-wave role-swap schedule for remote-FP8 MoE GEMM

This candidate transplants the canonical HipKittens 8-wave ping-pong schedule
(`<HK_ROOT>/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu`) onto the IRIS remote-FP8 MoE
expert-GEMM problem that V4 (`irisx/v4_astationary_kernel/`) solves with a permanent
producer/consumer split.

---

## 1. Role-SWAP vs V4's role-SPLIT (the core difference)

| | V4 (a-stationary) | sched_8wave (this) |
|---|---|---|
| warp partition | `warp_id < 4` = **producer forever**, `>= 4` = **consumer forever** | `warp_m = warp_id/WARPS_COL` in {0,1} = two **symmetric** wavegroups |
| who issues `G::load` | only the 4 producers | **every** wave (both wavegroups feed) |
| who issues `mma_ABt` | only the 4 consumers | **every** wave (both wavegroups compute) |
| overlap mechanism | producer fills a double-buffer the consumer drains | the two wavegroups **swap** feed/compute roles every `s_barrier()` |
| half the waves idle on MMA? | yes (producers never MMA) | **no** — all 8 waves do MMAs |
| half the waves idle on load? | yes (consumers never load) | **no** — all 8 waves do loads |

V4 dedicates 4 of 8 waves to memory and 4 to math permanently, so peak MFMA throughput is
capped at half the waves. The HK ping-pong keeps **all 8 waves doing math**, and hides the
feed latency by having whichever wavegroup is *not* currently in its MMA window do the next
feed. The two windows are interleaved by a half-phase, so the matrix units stay busy.

---

## 2. The two wavegroups and the ping-pong phase offset

```
warp_id :  0   1   2   3 | 4   5   6   7
warp_m  :  0   0   0   0 | 1   1   1   1     <- warp_m = warp_id / WARPS_COL   (WARPS_COL=4)
warp_n  :  0   1   2   3 | 0   1   2   3     <- warp_n = warp_id % WARPS_COL
group   : <--- WG0 ---->  <--- WG1 ---->
```

- WG0 (warp_m==0) owns output rows `[block_row, block_row+HALF_BM)`.
- WG1 (warp_m==1) owns output rows `[block_row+HALF_BM, block_row+BM)`.

The ping-pong is **seeded** in the prologue by a conditional barrier that only WG1 executes:

```c
if (warp_m == 1) __builtin_amdgcn_s_barrier();   // only WG1
__builtin_amdgcn_s_barrier();                     // both
```

WG1 hits `s_barrier()` twice, WG0 once, before entering the K loop. Because `s_barrier()` is a
workgroup-wide rendezvous, this puts WG0 and WG1 **a half-phase out of step**: when WG0 arrives
at the loop's first barrier, WG1 is already one barrier ahead, so the two groups alternate which
side of each barrier-pair they are on. That alternation is the ping-pong.

It is **rebalanced** in the epilogue by the mirror-image conditional barrier (only WG0), so both
wavegroups have executed the SAME total number of `s_barrier()`s before the stores — required
because `s_barrier()` participation must be balanced across the workgroup or the hardware barrier
counter desyncs.

```c
if (warp_m == 0) __builtin_amdgcn_s_barrier();   // only WG0 (mirror of the prologue seed)
```

---

## 3. K-loop barrier lattice (the maintained ping-pong)

Per K-tile, the A row-half is loaded into registers **once** (a-stationary reuse), then for each
of the NSUB N-subtiles the kernel runs a **feed -> barrier -> compute -> barrier** couplet:

```
for tile in 0..num_tiles:
    load a_frag  <- As[cur][warp_m]          # this WG's own A row-half, reused over NSUB
    s_waitcnt lgkmcnt(0)
    for sub in 0..NSUB:
        # ---- FEED phase ----
        if fetch < num_tiles:
            if sub==0: gather_dequant_A_half -> As[slot][warp_m]   # remote IRIS FP8 + dequant
            G::load   -> Bs[slot][sub]                              # local-HBM bf16
        s_barrier()                          # (A) hand LDS/compute to the OTHER wavegroup
        # ---- COMPUTE phase ----
        load b_frag <- Bs[cur][sub] subtile {warp_n}
        s_waitcnt lgkmcnt(0)
        setprio(1); mma_ABt(C_accum[sub], a_frag, b_frag, C_accum[sub]); setprio(0)
        sched_barrier(0)
        s_barrier()                          # (B) hand back to the OTHER wavegroup
```

Because of the half-phase seed, at barrier (A) one wavegroup is *finishing a feed* while the
other is *about to start its MMA*; at barrier (B) the situation is swapped. So at any moment one
wavegroup occupies the matrix units (MMA window) while the other occupies the
memory/LDS path (feed window). `setprio(1)` around the MMA tells the scheduler to favor the
compute wavegroup's matrix instructions over the feeding wavegroup's vector/memory ops, which is
exactly the HK source's trick.

ASCII timeline (one K-tile, one subtile; F=feed, M=mma, .=stall-free wait):

```
        |---- barrier(A) ----|---- barrier(B) ----|
WG0 :   [   M M M M  ]        [   F F F F  ]
WG1 :   [   F F F F  ]        [   M M M M  ]
            ^compute              ^compute
            (matrix units never idle: one WG always computing)
```

---

## 4. Output ownership — why there is NO partial-K reduction

This is the property that makes the role-swap legal for a GEMM.

Naive worry: "if two wavegroups split the work, don't they each compute a partial sum over part
of K that must then be added?" That would be a **split-K** scheme and would need a reduction
(atomic add or a second pass). **We do not do split-K.**

Instead we **split the output ROWS**:

```
A[block_row .. block_row+BM, :]   (BM rows, full K)
        |
        +-- WG0 gathers rows [block_row,          block_row+HALF_BM)   -> As[*][0]
        +-- WG1 gathers rows [block_row+HALF_BM,  block_row+BM)        -> As[*][1]

Both wavegroups walk the FULL K loop (tile = 0 .. K/BK).
WG0's accumulator C_accum[sub] = sum_k  A[WG0 rows, k] . B[n, k]   over ALL k
WG1's accumulator C_accum[sub] = sum_k  A[WG1 rows, k] . B[n, k]   over ALL k
```

Each accumulator is therefore a **complete** dot product over the entire K dimension for its own
disjoint set of rows. The stores write disjoint row ranges:

```c
store(g.c, C_accum[sub], {0,0, row_base/HALF_BM, out_col0/CONS_N});
// row_base = block_row + warp_m*HALF_BM  -> WG0 and WG1 target DIFFERENT row tiles
```

No element of C is written by both wavegroups => no cross-wavegroup add, no atomics, no second
reduction kernel. The canonical HK source does the same thing with its four accumulators cA/cB/
cC/cD stored to `block_row*WARPS_ROW*2 + warp_m` vs `... + WARPS_ROW + warp_m` (disjoint rows).

This is a different partition than V4: V4's 8 accumulators all cover the **same BM rows** but
**different N subtiles**; here the NSUB accumulators cover the same N-subtiles but each wavegroup
owns a **disjoint row-half**. Both avoid partial-K reduction; they differ in which axis is split.

---

## 5. Remote-MoE adaptation of the HK feed/compute primitives

| HK FP8_8wave primitive | sched_8wave replacement |
|---|---|
| `G::load(As, A, ...)` from local HBM fp8 | `gather_dequant_A_half<16>()`: IRIS `ctx.load(uint4, src_rank)` remote FP8 gather + per-128-group fp32 dequant into swizzled shared bf16 (reuses V4 math), each WG gathers its own HALF_BM row-half |
| `G::load(Bs, B, ...)` fp8 weights | `G::load<2,false>(Bs, g.b, ...)` local-HBM **bf16** weights (cheap ~7 TB/s); all 8 waves cooperate |
| `load_st_to_rt` + `mma_ABt` fp8xfp8 | `load` (bf16) + `mma_ABt` bf16xbf16; A already dequantized in the gather |
| double-buffer `As[2][2]`,`Bs[2][2]` | `As[NSTAGE][WARPS_ROW]` (one A tile per WG per stage) + `Bs[NSTAGE][NSUB]` |

A is the scarce cross-GPU traffic (128 GB/s IRIS link); B is local. The a-stationary reuse
(load A row-half once per K-tile, reuse across NSUB N-subtiles) is inherited from V4 so the
remote-gather amortization V4 proved (8x less redundant A traffic at NSUB=8) is preserved while
adding the all-waves-compute ping-pong on top.

---

## 6. Parameter space

| param | values | meaning |
|---|---|---|
| BM | 32, 64 | output rows / block (split into 2 disjoint HALF_BM row-halves) |
| BN | 32, 64 | output cols / N-subtile |
| BK | 32, 64 | K-step |
| NSUB | 2, 4, 8 | N-subtiles / block (a-stationary A reuse factor) |
| NSTAGE | 2, 3 | shared double/triple buffer depth |
| WARPS_COL | 4 (fixed) | warps / wavegroup => 8 total warps, 2 wavegroups |

Default in `kernel.cpp`: BM=BN=BK=64, NSUB=4, NSTAGE=2.

Constraints: `BM % WARPS_ROW == 0` (HALF_BM integral); `BN % WARPS_COL == 0` (CONS_N integral);
`K % BK == 0`. With BM=64 => HALF_BM=32; with BN=64,WARPS_COL=4 => CONS_N=16 (one 16x16 MFMA col
per wave per subtile).

---

## 7. Risks specific to the ping-pong (see AGENT_REPORT.md for the full risk list)

1. **Barrier-count balance.** Every `s_barrier()` in the K loop is hit by ALL 8 waves; only the
   two conditional prologue/epilogue barriers are per-wavegroup, and they are mirror images
   (one WG1, one WG0) so totals match. If a config makes the loop body conditionally skip a
   barrier (it does not, currently) the hardware barrier counter would desync and hang.
2. **A-half gather imbalance.** Each WG gathers HALF_BM rows with WARPS_COL warps; at BM=32,
   HALF_BM=16, the gather may underutilize threads. Covered by the BM=64 default.
3. **setprio fairness.** `setprio(1)` favors the computing WG; under triple-buffering (NSTAGE=3)
   the feeding WG has more slack, which should help — verify occupancy on node.
