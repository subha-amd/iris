# INTERLEAVE_4WAVE — symmetric 4-wave latency kernel (small-M MoE expert-GEMM)

Candidate dir: `irisx/sched_4wave/`. Generalizes V4 (`irisx/v4_astationary_kernel`) for the
**small-M latency path** (M <= 256), where V4's producer/consumer split loses (0.85-0.90x vs
baseline). Same fp8-e4m3 / per-128-scale / RMS-rel + zero-sentinel ABI as V4.

## 1. Why V4 loses at small M

V4 has **4 permanent producer waves + 4 permanent consumer waves** that hand off across an
`s_barrier` every K-tile:
- producers gather+dequant the remote fp8 A-tile over IRIS, consumers MFMA;
- at large M there is enough MFMA work to hide the gather, so V4 wins (M>=512);
- at small M (M<=256) the MFMA per block is tiny, so (a) it cannot cover the gather latency, and
  (b) half the waves are idle during each phase while the per-K-tile barrier serializes the two
  halves. The hand-off overhead dominates -> V4 = 0.85-0.90x.

## 2. The symmetric fix — every wave does BOTH

There is **no producer/consumer split**. All `NUM_WORKERS=4` waves (one per SIMD on the CU) are
identical. Each wave:
- owns its OWN strip of `CONS_M = BM/4` output rows;
- issues BOTH the IRIS remote fp8 **gather+dequant** of its rows AND the **MFMA** on its rows;
- carries its own in-flight cross-GPU load.

Because each wave both loads and computes, the gather latency of wave *w* on K-tile *(t+PREFETCH)*
is hidden by wave *w*'s OWN MFMA on K-tile *t* — **within the same wave**, not across a wasted
producer/consumer `s_barrier`. This mirrors HipKittens' `FP8_4wave/4_wave.cu`
(`do_interleaved_cluster` fine-grains `load_one` against `mma_one` under `sched_barrier`), adapted
to the IRIS remote-gather MoE setting.

## 3. Per-wave instruction pipeline (steady state, NOT all-load-then-compute)

Every wave runs this identical software pipeline (`kernel.cpp:micro_tk`):

```
PROLOGUE (per wave):
  gather_dequant_A_strip(As[0], tile=0, my CONS_M rows)     # remote fp8 -> dequant -> LDS
  for sub in 0..NSUB-1: load_B_subtile(Bs[0][sub], tile=0)  # local HBM B -> LDS
  s_waitcnt(0); __syncthreads()                             # sibling waves' A strips visible

STEADY LOOP, tile t = 0..num_tiles-1:
  cur   = t % NSTAGE
  fetch = t + PREFETCH
  # (1) ISSUE next K-tile's loads into the OTHER LDS slot — non-blocking VMEM, the latency we hide
  if fetch < num_tiles:
     gather_dequant_A_strip(As[fetch%NSTAGE], fetch, my CONS_M rows)   # in-flight remote gather
     for sub: load_B_subtile(Bs[fetch%NSTAGE][sub], fetch)            # in-flight local B
  # (2) Load THIS wave's A fragment (its CONS_M rows) ONCE; reuse across NSUB N-subtiles
  load(a_frag, subtile<CONS_M,BK>(As[cur], {warp_id,0})); s_waitcnt lgkmcnt(0)
  # (3) Interleave per-subtile B-fragment load with MFMA under s_setprio
  for sub in 0..NSUB-1:
     load(b_frag, Bs[cur][sub]); s_waitcnt lgkmcnt(0)
     s_setprio(1); mma_ABt(C_accum[sub], a_frag, b_frag, C_accum[sub]); s_setprio(0)
     sched_barrier(0)                       # pin the load/mma interleave order
  s_barrier()                               # cheap: all waves both produced+consumed this iter

EPILOGUE (per wave): store NSUB accumulator tiles (CONS_M rows each)
```

Key point: step (1)'s remote gather for tile *(t+PREFETCH)* is in flight (VMEM) across the whole of
step (3)'s MFMA for tile *t*. The `s_setprio(1)` around `mma_ABt` keeps the MFMA hot while the
`load`s and the background VMEM gather drain. The trailing `s_barrier()` is NOT a producer/consumer
hand-off — every wave produced (gathered its strip) and consumed (MFMA'd its strip) this iteration;
it only orders sibling waves' LDS writes to the next-tile A buffer before the next read.

## 4. How register pressure stays low (=> 4 waves/SIMD occupancy)

V4 holds **8 live fp32 accumulator tiles** per consumer wave (NSUB=8 of `rt_fl<BM,CONS_N>`), driving
~222 VGPR and only ~2 waves/SIMD with ~160B scratch spill.

This kernel:
- **splits M across waves**: each wave's accumulator is `rt_fl<CONS_M=BM/4, BN>`, i.e. 1/4 the rows;
- keeps only **NSUB (1-2) live accumulators** per wave (default NSUB=2), not 8;
- with `BM=32, NSUB=2` -> `CONS_M=8`, so each wave holds 2x `rt_fl<8,64>` fp32 tiles — a tiny
  accumulator footprint (~16 fp32 regs/lane worth of C, plus one `a_frag` + one transient `b_frag`).

That low VGPR footprint targets **4 waves/SIMD** resident occupancy. Latency hiding then comes from
TWO sources stacked: (a) the per-wave software pipeline (gather of t+PREFETCH behind MFMA of t), and
(b) occupancy (4 resident waves/SIMD swap on stalls). A is still gathered **once per K-tile** into
LDS and reused across NSUB N-subtiles, so the cross-GPU traffic win of V4 is preserved: A crosses
the interconnect `N/(NSUB*BN)` times, not `N/BN` times.

## 5. Tunables (sweep these)

| knob   | default | sweep            | effect                                                        |
|--------|---------|------------------|---------------------------------------------------------------|
| BM     | 32      | {16, 32}         | rows/block; CONS_M=BM/4 -> per-wave accumulator height        |
| BN     | 64      | {32, 64}         | cols/N-subtile; accumulator width                             |
| BK     | 64      | {32, 64}         | K-tile depth; gather granularity vs MFMA chunk                |
| NSUB   | 2       | {1, 2, 4}        | N-subtiles/block; A-reuse factor & live accumulators/wave     |
| NSTAGE | 2       | 2                | LDS double-buffer depth (PREFETCH = NSTAGE-1)                 |

Priority M values: **{8, 16, 32, 64, 128, 256}** (the regime V4 loses in).

## 6. Correctness / ABI (unchanged from V4)

- fp8 e4m3 OCP (`__HIP_E4M3`, NOT fnuz), per-128-group fp32 scales, B/C bf16.
- Remote gather via `iris_device_view::load(..., src_rank)`; non-source rank holds zeros
  (zero-sentinel: a correct result proves the data was pulled cross-GPU).
- `store(g.c, accum, {0,0,row_tile,col_tile})` indexes in accumulator-tile units:
  `row_tile = out_row0/CONS_M`, `col_tile = out_col0/BN`.
- Pass criterion: RMS-rel < 0.10 (expect ~0.003), `local_A_zero=True`, `C_zero=False`.
