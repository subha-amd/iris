# V4 — A-stationary fused MoE expert GEMM (the significant-speedup win)

Builds on V3 (full fused remote-gather + dequant + MFMA). V3 won only 1.02x because it was
**gather-bound from redundant cross-GPU traffic**: with grid `(N/BN, M/BM)`, every one of the
`N/BN = 32` N-blocks re-gathered the SAME A rows from the remote GPU — so each token tile crossed
the 128 GB/s interconnect ~32x more than necessary.

**V4 fix — A-stationary, wide-N per block:** each block owns one M-block and `NSUB` N-subtiles
(`N_PER_BLOCK = NSUB*BN`). Producers gather `A[BM,BK]` once per K-tile; consumers reuse it across
all NSUB subtiles (NSUB register accumulators + NSUB local B tiles). At N=2048, NSUB=8 cuts A's
cross-GPU crossings from 32x → 4x (8x less redundant gather).

Mirrored here for tracking; built/run inside a HipKittens checkout + ATOM container. Place under
`HipKittens/distributed-kernels/fmoe_fused_v4_astationary/`.

## Verified result (8×MI355X gfx950, np=2 — independently re-run 3×, stable)
- Correctness: fused == baseline, RMS-rel **0.00331**; zero-sentinel proves remote gather is real.
- **Winning shape M=1024, N=2048, K=7168: fused ~677µs vs baseline ~1227µs = 1.80–1.83x** (stable
  across 3 runs; 24→45 TFLOP/s).
- Wins 1.05–1.83x for M≥512; **loses (0.85–0.90x) for M≤256** — small M starves the 256-CU grid
  (A-stationary removes the redundant blocks that were accidentally hiding IRIS load latency in V3).

## Honest bottleneck analysis
The win is governed by **block count (latency hiding) × gather reuse**, not raw occupancy. At the
winner it's compute/occupancy-bound (occupancy stuck at 2 from 8 fp32 accumulators; VGPR 222 +
scratch spill). Small-M shapes remain comm-latency-bound. So the speedup is real and large in the
batched-decode regime (many tokens/expert), and the comm-latency regime needs a different tactic.

## Build / run
```
docker exec r1_c4 bash -lc 'cd <HK_ROOT>/distributed-kernels && cmake -B build -DDK_BUILD=fmoe_fused_v4_astationary && cmake --build build -j16 --target fmoe_fused_v4_astationary'
docker exec r1_c4 bash -lc 'cd <HK_ROOT>/distributed-kernels/fmoe_fused_v4_astationary && source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64; M=1024 K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 example.py'
```

## Next steps
Restore occupancy at NSUB=8 (fewer live accumulators); adaptive NSUB by shape; cache-on-first-touch
to also win at small M; then real multi-rank expert routing (per-tile src_rank), full FFN
(gate/up→SiLU→down), fp8 weights, 8-rank scale.
