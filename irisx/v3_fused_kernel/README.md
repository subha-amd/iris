# V3 — Full fused MoE expert GEMM (remote fp8 gather + dequant + MFMA in one kernel)

The capstone: merges **V1** (per-128-group FP8 e4m3 quant/dequant) with **V2.1** (remote-gather
producer/consumer GEMM) into ONE kernel, and measures it head-to-head against the unfused
two-phase design (the production-style `dispatch+quant` then separate `fmoe` GEMM).

Mirrored here for tracking; built inside a HipKittens checkout's `distributed-kernels/` (depends
on HK + IRIS), run inside the ATOM container. Place under
`HipKittens/distributed-kernels/fmoe_fused_v3/`.

## What it does
On the consumer rank: 4 producer warps pull fp8 e4m3 activation tiles (vectorized 16 bytes/thread
via uint4) + per-128 fp32 scales DIRECTLY from the remote rank's IRIS heap, dequantize fp8→bf16 in
the producer, write the swizzled shared tile; 4 consumer warps MFMA the previous tile. Cross-GPU
gather + dequant + matmul overlap. B/C local. AMD ping-pong scheduling (no idle-producer wave spec).

The **baseline** in the same file (`micro_tk_baseline`) is identical code with the overlap removed
(gather-all, barrier, then compute) — a fair apples-to-apples two-phase comparison.

## Verified result (8×MI355X gfx950, np=2 — independently re-run 3×, stable)
- Correctness: fused == baseline, RMS-rel 0.00331 vs bf16 reference; zero-sentinel proves remote gather.
- Head-to-head (M=256,K=7168,N=2048): baseline ~488µs, fused ~478µs → **1.02× (fused wins), stable to <0.3µs across runs.** Wins on every shape tried (1.01–1.02×).
- **Honest bottleneck: gather-bound.** Each A-tile is re-fetched N/BN=32× → ~59MB fp8 over the
  128 GB/s link ≈ 459µs, matching measured ~478µs. MFMA is nearly free here (doubling N barely
  moves wall-time), so there's little compute to hide the gather under — the fusion *ceiling* at
  these shapes is only a few %, and the kernel realizes essentially all of it. Same family as V1's
  quant-bound finding: the limiter is data movement, and overlap can only hide so much when compute
  is tiny relative to comm.

## Build / run
```
docker exec r1_c4 bash -lc 'cd <HK_ROOT>/distributed-kernels && cmake -B build -DDK_BUILD=fmoe_fused_v3 && cmake --build build -j16 --target fmoe_fused_v3'
docker exec r1_c4 bash -lc 'cd <HK_ROOT>/distributed-kernels/fmoe_fused_v3 && source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64; M=256 K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 example.py'
```

## Next step (the real win is here)
The small margin is because A is re-gathered 32× (once per N-block), so the kernel pays 32× the
necessary cross-GPU traffic. **Cache/reuse each A-tile across all N-blocks** (A-stationary) so each
token tile crosses the interconnect ONCE — that cuts gather traffic ~32×, making compute the
limiter and giving the overlap real headroom (tens of %). Then: bulk async IRIS tile-get to
saturate the 128 GB/s link, real multi-rank expert routing (per-tile src_rank), full FFN
(gate/up→SiLU→down), 8-rank scale.
