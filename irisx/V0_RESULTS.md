# V0 MoE dispatch+pack — results (8x MI355X, gfx950, 2026-06-24)

## Build (verified)
```
source /usr/share/Modules/init/bash && module load mpi/openmpi-x86_64
cd <repo>/irisx   # node path: see ../NODE_ACCESS.local.md
cmake -B build -DIRIS_BUILD_BENCHMARKS=ON -DIRIS_BUILD_TESTS=ON -DIRIS_HIP_ARCHITECTURES=gfx950
cmake --build build --parallel 8 --target moe_dispatch_pack test_moe_dispatch_pack
```

## Run (note: node IB is unusable -> force shared-mem MPI transport)
The mlx5 HCA cannot register memory (R1 server holds ~96% VRAM), so the default
OFI/openib transports fail at MPI_Init. Use ob1 + self,vader (intra-node IPC needs no IB):
```
mpirun --mca pml ob1 --mca btl self,vader -np 8 ./build/tests/test_moe_dispatch_pack
mpirun --mca pml ob1 --mca btl self,vader -np 8 ./build/benchmarks/moe_dispatch_pack     # add arg "1" for in-kernel profile
```

## Correctness
test_moe_dispatch_pack: ALL TESTS PASSED on 8 ranks (9 assertions/rank).
  [V0b] checked 496 assignments, 0 bucket errors, 0 bad route_slot
  [V0a] 0 expert-bucket errors
Both packed outputs match a CPU reference rebuilt from MPI-gathered topk_ids (multiset of
per-(src,expert) token signatures); route_slot[token][topk] verified in-range.

## Bandwidth (H=7168 bf16, topk=8, 256 experts, T_local=256, world=8; 2048 assignments/rank = 29.36 MB/rank/iter)
device multiProcessorCount = 256 CUs -> grid = 256 blocks (queried, not hardcoded)

| variant | ms/iter | GB/s  | us/instance |
|---------|--------:|------:|------------:|
| V0a (remote-atomic slot claim)    | 0.1008 | 291 | 100.8 |
| V0b (precomputed-offset, no atomics) | 0.0978 | 300 | 97.8 |

In-kernel timestamp profile (PROFILE=1): remote-store cycles (1.15e9) >> slot-claim cycles
(0.22e9) -> the path is bandwidth-bound, not atomic-bound at this scale. V0a's remote
fetch_add still costs ~8% vs V0b (273 vs 297 GB/s in the profile run), so V0b is the perf path.

## vs references
- all_put ceiling: NOT directly comparable / could not capture a stable number. all_put
  reallocates a fresh 1 GB IRIS heap each of 10 experiments and OOMs (~experiment 7) on this
  VRAM-starved node; it also moves only 4 KB/rank (latency-bound), not a 14 KB/token payload.
  The achieved 300 GB/s on a 29 MB transfer is the meaningful XGMI figure here.
- production EpDispatch + opus_moe_sorting x2 ~= 40 us/instance. Our 98 us/instance moves the
  FULL bf16 payload (no quant) and is intra-node IPC microbench (no graph capture / no overlap).
  V1 (FP8) will ~halve bytes; the real apples-to-apples needs the production data sizes + graph.

## Prior-art requirements applied
- route_slot[token][topk] first-class output (combine needs it). DONE.
- grid = device_props.multiProcessorCount (256), grid-strided kernels. DONE.
- bulk cross-rank completion signal (release per-source flag to all ranks + acquire-spin), not
  a heavy host barrier. DONE (completion_signal kernel).
- vectorized 16B stores (bf16x8). DONE.
- compile-time in-kernel timestamp profiling mode. DONE.

## Files
- benchmarks/moe_dispatch_pack.hip  (V0a + V0b + completion_signal + profile)
- tests/test_moe_dispatch_pack.hip  (Catch2, 8-rank)
- benchmarks/CMakeLists.txt, tests/CMakeLists.txt  (already registered)

## Recommended next step
V1: fold per-128-group FP8 e4m3 quant into V0b's store path (write fmoe_bf16_blockscaleFp8
layout: fp8 + 56 scales/token), ~halving XGMI bytes. Keep route_slot for the eventual V3 combine.
