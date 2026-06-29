# b1_overlap — P4 host-scheduled chunk overlap

This is the next overlap attempt after P1/P2/P3.  It keeps the verified dataflow but changes the scheduling:

```text
bulk same-kernel serial baseline:
  gather/dequant full M -> grouped B0 GEMM full task list

chunked serial baseline:
  gather/dequant all chunks -> grouped B0 GEMM all chunks

pipeline candidate:
  stream 0: gather/dequant chunk i
  stream 1: wait event_i -> grouped B0 GEMM chunk i
```

The wait is a HIP stream dependency, not a device-side spin loop.  That is the main design choice: a GEMM block never occupies a CU polling for data that has not arrived.

## Constraints

This candidate is built to satisfy the three constraints from the discussion:

- A crosses XGMI once: `dispatch_gather_pack_range` writes the same local packed fp8+scale buffer as `b1_dispatch`, just chunked by row range.
- B0 remains B0: `grouped_b0_chunk` runs the 256x256x64 8-wave ping-pong body over local bf16 A.  There is no permanent producer/consumer wave split inside the GEMM block.
- No P1/P2-style spin waste: overlap is expressed with host stream events in `example.py`, so waiting work is not resident on the GPU.

## Comparison Target

Use the updated `EXPERIMENT_LEDGER.md` numbers as the bar:

- Current verified B1 path: `b1_dispatch SCHEDULE=b0`, `ROUTE=uniform TOTAL_M=8192 N=2048 K=7168`, `np=8`:
  `714 us` end to end, with gather RMS 0 and e2e RMS about 0.0037.
- Local production unfused baseline: `b2_production/b2_aiter.py` on the same node/container:
  `1255 us` for `aiter.fused_moe` with sorting, dynamic quant, native-fp8 full FFN, and local combine.

The first number is the immediate benchmark this overlap candidate must beat.  If it does not beat
`714 us`, it is not an improvement over the current verified approach, even if it beats the AITER
number.

The second number is useful context, but this folder still implements the B1 gather plus one B0-class
grouped GEMM leg for the selected `(N,K)`.  A full drop-in production replacement still needs the full
MoE FFN path: W13/gate-up, activation, W2/down, and routing outputs back to token order.

## Files

- `kernel.cpp`: range gather/pack, range dequant, and grouped B0 GEMM over a task subset.
- `example.py`: np=8 MPI/IRIS driver with `MODE=bulk|serial|pipeline|both|all`.

The driver imports the existing B1 route/task builders from `../b1_dispatch` so it stays comparable with the current B1-dispatch layout.

## Build

Build inside the HipKittens `distributed-kernels` tree, the same way as `b1_dispatch`:

```bash
cd <HK_ROOT>/distributed-kernels
cp -r <repo>/irisx/b1_overlap .
cp -r <repo>/irisx/b1_dispatch .   # needed for the Python route/task builders
cmake -B build_b1_overlap -DDK_BUILD=b1_overlap -DIRIS_HIP_ARCHITECTURES=gfx950
cmake --build build_b1_overlap -j16 --target b1_overlap
```

## Run

```bash
cd <HK_ROOT>/distributed-kernels/b1_overlap
ROUTE=uniform TOTAL_M=8192 N=2048 K=7168 CHUNK_ROWS=1024 MODE=all \
  mpirun -np 8 python3 example.py
```

To make the run fail unless the pipeline beats the current verified B1 path:

```bash
ROUTE=uniform TOTAL_M=8192 N=2048 K=7168 CHUNK_ROWS=1024 MODE=all REQUIRE_BEATS=b1_b0 \
  mpirun -np 8 python3 example.py
```

Useful knobs:

- `MODE=bulk|serial|pipeline|both|all`: `bulk` is the full-buffer same-kernel serial baseline, `serial` isolates chunking overhead, and `pipeline` is the overlap candidate.
- `CHUNK_ROWS=512|1024|2048`: overlap granularity.  Must be a multiple of 256.
- `ALL_CONSUMERS=1`: default; all 8 ranks run the pipeline.
- `ALL_CONSUMERS=0 CONSUMER=7`: single-consumer compatibility mode.
- `CHECK=0`: skip the CPU reference when only timing.
- `B1_DISPATCH_B0_US=714`: current verified B1-dispatch B0 baseline used for printed speedups.
- `AITER_UNFUSED_US=1255`: local production AITER baseline used for printed speedups.
- `REQUIRE_BEATS=none|bulk|serial|b1_b0|aiter`: nonzero exit if `pipeline` does not beat the selected target.

The timing printed is the max across ranks, so the reported number is the EP step limiter rather than just rank 0.
