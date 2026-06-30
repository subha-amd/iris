# V2.1 — IRIS remote-gather fused GEMM (the fusion win)

**The core research result of the project**: a producer/consumer GEMM whose producer warps
pull each A-tile **directly from a remote GPU's IRIS heap** (`iris_ctx.load(ptr, src_rank)`),
double-buffered so the cross-GPU gather of tile (t+1) overlaps the consumer warps' MFMA of
tile (t). This collapses the old two-phase "dispatch kernel writes to peer, then peer runs a
separate GEMM" into one fused kernel — the tile-level communication-inside-compute abstraction
the whole project set out to demonstrate.

Mirrored here for tracking; built inside a HipKittens checkout (depends on HK + IRIS).
Built/run inside the ATOM container (needs torch/pybind11). Place under
`HipKittens/distributed-kernels/fmoe_gather_gemm/`.

## Build / run (2-rank, gfx950)
```
pip install mpi4py
cmake -B build -DDK_BUILD=fmoe_gather_gemm -DCPM_iris_SOURCE=<IRIS_ROOT>   # parent of irisx/
cmake --build build --target fmoe_gather_gemm -j 16
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:64
mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 example.py
```
(`--mca pml ob1 --mca btl self,vader` is required — the node's IB can't register memory.)

## Verified result (8×MI355X, gfx950, np=2 — independently re-run)
- rank 1's **local A is all-zero sentinel**; rank 0 holds the only real A.
- kernel output is **non-zero and matches `A_rank0·Bᵀ`** → the GEMM pulled A from rank 0 over IRIS.
- RMS-rel error **0.0033** (bf16 tolerance). PASSED.
- Also a direct `iris.translate` dump confirmed rank 1 read rank 0's distinct heap base (true IPC peer read, not a same-pointer no-op).

## Scope / deferred
This isolates and PROVES the remote gather: A is gathered element-wise (B stays local fast load),
bf16 (no fp8), 2 ranks, gate/up only. Next steps: vectorize the remote load, time the overlap vs
a two-phase baseline, per-tile `src_rank` mapping (real expert→rank routing), re-add the V1 fp8
dequant, scale to 8 ranks / full expert FFN (down-proj + SiLU).
