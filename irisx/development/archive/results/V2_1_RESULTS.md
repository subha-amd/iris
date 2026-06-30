# V2.1 HipKittens remote-gather-fused GEMM — results (MI355X, gfx950, 2026-06-25)

V2.1 = the actual fusion win. A producer/consumer GEMM on **rank 1** whose **producer warps pull
each A-tile DIRECTLY from rank 0's IRIS heap** via `iris_ctx.load(&A[...], src_rank=0)`, overlapping
the cross-GPU gather of tile (t+1) with the consumer warps' MFMA of tile (t). This collapses the
old two-phase pattern (dispatch kernel remote-writes A into rank B's heap, THEN a separate GEMM
reads rank B's local heap) into a single fused kernel: a tile-level communication abstraction
inside a compute kernel.

Scope (as briefed): 2-rank remote-gather **correctness** only. bf16 A (no fp8 dequant — that was
proven in V2; bf16 isolates the new variable = the gather). No 8-expert pipeline, no down-proj/SiLU.

## What it computes
```
A : [M, K] bf16 — the ONLY initialized copy lives on rank 0's IRIS heap.
B : [N, K] bf16 — weights, local on rank 1.
C : [M, N] bf16 — output, local on rank 1.  C = A · B^T   (mma_ABt: C[M,N] += A[M,K]·B[N,K]^T)
M=512, K=2048, N=512   (modest, fits the tight VRAM left by the R1 server)
```
Tiling: BM=BN=BK=64. 4 producer warps gather A + load B; 4 consumer warps MFMA. Double-buffered
shared A/B tiles, AMD producer/consumer scheduling (s_waitcnt / s_barrier discipline) preserved.

## The mechanism (why the gather is a one-line change to the address)
IRIS `iris_device_view::translate(ptr, remote_rank)` does
`remote_ptr = heap_bases_[remote_rank] + (ptr - heap_bases_[cur_rank])`.
Because A is allocated with the **same allocation sequence on both ranks**, it sits at the **same
offset** in each rank's symmetric heap. So on rank 1, forming the *local* pointer `&g.a[{0,0,r,k}]`
and calling `iris_ctx.load(ptr, src_rank=0)` reads rank 0's A at that same offset over the IPC peer
mapping. The producer warps do exactly this, element-by-element, into shared memory; the consumer
warps then `load(rt, st)` + `mma_ABt` the previously-gathered tile.

## Files created (all NEW; nothing else edited)
- `<HK_ROOT>/distributed-kernels/fmoe_gather_gemm/kernel.cpp` — the V2.1 kernel
  (remote-gather producer/consumer GEMM + pybind `tk_kernel`). Mirrors bf16_gemm's structure.
- `<HK_ROOT>/distributed-kernels/fmoe_gather_gemm/example.py` — np=2 driver:
  IRIS world=2, rank-0-only A init, rank-1 sentinel A, runs the kernel on rank 1, builds the
  reference, reports max/RMS error and the remote-gather proof.
- `<HK_ROOT>/V2_1_RESULTS.md` — this file.

## Build (verified, gfx950 / CDNA4 default)
Done **inside the running `r1_c4` container** (it has the repo mounted at the same path plus
torch 2.10/rocm7.2.4, pybind11 3.0.4, hipcc, cmake, mpicxx). The bare host has no torch/pybind11.
One missing dep installed: `pip install mpi4py` (compiled against the container's OpenMPI).

```
# inside container r1_c4, cwd = <HK_ROOT>/distributed-kernels
cmake -B build -DDK_BUILD=fmoe_gather_gemm -DCPM_iris_SOURCE=<IRIS_ROOT>
cmake --build build --target fmoe_gather_gemm -j 16
# -> distributed-kernels/fmoe_gather_gemm/tk_kernel.cpython-312-x86_64-linux-gnu.so   (no reg spills)
```
**CPM local override note (important):** point `CPM_iris_SOURCE` at the PARENT `<IRIS_ROOT>`,
NOT at `<IRIS_ROOT>/irisx`. The CMakeLists has `CPMAddPackage(... SOURCE_SUBDIR irisx)`, so
CPM appends `irisx` to the source root. Pointing at `.../irisx` makes it look for `.../irisx/irisx`
and the `iris::iris` target never gets created. Pointing at the parent resolves to `.../iris/irisx`
and the `iris::iris` ALIAS target is defined correctly. (irisx still CPM-fetches spdlog/catch2 from
GitHub — the node has outbound HTTPS, so that works; no further override needed.)

## Run (verified, np=2)
```
# inside container r1_c4, cwd = .../distributed-kernels/fmoe_gather_gemm
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:64    # be gentle on the ~8GB free VRAM
mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 2 python3 example.py
```
The `--mca pml ob1 --mca btl self,vader` flags are REQUIRED: IB/mlx5 can't register memory while
the R1 server holds ~96% VRAM, so the default MPI transports fail at MPI_Init.

mpi4py + IRIS double-MPI_Init gotcha: the driver sets `mpi4py.rc.initialize=False` /
`rc.finalize=False` before `from mpi4py import MPI`, so IRIS (`iris::mpi::initialize()` in the Iris
ctor) owns the MPI lifecycle and `MPI_Init` is not called twice.

## Correctness result (PASSED)
```
[Rank 1] fmoe_gather_gemm  M=512 K=2048 N=512  src_rank=0
  rank1 local-A is zero sentinel : True
  kernel output all-zero?        : False   (False => gather pulled real A)
  max abs error                  : 0.015623
  max rel error                  : 0.054922
  RMS rel error                  : 0.003305
  RESULT                         : PASSED
```
RMS relative error 0.0033 (max-rel 0.055 on the smallest-magnitude entries) — well within bf16
tolerance for a K=2048 reduction.

## How we PROVED the remote gather was genuinely exercised (the crux)
Three independent pieces of evidence, all in the verified run:
1. **Sentinel isolation.** Rank 1's *local* A buffer is filled with **zeros** (`local-A is zero
   sentinel = True`); only rank 0 holds the real A. If the kernel had read rank 1's local A (a
   local fallback), C would be all-zero. It is **not** all-zero (`False`) and it **matches the
   reference built from rank 0's A** → the data must have come from rank 0 over IRIS.
2. **Direct translate/load dump.** A debug build (`-DGATHER_DEBUG`, off by default) printed from
   thread 0 on rank 1:
   ```
   [DBG] cur_rank=1 src_rank=0 aptr=0x..ee00000 base_cur=0x..ee00000 base_src=0x..ac00000
         off=0x0 remote=0x..ac00000 A[0,0]_remote=-0.202148
   ```
   i.e. rank 1 formed a local pointer (= its own heap base, off=0), IRIS translated it to rank 0's
   distinct heap base, and the load returned rank 0's A[0,0]. The two heap bases differ → it is a
   true cross-GPU IPC peer read, not a same-pointer no-op.
3. **End-to-end value match.** The full output equals `A_rank0 · B^T`, so every gathered A tile
   (not just element 0) carried rank 0's values.

## Overlap timing
Deferred (optional per brief; correctness was the required deliverable). The structure for overlap
is in place: producers prefetch tile (t+1)'s remote A gather + local B while consumers MFMA tile
(t), with the double-buffered tic/toc shared tiles and the s_barrier between stages. A timing A/B
vs a two-phase (separate remote-copy-then-local-GEMM) baseline is the natural next measurement.

## Deviations from the template / design
- **A is gathered element-wise** via `iris_ctx.load` into shared memory (then written at the
  swizzled LDS location so the consumer's swizzle-aware `load(rt,st)` round-trips), rather than
  reusing bf16_gemm's fast buffer-resource `G::load` for A. Reason: `G::load` derives its base
  pointer from the `gl<>` SRD set up for the *local* tensor; injecting a translated remote base
  into the SRD is possible but fragile, whereas the explicit element gather makes the remote path
  unambiguous and easy to prove. **B still uses the fast local `G::load`.** This keeps the new
  variable (the gather) isolated, exactly as the brief asked. (Optimizing A's gather to a
  vectorized/buffer-resource remote load is a clear follow-up.)
- Plain row-major A/B/C layouts (not bf16_gemm's pre-swizzled A indexing) so the gather offset math
  is transparent.
- Smaller tiling (64³, 4+4 warps) than the production 8-wave config — appropriate for a 2-rank
  correctness bring-up under tight VRAM.

## Deferred / next steps
1. **Overlap timing** vs the two-phase baseline (quantify the gather/compute overlap win).
2. **Vectorize the remote A gather** (buffer-resource / 128-bit remote loads instead of scalar
   `iris_ctx.load` per element) and prefetch deeper to actually hide IPC latency under MFMA.
3. Scale `src_rank` to a real per-tile mapping (the V2 buffer's `[local_e][src_rank][slot][H]`
   axis), i.e. different A tiles gathered from different producing ranks within one GEMM.
4. Reintroduce fp8 + per-128 dequant (from V2) on the gathered tiles, then the gate/up→down chain.

## Environment
gfx950 (256 CUs), ROCm 7.2.4 / hipcc HIP 7.2, container `r1_c4`
(`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`), torch 2.10+rocm7.2.4, python3.12, pybind11 3.0.4,
mpi4py 4.1.2, OpenMPI 3.1. IRIS from local `<IRIS_ROOT>/irisx` (branch muhaawd/irisx),
max_world_size=8, run at world=2.
