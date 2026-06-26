# v4_bsingle_buffer — asymmetric buffering (A double, B single)

A copy of `irisx/v4_astationary_kernel/kernel.cpp` with **one** logical change: the local B
subtiles are single-buffered while the gathered A tile stays double-buffered.

## Exact diff from canonical V4

1. **Shared declaration** (`micro_tk`):
   ```diff
   - ST_A (&As)[NSTAGE]       = al.allocate<ST_A, NSTAGE>();
   - ST_B (&Bs)[NSTAGE][NSUB] = al.allocate<ST_B, NSTAGE, NSUB>();
   + ST_A (&As)[NSTAGE] = al.allocate<ST_A, NSTAGE>();   // A double-buffered (unchanged)
   + ST_B (&Bs)[NSUB]   = al.allocate<ST_B, NSUB>();      // B SINGLE-buffered
   ```

2. **`dynamic_shared_memory()`**:
   ```diff
   - return (size_t)NSTAGE * (sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
   + return (size_t)NSTAGE * sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B) + 1024;
   ```

3. **Prologue**: only A is prefetched (`As[s]`); the B load is removed from the prologue (one B
   buffer cannot be prefetched ahead).

4. **Main loop**: B is loaded into `Bs[sub]` (no stage index) for the CURRENT `tile`; A is still
   gathered ahead into `As[fetch % NSTAGE]`. Consumer indexes `Bs[sub]` and `As[cur]`.
   The existing per-tile `s_barrier` now also serializes producer-write/consumer-read of the single
   B buffer (correctness-preserving).

Everything else — the gather/dequant, swizzle, MFMA, accumulator layout, the baseline kernel, the
pybind module — is byte-identical to V4.

## LDS budget (st_bf<64,64> = 8192 B)

| config | formula | bytes | ~KB | blocks/CU @160KB |
|---|---|---|---|---|
| canonical V4 | `2*(8192 + 8*8192) + 1024` | 148480 | 145 | 1 |
| **this (A2,B1, NSUB8)** | `2*8192 + 8*8192 + 1024` | 82944 | 81 | 1 (160/82.9=1.93) |
| this + NSUB6 | `2*8192 + 6*8192 + 1024` | 66560 | 65 | 2 (160/65=2.46) |
| this + NSUB4 | `2*8192 + 4*8192 + 1024` | 50176 | 49 | 3 |

LDS saved by B single-buffering at NSUB=8 = **65536 B (~64 KB)** — the entire second B buffer.

## Build (main agent, on <NODE>, gentle/locked)

```
docker exec r1_c4 bash -lc '
  cp -r <repo>/irisx/lds_analysis/v4_bsingle_buffer \
        <HK_ROOT>/distributed-kernels/v4_bsingle_buffer
  cd <HK_ROOT>/distributed-kernels
  flock /tmp/mi355x_compile.lock -c "
    cmake -B build_10 -DDK_BUILD=v4_bsingle_buffer ;
    cmake --build build_10 -j8 --target v4_bsingle_buffer"'
```

Static resource check (subagent-safe, confirms LDS bytes):
```
HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage -c kernel.cpp -I<HK_ROOT>/include ...
# expect LDS ~82944 B for the fused micro_tk (vs ~148480 for canonical V4).
```

## Test (MAIN AGENT ONLY, GPU, serialized)

```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/v4_bsingle_buffer
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    M=1024 K=7168 N=2048 mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 2 python3 example.py"'
```
Copy V4's `example.py` (or symlink) into this dir — the ABI is identical.

## Correctness criteria
- RMS-rel vs reference ~0.0033 (same as V4; the change is LDS layout only, not math).
- zero-sentinel still proves the remote gather path.

## What to measure (the actual experiment)
- fused µs vs canonical V4 fused µs at M=1024/N=2048/K=7168.
- Static LDS bytes (confirm ~82944) and occupancy (rocprof: blocks/CU resident).
- HYPOTHESIS: at occ 1 perf is within noise of V4 (B prefetch overlap was cheap); the WIN is the
  freed ~64 KB. To convert that to occupancy, sweep NSUB=6 (this dir's `NSUB6` define) which should
  reach 2 blocks/CU while KEEPING A double-buffered. Watch the 384∤2048 tail (see Agent 07 note).
