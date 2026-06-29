# BUG: b1_dispatch phase-1 gather produces wrong values for all remote (XGMI) rows

**Status:** Open — root cause not yet identified despite extensive investigation.
**Severity:** Blocks verified end-to-end Level-2 benchmark (gather timing measurements are
real, but gathered data is wrong, so results cannot be reported as verified).
**Last known good:** commit `1dca7c0e` ("B1-dispatch V0 PASSES all 5 routes") on node
`cv350-1e707-b02-2.mkm.dcgpu` inside container `r1_c4` (`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`).

---

## 1. Why this matters — the benchmark goal

This repo (`irisx`) implements a custom MoE expert dispatch kernel for 8-GPU EP8 inference
on AMD MI355X. The project goal is to compare our approach against the production unfused MoE
pipeline to show a verified end-to-end speedup.

The two pipeline designs being compared:

**Ours (b1_dispatch):**
```
Phase 1: dispatch_gather_pack  — multi-source XGMI gather of each expert's tokens, packed
                                  once into a local fp8 buffer (copy-once-then-compute)
Phase 2: grouped_gemm_b0       — 8-wave 256x256 B0-class grouped GEMM over the local buffer
```

**Production (aiter.fused_moe):**
```
EpDispatch → moe_sorting (×2) → dynamic_quant → fmoe_fp8_blockscale_g1u1 → EpCombine
(DeepSeek-R1-0528, E=32 local experts, fp8 e4m3 per-1x128, top-k=8, EP=8 ranks)
```

The claim to test: "copy tokens once over XGMI, then do a fast local GEMM" beats the
production approach of "dispatch + sort + quant + fused GEMM" per decode step.

### What we have so far (unconfirmed numbers)

**Level 1 — GEMM efficiency (verified ✅):**
grouped_b0 standalone correctness passes (RMS~0.0037, contamination 0) at all production
shapes on the old node. TFLOP/s at 8192 rows, E=8:

| shape | N | K | TFLOP/s (real rows) |
|---|---|---|---|
| default | 2048 | 7168 | 748 |
| fc1 (g1u1) | 4096 | 7168 | **840** |
| fc2 (down) | 7168 | 2048 | 709 |

Production aiter.fused_moe (E=32, full shape, TOKEN=1024, native fp8): **574.8 TFLOP/s**
At E=32 with the matched M_e distribution (~256 per expert), grouped_b0 produces:
420.2 TFLOP/s real (33% padding waste from BM=256) / 630.3 TFLOP/s padded.
Ratio vs production: 0.73 real-row. Gap = native-fp8 + fused fc2 advantage.

**Level 2 — full pipeline (unconfirmed ❌ — gather correctness failed):**
Measured on the failing node (gather corrupted, output wrong — timing still valid):

| SCHEDULE | T_gather | T_gemm | TFLOP/s (gemm) | T_total | TFLOP/s (e2e) |
|---|---|---|---|---|---|
| microtk (old 64×64) | 209 µs | 3457 µs | 69.6 | 3671 µs | 65.5 |
| **b0 (new 256×256)** | 211 µs | 342 µs | **703** | **558 µs** | **431** |
| production (aiter) | — | — | 574.8 | **1255 µs** | — |

**10.1× GEMM speedup** (b0 vs microtk) is independent of gather and verified.
**6.6× pipeline speedup** (558 µs vs 3671 µs) is timing on corrupted gather output — unconfirmed.
The production comparison (558 µs vs 1255 µs = 2.25×) looks promising but also unconfirmed
because the gather output feeding the GEMM is wrong.

**The gather bug is the single blocker** preventing these numbers from being reported
as a verified result.

---

## 2. Symptom

Running `b1_dispatch/example.py` with `mpirun -np 8` produces:

```
[b1-dispatch] route=uniform E=32 TOTAL_M=8192 -> Mpacked=8192 NSUB=8
              segs=420 tiles=128 single_src_tiles=0 multi_src_tiles=128
[phase1-probe] packed-A vs ref A_deq: RMS=0.931706  rows_mismatch=7111/8192
remote (XGMI) gathered rows = 7111
packed_A_nonzero=True  C_zero=False  remote_path=True
-> FAILED
T_gather (phase1): 209.44 us
T_gemm   (phase2): 342.09 us   703.09 TFLOP/s
T_total:           558.28 us   430.82 TFLOP/s (e2e)
```

Key observations:
- **Exactly 7111 rows are wrong**, which equals the number of rows that came from remote ranks
  (all ranks except CONSUMER=7). Local rows are correct.
- `packed_A_nonzero=True` — the kernel IS writing something, just wrong data.
- `RMS=0.931706` — stable and consistent across runs, not random noise.
- `single_src_tiles=0` — the fast single-source Path 2 never triggers on any route.
- On some runs `RMS=inf` (not 0.93) — this happens when `deq_all` (the MPI-gathered reference)
  is empty on the consumer, making the reference `A_deq_ref` all zeros. Division by zero in the
  RMS formula gives inf. The underlying gather failure is the same.

---

## 3. Environment where it PASSES

- **Node:** `cv350-1e707-b02-2.mkm.dcgpu` (8× gfx950 MI355X)
- **Container:** `r1_c4` (`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`)
- **ROCm:** HIP 7.2.53211, clang 22 (`roc-7.2.4`), XNACK enabled
- **iris build:** tk_kernel.so built Jun 26 (CPM fetch of `muhaawd/irisx` at that time)
- **Result:** RMS=0.000000, rows_mismatch=0/8192, all 5 routes PASSED

## 4. Environment where it FAILS

- **Node:** `cv350-rck-g03-f03-18.rck.dcgpu` (8× gfx950 MI355X)
- **Container:** `qilihuan-dsv4-dp8-ep-vllm0617` (`sabreshao/vllm:dsv4_0615n`)
- **ROCm:** HIP 7.2.53211, clang 22 (`roc-7.2.3`)
- **XNACK:** Initially disabled (amdgpu `noretry=1`). Manually re-enabled: `sudo rmmod amdgpu &&
  sudo modprobe amdgpu noretry=0`, then `HSA_XNACK=1` passed to all MPI workers via `-x`.
  Fine-grained alloc now works, but gather still fails.
- **iris build:** tk_kernel.so built Jun 29 (CPM fetch of `muhaawd/irisx` at that time)
- **Result:** RMS=0.931706, 7111/8192 rows wrong

---

## 5. Exact reproduction steps

```bash
# === On the failing node cv350-rck-g03-f03-18.rck.dcgpu ===

# 1. Clone HipKittens — the AMD HIP port (NOT ThunderKittens which is NVIDIA-only)
git clone --depth=1 https://github.com/HazyResearch/HipKittens.git ~/HipKittens

# 2. Copy irisx/b1_dispatch into distributed-kernels
cp -r <iris_repo>/irisx/b1_dispatch ~/HipKittens/distributed-kernels/b1_dispatch

# 3. Copy into the container
docker cp ~/HipKittens qilihuan-dsv4-dp8-ep-vllm0617:/tmp/HipKittens

# 4. Build inside the container (fetches iris via CPM from ROCm/iris:muhaawd/irisx)
docker exec qilihuan-dsv4-dp8-ep-vllm0617 bash -c '
  cd /tmp/HipKittens/distributed-kernels
  cmake -B build -DGPU_TARGET=CDNA4 -DDK_BUILD=b1_dispatch
  cmake --build build -j16 --target b1_dispatch
  cmake --build build -j16 --target iris_py
'
# Build is clean: no errors, no VGPR spill, LDS=131072, occupancy=2 waves/SIMD

# 5. Enable XNACK (requires sudo; node should be exclusively reserved)
sudo rmmod amdgpu
sudo modprobe amdgpu noretry=0

# 6. Install mpi4py
docker exec qilihuan-dsv4-dp8-ep-vllm0617 pip install mpi4py -q

# 7. Run the benchmark — this produces the failure
docker exec qilihuan-dsv4-dp8-ep-vllm0617 bash -c '
  cd /tmp/HipKittens/distributed-kernels/b1_dispatch
  mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 \
    -x HSA_XNACK=1 -x PYTHONPATH=/tmp/HipKittens/distributed-kernels \
    -x ROUTE=uniform -x TOTAL_M=8192 -x N=2048 -x SCHEDULE=microtk \
    python3 example.py
'
```

---

## 6. What was investigated and ruled out

### ✅ RULED OUT: Signed-char `row_seg` overflow

The historical bug was `row_seg[]` stored as `signed char`, overflowing past index 127 with
~420 segments. The fix is present in both `ep8_gather.h` and `kernel.cpp`:
```cpp
// ep8_gather.h:117
void build_row_seg_map(int (&row_seg)[BM], ...)

// kernel.cpp:102
__shared__ int row_seg[GP_BM];

// ep8_gather.h:51
static constexpr int SEG_NONE = -1;
```
Confirmed by grep on the compiled source inside the container.

### ✅ RULED OUT: iris flag value mismatch

The iris source uses the macro `hipDeviceMallocFinegrained` (not a hardcoded `0x8`):
```cpp
// iris.hpp:49
const auto flags = hipDeviceMallocFinegrained;
hip_try(hipExtMallocWithFlags(&ptr, bytes, flags));
```
The container's ROCm defines this correctly as `0x1`:
```
/opt/rocm/include/hip/hip_runtime_api.h:#define hipDeviceMallocFinegrained 0x1
```

### ✅ RULED OUT: Fine-grained alloc fails in MPI workers

After enabling XNACK, `hipExtMallocWithFlags` with flag `0x1` returns 0 (success) on all 8
MPI workers:
```
rank=0 fg_ret=0 (b'no error') ptr=124639571345408 xnack=1
rank=1 fg_ret=0 (b'no error') ptr=132693507440640 xnack=1
... (all 8 ranks: ret=0)
```

### ✅ RULED OUT: IPC cross-rank GPU reads fail at the host level

`test_ipc.py` manually tests `hipIpcOpenMemHandle` + `hipMemcpy` cross-rank:
```
rank=0 reads from rank=1: got=101.0 expected=101.0 OK
rank=1 reads from rank=2: got=201.0 expected=201.0 OK
... all 8 pairs: OK
```

### ✅ RULED OUT: `ctx.load()` inside a `__global__` kernel returns wrong values

A minimal HIP kernel (`test_kernel_load.cu`) was written and built that:
1. Rank N writes `N*100 + i` to a fine-grained iris heap buffer.
2. A kernel on rank 0 calls `ctx.load(&src[i], 1)` for each element.
3. Result is verified against `1*100 + i = 100+i`.

Result: all 8 values correct. `ctx.load()` works inside a kernel on this platform.
```
rank=0 reading from rank=1:
  [0] got=100.0 expected=100.0 OK
  [1] got=101.0 expected=101.0 OK
  ... all 8: OK
Result: PASS
```

### ✅ RULED OUT: `make_fp8` torch tensor not on iris heap

`make_fp8` in `example.py` calls `iris.empty()` then wraps it with `__cuda_array_interface__`.
Tested that `data_ptr()` of the returned tensor equals the iris allocation pointer:
```
iris_ptr    = 0x754ebee00000
bf16.data_ptr = 0x754ebee00000  (same as iris: True)
fp8.data_ptr  = 0x754ebee00000  (same as iris: True)
  bf16 offset from heap=0  in_heap=True
  fp8 offset from heap=0  in_heap=True
```

### ✅ RULED OUT: P2P peer access disabled

`torch.cuda.can_device_access_peer(i, j)` returns True for all GPU pairs.

### ✅ RULED OUT: Container lacks IPC/privilege

`docker inspect`: `Privileged=true`, `IpcMode=host`. Full access confirmed.

### ✅ RULED OUT: XNACK not reaching MPI workers

`mpirun -x HSA_XNACK=1` propagation confirmed:
```
rank=0 HSA_XNACK=1
rank=1 HSA_XNACK=1
... all 8 ranks
```

### ✅ RULED OUT: Iris heap bases are zero or wrong

With `verbose=True`, iris prints heap bases from rank 0's perspective after IPC exchange.
All 8 ranks see identical heap base addresses (correct for symmetric IPC mapping):
```
Rank: 0 GPU: 0 heap base: 137286463258624
Rank: 0 GPU: 1 heap base: 128567948083200
Rank: 0 GPU: 2 heap base: 133443943923712
...
Rank: 4 GPU: 0 heap base: 137286463258624  ← identical across all ranks
Rank: 4 GPU: 1 heap base: 128567948083200
...
```
All heap bases are non-zero and consistent across all 8 ranks.

### ✅ RULED OUT: Tile metadata (`seg_begin`, `seg_count`) read incorrectly by kernel

A debug printf was added to `gather_pack_kernel` at tile 0, thread 0. The kernel reads:
```
[DBG tile0] seg_begin=0 seg_count=4 tile_dst0=0 valid_rows=64
[DBG seg0] expert=0 src_rank=1 src_row=0 dst_row=0 rc=16
```
These match the Python-side `tile_arr` and `seg_arr` exactly. The kernel reads the
correct metadata from `tilemeta` and `seg` globals.

---

## 7. What the kernel debug revealed — the current state of understanding

After instrumenting the gather kernel at the `ctx.load()` call site:

```
[DBG load] src_rank=1 src_row=0 cur_rank=7 sp=0x7e561778a0c0 a_src_base=0x7e561778a0c0 vbytes.x=3974380906
```

Key observations:
- `sp == a_src_base` — for `src_row=0, kc=0`, this is expected (`sp = a_src_base + 0*K + 0`)
- `sp = 0x7e561778a0c0` — this is rank 7 (CONSUMER)'s `A_src` buffer pointer
- `vbytes.x = 3974380906` — a non-zero, non-garbage-looking value is returned
- The reference says this row should come from rank 1's source, but we don't yet know what
  value `vbytes.x` *should* be to say whether it's right or wrong at this level

The debug shows `ctx.load()` is executing (not zeroing, not crashing). The `translate()` call
uses `heap_bases_[1]` (rank 1's IPC-mapped address) and `heap_bases_[7]` (consumer's own heap
base) to compute the remote pointer. Since all ranks share the same IPC heap base addresses
(confirmed above), the offset `sp - heap_bases_[7]` should correctly map to rank 1's `A_src`.

The investigation stopped here because we have not yet compared `vbytes.x` to what rank 1
actually wrote into `A_src_fp8[0, 0]` at the time of the load.

---

## 8. The kernel code path being executed

The gather kernel (`kernel.cpp:101–196`) for each 16-byte chunk of each packed row:

```cpp
// 1. Resolve source from segment map (Path 1, multi-source)
const int sidx = row_seg[r];               // from build_row_seg_map
const route_segment s = v.segs[sidx];
src_rank = s.src_rank;
src_row  = s.src_row_begin + local_dst;

// 2. Compute source pointer (in consumer rank's address space)
const uint4* sp = reinterpret_cast<const uint4*>(a_src_base + src_row * K + kc);

// 3. Remote load via iris translate()
vbytes = ctx.load(sp, src_rank);
//  → translate(sp, src_rank):
//       offset = sp - heap_bases_[cur_rank_]   ← cur_rank_ = 7
//       return heap_bases_[src_rank] + offset   ← src_rank = 0..6

// 4. Write to packed destination
*reinterpret_cast<uint4*>(a_dst_base + dst_row * K + kc) = vbytes;
```

`a_src_base` is from rank 7's `A_src_bf16` iris allocation (confirmed on-heap).
`heap_bases_[7]` is rank 7's own heap base (confirmed non-zero).
`heap_bases_[src_rank]` is the IPC-mapped base for remote rank (confirmed non-zero, consistent).
`ctx.load()` works correctly in isolation (confirmed by `test_kernel_load`).

The math appears correct. The data appears non-zero. Yet the reference comparison fails for
7111/8192 rows.

---

## 9. What has NOT yet been checked

The next thing to verify — not yet done because the session ended here:

**Does `vbytes.x` at the first remote load actually equal what rank 1 wrote into `A_src_fp8[0, 0]`?**

This would require printing from two sides: the consumer kernel's load result AND the source
rank's written value. The infrastructure to do this is in place (debug printf already works);
it just needs one more run with a cross-rank comparison print.

If `vbytes.x` equals the correct value, then the kernel is loading correctly and the bug
is in how the reference (`A_deq_ref`) is being constructed or compared.

If `vbytes.x` does NOT equal the correct value, then `ctx.load()` returns wrong data despite
the standalone test passing — pointing to a subtle difference between the standalone test
context (simple C++ MPI binary) and the pybind/Python-launched kernel context.

---

## 10. Relevant file locations

| File | Content |
|---|---|
| `b1_dispatch/kernel.cpp` | Full gather + B0 GEMM kernel. Lines 101–196: gather. Lines 119–132: tilemeta + ptr setup. Line 181: `ctx.load()` call. |
| `b1_dispatch/ep8_gather.h` | `route_segment` struct (5 fields: expert_id, src_rank, src_row_begin, dst_row_begin, row_count). `tile_is_single_source()`. `build_row_seg_map()`. |
| `b1_dispatch/example.py` | Full Python driver. Line 74: iris init. Lines 134–144: `make_fp8`. Lines 190–193: phase1 call. Lines 241–253: phase1 isolation probe. |
| `b1_dispatch/b1_dispatch_route.py` | `build_multisource_route()` — builds `segs` + `tiles`. `segs_to_int_array()` — 5-field layout matching `route_segment`. |
| `irisx/test_ipc.py` | Standalone IPC cross-rank read test (PASSES). |
| `irisx/test_kernel_load.cu` | Minimal HIP kernel `ctx.load()` test (PASSES). |
| `irisx/test_fg.py` | Fine-grained alloc test across 8 MPI workers (PASSES). |
| `irisx/test_heap_range.py` | Confirms iris.empty() data_ptr is within heap (PASSES). |
| `irisx/test_iris_bases.py` | Confirms all 8 ranks see identical non-zero heap bases (PASSES). |
| iris source (in container) | `/tmp/HipKittens/distributed-kernels/build/_deps/iris-src/irisx/include/iris/iris.hpp` |

---

## 11. Git context

```
Branch:   subha/moe-dispatch-v0
Repo:     https://github.com/subha-amd/iris

Last commit with gather passing: 1dca7c0e  B1-dispatch V0 PASSES all 5 routes
Current HEAD:                    7b4f46d5  b1_dispatch: add BUG.md documenting gather correctness failure

Gather kernel code (kernel.cpp gather section + ep8_gather.h) is unchanged
between 1dca7c0e and HEAD — confirmed by git diff.
```

The additions between `1dca7c0e` and HEAD: `grouped_gemm_b0` GEMM path in `kernel.cpp`,
`b0_tasks.py`, `example.py` SCHEDULE knob, `b2_aiter.py`. None touch the gather.
The one `kernel.cpp` change that affected gather: `g.a.raw_ptr` fix in
`dispatch_grouped_gemm_b0` (host function), which does not affect `gather_pack_kernel`.

---

## 12. Failing node configuration summary

```
Node:        cv350-rck-g03-f03-18.rck.dcgpu
GPUs:        8× gfx950 (MI355X), ~250 GB VRAM free per GPU
ROCm:        7.2.53211 (roc-7.2.3)
Container:   qilihuan-dsv4-dp8-ep-vllm0617 (sabreshao/vllm:dsv4_0615n)
             Privileged=true, IpcMode=host, all 8 GPUs exposed
XNACK:       Enabled via sudo modprobe amdgpu noretry=0 + HSA_XNACK=1
iris build:  CPM fetches ROCm/iris branch muhaawd/irisx at build time
             (exact commit SHA not pinned — could differ from Jun 26 build)
MPI:         OpenMPI 4.1.x, mpirun with --mca btl self,vader
HipKittens:  github.com/HazyResearch/HipKittens (cdna4 port), depth=1 clone
```
