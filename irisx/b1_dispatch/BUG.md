# BUG: b1_dispatch phase-1 gather produces wrong values for all remote (XGMI) rows

**Status:** Open. Code is believed correct; failure is suspected environmental/configuration.  
**Severity:** Blocks verified end-to-end Level-2 benchmark (gather timing is valid, output is wrong).  
**Last known good:** commit `1dca7c0e` ("B1-dispatch V0 PASSES all 5 routes") on node
`cv350-1e707-b02-2.mkm.dcgpu` inside container `r1_c4` (`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`).

---

## 1. Symptom

Running `b1_dispatch/example.py` with `mpirun -np 8` produces:

```
[phase1-probe] packed-A vs ref A_deq: RMS=0.931706  rows_mismatch=7111/8192
remote (XGMI) gathered rows = 7111
-> FAILED
```

- Exactly **7111 rows are wrong** — which equals the number of rows gathered from **remote ranks**
  (i.e. all ranks other than `CONSUMER`). Local rows (from `CONSUMER`'s own source buffer) are correct.
- `packed_A_nonzero=True` — the packed buffer is non-zero, so *something* is being written.
- `C_zero=False` — the GEMM ran on the garbage data and produced non-zero output.
- The RMS of ~0.93 indicates the packed values bear no relationship to the correct source values.
- On some runs (when `deq_all` is empty due to an upstream failure) `RMS=inf` instead of `0.93`.

---

## 2. What the gather is supposed to do

**Phase 1 (`dispatch_gather_pack`)** is a multi-source EP8 gather. The CONSUMER rank
(`CONSUMER=7` by default) runs a CUDA/HIP kernel that:

1. For each 64-row tile of the output packed buffer, reads `tilemeta` to find which route
   segments contribute to that tile.
2. For each row in the tile, resolves `(src_rank, src_row)` — which source rank holds this
   token and at which row.
3. If `src_rank == CONSUMER` (local): reads directly from the local IRIS heap.
4. If `src_rank != CONSUMER` (remote): uses `ctx.load(ptr, src_rank)` — the IRIS remote load
   primitive — to read the raw fp8 bytes + fp32 scales from the remote rank's IRIS heap over XGMI.

The IRIS remote load works by:
```cpp
// iris.hpp::iris_device_view::translate()
offset = reinterpret_cast<uintptr_t>(ptr) - heap_bases_[cur_rank_];
return reinterpret_cast<T*>(heap_bases_[remote_rank] + offset);
```
Each rank has a fine-grained (`hipExtMallocWithFlags(hipDeviceMallocFinegrained, 0x1)`) heap.
All allocations are symmetric (identical order on every rank) so the offset from heap base to
any given buffer is the same on every rank. IPC handles (`hipIpcOpenMemHandle`) give each rank
a mapped pointer into every other rank's heap.

---

## 3. Environment where it PASSES

- **Node:** `cv350-1e707-b02-2.mkm.dcgpu` (8× MI355X gfx950)
- **Container:** `r1_c4` (`rocm/atom-dev:vllm-v0.22.0-nightly_20260610`)
- **ROCm:** HIP 7.2.53211, hipcc clang 22 (`roc-7.2.4`)
- **Commit:** `1dca7c0e` (the gather kernel code at that commit is byte-identical to HEAD)
- **Result:** RMS=0.000000, zero rows mismatched, all 5 routes PASSED

## 4. Environment where it FAILS

- **Node:** `cv350-rck-g03-f03-18.rck.dcgpu` (8× MI355X gfx950)
- **Container:** `qilihuan-dsv4-dp8-ep-vllm0617` (`sabreshao/vllm:dsv4_0615n`)
- **ROCm:** HIP 7.2.53211, hipcc clang 22 (`roc-7.2.3`)
- **XNACK at test time:** Enabled via `HSA_XNACK=1` + kernel module reloaded with `noretry=0`
- **Result:** RMS=0.931706, 7111/8192 rows wrong (all XGMI rows)

---

## 5. Exact reproduction steps

### 5a. Build

```bash
# On node cv350-rck-g03-f03-18.rck.dcgpu (or equivalent 8xMI355X with qilihuan container)
# 1. Clone HipKittens (AMD HIP port — NOT ThunderKittens)
git clone --depth=1 https://github.com/HazyResearch/HipKittens.git ~/HipKittens

# 2. Sync irisx/b1_dispatch into HipKittens distributed-kernels
cp -r <path-to-iris-repo>/irisx/b1_dispatch ~/HipKittens/distributed-kernels/b1_dispatch

# 3. Copy HipKittens tree into container (it lives in /tmp inside the container)
docker cp ~/HipKittens qilihuan-dsv4-dp8-ep-vllm0617:/tmp/HipKittens

# 4. Build b1_dispatch + iris_py inside container
docker exec qilihuan-dsv4-dp8-ep-vllm0617 bash -c '
  cd /tmp/HipKittens/distributed-kernels
  cmake -B build -DGPU_TARGET=CDNA4 -DDK_BUILD=b1_dispatch
  cmake --build build -j16 --target b1_dispatch
  cmake --build build -j16 --target iris_py
'
# Build is clean — no errors, no VGPRs spill.
```

### 5b. Reproduce the failure

```bash
# First: enable XNACK at kernel driver level (requires sudo, node reserved exclusively)
sudo rmmod amdgpu
sudo modprobe amdgpu noretry=0

# Install mpi4py inside container if not present
docker exec qilihuan-dsv4-dp8-ep-vllm0617 pip install mpi4py -q

# Run the gather benchmark
docker exec qilihuan-dsv4-dp8-ep-vllm0617 bash -c '
  cd /tmp/HipKittens/distributed-kernels/b1_dispatch
  mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 \
    -x HSA_XNACK=1 -x PYTHONPATH=/tmp/HipKittens/distributed-kernels \
    -x ROUTE=uniform -x TOTAL_M=8192 -x N=2048 -x SCHEDULE=microtk \
    python3 example.py
'
```

**Expected output (passing):**
```
[phase1-probe] packed-A vs ref A_deq: RMS=0.000000  rows_mismatch=0/8192
-> PASSED
```

**Actual output (failing):**
```
[phase1-probe] packed-A vs ref A_deq: RMS=0.931706  rows_mismatch=7111/8192
remote (XGMI) gathered rows = 7111
-> FAILED
```

### 5c. Verify fine-grained memory works (it does — this is NOT the bug)

```python
# test_fg.py — run with: mpirun -np 8 -x HSA_XNACK=1 python3 test_fg.py
import os, ctypes
lib = ctypes.CDLL("libamdhip64.so")
ptr0 = ctypes.c_void_p()
lib.hipMalloc(ctypes.byref(ptr0), ctypes.c_size_t(4096))
ptr = ctypes.c_void_p()
ret = lib.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(4096), ctypes.c_uint(0x1))
lib.hipGetErrorString.restype = ctypes.c_char_p
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
print(f"rank={rank} fg_ret={ret} ({lib.hipGetErrorString(ret)}) ptr={ptr.value}", flush=True)
# Result: fg_ret=0 (no error) on all 8 ranks. Fine-grained alloc WORKS.
```

### 5d. Verify IPC cross-rank reads work at the Python/ctypes level (they do)

```python
# test_ipc.py — see irisx/test_ipc.py for full source
# Result: every rank reads the correct value from its neighbor's GPU buffer via hipIpcOpenMemHandle.
# rank=0 reads from rank=1: got=101.0 expected=101.0 OK
# rank=1 reads from rank=2: got=201.0 expected=201.0 OK
# ... all 8 ranks: OK
```

So: fine-grained alloc works, IPC cross-rank GPU reads work at the host level. The failure is
inside the HIP `__global__` kernel when `ctx.load(ptr, src_rank)` is called.

---

## 6. Key facts that narrow the cause

1. **The gather kernel code is byte-identical to the last-passing commit (`1dca7c0e`).** This was
   verified with `git diff 1dca7c0e HEAD -- irisx/b1_dispatch/kernel.cpp`. The gather kernel
   (`gather_pack_kernel`) and all its helpers (`ep8_gather.h`) are unchanged.

2. **The signed-char overflow bug is NOT the cause.** The known historical bug was `row_seg[]`
   stored as `signed char`, which overflows past 127 with ~420 segments, corrupting ~70% of rows.
   That fix is present: `ep8_gather.h:117` uses `int (&row_seg)[BM]`,
   `kernel.cpp:102` declares `__shared__ int row_seg[GP_BM]`, `SEG_NONE` is `int(-1)`.
   Confirmed by grep on the node's compiled source.

3. **`rows_mismatch` count equals exactly the XGMI row count.** `remote (XGMI) gathered rows =
   7111` and `rows_mismatch=7111/8192`. This is not a coincidence — local rows are fine, remote
   rows are wrong. Every route produces this pattern.

4. **`RMS=0.931706` is stable and consistent.** The wrong values are not random noise — they
   appear to be systematic garbage (possibly zeros, or values from another buffer).

5. **`packed_A_nonzero=True`** — the kernel IS writing something to the remote rows, it's just
   the wrong data.

6. **`single_src_tiles=0` on every route.** The `tile_is_single_source()` predicate always
   returns false, even for `one_hot` routing. This means Path 2 (the fast single-source path)
   never executes — all traffic goes through Path 1 (the multi-source segment iterator). This
   is worth investigating independently: either the tile metadata is built incorrectly, or the
   predicate condition is wrong for this configuration.

7. **XNACK was initially disabled (`noretry=1`).** After reloading the driver with `noretry=0`
   and setting `HSA_XNACK=1`, fine-grained alloc works, but the gather still fails. So XNACK
   being initially off may have masked the real bug, but enabling XNACK alone doesn't fix it.

8. **The iris CMakeLists fetches iris from `ROCm/iris:muhaawd/irisx` via CPM.** This branch
   may differ from the branch that was compiled on the passing node. Specifically, the passing
   node used an already-compiled `tk_kernel.so` from a prior session (Jun 26 build) that may
   have pulled a different iris commit. The iris `translate()` function, `malloc_fine_grained()`
   flag value, and IPC handle exchange are all in iris — if any of these differ, the heap bases
   could be set up incorrectly.

9. **The ROCm minor version differs slightly.** Passing node: `roc-7.2.4`. Failing node:
   `roc-7.2.3`. This is unlikely to matter for the gather logic but worth noting.

---

## 7. Most likely hypotheses (ordered by confidence)

### H1 (HIGH): iris CMakeLists fetches a different commit of iris that has a bug in fine-grained heap setup

The CMakeLists.txt:
```cmake
CPMAddPackage(
  NAME iris
  GITHUB_REPOSITORY ROCm/iris
  GIT_TAG muhaawd/irisx
  SOURCE_SUBDIR irisx
  ...
)
```

This fetches the tip of `muhaawd/irisx` at build time. If that branch changed between when the
passing node built (Jun 26) and when the failing node built (Jun 29), the iris heap setup could
differ. Specifically:
- The `hipDeviceMallocFinegrained` flag value changed between ROCm versions:
  old version = `0x8`, new (this node's ROCm) = `0x1`.
- If the fetched iris uses `0x8` (hardcoded) rather than the system header's `hipDeviceMallocFinegrained`
  macro, the alloc silently fails and `heap_bases_[i]` is 0 for remote ranks.
- `translate()` would then compute `heap_bases_[remote_rank] + offset = 0 + offset = offset` —
  a small integer — dereferencing which reads garbage or triggers an access at a nonsensical address.

**How to check:** Inside the container, read the compiled iris source:
```bash
cat /tmp/HipKittens/distributed-kernels/build/_deps/iris-src/irisx/include/iris/iris.hpp \
  | grep -A3 'malloc_fine_grained\|hipExtMallocWithFlags'
```
If it contains a hardcoded `0x8` instead of `hipDeviceMallocFinegrained`, that's the bug.

**Fix:** Patch iris to use `0x1` (the correct flag for this ROCm), or update the iris branch.
Alternatively, point CPM at a pinned commit that's known to work.

### H2 (MEDIUM): IPC handle exchange in iris succeeds but the opened handles map to wrong addresses inside the kernel

The iris constructor calls `hipIpcOpenMemHandle` for each remote rank's fine-grained heap, storing
the mapped base in `heap_bases_[i]`. In the host-level test (`test_ipc.py`) this works correctly.
But the host test uses `hipMemcpy` (D2H copy) to read, whereas the kernel uses a direct pointer
dereference (`*remote_ptr`). On gfx950 with some configurations, fine-grained memory opened via
IPC may not be directly dereferenceable from inside a kernel without additional setup (e.g.
`hipDeviceEnablePeerAccess` or specific SVM flags). The host `hipMemcpy` goes through the driver
which handles the translation; a direct kernel dereference may not.

**How to check:** Write a minimal HIP kernel that:
1. Takes a fine-grained pointer opened via IPC from another rank.
2. Reads a value from it with a direct dereference.
3. Writes the result to a host-visible buffer.
Compare the value to what `hipMemcpy` reads from the same address.

### H3 (MEDIUM): `tile_is_single_source()` always returning false corrupts Path 1

`single_src_tiles=0` for every route including `one_hot`. The `tile_is_single_source()` predicate
is in `ep8_gather.h`. If the tilemeta passed to the kernel has `seg_count=0` for all tiles (e.g.
a metadata-build bug where all segments are unrouted), then `tile_is_single_source()` might
consistently return false AND the segment iterator finds no valid routes, writing zeros everywhere —
but zeros would give `packed_A_nonzero=False`, which we don't see. So this is less likely as the
primary cause but the `single_src_tiles=0` across all routes is suspicious and worth investigating.

**How to check:** Print the first few entries of `tile_arr` (the tilemeta numpy array) before
copying to the GPU. Specifically check `tile_arr[0]` = `[seg_begin, seg_count, tile_dst0, valid_rows]`
for the first tile. If `seg_count=0` on all tiles, the metadata build is wrong.

### H4 (LOW): MPI `vader` (shared-memory) transport corrupts the IPC handle exchange

The `vader` MPI transport uses shared memory for intra-node communication. During iris init,
MPI allgather is used to exchange IPC handles (64-byte opaque structs). If `vader` corrupts these
bytes (e.g. alignment issues), the opened IPC handles would map to wrong addresses. Unlikely since
this is the same transport used on the passing node, but worth ruling out with `--mca btl tcp`
to force TCP transport.

---

## 8. Relevant code locations

| File | Lines | What it does |
|---|---|---|
| `b1_dispatch/kernel.cpp` | 86–196 | `gather_pack_kernel` + `dispatch_gather_pack` |
| `b1_dispatch/kernel.cpp` | 101–103 | `__shared__ int row_seg[GP_BM]` — the int fix |
| `b1_dispatch/kernel.cpp` | 117–126 | tilemeta read + pointer setup from `gl` |
| `b1_dispatch/kernel.cpp` | 131 | `build_row_seg_map` call (Path 1) |
| `b1_dispatch/kernel.cpp` | 148–162 | Per-row `(src_rank, src_row)` resolution |
| `b1_dispatch/kernel.cpp` | 168–174 | `ctx.load(sp, src_rank)` — the remote read |
| `b1_dispatch/ep8_gather.h` | 51 | `constexpr int SEG_NONE = -1` |
| `b1_dispatch/ep8_gather.h` | 109–131 | `build_row_seg_map` — fills `row_seg[]` |
| `b1_dispatch/ep8_gather.h` | 140–190 | `gather_copy_row` — Path 1 vs Path 2 dispatch |
| `b1_dispatch/ep8_gather.h` | 375–382 | `translate()` in iris_device_view |
| `b1_dispatch/example.py` | 74 | `iris = iris_py.Iris(heap_size_mb=512)` |
| `b1_dispatch/example.py` | 120–132 | `make_iris` — symmetric-heap tensor allocation |
| `b1_dispatch/example.py` | 134–144 | `make_fp8` — bf16-aliased fp8 on iris heap |
| `b1_dispatch/example.py` | 186–193 | `iris_ctx = iris.get_device_view()` + phase1 call |
| `b1_dispatch/example.py` | 206–225 | `build_reference()` — CPU reference construction |
| `b1_dispatch/example.py` | 241–253 | Phase-1 isolation probe (RMS + rows_mismatch) |

**The iris library source** (fetched via CPM during build) lives at:
`<build_dir>/_deps/iris-src/irisx/include/iris/iris.hpp`

Key functions in iris:
- `detail::malloc_fine_grained()` — allocates the heap with `hipExtMallocWithFlags`
- `iris::iris()` constructor — allgathers IPC handles, opens remote heaps
- `iris_device_view::translate()` — computes remote pointer from local pointer + heap bases
- `iris_device_view::load()` — dereferences translated pointer

---

## 9. What was NOT the problem (ruled out)

| Hypothesis | Status | Evidence |
|---|---|---|
| Signed-char `row_seg` overflow | RULED OUT | Code uses `int row_seg[GP_BM]`, confirmed by grep |
| Fine-grained alloc fails entirely | RULED OUT | `hipExtMallocWithFlags(0x1)` returns 0 in all 8 MPI workers with `HSA_XNACK=1` |
| IPC remote reads fail at host level | RULED OUT | `test_ipc.py`: all 8 ranks read correct values from neighbors via `hipMemcpy` through IPC handles |
| P2P peer access disabled | RULED OUT | `torch.cuda.can_device_access_peer(i,j)` returns True for all GPU pairs |
| Container lacks privileges | RULED OUT | `Privileged=true`, `IpcMode=host` confirmed via `docker inspect` |
| XNACK disabled (original suspicion) | PARTIALLY: was a prerequisite | Reloaded driver with `noretry=0`, set `HSA_XNACK=1` — fine-grained alloc works, but gather still fails |
| HSA_XNACK not reaching MPI workers | RULED OUT | `mpirun -x HSA_XNACK=1` confirmed via `os.environ` print in all 8 workers |
| Wrong HipKittens repo (ThunderKittens) | WAS A BUG, NOW FIXED | Initially cloned wrong repo; corrected to `HazyResearch/HipKittens` (cdna4 port) |
| `gl::operator[]` from host code | WAS A BUG, NOW FIXED | Fixed in `dispatch_grouped_gemm_b0` (host function). Gather kernel calls are from device — fine. |

---

## 10. Recommended next debugging steps for the agent

**Step 1 (most likely fix): Check and patch the iris flag value.**

Inside the failing container, check:
```bash
grep -n 'hipExtMallocWithFlags\|Finegrained\|fine_grained\|0x8\|0x1' \
  /tmp/HipKittens/distributed-kernels/build/_deps/iris-src/irisx/include/iris/iris.hpp
```
If `malloc_fine_grained` uses `hipDeviceMallocFinegrained` (the macro), check the macro value
in the container's ROCm headers:
```bash
grep 'hipDeviceMallocFinegrained' /opt/rocm/include/hip/hip_runtime_api.h
# Expected on this ROCm: #define hipDeviceMallocFinegrained 0x1
```
If the iris source hardcodes `0x8`, or if there's a version mismatch, patch iris and rebuild.

**Step 2: Check tile metadata for the first tile.**

Add a debug print to `example.py` before the GPU copy:
```python
if rank == CONSUMER:
    print(f"tile_arr[0] = {tile_arr[0]}")  # [seg_begin, seg_count, tile_dst0, valid_rows]
    print(f"seg_arr[0] = {seg_arr[0]}")    # [src_rank, src_row_begin, dst_row_begin, row_count, pad]
    print(f"Ntile={Ntile} Nseg={Nseg}")
```
Confirm `seg_count > 0` for the first tile and that `seg_arr` contains non-zero `row_count` values.
If `seg_count=0` everywhere, the metadata build is wrong and `tile_is_single_source()` is vacuously
returning false (no segments → not definitively single-source → Path 1 → Path 1 finds nothing →
writes zeros → but `packed_A_nonzero=True` contradicts this... so this is unlikely).

**Step 3: Write a minimal failing HIP kernel that tests the iris remote load.**

Write a standalone HIP kernel (no pybind, no full gather) that:
1. Accepts two `iris_device_view` structs and two pointers (`local_src`, `remote_dst`).
2. Has rank 0 write `42.0f` to its fine-grained buffer.
3. Has rank 1 read it via `ctx.load()` and write the result to a host-visible output.
4. Verify the output equals `42.0f`.

This isolates whether `ctx.load()` inside a `__global__` kernel works on this platform,
independent of the gather logic.

**Step 4: Try pinning the iris CPM fetch to a known-good commit.**

In `HipKittens/distributed-kernels/CMakeLists.txt`, change:
```cmake
GIT_TAG muhaawd/irisx
```
to the exact commit SHA that was used on the passing node. You can find it by checking the
CPM lock file or the `_deps/iris-src` git log on the passing node. Then rebuild and retest.

**Step 5: Try with `--mca btl tcp` instead of `btl self,vader`.**

This rules out the MPI shared-memory transport corrupting the IPC handle exchange:
```bash
mpirun -np 8 -x HSA_XNACK=1 --mca pml ob1 --mca btl self,tcp python3 example.py
```

---

## 11. Git context

```
Branch: subha/moe-dispatch-v0
Last commit with gather passing: 1dca7c0e  "B1-dispatch V0 PASSES all 5 routes"
Current HEAD:                    7ebdf127  "benchmarking: b2_aiter + b1_dispatch Level-1/Level-2"
```

The gather kernel (`gather_pack_kernel` in `kernel.cpp` and `ep8_gather.h`) is unchanged between
these commits. The additions between them are: `grouped_gemm_b0` GEMM path, `b0_tasks.py`,
`example.py` `SCHEDULE` knob, `b2_aiter.py`. None of these touch the gather.

---

## 12. Performance context (why fixing this matters)

With the gather working correctly, the Level-2 end-to-end comparison would be:

| pipeline | T_total (measured, gather unverified) |
|---|---|
| **ours: phase1 gather + phase2 grouped_b0** | **558 µs** |
| **production: aiter.fused_moe (sort+quant+fmoe)** | **1255 µs** |

The GEMM speedup (b0 vs microtk) of **10.1×** is verified and independent of the gather.
The **6.6× end-to-end speedup** is timing on corrupted data — real once the gather passes.
