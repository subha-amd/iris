# AGENT 06 REPORT — cache-on-first-touch / local tile inbox

Two concurrent kernels (`cft_producer` + `cft_consumer`) communicating through a **local HBM
symmetric-heap tile inbox** so each remote A tile crosses XGMI **exactly once**, while the consumer
keeps a **full grid** for latency hiding. Builds on V4 (a-stationary); V3/V4 untouched.

Files: `kernel.cpp`, `tile_inbox_abi.h`, `example.py`, `CACHE_FIRST_TOUCH.md`, this report.

---

## 1. Exact build command (node, gfx950 / CDNA4)

Mirror this dir under the node's distributed-kernels tree (auto-discovers `*/kernel.cpp`), then:

```bash
# on the node, in container r1_c4, GPU DISABLED for compile:
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels
  cp -r <repo>/irisx/cache_first_touch ./cache_first_touch    # keep node + repo identical
  cmake -B build_06 -DDK_BUILD=cache_first_touch -DIRIS_HIP_ARCHITECTURES=gfx950
  flock /tmp/mi355x_compile.lock -c "cmake --build build_06 -j8 --target cache_first_touch"'
```

Quick single-TU syntax check (no full cmake):
```bash
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
  cd <HK_ROOT>/distributed-kernels/cache_first_touch
  hipcc --offload-arch=gfx950 -std=c++20 -fsyntax-only \
    -I<HK_ROOT>/include -I<HK_ROOT>/include/iris \
    kernel.cpp'
```

**Compile status: [NEEDS-NODE].** Agent 06 verified SSH reachability to the node, but the
follow-up compile/inspect commands were blocked by the local shell permission policy in this run, so
no build was executed. The code targets the **exact** V4 toolchain (kittens.cuh + pyutils + iris,
gfx950) and reuses V4's proven gather/dequant/MFMA paths verbatim, so it is expected to compile; the
main agent should run the syntax check above to confirm.

## 2. Proposed GPU test command (MAIN AGENT ONLY, serialized)

```bash
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/cache_first_touch
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    M=1024 K=7168 N=2048 PROD_BLOCKS=16 mpirun --allow-run-as-root --mca pml ob1 \
      --mca btl self,vader -np 2 python3 example.py"'
```

### Launch ORDER + stream/event test plan (the load-bearing part)
- `dispatch_cft` (host, in kernel.cpp) creates **two non-blocking streams** and launches:
  1. `cft_producer` on `prod_stream` (grid = `PROD_BLOCKS`, modest → reserves CUs first),
  2. `cft_consumer` on `cons_stream` (FULL `(N/N_PER_BLOCK, M/BM)` grid) **immediately, no sync**,
  3. syncs both at the end.
- **Do NOT** insert `hipStreamSynchronize(prod_stream)` before the consumer launch — that serializes
  and defeats overlap (still correct, just slower; useful as an A/B sanity toggle).
- Per generation, the host resets `READY=0, CLAIM=0, CURSOR=0, INBOX=0` (done in `example.py
  reset_inbox()`), then bumps `GEN` so flags are written as `READY(gen)=(gen<<1)|1`.
- Suggested sweeps for the main agent: `PROD_BLOCKS ∈ {4,8,16,32}` (occupancy/latency trade);
  `M ∈ {256,512,1024}`, `N ∈ {2048}`, `K=7168`. Toggle serialized-vs-overlapped launch to measure
  the overlap benefit. Optionally insert a hipEvent on prod_stream and `hipStreamWaitEvent` to verify
  the (slower) serialized path still passes — confirms correctness is independent of overlap.

## 3. Expected output

```
==============================================================================
[CFT     ] M=1024 K=7168 N=2048  max_rel=...  RMS_rel=0.00xxx  local_A_zero=True  C_zero=False  -> PASSED
------------------------------------------------------------------------------
  CACHE-FIRST-TOUCH (2-kernel, 1x XGMI/tile):   XXX.XX us/iter   YY.YY TFLOP/s
==============================================================================
```

## 4. Correctness criteria

- `RMS_rel < 0.10` vs the bf16 reference `dequant(A_fp8) @ Bᵀ` (V4 achieves ~0.0033; expect similar).
- `local_A_zero == True`: rank1's local A is the zero **sentinel**, so a non-zero correct C **proves**
  A was gathered over XGMI by the producer (not read locally).
- `C_zero == False`.
- **XGMI-once invariant** (the point of this design): with `CLAIM`/`CURSOR`, no tile is gathered
  twice. Verify by having the producer `atomic_add` a per-task gather-count and asserting all counts
  == 1 (debug build), or compare measured XGMI bytes ≈ `num_m_tiles·num_k_tiles·BM·BK` (vs V4's
  `× N/N_PER_BLOCK`).

## 5. Assumptions

- HK pybind (`pyutils from_object`) builds a `gl` only from a torch.Tensor (or casts a scalar); it
  **cannot** bind a raw `int*`. Therefore `ready/claim/cursor` are passed as **int32 `gl` tensors**
  and the kernel extracts raw pointers via `&x[{0,0,0,0}]`. (Confirmed against
  `HipKittens/include/pyutils/pyutils.cuh`.)
- Producer + consumer both run on the **consumer rank**; only the producer touches the remote rank
  over IRIS. Same-rank cross-kernel HBM visibility uses `memory_scope_system` + release/acquire +
  a system fence before the flag store (canonical message_passing protocol).
- Inbox tiles are stored byte-for-byte in the **same ST_A swizzle** the consumer `load(frag,...)`
  expects, so producer→inbox→consumer is a plain memcpy with no re-layout.
- Symmetric allocation order is identical on both ranks (heap offsets match); inbox is allocated on
  the heap but read as **plain local memory** by the consumer (no XGMI).
- `IRIS::compare_exchange_strong` takes `expected` by reference (header confirmed) — used for CAS.

## 6. Known risks

- **Producer/consumer co-residency**: keep `PROD_BLOCKS ≤ CU count` so all producer blocks are
  schedulable; the two-launch design (separate streams) is what prevents the consumer grid from
  starving producers of CUs (see CACHE_FIRST_TOUCH.md §4). Correctness is guaranteed by the
  cursor+CAS drain; only liveness depends on this knob.
- **Spin cost**: consumer thread-0 acquire-spins per K-tile; fine for modest slot counts, but a long
  producer stall (e.g. too few PROD_BLOCKS) shows up as consumer spin time.
- **Compile not yet run on node** ([NEEDS-NODE]) — see §1.
- HBM inbox sizing: `num_slots·BM·BK·2` B (~117 MB at M1024/K7168) — within HBM but verify it does
  not pressure the live R1 server's VRAM (it lives on the bench's own IRIS heap, not the server's).

## 7. Files changed (new candidate dir only — no V3/V4 edits)

- `irisx/cache_first_touch/kernel.cpp`      — two-kernel producer/consumer + host `dispatch_cft`.
- `irisx/cache_first_touch/tile_inbox_abi.h` — flag/claim encoding, cache key, slot index, inbox desc.
- `irisx/cache_first_touch/example.py`        — np=2 driver scaffold (alloc inbox, reset+gen, launch).
- `irisx/cache_first_touch/CACHE_FIRST_TOUCH.md` — protocol, deadlock analysis, persistent sketch,
  storage discussion.
- `irisx/cache_first_touch/AGENT_REPORT.md`   — this file.

## 8. Static resource info

Not extracted ([NEEDS-NODE]; no compile in this run). Expected to be close to V4's consumer kernel
(it shares V4's MFMA/shared-tile structure: ~222 VGPR, ~2 waves/SIMD occupancy, ~160 B scratch); the
producer kernel is lighter (no MFMA, one shared A tile, gather+memcpy). Main agent should capture
`-Rpass-analysis=kernel-resource-usage` for both `cft_producer` and `cft_consumer` when it builds.
