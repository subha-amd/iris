# AGENT 03 — EP8 multi-source remote gather — REPORT

Deliverable: a drop-in, multi-source generalization of V4's single-rank A-tile gather, driven by the
shared `route_segment[]` ABI, so one expert's packed M-region can be gathered from up to 8 source
ranks. Header-only device fn reusable by V5, plus a standalone probe + np=8 driver + CPU reference.

## Files changed (new candidate dir `irisx/ep8_gather/`)
- `irisx/ep8_gather/ep8_gather.h` — device gather (both paths) + optional system-scope release/acquire.
- `irisx/ep8_gather/kernel.cpp` — standalone probe exercising the gather; pybind `dispatch_gather`.
- `irisx/ep8_gather/example.py` — np=8 (EP8) driver, per-rank zero-sentinel, CPU check.
- `irisx/ep8_gather/ep8_multisource_ref.py` — CPU reference + routing/segment generator.
- `irisx/ep8_gather/EP8_MULTISOURCE.md` — metadata format, memory-order semantics, the two paths.
- `AGENT_REPORT.md` — this file.

No edits to Agent 02's files or to V3/V4 (coordinate by ABI only).

## Build command (node, compile-only, MAIN AGENT or subagent under locks)
Mirror the dir under `<HK_ROOT>/distributed-kernels/ep8_gather/` (build auto-discovers `*/kernel.cpp`).
```
export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels
  cmake -B build_03 -DDK_BUILD=ep8_gather -DIRIS_HIP_ARCHITECTURES=gfx950
  flock /tmp/mi355x_compile.lock -c "cmake --build build_03 -j8 --target ep8_gather"'
```
Single-TU syntax check alternative:
`hipcc --offload-arch=gfx950 -fsyntax-only -I<HK>/include -I<iris>/include irisx/ep8_gather/kernel.cpp`

### Compile status: [NEEDS-NODE]
I could NOT compile. Bash execution was denied in this environment AND the GPU/compile rule forbids
me from running device tests; node SSH is gated. So the gather was written against the verified APIs
(IRIS `iris.hpp` device view: `translate`/`load`/`atomic_load`/`atomic_store`/`fence` with
`memory_order`+`memory_scope`, `cur_rank()`; V4 `gather_dequant_A_tile<16>` swizzle/sub-tile math)
but has NOT been through hipcc. Main agent should compile first and report any HK `gl{}`/`st_bf`
template mismatches.

## Proposed GPU test command (MAIN AGENT ONLY, serialized under flock)
```
docker exec r1_c4 bash -lc '
  cd <HK_ROOT>/distributed-kernels/ep8_gather
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  flock /tmp/mi355x_project_gpu.lock -c "
    MSRC=128 MPACKED=256 K=256 BM=64 \
      mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader \
      -np 8 python3 example.py"'
```
`--mca pml ob1 --mca btl self,vader` is mandatory (R1 server holds VRAM; IB registration would fail).

## Expected output (consumer rank 0)
```
[EP8 gather] Msrc=128 Mpacked=256 K=256 world=8 segs=N tiles=4
  fast-path tiles (single-source)  = >0
  segment-iterator tiles (straddle)= >0
  remote (XGMI) gathered rows      = >0
  RMS_rel=<~1e-3>  max_rel=<small>
  zero_sentinel_rows_exact_zero    = True
  routed_rows_nonzero              = True
  -> PASSED
```

## Correctness criteria
- `D == D_ref` (dequantized multi-source gather), RMS-rel < 1e-2 (fp8 dequant error only).
- Every unrouted/tail packed row EXACTLY zero (per-rank zero-sentinel) — `zero_sentinel_rows_exact_zero=True`.
- At least one routed row from a REMOTE rank — proves the XGMI `ctx.load` path executed.
- Routing exercises BOTH paths (single-source tiles AND straddling tiles both > 0).

## Path conditions (summary)
- Fast path (Path 2): `seg_count==1` and that segment covers `[0,valid_rows)`; constant
  `src_row = tile_dst0 + r + (src_row_begin - dst_row_begin)`; local short-circuit when
  `src_rank == cur_rank` (direct HBM, no translate/XGMI).
- Segment iterator (Path 1): otherwise; `build_row_seg_map<BM>` fills `signed char row_seg[BM]` once
  (disjoint+sorted spans, race-free), one lookup per row -> `(src_rank, src_row)`; `SEG_NONE`/overflow
  -> zero-sentinel uint4.

## Memory-order choices
- Default: read-only gather; the **host `iris.barrier()`** (`hipDeviceSync` + `MPI_Barrier`) between
  producer writes and consumer launch is the happens-before -> plain `ctx.load` (no per-load acquire).
- Optional in-launch handoff: `release_segment_ready` / `acquire_segment_ready` use IRIS
  `atomic_store`/`atomic_load` + `fence` at **`memory_scope_system`** (cross-GPU coherence over XGMI),
  release-after-write / acquire-before-read. Unused in the default path.

## Assumptions
- `route_segment` / `expert_task` / `expert_offsets` as in AGENT_COMMON.md §3 (Agent 00 may refine;
  consume-only). `seg_tile_view.tile_dst0 = expert_offsets[e] + m_tile_begin`.
- Segments disjoint + sorted by `dst_row_begin`; src rows contiguous per segment; gaps = zero rows.
- fp8 e4m3 OCP (gfx950, not fnuz) + per-128 fp32 scales, exactly as V4. bf16 dequant output.
- `BM <= 127` so a `signed char` segment index fits; current segments-per-tile easily < 127.
- IRIS symmetric heap: identical allocation ORDER on all ranks => identical offsets (driver enforces).

## Known risks
- [NEEDS-NODE] not compiled — possible HK template/signature drift (`st_bf` swizzle accessors,
  `gl{}` indexing of the int metadata arrays). The swizzle/sub-tile code is copied verbatim from V4
  so risk is mostly in the new int `gl` plumbing in `kernel.cpp`.
- `signed char` segment-index map caps **absolute** segment index at 127 within a tile's window;
  fine for the probe, but V5 should switch `row_seg` to a per-tile-relative `short` if global segment
  counts grow (use `seg_begin + local_index`, store local index). Flagged for V5 integration.
- Probe reads the tile back to HBM for checking (not fused); it validates the gather, not GEMM perf.
- If Agent 00 changes `route_segment` field order/types, regenerate `segs_to_int_array` + the
  `seg_tile_view` loads (ABI-only coupling, no code shared).
