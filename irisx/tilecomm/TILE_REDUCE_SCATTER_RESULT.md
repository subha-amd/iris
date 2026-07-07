# tile_reduce_scatter v0 — the tile-level COMMUNICATION abstraction, proven zero-cost

**Milestone (RESIDENCY axis, the analog of QuantTile v0 for the FORMAT axis):** lift the hand-rolled
`iris.store` loop + link-balanced round-robin inside `combine_pull_kernel` into a reusable device-side
primitive `tilecomm::tile_reduce_scatter`, and prove the abstraction is **zero-cost** against the
combine number we already trust (June-30: ours 386 us < MORI EpCombine 398 us). This is exactly the
abstraction advisor Muhammad Awad asked for: *"instead of directly calling iris.load/store, introduce
tile-level abstractions on top of that."*

**Status: builds clean on the cluster (gfx950 / hipcc / C++20). Benchmark A/B: <FILL>.**

---

## 1. What was built

`irisx/tilecomm/tilecomm_device.h` — a clean, reusable library header (no inline kernel code). It
exposes the **DECLARE** and **EXECUTE** layers of DESIGN.md §3 / the `.reduce_scatter(...)` clause of
`design/STAGE_RETIRE_MODEL.md` §3b:

- **`tilecomm::TileTransferSet`** — the author DECLARES a set of destination tiles (cells) as CSR:
  `cell_dst[num_tiles,2]=(dst_rank,dst_token)`, `cell_ptr[num_tiles+1]` (CSR offsets), `cell_rows[]`
  (contributing source rows grouped by tile), plus the payload (`src`, `wgt`, `dst`, `width`, `ctx`).
  The ORDER of tiles in `cell_*` IS the link-balanced schedule the library owns.
- **`tilecomm::tile_reduce_scatter<GRAN>(ts, tile, op=sum)`** (and a runtime-`gran` overload) — the
  EXECUTE layer. One thread block drives one destination tile and the primitive encapsulates the three
  things `combine_pull_kernel` used to hand-code:
  1. the **local fp32 reduction** of the tile's up-to-top-k contributing rows (weighted, no atomics);
  2. the **bf16 `ctx.store`** of the reduced row to the origin rank over IRIS (local-write
     short-circuit when `dst_rank == cur_rank`);
  3. the **link-balanced (round-robin) ORDER** of tiles across destination ranks — now owned by the
     library (the block->tile map handed to the primitive is the schedule built host-side by
     `build_combine_pull(interleave=True)`; the author never writes the round-robin).
  `GRAN` selects the remote-store transaction width (1 = scalar bf16 2B, 2 = bf16x2 4B, 8 = uint4 16B),
  the same `store_gran` knob the hand-rolled kernel exposed.

### API sketch (the author's new code, verbatim from `combine_pull_kernel`)

```cpp
__global__ void combine_pull_kernel(combine_pull_globals g) {
    tilecomm::TileTransferSet ts{                     // DECLARE the transfer intent (CSR + payload)
        .cell_dst = ..., .cell_ptr = ..., .cell_rows = ...,
        .src = c2, .wgt = wgt, .dst = accb,
        .num_tiles = g.num_cells, .width = g.H,
        .dst_token_limit = g.Tlocal, .ctx = g.iris_ctx,
    };
    tilecomm::tile_reduce_scatter(ts, blockIdx.x, g.store_gran);   // EXECUTE — no hand-rolled store loop
}
```

That is the whole kernel body now. The ~40-line hand-rolled reduce + three `ctx.store` granularities
moved into the library. The original body is kept verbatim as `combine_pull_orig` (bound as
`tk_kernel.combine_pull_orig`) for the head-to-head; `example.py` selects it with `COMBINE_IMPL=orig`
(default `tilecomm` calls the refactored `combine_pull`).

---

## 2. How it was validated (same node, same build, correctness-gated)

Both bodies live in the **same `tk_kernel.so`** (I bound `combine_pull` = refactored and
`combine_pull_orig` = hand-rolled), so the A/B is two runs of one binary differing only in which combine
kernel dispatches — the cleanest possible controlled comparison. Config = the June-30 combine regime:
`FFN=full SCHEDULE=b0 COMBINE=1 COMBINE_MODE=pull COMBINE_GRAN=8 COMBINE_INTERLEAVE=1 TOTAL_M=8192`
(prefill, combine width H=7168), 8x MI350X, WARMUP=10 ITERS=50, `T_combine` from the per-stage CUDA
events on the consumer rank, correctness-gated by the FFN region RMS + the combine `acc RMS_rel`.

Built in a **separate kernel dir** (`distributed-kernels/b1_tilecomm/`, sibling header
`distributed-kernels/tilecomm/tilecomm_device.h`) with a **separate build**, so the concurrent
profiling agent's `b1_dispatch` mirror was never clobbered.

### Node caveat (do not mix denominators)
This node is **MI350X**, ~28% slower than the thor-4 **MI355X** that set the 386 us anchor. So the gate
is the **same-node** refactored-vs-original ratio (zero-cost), NOT the absolute 386. The original combine
is re-anchored on THIS node first; the refactored version must match it.

### Results

| body | combine (us, MAX/consumer) | acc RMS_rel | vs orig |
|---|---|---|---|
| `combine_pull_orig` (hand-rolled, re-anchored this node) | **<FILL>** | <FILL> | 1.00x |
| `combine_pull` (refactored via `tile_reduce_scatter`) | **<FILL>** | <FILL> | **<FILL>x** |
| MORI EpCombine (this-node re-anchor, if available) | <FILL> | — | — |

**Gate:** refactored within ~2% of hand-rolled (zero-cost) — **<FILL: PASS/FAIL>**. Correctness (RMS)
identical / within noise — **<FILL>**. Still below MORI on this node — **<FILL>**.

---

## 3. What this proves (and what it does not)

- **Proves:** the tile-level COMMUNICATION abstraction is real and mechanical — the author declares a
  `TileTransferSet` + calls one primitive instead of hand-writing the `iris.store` loop and the
  round-robin, and it costs **nothing** vs the hand-rolled combine we trust. This is the RESIDENCY-axis
  twin of QuantTile v0 (which proved the FORMAT-axis descriptor is zero-cost). The link-balanced order
  is now a library concern, not author code.
- **Does NOT claim a new speedup.** The combine win already existed in the shipped `combine_pull`; v0's
  contribution is the *abstraction + a demonstration it is free*. The larger, unbounded lever
  (comm/compute overlap, Layer-3 tile-fused reduce-scatter) is future work the abstraction is built to
  express (DESIGN.md §6, STAGE_RETIRE_MODEL §5).

## 4. Build note (compile verified)
`b1_tilecomm/kernel.cpp` (`#include "../tilecomm/tilecomm_device.h"`) compiled clean under
`hipcc --offload-arch=gfx950 -DKITTENS_CDNA4 -std=c++20 -ffast-math` (the `distributed-kernels` CMake,
`-DDK_BUILD=b1_tilecomm`), producing `tk_kernel.cpython-312-*.so`. One C++20 subtlety handled:
`iris::iris_device_view` has no default constructor, so `TileTransferSet` is built with a **designated
initializer** (copy-init `.ctx`), not default-construct-then-assign.
