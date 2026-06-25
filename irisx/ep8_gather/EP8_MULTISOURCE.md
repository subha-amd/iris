# EP8 multi-source remote A-tile gather

Generalizes V4's single-source A-stationary gather to the real EP8 MoE case, where one expert's
packed M-region was routed from up to 8 different source ranks (top-k=8, EP_SIZE=8). A single
`BM`-tall A-tile can therefore straddle a rank boundary. This component supplies a drop-in device
function (`gather_dequant_A_tile_multisource`) that V5 calls in place of V4's
`gather_dequant_A_tile<VEC>`, plus the per-tile setup it needs.

Files: `ep8_gather.h` (device fns, header-only, reusable by V5), `kernel.cpp` (standalone probe),
`example.py` (np=8 driver), `ep8_multisource_ref.py` (CPU reference + routing generator).

## 1. Metadata format (consumed from the shared ABI — AGENT_COMMON.md §3)

`route_segment` = one contiguous run of rows for one expert from one source rank:

```c
struct route_segment {
    int expert_id;       // local expert [0,32)
    int src_rank;        // owning rank of the rows [0,8)
    int src_row_begin;   // first row in src_rank's activation buffer
    int dst_row_begin;   // first row in the packed output region (ABSOLUTE packed space)
    int row_count;       // contiguous rows
};
```

ABI guarantees we rely on: segments are **disjoint** and **sorted by `dst_row_begin`**; each
segment's source rows are contiguous on its `src_rank`. Gaps between segments are **unrouted**
packed rows that must read back as exactly zero (zero-sentinel).

Per-tile, the kernel receives a `seg_tile_view`:

```c
struct seg_tile_view {
    const route_segment* segs;  // local (replicated) route_segment array
    int seg_begin, seg_count;   // == expert_task.segment_begin / segment_count
    int tile_dst0;              // absolute packed row of this tile's row 0
    int valid_rows;            // real rows in tile (<= BM); [valid_rows,BM) is tail
};
```

In V5 these fields come straight from `expert_task` + `expert_offsets`
(`tile_dst0 = expert_offsets[e] + m_tile_begin`). In the standalone probe they are passed in a
flat `tilemeta[Ntile,4]` int array, and `route_segment[]` in a flat `seg[Nseg,5]` int array.

## 2. The two paths

Path selection happens **once per M-tile**, before the K loop (cost amortized over all K-tiles).

### Path 2 — fast path (`tile_is_single_source`)
True when `seg_count == 1` AND that one segment covers `[0, valid_rows)` of the tile. Then the whole
tile maps to one `src_rank` with a constant `row_off = src_row_begin - dst_row_begin`:
`src_row = (tile_dst0 + r) + row_off`. The gather loop is **V4-identical** — one source rank, one
base row, no per-row branch. Plus a **local short-circuit**: if `src_rank == ctx.cur_rank()` the
fp8 uint4 and the fp32 scale are read by direct HBM deref (no `translate`, no XGMI). This is the
common case (most expert tiles are dominated by a single source).

### Path 1 — segment iterator (general / straddling tile)
Used when the tile spans >= 2 source ranks. `build_row_seg_map<BM>` fills a shared
`signed char row_seg[BM]` **once per M-tile**: for each tile-local dst row it stores the **absolute
segment index** owning that row, or `SEG_NONE (-1)` for tail/unrouted rows. Because segments are
disjoint+sorted, each segment stamps a disjoint contiguous span, so the whole block can fill the map
with **no write races** (two `__syncthreads()` bracket the fill). The per-element gather then does
**one lookup per row** -> `(src_rank, src_row)`, reading the segment's source metadata once; rows
mapped to `SEG_NONE` (and any `src_row >= Msrc` / K-overflow) load the **zero-sentinel** `uint4` so
they dequant to exactly 0. Straddling tiles are handled naturally — different rows resolve to
different segments/ranks within the same tile. The fp8 uint4 load + per-128 fp32 scale + `__HIP_E4M3`
dequant into the swizzled `ST_A` are reused verbatim from V4 (bit-identical LDS layout).

## 3. Memory-order semantics

The gather is **read-only** of remote activations. The happens-before that makes a producer rank's
written activation bytes visible to the consumer's `ctx.load` is the **host-side `iris.barrier()`**
(`hipDeviceSynchronize()` + `MPI_Barrier`) issued between the producers finishing their writes and
the consumer launching this kernel. That host barrier is a full system fence on every rank, so
inside the kernel a plain `ctx.load` (relaxed deref) is correct; **no per-load acquire is needed**.

For callers that want **in-launch** producer/consumer handoff (no host barrier between the remote
write and this read), `ep8_gather.h` provides optional system-scope helpers:

- `release_segment_ready(ctx, flags, consumer_rank, my_src_rank)` — producer issues
  `fence<system>(release)` after writing its rows, then `atomic_store<system>(release)` of a ready
  flag onto the consumer's heap.
- `acquire_segment_ready(ctx, flags, src_rank)` — consumer spins on
  `atomic_load<system>(acquire)` of that flag before any `ctx.load` from `src_rank`.

`memory_scope_system` is required because the flag and the data it guards must be coherent **across
GPUs over XGMI** (device scope would not order cross-GPU visibility). These helpers are optional and
unused in the default barrier-synchronized path.

## 4. Correctness criteria (probe)

- `D == D_ref` (dequantized gather), RMS-rel < 1e-2 (fp8 dequant error only).
- Every unrouted/tail packed row is **exactly 0** (per-rank zero-sentinel) — proves we never read
  stale/garbage and that masking is correct.
- At least one routed row came from a **remote** rank (`src_rank != consumer`) — proves the XGMI
  `ctx.load` path actually ran, not just the local short-circuit.
- The generated routing exercises **both** paths: some single-source tiles, some straddling tiles.
