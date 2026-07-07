#pragma once
// ================================================================================================
// tilecomm_device.h — the FIRST tile-level COMMUNICATION abstraction for the fused MoE region.
//
// This is the RESIDENCY-axis analog of QuantTile v0 (the FORMAT axis, quanttile_decode.h): a clean,
// reusable device-side library layer so a kernel author declares a tile-transfer INTENT instead of
// hand-writing the iris.load/store loop + the link-balanced round-robin by hand. It is the concrete
// v0 of the `stage`/`retire` model's `.reduce_scatter(...)` clause (design/STAGE_RETIRE_MODEL.md §3b)
// and the `tile_reduce_scatter` intent of DESIGN.md §3 (Layer 1 DECLARE / Layer 3 EXECUTE).
//
// WHAT IT ENCAPSULATES (exactly the three things combine_pull_kernel used to hand-code):
//   (1) the LOCAL fp32 reduction of a destination tile's up-to-top-k contributing rows (weighted),
//   (2) the bf16 ctx.store of the reduced row to the ORIGIN rank over IRIS (local-write short-circuit
//       when the destination is this rank),
//   (3) the LINK-BALANCED (round-robin) ORDER of tiles across destination ranks — the library now
//       OWNS that order: the block->tile map fed to this primitive IS the schedule (built host-side by
//       build_combine_pull(interleave=True) / tilesched.py). The author never chooses it.
//
// THE BAR (like QuantTile v0): this abstraction must be ZERO-COST. combine_pull_kernel is refactored
// to call tile_reduce_scatter<GRAN>(...) and the original body is kept verbatim as combine_pull_orig
// for a same-node, same-build A/B. The lowered code is instruction-identical to the hand-rolled body,
// so the refactor is expected to reproduce the June-30 combine number (386 us, < MORI EpCombine 398 us).
//
// DEPENDENCIES: expects the TU to have already included HipKittens' kittens.cuh (for kittens::bf16 /
// bf16_2 / base_types::convertor) and <iris/iris.hpp> (for iris::iris_device_view). All kittens names
// are fully qualified so this header needs no `using namespace kittens`.
//
// IRIS gotcha note: a data-dependent destination rank is SAFE on the C++ iris_device_view path — its
// store()/load() `translate()` is plain symmetric-heap pointer arithmetic (heap_bases_[rank] + offset),
// NOT the Triton `iris.store` that faults on a non-constexpr to_rank (that gotcha is the Python/Triton
// path, examples/07). combine_pull_kernel has always passed a runtime dst_rank straight to ctx.store.
// ================================================================================================

#include <iris/iris.hpp>
// NOTE: bf16 support (__hip_bfloat16 / __float2bfloat16 / kittens::bf16 / base_types::convertor) comes
// from kittens.cuh, which the including TU pulls in BEFORE this header (see kernel.cpp include order).
// We do not re-include a hip bf16 header (its path varies across ROCm versions) — the TU already has it.

namespace tilecomm {

// Reduction operator for the reduce-scatter. v0 implements `sum` (the MoE top-k combine).
enum class reduce_op { sum };

// ================================================================================================
// Layer 1 — DECLARE.  TileTransferSet: the kernel author fills this POD to declare a reduce-scatter
// of a set of destination TILES ("cells"). Each tile is a (dst_rank, dst_token) that GATHERS its
// contributing SOURCE rows (grouped in CSR form), REDUCES them (op) in fp32 locally weighted, and
// stores ONE reduced row of `width` bf16 elements to the destination rank's symmetric-heap buffer.
//
// CSR layout (built once on the host, NOT in the timed region — it is the transpose of the per-row
// reverse routing map, the same metadata MORI's EpCombine precomputes):
//   cell_dst [num_tiles, 2]  : (dst_rank, dst_token) per tile
//   cell_ptr [num_tiles+1]   : CSR offsets into cell_rows
//   cell_rows[total_rows]    : source-row indices, grouped by tile  (cell_rows[cell_ptr[t] .. cell_ptr[t+1]))
// The ORDER of tiles in cell_* IS the link-balanced schedule the library owns.
//
// CONSTRUCTION: this is an aggregate; construct it with a designated initializer (C++20). The `ctx`
// member (iris::iris_device_view) has no default constructor, so default-construct-then-assign will
// NOT compile — always brace-init all fields, e.g. `TileTransferSet ts{ .cell_dst=..., ..., .ctx=v };`.
// ================================================================================================
struct TileTransferSet {
    // --- destination map + CSR grouping of contributing source rows per tile ---
    const int* cell_dst;         // [num_tiles, 2] : (dst_rank, dst_token)
    const int* cell_ptr;         // [num_tiles+1]  : CSR offsets into cell_rows
    const int* cell_rows;        // [total_rows]   : source-row indices, grouped by tile
    // --- payload ---
    const kittens::bf16* src;    // [*, width] LOCAL source rows (the reduction inputs, e.g. fc2 output)
    const float*         wgt;    // [*] per-source-row weight (route weight); must be non-null
    kittens::bf16*       dst;    // [*, width] symmetric-heap destination base (the remote store target)
    int num_tiles;               // number of destination tiles (== num_cells)
    int width;                   // elements per tile row (== H)
    int dst_token_limit;         // dst_token guard (== Tlocal); tiles past it are skipped
    iris::iris_device_view ctx;  // IRIS RMA handle (transport backend)
};

// Local fp32 reduction (op = sum) of one output element `h` of a tile whose contributing source rows
// are cell_rows[lo..hi). No atomics — a private accumulator per (tile, element). Weighted by wgt[row].
// Mirrors combine_pull_kernel's combine_reduce_elem exactly (so the lowered code is identical).
__device__ __forceinline__ float
trs_reduce_elem(const int* rows, const kittens::bf16* src, const float* wgt,
                int lo, int hi, int width, int h) {
    float acc = 0.f;
    for (int r = lo; r < hi; ++r) {
        const int row = rows[r];
        acc += wgt[row] * (float)src[(size_t)row * width + h];
    }
    return acc;
}

// ================================================================================================
// Layer 3 — EXECUTE (bulk-synchronous lowering). One THREAD BLOCK drives one destination tile `tile`
// of the transfer set. The whole ctx.store loop + local reduce is now this ONE call; the author no
// longer hand-writes iris.store. GRAN selects the remote-store transaction width (the same knob the
// hand-rolled kernel exposed as store_gran): 1 = scalar bf16 (2B), 2 = bf16x2 (4B), 8 = uint4 (16B).
// op = sum only in v0. Threads of the block stride the `width` output elements.
// ================================================================================================
template <int GRAN>
__device__ __forceinline__ void
tile_reduce_scatter(const TileTransferSet& ts, int tile, reduce_op op = reduce_op::sum) {
    (void)op;  // sum only in v0 (asserted at the API level)
    if (tile >= ts.num_tiles) return;

    const int dst_rank  = ts.cell_dst[tile * 2 + 0];
    const int dst_token = ts.cell_dst[tile * 2 + 1];
    if (dst_rank < 0 || dst_token < 0 || dst_token >= ts.dst_token_limit) return;  // unrouted/padding

    const int lo = ts.cell_ptr[tile];
    const int hi = ts.cell_ptr[tile + 1];
    const int W  = ts.width;
    iris::iris_device_view ctx = ts.ctx;
    const bool local = (dst_rank == ctx.cur_rank());

    const int*           rows = ts.cell_rows;
    const kittens::bf16* src  = ts.src;
    const float*         wgt  = ts.wgt;
    kittens::bf16*       drow = ts.dst + (size_t)dst_token * W;

    if constexpr (GRAN == 1) {
        // scalar bf16 store (2B); W stores/block.
        for (int h = threadIdx.x; h < W; h += blockDim.x) {
            const kittens::bf16 v = __float2bfloat16(trs_reduce_elem(rows, src, wgt, lo, hi, W, h));
            kittens::bf16* d = drow + h;
            if (local) *d = v; else ctx.store<kittens::bf16>(d, v, dst_rank);
        }
    } else if constexpr (GRAN == 2) {
        // bf16x2 store (4B); W/2 stores/block.
        const int Wv = W >> 1;
        for (int hv = threadIdx.x; hv < Wv; hv += blockDim.x) {
            const int h = hv << 1;
            const float a0 = trs_reduce_elem(rows, src, wgt, lo, hi, W, h);
            const float a1 = trs_reduce_elem(rows, src, wgt, lo, hi, W, h + 1);
            const kittens::bf16_2 v =
                kittens::base_types::convertor<kittens::bf16_2, float2>::convert(make_float2(a0, a1));
            kittens::bf16_2* d = reinterpret_cast<kittens::bf16_2*>(drow + h);
            if (local) *d = v; else ctx.store<kittens::bf16_2>(d, v, dst_rank);
        }
    } else {
        // uint4 store (16B == 8 bf16); W/8 stores/block (fewest transactions).
        const int Wv = W >> 3;
        for (int hv = threadIdx.x; hv < Wv; hv += blockDim.x) {
            const int h = hv << 3;
            float acc[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] = trs_reduce_elem(rows, src, wgt, lo, hi, W, h + j);
            union { uint4 v; kittens::bf16_2 h2[4]; } out;
            out.h2[0] = kittens::base_types::convertor<kittens::bf16_2, float2>::convert(make_float2(acc[0], acc[1]));
            out.h2[1] = kittens::base_types::convertor<kittens::bf16_2, float2>::convert(make_float2(acc[2], acc[3]));
            out.h2[2] = kittens::base_types::convertor<kittens::bf16_2, float2>::convert(make_float2(acc[4], acc[5]));
            out.h2[3] = kittens::base_types::convertor<kittens::bf16_2, float2>::convert(make_float2(acc[6], acc[7]));
            uint4* d = reinterpret_cast<uint4*>(drow + h);
            if (local) *d = out.v; else ctx.store<uint4>(d, out.v, dst_rank);
        }
    }
}

// Runtime-GRAN convenience: a SINGLE call site for the kernel author when the store granularity is a
// host-chosen runtime value (store_gran). Dispatches to the compile-time specialization above.
__device__ __forceinline__ void
tile_reduce_scatter(const TileTransferSet& ts, int tile, int gran, reduce_op op = reduce_op::sum) {
    if (gran == 1)      tile_reduce_scatter<1>(ts, tile, op);
    else if (gran == 2) tile_reduce_scatter<2>(ts, tile, op);
    else                tile_reduce_scatter<8>(ts, tile, op);
}

}  // namespace tilecomm
