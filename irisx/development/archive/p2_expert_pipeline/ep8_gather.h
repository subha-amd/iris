// ep8_gather.h  (P2 — trimmed self-contained subset of Agent 03's ep8_gather/ep8_gather.h)
// ================================================================================================
// P2 only needs the route_segment ABI + seg_tile_view + the single-source fast-path predicate from
// Agent 03's multi-source gather. The full gather_dequant_A_tile_multisource() is NOT used here
// (P2's producer does its own row-major copy-once gather into the expert slot), so it is omitted to
// keep this candidate dir buildable standalone. Semantics are IDENTICAL to the canonical header; if
// the canonical header changes, re-sync this subset.
//
// route_segment matches AGENT_COMMON.md §3 exactly. Segments are disjoint + sorted per ABI.
// ================================================================================================
#pragma once
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>

namespace ep8_gather {

using fp8_t = __hip_fp8_storage_t;
static constexpr int   QGROUP   = 128;
static constexpr signed char SEG_NONE = -1;

// One contiguous run of rows for one expert coming from one source rank (AGENT_COMMON §3).
struct route_segment {
    int expert_id;
    int src_rank;
    int src_row_begin;
    int dst_row_begin;
    int row_count;
};

// Kernel-side view of one M-tile's slice of route_segment[].
//   tile_dst0 here is SLOT-LOCAL (expert region starts at row 0 inside its slot), and the segments'
//   dst_row_begin are likewise expressed slot-local (host builds expert_meta+segs that way for P2).
struct seg_tile_view {
    const route_segment* segs;
    int seg_begin;
    int seg_count;
    int tile_dst0;
    int valid_rows;
};

__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// True iff exactly one segment covers EVERY valid row of the tile (V4-identical fast path).
// out_row_off: src_row = (tile_dst0 + r) + out_row_off.
__device__ __forceinline__ bool tile_is_single_source(
        const seg_tile_view& v, int* out_src_rank, int* out_row_off) {
    if (v.seg_count != 1) return false;
    const route_segment s = v.segs[v.seg_begin];
    const int seg_lo = s.dst_row_begin - v.tile_dst0;
    const int seg_hi = seg_lo + s.row_count;
    if (seg_lo > 0 || seg_hi < v.valid_rows) return false;
    *out_src_rank = s.src_rank;
    *out_row_off  = s.src_row_begin - s.dst_row_begin;
    return true;
}

}  // namespace ep8_gather
