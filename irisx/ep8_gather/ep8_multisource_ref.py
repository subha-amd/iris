#!/usr/bin/env python3
# ep8_multisource_ref.py
# ================================================================================================
# CPU reference + routing/segment generator for the EP8 multi-source gather.
#
# What it models: EP8 MoE token routing produces, on a CONSUMER rank, a packed activation region of
# Mpacked rows.  Each packed row originally lived on some SOURCE rank's activation buffer.  Contiguous
# runs of packed rows that share (src_rank, contiguous src rows) form a `route_segment`.  Some packed
# rows are intentionally UNROUTED (zero-sentinel): they must come back as exactly 0.
#
# This module is self-contained (numpy only) so it can be imported by example.py to:
#   (1) generate a random-but-valid routing -> route_segments + per-tile metadata,
#   (2) quantize each rank's activations exactly like V4 (per-128 group, amax/448),
#   (3) compute the EXPECTED dequantized gather output D_ref[Mpacked,K] (incl zero rows),
#   (4) verify the kernel's D against D_ref.
#
# Segment invariants (must match ep8_gather.h's assumptions):
#   * segments are disjoint and SORTED by dst_row_begin,
#   * each segment's src rows are contiguous on its src_rank,
#   * gaps between segments == unrouted (zero-sentinel) packed rows.
# ================================================================================================
import numpy as np

QGROUP = 128
FP8_MAX = 448.0


def quantize_v1(A, K):
    """V4/V1 quant: per-128-group scale = amax/448, q = round-to-fp8(x/scale).
    Returns (q_uint8 fp8 bytes [M,K], scale_f32 [M,NG], deq_f32 [M,K])."""
    import ml_dtypes  # fp8 e4m3 on CPU
    M = A.shape[0]
    NG = K // QGROUP
    Ag = A.reshape(M, NG, QGROUP).astype(np.float32)
    amax = np.abs(Ag).max(axis=2, keepdims=True)
    scale = np.clip(amax / FP8_MAX, 1e-12, None)            # [M,NG,1]
    q = (Ag / scale).astype(ml_dtypes.float8_e4m3)          # fp8 e4m3 (OCP)
    deq = (q.astype(np.float32) * scale).reshape(M, K)
    return (q.view(np.uint8).reshape(M, K),
            scale.reshape(M, NG).astype(np.float32),
            deq.astype(np.float32))


def make_routing(world, Msrc, Mpacked, BM, seed=0, unrouted_frac=0.15, max_seg_rows=40):
    """Generate a valid routing: a list of route_segments (disjoint, sorted by dst_row_begin) that
    cover SOME of [0,Mpacked); gaps are unrouted zero-sentinel rows.  Also returns per-tile metadata
    aligned to BM tiles.  Segments are deliberately built so that SOME M-tiles are single-source
    (fast path) and SOME straddle a rank boundary (segment-iterator path).

    Returns:
      segs : list of dicts {expert_id, src_rank, src_row_begin, dst_row_begin, row_count}
      tiles: list of dicts {seg_begin, seg_count, tile_dst0, valid_rows}  (one per BM tile)
    """
    rng = np.random.default_rng(seed)
    segs = []
    dst = 0
    # round-robin-ish source assignment so tiles straddle boundaries
    src_cursor = [0] * world
    while dst < Mpacked:
        if rng.random() < unrouted_frac:
            dst += int(rng.integers(1, 8))                  # an unrouted gap (zero rows)
            continue
        src_rank = int(rng.integers(0, world))
        rc = int(rng.integers(1, max_seg_rows + 1))
        rc = min(rc, Mpacked - dst)
        if src_cursor[src_rank] + rc > Msrc:
            rc = Msrc - src_cursor[src_rank]
        if rc <= 0:
            dst += 1
            continue
        segs.append(dict(expert_id=0, src_rank=src_rank,
                         src_row_begin=src_cursor[src_rank],
                         dst_row_begin=dst, row_count=rc))
        src_cursor[src_rank] += rc
        dst += rc

    # per-BM-tile metadata
    Ntile = (Mpacked + BM - 1) // BM
    tiles = []
    for t in range(Ntile):
        lo = t * BM
        hi = min(lo + BM, Mpacked)
        valid_rows = hi - lo
        # segments overlapping [lo,hi)
        idxs = [i for i, s in enumerate(segs)
                if s["dst_row_begin"] < hi and s["dst_row_begin"] + s["row_count"] > lo]
        if idxs:
            seg_begin = idxs[0]
            seg_count = idxs[-1] - idxs[0] + 1
        else:
            seg_begin, seg_count = 0, 0
        tiles.append(dict(seg_begin=seg_begin, seg_count=seg_count,
                          tile_dst0=lo, valid_rows=valid_rows))
    return segs, tiles


def reference_gather(segs, Mpacked, K, deq_per_rank):
    """Build expected dequantized gather output D_ref[Mpacked,K].
    deq_per_rank[r] = the dequantized activation buffer [Msrc,K] for rank r.
    Unrouted packed rows stay exactly 0 (zero-sentinel)."""
    D = np.zeros((Mpacked, K), dtype=np.float32)
    for s in segs:
        src = deq_per_rank[s["src_rank"]]
        sr = s["src_row_begin"]
        dr = s["dst_row_begin"]
        rc = s["row_count"]
        D[dr:dr + rc, :] = src[sr:sr + rc, :]
    return D


def segs_to_int_array(segs):
    """Flatten route_segment list to an int32 [Nseg,5] array in ABI field order:
    expert_id, src_rank, src_row_begin, dst_row_begin, row_count."""
    if not segs:
        return np.zeros((1, 5), dtype=np.int32)
    return np.array([[s["expert_id"], s["src_rank"], s["src_row_begin"],
                      s["dst_row_begin"], s["row_count"]] for s in segs], dtype=np.int32)


def tiles_to_int_array(tiles):
    """Flatten per-tile metadata to int32 [Ntile,4]: seg_begin, seg_count, tile_dst0, valid_rows."""
    return np.array([[t["seg_begin"], t["seg_count"], t["tile_dst0"], t["valid_rows"]]
                     for t in tiles], dtype=np.int32)


if __name__ == "__main__":
    # tiny self-test of the generator + reference (no GPU)
    world, Msrc, Mpacked, BM, K = 8, 128, 256, 64, 256
    segs, tiles = make_routing(world, Msrc, Mpacked, BM, seed=3)
    covered = sum(s["row_count"] for s in segs)
    # check disjoint+sorted
    last = -1
    for s in segs:
        assert s["dst_row_begin"] >= last, "segments not sorted"
        last = s["dst_row_begin"] + s["row_count"]
    n_single = sum(1 for t in tiles if t["seg_count"] == 1)
    n_multi  = sum(1 for t in tiles if t["seg_count"] > 1)
    print(f"segments={len(segs)} covered_rows={covered}/{Mpacked} "
          f"single_source_tiles={n_single} multi_source_tiles={n_multi}")
    assert n_single > 0 and n_multi > 0, "want both paths exercised"
    print("ep8_multisource_ref self-test OK")
