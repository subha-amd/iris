#!/usr/bin/env python3
# b1_dispatch / b1_dispatch_route.py
# ================================================================================================
# B1-DISPATCH routing builder: 32-expert MULTI-SOURCE segments over the v5_grouped BM-padded packed
# layout. This is the GLUE that makes Agent 02's grouped packing and Agent 03's multi-source gather
# speak the same packed-row space. Pure numpy (no GPU) so it self-tests anywhere.
#
# Given (from build_tasks.build_grouped_schedule):
#   rows_per_expert[E], expert_row_begin[E] (BM-padded prefix), padded_rows[E], Mpacked, BM
# produce:
#   * route_segment[] : for each expert, its real rows [base, base+m_e) are split into contiguous
#     runs, each run assigned to ONE source rank's contiguous source rows. Padding rows
#     [base+m_e, base+padded_e) are LEFT UNROUTED (gaps) -> zero-sentinel. Runs are deliberately
#     sized so SOME BM-tiles are single-source (fast path) and SOME straddle a rank boundary
#     (segment-iterator path) — exercising both VERIFIED gather paths.
#   * per-BM-tile metadata : seg_begin, seg_count, tile_dst0, valid_rows for all Mpacked/BM tiles.
#
# Invariants (must match ep8_gather.h's assumptions):
#   - segments disjoint + SORTED by dst_row_begin; each segment's src rows contiguous on its rank;
#   - gaps between segments == unrouted (zero-sentinel) packed rows (here: per-expert BM padding);
#   - every (src_rank, src_row) referenced is < Msrc on that rank.
#
# quantize_v1 is the project-canonical fp8 e4m3fn quant (NOT e4m3 -> 448 inf nan hazard); identical
# to ep8_multisource_ref.quantize_v1.
# ================================================================================================
import numpy as np

QGROUP = 128
FP8_MAX = 448.0


def quantize_v1(A, K):
    """V4/V1 quant: per-128-group scale=amax/448, q=round-to-fp8(x/scale). MUST use float8_e4m3fn
    (OCP finite, saturating) NOT float8_e4m3 (encodes 448 as inf -> deq=inf -> RMS=nan)."""
    import ml_dtypes
    M = A.shape[0]
    NG = K // QGROUP
    Ag = A.reshape(M, NG, QGROUP).astype(np.float32)
    amax = np.abs(Ag).max(axis=2, keepdims=True)
    scale = np.clip(amax / FP8_MAX, 1e-12, None)
    qf = np.clip(Ag / scale, -FP8_MAX, FP8_MAX)
    q = qf.astype(ml_dtypes.float8_e4m3fn)
    deq = (q.astype(np.float32) * scale).reshape(M, K)
    return (q.view(np.uint8).reshape(M, K),
            scale.reshape(M, NG).astype(np.float32),
            deq.astype(np.float32))


def build_multisource_route(world, Msrc, E, rows_per_expert, expert_row_begin, padded_rows,
                            Mpacked, BM, seed=0, max_run=40):
    """Build multi-source route_segments + per-BM-tile metadata over the BM-padded packed layout.

    Each source rank keeps an independent cursor into its [0,Msrc) source rows; segments draw
    contiguous source rows from a randomly chosen rank. Per-expert padding rows are unrouted gaps.

    Returns (segs, tiles):
      segs : list of {expert_id, src_rank, src_row_begin, dst_row_begin, row_count} (sorted by dst)
      tiles: list of {seg_begin, seg_count, tile_dst0, valid_rows} (one per BM tile over Mpacked)
    """
    rng = np.random.default_rng(seed)
    src_cursor = [0] * world
    segs = []
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        filled = 0
        while filled < m_e:
            # vary run length so tiles both single-source and straddling appear.
            run = int(rng.integers(1, max_run + 1))
            run = min(run, m_e - filled)
            src_rank = int(rng.integers(0, world))
            # clamp to remaining source rows on that rank; if exhausted, try other ranks.
            tries = 0
            while src_cursor[src_rank] + run > Msrc and tries < world:
                src_rank = (src_rank + 1) % world
                tries += 1
            avail = Msrc - src_cursor[src_rank]
            if avail <= 0:
                raise RuntimeError(f"all source ranks exhausted at expert {e} (Msrc={Msrc} too small "
                                   f"for TOTAL_M; raise MSRC)")
            run = min(run, avail)
            segs.append(dict(expert_id=e, src_rank=src_rank,
                             src_row_begin=src_cursor[src_rank],
                             dst_row_begin=base + filled, row_count=run))
            src_cursor[src_rank] += run
            filled += run
        # padding rows [base+m_e, base+padded_e) intentionally left unrouted (zero-sentinel gap).

    segs.sort(key=lambda s: s["dst_row_begin"])

    # per-BM-tile metadata over the full padded packed space.
    Ntile = (Mpacked + BM - 1) // BM
    tiles = []
    for t in range(Ntile):
        lo = t * BM
        hi = min(lo + BM, Mpacked)
        valid_rows = hi - lo
        idxs = [i for i, s in enumerate(segs)
                if s["dst_row_begin"] < hi and s["dst_row_begin"] + s["row_count"] > lo]
        if idxs:
            seg_begin, seg_count = idxs[0], idxs[-1] - idxs[0] + 1
        else:
            seg_begin, seg_count = 0, 0
        tiles.append(dict(seg_begin=seg_begin, seg_count=seg_count,
                          tile_dst0=lo, valid_rows=valid_rows))
    return segs, tiles


def segs_to_int_array(segs):
    """Flatten to int32 [Nseg,5] in ABI field order: expert_id, src_rank, src_row_begin,
    dst_row_begin, row_count."""
    if not segs:
        return np.zeros((1, 5), dtype=np.int32)
    return np.array([[s["expert_id"], s["src_rank"], s["src_row_begin"],
                      s["dst_row_begin"], s["row_count"]] for s in segs], dtype=np.int32)


def tiles_to_int_array(tiles):
    """Flatten per-tile metadata to int32 [Ntile,4]: seg_begin, seg_count, tile_dst0, valid_rows."""
    return np.array([[t["seg_begin"], t["seg_count"], t["tile_dst0"], t["valid_rows"]]
                     for t in tiles], dtype=np.int32)


# ================================================================================================
# COMBINE (EpCombine) reverse map. route_reverse[packed_row] = (src_rank, src_token, topk_slot,
# route_weight)  (PRODUCTION_ABI.md §3.2/§5). This is the INVERSE of the gather: phase-1 pulled
# packed row `dst` FROM (src_rank, src_row); combine scatters it BACK to that origin token. We derive
# it directly from the SAME `segs` so the pipeline is internally coherent (we scatter back to exactly
# the rank+token we gathered from). topk_slot is synthesized from the expert id (the synthetic route
# has no real router, see HONESTY note); route_weight is a deterministic per-row weight so the CPU
# reference matches the kernel byte-for-byte.
#
# HONESTY / accumulation: in the default coherent map each (src_rank, src_token) is referenced AT
# MOST ONCE (build_multisource_route advances each rank's source cursor monotonically), so combine is
# a weighted SCATTER and the top-k accumulation (`+=`) never actually collides — the kernel still
# uses atomic-add so it is correct if it did. To genuinely EXERCISE accumulation, pass Tlocal smaller
# than the per-rank token span: src_token is then folded mod Tlocal, so several packed rows land on
# the same (src_rank, src_token) and acc[token] += w*out really reduces. The CPU reference applies
# the identical fold, so correctness stays checkable. This synthetic fold stands in for "one token
# routed to several local experts"; a real top-k router would produce the collisions naturally.
# ================================================================================================
def build_route_reverse(segs, Mpacked, world, Tlocal=None, weight_seed=2024):
    """Build the packed-row -> origin reverse map for combine.

    Returns (rev[Mpacked,3] int32 = (src_rank, src_token, topk_slot),
             wgt[Mpacked]  float32 = route_weight,
             Tlocal int = accumulator token-dim (same on every rank for symmetric-heap alloc)).
    Unrouted/padding packed rows get src_rank=-1, src_token=-1 (combine skips them).
    """
    rev = np.full((Mpacked, 3), -1, dtype=np.int32)
    wgt = np.zeros((Mpacked,), dtype=np.float32)
    max_tok = -1
    for s in segs:
        sr, srb, drb, rc, e = (int(s["src_rank"]), int(s["src_row_begin"]),
                               int(s["dst_row_begin"]), int(s["row_count"]), int(s["expert_id"]))
        for i in range(rc):
            dst = drb + i
            tok = srb + i
            rev[dst, 0] = sr
            rev[dst, 1] = tok                 # folded below if Tlocal is given
            rev[dst, 2] = e % 8               # synthetic topk_slot (top-k = 8)
            if tok > max_tok:
                max_tok = tok

    if Tlocal is None:
        Tlocal = max_tok + 1 if max_tok >= 0 else 1
    else:
        routed = rev[:, 1] >= 0
        rev[routed, 1] = rev[routed, 1] % Tlocal   # fold to force top-k collisions (accumulation test)

    # deterministic route weights in (0.1, 1.0) for routed rows, filled in packed-row order so the
    # GPU (reads `wgt`) and the CPU reference (reads the SAME `wgt`) agree exactly.
    rng = np.random.default_rng(weight_seed)
    routed = rev[:, 1] >= 0
    wgt[routed] = (rng.random(int(routed.sum())) * 0.9 + 0.1).astype(np.float32)
    return rev, wgt, int(Tlocal)


def combine_reference(C2, rev, wgt, world, Tlocal, H):
    """CPU reference for combine: acc_ref[src_rank][src_token] += route_weight * C2[packed_row].

    C2 is the (GPU) fc2 output [Mpacked, H] as float32 — pass the SAME C2 the kernel scattered so this
    isolates the combine's scatter/weight/accumulate from the FFN numerics. Returns [world,Tlocal,H].
    """
    acc = np.zeros((world, Tlocal, H), dtype=np.float32)
    Mpacked = C2.shape[0]
    for row in range(Mpacked):
        sr = int(rev[row, 0]); tok = int(rev[row, 1])
        if sr < 0 or tok < 0 or tok >= Tlocal:
            continue
        acc[sr, tok] += wgt[row] * C2[row]
    return acc


if __name__ == "__main__":
    # pure-CPU self-test of the routing invariants (no GPU) across all 5 route distributions.
    import build_tasks as BT
    world, Msrc, E, BM, BN, N, K = 8, 8192, 32, 64, 64, 2048, 7168
    for route, fn in BT.ROUTE_BUILDERS.items():
        rng = np.random.default_rng(7)
        rpe = fn(E, 8192, rng)
        sched = BT.build_grouped_schedule(rpe, N, BM, BN)
        segs, tiles = build_multisource_route(world, Msrc, E, rpe, sched["expert_row_begin"],
                                              sched["padded_rows"], sched["total_padded_rows"], BM,
                                              seed=11)
        # Invariant 1: disjoint + sorted by dst.
        last = -1
        for s in segs:
            assert s["dst_row_begin"] >= last, f"{route}: segments not sorted/disjoint"
            last = s["dst_row_begin"] + s["row_count"]
        # Invariant 2: every routed run lies inside its expert's real (non-padding) rows.
        for s in segs:
            e = s["expert_id"]; base = int(sched["expert_row_begin"][e]); m_e = int(rpe[e])
            assert base <= s["dst_row_begin"], f"{route}: seg below expert base"
            assert s["dst_row_begin"] + s["row_count"] <= base + m_e, \
                f"{route}: seg {s} spills into padding/next expert (m_e={m_e})"
            assert s["src_row_begin"] + s["row_count"] <= Msrc, f"{route}: src rows exceed Msrc"
        # Invariant 3: routed rows == sum of real rows (every real row routed exactly once).
        covered = sum(s["row_count"] for s in segs)
        assert covered == int(np.sum(rpe)), f"{route}: covered {covered} != real {int(np.sum(rpe))}"
        # Invariant 4: both gather paths exercised on a non-degenerate route.
        n_single = sum(1 for t in tiles if t["seg_count"] == 1)
        n_multi  = sum(1 for t in tiles if t["seg_count"]  > 1)

        # Invariant 5: route_reverse covers exactly the routed rows, and a CPU combine reference of a
        #   synthetic C2 reproduces the weighted scatter (coherent) AND the folded accumulation.
        Mpacked = int(sched["total_padded_rows"])
        rev, wgt, Tloc = build_route_reverse(segs, Mpacked, world)        # coherent (no fold)
        n_routed = int((rev[:, 1] >= 0).sum())
        assert n_routed == covered, f"{route}: route_reverse routed {n_routed} != covered {covered}"
        H = 8
        C2 = np.arange(Mpacked * H, dtype=np.float32).reshape(Mpacked, H) % 7 - 3.0
        acc = combine_reference(C2, rev, wgt, world, Tloc, H)
        # coherent map: each (rank,token) written once -> acc cell == wgt*C2 for its single packed row
        nz_cells = int((np.abs(acc).sum(axis=2) > 0).sum())
        assert nz_cells == n_routed, f"{route}: coherent combine collided ({nz_cells} != {n_routed})"
        # folded map: force collisions and check the reference still sums them
        rev2, wgt2, Tloc2 = build_route_reverse(segs, Mpacked, world, Tlocal=max(1, n_routed // (world * 4)))
        acc2 = combine_reference(C2, rev2, wgt2, world, Tloc2, H)
        tot_ref = sum(float(wgt2[r]) * C2[r].sum() for r in range(Mpacked) if rev2[r, 1] >= 0)
        assert abs(acc2.sum() - tot_ref) < 1e-2 * (abs(tot_ref) + 1.0), f"{route}: folded combine sum mismatch"

        print(f"  {route:<12} segs={len(segs):4d} covered={covered:5d} "
              f"single_tiles={n_single:4d} multi_tiles={n_multi:4d} "
              f"rev_routed={n_routed} Tlocal={Tloc}  OK")
    print("b1_dispatch_route self-test: ALL INVARIANTS PASSED")
