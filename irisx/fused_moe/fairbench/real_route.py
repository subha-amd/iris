#!/usr/bin/env python3
# fairbench/real_route.py
# ================================================================================================
# Build gather_pack's routing plan (SEG / TILE / rowmap) from a REAL top-k router's output, so the
# fused gather is fed exactly the routing an actual MoE layer produces — not the synthetic
# contiguous-run route of `b1_dispatch_route.build_multisource_route`.
#
# WHY THIS EXISTS
# ---------------
# `build_multisource_route` advances a monotone per-rank cursor and emits contiguous runs of 1..40
# source rows. That has two effects the real router does NOT reproduce:
#   1. `tile_is_single_source()` (gather_pack's Path 2 fast path) fires for many tiles.
#   2. every (src_rank, src_token) is referenced AT MOST ONCE  ==>  effectively top-1 routing.
# With real top-8 routing over 256 global experts, runs collapse to length ~1 and a token that hits
# two experts on the same destination rank IS pulled twice. Both matter for the honest number.
#
# THE MAPPING
# -----------
# Global expert `g` lives on rank `g // Eloc`, as local expert `g % Eloc`.
# For destination rank `d`, the routed rows are every (src_rank, src_token, topk_slot) triple whose
# global expert satisfies `g // Eloc == d`. One triple == one packed row.
#
# PACKED ROW ORDER (within one local expert): sorted by (src_rank, src_token). This is deterministic
# and maximises route_segment run length, i.e. it gives gather_pack the BEST case it can get from a
# real router — so any remaining Path-1 cost is intrinsic, not an artefact of our ordering.
#
# OUTPUTS (all for ONE destination rank `d`):
#   rows_per_expert[Eloc]         int32  routed rows per local expert
#   expert_row_begin[Eloc]        int32  BM-padded prefix (BM=256, matches b0_tasks/build_packed_layout)
#   Mpacked                       int    total padded packed rows
#   packed_src[Mpacked, 3]        int32  (src_rank, src_token, topk_slot); (-1,-1,-1) for padding
#   segs[Nseg, 5]                 int32  (expert_id, src_rank, src_row_begin, dst_row_begin, row_count)
#   tiles[Ntile, 4]               int32  (seg_begin, seg_count, tile_dst0, valid_rows)  GP_BM=64 tiles
#   rowmap[Mpacked, 2]            int32  (src_rank, src_row); (-1,-1) for padding  <- gather_pack_rowmap
#   topk_weights_packed[Mpacked]  f32    the gate weight of each packed row (for combine)
# ================================================================================================
import numpy as np

GP_BM = 64     # gather_pack tile height (kernel.cpp GP_BM)
PACK_BM = 256  # per-expert packed-region padding (b0_tasks.B0_BM)


def build_packed_layout(rows_per_expert, BM=PACK_BM):
    """Per-expert BM-padded prefix. Mirrors build_tasks.build_packed_layout."""
    E = len(rows_per_expert)
    padded = np.array([int(np.ceil(m / BM)) * BM for m in rows_per_expert], dtype=np.int64)
    begin = np.zeros(E, dtype=np.int64)
    begin[1:] = np.cumsum(padded)[:-1]
    return begin, padded, int(padded.sum())


def route_for_rank(all_topk_ids, dst_rank, world, e_global, pack_bm=PACK_BM, gp_bm=GP_BM,
                   all_topk_weights=None):
    """Build the full routing plan that destination rank `dst_rank` needs.

    all_topk_ids : [world, T, TOPK] int  -- every rank's router output (global expert ids)
    Returns a dict (see module docstring).
    """
    all_topk_ids = np.asarray(all_topk_ids)
    W, T, TOPK = all_topk_ids.shape
    assert W == world
    eloc = e_global // world
    lo, hi = eloc * dst_rank, eloc * (dst_rank + 1)

    # ---- 1. collect every (local_expert, src_rank, src_token, slot) triple routed here ----------
    sr, st, sl = np.nonzero((all_topk_ids >= lo) & (all_topk_ids < hi))
    le = all_topk_ids[sr, st, sl] - lo                    # local expert id
    # order: expert-major, then (src_rank, src_token) -> longest possible runs
    order = np.lexsort((sl, st, sr, le))
    le, sr, st, sl = le[order], sr[order], st[order], sl[order]

    rows_per_expert = np.bincount(le, minlength=eloc).astype(np.int32)
    begin, padded, mpacked = build_packed_layout(rows_per_expert, pack_bm)

    # ---- 2. scatter the triples into their expert's packed region --------------------------------
    packed_src = np.full((mpacked, 3), -1, dtype=np.int32)
    # dst row of triple i = begin[le[i]] + (rank of i within its expert)
    within = np.arange(len(le), dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(rows_per_expert)[:-1])), rows_per_expert)
    dst = begin[le] + within
    packed_src[dst, 0] = sr
    packed_src[dst, 1] = st
    packed_src[dst, 2] = sl

    wgt = np.zeros(mpacked, dtype=np.float32)
    if all_topk_weights is not None:
        wgt[dst] = np.asarray(all_topk_weights)[sr, st, sl]

    # ---- 3. route_segments = maximal runs of (same expert, same src_rank, consecutive src_token,
    #         consecutive dst row). Reduces to length-1 runs under a real router (that is the point).
    segs = []
    n = len(le)
    i = 0
    while i < n:
        j = i + 1
        while (j < n and le[j] == le[i] and sr[j] == sr[i]
               and st[j] == st[i] + (j - i) and dst[j] == dst[i] + (j - i)):
            j += 1
        segs.append((int(le[i]), int(sr[i]), int(st[i]), int(dst[i]), j - i))
        i = j
    segs_np = (np.asarray(segs, dtype=np.int32) if segs
               else np.zeros((1, 5), dtype=np.int32))

    # ---- 4. per-GP_BM-tile metadata (seg_begin, seg_count, tile_dst0, valid_rows) -----------------
    #  segs are sorted by dst_row_begin (expert-major, then within-expert) -> the ABI's requirement.
    ntile = (mpacked + gp_bm - 1) // gp_bm
    tiles = np.zeros((ntile, 4), dtype=np.int32)
    if segs:
        seg_lo = segs_np[:, 3]                      # dst_row_begin
        seg_hi = seg_lo + segs_np[:, 4]             # one-past-last
        for t in range(ntile):
            a, b = t * gp_bm, min((t + 1) * gp_bm, mpacked)
            idx = np.nonzero((seg_lo < b) & (seg_hi > a))[0]
            if len(idx):
                tiles[t] = (idx[0], idx[-1] - idx[0] + 1, a, b - a)
            else:
                tiles[t] = (0, 0, a, b - a)
    else:
        for t in range(ntile):
            tiles[t] = (0, 0, t * gp_bm, min(gp_bm, mpacked - t * gp_bm))

    # ---- 5. flat rowmap (the natural production ABI for a pull gather) ---------------------------
    rowmap = packed_src[:, :2].copy()               # (src_rank, src_row); (-1,-1) = padding

    # ---- 6. honest stats -------------------------------------------------------------------------
    routed = int(len(le))
    distinct = len(set(zip(sr.tolist(), st.tolist())))  # tokens MORI's push would dedup to
    single_src_tiles = int(np.sum(tiles[:, 1] == 1))
    stats = dict(
        routed_rows=routed,
        distinct_src_tokens=distinct,
        dup_factor=routed / max(distinct, 1),        # pull re-reads this many x the deduped bytes
        n_segments=int(segs_np.shape[0]),
        mean_run=routed / max(len(segs), 1),
        n_tiles=int(ntile),
        single_source_tiles=single_src_tiles,
        max_seg_count=int(tiles[:, 1].max()) if ntile else 0,
        mpacked=int(mpacked),
        pad_frac=1.0 - routed / max(mpacked, 1),
    )
    return dict(rows_per_expert=rows_per_expert,
                expert_row_begin=begin.astype(np.int32),
                padded_rows=padded.astype(np.int32),
                Mpacked=int(mpacked),
                packed_src=packed_src,
                segs=segs_np,
                tiles=tiles,
                rowmap=rowmap,
                wgt=wgt,
                stats=stats)


# ================================================================================================
def _selftest():
    rng = np.random.default_rng(0)
    world, T, TOPK, E_GLOBAL = 8, 1024, 8, 256
    eloc = E_GLOBAL // world
    # a real-ish router: top-8 distinct experts per token
    all_ids = np.stack([np.stack([rng.choice(E_GLOBAL, TOPK, replace=False) for _ in range(T)])
                        for _ in range(world)]).astype(np.int32)
    all_w = rng.random((world, T, TOPK)).astype(np.float32)

    for d in range(world):
        r = route_for_rank(all_ids, d, world, E_GLOBAL, all_topk_weights=all_w)
        s, segs, tiles, pm = r["stats"], r["segs"], r["tiles"], r["packed_src"]
        beg, rpe, mp = r["expert_row_begin"], r["rows_per_expert"], r["Mpacked"]

        # I1: every triple routed to this rank appears exactly once
        want = int(((all_ids // eloc) == d).sum())
        assert s["routed_rows"] == want, f"d{d}: {s['routed_rows']} != {want}"
        assert (pm[:, 0] >= 0).sum() == want

        # I2: expert regions are 256-aligned, disjoint, and hold exactly rows_per_expert real rows
        for e in range(eloc):
            assert beg[e] % 256 == 0
            region = pm[beg[e]:beg[e] + rpe[e]]
            assert (region[:, 0] >= 0).all(), f"d{d} e{e}: hole inside real rows"
            pad = pm[beg[e] + rpe[e]: beg[e] + int(np.ceil(rpe[e] / 256) * 256)]
            assert (pad[:, 0] == -1).all(), f"d{d} e{e}: padding not unrouted"

        # I3: segments disjoint, sorted by dst, cover exactly the real rows, src rows contiguous
        last, cov = -1, 0
        for (e, sr, srb, drb, rc) in segs:
            assert drb >= last, f"d{d}: segs not sorted"
            last = drb + rc
            cov += rc
            for i in range(rc):
                assert tuple(pm[drb + i, :2]) == (sr, srb + i), f"d{d}: seg/packed_src disagree"
        assert cov == want, f"d{d}: segs cover {cov} != {want}"

        # I4: tilemeta really brackets the overlapping segments
        for t, (sb, sc, d0, vr) in enumerate(tiles):
            hits = [i for i, (e, sr, srb, drb, rc) in enumerate(segs)
                    if drb < d0 + vr and drb + rc > d0]
            if hits:
                assert (sb, sc) == (hits[0], hits[-1] - hits[0] + 1), f"d{d} t{t}: tilemeta wrong"
            else:
                assert sc == 0

        # I5: rowmap agrees with packed_src
        assert (r["rowmap"] == pm[:, :2]).all()

        if d == 0:
            print(f"  rank{d}: routed={s['routed_rows']} distinct={s['distinct_src_tokens']} "
                  f"dup={s['dup_factor']:.3f} segs={s['n_segments']} mean_run={s['mean_run']:.2f} "
                  f"tiles={s['n_tiles']} single_src={s['single_source_tiles']} "
                  f"max_seg_count={s['max_seg_count']} Mpacked={mp} pad={s['pad_frac']*100:.1f}%")
    print("real_route self-test: ALL INVARIANTS PASSED (all 8 dst ranks)")

    # what the synthetic route claimed, for contrast
    print("\n  contrast — synthetic build_multisource_route (runs 1..40, monotone cursors):")
    print("    mean_run ~20.5, single_source_tiles > 0, dup_factor == 1.000 (top-1, not top-8)")


def combine_fanout(all_topk_ids, world, e_global):
    """How many DISTINCT expert-owner ranks does each origin token route to?

    This is the number of producer ranks that will hold fc2 rows for one (origin_rank, token) cell.

    `combine_pull_kernel` (kernel.cpp:1971 -> tilecomm_device.h:123/135/151) reduces a cell's rows
    LOCALLY and then does a plain `ctx.store` of ONE bf16 row to the origin rank -- **not** an atomic
    accumulate ("No atomics -- a private accumulator per (tile, element)", tilecomm_device.h:77).
    That is correct only when a cell's contributions all live on ONE producer rank. Whenever fanout > 1,
    several producer ranks store to the same `accb[token]` and all but one contribution is LOST.

    The synthetic route hides this twice over: (a) example.py runs the region on ONE rank, and
    (b) build_multisource_route references each (src_rank, src_token) at most once, so fanout == 1.
    """
    a = np.asarray(all_topk_ids)
    owner = a // (e_global // world)                    # [W,T,TOPK] -> owning rank of each expert
    fan = np.array([[len(set(owner[r, t])) for t in range(a.shape[1])] for r in range(a.shape[0])])
    return dict(mean=float(fan.mean()), max=int(fan.max()),
                frac_gt1=float((fan > 1).mean()), hist=np.bincount(fan.ravel(), minlength=world + 1))


if __name__ == "__main__":
    _selftest()

    print("\n  combine fanout (distinct expert-owner ranks per origin token):")
    rng = np.random.default_rng(0)
    world, T, TOPK, E_GLOBAL = 8, 1024, 8, 256
    all_ids = np.stack([np.stack([rng.choice(E_GLOBAL, TOPK, replace=False) for _ in range(T)])
                        for _ in range(world)]).astype(np.int32)
    f = combine_fanout(all_ids, world, E_GLOBAL)
    print(f"    mean={f['mean']:.3f}  max={f['max']}  fraction of tokens with fanout>1: "
          f"{f['frac_gt1']*100:.1f}%")
    print(f"    histogram over fanout 0..{world}: {list(f['hist'])}")
    print("    => combine_pull's plain ctx.store loses all but one producer rank's partial sum for")
    print(f"       {f['frac_gt1']*100:.1f}% of tokens under REAL top-8 routing. The synthetic route has fanout==1.")
