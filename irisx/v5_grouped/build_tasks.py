#!/usr/bin/env python3
# fmoe_fused_v5_grouped / build_tasks.py
# ------------------------------------------------------------------------------------------------
# Host-side GROUPED scheduler support for V5.
#
# V4 ran ONE expert per launch: grid = (N/N_PER_BLOCK, M/BM), a fixed M and N.  A real MoE has
# E=32 local experts with WILDLY uneven rows-per-expert (M_e), so launching one V4 grid per expert
# either (a) serializes 32 tiny launches, or (b) pads every expert to a common M and wastes work on
# empty/near-empty experts.  V5 fuses all experts into ONE grid by flattening their tiles into a
# flat task list and launching grid.x = num_tasks blocks.
#
# This module is pure host Python (numpy only) so it can be unit-tested with NO GPU.  It produces:
#   1. expert_offsets[E+1], rows_per_expert[E]   -- the packed-row prefix sum (padded to BM).
#   2. tasks[num_tasks][6] int32                 -- the flat work list consumed by the kernel.
#   3. the adaptive NSUB choice for the given route distribution.
#
# tasks[i] = (local_expert, m_tile_begin, valid_rows, n_superblock, nsub, expert_row_begin)
#   local_expert     : which expert e in [0,E)            -> selects B base row e*N
#   m_tile_begin     : row offset WITHIN expert e's padded region, multiple of BM
#   valid_rows       : <= BM real rows in this M-tile (tail mask; padding rows read zero in gather)
#   n_superblock     : which NSUB-wide N panel (0 .. ceil(N/(NSUB*BN))-1)
#   nsub             : NSUB used for THIS launch (same for all tasks; carried for the kernel/ABI)
#   expert_row_begin : expert e's first row in the GLOBAL padded packed-A/C space (= padded prefix)
#
# NO-CONTAMINATION INVARIANT (host side): each expert's region in the packed A/C space is padded UP
# to a multiple of BM.  So m_tile_begin .. m_tile_begin+BM always lies inside ONE expert's padded
# rows; a full-tile store can never spill into the next expert.  The gather still masks at the TRUE
# valid_rows (gr < expert_row_begin + valid_rows) so the BM-valid_rows padding rows read zero and
# contribute zero to C.  The padding C rows are written (zeros) but live in dead padding space that
# no consumer reads.
# ------------------------------------------------------------------------------------------------
import numpy as np

TASK_W = 6  # columns per task row (keep in sync with kernel.cpp expert_task / TASK_W)


def ceil_div(a, b):
    return (a + b - 1) // b


def build_packed_layout(rows_per_expert, BM):
    """Pad each expert's rows UP to a multiple of BM and prefix-sum into the packed space.

    Returns (expert_row_begin[E], padded_rows_per_expert[E], total_padded_rows).
    expert_row_begin[e] is expert e's first GLOBAL row in the padded packed-A/C buffer.
    """
    E = len(rows_per_expert)
    padded = [ceil_div(int(m), BM) * BM if m > 0 else 0 for m in rows_per_expert]
    begin = [0] * E
    acc = 0
    for e in range(E):
        begin[e] = acc
        acc += padded[e]
    return begin, padded, acc


def aggregate_blocks(rows_per_expert, N, BM, BN, nsub):
    """Number of runnable (M-tile x N-superblock) blocks across ALL experts for a given NSUB.

    aggregate_blocks = sum_e ceil(M_e/BM) * ceil(N/(nsub*BN))
    Empty experts (M_e==0) contribute zero blocks (zero tasks).
    """
    n_super = ceil_div(N, nsub * BN)
    total = 0
    for m in rows_per_expert:
        if m > 0:
            total += ceil_div(int(m), BM) * n_super
    return total


def choose_nsub(rows_per_expert, N, BM, BN,
                candidates=(8, 4, 2, 1), min_blocks=256, prefer_blocks=512):
    """Adaptive NSUB selector (host).

    Pick the LARGEST NSUB in `candidates` that still leaves >= min_blocks runnable blocks
    (so the 256-CU grid is saturated and IRIS gather latency is hidden), preferring a config
    with >= prefer_blocks blocks when one exists.  Larger NSUB = more A-reuse (less redundant
    cross-GPU gather) but FEWER blocks; small/skewed routes need smaller NSUB to keep the grid full.

    Returns (nsub, n_blocks_for_that_nsub).  Falls back to the smallest candidate (most blocks)
    if even that can't reach min_blocks (a genuinely tiny route -- nothing better is possible).
    """
    scored = []  # (nsub, blocks)
    for nsub in sorted(candidates, reverse=True):
        blocks = aggregate_blocks(rows_per_expert, N, BM, BN, nsub)
        scored.append((nsub, blocks))

    # First choice: largest NSUB reaching prefer_blocks.
    for nsub, blocks in scored:  # already descending NSUB
        if blocks >= prefer_blocks:
            return nsub, blocks
    # Second choice: largest NSUB reaching min_blocks.
    for nsub, blocks in scored:
        if blocks >= min_blocks:
            return nsub, blocks
    # Fallback: smallest NSUB (the most blocks we can possibly make).
    nsub, blocks = min(scored, key=lambda x: x[0])
    return nsub, blocks


def build_tasks(rows_per_expert, N, BM, BN, nsub):
    """Flatten all experts into the flat tasks[num_tasks][6] int32 array.

    Empty experts emit ZERO tasks.  Tile order: expert-major, then M-tile, then N-superblock.
    """
    E = len(rows_per_expert)
    begin, padded, total_padded = build_packed_layout(rows_per_expert, BM)
    n_super = ceil_div(N, nsub * BN)

    tasks = []
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e <= 0:
            continue  # empty expert -> no tasks (no contamination, no wasted blocks)
        n_m_tiles = ceil_div(m_e, BM)
        for mt in range(n_m_tiles):
            m_tile_begin = mt * BM
            valid_rows = min(BM, m_e - m_tile_begin)   # tail mask for last tile
            for ns in range(n_super):
                tasks.append((e, m_tile_begin, valid_rows, ns, nsub, begin[e]))

    if len(tasks) == 0:
        tasks_np = np.zeros((0, TASK_W), dtype=np.int32)
    else:
        tasks_np = np.asarray(tasks, dtype=np.int32)
    return tasks_np, begin, padded, total_padded


def build_grouped_schedule(rows_per_expert, N, BM, BN,
                           candidates=(8, 4, 2, 1), min_blocks=256, prefer_blocks=512):
    """One-call entry point: choose NSUB then build the task list + packed layout.

    Returns a dict with everything the driver needs:
      nsub, n_super, tasks[num_tasks][6], expert_row_begin[E], padded_rows[E],
      total_padded_rows, num_tasks, n_blocks.
    """
    nsub, n_blocks = choose_nsub(rows_per_expert, N, BM, BN,
                                 candidates=candidates,
                                 min_blocks=min_blocks, prefer_blocks=prefer_blocks)
    tasks, begin, padded, total = build_tasks(rows_per_expert, N, BM, BN, nsub)
    return {
        "nsub": nsub,
        "n_super": ceil_div(N, nsub * BN),
        "tasks": tasks,
        "expert_row_begin": np.asarray(begin, dtype=np.int32),
        "padded_rows": np.asarray(padded, dtype=np.int32),
        "total_padded_rows": int(total),
        "num_tasks": int(tasks.shape[0]),
        "n_blocks": int(n_blocks),
    }


# ------------------------------------------------------------------------------------------------
# Synthetic route distributions for the serial test matrix (np=2 first).  Each returns
# rows_per_expert[E] (an int array summing to ~total_rows).  Used by example.py and self-test.
# ------------------------------------------------------------------------------------------------
def route_uniform(E, total_rows, rng):
    base = total_rows // E
    rpe = np.full(E, base, dtype=np.int64)
    rpe[: total_rows - base * E] += 1   # spread remainder
    return rpe


def route_zipf(E, total_rows, rng, a=1.2):
    w = 1.0 / np.power(np.arange(1, E + 1), a)
    w = w / w.sum()
    rpe = np.floor(w * total_rows).astype(np.int64)
    rpe[0] += total_rows - rpe.sum()    # dump remainder on the hottest expert
    return np.maximum(rpe, 0)


def route_one_hot(E, total_rows, rng):
    rpe = np.zeros(E, dtype=np.int64)
    rpe[0] = total_rows                 # ALL rows to one expert (worst skew)
    return rpe


def route_several_hot(E, total_rows, rng, hot=4):
    rpe = np.zeros(E, dtype=np.int64)
    per = total_rows // hot
    for i in range(hot):
        rpe[i] = per
    rpe[0] += total_rows - rpe.sum()
    return rpe


def route_many_empty(E, total_rows, rng, active=6):
    rpe = np.zeros(E, dtype=np.int64)
    per = total_rows // active
    for i in range(active):
        rpe[i] = per
    rpe[0] += total_rows - rpe.sum()    # rest of the E-active experts stay empty (0 tasks)
    return rpe


ROUTE_BUILDERS = {
    "uniform": route_uniform,
    "zipf": route_zipf,
    "one_hot": route_one_hot,
    "several_hot": route_several_hot,
    "many_empty": route_many_empty,
}


def _selftest():
    """Pure-CPU self-test of the scheduler invariants (NO GPU)."""
    BM, BN, N, E = 64, 64, 2048, 32
    rng = np.random.default_rng(0)
    print(f"build_tasks self-test: E={E} N={N} BM={BM} BN={BN}")
    for name, fn in ROUTE_BUILDERS.items():
        rpe = fn(E, 8192, rng)
        sched = build_grouped_schedule(rpe, N, BM, BN)
        tasks = sched["tasks"]
        begin = sched["expert_row_begin"]
        padded = sched["padded_rows"]
        nsub = sched["nsub"]
        n_super = sched["n_super"]

        # Invariant 1: empty experts emit zero tasks.
        emitted = set(int(t[0]) for t in tasks)
        for e in range(E):
            if rpe[e] == 0:
                assert e not in emitted, f"{name}: empty expert {e} emitted tasks"

        # Invariant 2: every task's BM-tile lands inside its expert's padded region (no spill).
        for t in tasks:
            e, m0, valid, ns, k_nsub, erb = (int(x) for x in t)
            assert k_nsub == nsub
            assert erb == begin[e]
            assert m0 % BM == 0
            assert m0 + BM <= padded[e], f"{name}: tile {m0}+{BM} spills expert {e} pad {padded[e]}"
            assert 0 < valid <= BM
            assert 0 <= ns < n_super

        # Invariant 3: task count == aggregate_blocks for the chosen NSUB.
        assert sched["num_tasks"] == aggregate_blocks(rpe, N, BM, BN, nsub)
        # Invariant 4: padded regions are disjoint + monotonic.
        for e in range(1, E):
            assert begin[e] == begin[e - 1] + padded[e - 1]

        print(f"  {name:<12} rows[0:6]={list(rpe[:6])}  NSUB={nsub} "
              f"n_super={n_super} num_tasks={sched['num_tasks']} "
              f"total_padded_rows={sched['total_padded_rows']}  OK")
    print("ALL INVARIANTS PASSED")


if __name__ == "__main__":
    _selftest()
