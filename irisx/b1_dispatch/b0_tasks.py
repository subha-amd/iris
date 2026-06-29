#!/usr/bin/env python3
# b1_dispatch / b0_tasks.py
# ------------------------------------------------------------------------------------------------
# Host-side task-list builder for the B0-class phase-2 GEMM (grouped_gemm_b0 in kernel.cpp).
#
# The B0 8-wave body computes one 256x256 output tile per block, so its task tuple is simpler than
# the v5 micro_tk one (no NSUB / N-superblock): a flat (expert, m_tile, n_tile, expert_row_begin).
# This is ADDITIVE — it does NOT replace build_tasks.py (which still serves micro_tk). It reuses
# build_tasks.build_packed_layout so the packed-row prefix sum matches, but at BM=256.
#
# tasks_b0[i] = (local_expert, m_tile, n_tile, expert_row_begin)   int32, B0_TASK_W=4
#   local_expert     : expert e in [0,E)               -> B base row e*N
#   m_tile           : 256-row tile INDEX within expert e's padded region (0-based)
#   n_tile           : 256-col tile INDEX over N        (0 .. N/256 - 1)
#   expert_row_begin : expert e's first GLOBAL packed row (multiple of BM=256)
#
# Invariants (mirrors build_tasks): each expert's region padded to a multiple of BM=256 so a 256-row
# tile stays inside ONE expert (no cross-expert contamination). expert_row_begin is therefore a
# multiple of 256, which the kernel relies on for exact integer tile-base math (ERB/128, ERB/64).
# Padding rows are zero in the packed buffer (phase-1 zero-sentinel) -> MFMA to 0 into dead C rows.
# ------------------------------------------------------------------------------------------------
import numpy as np
from build_tasks import build_packed_layout, ceil_div

B0_TASK_W = 4   # keep in sync with kernel.cpp B0_TASK_W
B0_BM = 256     # keep in sync with kernel.cpp B0_BM (== the B0 body's BLOCK_SIZE_ROW)
B0_BN = 256     # the B0 body's BLOCK_SIZE_COL (one task == one 256-wide N tile)


def build_b0_tasks(rows_per_expert, N, BM=B0_BM, BN=B0_BN):
    """Flatten all experts into the flat tasks_b0[num_tasks][4] int32 array for grouped_gemm_b0.

    Returns (tasks[num_tasks,4] int32, expert_row_begin[E] int32, padded_rows[E] int32,
             total_padded_rows int).  Empty experts emit ZERO tasks.
    Tile order: expert-major, then m_tile, then n_tile.
    """
    assert N % BN == 0, f"N={N} must be a multiple of B0 tile width {BN}"
    E = len(rows_per_expert)
    begin, padded, total = build_packed_layout(rows_per_expert, BM)
    n_n_tiles = N // BN

    tasks = []
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e <= 0:
            continue  # empty expert -> no tasks
        n_m_tiles = ceil_div(m_e, BM)          # 256-row tiles (incl. the tail tile, padded to zero)
        for mt in range(n_m_tiles):
            for nt in range(n_n_tiles):
                tasks.append((e, mt, nt, begin[e]))

    if len(tasks) == 0:
        tasks_np = np.zeros((0, B0_TASK_W), dtype=np.int32)
    else:
        tasks_np = np.asarray(tasks, dtype=np.int32)
    return (tasks_np,
            np.asarray(begin, dtype=np.int32),
            np.asarray(padded, dtype=np.int32),
            int(total))


def _selftest():
    """Pure-CPU self-test of the B0 task-list invariants (NO GPU)."""
    import build_tasks as bt
    N, E = 2048, 32
    rng = np.random.default_rng(0)
    print(f"b0_tasks self-test: E={E} N={N} BM={B0_BM} BN={B0_BN}")
    for name, fn in bt.ROUTE_BUILDERS.items():
        rpe = fn(E, 8192, rng)
        tasks, begin, padded, total = build_b0_tasks(rpe, N)

        # Invariant 1: empty experts emit zero tasks.
        emitted = set(int(t[0]) for t in tasks)
        for e in range(E):
            if rpe[e] == 0:
                assert e not in emitted, f"{name}: empty expert {e} emitted tasks"

        # Invariant 2: expert_row_begin is 256-aligned + disjoint/monotonic padded regions.
        for e in range(E):
            assert begin[e] % B0_BM == 0, f"{name}: ERB[{e}]={begin[e]} not 256-aligned"
        for e in range(1, E):
            assert begin[e] == begin[e - 1] + padded[e - 1]

        # Invariant 3: every (m_tile,n_tile) lands inside its expert's padded region / N.
        n_n_tiles = N // B0_BN
        for t in tasks:
            e, mt, nt, erb = (int(x) for x in t)
            assert erb == begin[e]
            assert 0 <= mt < padded[e] // B0_BM, f"{name}: m_tile {mt} out of expert {e} pad {padded[e]}"
            assert 0 <= nt < n_n_tiles

        # Invariant 4: task count == sum_e ceil(M_e/256) * (N/256).
        expect = sum(ceil_div(int(m), B0_BM) * n_n_tiles for m in rpe if m > 0)
        assert tasks.shape[0] == expect, f"{name}: {tasks.shape[0]} != {expect}"

        print(f"  {name:<12} rows[0:6]={list(rpe[:6])}  num_tasks={tasks.shape[0]} "
              f"total_padded_rows={total}  OK")
    print("ALL B0 TASK INVARIANTS PASSED")


if __name__ == "__main__":
    _selftest()
