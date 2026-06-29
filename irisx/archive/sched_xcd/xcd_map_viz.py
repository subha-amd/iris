#!/usr/bin/env python3
# xcd_map_viz.py  —  CPU-only predictor of the XCD/CU assignment produced by kernel.cpp's
# xcd_map_block().  No GPU, no HIP; pure Python index math that mirrors the device code 1:1.
#
# Purpose: SHOW (before running on hardware) that the XCD remap co-locates an M-tile's whole
# A-sharing superblock set onto ONE XCD, and verify the map is a BIJECTION (every logical task
# executed exactly once) so numerics are provably unchanged.
#
# HW model (MI355X / gfx950): NUM_XCDS=8, CUS_PER_XCD=32.  The HW launcher round-robins the raw
# row-major linear block id to XCDs by (wgid % NUM_XCDS).  chiplet_transform_chunked is the inverse
# permutation that places a contiguous chunk of logical ids on a single XCD.
#
# Run:  python3 xcd_map_viz.py            # default representative grids
#       python3 xcd_map_viz.py M N        # custom (uses K only via fixed tile sizes)

import sys

NUM_XCDS   = 8
CUS_PER_XCD = 32
BM = 64
BN = 64
NSUB = 8
N_PER_BLOCK = NSUB * BN          # 512


def ceil_div(a, b):
    return (a + b - 1) // b


def chiplet_transform_chunked(workgroup_id, num_workgroups, num_xcds, chunk_size):
    """Exact port of HipKittens include/cdna4/common/util.cuh."""
    xcd = workgroup_id % num_xcds
    block = num_xcds * chunk_size
    limit = (num_workgroups // block) * block
    if workgroup_id > limit:
        return workgroup_id
    local_pid = workgroup_id // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk


def xcd_map_block(blk_y, blk_x, num_m, num_n, W, C, remap=True):
    """Port of kernel.cpp xcd_map_block -> (pid_m, pid_n)."""
    if not remap:
        return blk_y, blk_x
    num_wgs = num_m * num_n
    chunk = C if C > 0 else num_n
    wgid = blk_y * num_n + blk_x
    tid = chiplet_transform_chunked(wgid, num_wgs, NUM_XCDS, chunk)
    blocks_per_grp = W * num_n
    group_id = tid // blocks_per_grp
    first_m = group_id * W
    grp_rows = min(num_m - first_m, W)
    idx_in_grp = tid % blocks_per_grp
    pid_m = first_m + (idx_in_grp % grp_rows)
    pid_n = idx_in_grp // grp_rows
    return pid_m, pid_n


def hw_xcd_of(blk_y, blk_x, num_n):
    """Which XCD the HW puts this PHYSICAL block on (round-robin on raw row-major linear id)."""
    wgid = blk_y * num_n + blk_x
    return wgid % NUM_XCDS


def analyze(M, N, W, C):
    num_m = ceil_div(M, BM)
    num_n = ceil_div(N, N_PER_BLOCK)
    chunk = C if C > 0 else num_n
    total = num_m * num_n

    print("=" * 78)
    print(f" M={M} N={N}  -> grid (gridDim.y=num_m={num_m}, gridDim.x=num_n={num_n})  "
          f"total blocks={total}")
    print(f" BM={BM} N_PER_BLOCK={N_PER_BLOCK}  XCD_W={W}  XCD_C={'auto(num_n)=' + str(chunk) if C == 0 else C}  "
          f"NUM_XCDS={NUM_XCDS} CUS_PER_XCD={CUS_PER_XCD}")
    print("-" * 78)

    # --- bijection / correctness check (the numerics guarantee, mechanically verified) ---
    seen = {}
    for by in range(num_m):
        for bx in range(num_n):
            pm, pn = xcd_map_block(by, bx, num_m, num_n, W, C, remap=True)
            in_range = (pm < num_m and pn < num_n)
            if in_range:
                key = (pm, pn)
                seen[key] = seen.get(key, 0) + 1
    expected = num_m * num_n
    dup = {k: v for k, v in seen.items() if v > 1}
    covered = len(seen)
    bij = (covered == expected and not dup)
    print(f" BIJECTION CHECK (remap=1): in-range tasks covered={covered}/{expected}  "
          f"duplicates={len(dup)}  -> {'OK (pure permutation, numerics identical)' if bij else 'FAIL'}")
    if dup:
        print(f"   !! duplicate tasks: {list(dup.items())[:8]}")

    # --- A-sharing locality: for each XCD, how many distinct M-tiles' A does it touch? ---
    # For the remapped schedule, group physical blocks by HW XCD, list the logical pid_m they run.
    xcd_to_mtiles = {x: set() for x in range(NUM_XCDS)}
    xcd_block_count = {x: 0 for x in range(NUM_XCDS)}
    for by in range(num_m):
        for bx in range(num_n):
            x = hw_xcd_of(by, bx, num_n)
            pm, pn = xcd_map_block(by, bx, num_m, num_n, W, C, remap=True)
            if pm < num_m and pn < num_n:
                xcd_to_mtiles[x].add(pm)
                xcd_block_count[x] += 1

    # Stock V4 (remap=0): pid_m == blk_y, so an M-tile's num_n superblocks are
    # consecutive linear ids -> sprayed across all 8 XCDs.
    print("-" * 78)
    print(" Per-XCD logical M-tiles touched (the A-sharing footprint each XCD must gather):")
    print(f"   {'XCD':>4} {'#blocks':>8} {'#distinctM':>11}   M-tiles (first 12)")
    for x in range(NUM_XCDS):
        ms = sorted(xcd_to_mtiles[x])
        print(f"   {x:>4} {xcd_block_count[x]:>8} {len(ms):>11}   {ms[:12]}")

    # Contrast: a single M-tile's superblocks under remap vs stock — which XCDs run them.
    print("-" * 78)
    sample_m = min(1, num_m - 1)
    print(f" Where do the {num_n} A-sharing superblocks of M-tile {sample_m} physically run?")
    remap_xcds = []
    stock_xcds = []
    for by in range(num_m):
        for bx in range(num_n):
            pm, pn = xcd_map_block(by, bx, num_m, num_n, W, C, remap=True)
            if pm == sample_m and pn < num_n:
                remap_xcds.append(hw_xcd_of(by, bx, num_n))
    # stock: M-tile sample_m's superblocks are physical blocks (by=sample_m, bx=0..num_n-1)
    for bx in range(num_n):
        stock_xcds.append(hw_xcd_of(sample_m, bx, num_n))
    print(f"   stock V4 (remap=0): XCDs = {sorted(set(stock_xcds))}  "
          f"(superblocks sprayed across {len(set(stock_xcds))} XCDs)")
    print(f"   XCD remap (remap=1): XCDs = {sorted(set(remap_xcds))}  "
          f"(superblocks on {len(set(remap_xcds))} XCD(s))")
    print("=" * 78)
    print()


def main():
    if len(sys.argv) >= 3:
        grids = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        # representative grids; canonical XCD A/B point is M=1024,N=2048.
        grids = [(1024, 2048), (512, 2048), (256, 2048), (2048, 2048), (1024, 7168)]
    W = 8   # XCD_W default
    C = 0   # XCD_C auto (= num_n)
    for (M, N) in grids:
        analyze(M, N, W, C)


if __name__ == "__main__":
    main()
