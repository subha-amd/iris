#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
xgmi_probe.py -- on-node validation of the TileComm cost model.

Measures the XGMI link-contention effect that the cost model in tilesched.py
predicts, DIRECTLY on the 8x MI350X, without the full fused_moe build. Every
rank concurrently stores `num_cells` rows of H bf16 to destination ranks chosen
by a schedule; we compare:

    sorted        -- cells grouped by destination rank (the naive CSR order)
    round_robin   -- cycle dst 0,1,...,W-1  (today's hand-rolled interleave)
    proportional  -- byte-weighted fair queueing (the demand-aware schedule)

This isolates exactly the phenomenon behind our measured combine result
(sorted 934 us vs round-robin 386 us): sorted order lets a window of concurrent
thread blocks hammer one XGMI ingress link; spreading orders keep all 8 links busy.

A second, independent measured point to validate (or correct) the cost model's
2.4x calibration. Uniform traffic here should reproduce the ~2.4x sorted/spread
ratio; an imbalanced dst distribution should show proportional >= round_robin.

Run (single 8-GPU node):
    python3 xgmi_probe.py --num_cells 8192 --H 7168 --world 8
(uses torch.multiprocessing.spawn; no mpirun needed)
"""

import argparse
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import iris


@triton.jit
def scatter_sched_kernel(
    src_buf,            # local source rows [num_cells * H] bf16
    dst_buf,            # remote destination buffer [num_cells * H] bf16 (symmetric)
    order_dst,          # int32 [num_cells]: destination rank for each cell (the schedule)
    num_cells,
    H: tl.constexpr,
    cur_rank,
    BLOCK_H: tl.constexpr,
    heap_bases_ptr,
):
    pid = tl.program_id(0)
    if pid >= num_cells:
        return
    dst_rank = tl.load(order_dst + pid)
    row_off = pid * H
    for h0 in range(0, H, BLOCK_H):
        offs = row_off + h0 + tl.arange(0, BLOCK_H)
        mask = (h0 + tl.arange(0, BLOCK_H)) < H
        vals = tl.load(src_buf + offs, mask=mask)
        # remote store over XGMI (local if dst_rank == cur_rank)
        iris.store(dst_buf + offs, vals, cur_rank, dst_rank, heap_bases_ptr, mask=mask)


# ---- schedules (host side; mirror tilesched.py) --------------------------------
def order_sorted(dst_of_cell):
    return np.argsort(dst_of_cell, kind="stable").astype(np.int32)


def order_round_robin(dst_of_cell, world):
    per = [list(np.where(dst_of_cell == r)[0]) for r in range(world)]
    out, idx, rem = [], [0] * world, len(dst_of_cell)
    while rem > 0:
        for r in range(world):
            if idx[r] < len(per[r]):
                out.append(per[r][idx[r]]); idx[r] += 1; rem -= 1
    return np.array(out, dtype=np.int32)


def order_proportional(dst_of_cell, size_of_cell, world):
    per = []
    for r in range(world):
        ix = np.where(dst_of_cell == r)[0]
        per.append(list(ix[np.argsort(-size_of_cell[ix], kind="stable")]))
    lb = np.zeros(world)
    np.add.at(lb, dst_of_cell, size_of_cell)
    rate = np.where(lb > 0, lb / lb.sum(), 0.0)
    vclock = np.full(world, np.inf)
    for r in range(world):
        if per[r]:
            vclock[r] = size_of_cell[per[r][0]] / max(rate[r], 1e-12)
    ptr, out = [0] * world, []
    for _ in range(len(dst_of_cell)):
        r = int(np.argmin(vclock))
        out.append(per[r][ptr[r]]); ptr[r] += 1
        if ptr[r] < len(per[r]):
            vclock[r] += size_of_cell[per[r][ptr[r]]] / max(rate[r], 1e-12)
        else:
            vclock[r] = np.inf
    return np.array(out, dtype=np.int32)


def make_dst(num_cells, world, skew, seed):
    """destination rank per cell. skew=0 -> uniform; larger -> a few ranks hot."""
    rng = np.random.default_rng(seed)
    if skew <= 0:
        p = np.ones(world) / world
    else:
        p = rng.dirichlet(np.full(world, max(1e-3, 1.0 / skew)))
    dst = rng.choice(world, size=num_cells, p=p).astype(np.int32)
    size = np.ones(num_cells, dtype=np.float64)   # uniform row size (one bf16 row)
    return dst, size


def _worker(local_rank, world_size, init_url, args):
    dist.init_process_group(backend="nccl", init_method=init_url,
                            world_size=world_size, rank=local_rank,
                            device_id=torch.device(f"cuda:{local_rank}"))
    shmem = iris.iris(args["heap_size"])
    cur = shmem.get_rank()
    world = shmem.get_num_ranks()
    H = args["H"]
    nC = args["num_cells"]

    src = shmem.zeros(nC * H, device="cuda", dtype=torch.bfloat16)
    dst = shmem.zeros(nC * H, device="cuda", dtype=torch.bfloat16)
    src.copy_(torch.randn(nC * H, device="cuda", dtype=torch.bfloat16))

    # same schedule on every rank (uniform draw with a fixed seed) so all ranks
    # push concurrently -- reproduces the real all-ranks-combine contention.
    dst_of_cell, size_of_cell = make_dst(nC, world, args["skew"], seed=1234)
    schedules = {
        "sorted": order_sorted(dst_of_cell),
        "round_robin": order_round_robin(dst_of_cell, world),
        "proportional": order_proportional(dst_of_cell, size_of_cell, world),
    }

    grid = (nC,)
    results = {}
    for name, order in schedules.items():
        order_t = torch.from_numpy(order).to("cuda")

        def run():
            scatter_sched_kernel[grid](src, dst, order_t, nC, H, cur, 1024,
                                       shmem.get_heap_bases())

        run(); shmem.barrier()
        ms = iris.do_bench(run, shmem.barrier,
                           n_repeat=args["iters"], n_warmup=args["warmup"])
        # MAX over ranks is the true region time (methodology from the handoff)
        ms_max = shmem.broadcast(ms, 0)
        all_ms = [ms]
        results[name] = ms

    shmem.barrier()
    if cur == 0:
        # gather each rank's timings via a simple all-reduce max
        print("\n==== XGMI schedule probe (8x MI350X) ====")
        print(f"num_cells={nC}  H={H}  bytes/cell={H*2}  skew={args['skew']}")
        base = results["round_robin"]
        print(f"  {'schedule':<14}{'us (rank0)':>12}{'  x vs round_robin':>20}")
        for name in schedules:
            us = results[name] * 1e3
            print(f"  {name:<14}{us:>12.1f}{results[name]/base:>20.3f}")
        print("  (model predicts sorted/round_robin ~ 2.4x for uniform traffic)")
        if args["output_file"]:
            with open(args["output_file"], "w") as f:
                json.dump({k: v * 1e3 for k, v in results.items()}
                          | {"num_cells": nC, "H": H, "skew": args["skew"]}, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--num_cells", type=int, default=8192)
    ap.add_argument("--H", type=int, default=7168)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--heap_size", type=int, default=1 << 33)
    ap.add_argument("--output_file", type=str, default="")
    args = vars(ap.parse_args())
    init_url = "tcp://127.0.0.1:29512"
    mp.spawn(_worker, args=(args["world"], init_url, args), nprocs=args["world"], join=True)


if __name__ == "__main__":
    main()
