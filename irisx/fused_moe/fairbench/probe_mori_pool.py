#!/usr/bin/env python3
# probe_mori_pool.py — does MORI bootstrap under b3's mp.Pool recipe (no mpirun)?
#
# probe_boot.py MODE=mori_only showed mori.shmem.shmem_init_attr() hangs for ALL 8 ranks under
# `mpirun`, with or without IRIS.  b3_ep8_unfused.py uses mp.Pool(spawn) + aiter.init_dist_env +
# shmem_torch_process_group_init and is known to work on this cluster.  If that still holds, the fair
# benchmark must be split into two processes (MORI side under mp.Pool, IRIS side under mpirun) rather
# than one.
#
# Run: python3 probe_mori_pool.py
import os
os.environ["MORI_GPU_ARCHS"] = "gfx950"
import torch, multiprocessing as mp
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk
import mori

WORLD, HID, E_GLOBAL, TOPK, T = 8, 7168, 256, 8, 64


def worker(rank):
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(1000 + rank)
    aiter.init_dist_env(WORLD, rank)
    Eloc = E_GLOBAL // WORLD

    tokens = torch.randn((T, HID), dtype=dtypes.bf16, device=dev)
    score = torch.randn((T, E_GLOBAL), dtype=dtypes.bf16, device=dev)
    topk_w, topk_ids = fused_topk(tokens, score, TOPK, True)

    wg = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", wg)
    mori.shmem.shmem_torch_process_group_init("default")

    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=tokens.dtype, rank=rank, world_size=WORLD, hidden_dim=HID,
        scale_dim=0, scale_type_size=0, max_token_type_size=dtypes.bf16.itemsize,
        max_num_inp_token_per_rank=T,
        num_experts_per_rank=Eloc, num_experts_per_token=TOPK,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode)
    op = mori.ops.EpDispatchCombineOp(cfg)

    do, dw, ds, di, drn = op.dispatch(tokens, topk_w, None, topk_ids)
    torch.cuda.synchronize()
    R = int(drn.item())

    # replay-mode routing handle (the tier-1 fairness analog of our precomputed SEG/TILE)
    *_, routing = op.dispatch(tokens, topk_w, None, topk_ids, return_routing=True)
    torch.cuda.synchronize()
    _, _, _, _, drn2 = op.dispatch(tokens, topk_w, None, topk_ids, routing=routing)
    torch.cuda.synchronize()
    R2 = int(drn2.item())

    aiter.destroy_dist_env()
    return (rank, R, R2)


if __name__ == "__main__":
    mp.freeze_support()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "49611")
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=WORLD) as pool:
        rows = [r.get() for r in [pool.apply_async(worker, args=(i,)) for i in range(WORLD)]]
    for r in sorted(rows):
        print(f"  rank{r[0]}: dispatch recv={r[1]}  replay recv={r[2]}")
    print("PROBE MORI/mp.Pool: PASS (dispatch + replay both live)")
