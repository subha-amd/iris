#!/usr/bin/env python3
# probe_drn.py — does MORI's `totalRecvTokenNum` accumulate across dispatches?
#
# Source reading says YES:
#   intranode.hpp:236        atomicAdd(args.totalRecvTokenNum, recvTokenNum)   <- every dispatch ADDS
#   dispatch_combine.cpp:328 hipMemset(totalRecvTokenNum, 0, ...)              <- ONCE, at construction
#   dispatch_combine.cpp:460 void EpDispatchCombineHandle::LaunchReset(...) {} <- EMPTY STUB
#   dispatch_combine.py:864  combine(..., call_reset: bool = False)            <- and dispatch() never resets
#
# If so, b3_ep8_unfused.py's timing loop hands `fused_moe(..., num_local_tokens=drn_)` a value that
# grows every iteration -> moe_sorting's moe_buf zeroing + moe_align work grow linearly -> the median
# fmoe time is taken over a monotonically increasing sequence, INFLATING the unfused baseline.
#
# Run: python3 probe_drn.py    (mp.Pool, exactly b3's launch recipe)
import os
os.environ["MORI_GPU_ARCHS"] = "gfx950"
import torch, multiprocessing as mp
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk
import mori

WORLD, HID, E_GLOBAL, TOPK, T = 8, 7168, 256, 8, 64
N_DISPATCH = 6


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
        max_num_inp_token_per_rank=max(8192, T * 4),        # exactly b3's setting
        num_experts_per_rank=Eloc, num_experts_per_token=TOPK,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode)
    op = mori.ops.EpDispatchCombineOp(cfg)

    seq = []
    for _ in range(N_DISPATCH):
        _, _, _, _, drn = op.dispatch(tokens, topk_w, None, topk_ids)
        torch.cuda.synchronize()
        seq.append(int(drn.item()))

    # does an explicit reset() fix it?
    op.reset()
    torch.cuda.synchronize()
    _, _, _, _, drn = op.dispatch(tokens, topk_w, None, topk_ids)
    torch.cuda.synchronize()
    after_reset = int(drn.item())

    aiter.destroy_dist_env()
    return (rank, seq, after_reset)


if __name__ == "__main__":
    mp.freeze_support()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "49717")
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=WORLD) as pool:
        rows = [r.get() for r in [pool.apply_async(worker, args=(i,)) for i in range(WORLD)]]
    print("\n=== MORI totalRecvTokenNum across N successive dispatch() calls (no combine) ===")
    for rank, seq, ar in sorted(rows):
        d = seq[1] - seq[0] if len(seq) > 1 else 0
        verdict = "ACCUMULATES" if (len(seq) > 1 and seq[1] > seq[0]) else "stable"
        print(f"  rank{rank}: {seq}   after op.reset()+dispatch -> {ar}   [{verdict}, delta={d}]")
    s0 = rows[0][1]
    if len(s0) > 1 and s0[1] > s0[0]:
        print("\nCONFIRMED: drn grows by ~R per dispatch. b3's timing loop feeds fused_moe a")
        print("num_local_tokens that climbs every iteration -> the unfused baseline is INFLATED.")
    else:
        print("\nNOT reproduced: drn is stable across dispatches. The source reading was wrong;")
        print("something else zeroes totalRecvTokenNum. Remove finding 2.2 from FAIRNESS_AUDIT.md.")
