#!/usr/bin/env python3
# ================================================================================================
# fairbench/bench_unfused_prefix.py — the UNFUSED half of the dispatch-prefix race.
#
# Runs under mp.Pool (b3_ep8_unfused.py's PROVEN recipe: aiter.init_dist_env +
# shmem_torch_process_group_init), because MORI's shmem bootstrap deadlocks under mpirun. The FUSED
# half (gather_pack) runs separately under mpirun in bench_gather_only.py. Both use the SAME node, the
# SAME per-rank route (same SEED -> identical topk_ids), the SAME T, and MAX over the 8 ranks, so the
# two prefix totals are directly comparable even though they are different processes.
#
# WHAT IS TIMED (all 8 ranks, median over ITERS, then MAX over ranks):
#   EpDispatch bf16  cache  : routing built on device every call            (tier-2, full cost)
#   EpDispatch bf16  REPLAY : routing cached (this is "MORI already cached") (tier-1, plan amortized)
#   dynamic_quant[R]        : aiter per_1x128 of the dispatch output [R,H]   (the fused side gets fp8 free)
#   moe_sorting             : the two sort passes gather_pack claims to replace
#
# The UNFUSED PREFIX = EpDispatch + quant + moe_sorting, i.e. everything MORI+aiter do between the
# router and the fc1 MFMA. Compare against bench_gather_only.py's fused prefix (quant + gather_pack).
#
# ROUTE: to match the fused side EXACTLY, we DON'T use fused_topk (softmax) -- we feed op.dispatch the
# same synthetic top-8 indices the fused side uses: each token picks 8 DISTINCT experts of E_GLOBAL,
# rng(SEED+rank). Weights are uniform (unused by the prefix). This makes both sides' recv-count
# distribution identical.
#
# Run:  python3 bench_unfused_prefix.py        # (NO mpirun -- it spawns its own 8 workers)
# Env:  T=1024 (prefill) | 64 (decode), ITERS, WARMUP, MAX_INP=real|b3, BLOCK_M=32, SEED
# ================================================================================================
import os
os.environ["MORI_GPU_ARCHS"] = "gfx950"
import statistics
import numpy as np
import torch
import multiprocessing as mp
import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
import mori

WORLD = int(os.environ.get("WORLD", "8"))
HID = int(os.environ.get("HID", "7168"))
E_GLOBAL = int(os.environ.get("E", "256"))
TOPK = int(os.environ.get("TOPK", "8"))
T = int(os.environ.get("T", "1024"))
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))
BLOCK_M = int(os.environ.get("BLOCK_M", "32"))
MAX_INP = os.environ.get("MAX_INP", "real")            # real -> T ; b3 -> max(8192, 4T)
SEED = int(os.environ.get("SEED", "1234"))
NG = HID // 128


def worker(rank):
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    aiter.init_dist_env(WORLD, rank)
    Eloc = E_GLOBAL // WORLD
    max_inp = T if MAX_INP == "real" else max(8192, T * 4)

    # ---- the SAME synthetic top-8 route the fused side uses (each token: 8 distinct of E_GLOBAL) ----
    rng = np.random.default_rng(SEED + rank)
    ids = np.stack([rng.choice(E_GLOBAL, TOPK, replace=False) for _ in range(T)]).astype(np.int32)
    topk_ids = torch.from_numpy(ids).to(dev)
    topk_w = torch.ones((T, TOPK), dtype=torch.float32, device=dev)     # uniform; prefix ignores values
    tokens = (torch.randn((T, HID), dtype=dtypes.bf16, device=dev) / 8.0)

    wg = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", wg)
    mori.shmem.shmem_torch_process_group_init("default")

    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=tokens.dtype, rank=rank, world_size=WORLD, hidden_dim=HID,
        scale_dim=0, scale_type_size=0, max_token_type_size=dtypes.bf16.itemsize,
        max_num_inp_token_per_rank=max_inp,
        num_experts_per_rank=Eloc, num_experts_per_token=TOPK,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode)
    op = mori.ops.EpDispatchCombineOp(cfg)

    expert_mask = torch.zeros((E_GLOBAL,), dtype=dtypes.i32, device=dev)
    expert_mask[Eloc * rank: Eloc * (rank + 1)] = 1

    # one dispatch to get R + a replay routing handle (the "already cached" routing)
    do, dw, ds, di, drn = op.dispatch(tokens, topk_w, None, topk_ids)
    torch.cuda.synchronize()
    R = int(drn.item())
    drn_fixed = torch.tensor([R], dtype=dtypes.i32, device=dev)          # avoid the accumulating-counter bug
    *_, routing = op.dispatch(tokens, topk_w, None, topk_ids, return_routing=True)
    torch.cuda.synchronize()

    aq = aiter.get_hip_quant(aiter.QuantType.per_1x128)

    dist = torch.distributed

    def timed(fn):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        ts = []
        for _ in range(ITERS):
            e0.record(); fn(); e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
            dist.barrier()
        return statistics.median(ts)

    stages = {
        "dispatch_cache": lambda: op.dispatch(tokens, topk_w, None, topk_ids),
        "dispatch_replay": lambda: op.dispatch(tokens, topk_w, None, topk_ids, routing=routing),
        "quant_R": lambda: aq(do[:R], quant_dtype=dtypes.fp8),
        "moe_sorting": lambda: moe_sorting(di, dw, E_GLOBAL, HID, dtypes.bf16, BLOCK_M,
                                           expert_mask, drn_fixed, 0),
    }
    res = {k: timed(fn) for k, fn in stages.items()}
    res["R"] = R
    aiter.destroy_dist_env()
    return (rank, res)


if __name__ == "__main__":
    mp.freeze_support()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "49811")
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=WORLD) as pool:
        rows = dict(r.get() for r in [pool.apply_async(worker, args=(i,)) for i in range(WORLD)])

    keys = ["dispatch_cache", "dispatch_replay", "quant_R", "moe_sorting"]
    mx = {k: max(rows[r][k] for r in rows) for k in keys}
    av = {k: sum(rows[r][k] for r in rows) / WORLD for k in keys}
    Rmean = sum(rows[r]["R"] for r in rows) / WORLD

    print(f"\n{'='*96}")
    print(f"UNFUSED DISPATCH-PREFIX (MORI+aiter)  world={WORLD} T={T}/rank HID={HID} E={E_GLOBAL} "
          f"topk={TOPK}  MAX_INP={MAX_INP}  BLOCK_M={BLOCK_M}")
    print(f"recv/rank R (mean) = {Rmean:.0f}")
    print(f"{'='*96}")
    print(f"{'stage':<26} {'MAX us':>9} {'mean us':>9}    (MAX over 8 ranks = production denominator)")
    for k in keys:
        print(f"{k:<26} {mx[k]:>9.2f} {av[k]:>9.2f}")
    print(f"{'-'*96}")
    print(f"  UNFUSED PREFIX tier-1 (routing cached): dispatch_replay + quant_R + moe_sorting")
    print(f"     = {mx['dispatch_replay']:.1f} + {mx['quant_R']:.1f} + {mx['moe_sorting']:.1f} "
          f"= {mx['dispatch_replay']+mx['quant_R']+mx['moe_sorting']:.1f} us")
    print(f"  UNFUSED PREFIX tier-2 (routing on device): dispatch_cache + quant_R + moe_sorting")
    print(f"     = {mx['dispatch_cache']:.1f} + {mx['quant_R']:.1f} + {mx['moe_sorting']:.1f} "
          f"= {mx['dispatch_cache']+mx['quant_R']+mx['moe_sorting']:.1f} us")
    print(f"\n  compare vs bench_gather_only.py fused prefix (quant[T] + gather_pack) at the same T.")
    print(f"{'='*96}\n")
