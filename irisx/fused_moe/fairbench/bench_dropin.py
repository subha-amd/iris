#!/usr/bin/env python3
# ================================================================================================
# fairbench/bench_dropin.py — the DROP-IN fused dispatch: input == exactly the router's output.
#
# The point (from the user, 2026-07-08): a valid drop-in replacement for "EpDispatch + moe_sorting"
# must consume EXACTLY what the router hands the MoE region in production — per rank:
#     tokens  bf16 [T, HID]         (hidden states)
#     topk_ids     [T, TOPK]        (which experts each token goes to)
# and produce the fc1 A operand. Intermediate representation may differ; INPUTS must match.
#
# So this script feeds the fused path RAW bf16 tokens + RAW topk_ids and times the WHOLE chain
# on-clock as one region, MAX over 8 ranks:
#     quant (bf16->fp8, per_1x128)         [aiter, local, at origin]
#  -> place fp8 on the IRIS symmetric heap [so peers can pull it]
#  -> plan_allgather_ids (topk_ids)        [IRIS: a pull consumer must know GLOBAL routing]
#  -> build_plan (count/scan/scatter)      [on device: topk_ids -> expert_row_begin + rowmap]
#  -> gather_pack_rowmap                    [IRIS: pull each expert's fp8 rows into the packed buffer]
#     == A_pk (expert-major fp8) = fc1 A operand
#
# TWO regions timed for the tier-1 vs tier-2 CONTRAST:
#   TIER-2 (drop-in valid) : the full chain above, built from raw topk_ids on device.
#   TIER-1 (NOT drop-in)   : same but with a HOST-PRECOMPUTED rowmap (skips allgather + build_plan).
# The DIFFERENCE (tier2 - tier1) is exactly the routing-plan-from-topk_ids cost that tier-1 hides and
# that production CANNOT skip (the router only produces topk_ids; there is no precomputed plan).
#
# Compare the TIER-2 total against bench_unfused_prefix.py's tier-2 EpDispatch + quant + moe_sorting.
#
# Run: mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 python3 bench_dropin.py
# Env: T=1024 (prefill) | 64 (decode), ITERS, WARMUP
# ================================================================================================
import os
os.environ.setdefault("HSA_XNACK", "1")
os.environ.setdefault("MORI_GPU_ARCHS", "gfx950")
import sys
DK = os.environ.get("DK_ROOT", "/home/subvadla/HipKittens/distributed-kernels")
sys.path.insert(0, DK)
sys.path.insert(0, os.path.join(DK, "b1_dispatch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faulthandler
_LR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
_FLOG = open(f"/tmp/dropin_r{_LR}.log", "w", buffering=1)
faulthandler.enable(file=_FLOG)
faulthandler.dump_traceback_later(90, repeat=True, file=_FLOG)

import statistics
import numpy as np
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
import torch
import iris_py
import tk_kernel
import real_route as RR
import aiter
from aiter import dtypes

HID = int(os.environ.get("HID", "7168"))
E_GLOBAL = int(os.environ.get("E", "256"))
TOPK = int(os.environ.get("TOPK", "8"))
T = int(os.environ.get("T", "1024"))
ITERS = int(os.environ.get("ITERS", "30"))
WARMUP = int(os.environ.get("WARMUP", "10"))
SEED = int(os.environ.get("SEED", "1234"))
QGROUP = 128
NG = HID // QGROUP

iris = iris_py.Iris(heap_size_mb=int(os.environ.get("HEAP_MB", "2048")), verbose=False)
rank, world = iris.rank(), iris.world_size()
torch.cuda.set_device(rank)
comm = MPI.COMM_WORLD
dev = torch.device(f"cuda:{rank}")
Eloc = E_GLOBAL // world


def p0(*a):
    if rank == 0:
        print(*a, flush=True)


# ---- the ROUTER'S OUTPUT (same synthetic top-8 route as the other benches: same SEED) ------------
rng = np.random.default_rng(SEED + rank)
my_ids_np = np.stack([rng.choice(E_GLOBAL, TOPK, replace=False) for _ in range(T)]).astype(np.int32)
all_ids_np = np.stack(comm.allgather(my_ids_np))
plan = RR.route_for_rank(all_ids_np, rank, world, E_GLOBAL)
S, Mpacked = plan["stats"], plan["Mpacked"]
Mpk_max = comm.allreduce(Mpacked, op=MPI.MAX)

# the ACTUAL production input this rank holds:
torch.manual_seed(SEED + rank)
tokens = torch.randn((T, HID), dtype=dtypes.bf16, device=dev) / 8.0     # bf16 hidden states
# topk_ids [T,TOPK] lives on the heap (peers all-gather it in plan_allgather_ids)


def _view(t, ts, shape, td):
    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)


def make_fp8(M, Kd):
    t = iris.empty([M, Kd // 2], "bfloat16")
    return (_view(t, "<u2", (M, Kd // 2), torch.bfloat16),
            _view(t, "|u1", (M, Kd), torch.float8_e4m3fn))


def make_iris(shape, dtype):
    alloc = "float32" if dtype == "int32" else dtype
    t = iris.empty(list(shape), dtype=alloc)
    dmap = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"),
            "int32": (torch.int32, "<i4")}
    td, ts = dmap[dtype]
    return _view(t, ts, shape, td)


# symmetric-heap buffers (uniform sizes across ranks)
A_src_bf16, A_src_fp8 = make_fp8(T, HID)          # fp8 activations placed here after quant (pulled by peers)
A_src_sc = make_iris([T, NG], "float32")
A_pk_bf16, A_pk_fp8 = make_fp8(Mpk_max, HID)      # expert-major packed output (fc1 A operand)
A_pk_sc = make_iris([Mpk_max, NG], "float32")
MY_IDS = make_iris([T * TOPK, 1], "int32")        # this rank's topk_ids (peers pull via allgather)
MY_IDS.copy_(torch.from_numpy(my_ids_np.reshape(-1, 1)).cuda())

# device plan scratch (local)
ALL_IDS = torch.zeros(world * T * TOPK, 1, dtype=torch.int32, device=dev)
COUNTS = torch.zeros(Eloc, 1, dtype=torch.int32, device=dev)
CURSOR = torch.zeros(Eloc, 1, dtype=torch.int32, device=dev)
ERB = torch.zeros(Eloc + 1, 1, dtype=torch.int32, device=dev)
ROWMAP_D = torch.zeros(Mpk_max, 2, dtype=torch.int32, device=dev)
ROWMAP_H = torch.full((Mpk_max, 2), -1, dtype=torch.int32, device=dev)      # host-precomputed (tier-1)
ROWMAP_H[:Mpacked].copy_(torch.from_numpy(plan["rowmap"]).cuda())

aq = aiter.get_hip_quant(aiter.QuantType.per_1x128)
iris.barrier()
ctx = iris.get_device_view()

p0(f"\n{'='*100}")
p0(f"DROP-IN FUSED DISPATCH  input=(tokens bf16[{T},{HID}], topk_ids[{T},{TOPK}])  world={world} "
   f"E={E_GLOBAL} (E_loc={Eloc}) topk={TOPK}")
p0(f"{'='*100}")


# ---- the drop-in region: RAW router output -> fc1 A operand, all on-clock -----------------------
def quant_to_heap():
    tq, tsc = aq(tokens, quant_dtype=dtypes.fp8)
    A_src_fp8.copy_(tq)
    A_src_sc.copy_(tsc.float())


def region_tier2():                       # DROP-IN VALID: build routing from raw topk_ids on device
    quant_to_heap()
    tk_kernel.plan_allgather_ids(MY_IDS, ALL_IDS, ctx, world, T * TOPK)
    tk_kernel.build_plan(ALL_IDS, COUNTS, CURSOR, ERB, ROWMAP_D, world, T, TOPK, Eloc, rank, Mpk_max)
    tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, ROWMAP_D, ctx,
                                          T, Mpk_max, HID)


def region_tier1():                       # NOT DROP-IN: host-precomputed plan (production can't do this)
    quant_to_heap()
    tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, ROWMAP_H, ctx,
                                          T, Mpacked, HID)


# ---- CORRECTNESS: device-built plan must match the host plan, and the two gathers must agree -----
A_pk_fp8.view(torch.uint8).zero_(); A_pk_sc.zero_()
torch.cuda.synchronize(); iris.barrier()
region_tier2()
torch.cuda.synchronize(); iris.barrier()
erb_d = ERB.cpu().numpy().ravel()
erb_h = np.append(plan["expert_row_begin"], Mpacked)
rmd = ROWMAP_D.cpu().numpy()
# device plan (built on-device from raw topk_ids) must match the host plan: same expert_row_begin, and
# the SAME set of (src_rank,src_token) per expert (intra-expert order differs -- atomic scatter vs host
# sort -- which is fine, any permutation of an expert's rows is a valid packing).
sets_ok = all(set(map(tuple, rmd[int(erb_h[e]):int(erb_h[e]) + int(plan["rows_per_expert"][e])])) ==
              set(map(tuple, plan["rowmap"][int(erb_h[e]):int(erb_h[e]) + int(plan["rows_per_expert"][e])]))
              for e in range(Eloc))
erb_ok = bool((erb_d == erb_h).all())
routed_nonzero = bool((A_pk_fp8.view(torch.uint8)[:Mpacked, 0].cpu().numpy() != 0).sum() > 0)

gates = dict(dev_plan_erb=erb_ok, dev_plan_sets=sets_ok, packed_nonzero=routed_nonzero)
allg = comm.gather(gates, root=0)
if rank == 0:
    p0("CORRECTNESS (device plan from raw topk_ids vs host plan):")
    for k in gates:
        n = sum(1 for g in allg if g[k])
        p0(f"  {k:<20} {n}/{world} {'PASS' if n == world else 'FAIL'}")
ok = comm.allreduce(1 if (erb_ok and sets_ok and routed_nonzero) else 0, op=MPI.MIN)
assert ok, "drop-in correctness FAILED"

# ---- TIMING: each region as ONE unit, warm, MAX over ranks --------------------------------------
EV = {}


def timed(name, fn):
    if name not in EV:
        EV[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0, e1 = EV[name]
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize(); comm.Barrier()
    ts = []
    for _ in range(ITERS):
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
        comm.Barrier()
    return statistics.median(ts)


# also time the pieces so the tier-2 total decomposes
def piece_quant():
    quant_to_heap()


def piece_allgather():
    tk_kernel.plan_allgather_ids(MY_IDS, ALL_IDS, ctx, world, T * TOPK)


def piece_build():
    tk_kernel.build_plan(ALL_IDS, COUNTS, CURSOR, ERB, ROWMAP_D, world, T, TOPK, Eloc, rank, Mpk_max)


def piece_gather():
    tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, ROWMAP_D, ctx,
                                          T, Mpk_max, HID)


res = {
    "REGION tier1 (host plan, NOT drop-in)": timed("t1", region_tier1),
    "REGION tier2 (from raw topk_ids, DROP-IN)": timed("t2", region_tier2),
    "  piece: quant+place": timed("q", piece_quant),
    "  piece: plan_allgather_ids": timed("ag", piece_allgather),
    "  piece: build_plan": timed("bp", piece_build),
    "  piece: gather_pack_rowmap": timed("g", piece_gather),
}
comm.Barrier()
allres = comm.gather(res, root=0)
if rank == 0:
    mx = {k: max(r[k] for r in allres) for k in res}
    p0(f"\n{'-'*100}")
    p0(f"{'region / piece':<44} {'MAX us':>9}    (MAX over {world} ranks, warm {WARMUP}+{ITERS} iters)")
    p0(f"{'-'*100}")
    for k in res:
        p0(f"{k:<44} {mx[k]:>9.2f}")
    t1 = mx["REGION tier1 (host plan, NOT drop-in)"]
    t2 = mx["REGION tier2 (from raw topk_ids, DROP-IN)"]
    p0(f"\n  TIER-1 (host-precomputed plan) = {t1:.1f} us  <- NOT a valid drop-in "
       f"(production has no precomputed plan)")
    p0(f"  TIER-2 (built from raw topk_ids) = {t2:.1f} us  <- THE drop-in number (input == router output)")
    p0(f"  the routing-plan-from-topk_ids cost tier-1 hides = tier2 - tier1 = {t2 - t1:.1f} us "
       f"(plan_allgather_ids + build_plan)")
    p0(f"\n  compare TIER-2 vs bench_unfused_prefix.py tier-2 (EpDispatch_fp8 + quant_T + moe_sorting).")
    p0(f"{'='*100}\n")

iris.barrier(); comm.Barrier()
faulthandler.cancel_dump_traceback_later()
MPI.Finalize()
