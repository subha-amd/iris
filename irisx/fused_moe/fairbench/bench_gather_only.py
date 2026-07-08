#!/usr/bin/env python3
# ================================================================================================
# fairbench/bench_gather_only.py — the FUSED half of the dispatch-prefix race, with no MORI and no
# aiter in the process (so it is not blocked on the MORI shmem_init_attr deadlock).
#
# What this measures, on ALL 8 RANKS, MAX over ranks, under a REAL top-8 route:
#   1. gather_pack           — the shipped kernel, host-precomputed SEG/TILE (route_segment runs)
#   2. gather_pack_rowmap    — same movement, flat (src_rank, src_row) per packed row
#   3. plan_allgather_ids    — IRIS all-gather of topk_ids  (tier-2: the routing metadata exchange)
#   4. build_plan            — on-device count/scan/scatter producing the rowmap (tier-2)
#
# WHY 1 vs 2 MATTERS.  `route_segment` encodes a CONTIGUOUS RUN of source rows. The synthetic
# `b1_dispatch_route.build_multisource_route` produces runs of mean length ~20.5. A REAL top-8 router
# produces mean run **1.03**, so Nseg ~= Mpacked, `tile_is_single_source()` (Path 2) fires for 3 of 200
# tiles, and `build_row_seg_map()` degenerates to a 64-iteration SERIAL scan per tile. The rowmap ABI
# is what an on-device plan builder naturally emits (a counting sort with atomics has no "runs"), so
# (2) is the honest number for a production pull gather. (1) is what the deck measured.
#
# WHY 3+4 MATTER.  example.py builds SEG/TILE on the HOST, off the clock. A real router cannot: topk_ids
# lives on the GPU and changes every token, every layer, every decode step. MORI builds its placement
# on device inside the dispatch kernel (intranode.hpp:145-170). (3)+(4) is what gather_pack must pay to
# be compared against a full-cost MORI dispatch.
#
# Correctness gates (timings are meaningless without them):
#   A. rowmap gather produces byte-identical A_pk to the SEG gather
#   B. the on-device plan's expert_row_begin matches the host plan's, and per expert the SET of
#      (src_rank, src_token) matches (order within an expert is atomics-nondeterministic -- and the
#      pull gather does not care, which is exactly why rowmap is the right ABI for a device plan)
#   C. every routed packed row holds the source token's fp8 bytes; every padding row is exactly zero
#
# Run: mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 python3 bench_gather_only.py
# Env: T=1024 (prefill) | 64 (decode), ITERS, WARMUP
# ================================================================================================
import os
os.environ.setdefault("HSA_XNACK", "1")
import sys
DK = os.environ.get("DK_ROOT", "/home/subvadla/HipKittens/distributed-kernels")
sys.path.insert(0, DK)
sys.path.insert(0, os.path.join(DK, "b1_dispatch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faulthandler
_LR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
_FLOG = open(f"/tmp/gather_r{_LR}.log", "w", buffering=1)
faulthandler.enable(file=_FLOG)
faulthandler.dump_traceback_later(90, repeat=True, file=_FLOG)   # a hang self-reports every 90s

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

HID = int(os.environ.get("HID", "7168"))
E_GLOBAL = int(os.environ.get("E", "256"))
TOPK = int(os.environ.get("TOPK", "8"))
T = int(os.environ.get("T", "1024"))
ITERS = int(os.environ.get("ITERS", "50"))
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


# ---- a REAL top-8 router's output (each token picks 8 DISTINCT experts of 256) --------------------
rng = np.random.default_rng(SEED + rank)
my_ids_np = np.stack([rng.choice(E_GLOBAL, TOPK, replace=False) for _ in range(T)]).astype(np.int32)
all_ids_np = np.stack(comm.allgather(my_ids_np))                       # [W,T,TOPK]
plan = RR.route_for_rank(all_ids_np, rank, world, E_GLOBAL)
S, Mpacked = plan["stats"], plan["Mpacked"]
Nseg, Ntile = int(plan["segs"].shape[0]), int(plan["tiles"].shape[0])

Mpk_max = comm.allreduce(Mpacked, op=MPI.MAX)
Nseg_max = comm.allreduce(Nseg, op=MPI.MAX)
Ntile_max = comm.allreduce(Ntile, op=MPI.MAX)


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


# symmetric heap: identical alloc ORDER and SIZE on every rank
A_src_bf16, A_src_fp8 = make_fp8(T, HID)
A_src_sc = make_iris([T, NG], "float32")
A_pk_bf16, A_pk_fp8 = make_fp8(Mpk_max, HID)
A_pk_sc = make_iris([Mpk_max, NG], "float32")
SEG = make_iris([max(Nseg_max, 1), 5], "int32")
TILE = make_iris([max(Ntile_max, 1), 4], "int32")
MY_IDS = make_iris([T * TOPK, 1], "int32")            # remote-read by plan_allgather_ids

# local (non-heap) plan scratch
ROWMAP_H = torch.zeros(Mpk_max, 2, dtype=torch.int32, device=dev)     # host-built plan
ROWMAP_D = torch.zeros(Mpk_max, 2, dtype=torch.int32, device=dev)     # device-built plan
ALL_IDS = torch.zeros(world * T * TOPK, 1, dtype=torch.int32, device=dev)
COUNTS = torch.zeros(Eloc, 1, dtype=torch.int32, device=dev)
CURSOR = torch.zeros(Eloc, 1, dtype=torch.int32, device=dev)
ERB = torch.zeros(Eloc + 1, 1, dtype=torch.int32, device=dev)

SEG[:Nseg].copy_(torch.from_numpy(plan["segs"]).cuda())
TILE[:Ntile].copy_(torch.from_numpy(plan["tiles"]).cuda())
ROWMAP_H[:Mpacked].copy_(torch.from_numpy(plan["rowmap"]).cuda())
ROWMAP_H[Mpacked:].fill_(-1)
MY_IDS.copy_(torch.from_numpy(my_ids_np.reshape(-1, 1)).cuda())

# Synthetic fp8 activations chosen so the gather is CHEAPLY CHECKABLE with NO cross-rank allgather:
# every byte of (src_rank r, src_token t) == encode(r,t) = (r*97 + (t%251) + 1) & 0xFF (never 0, so
# padding rows are provably zero). The scale of row t on rank r == r*1000 + t. A packed row that
# should come from (r,t) is correct iff all its bytes == encode(r,t) and its scale == r*1000+t --
# checkable locally from the plan alone. (An earlier version allgathered 8x7MB of source bytes per
# rank, which stalled the run; this needs no collective at all.)
def _enc(r, t):
    return ((r * 97 + (t % 251) + 1) & 0xFF)


g = np.random.default_rng(7000 + rank)
tok = np.arange(T)
src_bytes = np.broadcast_to(np.array([_enc(rank, int(t)) for t in tok], dtype=np.uint8)[:, None],
                            (T, HID)).copy()
src_sc = np.broadcast_to((rank * 1000 + tok).astype(np.float32)[:, None], (T, NG)).copy()
A_src_fp8.view(torch.uint8).copy_(torch.from_numpy(src_bytes).cuda())
A_src_sc.copy_(torch.from_numpy(src_sc).cuda())
A_pk_fp8.view(torch.uint8).zero_()
A_pk_sc.zero_()
iris.barrier()
ctx = iris.get_device_view()

p0(f"\n{'='*104}")
p0(f"FUSED GATHER BENCH (no MORI/aiter)   world={world} T={T}/rank HID={HID} E={E_GLOBAL} topk={TOPK}")
p0(f"{'='*104}")
allS = comm.gather(S, root=0)
if rank == 0:
    p0("real-route stats per destination rank (from a real top-8 router, NOT build_multisource_route):")
    p0(f"  {'rank':>4} {'routed':>7} {'distinct':>8} {'dup':>6} {'segs':>7} {'run':>6} "
       f"{'tiles':>6} {'1src':>5} {'maxsegc':>7} {'Mpacked':>8} {'pad%':>6}")
    for r, s in enumerate(allS):
        p0(f"  {r:>4} {s['routed_rows']:>7} {s['distinct_src_tokens']:>8} {s['dup_factor']:>6.3f} "
           f"{s['n_segments']:>7} {s['mean_run']:>6.2f} {s['n_tiles']:>6} {s['single_source_tiles']:>5} "
           f"{s['max_seg_count']:>7} {s['mpacked']:>8} {s['pad_frac']*100:>5.1f}%")
    p0("  (synthetic route would read: dup=1.000, run~20.5, many 1src tiles, pad=0.0%)")

# =================================================================================================
# CORRECTNESS -- checked LOCALLY from the plan + the analytic encode(), NO cross-rank allgather.
# =================================================================================================
ps = plan["packed_src"]                                               # [Mpacked,3] (src_rank,src_token,slot)
routed = ps[:, 0] >= 0
want_bytes = np.where(routed, ((ps[:, 0] * 97 + (ps[:, 1] % 251) + 1) & 0xFF), 0).astype(np.uint8)
want_sc = np.where(routed, ps[:, 0] * 1000 + ps[:, 1], 0).astype(np.float32)


def check_pack(tag):
    got = A_pk_fp8.view(torch.uint8)[:Mpacked].cpu().numpy()          # [Mpacked, HID]
    got_sc = A_pk_sc[:Mpacked].cpu().numpy()                          # [Mpacked, NG]
    ok_b = bool((got[routed] == want_bytes[routed, None]).all())      # every byte == encode(r,t)
    ok_s = bool((got_sc[routed] == want_sc[routed, None]).all())
    ok_p = bool((got[~routed] == 0).all())                            # padding rows exactly zero
    return ok_b, ok_s, ok_p, got


# --- A. SEG gather (the shipped path) ---
tk_kernel.dispatch_gather_pack(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, ctx,
                               T, Mpacked, HID, Nseg, Ntile)
torch.cuda.synchronize(); iris.barrier()
seg_b, seg_s, seg_p, ref_pack = check_pack("seg")

# --- B. rowmap gather (host plan) must be byte-identical ---
A_pk_fp8.view(torch.uint8).zero_(); A_pk_sc.zero_()
torch.cuda.synchronize(); iris.barrier()
tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, ROWMAP_H, ctx,
                                      T, Mpacked, HID)
torch.cuda.synchronize(); iris.barrier()
rm_b, rm_s, rm_p, rm_pack = check_pack("rowmap")
identical = bool((rm_pack == ref_pack).all())

# --- C. on-device plan must agree with the host plan ---
tk_kernel.plan_allgather_ids(MY_IDS, ALL_IDS, ctx, world, T * TOPK)
torch.cuda.synchronize(); iris.barrier()
ag_ok = bool((ALL_IDS.cpu().numpy().reshape(world, T, TOPK) == all_ids_np).all())

tk_kernel.build_plan(ALL_IDS, COUNTS, CURSOR, ERB, ROWMAP_D, world, T, TOPK, Eloc, rank, Mpk_max)
torch.cuda.synchronize()
erb_d = ERB.cpu().numpy().ravel()
erb_h = np.append(plan["expert_row_begin"], Mpacked)
erb_ok = bool((erb_d == erb_h).all())
rmd = ROWMAP_D.cpu().numpy()
sets_ok = True
for e in range(Eloc):
    b, n = int(erb_h[e]), int(plan["rows_per_expert"][e])
    if set(map(tuple, rmd[b:b + n])) != set(map(tuple, plan["rowmap"][b:b + n])):
        sets_ok = False
        break
pad_ok = bool((rmd[Mpacked:] == -1).all()) if Mpk_max > Mpacked else True

# --- D. gather through the DEVICE-built plan ---
A_pk_fp8.view(torch.uint8).zero_(); A_pk_sc.zero_()
torch.cuda.synchronize(); iris.barrier()
tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, ROWMAP_D, ctx,
                                      T, Mpacked, HID)
torch.cuda.synchronize(); iris.barrier()
gotd = A_pk_fp8.view(torch.uint8)[:Mpacked].cpu().numpy()
rd = rmd[:Mpacked]
routed_d = rd[:, 0] >= 0
want_d = ((rd[routed_d, 0] * 97 + (rd[routed_d, 1] % 251) + 1) & 0xFF).astype(np.uint8)
dev_b = bool((gotd[routed_d] == want_d[:, None]).all())
dev_p = bool((gotd[~routed_d] == 0).all())

gates = dict(seg_bytes=seg_b, seg_scales=seg_s, seg_zeropad=seg_p,
             rowmap_bytes=rm_b, rowmap_scales=rm_s, rowmap_zeropad=rm_p,
             rowmap_identical_to_seg=identical,
             dev_allgather_ids=ag_ok, dev_plan_erb=erb_ok, dev_plan_sets=sets_ok,
             dev_plan_pad=pad_ok, dev_plan_gather_bytes=dev_b, dev_plan_gather_zeropad=dev_p)
allg = comm.gather(gates, root=0)
if rank == 0:
    p0("\nCORRECTNESS (all 8 ranks must pass every gate):")
    for k in gates:
        n_ok = sum(1 for g_ in allg if g_[k])
        p0(f"  {k:<28} {n_ok}/8 {'PASS' if n_ok == world else 'FAIL'}")
ok_all = comm.allreduce(1 if all(gates.values()) else 0, op=MPI.MIN)
assert ok_all, "correctness gate FAILED -- timings would be meaningless"

# =================================================================================================
# TIMING — every rank, every stage, every iteration. median over ITERS, then MAX over ranks.
# =================================================================================================
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


STAGES = [
    ("gather_pack        (SEG/TILE, host plan)",
     lambda: tk_kernel.dispatch_gather_pack(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, ctx,
                                            T, Mpacked, HID, Nseg, Ntile)),
    ("gather_pack_rowmap (flat plan, host)",
     lambda: tk_kernel.dispatch_gather_pack_rowmap(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc,
                                                   ROWMAP_H, ctx, T, Mpacked, HID)),
    ("plan_allgather_ids (IRIS, tier-2)",
     lambda: tk_kernel.plan_allgather_ids(MY_IDS, ALL_IDS, ctx, world, T * TOPK)),
    ("build_plan         (on device, tier-2)",
     lambda: tk_kernel.build_plan(ALL_IDS, COUNTS, CURSOR, ERB, ROWMAP_D, world, T, TOPK, Eloc,
                                  rank, Mpk_max)),
]

res = {name: timed(name, fn) for name, fn in STAGES}
comm.Barrier()
allres = comm.gather(res, root=0)
if rank == 0:
    mx = {k: max(r[k] for r in allres) for k in res}
    av = {k: sum(r[k] for r in allres) / world for k in res}
    p0(f"\n{'-'*104}")
    p0(f"{'stage':<48} {'MAX us':>9} {'mean us':>9}   (MAX over 8 ranks = the production denominator)")
    p0(f"{'-'*104}")
    for name, _ in STAGES:
        p0(f"{name:<48} {mx[name]:>9.2f} {av[name]:>9.2f}")
    gseg, grm, gag, gbp = (mx[s[0]] for s in STAGES)
    p0(f"\n  rowmap vs SEG ABI               : {gseg/grm:.3f}x  "
       f"({'rowmap faster' if grm < gseg else 'SEG faster'})")
    p0(f"  tier-2 routing-plan cost        : {gag + gbp:.2f} us "
       f"(allgather {gag:.2f} + build {gbp:.2f}) -- what example.py hides on the host")
    p0(f"  fused prefix, tier 1 (plan free) : gather_pack_rowmap = {grm:.2f} us")
    p0(f"  fused prefix, tier 2 (full cost) : {grm + gag + gbp:.2f} us")

    s0 = allS[0]
    rr, dd, mp = s0['routed_rows'], s0['distinct_src_tokens'], s0['mpacked']
    p0(f"\n  bytes across XGMI (rank0 route): pull {rr} rows x (7168 fp8 + 56 f32) = "
       f"{(rr*HID + rr*NG*4)/1e6:.2f} MB")
    p0(f"    a deduplicating push would move {dd} distinct tokens ({rr/dd:.3f}x fewer rows)")
    p0(f"  local HBM written: {mp} x 7168 fp8 = {mp*HID/1e6:.2f} MB "
       f"({s0['pad_frac']*100:.0f}% BM=256 zero padding)")
    p0(f"{'='*104}\n")

iris.barrier()
comm.Barrier()
faulthandler.cancel_dump_traceback_later()
MPI.Finalize()
