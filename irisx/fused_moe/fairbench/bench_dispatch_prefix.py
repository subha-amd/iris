#!/usr/bin/env python3
# ================================================================================================
# fairbench/bench_dispatch_prefix.py
#
# A FAIR, same-input / same-output race between
#     FUSED    : gather_pack                                   (irisx/fused_moe/kernel.cpp)
# and UNFUSED  : MORI EpDispatch  ->  aiter dynamic_quant  ->  aiter moe_sorting
#
# ------------------------------------------------------------------------------------------------
# WHY THE OLD COMPARISON WAS NOT FAIR (all five verified against the sources, 2026-07-08)
#
#  1. CONCURRENCY.  `fused_moe/example.py:644` runs the whole region on `rank == CONSUMER` only;
#     the other 7 ranks sit at a barrier. The b3 baseline runs all 8 ranks' all-to-all at once and
#     reduces with MAX over ranks. Here EVERY rank runs EVERY stage, and every number is MAX-over-ranks.
#
#  2. INPUT DTYPE.  gather_pack is handed activations that are ALREADY fp8 + per-128 scales
#     (`example.py:353`, quantized on the host, off the clock).  b3 dispatches bf16 and pays
#     `dynamic_quant` inside `fused_moe`.  Here BOTH paths start from bf16 `tokens[T,7168]` and the
#     per_1x128 quant is timed on whichever side pays it.
#
#  3. ROUTING PLAN.  gather_pack consumes host-precomputed SEG/TILE. MORI computes the placement
#     on device on every dispatch.  Two tiers are reported:
#        tier-1 (plan amortized) : MORI *replay-mode* dispatch   vs  gather_pack w/ precomputed plan
#        tier-2 (full cost)      : MORI cache-mode dispatch      vs  gather_pack + on-device plan build
#
#  4. ROUTE REALISM.  The synthetic `b1_dispatch_route.build_multisource_route` emits contiguous
#     source runs (so `tile_is_single_source()` Path-2 fires) and references each source token at
#     most once (= top-1).  Here the plan comes from the REAL `fused_topk` output over 256 experts,
#     top-8 (`real_route.py`): runs collapse to length ~1 and a pull gather re-reads ~1.5x the bytes
#     a deduplicating push moves.
#
#  5. WHAT IS MATERIALIZED.  moe_sorting emits only INDEX ARRAYS -- the unfused fmoe GEMM applies the
#     permutation for free in its A-load.  gather_pack physically writes Mpacked x 7168 fp8 bytes.
#     The stage race below is therefore "everything between the router and the fc1 MFMA", and we
#     print the bytes each side moves so the reader can see who deferred what.
#
# Also fixed: b3 sets `max_num_inp_token_per_rank = max(8192, 4*T)`, which drives MORI's recv buffer
# to world*8192 = 65536 rows AND aiter moe_sorting's workspace/grid (`moe_sorting_opus.h:1364`,
# `k.tokens = h.tokens`).  That inflates the baseline.  `MAX_INP=real` (default) uses T.
# ------------------------------------------------------------------------------------------------
# Run (8x MI350, inside the aiter/MORI container, which also has iris_py + tk_kernel):
#   mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 python3 bench_dispatch_prefix.py
# Env: T=1024 (prefill) | 64 (decode), ITERS, WARMUP, MAX_INP=real|b3, BLOCK_M=32
# ================================================================================================
import os
os.environ.setdefault("MORI_GPU_ARCHS", "gfx950")
os.environ.setdefault("HSA_XNACK", "1")

import sys
DK = os.environ.get("DK_ROOT", "/home/subvadla/HipKittens/distributed-kernels")
sys.path.insert(0, DK)
sys.path.insert(0, os.path.join(DK, "b1_dispatch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faulthandler
_LR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
_LOG = open(f"/tmp/prefix_r{_LR}.log", "w", buffering=1)
faulthandler.enable(file=_LOG)
faulthandler.dump_traceback_later(300, repeat=True, file=_LOG)

import statistics
import numpy as np
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI                            # noqa: E402
import torch                                      # noqa: E402
# torch MUST be imported before iris_py / tk_kernel -- this is the order example.py and probe_boot.py
# use, and both work. (Bringing the HK/IRIS extension up first is one of the two things that made
# mori.shmem.shmem_init_attr() hang forever on all 8 ranks; see FAIRNESS_AUDIT.md §3.)
import iris_py                                    # noqa: E402
import tk_kernel                                  # noqa: E402

import real_route as RR                           # noqa: E402

# ---- config -------------------------------------------------------------------------------------
HID = int(os.environ.get("HID", "7168"))
E_GLOBAL = int(os.environ.get("E", "256"))
TOPK = int(os.environ.get("TOPK", "8"))
T = int(os.environ.get("T", "1024"))              # tokens per rank (prefill 1024 / decode 64)
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))
BLOCK_M = int(os.environ.get("BLOCK_M", "32"))    # aiter moe_sorting unit_size (fused_moe BLOCK_SIZE_M)
MAX_INP = os.environ.get("MAX_INP", "real")       # real -> T ; b3 -> max(8192, 4T) (reproduce b3)
SEED = int(os.environ.get("SEED", "1234"))
QGROUP = 128
NG = HID // QGROUP

iris = iris_py.Iris(heap_size_mb=int(os.environ.get("HEAP_MB", "2048")), verbose=False)
rank, world = iris.rank(), iris.world_size()
torch.cuda.set_device(rank)
comm = MPI.COMM_WORLD
dev = torch.device(f"cuda:{rank}")


def p0(*a):
    if rank == 0:
        print(*a, flush=True)
        print(*a, file=_LOG, flush=True)


import mori                                        # noqa: E402
import mori.shmem                                  # noqa: E402

# MORI bootstrap: uid from rank0, broadcast with mpi4py.
#   * NOT shmem_torch_process_group_init() -- its uid broadcast goes through a torch.distributed NCCL
#     collective, which we don't have here (and which deadlocks under mpirun).
#   * NOT shmem_mpi_init() -- not compiled into this container's mori_cpp.
# Kept ahead of `import aiter` for hygiene, though that ordering was NOT the cause of the
# shmem_init_attr hang (tested both ways -- see FAIRNESS_AUDIT.md §3 trap 4).
_uid = mori.shmem.shmem_get_unique_id() if rank == 0 else None
_uid = comm.bcast(_uid, root=0)
mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world, _uid)
comm.Barrier()

import aiter                                       # noqa: E402
from aiter import dtypes                           # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting  # noqa: E402

Eloc = E_GLOBAL // world
MAX_INP_TOK = T if MAX_INP == "real" else max(8192, T * 4)

# ---- inputs: identical for BOTH paths ------------------------------------------------------------
torch.manual_seed(SEED + rank)
tokens = torch.randn((T, HID), dtype=dtypes.bf16, device=dev) / 8.0
score = torch.randn((T, E_GLOBAL), dtype=dtypes.bf16, device=dev)
topk_w, topk_ids = fused_topk(tokens, score, TOPK, True)     # [T,8] f32 / [T,8] i32 (global expert ids)

# every rank needs every rank's routing to build its own pull plan (host, off-clock for tier 1)
all_ids = np.stack(comm.allgather(topk_ids.cpu().numpy().astype(np.int32)))       # [W,T,8]
all_w = np.stack(comm.allgather(topk_w.float().cpu().numpy()))                    # [W,T,8]
plan = RR.route_for_rank(all_ids, rank, world, E_GLOBAL, all_topk_weights=all_w)
S = plan["stats"]
Mpacked = plan["Mpacked"]

# symmetric heap => every rank must allocate the SAME sizes in the SAME order
Mpk_max = comm.allreduce(Mpacked, op=MPI.MAX)
Nseg_max = comm.allreduce(int(plan["segs"].shape[0]), op=MPI.MAX)
Ntile_max = comm.allreduce(int(plan["tiles"].shape[0]), op=MPI.MAX)
Nseg, Ntile = int(plan["segs"].shape[0]), int(plan["tiles"].shape[0])

p0(f"\n{'='*100}")
p0(f"FAIR DISPATCH-PREFIX BENCH   world={world} T={T}/rank  HID={HID} E={E_GLOBAL} (E_loc={Eloc}) "
   f"topk={TOPK}  MAX_INP={MAX_INP}({MAX_INP_TOK})  BLOCK_M={BLOCK_M}")
p0(f"{'='*100}")
allS = comm.gather(S, root=0)
if rank == 0:
    p0("real-route stats (per destination rank), from fused_topk -- NOT the synthetic route:")
    p0(f"  {'rank':>4} {'routed':>7} {'distinct':>8} {'dup':>6} {'segs':>7} {'mean_run':>8} "
       f"{'tiles':>6} {'1src':>5} {'maxsegc':>7} {'Mpacked':>8} {'pad%':>6}")
    for r, s in enumerate(allS):
        p0(f"  {r:>4} {s['routed_rows']:>7} {s['distinct_src_tokens']:>8} {s['dup_factor']:>6.3f} "
           f"{s['n_segments']:>7} {s['mean_run']:>8.2f} {s['n_tiles']:>6} {s['single_source_tiles']:>5} "
           f"{s['max_seg_count']:>7} {s['mpacked']:>8} {s['pad_frac']*100:>5.1f}%")
    p0("  (synthetic build_multisource_route would show: dup=1.000, mean_run~20.5, many 1src tiles, pad=0%)")

# ---- IRIS heap buffers (identical alloc order + size on every rank) ------------------------------


def _view(t, ts, shape, td):
    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)


def make_fp8(M, Kdim):
    t = iris.empty([M, Kdim // 2], "bfloat16")
    return (_view(t, "<u2", (M, Kdim // 2), torch.bfloat16),
            _view(t, "|u1", (M, Kdim), torch.float8_e4m3fn))


def make_iris(shape, dtype):
    alloc = "float32" if dtype == "int32" else dtype
    t = iris.empty(list(shape), dtype=alloc)
    dmap = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"),
            "int32": (torch.int32, "<i4")}
    td, ts = dmap[dtype]
    return _view(t, ts, shape, td)


A_src_bf16, A_src_fp8 = make_fp8(T, HID)                 # this rank's quantized tokens (pulled remotely)
A_src_sc = make_iris([T, NG], "float32")
A_pk_bf16, A_pk_fp8 = make_fp8(Mpk_max, HID)             # expert-major packed output
A_pk_sc = make_iris([Mpk_max, NG], "float32")
SEG = make_iris([max(Nseg_max, 1), 5], "int32")
TILE = make_iris([max(Ntile_max, 1), 4], "int32")

SEG[:Nseg].copy_(torch.from_numpy(plan["segs"]).cuda())
TILE[:Ntile].copy_(torch.from_numpy(plan["tiles"]).cuda())
A_pk_fp8.view(torch.uint8).zero_()
A_pk_sc.zero_()

# ---- the quant kernel BOTH paths must pay --------------------------------------------------------
aq = aiter.get_hip_quant(aiter.QuantType.per_1x128)
tq, tsc = aq(tokens, quant_dtype=dtypes.fp8)
torch.cuda.synchronize()
p0(f"\naiter per_1x128 quant: tq{tuple(tq.shape)} {tq.dtype} | tsc{tuple(tsc.shape)} {tsc.dtype} "
   f"stride={tsc.stride()}")
if tsc.shape != (T, NG):
    raise SystemExit(f"aiter per_1x128 scale layout is {tuple(tsc.shape)}, expected token-major "
                     f"[{T},{NG}] -- gather_pack's sc_src ABI assumes token-major (DATA_FLOW_AND_ABI.md "
                     f"trap #1). Transpose here before comparing.")

A_src_fp8.copy_(tq)                       # off-clock: a production integrator would quant straight
A_src_sc.copy_(tsc.float())               # into the symmetric heap (see NOTE in the writeup)
iris.barrier()
ctx = iris.get_device_view()

# ---- MORI ops -------------------------------------------------------------------------------------
expert_mask = torch.zeros((E_GLOBAL,), dtype=dtypes.i32, device=dev)
expert_mask[Eloc * rank: Eloc * (rank + 1)] = 1


def make_op(dtype, scale_dim, scale_ts):
    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=dtype, rank=rank, world_size=world, hidden_dim=HID,
        scale_dim=scale_dim, scale_type_size=scale_ts,
        max_token_type_size=dtypes.bf16.itemsize,
        max_num_inp_token_per_rank=MAX_INP_TOK,
        num_experts_per_rank=Eloc, num_experts_per_token=TOPK,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode)
    return mori.ops.EpDispatchCombineOp(cfg)


op_bf16 = make_op(dtypes.bf16, 0, 0)
op_fp8 = make_op(dtypes.fp8, NG, tsc.dtype.itemsize)

do, dw, ds, di, drn = op_bf16.dispatch(tokens, topk_w, None, topk_ids)
torch.cuda.synchronize()
R = int(drn.item())
*_, routing_bf16 = op_bf16.dispatch(tokens, topk_w, None, topk_ids, return_routing=True)
torch.cuda.synchronize()

do8, dw8, ds8, di8, drn8 = op_fp8.dispatch(tq, topk_w, tsc, topk_ids)
torch.cuda.synchronize()
*_, routing_fp8 = op_fp8.dispatch(tq, topk_w, tsc, topk_ids, return_routing=True)
torch.cuda.synchronize()

p0(f"MORI recv/rank R={R}  (expected distinct_src_tokens={S['distinct_src_tokens']})  "
   f"dispatch_out{tuple(do.shape)}")

# ================================================================================================
# CORRECTNESS: both prefixes must present fc1 with the SAME (expert -> source rows) mapping.
# ================================================================================================
# 1. fused: A_pk[dst] must be the raw fp8 bytes of tokens[src_rank][src_token]
tk_kernel.dispatch_gather_pack(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, ctx,
                               T, Mpacked, HID, Nseg, Ntile)
torch.cuda.synchronize()
iris.barrier()

all_tq = np.stack(comm.allgather(tq.view(torch.uint8).cpu().numpy()))            # [W,T,HID] u8
all_tsc = np.stack(comm.allgather(tsc.float().cpu().numpy()))                    # [W,T,NG]
got = A_pk_fp8.view(torch.uint8)[:Mpacked].cpu().numpy()
got_sc = A_pk_sc[:Mpacked].cpu().numpy()
ps = plan["packed_src"]
routed = ps[:, 0] >= 0
want = all_tq[ps[routed, 0], ps[routed, 1]]
want_sc = all_tsc[ps[routed, 0], ps[routed, 1]]
gp_bytes_ok = bool((got[routed] == want).all())
gp_sc_ok = bool(np.allclose(got_sc[routed], want_sc, rtol=0, atol=0))
gp_pad_ok = bool((got[~routed] == 0).all())

# 2. unfused: MORI's dispatch slots -> (src_rank, src_token); the multiset routed to each local
#    expert must equal the fused plan's.  dispTokIdToSrcTokId = FlatTokenIndex(pe, tok).
src_pos = op_bf16.get_dispatch_src_token_pos()[:R].cpu().numpy()
max_send = world * MAX_INP_TOK
u_rank, u_tok = src_pos // max_send, src_pos % max_send
u_ids = di[:R].cpu().numpy()                                                     # [R,8] global expert ids
mori_pairs = {e: set() for e in range(Eloc)}
for i in range(R):
    for g in u_ids[i]:
        if Eloc * rank <= g < Eloc * (rank + 1):
            mori_pairs[int(g) - Eloc * rank].add((int(u_rank[i]), int(u_tok[i])))
fused_pairs = {e: set() for e in range(Eloc)}
beg, rpe = plan["expert_row_begin"], plan["rows_per_expert"]
for e in range(Eloc):
    for row in range(int(beg[e]), int(beg[e]) + int(rpe[e])):
        fused_pairs[e].add((int(ps[row, 0]), int(ps[row, 1])))
same_map = all(mori_pairs[e] == fused_pairs[e] for e in range(Eloc))

ok = gp_bytes_ok and gp_sc_ok and gp_pad_ok and same_map
oks = comm.gather((rank, gp_bytes_ok, gp_sc_ok, gp_pad_ok, same_map), root=0)
p0("\nCORRECTNESS (each rank):  gather_pack bytes | scales | zero-pad | expert->src map == MORI's")
if rank == 0:
    for r, a, b, c, d in oks:
        p0(f"  rank{r}: {'PASS' if a else 'FAIL'} {'PASS' if b else 'FAIL'} "
           f"{'PASS' if c else 'FAIL'} {'PASS' if d else 'FAIL'}")
assert comm.allreduce(1 if ok else 0, op=MPI.MIN), "correctness gate FAILED -- timings meaningless"

# ================================================================================================
# TIMING.  every rank runs every stage, every iteration; median over ITERS; MAX over ranks.
# ================================================================================================
EV = {}


def timed(name, fn):
    """Run fn() ITERS times under cuda events; return median us on THIS rank."""
    if name not in EV:
        EV[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0, e1 = EV[name]
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    comm.Barrier()
    ts = []
    for _ in range(ITERS):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)      # ms -> us
        comm.Barrier()                            # keep the 8 ranks in lockstep (a2a needs it)
    return statistics.median(ts)


STAGES = [
    ("quant[T]        fp8 per_1x128 of tokens[T,H]", lambda: aq(tokens, quant_dtype=dtypes.fp8)),
    ("EpDispatch bf16 (routing on device)", lambda: op_bf16.dispatch(tokens, topk_w, None, topk_ids)),
    ("EpDispatch bf16 REPLAY (plan cached)",
     lambda: op_bf16.dispatch(tokens, topk_w, None, topk_ids, routing=routing_bf16)),
    ("quant[R]        fp8 per_1x128 of dispatch_out[R,H]", lambda: aq(do[:R], quant_dtype=dtypes.fp8)),
    ("EpDispatch fp8  (routing on device)", lambda: op_fp8.dispatch(tq, topk_w, tsc, topk_ids)),
    ("EpDispatch fp8  REPLAY (plan cached)",
     lambda: op_fp8.dispatch(tq, topk_w, tsc, topk_ids, routing=routing_fp8)),
    ("moe_sorting     (bf16 dispatch out)",
     lambda: moe_sorting(di, dw, E_GLOBAL, HID, dtypes.bf16, BLOCK_M, expert_mask, drn, 0)),
    ("gather_pack     SEG/TILE plan precomputed",
     lambda: tk_kernel.dispatch_gather_pack(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, ctx,
                                            T, Mpacked, HID, Nseg, Ntile)),
]

res = {}
for name, fn in STAGES:
    res[name] = timed(name, fn)
    comm.Barrier()

allres = comm.gather(res, root=0)
if rank == 0:
    mx = {k: max(r[k] for r in allres) for k in res}
    av = {k: sum(r[k] for r in allres) / world for k in res}
    p0(f"\n{'-'*100}")
    p0(f"{'stage':<52} {'MAX us':>9} {'mean us':>9}    (MAX over 8 ranks = the production denominator)")
    p0(f"{'-'*100}")
    for name, _ in STAGES:
        p0(f"{name:<52} {mx[name]:>9.2f} {av[name]:>9.2f}")

    q_T = mx[STAGES[0][0]]
    d_bf = mx[STAGES[1][0]]
    d_bf_rp = mx[STAGES[2][0]]
    q_R = mx[STAGES[3][0]]
    d_f8 = mx[STAGES[4][0]]
    d_f8_rp = mx[STAGES[5][0]]
    srt = mx[STAGES[6][0]]
    gp = mx[STAGES[7][0]]

    p0(f"\n{'='*100}")
    p0("PREFIX TOTALS -- 'everything between the router and the fc1 MFMA', same inputs, same 8 GPUs")
    p0(f"{'='*100}")
    p0(f"  TIER 2  (full cost: routing plan built every step)")
    p0(f"    unfused bf16 : EpDispatch({d_bf:.1f}) + quant[R]({q_R:.1f}) + moe_sorting({srt:.1f})"
       f"          = {d_bf+q_R+srt:8.1f} us")
    p0(f"    unfused fp8  : quant[T]({q_T:.1f}) + EpDispatch_fp8({d_f8:.1f}) + moe_sorting({srt:.1f})"
       f"      = {q_T+d_f8+srt:8.1f} us")
    p0(f"    fused        : quant[T]({q_T:.1f}) + gather_pack({gp:.1f}) + <plan build: NOT YET ON THE CLOCK>"
       f" = {q_T+gp:8.1f} us +?")
    p0(f"  TIER 1  (plan amortized on BOTH sides)")
    p0(f"    unfused bf16 : EpDispatch_REPLAY({d_bf_rp:.1f}) + quant[R]({q_R:.1f}) + moe_sorting({srt:.1f})"
       f"  = {d_bf_rp+q_R+srt:8.1f} us")
    p0(f"    unfused fp8  : quant[T]({q_T:.1f}) + EpDispatch_fp8_REPLAY({d_f8_rp:.1f}) + moe_sorting({srt:.1f})"
       f" = {q_T+d_f8_rp+srt:8.1f} us")
    p0(f"    fused        : quant[T]({q_T:.1f}) + gather_pack({gp:.1f})"
       f"                          = {q_T+gp:8.1f} us")

    # honest byte accounting
    rr, dd, mp = S['routed_rows'], S['distinct_src_tokens'], S['mpacked']
    pull_b = rr * HID * 1 + rr * NG * 4
    push_bf = dd * HID * 2 + dd * (TOPK * 8)
    push_f8 = dd * HID * 1 + dd * (NG * 4 + TOPK * 8)
    p0(f"\n  bytes across XGMI per rank (rank0 route):")
    p0(f"    fused pull  : {rr} rows x (7168 fp8 + 56 f32)      = {pull_b/1e6:7.2f} MB  "
       f"(re-reads each of {dd} distinct tokens {rr/dd:.2f}x -- a push dedups, a pull cannot)")
    p0(f"    MORI push bf16: {dd} deduped tokens x 7168 bf16    = {push_bf/1e6:7.2f} MB")
    p0(f"    MORI push fp8 : {dd} deduped tokens x 7168 fp8+sc  = {push_f8/1e6:7.2f} MB")
    p0(f"  bytes WRITTEN to local HBM:")
    p0(f"    fused gather_pack materializes {mp} x 7168 fp8    = {mp*HID/1e6:7.2f} MB  "
       f"({S['pad_frac']*100:.0f}% of it is BM=256 zero padding)")
    p0(f"    unfused moe_sorting writes index arrays only; the fmoe GEMM applies the permutation "
       f"for free in its A-load")
    p0(f"{'='*100}\n")

iris.barrier()
faulthandler.cancel_dump_traceback_later()
comm.Barrier()
mori.shmem.shmem_finalize()
MPI.Finalize()          # iris_py called MPI_Init; without this mpirun reports rank exit != 0
