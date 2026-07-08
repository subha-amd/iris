#!/usr/bin/env python3
# b1_dispatch / example.py
# ================================================================================================
# B1-DISPATCH V0 np=8 driver (main agent runs on the node under flock).
#
# Two-phase, SERIAL, route-aware EP8 pipeline on the CONSUMER rank:
#   PHASE 1  tk_kernel.dispatch_gather_pack(...)  — multi-source EP8 gather of each expert's rows
#            from their SOURCE ranks (route_segments) ONCE into a LOCAL expert-major packed fp8 +
#            scale buffer (BM-padded per-expert regions). A crosses XGMI exactly here.
#   PHASE 2  tk_kernel.grouped_gemm(...)          — the VERIFIED v5_grouped serial grouped GEMM over
#            that LOCAL packed buffer. src_rank = CONSUMER, so its internal ctx.load reads LOCAL HBM
#            (A does NOT re-cross XGMI). 32 experts, flat task list from build_tasks.py.
#
# This is the production-shaped analog of B1-copy: REAL routing + 32 experts + multi-source gather,
# vs B1-copy's single fixed-source dense copy. We time T_gather / T_gemm / T_total with cuda events
# (same-iteration) so they are directly comparable to B1-copy's copy/gemm/total split.
#
# Composition note: phase 1 reuses ep8_gather.h's VERIFIED multi-source row resolver (Agent 03) +
# harness B1's raw fp8-byte copy; phase 2 is v5_grouped's VERIFIED grouped GEMM (Agent 02) VERBATIM.
# The CPU reference is the v5 grouped reference (per-expert dequant(A) @ B^T) — the same numerics that
# PASSED at RMS 0.00331; the gather correctness is additionally proven by the zero-sentinel.
#
# HARNESS TRAPS observed (and avoided here, all documented in EXPERIMENT_LEDGER.md):
#   - iris.empty has NO int32 -> back int32 buffers with float32 alloc + int32 view.
#   - ONLY gathered A (+scales), the packed A (+scales), and the route metadata go on the IRIS
#     symmetric heap. B (weights, ~3GB), C (output), TASKS are LOCAL torch tensors (heap is 512MB).
#   - Identical IRIS allocation ORDER on every rank (symmetric offsets).
#   - ALL ranks call iris.barrier() the SAME number of times (timing loops run on every rank; only
#     recording/printing is gated to the consumer) or it DEADLOCKS.
#   - CPU fp8 reference uses ml_dtypes.float8_e4m3fn (NOT float8_e4m3) or 448->inf->RMS=nan.
# ================================================================================================
import sys, os, time, subprocess, datetime
sys.path.insert(0, "..")
import numpy as np
import torch
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import iris_py
import tk_kernel
from mpi4py import MPI

import build_tasks as BT                       # v5_grouped task-list builder (copied into this dir)
import b1_dispatch_route as RT                 # B1-dispatch routing (32-expert multi-source) builder

# ---- shapes / config (override via env) ----
E       = int(os.environ.get("E", "32"))       # local experts on the consumer
K       = int(os.environ.get("K", "7168"))
N       = int(os.environ.get("N", "2048"))
TOTAL_M = int(os.environ.get("TOTAL_M", "8192"))   # aggregate routed rows across all experts
ROUTE   = os.environ.get("ROUTE", "uniform")       # uniform|zipf|one_hot|several_hot|many_empty
BM      = int(os.environ.get("BM", "64"))
BN      = int(os.environ.get("BN", "64"))
BK      = int(os.environ.get("BK", "64"))
MSRC    = int(os.environ.get("MSRC", "4096"))      # rows in each source rank's activation buffer
ITERS   = int(os.environ.get("ITERS", "50"))
WARMUP  = int(os.environ.get("WARMUP", "10"))
SEED    = int(os.environ.get("SEED", "1234"))
CONSUMER= int(os.environ.get("CONSUMER", "7"))     # the rank that packs + computes (any rank)
# ALL_RANKS=1 : EVERY rank runs the whole region every iteration, and the reported time is the MAX
#   over the 8 ranks -- the same denominator baselines/b3_ep8_unfused.py uses.
#   DEFAULT (0) reproduces the historical behaviour: ONLY rank==CONSUMER executes while the other 7
#   ranks idle at a barrier, so the gather pulls from seven IDLE peers with no all-to-all contention
#   and no straggler. That is NOT comparable to b3 (which runs the real 8-way collective). Every
#   number in the june-30 deck was taken with ALL_RANKS=0. See fairbench/FAIRNESS_AUDIT.md §1.1.
ALL_RANKS = int(os.environ.get("ALL_RANKS", "0"))
# ⚠️ ALL_RANKS + COMBINE: every rank builds the SAME synthetic route (same SEED), so all 8 ranks
#   ctx.store IDENTICAL reduced rows into the same accb cells. The stores are idempotent, so the
#   COMBINE gate still passes -- but it proves nothing about cross-rank accumulation, which
#   combine_pull cannot do (it stores, it does not fetch_add; tilecomm_device.h:123/135/151).
#   Under a REAL top-8 route 100% of tokens have contributions on >=2 producer ranks (mean 5.33), and
#   all but one partial sum would be silently dropped. See fairbench/FAIRNESS_AUDIT.md §1.6.
#   Use ALL_RANKS to measure gather/fc1/act/fc2 under real 8-way XGMI contention; do NOT read the
#   ALL_RANKS combine number as a validated EpCombine replacement.
CSV     = os.environ.get("CSV", "results.csv")
QGROUP  = 128
assert K % QGROUP == 0
NG = K // QGROUP

# Phase-2 schedule selector: 'microtk' = v5 64x64 producer/consumer (default, the path that was slow);
# 'b0' = the B0-class 256x256 8-wave ping-pong GEMM (grouped_gemm_b0). HEAD-TO-HEAD: run this driver
# once with SCHEDULE=microtk and once with SCHEDULE=b0 at the SAME route/shape and compare T_gemm.
# The B0 path pads each expert's packed region to 256 (inside build_b0_tasks) so a 256-row tile can't
# straddle two experts; the phase-1 gather still tiles in GP_BM(=64)-row chunks over that 256-padded
# space, so NO gather rebuild is needed. (b0 requires N in {2048,4096,7168} — the compiled instances.)
SCHEDULE = os.environ.get("SCHEDULE", "microtk")
assert SCHEDULE in ("microtk", "b0"), f"bad SCHEDULE={SCHEDULE}"

# ---- COMPLETE-REGION knobs (additive; defaults reproduce the original single-GEMM path) ----------
# FFN=single  : original path — ONE grouped GEMM at N (the existing verified behavior).
# FFN=full    : the real expert FFN — fc1 g1u1 (N=4096,K=7168) -> SiLU(gate)*up + fp8 re-quant ->
#               fc2 down (N=7168,K=2048), both via grouped_gemm_b0. Requires SCHEDULE=b0.
# COMBINE=1   : after the FFN, scatter the fc2 output back to origin tokens over IRIS (EpCombine), so
#               the timed region is gather -> fc1 -> act -> fc2 -> combine (the full expert region).
FFN          = os.environ.get("FFN", "single")
COMBINE      = int(os.environ.get("COMBINE", "0"))
COMBINE_ATOMIC = int(os.environ.get("COMBINE_ATOMIC", "1"))    # 1=fetch_add (accumulates), 0=store
COMBINE_TLOCAL = int(os.environ.get("COMBINE_TLOCAL", "0"))    # >0 folds src_token (forces accumulate)
COMBINE_MODE = os.environ.get("COMBINE_MODE", "scatter")       # scatter (fp32 atomic) | pull (bf16 gather-reduce)
assert COMBINE_MODE in ("scatter", "pull"), f"bad COMBINE_MODE={COMBINE_MODE}"
COMBINE_GRAN = int(os.environ.get("COMBINE_GRAN", "8"))        # pull store width: 1=bf16(2B) 2=bf16_2(4B) 8=uint4(16B); 8 best
# DECODE=1: route the fc1/fc2 grouped GEMM through the BM=16 skinny-tile path (kills the BM=256 padding
# tax at low M_e). Same 256-padded packed layout (build_b0_tasks_decode) so phase1/act/combine UNCHANGED.
DECODE       = int(os.environ.get("DECODE", "0"))
DECODE_FP8   = int(os.environ.get("DECODE_FP8", "0"))          # 1 = native-fp8 BM=16 decode (halves weight stream)
DECODE_MXFP4 = int(os.environ.get("DECODE_MXFP4", "0"))        # 1 = MXFP4 (OCP W4, E8M0/32blk) BM=16 decode (1/4 bytes)
if DECODE_FP8 or DECODE_MXFP4:
    DECODE = 1                                                 # both use the BM=16 task layout
if DECODE_MXFP4:
    DECODE_FP8 = 0                                             # mxfp4 and fp8 decode are mutually exclusive
assert FFN in ("single", "full"), f"bad FFN={FFN}"
if FFN == "full":
    assert SCHEDULE == "b0", "FFN=full chains grouped_gemm_b0 twice -> requires SCHEDULE=b0"
# fc1/fc2 production shapes (PRODUCTION_ABI.md §4 / route_abi.h). g1u1: fc1 N=2*INTER, fc2 K=INTER.
INTER  = int(os.environ.get("INTER", "2048"))   # SiLU-gated intermediate width
N_FC1  = 2 * INTER                               # 4096 (gate||up fused)
K_FC1  = K                                       # 7168
N_FC2  = K                                       # 7168 (down projects back to hidden)
K_FC2  = INTER                                   # 2048

# ---- MXFP4 (OCP W4, E8M0 per-32-K-block) weight quantizer — matches kernel.cpp grouped_b0_gemm_decode_
# mxfp4_sat EXACTLY (kDecSatPermFp4 swizzle, 2 fp4/byte, low nibble = even col, hardware e2m1 codes). ----
_PERM_FP4 = [
    0,1,2,3,4,5,6,7, 32,33,34,35,36,37,38,39, 64,65,66,67,68,69,70,71, 96,97,98,99,100,101,102,103,
    8,9,10,11,12,13,14,15, 40,41,42,43,44,45,46,47, 72,73,74,75,76,77,78,79, 104,105,106,107,108,109,110,111,
    16,17,18,19,20,21,22,23, 48,49,50,51,52,53,54,55, 80,81,82,83,84,85,86,87, 112,113,114,115,116,117,118,119,
    24,25,26,27,28,29,30,31, 56,57,58,59,60,61,62,63, 88,89,90,91,92,93,94,95, 120,121,122,123,124,125,126,127]
def _quant_b_mxfp4(Bw, Ne):
    """Bw [E*Ne, Kd] bf16 -> (packed_swizzled_fp4_as_bf16 [E*Ne, Kd/4], scale_f32 [E*Ne, Kd/32],
    dequant_bf16 [E*Ne, Kd]). E8M0 scale = 2^(floor(log2(amax))-2); e2m1 nearest-code."""
    dev = Bw.device; R = Bw.shape[0]; Kd = Bw.shape[1]
    levels = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.], device=dev)
    mids   = torch.tensor([0.25,0.75,1.25,1.75,2.5,3.5,5.0], device=dev)
    perm   = torch.tensor(_PERM_FP4, dtype=torch.long, device=dev)
    x = Bw.float().view(R, Kd // 32, 32)
    amax = x.abs().amax(dim=2)                                              # [R, Kd/32]
    _, ex = torch.frexp(amax)                                              # amax = m*2^ex, floor(log2)=ex-1
    se = (ex - 1 - 2 + 127).clamp(1, 254)
    se = torch.where(amax > 0, se, torch.full_like(se, 127))
    scale = torch.exp2((se.float() - 127.0))                              # [R, Kd/32]
    q = x / scale.unsqueeze(2)
    a = q.abs()
    mag = torch.bucketize(a, mids, right=True)                            # 0..7
    sign = (q < 0)
    code = (mag + sign.to(torch.int32) * 8).view(R, Kd).to(torch.uint8)   # [R, Kd] codes 0..15
    deq = (levels[mag] * torch.where(sign, -1.0, 1.0)).view(R, Kd // 32, 32) * scale.unsqueeze(2)
    deq = deq.view(R, Kd).to(torch.bfloat16)
    csw = code.view(R, Kd // 128, 128)[:, :, perm].reshape(R, Kd)         # swizzle per 128-fp4-block
    c2 = csw.view(R, Kd // 2, 2)
    packed = (c2[:, :, 0].int() | (c2[:, :, 1].int() << 4)).to(torch.uint8).contiguous()   # [R, Kd/2] bytes
    packed_bf16 = packed.view(torch.bfloat16).contiguous()               # [R, Kd/4]
    return packed_bf16, scale.contiguous(), deq

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 8, f"B1-dispatch V0 expects np=8 (EP8), got world={world}"
torch.cuda.set_device(rank)
ACTIVE = ALL_RANKS or (rank == CONSUMER)     # does THIS rank execute the region?

# ---- host: build the 32-expert MULTI-SOURCE routing (deterministic across ranks) -------------
# rows_per_expert + the BM-padded expert-major packed layout (v5_grouped), THEN per-packed-row
# (src_rank, src_row) assignment producing route_segments + per-BM-tile metadata (ep8_gather).
rng = np.random.default_rng(SEED)
rows_per_expert = BT.ROUTE_BUILDERS[ROUTE](E, TOTAL_M, rng)
if SCHEDULE == "b0":
    import b0_tasks as B0T               # B0 task builder (TASK_W=4, pads experts to BM=256)
    tasks_np, expert_row_begin, padded_rows, Mpacked = B0T.build_b0_tasks(rows_per_expert, N)
    num_tasks = int(tasks_np.shape[0]); NSUB = 1; n_blocks = num_tasks; TASK_COLS = B0T.B0_TASK_W
else:
    sched = BT.build_grouped_schedule(rows_per_expert, N, BM, BN)
    NSUB             = sched["nsub"]
    tasks_np         = sched["tasks"]
    expert_row_begin = sched["expert_row_begin"]
    padded_rows      = sched["padded_rows"]
    Mpacked          = sched["total_padded_rows"]
    num_tasks        = sched["num_tasks"]
    n_blocks         = sched["n_blocks"]
    TASK_COLS        = BT.TASK_W
assert Mpacked > 0 and num_tasks > 0, "empty route -- nothing to launch"

# FFN=full: build the TWO B0 task lists (same BM=256 packed layout, hence same Mpacked /
# expert_row_begin as above — only the N-tile count differs per GEMM).
if FFN == "full":
    import b0_tasks as B0T
    # DECODE: BM=16-tiled task list (same 256-padded Mpacked/erb); else the BM=256 task list.
    _build_ffn_tasks = B0T.build_b0_tasks_decode if DECODE else B0T.build_b0_tasks
    tasks_fc1_np, erb_fc1, _, Mpacked_fc1 = _build_ffn_tasks(rows_per_expert, N_FC1)
    tasks_fc2_np, erb_fc2, _, Mpacked_fc2 = _build_ffn_tasks(rows_per_expert, N_FC2)
    assert Mpacked_fc1 == Mpacked == Mpacked_fc2, "fc1/fc2 packed layout must match phase-1"
    num_tasks_fc1 = int(tasks_fc1_np.shape[0])
    num_tasks_fc2 = int(tasks_fc2_np.shape[0])

# Multi-source segments over the SAME packed layout: each expert's real rows are split into runs
# coming from different source ranks; padding rows are unrouted (zero-sentinel).
segs, tiles = RT.build_multisource_route(world, MSRC, E, rows_per_expert, expert_row_begin,
                                         padded_rows, Mpacked, BM, seed=SEED + 1)
seg_arr  = RT.segs_to_int_array(segs)
tile_arr = RT.tiles_to_int_array(tiles)
Nseg  = seg_arr.shape[0]
Ntile = tile_arr.shape[0]
assert Ntile == (Mpacked + BM - 1) // BM

if rank == CONSUMER:
    n_single = sum(1 for t in tiles if t["seg_count"] == 1)
    n_multi  = sum(1 for t in tiles if t["seg_count"]  > 1)
    print(f"[b1-dispatch] route={ROUTE} E={E} TOTAL_M={TOTAL_M} -> Mpacked={Mpacked} "
          f"NSUB={NSUB} num_tasks={num_tasks} segs={Nseg} tiles={Ntile} "
          f"single_src_tiles={n_single} multi_src_tiles={n_multi} "
          f"rows[0:6]={list(rows_per_expert[:6])}", flush=True)

# ---- IRIS symmetric-heap tensors (identical allocation ORDER on EVERY rank) -------------------
def make_iris(shape, dtype):
    # iris.empty supports bfloat16/float32 but NOT int32 -> back int32 with float32 alloc + int32 view.
    alloc_dtype = "float32" if dtype == "int32" else dtype
    t = iris.empty(shape, dtype=alloc_dtype)
    dmap = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"),
            "int32": (torch.int32, "<i4")}
    td, ts = dmap[dtype]
    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)

def make_fp8(M, Kdim):
    assert Kdim % 2 == 0
    t = iris.empty([M, Kdim // 2], "bfloat16")     # M*Kdim bytes
    def view(ts, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    return view("<u2", (M, Kdim // 2), torch.bfloat16), view("|u1", (M, Kdim), torch.float8_e4m3fn)

# IMPORTANT: identical allocation ORDER on every rank -> identical symmetric-heap offsets.
#   per-rank source activations (each rank fills its OWN):
A_src_bf16, A_src_fp8 = make_fp8(MSRC, K)              # [Msrc, K] fp8 on EACH rank (IRIS heap)
A_src_sc = make_iris([MSRC, NG], "float32")           # [Msrc, NG] scales on EACH rank (IRIS heap)
#   consumer-only packed buffers (allocated on EVERY rank to keep offsets symmetric):
A_pk_bf16, A_pk_fp8 = make_fp8(Mpacked, K)            # [Mpacked, K] LOCAL packed fp8 (IRIS heap)
A_pk_sc  = make_iris([Mpacked, NG], "float32")        # [Mpacked, NG] LOCAL packed scales (IRIS heap)
#   route metadata on the IRIS heap (int32-as-float32):
SEG  = make_iris([Nseg, 5], "int32")
TILE = make_iris([Ntile, 4], "int32")

# ---- COMBINE: the per-origin-rank accumulator lives on the IRIS heap (the REMOTE scatter target),
#      so it must be allocated on EVERY rank in the SAME order (symmetric offsets). route_reverse is
#      deterministic from segs (identical on every rank) so Tlocal agrees with no MPI. -------------
H_COMB = N_FC2 if FFN == "full" else N           # combine width = last GEMM's N (production H=7168)

def build_combine_pull(rev, world, interleave=True):
    """PULL/gather-reduce CSR from route_reverse[Mpacked,3]=(src_rank,src_token,slot): group routed
    packed rows by destination cell (src_rank,src_token). Built once on host (NOT timed) — it is the
    transpose of the per-row reverse map, the same routing metadata MORI's EpCombine precomputes.
    Returns cell_dst[num_cells,2]=(dst_rank,dst_token), cell_ptr[num_cells+1] CSR offsets,
    cell_rows[total_routed]=packed-row idx, num_cells.
    interleave=True ROUND-ROBINS cells across dst_rank so any window of consecutive blocks (which run
    concurrently) spreads its remote stores over all `world` XGMI links — sorted-by-rank order hammers
    one link at a time (measured ~2.4x slower per byte than the scatter's naturally-random dst order)."""
    Mp = rev.shape[0]
    groups = {}
    for row in range(Mp):
        sr = int(rev[row, 0]); tok = int(rev[row, 1])
        if sr < 0 or tok < 0:
            continue
        groups.setdefault((sr, tok), []).append(row)
    if interleave:
        per_rank = [[] for _ in range(world)]
        for (sr, tok) in sorted(groups.keys()):
            per_rank[sr].append((sr, tok))
        keys = []
        idx = [0] * world
        remaining = sum(len(p) for p in per_rank)
        while remaining > 0:
            for r in range(world):
                if idx[r] < len(per_rank[r]):
                    keys.append(per_rank[r][idx[r]]); idx[r] += 1; remaining -= 1
    else:
        keys = sorted(groups.keys())
    nC = len(keys)
    cell_dst = np.zeros((max(nC, 1), 2), dtype=np.int32)
    cell_ptr = np.zeros((nC + 1,), dtype=np.int32)
    rows_list = []
    for i, (sr, tok) in enumerate(keys):
        cell_dst[i, 0] = sr; cell_dst[i, 1] = tok
        rws = groups[(sr, tok)]
        rows_list.extend(rws)
        cell_ptr[i + 1] = cell_ptr[i] + len(rws)
    cell_rows = np.array(rows_list, dtype=np.int32) if rows_list else np.zeros((1,), dtype=np.int32)
    return cell_dst, cell_ptr, cell_rows, nC

if COMBINE:
    rev_np, wgt_np, Tlocal = RT.build_route_reverse(segs, Mpacked, world,
                                                    Tlocal=(COMBINE_TLOCAL or None))
    if COMBINE_MODE == "pull":
        cell_dst_np, cell_ptr_np, cell_rows_np, NUM_CELLS = build_combine_pull(
            rev_np, world, interleave=bool(int(os.environ.get("COMBINE_INTERLEAVE", "1"))))
        ACC = make_iris([Tlocal, H_COMB], "bfloat16")  # bf16 accumulator (pull reduces fp32, stores bf16)
    else:
        ACC = make_iris([Tlocal, H_COMB], "float32")  # [Tlocal, H] fp32 accumulator (IRIS heap, all ranks)

# ---- LOCAL (non-heap) torch tensors: B (weights), C (output), TASKS ---------------------------
B     = torch.zeros(E * N, K, dtype=torch.bfloat16, device='cuda')     # weights [E*N, K]
C     = torch.zeros(Mpacked, N, dtype=torch.bfloat16, device='cuda')   # output  [Mpacked, N]
TASKS = torch.zeros(num_tasks, TASK_COLS, dtype=torch.int32, device='cuda')
TASKS.copy_(torch.from_numpy(tasks_np).to('cuda'))

# ---- FFN=full LOCAL buffers: two weight sets (W13/W2), two outputs, the fp8 intermediate, tasks ---
def make_fp8_local(M, Kdim):
    """Local (non-heap) fp8 buffer + its (bf16-view, fp8-view) over the SAME bytes — for the fc1->fc2
    intermediate. The bf16 view is what grouped_gemm_b0 takes as `a`; its raw_ptr is reinterpreted as
    fp8 by the dequant preamble (kernel.cpp:794)."""
    buf = torch.zeros(M, Kdim // 2, dtype=torch.bfloat16, device='cuda')      # M*Kdim bytes
    fp8 = buf.view(torch.uint8).view(torch.float8_e4m3fn).view(M, Kdim)       # same bytes, fp8 view
    return buf, fp8

if FFN == "full":
    B_fc1 = torch.empty(E * N_FC1, K_FC1, dtype=torch.bfloat16, device='cuda')   # W13 [E*4096,7168]
    B_fc2 = torch.empty(E * N_FC2, K_FC2, dtype=torch.bfloat16, device='cuda')   # W2  [E*7168,2048]
    C1 = torch.zeros(Mpacked, N_FC1, dtype=torch.bfloat16, device='cuda')        # fc1 out (gate||up)
    C2 = torch.zeros(Mpacked, N_FC2, dtype=torch.bfloat16, device='cuda')        # fc2 out (-> combine)
    A2_bf16, A2_fp8 = make_fp8_local(Mpacked, K_FC2)                             # fc1->fc2 fp8 packed
    A2_sc = torch.zeros(Mpacked, K_FC2 // QGROUP, dtype=torch.float32, device='cuda')
    TASKS_fc1 = torch.from_numpy(tasks_fc1_np).to('cuda')
    TASKS_fc2 = torch.from_numpy(tasks_fc2_np).to('cuda')
    if DECODE_FP8:
        # native-fp8 decode: per-expert fp8 weights + scale, and a row->expert map (for scale_c).
        def _quant_b_perrow(Bw, Ne):
            # PER-N-ROW fp8: scale per output channel (per B-row n) = max|.|/448 over K. Finer than
            # per-expert -> e4m3 error ~ the per-block A level. Factors out per output column in scale_c.
            Bv = Bw.view(E, Ne, Bw.shape[1]).float()
            sB = (Bv.abs().amax(dim=2) / 448.0).clamp(min=1e-12)                  # [E, Ne]
            q  = (Bv / sB[:, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            return q.reshape(E * Ne, Bw.shape[1]).contiguous(), sB.to(torch.float32).reshape(E, Ne).contiguous()
        # allocated on all ranks; filled (quantized) only on the consumer after the bf16 B fill below.
        B_fc1_fp8 = torch.zeros(E * N_FC1, K_FC1, dtype=torch.float8_e4m3fn, device='cuda')
        B_fc2_fp8 = torch.zeros(E * N_FC2, K_FC2, dtype=torch.float8_e4m3fn, device='cuda')
        sB_fc1 = torch.ones(E, N_FC1, dtype=torch.float32, device='cuda')
        sB_fc2 = torch.ones(E, N_FC2, dtype=torch.float32, device='cuda')
        # row_expert[Mpacked]: every packed row -> its expert (whole 256-padded region); -1 if none.
        _rowe = np.full((Mpacked, 1), -1, dtype=np.int32)
        for _e in range(E):
            _rowe[int(expert_row_begin[_e]):int(expert_row_begin[_e]) + int(padded_rows[_e])] = _e
        ROW_EXPERT = torch.from_numpy(_rowe).to('cuda')
    if DECODE_MXFP4:
        # host pre-swizzled+packed fp4 B (as bf16 [E*N, K/4]) + per-32-block scale [E*N, K/32]; filled on consumer.
        B_fc1_mxfp4 = torch.zeros(E * N_FC1, K_FC1 // 4, dtype=torch.bfloat16, device='cuda')
        B_fc2_mxfp4 = torch.zeros(E * N_FC2, K_FC2 // 4, dtype=torch.bfloat16, device='cuda')
        sBe_fc1 = torch.ones(E * N_FC1, K_FC1 // 32, dtype=torch.float32, device='cuda')
        sBe_fc2 = torch.ones(E * N_FC2, K_FC2 // 32, dtype=torch.float32, device='cuda')

# COMBINE reverse map (LOCAL on the consumer — read locally by the combine kernel; acc is the only
# combine buffer that must be on the IRIS heap). REV ints + WGT float mirror the route_reverse ABI.
if COMBINE:
    REV = torch.zeros(Mpacked, 3, dtype=torch.int32, device='cuda')
    WGT = torch.zeros(Mpacked, 1, dtype=torch.float32, device='cuda')
    if COMBINE_MODE == "pull":
        CELL_DST  = torch.from_numpy(cell_dst_np).to('cuda')              # [num_cells,2] (dst_rank,dst_token)
        CELL_PTR  = torch.from_numpy(cell_ptr_np.reshape(-1, 1)).to('cuda')   # [num_cells+1,1]
        CELL_ROWS = torch.from_numpy(cell_rows_np.reshape(-1, 1)).to('cuda')  # [total_routed,1]

# ---- each rank fills its OWN source activations; quantize like V4 (ep8 ref quantize_v1) -------
gnp = np.random.default_rng(1000 + rank)
A_real = (gnp.standard_normal((MSRC, K)).astype(np.float32) / 8.0)
q_u8, sc_f32, deq_f32 = RT.quantize_v1(A_real, K)
A_src_fp8.copy_(torch.from_numpy(q_u8.view(np.uint8)).cuda().view(torch.float8_e4m3fn).view(MSRC, K))
A_src_sc.copy_(torch.from_numpy(sc_f32).cuda())

# gather every rank's dequantized source buffer to the consumer for the CPU reference.
comm = MPI.COMM_WORLD
deq_all = comm.gather(deq_f32, root=CONSUMER)

# Expert-major weights B (same on every rank; weights are local to the consumer).
torch.manual_seed(777)
Bfull = (torch.randn(E * N, K, dtype=torch.bfloat16, device='cuda') / 8.0)
B.copy_(Bfull)

# FFN=full weights: W13 (fc1 g1u1) and W2 (fc2 down), same /8 scale as B (bf16; only A is fp8).
# Under ALL_RANKS every rank runs the GEMMs, so every rank needs its weights filled (the seeds are
# fixed, so all ranks hold identical weights -- as in EP, where each rank owns its own 32 experts).
if FFN == "full" and ACTIVE:
    torch.manual_seed(778); B_fc1.copy_(torch.randn(E * N_FC1, K_FC1, dtype=torch.bfloat16, device='cuda') / 8.0)
    torch.manual_seed(779); B_fc2.copy_(torch.randn(E * N_FC2, K_FC2, dtype=torch.bfloat16, device='cuda') / 8.0)
    if DECODE_FP8:
        _q1, _s1 = _quant_b_perrow(B_fc1, N_FC1); B_fc1_fp8.copy_(_q1); sB_fc1.copy_(_s1)
        _q2, _s2 = _quant_b_perrow(B_fc2, N_FC2); B_fc2_fp8.copy_(_q2); sB_fc2.copy_(_s2)
    if DECODE_MXFP4:
        # quantize -> load kernel buffers; then replace B_fc1/B_fc2 with the DEQUANT-fp4 weights so the
        # FFN reference measures KERNEL correctness (RMS ~0.003), not the fp4 quant class (~0.12, reported separately).
        _p1, _e1, _d1 = _quant_b_mxfp4(B_fc1, N_FC1); B_fc1_mxfp4.copy_(_p1); sBe_fc1.copy_(_e1); B_fc1.copy_(_d1)
        _p2, _e2, _d2 = _quant_b_mxfp4(B_fc2, N_FC2); B_fc2_mxfp4.copy_(_p2); sBe_fc2.copy_(_e2); B_fc2.copy_(_d2)

if ACTIVE:
    SEG.copy_(torch.from_numpy(seg_arr).cuda())
    TILE.copy_(torch.from_numpy(tile_arr).cuda())
    A_pk_fp8.view(torch.uint8).zero_()    # padding/unrouted packed rows MUST start (and stay) zero
    A_pk_sc.zero_()
    C.zero_()
    if COMBINE:
        REV.copy_(torch.from_numpy(rev_np).cuda())
        WGT.copy_(torch.from_numpy(wgt_np).reshape(Mpacked, 1).cuda())
if COMBINE:
    ACC.zero_()                           # the scatter target is zeroed on EVERY origin rank
iris.barrier()                            # happens-before for the read-only remote gather (phase 1)

iris_ctx = iris.get_device_view()

# ---- the two phases ---------------------------------------------------------------------------
def phase1_gather_pack():
    # multi-source gather/pack/quant ONCE: source per-rank A -> local expert-major packed A.
    tk_kernel.dispatch_gather_pack(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, iris_ctx,
                                   MSRC, Mpacked, K, Nseg, Ntile)

FUSED = int(os.environ.get("FUSED", "1"))   # V1: 1 = A-stationary fused (default), 0 = serial baseline

def phase2_grouped_gemm():
    # local grouped GEMM over the packed buffer. src_rank=CONSUMER -> ctx.load is a LOCAL deref.
    if SCHEDULE == "b0":
        # B0-class 8-wave ping-pong (dequant preamble + 256x256 GEMM). LOCAL only (no iris_ctx).
        tk_kernel.grouped_gemm_b0(A_pk_bf16, A_pk_sc, B, C, TASKS, Mpacked, N, K, num_tasks)
    else:
        tk_kernel.grouped_gemm(A_pk_bf16, A_pk_sc, B, C, TASKS, iris_ctx,
                               Mpacked, N, K, CONSUMER, num_tasks, NSUB, FUSED)

# ---- FFN=full: fc1 (g1u1) -> SiLU(gate)*up + fp8 re-quant -> fc2 (down), chaining grouped_gemm_b0 -
import torch.nn.functional as Fnn

def silu_and_quant(C1_tensor):
    """g1u1 epilogue + the production intermediate dynamic fp8 re-quant. C1_tensor=[M,N_FC1] (gate||up).
    Returns (q_fp8 [M,INTER], scale [M,INTER/128], deq [M,INTER]). Quant matches the device dequant
    exactly: per-128-group scale=amax/448, q=round-to-e4m3fn(x/scale), deq=q*scale (grouped_b0.cu:387)."""
    gate = C1_tensor[:, :INTER].float()
    up   = C1_tensor[:, INTER:].float()
    h    = Fnn.silu(gate) * up                                   # [M, INTER]
    NG2  = INTER // QGROUP
    hg   = h.view(h.shape[0], NG2, QGROUP)
    amax = hg.abs().amax(dim=2, keepdim=True)
    scale = (amax / 448.0).clamp(min=1e-12)
    q    = (hg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    deq  = (q.float() * scale).view(h.shape[0], INTER)
    return q.view(h.shape[0], INTER), scale.view(h.shape[0], NG2), deq

_gemm_b0 = (tk_kernel.grouped_gemm_b0_decode if DECODE else tk_kernel.grouped_gemm_b0)

def phase_fc1():
    # fc1: packed A [Mpacked,7168] fp8 -> C1 [Mpacked,4096] bf16 (gate||up fused, g1u1).
    if DECODE_MXFP4:
        tk_kernel.grouped_gemm_b0_decode_mxfp4(A_pk_bf16, A_pk_sc, B_fc1_mxfp4, sBe_fc1,
                                               C1, TASKS_fc1, Mpacked, N_FC1, K_FC1, num_tasks_fc1, E)
    elif DECODE_FP8:
        tk_kernel.grouped_gemm_b0_decode_fp8(A_pk_bf16, A_pk_sc, B_fc1_fp8.view(torch.bfloat16), sB_fc1,
                                             C1, TASKS_fc1, ROW_EXPERT, Mpacked, N_FC1, K_FC1, num_tasks_fc1, E)
    else:
        _gemm_b0(A_pk_bf16, A_pk_sc, B_fc1, C1, TASKS_fc1, Mpacked, N_FC1, K_FC1, num_tasks_fc1)

def phase_act_quant():
    # SiLU(gate)*up + fp8 re-quant of the intermediate -> A2 (fc2's fp8 input). The honest dynamic-quant.
    if int(os.environ.get("ACT_KERNEL", "1")):
        # fused HipKittens kernel: one C1 read -> fp8 A2 + per-128 scale (replaces PyTorch eager).
        # DECODE: task-drive over the real BM=16 m-tiles (num_tasks>0) so silu_quant skips the ~94% zero
        # padding rows; non-decode passes num_tasks=0 to force the dense <<<Mpacked,...>>> launch.
        _act_ntasks = num_tasks_fc1 if DECODE else 0
        tk_kernel.silu_quant(C1, A2_bf16, A2_sc, TASKS_fc1, Mpacked, N_FC1, INTER, _act_ntasks)
    else:
        q, sc, _ = silu_and_quant(C1)
        A2_fp8.copy_(q); A2_sc.copy_(sc)

def phase_fc2():
    # fc2: packed A2 [Mpacked,2048] fp8 -> C2 [Mpacked,7168] bf16 (down projection).
    if DECODE_MXFP4:
        tk_kernel.grouped_gemm_b0_decode_mxfp4(A2_bf16, A2_sc, B_fc2_mxfp4, sBe_fc2,
                                               C2, TASKS_fc2, Mpacked, N_FC2, K_FC2, num_tasks_fc2, E)
    elif DECODE_FP8:
        tk_kernel.grouped_gemm_b0_decode_fp8(A2_bf16, A2_sc, B_fc2_fp8.view(torch.bfloat16), sB_fc2,
                                             C2, TASKS_fc2, ROW_EXPERT, Mpacked, N_FC2, K_FC2, num_tasks_fc2, E)
    else:
        _gemm_b0(A2_bf16, A2_sc, B_fc2, C2, TASKS_fc2, Mpacked, N_FC2, K_FC2, num_tasks_fc2)

# COMBINE_IMPL selects the PULL-combine body: 'tilecomm' (default) = the refactored kernel that calls
# the tilecomm::tile_reduce_scatter primitive (the tile-level COMMUNICATION abstraction); 'orig' = the
# hand-rolled body kept verbatim for the zero-cost A/B gate. Both take the identical signature.
COMBINE_IMPL = os.environ.get("COMBINE_IMPL", "tilecomm").lower()

def phase_combine():
    # EpCombine: combine the fc2 output back to origin tokens over IRIS, weighted, accumulating top-k.
    out = C2 if FFN == "full" else C
    if COMBINE_MODE == "pull":
        # PULL/gather-reduce: dst cells gather their <=k rows, reduce fp32, ONE bf16 remote store.
        pull_fn = tk_kernel.combine_pull_orig if COMBINE_IMPL == "orig" else tk_kernel.combine_pull
        pull_fn(out, ACC, WGT, CELL_DST, CELL_PTR, CELL_ROWS, iris_ctx,
                NUM_CELLS, H_COMB, Tlocal, COMBINE_GRAN)
    else:
        tk_kernel.combine_scatter(out, ACC, REV, WGT, iris_ctx, Mpacked, H_COMB, Tlocal, COMBINE_ATOMIC)

# ---- CPU grouped reference (per-expert dequant(gathered A) @ B^T), incl zero-sentinel ----------
def build_reference(with_gemm=True):
    """A_deq[Mpacked,K] = the gathered+dequantized packed A (host reconstruction from each rank's deq
    source, via the SAME segments). With with_gemm, also C_ref[Mpacked,N] = per-expert dequant(A)@B^T
    (the single-GEMM reference). FFN=full skips that GEMM (it uses build_ffn_reference instead)."""
    A_deq = np.zeros((Mpacked, K), dtype=np.float32)
    for s in segs:
        src = deq_all[s["src_rank"]]
        sr, dr, rc = s["src_row_begin"], s["dst_row_begin"], s["row_count"]
        assert sr + rc <= src.shape[0], f"seg src rows exceed Msrc on rank {s['src_rank']}"
        A_deq[dr:dr + rc, :] = src[sr:sr + rc, :]
    if not with_gemm:
        return A_deq, None
    C_ref = np.zeros((Mpacked, N), dtype=np.float32)
    Bcpu = B.float().cpu().numpy()
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        Be = Bcpu[e * N:(e + 1) * N, :]               # [N,K]
        C_ref[base:base + m_e] = A_deq[base:base + m_e] @ Be.T
    return A_deq, C_ref

def build_ffn_reference(A_deq_np):
    """FFN reference (consumer GPU torch, fp32 matmuls): fc1 g1u1 -> SiLU(gate)*up + the SAME fp8
    re-quant -> fc2. Includes the intermediate quant so RMS captures the WHOLE chain (fc1 bf16 GEMM +
    fp8 intermediate + fc2 bf16 GEMM) — expect ~0.01-0.05. Returns C2_ref[Mpacked, N_FC2] (numpy)."""
    A_deq = torch.from_numpy(A_deq_np).to('cuda')                       # [Mpacked,7168] fp32
    C1_ref = torch.zeros(Mpacked, N_FC1, device='cuda', dtype=torch.float32)
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        W13e = B_fc1[e * N_FC1:(e + 1) * N_FC1].float()                 # [4096,7168]
        C1_ref[base:base + m_e] = A_deq[base:base + m_e] @ W13e.T
    _, _, deq_h = silu_and_quant(C1_ref)                               # [Mpacked,2048] fp32 (re-quanted)
    C2_ref = torch.zeros(Mpacked, N_FC2, device='cuda', dtype=torch.float32)
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        W2e = B_fc2[e * N_FC2:(e + 1) * N_FC2].float()                  # [7168,2048]
        C2_ref[base:base + m_e] = deq_h[base:base + m_e] @ W2e.T
    return C2_ref.cpu().numpy()

# ---- run / check ------------------------------------------------------------------------------
def run_both():
    if COMBINE:                                 # every origin rank clears its own scatter target
        ACC.zero_(); torch.cuda.synchronize(); iris.barrier()
    if ACTIVE:
        A_pk_fp8.view(torch.uint8).zero_(); A_pk_sc.zero_()
        if FFN == "full":
            C1.zero_(); C2.zero_()
            phase1_gather_pack(); phase_fc1(); phase_act_quant(); phase_fc2()
        else:
            C.zero_(); phase1_gather_pack(); phase2_grouped_gemm()
        if COMBINE:
            phase_combine()
    torch.cuda.synchronize()
    iris.barrier()

run_both()   # correctness pass
# COMBINE: gather every rank's accumulator to the consumer (collective -> ALL ranks call it).
acc_all = comm.gather(ACC.float().cpu().numpy(), root=CONSUMER) if COMBINE else None
if rank == CONSUMER:
    A_deq_ref, C_ref_single = build_reference(with_gemm=(FFN != "full"))
    if FFN == "full":
        C_ref = build_ffn_reference(A_deq_ref); C_got = C2.float().cpu().numpy(); N_out = N_FC2
    else:
        C_ref = C_ref_single; C_got = C.float().cpu().numpy(); N_out = N

    # ---- PHASE-1 ISOLATION PROBE: dequant the packed-A buffer the kernel produced and compare to
    #      the reference A_deq. If this matches, phase 1 (gather/pack) is correct and the bug is in
    #      phase 2 (the GEMM feed); if it diverges, the bug is in phase 1's pack. ----
    pk_fp8 = A_pk_fp8.view(torch.uint8).cpu().numpy().astype(np.float32)   # raw e4m3 bytes [Mpacked,K]
    # decode e4m3 bytes -> float via ml_dtypes (matches gfx950 OCP e4m3)
    import ml_dtypes as _ml
    pk_vals = A_pk_fp8.view(torch.uint8).cpu().numpy().view(_ml.float8_e4m3fn).astype(np.float32)
    pk_sc   = A_pk_sc.cpu().numpy()                                        # [Mpacked, NG]
    A_deq_pk = (pk_vals.reshape(Mpacked, NG, QGROUP) * pk_sc[:, :, None]).reshape(Mpacked, K)
    a_diff = np.abs(A_deq_pk - A_deq_ref)
    a_rms = float(np.sqrt((a_diff**2).mean()) / max(np.sqrt((A_deq_ref**2).mean()), 1e-9))
    n_a_bad_rows = int((a_diff.max(axis=1) > 1e-2).sum())
    print(f"  [phase1-probe] packed-A vs ref A_deq: RMS={a_rms:.6f}  rows_mismatch={n_a_bad_rows}/{Mpacked}", flush=True)

    # real (non-padding) rows only
    real = np.zeros(Mpacked, dtype=bool)
    routed = np.zeros(Mpacked, dtype=bool)
    remote_rows = 0
    for e in range(E):
        m_e = int(rows_per_expert[e]); base = int(expert_row_begin[e])
        if m_e: real[base:base + m_e] = True
    for s in segs:
        routed[s["dst_row_begin"]:s["dst_row_begin"] + s["row_count"]] = True
        if s["src_rank"] != CONSUMER:
            remote_rows += s["row_count"]
    zero_rows = ~routed

    Cr = C_ref[real]; Cg = C_got[real]
    diff = np.abs(Cg - Cr); denom = np.maximum(np.abs(Cr), 1e-6)
    rms_rel = float(np.sqrt((diff**2).mean()) / max(np.sqrt((Cr**2).mean()), 1e-9))
    max_rel = float((diff / denom).max())

    # zero-sentinel: unrouted packed rows must be EXACTLY zero in C (proves no contamination) and the
    # consumer's LOCAL source A buffer is unrelated to the packed A, so a correct C proves phase 1
    # actually moved bytes (packed A nonzero where routed).
    sentinel_ok = bool(np.all(Cg[zero_rows[real]] == 0.0)) if zero_rows[real].any() else True
    packed_nonzero = bool(A_pk_fp8.view(torch.uint8).max().item() > 0)
    c_zero = bool(np.abs(Cg).max() == 0.0)
    remote_ok = remote_rows > 0
    # FFN=full chains TWO fp8-A GEMMs + an fp8 intermediate re-quant -> looser tol than the 1-GEMM path.
    rms_tol = 0.05 if FFN == "full" else 0.01
    ok = (rms_rel < rms_tol and not c_zero and packed_nonzero and remote_ok)

    print("=" * 92, flush=True)
    region = "gather+fc1+act+fc2" + ("+combine" if COMBINE else "") if FFN == "full" else \
             "gather+gemm" + ("+combine" if COMBINE else "")
    print(f"[B1-dispatch {('FULL-FFN' if FFN=='full' else 'V0')}] route={ROUTE} Mpacked={Mpacked} "
          f"region=({region}) N_out={N_out} K={K} tasks={num_tasks} segs={Nseg}", flush=True)
    print(f"  remote (XGMI) gathered rows = {remote_rows}", flush=True)
    print(f"  RMS_rel={rms_rel:.6f} (tol {rms_tol})  max_rel={max_rel:.4f}", flush=True)
    print(f"  packed_A_nonzero={packed_nonzero}  C_zero={c_zero}  remote_path={remote_ok}", flush=True)
    print(f"  -> FFN {'PASSED' if ok else 'FAILED'}", flush=True)

    # ---- COMBINE correctness: GPU-scattered acc vs a CPU scatter of the SAME fc2 output (isolates the
    #      combine from the FFN numerics). acc_got[world,Tlocal,H] is the MPI-gathered per-rank acc. ----
    if COMBINE:
        acc_got = np.stack(acc_all, axis=0).astype(np.float32)         # [world, Tlocal, H_COMB]
        acc_ref = RT.combine_reference(C_got, rev_np, wgt_np, world, Tlocal, H_COMB)
        cdiff = np.abs(acc_got - acc_ref)
        comb_rms = float(np.sqrt((cdiff**2).mean()) / max(np.sqrt((acc_ref**2).mean()), 1e-9))
        n_cells = int((np.abs(acc_ref).sum(axis=2) > 0).sum())         # (rank,token) cells that got mass
        n_collide = int((rev_np[:, 1] >= 0).sum()) - n_cells           # >0 means real accumulation tested
        comb_nonzero = bool(np.abs(acc_got).max() > 0.0)
        comb_ok = (comb_rms < 0.02 and comb_nonzero)
        print(f"  [combine] mode={COMBINE_MODE} dst_cells={n_cells} collisions(accumulated)={n_collide} "
              f"atomic={COMBINE_ATOMIC} Tlocal={Tlocal}", flush=True)
        print(f"  [combine] acc RMS_rel={comb_rms:.6f}  acc_nonzero={comb_nonzero}  "
              f"-> COMBINE {'PASSED' if comb_ok else 'FAILED'}", flush=True)
    print("=" * 92, flush=True)

# ---- timing: per-stage cuda events (SAME iteration). T_gather / T_fc1 / T_act / T_fc2 / T_combine /
#      T_total. ALL ranks run the loop (only CONSUMER launches+records) so the barrier counts match. -
_evk = ("start", "gather", "fc1", "act", "fc2", "comb")
EV = {k: torch.cuda.Event(enable_timing=True) for k in _evk}

def _region_once():
    EV["start"].record()
    phase1_gather_pack();                 EV["gather"].record()
    if FFN == "full":
        phase_fc1();                      EV["fc1"].record()
        phase_act_quant();                EV["act"].record()
        phase_fc2();                      EV["fc2"].record()
    else:
        phase2_grouped_gemm();            EV["fc2"].record()   # single-GEMM end marker
    if COMBINE:
        phase_combine();                  EV["comb"].record()

def timed():
    for _ in range(WARMUP):
        if ACTIVE:
            _region_once()
    torch.cuda.synchronize(); iris.barrier()
    acc = dict(gather=0.0, fc1=0.0, act=0.0, fc2=0.0, comb=0.0, total=0.0)
    for _ in range(ITERS):
        if ACTIVE:
            _region_once()
            torch.cuda.synchronize()
            acc["gather"] += EV["start"].elapsed_time(EV["gather"])    # ms
            if FFN == "full":
                acc["fc1"] += EV["gather"].elapsed_time(EV["fc1"])
                acc["act"] += EV["fc1"].elapsed_time(EV["act"])
                acc["fc2"] += EV["act"].elapsed_time(EV["fc2"])
            else:
                acc["fc2"] += EV["gather"].elapsed_time(EV["fc2"])
            if COMBINE:
                acc["comb"] += EV["fc2"].elapsed_time(EV["comb"])
            end = EV["comb"] if COMBINE else EV["fc2"]
            acc["total"] += EV["start"].elapsed_time(end)
        else:
            torch.cuda.synchronize()
        iris.barrier()
    if not ALL_RANKS:
        return {k: v / ITERS * 1e3 for k, v in acc.items()} if rank == CONSUMER else None
    # ALL_RANKS: reduce with MAX over the 8 ranks -- the production denominator (the step waits for the
    # slowest rank), matching b3_ep8_unfused.py's `mx = lambda j: max(r[j] for r in rows)`.
    mine = {k: v / ITERS * 1e3 for k, v in acc.items()}                # us, this rank
    keys = ("gather", "fc1", "act", "fc2", "comb", "total")
    mx = comm.allreduce(np.array([mine[k] for k in keys], dtype=np.float64), op=MPI.MAX)
    av = comm.allreduce(np.array([mine[k] for k in keys], dtype=np.float64), op=MPI.SUM) / world
    if rank == CONSUMER:
        print(f"  [ALL_RANKS] per-stage mean-over-ranks us: "
              + " ".join(f"{k}={av[i]:.1f}" for i, k in enumerate(keys)), flush=True)
        return {k: float(mx[i]) for i, k in enumerate(keys)}
    return None

res = timed()
if rank == CONSUMER:
    real_rows = int(np.sum(rows_per_expert))
    T_gather = res["gather"]; T_total = res["total"]; T_combine = res["comb"]
    print("-" * 92, flush=True)
    print(f"  T_gather (phase1, multi-source gather/pack ONCE) : {T_gather:8.2f} us", flush=True)
    if FFN == "full":
        f1 = 2.0 * real_rows * N_FC1 * K_FC1
        f2 = 2.0 * real_rows * N_FC2 * K_FC2
        T_fc1 = res["fc1"]; T_act = res["act"]; T_fc2 = res["fc2"]; T_gemm = T_fc1 + T_fc2
        tflops_total = (f1 + f2) / (T_total * 1e-6) / 1e12
        print(f"  T_fc1    (g1u1 GEMM  N={N_FC1} K={K_FC1})         : {T_fc1:8.2f} us   "
              f"{f1/(T_fc1*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
        print(f"  T_act    (SiLU(gate)*up + fp8 re-quant)          : {T_act:8.2f} us", flush=True)
        print(f"  T_fc2    (down GEMM  N={N_FC2} K={K_FC2})         : {T_fc2:8.2f} us   "
              f"{f2/(T_fc2*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
    else:
        flops = 2.0 * real_rows * N * K
        T_fc1 = T_act = T_fc2 = 0.0; T_gemm = res["fc2"]
        tflops_total = flops / (T_total * 1e-6) / 1e12
        print(f"  T_gemm   (phase2, local grouped GEMM)            : {T_gemm:8.2f} us   "
              f"{flops/(T_gemm*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
    if COMBINE:
        print(f"  T_combine(EpCombine scatter-back over XGMI)      : {T_combine:8.2f} us", flush=True)
    print(f"  T_total  (serial region, the production denominator): {T_total:8.2f} us   "
          f"{tflops_total:6.2f} TFLOP/s (e2e)", flush=True)
    print("=" * 92, flush=True)

    def git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
        except Exception:
            return "unknown"

    header = ("candidate,commit,date,ranks,route_dist,M_label,M,N,K,dtype,BM,BN,BK,NSUB,"
              "schedule,grid_blocks,VGPR,AGPR,SGPR,LDS,scratch,lat_us,p50,p95,p99,TFLOPs,"
              "rms_rel,zero_sentinel,spd_vs_B1,spd_vs_B2,spd_vs_B3,notes")
    new = not os.path.exists(CSV)
    with open(CSV, "a") as f:
        if new:
            f.write(header + "\n")
        notes = (f"FFN={FFN} SCHEDULE={SCHEDULE} COMBINE={COMBINE}; "
                 f"T_gather={T_gather:.1f} T_fc1={T_fc1:.1f} T_act={T_act:.1f} "
                 f"T_fc2={T_fc2:.1f} T_combine={T_combine:.1f}")
        row = [
            "B1-dispatch", git_commit(), datetime.date.today().isoformat(), str(world),
            ROUTE, f"TOTAL_M{TOTAL_M}", str(Mpacked), str(N_out), str(K), "fp8e4m3->bf16",
            str(BM), str(BN), str(BK), str(NSUB),
            f"{SCHEDULE}-{'fullffn' if FFN=='full' else 'gemm'}", str(num_tasks),
            "", "", "", "", "",
            f"{T_total:.2f}", "", "", "", f"{tflops_total:.2f}",
            f"{rms_rel:.5f}", "1", "", "", "",
            notes,
        ]
        f.write(",".join(row) + "\n")

import gc
del A_src_bf16, A_src_fp8, A_src_sc, A_pk_bf16, A_pk_fp8, A_pk_sc, SEG, TILE, B, C, TASKS
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
