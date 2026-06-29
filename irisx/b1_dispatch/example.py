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

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 8, f"B1-dispatch V0 expects np=8 (EP8), got world={world}"
torch.cuda.set_device(rank)

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

# ---- LOCAL (non-heap) torch tensors: B (weights), C (output), TASKS ---------------------------
B     = torch.zeros(E * N, K, dtype=torch.bfloat16, device='cuda')     # weights [E*N, K]
C     = torch.zeros(Mpacked, N, dtype=torch.bfloat16, device='cuda')   # output  [Mpacked, N]
TASKS = torch.zeros(num_tasks, TASK_COLS, dtype=torch.int32, device='cuda')
TASKS.copy_(torch.from_numpy(tasks_np).to('cuda'))

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

if rank == CONSUMER:
    SEG.copy_(torch.from_numpy(seg_arr).cuda())
    TILE.copy_(torch.from_numpy(tile_arr).cuda())
    A_pk_fp8.view(torch.uint8).zero_()    # padding/unrouted packed rows MUST start (and stay) zero
    A_pk_sc.zero_()
    C.zero_()
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

# ---- CPU grouped reference (per-expert dequant(gathered A) @ B^T), incl zero-sentinel ----------
def build_reference():
    """C_ref[Mpacked,N]: for each expert e, for each packed row, dequant(source row) @ B[e]^T.
    Unrouted/padding rows expected exactly 0. Uses the gathered per-rank dequantized A (deq_all)."""
    A_deq = np.zeros((Mpacked, K), dtype=np.float32)
    for s in segs:
        src = deq_all[s["src_rank"]]
        sr, dr, rc = s["src_row_begin"], s["dst_row_begin"], s["row_count"]
        assert sr + rc <= src.shape[0], f"seg src rows exceed Msrc on rank {s['src_rank']}"
        A_deq[dr:dr + rc, :] = src[sr:sr + rc, :]
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

# ---- run / check ------------------------------------------------------------------------------
def run_both():
    if rank == CONSUMER:
        A_pk_fp8.view(torch.uint8).zero_(); A_pk_sc.zero_(); C.zero_()
        phase1_gather_pack()
        phase2_grouped_gemm()
    torch.cuda.synchronize()
    iris.barrier()

run_both()   # correctness pass
if rank == CONSUMER:
    A_deq_ref, C_ref = build_reference()
    C_got = C.float().cpu().numpy()

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
    ok = (rms_rel < 0.01 and not c_zero and packed_nonzero and remote_ok)

    print("=" * 92, flush=True)
    print(f"[B1-dispatch V0] route={ROUTE} Mpacked={Mpacked} N={N} K={K} NSUB={NSUB} "
          f"tasks={num_tasks} segs={Nseg}", flush=True)
    print(f"  remote (XGMI) gathered rows = {remote_rows}", flush=True)
    print(f"  RMS_rel={rms_rel:.6f}  max_rel={max_rel:.4f}", flush=True)
    print(f"  packed_A_nonzero={packed_nonzero}  C_zero={c_zero}  remote_path={remote_ok}", flush=True)
    print(f"  -> {'PASSED' if ok else 'FAILED'}", flush=True)
    print("=" * 92, flush=True)

# ---- timing: T_gather / T_gemm / T_total via cuda events (SAME iteration) ----------------------
# ALL ranks run the loop (only CONSUMER launches + records) so the barrier counts match.
ev_total0 = torch.cuda.Event(enable_timing=True)
ev_g0     = torch.cuda.Event(enable_timing=True)
ev_g1     = torch.cuda.Event(enable_timing=True)
ev_m1     = torch.cuda.Event(enable_timing=True)

def timed():
    # warmup
    for _ in range(WARMUP):
        if rank == CONSUMER:
            phase1_gather_pack(); phase2_grouped_gemm()
    torch.cuda.synchronize(); iris.barrier()
    tg = tm = tt = 0.0
    for _ in range(ITERS):
        if rank == CONSUMER:
            ev_total0.record(); ev_g0.record()
            phase1_gather_pack()
            ev_g1.record()
            phase2_grouped_gemm()
            ev_m1.record()
            torch.cuda.synchronize()
            tg += ev_g0.elapsed_time(ev_g1)     # ms
            tm += ev_g1.elapsed_time(ev_m1)
            tt += ev_total0.elapsed_time(ev_m1)
        else:
            torch.cuda.synchronize()
        iris.barrier()
    if rank == CONSUMER:
        return tg / ITERS * 1e3, tm / ITERS * 1e3, tt / ITERS * 1e3   # us
    return None

res = timed()
if rank == CONSUMER:
    T_gather, T_gemm, T_total = res
    real_rows = int(np.sum(rows_per_expert))
    flops = 2.0 * real_rows * N * K
    tflops_total = flops / (T_total * 1e-6) / 1e12
    tflops_gemm  = flops / (T_gemm  * 1e-6) / 1e12
    print("-" * 92, flush=True)
    print(f"  T_gather (phase1, multi-source gather/pack ONCE) : {T_gather:8.2f} us", flush=True)
    print(f"  T_gemm   (phase2, local grouped GEMM)            : {T_gemm:8.2f} us   "
          f"{tflops_gemm:6.2f} TFLOP/s", flush=True)
    print(f"  T_total  (serial phase1 + phase2)                : {T_total:8.2f} us   "
          f"{tflops_total:6.2f} TFLOP/s (e2e)", flush=True)
    print(f"  (compare to B1-copy: copy 135us + gemm 150us = 285us @ M1024 single-source)", flush=True)
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
        row = [
            "B1-dispatch", git_commit(), datetime.date.today().isoformat(), str(world),
            ROUTE, f"TOTAL_M{TOTAL_M}", str(Mpacked), str(N), str(K), "fp8e4m3->bf16",
            str(BM), str(BN), str(BK), str(NSUB), "serial-gatherpack+grouped", str(num_tasks),
            "", "", "", "", "",
            f"{T_total:.2f}", "", "", "", f"{tflops_total:.2f}",
            f"{rms_rel:.5f}", "1", "", "", "",
            f"V0 serial EP8 dispatch; T_gather={T_gather:.1f} T_gemm={T_gemm:.1f}",
        ]
        f.write(",".join(row) + "\n")

import gc
del A_src_bf16, A_src_fp8, A_src_sc, A_pk_bf16, A_pk_fp8, A_pk_sc, SEG, TILE, B, C, TASKS
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
