#!/usr/bin/env python3
# fmoe_fused_v5_grouped / example.py
# ------------------------------------------------------------------------------------------------
# V5 GROUPED head-to-head driver (np=2).  Generalizes V4 to ALL E local experts in ONE launch.
#
#   Rank 0 ("producer rank") holds the ONLY real copy of the QUANTIZED fp8 packed activations
#     A_fp8[Mpacked, K] + per-128-group fp32 scales A_sc[Mpacked, K/128] on its IRIS heap, where
#     Mpacked = sum_e padded(M_e) packs all experts' routed rows back-to-back, each expert's region
#     padded UP to a multiple of BM (the no-contamination layout).
#   Rank 1 ("consumer rank") runs the GROUPED expert GEMM: one flat task grid over all experts.
#     Its kernel PRODUCER warps pull fp8 A-tiles + scales from rank 0 over IRIS, dequantize, and the
#     CONSUMER warps MFMA -- overlapping cross-GPU gather+dequant+matmul (FUSED) vs a two-phase
#     gather-then-MFMA (BASELINE) on the SAME grouped layout.
#
# The host builds the task list + adaptive NSUB with build_tasks.py (pure CPU, no GPU).  Correctness
# is checked against a NUMPY-CPU grouped reference (each expert e: C[rows_e] = dequant(A[rows_e]) @
# B[e]^T), plus the zero-sentinel (rank-1's local A is zeros, so a correct C proves the remote
# gather is real).  Results are appended to results.csv with the canonical columns.
# ------------------------------------------------------------------------------------------------
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

import build_tasks as BT

torch.manual_seed(0)

# ---- shapes / config (override via env) ----
E       = int(os.environ.get("E", "32"))            # local experts/rank
K       = int(os.environ.get("K", "7168"))
N       = int(os.environ.get("N", "2048"))
TOTAL_M = int(os.environ.get("TOTAL_M", "8192"))    # aggregate routed rows across all experts
ROUTE   = os.environ.get("ROUTE", "uniform")        # uniform|zipf|one_hot|several_hot|many_empty
BM      = int(os.environ.get("BM", "64"))
BN      = int(os.environ.get("BN", "64"))
BK      = int(os.environ.get("BK", "64"))
ITERS   = int(os.environ.get("ITERS", "50"))
WARMUP  = int(os.environ.get("WARMUP", "10"))
CSV     = os.environ.get("CSV", "results.csv")
QGROUP  = 128
assert K % QGROUP == 0
NG = K // QGROUP

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 2, f"this bring-up expects np=2, got world={world}"
torch.cuda.set_device(rank)
SRC_RANK = 0

# ---- host: build the route + grouped schedule (deterministic across ranks) ----
rng = np.random.default_rng(20260625)
rows_per_expert = BT.ROUTE_BUILDERS[ROUTE](E, TOTAL_M, rng)
sched = BT.build_grouped_schedule(rows_per_expert, N, BM, BN)
NSUB             = sched["nsub"]
tasks_np         = sched["tasks"]                    # [num_tasks, 6] int32
expert_row_begin = sched["expert_row_begin"]
padded_rows      = sched["padded_rows"]
Mpacked          = sched["total_padded_rows"]
num_tasks        = sched["num_tasks"]
n_blocks         = sched["n_blocks"]
assert Mpacked > 0 and num_tasks > 0, "empty route -- nothing to launch"

if rank != SRC_RANK:
    print(f"[grouped] route={ROUTE} E={E} TOTAL_M={TOTAL_M} -> Mpacked={Mpacked} "
          f"NSUB={NSUB} num_tasks={num_tasks} n_blocks={n_blocks} "
          f"rows[0:6]={list(rows_per_expert[:6])}", flush=True)

# ---- IRIS symmetric-heap tensors (identical allocation ORDER on both ranks) ----
def make_iris_tensor(shape, dtype):
    t = iris.empty(shape, dtype=dtype)
    dtype_map = {"bfloat16": (torch.bfloat16, "<u2"),
                 "float32":  (torch.float32, "<f4"),
                 "int32":    (torch.int32,   "<i4")}
    torch_dtype, typestr = dtype_map[dtype]
    class W:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': typestr,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    w = W(t.data_ptr(), shape, typestr)
    return torch.as_tensor(w, device='cuda').view(torch_dtype).view(*shape)

def make_fp8_iris_tensor(M, K):
    assert K % 2 == 0
    t = iris.empty([M, K // 2], "bfloat16")              # M*K bytes
    def view_as(typestr, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': typestr,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    bf16_view = view_as("<u2", (M, K // 2), torch.bfloat16)
    fp8_view  = view_as("|u1", (M, K), torch.float8_e4m3fn)
    return bf16_view, fp8_view

# allocate in identical order on both ranks.
# ONLY A (gathered remotely) needs the IRIS symmetric heap. B (weights) and C (output) are LOCAL to
# the consumer rank -> plain torch tensors (putting 3GB of B on the 512MB heap OOMs and is pointless;
# the kernel reads B/C via a gl built from any device pointer).
A_fp8_bf16, A_fp8 = make_fp8_iris_tensor(Mpacked, K)     # packed activations (padded) - IRIS heap
A_sc   = make_iris_tensor([Mpacked, NG], "float32")      # scales - IRIS heap
B      = torch.zeros(E * N, K, dtype=torch.bfloat16, device='cuda')   # LOCAL weights B[E*N, K]
C      = torch.zeros(Mpacked, N, dtype=torch.bfloat16, device='cuda') # LOCAL output
TASKS  = torch.zeros(num_tasks, BT.TASK_W, dtype=torch.int32, device='cuda')  # LOCAL task list (consumer-only; IRIS has no int32)
TASKS.copy_(torch.from_numpy(tasks_np).to('cuda'))

# ---- build reference data on the host (numpy/torch CPU+GPU), pack with BM padding ----
torch.manual_seed(1234)

def quantize_v1(A_real_2d, m):
    """V1 quant: per-128-group scale = amax/448, q = round-to-fp8(x/scale).  A_real_2d:[m,K]."""
    Ag = A_real_2d.view(m, NG, QGROUP)
    amax = Ag.abs().amax(dim=2, keepdim=True)
    scale = (amax / 448.0).clamp_min(1e-12)
    q = (Ag / scale).to(torch.float8_e4m3fn)
    return q.view(m, K), scale.view(m, NG).contiguous()

# Build per-expert real activations, quantize, and scatter into the PADDED packed buffer.
# A_deq_packed is the bf16-dequantized A the kernel reconstructs (the TRUE matmul input).
A_deq_packed_host = None
if rank == SRC_RANK:
    A_fp8.view(torch.uint8).zero_()                      # padding rows stay zero
    A_sc.zero_()
    A_deq_packed = torch.zeros(Mpacked, K, dtype=torch.bfloat16, device='cuda')
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        A_real_e = (torch.randn(m_e, K, dtype=torch.float32, device='cuda') / 8.0)
        qa, qs = quantize_v1(A_real_e, m_e)
        A_fp8[base:base + m_e].copy_(qa)
        A_sc[base:base + m_e].copy_(qs)
        deq = (qa.to(torch.float32).view(m_e, NG, QGROUP) * qs.view(m_e, NG, 1)).view(m_e, K)
        A_deq_packed[base:base + m_e].copy_(deq.to(torch.bfloat16))
    A_deq_packed_host = A_deq_packed.float().cpu().numpy()
else:
    A_fp8.view(torch.uint8).zero_()                      # SENTINEL: rank1 local A = zeros
    A_sc.zero_()

# Expert-major B (same on both ranks; weights are local to consumer rank).
torch.manual_seed(777)
Bfull = (torch.randn(E * N, K, dtype=torch.bfloat16, device='cuda') / 8.0)
B.copy_(Bfull)
C.zero_()
iris.barrier()

iris_ctx = iris.get_device_view()

def launch(fused):
    tk_kernel.dispatch_micro(A_fp8_bf16, A_sc, B, C, TASKS, iris_ctx,
                             Mpacked, N, K, SRC_RANK, num_tasks, NSUB, int(fused))

def run(fused):
    C.zero_()
    if rank != SRC_RANK:
        launch(fused)
    torch.cuda.synchronize()
    iris.barrier()

def timed(fused):
    for _ in range(WARMUP):
        if rank != SRC_RANK:
            launch(fused)
    torch.cuda.synchronize(); iris.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        if rank != SRC_RANK:
            launch(fused)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e6     # us/iter
    iris.barrier()
    return dt

# ---- NUMPY-CPU grouped reference (broadcast dequant A + B to consumer rank) ----
comm = MPI.COMM_WORLD
A_deq_packed_host = comm.bcast(A_deq_packed_host, root=SRC_RANK)

def grouped_reference():
    """Per-expert C[rows_e] = dequant(A[rows_e]) @ B[e]^T; padding rows expected zero."""
    A_ref = torch.from_numpy(A_deq_packed_host).to('cuda').to(torch.bfloat16)
    C_ref = torch.zeros(Mpacked, N, dtype=torch.float32, device='cuda')
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        Be = B[e * N:(e + 1) * N].float()               # [N,K]
        C_ref[base:base + m_e] = torch.matmul(A_ref[base:base + m_e].float(), Be.t())
    return C_ref

def check(fused, label):
    run(fused)
    if rank == SRC_RANK:
        return None
    C_ref = grouped_reference()
    C_got = C.float()
    # Only compare REAL (non-padding) rows; padding C rows are dead space.
    real_mask = torch.zeros(Mpacked, dtype=torch.bool, device='cuda')
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        real_mask[base:base + m_e] = True
    Cr = C_ref[real_mask]
    Cg = C_got[real_mask]
    diff = (Cg - Cr).abs()
    denom = Cr.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    rms_rel = (diff.pow(2).mean().sqrt() / Cr.pow(2).mean().sqrt()).item()
    max_abs = diff.max().item()
    c_zero = bool(Cg.abs().max().item() == 0.0)
    a_zero = bool(A_fp8.view(torch.uint8).max().item() == 0)     # local A must be all zeros
    ok = (rms_rel < 0.10 and not c_zero and a_zero)
    print(f"[{label}] route={ROUTE} Mpacked={Mpacked} N={N} K={K} NSUB={NSUB} tasks={num_tasks}  "
          f"max_abs={max_abs:.4f} max_rel={max_rel:.4f} RMS_rel={rms_rel:.5f} "
          f"local_A_zero={a_zero} C_zero={c_zero} -> {'PASSED' if ok else 'FAILED'}", flush=True)
    return rms_rel, a_zero

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__))
                                       ).decode().strip()
    except Exception:
        return "unknown"

def append_csv(schedule, lat_us, tflops, rms_rel, zero_sentinel):
    if rank == SRC_RANK:
        return
    header = ("candidate,commit,date,ranks,route_dist,M_label,M,N,K,dtype,BM,BN,BK,NSUB,"
              "schedule,grid_blocks,VGPR,AGPR,SGPR,LDS,scratch,lat_us,p50,p95,p99,TFLOPs,"
              "rms_rel,zero_sentinel,spd_vs_B1,spd_vs_B2,spd_vs_B3,notes")
    new = not os.path.exists(CSV)
    with open(CSV, "a") as f:
        if new:
            f.write(header + "\n")
        row = [
            "v5_grouped", git_commit(), datetime.date.today().isoformat(), str(world),
            ROUTE, f"TOTAL_M{TOTAL_M}", str(Mpacked), str(N), str(K), "fp8e4m3->bf16",
            str(BM), str(BN), str(BK), str(NSUB), schedule, str(num_tasks),
            "", "", "", "", "",                          # VGPR/AGPR/SGPR/LDS/scratch (compile-time)
            f"{lat_us:.2f}", "", "", "", f"{tflops:.2f}",
            f"{rms_rel:.5f}", str(int(zero_sentinel)), "", "", "",
            "grouped all-experts-in-one-grid; BM-padded no-contamination",
        ]
        f.write(",".join(row) + "\n")

if rank != SRC_RANK:
    print("=" * 90, flush=True)
res_fused = check(True,  "FUSED   ")
res_base  = check(False, "BASELINE")

t_fused = timed(True)
t_base  = timed(False)

if rank != SRC_RANK:
    # Real-row FLOPs only (padding rows do no useful work).
    flops = 2.0 * int(np.sum(rows_per_expert)) * N * K
    tflops_fused = flops / (t_fused * 1e-6) / 1e12
    tflops_base  = flops / (t_base  * 1e-6) / 1e12
    print("-" * 90, flush=True)
    print(f"  BASELINE (two-phase, no overlap) : {t_base:8.2f} us/iter   {tflops_base:6.2f} TFLOP/s", flush=True)
    print(f"  FUSED    (grouped A-stationary)  : {t_fused:8.2f} us/iter   {tflops_fused:6.2f} TFLOP/s", flush=True)
    spd = t_base / t_fused
    print(f"  SPEEDUP (baseline/fused)         : {spd:6.3f}x   ({'FUSED WINS' if spd>1 else 'baseline wins'})", flush=True)
    print("=" * 90, flush=True)
    rms_f, a_zero_f = res_fused
    append_csv("4P4C-grouped-fused", t_fused, tflops_fused, rms_f, a_zero_f)

import gc
del A_fp8_bf16, A_fp8, A_sc, B, C, TASKS
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
