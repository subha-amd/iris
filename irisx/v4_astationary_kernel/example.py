#!/usr/bin/env python3
# fmoe_fused_v3 / example.py
# ------------------------------------------------------------------------------------------------
# V3 head-to-head driver (np=2):
#   Rank 0 ("producer rank") holds the ONLY real copy of the QUANTIZED fp8 activations A_fp8[M,K]
#     and their per-128-group fp32 scales A_sc[M,K/128] on its IRIS heap.
#   Rank 1 ("consumer rank") runs the expert GEMM C = dequant(A_fp8) @ B^T.  Its kernel's PRODUCER
#     warps pull fp8 A-tiles + scales from rank 0 over IRIS, DEQUANTIZE in the producer, and the
#     CONSUMER warps MFMA — overlapping cross-GPU gather + dequant + matmul (FUSED), vs a two-phase
#     gather-then-MFMA (BASELINE).  Same shapes/layout/gather/dequant; only overlap differs.
#
# Reports: correctness (RMS-rel err vs bf16 reference) for both, and head-to-head wall-clock.
# ------------------------------------------------------------------------------------------------
import sys, os, time
sys.path.insert(0, "..")
import torch
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import iris_py
import tk_kernel
from mpi4py import MPI

torch.manual_seed(0)

# ---- shapes (override via env) ----
M = int(os.environ.get("M", "256"))
K = int(os.environ.get("K", "2048"))
N = int(os.environ.get("N", "2048"))
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))
QGROUP = 128
assert K % QGROUP == 0
NG = K // QGROUP

iris = iris_py.Iris(heap_size_mb=128, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 2, f"this bring-up expects np=2, got world={world}"
torch.cuda.set_device(rank)
SRC_RANK = 0

def make_iris_tensor(shape, dtype):
    t = iris.empty(shape, dtype=dtype)
    dtype_map = {"bfloat16": (torch.bfloat16, "<u2"),
                 "float32":  (torch.float32, "<f4"),
                 "float16":  (torch.float16, "<u2")}
    torch_dtype, typestr = dtype_map[dtype]
    class W:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': typestr,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    w = W(t.data_ptr(), shape, typestr)
    return torch.as_tensor(w, device='cuda').view(torch_dtype).view(*shape)

# fp8 buffer: iris.empty has no fp8 dtype, AND HK's gl rejects 1-byte element types, so we
# allocate M*K BYTES (as bf16 [M,K//2]) on the heap.  We keep TWO views over the SAME storage:
#   - A_fp8_bf16 [M,K//2] bf16  : the tensor PASSED to the kernel (gl<bf16> reinterprets bytes)
#   - A_fp8      [M,K]    fp8    : used only here to quantize/copy real fp8 values in
# Symmetric-heap offsets stay identical on ranks (same allocation order).
def make_fp8_iris_tensor(M, K):
    assert K % 2 == 0
    t = iris.empty([M, K // 2], "bfloat16")         # M*K bytes
    def view_as(typestr, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': typestr,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    bf16_view = view_as("<u2", (M, K // 2), torch.bfloat16)   # kernel arg
    fp8_view  = view_as("|u1", (M, K), torch.float8_e4m3fn)   # quant/copy helper
    return bf16_view, fp8_view

# IMPORTANT: identical allocation ORDER on both ranks -> identical heap offsets (symmetric heap).
A_fp8_bf16, A_fp8 = make_fp8_iris_tensor(M, K)      # bf16-view (kernel) + fp8-view (same storage)
A_sc  = make_iris_tensor([M, NG], "float32")        # per-group scales [M,K/128]
B     = make_iris_tensor([N, K], "bfloat16")        # weights local on rank1
C     = make_iris_tensor([M, N], "bfloat16")        # output  local on rank1

# ---- build reference data ----
torch.manual_seed(1234)
A_real = (torch.randn(M, K, dtype=torch.float32, device='cuda') / 8.0)   # the true (pre-quant) A

# V1 quant scheme: per-128 group scale = max(|.|)/448, q = round-to-fp8(x/scale).
def quantize_v1(A):
    Ag = A.view(M, NG, QGROUP)
    amax = Ag.abs().amax(dim=2, keepdim=True)                 # [M,NG,1]
    scale = (amax / 448.0).clamp_min(1e-12)                   # [M,NG,1]
    q = (Ag / scale).to(torch.float8_e4m3fn)                  # fp8 e4m3
    return q.view(M, K), scale.view(M, NG).contiguous()

if rank == SRC_RANK:
    qa, qs = quantize_v1(A_real)
    A_fp8.copy_(qa)
    A_sc.copy_(qs)
    # bf16-dequantized A (this is exactly what the kernel reconstructs) -> the TRUE matmul input.
    A_deq = (qa.to(torch.float32).view(M, NG, QGROUP) * qs.view(M, NG, 1)).view(M, K).to(torch.bfloat16)
else:
    A_fp8.view(torch.uint8).zero_()                           # SENTINEL: rank1 local A = zeros
    A_sc.zero_()
    A_deq = None

torch.manual_seed(777)
Bfull = (torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 8.0)
B.copy_(Bfull)
C.zero_()
iris.barrier()

iris_ctx = iris.get_device_view()

def run(fused):
    C.zero_()
    if rank != SRC_RANK:
        tk_kernel.dispatch_micro(A_fp8_bf16, A_sc, B, C, iris_ctx, M, N, K, SRC_RANK, int(fused))
    torch.cuda.synchronize()
    iris.barrier()

def timed(fused):
    # warmup
    for _ in range(WARMUP):
        if rank != SRC_RANK:
            tk_kernel.dispatch_micro(A_fp8_bf16, A_sc, B, C, iris_ctx, M, N, K, SRC_RANK, int(fused))
    torch.cuda.synchronize(); iris.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        if rank != SRC_RANK:
            tk_kernel.dispatch_micro(A_fp8_bf16, A_sc, B, C, iris_ctx, M, N, K, SRC_RANK, int(fused))
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e6     # us/iter
    iris.barrier()
    return dt

# ---- correctness: build bf16 reference from the dequantized A (broadcast true A for ref) ----
comm = MPI.COMM_WORLD
A_deq_host = A_deq.float().cpu().numpy() if rank == SRC_RANK else None
A_deq_host = comm.bcast(A_deq_host, root=SRC_RANK)

def check(fused, label):
    run(fused)
    if rank != SRC_RANK:
        A_ref = torch.from_numpy(A_deq_host).to('cuda').to(torch.bfloat16)
        C_ref = torch.matmul(A_ref.float(), B.float().t())
        C_got = C.float()
        diff = (C_got - C_ref).abs()
        denom = C_ref.abs().clamp_min(1e-6)
        max_rel = (diff / denom).max().item()
        rms_rel = (diff.pow(2).mean().sqrt() / C_ref.pow(2).mean().sqrt()).item()
        max_abs = diff.max().item()
        c_zero = bool(C_got.abs().max().item() == 0.0)
        a_zero = bool(A_fp8.view(torch.uint8).max().item() == 0)
        ok = (rms_rel < 0.10 and not c_zero and a_zero)
        print(f"[{label}] M={M} K={K} N={N}  max_abs={max_abs:.4f}  max_rel={max_rel:.4f}  "
              f"RMS_rel={rms_rel:.5f}  local_A_zero={a_zero}  C_zero={c_zero}  -> "
              f"{'PASSED' if ok else 'FAILED'}", flush=True)
        return rms_rel
    return None

if rank != SRC_RANK:
    print("="*78, flush=True)
e_fused = check(True,  "FUSED   ")
e_base  = check(False, "BASELINE")

t_fused = timed(True)
t_base  = timed(False)

if rank != SRC_RANK:
    flops = 2.0 * M * N * K
    print("-"*78, flush=True)
    print(f"  BASELINE (two-phase, no overlap) : {t_base:8.2f} us/iter   {flops/(t_base*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
    print(f"  FUSED    (gather+dequant+MFMA)   : {t_fused:8.2f} us/iter   {flops/(t_fused*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
    spd = t_base / t_fused
    print(f"  SPEEDUP (baseline/fused)         : {spd:6.3f}x   ({'FUSED WINS' if spd>1 else 'baseline wins'})", flush=True)
    print("="*78, flush=True)

import gc
del A_fp8_bf16, A_fp8, A_sc, B, C
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
