#!/usr/bin/env python3
# irisx/p3_singlekernel/example.py
# ------------------------------------------------------------------------------------------------
# P3 single-kernel copy-once + in-block overlap driver (np=2).
#
#   Rank 0 ("src rank")      holds the ONLY real fp8 activations A_fp8[M,K] + per-128 fp32 scales
#                            A_sc[M,K/128] on its IRIS heap.
#   Rank 1 ("consumer rank") runs the P3 expert GEMM C = dequant(A_fp8) @ B^T.  Its SINGLE kernel
#                            gathers each remote A K-tile EXACTLY ONCE (overlapped with MFMA via an
#                            in-block double-buffer), caches it to local HBM, and reuses it across
#                            all N-panels.  No second kernel, no cross-stream flag => no P1/P2 hazard.
#
# Reports: correctness (RMS-rel vs bf16 reference + zero-sentinel) for the FUSED (P3 overlap) path
# and the in-module SERIAL baseline (gather-all-then-local-GEMM = B1-copy reproduction); and
# head-to-head wall-clock.  The AUTHORITATIVE B1 reference is the harness (run separately by the
# main agent); this in-module baseline is for a self-contained sanity comparison.
#
# Driver mirrors v4_astationary_kernel/example.py EXACTLY (same iris alloc tricks, same symmetric
# barriers, same correctness math) so it slots into the existing runner unchanged.  ALL ranks run
# the timing loops symmetrically (asymmetric iris.barrier() deadlocks — see ledger Gate-1 note).
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
M = int(os.environ.get("M", "1024"))
K = int(os.environ.get("K", "7168"))
N = int(os.environ.get("N", "2048"))
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))
CSV = os.environ.get("CSV", "")            # optional: append a results.csv row
QGROUP = 128
assert K % QGROUP == 0
NG = K // QGROUP

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
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

# fp8 buffer: iris.empty has no fp8 dtype AND HK gl rejects 1-byte elems -> allocate M*K BYTES as
# bf16 [M,K//2] on the heap, keep two views over the SAME storage (bf16 = kernel arg, fp8 = quant).
def make_fp8_iris_tensor(M, K):
    assert K % 2 == 0
    t = iris.empty([M, K // 2], "bfloat16")        # M*K bytes
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

# identical allocation ORDER on both ranks -> identical heap offsets (symmetric heap).
A_fp8_bf16, A_fp8 = make_fp8_iris_tensor(M, K)
A_sc  = make_iris_tensor([M, NG], "float32")
B     = make_iris_tensor([N, K], "bfloat16")
C     = make_iris_tensor([M, N], "bfloat16")

# ---- build reference data ----
torch.manual_seed(1234)
A_real = (torch.randn(M, K, dtype=torch.float32, device='cuda') / 8.0)

def quantize_v1(A):
    Ag = A.view(M, NG, QGROUP)
    amax = Ag.abs().amax(dim=2, keepdim=True)
    scale = (amax / 448.0).clamp_min(1e-12)
    q = (Ag / scale).to(torch.float8_e4m3fn)
    return q.view(M, K), scale.view(M, NG).contiguous()

if rank == SRC_RANK:
    qa, qs = quantize_v1(A_real)
    A_fp8.copy_(qa)
    A_sc.copy_(qs)
    A_deq = (qa.to(torch.float32).view(M, NG, QGROUP) * qs.view(M, NG, 1)).view(M, K).to(torch.bfloat16)
else:
    A_fp8.view(torch.uint8).zero_()                # SENTINEL: consumer-rank local A = zeros
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
    for _ in range(WARMUP):
        if rank != SRC_RANK:
            tk_kernel.dispatch_micro(A_fp8_bf16, A_sc, B, C, iris_ctx, M, N, K, SRC_RANK, int(fused))
    torch.cuda.synchronize(); iris.barrier()
    # one enclosing cuda event around the whole P3 launch (the single kernel = total T)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(ITERS):
        if rank != SRC_RANK:
            tk_kernel.dispatch_micro(A_fp8_bf16, A_sc, B, C, iris_ctx, M, N, K, SRC_RANK, int(fused))
    ev1.record()
    torch.cuda.synchronize()
    dt = ev0.elapsed_time(ev1) / ITERS * 1e3      # us/iter
    iris.barrier()
    return dt

# ---- correctness reference (broadcast true dequant A from src rank) ----
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
        return rms_rel, a_zero, (not c_zero)
    return None, None, None

if rank != SRC_RANK:
    print("="*78, flush=True)
e_fused, a_zero_f, sentinel_f = check(True,  "P3-FUSED ")
e_base,  _,        _          = check(False, "SERIAL-B1")

t_fused = timed(True)
t_base  = timed(False)

if rank != SRC_RANK:
    flops = 2.0 * M * N * K
    tf_fused = flops/(t_fused*1e-6)/1e12
    tf_base  = flops/(t_base*1e-6)/1e12
    print("-"*78, flush=True)
    print(f"  SERIAL  (gather-all + local GEMM, ~B1) : {t_base:8.2f} us/iter   {tf_base:6.2f} TFLOP/s", flush=True)
    print(f"  P3-FUSED (copy-once + in-block overlap): {t_fused:8.2f} us/iter   {tf_fused:6.2f} TFLOP/s", flush=True)
    spd = t_base / t_fused
    print(f"  SPEEDUP (serial/fused)                 : {spd:6.3f}x   ({'P3 WINS' if spd>1 else 'serial wins'})", flush=True)
    print("="*78, flush=True)
    if CSV:
        hdr = ("candidate,M,N,K,dtype,BM,BN,BK,schedule,lat_us,TFLOPs,rms_rel,"
               "zero_sentinel,serial_us,serial_TFLOPs,spd_vs_serial\n")
        new = not os.path.exists(CSV)
        with open(CSV, "a") as f:
            if new: f.write(hdr)
            f.write(f"p3_singlekernel,{M},{N},{K},fp8e4m3->bf16,256,256,64,"
                    f"copyonce_inblock_overlap,{t_fused:.2f},{tf_fused:.2f},{e_fused:.5f},"
                    f"{bool(sentinel_f)},{t_base:.2f},{tf_base:.2f},{spd:.3f}\n")

import gc
del A_fp8_bf16, A_fp8, A_sc, B, C
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
