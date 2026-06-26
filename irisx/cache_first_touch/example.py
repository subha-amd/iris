#!/usr/bin/env python3
# cache_first_touch / example.py  (SCAFFOLD for the MAIN AGENT to finish + run on the node)
# ------------------------------------------------------------------------------------------------
# Agent 06 — cache-on-first-touch / local tile inbox. TWO concurrent kernels (producer + consumer)
# communicating through a LOCAL HBM symmetric-heap tile inbox so each remote A tile crosses XGMI
# exactly ONCE while a FULL consumer grid still hides latency.
#
# Topology (np=2, mirrors V4):
#   Rank 0 (SRC_RANK)  = "producer rank": holds the ONLY real fp8 A[M,K] + per-128 fp32 scales.
#   Rank 1             = "consumer rank": runs BOTH the cft_producer (gather/dequant -> inbox) and
#                        the cft_consumer (inbox -> MFMA) kernels, concurrently, on two streams.
#                        Only the cft_producer touches rank 0 over IRIS; cft_consumer is local-only.
#
# >>> LAUNCH ORDER / STREAM PLAN (the load-bearing part for the main agent) <<<
#   dispatch_cft (kernel.cpp host side) creates TWO non-blocking streams and:
#     1. launches cft_producer on prod_stream  (modest grid -> reserves few CUs first)
#     2. IMMEDIATELY launches cft_consumer on cons_stream (full grid) with NO sync between them
#     3. syncs both at the end.
#   The two SEPARATE launches are the deadlock-avoidance mechanism: the producer reserves its CUs
#   independently, so the full consumer grid can never occupy all CUs before producers run. The
#   consumer acquire-spins on per-slot ready flags; every awaited flag is eventually written because
#   the producer cursor hands out every task index and CAS makes exactly one producer gather+signal
#   each. Do NOT add a hipStreamSynchronize(prod) before the consumer launch — that serializes them
#   and defeats overlap (though it would still be correct, just slower).
#
# >>> PER-GENERATION RESET (host responsibility, done below) <<<
#   Before each dispatch for generation `gen`:
#     ready[:]  = CFT_FLAG_EMPTY (0)
#     claim[:]  = CFT_CLAIM_FREE (0)
#     cursor[:] = 0
#   Generation tagging (ready=(gen<<1)|1) is belt-and-suspenders: even if a reset were skipped, a
#   stale READY(gen-1) is a different int and can never satisfy a gen waiter.
#
# THIS IS A SCAFFOLD: the GEMM correctness harness mirrors V4's. The main agent should (a) confirm
# the HK pybind path accepts the int32 gl tensors for ready/claim/cursor, (b) tune num_producer_blocks,
# (c) run under flock on the node. Agent 06 did NOT run anything on the GPU (per AGENT_COMMON rule).
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
NUM_PRODUCER_BLOCKS = int(os.environ.get("PROD_BLOCKS", "16"))
QGROUP = 128
BM, BK = 64, 64
assert K % QGROUP == 0
NG = K // QGROUP

NUM_M_TILES = (M + BM - 1) // BM
NUM_K_TILES = (K + BK - 1) // BK
NUM_SLOTS   = NUM_M_TILES * NUM_K_TILES

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
    t = iris.empty([M, K // 2], "bfloat16")
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

# IMPORTANT: identical allocation ORDER on both ranks -> identical symmetric-heap offsets.
A_fp8_bf16, A_fp8 = make_fp8_iris_tensor(M, K)      # remote fp8 source (rank0)
A_sc  = make_iris_tensor([M, NG], "float32")        # remote scales
B     = make_iris_tensor([N, K], "bfloat16")        # local weights (rank1)
C     = make_iris_tensor([M, N], "bfloat16")        # local output  (rank1)

# ---- the LOCAL tile inbox (rank1) ----
INBOX = make_iris_tensor([NUM_SLOTS, BM * BK], "bfloat16")  # dequantized A tiles, ST_A-swizzled
READY = make_iris_tensor([NUM_SLOTS], "int32")             # per-slot ready flags
CLAIM = make_iris_tensor([NUM_SLOTS], "int32")             # per-task claim words
CURSOR = make_iris_tensor([1], "int32")                    # producer task dispenser

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
    A_fp8.view(torch.uint8).zero_()    # SENTINEL: rank1 has NO local A -> proves remote gather
    A_sc.zero_()
    A_deq = None

torch.manual_seed(777)
Bfull = (torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 8.0)
B.copy_(Bfull)
C.zero_()
iris.barrier()

iris_ctx = iris.get_device_view()

GEN = 0
def reset_inbox():
    # per-generation reset (host responsibility). Only the consumer rank owns the inbox.
    if rank != SRC_RANK:
        READY.zero_(); CLAIM.zero_(); CURSOR.zero_(); INBOX.zero_()
    torch.cuda.synchronize(); iris.barrier()

def run():
    global GEN
    GEN += 1
    reset_inbox()
    if rank != SRC_RANK:
        # dispatch_cft launches BOTH kernels (producer then consumer, two streams, no sync between).
        tk_kernel.dispatch_cft(
            A_fp8_bf16, A_sc, B, C,
            INBOX, READY, CLAIM, CURSOR,
            iris_ctx, M, N, K, SRC_RANK, GEN,
            NUM_M_TILES, NUM_K_TILES, NUM_SLOTS, NUM_PRODUCER_BLOCKS)
    torch.cuda.synchronize(); iris.barrier()

def timed():
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize(); iris.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e6
    iris.barrier()
    return dt

# ---- correctness ----
comm = MPI.COMM_WORLD
A_deq_host = A_deq.float().cpu().numpy() if rank == SRC_RANK else None
A_deq_host = comm.bcast(A_deq_host, root=SRC_RANK)

def check(label):
    C.zero_()
    run()
    if rank != SRC_RANK:
        A_ref = torch.from_numpy(A_deq_host).to('cuda').to(torch.bfloat16)
        C_ref = torch.matmul(A_ref.float(), B.float().t())
        C_got = C.float()
        diff = (C_got - C_ref).abs()
        denom = C_ref.abs().clamp_min(1e-6)
        max_rel = (diff / denom).max().item()
        rms_rel = (diff.pow(2).mean().sqrt() / C_ref.pow(2).mean().sqrt()).item()
        a_zero = bool(A_fp8.view(torch.uint8).max().item() == 0)   # local A still zero (remote gather)
        c_zero = bool(C_got.abs().max().item() == 0.0)
        ok = (rms_rel < 0.10 and not c_zero and a_zero)
        print(f"[{label}] M={M} K={K} N={N}  max_rel={max_rel:.4f}  RMS_rel={rms_rel:.5f}  "
              f"local_A_zero={a_zero}  C_zero={c_zero}  -> {'PASSED' if ok else 'FAILED'}", flush=True)
        return rms_rel
    return None

if rank != SRC_RANK:
    print("=" * 78, flush=True)
e = check("CFT     ")
t = timed()
if rank != SRC_RANK:
    flops = 2.0 * M * N * K
    print("-" * 78, flush=True)
    print(f"  CACHE-FIRST-TOUCH (2-kernel, 1x XGMI/tile): {t:8.2f} us/iter   "
          f"{flops/(t*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
    print("=" * 78, flush=True)

import gc
del A_fp8_bf16, A_fp8, A_sc, B, C, INBOX, READY, CLAIM, CURSOR
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
