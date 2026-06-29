#!/usr/bin/env python3
# p1_tile_inbox / example.py  (SCAFFOLD; the MAIN AGENT runs on the node under flock, np=2)
# ================================================================================================
# CANDIDATE P1 — copy-once tile inbox. PRODUCER copies remote A (fp8+scales) ONCE into a LOCAL
# dequantized bf16 A[M,K] inbox + per-row-band ready flags; CONSUMER = the EXACT B0 GEMM reading A
# from the local inbox. Two concurrent kernels, two streams, overlapped. Metric = T_pipeline.
#
# Topology (np=2, mirrors B1 / cache_first_touch):
#   Rank 0 (SRC_RANK) = "producer rank": holds the ONLY real fp8 A[M,K] + per-128 fp32 scales.
#   Rank 1            = "consumer rank": runs BOTH p1_producer (remote gather/dequant -> inbox) and
#                       p1_consumer (B0 GEMM over the local inbox), concurrently on two streams.
#                       Only p1_producer touches rank 0 over IRIS; p1_consumer is local-only.
#
# DEADLOCK AVOIDANCE: dispatch_p1 (kernel.cpp) launches producer FIRST on prod_stream (reserves CUs),
#   then consumer on cons_stream with NO sync between. Producer cursor hands out every band once;
#   CAS makes exactly one block materialize+signal each band; consumers only wait on bands, never on
#   other consumer blocks. So the pool drains even with a single resident producer block.
#
# SYMMETRIC BARRIERS (this bit Agent 01): BOTH ranks run the SAME number of run()/iris.barrier()
#   calls. Only rank1 does GPU work inside run(); rank0 just participates in the barriers. An
#   asymmetric barrier count deadlocks.
#
# CSV columns (irisx/results/results.csv):
#   candidate,commit,date,ranks,route_dist,M_label,M,N,K,dtype,BM,BN,BK,NSUB,schedule,grid_blocks,
#   VGPR,AGPR,SGPR,LDS,scratch,lat_us,p50,p95,p99,TFLOPs,rms_rel,zero_sentinel,spd_vs_B1,spd_vs_B2,
#   spd_vs_B3,notes
# ================================================================================================
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
ITERS  = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))
NUM_PRODUCER_BLOCKS = int(os.environ.get("PROD_BLOCKS", "16"))
BM_PROD = int(os.environ.get("BM_PROD", "64"))   # MUST match kernel.cpp BM_PROD
QGROUP = 128
assert K % QGROUP == 0
NG = K // QGROUP
NUM_BANDS = (M + BM_PROD - 1) // BM_PROD

iris = iris_py.Iris(heap_size_mb=1024, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 2, f"P1 bring-up expects np=2, got world={world}"
torch.cuda.set_device(rank)
SRC_RANK = 0

def make_iris_tensor(shape, dtype):
    # IRIS python iris.empty supports bfloat16/float32 but NOT int32. int32 and float32 are both
    # 4 bytes, so back int32 flag buffers with a float32 IRIS allocation and expose an int32 view
    # (the device-side gl<int> just needs the symmetric-heap byte pointer; storage dtype is irrelevant).
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

def make_fp8_iris_tensor(M, K):
    assert K % 2 == 0
    t = iris.empty([M, K // 2], "bfloat16")
    def view(ts, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    return view("<u2", (M, K // 2), torch.bfloat16), view("|u1", (M, K), torch.float8_e4m3fn)

# IMPORTANT: identical allocation ORDER on both ranks -> identical symmetric-heap offsets.
A_fp8_bf16, A_fp8 = make_fp8_iris_tensor(M, K)     # remote fp8 source (rank0)
A_sc  = make_iris_tensor([M, NG], "float32")       # remote scales
B     = make_iris_tensor([N, K], "bfloat16")       # local weights (rank1)
C     = make_iris_tensor([M, N], "bfloat16")       # local output  (rank1)
INBOX = make_iris_tensor([M, K], "bfloat16")       # LOCAL dequantized bf16 A (row-major)
READY = make_iris_tensor([NUM_BANDS], "int32")
CLAIM = make_iris_tensor([NUM_BANDS], "int32")
CURSOR = make_iris_tensor([1], "int32")

# ---- reference data ----
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
    if rank != SRC_RANK:
        READY.zero_(); CLAIM.zero_(); CURSOR.zero_(); INBOX.zero_()
    torch.cuda.synchronize(); iris.barrier()

def run():
    """Symmetric across ranks: both call iris.barrier() the same # of times; only rank1 launches."""
    global GEN
    GEN += 1
    reset_inbox()
    if rank != SRC_RANK:
        tk_kernel.dispatch_p1(
            A_fp8_bf16, A_sc, B, C,
            INBOX, READY, CLAIM, CURSOR,
            iris_ctx, M, N, K, SRC_RANK, GEN,
            NUM_BANDS, NUM_PRODUCER_BLOCKS)
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
        a_zero = bool(A_fp8.view(torch.uint8).max().item() == 0)   # local A still zero -> remote gather
        c_zero = bool(C_got.abs().max().item() == 0.0)
        ok = (rms_rel < 0.10 and not c_zero and a_zero)
        print(f"[{label}] M={M} K={K} N={N}  max_rel={max_rel:.4f}  RMS_rel={rms_rel:.5f}  "
              f"local_A_zero={a_zero}  C_zero={c_zero}  -> {'PASSED' if ok else 'FAILED'}", flush=True)
        return rms_rel, a_zero
    return None, None

if rank != SRC_RANK:
    print("=" * 78, flush=True)
e, sentinel = check("P1      ")
t = timed()
if rank != SRC_RANK:
    flops = 2.0 * M * N * K
    B0 = 164.2; B1 = 291.4   # from EXPERIMENT_LEDGER.md (M1024/N2048/K7168)
    print("-" * 78, flush=True)
    print(f"  P1 COPY-ONCE TILE INBOX (B0 consumer, overlap): {t:8.2f} us/iter   "
          f"{flops/(t*1e-6)/1e12:6.2f} TFLOP/s", flush=True)
    print(f"  T_pipeline={t:.1f}us   vs B0={B0}us  vs B1={B1}us   "
          f"spd_vs_B1={B1/t:.3f}x   overlap_vs_floor(162us)={t/162.0:.3f}", flush=True)
    print(f"  compute_retention ~= B0/T_pipeline (upper bound) = {B0/t:.3f}", flush=True)
    # CSV row (paste into results/results.csv; main agent fills commit + static resource cols).
    print(f"CSV,P1,<commit>,2,fixed_src,per_expert,{M},{N},{K},fp8_bf16,256,256,64,1,"
          f"prod+B0consumer,{(N+255)//256*((M+255)//256)},,,,,,"
          f"{t:.1f},,,,{flops/(t*1e-6)/1e12:.1f},{e:.5f},{sentinel},{B1/t:.3f},,,"
          f"copy_once_2kernel_overlap", flush=True)
    print("=" * 78, flush=True)

import gc
del A_fp8_bf16, A_fp8, A_sc, B, C, INBOX, READY, CLAIM, CURSOR
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
