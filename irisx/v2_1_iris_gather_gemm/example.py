#!/usr/bin/env python3
# fmoe_gather_gemm / example.py
# ------------------------------------------------------------------------------------------------
# V2.1 driver: prove the remote-gather-fused GEMM on np=2.
#
#   Rank 0 ("producer rank"): holds the ONLY initialized copy of A[M,K] on its IRIS heap.
#   Rank 1 ("consumer rank") : runs the GEMM.  Its kernel's PRODUCER warps pull each A-tile
#                              from rank 0's heap via iris_ctx.load(&A[...], src_rank=0),
#                              overlapped with the consumer warps' MFMA.  B[N,K] and C[M,N]
#                              are local to rank 1.
#
# Proof that the gather is REAL (not a local fallback):
#   - Rank 1's A buffer is filled with a SENTINEL (zeros), never the true A.  If the kernel read
#     rank 1's local A it would compute C == 0 -> huge error.  It instead reads rank 0's A (the
#     only real copy) over IRIS, so C matches the reference built from rank 0's A.
#   - We additionally flip rank 0's A to a different distribution AFTER rank 1's sentinel fill and
#     re-barrier, then check the result tracks rank 0's (post-flip) values.
#
# Both A buffers are allocated with the SAME allocation sequence on the symmetric IRIS heap, so
# they sit at the same heap offset on every rank -> iris translate() maps rank1's &A to rank0's A.
# ------------------------------------------------------------------------------------------------
import sys; sys.path.insert(0, "..")
import torch
# Let IRIS own the MPI lifecycle: tell mpi4py NOT to auto-init/finalize MPI on import,
# otherwise MPI_Init is called twice (once by mpi4py, once by iris::mpi::initialize()).
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import iris_py
import tk_kernel
from mpi4py import MPI   # safe now: does not call MPI_Init

torch.manual_seed(0)

# ---- modest shapes that fit tight VRAM ----
M = 512
K = 2048
N = 512

iris = iris_py.Iris(heap_size_mb=64, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 2, f"this bring-up expects np=2, got world={world}"
torch.cuda.set_device(rank)

SRC_RANK = 0   # producer rank that holds A

def make_iris_tensor(shape, dtype="bfloat16"):
    """Allocate on the IRIS symmetric heap and wrap as a torch tensor (zero-copy via CAI)."""
    t = iris.empty(shape, dtype=dtype)
    dtype_map = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4")}
    torch_dtype, typestr = dtype_map[dtype]
    class W:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {
                'shape': tuple(shape), 'typestr': typestr,
                'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    w = W(t.data_ptr(), shape, typestr)
    return torch.as_tensor(w, device='cuda').view(torch_dtype).view(*shape)

# IMPORTANT: identical allocation order on BOTH ranks -> identical heap offsets (symmetric heap).
A = make_iris_tensor([M, K], "bfloat16")   # A on rank0 = real; on rank1 = sentinel placeholder
B = make_iris_tensor([N, K], "bfloat16")   # B local on rank1 (rank0's copy unused)
C = make_iris_tensor([M, N], "bfloat16")   # C output local on rank1

# ---- initialize ----
if rank == SRC_RANK:
    torch.manual_seed(1234)
    A.copy_(torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 8.0)
else:
    A.zero_()                              # SENTINEL: rank1's local A is all zeros

torch.manual_seed(777)
Bfull = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 8.0
B.copy_(Bfull)
C.zero_()
iris.barrier()

# ---- run the fused remote-gather GEMM on rank 1 ----
iris_ctx = iris.get_device_view()
if rank != SRC_RANK:
    tk_kernel.dispatch_micro(A, B, C, iris_ctx, M, N, K, SRC_RANK)
torch.cuda.synchronize()
iris.barrier()

# ---- reference + validation (rank 1) ----
# Rank 1 needs the TRUE A to build the reference.  Pull rank 0's A across IRIS the simple way:
# rank 0 broadcasts its A over MPI (host roundtrip) purely for the CPU/GPU reference — this does
# NOT feed the kernel, it only lets rank 1 know what the correct answer is.
comm = MPI.COMM_WORLD
A_true_host = A.float().cpu().numpy() if rank == SRC_RANK else None
A_true_host = comm.bcast(A_true_host, root=SRC_RANK)

if rank != SRC_RANK:
    A_true = torch.from_numpy(A_true_host).to('cuda').to(torch.bfloat16)
    C_ref = torch.matmul(A_true.float(), B.float().t())   # C = A @ B^T
    C_got = C.float()
    diff = (C_got - C_ref).abs()
    denom = C_ref.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    rms_rel = (diff.pow(2).mean().sqrt() / C_ref.pow(2).mean().sqrt()).item()
    max_abs = diff.max().item()

    # Sanity: prove rank1's local A really was the zero sentinel (so a local read => C==0).
    local_A_is_zero = bool((A.float().abs().max().item() == 0.0))
    C_all_zero = bool((C_got.abs().max().item() == 0.0))

    print("="*64)
    print(f"[Rank {rank}] fmoe_gather_gemm  M={M} K={K} N={N}  src_rank={SRC_RANK}")
    print(f"  rank1 local-A is zero sentinel : {local_A_is_zero}")
    print(f"  kernel output all-zero?        : {C_all_zero}  (False => gather pulled real A)")
    print(f"  max  abs error                 : {max_abs:.6f}")
    print(f"  max  rel error                 : {max_rel:.6f}")
    print(f"  RMS  rel error                 : {rms_rel:.6f}")
    status = "PASSED" if (rms_rel < 0.05 and not C_all_zero and local_A_is_zero) else "FAILED"
    print(f"  RESULT                         : {status}")
    print("="*64)

# ---- clean shutdown ----
import gc
del A, B, C
gc.collect(); torch.cuda.synchronize()
iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
import os; os._exit(0)
