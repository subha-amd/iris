#!/usr/bin/env python3
# probe_coexist.py — can ONE process host IRIS (MPI symmetric heap + tk_kernel) *and*
# MORI (shmem) + aiter at the same time?
#
# If yes, the fair dispatch-prefix benchmark can run both paths on the SAME tensors, in the
# SAME iteration, on the SAME 8 GPUs -> the only honest way to race gather_pack against
# EpDispatch + moe_sorting.
#
# FINDING (run 1): `mori.shmem.shmem_torch_process_group_init("default")` HANGS under mpirun
#   (all 8 ranks spinning on CPU, 0% GPU) — its uid broadcast goes through a torch.distributed
#   NCCL collective, and RCCL comm creation deadlocks alongside an already-initialised
#   OpenMPI + IRIS IPC heap. MORI ships `shmem_mpi_init()` (bootstraps from MPI_COMM_WORLD),
#   which is the right entry point when the process is already under mpirun.
#
# Run:  mpirun --allow-run-as-root -np 8 python3 probe_coexist.py
import os
os.environ.setdefault("MORI_GPU_ARCHS", "gfx950")
os.environ.setdefault("HSA_XNACK", "1")

import sys
DK = "/home/subvadla/HipKittens/distributed-kernels"
sys.path.insert(0, DK)
sys.path.insert(0, os.path.join(DK, "b1_dispatch"))

import faulthandler
_LR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
_LOG = open(f"/tmp/coexist_r{_LR}.log", "w", buffering=1)
faulthandler.enable(file=_LOG)
faulthandler.dump_traceback_later(120, repeat=True, file=_LOG)   # hung? dump a traceback


def _p(*a):
    print(*a, file=_LOG, flush=True)
    print(*a, flush=True)


_p(f"[r{_LR}] START")

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import iris_py                      # noqa: E402
import tk_kernel                    # noqa: E402
from mpi4py import MPI              # noqa: E402
import numpy as np                  # noqa: E402
import torch                        # noqa: E402
_p(f"[r{_LR}] torch imported")

# ---- 1. IRIS first (it calls MPI_Init) --------------------------------------------------------
iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank, world = iris.rank(), iris.world_size()
torch.cuda.set_device(rank)
_p(f"[r{rank}] IRIS ok  world={world}")

comm = MPI.COMM_WORLD
assert comm.Get_rank() == rank and comm.Get_size() == world

# ---- 2. aiter + MORI, bootstrapped from MPI (NOT torch.distributed) ---------------------------
_p(f"[r{rank}] importing aiter/mori")
import aiter                        # noqa: E402
from aiter import dtypes            # noqa: E402
from aiter.fused_moe import fused_topk  # noqa: E402
import mori                         # noqa: E402
import mori.shmem                   # noqa: E402
_p(f"[r{rank}] aiter+mori imported")

BOOT = os.environ.get("MORI_BOOT", "mpi")
if BOOT == "mpi":
    _p(f"[r{rank}] mori.shmem.shmem_mpi_init() ...")
    mori.shmem.shmem_mpi_init()
else:   # uid: get on rank0, broadcast with mpi4py, init everywhere (no torch.distributed)
    uid = mori.shmem.shmem_get_unique_id() if rank == 0 else None
    uid = comm.bcast(uid, root=0)
    _p(f"[r{rank}] shmem_init_attr(uid) ...")
    mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world, uid)
_p(f"[r{rank}] mori shmem ok  mype={mori.shmem.shmem_mype()} npes={mori.shmem.shmem_npes()}")

# ---- 3. a real EpDispatch on R1 shapes ---------------------------------------------------------
HID, E_GLOBAL, TOPK, T = 7168, 256, 8, 64
Eloc = E_GLOBAL // world
dev = torch.device(f"cuda:{rank}")
torch.manual_seed(1000 + rank)
tokens = torch.randn((T, HID), dtype=dtypes.bf16, device=dev)
score = torch.randn((T, E_GLOBAL), dtype=dtypes.bf16, device=dev)
topk_w, topk_ids = fused_topk(tokens, score, TOPK, True)
_p(f"[r{rank}] fused_topk ok  ids{tuple(topk_ids.shape)} w{tuple(topk_w.shape)} {topk_w.dtype}")

cfg = mori.ops.EpDispatchCombineConfig(
    data_type=tokens.dtype, rank=rank, world_size=world, hidden_dim=HID,
    scale_dim=0, scale_type_size=0, max_token_type_size=dtypes.bf16.itemsize,
    max_num_inp_token_per_rank=T,          # realistic, NOT b3's max(8192, 4*T)
    num_experts_per_rank=Eloc, num_experts_per_token=TOPK,
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode)
_p(f"[r{rank}] building EpDispatchCombineOp (JIT may take minutes on a cold cache)")
op = mori.ops.EpDispatchCombineOp(cfg)
_p(f"[r{rank}] op built")

do, dw, ds, di, drn = op.dispatch(tokens, topk_w, None, topk_ids)
torch.cuda.synchronize()
_p(f"[r{rank}] EpDispatch ok  recv={int(drn.item())}  do{tuple(do.shape)}")

# ---- 4. replay mode ----------------------------------------------------------------------------
*_, routing = op.dispatch(tokens, topk_w, None, topk_ids, return_routing=True)
torch.cuda.synchronize()
_, _, _, _, drn3 = op.dispatch(tokens, topk_w, None, topk_ids, routing=routing)
torch.cuda.synchronize()
_p(f"[r{rank}] EpDispatch REPLAY ok  recv={int(drn3.item())}")

# ---- 5. aiter per_1x128 quant (the kernel BOTH paths must pay) ----------------------------------
aq = aiter.get_hip_quant(aiter.QuantType.per_1x128)
tq, tsc = aq(tokens, quant_dtype=dtypes.fp8)
torch.cuda.synchronize()
_p(f"[r{rank}] quant ok  tq{tuple(tq.shape)} {tq.dtype}  tsc{tuple(tsc.shape)} {tsc.dtype} "
   f"contig={tsc.is_contiguous()} stride={tsc.stride()}")

# ---- 6. gather_pack in the SAME process --------------------------------------------------------
K, NG, BM = HID, HID // 128, 64
Msrc, Mpacked = 256, 256


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
    t = iris.empty(shape, dtype=alloc)
    dmap = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"),
            "int32": (torch.int32, "<i4")}
    td, ts = dmap[dtype]
    return _view(t, ts, shape, td)


A_src_bf16, A_src_fp8 = make_fp8(Msrc, K)
A_src_sc = make_iris([Msrc, NG], "float32")
A_pk_bf16, A_pk_fp8 = make_fp8(Mpacked, K)
A_pk_sc = make_iris([Mpacked, NG], "float32")

# one segment per rank: packed rows [32r,32r+32) come from rank r's source rows [0,32)
segs = np.array([[0, r, 0, 32 * r, 32] for r in range(world)], dtype=np.int32)
Ntile = (Mpacked + BM - 1) // BM
tiles = np.array([[0, world, t * BM, BM] for t in range(Ntile)], dtype=np.int32)
SEG = make_iris([segs.shape[0], 5], "int32")
TILE = make_iris([Ntile, 4], "int32")
SEG.copy_(torch.from_numpy(segs).cuda())
TILE.copy_(torch.from_numpy(tiles).cuda())

A_src_fp8.view(torch.uint8).copy_(torch.full((Msrc, K), rank + 1, dtype=torch.uint8, device=dev))
A_src_sc.fill_(1.0)
A_pk_fp8.view(torch.uint8).zero_()
iris.barrier()

ctx = iris.get_device_view()
tk_kernel.dispatch_gather_pack(A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, ctx,
                               Msrc, Mpacked, K, segs.shape[0], Ntile)
torch.cuda.synchronize()
iris.barrier()

got = A_pk_fp8.view(torch.uint8)[:, 0].cpu().numpy()
want = np.repeat(np.arange(1, world + 1, dtype=np.uint8), 32)
ok = bool((got[:world * 32] == want).all())
_p(f"[r{rank}] gather_pack after mori init: {'PASS' if ok else 'FAIL'}  got[:9]={got[:9]}")

# ---- 7. moe_sorting, exactly as fused_moe calls it ----------------------------------------------
from aiter.fused_moe import moe_sorting   # noqa: E402
expert_mask = torch.zeros((E_GLOBAL,), dtype=dtypes.i32, device=dev)
expert_mask[Eloc * rank: Eloc * (rank + 1)] = 1
sr = moe_sorting(di, dw, E_GLOBAL, HID, dtypes.bf16, 32, expert_mask, drn, 0)
torch.cuda.synchronize()
_p(f"[r{rank}] moe_sorting ok  sorted_ids{tuple(sr[0].shape)} sorted_eids{tuple(sr[2].shape)} "
   f"num_valid={sr[3].tolist()} moe_buf{tuple(sr[4].shape)}")

iris.barrier()
ok_all = comm.allreduce(1 if ok else 0, op=MPI.MIN)
if rank == 0:
    _p("\n=== COEXIST PROBE PASSED: IRIS + tk_kernel + MORI(shmem_mpi_init) + aiter, one process ===")
    _p(f"    gather_pack correctness on all ranks: {'PASS' if ok_all else 'FAIL'}")
faulthandler.cancel_dump_traceback_later()
