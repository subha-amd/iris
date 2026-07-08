#!/usr/bin/env python3
# probe_boot.py — isolate WHY mori.shmem's bootstrap deadlocks in our process.
#
# Observed: under `mpirun -np 8`, with iris_py.Iris() already up, BOTH
#   mori.shmem.shmem_torch_process_group_init("default")   (nccl uid broadcast)
#   mori.shmem.shmem_init_attr(UNIQUEID, rank, world, uid) (mpi4py uid broadcast)
# hang with all 8 ranks spinning on CPU at 0% GPU.
#
# MODE:
#   mori_only   : MORI alone under mpirun (no IRIS at all)      -> is mpirun the problem?
#   mori_first  : MORI shmem init, THEN iris_py.Iris()          -> is the ORDER the problem?
#   iris_first  : iris_py.Iris(), THEN MORI shmem init          -> reproduces the hang
#
# Run: MODE=mori_only mpirun --allow-run-as-root -np 8 python3 probe_boot.py
import os, sys, faulthandler
os.environ.setdefault("MORI_GPU_ARCHS", "gfx950")
os.environ.setdefault("HSA_XNACK", "1")
DK = "/home/subvadla/HipKittens/distributed-kernels"
sys.path.insert(0, DK)
sys.path.insert(0, os.path.join(DK, "b1_dispatch"))

MODE = os.environ.get("MODE", "iris_first")
_LR = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
_WS = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "8"))
_LOG = open(f"/tmp/boot_{MODE}_r{_LR}.log", "w", buffering=1)
faulthandler.enable(file=_LOG)
faulthandler.dump_traceback_later(90, repeat=True, file=_LOG)


def _p(*a):
    print(*a, flush=True)
    print(*a, file=_LOG, flush=True)


_p(f"[r{_LR}] MODE={MODE} START")

import mpi4py
mpi4py.rc.initialize = (MODE == "mori_only" or MODE == "mori_first")
mpi4py.rc.finalize = False
from mpi4py import MPI
import torch

iris = None


def start_iris():
    global iris
    import iris_py
    import tk_kernel  # noqa: F401
    iris = iris_py.Iris(heap_size_mb=256, verbose=False)
    _p(f"[r{_LR}] IRIS up: rank={iris.rank()} world={iris.world_size()}")


def start_mori(rank, world, comm):
    import mori, mori.shmem
    torch.cuda.set_device(rank)
    uid = mori.shmem.shmem_get_unique_id() if rank == 0 else None
    uid = comm.bcast(uid, root=0)
    _p(f"[r{rank}] shmem_init_attr(uid={len(uid) if uid else 0}B) ...")
    mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world, uid)
    _p(f"[r{rank}] MORI shmem up: mype={mori.shmem.shmem_mype()} npes={mori.shmem.shmem_npes()}")


if MODE == "mori_only":
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    start_mori(rank, world, comm)

elif MODE == "mori_first":
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    start_mori(rank, world, comm)
    start_iris()

else:  # iris_first
    start_iris()
    comm = MPI.COMM_WORLD
    rank, world = iris.rank(), iris.world_size()
    torch.cuda.set_device(rank)
    start_mori(rank, world, comm)

_p(f"[r{_LR}] MODE={MODE} SUCCESS")
faulthandler.cancel_dump_traceback_later()
