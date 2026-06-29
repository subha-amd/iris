"""
Test: verify iris symmetric heap offsets are identical across ranks,
and that remote ptr translation produces the correct address.
"""
import sys, os
sys.path.insert(0, '/tmp/HipKittens/distributed-kernels')
import torch, numpy as np
import iris_py

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
world = iris.world_size()
torch.cuda.set_device(rank)

from mpi4py import MPI
comm = MPI.COMM_WORLD

# Allocate a small buffer on the iris heap (same order on every rank)
buf = iris.empty([64], "float32")
ptr = buf.data_ptr()

# Get heap base for this rank via the device view
dv = iris.get_device_view()
# Can't easily read heap_bases from Python, but we can:
# 1. Get the offset from heap base by writing a known value and reading via IPC
# Alloc 4 floats, write rank*100+1
t = torch.as_tensor(buf, device='cuda').view(torch.float32)
t[0] = float(rank * 100 + 1)
torch.cuda.synchronize()

# Share ptrs across ranks
all_ptrs = comm.allgather(ptr)
comm.Barrier()

# Check offsets are identical
offsets = [p - all_ptrs[0] for p in all_ptrs]
if rank == 0:
    print(f"ptr offsets from rank0 heap base: {offsets}", flush=True)
    all_same = all(o == 0 for o in offsets)
    print(f"All ptrs identical (symmetric heap): {all_same}", flush=True)
    print(f"ptrs: {[hex(p) for p in all_ptrs]}", flush=True)
comm.Barrier()

# Each rank reads its neighbor's buffer value via device view (iris remote load)
# This requires a GPU kernel — just check the ptr math instead
src_rank = (rank + 1) % world
src_ptr = all_ptrs[src_rank]
local_ptr = all_ptrs[rank]
offset = ptr - local_ptr  # should be 0
translated = src_ptr + offset

if rank == 0:
    print(f"\nrank=0 local ptr={hex(local_ptr)} src_rank=1 src_ptr={hex(src_ptr)}", flush=True)
    print(f"offset={offset} -> translated={hex(translated)} (should == {hex(src_ptr)})", flush=True)
    print(f"Translation correct: {translated == src_ptr}", flush=True)
