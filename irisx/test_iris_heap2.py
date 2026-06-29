import sys, os
sys.path.insert(0, '/tmp/HipKittens/distributed-kernels')
import iris_py
from mpi4py import MPI

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
comm = MPI.COMM_WORLD

# First allocation on the iris heap
buf = iris.empty([1024], "float32")
ptr = buf.data_ptr()

all_ptrs = comm.allgather(ptr)
if rank == 0:
    print(f"First alloc ptrs across ranks: {[hex(p) for p in all_ptrs]}", flush=True)
    offsets = [p - all_ptrs[0] for p in all_ptrs]
    print(f"Offsets from rank0: {offsets}", flush=True)
    print(f"All same offset from respective heap bases: {len(set(offsets)) == 1}", flush=True)

# Second allocation - check offsets are still symmetric
buf2 = iris.empty([512], "float32")
ptr2 = buf2.data_ptr()
all_ptrs2 = comm.allgather(ptr2)
if rank == 0:
    offsets2 = [all_ptrs2[i] - all_ptrs[i] for i in range(len(all_ptrs))]
    print(f"Offset between alloc1 and alloc2 per rank: {offsets2}", flush=True)
    print(f"Offset symmetric: {len(set(offsets2)) == 1}", flush=True)
