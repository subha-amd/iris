import sys
sys.path.insert(0, "/tmp/HipKittens/distributed-kernels")
import iris_py, torch

iris = iris_py.Iris(heap_size_mb=64, verbose=True)
rank = iris.rank()
world = iris.world_size()
torch.cuda.set_device(rank)

buf = iris.empty([4], "float32")
ptr = buf.data_ptr()

# Use iris barrier + a shared output approach: each rank prints its ptr,
# rank 0 reads the verbose output from iris init to see the IPC bases.
iris.barrier()
print("rank=%d ptr=0x%x" % (rank, ptr), flush=True)
iris.barrier()

# Check the device view - it contains the heap_bases_ we care about.
# We can't easily read them from Python, but verbose=True in the iris constructor
# prints them if the relevant log level is enabled.
dv = iris.get_device_view()
print("rank=%d device_view_rank=%d" % (rank, dv.rank()), flush=True)
