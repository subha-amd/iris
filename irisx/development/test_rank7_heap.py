import sys
sys.path.insert(0, "/tmp/HipKittens/distributed-kernels")
import iris_py, torch

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
torch.cuda.set_device(rank)

MSRC = 4096
K = 7168

t = iris.empty([MSRC, K // 2], "bfloat16")
iris_ptr = t.data_ptr()

iris.barrier()
# Every rank prints its A_src ptr
print("rank=%d A_src iris_ptr=0x%x" % (rank, iris_ptr), flush=True)
iris.barrier()
