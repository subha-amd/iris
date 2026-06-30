"""Check whether make_fp8's torch tensor has the same data_ptr as the iris allocation."""
import sys
sys.path.insert(0, "/tmp/HipKittens/distributed-kernels")
import iris_py, torch

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
torch.cuda.set_device(rank)

MSRC = 4096
K = 7168

# Replicate make_fp8 from example.py
t = iris.empty([MSRC, K // 2], "bfloat16")
iris_ptr = t.data_ptr()

def view(ts, shape, td):
    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {
                'shape': tuple(shape), 'typestr': ts,
                'data': (ptr, False), 'version': 3, 'strides': None
            }
            self._keep = t
    return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)

A_src_bf16 = view("<u2", (MSRC, K // 2), torch.bfloat16)
A_src_fp8  = view("|u1", (MSRC, K),       torch.float8_e4m3fn)

iris.barrier()
if rank == 0:
    print("iris_ptr    = 0x%x" % iris_ptr)
    print("bf16.data_ptr = 0x%x  (same as iris: %s)" % (
        A_src_bf16.data_ptr(), A_src_bf16.data_ptr() == iris_ptr))
    print("fp8.data_ptr  = 0x%x  (same as iris: %s)" % (
        A_src_fp8.data_ptr(), A_src_fp8.data_ptr() == iris_ptr))
    # Heap base approx
    heap_base = iris_ptr  # first alloc is at heap base
    for name, ptr in [("bf16", A_src_bf16.data_ptr()), ("fp8", A_src_fp8.data_ptr())]:
        offset = ptr - heap_base
        in_heap = 0 <= offset < 512 * 1024 * 1024
        print("  %s offset from heap=%d  in_heap=%s" % (name, offset, in_heap))
iris.barrier()
