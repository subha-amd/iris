import sys
sys.path.insert(0, "/tmp/HipKittens/distributed-kernels")
import iris_py, torch

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
torch.cuda.set_device(rank)

# Allocate in same order as example.py
MSRC = 4096
K = 7168
NG = K // 128
Mpacked = 8192

# Replicate example.py's make_fp8 and make_iris allocation order
A_src_buf = iris.empty([MSRC, K // 2], "bfloat16")    # A_src_bf16 / A_src_fp8
A_src_sc  = iris.empty([MSRC, NG], "float32")          # A_src_sc
A_pk_buf  = iris.empty([Mpacked, K // 2], "bfloat16")  # A_pk_bf16 / A_pk_fp8
A_pk_sc   = iris.empty([Mpacked, NG], "float32")        # A_pk_sc

# Get heap base for rank (first allocation pointer approximates it)
heap_approx = A_src_buf.data_ptr()

iris.barrier()
if rank == 0:
    print("rank=%d:" % rank)
    print("  A_src_buf ptr = 0x%x  (heap_approx=0x%x  offset=%d)" % (
        A_src_buf.data_ptr(), heap_approx, A_src_buf.data_ptr() - heap_approx))
    print("  A_src_sc  ptr = 0x%x  offset=%d" % (
        A_src_sc.data_ptr(), A_src_sc.data_ptr() - heap_approx))
    print("  A_pk_buf  ptr = 0x%x  offset=%d" % (
        A_pk_buf.data_ptr(), A_pk_buf.data_ptr() - heap_approx))
    print("  A_pk_sc   ptr = 0x%x  offset=%d" % (
        A_pk_sc.data_ptr(), A_pk_sc.data_ptr() - heap_approx))
    print("  heap_size = 512MB = %d bytes" % (512 * 1024 * 1024))
    heap_end = heap_approx + 512 * 1024 * 1024
    for name, ptr in [("A_src", A_src_buf.data_ptr()), ("A_pk", A_pk_buf.data_ptr())]:
        in_heap = heap_approx <= ptr < heap_end
        print("  %s in heap: %s" % (name, in_heap))
iris.barrier()
