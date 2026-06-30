import os, ctypes, struct
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = comm.Get_size()

lib = ctypes.CDLL("libamdhip64.so")
lib.hipGetErrorString.restype = ctypes.c_char_p
lib.hipSetDevice(rank % 8)

# Alloc fine-grained memory
ptr = ctypes.c_void_p()
ret = lib.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(4096), ctypes.c_uint(0x1))
assert ret == 0, f"rank={rank} fine-grained alloc failed: {lib.hipGetErrorString(ret)}"

# Write rank value into the buffer
val_arr = (ctypes.c_float * 1)(float(rank * 100 + 1))
lib.hipMemcpy(ptr, ctypes.cast(val_arr, ctypes.c_void_p), ctypes.c_size_t(4), ctypes.c_int(1))  # H2D

# Get IPC handle (64 bytes)
class IpcHandle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_byte * 64)]
handle = IpcHandle()
ret = lib.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
assert ret == 0, f"rank={rank} IpcGetMemHandle failed: {lib.hipGetErrorString(ret)}"

handle_bytes = bytes(handle.data)
all_handles = comm.allgather(handle_bytes)
all_ptrs_int = comm.allgather(ptr.value)
comm.Barrier()

# Each rank reads from rank (rank+1)%world via IPC
src_rank = (rank + 1) % world
remote_handle_bytes = all_handles[src_rank]
remote_handle = IpcHandle()
ctypes.memmove(remote_handle.data, remote_handle_bytes, 64)

remote_ptr = ctypes.c_void_p()
ret = lib.hipIpcOpenMemHandle(ctypes.byref(remote_ptr), remote_handle, ctypes.c_uint(1))
if ret != 0:
    print(f"rank={rank} IpcOpenMemHandle FAILED ret={ret} ({lib.hipGetErrorString(ret)})", flush=True)
else:
    # Read the value
    val_out = (ctypes.c_float * 1)(0.0)
    lib.hipMemcpy(ctypes.cast(val_out, ctypes.c_void_p), remote_ptr, ctypes.c_size_t(4), ctypes.c_int(2))  # D2H
    expected = float(src_rank * 100 + 1)
    ok = abs(val_out[0] - expected) < 0.1
    print(f"rank={rank} reads from rank={src_rank}: got={val_out[0]} expected={expected} {'OK' if ok else 'WRONG'}", flush=True)
