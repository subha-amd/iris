import os, ctypes
lib = ctypes.CDLL("libamdhip64.so")
ptr0 = ctypes.c_void_p()
lib.hipMalloc(ctypes.byref(ptr0), ctypes.c_size_t(4096))
ptr = ctypes.c_void_p()
ret = lib.hipExtMallocWithFlags(ctypes.byref(ptr), ctypes.c_size_t(4096), ctypes.c_uint(0x1))
lib.hipGetErrorString.restype = ctypes.c_char_p
rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
xnack = os.environ.get("HSA_XNACK", "UNSET")
err = lib.hipGetErrorString(ctypes.c_int(ret))
print(f"rank={rank} fg_ret={ret} ({err}) ptr={ptr.value} xnack={xnack}", flush=True)
