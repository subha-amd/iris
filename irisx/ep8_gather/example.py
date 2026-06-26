#!/usr/bin/env python3
# ep8_gather / example.py
# ================================================================================================
# np=8 (EP8) driver for the multi-source gather PROBE (main agent runs this on the node under flock).
#
# Layout (symmetric IRIS heap; IDENTICAL allocation order on every rank => identical offsets):
#   per rank r:   A_fp8[Msrc,K] (fp8 bytes via bf16 view) + A_sc[Msrc,NG]   <- rank r's OWN activations
#   consumer (rank 0) additionally: D[Mpacked,K] (output), SEG[Nseg,5], TILE[Ntile,4]
#
# Each rank fills its own A_fp8/A_sc with quantized random activations.  The CPU reference builds the
# routing (route_segments) + per-tile metadata and the expected dequantized gather D_ref, then we
# launch the probe ON THE CONSUMER RANK ONLY, which gathers rows from all 8 ranks via IRIS and writes
# D.  Verify: D == D_ref (RMS-rel small) AND every unrouted/tail packed row is EXACTLY zero
# (per-rank zero-sentinel), AND at least one gathered row came from a REMOTE rank (proves XGMI path).
# ================================================================================================
import sys, os
sys.path.insert(0, "..")
import numpy as np
import torch
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import iris_py
import tk_kernel
from mpi4py import MPI
import ep8_multisource_ref as ref

# ---- shapes (override via env) ----
Msrc    = int(os.environ.get("MSRC", "128"))      # rows per source rank's activation buffer
Mpacked = int(os.environ.get("MPACKED", "256"))   # packed rows on the consumer
K       = int(os.environ.get("K", "256"))
BM      = int(os.environ.get("BM", "64"))
CONSUMER = int(os.environ.get("CONSUMER", "0"))
SEED    = int(os.environ.get("SEED", "3"))
QGROUP  = 128
assert K % QGROUP == 0
NG = K // QGROUP

iris = iris_py.Iris(heap_size_mb=256, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 8, f"EP8 probe expects np=8, got world={world}"
torch.cuda.set_device(rank)

def make_iris(shape, dtype):
    t = iris.empty(shape, dtype=dtype)
    dmap = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"),
            "int32": (torch.int32, "<i4")}
    td, ts = dmap[dtype]
    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)

def make_fp8(M, K):
    t = iris.empty([M, K // 2], "bfloat16")        # M*K bytes
    def view(ts, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    return view("<u2", (M, K // 2), torch.bfloat16), view("|u1", (M, K), torch.float8_e4m3fn)

# IMPORTANT: identical allocation ORDER on every rank -> identical symmetric-heap offsets.
A_fp8_bf16, A_fp8 = make_fp8(Msrc, K)
A_sc = make_iris([Msrc, NG], "float32")
# Consumer-only buffers still allocated on EVERY rank (keep offsets symmetric); only consumer uses them.
D    = make_iris([Mpacked, K], "bfloat16")
Ntile = (Mpacked + BM - 1) // BM
# routing built by reference on rank CONSUMER, broadcast so SEG/TILE sizes match everywhere.
comm = MPI.COMM_WORLD
if rank == CONSUMER:
    segs, tiles = ref.make_routing(world, Msrc, Mpacked, BM, seed=SEED)
    seg_arr  = ref.segs_to_int_array(segs)
    tile_arr = ref.tiles_to_int_array(tiles)
else:
    segs = tiles = None
    seg_arr = tile_arr = None
seg_arr  = comm.bcast(seg_arr,  root=CONSUMER)
tile_arr = comm.bcast(tile_arr, root=CONSUMER)
Nseg = seg_arr.shape[0]
SEG  = make_iris([Nseg, 5], "int32")
TILE = make_iris([Ntile, 4], "int32")

# ---- each rank fills its OWN activations; quantize like V4 ----
g = np.random.default_rng(1000 + rank)
A_real = (g.standard_normal((Msrc, K)).astype(np.float32) / 8.0)
q_u8, sc_f32, deq_f32 = ref.quantize_v1(A_real, K)
A_fp8.copy_(torch.from_numpy(q_u8.view(np.uint8)).cuda().view(torch.float8_e4m3fn).view(Msrc, K))
A_sc.copy_(torch.from_numpy(sc_f32).cuda())

# gather every rank's dequantized buffer to the consumer for the reference
deq_all = comm.gather(deq_f32, root=CONSUMER)

if rank == CONSUMER:
    SEG.copy_(torch.from_numpy(seg_arr).cuda())
    TILE.copy_(torch.from_numpy(tile_arr).cuda())
    D.view(torch.uint16).zero_()      # start zeroed: unrouted rows must STAY zero
iris.barrier()                        # <-- the happens-before for the read-only remote gather

ctx = iris.get_device_view()
if rank == CONSUMER:
    tk_kernel.dispatch_gather(A_fp8_bf16, A_sc, D, SEG, TILE, ctx,
                              Msrc, Mpacked, K, Nseg, Ntile)
torch.cuda.synchronize()
iris.barrier()

# ---- verify on the consumer ----
if rank == CONSUMER:
    D_ref = ref.reference_gather(segs, Mpacked, K, deq_all).astype(np.float32)
    D_got = D.float().cpu().numpy()

    # which packed rows are routed vs zero-sentinel
    routed = np.zeros(Mpacked, dtype=bool)
    remote_rows = 0
    for s in segs:
        routed[s["dst_row_begin"]:s["dst_row_begin"] + s["row_count"]] = True
        if s["src_rank"] != CONSUMER:
            remote_rows += s["row_count"]
    zero_rows = ~routed

    diff = np.abs(D_got - D_ref)
    denom = np.maximum(np.abs(D_ref), 1e-6)
    rms_rel = float(np.sqrt((diff**2).mean()) / max(np.sqrt((D_ref**2).mean()), 1e-9))
    max_rel = float((diff / denom).max())

    sentinel_ok = bool(np.all(D_got[zero_rows] == 0.0)) if zero_rows.any() else True
    routed_nonzero = bool(np.any(D_got[routed] != 0.0)) if routed.any() else False
    remote_ok = remote_rows > 0

    n_single = sum(1 for t in tiles if t["seg_count"] == 1)
    n_multi  = sum(1 for t in tiles if t["seg_count"] > 1)

    ok = (rms_rel < 1e-2 and sentinel_ok and routed_nonzero and remote_ok)
    print("=" * 84, flush=True)
    print(f"[EP8 gather] Msrc={Msrc} Mpacked={Mpacked} K={K} world={world} segs={Nseg} "
          f"tiles={Ntile}", flush=True)
    print(f"  fast-path tiles (single-source)  = {n_single}", flush=True)
    print(f"  segment-iterator tiles (straddle)= {n_multi}", flush=True)
    print(f"  remote (XGMI) gathered rows      = {remote_rows}", flush=True)
    print(f"  RMS_rel={rms_rel:.6f}  max_rel={max_rel:.4f}", flush=True)
    print(f"  zero_sentinel_rows_exact_zero    = {sentinel_ok}", flush=True)
    print(f"  routed_rows_nonzero              = {routed_nonzero}", flush=True)
    print(f"  -> {'PASSED' if ok else 'FAILED'}", flush=True)
    print("=" * 84, flush=True)

import gc
del A_fp8_bf16, A_fp8, A_sc, D, SEG, TILE
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
