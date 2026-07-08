#!/usr/bin/env python3
# ================================================================================================
# probe_combine_fanout.py — DIRECT on-device demonstration that `combine_pull` drops cross-rank
# contributions, which is exactly what real EP8 top-8 routing produces on every token.
#
# THE CLAIM UNDER TEST
#   combine_pull_kernel (kernel.cpp:1971) -> tilecomm::tile_reduce_scatter (tilecomm_device.h)
#   reduces a destination cell's LOCAL rows in fp32 and then does a PLAIN STORE of one bf16 row:
#       tilecomm_device.h:151   if (local) *d = out.v; else ctx.store<uint4>(d, out.v, dst_rank);
#   ("No atomics -- a private accumulator per (tile, element)", tilecomm_device.h:77.)
#   So if TWO producer ranks both hold rows for the same (dst_rank, dst_token), they both store and
#   one partial sum is silently lost.
#
#   real_route.combine_fanout(): under a real top-8 router over 256 experts / 32 per rank,
#       mean distinct expert-owner ranks per origin token = 5.33, and 100.0% of tokens have fanout > 1.
#   The synthetic route (b1_dispatch_route.build_multisource_route) has fanout == 1, and example.py
#   runs the region on ONE rank -- so the benchmark can never observe this.
#
# THE EXPERIMENT (no aiter, no MORI -- just IRIS + tk_kernel)
#   Every rank r owns ONE packed row whose value is (r+1), weight 1.0, all routed to the SAME
#   destination cell (dst_rank=0, dst_token=0).
#     correct answer  : accb[0,0] = sum_{r=0..7} (r+1) = 36
#     plain-store answer: accb[0,0] = (some single r)+1  in {1..8}
#
#   Then repeat with fanout=1 (only rank 0 holds a row) to show the kernel IS correct there.
#
# Run: mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 python3 probe_combine_fanout.py
# ================================================================================================
import os
os.environ.setdefault("HSA_XNACK", "1")
import sys
DK = os.environ.get("DK_ROOT", "/home/subvadla/HipKittens/distributed-kernels")
sys.path.insert(0, DK)
sys.path.insert(0, os.path.join(DK, "b1_dispatch"))

import numpy as np
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
import torch
import iris_py
import tk_kernel

H = int(os.environ.get("H", "512"))          # small; the bug is independent of width
TLOCAL = 4
STORE_GRAN = int(os.environ.get("COMBINE_GRAN", "8"))

iris = iris_py.Iris(heap_size_mb=64, verbose=False)
rank, world = iris.rank(), iris.world_size()
torch.cuda.set_device(rank)
comm = MPI.COMM_WORLD


def _view(t, ts, shape, td):
    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)


def make_iris(shape, dtype):
    alloc = "float32" if dtype == "int32" else dtype
    t = iris.empty(list(shape), dtype=alloc)
    dmap = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"),
            "int32": (torch.int32, "<i4")}
    td, ts = dmap[dtype]
    return _view(t, ts, shape, td)


ACC = make_iris([TLOCAL, H], "bfloat16")     # the remote store target, on every rank
ctx = iris.get_device_view()


def run_case(fanout_all_ranks: bool):
    """Every rank (or only rank 0) contributes one row to cell (dst_rank=0, dst_token=0)."""
    ACC.zero_()
    torch.cuda.synchronize()
    iris.barrier()

    contributes = fanout_all_ranks or (rank == 0)
    if contributes:
        Mp = 1
        c2 = torch.full((Mp, H), float(rank + 1), dtype=torch.bfloat16, device='cuda')
        wgt = torch.ones(Mp, 1, dtype=torch.float32, device='cuda')
        cell_dst = torch.tensor([[0, 0]], dtype=torch.int32, device='cuda')      # -> rank 0, token 0
        cell_ptr = torch.tensor([[0], [1]], dtype=torch.int32, device='cuda')    # CSR: cell 0 = rows[0:1]
        cell_rows = torch.tensor([[0]], dtype=torch.int32, device='cuda')
        tk_kernel.combine_pull(c2, ACC, wgt, cell_dst, cell_ptr, cell_rows, ctx,
                               1, H, TLOCAL, STORE_GRAN)
    torch.cuda.synchronize()
    iris.barrier()

    got = float(ACC[0, 0].float().item()) if rank == 0 else 0.0
    return comm.bcast(got, root=0)


# ---- case A: fanout = 8 (what REAL top-8 routing produces on 100% of tokens) --------------------
got_a = run_case(fanout_all_ranks=True)
want_a = sum(r + 1 for r in range(world))          # 36

# ---- case B: fanout = 1 (what the SYNTHETIC route produces) -------------------------------------
got_b = run_case(fanout_all_ranks=False)
want_b = 1.0

if rank == 0:
    print("\n" + "=" * 92)
    print("combine_pull cross-rank accumulation probe   (store_gran=%d, H=%d, world=%d)" %
          (STORE_GRAN, H, world))
    print("=" * 92)
    print(f"  case A  fanout={world} (REAL top-8 routing: 100% of tokens, mean fanout 5.33)")
    print(f"          expected accb[0,0] = {want_a:.1f}   got = {got_a:.1f}   "
          f"-> {'PASS' if abs(got_a - want_a) < 0.5 else 'FAIL — contributions DROPPED'}")
    print(f"  case B  fanout=1  (SYNTHETIC route: build_multisource_route, and the only case")
    print(f"                     example.py's single-CONSUMER run can ever produce)")
    print(f"          expected accb[0,0] = {want_b:.1f}   got = {got_b:.1f}   "
          f"-> {'PASS' if abs(got_b - want_b) < 0.5 else 'FAIL'}")
    print()
    if abs(got_a - want_a) >= 0.5 and abs(got_b - want_b) < 0.5:
        print("  CONFIRMED: combine_pull is correct at fanout=1 and LOSES partial sums at fanout>1.")
        print("  It reduces a cell's LOCAL rows then plain-stores (tilecomm_device.h:123/135/151);")
        print("  it never accumulates across producer ranks. MORI's EpCombineIntraNodeKernel does")
        print("  (intranode.hpp:674-698: pull from every peer's staging buffer + core::WarpAccum).")
        print("  => the '386 us pull-combine beats MORI's 398 us' claim compares different work.")
        print("     The cross-rank-correct variant is combine_scatter (ctx.fetch_add) at 788 us.")
    print("=" * 92 + "\n")

iris.barrier()
comm.Barrier()
MPI.Finalize()
