#!/usr/bin/env python3
# b1_overlap / example.py
# ================================================================================================
# P4 host-scheduled overlap harness.
#
# Runs on np=8.  By default every rank is an active consumer, so all 8 GPUs execute the same shaped
# gather/dequant/GEMM pipeline over their local packed buffers.  The route is deterministic and shared
# across ranks to keep IRIS symmetric heap allocation sizes identical.
#
# Three directly comparable modes use the SAME kernels:
#
#   bulk:
#       gather_pack_range full M + dequant_range full M + grouped_b0_chunk all tasks
#
#   serial:
#       for all chunks: gather_pack_range + dequant_range
#       for all chunks: grouped_b0_chunk
#
#   pipeline:
#       stream_gather: gather_pack_range + dequant_range chunk i
#       stream_gemm:   wait event_i, grouped_b0_chunk chunk i
#
# Pipeline waits are HIP stream events.  No GPU block spin-waits for data.
# ================================================================================================
import os
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../b1_dispatch")

import numpy as np
import torch
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

import iris_py
import tk_kernel
import build_tasks as BT
import b0_tasks as B0T
import b1_dispatch_route as RT


E = int(os.environ.get("E", "32"))
K = int(os.environ.get("K", "7168"))
N = int(os.environ.get("N", "2048"))
TOTAL_M = int(os.environ.get("TOTAL_M", "8192"))
ROUTE = os.environ.get("ROUTE", "uniform")
MSRC = int(os.environ.get("MSRC", "4096"))
SEED = int(os.environ.get("SEED", "1234"))
ITERS = int(os.environ.get("ITERS", "30"))
WARMUP = int(os.environ.get("WARMUP", "5"))
CHUNK_ROWS = int(os.environ.get("CHUNK_ROWS", "1024"))
MODE = os.environ.get("MODE", "all")  # bulk|serial|pipeline|both|all
CHECK = int(os.environ.get("CHECK", "1"))
ALL_CONSUMERS = int(os.environ.get("ALL_CONSUMERS", "1"))
CONSUMER = int(os.environ.get("CONSUMER", "7"))
CSV = os.environ.get("CSV", "b1_overlap_results.csv")
B1_DISPATCH_B0_US = float(os.environ.get("B1_DISPATCH_B0_US", "714.0"))
AITER_UNFUSED_US = float(os.environ.get("AITER_UNFUSED_US", "1255.0"))
REQUIRE_BEATS = os.environ.get("REQUIRE_BEATS", "none").lower()

BM_GATHER = 64
B0_BM = B0T.B0_BM
B0_T_MTILE = 1
B0_T_EROWBEG = 3
QGROUP = 128
NG = K // QGROUP

assert MODE in ("bulk", "serial", "pipeline", "both", "all")
assert REQUIRE_BEATS in ("none", "bulk", "serial", "b1_b0", "aiter")
assert K % QGROUP == 0
assert CHUNK_ROWS % B0_BM == 0, "CHUNK_ROWS must be B0_BM-aligned"
assert CHUNK_ROWS % BM_GATHER == 0, "CHUNK_ROWS must be gather-tile aligned"

iris = iris_py.Iris(heap_size_mb=768, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 8, f"b1_overlap expects np=8, got {world}"
torch.cuda.set_device(rank)
comm = MPI.COMM_WORLD
active = bool(ALL_CONSUMERS or rank == CONSUMER)
CHECK_RANK = 0 if ALL_CONSUMERS else CONSUMER


def make_iris(shape, dtype):
    alloc_dtype = "float32" if dtype == "int32" else dtype
    t = iris.empty(shape, dtype=alloc_dtype)
    dmap = {
        "bfloat16": (torch.bfloat16, "<u2"),
        "float32": (torch.float32, "<f4"),
        "int32": (torch.int32, "<i4"),
    }
    td, ts = dmap[dtype]

    class W:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {
                "shape": tuple(shape),
                "typestr": ts,
                "data": (ptr, False),
                "version": 3,
                "strides": None,
            }
            self._keep = t

    return torch.as_tensor(W(t.data_ptr()), device="cuda").view(td).view(*shape)


def make_fp8(M, Kdim):
    assert Kdim % 2 == 0
    t = iris.empty([M, Kdim // 2], "bfloat16")

    def view(ts, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {
                    "shape": tuple(shape),
                    "typestr": ts,
                    "data": (ptr, False),
                    "version": 3,
                    "strides": None,
                }
                self._keep = t

        return torch.as_tensor(W(t.data_ptr()), device="cuda").view(td).view(*shape)

    return view("<u2", (M, Kdim // 2), torch.bfloat16), view("|u1", (M, Kdim), torch.float8_e4m3fn)


# Host routing and task metadata.  Same route on every rank keeps symmetric heap shapes identical.
rng = np.random.default_rng(SEED)
rows_per_expert = BT.ROUTE_BUILDERS[ROUTE](E, TOTAL_M, rng)
tasks_np, expert_row_begin, padded_rows, Mpacked = B0T.build_b0_tasks(rows_per_expert, N)
num_tasks = int(tasks_np.shape[0])
assert Mpacked > 0 and num_tasks > 0
assert Mpacked % B0_BM == 0

segs, tiles = RT.build_multisource_route(
    world, MSRC, E, rows_per_expert, expert_row_begin, padded_rows,
    Mpacked, BM_GATHER, seed=SEED + 1,
)
seg_arr = RT.segs_to_int_array(segs)
tile_arr = RT.tiles_to_int_array(tiles)
Nseg = int(seg_arr.shape[0])
Ntile = int(tile_arr.shape[0])
assert Ntile == (Mpacked + BM_GATHER - 1) // BM_GATHER


def task_row(t):
    return int(t[B0_T_EROWBEG]) + int(t[B0_T_MTILE]) * B0_BM


chunks = []
for row0 in range(0, Mpacked, CHUNK_ROWS):
    row1 = min(row0 + CHUNK_ROWS, Mpacked)
    mask = np.array([(row0 <= task_row(t) < row1) for t in tasks_np], dtype=bool)
    ctasks = np.ascontiguousarray(tasks_np[mask])
    tile0 = row0 // BM_GATHER
    tile_count = (row1 - row0) // BM_GATHER
    chunks.append({
        "row0": row0,
        "row_count": row1 - row0,
        "tile0": tile0,
        "tile_count": tile_count,
        "tasks_np": ctasks,
    })

if rank == 0:
    print(
        f"[b1-overlap] mode={MODE} active={'all' if ALL_CONSUMERS else CONSUMER} "
        f"route={ROUTE} E={E} TOTAL_M={TOTAL_M} Mpacked={Mpacked} N={N} K={K} "
        f"chunks={len(chunks)} chunk_rows={CHUNK_ROWS} tasks={num_tasks} segs={Nseg}",
        flush=True,
    )


# Symmetric heap tensors.  Every rank allocates the same shapes in the same order.
A_src_bf16, A_src_fp8 = make_fp8(MSRC, K)
A_src_sc = make_iris([MSRC, NG], "float32")
A_pk_bf16, A_pk_fp8 = make_fp8(Mpacked, K)
A_pk_sc = make_iris([Mpacked, NG], "float32")
SEG = make_iris([Nseg, 5], "int32")
TILE = make_iris([Ntile, 4], "int32")

# Local tensors.  A_bf16 is the explicit local dequant cache consumed by grouped_b0_chunk.
B = torch.zeros(E * N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(Mpacked, N, dtype=torch.bfloat16, device="cuda")
A_bf16 = torch.zeros(Mpacked, K, dtype=torch.bfloat16, device="cuda")
TASK_CHUNKS = [
    torch.from_numpy(ch["tasks_np"]).to(device="cuda", dtype=torch.int32).contiguous()
    for ch in chunks
]
TASK_ALL = torch.from_numpy(tasks_np).to(device="cuda", dtype=torch.int32).contiguous()

# Fill each rank's source activation buffer.
gnp = np.random.default_rng(1000 + rank)
A_real = (gnp.standard_normal((MSRC, K)).astype(np.float32) / 8.0)
q_u8, sc_f32, deq_f32 = RT.quantize_v1(A_real, K)
A_src_fp8.copy_(torch.from_numpy(q_u8.view(np.uint8)).cuda().view(torch.float8_e4m3fn).view(MSRC, K))
A_src_sc.copy_(torch.from_numpy(sc_f32).cuda())

deq_all = comm.gather(deq_f32, root=CHECK_RANK) if CHECK else None

torch.manual_seed(777)
B.copy_(torch.randn(E * N, K, dtype=torch.bfloat16, device="cuda") / 8.0)

SEG.copy_(torch.from_numpy(seg_arr).cuda())
TILE.copy_(torch.from_numpy(tile_arr).cuda())
iris.barrier()

iris_ctx = iris.get_device_view()
gather_stream = torch.cuda.Stream()
gemm_stream = torch.cuda.Stream()
chunk_events = [torch.cuda.Event(blocking=False) for _ in chunks]


def clear_outputs():
    A_pk_fp8.view(torch.uint8).zero_()
    A_pk_sc.zero_()
    A_bf16.zero_()
    C.zero_()


def schedule_gather_dequant(ch):
    tk_kernel.dispatch_gather_pack_range(
        A_src_bf16, A_src_sc, A_pk_bf16, A_pk_sc, SEG, TILE, iris_ctx,
        MSRC, Mpacked, K, Nseg, Ntile, ch["tile0"], ch["tile_count"],
    )
    tk_kernel.dequant_packed_range(
        A_pk_bf16, A_pk_sc, A_bf16,
        Mpacked, K, ch["row0"], ch["row_count"],
    )


def schedule_gemm_chunk(i):
    tasks = TASK_CHUNKS[i]
    if tasks.numel() == 0:
        return
    tk_kernel.grouped_b0_chunk(A_bf16, B, C, tasks, N, K, int(tasks.shape[0]))


def run_bulk():
    if not active:
        return
    schedule_gather_dequant({
        "tile0": 0,
        "tile_count": Ntile,
        "row0": 0,
        "row_count": Mpacked,
    })
    tk_kernel.grouped_b0_chunk(A_bf16, B, C, TASK_ALL, N, K, num_tasks)


def run_serial():
    if not active:
        return
    for ch in chunks:
        schedule_gather_dequant(ch)
    for i in range(len(chunks)):
        schedule_gemm_chunk(i)


def run_pipeline():
    if not active:
        return
    default = torch.cuda.current_stream()
    gather_stream.wait_stream(default)
    gemm_stream.wait_stream(default)
    for i, ch in enumerate(chunks):
        with torch.cuda.stream(gather_stream):
            schedule_gather_dequant(ch)
            chunk_events[i].record(gather_stream)
        with torch.cuda.stream(gemm_stream):
            gemm_stream.wait_event(chunk_events[i])
            schedule_gemm_chunk(i)
    default.wait_stream(gather_stream)
    default.wait_stream(gemm_stream)


def timed(fn):
    for _ in range(WARMUP):
        if active:
            clear_outputs()
            fn()
        torch.cuda.synchronize()
        iris.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_us = 0.0
    for _ in range(ITERS):
        if active:
            clear_outputs()
            start.record()
            fn()
            end.record()
            end.synchronize()
            total_us += start.elapsed_time(end) * 1e3
        torch.cuda.synchronize()
        iris.barrier()
    local = total_us / max(ITERS, 1) if active else 0.0
    return comm.allreduce(local, op=MPI.MAX)


def build_reference_check_rank():
    A_deq = np.zeros((Mpacked, K), dtype=np.float32)
    for s in segs:
        src = deq_all[s["src_rank"]]
        sr, dr, rc = s["src_row_begin"], s["dst_row_begin"], s["row_count"]
        A_deq[dr:dr + rc, :] = src[sr:sr + rc, :]
    C_ref = np.zeros((Mpacked, N), dtype=np.float32)
    Bcpu = B.float().cpu().numpy()
    for e in range(E):
        m_e = int(rows_per_expert[e])
        if m_e == 0:
            continue
        base = int(expert_row_begin[e])
        Be = Bcpu[e * N:(e + 1) * N, :]
        C_ref[base:base + m_e] = A_deq[base:base + m_e] @ Be.T
    return A_deq, C_ref


def check_on_check_rank(label):
    if not CHECK:
        return None
    if active:
        clear_outputs()
        run_pipeline()
    torch.cuda.synchronize()
    iris.barrier()
    if rank != CHECK_RANK:
        return None

    A_ref, C_ref = build_reference_check_rank()
    A_got = A_bf16.float().cpu().numpy()
    C_got = C.float().cpu().numpy()

    real = np.zeros(Mpacked, dtype=bool)
    routed = np.zeros(Mpacked, dtype=bool)
    remote_rows = 0
    for e in range(E):
        m_e = int(rows_per_expert[e])
        base = int(expert_row_begin[e])
        if m_e:
            real[base:base + m_e] = True
    for s in segs:
        routed[s["dst_row_begin"]:s["dst_row_begin"] + s["row_count"]] = True
        if s["src_rank"] != rank:
            remote_rows += s["row_count"]

    a_diff = np.abs(A_got - A_ref)
    a_rms = float(np.sqrt((a_diff ** 2).mean()) / max(np.sqrt((A_ref ** 2).mean()), 1e-9))
    c_diff = np.abs(C_got[real] - C_ref[real])
    c_rms = float(np.sqrt((c_diff ** 2).mean()) / max(np.sqrt((C_ref[real] ** 2).mean()), 1e-9))
    zero_rows = ~routed
    sentinel_ok = bool(np.all(C_got[zero_rows] == 0.0)) if zero_rows.any() else True
    ok = bool(a_rms < 0.01 and c_rms < 0.02 and sentinel_ok and remote_rows > 0)
    print(
        f"[{label}] phaseA_bf16_RMS={a_rms:.6f} C_RMS={c_rms:.6f} "
        f"zero_sentinel={sentinel_ok} remote_rows={remote_rows} -> {'PASSED' if ok else 'FAILED'}",
        flush=True,
    )
    return c_rms, ok


check_on_check_rank("b1-overlap pipeline")

results = {}
if MODE in ("bulk", "all"):
    results["bulk"] = timed(run_bulk)
if MODE in ("serial", "both", "all"):
    results["serial"] = timed(run_serial)
if MODE in ("pipeline", "both", "all"):
    results["pipeline"] = timed(run_pipeline)

exit_code = 0
if "pipeline" in results:
    targets = {
        "bulk": results.get("bulk"),
        "serial": results.get("serial"),
        "b1_b0": B1_DISPATCH_B0_US,
        "aiter": AITER_UNFUSED_US,
    }
    target = targets.get(REQUIRE_BEATS)
    if REQUIRE_BEATS != "none":
        if target is None:
            exit_code = 2
        elif results["pipeline"] >= target:
            exit_code = 3

if rank == 0:
    real_rows = int(np.sum(rows_per_expert))
    flops = 2.0 * real_rows * N * K
    print("=" * 92, flush=True)
    for name, us in results.items():
        tflops = flops / (us * 1e-6) / 1e12
        print(f"  {name:<8} total: {us:8.2f} us   {tflops:7.2f} TFLOP/s e2e", flush=True)
    if "pipeline" in results and "bulk" in results:
        spd = results["bulk"] / results["pipeline"] if results["pipeline"] > 0 else 0.0
        print(f"  speedup pipeline vs bulk:   {spd:.3f}x", flush=True)
    if "serial" in results and "pipeline" in results:
        spd = results["serial"] / results["pipeline"] if results["pipeline"] > 0 else 0.0
        print(f"  speedup pipeline vs serial: {spd:.3f}x", flush=True)
    if "pipeline" in results:
        p = results["pipeline"]
        print(
            f"  speedup pipeline vs verified b1_dispatch b0 ({B1_DISPATCH_B0_US:.1f} us): "
            f"{B1_DISPATCH_B0_US / p:.3f}x",
            flush=True,
        )
        print(
            f"  speedup pipeline vs local aiter unfused ({AITER_UNFUSED_US:.1f} us): "
            f"{AITER_UNFUSED_US / p:.3f}x",
            flush=True,
        )
        if REQUIRE_BEATS != "none":
            target = {
                "bulk": results.get("bulk"),
                "serial": results.get("serial"),
                "b1_b0": B1_DISPATCH_B0_US,
                "aiter": AITER_UNFUSED_US,
            }.get(REQUIRE_BEATS)
            if target is None:
                print(f"  REQUIRE_BEATS={REQUIRE_BEATS}: FAILED (target mode was not run)", flush=True)
            else:
                verdict = "PASSED" if p < target else "FAILED"
                print(f"  REQUIRE_BEATS={REQUIRE_BEATS}: {verdict} ({p:.2f} us vs {target:.2f} us)", flush=True)
    print("=" * 92, flush=True)

    new = not os.path.exists(CSV)
    with open(CSV, "a") as f:
        if new:
            f.write(
                "candidate,mode,ranks,route,Mpacked,N,K,chunk_rows,chunks,total_us,tflops,"
                "b1_dispatch_b0_us,aiter_unfused_us,notes\n"
            )
        for name, us in results.items():
            tflops = flops / (us * 1e-6) / 1e12
            f.write(
                f"b1-overlap,{name},{world},{ROUTE},{Mpacked},{N},{K},"
                f"{CHUNK_ROWS},{len(chunks)},{us:.2f},{tflops:.2f},"
                f"{B1_DISPATCH_B0_US:.2f},{AITER_UNFUSED_US:.2f},"
                "host-stream-event chunk pipeline\n"
            )

import gc
del A_src_bf16, A_src_fp8, A_src_sc, A_pk_bf16, A_pk_fp8, A_pk_sc, A_bf16
del SEG, TILE, B, C, TASK_CHUNKS, TASK_ALL
gc.collect()
torch.cuda.synchronize()
iris.barrier()
del iris_ctx, iris
gc.collect()
torch.cuda.synchronize()
MPI.Finalize()
os._exit(exit_code)
