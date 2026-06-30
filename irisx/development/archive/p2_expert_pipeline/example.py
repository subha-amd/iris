#!/usr/bin/env python3
# p2_expert_pipeline / example.py  (SCAFFOLD; MAIN AGENT runs on the node under flock, np=2 first)
# ================================================================================================
# CANDIDATE P2 — expert-granular copy-once double-buffer. PRODUCER gathers each expert's A region
# ONCE into a 2-slot rotating LOCAL bf16 buffer (slot=e%2) with a per-expert ready + 2-deep slot-
# recycle handshake; CONSUMER = B0 grouped GEMM per expert reading from the slot. Overlaps gather of
# expert e+1 with the GEMM of expert e. Metric = T_pipeline.
#
# np=2 topology (single source rank, like B1/v5_grouped — the np=8 multi-source variant reuses the
# SAME kernel via route_segments spanning ranks; bring up np=2 first):
#   Rank 0 (SRC_RANK) = "producer rank": holds the ONLY real fp8 packed-A[Mpacked,K] + scales.
#   Rank 1            = "consumer rank": runs p2_producer + p2_consumer concurrently on two streams.
#
# DEADLOCK AVOIDANCE: two separate launches / two non-blocking streams (producer first reserves CUs);
#   strict depth-2 slot queue (producer waits done[e]; consumer waits ready[e]; last consumer tile of
#   e sets done[e+2]); host primes done[0],done[1]=ready and pre-sets done[e+2] for EMPTY experts so
#   the recycle chain never stalls. No consumer-to-consumer dependency.
#
# SYMMETRIC BARRIERS: both ranks run the SAME number of run()/iris.barrier() calls; only rank1
#   launches kernels. Asymmetric barrier counts deadlock (bit Agent 01).
#
# THIS IS A SCAFFOLD: host-side route/schedule build is sketched; the main agent fills
# build_p2_schedule() (pure CPU) to emit expert_meta / segs / tasks / arrive consistent with the ABI
# below, and runs under flock. The np=2 single-source case is the simplest: ONE segment per expert,
# src_rank=0, dst_row_begin slot-local.
#
# ABI (must match kernel.cpp):
#   slots       : [2*slot_rows, K] bf16     two rotating expert slots
#   segs        : [Nseg, 5] int32           (expert_id,src_rank,src_row_begin,dst_row_begin,row_count)
#                                            dst_row_begin is SLOT-LOCAL (expert region starts at 0)
#   expert_meta : [E, 4] int32              (valid_rows, seg_begin, seg_count, padded_rows)
#   tasks       : [num_tasks, 5] int32      (expert, m_tile, n_tile, slot_rows, erow_begin)
#   ready       : [E] int32                 producer->consumer
#   done        : [E+2] int32               consumer->producer (done[0],done[1] primed)
#   arrive      : [E] int32                 per-expert tile count, consumer counts DOWN to 0
#   ecursor     : [1] int32                 producer expert dispenser
# ================================================================================================
import sys, os, time
sys.path.insert(0, "..")
import numpy as np
import torch
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import iris_py
import tk_kernel
from mpi4py import MPI

torch.manual_seed(0)

# ---- shapes / route (override via env) ----
E       = int(os.environ.get("E", "32"))
K       = int(os.environ.get("K", "7168"))
N       = int(os.environ.get("N", "2048"))
TOTAL_M = int(os.environ.get("TOTAL_M", "8192"))
ROUTE   = os.environ.get("ROUTE", "uniform")       # uniform|zipf|one_hot|many_empty
ITERS   = int(os.environ.get("ITERS", "50"))
WARMUP  = int(os.environ.get("WARMUP", "10"))
NUM_PRODUCER_BLOCKS = int(os.environ.get("PROD_BLOCKS", "8"))
B0_BM, B0_BN = 256, 256
GBM = 64   # ep8_gather_BM (must match kernel.cpp)
QGROUP = 128
assert K % QGROUP == 0
NG = K // QGROUP

iris = iris_py.Iris(heap_size_mb=2048, verbose=False)
rank = iris.rank()
world = iris.world_size()
assert world == 2, f"P2 np=2 bring-up expects world=2, got {world}"
torch.cuda.set_device(rank)
SRC_RANK = 0

# ---- host: build a deterministic route + grouped B0 schedule (pure CPU) ----
rng = np.random.default_rng(20260625)
def build_route(E, TOTAL_M, route):
    if route == "uniform":
        base = TOTAL_M // E
        rows = np.full(E, base, dtype=np.int64); rows[: TOTAL_M - base * E] += 1
    elif route == "many_empty":
        rows = np.zeros(E, dtype=np.int64); rows[:E // 4] = TOTAL_M // (E // 4)
    elif route == "one_hot":
        rows = np.zeros(E, dtype=np.int64); rows[0] = TOTAL_M
    else:  # zipf-ish
        w = 1.0 / (1.0 + np.arange(E)); w /= w.sum()
        rows = np.floor(w * TOTAL_M).astype(np.int64)
    return rows

rows_per_expert = build_route(E, TOTAL_M, ROUTE)
# pad each expert UP to a multiple of B0_BM (no-contamination layout); slot_rows = max padded.
padded = ((rows_per_expert + B0_BM - 1) // B0_BM) * B0_BM
slot_rows = int(max(padded.max(), B0_BM))
# global packed C row begin per expert (back-to-back padded regions on rank0's Mpacked buffer).
erow_begin = np.zeros(E, dtype=np.int64)
erow_begin[1:] = np.cumsum(padded)[:-1]
Mpacked = int(erow_begin[-1] + padded[-1]) if E > 0 else 0
Msrc = Mpacked   # np=2 single-source: src row == dst packed row (one segment per expert, off=0)

# segs: ONE segment per non-empty expert (np=2 single-source). dst_row_begin SLOT-LOCAL (=0).
segs = []
expert_meta = np.zeros((E, 4), dtype=np.int32)
for e in range(E):
    vr = int(rows_per_expert[e]); pr = int(padded[e])
    seg_begin = len(segs)
    if vr > 0:
        segs.append((e, SRC_RANK, int(erow_begin[e]), 0, vr))  # src_row_begin = global packed begin
        seg_count = 1
    else:
        seg_count = 0
    expert_meta[e] = (vr, seg_begin, seg_count, pr)
segs_np = np.array(segs, dtype=np.int32).reshape(-1, 5) if segs else np.zeros((1, 5), np.int32)
Nseg = max(1, len(segs))

# tasks: one B0 256x256 tile per (expert, m_tile, n_tile). arrive[e] = tiles_of_e.
n_ntiles = (N + B0_BN - 1) // B0_BN
tasks = []
arrive = np.zeros(E, dtype=np.int32)
for e in range(E):
    pr = int(padded[e])
    n_mtiles = pr // B0_BM
    cnt = 0
    for mt in range(n_mtiles):
        for nt in range(n_ntiles):
            tasks.append((e, mt, nt, slot_rows, int(erow_begin[e]))); cnt += 1
    arrive[e] = cnt
tasks_np = np.array(tasks, dtype=np.int32).reshape(-1, 5) if tasks else np.zeros((1, 5), np.int32)
num_tasks = len(tasks)

if rank != SRC_RANK:
    print(f"[P2] route={ROUTE} E={E} TOTAL_M={TOTAL_M} -> Mpacked={Mpacked} slot_rows={slot_rows} "
          f"Nseg={Nseg} num_tasks={num_tasks} empty={(rows_per_expert==0).sum()}", flush=True)

# ---- IRIS symmetric-heap tensors (identical allocation ORDER on both ranks) ----
def make_iris(shape, dtype):
    # IRIS python iris.empty supports bfloat16/float32 but NOT int32. int32 and float32 are both
    # 4 bytes -> back int32 buffers with a float32 IRIS alloc + int32 view (device gl<int> only needs
    # the symmetric-heap byte pointer; storage dtype is irrelevant).
    alloc_dtype = "float32" if dtype == "int32" else dtype
    t = iris.empty(shape, dtype=alloc_dtype)
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
    t = iris.empty([M, K // 2], "bfloat16")
    def view(ts, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': ts,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    return view("<u2", (M, K // 2), torch.bfloat16), view("|u1", (M, K), torch.float8_e4m3fn)

Mpk = max(Mpacked, B0_BM)
A_fp8_bf16, A_fp8 = make_fp8(Mpk, K)               # remote fp8 packed-A (rank0)
A_sc  = make_iris([Mpk, NG], "float32")
Bw    = make_iris([E * N, K], "bfloat16")          # local expert-major weights
C     = make_iris([Mpk, N], "bfloat16")            # local output
SLOTS = make_iris([2 * slot_rows, K], "bfloat16")  # two rotating expert slots
SEG   = make_iris([Nseg, 5], "int32")
EMETA = make_iris([E, 4], "int32")
TASKS = make_iris([max(1, num_tasks), 5], "int32")
READY = make_iris([E], "int32")
DONE  = make_iris([E + 2], "int32")
ARRIVE = make_iris([E], "int32")
ECUR  = make_iris([1], "int32")

# ---- reference data ----
gpt = np.random.default_rng(1234)
A_real = (gpt.standard_normal((Mpk, K)).astype(np.float32) / 8.0)
def quantize_v1(A):
    Ag = A.reshape(Mpk, NG, QGROUP)
    amax = np.abs(Ag).max(axis=2, keepdims=True)
    scale = np.clip(amax / 448.0, 1e-12, None)
    q = (Ag / scale)
    return q.reshape(Mpk, K), scale.reshape(Mpk, NG)

q_f, sc_f = quantize_v1(A_real)
q_u8 = torch.from_numpy(q_f).cuda().to(torch.float8_e4m3fn)
A_deq_host = (q_u8.float().cpu().numpy().reshape(Mpk, NG, QGROUP) * sc_f.reshape(Mpk, NG, 1)).reshape(Mpk, K)

if rank == SRC_RANK:
    A_fp8.copy_(q_u8)
    A_sc.copy_(torch.from_numpy(sc_f.astype(np.float32)).cuda())
else:
    A_fp8.view(torch.uint8).zero_()    # SENTINEL: rank1 has no local A -> proves remote gather
    A_sc.zero_()

gw = np.random.default_rng(777)
Bw_np = (gw.standard_normal((E * N, K)).astype(np.float32) / 8.0)
Bw.copy_(torch.from_numpy(Bw_np).cuda().to(torch.bfloat16))
C.zero_()

if rank != SRC_RANK:
    SEG.copy_(torch.from_numpy(segs_np).cuda())
    EMETA.copy_(torch.from_numpy(expert_meta).cuda())
    TASKS.copy_(torch.from_numpy(tasks_np).cuda())
iris.barrier()

iris_ctx = iris.get_device_view()
GEN = 0
def ready_flag(g): return (g << 1) | 1

def reset_state():
    if rank != SRC_RANK:
        rv = ready_flag(GEN)
        READY.zero_(); ECUR.zero_(); SLOTS.zero_()
        ARRIVE.copy_(torch.from_numpy(arrive).cuda())
        d = torch.zeros(E + 2, dtype=torch.int32, device='cuda')
        d[0] = rv; d[1] = rv                       # prime first two slots free
        for e in range(E):                         # empty experts: pre-free their downstream slot
            if expert_meta[e, 0] == 0:
                d[e + 2] = rv
        DONE.copy_(d)
    torch.cuda.synchronize(); iris.barrier()

def run():
    global GEN
    GEN += 1
    reset_state()
    if rank != SRC_RANK:
        tk_kernel.dispatch_p2(
            A_fp8_bf16, A_sc, Bw, C, SLOTS, SEG, EMETA, TASKS,
            READY, DONE, ARRIVE, ECUR, iris_ctx,
            E, N, K, Mpacked, Msrc, slot_rows, GEN, num_tasks, NUM_PRODUCER_BLOCKS)
    torch.cuda.synchronize(); iris.barrier()

def timed():
    for _ in range(WARMUP): run()
    torch.cuda.synchronize(); iris.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS): run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1e6
    iris.barrier()
    return dt

# ---- correctness: per-expert C[rows_e] = dequant(A[packed rows_e]) @ B[e]^T ----
def check(label):
    C.zero_()
    run()
    if rank != SRC_RANK:
        C_got = C.float().cpu().numpy()
        ok = True; max_rel = 0.0; rms_num = 0.0; rms_den = 0.0
        for e in range(E):
            vr = int(rows_per_expert[e])
            if vr == 0: continue
            r0 = int(erow_begin[e])
            A_e = A_deq_host[r0:r0 + vr]                       # [vr, K]
            B_e = Bw_np[e * N:(e + 1) * N]                     # [N, K]
            C_ref = A_e @ B_e.T                                # [vr, N]
            C_e = C_got[r0:r0 + vr]
            diff = np.abs(C_e - C_ref)
            denom = np.maximum(np.abs(C_ref), 1e-6)
            max_rel = max(max_rel, float((diff / denom).max()))
            rms_num += float((diff ** 2).sum()); rms_den += float((C_ref ** 2).sum())
        rms_rel = (rms_num ** 0.5) / max(rms_den ** 0.5, 1e-9)
        a_zero = bool(A_fp8.view(torch.uint8).max().item() == 0)
        c_zero = bool(np.abs(C_got).max() == 0.0)
        ok = (rms_rel < 0.10 and not c_zero and a_zero)
        print(f"[{label}] E={E} TOTAL_M={TOTAL_M} K={K} N={N}  max_rel={max_rel:.4f}  "
              f"RMS_rel={rms_rel:.5f}  local_A_zero={a_zero}  C_zero={c_zero}  "
              f"-> {'PASSED' if ok else 'FAILED'}", flush=True)
        return rms_rel, a_zero
    return None, None

if rank != SRC_RANK:
    print("=" * 84, flush=True)
e, sentinel = check("P2      ")
t = timed()
if rank != SRC_RANK:
    flops = 2.0 * float(rows_per_expert.sum()) * N * K
    print("-" * 84, flush=True)
    print(f"  P2 EXPERT DOUBLE-BUFFER (B0 consumer, overlap): {t:8.2f} us/iter   "
          f"{flops/(t*1e-6)/1e12:6.2f} TFLOP/s (routed-rows)", flush=True)
    print(f"  T_pipeline={t:.1f}us   (compare to a same-route B1-dispatch serial baseline)", flush=True)
    print(f"CSV,P2,<commit>,2,{ROUTE},aggregate,{int(rows_per_expert.sum())},{N},{K},fp8_bf16,"
          f"256,256,64,1,prod+B0grouped,{num_tasks},,,,,,{t:.1f},,,,"
          f"{flops/(t*1e-6)/1e12:.1f},{e:.5f},{sentinel},,,,"
          f"expert_double_buffer_overlap", flush=True)
    print("=" * 84, flush=True)

import gc
del A_fp8_bf16, A_fp8, A_sc, Bw, C, SLOTS, SEG, EMETA, TASKS, READY, DONE, ARRIVE, ECUR
gc.collect(); torch.cuda.synchronize(); iris.barrier()
del iris_ctx, iris
gc.collect(); torch.cuda.synchronize()
MPI.Finalize()
os._exit(0)
