#!/usr/bin/env python3
# irisx/harness/harness_common.py
# ================================================================================================
# UNIFIED HARNESS — shared tensor / quant / correctness / timing utilities for ALL B-cases.
#
# This is the single source of truth that guarantees apples-to-apples comparison across
# B0..B5: identical M,N,K,dtype; identical A fp8 bytes + per-128 fp32 scales; identical B
# layout/dtype; identical output dtype; identical RMS-rel correctness check; identical
# zero-sentinel; per_expert-vs-aggregate M labelling never silently mixed.
#
# NO GPU CODE RUNS AT IMPORT.  GPU work happens only when a B-case driver (run_harness.py)
# actually calls these helpers, and ONLY the main agent runs that on-device.
#
# The numerics here are copied EXACTLY from the canonical kernels so the reference matches what
# the device produces:
#   - quant scheme  == v3_fused_kernel/example.py::quantize_v1  (per-128 group, scale=amax/448)
#   - dequant       == kernel.cpp fp8_to_f32 * scale            (OCP e4m3 on gfx950)
#   - correctness   == v3/v4 example.py check (RMS-rel vs bf16-of-dequant reference)
#   - zero-sentinel == rank!=src local A buffer forced to 0 -> a local read yields C==0
# ================================================================================================
import os
import time

# torch is imported lazily inside functions so this module can be imported (and the CSV header
# emitted, --dry-run flows exercised) WITHOUT a GPU / torch-HIP present.  Design-only / compile
# checks therefore never touch the device.
QGROUP = 128
FP8_E4M3_MAX = 448.0

# Exact CSV header from irisx/results/results.csv (DO NOT REORDER — main agent appends rows).
CSV_HEADER = [
    "candidate", "commit", "date", "ranks", "route_dist", "M_label", "M", "N", "K", "dtype",
    "BM", "BN", "BK", "NSUB", "schedule", "grid_blocks", "VGPR", "AGPR", "SGPR", "LDS", "scratch",
    "lat_us", "p50", "p95", "p99", "TFLOPs", "rms_rel", "zero_sentinel",
    "spd_vs_B1", "spd_vs_B2", "spd_vs_B3", "notes",
]

# M_label is an ENUM — never silently mix per-expert and aggregate rows.
M_LABEL_PER_EXPERT = "per_expert"   # M is one expert's routed-row count M_e
M_LABEL_AGGREGATE = "aggregate"     # M is total routed rows across the packed buffer


# ----------------------------------------------------------------------------------------------
# IRIS symmetric-heap tensor wrappers (copied verbatim from v3/v4 example.py so heap-offset
# semantics are byte-identical across cases).  Only used by the comm cases (B1..B5).
# ----------------------------------------------------------------------------------------------
def make_iris_tensor(iris, shape, dtype):
    import torch
    t = iris.empty(shape, dtype=dtype)
    dtype_map = {"bfloat16": (torch.bfloat16, "<u2"),
                 "float32":  (torch.float32, "<f4"),
                 "float16":  (torch.float16, "<u2")}
    torch_dtype, typestr = dtype_map[dtype]

    class W:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': typestr,
                                             'data': (ptr, False), 'version': 3, 'strides': None}
            self._keep = t
    w = W(t.data_ptr(), shape, typestr)
    return torch.as_tensor(w, device='cuda').view(torch_dtype).view(*shape)


def make_fp8_iris_tensor(iris, M, K):
    """Two views over the SAME M*K-byte storage (allocated as bf16[M,K/2]):
       - bf16 view [M,K/2]  : the tensor passed to the kernel (gl<bf16> reinterprets bytes)
       - fp8  view [M,K]    : used here to quantize/copy real fp8 values in
    Identical allocation order on all ranks -> identical heap offsets (symmetric heap)."""
    import torch
    assert K % 2 == 0
    t = iris.empty([M, K // 2], "bfloat16")

    def view_as(typestr, shape, td):
        class W:
            def __init__(self, ptr):
                self.__cuda_array_interface__ = {'shape': tuple(shape), 'typestr': typestr,
                                                 'data': (ptr, False), 'version': 3, 'strides': None}
                self._keep = t
        return torch.as_tensor(W(t.data_ptr()), device='cuda').view(td).view(*shape)
    bf16_view = view_as("<u2", (M, K // 2), torch.bfloat16)      # kernel arg
    fp8_view = view_as("|u1", (M, K), torch.float8_e4m3fn)       # quant/copy helper (same storage)
    return bf16_view, fp8_view


# ----------------------------------------------------------------------------------------------
# Canonical quant (must equal v3 example.py quantize_v1 EXACTLY).
# ----------------------------------------------------------------------------------------------
def quantize_v1(A, M, K):
    """A[M,K] fp32 -> (q fp8 e4m3 [M,K], scale fp32 [M,K/128]).  scale = amax/448 per 128-group."""
    import torch
    NG = K // QGROUP
    Ag = A.view(M, NG, QGROUP)
    amax = Ag.abs().amax(dim=2, keepdim=True)
    scale = (amax / FP8_E4M3_MAX).clamp_min(1e-12)
    q = (Ag / scale).to(torch.float8_e4m3fn)
    return q.view(M, K), scale.view(M, NG).contiguous()


def dequant_v1(q, scale, M, K):
    """fp8 q + per-128 scale -> bf16 A, EXACTLY what every kernel reconstructs."""
    import torch
    NG = K // QGROUP
    return (q.to(torch.float32).view(M, NG, QGROUP)
            * scale.view(M, NG, 1)).view(M, K).to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------
# Shared correctness + sentinel check.  IDENTICAL math for every B-case.
#   A_ref      : bf16-of-dequant of the TRUE A (what the kernel should reconstruct)
#   C_got      : kernel output [M,N] (any float dtype)
#   local_a_max: max(|local A buffer|) on the consumer rank (0 == zero-sentinel intact)
# Returns dict: rms_rel, max_rel, max_abs, c_zero, zero_sentinel (True == gather proven), ok.
# ----------------------------------------------------------------------------------------------
def correctness(A_ref_bf16, B_bf16, C_got, local_a_max, tol=0.10):
    import torch
    C_ref = torch.matmul(A_ref_bf16.float(), B_bf16.float().t())
    C_got = C_got.float()
    diff = (C_got - C_ref).abs()
    denom = C_ref.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    rms_rel = (diff.pow(2).mean().sqrt() / C_ref.pow(2).mean().sqrt()).item()
    max_abs = diff.max().item()
    c_zero = bool(C_got.abs().max().item() == 0.0)
    # zero_sentinel == True means: consumer's LOCAL A really was zero, so a non-zero C proves the
    # value came over IRIS from the remote rank (not a local fallback).  For no-comm cases (B0)
    # there is no remote read, so local_a_max is None -> sentinel field is "n/a".
    if local_a_max is None:
        zero_sentinel = "n/a"
        ok = (rms_rel < tol and not c_zero)
    else:
        a_zero = bool(local_a_max == 0)
        zero_sentinel = a_zero
        ok = (rms_rel < tol and not c_zero and a_zero)
    return dict(rms_rel=rms_rel, max_rel=max_rel, max_abs=max_abs,
                c_zero=c_zero, zero_sentinel=zero_sentinel, ok=ok)


# ----------------------------------------------------------------------------------------------
# Timing.  SEPARATE transfer / compute / combined where the case exposes them.
#   run_fn()        : callable that performs ONE iteration of the measured op (sync handled here)
#   sync_fn()       : torch.cuda.synchronize (+ optional iris.barrier) — passed in by driver
# Returns us/iter (p50/p95/p99 computed from the per-iter samples).
# ----------------------------------------------------------------------------------------------
def timed(run_fn, sync_fn, iters=50, warmup=10):
    for _ in range(warmup):
        run_fn()
    sync_fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_fn()
        sync_fn()
        samples.append((time.perf_counter() - t0) * 1e6)   # us
    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    p50 = samples[int(0.50 * (n - 1))]
    p95 = samples[int(0.95 * (n - 1))]
    p99 = samples[int(0.99 * (n - 1))]
    return dict(lat_us=mean, p50=p50, p95=p95, p99=p99)


def tflops(M, N, K, lat_us):
    if lat_us <= 0:
        return 0.0
    return (2.0 * M * N * K) / (lat_us * 1e-6) / 1e12


# ----------------------------------------------------------------------------------------------
# Grouped 32-expert M_e distributions.  Returns a python list of 32 per-expert row counts.
# route_dist in {single, uniform, zipf, onehot, captured}.  Total respects an optional cap.
# ----------------------------------------------------------------------------------------------
def expert_row_counts(route_dist, n_experts=32, m_total=None, m_each=None, captured=None, seed=0):
    import random
    rng = random.Random(seed)
    if route_dist == "single":
        # single-expert mode: one expert holds m_each rows, others 0 (M_label=per_expert path
        # usually drives this directly with a scalar M instead).
        counts = [0] * n_experts
        counts[0] = m_each if m_each is not None else (m_total or 0)
        return counts
    if route_dist == "uniform":
        per = (m_total // n_experts) if m_total else (m_each or 0)
        return [per] * n_experts
    if route_dist == "zipf":
        # Zipf-ish: weight ~ 1/(rank+1), normalized to m_total.
        w = [1.0 / (i + 1) for i in range(n_experts)]
        s = sum(w)
        tot = m_total or (m_each * n_experts if m_each else 0)
        return [max(0, int(round(tot * wi / s))) for wi in w]
    if route_dist == "onehot":
        counts = [0] * n_experts
        counts[rng.randrange(n_experts)] = m_total or (m_each or 0)
        return counts
    if route_dist == "captured":
        assert captured is not None, "route_dist=captured needs --captured <path-to-json-list>"
        assert len(captured) == n_experts
        return list(captured)
    raise ValueError(f"unknown route_dist {route_dist}")


def git_commit_short():
    """Best-effort short commit; '<unknown>' if git not available (e.g. on the node container)."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "<unknown>"


def today():
    return time.strftime("%Y-%m-%d")
