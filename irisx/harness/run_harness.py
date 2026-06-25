#!/usr/bin/env python3
# irisx/harness/run_harness.py
# ================================================================================================
# UNIFIED B-CASE RUNNER.  One driver that benchmarks every candidate (B0..B5) on IDENTICAL
# tensors/layouts/precision/correctness, emitting rows in the exact results.csv schema.
#
#   B0  local V2 HK GEMM only, NO comm                       -> compute ceiling
#   B1  IRISX dispatch-pack-quant A EXACTLY ONCE into a local fp8+scale buffer, then local V2 GEMM
#                                                            -> STRONG unfused IRISX baseline (HEADLINE cites this)
#   B2  production MORI dispatch/pack/quant + AITER/CK fmoe   -> production baseline (STUB; needs node wiring)
#   B3  V3 direct-pull, no overlap (fused=0)                  -> historic weak baseline
#   B4  A-stationary remote pull, no overlap                  -> isolates reuse from overlap
#   B5  V4 A-stationary + overlap (fused=1)                   -> the fused candidate
#
# GPU RULE: this script is run ONLY by the main agent, under flock, np=2 first (np=8 later).
# Subagents NEVER run it.  It imports torch/iris lazily so --list / --dry-run touch no GPU.
#
# Usage (main agent, on node, inside r1_c4, under flock — see AGENT_REPORT.md for exact cmds):
#   M=256 N=2048 K=7168 CASE=B1 mpirun ... -np 2 python3 run_harness.py
#   CASE=all  ROUTE=single  M=256 ...                      (run every callable case)
#   CASE=B5   ROUTE=zipf    M_TOTAL=8192  N_EXPERTS=32 ... (grouped, aggregate M)
#
# Env knobs: M,N,K,ITERS,WARMUP,CASE,ROUTE,M_LABEL,M_TOTAL,N_EXPERTS,CAPTURED,CSV,SRC_RANK,TOL
# ================================================================================================
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")   # so the per-candidate tk_kernel built next to example.py is importable
import harness_common as hc

# ---- shapes / config (env-overridable) -------------------------------------------------------
M = int(os.environ.get("M", "256"))
K = int(os.environ.get("K", "7168"))
N = int(os.environ.get("N", "2048"))
ITERS = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "10"))
CASE = os.environ.get("CASE", "B1")            # B0|B1|B2|B3|B4|B5|all
ROUTE = os.environ.get("ROUTE", "single")      # single|uniform|zipf|onehot|captured
M_LABEL = os.environ.get("M_LABEL", hc.M_LABEL_PER_EXPERT)
N_EXPERTS = int(os.environ.get("N_EXPERTS", "32"))
SRC_RANK = int(os.environ.get("SRC_RANK", "0"))
TOL = float(os.environ.get("TOL", "0.10"))
CSV = os.environ.get("CSV", os.path.join("..", "results", "results.csv"))

ALL_CALLABLE = ["B0", "B1", "B3", "B4", "B5"]   # B2 is a documented stub (needs node wiring)


def _emit_csv_row(row: dict):
    """Append one row to CSV in the canonical column order, creating header if missing."""
    import csv
    new = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    os.makedirs(os.path.dirname(CSV) or ".", exist_ok=True)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hc.CSV_HEADER, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _base_row(candidate, schedule, bm, bn, bk, nsub, grid_blocks, notes):
    return {
        "candidate": candidate, "commit": hc.git_commit_short(), "date": hc.today(),
        "ranks": None, "route_dist": ROUTE, "M_label": M_LABEL, "M": M, "N": N, "K": K,
        "dtype": "fp8e4m3_in/bf16_w/bf16_out",
        "BM": bm, "BN": bn, "BK": bk, "NSUB": nsub, "schedule": schedule,
        "grid_blocks": grid_blocks,
        "VGPR": "[NEEDS-NODE]", "AGPR": "[NEEDS-NODE]", "SGPR": "[NEEDS-NODE]",
        "LDS": "[NEEDS-NODE]", "scratch": "[NEEDS-NODE]",
        "lat_us": None, "p50": None, "p95": None, "p99": None, "TFLOPs": None,
        "rms_rel": None, "zero_sentinel": None,
        "spd_vs_B1": None, "spd_vs_B2": None, "spd_vs_B3": None, "notes": notes,
    }


# ==============================================================================================
# Tensor setup.  Built ONCE per process and shared across cases so every B-case sees the SAME
# fp8 bytes, the SAME scales, the SAME B, the SAME output dtype.  Two regimes:
#   - comm cases (B1,B3,B4,B5): A on IRIS symmetric heap, src_rank holds real A, others sentinel.
#   - no-comm case (B0): everything local on a single rank (run with -np 1 OR ignored on rank!=0).
# ==============================================================================================
class Tensors:
    def __init__(self, iris, rank, world, need_comm):
        import torch
        self.torch = torch
        self.iris = iris
        self.rank = rank
        self.world = world
        self.need_comm = need_comm
        NG = K // hc.QGROUP

        # Deterministic true A (pre-quant) and weights B — same seeds as v3/v4 example.py.
        torch.manual_seed(1234)
        A_real = (torch.randn(M, K, dtype=torch.float32, device='cuda') / 8.0)
        qa, qs = hc.quantize_v1(A_real, M, K)
        self.A_deq = hc.dequant_v1(qa, qs, M, K)               # bf16 reference input

        torch.manual_seed(777)
        Bfull = (torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 8.0)

        if need_comm:
            self.A_fp8_bf16, self.A_fp8 = hc.make_fp8_iris_tensor(iris, M, K)
            self.A_sc = hc.make_iris_tensor(iris, [M, NG], "float32")
            self.B = hc.make_iris_tensor(iris, [N, K], "bfloat16")
            self.C = hc.make_iris_tensor(iris, [M, N], "bfloat16")
            if rank == SRC_RANK:
                self.A_fp8.copy_(qa)
                self.A_sc.copy_(qs)
            else:
                self.A_fp8.view(torch.uint8).zero_()           # ZERO SENTINEL on consumer rank
                self.A_sc.zero_()
            self.B.copy_(Bfull)
            self.C.zero_()
            iris.barrier()
            self.iris_ctx = iris.get_device_view()
        else:
            # B0: local-only, no heap, no sentinel (compute ceiling).
            self.A_fp8 = qa                                    # local fp8
            self.A_sc = qs
            self.B = Bfull
            self.C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            self.iris_ctx = None

        # Broadcast the true dequant-A to the consumer rank for the reference (host roundtrip,
        # does NOT feed any kernel — only builds the correct answer).
        if need_comm:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            host = self.A_deq.float().cpu().numpy() if rank == SRC_RANK else None
            host = comm.bcast(host, root=SRC_RANK)
            self.A_ref = torch.from_numpy(host).to('cuda').to(torch.bfloat16)
        else:
            self.A_ref = self.A_deq

    def fp8_as_bf16(self, fp8_tensor):
        """Reinterpret a fp8 [M,K] tensor as bf16 [M,K/2] over the SAME bytes (for gl<bf16>)."""
        t = self.torch
        return fp8_tensor.view(t.uint8).view(t.bfloat16).view(M, K // 2)

    def local_a_max(self):
        if not self.need_comm:
            return None
        return float(self.A_fp8.view(self.torch.uint8).max().item())


# ==============================================================================================
# Per-case run closures.  Each returns (run_fn, schedule_meta) or raises NotImplementedError /
# returns a STUB marker for B2.  run_fn does exactly one measured iteration.
# ==============================================================================================
def make_case(case, T):
    """Return dict(run_fn, meta) where meta carries schedule/tile/grid for the CSV row.
    For comm cases the consumer rank (rank != SRC_RANK) does the work; producer rank no-ops."""
    is_consumer = (not T.need_comm) or (T.rank != SRC_RANK)

    # ---- B3 / B4 / B5 : existing pybind kernels (tk_kernel.dispatch_micro) -------------------
    # B3 = V3 file built as tk_kernel, fused=0.   B5 = V4 file built as tk_kernel, fused=1.
    # B4 = V4 file built as tk_kernel, fused=0 (A-stationary reuse WITHOUT overlap).
    # The runner is told WHICH tk_kernel is loaded via env KERNEL_VARIANT (v3|v4); see report.
    if case in ("B3", "B4", "B5"):
        import tk_kernel
        variant = os.environ.get("KERNEL_VARIANT",
                                 "v3" if case == "B3" else "v4")
        fused = 1 if case == "B5" else 0
        bm, bn, bk = 64, 64, 64
        nsub = 1 if variant == "v3" else int(os.environ.get("NSUB", "8"))
        gx = (N + (bn * nsub) - 1) // (bn * nsub) if variant == "v4" else (N + bn - 1) // bn
        gy = (M + bm - 1) // bm
        meta = dict(schedule=f"{variant}/4P4C/{'overlap' if fused else 'two-phase'}",
                    bm=bm, bn=bn, bk=bk, nsub=nsub, grid_blocks=gx * gy)

        def run_fn():
            if is_consumer:
                tk_kernel.dispatch_micro(T.A_fp8_bf16, T.A_sc, T.B, T.C,
                                         T.iris_ctx, M, N, K, SRC_RANK, int(fused))
        return dict(run_fn=run_fn, meta=meta)

    # ---- B0 : local V2 HK GEMM only, NO comm (compute ceiling) -------------------------------
    # Calls the new harness pybind op `local_gemm` (harness_kernels.cpp), which wraps the SAME
    # bf16 8-wave MMA core as v2_hk_expert_gemm: dequant fp8->bf16 (preamble) then GEMM.  No IRIS.
    if case == "B0":
        try:
            import harness_kernel as hk
        except ImportError as e:
            raise NotImplementedError(
                "B0 needs the harness_kernel pybind module (build harness_kernels.cpp). "
                f"[NEEDS-NODE] import failed: {e}")
        bm, bn, bk = 256, 256, 64
        meta = dict(schedule="v2/8wave-pingpong/local", bm=bm, bn=bn, bk=bk, nsub=1,
                    grid_blocks=((M + 255) // 256) * ((N + 255) // 256))

        def run_fn():
            # local_gemm's gl<bf16> reinterprets the fp8 bytes; pass the bf16-view [M,K/2].
            hk.local_gemm(T.fp8_as_bf16(T.A_fp8), T.A_sc, T.B, T.C, M, N, K)
        return dict(run_fn=run_fn, meta=meta)

    # ---- B1 : IRISX dispatch-pack-quant ONCE -> local fp8+scale buffer -> local V2 GEMM ------
    # STRONG unfused IRISX baseline.  Two phases, each timed SEPARATELY then combined:
    #   phase T (transfer): IRIS dispatch+pack+quant of A from src_rank into a LOCAL fp8 buffer,
    #                       performed EXACTLY ONCE (not per N-tile — that's the whole point).
    #   phase C (compute) : the SAME local V2 GEMM as B0 over the now-local fp8 buffer.
    if case == "B1":
        try:
            import harness_kernel as hk
        except ImportError as e:
            raise NotImplementedError(
                "B1 needs the harness_kernel pybind module (build harness_kernels.cpp). "
                f"[NEEDS-NODE] import failed: {e}")
        import torch
        # Local destination fp8 buffer (NOT on the symmetric heap — it's the gathered copy).
        # Allocate as bf16[M,K/2] so its bf16-view matches the kernels' gl<bf16>; fp8-view shares
        # the same storage for scale-free byte copies.
        A_local_bf16 = torch.zeros(M, K // 2, dtype=torch.bfloat16, device='cuda')
        A_local_fp8 = A_local_bf16.view(torch.uint8).view(torch.float8_e4m3fn).view(M, K)
        A_local_sc = torch.zeros(M, K // hc.QGROUP, dtype=torch.float32, device='cuda')
        bm, bn, bk = 256, 256, 64
        meta = dict(schedule="B1/dispatch-once+v2gemm/two-phase-separate-timers",
                    bm=bm, bn=bn, bk=bk, nsub=1,
                    grid_blocks=((M + 255) // 256) * ((N + 255) // 256))

        def _phase_T():
            # phase T: gather A's fp8 bytes + scales ONCE from src_rank into the local buffer.
            hk.dispatch_pack_quant_once(T.A_fp8_bf16, T.A_sc, A_local_bf16, A_local_sc,
                                        T.iris_ctx, M, K, SRC_RANK)

        def _phase_C():
            # phase C: local GEMM over the gathered fp8 (identical numerics to B0's compute).
            hk.local_gemm(A_local_bf16, A_local_sc, T.B, T.C, M, N, K)

        def run_fn():
            if is_consumer:
                _phase_T()
                _phase_C()
        return dict(run_fn=run_fn, meta=meta,
                    phase_T=(lambda: _phase_T()) if is_consumer else (lambda: None),
                    phase_C=(lambda: _phase_C()) if is_consumer else (lambda: None))

    # ---- B2 : production MORI dispatch + AITER/CK fmoe (STUB) --------------------------------
    if case == "B2":
        raise NotImplementedError(
            "B2 is a STUB. Production MORI dispatch/pack/quant + AITER/CK fmoe are NOT callable "
            "from this harness without node wiring. See BENCHMARK_METHODOLOGY.md 'B2 wiring gap' "
            "and AGENT_REPORT.md for the exact symbols/imports the main agent must provide "
            "(mori.EpDispatch, aiter.fmoe / ck fmoe_bf16_blockscaleFp8). [NEEDS-NODE]")

    raise ValueError(f"unknown case {case}")


# ==============================================================================================
def run_case(case, iris, rank, world):
    need_comm = case in ("B1", "B3", "B4", "B5")
    T = Tensors(iris, rank, world, need_comm)
    handle = make_case(case, T)
    run_fn = handle["run_fn"]
    meta = handle["meta"]

    def sync_fn():
        import torch
        torch.cuda.synchronize()
        if need_comm:
            iris.barrier()

    # correctness (one run, then measure local A buffer for sentinel)
    run_fn()
    sync_fn()
    is_consumer = (not need_comm) or (rank != SRC_RANK)
    row = _base_row(case, meta["schedule"], meta["bm"], meta["bn"], meta["bk"],
                    meta["nsub"], meta["grid_blocks"], notes="")
    row["ranks"] = world

    if is_consumer:
        chk = hc.correctness(T.A_ref, T.B, T.C, T.local_a_max(), tol=TOL)
        row["rms_rel"] = round(chk["rms_rel"], 6)
        row["zero_sentinel"] = chk["zero_sentinel"]
        timing = hc.timed(run_fn, sync_fn, iters=ITERS, warmup=WARMUP)
        row["lat_us"] = round(timing["lat_us"], 3)
        row["p50"] = round(timing["p50"], 3)
        row["p95"] = round(timing["p95"], 3)
        row["p99"] = round(timing["p99"], 3)
        row["TFLOPs"] = round(hc.tflops(M, N, K, timing["lat_us"]), 2)

        # B1 split timers (transfer vs compute) -> noted in 'notes' for the strong baseline.
        if case == "B1" and "phase_T" in handle:
            tT = hc.timed(handle["phase_T"], sync_fn, iters=ITERS, warmup=WARMUP)
            tC = hc.timed(handle["phase_C"], sync_fn, iters=ITERS, warmup=WARMUP)
            row["notes"] = (f"transfer={tT['lat_us']:.1f}us compute={tC['lat_us']:.1f}us "
                            f"(dispatch-once); combined above")

        print(f"[{case}] M={M} N={N} K={K} M_label={M_LABEL} route={ROUTE}  "
              f"lat={row['lat_us']}us  TFLOPs={row['TFLOPs']}  rms_rel={row['rms_rel']}  "
              f"zero_sentinel={row['zero_sentinel']}  "
              f"{'OK' if chk['ok'] else 'FAIL'}", flush=True)
        _emit_csv_row(row)
    return row


def main():
    if "--list" in sys.argv:
        print("callable cases:", ", ".join(ALL_CALLABLE), "| B2 = STUB [NEEDS-NODE]")
        print("CSV header:", ",".join(hc.CSV_HEADER))
        return
    if "--dry-run" in sys.argv:
        # No GPU: print the row schema a real run WOULD emit for the requested case/shape.
        cases = ALL_CALLABLE if CASE == "all" else [CASE]
        for c in cases:
            r = _base_row(c, f"{c}/<schedule>", 64, 64, 64, 1, "<grid>",
                          notes="DRY-RUN: no device executed")
            print(c, "->", {k: r[k] for k in ("candidate", "M", "N", "K", "M_label",
                                              "route_dist", "dtype")})
        return

    # Real run: IRIS owns MPI lifecycle (mpi4py must not auto-init).
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    import torch
    import iris_py
    from mpi4py import MPI  # noqa: F401  (safe: no MPI_Init here)

    iris = iris_py.Iris(heap_size_mb=int(os.environ.get("HEAP_MB", "256")), verbose=False)
    rank = iris.rank()
    world = iris.world_size()
    torch.cuda.set_device(rank)

    cases = ALL_CALLABLE if CASE == "all" else [CASE]
    for c in cases:
        try:
            run_case(c, iris, rank, world)
        except NotImplementedError as e:
            if rank == 0 or rank != SRC_RANK:
                print(f"[{c}] SKIPPED: {e}", flush=True)

    import gc
    gc.collect(); torch.cuda.synchronize(); iris.barrier()
    del iris
    gc.collect(); torch.cuda.synchronize()
    from mpi4py import MPI as _MPI
    _MPI.Finalize()
    os._exit(0)


if __name__ == "__main__":
    main()
