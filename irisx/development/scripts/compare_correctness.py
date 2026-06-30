#!/usr/bin/env python3
"""Correctness comparator for candidate outputs. CPU/numpy only — NO GPU, safe anywhere.

Each candidate's example.py should dump (a) the kernel output and (b) the bf16 reference it was
checked against, as .npy, plus a small JSON with the zero-sentinel result. This script recomputes
the project's standard RMS-relative error and re-asserts the gate, independent of the kernel's own
self-report (trust-but-verify).

Usage:
  compare_correctness.py --out kernel_C.npy --ref ref_C.npy [--rms-tol 0.02] \
      [--sentinel sentinel.json]
"""
import argparse, json, sys
import numpy as np


def rms_rel(out: np.ndarray, ref: np.ndarray) -> float:
    out = out.astype(np.float64); ref = ref.astype(np.float64)
    num = np.sqrt(np.mean((out - ref) ** 2))
    den = np.sqrt(np.mean(ref ** 2)) + 1e-12
    return float(num / den)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--rms-tol", type=float, default=0.02,
                    help="RMS-rel gate; V3/V4 land ~0.00331, e4m3+bf16 tol is generous")
    ap.add_argument("--sentinel", default=None,
                    help="JSON with {local_A_zero:bool, C_zero:bool} — proves remote gather is real")
    args = ap.parse_args()

    out = np.load(args.out); ref = np.load(args.ref)
    if out.shape != ref.shape:
        print(f"FAIL shape mismatch {out.shape} vs {ref.shape}"); return 2

    r = rms_rel(out, ref)
    max_abs = float(np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64))))
    ok = r <= args.rms_tol
    print(f"rms_rel={r:.6f} (tol {args.rms_tol})  max_abs={max_abs:.4g}  -> {'PASS' if ok else 'FAIL'}")

    if args.sentinel:
        s = json.load(open(args.sentinel))
        sent_ok = bool(s.get("local_A_zero")) and not bool(s.get("C_zero"))
        print(f"zero_sentinel local_A_zero={s.get('local_A_zero')} C_zero={s.get('C_zero')} "
              f"-> {'REMOTE GATHER REAL' if sent_ok else 'SENTINEL FAIL'}")
        ok = ok and sent_ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
