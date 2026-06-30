#!/usr/bin/env python3
# =============================================================================
# b2_unfused_region.py — the FAIR unfused-production MoE baseline (single GPU)
# =============================================================================
# Purpose (answers "run the same aiter kernels in sequence and measure my fused
# kernel against the unfused baseline"):
#
#   This script reconstructs the EXACT C4 (TP4/DP2+EP) DeepSeek-R1 decode MoE
#   expert region as a sequence of the real, stock aiter kernels, times it as ONE
#   region (and per-kernel), at a configurable operating point (decode M_e small
#   -> prefill M_e large), and prints a REGION-ACCOUNTING table that adds the two
#   cross-GPU kernels (EpDispatch / EpCombine) whose durations come from the C4
#   trace. That total is the denominator your fused b1_dispatch region must beat.
#
# Why this is the honest baseline (vs the ledger's 1255us-vs-714us):
#   * aiter.fused_moe is a PYTHON ORCHESTRATOR, not one fused GPU kernel. On the
#     GPU it launches exactly: moe_sorting (P0,P23) -> dynamic_per_group_scaled_quant
#     -> fmoe_fp8_blockscale_g1u1 -> moe_sum. Those are the SAME kernel *names*
#     the C4 trace shows between EpDispatch and EpCombine. So timing fused_moe IS
#     "run the unfused kernels in sequence" for the LOCAL part of the region.
#   * The only C4 kernels it does NOT contain are the two CROSS-GPU MORI ops,
#     EpDispatchIntraNodeKernel_bf16 (token scatter-out) and
#     EpCombineIntraNodeKernel_bf16_nop2p (result gather-back). Those cannot be
#     reproduced single-GPU, so we add their measured trace durations as explicit,
#     overridable terms (defaults below; re-pin them at the matched M_e — see §C6).
#
# THE REGION BOUNDARY (must be identical on both sides for the ratio to mean
# anything):
#   ENTER: routed bf16 tokens, present locally, in arbitrary (unsorted) order,
#          plus the top-k routing decision (ids + weights).
#   EXIT:  bf16 expert outputs, combined over the top-k, in original token order.
#
#   Production covers it as:  EpDispatch | sort | quant | fmoe | EpCombine(+moe_sum)
#   Your fused path must cover the SAME enter->exit:
#                             ep8_gather(=dispatch+sort+quant-amortized) | grouped_b0 | combine
#   ==> b1_dispatch CURRENTLY HAS NO COMBINE. Until it does, it is measuring a
#       STRICT SUBSET of this region and its number is not yet comparable. (§C6)
#
# Run (single GPU; aiter installed; works alongside the vLLM server for small E):
#   python3 b2_unfused_region.py                 # sweep TOKEN to show op-point dep.
#   TOKEN=16 python3 b2_unfused_region.py        # one decode-like point
#   PROFILE=1 TOKEN=64 python3 b2_unfused_region.py   # + per-kernel attribution
#   EP_DISPATCH_US=30.7 EP_COMBINE_US=23.2 python3 b2_unfused_region.py
#
# This file is the sibling of b2_aiter.py (which reports FLOP-normalized TFLOP/s
# for the GEMM-efficiency Level-1 question). THIS file reports LATENCY for the
# region-comparison Level-2 question. See ../BENCHMARKING_METHODOLOGY.md.
# =============================================================================
import os, json, torch
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_topk, QuantType
from aiter.test_common import run_perftest

# ---- problem constants (DeepSeek-R1-0528 decode, per-GPU local experts) ------
E     = int(os.environ.get("E", "32"))       # local experts / GPU (EP8 -> 32)
K     = int(os.environ.get("K", "7168"))     # model_dim = fc1 contraction K
INTER = int(os.environ.get("INTER", "2048")) # inter_dim (W13 emits 2*INTER=4096)
TOPK  = int(os.environ.get("TOPK", "8"))
dtype = torch.bfloat16
WQDType = dtypes.fp8

# ---- cross-GPU terms (the two MORI kernels we can't run single-GPU) -----------
# Defaults are the C4 TP4/DP2 trace AVERAGES (c4-highthroughput-query2):
#   EpDispatchIntraNodeKernel_bf16        avg 30.735 us  (150 calls)
#   EpCombineIntraNodeKernel_bf16_nop2p   avg 23.172 us  (150 calls)
# THESE WERE MEASURED AT THE TRACE'S DECODE BATCH. If you sweep TOKEN here to a
# different operating point, these terms are NOT automatically valid at that M_e
# (all-to-all is latency-bound at tiny M, BW-bound at large M). Re-pin them at the
# matched M_e with a MORI/rccl-tests all-to-all microbench (§C6) before quoting a
# full-region speedup. They are printed separately so the local ratio stays clean.
EP_DISPATCH_US = float(os.environ.get("EP_DISPATCH_US", "30.735"))
EP_COMBINE_US  = float(os.environ.get("EP_COMBINE_US",  "23.172"))

PROFILE = os.environ.get("PROFILE", "0") == "1"

torch.manual_seed(0)
torch.cuda.set_device(0)

# weight per-128x128 block quant — VERBATIM from b2_aiter.py / aiter op_tests -----
def weight_per_128x128_quant(weight, quant_dtype):
    E_, d1, d2 = weight.shape
    wb = weight.view(E_, d1 // 128, 128, d2 // 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    wb = wb.view(E_, -1, 128 * 128)
    wqt, wsc = aiter.pertoken_quant(wb, quant_dtype=quant_dtype)
    wqt = wqt.view(E_, d1 // 128, d2 // 128, 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    wqt = wqt.view(E_, d1, d2)
    wsc = wsc.view(E_, d1 // 128, d2 // 128)
    return wqt, wsc

def build(TOKEN):
    inp = torch.randn((TOKEN, K), dtype=dtype, device="cuda")
    w1  = torch.randn((E, INTER * 2, K), dtype=dtype, device="cuda") / (K ** 0.5)
    w2  = torch.randn((E, K, INTER), dtype=dtype, device="cuda") / (INTER ** 0.5)
    score = torch.randn((TOKEN, E), dtype=dtype, device="cuda")
    tw, tid = fused_topk(inp, score, TOPK, True)
    w1q, w1s = weight_per_128x128_quant(w1, WQDType)
    w2q, w2s = weight_per_128x128_quant(w2, WQDType)
    m_e = torch.bincount(tid.flatten().to(torch.int64), minlength=E).cpu().tolist()
    def call():
        # the UNFUSED sequence: GPU sees moe_sorting -> dynamic_quant -> fmoe -> moe_sum
        return fused_moe(inp, w1q, w2q, tw, tid, quant_type=QuantType.per_1x128,
                         w1_scale=w1s, w2_scale=w2s)
    return call, m_e

def per_kernel_split(call):
    """Optional: torch.profiler gives the same per-kernel attribution the trace
    has (moe_sorting / dynamic_quant / fmoe / moe_sum), version-independently."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(5): call()           # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20): call()
    torch.cuda.synchronize()
    rows = []
    for k in prof.key_averages():
        nm = k.key
        if nm.startswith(("aten::", "cuda", "hip", "Context", "Runtime")):
            continue                       # drop host/dispatch rows; keep GPU kernels
        us = (k.self_device_time_total if hasattr(k, "self_device_time_total")
              else k.self_cuda_time_total) / 20.0
        if us > 0.05:
            rows.append((nm[:64], us))
    return sorted(rows, key=lambda r: -r[1])[:12]

def main():
    tokens = os.environ.get("TOKEN")
    sweep = [int(tokens)] if tokens else [16, 64, 256, 1024]
    print(f"[b2-unfused-region] E={E} K={K} INTER={INTER} TOPK={TOPK} fp8 per_1x128 | "
          f"cross-GPU terms: EpDispatch={EP_DISPATCH_US}us EpCombine={EP_COMBINE_US}us\n")
    print(f"{'TOKEN':>6} {'routed':>7} {'M_e mean':>9} {'M_e max':>8} "
          f"{'T_local us':>11} {'+EpDisp':>8} {'+EpComb':>8} {'T_region us':>12}")
    results = []
    for T in sweep:
        call, m_e = build(T)
        out = call()
        assert bool(torch.isfinite(out).all()), "non-finite output"
        _, us = run_perftest(call)               # median local-region latency
        region = EP_DISPATCH_US + us + EP_COMBINE_US
        routed = T * TOPK
        print(f"{T:>6} {routed:>7} {sum(m_e)/len(m_e):>9.1f} {max(m_e):>8} "
              f"{us:>11.2f} {EP_DISPATCH_US:>8.1f} {EP_COMBINE_US:>8.1f} {region:>12.2f}")
        results.append(dict(token=T, routed=routed, m_e_mean=sum(m_e)/len(m_e),
                            m_e_max=max(m_e), t_local_us=us, t_region_us=region, m_e=m_e))
        if PROFILE:
            print("        per-kernel (us/call):")
            for nm, kus in per_kernel_split(call):
                print(f"          {kus:8.2f}  {nm}")
    # dump M_e of the largest point so the fused side can match it exactly
    with open("b2_unfused_region_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n--- FAIRNESS CONTRACT (read before quoting any ratio) -----------------")
    print("  * T_local  = stock aiter sort+quant+fmoe+moe_sum (the in-trace local chain).")
    print("  * T_region = T_local + EpDispatch + EpCombine (the FULL C4 MoE region).")
    print("  * Your fused b1_dispatch number is comparable to T_region ONLY when it")
    print("    covers ENTER->EXIT: gather(=dispatch+sort+quant) + grouped_b0 + COMBINE.")
    print("    b1_dispatch has no combine yet -> compare its (gather+GEMM) to")
    print("    (EpDispatch + sort + quant + fmoe), NOT to the whole region, until then.")
    print("  * Match M_e: feed results['m_e'] of the SAME TOKEN into grouped_b0 / b1_dispatch.")
    print("  * Decode reality: real C4 decode M_e is SINGLE DIGITS, not 256. The TOKEN=16")
    print("    row is closer to decode than TOKEN=1024 (which is prefill-like). Quote the")
    print("    operating point with every number.  -> wrote b2_unfused_region_results.json")

if __name__ == "__main__":
    main()
