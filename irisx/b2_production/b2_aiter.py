#!/usr/bin/env python3
# B2 — PRODUCTION local MoE baseline via aiter.fused_moe (sort + dynamic_quant + fmoe + combine).
# This is the UNFUSED PRODUCTION compute path on already-local tokens — the real bar our kernels are
# measured against (NOT B0, which is just the compute ceiling with no comm). It is the production
# "sequence of unfused kernels": grouped_topk's output -> moe_sorting -> dynamic_quant -> fmoe (the
# fused gate/up + SiLU + down) -> moe_sum combine, all inside aiter.fused_moe on ONE GPU.
#
# WHAT TO COMPARE (see ../BENCHMARKING_HANDOFF.md):
#   LEVEL 1 (now): GEMM efficiency. This prints production TFLOP/s (FLOP-normalized over the full
#     fc1+fc2 FFN). grouped_b0/grouped_b0.cu prints OUR TFLOP/s for one grouped GEMM. TFLOP/s is
#     FLOP-normalized so it is a fair "matmul efficiency" comparison regardless of how many GEMMs each
#     kernel fuses. Expect production (native fp8) > grouped_b0 (bf16 dequant) — that gap is the
#     native-fp8 track, separate from the tile/schedule fix.
#   To make it APPLES-TO-APPLES, feed grouped_b0 the SAME per-expert token counts this script prints
#     below ("M_e per expert"), at the SAME shapes (K, INTER).
#
# Shapes: DeepSeek-R1 decode, per-GPU local experts. model_dim K=7168, inter_dim=2048 (W13 g1u1 ->
# inter*2=4096), E=32 local experts, top-k=8, fp8 e4m3 per-1x128 block-scale (QuantType.per_1x128).
# Weight block-scale per-128x128 (production layout, copied from aiter op_tests/test_moe_2stage.py).
#
# VRAM: weights are fp8 [E,2*INTER,K] + [E,K,INTER] ~= E*1.5*INTER*K bytes (E=32 -> ~1.4 GB). If the
# live vLLM server holds VRAM, reduce E (env) or pause the server — see BENCHMARKING_HANDOFF.md §6.
#
# Run (single GPU): python3 b2_aiter.py    (env: TOKEN, E, K, INTER, TOPK)
import os, torch
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.fused_moe import QuantType
from aiter.test_common import run_perftest

TOKEN = int(os.environ.get("TOKEN", "1024"))     # local tokens (pre-routing); routed rows = TOKEN*TOPK
E     = int(os.environ.get("E", "32"))           # local experts
K     = int(os.environ.get("K", "7168"))         # model_dim
INTER = int(os.environ.get("INTER", "2048"))     # inter_dim (W13 produces inter*2=4096)
TOPK  = int(os.environ.get("TOPK", "8"))
dtype = torch.bfloat16
WQDType = dtypes.fp8
AQDType = dtypes.fp8

torch.manual_seed(0)
torch.cuda.set_device(0)

inp = torch.randn((TOKEN, K), dtype=dtype, device="cuda")
w1  = torch.randn((E, INTER * 2, K), dtype=dtype, device="cuda") / (K ** 0.5)   # g1u1: [E, 2*inter, K]
w2  = torch.randn((E, K, INTER), dtype=dtype, device="cuda") / (INTER ** 0.5)   # [E, K, inter]
score = torch.randn((TOKEN, E), dtype=dtype, device="cuda")
topk_weights, topk_ids = fused_topk(inp, score, TOPK, True)

# --- the per-expert routed-token distribution (M_e) — print it so grouped_b0 can be fed the SAME ---
# distribution for a fair head-to-head (this is the routing both sides must share).
m_e = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=E).cpu().tolist()

# weight per-128x128 block quant (VERBATIM convention from op_tests/test_moe_2stage.py) ------------
def weight_per_128x128_quant(weight, quant_dtype):
    E_, dim1, dim2 = weight.shape
    wb = weight.view(E_, dim1 // 128, 128, dim2 // 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    wb = wb.view(E_, -1, 128 * 128)
    wqt, wsc = aiter.pertoken_quant(wb, quant_dtype=quant_dtype)
    wqt = wqt.view(E_, dim1 // 128, dim2 // 128, 128, 128).permute(0, 1, 3, 2, 4).contiguous()
    wqt = wqt.view(E_, dim1, dim2)
    wsc = wsc.view(E_, dim1 // 128, dim2 // 128)
    return wqt, wsc

w1_qt, w1_scale = weight_per_128x128_quant(w1, WQDType)
w2_qt, w2_scale = weight_per_128x128_quant(w2, WQDType)

def call_b2():
    return fused_moe(
        inp, w1_qt, w2_qt, topk_weights, topk_ids,
        quant_type=QuantType.per_1x128,
        w1_scale=w1_scale, w2_scale=w2_scale,
    )

# correctness smoke: output finite + nonzero
out = call_b2()
finite = bool(torch.isfinite(out).all()); nonzero = bool(out.abs().sum() > 0)
out2, us = run_perftest(call_b2)   # aiter's own perf harness (warmup+median us)
routed = TOKEN * TOPK
# FFN FLOPs: W13 (K->2*inter) + W2 (inter->K) per routed row, x2 for MAC
flops = routed * (K * INTER * 2 + INTER * K) * 2
tflops = flops / (us * 1e-6) / 1e12
print(f"[B2-aiter] TOKEN={TOKEN} routed={routed} E={E} K={K} INTER={INTER} topk={TOPK} "
      f"fp8 per_1x128 | finite={finite} nonzero={nonzero} | {us:.2f} us  {tflops:.2f} TFLOP/s", flush=True)
print(f"[B2-aiter] full-FFN production GEMM efficiency = {tflops:.1f} TFLOP/s  "
      f"(compare to grouped_b0's TFLOP/s; gap above bf16 grouped_b0 = the native-fp8 headroom)", flush=True)
# The routing both sides must share for a fair comparison:
print(f"[B2-aiter] M_e per expert (feed these to grouped_b0 for a MATCHED comparison):", flush=True)
print(f"           min={min(m_e)} max={max(m_e)} mean={sum(m_e)/len(m_e):.1f} sum={sum(m_e)}", flush=True)
print(f"           M_e={m_e}", flush=True)
