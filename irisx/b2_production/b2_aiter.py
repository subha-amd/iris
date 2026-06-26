#!/usr/bin/env python3
# B2 — PRODUCTION local MoE baseline via aiter.fused_moe (sort + dynamic_quant + fmoe + combine).
# This is the UNFUSED PRODUCTION compute path on already-local tokens, the real bar B1-dispatch's
# pipeline total must be compared against (NOT B0, which is just the compute ceiling with no comm).
#
# Shapes: DeepSeek-R1 decode, per-GPU local experts. model_dim K=7168, inter_dim=2048 (W13 g1u1 ->
# inter*2=4096), E=32 local experts, top-k=8, fp8 e4m3 per-1x128 block-scale (QuantType.per_1x128).
# Weight block-scale per-128x128 (production layout, copied from aiter op_tests/test_moe_2stage.py).
#
# Run (single GPU): python3 b2_aiter.py   (env: TOKEN, E, K, INTER, TOPK)
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
