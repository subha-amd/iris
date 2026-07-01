#!/usr/bin/env python3
# =============================================================================
# b3_ep8_unfused.py — the REAL 8-GPU unfused production EP MoE region
# =============================================================================
# Answers "where does the unfused variant stand on all 8 GPUs in a realistic
# production setting." This runs the ACTUAL production EP8 pipeline across 8 MI350
# [b3_a4w4 variant] QUANT=mxfp4 -> aiter a4w4 (OCP MXFP4) fmoe path.
# GPUs with the real MORI all-to-all (the same EpDispatchIntraNodeKernel /
# EpCombineIntraNodeKernel kernels the C4 trace shows), NOT a single-GPU proxy:
#
#   per rank:  tokens --MORI dispatch (all-to-all, XGMI)--> sort+quant+fmoe (aiter)
#                     --MORI combine (all-to-all, XGMI)--> combined output
#
# It is adapted from the AMD MORI op_test `multigpu_tests/test_dispatch_combine.py`
# (the known-good driver: aiter.init_dist_env + shmem_torch_process_group_init +
# EpDispatchCombineOp.dispatch/.combine + aiter.fused_moe), changed to:
#   * R1-0528 shapes (hidden=7168, inter=2048, 256 global experts, 32/rank, topk=8),
#   * time dispatch / fmoe / combine SEPARATELY and the FULL region together,
#   * reduce the region time with MAX over the 8 ranks (the production denominator —
#     the decode step waits for the slowest rank),
#   * sweep per-rank token count (decode-small -> prefill-large).
#
# Single launch spawns all 8 ranks (mp.Pool, like the MORI test) — no mpirun needed.
# Run:  python3 b3_ep8_unfused.py                  # sweep
#       TOKENS_PER_RANK=16 python3 b3_ep8_unfused.py
#       QUANT=per_128x128 python3 b3_ep8_unfused.py
# =============================================================================
import os
# This gfx950 (MI350) node's container defaults MORI_GPU_ARCHS to "gfx942;gfx950"
# (gfx942 first) -> MORI JITs for gfx942 and hipModuleLoad rejects the image on gfx950.
# Force gfx950 BEFORE importing mori.
os.environ["MORI_GPU_ARCHS"] = "gfx950"
import torch, argparse, multiprocessing as mp
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter import get_hip_quant
from aiter.test_common import run_perftest
import mori

WORLD = int(os.environ.get("WORLD", "8"))
E_GLOBAL = int(os.environ.get("E", "256"))          # global experts (32/rank @ world=8)
HID   = int(os.environ.get("HID", "7168"))
IDIM  = int(os.environ.get("IDIM", "2048"))
TOPK  = int(os.environ.get("TOPK", "8"))
QUANT = os.environ.get("QUANT", "per_1x128")          # C4/ATOM production = per_1x128
# DISPATCH=bf16 mirrors the C4 trace: MORI moves bf16 tokens (EpDispatchIntraNodeKernel_bf16),
#   then dynamic_quant runs IN-region inside fused_moe. DISPATCH=fp8 = the cheaper pre-quant variant.
DISPATCH = os.environ.get("DISPATCH", "bf16")
NWARM = int(os.environ.get("NWARM", "10"))
NITER = int(os.environ.get("NITER", "50"))

def qtype():
    return {"per_1x128":   aiter.QuantType.per_1x128,
            "per_128x128": aiter.QuantType.per_128x128,
            "per_Token":   aiter.QuantType.per_Token,
            "mxfp4":       aiter.QuantType.per_1x32,
            "No":          aiter.QuantType.No}[QUANT]

def weight_per_128x128_quant(weight, quant_dtype):
    E, d1, d2 = weight.shape
    wb = weight.view(E, d1//128, 128, d2//128, 128).permute(0,1,3,2,4).contiguous()
    wb = wb.view(E, -1, 128*128)
    wqt, wsc = aiter.pertoken_quant(wb, quant_dtype=quant_dtype)
    wqt = wqt.view(E, d1//128, d2//128, 128, 128).permute(0,1,3,2,4).contiguous().view(E, d1, d2)
    wsc = wsc.view(E, d1//128, d2//128)
    return wqt, wsc

def worker(rankID, tokens_per_rank):
    dev = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(dev)
    torch.manual_seed(1000 + rankID)
    aiter.init_dist_env(WORLD, rankID)
    qt = qtype()
    Eloc = E_GLOBAL // WORLD

    # this rank's input tokens (a DP replica) + global routing over E_GLOBAL experts
    tokens = torch.randn((tokens_per_rank, HID), dtype=dtypes.bf16, device=dev)
    score  = torch.randn((tokens_per_rank, E_GLOBAL), dtype=dtypes.bf16, device=dev)
    topk_weights, topk_ids = fused_topk(tokens, score, TOPK, True)

    # this rank's 32 local experts' weights (fp8 + 128x128 block-scale, preshuffled)
    w1 = torch.randn((Eloc, 2*IDIM, HID), dtype=dtypes.bf16, device=dev) / (HID**0.5)
    w2 = torch.randn((Eloc, HID, IDIM), dtype=dtypes.bf16, device=dev) / (IDIM**0.5)
    if QUANT == "mxfp4":
        # OCP MXFP4 W4 (aiter a4w4 path): fp4x2 weights + E8M0 per-32-block scale, a16w4-preshuffled.
        # DISPATCH=bf16 => bf16 activations dispatched; fused_moe quantizes A in-region (a1_scale=None).
        from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
        qf = aiter.get_torch_quant(aiter.QuantType.per_1x32)
        w1_qt, w1_scale = qf(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = qf(w2, quant_dtype=dtypes.fp4x2)
        w1_qt = shuffle_weight_a16w4(w1_qt, 16, True);  w1_scale = shuffle_scale_a16w4(w1_scale, Eloc, True)
        w2_qt = shuffle_weight_a16w4(w2_qt, 16, False); w2_scale = shuffle_scale_a16w4(w2_scale, Eloc, False)
    elif qt in (aiter.QuantType.per_128x128, aiter.QuantType.per_1x128):
        w1_qt, w1_scale = weight_per_128x128_quant(w1, dtypes.fp8)   # weights are 128x128-block
        w2_qt, w2_scale = weight_per_128x128_quant(w2, dtypes.fp8)
        w1_qt = shuffle_weight(w1_qt); w2_qt = shuffle_weight(w2_qt)
    else:
        qf = aiter.get_torch_quant(qt)
        w1_qt, w1_scale = qf(w1, quant_dtype=dtypes.fp8)
        w2_qt, w2_scale = qf(w2, quant_dtype=dtypes.fp8)
        w1_qt = shuffle_weight(w1_qt); w2_qt = shuffle_weight(w2_qt)

    # --- DISPATCH dtype selects the pipeline shape ---
    # bf16 (C4 trace): MORI moves bf16 tokens (no scales); dynamic_quant runs IN-region inside
    #   fused_moe (a1_scale=None). This is the faithful EpDispatch_bf16 -> sort -> quant -> fmoe path.
    # fp8: pre-quant before dispatch (cheaper movement), fused_moe uses the dispatched scale.
    if DISPATCH == "bf16":
        disp_inp, disp_scale_in, scdim, scsz, use_a1 = tokens, None, 0, 0, False
    else:
        aq = get_hip_quant(aiter.QuantType.per_1x128)
        tokens_qt, scale = aq(tokens, quant_dtype=dtypes.fp8)
        disp_inp, disp_scale_in, scdim, scsz, use_a1 = tokens_qt, scale, scale.shape[-1], scale.dtype.itemsize, True

    wg = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", wg)
    mori.shmem.shmem_torch_process_group_init("default")
    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=disp_inp.dtype, rank=rankID, world_size=WORLD, hidden_dim=HID,
        scale_dim=scdim, scale_type_size=scsz,
        max_token_type_size=dtypes.bf16.itemsize,
        max_num_inp_token_per_rank=max(8192, tokens_per_rank * 4),
        num_experts_per_rank=Eloc, num_experts_per_token=TOPK,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode)  # single-node, matches C4
    op = mori.ops.EpDispatchCombineOp(cfg)
    expert_mask = torch.zeros((E_GLOBAL,), dtype=dtypes.i32, device=dev)
    expert_mask[Eloc*rankID : Eloc*(rankID+1)] = 1

    def do_dispatch():
        return op.dispatch(disp_inp, topk_weights, disp_scale_in, topk_ids)
    def do_fmoe(do_, dw_, ds_, di_, drn_):
        return fused_moe(do_, w1_qt, w2_qt, dw_, di_, expert_mask,
                         num_local_tokens=drn_, w1_scale=w1_scale, w2_scale=w2_scale,
                         a1_scale=(ds_ if use_a1 else None), quant_type=qt, dtype=dtypes.bf16)
    def region():
        do_, dw_, ds_, di_, drn_ = op.dispatch(disp_inp, topk_weights, disp_scale_in, topk_ids)
        out_ = do_fmoe(do_, dw_, ds_, di_, drn_)
        return op.combine(out_, topk_weights, topk_ids)

    # ONE unified timing loop (no separate collective perftests — those desync the 8
    # ranks and deadlock MORI's all-to-all). Per-iter cuda events give the dispatch/
    # fmoe/combine breakdown; the per-iter device sync + the collective in every iter
    # keep all ranks in lockstep. dist.barrier() bookends a clean window.
    import statistics
    dist = torch.distributed
    def timed_iter():
        e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        e[0].record()
        do_, dw_, ds_, di_, drn_ = op.dispatch(disp_inp, topk_weights, disp_scale_in, topk_ids)
        e[1].record()
        out_ = do_fmoe(do_, dw_, ds_, di_, drn_)
        e[2].record()
        _ = op.combine(out_, topk_weights, topk_ids)
        e[3].record()
        torch.cuda.synchronize()
        return (e[0].elapsed_time(e[1]), e[1].elapsed_time(e[2]), e[2].elapsed_time(e[3]))

    do, dw, ds, di, drn = do_dispatch()
    recv = int(drn.item()) if hasattr(drn, "item") else int(drn)
    for _ in range(NWARM):
        timed_iter()
    dist.barrier()
    dl, fl, cl = [], [], []
    for _ in range(NITER):
        a, b, c = timed_iter(); dl.append(a); fl.append(b); cl.append(c)
    dist.barrier()
    us_disp = statistics.median(dl) * 1000.0   # ms -> us
    us_fmoe = statistics.median(fl) * 1000.0
    us_comb = statistics.median(cl) * 1000.0
    us_region = us_disp + us_fmoe + us_comb
    aiter.destroy_dist_env()
    return (rankID, recv, us_disp, us_fmoe, us_comb, us_region)

def run_point(tokens_per_rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "49401")
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=WORLD) as pool:
        res = [pool.apply_async(worker, args=(i, tokens_per_rank)) for i in range(WORLD)]
        rows = [r.get() for r in res]
    rows.sort()
    # MAX over ranks = the production region latency (step waits for the slowest rank)
    mx = lambda j: max(r[j] for r in rows)
    av = lambda j: sum(r[j] for r in rows)/len(rows)
    recv_mean = sum(r[1] for r in rows)/len(rows)
    print(f"\n### per-rank tokens={tokens_per_rank}  recv/rank≈{recv_mean:.0f}  "
          f"(world={WORLD}, E={E_GLOBAL}, topk={TOPK}, {QUANT}, dispatch={DISPATCH}) ###")
    print(f"{'stage':12} {'MAX us':>9} {'mean us':>9}   (MAX over ranks = the production denominator)")
    for nm, j in [("dispatch",2),("fmoe(local)",3),("combine",4),("REGION(d+f+c)",5)]:
        print(f"{nm:12} {mx(j):>9.2f} {av(j):>9.2f}")
    print(f"  dispatch+combine (the all-to-all) MAX = {mx(2)+mx(4):.2f} us")
    return dict(tokens=tokens_per_rank, recv=recv_mean,
                dispatch=mx(2), fmoe=mx(3), combine=mx(4), region=mx(5))

if __name__ == "__main__":
    mp.freeze_support()
    tpr = os.environ.get("TOKENS_PER_RANK")
    sweep = [int(tpr)] if tpr else [16, 64, 256, 1024]
    summ = []
    for t in sweep:
        try:
            summ.append(run_point(t))
        except Exception as e:
            print(f"  tokens={t} FAILED: {type(e).__name__}: {e}")
    print("\n===== 8-GPU UNFUSED EP MoE REGION (MAX over ranks) =====")
    print(f"{'tok/rank':>8} {'recv':>6} {'dispatch':>9} {'fmoe':>9} {'combine':>9} {'REGION':>9}")
    for s in summ:
        print(f"{s['tokens']:>8} {s['recv']:>6.0f} {s['dispatch']:>9.1f} {s['fmoe']:>9.1f} "
              f"{s['combine']:>9.1f} {s['region']:>9.1f}")
