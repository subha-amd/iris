#!/usr/bin/env python3
# =============================================================================
# moe_cost_model.py — analytic cost model for the DeepSeek-R1 EP8 MoE region
# =============================================================================
# Simran asked for a cost model. This is a first-principles roofline of the C4
# decode MoE expert region. It does TWO things:
#
#   (1) PREDICT, as a function of the per-expert row counts M_e and the hardware
#       constants, the time of each stage (dispatch / sort / quant / fc1 / fc2 /
#       combine), and flag whether each stage is COMPUTE-bound or BANDWIDTH/
#       LATENCY-bound. This tells you, before you write a line of kernel, where
#       the time is and therefore what fusion can and cannot buy.
#
#   (2) Compute the FUSION CEILING: the best case if the gather+pack+quant
#       envelope is perfectly hidden under the GEMM. You cannot beat this; if your
#       measured fused number is far below the ceiling, the win is real headroom,
#       if it's near the ceiling you're done.
#
# It is deliberately simple (bytes/BW + flops/peak, max() for roofline, additive
# fixed launch cost). The POINT is the structure and the sensitivity to M_e, not
# 5-digit accuracy. PIN THE CONSTANTS with a microbench (see PIN markers) before
# trusting absolute numbers; the SHAPE of the answer (who dominates at which M_e)
# is robust to constant error.
#
# Run:  python3 moe_cost_model.py                       # default decode + prefill points
#       ME=4   python3 moe_cost_model.py                # uniform M_e=4  (deep decode)
#       MEFILE=b2_unfused_region_results.json python3 moe_cost_model.py  # real M_e
# =============================================================================
import os, json, sys

# ---- HARDWARE CONSTANTS (MI355X / gfx950 / CDNA4) ----------------------------
# >>> PIN THESE. Defaults are spec-sheet ballparks, not measured. <<<
HBM_BW_TBs      = float(os.environ.get("HBM_BW_TBS", "8.0"))     # TB/s   PIN: STREAM/copy bench
XGMI_BW_GBs     = float(os.environ.get("XGMI_BW_GBS", "400.0"))  # GB/s   PIN: rccl-tests all2all, per-rank effective
FP8_PFLOPs      = float(os.environ.get("FP8_PFLOPS", "5.0"))     # PFLOP/s native fp8 dense  PIN: GEMM roofline
BF16_PFLOPs     = float(os.environ.get("BF16_PFLOPS", "2.5"))    # PFLOP/s bf16 (dequant path)  PIN
LAUNCH_US       = float(os.environ.get("LAUNCH_US", "3.0"))      # per-kernel fixed cost (decode is launch-heavy)
EFF             = float(os.environ.get("EFF", "0.55"))           # achieved/peak fraction (real kernels)

# ---- PROBLEM CONSTANTS (R1-0528, EP8) ----------------------------------------
K       = 7168          # fc1 contraction / model dim
N_FC1   = 4096          # fused gate|up output (2*INTER)
INTER   = 2048
N_FC2   = 7168          # down output = H
TOPK    = 8
EP      = 8
B_BF16  = 2
B_FP8   = 1
GROUPS  = K // 128      # 56 fp8 block-scale groups along K

GByte = 1e9
def us(seconds): return seconds * 1e6

def gemm_time(rows, N, Kc, peak_pflops, in_bytes, n_active):
    """roofline: max(compute, memory) for one grouped GEMM over `rows` rows.
    THE DECODE MEMORY WALL: the expert weights do NOT fit in L2 (~1.4 GB fp8 for
    32 experts), so they are streamed from HBM every decode step REGARDLESS of how
    few tokens route to each expert. This weight term dominates at small M_e and is
    why decode MoE is weight-bandwidth-bound. Measured on MI350: fmoe @ M_e=16 = 262us,
    of which ~176us is just reading 1.4GB of weights at ~8TB/s. (Verified vs the
    per-kernel profile in b2_unfused_region.py; the old 'weights resident' assumption
    under-predicted the GEMM by ~12x at decode.)"""
    flops = 2.0 * rows * N * Kc
    t_compute = flops / (peak_pflops * 1e15 * EFF)
    a = rows * Kc * in_bytes               # activations read
    c = rows * N * B_BF16                  # output written
    w = n_active * N * Kc * B_FP8          # ALL active experts' weights streamed from HBM
    t_mem = (a + c + w) / (HBM_BW_TBs * 1e12)
    return max(t_compute, t_mem), ("compute" if t_compute >= t_mem else "memory(weights)")

def stage_table(m_e, label):
    Mtot = sum(m_e)                 # routed rows landing on THIS rank
    nexp = sum(1 for x in m_e if x > 0)
    # --- dispatch: bf16 tokens scattered out over XGMI (this rank's share) -----
    # at decode this is latency-bound; model as max(BW term, a floor latency)
    disp_bytes = Mtot * K * B_BF16
    t_disp = max(disp_bytes / (XGMI_BW_GBs * GByte), 8e-6) + LAUNCH_US * 1e-6
    # --- sort: two full read+write passes over the routed bf16 activations ------
    sort_bytes = 2 * (Mtot * K * B_BF16) * 2   # 2 passes, each R+W
    t_sort = sort_bytes / (HBM_BW_TBs * 1e12) + 2 * LAUNCH_US * 1e-6
    # --- quant: read bf16, write fp8 + fp32 scales ------------------------------
    q_bytes = Mtot * K * B_BF16 + Mtot * K * B_FP8 + Mtot * GROUPS * 4
    t_quant = q_bytes / (HBM_BW_TBs * 1e12) + LAUNCH_US * 1e-6
    # --- fc1 (g1u1, native fp8) and fc2 (down, native fp8) ---------------------
    # n_active experts' weights stream from HBM every step -> the decode weight wall
    t_fc1, b_fc1 = gemm_time(Mtot, N_FC1, K,    FP8_PFLOPs, B_FP8, nexp)
    t_fc2, b_fc2 = gemm_time(Mtot, N_FC2, INTER, FP8_PFLOPs, B_FP8, nexp)
    t_fc1 += LAUNCH_US * 1e-6; t_fc2 += LAUNCH_US * 1e-6
    # --- combine: bf16 results gathered back over XGMI + weighted accumulate -----
    comb_bytes = Mtot * N_FC2 * B_BF16
    t_comb = max(comb_bytes / (XGMI_BW_GBs * GByte), 6e-6) + LAUNCH_US * 1e-6

    stages = [("EpDispatch (XGMI)", t_disp, "xgmi/lat"),
              ("moe_sorting",       t_sort, "hbm"),
              ("dynamic_quant",     t_quant, "hbm"),
              ("fmoe fc1 g1u1",     t_fc1, b_fc1),
              ("fmoe fc2 down",     t_fc2, b_fc2),
              ("EpCombine (XGMI)",  t_comb, "xgmi/lat")]
    total = sum(s[1] for s in stages)
    print(f"\n=== {label}: M_e mean={Mtot/len(m_e):.1f} max={max(m_e)} "
          f"active_experts={nexp}/{len(m_e)} routed_rows={Mtot} ===")
    print(f"  {'stage':20} {'time us':>9} {'% region':>9}  bound")
    for nm, t, b in stages:
        print(f"  {nm:20} {us(t):>9.2f} {100*t/total:>8.1f}%  {b}")
    print(f"  {'TOTAL REGION':20} {us(total):>9.2f} {100:>8.1f}%")

    # --- fusion ceiling: hide dispatch+sort+quant under the GEMM ---------------
    gemm = t_fc1 + t_fc2
    envelope = t_disp + t_sort + t_quant            # the gather/pack/quant surface
    floor = max(gemm + t_comb, envelope + t_comb)   # perfect overlap of envelope w/ GEMM
    print(f"  ---- fusion analysis ----")
    print(f"  gather/pack/quant envelope : {us(envelope):8.2f} us  ({100*envelope/total:.0f}% of region)")
    print(f"  expert GEMM (fc1+fc2)      : {us(gemm):8.2f} us  ({100*gemm/total:.0f}% of region)")
    print(f"  perfect-overlap floor      : {us(floor):8.2f} us  => MAX speedup vs unfused = {total/floor:.2f}x")
    w_floor = nexp * (N_FC1 * K + N_FC2 * INTER) * B_FP8 / (HBM_BW_TBs * 1e12)
    print(f"  expert-weight HBM floor    : {us(w_floor):8.2f} us  (stream {nexp} experts' fp8 weights; "
          f"independent of M_e)")
    weight_bound = (b_fc1.startswith("memory") or b_fc2.startswith("memory"))
    if weight_bound:
        print(f"  NOTE: GEMM is WEIGHT-MEMORY-bound -> the {us(w_floor):.0f}us expert-weight HBM stream")
        print(f"        (all {nexp} experts, every step) IS the floor, not token compute and not the")
        print(f"        gather. This is the decode memory wall. A better GATHER kernel only helps by")
        print(f"        HIDING the {us(envelope):.0f}us gather+sort+quant envelope UNDER this stream")
        print(f"        (ceiling {total/floor:.2f}x). Bigger decode levers: shard weights across more EP")
        print(f"        ranks, fp4 weights, or batch more tokens so the GEMM turns compute-bound.")
    elif envelope > gemm:
        print(f"  NOTE: envelope > GEMM -> COMMUNICATION/OVERHEAD-bound. The win is collapsing the")
        print(f"        gather+sort+quant+launch overhead (this is where a fused gather pays most).")
    else:
        print(f"  NOTE: GEMM > envelope and COMPUTE-bound (prefill-like). Overlap hides the envelope")
        print(f"        under the GEMM; ceiling is the GEMM itself. Native-fp8 GEMM matters here.")

def main():
    mefile = os.environ.get("MEFILE")
    me_env = os.environ.get("ME")
    if mefile and os.path.exists(mefile):
        data = json.load(open(mefile))
        for row in data:
            stage_table(row["m_e"], f"TOKEN={row['token']} (from {mefile})")
        return
    if me_env:
        m = int(me_env); stage_table([m]*32, f"uniform M_e={m}"); return
    # default: a decode point and a prefill point to show the regime flip
    stage_table([4]*32 + [0]*0,  "DECODE-like  (uniform M_e=4, ~16 tokens topk8)")
    stage_table([40]*32,         "DECODE-heavy (uniform M_e=40)")
    stage_table([256]*32,        "PREFILL-like (uniform M_e=256, the ledger's 8192-row point)")
    print("\nReminder: constants are spec ballparks. PIN HBM_BW/XGMI_BW/FP8_PFLOPS with a")
    print("microbench, then re-run. The REGIME (who dominates at which M_e) is the deliverable.")

if __name__ == "__main__":
    main()
