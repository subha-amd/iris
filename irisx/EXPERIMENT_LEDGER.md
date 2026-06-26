# EXPERIMENT LEDGER — IRISX/HK MoE on 8×MI355X

Durable, committed record of every on-device measurement. The main agent is the ONLY writer here,
and the only actor that runs GPU jobs (serialized under `flock /tmp/mi355x_project_gpu.lock`).
Raw rows also go to `results/results.csv` (gitignored; regenerable — `git add -f` if you want it
tracked). **Record negative results too — never keep only the best number.**

## Columns (also the CSV header)
`candidate, commit, date, ranks, route_dist, M_label(per_expert|aggregate), M, N, K, dtype,
BM, BN, BK, NSUB, schedule, grid_blocks, VGPR, AGPR, SGPR, LDS, scratch, lat_us, p50, p95, p99,
TFLOPs, rms_rel, zero_sentinel, spd_vs_B1, spd_vs_B2, spd_vs_B3, notes`

## Baseline definitions (canonical names — enforced 2026-06-26 after Gate 1)
- **B0** local V2 HK GEMM, no comm — compute ceiling (~164us / 183 TFLOP/s @ M1024).
- **B1-copy** one remote copy of A (+scales) into a local buffer, then exact local B0 GEMM — the
  strong copy-once baseline. **Headline speedups must cite B1-copy (or B1-dispatch), never B3.**
- **B1-dispatch** route-aware dispatch/pack/quant + local GROUPED GEMM (the real EP baseline once
  grouped lands). No candidate is a production winner until it beats B1-dispatch.
- **B2** MORI/ATOM dispatch + AITER/CK fmoe — production baseline (must be compared, not just beaten).
- **B3** repeated-direct-pull serial baseline (refetches A per N-tile, ~32x). NOT "production/strong".
- **B4** A-stationary remote pull, SERIAL (no overlap) — isolates reuse from overlap.
- **B5** A-stationary remote pull + overlap = the V4 "direct-pull A-stationary mechanism proof".
- **P1** copy-once tile-inbox overlap (producer copies A 1x + ready flag; consumer = B0 GEMM).
- **P2** expert-granular double-buffer overlap (gather expert e+1 while GEMM computes e).
Decision: P1/P2 < B1 -> continue tile-overlap research; ~= B1 -> abstraction useful, perf weak;
> B1 -> bulk dispatch+local GEMM is the preferred dataflow (optimize dispatch, not a fused GEMM).

## Decision gates (no claim crosses a gate unmet)
- G1 baseline validity: V4 compared to B1 + B2 before any headline.
- G2 production shape: W13/W2 shapes+layouts source-verified (Agent 00) before "R1 expert kernel".
- G3 decode realism: tested on captured EP8 per-expert distributions before "decode win".
- G4 real multi-rank: no production claim from fixed-source np=2.
- G5 end-to-end: no model-level claim until ATOM-integrated under C4/C6.

---

## Phase A — rebaseline canonical V4
_status: DONE 2026-06-25 (main agent, node cv350 / r1_c4, np=2, --mca pml ob1 --mca btl self,vader)_

Build clean (gfx950). Baseline here = V4's in-file `micro_tk_baseline` (= B3-family direct-pull,
no overlap). All RMS_rel=0.00331, local_A_zero=True, C_zero=False (remote gather real).

| candidate | M | N | K | baseline_us | fused_us | spd_vs_B3 | notes |
|---|---|---|---|---|---|---|---|
| V4 astationary | 1024 | 2048 | 7168 | 1240.6 | 679.9 | 1.825x | run1/3 |
| V4 astationary | 1024 | 2048 | 7168 | 1222.3 | 675.6 | 1.809x | run2/3 |
| V4 astationary | 1024 | 2048 | 7168 | 1243.2 | 682.6 | 1.821x | run3/3 (stable, ~44.2 TFLOP/s) |
| V4 astationary | 512  | 2048 | 7168 | 642.5  | 563.0 | 1.141x | crossover into win |
| V4 astationary | 256  | 2048 | 7168 | 490.4  | 546.6 | 0.897x | LOSS (grid-starved) |
| V4 astationary | 128  | 2048 | 7168 | 464.7  | 549.2 | 0.846x | LOSS (grid-starved) |

CONFIRMED: 1.80-1.83x stable at M=1024; wins M>=512; loses M<=256. Matches V4_ASTATIONARY_RESULTS.
CAVEAT (Gate 1 unmet): this baseline refetches A per N-tile (weak). Honest speedup needs B1
(gather-once + local GEMM) from Agent 01's harness — see Phase B below.

## Phase B — strong baselines B0/B1 (GATE 1) — THE HONEST COMPARISON
_status: DONE 2026-06-26 (main agent, cv350/r1_c4, np=2; B0 np=2 consumer-rank). Stable 2 runs._

Harness had 3 bugs found+fixed on the way (host gl-indexing, pybind module name, B0 GEMM OOB
tiling -> replaced with the proven V2 kernel) PLUS a barrier-asymmetry DEADLOCK in run_harness
(only the consumer rank ran the timing loop, so it waited forever on iris.barrier() the producer
never reached; fixed so all ranks run the timing loops symmetrically).

| case | what | M | N | K | lat_us | TFLOPs | rms_rel | sentinel |
|---|---|---|---|---|---|---|---|---|
| B0 | local GEMM, NO comm (compute ceiling) | 1024 | 2048 | 7168 | 164.2 | 183.1 | 0.00331 | n/a |
| B1-copy | one remote copy + exact local B0 GEMM | 1024 | 2048 | 7168 | 291.4 | 103.2 (end-to-end) | 0.00331 | True |
| B5 (V4) | direct-pull A-stationary mechanism proof (fused) | 1024 | 2048 | 7168 | 678.0 | 44.2 (end-to-end) | 0.00331 | True |

CORRECTIONS (per redirection 2026-06-26):
- TFLOP/s columns for B1-copy (103) and B5 (44) are END-TO-END EFFECTIVE throughput, NOT GEMM
  efficiency. B1-copy's COMPUTE PHASE (~162us) is ~185 TFLOP/s = matches B0's 183. So B1's GEMM is
  B0-class; the 103 number is dragged down by the serial 143us transfer. Do NOT say "B1 GEMM = 103".
- 291 (T_total) vs 143+162=305 (separate timed() medians): these are NOT a verified same-iteration
  serial decomposition — they are medians from SEPARATE timed() loops, each of which re-runs the
  full run_fn (the split timers call phase_T and phase_C as independent run_fns, so each carries its
  own warmup/launch/sync overhead -> their sum overcounts vs the single combined loop). Marked
  STILL-UNKNOWN until same-iteration cuda-event instrumentation (task #44) confirms the split.
- NAMING (enforced from here): B3 = "repeated-direct-pull serial baseline" (NOT production/strong).
  V4 = "direct-pull A-stationary mechanism proof" = B5 (NOT production fused winner).

*** GATE 1 RESULT (reverses the headline) ***
- vs the repeated-direct-pull serial baseline B3 (refetch A 32x): V4/B5 = 1.82x FASTER.
- vs the strong copy-once baseline B1-copy: V4/B5 = 291/678 = **0.43x (V4 is ~2.33x SLOWER).**
WHY: B5 still re-crosses XGMI for A (~4x at NSUB=8) AND runs at only 44 TFLOP/s end-to-end (LDS-bound,
occ 1-2). B1-copy moves A once then runs a B0-class GEMM. Fusion/overlap does NOT beat not-refetching.
DECISION RULE (M1024/N2048/K7168): perfect-overlap floor = max(T_copy,T_gemm)=max(143,162)=162us;
B1 total 291us => theoretical MAX speedup over B1 ~= 291/162 ~= 1.80x (a CEILING, not a target).
NEW DIRECTION: a competitive design needs simultaneously (1) ~1x remote A traffic, (2) B0-class GEMM
efficiency, (3) comm/compute overlap. That is the copy-once tile-inbox pipeline (P1) or expert-
granular double-buffer (P2), NOT V4-as-written. V4 is now a schedule ablation, not the main line.
B1 transfer = 7,569,408 B / 143us ~= 53 GB/s (payload-only) -- check if B1 transfer itself can go
faster; a faster B1 raises the bar and shrinks the overlap headroom.

## Phase B — strong baselines (B0/B1/B2) + recomputed V4 speedup
_status: PENDING_

## Phase B sweep + same-iteration event split (2026-06-26, cv350/r1_c4, np=2)

Full B0/B1 sweep. B0 = compute ceiling (no comm), B1-copy = copy-A-once + B0 GEMM (serial).
All RMS-rel ~0.0033, B1 zero_sentinel=True. M=128 SKIPPED (V2 GEMM requires M>=BM=256; M<256 faults).

| M | N | K | B0 us | B0 TFLOPs | B1 us | B1 TFLOPs(e2e) | transfer (B1-B0) us |
|---|---|---|---|---|---|---|---|
| 256 | 2048 | 7168 | 160.9 | 46.7 | 192.6 | 39.0 | ~32 |
| 512 | 2048 | 7168 | 161.9 | 92.8 | 225.5 | 66.7 | ~64 |
| 1024 | 2048 | 7168 | 165.6 | 181.5 | 291.1 | 103.3 | ~126 |
| 256 | 4096 | 7168 | 160.0 | 94.0 | 192.3 | 78.2 | ~32 |
| 512 | 4096 | 7168 | 162.9 | 184.5 | 225.8 | 133.1 | ~63 |
| 1024 | 4096 | 7168 | 168.4 | 357.2 | 292.7 | 205.4 | ~124 |
| 1024 | 7168 | 2048 (W2) | 72.5 | 414.7 | 108.4 | 277.4 | ~36 |

Observations (VERIFIED):
- B0 latency ~FLAT in M (160-170us) at N=2048 — the 256x256 tile floor dominates; small M wastes the
  tile (M256=47 TFLOP/s vs M1024=181 at ~same 163us). Small-M is GEMM-fixed-cost-bound, NOT comm-bound.
  Larger N amortizes (N4096 M1024 = 357 TFLOP/s).
- B1 transfer (B1-B0) ~LINEAR in M: 32/64/126us for M=256/512/1024; ~independent of N (A doesn't
  depend on N) — 32us at M256 for both N2048 and N4096.
- W2 shape (K=2048) cheap: B0=72us, transfer ~36us.

### SAME-ITERATION cuda-event split (B1_EVENTS=1) — resolves 291-vs-305, VERIFIED
copy + gemm measured around the SAME iteration + one enclosing event:

| M | N | K | copy us | gemm us | total_evt us | copy+gemm | overlap floor | max spd vs B1 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 2048 | 7168 | 134.9 | 150.3 | 285.2 | 285.2 (exact) | 150.3 | 1.90x |
| 512 | 2048 | 7168 | 71.7 | 147.6 | 219.3 | 219.3 (exact) | 147.6 | 1.49x |
| 256 | 2048 | 7168 | 41.3 | 145.5 | 186.8 | 186.8 (exact) | 145.5 | 1.28x |
| 1024 | 7168 | 2048 (W2) | 46.0 | 54.0 | 99.9 | 99.9 (exact) | 54.0 | 1.85x |

RESOLUTION of 291 vs 143+162=305: the 143/162 were medians from SEPARATE timed() loops (each carrying
its own per-iter sync overhead -> overcount). TRUE same-iteration split = copy 135 + gemm 150 = 285
EXACTLY (sums to the enclosing event). B1 IS literally serial copy-then-compute. [VERIFIED]
KEY CONSEQUENCE: at M1024 copy(135) < gemm(150) — GEMM is the longer pole. Perfect overlap hides the
135us copy under the 150us compute => floor 150us => MAX 1.90x over B1 (hard ceiling). At smaller M
copy shrinks (transfer~linear) while compute stays ~flat, so the overlap ceiling DROPS to 1.28x@M256.
=> Overlap only pays at LARGE M. At small M, copy is already small vs the fixed GEMM cost, so plain
copy-once+serial (B1) is near-optimal and overlap buys little. P1/P2 must target large-M to matter.

## B3/B4/B5 reuse-vs-overlap decomposition (2026-06-26, M1024/N2048/K7168) — VERIFIED, SURPRISING
| code | what | us | TFLOPs |
|---|---|---|---|
| B3 | repeated-direct-pull SERIAL (V3 baseline, refetch A per N-tile) | 1235.7 | 24.3 |
| B4 | A-stationary SERIAL (V4 baseline, no overlap) | 1230.7 | 24.4 |
| B5 | A-stationary + OVERLAP (V4 fused) | 676.6 | 44.4 |

- reuse gain  B3/B4 = 1235.7/1230.7 = **1.00x**  (A-stationary reuse alone bought ~NOTHING serially!)
- overlap gain B4/B5 = 1230.7/676.6 = **1.82x**  (the ENTIRE win is comm/compute overlap)
- combined    B3/B5 = 1.83x

*** OVERTURNS the V4 narrative. *** V4_ASTATIONARY_RESULTS claimed the win came from cutting redundant
A traffic 32x->4x (reuse). The data says reuse SERIALLY = 1.00x; the whole 1.82x is OVERLAP. Why
reuse looks free here: B4 still issues the same total remote bytes as B3 in this kernel (the
"A-stationary" rework changed the loop nest but the serial direct-pull still streams A through the
same blocking deref latency, so wall-time is unchanged); only when overlap hides that latency under
MFMA does it help. IMPLICATION for the new direction: the lever that matters is OVERLAP, and B1-copy
already shows that moving A ONCE (135us) is far cheaper than B3/B4's repeated pulls (~1070us of A
traffic). So P1/P2 = (copy-once like B1) + (overlap like B5's mechanism) is exactly the right combine;
neither the V4 reuse rework nor the direct-pull dataflow is worth keeping. [VERIFIED]

## Agent 10 bounded LDS diagnostic (v4_bsingle_buffer) — REFUTED, one pass only
Static resource (compile): fused 244 VGPR / occ 2 / 160B scratch / 0 spill (LDS dynamic -> static
pass shows 0; runtime LDS ~82KB by Agent 10's formula). Run M1024/N2048/K7168:
- FUSED **FAILED correctness** (RMS_rel=1.16, max_rel 1e6) — single-buffering B introduced a race
  (consumer reads B subtiles while producer overwrites them; B needs its buffering or a barrier).
- Timing (ignoring the bug): fused 671.7us vs V4 676.6us = within noise. **No throughput gain.**
VERDICT [REFUTED]: reducing LDS via B-single-buffer did NOT restore GEMM throughput AND broke
correctness. Per the one-pass rule, NOT pursued further. (Confirms the real V4 limiter is the
direct-pull dataflow/overlap mechanism, not LDS occupancy — consistent with B3/B4/B5 showing the win
is overlap, not reuse.) Occupancy was already 2 in stock V4; lifting it didn't help because the
kernel is comm-latency-bound, not compute-occupancy-bound, at these shapes.

## P1 / P2 copy-once overlap candidates (Agent 11) — BOTH FAIL, architecture suspect
_status: BLOCKED 2026-06-26 — stopping blind iteration per redirection's correctness-failure rule_
Two bring-up rounds + one focused fixer agent. Both candidates share a two-kernel cross-stream
producer/consumer-flag architecture and BOTH fail with the SAME signature:
- P1 tile-inbox: RMS_rel=inf (was 1.15), 4434us/iter (~15x SLOWER than B1's 291; goal <291).
- P2 expert-pipeline: RMS_rel=inf, 37568us/iter (E=32 TOTAL_M=8192 uniform).
- Fixes already tried (did NOT work): example int32->float32 IRIS alloc (real, needed); P1
  system-acquire fence + LDS scale-hoist; P2 host-side gl (compile fix, needed). Numerics still inf
  and perf still catastrophic on BOTH.
- DIAGNOSIS: the shared flaw is the two-kernel cross-stream flag handshake itself, NOT a local bug:
  (1) the producer materializes bands/experts with very few resident blocks while many consumer
  blocks spin-wait -> NO real overlap, effectively serialized + huge spin waste (explains 15-100x
  slowdown); (2) RMS=inf means the consumer GEMM reads inbox memory that is never correctly published
  for its tiles (flag/visibility or band<->row mapping wrong). Same family on P1 and P2 => design bug.
- DECISION: do NOT spawn a 3rd blind fixer into the same architecture. Two cleaner paths to evaluate
  next (see report): (A) a SINGLE-kernel persistent producer/consumer (no cross-stream, no flag spin
  -- in-block double-buffer like V4/B5 but consuming a LOCAL once-copied tile), or (B) accept B1's
  bulk-synchronous copy-then-GEMM as the dataflow and just SPEED UP the copy (it's only 56 GB/s of
  128 avail) -- per the decision rule, if overlap can't beat B1, bulk dispatch+local GEMM wins.
- Note from data: B5(V4) ALREADY achieves real in-kernel overlap (the 1.82x over B3 was ALL overlap).
  The working overlap mechanism is V4's in-block producer/consumer warps, NOT a two-kernel handshake.
  => Path A (single-kernel, copy-once into LOCAL tile, V4-style in-block overlap) is the most promising
  and reuses a MECHANISM WE KNOW WORKS. Schedule variants (04/05/07/08) stay PARKED.

## P3 single-kernel copy-once (Agent 12) — FAILS; reveals the PATTERN across P1/P2/P3
_status: FAILED 2026-06-26. M1024/N2048/K7168: P3-fused RMS=1.39 / 2535us; its own SERIAL-B1 repro
RMS=1.37 / 3369us. P3's "overlap" is 1.33x over its OWN broken serial = meaningless.

*** THE PATTERN (now 3/3) ***: P1, P2, P3 each had an AGENT REWRITE the remote fp8 gather + dequant
+ the in-module B1/reference from scratch, and ALL THREE produced RMS ~1.0-1.4 garbage AND 10-25x
slowdown. The AUTHORITATIVE working gather already exists: harness dispatch_pack_quant_once (B1) =
RMS 0.0033, copy 135us. P3's serial repro of "the same B1" is 25x slower (3369 vs 135) and wrong
(1.37 vs 0.0033) => the agents are NOT reproducing the working gather; they re-derive a broken/slow
one (likely per-element remote scalar scale loads + a scale-layout/transpose bug — the token-major vs
group-major trap Agent 00 flagged). The OVERLAP idea is not what's failing; the rewritten GATHER is.

CORRECTIVE PLAN (do NOT spawn a 4th from-scratch attempt):
- Build P4 by COMPOSITION, reusing VERIFIED binaries unchanged: harness dispatch_pack_quant_once
  (B1 gather, correct+fast) + harness local_gemm (B0, 183 TFLOP/s). First just CALL them back-to-back
  from one driver to reproduce B1=291us EXACTLY (sanity that composition works). THEN add overlap by
  the minimal correct means (e.g. tile the gather by m-strip and launch gemm per-strip behind it on
  the same stream, or a 2-stream dependency on cuda events) WITHOUT touching the gather/gemm internals.
- Reality check from data: B1 copy(135) < gemm(150) at M1024 => overlap ceiling = 1.90x, and ONLY at
  large M (1.28x @ M256). B1 already works, is simple, and is correct. If P4 overlap can't clearly
  beat B1, the DECISION RULE says bulk-synchronous copy-once + local GEMM (= B1 / B1-dispatch) IS the
  production dataflow and we optimize the COPY (56 GB/s of 128 avail) + the grouped dispatch instead.

## B1-dispatch main line — grouped 32-expert CORRECTNESS (Agent 02 V5) — PASS (2026-06-26)
CPU self-test (build_tasks.py): all 5 routes OK (uniform/zipf/one_hot/several_hot/many_empty),
adaptive NSUB=8, no layout contamination. Then np=2 GPU grouped GEMM, E=32 TOTAL_M=8192 N2048 K7168:

| route | Mpacked | tasks | RMS_rel | sentinel | verdict |
|---|---|---|---|---|---|
| uniform | 8192 | 512 | 0.00331 | True | PASSED |
| zipf | 8960 | 560 | 0.00331 | True | PASSED |
| one_hot | 8192 | 512 | 0.00331 | True | PASSED |
| several_hot | 8192 | 512 | 0.00331 | True | PASSED |
| many_empty | 8448 | 528 | 0.00331 | True | PASSED |

GROUPED 32-EXPERT GEMM IS CORRECT for all distributions: no cross-expert contamination, empty
experts safe, BM-padding works, route-row order preserved, zero-sentinel proves remote gather. This
is the LOCAL-GROUPED-GEMM half of B1-dispatch. [VERIFIED]
Test-harness fixes needed (NOT kernel bugs): B/C/TASKS moved off the 512MB IRIS heap to LOCAL torch
(only gathered A needs the symmetric heap; B was 3GB -> OOM); TASKS int32 local (IRIS has no int32).
PERF NOTE (expected, not the point of this task): the grouped kernel is still V4's DIRECT-PULL
A-stationary fused body, so it shows the same ~2x-over-weak-baseline / ~17 TFLOP/s as V4 (fused
14.5ms vs its own direct-pull baseline 32ms). The low throughput is the direct-pull dataflow, to be
replaced by copy-once gather when grouped is paired into B1-dispatch. Correctness was the goal here.

## B1-dispatch main line — EP8 multi-source gather CORRECTNESS (Agent 03) — PASS (2026-06-26)
np=8, Msrc=128 Mpacked=256 K=256, 10 route_segments, 4 tiles:
- RMS_rel = 0.00167, max_rel 0.0039 -> PASSED
- zero_sentinel_rows_exact_zero = True (unrouted/tail rows exactly 0)
- remote (XGMI) gathered rows = 207 from MULTIPLE source ranks (proves real multi-source EP8 path)
- segment-iterator straddle tiles = 4 (the cross-source-boundary path exercised)
EP8 MULTI-SOURCE GATHER IS CORRECT across all 8 ranks. This is the GATHER half of B1-dispatch. [VERIFIED]
Bugs fixed en route (all in the TEST/REF, not the gather kernel): (1) IRIS int32 alloc -> float32-
backed; (2) the nan was the CPU reference using ml_dtypes.float8_e4m3 (has inf; encodes 448 as inf ->
deq inf, 4135 non-finite) instead of float8_e4m3fn (OCP finite/saturating, what gfx950+torch use) +
saturating clip; (3) comm.gather -> allgather + per-rank finite asserts to localize. The gather
MECHANISM was correct from the first run (sentinel + 207 remote rows); only the reference was wrong.

### B1-dispatch status: BOTH HALVES VERIFIED
- LOCAL grouped 32-expert GEMM (Agent 02 V5): PASS all 5 routes (RMS 0.00331). [VERIFIED]
- EP8 multi-source gather/pack (Agent 03): PASS np=8 (RMS 0.00167). [VERIFIED]
NEXT: B1-dispatch V0 = wire EP8-gather-once (expert-major pack + route_reverse) -> local grouped GEMM,
measure vs B1-copy. (fp8 e4m3fn saturation is a PROJECT-WIDE ref hazard — note for any CPU reference.)

## B1-dispatch V0 — production-shaped EP8 pipeline — CORRECTNESS PASS (2026-06-26) *** MILESTONE ***
np=8, E=32, TOTAL_M=8192, N=2048, K=7168, MSRC=4096/rank. Composition of the two VERIFIED components
(EP8 multi-source gather + v5 grouped GEMM), NO rewrite. Phase1 = gather/pack/quant ONCE into local
expert-major buffer; Phase2 = local grouped GEMM (src_rank=CONSUMER -> ctx.load is local).

| route | Mpacked | segs | phase1 packed-A RMS | e2e RMS_rel | verdict |
|---|---|---|---|---|---|
| uniform | 8192 | 420 | 0.000000 (0 mismatch) | 0.003702 | PASSED |
| zipf | 8960 | 421 | 0.000000 | 0.003701 | PASSED |
| one_hot | 8192 | 402 | 0.000000 | 0.003703 | PASSED |
| several_hot | 8192 | 406 | 0.000000 | 0.003703 | PASSED |
| many_empty | 8448 | 406 | 0.000000 | 0.003702 | PASSED |

B1-DISPATCH V0 IS CORRECT: real EP8 multi-source routing -> gather/pack once -> grouped 32-expert
GEMM, all distributions, 7111 rows over XGMI from multiple ranks. [VERIFIED]

*** BUG FOUND + FIXED BY MAIN AGENT (directly, on the node) ***
First run: e2e RMS=0.95, ~70% rows wrong. A phase-1 isolation probe (dequant the packed buffer, diff
vs reference) localized it to PHASE 1: packed-A RMS=0.95, 5768/8192 rows mismatched. ROOT CAUSE:
build_row_seg_map stored the ABSOLUTE segment index (seg_begin+si, up to ~420 at 32 experts) in a
`signed char` row_seg[] -> overflow at 127 -> wrong segment for every row whose seg index >127 (~70%).
This is EXACTLY the risk Agent 03 flagged in its original report. FIX: row_seg + SEG_NONE +
build_row_seg_map + the reader changed signed char -> int (kernel.cpp AND ep8_gather.h). -> RMS 0.000.

PERF (NOT optimized yet — next step): T_gather ~207us (phase1, the gather/pack ONCE) + T_gemm ~7400us
(phase2) = ~7640us e2e at 31 TFLOP/s. The gather is cheap; T_gemm is huge because phase2 uses v5's
DIRECT-PULL serial baseline (micro_tk_baseline re-reads A per N-tile via ctx.load — even though local,
it re-reads K*N/BN bytes from HBM and runs at ~32 TFLOP/s, far below B0's 183). NEXT: replace phase2's
per-tile A re-read with a copy-once-into-LDS local grouped GEMM (B0-class) so T_gemm approaches the B0
ceiling. Then B1-dispatch total should approach T_gather + (B0-class grouped GEMM). Compare vs B1-copy.

## B1-dispatch V1 — fused A-stationary phase2 (2026-06-26) — 2.1x over V0, correct all 5 routes
Ported v5's micro_tk (A-stationary: gather A once per K-tile, reuse across NSUB) into b1_dispatch;
dispatch branches on g.fused; example FUSED=1. Phase1 unchanged (RMS 0.000000 all routes).

| route | V0 T_total us | V1 T_total us | V1 TFLOPs(e2e) | RMS_rel |
|---|---|---|---|---|
| uniform | 7641 | 3684 | 65.3 | 0.0037 |
| zipf | (n/a) | 4602 | 52.3 | 0.0037 |
| one_hot | (n/a) | 3310 | 72.7 | 0.0037 |
| several_hot | (n/a) | 3386 | 71.0 | 0.0037 |
| many_empty | (n/a) | 4640 | 51.8 | 0.0037 |

uniform breakdown: T_gather 216us + T_gemm 3462us (69 TFLOP/s, was 32). The A-stationary fusion gave
~2.1x on the GEMM. All routes correct. [VERIFIED]
STILL below B0's 183 TFLOP/s: the GEMM reads A from HBM via ctx.load per K-tile (not the clean B0
LDS-resident path) and is LDS-bound at occ-2 (the V4 family limit). The gather (216us) is now a small
fraction of total. NEXT (V2, if pursued): an LDS-resident B0-class grouped GEMM over the packed buffer
to push TFLOP/s toward 183 — but note B1-dispatch is already a CORRECT production-shaped EP8 pipeline.

## Phase C — single-expert schedule ablations (4P4C / 8-wave / 4-wave / occupancy / XCD / cache)
_status: PARKED per redirection — schedule variants (04/05/07/08) park until copy-once pipeline works_

## Phase D — grouped 32-expert, np=2 (uniform/Zipf/hot/empty)
_status: PENDING_

## Phase E — grouped EP8 multi-source (np=8)
_status: PENDING_

## Phase F — captured production distributions (C4/C6 replay)
_status: PENDING_

## Phase G — full production shapes (W13 / W2 / full FFN)
_status: PENDING_

## Phase H — ATOM integration (TPOT/TTFT/throughput)
_status: PENDING_
