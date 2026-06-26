# B1-dispatch V0 — production-shaped EP8 MoE expert kernel (by composition)

B1-dispatch is the route-aware EP8 analog of B1-copy: REAL top-k routing, 32 local experts, and
MULTI-SOURCE gather (each expert's rows were routed there from up to 8 ranks), kept SERIAL (no
comm/compute overlap yet — overlap is a later experiment, pursued only if it beats this).

It is the back-to-back COMPOSITION of two already-verified components. The #1 rule of this task is to
REUSE the verified gather + GEMM, NOT rewrite them: the three prior overlap candidates (P1/P2/P3) all
failed precisely because each agent rewrote the remote gather from scratch -> RMS~1.0 garbage + 10-25x
slow (see EXPERIMENT_LEDGER "THE PATTERN (now 3/3)"). B1-dispatch sidesteps that by composing the
verified code unchanged.

## Two-phase structure (serial, on the consumer rank)

```
PHASE 1  dispatch_gather_pack(...)   A crosses XGMI EXACTLY ONCE here
  multi-source EP8 gather of each expert's rows from their SOURCE ranks (route_segments) ONCE into a
  LOCAL expert-major packed fp8 + scale buffer (v5_grouped layout: BM-padded per-expert regions;
  padding/unrouted rows stay zero = zero-sentinel).
      -> A_packed_fp8[Mpacked, K] (fp8-as-bf16) + A_packed_sc[Mpacked, NG], on the consumer's heap.

PHASE 2  grouped_gemm(...)           A does NOT re-cross XGMI
  the VERIFIED v5_grouped SERIAL grouped GEMM (micro_tk_baseline) over that LOCAL packed buffer.
  Called with src_rank = CONSUMER so the kernel's internal ctx.load(..., src_rank) is a LOCAL HBM
  deref. 32 experts, one flat task grid (build_tasks.py).
      -> C[Mpacked, N] (bf16).
```

This is the production-shaped analog of B1-copy (which was a single fixed-source dense [M,K] copy +
a single dense B0 GEMM). B1-dispatch replaces that with REAL routing, 32 variable-M_e experts, and a
multi-source gather, while keeping the same "move A once, then a local GEMM" dataflow that beat V4 at
Gate 1 (B1-copy 291us vs V4 678us).

## What is reused VERBATIM vs new glue

| piece | source | status |
|---|---|---|
| multi-source row resolver: `seg_tile_view`, `tile_is_single_source`, `build_row_seg_map`, `route_segment` ABI, the per-row (src_rank,src_row) resolution | `ep8_gather/ep8_gather.h` (Agent 03, np=8 RMS 0.00167) | included unchanged; resolution lifted verbatim into the phase-1 loop |
| raw fp8-byte + scale movement (uint4 remote load -> local store, local short-circuit) | harness `gather_once_kernel` body (B1-copy, RMS 0.0033, 135us) | copied verbatim |
| grouped 32-expert GEMM `micro_tk_baseline` + `gather_dequant_A_tile` + `fp8_to_f32` + `micro_globals` | `v5_grouped/kernel.cpp` (Agent 02, all 5 routes RMS 0.00331) | copied verbatim (the whole phase-2 block) |
| task list + adaptive NSUB + packed layout | `v5_grouped/build_tasks.py` | copied verbatim into this dir |

NEW GLUE (the only new code):
1. `gather_pack_kernel` driver loop — walks packed BM-tiles, calls the verified resolver per row,
   copies raw fp8+scale into the packed buffer, writes the zero-sentinel for unrouted/tail rows.
2. `b1_dispatch_route.py` — generates 32-expert MULTI-SOURCE route_segments over the v5 BM-padded
   packed layout (ties Agent 02's packing to Agent 03's gather ABI) + per-BM-tile metadata.
3. the two pybind wrappers (`dispatch_gather_pack`, `grouped_gemm`) and the example.py driver.

## Phase1 -> Phase2 handoff (byte-exact layout match)

Phase 1 writes, and phase 2 reads, the SAME two buffers in the SAME layout:
- `A_packed_fp8[Mpacked, K]` fp8 e4m3 row-major (fp8-as-bf16 view), token-major.
- `A_packed_sc[Mpacked, NG]` fp32 per-128-group scales, token-major (NG = K/128).
- `Mpacked = sum_e padded(M_e)`, each expert's region padded UP to a multiple of BM
  (`build_tasks.build_packed_layout`); `expert_row_begin[e]` is the per-expert prefix.

Token-major scales `[Mpacked, NG]` is exactly what v5_grouped's `gather_dequant_A_tile` expects
(`g.sc[{0,0,gr,grp}]`). Production's group-major scale transpose (FMOE_LAYOUT.md §5) is a LATER
concern; V0 keeps phase 1 and phase 2 in the SAME order so they compose correctly. Padding/unrouted
rows are zero in both fp8 and scale -> dequant to 0 -> contribute 0 to C (no cross-expert
contamination; the same guarantee v5 relies on).

## Correctness

CPU reference = the v5 grouped reference: for each expert e, `C[rows_e] = dequant(gathered A[rows_e])
@ B[e]^T`; the gathered A is reconstructed on the host from each rank's dequantized source buffer via
the same route_segments. Expected:
- `RMS_rel < 0.01` (v5 measured 0.00331; same dequant + GEMM).
- `packed_A_nonzero = True` and `C_zero = False` (phase 1 actually moved bytes; phase 2 wrote output).
- zero-sentinel: unrouted packed rows produce EXACTLY-zero C rows.
- `remote (XGMI) gathered rows > 0` (real multi-source path exercised, not all-local).

## Timing

`T_gather` (phase 1) / `T_gemm` (phase 2) / `T_total` (serial sum) measured same-iteration with cuda
events, directly comparable to B1-copy's copy/gemm/total split (135 / 150 / 285us at M1024
single-source). B1-dispatch's gather is multi-source over 32 experts, so T_gather is the honest
production gather cost; T_gemm is the grouped-GEMM cost. Per the project decision rule: if a future
overlapped design cannot clearly beat this serial B1-dispatch, bulk gather-once + local grouped GEMM
IS the production dataflow and we optimize the gather (currently ~56 GB/s of ~128 available) and the
grouped schedule, not a fused GEMM.
