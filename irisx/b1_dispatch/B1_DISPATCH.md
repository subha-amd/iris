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

## Complete region — full FFN (fc1 g1u1 + SiLU + fc2) + combine (EpCombine)

The single-GEMM path above is a *partial* region (gather + one projection). For a head-to-head vs the
production unfused region (`b2_production/b3_ep8_unfused.py`: dispatch / fmoe / combine), `example.py`
also runs the **complete** expert region, gated by env, all ADDITIVE (defaults reproduce the original
single-GEMM path):

```
FFN=full   : gather -> fc1 (g1u1, N=4096 K=7168) -> SiLU(gate)*up + fp8 re-quant -> fc2 (N=7168 K=2048)
             both projections via grouped_gemm_b0 (already compiled for both (N,K)); requires SCHEDULE=b0.
COMBINE=1  : after fc2, scatter the output back to origin tokens over IRIS (combine_scatter): per row,
             acc[src_token] += route_weight * out[row], accumulating top-k contributions on origin ranks.
```

The intermediate `SiLU(gate)*up` + fp8 per-128-group re-quant (the production `dynamic_quant`) is a
torch op between the two GEMMs and is timed as **T_act** so the quant cost is accounted honestly.

**Build the module** (on the node, inside the HK checkout):
```bash
cd /tmp/HipKittens/distributed-kernels
cmake -B build -DGPU_TARGET=CDNA4 -DDK_BUILD=b1_dispatch
cmake --build build -j16 --target b1_dispatch && cmake --build build -j16 --target iris_py
```

**Run the complete region** (8× MI350, gfx950; prints T_gather / T_fc1 / T_act / T_fc2 / T_combine / T_total):
```bash
cd /tmp/HipKittens/distributed-kernels/b1_dispatch
ROUTE=uniform TOTAL_M=8192 SCHEDULE=b0 FFN=full COMBINE=1 \
  mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 \
    -x HSA_XNACK=1 -x MORI_GPU_ARCHS=gfx950 -x PYTHONPATH=/tmp/HipKittens/distributed-kernels \
    python3 example.py
```

Two correctness gates print: **FFN** (C2 vs a CPU fc1→SiLU→requant→fc2 reference; tol 0.05 for the
two fp8-A GEMMs + intermediate quant) and **COMBINE** (the GPU-scattered accumulator vs a CPU scatter
of the SAME fc2 output; tol 0.02 — isolates the scatter/weight/accumulate from the FFN numerics).

New pieces (all additive; single-GEMM path untouched):
- `kernel.cpp::combine_scatter` — the IRIS scatter-back kernel (mirror of phase-1 gather, reversed),
  bound as `tk_kernel.combine_scatter(c2, acc, rev, wgt, iris_ctx, Mpacked, H, Tlocal, atomic)`.
- `b1_dispatch_route.py::build_route_reverse` / `combine_reference` — the host reverse-map builder +
  CPU combine reference. `route_reverse` is derived from the SAME segments (coherent with the gather).
- `example.py` — `FFN`/`COMBINE`/`COMBINE_ATOMIC`/`COMBINE_TLOCAL` knobs, the two-GEMM FFN chain with
  the SiLU+re-quant epilogue, and the per-stage timing.

HONESTY / accumulation: the synthetic route advances each source rank's cursor monotonically, so by
default each (src_rank, src_token) is referenced once — combine is a weighted SCATTER and the top-k
`+=` never collides (the kernel still uses atomic-add, so it is correct if it did). Set
`COMBINE_TLOCAL=N` (< per-rank token span) to FOLD src_token and force real accumulation; the CPU
reference applies the identical fold so correctness stays checkable. A real top-k router would produce
those collisions naturally — see the note in `b1_dispatch_route.py::build_route_reverse`.
