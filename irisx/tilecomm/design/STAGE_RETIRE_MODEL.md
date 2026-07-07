# The `stage` / `retire` typed-tile-edge programming model

> **What this is.** A programming model — a *type system for tile edges* — in which a tile's
> **RESIDENCY** (it may physically live on another rank) and its **FORMAT** (it may be a compressed,
> per-block-scaled encoding) are first-class type parameters, and the kernel author never writes
> `load` / `store` / `get` / `put` / dequant by hand. The author declares an *edge* into or out of an
> HK register/shared tile; the compiler lowers that edge to the right backend (CDNA4 scaled-MFMA on the
> FORMAT axis, IRIS RMA + a pipeline schedule on the RESIDENCY axis).
>
> This is codex's proposal (the "empty quadrant = representation + residency, not overlap" cell,
> `scratchpad/MEASURED_FINDINGS.md §D`). **It is an API-level unifier, not one coherent cost model** —
> §6 states that boundary precisely and shows how to frame the thesis so a reviewer cannot ask
> "which cost model arbitrates XGMI-overlap vs HBM-bytes vs scale-layout vs MFMA-occupancy vs accuracy."
>
> Status: **design only.** No line of this API is built. The numbers it cites are measured on
> 8× MI350X (gfx950); the model is the notation we would build the two *already-measured* lowerings
> (QuantTile / decode, Overlap-A / prefill) underneath. Every unknown is flagged in §7.

---

## 1. The one idea

Today an HK kernel that participates in a distributed MoE region has three concerns tangled into its
inner loop:

1. **where** an operand physically is (local HBM, or a peer's symmetric heap reached by an IRIS
   `get`/`load`),
2. **how** it is encoded (bf16, fp8-e4m3 with per-128-K fp32 scales, MXFP4 with per-32-K E8M0 scales),
3. **when** its movement runs relative to the MFMA that consumes or produces it.

Those three are hand-coded, per kernel, per format. The shipped fused MoE region proves the cost of
that tangle: the `_sat` decode path, the `mxfp4_sat` decode path, and the bf16 prefill path are
**three separate GEMM bodies** that differ only in the decode of one operand
(`MASTER_HANDOFF.md §5`), and the combine collective's schedule is a hand-rolled round-robin baked
into `build_combine_pull(interleave=True)` (`DESIGN.md §1`).

`stage`/`retire` collapses (1)–(3) into **two verbs on a typed tile edge**:

- `stage(dst_tile)` — an **inbound** edge that *fills* an HK tile from a declared residency + format.
- `retire(c_tile)`  — an **outbound** edge that *disposes* of a produced tile via a declared
  collective + overlap schedule.

The tile edge is parameterised on two **orthogonal type axes**:

| axis | what varies | backend it lowers to | the currency it trades in |
|---|---|---|---|
| **RESIDENCY / transport** | `local` vs `remote<rank>` vs "reduction of a group" | IRIS RMA (`get`/`store`/`atomic`) + a tile pipeline schedule | XGMI-overlap, link contention |
| **FORMAT** | `bf16` / `fp8_e4m3{scale}` / `mx_fp4{scale}` / … | CDNA4 scaled-MFMA (`mma_ABt_scaled` → `mfma_scale_f32_16x16x128_f8f6f4`) | HBM bytes, scale-layout, MFMA occupancy, accuracy |

The axes are **orthogonal by construction**: residency does not change the numerics, format does not
change the topology. That orthogonality is exactly what lets the two axes keep *separate* cost models
(§6) — it is the load-bearing property of the whole design.

---

## 2. The two axes are real, measured levers (not hypotheticals)

The model exists to notate two lowerings we have *already measured the ceiling of*:

- **FORMAT axis → the decode weight wall.** Decode streams all 32 local experts' ~1.4 GB of fp8
  weights every step — a fixed ~176 µs HBM floor at 8 TB/s — and gather+combine are only ~18% of the
  region, so the decode region is ≈80% weight floor + 20% boundaries (`MASTER_HANDOFF.md §5.1`).
  The *only* high-ceiling decode lever is **fewer weight bytes**. Route-1 W4A16 MXFP4 already shipped
  **~1.6× over the fp8 `_sat` decode GEMM** (standalone, same node) by streaming ¼ the bytes and
  decoding fp4→bf16 in-register with the gfx950 `cvt_scalef32_pk_bf16_fp4` builtin
  (`MASTER_HANDOFF.md §10.1`). This is the FORMAT axis with `residency = local`.

- **RESIDENCY / transport axis → the TP4 prefill all-reduce.** On the TP4 prefill path the all-reduce
  after the GEMM is **59–85% of the serial GEMM+AR time** (measured, `MEASURED_FINDINGS.md §A`): it is
  1.2–4.5× the GEMM itself. If the GEMM's output tiles are reduce-scattered *as they are produced*, the
  short GEMM hides under the long AR and the fused sequence approaches `max(GEMM, AR) = AR`, giving a
  **~1.2–1.7× prefill ceiling** (down-proj 1.43×, attn out-proj 1.18×, example-default 1.69×,
  small-prefill 1.49×). This is the RESIDENCY axis with `format = bf16`.

Both are *ceilings*, and the honest state of each is in §7. The point of §2 is only that the two type
axes name two levers that measurement has already sized — the notation is not decorative.

---

## 3. The API surface

The model lives at the **HK (C++ tile) layer**; the transport backend is **IRIS** (a Triton device-side
RMA library), and the format backend is the **CDNA4 scaled-MFMA**. The fluent chain below is C++-flavored
pseudo-syntax; the RESIDENCY clauses map 1:1 onto IRIS calls, the FORMAT clauses onto HK tile descriptors.

### 3a. Inbound edge — `stage`

```cpp
//  stage(dst)  opens an INBOUND edge that FILLS the HK tile `dst`.
//  RT is the destination register/shared tile the consuming MMA reads.
template <typename RT>
staged_edge<RT, Local, Bf16, Unbound>   stage(RT& dst);

//  .from(src) — RESIDENCY clause.  Binds WHERE the bytes physically are.
//     local()              -> this rank's symmetric-heap / HBM base pointer (no RMA)
//     remote(int rank)     -> a peer's symmetric-heap slot, reached by IRIS get/load
//     remote(rank_expr)    -> MUST resolve through a constexpr tl.static_range(WORLD) loop;
//                             a data-dependent to_rank FAULTS ("write to read-only page")
//                             — the measured IRIS gotcha (MASTER_HANDOFF §0.5, DESIGN §6.1)
template <typename R>
staged_edge<RT, R, F, C>&   from(residency<R> src);

//  .format(F) — FORMAT clause.  Binds the on-the-wire / in-HBM ENCODING + its scales.
//     bf16{}                                   -> no decode; direct buffer_load_b128
//     fp8_e4m3{ scale_ptr, PER_128K, FP32 }    -> the `_sat` path: memory fp8, math bf16,
//                                                 measured 3.97 TB/s on the BM=16 tile
//     mx_fp4{ scale_ptr, block_k=32, E8M0 }    -> scaled-MFMA operand; fp4 e2m1 packed 2/byte,
//                                                 E8M0 per-32-K scale, decoded in-register
//     mx_fp8{ scale_ptr, block_k=32, E8M0 }    -> the HK `mxfp8/MXFP8_8wave` body (421 TFLOPS)
template <typename Fnew>
staged_edge<RT, R, Fnew, C>&   format(Fnew desc);

//  .for_mma<Role>() — CONSUMER clause.  Declares the tile is an MMA OPERAND, so the lowering
//     targets the register-tile K-major layout the MFMA wants (Role = A | Bt), and for scaled
//     formats EMITS THE E8M0 SCALE OPERAND alongside `dst`, ready for mma_ABt_scaled.
//     Terminal: materializes and returns the filled tile.
template <mma_role Role = Bt>
RT&   for_mma();
```

**What each inbound clause controls, precisely:**

| clause | axis | controls | trivial (default) value |
|---|---|---|---|
| `.from(...)` | RESIDENCY | which IRIS primitive (none / `load` / `get`), and the constexpr-`static_range` guard on `to_rank` | `local()` → no RMA |
| `.format(...)` | FORMAT | the HBM byte footprint, the pre-swizzle, the decode builtin, and whether a scale operand is materialized | `bf16{}` → no decode |
| `.for_mma<Role>()` | CONSUMER | the destination register layout (A vs Bt, K-major) and scale-operand emission for `mma_ABt_scaled` | (terminal; required) |

### 3b. Outbound edge — `retire`

```cpp
//  retire(c)  opens an OUTBOUND edge for a PRODUCED tile (a GEMM accumulator).
template <typename RT>
retiring_edge<RT, LocalStore, BulkSync>   retire(RT& c);

//  .reduce_scatter(group, op) — TRANSPORT clause.  Declares the tile's destination is the
//     REDUCTION of this group's contributions, scattered to each element's owner rank.
//     group = tp_group{ranks...} | ep_group{...};  op = sum.
//     siblings: .scatter(dst_map) | .gather(src_map) | .all_reduce(group, op)
//     (all_reduce == reduce_scatter + all_gather — the decomposition Osama named.)
template <typename Dest>
retiring_edge<RT, Dest, S>&   reduce_scatter(comm_group group, reduce_op op);

//  .format(F) — FORMAT clause on the OUTBOUND edge (the compositional cell, §5).
//     Declares the tile is REDUCED / SHIPPED in a compressed encoding, e.g. combine in
//     mx_fp4 = ½–¼ the XGMI bytes.  UNMEASURED (§7) — the design's aspiration, not a result.
template <typename Fnew>
retiring_edge<RT, Dest, S>&   format(Fnew desc);

//  .overlap_with(next) — SCHEDULE clause.  Declares this tile's EGRESS may run concurrently
//     with the compute that produces `next` (the next GEMM output tile).  This is the ONLY
//     clause that makes a claim the transport cost model must arbitrate: it opens the
//     "how many tiles produced before we reduce" pipeline space (Osama's prize).
//     Terminal.  Omitting it == .commit() == bulk-synchronous (the safe default).
void   overlap_with(tile_ref next);
void   commit();   // bulk-synchronous terminal
```

**What each outbound clause controls, precisely:**

| clause | axis | controls | trivial (default) value |
|---|---|---|---|
| `.reduce_scatter / .scatter / .gather / .all_reduce` | RESIDENCY | the collective intent → the IRIS `store`/`atomic` pattern + the link-balanced tile schedule (`tilesched.py`) | `LocalStore` → plain write-back |
| `.format(...)` | FORMAT | whether the tile is *compressed before* egress and *decoded on arrival* (the §5 cell) | uncompressed |
| `.overlap_with(next)` | SCHEDULE | tiles-produced-before-reduce; whether egress is emitted *inside* the GEMM tile loop | `BulkSync` → standalone collective kernel |

The **type of the edge** after the chain — e.g.
`staged_edge<rt_bf<32,32>, Remote<r>, MxFp4<32,E8M0>, ForMma<Bt>>` — is what the compiler dispatches the
lowering on. The author writes the fluent chain; **the lowering is template specialization on the two
type axes.** That is the mechanical meaning of "two lowerings of the same declaration" in §4.

---

## 4. QuantTile and Overlap-A are two lowerings of the SAME declaration

Both are `stage`/`retire` chains. They differ only in *which type axis is non-trivial*, and the compiler
picks the backend from that.

### 4a. QuantTile (B) — the FORMAT axis is non-trivial, residency is `local` (DECODE)

```cpp
// decode expert-weight operand: fp4 in HBM, decoded at MFMA time, ONE body for fp8 AND mxfp4
auto& B = stage(b_rt)
            .from( local() )                       // RESIDENCY trivial: no RMA
            .format( mx_fp4{ w_scale, /*block_k=*/32, E8M0 } )   // FORMAT non-trivial
            .for_mma<Bt>();
mma_ABt_scaled(acc, A, B, acc, a_scale, w_scale);  // mfma_scale_f32_16x16x128_f8f6f4
```

**Lowering (FORMAT-axis backend = CDNA4 scaled-MFMA):** `buffer_load` the packed fp4 bytes + the E8M0
scales into a register tile, decode in-register (`cvt_scalef32_pk_bf16_fp4`, or feed the scaled-MFMA
directly for a true fp4×fp4 body), no RMA anywhere. Swapping `mx_fp4{}` → `fp8_e4m3{PER_128K,FP32}`
changes *only the `.format(...)` argument* and re-targets the same body to the `_sat` fp8 path. **This is
codex's falsifiable v0** (`MEASURED_FINDINGS.md §D`): one HK grouped decode GEMM body serving both fp8-sat
and MXFP4 through a tile descriptor, **no per-format kernel fork** — collapsing the three shipped bodies
of `MASTER_HANDOFF.md §5` into one.

The currency this lowering trades in is entirely on the FORMAT side: **HBM bytes** (¼ for fp4),
**scale-layout** (per-128-K fp32 → per-32-K E8M0 replication is lossless; fp32→E8M0 rounding is lossy
unless the model is genuinely MX-quantized — the real remaining gap, `MASTER_HANDOFF.md §5.6`),
**MFMA occupancy**, and **accuracy** (decode fp4 precision class RMS 0.117 vs fp8 0.057 vs bf16 0.018).

### 4b. Overlap-A (A) — the RESIDENCY/transport axis is non-trivial, format is `bf16` (PREFILL)

```cpp
// prefill TP4: reduce-scatter each output tile as it is produced, overlapped with the next tile
for (int t = 0; t < n_tiles; ++t) {
    mma_ABt(acc[t], A[t], B[t], acc[t]);           // FORMAT trivial: bf16 MFMA
    retire( acc[t] )
        .reduce_scatter( tp_group, sum )           // RESIDENCY non-trivial: IRIS store+reduce
        .overlap_with( tile(t + 1) );              // SCHEDULE: hide egress under next MFMA
}
```

**Lowering (RESIDENCY-axis backend = IRIS RMA + tile pipeline schedule):** emit tile-`t`'s partial into
peers' symmetric heap between GEMM iterations, so tile-`t`'s egress overlaps tile-`(t+1)`'s MFMA; the
schedule chooses tiles-produced-before-reduce and the link-balanced store order (`tilesched.py`). The
`.format(...)` clause is absent, so the FORMAT backend is never invoked. This is the ~1.2–1.7× prefill
lever of §2.

The currency this lowering trades in is entirely on the RESIDENCY side: **XGMI-overlap** (the pipeline
depth) and **link contention** (the round-robin/proportional schedule).

### The punchline

`stage`/`retire` is **one declaration surface**; QuantTile and Overlap-A are the *format-only* and
*transport-only* specializations of it. The author writes the same two verbs; the compiler dispatches on
the type axes and never invokes the backend for a trivial axis. This is what makes it a unifier at the
API level — and *only* at the API level (§6).

---

## 5. The compositional cell — the genuinely-empty quadrant (UNMEASURED)

The reason the two verbs are worth unifying (rather than shipping as two unrelated features) is the cell
where **both** axes are non-trivial — a tile whose *remote residency AND compressed representation* are
both first-class:

```cpp
// EP combine shipped in fp4: ¼ the XGMI bytes AND reduced on arrival — "preserve compression
// until the last responsible moment."  This cell is codex's "empty quadrant."
retire( expert_out )
    .format( mx_fp4{ out_scale, 32, E8M0 } )   // FORMAT: ship compressed
    .reduce_scatter( ep_group, sum )           // RESIDENCY: reduce on the destination
    .overlap_with( next_expert_tile );
```

This is the cell prior art does *not* occupy: overlap-only (Flux / CoCoNet / MSCCL++ / TRT-LLM fused
AR+RMSNorm+quant) and format-only (per-format GEMM kernels) each take one axis; the *union* — a typed tile
that carries transport *and* format together — is open (`MEASURED_FINDINGS.md §D`). The combine already
ships fp8 over XGMI today (`gather_pack`), so shipping fp4 for ½–¼ the bytes is *plausible*.

**Honesty gate (do not soften):** this cell is **unmeasured**. It is the design's aspiration, not a
result. Worse, it is the *only* place the two axes interact — decoding-on-arrival vs reducing-in-flight
couples the FORMAT decode into the RESIDENCY schedule — so it is exactly the cell the decoupled cost
models of §6 do **not** cover. Framing it as "future work whose value the model exists to *express*"
(not to *deliver*) is mandatory.

---

## 6. The honest boundary — an API-level unifier, NOT one coherent cost model

This is codex's sharpest point (`MEASURED_FINDINGS.md §D`, `RESEARCH_BRIEF.md §5`): "transport + format"
is an **API-level unifier only, NOT one coherent optimization problem." A is a dependency/pipeline
scheduler; B is a representation/MFMA lowering. Pitching them as one invites the fatal reviewer question:

> *"Which single cost model arbitrates XGMI-overlap vs HBM-bytes vs scale-layout vs MFMA-occupancy vs
> accuracy?"*

**We do not have that model, and we should not claim to.** Those are five incommensurable currencies:

| currency | axis | nature | its cost model |
|---|---|---|---|
| XGMI-overlap (µs hidden under compute) | RESIDENCY | pipeline / scheduling | `tilesched.py` wave model (calibrated to combine 934/386 µs) |
| link contention (bytes/link) | RESIDENCY | traffic engineering | same `tilesched.py` |
| HBM bytes streamed | FORMAT | representation | roofline (176 µs weight floor @ 8 TB/s) |
| scale-layout (per-128 fp32 vs per-32 E8M0) | FORMAT | layout / correctness | lossless-replication vs lossy-rounding rule (§5.6) |
| MFMA occupancy (VGPR / waves) | FORMAT | codegen | HK 8-wave roofline (TFLOPS) |
| accuracy (RMS from E8M0 rounding) | FORMAT | numerics | the RMS ledger (0.018 / 0.057 / 0.117) |

There is **no scalar objective** over these. You cannot write `minimize(cost)` across a µs of hidden
comm, a byte of HBM traffic, and a unit of RMS error. Any thesis that implies one is over-claiming, and a
reviewer will find the seam in thirty seconds.

### How to frame the thesis so the question never lands

Position the contribution as a **type system / programming model — an *interface* contribution,
explicitly not an *optimizer* contribution.** The exact claim, verbatim-safe:

1. **The novelty is the interface, not a solver.** `stage`/`retire` makes RESIDENCY and FORMAT
   first-class type parameters and hides `load`/`store`/`get`/`put`/dequant. One declaration drives two
   backends. That is the whole claim.

2. **Each axis lowers independently to its own, separately-validated cost model.** The RESIDENCY axis
   lowers to IRIS RMA governed by `tilesched.py`; the FORMAT axis lowers to CDNA4 scaled-MFMA governed by
   the roofline + RMS ledger. **We deliberately keep the cost models separate** — this is a design
   decision, stated up front, not a gap.

3. **The decoupling is *sound*, not a punt, because the axes are orthogonal by construction** (§1):
   residency does not change the numerics, format does not change the topology. Two orthogonal axes are
   *entitled* to two independent cost models; a unified model would be needed only if they interacted
   everywhere, and they interact in exactly one cell (§5), which we scope as future work.

4. **We measure the two single-axis lowerings against numbers we already trust** — QuantTile-v0 against
   the fp8 `_sat` 3.97 TB/s decode GEMM (codex's falsifiable milestone: fp8 within ~5% of standalone,
   MXFP4 keeping ~1.6×, no per-format fork), and Overlap-A against the measured ~1.2–1.7× prefill AR
   ceiling — and we do **not** claim the compositional cell.

This converts the reviewer's attack ("you have no unified cost model") into a *stated design principle*
("we deliberately decoupled two orthogonal axes; each keeps its own validated model; unifying them is
explicitly out of scope"). The word "unify" applies to the **declaration surface**, never to the
**objective function.** Hold that line and the question in the box above has no purchase.

---

## 7. What is measured, what is claimed, what is unknown (the honest ledger)

| item | status |
|---|---|
| FORMAT-axis lever real (decode weight wall, fp4 = fewer bytes) | **MEASURED.** Route-1 MXFP4 ~1.6× over fp8 `_sat` standalone decode GEMM; fused region 1.11× over fp8 / 1.69× over bf16 (8× MI350, same node). |
| fp8 `_sat` decode baseline = 3.97 TB/s | **FLAGGED FOR VERIFY** (codex): needs the HK build to re-confirm before the QuantTile-v0 delta rests on it. `MEASURED_FINDINGS.md §C`. |
| RESIDENCY-axis lever real (TP4 prefill AR = 59–85% of GEMM+AR) | **MEASURED.** `MEASURED_FINDINGS.md §A`. |
| Overlap-A ~1.2–1.7× prefill ceiling | **CEILING, not achieved.** It is `max(GEMM,AR)=AR` reasoning, not a fused measurement. |
| A realizable on the *current* IRIS substrate | **NO.** Shipped IRIS fused examples are 4.7× (ex.09, one-shot AR) to 280× (ex.08, atomic AR) slower than unfused torch+RCCL; ex.09's Triton GEMM is 471 vs torch's 1149 TFLOP/s. Realizing A needs a competitive producer/consumer-warp GEMM body **and** an in-kernel reduce-scatter matching RCCL BW — both substantial, the reduce-scatter unproven. `MEASURED_FINDINGS.md §B`. |
| The compositional cell (fp4-over-XGMI combine, §5) | **UNMEASURED.** Design aspiration; the one cell the decoupled cost models do not cover. |
| One unified cost model across the 5 currencies | **DOES NOT EXIST, and the thesis must not claim it** (§6). |
| Traffic-shaping / schedule as the headline lever | **REJECTED.** 2.4× over the naive order but only 1–4% over hand-rolled round-robin; XGMI probe still issue-bound. Demoted to a substrate/guardrail. `DESIGN.md §7`, `MASTER_HANDOFF.md §0.5`. |
| IRIS `to_rank` must be constexpr `static_range(WORLD)` | **MEASURED GOTCHA** — a data-dependent `to_rank` faults; encoded in the `.from(remote(...))` lowering. |

### First milestone (unchanged from codex): **build QuantTile-v0 FIRST.**
One HK grouped decode GEMM body serving both fp8-sat and MXFP4 through a tile descriptor; fp8 within ~5%
of the standalone 3.97 TB/s (re-verified), MXFP4 keeping ~1.6×, **no per-format kernel fork.** Explicitly
do **not** start by reproducing the 386 µs combine, and do **not** start with Overlap-A (it needs a new
GEMM body first). QuantTile-v0 is the format-axis lowering of §4a, measurable against a trusted number,
and it is the proof that `stage(...).format(...)` is a real compiler dispatch and not a slide.
