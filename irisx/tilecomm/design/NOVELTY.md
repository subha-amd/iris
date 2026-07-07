# NOVELTY.md — Adversarial web-search verification of the "empty quadrant" claim

> **Method.** Adversarial prior-art search (WebSearch/WebFetch, 2026-07-07). For each named
> system I read what it *actually* does and located the *precise* gap versus our claim. The
> discipline: assume the claim is scooped until proven otherwise; downgrade any wording that a
> reviewer could break. Numbers below are cited to URLs; where a fast-model page read is the
> only source I mark it "(page-read)".

## The claim under test

> *"A tile-level programming model where a tile's remote **RESIDENCY** and quantized
> **REPRESENTATION** are both first-class, and the runtime lowers it to device-side RMA (IRIS)
> + CDNA4 scaled-MFMA, preserving compression until the last responsible moment."*

Two axes must **both** be first-class **in one typed-tile abstraction**, and the compressed
bytes must be **the unit of transfer AND the unit of matmul** (dequant folded into the
scaled-MFMA), so compression survives HBM → XGMI → MFMA accumulate.

---

## What each prior work actually does, and the precise gap

### Overlap / comm-fusion line — owns RESIDENCY, has NO representation axis
- **FLUX** (arXiv [2406.06858](https://arxiv.org/abs/2406.06858), ByteDance,
  [github](https://github.com/bytedance/flux)): over-decomposes a GEMM + its collective into
  fine tiles and fuses them into one kernel for warp-level comm/compute interleave (≤1.38×
  train, >2× infer vs TransformerEngine). **Gap:** pure *overlap of dependent compute+comm*;
  data crosses the wire in its native dtype. No format/scale as a tile property, no
  compression-into-MFMA. NVIDIA-centric.
- **CoCoNet** (arXiv [2105.05720](https://arxiv.org/abs/2105.05720)): a DSL + scheduling
  language making collectives first-class so fusion/overlap transforms are expressible. **Gap:**
  first-class *collectives*, not first-class *quantized tiles*; the co-optimized object is the
  {compute, communication} schedule, never the {format, scale, layout} of the moving data.
- **Google/ASPLOS-23 "Overlap Communication with Dependent Computation via Decomposition"**
  (Wang et al., [dl.acm 3567955.3567959](https://dl.acm.org/doi/abs/10.1145/3567955.3567959)):
  the canonical decomposition-for-overlap paper. **Gap:** overlap only; no precision axis at all.
- **Triton-distributed** (arXiv [2504.19442](https://arxiv.org/pdf/2504.19442), ByteDance) +
  **PyTorch SymmetricMemory** / **IRIS** (arXiv [2511.12500](https://arxiv.org/html/2511.12500v1)):
  the *closest* thing to our residency axis — tile-based symmetric memory, per-tile readiness
  flags, single-source kernels that interleave compute+comm (30–40% over PyTorch+RCCL).
  **Gap (verified):** the IRIS paper has **zero** references to quantization/fp4/fp8/MX; it
  treats communicated data as **opaque bytes / standard dtypes** (page-read of the paper). It is
  *our substrate*, and it explicitly does **not** carry a representation. This is the strongest
  evidence the representation axis is genuinely unoccupied in the tile-RMA world.
- **MSCCL / MSCCL++** ([github](https://github.com/microsoft/mscclpp), arXiv
  [2504.09014](https://arxiv.org/html/2504.09014)): programmable collectives via a DSL, but at
  **buffer / threadblock** granularity with send/recv/reduce/copy instructions over opaque
  buffers. **Gap:** you program the *schedule*, not a typed tile; no quant-awareness, no
  fusion into a compute kernel's tile loop.
- **NVSHMEM / rocSHMEM / DeepEP** (device-initiated RMA;
  [DeepEP](https://github.com/deepseek-ai/DeepEP)): symmetric-heap one-sided put/get; DeepEP
  adds fp8/MXFP4 *dispatch*. **Gap:** a *transport substrate* (like IRIS), not a programming
  model; DeepEP's low-precision dispatch is a fixed collective, and compute-side it "preserves
  FP8 precision in compute-intensive GEMM" — dispatch precision and GEMM precision are managed
  as **separate** concerns, not one typed tile.

### Fused-quant-collective line — quantizes comm, but DEQUANTS before compute (or after reduce)
- **TensorRT-LLM fused AllReduce+RMSNorm+quant**
  ([allReduceFusionKernels.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/d6b741ddfe7f8a80718c10d49773c42abc0a254f/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu)):
  **verified from source** — the all-reduce **transmits full-precision** data; sum → residual
  → RMSNorm run in fp32/float4; **quantization is applied only to the OUTPUT, post-norm**, as a
  producer of the *next* layer's fp4/fp8 input. **Gap:** compression is a post-reduce
  *store* step, never carried *through* the collective or *into* the reduce/matmul.
- **EQuARX** (arXiv [2506.17615](https://arxiv.org/pdf/2506.17615), Google/XLA): a genuinely
  *quantized* all-reduce. **Gap (verified):** it **dequantizes back to full precision
  immediately after the reduction**; subsequent matmuls run on restored full precision. Collective
  implementation, not a tile programming model. So "quantized communication" alone is old — the
  novel part is *not* dequantizing before the matmul.
- **MORI / aiter** (LMSYS [MoRI blog](https://www.lmsys.org/blog/2026-05-28-mori/), MI355X):
  **FP4 dispatch + FP8 combine**, 2.56× round-trip BW cut — i.e. compressed bytes *do* cross the
  AMD MoE all-to-all in production **today**. **Gap (verified page-read):** MORI is a **fixed
  dispatch/combine collective library**, quant is **orthogonal config** (`SGLANG_MORI_DISPATCH_DTYPE`
  env var), and expert compute runs on **dequantized** intermediates. So *compression-over-the-AMD-
  MoE-wire is not novel*; what MORI does **not** do is make the format a tile-type property or keep
  it live *into* the scaled-MFMA.
- **FlashInfer** (arXiv [2501.01005](https://arxiv.org/pdf/2501.01005),
  [docs](https://docs.flashinfer.ai/)): a unified kernel *library* — fp4/fp8/MXFP4 MoE GEMM,
  grouped GEMM, all-to-all, and an AllReduce-fusion API, behind one interface. **Gap:** a menu of
  best-of-breed **separate** kernels selected behind an API, not a tile abstraction; quant and
  comm are composed, not co-lowered from one typed tile; NVIDIA tensor cores.

### Low-precision tile line — owns REPRESENTATION, has NO residency axis
- **Tilus** (arXiv [2504.12984](https://arxiv.org/pdf/2504.12984),
  [NVIDIA/tilus](https://github.com/NVIDIA/tilus)): "A Tile-Level GPGPU Programming Language for
  Low-Precision Computation" — the *closest* thing to our representation axis as a first-class
  programming model. **Gap (verified):** **single-GPU, zero** multi-GPU / RMA / collective /
  remote-memory concept. NVIDIA/PTX.
- **HipKittens** (arXiv [2511.08083](https://arxiv.org/html/2511.08083v1)): CDNA3/4 tile DSL with
  register-tiles → MFMA and the scaled-MFMA path we need. **Gap:** **no communication concept at
  all** (confirmed in our own source audit: `MASTER_HANDOFF §5`). It is the compute half of our
  substrate; the distributed half does not exist in HK.
- **LiquidGEMM / TileFuse / Multi-Scale Dequant** (arXiv
  [2509.01229](https://arxiv.org/html/2509.01229v1), [2606.11357](https://arxiv.org/pdf/2606.11357),
  [2605.13915](https://arxiv.org/pdf/2605.13915)): dequant fused into the MMA mainloop — i.e.
  "preserve compression until the last responsible moment" — but **single-GPU**. **Gap:** the
  "last responsible moment" is inside one GPU; there is no transport boundary to preserve across.

### ⚠️ The one that gets closest on the MECHANISM — DeepGEMM "Mega MoE"
- **DeepGEMM Mega MoE / DeepSeek-V4** (LMSYS
  [DeepSeek-V4 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)): **verified quote** — a
  kernel that "fuses EP dispatch, the first FP8xFP4 expert GEMM, SwiGLU, the second FP8xFP4 expert
  GEMM, and EP combine into a single **symmetric-memory-based mega-kernel**." Pairs MXFP8
  activations with MXFP4 weights on **Blackwell** ("relies on Blackwell-specific FP4 tensor-core
  machinery"). **This is the strongest threat: the *mechanism* — symmetric-memory residency + fp4
  tensor-core compute + fused MoE dispatch/combine in one kernel — already exists on NVIDIA.**
  **Gap (precise):** (1) it is a **hand-written monolithic mega-kernel**, *not a programming model
  / typed-tile abstraction* where residency and representation are first-class and a runtime
  co-lowers them; (2) **Blackwell-specific FP4 tensor cores + NVLink**, not IRIS device-RMA +
  CDNA4 scaled-MFMA (`mfma_scale_f32_16x16x128_f8f6f4`, per-32-block E8M0); (3) MoE dispatch/combine
  only — not one abstraction that *also* covers the TP-prefill all-reduce; (4) the blog does **not**
  state that FP4 activations stay compressed over the wire *into* the GEMM (it names MXFP8
  activations + MXFP4 *weights*), so even the "compression through the transport" specifics are
  unconfirmed there.

---

## The single strongest DEFENSIBLE novelty sentence

> **A typed-tile *programming model* in which a tile's remote residency and its quantized
> representation (format + block-scale + layout) are both first-class, so that a single
> declaration is co-lowered by the runtime to IRIS device-side RMA *and* a CDNA4 scaled-MFMA —
> keeping the compressed bytes as one-and-the-same unit of transfer and of matmul, so that
> compression is preserved from HBM across XGMI into the MFMA accumulate — a combination that on
> AMD exists in neither the tile-RMA world (IRIS/HipKittens carry no format) nor the low-precision
> tile world (Tilus/HipKittens carry no transport), and that elsewhere exists only as a
> hand-written Blackwell mega-kernel, never as an abstraction.**

Notice what this sentence deliberately does **not** claim: not "first to send fp4 over the wire"
(MORI/DeepEP), not "first quantized collective" (EQuARX), not "first fused compressed MoE kernel"
(DeepGEMM Mega MoE), not "first lazy-dequant-into-MMA" (LiquidGEMM). The load-bearing words are
**programming model / first-class / co-lowered / on AMD (IRIS+CDNA4)**.

## Top-3 NEAREST prior works we MUST explicitly distinguish

1. **DeepGEMM "Mega MoE" (DeepSeek-V4, Blackwell)** — the mechanism twin. Distinguish on:
   *abstraction vs monolithic hand-kernel*, *AMD IRIS-RMA + CDNA4 scaled-MFMA vs Blackwell/NVLink*,
   *general tile edge (MoE combine **and** TP all-reduce) vs a single MoE mega-kernel*.
2. **Tilus** — the representation-axis twin (first-class low-precision tile programming language),
   but **single-GPU, no transport**. Distinguish on: *residency as a first-class tile property +
   RMA lowering*.
3. **Triton-distributed / IRIS** — the residency-axis twin (first-class tile RMA + overlap), but
   data is **opaque standard dtypes, zero quant-awareness**. Distinguish on: *representation as a
   first-class tile property + scaled-MFMA lowering that keeps compression live through the wire*.
   *(Runner-up to name: **MORI** — fp4-dispatch/fp8-combine already in production on AMD, but a
   fixed collective with quant-as-env-config and dequant-before-GEMM.)*

## What THREATENS the novelty (flag honestly — do not paper over)

- **DeepGEMM Mega MoE reduces our claim to "abstraction + AMD port."** A reviewer can say the
  *mechanism* (symmetric-memory + fp4 tensor-core + fused MoE) is demonstrated on NVIDIA, so our
  contribution must stand or fall on the **programming-model** framing and the honest AMD
  measurements — not on inventing the mechanism. Lead with the abstraction; never headline "first
  to fuse compressed comm with a compressed GEMM."
- **Compression-over-the-wire is already production on AMD (MORI) and NVIDIA (DeepEP/DeepGEMM).**
  Any sentence implying we invented low-precision MoE communication is false and will be shot down.
  Our delta is narrow-but-real: *keeping the compressed form live INTO the scaled-MFMA as a typed-
  tile lowering, rather than dequant-before-GEMM (MORI) or dequant-after-reduce (EQuARX/TRT-LLM).*
- **"Just compose Tilus + Triton-distributed."** The obvious reviewer reflex. Rebuttal must be
  concrete: neither can *express* the crux — the format+block-scale+layout must **survive the RMA
  and arrive in exactly the per-32-K-block E8M0 layout the CDNA4 scaled-MFMA consumes** (IRIS =
  opaque bytes; Tilus = no transport). The research content is that cross-boundary contract, not
  the two endpoints.
- **"Last responsible moment" is a known idea** (LiquidGEMM/TileFuse lazy-dequant-in-mainloop).
  Novel only when the "moment" is pushed *across a transport boundary* — must be stated that way.
- **Substrate risk, not prior-art risk, but it caps the claim:** the IRIS RMA + scaled-MFMA
  co-lowering is, as of now, **unbuilt and unmeasured on our node** (the fused MoE ships bf16-MFMA
  after dequant; the tile-fused reduce-scatter is unproven — `MEASURED_FINDINGS §B`, `MASTER_HANDOFF
  §6`). Novelty of an *abstraction* is weak without a working lowering + a number. QuantTile-v0
  (one HK grouped GEMM body serving fp8-sat AND MXFP4 via a tile descriptor) is the falsifiable
  first proof that the representation axis is real; the residency+representation co-lowering is the
  paper's actual bar.
