# QuantTile v0 — a format-carrying weight tile that lowers to ONE grouped-decode GEMM body

> **Status:** design (2026-07-07). This is the recommended **first** milestone of the tile-level
> abstraction, per the codex critique ("build QuantTile v0 FIRST … explicitly do NOT start by
> reproducing the 386 µs combine") and the measured finding that decode is a **weight-memory wall**
> where the *only* high-ceiling lever is fewer weight bytes (fp4). It is falsifiable against numbers
> we already trust.
>
> **Scope discipline (read first).** QuantTile v0 is a **weight-side, decode-path, in-register-dequant**
> tile. It unifies the **already-shipped and already-measured** `grouped_b0_gemm_decode_fp8_sat` and
> `grouped_b0_gemm_decode_mxfp4_sat` kernels (which today are a copy-paste fork) into **one
> descriptor-dispatched body**, and folds the plain bf16 decode GEMM in as the reference specialization.
> It does **not** ship a native fp4×fp4 scaled-MFMA path — that (Route-2) has two unresolved on-device
> unknowns (§7.2) and is v1. Everything claimed here is backed by a measured number or is flagged as
> unknown.

---

## 0. Why this, and why decode-weight-only first

Measured, same-node, correctness-gated (see `MASTER_HANDOFF.md` §4/§5, `MEASURED_FINDINGS.md` C):

- **Decode is ~80% weight floor + ~20% boundaries.** The expert GEMM streams all 32 local experts'
  ~1.4 GB fp8 weights every step (~176 µs HBM floor at 8 TB/s), independent of token count. Even if
  fusion made every boundary free, decode caps at ~1.25×. **The only way to move the 80% is to move
  fewer weight bytes.** That is a *representation* lever, not a comm/overlap lever.
- **Shipped evidence the lever works:**
  - fp8 `_sat` decode GEMM (store B as fp8 = ½ the bf16 bytes): **3.97 TB/s standalone, 1.46× over
    bf16**, RMS-vs-fp8-dequant **0.0037**.
  - MXFP4 `_sat` decode GEMM (store B as fp4 = ¼ the bf16 bytes): **~1.6× over the fp8 `_sat`
    decode**, ~2× over bf16, RMS-vs-fp4-dequant **0.0033**.
  - Fused decode region (8× MI350, TOTAL_M=512): MXFP4 **520.6 µs** vs fp8 578.9 vs bf16 881.3 →
    **1.11× over fp8, 1.69× over bf16** (GEMM-only 1.16×/1.96×; region diluted by the shared A-dequant
    + gather/act/combine boundaries).
- **The problem those two kernels expose:** they are **~90% identical** — same BM=16/BN=256/BK=128
  8-warp skeleton, same bf16 `mma_ABt`, same task loop, same A-load, same store — and differ **only**
  in (a) the packed-B register width, (b) the in-register unpack routine, (c) the scale layout, (d) the
  offline column permutation. HipKittens itself hard-codes a **separate GEMM per quant format**
  (`kernels/gemm/fp8fp32`, `kernels/gemm/mxfp8`). **A single tile that carries its format and lowers to
  one body is the genuine HK-paradigm contribution**, and the two near-identical decode kernels are the
  proof that the fork is accidental, not essential.

QuantTile v0 makes the fork **impossible by construction**: the format lives in a compile-time
descriptor; the GEMM body is written once.

---

## 1. Deliverable (1): the tile descriptor and a concrete C++/HK type sketch

### 1.1 What a QuantTile carries

A QuantTile is a **weight tile** (the B operand) tagged with everything the GEMM needs to consume it
without a per-format kernel:

| field | v0 values | meaning |
|---|---|---|
| `format` | `bf16`, `fp8e4m3`, `mxfp4`, `mxfp8` | storage element type of B in HBM |
| `bits` | 16, 8, 4, 8 | storage bits/element → sets the reinterpret-as-bf16 load width `K·bits/16` |
| `lower` | `dequant_bf16_mfma` \| `native_scaled_mfma` | which MFMA the body emits (§2.3) |
| `scale_layout` | `none`, `per_out_channel_f32`, `per_k32_block_e8m0` | how B's scales are laid out + applied |
| `preperm` | `nullptr`, `kDecSatPerm[128]`, `kDecSatPermFp4[128]` | offline column permutation that makes the reinterpret-as-bf16 load land in HK's register-fragment order |

Two facts from the real HK source pin the design:

1. **HK forbids fp8 (and any 1-byte) global→register loads** — a hard
   `static_assert(!std::is_same_v<…,fp8e4m3>, "Unsupported type for load")` in
   `include/cdna4/ops/warp/memory/tile/global_to_register.cuh:30` (row-major) and `:139` (col-major).
   The fast `buffer_load_b64/b128` path is bf16/float only. **This is by design.** Therefore a
   QuantTile's *load staging type is ALWAYS `rt_bf`* (never `rt_fp8e4m3`): compressed B is
   **reinterpreted** as a narrower bf16 tile and loaded through the fast path. The `preperm` exists
   precisely so that reinterpret is bit-correct (it inverts the byte→K-fragment scramble that the fast
   load imposes).
2. **The scaled MFMA is already in HK** (`mma_ABt_scaled` → `mfma1616128_scaled` →
   `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`, `mma.cuh:529/128`), consuming **E8M0, per-32-K**
   scales via `pack_scales`, with `cbsz/blgp` selecting the operand format (0=fp8e4m3, 4=fp4). This is
   the `native_scaled_mfma` lowering — but its scale format (E8M0) is lossy for non-MX checkpoints
   (§7.1) and its fp4 layout is unverified (§7.2), so v0 keeps it behind a flag.

### 1.2 C++/HK type sketch

```cpp
// ---- QuantTile descriptor: a compile-time policy the decode GEMM dispatches on. --------------------
namespace qt {

enum class format { bf16, fp8e4m3, mxfp4, mxfp8 };
enum class lower  { dequant_bf16_mfma,      // store compressed, unpack in-register, ONE bf16 mma_ABt
                    native_scaled_mfma };   // mma_ABt_scaled<cbsz,blgp> w/ E8M0 per-32 scales (v1)
enum class scale  { none, per_out_channel_f32, per_k32_block_e8m0 };

template <format F> struct traits;   // one specialization per format — the ONLY per-format code

// bf16 — the reference / correctness anchor (2× the fp8 bytes; never a serving target)
template <> struct traits<format::bf16> {
    static constexpr int   bits   = 16;
    static constexpr lower low     = lower::dequant_bf16_mfma;
    static constexpr scale slay    = scale::none;
    static constexpr const int* preperm = nullptr;
    // staging tile == compute tile; unpack is identity
    template <int RN, int BK> using RT_PACK = rt_bf<RN, BK,   row_l, rt_16x32_s>;
    using scale_t = float;   // unused
};

// fp8e4m3 — DeepSeek/aiter native: per-output-channel fp32 scale, applied in float during unpack.
// (Deliberately NOT E8M0 — see §7.1. This is the SAFE, checkpoint-agnostic route.)
template <> struct traits<format::fp8e4m3> {
    static constexpr int   bits   = 8;
    static constexpr lower low     = lower::dequant_bf16_mfma;
    static constexpr scale slay    = scale::per_out_channel_f32;   // sB : [E*N] e-major
    static constexpr const int* preperm = kDecSatPerm;             // 128-wide column perm (mma.cuh order)
    template <int RN, int BK> using RT_PACK = rt_bf<RN, BK/2, row_l, rt_16x32_s>;  // K/2 bf16 = K fp8
    using scale_t = float;
};

// mxfp4 — OCP MXFP4: per-32-K-block E8M0 (decoded to f32 host-side), folded by the gfx950 HW cvt.
template <> struct traits<format::mxfp4> {
    static constexpr int   bits   = 4;
    static constexpr lower low     = lower::dequant_bf16_mfma;
    static constexpr scale slay    = scale::per_k32_block_e8m0;    // sB : [E*N, K/32]
    static constexpr const int* preperm = kDecSatPermFp4;          // algebraic compose of kDecSatPerm
    template <int RN, int BK> using RT_PACK = rt_bf<RN, BK/4, row_l, rt_16x32_s>;  // K/4 bf16 = K fp4
    using scale_t = float;
};

// mxfp8 — native scaled MFMA (E8M0 per-32). v1 vehicle for prefill; the mma_ABt_scaled lowering.
template <> struct traits<format::mxfp8> {
    static constexpr int   bits   = 8;
    static constexpr lower low     = lower::native_scaled_mfma;
    static constexpr scale slay    = scale::per_k32_block_e8m0;
    static constexpr const int* preperm = nullptr;                 // LDS-staged (mxfp8 8-wave swizzle)
    template <int RN, int BK> using RT_PACK = rt_fp8e4m3<RN, BK>;  // native fp8 register operand
    using scale_t = fp8e8m0;
};

// ---- The per-format unpack: the ONLY thing that varies inside dequant_bf16_mfma. ------------------
// Given this warp's compressed strip staged as bf16 (`bpk`), fill the 32x128 bf16 compute tile `b`,
// folding B's scale in. `sB` + `so` (scale offsets) are format-specific; `b` is the SAME type for all.
template <format F, class RT_B, class RT_PACK, class SB>
__device__ __forceinline__ void unpack(RT_B& b, const RT_PACK& bpk, const SB& sB, /*offsets*/ ...);
// fp8e4m3 : convertor<float4,fp8e4m3_4> * sc(per-N-row)                 [kernel.cpp:1310-1325]
// mxfp4   : __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(reg, sc, SEL)     [kernel.cpp:1432-1493]
// bf16    : memcpy (identity)

} // namespace qt
```

The descriptor is **pure compile-time**; there is no runtime cost to carrying it. `format` is chosen by
a single host-side dispatch (§2.4) that replaces today's `DECODE_SAT` / `DECODE_MXFP4` env fork.

---

## 2. Deliverable (2): lowering to ONE grouped-decode-GEMM body

### 2.1 The shared skeleton (this is already what both shipped kernels are)

Both `grouped_b0_gemm_decode_fp8_sat` (kernel.cpp:1272) and `grouped_b0_gemm_decode_mxfp4_sat`
(kernel.cpp:1438) are the identical loop below. The unified body is this, parameterized by
`qt::traits<F>`:

```cpp
template <int NN, int KK, qt::format F>
__global__ __launch_bounds__(512, 2)
void grouped_b0_gemm_decode_qt(
        const gl<bf16, -1,-1,-1,-1> A,     // [Mpacked, K]   bf16 activations (A stays bf16 in v0)
        const gl<bf16, -1,-1,-1,-1> Bpk,   // [E*N, K*bits/16] PRE-SWIZZLED compressed B reinterpreted as bf16
        const gl<bf16, -1,-1,-1,-1> C,     // [Mpacked, N]   final scaled output
        const typename qt::traits<F>::scale_t* __restrict__ sB,
        const int* __restrict__ tasks, int num_tasks) {
    using QT = qt::traits<F>;
    constexpr int BM_DEC=16, BN_DEC=256, BLOCK_K=128, WARPS_COL=8, REG_BLOCK_N=BN_DEC/WARPS_COL; // 32
    constexpr int k_iters = KK / BLOCK_K;

    using RT_A    = rt_bf<BM_DEC,      BLOCK_K, row_l, rt_16x32_s>;              // 16x128 bf16 A
    using RT_PACK = typename QT::template RT_PACK<REG_BLOCK_N, BLOCK_K>;        // 32 x (128*bits/16) bf16
    using RT_B    = rt_bf<REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s>;             // 32x128 bf16 (unpacked)
    using RT_C    = rt_fl<BM_DEC, REG_BLOCK_N, col_l, rt_16x16_s>;              // 16x32 fp32 acc
    RT_A a; RT_PACK bpk; RT_B b; RT_C c;

    /* ---- task decode: e, mt, nt, ERB from tasks[task] — IDENTICAL for every format ---- */
    /* ---- scale-offset setup from QT::slay (per-N-row scalar OR per-32-block base) ---- */

    zero(c);
    #pragma unroll 4
    for (int k = 0; k < k_iters; k++) {
        kittens::load<2>(a,   A,   {0,0, a_row16, k});   // 16x128 bf16 A (broadcast) — IDENTICAL
        kittens::load<2>(bpk, Bpk, {0,0, b_row32, k});   // this warp's compressed strip as bf16
        qt::unpack<F>(b, bpk, sB, /*offsets, k*/ ...);   // <-- the ONLY format-specific line
        mma_ABt(c, a, b, c);                             // ONE bf16 MFMA (16x16x32) — IDENTICAL
    }
    store(C, c, {0,0, c_row16, c_col32});                // IDENTICAL
}
```

`mma_ABt` here is exactly the HK op in `mma.cuh:471`; `kittens::load<2>` is the fast bf16 buffer-load
in `global_to_register.cuh:24`. **Nothing in the pipeline, the barriers, the accumulator, or the store
changes between fp8 and mxfp4** — only `RT_PACK`'s width and the one `qt::unpack` call.

### 2.2 In-register dequant lowering (`dequant_bf16_mfma`) — the v0 path

This is the shipped `_sat` trick, generalized. For each format the descriptor supplies:

- **width:** `RT_PACK = rt_bf<32, 128·bits/16>` — fp8 → `rt_bf<32,64>`, mxfp4 → `rt_bf<32,32>`, bf16 →
  `rt_bf<32,128>`. The `Bpk` gl view's inner dim is `K·bits/16`.
- **preperm:** an offline column permutation applied **once** to B so the reinterpret-as-bf16 fast load
  delivers bytes in the fragment order the unpack expects. fp8 uses `kDecSatPerm[128]` (kernel.cpp:1252,
  round-trip verified to reconstruct identity for all 128 K columns); mxfp4 uses `kDecSatPermFp4` (the
  algebraic compose of `kDecSatPerm`). **v0 normalizes where this runs** (§2.4): today fp8 pre-swizzles
  on-device (pointer-cached, kernel.cpp:1366) while mxfp4 arrives pre-swizzled from the host — an
  accidental inconsistency the descriptor removes.
- **unpack:** fp8 → `convertor<float4,fp8e4m3_4>` then multiply by the per-N-row fp32 scale
  (kernel.cpp:1317-1322); mxfp4 → `__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(reg, sc, SEL)` with the
  per-32-block scale folded in (kernel.cpp:1434) — the gfx950 **hardware** cvt (the software
  `float4(fp4e2m1_4)` path is 4× slower and is the key perf lever); bf16 → identity `memcpy`.

**Precision posture of this lowering (honest):** A is bf16 (dequantized by `dequant_packed_mtile`
before the GEMM), and B's scale is applied **in float, outside the MFMA**. This sidesteps the E8M0
constraint entirely (§7.1) — the fp8 route uses the model's true fp32 per-channel scale, the mxfp4
route uses the true per-32 E8M0 magnitude decoded to f32. That is *why* the `_sat` fp8 route hits RMS
0.0037 (vs 0.073 for the native per-row-requant path, DECODE_SAT=0) — no scale coarsening.

### 2.3 Native scaled-MFMA lowering (`native_scaled_mfma`) — declared, v1

The descriptor's second lowering emits `mma_ABt_scaled<cbsz,blgp>` (mma.cuh:529) with `pack_scales`
(mma.cuh:253) feeding E8M0 per-32-K scales — exactly the `kernels/gemm/mxfp8/MXFP8_8wave` body
(421 TFLOPS stock, passes correctness). Here **both A and B are compressed** and scales are applied
*inside* the accumulate (no float fold, no requant). `cbsz=blgp=0` → fp8, `=4` → fp4.

The unified caller declares this lowering in the descriptor; the body selects it with
`if constexpr (QT::low == lower::native_scaled_mfma)`. The task loop, expert grouping, A-side handling,
and store are shared with 2.2; the inner load/mma differ (LDS-staged `st_fp8e4m3` + `load_st_to_rt` +
`mma_ABt_scaled` vs reinterpret-bf16 register load + `mma_ABt`). **This is where the "one descriptor,
two lowerings" claim is weaker than "one body"** — honestly, across lowerings it is one *declaration*
that picks one of two inner-loop templates, not one literal loop. Within the v0 `dequant_bf16_mfma`
lowering, fp8 + mxfp4 (+ bf16) genuinely share one loop. v0 ships only that.

### 2.4 Host dispatch — replaces the env fork

```cpp
void dispatch_grouped_gemm_b0_decode_qt(b0_dec_qt_globals g) {   // one entry, one enum
    switch (g.format) {                                          // was: if(DECODE_SAT)…elif(DECODE_MXFP4)…
      case qt::format::fp8e4m3: launch<qt::format::fp8e4m3>(g); break;
      case qt::format::mxfp4:   launch<qt::format::mxfp4>(g);   break;
      case qt::format::bf16:    launch<qt::format::bf16>(g);    break;
      case qt::format::mxfp8:   launch<qt::format::mxfp8>(g);   break;   // native_scaled_mfma (v1)
    }
}
```

`launch<F>` does the three shared host steps (all already present, kernel.cpp:1346/1511): (1) dequant
packed-fp8 A → bf16 via `dequant_packed_mtile` (task-driven, skips the ~94% padding rows); (2) apply
`qt::traits<F>::preperm` to B **once** (pointer-cached associative slot cache, unified across formats);
(3) launch `grouped_b0_gemm_decode_qt<N,K,F>` for the supported (N,K) ∈ {(2048,7168),(4096,7168),
(7168,2048)}.

---

## 3. Deliverable (3): the falsifiable v0 milestone + gates

**Milestone.** One templated `grouped_b0_gemm_decode_qt<N,K,F>` body + one `qt::traits<F>` descriptor
set replaces `grouped_b0_gemm_decode_fp8_sat`, `grouped_b0_gemm_decode_mxfp4_sat`, and (as the
reference specialization) `grouped_b0_gemm_decode`. **No `#ifdef`/copy-paste per-format kernel remains.**
It is falsified if any gate below misses.

Run **standalone** (the trusted GEMM harness, same-node, rotating buffers, best-of-N CUDA-event) at the
decode shape N=4096,K=7168, BM=16, at the measured `M_e` distribution, plus the **in-region** decode
gate (TOTAL_M=512, 8× MI350, same node, MAX-over-ranks).

| # | gate | threshold | anchor (measured) |
|---|---|---|---|
| **G1** | **fp8 no-regression** — unified body @ format=fp8e4m3 vs the trusted standalone fp8 `_sat` | within **±5%** of **3.97 TB/s** (i.e. ≥ 3.77 TB/s) | 3.97 TB/s (`_sat` standalone) |
| **G2** | **mxfp4 win preserved** — unified body @ format=mxfp4 GEMM time vs @ format=fp8e4m3, same shape | mxfp4 ≤ fp8_time / **1.5** (≥1.5× win; 5% slack on the measured 1.6×) | 1.6× (mxfp4 over fp8 `_sat`) |
| **G3a** | **correctness vs format-faithful dequant** — unified body per format vs a CPU/torch reference that dequantizes B in that exact format | fp8 RMS ≤ **5e-3**; mxfp4 RMS ≤ **5e-3** | fp8 0.0037, mxfp4 0.0033 |
| **G3b** | **unification is bit-stable** — unified body output vs the pre-unification per-format kernel output, same inputs | max\|Δ\| ≤ fp-reassoc noise (bf16 ULP-level; effectively RMS ≤ 1e-4) | — (regression guard) |
| **G4** | **region still wins** — fused decode region (TOTAL_M=512) with the unified body @ mxfp4 vs @ fp8 | ≥ **1.10×** (matches shipped) | 520.6 vs 578.9 µs = 1.11× |

- **G1** is the load-bearing gate: it proves the descriptor abstraction is **zero-cost** — a `constexpr`
  policy, not a dynamic dispatch tax. If G1 fails, the templating introduced overhead (spills, lost
  scheduling) and the unification is not free.
- **G2** proves the abstraction did not quietly disable the fp4 byte-saving (e.g. by forcing a common
  code path that widens the fp4 load). The win must survive unification.
- **G3a** is the "am I computing the format's intended math" gate (tight, ~5e-3). **G3b** is the
  stronger "did unification change any bit" gate. **Note:** G3a is NOT the precision-class number
  (fp8 vs true-bf16 = 0.057; fp4 = 0.117) — those are the inherent quant errors and are *expected*;
  G3 checks the GEMM faithfully realizes the chosen format, not that the format is lossless.
- **Explicitly out of scope for v0 (so the milestone stays falsifiable):** no native fp4×fp4 MFMA
  (§7.2), no A-side quant (A stays bf16), no prefill, no combine/gather changes, no new external
  baseline (§6 — the a4w4 comparison is untrusted; do not headline any number against it).

---

## 4. What is genuinely new here (novelty, honest)

- **HK today hard-codes one GEMM per format** (`fp8fp32`, `mxfp8`) and the MoE fork hard-codes
  `_fp8_sat` vs `_mxfp4_sat`. A **format-carrying tile that lowers to one body** is the HK-paradigm
  contribution — it turns "write a kernel per quant format" into "declare a tile's format."
- The **empty quadrant** (per the codex read) is *representation as a first-class tile property*,
  lowering to CDNA4 scaled-MFMA / in-register dequant, "preserving compression until the last
  responsible moment." Overlap-based fusion (Flux/CoCoNet/TRT-LLM AR+RMSNorm+quant) is taken; this axis
  is not.
- It is the **first lowering of the `stage`/`retire` typed-tile notation** the abstraction proposes:
  QuantTile is the `format` half; the tile-fused reduce-scatter (abstraction A, prefill) is the
  `transport` half. v0 proves the format half against the decode weight wall — the bottleneck that
  actually sets R1 serving latency.

**What is NOT novel / must not be oversold:** the *speedups* (3.97 TB/s, 1.6×) already exist in the two
shipped kernels; v0 does not make decode faster, it makes the two fast kernels **one** kernel at
**zero cost** (G1). The contribution is the abstraction + the demonstration that it is free, not a new
perf number.

---

## 5. Where the code lives / the refactor surface

| today | becomes |
|---|---|
| `grouped_b0_gemm_decode_fp8_sat<N,K>` (kernel.cpp:1272) | `grouped_b0_gemm_decode_qt<N,K,fp8e4m3>` |
| `grouped_b0_gemm_decode_mxfp4_sat<N,K>` (kernel.cpp:1438) | `grouped_b0_gemm_decode_qt<N,K,mxfp4>` |
| `grouped_b0_gemm_decode<N,K>` (kernel.cpp:1014) | `grouped_b0_gemm_decode_qt<N,K,bf16>` (reference) |
| fp8 unpack inner loop (kernel.cpp:1310-1325) | `qt::unpack<fp8e4m3>` |
| mxfp4 unpack inner loop (kernel.cpp:1478-1492) | `qt::unpack<mxfp4>` |
| `kDecSatPerm` (kernel.cpp:1252) / `kDecSatPermFp4` (host) | `qt::traits<F>::preperm` (unified: on-device, pointer-cached, for both) |
| `dispatch_grouped_gemm_b0_decode_fp8_sat_impl` + `..._mxfp4` + `DECODE_SAT`/`DECODE_MXFP4` env | one `dispatch_grouped_gemm_b0_decode_qt` switching on `format` |

The `native_scaled_mfma` lowering reuses `kernels/gemm/mxfp8/MXFP8_8wave` (`mma_ABt_scaled` +
`pack_scales`) verbatim as a second inner-loop template — no new MFMA path to write.

---

## 6. Baseline discipline (do not re-earn a fake win)

v0's gates are **self-referential on purpose** (G1/G2/G3b compare the unified body to the *trusted*
per-format kernels we already measured same-node). Do **not** gate v0 against the captured aiter `a4w4`
region — it is untuned + buggy for our EP shape (MASTER_HANDOFF §10.1: every `block_size_M` sweep falls
back to `2stage default`; ROCm/aiter #3632/#2343). Any external-baseline claim waits for the
reproducible SGLang DeepSeek-R1-FP4 stack or Simran's e2e run (MASTER_HANDOFF §10 ⚠️). v0 proves
**abstraction correctness + zero cost**, not a new external speedup.

---

## 7. Honest risks

### 7.1 The scale-format mismatch: aiter fp32-per-128 vs HK E8M0-per-32 (the correctness trap)

- **The facts.** DeepSeek/aiter fp8 checkpoints carry **per-128-K, fp32** scales. The CDNA4 scaled
  MFMA (`mma_ABt_scaled`, mma.cuh:529) and HK's `pack_scales` consume **per-32-K, E8M0** (8-bit
  exponent, **power-of-two only, no mantissa**) scales.
- **What converts and what doesn't.** per-128 → per-32 is **lossless** (replicate the coarse scale into
  4 fine blocks). fp32 → E8M0 is **lossy** — a fp32 scale of, say, 1.3 must snap to a power of two
  (1.0 or 2.0). It is only lossless if the checkpoint was **genuinely MX-quantized** (scales already
  powers of two), e.g. `amd/DeepSeek-R1-MXFP4`.
- **The consequence for v0.** The **`native_scaled_mfma` lowering is only correct for MX checkpoints.**
  Feeding a DeepSeek fp8-per-128-fp32 scale into E8M0 corrupts the result. **This is exactly why v0's
  fp8 path uses `dequant_bf16_mfma` with a `per_out_channel_f32` scale applied in float** — it consumes
  the model's true fp32 scale and never touches E8M0. So v0 is **checkpoint-agnostic for fp8** and
  **MX-correct for mxfp4** by construction. The risk is real but *designed around*: the descriptor's
  `scale_layout` field is what encodes "this format's scales are fp32-per-channel, do not E8M0 them."
  The trap only bites if someone later routes an fp8-per-128 checkpoint through `native_scaled_mfma`
  — G3a would catch it, but the descriptor should also `static_assert` the (format, scale_layout,
  lower) triple is a known-good combination.
- **Unknown:** whether the production R1 fp8 checkpoint Simran targets is per-128-fp32 (aiter/DeepSeek
  native) or already MX. v0 does not depend on the answer (it uses the fp32-fold route); v1's native
  path does. **Flag for Simran.**

### 7.2 The fp4 `cbsz=4` unknowns — why native fp4×fp4 is v1, not v0 (MASTER_HANDOFF §10.1, Route-2)

Route-2 (a true fp4×fp4 `mma_ABt_scaled<cbsz=4,blgp=4>`, the high-throughput **prefill** fp4 path) has
two unresolved, on-device unknowns. The `cbsz=4` variant **compiles clean** on the existing
`rt_fp8e4m3` tiles, but two things are undocumented and unverified:

1. **fp4 operand byte→K layout.** With `cbsz=4` the MFMA's K-reach doubles from 128 (fp8) to **256**
   (fp4 packs 2 nibbles/byte). The exact in-register byte→K-index mapping the intrinsic expects is
   undocumented — the same kind of scramble the fp8 `_sat` swizzle (`kDecSatPerm`) had to be reverse-
   engineered for. Until it is lane-by-lane round-trip verified (as `kDecSatPerm` was), a native fp4
   MFMA can silently transpose K.
2. **Scale granularity per MFMA.** `pack_scales` (mma.cuh:253) packs **4 E8M0** for a K=128 fp8 tile
   (one per 32). fp4's K=256 needs **8 per-32 blocks**. Whether the `opsel`/scale encoding accepts 8
   per-32 scales, or only 4 (⇒ per-64, **lossy vs MXFP4's per-32**), is unverified. A per-64 fallback
   would break MXFP4 numerics.

**Why this does not block v0:** v0's mxfp4 uses the `dequant_bf16_mfma` lowering — it unpacks fp4→bf16
in-register with the **hardware `cvt_scalef32_pk_bf16_fp4`** (per-32 E8M0 folded, already verified,
RMS 0.0033) and runs the **standard bf16 `mma_ABt`** (K=128, no cbsz). It never touches the fp4-native
MFMA, so neither unknown applies. Route-2's value is **prefill** (decode is weight-bound where the
in-register dequant already wins); v0 is decode-only. **The two unknowns are v1 gating items, resolved
by a standalone `mxfp8`-body probe with cbsz flipped to 4, verified against a per-32 fp4 CPU reference —
NOT on the critical path for the v0 milestone.**

### 7.3 Other honest caveats

- **A stays bf16 in v0.** This is weight-only quant (W4A16 / W8A16-ish). It is the *correct* decode
  choice (literature + our measurement: 4-bit **activation** quant gives ~no decode speedup because
  decode is weight-bound), but it means v0 is **not** the model's true W4A4/W8A8 math — it is *more*
  accurate than the model, and a true A-quant (native scaled MFMA) is a separate v1 item.
- **The G1 3.97 TB/s anchor is a same-node standalone number**; the region dilutes it (shared A-dequant
  + boundaries). Gate G1 on the standalone GEMM, G4 on the region — do not mix denominators.
- **Zero-cost is a hypothesis, not a guarantee.** Templating a body that was hand-scheduled per format
  can perturb the compiler's instruction schedule (the fp8 body relies on a specific
  `sched_barrier(0)`/`s_waitcnt` placement; kernel.cpp notes the compiler was hoisting the mma above
  the data wait). If G1 fails, the fix is likely to keep the schedule identical and only vary the
  `qt::unpack` call — which is exactly how the two kernels already differ, so the risk is low but real.
```
