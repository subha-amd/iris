# BENCHMARK METHODOLOGY — unified IRISX/HK MoE B-case harness (Agent 01)

The harness (`run_harness.py` + `harness_common.py` + `harness_kernels.cpp`) compares every
candidate on **identical tensors, layouts, precision, and correctness**, so a speedup number is
an apples-to-apples ratio and not an artifact of differing inputs. This file documents what each
B-case *is*, the timing methodology, the per_expert-vs-aggregate rule, and the B2 wiring gap.

## What is held identical across all comparable cases
- **Shapes** `M, N, K` from env; identical across B0..B5 in one invocation.
- **A activation** identical fp8 e4m3 bytes + identical per-128-group fp32 scales. Produced once by
  `harness_common.quantize_v1` (= `v3_fused_kernel/example.py::quantize_v1`, scale = amax/448).
- **B weights** identical bf16 `[N,K]`, same seed (777).
- **Output** identical bf16 `[M,N]`, `C = A · Bᵀ`.
- **Reference** bf16-of-dequant of the *true* A (`dequant_v1`), broadcast to the consumer rank by
  host MPI roundtrip — this builds the correct answer only, it never feeds a kernel.
- **Correctness** identical `harness_common.correctness`: RMS-rel `||C-ref||/||ref||`, tol 0.10,
  plus `c_zero` guard.
- **Zero-sentinel** the consumer rank's *local* A buffer is forced to zero. A non-zero, correct C
  therefore proves the activation came over IRIS from the remote rank (not a local fallback). For
  the no-comm case (B0) there is no remote read, so the field is `n/a`.

## The B-cases (exactly as implemented)
| Case | What it runs | Comm? | Where implemented |
|------|--------------|-------|-------------------|
| **B0** | dequant fp8→bf16 + local bf16 GEMM, **no comm** — compute ceiling | no | `harness_kernel.local_gemm` (`harness_kernels.cpp`) |
| **B1** | IRIS dispatch-pack-quant of A **exactly once** into a local fp8+scale buffer, then the **same** local GEMM as B0 — the **STRONG unfused IRISX baseline** | yes (once) | `harness_kernel.dispatch_pack_quant_once` + `local_gemm` |
| **B2** | production **MORI** dispatch/pack/quant + **AITER/CK** fmoe wrapper — production baseline | yes | **STUB** — see "B2 wiring gap" below |
| **B3** | V3 direct-pull, **no overlap** (`fused=0`) — historic weak baseline (re-gathers A per N-tile) | yes | existing `v3_fused_kernel` `tk_kernel.dispatch_micro` |
| **B4** | V4 A-stationary remote pull, **no overlap** (`fused=0`) — isolates A-reuse from overlap | yes | existing `v4_astationary_kernel` `tk_kernel.dispatch_micro` |
| **B5** | V4 A-stationary **+ overlap** (`fused=1`) — the fused candidate | yes | existing `v4_astationary_kernel` `tk_kernel.dispatch_micro` |

**Headline rule (decision gate G1):** any headline speedup of V4 (B5) must be cited against **B1**
(and B2 once wired), NOT against B3. B3 re-gathers A N/BN≈32× and is a known-weak baseline; quoting
1.8× vs B3 over-claims. B4 vs B5 isolates the *overlap* contribution from the *A-reuse* contribution.

### Loading the right kernel module for B3/B4/B5
B3 uses the V3 `tk_kernel`; B4/B5 use the V4 `tk_kernel`. Both candidate dirs build a module named
`tk_kernel`, so only one can be importable per process. The runner picks the variant via
`KERNEL_VARIANT={v3|v4}` and you run B3 in a V3 build context and B4/B5 in a V4 build context (see
AGENT_REPORT.md for the exact per-case commands). They are separate processes; results land in the
same CSV.

## Timing methodology
`harness_common.timed` does `WARMUP` unmeasured iters, then `ITERS` measured iters each bracketed by
`torch.cuda.synchronize()` (+ `iris.barrier()` for comm cases). It reports mean `lat_us` and
`p50/p95/p99` from the per-iter samples. TFLOPs = `2·M·N·K / lat`.

**Separate transfer/compute/combined timers.** B1 is the case where the split is meaningful and the
harness measures all three: the combined `run_fn` (transfer-once + compute) is the headline latency;
`phase_T` (IRIS gather of A once) and `phase_C` (local GEMM) are timed separately and written to the
`notes` column (`transfer=… compute=…`). B0 has only compute. B3/B4/B5 are single fused/loop kernels
whose internal transfer/compute overlap cannot be cleanly split at the Python level — they report a
single combined latency (the device-side split is a kernel-internal profiling mode, out of scope
here).

## per_expert vs aggregate — never silently mixed
`M_label` is an explicit enum (`harness_common.M_LABEL_PER_EXPERT` / `_AGGREGATE`) written into every
CSV row from the `M_LABEL` env var:
- **per_expert** — `M` is one expert's routed-row count `M_e`. Single-expert sweeps
  (`ROUTE=single`, `M ∈ {8,16,32,64,128,256,512,1024}`) use this.
- **aggregate** — `M` is the *total* routed rows across the packed buffer. Grouped 32-expert runs
  (`ROUTE=uniform|zipf|onehot|captured`, `M_TOTAL=…`) use this.

The runner never converts between them and never emits a row that mixes the two. Grouped per-expert
`M_e` distributions come from `harness_common.expert_row_counts` (uniform / Zipf / one-hot / captured).
The current harness benchmarks the **per-expert tile** at the given `M`/`M_e`; the full grouped
scheduler that fuses 32 variable-M experts into one launch is **Agent 02's** deliverable — this
harness provides the shared tensors/correctness/timing it will plug into, and records the route
distribution and `M_label` so grouped rows are unambiguous.

## Test shapes (driven by env, see AGENT_REPORT.md)
- single-expert: `M ∈ {8,16,32,64,128,256,512,1024}`, `K=7168`, `N ∈ {2048,4096}` (W13 gate/up);
  W2/down: `K=2048, N=7168`.
- grouped 32-expert: variable `M_e` via `ROUTE` (uniform / Zipf / one-hot / captured), `M_label=aggregate`.

## B2 wiring gap (production MORI + AITER/CK) — [NEEDS-NODE]
B2 is intentionally a **stub** that raises `NotImplementedError`. It cannot be called from this
harness without node-side wiring because the production dispatch/quant and fmoe live in libraries not
importable from the repo here:
- **Dispatch/pack/quant:** `mori` EP-dispatch (`mori.EpDispatch` / the ATOM dispatch op) producing
  the expert-major `packed_fp8 [local_e][src_rank][slot][H]` + `packed_sc [...][N_GROUPS]` layout
  (see `irisx/FMOE_LAYOUT.md`). The harness's B1 buffer layout is row-major `[M,K]` fp8 + `[M,K/128]`
  scales; the main agent must map MORI's expert-major packed layout onto the same logical A so the
  correctness reference still matches (same dequant).
- **Expert GEMM:** AITER `fmoe` / CK `fmoe_bf16_blockscaleFp8` invoked over that packed buffer.

**What the main agent must wire for B2** (then flip the stub to a real `make_case` branch):
1. `import` the MORI/ATOM dispatch entry point and the AITER/CK fmoe op available in `r1_c4`.
2. Feed them the SAME `A_fp8`/`A_sc`/`B` the harness built (or document the layout transform and
   apply it to both the op input and the reference identically).
3. Time dispatch (transfer) and fmoe (compute) with the same `timed()` helper; emit a B2 row.
4. Correctness via the same `correctness()` (RMS-rel, zero-sentinel where a remote dispatch applies).
Until then, B2 latency/TFLOPs are `[NEEDS-NODE]` and no `spd_vs_B2` can be filled.

## Static resource columns
VGPR/AGPR/SGPR/LDS/scratch are `[NEEDS-NODE]` in emitted rows — they come from compiling on the node
with `-Rpass-analysis=kernel-resource-usage` / `llvm-objdump` / `roc-obj` (subagent compile-only is
allowed; see AGENT_REPORT.md for the extraction commands).
