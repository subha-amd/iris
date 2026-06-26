# AGENT REPORT — P3 single-kernel copy-once + in-block overlap (Agent 12)

Candidate: `irisx/p3_singlekernel/` — module `tk_kernel`. Branch `agent/12-singlekernel-copyonce`.
Subagent wrote files only; the MAIN AGENT compiles, runs (under the GPU lock), and commits.

---

## 1. Exact build command (mirror to node, then build)

Mirror the candidate dir into the HK distributed-kernels checkout (build auto-discovers
`*/kernel.cpp`), keeping the node copy byte-identical to the repo copy:

```bash
# on <NODE>, inside the resident container, with the GPU HIDDEN for compile-only:
export HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
mkdir -p <HK_ROOT>/distributed-kernels/p3_singlekernel
cp irisx/p3_singlekernel/kernel.cpp   <HK_ROOT>/distributed-kernels/p3_singlekernel/kernel.cpp
cp irisx/p3_singlekernel/example.py   <HK_ROOT>/distributed-kernels/p3_singlekernel/example.py

cd <HK_ROOT>/distributed-kernels
flock /tmp/mi355x_compile.lock -c \
  "cmake -B build_12 -DDK_BUILD=p3_singlekernel -DCMAKE_HIP_ARCHITECTURES=gfx950 && \
   cmake --build build_12 -j8 --target p3_singlekernel"
```

Quick syntax-only check (no full build):
```bash
hipcc --offload-arch=gfx950 -std=c++20 -fsyntax-only \
  -I<HK_ROOT>/include -I<HK_ROOT>/ThunderKittens/include \
  -Iirisx/include irisx/p3_singlekernel/kernel.cpp
```

## 2. Proposed GPU test command (for the MAIN AGENT)

Single-expert, headline shape M1024/N2048/K7168 (np=2, rank0 = fp8 A source, rank1 = consumer):

```bash
flock /tmp/mi355x_project_gpu.lock -c '
  cd <HK_ROOT>/distributed-kernels/p3_singlekernel
  M=1024 N=2048 K=7168 ITERS=50 WARMUP=10 CSV=../../results/p3_results.csv \
  mpirun -np 2 --mca pml ob1 --mca btl self,vader python example.py
'
```

Sweep (single-expert) to map the copy-once/overlap regime and the grid-starvation risk:
```bash
for M in 256 512 1024; do for N in 2048 4096; do
  M=$M N=$N K=7168 ITERS=50 WARMUP=10 CSV=../../results/p3_results.csv \
  mpirun -np 2 --mca pml ob1 --mca btl self,vader python example.py
done; done
```

## 3. Expected output

```
==============================================================================
[P3-FUSED ] M=1024 K=7168 N=2048  max_abs=...  max_rel=...  RMS_rel=0.00331  local_A_zero=True  C_zero=False  -> PASSED
[SERIAL-B1] M=1024 K=7168 N=2048  ...                       RMS_rel=0.00331  local_A_zero=True  C_zero=False  -> PASSED
------------------------------------------------------------------------------
  SERIAL  (gather-all + local GEMM, ~B1) :   ~285 us/iter   ~103 TFLOP/s
  P3-FUSED (copy-once + in-block overlap):   <285 us/iter   >103 TFLOP/s   (target 165-220us)
  SPEEDUP (serial/fused)                 :   >1.0x   (P3 WINS)
==============================================================================
```

## 4. Correctness criteria (must ALL hold)

- `RMS_rel < 0.10` (expect ~0.00331, identical to B0/B1/V4) for BOTH P3-FUSED and SERIAL-B1.
- `local_A_zero = True` (consumer-rank local A is the zero sentinel).
- `C_zero = False` (C nonzero ⇒ the remote gather actually crossed XGMI).
- P3-FUSED and SERIAL-B1 produce the same C (both pass) — proves the overlap path didn't corrupt data.

## 5. Headline / gate rule

Per the ledger, the HONEST comparison is **vs B1** (NOT vs the in-module serial, which approximates
B1). The main agent should ALSO run the harness B1 (`dispatch_pack_quant_once` + `local_gemm`) at the
same shape and compare P3-FUSED T against the harness B1 T (~285–291µs). P3 is interesting only if
`T_P3 < T_B1`. Overlap hard ceiling = 1.90× over B1 at M1024/N2048.

## 6. Assumptions

- The harness `b0_gemm` MFMA layout (256×256×64, 8 warps, WARPS_COL=4×WARPS_ROW=2, single `cacc`,
  per-warp `{block_row*2+warp_m, k}` A indexing) is the measured-correct 183 TFLOP/s path. P3's MFMA
  core + store indexing are copied from it verbatim; only the A *source* is swapped. **The main agent
  should first confirm P3-FUSED matches harness B0/B1 numerics before trusting the timing.**
- `iris::iris_device_view::load<uint4>` does a single remote read (it does — see `iris.hpp:313`,
  `translate` + deref). Same gather primitive V3/V4 use.
- `iris.empty` has no int32/fp8 dtype → fp8 backed by a bf16 alloc with a `|u1` CAI view (mirrors
  v4 example). No int32 flag tensors are needed (single-kernel, no cross-block handshake).
- bf16 B weights, output bf16, per-128 fp32 scales, fp8 e4m3 OCP (NOT fnuz) — project constants.

## 7. Known risks (detail in P3_DESIGN.md §6)

- **A. Grid starvation (primary).** M-only grid = M/256 blocks (4 @ M1024) on 256 CUs. If T > B1,
  the fix is to split N across blocks (`N_PER_BLOCK` tunable, V4-style) trading some copy-once for
  occupancy. Measure M-only first (true copy-once), then raise block count if starved.
- **B. HBM A-cache round-trip** (panel-0 write + panels-1..n reads). Cheap at ~7 TB/s vs the 135µs
  remote gather, but if it dominates, cache fp8 (½ bytes) and dequant per panel.
- **C. Overlap depth.** Panel 0 carries the only remote-vs-MFMA overlap; large N amortizes via
  copy-once so even imperfect overlap beats B1's fully-serial copy-then-GEMM.
- **D. Static-only validation.** No GPU access for this subagent; the kernel is statically reasoned to
  mirror b0_gemm. First on-device run must verify correctness, not just speed.

## 8. Why this avoids P1/P2's failure modes

ONE kernel, ONE grid, ONE block-per-M-tile. No second kernel, no cross-stream, no inter-block flag,
no spin-wait. Overlap is in-block warp double-buffer (the V4/B5 mechanism the ledger proves works).
No cross-block publish ⇒ the P1/P2 inf-RMS visibility hazard cannot occur; no producer kernel ⇒ no
underfill/spin-waste serialization. (Table in P3_DESIGN.md §7.)

## 9. Files written (this candidate)

- `irisx/p3_singlekernel/kernel.cpp`   — P3 fused kernel + in-module serial baseline, module `tk_kernel`.
- `irisx/p3_singlekernel/example.py`   — np=2 driver: timing (one enclosing cuda event), RMS + sentinel, CSV.
- `irisx/p3_singlekernel/P3_DESIGN.md` — full design rationale, structure, risks, P1/P2-avoidance.
- `irisx/p3_singlekernel/AGENT_REPORT.md` — this file.

NOT touched: `v4_astationary_kernel/`, `harness/`, `v3_fused_kernel/`, `v2_*`, or any other agent dir.

## 10. Secret scan

No `hf_...` tokens, no real hostnames/paths, no IPs. Node specifics use placeholders
`<NODE>`/`<HK_ROOT>` only. (`NODE_ACCESS.local.md` is gitignored and was NOT copied into any committed
file.)

## 11. Static resource info

Not compiled by this subagent (no node access from here). Expected ≈ B0's resource profile (B0 is
`__launch_bounds__(512, 2)`, occ 2; P3 adds the gather helper's VGPR for the `uint4` remote load +
dequant temporaries — similar to V4's gather path). The main agent should capture the real
`-Rpass-analysis=kernel-resource-usage` VGPR/LDS/scratch when it compiles `build_12`.
```
