# grouped_b0 — B0-class grouped MoE GEMM (the "fix the tile + schedule" kernel)

**Status:** code written on 2026-06-29, **NOT yet built/run on the node** (this machine has no GPU).
First on-device build + correctness + timing is the immediate next step.

## What this is
A drop-in-quality replacement for the b1_dispatch **phase-2** grouped GEMM. It computes the same
thing — `C[m,:] = A[m,:] · B[expert(m)·N + :, :]ᵀ` for every active packed row — but uses the
**proven B0-class 8-wave ping-pong schedule** instead of the slow V4-lineage producer/consumer body.

| | `micro_tk` (today, `reference/v5_grouped`) | **`grouped_b0` (this)** |
|---|---|---|
| Tile | 64×64×64 | **256×256×64** |
| Schedule | 4 producer + 4 consumer (permanent split) | **8-wave ping-pong (all 8 waves MFMA)** |
| Waves issuing MFMA | 4 / 8 | **8 / 8** |
| Body source | V3/V4 fused direct-pull | `reference/v2_hk_expert_gemm/fmoe_expert_v2.cu` (B0) |
| Measured | ~32–69 TFLOP/s | target ≈ B0's ~183 TFLOP/s (bf16 ceiling) |

## Design — a pure index remap, nothing else
`grouped_b0.cu::grouped_expert_gemm` is `expert_gemm_bf16` **byte-for-byte**, except the four
per-block tile-base indices, which now come from a per-task `(expert, m_tile, n_tile, expert_row_begin)`
tuple instead of a flat `blockIdx` decode. Because each expert's `expert_row_begin` is padded to a
multiple of `BM=256`, every offset is exact integer arithmetic:

```
a_row_tile = ERB/128 + mt*2          // A half-tiles are 128 rows   (was block_row*2)
b_row_tile = (e*N)/128 + nt*2        // B is expert-major [E*N,K]    (was block_col*2)
c_row_tile = ERB/64  + mt*4          // C reg-tiles are 64 rows      (was block_row*WARPS_ROW*2)
c_col_tile = nt*8                    // C reg-tiles are 32 cols      (was block_col*WARPS_COL*2)
```

The delicate part of the kernel — the prologue, the `#pragma unroll 2` steady state, the drain
epilogue, and every `s_waitcnt` / `s_setprio` / `s_barrier` / `sched_barrier` — is the verified
schedule, **untouched**. That is deliberate: the throughput lives in that schedule, and rewriting it
is exactly how P1/P2/P3 produced garbage (see `EXPERIMENT_LEDGER.md`).

**No-contamination invariant:** the host pads each expert's packed-row count up to a multiple of
`BM=256`, so a 256-row block always lies inside one expert. Padding rows are 0 in the packed buffer
(the phase-1 gather masks them), so they MFMA to 0 into dead C rows. Correctness comes from disjoint
padded regions + zeroed padding, not from per-row store predication.

## Scope (first cut, on purpose)
- **Phase-2 GEMM only.** A is assumed already gathered/packed **locally** (expert-major, BM-padded) —
  the buffer b1_dispatch phase-1 produces. The cross-GPU gather stays a separate kernel; the GEMM
  does no remote access. This is the "copy-once then local GEMM" dataflow B1 proved is strong.
- **Dequant fp8→bf16 preamble, then bf16 MMA** (identical numerics to B0/B1-copy). Native-FP8 MMA is
  a separate later track — HK's in-MMA scaled path is MX-e8m0/32 only, our scale is fp32/128
  (`V2_HK_ANALYSIS.md` §4).
- **Single GPU, self-contained.** No MPI / IRIS needed (the GEMM is local), so it builds and verifies
  with one `hipcc` line.

## Build + run (on the node, inside the HK checkout)
```bash
# single source, no MPI/IRIS — same toolchain as the V2 probe (V2_HK_ANALYSIS.md §1a)
/opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
    -I<HK_ROOT>/include -I/opt/rocm/include/hip grouped_b0.cu -o grouped_b0
./grouped_b0
```
It runs two cases and prints, for each, `TFLOP/s (real)` / `TFLOP/s (padded)` and an RMS-rel /
contamination correctness line:
1. **ragged-correctness** — `M_e = {40,256,130,300,512,7,256,99}`: exercises BM-padding, masking, and
   the no-cross-expert-contamination guard with a full CPU reference.
2. **perf-8×1024** — 8 experts × 1024 rows = 8192 packed rows, 256 tiles (fills the GPU). Its
   headline TFLOP/s is directly comparable to the `EXPERIMENT_LEDGER` grouped case.

Expected: RMS-rel ≈ 0.003–0.03 (bf16 round-trip), `padded/contaminated nonzero = 0`, and a
TFLOP/s **well above micro_tk's ~69**, approaching B0's ~183.

## What to verify on the node (highest-risk items first)
1. **It compiles.** The one unproven choice is the mixed/dynamic `gl<bf16,-1,-1,-1,-1>` with the
   `template<int NN,int KK>` schedule constants. If the dynamic gl + `prefill_swizzled_offsets` combo
   complains, the harness `b0_gemm` (all-dynamic gl, also 256×256 8-wave) is the proven fallback form.
2. **Correctness** (RMS-rel < 0.05, contamination 0) on the ragged case — this validates the index
   remap end-to-end. If RMS is ~1.0, suspect a tile-base index (most likely `b_row_tile` expert
   stride or a `c_*` store index).
3. **TFLOP/s** on perf-8×1024 vs the ledger's micro_tk number. This is the headline.

## Next steps (after it's green)
- **Sweep BM** ∈ {64, 128, 256}: decode's small per-expert M_e means 256 may lose to a tall-skinny
  tile (less padding waste). NB: the current body is hardwired to 256×256; a smaller BM needs a
  re-derived schedule (a real follow-up, not a constant change).
- **Wire into the b1_dispatch harness** for a same-process head-to-head: `T_gather` (phase-1) +
  grouped_b0 (phase-2) vs the current micro_tk phase-2, identical packed buffer.
- **Native-FP8 grouped GEMM** (separate track): removes the dequant-to-bf16 2× MMA-rate penalty vs
  AITER fmoe.

---

## Update 2026-06-30 — built, run, and verified on-device (8×/single MI350, gfx950)

The kernels below are now **built + correctness-verified + timed** on node B (mi355x-thor-4). The
"NOT yet built" status above is superseded.

### Decode (BM=16) GEMM family added to `grouped_b0.cu`
- `grouped_expert_gemm_decode` — bf16 BM=16 skinny decode tile (kills the BM=256 padding tax at low
  M_e). Decode case (E32, 512 real rows): **0.487 ms / 5.43 TB/s** B-stream.
- `grouped_expert_gemm_decode_fp8` — **LDS-staged NATIVE-fp8 MMA** (exp_12 double-buffer, the only
  HK-legal fp8 global path: `buffer_load_lds` → `ds_read_b128` → fp8×fp8 mma). Same decode case:
  **0.176 ms / 171 TFLOP/s ≈ aiter's 171.6**, RMS 0.0037. VGPR 60, occ 4. (Its effective B-stream
  exceeds the physical HBM ceiling → it benefits from L2 reuse of hot expert weights.)
- `grouped_expert_gemm_fp8` + `scale_c` — native-fp8 256×256 path (prefill).

### `sat_decode.cu` — SATURATING fp8 decode GEMM (new file)
Stores B fp8 (half the bytes) but PRE-SWIZZLES it offline (`PERM128`, round-trip verified) so it loads
through the fast half-width bf16 global→register path (no LDS, no barriers), unpacks fp8→bf16 in-register
with a per-128-K-block scale, then bf16×bf16 mma. **No LDS, no spill, VGPR 91, occ 5.** Decode case:
**0.333 ms / 3.97 TB/s, 1.46× over bf16**, RMS 0.0037 PASS. This is the conservative real-HBM floor
(no cache assumption); the ~27% gap to the bf16 ceiling is in-register unpack throughput.

Run: `SAT_VS_BF16=1 ./sat_decode` (carries its own bf16-ref head-to-head on the identical task list).

### Honest verdict (see `EXPERIMENT_LEDGER.md`)
Both fp8 decode kernels are genuine wins over bf16, but neither flips the verified COMPLETE-region
result (fused b1 is 2.7–2.8× slower than MORI+aiter): the GEMM is one of three lagging components and
aiter fuses the whole fc1+SiLU+fc2 FFN into one kernel. The fp8 GEMM is necessary but not sufficient.
