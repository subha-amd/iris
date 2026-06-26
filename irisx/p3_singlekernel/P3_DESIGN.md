# P3 — Single-Kernel Copy-Once + In-Block Overlap MoE Expert-GEMM

**Candidate dir:** `irisx/p3_singlekernel/` (module `tk_kernel`)
**Built on:** B0 (`harness/harness_kernels.cpp::b0_gemm`, the measured 183 TFLOP/s compute path) +
V4's in-block producer/consumer double-buffer overlap mechanism (`v4_astationary_kernel`).
**Does NOT edit** v4/harness/v3/v2 in place.

---

## 1. Why P3 (after P1/P2 failed)

P1/P2 used a **two-kernel cross-stream producer/consumer flag handshake**. Both failed the same way
(ledger): RMS=inf (consumer GEMM read inbox tiles never correctly *published* for its rows) and
15–100× slower than B1 (producer kernel had few resident blocks while consumer blocks spin-waited →
no real overlap, huge spin waste). The flaw is the architecture, not a local bug.

P3 removes the architecture entirely:
- **ONE kernel, ONE launch, ONE grid.** No second kernel, no cross-stream, no inter-block flag, no
  spin-wait. The only synchronization is `s_barrier` between warps **of the same block** sharing
  LDS — the exact mechanism V4/B5 already uses and that the ledger proves works (B3→B5 = 1.82×, "ALL
  overlap"). No cross-block visibility is relied upon, so the inf-RMS publish hazard *cannot* occur.

## 2. The two levers P3 fixes vs V4/B5 (which ran at only 44 TFLOP/s)

### (1) COPY-ONCE — each remote A element crosses XGMI exactly once
- **Grid is 1-D over M-tiles only** (`M / BM` blocks, BM=256). Each block owns one BM row-strip and
  **all N columns**, so no two blocks ever gather the same A. (V4 gathered each A tile once per
  N-subtile-group → A crossed XGMI 4–8×.)
- Within a block, the **outer loop walks N-panels** (BN=256 each). On **panel 0**, each A K-tile is
  gathered from **remote** once and **cached (already dequantized to bf16) into a block-private
  LOCAL-HBM A-strip**. **Panels 1..n-1** read A from that local HBM strip (~7 TB/s) and **never
  re-cross XGMI**. ⇒ A traffic = B1-class (each A element crosses once).

### (2) NEAR-B0 COMPUTE — all 8 warps MFMA, no permanent producer warps
- **No producer/consumer warp split.** (V4 burned 4 of 8 warps as permanent gather warps → half the
  MFMA throughput.) All 8 warps run **B0's exact 256×256×64 ping-pong MFMA**, and all 8 warps also
  cooperatively issue the remote gather.
- The A K-tile for `k+1` is **prefetched while MFMA-ing `k`** (in-block double-buffer, `tic^=1`), so
  the cross-GPU latency of A[k+1] is hidden under the MFMA of A[k]. The MFMA math/occupancy is
  byte-for-byte B0; only the *source* of the A LDS tile differs (remote on panel 0, local HBM after).

This is the prompt's **KEY SIMPLIFICATION**: *"B0's GEMM, but the A it reads is gathered from remote
into LDS one K-tile ahead (double-buffered) instead of read from local HBM."*

## 3. Kernel structure (`p3_gemm`)

```
grid = (M/256) blocks, 8 warps/block (512 threads), launch_bounds(512, 2)   // == B0
LDS: As[2] (st_bf<128,64>), Bs[2] (st_bf<128,64>)     // double-buffered A + B, == B0
local-HBM A-cache: [grid_blocks][256][K] bf16, block-private (alloc once, cached in dispatch)

for np in 0..n_panels-1:                 // outer: N-panels (copy-once boundary)
    panel0 = (np == 0)
    zero(cacc)
    // prologue: fetch K-tile 0 into stage 0
    if panel0: gather_dequant_A_half(As[0], k=0, warp_m)      // REMOTE, all 8 warps
    else:      load_A_half_from_cache(As[0], k=0, warp_m)     // LOCAL HBM
    P3G::load(Bs[0], k=0)                                     // LOCAL HBM
    s_barrier; s_waitcnt
    if panel0: store_A_half_to_cache(As[0], k=0)             // persist for later panels

    for k in 0..k_iters-1, tic^=1:
        nxt = tic^1
        // PREFETCH A[k+1] + B[k+1] into the OTHER stage (overlaps this iter's MFMA)
        if k+1 < k_iters:
            if panel0: gather_dequant_A_half(As[nxt], k+1, warp_m)   // REMOTE, hidden under MFMA
            else:      load_A_half_from_cache(As[nxt], k+1, warp_m)
            P3G::load(Bs[nxt], k+1)
        // MFMA the CURRENT stage  (== B0)
        load(a, subtile(As[tic])); load(b0, subtile(Bs[tic])); s_waitcnt
        mma_ABt(cacc, a, b0, cacc)
        s_waitcnt; s_barrier
        if panel0 and k+1<k_iters: store_A_half_to_cache(As[nxt], k+1)  // cache the prefetched tile
    store(C, cacc, ...)                  // == B0 store indexing
```

`gather_dequant_A_half` is V4's proven remote fp8 gather + per-128 dequant (vectorized `uint4`
IRIS `ctx.load`), writing the **B0 swizzle** (`P3_ST_A::swizzle`) so the MFMA `load` consumes it
identically to B0. `load_A_half_from_cache` / `store_A_half_to_cache` mirror that swizzle for the
local-HBM round-trip.

## 4. In-module SERIAL baseline (`fused=0`)

Reproduces **B1-copy** inside the module for a self-contained comparison: `p3_gather_all` gathers +
dequants the whole A strip to local HBM (NO overlap), then `p3_gemm_local` runs the pure-local
B0 GEMM over the cache. The **authoritative** B1 reference is still the harness
(`dispatch_pack_quant_once` + `local_gemm`), which the main agent runs separately.

## 5. Correctness

- Same fp8 e4m3 (OCP gfx950) + per-128 fp32 scale + bf16 ABI as B0/B1/V4.
- RMS-rel vs the same bf16 reference target ~0.0033 (`< 0.10` PASS).
- Zero-sentinel: A real only on `src_rank`, zeros on the consumer rank; nonzero C + `local_A_zero=True`
  proves the remote gather happened.

## 6. Targets & risks

- **Target:** T < 285µs (beat B1 same-iteration total). Overlap floor at M1024/N2048/K7168 = 150µs
  (copy 135 < gemm 150) ⇒ realistic ~165–220µs, hard ceiling ~1.90× over B1.
- **RISK A — grid starvation (the #1 risk).** Grid is M/256 blocks: at M1024 that is **only 4 blocks**
  on a 256-CU GPU. V4's data shows block count drives latency hiding; 4 blocks may not hide the panel-0
  remote-gather latency well, and panels 1..7 are serialized *within* each of the 4 blocks. **Mitigation
  if P3 is grid-starved:** split the N range across blocks too (grid = (M/256)×(N_GROUPS)), giving up
  some copy-once (A re-crosses XGMI once per N-group) for occupancy — a tunable `N_PER_BLOCK` exactly
  like V4's NSUB sweep. Start with M-only (true copy-once) and measure; if T > B1, raise block count.
- **RISK B — HBM A-cache round-trip.** Panel 0 writes the dequantized A strip to HBM and panels 1..7
  read it back: extra `256×K×2 B` write + 7× read per block. At ~7 TB/s this is cheap vs the 135µs
  remote gather it replaces, but it is not free; if it dominates, cache the *fp8* bytes (½ the
  bytes) and dequant per panel instead.
- **RISK C — single-block-pass overlap depth.** With only 1 N-panel worth of MFMA to hide panel-0's
  remote gather, and panels 1..n purely local, the *overlap* benefit is concentrated in panel 0. For
  large N (more panels) the amortized remote cost shrinks (copy-once), so even imperfect panel-0
  overlap still beats B1's fully-serial copy-then-GEMM. This is the design's main bet.

## 7. Why this avoids P1/P2's failure modes (explicit)

| P1/P2 failure | P3 |
|---|---|
| Two kernels, cross-stream flag handshake | ONE kernel, no flags, no cross-stream |
| Consumer spin-waits on producer flag → serialized + spin waste | No spin; overlap is in-block warp double-buffer (V4-proven) |
| Flag/visibility mis-publish → RMS=inf | No cross-block publish; all data flows LDS↔registers in one block |
| Producer underfills (few resident blocks) | No producer kernel; every block does its own gather+MFMA |
