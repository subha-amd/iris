# XCD-aware grid scheduling for the V4 A-stationary fused MoE GEMM

Candidate: `irisx/sched_xcd/` (built on `irisx/v4_astationary_kernel/`, which is left untouched).
Target: MI355X / gfx950 — `NUM_XCDS = 8`, `CUS_PER_XCD = 32`, `NUM_CUS = 256`.

---

## 1. The reuse opportunity

V4 launches grid `(gridDim.x = ceil(N / N_PER_BLOCK), gridDim.y = ceil(M / BM))` with
`BM = 64`, `N_PER_BLOCK = NSUB*BN = 8*64 = 512`.

* `blockIdx.y` selects an **M-tile** (`block_row = blockIdx.y*BM`).
* `blockIdx.x` selects an **N-superblock** (`NSUB=8` adjacent BN=64 N-subtiles).

For one fixed M-tile, **all `gridDim.x` N-superblocks gather the SAME remote
`A[block_row : block_row+BM, :]` rows** over IRIS (FP8 e4m3 bytes + per-128 fp32 scales). That A
strip is the scarce cross-GPU (XGMI / P2P) traffic V4 already amortizes *within* a block (A loaded
once per K-tile, reused across NSUB N-subtiles). The remaining redundancy is *across* blocks: the
`gridDim.x` superblocks of one M-tile each re-cross XGMI for identical A bytes.

### How the hardware places blocks
gfx950 round-robins the **raw row-major linear block id** to XCDs:
`xcd = (blockIdx.y*gridDim.x + blockIdx.x) % NUM_XCDS`.
With the stock launch, the `gridDim.x` superblocks of one M-tile are **consecutive** linear ids, so
they are **sprayed across all 8 XCDs**. Each XCD owns its own L2 slice / LLC path, so any cacheable
remote A line is duplicated up to 8x — no cross-superblock reuse is possible.

---

## 2. The remap

We do **not** change the launch grid or the per-task work. We change *which physical block* (the one
HW already placed on a given XCD) runs *which logical `(m_tile, n_super)` task*, using HipKittens'
`chiplet_transform_chunked()` (the inverse of the HW round-robin) plus a reference-GEMM "W"-window
M super-grouping.

```
xcd_map_block(blockIdx.y, blockIdx.x, num_m=gridDim.y, num_n=gridDim.x):
    num_wgs = num_m * num_n
    chunk   = XCD_C>0 ? XCD_C : num_n          # default: one M-tile's superblocks = one chunk
    wgid    = blockIdx.y*num_n + blockIdx.x     # raw row-major linear HW id
    tid     = chiplet_transform_chunked(wgid, num_wgs, NUM_XCDS=8, chunk)   # inverse round-robin
    # reference-GEMM W-window decode of the permuted id -> (pid_m, pid_n):
    blocks_per_grp = XCD_W * num_n
    group_id       = tid / blocks_per_grp
    first_m        = group_id * XCD_W
    grp_rows       = min(num_m - first_m, XCD_W)
    idx_in_grp     = tid % blocks_per_grp
    pid_m = first_m + (idx_in_grp % grp_rows)   # column-major within window: A-sharing stays contiguous
    pid_n = idx_in_grp / grp_rows
block_row = pid_m*BM ;  block_n0 = pid_n*N_PER_BLOCK ;  n_tile0 = pid_n*NSUB
```

`chiplet_transform_chunked` (HipKittens `include/cdna4/common/util.cuh`): the HW assigns blocks to
XCDs by `id % num_xcds`. This function is the inverse permutation: it places a **contiguous chunk of
`chunk_size` logical ids on a single XCD**. Within each full `block = num_xcds*chunk_size` region it
splits `id` into `(xcd = id%num_xcds, local = id//num_xcds)`, then `(chunk_idx, pos)` from `local`,
and reassembles `chunk_idx*block + xcd*chunk + pos`. Ids past the last full block are left unchanged.

### Parameters (compile-time `#define`, overridable via `-D`)
| param | default | meaning |
|---|---|---|
| `XCD_REMAP` | `1` | master switch. **`0` = exact stock V4** (pid_m=blockIdx.y, pid_n=blockIdx.x, no transform) — the A/B control. |
| `XCD_W` | `8` | M-tile super-group window (reference-GEMM WGM / GROUP_SIZE_M). |
| `XCD_C` | `0` | chunk_size sentinel; `0` => use `gridDim.x` at runtime so **a whole M-tile's `gridDim.x` A-sharing superblocks become one contiguous chunk → land on one XCD**. |

With `XCD_C = num_n`, the chiplet transform pulls all of one M-tile's N-superblocks onto a single
XCD instead of spraying them — so the 2nd … `gridDim.x`-th superblock could (hypothesis below) hit a
cache line the 1st already pulled, on the *same* XCD's L2/LLC.

---

## 3. Predicted mapping (canonical R1 point M=1024, N=2048, K=7168)

`num_m = 1024/64 = 16`, `num_n = 2048/512 = 4`, total `= 64` blocks. `chunk = num_n = 4`,
`block = 8*4 = 32`, two full chiplet blocks (no tail). `XCD_W=8` → 2 full M-windows.

* **Stock V4 (`XCD_REMAP=0`)**: M-tile 0's 4 superblocks are linear ids `0,1,2,3` → XCDs
  `0,1,2,3`. Sprayed across 4 different XCDs; no shared-A cache reuse possible.
* **XCD remap (`XCD_REMAP=1`)**: the inverse-RR pulls each M-tile's 4 superblocks onto **one XCD**;
  consecutive M-tiles fill the 8 XCDs round-robin within the W-window. Each XCD then gathers the
  A strips of only `num_m / NUM_XCDS = 16/8 = 2` M-tiles (vs touching all 16 under stock spray).

`xcd_map_viz.py` prints this per-XCD footprint and mechanically verifies the bijection for
`(1024,2048), (512,2048), (256,2048), (2048,2048), (1024,7168)`.

---

## 4. Numerics are provably unchanged (bit-for-bit V4)

The map `wgid → tid → (pid_m, pid_n)` is a **pure permutation** of which physical block executes
which logical `(m_tile, n_super)` task:

1. `chiplet_transform_chunked` is a bijection on each full `8*chunk` region and identity on the
   (disjoint, higher-numbered) tail, so `tid` is a permutation of `[0, num_wgs)`.
2. The W-window decode is a bijection from `[0, num_wgs)` onto the in-range `(pid_m, pid_n)` lattice
   **when `XCD_W` divides `num_m`, or `num_m ≤ XCD_W`** — true for all canonical R1 shapes
   (`num_m ∈ {8,16,32}`, `XCD_W=8`). Out-of-range `(pid_m,pid_n)` are masked with the same edge
   rule V4 uses for ragged tiles.

Therefore **every in-range `(m_tile, n_super)` task runs exactly once**, and the per-task work
(remote FP8 gather, per-128 dequant, double-buffered MFMA pipeline, store) is the SAME instruction
stream as V4. The remap changes only the XCD/CU a task lands on, never its arithmetic. Output is
identical: **RMS-rel 0.00331** vs the bf16 reference, and the **zero-sentinel** (rank-1 local A = 0
⇒ a correct result proves the bytes were gathered remotely) is preserved.

> **Constraint to respect:** keep `XCD_W` a divisor of `num_m` (or `≥ num_m`). For arbitrary M not a
> multiple of `XCD_W*BM`, set `XCD_W` to a divisor of `num_m`; otherwise tids in a ragged last
> window decode to `pid_n ≥ num_n` and would be dropped, losing tasks. The 5 canonical shapes are
> all safe; `xcd_map_viz.py`'s bijection check will flag any unsafe `(M, XCD_W)` combo.

---

## 5. The L2-reuse HYPOTHESIS — to be MEASURED, not asserted

**Hypothesis:** co-locating an M-tile's whole A-sharing superblock set onto one XCD lets the
2nd…`gridDim.x` superblocks **hit L2/LLC** for the shared remote A instead of re-crossing XGMI,
reducing XGMI read bytes and raising L2-TCC / LLC(MALL) hit% — at unchanged numerics.

**This may be false.** Remote P2P / XGMI loads on MI355X may not populate a *reusable* cache line on
the consumer XCD (they can be treated as uncached fabric reads, or land in a different cache domain),
in which case the remap will move XGMI bytes around without reducing them. **The main agent must
measure** the counters in `AGENT_REPORT.md` (XGMI read bytes/packets, L2-TCC hit%, LLC/MALL hit%)
for `XCD_REMAP=1` vs `XCD_REMAP=0` at M=1024/N=2048/K=7168 to confirm or refute. A neutral result is
a valid, reportable outcome.
