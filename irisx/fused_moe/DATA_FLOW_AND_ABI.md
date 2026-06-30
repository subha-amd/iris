# Fused MoE Expert Region — Data Flow, Input ABI & Warp/SIMD/XCD Map

DeepSeek-R1 · EP8 · AMD **MI355X (gfx950 / CDNA4)** · **HipKittens** (on-GPU tile compute) + **IRIS** (cross-GPU XGMI).

> Rendered version with diagrams: **[`DATA_FLOW_AND_ABI.html`](DATA_FLOW_AND_ABI.html)** (open in a browser).
> Every grid/block/wave number here is taken verbatim from `kernel.cpp`, `example.py`, `b0_tasks.py`, `build_tasks.py`, `ep8_gather.h`.

Shapes (R1): `H=7168`, `INTER=2048`, `N_FC1=4096` (gate‖up), `N_FC2=7168`, `E=32` local experts/rank, `top-k=8`, fp8 e4m3 (max 448), per-128 block scales (`QGROUP=128`, `NG=56`), bf16 activations/output, fp32 scales.

---

## 1. The region — 5 kernels, two XGMI crossings

The production **unfused** region is 7 kernels: `MORI EpDispatch → moe_sorting ×2 → dynamic_quant → fmoe_g1u1_silu → moe_sum → MORI EpCombine`.
We collapse it to **5**. The whole region runs on the **consumer** rank against an **expert-major packed buffer**; a token crosses XGMI **exactly twice** — once in (gather), once out (combine).

```
HOST METADATA (built once, off the clock):  route_segments · tilemeta → ①   tasks_fc1/fc2 → ②④   CSR(cell_*)+wgt → ⑤
                                                  │                          │                        │
   8 source ranks            ┌── XGMI in ──┐      ▼          ▼               ▼          ▼             ▼
   A_src fp8 [Msrc,7168] ───▶ ① gather_pack ──▶ ② GEMM fc1 ─▶ ③ silu_quant ─▶ ④ GEMM fc2 ──▶ ⑤ combine_pull ──┐
   + scales  (IRIS heap)      IRIS              HK 8-wave      HK (no MFMA)    HK 8-wave       IRIS pull-reduce  │
                              raw fp8 copy      dequant→       SiLU(g)·u       dequant→        fp32 reduce +     │
                              zero-sentinel     256×256×64     per-128 fp8     256×256×64      1 bf16 store/tok  │
                                  │             w13[E·4096,    requant            w2[E·7168,         │           │
                                  ▼             7168]            │                2048]              ▼           │
                              A_pk fp8        C1 bf16          A2 fp8 [Mpk,    C2 bf16          ACC bf16          │
                              [Mpk,7168]      [Mpk,4096]       2048]+sc        [Mpk,7168]       [Tlocal,7168]     │
                              expert-major     gate‖up                          down proj        on origin ranks  │
                                                                                                                  │
                              └──────────────────────── XGMI back (weighted top-k scatter) ◀───────────────────────┘

iris.barrier() → ① → ② → ③ → ④ → ⑤ → iris.barrier()      (single stream; barriers bracket the two XGMI phases)
```

**Why there is no `moe_sorting` in the fused path** (Awad's question): the unfused GEMM needs `sorted_ids/sorted_expert_ids` to group a flat `[token,H]` buffer by expert. **Our gather already groups** — `gather_pack` writes each expert's rows *contiguously* using the host `route_segments` map, so "sorting" *is* the placement. The GEMM indexes experts by `expert_row_begin`; `combine_pull` inverts the placement with a host CSR. No sort/scan kernel, no scatter-to-sorted, no `moe_sum`.

**Decode (low `M_e`):** same ①③⑤; ②④ swap the 256×256 tile for a skinny **BM=16** tile (`grouped_b0_gemm_decode` / `_fp8_sat`). At ~16 real rows/expert a 256-row tile is ~94% zero-padding and runs *compute-bound on padding*; BM=16 (MFMA's min M) drops padded compute below the weight-stream floor → decode runs *weight-memory-bound* (the right regime). The packed layout stays 256-aligned so ①③⑤ are byte-identical. `_fp8_sat` stores weights fp8 (half the HBM bytes that bound decode) — the decode lever; fp4 is next.

---

## 2. The 8-XCD execution model

An MI355X is **8 XCDs × 32 CUs = 256 CUs**. Every kernel launches a **1-D grid of independent blocks** (one block per expert-tile / packed-row / destination-cell). The gfx950 hardware workgroup scheduler sprays blocks **round-robin across the 8 XCDs** (block `i` → XCD `i mod 8`), then round-robins across the 32 CUs in each XCD. **Consecutive `blockIdx.x` land on *different* XCDs** — there is no per-XCD affinity in the code; balance comes from the round-robin + the task *count*.

```
1-D grid:  blk0 blk1 blk2 blk3 blk4 blk5 blk6 blk7 blk8 ...
              │    │    │    │    │    │    │    │    └─▶ XCD0 (wraps)
              ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
            XCD0 XCD1 XCD2 XCD3 XCD4 XCD5 XCD6 XCD7      each XCD: 32 CUs, own L2
              └──────────── shared LLC + 288GB HBM3E @ ~8 TB/s ──────────┘   ← weights B stream here (weight wall)
              └──────────── 8 XGMI links → 7 peer GPUs (only ① and ⑤) ───┘

1 CU = 4×SIMD32, wave64, 64KB LDS.  A 512-thread block = 8 wave64 ≈ one CU's worth (≤2 such blocks/CU).
```

**Three load-balancing levers — all in the host task lists, none in the kernels:**
1. **Constant tile, variable count.** Every expert uses the *same* tile height (BM=256 prefill / 16 decode). Imbalance is absorbed by `⌈M_e/BM⌉` tiles — a hot expert emits more blocks. One block = one `tasks[i]=(expert,m_tile,n_tile,expert_row_begin)`.
2. **Spray over XCDs.** A hot expert's many tiles land on different XCDs (round-robin), so no single XCD bottlenecks one expert. The ≤BM−M_e padding tail is masked / MFMA'd to zero, never read downstream.
3. **Round-robin XGMI.** The combine CSR orders consecutive cells to *different* dst ranks, spreading remote-store bytes over all 8 links (sorted-by-rank was ~2.4× slower per byte).

---

## 3. Per-kernel warp / SIMD / XCD map

### ① `gather_pack_kernel` — IRIS, XGMI gather  (`kernel.cpp:108`)
- **grid** `(Ntile, GP_SPLIT=8)`, Ntile = ⌈Mpacked/64⌉; **8 split-blocks cooperate per BM=64 tile**. **block** 256 = 4 wave64.
- **per lane:** one `uint4 = 16 fp8 bytes`, **raw copy, no dequant** — `vbytes = (src==cur)? *sp : ctx.load(sp, src_rank)` (XGMI remote load, local deref when same rank). +1 fp32 scale per 128-group (idempotent write). **No MFMA.**
- **routing (per tile, once):** `tile_is_single_source` → Path 2 (constant map, common case) else Path 1 builds `row_seg[64]` in LDS.
- **zero-sentinel:** unrouted/tail rows → `uint4(0)` → fp8 `0x00` dequants to 0 (no contamination).
- **across XCDs:** Ntile×8 blocks spray over all 8 XCDs → 8 XCDs issue XGMI reads at once; `GP_SPLIT=8` fills 256 CUs & hides remote-load latency (grid=Ntile alone ≈ 50% of the chip).

### ② / ④ `grouped_b0_gemm<N,K>` — HipKittens, 8-wave GEMM  (`kernel.cpp:749`; dequant preamble `:556`)
- **grid** `num_tasks` — one block per `(expert, 256-row m_tile, 256-col n_tile)`, expert-major. **block** 512 = **8 wave64**, `__launch_bounds__(512,2)`.
- **macro-tile:** 256(M)×256(N), contracting K=7168(fc1)/2048(fc2) in `BLOCK_K=64` steps (`k_iters`=112 fc1 / 32 fc2).
- **8 waves:** `warp_m=wid/4 ∈{0,1}`, `warp_n=wid%4 ∈{0..3}` → 2×4 wave grid; each wave holds **4 accumulators** `cA,cB,cC,cD` of `rt_fl<64,32>` (a checkerboard 128×64 of the tile).
- **SIMD work:** `mma_ABt` = **16×16×32 bf16 MFMA** (CDNA4 `v_mfma_f32_16x16x32`); ~64 MFMAs/wave/K-iter.
- **LDS:** double-buffered `As[2][2],Bs[2][2]` of `st_bf<128,64>`; `group<8>` all-wave swizzled HBM→LDS load, ping-pong `tic/toc`; hand-placed `s_waitcnt`+`s_barrier`+`s_setprio(1)` keep all 8 MFMA pipes full while next tiles stream.
- **why dequant:** HK has no fp8 global→register load on this path, so the preamble expands packed fp8 → bf16 scratch first.
- **across XCDs:** num_tasks blocks spray over 8 XCDs. Consecutive tasks (same expert, different n_tile) land on *different* XCDs, so one expert's `B[e·N:(e+1)·N]` rows are pulled by tiles spread across XCDs — each XCD caches its slice in its own L2, shared LLC/HBM backs reuse. **B is the dominant HBM stream = the weight wall.**

### ③ `silu_quant_kernel` — HipKittens, activation (no MFMA)  (`kernel.cpp:625`; decode `_mtile:677`)
- **grid** `Mpacked` (prefill) / ~512 real rows (decode `_mtile`, skips 94% padding). **block** 256 = 4 wave64. One packed row per block.
- reads `C1[row,:4096] = gate(0:2048)‖up(2048:4096)`; each lane owns `ELEMS=2048/256=8` consecutive cols (one 128-group).
- **SIMD work:** `h = silu(gate)·up = (g/(1+e^−g))·u` (transcendental on the SIMD ALU); stash in LDS.
- **reduction:** per-lane local amax → **one** `atomicMax` into `samax[group]` ⇒ **16-way** contention (16 groups), not 128-way; `scale=amax/448`; requant `q=e4m3(clamp(h/scale,±448))` → A2 fp8 + 16 scales/row. Replaces ~7 `[M,INTER]` fp32 temporaries with **1 read + 1 fp8 write**.
- **across XCDs:** embarrassingly parallel — independent rows spray over all 8 XCDs.

### ⑤ `combine_pull_kernel` — IRIS, XGMI combine / EpCombine  (`kernel.cpp:1558`)
- **grid** `num_cells` — one block per **destination cell** = unique `(dst_rank,dst_token)`. **block** 256 = 4 wave64.
- reads CSR span `[lo,hi)=cell_ptr[c..c+1]` → `cell_rows` = the ≤8 packed rows routing to this token.
- **SIMD work:** **local fp32 reduce** `Σ wgt[row]·c2[row,h]` over ≤8 rows (no atomics) — this is what makes a bf16 output correct under top-k collisions.
- **XGMI store:** **one** store of the reduced row to `accb[dst_token]` on `dst_rank` (`local *dst` if same rank else `ctx.store(dst,v,dst_rank)`). `store_gran`: bf16 / bf16×2 / uint4(16B). Shipped = **bf16** ⇒ half the scatter's fp32 bytes (combine is write-BW-bound; fewer bytes is the only lever).
- **across XCDs:** num_cells blocks spray over 8 XCDs; the host CSR round-robins cells across dst_rank so the 8 XCDs × 8 XGMI links stay busy together. Reads of `c2` are local HBM; only the store crosses XGMI.

### D. Decode GEMM (replaces ②④ at low `M_e`) — `grouped_b0_gemm_decode` / `_fp8_sat`  (`kernel.cpp:1013 / 1273`)
- **grid** ≈512 — one block per `(expert, 16-row m_tile, 256-col n_tile)`. **block** 512 = 8 wave64, **WARPS_COL=8** (warp_m=0).
- **tile** 16(M)×256(N): the 8 waves split N into **8 disjoint 32-col strips** (8-way weight parallelism); A(16×K) broadcast to all waves.
- **SIMD work:** bf16 → 16×16×32 MFMA. `_fp8_sat` → load fp8 B through the fast bf16 path, unpack fp8→bf16 in-register with per-N scale, then bf16 MFMA (halves weight HBM bytes). **Critical** `sched_barrier(0)` before each mma forbids the async ds_read hoist (else RMS≈0.71 + >512-block hang).
- **across XCDs:** ≈512 blocks → ~2/CU across 256 CUs; BM=16 makes padded compute fall below the weight-stream floor → decode runs weight-memory-bound.

---

## 4. Input ABI

**Where:** *IRIS symmetric heap* (allocated in identical order on every rank → identical offsets, so `ctx.load/store(ptr,rank)` resolves a peer's copy) vs *local HBM* (never crosses XGMI).

### 4a. Activation & weight tensors
| buffer | shape | dtype | where | built by | consumed by |
|---|---|---|---|---|---|
| `A_src` (+`A_src_sc`) | `[Msrc,7168]` (+`[Msrc,56]`) | fp8 e4m3 (+f32) | **IRIS heap** | each rank's routed activations | ① gather (remote read) |
| `A_pk` (+`A_pk_sc`) | `[Mpacked,7168]` (+`[Mpacked,56]`) | fp8 e4m3 (+f32) | **IRIS heap*** | ① gather (padding pre-zeroed) | ② fc1 dequant |
| `B_fc1` (w13) | `[E·4096, 7168]` | bf16 / fp8 | local HBM | model weights (preshuffled) | ② fc1 GEMM |
| `C1` | `[Mpacked, 4096]` | bf16 | local HBM | ② fc1 (gate‖up) | ③ silu_quant |
| `A2` (+`A2_sc`) | `[Mpacked,2048]` (+`[Mpacked,16]`) | fp8 e4m3 (+f32) | local HBM | ③ silu_quant | ④ fc2 dequant |
| `B_fc2` (w2) | `[E·7168, 2048]` | bf16 / fp8 | local HBM | model weights (preshuffled) | ④ fc2 GEMM |
| `C2` | `[Mpacked, 7168]` | bf16 | local HBM | ④ fc2 (down) | ⑤ combine (local read) |
| `ACC`/`accb` | `[Tlocal, 7168]` | bf16 (or f32) | **IRIS heap** | zeroed on every origin rank | ⑤ combine (remote accumulate) → next layer |

\* `A_pk` is on the heap only to keep allocation order symmetric; the GEMM reads it as plain local HBM (`src_rank=consumer`). The fp8 buffer is a bf16 tensor aliased to a fp8 view over the same bytes (`make_fp8`), so HK takes a bf16 `gl<>` and the dequant preamble reinterprets the bytes as fp8.

### 4b. Routing metadata (host-built once, off the timed clock)
| buffer | shape / record | feeds | meaning |
|---|---|---|---|
| `route_segments` (SEG) | `[Nseg,5]=(expert_id,src_rank,src_row_begin,dst_row_begin,row_count)` | ① | one contiguous run of rows for *one expert from one source rank*; disjoint+sorted. Makes the gather group-by-expert. |
| `tilemeta` (TILE) | `[Ntile,4]=(seg_begin,seg_count,tile_dst0,valid_rows)` | ① | per BM=64 tile: which segments touch it + true row count (tail mask). |
| `tasks_fc1`/`tasks_fc2` | `[num_tasks,4]=(expert,m_tile,n_tile,expert_row_begin)` | ②④ | the flat work list; `expert_row_begin` = expert's first packed row (mult of 256). One row = one block. |
| `expert_row_begin` | `[E]` (256-padded prefix of `M_e`) | ②④⑤ | each expert's region base in packed space; padding keeps tiles inside one expert (no contamination). |
| `cell_dst`/`cell_ptr`/`cell_rows` | `[nC,2]·[nC+1]·[total_routed]` | ⑤ | CSR transpose of the reverse route: group packed rows by `(dst_rank,dst_token)`; round-robined across dst_rank. |
| `wgt` | `[Mpacked,1]` f32 | ⑤ | per packed row's softmax gate weight (top-k weight the combine reduces by). |

### 4c. The two XGMI primitives (all of "IRIS" the region uses)
| call | used by | semantics |
|---|---|---|
| `ctx.load(ptr, src_rank)` | ① gather | read 128 bits from the *same symmetric offset* on a peer GPU over XGMI (local deref when `src_rank==cur_rank`). |
| `ctx.store(ptr, v, dst_rank)` | ⑤ combine | write bf16/bf16₂/uint4 to a peer's accumulator. Cross-rank visibility comes from the host `iris.barrier()` (hipDeviceSynchronize + MPI_Barrier = system fence) bracketing each phase, so in-kernel ops are **relaxed** — no per-element fences. |

The combine also has a **rejected** `ctx.fetch_add` scatter variant (`kernel.cpp:1464`): correct but XGMI-write-BW-bound at fp32 (788 µs). The shipped pull accumulates *locally* in fp32 and stores *bf16* → 386 µs, beating MORI EpCombine's 398 µs.

---

## ⚠️ Two ABI traps a vLLM integrator must know (from `../development/abi/PRODUCTION_ABI.md`)
1. **Scale layout.** We pack scales **token-major** `[Mpacked,56]`. Production `fmoe` reads **group-major** `[56,M_pad]` — a transpose. Handing our packed scales to stock fmoe (or vice-versa) without transposing reads numerically-wrong scales **with no crash**.
2. **fc1 N-split.** We assume the fused 4096 is `[gate(0:2048) ‖ up(2048:4096)]` (blocked, not interleaved). If the production w13 preshuffle interleaves per-128-block, `silu_quant`'s gate/up split must match. (Flagged **[NEEDS-NODE]** in the ABI.)

---

*Sources:* `kernel.cpp` (gather_pack:108 · grouped_b0_gemm:749 · silu_quant:625 · combine_pull:1558 · decode:1013/1273), `ep8_gather.h`, `build_tasks.py` & `b0_tasks.py`, `example.py`, `development/abi/PRODUCTION_ABI.md`. MI355X: 8 XCD × 32 CU = 256 CU, per-XCD L2, shared LLC, ~8 TB/s HBM3E.
