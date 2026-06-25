# V2 HipKittens Fused-MoE Expert-GEMM — Feasibility & Design Map

Target: write a "V2" HK (C++/HIP) grouped FP8 block-scale expert GEMM on **8×MI355X (gfx950, CDNA4)**
that consumes the **expert-major FP8 buffer** our IRISX MoE-dispatch kernel produces, instead of
matching production's row-major+sorted-ids contract.

All claims tagged: **[SRC]** confirmed from source on this node, **[INF]** inferred from source,
**[RT]** needs a runtime check. Investigated `<HK_ROOT>` and `<IRISX_ROOT>`,
read-only except the build dir and this doc. Build probe run + verified.

---

## 0. TL;DR

- **HK builds and runs on gfx950.** [SRC] Verified by probe: the standalone FP8 GEMM compiled clean and
  the resulting ELF ran with exit 0. Recipe in §1.
- **Best starting template: `kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu`** for the math core, **layered with the
  `distributed-kernels/bf16_gemm` IRIS pattern** for remote tile access. [SRC] Both confirmed present; the
  fp8 core is build-verified.
- **HK has FP8 e4m3 tiles + MMA, and a block-scaled MMA — but the scaled path is MX-only (e8m0, 32-wide K-group).**
  Our buffer is **fp32 per-128-K-group** (DeepSeek block-scale). HK's `mma_ABt_scaled` will NOT accept our scales;
  **we apply the fp32 per-128 dequant ourselves** (epilogue / per-K-group accumulator scaling, like `torch_scaled`). [SRC]
- **IRIS is already integrated into an HK GEMM.** `distributed-kernels/bf16_gemm/kernel.cpp` passes an
  `iris::iris_device_view` into the kernel and does device-side `iris_ctx.store(ptr, val, rank)`. The device view also
  exposes `load(ptr, remote_rank)`, atomics, `fetch_add`, `fence`. This is the tile-level remote-access primitive for V2. [SRC]
- **No grouped/variable-M GEMM exists.** All GEMMs are single fixed-shape (M,N,K template params). V2 must add the
  per-expert loop itself (host launch-per-expert, or one grid with an expert-id to tile map). [SRC]

---

## 1. Build recipe for gfx950 (VERIFIED by probe)

**Toolchain on node** [SRC]: ROCm **7.2.4**, `hipcc` HIP 7.2 (`7.2.53211`), AMD clang **22.0.0**, cmake 3.31.
`/opt/rocm/.info/version` = 7.2.4. `which hipcc` = `/usr/bin/hipcc`.

**Arch flag** [SRC]: HK selects arch by `GPU_TARGET`. In `kernels/common.mk` and
`distributed-kernels/CMakeLists.txt`:
- `GPU_TARGET=CDNA4` (the default) -> `-DKITTENS_CDNA4 --offload-arch=gfx950`
  (CMake also adds `-DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math`).
- CDNA3 -> gfx942, CDNA2 -> gfx90a, UDNA1 -> gfx1250. **gfx950 == CDNA4 is the default — no override needed.**

**Headers**: `include/` has `cdna4/` (gfx950) and `udna1/` (gfx1250) backends, switched by the `KITTENS_*` define.
`include/kittens.cuh` is the umbrella header. `THUNDERKITTENS_ROOT` **must** point at the repo root so `common.mk`
adds `-I$(ROOT)/include` (set by env.src; the probe had to set it explicitly when not sourcing env.src).

**Submodules**: `.gitmodules` exists but is **empty** [SRC] — no missing-submodule blocker despite the no-`.git` deploy.

### 1a. The probe (standalone, no MPI/IRIS, single source) — PASSED
```bash
ssh <node>   # see ../NODE_ACCESS.local.md
cd <HK_ROOT>/kernels/gemm/fp8fp32/FP8_8wave
THUNDERKITTENS_ROOT=<HK_ROOT> make        # GPU_TARGET defaults to CDNA4=gfx950
# -> /opt/rocm/bin/hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -w -O3 \
#       -I<HK_ROOT>/include -I/opt/rocm/include/hip -c 8_wave.cu ...
# -> HK_BUILD_EXIT=0 ; produced ELF "tk_kernel"
./tk_kernel    # RUN_EXIT=0   (8192^3; ran on an available slice)
```
Note the `No module named pybind11` line is **benign** — it only matters for the `BUILD_MODE=pyext` path; the
standalone GEMM does not import it.

### 1b. The IRIS/distributed path (needed for the real V2; NOT probed — needs network + MPI)
```bash
source /usr/share/Modules/init/bash && module load mpi/openmpi-x86_64
cd <HK_ROOT>/distributed-kernels
cmake -B build -DDK_BUILD=bf16_gemm          # GPU_TARGET defaults CDNA4=gfx950
cmake --build build -j 16
```
**Blocker to flag** [SRC/RT]: `distributed-kernels/CMakeLists.txt` fetches IRIS via **CPM** from GitHub
`ROCm/iris` tag `muhaawd/irisx`, SOURCE_SUBDIR `irisx`. This needs outbound network (and downloads CPM.cmake).
If the node is offline, point CPM at the **local** checkout already present at `<IRISX_ROOT>`
(e.g. `-DCPM_iris_SOURCE=<IRISX_ROOT>`) — **[RT]** not verified. MPI is gated behind the module above. **[SRC]**

---

## 2. Closest existing kernel = the V2 starting template

| candidate | dtype | scale | distributed | verdict |
|---|---|---|---|---|
| `kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp` | bf16->fp32 | none | no | fastest bf16; wrong dtype |
| `kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu` | **fp8e4m3->fp32->bf16** | none | no | **math-core template (build-verified)** |
| `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu` | fp8e4m3 | none | no | ~3% faster, custom load/store, less extensible |
| `kernels/torch_scaled/scaled_matmul.cu` | fp8e4m3 | **per-row / per-col vectors** (1xM, 1xN) | no | **epilogue-dequant pattern to copy** |
| `kernels/gemm/mxfp8/MXFP8_8wave/8_wave_*.cu` | fp8e4m3 | **MX e8m0, 32-wide, in-MMA** | no | scale format wrong for us, but shows scale-tile staging |
| `distributed-kernels/bf16_gemm/kernel.cpp` | bf16 | none | **IRIS** | **the remote-tile/IRIS pattern to copy** |
| `kernels/gemm/bf16fp32/gfx1250/gemm_expert.cpp` | bf16 | none | no | **false friend** — "expert" = scheduler expert-mode, NOT MoE; also gfx1250-only |

**Recommendation** [INF]: start from **`fp8fp32/FP8_8wave/8_wave.cu`** (clean fp8 MFMA core on standard HK memory
ops, 256x256x128 tiles, 8-wave ping-pong) and graft on:
1. the **per-128-K-group fp32 dequant** (combine `torch_scaled`'s `mul_row`/`mul_col` epilogue idea with
   per-K-group accumulation — see §4), and
2. (for the overlapped/gathered variant) the **IRIS device-view tile access** from `distributed-kernels/bf16_gemm`.
The README itself says the **8-wave** version is "more programmable... easier to extend," vs the 4-wave one built
on kernel-specific load/store. [SRC]

---

## 3. HK tile / MMA API + AMD scheduling pattern

### Tile types [SRC] (`include/cdna4/types/`)
- **Global**: `gl<dtype, B, D, R, C>` — typed global-memory view with compile-time or `-1` dims; indexed by a
  4-coord `{b,d,r,c}` in *tile* units.
- **Shared**: `st_fp8e4m3<ROWS, COLS, st_16x128_s>`, `st_bf<..., st_16x32_s>`, `st<dtype,R,C,swizzle>` — the swizzle
  tag (`st_16x128_s`, `st_16x32_s`, `st_16x64_s`) encodes the bank-conflict-free LDS layout matched to the MFMA.
- **Register**: `rt_fp8e4m3<M,K>` / `rt_bf<M,K,row_l,rt_16x32_s>` operands; `rt_fl<M,N,col_l,rt_16x16_s>`
  accumulator (note **col_l** accumulator, 16x16 base). `RT::col_vec` / `RT::row_vec` for per-row/col scale vectors.

### load -> mma -> store flow [SRC] (from `8_wave.cu`)
1. `G::prefill_swizzled_offsets(As[..], A, sw_A)` once — precompute swizzled LDS write offsets per thread.
2. `G::load(As[tic][..], A, {b,d,r,c}, sw_A)` — group (8-warp) coalesced **global->shared** with double buffer
   (`tic/toc`), software-pipelined ahead of compute.
3. `subtile_inplace<REG_M,BLOCK_K>(As[tic][i], {warp_m,0})` then `load_st_to_rt<RT_A>(a, subtile)` —
   **shared->register** per warp.
4. `mma_ABt(cX, a, b, cX)` — MFMA. Contract: `D[N,M] += A[N,K] . B^T[M,K]`; **B is passed (M,K) row-major**, A is
   (N,K); the reduction dim K must match. `static_assert` enforces dtypes: bf16.bf16->fp32, half.half->half, or
   **fp8e4m3.fp8e4m3->fp32**. [SRC `mma.cuh:471`]
5. `store(C, cX, {...})` — register->global.
- The 256x256 output tile is split across 8 warps (WARPS_ROW=2 x WARPS_COL=4), each owning 4 accumulators
  (cA,cB,cC,cD), stored to the 4 quadrants at the end.

### AMD scheduling = 8-wave ping-pong / 4-wave interleave [SRC]
HK does **NOT** use NVIDIA-style warp specialization in the GEMMs. Instead, inside the K-loop each step is hand-bracketed:
```
asm volatile("s_waitcnt lgkmcnt(0)");        // wait LDS reads complete
__builtin_amdgcn_s_setprio(1);               // raise priority around the MFMA
mma_ABt(cX, a, b, cX);
__builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();                // ping-pong handoff between wave halves
__builtin_amdgcn_sched_barrier(0);           // pin instruction order vs the compiler
```
Loads for iteration k+1/k+2 are issued *between* the MFMAs (`G::load(... k+1 ...)`), and `s_waitcnt vmcnt(N)`
gates how many outstanding global loads may remain — this is the explicit memory/compute overlap. `warp_m==0/1`
half-barriers create the two-group ping-pong. The distributed kernel instead uses an explicit
**producer/consumer warp split** (`is_producer`/`is_consumer`, warp_group_id) — the other documented pattern. [SRC]
There is also a richer scheduler API on the gfx1250 path (`kittens::sched::expert_scope`, `wait_alu`, `wait_ds`,
`load_async`) but **`load_async`/`sched::*` are gfx1250-only — not in `include/cdna4/`** [SRC], so V2 (gfx950)
uses the `s_waitcnt`/`s_barrier`/`setprio` idiom above.

---

## 4. FP8 + block-scale support — the central mismatch

**What HK has** [SRC]:
- **FP8 e4m3 tiles + MFMA**: `st_fp8e4m3`, `rt_fp8e4m3`, `mma_ABt(fp8,fp8)->fp32`. (`fp8fp32`, `torch_scaled`.)
- **Per-tensor/row/col scale** (`torch_scaled`): scales are 1xM and 1xN fp32 vectors, applied **in the epilogue**
  after the unscaled fp8 MMA via `mul_row(c, c, scale_a_rv)` / `mul_col(c, c, scale_b_rv)`. No K-dependence.
- **MX block-scale in the MMA** (`mxfp8` + `mma_ABt_scaled`, `pack_scales`): scales are **e8m0 (`fp8e8m0`)**,
  **one per 32 contiguous K elements**, staged into LDS as `st<fp8e8m0,16,64,...>`, packed to `fp8e8m0_4`, and fed
  *inside* the MFMA (`mma_ABt_scaled(d,a,b,c,&sa,&sb)`). The header `static_assert`s and the `pack_scales`/opsel
  logic are hard-wired to e8m0x32. [SRC `mma.cuh:253,529`]

**What our buffer is** [SRC, from `irisx/.../test_moe_dispatch_pack_quant.hip` + V1_RESULTS/FMOE_LAYOUT]:
- **fp8 OCP e4m3 (max 448)** values, `packed_fp8[base*H]`.
- **fp32** scale, **one per 128 contiguous K (= hidden H) elements**, `scale = max(|group|)/448`,
  `packed_sc[base*N_GROUPS]`, `N_GROUPS = H/128 = 56`. This is DeepSeek-style **per-(token,128-K-group)** block-scale.

**Conclusion** [INF]: HK's in-MMA scaled path does **not** fit (wrong scale dtype e8m0!=fp32 and wrong group 32!=128).
So **V2 dequantizes itself**. Two viable shapes, both fp32 per-128:
- **(a) K-group accumulation (recommended for final perf).** Pick `BLOCK_K = 128` = exactly one scale group. For each
  K-step do the unscaled `mma_ABt` into a *temporary* fp32 accumulator, then `C_acc += temp * (scale_a_group *
  scale_b_group)` before the next K-step — i.e. fold the two fp32 group scales as a per-tile multiply (reuse
  `torch_scaled`'s `mul_row`/`mul_col` mechanics, but applied **per K-iteration** rather than once at the end).
  For activation A the scale is per-row (per token) x per-group; for weight B the per-128 weight scale is
  per-(N,group). Exact register-vec broadcast layout is **[RT]**.
- **(b) Preamble dequant to bf16 (recommended for bring-up).** Convert the whole BLOCK_M x 128 fp8 A-tile to bf16 by
  multiplying each 128-group by its fp32 scale in shared/registers, then run the **bf16** MMA core (the fastest,
  best-tested HK path). Simpler, costs one extra convert; likely the right first cut.

---

## 5. The IRIS bridge — how HK already pulls/pushes remote tiles  (KEY for V2)

From `distributed-kernels/bf16_gemm/kernel.cpp` + `iris_py.cpp` + `<IRISX_ROOT>/include/iris/iris.hpp`: [SRC]

- The kernel's `globals` struct carries **`iris::iris_device_view iris_ctx;`** alongside the `gl<>` tensors. It is
  built host-side by `IrisInstance` (MPI init -> `iris(heap_bytes, rank, world)` -> `get_device_view()`), bound to
  Python via `iris_py` (`Iris.empty(shape,dtype)` allocates on the **symmetric IRIS heap**, `barrier()`, `rank()`).
- **Device API** (`iris.hpp` `class iris_device_view`, all `__host__ __device__`):
  ```
  T    load(const T* ptr, int remote_rank);
  void store(T* ptr, T value, int remote_rank);
  T    atomic_load(const T*, int rank, order, scope);
  void atomic_store(T*, T, int rank, ...);
  T    fetch_add / fetch_sub (T*, T, int rank, ...);
  bool compare_exchange_strong(..., int rank, ...);
  void fence(order); int cur_rank(); int world_size();
  ```
  Internally each call does `remote_ptr = heap_bases_[remote_rank] + (ptr - heap_bases_[cur_rank])` then a normal
  HIP load/store/atomic — i.e. **IPC-mapped peer pointers** (built via `hipIpcMemHandle` exchange in the ctor;
  `max_world_size = 8`, matching our 8-GPU EP). So a single GPU thread can read/write another GPU's heap by passing
  that rank. [SRC]
- **How the bf16 GEMM uses it today**: only in the **store** epilogue — `kittens_store(g.c, C_accum, idx, iris_ctx)`
  walks the register accumulator and does `iris_ctx.store(&dst_ptr[...], val, cur_rank)` element-by-element
  (it uses `cur_rank()`, i.e. writes locally through the IRIS path; the *plumbing* to target a remote rank is the
  same call with a different rank arg). **Caveat** [SRC]: `kittens_store` has
  `static_assert(!std::is_same_v<T, fp8e4m3>)` — the current IRIS store helper does **not** support fp8 output;
  for V2 we add an fp8/bf16 store specialization (our GEMM output is bf16 anyway, so fine).
- **The V2-relevant pattern** [INF]: because our `packed_fp8`/`packed_sc` are **allocated on the IRIS heap**
  (`iris_obj.allocate<fp8_t>(...)` in the dispatch test), a V2 GEMM threadblock can **gather an A-tile (and its
  scales) directly from the producing rank's heap** with `iris_ctx.load(&packed_fp8[base*H + ...], src_rank)` —
  enabling **tile-level overlap of gather + GEMM** (issue the next rank's/expert's remote tile load while MFMA-ing
  the current one), instead of waiting for the whole dispatch to land. Our buffer's `src_rank` axis maps directly
  to the IRIS `remote_rank` arg. The exact `s_waitcnt`/fence discipline for remote loads is **[RT]**.

---

## 6. Grouped / variable-M (32 experts, different token counts)

- **No grouped/batched/variable-M GEMM exists in HK.** [SRC] Every GEMM is a single fixed `<M,N,K>` template with
  `grid = (M/BLOCK)*(N/BLOCK)` and compile-time tile counts. There is no segment/offset table, no `sorted_expert_ids`
  consumer, no ragged-M handling. (`gemm_expert.cpp` is scheduler "expert mode," unrelated.)
- Our experts: **32 local experts/GPU**, each `[n_tokens_e, 7168] x [7168, 2048]` (gate/up) then
  `[.,2048] x [2048,7168]` (down); `n_tokens_e` varies. [SRC shapes]
- **How V2 expresses it** [INF]:
  - Our buffer gives **fixed capacity per (expert,src): `PER_SRC_CAPACITY` slots**, zero-padded (the dispatch test
    `hipMemset`s the whole buffer to 0). So the **padded M per expert is constant = `world * PER_SRC_CAPACITY`** —
    V2 can treat each expert as a *fixed-shape* GEMM over the padded slot grid and simply let the zero rows produce
    zero (and skip storing padded outputs). This sidesteps true ragged-M entirely. **The real (unpadded) per-expert
    token count for early-exit / store-masking is `send_counts[expert]`** (already computed by `k_count`). [SRC]
  - Launch options: (i) **one kernel launch per expert** (32 launches, simplest, host loop over `local_e`,
    pointer-offset into `packed_fp8 + (local_e*world*PER_SRC_CAPACITY)*H`); or (ii) **one grid, expert-major tile
    map** — `grid.z = 32` (or fold expert into `blockIdx`), each block reads its `local_e` and offsets the A base +
    the per-expert weight B. Option (ii) gives better occupancy/overlap and is the V2 target; (i) is the bring-up step.
  - Because M is the expert-major **token/slot** dim and BLOCK_M tiles it, **variable real-M only affects how many
    M-tiles do useful work** — handle by masking/early-return on `tile_m * BLOCK_M >= send_counts[e]`.

---

## 7. Proposed V2 design sketch

**Copy** `kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu` -> new `distributed-kernels/fmoe_expert_gemm/kernel.cpp`
(so it picks up the IRIS CMake target convention: a dir with `kernel.cpp` + pybind `tk_kernel`). **Change:**

1. **Globals**: add `iris::iris_device_view iris_ctx;` and our buffer views:
   `gl<fp8e4m3,...> packed_fp8` (A, expert-major), `gl<float,...> packed_sc` (per-128 scales),
   `gl<fp8e4m3,...> W_gate/W_up/W_down` (B), `gl<bf16,...> out`. Add `int n_experts, world, per_src_cap, send_counts*`.
2. **Tiling**: `BLOCK_M x BLOCK_N x BLOCK_K = 256x256x128`; **`BLOCK_K = 128` deliberately = one scale group**.
   Per-expert M-extent = `world*per_src_cap` (padded). K = H = 7168 -> 56 K-iters -> 56 scale groups (matches `N_GROUPS`).
3. **Expert loop**: `blockIdx.z = local_e` (or host loop for bring-up). Offset A base by
   `(local_e*world*per_src_cap)*H`, scales by `*N_GROUPS`, and B by this expert's weight slab.
   Early-return M-tiles beyond `send_counts[local_e]`; zero-pad rows are harmless.
4. **Load**: reuse `G::load`+`prefill_swizzled_offsets` for the fp8 A/B tiles. For the **gathered** variant, replace
   the A-tile global load with `iris_ctx.load(&packed_fp8[...], src_rank)` where `src_rank = (slot_row / per_src_cap)`
   — i.e. the buffer's `src_rank` axis is the IRIS rank. Prefetch next tile's remote load across the MFMA. **[RT]**
5. **Dequant**: load the two fp32 group scales for the current `BLOCK_K=128` window
   (`packed_sc` for A per (slot-row, k); weight scale for B per (n, k)); run unscaled `mma_ABt` into a temp fp32 tile,
   then fold `temp * scale_a_group * scale_b_group` into `C_acc` per K-iter (start from `torch_scaled`'s
   `mul_row`/`mul_col`, but per-K-group). First cut may instead **dequant fp8->bf16 in the preamble** and use the bf16
   core (§4b). Activation (SiLU(gate).up) is a fused epilogue between the gate/up GEMM and the down GEMM.
6. **Store**: bf16 output via plain `store` (local) or an fp8/bf16-enabled `kittens_store` over IRIS for cross-rank
   combine — add the missing non-fp8 store specialization (current helper `static_assert`s out fp8). **[SRC caveat]**

**Build V2**: `distributed-kernels/` CMake, `-DDK_BUILD=fmoe_expert_gemm`, default CDNA4=gfx950, IRIS from the local
`irisx` checkout if offline (§1b).

---

## 8. Unknowns & confidence

| item | confidence |
|---|---|
| gfx950 build flag + toolchain; standalone fp8 GEMM compiles **and runs** | **[SRC] verified by probe** |
| HK has fp8 tiles + fp8 MMA; scaled MMA is MX-e8m0-32 only (!= our fp32-128) | **[SRC]** |
| Our buffer = `[local_e][src_rank][slot][H]` fp8 + `[...][N_GROUPS=56]` fp32, scale=max/448 | **[SRC]** (dispatch test) |
| IRIS device-view `load/store/atomic(ptr,rank)` exists; IPC peer-pointer impl; max_world=8 | **[SRC]** |
| HK GEMM already carries `iris_device_view` + does device-side IRIS store | **[SRC]** |
| No grouped/variable-M GEMM; must add expert loop + send_counts masking | **[SRC]** |
| 8-wave ping-pong via s_waitcnt/s_barrier/setprio/sched_barrier; `load_async`/`sched::*` gfx1250-only | **[SRC]** |
| Distributed CMake fetches IRIS over network (CPM, ROCm/iris@muhaawd/irisx) | **[SRC]**; offline-local-override **[RT]** |
| Per-128 fp32 dequant via per-K-group accumulator scaling (option a/b) | **[INF]** design; exact reg-vec broadcast layout **[RT]** |
| Remote-tile-gather overlap (`iris_ctx.load` of A-tile from src_rank) correctness/waitcnt discipline | **[INF]** plumbing exists; **[RT]** behavior |
| Padded-M (world*per_src_cap) vs ragged-M handling | **[INF]** from zero-init + send_counts |
| Distributed/IRIS GEMM build itself (needs MPI module + network) | **[RT]** — not probed (probe was the standalone fp8 core) |
| Running on a busy node (96% VRAM held) | standalone probe ran; large/multi-GPU runs **[RT]** |
