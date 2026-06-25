# Handoff — Build the V0 MoE dispatch+pack kernel in IRISX

> For a fresh Claude (or human) starting the kernel implementation. Read this first, then
> `KERNEL_PLAN.md` (the full design + profiling basis). This doc = where to look, what
> server to use, how to build/run, and the first concrete coding steps.

## 0. One-paragraph context
We profiled DeepSeek-R1-0528 decode on 8×MI355X and found the MoE **expert gather/scatter**
(`EpDispatchIntraNodeKernel` / `EpCombineIntraNodeKernel`) is ~14–16% of decode GPU time and
sits right around the expert GEMM. The job: write a **device-side IRISX kernel** that
collapses the *pre-GEMM dispatch-prep chain* (`EpDispatch → opus_moe_sorting ×2 → dynamic_quant`)
into one kernel that routes top-k token activations to expert-owning GPUs and writes them
directly in the expert-major layout the GEMM expects. V0 = bf16 dispatch+pack. This is the
simpler device-side prototype the mentors (Osama/Simran) asked for, before reduce_scatter, and
it validates the IRIS symmetric-memory load/store abstraction that later HipKittens-fused
kernels will reuse. **Full design, cost model, and the measured numbers are in `KERNEL_PLAN.md`
— do not re-derive them; read that file.**

## 1. What to read, in order
1. `KERNEL_PLAN.md` (this repo root) — the plan. §4 has the V0/V1 kernel specs (V0a atomic,
   V0b precomputed-offset), §2 the R1 shapes, §1 the profiling basis + 3 cost surfaces.
1b. `PRIOR_ART.md` (this repo root) — **READ THIS.** RadeonFlow/Gau/DeepEP/PPLX/MoRI have built
   this kernel family already. It lists the proven design ideas to copy and the baselines to
   beat. Key consequences are folded into the V0 spec below; the doc has the full reasoning.
2. `irisx/include/iris/iris.hpp` — the entire IRISX API (header-only). Read the
   `iris_device_view` class: `store`, `load`, `fetch_add`, `translate`, memory orders/scopes.
3. `irisx/benchmarks/all_put.hip` — **the template for V0.** A grid-stride kernel that does
   `iris_view.store(...)` to remote ranks + a GB/s timer. V0 is a smarter all_put.
4. `irisx/benchmarks/put.hip` — single-pair put + bandwidth measurement pattern.
5. `irisx/tests/test_remote_store.hip` — Catch2 test pattern (producer rank stores to
   consumer rank's buffer, consumer verifies). Model the V0 correctness test on this.
6. `irisx/README.md` — build/run overview (but use the EXACT commands in §3 below — the
   README's default arch is wrong for our node).

Companion profiling docs (in the sibling `kernels-testing` repo, not this one):
`EP_PROFILING_RESULTS.md` (consolidated C4/C6/C4-LL), `results/C4/v1_envelope_C4_ht.md`
(the measured pre-GEMM envelope), `results/C4/sql7_gaps_C4_ht.md` (transition gaps).

## 2. The server
**SSH details, host, key, user, and node paths are in `NODE_ACCESS.local.md`** (gitignored —
not in this public fork). Read that file for the exact `ssh`/`scp` commands and paths.
- 8× MI355X, **gfx950** (CDNA4), 384 CPU / 3TB RAM. The node is ours; no Slurm gating.
- Ignore the Conductor SSH banner noise; auth works with the key.
- Write under the home dir (1.8 TB free on `/`). `/data` and `/data2` are NOT writable.
- The R1 model + the production ATOM server live here too (see KERNEL_PLAN/EP_PROFILING docs),
  but **you don't need them for V0** — V0 is a standalone microbenchmark on synthetic data.

### Verified toolchain (all present on the node, 2026-06-24)
| tool | version | note |
|---|---|---|
| hipcc | HIP 7.2 (clang 22) | `/usr/bin/hipcc` |
| cmake | 3.31.8 | `/usr/bin/cmake` |
| git | present | |
| OpenMPI | 4.1.1 | **NOT on PATH by default** — must `module load` (below) |
| GPU arch | **gfx950** | the CMake default `gfx942` is WRONG; override it |

## 3. Build & run IRISX on the node (exact, verified commands)
IRISX hard-requires MPI (`find_package(MPI REQUIRED)`, `#include <mpi.h>`). MPI is installed
but gated behind a module. Always load it first:
```bash
# SSH in + deploy the repo: see NODE_ACCESS.local.md for the exact commands/paths.
source /usr/share/Modules/init/bash
module load mpi/openmpi-x86_64        # puts mpirun/mpicc on PATH (OpenMPI 4.1.1)
module load rocm/7.2.4                # (optional; rocm already default)

cd <repo>/irisx                       # repo path: see NODE_ACCESS.local.md
# IMPORTANT: override the gfx942 default for our MI355X node
cmake -B build -DIRIS_BUILD_BENCHMARKS=ON -DIRIS_BUILD_TESTS=ON \
      -DIRIS_HIP_ARCHITECTURES=gfx950
cmake --build build --parallel 8

# sanity: run the existing all_put bench on 8 GPUs (this is the bandwidth ceiling for V0)
./scripts/iris_run 8 ./build/benchmarks/all_put 8
# (iris_run is just: mpirun -np <N> <binary>)
```
**First step for the new agent: get this baseline building and `all_put` running before
writing any new code.** If CMake/CPM can't fetch spdlog/catch2 (it pulls them via CPM from
GitHub), check the node has outbound HTTPS (it does — HF worked) or pre-seed the CPM cache.

## 4. R1 shapes the kernel is specialized for (from KERNEL_PLAN §2)
```
hidden H            = 7168     (bf16 token vector; 56 groups of 128)
top-k               = 8        (expert assignments per token)
routed experts      = 256  →   32 experts per GPU at EP=8
EP_SIZE             = 8        (TP4×DP2 or DP8)
quant               = FP8 e4m3, block 128  (V1 only)
bf16 token bytes    = 14,336   |  fp8+scales = 7,392  (~2× cut, V1)
```
For V0, use synthetic data with these shapes: a `[T_local, 7168]` bf16 hidden buffer and a
`[T_local, 8]` random `topk_ids` table. Pick T_local ~ a few hundred (decode batch per rank).

## 5. Concrete first coding steps (V0)
Create `irisx/benchmarks/moe_dispatch_pack.hip` (add it to `benchmarks/CMakeLists.txt` via
`add_iris_benchmark(moe_dispatch_pack)`). Build **V0a first** (correctness), then **V0b**.

**V0a — remote-atomic slot claim (correctness baseline):**
- Allocate on the IRIS symmetric heap: per-rank source hidden `[T_local,H]`, and a
  destination expert-major packed buffer sized `[32 experts][capacity][H]` + a
  `[32]` atomic counter array, all via `iris.allocate<...>`.
- Kernel: one warp (or block) owns one (token, k) assignment. Compute
  `expert=topk_ids[t,k]; dst_rank=expert/32; local_e=expert%32`. Claim a slot with
  `iris_view.fetch_add(&count[local_e], 1, dst_rank)`. Then a warp-strided loop of
  `iris_view.store(&packed[local_e][slot][h], hidden[t][h], dst_rank)` over H=7168.
- Verify like `tests/test_remote_store.hip`: after a barrier, each rank checks its packed
  buffer contains exactly the tokens routed to its experts (compare against a CPU reference
  built from the same topk_ids).
- Time it; report GB/s vs the `all_put` ceiling and vs the production ~40 µs/instance
  (`EpDispatch + sorting×2`, from the profiling).

**V0b — precomputed-offset (the likely performance path):**
- Layout `packed[local_expert][source_rank][local_slot][H]` so each source rank owns a
  deterministic slice → **no remote atomics**. Needs a small pre-pass kernel computing
  per-(source,expert) send counts + prefix offsets (a two-kernel design is fine — see
  KERNEL_PLAN §1: graph gaps are ~1 ns, so removing the remote-atomic bottleneck matters
  more than kernel count).

**Why two variants:** remote `fetch_add` on a hot expert serializes across XGMI, and MoE
routing is skewed — V0a may be atomics-bound. V0b avoids that (PPLX-style sender-owned slices).
Benchmark both. RadeonFlow and Gau both pay one atomic/assignment and hit this; PPLX avoids it.

**Prior-art requirements (from `PRIOR_ART.md` — apply in V0):**
- Emit **`route_slot[token][topk]`** as a first-class output (needed for V3 combine later).
- **Don't hardcode grid size** — MI355X has 256 CUs (RadeonFlow's `NUM_SMS=304` is MI300X).
  Use `device_props.multiProcessorCount`.
- End dispatch with a **bulk cross-rank completion signal** (local stream order ≠ peers done
  writing into my memory). Lightweight signal, not a full 40–90 µs barrier.
- Add an **in-kernel timestamp profiling mode** early (torch/rocprof is unreliable multi-GPU).
- Try **128-element hidden chunking** if one-wave-per-full-vector under-utilizes at small batch.
- Use **vectorized (16B) non-temporal** loads/stores for the hidden copy.

## 6. What NOT to do yet
- **No HipKittens, no GEMM, no MFMA in V0/V1.** Those are V2+. V0/V1 are pure IRIS
  data-movement. (See KERNEL_PLAN §4 V2 for where HK actually enters.)
- **Don't claim the ~24% envelope.** V0 targets the dispatch+sort slice (~40 µs/instance
  measured); V1 adds quant (~46 µs). The combine side and dense/attention quant are out of scope.
- **Don't fuse into the GEMM.** The trace proves the gather's consumer is the sorter, not the GEMM.
- Don't pitch the win as "filling idle bubbles" — SQL7 shows ~1 ns gaps. The win is removing
  intermediate HBM layouts + graph nodes (+ XGMI bytes in V1).

## 7. Definition of done for V0
1. `moe_dispatch_pack.hip` (V0a + V0b) builds for gfx950 and runs on 8 GPUs via `iris_run`.
2. A Catch2 correctness test (model on `test_remote_store.hip`) passes: packed output matches
   a CPU reference for a known `topk_ids`.
3. A bandwidth/latency number for V0a and V0b, compared to the `all_put` ceiling and the
   production dispatch+sort time. Record it in a short results note.
4. Then proceed to V1 (add FP8 quant on the better V0 layout) per KERNEL_PLAN §4.

## 8. Gotchas (learned the hard way)
- **Arch**: always pass `-DIRIS_HIP_ARCHITECTURES=gfx950`. The default gfx942 silently builds
  the wrong ISA.
- **MPI**: `module load mpi/openmpi-x86_64` every fresh shell, or builds fail at
  `find_package(MPI)` and runs fail with "mpirun: command not found".
- **Symmetric heap**: IRIS requires every rank to allocate the same buffers in the same order
  (allocations are offset-matched across ranks). Allocate identically on all ranks.
- **`max_world_size` is 8** in `iris.hpp` — fine for one node, don't exceed.
- IRISX is **intra-node IPC** (`hipIpcOpenMemHandle`); single 8-GPU node only. No multi-node.
