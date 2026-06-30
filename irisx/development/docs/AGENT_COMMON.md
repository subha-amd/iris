# AGENT_COMMON — shared rules + ABI for all IRISX/HK MoE subagents

Every subagent on this project (`agent/00-*` … `agent/09-*`) MUST read this file first and obey
it. It is the single source of truth for the rules that would otherwise be repeated in every
prompt. Your task-specific prompt layers on top of this.

---

## 0. The one hard rule: NO subagent touches the GPU

You MAY: read code, write code, create tests/harnesses, **ssh to the node and COMPILE**,
disassemble, extract VGPR/AGPR/SGPR/LDS/scratch metadata statically, write docs, write the exact
GPU commands the main agent will later run.

You MUST NOT, anywhere (local or on the node): run `mpirun`, run any `example.py` / Python that
imports torch-GPU or initializes HIP tensors, run any HIP executable, run `rocprof`/`rocprof-compute`,
start ATOM/vLLM, or call anything that does `hipSetDevice`/`hipMalloc`/launches a kernel.

Before ANY shell command on the node, export:
```
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
```
Static resource extraction is allowed and encouraged:
```
# compile-only resource report
hipcc ... -Rpass-analysis=kernel-resource-usage -c kernel.cpp
# or inspect the built object
llvm-objdump -d --mcpu=gfx950 <obj>     # disasm
roc-obj / readelf on the .hsaco         # metadata (vgpr/sgpr/lds/scratch)
```
The MAIN AGENT runs all on-device tests, one at a time, under `flock /tmp/mi355x_project_gpu.lock`.
The node also runs a live R1 vLLM server — never disturb it, never take VRAM, never kill processes.

### Compile coordination (node is shared — even compile-only must be gentle)
The node serves a live model. Do NOT launch a wide parallel build. When you compile on the node:
- use YOUR OWN build dir so agents don't stomp each other: `cmake -B build_<NN>` (NN = your agent
  number), never the shared `build/`;
- cap parallelism: `cmake --build build_<NN> -j8` (NOT -j16);
- serialize against other agents with the COMPILE lock (separate from the GPU lock):
  `flock /tmp/mi355x_compile.lock -c "cmake --build build_<NN> -j8 --target <dir>"`;
- one compile at a time per agent; if you only need a syntax check, prefer a single-TU
  `hipcc --offload-arch=gfx950 -fsyntax-only` over a full cmake build.
Compiling is OPTIONAL for design-only agents (00/01/09). For schedule/occupancy agents (04/05/07)
a static resource report IS the deliverable, so compiling is expected — just lock + low -j.

Connection details (host/container/paths/build+run commands): see `NODE_ACCESS.local.md`
(gitignored — do NOT copy its contents into any committed file; use the placeholders
`<NODE>`,`<USER>`,`<HK_ROOT>`,`r1_c4` in committed docs).

---

## 1. Git / file isolation

- Work ONLY in your own branch + worktree. Branch names are fixed:
  `agent/00-production-abi`, `agent/01-strong-baseline`, `agent/02-grouped-scheduler`,
  `agent/03-ep8-multisource`, `agent/04-8wave-pingpong`, `agent/05-4wave-interleave`,
  `agent/06-cache-first-touch`, `agent/07-register-occupancy`, `agent/08-xcd-scheduling`,
  `agent/09-irisx-tile-api`.
- Branch off `irisx` (the working branch), in the iris repo at `C:/Users/subvadla/repos/iris`.
- Create a NEW candidate directory under `irisx/` (e.g. `irisx/v5_grouped/`). NEVER edit the
  canonical `irisx/v3_fused_kernel/` or `irisx/v4_astationary_kernel/` in place.
- Do NOT merge any other agent's branch. The main agent owns all merges and the final push to
  `fork` (`github.com/subha-amd/iris`). NEVER push to `origin` (public ROCm/iris).
- Do NOT commit secrets: no `hf_...` tokens, no real hostnames/paths (use placeholders). Secret-scan
  your diff before committing. No Claude co-author trailer on commits (user preference).

If on the node you need a buildable tree, mirror your candidate dir under
`<HK_ROOT>/distributed-kernels/<your_dir>/` (the build system auto-discovers `*/kernel.cpp`). Keep
the node copy and the repo copy identical.

---

## 2. Mandatory deliverables (every agent)

Commit to your branch:
1. the code (in your new candidate dir);
2. `AGENT_REPORT.md` containing, in this order:
   exact build command · exact PROPOSED gpu test command (for main agent) · expected output ·
   correctness criteria · assumptions · known risks · files changed · static resource info if you
   compiled it.
3. plus the task-specific docs your prompt names (e.g. `PRODUCTION_ABI.md`).

Record negative results too. If something doesn't compile or a design can't meet a constraint, say
so explicitly — a rigorous negative result is a valid deliverable.

---

## 3. The shared metadata ABI (canonical — all kernels must speak this)

Agent 00 owns/refines this; everyone else consumes it. Until 00 finalizes, use exactly these
definitions so branches stay compatible:

```c
// One contiguous run of rows for one expert coming from one source rank.
struct route_segment {
    int expert_id;       // local expert index [0,32)
    int src_rank;        // owning rank of the activation rows [0,8)
    int src_row_begin;   // first row in the source rank's activation buffer
    int dst_row_begin;   // first row in this expert's packed output region
    int row_count;       // number of contiguous rows
};

// Prefix-sum layout of the 32 local experts in the packed activation/output space.
int expert_offsets[33]; // expert_offsets[e]..expert_offsets[e+1] = rows of expert e
int rows_per_expert[32];

// Flattened GEMM work unit (Agent 02 grouped scheduler).
struct expert_task {
    int local_expert;
    int m_tile_begin;    // row offset within the expert's region (multiple of BM)
    int valid_rows;      // <= BM; for tail masking
    int n_superblock;    // which NSUB-wide N panel
    int segment_begin;   // index into route_segment[] for this tile
    int segment_count;
};
```
Reverse-route / combine metadata (route_slot mapping packed-row -> original token) is defined by
Agent 00 in `PRODUCTION_ABI.md`; consumers must preserve it through the kernel.

---

## 4. Fixed problem constants (R1, EP8)

```
EP_SIZE=8 · 256 global routed experts · 32 local experts/rank · top-k=8
K=7168 (W13 input)   W13: gate+up, N=2048 each (verify fused-vs-split in Agent 00)
W2/down: K=2048, N=7168 (VERIFY in Agent 00)
activations FP8 e4m3 (OCP on gfx950, NOT fnuz) · per-128-group fp32 scales (DeepSeek style)
B/weights bf16 for now (fp8 weights later) · output bf16
```
M is **per-expert rows (M_e)** unless a doc explicitly says "aggregate routed rows". Never compare
per-expert and aggregate M silently.

---

## 5. Canonical V4 facts (the thing we're generalizing)

`irisx/v4_astationary_kernel/kernel.cpp`: BM=BN=BK=64, NSUB=8, NSTAGE=2, 4 permanent producer
waves + 4 permanent consumer waves, 8 live fp32 accumulator tiles, ~222 VGPR, occupancy ~2
waves/SIMD, ~160B scratch spill. Result: M1024/N2048/K7168 fused ~677µs vs V3-direct-pull baseline
~1227µs = 1.80–1.83×; wins M≥512, loses M≤256. RMS-rel 0.00331; zero-sentinel proves remote gather.
This baseline is WEAK (refetches A once per N tile). The honest comparison is vs Agent 01's B1
(gather-once-then-local-GEMM) and B2 (MORI+AITER). Do not over-claim 1.82×.
