# AGENT 00 REPORT — production ABI + routing

Branch `agent/00-production-abi`. Design-only agent (00): compiling is optional per
AGENT_COMMON §0; no GPU touched. Deliverables under `irisx/abi/`.

## Build command (for the header / capture helper, main agent, node, GPU disabled)
```
docker exec r1_c4 bash -lc '
  export HIP_VISIBLE_DEVICES=""; export ROCR_VISIBLE_DEVICES=""
  cd /home/<USER>/HipKittens/distributed-kernels
  hipcc --offload-arch=gfx950 -fsyntax-only -x c++ \
    -I <repo>/irisx/abi <repo>/irisx/abi/route_capture.hpp
  # pure-C header check:
  gcc -fsyntax-only -x c <repo>/irisx/abi/route_abi.h'
```
Generators (CPU only, run anywhere, no GPU):
```
python3 irisx/abi/route_generators.py --kind zipf  --t-local 64 --out-bin /tmp/z.bin --out-json /tmp/z.json
python3 irisx/abi/route_generators.py --kind hot   --hot-expert 3 --t-local 128
python3 irisx/abi/route_generators.py --kind many-empty --n-empty 220 --t-local 32
python3 irisx/abi/route_generators.py --kind uniform --t-local 16
```

## Proposed GPU capture commands (MAIN AGENT ONLY, under flock)
Capture is opt-in via `ROUTE_CAPTURE` (see ROUTE_CAPTURE_SCHEMA.md). To capture real routing
from the IRISX dispatch bench (does NOT touch the live R1 server; runs the standalone bench):
```
docker exec r1_c4 bash -lc '
  cd /home/<USER>/HipKittens/distributed-kernels/<dirname>
  source /usr/share/Modules/init/bash; module load mpi/openmpi-x86_64
  # build with capture on:
  flock /tmp/mi355x_compile.lock -c "cmake --build build_00 -j8 --target <dirname> \
    -- CXXFLAGS=-DROUTE_CAPTURE=2"
  flock /tmp/mi355x_project_gpu.lock -c "
    ROUTE_CAPTURE_ENABLE=1 \
    mpirun --allow-run-as-root --mca pml ob1 --mca btl self,vader -np 8 python3 example.py"'
```
For PRODUCTION routing capture, wire the summary drain at ATOM `moe.py:609` (see
ROUTE_CAPTURE_SCHEMA §4) — that is a model-server change; coordinate so the live server is not
disturbed (capture into a ring, drain off the hot path).

## Expected output
- Generators: a line per run with `packed_rows`, `max_load`, `remote`, `m_pad`,
  `nonempty_experts/32`; optional `.bin`/`.json`. `many-empty` -> few nonempty experts;
  `hot` -> `max_load` >> mean; `uniform` -> roughly balanced; `zipf` -> heavy-tailed loads.
- Capture run: a `route_capture.bin` (+ summary JSON) whose records satisfy the §5 invariants.

## Correctness criteria
- `route_generators.validate()` passes (prefix-sum, monotonic offsets, total_packed_rows,
  route_reverse bounds) for all four kinds — assert in the script.
- Header compiles clean as C and as C++/HIP (main agent confirms on node).
- With `ROUTE_CAPTURE=0` the built kernel's disassembly / kernel set is identical to a
  baseline build (no-op proof): `roc-obj`/`llvm-objdump` diff shows no added kernels.
- A captured `route_capture_record`: `sum(rows_per_expert)==total_assignments` (uncapped),
  `expert_offsets[0]==0`, `expert_offsets[32]==total`.

## Assumptions
- R1-0528, EP8, top-k=8, H=7168, 32 local experts/rank (AGENT_COMMON §4) — fixed.
- fc1 is g1u1 fused gate||up, N=4096 -> SiLU(gate)*up -> 2048; fc2 K=2048,N=7168.
  N-split order and a fresh fc2 cite are NEEDS-NODE (below).
- Production A is row-major `[token,H]` fp8 OCP e4m3 + group-major (transposed) fp32 scale +
  sorted_ids indirection (FMOE_LAYOUT.md, source-cited).
- Single-rank synthetic generator sets src_rank==my_rank; multi-source (Agent 03) extends
  route_segment/route_reverse over real src_rank fan-in.

## Known risks
- **Scale-layout trap** (token-major vs group-major) is the #1 silent-correctness hazard;
  `route_params.scale_layout` makes it in-band but consumers must actually check it.
- `m_pad` is a guess (pad to 32) until NEEDS-NODE Q3 confirms; generators expose it as a knob.
- instrumentation.patch and route_capture.hpp are COMPILE-UNTESTED locally (no sandbox
  compiler); design-reviewed only. Main agent must syntax-check on node before GPU use.
- Capturing production routing requires a model-server hook — must not perturb the live server.

## Files changed (all new, under irisx/abi/ + repo root)
- `irisx/abi/PRODUCTION_ABI.md` — finalized ABI (structs, shapes, fp8/scale formats + trap,
  EpCombine needs, graph-capture notes, NEEDS-NODE table).
- `irisx/abi/ROUTE_CAPTURE_SCHEMA.md` — opt-in instrumentation schema + dump formats.
- `irisx/abi/route_abi.h` — C header, all structs (compile-untested locally).
- `irisx/abi/route_capture.hpp` — no-op-when-disabled capture helpers (the patch wires these).
- `irisx/abi/instrumentation.patch` — opt-in route-capture patch for moe_dispatch_pack_quant.hip.
- `irisx/abi/route_generators.py` — synthetic uniform/Zipf/hot/many-empty buffers (CPU/numpy).
- `AGENT_REPORT.md` — this file.

## [NEEDS-NODE-MAIN-AGENT] list (greps in PRODUCTION_ABI §1,§2,§5,§6,§7)
- Q1 fc1 fused-N split order (gate|up vs interleaved per-128).
- Q2 W2 K=2048,N=7168 fresh source cite.
- Q3 transposed-scale `M_pad` (pad to block_size_M=32?).
- Q4 EpCombine reverse-index tensor name/dtype.
- Q5 confirm `sorted_weights` == our `route_reverse.route_weight`.
- Q6 R1 actually graph-captures the MoE region + batch buckets.
- Q7 capacity policy (drop vs pad; V1 uses PER_SRC_CAPACITY=64).
- Q8 EpCombine metadata exact shapes.
- Q9 graph-capture pointer stability of the production heap.
- Q10 source-rank pre-grouping vs pure sorted_ids.
Exact grep commands for each are in PRODUCTION_ABI.md §1,§2,§5,§6,§7.
