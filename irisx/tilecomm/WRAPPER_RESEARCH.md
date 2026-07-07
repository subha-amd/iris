**Bottom Line**
Declare/schedule/execute is the right decomposition, but only if `execute` has explicit completion semantics. The current repo v0 is a useful refactor: `combine_pull_kernel` now builds a `tilecomm::TileTransferSet` and calls `tile_reduce_scatter` in [kernel.cpp](/Users/subha/repos/iris/irisx/fused_moe/kernel.cpp:1739), while `gather_pack_kernel` still directly issues `ctx.load` in [kernel.cpp](/Users/subha/repos/iris/irisx/fused_moe/kernel.cpp:149). The missing research contract is not “move tiles” but “move tiles, prove arrival, avoid illegal rank lowering, and make schedule reordering semantically invisible.”

**Recommended API**
`TileTransferSet` should become an immutable manifest plus a separate `Plan`: `{tile_id, src_rank, dst_rank, src_ptr, dst_ptr, bytes/layout/format, owner, reduce_op, epoch, completion_policy}`. Then `plan = tilecomm::schedule(set, topology, constraints)` and `execute(plan, mode=bulk_sync|tile_fused)`. Do not encode the schedule only by permuting `cell_dst`; that is fine for v0, but too implicit for a library boundary.

Completion policies should be first-class:

- `fire_and_forget`: allowed only for terminal outputs or externally proven barriers.
- `bulk_barrier`: after all participants return, every destination tile is visible and readable.
- `arrive_counter` / `arrive_signal`: payload store, system-scope release fence, signal/counter store, receiver acquire-waits before reading.
- `epoch_double_buffered`: required for fused mode so a late tile cannot satisfy a reused counter.

Ordering: no cross-tile order by default. The library may reorder any independent tiles. The only mandatory order is payload-before-arrival-signal. Reductions require declared algebra: `sum` is associative in math but not bitwise stable in fp; deterministic mode must pin a tree/order and will cost performance.

For correctness, the wrapper should lower rank selection through a safe backend-specific path. The design docs say Triton `iris.store` faults for data-dependent `to_rank`, requiring `static_range(WORLD)`; [tilecomm_device.h](/Users/subha/repos/iris/irisx/tilecomm/tilecomm_device.h:28) says C++ `iris_device_view` runtime rank is safe. Hide that split. The user API should never expose “maybe constexpr rank” as a footgun.

**Memory Model**
Use the pattern already present in [ep8_gather.h](/Users/subha/repos/iris/irisx/fused_moe/ep8_gather.h:29): host phase barriers for bulk mode; release/acquire system-scope flags for in-launch handoff. But do not blindly trust `iris.barrier()` as store completion. The XGMI probe says stores are fire-and-forget and issue-bound; wrapper semantics must require a real remote-completion primitive or a validated signal-after-data protocol. [EXPERT-Q] Q1: What exact ROCm/XGMI/IRIS operation establishes remote visibility of prior device stores: kernel completion, `ctx.fence`, atomic release, a readback, or an IRIS/SHMEM-style `quiet`?

Correct-by-construction rules: one writer per destination tile unless using a declared reduction; reductions should prefer local fp32 reduce then one store, like combine, not cross-rank atomics; all counters need epochs; waits inside kernels must be proven deadlock-free under occupancy. [EXPERT-Q] Q2: What progress/deadlock conditions apply when consumer blocks spin on arrival counters while producer blocks may not yet be resident?

**Prior Art Positioning**
This is not new RMA. SHMEM/OpenSHMEM/PGAS and GPU variants like NVSHMEM/rocSHMEM already provide symmetric heap, one-sided put/get, atomics, fences, and waits; see PGAS theory and NVSHMEM analyses ([PGAS](https://arxiv.org/abs/1307.6590), [NVSHMEM](https://arxiv.org/abs/2606.05951)). IRIS is the AMD/Triton substrate here, not the novelty ([IRIS](https://arxiv.org/abs/2511.12500)).

This is not new programmable collectives in the broad sense. SCCL/MSCCL synthesize topology-aware collective algorithms; MSCCL++ exposes primitive interfaces and DSLs while hiding synchronization/consistency complexity ([SCCL](https://arxiv.org/abs/2008.08708), [MSCCL++](https://arxiv.org/abs/2504.09014)). A reviewer can fairly call TileComm “MSCCL++-like, but tile/RMA/fused-kernel scoped.” The defensible distinction is dynamic sparse tile demand from MoE routing plus direct lowering inside an HK/IRIS tile loop.

This is not new overlap. Flux, CoCoNet, Google/AMD-style fused computation-collective work, and T3 all attack dependent communication overlap ([Flux](https://arxiv.org/abs/2406.06858), [CoCoNet](https://arxiv.org/abs/2105.05720), [fused computation-collectives](https://arxiv.org/abs/2305.06942), [T3](https://arxiv.org/abs/2401.16677)). Your novelty is narrower: tile-granular RMA collective wrappers over IRIS, with schedule/correctness owned by the library, and eventually emitted from the GEMM tile loop.

DeepEP/NCCL EP are direct threats for MoE dispatch/combine positioning: they already provide specialized expert-parallel communication and GPU-initiated paths ([UCCL-EP discussion of DeepEP](https://arxiv.org/abs/2512.19849), [NCCL EP](https://arxiv.org/abs/2603.13606)). TileComm must therefore avoid claiming “first MoE tile communication”; claim “general tile-level collective abstraction over IRIS RMA, reusable across MoE gather/combine and TP reduce-scatter/all-reduce.”

**Algorithm Answers**
For direct routing on a fully connected fabric, reorder-only scheduling cannot beat the hottest directed link plus injection/egress limits. With divisible tiles and independent links, weighted fair scheduling reaches `T >= max_e D_e/B_e`; `tilesched.py` is essentially enforcing that. The 2.4x sorted-to-round-robin win is avoiding a bad issue order, not beating a lower bound. [EXPERT-Q] Q3: Under finite resident blocks, IRIS issue queues, and nonuniform tile sizes, what is the exact approximation guarantee or hardness of the wave scheduling objective?

Multi-path 2-hop relay can beat the direct hottest-link bound only by changing the routing problem. For one hot `(s,d)` pair and idle relays, splitting via relays can reduce direct-link time by using `s->r` and `r->d` links, at the cost of extra bytes, relay storage, latency, and synchronization. For dense balanced all-to-all, it usually cannot help because all links or node injection are already loaded. [EXPERT-Q] Q4: On MI350X XGMI specifically, are per-GPU injection/ingress limits, routing hardware, and relay store-forward overhead low enough for 2-hop software routing to win under expert skew?

For reduce-scatter/all-reduce on 8 fully connected GPUs: direct all-push reduce-scatter is volume-optimal if every rank sends shard `j` to owner `j`, owner reduces locally, then all-gather distributes reduced shards. Ring/Rabenseifner/recursive-halving-doubling remain relevant when latency, channels, or injection constraints dominate; recent reduce-scatter/allreduce theory still distinguishes round-optimal and volume-optimal regimes ([Träff 2024](https://arxiv.org/abs/2410.14234)). [EXPERT-Q] Q5: For the measured TP4 shapes on MI350X, where are the breakpoints between direct all-push, ring, recursive-halving/doubling, and RCCL’s chosen algorithms?

Tile fusion has a simple batching model. If `G` tiles are produced before a reduce batch, per-tile steady cost is approximately `max(t_compute, t_comm + L/G)`, with fill/drain overhead. If `t_compute > t_comm`, choose `G >= L/(t_compute - t_comm)` to hide latency; if bandwidth dominates, larger `G` only amortizes latency and cannot hide the byte term. The repo’s [OVERLAP_A.md](/Users/subha/repos/iris/irisx/tilecomm/design/OVERLAP_A.md:143) derives a `G* ~= sqrt(P*L/(t_c+t_m))` sketch. [EXPERT-Q] Q6: Is that batching model valid for IRIS stores and XGMI, or does issue-queue/completion behavior require a different latency term?

**Adversarial Objections**
The strongest objection: “This is a nice API around known collectives, and the measured scheduling benefit over a competent round-robin is only 1-4%.” That objection is valid unless the paper leads with correctness-by-construction, dynamic sparse demand scheduling, and fused tile pipeline semantics rather than “better ordering.”

Second: bulk-synchronous TileComm will often lose to RCCL for standard dense all-reduce. Use RCCL/RCCL+MSCCL++ for ordinary whole-buffer collectives unless the wrapper exploits sparsity, MoE-specific ownership, in-kernel fusion, or avoids a known bad hand-written order.

Third: tile granularity can destroy compute. The measured naive in-GEMM gather was 13x slower; without producer/consumer warp specialization and nonblocking handoff, tile fusion is a regression machine.

**Open Questions For Experts**
1. [EXPERT-Q] Q1: What exact ROCm/XGMI/IRIS primitive gives remote store completion and visibility for device-initiated RMA?
2. [EXPERT-Q] Q4: Can 2-hop relay routing actually beat direct routing on fully connected MI350X under skew after injection, relay, and synchronization costs?
3. [EXPERT-Q] Q5: What reduce-scatter/all-reduce algorithm is optimal for these TP4 message sizes on 8x MI350X: direct all-push, ring, recursive-halving/doubling, or RCCL’s current choice?
4. [EXPERT-Q] Q2: What are the deadlock-free progress conditions for in-kernel arrival-counter waits under GPU block residency limits?
5. [EXPERT-Q] Q3: What is the right formal model for finite-wave tile scheduling with issue queues and heterogeneous tile sizes?
6. [EXPERT-Q] Q6: What is the correct software-pipeline batching model for `G` tiles before reduce when store completion is not synchronous?
7. [EXPERT-Q] Q7: What novelty framing would distributed-algorithms reviewers accept versus MSCCL++/SCCL/NCCL EP/DeepEP: API, schedule synthesis, memory-model safety, or fused tile execution?
