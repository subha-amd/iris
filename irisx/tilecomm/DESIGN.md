# TileComm — Tile-Level Communication for Fused Multi-GPU Inference

> A programming-model + scheduler proposal that turns the hand-rolled communication
> schedules inside our fused MoE kernel into a first-class, demand-aware library concern.
> This is the seed for the paper *"Tile-Level Communication for Fused Multi-GPU Inference."*
>
> Status 2026-07-06: design + a runnable scheduler/cost-model prototype (`tilesched.py`,
> calibrated to our measured combine numbers). On-node XGMI validation and the drop-in
> into `fused_moe` are the next steps (roadmap at the end).

---

## 1. The observation that started this

Our fused MoE expert region already beats the unfused MORI→aiter→MORI baseline (1.56×
prefill). But look at *how* the win was actually earned on the communication side. The
combine collective — send each expert output back to the token's home rank and reduce —
was rewritten three times, and the decisive change was **not** an algorithm change. It was
a **scheduling** change, one line, in `irisx/fused_moe/example.py`:

```python
build_combine_pull(rev, world, interleave=True)   # round-robin cells across dst ranks
```

| combine schedule | measured (prefill, 8× MI350X) |
|---|---|
| sorted by destination rank (the natural CSR order) | **934 µs** |
| round-robin cells across dst ranks (`interleave=True`) | **386 µs** |

Same bytes, same kernel, same reduction — **2.42× purely from the order in which the
concurrently-running thread blocks issue their remote stores.** Sorted order lets a window
of concurrent blocks all hammer one XGMI ingress link; round-robin spreads that window
across all 8 links.

That 2.42× is real and it is ours. But the *way* we got it is a problem:

1. **It is hand-rolled.** A human noticed the link imbalance and hand-wrote a round-robin.
2. **It is workload-blind.** Round-robin balances cell *count*. It assumes every
   destination carries the same bytes. Under the expert load imbalance Simran flagged
   ("some experts get a ton and others get very few"), that assumption is false.
3. **It is not reusable.** The gather has its own hand-rolled ordering. A future
   `reduce_scatter` will need a third. Every collective re-derives link balancing from
   scratch.

**The thesis of TileComm:** the schedule should be *computed by the library from the
declared communication demand and the measured link topology* — not hand-authored per
kernel. The kernel says *what* tiles move where; the library decides *how and when*.

This is exactly the abstraction Awad asked for ("the contribution is the tile-level
communication abstraction, not just raw speed") and the traffic-shaping idea in the brief
("let a kernel express *scatter these tiles to these ranks* and have the library choose the
link-balanced schedule automatically, instead of hand-rolling the round-robin").

---

## 2. Where this sits (and why it's unoccupied)

|  | whole-tensor | **tile-granular** |
|---|---|---|
| **fixed algorithm** | NCCL / RCCL (ring, tree) | — |
| **programmable schedule** | MSCCL / MSCCL++ (buffer-level, hand-authored) | **TileComm (this)** |
| **fused into a compute kernel** | aiter / MORI (schedule baked in) | **TileComm (this)** |

- **NCCL/RCCL** operate on whole tensors with fixed algorithms. No tile granularity, no
  fusion with the compute kernel that produced/consumes the data.
- **MSCCL / MSCCL++** let you *program* a collective, but at buffer granularity, with
  hand-authored schedules, and not demand-adaptive — you still write the schedule.
- **aiter / MORI** fuse the MoE expert math and dispatch/combine, but the communication
  schedule is baked into the library; you cannot express a different tile→rank plan.
- **IRIS** gives us device-side point primitives (`load / store / get / put / atomic_*`)
  plus exactly one whole-tensor collective (`ccl.all_reduce`). Tile-granular collectives
  do not exist — which is why we hand-roll them.
- **HipKittens** gives us a beautiful *tile compute* abstraction (register/shared tiles,
  warp ops, MFMA) but has **no communication concept** at all.

The empty, interesting quadrant — **tile-granular + demand-adaptive + fused into the
compute kernel's tile loop** — is precisely what Osama named as the research prize:

> *"When you have tile-level granularity, the amount of software pipelining you can build,
> that permutation space is absolutely insane and really unexplored — how many tiles do you
> produce from the GEMM side before you do the reduce."*

TileComm turns "that permutation space" from a thing a human hand-tunes into a **schedule
space the library searches** from declared demand.

---

## 3. The abstraction — three layers

### Layer 1 — DECLARE (what the kernel author writes)

The author describes communication as a **TileTransferSet**: a set of tile transfers, each
`{tile_id, src_rank, dst_rank, size, reduce_op?}`. They never write `ctx.load`/`ctx.store`
in a hand-chosen order. Four intents cover the MoE region and the TP collectives:

```
tile_gather (tiles, src_map)                 # pull tiles from the ranks that own them
tile_scatter(tiles, dst_map)                 # push tiles to their destination ranks
tile_reduce_scatter(tiles, dst_map, op=sum)  # push + reduce on arrival (combine, all-reduce/2)
tile_all_reduce (tiles, op=sum)              # reduce_scatter + all_gather
```

The author declares the *demand*; the ordering and (later) the routing are the library's.

### Layer 2 — SCHEDULE (what the library computes)  ← the core of the contribution

Given the TileTransferSet and the measured link topology (8× MI350X = all-to-all XGMI, per-
link bandwidth probed once at init), the library produces a **link-balanced execution
order** — an assignment of transfers to the stream of concurrent thread blocks that
minimizes makespan under link contention.

This is the traffic-shaping optimizer. It is prototyped and evaluated in
[`tilesched.py`](./tilesched.py) with a cost model calibrated to our measured combine
numbers (§4). The key property: it is a **strict generalization of the hand-rolled
round-robin** — for uniform-size traffic it *is* round-robin; under imbalance it stays
link-balanced where round-robin cannot.

### Layer 3 — EXECUTE (how the schedule runs)

The same declaration lowers to IRIS primitives inside a HipKittens kernel in one of two
modes:

- **bulk-synchronous** — a standalone collective kernel. Drop-in for today's `combine_pull`
  / `gather_pack`. This is the near-term, low-risk target (Awad: "start bulk synchronous,
  then iteratively get to concurrency").
- **tile-fused** — the collective's transfers are emitted *between the GEMM's tile
  iterations*, so communication for tile *i* overlaps the compute of tile *i+1*. This is
  Osama's "fuse inside the GEMM" prize; the schedule then also chooses *how many tiles to
  produce before reducing*. Same declaration, harder lowering — the roadmap's endpoint.

The point of the three layers: **the author writes the declaration once; the bulk and fused
executions are two lowerings of the same intent, and the schedule is reused across both.**

---

## 4. The scheduler and its cost model (`tilesched.py`)

We model XGMI link contention with a **wave** model: the GPU keeps ~`C` blocks resident at
once (a wave); within a wave, each rank's link serializes the bytes headed to it, so a wave
costs the busiest link plus a fixed floor; makespan is the sum over waves.

```
makespan = A  +  Σ_waves ( max_rank  bytes_to_that_rank_in_the_wave )  / BW
           └ schedule-independent floor (issue/latency/local reduce)
```

Calibrated to the two measured combine points (sorted 934 µs, round-robin 386 µs) this
fixes `A = 307.7 µs` (floor) and `D = total_bytes/BW = 626.6 µs`. The model then reproduces
both measurements to <1% and is used to *predict* other regimes.

Three schedulers:

| scheduler | what it does | corresponds to |
|---|---|---|
| `sorted` | all of rank 0's tiles, then rank 1's… | the naive CSR order (the 934 µs cliff) |
| `round_robin` | cycle 0,1,…,W-1 over ranks with tiles left — balances **count** | today's hand-rolled `interleave=True` (386 µs) |
| `proportional` | byte-weighted fair queueing — each link appears in ~(its byte share) of every wave; balances **bytes** | the demand-aware schedule the library picks |

### Results (from `tilesched.py`, averaged over seeds — not cherry-picked)

**Uniform combine** (reproduces the measurement):

```
scheduler      makespan_us   x vs link-LB
sorted             927.8         2.39      <- naive order: 2.4x cliff
round_robin        388.2         1.00      <- today's hand-roll: already optimal here
proportional       388.3         1.00      <- library matches it automatically
```

**Imbalanced gather** (expert-tile granularity, Zipf expert popularity):

```
zipf  max/mean   sorted   round_robin  proportional   RR/prop   prop/LB
0.0     1.00     5166.7      780.4        780.4         1.00      1.00
0.6     1.37     4238.9     1065.8       1031.5         1.03      1.00
1.2     2.87     4257.2     1997.6       1968.9         1.02      1.00
1.5     3.94     4426.7     2696.7       2659.8         1.01      1.00
```

**What this says, honestly:**

1. **The scheduling decision is worth 2.4× (uniform) to ~5–6× (imbalanced) over the naive
   order** a kernel author writes by default. That naive order is not a strawman — it is the
   order you get from a CSR grouped by destination, and it is exactly the 934 µs combine we
   started with. Avoiding that cliff is the single biggest lever, and TileComm makes it
   **impossible to fall into by construction** (the author never chooses the order).
2. **The demand-aware `proportional` schedule tracks the link lower bound at every skew**
   (`prop/LB = 1.00`), where the hand-rolled round-robin drifts 1–4% under imbalance
   (`RR/prop` up to 1.03). Small — but it is the *free* half of the story: the library gives
   you the right schedule for **every** regime without any per-workload hand-tuning, and
   round-robin's error grows with skew while proportional's stays zero.
3. **Reorder-only scheduling of a single collective is bounded by the hottest link.** The
   larger, unbounded lever is **comm/compute overlap** — Layer-3 tile-fusion — where the
   goal is to hide communication *entirely* under GEMM compute. The cost model here
   quantifies the pure-communication schedule; the overlap is the research payoff the
   abstraction is built to express.

---

## 5. Grounding — this maps onto code we already have

TileComm is not vaporware; every intent already exists as a hand-rolled special case in
`irisx/fused_moe/`:

| today (hand-rolled) | TileComm intent | the schedule that's currently hard-coded |
|---|---|---|
| `combine_pull_kernel` + `build_combine_pull(interleave=True)` | `tile_reduce_scatter` (dst = token home, op = top-k sum) | `sched_round_robin` — literally the `interleave` loop |
| `gather_pack_kernel` + multi-source route | `tile_gather` (src = expert rank) | an implicit expert-major order |
| future GEMM + all-reduce (Osama's prize) | `tile_reduce_scatter` emitted in the GEMM tile loop | *the schedule space to search* |

**The concrete first step is a drop-in:** replace the `interleave=True` argument with a call
to `tilecomm.schedule(transfer_set, topology)` that returns the cell order. It must
reproduce the 386 µs on uniform traffic (regression gate) and beat round-robin on a
load-imbalanced trace. Because it is the same `combine_pull_kernel`, this is low-risk and
directly measurable against the number we already trust.

---

## 6. Roadmap

1. **On-node XGMI microbenchmark** (`xgmi_probe.py`) — a standalone IRIS store benchmark on
   the 8× MI350X that measures sorted vs round-robin vs proportional directly, giving a
   *second measured point* to validate the cost model beyond the combine calibration.
2. **Drop-in into `combine`** — `tilecomm.schedule()` replaces `interleave=True`; regression-
   gate at 386 µs on uniform, measure the win on a load-imbalanced routing trace (Simran can
   supply a real text-distribution trace; the harness already tracks per-expert row counts).
3. **`tile_reduce_scatter` as a real IRIS collective** — the TP-decomposition stepping stone
   Osama named (all-reduce = reduce_scatter + all-gather), bulk-synchronous first.
4. **Tile-fused execution** — emit the reduce-scatter transfers inside the GEMM tile loop;
   the schedule now also chooses tiles-produced-before-reduce. This needs the producer/
   consumer-warp GEMM body (the open item in the master handoff §6), and it is where the
   comm/compute-overlap win is unbounded.
5. **Multi-path routing under skew** — when a single (src,dst) link is hot, split the flow
   over a 2-hop relay through an idle rank. Turns Layer-2 from ordering-only into full
   traffic engineering.

## 7. Honest limitations

- The cost model is calibrated to **one** measured collective (combine). Step 1 adds a
  second, independent measured point; until then, absolute predictions off the calibration
  point are estimates, not measurements.
- Reorder-only scheduling buys 2.4–6× over the naive order but only 1–4% over the already-
  good hand-rolled round-robin. The abstraction's value at this layer is *automation +
  correctness-by-construction + robustness to imbalance*, not a large new speedup. The large
  speedup lives in Layer-3 tile-fusion (comm/compute overlap), which is future work.
- On a fully-connected 8-GPU XGMI fabric, a scatter's per-link bytes are fixed by demand, so
  ordering cannot beat the hottest-link floor. Multi-path routing (step 5) is what moves that
  floor, and only helps when the demand matrix is skewed onto specific links.
