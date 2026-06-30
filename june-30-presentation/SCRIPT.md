# Speaking script — *Fusing the DeepSeek-R1 MoE Expert Region on MI355X*

**Target: ~25 min with interruptions.** Plan on ~18–20 min of *talking*; let the rest fill with Q&A — this is an expert audience and they *will* interrupt. Each slide has **SAY** (what to say, first person) and **IF ASKED** (pre-empted questions + answers). ★ = must-land, ⚡ = compressible. 22 slides.

**This talk answers the action items from our last meeting** — call these out as you go:
- *Simran:* "data-parallel TP4 is the right trade-off point" → **Slide 4** (we use C4 = TP4/DP2/EP/DP-attn).
- *Osama:* "do the MoE decode **gather** — build the model with HipKittens kernels + a load/store on the MoE side" → **Slides 9–13** (literally our design).
- *Osama:* "the real prize is **tile-level fusion inside the GEMM** — more to hide behind there, but harder, you're GEMM-resource-constrained" → **Slide 10** (we demonstrated it for the dequant) and **Slide 18** (we characterized exactly why the full version needs a new GEMM body).
- *Awad:* "the contribution is the **tile-level comm abstraction** (HK compute + IRIS comm), not just speed" → **Slides 9 & 21**.
- *Osama:* "what's the **net benefit** of fusing the gather — maybe it's small" → **Slides 5, 16, 17**.
- *Simran:* "long all-reduce times are usually **profiling artifacts** — use in-kernel / MAX-over-ranks timing" → **Slide 15**.
- *Simran/Osama:* "write a **cost model**" → **Slide 6**.

---

## Slide 1 — Title (~20 sec)
**SAY:** "This is the MoE expert-region work — taking the production DeepSeek-R1 serving path on MI355X and fusing it with HipKittens and IRIS. Headline: 1.56× on prefill, and I'll be precise about why decode is a different story. This is the follow-through on what we scoped last time."

---

## Slide 2 — The one-slide story ★ (~1.5 min)
**SAY:** "The whole arc up front. The production MoE region is a chain of *separate* library kernels — MORI dispatch, then aiter's fused MoE, then MORI combine. The expert GEMM in the middle is genuinely good; the problem is the *handoffs* around it. So the thesis is one line: **aiter fuses the expert *math*; we fuse the expert *region*** — we make the cross-GPU communication produce the GEMM's layout directly, we fuse the activation and requant, and we replace the scatter-combine with a bandwidth-efficient pull-combine. On 8 full MI355X, same-node, correctness-gated: prefill is 1.56× faster, our combine even beats AMD's own MORI combine, and decode is a marginal 2–3% — and the *why* of decode is the most interesting technical part."

---

## Slide 3 — Where the expert region lives ⚡ (~1 min)
**SAY:** "Quick grounding. After routing, with top-8, each token becomes up to 8 token-expert rows. Each row crosses to the GPU that owns its expert, runs the expert FFN, and is combined back. The FFN is two matmuls: fc1 — emits gate and up in one GEMM — then SiLU(gate)·up with an fp8 requant, then fc2 back to hidden. fc1 and fc2 are *inside one expert*. In expert-parallel, the **dispatch (gather)** sits before the GEMM and the **combine (scatter)** after — that's the all-to-all that replaces TP's all-reduce."
**IF ASKED — R1 specifics:** "H=7168, intermediate 2048, 256 experts, top-8 plus one shared; EP8 = 32 local experts per GPU; fp8 e4m3 weights, per-1×128 block scales."

---

## Slide 4 — Profiling setup ★ (~1 min) [Simran's config guidance]
**SAY:** "Config is straight from our last discussion. Simran's point: data-parallel TP4 is the better throughput/interactivity point, and the gather shows up in prefill-decode disaggregation with DP-attention. So C4 here is exactly that — TP4 × DP2, expert-parallel, DP attention. It's the *only* mode where the MoE all-to-all fires: you need dp>1 *and* the expert-parallel flag. C6 — DP8, TP1 — is the control: it strips the TP all-reduce, so the only cross-GPU traffic is the MoE all-to-all. That isolates the gather as an EP-flag artifact."
**IF ASKED — why not TP8:** "That's where I started, but it's all-reduce-dominated and isn't the decode config Simran pointed me to; the pattern we care about only appears with DP>1 + EP."

---

## Slide 5 — Profiling result ★ (~1.5 min) [Osama's "net benefit?"]
**SAY:** "What the profiler said, stable across all three variants: the MoE communication — dispatch gather plus combine scatter — is 14 to 16% of *total* GPU kernel time. Add the *adjacent* fusable sorting and quant, and you get a ~24.5% 'gather + pack + quant' envelope of the decode step. That envelope — everything *around* the expert GEMM — is the target."
**IF ASKED — Osama's "is the net benefit even big?":** "The gather *alone* is ~6%, so on its own, modest — which is why I attack the whole *envelope*. You don't just remove the gather; you remove the sort, the quant, the scatter combine, and at prefill that compounds to 1.56×. Slide 6 explains why decode is different."

---

## Slide 6 — Why decode is a *weight* wall ★★ (~2 min) [the cost model]
*Conceptual keystone — go slow.*
**SAY:** "Before writing a kernel I built a cost model by sweeping the region over batch size. One term first: M_e is the number of rows — token-expert pairs — each expert processes. At decode you're generating only a handful of tokens per step; with top-8 routing over 256 experts, 32 local per GPU, even a 64-token step lands only ~16 rows on each local expert. Now the table: as you shrink from prefill to decode, the region time *barely moves* — 292 versus 329 microseconds — and the GEMM is 95 to 97% of it. Why? **The expert GEMM has to read all 32 local experts' ~1.4 GB of fp8 weights from HBM *every single step*** — a fixed ~176-microsecond floor at 8 TB/s, no matter how few tokens you have. So decode is bottlenecked by *weight-memory bandwidth*, not the gather — the gather and combine together are only ~18% of the decode region. That single fact is why prefill wins big and decode is marginal: prefill has enough tokens that the GEMM is busy and removing the boundaries matters; decode is so token-starved that the weight wall dominates and there's almost nothing for fusion to take."
**IF ASKED — so is decode fusion pointless?** "Not pointless — ceiling ~1.1–1.4×, and you get it by *hiding* the gather under the weight stream, not removing it. The bigger decode levers are orthogonal: shard weights across more EP ranks, fp4 weights, or larger batch."
**IF ASKED — 1.4 GB math?** "940 MB fc1 + 469 MB fc2 across 32 local experts in fp8; ~176 µs at 8 TB/s, measured ~230."

---

## Slide 7 — The unfused baseline, drawn precisely ★ (~1 min)
**SAY:** "The baseline, exactly. `aiter.fused_moe` is *one Python call* but *many GPU kernels* — seven stages: MORI dispatch, two sorting passes, dynamic quant, the fmoe kernel — the good internally-fused fc1+SiLU+fc2 — then moe_sum, then MORI combine. The picture: a good expert kernel surrounded by expensive layout and materialization boundaries."
**IF ASKED — is aiter's fmoe the bottleneck?** "No, and I want to be fair: fmoe is well-optimized. We don't beat it with a faster matmul; we win by deleting the *boundaries around* it."

---

## Slide 8 — Why the unfused path bleeds ★ (~1.5 min)
**SAY:** "Let me make 'boundaries' concrete — each is an HBM round-trip in a *different* representation. Dispatch-to-sort: tokens arrive in communication order, but the GEMM needs expert-major order, so sorting is two passes of pure layout traffic, no matmul. Sort-to-quant: the dispatch moves *bf16* tokens, but fp8 fmoe needs fp8 plus scales, so dynamic_quant is a full read-bf16/write-fp8 pass. fmoe-to-combine: fmoe writes outputs to HBM, then moe_sum and combine read them back. And at decode the fixed launch and sync costs become visible. So unfused cost is: good expert kernel, plus standalone sort, plus standalone quant, plus a scatter combine — none of which is matmul."

---

## Slide 9 — What we actually built ★★ (~1.5 min) [Osama + Awad's HK+IRIS framing]
**SAY:** "Here's our design, and I want to be precise about which tool does what, because this is exactly the split we discussed. **HipKittens — the on-GPU tile library — does all the compute**: the fc1 and fc2 grouped GEMM, the SiLU-plus-requant kernel, and the local reduction in the combine. **IRIS — the multi-GPU library — does all the cross-GPU communication**: the XGMI load that pulls rows in the gather, and the XGMI store that writes results home in the combine. We use *no* aiter kernels — aiter's fmoe is the *baseline* we replace. This is literally the 'build the MoE with HipKittens kernels plus a load/store on the MoE side' design Osama proposed. And because *we* own the layouts in between, the IRIS communication *produces* the GEMM's input layout — which is what lets us delete the standalone sort, the quant pass, and the scatter combine."

---

## Slide 10 — gather_pack: communication *is* the sort ★★ (~2 min) [the comm/compute overlap]
*The user flagged this — explain the overlap carefully and concretely.*
**SAY:** "gather_pack replaces the dispatch and the two sort passes. For each routed row, we look up where it came from — which rank, which row — then issue *one* IRIS load: a direct read over XGMI, the inter-GPU link, of that row's 16 fp8 bytes plus its scale, landing it straight into our expert-major packed buffer. Rows that happen to be on this same GPU skip the network. The key: there's *no separate sort*, because the slot we write into *is* the expert-major row the GEMM wants — communication and layout are one pass.

Two things to land. First, 'zero sentinel.' To run all 32 experts in one kernel launch, we pad each expert's block of rows up to a tile boundary. The padding rows, and any unrouted rows, we write as fp8 zeros with a zero scale — so they multiply to *exactly* zero in the GEMM. That means a padding row can never leak into a real expert's output; it's a poison-free pad.

Second — and this is the communication/computation overlap idea that Osama and Muhammad cared about, so let me be concrete. The GEMM wants bf16 inputs, but what we pull over the network is fp8 — half-size bytes plus a scale. Converting fp8 back to bf16 — the 'dequant' — is a little arithmetic per number. Now, pulling a row over XGMI is *slow*: you issue the load and then the GPU sits waiting for the bytes to arrive — that's idle compute. So in a prototype, we did the dequant *inside the gather kernel*: the moment a row's bytes land, we convert them, *while the loads for the next rows are still in flight*. The conversion fills the cycles the GPU was otherwise spending waiting on the network. The proof it's free: the gather *alone* is 168 microseconds; the gather *with* the dequant folded in is 162 — the same number to within noise. The conversion cost *disappeared* because it hid in the shadow of the communication. That's tile-level comm/compute overlap, and it only works because we can express both at fine, per-row granularity. To be precise: the *shipped* region keeps the gather and the dequant as two separate kernels — both cheap and vectorized — for simplicity and correctness; this folded prototype is the proof of concept, and the *harder* version, folding the gather into the full MFMA GEMM, is where we hit a wall — slide 18."
**IF ASKED — "so the shipped path doesn't overlap?":** "Correct, and I want to be honest about that. The shipped 1.56× comes from *region* fusion — deleting the boundaries. The comm/compute *overlap* is a separately-demonstrated result that proves the abstraction works; we didn't ship the in-GEMM version because of the occupancy wall on slide 18."
**IF ASKED — "what is IRIS exactly?":** "A symmetric-memory multi-GPU library — every GPU exposes a heap at the same virtual address, and you do `ctx.load`/`ctx.store`/`ctx.fetch_add` against a (pointer, rank) pair, which compiles to an XGMI access. It's the load/store primitive Muhammad described; we build the collective out of it."

---

## Slide 11 — The expert GEMM grouped_b0 (HipKittens) ★ (~1.5 min) [NEW — the actual matmul]
**SAY:** "This is the kernel that does the actual expert matmuls — *both* fc1 and fc2. It's our own HipKittens GEMM, not aiter's. Mechanically: 8 waves per block, 256-by-256-by-64 tiles, double-buffered — it ping-pongs tiles from shared memory into registers while the MFMA matrix units chew on the previous tile — and it accumulates in fp32. All of that is written in HipKittens' tile types, so it's a few hundred lines, not thousands. The MoE-specific trick: *one thread-block per per-expert tile*, dispatched from a flat task list we build, so all 32 local experts compute in a *single* launch — that's what replaces aiter's grouped fmoe GEMM. One detail that matters for the next slides: this GEMM consumes *bf16*. The gather handed us fp8, so a small vectorized kernel converts fp8-to-bf16 right before it — the 'dequant preamble' — and vectorizing that conversion saved about 117 microseconds on fc1. Because we dequant to bf16 and then do a bf16 matmul, the prefill path is *bf16-precision*, which is where the clean 0.018 error comes from. Decode swaps in a native-fp8 version — that's the next slide."
**IF ASKED — "why not native fp8 everywhere?":** "Prefill isn't weight-bound, so bf16 is fine and gives better accuracy at no speed cost. Decode *is* weight-bound, so there the fp8 weight bytes are the whole game — and that's where the precision tradeoff shows up."

---

## Slide 12 — The activation silu_quant (HipKittens) ★ (~1.5 min)
*The user found the PyTorch number confusing — explain it directly.*
**SAY:** "This kernel is the *middle* of the FFN, between the two matmuls — it is *not* a matmul. It reads fc1's output, which is the gate and up halves concatenated, computes SiLU(gate) times up, and then re-quantizes that result back to fp8 — with a fresh per-128-element scale — so the second matmul can consume it. Now, you might ask why the comparison on this slide is against a *PyTorch* number rather than the production baseline. The reason: aiter does this activation step *inside* its monolithic fmoe kernel, so there's no standalone 'unfused activation' to race against. When we split the region into our own kernels, we had to implement this step ourselves — and our very first version just called PyTorch's built-in silu-and-quant, which was 471 microseconds. So this table is the optimization of *our own* kernel: 471 calling PyTorch, down to 127 once we fused it into one kernel, down to 38.6 with the real fix. That last jump is the interesting one: computing the per-128 maximum for the scale originally had *every* thread atomically updating one shared value — 128 threads fighting over it. We changed it so each thread first finds the max over *its own* slice and then does a *single* atomic — 16-way instead of 128-way contention. At decode a variant that skips the padded rows gets it to 7.2."
**IF ASKED — "why does activation even matter?":** "It doesn't in aiter — it's hidden inside fmoe. The moment we expose it as our own kernel, it's on our critical path; making it near-free is part of why the region holds together."

---

## Slide 13 — combine (IRIS): beating MORI's own combine ★ (~2 min)
**SAY:** "The combine is the *reverse* of the gather: take each expert-output row and send it back to the token it came from, on that token's home GPU, scaled by its routing weight — and since top-8 means a token gets contributions from up to 8 experts, you have to *sum* them. Our first version was the obvious one — a *scatter*. Each thread-block owns one expert-output row, and for every element it does an atomic add, over XGMI, into the destination token's accumulator on the home GPU. We used an atomic because several expert outputs land on the same token, so the adds can collide. That was 788 microseconds. Then we diagnosed it: we tried removing the atomic — only 4% faster; we tried packing the writes into bigger chunks — zero faster. So it's *not* the atomics and *not* the number of transactions. It's the sheer *volume* of remote writes: 8192 rows times 7168 elements times 4 bytes is about 234 megabytes pushed over the network, in fp32. The only lever is *fewer bytes*. So we inverted it — a *pull*. Instead of each source row pushing out, each *destination token* gathers its up-to-8 contributing rows from *local* memory, sums them locally in fp32 — no atomics, because one block owns the token — and then writes *one* result row home in bf16. bf16 halves the bytes, 234 down to 117 megabytes, and the local fp32 sum is what makes the bf16 output correct even with collisions. But here's the punchline detail: even the pull was *worse* — 934 microseconds — when we processed destination tokens in rank order, because that hammers one network link at a time. When we *round-robin* the destinations across ranks, so that the blocks running concurrently are spread across all 8 XGMI links, it drops to 386 — the same bytes, 2.4× faster, and now *below* MORI's own combine at 398."
**IF ASKED — "you beat MORI — fair?":** "Same region, same node, same data, correctness-gated. MORI is a general-purpose collective; we exploit that we own the layout, so we can choose bf16 output and the link-balanced schedule."
**IF ASKED — "what's an XGMI link?":** "The on-package interconnect between the GPUs — there are multiple links per GPU, and the win is keeping all of them busy instead of funneling through one."

---

## Slide 14 — The decode GEMM: native fp8 + a subtle bug ⚡ (~1.5 min)
**SAY:** "Decode is weight-bound, so the only lever is fp8 weights — half the bytes. The wrinkle: HipKittens has no fp8 *register* load path, so we store the weights as fp8 but *load them as if they were bf16*, hitting the fast load path, then unpack them in registers — with the weights pre-arranged offline into the exact order the matrix units expect. Standalone, that hits 3.97 TB/s, 1.46× over bf16, and matches aiter's fused fp8 at parity. The reason it's on a slide: it took 14 experiments to find a bug — a stable wrong answer plus an apparent hang. It turned out *not* to be a data bug — it was the compiler reordering instructions: it moved the matrix-multiply *ahead* of the wait that ensures the data has arrived, so the first multiply ran on garbage. A one-line scheduling barrier before the multiply fixed both the wrong answer and the hang."
**IF ASKED — "pre-swizzle offline — is that cheating?":** "It's a one-time weight reformat at load time, not per-step — standard for these kernels."

---

## Slide 15 — How we benchmarked ★ (~1.5 min) [Simran's artifact warning]
**SAY:** "Methodology — and this addresses Simran's point that long all-reduce times are usually profiling artifacts. We compare *regions*, not kernels: define the region by its inputs and outputs and measure both sides over it. Everything is *same-node, same operating point* — nodes here vary 1.8× in speed, and an early 'decode loss' of mine was just a throttled node. We *pre-warm* aiter, because it compiles its fmoe per-shape into a local cache the first time — a cold cache looks like a 10-minute hang but isn't. And timing is device-event based, taking the *max over the 8 ranks*, warmup plus median, with a correctness check *before* timing — exactly to avoid the per-rank trace artifacts you flagged."
**IF ASKED — "max over ranks, why?":** "The region only finishes when the slowest rank does; averaging understates it, and per-rank traces are where the bogus millisecond all-reduces come from."

---

## Slide 16 — Result: PREFILL 1.56× ★★ (~1.5 min)
**SAY:** "The headline. Same node, correctness-gated: our fused region is 1247 microseconds versus the unfused 1941 — 1.56×. The per-stage table — and the caveat, don't mix denominators across machines, this decomposition is from a different node — shows where it comes from: we roughly halve the dispatch, the activation is near-free, and the combine is below MORI. Why prefill specifically: the GEMM is busy enough that eliminating the sort, the quant materialization, and the scatter combine is pure savings. And a bonus — our prefill GEMM is bf16, so it's actually *more* accurate than aiter's fp8, and still faster."
**IF ASKED — "1.54 vs 1.56 vs 1.64?":** "1.56 is the clean same-node number; 1.64 is an earlier node's decomposition; same story, different machines. I quote 1.56."

---

## Slide 17 — Result: DECODE, honest ★★ (~1.5 min)
**SAY:** "Decode, precisely. Same node, warm aiter: 516.5 versus 527 — a 2–3% win. Per-stage, the two matmuls are 424 of those 516 microseconds — just the fp8 weight streams, the weight wall from slide 6. The boundaries we could remove were only ~18% to begin with, so we recovered the activation tax and edged ahead. The precision caveat, because someone will ask: the region scores 0.057 error against a *bf16-weight* reference, where bf16 weights would score 0.018 — and that jump *is* the fp8 weight quantization, which is scale-invariant, finer scales don't fix it. But the aiter baseline *also* runs fp8 weights, so 0.057 is production's *own* precision class. Getting under 0.05 would require a bf16 matmul, which is +135 microseconds — a decode *loss*. So it's a genuine speed-versus-precision tradeoff, and at *matched* precision, decode wins."
**IF ASKED — "win or not?":** "At production-matched precision, marginally yes. At bf16-reference accuracy, no — and no fused *or* unfused fp8 path passes that bar. I'd rather state it precisely than oversell."

---

## Slide 18 — What didn't work, part 1 ★ (~1.5 min) [Osama's in-GEMM-fusion question]
*Osama will care most about this — it's the honest result on his "fuse inside the GEMM" idea.*
**SAY:** "Two we tried and rejected, with the architectural reason — because the *why* is the contribution. First, host-stream overlap: run the gather and the GEMM on two streams so they overlap. Even after fixing a grid bug, concurrent was *negative* versus serial. The reason is what Osama predicted: the GEMM already saturates the compute units, so two streams just time-slice. Second — and this is the tile-level fusion *inside* the GEMM you flagged as the real prize — the only existing kernel that gathers remote data *under* the matrix-multiply runs 13× slower, 36 versus 466 TFLOP/s. The reason: our fast GEMM body has no producer/consumer split between warps and no reuse of the gathered data across output tiles, so a blocking remote load just stalls the whole wave instead of hiding behind compute. So gather-under-MFMA isn't a patch to today's kernel — it needs a *different* GEMM body that exposes an asynchronous fill and reuse. That's a real, scoped open direction, exactly the abstraction work Awad pointed at — not a dead end."
**IF ASKED — "so the in-GEMM fusion dream is dead?":** "No — *characterized*. We proved comm/compute overlap works where compute sits in comm's shadow — the dequant on slide 10. The full warp-level version needs a GEMM built for it from the start. I can scope that as next."

---

## Slide 19 — What didn't work, part 2 ⚡ (~1 min)
**SAY:** "Four more, one line each. Native fp8 at the big tile: neutral, because there decode is compute-bound on *padding*, so fewer weight bytes don't help until a small tile makes it memory-bound. Register double-buffering the decode tile: 3× *slower* — it blew the register budget and collapsed occupancy; at this tile size, latency-hiding is *many warps*, not a wider per-thread pipe. Compacting the padding: we measured it mostly hides under the weight stream, the only exposed cost was the activation tax we already took, and removing it would break tile-alignment — so not worth it. And a strict sub-0.05 decode: fundamental, it needs bf16 and that's a loss. The theme: *profile before you pay*."

---

## Slide 20 — The padding aside ⚡ (~45 sec) [pre-empts a sharp question]
*Only dwell here if asked "isn't padding 16× unfair?"*
**SAY (short):** "One thing experts always ask — why pad each expert to 256 rows. Flattening 32 experts into one launch needs each expert rounded up to a tile boundary so a tile never straddles two experts. Production aiter pads too, just finer — to ~32 — so it's a self-inflicted, fixable handicap, *not* an unfair benchmark. And we measured that at decode it mostly hides under the weight stream, so it wasn't worth the alignment risk to remove."

---

## Slide 21 — Summary ★ (~1 min)
**SAY:** "To close. The fusion buys 1.56× on prefill over the production chain, a combine that beats MORI's own, near-free activation, and a dequant we showed can hide under the gather — and decode goes from a loss to a marginal win at production-matched precision. The honest boundaries: decode is a weight wall, so fusion's ceiling there is modest; the warp-level gather-under-GEMM needs a new GEMM body; and the bigger decode levers are orthogonal. And the substrate is the point Awad and Osama emphasized: HipKittens for tile-level compute, IRIS for tile-level communication. One line: **make the communication produce the compute layout, hide the conversions in its shadow, and reduce on pull not scatter — and the fused region beats the production baseline, decisively at prefill.**"
**IF ASKED — "what's next?":** "Three: the gather-under-GEMM body for true in-kernel overlap; the reduce_scatter collective as the second primitive; and end-to-end integration to measure TPOT, not just the region."

---

## Slide 22 — Backup numbers (only if asked)
Keep up if someone wants the full table.

---

## "Gotcha" questions — quick answers
- **"Is the baseline production-faithful?"** Yes — MORI bf16 dispatch + aiter `fused_moe` (the C4 default) + MORI combine; same config Simran flagged; max-over-ranks; warm caches.
- **"1.56× at what batch?"** Prefill, ~8192 tokens, M_e≈256. Decode (M_e≈4) is the 2–3% number.
- **"Why bf16 prefill but fp8 decode?"** Prefill isn't weight-bound → keep precision and still win; decode *is* → fp8's halved bytes are the only lever, at the cost of the RMS floor.
- **"End-to-end TPOT?"** Not yet — region-level so far; that's next and the gold-standard denominator.
- **"How much is HipKittens vs hand CUDA?"** All on-GPU compute is HipKittens tiles (8-wave MFMA, fp8/bf16, the scheduling-barrier fix); all comm is IRIS XGMI primitives. No aiter, no hand-rolled MFMA.
- **"Combine beats MORI — holds at scale?"** It's link-balancing-bound; should hold while the XGMI mesh is the bottleneck, but untested beyond 8 GPUs — fair caveat.
