# Speaker notes — HipKittens × IRIS, Meeting 3

Deck is `meeting3_talk.pdf`, ~12–15 min then discussion. Quick map of where I am so it's clear what's real: QuantTile is built and validated, MXFP4 is ported with real region numbers, the comm wrapper (tile_reduce_scatter) is written and about to be measured, and the prefill overlap is the honest not-yet. I'll call out measured vs still-building as I go.

---

## Slide 1 — Title
Just say what this is. This is a progress update, not a finished thing. Last time I showed the fused MoE region — today I want to go through what the profiling actually told me, the QuantTile primitive I built and validated, where MXFP4 landed, and the communication wrapper I started. Some of it's measured and done, some is still building, I'll flag it.

## Slide 2 — Since last time (spend a bit here, there are new folks)
Since some people are new, let me back up and say what the kernel even is and what I did last time.

The thing we're optimizing is the MoE expert region in DeepSeek-R1. After the router runs, each token picks its top-8 experts out of 256, so each token becomes up to 8 (token, expert) pairs — I call each of those a routed row, it's basically a copy of the token's hidden vector headed to one specific expert. We run expert-parallel, EP8, so there are 32 experts sitting on each of the 8 GPUs. Every routed row has to get to the GPU that owns its expert, run that expert's FFN — fc1, then SiLU, then fc2 — and then come back to the GPU the token started on.

The production way to do this (aiter + MORI) is 7 separate kernels: EpDispatch to send the rows out, then moe_sorting twice, then dynamic_quant, then fmoe (that's the good one — it does fc1 + SiLU + fc2 fused), then moe_sum, then EpCombine to send the results home. The expert GEMM itself is fine. The problem is all the boundary kernels around it.

What I did last time was collapse that into basically 3 stages: gather_pack in IRIS → the two grouped GEMMs with silu_quant in between, all in HipKittens → combine_pull in IRIS. Most of the win is in the two comm kernels, so let me explain those:

- **gather_pack.** In the unfused path the *source* GPU drives the transfer — it pushes its tokens out, and they land on the destination in whatever order they showed up, interleaved by whoever sent when, not grouped by expert. That's the whole reason the destination then has to run 2 sort passes — to get the rows into expert-major order before the GEMM. Ours flips it around: the *destination*, the expert-owner GPU, drives it. It reaches out and reads exactly the rows it needs with `ctx.load(src_rank)` over XGMI, and because the destination controls where each row lands, it drops them straight into expert-major slots as it reads them. So there's no sort pass at all — the communication produces the layout the GEMM wants. On top of that I quantize the activations to fp8 on the producing GPU *before* dispatch, so the gather only moves half the XGMI bytes, and that also kills the dynamic_quant kernel.
- **combine_pull.** Just the reverse — each destination token pulls its up-to-8 expert outputs from local memory, sums them in fp32, and writes one row home.

Net result was 1.56× on prefill. Decode barely moved, and I said at the time that's because decode is weight-bound — you're streaming all 32 experts' weights every single step no matter how few tokens you have, so killing the boundary kernels doesn't buy you much there.

So this week: instead of arguing from priors about which abstraction to build, I measured. And the two directions I kept going back and forth on turned out to be for two different regimes — decode wants fewer weight bytes, prefill wants the all-reduce fused. Only one of them is landable right now, which is the rest of the talk.

## Slide 3 — What I profiled, and what came out
Setup first, because people will poke at this. TP4, bf16, on the MI350X. My baseline is what production does when it can't fuse — a full 256-CU torch matmul of each rank's K-slice, then a separate RCCL all-reduce. Median of 30, and I take the max over the 4 ranks because the slowest rank is what gates the collective.

Read one row so it's not just a wall of numbers: take the down-proj, 8192 by 7168 by 18432. The GEMM is 0.565 ms, but the all-reduce sitting right after it is 1.148 — more than twice the GEMM. So the all-reduce is 70% of that GEMM+all-reduce.

The point: across all three shapes the all-reduce is 59 to 85% of the GEMM+AR. That's the thing I had backwards. I'd assumed comm was small next to the matmul — it's not, it's the expensive half. So if you could hide the GEMM under the all-reduce — reduce tiles as you produce them — the ceiling is 1.4 to 1.7×.

Decode is a totally separate story though. Under DP-attention there's no TP all-reduce at all, so there's nothing to fuse — decode just streams the experts' weights every step. So decode and prefill genuinely need different fixes, which is why the next slides split.

## Slide 4 — Where HipKittens is today on quantized loads
Two things about HK. One, it hard-codes a separate GEMM per quant format — fp8fp32 and mxfp8 are two different kernel bodies. Two, and this is the one that matters for speed: there's no fast direct HBM→register fp8 load. The fast `buffer_load` path is a hard static_assert that rejects fp8 for the direct-to-register case, so HK's fp8 kernels stage through shared — global to shared, shared to register, then the fp8 MFMA. I measured that staged path at about 2.6 TB/s on the skinny BM=16 decode tile.

Be careful how I say this, because someone will push on it: it's *not* "HK has slow fp8 loads." global→shared actually does support fp8 — it's specifically global→*register* that forbids it. And staging through shared is the correct, fast choice for a big square GEMM tile, because you reuse each loaded K-tile across a bunch of MFMAs, so the load cost amortizes — that's how every good GEMM works. The 2.6 TB/s only looks slow in the decode case, where the tile is skinny (BM=16) and there's basically no reuse — so the shared-memory hop is pure overhead you can't amortize. Different regime, not a knock on HK's mainline path. But it's exactly the regime MoE decode lives in.

## Slide 5 — What I built, the sat trick
So here's the workaround. Keep the weights compressed in HBM — fp8 is half the bf16 bytes, fp4 a quarter — but load them through the fast bf16 register path anyway. I reinterpret the packed bytes as a half- or quarter-width bf16 tile, unpack in-register, and hand a clean bf16 tile to a normal MFMA. No shared-memory hop. The one piece of real engineering is that I pre-swizzle the weights offline so the reinterpreted bytes land in exactly the MFMA fragment order — that's what makes the reinterpret correct. For fp8 the in-register step is just applying the per-block scale; for fp4 it's one hardware instruction, cvt_scalef32_pk_bf16_fp4, that converts and folds the scale in.

Numbers, with the honesty flag: on thor-4, sat fp8 hits 3.97 TB/s vs HK's 2.6, so about 1.5×, just from skipping the shared hop. Flag: this table is the thor-4 anchor, which is an MI355X — a faster box than the MI350X the rest of the deck runs on — and I didn't re-verify these exact numbers this week. The ratios hold on both nodes though, and that's the actual claim. sat fp8 also beats a plain bf16 load by 1.46×, and fp4 is roughly another 1.6× on top of fp8 because it carries half the bytes.

If someone asks "is this a real contribution": the reinterpret-a-narrow-type-as-a-wider-one idea isn't novel in the abstract. The useful part is the offline pre-swizzle into MFMA-fragment order plus packaging it for the memory-bound, low-reuse, skinny decode tile — which is the regime HK's reuse-oriented staging doesn't cover, and it only gets more valuable at fp4. That's the piece worth writing up / pushing upstream.

## Slide 6 — QuantTile, the primitive and the validation
QuantTile is basically me abstracting a thing I'd already built twice. My decode path had two bodies — grouped_expert_gemm_decode_fp8_sat and grouped_expert_gemm_decode_mxfp4_sat — and they're about 90% the same code. They only differ in three spots: the packed-tile width, the scale layout, and the in-register unpack. That's a copy-paste fork that's going to rot the second I add a third format.

So QuantTile makes the format a compile-time property of the tile — one descriptor carries the width, the scale layout, and the unpack — and then one GEMM body serves both fp8 and mxfp4 instead of the per-format fork HK ships. There's no HK tile today that can say "I'm fp4 but I turn into bf16 at MFMA time." That's the gap.

The point isn't a speedup — the speedups already live in the two kernels. The point is proving the abstraction is free. Same node, identical inputs: the unified fp8 body comes out within 0.07% of the hand-written one, 2.874 TB/s, a wash. mxfp4 keeps its 1.63× over fp8. And the output matches the old kernels to 5 decimal places. So the descriptor is a genuine compile-time policy, not a dispatch tax. That's the QuantTile v0 result — the format axis, done and measured. And it's aimed straight at decode, which is a weight-memory wall — ~80% of that region is just streaming weights, so the only lever is fewer weight bytes, which is exactly what a compressed tile carries.

## Slide 7 — MXFP4, porting the fused region to 4-bit weights
The model is amd/DeepSeek-R1-MXFP4 — OCP MXFP4, W4A4, so fp4 weights and activations, group size 32, an E8M0 scale per 32-element K-block. My Route-1 keeps the activations in bf16 — weight-only fp4. That's both more accurate and the right decode call, because decode is weight-bound, so what matters is the weight bytes, not the activation bytes.

Numbers at the full fused decode region — 8×MI350, same node, 512 tokens: bf16 is 881 µs, fp8 is 579, MXFP4 is 521. So MXFP4 is 1.11× over fp8, 1.69× over bf16. The GEMM alone is a bigger win, 1.16 and 1.96×, but the region dilutes that because the activation dequant and the gather/act/combine boundaries are shared cost that fp4 doesn't touch. FFN correctness passes at RMS 0.018.

Nice part: the GEMM already exists — I reuse HK's mxfp8 8-wave scaled-MFMA body, which already eats the MX per-32-block E8M0 format, and wrap it with my expert-grouping and the IRIS gather/combine. So the work is the MX wiring, not a new MFMA path. Honest caveat: I don't have an fp4-vs-SOTA number I trust yet, because aiter's a4w4 EP path is untuned and buggy for our shape. The real baseline is the reproducible SGLang R1-MXFP4 end-to-end run, which I'm setting up. And note fp4's activation-quant throughput win is really a prefill effect — for decode, weight-only is the one that matters.

## Slide 8 — The other lever, fusing the prefill all-reduce
The prefill idea is to fold the TP4 all-reduce into the GEMM — reduce output tiles as they're produced, so the short GEMM hides under the long all-reduce. Slide 3 says the ceiling's real, 1.4 to 1.7×. It's the tiles-before-reduce thing — how many output tiles you pile up before you fire a reduce — as one knob.

But I want to be straight: the substrate isn't there. The IRIS fused-collective examples I actually ran are 4.7× to 280× slower than plain unfused. They're pedagogical — a slow Triton GEMM plus a naive in-kernel all-reduce. To make this real I'd need a proper HK producer/consumer GEMM body at torch rate whose epilogue emits the tile transfers, plus an in-kernel reduce-scatter at RCCL bandwidth. Both are a lot of work, and it's prefill-only. So this is the harder direction and it's not done.

## Slide 9 — The communication abstraction, tile_reduce_scatter
Right now the combine hand-writes a loop of iris.store calls. Each destination token pulls its up-to-8 contributing rows, reduces them locally in fp32 — weighted, no cross-rank atomics — and writes one bf16 row home, over a round-robin of tiles across the 8 destination links that I hand-rolled by hand.

The wrapper lifts that into a reusable primitive. The author declares a TileTransferSet — the destination map, a CSR grouping of each tile's contributing rows, the payload, and the IRIS handle — and calls tile_reduce_scatter. The library does the fp32 reduce, the store home with a local short-circuit, and it owns the link ordering. So you stop hand-writing iris.load and iris.store — you say "reduce-scatter these tiles" and the library lowers it.

Say this part clearly so it doesn't get read as the old traffic-shaping thing: the library owning the schedule is about correctness — you can't accidentally pick the bad order — it's not a speedup pitch. Same shape as QuantTile: QuantTile made the tile carry its format, this makes the tile carry where it lives. Declare the intent, the library lowers it. I'm doing bulk-synchronous first as a low-risk drop-in, and the same declaration can later go inside the GEMM's tile loop for the overlap version. There's a store-granularity knob for the remote transaction width.

Status: the code's written and it compiles — I refactored the combine onto it and kept the original body for a same-build A/B. Not profiled yet, the GPU's busy on the QuantTile-vs-SOTA run. The bar is the same as QuantTile — zero cost — so it should reproduce the June-30 combine, 386 µs, below MORI's 398. That measurement's pending.

If novelty comes up: this isn't the first MoE communication — DeepEP and NCCL-EP exist. The honest claim is a general tile-level wrapper over IRIS RMA that's reusable across both the MoE gather/combine and the TP reduce-scatter/all-reduce, not "first MoE comm."

## Slide 10 — How the two fit together
Both QuantTile and the collective are the same move on the tile. QuantTile touches format, the collective touches residency. A tile that carries its format lowers to an in-register unpack plus a bf16 MFMA, and it goes at the decode weight wall. A tile that carries its residency lowers to an in-kernel reduce-scatter over IRIS, and it goes at the prefill all-reduce.

I don't want to oversell it though — I'm not claiming one cost model tunes both. Format and residency stay two separate problems. What they share is the tile descriptor, the way you declare it. That's the connection, and it's an honest one, not a grand unified theory.

## Slide 11 — Open questions for the experts
A few of these are past where I can answer from one node, so I want to actually bring them to the room. Walk the list and say what each is really asking:

- **Store completion** — what actually tells me a device-initiated store has landed on the remote GPU on ROCm/XGMI? My combine stores are fire-and-forget, and my earlier bandwidth probe came out issue-bound for exactly this reason, so I don't have a trustworthy remote-completion primitive.
- **Reduce-scatter algorithm** — for my TP4 message sizes on 8 fully-connected GPUs, which reduce-scatter is right — direct all-push, ring, recursive halving/doubling, or whatever RCCL picks — and where are the breakpoints?
- **Relay routing** — on this fabric, can a 2-hop relay through an idle GPU beat direct routing when the expert load is skewed, once you pay for the extra bytes and the sync?
- **Deadlock** — if I move to in-kernel arrival counters, producer signals and consumer waits, what keeps it deadlock-free when a producer block might not even be resident yet?
- **Scheduling model** — what's the right formal model for scheduling a finite wave of tiles with issue queues and uneven tile sizes?
- **Batching model** — is the simple tiles-before-reduce batching model even valid for IRIS stores over XGMI, or does the completion behavior need a different latency term?
- **Novelty framing** — honestly, what would a distributed-algorithms reviewer accept as the contribution here, vs MSCCL++, SCCL, NCCL-EP, DeepEP?

Close: that's where I'm at. QuantTile's done and validated, MXFP4's ported with real region numbers, the comm wrapper's written and about to be measured, and the prefill overlap is the honest not-yet. Happy to dig into any of the open questions.
