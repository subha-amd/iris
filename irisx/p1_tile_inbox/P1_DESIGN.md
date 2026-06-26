# P1 — Two-Kernel Copy-Once Tile Inbox (B0-GEMM consumer)

## Goal (from EXPERIMENT_LEDGER Gate-1)
B0 (local GEMM) ~164us; B1-copy (copy A once, then B0, SERIAL) ~291us = ~143us transfer + ~162us
compute; perfect-overlap floor = max(143,162) = ~162us; ceiling speedup over B1 ~= 1.80x.
A competitive design must hit ALL THREE at once: (1) ~1x remote A traffic, (2) B0-class GEMM
efficiency, (3) comm/compute overlap. P1 = copy A exactly once (producer) + overlap with the EXACT
B0 GEMM (consumer). Target: T_pipeline meaningfully below 291us toward ~162-200us.

## The one change vs Agent 06 cache_first_touch
Agent 06 already moves each A tile once and overlaps, but its CONSUMER is a low-occupancy
4-producer/4-consumer fused body (NSUB N-grouping, CONS_N=BN/4, occ ~2) — i.e. the V4 body, not B0.
That body is exactly what Gate-1 showed runs at ~44 TFLOP/s. **P1's consumer is the verbatim B0
kernel** (256x256x64, 8 warps, `__launch_bounds__(512,2)`, 2-stage shared double-buffer). The only
added device code is a single per-block acquire-spin before the K-loop. So
`compute_retention = B0_lat / consumer_compute_lat` should be ~1.

## Inbox = full dequantized bf16 A[M,K]
Instead of a swizzled per-tile staging area, the inbox is the **full dequantized bf16 A[M,K]
row-major buffer** — byte-identical to what B0's own `dequant_a_dense` preamble produces. The
producer performs B0's dequant preamble REMOTELY and ONCE; the consumer is then literally B0 reading
A from this local buffer. No swizzle reinterpretation, no ABI drift between producer and consumer.

## Producer/consumer protocol
- **Flags:** one `ready[band]` per `BM_PROD`(=64)-row band; a band spans the WHOLE K, so
  `ready[band]==(gen<<1)|1` means all K columns of those 64 rows are locally materialized.
  EMPTY=0; anti-stale generation tag (a stale READY(gen-1) is a different int).
- **Producer** (modest grid, own stream, launched first): `fetch_add(cursor)` dispenses a band index;
  `CAS(claim[band], FREE->TAKEN)` makes exactly one block materialize it; the block IRIS-loads the
  fp8 bytes (uint4, 16 bytes/thread) + per-128 fp32 scales for all K of those rows ONCE, dequants to
  bf16, writes row-major into the inbox; `fence_system(release)` then `atomic_store(release)` the
  ready flag. Copy amplification ~1.0x.
- **Consumer** = B0. Each B0 block owns a `B0_BM`(=256)-row x `B0_BN`(=256)-col output tile, i.e. 256
  A rows over all K = `B0_BM/BM_PROD`=4 producer bands. Thread 0 acquire-spins those 4 flags, then
  `__syncthreads()` (the happens-before that publishes the producer's A bytes to all threads), then
  runs the EXACT B0 inner loop reading A from the local inbox. Zero remote reads.

## Deadlock-avoidance argument
1. **CU reservation:** two SEPARATE launches on two non-blocking streams; the producer launches first
   and reserves its (few) CUs independently, so the large consumer grid can never occupy all CUs
   before the producer runs. No `hipStreamSynchronize` between the launches (that would serialize and
   defeat overlap, though it would still be correct).
2. **Liveness:** the producer cursor hands out EVERY band index exactly once; CAS guarantees exactly
   one block materializes+signals each band. There is NO consumer-to-consumer dependency — a consumer
   block only waits on `ready[band]`, never on another consumer block. Therefore even a single
   resident producer block drains the whole pool, and every flag a consumer awaits is eventually
   written. No cycle exists -> deadlock-free.
3. **Anti-stale + reset:** host bumps `gen` and zeroes ready/claim/cursor each step; a stale flag from
   gen-1 can never satisfy a gen waiter.

## Metrics
- Primary: `T_pipeline` (combined two-kernel wallclock, single timing loop; symmetric barriers).
- `compute_retention = B0_lat(164us) / consumer_compute_lat` (upper-bounded by `B0/T_pipeline`).
- `overlap_efficiency = (T_copy + T_gemm - T_pipeline) / min(T_copy, T_gemm)` using B1's measured
  T_copy~=143us, T_gemm~=162us.
- Correctness: RMS-rel vs the bf16 reference C = dequant(A) @ B^T (same as B0/B1); zero-sentinel
  (consumer rank's local A is zero -> a correct C proves the remote gather is real).

## Risks
- `INBOX = [M,K]` bf16 is M*K*2 bytes (1024*7168*2 = ~14.7 MB) — fine; heap sized 1024 MB.
- The producer's scalar fp8->bf16 dequant loop may be slower than the parallel `dequant_a_dense`; if
  the producer becomes the bottleneck (T_copy >> 162us) raise PROD_BLOCKS and/or vectorize the
  dequant. The gather itself is the same uint4 traffic as B1.
- If `PROD_BLOCKS` is too large it competes with consumer CUs; if too small the producer can't keep
  up with the consumer. Sweep PROD_BLOCKS in {8,16,32}.
- Tuning knob `BM_PROD` MUST match between kernel.cpp (`#define BM_PROD`) and example.py.
