# Cache-on-First-Touch / Local Tile Inbox (Agent 06)

Keep a **large consumer block grid** (good latency hiding) while making each remote A tile cross
XGMI **exactly once**. Achieved by splitting *gather* from *compute* into two concurrent kernels that
communicate through a **local HBM symmetric-heap tile inbox**.

This builds on `irisx/v4_astationary_kernel` (a-stationary). V3/V4 are NOT edited.

---

## 1. The problem this fixes

- **V3**: grid `(N/BN, M/BM)`; every output-tile block re-gathers its own A strip remotely → A
  crosses XGMI `N/BN`× more than necessary. Gather-bound (~459/478 µs).
- **V4 (a-stationary)**: one block owns `NSUB` N-subtiles and reuses a single gathered A tile across
  them → A crosses `N/(NSUB·BN)`× instead of `N/BN`×. Better, but the **same** remote
  `A[m_tile,k_tile]` is still re-gathered by **every distinct block on that m-band**. To hide
  latency we *want* many blocks — directly in tension with "read each tile once".
- **Cache-on-first-touch**: a tile is gathered **once** by a producer into local HBM; *all* consumer
  blocks that need it read it locally. Grid size is now **decoupled** from XGMI traffic.

Total XGMI A traffic drops to the theoretical minimum: `num_m_tiles · num_k_tiles` tiles, each read
once, regardless of N or how many consumer blocks exist.

---

## 2. Two-kernel protocol (the implemented prototype)

Both kernels run on the **consumer rank**. Only the producer reads the remote rank over IRIS.

### Producer (`cft_producer`, modest grid)
```
loop:
  idx = atomic_fetch_add(cursor, 1)            # device scope, relaxed — dispense a task
  if idx >= num_tasks: break                   # pool drained → exit
  if !CAS(claim[idx], FREE -> TAKEN): continue # lost race → someone else gathers it
  (m_tile, k_tile) = decode(idx)
  gather_dequant_A_tile(...)                    # fp8 + scales from src_rank over IRIS, ONCE
  copy swizzled shared tile -> inbox[idx]       # local HBM, byte-for-byte, ST_A swizzle
  fence(system, release)                        # publish inbox bytes before the flag
  atomic_store(ready[idx], (gen<<1)|1, release) # generation-tagged ready flag
```

### Consumer (`cft_consumer`, FULL `(N/N_PER_BLOCK, M/BM)` grid, a-stationary N grouping)
```
for k_tile in 0..K/BK:
  slot = m_tile * num_k_tiles + k_tile
  spin: while atomic_load(ready[slot], acquire) != (gen<<1)|1   # acquire pairs w/ producer release
  load A from inbox[slot]  (LOCAL HBM/L2)        # ZERO remote reads
  load NSUB local B subtiles
  for sub in 0..NSUB: mma_ABt(C[sub], A, B[sub])
store C
```

The acquire on the consumer load pairs with the producer's release store + system fence, so the
inbox bytes are guaranteed visible once the flag reads READY. This is exactly the canonical IRIS
release/acquire protocol from `examples/01_message_passing/message_passing.hip`, with
`memory_scope_system` because the two **separate kernels** communicate through HBM and need
cross-kernel visibility.

---

## 3. Flag & cache-key scheme

**Flag encoding (anti-alias, generation-tagged)** — `tile_inbox_abi.h`:
- `EMPTY = 0`
- `READY(gen) = (gen << 1) | 1`

Low bit = valid; high bits = generation. A waiter for generation `g` requires *exactly* `READY(g)`.
A leftover `READY(g-1) = ((g-1)<<1)|1` is a **different integer**, so it can never satisfy a `g`
waiter — even if the per-generation reset were skipped. `gen` starts at 1 (so `READY` is always `≠
EMPTY`).

**Claim word** — `FREE=0`, `TAKEN=1`. `fetch_add(cursor)` bounds the number of indices handed out;
the `CAS(FREE→TAKEN)` guarantees that even under a race exactly one producer gathers each task. The
cursor alone suffices when each index is dispensed once; the CAS is belt-and-suspenders and makes the
persistent variant (cursor wraps) correct by construction.

**Cache key** — `cft_tile_key { generation, expert_id, m_tile, k_tile, src_rank, src_segment }`
uniquely identifies a remote tile so it is gathered once. In the two-kernel prototype one launch
processes one expert's tile space, so the **slot index** is just
`slot = m_tile * num_k_tiles + k_tile`; `expert_id/generation/src_rank/src_segment` are carried for
correctness/debug and feed the slot **hash** in the persistent multi-expert variant (§5).

---

## 4. Deadlock-avoidance analysis

The consumer **spins** on flags the producer must write. Two independent guarantees prevent deadlock:

1. **CU reservation via TWO separate launches.** Producer and consumer are launched as **two
   distinct grid launches on two non-blocking streams** (`dispatch_cft`), producer **first**. The
   producer's blocks are scheduled and reserve their CUs independently of the consumer grid, so the
   full consumer grid can **never** occupy all CUs and starve the producer before it runs. This is
   the core reason for two kernels rather than one mega-kernel: a single launch where consumer blocks
   could be co-scheduled ahead of producer blocks risks a classic occupancy deadlock (all CUs full of
   spinning consumers, no room for producers). Separate launches make CU allocation a scheduler-level
   guarantee, not a within-kernel gamble.

2. **Every awaited flag is eventually written.** The cursor `fetch_add` hands out **every** index in
   `[0, num_tasks)` exactly once across all producer blocks; the CAS ensures exactly one producer
   gathers+signals each index; producers loop until the cursor drains (`idx >= num_tasks`). Therefore
   every `slot` a consumer can await is guaranteed to receive its `READY(gen)` store. No consumer can
   wait on a slot no producer will fill.

Combined: producers always have CUs to make progress (1), and they collectively write every flag a
consumer waits on (2) ⇒ no deadlock and no livelock. (Spin uses `memory_order_acquire` loads; on
CDNA4 this is a plain global load loop, fine for the modest slot counts here.)

**Liveness caveat / risk.** If `num_producer_blocks` is set so high that the producer grid alone
exceeds device occupancy *and* HIP does not guarantee forward progress of all blocks, a producer
block could in principle be unscheduled. Keep `num_producer_blocks` ≤ the CU count (default 16) so
all producer blocks are co-resident. This is a tuning knob, not a correctness hazard, given (2).

---

## 5. One-kernel persistent variant (DESIGN SKETCH — not implemented here)

A single persistent launch with **fixed producer blocks** and **fixed consumer blocks**:

```
grid = num_producer_blocks + num_consumer_blocks   # all co-resident, persistent
if blockIdx < num_producer_blocks:   role = PRODUCER
else:                                role = CONSUMER
```

- **Producer blocks** loop forever: pull a task from a **global work queue** (the same
  `fetch_add(cursor)` + CAS claim), gather→inbox→release flag, repeat until the queue for the current
  generation is drained, then wait at a generation barrier.
- **Consumer blocks** loop over their assigned output tiles, acquire-spinning the per-slot flags,
  exactly as in §2.
- **Generation rollover.** A global `generation` counter advances per step. Flags use `READY(gen)`;
  on rollover the producer resets `ready/claim/cursor` for the next gen behind a grid-wide barrier
  (or uses a 2-slot ping-pong: `slot_base = (gen & 1) * num_slots`). Because `gen` is folded into the
  flag value, a single stale `READY(gen-1)` never aliases — the reset can even be lazy. `gen` must
  stay `< 2^31` (low bit is the valid bit); with one increment per decode step that is ~10^9 steps,
  effectively unbounded, but a modular `gen' = 1 + (gen % MAXGEN)` with a barrier'd flag-array clear
  every `MAXGEN` makes it truly safe.

**Why we shipped two kernels first, not the persistent one.** The persistent variant must hand-manage
CU partitioning *within one launch* (producer vs consumer blocks co-resident without one starving the
other) and forward-progress of spinning consumer blocks while producers run — both are fragile on HIP
without guaranteed block forward progress. The **two-launch** version delegates CU partitioning to the
scheduler (§4.1), which is far more robust for a first correct prototype. The persistent variant is
the latency-optimal follow-up once two-kernel correctness + speedup are confirmed (it removes the
second launch's overhead and lets producers prefetch the next generation while consumers finish the
current one).

---

## 6. Tile-cache storage: where does the inbox live?

| Option | Latency to consumer | Capacity | Coherence/effort | Verdict |
|---|---|---|---|---|
| **Local HBM** (symmetric heap) | ~HBM (then L2-cached on reuse) | large (GBs) | producer write + `fence(system)` + flag; consumer acquire-load | **chosen** — simple, correct, plenty of capacity for `num_m_tiles·num_k_tiles` BM×BK bf16 tiles |
| **LLC/L2-assisted** | best on reuse (tile stays hot) | small (MBs) | rely on L2 keeping recently-written tiles resident; no explicit API to *pin* | use as a *bonus*: writing to local HBM naturally populates L2, and the many consumer blocks reading the same slot hit L2 — we get this **for free** on top of the HBM inbox |
| **Symmetric heap (remote-readable)** | remote reads = XGMI (defeats the point for the consumer) | large | needed only if a *different rank's* consumers must read the cached tile | **not** for same-rank consumers; relevant only for a cross-rank shared cache (future) |

**Decision:** inbox in **local HBM on the consumer rank's IRIS symmetric heap**. It is allocated on
the heap purely for symmetric allocation order / IRIS pointer translation convenience; the consumer
reads it as **plain local memory** (no IRIS translation, no XGMI). The **L2/LLC win is implicit**:
the first consumer block to touch a slot pulls it from HBM into L2, and the remaining `N/N_PER_BLOCK`
blocks on that m-band hit it in L2 — this is precisely the "large grid shares one tile cheaply"
property we wanted. Sizing: `num_slots · BM · BK · 2` bytes; e.g. M=1024,K=7168,BM=BK=64 ⇒
16·112·64·64·2 ≈ 117 MB — comfortably within HBM, and the *hot working set* per m-band is just the
`num_k_tiles` tiles a consumer block streams, which L2 handles well.

---

## 7. Honest comparison & limits

- The win is **XGMI traffic** (each A tile crossed once) — compare against Agent 01's B1
  (gather-once-then-local-GEMM) and B2 (MORI+AITER), **not** the weak V3-direct-pull baseline.
- Extra cost vs V4: one HBM round-trip for A (producer writes, consumer reads) and the flag
  synchronization. This pays off when redundant remote gather dominates (large N, many m-band
  blocks) and may be a wash at tiny M/N. Record both wins and losses.
- Producer/consumer co-residency and spin overhead are the main risks; `num_producer_blocks` is the
  tuning knob (§4).
