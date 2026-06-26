// tile_inbox_abi.h
// ------------------------------------------------------------------------------------------------
// Agent 06 — cache-on-first-touch / local tile inbox ABI.
//
// Shared host+device definitions for the TWO-KERNEL cache-first-touch MoE expert-GEMM prototype.
//
// Big picture (why this exists):
//   V4 (a-stationary) already amortizes the cross-XGMI A gather by NSUB (reuse one A tile across
//   NSUB N-subtiles inside ONE block).  But the SAME remote A[m_tile,k_tile] is STILL re-gathered
//   by every DISTINCT consumer block that owns that (m_tile) row band — and to keep a large grid
//   for latency hiding we WANT many such blocks.  Large grid <-> single remote read are in tension.
//
//   Resolution: split producer (gather) from consumer (compute) into TWO concurrent kernels that
//   communicate through a LOCAL HBM symmetric-heap "tile inbox":
//     * PRODUCER claims each DISTINCT (m_tile,k_tile) A-task exactly once (fetch_add cursor + CAS
//       claim word), gathers fp8 tile + scales ONCE from the src rank over IRIS, dequants to bf16,
//       writes it into a local inbox slot, then atomic_store(release) a generation-tagged ready
//       flag.
//     * CONSUMER keeps the FULL (M/BM x N/BN) grid; for each (m_tile,k_tile) it needs, it
//       atomic_load(acquire)-spins on that slot's flag, then reads A from LOCAL HBM/L2 and MFMAs.
//       Consumers issue ZERO remote reads.
//
//   So each remote A tile crosses XGMI exactly ONCE (producer), while N/BN consumer blocks share it
//   from local HBM.  Grid size is decoupled from XGMI traffic.
//
// This header is the contract between kernel.cpp (device) and example.py (host launcher).
// No secrets; no real hostnames/paths.
// ------------------------------------------------------------------------------------------------
#pragma once
#include <cstdint>

// ------------------------------------------------------------------------------------------------
// Tile geometry (must match the GEMM tiling in kernel.cpp). BM/BK chosen to match V4 defaults.
// ------------------------------------------------------------------------------------------------
#ifndef CFT_BM
#define CFT_BM 64
#endif
#ifndef CFT_BN
#define CFT_BN 64
#endif
#ifndef CFT_BK
#define CFT_BK 64
#endif

// ------------------------------------------------------------------------------------------------
// Flag encoding (anti-alias, generation-tagged).
//
//   EMPTY  = 0                      slot has never been filled for the current generation
//   READY  = (gen << 1) | 1         slot filled & visible for generation `gen`
//
// The low bit is the "valid" bit; the high bits carry the generation. A waiter for generation `g`
// requires EXACTLY ready_flag(g); a stale READY from generation g-1 is a DIFFERENT integer
// ((g-1)<<1)|1 and therefore can NEVER satisfy the g waiter. Generation 0 is reserved-as-unused so
// that EMPTY(0) is never confused with a real ready value (ready_flag(0) == 1 != 0, still distinct,
// but we start live generations at 1 to keep the invariant obvious).
//
// gen must stay < 2^31 within a single launch (trivially true: one generation per launch here).
// Rollover is only a concern for the persistent one-kernel variant (see CACHE_FIRST_TOUCH.md).
// ------------------------------------------------------------------------------------------------
#define CFT_FLAG_EMPTY 0

__host__ __device__ inline int cft_ready_flag(int gen) { return (gen << 1) | 1; }

// ------------------------------------------------------------------------------------------------
// Claim-word encoding for the producer task pool (one int32 per A-task).
//   CFT_CLAIM_FREE = 0   task not yet claimed
//   CFT_CLAIM_TAKEN = 1  some producer block won the CAS and is gathering it
// A producer first grabs a candidate index via fetch_add on a global cursor, then CAS-claims the
// task word FREE->TAKEN. The fetch_add cursor bounds total CAS attempts; the CAS guarantees that
// even if two producers race the same index, exactly one gathers it. (Cursor alone is enough when
// each index is handed out once; the CAS is belt-and-suspenders against any cursor reuse and makes
// the persistent variant — where the cursor wraps — correct by construction.)
// ------------------------------------------------------------------------------------------------
#define CFT_CLAIM_FREE  0
#define CFT_CLAIM_TAKEN 1

// ------------------------------------------------------------------------------------------------
// Cache key for an A-tile task. Identifies a UNIQUE remote tile so it is gathered exactly once.
//
//   generation : monotonically increasing epoch (one decode/prefill step). Tags the ready flag.
//   expert_id  : local expert [0,32). Different experts have disjoint row regions.
//   m_tile     : A row tile index within the expert's packed region (row band = m_tile*BM).
//   k_tile     : A column tile index along K (col band = k_tile*BK).
//   src_rank   : owning rank of the activation rows [0,8).
//   src_segment: route_segment index (AGENT_COMMON ABI) that this (m_tile) maps into; disambiguates
//                tiles whose rows come from different contiguous source runs.
//
// Slot index (into the inbox / flag arrays) is purely (m_tile,k_tile) within one expert+generation:
//      slot = m_tile * num_k_tiles + k_tile
// expert_id/generation/src_rank/src_segment are carried for correctness + debugging but do NOT
// change the slot index, because one launch processes ONE expert's tile space at a time in this
// two-kernel prototype (the persistent variant adds expert_id into the slot hash; see the .md).
// ------------------------------------------------------------------------------------------------
struct cft_tile_key {
    int generation;
    int expert_id;
    int m_tile;
    int k_tile;
    int src_rank;
    int src_segment;
};

__host__ __device__ inline int cft_slot_index(int m_tile, int k_tile, int num_k_tiles) {
    return m_tile * num_k_tiles + k_tile;
}

// ------------------------------------------------------------------------------------------------
// Inbox layout. All buffers live on the LOCAL (consumer) rank's IRIS symmetric heap. Producers and
// consumers run on the SAME (consumer) rank in this prototype, so these are plain local pointers;
// only the fp8 A source + scales are remote (read by the producer over IRIS from src_rank).
//
//   inbox_A   : [num_m_tiles * num_k_tiles] dequantized bf16 A tiles, BM*BK bf16 each, laid out
//               contiguously and tile-swizzled exactly like ST_A so the consumer's `load(frag,...)`
//               can ingest a slot directly. Size = num_slots * BM * BK * sizeof(bf16).
//   ready     : [num_slots] int32 flags, EMPTY/READY(gen) per cft_ready_flag().
//   claim     : [num_tasks] int32 claim words, FREE/TAKEN. num_tasks == num_slots here.
//   cursor    : single int32, producer fetch_add task dispenser. Starts at 0.
//
// Producer count vs consumer count: separate launches reserve CUs independently (see deadlock
// analysis in CACHE_FIRST_TOUCH.md) so consumers can never starve producers of CUs.
// ------------------------------------------------------------------------------------------------
struct cft_inbox_desc {
    int   num_m_tiles;
    int   num_k_tiles;
    int   num_slots;     // == num_m_tiles * num_k_tiles
    int   generation;    // current epoch; flags written as cft_ready_flag(generation)
};
