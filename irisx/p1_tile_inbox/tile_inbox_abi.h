// tile_inbox_abi.h  (P1 — subset reused from Agent 06's cache_first_touch ABI)
// ------------------------------------------------------------------------------------------------
// Minimal, self-contained copy of the flag/claim encoding P1 needs, so this candidate dir mirrors to
// the node standalone (the build auto-discovers */kernel.cpp; the include must resolve within the
// candidate dir). Semantics are IDENTICAL to cache_first_touch/tile_inbox_abi.h.
//
//   ready=(gen<<1)|1, EMPTY=0  -> a stale READY(gen-1) is a DIFFERENT int and can never satisfy a gen
//   waiter (anti-stale generation tag). claim FREE/TAKEN guards exactly-once materialization.
// ------------------------------------------------------------------------------------------------
#pragma once
#include <cstdint>

#define CFT_FLAG_EMPTY  0
#define CFT_CLAIM_FREE  0
#define CFT_CLAIM_TAKEN 1

__host__ __device__ inline int cft_ready_flag(int gen) { return (gen << 1) | 1; }
