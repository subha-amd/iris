// expert_pipeline_abi.h  (P2 — flag encoding + tile constant shared host/device)
// ------------------------------------------------------------------------------------------------
// Self-contained so the candidate dir mirrors to the node standalone. Same anti-stale generation tag
// as Agent 06's tile_inbox_abi: ready=(gen<<1)|1, EMPTY=0. A stale flag from gen-1 is a different int
// and can never satisfy a gen waiter; host bumps gen + resets each step.
// ------------------------------------------------------------------------------------------------
#pragma once
#include <cstdint>

#define EP_FLAG_EMPTY 0
__host__ __device__ inline int ep_ready_flag(int gen) { return (gen << 1) | 1; }

// BM tile height used by the multi-source gather inside the producer (per-segment resolution unit).
// Must divide the per-expert padded region cleanly (host pads padded_rows to a multiple of this).
#ifndef ep8_gather_BM
#define ep8_gather_BM 64
#endif
