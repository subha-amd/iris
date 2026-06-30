// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// route_capture.hpp -- compile-time-gated route capture (ROUTE_CAPTURE_SCHEMA.md).
// At ROUTE_CAPTURE=0 every macro is empty -> the captured HIP graph is byte-identical to a
// baseline build. At >=1 it writes summary records into a pre-allocated device ring; at >=2
// it also stores route_reverse / topk weights. No device-side allocation, no host sync inside
// the captured region -> graph-safe (PRODUCTION_ABI 6).
//
// DESIGN-REVIEWED, COMPILE-UNTESTED locally (no compiler in sandbox). Main agent: syntax-check
// on node with HIP_VISIBLE_DEVICES="" before any GPU run.
#ifndef IRISX_ROUTE_CAPTURE_HPP
#define IRISX_ROUTE_CAPTURE_HPP

#include "route_abi.h"

#ifndef ROUTE_CAPTURE
#define ROUTE_CAPTURE 0   // 0=off, 1=summary, 2=full
#endif

#if ROUTE_CAPTURE == 0
// ---- OFF: all hooks vanish. No buffer, no branch, no graph-shape change. ----
#define ROUTE_CAPTURE_REVERSE(packed_row, src_rank, src_token, topk_slot, weight) ((void)0)
#define ROUTE_CAPTURE_SUMMARY(ring, rank, layer, step, send_counts, world, epr, t_local) ((void)0)

#else  // ROUTE_CAPTURE >= 1

#include <hip/hip_runtime.h>

// Pre-allocated device ring for summary records (pointer-stable for graph capture).
struct route_capture_ring {
    route_capture_record* records;   // [n_slots], allocated once at init
    route_reverse*        reverse;   // [max_packed_rows], FULL only (ROUTE_CAPTURE==2)
    unsigned int*         head;      // device atomic, next summary slot
    int                   n_slots;
    int                   max_reverse;
    int                   enable;    // runtime gate (ROUTE_CAPTURE_ENABLE); single predicated store
};

// Global handle (the host sets g_capture_ring once; nullptr-safe).
__device__ __host__ inline route_capture_ring*& route_capture_handle() {
    static route_capture_ring* h = nullptr;  // host side
    return h;
}

#if ROUTE_CAPTURE >= 2
// FULL: store one route_reverse entry. Device-side, predicated, no alloc.
__device__ inline void route_capture_reverse_impl(route_capture_ring* r, int packed_row,
                                                  int src_rank, int src_token,
                                                  int topk_slot, float weight) {
    if (!r || !r->enable || !r->reverse) return;
    if (packed_row < 0 || packed_row >= r->max_reverse) return;
    route_reverse e; e.src_rank = src_rank; e.src_token = src_token;
    e.topk_slot = topk_slot; e.route_weight = weight;
    r->reverse[packed_row] = e;
}
#define ROUTE_CAPTURE_REVERSE(packed_row, src_rank, src_token, topk_slot, weight) \
    route_capture_reverse_impl(route_capture_handle(), (packed_row), (src_rank), \
                               (src_token), (topk_slot), (weight))
#else
#define ROUTE_CAPTURE_REVERSE(packed_row, src_rank, src_token, topk_slot, weight) ((void)0)
#endif

// SUMMARY: device kernel that folds send_counts -> one route_capture_record into the ring.
// Launched 1 block; reads device-resident send_counts (no host sync) -> graph-safe.
__global__ inline void route_capture_summary_kernel(route_capture_ring r, int rank, int layer,
                                                    long long step, const int* send_counts,
                                                    int world, int epr, int t_local) {
    if (!r.enable || !r.records || threadIdx.x != 0 || blockIdx.x != 0) return;
    route_capture_record rec;
    rec.magic = ROUTE_ABI_MAGIC; rec.version = ROUTE_ABI_VERSION;
    rec.rank = rank; rec.layer = layer; rec.step = step;
    rec.t_local = t_local; rec.top_k = ROUTE_TOP_K; rec.n_local_experts = epr;
    rec.total_assignments = t_local * ROUTE_TOP_K;
    rec.pad_ = 0;
    int total = 0, maxload = 0;
    rec.expert_offsets[0] = 0;
    for (int e = 0; e < epr; ++e) {
        // rows for local expert e summed across all source ranks: send_counts[src*epr + e]
        int rows = 0;
        for (int s = 0; s < world; ++s) rows += send_counts[s * epr + e];
        rec.rows_per_expert[e] = rows;
        total += rows;
        if (rows > maxload) maxload = rows;
        rec.expert_offsets[e + 1] = total;
    }
    rec.max_expert_load = maxload;
    rec.remote_assignments = 0;   // filled by host (needs topk_ids); 0 here
    rec.dropped_assignments = 0;  // filled by host if capacity enforced
    unsigned int slot = atomicAdd(r.head, 1u) % (unsigned)r.n_slots;
    r.records[slot] = rec;
}
#define ROUTE_CAPTURE_SUMMARY(ring, rank, layer, step, send_counts, world, epr, t_local) \
    do { route_capture_ring* _r = route_capture_handle(); \
         if (_r) route_capture_summary_kernel<<<1,1,0,0>>>(*_r, (rank), (layer), \
                 (long long)(step), (send_counts), (world), (epr), (t_local)); } while (0)

#endif  // ROUTE_CAPTURE >= 1

#endif  // IRISX_ROUTE_CAPTURE_HPP
