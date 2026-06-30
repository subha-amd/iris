/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * route_abi.h  -- finalized IRISX MoE dispatch/route ABI (R1-0528, EP8, gfx950).
 * Authoritative struct definitions for PRODUCTION_ABI.md. Plain C, host+device safe
 * (no STL, fixed-width ints). Include from .cpp/.hip and from CPU tools.
 *
 * Constants and the scale-layout trap are documented in PRODUCTION_ABI.md (Agent 00).
 */
#ifndef IRISX_ROUTE_ABI_H
#define IRISX_ROUTE_ABI_H

#ifdef __cplusplus
extern "C" {
#endif

/* -------- problem constants (binding; see AGENT_COMMON 4 / PRODUCTION_ABI 0) -------- */
#define ROUTE_EP_SIZE          8
#define ROUTE_N_GLOBAL_EXPERTS 256
#define ROUTE_N_LOCAL_EXPERTS  32
#define ROUTE_TOP_K            8
#define ROUTE_HIDDEN           7168
#define ROUTE_GROUP            128
#define ROUTE_N_GROUPS         56        /* ROUTE_HIDDEN / ROUTE_GROUP */
#define ROUTE_FP8_E4M3_MAX_OCP 448.0f    /* gfx950 OCP e4m3 */
#define ROUTE_FP8_E4M3_MAX_FNUZ 240.0f   /* gfx942 fnuz (guard only) */

/* fc1 fused gate||up (g1u1) and fc2 down shapes (PRODUCTION_ABI 4). */
#define ROUTE_FC1_N            4096      /* gate(2048) || up(2048); SiLU(gate)*up -> 2048 */
#define ROUTE_FC1_K            7168      /* == ROUTE_HIDDEN */
#define ROUTE_FC2_N            7168
#define ROUTE_FC2_K            2048

/* -------- self-describing header -------- */
#define ROUTE_ABI_MAGIC   0x52415831u    /* "RAX1" little-endian */
#define ROUTE_ABI_VERSION 1u

/* scale_layout enum */
#define ROUTE_SCALE_TOKEN_MAJOR 0        /* IRISX native: scale[token*NG + g] (the trap) */
#define ROUTE_SCALE_GROUP_MAJOR 1        /* production fmoe: scale[g*M_pad + token]      */

typedef struct route_params {
    unsigned int  magic;        /* ROUTE_ABI_MAGIC */
    unsigned int  version;      /* ROUTE_ABI_VERSION */
    int           ep_size;      /* 8 */
    int           my_rank;      /* [0,ep_size) */
    int           n_local_experts; /* 32 */
    int           top_k;        /* 8 */
    int           hidden;       /* 7168 */
    int           group;        /* 128 */
    int           n_groups;     /* 56 */
    int           fp8_is_ocp;   /* 1=e4m3fn/448 (gfx950); 0=e4m3fnuz/240 (gfx942) */
    int           scale_layout; /* ROUTE_SCALE_TOKEN_MAJOR | ROUTE_SCALE_GROUP_MAJOR */
    int           m_pad;        /* token-dim pad of group-major scale; 0 = unknown [NEEDS-NODE] */
    int           total_packed_rows; /* == expert_offsets[n_local_experts] */
    int           t_local;      /* tokens this rank dispatched */
} route_params;

/* -------- inherited from AGENT_COMMON 3 (unchanged) -------- */

/* One contiguous run of rows for one expert coming from one source rank. */
typedef struct route_segment {
    int expert_id;       /* local expert index [0,32) */
    int src_rank;        /* owning rank of the activation rows [0,8) */
    int src_row_begin;   /* first row in the source rank's activation buffer */
    int dst_row_begin;   /* first row in this expert's packed output region */
    int row_count;       /* number of contiguous rows */
} route_segment;

/* Flattened GEMM work unit (Agent 02 grouped scheduler). */
typedef struct expert_task {
    int local_expert;
    int m_tile_begin;    /* row offset within the expert's region (multiple of BM) */
    int valid_rows;      /* <= BM; for tail masking */
    int n_superblock;    /* which NSUB-wide N panel */
    int segment_begin;   /* index into route_segment[] for this tile */
    int segment_count;
} expert_task;

/* expert_offsets[33] (prefix sum) and rows_per_expert[32] are plain int arrays;
 * size macros provided for allocators. */
#define ROUTE_EXPERT_OFFSETS_LEN (ROUTE_N_LOCAL_EXPERTS + 1)  /* 33 */
#define ROUTE_ROWS_PER_EXPERT_LEN ROUTE_N_LOCAL_EXPERTS       /* 32 */

/* -------- Agent-00 additions (PRODUCTION_ABI 3.2) -------- */

/* packed-row -> original token. One entry per packed (GEMM output) row, in packed-row order.
 * The inverse of V1's route_slot[token][topk]; combine/EpCombine walks this. */
typedef struct route_reverse {
    int   src_rank;      /* rank owning the original activation row [0,8) */
    int   src_token;     /* original token index on src_rank [0, t_local) */
    int   topk_slot;     /* which of the token's top-k picks this row is [0,8) */
    float route_weight;  /* softmax gate weight; combine accumulates weight * out_row */
} route_reverse;

/* -------- route-capture instrumentation record (ROUTE_CAPTURE_SCHEMA.md 1) -------- */
typedef struct route_capture_record {
    unsigned int magic;          /* ROUTE_ABI_MAGIC */
    unsigned int version;        /* ROUTE_ABI_VERSION */
    int   rank;
    int   layer;
    long long step;              /* monotonic forward-step counter */
    int   t_local;
    int   top_k;
    int   n_local_experts;
    int   total_assignments;     /* t_local * top_k */
    int   remote_assignments;    /* dst_rank != rank */
    int   dropped_assignments;   /* past capacity; 0 if uncapped */
    int   max_expert_load;       /* max rows_per_expert */
    int   pad_;                  /* keep 8B alignment */
    int   rows_per_expert[ROUTE_ROWS_PER_EXPERT_LEN];   /* 32 */
    int   expert_offsets[ROUTE_EXPERT_OFFSETS_LEN];     /* 33 */
} route_capture_record;

/* binary dump file header (ROUTE_CAPTURE_SCHEMA.md 3.2) */
typedef struct route_capture_file_header {
    unsigned int magic;          /* ROUTE_ABI_MAGIC */
    unsigned int version;        /* ROUTE_ABI_VERSION */
    unsigned int record_count;
    unsigned int flags;          /* bit0: FULL arrays present */
} route_capture_file_header;

/* FULL-array kinds (prefix each variable array in the FULL arena) */
#define ROUTE_FULL_KIND_TOPK_IDS      0u
#define ROUTE_FULL_KIND_TOPK_WEIGHTS  1u
#define ROUTE_FULL_KIND_ROUTE_REVERSE 2u

typedef struct route_full_array_prefix {
    unsigned int record_index;
    unsigned int kind;           /* ROUTE_FULL_KIND_* */
    unsigned int len;            /* element count */
} route_full_array_prefix;

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* IRISX_ROUTE_ABI_H */
