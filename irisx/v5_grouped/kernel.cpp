// fmoe_fused_v5_grouped / kernel.cpp
// ------------------------------------------------------------------------------------------------
// V5 — GROUPED (all-experts-in-one-grid) fused MoE expert-GEMM.  Built on V4 (do NOT edit V4).
//
// V4 ran ONE expert per launch (fixed M,N; grid = (N/N_PER_BLOCK, M/BM)).  A real EP-MoE has
// E=32 local experts with wildly uneven rows-per-expert (M_e), so per-expert launches either
// serialize 32 tiny kernels or pad every expert to a common M (wasting work on empty experts).
//
// V5 fuses ALL experts into ONE grid.  The HOST flattens the per-expert tiles into a flat task
// list tasks[num_tasks][6] and launches grid.x = num_tasks.  Each block reads its task tuple:
//   (local_expert, m_tile_begin, valid_rows, n_superblock, nsub, expert_row_begin)
// and behaves EXACTLY like one V4 block:
//   - global packed-A/C row = expert_row_begin + m_tile_begin   (block_row)
//   - per-expert B base row = local_expert * N                  (B packed [E*N, K])
//   - owns NSUB N-subtiles of N-superblock n_superblock         (A reused NSUB times)
//
// NO CROSS-EXPERT CONTAMINATION (host-padded regions):  the host pads each expert's packed A/C
// region UP to a multiple of BM (see build_tasks.py).  So a block's [block_row, block_row+BM)
// always lies inside ONE expert's padded rows -- a FULL-tile store can never spill into the next
// expert.  The A gather still masks at the TRUE valid_rows boundary (gr < row_limit) so the
// BM-valid_rows padding rows read ZERO and contribute zero to C.  This is chosen over an
// element-wise masked store: the store stays a single fast full-tile store; correctness comes from
// disjoint padded regions + zeroed gather, not from per-row store predication.
//
// EVERYTHING ELSE PRESERVED VERBATIM FROM V4:  fp8 e4m3 remote gather + per-128-group dequant in
// the producer, NSTAGE double-buffered shared tiles, the 4-producer / 4-consumer permanent-wave
// overlap, the inner K-loop, and the s_waitcnt/s_barrier AMD scheduling.  Agents 04/05 own the
// schedule redesign; V5 only changes WHICH (expert,m,n) tile each block computes.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte
static constexpr int QGROUP = 128;   // V1 FP8 block-scale quant group

// flat task layout (keep in sync with build_tasks.py TASK_W / column order).
static constexpr int TASK_W = 6;
enum { T_EXPERT = 0, T_MBEGIN = 1, T_VALID = 2, T_NSUPER = 3, T_NSUB = 4, T_EROWBEG = 5 };

// ----------------------------- tile / block configuration ---------------------------------------
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef NSUB
#define NSUB 8                  // COMPILE-TIME max NSUB (host picks runtime nsub <= this)
#endif
#ifndef NUM_PRODUCER_WORKERS
#define NUM_PRODUCER_WORKERS 4
#endif
#ifndef NUM_CONSUMER_WORKERS
#define NUM_CONSUMER_WORKERS 4
#endif
#ifndef NSTAGE
#define NSTAGE 2            // shared-tile buffering depth (2 = double-buffer)
#endif

#define N_PER_BLOCK (NSUB * BN)
#define NUM_WARPS (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS)
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using PG = kittens::group<NUM_PRODUCER_WORKERS>;   // producer warp group (fast local B load)

// Shared tile types (bf16 — A is dequantized into bf16 before MFMA).
using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

struct micro_globals {
    gl<bf16,  -1, -1, -1, -1> a;       // [Mpacked, K/2]  (reinterpreted fp8 e4m3 [Mpacked,K])
    gl<float, -1, -1, -1, -1> sc;      // [Mpacked, K/128]  fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[E*N, K] local (expert-major), C[Mpacked, N] local
    gl<int,   -1, -1, -1, -1> tasks;   // [num_tasks, TASK_W] int32 flat work list
    iris::iris_device_view iris_ctx;
    int Mpacked, N, K, src_rank;
    int num_tasks;
    int nsub;                           // runtime NSUB chosen by host (<= compile-time NSUB)
    int fused;                          // 1 = fused grouped path, 0 = per-tile baseline path
    hipStream_t stream;
    // V5: grid.x walks the flat task list; one block per task.
    dim3 grid()  { return dim3(num_tasks > 0 ? num_tasks : 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    // shared: NSTAGE A tiles (1 each, reused across N) + NSTAGE * NSUB B tiles (compile-time NSUB).
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * (sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
    }
};

// fp8 -> float using OCP e4m3 (matches V1's __HIP_E4M3).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of a BM x BK A-tile into the swizzled shared bf16 tile `dst`.
//
// V5 generalization of V4's gather: `block_row` is the GLOBAL packed-A row (expert_row_begin +
// m_tile_begin), and `row_limit` is the TRUE last-valid row (block_row + valid_rows).  Rows in
// [row_limit, block_row+BM) are this expert's PADDING -> read ZERO so they contribute nothing to
// the MFMA (and the padded C rows they produce live in dead space no consumer reads).  This is the
// no-contamination guarantee on the gather side.
// ------------------------------------------------------------------------------------------------
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_tile(
        ST_A &dst, int tile, int block_row, int row_limit, int src_rank,
        const micro_globals &g, int warp_id, int laneid) {
    const int k0 = tile * BK;
    const int tid = (warp_id * kittens::WARP_THREADS) + laneid;
    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    const int K = g.K;
    const int NG = K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;

    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = BM * CHUNKS_PER_ROW;

    for (int ci = tid; ci < total_chunks; ci += NUM_PRODUCER_THREADS) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gr = block_row + r;                 // GLOBAL packed-A row
        const int gk = k0 + kc;

        uint4 packed;
        // Mask at the TRUE valid boundary (row_limit), NOT g.M: padding rows of THIS expert read 0.
        if (gr < row_limit && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        const int grp = (gk) / QGROUP;
        float scale = 1.0f;
        if (gr < row_limit && grp < NG) {
            const float* sptr = &g.sc[{0, 0, gr, grp}];
            scale = ctx.load(sptr, src_rank);
        }

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;
            bf16 val = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_A::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_A::swizzle({rr, cc});
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// FUSED GROUPED kernel.  One block per task.  Producer gathers A[BM,BK] ONCE per K-tile + nsub
// local B subtiles (B base row = local_expert*N); consumer MFMAs nsub accumulators against the
// single shared A tile (A reused nsub times).  Inner K-loop + 4P/4C wave schedule == V4 VERBATIM.
//
// NOTE: nsub is a RUNTIME value (g.nsub) <= compile-time NSUB.  We allocate NSUB shared/register
// slots (compile-time) but only iterate the first g.nsub of them.  The host's adaptive selector
// keeps g.nsub in {8,4,2,1}; for the grouped V5 launch the host sets compile-time NSUB to the max
// it will request (default 8) so the buffers are always large enough.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    const int task = blockIdx.x;
    if (task >= g.num_tasks) return;

    // ---- decode this block's task tuple ----
    const int* tk = &g.tasks[{0, 0, task, 0}];
    const int local_expert    = tk[T_EXPERT];
    const int m_tile_begin    = tk[T_MBEGIN];
    const int valid_rows      = tk[T_VALID];
    const int n_superblock    = tk[T_NSUPER];
    const int nsub            = tk[T_NSUB];           // == g.nsub; carried per-task for ABI clarity
    const int expert_row_begin= tk[T_EROWBEG];

    const int block_row = expert_row_begin + m_tile_begin;   // GLOBAL packed-A/C row
    const int row_limit = block_row + valid_rows;            // true valid boundary (gather mask)
    const int block_n0  = n_superblock * (nsub * BN);        // first N column this block owns
    const int b_row0    = local_expert * g.N;                // B base row for this expert
    const int n_tile0   = block_n0 / BN;                     // first B N-tile index (in expert B)

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE]       = al.allocate<ST_A, NSTAGE>();
    ST_B (&Bs)[NSTAGE][NSUB] = al.allocate<ST_B, NSTAGE, NSUB>();

    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    // B is expert-major [E*N, K]; the producer's row index must be offset by b_row0.  The HK
    // gl<bf16> for B is [1,1,E*N,K] reshaped to [1,1,(E*N)/BN, ...] tiles -> the tile row index is
    // (b_row0 + sub*BN)/BN = b_row0/BN + sub.  We bake b_row0/BN into the B tile index below.
    PG::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);
    const int b_tile_row0 = b_row0 / BN;                     // B-tile row base for this expert

    constexpr int PREFETCH = NSTAGE - 1;

    // Prologue: producers prefetch the first PREFETCH K-tiles (A once + nsub B subtiles each).
    if (is_producer) {
        #pragma unroll
        for (int s = 0; s < PREFETCH; ++s) {
            if (s < num_tiles) {
                gather_dequant_A_tile<16>(As[s], s, block_row, row_limit, src_rank, g, warp_id, laneid);
                for (int sub = 0; sub < nsub; ++sub)
                    PG::load<2, false>(Bs[s][sub], g.b, {0, 0, b_tile_row0 + n_tile0 + sub, s}, swizzled_offsets_B);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    // NSUB accumulators per consumer warp (compile-time max; only first nsub used).
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum[NSUB];
    if (is_consumer) {
        for (int sub = 0; sub < nsub; ++sub) zero(C_accum[sub]);
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile % NSTAGE;
        const int fetch = tile + PREFETCH;
        if (is_producer && fetch < num_tiles) {
            const int slot = fetch % NSTAGE;
            gather_dequant_A_tile<16>(As[slot], fetch, block_row, row_limit, src_rank, g, warp_id, laneid);
            for (int sub = 0; sub < nsub; ++sub)
                PG::load<2, false>(Bs[slot][sub], g.b, {0, 0, b_tile_row0 + n_tile0 + sub, fetch}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            // Load the single shared A tile ONCE, reuse across all nsub N-subtiles.
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            load(a_frag, As[cur]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            for (int sub = 0; sub < nsub; ++sub) {
                rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
                auto b_sub = subtile_inplace<CONS_N, BK>(Bs[cur][sub], {cons_id, 0});
                load(b_frag, b_sub);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum[sub], a_frag, b_frag, C_accum[sub]);
                __builtin_amdgcn_s_setprio(0);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        for (int sub = 0; sub < nsub; ++sub) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            // Full-tile store: block_row..block_row+BM is guaranteed inside this expert's PADDED
            // region (host pads M_e up to a multiple of BM), so no spill into the next expert.
            store(g.c, C_accum[sub], {0, 0, block_row / BM, out_col0 / CONS_N});
        }
    }
}

// ------------------------------------------------------------------------------------------------
// BASELINE kernel (two-phase, NO overlap) — the grouped head-to-head reference.  Same flat task
// list / grid as the fused path, but each block runs the V4-baseline two-phase chain (gather full
// A tile, sync, MFMA) per K-tile for each of its nsub N-subtiles SEQUENTIALLY.  This is the
// unfused reference V5-fused must beat on the SAME grouped layout (same experts, same packing,
// same gather/dequant) so the head-to-head isolates fusion+A-stationary, not the grouping.
// ------------------------------------------------------------------------------------------------
using ST_A_b = st_bf<BM, BK, st_16x32_s>;
using ST_B_b = st_bf<BN, BK, st_16x32_s>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk_baseline(micro_globals g) {
    const int task = blockIdx.x;
    if (task >= g.num_tasks) return;

    const int* tk = &g.tasks[{0, 0, task, 0}];
    const int local_expert    = tk[T_EXPERT];
    const int m_tile_begin    = tk[T_MBEGIN];
    const int valid_rows      = tk[T_VALID];
    const int n_superblock    = tk[T_NSUPER];
    const int nsub            = tk[T_NSUB];
    const int expert_row_begin= tk[T_EROWBEG];

    const int block_row = expert_row_begin + m_tile_begin;
    const int row_limit = block_row + valid_rows;
    const int block_n0  = n_superblock * (nsub * BN);
    const int b_row0    = local_expert * g.N;
    const int n_tile0   = block_n0 / BN;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_b (&As)[1] = al.allocate<ST_A_b, 1>();
    ST_B_b (&Bs)[1] = al.allocate<ST_B_b, 1>();

    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);
    const int b_tile_row0 = b_row0 / BN;

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;

    // Process the nsub N-subtiles of this block sequentially (no A reuse, no overlap) -> baseline.
    for (int sub = 0; sub < nsub; ++sub) {
        rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum;
        zero(C_accum);
        for (int tile = 0; tile < num_tiles; ++tile) {
            if (is_producer) {
                gather_dequant_A_tile<16>(As[0], tile, block_row, row_limit, src_rank, g, warp_id, laneid);
                PG::load<2, false>(Bs[0], g.b, {0, 0, b_tile_row0 + n_tile0 + sub, tile}, swizzled_offsets_B);
                __builtin_amdgcn_s_waitcnt(0);
            }
            __syncthreads();
            if (is_consumer) {
                rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
                rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
                load(a_frag, As[0]);
                auto b_sub = subtile_inplace<CONS_N, BK>(Bs[0], {cons_id, 0});
                load(b_frag, b_sub);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum, a_frag, b_frag, C_accum);
                __builtin_amdgcn_s_setprio(0);
            }
            __builtin_amdgcn_sched_barrier(0);
            __syncthreads();
        }
        if (is_consumer) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N});
        }
        __syncthreads();
    }
}

void dispatch_micro(micro_globals g) {
    if (g.num_tasks <= 0) return;   // empty route -> nothing to launch (all experts empty).
    if (g.fused) {
        const unsigned long mem_size = g.dynamic_shared_memory();
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    } else {
        const unsigned long mem_size = (unsigned long)(sizeof(ST_A_b) + sizeof(ST_B_b)) + 1024;
        hipFuncSetAttribute((void*)micro_tk_baseline, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk_baseline<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    }
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "fmoe_fused_v5_grouped tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a,
        &micro_globals::sc,
        &micro_globals::b,
        &micro_globals::c,
        &micro_globals::tasks,
        &micro_globals::iris_ctx,
        &micro_globals::Mpacked,
        &micro_globals::N,
        &micro_globals::K,
        &micro_globals::src_rank,
        &micro_globals::num_tasks,
        &micro_globals::nsub,
        &micro_globals::fused
    );
}
