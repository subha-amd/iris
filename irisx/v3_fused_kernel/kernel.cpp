// fmoe_fused_v3 / kernel.cpp
// ------------------------------------------------------------------------------------------------
// V3 — the FULL FUSED MoE expert-GEMM.  Merges V1 (FP8 quant/dequant) with V2.1 (remote-gather
// fused GEMM) into a SINGLE kernel.
//
// Thesis:  rank 1 runs a producer/consumer GEMM whose PRODUCER warps pull each A-tile of
// QUANTIZED fp8 e4m3 activations DIRECTLY from rank 0's IRIS heap (vectorized 16B gathers),
// DEQUANTIZE fp8->bf16 in the producer using V1's per-128-element-group fp32 scales (also pulled
// remotely), and write bf16 into the swizzled shared tile, while the CONSUMER warps MFMA the
// previous tile.  So cross-GPU movement + dequant + matmul all overlap in one kernel — replacing
// the production two-phase  EpDispatch + dynamic_quant + fmoe  chain.
//
// This is the FUSED side of the head-to-head.  The UNFUSED two-phase baseline (gather-then-gemm,
// no overlap) is implemented as a second kernel in this same file (micro_tk_baseline), driven by
// example.py, so both run identical shapes/layouts for an apples-to-apples wall-clock compare.
//
// Layouts (row-major):
//   A_fp8  : [M, K]      fp8 e4m3  — quantized activations; ONLY real copy on rank `src_rank`.
//   A_sc   : [M, K/128]  fp32      — per-128-group scales; ONLY real copy on rank `src_rank`.
//   B      : [N, K]      bf16      — weights, local on the consumer rank.  C = A @ B^T.
//   C      : [M, N]      bf16      — output, local on the consumer rank.
//
// Dequant matches V1 exactly:  x_bf16 = float(fp8) * scale[group],  scale = max(|group|)/448.
//
// AMD scheduling preserved from V2.1: producer/consumer warpgroups, double-buffered shared tiles,
// s_waitcnt discipline, s_barrier between stages.  No NVIDIA-style idle-producer wave specialization.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte
static constexpr int QGROUP = 128;   // V1 FP8 block-scale quant group

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
#ifndef NUM_PRODUCER_WORKERS
#define NUM_PRODUCER_WORKERS 4
#endif
#ifndef NUM_CONSUMER_WORKERS
#define NUM_CONSUMER_WORKERS 4
#endif
#ifndef NSTAGE
#define NSTAGE 2            // shared-tile buffering depth (2 = double-buffer; try 3/4 with big LDS)
#endif

#define NUM_WARPS (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS)
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using PG = kittens::group<NUM_PRODUCER_WORKERS>;   // producer warp group (fast local B load)

// Shared tile types (bf16 — A is dequantized into bf16 before MFMA).
using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

struct micro_globals {
    // a is the fp8 activation buffer (remote on src_rank): M*K BYTES, but HK's gl rejects 1-byte
    // element types (no packed_type for unsigned char), so we declare it as gl<bf16,[M,K/2]> — the
    // SAME storage, 2 fp8 bytes per bf16 slot — and index raw bytes ourselves in the gather.
    gl<bf16,  -1, -1, -1, -1> a;       // [M, K/2]  (reinterpreted fp8 e4m3 [M,K])
    gl<float, -1, -1, -1, -1> sc;      // [M, K/128]  fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[N,K] local, C[M,N] local
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int fused;                          // 1 = fused overlap path, 0 = two-phase baseline path
    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(N, (int)BN), ceil_div(M, (int)BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return (size_t)NSTAGE * (sizeof(ST_A) + sizeof(ST_B)) + 1024; }
};

// fp8 -> float using OCP e4m3 (matches V1's __HIP_E4M3).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of a BM x BK A-tile into the swizzled shared bf16 tile `dst`.
// Producer threads stride over the BM*BK tile elements, VECTORIZED 16 fp8 bytes/thread (uint4)
// across K, dequantize each with its per-128-group scale (also pulled remotely; cached per group),
// and write bf16 into the swizzled shared layout the consumer's load(rt,st) expects.
// ------------------------------------------------------------------------------------------------
template<int VEC>   // fp8 elements fetched per thread per inner step (16 = one uint4)
__device__ __forceinline__ void gather_dequant_A_tile(
        ST_A &dst, int tile, int block_row, int src_rank,
        const micro_globals &g, int warp_id, int laneid) {
    const int k0 = tile * BK;
    const int tid = (warp_id * kittens::WARP_THREADS) + laneid;  // 0..NUM_PRODUCER_THREADS-1
    constexpr int SUBR = ST_A::underlying_subtile_rows;          // 16
    constexpr int SUBC = ST_A::underlying_subtile_cols;          // 32
    constexpr int SUBN = ST_A::underlying_subtile_elements;      // 512
    const int K = g.K;
    const int NG = K / QGROUP;
    // Base byte pointer of A's fp8 storage (declared as bf16[M,K/2] -> M*K bytes, row stride = K).
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;   // local mutable copy (load() is non-const)

    // Number of VEC-chunks across the BK columns; each chunk is 16 contiguous K elements.
    constexpr int CHUNKS_PER_ROW = BK / VEC;                     // 64/16 = 4
    const int total_chunks = BM * CHUNKS_PER_ROW;

    for (int ci = tid; ci < total_chunks; ci += NUM_PRODUCER_THREADS) {
        const int r  = ci / CHUNKS_PER_ROW;       // local tile row 0..BM-1
        const int kc = (ci % CHUNKS_PER_ROW) * VEC; // local tile col base 0..BK-VEC
        const int gr = block_row + r;             // global A row
        const int gk = k0 + kc;                   // global A col base

        // Vectorized remote gather of VEC=16 fp8 bytes -> one uint4 (16 bytes).
        uint4 packed;
        if (gr < g.M && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;          // fp8 byte address
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        // Dequant: each of the 16 elements belongs to group (gk+j)/128.  Within a 16-wide chunk
        // all 16 share the same group unless the chunk straddles a 128 boundary (it never does for
        // BK | 128 and gk % 16 == 0 with 128 % 16 == 0), so one scale fetch covers the chunk.
        const int grp = (gk) / QGROUP;
        float scale = 1.0f;
        if (gr < g.M && grp < NG) {
            const float* sptr = &g.sc[{0, 0, gr, grp}];
            scale = ctx.load(sptr, src_rank);
        }

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int k = kc + j;                 // local tile col
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
// FUSED kernel: producer gathers+dequants tile (t+1) while consumer MFMAs tile (t).  Overlap.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE] = al.allocate<ST_A, NSTAGE>();
    ST_B (&Bs)[NSTAGE] = al.allocate<ST_B, NSTAGE>();

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
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

    constexpr int PREFETCH = NSTAGE - 1;   // tiles the producer runs ahead of the consumer

    // Prologue: producers prefetch the first PREFETCH tiles (deep pipeline => many remote
    // loads in flight, hiding the latency-bound IRIS gather under both more gather AND compute).
    if (is_producer) {
        #pragma unroll
        for (int s = 0; s < PREFETCH; ++s) {
            if (s < num_tiles) {
                gather_dequant_A_tile<16>(As[s], s, block_row, src_rank, g, warp_id, laneid);
                PG::load<2, false>(Bs[s], g.b, {0, 0, (int)blockIdx.x, s}, swizzled_offsets_B);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum;
    if (is_consumer) zero(C_accum);

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile % NSTAGE;
        const int fetch = tile + PREFETCH;          // tile the producer fetches this step
        if (is_producer && fetch < num_tiles) {
            const int slot = fetch % NSTAGE;
            gather_dequant_A_tile<16>(As[slot], fetch, block_row, src_rank, g, warp_id, laneid);
            PG::load<2, false>(Bs[slot], g.b, {0, 0, (int)blockIdx.x, fetch}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
            load(a_frag, As[cur]);
            auto b_sub = subtile_inplace<CONS_N, BK>(Bs[cur], {cons_id, 0});
            load(b_frag, b_sub);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a_frag, b_frag, C_accum);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        const int out_col0 = block_col + cons_id * CONS_N;
        store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N});
    }
}

// ------------------------------------------------------------------------------------------------
// BASELINE kernel (two-phase, NO overlap): ALL producer threads first gather+dequant the ENTIRE
// K dimension of this block's A-tile into a local scratch in shared, with a full barrier, THEN
// the consumers MFMA — no producer/consumer overlap.  This mimics the production design where
// dispatch+quant writes the whole buffer, a barrier, then the GEMM reads it back.  Same shapes,
// same gather, same dequant — the ONLY difference is overlap, so the head-to-head isolates the
// fusion win.  (We serialize per-tile: gather tile t (all warps), barrier, MFMA tile t, barrier.)
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk_baseline(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[1] = al.allocate<ST_A, 1>();
    ST_B (&Bs)[1] = al.allocate<ST_B, 1>();

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int warp_id   = kittens::warpid();
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

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum;
    zero(C_accum);   // zero on ALL warps (harmless on producers; avoids any uninit path)

    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    for (int tile = 0; tile < num_tiles; ++tile) {
        // PHASE A: gather+dequant (producers) + load B (producers).  Consumers idle.
        if (is_producer) {
            gather_dequant_A_tile<16>(As[0], tile, block_row, src_rank, g, warp_id, laneid);
            PG::load<2, false>(Bs[0], g.b, {0, 0, (int)blockIdx.x, tile}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        }
        __syncthreads();   // hard separation: comm finishes before compute starts (no overlap)
        // PHASE B: MFMA (consumers).  Producers idle.
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
        __syncthreads();   // make sure MFMA done reading As[0] before next gather overwrites it
    }

    if (is_consumer) {
        const int out_col0 = block_col + cons_id * CONS_N;
        store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    if (g.fused) {
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    } else {
        hipFuncSetAttribute((void*)micro_tk_baseline, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk_baseline<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    }
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "fmoe_fused_v3 tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a,
        &micro_globals::sc,
        &micro_globals::b,
        &micro_globals::c,
        &micro_globals::iris_ctx,
        &micro_globals::M,
        &micro_globals::N,
        &micro_globals::K,
        &micro_globals::src_rank,
        &micro_globals::fused
    );
}
