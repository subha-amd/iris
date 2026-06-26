// p1_tile_inbox / kernel.cpp
// ================================================================================================
// CANDIDATE P1 — TWO-KERNEL COPY-ONCE TILE INBOX (producer copies A 1x; consumer = the REAL B0 GEMM)
//
// WHY (Gate-1 result, EXPERIMENT_LEDGER.md):
//   B0 (local GEMM, no comm)              ~164us / 183 TFLOP/s @ M1024/N2048/K7168  (compute ceiling)
//   B1-copy (copy A once + B0 GEMM, SERIAL) ~291us = ~143us transfer + ~162us B0-class compute
//   V4/B5 fused (refetch A ~4x, occ 1-2)  ~678us  => 2.33x SLOWER than B1. Fusion lost.
//   Perfect-overlap floor = max(143,162) = ~162us. Best possible over B1 ~= 291/162 = 1.80x CEILING.
//
//   The lesson: a competitive design needs ALL THREE at once: (1) ~1x remote A traffic,
//   (2) B0-class GEMM efficiency, (3) comm/compute overlap. Agent 06's cache_first_touch got (1)+(3)
//   but its CONSUMER is a low-occupancy 4-producer/4-consumer fused body (NSUB N-grouping, occ ~2,
//   CONS_N=BN/4) — NOT the B0 GEMM. P1's whole point is to make the consumer the EXACT B0 kernel
//   (256x256x64, 8 warps, __launch_bounds__(512,2)) so compute_retention = B0_lat/consumer_lat ~= 1.
//
// DESIGN (the one change from Agent 06 that matters):
//   * The inbox is NOT a swizzled per-tile staging area; it is the FULL dequantized bf16 A[M,K]
//     row-major buffer — byte-identical to what B0's own dequant_a_dense preamble produces. So the
//     consumer is verbatim B0 (same loads, same swizzle, same MFMA, same occupancy). The producer
//     simply does B0's dequant preamble REMOTELY+ONCE and publishes per-row-band ready flags.
//
//   PRODUCER kernel (modest grid; reserves few CUs first on its own stream):
//     - dispenses distinct A row-bands (BM_PROD-row chunks) via fetch_add on a global cursor;
//     - for each row-band, IRIS-loads the fp8 bytes + per-128 fp32 scales for ALL K columns of those
//       rows EXACTLY ONCE from src_rank (uint4 / 16 fp8 bytes per thread), dequants to bf16, writes
//       row-major into the LOCAL inbox A[M,K]; copy amplification ~= 1.0x;
//     - threadfence_system, then release-store ready[band] = (gen<<1)|1 (anti-stale gen tag).
//       One flag per BM_PROD-row band; the band spans the WHOLE K, so a set flag means the entire
//       A sub-block (all K) for those rows is locally materialized.
//
//   CONSUMER kernel = B0 (UNCHANGED inner loop). Each B0 block owns a B0_BM x B0_BN output tile, i.e.
//     B0_BM (=256) A rows over all K. Before its K-loop it ACQUIRE-spins the ready flags for the
//     B0_BM/BM_PROD producer bands that cover its rows. After that single wait it reads A from the
//     LOCAL inbox and runs the EXACT B0 schedule. ZERO remote reads in the consumer.
//
// FLAG ENCODING: ready=(gen<<1)|1, EMPTY=0 (Agent 06's tile_inbox_abi). A stale READY(gen-1) is a
//   different int, can never satisfy a gen waiter; host bumps gen each step + resets flags.
//
// DEADLOCK AVOIDANCE (argued in P1_DESIGN.md): TWO SEPARATE LAUNCHES on TWO non-blocking streams.
//   The producer launch reserves its CUs independently of the consumer, so the (large) consumer grid
//   can never occupy all CUs before the producer runs. Every flag a consumer awaits is eventually
//   written because the producer cursor hands out EVERY band index exactly once (fetch_add) and a
//   CAS makes exactly one producer block materialize+signal each band. No consumer-block-to-consumer
//   dependency exists, so even a single resident producer block drains the whole pool.
// ================================================================================================
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
#include "tile_inbox_abi.h"   // reused from Agent 06: cft_ready_flag / CFT_FLAG_EMPTY / claim words
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;     // unsigned char, 1 byte
static constexpr int QGROUP = 128;     // per-128-group fp8 block-scale quant group

__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Producer row-band height. One ready flag per BM_PROD rows (spanning all K). 64 keeps the gather
// grid wide and matches the consumer's natural 64-row tiling so the consumer's wait is a small set.
// ------------------------------------------------------------------------------------------------
#ifndef BM_PROD
#define BM_PROD 64
#endif
#ifndef NUM_PRODUCER_THREADS_P1
#define NUM_PRODUCER_THREADS_P1 256
#endif

// ================================================================================================
// CONSUMER GEMM CORE == B0 (harness_kernels.cpp::b0_gemm), VERBATIM tiling/occupancy. The ONLY
// addition is the per-block acquire-spin on the inbox ready flags before the K-loop. Everything that
// determines VGPR/LDS/occupancy (256x256x64, 8 warps, 2 stages, __launch_bounds__(512,2)) is kept.
// ================================================================================================
constexpr int B0_WARPS = 8;
using B0G = kittens::group<B0_WARPS>;
constexpr int B0_BM = 256, B0_BN = 256, B0_BK = 64;
using B0_ST_A = st_bf<B0_BM / 2, B0_BK, st_16x32_s>;
using B0_ST_B = st_bf<B0_BN / 2, B0_BK, st_16x32_s>;

struct p1_globals {
    // remote fp8 source on src_rank (fp8 bytes reinterpreted as bf16[M,K/2]) + per-128 scales.
    gl<bf16,  -1, -1, -1, -1> a_src;    // [M, K/2]
    gl<float, -1, -1, -1, -1> sc_src;   // [M, K/128]
    gl<bf16,  -1, -1, -1, -1> b, c;     // local B[N,K], C[M,N]
    // LOCAL inbox: full dequantized bf16 A[M,K] (row-major, byte-identical to B0's dequant output),
    // plus per-row-band ready flags. All on the consumer rank's symmetric heap.
    gl<bf16,  -1, -1, -1, -1> inbox;    // [M, K] dequantized bf16 A
    gl<int,   -1, -1, -1, -1> ready_gl; // [num_bands]
    gl<int,   -1, -1, -1, -1> claim_gl; // [num_bands]
    gl<int,   -1, -1, -1, -1> cursor_gl;// [1]
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int generation;
    int num_bands;            // == ceil(M / BM_PROD)
    int num_producer_blocks;  // modest; reserves CUs

    dim3 producer_grid() { return dim3(num_producer_blocks > 0 ? num_producer_blocks : 1); }
    dim3 producer_block() { return dim3(NUM_PRODUCER_THREADS_P1); }
    // consumer grid: EXACT B0 grid (one block per 256x256 output tile, flattened).
    dim3 consumer_grid() {
        const int bc = (N + B0_BN - 1) / B0_BN;
        const int br = (M + B0_BM - 1) / B0_BM;
        return dim3(bc * br);
    }
    dim3 consumer_block() { return dim3(B0_WARPS * 64); }
    size_t consumer_shared() { return 2 * (sizeof(B0_ST_A) + sizeof(B0_ST_B)) + 1024; }
};

// ------------------------------------------------------------------------------------------------
// PRODUCER: copy-once remote fp8 gather + dequant of a BM_PROD-row x full-K band into LOCAL inbox.
// One block claims a band via fetch_add(cursor)+CAS(claim); the whole block cooperates on the gather.
// fp8 uint4 loads (16 bytes/thread) mirror B1's gather_once_kernel; scales loaded per-128-group.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_PRODUCER_THREADS_P1, 1)
void p1_producer(p1_globals g) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    iris::iris_device_view ctx = g.iris_ctx;
    const int local_rank = ctx.cur_rank();     // producer + consumer share this rank
    const int K = g.K, M = g.M, NG = K / QGROUP;
    const int num_bands = g.num_bands;
    const int ready_val = cft_ready_flag(g.generation);

    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a_src[{0, 0, 0, 0}]);
    const float* sc_base = &g.sc_src[{0, 0, 0, 0}];
    bf16* inbox = &g.inbox[{0, 0, 0, 0}];
    int* ready  = &g.ready_gl[{0, 0, 0, 0}];
    int* claim  = &g.claim_gl[{0, 0, 0, 0}];
    int* cursor = &g.cursor_gl[{0, 0, 0, 0}];

    __shared__ int s_band;
    __shared__ bool s_won;
    // Per-row scale cache: NG (=K/128) fp32 scales fetched ONCE per row from src_rank, reused for all
    // K columns. NG=56 for K=7168 -> 224 B of LDS. This removes the original 16x-per-chunk (one per
    // fp8 element) redundant REMOTE scalar scale loads that made the producer ~10-100x too slow.
    extern __shared__ float s_scales[];   // [NG]

    while (true) {
        if (tid == 0) {
            s_band = ctx.fetch_add<int, iris::memory_scope_device>(
                         cursor, 1, local_rank, iris::memory_order_relaxed);
            s_won = false;
            if (s_band < num_bands) {
                int expected = CFT_CLAIM_FREE;
                s_won = ctx.compare_exchange_strong<int, iris::memory_scope_device>(
                            &claim[s_band], expected, CFT_CLAIM_TAKEN, local_rank,
                            iris::memory_order_relaxed);
            }
        }
        __syncthreads();
        const int band = s_band;
        if (band >= num_bands) break;
        if (!s_won) { __syncthreads(); continue; }

        const int row0 = band * BM_PROD;
        const int row1 = (row0 + BM_PROD < M) ? (row0 + BM_PROD) : M;

        // gather + dequant all K columns of rows [row0,row1) ONCE -> local inbox row-major bf16.
        // One REMOTE uint4 load (16 fp8 bytes) per chunk; one REMOTE scale load per 128-group per row
        // (cached in LDS). A 16-byte chunk never straddles a 128-group boundary (16 | 128), so a chunk
        // uses exactly one cached scale (kk/QGROUP == k/QGROUP for all 16 elements).
        const int chunks_per_row = K / 16;                 // 16 fp8 bytes (uint4) per chunk
        for (int row = row0; row < row1; ++row) {
            const fp8_t* a_row = a_base + (size_t)row * K;
            const float* sc_row = sc_base + (size_t)row * NG;
            bf16* o_row = inbox + (size_t)row * K;
            // Fetch this row's NG scales ONCE (remote) into LDS, shared by the whole block.
            for (int gi = tid; gi < NG; gi += nthreads)
                s_scales[gi] = ctx.load(sc_row + gi, g.src_rank);
            __syncthreads();
            for (int c = tid; c < chunks_per_row; c += nthreads) {
                const int k = c * 16;
                uint4 v = ctx.load(reinterpret_cast<const uint4*>(a_row + k), g.src_rank);
                const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&v);
                const float scale = s_scales[k / QGROUP];   // constant across the 16 chunk elements
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    o_row[k + j] = __float2bfloat16(fp8_to_f32(bytes[j]) * scale);
                }
            }
            __syncthreads();   // scale cache reusable for the next row
        }
        __syncthreads();

        // publish: make inbox writes globally visible, THEN release the band flag.
        if (tid == 0) {
            ctx.fence<iris::memory_scope_system>(iris::memory_order_release);
            ctx.atomic_store<int, iris::memory_scope_system>(
                &ready[band], ready_val, local_rank, iris::memory_order_release);
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------
// CONSUMER == B0 GEMM. Reads A from the LOCAL inbox (zero remote reads). The added prelude is a
// single acquire-spin over the B0_BM/BM_PROD bands this block's rows cover. Thread 0 of warp 0 polls
// (one int load per band per poll), then __syncthreads() makes the producer's A bytes visible to all.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(B0_WARPS * 64, 2)
void p1_consumer(p1_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    B0_ST_A (&As)[2] = al.allocate<B0_ST_A, 2>();
    B0_ST_B (&Bs)[2] = al.allocate<B0_ST_B, 2>();

    const int M = g.M, N = g.N, K = g.K;
    constexpr int WARPS_COL = 4, WARPS_ROW = 2;
    constexpr int REG_M = B0_BM / WARPS_ROW / 2;
    constexpr int REG_N = B0_BN / WARPS_COL / 2;
    const int k_iters = K / B0_BK;
    const int blocks_per_col = (N + B0_BN - 1) / B0_BN;
    const int block_row = blockIdx.x / blocks_per_col;     // B0 256-row tile index
    const int block_col = blockIdx.x % blocks_per_col;
    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;

    iris::iris_device_view ctx = g.iris_ctx;
    const int local_rank = ctx.cur_rank();
    const int ready_val = cft_ready_flag(g.generation);
    int* ready = &g.ready_gl[{0, 0, 0, 0}];

    // ---- acquire-spin: wait for every producer band covering this block's B0_BM rows ----
    // This block's rows are [block_row*B0_BM, block_row*B0_BM + B0_BM). They span
    // B0_BM/BM_PROD producer bands. Wait for all of them (thread 0 polls; barrier publishes).
    const int row_base = block_row * B0_BM;
    const int band_lo = row_base / BM_PROD;
    int band_hi = (row_base + B0_BM + BM_PROD - 1) / BM_PROD;
    if (band_hi > g.num_bands) band_hi = g.num_bands;
    if (threadIdx.x == 0) {
        for (int band = band_lo; band < band_hi; ++band) {
            int f = ctx.atomic_load<int, iris::memory_scope_system>(
                        &ready[band], local_rank, iris::memory_order_acquire);
            while (f != ready_val) {
                f = ctx.atomic_load<int, iris::memory_scope_system>(
                        &ready[band], local_rank, iris::memory_order_acquire);
            }
        }
    }
    __syncthreads();   // make thread-0's acquired view visible to all threads in the block.
    // Explicit system-scope ACQUIRE FENCE before reading the inbox. The B0 inbox read uses
    // buffer_load_lds (a global->LDS DMA on the cache_all path); the per-band atomic acquire above is
    // on a DIFFERENT address (ready[band]), so without this fence the DMA may read STALE (pre-fill /
    // zeroed) L1/L2 lines for the inbox even though the flag was observed set. This pairs with the
    // producer's fence<system>(release) and is the likely fix for the RMS_rel~1.15 garbage-but-nonzero
    // C (handshake/layout are otherwise byte-identical to the verified B0/B1 path).
    ctx.fence<iris::memory_scope_system>(iris::memory_order_acquire);

    // ---- EXACT B0 inner loop, reading A from the LOCAL inbox gl ----
    rt_bf<REG_M, B0_BK, row_l, rt_16x32_s> a;
    rt_bf<REG_N, B0_BK, row_l, rt_16x32_s> b0;
    rt_fl<REG_M, REG_N, col_l, rt_16x16_s> cacc;
    zero(cacc);

    uint32_t soA[64], soB[64];
    B0G::prefill_swizzled_offsets(As[0], g.inbox, soA);
    B0G::prefill_swizzled_offsets(Bs[0], g.b, soB);

    int tic = 0;
    for (int k = 0; k < k_iters; ++k, tic ^= 1) {
        B0G::load(As[tic], g.inbox, {0, 0, block_row * 2 + warp_m, k}, soA);
        B0G::load(Bs[tic], g.b,    {0, 0, block_col * 2 + warp_n, k}, soB);
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        auto as = kittens::subtile_inplace<REG_M, B0_BK>(As[tic], {0, 0});
        auto bs = kittens::subtile_inplace<REG_N, B0_BK>(Bs[tic], {0, 0});
        load(a, as);
        load(b0, bs);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cacc, a, b0, cacc);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }
    store(g.c, cacc, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
}

// ------------------------------------------------------------------------------------------------
// HOST: two NON-BLOCKING streams; producer launched FIRST (reserves CUs), consumer immediately
// after with NO sync between (so they overlap). Host must reset ready/claim/cursor + bump gen before
// each call (done in example.py). No hipStreamSynchronize between the two launches.
// ------------------------------------------------------------------------------------------------
void dispatch_p1(p1_globals g) {
    const size_t sc = g.consumer_shared();
    hipFuncSetAttribute((void*)p1_consumer, hipFuncAttributeMaxDynamicSharedMemorySize, sc);

    // Producer LDS: NG (=K/128) fp32 scales cached per row.
    const size_t prod_smem = (size_t)(g.K / QGROUP) * sizeof(float) + 16;

    hipStream_t prod_stream, cons_stream;
    hipStreamCreateWithFlags(&prod_stream, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&cons_stream, hipStreamNonBlocking);

    // Producer FIRST on its own stream (reserves CUs), consumer immediately after, NO sync between
    // (overlap). Producer is now fast (1 remote uint4/chunk + 1 remote scale/group/row, cached) so it
    // is not starved by the consumer grid.
    p1_producer<<<g.producer_grid(), g.producer_block(), prod_smem, prod_stream>>>(g);
    p1_consumer<<<g.consumer_grid(), g.consumer_block(), sc, cons_stream>>>(g);

    hipStreamSynchronize(prod_stream);
    hipStreamSynchronize(cons_stream);
    hipStreamDestroy(prod_stream);
    hipStreamDestroy(cons_stream);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "p1_tile_inbox tk_kernel: copy-once producer + B0-GEMM consumer (overlap)";
    py::bind_function<dispatch_p1>(m, "dispatch_p1",
        &p1_globals::a_src,
        &p1_globals::sc_src,
        &p1_globals::b,
        &p1_globals::c,
        &p1_globals::inbox,
        &p1_globals::ready_gl,
        &p1_globals::claim_gl,
        &p1_globals::cursor_gl,
        &p1_globals::iris_ctx,
        &p1_globals::M,
        &p1_globals::N,
        &p1_globals::K,
        &p1_globals::src_rank,
        &p1_globals::generation,
        &p1_globals::num_bands,
        &p1_globals::num_producer_blocks
    );
}
