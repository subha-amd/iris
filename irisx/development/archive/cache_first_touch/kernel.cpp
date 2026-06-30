// cache_first_touch / kernel.cpp
// ------------------------------------------------------------------------------------------------
// Agent 06 — CACHE-ON-FIRST-TOUCH expert-GEMM, TWO-KERNEL prototype.
//
// Goal: keep a LARGE consumer grid (good latency hiding) but make every remote A tile cross XGMI
// EXACTLY ONCE. Built on V4 (a-stationary) — do NOT edit V3/V4 in place.
//
//   V4 reuses one gathered A tile across NSUB N-subtiles WITHIN a block. But the same remote
//   A[m_tile,k_tile] is still re-gathered by every distinct consumer block on that m-band. To keep
//   many blocks (latency hiding) without paying many remote reads, we DECOUPLE gather from compute:
//
//   PRODUCER kernel (few blocks, reserves CUs first via separate launch):
//     - dispenses distinct A-tasks via fetch_add on a global cursor;
//     - CAS-claims each task FREE->TAKEN so no tile is gathered twice;
//     - gathers the fp8 A tile + per-128 scales ONCE from src_rank over IRIS, dequants to bf16;
//     - stores the bf16 tile into a LOCAL HBM symmetric-heap inbox slot (tile-swizzled like ST_A);
//     - atomic_store(release) ready = (gen<<1)|1 on that slot's flag.
//
//   CONSUMER grouped-GEMM kernel (FULL M/BM x N/BN grid):
//     - for each (m_tile,k_tile) it needs, atomic_load(acquire)-spins the slot flag == ready(gen);
//     - reads A from LOCAL HBM/L2 (zero remote reads), loads local B, MFMAs;
//     - writes C.
//
// Flag anti-alias: ready=(gen<<1)|1, EMPTY=0; a stale READY(gen-1) is a different int and can never
// satisfy a gen waiter. See tile_inbox_abi.h.
//
// Deadlock avoidance: TWO SEPARATE LAUNCHES on TWO STREAMS. The producer launch reserves its CUs
// independently of the consumer launch, so consumers can never occupy all CUs before producers run.
// Every flag a consumer awaits is eventually written because the producer's cursor hands out every
// task index and CAS guarantees exactly one producer gathers+signals each one. (Full argument in
// CACHE_FIRST_TOUCH.md.)
//
// IRIS atomic/flag protocol is the canonical examples/01_message_passing one:
//   release on the producer store, acquire on the consumer spin, memory_scope_system.
// Here producer+consumer are on the SAME rank, so the flag is signaled with remote_rank = the
// consumer (local) rank; system scope still required because two concurrent kernels on the device
// communicate through HBM and we need cross-kernel visibility.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
#include "tile_inbox_abi.h"
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte
static constexpr int QGROUP = 128;   // FP8 block-scale quant group

// ----------------------------- tile / block configuration ---------------------------------------
#ifndef BM
#define BM CFT_BM
#endif
#ifndef BN
#define BN CFT_BN
#endif
#ifndef BK
#define BK CFT_BK
#endif
#ifndef NSUB
#define NSUB 8
#endif
#ifndef NUM_PRODUCER_WORKERS
#define NUM_PRODUCER_WORKERS 4
#endif
#ifndef NUM_CONSUMER_WORKERS
#define NUM_CONSUMER_WORKERS 4
#endif

// Producer kernel uses its own warp count (all warps gather). Reuse PRODUCER_WORKERS as its width.
#define NUM_WARPS (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS)
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)
#define N_PER_BLOCK (NSUB * BN)

using PG = kittens::group<NUM_PRODUCER_WORKERS>;

// Shared tile types (bf16 — A dequantized into bf16 before MFMA). Identical swizzle to V4 so the
// inbox bytes the producer writes are exactly what the consumer's `load(frag, ...)` expects.
using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

// ------------------------------------------------------------------------------------------------
// Globals shared by both kernels. A is the REMOTE fp8 source (read over IRIS by the producer only);
// inbox_A / ready / claim / cursor live on the LOCAL consumer-rank heap.
// ------------------------------------------------------------------------------------------------
struct cft_globals {
    // remote fp8 source on src_rank (bf16-reinterpreted bytes, like V4): [M, K/2]
    gl<bf16,  -1, -1, -1, -1> a;
    gl<float, -1, -1, -1, -1> sc;    // remote per-128 scales [M, K/128]
    gl<bf16,  -1, -1, -1, -1> b, c;  // local B[N,K], C[M,N]

    // local tile inbox (all on consumer rank's symmetric heap). NOTE: ready/claim/cursor are passed
    // as gl<int> (torch int32 tensors) because the HK pybind path (pyutils from_object) only knows
    // how to build a gl from a torch.Tensor or cast a scalar — it CANNOT bind a raw `int*`. We pull
    // the raw pointer out inside the kernel via &x[{0,0,0,0}].
    gl<bf16, -1, -1, -1, -1> inbox;    // [num_slots, BM*BK] flattened bf16, swizzled per ST_A
    gl<int,  -1, -1, -1, -1> ready_gl; // [num_slots]
    gl<int,  -1, -1, -1, -1> claim_gl; // [num_tasks == num_slots]
    gl<int,  -1, -1, -1, -1> cursor_gl;// [1]

    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int generation;
    int num_m_tiles, num_k_tiles, num_slots;

    dim3 block() { return dim3(NUM_THREADS); }
    // producer grid: a MODEST number of blocks (reserves few CUs, dispenses all tasks via cursor).
    dim3 producer_grid() { return dim3(num_producer_blocks); }
    // consumer grid: FULL M/BM x N/BN (a-stationary N grouping like V4: x walks N in N_PER_BLOCK).
    dim3 consumer_grid() { return dim3(ceil_div(N, (int)N_PER_BLOCK), ceil_div(M, (int)BM)); }
    int num_producer_blocks;
    size_t shared_producer() { return sizeof(ST_A) + 1024; }
    size_t shared_consumer() { return sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B) + 1024; }
};

__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of a BM x BK A-tile into shared bf16 tile `dst` (V4-identical: this
// is the scarce XGMI traffic we are reducing to ONCE-per-tile).
// ------------------------------------------------------------------------------------------------
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_tile(
        ST_A &dst, int k_tile, int block_row, int src_rank,
        const cft_globals &g, int tid, int nthreads) {
    const int k0 = k_tile * BK;
    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    const int K = g.K;
    const int NG = K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a[{0, 0, 0, 0}]);
    iris::iris_device_view ctx = g.iris_ctx;

    constexpr int CHUNKS_PER_ROW = BK / VEC;
    const int total_chunks = BM * CHUNKS_PER_ROW;

    for (int ci = tid; ci < total_chunks; ci += nthreads) {
        const int r  = ci / CHUNKS_PER_ROW;
        const int kc = (ci % CHUNKS_PER_ROW) * VEC;
        const int gr = block_row + r;
        const int gk = k0 + kc;

        uint4 packed;
        if (gr < g.M && (gk + VEC) <= K) {
            const fp8_t* aptr = a_base + (size_t)gr * K + gk;
            packed = ctx.load(reinterpret_cast<const uint4*>(aptr), src_rank);
        } else {
            packed = make_uint4(0u, 0u, 0u, 0u);
        }
        const fp8_t* bytes = reinterpret_cast<const fp8_t*>(&packed);

        const int grp = gk / QGROUP;
        float scale = 1.0f;
        if (gr < g.M && grp < NG) {
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
// PRODUCER KERNEL
//   while (idx = fetch_add(cursor, 1)) < num_tasks:
//     CAS claim[idx] FREE->TAKEN; if lost, continue (someone else gathers it);
//     decode idx -> (m_tile, k_tile); gather+dequant A tile ONCE into shared; copy shared->inbox
//     slot in LOCAL HBM; threadfence_system; atomic_store(release) ready[slot] = ready(gen).
// Only thread 0 of the block drives the cursor/CAS; the whole block cooperates on the gather + copy.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void cft_producer(cft_globals g) {
    extern __shared__ alignment_dummy __shm_p[];
    shared_allocator al((int*)&__shm_p[0]);
    ST_A (&As)[1] = al.allocate<ST_A, 1>();

    const int tid = threadIdx.x;
    iris::iris_device_view ctx = g.iris_ctx;
    const int local_rank = ctx.cur_rank();   // producer + consumer share this rank
    const int num_tasks  = g.num_slots;
    const int ready_val  = cft_ready_flag(g.generation);
    int* ready  = &g.ready_gl[{0, 0, 0, 0}];
    int* claim  = &g.claim_gl[{0, 0, 0, 0}];
    int* cursor = &g.cursor_gl[{0, 0, 0, 0}];

    __shared__ int s_idx;
    __shared__ bool s_won;

    while (true) {
        if (tid == 0) {
            // dispense a candidate task index.
            s_idx = ctx.fetch_add<int, iris::memory_scope_device>(
                        cursor, 1, local_rank, iris::memory_order_relaxed);
            s_won = false;
            if (s_idx < num_tasks) {
                // CAS-claim FREE->TAKEN so no tile is gathered twice.
                int expected = CFT_CLAIM_FREE;
                s_won = ctx.compare_exchange_strong<int, iris::memory_scope_device>(
                            &claim[s_idx], expected, CFT_CLAIM_TAKEN, local_rank,
                            iris::memory_order_relaxed);
            }
        }
        __syncthreads();
        const int idx = s_idx;
        if (idx >= num_tasks) break;     // pool drained — all producers exit
        if (!s_won) { __syncthreads(); continue; }   // lost the race; try next index

        const int m_tile = idx / g.num_k_tiles;
        const int k_tile = idx % g.num_k_tiles;
        const int block_row = m_tile * BM;

        // gather+dequant ONCE over IRIS into shared.
        gather_dequant_A_tile<16>(As[0], k_tile, block_row, g.src_rank, g, tid, NUM_THREADS);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();

        // copy swizzled shared tile -> local HBM inbox slot (byte-for-byte; same swizzle as ST_A,
        // so consumer load(frag, inbox_slot) ingests it directly).
        constexpr int TILE_BYTES = (int)sizeof(ST_A);
        char* dst = reinterpret_cast<char*>(&g.inbox[{0, 0, idx, 0}]);
        const char* src = reinterpret_cast<const char*>(&As[0].data[0]);
        for (int o = tid * (int)sizeof(uint4); o < TILE_BYTES; o += NUM_THREADS * (int)sizeof(uint4)) {
            if (o + (int)sizeof(uint4) <= TILE_BYTES)
                *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(src + o);
        }
        __syncthreads();

        // publish: make the inbox writes globally visible, THEN release the flag.
        if (tid == 0) {
            ctx.fence<iris::memory_scope_system>(iris::memory_order_release);
            ctx.atomic_store<int, iris::memory_scope_system>(
                &ready[idx], ready_val, local_rank, iris::memory_order_release);
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------
// CONSUMER KERNEL — FULL grid, a-stationary N grouping (like V4). For each K-tile: acquire-spin the
// inbox slot flag, load A from LOCAL inbox, MFMA NSUB N-subtiles. ZERO remote reads.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void cft_consumer(cft_globals g) {
    extern __shared__ alignment_dummy __shm_c[];
    shared_allocator al((int*)&__shm_c[0]);
    ST_A (&As)[1]      = al.allocate<ST_A, 1>();
    ST_B (&Bs)[NSUB]   = al.allocate<ST_B, NSUB>();

    const int block_row = blockIdx.y * BM;
    const int block_n0  = blockIdx.x * N_PER_BLOCK;
    const int m_tile    = blockIdx.y;
    const int warp_id   = kittens::warpid();
    const bool is_producer_warp = (warp_id < NUM_PRODUCER_WORKERS); // local B loaders
    const bool is_consumer_warp = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id  = is_consumer_warp ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  num_tiles = g.K / BK;
    const int  n_tile0   = blockIdx.x * NSUB;
    const int  tid       = threadIdx.x;
    const int  ready_val = cft_ready_flag(g.generation);
    iris::iris_device_view ctx = g.iris_ctx;
    const int  local_rank = ctx.cur_rank();
    int* ready = &g.ready_gl[{0, 0, 0, 0}];

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum[NSUB];
    if (is_consumer_warp) {
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) zero(C_accum[sub]);
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int slot = cft_slot_index(m_tile, tile, g.num_k_tiles);

        // ACQUIRE-spin the slot flag (thread 0), then make A visible to the whole block.
        if (tid == 0) {
            int f = ctx.atomic_load<int, iris::memory_scope_system>(
                        &ready[slot], local_rank, iris::memory_order_acquire);
            while (f != ready_val) {
                f = ctx.atomic_load<int, iris::memory_scope_system>(
                        &ready[slot], local_rank, iris::memory_order_acquire);
            }
        }
        __syncthreads();

        // load A ONCE from LOCAL inbox slot into shared; load NSUB local B subtiles.
        if (is_producer_warp) {
            // copy local HBM inbox slot -> shared As (same swizzle).
            constexpr int TILE_BYTES = (int)sizeof(ST_A);
            char* dst = reinterpret_cast<char*>(&As[0].data[0]);
            const char* src = reinterpret_cast<const char*>(&g.inbox[{0, 0, slot, 0}]);
            for (int o = tid * (int)sizeof(uint4); o < TILE_BYTES;
                 o += NUM_PRODUCER_THREADS * (int)sizeof(uint4)) {
                if (o + (int)sizeof(uint4) <= TILE_BYTES)
                    *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(src + o);
            }
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub)
                PG::load<2, false>(Bs[sub], g.b, {0, 0, n_tile0 + sub, tile}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        }
        __syncthreads();

        if (is_consumer_warp) {
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            load(a_frag, As[0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub) {
                rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
                auto b_sub = subtile_inplace<CONS_N, BK>(Bs[sub], {cons_id, 0});
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

    if (is_consumer_warp) {
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            store(g.c, C_accum[sub], {0, 0, block_row / BM, out_col0 / CONS_N});
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Host dispatch. TWO SEPARATE LAUNCHES on TWO STREAMS so the producer reserves CUs independently of
// the consumer (deadlock avoidance). The producer launch is fire-and-forget on its own stream; the
// consumer launch runs concurrently on a second stream and acquire-spins on the flags. The host
// must NOT serialize them (no hipStreamSynchronize between the two launches) — they overlap.
//
// IMPORTANT: ready/claim/cursor and the inbox MUST be reset to EMPTY/FREE/0 by the host BEFORE this
// is called for a given generation (done in example.py). Generation is bumped each step so stale
// READY flags from a prior step can never satisfy a new waiter even if a reset is skipped.
// ------------------------------------------------------------------------------------------------
void dispatch_cft(cft_globals g) {
    const unsigned long sp = g.shared_producer();
    const unsigned long sc = g.shared_consumer();
    hipFuncSetAttribute((void*)cft_producer, hipFuncAttributeMaxDynamicSharedMemorySize, sp);
    hipFuncSetAttribute((void*)cft_consumer, hipFuncAttributeMaxDynamicSharedMemorySize, sc);

    // Two NON-BLOCKING streams so producer and consumer are CO-RESIDENT on the device. We create
    // them here rather than binding hipStream_t through pybind (pyutils can't bind a raw stream
    // handle). Producer launched FIRST so it reserves its CUs before the full consumer grid floods
    // the device; consumer launched immediately after with NO sync in between so they overlap.
    hipStream_t prod_stream, cons_stream;
    hipStreamCreateWithFlags(&prod_stream, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&cons_stream, hipStreamNonBlocking);

    cft_producer<<<g.producer_grid(), g.block(), sp, prod_stream>>>(g);
    cft_consumer<<<g.consumer_grid(), g.block(), sc, cons_stream>>>(g);

    hipStreamSynchronize(prod_stream);
    hipStreamSynchronize(cons_stream);
    hipStreamDestroy(prod_stream);
    hipStreamDestroy(cons_stream);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "cache_first_touch tk_kernel python module (two-kernel producer/consumer)";
    py::bind_function<dispatch_cft>(m, "dispatch_cft",
        &cft_globals::a,
        &cft_globals::sc,
        &cft_globals::b,
        &cft_globals::c,
        &cft_globals::inbox,
        &cft_globals::ready_gl,
        &cft_globals::claim_gl,
        &cft_globals::cursor_gl,
        &cft_globals::iris_ctx,
        &cft_globals::M,
        &cft_globals::N,
        &cft_globals::K,
        &cft_globals::src_rank,
        &cft_globals::generation,
        &cft_globals::num_m_tiles,
        &cft_globals::num_k_tiles,
        &cft_globals::num_slots,
        &cft_globals::num_producer_blocks
    );
}
