// fmoe_fused_v4_bsingle_buffer / kernel.cpp
// ------------------------------------------------------------------------------------------------
// V4 ASYMMETRIC-BUFFERING variant (Agent 10, LDS-pressure).
//
// HYPOTHESIS: V4 double-buffers BOTH the gathered A tile (ST_A As[NSTAGE]) and ALL NSUB local B
// subtiles (ST_B Bs[NSTAGE][NSUB]).  A is the SCARCE cross-GPU traffic (IRIS remote gather, ~128
// GB/s) and benefits from double-buffering so the consumer can MFMA tile t while the producer
// gathers tile t+1.  But B is LOCAL HBM (~7.2 TB/s) with no remote latency to hide; the NSUB B
// double-buffers (NSTAGE*NSUB = 16 ST_B tiles = 128 KB of the 145 KB budget) are the LDS hog and
// their second buffer mostly burns LDS for little overlap benefit.
//
// CHANGE vs canonical V4 (the ONLY change):
//   ST_A  As[NSTAGE]        -> UNCHANGED   (A stays double-buffered: hides remote gather latency)
//   ST_B  Bs[NSTAGE][NSUB]  -> Bs[NSUB]    (B single-buffered: one set of NSUB subtiles)
// The A-stationary win is fully preserved: A is still gathered ONCE per K-tile and reused across
// all NSUB N-subtiles (N_PER_BLOCK = NSUB*BN unchanged), so A crosses the interconnect exactly as
// often as canonical V4.  Only B's redundant second buffer is removed.
//
// LDS budget (st_bf<64,64> = 64*64*2 = 8192 B):
//   canonical  : NSTAGE*(ST_A + NSUB*ST_B) + 1024 = 2*(8192 + 8*8192) + 1024 = 148480 B (~145 KB)
//   this variant: NSTAGE*ST_A + NSUB*ST_B + 1024  = 2*8192 + 8*8192 + 1024   =  82944 B (~81 KB)
//   -> LDS saved ~64 KB (the entire second B buffer: NSUB*ST_B = 65536 B).
//   At 160 KB/CU: 160/82.9 = 1.93 -> still 1 block/CU by itself; pair with NSUB<=6 or NSTAGE-relief
//   on A to clear 2 blocks/CU.  Even at occ 1 it FREES ~64 KB that a co-resident block (e.g. the
//   two-phase baseline or a second expert tile) can use, and removes LDS-bank contention.
//
// PIPELINE RISK: with B single-buffered, the producer cannot overwrite Bs[sub] for tile t+1 until
// the consumer has finished reading it for tile t.  We therefore keep the per-K-tile s_barrier
// (already present) which serializes producer-write / consumer-read of B within a tile; A's double
// buffer still lets the A *gather* (the long-latency op) run ahead.  Net: we keep the expensive
// remote-gather overlap, give up only the cheap local-B prefetch overlap.
//
// Build/run: see README.md in this directory.  Correctness criteria identical to V4
// (RMS-rel ~0.0033, zero-sentinel proves remote gather).
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
#ifndef NSUB
#define NSUB 8
#endif
#ifndef NUM_PRODUCER_WORKERS
#define NUM_PRODUCER_WORKERS 4
#endif
#ifndef NUM_CONSUMER_WORKERS
#define NUM_CONSUMER_WORKERS 4
#endif
#ifndef NSTAGE
#define NSTAGE 2            // shared-tile buffering depth FOR A ONLY (2 = double-buffer A)
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
    gl<bf16,  -1, -1, -1, -1> a;       // [M, K/2]  (reinterpreted fp8 e4m3 [M,K])
    gl<float, -1, -1, -1, -1> sc;      // [M, K/128]  fp32 scales
    gl<bf16,  -1, -1, -1, -1> b, c;    // B[N,K] local, C[M,N] local
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int fused;                          // 1 = fused overlap path, 0 = two-phase baseline path
    hipStream_t stream;
    // V4: grid x dim walks N in N_PER_BLOCK steps (each block owns NSUB N-subtiles).
    dim3 grid()  { return dim3(ceil_div(N, (int)N_PER_BLOCK), ceil_div(M, (int)BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    // shared: NSTAGE A tiles (double-buffered, reused across N) + ONE set of NSUB B tiles (single-buffered).
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B) + 1024;
    }
};

// fp8 -> float using OCP e4m3 (matches V1's __HIP_E4M3).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of a BM x BK A-tile into the swizzled shared bf16 tile `dst`.
// (Identical to V3/V4 — this is the EXACT scarce cross-GPU traffic we are amortizing.)
// ------------------------------------------------------------------------------------------------
template<int VEC>
__device__ __forceinline__ void gather_dequant_A_tile(
        ST_A &dst, int tile, int block_row, int src_rank,
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

        const int grp = (gk) / QGROUP;
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
// FUSED A-STATIONARY kernel, ASYMMETRIC BUFFERING:
//   A double-buffered (As[NSTAGE]) to hide the remote gather; B single-buffered (Bs[NSUB]).
// Producer gathers A[BM,BK] ONCE per K-tile + NSUB local B subtiles; consumer MFMAs NSUB
// accumulators against the single shared A tile (A reused NSUB times).
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE] = al.allocate<ST_A, NSTAGE>();   // A: double-buffered (unchanged from V4)
    ST_B (&Bs)[NSUB]   = al.allocate<ST_B, NSUB>();     // B: SINGLE-buffered (was Bs[NSTAGE][NSUB])

    const int block_row = blockIdx.y * BM;
    const int block_n0  = blockIdx.x * N_PER_BLOCK;     // first N column this block owns
    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;
    const int  n_tile0   = blockIdx.x * NSUB;           // first B N-tile index this block owns

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);

    constexpr int PREFETCH = NSTAGE - 1;   // prefetch depth for A (B is single-buffered)

    // Prologue: producers prefetch the first PREFETCH A K-tiles into As[] (A double-buffer fill).
    // B is single-buffered, so B for tile 0 is loaded inside the main loop's tile==0 iteration
    // (it cannot be prefetched ahead — there is only one B buffer).
    if (is_producer) {
        #pragma unroll
        for (int s = 0; s < PREFETCH; ++s) {
            if (s < num_tiles) {
                gather_dequant_A_tile<16>(As[s], s, block_row, src_rank, g, warp_id, laneid);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    // NSUB accumulators per consumer warp (one per N-subtile it co-owns).
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum[NSUB];
    if (is_consumer) {
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) zero(C_accum[sub]);
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile % NSTAGE;          // A slot for this tile (double-buffered)
        const int fetch = tile + PREFETCH;      // A slot to gather ahead (double-buffered)
        if (is_producer) {
            // Load B for THIS tile into the single B buffer (cannot be prefetched: 1 buffer).
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub)
                PG::load<2, false>(Bs[sub], g.b, {0, 0, n_tile0 + sub, tile}, swizzled_offsets_B);
            // Gather the NEXT A tile ahead into its double-buffer slot (hides remote latency).
            if (fetch < num_tiles) {
                const int slot = fetch % NSTAGE;
                gather_dequant_A_tile<16>(As[slot], fetch, block_row, src_rank, g, warp_id, laneid);
            }
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            // Load the single shared A tile (current double-buffer slot) ONCE, reuse across NSUB.
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            load(a_frag, As[cur]);
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
        // The per-tile barrier serializes producer-write / consumer-read of the SINGLE B buffer:
        // the producer for tile t+1 must not overwrite Bs[] until the consumer has read it for t.
        // (A is double-buffered so its gather can still run ahead across this barrier.)
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            store(g.c, C_accum[sub], {0, 0, block_row / BM, out_col0 / CONS_N});
        }
    }
}

// ------------------------------------------------------------------------------------------------
// BASELINE kernel (two-phase, NO overlap) — IDENTICAL to V3/V4's baseline.  Unchanged.
// ------------------------------------------------------------------------------------------------
using ST_A_b = st_bf<BM, BK, st_16x32_s>;
using ST_B_b = st_bf<BN, BK, st_16x32_s>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk_baseline(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_b (&As)[1] = al.allocate<ST_A_b, 1>();
    ST_B_b (&Bs)[1] = al.allocate<ST_B_b, 1>();

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
    zero(C_accum);

    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    for (int tile = 0; tile < num_tiles; ++tile) {
        if (is_producer) {
            gather_dequant_A_tile<16>(As[0], tile, block_row, src_rank, g, warp_id, laneid);
            PG::load<2, false>(Bs[0], g.b, {0, 0, (int)blockIdx.x, tile}, swizzled_offsets_B);
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
        const int out_col0 = block_col + cons_id * CONS_N;
        store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N});
    }
}

void dispatch_micro(micro_globals g) {
    if (g.fused) {
        const unsigned long mem_size = g.dynamic_shared_memory();
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    } else {
        const unsigned long mem_size = (unsigned long)(sizeof(ST_A_b) + sizeof(ST_B_b)) + 1024;
        hipFuncSetAttribute((void*)micro_tk_baseline, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        dim3 bgrid(ceil_div(g.N, (int)BN), ceil_div(g.M, (int)BM));
        micro_tk_baseline<<<bgrid, g.block(), mem_size, g.stream>>>(g);
    }
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "fmoe_fused_v4_bsingle_buffer tk_kernel python module";
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
