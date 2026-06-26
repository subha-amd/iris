// v4_variant_base.cuh
// ------------------------------------------------------------------------------------------------
// Single source of truth for the occupancy/register-pressure variants of the V4 A-stationary
// fused MoE expert-GEMM kernel (Agent 07).  This is a verbatim copy of
// irisx/v4_astationary_kernel/kernel.cpp restructured so that the tile/block configuration is
// taken ENTIRELY from -D macros on the compile line (or from a tiny wrapper .cpp that #defines the
// macros then #includes this header).  We do NOT edit the canonical v4 in place (AGENT_COMMON 1).
//
// Each variant = one set of {BM,BN,BK,NSUB,NUM_PRODUCER_WORKERS,NUM_CONSUMER_WORKERS,NSTAGE}.
// The ONLY structural variant that needs its own file (because it changes the consumer loop body,
// not just a constant) is the staged-consumer kernel (v4_staged_consumer.cpp); everything else is a
// pure -D reparameterization of this header.
//
// Register math (confirmed against HipKittens cdna4 rt_base.cuh / rt_shape.cuh / rt.cuh):
//   rt_16x16 base tile, float accumulator : registers_per_thread = 4 VGPR / base tile
//     C_accum = rt_fl<BM,CONS_N,col_l,rt_16x16> : height=BM/16, width=CONS_N/16
//        VGPR(C_accum)        = 4 * (BM/16) * (CONS_N/16)
//        VGPR(C_accum[NSUB])  = NSUB * 4 * (BM/16) * (CONS_N/16)        <-- spill driver
//   rt_16x32 base tile, bf16 : registers_per_thread = 4 VGPR / base tile
//     a_frag  = rt_bf<BM,BK,row_l,rt_16x32> : VGPR = 4 * (BM/16) * (BK/32)   (live across NSUB loop)
//     b_frag  = rt_bf<CONS_N,BK,row_l,rt_16x32> : VGPR = 4 * (CONS_N/16) * (BK/32) (short-lived)
//   __launch_bounds__(NUM_THREADS,1) => single fn, VGPR = max(producer,consumer); producer
//     gather/dequant overhead measured ~50-54 VGPR (uint4 packed + scale + addr math).
//
// IMPORTANT feasibility constraint discovered from rt.cuh static_assert
//   (cols % base_tile_cols == 0): with rt_16x16 accumulators, CONS_N MUST be a multiple of 16.
//   CONS_N = BN / NUM_CONSUMER_WORKERS, so for BN=64 only NUM_CONSUMER_WORKERS in {1,2,4} are legal
//   (CONS_N in {64,32,16}). NUM_CONSUMER_WORKERS=8 => CONS_N=8 => NOT divisible by 16 => WILL NOT
//   COMPILE. The "cons8" idea is therefore infeasible as stated; see REGISTER_OCCUPANCY.md.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include <cstdio>
using namespace kittens;

using fp8_t = __hip_fp8_storage_t;   // unsigned char, 1 byte
static constexpr int QGROUP = 128;   // V1 FP8 block-scale quant group

// ----------------------------- tile / block configuration (ALL from -D) -------------------------
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
#define NSTAGE 2
#endif

#define N_PER_BLOCK (NSUB * BN)
#define NUM_WARPS (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS)
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using PG = kittens::group<NUM_PRODUCER_WORKERS>;

using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

struct micro_globals {
    gl<bf16,  -1, -1, -1, -1> a;
    gl<float, -1, -1, -1, -1> sc;
    gl<bf16,  -1, -1, -1, -1> b, c;
    iris::iris_device_view iris_ctx;
    int M, N, K, src_rank;
    int fused;
    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(N, (int)N_PER_BLOCK), ceil_div(M, (int)BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * (sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
    }
};

__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

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
// FUSED A-STATIONARY kernel (config from macros).
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE]       = al.allocate<ST_A, NSTAGE>();
    ST_B (&Bs)[NSTAGE][NSUB] = al.allocate<ST_B, NSTAGE, NSUB>();

    const int block_row = blockIdx.y * BM;
    const int block_n0  = blockIdx.x * N_PER_BLOCK;
    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;
    const int  n_tile0   = blockIdx.x * NSUB;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    constexpr int PREFETCH = NSTAGE - 1;

    if (is_producer) {
        #pragma unroll
        for (int s = 0; s < PREFETCH; ++s) {
            if (s < num_tiles) {
                gather_dequant_A_tile<16>(As[s], s, block_row, src_rank, g, warp_id, laneid);
                #pragma unroll
                for (int sub = 0; sub < NSUB; ++sub)
                    PG::load<2, false>(Bs[s][sub], g.b, {0, 0, n_tile0 + sub, s}, swizzled_offsets_B);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;
    static_assert(CONS_N % 16 == 0,
        "CONS_N must be a multiple of 16 for rt_16x16 accumulators "
        "(BN / NUM_CONSUMER_WORKERS); NUM_CONSUMER_WORKERS=8 on BN=64 is infeasible.");
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum[NSUB];
    if (is_consumer) {
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) zero(C_accum[sub]);
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int cur = tile % NSTAGE;
        const int fetch = tile + PREFETCH;
        if (is_producer && fetch < num_tiles) {
            const int slot = fetch % NSTAGE;
            gather_dequant_A_tile<16>(As[slot], fetch, block_row, src_rank, g, warp_id, laneid);
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub)
                PG::load<2, false>(Bs[slot][sub], g.b, {0, 0, n_tile0 + sub, fetch}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            load(a_frag, As[cur]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            #pragma unroll
            for (int sub = 0; sub < NSUB; ++sub) {
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
        #pragma unroll
        for (int sub = 0; sub < NSUB; ++sub) {
            const int out_col0 = block_n0 + sub * BN + cons_id * CONS_N;
            store(g.c, C_accum[sub], {0, 0, block_row / BM, out_col0 / CONS_N});
        }
    }
}

// ------------------------------------------------------------------------------------------------
// BASELINE (V3-identical, unchanged) — kept so each variant can still run a head-to-head.
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
    static_assert(CONS_N % 16 == 0, "CONS_N must be a multiple of 16 for rt_16x16 accumulators.");
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
    m.doc() = "fmoe_fused_v4_astationary occ-variant tk_kernel python module";
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
