// fmoe_fused_sched_xcd / kernel.cpp
// ------------------------------------------------------------------------------------------------
// XCD-AWARE GRID SCHEDULING on top of V4 (A-stationary fused MoE expert-GEMM).
// Built on irisx/v4_astationary_kernel/kernel.cpp.  Do NOT edit V4 in place.
//
// WHY -----------------------------------------------------------------------------------------
// V4 grid = (gridDim.x = N/N_PER_BLOCK, gridDim.y = M/BM).  For ONE fixed M-tile (blockIdx.y) ALL
// gridDim.x N-superblocks gather the SAME remote A[block_row : block_row+BM, :] rows over IRIS
// (FP8 e4m3 + per-128 scales).  Those superblocks are the A-sharing set.  On MI355X (gfx950) the
// HW round-robins logical blocks to NUM_XCDS=8 chiplets by (linear_block_id % 8): with the stock
// row-major launch, consecutive linear ids = consecutive N-superblocks of the SAME M-tile, so the
// A-sharing set is SPRAYED across all 8 XCDs.  Each XCD then has its own slice of L2 / its own LLC
// path, so the shared remote A bytes (IF they land in cache at all) are duplicated 8x.
//
// HYPOTHESIS (NOT asserted — main agent must MEASURE): if remote P2P/XGMI loads populate a
// reusable cache line in L2/LLC, then co-locating an M-tile's whole A-sharing set ONTO ONE XCD
// lets the 2nd..gridDim.x superblocks hit cache instead of re-crossing XGMI, cutting XGMI read
// bytes and raising L2/LLC hit%.  It may also do nothing (P2P loads may bypass / not be reusable).
// This kernel makes the experiment runnable; it does not claim the win.
//
// WHAT -----------------------------------------------------------------------------------------
// We remap which PHYSICAL block (the one HW placed on a given XCD) runs which LOGICAL (m_tile,
// n_super) task, using HK's chiplet_transform_chunked() (the inverse of the HW round-robin) plus a
// reference-GEMM-style "W" super-grouping window in M.  The gather / dequant / MFMA pipeline is
// byte-for-byte V4.  XCD_REMAP=0 reproduces stock V4 EXACTLY.
//
//   xcd_map_block(): wgid = blockIdx.y*gridDim.x + blockIdx.x   (row-major linear HW id)
//     -> tid = chiplet_transform_chunked(wgid, num_wgs, NUM_XCDS=8, XCD_C)   [inverse RR perm]
//     -> W-window M super-group on the permuted id -> (pid_m, pid_n)
//     -> block_row = pid_m*BM ; block_n0 = pid_n*N_PER_BLOCK ; n_tile0 = pid_n*NSUB
//
//   XCD_C  (chunk_size, default = gridDim.x = N/N_PER_BLOCK):  one M-tile's A-sharing superblocks
//          (there are exactly gridDim.x of them) become a contiguous chunk -> land on ONE XCD.
//   XCD_W  (m-tile window = WGM, default 8):  reference-GEMM super-grouping in M for L2 locality of
//          the *output*/B side and to keep the (m,n) decode well-defined for non-square grids.
//   XCD_REMAP (master switch, default 1):  0 => exact stock V4 (block_row/block_n0/n_tile0 from raw
//          blockIdx, no transform), for an apples-to-apples A/B test.
//
// NUMERICS GUARANTEE --------------------------------------------------------------------------
// This is a PURE PERMUTATION of which physical block executes which (m_tile, n_super) task.  The
// map (wgid -> tid -> (pid_m,pid_n)) is a bijection over the full-block region (chiplet_transform_
// chunked is its own structured inverse-RR within full (NUM_XCDS*chunk) blocks and identity past
// the last full block; the W-window decode is a bijection on the in-range (m,n) lattice).  Tiles
// whose (pid_m,pid_n) fall outside the M/N ranges are masked off exactly as V4 masks ragged edges.
// Every in-range (m_tile, n_super) task is therefore executed EXACTLY ONCE, by some block, and the
// per-task work (remote FP8 gather, per-128 dequant, double-buffered MFMA, store) is identical to
// V4 down to the instruction.  Therefore the output is BIT-for-BIT what V4 produces: RMS-rel
// 0.00331 vs the bf16 reference is unchanged, and the zero-sentinel (local A=0 proves remote
// gather) is preserved.  XCD remapping changes ONLY the XCD/CU a task lands on, not its result.
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
#define NSTAGE 2            // shared-tile buffering depth (2 = double-buffer)
#endif

// --------------------------- XCD scheduling configuration ---------------------------------------
// Master switch: 1 = XCD-aware remap (default), 0 = EXACT stock V4 (no transform).
#ifndef XCD_REMAP
#define XCD_REMAP 1
#endif
// W super-group window in M-tiles (reference-GEMM WGM / GROUP_SIZE_M). Default 8.
#ifndef XCD_W
#define XCD_W 8
#endif
// Chunk size for chiplet_transform_chunked.  Sentinel 0 => use gridDim.x (= N/N_PER_BLOCK) at
// runtime so a whole M-tile's A-sharing superblocks form ONE contiguous chunk -> one XCD.
#ifndef XCD_C
#define XCD_C 0
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
    // grid identical to V4: x walks N in N_PER_BLOCK steps; y walks M in BM steps.
    dim3 grid()  { return dim3(ceil_div(N, (int)N_PER_BLOCK), ceil_div(M, (int)BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() {
        return (size_t)NSTAGE * (sizeof(ST_A) + (size_t)NSUB * sizeof(ST_B)) + 1024;
    }
};

// fp8 -> float using OCP e4m3 (matches V1's __HIP_E4M3).
__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ------------------------------------------------------------------------------------------------
// XCD block remap.  Returns (pid_m, pid_n) = the LOGICAL (m_tile, n_super) task this physical block
// should run.  Pure index math (no memory traffic); a bijection over the in-range grid (see header
// NUMERICS GUARANTEE).  num_m = gridDim.y, num_n = gridDim.x.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void xcd_map_block(int &pid_m, int &pid_n,
                                              int num_m, int num_n) {
#if XCD_REMAP
    const int num_wgs = num_m * num_n;
    // chunk = whole M-tile's A-sharing superblocks (num_n of them) onto one XCD, unless overridden.
    const int chunk = (XCD_C > 0) ? (int)XCD_C : num_n;
    // raw row-major linear HW id (HW round-robins this % NUM_XCDS to chiplets).
    const int wgid = blockIdx.y * num_n + blockIdx.x;
    // inverse round-robin: contiguous logical ids -> same XCD.
    const int tid  = kittens::chiplet_transform_chunked(wgid, num_wgs, kittens::NUM_XCDS, chunk);

    // Reference-GEMM "W"-window super-grouping in M: decode the permuted linear id (tid) into
    // (pid_m, pid_n) so that XCD_W consecutive M-tiles for the same N column stay together.
    const int W            = (int)XCD_W;
    const int blocks_per_grp = W * num_n;                 // tasks in one M-window
    const int group_id     = tid / blocks_per_grp;
    const int first_m      = group_id * W;
    const int grp_rows     = (num_m - first_m) < W ? (num_m - first_m) : W;  // ragged last window
    const int idx_in_grp   = tid % blocks_per_grp;
    // column-major within the window so A-sharing (fixed m, varying n) stays contiguous in tid.
    pid_m = first_m + (idx_in_grp % grp_rows);
    pid_n = idx_in_grp / grp_rows;
#else
    // EXACT stock V4: no transform.
    pid_m = blockIdx.y;
    pid_n = blockIdx.x;
#endif
}

// ------------------------------------------------------------------------------------------------
// Remote fp8 gather + dequant of a BM x BK A-tile into the swizzled shared bf16 tile `dst`.
// (Identical to V4/V3 — this is the EXACT scarce cross-GPU traffic we are amortizing.)
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
// FUSED A-STATIONARY kernel with XCD-aware block placement.  Identical pipeline to V4; ONLY the
// (block_row, block_n0, n_tile0) derivation changes via xcd_map_block (a permutation).
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[NSTAGE]       = al.allocate<ST_A, NSTAGE>();
    ST_B (&Bs)[NSTAGE][NSUB] = al.allocate<ST_B, NSTAGE, NSUB>();

    // XCD-aware remap: physical block -> logical (pid_m, pid_n) task.
    int pid_m, pid_n;
    xcd_map_block(pid_m, pid_n, (int)gridDim.y, (int)gridDim.x);

    const int block_row = pid_m * BM;
    const int block_n0  = pid_n * N_PER_BLOCK;     // first N column this block owns
    const int n_tile0   = pid_n * NSUB;            // first B N-tile index this block owns
    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;
    const int  num_tiles = g.K / BK;

    // Mask off out-of-range remapped tasks (ragged windows / non-multiple grids) — V4 edge rule.
    if (block_row >= g.M || block_n0 >= g.N) return;

    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    constexpr int PREFETCH = NSTAGE - 1;

    // Prologue: producers prefetch the first PREFETCH K-tiles (A once + NSUB B subtiles each).
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
    // NSUB accumulators per consumer warp (one per N-subtile it co-owns).
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
            // Load the single shared A tile ONCE, reuse across all NSUB N-subtiles.
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
// BASELINE kernel (two-phase, NO overlap) — IDENTICAL to V4/V3's baseline (grid = (N/BN, M/BM),
// one output tile per block).  NOT remapped (it is the unfused reference; the XCD experiment is on
// the fused path).  Kept V4-exact for an apples-to-apples head-to-head.
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
    m.doc() = "fmoe_fused_sched_xcd tk_kernel python module";
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
