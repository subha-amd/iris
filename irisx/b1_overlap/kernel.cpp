// b1_overlap / kernel.cpp
// ================================================================================================
// P4 — host-scheduled chunk overlap for B1-dispatch + B0-class grouped GEMM.
//
// This candidate is deliberately conservative.  It does NOT introduce device-side ready flags, and it
// does NOT make GEMM blocks spin on unavailable data.  Instead, the Python driver splits the packed
// row space into B0-aligned chunks and schedules:
//
//   gather/dequant chunk i on stream_gather -> record event_i
//   grouped B0 GEMM chunk i on stream_gemm -> wait(event_i), then run
//
// Waiting happens in the HIP stream dependency graph, not inside resident GPU blocks.  That is the key
// difference from P1/P2: no consumer block occupies a CU just to poll a flag.
//
// The three intended invariants are:
//   1. A crosses XGMI once: gather_pack_range writes the same local packed fp8+scale buffer as
//      b1_dispatch phase 1, one row range at a time.
//   2. The expert GEMM is B0-class: grouped_b0_chunk consumes bf16 A from local HBM and uses the same
//      256x256x64 8-wave ping-pong body as grouped_b0.
//   3. No device-side spin-wait: inter-chunk dependencies are stream events owned by the host driver.
//
// The directly comparable same-kernel baseline is in example.py: it launches the exact same chunk
// kernels serially (all gather/dequant chunks, then all GEMM chunks).  The only difference in pipeline
// mode is stream/event scheduling.
// ================================================================================================
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_fp8.h>
#include "../b1_dispatch/ep8_gather.h"
#include <cstdio>

using namespace kittens;

using fp8_t = __hip_fp8_storage_t;
static constexpr int QGROUP = 128;

// -------------------------------------------------------------------------------------------------
// Multi-source route resolver reused from b1_dispatch/ep8_gather.h.
//
// Keep this candidate composition-based: previous overlap attempts failed when they rewrote the
// route resolver.  The README build flow copies b1_overlap and b1_dispatch as sibling directories,
// so this relative include resolves in both this repo and the HipKittens distributed-kernels tree.
// -------------------------------------------------------------------------------------------------
using ep8_gather::route_segment;
using ep8_gather::seg_tile_view;
using ep8_gather::tile_is_single_source;
using ep8_gather::build_row_seg_map;
using ep8_gather::SEG_NONE;

__device__ __forceinline__ float fp8_to_f32(fp8_t b) {
    return (float)__half(__hip_cvt_fp8_to_halfraw(b, __HIP_E4M3));
}

// ================================================================================================
// Stage A: gather/pack a contiguous range of route tiles into local packed fp8+scale HBM.
// ================================================================================================
#ifndef GP_BM
#define GP_BM 64
#endif
#ifndef GP_THREADS
#define GP_THREADS 256
#endif

struct gather_range_globals {
    gl<bf16,  -1, -1, -1, -1> a_src;    // [Msrc, K/2] fp8-as-bf16 on every rank
    gl<float, -1, -1, -1, -1> sc_src;   // [Msrc, K/128]
    gl<bf16,  -1, -1, -1, -1> a_dst;    // [Mpacked, K/2] local packed fp8-as-bf16
    gl<float, -1, -1, -1, -1> sc_dst;   // [Mpacked, K/128]
    gl<int,   -1, -1, -1, -1> seg;      // [Nseg,5]
    gl<int,   -1, -1, -1, -1> tilemeta; // [Ntile,4]
    iris::iris_device_view iris_ctx;
    int Msrc, Mpacked, K, Nseg, Ntile;
    int tile_begin, tile_count;
    hipStream_t stream;
    dim3 grid()  { return dim3(tile_count > 0 ? tile_count : 1); }
    dim3 block() { return dim3(GP_THREADS); }
};

__global__ __launch_bounds__(GP_THREADS, 1)
void gather_pack_range_kernel(gather_range_globals g) {
    __shared__ int row_seg[GP_BM];

    const int local_tile = blockIdx.x;
    const int m_tile = g.tile_begin + local_tile;
    const int tid = threadIdx.x;
    const int nthreads = GP_THREADS;
    const int K = g.K;
    const int NG = K / QGROUP;
    iris::iris_device_view ctx = g.iris_ctx;
    const int cur_rank = ctx.cur_rank();

    if (local_tile >= g.tile_count || m_tile >= g.Ntile) return;

    seg_tile_view v;
    v.segs       = reinterpret_cast<const route_segment*>(&g.seg[{0, 0, 0, 0}]);
    v.seg_begin  = g.tilemeta[{0, 0, m_tile, 0}];
    v.seg_count  = g.tilemeta[{0, 0, m_tile, 1}];
    v.tile_dst0  = g.tilemeta[{0, 0, m_tile, 2}];
    v.valid_rows = g.tilemeta[{0, 0, m_tile, 3}];

    const fp8_t* a_src_base = reinterpret_cast<const fp8_t*>(&g.a_src[{0, 0, 0, 0}]);
    const float* sc_src_base = &g.sc_src[{0, 0, 0, 0}];
    fp8_t* a_dst_base = reinterpret_cast<fp8_t*>(&g.a_dst[{0, 0, 0, 0}]);
    float* sc_dst_base = &g.sc_dst[{0, 0, 0, 0}];

    int fast_src_rank = 0, fast_row_off = 0;
    const bool fast = tile_is_single_source(v, &fast_src_rank, &fast_row_off);
    if (!fast) build_row_seg_map<GP_BM>(row_seg, v, tid, nthreads);
    else       __syncthreads();

    const int chunks_per_row = K / 16;
    const long total_chunks = (long)GP_BM * chunks_per_row;

    for (long ci = tid; ci < total_chunks; ci += nthreads) {
        const int r = (int)(ci / chunks_per_row);
        const int kc = (int)(ci % chunks_per_row) * 16;
        const int dst_row = v.tile_dst0 + r;
        if (dst_row >= g.Mpacked) continue;

        int src_rank, src_row;
        bool valid;
        if (fast) {
            valid = (r < v.valid_rows);
            src_rank = fast_src_rank;
            src_row = (v.tile_dst0 + r) + fast_row_off;
        } else {
            const int sidx = row_seg[r];
            if (sidx == SEG_NONE) {
                valid = false;
                src_rank = cur_rank;
                src_row = 0;
            } else {
                const route_segment s = v.segs[sidx];
                const int local_dst = (v.tile_dst0 + r) - s.dst_row_begin;
                src_rank = s.src_rank;
                src_row = s.src_row_begin + local_dst;
                valid = true;
            }
        }

        uint4 vbytes;
        if (valid && src_row < g.Msrc && (kc + 16) <= K) {
            const uint4* sp = reinterpret_cast<const uint4*>(
                a_src_base + (size_t)src_row * K + kc);
            vbytes = (src_rank == cur_rank) ? *sp : ctx.load(sp, src_rank);
        } else {
            vbytes = make_uint4(0u, 0u, 0u, 0u);
        }
        *reinterpret_cast<uint4*>(a_dst_base + (size_t)dst_row * K + kc) = vbytes;

        const int grp = kc / QGROUP;
        if (grp < NG) {
            float scale = 0.0f;
            if (valid && src_row < g.Msrc) {
                const float* spc = sc_src_base + (size_t)src_row * NG + grp;
                scale = (src_rank == cur_rank) ? *spc : ctx.load(spc, src_rank);
            }
            sc_dst_base[(size_t)dst_row * NG + grp] = scale;
        }
    }
}

void dispatch_gather_pack_range(gather_range_globals g) {
    if (g.tile_count <= 0) return;
    gather_pack_range_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
// Stage B: dequant a contiguous packed row range from fp8+scale HBM into local bf16 HBM.
// ================================================================================================
struct dequant_range_globals {
    gl<bf16,  -1, -1, -1, -1> a_fp8;   // [Mpacked, K/2] fp8-as-bf16
    gl<float, -1, -1, -1, -1> a_sc;    // [Mpacked, K/128]
    gl<bf16,  -1, -1, -1, -1> a_bf16;  // [Mpacked, K]
    int Mpacked, K, row_begin, row_count;
    hipStream_t stream;
    dim3 grid()  { return dim3(row_count > 0 ? row_count : 1); }
    dim3 block() { return dim3(256); }
};

__global__ void dequant_packed_range_kernel(dequant_range_globals g) {
    const int row = g.row_begin + blockIdx.x;
    if (blockIdx.x >= g.row_count || row >= g.Mpacked) return;
    const int NG = g.K / QGROUP;
    const fp8_t* a_base = reinterpret_cast<const fp8_t*>(&g.a_fp8[{0, 0, 0, 0}]);
    const fp8_t* frow = a_base + (size_t)row * g.K;
    const float* srow = &g.a_sc[{0, 0, row, 0}];
    bf16* orow = &g.a_bf16[{0, 0, row, 0}];
    for (int h = threadIdx.x; h < g.K; h += blockDim.x)
        orow[h] = (bf16)(fp8_to_f32(frow[h]) * srow[h / QGROUP]);
}

void dequant_packed_range(dequant_range_globals g) {
    if (g.row_count <= 0) return;
    dequant_packed_range_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
// Stage C: B0-class grouped GEMM over an arbitrary task subset.
// ================================================================================================
static constexpr int B0_TASK_W = 4;
enum { B0_T_EXPERT = 0, B0_T_MTILE = 1, B0_T_NTILE = 2, B0_T_EROWBEG = 3 };

using B0G = kittens::group<8>;

template <int NN, int KK>
__global__ __launch_bounds__(512, 2)
void grouped_b0_gemm_bf16(const gl<bf16, -1, -1, -1, -1> A,   // [Mpacked,K]
                          const gl<bf16, -1, -1, -1, -1> B,   // [E*N,K]
                          const gl<bf16, -1, -1, -1, -1> C,   // [Mpacked,N]
                          const int* __restrict__ tasks,
                          int num_tasks) {
    constexpr int WARPS_COL = 4, WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256, BLOCK_SIZE_COL = 256, BLOCK_K = 64;
    constexpr int k_iters = KK / BLOCK_K;
    constexpr int HALF_BLOCK_SIZE_ROW = BLOCK_SIZE_ROW / 2;
    constexpr int HALF_BLOCK_SIZE_COL = BLOCK_SIZE_COL / 2;
    constexpr int REG_BLOCK_M = BLOCK_SIZE_ROW / WARPS_ROW / 2;
    constexpr int REG_BLOCK_N = BLOCK_SIZE_COL / WARPS_COL / 2;

    using B0_ST_A = st_bf<HALF_BLOCK_SIZE_ROW, BLOCK_K, st_16x32_s>;
    using B0_ST_B = st_bf<HALF_BLOCK_SIZE_COL, BLOCK_K, st_16x32_s>;
    __shared__ B0_ST_A As[2][2];
    __shared__ B0_ST_B Bs[2][2];

    using B0_RT_A = rt_bf<REG_BLOCK_M, BLOCK_K, row_l, rt_16x32_s>;
    using B0_RT_B = rt_bf<REG_BLOCK_N, BLOCK_K, row_l, rt_16x32_s>;
    using B0_RT_C = rt_fl<REG_BLOCK_M, REG_BLOCK_N, col_l, rt_16x16_s>;
    B0_RT_A a;
    B0_RT_B b0, b1;
    B0_RT_C cA, cB, cC, cD;

    const int task = blockIdx.x;
    if (task >= num_tasks) return;
    const int* tk = tasks + (size_t)task * B0_TASK_W;
    const int e = tk[B0_T_EXPERT];
    const int mt = tk[B0_T_MTILE];
    const int nt = tk[B0_T_NTILE];
    const int ERB = tk[B0_T_EROWBEG];

    const int a_row_tile = ERB / 128 + mt * 2;
    const int b_row_tile = (e * NN) / 128 + nt * 2;
    const int c_row_tile = ERB / 64 + mt * 4;
    const int c_col_tile = nt * 8;

    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);
    int tic = 0, toc = 1;

    uint32_t swizzled_offsets_A[64];
    uint32_t swizzled_offsets_B[64];
    B0G::prefill_swizzled_offsets(As[tic][0], A, swizzled_offsets_A);
    B0G::prefill_swizzled_offsets(Bs[tic][0], B, swizzled_offsets_B);

    zero(cA); zero(cB); zero(cC); zero(cD);

    B0G::load(Bs[tic][0], B, {0, 0, b_row_tile,     0}, swizzled_offsets_B);
    B0G::load(As[tic][0], A, {0, 0, a_row_tile,     0}, swizzled_offsets_A);
    B0G::load(Bs[tic][1], B, {0, 0, b_row_tile + 1, 0}, swizzled_offsets_B);
    B0G::load(As[tic][1], A, {0, 0, a_row_tile + 1, 0}, swizzled_offsets_A);

    if (warp_m == 1) { __builtin_amdgcn_s_barrier(); }
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    B0G::load(Bs[toc][0], B, {0, 0, b_row_tile,     1}, swizzled_offsets_B);
    B0G::load(As[toc][0], A, {0, 0, a_row_tile,     1}, swizzled_offsets_A);
    B0G::load(Bs[toc][1], B, {0, 0, b_row_tile + 1, 1}, swizzled_offsets_B);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
        auto bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        B0G::load(As[toc][1], A, {0, 0, a_row_tile + 1, k + 1}, swizzled_offsets_A);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        B0G::load(Bs[tic][0], B, {0, 0, b_row_tile, k + 2}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        B0G::load(As[tic][0], A, {0, 0, a_row_tile, k + 2}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        B0G::load(Bs[tic][1], B, {0, 0, b_row_tile + 1, k + 2}, swizzled_offsets_B);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        constexpr int k = k_iters - 2;
        auto bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][0], {warp_n, 0});
        load(b0, bs0);
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        B0G::load(As[toc][1], A, {0, 0, a_row_tile + 1, k + 1}, swizzled_offsets_A);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        bs0 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[toc][0], {warp_n, 0});
        load(b0, bs0);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1, toc ^= 1;
    }

    {
        auto as0 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][0], {warp_m, 0});
        load(a, as0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto bs1 = kittens::subtile_inplace<REG_BLOCK_N, BLOCK_K>(Bs[tic][1], {warp_n, 0});
        load(b1, bs1);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        auto as1 = kittens::subtile_inplace<REG_BLOCK_M, BLOCK_K>(As[tic][1], {warp_m, 0});
        load(a, as1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(cC, a, b0, cC);
        mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (warp_m == 0) { __builtin_amdgcn_s_barrier(); }

    store(C, cA, {0, 0, c_row_tile + warp_m,             c_col_tile + warp_n});
    store(C, cB, {0, 0, c_row_tile + warp_m,             c_col_tile + WARPS_COL + warp_n});
    store(C, cC, {0, 0, c_row_tile + WARPS_ROW + warp_m, c_col_tile + warp_n});
    store(C, cD, {0, 0, c_row_tile + WARPS_ROW + warp_m, c_col_tile + WARPS_COL + warp_n});
}

struct b0_chunk_globals {
    gl<bf16, -1, -1, -1, -1> a;       // [Mpacked,K] local bf16 dequanted A
    gl<bf16, -1, -1, -1, -1> b, c;    // B[E*N,K], C[Mpacked,N]
    gl<int,  -1, -1, -1, -1> tasks;   // [num_tasks,4], already filtered to this chunk
    int N, K, num_tasks;
    hipStream_t stream;
};

void dispatch_grouped_b0_chunk(b0_chunk_globals g) {
    if (g.num_tasks <= 0) return;
    const int threads = 8 * 64;
    const int* tasks = g.tasks.raw_ptr;
    if      (g.N == 2048 && g.K == 7168) grouped_b0_gemm_bf16<2048, 7168><<<g.num_tasks, threads, 0, g.stream>>>(g.a, g.b, g.c, tasks, g.num_tasks);
    else if (g.N == 4096 && g.K == 7168) grouped_b0_gemm_bf16<4096, 7168><<<g.num_tasks, threads, 0, g.stream>>>(g.a, g.b, g.c, tasks, g.num_tasks);
    else if (g.N == 7168 && g.K == 2048) grouped_b0_gemm_bf16<7168, 2048><<<g.num_tasks, threads, 0, g.stream>>>(g.a, g.b, g.c, tasks, g.num_tasks);
    else printf("b1_overlap grouped_b0_chunk: unsupported (N=%d,K=%d)\n", g.N, g.K);
}

// ================================================================================================
PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "b1_overlap P4: chunked gather/dequant + B0 grouped GEMM with host stream-event overlap";

    py::bind_function<dispatch_gather_pack_range>(m, "dispatch_gather_pack_range",
        &gather_range_globals::a_src,
        &gather_range_globals::sc_src,
        &gather_range_globals::a_dst,
        &gather_range_globals::sc_dst,
        &gather_range_globals::seg,
        &gather_range_globals::tilemeta,
        &gather_range_globals::iris_ctx,
        &gather_range_globals::Msrc,
        &gather_range_globals::Mpacked,
        &gather_range_globals::K,
        &gather_range_globals::Nseg,
        &gather_range_globals::Ntile,
        &gather_range_globals::tile_begin,
        &gather_range_globals::tile_count
    );

    py::bind_function<dequant_packed_range>(m, "dequant_packed_range",
        &dequant_range_globals::a_fp8,
        &dequant_range_globals::a_sc,
        &dequant_range_globals::a_bf16,
        &dequant_range_globals::Mpacked,
        &dequant_range_globals::K,
        &dequant_range_globals::row_begin,
        &dequant_range_globals::row_count
    );

    py::bind_function<dispatch_grouped_b0_chunk>(m, "grouped_b0_chunk",
        &b0_chunk_globals::a,
        &b0_chunk_globals::b,
        &b0_chunk_globals::c,
        &b0_chunk_globals::tasks,
        &b0_chunk_globals::N,
        &b0_chunk_globals::K,
        &b0_chunk_globals::num_tasks
    );
}
