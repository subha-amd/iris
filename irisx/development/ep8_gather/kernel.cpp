// ep8_gather / kernel.cpp
// ================================================================================================
// Standalone candidate that EXERCISES ep8_gather.h's multi-source A-tile gather.  This is NOT the
// fully-fused V5 GEMM — it is a focused correctness harness for the gather itself: for each
// (M-tile, K-tile) it runs the multi-source gather+dequant into the swizzled shared ST_A tile, then
// reads the tile back into a flat row-major dequantized output D[M_packed, K] in LOCAL HBM so the
// CPU reference can compare it bit-for-bit (modulo fp8 dequant) against the routed/gathered ground
// truth — including per-rank ZERO-SENTINEL rows (unrouted/tail rows must come back exactly 0).
//
// The harness chooses Path 1 vs Path 2 per M-tile exactly as V5 would (tile_is_single_source), so a
// single run validates BOTH paths plus the local-source short-circuit (src_rank == cur_rank).
//
// np=8 (EP8): every rank holds, on its symmetric IRIS heap, its OWN activation buffer A_fp8[Msrc,K]
// + scales A_sc[Msrc,K/128].  The consumer rank (default 0) additionally holds the packed metadata
// (route_segment[] + the tile's seg_tile_view fields) and the output D.  The kernel gathers rows
// from up to 8 source ranks into D and we check on the CPU.
// ================================================================================================
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "ep8_gather.h"
#include <cstdio>
using namespace kittens;
using namespace ep8_gather;

#ifndef BM
#define BM 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef NUM_WORKERS
#define NUM_WORKERS 4
#endif
#define NUM_THREADS (NUM_WORKERS * kittens::WARP_THREADS)

using ELEM = bf16;
using ST_A = st_bf<BM, BK, st_16x32_s>;

struct gather_globals {
    gl<bf16,  -1, -1, -1, -1> a;       // [Msrc, K/2] fp8 e4m3 reinterpreted as bf16 (local heap)
    gl<float, -1, -1, -1, -1> sc;      // [Msrc, K/128] fp32 scales (local heap)
    gl<bf16,  -1, -1, -1, -1> d;       // [Mpacked, K] dequantized gather OUTPUT (local heap)
    gl<int,   -1, -1, -1, -1> seg;     // route_segment[] flattened: [Nseg, 5] ints (local heap)
    gl<int,   -1, -1, -1, -1> tilemeta;// per-tile [Ntile, 4]: seg_begin, seg_count, tile_dst0, valid_rows
    iris::iris_device_view iris_ctx;
    int Msrc, Mpacked, K, Nseg, Ntile;
    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(K, (int)BK), Ntile); }   // x: K-tiles, y: M-tiles
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return sizeof(ST_A) + BM + 1024; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void gather_probe(gather_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As) = al.allocate<ST_A>();
    signed char* row_seg_mem = reinterpret_cast<signed char*>(al.allocate<char, BM>());
    signed char (&row_seg)[BM] = *reinterpret_cast<signed char(*)[BM]>(row_seg_mem);

    const int m_tile = blockIdx.y;
    const int k_tile = blockIdx.x;
    const int tid    = threadIdx.x;
    const int nthreads = NUM_THREADS;
    const int k0 = k_tile * BK;

    // ---- load this tile's seg_tile_view from tilemeta ----
    seg_tile_view v;
    v.segs       = reinterpret_cast<const route_segment*>(&g.seg[{0,0,0,0}]);
    v.seg_begin  = g.tilemeta[{0,0,m_tile,0}];
    v.seg_count  = g.tilemeta[{0,0,m_tile,1}];
    v.tile_dst0  = g.tilemeta[{0,0,m_tile,2}];
    v.valid_rows = g.tilemeta[{0,0,m_tile,3}];

    const fp8_t* a_base  = reinterpret_cast<const fp8_t*>(&g.a[{0,0,0,0}]);
    const float* sc_base = &g.sc[{0,0,0,0}];

    // ---- choose path ----
    int fast_src_rank = 0, fast_row_off = 0;
    const bool fast = tile_is_single_source(v, &fast_src_rank, &fast_row_off);
    if (!fast) build_row_seg_map<BM>(row_seg, v, tid, nthreads);
    else       __syncthreads();

    gather_dequant_A_tile_multisource<ST_A, ELEM, BM, BK, 16>(
        As, row_seg, v, a_base, sc_base, g.K, g.Msrc, k0,
        g.iris_ctx, tid, nthreads, fast, fast_src_rank, fast_row_off);
    __syncthreads();

    // ---- read the swizzled ST_A tile back into row-major D[Mpacked,K] for CPU check ----
    constexpr int SUBR = ST_A::underlying_subtile_rows;
    constexpr int SUBC = ST_A::underlying_subtile_cols;
    constexpr int SUBN = ST_A::underlying_subtile_elements;
    for (int idx = tid; idx < BM * BK; idx += nthreads) {
        const int r = idx / BK, k = idx % BK;
        const int gr = v.tile_dst0 + r;
        if (gr >= g.Mpacked) continue;
        const int sub_row = r / SUBR, sub_col = k / SUBC;
        const int sub_id  = sub_row * ST_A::underlying_subtiles_per_row + sub_col;
        const int rr = r % SUBR, cc = k % SUBC;
        const uint32_t intra_byte = ST_A::swizzle({rr, cc});
        const char* base = reinterpret_cast<const char*>(&As.data[0]) + (size_t)sub_id * SUBN * sizeof(ELEM);
        const ELEM val = *reinterpret_cast<const ELEM*>(base + intra_byte);
        g.d[{0,0,gr,k0 + k}] = val;
    }
}

void dispatch_gather(gather_globals g) {
    const unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)gather_probe, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    gather_probe<<<g.grid(), g.block(), mem, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "ep8_gather multi-source gather probe";
    py::bind_function<dispatch_gather>(m, "dispatch_gather",
        &gather_globals::a,
        &gather_globals::sc,
        &gather_globals::d,
        &gather_globals::seg,
        &gather_globals::tilemeta,
        &gather_globals::iris_ctx,
        &gather_globals::Msrc,
        &gather_globals::Mpacked,
        &gather_globals::K,
        &gather_globals::Nseg,
        &gather_globals::Ntile
    );
}
