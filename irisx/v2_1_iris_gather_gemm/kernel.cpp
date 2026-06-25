// fmoe_gather_gemm / kernel.cpp
// ------------------------------------------------------------------------------------------------
// V2.1 — the remote-gather-fused GEMM.
//
// Thesis being proven:  rank 1 runs a producer/consumer GEMM whose PRODUCER warps pull each A-tile
// DIRECTLY from rank 0's IRIS heap via iris_ctx.load(&A[...], src_rank=0) — overlapping the
// cross-GPU gather of tile (t+1) with the consumer warps' MFMA of tile (t).  B (weights) and C
// (output) are local to rank 1.  This collapses the old two-phase "dispatch-then-GEMM" into a
// single fused kernel: a tile-level communication abstraction inside a compute kernel.
//
// Layouts (all row-major, plain — we deliberately avoid the bf16_gemm SRD-swizzle for A so the
// remote element gather is explicit and unambiguous):
//   A : [M, K] bf16  — the ONLY initialized copy lives on rank 0's heap; rank 1's A buffer is a
//                      same-offset placeholder used only to form the local pointer that IRIS
//                      translates to rank 0 (and, separately, to build the host reference).
//   B : [N, K] bf16  — weights, local on rank 1 (we compute C = A @ B^T).
//   C : [M, N] bf16  — output, local on rank 1.
//
// AMD scheduling is preserved: producer/consumer warpgroups, double-buffered shared tiles,
// s_waitcnt discipline, s_barrier between stages.  No NVIDIA-style wave specialization imported.
// ------------------------------------------------------------------------------------------------

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <cstdio>
using namespace kittens;

// Define GATHER_DEBUG (e.g. -DGATHER_DEBUG) to print a thread-0 diagnostic that proves the
// remote translate/load: it shows rank1's heap base, rank0's heap base, the translated remote
// address, and the value loaded from rank0's A[0,0].  Off by default.
// #define GATHER_DEBUG 1

// ----------------------------- tile / block configuration ---------------------------------------
// Per threadblock: BM x BN output tile, BK reduction tile.
constexpr int BM = 64;   // output rows per block
constexpr int BN = 64;   // output cols per block
constexpr int BK = 64;   // reduction depth per tile

// One MFMA-friendly subtile is 16x16 accum; we use HK's rt_bf 16x32 fragments and tile them.
// Keep it simple: 64x64 tiles built from 16x16 mma operations via HK rt/st abstractions.

#define NUM_PRODUCER_WORKERS (4)            // 4 producer warps gather A (+ load B)
#define NUM_CONSUMER_WORKERS (4)            // 4 consumer warps MFMA
#define NUM_WARPS (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS)
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using PG = kittens::group<NUM_PRODUCER_WORKERS>;   // producer warp group (for fast local B load)

// Shared tile types (bf16).
using ST_A = st_bf<BM, BK, st_16x32_s>;
using ST_B = st_bf<BN, BK, st_16x32_s>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b, c;   // A[M,K] (remote on rank0), B[N,K] local, C[M,N] local
    iris::iris_device_view iris_ctx;

    int M;
    int N;
    int K;
    int src_rank;        // the producing rank we gather A from (0)

    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(N, BN), ceil_div(M, BM)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

// ------------------------------------------------------------------------------------------------
// The kernel.
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // Double-buffered shared A/B tiles.
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    const int block_row = blockIdx.y * BM;   // top row of this block's output tile (in A/C row space)
    const int block_col = blockIdx.x * BN;   // top col of this block's output tile (= B row space)

    const int warp_id   = kittens::warpid();
    const bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    const bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS);
    const int  cons_id   = is_consumer ? (warp_id - NUM_PRODUCER_WORKERS) : 0;  // 0..3
    const int  laneid    = kittens::laneid();
    const int  src_rank  = g.src_rank;

    const int num_tiles = g.K / BK;

    // Fast local-B swizzled offsets (producer group collaborative load).
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile  = BN * BK * sizeof(bf16) / bytes_per_memcpy;
    uint32_t swizzled_offsets_B[memcpy_per_tile > 0 ? memcpy_per_tile : 1];
    PG::prefill_swizzled_offsets(Bs[0], g.b, swizzled_offsets_B);

    // ---- REMOTE A GATHER --------------------------------------------------------------------
    // Producer warps cooperatively pull a BM x BK A-tile from rank `src_rank`'s heap into shared
    // memory.  Each of the NUM_PRODUCER_THREADS strides over the BM*BK elements; for each element
    // it forms the LOCAL pointer &g.a[{0,0,r,k}] (an address inside rank `cur_rank`'s heap) and
    // calls iris_ctx.load(ptr, src_rank), which IRIS translates to
    //     heap_bases_[src_rank] + (ptr - heap_bases_[cur_rank])
    // i.e. the SAME offset in rank src_rank's heap.  This is the cross-GPU gather.
    auto gather_A_tile = [&](ST_A &dst, int tile) {
        const int k0 = tile * BK;
        const int tid = (warp_id * kittens::WARP_THREADS) + laneid;  // 0..NUM_PRODUCER_THREADS-1
        #pragma unroll
        for (int e = tid; e < BM * BK; e += NUM_PRODUCER_THREADS) {
            const int r = e / BK;             // local tile row
            const int k = e % BK;             // local tile col
            const int gr = block_row + r;     // global A row
            const int gk = k0 + k;            // global A col (K)
            bf16 val;
            if (gr < g.M && gk < g.K) {
                const bf16* aptr = &g.a[{0, 0, gr, gk}];          // local heap pointer
                val = g.iris_ctx.load(aptr, src_rank);            // <-- REMOTE pull from rank src_rank
            } else {
                val = base_types::constants<bf16>::zero();
            }
            // Write into the shared tile at the SWIZZLED location so the consumer's
            // swizzle-aware load(rt, st) reads it back correctly.  An st<bf16,64,64,16x32_s>
            // stores data subtile-by-subtile (16x32 base subtiles, row-major grid), with an
            // intra-subtile XOR swizzle.  Reconstruct the full element offset exactly as
            // global_to_shared does.
            constexpr int SUBR = ST_A::underlying_subtile_rows;   // 16
            constexpr int SUBC = ST_A::underlying_subtile_cols;   // 32
            constexpr int SUBN = ST_A::underlying_subtile_elements; // 512
            const int sub_row = r / SUBR, sub_col = k / SUBC;
            const int sub_id  = sub_row * ST_A::underlying_subtiles_per_row + sub_col;
            const int rr = r % SUBR, cc = k % SUBC;
            const uint32_t intra_byte = ST_A::swizzle({rr, cc});  // byte offset within the 16x32 subtile
            char* base = reinterpret_cast<char*>(&dst.data[0]) + (size_t)sub_id * SUBN * sizeof(bf16);
            *reinterpret_cast<bf16*>(base + intra_byte) = val;
        }
    };

    int tic = 0, toc = 1;

#ifdef GATHER_DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 0 && laneid == 0) {
        const bf16* aptr = &g.a[{0, 0, 0, 0}];
        uintptr_t base_cur = g.iris_ctx.get_heap_base(g.iris_ctx.cur_rank());
        uintptr_t base_src = g.iris_ctx.get_heap_base(src_rank);
        uintptr_t off = (uintptr_t)aptr - base_cur;
        uintptr_t remote = base_src + off;
        float v = (float)g.iris_ctx.load(aptr, src_rank);
        printf("[DBG] cur_rank=%d src_rank=%d aptr=%p base_cur=0x%lx base_src=0x%lx off=0x%lx remote=0x%lx A[0,0]_remote=%f\n",
               g.iris_ctx.cur_rank(), src_rank, (void*)aptr, base_cur, base_src, off, remote, v);
    }
#endif

    // Prologue: gather tile 0 of A (remote) and load tile 0 of B (local).
    if (is_producer) {
        gather_A_tile(As[tic], 0);
        // G::load COORD is in TILE units: row index = blockIdx.x (each unit = ST_B::rows = BN),
        // col index = k-tile (each unit = BK).  Do NOT pass element offsets here.
        PG::load<2, false>(Bs[tic], g.b, {0, 0, (int)blockIdx.x, 0}, swizzled_offsets_B);
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    // Consumer accumulator: BM x BN tile, split across 4 consumer warps along N.
    // Each consumer warp owns a BM x (BN/4) column strip => 64 x 16.
    constexpr int CONS_N = BN / NUM_CONSUMER_WORKERS;   // 16
    rt_fl<BM, CONS_N, col_l, rt_16x16_s> C_accum;
    if (is_consumer) zero(C_accum);

    for (int tile = 0; tile < num_tiles; ++tile, tic ^= 1, toc ^= 1) {
        // Producers prefetch the NEXT tile (remote A gather + local B) while consumers MFMA current.
        if (is_producer && tile + 1 < num_tiles) {
            gather_A_tile(As[toc], tile + 1);
            PG::load<2, false>(Bs[toc], g.b, {0, 0, (int)blockIdx.x, tile + 1}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            // Consume the current (tic) tile already in shared memory.
            // A operand: this warp reads the full BM x BK A tile.
            // B operand: this warp's CONS_N-wide column strip of the BN x BK B tile.
            rt_bf<BM, BK, row_l, rt_16x32_s> a_frag;
            rt_bf<CONS_N, BK, row_l, rt_16x32_s> b_frag;
            load(a_frag, As[tic]);
            auto b_sub = subtile_inplace<CONS_N, BK>(Bs[tic], {cons_id, 0});
            load(b_frag, b_sub);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a_frag, b_frag, C_accum);   // C += A * B^T  (B^T since B is [N,K])
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue: store C strip locally.  Output rows block_row..+BM, cols block_col + cons_id*CONS_N.
    if (is_consumer) {
        const int out_col0 = block_col + cons_id * CONS_N;
        // Write C_accum (BM x CONS_N) to g.c[{0,0, block_row+.., out_col0+..}] locally.
        store(g.c, C_accum, {0, 0, block_row / BM, out_col0 / CONS_N});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "fmoe_gather_gemm tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro",
        &micro_globals::a,
        &micro_globals::b,
        &micro_globals::c,
        &micro_globals::iris_ctx,
        &micro_globals::M,
        &micro_globals::N,
        &micro_globals::K,
        &micro_globals::src_rank
    );
}
