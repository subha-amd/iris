// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// V1 MoE dispatch + pack + FP8 quant kernel for the IRISX symmetric-heap RMA library.
//
// Builds directly on V0b (precomputed-offset, sender-owned-slice, no remote atomics).
// Folds per-128-element-group FP8 e4m3 quantization into the remote-store path so the
// kernel writes the EXACT input layout the production fmoe_bf16_blockscaleFp8 expert
// GEMM consumes: fp8 e4m3 values (expert-major) + per-128-group fp32 scales.
//
// This collapses the production pre-GEMM chain:
//   EpDispatch -> opus_moe_sorting(P0) -> opus_moe_sorting(P23) -> dynamic_per_group_scaled_quant
// into a single device-side dispatch_pack_quant kernel, and ~halves the XGMI payload:
//   bf16 token       7168 * 2          = 14,336 B
//   fp8 + 56 scales  7168 + 56*4       =  7,392 B   (~51.6%)
//
// R1-0528 shapes: H=7168, top-k=8, 256 experts => 32 experts/GPU at EP=8, world=8.
// Quant: FP8 e4m3 (OCP), block group = 128 -> 56 groups per 7168-element token.
//
// Prior-art requirements (PRIOR_ART.md) carried over from V0b:
//   * route_slot[token][topk] first-class output (V3 combine needs it).
//   * grid = device_props.multiProcessorCount (not hardcoded).
//   * bulk cross-rank completion signal (not a heavy barrier).
//   * vectorized remote stores; the natural V1 chunk is the 128-elem quant group.
//   * compile-time in-kernel timestamp profiling mode (PROFILE=1): quant vs store cycles.

#include "iris/iris.hpp"
#include "utils.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using bf16 = __hip_bfloat16;
using fp8_t = __hip_fp8_storage_t;  // unsigned char

// 16B vector type for coalesced bf16 loads (8 bf16 = 16 bytes).
struct __attribute__((aligned(16))) bf16x8 { bf16 v[8]; };

// ----- R1-0528 problem shapes -----------------------------------------------
static constexpr int H = 7168;
static constexpr int TOPK = 8;
static constexpr int N_ROUTED_EXPERTS = 256;
static constexpr int EXPERTS_PER_RANK = 32;
static constexpr int GROUP = 128;             // FP8 block-scale quant group
static constexpr int N_GROUPS = H / GROUP;    // 56 scales per token
static constexpr float FP8_E4M3_MAX = 448.0f; // OCP e4m3 max finite magnitude

// V0b: per-(expert, src_rank) capacity (sender-owned slice).
static constexpr int PER_SRC_CAPACITY = 64;

// =====================================================================================
// Pass 1 (local): per-assignment local slot index + per-(dst_rank,local_e) send counts.
// Identical machinery to V0b's count kernel.
// =====================================================================================
__global__ void moe_dispatch_count(const int* __restrict__ topk_ids,  // [T_LOCAL,TOPK]
                                   int* __restrict__ assign_slot,      // [T_LOCAL*TOPK] out
                                   int* __restrict__ send_counts,      // [world*EPR] out
                                   int T_LOCAL) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = T_LOCAL * TOPK;
  if (tid >= total) return;
  const int t = tid / TOPK;
  const int k = tid % TOPK;
  const int expert = topk_ids[t * TOPK + k];
  const int dst_rank = expert / EXPERTS_PER_RANK;
  const int local_e = expert % EXPERTS_PER_RANK;
  const int bucket = dst_rank * EXPERTS_PER_RANK + local_e;
  assign_slot[tid] = atomicAdd(&send_counts[bucket], 1);
}

// =====================================================================================
// V1 pass 2 (remote): for each routed (token,k), quantize each 128-elem group to fp8
// e4m3 and remote-store the fp8 values + per-group fp32 scale into the destination
// expert-major buffers (sender-owned slice, deterministic offset -> NO remote atomics).
//
// One block owns one assignment. The 128-thread block handles one quant group at a
// time (group-strided over the 56 groups). Each thread: loads one bf16 element, the
// block reduces |.| to a group max -> scale, then each thread quantizes its element and
// remote-stores one fp8 byte; thread 0 remote-stores the fp32 scale.
//
// Output layout (the fmoe_bf16_blockscaleFp8 input we reproduce; see V1_RESULTS.md
// validation #2 -- flagged for ATOM byte-confirmation):
//   packed_fp8 [local_e][src_rank][slot][H]        fp8 e4m3, expert-major
//   packed_sc  [local_e][src_rank][slot][N_GROUPS] fp32 per-128-group scale
// =====================================================================================
// Design: a 256-thread block holds 8 "tiles" of 32 lanes. Each tile owns ONE 128-elem
// quant group and processes 4 CONTIGUOUS elements per lane (32*4 = 128). The group max
// is a pure 32-lane sub-warp reduction (NO __syncthreads -> groups run independently),
// and each lane packs its 4 fp8 bytes into one 32-bit word for a single 4-byte remote
// store (32 stores/group instead of 128). Tiles stride over the 56 groups of the token.
static constexpr int TILE = 32;            // lanes per quant group
static constexpr int ELEMS_PER_LANE = GROUP / TILE;  // 4
static constexpr int TILES_PER_BLOCK = 256 / TILE;   // 8 groups in flight per block

template <int PROFILE = 0>
__global__ void ep_dispatch_pack_quant_fp8(const bf16* __restrict__ hidden,    // [T_LOCAL,H]
                                           const int* __restrict__ topk_ids,    // [T_LOCAL,TOPK]
                                           const int* __restrict__ assign_slot, // [T_LOCAL*TOPK]
                                           fp8_t* packed_fp8,  // [EPR][world][PER_SRC][H]
                                           float* packed_sc,   // [EPR][world][PER_SRC][N_GROUPS]
                                           int* route_slot,    // [T_LOCAL,TOPK] out
                                           unsigned long long* prof,  // [2] (quant, store)
                                           int T_LOCAL,
                                           iris::iris_device_view view) {
  const int my_rank = view.cur_rank();
  const int world = view.world_size();
  const int n_assign = T_LOCAL * TOPK;
  const int tile = threadIdx.x / TILE;       // which group-tile in this block [0,8)
  const int lane = threadIdx.x % TILE;       // lane within the tile [0,32)

  for (int assignment = blockIdx.x; assignment < n_assign; assignment += gridDim.x) {
    const int t = assignment / TOPK;
    const int k = assignment % TOPK;

    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK;
    const int local_e = expert % EXPERTS_PER_RANK;
    const int slot = assign_slot[assignment];
    if (slot >= PER_SRC_CAPACITY) continue;
    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;

    const std::size_t base =
        ((static_cast<std::size_t>(local_e) * world + my_rank) * PER_SRC_CAPACITY + slot);
    fp8_t* dst_v = packed_fp8 + base * H;
    float* dst_s = packed_sc + base * N_GROUPS;
    const bf16* src = hidden + static_cast<std::size_t>(t) * H;

    // each tile strides over the 56 groups
    for (int g = tile; g < N_GROUPS; g += TILES_PER_BLOCK) {
      const int e0 = g * GROUP + lane * ELEMS_PER_LANE;  // first of 4 contiguous elems

      unsigned long long c0 = 0;
      if (PROFILE && lane == 0) c0 = clock64();

      // load 4 contiguous bf16 (vectorized) and find this lane's |max|
      float x[ELEMS_PER_LANE];
      float a = 0.0f;
      #pragma unroll
      for (int j = 0; j < ELEMS_PER_LANE; j++) {
        x[j] = static_cast<float>(src[e0 + j]);
        a = fmaxf(a, fabsf(x[j]));
      }
      // 32-lane sub-warp max reduction (no block sync)
      #pragma unroll
      for (int o = TILE / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, TILE));
      const float scale = (a > 0.0f) ? (a / FP8_E4M3_MAX) : 1.0f;
      const float inv = 1.0f / scale;
      if (PROFILE && lane == 0) atomicAdd(&prof[0], clock64() - c0);

      // quantize 4 elems -> pack into one 32-bit word -> single 4-byte remote store
      unsigned long long s0 = 0;
      if (PROFILE && lane == 0) s0 = clock64();
      unsigned int word = 0;
      #pragma unroll
      for (int j = 0; j < ELEMS_PER_LANE; j++) {
        const fp8_t q = __hip_cvt_float_to_fp8(x[j] * inv, __HIP_SATFINITE, __HIP_E4M3);
        word |= static_cast<unsigned int>(q) << (8 * j);
      }
      view.store(reinterpret_cast<unsigned int*>(dst_v + e0), word, dst_rank);
      if (lane == 0) view.store(dst_s + g, scale, dst_rank);  // one fp32 scale / group
      if (PROFILE && lane == 0) atomicAdd(&prof[1], clock64() - s0);
    }
  }
}

// =====================================================================================
// Bulk cross-rank completion signal (identical to V0b).
// =====================================================================================
__global__ void completion_signal(int* signals, int epoch, iris::iris_device_view view) {
  const int my_rank = view.cur_rank();
  const int world = view.world_size();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int r = 0; r < world; r++)
      view.atomic_store<int, iris::memory_scope_system>(
          &signals[my_rank], epoch, r, iris::memory_order_release);
    for (int s = 0; s < world; s++)
      while (view.atomic_load<int, iris::memory_scope_system>(
                 &signals[s], my_rank, iris::memory_order_acquire) < epoch) {}
  }
}

// =====================================================================================
// V0b bf16 reference path (for in-isolation FP8-payload-win comparison at same T_local).
// =====================================================================================
__global__ void moe_dispatch_pack_v0b(const bf16* __restrict__ hidden,
                                      const int* __restrict__ topk_ids,
                                      const int* __restrict__ assign_slot,
                                      bf16* packed, int* route_slot, int T_LOCAL,
                                      iris::iris_device_view view) {
  const int my_rank = view.cur_rank();
  const int world = view.world_size();
  const int HVEC = H / 8;
  const int n_assign = T_LOCAL * TOPK;
  for (int a = blockIdx.x; a < n_assign; a += gridDim.x) {
    const int t = a / TOPK, k = a % TOPK;
    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK, local_e = expert % EXPERTS_PER_RANK;
    const int slot = assign_slot[a];
    if (slot >= PER_SRC_CAPACITY) continue;
    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;
    const std::size_t idx = ((static_cast<std::size_t>(local_e) * world + my_rank) * PER_SRC_CAPACITY + slot);
    const bf16x8* s = reinterpret_cast<const bf16x8*>(hidden + static_cast<std::size_t>(t) * H);
    bf16x8* d = reinterpret_cast<bf16x8*>(packed + idx * H);
    for (int h = threadIdx.x; h < HVEC; h += blockDim.x) view.store(d + h, s[h], dst_rank);
  }
}

// ----- host helpers ---------------------------------------------------------
int main(int argc, char** argv) {
  const auto [rank, world_size] = iris::mpi::initialize();
  const bool profile = (argc > 1 && std::atoi(argv[1]) == 1);

  int dev = 0; hip_try(hipGetDevice(&dev));
  hipDeviceProp_t props; hip_try(hipGetDeviceProperties(&props, dev));
  const int num_cu = props.multiProcessorCount;
  const int grid = num_cu;

  if (rank == 0) {
    std::printf("=== V1 MoE dispatch+pack+FP8quant (H=%d, topk=%d, experts=%d, world=%d) ===\n",
                H, TOPK, N_ROUTED_EXPERTS, world_size);
    std::printf("group=%d -> %d scales/token; FP8_E4M3_MAX=%.0f; grid=%d CUs\n",
                GROUP, N_GROUPS, FP8_E4M3_MAX, grid);
    std::printf("payload/token: bf16=%d B  fp8+scales=%d B  (%.1f%%)\n",
                H * 2, H + N_GROUPS * 4, 100.0 * (H + N_GROUPS * 4) / (H * 2));
  }

  // T_local sweep matching R1 decode batch buckets.
  const int sweep[] = {8, 16, 32, 64, 128};
  const int n_sweep = sizeof(sweep) / sizeof(int);
  const int T_MAX = 128;

  const std::size_t heap_size = 1024ULL * 1024 * 1024;  // 1 GB
  iris::iris iris_obj(heap_size, rank, world_size);
  auto view = iris_obj.get_device_view();

  // allocate for the MAX T_local; reuse across sweep.
  auto* hidden = iris_obj.allocate<bf16>(static_cast<std::size_t>(T_MAX) * H);
  auto* packed_fp8 = iris_obj.allocate<fp8_t>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  auto* packed_sc = iris_obj.allocate<float>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * N_GROUPS);
  auto* packed_bf = iris_obj.allocate<bf16>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  auto* assign_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_MAX) * TOPK);
  auto* send_counts = iris_obj.allocate<int>(static_cast<std::size_t>(world_size) * EXPERTS_PER_RANK);
  auto* route_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_MAX) * TOPK);
  auto* signals = iris_obj.allocate<int>(world_size);
  auto* prof = iris_obj.allocate<unsigned long long>(2);
  auto* topk_ids = iris_obj.allocate<int>(static_cast<std::size_t>(T_MAX) * TOPK);
  auto* remote_cnt = iris_obj.allocate<int>(1);

  if (!hidden || !packed_fp8 || !packed_sc || !packed_bf || !assign_slot || !send_counts ||
      !route_slot || !signals || !prof || !topk_ids || !remote_cnt) {
    std::printf("rank %d: allocation failed\n", rank);
    iris::mpi::finalize();
    return 1;
  }

  // init synthetic data (matches V0 scheme) for T_MAX tokens.
  std::vector<bf16> h_hidden(static_cast<std::size_t>(T_MAX) * H);
  std::vector<int> h_topk(static_cast<std::size_t>(T_MAX) * TOPK);
  std::mt19937 gen(1234 + rank);
  std::uniform_int_distribution<int> edist(0, N_ROUTED_EXPERTS - 1);
  for (int t = 0; t < T_MAX; t++) {
    for (int h = 0; h < H; h++)
      h_hidden[static_cast<std::size_t>(t) * H + h] =
          static_cast<bf16>(static_cast<float>((rank * 131 + t * 7 + h) % 17) * 0.5f);
    int picked[TOPK];
    for (int k = 0; k < TOPK; k++) {
      int e; bool dup;
      do { e = edist(gen); dup = false; for (int j = 0; j < k; j++) if (picked[j] == e) dup = true; } while (dup);
      picked[k] = e; h_topk[static_cast<std::size_t>(t) * TOPK + k] = e;
    }
  }
  hip_try(hipMemcpy(hidden, h_hidden.data(), h_hidden.size() * sizeof(bf16), hipMemcpyHostToDevice));
  hip_try(hipMemcpy(topk_ids, h_topk.data(), h_topk.size() * sizeof(int), hipMemcpyHostToDevice));

  const int block = 256;        // bf16 path block
  const int qblock = 256;       // 8 tiles x 32 lanes; each tile owns one 128-elem group
  const int iters = 20;

  if (rank == 0) {
    std::printf("\n%-8s | %-28s | %-28s | %-10s\n",
                "T_local", "V1 fp8 (ms / GB/s / us)", "V0b bf16 (ms / GB/s / us)", "remote%");
    std::printf("---------+------------------------------+------------------------------+--------\n");
  }

  for (int si = 0; si < n_sweep; si++) {
    const int T_LOCAL = sweep[si];
    const int n_assign = T_LOCAL * TOPK;

    // measure remote fraction (assignments whose dst_rank != my_rank)
    int local_remote = 0;
    for (int t = 0; t < T_LOCAL; t++)
      for (int k = 0; k < TOPK; k++) {
        const int e = h_topk[static_cast<std::size_t>(t) * TOPK + k];
        if (e / EXPERTS_PER_RANK != rank) local_remote++;
      }
    int sum_remote = 0, sum_assign = 0;
    MPI_Allreduce(&local_remote, &sum_remote, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int la = n_assign; MPI_Allreduce(&la, &sum_assign, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const double remote_frac = 100.0 * sum_remote / sum_assign;

    // ---------- V1 FP8 ----------
    hip_try(hipMemset(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int)));
    hip_try(hipMemset(signals, 0, world_size * sizeof(int)));
    hip_try(hipMemset(prof, 0, 2 * sizeof(unsigned long long)));
    iris_obj.barrier();

    auto run_v1 = [&](int epoch) {
      hipMemsetAsync(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int));
      moe_dispatch_count<<<(n_assign + block - 1) / block, block>>>(topk_ids, assign_slot, send_counts, T_LOCAL);
      if (profile)
        ep_dispatch_pack_quant_fp8<1><<<grid, qblock>>>(hidden, topk_ids, assign_slot, packed_fp8, packed_sc, route_slot, prof, T_LOCAL, view);
      else
        ep_dispatch_pack_quant_fp8<0><<<grid, qblock>>>(hidden, topk_ids, assign_slot, packed_fp8, packed_sc, route_slot, prof, T_LOCAL, view);
      completion_signal<<<1, 1>>>(signals, epoch, view);
    };
    run_v1(1);
    iris_obj.barrier();
    gpu_timer t1; t1.start();
    for (int i = 0; i < iters; i++) run_v1(2 + i);
    iris_obj.barrier();
    t1.stop();
    const float v1_ms = t1.elapsed_ms() / iters;

    // ---------- V0b bf16 ----------
    hip_try(hipMemset(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int)));
    hip_try(hipMemset(signals, 0, world_size * sizeof(int)));
    iris_obj.barrier();
    auto run_b = [&](int epoch) {
      hipMemsetAsync(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int));
      moe_dispatch_count<<<(n_assign + block - 1) / block, block>>>(topk_ids, assign_slot, send_counts, T_LOCAL);
      moe_dispatch_pack_v0b<<<grid, block>>>(hidden, topk_ids, assign_slot, packed_bf, route_slot, T_LOCAL, view);
      completion_signal<<<1, 1>>>(signals, epoch, view);
    };
    run_b(1);
    iris_obj.barrier();
    gpu_timer tb; tb.start();
    for (int i = 0; i < iters; i++) run_b(2 + i);
    iris_obj.barrier();
    tb.stop();
    const float b_ms = tb.elapsed_ms() / iters;

    // bytes moved / rank / iter (all assignments move their payload)
    const double v1_bytes = static_cast<double>(n_assign) * (H + N_GROUPS * 4);
    const double b_bytes = static_cast<double>(n_assign) * (H * 2);
    const double v1_gbs = v1_bytes / (v1_ms / 1e3) / 1e9;
    const double b_gbs = b_bytes / (b_ms / 1e3) / 1e9;

    if (rank == 0) {
      std::printf("%-8d | %7.4f / %6.1f / %6.1f   | %7.4f / %6.1f / %6.1f   | %5.1f\n",
                  T_LOCAL, v1_ms, v1_gbs, v1_ms * 1e3, b_ms, b_gbs, b_ms * 1e3, remote_frac);
    }

    if (profile && rank == 0) {
      unsigned long long h_prof[2];
      hip_try(hipMemcpy(h_prof, prof, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
      std::printf("  [profile T=%d] quant-cycles=%llu  remote-store-cycles=%llu  (store/quant=%.2fx)\n",
                  T_LOCAL, h_prof[0], h_prof[1],
                  h_prof[0] ? (double)h_prof[1] / h_prof[0] : 0.0);
    }
  }

  if (rank == 0) {
    std::printf("\nBaselines: production EpDispatch+sort x2+MoE-input quant ~= 46 us/instance (pre-GEMM envelope);\n");
    std::printf("           V0b bf16 reference shown per-row (FP8 payload win in isolation).\n");
  }

  iris_obj.barrier();
  iris::mpi::finalize();
  return 0;
}
