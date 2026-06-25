// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// V1.1 MoE dispatch + pack + FP8 quant kernel for IRISX.
//
// V1.1 = V1 + two optimizations targeting V1's QUANT bound (V1_RESULTS.md validation #5:
// halving payload only gave ~23%, not ~50%, because bf16->fp8 conversion dominates):
//
//   OPT1 -- Software-pipeline quant(g+1) over the fire-and-forget remote store(g).
//           V1 did pack(g)->store(g)->pack(g+1)->store(g+1)... sequentially. Remote
//           stores are non-blocking (iris view.store is *ptr=val, no wait), so we hold
//           the produced 32-bit word + scale of group g in registers, ISSUE its store,
//           then immediately run the quant math of group g+1 (prologue / steady-state /
//           epilogue). Total time approaches max(quant, store) instead of sum(quant,store).
//
//   OPT2 -- Packed fp8x2 convert: replace 4x __hip_cvt_float_to_fp8 per lane with
//           2x __hip_cvt_float2_to_fp8x2 (each emits a 16-bit fp8x2) to halve convert
//           ops. NOTE: in this ROCm (7.2.4) hip_fp8.h, __hip_cvt_float2_to_fp8x2 is a
//           SOFTWARE wrapper over two scalar converts (amd_hip_fp8.h:743-745), not a
//           single hardware fp8x2 instruction. We use it anyway (lets the compiler see
//           the 2-wide intent / pack); honest attribution is in V1_1_RESULTS.md.
//
// Identical output layout, count/offset machinery, sender-owned slice (no remote
// atomics), bulk completion signal, device-queried grid, PROFILE mode as V1. Pipelining
// must NOT change numerical results (same scale, same per-element quant).

#include "iris/iris.hpp"
#include "utils.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using bf16 = __hip_bfloat16;
using fp8_t = __hip_fp8_storage_t;

struct __attribute__((aligned(16))) bf16x8 { bf16 v[8]; };

static constexpr int H = 7168;
static constexpr int TOPK = 8;
static constexpr int N_ROUTED_EXPERTS = 256;
static constexpr int EXPERTS_PER_RANK = 32;
static constexpr int GROUP = 128;
static constexpr int N_GROUPS = H / GROUP;
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr int PER_SRC_CAPACITY = 64;

// Pass 1 (local): per-assignment slot index + per-(dst,local_e) counts. (== V1)
__global__ void moe_dispatch_count(const int* __restrict__ topk_ids,
                                   int* __restrict__ assign_slot,
                                   int* __restrict__ send_counts,
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

static constexpr int TILE = 32;
static constexpr int ELEMS_PER_LANE = GROUP / TILE;   // 4
static constexpr int TILES_PER_BLOCK = 256 / TILE;    // 8 groups in flight per block

// ----- the per-group quant math, factored out so the pipeline can call it ahead -----
// Loads 4 contiguous bf16 from src at group g for this lane, computes the group scale
// (32-lane sub-warp max), quantizes via packed fp8x2, returns the packed 32-bit word and
// (for lane 0) the fp32 scale. NO remote traffic here -- pure on-chip quant.
__device__ __forceinline__ void quant_group(const bf16* __restrict__ src, int g, int lane,
                                            unsigned int& word_out, float& scale_out) {
  const int e0 = g * GROUP + lane * ELEMS_PER_LANE;   // first of 4 contiguous elems
  float x[ELEMS_PER_LANE];
  float a = 0.0f;
  #pragma unroll
  for (int j = 0; j < ELEMS_PER_LANE; j++) {
    x[j] = static_cast<float>(src[e0 + j]);
    a = fmaxf(a, fabsf(x[j]));
  }
  #pragma unroll
  for (int o = TILE / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, TILE));
  const float scale = (a > 0.0f) ? (a / FP8_E4M3_MAX) : 1.0f;
  const float inv = 1.0f / scale;
  // OPT2: packed fp8x2 -- two 2-wide converts instead of four scalar converts.
  const __hip_fp8x2_storage_t p01 =
      __hip_cvt_float2_to_fp8x2(float2{x[0] * inv, x[1] * inv}, __HIP_SATFINITE, __HIP_E4M3);
  const __hip_fp8x2_storage_t p23 =
      __hip_cvt_float2_to_fp8x2(float2{x[2] * inv, x[3] * inv}, __HIP_SATFINITE, __HIP_E4M3);
  word_out = static_cast<unsigned int>(p01) | (static_cast<unsigned int>(p23) << 16);
  scale_out = scale;
}

// V1.1 pass 2 (remote): software-pipelined quant/store across the 56 groups.
template <int PROFILE = 0>
__global__ void ep_dispatch_pack_quant_fp8_pipelined(
    const bf16* __restrict__ hidden,
    const int* __restrict__ topk_ids,
    const int* __restrict__ assign_slot,
    fp8_t* packed_fp8, float* packed_sc, int* route_slot,
    unsigned long long* prof, int T_LOCAL, iris::iris_device_view view) {
  const int my_rank = view.cur_rank();
  const int world = view.world_size();
  const int n_assign = T_LOCAL * TOPK;
  const int tile = threadIdx.x / TILE;
  const int lane = threadIdx.x % TILE;

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

    // Software pipeline over this tile's groups (g = tile, tile+8, tile+16, ...).
    // Prologue: quant the first group into registers. Steady state: ISSUE the store of
    // the previously-quantized group (fire-and-forget), THEN quant the next group while
    // that store is in flight. Epilogue: store the last quantized group.
    unsigned long long cq = 0, cs = 0;

    if (PROFILE && lane == 0) cq = clock64();
    unsigned int word; float scale;
    int g = tile;
    if (g < N_GROUPS) quant_group(src, g, lane, word, scale);
    if (PROFILE && lane == 0) atomicAdd(&prof[0], clock64() - cq);

    for (int gn = g + TILES_PER_BLOCK; gn < N_GROUPS; gn += TILES_PER_BLOCK) {
      // issue store of group g (non-blocking) -- overlaps with quant of group gn below
      if (PROFILE && lane == 0) cs = clock64();
      view.store(reinterpret_cast<unsigned int*>(dst_v + g * GROUP + lane * ELEMS_PER_LANE), word, dst_rank);
      if (lane == 0) view.store(dst_s + g, scale, dst_rank);
      if (PROFILE && lane == 0) atomicAdd(&prof[1], clock64() - cs);

      // quant next group while the store above is in flight
      if (PROFILE && lane == 0) cq = clock64();
      unsigned int word_n; float scale_n;
      quant_group(src, gn, lane, word_n, scale_n);
      if (PROFILE && lane == 0) atomicAdd(&prof[0], clock64() - cq);

      g = gn; word = word_n; scale = scale_n;
    }

    // epilogue: store the final quantized group
    if (g < N_GROUPS) {
      if (PROFILE && lane == 0) cs = clock64();
      view.store(reinterpret_cast<unsigned int*>(dst_v + g * GROUP + lane * ELEMS_PER_LANE), word, dst_rank);
      if (lane == 0) view.store(dst_s + g, scale, dst_rank);
      if (PROFILE && lane == 0) atomicAdd(&prof[1], clock64() - cs);
    }
  }
}

// Bulk cross-rank completion signal (== V1).
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

int main(int argc, char** argv) {
  const auto [rank, world_size] = iris::mpi::initialize();
  const bool profile = (argc > 1 && std::atoi(argv[1]) == 1);

  int dev = 0; hip_try(hipGetDevice(&dev));
  hipDeviceProp_t props; hip_try(hipGetDeviceProperties(&props, dev));
  const int grid = props.multiProcessorCount;

  if (rank == 0) {
    std::printf("=== V1.1 MoE dispatch+pack+FP8quant PIPELINED (H=%d, topk=%d, experts=%d, world=%d) ===\n",
                H, TOPK, N_ROUTED_EXPERTS, world_size);
    std::printf("group=%d -> %d scales/token; FP8_E4M3_MAX=%.0f; grid=%d CUs\n",
                GROUP, N_GROUPS, FP8_E4M3_MAX, grid);
    std::printf("OPT1=pipeline quant(g+1) over store(g)  OPT2=packed fp8x2 convert\n");
    std::printf("payload/token: bf16=%d B  fp8+scales=%d B  (%.1f%%)\n",
                H * 2, H + N_GROUPS * 4, 100.0 * (H + N_GROUPS * 4) / (H * 2));
  }

  const int sweep[] = {8, 16, 32, 64, 128};
  const int n_sweep = sizeof(sweep) / sizeof(int);
  const int T_MAX = 128;

  const std::size_t heap_size = 1024ULL * 1024 * 1024;
  iris::iris iris_obj(heap_size, rank, world_size);
  auto view = iris_obj.get_device_view();

  auto* hidden = iris_obj.allocate<bf16>(static_cast<std::size_t>(T_MAX) * H);
  auto* packed_fp8 = iris_obj.allocate<fp8_t>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  auto* packed_sc = iris_obj.allocate<float>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * N_GROUPS);
  auto* assign_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_MAX) * TOPK);
  auto* send_counts = iris_obj.allocate<int>(static_cast<std::size_t>(world_size) * EXPERTS_PER_RANK);
  auto* route_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_MAX) * TOPK);
  auto* signals = iris_obj.allocate<int>(world_size);
  auto* prof = iris_obj.allocate<unsigned long long>(2);
  auto* topk_ids = iris_obj.allocate<int>(static_cast<std::size_t>(T_MAX) * TOPK);

  if (!hidden || !packed_fp8 || !packed_sc || !assign_slot || !send_counts ||
      !route_slot || !signals || !prof || !topk_ids) {
    std::printf("rank %d: allocation failed\n", rank);
    iris::mpi::finalize();
    return 1;
  }

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

  const int block = 256;
  const int qblock = 256;
  const int iters = 20;

  if (rank == 0) {
    std::printf("\n%-8s | %-30s | %-10s\n", "T_local", "V1.1 fp8 (ms / GB/s / us)", "remote%");
    std::printf("---------+--------------------------------+--------\n");
  }

  for (int si = 0; si < n_sweep; si++) {
    const int T_LOCAL = sweep[si];
    const int n_assign = T_LOCAL * TOPK;

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

    hip_try(hipMemset(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int)));
    hip_try(hipMemset(signals, 0, world_size * sizeof(int)));
    hip_try(hipMemset(prof, 0, 2 * sizeof(unsigned long long)));
    iris_obj.barrier();

    auto run_v11 = [&](int epoch) {
      hipMemsetAsync(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int));
      moe_dispatch_count<<<(n_assign + block - 1) / block, block>>>(topk_ids, assign_slot, send_counts, T_LOCAL);
      if (profile)
        ep_dispatch_pack_quant_fp8_pipelined<1><<<grid, qblock>>>(hidden, topk_ids, assign_slot, packed_fp8, packed_sc, route_slot, prof, T_LOCAL, view);
      else
        ep_dispatch_pack_quant_fp8_pipelined<0><<<grid, qblock>>>(hidden, topk_ids, assign_slot, packed_fp8, packed_sc, route_slot, prof, T_LOCAL, view);
      completion_signal<<<1, 1>>>(signals, epoch, view);
    };
    run_v11(1);
    iris_obj.barrier();
    gpu_timer t1; t1.start();
    for (int i = 0; i < iters; i++) run_v11(2 + i);
    iris_obj.barrier();
    t1.stop();
    const float v11_ms = t1.elapsed_ms() / iters;

    const double v11_bytes = static_cast<double>(n_assign) * (H + N_GROUPS * 4);
    const double v11_gbs = v11_bytes / (v11_ms / 1e3) / 1e9;

    if (rank == 0)
      std::printf("%-8d | %7.4f / %6.1f / %6.1f      | %5.1f\n",
                  T_LOCAL, v11_ms, v11_gbs, v11_ms * 1e3, remote_frac);

    if (profile && rank == 0) {
      unsigned long long h_prof[2];
      hip_try(hipMemcpy(h_prof, prof, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
      std::printf("  [profile T=%d] quant-cycles=%llu  remote-store-cycles=%llu  (store/quant=%.2fx)\n",
                  T_LOCAL, h_prof[0], h_prof[1],
                  h_prof[0] ? (double)h_prof[1] / h_prof[0] : 0.0);
    }
  }

  iris_obj.barrier();
  iris::mpi::finalize();
  return 0;
}
