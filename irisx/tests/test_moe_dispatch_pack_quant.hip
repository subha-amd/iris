// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Correctness test for the V1 MoE dispatch+pack+FP8quant kernel.
//
// Each rank dispatches synthetic hidden activations with FP8 e4m3 per-128-group quant.
// The test then, on the receiver side, dequantizes the packed fp8+scales and compares
// against the bf16 reference token vector (rebuilt from MPI-gathered hidden data) WITHIN
// FP8 e4m3 tolerance (per-128-group scaled quant -> max relative error ~ 1/2^3 = ~0.0625
// for the mantissa step; we assert a generous bound and report the measured max rel err).
//
// Also validates route_slot[token][topk] is in range, and that the per-(src,expert)
// token MULTISET matches the reference (slot order within a bucket is nondeterministic).

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdint>

using bf16 = __hip_bfloat16;
using fp8_t = __hip_fp8_storage_t;

static constexpr int H = 7168;
static constexpr int TOPK = 8;
static constexpr int N_ROUTED_EXPERTS = 256;
static constexpr int EXPERTS_PER_RANK = 32;
static constexpr int GROUP = 128;
static constexpr int N_GROUPS = H / GROUP;          // 56
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr int T_LOCAL = 64;                  // small for a fast test
static constexpr int PER_SRC_CAPACITY = 64;

__global__ void k_count(const int* topk_ids, int* assign_slot, int* send_counts) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= T_LOCAL * TOPK) return;
  const int t = tid / TOPK, k = tid % TOPK;
  const int expert = topk_ids[t * TOPK + k];
  const int dst_rank = expert / EXPERTS_PER_RANK, local_e = expert % EXPERTS_PER_RANK;
  assign_slot[tid] = atomicAdd(&send_counts[dst_rank * EXPERTS_PER_RANK + local_e], 1);
}

// Mirrors ep_dispatch_pack_quant_fp8 in moe_dispatch_pack_quant.hip (tiled, vectorized).
static constexpr int TILE = 32;
static constexpr int ELEMS_PER_LANE = GROUP / TILE;        // 4
static constexpr int TILES_PER_BLOCK = 256 / TILE;         // 8
__global__ void k_quant(const bf16* hidden, const int* topk_ids, const int* assign_slot,
                        fp8_t* packed_fp8, float* packed_sc, int* route_slot,
                        iris::iris_device_view view) {
  const int my_rank = view.cur_rank(), world = view.world_size();
  const int n_assign = T_LOCAL * TOPK;
  const int tile = threadIdx.x / TILE, lane = threadIdx.x % TILE;
  for (int a = blockIdx.x; a < n_assign; a += gridDim.x) {
    const int t = a / TOPK, k = a % TOPK;
    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK, local_e = expert % EXPERTS_PER_RANK;
    const int slot = assign_slot[a];
    if (slot >= PER_SRC_CAPACITY) continue;
    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;
    const std::size_t base = ((static_cast<std::size_t>(local_e) * world + my_rank) * PER_SRC_CAPACITY + slot);
    fp8_t* dst_v = packed_fp8 + base * H;
    float* dst_s = packed_sc + base * N_GROUPS;
    const bf16* src = hidden + static_cast<std::size_t>(t) * H;
    for (int g = tile; g < N_GROUPS; g += TILES_PER_BLOCK) {
      const int e0 = g * GROUP + lane * ELEMS_PER_LANE;
      float x[ELEMS_PER_LANE]; float aa = 0.0f;
      #pragma unroll
      for (int j = 0; j < ELEMS_PER_LANE; j++) { x[j] = static_cast<float>(src[e0 + j]); aa = fmaxf(aa, fabsf(x[j])); }
      #pragma unroll
      for (int o = TILE / 2; o > 0; o >>= 1) aa = fmaxf(aa, __shfl_xor(aa, o, TILE));
      const float scale = (aa > 0.0f) ? (aa / FP8_E4M3_MAX) : 1.0f;
      const float inv = 1.0f / scale;
      unsigned int word = 0;
      #pragma unroll
      for (int j = 0; j < ELEMS_PER_LANE; j++) {
        const fp8_t q = __hip_cvt_float_to_fp8(x[j] * inv, __HIP_SATFINITE, __HIP_E4M3);
        word |= static_cast<unsigned int>(q) << (8 * j);
      }
      view.store(reinterpret_cast<unsigned int*>(dst_v + e0), word, dst_rank);
      if (lane == 0) view.store(dst_s + g, scale, dst_rank);
    }
  }
}

// host-side fp8 e4m3 dequant
static inline float deq(fp8_t v, float scale) {
  __hip_fp8_e4m3 f; f.__x = v;
  return static_cast<float>(f) * scale;
}

// token signature on dequantized buffer (sampled lanes), and on raw bf16
static float sig_deq(const std::vector<fp8_t>& v, const std::vector<float>& sc, std::size_t vbase, std::size_t sbase) {
  float s = 0;
  for (int h = 0; h < H; h += 257) s += deq(v[vbase + h], sc[sbase + h / GROUP]);
  return s;
}
static float sig_bf(const std::vector<bf16>& h, std::size_t tb) {
  float s = 0;
  for (int hi = 0; hi < H; hi += 257) s += static_cast<float>(h[tb + hi]);
  return s;
}

TEST_CASE("V1 MoE dispatch+pack+FP8quant correctness", "[moe_dispatch_quant]") {
  const auto [rank, world_size] = iris::mpi::initialize();
  REQUIRE(world_size == 8);

  int dev = 0; hip_try(hipGetDevice(&dev));
  hipDeviceProp_t props; hip_try(hipGetDeviceProperties(&props, dev));
  const int grid = props.multiProcessorCount;

  const std::size_t heap = 1024ULL * 1024 * 1024;
  iris::iris iris_obj(heap, rank, world_size);
  auto view = iris_obj.get_device_view();

  auto* hidden = iris_obj.allocate<bf16>(static_cast<std::size_t>(T_LOCAL) * H);
  auto* packed_fp8 = iris_obj.allocate<fp8_t>(static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  auto* packed_sc = iris_obj.allocate<float>(static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * N_GROUPS);
  auto* assign_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  auto* send_counts = iris_obj.allocate<int>(static_cast<std::size_t>(world_size) * EXPERTS_PER_RANK);
  auto* route_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  auto* topk_ids = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  REQUIRE(hidden); REQUIRE(packed_fp8); REQUIRE(packed_sc); REQUIRE(route_slot); REQUIRE(topk_ids);

  std::vector<bf16> h_hidden(static_cast<std::size_t>(T_LOCAL) * H);
  std::vector<int> h_topk(static_cast<std::size_t>(T_LOCAL) * TOPK);
  std::mt19937 gen(1234 + rank);
  std::uniform_int_distribution<int> edist(0, N_ROUTED_EXPERTS - 1);
  for (int t = 0; t < T_LOCAL; t++) {
    for (int hi = 0; hi < H; hi++)
      h_hidden[static_cast<std::size_t>(t) * H + hi] =
          static_cast<bf16>(static_cast<float>((rank * 131 + t * 7 + hi) % 17) * 0.5f);
    int picked[TOPK];
    for (int k = 0; k < TOPK; k++) {
      int e; bool dup;
      do { e = edist(gen); dup = false; for (int j = 0; j < k; j++) if (picked[j] == e) dup = true; } while (dup);
      picked[k] = e; h_topk[static_cast<std::size_t>(t) * TOPK + k] = e;
    }
  }
  hip_try(hipMemcpy(hidden, h_hidden.data(), h_hidden.size() * sizeof(bf16), hipMemcpyHostToDevice));
  hip_try(hipMemcpy(topk_ids, h_topk.data(), h_topk.size() * sizeof(int), hipMemcpyHostToDevice));

  // Gather all ranks' topk + per-token signature for the multiset check.
  std::vector<int> all_topk(static_cast<std::size_t>(world_size) * T_LOCAL * TOPK);
  std::vector<float> all_sig(static_cast<std::size_t>(world_size) * T_LOCAL);
  std::vector<float> my_sig(T_LOCAL);
  for (int t = 0; t < T_LOCAL; t++) my_sig[t] = sig_bf(h_hidden, static_cast<std::size_t>(t) * H);
  MPI_Allgather(h_topk.data(), T_LOCAL * TOPK, MPI_INT, all_topk.data(), T_LOCAL * TOPK, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(my_sig.data(), T_LOCAL, MPI_FLOAT, all_sig.data(), T_LOCAL, MPI_FLOAT, MPI_COMM_WORLD);

  const int n_assign = T_LOCAL * TOPK, block = 256;
  hip_try(hipMemset(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int)));
  hip_try(hipMemset(packed_fp8, 0, static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H * sizeof(fp8_t)));
  hip_try(hipMemset(packed_sc, 0, static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * N_GROUPS * sizeof(float)));
  hip_try(hipMemset(route_slot, 0xff, n_assign * sizeof(int)));
  iris_obj.barrier();
  k_count<<<(n_assign + block - 1) / block, block>>>(topk_ids, assign_slot, send_counts);
  hip_try(hipDeviceSynchronize());
  k_quant<<<grid, 256>>>(hidden, topk_ids, assign_slot, packed_fp8, packed_sc, route_slot, view);
  iris_obj.barrier();

  std::vector<fp8_t> hv(static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  std::vector<float> hs(static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * N_GROUPS);
  hip_try(hipMemcpy(hv.data(), packed_fp8, hv.size() * sizeof(fp8_t), hipMemcpyDeviceToHost));
  hip_try(hipMemcpy(hs.data(), packed_sc, hs.size() * sizeof(float), hipMemcpyDeviceToHost));

  // (a) per-(src,expert) multiset of dequantized token signatures vs bf16 reference.
  int errors = 0, checked = 0;
  for (int src = 0; src < world_size; src++)
    for (int local_e = 0; local_e < EXPERTS_PER_RANK; local_e++) {
      std::vector<float> expected, got;
      for (int t = 0; t < T_LOCAL; t++)
        for (int k = 0; k < TOPK; k++) {
          const int e = all_topk[((static_cast<std::size_t>(src) * T_LOCAL) + t) * TOPK + k];
          if (e / EXPERTS_PER_RANK == rank && e % EXPERTS_PER_RANK == local_e) {
            expected.push_back(all_sig[static_cast<std::size_t>(src) * T_LOCAL + t]);
            checked++;
          }
        }
      for (int slot = 0; slot < (int)expected.size(); slot++) {
        const std::size_t b = ((static_cast<std::size_t>(local_e) * world_size + src) * PER_SRC_CAPACITY + slot);
        got.push_back(sig_deq(hv, hs, b * H, b * N_GROUPS));
      }
      std::sort(expected.begin(), expected.end());
      std::sort(got.begin(), got.end());
      if (expected.size() != got.size()) { errors++; continue; }
      for (size_t i = 0; i < expected.size(); i++)
        // signature is a sum of ~28 sampled dequantized values; allow FP8 group error
        if (std::abs(expected[i] - got[i]) > 0.06f * (std::abs(expected[i]) + 1.0f)) { errors++; break; }
    }

  // (b) element-wise max relative error on rank's own self-routed tokens (local_e where
  //     we can pair fp8 slot back to a known source token via assign_slot of this rank).
  // We recompute this rank's own assignment slots to locate its tokens in the buffer.
  std::vector<int> h_assign(n_assign);
  hip_try(hipMemcpy(h_assign.data(), assign_slot, n_assign * sizeof(int), hipMemcpyDeviceToHost));
  double max_rel = 0.0; double sum_rel = 0.0; long cnt_rel = 0;
  for (int t = 0; t < T_LOCAL; t++)
    for (int k = 0; k < TOPK; k++) {
      const int e = h_topk[static_cast<std::size_t>(t) * TOPK + k];
      if (e / EXPERTS_PER_RANK != rank) continue;  // only buckets we own + we are src
      const int local_e = e % EXPERTS_PER_RANK;
      const int slot = h_assign[t * TOPK + k];
      if (slot >= PER_SRC_CAPACITY) continue;
      const std::size_t b = ((static_cast<std::size_t>(local_e) * world_size + rank) * PER_SRC_CAPACITY + slot);
      for (int hi = 0; hi < H; hi += 64) {
        const float ref = static_cast<float>(h_hidden[static_cast<std::size_t>(t) * H + hi]);
        const float dq = deq(hv[b * H + hi], hs[b * N_GROUPS + hi / GROUP]);
        const float denom = std::abs(ref) + 1e-3f;
        const float rel = std::abs(ref - dq) / denom;
        max_rel = std::max(max_rel, (double)rel); sum_rel += rel; cnt_rel++;
      }
    }
  double gmax = 0; MPI_Allreduce(&max_rel, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double gsum = 0; MPI_Allreduce(&sum_rel, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long gcnt = 0; MPI_Allreduce(&cnt_rel, &gcnt, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  // route_slot validity
  std::vector<int> rslot(n_assign);
  hip_try(hipMemcpy(rslot.data(), route_slot, n_assign * sizeof(int), hipMemcpyDeviceToHost));
  int route_bad = 0;
  for (int a = 0; a < n_assign; a++) if (rslot[a] < 0 || rslot[a] >= PER_SRC_CAPACITY) route_bad++;

  if (rank == 0) {
    std::printf("[V1] checked %d assignments, %d multiset errors, %d bad route_slot\n", checked, errors, route_bad);
    std::printf("[V1] FP8 e4m3 dequant vs bf16: max_rel_err=%.4f  mean_rel_err=%.5f  (n=%ld samples)\n",
                gmax, gcnt ? gsum / gcnt : 0.0, gcnt);
  }
  REQUIRE(errors == 0);
  REQUIRE(route_bad == 0);
  // FP8 e4m3 has 3 mantissa bits -> per-element relative step ~ 2^-3 = 0.125; with
  // per-128 max scaling the worst-case rounding rel err <= ~0.125. Assert a safe bound.
  REQUIRE(gmax < 0.13);

  iris_obj.barrier();
  iris::mpi::finalize();
}
