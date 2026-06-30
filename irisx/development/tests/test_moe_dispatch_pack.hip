// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Correctness test for the V0 MoE dispatch+pack kernels.
//
// Each rank generates synthetic hidden activations + random top-k routing.
// After dispatch, each rank checks that its expert-major packed buffer contains
// exactly the token vectors that were routed (from every source rank) to the
// experts it owns -- compared against a CPU reference rebuilt from the same
// topk_ids gathered over MPI.
//
// V0b (precomputed-offset) and V0a (atomic slot-claim) both verified as a multiset
// of per-(src,expert) token signatures: slot order within a bucket is
// nondeterministic across blocks, but the SET of tokens landing in each expert
// bucket is deterministic. route_slot output is also checked for validity.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"

#include <hip/hip_bf16.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

using bf16 = __hip_bfloat16;

struct __attribute__((aligned(16))) bf16x8 { bf16 v[8]; };

static constexpr int H = 7168;
static constexpr int TOPK = 8;
static constexpr int N_ROUTED_EXPERTS = 256;
static constexpr int EXPERTS_PER_RANK = 32;
static constexpr int T_LOCAL = 64;          // small for a fast test
static constexpr int HVEC = H / 8;
static constexpr int EXPERT_CAPACITY = 256;
static constexpr int PER_SRC_CAPACITY = 64;

// --- kernels (mirror moe_dispatch_pack.hip; vectorized 16B stores) ---
__global__ void k_v0a(const bf16* hidden, const int* topk_ids, bf16* packed,
                      int* counts, int* route_slot, iris::iris_device_view view) {
  const int n_assign = T_LOCAL * TOPK;
  for (int a = blockIdx.x; a < n_assign; a += gridDim.x) {
    const int t = a / TOPK, k = a % TOPK;
    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK, local_e = expert % EXPERTS_PER_RANK;
    __shared__ int slot_s;
    if (threadIdx.x == 0)
      slot_s = view.fetch_add<int, iris::memory_scope_system>(
          &counts[local_e], 1, dst_rank, iris::memory_order_relaxed);
    __syncthreads();
    const int slot = slot_s;
    if (slot >= EXPERT_CAPACITY) continue;
    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;
    const bf16x8* src = reinterpret_cast<const bf16x8*>(hidden + t * H);
    bf16x8* dst = reinterpret_cast<bf16x8*>(
        packed + (static_cast<std::size_t>(local_e) * EXPERT_CAPACITY + slot) * H);
    for (int h = threadIdx.x; h < HVEC; h += blockDim.x) view.store(dst + h, src[h], dst_rank);
  }
}

__global__ void k_count_v0b(const int* topk_ids, int* assign_slot, int* send_counts) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= T_LOCAL * TOPK) return;
  const int t = tid / TOPK, k = tid % TOPK;
  const int expert = topk_ids[t * TOPK + k];
  const int dst_rank = expert / EXPERTS_PER_RANK, local_e = expert % EXPERTS_PER_RANK;
  assign_slot[tid] = atomicAdd(&send_counts[dst_rank * EXPERTS_PER_RANK + local_e], 1);
}

__global__ void k_v0b(const bf16* hidden, const int* topk_ids, const int* assign_slot,
                      bf16* packed, int* route_slot, iris::iris_device_view view) {
  const int my_rank = view.cur_rank(), world = view.world_size();
  const int n_assign = T_LOCAL * TOPK;
  for (int a = blockIdx.x; a < n_assign; a += gridDim.x) {
    const int t = a / TOPK, k = a % TOPK;
    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK, local_e = expert % EXPERTS_PER_RANK;
    const int slot = assign_slot[a];
    if (slot >= PER_SRC_CAPACITY) continue;
    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;
    const std::size_t idx = ((static_cast<std::size_t>(local_e) * world + my_rank) * PER_SRC_CAPACITY + slot);
    const bf16x8* src = reinterpret_cast<const bf16x8*>(hidden + t * H);
    bf16x8* dst = reinterpret_cast<bf16x8*>(packed + idx * H);
    for (int h = threadIdx.x; h < HVEC; h += blockDim.x) view.store(dst + h, src[h], dst_rank);
  }
}

// Token signature: cheap fingerprint of an H-vector (sum of sampled lanes).
static float sig_host(const std::vector<bf16>& h, std::size_t tok_base) {
  float s = 0;
  for (int h_i = 0; h_i < H; h_i += 257) s += static_cast<float>(h[tok_base + h_i]);
  return s;
}
static float sig_buf(const std::vector<bf16>& buf, std::size_t vec_base) {
  float s = 0;
  for (int h_i = 0; h_i < H; h_i += 257) s += static_cast<float>(buf[vec_base + h_i]);
  return s;
}

TEST_CASE("V0 MoE dispatch+pack correctness", "[moe_dispatch]") {
  const auto [rank, world_size] = iris::mpi::initialize();
  REQUIRE(world_size == 8);

  int dev = 0; hip_try(hipGetDevice(&dev));
  hipDeviceProp_t props; hip_try(hipGetDeviceProperties(&props, dev));
  const int grid = props.multiProcessorCount;  // device-derived grid (no hardcode)

  const std::size_t heap = 1024ULL * 1024 * 1024;
  iris::iris iris_obj(heap, rank, world_size);
  auto view = iris_obj.get_device_view();

  auto* hidden = iris_obj.allocate<bf16>(static_cast<std::size_t>(T_LOCAL) * H);
  auto* packed_a = iris_obj.allocate<bf16>(static_cast<std::size_t>(EXPERTS_PER_RANK) * EXPERT_CAPACITY * H);
  auto* counts = iris_obj.allocate<int>(EXPERTS_PER_RANK);
  auto* packed_b = iris_obj.allocate<bf16>(static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  auto* assign_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  auto* send_counts = iris_obj.allocate<int>(static_cast<std::size_t>(world_size) * EXPERTS_PER_RANK);
  auto* route_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  auto* topk_ids = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  REQUIRE(hidden); REQUIRE(packed_a); REQUIRE(packed_b); REQUIRE(route_slot); REQUIRE(topk_ids);

  // synthetic data (matches benchmark generation scheme)
  std::vector<bf16> h_hidden(static_cast<std::size_t>(T_LOCAL) * H);
  std::vector<int> h_topk(static_cast<std::size_t>(T_LOCAL) * TOPK);
  std::mt19937 gen(1234 + rank);
  std::uniform_int_distribution<int> edist(0, N_ROUTED_EXPERTS - 1);
  for (int t = 0; t < T_LOCAL; t++) {
    for (int h_i = 0; h_i < H; h_i++)
      h_hidden[static_cast<std::size_t>(t) * H + h_i] =
          static_cast<bf16>(static_cast<float>((rank * 131 + t * 7 + h_i) % 17) * 0.5f);
    int picked[TOPK];
    for (int k = 0; k < TOPK; k++) {
      int e; bool dup;
      do { e = edist(gen); dup = false; for (int j = 0; j < k; j++) if (picked[j] == e) dup = true; } while (dup);
      picked[k] = e; h_topk[static_cast<std::size_t>(t) * TOPK + k] = e;
    }
  }
  hip_try(hipMemcpy(hidden, h_hidden.data(), h_hidden.size() * sizeof(bf16), hipMemcpyHostToDevice));
  hip_try(hipMemcpy(topk_ids, h_topk.data(), h_topk.size() * sizeof(int), hipMemcpyHostToDevice));

  // Gather every rank's topk + per-token signature so each rank can build a CPU reference.
  std::vector<int> all_topk(static_cast<std::size_t>(world_size) * T_LOCAL * TOPK);
  std::vector<float> all_sig(static_cast<std::size_t>(world_size) * T_LOCAL);
  std::vector<float> my_sig(T_LOCAL);
  for (int t = 0; t < T_LOCAL; t++) my_sig[t] = sig_host(h_hidden, static_cast<std::size_t>(t) * H);
  MPI_Allgather(h_topk.data(), T_LOCAL * TOPK, MPI_INT, all_topk.data(), T_LOCAL * TOPK, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(my_sig.data(), T_LOCAL, MPI_FLOAT, all_sig.data(), T_LOCAL, MPI_FLOAT, MPI_COMM_WORLD);

  const int n_assign = T_LOCAL * TOPK;
  const int block = 256;

  // ===== V0b =====
  hip_try(hipMemset(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int)));
  hip_try(hipMemset(packed_b, 0, static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H * sizeof(bf16)));
  hip_try(hipMemset(route_slot, 0xff, n_assign * sizeof(int)));
  iris_obj.barrier();
  k_count_v0b<<<(n_assign + block - 1) / block, block>>>(topk_ids, assign_slot, send_counts);
  hip_try(hipDeviceSynchronize());
  k_v0b<<<grid, block>>>(hidden, topk_ids, assign_slot, packed_b, route_slot, view);
  iris_obj.barrier();

  std::vector<bf16> hb(static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  hip_try(hipMemcpy(hb.data(), packed_b, hb.size() * sizeof(bf16), hipMemcpyDeviceToHost));

  int errors_b = 0, checked_b = 0;
  for (int src = 0; src < world_size; src++) {
    for (int local_e = 0; local_e < EXPERTS_PER_RANK; local_e++) {
      std::vector<float> expected, got;
      for (int t = 0; t < T_LOCAL; t++)
        for (int k = 0; k < TOPK; k++) {
          const int expert = all_topk[((static_cast<std::size_t>(src) * T_LOCAL) + t) * TOPK + k];
          if (expert / EXPERTS_PER_RANK == rank && expert % EXPERTS_PER_RANK == local_e) {
            expected.push_back(all_sig[static_cast<std::size_t>(src) * T_LOCAL + t]);
            checked_b++;
          }
        }
      for (int slot = 0; slot < (int)expected.size(); slot++) {
        const std::size_t base = ((static_cast<std::size_t>(local_e) * world_size + src) * PER_SRC_CAPACITY + slot) * H;
        got.push_back(sig_buf(hb, base));
      }
      std::sort(expected.begin(), expected.end());
      std::sort(got.begin(), got.end());
      if (expected.size() != got.size()) { errors_b++; continue; }
      for (size_t i = 0; i < expected.size(); i++)
        if (std::abs(expected[i] - got[i]) > 1e-2f) { errors_b++; break; }
    }
  }
  // route_slot validity: every assignment must have been written a non-negative slot.
  std::vector<int> rslot(n_assign);
  hip_try(hipMemcpy(rslot.data(), route_slot, n_assign * sizeof(int), hipMemcpyDeviceToHost));
  int route_bad = 0;
  for (int a = 0; a < n_assign; a++) if (rslot[a] < 0 || rslot[a] >= PER_SRC_CAPACITY) route_bad++;
  if (rank == 0) std::printf("[V0b] rank0 checked %d assignments, %d bucket errors, %d bad route_slot\n",
                             checked_b, errors_b, route_bad);
  REQUIRE(errors_b == 0);
  REQUIRE(route_bad == 0);

  // ===== V0a =====
  hip_try(hipMemset(counts, 0, EXPERTS_PER_RANK * sizeof(int)));
  hip_try(hipMemset(packed_a, 0, static_cast<std::size_t>(EXPERTS_PER_RANK) * EXPERT_CAPACITY * H * sizeof(bf16)));
  iris_obj.barrier();
  k_v0a<<<grid, block>>>(hidden, topk_ids, packed_a, counts, route_slot, view);
  iris_obj.barrier();

  std::vector<bf16> ha(static_cast<std::size_t>(EXPERTS_PER_RANK) * EXPERT_CAPACITY * H);
  std::vector<int> hcounts(EXPERTS_PER_RANK);
  hip_try(hipMemcpy(ha.data(), packed_a, ha.size() * sizeof(bf16), hipMemcpyDeviceToHost));
  hip_try(hipMemcpy(hcounts.data(), counts, EXPERTS_PER_RANK * sizeof(int), hipMemcpyDeviceToHost));

  int errors_a = 0;
  for (int local_e = 0; local_e < EXPERTS_PER_RANK; local_e++) {
    std::vector<float> expected, got;
    for (int src = 0; src < world_size; src++)
      for (int t = 0; t < T_LOCAL; t++)
        for (int k = 0; k < TOPK; k++) {
          const int expert = all_topk[((static_cast<std::size_t>(src) * T_LOCAL) + t) * TOPK + k];
          if (expert == rank * EXPERTS_PER_RANK + local_e)
            expected.push_back(all_sig[static_cast<std::size_t>(src) * T_LOCAL + t]);
        }
    if ((int)expected.size() != hcounts[local_e]) { errors_a++; }
    for (int slot = 0; slot < hcounts[local_e] && slot < EXPERT_CAPACITY; slot++)
      got.push_back(sig_buf(ha, (static_cast<std::size_t>(local_e) * EXPERT_CAPACITY + slot) * H));
    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());
    if (expected.size() != got.size()) { errors_a++; continue; }
    for (size_t i = 0; i < expected.size(); i++)
      if (std::abs(expected[i] - got[i]) > 1e-2f) { errors_a++; break; }
  }
  if (rank == 0) std::printf("[V0a] rank0 %d expert-bucket errors\n", errors_a);
  REQUIRE(errors_a == 0);

  iris_obj.barrier();
  iris::mpi::finalize();
}
