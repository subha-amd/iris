// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// V0 MoE dispatch + pack kernel for the IRISX symmetric-heap RMA library.
//
// Routes top-k token activations to expert-owning GPUs and writes them in
// expert-major layout, collapsing the production EpDispatch -> opus_moe_sorting(x2)
// chain into a single device-side kernel. Pure data movement (no GEMM/MFMA/HK).
//
// R1-0528 shapes: H=7168 (bf16), top-k=8, 256 routed experts => 32 experts/GPU
// at EP=8, world_size=8. Synthetic data.
//
// Two variants:
//   V0a -- remote-atomic slot claim (correctness baseline). One block owns one
//          (token,k) assignment; claims a slot via remote fetch_add then does a
//          vectorized (16B) block-strided remote store of the H-vector.
//   V0b -- precomputed-offset layout packed[local_expert][src_rank][slot][H], so
//          each source rank owns a deterministic slice => NO remote atomics
//          (PPLX-style). A local counting pre-pass computes per-(src,expert) send
//          counts; each rank writes into its own reserved sub-slice.
//
// Prior-art requirements applied (see PRIOR_ART.md):
//   * route_slot[token][topk] emitted as a first-class output (V3 combine needs it).
//   * grid size from device_props.multiProcessorCount (NOT hardcoded 304).
//   * bulk cross-rank completion signal at end of dispatch (lightweight, not a barrier).
//   * vectorized 16B non-temporal remote stores for the hidden copy.
//   * compile-time in-kernel timestamp profiling mode (PROFILE=1).

#include "iris/iris.hpp"
#include "utils.hpp"

#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using bf16 = __hip_bfloat16;

// 16B vector type for coalesced/non-temporal remote stores (8 bf16 = 16 bytes).
struct __attribute__((aligned(16))) bf16x8 {
  bf16 v[8];
};

// ----- R1-0528 problem shapes -----------------------------------------------
static constexpr int H = 7168;             // hidden dim (bf16 token vector)
static constexpr int TOPK = 8;             // expert assignments per token
static constexpr int N_ROUTED_EXPERTS = 256;
static constexpr int EXPERTS_PER_RANK = 32;  // 256 / 8
static constexpr int T_LOCAL = 256;          // tokens per rank (decode batch)
static constexpr int HVEC = H / 8;           // 896 vectorized (16B) elements

// Per-expert capacity (max tokens any single local expert may receive across all
// source ranks). Avg load = T_LOCAL*world*TOPK/N_EXPERTS = 64 at world=8; allow skew.
static constexpr int EXPERT_CAPACITY = 256;
// V0b: per-(expert, src_rank) capacity. A source sends <= a handful of its tokens
// to any one expert (distinct top-k); 64 is generous headroom.
static constexpr int PER_SRC_CAPACITY = 64;

// =====================================================================================
// V0a: remote-atomic slot claim. grid-strided over assignments so grid size is decoupled
// from T_LOCAL*TOPK and set from device_props.multiProcessorCount.
// =====================================================================================
template <int PROFILE = 0>
__global__ void moe_dispatch_pack_v0a(const bf16* __restrict__ hidden,   // [T_LOCAL, H]
                                      const int* __restrict__ topk_ids,   // [T_LOCAL, TOPK]
                                      bf16* packed,                       // [EPR, CAP, H]
                                      int* counts,                        // [EPR]
                                      int* route_slot,                    // [T_LOCAL, TOPK] out
                                      unsigned long long* prof,           // [2] out (claim, store cycles)
                                      iris::iris_device_view view) {
  const int n_assign = T_LOCAL * TOPK;
  for (int assignment = blockIdx.x; assignment < n_assign; assignment += gridDim.x) {
    const int t = assignment / TOPK;
    const int k = assignment % TOPK;

    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK;
    const int local_e = expert % EXPERTS_PER_RANK;

    unsigned long long c0 = 0;
    if (PROFILE && threadIdx.x == 0) c0 = clock64();

    __shared__ int slot_s;
    if (threadIdx.x == 0) {
      slot_s = view.fetch_add<int, iris::memory_scope_system>(
          &counts[local_e], 1, dst_rank, iris::memory_order_relaxed);
    }
    __syncthreads();
    const int slot = slot_s;

    if (PROFILE && threadIdx.x == 0) {
      atomicAdd(&prof[0], clock64() - c0);
    }
    if (slot >= EXPERT_CAPACITY) continue;  // overflow guard

    // record where this assignment landed (first-class output for combine)
    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;

    unsigned long long s0 = 0;
    if (PROFILE && threadIdx.x == 0) s0 = clock64();

    // vectorized 16B remote store of the H-vector
    const bf16x8* src = reinterpret_cast<const bf16x8*>(hidden + t * H);
    bf16x8* dst = reinterpret_cast<bf16x8*>(
        packed + (static_cast<std::size_t>(local_e) * EXPERT_CAPACITY + slot) * H);
    for (int h = threadIdx.x; h < HVEC; h += blockDim.x) {
      view.store(dst + h, src[h], dst_rank);
    }
    if (PROFILE && threadIdx.x == 0) {
      atomicAdd(&prof[1], clock64() - s0);
    }
  }
}

// =====================================================================================
// V0b pass 1 (local): per-assignment local slot index + per-(dst_rank,local_e) counts.
// =====================================================================================
__global__ void moe_dispatch_count_v0b(const int* __restrict__ topk_ids,  // [T_LOCAL,TOPK]
                                        int* __restrict__ assign_slot,     // [T_LOCAL*TOPK] out
                                        int* __restrict__ send_counts) {   // [world*EPR] out
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
// V0b pass 2 (remote): deterministic store into [local_e][my_rank][slot][H] on dst_rank.
// grid-strided over assignments.
// =====================================================================================
template <int PROFILE = 0>
__global__ void moe_dispatch_pack_v0b(const bf16* __restrict__ hidden,    // [T_LOCAL,H]
                                      const int* __restrict__ topk_ids,   // [T_LOCAL,TOPK]
                                      const int* __restrict__ assign_slot,// [T_LOCAL*TOPK]
                                      bf16* packed,  // [EPR][world][PER_SRC][H]
                                      int* route_slot,                    // [T_LOCAL,TOPK] out
                                      unsigned long long* prof,           // [2] out
                                      iris::iris_device_view view) {
  const int my_rank = view.cur_rank();
  const int world = view.world_size();
  const int n_assign = T_LOCAL * TOPK;
  for (int assignment = blockIdx.x; assignment < n_assign; assignment += gridDim.x) {
    const int t = assignment / TOPK;
    const int k = assignment % TOPK;

    const int expert = topk_ids[t * TOPK + k];
    const int dst_rank = expert / EXPERTS_PER_RANK;
    const int local_e = expert % EXPERTS_PER_RANK;
    const int slot = assign_slot[assignment];
    if (slot >= PER_SRC_CAPACITY) continue;

    if (threadIdx.x == 0) route_slot[t * TOPK + k] = slot;

    unsigned long long s0 = 0;
    if (PROFILE && threadIdx.x == 0) s0 = clock64();

    const std::size_t idx =
        ((static_cast<std::size_t>(local_e) * world + my_rank) * PER_SRC_CAPACITY + slot);
    const bf16x8* src = reinterpret_cast<const bf16x8*>(hidden + t * H);
    bf16x8* dst = reinterpret_cast<bf16x8*>(packed + idx * H);
    for (int h = threadIdx.x; h < HVEC; h += blockDim.x) {
      view.store(dst + h, src[h], dst_rank);
    }
    if (PROFILE && threadIdx.x == 0) atomicAdd(&prof[1], clock64() - s0);
  }
}

// =====================================================================================
// Bulk cross-rank completion signal. After a rank finishes its remote stores it
// release-bumps a per-source flag on EVERY destination rank, then each rank spins
// until it has seen all `world` source flags arrive in its own inbox. This guarantees
// "all peers finished writing into my memory" without a heavy host barrier.
// signals layout (per rank, on symmetric heap): [world] ints; src s writes index s.
// =====================================================================================
__global__ void completion_signal(int* signals, int epoch, iris::iris_device_view view) {
  const int my_rank = view.cur_rank();
  const int world = view.world_size();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // tell every destination rank that my writes (this epoch) are visible
    for (int r = 0; r < world; r++) {
      view.atomic_store<int, iris::memory_scope_system>(
          &signals[my_rank], epoch, r, iris::memory_order_release);
    }
    // wait until every source has signalled me
    for (int s = 0; s < world; s++) {
      while (view.atomic_load<int, iris::memory_scope_system>(
                 &signals[s], my_rank, iris::memory_order_acquire) < epoch) {
        // spin
      }
    }
  }
}

// ----- host helpers ---------------------------------------------------------
static double bytes_moved_per_rank() {
  const double bytes_per_vec = static_cast<double>(H) * sizeof(bf16);
  const double assignments = static_cast<double>(T_LOCAL) * TOPK;
  return assignments * bytes_per_vec;  // all assignments move their H-vector
}

int main(int argc, char** argv) {
  const auto [rank, world_size] = iris::mpi::initialize();
  const bool profile = (argc > 1 && std::atoi(argv[1]) == 1);

  int dev = 0; hip_try(hipGetDevice(&dev));
  hipDeviceProp_t props; hip_try(hipGetDeviceProperties(&props, dev));
  const int num_cu = props.multiProcessorCount;          // MI355X = 256 CUs
  const int grid = num_cu;                                // grid-strided kernels

  if (rank == 0) {
    std::printf("=== V0 MoE dispatch+pack (H=%d, topk=%d, experts=%d, T_local=%d, world=%d) ===\n",
                H, TOPK, N_ROUTED_EXPERTS, T_LOCAL, world_size);
    std::printf("device CUs (multiProcessorCount) = %d -> grid = %d blocks\n", num_cu, grid);
  }

  const std::size_t heap_size = 1024ULL * 1024 * 1024;  // 1 GB (node VRAM is busy)
  iris::iris iris_obj(heap_size, rank, world_size);
  auto view = iris_obj.get_device_view();

  // ---- allocate IDENTICALLY on every rank (symmetric heap) ----
  auto* hidden = iris_obj.allocate<bf16>(static_cast<std::size_t>(T_LOCAL) * H);
  auto* packed_a = iris_obj.allocate<bf16>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * EXPERT_CAPACITY * H);
  auto* counts = iris_obj.allocate<int>(EXPERTS_PER_RANK);
  auto* packed_b = iris_obj.allocate<bf16>(
      static_cast<std::size_t>(EXPERTS_PER_RANK) * world_size * PER_SRC_CAPACITY * H);
  auto* assign_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  auto* send_counts = iris_obj.allocate<int>(static_cast<std::size_t>(world_size) * EXPERTS_PER_RANK);
  auto* route_slot = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);
  auto* signals = iris_obj.allocate<int>(world_size);
  auto* prof = iris_obj.allocate<unsigned long long>(2);
  auto* topk_ids = iris_obj.allocate<int>(static_cast<std::size_t>(T_LOCAL) * TOPK);

  if (!hidden || !packed_a || !packed_b || !counts || !assign_slot || !send_counts ||
      !route_slot || !signals || !prof || !topk_ids) {
    std::printf("rank %d: allocation failed (heap too small / VRAM busy)\n", rank);
    iris::mpi::finalize();
    return 1;
  }

  // ---- init synthetic data on host, copy to device ----
  std::vector<bf16> h_hidden(static_cast<std::size_t>(T_LOCAL) * H);
  std::vector<int> h_topk(static_cast<std::size_t>(T_LOCAL) * TOPK);
  std::mt19937 gen(1234 + rank);
  std::uniform_int_distribution<int> edist(0, N_ROUTED_EXPERTS - 1);
  for (int t = 0; t < T_LOCAL; t++) {
    for (int h = 0; h < H; h++) {
      h_hidden[static_cast<std::size_t>(t) * H + h] =
          static_cast<bf16>(static_cast<float>((rank * 131 + t * 7 + h) % 17) * 0.5f);
    }
    int picked[TOPK];
    for (int k = 0; k < TOPK; k++) {
      int e; bool dup;
      do { e = edist(gen); dup = false; for (int j = 0; j < k; j++) if (picked[j] == e) dup = true; } while (dup);
      picked[k] = e;
      h_topk[static_cast<std::size_t>(t) * TOPK + k] = e;
    }
  }
  hip_try(hipMemcpy(hidden, h_hidden.data(), h_hidden.size() * sizeof(bf16), hipMemcpyHostToDevice));
  hip_try(hipMemcpy(topk_ids, h_topk.data(), h_topk.size() * sizeof(int), hipMemcpyHostToDevice));

  const int n_assign = T_LOCAL * TOPK;
  const int block = 256;
  const int iters = 20;

  // ============================ V0a ============================
  hip_try(hipMemset(counts, 0, EXPERTS_PER_RANK * sizeof(int)));
  hip_try(hipMemset(signals, 0, world_size * sizeof(int)));
  hip_try(hipMemset(prof, 0, 2 * sizeof(unsigned long long)));
  iris_obj.barrier();

  auto run_v0a = [&](int epoch) {
    hipMemsetAsync(counts, 0, EXPERTS_PER_RANK * sizeof(int));
    if (profile)
      moe_dispatch_pack_v0a<1><<<grid, block>>>(hidden, topk_ids, packed_a, counts, route_slot, prof, view);
    else
      moe_dispatch_pack_v0a<0><<<grid, block>>>(hidden, topk_ids, packed_a, counts, route_slot, prof, view);
    completion_signal<<<1, 1>>>(signals, epoch, view);
  };
  // warmup
  run_v0a(1);
  iris_obj.barrier();
  gpu_timer ta; ta.start();
  for (int i = 0; i < iters; i++) run_v0a(2 + i);
  iris_obj.barrier();
  ta.stop();
  float v0a_ms = ta.elapsed_ms() / iters;

  // ============================ V0b ============================
  hip_try(hipMemset(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int)));
  hip_try(hipMemset(signals, 0, world_size * sizeof(int)));
  iris_obj.barrier();

  auto run_v0b = [&](int epoch) {
    hipMemsetAsync(send_counts, 0, world_size * EXPERTS_PER_RANK * sizeof(int));
    moe_dispatch_count_v0b<<<(n_assign + block - 1) / block, block>>>(topk_ids, assign_slot, send_counts);
    if (profile)
      moe_dispatch_pack_v0b<1><<<grid, block>>>(hidden, topk_ids, assign_slot, packed_b, route_slot, prof, view);
    else
      moe_dispatch_pack_v0b<0><<<grid, block>>>(hidden, topk_ids, assign_slot, packed_b, route_slot, prof, view);
    completion_signal<<<1, 1>>>(signals, epoch, view);
  };
  run_v0b(1);
  iris_obj.barrier();
  gpu_timer tb; tb.start();
  for (int i = 0; i < iters; i++) run_v0b(2 + i);
  iris_obj.barrier();
  tb.stop();
  float v0b_ms = tb.elapsed_ms() / iters;

  // ---- report (rank 0) ----
  const double bytes = bytes_moved_per_rank();
  const double v0a_gbs = bytes / (v0a_ms / 1e3) / 1e9;
  const double v0b_gbs = bytes / (v0b_ms / 1e3) / 1e9;
  if (rank == 0) {
    std::printf("bytes moved / rank / iter: %.2f MB (%d assignments x %d bf16)\n",
                bytes / 1e6, n_assign, H);
    std::printf("V0a (atomic slot-claim) : %8.4f ms/iter   %8.2f GB/s   %.1f us/instance\n",
                v0a_ms, v0a_gbs, v0a_ms * 1e3);
    std::printf("V0b (precomputed-offset): %8.4f ms/iter   %8.2f GB/s   %.1f us/instance\n",
                v0b_ms, v0b_gbs, v0b_ms * 1e3);
    std::printf("Reference: production EpDispatch+sorting x2 ~= 40 us/instance\n");
    if (profile) {
      unsigned long long h_prof[2];
      hip_try(hipMemcpy(h_prof, prof, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
      std::printf("[profile] accumulated cycles: slot-claim=%llu, remote-store=%llu\n",
                  h_prof[0], h_prof[1]);
    }
  }

  iris_obj.barrier();
  iris::mpi::finalize();
  return 0;
}
