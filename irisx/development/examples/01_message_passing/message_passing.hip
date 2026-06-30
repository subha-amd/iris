// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <cstring>

#include "iris/iris.hpp"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define hip_try(error)                                                         \
  if (error != hipSuccess) {                                                   \
    std::cerr << "Hip error: " << hipGetErrorString(error) << " at line "      \
              << __LINE__ << std::endl;                                        \
    exit(-1);                                                                  \
  }

// Step 2: Producer kernel with atomic flag signaling
template <typename T>
__global__ void producer_kernel(T *source_buffer, T *target_buffer, int *flag,
                                int buffer_size, int producer_rank,
                                int consumer_rank, int BLOCK_SIZE,
                                iris::iris_device_view iris_view) {
  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_start = pid * BLOCK_SIZE;
  const int idx = block_start + tid;

  if (idx < buffer_size) {
    T value = iris_view.load(source_buffer + idx, producer_rank);
    iris_view.store(target_buffer + idx, value, consumer_rank);
  }

  // Signal completion with release semantics (only thread 0 in block)
  if (tid == 0) {
    iris_view.atomic_store<int, iris::memory_scope_system>(
        flag + pid, 1, consumer_rank, iris::memory_order_release);
  }
}

// Step 3: Consumer kernel with atomic flag waiting
template <typename T>
__global__ void consumer_kernel(T *buffer, int *flag, int buffer_size,
                                int consumer_rank, int BLOCK_SIZE,
                                iris::iris_device_view iris_view) {
  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_start = pid * BLOCK_SIZE;
  const int idx = block_start + tid;

  // Spin-wait for producer with acquire semantics (only thread 0 in block)
  if (tid == 0) {
    int flag_value = 0;
    while (flag_value != 1) {
      flag_value = iris_view.atomic_load<int, iris::memory_scope_system>(
          flag + pid, consumer_rank, iris::memory_order_acquire);
    }
  }
  __syncthreads();

  // Process the data
  if (idx < buffer_size) {
    T value = iris_view.load(buffer + idx, consumer_rank);
    value = value * T(2);  // Example processing
    iris_view.store(buffer + idx, value, consumer_rank);
  }
}

struct Args {
  int buffer_size = 1024;
  int block_size = 512;
};

Args parse_args(int argc, char *argv[]) {
  Args args;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--buffer_size") == 0 && i + 1 < argc) {
      args.buffer_size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--block_size") == 0 && i + 1 < argc) {
      args.block_size = std::atoi(argv[++i]);
    }
  }
  return args;
}

template <typename T>
void run_example(int argc, char *argv[]) {
  auto args = parse_args(argc, argv);

  const auto [cur_rank, world_size] = iris::mpi::initialize();
  if (world_size != 2) {
    std::cerr << "This example requires exactly 2 ranks" << std::endl;
    iris::mpi::finalize();
    exit(1);
  }

  const int producer_rank = 0;
  const int consumer_rank = 1;

  const std::size_t heap_size = 1ULL << 30;
  iris::iris iris(heap_size, cur_rank, world_size, false);

  const int n_elements = args.buffer_size;
  const int num_blocks = (n_elements + args.block_size - 1) / args.block_size;

  // Allocate buffers
  T *source_buffer = iris.allocate<T>(n_elements);
  T *destination_buffer = iris.allocate<T>(n_elements);
  int *flags = iris.allocate<int>(num_blocks);

  // Initialize source buffer with sequential values using thrust
  if (cur_rank == producer_rank) {
    thrust::sequence(thrust::device, source_buffer, source_buffer + n_elements, T(0));
  }

  // Initialize destination buffer to zeros using thrust
  if (cur_rank == consumer_rank) {
    thrust::fill(thrust::device, destination_buffer, destination_buffer + n_elements, T(0));
    thrust::fill(thrust::device, flags, flags + num_blocks, 0);
  }

  auto device_view = iris.get_device_view();

  iris.barrier();

  // Launch producer kernel
  if (cur_rank == producer_rank) {
    std::cout << "Rank " << cur_rank << ": Launching producer kernel with "
              << num_blocks << " blocks, " << args.block_size << " threads, "
              << n_elements << " elements" << std::endl;
    producer_kernel<<<num_blocks, args.block_size>>>(
        source_buffer, destination_buffer, flags, n_elements, producer_rank,
        consumer_rank, args.block_size, device_view);
    hip_try(hipDeviceSynchronize());
    std::cout << "Rank " << cur_rank << ": Producer kernel completed" << std::endl;
  }

  // Launch consumer kernel
  if (cur_rank == consumer_rank) {
    std::cout << "Rank " << cur_rank << ": Launching consumer kernel with "
              << num_blocks << " blocks, " << args.block_size << " threads" << std::endl;
    consumer_kernel<<<num_blocks, args.block_size>>>(
        destination_buffer, flags, n_elements, consumer_rank,
        args.block_size, device_view);
    hip_try(hipDeviceSynchronize());
    std::cout << "Rank " << cur_rank << ": Consumer kernel completed" << std::endl;
  }

  iris.barrier();

  // Validate output on consumer (should be doubled now)
  if (cur_rank == consumer_rank) {
    std::vector<T> dest_host(n_elements);
    hip_try(hipMemcpy(dest_host.data(), destination_buffer,
                      n_elements * sizeof(T), hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < n_elements; i++) {
      T expected = T(i) * T(2);  // Should be doubled by consumer
      if (std::abs(dest_host[i] - expected) > T(0.01)) {
        std::cerr << "Rank " << cur_rank << ": Validation failed at index " << i
                  << ": expected " << expected << ", got " << dest_host[i] << std::endl;
        success = false;
        break;
      }
    }

    if (success) {
      std::cout << "Rank " << cur_rank << ": ✓ Validation passed! All " << n_elements
                << " elements correct (producer+consumer with atomics)" << std::endl;
    } else {
      std::cerr << "Rank " << cur_rank << ": ✗ Validation failed!" << std::endl;
    }
  }

  iris.barrier();
  iris::mpi::finalize();
}

int main(int argc, char *argv[]) {
  std::cout << "Running producer-consumer example with fp32 datatype" << std::endl;
  run_example<float>(argc, argv);
  return 0;
}
