// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Producer kernel from original code
template <typename T>
__global__ void producer_kernel(T *source_buffer,
                                T *target_buffer,
                                int *flag,
                                int buffer_size,
                                int producer_rank,
                                int consumer_rank,
                                int BLOCK_SIZE,
                                iris::iris_device_view iris_view) {
  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_start = pid * BLOCK_SIZE;
  const int idx = block_start + tid;

  if (idx < buffer_size) {
    T value = iris_view.load(source_buffer + idx, producer_rank);
    iris_view.store(target_buffer + idx, value, consumer_rank);
  }

  if (tid == 0) {
    int desired = 1;
    iris_view.store(flag + pid, desired, consumer_rank);
  }
}

TEST_CASE("Test 4: Producer kernel", "[producer]") {
    const auto [rank, world_size] = iris::mpi::initialize();

    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30;
    iris::iris iris_obj(heap_size, rank, world_size, true);

    auto device_view = iris_obj.get_device_view();

    const int producer_rank = 0;
    const int consumer_rank = 1;
    const int buffer_size = 512;
    const int block_size = 256;
    const int num_blocks = (buffer_size + block_size - 1) / block_size;

    // Allocate buffers
    float *source_buffer = iris_obj.allocate<float>(buffer_size);
    float *target_buffer = iris_obj.allocate<float>(buffer_size);
    int *flags = iris_obj.allocate<int>(num_blocks);

    // Initialize source buffer on producer
    if (rank == producer_rank) {
        std::vector<float> source_data(buffer_size);
        for (int i = 0; i < buffer_size; i++) {
            source_data[i] = (float)i; // Simple sequence
        }
        hip_try(hipMemcpy(source_buffer, source_data.data(),
                        buffer_size * sizeof(float), hipMemcpyHostToDevice));
    }

    // Initialize target and flags to 0
    std::vector<float> zeros_float(buffer_size, 0.0f);
    std::vector<int> zeros_int(num_blocks, 0);
    hip_try(hipMemcpy(target_buffer, zeros_float.data(),
                    buffer_size * sizeof(float), hipMemcpyHostToDevice));
    hip_try(hipMemcpy(flags, zeros_int.data(),
                    num_blocks * sizeof(int), hipMemcpyHostToDevice));

    iris_obj.barrier();

    if (rank == producer_rank) {
        std::cout << "Rank " << rank << ": Launching producer kernel with "
                 << num_blocks << " blocks" << std::endl;

        producer_kernel<<<num_blocks, block_size>>>(
            source_buffer, target_buffer, flags, buffer_size,
            producer_rank, consumer_rank, block_size, device_view);

        hip_try(hipDeviceSynchronize());
        std::cout << "Rank " << rank << ": Producer kernel complete" << std::endl;
    }

    iris_obj.barrier();

    if (rank == consumer_rank) {
        // Check flags
        std::vector<int> flag_results(num_blocks);
        hip_try(hipMemcpy(flag_results.data(), flags,
                        num_blocks * sizeof(int), hipMemcpyDeviceToHost));

        std::cout << "Rank " << rank << ": Checking flags..." << std::endl;
        for (int i = 0; i < num_blocks; i++) {
            std::cout << "  Flag[" << i << "] = " << flag_results[i] << std::endl;
            REQUIRE(flag_results[i] == 1);
        }

        // Check data
        std::vector<float> target_results(buffer_size);
        hip_try(hipMemcpy(target_results.data(), target_buffer,
                        buffer_size * sizeof(float), hipMemcpyDeviceToHost));

        std::cout << "Rank " << rank << ": Checking data..." << std::endl;
        bool data_correct = true;
        for (int i = 0; i < buffer_size; i++) {
            if (target_results[i] != (float)i) {
                std::cout << "  Mismatch at index " << i
                         << ": expected " << (float)i
                         << ", got " << target_results[i] << std::endl;
                data_correct = false;
                if (i > 10) break; // Only show first few errors
            }
        }
        REQUIRE(data_correct);
    }

    iris_obj.barrier();

    iris::mpi::finalize();
}

