// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Kernel that stores values to a REMOTE rank's buffer
template <typename T>
__global__ void remote_store_kernel(T *buffer, int size, int target_rank,
                                   iris::iris_device_view iris_view) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        T value = static_cast<T>(idx * 100); // Store idx * 100
        // Store to REMOTE rank's memory
        iris_view.store(buffer + idx, value, target_rank);
    }
}

TEST_CASE("Test 7: Remote store operation", "[remote_store]") {
    const auto [rank, world_size] = iris::mpi::initialize();

    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30;
    iris::iris iris_obj(heap_size, rank, world_size, true);

    auto device_view = iris_obj.get_device_view();

    const int producer_rank = 0;
    const int consumer_rank = 1;

    const int buffer_size = 1024;
    const int block_size = 256;
    const int num_blocks = (buffer_size + block_size - 1) / block_size;

    // Allocate shared buffer
    float *buffer = iris_obj.allocate<float>(buffer_size);

    // Initialize to zeros on both ranks
    std::vector<float> zeros(buffer_size, 0.0f);
    hip_try(hipMemcpy(buffer, zeros.data(),
                    buffer_size * sizeof(float), hipMemcpyHostToDevice));

    iris_obj.barrier();

    if (rank == producer_rank) {
        std::cout << "Rank " << rank << ": Storing to Rank " << consumer_rank
                 << "'s buffer..." << std::endl;

        // Rank 0 stores to Rank 1's buffer
        remote_store_kernel<<<num_blocks, block_size>>>(
            buffer, buffer_size, consumer_rank, device_view);

        hip_try(hipDeviceSynchronize());
        std::cout << "Rank " << rank << ": Remote store complete" << std::endl;
    }

    iris_obj.barrier();

    if (rank == consumer_rank) {
        std::cout << "Rank " << rank << ": Checking received data..." << std::endl;

        // Rank 1 checks its buffer
        std::vector<float> results(buffer_size);
        hip_try(hipMemcpy(results.data(), buffer,
                        buffer_size * sizeof(float), hipMemcpyDeviceToHost));

        // Verify values
        bool all_correct = true;
        int errors = 0;
        for (int i = 0; i < buffer_size; i++) {
            float expected = (float)(i * 100);
            if (results[i] != expected) {
                if (errors < 10) {
                    std::cout << "Rank " << rank << ": ERROR at index " << i
                             << " - expected " << expected
                             << ", got " << results[i] << std::endl;
                }
                all_correct = false;
                errors++;
            }
        }

        if (all_correct) {
            std::cout << "Rank " << rank << ": All values received correctly!" << std::endl;
        } else {
            std::cout << "Rank " << rank << ": Found " << errors << " errors" << std::endl;
        }

        REQUIRE(all_correct);
    }

    iris_obj.barrier();

    iris::mpi::finalize();
}
