// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Kernel that stores values locally using iris_view.store()
template <typename T>
__global__ void local_store_kernel(T *buffer, int size, int my_rank,
                                   iris::iris_device_view iris_view) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        T value = static_cast<T>(idx * 10); // Store idx * 10
        // Store to local rank's memory
        iris_view.store(buffer + idx, value, my_rank);
    }
}

TEST_CASE("Test 5: Local store operation", "[local_store]") {
    const auto [rank, world_size] = iris::mpi::initialize();

    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30;
    iris::iris iris_obj(heap_size, rank, world_size, true);

    auto device_view = iris_obj.get_device_view();

    const int buffer_size = 1024;
    const int block_size = 256;
    const int num_blocks = (buffer_size + block_size - 1) / block_size;

    // Allocate buffer
    float *buffer = iris_obj.allocate<float>(buffer_size);

    // Initialize to zeros
    std::vector<float> zeros(buffer_size, 0.0f);
    hip_try(hipMemcpy(buffer, zeros.data(),
                    buffer_size * sizeof(float), hipMemcpyHostToDevice));

    iris_obj.barrier();

    std::cout << "Rank " << rank << ": Launching local store kernel" << std::endl;

    // Launch kernel to store values locally
    local_store_kernel<<<num_blocks, block_size>>>(
        buffer, buffer_size, rank, device_view);

    hip_try(hipDeviceSynchronize());

    std::cout << "Rank " << rank << ": Kernel complete, checking results..." << std::endl;

    // Copy back and verify
    std::vector<float> results(buffer_size);
    hip_try(hipMemcpy(results.data(), buffer,
                    buffer_size * sizeof(float), hipMemcpyDeviceToHost));

    // Check values
    bool all_correct = true;
    int errors = 0;
    for (int i = 0; i < buffer_size; i++) {
        float expected = (float)(i * 10);
        if (results[i] != expected) {
            if (errors < 10) { // Only print first 10 errors
                std::cout << "Rank " << rank << ": ERROR at index " << i
                         << " - expected " << expected
                         << ", got " << results[i] << std::endl;
            }
            all_correct = false;
            errors++;
        }
    }

    if (all_correct) {
        std::cout << "Rank " << rank << ": All values correct!" << std::endl;
    } else {
        std::cout << "Rank " << rank << ": Found " << errors << " errors" << std::endl;
    }

    REQUIRE(all_correct);

    iris_obj.barrier();

    iris::mpi::finalize();
}
