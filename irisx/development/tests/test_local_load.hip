// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Kernel that loads values locally using iris_view.load() and writes to output
template <typename T>
__global__ void local_load_kernel(T *input_buffer, T *output_buffer,
                                 int size, int my_rank,
                                 iris::iris_device_view iris_view) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Load from local rank's memory using iris_view.load()
        T value = iris_view.load(input_buffer + idx, my_rank);

        // Write to output buffer (regular store, not using iris_view)
        output_buffer[idx] = value;
    }
}

TEST_CASE("Test 6: Local load operation", "[local_load]") {
    const auto [rank, world_size] = iris::mpi::initialize();

    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30;
    iris::iris iris_obj(heap_size, rank, world_size, true);

    auto device_view = iris_obj.get_device_view();

    const int buffer_size = 1024;
    const int block_size = 256;
    const int num_blocks = (buffer_size + block_size - 1) / block_size;

    // Allocate buffers
    float *input_buffer = iris_obj.allocate<float>(buffer_size);
    float *output_buffer = iris_obj.allocate<float>(buffer_size);

    // Initialize input buffer with known values from host
    std::vector<float> input_data(buffer_size);
    for (int i = 0; i < buffer_size; i++) {
        input_data[i] = (float)(i * 7 + rank * 1000); // Different values per rank
    }
    hip_try(hipMemcpy(input_buffer, input_data.data(),
                    buffer_size * sizeof(float), hipMemcpyHostToDevice));

    // Initialize output to zeros
    std::vector<float> zeros(buffer_size, 0.0f);
    hip_try(hipMemcpy(output_buffer, zeros.data(),
                    buffer_size * sizeof(float), hipMemcpyHostToDevice));

    iris_obj.barrier();

    std::cout << "Rank " << rank << ": Launching local load kernel" << std::endl;

    // Launch kernel to load values locally
    local_load_kernel<<<num_blocks, block_size>>>(
        input_buffer, output_buffer, buffer_size, rank, device_view);

    hip_try(hipDeviceSynchronize());

    std::cout << "Rank " << rank << ": Kernel complete, checking results..." << std::endl;

    // Copy output back and verify
    std::vector<float> results(buffer_size);
    hip_try(hipMemcpy(results.data(), output_buffer,
                    buffer_size * sizeof(float), hipMemcpyDeviceToHost));

    // Check that output matches input
    bool all_correct = true;
    int errors = 0;
    for (int i = 0; i < buffer_size; i++) {
        float expected = input_data[i];
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
        std::cout << "Rank " << rank << ": All values loaded correctly!" << std::endl;
    } else {
        std::cout << "Rank " << rank << ": Found " << errors << " errors" << std::endl;
    }

    REQUIRE(all_correct);

    iris_obj.barrier();

    iris::mpi::finalize();
}
