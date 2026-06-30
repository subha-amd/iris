// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Kernel that loads values from a REMOTE rank's buffer
template <typename T>
__global__ void remote_load_kernel(T *source_buffer, T *dest_buffer,
                                  int size, int source_rank,
                                  iris::iris_device_view iris_view) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Load from REMOTE rank's memory using iris_view.load()
        T value = iris_view.load(source_buffer + idx, source_rank);

        // Write to local destination buffer (regular store)
        dest_buffer[idx] = value;
    }
}

TEST_CASE("Test 8: Remote load operation", "[remote_load]") {
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

    // Allocate buffers
    float *source_buffer = iris_obj.allocate<float>(buffer_size);
    float *dest_buffer = iris_obj.allocate<float>(buffer_size);

    // Rank 0 initializes its buffer with data
    if (rank == producer_rank) {
        std::vector<float> source_data(buffer_size);
        for (int i = 0; i < buffer_size; i++) {
            source_data[i] = (float)(i * 50); // Source data
        }
        hip_try(hipMemcpy(source_buffer, source_data.data(),
                        buffer_size * sizeof(float), hipMemcpyHostToDevice));

        std::cout << "Rank " << rank << ": Initialized source buffer" << std::endl;
    }

    // Rank 1 initializes dest buffer to zeros
    if (rank == consumer_rank) {
        std::vector<float> zeros(buffer_size, 0.0f);
        hip_try(hipMemcpy(dest_buffer, zeros.data(),
                        buffer_size * sizeof(float), hipMemcpyHostToDevice));
    }

    iris_obj.barrier();

    if (rank == consumer_rank) {
        std::cout << "Rank " << rank << ": Loading from Rank " << producer_rank
                 << "'s buffer..." << std::endl;

        // Rank 1 loads from Rank 0's buffer
        remote_load_kernel<<<num_blocks, block_size>>>(
            source_buffer, dest_buffer, buffer_size, producer_rank, device_view);

        hip_try(hipDeviceSynchronize());
        std::cout << "Rank " << rank << ": Remote load complete" << std::endl;

        // Verify loaded data
        std::vector<float> results(buffer_size);
        hip_try(hipMemcpy(results.data(), dest_buffer,
                        buffer_size * sizeof(float), hipMemcpyDeviceToHost));

        bool all_correct = true;
        int errors = 0;
        for (int i = 0; i < buffer_size; i++) {
            float expected = (float)(i * 50);
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
            std::cout << "Rank " << rank << ": All values loaded correctly!" << std::endl;
        } else {
            std::cout << "Rank " << rank << ": Found " << errors << " errors" << std::endl;
        }

        REQUIRE(all_correct);
    }

    iris_obj.barrier();

    iris::mpi::finalize();
}
