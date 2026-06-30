// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Kernel that does atomic operations remotely
template <typename T>
__global__ void remote_atomic_kernel(T *buffer, int target_rank, iris::iris_device_view iris_view) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 32) {
        // Each thread atomically adds 1 to remote rank's buffer
        iris_view.fetch_add(buffer, static_cast<T>(1), target_rank);
    }
}

TEST_CASE("Test: Remote atomic operations", "[remote_atomic]") {
    const auto [rank, world_size] = iris::mpi::initialize();
    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30;
    iris::iris iris_obj(heap_size, rank, world_size, false);
    auto device_view = iris_obj.get_device_view();

    const int producer_rank = 0;
    const int consumer_rank = 1;

    int *buffer = iris_obj.allocate<int>(1);
    REQUIRE(buffer != nullptr);

    int zero = 0;
    hip_try(hipMemcpy(buffer, &zero, sizeof(int), hipMemcpyHostToDevice));

    iris_obj.barrier();

    if (rank == producer_rank) {
        std::cout << "Rank " << rank << ": Launching remote atomic to rank " << consumer_rank << std::endl;
        remote_atomic_kernel<<<1, 32>>>(buffer, consumer_rank, device_view);
        hip_try(hipDeviceSynchronize());
    }

    iris_obj.barrier();

    if (rank == consumer_rank) {
        int result;
        hip_try(hipMemcpy(&result, buffer, sizeof(int), hipMemcpyDeviceToHost));
        std::cout << "Rank " << rank << ": Result = " << result << " (expected 32)" << std::endl;
        REQUIRE(result == 32);
    }

    iris_obj.barrier();
    iris::mpi::finalize();
}

