// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

// Kernel that does atomic operations locally
template <typename T>
__global__ void local_atomic_kernel(T *buffer, int my_rank, iris::iris_device_view iris_view) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 32) {
        // Each thread atomically adds 1
        iris_view.fetch_add(buffer, static_cast<T>(1), my_rank);
    }
}

TEST_CASE("Test: Local atomic operations", "[local_atomic]") {
    const auto [rank, world_size] = iris::mpi::initialize();
    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30;
    iris::iris iris_obj(heap_size, rank, world_size, false);
    auto device_view = iris_obj.get_device_view();

    int *buffer = iris_obj.allocate<int>(1);
    REQUIRE(buffer != nullptr);

    int zero = 0;
    hip_try(hipMemcpy(buffer, &zero, sizeof(int), hipMemcpyHostToDevice));

    iris_obj.barrier();

    std::cout << "Rank " << rank << ": Launching local atomic kernel" << std::endl;
    local_atomic_kernel<<<1, 32>>>(buffer, rank, device_view);
    hip_try(hipDeviceSynchronize());

    int result;
    hip_try(hipMemcpy(&result, buffer, sizeof(int), hipMemcpyDeviceToHost));

    std::cout << "Rank " << rank << ": Result = " << result << " (expected 32)" << std::endl;
    REQUIRE(result == 32);

    iris_obj.barrier();
    iris::mpi::finalize();
}

