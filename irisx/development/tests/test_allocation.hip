// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include "iris/iris.hpp"
#include "common.hpp"

TEST_CASE("Test 2: Memory allocation", "[allocation]") {
    const auto [rank, world_size] = iris::mpi::initialize();

    REQUIRE(world_size == 2);

    const std::size_t heap_size = 1ULL << 30; // 1 GB
    iris::iris iris_obj(heap_size, rank, world_size, true);

    // Test 1: Allocate small buffer
    const int size = 100;
    float *buffer = iris_obj.allocate<float>(size);

    REQUIRE(buffer != nullptr);

    // Try to write to it from host
    std::vector<float> host_data(size, 42.0f);
    hip_try(hipMemcpy(buffer, host_data.data(), size * sizeof(float),
                     hipMemcpyHostToDevice));

    // Read it back
    std::vector<float> result(size);
    hip_try(hipMemcpy(result.data(), buffer, size * sizeof(float),
                     hipMemcpyDeviceToHost));

    // Check values
    for (int i = 0; i < size; i++) {
        REQUIRE(result[i] == 42.0f);
    }

    // Test 2: Allocate multiple buffers
    const int size2 = 256;
    float *buffer1 = iris_obj.allocate<float>(size2);
    float *buffer2 = iris_obj.allocate<float>(size2);
    int *buffer3 = iris_obj.allocate<int>(size2);

    REQUIRE(buffer1 != nullptr);
    REQUIRE(buffer2 != nullptr);
    REQUIRE(buffer3 != nullptr);

    // Buffers should be different
    REQUIRE(buffer1 != buffer2);

    iris::mpi::finalize();
}

