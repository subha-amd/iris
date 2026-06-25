# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Add CPM for dependency management
include(../cmake/CPM.cmake)

# Add Catch2
CPMAddPackage("gh:catchorg/Catch2@3.4.0")

# Function to add an Iris test
function(add_iris_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.hip)
    target_link_libraries(${TEST_NAME} PRIVATE iris::iris Catch2::Catch2WithMain)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# Add all tests in logical order
# 1. Allocation
# 2. Loads (local, remote)
# 3. Stores (local, remote)
# 4. Atomics (local, remote)
# 5. Producer/consumer
set(TESTS
    test_allocation
    test_local_load
    test_remote_load
    test_local_store
    test_remote_store
    test_local_atomic
    test_remote_atomic
    test_producer
    test_moe_dispatch_pack
)

foreach(test ${TESTS})
    add_iris_test(${test})
endforeach()

# Helper to run all tests
add_custom_target(run_all_tests
    COMMAND echo "=== Running All Iris Tests ==="
    COMMAND ${CMAKE_COMMAND} -E env sh -c "for test in ${TESTS}\; do echo \"--- $$test ---\"\; mpirun -np 2 ./$$test || exit 1\; done"
    DEPENDS ${TESTS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

