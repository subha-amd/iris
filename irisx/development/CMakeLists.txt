# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

project(iris LANGUAGES CXX HIP C)

cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set GPU architecture - optional, can be overridden by user
# Example: cmake -DIRIS_HIP_ARCHITECTURES=gfx90a ..
if(NOT DEFINED IRIS_HIP_ARCHITECTURES)
  set(IRIS_HIP_ARCHITECTURES "gfx942" CACHE STRING "Target GPU architecture(s)")
endif()
set(CMAKE_HIP_ARCHITECTURES ${IRIS_HIP_ARCHITECTURES})

set(IRIS_BUILD_EXAMPLES ON)
set(IRIS_BUILD_BENCHMARKS ON)
set(IRIS_BUILD_TESTS ON)

# Add CPM for dependency management
include(cmake/CPM.cmake)

# Add spdlog for logging
CPMAddPackage("gh:gabime/spdlog#v1.16.0")

# Find MPI
find_package(MPI REQUIRED)

# Create header-only library target
add_library(iris INTERFACE)
target_include_directories(iris INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(iris INTERFACE MPI::MPI_CXX spdlog::spdlog)

add_library(iris::iris ALIAS iris)

if (IRIS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if (IRIS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if (IRIS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
