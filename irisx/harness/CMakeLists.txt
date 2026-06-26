# irisx/harness/CMakeLists.txt
# ------------------------------------------------------------------------------------------------
# Builds the harness pybind module `harness_kernel` (B0 local_gemm + B1 dispatch_pack_quant_once).
# Mirrors the v3/v4 candidate build pattern: place this dir under a HipKittens distributed-kernels
# checkout (it auto-discovers */kernel.cpp; here the source is harness_kernels.cpp, so add it
# explicitly).  Depends on HipKittens tile headers + IRIS, built inside the ATOM container (r1_c4).
#
# B3/B4/B5 reuse the EXISTING per-candidate tk_kernel modules (v3_fused_kernel / v4_astationary_kernel)
# — this file only builds the NEW B0/B1 op.  See BENCHMARK_METHODOLOGY.md for how the runner loads
# each module.
#
# Build (node, compile-only ok for subagents; GPU run main-agent-only):
#   docker exec r1_c4 bash -lc '
#     cd <HK_ROOT>/distributed-kernels
#     cmake -B build_01 -DDK_BUILD=harness -DIRIS_HIP_ARCHITECTURES=gfx950
#     flock /tmp/mi355x_compile.lock -c "cmake --build build_01 -j8 --target harness_kernel"'
# ------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)

# This assumes the surrounding distributed-kernels CMake provides hk_add_python_kernel() (the
# same helper v3/v4 use) OR pybind11_add_module + HK/IRIS include dirs.  Adjust the helper name to
# match the host tree; the v3/v4 dirs' build is the reference.
if(COMMAND hk_add_python_kernel)
    hk_add_python_kernel(harness_kernel harness_kernels.cpp)
else()
    # Fallback explicit form (mirror flags used by v3_fused_kernel).
    find_package(pybind11 REQUIRED)
    pybind11_add_module(harness_kernel harness_kernels.cpp)
    target_compile_options(harness_kernel PRIVATE --offload-arch=gfx950 -O3)
    # HK_ROOT / IRIS_ROOT include dirs are injected by the parent distributed-kernels CMake;
    # if building standalone, add:  target_include_directories(harness_kernel PRIVATE <HK_ROOT>/include <IRIS_ROOT>)
endif()
