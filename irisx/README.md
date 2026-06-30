# Irisx

> [!IMPORTANT]
> This is a WIP project and a POC at the moment to explore simple R(D)MA APIs.

A modern C++ take on RMA/RDMA operations using AMD ROCm HIP. This is the C++ version of [Iris](https://github.com/ROCm/iris), designed for high-performance distributed computing applications with simple, intuitive and modern APIs.

> **Fork note — DeepSeek-R1 MoE expert region (HipKittens + IRIS).** This fork layers a fused MoE
> expert-region effort on top of the IRIS library. **➜ START AT [`../MASTER_HANDOFF.md`](../MASTER_HANDOFF.md)**
> — goal, results, cluster setup, and how to resume. The final, usable kernel is **`fused_moe/`** (prefill
> **1.56×** over the unfused baseline; decode ~2–3% at matched fp8 precision); the unfused baseline is
> **`baselines/`**; archived dev kernels + infra are in **`development/`**; authoritative measurements are in
> **[`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)**. The detailed notes below are **historical** (they predate
> the 2026-06-30 reorg: `b1_dispatch/` → `fused_moe/`, `b2_production/` → `baselines/`, dev → `development/`);
> the upstream IRIS library docs follow them.
>
> **Complete expert region (2026-06-29).** `b1_dispatch/` now runs the FULL MoE expert region, not
> just gather + one projection: `FFN=full` chains `grouped_gemm_b0` twice — **fc1 g1u1** (N=4096,
> K=7168) → `SiLU(gate)*up` + fp8 intermediate re-quant → **fc2 down** (N=7168, K=2048) — and
> `COMBINE=1` adds an IRIS scatter-back (`combine_scatter`, the EpCombine equivalent: per-row
> `acc[src_token] += route_weight * out` over origin ranks). The timed region is then
> **gather → fc1 → act → fc2 → combine**, head-to-head against the production unfused region in
> `b2_production/b3_ep8_unfused.py` (dispatch / fmoe / combine). Recipe: `b1_dispatch/B1_DISPATCH.md`
> §"Complete region". All additive — `FFN=single` is the original verified single-GEMM path.
>
> **Benchmarking the fused MoE kernel fairly (C4 = TP4/DP2+EP).** The 714µs-vs-1255µs result in the
> ledger compares mismatched regions at a prefill-like operating point. The fair-baseline toolkit:
> **`BENCHMARKING_METHODOLOGY.md`** (region-not-kernels rule + the 3-tier baseline recipe + how
> people benchmark multi-GPU fused-vs-unfused), **`b2_production/b2_unfused_region.py`** (Tier-2
> replay harness: runs the exact stock aiter sort→quant→fmoe chain in sequence + the trace's
> EpDispatch/EpCombine terms, sweeps `M_e`), **`b2_production/moe_cost_model.py`** (analytic
> roofline — predicts the decode regime is comm/overhead-bound and the fusion ceiling), and
> **`C4_RESULTS_DECK.html`** (mentor-facing deck of the C4 results + benchmarking plan). Prior
> baseline analysis: `UNFUSED_FUSED_BASELINE_FINDINGS.md`.

## Overview

Irisx provides a high-level C++ interface for performing remote memory operations across multiple GPUs in a distributed system.

## Features

- **Header-only library** - Easy integration with existing C++ projects
- **Simple and Intuitive APIs** - MultiGPU programming doesn't have to be that complicated 
- **Designed for HIP-like memory model** - Support for various memory ordering and scoping
- **Remote load, store and atomics** - Complete set of remote memory operations

## Requirements

- **C++20** compatible compiler
- **CMake 3.20** or higher
- **MPI** implementation (OpenMPI, MPICH, etc.)
- **AMD ROCm** with HIP support
- **Linux** operating system

## Building

### Prerequisites

Ensure you have the following installed:
- AMD ROCm drivers and HIP runtime that are C++20 compatible
- MPI implementation
- CMake 3.20+

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/ROCm/irisx
cd irisx

cmake -B build
cmake --build build --parallel 8
```

### Build Options

- `IRIS_BUILD_EXAMPLES` (default: ON) - Build example programs
- `IRIS_BUILD_BENCHMARKS` (default: ON) - Build benchmark programs

## Usage

### Basic Example

```cpp
#include "iris/iris.hpp"
#include <iostream>

// Kernel demonstrating both put and get operations
template <typename T>
__global__ void remote_operations_kernel(T* local_data, T* remote_data, 
                                        int data_size, int target_rank,
                                        iris::iris_device_view iris_view) {
    const int tid = threadIdx.x;
    const int rank = iris_view.cur_rank();
    
    if (tid < data_size) {
        // PUT operation: Send data to remote rank
        T value_to_send = rank * 1000 + tid;
        iris_view.store(&remote_data[tid], value_to_send, target_rank);
        
        // GET operation: Load data from remote rank
        T received_value = iris_view.load(&local_data[tid], target_rank);
        
        // Print results (only first few elements to avoid spam)
        if (tid < 3) {
            printf("Rank %d: sent %d to rank %d, received %d from rank %d\n", 
                   rank, value_to_send, target_rank, received_value, target_rank);
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    const auto [rank, world_size] = iris::mpi::initialize();
    
    if (world_size < 2) {
        std::cerr << "This example requires at least 2 ranks" << std::endl;
        iris::mpi::finalize();
        return 1;
    }
    
    std::cout << "Rank " << rank << " of " << world_size << " starting remote operations example" << std::endl;
    
    // Create Iris instance with 1GB heap
    static constexpr std::size_t heap_size_bytes = 1024 * 1024 * 1024;
    iris::iris iris(heap_size_bytes, rank, world_size);
    
    // Allocate memory for local and remote operations
    const int data_size = 16;
    auto* local_data = iris.allocate<int>(data_size);
    auto* remote_data = iris.allocate<int>(data_size);
    
    // Initialize local data
    for (int i = 0; i < data_size; ++i) {
        local_data[i] = rank * 1000 + i;
    }
    
    // Get device view for GPU operations
    auto device_view = iris.get_device_view();
    
    // Determine target rank (send to next rank in ring)
    const int target_rank = (rank + 1) % world_size;
    
    std::cout << "Rank " << rank << ": Starting remote operations with target rank " << target_rank << std::endl;
    
    // Launch kernel to perform remote operations
    remote_operations_kernel<<<1, data_size>>>(local_data, remote_data, data_size, target_rank, device_view);
    
    // Synchronize to ensure all operations are complete
    iris.barrier();
    
    // Verify results by copying data back to host
    std::vector<int> local_results(data_size);
    std::vector<int> remote_results(data_size);
    
    hip_try(hipMemcpy(local_results.data(), local_data, data_size * sizeof(int), hipMemcpyDeviceToHost));
    hip_try(hipMemcpy(remote_results.data(), remote_data, data_size * sizeof(int), hipMemcpyDeviceToHost));
    
    // Print verification results
    std::cout << "Rank " << rank << " verification:" << std::endl;
    std::cout << "  Local data (first 3): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << local_results[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  Remote data (first 3): ";
    for (int i = 0; i < 3; ++i) {
        std::cout << remote_results[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Rank " << rank << ": Remote operations completed successfully!" << std::endl;
    
    // Cleanup
    iris::mpi::finalize();
    return 0;
}
```

### Running Programs

Use the provided script to run multi-GPU programs:

```bash
# Run with 4 GPUs
./scripts/iris_run 4 ./build/benchmarks/put

# Or manually with mpirun
mpirun -np 4 ./build/benchmarks/put
```

## API Reference

### Core Classes

#### `iris::iris`
Main class for managing distributed memory and communication.

```cpp
iris::iris(heap_size_bytes, rank, world_size, verbose = false)
```

**Methods:**
- `allocate<T>(num_elements)` - Allocate memory
- `deallocate(ptr)` - Free allocated memory
- `barrier()` - Synchronize all ranks
- `get_device_view()` - Get device view for GPU operations

#### `iris::iris_device_view`
Device-side interface for remote memory operations.

**Atomic Operations:**
- `atomic_load<T>(ptr, remote_rank, order)` - Atomic load
- `atomic_store<T>(ptr, value, remote_rank, order)` - Atomic store
- `fetch_add<T>(ptr, value, remote_rank, order)` - Atomic add
- `fetch_sub<T>(ptr, value, remote_rank, order)` - Atomic subtract
- `compare_exchange_strong<T>(ptr, expected, desired, remote_rank, order)` - Compare and swap

**Non-atomic Operations:**
- `load<T>(ptr, remote_rank)` - Load value
- `store<T>(ptr, value, remote_rank)` - Store value

**Synchronization:**
- `fence<scope>(order)` - Memory fence

### Memory Ordering and Scoping

```cpp
// Memory ordering options
iris::memory_order_relaxed
iris::memory_order_consume
iris::memory_order_acquire
iris::memory_order_release
iris::memory_order_acq_rel
iris::memory_order_seq_cst

// Memory scoping options
iris::memory_scope_thread
iris::memory_scope_warp
iris::memory_scope_block
iris::memory_scope_device
iris::memory_scope_system
```

## Benchmarks

The project includes several benchmark programs:

- `put.hip` - Basic remote store operations
- `all_put.hip` - All-to-all remote store operations

Run benchmarks with:
```bash
cd build
./scripts/iris_run 4 ./benchmarks/put
```

## Examples

See the `examples/` directory for complete working examples demonstrating:
- **Remote Operations** - Core PUT and GET operations in a single kernel
- **Point-to-Point Communication** - Basic communication between neighboring ranks
- **Ring Communication** - Data circulation around all ranks in a ring topology
- **Broadcast Pattern** - One-to-many communication from rank 0 to all others
- Basic remote memory operations
- Atomic operations with different memory ordering
- Multi-GPU communication patterns

### Communication Examples

The examples demonstrate fundamental distributed communication patterns:

#### Remote Operations (`examples/remote_operations.hip`)
- Demonstrates both PUT and GET operations in a single kernel
- Shows core remote memory operations using `iris_view.store()` and `iris_view.load()`
- Bidirectional communication between ranks

#### Point-to-Point (`examples/point_to_point.hip`)
- Simple communication between neighboring ranks
- Shows basic remote memory operations using `iris_view.store()`

#### Ring Communication (`examples/ring_communication.hip`)
- Data circulates around all ranks in a ring topology
- Demonstrates collective communication patterns

#### Broadcast (`examples/broadcast.hip`)
- Rank 0 broadcasts data to all other ranks
- Shows one-to-many communication patterns

Run any example:
```bash
./scripts/iris_run 4 ./build/examples/remote_operations
./scripts/iris_run 4 ./build/examples/point_to_point
./scripts/iris_run 4 ./build/examples/ring_communication
./scripts/iris_run 4 ./build/examples/broadcast
```


## Support

Need help? We're here to support you! Here are a few ways to get in touch:

1. **Open an Issue**: Found a bug or have a feature request? [Open an issue](https://github.com/ROCm/irisx/issues/new/choose) on GitHub
2. **Contact the Team**: If GitHub issues aren't working for you or you need to reach us directly, feel free to contact our development team

We welcome your feedback and contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


