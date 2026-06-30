// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "iris/iris.hpp"
#include "utils.hpp"

template <typename T>
__global__ void put_kernel(T* data, T value, std::size_t num_elements,
                           int remote_rank, iris::iris_device_view iris_view) {
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto grid_stride = blockDim.x * gridDim.x;

  for (int i = thread_id; i < num_elements; i += grid_stride) {
    iris_view.store(data + i, value, remote_rank);
  }
}

template <typename T>
std::pair<float, double> run_experiment(int source_rank, int remote_rank,
                                        int rank, int world_size) {
  static constexpr std::size_t heap_size_bytes = 1024 * 1024 * 1024;
  iris::iris iris(heap_size_bytes, rank, world_size);

  static constexpr std::size_t num_elements = 1024;
  auto* data = iris.allocate<T>(num_elements);

  gpu_timer timer;
  timer.start();

  if (rank == source_rank) {
    const int block_size = 1024;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    put_kernel<<<num_blocks, block_size>>>(data, 42, num_elements, remote_rank,
                                           iris.get_device_view());
  }
  iris.barrier();
  timer.stop();

  const auto elapsed_ms = timer.elapsed_ms();
  std::printf("rank: %d, data[0]: %d, elapsed_ms: %f\n", rank, data[0],
              elapsed_ms);

  const double bytes_transfered = static_cast<double>(num_elements) * sizeof(T);
  return {elapsed_ms, bytes_transfered};
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <source_rank> <remote_rank>\n", argv[0]);
    return 1;
  }
  const int source_rank = std::atoi(argv[1]);
  const int remote_rank = std::atoi(argv[2]);

  const auto [rank, world_size] = iris::mpi::initialize();

  std::printf("Running Put from rank %d to rank %d\n", source_rank,
              remote_rank);

  static constexpr auto num_experiments = 10;
  using T = int;

  float total_elapsed_ms = 0;
  double total_bytes_transfered = 0;
  for (int i = 0; i < num_experiments; i++) {
    std::printf("Experiment %d\n", i);
    const auto [elapsed_ms, bytes_transfered] =
        run_experiment<T>(source_rank, remote_rank, rank, world_size);
    total_elapsed_ms += elapsed_ms;
    total_bytes_transfered += bytes_transfered;
  }
  float average_elapsed_ms = total_elapsed_ms / num_experiments;
  const double average_bytes_transfered =
      total_bytes_transfered / num_experiments;
  const double average_bytes_per_ms = average_bytes_transfered / average_elapsed_ms;

  std::printf("Average elapsed_ms: %f\n", average_elapsed_ms);
  std::printf("Average bytes_transfered: %f\n", average_bytes_transfered);
  std::printf("Average GB/s: %f\n",
              average_bytes_per_ms * 1000.0 / 1024.0 / 1024.0 / 1024.0);

  iris::mpi::finalize();
  return 0;
}
