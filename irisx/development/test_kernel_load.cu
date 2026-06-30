// Minimal test: does iris ctx.load() inside a __global__ kernel return correct remote values?
// Build: hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O0
//   -I$HOME/HipKittens/include -I/opt/rocm/include/hip
//   -I<iris_src>/irisx/include
//   test_kernel_load.cu -o test_kernel_load
//   Then run with: mpirun -np 8 -x HSA_XNACK=1 ./test_kernel_load

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include "iris/iris.hpp"

#define hip_check(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { printf("HIP error %d: %s at line %d\n", _e, hipGetErrorString(_e), __LINE__); exit(1); } \
} while(0)

// Simple kernel: for each element in dst, read from the remote rank's src via ctx.load()
__global__ void remote_read_kernel(
    iris::iris_device_view ctx,
    const float* local_src,   // this rank's src buffer (on iris heap)
    float* local_dst,         // this rank's dst buffer (regular GPU mem)
    int src_rank,             // which rank to read from
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Translate local_src pointer to remote rank's equivalent pointer and load
    local_dst[i] = ctx.load(&local_src[i], src_rank);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Init iris (allocates fine-grained heap, exchanges IPC handles)
    iris::iris iris_ctx(512 * 1024 * 1024, rank, world);
    hipSetDevice(rank);

    const int N = 8;

    // Allocate src on iris heap (symmetric order across ranks)
    float* src = iris_ctx.allocate<float>(N);

    // Each rank writes rank*100 + i to its src
    float host_src[N];
    for (int i = 0; i < N; i++) host_src[i] = rank * 100.0f + i;
    hip_check(hipMemcpy(src, host_src, N * sizeof(float), hipMemcpyHostToDevice));
    hip_check(hipDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0: read from rank 1's src buffer
    if (rank == 0) {
        int read_from = 1;
        float* dst;
        hip_check(hipMalloc(&dst, N * sizeof(float)));
        hip_check(hipMemset(dst, 0, N * sizeof(float)));

        auto dv = iris_ctx.get_device_view();
        remote_read_kernel<<<1, N>>>(dv, src, dst, read_from, N);
        hip_check(hipDeviceSynchronize());

        float host_dst[N];
        hip_check(hipMemcpy(host_dst, dst, N * sizeof(float), hipMemcpyDeviceToHost));

        printf("rank=0 reading from rank=%d:\n", read_from);
        bool all_ok = true;
        for (int i = 0; i < N; i++) {
            float expected = read_from * 100.0f + i;
            bool ok = (host_dst[i] == expected);
            printf("  [%d] got=%.1f expected=%.1f %s\n", i, host_dst[i], expected, ok ? "OK" : "WRONG");
            if (!ok) all_ok = false;
        }
        printf("Result: %s\n", all_ok ? "PASS" : "FAIL");
        hip_check(hipFree(dst));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
