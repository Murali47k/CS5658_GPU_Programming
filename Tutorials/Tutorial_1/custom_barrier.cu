#include <stdio.h>
#include <cuda_runtime.h>

__device__ int barrierCount = 0;

// Custom Barrier
__device__ void customBarrier(int num_blocks)
{
    // Make sure all threads have reached this point
    __syncthreads();

    // One thread increments the global counter
    if (threadIdx.x == 0)
    {
        atomicAdd(&barrierCount, 1);
    }

    // Wait until the counter indicates that the block has reached the barrier
    while (atomicAdd(&barrierCount, 0) < num_blocks)
    {
        // Busy wait
    }

    // Synchronize all threads after the atomic operation
    __syncthreads();
}


__global__ void KernelwithBarrier(int num_blocks)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    printf("Block %d Thread %d reached BEFORE barrier\n", bid, tid);

    // Custom barrier
    customBarrier(num_blocks);

    printf("Block %d Thread %d reached AFTER barrier\n", bid, tid);
}


int main()
{
    int threads = 8;
    int num_blocks = 2;

    printf("Launching CUDA kernel with %d threads...\n\n",threads);

    KernelwithBarrier<<<2, threads>>>(num_blocks);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s\n",cudaGetErrorString(error));
        return 1;
    }

    return 0;
}