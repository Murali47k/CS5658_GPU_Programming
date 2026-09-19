#include <stdio.h>
#include <cuda_runtime.h>

// Implement atomicAdd using atomicCAS
__device__ int myAtomicAdd(int *address, int value) {
    int old = *address;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address,assumed,assumed + value);
    } while (old != assumed);

    return old;
}

// Kernel
__global__ void incrementCounter(int *counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d is adding 1\n", tid);
    myAtomicAdd(counter, 1);
}


int main() {
    int numBlocks = 4;
    int threadsPerBlock = 8;

    int totalThreads = numBlocks * threadsPerBlock;

    printf("Total threads = %d\n", totalThreads);

    int h_counter = 0;
    int *d_counter;

    cudaMalloc((void **)&d_counter, sizeof(int));
    cudaMemcpy(d_counter,&h_counter,sizeof(int),cudaMemcpyHostToDevice);

    incrementCounter<<<numBlocks, threadsPerBlock>>>(d_counter);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_counter,d_counter,sizeof(int),cudaMemcpyDeviceToHost);

    printf("\nFinal counter = %d\n", h_counter);
    printf("Expected counter = %d\n", totalThreads);

    cudaFree(d_counter);

    return 0;
}