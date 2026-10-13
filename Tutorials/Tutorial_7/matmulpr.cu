#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define MATRIX_SIZE 1024

__global__ void matrixMul(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void initMatrix(int *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 10;
    }
}

int main() {
    int N = MATRIX_SIZE;
    size_t size = (size_t)N * N * sizeof(int);

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);
    int *h_C_ref = (int *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL || h_C_ref == NULL) {
        printf("Host memory allocation failed\n");
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
        return 1;
    }

    srand((unsigned int)time(NULL));
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // CPU reference multiplication and timing
    clock_t cpu_start = clock();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }

    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Each thread calculates one output element; no tiling/shared memory
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    double gpu_time = gpu_time_ms / 1000.0;

    // Copy output back to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify GPU output against CPU reference
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != h_C_ref[i]) {
            correct = false;
            printf("Mismatch at index %d: GPU=%d, CPU=%d\n",
                   i, h_C[i], h_C_ref[i]);
            break;
        }
    }

    printf("Matrix Size: %d x %d\n", N, N);
    printf("Data Type: int\n");
    printf("Kernel: Naive matrix multiplication (no tiling)\n");
    printf("CPU Time: %.4f seconds\n", cpu_time);
    printf("GPU Time: %.6f seconds\n", gpu_time);

    if (gpu_time > 0.0) {
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    }

    printf("Results %s\n", correct ? "match!" : "do not match!");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return correct ? 0 : 1;
}
