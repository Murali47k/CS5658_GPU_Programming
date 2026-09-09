#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TILE_SIZE 16
#define MATRIX_SIZE 512

__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate global thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate row and column of the output matrix
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int m = 0; m < N / TILE_SIZE; m++) {
        
        // Load tiles into shared memory
        int aRow = by * TILE_SIZE + ty;
        int aCol = m * TILE_SIZE + tx;
        As[ty][tx] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
        
        int bRow = m * TILE_SIZE + ty;
        int bCol = bx * TILE_SIZE + tx;
        Bs[ty][tx] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        __syncthreads();  // Ensure all threads have loaded tiles
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();  // Ensure all threads have finished before next iteration
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to initialize matrix with random numbers 0-9
void initMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)(rand() % 10);
    }
}

int main() {
    int N = MATRIX_SIZE;
    size_t size = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);
    
    // Initialize matrices with random numbers
    srand(time(NULL));
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    
    // Compute reference result on CPU and time it
    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_C_ref[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                h_C_ref[i * N + j] += h_A[i * N + k] * h_B[k * N + j];
            }
        }
    }
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE);
    
    // Launch kernel and time it
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time = gpu_time / 1000.0; // Convert to seconds
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
            correct = false;
            break;
        }
    }
    
    // Print results
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Tile Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("CPU Time: %.4f seconds\n", cpu_time);
    printf("GPU Time: %.4f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    if (correct) {
        printf("Results match!\n");
    } else {
        printf("Results don't match\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}