#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 2
#define MATRIX_SIZE 4

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

// Function to print matrix
void printMatrix(const char *name, float *matrix, int N) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int N = MATRIX_SIZE;
    size_t size = N * N * sizeof(float);
    
    float h_A[16] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    
    float h_B[16] = {
        16.0, 15.0, 14.0, 13.0,
        12.0, 11.0, 10.0, 9.0,
        8.0, 7.0, 6.0, 5.0,
        4.0, 3.0, 2.0, 1.0
    };
    
    float h_C[16];
    float h_C_ref[16];
    
    printf("Input matrices (4x4):\n");
    printMatrix("Matrix A", h_A, N);
    printMatrix("Matrix B", h_B, N);
    
    // Compute reference result on CPU
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_C_ref[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                h_C_ref[i * N + j] += h_A[i * N + k] * h_B[k * N + j];
            }
        }
    }
    
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
    
    // Launch kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("\nGPU Result (using tiling):\n");
    printMatrix("Matrix C (GPU)", h_C, N);
    
    printf("CPU Reference result:\n");
    printMatrix("Matrix C (CPU)", h_C_ref, N);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Results match!\n");
    } else {
        printf("Results don't match\n");
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}