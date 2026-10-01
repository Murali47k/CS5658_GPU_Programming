#include <cstdio>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define N 64
#define TILE 16

__global__ void matmul_tensor(const half *A,const half *B,float *C) {

    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    int row = blockIdx.y * 32 + warp_row * TILE;
    int col = blockIdx.x * 32 + warp_col * TILE;

    if (row >= N || col >= N)
        return;

    // Accumulator fragment
    wmma::fragment< wmma::accumulator, TILE, TILE, TILE, float > acc;
    wmma::fill_fragment(acc, 0.0f);

    // Iterate over K dimension
    for (int k = 0; k < N; k += TILE) {

        wmma::fragment< wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major > a_frag;
        wmma::fragment< wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major > b_frag;

        const half *a_ptr = A + row * N + k;
        const half *b_ptr = B + k * N + col;

        wmma::load_matrix_sync( a_frag, a_ptr, N );
        wmma::load_matrix_sync( b_frag, b_ptr, N );
        wmma::mma_sync( acc, a_frag, b_frag, acc );
    }

    // Store result
    float *c_ptr = C + row * N + col;
    wmma::store_matrix_sync( c_ptr, acc, N, wmma::mem_row_major );
}


int main() {
    const int sizeA = N * N;
    const int sizeB = N * N;
    const int sizeC = N * N;

    half *h_A = new half[sizeA];
    half *h_B = new half[sizeB];
    float *h_C = new float[sizeC];

    
    for (int i = 0; i < sizeA; i++)
        h_A[i] = __float2half(1.0f);

    for (int i = 0; i < sizeB; i++)
        h_B[i] = __float2half(1.0f);

    half *d_A;
    half *d_B;
    float *d_C;

    cudaMalloc(&d_A, sizeA * sizeof(half));
    cudaMalloc(&d_B, sizeB * sizeof(half));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy( d_A, h_A, sizeA * sizeof(half),cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, sizeB * sizeof(half),cudaMemcpyHostToDevice );


    dim3 block(128);
    dim3 grid( N / 32, N / 32);

    matmul_tensor<<<grid, block>>>(d_A,d_B,d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,sizeC * sizeof(float),cudaMemcpyDeviceToHost);

    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}