#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// 故意制造 bank conflict 的版本：tile[TILE_DIM][TILE_DIM]
__global__ void conflict_read(float* out, const float* in) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int col = threadIdx.x;
    for (int row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        tile[row + threadIdx.y][col] = in[(row + threadIdx.y) * TILE_DIM + col];
    }
    __syncthreads();

    // 同一 warp 内线程访问同一 column，产生 bank conflict
    for (int row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        out[(row + threadIdx.y) * TILE_DIM + col] = tile[col][row + threadIdx.y];
    }
}

// 使用 padding 消除 bank conflict：tile[TILE_DIM][TILE_DIM + 1]
__global__ void no_conflict_read(float* out, const float* in) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int col = threadIdx.x;
    for (int row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        tile[row + threadIdx.y][col] = in[(row + threadIdx.y) * TILE_DIM + col];
    }
    __syncthreads();

    for (int row = 0; row < TILE_DIM; row += BLOCK_ROWS) {
        out[(row + threadIdx.y) * TILE_DIM + col] = tile[col][row + threadIdx.y];
    }
}

int main() {
    const int N = TILE_DIM * TILE_DIM;
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(1, 1);

    conflict_read<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();

    no_conflict_read<<<grid, block>>>(d_out, d_in);
    cudaDeviceSynchronize();

    printf("Bank conflict kernels finished. Use ncu to compare metrics.\n");

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
