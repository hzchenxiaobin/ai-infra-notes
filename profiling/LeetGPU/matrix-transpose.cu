// matrix_transpose.cu —— Shared Memory 优化的矩阵转置
// 编译命令: nvcc -o matrix_transpose matrix_transpose.cu -O3 -arch=sm_80

#include <cuda_runtime.h>
#include <cstdio>

#define TILE_DIM 32

__global__ void transpose_naive(const float* in, float* out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M)
        out[x * M + y] = in[y * N + x];
}

__global__ void transpose_shared(const float* in, float* out, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < M && y < N)
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int M = 2048, N = 2048;
    size_t bytes = M * N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < M * N; i++) h_in[i] = (float)rand() / RAND_MAX;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Naive
    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);
    cudaEventRecord(s1);
    transpose_naive<<<grid, block>>>(d_in, d_out, M, N);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_naive; cudaEventElapsedTime(&ms_naive, s1, s2);

    // Shared Memory
    cudaEventRecord(s1);
    transpose_shared<<<grid, block>>>(d_in, d_out, M, N);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_shared; cudaEventElapsedTime(&ms_shared, s1, s2);

    printf("Naive:  %.3f ms (%.1f GB/s)\n", ms_naive, 2.0f * bytes / (ms_naive * 1e6));
    printf("Shared: %.3f ms (%.1f GB/s)\n", ms_shared, 2.0f * bytes / (ms_shared * 1e6));
    printf("Speedup: %.2fx\n", ms_naive / ms_shared);

    free(h_in); cudaFree(d_in); cudaFree(d_out);
    return 0;
}
