// matrix_multiplication.cu —— Shared Memory Tiling GEMM
// 编译命令: nvcc -o matmul matmul.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_SIZE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_SIZE) {
        if (row < M && bk + threadIdx.x < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + bk + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            s_B[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Bank-conflict-free version: pad the shared memory arrays by one column.
// Without padding, s_A[threadIdx.y][k] causes a 2-way bank conflict because
// consecutive rows are 16 floats (64 bytes) apart, which is a multiple of the
// 32-bank shared-memory stride (4 bytes/bank). Padding breaks the alignment.
__global__ void matmul_tiled_nobc(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_SIZE) {
        if (row < M && bk + threadIdx.x < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + bk + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            s_B[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)rand() / RAND_MAX * 2 - 1;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);

    cudaEventRecord(s1);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(s2);
    cudaEventSynchronize(s2);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, s1, s2);

    cudaEventRecord(s1);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(s2);
    cudaEventSynchronize(s2);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, s1, s2);

    float gflops_naive = 2.0f * M * N * K / (ms_naive * 1e6);
    float gflops_tiled = 2.0f * M * N * K / (ms_tiled * 1e6);

    printf("Naive:  %.3f ms (%.1f GFLOPS)\n", ms_naive, gflops_naive);
    printf("Tiled:  %.3f ms (%.1f GFLOPS)\n", ms_tiled, gflops_tiled);
    printf("Speedup: %.2fx\n", ms_naive / ms_tiled);

    cudaEventRecord(s1);
    matmul_tiled_nobc<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(s2);
    cudaEventSynchronize(s2);
    float ms_tiled_nobc;
    cudaEventElapsedTime(&ms_tiled_nobc, s1, s2);
    float gflops_tiled_nobc = 2.0f * M * N * K / (ms_tiled_nobc * 1e6);

    printf("Tiled (no bank conflict): %.3f ms (%.1f GFLOPS)\n", ms_tiled_nobc, gflops_tiled_nobc);
    printf("Speedup vs Tiled: %.2fx\n", ms_tiled / ms_tiled_nobc);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
