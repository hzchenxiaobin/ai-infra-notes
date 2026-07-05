// histogram.cu —— Global vs Shared Memory Histogram 对比
// 编译命令: nvcc -o histogram histogram.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>

// Version 1: Global atomic (baseline)
__global__ void histogram_global(const int* input, int* hist, int N, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&hist[input[idx]], 1);
}

// Version 2: Shared memory privatization (optimized)
__global__ void histogram_shared(const int* input, int* hist, int N, int B) {
    __shared__ int s_hist[256];  // B <= 256

    // 初始化 shared histogram
    for (int i = threadIdx.x; i < B; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // 每个 block 累加到 shared memory (atomic 在 shared mem 上, 快得多)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x) {
        atomicAdd(&s_hist[input[i]], 1);
    }
    __syncthreads();

    // 合并到 global histogram
    for (int i = threadIdx.x; i < B; i += blockDim.x)
        atomicAdd(&hist[i], s_hist[i]);
}

int main() {
    const int N = 1 << 20;
    const int B = 256;
    int *h_in = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = rand() % B;

    int *d_in, *d_hist;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_hist, B * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = 256;

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);

    // Global atomic
    cudaMemset(d_hist, 0, B * sizeof(int));
    cudaEventRecord(s1);
    histogram_global<<<grid, block>>>(d_in, d_hist, N, B);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_global; cudaEventElapsedTime(&ms_global, s1, s2);

    // Shared memory privatization
    cudaMemset(d_hist, 0, B * sizeof(int));
    cudaEventRecord(s1);
    histogram_shared<<<grid, block>>>(d_in, d_hist, N, B);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_shared; cudaEventElapsedTime(&ms_shared, s1, s2);

    printf("Global atomic: %.3f ms\n", ms_global);
    printf("Shared privat: %.3f ms\n", ms_shared);
    printf("Speedup: %.2fx\n", ms_global / ms_shared);

    free(h_in); cudaFree(d_in); cudaFree(d_hist);
    return 0;
}
