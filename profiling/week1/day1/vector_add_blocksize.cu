// vector_add_blocksize.cu —— Vector Add 不同 block size 性能对比（Day 1 LeetGPU 题目）
// 编译命令: nvcc -o vector_add_blocksize vector_add_blocksize.cu -O3 -arch=sm_120 -lineinfo
// 运行命令: ./vector_add_blocksize
// profiling: make profile

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 带计时和正确性验证的 benchmark 函数
float benchmark_blocksize(const float* d_A, const float* d_B, float* d_C, int N, int block_size, int iters = 100) {
    int grid_size = (N + block_size - 1) / block_size;

    // warmup
    vector_add<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        vector_add<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    const int N = 1 << 20; // 1M 元素
    size_t bytes = N * sizeof(float);

    printf("=== Vector Add: block size 性能对比 ===\n");
    printf("N = %d (%.1f MB per array)\n\n", N, (double)bytes / 1024 / 1024);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 1000) * 0.001f;
        h_B[i] = (float)(rand() % 1000) * 0.001f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 测试不同 block size
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("%-12s %-12s %-14s %-14s\n", "block_size", "grid_size", "time(ms)", "bandwidth(GB/s)");
    printf("--------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        int bs = block_sizes[i];
        float ms = benchmark_blocksize(d_A, d_B, d_C, N, bs);
        int grid = (N + bs - 1) / bs;
        // 带宽 = (读 A + 读 B + 写 C) / time = 3 * N * 4 / time
        double bandwidth = 3.0 * N * sizeof(float) / (ms * 1e-3) / 1e9;
        printf("%-12d %-12d %-14.4f %-14.1f\n", bs, grid, ms, bandwidth);
    }

    // 正确性验证
    float* h_C = (float*)malloc(bytes);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            pass = false;
            break;
        }
    }
    printf("\n正确性: %s\n", pass ? "PASS" : "FAIL");

    printf("\n=== ncu 分析命令 ===\n");
    printf("ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  launch__registers_per_thread \\\n");
    printf("  --kernel-name regex:vector_add \\\n");
    printf("  ./vector_add_blocksize\n");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
