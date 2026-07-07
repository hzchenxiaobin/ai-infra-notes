// histogram.cu —— Histogram: Global atomic vs Shared memory privatization（ncu profiling 版）
// 编译命令: nvcc -o histogram histogram.cu -O3 -arch=sm_80 -g -lineinfo
// 运行命令: ./histogram

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Version 1: Global atomic (baseline)
__global__ void histogram_global(const int* input, int* hist, int N, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&hist[input[idx]], 1);
}

// Version 2: Shared memory privatization (optimized)
__global__ void histogram_shared(const int* input, int* hist, int N, int B) {
    __shared__ int s_hist[256];  // B <= 256
    int tid = threadIdx.x;

    // 初始化 shared histogram
    for (int i = tid; i < B; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // 每个 block 累加到 shared memory
    for (int i = blockIdx.x * blockDim.x + tid; i < N;
         i += gridDim.x * blockDim.x) {
        atomicAdd(&s_hist[input[i]], 1);
    }
    __syncthreads();

    // 合并到 global histogram
    for (int i = tid; i < B; i += blockDim.x)
        atomicAdd(&hist[i], s_hist[i]);
}

void initData(int* data, int n, int B) {
    srand(42);
    for (int i = 0; i < n; i++)
        data[i] = rand() % B;
}

bool checkResult(const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at bin %d: global=%d, shared=%d\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1 << 20;  // 1M 元素
    const int B = 256;       // 256 bins
    const int threads = 256;
    const int blocks = min((N + threads - 1) / threads, 1024);

    printf("=== Histogram: Global atomic vs Shared memory ===\n");
    printf("N = %d, B = %d, blocks = %d, threads = %d\n\n", N, B, blocks, threads);

    int* h_input = (int*)malloc(N * sizeof(int));
    initData(h_input, N, B);

    int *d_input, *d_hist_global, *d_hist_shared;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_hist_global, B * sizeof(int));
    cudaMalloc(&d_hist_shared, B * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ===== Global atomic =====
    cudaMemset(d_hist_global, 0, B * sizeof(int));
    cudaEventRecord(start);
    histogram_global<<<blocks, threads>>>(d_input, d_hist_global, N, B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_global;
    cudaEventElapsedTime(&ms_global, start, stop);

    // ===== Shared memory privatization =====
    cudaMemset(d_hist_shared, 0, B * sizeof(int));
    cudaEventRecord(start);
    histogram_shared<<<blocks, threads>>>(d_input, d_hist_shared, N, B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_shared;
    cudaEventElapsedTime(&ms_shared, start, stop);

    // 验证
    int* h_hist_global = (int*)calloc(B, sizeof(int));
    int* h_hist_shared = (int*)calloc(B, sizeof(int));
    cudaMemcpy(h_hist_global, d_hist_global, B * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist_shared, d_hist_shared, B * sizeof(int), cudaMemcpyDeviceToHost);
    bool correct = checkResult(h_hist_global, h_hist_shared, B);

    printf("Global atomic:  %.3f ms\n", ms_global);
    printf("Shared memory:   %.3f ms\n", ms_shared);
    printf("Speedup:         %.2fx\n", ms_global / ms_shared);
    printf("Correctness:     %s\n\n", correct ? "PASS" : "FAIL");

    printf("=== ncu 分析命令 ===\n");
    printf("# Global atomic\n");
    printf("ncu --kernel-name regex:histogram_global \\\n");
    printf("  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\\\n");
    printf("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\\n");
    printf("sm__occupancy.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("  ./histogram\n\n");
    printf("# Shared memory privatization\n");
    printf("ncu --kernel-name regex:histogram_shared \\\n");
    printf("  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\\\n");
    printf("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\\n");
    printf("sm__occupancy.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("  ./histogram\n\n");
    printf("# 对比两个 kernel 的 HBM 读写量\n");
    printf("ncu --kernel-name regex:\"histogram_global|histogram_shared\" \\\n");
    printf("  --metrics dram__bytes_read.sum,dram__bytes_write.sum \\\n");
    printf("  ./histogram\n");

    free(h_input); free(h_hist_global); free(h_hist_shared);
    cudaFree(d_input); cudaFree(d_hist_global); cudaFree(d_hist_shared);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
