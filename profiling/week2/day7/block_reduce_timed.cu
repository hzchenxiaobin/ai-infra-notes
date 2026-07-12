// block_reduce_timed.cu —— 30 分钟手撕 Block Reduce（ncu profiling 版）
// 编译: nvcc -o block_reduce block_reduce_timed.cu -O3 -arch=sm_120 -g -lineinfo
// 运行: ./block_reduce

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void blockReduceSum(const float* in, float* out, int n) {
    __shared__ float warpSums[32];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    // Step 1: grid-stride 累加
    float sum = 0.0f;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }

    // Step 2: Warp 级归约
    sum = warpReduceSum(sum);

    // Step 3: lane 0 写入 Shared Memory
    if (lane == 0)
        warpSums[wid] = sum;
    __syncthreads();

    // Step 4: Warp 0 做最终归约
    if (wid == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0)
            out[blockIdx.x] = sum;
    }
}

int main() {
    const int N = 1 << 22; // 4M 元素
    printf("=== Block Reduce (Warp Shuffle + 两级归约) ===\n");
    printf("N = %d (%.2f MB)\n\n", N, N * sizeof(float) / (1024.0 * 1024.0));

    float* h_in = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 1000) * 0.001f;
    }

    float *d_in, *d_tmp, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_tmp, 1024 * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    blockReduceSum<<<blocks, threads>>>(d_in, d_tmp, N);
    blockReduceSum<<<1, 256>>>(d_tmp, d_out, blocks);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        blockReduceSum<<<blocks, threads>>>(d_in, d_tmp, N);
        blockReduceSum<<<1, 256>>>(d_tmp, d_out, blocks);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100;

    float gpuSum;
    cudaMemcpy(&gpuSum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    double cpuSum = 0.0;
    for (int i = 0; i < N; i++) {
        cpuSum += h_in[i];
    }

    printf("GPU Sum: %.4f\n", gpuSum);
    printf("CPU Sum: %.4f\n", (float)cpuSum);
    printf("Diff:    %.6f (%s)\n", fabs(gpuSum - (float)cpuSum), fabs(gpuSum - (float)cpuSum) < 1e-3 ? "PASS" : "FAIL");
    printf("Time:    %.3f ms (%.2f GB/s bandwidth)\n", ms, N * sizeof(float) / (ms * 1e6));

    printf("\n=== ncu 分析命令 ===\n");
    printf("ncu --kernel-name regex:blockReduceSum \\\n");
    printf("  --metrics \\\n");
    printf("    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    launch__registers_per_thread,\\\n");
    printf("    smsp__average_warps_issue_stalled_long_scoreboard.pct \\\n");
    printf("  ./block_reduce\n");

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
