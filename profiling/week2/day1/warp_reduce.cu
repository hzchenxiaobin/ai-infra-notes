// warp_reduce.cu —— Warp 级 + Block 级两级归约完整实现（ncu profiling 版）
// 编译命令: nvcc -o warp_reduce warp_reduce.cu -O3 -arch=sm_80 -lineinfo
// 运行命令: ./warp_reduce

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// --------------------------------------------------
// Warp 级归约：使用 __shfl_down_sync 折半累加
// --------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------
// Warp 级归约（XOR 模式）：使用 __shfl_xor_sync
// 用途：when you need reduction result in ALL lanes, not just lane 0
// --------------------------------------------------
__inline__ __device__ float warpReduceSumXor(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------
// Block 级归约：每个 warp 先归约，然后 warp 0 做最终归约
// --------------------------------------------------
__global__ void blockReduceSum(const float* input, float* output, int n) {
    __shared__ float warpSums[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Step 1: 每个线程从 global memory 读取并做 per-thread 累加
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Step 2: Warp 级归约（每个 warp 的 32 个线程累加到 lane 0）
    sum = warpReduceSum(sum);

    // Step 3: lane 0 将 warp 的部分和写入 shared memory
    if (lane == 0) {
        warpSums[wid] = sum;
    }
    __syncthreads();

    // Step 4: Warp 0 做最终归约
    if (wid == 0) {
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// --------------------------------------------------
// 多 block 版本：需要第二次 kernel 调用汇总
// --------------------------------------------------
float launchReduce(const float* d_input, float* d_temp, float* d_output,
                   int n, int threadsPerBlock) {
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    blocks = min(blocks, 1024);

    blockReduceSum<<<blocks, threadsPerBlock>>>(d_input, d_temp, n);
    cudaDeviceSynchronize();

    blockReduceSum<<<1, 256>>>(d_temp, d_output, blocks);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// --------------------------------------------------
// Host 辅助函数
// --------------------------------------------------
void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
}

float cpuReduceSum(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return static_cast<float>(sum);
}

int main() {
    const int n = 1 << 22;  // 4,194,304 个元素
    printf("=== Warp Shuffle Block Reduce ===\n");
    printf("Array size: %d (%.2f MB)\n", n, n * sizeof(float) / (1024.0 * 1024.0));

    float* h_input = (float*)malloc(n * sizeof(float));
    initData(h_input, n);

    float *d_input, *d_temp, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_temp, 1024 * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float gpuSum = launchReduce(d_input, d_temp, d_output, n, 256);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float cpuSum = cpuReduceSum(h_input, n);
    float diff = fabs(gpuSum - cpuSum);

    printf("GPU Sum: %.6f\n", gpuSum);
    printf("CPU Sum: %.6f\n", cpuSum);
    printf("Diff:    %.6f (%s)\n", diff, diff < 1e-3 ? "PASS" : "FAIL");
    printf("Time:    %.3f ms (%.2f GB/s bandwidth)\n",
           ms, n * sizeof(float) / (ms * 1e6));

    printf("\n=== ncu 分析命令 ===\n");
    printf("ncu --metrics \\\n");
    printf("  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  launch__registers_per_thread,\\\n");
    printf("  smsp__average_warps_issue_stalled_long_scoreboard.pct \\\n");
    printf("  --kernel-name regex:blockReduceSum \\\n");
    printf("  ./warp_reduce\n");

    free(h_input);
    cudaFree(d_input); cudaFree(d_temp); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
