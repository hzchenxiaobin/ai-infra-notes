// warp_vs_block_dscan.cu —— Softmax warp 级 vs block 级不同 D 值性能对比（实验 2）
// 编译命令: nvcc -o warp_vs_block warp_vs_block_dscan.cu -O3 -arch=sm_120 -lineinfo
// 运行命令: ./warp_vs_block

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}
__inline__ __device__ float blockReduceSum(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
__inline__ __device__ float blockReduceMax(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// Block 级 Softmax（Day 16 基准）：一个 block 处理一行
__global__ void softmax_block_kernel(const float* __restrict__ input,
                                      float* __restrict__ output, int M, int D) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = input + row * D;
    float* out_row = output + row * D;
    __shared__ float smem[32];
    __shared__ float row_max, row_sum;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x)
        local_max = fmaxf(local_max, in_row[i]);
    local_max = blockReduceMax(local_max, smem);
    if (tid == 0) row_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x)
        local_sum += expf(in_row[i] - row_max);
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_sum = local_sum;
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < D; i += blockDim.x)
        out_row[i] = expf(in_row[i] - row_max) * inv_sum;
}

// Warp 级 Softmax（优化版）：一个 warp 处理一行，无 shared memory
__global__ void softmax_warp_kernel(const float* __restrict__ input,
                                     float* __restrict__ output, int M, int D) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (global_warp_id >= M) return;
    int lane = threadIdx.x % 32;
    const float* in_row = input + global_warp_id * D;
    float* out_row = output + global_warp_id * D;

    float local_max = -INFINITY;
    for (int i = lane; i < D; i += 32)
        local_max = fmaxf(local_max, in_row[i]);
    local_max = warpReduceMax(local_max);
    local_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    float local_sum = 0.0f;
    for (int i = lane; i < D; i += 32)
        local_sum += expf(in_row[i] - local_max);
    local_sum = warpReduceSum(local_sum);
    local_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);

    float inv_sum = 1.0f / local_sum;
    for (int i = lane; i < D; i += 32)
        out_row[i] = expf(in_row[i] - local_max) * inv_sum;
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
}

int main() {
    const int M = 1024;
    const int threads_block = 256;
    const int threads_warp = 128;  // 4 warps per block
    int test_D[] = {256, 512, 1024, 2048, 4096};
    int num_D = sizeof(test_D) / sizeof(test_D[0]);

    printf("=== Softmax: warp-level vs block-level D-scan ===\n");
    printf("M=%d\n\n", M);
    printf("%-8s %-16s %-16s %-12s\n", "D", "Block(ms)", "Warp(ms)", "Speedup");
    printf("--------------------------------------------------------\n");

    for (int d = 0; d < num_D; d++) {
        int D = test_D[d];
        size_t bytes = (size_t)M * D * sizeof(float);
        float *h_in = (float*)malloc(bytes);
        float *h_out = (float*)malloc(bytes);
        initData(h_in, M * D);

        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Block 级
        for (int i = 0; i < 3; i++) softmax_block_kernel<<<M, threads_block>>>(d_in, d_out, M, D);
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) softmax_block_kernel<<<M, threads_block>>>(d_in, d_out, M, D);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_block; cudaEventElapsedTime(&ms_block, start, stop); ms_block /= 50;

        // Warp 级
        int warps_per_block = threads_warp / 32;
        int grid_warp = (M + warps_per_block - 1) / warps_per_block;
        for (int i = 0; i < 3; i++) softmax_warp_kernel<<<grid_warp, threads_warp>>>(d_in, d_out, M, D);
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) softmax_warp_kernel<<<grid_warp, threads_warp>>>(d_in, d_out, M, D);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_warp; cudaEventElapsedTime(&ms_warp, start, stop); ms_warp /= 50;

        printf("%-8d %-16.4f %-16.4f %-12.2f\n", D, ms_block, ms_warp, ms_block / ms_warp);

        free(h_in); free(h_out);
        cudaFree(d_in); cudaFree(d_out);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n观察要点：\n");
    printf("1. D<=1024 时 warp 级通常更快（无 __syncthreads 开销）\n");
    printf("2. D=4096 时 warp 级每 lane 处理 128 元素，并行度下降\n");
    printf("3. 这就是 PyTorch 用 D=1024 做 dispatch 分界的原因\n\n");
    printf("=== ncu 命令 ===\n");
    printf("# 对比 warp vs block 在特定 D 下的指标\n");
    printf("ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__occupancy.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("  --kernel-name regex:\"softmax_warp_kernel|softmax_block_kernel\" \\\n");
    printf("  ./warp_vs_block\n");

    return 0;
}
