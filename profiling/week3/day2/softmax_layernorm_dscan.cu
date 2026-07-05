// softmax_layernorm_dscan.cu —— Softmax + LayerNorm 不同 D 值性能扫描（实验 1）
// 编译命令: nvcc -o sl_dscan softmax_layernorm_dscan.cu -O3 -arch=sm_80 -lineinfo
// 运行命令: ./sl_dscan

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

__global__ void softmax_kernel(const float* __restrict__ input,
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

__global__ void layernorm_kernel(const float* __restrict__ input,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ output,
                                  int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = input + row * N;
    float* out_row = output + row * N;
    __shared__ float smem[32];
    __shared__ float row_mean, row_rstd;
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) local_sum += in_row[i];
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_mean = local_sum / N;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = in_row[i] - row_mean;
        local_sq += diff * diff;
    }
    local_sq = blockReduceSum(local_sq, smem);
    if (tid == 0) row_rstd = rsqrtf(local_sq / N + eps);
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x)
        out_row[i] = (in_row[i] - row_mean) * row_rstd * gamma[i] + beta[i];
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
}

int main() {
    const int M = 128;
    const int threads = 256;
    const float eps = 1e-5f;
    int test_D[] = {256, 512, 768, 1024, 2048, 4096};
    int num_D = sizeof(test_D) / sizeof(test_D[0]);

    printf("=== Softmax + LayerNorm D-scan (memory-bound scale law) ===\n");
    printf("M=%d, threads=%d\n\n", M, threads);
    printf("%-8s %-14s %-14s %-14s %-14s\n", "D", "SM time(ms)", "LN time(ms)", "SM BW(GB/s)", "LN BW(GB/s)");
    printf("------------------------------------------------------------------------\n");

    for (int d = 0; d < num_D; d++) {
        int D = test_D[d];
        size_t bytes = (size_t)M * D * sizeof(float);

        float *h_in = (float*)malloc(bytes);
        float *h_out = (float*)malloc(bytes);
        float *h_gamma = (float*)malloc(D * sizeof(float));
        float *h_beta = (float*)malloc(D * sizeof(float));
        initData(h_in, M * D);
        for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

        float *d_in, *d_out, *d_gamma, *d_beta;
        cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
        cudaMalloc(&d_gamma, D * sizeof(float)); cudaMalloc(&d_beta, D * sizeof(float));
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Softmax
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++)
            softmax_kernel<<<M, threads>>>(d_in, d_out, M, D);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_sm; cudaEventElapsedTime(&ms_sm, start, stop); ms_sm /= 50;

        // LayerNorm
        cudaEventRecord(start);
        for (int i = 0; i < 50; i++)
            layernorm_kernel<<<M, threads>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_ln; cudaEventElapsedTime(&ms_ln, start, stop); ms_ln /= 50;

        // 带宽 = 3 * M * D * 4B / time（三遍扫描约读 3 遍）
        double sm_bw = 3.0 * M * D * sizeof(float) / (ms_sm * 1e6);
        double ln_bw = 3.0 * M * D * sizeof(float) / (ms_ln * 1e6);

        printf("%-8d %-14.4f %-14.4f %-14.1f %-14.1f\n", D, ms_sm, ms_ln, sm_bw, ln_bw);

        free(h_in); free(h_out); free(h_gamma); free(h_beta);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n观察要点：\n");
    printf("1. D 翻倍 → 时间接近翻倍（memory-bound，时间 ≈ Bytes/Bandwidth）\n");
    printf("2. 带宽利用率应相对稳定（受 DRAM 带宽限制）\n");
    printf("3. ncu 验证：DRAM Throughput >> SM Throughput → memory-bound\n\n");
    printf("=== ncu 命令（指定 D=4096）===\n");
    printf("ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  smsp__average_warps_issue_stalled_long_scoreboard.pct \\\n");
    printf("  --kernel-name regex:\"softmax_kernel|layernorm_kernel\" \\\n");
    printf("  ./sl_dscan\n");

    return 0;
}
