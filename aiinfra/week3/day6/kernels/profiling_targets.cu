// profiling_targets.cu —— 端到端 Profiling 靶点：memory-bound Softmax + compute-bound GEMM
// 编译命令: nvcc -o profiling_targets kernels/profiling_targets.cu -O3 -arch=sm_120 -lineinfo
// 运行命令: ./profiling_targets
// ncu 分析: ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed --kernel-name regex:"softmax_kernel|gemm_kernel" ./profiling_targets

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// Warp / Block reduce 原语（复用 Week 2 Day 1 / Week 3 Day 2）
// ============================================================
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
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

// ============================================================
// [Memory-bound] Softmax Kernel：一行一个 block，三遍扫描 safe softmax
// 预期 ncu：DRAM Throughput >> SM Throughput
// ============================================================
__global__ void softmax_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int M, int D) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = input + row * D;
    float* out_row = output + row * D;
    __shared__ float smem[32];
    __shared__ float row_max, row_sum;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    local_max = blockReduceMax(local_max, smem);
    if (tid == 0) row_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_sum += expf(in_row[i] - row_max);
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_sum = local_sum;
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < D; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - row_max) * inv_sum;
    }
}

// ============================================================
// [Compute-bound] Naive GEMM Kernel：C = A·B，A∈R^{M×K}, B∈R^{K×N}
// 故意不做 tiling，但仍能体现 compute-bound 特征（SM >> DRAM）
// ============================================================
__global__ void gemm_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ============================================================
// Host 辅助
// ============================================================
void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    }
}

void cpuSoftmax(const float* in, float* out, int M, int D) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * D;
        float* orow = out + r * D;
        float mx = ir[0];
        for (int i = 1; i < D; i++) {
            mx = fmaxf(mx, ir[i]);
        }
        float s = 0.0f;
        for (int i = 0; i < D; i++) { orow[i] = expf(ir[i] - mx); s += orow[i]; }
        for (int i = 0; i < D; i++) {
            orow[i] /= s;
        }
    }
}

int main() {
    // Softmax 配置（memory-bound 靶点）
    const int SM_M = 256, SM_D = 1024;
    // GEMM 配置（compute-bound 靶点，规模足够大让 compute 特征显现）
    const int G_M = 512, G_N = 512, G_K = 512;

    printf("=== Profiling Targets: Softmax(memory-bound) + GEMM(compute-bound) ===\n");
    printf("Softmax: M=%d, D=%d\n", SM_M, SM_D);
    printf("GEMM:    M=%d, N=%d, K=%d\n\n", G_M, G_N, G_K);

    // ---- Softmax 内存 ----
    size_t sm_bytes = (size_t)SM_M * SM_D * sizeof(float);
    float *h_sm_in = (float*)malloc(sm_bytes), *h_sm_out = (float*)malloc(sm_bytes), *h_sm_ref = (float*)malloc(sm_bytes);
    initData(h_sm_in, SM_M * SM_D);
    float *d_sm_in, *d_sm_out;
    cudaMalloc(&d_sm_in, sm_bytes); cudaMalloc(&d_sm_out, sm_bytes);
    cudaMemcpy(d_sm_in, h_sm_in, sm_bytes, cudaMemcpyHostToDevice);

    // ---- GEMM 内存 ----
    size_t a_bytes = (size_t)G_M * G_K * sizeof(float);
    size_t b_bytes = (size_t)G_K * G_N * sizeof(float);
    size_t c_bytes = (size_t)G_M * G_N * sizeof(float);
    float *h_A = (float*)malloc(a_bytes), *h_B = (float*)malloc(b_bytes), *h_C = (float*)malloc(c_bytes);
    initData(h_A, G_M * G_K); initData(h_B, G_K * G_N);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_bytes); cudaMalloc(&d_B, b_bytes); cudaMalloc(&d_C, c_bytes);
    cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // ---- 运行 Softmax ----
    cudaEventRecord(start);
    softmax_kernel<<<SM_M, 256>>>(d_sm_in, d_sm_out, SM_M, SM_D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sm_ms;
    cudaEventElapsedTime(&sm_ms, start, stop);
    cudaMemcpy(h_sm_out, d_sm_out, sm_bytes, cudaMemcpyDeviceToHost);
    cpuSoftmax(h_sm_in, h_sm_ref, SM_M, SM_D);
    float sm_diff = 0.0f;
    for (int i = 0; i < SM_M * SM_D; i++) {
        sm_diff = fmaxf(sm_diff, fabsf(h_sm_out[i] - h_sm_ref[i]));
    }
    printf("[Softmax] time=%.3f ms  maxDiff=%.2e  (%s)\n", sm_ms, sm_diff,
           sm_diff < 1e-5f ? "PASS" : "FAIL");

    // ---- 运行 GEMM ----
    dim3 block(16, 16);
    dim3 grid((G_N + block.x - 1) / block.x, (G_M + block.y - 1) / block.y);
    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, G_M, G_N, G_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gemm_ms;
    cudaEventElapsedTime(&gemm_ms, start, stop);
    double gemm_flops = 2.0 * G_M * G_N * G_K;
    printf("[GEMM]    time=%.3f ms  TFLOPS=%.2f  (naive, no tiling)\n",
           gemm_ms, gemm_flops / gemm_ms / 1e9);

    printf("\n--- ncu 分析指引 ---\n");
    printf("ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  smsp__average_warps_issue_stalled_long_scoreboard.pct,\\\n");
    printf("  gpu__time_duration.sum \\\n");
    printf("  --kernel-name regex:\"softmax_kernel|gemm_kernel\" ./profiling_targets\n\n");
    printf("预期：softmax_kernel -> DRAM%% >> SM%% (memory-bound)\n");
    printf("      gemm_kernel    -> SM%%   >> DRAM%% (compute-bound)\n");

    free(h_sm_in); free(h_sm_out); free(h_sm_ref); cudaFree(d_sm_in); cudaFree(d_sm_out);
    free(h_A); free(h_B); free(h_C); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
