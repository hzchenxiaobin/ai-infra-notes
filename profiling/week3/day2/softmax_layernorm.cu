// softmax_layernorm.cu —— Softmax + LayerNorm 完整实现（三遍扫描 + 两级 reduce）
// 编译命令: nvcc -o softmax_layernorm softmax_layernorm.cu -O3 -arch=sm_120
// 运行命令: ./softmax_layernorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// 复用 Week 2 Day 1 的 Warp Shuffle 原语
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

// ============================================================
// Block 级 reduce：warp 级 → shared memory → warp 0 最终 reduce
// smem 为外部传入的 shared memory 缓冲区（至少 32 个 float）
// 注意：返回后只有 warp 0 的线程持有正确结果，调用方需用 shared 变量广播
// ============================================================
__inline__ __device__ float blockReduceSum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0)
        smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : 0.0f;
    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}

__inline__ __device__ float blockReduceMax(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0)
        smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : -INFINITY;
    if (wid == 0)
        val = warpReduceMax(val);
    return val;
}

// ============================================================
// Softmax Kernel：一行一个 block，三遍扫描 safe softmax
// 输入: input[M][D]，输出: output[M][D]
// ============================================================
__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int D) {
    int row = blockIdx.x;
    if (row >= M)
        return;
    const float* in_row = input + row * D;
    float* out_row = output + row * D;

    __shared__ float smem[32]; // warp 间 reduce 缓冲区
    __shared__ float row_max;
    __shared__ float row_sum;

    int tid = threadIdx.x;

    // Step 1: 求 max（数值稳定性）
    float local_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    local_max = blockReduceMax(local_max, smem);
    if (tid == 0)
        row_max = local_max;
    __syncthreads();

    // Step 2: 求 sum(exp(x - max))
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_sum += expf(in_row[i] - row_max);
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0)
        row_sum = local_sum;
    __syncthreads();

    // Step 3: 归一化写出
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < D; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - row_max) * inv_sum;
    }
}

// ============================================================
// LayerNorm Kernel：一行一个 block，两次 reduce
// 输入: input[M][N]，参数: gamma[N], beta[N]，输出: output[M][N]
// ============================================================
__global__ void layernorm_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, float* __restrict__ output, int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M)
        return;
    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    __shared__ float smem[32];
    __shared__ float row_mean;
    __shared__ float row_rstd;

    int tid = threadIdx.x;

    // Step 1: 求 mean = sum(x) / N
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += in_row[i];
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0)
        row_mean = local_sum / N;
    __syncthreads();

    // Step 2: 求 variance = sum((x - mean)^2) / N，rstd = 1/sqrt(var + eps)
    float local_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = in_row[i] - row_mean;
        local_sq += diff * diff;
    }
    local_sq = blockReduceSum(local_sq, smem);
    if (tid == 0)
        row_rstd = rsqrtf(local_sq / N + eps);
    __syncthreads();

    // Step 3: 归一化 + affine: y = (x - mean) * rstd * gamma + beta
    for (int i = tid; i < N; i += blockDim.x) {
        out_row[i] = (in_row[i] - row_mean) * row_rstd * gamma[i] + beta[i];
    }
}

// ============================================================
// Host 辅助函数与验证
// ============================================================
void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
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
        for (int i = 0; i < D; i++) {
            orow[i] = expf(ir[i] - mx);
            s += orow[i];
        }
        for (int i = 0; i < D; i++) {
            orow[i] /= s;
        }
    }
}

void cpuLayerNorm(const float* in, const float* gamma, const float* beta, float* out, int M, int N, float eps) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * N;
        float* orow = out + r * N;
        float mean = 0.0f;
        for (int i = 0; i < N; i++) {
            mean += ir[i];
        }
        mean /= N;
        float var = 0.0f;
        for (int i = 0; i < N; i++) {
            float d = ir[i] - mean;
            var += d * d;
        }
        var /= N;
        float rstd = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < N; i++) {
            orow[i] = (ir[i] - mean) * rstd * gamma[i] + beta[i];
        }
    }
}

bool checkResult(const float* a, const float* b, int n, float eps, const char* name) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }
    bool ok = maxDiff < eps;
    printf("%s: maxDiff = %.2e (%s)\n", name, maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    // 测试配置
    const int M = 128;  // 行数（batch * seq_len）
    const int D = 1024; // 特征维（feature dim）
    const float eps = 1e-5f;
    const int threads = 256;

    printf("=== Softmax + LayerNorm Kernel Test ===\n");
    printf("Config: M=%d, D=%d, threads=%d\n\n", M, D, threads);

    size_t bytes = (size_t)M * D * sizeof(float);

    // Host 内存
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_gamma = (float*)malloc(D * sizeof(float));
    float* h_beta = (float*)malloc(D * sizeof(float));
    initData(h_in, M * D);
    for (int i = 0; i < D; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    // Device 内存
    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_gamma, D * sizeof(float));
    cudaMalloc(&d_beta, D * sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Softmax 测试 ----
    printf("[Softmax]\n");
    cudaEventRecord(start);
    softmax_kernel<<<M, threads>>>(d_in, d_out, M, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float smMs;
    cudaEventElapsedTime(&smMs, start, stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuSoftmax(h_in, h_ref, M, D);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  Softmax vs CPU");
    printf("  Time: %.3f ms\n", smMs);

    // ---- LayerNorm 测试 ----
    printf("[LayerNorm]\n");
    cudaEventRecord(start);
    layernorm_kernel<<<M, threads>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float lnMs;
    cudaEventElapsedTime(&lnMs, start, stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuLayerNorm(h_in, h_gamma, h_beta, h_ref, M, D, eps);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  LayerNorm vs CPU");
    printf("  Time: %.3f ms\n", lnMs);

    // 释放
    free(h_in);
    free(h_out);
    free(h_ref);
    free(h_gamma);
    free(h_beta);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
