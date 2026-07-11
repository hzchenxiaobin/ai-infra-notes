// softmax_layernorm_opt.cu —— 优化版 Softmax + LayerNorm（warp 级 + float4 向量化）
// 对比 Day 16 的 block 级 + 逐元素加载版本，验证工业级优化手法
// 编译命令: nvcc -o softmax_layernorm_opt softmax_layernorm_opt.cu -O3 -arch=sm_120 -lineinfo
// 运行命令: ./softmax_layernorm_opt

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// Warp Shuffle 原语（复用 Week 2 Day 1）
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

// Block 级 reduce（Day 16 版本，用于对比基准）
__inline__ __device__ float blockReduceSum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__inline__ __device__ float blockReduceMax(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// ============================================================
// 优化 1：Warp 级 Softmax（参考 PyTorch softmax_warp_forward）
// 一个 warp 处理一行（D ≤ 1024），无需 shared memory 和 __syncthreads
// ============================================================
__global__ void softmax_warp_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int M, int D) {
    // 每个 warp 处理一行，warp 数 = M * (blockDim.x / 32)
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (global_warp_id >= M) return;
    int lane = threadIdx.x % 32;

    const float* in_row = input + global_warp_id * D;
    float* out_row = output + global_warp_id * D;

    // 每个 lane 处理 D/32 个元素（D=1024 时每 lane 32 个）
    float local_max = -INFINITY;
    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    local_max = warpReduceMax(local_max);
    // warp 内 shuffle 后所有 lane 都有 local_max（用 __shfl_sync 广播）
    local_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        local_sum += expf(in_row[i] - local_max);
    }
    local_sum = warpReduceSum(local_sum);
    local_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);

    float inv_sum = 1.0f / local_sum;
    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        out_row[i] = expf(in_row[i] - local_max) * inv_sum;
    }
}

// ============================================================
// 基准：Day 16 的 block 级 Softmax（用于对比）
// ============================================================
__global__ void softmax_block_kernel(const float* __restrict__ input,
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
// 优化 2：float4 向量化 LayerNorm（参考 FasterTransformer 向量化加载）
// 一次加载 4 个 float（128-bit），减少 4x 加载指令
// ============================================================
__global__ void layernorm_float4_kernel(const float* __restrict__ input,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ output,
                                         int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    // float4 指针：把 float* 按 4 个一组 reinterpret
    const float4* in4 = reinterpret_cast<const float4*>(input + row * N);
    const float4* g4  = reinterpret_cast<const float4*>(gamma);
    const float4* b4  = reinterpret_cast<const float4*>(beta);
    float4* out4      = reinterpret_cast<float4*>(output + row * N);

    int N4 = N / 4;   // float4 元素数
    int tid = threadIdx.x;

    __shared__ float smem[32];
    __shared__ float row_mean, row_rstd;

    // Step 1: 求 mean（用 float4 批量加载累加）
    float local_sum = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 v = in4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_mean = local_sum / N;
    __syncthreads();

    // Step 2: 求 variance（同样 float4 批量加载）
    float local_sq = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 v = in4[i];
        float dx = v.x - row_mean;
        float dy = v.y - row_mean;
        float dz = v.z - row_mean;
        float dw = v.w - row_mean;
        local_sq += dx * dx + dy * dy + dz * dz + dw * dw;
    }
    local_sq = blockReduceSum(local_sq, smem);
    if (tid == 0) row_rstd = rsqrtf(local_sq / N + eps);
    __syncthreads();

    // Step 3: 归一化 + affine（float4 批量写出）
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 v = in4[i];
        float4 g = g4[i];
        float4 b = b4[i];
        float4 r;
        r.x = (v.x - row_mean) * row_rstd * g.x + b.x;
        r.y = (v.y - row_mean) * row_rstd * g.y + b.y;
        r.z = (v.z - row_mean) * row_rstd * g.z + b.z;
        r.w = (v.w - row_mean) * row_rstd * g.w + b.w;
        out4[i] = r;
    }
}

// ============================================================
// 基准：Day 16 的逐元素 LayerNorm（用于对比）
// ============================================================
__global__ void layernorm_scalar_kernel(const float* __restrict__ input,
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
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += in_row[i];
    }
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
        for (int i = 0; i < D; i++) { orow[i] = expf(ir[i] - mx); s += orow[i]; }
        for (int i = 0; i < D; i++) {
            orow[i] /= s;
        }
    }
}

void cpuLayerNorm(const float* in, const float* gamma, const float* beta,
                  float* out, int M, int N, float eps) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * N;
        float* orow = out + r * N;
        float mean = 0.0f;
        for (int i = 0; i < N; i++) {
            mean += ir[i];
        }
        mean /= N;
        float var = 0.0f;
        for (int i = 0; i < N; i++) { float d = ir[i] - mean; var += d * d; }
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
        maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    }
    bool ok = maxDiff < eps;
    printf("%s: maxDiff = %.2e (%s)\n", name, maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

// 计时辅助
float timeKernel(cudaEvent_t& s, cudaEvent_t& e) {
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

int main() {
    const int M = 1024;      // 行数（增大 M 让 warp 级有足够并行度）
    const int D = 1024;      // 特征维（warp 级 softmax 要求 D ≤ 1024）
    const float eps = 1e-5f;
    const int threads_block = 256;
    const int threads_warp  = 128;   // 4 个 warp per block

    printf("=== Softmax + LayerNorm Optimization Comparison ===\n");
    printf("Config: M=%d, D=%d (D must be multiple of 4 for float4)\n\n", M, D);

    size_t bytes = (size_t)M * D * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
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
    int iters = 50;

    // ---- Softmax 对比 ----
    printf("[Softmax: block-level (Day16) vs warp-level (optimized)]\n");

    // warmup + bench block
    for (int i = 0; i < 3; i++) {
        softmax_block_kernel<<<M, threads_block>>>(d_in, d_out, M, D);
    }
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        softmax_block_kernel<<<M, threads_block>>>(d_in, d_out, M, D);
    }
    cudaEventRecord(stop);
    float ms_block = timeKernel(start, stop) / iters;

    // warmup + bench warp
    int warps_per_block = threads_warp / 32;
    int grid_warp = (M + warps_per_block - 1) / warps_per_block;
    for (int i = 0; i < 3; i++) {
        softmax_warp_kernel<<<grid_warp, threads_warp>>>(d_in, d_out, M, D);
    }
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        softmax_warp_kernel<<<grid_warp, threads_warp>>>(d_in, d_out, M, D);
    }
    cudaEventRecord(stop);
    float ms_warp = timeKernel(start, stop) / iters;

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuSoftmax(h_in, h_ref, M, D);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  warp-level correctness");
    printf("  block-level (Day16): %.4f ms\n", ms_block);
    printf("  warp-level (optim) : %.4f ms\n", ms_warp);
    printf("  speedup            : %.2fx\n\n", ms_block / ms_warp);

    // ---- LayerNorm 对比 ----
    printf("[LayerNorm: scalar load (Day16) vs float4 vectorized]\n");

    for (int i = 0; i < 3; i++) {
        layernorm_scalar_kernel<<<M, threads_block>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
    }
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        layernorm_scalar_kernel<<<M, threads_block>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
    }
    cudaEventRecord(stop);
    float ms_scalar = timeKernel(start, stop) / iters;

    for (int i = 0; i < 3; i++) {
        layernorm_float4_kernel<<<M, threads_block>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
    }
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        layernorm_float4_kernel<<<M, threads_block>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
    }
    cudaEventRecord(stop);
    float ms_f4 = timeKernel(start, stop) / iters;

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuLayerNorm(h_in, h_gamma, h_beta, h_ref, M, D, eps);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  float4 correctness");
    printf("  scalar (Day16) : %.4f ms\n", ms_scalar);
    printf("  float4 (optim) : %.4f ms\n", ms_f4);
    printf("  speedup        : %.2fx\n\n", ms_scalar / ms_f4);

    printf("=== ncu 验证命令 ===\n");
    printf("ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("  gpu__time_duration.sum \\\n");
    printf("  --kernel-name regex:\"softmax_warp_kernel|layernorm_float4_kernel\" \\\n");
    printf("  ./softmax_layernorm_opt\n");

    free(h_in); free(h_out); free(h_ref); free(h_gamma); free(h_beta);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
