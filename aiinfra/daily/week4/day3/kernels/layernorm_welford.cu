// layernorm_welford.cu —— Day 3: Welford 单 pass LayerNorm（对比 Day 2 三遍扫描基线）
// 一行一个 block：
//   - layernorm_threepass_kernel: mean / variance / normalize 三遍读 x（Day 2 基线，HBM 3 读 1 写）
//   - layernorm_welford_kernel:   Welford 在线算法一次遍历同时求 mean/var，再一遍归一化写出
//                                 （HBM 1 读 1 写，归一化遍命中 L2/L1 缓存）
// 正确性: 与 CPU 参考对比；性能: cudaEvent 计时（预热 10 次 + 计时 100 次）
//
// 编译命令: nvcc -O3 -arch=sm_120 kernels/layernorm_welford.cu -o layernorm_welford
// 运行命令: ./layernorm_welford
//
// ⚠️ 本机无 GPU，未编译实测——README 中"预期输出"表为量级预估，实测数据待 GPU 环境回填。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// 复用 Day 2 的 Warp Shuffle 原语（三遍扫描基线的 block reduce 用）
// ============================================================
__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block 级 reduce：warp 级 → shared memory → warp 0 最终 reduce
// 返回后只有 warp 0 的线程持有正确结果，调用方需用 shared 变量广播
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

// ============================================================
// Welford 在线算法原语（与 week10/_supplementary/from_w4d1/kernels/softmax_layernorm_opt.cu 同一套写法）
// 一次遍历同时维护 (count, mean, M2)，最终 var = M2 / count
// ============================================================
struct WelfordData {
    float mean;
    float m2;
    int count;
};

// 单元素在线更新：delta = x - mean_old; mean += delta / count; M2 += delta * (x - mean_new)
__inline__ __device__ WelfordData welfordUpdate(WelfordData w, float x) {
    w.count += 1;
    float delta = x - w.mean;
    w.mean += delta / (float)w.count;
    float delta2 = x - w.mean;
    w.m2 += delta * delta2;
    return w;
}

// 并行合并两个统计块，按 count 加权：
// delta = mean_b - mean_a; mean = mean_a + delta * n_b / n; M2 = M2_a + M2_b + delta^2 * n_a * n_b / n
__inline__ __device__ WelfordData welfordMerge(WelfordData a, WelfordData b) {
    if (a.count == 0)
        return b;
    if (b.count == 0)
        return a;
    float delta = b.mean - a.mean;
    int n = a.count + b.count;
    WelfordData r;
    r.mean = a.mean + delta * (float)b.count / (float)n;
    r.m2 = a.m2 + b.m2 + delta * delta * (float)a.count * (float)b.count / (float)n;
    r.count = n;
    return r;
}

// warp 内折半合并：shuffle 3 个字段（mean / m2 / count）
__inline__ __device__ WelfordData warpReduceWelford(WelfordData w) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordData other;
        other.mean = __shfl_down_sync(0xFFFFFFFF, w.mean, offset);
        other.m2 = __shfl_down_sync(0xFFFFFFFF, w.m2, offset);
        other.count = __shfl_down_sync(0xFFFFFFFF, w.count, offset);
        w = welfordMerge(w, other);
    }
    return w;
}

// block 级合并：warp 内 shuffle + 每 warp 结果写 shared memory，再由 warp 0 合并
// 返回值只在 warp 0 的 lane 0 有效
__inline__ __device__ WelfordData blockReduceWelford(WelfordData w, WelfordData* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    w = warpReduceWelford(w);
    if (lane == 0)
        smem[wid] = w;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    if (wid == 0) {
        w = (lane < numWarps) ? smem[lane] : WelfordData{0.0f, 0.0f, 0};
        w = warpReduceWelford(w);
    }
    return w;
}

// ============================================================
// 基线：Day 2 三遍扫描 LayerNorm（mean → variance → normalize，3 次读 x）
// 输入: input[M][N]，参数: gamma[N], beta[N]，输出: output[M][N]
// ============================================================
__global__ void layernorm_threepass_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                                           const float* __restrict__ beta, float* __restrict__ output, int M, int N,
                                           float eps) {
    int row = blockIdx.x;
    if (row >= M)
        return;
    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    __shared__ float smem[32];
    __shared__ float row_mean;
    __shared__ float row_rstd;

    int tid = threadIdx.x;

    // Pass 1: 求 mean = sum(x) / N
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += in_row[i];
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0)
        row_mean = local_sum / N;
    __syncthreads();

    // Pass 2: 求 variance = sum((x - mean)^2) / N，rstd = 1/sqrt(var + eps)
    float local_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = in_row[i] - row_mean;
        local_sq += diff * diff;
    }
    local_sq = blockReduceSum(local_sq, smem);
    if (tid == 0)
        row_rstd = rsqrtf(local_sq / N + eps);
    __syncthreads();

    // Pass 3: 归一化 + affine: y = (x - mean) * rstd * gamma + beta
    for (int i = tid; i < N; i += blockDim.x) {
        out_row[i] = (in_row[i] - row_mean) * row_rstd * gamma[i] + beta[i];
    }
}

// ============================================================
// Welford 单 pass LayerNorm（README §3.2 的完整版，补齐跨 warp 合并）
// Pass 1 一次遍历同时求 mean/var（Welford 递推 + shuffle/smem 两级合并）
// Pass 2 归一化写出（这一遍读命中缓存；HBM 层面只读一遍 x）
// ============================================================
__global__ void layernorm_welford_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                                         const float* __restrict__ beta, float* __restrict__ output, int M, int N,
                                         float eps) {
    int row = blockIdx.x;
    if (row >= M)
        return;
    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    __shared__ WelfordData wsmem[32];
    __shared__ float row_mean, row_rstd;

    int tid = threadIdx.x;

    // 1. 每线程做局部 Welford（处理 x[row][tid], x[row][tid+blockDim.x], ...）
    WelfordData w = {0.0f, 0.0f, 0};
    for (int i = tid; i < N; i += blockDim.x) {
        w = welfordUpdate(w, in_row[i]);
    }

    // 2. warp 内 __shfl_down_sync 合并 + 3. shared memory 跨 warp 合并（结果在 warp 0 的 lane 0）
    w = blockReduceWelford(w, wsmem);

    // 4. 最终 mean / rstd 广播到所有线程
    if (tid == 0) {
        row_mean = w.mean;
        row_rstd = rsqrtf(w.m2 / N + eps);
    }
    __syncthreads();

    // 5. 归一化 + affine（读 x 一遍、写 y 一遍）
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

// gamma 初始化在 1 附近、beta 在 0 附近（比全 1/全 0 更强的正确性检验）
void initParams(float* gamma, float* beta, int n) {
    srand(123);
    for (int i = 0; i < n; i++) {
        gamma[i] = 1.0f + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        beta[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
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

float maxDiff(const float* a, const float* b, int n) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > md)
            md = diff;
    }
    return md;
}

// which: 0 = three-pass, 1 = welford
static void launchKernel(int which, const float* in, const float* g, const float* b, float* out, int M, int N,
                         float eps, int threads) {
    if (which == 0)
        layernorm_threepass_kernel<<<M, threads>>>(in, g, b, out, M, N, eps);
    else
        layernorm_welford_kernel<<<M, threads>>>(in, g, b, out, M, N, eps);
}

// cudaEvent 计时：预热 warmup 次，再计 iters 次的平均耗时（ms）
static float benchMs(int which, const float* in, const float* g, const float* b, float* out, int M, int N, float eps,
                     int threads, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        launchKernel(which, in, g, b, out, M, N, eps, threads);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launchKernel(which, in, g, b, out, M, N, eps, threads);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main() {
    const int D = 1024;       // 特征维
    const float eps = 1e-5f;
    const int threads = 256;
    const int warmup = 10;
    const int iters = 100;
    const int rows_list[] = {1024, 4096, 16384};
    const int n_cases = sizeof(rows_list) / sizeof(rows_list[0]);

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    printf("=== LayerNorm: Three-pass vs Welford (%s, D=%d) ===\n", prop.name, D);
    printf("M(rows)  | Three-pass(ms)  Welford(ms)  | Speedup  max_diff\n");
    printf("---------|--------------------------------|----------------------\n");

    float worst_cpu_diff = 0.0f;
    for (int c = 0; c < n_cases; c++) {
        int M = rows_list[c];
        size_t bytes = (size_t)M * D * sizeof(float);

        float* h_in = (float*)malloc(bytes);
        float* h_out3 = (float*)malloc(bytes);
        float* h_outw = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        float* h_gamma = (float*)malloc(D * sizeof(float));
        float* h_beta = (float*)malloc(D * sizeof(float));
        initData(h_in, M * D);
        initParams(h_gamma, h_beta, D);

        float *d_in, *d_out, *d_gamma, *d_beta;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cudaMalloc(&d_gamma, D * sizeof(float));
        cudaMalloc(&d_beta, D * sizeof(float));
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice);

        // Three-pass 基线
        float ms3 = benchMs(0, d_in, d_gamma, d_beta, d_out, M, D, eps, threads, warmup, iters);
        cudaMemcpy(h_out3, d_out, bytes, cudaMemcpyDeviceToHost);

        // Welford 单 pass
        float msw = benchMs(1, d_in, d_gamma, d_beta, d_out, M, D, eps, threads, warmup, iters);
        cudaMemcpy(h_outw, d_out, bytes, cudaMemcpyDeviceToHost);

        // 正确性：两种 GPU 实现互相对比 + Welford 对比 CPU 参考
        cpuLayerNorm(h_in, h_gamma, h_beta, h_ref, M, D, eps);
        float diff_gpu = maxDiff(h_out3, h_outw, M * D);
        float diff_cpu = maxDiff(h_outw, h_ref, M * D);
        if (diff_cpu > worst_cpu_diff)
            worst_cpu_diff = diff_cpu;

        printf("%-9d| %-15.3f %-15.3f | %-7.2fx %.2e\n", M, ms3, msw, ms3 / msw, diff_gpu);

        free(h_in);
        free(h_out3);
        free(h_outw);
        free(h_ref);
        free(h_gamma);
        free(h_beta);
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_gamma);
        cudaFree(d_beta);
    }

    printf("\n[正确性] Welford vs CPU 参考: max_diff = %.2e (< 1e-4 %s)\n", worst_cpu_diff,
           worst_cpu_diff < 1e-4f ? "PASS" : "FAIL");
    printf("[说明] 表中 max_diff 为两种 GPU 实现之间的差异，< 1e-5 为一致（README 口径）。\n");
    printf("[说明] Welford 预期 ~2x 加速（HBM 读写 4x2D -> 2x2D）；HBM 访问量用 ncu 验证（README 任务 2）。\n");
    return 0;
}
