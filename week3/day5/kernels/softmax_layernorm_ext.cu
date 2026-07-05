// softmax_layernorm_ext.cu —— 自定义 Softmax/LayerNorm（含 launch wrapper + PyTorch C++ Extension 绑定）
// 编译命令（独立）: nvcc -o softmax_layernorm_ext kernels/softmax_layernorm_ext.cu -O3 -arch=sm_120
// 集成编译（PyTorch load_inline）: 见 mini_engine.py
// 运行命令: ./softmax_layernorm_ext

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// 复用 Week 2 Day 1 / Day 16 的 Warp Shuffle 原语
// ============================================================
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

// ============================================================
// Softmax Kernel：一行一个 block，三遍扫描 safe softmax（Day 16 实现）
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

// ============================================================
// LayerNorm Kernel：一行一个 block，两次 reduce（Day 16 实现）
// ============================================================
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

// ============================================================
// Launch Wrappers：供 C++ Extension 和独立 main 共用
// 封装 grid/block 配置 + stream 传递
// ============================================================
void launch_softmax(const float* input, float* output, int M, int D, cudaStream_t stream) {
    int threads = 256;
    softmax_kernel<<<M, threads, 0, stream>>>(input, output, M, D);
}

void launch_layernorm(const float* input, const float* gamma, const float* beta,
                      float* output, int M, int N, float eps, cudaStream_t stream) {
    int threads = 256;
    layernorm_kernel<<<M, threads, 0, stream>>>(input, gamma, beta, output, M, N, eps);
}

// ============================================================
// 独立验证 main（不依赖 PyTorch，可直接 nvcc 编译）
// ============================================================
void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
}

void cpuSoftmax(const float* in, float* out, int M, int D) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * D;
        float* orow = out + r * D;
        float mx = ir[0];
        for (int i = 1; i < D; i++) mx = fmaxf(mx, ir[i]);
        float s = 0.0f;
        for (int i = 0; i < D; i++) { orow[i] = expf(ir[i] - mx); s += orow[i]; }
        for (int i = 0; i < D; i++) orow[i] /= s;
    }
}

void cpuLayerNorm(const float* in, const float* gamma, const float* beta,
                  float* out, int M, int N, float eps) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * N;
        float* orow = out + r * N;
        float mean = 0.0f;
        for (int i = 0; i < N; i++) mean += ir[i];
        mean /= N;
        float var = 0.0f;
        for (int i = 0; i < N; i++) { float d = ir[i] - mean; var += d * d; }
        var /= N;
        float rstd = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < N; i++)
            orow[i] = (ir[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

bool checkResult(const float* a, const float* b, int n, float eps, const char* name) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    bool ok = maxDiff < eps;
    printf("%s: maxDiff = %.2e (%s)\n", name, maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const int M = 128, D = 1024;
    const float eps = 1e-5f;
    size_t bytes = (size_t)M * D * sizeof(float);

    printf("=== Softmax + LayerNorm (ext version, launch wrappers) ===\n");
    printf("Config: M=%d, D=%d\n\n", M, D);

    float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(D * sizeof(float)), *h_beta = (float*)malloc(D * sizeof(float));
    initData(h_in, M * D);
    for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_gamma, D * sizeof(float)); cudaMalloc(&d_beta, D * sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice);

    // 通过 launch wrapper 调用（与 C++ Extension 路径一致）
    printf("[Softmax]\n");
    launch_softmax(d_in, d_out, M, D, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuSoftmax(h_in, h_ref, M, D);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  Softmax vs CPU");

    printf("[LayerNorm]\n");
    launch_layernorm(d_in, d_gamma, d_beta, d_out, M, D, eps, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuLayerNorm(h_in, h_gamma, h_beta, h_ref, M, D, eps);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  LayerNorm vs CPU");

    printf("\n这两个 launch wrapper 就是 PyTorch C++ Extension 要调用的入口。\n");
    printf("集成方式见 mini_engine.py 的 load_inline 调用。\n");

    free(h_in); free(h_out); free(h_ref); free(h_gamma); free(h_beta);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    return 0;
}

// ============================================================
// PyTorch C++ Extension 绑定（仅 load_inline 编译时启用）
// 用 __TORCH_EXTENSION 宏区分独立编译和 PyTorch 集成
// ============================================================
#ifdef WITH_TORCH
#include <torch/extension.h>

at::Tensor softmax_forward(at::Tensor input) {
    int M = input.size(0), D = input.size(1);
    auto output = at::empty_like(input);
    launch_softmax(input.data_ptr<float>(), output.data_ptr<float>(),
                   M, D, at::cuda::getCurrentCUDAStream());
    return output;
}

at::Tensor layernorm_forward(at::Tensor input, at::Tensor gamma, at::Tensor beta, double eps) {
    int M = input.size(0), N = input.size(1);
    auto output = at::empty_like(input);
    launch_layernorm(input.data_ptr<float>(), gamma.data_ptr<float>(),
                     beta.data_ptr<float>(), output.data_ptr<float>(),
                     M, N, (float)eps, at::cuda::getCurrentCUDAStream());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_forward", &softmax_forward, "Softmax forward (CUDA)");
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
}
#endif
