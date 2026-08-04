// w8a16_dequant.cu —— W8A16 Weight-only 量化：INT8 权重 + FP16 激活，GEMM 内在线反量化
// 编译命令: nvcc -o w8a16_dequant w8a16_dequant.cu -O3 -arch=sm_120
// 运行命令: ./w8a16_dequant
//
// 演示 W8A16 量化的三大要点：
//   1. 权重 W 以 INT8 存储（per-channel scale），激活 X 保持 FP16
//   2. GEMM 内在线反量化：Y[m,n] = scale[n] * Σ_k X[m,k] * W_int8[n,k]
//      —— per-channel scale 可提到求和外面，只需一次 INT8 点积（FP32 累加）+ 一次乘 scale
//   3. 对比 FP16 GEMM：权重显存减半，memory-bound 的 Decode 阶段带宽收益明显

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

// ---------- W8A16 GEMV kernel（M=1, Decode 场景） ----------
// X:      [K]       FP16 激活
// W_int8: [N, K]    INT8 权重（row-major: N=out_features, K=in_features）
// scale:  [N]       FP16 per-channel scale（每行一个）
// Y:      [N]       FP16 输出
// 数学: Y[n] = scale[n] * Σ_k X[k] * W_int8[n,k]
//      —— 先算 INT8 点积（FP32 累加），最后乘 scale[n]，省掉逐元素反量化
__global__ void w8a16_gemv_kernel(
    const __half* __restrict__ X,
    const int8_t* __restrict__ W_int8,
    const __half* __restrict__ scale,
    __half* __restrict__ Y,
    int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(X[k]) * (float)W_int8[n * K + k];
    Y[n] = __float2half(acc * __half2float(scale[n]));
}

// ---------- FP16 GEMV kernel（基线，权重存 FP16） ----------
__global__ void fp16_gemv_kernel(
    const __half* __restrict__ X,
    const __half* __restrict__ W,
    __half* __restrict__ Y,
    int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(X[k]) * __half2float(W[n * K + k]);
    Y[n] = __float2half(acc);
}

// ---------- per-channel 对称量化：scale[n] = max|W[n,:]| / 127 ----------
void quantize_w8a16(const std::vector<float>& W, std::vector<int8_t>& W_int8,
                    std::vector<float>& scale, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float mx = 0.f;
        for (int k = 0; k < K; ++k) mx = fmaxf(mx, fabsf(W[n * K + k]));
        float s = mx / 127.f;
        scale[n] = s;
        for (int k = 0; k < K; ++k) {
            int q = (int)lroundf(W[n * K + k] / s);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            W_int8[n * K + k] = (int8_t)q;
        }
    }
}

// ---------- CPU 参考（FP32 GEMV） ----------
void cpu_gemv(const std::vector<float>& X, const std::vector<float>& W,
              std::vector<float>& Y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += X[k] * W[n * K + k];
        Y[n] = acc;
    }
}

int main() {
    const int N = 1024, K = 1024;
    printf("=== W8A16 Weight-only Dequant Test ===\n");
    printf("N=%d (out), K=%d (in), M=1 (Decode GEMV)\n\n", N, K);

    std::vector<float> h_X(K), h_W(N * K);
    for (int i = 0; i < K; ++i) h_X[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < N * K; ++i) h_W[i] = ((float)(rand() % 200) - 100) / 100.0f;

    std::vector<int8_t> h_W_int8(N * K);
    std::vector<float> h_scale(N);
    quantize_w8a16(h_W, h_W_int8, h_scale, N, K);

    std::vector<__half> h_X16(K), h_W16(N * K), h_scale16(N);
    for (int i = 0; i < K; ++i) h_X16[i] = __float2half(h_X[i]);
    for (int i = 0; i < N * K; ++i) h_W16[i] = __float2half(h_W[i]);
    for (int i = 0; i < N; ++i) h_scale16[i] = __float2half(h_scale[i]);

    __half *d_X, *d_W, *d_Y_fp16, *d_Y_w8a16, *d_scale;
    int8_t *d_W_int8;
    cudaMalloc(&d_X, K * sizeof(__half));
    cudaMalloc(&d_W, (size_t)N * K * sizeof(__half));
    cudaMalloc(&d_Y_fp16, N * sizeof(__half));
    cudaMalloc(&d_Y_w8a16, N * sizeof(__half));
    cudaMalloc(&d_scale, N * sizeof(__half));
    cudaMalloc(&d_W_int8, (size_t)N * K * sizeof(int8_t));
    cudaMemcpy(d_X, h_X16.data(), K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W16.data(), (size_t)N * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale16.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_int8, h_W_int8.data(), (size_t)N * K, cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float t_fp16 = 0, t_w8a16 = 0;

    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(s);
        fp16_gemv_kernel<<<blocks, 256>>>(d_X, d_W, d_Y_fp16, N, K);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&t_fp16, s, e);
    }
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(s);
        w8a16_gemv_kernel<<<blocks, 256>>>(d_X, d_W_int8, d_scale, d_Y_w8a16, N, K);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&t_w8a16, s, e);
    }

    std::vector<__half> h_Y_fp16(N), h_Y_w8a16(N);
    cudaMemcpy(h_Y_fp16.data(), d_Y_fp16, N * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y_w8a16.data(), d_Y_w8a16, N * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> h_ref(N);
    cpu_gemv(h_X, h_W, h_ref, N, K);

    float max_diff_w8a16 = 0.f, max_diff_fp16 = 0.f;
    for (int n = 0; n < N; ++n) {
        max_diff_w8a16 = fmaxf(max_diff_w8a16, fabsf(__half2float(h_Y_w8a16[n]) - h_ref[n]));
        max_diff_fp16 = fmaxf(max_diff_fp16, fabsf(__half2float(h_Y_fp16[n]) - h_ref[n]));
    }

    printf("[Correctness vs FP32 CPU ref]\n");
    printf("  FP16 GEMV  max_diff: %.4e\n", max_diff_fp16);
    printf("  W8A16 GEMV max_diff: %.4e  (量化引入的误差)\n", max_diff_w8a16);
    printf("  result: %s\n\n", max_diff_w8a16 < 0.5f ? "PASS" : "FAIL");

    printf("[Memory: weight only]\n");
    printf("  FP16 weight: %zu bytes\n", (size_t)N * K * 2);
    printf("  INT8 weight: %zu bytes (+ scale %zu B)\n", (size_t)N * K, (size_t)N * 2);
    printf("  savings:     2.0x\n\n");

    printf("[Latency (M=1 Decode GEMV, naive kernel)]\n");
    printf("  FP16 GEMV:  %.3f ms\n", t_fp16);
    printf("  W8A16 GEMV: %.3f ms\n", t_w8a16);
    printf("  speedup:    %.2fx (权重带宽减半, memory-bound Decode 受益)\n", t_fp16 / t_w8a16);

    cudaFree(d_X); cudaFree(d_W); cudaFree(d_W_int8); cudaFree(d_scale);
    cudaFree(d_Y_fp16); cudaFree(d_Y_w8a16);
    return 0;
}
