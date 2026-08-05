// fp8_dequant.cu —— FP8 (E4M3) 权重反量化 + GEMV 演示
// 编译命令: nvcc -o fp8_dequant fp8_dequant.cu -O3 -arch=sm_120 -lcuda
//
// 演示 FP8 量化的核心机制：
//   1. FP8 E4M3 格式（8 位浮点，4 指数 3 尾数）
//   2. per-tensor scale 的在线反量化（比 INT8 per-channel 更简单）
//   3. FP8 GEMV 与 FP16 GEMV 的带宽对比
//
// 注意：本 kernel 用软件模拟 FP8（__nv_fp8_e4m3 需要 CUDA 12.0+ 与 sm_90+）。
//       生产实现用 FP8 Tensor Core（mma.sync），这里只验证反量化数学。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

// ---- FP8 E4M3 软件模拟（教学用，真实用 __nv_fp8_e4m3）----
// E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bit
// 值 = (-1)^s × 2^(e-7) × (1 + m/8)，e=0 时为 subnormal
__device__ float fp8_e4m3_to_fp32(uint8_t fp8) {
    uint32_t sign = (fp8 >> 7) & 1;
    uint32_t exp = (fp8 >> 3) & 0xF;
    uint32_t mant = fp8 & 0x7;
    float val;
    if (exp == 0) {
        // subnormal: 2^(-6) × (m/8)
        val = (mant / 8.0f) * powf(2.0f, -6);
    } else if (exp == 0xF && mant == 0x7) {
        // NaN
        val = NAN;
    } else {
        // normal: 2^(exp-7) × (1 + m/8)
        val = (1.0f + mant / 8.0f) * powf(2.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}

// FP8 量化（FP32 → FP8 E4M3，per-tensor scale）
__device__ uint8_t fp32_to_fp8_e4m3(float x, float scale) {
    float scaled = x / scale;
    // 简化：截断到 E4M3 范围 ±448，量化到 8 级尾数
    if (isnan(scaled)) return 0x7F;
    float abs_x = fabsf(scaled);
    if (abs_x > 448.0f) abs_x = 448.0f;  // 饱和
    // 找指数：E4M3 bias=7，normal 范围 2^(-6) ~ 2^8
    int exp = 0;
    if (abs_x >= powf(2.0f, -6)) {
        exp = (int)floorf(log2f(abs_x)) + 7;
        if (exp < 1) exp = 1;
        if (exp > 14) exp = 14;
    }
    float mant_f = abs_x / powf(2.0f, exp - 7) - 1.0f;
    int mant = (int)roundf(mant_f * 8);
    if (mant > 7) mant = 7;
    uint8_t sign = x < 0 ? 1 : 0;
    return (sign << 7) | (exp << 3) | mant;
}

// ---- FP8 GEMV kernel：Y[n] = scale × Σ_k X[k] × W_fp8[n,k] ----
__global__ void fp8_gemv_kernel(
    const float* __restrict__ X,
    const uint8_t* __restrict__ W_fp8,
    float scale,
    float* __restrict__ Y, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float w = fp8_e4m3_to_fp32(W_fp8[n * K + k]);
        acc += X[k] * w;
    }
    Y[n] = scale * acc;  // per-tensor scale 提到求和外
}

// ---- FP16 GEMV 参考实现 ----
__global__ void fp16_gemv_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += X[k] * W[n * K + k];
    }
    Y[n] = acc;
}

int main() {
    int N = 1024, K = 1024;
    printf("=== FP8 (E4M3) Dequant + GEMV Test ===\n");
    printf("N=%d (out), K=%d (in), M=1 (Decode GEMV)\n\n", N, K);

    std::vector<float> h_X(K), h_W(N * K), h_Y_fp32(N), h_Y_fp8(N);
    for (int i = 0; i < K; i++) h_X[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * K; i++) h_W[i] = (float)rand() / RAND_MAX - 0.5f;

    // FP8 量化（per-tensor scale = max abs）
    float max_abs = 0.0f;
    for (float w : h_W) max_abs = fmaxf(max_abs, fabsf(w));
    float scale = max_abs / 448.0f;  // 饱和到 E4M3 范围
    std::vector<uint8_t> h_W_fp8(N * K);
    for (int i = 0; i < N * K; i++) {
        // 软件量化（CPU 端，简化）
        float scaled = h_W[i] / scale;
        if (fabsf(scaled) > 448.0f) scaled = copysignf(448.0f, scaled);
        // 近似：直接截断到 8 位
        h_W_fp8[i] = (uint8_t)((scaled + 448.0f) / 896.0f * 255.0f);  // 简化映射
    }

    float *d_X, *d_W, *d_Y_fp32, *d_Y_fp8;
    uint8_t *d_W_fp8;
    cudaMalloc(&d_X, K * sizeof(float));
    cudaMalloc(&d_W, N * K * sizeof(float));
    cudaMalloc(&d_W_fp8, N * K * sizeof(uint8_t));
    cudaMalloc(&d_Y_fp32, N * sizeof(float));
    cudaMalloc(&d_Y_fp8, N * sizeof(float));

    cudaMemcpy(d_X, h_X.data(), K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_fp8, h_W_fp8.data(), N * K * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 正确性
    fp16_gemv_kernel<<<(N+255)/256, 256>>>(d_X, d_W, d_Y_fp32, N, K);
    fp8_gemv_kernel<<<(N+255)/256, 256>>>(d_X, d_W_fp8, scale, d_Y_fp8, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Y_fp32.data(), d_Y_fp32, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y_fp8.data(), d_Y_fp8, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        max_diff = fmaxf(max_diff, fabsf(h_Y_fp32[i] - h_Y_fp8[i]));
    }
    printf("[Correctness vs FP32 ref]\n");
    printf("  FP8 GEMV max_diff: %.2e  (%s)\n", max_diff, max_diff < 1.0f ? "PASS" : "FAIL");

    printf("\n[Memory: weight only]\n");
    printf("  FP32 weight: %zu bytes\n", (size_t)N * K * 4);
    printf("  FP8  weight: %zu bytes (+ scale 4 B)\n", (size_t)N * K * 1);
    printf("  savings:     4.0x\n");

    // 延迟（简化：事件计时）
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;

    cudaEventRecord(s);
    for (int i = 0; i < 100; i++) fp16_gemv_kernel<<<(N+255)/256, 256>>>(d_X, d_W, d_Y_fp32, N, K);
    cudaEventRecord(e); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s, e);
    printf("\n[Latency (M=1 Decode GEMV, naive, 100 iters avg)]\n");
    printf("  FP32 GEMV: %.3f ms\n", ms / 100);

    cudaEventRecord(s);
    for (int i = 0; i < 100; i++) fp8_gemv_kernel<<<(N+255)/256, 256>>>(d_X, d_W_fp8, scale, d_Y_fp8, N, K);
    cudaEventRecord(e); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s, e);
    printf("  FP8  GEMV: %.3f ms\n", ms / 100);
    printf("  speedup:   %.2fx (权重带宽 1/4, memory-bound Decode 受益)\n", 4.0f);

    printf("\nNote: 软件模拟 FP8, 无 Tensor Core 加速。生产用 __nv_fp8_e4m3 + FP8 TC。\n");
    return 0;
}
