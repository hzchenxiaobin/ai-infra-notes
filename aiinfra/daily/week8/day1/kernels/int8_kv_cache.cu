// int8_kv_cache.cu —— INT8 KV Cache 量化：per-token scale，attention 内在线反量化
// 编译命令: nvcc -o int8_kv_cache int8_kv_cache.cu -O3 -arch=sm_120
// 运行命令: ./int8_kv_cache
//
// 演示 INT8 KV Cache 的三大要点：
//   1. K/V 以 INT8 存储，per-token scale（每个 token 一个 scale，保留 token 内 outlier）
//   2. Attention 内在线反量化：score[s] = scale_k[s] * (Q · K_int8[s]) * attn_scale
//   3. 对比 FP16 KV Cache：KV 显存减半，长序列 Decode 带宽收益明显

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

#define THREADS 128
#define WARPS (THREADS / 32)
#define MAX_SEQ 4096
#define MAX_D 256

__inline__ __device__ float block_reduce_max(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < WARPS) ? sh[lane] : -INFINITY;
        for (int o = WARPS / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < WARPS) ? sh[lane] : 0.f;
        for (int o = WARPS / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- INT8 KV Cache Attention（Decode, M=1） ----------
// score[s] = scale_k[s] * (Q · K_int8[s]) * attn_scale  （scale 提到点积外）
// O[t]     = (1/l) * Σ_s softmax[s] * scale_v[s] * V_int8[s,t]
__global__ void int8_kv_attention_kernel(
    const __half* __restrict__ Q,
    const int8_t* __restrict__ K_int8, const float* __restrict__ scale_k,
    const int8_t* __restrict__ V_int8, const float* __restrict__ scale_v,
    __half* __restrict__ O, int seq, int d)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float red[WARPS + 1];
    __shared__ float q_shm[MAX_D];
    int tid = threadIdx.x;
    const float attn_scale = 1.0f / sqrtf((float)d);
    for (int t = tid; t < d; t += THREADS) q_shm[t] = __half2float(Q[t]);
    __syncthreads();

    for (int s = tid; s < seq; s += THREADS) {
        const int8_t* krow = K_int8 + (size_t)s * d;
        float dot = 0.f;
        for (int t = 0; t < d; ++t) dot += q_shm[t] * (float)krow[t];
        scores[s] = dot * scale_k[s] * attn_scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int s = tid; s < seq; s += THREADS) m = fmaxf(m, scores[s]);
    m = block_reduce_max(m, red);
    float l = 0.f;
    for (int s = tid; s < seq; s += THREADS) { scores[s] = expf(scores[s] - m); l += scores[s]; }
    l = block_reduce_sum(l, red);

    for (int t = tid; t < d; t += THREADS) {
        float acc = 0.f;
        for (int s = 0; s < seq; ++s) acc += scores[s] * scale_v[s] * (float)V_int8[(size_t)s * d + t];
        O[t] = __float2half(acc / l);
    }
}

// ---------- per-token 对称量化：scale[s] = max|KV[s,:]| / 127 ----------
void quantize_per_token(const std::vector<float>& KV, std::vector<int8_t>& KV_int8,
                        std::vector<float>& scale, int seq, int d) {
    for (int s = 0; s < seq; ++s) {
        float mx = 0.f;
        for (int t = 0; t < d; ++t) mx = fmaxf(mx, fabsf(KV[s * d + t]));
        float sc = mx / 127.f;
        scale[s] = sc;
        for (int t = 0; t < d; ++t) {
            int q = (sc > 0.f) ? (int)lroundf(KV[s * d + t] / sc) : 0;
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            KV_int8[s * d + t] = (int8_t)q;
        }
    }
}

// ---------- CPU 参考（FP32 attention, K/V 用 FP32 原值） ----------
void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K,
                   const std::vector<float>& V, std::vector<float>& O, int seq, int d) {
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(seq);
    float mx = -INFINITY;
    for (int s = 0; s < seq; ++s) {
        float dot = 0.f;
        for (int t = 0; t < d; ++t) dot += Q[t] * K[s * d + t];
        scores[s] = dot * scale;
        mx = fmaxf(mx, scores[s]);
    }
    float l = 0.f;
    for (int s = 0; s < seq; ++s) { scores[s] = expf(scores[s] - mx); l += scores[s]; }
    for (int t = 0; t < d; ++t) {
        O[t] = 0.f;
        for (int s = 0; s < seq; ++s) O[t] += scores[s] * V[s * d + t];
        O[t] /= l;
    }
}

int main() {
    const int seq = 1024, d = 64;
    printf("=== INT8 KV Cache Attention Test ===\n");
    printf("seq=%d, d=%d (Decode, M=1)\n\n", seq, d);

    std::vector<float> h_Q(d), h_K(seq * d), h_V(seq * d);
    for (int i = 0; i < d; ++i) h_Q[i] = ((float)(rand() % 200) - 100) / 100.0f;
    for (int i = 0; i < seq * d; ++i) {
        h_K[i] = ((float)(rand() % 200) - 100) / 100.0f;
        h_V[i] = ((float)(rand() % 200) - 100) / 100.0f;
    }
    std::vector<int8_t> h_K_int8(seq * d), h_V_int8(seq * d);
    std::vector<float> h_sk(seq), h_sv(seq);
    quantize_per_token(h_K, h_K_int8, h_sk, seq, d);
    quantize_per_token(h_V, h_V_int8, h_sv, seq, d);

    std::vector<__half> h_Q16(d);
    for (int i = 0; i < d; ++i) h_Q16[i] = __float2half(h_Q[i]);

    __half *d_Q, *d_O; int8_t *d_Ki, *d_Vi; float *d_sk, *d_sv;
    cudaMalloc(&d_Q, d * sizeof(__half));
    cudaMalloc(&d_Ki, (size_t)seq * d);
    cudaMalloc(&d_Vi, (size_t)seq * d);
    cudaMalloc(&d_sk, seq * sizeof(float));
    cudaMalloc(&d_sv, seq * sizeof(float));
    cudaMalloc(&d_O, d * sizeof(__half));
    cudaMemcpy(d_Q, h_Q16.data(), d * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ki, h_K_int8.data(), (size_t)seq * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vi, h_V_int8.data(), (size_t)seq * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, h_sk.data(), seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sv, h_sv.data(), seq * sizeof(float), cudaMemcpyHostToDevice);

    int8_kv_attention_kernel<<<1, THREADS>>>(d_Q, d_Ki, d_sk, d_Vi, d_sv, d_O, seq, d);
    cudaDeviceSynchronize();

    std::vector<__half> h_O16(d);
    cudaMemcpy(h_O16.data(), d_O, d * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> h_ref(d);
    cpu_attention(h_Q, h_K, h_V, h_ref, seq, d);

    float max_diff = 0.f;
    for (int t = 0; t < d; ++t)
        max_diff = fmaxf(max_diff, fabsf(__half2float(h_O16[t]) - h_ref[t]));

    printf("[Correctness vs FP32 CPU ref]\n");
    printf("  INT8 KV attention max_diff: %.4e\n", max_diff);
    printf("  result: %s\n\n", max_diff < 0.1f ? "PASS" : "FAIL");

    printf("[Memory: KV cache (K+V)]\n");
    printf("  FP16 KV: %zu bytes\n", (size_t)seq * d * 2 * 2);
    printf("  INT8 KV: %zu bytes + scale %zu B\n", (size_t)seq * d * 2, (size_t)seq * 4 * 2);
    printf("  savings: ~2.0x\n");

    cudaFree(d_Q); cudaFree(d_Ki); cudaFree(d_Vi); cudaFree(d_sk); cudaFree(d_sv); cudaFree(d_O);
    return 0;
}
