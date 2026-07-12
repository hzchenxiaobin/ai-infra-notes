// softmax.cu —— 三遍扫描 Softmax + Profiling
// 编译命令: nvcc -o softmax softmax.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// 三遍扫描: 每行独立处理
__global__ void softmax_three_pass(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // Pass 1: 求 max (数值稳定性)
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Pass 2: 求 sum(exp(x - max))
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(input[i] - max_val);
    }

    // Pass 3: 归一化
    output[idx] = expf(input[idx] - max_val) / sum;
}

// 优化版: Online Softmax (两遍, 减少 1 次全局内存读取)
__global__ void softmax_online(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // Pass 1: 同时求 max 和 sum(exp(x - running_max))
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float x = input[i];
        float new_max = fmaxf(max_val, x);
        sum = sum * expf(max_val - new_max) + expf(x - new_max);
        max_val = new_max;
    }

    // Pass 2: 归一化
    output[idx] = expf(input[idx] - max_val) / sum;
}

int main() {
    const int N = 1 << 16;
    float* h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 2000 - 1000) * 0.01f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);

    cudaEventRecord(s1);
    softmax_three_pass<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(s2);
    cudaEventSynchronize(s2);
    float ms_3pass;
    cudaEventElapsedTime(&ms_3pass, s1, s2);

    cudaEventRecord(s1);
    softmax_online<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(s2);
    cudaEventSynchronize(s2);
    float ms_online;
    cudaEventElapsedTime(&ms_online, s1, s2);

    printf("Three-pass: %.3f ms\n", ms_3pass);
    printf("Online:      %.3f ms\n", ms_online);
    printf("Speedup:     %.2fx\n", ms_3pass / ms_online);

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
