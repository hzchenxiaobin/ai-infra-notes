// relu.cu —— ReLU 激活函数
// 编译命令: nvcc -o relu relu.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 2000 - 1000) * 0.001f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // 测试不同 block size
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int bs : block_sizes) {
        int grid = (N + bs - 1) / bs;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        relu_kernel<<<grid, bs>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("block=%4d  time=%.3f ms  bw=%.1f GB/s\n",
               bs, ms, 2.0f * N * sizeof(float) / (ms * 1e6));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != fmaxf(h_in[i], 0.0f)) { ok = false; break; }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
