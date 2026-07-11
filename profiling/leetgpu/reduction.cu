// reduction.cu —— 并行归约（Warp Shuffle + 两级归约）
// 编译命令: nvcc -o reduction reduction.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float warpSums[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float sum = 0.0f;
    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        sum += input[i];
    }

    sum = warpReduceSum(sum);

    if (lane == 0) warpSums[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        int numWarps = (blockDim.x + 31) / 32;
        sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) output[blockIdx.x] = sum;
    }
}

int main() {
    const int N = 1 << 22;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 1000) * 0.001f;
    }

    float *d_in, *d_temp, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_temp, 1024 * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduction_kernel<<<blocks, threads>>>(d_in, d_temp, N);
    reduction_kernel<<<1, 256>>>(d_temp, d_out, blocks);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        cpu_sum += h_in[i];
    }

    printf("GPU=%.4f CPU=%.4f diff=%.6f %s\n",
           gpu_sum, (float)cpu_sum, fabs(gpu_sum - (float)cpu_sum),
           fabs(gpu_sum - (float)cpu_sum) < 1e-3 ? "PASS" : "FAIL");
    printf("Time: %.3f ms (%.1f GB/s)\n", ms, N * sizeof(float) / (ms * 1e6));

    free(h_in); cudaFree(d_in); cudaFree(d_temp); cudaFree(d_out);
    return 0;
}
