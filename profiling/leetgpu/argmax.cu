// argmax.cu —— Argmax 归约（两级归约 + Warp Shuffle）
// 编译命令: nvcc -o argmax argmax.cu -O3 -arch=sm_80

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// 将 (value, -index) 打包成 64-bit，用 atomicMax 同时比较值和索引
static __device__ __forceinline__ unsigned long long packArgmax(float val, int idx) {
    unsigned int uv = __float_as_uint(val);
    unsigned int ui = static_cast<unsigned int>(-idx);
    return (static_cast<unsigned long long>(uv) << 32) | ui;
}

__global__ void argmax_kernel(const float* input, int* out_idx, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        if (input[i] > local_max) {
            local_max = input[i];
            local_idx = i;
        }
    }

    __shared__ float s_val[32];
    __shared__ int   s_idx[32];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    // Warp 级归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int   other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
        if (other_val > local_max ||
            (other_val == local_max && other_idx < local_idx)) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    if (lane == 0) { s_val[wid] = local_max; s_idx[wid] = local_idx; }
    __syncthreads();

    // Warp 0 最终归约
    if (wid == 0) {
        int numWarps = (blockDim.x + 31) / 32;
        local_max = (lane < numWarps) ? s_val[lane] : -FLT_MAX;
        local_idx = (lane < numWarps) ? s_idx[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            int   other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max ||
                (other_val == local_max && other_idx < local_idx)) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            unsigned long long packed = packArgmax(local_max, local_idx);
            atomicMax(reinterpret_cast<unsigned long long*>(out_idx), packed);
        }
    }
}

int main() {
    const int N = 1 << 20;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) * 0.001f;
    h_in[N / 2] = 999.0f;  // 确保最大值在 N/2

    float *d_in; cudaMalloc(&d_in, N * sizeof(float));
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(unsigned long long));

    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);
    argmax_kernel<<<blocks, threads>>>(d_in, reinterpret_cast<int*>(d_out), N);

    unsigned long long packed;
    cudaMemcpy(&packed, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    int gpu_idx = -(static_cast<int>(packed & 0xFFFFFFFFULL));

    printf("GPU argmax idx = %d (expected %d) %s\n",
           gpu_idx, N / 2, gpu_idx == N / 2 ? "PASS" : "FAIL");

    free(h_in); cudaFree(d_in); cudaFree(d_out);
    return 0;
}
