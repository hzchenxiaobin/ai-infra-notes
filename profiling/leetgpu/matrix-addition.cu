// matrix_addition.cu —— Matrix Addition（1D grid-stride + float4 向量化）
// 编译命令: nvcc -o matrix_addition matrix_addition.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void matrix_add_float4(const float* A, const float* B, float* C, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 每个线程处理 4 个元素
    int vec_count = num_elements / 4;

    for (int i = tid; i < vec_count; i += stride) {
        float4 a = reinterpret_cast<const float4*>(A)[i];
        float4 b = reinterpret_cast<const float4*>(B)[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[i] = c;
    }
}

// 处理剩余不足 4 个的元素
__global__ void matrix_add_tail(const float* A, const float* B, float* C,
                                int vec_count, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = vec_count * 4 + tid; i < num_elements; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int num_elements = M * N;
    const size_t bytes = num_elements * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < num_elements; ++i) {
        h_A[i] = (float)(rand() % 100) * 0.01f;
        h_B[i] = (float)(rand() % 100) * 0.01f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = min((num_elements / 4 + threads - 1) / threads, 1024);

    matrix_add_float4<<<blocks, threads>>>(d_A, d_B, d_C, num_elements);
    matrix_add_tail<<<blocks, threads>>>(d_A, d_B, d_C, num_elements / 4, num_elements);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 简单验证
    bool pass = true;
    for (int i = 0; i < num_elements; ++i) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            pass = false;
            break;
        }
    }
    printf("Matrix Addition %s\n", pass ? "PASS" : "FAIL");

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
