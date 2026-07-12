#include <cuda_runtime.h>
#include <stdio.h>

__global__ void compute_intensive(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    float e = 0.0f, f = 0.0f, g = 0.0f, h = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = in[(idx + i) % n];
        a += v;
        b += v * 2;
        c += v * 3;
        d += v * 4;
        e += v * 5;
        f += v * 6;
        g += v * 7;
        h += v * 8;
    }
    out[idx] = a + b + c + d + e + f + g + h;
}

int main() {
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, compute_intensive);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("=== Kernel Attributes ===\n");
    printf("Registers per thread: %d\n", attr.numRegs);
    printf("Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    printf("Constant memory per block: %zu bytes\n", attr.constSizeBytes);
    printf("Local memory per thread: %zu bytes\n", attr.localSizeBytes);
    printf("Max threads per block: %d\n", attr.maxThreadsPerBlock);
    printf("=========================\n");

    const int N = 1 << 20;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    compute_intensive<<<grid_size, block_size>>>(d_in, d_out, 64);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}