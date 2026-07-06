#include <cuda_runtime.h>
#include <stdio.h>

// 这个 kernel 故意使用大量局部变量，并通过 __launch_bounds__ 限制寄存器数量，
// 迫使编译器将变量溢出到 local memory（实际在 global memory 中），
// 从而产生 register spilling。
//
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// 含义：每个 block 最多 128 线程，每个 SM 至少同时运行 8 个 block。
// 以 sm_52 为例，每个 SM 有 64K 个 32-bit 寄存器：
//   每个线程最多可用寄存器 ≈ 65536 / (128 * 8) = 64 个
// 但下面代码中同时活跃的 float 变量超过 64 个，编译器只能把部分变量 spill 到 local memory。
__launch_bounds__(128, 8)
__global__ void spill_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 通过常量索引访问数组，编译器会尝试把 acc[80] 都保留在寄存器中
    float acc[80];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        acc[i] = in[(idx + i) % n];
    }

    // 对所有元素做一次规约，制造大量同时 live 的变量
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        sum += acc[i] * acc[i] + 1.0f;
    }

    out[idx] = sum;
}

// 作为对比：同样的计算逻辑，但不限制寄存器，通常不会发生 spilling
__global__ void no_spill_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float acc[80];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        acc[i] = in[(idx + i) % n];
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        sum += acc[i] * acc[i] + 1.0f;
    }

    out[idx] = sum;
}

int main() {
    const int N = 1 << 20;  // 约 100 万元素
    size_t bytes = N * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    int block_size = 128;
    int grid_size = (N + block_size - 1) / block_size;

    printf("=== Register Spill Demo ===\n");
    printf("Launching: grid=%d, block=%d, total_threads=%d\n",
           grid_size, block_size, grid_size * block_size);

    spill_kernel<<<grid_size, block_size>>>(d_in, d_out, N);
    no_spill_kernel<<<grid_size, block_size>>>(d_in, d_out, N);

    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);

    printf("Done. Use the command below to check spilling:\n");
    printf("  nvcc -Xptxas -v week1/day2/exercise/register_spill.cu\n");
    printf("Look for 'spill stores' and 'spill loads' in the output.\n");

    return 0;
}
