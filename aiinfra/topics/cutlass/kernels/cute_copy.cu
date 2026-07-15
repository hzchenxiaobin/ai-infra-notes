// cute_copy.cu —— CuTe copy 练习：Global -> Shared -> Global
// 编译: nvcc -o cute_copy cute_copy.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17

#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cute;

__global__ void copy_kernel(float const* gmem_src, float* gmem_dst, int M, int N) {
    auto gmem_layout = make_layout(make_shape(M, N), make_stride(N, 1));
    auto gA = make_tensor(make_gmem_ptr(gmem_src), gmem_layout);

    __shared__ float smem[16 * 16];
    auto smem_layout = make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{}));
    auto sA = make_tensor(make_smem_ptr(smem), smem_layout);

    int block_m = blockIdx.y;
    int block_n = blockIdx.x;

    auto gA_tile = gA(make_range(block_m * 16, block_m * 16 + 16),
                      make_range(block_n * 16, block_n * 16 + 16));

    copy(gA_tile, sA);
    __syncthreads();

    auto gD = make_tensor(make_gmem_ptr(gmem_dst), gmem_layout);
    auto gD_tile = gD(make_range(block_m * 16, block_m * 16 + 16),
                      make_range(block_n * 16, block_n * 16 + 16));

    copy(sA, gD_tile);
}

int main() {
    int M = 32, N = 32;
    int size = M * N;
    size_t bytes = size * sizeof(float);

    float *h_src = (float*)malloc(bytes);
    float *h_dst = (float*)malloc(bytes);
    for (int i = 0; i < size; ++i) h_src[i] = (float)i;

    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, bytes);

    dim3 grid(N / 16, M / 16);
    dim3 block(16, 16);
    copy_kernel<<<grid, block>>>(d_src, d_dst, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < size; ++i) {
        if (h_src[i] != h_dst[i]) errors++;
    }
    std::cout << "Copy test: " << (errors == 0 ? "PASSED" : "FAILED");
    std::cout << " (" << (size - errors) << "/" << size << " correct)" << std::endl;

    cudaFree(d_src); cudaFree(d_dst);
    free(h_src); free(h_dst);
    return 0;
}
