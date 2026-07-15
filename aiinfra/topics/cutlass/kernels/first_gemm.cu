// first_gemm.cu —— 用 CUTLASS 3.x API 写的第一个 GEMM
// 编译: nvcc -o first_gemm first_gemm.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17 -lcudart
// 运行: ./first_gemm

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cutlass;

using ElementA = half_t;
using ElementB = half_t;
using ElementC = half_t;
using ElementAccumulator = float;

using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;

using ArchTag = arch::Sm90;
using OpClass = arch::OpClassTensorOp;

using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    LayoutA, LayoutB,
    ElementA, ElementB,
    ElementAccumulator,
    LayoutC
>::Type;

using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, ElementB, ElementAccumulator,
    ElementC, LayoutC, 1
>::Type;

using ProblemShape = Shape<int, int, int, int>;

using GemmKernel = gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = gemm::device::GemmUniversal<GemmKernel>;

void init_matrix(half_t* h_ptr, int size, float scale = 1.0f) {
    for (int i = 0; i < size; ++i) {
        h_ptr[i] = half_t(((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale);
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;

    printf("=== CUTLASS 3.x First GEMM ===\n");
    printf("Problem: %d x %d x %d (FP16)\n\n", M, N, K);

    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;
    size_t bytes_A = size_A * sizeof(half_t);
    size_t bytes_B = size_B * sizeof(half_t);
    size_t bytes_C = size_C * sizeof(half_t);

    half_t *h_A = (half_t*)malloc(bytes_A);
    half_t *h_B = (half_t*)malloc(bytes_B);
    half_t *h_C = (half_t*)malloc(bytes_C);
    half_t *h_D = (half_t*)malloc(bytes_C);

    init_matrix(h_A, size_A);
    init_matrix(h_B, size_B);
    init_matrix(h_C, size_C, 0.0f);

    half_t *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_D, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice);

    typename Gemm::Arguments args{
        gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, {K, 1}},
        {d_B, {N, 1}},
        {d_C, {N, 1}},
        {d_D, {N, 1}},
        {1.0f, 0.0f}
    };

    Gemm gemm;
    size_t workspace_size = gemm.get_workspace_size(args);
    void *d_workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    }

    cutlass::Status status = gemm.initialize(args, d_workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("GEMM init failed: %d\n", (int)status);
        return -1;
    }

    gemm.run();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemm.run();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms / 1000.0) / 1e12;

    printf("Result:\n");
    printf("  Duration: %.3f ms\n", ms);
    printf("  TFLOPS:   %.1f\n", tflops);
    printf("  Bandwidth: %.1f GB/s\n",
           (bytes_A + bytes_B + bytes_C) / (ms / 1000.0) / 1e9);

    cudaMemcpy(h_D, d_D, bytes_C, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < 5 && errors < 5; ++i) {
        for (int j = 0; j < 5 && errors < 5; ++j) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k) {
                ref += float(h_A[i * K + k]) * float(h_B[k * N + j]);
            }
            float got = float(h_D[i * N + j]);
            if (fabsf(ref - got) > 0.5f) {
                printf("  MISMATCH D[%d][%d]: ref=%.2f, got=%.2f\n", i, j, ref, got);
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("  PASSED (first 5x5 elements correct)\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    if (d_workspace) cudaFree(d_workspace);
    free(h_A); free(h_B); free(h_C); free(h_D);

    printf("\nDone!\n");
    return 0;
}
