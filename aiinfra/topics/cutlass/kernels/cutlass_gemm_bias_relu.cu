// cutlass_gemm_bias_relu.cu —— 融合 Epilogue GEMM（GEMM+Bias+ReLU）
// 编译: nvcc -o cutlass_gemm_bias_relu cutlass_gemm_bias_relu.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
// 运行: ./cutlass_gemm_bias_relu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

using namespace cutlass;

using ElementA    = half_t;
using ElementB    = half_t;
using ElementC    = half_t;
using ElementD    = half_t;
using ElementBias = half_t;
using ElementAcc  = float;

using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;

using ArchTag = arch::Sm90;
using OpClass = arch::OpClassTensorOp;

using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    LayoutA, LayoutB,
    ElementA, ElementB,
    ElementAcc,
    LayoutC
>::Type;

using EpilogueOp = epilogue::thread::LinearCombinationBiasRelu<
    ElementD,
    ElementAcc,
    ElementBias,
    ElementC,
    epilogue::thread::ReLu<ElementAcc>
>;

using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, ElementB, ElementAcc,
    ElementD, LayoutC, 1,
    EpilogueOp
>::Type;

using ProblemShape = Shape<int, int, int, int>;

using GemmKernel = gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = gemm::device::GemmUniversal<GemmKernel>;

void init_matrix(half_t* ptr, int size) {
    for (int i = 0; i < size; ++i)
        ptr[i] = half_t(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
}

void init_bias(half_t* ptr, int size) {
    for (int i = 0; i < size; ++i)
        ptr[i] = half_t(((float)rand() / RAND_MAX) * 0.1f);
}

int main() {
    int M = 4096, N = 4096, K = 4096;

    printf("=== CUTLASS GEMM + Bias + ReLU (Fused) ===\n");
    printf("Problem: %d x %d x %d (FP16)\n\n", M, N, K);

    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;
    size_t size_D = (size_t)M * N;
    size_t size_bias = (size_t)N;

    half_t *h_A = (half_t*)malloc(size_A * sizeof(half_t));
    half_t *h_B = (half_t*)malloc(size_B * sizeof(half_t));
    half_t *h_C = (half_t*)malloc(size_C * sizeof(half_t));
    half_t *h_D = (half_t*)malloc(size_D * sizeof(half_t));
    half_t *h_bias = (half_t*)malloc(size_bias * sizeof(half_t));

    init_matrix(h_A, size_A);
    init_matrix(h_B, size_B);
    init_matrix(h_C, size_C);
    init_bias(h_bias, size_bias);

    half_t *d_A, *d_B, *d_C, *d_D, *d_bias;
    cudaMalloc(&d_A, size_A * sizeof(half_t));
    cudaMalloc(&d_B, size_B * sizeof(half_t));
    cudaMalloc(&d_C, size_C * sizeof(half_t));
    cudaMalloc(&d_D, size_D * sizeof(half_t));
    cudaMalloc(&d_bias, size_bias * sizeof(half_t));

    cudaMemcpy(d_A, h_A, size_A * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, size_bias * sizeof(half_t), cudaMemcpyHostToDevice);

    typename Gemm::Arguments args{
        gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, {K, 1}},
        {d_B, {N, 1}},
        {d_bias, {1, 1}},
        {d_C, {N, 1}},
        {d_D, {N, 1}},
        {1.0f, 0.0f}
    };

    Gemm gemm;
    size_t ws_size = gemm.get_workspace_size(args);
    void *d_ws = nullptr;
    if (ws_size > 0) cudaMalloc(&d_ws, ws_size);

    cutlass::Status status = gemm.initialize(args, d_ws);
    if (status != cutlass::Status::kSuccess) {
        printf("INIT FAILED: %d\n", (int)status);
        return -1;
    }

    gemm.run();
    cudaDeviceSynchronize();

    float best_ms = 1e9f;
    for (int i = 0; i < 5; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemm.run();
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) best_ms = ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    double flops = 2.0 * M * N * K;
    double tflops = flops / (best_ms / 1000.0) / 1e12;

    printf("Fused GEMM+Bias+ReLU:\n");
    printf("  Duration: %.3f ms\n", best_ms);
    printf("  TFLOPS:   %.1f\n", tflops);

    cudaMemcpy(h_D, d_D, size_D * sizeof(half_t), cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < 5 && errors < 5; ++i) {
        for (int j = 0; j < 5 && errors < 5; ++j) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k)
                ref += float(h_A[i * K + k]) * float(h_B[k * N + j]);
            ref += float(h_bias[j]);
            ref = fmaxf(ref, 0.0f);
            float got = float(h_D[i * N + j]);
            if (fabsf(ref - got) > 1.0f) {
                printf("  MISMATCH D[%d][%d]: ref=%.2f, got=%.2f\n", i, j, ref, got);
                errors++;
            }
        }
    }
    printf("  Verify:   %s\n", errors == 0 ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_bias);
    if (d_ws) cudaFree(d_ws);
    free(h_A); free(h_B); free(h_C); free(h_D); free(h_bias);

    return 0;
}
