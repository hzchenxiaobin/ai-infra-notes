// cutlass_gemm_3x.cu —— CUTLASS 3.x GEMM（含计时 + 多尺寸 + TFLOPS）
// 编译: nvcc -o cutlass_gemm_3x cutlass_gemm_3x.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
// 运行: ./cutlass_gemm_3x

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cutlass;

using ElementA   = half_t;
using ElementB   = half_t;
using ElementC   = half_t;
using ElementAcc = float;

using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;

using ArchTag  = arch::Sm90;
using OpClass  = arch::OpClassTensorOp;

using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    LayoutA, LayoutB,
    ElementA, ElementB,
    ElementAcc,
    LayoutC
>::Type;

using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, ElementB, ElementAcc,
    ElementC, LayoutC, 1
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

double run_gemm(int M, int N, int K) {
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
    init_matrix(h_C, size_C);

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
    size_t ws_size = gemm.get_workspace_size(args);
    void *d_ws = nullptr;
    if (ws_size > 0) cudaMalloc(&d_ws, ws_size);

    cutlass::Status status = gemm.initialize(args, d_ws);
    if (status != cutlass::Status::kSuccess) {
        printf("  INIT FAILED: %d\n", (int)status);
        return -1.0;
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

    cudaMemcpy(h_D, d_D, bytes_C, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < 10 && errors < 5; ++i) {
        for (int j = 0; j < 10 && errors < 5; ++j) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k)
                ref += float(h_A[i * K + k]) * float(h_B[k * N + j]);
            if (fabsf(ref - float(h_D[i * N + j])) > 1.0f) errors++;
        }
    }

    printf("  %4d x %4d x %4d | %8.3f ms | %7.1f TFLOPS | %s\n",
           M, N, K, best_ms, tflops,
           errors == 0 ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    if (d_ws) cudaFree(d_ws);
    free(h_A); free(h_B); free(h_C); free(h_D);

    return tflops;
}

int main() {
    printf("=== CUTLASS 3.x GEMM Benchmark ===\n\n");
    printf("  Size (M x N x K) | Duration  | TFLOPS     | Verify\n");
    printf("  -----------------+-----------+------------+-------\n");

    struct { int M, N, K; } sizes[] = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };

    for (auto& s : sizes) {
        run_gemm(s.M, s.N, s.K);
    }

    return 0;
}
