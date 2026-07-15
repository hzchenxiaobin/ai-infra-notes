// cutlass_gemm_tiles.cu —— 对比不同 TileShape 的性能
// 编译: nvcc -o cutlass_gemm_tiles cutlass_gemm_tiles.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
// 运行: ./cutlass_gemm_tiles

#include <cuda_runtime.h>
#include <stdio.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cutlass;

template <typename TileShape>
double run_with_tile(int M, int N, int K) {
    using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
        arch::Sm90, arch::OpClassTensorOp,
        layout::RowMajor, layout::ColumnMajor,
        half_t, half_t, float,
        layout::RowMajor,
        TileShape
    >::Type;

    using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
        arch::Sm90, arch::OpClassTensorOp,
        half_t, half_t, float,
        half_t, layout::RowMajor, 1
    >::Type;

    using GemmKernel = gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue
    >;
    using Gemm = gemm::device::GemmUniversal<GemmKernel>;

    half_t *d_A, *d_B, *d_C, *d_D;
    size_t bytes_A = (size_t)M * K * sizeof(half_t);
    size_t bytes_B = (size_t)K * N * sizeof(half_t);
    size_t bytes_C = (size_t)M * N * sizeof(half_t);
    cudaMalloc(&d_A, bytes_A); cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C); cudaMalloc(&d_D, bytes_C);

    typename Gemm::Arguments args{
        gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, {K, 1}}, {d_B, {N, 1}},
        {d_C, {N, 1}}, {d_D, {N, 1}},
        {1.0f, 0.0f}
    };

    Gemm gemm;
    size_t ws_size = gemm.get_workspace_size(args);
    void *d_ws = nullptr;
    if (ws_size > 0) cudaMalloc(&d_ws, ws_size);
    gemm.initialize(args, d_ws);

    gemm.run(); cudaDeviceSynchronize();

    float best_ms = 1e9f;
    for (int i = 0; i < 5; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemm.run();
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        float ms; cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) best_ms = ms;
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    double tflops = 2.0 * M * N * K / (best_ms / 1000.0) / 1e12;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    if (d_ws) cudaFree(d_ws);
    return tflops;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("=== TileShape Comparison (M=N=K=%d) ===\n\n", M);

    double t1 = run_with_tile<Shape<_128, _128, _64>>(M, N, K);
    printf("  128x128x64:  %.1f TFLOPS\n", t1);

    double t2 = run_with_tile<Shape<_128, _256, _64>>(M, N, K);
    printf("  128x256x64:  %.1f TFLOPS\n", t2);

    double t3 = run_with_tile<Shape<_256, _128, _64>>(M, N, K);
    printf("  256x128x64:  %.1f TFLOPS\n", t3);

    double t4 = run_with_tile<Shape<_128, _128, _128>>(M, N, K);
    printf("  128x128x128: %.1f TFLOPS\n", t4);

    return 0;
}
