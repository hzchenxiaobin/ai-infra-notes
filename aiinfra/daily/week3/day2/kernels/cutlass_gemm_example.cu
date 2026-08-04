// cutlass_gemm_example.cu —— CUTLASS GEMM 实例化 + cuBLAS 对比
// 编译命令: nvcc -O3 -arch=sm_120 -I/path/to/cutlass/include -lcublas cutlass_gemm_example.cu -o cutlass_gemm
//
// 需要先 clone CUTLASS:
//   git clone https://github.com/NVIDIA/cutlass.git
//   export CUTLASS_PATH=/path/to/cutlass

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/numeric_types.h>

// 定义 CUTLASS GEMM 类型
// FP16 输入 (row-major A, col-major B), FP32 累加, FP32 输出
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                                    // InputType A
    cutlass::layout::RowMajor,                          // LayoutA
    cutlass::half_t,                                    // InputType B
    cutlass::layout::ColumnMajor,                       // LayoutB
    float,                                              // OutputType C
    cutlass::layout::RowMajor,                          // LayoutC
    float,                                              // AccumulatorType
    cutlass::arch::OpClassTensorOp,                     // OpClass: Tensor Core
    cutlass::arch::Sm80,                                // ArchTag (compatible with sm_120)
    cutlass::gemm::GemmShape<128, 128, 32>,             // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,               // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,                // InstructionShape
    cutlass::epilogue::thread::LinearCombination<float, float>,  // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
    2                                                   // NumStages (double buffer)
>;

void run_cutlass_gemm(
    int M, int N, int K,
    const cutlass::half_t* d_A,
    const cutlass::half_t* d_B,
    float* d_C,
    float alpha = 1.0f, float beta = 0.0f)
{
    Gemm gemm;

    Gemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, K},
        {d_C, N},
        {d_C, N},
        {alpha, beta}
    );

    cutlass::Status status = gemm.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS initialize failed: %d\n", (int)status);
        return;
    }

    status = gemm();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS run failed: %d\n", (int)status);
    }
}

void run_cublas_sgemm(
    cublasHandle_t handle, int M, int N, int K,
    const float* d_A, const float* d_B, float* d_C)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
}

int main(int argc, char** argv)
{
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("CUTLASS vs cuBLAS benchmark (FP16 input, FP32 accumulate)\n");
    printf("%-8s | %-14s %-14s %-8s | %-12s\n",
           "M=N=K", "CUTLASS(ms)", "cuBLAS(ms)", "Ratio", "CUTLASS TFLOPS");
    printf("---------|--------------------------------------------------|---------------\n");

    for (int si = 0; si < num_sizes; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];

        // FP32 data for cuBLAS
        float *h_A_f32 = (float*)malloc(M * K * sizeof(float));
        float *h_B_f32 = (float*)malloc(K * N * sizeof(float));
        for (int i = 0; i < M * K; i++) h_A_f32[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++) h_B_f32[i] = (float)(rand() % 100) / 100.0f;

        float *d_A_f32, *d_B_f32, *d_C_cublas;
        cudaMalloc(&d_A_f32, M * K * sizeof(float));
        cudaMalloc(&d_B_f32, K * N * sizeof(float));
        cudaMalloc(&d_C_cublas, M * N * sizeof(float));
        cudaMemcpy(d_A_f32, h_A_f32, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_f32, h_B_f32, K * N * sizeof(float), cudaMemcpyHostToDevice);

        // FP16 data for CUTLASS
        cutlass::half_t *h_A_f16 = (cutlass::half_t*)malloc(M * K * sizeof(cutlass::half_t));
        cutlass::half_t *h_B_f16 = (cutlass::half_t*)malloc(K * N * sizeof(cutlass::half_t));
        for (int i = 0; i < M * K; i++) h_A_f16[i] = (cutlass::half_t)h_A_f32[i];
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                h_B_f16[k + n * K] = (cutlass::half_t)h_B_f32[k * N + n];

        cutlass::half_t *d_A_f16, *d_B_f16;
        float *d_C_cutlass;
        cudaMalloc(&d_A_f16, M * K * sizeof(cutlass::half_t));
        cudaMalloc(&d_B_f16, K * N * sizeof(cutlass::half_t));
        cudaMalloc(&d_C_cutlass, M * N * sizeof(float));
        cudaMemcpy(d_A_f16, h_A_f16, M * K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_f16, h_B_f16, K * N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // --- CUTLASS GEMM ---
        run_cutlass_gemm(M, N, K, d_A_f16, d_B_f16, d_C_cutlass);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            run_cutlass_gemm(M, N, K, d_A_f16, d_B_f16, d_C_cutlass);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_cutlass = 0;
        cudaEventElapsedTime(&ms_cutlass, start, stop);
        ms_cutlass /= 10.0f;

        // --- cuBLAS sgemm ---
        run_cublas_sgemm(handle, M, N, K, d_A_f32, d_B_f32, d_C_cublas);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            run_cublas_sgemm(handle, M, N, K, d_A_f32, d_B_f32, d_C_cublas);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_cublas = 0;
        cudaEventElapsedTime(&ms_cublas, start, stop);
        ms_cublas /= 10.0f;

        double flops = 2.0 * M * N * K;
        double tflops_cutlass = flops / (ms_cutlass * 1e9);
        double ratio = ms_cublas / ms_cutlass * 100.0;

        printf("%-8d | %-14.3f %-14.3f %-8.1f | %-12.1f\n",
               M, ms_cutlass, ms_cublas, ratio, tflops_cutlass);

        // Correctness check (first size only)
        if (M == sizes[0]) {
            float *h_C_cutlass = (float*)malloc(M * N * sizeof(float));
            float *h_C_cublas = (float*)malloc(M * N * sizeof(float));
            cudaMemcpy(h_C_cutlass, d_C_cutlass, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

            float max_diff = 0.0f;
            for (int i = 0; i < M * N; i++) {
                float diff = fabsf(h_C_cutlass[i] - h_C_cublas[i]);
                if (diff > max_diff) max_diff = diff;
            }
            printf("  CUTLASS vs cuBLAS max_diff = %.2e (FP16 input precision loss expected)\n", max_diff);
            free(h_C_cutlass);
            free(h_C_cublas);
        }

        free(h_A_f32); free(h_B_f32);
        free(h_A_f16); free(h_B_f16);
        cudaFree(d_A_f32); cudaFree(d_B_f32); cudaFree(d_C_cublas);
        cudaFree(d_A_f16); cudaFree(d_B_f16); cudaFree(d_C_cutlass);
    }

    cublasDestroy(handle);
    printf("\nNote: CUTLASS uses FP16 input + FP32 accumulate (Tensor Core).\n");
    printf("      cuBLAS uses FP32 (sgemm) for baseline comparison.\n");
    printf("      ThreadblockShape<128,128,32>, WarpShape<64,64,32>, NumStages=2\n");
    return 0;
}
