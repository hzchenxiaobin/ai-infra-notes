// cublas_bench.cu —— cuBLAS GEMM benchmark (供 ncu profiling 用)
// 仅运行 cuBLAS sgemm (TF32 模式), 供 Day 6 任务 3 的 ncu 命令分析
// 编译: nvcc -O3 -arch=sm_120 -lcublas cublas_bench.cu -o cublas_bench

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("cuBLAS sgemm (TF32 mode) benchmark\n");
    printf("M=N=K    | cublas_ms   TFLOPS\n");
    printf("---------|----------------------\n");

    for (int si = 0; si < 4; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t a_sz = (size_t)M * K, b_sz = (size_t)K * N, c_sz = (size_t)M * N;
        float *dA, *dB, *dC;
        cudaMalloc(&dA, a_sz * sizeof(float));
        cudaMalloc(&dB, b_sz * sizeof(float));
        cudaMalloc(&dC, c_sz * sizeof(float));
        float *hA = (float*)malloc(a_sz * sizeof(float));
        float *hB = (float*)malloc(b_sz * sizeof(float));
        for (size_t i = 0; i < a_sz; i++) hA[i] = (float)((rand() % 200) - 100) / 100.0f;
        for (size_t i = 0; i < b_sz; i++) hB[i] = (float)((rand() % 200) - 100) / 100.0f;
        cudaMemcpy(dA, hA, a_sz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, b_sz * sizeof(float), cudaMemcpyHostToDevice);

        float alpha = 1.0f, beta = 0.0f;
        // warmup
        for (int w = 0; w < 3; w++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, dB, N, dA, K, &beta, dC, N);
        cudaDeviceSynchronize();

        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int it = 0; it < 10; it++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, dB, N, dA, K, &beta, dC, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        ms /= 10.0f;
        cudaEventDestroy(s);
        cudaEventDestroy(e);

        double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        printf("%-8d | %-10.4f  %.1f\n", M, ms, tflops);

        free(hA);
        free(hB);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }
    cublasDestroy(handle);
    return 0;
}
