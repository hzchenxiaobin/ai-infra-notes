// wmma_gemm.cu —— Tensor Core (WMMA) GEMM: FP16 输入, FP32 累加
// 编译命令: nvcc -O3 -arch=sm_120 -lcublas wmma_gemm.cu -o wmma_gemm
// 对比 FMA GEMM vs WMMA GEMM vs cuBLAS

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

// ---------- WMMA GEMM (Tensor Core) ----------
// C = A @ B, A: [M, K] row-major (half), B: [K, N] col-major (half), C: [M, N] (float)
// 使用 WMMA m16n16k16 fragment, 每个 warp 计算 WMMA_M x WMMA_N 的输出 tile
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        const __half* tileA = A + warpM * WMMA_M * K + k;
        const __half* tileB = B + k + warpN * WMMA_N * K;

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, K);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* tileC = C + warpM * WMMA_M * N + warpN * WMMA_N;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

// ---------- FMA GEMM (Register Blocking baseline, FP32) ----------
#define BLOCK_SIZE 16
__global__ void fma_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ---------- Helper: cuBLAS sgemm ----------
void cublas_sgemm(
    cublasHandle_t handle, int M, int N, int K,
    const float* d_A, const float* d_B, float* d_C)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
}

// ---------- Main: benchmark FMA vs WMMA vs cuBLAS ----------
int main(int argc, char** argv)
{
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("%-8s | %-12s %-12s %-12s | %-8s %-8s %-8s\n",
           "M=N=K", "FMA(ms)", "WMMA(ms)", "cuBLAS(ms)",
           "FMA%", "WMMA%", "WMMA/FMA");
    printf("---------|------------------------------------------------|----------------------------\n");

    for (int si = 0; si < num_sizes; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t bytes_f32 = (size_t)M * N * sizeof(float);
        size_t bytes_f16 = (size_t)M * K * sizeof(__half);

        // FP32 buffers for FMA + cuBLAS
        float *h_A_f32 = (float*)malloc(M * K * sizeof(float));
        float *h_B_f32 = (float*)malloc(K * N * sizeof(float));
        float *h_C_f32 = (float*)malloc(M * N * sizeof(float));
        for (int i = 0; i < M * K; i++) h_A_f32[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++) h_B_f32[i] = (float)(rand() % 100) / 100.0f;

        float *d_A_f32, *d_B_f32, *d_C_f32;
        cudaMalloc(&d_A_f32, M * K * sizeof(float));
        cudaMalloc(&d_B_f32, K * N * sizeof(float));
        cudaMalloc(&d_C_f32, M * N * sizeof(float));
        cudaMemcpy(d_A_f32, h_A_f32, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_f32, h_B_f32, K * N * sizeof(float), cudaMemcpyHostToDevice);

        // FP16 buffers for WMMA (A row-major, B col-major for WMMA)
        __half *h_A_f16 = (__half*)malloc(M * K * sizeof(__half));
        __half *h_B_f16 = (__half*)malloc(K * N * sizeof(__half));
        for (int i = 0; i < M * K; i++) h_A_f16[i] = __float2half(h_A_f32[i]);
        // B stored col-major: B[k*N+n] -> B_col[k + n*K]
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                h_B_f16[k + n * K] = __float2half(h_B_f32[k * N + n]);

        __half *d_A_f16, *d_B_f16;
        cudaMalloc(&d_A_f16, M * K * sizeof(__half));
        cudaMalloc(&d_B_f16, K * N * sizeof(__half));
        cudaMemcpy(d_A_f16, h_A_f16, M * K * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_f16, h_B_f16, K * N * sizeof(__half), cudaMemcpyHostToDevice);

        float *d_C_wmma;
        cudaMalloc(&d_C_wmma, M * N * sizeof(float));

        // --- FMA GEMM ---
        dim3 fma_block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 fma_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // warmup
        fma_gemm_kernel<<<fma_grid, fma_block>>>(d_A_f32, d_B_f32, d_C_f32, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            fma_gemm_kernel<<<fma_grid, fma_block>>>(d_A_f32, d_B_f32, d_C_f32, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_fma = 0;
        cudaEventElapsedTime(&ms_fma, start, stop);
        ms_fma /= 10.0f;

        // --- WMMA GEMM ---
        dim3 wmma_grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        dim3 wmma_block(32, 1); // 1 warp per block

        // warmup
        wmma_gemm_kernel<<<wmma_grid, wmma_block>>>(d_A_f16, d_B_f16, d_C_wmma, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            wmma_gemm_kernel<<<wmma_grid, wmma_block>>>(d_A_f16, d_B_f16, d_C_wmma, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_wmma = 0;
        cudaEventElapsedTime(&ms_wmma, start, stop);
        ms_wmma /= 10.0f;

        // --- cuBLAS sgemm ---
        cublas_sgemm(handle, M, N, K, d_A_f32, d_B_f32, d_C_f32);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            cublas_sgemm(handle, M, N, K, d_A_f32, d_B_f32, d_C_f32);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_cublas = 0;
        cudaEventElapsedTime(&ms_cublas, start, stop);
        ms_cublas /= 10.0f;

        // --- TFLOPS ---
        double flops = 2.0 * M * N * K;
        double tflops_fma = flops / (ms_fma * 1e9);
        double tflops_wmma = flops / (ms_wmma * 1e9);
        double tflops_cublas = flops / (ms_cublas * 1e9);

        printf("%-8d | %-12.3f %-12.3f %-12.3f | %-8.1f %-8.1f %-8.1f\n",
               M, ms_fma, ms_wmma, ms_cublas,
               tflops_fma, tflops_wmma, tflops_cublas);

        // --- Correctness check (WMMA vs cuBLAS) ---
        float *h_C_wmma = (float*)malloc(M * N * sizeof(float));
        float *h_C_cublas = (float*)malloc(M * N * sizeof(float));
        cudaMemcpy(h_C_wmma, d_C_wmma, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_cublas, d_C_f32, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float diff = fabsf(h_C_wmma[i] - h_C_cublas[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (M == sizes[0]) // only print for first size
            printf("  WMMA vs cuBLAS max_diff = %.2e (FP16 input precision loss expected)\n", max_diff);

        free(h_C_wmma);
        free(h_C_cublas);
        free(h_A_f32); free(h_B_f32); free(h_C_f32);
        free(h_A_f16); free(h_B_f16);
        cudaFree(d_A_f32); cudaFree(d_B_f32); cudaFree(d_C_f32);
        cudaFree(d_A_f16); cudaFree(d_B_f16); cudaFree(d_C_wmma);
    }

    cublasDestroy(handle);
    printf("\nNote: WMMA uses FP16 input + FP32 accumulate.\n");
    printf("      FMA uses FP32 throughout. cuBLAS uses FP32 (sgemm).\n");
    printf("      RTX 5090 FP16 Tensor Core peak ~209 TFLOPS (dense).\n");
    return 0;
}
