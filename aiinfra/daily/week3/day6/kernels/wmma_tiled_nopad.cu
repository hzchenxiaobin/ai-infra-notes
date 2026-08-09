// wmma_tiled_nopad.cu —— Day 2 tiled WMMA GEMM 的无 padding 变体
// 用于 Day 6 实验 1：对比有 padding vs 无 padding 的 bank conflict
// 与 wmma_gemm_tiled.cu 唯一区别：smemA/smemB 不加 SMEM_PAD
// 编译: nvcc -O3 -arch=sm_120 -lcublas wmma_tiled_nopad.cu -o wmma_tiled_nopad

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 16
#define WM 32
#define WN 32
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_tiled_nopad_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_x = warp_id % 2;
    int warp_y = warp_id / 2;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    __shared__ __half smemA[BM][BK];
    __shared__ __half smemB[BK][BN];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    for (int k = 0; k < K; k += BK) {
        int load_row = warp_id * 16;
        for (int i = 0; i < 16; i++)
            for (int j = threadIdx.x; j < BK; j += WARP_SIZE) {
                int gr = block_row + load_row + i;
                if (gr < M && (k + j) < K)
                    smemA[load_row + i][j] = A[gr * K + k + j];
            }
        int load_col = warp_id * 16;
        for (int i = threadIdx.x; i < BK; i += WARP_SIZE)
            for (int j = 0; j < 16; j++) {
                int gc = block_col + load_col + j;
                if ((k + i) < K && gc < N)
                    smemB[i][load_col + j] = B[(k + i) * N + gc];
            }
        __syncthreads();

        for (int i = 0; i < 2; i++)
            wmma::load_matrix_sync(a_frag[i],
                &smemA[warp_y * 16 + i * 16][0], BK);
        for (int j = 0; j < 2; j++)
            wmma::load_matrix_sync(b_frag[j],
                &smemB[0][warp_x * 16 + j * 16], BN);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        __syncthreads();
    }

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            int row = block_row + warp_y * 16 + i * 16;
            int col = block_col + warp_x * 16 + j * 16;
            if (row < M && col < N)
                wmma::store_matrix_sync(C + row * N + col,
                    c_frag[i][j], N, wmma::mem_row_major);
        }
}

int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("wmma_tiled_nopad (no bank conflict padding) vs cuBLAS TF32\n");
    printf("M=N=K    | nopad_ms   TF32cub_ms  | nopad%%TF32  max_diff\n");
    printf("---------|-----------------------------------------|-------------------------\n");

    for (int si = 0; si < 4; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t a_sz = (size_t)M * K, b_sz = (size_t)K * N, c_sz = (size_t)M * N;
        __half *dA, *dB;
        float *dC, *dCref, *dAf, *dBf;
        cudaMalloc(&dA, a_sz * sizeof(__half));
        cudaMalloc(&dB, b_sz * sizeof(__half));
        cudaMalloc(&dC, c_sz * sizeof(float));
        cudaMalloc(&dCref, c_sz * sizeof(float));
        cudaMalloc(&dAf, a_sz * sizeof(float));
        cudaMalloc(&dBf, b_sz * sizeof(float));

        __half *hA = (__half*)malloc(a_sz * sizeof(__half));
        __half *hB = (__half*)malloc(b_sz * sizeof(__half));
        float *hAf = (float*)malloc(a_sz * sizeof(float));
        float *hBf = (float*)malloc(b_sz * sizeof(float));
        for (size_t i = 0; i < a_sz; i++) {
            hA[i] = __float2half((float)((rand() % 200) - 100) / 100.0f);
            hAf[i] = __half2float(hA[i]);
        }
        for (size_t i = 0; i < b_sz; i++) {
            hB[i] = __float2half((float)((rand() % 200) - 100) / 100.0f);
            hBf[i] = __half2float(hB[i]);
        }
        cudaMemcpy(dA, hA, a_sz * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, b_sz * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dAf, hAf, a_sz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dBf, hBf, b_sz * sizeof(float), cudaMemcpyHostToDevice);

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(128);

        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        // warmup
        wmma_gemm_tiled_nopad_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        wmma_gemm_tiled_nopad_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float np_ms;
        cudaEventElapsedTime(&np_ms, s, e);
        cudaEventDestroy(s);
        cudaEventDestroy(e);

        float alpha = 1.0f, beta = 0.0f;
        for (int w = 0; w < 3; w++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, dBf, N, dAf, K, &beta, dCref, N);
        cudaDeviceSynchronize();
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int it = 0; it < 10; it++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, dBf, N, dAf, K, &beta, dCref, N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float cub_ms;
        cudaEventElapsedTime(&cub_ms, s, e);
        cub_ms /= 10.0f;
        cudaEventDestroy(s);
        cudaEventDestroy(e);

        float *hC = (float*)malloc(c_sz * sizeof(float));
        float *hCref = (float*)malloc(c_sz * sizeof(float));
        cudaMemcpy(hC, dC, c_sz * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hCref, dCref, c_sz * sizeof(float), cudaMemcpyDeviceToHost);
        float max_diff = 0.0f;
        for (size_t i = 0; i < c_sz; i++) {
            float d = fabsf(hC[i] - hCref[i]);
            if (d > max_diff) max_diff = d;
        }

        printf("%-8d | %-10.4f  %-11.4f | %-9.1f  %.2e\n",
               M, np_ms, cub_ms, 100.0f * cub_ms / np_ms, max_diff);

        free(hA);
        free(hB);
        free(hAf);
        free(hBf);
        free(hC);
        free(hCref);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFree(dCref);
        cudaFree(dAf);
        cudaFree(dBf);
    }
    cublasDestroy(handle);
    return 0;
}
