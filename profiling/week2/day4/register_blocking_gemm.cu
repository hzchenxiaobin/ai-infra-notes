// register_blocking_gemm.cu —— Register Blocking 矩阵乘法完整实现
// 编译命令: nvcc -o register_gemm register_blocking_gemm.cu -O3 -arch=sm_120 -lcublas
// 运行命令: ./register_gemm

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))

__global__ void gemmRegisterBlocking(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    float r_A[TM];
    float r_B[TN];
    float acc[TM][TN] = {0};

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A tile
        int aRow = threadIdx.x / BK;
        int aCol = threadIdx.x % BK;
        #pragma unroll
        for (int i = 0; i < BM; i += NUM_THREADS / BK) {
            int loadRow = aRow + i;
            if (loadRow < BM && (cRow + loadRow) < M && (bk + aCol) < K) {
                s_A[loadRow][aCol] = A[(cRow + loadRow) * K + (bk + aCol)];
            } else if (loadRow < BM) {
                s_A[loadRow][aCol] = 0.0f;
            }
        }

        // 协作加载 B tile
        int bRow = threadIdx.x / BN;
        int bCol = threadIdx.x % BN;
        #pragma unroll
        for (int i = 0; i < BK; i += NUM_THREADS / BN) {
            int loadRow = bRow + i;
            if (loadRow < BK && (bk + loadRow) < K && (cCol + bCol) < N) {
                s_B[loadRow][bCol] = B[(bk + loadRow) * N + (cCol + bCol)];
            } else if (loadRow < BK) {
                s_B[loadRow][bCol] = 0.0f;
            }
        }

        __syncthreads();

        // Register Blocking 计算
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                r_A[m] = s_A[threadRow * TM + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                r_B[n] = s_B[k][threadCol * TN + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] += r_A[m] * r_B[n];
                }
            }
        }
        __syncthreads();
    }

    // 写回 Global Memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gRow = cRow + threadRow * TM + m;
            int gCol = cCol + threadCol * TN + n;
            if (gRow < M && gCol < N) {
                C[gRow * N + gCol] = acc[m][n];
            }
        }
    }
}

float runCuBLAS(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void initMatrix(float* mat, int rows, int cols) {
    srand(42);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
    }
}

bool checkResult(const float* gpu, const float* cpu, int M, int N, float eps) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(gpu[i] - cpu[i]) > eps) {
            printf("Mismatch at [%d][%d]: GPU=%.6f, CPU=%.6f\n",
                   i / N, i % N, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

float getGFLOPS(int M, int N, int K, float ms) {
    return 2.0f * M * N * K / (ms * 1e6);
}

int main() {
    int sizes[][3] = {{1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096}};

    printf("=== Register Blocking GEMM ===\n");
    printf("Parameters: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d, Threads=%d\n",
           BM, BN, BK, TM, TN, NUM_THREADS);
    printf("%-10s %-10s %-10s %-12s %-12s %-10s\n",
           "M", "N", "K", "Our(ms)", "cuBLAS(ms)", "Percent");
    printf("------------------------------------------------------------\n");

    for (int s = 0; s < 3; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);

        float *h_A = (float*)malloc(sizeA);
        float *h_B = (float*)malloc(sizeB);
        float *h_C = (float*)malloc(sizeC);
        float *h_C_ref = (float*)malloc(sizeC);
        initMatrix(h_A, M, K);
        initMatrix(h_B, K, N);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);
        cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 blockDim(NUM_THREADS);

        gemmRegisterBlocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemmRegisterBlocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ourMs;
        cudaEventElapsedTime(&ourMs, start, stop);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

        float cublasMs = runCuBLAS(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C_ref, d_C, sizeC, cudaMemcpyDeviceToHost);

        bool correct = checkResult(h_C, h_C_ref, M, N, 1e-2);
        float percent = (cublasMs / ourMs) * 100;

        printf("%-10d %-10d %-10d %-12.3f %-12.3f %-9.1f%% %s\n",
               M, N, K, ourMs, cublasMs, percent, correct ? "PASS" : "FAIL");

        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}
