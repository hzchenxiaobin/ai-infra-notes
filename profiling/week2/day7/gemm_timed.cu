// gemm_timed.cu —— 60 分钟手撕 Register Blocking GEMM（ncu profiling 版）
// 编译: nvcc -o gemm_timed gemm_timed.cu -O3 -arch=sm_120 -lcublas -g -lineinfo
// 运行: ./gemm_timed

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM/TM)*(BN/TN))   // 256

__global__ void gemmRegisterBlocking(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    float r_A[TM];
    float r_B[TN];
    float acc[TM][TN] = {{0}};

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A tile
        #pragma unroll
        for (int i = 0; i < BM; i += NUM_THREADS / BK) {
            int row = threadIdx.x / BK + i;
            int col = threadIdx.x % BK;
            if (cRow + row < M && bk + col < K)
                s_A[row][col] = A[(cRow + row) * K + (bk + col)];
            else
                s_A[row][col] = 0.0f;
        }
        // 协作加载 B tile
        #pragma unroll
        for (int i = 0; i < BK; i += NUM_THREADS / BN) {
            int row = threadIdx.x / BN + i;
            int col = threadIdx.x % BN;
            if (bk + row < K && cCol + col < N)
                s_B[row][col] = B[(bk + row) * N + (cCol + col)];
            else
                s_B[row][col] = 0.0f;
        }
        __syncthreads();

        // Register Blocking 计算
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) r_A[m] = s_A[threadRow*TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) r_B[n] = s_B[k][threadCol*TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += r_A[m] * r_B[n];
        }
        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gRow = cRow + threadRow * TM + m;
            int gCol = cCol + threadCol * TN + n;
            if (gRow < M && gCol < N) C[gRow * N + gCol] = acc[m][n];
        }
    }
}

// cuBLAS 基准
float runCuBLAS(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cublasDestroy(handle); cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

void initMatrix(float* mat, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++)
        if (fabs(a[i] - b[i]) > eps) return false;
    return true;
}

int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    printf("=== Register Blocking GEMM (手撕验收) ===\n");
    printf("BM=%d BN=%d BK=%d TM=%d TN=%d Threads=%d\n\n", BM, BN, BK, TM, TN, NUM_THREADS);
    printf("%-8s %-10s %-10s %-10s %-8s\n", "N", "Our(ms)", "cuBLAS(ms)", "GFLOPS", "Percent");
    printf("------------------------------------------------\n");

    for (int s = 0; s < 4; s++) {
        int N = sizes[s], M = N, K = N;
        size_t bytes = (size_t)M * N * sizeof(float);
        float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes), *h_C_ref = (float*)malloc(bytes);
        initMatrix(h_A, M * K); initMatrix(h_B, K * N);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);

        gemmRegisterBlocking<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEvent_t st, sp;
        cudaEventCreate(&st); cudaEventCreate(&sp);
        cudaEventRecord(st);
        gemmRegisterBlocking<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float ourMs; cudaEventElapsedTime(&ourMs, st, sp);
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

        float cublasMs = runCuBLAS(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C_ref, d_C, bytes, cudaMemcpyDeviceToHost);

        bool ok = checkResult(h_C, h_C_ref, M * N, 1e-2);
        float gflops = 2.0f * M * N * K / (ourMs * 1e6);
        float pct = (cublasMs / ourMs) * 100;

        printf("%-8d %-10.3f %-10.3f %-10.1f %-7.1f%% %s\n",
               N, ourMs, cublasMs, gflops, pct, ok ? "PASS" : "FAIL");

        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(st); cudaEventDestroy(sp);
    }

    printf("\n=== ncu 分析命令 ===\n");
    printf("ncu --kernel-name regex:gemmRegisterBlocking \\\n");
    printf("  --metrics \\\n");
    printf("    sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("    launch__registers_per_thread,\\\n");
    printf("    smsp__average_warps_issue_stalled_long_scoreboard.pct \\\n");
    printf("  ./gemm_timed\n");

    return 0;
}
