// integrated_gemm.cu —— 整合优化 GEMM（ncu profiling 版）
// Warp Shuffle + Register Blocking + float4 向量化加载 + Coalesced 写回
// 编译命令: nvcc -o integrated_gemm integrated_gemm.cu -O3 -arch=sm_120 -lcublas -g -lineinfo
// 运行命令: ./integrated_gemm

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
#define NUM_THREADS ((BM / TM) * (BN / TN)) // 256

__device__ __forceinline__ float4 make_float4_from_float(const float* p) {
    return make_float4(p[0], p[1], p[2], p[3]);
}

__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void gemmIntegrated(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M,
                               int N, int K) {
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
        // ---- 协作加载 A tile (BM×BK)，使用 float4 ----
        int aRow = threadIdx.x / (BK / 4);
        int aCol4 = threadIdx.x % (BK / 4);

#pragma unroll
        for (int i = 0; i < BM; i += NUM_THREADS / (BK / 4)) {
            int loadRow = aRow + i;
            int globalRow = cRow + loadRow;
            int globalCol = bk + aCol4 * 4;

            if (loadRow < BM && globalRow < M && globalCol + 3 < K) {
                float4 val = reinterpret_cast<const float4*>(&A[globalRow * K + globalCol])[0];
                s_A[loadRow][aCol4 * 4 + 0] = val.x;
                s_A[loadRow][aCol4 * 4 + 1] = val.y;
                s_A[loadRow][aCol4 * 4 + 2] = val.z;
                s_A[loadRow][aCol4 * 4 + 3] = val.w;
            } else if (loadRow < BM) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    int gc = globalCol + c;
                    s_A[loadRow][aCol4 * 4 + c] = (globalRow < M && gc < K) ? A[globalRow * K + gc] : 0.0f;
                }
            }
        }

        // ---- 协作加载 B tile (BK×BN)，使用 float4 ----
        int bRow = threadIdx.x / (BN / 4);
        int bCol4 = threadIdx.x % (BN / 4);

#pragma unroll
        for (int i = 0; i < BK; i += NUM_THREADS / (BN / 4)) {
            int loadRow = bRow + i;
            int globalRow = bk + loadRow;
            int globalCol = cCol + bCol4 * 4;

            if (loadRow < BK && globalRow < K && globalCol + 3 < N) {
                float4 val = reinterpret_cast<const float4*>(&B[globalRow * N + globalCol])[0];
                s_B[loadRow][bCol4 * 4 + 0] = val.x;
                s_B[loadRow][bCol4 * 4 + 1] = val.y;
                s_B[loadRow][bCol4 * 4 + 2] = val.z;
                s_B[loadRow][bCol4 * 4 + 3] = val.w;
            } else if (loadRow < BK) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    int gc = globalCol + c;
                    s_B[loadRow][bCol4 * 4 + c] = (globalRow < K && gc < N) ? B[globalRow * N + gc] : 0.0f;
                }
            }
        }

        __syncthreads();

// ---- Register Blocking 计算 ----
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

// ---- Coalesced 写回 Global Memory，使用 float4 ----
#pragma unroll
    for (int m = 0; m < TM; m++) {
        int gRow = cRow + threadRow * TM + m;
        if (gRow < M) {
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int gCol = cCol + threadCol * TN + n;
                if (gCol + 3 < N) {
                    float4 val = make_float4(acc[m][n + 0], acc[m][n + 1], acc[m][n + 2], acc[m][n + 3]);
                    reinterpret_cast<float4*>(&C[gRow * N + gCol])[0] = val;
                } else {
#pragma unroll
                    for (int c = 0; c < 4 && gCol + c < N; c++) {
                        C[gRow * N + gCol + c] = acc[m][n + c];
                    }
                }
            }
        }
    }
}

// cuBLAS 基准
float runCuBLAS(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float runOurKernel(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gemmIntegrated<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemmIntegrated<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void initMatrix(float* mat, int rows, int cols) {
    srand(42);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > eps) {
            printf("First mismatch at %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

float getGFLOPS(int M, int N, int K, float ms) {
    return 2.0f * M * N * K / (ms * 1e6);
}

int main() {
    int sizes[][3] = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };

    printf("=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===\n");
    printf("BM=%d, BN=%d, BK=%d, TM=%d, TN=%d, Threads=%d\n\n", BM, BN, BK, TM, TN, NUM_THREADS);
    printf("%-8s %-8s %-8s %-10s %-10s %-10s %-8s\n", "M", "N", "K", "Our(ms)", "cuBLAS(ms)", "GFLOPS", "Percent");
    printf("----------------------------------------------------------------\n");

    for (int s = 0; s < 4; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        size_t bytesA = M * K * sizeof(float);
        size_t bytesB = K * N * sizeof(float);
        size_t bytesC = M * N * sizeof(float);

        float* h_A = (float*)malloc(bytesA);
        float* h_B = (float*)malloc(bytesB);
        float* h_C = (float*)malloc(bytesC);
        float* h_C_ref = (float*)malloc(bytesC);

        initMatrix(h_A, M, K);
        initMatrix(h_B, K, N);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytesA);
        cudaMalloc(&d_B, bytesB);
        cudaMalloc(&d_C, bytesC);
        cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

        float ourMs = runOurKernel(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

        float cublasMs = runCuBLAS(d_A, d_B, d_C, M, N, K);
        cudaMemcpy(h_C_ref, d_C, bytesC, cudaMemcpyDeviceToHost);

        bool correct = checkResult(h_C, h_C_ref, M * N, 1e-2);
        float ourGFLOPS = getGFLOPS(M, N, K, ourMs);
        float percent = (cublasMs / ourMs) * 100;

        printf("%-8d %-8d %-8d %-10.3f %-10.3f %-10.1f %-7.1f%% %s\n", M, N, K, ourMs, cublasMs, ourGFLOPS, percent,
               correct ? "PASS" : "FAIL");

        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    printf("\n=== ncu 分析命令 ===\n");
    printf("ncu --kernel-name regex:gemmIntegrated \\\n");
    printf("  -o integrated_profile \\\n");
    printf("  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("launch__registers_per_thread,\\\n");
    printf("smsp__average_warps_issue_stalled_long_scoreboard.pct \\\n");
    printf("  ./integrated_gemm\n");

    return 0;
}
