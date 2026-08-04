// integrated_gemm_leetgpu.cu —— 整合版 GEMM (Week2 Day6) 适配 LeetGPU 平台
// 在 Register Blocking + float4 向量化加载 + Coalesced 写回的基础上：
//   - 输入 A/B 为 half，累加在 float（满足题目 FP32 累加要求）
//   - epilogue 处理 C = alpha * (A@B) + beta * C
//   - 签名: solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
// 对应题目: https://leetgpu.com/challenges/general-matrix-multiplication-gemm
// 编译命令: nvcc -O3 -arch=sm_120 integrated_gemm_leetgpu.cu -o integrated_gemm_leetgpu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))

__global__ void gemmIntegrated(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                               int M, int N, int K, float alpha, float beta) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rA[TM];
    float rB[TN];
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // ---- 协作加载 A tile (BM×BK): half -> float, 每线程 4 halfs (2×__half2) ----
        {
            int row = threadIdx.x / (BK / 4);
            int col4 = threadIdx.x % (BK / 4);
            int col = col4 * 4;
            int gRow = cRow + row;
            int gCol = bk + col;
            if (gRow < M && gCol + 3 < K) {
                const __half2* p = reinterpret_cast<const __half2*>(&A[gRow * K + gCol]);
                float2 f0 = __half22float2(p[0]);
                float2 f1 = __half22float2(p[1]);
                sA[row][col + 0] = f0.x;
                sA[row][col + 1] = f0.y;
                sA[row][col + 2] = f1.x;
                sA[row][col + 3] = f1.y;
            } else {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    int gc = gCol + c;
                    sA[row][col + c] = (gRow < M && gc < K) ? __half2float(A[gRow * K + gc]) : 0.0f;
                }
            }
        }

        // ---- 协作加载 B tile (BK×BN): half -> float, 每线程 4 halfs (2×__half2) ----
        {
            int row = threadIdx.x / (BN / 4);
            int col4 = threadIdx.x % (BN / 4);
            int col = col4 * 4;
            int gRow = bk + row;
            int gCol = cCol + col;
            if (gRow < K && gCol + 3 < N) {
                const __half2* p = reinterpret_cast<const __half2*>(&B[gRow * N + gCol]);
                float2 f0 = __half22float2(p[0]);
                float2 f1 = __half22float2(p[1]);
                sB[row][col + 0] = f0.x;
                sB[row][col + 1] = f0.y;
                sB[row][col + 2] = f1.x;
                sB[row][col + 3] = f1.y;
            } else {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    int gc = gCol + c;
                    sB[row][col + c] = (gRow < K && gc < N) ? __half2float(B[gRow * N + gc]) : 0.0f;
                }
            }
        }

        __syncthreads();

        // ---- Register Blocking: 每线程算 TM×TN 个输出 ----
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++)
                rA[m] = sA[threadRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                rB[n] = sB[k][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += rA[m] * rB[n];
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: alpha*acc + beta*C_init -> half, Coalesced 写回 (8 halfs = 128-bit) ----
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int gRow = cRow + threadRow * TM + m;
        if (gRow < M) {
            int gCol = cCol + threadCol * TN;
            if (gCol + TN <= N) {
                float4 outv;
                __half2* h2 = reinterpret_cast<__half2*>(&outv);
                #pragma unroll
                for (int n = 0; n < TN; n += 2) {
                    float ci0 = (beta != 0.0f) ? __half2float(C[gRow * N + gCol + n]) : 0.0f;
                    float ci1 = (beta != 0.0f) ? __half2float(C[gRow * N + gCol + n + 1]) : 0.0f;
                    h2[n / 2] = __floats2half2_rn(alpha * acc[m][n] + beta * ci0,
                                                   alpha * acc[m][n + 1] + beta * ci1);
                }
                *reinterpret_cast<float4*>(&C[gRow * N + gCol]) = outv;
            } else {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    if (gCol + n < N) {
                        float ci = (beta != 0.0f) ? __half2float(C[gRow * N + gCol + n]) : 0.0f;
                        C[gRow * N + gCol + n] = __float2half(alpha * acc[m][n] + beta * ci);
                    }
                }
            }
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    gemmIntegrated<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
