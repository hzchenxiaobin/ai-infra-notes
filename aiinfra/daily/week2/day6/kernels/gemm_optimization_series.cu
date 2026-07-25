// gemm_optimization_series.cu —— GEMM 优化全系列对比 (Week2 Day6)
// 包含 6 个优化版本 + cuBLAS 基线，逐层展示 GEMM 优化的收益来源:
//   v1 Naive            —— 每 thread 算一个 C[i][j]，纯 global 读
//   v2 SharedMem Tiling —— A/B tile 预取到 shared memory，K 维复用
//   v3 Register Blocking—— TM×TN thread tile，acc 驻留寄存器
//   v4 + float4 Load    —— 向量化 128-bit global→shared 加载
//   v5 Integrated       —— + float4 coalesced 写回
//   v6 Double Buffering —— 软件流水线，计算掩盖加载
// 编译命令: nvcc -O3 -arch=sm_120 gemm_optimization_series.cu -o gemm_series -lcublas
// 运行命令: ./gemm_series
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

// ============ 公共参数 ============
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))  // 256

#define CHECK_CUDA(call)                                                                   \
    do {                                                                                   \
        cudaError_t e = (call);                                                            \
        if (e != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// ============ v1: Naive GEMM (每 thread 算一个 C[i][j]) ============
__global__ void gemmNaive(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============ v2: Shared Memory Tiling (32×32 tile, 每 thread 算一个 C[i][j]) ============
#define SM_TILE 32
__global__ void gemmSharedMem(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[SM_TILE][SM_TILE];
    __shared__ float sB[SM_TILE][SM_TILE];

    int tx = threadIdx.x;  // 0..SM_TILE-1
    int ty = threadIdx.y;  // 0..SM_TILE-1
    int row = blockIdx.y * SM_TILE + ty;
    int col = blockIdx.x * SM_TILE + tx;
    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += SM_TILE) {
        // 协作加载 A tile (SM_TILE×SM_TILE): 每个 thread 加载一个元素
        sA[ty][tx] = (row < M && bk + tx < K) ? A[row * K + bk + tx] : 0.0f;
        // 协作加载 B tile (SM_TILE×SM_TILE): 每个 thread 加载一个元素
        sB[ty][tx] = (bk + ty < K && col < N) ? B[(bk + ty) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < SM_TILE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ============ v3: Register Blocking (TM×TN thread tile, acc 驻留寄存器) ============
__global__ void gemmRegisterBlocking(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rA[TM];
    float rB[TN];
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A tile (BM×BK): 每 thread 加载若干元素
        {
            int aRow = threadIdx.x / BK;
            int aCol = threadIdx.x % BK;
            #pragma unroll
            for (int i = 0; i < BM; i += NUM_THREADS / BK) {
                int loadRow = aRow + i;
                if (loadRow < BM) {
                    sA[loadRow][aCol] =
                        (cRow + loadRow < M && bk + aCol < K) ? A[(cRow + loadRow) * K + (bk + aCol)] : 0.0f;
                }
            }
        }
        // 协作加载 B tile (BK×BN)
        {
            int bRow = threadIdx.x / BN;
            int bCol = threadIdx.x % BN;
            #pragma unroll
            for (int i = 0; i < BK; i += NUM_THREADS / BN) {
                int loadRow = bRow + i;
                if (loadRow < BK) {
                    sB[loadRow][bCol] =
                        (bk + loadRow < K && cCol + bCol < N) ? B[(bk + loadRow) * N + (cCol + bCol)] : 0.0f;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) rA[m] = sA[threadRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) rB[n] = sB[k][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) acc[m][n] += rA[m] * rB[n];
            }
        }
        __syncthreads();
    }

    // 标量写回
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

// ============ v4: Register Blocking + float4 向量化加载 ============
__global__ void gemmRegisterBlockingF4(const float* __restrict__ A, const float* __restrict__ B,
                                       float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rA[TM];
    float rB[TN];
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A tile (BM×BK): float4 向量化 (BK=8 → 2 个 float4 / 行)
        {
            int aRow = threadIdx.x / (BK / 4);
            int aCol4 = threadIdx.x % (BK / 4);
            int aCol = aCol4 * 4;
            #pragma unroll
            for (int i = 0; i < BM; i += NUM_THREADS / (BK / 4)) {
                int loadRow = aRow + i;
                int gRow = cRow + loadRow;
                int gCol = bk + aCol;
                if (loadRow < BM && gRow < M && gCol + 3 < K) {
                    float4 val = reinterpret_cast<const float4*>(&A[gRow * K + gCol])[0];
                    sA[loadRow][aCol + 0] = val.x;
                    sA[loadRow][aCol + 1] = val.y;
                    sA[loadRow][aCol + 2] = val.z;
                    sA[loadRow][aCol + 3] = val.w;
                } else if (loadRow < BM) {
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        int gc = gCol + c;
                        sA[loadRow][aCol + c] = (gRow < M && gc < K) ? A[gRow * K + gc] : 0.0f;
                    }
                }
            }
        }
        // 协作加载 B tile (BK×BN): float4 向量化 (BN=128 → 32 个 float4 / 行)
        {
            int bRow = threadIdx.x / (BN / 4);
            int bCol4 = threadIdx.x % (BN / 4);
            int bCol = bCol4 * 4;
            #pragma unroll
            for (int i = 0; i < BK; i += NUM_THREADS / (BN / 4)) {
                int loadRow = bRow + i;
                int gRow = bk + loadRow;
                int gCol = cCol + bCol;
                if (loadRow < BK && gRow < K && gCol + 3 < N) {
                    float4 val = reinterpret_cast<const float4*>(&B[gRow * N + gCol])[0];
                    sB[loadRow][bCol + 0] = val.x;
                    sB[loadRow][bCol + 1] = val.y;
                    sB[loadRow][bCol + 2] = val.z;
                    sB[loadRow][bCol + 3] = val.w;
                } else if (loadRow < BK) {
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        int gc = gCol + c;
                        sB[loadRow][bCol + c] = (gRow < K && gc < N) ? B[gRow * N + gc] : 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) rA[m] = sA[threadRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) rB[n] = sB[k][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) acc[m][n] += rA[m] * rB[n];
            }
        }
        __syncthreads();
    }

    // 标量写回 (与 v3 相同，仅加载阶段向量化)
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

// ============ v5: Integrated (Register Blocking + float4 加载 + float4 coalesced 写回) ============
__global__ void gemmIntegrated(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rA[TM];
    float rB[TN];
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // float4 加载 A
        {
            int aRow = threadIdx.x / (BK / 4);
            int aCol4 = threadIdx.x % (BK / 4);
            int aCol = aCol4 * 4;
            #pragma unroll
            for (int i = 0; i < BM; i += NUM_THREADS / (BK / 4)) {
                int loadRow = aRow + i;
                int gRow = cRow + loadRow;
                int gCol = bk + aCol;
                if (loadRow < BM && gRow < M && gCol + 3 < K) {
                    float4 val = reinterpret_cast<const float4*>(&A[gRow * K + gCol])[0];
                    sA[loadRow][aCol + 0] = val.x;
                    sA[loadRow][aCol + 1] = val.y;
                    sA[loadRow][aCol + 2] = val.z;
                    sA[loadRow][aCol + 3] = val.w;
                } else if (loadRow < BM) {
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        int gc = gCol + c;
                        sA[loadRow][aCol + c] = (gRow < M && gc < K) ? A[gRow * K + gc] : 0.0f;
                    }
                }
            }
        }
        // float4 加载 B
        {
            int bRow = threadIdx.x / (BN / 4);
            int bCol4 = threadIdx.x % (BN / 4);
            int bCol = bCol4 * 4;
            #pragma unroll
            for (int i = 0; i < BK; i += NUM_THREADS / (BN / 4)) {
                int loadRow = bRow + i;
                int gRow = bk + loadRow;
                int gCol = cCol + bCol;
                if (loadRow < BK && gRow < K && gCol + 3 < N) {
                    float4 val = reinterpret_cast<const float4*>(&B[gRow * N + gCol])[0];
                    sB[loadRow][bCol + 0] = val.x;
                    sB[loadRow][bCol + 1] = val.y;
                    sB[loadRow][bCol + 2] = val.z;
                    sB[loadRow][bCol + 3] = val.w;
                } else if (loadRow < BK) {
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        int gc = gCol + c;
                        sB[loadRow][bCol + c] = (gRow < K && gc < N) ? B[gRow * N + gc] : 0.0f;
                    }
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) rA[m] = sA[threadRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) rB[n] = sB[k][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) acc[m][n] += rA[m] * rB[n];
            }
        }
        __syncthreads();
    }

    // float4 coalesced 写回 (TN=8 → 2 个 float4)
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int gRow = cRow + threadRow * TM + m;
        if (gRow < M) {
            int gCol = cCol + threadCol * TN;
            if (gCol + TN <= N) {
                #pragma unroll
                for (int n = 0; n < TN; n += 4) {
                    float4 val = make_float4(acc[m][n + 0], acc[m][n + 1], acc[m][n + 2], acc[m][n + 3]);
                    *reinterpret_cast<float4*>(&C[gRow * N + gCol + n]) = val;
                }
            } else {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    if (gCol + n < N) C[gRow * N + gCol + n] = acc[m][n];
                }
            }
        }
    }
}

// ============ v6: Double Buffering (软件流水线) ============
// 在 v5 基础上用双 shared buffer，让下一 tile 的加载与当前 tile 的计算重叠
// 用宏内联加载逻辑，避免 device lambda 捕获 shared memory 的可移植性问题
#define LOAD_TILE(buf, bk)                                                                       \
    do {                                                                                         \
        {                                                                                        \
            int aRow = threadIdx.x / (BK / 4);                                                   \
            int aCol4 = threadIdx.x % (BK / 4);                                                  \
            int aCol = aCol4 * 4;                                                                \
            _Pragma("unroll")                                                                    \
            for (int i = 0; i < BM; i += NUM_THREADS / (BK / 4)) {                               \
                int loadRow = aRow + i;                                                          \
                int gRow = cRow + loadRow;                                                       \
                int gCol = (bk) + aCol;                                                          \
                if (loadRow < BM && gRow < M && gCol + 3 < K) {                                  \
                    float4 val = reinterpret_cast<const float4*>(&A[gRow * K + gCol])[0];        \
                    sA[(buf)][loadRow][aCol + 0] = val.x;                                        \
                    sA[(buf)][loadRow][aCol + 1] = val.y;                                        \
                    sA[(buf)][loadRow][aCol + 2] = val.z;                                        \
                    sA[(buf)][loadRow][aCol + 3] = val.w;                                        \
                } else if (loadRow < BM) {                                                       \
                    _Pragma("unroll")                                                            \
                    for (int c = 0; c < 4; c++) {                                                \
                        int gc = gCol + c;                                                       \
                        sA[(buf)][loadRow][aCol + c] = (gRow < M && gc < K) ? A[gRow * K + gc] : 0.0f; \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        {                                                                                        \
            int bRow = threadIdx.x / (BN / 4);                                                   \
            int bCol4 = threadIdx.x % (BN / 4);                                                  \
            int bCol = bCol4 * 4;                                                                \
            _Pragma("unroll")                                                                    \
            for (int i = 0; i < BK; i += NUM_THREADS / (BN / 4)) {                               \
                int loadRow = bRow + i;                                                          \
                int gRow = (bk) + loadRow;                                                       \
                int gCol = cCol + bCol;                                                          \
                if (loadRow < BK && gRow < K && gCol + 3 < N) {                                  \
                    float4 val = reinterpret_cast<const float4*>(&B[gRow * N + gCol])[0];        \
                    sB[(buf)][loadRow][bCol + 0] = val.x;                                        \
                    sB[(buf)][loadRow][bCol + 1] = val.y;                                        \
                    sB[(buf)][loadRow][bCol + 2] = val.z;                                        \
                    sB[(buf)][loadRow][bCol + 3] = val.w;                                        \
                } else if (loadRow < BK) {                                                       \
                    _Pragma("unroll")                                                            \
                    for (int c = 0; c < 4; c++) {                                                \
                        int gc = gCol + c;                                                       \
                        sB[(buf)][loadRow][bCol + c] = (gRow < K && gc < N) ? B[gRow * N + gc] : 0.0f; \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    } while (0)

__global__ void gemmDoubleBuffer(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[2][BM][BK];
    __shared__ float sB[2][BK][BN];

    float rA[TM];
    float rB[TN];
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    int numTiles = (K + BK - 1) / BK;

    // Prologue: 预取第 0 个 tile 到 buf 0
    LOAD_TILE(0, 0);
    __syncthreads();

    for (int t = 0; t < numTiles; t++) {
        int bk = t * BK;
        int curBuf = t & 1;
        int nxtBuf = 1 - curBuf;

        // 预取下一 tile (如果有) —— 与当前 tile 的计算重叠
        if (t + 1 < numTiles) {
            LOAD_TILE(nxtBuf, (t + 1) * BK);
        }
        // 计算当前 tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) rA[m] = sA[curBuf][threadRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) rB[n] = sB[curBuf][k][threadCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) acc[m][n] += rA[m] * rB[n];
            }
        }
        __syncthreads();  // 等下一 tile 加载完且当前 tile 算完，才能覆盖
    }

    // float4 coalesced 写回
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int gRow = cRow + threadRow * TM + m;
        if (gRow < M) {
            int gCol = cCol + threadCol * TN;
            if (gCol + TN <= N) {
                #pragma unroll
                for (int n = 0; n < TN; n += 4) {
                    float4 val = make_float4(acc[m][n + 0], acc[m][n + 1], acc[m][n + 2], acc[m][n + 3]);
                    *reinterpret_cast<float4*>(&C[gRow * N + gCol + n]) = val;
                }
            } else {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    if (gCol + n < N) C[gRow * N + gCol + n] = acc[m][n];
                }
            }
        }
    }
}
#undef LOAD_TILE

// ============ Host 辅助函数 ============
void initMatrix(float* mat, int rows, int cols) {
    srand(42);
    for (int i = 0; i < rows * cols; i++)
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps) {
            printf("  First mismatch at %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

float getGFLOPS(int M, int N, int K, float ms) {
    return 2.0f * M * N * K / (ms * 1e6);
}

float runCuBLAS(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaDeviceSynchronize();

    cudaEvent_t st, en;
    cudaEventCreate(&st);
    cudaEventCreate(&en);
    cudaEventRecord(st);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaEventRecord(en);
    cudaEventSynchronize(en);
    float ms;
    cudaEventElapsedTime(&ms, st, en);
    cublasDestroy(handle);
    cudaEventDestroy(st);
    cudaEventDestroy(en);
    return ms;
}

// 统一的 kernel 计时封装
template <typename KernelFunc>
float runKernel(KernelFunc kernel, const float* dA, const float* dB, float* dC, int M, int N, int K,
                dim3 grid, dim3 block) {
    kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t st, en;
    cudaEventCreate(&st);
    cudaEventCreate(&en);
    cudaEventRecord(st);
    kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(en);
    CHECK_CUDA(cudaEventSynchronize(en));
    float ms;
    cudaEventElapsedTime(&ms, st, en);
    cudaEventDestroy(st);
    cudaEventDestroy(en);
    return ms;
}

int main() {
    int sizes[][3] = {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}};
    int numSizes = 3;

    printf("=== GEMM Optimization Series (RTX 5090, sm_120) ===\n");
    printf("BM=%d BN=%d BK=%d TM=%d TN=%d Threads=%d\n\n", BM, BN, BK, TM, TN, NUM_THREADS);
    printf("  M      N      K      | Naive     SharedMem | RegBlk    RegF4     Integ     DblBuf    cuBLAS    verify\n");
    printf("  -------------------------------------------------------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        size_t sa = (size_t)M * K * sizeof(float);
        size_t sb = (size_t)K * N * sizeof(float);
        size_t sc = (size_t)M * N * sizeof(float);

        float *hA = (float*)malloc(sa), *hB = (float*)malloc(sb);
        float *hC = (float*)malloc(sc), *hRef = (float*)malloc(sc);
        initMatrix(hA, M, K);
        initMatrix(hB, K, N);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, sa));
        CHECK_CUDA(cudaMalloc(&dB, sb));
        CHECK_CUDA(cudaMalloc(&dC, sc));
        CHECK_CUDA(cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice));

        dim3 gridTiled((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 blockTiled(NUM_THREADS);
        dim3 gridNaive((N + 15) / 16, (M + 15) / 16);
        dim3 blockNaive(16, 16);
        dim3 gridShared((N + SM_TILE - 1) / SM_TILE, (M + SM_TILE - 1) / SM_TILE);
        dim3 blockShared(SM_TILE, SM_TILE);

        // cuBLAS 基线
        float msCu = runCuBLAS(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaMemcpy(hRef, dC, sc, cudaMemcpyDeviceToHost));

        // v1 Naive
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        float msNaive = runKernel(gemmNaive, dA, dB, dC, M, N, K, gridNaive, blockNaive);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool ok1 = checkResult(hC, hRef, M * N, 1e-1f);

        // v2 SharedMem
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        float msShared = runKernel(gemmSharedMem, dA, dB, dC, M, N, K, gridShared, blockShared);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool ok2 = checkResult(hC, hRef, M * N, 1e-2f);

        // v3 RegisterBlocking
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        float msReg = runKernel(gemmRegisterBlocking, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool ok3 = checkResult(hC, hRef, M * N, 1e-2f);

        // v4 RegF4
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        float msRegF4 = runKernel(gemmRegisterBlockingF4, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool ok4 = checkResult(hC, hRef, M * N, 1e-2f);

        // v5 Integrated
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        float msInteg = runKernel(gemmIntegrated, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool ok5 = checkResult(hC, hRef, M * N, 1e-2f);

        // v6 DoubleBuffer
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        float msDbl = runKernel(gemmDoubleBuffer, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool ok6 = checkResult(hC, hRef, M * N, 1e-2f);

        bool allOk = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
        printf("%-6d %-6d %-6d | %-10.3f %-10.3f | %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f %s\n",
               M, N, K, msNaive, msShared, msReg, msRegF4, msInteg, msDbl, msCu,
               allOk ? "PASS" : "FAIL");

        free(hA); free(hB); free(hC); free(hRef);
        CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    }

    // ============ 第二部分: cuBLAS 百分比汇总 ============
    printf("\n=== cuBLAS %% (Our TFLOPS / cuBLAS TFLOPS) ===\n");
    printf("%-6s %-6s %-6s | %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "M", "N", "K", "Naive", "Shared", "RegBlk", "RegF4", "Integ", "DblBuf");
    printf("-------------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        size_t sa = (size_t)M * K * sizeof(float);
        size_t sb = (size_t)K * N * sizeof(float);
        size_t sc = (size_t)M * N * sizeof(float);

        float *hA = (float*)malloc(sa), *hB = (float*)malloc(sb);
        initMatrix(hA, M, K);
        initMatrix(hB, K, N);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, sa));
        CHECK_CUDA(cudaMalloc(&dB, sb));
        CHECK_CUDA(cudaMalloc(&dC, sc));
        CHECK_CUDA(cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice));

        dim3 gridTiled((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 blockTiled(NUM_THREADS);
        dim3 gridNaive((N + 15) / 16, (M + 15) / 16);
        dim3 blockNaive(16, 16);
        dim3 gridShared((N + SM_TILE - 1) / SM_TILE, (M + SM_TILE - 1) / SM_TILE);
        dim3 blockShared(SM_TILE, SM_TILE);

        float msCu = runCuBLAS(dA, dB, dC, M, N, K);
        float msNaive = runKernel(gemmNaive, dA, dB, dC, M, N, K, gridNaive, blockNaive);
        float msShared = runKernel(gemmSharedMem, dA, dB, dC, M, N, K, gridShared, blockShared);
        float msReg = runKernel(gemmRegisterBlocking, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        float msRegF4 = runKernel(gemmRegisterBlockingF4, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        float msInteg = runKernel(gemmIntegrated, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        float msDbl = runKernel(gemmDoubleBuffer, dA, dB, dC, M, N, K, gridTiled, blockTiled);

        printf("%-6d %-6d %-6d | %-7.1f%% %-7.1f%% %-7.1f%% %-7.1f%% %-7.1f%% %-7.1f%%\n",
               M, N, K, 100 * msCu / msNaive, 100 * msCu / msShared, 100 * msCu / msReg,
               100 * msCu / msRegF4, 100 * msCu / msInteg, 100 * msCu / msDbl);

        free(hA); free(hB);
        CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    }

    printf("\n=== TFLOPS Summary ===\n");
    printf("%-6s %-6s %-6s | %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "M", "N", "K", "Naive", "Shared", "RegBlk", "RegF4", "Integ", "DblBuf", "cuBLAS");
    printf("----------------------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];
        size_t sa = (size_t)M * K * sizeof(float);
        size_t sb = (size_t)K * N * sizeof(float);
        size_t sc = (size_t)M * N * sizeof(float);

        float *hA = (float*)malloc(sa), *hB = (float*)malloc(sb);
        initMatrix(hA, M, K);
        initMatrix(hB, K, N);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, sa));
        CHECK_CUDA(cudaMalloc(&dB, sb));
        CHECK_CUDA(cudaMalloc(&dC, sc));
        CHECK_CUDA(cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice));

        dim3 gridTiled((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 blockTiled(NUM_THREADS);
        dim3 gridNaive((N + 15) / 16, (M + 15) / 16);
        dim3 blockNaive(16, 16);
        dim3 gridShared((N + SM_TILE - 1) / SM_TILE, (M + SM_TILE - 1) / SM_TILE);
        dim3 blockShared(SM_TILE, SM_TILE);

        float msCu = runCuBLAS(dA, dB, dC, M, N, K);
        float msNaive = runKernel(gemmNaive, dA, dB, dC, M, N, K, gridNaive, blockNaive);
        float msShared = runKernel(gemmSharedMem, dA, dB, dC, M, N, K, gridShared, blockShared);
        float msReg = runKernel(gemmRegisterBlocking, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        float msRegF4 = runKernel(gemmRegisterBlockingF4, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        float msInteg = runKernel(gemmIntegrated, dA, dB, dC, M, N, K, gridTiled, blockTiled);
        float msDbl = runKernel(gemmDoubleBuffer, dA, dB, dC, M, N, K, gridTiled, blockTiled);

        printf("%-6d %-6d %-6d | %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f\n",
               M, N, K, getGFLOPS(M, N, K, msNaive) / 1000, getGFLOPS(M, N, K, msShared) / 1000,
               getGFLOPS(M, N, K, msReg) / 1000, getGFLOPS(M, N, K, msRegF4) / 1000,
               getGFLOPS(M, N, K, msInteg) / 1000, getGFLOPS(M, N, K, msDbl) / 1000,
               getGFLOPS(M, N, K, msCu) / 1000);

        free(hA); free(hB);
        CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    }

    return 0;
}
