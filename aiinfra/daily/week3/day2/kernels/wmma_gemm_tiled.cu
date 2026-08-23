// wmma_gemm_tiled.cu —— Day 2: Shared Memory Tiling 版 WMMA GEMM
// FP16 输入 / FP32 累加，含 Day 1 naive 版与 cuBLAS 三方对比 + 正确性验证
//
// 编译命令: nvcc -O3 -arch=sm_120 -lcublas wmma_gemm_tiled.cu -o wmma_tiled
//
// ⚠️ 本文件由 README 内嵌代码补全而来（host 端初始化/验证/计时为补齐部分），
//    本机无 GPU，未编译实测——实测数据待 GPU 环境回填。
//    README 中的性能表（4096: tiled 8.12ms / TF32 cuBLAS 16%）为此前
//    RTX 5090 实测留档（注意：该数据为修复前代码的实测，存在 warp 偏移
//    和线程索引 bug，修复后性能与精度均会变化）。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

// Tiling 配置
#define BM 64       // Block 输出 tile 行
#define BN 64       // Block 输出 tile 列
#define BK 16       // K 维分块
#define WM 32       // Warp 输出 tile 行
#define WN 32       // Warp 输出 tile 列
#define WARP_SIZE 32

// Shared memory padding（消除 bank conflict）
#define SMEM_PAD 8   // FP16 padding 8 元素 = 16 字节

// ---------- Day 1 naive 版（每 block 1 warp，直接从 global memory load fragment） ----------
// 用于同文件对比，替代原计划的 wmma_gemm_compare.cu
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm_naive_kernel(
    const __half* __restrict__ A,    // M×K, row-major
    const __half* __restrict__ B,    // K×N, row-major（本文件统一用 row-major）
    float* __restrict__ C,           // M×N, row-major
    int M, int N, int K)
{
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        const __half* tileA = A + warpM * WMMA_M * K + k;
        const __half* tileB = B + k * N + warpN * WMMA_N;   // B row-major: ld = N
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* tileC = C + warpM * WMMA_M * N + warpN * WMMA_N;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

// ---------- Day 2 tiled 版（4 warp/block 协作，smem tiling + padding） ----------
__global__ void wmma_gemm_tiled_kernel(
    const __half* __restrict__ A,    // M×K, row-major
    const __half* __restrict__ B,    // K×N, row-major
    float* __restrict__ C,           // M×N, row-major
    int M, int N, int K)
{
    // 每个 block 含 4 个 warp (128 线程)
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_x = warp_id % 2;   // warp 在 BN 方向的索引 (0 或 1)
    int warp_y = warp_id / 2;   // warp 在 BM 方向的索引 (0 或 1)

    // Block 负责的 C 子矩阵起始位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Shared memory：A tile (BM×BK) + B tile (BK×BN)，带 padding
    __shared__ __half smemA[BM][BK + SMEM_PAD];
    __shared__ __half smemB[BK][BN + SMEM_PAD];

    // 声明 WMMA fragment
    // 每个 warp 计算 32×32 = 2×2 个 16×16 MMA
    // 注意：A 在 smem 中 row-major；B 从 row-major global 加载，
    //   在 smem 中按 B[k][n] 存放（k 行 n 列，即 row-major），
    //   故 b_frag 声明为 row_major，与 smem 布局一致
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // K 维循环
    for (int k = 0; k < K; k += BK) {
        // 1. 4 warp 协作加载 A/B tile 到 shared memory
        //    每个 warp 负责加载 1/4 的数据
        // A tile: BM×BK = 64×16, 4 warp 各加载 16 行
        int load_row = warp_id * 16;
        for (int i = 0; i < 16; i++) {
            for (int j = threadIdx.x % WARP_SIZE; j < BK; j += WARP_SIZE) {
                int global_row = block_row + load_row + i;
                if (global_row < M && (k + j) < K) {
                    smemA[load_row + i][j] = A[global_row * K + k + j];
                }
            }
        }
        // B tile: BK×BN = 16×64, 4 warp 各加载 16 列
        // 按 B[k][n] 存放（k 行 n 列，即 row-major），b_frag 声明为 row_major
        int load_col = warp_id * 16;
        for (int i = threadIdx.x % WARP_SIZE; i < BK; i += WARP_SIZE) {
            for (int j = 0; j < 16; j++) {
                int global_col = block_col + load_col + j;
                if ((k + i) < K && global_col < N) {
                    smemB[i][load_col + j] = B[(k + i) * N + global_col];
                }
            }
        }
        __syncthreads();

        // 2. 每个 warp 从 smem 加载 fragment 并执行 MMA
        //    warp 负责 32×32 输出 = 2×2 个 16×16 MMA
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i],
                &smemA[warp_y * WM + i * 16][0], BK + SMEM_PAD);
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j],
                &smemB[0][warp_x * WN + j * 16], BN + SMEM_PAD);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        __syncthreads();
    }

    // 3. 存储结果到 C
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int row = block_row + warp_y * WM + i * 16;
            int col = block_col + warp_x * WN + j * 16;
            if (row < M && col < N) {
                wmma::store_matrix_sync(C + row * N + col,
                    c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// ---------- Host 端：计时工具 ----------
static float time_kernel(void (*launch)(const __half*, const __half*, float*, int, int, int),
                         const __half* d_A, const __half* d_B, float* d_C,
                         int M, int N, int K)
{
    // warmup
    launch(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 10; iter++) {
        launch(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 10.0f;
}

static void launch_naive(const __half* A, const __half* B, float* C,
                         int M, int N, int K)
{
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32, 1);
    wmma_gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

static void launch_tiled(const __half* A, const __half* B, float* C,
                         int M, int N, int K)
{
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);  // 4 warp = 128 线程
    wmma_gemm_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// ---------- Host 端：主流程（数据准备 → 三方对比 → 正确性验证） ----------
int main(int argc, char** argv)
{
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // TF32 模式：cuBLAS FP32 输入启用 TF32 Tensor Core（本课程的基准口径）
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("M=N=K    | naive_ms   tiled_ms   TF32cub_ms  | naive%%TF32  tiled%%TF32  max_diff\n");
    printf("---------|-----------------------------------------|-------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];

        // --- 1. Host 端准备数据（A/B 均 row-major） ---
        __half* h_A = (__half*)malloc(M * K * sizeof(__half));
        __half* h_B = (__half*)malloc(K * N * sizeof(__half));
        srand(42);
        for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 100) / 100.0f);
        for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(rand() % 100) / 100.0f);

        // --- 2. Device 端分配 + 拷贝 ---
        __half *d_A, *d_B;
        float *d_C, *d_C_ref;
        cudaMalloc(&d_A, M * K * sizeof(__half));
        cudaMalloc(&d_B, K * N * sizeof(__half));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMalloc(&d_C_ref, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(__half), cudaMemcpyHostToDevice);

        // --- 3. 三个实现计时 ---
        float naive_ms = time_kernel(launch_naive, d_A, d_B, d_C, M, N, K);
        float tiled_ms = time_kernel(launch_tiled, d_A, d_B, d_C, M, N, K);

        // cuBLAS TF32 参考（FP32 输入 + TF32 Tensor Core）
        // cuBLAS 列主序：C_row = A_row @ B_row 等价于 C_col = B_row^T @ A_row^T，
        // 即 cublasSgemm(OP_N, OP_N, N, M, K, B, A)
        float *h_Af = (float*)malloc(M * K * sizeof(float));
        float *h_Bf = (float*)malloc(K * N * sizeof(float));
        for (int i = 0; i < M * K; i++) h_Af[i] = __half2float(h_A[i]);
        for (int i = 0; i < K * N; i++) h_Bf[i] = __half2float(h_B[i]);
        float *d_Af, *d_Bf;
        cudaMalloc(&d_Af, M * K * sizeof(float));
        cudaMalloc(&d_Bf, K * N * sizeof(float));
        cudaMemcpy(d_Af, h_Af, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bf, h_Bf, K * N * sizeof(float), cudaMemcpyHostToDevice);

        float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // warmup
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_Bf, N, d_Af, K, &beta, d_C_ref, N);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, d_Bf, N, d_Af, K, &beta, d_C_ref, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cub_ms = 0;
        cudaEventElapsedTime(&cub_ms, start, stop);
        cub_ms /= 10.0f;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // --- 4. 正确性验证（tiled vs cuBLAS TF32，max_diff） ---
        launch_tiled(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        float* h_C = (float*)malloc(M * N * sizeof(float));
        float* h_C_ref = (float*)malloc(M * N * sizeof(float));
        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        float max_diff = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float diff = fabsf(h_C[i] - h_C_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("%-8d | %-10.4f %-10.4f %-10.4f | %-10.1f %-10.1f %.2e\n",
               M, naive_ms, tiled_ms, cub_ms,
               100.0f * cub_ms / naive_ms,   // 时间比 = 性能比
               100.0f * cub_ms / tiled_ms,
               max_diff);

        // --- 5. 释放资源 ---
        free(h_A); free(h_B); free(h_Af); free(h_Bf); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
        cudaFree(d_Af); cudaFree(d_Bf);
    }

    cublasDestroy(handle);
    return 0;
}
