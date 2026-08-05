// wmma_gemm.cu —— Tensor Core (WMMA) GEMM: FP16 输入, FP32 累加
// 编译命令: nvcc -O3 -arch=sm_120 wmma_gemm.cu -o wmma_gemm

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

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
    // 1. 根据 block 索引确定本 warp 负责的输出 tile 位置
    int warpM = blockIdx.y;  // tile 行索引
    int warpN = blockIdx.x;  // tile 列索引

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // 2. 声明 fragment（编译时确定形状和精度）
    //    A: FP16 row-major, B: FP16 col-major, C: FP32 累加器
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 3. 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 4. K 维循环：每次加载 16×16 的 A/B tile 并执行 MMA
    for (int k = 0; k < K; k += WMMA_K) {
        // A tile 起始地址：第 warpM*16 行、第 k 列（row-major，ld=K）
        const __half* tileA = A + warpM * WMMA_M * K + k;
        // B tile 起始地址：第 k 行、第 warpN*16 列（col-major，ld=K）
        const __half* tileB = B + k + warpN * WMMA_N * K;

        // 从 global memory 加载 fragment（warp 内 32 线程协作）
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, K);

        // 执行矩阵乘加：C = A × B + C（Tensor Core 硬件执行）
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 5. 存储结果到 C（row-major，ld=N）
    float* tileC = C + warpM * WMMA_M * N + warpN * WMMA_N;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

// ---------- Host 端：数据准备 + kernel launch + 计时 ----------
int main(int argc, char** argv)
{
    int M = 4096, N = 4096, K = 4096;

    // --- 1. Host 端准备数据 ---
    // FP16 数据：A row-major, B col-major
    __half *h_A = (__half*)malloc(M * K * sizeof(__half));
    __half *h_B = (__half*)malloc(K * N * sizeof(__half));
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 100) / 100.0f);
    // B col-major: B_col[k + n*K] = B_row[k*N + n]
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            h_B[k + n * K] = __float2half((float)(rand() % 100) / 100.0f);

    // --- 2. Device 端分配 + 拷贝 ---
    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__half), cudaMemcpyHostToDevice);

    // --- 3. WMMA kernel launch ---
    // Grid: (N/16, M/16)，每个 block = 1 warp (32 threads)
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32, 1);

    // warmup
    wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 计时（10 次取平均）
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 10; iter++)
        wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;

    // --- 4. 输出结果 ---
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    printf("WMMA GEMM %dx%dx%d: %.3f ms, %.1f TFLOPS\n", M, N, K, ms, tflops);

    // --- 5. 释放资源 ---
    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
