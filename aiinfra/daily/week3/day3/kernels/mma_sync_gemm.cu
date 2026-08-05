// mma_sync_gemm.cu —— Day 3: mma.sync + ldmatrix PTX 级 GEMM
// FP16 输入 / FP32 累加，每 block 1 warp 计算 16×8 输出 tile，
// 与 cuBLAS(TF32) 对比并做正确性验证。
//
// 编译命令: nvcc -O3 -arch=sm_120 -lcublas mma_sync_gemm.cu -o mma_sync_gemm
//
// ⚠️ 本文件由 README 3.4 节的"概念示意"代码补全而来（smem 加载、写回、
//    host 端为补齐部分），本机无 GPU，未编译实测——实测数据待 GPU 环境回填。
//    README 中原"预期输出"性能表为预估口径，不作为实测引用。
//
// fragment 线程-数据映射（m16n8k16, PTX ISA 固定）：
//   线程 tid: groupID = tid/4, tig = tid%4
//   A fragment (16×16, 4 regs): a0/a1/a2/a3 分别对应
//     (row=groupID, col=2*tig..+1), (row=groupID+8, 同列),
//     (row=groupID, col=2*tig+8..+9), (row=groupID+8, 同列)
//   B fragment (16×8, 2 regs): b0 = (k=2*tig..+1, n=groupID), b1 = k+8
//   C/D (16×8, 4 floats): c0/c1 = (row=groupID, col=2*tig..+1),
//     c2/c3 = (row=groupID+8, 同列)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#define TILE_M 16
#define TILE_N 8
#define TILE_K 16
#define PAD 8   // FP16 padding：对齐 16 字节 + 缓解 bank conflict

// ldmatrix.x4：加载 A fragment（4 个 8×8 矩阵 = 16×16）
// 线程地址分工：t0-7 → 左上 8×8 的行 0-7；t8-15 → 左下；t16-23 → 右上；t24-31 → 右下
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const __half* base, int ld)
{
    int tid = threadIdx.x;
    const __half* addr = base + (tid % 16) * ld + (tid / 16) * 8;
    uint32_t smem_addr = __cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr));
}

// ldmatrix.x2.trans：加载 B fragment（2 个 8×8 = 16(k)×8(n)）
// B 在 smem 中按 [k][n] 存放（row-major），.trans 在加载时转置成
// mma.sync 要求的 col-major fragment
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const __half* base, int ld)
{
    int tid = threadIdx.x;
    const __half* addr = base + (tid % 16) * ld;   // t0-15 提供 16 个 k 行地址
    uint32_t smem_addr = __cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_addr));
}

// mma.sync m16n8k16: D = A × B + C（FP16 输入, FP32 累加）
__device__ __forceinline__ void mma_sync_f32(float (&c)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__global__ void mma_sync_gemm_kernel(
    const __half* __restrict__ A,    // M×K, row-major
    const __half* __restrict__ B,    // K×N, row-major
    float* __restrict__ C,           // M×N, row-major
    int M, int N, int K)
{
    // Shared memory：A tile (16×16) + B tile (16×8)，16 字节对齐
    __shared__ __half smemA[TILE_M][TILE_K + PAD];
    __shared__ __half smemB[TILE_K][TILE_N + PAD];

    uint32_t a_reg[4];
    uint32_t b_reg[2];
    float c_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int tid = threadIdx.x;           // block = 32 线程（1 warp）
    int row = blockIdx.y * TILE_M;
    int col = blockIdx.x * TILE_N;

    for (int k = 0; k < K; k += TILE_K) {
        // 1. 加载 A/B tile 到 shared memory
        //    A: 16×16 = 256 halfs，32 线程各加载 8 个（一行的一半）
        {
            int r = tid / 2, c0 = (tid % 2) * 8;
            for (int j = 0; j < 8; j++) {
                smemA[r][c0 + j] = A[(row + r) * K + k + c0 + j];
            }
        }
        //    B: 16×8 = 128 halfs，32 线程各加载 4 个，按 [k][n] 存放
        {
            int r = tid / 2, c0 = (tid % 2) * 4;
            for (int j = 0; j < 4; j++) {
                smemB[r][c0 + j] = B[(k + r) * N + col + c0 + j];
            }
        }
        __syncthreads();

        // 2. ldmatrix 加载 fragment（A 用 x4，B 用 x2.trans）
        ldmatrix_x4(a_reg, &smemA[0][0], TILE_K + PAD);
        ldmatrix_x2_trans(b_reg, &smemB[0][0], TILE_N + PAD);

        // 3. mma.sync 执行矩阵乘加
        mma_sync_f32(c_reg, a_reg, b_reg);

        __syncthreads();
    }

    // 4. 存储结果：c_reg 的 4 个 float 按 fragment 布局写回
    //    c0/c1 → (groupID, 2*tig..+1)，c2/c3 → (groupID+8, 同列)
    int groupID = tid / 4;
    int tig = tid % 4;
    C[(row + groupID) * N + col + tig * 2]     = c_reg[0];
    C[(row + groupID) * N + col + tig * 2 + 1] = c_reg[1];
    C[(row + groupID + 8) * N + col + tig * 2]     = c_reg[2];
    C[(row + groupID + 8) * N + col + tig * 2 + 1] = c_reg[3];
}

// ---------- Host 端：数据准备 + cuBLAS 对比 + 正确性验证 ----------
int main(int argc, char** argv)
{
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("M=N=K    | mma.sync_ms  TF32cub_ms  | mma%%TF32  max_diff\n");
    printf("---------|---------------------------|---------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        // 本教学 kernel 未做边界处理，要求整除（本表 sizes 均满足）
        assert(M % TILE_M == 0 && N % TILE_N == 0 && K % TILE_K == 0);

        __half* h_A = (__half*)malloc(M * K * sizeof(__half));
        __half* h_B = (__half*)malloc(K * N * sizeof(__half));
        srand(42);
        for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 100) / 100.0f);
        for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(rand() % 100) / 100.0f);

        __half *d_A, *d_B;
        float *d_C, *d_C_ref;
        cudaMalloc(&d_A, M * K * sizeof(__half));
        cudaMalloc(&d_B, K * N * sizeof(__half));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMalloc(&d_C_ref, M * N * sizeof(float));
        cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(__half), cudaMemcpyHostToDevice);

        // --- mma.sync kernel 计时 ---
        dim3 grid(N / TILE_N, M / TILE_M);
        dim3 block(32);
        mma_sync_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            mma_sync_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float mma_ms = 0;
        cudaEventElapsedTime(&mma_ms, start, stop);
        mma_ms /= 10.0f;

        // --- cuBLAS TF32 参考 ---
        float* h_Af = (float*)malloc(M * K * sizeof(float));
        float* h_Bf = (float*)malloc(K * N * sizeof(float));
        for (int i = 0; i < M * K; i++) h_Af[i] = __half2float(h_A[i]);
        for (int i = 0; i < K * N; i++) h_Bf[i] = __half2float(h_B[i]);
        float *d_Af, *d_Bf;
        cudaMalloc(&d_Af, M * K * sizeof(float));
        cudaMalloc(&d_Bf, K * N * sizeof(float));
        cudaMemcpy(d_Af, h_Af, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bf, h_Bf, K * N * sizeof(float), cudaMemcpyHostToDevice);

        float alpha = 1.0f, beta = 0.0f;
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

        // --- 正确性验证（max_diff vs cuBLAS TF32） ---
        float* h_C = (float*)malloc(M * N * sizeof(float));
        float* h_C_ref = (float*)malloc(M * N * sizeof(float));
        cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        float max_diff = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float diff = fabsf(h_C[i] - h_C_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("%-8d | %-12.4f %-12.4f | %-8.1f %.2e\n",
               M, mma_ms, cub_ms, 100.0f * cub_ms / mma_ms, max_diff);

        free(h_A); free(h_B); free(h_Af); free(h_Bf); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
        cudaFree(d_Af); cudaFree(d_Bf);
    }

    cublasDestroy(handle);
    return 0;
}
