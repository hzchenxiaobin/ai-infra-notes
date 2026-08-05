// wmma_gemm_dbuf.cu —— Day 5: Double Buffer WMMA GEMM（cp.async 真双缓冲）
// FP16 输入 / FP32 累加，2-stage pipeline：
//   cp.async 异步加载 tile[k+1] 与 当前 tile[k] 的 MMA 计算重叠。
// 与 cuBLAS(TF32) 对比并做正确性验证。
//
// 编译命令: nvcc -O3 -arch=sm_120 -lcublas wmma_gemm_dbuf.cu -o wmma_dbuf
//
// ⚠️ 本机无 GPU，未编译实测——实测数据待 GPU 环境回填。
//    README 中原"预期输出"表为预估口径，不作为实测引用。
//
// 与 README 旧版内嵌代码的区别：旧版在 pipeline 里包的是普通赋值拷贝
// （global→register→smem，同步），pipeline 注解无效、无任何重叠收益。
// 本版用 cp.async PTX（global→smem 直达、异步）实现真正的 load/compute 重叠，
// 与 week10/day4 实测表"同步实现未重叠，需 cp.async/TMA"的口径一致。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 16
#define PAD 8
#define NUM_STAGES 2
#define WARP_SIZE 32

// ---------- cp.async PTX 封装 ----------
// 异步拷贝 16 字节：global → shared（不经寄存器）
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" :::);
}

// 等待至多 N 组 cp.async 未完成（N=0 即全部完成）
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---------- 异步加载一个 K-step 的 A/B tile 到 smem[stage] ----------
// A tile: BM×BK = 64×16 halfs = 64 行 × 32B = 128 个 16B chunk，128 线程每线程 1 个
// B tile: BK×BN = 16×64 halfs = 16 行 × 128B = 128 个 16B chunk，128 线程每线程 1 个
__device__ __forceinline__ void load_tile_async(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half (&smemA)[BM][BK + PAD], __half (&smemB)[BK][BN + PAD],
    int block_row, int block_col, int k, int K, int N)
{
    // A：chunk = row*2 + half8（每行 2 个 16B chunk）
    {
        int chunk = threadIdx.x;
        int r = chunk / 2, c8 = (chunk % 2) * 8;
        uint32_t addr = __cvta_generic_to_shared(&smemA[r][c8]);
        cp_async_16(addr, A + (block_row + r) * K + k + c8);
    }
    // B：chunk = row*8 + c8（每行 8 个 16B chunk），按 [k][n] 存放
    {
        int chunk = threadIdx.x;
        int r = chunk / 8, c8 = (chunk % 8) * 8;
        uint32_t addr = __cvta_generic_to_shared(&smemB[r][c8]);
        cp_async_16(addr, B + (k + r) * N + block_col + c8);
    }
}

// ---------- 从 smem[stage] 计算 2×2 个 MMA ----------
__device__ __forceinline__ void compute_mma(
    __half (&smemA)[BM][BK + PAD], __half (&smemB)[BK][BN + PAD],
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> (&a_frag)[2],
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> (&b_frag)[2],
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&c_frag)[2][2],
    int warp_x, int warp_y)
{
    #pragma unroll
    for (int i = 0; i < 2; i++)
        wmma::load_matrix_sync(a_frag[i],
            &smemA[warp_y * 16 + i * 16][0], BK + PAD);
    #pragma unroll
    for (int j = 0; j < 2; j++)
        wmma::load_matrix_sync(b_frag[j],
            &smemB[0][warp_x * 16 + j * 16], BN + PAD);
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
}

__global__ void wmma_gemm_dbuf_kernel(
    const __half* __restrict__ A,    // M×K, row-major
    const __half* __restrict__ B,    // K×N, row-major
    float* __restrict__ C,           // M×N, row-major
    int M, int N, int K)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_x = warp_id % 2;
    int warp_y = warp_id / 2;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 双缓冲 shared memory
    __shared__ __half smemA[NUM_STAGES][BM][BK + PAD];
    __shared__ __half smemB[NUM_STAGES][BK][BN + PAD];

    // B 在 smem 中按 [k][n] 存放（k 行 n 列），对 matrix_b 是 col-major 布局
    // （ld = BN + PAD），故 b_frag 声明为 col_major——与 Day 2 落盘版一致
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // 1. 预加载 tile[0] 到 stage 0（pipeline 预热）
    load_tile_async(A, B, smemA[0], smemB[0], block_row, block_col, 0, K, N);
    cp_async_commit();

    int buf = 0;
    // 2. 流水线主循环：加载 tile[k] 与计算 tile[k-BK] 重叠
    for (int k = BK; k < K; k += BK) {
        int next = buf ^ 1;
        // 异步加载下一块（提交后立即返回，不阻塞下面的计算）
        load_tile_async(A, B, smemA[next], smemB[next], block_row, block_col, k, K, N);
        cp_async_commit();
        // 等当前 buffer 的数据就绪（至多 1 组在途 = 刚提交的 next 组）
        cp_async_wait<1>();
        __syncthreads();
        // 从当前 buffer 计算（与 next 的 cp.async 在途传输重叠）
        compute_mma(smemA[buf], smemB[buf], a_frag, b_frag, c_frag, warp_x, warp_y);
        // 全部 warp 读完 buf 后，下一轮才能往其中覆写
        __syncthreads();
        buf = next;
    }

    // 3. 计算最后一块（pipeline 排空）
    cp_async_wait<0>();
    __syncthreads();
    compute_mma(smemA[buf], smemB[buf], a_frag, b_frag, c_frag, warp_x, warp_y);

    // 4. 存储结果
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int row = block_row + warp_y * 16 + i * 16;
            int col = block_col + warp_x * 16 + j * 16;
            wmma::store_matrix_sync(C + row * N + col,
                c_frag[i][j], N, wmma::mem_row_major);
        }
}

// ---------- Host 端：数据准备 + cuBLAS 对比 + 正确性验证 ----------
int main(int argc, char** argv)
{
    int sizes[] = {512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("M=N=K    | dbuf_ms      TF32cub_ms  | dbuf%%TF32  max_diff\n");
    printf("---------|---------------------------|---------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        // 教学 kernel 未做边界处理，要求 M%BM==0, N%BN==0, K%BK==0（本表均满足）
        assert(M % BM == 0 && N % BN == 0 && K % BK == 0);

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

        // --- double buffer kernel 计时 ---
        dim3 grid(N / BN, M / BM);
        dim3 block(128);   // 4 warp = 128 线程
        wmma_gemm_dbuf_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++)
            wmma_gemm_dbuf_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float dbuf_ms = 0;
        cudaEventElapsedTime(&dbuf_ms, start, stop);
        dbuf_ms /= 10.0f;

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

        printf("%-8d | %-12.4f %-12.4f | %-9.1f %.2e\n",
               M, dbuf_ms, cub_ms, 100.0f * cub_ms / dbuf_ms, max_diff);

        free(h_A); free(h_B); free(h_Af); free(h_Bf); free(h_C); free(h_C_ref);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
        cudaFree(d_Af); cudaFree(d_Bf);
    }

    cublasDestroy(handle);
    return 0;
}
