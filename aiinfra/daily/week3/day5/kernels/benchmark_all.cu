#include <cstdint>
// benchmark_all.cu —— Week3 性能演进统一 benchmark
// 复用 Day1(naive)/Day2(tiled)/Day3(mma.sync)/Day5(dbuf) 四个 kernel
// 编译: nvcc -O3 -arch=sm_120 -lcublas benchmark_all.cu -o bench_all

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define BM 64
#define BN 64
#define BK 16
#define WM 32
#define WN 32
#define SMEM_PAD 8

__global__ void wmma_gemm_naive_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    int warpM = blockIdx.y, warpN = blockIdx.x;
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM*WMMA_M*K + k, K);
        wmma::load_matrix_sync(b_frag, B + k + warpN*WMMA_N*K, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + warpM*WMMA_M*N + warpN*WMMA_N, c_frag, N, wmma::mem_row_major);
}

__global__ void wmma_gemm_tiled_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_x = warp_id % 2, warp_y = warp_id / 2;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    __shared__ __half smemA[BM][BK + SMEM_PAD];
    __shared__ __half smemB[BK][BN + SMEM_PAD];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(c_frag[i][j], 0.0f);
    for (int k = 0; k < K; k += BK) {
        int load_row = warp_id * 16, load_col = warp_id * 16;
        for (int i = 0; i < 16; i++)
            for (int j = threadIdx.x; j < BK; j += WARP_SIZE) {
                int gr = block_row + load_row + i;
                if (gr < M && (k+j) < K) smemA[load_row+i][j] = A[gr*K + k + j];
            }
        for (int i = threadIdx.x; i < BK; i += WARP_SIZE)
            for (int j = 0; j < 16; j++) {
                int gc = block_col + load_col + j;
                if ((k+i) < K && gc < N) smemB[i][load_col+j] = B[(k+i)*N + gc];
            }
        __syncthreads();
        for (int i = 0; i < 2; i++)
            wmma::load_matrix_sync(a_frag[i], &smemA[warp_y*16 + i*16][0], BK + SMEM_PAD);
        for (int j = 0; j < 2; j++)
            wmma::load_matrix_sync(b_frag[j], &smemB[0][warp_x*16 + j*16], BN + SMEM_PAD);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
            wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        __syncthreads();
    }
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        int row = block_row + warp_y*16 + i*16, col = block_col + warp_x*16 + j*16;
        if (row < M && col < N) wmma::store_matrix_sync(C + row*N + col, c_frag[i][j], N, wmma::mem_row_major);
    }
}

__global__ void mma_sync_gemm_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    __shared__ __half smemA[16][16 + 8];
    __shared__ __half smemB[16][8 + 8];
    uint32_t a_reg[4], b_reg[2];
    float c_reg[4] = {0,0,0,0};
    int tid = threadIdx.x;
    int row = blockIdx.y * 16, col = blockIdx.x * 8;
    for (int k = 0; k < K; k += 16) {
        if (tid < 16) {
            for (int j = 0; j < 16; j++) smemA[tid][j] = A[(row+tid)*K + k + j];
            for (int j = 0; j < 8; j++) smemB[tid][j] = B[(k+tid)*N + col + j];
        }
        __syncthreads();
        uint32_t sa = __cvta_generic_to_shared(&smemA[0][0]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a_reg[0]),"=r"(a_reg[1]),"=r"(a_reg[2]),"=r"(a_reg[3]) : "r"(sa));
        uint32_t sb = __cvta_generic_to_shared(&smemB[0][0]);
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b_reg[0]),"=r"(b_reg[1]) : "r"(sb));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c_reg[0]),"+f"(c_reg[1]),"+f"(c_reg[2]),"+f"(c_reg[3])
            : "r"(a_reg[0]),"r"(a_reg[1]),"r"(a_reg[2]),"r"(a_reg[3]),"r"(b_reg[0]),"r"(b_reg[1]));
        __syncthreads();
    }
    int wr = tid / 4, wc = (tid % 4) * 2;
    if (wr < 16 && (col + wc) < N) {
        C[(row+wr)*N + col + wc] = c_reg[0];
        if (col + wc + 1 < N) C[(row+wr)*N + col + wc + 1] = c_reg[1];
    }
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" :::); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#define NUM_STAGES 2
__device__ __forceinline__ void load_tile_async(
    const __half* A, const __half* B,
    __half (&smemA)[BM][BK + SMEM_PAD], __half (&smemB)[BK][BN + SMEM_PAD],
    int block_row, int block_col, int k, int K, int N)
{
    {
        int chunk = threadIdx.x;
        int r = chunk / 2, c8 = (chunk % 2) * 8;
        uint32_t addr = __cvta_generic_to_shared(&smemA[r][c8]);
        cp_async_16(addr, A + (block_row + r) * K + k + c8);
    }
    {
        int chunk = threadIdx.x;
        int r = chunk / 8, c8 = (chunk % 8) * 8;
        uint32_t addr = __cvta_generic_to_shared(&smemB[r][c8]);
        cp_async_16(addr, B + (k + r) * N + block_col + c8);
    }
}

__device__ __forceinline__ void compute_mma(
    __half (&smemA)[BM][BK + SMEM_PAD], __half (&smemB)[BK][BN + SMEM_PAD],
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> (&a_frag)[2],
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> (&b_frag)[2],
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&c_frag)[2][2],
    int warp_x, int warp_y)
{
    #pragma unroll
    for (int i = 0; i < 2; i++)
        wmma::load_matrix_sync(a_frag[i], &smemA[warp_y*16 + i*16][0], BK + SMEM_PAD);
    #pragma unroll
    for (int j = 0; j < 2; j++)
        wmma::load_matrix_sync(b_frag[j], &smemB[0][warp_x*16 + j*16], BN + SMEM_PAD);
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
}

__global__ void wmma_gemm_dbuf_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_x = warp_id % 2, warp_y = warp_id / 2;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    __shared__ __half smemA[NUM_STAGES][BM][BK + SMEM_PAD];
    __shared__ __half smemB[NUM_STAGES][BK][BN + SMEM_PAD];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(c_frag[i][j], 0.0f);
    load_tile_async(A, B, smemA[0], smemB[0], block_row, block_col, 0, K, N);
    cp_async_commit();
    int buf = 0;
    for (int k = BK; k < K; k += BK) {
        int next = buf ^ 1;
        load_tile_async(A, B, smemA[next], smemB[next], block_row, block_col, k, K, N);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        compute_mma(smemA[buf], smemB[buf], a_frag, b_frag, c_frag, warp_x, warp_y);
        __syncthreads();
        buf = next;
    }
    cp_async_wait<0>();
    __syncthreads();
    compute_mma(smemA[buf], smemB[buf], a_frag, b_frag, c_frag, warp_x, warp_y);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
        int row = block_row + warp_y*16 + i*16, col = block_col + warp_x*16 + j*16;
        if (row < M && col < N) wmma::store_matrix_sync(C + row*N + col, c_frag[i][j], N, wmma::mem_row_major);
    }
}

typedef void (*launch_fn)(const __half*, const __half*, float*, int, int, int);
static void launch_naive(const __half* A, const __half* B, float* C, int M, int N, int K) {
    dim3 g((N+15)/16, (M+15)/16); wmma_gemm_naive_kernel<<<g, 32>>>(A, B, C, M, N, K);
}
static void launch_tiled(const __half* A, const __half* B, float* C, int M, int N, int K) {
    dim3 g((N+BN-1)/BN, (M+BM-1)/BM); wmma_gemm_tiled_kernel<<<g, 128>>>(A, B, C, M, N, K);
}
static void launch_mma(const __half* A, const __half* B, float* C, int M, int N, int K) {
    dim3 g((N+7)/8, (M+15)/16); mma_sync_gemm_kernel<<<g, 32>>>(A, B, C, M, N, K);
}
static void launch_dbuf(const __half* A, const __half* B, float* C, int M, int N, int K) {
    dim3 g((N+BN-1)/BN, (M+BM-1)/BM); wmma_gemm_dbuf_kernel<<<g, 128>>>(A, B, C, M, N, K);
}

static float bench(const __half* A, const __half* B, float* C, int M, int N, int K, launch_fn fn) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    fn(A, B, C, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);

    printf("=== WMMA GEMM Performance Evolution (RTX 5090, sm_120, TF32 cuBLAS baseline) ===\n");
    printf("Size     | Day1_naive  Day2_tiled  Day3_mma   Day5_dbuf  | Best%%   Best_impl\n");
    printf("---------|----------------------------------------------------------------|------------------\n");

    for (int si = 0; si < 4; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        size_t a_sz=(size_t)M*K, b_sz=(size_t)K*N, c_sz=(size_t)M*N;
        __half *dA, *dB; float *dC, *dCref, *dAf, *dBf;
        cudaMalloc(&dA, a_sz*sizeof(__half));
        cudaMalloc(&dB, b_sz*sizeof(__half));
        cudaMalloc(&dC, c_sz*sizeof(float));
        cudaMalloc(&dCref, c_sz*sizeof(float));
        cudaMalloc(&dAf, a_sz*sizeof(float));
        cudaMalloc(&dBf, b_sz*sizeof(float));
        __half* hA = (__half*)malloc(a_sz*sizeof(__half));
        __half* hB = (__half*)malloc(b_sz*sizeof(__half));
        float* hAf = (float*)malloc(a_sz*sizeof(float));
        float* hBf = (float*)malloc(b_sz*sizeof(float));
        for (size_t i=0;i<a_sz;i++) { hA[i] = __float2half((float)((rand()%200)-100)/100.0f); hAf[i] = __half2float(hA[i]); }
        for (size_t i=0;i<b_sz;i++) { hB[i] = __float2half((float)((rand()%200)-100)/100.0f); hBf[i] = __half2float(hB[i]); }
        cudaMemcpy(dA, hA, a_sz*sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, b_sz*sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dAf, hAf, a_sz*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dBf, hBf, b_sz*sizeof(float), cudaMemcpyHostToDevice);

        float alpha=1.0f, beta=0.0f;
        for (int w=0;w<3;w++) cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N,M,K, &alpha, dBf, N, dAf, K, &beta, dCref, N);
        cudaDeviceSynchronize();
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int it=0;it<10;it++) cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N,M,K, &alpha, dBf, N, dAf, K, &beta, dCref, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float cub_ms; cudaEventElapsedTime(&cub_ms, s, e); cub_ms /= 10.0f;
        cudaEventDestroy(s); cudaEventDestroy(e);

        launch_naive(dA,dB,dC,M,N,K); cudaDeviceSynchronize();
        launch_tiled(dA,dB,dC,M,N,K); cudaDeviceSynchronize();
        launch_mma(dA,dB,dC,M,N,K); cudaDeviceSynchronize();
        launch_dbuf(dA,dB,dC,M,N,K); cudaDeviceSynchronize();

        float n_ms = bench(dA,dB,dC,M,N,K, launch_naive);
        float t_ms = bench(dA,dB,dC,M,N,K, launch_tiled);
        float m_ms = bench(dA,dB,dC,M,N,K, launch_mma);
        float d_ms = bench(dA,dB,dC,M,N,K, launch_dbuf);

        float n_pct = 100.0f * cub_ms / n_ms;
        float t_pct = 100.0f * cub_ms / t_ms;
        float m_pct = 100.0f * cub_ms / m_ms;
        float d_pct = 100.0f * cub_ms / d_ms;
        struct { float p; const char* n; } arr[] = {{n_pct,"Day1_naive"},{t_pct,"Day2_tiled"},{m_pct,"Day3_mma"},{d_pct,"Day5_dbuf"}};
        int best=0; for(int i=1;i<4;i++) if(arr[i].p>arr[best].p) best=i;
        printf("%-8d | %-10.1f%%  %-10.1f%%  %-9.1f%%  %-10.1f%%| %-6.1f  %s\n",
            M, n_pct, t_pct, m_pct, d_pct, arr[best].p, arr[best].n);

        free(hA); free(hB); free(hAf); free(hBf);
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCref); cudaFree(dAf); cudaFree(dBf);
    }
    cublasDestroy(h);
    return 0;
}
