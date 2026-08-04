// gemm_backward.cu —— Naive GEMM Backward: dA = dC @ B^T, dB = A^T @ dC
// 编译命令: nvcc -o gemm_backward gemm_backward.cu -O3 -arch=sm_120
// 运行命令: ./gemm_backward
//
// 前向: C = A @ B, A: M×K (row-major), B: K×N, C: M×N
// 反向: dA = dC @ B^T (M×K), dB = A^T @ dC (K×N)
// 教学版：每个线程算一个输出元素，naive but correct，重点展示"反向即两个转置 GEMM"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// dA[i,k] = sum_j dC[i,j] * B[k,j]   (B^T 的第 k 行 = B 的第 k 行，B row-major K×N)
__global__ void gemm_backward_dA_kernel(const float* __restrict__ dC,
                                        const float* __restrict__ B,
                                        float* __restrict__ dA,
                                        int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // M 维
    int k = blockIdx.y * blockDim.y + threadIdx.y;   // K 维
    if (i >= M || k >= K) return;
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += dC[i * N + j] * B[k * N + j];
    }
    dA[i * K + k] = sum;
}

// dB[k,j] = sum_i A[i,k] * dC[i,j]   (A^T 的第 k 行 = A 的第 k 列)
__global__ void gemm_backward_dB_kernel(const float* __restrict__ A,
                                        const float* __restrict__ dC,
                                        float* __restrict__ dB,
                                        int M, int N, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;   // K 维
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // N 维
    if (k >= K || j >= N) return;
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += A[i * K + k] * dC[i * N + j];
    }
    dB[k * N + j] = sum;
}

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

void init_data(float* p, int n) {
    srand(42);
    for (int i = 0; i < n; i++) p[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.4f;
}

bool check(const float* a, const float* b, int n, float eps) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) md = fmaxf(md, fabsf(a[i] - b[i]));
    bool ok = md < eps;
    printf("  maxDiff = %.2e (%s)\n", md, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    int M = 64, N = 64, K = 32;
    printf("=== Naive GEMM Backward ===\n");
    printf("A: %dx%d, B: %dx%d, C: %dx%d\n\n", M, K, K, N, M, N);

    size_t sA = M * K, sB = K * N, sC = M * N;
    float *hA = (float*)malloc(sA * 4), *hB = (float*)malloc(sB * 4);
    float *hC = (float*)malloc(sC * 4), *hdC = (float*)malloc(sC * 4);
    float *hdA = (float*)malloc(sA * 4), *hdB = (float*)malloc(sB * 4);
    float *hdA_ref = (float*)malloc(sA * 4), *hdB_ref = (float*)malloc(sB * 4);
    float *hdA_fd = (float*)malloc(sA * 4);   // finite-difference reference

    init_data(hA, sA); init_data(hB, sB);
    cpu_gemm(hA, hB, hC, M, N, K);

    // 取 loss = sum(C)，则 dC = d loss / dC = ones(M,N)
    for (size_t i = 0; i < sC; i++) hdC[i] = 1.0f;

    // CPU 解析解: dA = dC @ B^T, dB = A^T @ dC
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float s = 0.0f;
            for (int j = 0; j < N; j++) s += hdC[i * N + j] * hB[k * N + j];
            hdA_ref[i * K + k] = s;
        }
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int i = 0; i < M; i++) s += hA[i * K + k] * hdC[i * N + j];
            hdB_ref[k * N + j] = s;
        }

    // 有限差分验证 dA: loss = sum(A @ B), d loss / d A[i,k] = sum_j B[k,j]
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float s = 0.0f;
            for (int j = 0; j < N; j++) s += hB[k * N + j];
            hdA_fd[i * K + k] = s;
        }

    float *dA, *dB, *dC, *ddA, *ddB;
    cudaMalloc(&dA, sA * 4); cudaMalloc(&dB, sB * 4); cudaMalloc(&dC, sC * 4);
    cudaMalloc(&ddA, sA * 4); cudaMalloc(&ddB, sB * 4);
    cudaMemcpy(dA, hA, sA * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hdC, sC * 4, cudaMemcpyHostToDevice);

    dim3 blk(16, 16);
    dim3 grd_dA((M + 15) / 16, (K + 15) / 16);
    dim3 grd_dB((K + 15) / 16, (N + 15) / 16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gemm_backward_dA_kernel<<<grd_dA, blk>>>(dC, dB, ddA, M, N, K);
    gemm_backward_dB_kernel<<<grd_dB, blk>>>(dA, dC, ddB, M, N, K);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(hdA, ddA, sA * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdB, ddB, sB * 4, cudaMemcpyDeviceToHost);

    printf("[dA = dC @ B^T] GPU vs CPU ref:\n");   check(hdA, hdA_ref, sA, 1e-4f);
    printf("[dA] CPU ref vs finite-diff:\n");        check(hdA_ref, hdA_fd, sA, 1e-4f);
    printf("[dB = A^T @ dC] GPU vs CPU ref:\n");     check(hdB, hdB_ref, sB, 1e-4f);
    printf("GPU Time (dA + dB kernels): %.3f ms\n", ms);

    free(hA); free(hB); free(hC); free(hdC); free(hdA); free(hdB);
    free(hdA_ref); free(hdB_ref); free(hdA_fd);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(ddA); cudaFree(ddB);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
