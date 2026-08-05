// flash_attention.cu —— FlashAttention 简化版 Forward Kernel
// 编译命令: nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_120
// 运行命令: ./flash_attention

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define Br 64 // Q tile 的行数；本实现一个 block 固定 Br 个线程，每个线程负责 Q tile 的一行
#define Bc 32 // K/V tile 的行数；Bc=32 时 SRAM 占 32 KB，Bc=64 会顶到 48 KB 静态上限、每 SM 只能驻留 1 个 block
#define D 64  // Head dimension

__global__ void flashAttentionFwd(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                                  float* __restrict__ O, int N, int numHeads) {
    __shared__ float s_Q[Br][D]; // Q tile: Br×D
    __shared__ float s_K[Bc][D]; // K tile: Bc×D
    __shared__ float s_V[Bc][D]; // V tile: Bc×D
    // 注意：S/P 中间结果不放 shared memory，每个线程用寄存器/local 保存自己那一行的值

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qTileRow = blockIdx.x * Br;

    int tid = threadIdx.x;        // 本线程负责的 Q 行（tile 内偏移）
    int qRow = qTileRow + tid;    // 全局行号
    int bhOffset = (batch * numHeads + head) * N;

    // 每个线程维护自己那一行的 running 状态
    float m = -1e30f;   // running max
    float l = 0.0f;     // running sum
    float acc[D] = {0}; // running output accumulator（每步归一化变体，末尾无需再除 l）

    // Step 1: 全 block 协作加载 Q tile 到 Shared Memory（全局内存合并访问）
    for (int idx = tid; idx < Br * D; idx += Br) {
        int r = idx / D, c = idx % D;
        s_Q[r][c] = (qTileRow + r < N) ? Q[bhOffset * D + (qTileRow + r) * D + c] : 0.0f;
    }
    __syncthreads();

    // Step 2: 内循环遍历 K/V tile
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
        // 2a: 协作加载 K 和 V tile
        for (int idx = tid; idx < Bc * D; idx += Br) {
            int r = idx / D, c = idx % D;
            s_K[r][c] = (kvStart + r < N) ? K[bhOffset * D + (kvStart + r) * D + c] : 0.0f;
            s_V[r][c] = (kvStart + r < N) ? V[bhOffset * D + (kvStart + r) * D + c] : 0.0f;
        }
        __syncthreads();

        // 2b+2c: 每个线程独立计算自己那一行的 score，并做 Online Softmax 更新
        if (qRow < N) {
            int kvLen = min(Bc, N - kvStart); // 最后一个 tile 可能不满

            // 2b: s_row[c] = Q[qRow] · K[kvStart+c]，本行对当前 KV tile 的 kvLen 个 score
            float s_row[Bc];
            float m_tile = -1e30f;
            for (int c = 0; c < kvLen; c++) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d++)
                    s += s_Q[tid][d] * s_K[c][d];
                s_row[c] = s; // 面试/LeetGPU 版本这里要乘 1/sqrtf(D)
                m_tile = fmaxf(m_tile, s);
            }

            // 公式1: max 更新
            float m_new = fmaxf(m, m_tile);

            // 公式2: sum 更新（l_scale 把旧 sum 从参考点 m 缩放到 m_new）
            float l_scale = expf(m - m_new);
            float l_new = l * l_scale;
            for (int c = 0; c < kvLen; c++) {
                s_row[c] = expf(s_row[c] - m_new); // p_c = exp(s_c - m_new)
                l_new += s_row[c];
            }

            // 公式3: output 更新（每步归一化变体）
            float o_scale = (l * l_scale) / l_new;
            #pragma unroll
            for (int d = 0; d < D; d++)
                acc[d] *= o_scale;
            for (int c = 0; c < kvLen; c++) {
                float p_norm = s_row[c] / l_new;
                #pragma unroll
                for (int d = 0; d < D; d++)
                    acc[d] += p_norm * s_V[c][d];
            }

            m = m_new;
            l = l_new;
        }
        __syncthreads(); // 等所有线程用完 s_K/s_V，再加载下一个 tile
    }

    // Step 3: 写回最终结果
    if (qRow < N) {
        for (int d = 0; d < D; d++)
            O[bhOffset * D + qRow * D + d] = acc[d];
    }
}

// 避免宏 D 与函数参数名冲突
#undef D

// CPU 参考实现（标准 Attention，用于验证正确性；与 kernel 同步省略 1/√d scale）
void cpuAttention(const float* Q, const float* K, const float* V, float* O, int N, int D) {
    float* S = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int d = 0; d < D; d++)
                sum += Q[i * D + d] * K[j * D + d];
            S[i * N + j] = sum;
        }
    }
    for (int i = 0; i < N; i++) {
        float maxVal = S[i * N];
        for (int j = 1; j < N; j++)
            maxVal = fmaxf(maxVal, S[i * N + j]);
        float sum = 0;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - maxVal);
            sum += S[i * N + j];
        }
        for (int j = 0; j < N; j++)
            S[i * N + j] /= sum;
    }
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += S[i * N + j] * V[j * D + d];
            O[i * D + d] = sum;
        }
    }
    free(S);
}

void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
}

bool checkResult(const float* gpu, const float* cpu, int n, float eps) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > eps) {
            printf("Mismatch at %d: GPU=%.6f, CPU=%.6f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 256;
    const int D = 64;
    const int batchSize = 1;
    const int numHeads = 1;

    printf("=== FlashAttention Simplified Forward ===\n");
    printf("Config: N=%d, D=%d, batch=%d, heads=%d\n", N, D, batchSize, numHeads);
    printf("SRAM usage per block: %.2f KB\n", (Br * D + Bc * D * 2) * sizeof(float) / 1024.0);

    size_t totalElements = batchSize * numHeads * N * D;
    size_t bytes = totalElements * sizeof(float);

    float* h_Q = (float*)malloc(bytes);
    float* h_K = (float*)malloc(bytes);
    float* h_V = (float*)malloc(bytes);
    float* h_O = (float*)malloc(bytes);
    float* h_O_CPU = (float*)malloc(bytes);

    srand(42); // 只播种一次：若在 initMatrix 里每次 srand(42)，Q/K/V 会被初始化成完全相同的矩阵
    initMatrix(h_Q, batchSize * numHeads * N, D);
    initMatrix(h_K, batchSize * numHeads * N, D);
    initMatrix(h_V, batchSize * numHeads * N, D);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    dim3 gridDim((N + Br - 1) / Br, numHeads, batchSize);
    dim3 blockDim(Br); // 一个 block Br 个线程，每个线程负责 Q tile 的一行

    printf("Grid: (%d, %d, %d), Block: %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    flashAttentionFwd<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, N, numHeads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

    cpuAttention(h_Q, h_K, h_V, h_O_CPU, N, D);
    bool correct = checkResult(h_O, h_O_CPU, totalElements, 1e-3);

    printf("GPU Time: %.3f ms\n", ms);
    printf("Result check: %s\n", correct ? "PASS" : "FAIL");

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O_CPU);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
