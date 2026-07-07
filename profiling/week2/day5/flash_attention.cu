// flash_attention.cu —— FlashAttention 简化版 Forward（ncu profiling 版）
// 编译命令: nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_80 -g -lineinfo
// 运行命令: ./flash_attention

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define Br 32   // Q tile 的行数（SRAM 可容纳）
#define Bc 32   // K/V tile 的行数（SRAM 可容纳）
#define D 64    // Head dimension

#define NUM_THREADS_X Bc   // 64
#define NUM_THREADS_Y 4    // Br/NUM_THREADS_Y = 64/4 = 16

__global__ void flashAttentionFwd(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    int N, int numHeads) {
    __shared__ float s_Q[Br][D];    // Q tile: Br×D
    __shared__ float s_K[Bc][D];    // K tile: Bc×D
    __shared__ float s_V[Bc][D];    // V tile: Bc×D
    __shared__ float s_S[Br][Bc];   // S = Q×K^T partial: Br×Bc

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qTileRow = blockIdx.x * Br;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int bhOffset = ((batch * numHeads + head) * N);

    // 每个线程维护的 running 状态（按 Q 行）
    float m = -1e30f;   // running max
    float l = 0.0f;     // running sum
    float acc[D] = {0}; // running output accumulator

    // Step 1: 加载 Q tile 到 Shared Memory
    for (int i = tid_y; i < Br; i += NUM_THREADS_Y) {
        int qRow = qTileRow + i;
        for (int d = tid_x; d < D; d += NUM_THREADS_X) {
            if (qRow < N) {
                s_Q[i][d] = Q[bhOffset * D + qRow * D + d];
            } else {
                s_Q[i][d] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Step 2: 内循环遍历 K/V tile
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
        // 2a: 加载 K 和 V tile
        for (int i = tid_y; i < Bc; i += NUM_THREADS_Y) {
            int kvRow = kvStart + i;
            for (int d = tid_x; d < D; d += NUM_THREADS_X) {
                if (kvRow < N) {
                    s_K[i][d] = K[bhOffset * D + kvRow * D + d];
                    s_V[i][d] = V[bhOffset * D + kvRow * D + d];
                } else {
                    s_K[i][d] = 0.0f;
                    s_V[i][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        // 2b: 计算 S_tile = Q_tile × K_tile^T (Br×Bc)
        for (int qi = tid_y; qi < Br; qi += NUM_THREADS_Y) {
            for (int ki = tid_x; ki < Bc; ki += NUM_THREADS_X) {
                float s_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    s_val += s_Q[qi][d] * s_K[ki][d];
                }
                s_S[qi][ki] = s_val;
            }
        }
        __syncthreads();

        // 2c: Online Softmax 更新（每个 Q 行独立处理）
        for (int qi = tid_y; qi < Br && (qTileRow + qi) < N; qi += NUM_THREADS_Y) {
            if (tid_x == 0) {
                // 公式1: 计算新块的局部 max
                float m_prev = m;
                float m_new = m_prev;
                for (int c = 0; c < Bc && (kvStart + c) < N; c++) {
                    m_new = fmaxf(m_new, s_S[qi][c]);
                }

                // 公式2: 更新 running sum
                float l_scale = expf(m_prev - m_new);
                float l_new = l * l_scale;

                float p[Bc];
                for (int c = 0; c < Bc && (kvStart + c) < N; c++) {
                    p[c] = expf(s_S[qi][c] - m_new);
                    l_new += p[c];
                }

                // 公式3: 更新 running output
                float o_scale = (l * l_scale) / l_new;
                for (int d = 0; d < D; d++) {
                    acc[d] = acc[d] * o_scale;
                }
                for (int c = 0; c < Bc && (kvStart + c) < N; c++) {
                    float p_norm = p[c] / l_new;
                    for (int d = 0; d < D; d++) {
                        acc[d] += p_norm * s_V[c][d];
                    }
                }

                m = m_new;
                l = l_new;
            }
        }
        __syncthreads();
    }

    // Step 3: 写回最终结果
    for (int qi = tid_y; qi < Br && (qTileRow + qi) < N; qi += NUM_THREADS_Y) {
        if (tid_x == 0) {
            for (int d = 0; d < D; d++) {
                int outRow = qTileRow + qi;
                O[bhOffset * D + outRow * D + d] = acc[d];
            }
        }
    }
}

// ============================================================
// 标准 Attention（物化 S/P，用于 ncu 对比）
// ============================================================
__global__ void standardAttentionFwd(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ S,
                                       float* __restrict__ P,
                                       float* __restrict__ O,
                                       int N, int numHeads) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (qRow >= N) return;

    int bhOffset = (batch * numHeads + head) * N;
    float scale = 1.0f / sqrtf((float)D);

    // S = Q × K^T
    for (int j = 0; j < N; j++) {
        float s_val = 0.0f;
        for (int d = 0; d < D; d++) {
            s_val += Q[bhOffset * D + qRow * D + d] * K[bhOffset * D + j * D + d];
        }
        S[bhOffset * N + qRow * N + j] = s_val * scale;
    }
    __syncthreads();

    // P = softmax(S)
    float maxVal = -1e30f;
    for (int j = 0; j < N; j++)
        maxVal = fmaxf(maxVal, S[bhOffset * N + qRow * N + j]);
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        P[bhOffset * N + qRow * N + j] = expf(S[bhOffset * N + qRow * N + j] - maxVal);
        sum += P[bhOffset * N + qRow * N + j];
    }

    // O = P × V
    for (int d = 0; d < D; d++) {
        float o_val = 0.0f;
        for (int j = 0; j < N; j++) {
            o_val += (P[bhOffset * N + qRow * N + j] / sum) * V[bhOffset * D + j * D + d];
        }
        O[bhOffset * D + qRow * D + d] = o_val;
    }
}

// CPU 参考实现（标准 Attention，用于验证正确性）
void cpuAttention(const float* Q, const float* K, const float* V,
                  float* O, int N, int headDim) {
    float* S = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int d = 0; d < headDim; d++)
                sum += Q[i * headDim + d] * K[j * headDim + d];
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
        for (int d = 0; d < headDim; d++) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += S[i * N + j] * V[j * headDim + d];
            O[i * headDim + d] = sum;
        }
    }
    free(S);
}

void initMatrix(float* mat, int rows, int cols) {
    srand(42);
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
    const int headDim = 64;
    const int batchSize = 1;
    const int numHeads = 1;

    printf("=== FlashAttention Simplified Forward ===\n");
    printf("Config: N=%d, D=%d, batch=%d, heads=%d\n", N, headDim, batchSize, numHeads);
    printf("SRAM usage per block: %.2f KB\n",
           (Br * headDim + Bc * headDim * 2 + Br * Bc) * sizeof(float) / 1024.0);

    size_t totalElements = batchSize * numHeads * N * headDim;
    size_t bytes = totalElements * sizeof(float);

    float *h_Q = (float*)malloc(bytes);
    float *h_K = (float*)malloc(bytes);
    float *h_V = (float*)malloc(bytes);
    float *h_O = (float*)malloc(bytes);
    float *h_O_CPU = (float*)malloc(bytes);

    initMatrix(h_Q, batchSize * numHeads * N, headDim);
    initMatrix(h_K, batchSize * numHeads * N, headDim);
    initMatrix(h_V, batchSize * numHeads * N, headDim);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    dim3 gridDim((N + Br - 1) / Br, numHeads, batchSize);
    dim3 blockDim(NUM_THREADS_X, NUM_THREADS_Y);

    printf("Grid: (%d, %d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ===== FlashAttention =====
    cudaEventRecord(start);
    flashAttentionFwd<<<gridDim, blockDim>>>(d_Q, d_K, d_V, d_O, N, numHeads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_flash;
    cudaEventElapsedTime(&ms_flash, start, stop);
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

    cpuAttention(h_Q, h_K, h_V, h_O_CPU, N, headDim);
    bool correct = checkResult(h_O, h_O_CPU, totalElements, 1e-3);

    printf("FlashAttention GPU Time: %.3f ms\n", ms_flash);
    printf("Result check: %s\n", correct ? "PASS" : "FAIL");

    // ===== Standard Attention（对比） =====
    size_t bytesSP = (size_t)batchSize * numHeads * N * N * sizeof(float);
    float *d_S, *d_P;
    cudaMalloc(&d_S, bytesSP);
    cudaMalloc(&d_P, bytesSP);

    dim3 stdGrid(N, numHeads, batchSize);
    dim3 stdBlock(256);

    cudaEventRecord(start);
    standardAttentionFwd<<<stdGrid, stdBlock>>>(d_Q, d_K, d_V, d_S, d_P, d_O, N, numHeads);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_std;
    cudaEventElapsedTime(&ms_std, start, stop);
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);
    bool correct2 = checkResult(h_O, h_O_CPU, totalElements, 1e-3);

    printf("Standard Attention GPU Time: %.3f ms\n", ms_std);
    printf("Result check: %s\n", correct2 ? "PASS" : "FAIL");
    printf("Speedup: %.2fx\n", ms_std / ms_flash);

    printf("\n=== ncu 分析命令 ===\n");
    printf("# FlashAttention\n");
    printf("ncu --kernel-name regex:flashAttentionFwd \\\n");
    printf("    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("sm__occupancy.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("    ./flash_attention\n\n");
    printf("# Standard Attention（对比）\n");
    printf("ncu --kernel-name regex:standardAttentionFwd \\\n");
    printf("    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");
    printf("sm__occupancy.avg.pct_of_peak_sustained_elapsed \\\n");
    printf("    ./flash_attention\n\n");
    printf("# HBM 读写量对比\n");
    printf("ncu --kernel-name regex:\"flashAttentionFwd|standardAttentionFwd\" \\\n");
    printf("    --metrics dram__bytes_read.sum,dram__bytes_write.sum \\\n");
    printf("    ./flash_attention\n");

    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_CPU);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_S); cudaFree(d_P);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
