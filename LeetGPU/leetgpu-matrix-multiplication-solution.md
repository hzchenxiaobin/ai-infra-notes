# LeetGPU Matrix Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Multiplication
- **链接**：https://leetgpu.com/challenges/matrix-multiplication
- **难度**：中等
- **标签**：CUDA、GEMM、Shared Memory Tiling、Roofline、Profiling

给定 `M×K` 矩阵 `A` 和 `K×N` 矩阵 `B`（行优先存储），计算 `C = A × B`，其中 `C` 为 `M×N` 矩阵，`C[i][j] = Σ(A[i][k] * B[k][j])`。

约束：`1 ≤ M, N, K ≤ 1024`，矩阵元素范围 `[-1.0, 1.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[i * K + k] * B[k * N + j];
        C[i * N + j] = sum;
    }
```

### 朴素 GPU 方法（无 Shared Memory）

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
```

- 每个线程读 A 的一行 + B 的一列，大量重复全局内存访问。

![Naive GEMM 流程](images/matmul_naive.svg)

*图 1：Naive GEMM — 每个线程读取 A 的一行与 B 的一列，计算 C 的一个元素。*

## 3. GPU 设计

### 3.1 并行化策略

**Shared Memory Tiling**：把 A/B 的子矩阵预取到 Shared Memory，实现 K 维度的数据复用。

![Shared Memory Tiling 流程](images/matmul_tiled.svg)

*图 2：Shared Memory Tiling — 沿 K 维分块加载 A/B tile 到 Shared Memory，每个 block 累加得到 C 的一个 tile。*

- Block tile：`TILE_SIZE × TILE_SIZE`（如 16×16）
- 每个 block 协作加载 A 的 `TILE_SIZE × TILE_SIZE` tile 和 B 的 `TILE_SIZE × TILE_SIZE` tile
- 在 Shared Memory 中做乘加，减少全局内存访问

### 3.2 Thread / Block 映射

每个 block 负责计算 C 中一个 `TILE_SIZE × TILE_SIZE` 的子块；block 内的每个线程对应一个输出元素。

![Thread / Block 映射](images/matmul_thread_block_mapping.svg)

*图 3：C 矩阵按 TILE 划分给 block；block 内部每个线程负责一个 `C[row][col]`。*

```
row = blockIdx.y * TILE_SIZE + threadIdx.y
col = blockIdx.x * TILE_SIZE + threadIdx.x
```

### 3.3 存储层次使用

| 层次 | 用途 | 效果 |
|------|------|------|
| Global Memory | 读 A、B，写 C | tile 粒度访问 |
| Shared Memory | 缓存 A/B tile | K 维度复用，减少全局访问 |
| Register | 累加器 `sum` | 避免反复读写 Shared Memory |

## 4. Kernel 实现

```cuda
// matrix_multiplication.cu —— Shared Memory Tiling GEMM
// 编译命令: nvcc -o matmul matmul.cu -O3 -arch=sm_80

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_SIZE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_SIZE) {
        if (row < M && bk + threadIdx.x < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + bk + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            s_B[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

// Bank-conflict-free version: pad the shared memory arrays by one column.
// Without padding, s_A[threadIdx.y][k] causes a 2-way bank conflict because
// consecutive rows are 16 floats (64 bytes) apart, which is a multiple of the
// 32-bank shared-memory stride (4 bytes/bank). Padding breaks the alignment.
__global__ void matmul_tiled_nobc(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_SIZE) {
        if (row < M && bk + threadIdx.x < K)
            s_A[threadIdx.y][threadIdx.x] = A[row * K + bk + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (bk + threadIdx.y < K && col < N)
            s_B[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX * 2 - 1;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX * 2 - 1;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA); cudaMalloc(&d_B, bytesB); cudaMalloc(&d_C, bytesC);
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);

    cudaEventRecord(s1);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_naive; cudaEventElapsedTime(&ms_naive, s1, s2);

    cudaEventRecord(s1);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_tiled; cudaEventElapsedTime(&ms_tiled, s1, s2);

    float gflops_naive = 2.0f * M * N * K / (ms_naive * 1e6);
    float gflops_tiled = 2.0f * M * N * K / (ms_tiled * 1e6);

    printf("Naive:  %.3f ms (%.1f GFLOPS)\n", ms_naive, gflops_naive);
    printf("Tiled:  %.3f ms (%.1f GFLOPS)\n", ms_tiled, gflops_tiled);
    printf("Speedup: %.2fx\n", ms_naive / ms_tiled);

    cudaEventRecord(s1);
    matmul_tiled_nobc<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_tiled_nobc; cudaEventElapsedTime(&ms_tiled_nobc, s1, s2);
    float gflops_tiled_nobc = 2.0f * M * N * K / (ms_tiled_nobc * 1e6);

    printf("Tiled (no bank conflict): %.3f ms (%.1f GFLOPS)\n", ms_tiled_nobc, gflops_tiled_nobc);
    printf("Speedup vs Tiled: %.2fx\n", ms_tiled / ms_tiled_nobc);

    free(h_A); free(h_B); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 完整 profiling

```bash
ncu --set full -o matmul_report ./matmul
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./matmul
```

### Roofline 分析

```
算术强度 = 2 * M * N * K / (M*K + K*N + M*N) / sizeof(float)
```

对于 M=N=K=512：AI ≈ 2*512³ / (3*512²*4) ≈ 85 FLOP/Byte → 接近 compute-bound

### Shared Memory Bank Conflict 分析

在 `matmul_tiled` 中，读取 `s_A[threadIdx.y][k]` 时，一个 warp 内的线程按 `threadIdx.y` 访问同一列的不同行。`TILE_SIZE = 16` 时，相邻两行在 Shared Memory 中相距 `16 × 4 = 64` 字节，恰好是 32 个 bank 的整数倍，因此偶数行落在 bank `k`，奇数行落在 bank `k + 16`，形成 **2-way bank conflict**。

解决方案是给 Shared Memory 数组加一列 padding：

```cuda
__shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
__shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];
```

这样行 stride 变成 `17 × 4 = 68` 字节，`68 / 4 = 17` 与 32 互质，同一列的相邻行会落到不同的 bank，conflict 消失。对 `s_B` 同样加 padding 可以保持代码对称，并避免加载阶段潜在的 bank conflict。

### 对比表

| 版本 | 时间(ms) | GFLOPS | SM Throughput | Memory Throughput | 瓶颈 |
|------|---------|--------|--------------|-------------------|------|
| Naive | | | | | memory-bound |
| Tiled | | | | | compute-bound |
| Tiled (no bank conflict) | | | | | compute-bound, fewer shared-memory stalls |

## 6. 复杂度分析

- **时间复杂度**：`O(M×N×K)`。
- **空间复杂度**：`O(M×K + K×N + M×N)` + `O(TILE_SIZE²)` Shared Memory。
- **算术强度**：`2MNK / (4(MK+KN+MN))`，大矩阵下接近 **compute-bound**。
