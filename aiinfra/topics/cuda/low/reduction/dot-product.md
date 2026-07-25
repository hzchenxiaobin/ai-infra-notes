# LeetGPU Dot Product 题解

> **面试考察度**：⭐⭐⭐ Dot Product 是 reduce 的直接应用，面试常作"热身题"考查 warp shuffle 归约基本功
> **面试形式**：手写 fused kernel（元素乘 + 归约）+ 讲清两阶段归约

## 1. 题目概述

- **标题 / 题号**：Dot Product（LeetGPU #17，medium）
- **链接**：https://leetgpu.com/challenges/dot-product
- **难度**：中等
- **标签**：CUDA、Dot Product、fused reduce、warp shuffle、两阶段归约、memory-bound

**题意**：给定两个长度 `N` 的 `float32` 数组 `A` 和 `B`，计算点积写入标量 `result[0]`：

$$\text{result}[0] = \sum_{i=0}^{N-1} A[i] \cdot B[i]$$

函数签名固定：

```cpp
extern "C" void solve(const float* A, const float* B, float* result, int N);
```

**关键约束**：

- `A`、`B` 为 FP32，`result` 为单元素 FP32
- 容差 `atol=1e-5, rtol=1e-5`（紧，需 `double` 累加）
- 性能测例 `N=5`（很小，实际考查通用性）

> 💡 **本质**：dot product = elementwise multiply + reduction。与 #4 Reduction 的唯一区别是多了一步"先乘后加"——可融合进同一个 grid-stride loop（fused kernel），省一次 global 读写。

## 2. CPU 基线 / 朴素 GPU 方法

```cpp
// CPU 串行
float dot_cpu(const float* A, const float* B, int N) {
    double acc = 0.0;
    for (int i = 0; i < N; ++i) acc += (double)A[i] * B[i];
    return (float)acc;
}
```

朴素 GPU：两阶段归约（同 #4 Reduction），但 grid-stride 内 `local_sum += A[i]*B[i]`（融合乘加）。

## 3. GPU 设计

与 #4 Reduction 骨架完全一致（两阶段：block reduce → final reduce），唯一差别是累加项从 `input[i]` 变成 `A[i]*B[i]`——**融合 elementwise mul 进归约**，省一次 global 读。

![两阶段归约：block reduce → global reduce](../../../images/cuda_reduction_overview.svg)

## 4. Kernel 实现

```cuda
// cuda_dot_product.cu —— 手撕 Dot Product：fused mul + 两阶段归约
// 编译命令: nvcc -O3 -arch=sm_120 cuda_dot_product.cu -o dot -lineinfo
// 运行:     ./dot 1000000

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS 4096

__inline__ __device__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ double block_reduce_sum(double val, double* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ---- Kernel 1：fused mul + grid-stride 累加 → partial ----
__global__ void dot_block_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 double* __restrict__ partial,
                                 int N) {
    __shared__ double shared[NUM_WARPS + 1];
    double local_sum = 0.0;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N;
         i += gridDim.x * BLOCK_SIZE)
        local_sum += (double)A[i] * B[i];        // ← 融合 mul 进归约
    double block_sum = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) partial[blockIdx.x] = block_sum;
}

// ---- Kernel 2：归约 partial → result ----
__global__ void dot_final_kernel(const double* __restrict__ partial,
                                 float* __restrict__ result, int M) {
    __shared__ double shared[NUM_WARPS + 1];
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < M; i += BLOCK_SIZE)
        local_sum += partial[i];
    double total = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) result[0] = (float)total;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t bytes = (size_t)N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) { hA[i] = (float)(rand()%200-100)/10.0f; hB[i] = (float)(rand()%200-100)/10.0f; }

    float *dA, *dB, *dResult; double *dPartial;
    CHECK_CUDA(cudaMalloc(&dA, bytes)); CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dResult, sizeof(float)));
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;
    CHECK_CUDA(cudaMalloc(&dPartial, numBlocks * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    dot_block_kernel<<<numBlocks, BLOCK_SIZE>>>(dA, dB, dPartial, N);
    dot_final_kernel<<<1, BLOCK_SIZE>>>(dPartial, dResult, numBlocks);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("N=%d, time: %.3f ms\n", N, ms);

    float res; CHECK_CUDA(cudaMemcpy(&res, dResult, sizeof(float), cudaMemcpyDeviceToHost));
    double ref = 0.0; for (int i = 0; i < N; ++i) ref += (double)hA[i]*hB[i];
    printf("gpu=%.4f ref=%.4f diff=%.2e (%s)\n", res, (float)ref, fabsf(res-(float)ref),
           fabsf(res-(float)ref) < 1e-3f ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dPartial)); CHECK_CUDA(cudaFree(dResult));
    free(hA); free(hB);
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
// dot_product_submit.cu
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS 4096

__inline__ __device__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffff, val, o);
    return val;
}
__inline__ __device__ double block_reduce_sum(double val, double* shared) {
    int lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void dot_block_kernel(const float* A, const float* B, double* partial, int N) {
    __shared__ double shared[NUM_WARPS + 1];
    double local_sum = 0.0;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N; i += gridDim.x * BLOCK_SIZE)
        local_sum += (double)A[i] * B[i];
    if (threadIdx.x == 0) partial[blockIdx.x] = block_reduce_sum(local_sum, shared);
}

__global__ void dot_final_kernel(const double* partial, float* result, int M) {
    __shared__ double shared[NUM_WARPS + 1];
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < M; i += BLOCK_SIZE) local_sum += partial[i];
    double total = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) result[0] = (float)total;
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;
    if (numBlocks < 1) numBlocks = 1;
    double* partial; cudaMalloc(&partial, numBlocks * sizeof(double));
    dot_block_kernel<<<numBlocks, BLOCK_SIZE>>>(A, B, partial, N);
    dot_final_kernel<<<1, BLOCK_SIZE>>>(partial, result, numBlocks);
    cudaFree(partial);
}
```

### 4.2 代码详解

与 #4 Reduction **逐行对称**，唯一差别：grid-stride 累加项 `local_sum += (double)A[i] * B[i]`（融合乘加），而非 `input[i]`。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **融合乘加** | `local_sum += A[i] * B[i]` | elementwise mul 融进归约，省一次 global 读 |
| **两阶段归约** | block reduce → partial → final reduce | 同 #4 Reduction |
| **double 累加** | `double local_sum` | 紧容差 1e-5 需 double |

> 💡 **关键洞察**：dot product = "reduce 的融合变体"。把 elementwise mul 融进 grid-stride 累加，省一次 global 读写（不用先算 `C[i]=A[i]*B[i]` 再归约）。这是 **kernel fusion** 的最简形态——面试常以此引出"为什么要融合"。

## 5-6. 性能与复杂度

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)` |
| **算术强度** | `2 FLOP / 8B = 0.25 FLOP/Byte`（读 A+B），memory-bound |
| **瓶颈类型** | **memory-bound** |
| **kernel 数** | 2（同 Reduction） |

> 💡 **一句话总结**：Dot Product = "fused mul + 两阶段归约"，是 #4 Reduction 的融合变体。掌握它就掌握了"elementwise + reduce"的融合范式，MSE / attention score 都是这个骨架。

## 面试考点

- **手撕要求**：默写 fused grid-stride（`A[i]*B[i]`）+ 两阶段归约。
- **高频追问**：
  - **为什么融合 mul 进归约？** 省一次 global 读写（不用中间 buffer `C[i]`），减带宽。
  - **为什么用 double？** 紧容差 1e-5，大 N 时 float 误差超限。
  - **与 Reduction 的区别？** 仅累加项不同（`A[i]*B[i]` vs `input[i]`），骨架完全一致。
- **进阶延伸**：MSE（平方差归约）、FP16 dot product（#58，半精度累加精度保证）、attention score（Q·Kᵀ 归约）都是 fused reduce 变体。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | 树形归约，dot product 的基础组件 |
| 58 | [FP16 Dot Product](https://leetgpu.com/challenges/fp16-dot-product) | 中等 | 半精度归约 |
| 27 | [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error) | 中等 | 平方差归约的变体 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | 同题，可对比不同归约写法 |

> 💡 **选题思路**：元素乘 + block 归约，练习融合 kernel 与归约。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
