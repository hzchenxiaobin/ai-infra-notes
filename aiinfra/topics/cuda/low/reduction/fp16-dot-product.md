# LeetGPU FP16 Dot Product 题解

> **面试考察度**：⭐⭐⭐ 半精度归约，练习 `__half` 类型转换与 FP32 累加精度保证
> **面试形式**：手写 + 讲清"为什么 FP16 输入要 FP32 累加"

## 1. 题目概述

- **标题 / 题号**：FP16 Dot Product（LeetGPU #58，medium）
- **链接**：https://leetgpu.com/challenges/fp16-dot-product
- **难度**：中等
- **标签**：CUDA、Dot Product、FP16、`__half`、FP32 累加、两阶段归约、memory-bound

**题意**：`result[0] = Σ A[i]·B[i]`，A/B 为 FP16（`__half`，传 `uint16_t`），result 为 FP16 单标量。FP32 累加后转回 FP16。签名 `solve(const uint16_t* A, const uint16_t* B, uint16_t* result, int N)`。容差 `0.05`，性能测例 `N=100,000,000`。

## 2-4. 设计与实现

与 #17 Dot Product 骨架一致（fused mul + 两阶段归约），差别：① A/B 类型 `__half`；② 累加用 `float`（FP16 累加误差超容差）；③ 结果转回 `__half`。

![两阶段归约：block reduce → global reduce](../../../images/cuda_reduction_overview.svg)

```cuda
// fp16_dot_product_submit.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS 4096

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffff, val, o);
    return val;
}
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void dot_block_kernel(const __half* A, const __half* B, float* partial, int N) {
    __shared__ float shared[NUM_WARPS + 1];
    float local_sum = 0.0f;   // ← FP32 累加
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N; i += gridDim.x * BLOCK_SIZE)
        local_sum += __half2float(A[i]) * __half2float(B[i]);   // 转 FP32 后乘加
    if (threadIdx.x == 0) partial[blockIdx.x] = block_reduce_sum(local_sum, shared);
}

__global__ void dot_final_kernel(const float* partial, __half* result, int M) {
    __shared__ float shared[NUM_WARPS + 1];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < M; i += BLOCK_SIZE) local_sum += partial[i];
    float total = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) result[0] = __float2half(total);   // ← 转回 FP16
}

extern "C" void solve(const uint16_t* A, const uint16_t* B, uint16_t* result, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > MAX_BLOCKS) numBlocks = MAX_BLOCKS;
    if (numBlocks < 1) numBlocks = 1;
    float* partial; cudaMalloc(&partial, numBlocks * sizeof(float));
    dot_block_kernel<<<numBlocks, BLOCK_SIZE>>>((const __half*)A, (const __half*)B, partial, N);
    dot_final_kernel<<<1, BLOCK_SIZE>>>(partial, (__half*)result, numBlocks);
    cudaFree(partial);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **FP16 读** | `A[i]` (`__half`) | 输入半精度 |
| **转 FP32** | `__half2float(A[i])` | 累加前转 FP32 |
| **FP32 累加** | `local_sum += ...` | 精度保证 |
| **FP16 写回** | `__float2half(total)` | 结果转回 FP16 |

> 💡 **关键洞察**：FP16 dot product 与 #17 的唯一差别是"输入 `__half`、累加 `float`、输出 `__half`"。FP16 尾数仅 10 位，直接累加 `N=1e8` 个元素误差爆炸；FP32 累加是混合精度的标准做法（与 #57 FP16 GEMM 一致）。

## 5-6. 性能与复杂度

`O(N)`，FP16 字节减半（读 `4N` vs FP32 的 `8N`），memory-bound。可进一步用 Tensor Core（`mma.sync` FP16 归约）。

> 💡 **一句话总结**：FP16 Dot Product = "#17 + `__half` 转换 + FP32 累加"，混合精度的归约变体。

## 面试考点

- **手撕要求**：默写 `__half2float` 转换 + FP32 累加 + 两阶段归约。
- **高频追问**：为什么 FP32 累加（尾数 10 位不够）；`__half2float`/`__float2half` 开销；Tensor Core 加速。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | 树形归约基础组件 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | FP32 版 dot product 对比 |
| 57 | [FP16 Batched Matrix Multiplication](https://leetgpu.com/challenges/fp16-batched-matmul) | 中等 | FP16 + Tensor Core，半精度 GEMM |
| 27 | [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error) | 中等 | 归约在损失函数中的应用 |

> 💡 **选题思路**：半精度归约，练习 __half 类型转换与 FP32 累加精度保证。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
