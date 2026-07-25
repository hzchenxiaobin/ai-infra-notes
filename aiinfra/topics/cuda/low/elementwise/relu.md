# LeetGPU ReLU 题解

> **面试考察度**：⭐⭐⭐ 最简激活函数，面试常作 ReLU/LN/RMSNorm 系列的入门
> **面试形式**：手写 elementwise + 讲清分支 vs 无分支

## 1. 题目概述

- **标题 / 题号**：ReLU（LeetGPU #21，easy）
- **链接**：https://leetgpu.com/challenges/relu
- **难度**：简单
- **标签**：CUDA、elementwise、激活函数、分支/无分支、memory-bound

**题意**：`output[i] = max(0, input[i])`，FP32，签名 `solve(const float* input, float* output, int N)`。容差 `1e-5`，性能测例 `N=25,000,000`。

## 2-4. 设计与实现

CPU 串行 `max(0, x)`。GPU：grid-stride loop，`fmaxf(x, 0.0f)`（无分支）。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// relu_submit.cu
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride)
        output[i] = fmaxf(input[i], 0.0f);   // 无分支，比 if(x>0) 快
}

extern "C" void solve(const float* input, float* output, int N) {
    int block = 256, grid = min((N + block - 1) / block, 4096);
    relu_kernel<<<grid, block>>>(input, output, N);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **grid-stride** | `for (i = tid; i < N; i += stride)` | 通用 elementwise 骨架 |
| **无分支** | `fmaxf(x, 0.0f)` | 避免 warp divergence（`if(x>0)` 会让正负 lane 分歧） |

> 💡 **关键洞察**：`fmaxf` 是无分支的硬件指令，比 `if (x > 0) x else 0` 快——后者在 warp 内正负值混合时产生 divergence（两分支都执行）。

## 5-6. 性能与复杂度

`O(N)`，`1 FLOP / 8B`，memory-bound。优化：float4 向量化、fused（Conv+ReLU epilogue）。

> 💡 **一句话总结**：ReLU = "grid-stride + fmaxf 无分支"，是激活函数 family 的最简形态。

## 面试考点

- **手撕要求**：默写 grid-stride + `fmaxf`。
- **高频追问**：为什么用 `fmaxf` 不用 `if`（warp divergence）；fused Conv+ReLU（epilogue）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 23 | [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) | 简单 | 带负斜率分支，对比无分支优化 |
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | 简单 | 融合 sigmoid+mul，练习 fused kernel |
| 68 | [Sigmoid](https://leetgpu.com/challenges/sigmoid) | 简单 | 纯数学函数逐元素，练习 exp 实现 |
| 65 | [GeGLU](https://leetgpu.com/challenges/geglu) | 简单 | GELU 激活，更复杂的逐元素融合 |

> 💡 **选题思路**：逐元素激活函数 family，练习分支/无分支 kernel 与合并访存。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
