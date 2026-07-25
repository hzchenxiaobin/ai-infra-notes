# LeetGPU Sigmoid 题解

> **面试考察度**：⭐⭐⭐ 数学函数逐元素，面试常追问 `__expf` 快速数学与精度
> **面试形式**：手写 + 讲清 `1/(1+exp(-x))` 的稳定实现

## 1. 题目概述

- **标题 / 题号**：Sigmoid Activation（LeetGPU #68，easy）
- **链接**：https://leetgpu.com/challenges/sigmoid
- **难度**：简单
- **标签**：CUDA、elementwise、激活函数、`__expf`、数值稳定、memory-bound

**题意**：`output[i] = 1 / (1 + exp(-input[i]))`，FP32，签名 `solve(const float* X, float* Y, size_t N)`。容差 `1e-5`，性能测例 `N=50,000,000`（`input` 在 `[-10,10]`）。

## 2-4. 设计与实现

朴素 `1/(1+expf(-x))`。优化：用 `__expf`（快速 exp，精度低但快）；对大负 x `exp(-x)` 溢出 → 用 `expf(-x)` 的 FP32 范围够（`x=-88` 时 `exp(88)` 才溢出，本题 `[-10,10]` 安全）。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// sigmoid_submit.cu
#include <cuda_runtime.h>
#include <cstdint>

__global__ void sigmoid_kernel(const float* X, float* Y, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) {
        float x = X[i];
        Y[i] = 1.0f / (1.0f + __expf(-x));   // __expf 快速 exp
    }
}

extern "C" void solve(const float* X, float* Y, size_t N) {
    int block = 256, grid = min((int)((N + block - 1) / block), 4096);
    sigmoid_kernel<<<grid, block>>>(X, Y, N);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **快速 exp** | `__expf(-x)` | 比 `expf` 快约 2×，精度 ~`1e-3`（本题容差 `1e-5` 可能紧张，保守用 `expf`） |
| **sigmoid** | `1/(1+exp(-x))` | 标准 sigmoid |

> ⚠️ `__expf` 精度约 `1e-3`，本题容差 `1e-5` 可能超限——保守用 `expf`（标准精度）。面试可讲"权衡 `__expf` 速度与精度"。

> 💡 **关键洞察**：sigmoid 是 SiLU/SwiGLU/GeGLU 的子组件。`__expf` 是 CUDA 的快速数学内建函数（`-use_fast_math`），比 `expf` 快但精度低——面试常追问"何时能用 `__expf`"。

## 5-6. 性能与复杂度

`O(N)`，memory-bound（exp 是计算但被访存掩盖）。优化：`__expf`、fused（SiLU = x*sigmoid(x) 融合）。

> 💡 **一句话总结**：Sigmoid = "grid-stride + `__expf` 快速数学"，是激活函数 family 的数学基础组件。

## 面试考点

- **手撕要求**：默写 `1/(1+exp(-x))` + 讲清 `__expf` 权衡。
- **高频追问**：`__expf` vs `expf`（速度 vs 精度）；数值稳定（大负 x 溢出）；fused SiLU。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | 最简激活函数对比 |
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | 简单 | 融合 sigmoid+mul，练习 fused kernel |
| 23 | [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) | 简单 | 分支激活对比 |
| 54 | [SwiGLU](https://leetgpu.com/challenges/swiglu) | 简单 | SwiGLU 使用 sigmoid 组件 |

> 💡 **选题思路**：逐元素数学函数，练习 __expf 快速数学与合并访存。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
