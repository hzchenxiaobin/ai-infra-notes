# LeetGPU Leaky ReLU 题解

> **面试考察度**：⭐⭐⭐ ReLU 的负斜率变体，面试常对比"分支 vs 无分支"写法
> **面试形式**：手写 + 讲清无分支实现（`x > 0 ? x : 0.01*x`）

## 1. 题目概述

- **标题 / 题号**：Leaky ReLU（LeetGPU #23，easy）
- **链接**：https://leetgpu.com/challenges/leaky-relu
- **难度**：简单
- **标签**：CUDA、elementwise、激活函数、无分支、memory-bound

**题意**：`f(x) = x if x > 0 else 0.01*x`（`alpha=0.01` 固定），FP32，签名 `solve(const float* input, float* output, int N)`。容差 `1e-6`（紧），性能测例 `N=50,000,000`。

## 2-4. 设计与实现

无分支实现：`fmaxf(x, 0.01f * x)`——当 `x>0` 时 `fmaxf` 取 `x`，当 `x<0` 时取 `0.01*x`（因 `0.01x > x` 对负 x）。等价且无 divergence。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// leaky_relu_submit.cu
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    const float alpha = 0.01f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        output[i] = fmaxf(x, alpha * x);   // 无分支：x>0 取 x，x<0 取 0.01x
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int block = 256, grid = min((N + block - 1) / block, 4096);
    leaky_relu_kernel<<<grid, block>>>(input, output, N);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **无分支** | `fmaxf(x, alpha*x)` | `x>0`: `fmaxf(x, 0.01x)=x`；`x<0`: `fmaxf(x, 0.01x)=0.01x`（0.01x > x） |
| **grid-stride** | `for (i = tid; i < N; i += stride)` | 通用骨架 |

> 💡 **关键洞察**：Leaky ReLU 的无分支技巧是 `fmaxf(x, alpha*x)`——利用"对负 x，`alpha*x > x`"的性质，一个 `fmaxf` 替代 `if-else`，消除 warp divergence。这是面试加分点。

## 5-6. 性能与复杂度

`O(N)`，memory-bound。与 ReLU 同骨架，多一次乘法。

> 💡 **一句话总结**：Leaky ReLU = "ReLU + 负斜率无分支技巧 `fmaxf(x, alpha*x)`"。

## 面试考点

- **手撕要求**：默写无分支 `fmaxf(x, alpha*x)`。
- **高频追问**：为什么无分支比 `if` 快（warp divergence）；alpha 对精度的影响。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | 最简激活函数对比，无负斜率 |
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | 简单 | 融合激活函数，练习 __expf |
| 68 | [Sigmoid](https://leetgpu.com/challenges/sigmoid) | 简单 | 数学函数逐元素，练习 exp 实现 |
| 65 | [GeGLU](https://leetgpu.com/challenges/geglu) | 简单 | GELU 门控变体，更复杂激活 |

> 💡 **选题思路**：逐元素激活函数 family，练习分支/无分支 kernel 与合并访存。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
