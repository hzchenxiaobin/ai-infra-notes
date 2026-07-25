# LeetGPU SiLU 题解

> **面试考察度**：⭐⭐⭐ 融合激活函数，LLaMA 的 FFN 组件；练习 fused kernel 思想
> **面试形式**：手写 fused `x * sigmoid(x)` + 讲清"为什么融合省一次访存"

## 1. 题目概述

- **标题 / 题号**：Sigmoid Linear Unit（LeetGPU #52，easy）
- **链接**：https://leetgpu.com/challenges/silu
- **难度**：简单
- **标签**：CUDA、elementwise、fused activation、SiLU/Swish、memory-bound

**题意**：`output[i] = input[i] * sigmoid(input[i])` = `x / (1 + exp(-x))`，FP32，签名 `solve(const float* input, float* output, int N)`。容差 `1e-5`，性能测例 `N=50,000`（`input` 在 `[-50,50]`）。

## 2-4. 设计与实现

朴素版：先算 `sigmoid(x)` 存中间 buffer，再乘 `x`——读 2 遍。**Fused 版**：一个 kernel 内 `x * sigmoid(x)`，只读 1 遍 input。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// silu_submit.cu
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        output[i] = x * (1.0f / (1.0f + expf(-x)));   // fused: 一次读，一次写
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int block = 256, grid = min((N + block - 1) / block, 4096);
    silu_kernel<<<grid, block>>>(input, output, N);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **fused** | `x * (1/(1+exp(-x)))` | sigmoid 和乘法在同一 thread，省中间 buffer 的读写 |
| **grid-stride** | `for (i = tid; i < N; i += stride)` | 通用骨架 |

> 💡 **关键洞察**：SiLU 的 fused 实现把"算 sigmoid + 乘 x"融合进一个 kernel——朴素版需读 input 2 遍（算 sigmoid + 乘），fused 版只读 1 遍，省一半 HBM 带宽。这是 kernel fusion 的典型示例。

## 5-6. 性能与复杂度

`O(N)`，fused 版访存 `2N×4B`（读 1 + 写 1），朴素版 `4N×4B`（读 2 + 写 1 + 中间）——fused 省 50% 带宽。

> 💡 **一句话总结**：SiLU = "fused x*sigmoid(x)"，是 LLaMA FFN 的激活组件，演示 kernel fusion 省 HBM 带宽。

## 面试考点

- **手撕要求**：默写 fused `x * sigmoid(x)`。
- **高频追问**：为什么 fused（省中间 buffer 读写）；与 ReLU 对比（无分支但多了 exp）；LLaMA 用 SiLU（GPT-2 用 GELU）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | 最简激活函数对比 |
| 68 | [Sigmoid](https://leetgpu.com/challenges/sigmoid) | 简单 | silu 的组件 |
| 54 | [SwiGLU](https://leetgpu.com/challenges/swiglu) | 简单 | 融合激活 + 门控进阶 |
| 23 | [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) | 简单 | 分支激活对比 |

> 💡 **选题思路**：融合 sigmoid + mul 逐元素，练习 fused activation kernel。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
