# LeetGPU SwiGLU 题解

> **面试考察度**：⭐⭐⭐⭐ SwiGLU 是 LLaMA FFN 的核心激活，面试常考"门控激活怎么融合"
> **面试形式**：手写 fused SwiGLU + 讲清"split input + 门控乘法"

## 1. 题目概述

- **标题 / 题号**：Swish-Gated Linear Unit（LeetGPU #54，easy）
- **链接**：https://leetgpu.com/challenges/swiglu
- **难度**：简单
- **标签**：CUDA、elementwise、fused activation、门控、SwiGLU、memory-bound

**题意**：输入 `input[N]` 拆成两半 `x1 = input[0..N/2-1]`、`x2 = input[N/2..N-1]`，输出 `output[i] = (x1[i] * sigmoid(x1[i])) * x2[i]`（即 `SiLU(x1) * x2`），`output` 长度 `N/2`。FP32，签名 `solve(const float* input, float* output, int N)`。容差 `atol=1e-4, rtol=1e-5`，性能测例 `N=100,000`。

## 2-4. 设计与实现

门控机制：`x1` 经 SiLU 激活后与 `x2` 相乘——`x2` 是门控信号。fused：一个 kernel 读 `x1[i]` 和 `x2[i]`，算 `SiLU(x1)*x2` 写回。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// swiglu_submit.cu
#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int N) {
    int half = N / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < half; i += stride) {
        float x1 = input[i];           // 前半
        float x2 = input[i + half];    // 后半（门控）
        float silu = x1 * (1.0f / (1.0f + expf(-x1)));
        output[i] = silu * x2;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int half = N / 2;
    int block = 256, grid = min((half + block - 1) / block, 4096);
    swiglu_kernel<<<grid, block>>>(input, output, N);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **split** | `x1 = input[i]; x2 = input[i + half]` | 输入拆两半，后半是门控 |
| **SiLU** | `x1 * sigmoid(x1)` | 融合激活 |
| **门控乘** | `silu * x2` | x2 门控缩放 |
| **输出半长** | `output[i]`，`i < half` | 输出长度 N/2 |

> 💡 **关键洞察**：SwiGLU = SiLU(x1) * x2，门控激活。LLaMA FFN 用 `SwiGLU(W1(x), W3(x))`——两个 GEMM 的输出做门控乘法。fused 把 sigmoid+mul+门控合并，省中间 buffer。

## 5-6. 性能与复杂度

`O(N)`，读 `N` 写 `N/2`，memory-bound。在 FFN 中常作 GEMM epilogue 融合。

> 💡 **一句话总结**：SwiGLU = "fused SiLU(x1)*x2 门控激活"，是 LLaMA FFN 的核心组件，演示门控融合。

## 面试考点

- **手撕要求**：默写 split + `SiLU(x1)*x2`。
- **高频追问**：SwiGLU vs GeGLU（SiLU vs GELU 激活）；为什么 LLaMA 用 SwiGLU（比 ReLU/GELU 表现好）；FFN 中怎么融合（GEMM epilogue）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | 简单 | SwiGLU 的激活组件 |
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | 最简激活对比 |
| 65 | [GeGLU](https://leetgpu.com/challenges/geglu) | 简单 | GELU 门控变体 |
| 84 | [SwiGLU MLP Block](https://leetgpu.com/challenges/swiglu-mlp-block) | 中等 | SwiGLU 的完整 MLP 应用 |

> 💡 **选题思路**：融合激活 + 门控乘法，练习 fused MLP 组件 kernel。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
