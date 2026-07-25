# LeetGPU GeGLU 题解

> **面试考察度**：⭐⭐⭐ GELU 门控激活，GPT-2 FFN 的核心组件；练习 erf 与门控融合
> **面试形式**：手写 fused GeGLU + 讲清"GELU 精确 erf vs tanh 近似"

## 1. 题目概述

- **标题 / 题号**：Gaussian Error Gated Linear Unit（LeetGPU #65，easy）
- **链接**：https://leetgpu.com/challenges/geglu
- **难度**：简单
- **标签**：CUDA、elementwise、fused activation、门控、GELU、erf、memory-bound

**题意**：输入 `input[N]` 拆两半 `x1 = input[0..N/2-1]`、`x2 = input[N/2..N-1]`，输出 `output[i] = x1[i] * GELU(x2[i])`，其中 `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`（**精确 erf**，非 tanh 近似）。`output` 长度 `N/2`，FP32，签名 `solve(const float* input, float* output, int N)`。容差 `1e-4`，性能测例 `N=1,000,000`。

## 2-4. 设计与实现

门控机制（与 SwiGLU 对称）：`x1` 是值，`x2` 经 GELU 激活后与 `x1` 相乘。GELU 用精确 `erf`（CUDA 提供 `erff` 内建函数）。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// geglu_submit.cu
#include <cuda_runtime.h>
#include <cmath>

__global__ void geglu_kernel(const float* input, float* output, int N) {
    int half = N / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < half; i += stride) {
        float x1 = input[i];           // 值
        float x2 = input[i + half];    // 门控
        float gelu = 0.5f * x2 * (1.0f + erff(x2 / 1.41421356f));   // 精确 erf
        output[i] = x1 * gelu;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int half = N / 2;
    int block = 256, grid = min((half + block - 1) / block, 4096);
    geglu_kernel<<<grid, block>>>(input, output, N);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **split** | `x1 = input[i]; x2 = input[i + half]` | 输入拆两半 |
| **GELU 精确** | `0.5*x*(1+erf(x/√2))` | 用 `erff` 内建函数，**非 tanh 近似** |
| **门控乘** | `x1 * gelu` | x1 值 × x2 的 GELU 激活 |

> ⚠️ **与 GPT-2 的区别**：GPT-2 用 GELU 的 **tanh 近似** `0.5x(1+tanh(√(2/π)(x+0.0447x³)))`；本题用**精确 erf**。面试要分清两者——erf 精度高但慢，tanh 近似快但精度低。

> 💡 **关键洞察**：GeGLU = x1 * GELU(x2)，与 SwiGLU（SiLU(x1)*x2）对称——门控方向相反。GELU 的精确 erf 与 tanh 近似是面试常考点。

## 5-6. 性能与复杂度

`O(N)`，`erff` 比 `expf` 略慢但仍被访存掩盖，memory-bound。

> 💡 **一句话总结**：GeGLU = "fused x1*GELU(x2) 门控激活"，GELU 用精确 erf。与 SwiGLU 对称（门控方向相反）。

## 面试考点

- **手撕要求**：默写 `0.5*x*(1+erf(x/√2))` + 门控乘。
- **高频追问**：GELU 精确 erf vs tanh 近似（精度 vs 速度）；GeGLU vs SwiGLU（GELU vs SiLU，门控方向）；GPT-2 用哪个（tanh 近似）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 54 | [SwiGLU](https://leetgpu.com/challenges/swiglu) | 简单 | 门控激活对比，SiLU vs GELU |
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | 简单 | SiLU 激活，GeGLU 的子组件 |
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | 最简激活对比 |
| 65 | [GeGLU](https://leetgpu.com/challenges/geglu) | 简单 | 同题，可对比不同实现 |

> 💡 **选题思路**：GELU 门控激活 + 逐元素融合，练习 fused MLP 组件 kernel。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
