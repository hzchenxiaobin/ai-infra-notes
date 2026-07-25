# LeetGPU RoPE Embedding 题解

> **面试考察度**：⭐⭐⭐⭐ RoPE 是 LLaMA 的位置编码，面试常考"rotate_half 的索引映射"
> **面试形式**：手写 + 讲清"复数旋转的实数实现"

## 1. 题目概述

- **标题 / 题号**：Rotary Positional Embedding（LeetGPU #61，medium）
- **链接**：https://leetgpu.com/challenges/rope-embedding
- **难度**：中等
- **标签**：CUDA、RoPE、位置编码、复数旋转、elementwise、memory-bound

**题意**：`output = Q * cos + rotate_half(Q) * sin`。`rotate_half(Q)` = `concat(-Q[D/2:], Q[:D/2])`（前后半交换并取反前半）。`Q` 为 `(M, D)`，`cos`/`sin` 为 `(M, D)` 或 `(D,)`。FP32，签名 `solve(const float* Q, const float* cos, const float* sin, float* output, int M, int D)`。容差 `1e-4`，性能测例 `M=1048576, D=128`。

## 2-4. 设计与实现

elementwise kernel，每元素查 `cos/sin` 并做 `rotate_half`。`rotate_half` 对维度 `D` 操作：前半 `d < D/2` 取 `-Q[d + D/2]`，后半 `d >= D/2` 取 `Q[d - D/2]`。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// rope_embedding_submit.cu
#include <cuda_runtime.h>

__global__ void rope_kernel(const float* Q, const float* cos, const float* sin,
                           float* output, int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * D;
    if (idx >= total) return;
    int half = D / 2;
    int d = idx % D;          // 特征维
    int m = idx / D;          // 样本维

    float q = Q[idx];
    float rotate;
    if (d < half) rotate = -Q[m * D + d + half];      // 前半取后半的负
    else          rotate =  Q[m * D + d - half];      // 后半取前半
    output[idx] = q * cos[m * D + d] + rotate * sin[m * D + d];
}

extern "C" void solve(const float* Q, const float* cos, const float* sin,
                      float* output, int M, int D) {
    int total = M * D;
    int block = 256, grid = (total + block - 1) / block;
    rope_kernel<<<grid, block>>>(Q, cos, sin, output, M, D);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **rotate_half** | `d < D/2 ? -Q[d+half] : Q[d-half]` | 前后半交换 + 前半取负 |
| **旋转** | `Q*cos + rotate*sin` | 复数旋转的实数形式 |

> 💡 **关键洞察**：RoPE 把位置编码融入复数旋转——对每对特征 `(d, d+D/2)` 视为复数实/虚部，乘 `e^{iθ}` = `cos θ + i sin θ`。`rotate_half` 是复数旋转的实数实现：`(a, b) → (a·cos - b·sin, a·sin + b·cos)`。LLaMA 默认位置编码。

## 5-6. 性能与复杂度

`O(M·D)`，读 `Q+cos+sin` 写 `output`，memory-bound。常与 QKV projection 融合。

> 💡 **一句话总结**：RoPE Embedding = "rotate_half + elementwise 旋转"，复数位置编码的实数实现。

## 面试考点

- **手撕要求**：默写 `rotate_half` 索引。
- **高频追问**：rotate_half 的物理意义（复数旋转）；为什么 RoPE 比 sinusoidal 好（相对位置）；与 GPT-2 的 learned PE 对比。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 106 | [Token Embedding](https://leetgpu.com/challenges/token-embedding-layer) | 中等 | 位置嵌入的另一种实现 |
| 54 | [SwiGLU](https://leetgpu.com/challenges/swiglu) | 简单 | 融合 elementwise 进阶 |
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | 简单 | fused elementwise |
| 50 | [RMS Normalization](https://leetgpu.com/challenges/rms-normalization) | 中等 | 归约 + elementwise |

> 💡 **选题思路**：复数旋转 + elementwise，练习位置编码的并行实现。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
