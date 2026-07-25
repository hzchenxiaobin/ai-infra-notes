# LeetGPU RGB to Grayscale 题解

> **面试考察度**：⭐⭐⭐ 多通道加权求和，CV 预处理基础；练习 coalesced 多通道访存
> **面试形式**：手写 + 讲清"通道维在最后的布局与加权系数"

## 1. 题目概述

- **标题 / 题号**：RGB to Grayscale（LeetGPU #66，easy）
- **链接**：https://leetgpu.com/challenges/rgb-to-grayscale
- **难度**：简单
- **标签**：CUDA、elementwise、多通道、coalesced、memory-bound

**题意**：`gray = 0.299*R + 0.587*G + 0.114*B`（ITU-R BT.601 luma）。输入 `(H, W, 3)` 展平，输出 `(H*W,)`。FP32，签名 `solve(const float* input, float* output, int width, int height)`。容差 `1e-5`，性能测例 `width=2048, height=2048`（像素值 0-255）。

## 2-4. 设计与实现

每像素读 3 个通道（RGB 连续），加权求和写 1 个灰度值。布局 `(H, W, 3)` 展平 → 像素 `p` 的 RGB 在 `input[p*3], input[p*3+1], input[p*3+2]`。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// rgb_to_grayscale_submit.cu
#include <cuda_runtime.h>

__global__ void rgb2gray_kernel(const float* input, float* output, int width, int height) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (p >= total) return;
    float r = input[p * 3 + 0];
    float g = input[p * 3 + 1];
    float b = input[p * 3 + 2];
    output[p] = 0.299f * r + 0.587f * g + 0.114f * b;   // BT.601 luma
}

extern "C" void solve(const float* input, float* output, int width, int height) {
    int total = width * height;
    int block = 256, grid = (total + block - 1) / block;
    rgb2gray_kernel<<<grid, block>>>(input, output, width, height);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **像素映射** | `p = tid` | 每 thread 算一个像素 |
| **读 3 通道** | `input[p*3 + 0/1/2]` | RGB 连续存放 |
| **加权求和** | `0.299R + 0.587G + 0.114B` | BT.601 luma 系数 |

> 💡 **关键洞察**：RGB→Gray 是"多通道加权求和"——每像素读 3 个 float 写 1 个。coalesced 取决于布局：`(H,W,3)` 展平时相邻 thread 读 `p*3` 和 `(p+1)*3`，间隔 12B → 不完全 coalesced（3 个 sector）。`float4` 向量化可改善。

## 5-6. 性能与复杂度

`O(H·W)`，读 `3N` 写 `N`，memory-bound。优化：`float4` 读 4 像素、NHWC 布局优化。

> 💡 **一句话总结**：RGB to Grayscale = "多通道加权求和"，CV 预处理基础，练习通道索引与 coalesced。

## 面试考点

- **手撕要求**：默写 `0.299R+0.587G+0.114B`。
- **高频追问**：为什么用 BT.601 系数（人眼对绿光敏感）；`(H,W,3)` vs `(3,H,W)` 布局对 coalesced 的影响；float4 向量化。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 1 | [Vector Addition](https://leetgpu.com/challenges/vector-addition) | 简单 | grid-stride + coalesced 基础 |
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | 逐元素 kernel，分支开销 |
| 7 | [Color Inversion](https://leetgpu.com/challenges/color-inversion) | 简单 | 多通道逐元素变换 |
| 8 | [Matrix Addition](https://leetgpu.com/challenges/matrix-addition) | 简单 | 2D grid 逐元素 |

> 💡 **选题思路**：多通道加权求和，练习 coalesced 访存与通道索引。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
