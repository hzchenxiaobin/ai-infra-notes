# LeetGPU Gaussian Blur 题解

> **面试考察度**：⭐⭐⭐ 可分离卷积的招牌题，练习"2D conv 拆成两遍 1D conv 省 FLOP"
> **面试形式**：手写可分离卷积 + 讲清"SAME padding 与行列分离"

## 1. 题目概述

- **标题 / 题号**：Gaussian Blur（LeetGPU #28，medium）
- **链接**：https://leetgpu.com/challenges/gaussian-blur
- **难度**：中等
- **标签**：CUDA、Convolution、可分离卷积、SAME padding、shared memory、memory-bound

**题意**：2D 卷积，**SAME padding**（`pad_h = kernel_rows//2, pad_w = kernel_cols//2`，输出与输入同尺寸）。FP32，签名 `solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)`。容差 `1e-5`，性能测例 `input 512×512, kernel 7×7`。

> 💡 **可分离性**：若 kernel 是高斯核（可分解为 `col_kernel × row_kernel` 的外积），2D 卷积可拆成两遍 1D：先沿行卷积，再沿列卷积。FLOP 从 `H·W·kr·kc` 降到 `H·W·(kr+kc)`，`k=7` 时省 `7/2 ≈ 3.5×`。本题 kernel 不保证可分离（随机生成），但面试标准做法是可分离。

## 2-4. 设计与实现

**SAME padding**：输出与输入同尺寸，边界补 0。朴素 2D conv（同 #10 但 padding 非零）。

![2D Convolution：shared memory halo](../../../images/cuda_convolution_overview.svg)

```cuda
// gaussian_blur_submit.cu —— 朴素 2D conv + SAME padding
#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(const float* input, const float* kernel,
                                     float* output, int ir, int ic, int kr, int kc) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ir || j >= ic) return;
    int pad_h = kr / 2, pad_w = kc / 2;
    float acc = 0.0f;
    for (int m = 0; m < kr; ++m)
        for (int n = 0; n < kc; ++n) {
            int ih = i + m - pad_h;
            int iw = j + n - pad_w;
            if (ih >= 0 && ih < ir && iw >= 0 && iw < ic)
                acc += input[ih * ic + iw] * kernel[m * kc + n];
        }
    output[i * ic + j] = acc;
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 block(16, 16);
    dim3 grid((input_cols + 15) / 16, (input_rows + 15) / 16);
    gaussian_blur_kernel<<<grid, block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **SAME padding** | `pad_h = kr/2` | 输出 = 输入尺寸 |
| **边界检查** | `if (ih>=0 && ih<ir ...)` | 越界补 0（不累加） |
| **卷积** | `acc += input[...] * kernel[...]` | 2D cross-correlation |

> 💡 **关键洞察**：Gaussian Blur 的 SAME padding 让输出与输入同尺寸（vs #10 的 VALID 缩小）。可分离优化把 2D conv 拆成两遍 1D（行 + 列），FLOP 从 `k²` 降到 `2k`——这是高斯模糊的标准优化。

## 5-6. 性能与复杂度

`O(H·W·kr·kc)`（朴素），可分离降到 `O(H·W·(kr+kc))`。memory-bound。

> 💡 **一句话总结**：Gaussian Blur = "2D conv + SAME padding"，可分离卷积拆成两遍 1D 省 FLOP。

## 面试考点

- **手撕要求**：默写 SAME padding 2D conv + 讲清可分离优化。
- **高频追问**：SAME vs VALID padding；可分离卷积省多少 FLOP（`k²→2k`）；为什么高斯核可分离（外积分解）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 10 | [2D Convolution](https://leetgpu.com/challenges/2d-convolution) | 中等 | 2D shared memory halo + tiling |
| 9 | [1D Convolution](https://leetgpu.com/challenges/1d-convolution) | 简单 | 1D 卷积，halo 基础 |
| 42 | [2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling) | 中等 | 滑窗 reduction，类似 tiling 模式 |
| 11 | [3D Convolution](https://leetgpu.com/challenges/3d-convolution) | 中等 | 3D 体数据 halo |

> 💡 **选题思路**：可分离卷积，练习 shared memory + 常数内存。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
