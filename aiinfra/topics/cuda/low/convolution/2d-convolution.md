# LeetGPU 2D Convolution 题解

> **面试考察度**：⭐⭐⭐⭐ 2D 卷积是 CV 部署的核心，面试常考"shared memory halo + 常数内存"
> **面试形式**：手写 2D conv + 讲清 halo 四周边界与 tiling

## 1. 题目概述

- **标题 / 题号**：2D Convolution（LeetGPU #10，medium）
- **链接**：https://leetgpu.com/challenges/2d-convolution
- **难度**：中等
- **标签**：CUDA、Convolution、shared memory halo、常数内存、VALID padding、memory-bound

**题意**：2D cross-correlation，VALID padding。`output[i][j] = ΣΣ input[i+m][j+n] * kernel[m][n]`，输出 `(ir-kr+1) × (ic-kc+1)`，行主序展平。FP32，签名 `solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)`。容差 `1e-5`，性能测例 `input 3072×3072, kernel 15×15`。

## 2-4. 设计与实现

朴素版每输出像素读 `kr×kc` 个 input。优化：shared memory tile + 四周 halo，全 block 复用。kernel 存 `__constant__`（15×15=225 个 float，远小于 64KB 上限）。

![2D Convolution：shared memory halo](../../../images/cuda_convolution_overview.svg)

```cuda
// 2d_convolution_submit.cu —— 朴素版（每 thread 算一个输出像素）
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel,
                              float* output, int ir, int ic, int kr, int kc) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int out_rows = ir - kr + 1, out_cols = ic - kc + 1;
    if (i >= out_rows || j >= out_cols) return;
    float acc = 0.0f;
    for (int m = 0; m < kr; ++m)
        for (int n = 0; n < kc; ++n)
            acc += input[(i + m) * ic + (j + n)] * kernel[m * kc + n];
    output[i * out_cols + j] = acc;
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    dim3 block(16, 16);
    dim3 grid((out_cols + 15) / 16, (out_rows + 15) / 16);
    conv2d_kernel<<<grid, block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **2D 映射** | `i = by*bdy+ty, j = bx*bdx+tx` | 每 thread 算一个输出像素 |
| **卷积** | `acc += input[(i+m)*ic + (j+n)] * kernel[m*kc+n]` | 2D cross-correlation |
| **VALID** | `out = (ir-kr+1) × (ic-kc+1)` | 无 padding |

> 💡 **关键洞察**：2D conv 的 halo 在四周（上下左右各 `kr/2`/`kc/2`）。朴素版每个像素读 `kr×kc` 个 input，相邻像素重复读；shared tile + halo 让全 block 从 shared 读，global 读降到 `ir×ic`。`__constant__` 缓存 kernel 全广播（所有 thread 读相同 kernel）。

## 5-6. 性能与复杂度

`O(ir·ic·kr·kc)`，朴素版 `K` 大时 compute-bound。halo 优化把 global 读从 `ir·ic·kr·kc` 降到 `ir·ic`。

> 💡 **一句话总结**：2D Convolution = "1D conv 扩展到二维 + 四周 halo + `__constant__` kernel"，CV 部署核心。

## 面试考点

- **手撕要求**：默写朴素 2D conv + 讲清 halo 四周边界。
- **高频追问**：halo 四周怎么加载；kernel 为什么存 `__constant__`；与 im2col 对比（内存换计算）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 9 | [1D Convolution](https://leetgpu.com/challenges/1d-convolution) | 简单 | halo 基础入门 |
| 11 | [3D Convolution](https://leetgpu.com/challenges/3d-convolution) | 中等 | 体数据 halo 扩展 |
| 28 | [Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur) | 中等 | 可分离卷积，行列分离优化 |
| 42 | [2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling) | 中等 | 滑窗 reduction，类似 tiling 模式 |

> 💡 **选题思路**：shared memory halo + 常数内存，练习卷积类 kernel 的边界处理与 tiling。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
