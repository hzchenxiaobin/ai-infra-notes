# LeetGPU 3D Convolution 题解

> **面试考察度**：⭐⭐⭐ 3D 卷积扩展 halo 到三维，体数据/视频处理基础
> **面试形式**：手写 3D conv + 讲清"高维 halo 的复杂度增长"

## 1. 题目概述

- **标题 / 题号**：3D Convolution（LeetGPU #11，medium）
- **链接**：https://leetgpu.com/challenges/3d-convolution
- **难度**：中等
- **标签**：CUDA、Convolution、3D halo、常数内存、VALID padding、memory-bound

**题意**：3D cross-correlation，VALID padding。输入 `(D, H, W)`，kernel `(kd, kh, kw)`，输出 `(D-kd+1, H-kh+1, W-kw+1)`，3D 张量（非展平）。FP32，签名 `solve(const float* input, const float* kernel, float* output, int input_depth, int input_rows, int input_cols, int kernel_depth, int kernel_rows, int kernel_cols)`。容差 `1e-5`，性能测例 `input 256×128×128, kernel 5×5×5`。

## 2-4. 设计与实现

2D conv 的高维扩展——halo 从四周变成六面（±kd/2, ±kh/2, ±kw/2）。朴素版每输出体素读 `kd×kh×kw` 个 input。

![2D Convolution：shared memory halo](../../../images/cuda_convolution_overview.svg)

```cuda
// 3d_convolution_submit.cu —— 朴素版
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                              int id, int ir, int ic, int kd, int kr, int kc) {
    int od = id - kd + 1, or_ = ir - kr + 1, oc = ic - kc + 1;
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= od || i >= or_ || j >= oc) return;
    float acc = 0.0f;
    for (int md = 0; md < kd; ++md)
        for (int mi = 0; mi < kr; ++mi)
            for (int mj = 0; mj < kc; ++mj)
                acc += input[((d+md)*ir + (i+mi))*ic + (j+mj)] * kernel[(md*kr + mi)*kc + mj];
    output[(d*or_ + i)*oc + j] = acc;
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_depth, int input_rows, int input_cols,
                      int kernel_depth, int kernel_rows, int kernel_cols) {
    int od = input_depth - kernel_depth + 1;
    int or_ = input_rows - kernel_rows + 1;
    int oc = input_cols - kernel_cols + 1;
    dim3 block(8, 8, 4);
    dim3 grid((oc+7)/8, (or_+7)/8, (od+3)/4);
    conv3d_kernel<<<grid, block>>>(input, kernel, output, input_depth, input_rows, input_cols,
                                    kernel_depth, kernel_rows, kernel_cols);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **3D 映射** | `(d, i, j)` 三维坐标 | 3D grid/block |
| **卷积** | 三重循环 `md, mi, mj` | 3D cross-correlation |
| **VALID** | `od/oe_/oc = input - kernel + 1` | 三维都缩小 |

> 💡 **关键洞察**：3D conv 的 halo 在六面，复杂度 `O(D·H·W·kd·kh·kw)`。shared memory halo 在 3D 下占用更大（`(BT+kd-1)·(BT+kr-1)·(BT+kc-1)`），需权衡 tile 大小与 shared 容量。

## 5-6. 性能与复杂度

`O(D·H·W·kd·kh·kw)`，halo 优化把 global 读降到 `D·H·W`。

> 💡 **一句话总结**：3D Convolution = "2D conv 扩展到三维 + 六面 halo"，体数据/视频处理基础。

## 面试考点

- **手撕要求**：默写朴素 3D conv + 讲清六面 halo。
- **高频追问**：3D halo 的 shared 占用怎么算；可分离 3D 卷积（三遍 1D）省多少。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 9 | [1D Convolution](https://leetgpu.com/challenges/1d-convolution) | 简单 | 1D halo 基础 |
| 10 | [2D Convolution](https://leetgpu.com/challenges/2d-convolution) | 中等 | 2D shared memory halo |
| 28 | [Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur) | 中等 | 可分离卷积 |
| 42 | [2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling) | 中等 | 滑窗 reduction |

> 💡 **选题思路**：3D 体数据 halo + 常数内存，练习高维卷积边界处理。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
