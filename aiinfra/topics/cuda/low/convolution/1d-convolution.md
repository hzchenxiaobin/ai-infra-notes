# LeetGPU 1D Convolution 题解

> **面试考察度**：⭐⭐⭐ 1D 卷积是 shared memory halo 的入门题，音频/时序处理基础
> **面试形式**：手写 1D conv + 讲清"halo 区域避免重复读 global"

## 1. 题目概述

- **标题 / 题号**：1D Convolution（LeetGPU #9，easy）
- **链接**：https://leetgpu.com/challenges/1d-convolution
- **难度**：简单
- **标签**：CUDA、Convolution、shared memory halo、常数内存、VALID padding、memory-bound

**题意**：1D cross-correlation（无翻转），VALID padding（无填充，stride 1）。`output[i] = Σ_{m=0}^{K-1} input[i+m] * kernel[m]`，输出长度 `input_size - kernel_size + 1`。FP32，签名 `solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size)`。容差 `1e-4`，性能测例 `input_size=1500000, kernel_size=2047`。

## 2-4. 设计与实现

朴素版每输出元素读 `K` 个 input——重复读 `K` 次。优化：shared memory halo——把 input 分 tile，每 tile 加载到 shared（含两侧 halo），全 block 复用。kernel 存 `__constant__` 内存（小且全广播）。

![2D Convolution：shared memory halo](../../../images/cuda_convolution_overview.svg)

```cuda
// 1d_convolution_submit.cu —— 朴素版（每个 thread 算一个输出元素）
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel,
                              float* output, int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = input_size - kernel_size + 1;
    if (i >= out_size) return;
    float acc = 0.0f;
    for (int m = 0; m < kernel_size; ++m)
        acc += input[i + m] * kernel[m];
    output[i] = acc;
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    int out_size = input_size - kernel_size + 1;
    int block = 256, grid = (out_size + block - 1) / block;
    conv1d_kernel<<<grid, block>>>(input, kernel, output, input_size, kernel_size);
}
```

> ⚠️ 朴素版每个输出元素读 `K` 个 input，`input` 被相邻输出重复读 `K` 次。优化版用 shared halo（tile + 两侧 `K/2` halo），全 block 复用——本题 `K=2047` 大，halo 优化收益显著。

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **输出映射** | `i = tid` | 每 thread 算一个 output[i] |
| **卷积累加** | `acc += input[i+m] * kernel[m]` | cross-correlation（无翻转） |
| **VALID** | `out_size = input_size - K + 1` | 无 padding，输出缩小 |

> 💡 **关键洞察**：1D conv 的优化核心是 halo——shared tile 加载时多读两侧 `K/2` 个元素，让边界元素只从 shared 读（而非每个输出都从 global 重复读）。`K` 越大，halo 优化收益越高。

## 5-6. 性能与复杂度

`O(N·K)`，朴素版算术强度 `2K / 8B`（K 大时 compute-bound）。halo 优化把 global 读从 `N·K` 降到 `N`。

> 💡 **一句话总结**：1D Convolution = "cross-correlation + shared memory halo"，halo 消除重复读 global，kernel 存 `__constant__`。

## 面试考点

- **手撕要求**：默写朴素 1D conv + 讲清 halo 优化思路。
- **高频追问**：cross-correlation vs convolution（是否翻转 kernel）；halo 怎么消除重复读；kernel 为什么存 `__constant__`（小且全广播）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 10 | [2D Convolution](https://leetgpu.com/challenges/2d-convolution) | 中等 | halo 扩展到二维 |
| 11 | [3D Convolution](https://leetgpu.com/challenges/3d-convolution) | 中等 | 体数据 halo |
| 90 | [Causal Depthwise Conv1d](https://leetgpu.com/challenges/causal-depthwise-conv1d) | 中等 | 因果卷积变体 |
| 28 | [Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur) | 中等 | 可分离卷积 |

> 💡 **选题思路**：1D shared memory halo，练习卷积边界填充与 tile 加载。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
