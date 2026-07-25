# LeetGPU Histogramming 题解

> **面试考察度**：⭐⭐⭐⭐ 直方图是 shared memory + atomic 的招牌题，面试常考"原子冲突怎么减少"
> **面试形式**：手写 shared memory 直方图 + 讲清"privatization 减少全局 atomic"

## 1. 题目概述

- **标题 / 题号**：Histogramming（LeetGPU #13，medium）
- **链接**：https://leetgpu.com/challenges/histogramming
- **难度**：中等
- **标签**：CUDA、Histogram、shared memory、atomicAdd、privatization、memory-bound

**题意**：统计 `input[N]`（INT32）中每个值 `[0, num_bins)` 的出现次数到 `histogram[num_bins]`（INT32），越界值忽略。需先清零 histogram。签名 `solve(const int* input, int* histogram, int N, int num_bins)`。容差 `1e-5`，性能测例 `N=50,000,000, num_bins=256`（值 0-255）。

## 2-4. 设计与实现

朴素版：全局 `atomicAdd(&histogram[v], 1)`——`N` 次 atomic 到 `num_bins` 个地址，冲突严重。优化：**privatization**——每 block 在 shared memory 维护私有直方图（`atomicAdd` 到 shared，快），最后合并到全局。

![Elementwise Kernel 数据流](../../../images/cuda_elementwise_overview.svg)

```cuda
// histogramming_submit.cu —— shared memory privatization
#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int s_hist[];
    int tid = threadIdx.x;
    // 初始化 shared 直方图
    for (int i = tid; i < num_bins; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // grid-stride 统计到 shared（atomic 到 shared 比 global 快）
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) {
        int v = input[i];
        if (v >= 0 && v < num_bins) atomicAdd(&s_hist[v], 1);
    }
    __syncthreads();

    // 合并到全局（每 block 贡献累加）
    for (int i = tid; i < num_bins; i += blockDim.x)
        atomicAdd(&histogram[i], s_hist[i]);
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, num_bins * sizeof(int));
    int block = 256, grid = min((N + block - 1) / block, 4096);
    histogram_kernel<<<grid, block, num_bins * sizeof(int)>>>(input, histogram, N, num_bins);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **privatization** | `s_hist[num_bins]` shared | 每 block 私有直方图 |
| **shared atomic** | `atomicAdd(&s_hist[v], 1)` | atomic 到 shared（快，无全局竞争） |
| **合并** | `atomicAdd(&histogram[i], s_hist[i])` | 每 block 贡献累加到全局 |

> 💡 **关键洞察**：直方图的瓶颈是 atomic 冲突——`N` 个值竞争 `num_bins` 个 bin。privatization 把竞争从全局（所有 block）降到 block 内（256 thread），shared atomic 比全局 atomic 快 ~10×。冲突仍存在（同 bin 的值串行化），但量级大降。

## 5-6. 性能与复杂度

`O(N)`，atomic 串行化是瓶颈。优化：privatization、bin 分桶减少冲突、scan 替代 atomic。

> 💡 **一句话总结**：Histogramming = "shared memory privatization + atomicAdd"，privatization 把全局冲突降到 block 内。

## 面试考点

- **手撕要求**：默写 shared privatization + 合并。
- **高频追问**：为什么用 shared 而非全局 atomic（冲突量级）；privatization 省多少；bin 冲突怎么进一步优化（分桶/scan）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 43 | [Count Array Element](https://leetgpu.com/challenges/count-array-element) | 中等 | 计数归约 + atomic，对比归约与 atomic |
| 44 | [Count 2D Array Element](https://leetgpu.com/challenges/count-2d-array-element) | 中等 | 2D 计数，扩展到多维 atomic |
| 29 | [Top K Selection](https://leetgpu.com/challenges/top-k-selection) | 中等 | bitonic 排序 + 堆归约，相关并行模式 |
| 36 | [Radix Sort](https://leetgpu.com/challenges/radix-sort) | 中等 | Radix Sort，histogram + scan 综合 |

> 💡 **选题思路**：shared memory 直方图 + atomic 冲突，练习计数类并行模式。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
