# LeetGPU MoE Top-K Gating 题解

> **面试考察度**：⭐⭐⭐⭐ MoE 是大模型扩展的核心，面试常考"top-k 选择 + softmax 路由"
> **面试形式**：手写 + 讲清"per-token 选 top-k expert + 重归一化"

## 1. 题目概述

- **标题 / 题号**：MoE Top-K Gating（LeetGPU #67，medium）
- **链接**：https://leetgpu.com/challenges/moe-topk-gating
- **难度**：中等
- **标签**：CUDA、MoE、Top-K、softmax、routing、排序归约

**题意**：每 token（共 `M` 个）从 `E` 个 expert 的 logits 中选 top-`k` 个，对选中的 `k` 个做 softmax 归一化。输出 `topk_weights[M×k]`（FP32）和 `topk_indices[M×k]`（INT32，降序排列）。签名 `solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k)`。容差 `1e-5`，性能测例 `M=1024, E=64, k=2`。

## 2-4. 设计与实现

一个 block 处理一个 token（`M` 个 block）。block 内：对 `E` 个 logits 找 top-k（朴素：`k` 轮选择排序），对选中的 `k` 个做 softmax，按降序写回。

![Scaled Dot-Product Attention 数据流](../../../images/cuda_softmax_attention_overview.svg)

```cuda
// moe_topk_gating_submit.cu —— 朴素版（一个 block 一个 token，k 轮选择）
#include <cuda_runtime.h>

__global__ void moe_gating_kernel(const float* logits, float* topk_weights, int* topk_indices,
                                  int M, int E, int k) {
    int m = blockIdx.x;
    if (m >= M) return;
    const float* lm = logits + m * E;
    float* wm = topk_weights + m * k;
    int* im = topk_indices + m * k;

    // 简化：单 thread 做 top-k 选择（生产用并行 sort）
    if (threadIdx.x == 0) {
        // 拷贝到 local（简化：直接在 global 操作，标记已选）
        bool selected[64] = {false};   // 假设 E <= 64
        for (int sel = 0; sel < k; ++sel) {
            int best = -1; float best_val = -INFINITY;
            for (int e = 0; e < E; ++e) {
                if (!selected[e] && lm[e] > best_val) { best_val = lm[e]; best = e; }
            }
            selected[best] = true;
            im[sel] = best; wm[sel] = best_val;
        }
        // softmax over k selected
        float mx = -INFINITY;
        for (int i = 0; i < k; ++i) mx = fmaxf(mx, wm[i]);
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) { wm[i] = expf(wm[i] - mx); sum += wm[i]; }
        for (int i = 0; i < k; ++i) wm[i] /= sum;
    }
}

extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices,
                      int M, int E, int k) {
    moe_gating_kernel<<<M, 32>>>(logits, topk_weights, topk_indices, M, E, k);
}
```

> ⚠️ 朴素版用单 thread 做 top-k 选择（`k` 轮扫描 `E`），生产级用并行 bitonic sort + warp shuffle。本题 `k=2` 很小，朴素版够用。

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **top-k 选择** | `k` 轮选最大 | 降序排列选中 |
| **softmax** | 对 k 个选中值 | 重归一化权重 |
| **写回** | weights + indices | 按降序 |

> 💡 **关键洞察**：MoE gating = "per-token top-k 选择 + softmax 路由"。每 token 选 `k` 个 expert（如 `k=2`），softmax 归一化后作为加权路由权重。这是 Mixtral/DeepSeek-MoE 的核心——稀疏激活只计算选中的 expert，省算力。

## 5-6. 性能与复杂度

`O(M·k·E)`（`k` 轮选择），`k` 小时 dominated by sort。优化：bitonic sort、warp 并行选择。

> 💡 **一句话总结**：MoE Top-K Gating = "per-token top-k 选择 + softmax 路由"，MoE 稀疏激活的核心。

## 面试考点

- **手撕要求**：讲清 top-k 选择 + softmax 流程。
- **高频追问**：top-k 怎么并行化（bitonic sort / warp shuffle）；为什么 softmax（归一化路由权重）；k=2 的意义（每 token 激活 2 个 expert）；load balancing（防 expert 不均）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 29 | [Top K Selection](https://leetgpu.com/challenges/top-k-selection) | 中等 | bitonic 排序 + 堆归约基础 |
| 60 | [Top-p Sampling](https://leetgpu.com/challenges/top-p-sampling) | 中等 | 排序 + 累积概率 + 采样 |
| 5 | [Softmax](https://leetgpu.com/challenges/softmax) | 中等 | softmax，top-k 后的归一化 |
| 84 | [SwiGLU MLP Block](https://leetgpu.com/challenges/swiglu-mlp-block) | 中等 | MoE 中的 MLP 组件 |

> 💡 **选题思路**：top-k 选择 + softmax，练习排序归约与 MoE 路由。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
