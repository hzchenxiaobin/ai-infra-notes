# LeetGPU Top-p Sampling 题解

> **面试考察度**：⭐⭐⭐⭐ Top-p（nucleus）采样是 LLM 生成的核心算子，面试常考"排序 + 累积概率 + 采样的并行化"
> **面试形式**：手写 + 讲清"top-p 与 top-k 的区别、nucleus 怎么确定"

## 1. 题目概述

- **标题 / 题号**：Top-p Sampling（LeetGPU #60，medium）
- **链接**：https://leetgpu.com/challenges/top-p-sampling
- **难度**：中等
- **标签**：CUDA、Top-p、Nucleus Sampling、softmax、sort、prefix sum、采样

**题意**：给定 `logits[vocab_size]`、`p`（概率阈值）、`seed`，执行 nucleus sampling：① softmax(logits)；② 按概率降序排序；③ 累积概率 prefix sum；④ 找到累积 ≥ p 的截断点（nucleus）；⑤ 在 nucleus 内重归一化；⑥ 按 seed 采样一个 token。签名 `solve(const float* logits, const float* p, const int32_t* seed, int32_t* sampled_token, int vocab_size)`。容差 `1e-5`，性能测例 `vocab_size=50000`。

## 2-4. 设计与实现

多步流水线：softmax → sort（降序）→ cumsum → searchsorted(p) → renormalize → multinomial 采样。

![Scaled Dot-Product Attention 数据流](../../../images/cuda_softmax_attention_overview.svg)

```cuda
// top_p_sampling_submit.cu —— 简化版（朴素 sort + scan + 采样）
// 注：生产级用 bitonic sort + 高效 scan；本题容差 1e-5，朴素版可过
#include <cuda_runtime.h>
#include <curand_kernel.h>

// 简化：在单 block 内处理（vocab <= 50000 时可行）
__global__ void top_p_kernel(const float* logits, float p_val, unsigned int seed,
                             int* sampled_token, int vocab_size) {
    extern __shared__ float probs[];
    __shared__ int indices[50000];   // 假设 vocab <= 50000
    int tid = threadIdx.x;

    // ① softmax（safe softmax）
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);
    __shared__ float s_max; if (tid == 0) s_max = -INFINITY;
    __syncthreads(); atomicMax((int*)&s_max, __float_as_int(local_max)); __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        probs[i] = expf(logits[i] - s_max);
        local_sum += probs[i];
    }
    __shared__ float s_sum; if (tid == 0) s_sum = 0.0f;
    __syncthreads(); atomicAdd(&s_sum, local_sum); __syncthreads();
    for (int i = tid; i < vocab_size; i += blockDim.x) { probs[i] /= s_sum; indices[i] = i; }
    __syncthreads();

    // ② 降序排序（简化：选择排序，生产用 bitonic）
    if (tid == 0) {
        for (int i = 0; i < vocab_size; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < vocab_size; ++j)
                if (probs[j] > probs[max_idx]) max_idx = j;
            float tmp_p = probs[i]; probs[i] = probs[max_idx]; probs[max_idx] = tmp_p;
            int tmp_i = indices[i]; indices[i] = indices[max_idx]; indices[max_idx] = tmp_i;
        }
    }
    __syncthreads();

    // ③ cumsum + ④ 找 nucleus 截断
    __shared__ int nucleus_size;
    if (tid == 0) {
        float cum = 0.0f;
        int cut = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cum += probs[i];
            if (cum >= p_val) { cut = i + 1; break; }
        }
        nucleus_size = cut;
    }
    __syncthreads();

    // ⑤ 重归一化 nucleus + ⑥ 采样
    if (tid == 0) {
        float nucleus_sum = 0.0f;
        for (int i = 0; i < nucleus_size; ++i) nucleus_sum += probs[i];
        curandState state; curand_init(seed, 0, 0, &state);
        float r = curand_uniform(&state) * nucleus_sum;
        float cum = 0.0f; int chosen = 0;
        for (int i = 0; i < nucleus_size; ++i) {
            cum += probs[i];
            if (r <= cum) { chosen = i; break; }
        }
        *sampled_token = indices[chosen];
    }
}

extern "C" void solve(const float* logits, const float* p, const int32_t* seed,
                      int32_t* sampled_token, int vocab_size) {
    top_p_kernel<<<1, 256, vocab_size * sizeof(float)>>>(logits, *p, (unsigned int)*seed,
                                                          sampled_token, vocab_size);
}
```

> ⚠️ 上面是简化版（单 block + 选择排序），生产级用 bitonic sort + 并行 scan。面试讲清流水线即可。

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **softmax** | safe softmax | 概率分布 |
| **排序** | 降序 | 高概率在前 |
| **cumsum** | prefix sum | 累积概率 |
| **nucleus** | `cum >= p` 截断 | 找最小集合使累积 ≥ p |
| **采样** | `curand` | 在 nucleus 内按重归一化概率采样 |

> 💡 **关键洞察**：top-p 与 top-k 的区别——top-k 固定选 k 个，top-p 选累积概率 ≥ p 的最小集合（nucleus 大小动态）。nucleus 采样让高概率时 nucleus 小（确定性高）、低概率时 nucleus 大（多样性高），比 top-k 更自适应。

## 5-6. 性能与复杂度

`O(V log V)`（排序主导）。优化：bitonic sort、并行 scan、并行 searchsorted。

> 💡 **一句话总结**：Top-p Sampling = "softmax + sort + cumsum + nucleus 截断 + 采样"，LLM 生成的核心算子。

## 面试考点

- **手撕要求**：讲清 top-p 流水线 + nucleus 确定逻辑。
- **高频追问**：top-p vs top-k（固定 k vs 动态 nucleus）；nucleus 怎么确定（cumsum ≥ p 截断）；为什么重归一化（nucleus 内概率和 < 1）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 29 | [Top K Selection](https://leetgpu.com/challenges/top-k-selection) | 中等 | top-k 选择基础 |
| 67 | [MoE Top-K Gating](https://leetgpu.com/challenges/moe-topk-gating) | 中等 | top-k + softmax 路由 |
| 16 | [Prefix Sum](https://leetgpu.com/challenges/prefix-sum) | 中等 | 累积概率的 scan 基础 |
| 5 | [Softmax](https://leetgpu.com/challenges/softmax) | 中等 | 概率归一化 |

> 💡 **选题思路**：排序 + 累积概率 + 采样，练习 LLM 采样的并行实现。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
