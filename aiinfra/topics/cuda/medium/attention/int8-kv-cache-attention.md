# LeetGPU INT8 KV-Cache Attention 题解

> **面试考察度**：⭐⭐⭐⭐ KV cache 量化是 LLM 推理显存优化的核心，面试常考"量化与 attention 的结合"
> **面试形式**：手写 + 讲清"per-(head,seq) scale + 反量化后做标准 attention"

## 1. 题目概述

- **标题 / 题号**：INT8 KV-Cache Attention（LeetGPU #96，medium）
- **链接**：https://leetgpu.com/challenges/int8-kv-cache-attention
- **难度**：中等
- **标签**：CUDA、Attention、INT8、KV-Cache、量化、反量化、memory-bound

**题意**：per-head 单 query attention，K/V 为 INT8 量化（per-(head,seq) scale）。反量化 `K_fp = K_int8 * k_scale[head, seq]`，`V_fp = V_int8 * v_scale[head, seq]`，然后标准 `scores = Q·K_fp^T / √d → softmax → ·V_fp`。`Q` 为 FP32 `(num_heads, head_dim)`，`K/V` 为 INT8 `(num_heads, seq_len, head_dim)`，scales FP32 `(num_heads, seq_len)`，output FP32 `(num_heads, head_dim)`。无 causal mask（全注意力）。签名 `solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8, const float* k_scale, const float* v_scale, float* output, int num_heads, int seq_len, int head_dim)`。容差 `1e-3`，性能测例 `num_heads=32, seq_len=8192, head_dim=128`。

## 2-4. 设计与实现

一个 block 一个 head。block 内：反量化 K/V（逐元素 `int8 * scale`），算 `Q·K^T / √d`（归约 head_dim），safe softmax（归约 seq_len），`·V`（归约 seq_len）。

![Scaled Dot-Product Attention 数据流](../../../images/cuda_softmax_attention_overview.svg)

```cuda
// int8_kv_cache_attention_submit.cu —— 朴素版（一个 block 一个 head）
#include <cuda_runtime.h>
#include <cmath>

__global__ void int8_kv_attn_kernel(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                                    const float* k_scale, const float* v_scale,
                                    float* output, int num_heads, int seq_len, int head_dim) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= num_heads) return;

    extern __shared__ float scores[];   // seq_len 个
    __shared__ float s_max, s_sum;

    const float* Qh = Q + h * head_dim;
    const int8_t* Kh = K_int8 + (size_t)h * seq_len * head_dim;
    const int8_t* Vh = V_int8 + (size_t)h * seq_len * head_dim;
    const float* ksh = k_scale + h * seq_len;
    const float* vsh = v_scale + h * seq_len;
    float scale = 1.0f / sqrtf((float)head_dim);

    // ① scores[j] = Σ Q[k] · (K_int8[j][k] * k_scale[j]) / √d
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < head_dim; ++k)
            s += Qh[k] * ((float)Kh[j * head_dim + k] * ksh[j]);
        scores[j] = s * scale;
    }
    __syncthreads();

    // ② safe softmax
    float local_max = -INFINITY;
    for (int j = tid; j < seq_len; j += blockDim.x) local_max = fmaxf(local_max, scores[j]);
    if (tid == 0) s_max = -INFINITY;
    __syncthreads(); atomicMax((int*)&s_max, __float_as_int(local_max)); __syncthreads();
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float e = expf(scores[j] - s_max); scores[j] = e; local_sum += e;
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads(); atomicAdd(&s_sum, local_sum); __syncthreads();
    float inv = 1.0f / s_sum;
    for (int j = tid; j < seq_len; j += blockDim.x) scores[j] *= inv;
    __syncthreads();

    // ③ output[k] = Σ attn[j] · (V_int8[j][k] * v_scale[j])
    for (int k = tid; k < head_dim; k += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j)
            acc += scores[j] * ((float)Vh[j * head_dim + k] * vsh[j]);
        output[h * head_dim + k] = acc;
    }
}

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output,
                      int num_heads, int seq_len, int head_dim) {
    int8_kv_attn_kernel<<<num_heads, 256, seq_len * sizeof(float)>>>(
        Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **反量化 K** | `Kh[j][k] * ksh[j]` | INT8 × per-(head,seq) scale |
| **score** | `Σ Q·K_fp / √d` | 标准 attention |
| **softmax** | safe softmax | 减 max 防 overflow |
| **反量化 V** | `Vh[j][k] * vsh[j]` | 同 K |
| **加权** | `Σ attn·V_fp` | 标准加权 |

> 💡 **关键洞察**：INT8 KV-cache attention = "标准 attention + K/V 反量化"。KV cache 是推理显存瓶颈（随 seq_len 线性增长），INT8 量化省 4× 显存与带宽。反量化可在加载时即时做（不落盘 FP32），保持 cache 的 INT8 存储。

## 5-6. 性能与复杂度

`O(num_heads · seq_len · head_dim)`，KV 字节减 4×，memory-bound（attention 本就 memory-bound，量化锦上添花）。

> 💡 **一句话总结**：INT8 KV-Cache Attention = "attention + K/V 反量化"，KV cache 量化省推理显存。

## 面试考点

- **手撕要求**：默写反量化 + 标准 attention 流程。
- **高频追问**：per-(head,seq) scale 粒度；为什么量化 KV（显存瓶颈）；反量化在哪做（加载时即时，不落盘）；精度损失。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 80 | [Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention) | 中等 | KV head 复用的 attention 基础 |
| 64 | [Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization) | 中等 | 反量化基础 |
| 53 | [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention) | 困难 | attention 基础 |
| 32 | [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul) | 中等 | INT8 计算基础 |

> 💡 **选题思路**：量化 KV cache + attention，练习低精度推理与 attention 的结合。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
