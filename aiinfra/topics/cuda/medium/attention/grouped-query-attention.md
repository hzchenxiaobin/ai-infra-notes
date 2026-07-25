# LeetGPU Grouped Query Attention 题解

> **面试考察度**：⭐⭐⭐⭐ GQA 是 LLaMA-2/3 的标准 attention，面试常考"KV head 怎么共享、和 MHA 的区别"
> **面试形式**：手写 GQA kernel + 讲清"num_groups 个 Q head 共享 1 个 KV head"的映射

## 1. 题目概述

- **标题 / 题号**：Grouped Query Attention（LeetGPU #80，medium）
- **链接**：https://leetgpu.com/challenges/grouped-query-attention
- **难度**：中等
- **标签**：CUDA、GQA、KV head 共享、batched attention、attention、memory-bound

**题意**：给定 `Q`（`num_q_heads × seq_len × head_dim`）、`K`/`V`（`num_kv_heads × seq_len × head_dim`），`num_q_heads % num_kv_heads == 0`。每组 `num_groups = num_q_heads / num_kv_heads` 个 Q head 共享 1 个 KV head（按连续分组）：

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i \cdot K_{g}^T}{\sqrt{d}}\right) \cdot V_{g}, \qquad g = \lfloor i / \text{num\_groups} \rfloor$$

函数签名固定：

```cpp
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim);
```

**关键约束**：

- 全 FP32，`scale = 1/sqrt(head_dim)`，3D 布局（head 维在前）
- 容差 `atol=1e-4, rtol=1e-4`
- 性能测例 `num_q_heads=32, num_kv_heads=8, seq_len=1024, head_dim=128`（LLaMA-3 8B 风格）
- KV 共享：`num_groups = 32/8 = 4`，每 4 个 Q head 共享 1 个 KV head

> ⚠️ **KV head 映射**：Q head `i` 对应的 KV head 是 `g = i / num_groups`（整数除法，连续分组）。写反成 `i % num_kv_heads` 会全错。

> 💡 **为什么 GQA？** KV cache 是 LLM 推理的显存瓶颈。MHA（`num_kv_heads = num_q_heads`）的 KV cache 随 head 数线性增长；GQA/MQA 减少KV head 数，显存和带宽都降，精度损失小——LLaMA-2/3 默认用 GQA。

## 2-3. CPU 基线与 GPU 设计

CPU 串行：`K`/`V` 用 `repeat_interleave(num_groups, dim=0)` 扩展到 `num_q_heads` 个，再对每个 Q head 独立做 attention（同 #6）。

**GPU 映射**：`grid = num_q_heads` 个 block，一个 block 一个 Q head。block 内对 head `i`：取 `Q[i]`、`K[g]`、`V[g]`（`g = i / num_groups`）做单 head attention。

![Scaled Dot-Product Attention 数据流](../../../images/cuda_softmax_attention_overview.svg)

> 💡 **与 #12 MHA 的区别**：MHA 的 Q/K/V head 数相等（`h`）；GQA 的 Q head 多、KV head 少，多个 Q head 共享同一 KV head——只是 KV 的索引从 `i` 变 `i / num_groups`，骨架同构。

## 4. Kernel 实现

```cuda
// cuda_grouped_query_attention.cu —— GQA：一个 block 一个 Q head，KV 按 group 共享
// 编译: nvcc -O3 -arch=sm_120 cuda_grouped_query_attention.cu -o gqa -lineinfo
// 运行: ./gqa 32 8 1024 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256

// ---- 一个 block 一个 Q head，KV 按 g = q_head / num_groups 共享 ----
__global__ void gqa_kernel(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           const float* __restrict__ V,
                           float* __restrict__ output,
                           int num_q_heads, int num_kv_heads,
                           int seq_len, int head_dim) {
    extern __shared__ float scores[];   // seq_len 个
    __shared__ float s_max, s_sum;
    int qh = blockIdx.x;                 // Q head 编号
    int tid = threadIdx.x;
    int num_groups = num_q_heads / num_kv_heads;
    int kvh = qh / num_groups;           // ← 共享的 KV head 编号
    int S = seq_len, d = head_dim;

    const float* Qh = Q + (size_t)qh * S * d;
    const float* Kh = K + (size_t)kvh * S * d;   // 多个 Q head 共享
    const float* Vh = V + (size_t)kvh * S * d;
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < S; ++i) {        // 逐 query 行（朴素版串行行）
        const float* Qi = Qh + i * d;
        float local_max = -INFINITY;
        for (int j = tid; j < S; j += BLOCK_SIZE) {
            const float* Kj = Kh + j * d;
            float s = 0.0f;
            for (int k = 0; k < d; ++k) s += Qi[k] * Kj[k];
            s *= scale; scores[j] = s; local_max = fmaxf(local_max, s);
        }
        if (tid == 0) s_max = -INFINITY;
        __syncthreads();
        atomicMax((int*)&s_max, __float_as_int(local_max));
        __syncthreads();
        float row_max = s_max;

        float local_sum = 0.0f;
        for (int j = tid; j < S; j += BLOCK_SIZE) {
            float e = expf(scores[j] - row_max); scores[j] = e; local_sum += e;
        }
        if (tid == 0) s_sum = 0.0f;
        __syncthreads();
        atomicAdd(&s_sum, local_sum);
        __syncthreads();
        float inv_sum = 1.0f / s_sum;
        for (int j = tid; j < S; j += BLOCK_SIZE) scores[j] *= inv_sum;
        __syncthreads();

        for (int k = tid; k < d; k += BLOCK_SIZE) {
            float acc = 0.0f;
            for (int j = 0; j < S; ++j) acc += scores[j] * Vh[j * d + k];
            output[((size_t)qh * S + i) * d + k] = acc;
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    int nqh=(argc>1)?atoi(argv[1]):32, nkvh=(argc>2)?atoi(argv[2]):8, S=(argc>3)?atoi(argv[3]):1024, d=(argc>4)?atoi(argv[4]):128;
    size_t bQ=(size_t)nqh*S*d*4, bKV=(size_t)nkvh*S*d*4, bO=bQ;
    float *hQ=(float*)malloc(bQ),*hK=(float*)malloc(bKV),*hV=(float*)malloc(bKV),*hO=(float*)malloc(bO);
    srand(42); for(int i=0;i<nqh*S*d;++i) hQ[i]=((float)(rand()%200)-100)/100.0f;
    for(int i=0;i<nkvh*S*d;++i){hK[i]=((float)(rand()%200)-100)/100.0f; hV[i]=hK[i];}
    float *dQ,*dK,*dV,*dO;
    CHECK_CUDA(cudaMalloc(&dQ,bQ)); CHECK_CUDA(cudaMalloc(&dK,bKV)); CHECK_CUDA(cudaMalloc(&dV,bKV)); CHECK_CUDA(cudaMalloc(&dO,bO));
    CHECK_CUDA(cudaMemcpy(dQ,hQ,bQ,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK,hK,bKV,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV,hV,bKV,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gqa_kernel<<<nqh, BLOCK_SIZE, S*sizeof(float)>>>(dQ,dK,dV,dO,nqh,nkvh,S,d);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("q_heads=%d kv_heads=%d seq=%d d=%d time=%.3fms\n", nqh,nkvh,S,d,ms);

    CHECK_CUDA(cudaMemcpy(hO,dO,bO,cudaMemcpyDeviceToHost));
    printf("output[0]=%.4f\n", hO[0]);
    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK)); CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    free(hQ); free(hK); free(hV); free(hO);
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
// grouped_query_attention_submit.cu
#include <cuda_runtime.h>
#include <cmath>
#define BLOCK_SIZE 256

__global__ void gqa_kernel(const float* Q, const float* K, const float* V,
                           float* output, int num_q_heads, int num_kv_heads,
                           int seq_len, int head_dim) {
    extern __shared__ float scores[];
    __shared__ float s_max, s_sum;
    int qh = blockIdx.x, tid = threadIdx.x;
    int num_groups = num_q_heads / num_kv_heads;
    int kvh = qh / num_groups;
    int S = seq_len, d = head_dim;
    const float* Qh = Q + (size_t)qh * S * d;
    const float* Kh = K + (size_t)kvh * S * d;
    const float* Vh = V + (size_t)kvh * S * d;
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < S; ++i) {
        const float* Qi = Qh + i * d;
        float local_max = -INFINITY;
        for (int j = tid; j < S; j += BLOCK_SIZE) {
            const float* Kj = Kh + j * d;
            float s = 0.0f;
            for (int k = 0; k < d; ++k) s += Qi[k] * Kj[k];
            s *= scale; scores[j] = s; local_max = fmaxf(local_max, s);
        }
        if (tid == 0) s_max = -INFINITY;
        __syncthreads();
        atomicMax((int*)&s_max, __float_as_int(local_max));
        __syncthreads();
        float row_max = s_max;
        float local_sum = 0.0f;
        for (int j = tid; j < S; j += BLOCK_SIZE) {
            float e = expf(scores[j] - row_max); scores[j] = e; local_sum += e;
        }
        if (tid == 0) s_sum = 0.0f;
        __syncthreads();
        atomicAdd(&s_sum, local_sum);
        __syncthreads();
        float inv_sum = 1.0f / s_sum;
        for (int j = tid; j < S; j += BLOCK_SIZE) scores[j] *= inv_sum;
        __syncthreads();
        for (int k = tid; k < d; k += BLOCK_SIZE) {
            float acc = 0.0f;
            for (int j = 0; j < S; ++j) acc += scores[j] * Vh[j * d + k];
            output[((size_t)qh * S + i) * d + k] = acc;
        }
        __syncthreads();
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    gqa_kernel<<<num_q_heads, BLOCK_SIZE, seq_len * sizeof(float)>>>(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
}
```

### 4.2 代码详解

与 #12 MHA **逐行对称**，唯一差别：KV head 索引从 `qh`（每 Q head 独立 KV）变为 `kvh = qh / num_groups`（组内共享）。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **Q head 映射** | `qh = blockIdx.x` | 一个 block 一个 Q head |
| **KV head 共享** | `kvh = qh / num_groups` | 组内 Q head 共享同一 KV head |
| **3D 布局** | `Q + qh*S*d` | head 维在前（与 #12 的 2D 不同） |
| **attention** | 同 #6 | scale 用 head_dim |

> 💡 **关键洞察**：GQA = MHA + "KV head 共享"。代码唯一改动是 KV 索引 `qh → qh/num_groups`——`num_groups` 个 Q head 共享同一 KV head。这大幅减少 KV cache 显存（LLM 推理瓶颈），是 LLaMA-2/3 的标准配置。

## 5-6. 性能与复杂度

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(num_q_heads · S² · d)` |
| **KV cache 节省** | `num_kv_heads / num_q_heads = 1/num_groups`（本题省 4×） |
| **瓶颈类型** | 同 #6/#12 |

> 💡 **一句话总结**：GQA = "MHA + KV head 共享"，`kvh = qh / num_groups` 是唯一改动。它减少 KV cache 显存与带宽，是 LLaMA-2/3 推理优化的标准配置。MQA（`num_kv_heads=1`）是 GQA 的极致特例。

## 面试考点

- **手撕要求**：默写 GQA（`kvh = qh / num_groups`）+ 讲清 KV 共享的动机。
- **高频追问**：
  - **GQA 和 MHA 的区别？** MHA 每 Q head 独立 KV head；GQA 多个 Q head 共享 KV head（`num_groups` 个共享 1 个）。
  - **为什么 GQA？** KV cache 是推理显存瓶颈，减少 KV head 数省显存和带宽，精度损失小。
  - **KV head 怎么映射？** `kvh = qh / num_groups`（整数除法，连续分组），不是取模。
  - **GQA vs MQA？** MQA 是 `num_kv_heads=1`（所有 Q head 共享 1 个 KV），GQA 是折中（分组共享）。
- **进阶延伸**：LLaMA-2/3 用 GQA，DeepSeek 用 MLA（更激进的 KV 压缩）；KV cache + paged attention（vLLM）是推理引擎标配。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 12 | [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention) | 困难 | MHA 基础版 |
| 53 | [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention) | 困难 | mask 变体 |
| 96 | [INT8 KV-Cache Attention](https://leetgpu.com/challenges/int8-kv-cache-attention) | 困难 | 量化 + KV cache |
| 59 | [Sliding Window Self-Attention](https://leetgpu.com/challenges/sliding-window-self-attention) | 困难 | 滑窗注意力变体 |

> 💡 **选题思路**：KV head 共享 + attention，练习 GQA 的分组调度。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
