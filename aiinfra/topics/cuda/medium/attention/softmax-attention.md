# LeetGPU Softmax Attention 题解

> **面试考察度**：⭐⭐⭐⭐⭐ Softmax Attention 是 FlashAttention 的入门版，推理岗越来越常考，至少能手写 fused softmax+matmul + online softmax 递推
> **面试形式**：手写 attention score 计算 + 讲清"为什么 scale、为什么 fused"

## 1. 题目概述

- **标题 / 题号**：Softmax Attention（LeetGPU #6，medium）
- **链接**：https://leetgpu.com/challenges/softmax-attention
- **难度**：中等
- **标签**：CUDA、Attention、Softmax、fused matmul、online softmax、数值稳定、memory-bound + compute-bound 混合

**题意**：给定 `Q`（`M×d`）、`K`（`N×d`）、`V`（`N×d`），计算 scaled dot-product attention（**无 causal mask**）：

$$\text{output} = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V$$

函数签名固定：

```cpp
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);
```

**关键约束**：

- 全 FP32，`scale = sqrt(d)`
- 容差 `atol=1e-4, rtol=1e-4`
- 性能测例 `M=512, N=256, d=128`（`input` 在 `[-0.1, 0.1]`）
- 无 mask（全注意力，非因果）

> ⚠️ **关键点**：① **scale** `1/√d` 防止点积过大导致 softmax 饱和；② **safe softmax** 减行最大值防 `exp` 溢出；③ **fused** 把 Q·Kᵀ/softmax/·V 融进单 kernel，避免 `scores (M×N)` 落盘 HBM。

## 2. CPU 基线 / 朴素 GPU 方法

```cpp
// CPU 串行（三步分离）
// scores = Q @ K^T / sqrt(d)   (M×N)
// attn = softmax(scores, dim=1)  逐行
// output = attn @ V            (M×d)
```

朴素 GPU：三个独立 kernel（GEMM → softmax → GEMM），`scores (M×N)` 落盘两次——`seq` 大时 HBM 带宽浪费。

## 3. GPU 设计

### 3.1 Fused 单 kernel

**核心映射**：一个 block 负责一行 query（`M` 个 block）。block 内对每行 `i`：算 `scores[i][0..N-1]` → safe softmax → 累加 `Σ attn[i][j]·V[j]`。

![Scaled Dot-Product Attention 数据流](../../../images/cuda_softmax_attention_overview.svg)

### 3.2 Online softmax（FlashAttention 思想）

朴素版需先算完整行 `scores` 再 softmax（两遍）。**Online softmax** 分块处理 K/V，每块增量更新 `(m, s)`：

$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}}), \quad s_{\text{new}} = s_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + s_{\text{block}} \cdot e^{m_{\text{block}} - m_{\text{new}}}$$

输出 `out[i] = Σ_j attn[i][j]·V[j]` 也分块累加，最后归一化 `out / s`。这样 `scores` 不落盘，只需 `O(N)` 寄存器遍历。

> 💡 **为什么 scale？** `Q·K` 的方差随 `d` 线性增长，不除 `√d` 会让点积量级大 → softmax 趋向 one-hot → 梯度消失。`scale=1/√d` 让方差稳定在 1。

## 4. Kernel 实现

```cuda
// cuda_softmax_attention.cu —— Fused Softmax Attention（朴素版，scores 留 shared）
// 编译: nvcc -O3 -arch=sm_120 cuda_softmax_attention.cu -o attn -lineinfo
// 运行: ./attn 512 256 128

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

// ---- 一个 block 负责一行 query，朴素 fused（scores 存 shared，适合 N 较小）----
__global__ void softmax_attention_kernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         float* __restrict__ output,
                                         int M, int N, int d) {
    __shared__ float scores[256];   // 假设 N <= 256（性能测例 N=256）
    int i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= M) return;

    const float* Qi = Q + i * d;
    float scale = 1.0f / sqrtf((float)d);

    // ---- ① scores[i][j] = Σ Q[i][k]·K[j][k] / √d ----
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        const float* Kj = K + j * d;
        float s = 0.0f;
        for (int k = 0; k < d; ++k) s += Qi[k] * Kj[k];
        s *= scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    // block reduce max（简化：用 shared + sync，实际用 warp shuffle）
    __shared__ float s_max;
    if (tid == 0) s_max = -INFINITY;
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(local_max));   // 简化：N 小时可用 atomic
    __syncthreads();
    float row_max = s_max;

    // ---- ② safe softmax: exp(scores - max), sum ----
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float inv_sum = 1.0f / s_sum;
    for (int j = tid; j < N; j += BLOCK_SIZE) scores[j] *= inv_sum;
    __syncthreads();

    // ---- ③ output[i] = Σ attn[i][j] · V[j] ----
    for (int k = tid; k < d; k += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += scores[j] * V[j * d + k];
        output[i * d + k] = acc;
    }
}

int main(int argc, char** argv) {
    int M = (argc>1)?atoi(argv[1]):512, N = (argc>2)?atoi(argv[2]):256, d = (argc>3)?atoi(argv[3]):128;
    size_t bQ=(size_t)M*d*4, bK=(size_t)N*d*4, bV=bK, bO=bQ;
    float *hQ=(float*)malloc(bQ),*hK=(float*)malloc(bK),*hV=(float*)malloc(bV),*hO=(float*)malloc(bO);
    srand(42); for(int i=0;i<M*d;++i) hQ[i]=((float)(rand()%20)-10)/100.0f;
    for(int i=0;i<N*d;++i){hK[i]=((float)(rand()%20)-10)/100.0f; hV[i]=((float)(rand()%20)-10)/100.0f;}

    float *dQ,*dK,*dV,*dO;
    CHECK_CUDA(cudaMalloc(&dQ,bQ)); CHECK_CUDA(cudaMalloc(&dK,bK)); CHECK_CUDA(cudaMalloc(&dV,bV)); CHECK_CUDA(cudaMalloc(&dO,bO));
    CHECK_CUDA(cudaMemcpy(dQ,hQ,bQ,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK,hK,bK,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV,hV,bV,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    softmax_attention_kernel<<<M, BLOCK_SIZE>>>(dQ,dK,dV,dO,M,N,d);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("M=%d N=%d d=%d time=%.3fms\n", M,N,d,ms);

    CHECK_CUDA(cudaMemcpy(hO,dO,bO,cudaMemcpyDeviceToHost));
    printf("done (sample output[0]=%.4f)\n", hO[0]);
    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK)); CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    free(hQ); free(hK); free(hV); free(hO);
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
// softmax_attention_submit.cu
#include <cuda_runtime.h>
#include <cmath>
#define BLOCK_SIZE 256

__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V,
                                         float* output, int M, int N, int d) {
    __shared__ float scores[256];
    __shared__ float s_max, s_sum;
    int i = blockIdx.x, tid = threadIdx.x;
    if (i >= M) return;
    const float* Qi = Q + i * d;
    float scale = 1.0f / sqrtf((float)d);

    float local_max = -INFINITY;
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        const float* Kj = K + j * d;
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
    for (int j = tid; j < N; j += BLOCK_SIZE) {
        float e = expf(scores[j] - row_max); scores[j] = e; local_sum += e;
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float inv_sum = 1.0f / s_sum;
    for (int j = tid; j < N; j += BLOCK_SIZE) scores[j] *= inv_sum;
    __syncthreads();

    for (int k = tid; k < d; k += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += scores[j] * V[j * d + k];
        output[i * d + k] = acc;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    softmax_attention_kernel<<<M, BLOCK_SIZE>>>(Q, K, V, output, M, N, d);
}
```

> ⚠️ 上面朴素版 `scores[256]` 假设 `N<=256`。大 `N` 需用 online softmax 分块（见 §3.2），scores 不落 shared 而是流式处理。

### 4.2 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **行映射** | `i = blockIdx.x` | 一个 block 一行 query |
| **算 scores** | `s = Σ Q[i][k]·K[j][k] · scale` | 归约 d 维 |
| **safe softmax** | `exp(s - max) / sum` | 减 max 防 overflow |
| **加权 V** | `acc += attn[j]·V[j][k]` | 归约 N 维 |
| **scale** | `1/√d` | 防点积过大 |

> 💡 **关键洞察**：attention 的本质是"两次归约"——d 维归约算 score，N 维归约加权 V。fused 版把 scores 留 shared，避免落盘 HBM。online softmax 进一步把 N 维也分块流式处理，是 FlashAttention 的核心。

## 5-6. 性能与复杂度

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M·N·d)`（Q·Kᵀ）+ `O(M·N·d)`（·V）= `O(2·M·N·d)` |
| **空间复杂度** | `O(M·d + N·d)` Q/K/V + `O(N)` shared scores |
| **瓶颈类型** | d 小时 memory-bound，d 大时 compute-bound；fused 减 HBM IO |
| **HBM IO（朴素分离）** | scores 落盘 `M·N·4B`（seq 大时浪费） |
| **HBM IO（fused）** | scores 留 shared，省落盘 |

> 💡 **一句话总结**：Softmax Attention = "Q·Kᵀ/√d → safe softmax → ·V"，fused 进单 kernel 避免 scores 落盘。online softmax 分块是 FlashAttention 的核心。掌握它就掌握了 attention 全家族（#12 MHA、#53 causal、#80 GQA）的骨架。

## 面试考点

- **手撕要求**：默写 fused attention（算 score → safe softmax → 加权 V）+ 讲清 scale 和 fused 的动机。
- **高频追问**：
  - **为什么 scale 1/√d？** 防点积过大导致 softmax 饱和、梯度消失。
  - **为什么 fused？** scores (M×N) 落盘浪费 HBM 带宽，fused 留 shared 省 IO。
  - **online softmax 怎么做？** 分块增量更新 (m, s)，旧和乘修正因子 e^(m_old-m_new)，scores 不落盘。
  - **safe softmax 为什么要减 max？** 防 exp 溢出（见 softmax 题解）。
- **进阶延伸**：FlashAttention v2/v3（分块 + 寄存器复用）、causal mask（#53）、MHA（#12 head 并行）、GQA（#80 KV 共享）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 12 | [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention) | 困难 | head 并行进阶 |
| 53 | [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention) | 困难 | 因果掩码，下三角掩码 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | attention 的基础组件 |
| 5 | [Softmax](https://leetgpu.com/challenges/softmax) | 中等 | attention 的基础组件 |

> 💡 **选题思路**：fused softmax+matmul + 数值稳定，练习 attention score 计算全流程。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
