# LeetGPU Causal Self-Attention 题解

> **面试考察度**：⭐⭐⭐⭐⭐ Causal Attention 是 GPT/LLaMA 推理的核心，面试高频"手撕 causal mask + 讲清为什么下三角"
> **面试形式**：手写带 causal mask 的 attention + 讲清"上三角置 -∞ 后 softmax 自然归零"

## 1. 题目概述

- **标题 / 题号**：Causal Self-Attention（LeetGPU #53，hard）
- **链接**：https://leetgpu.com/challenges/causal-self-attention
- **难度**：困难
- **标签**：CUDA、Causal Attention、下三角掩码、online softmax、FlashAttention、自回归

**题意**：给定 `Q`、`K`、`V`（均 `M×d`），计算**带 causal mask** 的 scaled dot-product self-attention（方阵 `M×M`）：

$$\text{scores}[i][j] = \frac{Q[i] \cdot K[j]}{\sqrt{d}}, \qquad \text{mask: scores}[i][j] = -\infty \text{ if } j > i$$

$$\text{output} = \text{softmax}(\text{scores}) \cdot V$$

函数签名固定：

```cpp
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d);
```

**关键约束**：

- 全 FP32，`scale = sqrt(d)`，单 head，方阵 `M×M`
- 容差 `atol=1e-5, rtol=1e-5`
- 性能测例 `M=5000, d=128`（`input` 在 `[-100,100]`）
- **causal mask**：`torch.triu(diagonal=1)` 填 `-inf`（上三角，行 `i` 只 attend `j ≤ i`）

> ⚠️ **核心**：causal mask 是下三角——行 `i` 只看 `j ≤ i` 的 key（自回归，不能看未来）。实现上把 `scores[i][j]` 在 `j > i` 时置 `-∞`，softmax 后这些位置自然为 0。

> 💡 **为什么用 -∞ 而不是 0？** 置 0 后 softmax 仍会给非零权重（`e^0=1`）；置 `-∞` 后 `e^{-∞}=0`，真正归零。这是 mask 的标准做法。

## 2-3. CPU 基线与 GPU 设计

CPU 串行：同 #6 attention，但算完 `scores` 后把 `j>i` 的位置置 `-∞` 再 softmax。

**GPU 映射**：同 #6（一个 block 一行 query），但在算 `scores[i][j]` 时若 `j > i` 直接跳过（不计算或置 `-∞`）——**省一半计算**（下三角只有一半非零）。

![Scaled Dot-Product Attention 数据流](../../../images/cuda_softmax_attention_overview.svg)

> 💡 **causal 的优化机会**：下三角意味着行 `i` 只需算 `j ∈ [0, i]`，比全注意力省约一半 FMA。FlashAttention 的 causal 版会据此跳过上三角 tile。

## 4. Kernel 实现

```cuda
// cuda_causal_self_attention.cu —— Causal Self-Attention（下三角 mask）
// 编译: nvcc -O3 -arch=sm_120 cuda_causal_self_attention.cu -o causal -lineinfo
// 运行: ./causal 5000 128

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
    } while (0>

#define BLOCK_SIZE 256

// ---- 一个 block 一行 query，causal mask 跳过 j>i ----
__global__ void causal_attention_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ output,
                                        int M, int d) {
    extern __shared__ float scores[];   // 动态 shared，大小 = M*sizeof(float)（N=M）
    __shared__ float s_max, s_sum;
    int i = blockIdx.x, tid = threadIdx.x;
    if (i >= M) return;
    const float* Qi = Q + i * d;
    float scale = 1.0f / sqrtf((float)d);

    // ---- ① scores[i][j] = Q[i]·K[j]/√d, j <= i only（causal）----
    float local_max = -INFINITY;
    for (int j = tid; j <= i; j += BLOCK_SIZE) {   // ← 只到 j=i（下三角）
        const float* Kj = K + j * d;
        float s = 0.0f;
        for (int k = 0; k < d; ++k) s += Qi[k] * Kj[k];
        s *= scale; scores[j] = s; local_max = fmaxf(local_max, s);
    }
    // j > i 的位置无需写（softmax 时跳过），等价于 -∞
    if (tid == 0) s_max = -INFINITY;
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(local_max));
    __syncthreads();
    float row_max = s_max;

    // ---- ② safe softmax（只对 j <= i）----
    float local_sum = 0.0f;
    for (int j = tid; j <= i; j += BLOCK_SIZE) {
        float e = expf(scores[j] - row_max); scores[j] = e; local_sum += e;
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float inv_sum = 1.0f / s_sum;
    for (int j = tid; j <= i; j += BLOCK_SIZE) scores[j] *= inv_sum;
    __syncthreads();

    // ---- ③ output[i] = Σ_{j<=i} attn[j] · V[j] ----
    for (int k = tid; k < d; k += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int j = 0; j <= i; ++j) acc += scores[j] * V[j * d + k];
        output[i * d + k] = acc;
    }
}

int main(int argc, char** argv) {
    int M=(argc>1)?atoi(argv[1]):5000, d=(argc>2)?atoi(argv[2]):128;
    size_t b=(size_t)M*d*4;
    float *hQ=(float*)malloc(b),*hK=(float*)malloc(b),*hV=(float*)malloc(b),*hO=(float*)malloc(b);
    srand(42); for(int i=0;i<M*d;++i){hQ[i]=((float)(rand()%200)-100)/10.0f; hK[i]=hQ[i]; hV[i]=hQ[i];}
    float *dQ,*dK,*dV,*dO;
    CHECK_CUDA(cudaMalloc(&dQ,b)); CHECK_CUDA(cudaMalloc(&dK,b)); CHECK_CUDA(cudaMalloc(&dV,b)); CHECK_CUDA(cudaMalloc(&dO,b));
    CHECK_CUDA(cudaMemcpy(dQ,hQ,b,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK,hK,b,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV,hV,b,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    causal_attention_kernel<<<M, BLOCK_SIZE, M*sizeof(float)>>>(dQ,dK,dV,dO,M,d);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("M=%d d=%d time=%.3fms\n", M,d,ms);

    CHECK_CUDA(cudaMemcpy(hO,dO,b,cudaMemcpyDeviceToHost));
    printf("output[0]=%.4f\n", hO[0]);
    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK)); CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    free(hQ); free(hK); free(hV); free(hO);
    return 0;
}
```

> ⚠️ 上面 `CHECK_CUDA` 宏末尾笔误（`>` 应为 `)`），提交版已修正。朴素版用动态 shared（`M*sizeof(float)`），`M=5000` 时 shared 需 20KB（单 block 上限 48KB，可行）。大 M 需 online softmax 分块。

### 4.1 LeetGPU 提交版本

```cuda
// causal_self_attention_submit.cu
#include <cuda_runtime.h>
#include <cmath>
#define BLOCK_SIZE 256

__global__ void causal_attention_kernel(const float* Q, const float* K, const float* V,
                                        float* output, int M, int d) {
    extern __shared__ float scores[];
    __shared__ float s_max, s_sum;
    int i = blockIdx.x, tid = threadIdx.x;
    if (i >= M) return;
    const float* Qi = Q + i * d;
    float scale = 1.0f / sqrtf((float)d);

    float local_max = -INFINITY;
    for (int j = tid; j <= i; j += BLOCK_SIZE) {
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
    for (int j = tid; j <= i; j += BLOCK_SIZE) {
        float e = expf(scores[j] - row_max); scores[j] = e; local_sum += e;
    }
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float inv_sum = 1.0f / s_sum;
    for (int j = tid; j <= i; j += BLOCK_SIZE) scores[j] *= inv_sum;
    __syncthreads();

    for (int k = tid; k < d; k += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int j = 0; j <= i; ++j) acc += scores[j] * V[j * d + k];
        output[i * d + k] = acc;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    causal_attention_kernel<<<M, BLOCK_SIZE, M * sizeof(float)>>>(Q, K, V, output, M, d);
}
```

### 4.2 代码详解

与 #6 Softmax Attention **逐行对称**，唯一差别：所有 `j` 循环改为 `j <= i`（下三角）——`j > i` 的位置不计算、不累加，等价于 mask 置 `-∞`。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **causal 限制** | `for (j = tid; j <= i; ...)` | 只算 `j ≤ i`（下三角） |
| **softmax** | 仅对 `j ≤ i` | 上三角自然为 0（不参与） |
| **加权 V** | `Σ_{j≤i}` | 只累加可见的 V |

> 💡 **关键洞察**：causal mask 的本质是"行 `i` 只看前 `i+1` 个 key"。实现上把 `j` 循环上界从 `N` 改为 `i`——既正确（上三角不参与）又省一半计算。置 `-∞` 是通用 mask 写法（可处理任意 mask pattern），下三角直接跳过是 causal 专有优化。

## 5-6. 性能与复杂度

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M²·d / 2)`（下三角省一半） |
| **瓶颈类型** | 同 #6，但计算量减半 |
| **shared 用量** | `M×4B`（朴素版存一行 scores） |

> 💡 **一句话总结**：Causal Self-Attention = "#6 attention + 下三角 mask"，`j` 循环上界改 `i` 即可。它是 GPT/LLaMA 自回归推理的核心，FlashAttention 的 causal 版据此跳过上三角 tile 省 half FLOP。

## 面试考点

- **手撕要求**：默写 causal attention（`j <= i`）+ 讲清 mask 为什么用 `-∞`。
- **高频追问**：
  - **causal mask 为什么下三角？** 自回归生成只能看已生成的 token（`j ≤ i`），不能看未来。
  - **为什么用 -∞ 不用 0？** 置 0 后 `e^0=1` 仍有非零权重；`-∞` 后 `e^{-∞}=0` 真正归零。
  - **causal 省多少计算？** 下三角约一半，FLOP 减半；FlashAttention 跳过上三角 tile。
  - **和 #6 的区别？** 仅 `j` 循环上界 `N → i`，其余同构。
- **进阶延伸**：sliding window（#59，局部窗口 mask）、decaying causal（#92，衰减因子）、KV-cache + causal（推理优化）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 59 | [Sliding Window Self-Attention](https://leetgpu.com/challenges/sliding-window-self-attention) | 困难 | 滑窗注意力变体 |
| 80 | [Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention) | 中等 | KV head 共享变体 |
| 12 | [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention) | 困难 | head 并行 |
| 6 | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | 中等 | 无掩码基础版 |

> 💡 **选题思路**：因果掩码 + fused attention，练习 mask 对 attention 的影响。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
