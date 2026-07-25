# LeetGPU INT8 Quantized MatMul 题解

> **面试考察度**：⭐⭐⭐⭐ INT8 GEMM 是推理优化的标配，面试常考"requantize 流程与 scale 链"
> **面试形式**：手写 + 讲清"INT8 累加 → scale 缩放 → requantize 回 INT8"

## 1. 题目概述

- **标题 / 题号**：INT8 Quantized MatMul（LeetGPU #32，medium）
- **链接**：https://leetgpu.com/challenges/int8-quantized-matmul
- **难度**：中等
- **标签**：CUDA、GEMM、INT8、量化、requantize、compute-bound

**题意**：INT8 矩阵乘 `C = A @ B`，带 per-tensor scale/zero_point。`A (M×K)`、`B (K×N)`、`C (M×N)` 均为 INT8。计算：反量化 `A_f = (A - zp_A) * scale_A`，`B_f = (B - zp_B) * scale_B`，矩阵乘 `C_f = A_f @ B_f`，requantize `C = round(C_f * scale_C / (scale_A * scale_B)) + zp_C`。签名 `solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C)`。容差 `0.05`，性能测例 `A 8192×2048, B 2048×4096`。

## 2-4. 设计与实现

tiled matmul（同 #2），但：① A/B 为 INT8，加载后转 FP32（`(A - zp) * scale`）；② 累加 FP32；③ 写回时 requantize `round(acc * scale_C / (scale_A*scale_B)) + zp_C` 转 INT8。

![量化与反量化](../../../images/cuda_quantization_overview.svg)

```cuda
// int8_quantized_matmul_submit.cu —— 朴素 INT8 GEMM + requantize
#include <cuda_runtime.h>
#include <cstdint>

#define BM 32
#define BK 32
#define BN 32
#define BLOCK_SIZE (BM * BK)

__global__ void int8_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C,
                                   int M, int N, int K,
                                   float scale_A, float scale_B, float scale_C,
                                   int zp_A, int zp_B, int zp_C) {
    __shared__ int8_t sA[BM][BN];
    __shared__ int8_t sB[BN][BK];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x / BK, ty = threadIdx.x % BK;
    int row = bx * BM + tx, col = by * BK + ty;
    float acc = 0.0f;

    float combined_scale = scale_A * scale_B / scale_C;
    for (int t = 0; t < (N + BN - 1) / BN; ++t) {
        sA[tx][ty] = (row < M && t * BN + ty < N) ? A[row * N + t * BN + ty] : 0;
        sB[tx][ty] = (t * BN + tx < N && col < K) ? B[(t * BN + tx) * K + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BN; ++k) {
            float a = (float)sA[tx][k] - zp_A;
            float b = (float)sB[k][ty] - zp_B;
            acc += a * b;
        }
        __syncthreads();
    }
    if (row < M && col < K) {
        int q = (int)roundf(acc * combined_scale) + zp_C;
        C[row * K + col] = (int8_t)max(-128, min(127, q));   // clamp 到 INT8
    }
}

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C) {
    dim3 grid((M + BM - 1) / BM, (K + BK - 1) / BK);
    int8_matmul_kernel<<<grid, BLOCK_SIZE>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C,
                                              zero_point_A, zero_point_B, zero_point_C);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **反量化** | `(A - zp_A) * scale_A` | INT8 → FP32（隐含在累加） |
| **累加** | `acc += a * b` | FP32 累加 |
| **requantize** | `round(acc * scale_C/(scale_A*scale_B)) + zp_C` | FP32 → INT8 |
| **clamp** | `max(-128, min(127, q))` | 防 INT8 溢出 |

> 💡 **关键洞察**：INT8 GEMM 的核心是 **requantize 链**——反量化 `A_f = (A-zp)*scale_A`，乘 `C_f = A_f·B_f`，再量化 `C = round(C_f / scale_C) + zp_C`。`combined_scale = scale_A*scale_B/scale_C` 可预计算，避免每元素除法。

## 5-6. 性能与复杂度

`O(M·N·K)`，INT8 字节减半（带宽翻倍），compute-bound。生产级用 INT8 Tensor Core（`mma.sync` INT8）。

> 💡 **一句话总结**：INT8 Quantized MatMul = "tiled GEMM + 反量化 + requantize"，INT8 省带宽/算力，requantize 保精度。

## 面试考点

- **手撕要求**：默写 requantize 链 + clamp。
- **高频追问**：requantize 公式推导；per-tensor vs per-channel scale；INT8 Tensor Core（DP4A/`mma.sync`）；为什么 clamp（防溢出）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 22 | [GEMM](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) | 中等 | GEMM tiling 基础 |
| 64 | [Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization) | 中等 | 反量化基础操作 |
| 81 | [INT4 MatMul](https://leetgpu.com/challenges/int4-matmul) | 中等 | 4-bit 量化进阶 |
| 57 | [FP16 Batched Matrix Multiplication](https://leetgpu.com/challenges/fp16-batched-matmul) | 中等 | FP16 + Tensor Core |

> 💡 **选题思路**：INT8 量化 GEMM，低精度计算与 requantize 流程。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
