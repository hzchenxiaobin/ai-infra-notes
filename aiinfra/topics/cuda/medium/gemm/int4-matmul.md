# LeetGPU INT4 Weight-Only Quantized MatMul 题解

> **面试考察度**：⭐⭐⭐⭐ INT4 是极致低精度，面试常考"4-bit 打包解包 + group-wise dequant"
> **面试形式**：手写 + 讲清"nibble 解包与 signed 转换"

## 1. 题目概述

- **标题 / 题号**：INT4 Weight-Only Quantized MatMul（LeetGPU #81，medium）
- **链接**：https://leetgpu.com/challenges/int4-matmul
- **难度**：中等
- **标签**：CUDA、GEMM、INT4、weight-only 量化、nibble 解包、group-wise dequant、compute-bound

**题意**：`weight-only` 量化——`x` 是 FP16，`w_q` 是打包的 INT4（每 byte 2 个 nibble），`scales` 是 FP16（per-group）。反量化 `w_dequant = (nibble - 8) * scale[group]`，然后 `y = x @ w_dequant.T`（FP32 累加，FP16 输出）。签名 `solve(const uint16_t* x, const uint8_t* w_q, const uint16_t* scales, uint16_t* y, int M, int N, int K, int group_size)`。容差 `0.01`，性能测例 `M=N=K=4096, group_size=128`。

**INT4 解包**：byte 的 high nibble（`byte>>4`）是 `w[n, 2i]`，low nibble（`byte&0xF`）是 `w[n, 2i+1]`。signed 值 = `nibble - 8`（范围 `[-8, 7]`）。`group_size` 个元素共享一个 scale。

## 2-4. 设计与实现

tiled matmul（同 #2），但：① `x` 是 FP16；② `w_q` 需解包 INT4 → `(nibble-8) * scale[group]` 反量化；③ FP32 累加；④ FP16 输出。

![量化与反量化](../../../images/cuda_quantization_overview.svg)

```cuda
// int4_matmul_submit.cu —— 朴素 INT4 weight-only GEMM
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__global__ void int4_matmul_kernel(const __half* x, const uint8_t* w_q, const __half* scales,
                                   __half* y, int M, int N, int K, int group_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // M 维
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // N 维
    if (i >= M || j >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float xv = __half2float(x[i * K + k]);
        // 解包 INT4：w_q[j, k] 在 byte j*K/2 + k/2 的 (k%2==0? high : low) nibble
        int byte_idx = j * K / 2 + k / 2;
        uint8_t byte = w_q[byte_idx];
        uint8_t nibble = (k % 2 == 0) ? (byte >> 4) : (byte & 0xF);
        float wv = ((float)(int)nibble - 8.0f) * __half2float(scales[j * (K / group_size) + k / group_size]);
        acc += xv * wv;
    }
    y[i * N + j] = __float2half(acc);
}

extern "C" void solve(const uint16_t* x, const uint8_t* w_q, const uint16_t* scales,
                      uint16_t* y, int M, int N, int K, int group_size) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    int4_matmul_kernel<<<grid, block>>>((const __half*)x, w_q, (const __half*)scales,
                                        (__half*)y, M, N, K, group_size);
}
```

### 4.1 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **nibble 解包** | `byte >> 4` 或 `byte & 0xF` | 2 个 INT4 共享 1 byte |
| **signed 转换** | `nibble - 8` | INT4 范围 `[-8, 7]` |
| **group dequant** | `* scales[k / group_size]` | 每 group_size 个元素一个 scale |
| **FP32 累加** | `acc += xv * wv` | x 是 FP16，w 反量化后 FP32 |

> 💡 **关键洞察**：INT4 weight-only 量化的难点在**解包**——每 byte 含 2 个 nibble，需按 `k%2` 取 high/low。signed 值 `nibble-8` 是对称量化（`[-8,7]`）。group-wise scale 每 `group_size` 个元素一个。INT4 比 INT8 再省 2× 带宽。

## 5-6. 性能与复杂度

`O(M·N·K)`，INT4 权重字节省 8×（vs FP32），compute-bound。生产级用 INT4 Tensor Core（Hopper）。

> 💡 **一句话总结**：INT4 MatMul = "nibble 解包 + group dequant + tiled GEMM"，极致低精度省带宽。

## 面试考点

- **手撕要求**：默写 nibble 解包 + signed 转换。
- **高频追问**：nibble high/low 怎么取；为什么 `nibble-8`（对称量化 `[-8,7]`）；weight-only vs 全量化（activation 也量化）；group_size 对精度影响。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 32 | [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul) | 中等 | INT8 量化 GEMM 对比 |
| 64 | [Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization) | 中等 | 反量化基础操作 |
| 22 | [GEMM](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) | 中等 | GEMM tiling 基础 |
| 57 | [FP16 Batched Matrix Multiplication](https://leetgpu.com/challenges/fp16-batched-matmul) | 中等 | FP16 + Tensor Core |

> 💡 **选题思路**：INT4 权重量化 GEMM，练习 4-bit 打包解包与低精度计算。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
