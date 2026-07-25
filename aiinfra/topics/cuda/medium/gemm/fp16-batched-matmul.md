# LeetGPU FP16 Batched Matrix Multiplication 题解

> **面试考察度**：⭐⭐⭐⭐ FP16 batched GEMM 是 Tensor Core + batch 调度的综合题，面试常考"FP16 输入为什么 FP32 累加、batch 维怎么映射"
> **面试形式**：手写 batched tiled GEMM + 讲清 `__half` 类型与 FP32 累加精度

## 1. 题目概述

- **标题 / 题号**：FP16 Batched Matrix Multiplication（LeetGPU #57，medium）
- **链接**：https://leetgpu.com/challenges/fp16-batched-matmul
- **难度**：中等
- **标签**：CUDA、GEMM、FP16、Batched、shared memory tiling、compute-bound

**题意**：给定 `BATCH` 组 FP16 矩阵 `A[b]`（`M×K`）、`B[b]`（`K×N`），对每个 batch 独立计算 `C[b] = A[b] @ B[b]`：

$$C[b][i][j] = \sum_{k=0}^{K-1} A[b][i][k] \cdot B[b][k][j], \qquad b \in [0, \text{BATCH})$$

函数签名固定（指针为 `uint16_t`，即 `__half` 位模式）：

```cpp
extern "C" void solve(const uint16_t* A, const uint16_t* B, uint16_t* C, int BATCH, int M, int N, int K);
```

**关键约束**：

- `A`、`B`、`C` 均为 **FP16（`__half`）**，行主序，3D 布局 `(BATCH, M, K)` 等
- 容差 `atol=0.05, rtol=0.05`（松，FP16 精度限制）
- 性能测例 `BATCH=32, M=N=K=256`
- **累加必须在 FP32**（参考 `torch.bmm` 内部升 FP32 累加再降回 FP16）

> ⚠️ **精度关键**：FP16 输入但**累加用 FP32**。`__half` 的尾数仅 10 位（~3 位十进制），直接 FP16 累加 `K=256` 个元素误差远超 0.05。必须 `__half2float` 转换后用 `float` 累加，结果再 `__float2half` 写回。

> ⚠️ **batch 维映射**：grid 加一维 `batch`，`batch_idx = blockIdx.z`，每个 batch 独立计算（batch 间无依赖）。

## 2-3. CPU 基线与 GPU 设计

CPU 串行：循环 `b`，每个 batch 做 `M×K @ K×N`（同 #2）。参考用 `torch.bmm`（内部 FP32 累加）。

**GPU 映射**：grid = `(M/BM) × (N/BN) × BATCH`（3D grid，z 维是 batch）。每个 block 负责一个 batch 的一个 C tile。block 内 tiled matmul（同 #2），但 A/B 加载时用 `__half` 读、转 `float` 累加，写回转 `__half`。

> 💡 **与 #2/#22 的区别**：#2 是 FP32 单矩阵；#22 是 FP16 单矩阵 + WMMA；#57 是 FP16 batch 矩阵——加 batch 维（z 维）+ FP32 累加。骨架同 tiled matmul。

![Tiled Matmul：shared memory 分块复用](../../../images/cuda_matrix_multiplication_tiled.svg)

## 4. Kernel 实现

```cuda
// cuda_fp16_batched_matmul.cu —— FP16 Batched Tiled GEMM（FP32 累加）
// 编译: nvcc -O3 -arch=sm_120 cuda_fp16_batched_matmul.cu -o fp16bmm -lineinfo
// 运行: ./fp16bmm 32 256 256 256

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BM 32
#define BK 32
#define BN 32
#define BLOCK_SIZE (BM * BK)

// ---- FP16 Batched Tiled GEMM（FP32 累加）----
__global__ void fp16_bmm_kernel(const __half* __restrict__ A,
                                const __half* __restrict__ B,
                                __half* __restrict__ C,
                                int BATCH, int M, int N, int K) {
    __shared__ __half sA[BM][BN];
    __shared__ __half sB[BN][BK];

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;   // bz = batch
    int tx = threadIdx.x / BK, ty = threadIdx.x % BK;
    int row = bx * BM + tx, col = by * BK + ty;

    // batch 偏移
    const __half* Ab = A + (size_t)bz * M * N;
    const __half* Bb = B + (size_t)bz * N * K;
    __half* Cb = C + (size_t)bz * M * K;

    float acc = 0.0f;   // ← FP32 累加
    int numTiles = (N + BN - 1) / BN;
    for (int t = 0; t < numTiles; ++t) {
        // 加载（FP16 读）
        sA[tx][ty] = (row < M && t * BN + ty < N) ? Ab[row * N + t * BN + ty] : __float2half(0.0f);
        sB[tx][ty] = (t * BN + tx < N && col < K) ? Bb[(t * BN + tx) * K + col] : __float2half(0.0f);
        __syncthreads();

        // 累加（转 FP32）
        #pragma unroll
        for (int k = 0; k < BN; ++k)
            acc += __half2float(sA[tx][k]) * __half2float(sB[k][ty]);
        __syncthreads();
    }

    if (row < M && col < K)
        Cb[row * K + col] = __float2half(acc);   // ← 转回 FP16 写
}

int main(int argc, char** argv) {
    int BATCH=(argc>1)?atoi(argv[1]):32, M=(argc>2)?atoi(argv[2]):256, N=(argc>3)?atoi(argv[3]):256, K=(argc>4)?atoi(argv[4]):256;
    size_t bA=(size_t)BATCH*M*N*sizeof(__half), bB=(size_t)BATCH*N*K*sizeof(__half), bC=(size_t)BATCH*M*K*sizeof(__half);
    __half *hA=(__half*)malloc(bA), *hB=(__half*)malloc(bB), *hC=(__half*)malloc(bC);
    srand(42);
    for (int i=0;i<BATCH*M*N;++i) hA[i]=__float2half(((float)(rand()%100)-50)/10.0f);
    for (int i=0;i<BATCH*N*K;++i) hB[i]=__float2half(((float)(rand()%100)-50)/10.0f);

    __half *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc(&dA,bA)); CHECK_CUDA(cudaMalloc(&dB,bB)); CHECK_CUDA(cudaMalloc(&dC,bC));
    CHECK_CUDA(cudaMemcpy(dA,hA,bA,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,bB,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    dim3 grid((M+BM-1)/BM, (K+BK-1)/BK, BATCH);
    fp16_bmm_kernel<<<grid, BLOCK_SIZE>>>(dA, dB, dC, BATCH, M, N, K);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double gflops = 2.0*BATCH*M*N*K/1e9/(ms/1e3);
    printf("BATCH=%d M=%d N=%d K=%d time=%.3fms %.1fGFLOP/s\n", BATCH,M,N,K,ms,gflops);

    CHECK_CUDA(cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost));
    float maxDiff=0;
    for (int b=0;b<2;++b) for (int i=0;i<8;++i) for(int j=0;j<8;++j){
        float ref=0; for(int k=0;k<N;++k) ref+=__half2float(hA[b*M*N+i*N+k])*__half2float(hB[b*N*K+k*K+j]);
        maxDiff=fmaxf(maxDiff, fabsf(__half2float(hC[b*M*K+i*K+j])-ref));
    }
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff<0.1f?"PASS":"FAIL");

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
// fp16_batched_matmul_submit.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define BM 32
#define BK 32
#define BN 32
#define BLOCK_SIZE (BM * BK)

__global__ void fp16_bmm_kernel(const __half* A, const __half* B, __half* C,
                                int BATCH, int M, int N, int K) {
    __shared__ __half sA[BM][BN];
    __shared__ __half sB[BN][BK];
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x / BK, ty = threadIdx.x % BK;
    int row = bx * BM + tx, col = by * BK + ty;
    const __half* Ab = A + (size_t)bz * M * N;
    const __half* Bb = B + (size_t)bz * N * K;
    __half* Cb = C + (size_t)bz * M * K;

    float acc = 0.0f;
    for (int t = 0; t < (N + BN - 1) / BN; ++t) {
        sA[tx][ty] = (row < M && t * BN + ty < N) ? Ab[row * N + t * BN + ty] : __float2half(0.0f);
        sB[tx][ty] = (t * BN + tx < N && col < K) ? Bb[(t * BN + tx) * K + col] : __float2half(0.0f);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BN; ++k) acc += __half2float(sA[tx][k]) * __half2float(sB[k][ty]);
        __syncthreads();
    }
    if (row < M && col < K) Cb[row * K + col] = __float2half(acc);
}

extern "C" void solve(const uint16_t* A, const uint16_t* B, uint16_t* C, int BATCH, int M, int N, int K) {
    dim3 grid((M + BM - 1) / BM, (K + BK - 1) / BK, BATCH);
    fp16_bmm_kernel<<<grid, BLOCK_SIZE>>>((const __half*)A, (const __half*)B, (__half*)C, BATCH, M, N, K);
}
```

### 4.2 代码详解

与 #2 Matrix Multiplication **逐行对称**，差别：① 加 batch 维（`bz = blockIdx.z`，指针偏移 `bz*M*N`）；② A/B/C 类型 `__half`，累加用 `float`。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **batch 维** | `bz = blockIdx.z` | 3D grid，z 维是 batch |
| **batch 偏移** | `Ab = A + bz*M*N` | 每 batch 独立矩阵 |
| **FP16 读** | `sA[tx][ty] = Ab[...]` | shared 存 `__half` |
| **FP32 累加** | `acc += __half2float(sA) * __half2float(sB)` | 精度生命线 |
| **FP16 写回** | `Cb[...] = __float2half(acc)` | 转回 FP16 |

> 💡 **关键洞察**：FP16 GEMM 的精度关键在"输入 FP16、累加 FP32、输出 FP16"。`__half` 尾数仅 10 位，FP16 累加 `K` 个元素误差累积超容差；FP32 累加误差可控。WMMA/Tensor Core 天然支持这个流程（fragment 是 FP32 accumulator），生产级 GEMM 用 WMMA 而非 CUDA Core。

## 5-6. 性能与复杂度

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(BATCH · M · N · K)` |
| **算术强度** | FP16 字节减半，AI 翻倍 → 更 compute-bound |
| **batch 并行** | z 维独立，无依赖 |
| **瓶颈类型** | **compute-bound**（tiled + FP16） |

> 💡 **一句话总结**：FP16 Batched GEMM = "#2 tiled matmul + batch 维 + FP32 累加"。`__half` 输入省带宽，FP32 累加保精度，batch 维用 z 维并行。生产级用 WMMA/Tensor Core 加速一个数量级。

## 面试考点

- **手撕要求**：默写 batched tiled GEMM（3D grid + `__half`/`float` 转换）。
- **高频追问**：
  - **为什么 FP16 输入要 FP32 累加？** `__half` 尾数 10 位，累加 K 个误差超容差；FP32 尾数 23 位，误差可控。这是混合精度的标准做法。
  - **batch 维怎么映射？** 3D grid 的 z 维，`blockIdx.z = batch_idx`，batch 间无依赖。
  - **和 #22 GEMM 的区别？** #22 单矩阵 + WMMA；#57 batch 矩阵 + CUDA Core（朴素版）。两者都 FP32 累加。
  - **为什么 FP16 比 FP32 快？** ① 字节减半，带宽翻倍；② Tensor Core（WMMA）单指令 8192 FLOP，吞吐高一个数量级。
- **进阶延伸**：WMMA/`mma.sync`（Tensor Core）、FlashAttention 的 FP16 attention、FP8（Hopper）是更激进的低精度。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 22 | [GEMM](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) | 中等 | GEMM tiling 基础 |
| 30 | [Batched Matrix Multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication) | 中等 | FP32 batched GEMM |
| 32 | [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul) | 中等 | INT8 量化 GEMM，低精度 + scale |
| 37 | [Matrix Power](https://leetgpu.com/challenges/matrix-power) | 中等 | 重复 matmul 调度 |

> 💡 **选题思路**：FP16 + Tensor Core，半精度 GEMM。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
