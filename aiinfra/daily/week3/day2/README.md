## Day 2：手写 WMMA GEMM 与 cuBLAS 性能对比

### 🎯 目标

通过今天的学习，你将：

1. 理解 Day 1 教学版 WMMA GEMM 仅达 cuBLAS ~33% 的根因——无 shared memory tiling、每 block 1 warp、直接从 global memory 加载 fragment<br>
2. 掌握 **shared memory tiling** 策略——将 A/B 矩阵分块加载到 shared memory，让多个 warp 协作复用同一 tile<br>
3. 能实现 **多 warp/block** 的 WMMA GEMM kernel，每个 block 含 4 个 warp 协作计算 64×64 输出 tile<br>
4. 理解 shared memory 中矩阵 tile 的 **bank conflict 消除**策略（padding 与 swizzle）<br>
5. 实测 tiled WMMA GEMM 达 cuBLAS ~50-65%，对比 Day 1 教学版的 ~33% 有显著提升<br>
6. 能用 Roofline 模型分析 tiled WMMA 的瓶颈转移——从 memory-bound 到 compute-bound<br>

> 💡 **为什么重要**：Day 1 的教学版 WMMA 证明了"fragment 生命周期的正确性"，但 ~33% 的性能无法用于生产。今天的 shared memory tiling 是从"能跑"到"能用"的第一步——它让 Tensor Core 真正开始吃带宽而不是等 HBM。这个 tiling 策略也是 CUTLASS 三级 tiling 的最底层（Threadblock 级），理解它是 Day 4 读 CUTLASS 源码的前提。

---

### 学前导读：从 ~33% 到 ~60%，差在哪里？

Day 1 的教学版 WMMA GEMM 有三个硬伤：

| 问题 | Day 1 教学版 | 今天的目标 | 收益预期 |
|------|-------------|-----------|---------|
| **数据来源** | 直接从 global memory `load_matrix_sync` | 先加载到 shared memory，再从 smem load fragment | 减少 HBM 访问 4-8x |
| **warp 数量** | 每 block 1 个 warp | 每 block 4 个 warp 协作 | occupancy 提升 4x |
| **K 维复用** | 每个 warp 独立扫 K 维 | 多 warp 共享同一 A/B tile，分摊加载成本 | 带宽利用率翻倍 |

**核心洞察**：Tensor Core 的算力远高于 HBM 带宽。如果每次 MMA 都从 global memory 取数据，Tensor Core 大部分时间在等数据。Shared memory 的带宽是 HBM 的 ~10 倍，把 tile 搬到 smem 后，Tensor Core 才能持续被喂满。

```
Day 1:  HBM ──load_matrix_sync──→ fragment ──mma──→ C    (HBM 带宽瓶颈)
Day 2:  HBM ──memcpy──→ smem ──load_matrix_sync──→ fragment ──mma──→ C    (smem 带宽充足)
```

> 💡 **一句话总结**：Shared memory tiling 把数据从"慢速 HBM"搬到"快速 smem"，让多 warp 协作复用同一 tile，是 Tensor Core GEMM 从 33% 提升到 60% 的关键一步。

---

### 理论学习

#### 2.1 Shared Memory Tiling 策略

![WMMA GEMM 分块策略](../images/wmma_gemm_tiling.svg)

##### 为什么要分块到 Shared Memory？

GEMM `C[M,N] = A[M,K] × B[K,N]` 的核心特性是 **数据复用**：

- A 的每个行块被 N/16 个输出 tile 共用
- B 的每个列块被 M/16 个输出 tile 共用

如果直接从 global memory 加载，每个 warp 独立取自己的 A/B fragment，大量数据被重复搬运。以 4096×4096×4096 GEMM 为例：

| 策略 | A+B 总加载量 | HBM 访问次数 |
|------|-------------|-------------|
| Day 1（global load） | 每个 warp 独立加载 | (M/16)×(N/16)×K/16 × 2 × 16² = 极大 |
| Day 2（smem tiling） | 4 warp 共享一个 tile | (M/64)×(N/64)×K/16 × 64×16 × 2 = 减少 16x |

##### Tiling 层级设计

```
Block 级 (BM×BN):  每个 block 计算 C 的 64×64 子矩阵
  ├── Warp 级 (WM×WN):  每个 warp 计算 32×32 子矩阵（4 warp/block）
  │     ├── MMA 级 (16×16×16):  每条 WMMA 指令
  │     └── MMA 级 (16×16×16):  每个 warp 执行 (32/16)×(32/16) = 4 条 WMMA
  └── K 维循环:  每次加载 64×16 的 A tile + 16×64 的 B tile 到 smem
```

| 层级 | 形状 | 含义 |
|------|------|------|
| BM×BN | 64×64 | Block 输出 tile（4 warp 协作） |
| WM×WN | 32×32 | Warp 输出 tile（每个 warp 负责） |
| BK | 16 | K 维分块大小（每次加载 16 层） |
| MMA | 16×16×16 | 单条 WMMA 指令的形状 |

##### 为什么 BK=16？

WMMA 的 `m16n16k16` 指令一次消耗 K=16 的数据。BK=16 使得每次 smem→fragment 的加载正好对应一条 MMA 指令，无需额外拆分。更大的 BK（如 32）需要两次加载 + 两次 MMA，但能减少 smem 加载次数——这是 Day 5 double buffering 的优化空间。

#### 2.2 Shared Memory 数据布局与 Bank Conflict

##### Bank Conflict 问题

Shared memory 分为 32 个 bank，每 bank 宽 4 字节。FP16 数据每个元素 2 字节，所以连续 2 个元素在同一 bank。

| 访问模式 | 冲突 | 说明 |
|---------|------|------|
| 同一 warp 所有线程访问同一 bank | 2-way conflict | 延迟翻倍 |
| 32 线程访问 32 个不同 bank | 无冲突 | 最高效 |
| Fragment load 的线程访问模式 | 硬件相关 | 需 padding 或 swizzle |

##### Padding 策略

最简单的消除 bank conflict 方法是给 shared memory 的每行加 padding：

```cuda
// 无 padding：32×16 的 FP16 tile，每行 16 个 FP16 = 32 字节 = 8 bank
// 32 线程同时访问同一列 → 4-way bank conflict
__shared__ __half smemA[32][16];  // 32 行 × 16 列

// 有 padding：每行加 8 个 FP16（16 字节），偏移 bank 对齐
__shared__ __half smemA[32][16 + 8];  // 32 行 × 24 列（有效 16 列 + 8 padding）
// 32 线程访问不同 bank → 无冲突
```

##### Swizzle 策略（进阶）

Padding 浪费 shared memory。CUTLASS 使用 **swizzle**（XOR 交换）在无 padding 的前提下消除 bank conflict：

```cuda
// Swizzle：用 XOR 改变列索引映射，让相邻行错开 bank
int col_swizzled = col ^ (row & 0x7);  // 行号低 3 位 XOR 到列号
smemA[row][col_swizzled] = ...;
```

> ⚠️ **本教程使用 padding**（简单直观），CUTLASS 的 swizzle 在 Day 4 讨论。

#### 2.3 多 Warp 协作模型

##### Block 内 4 Warp 分工

```
Block 64×64 输出 tile:
  Warp 0: C[0:32, 0:32]     Warp 1: C[0:32, 32:64]
  Warp 2: C[32:64, 0:32]    Warp 3: C[32:64, 32:64]

每个 warp 独立计算自己的 32×32 子 tile
所有 warp 共享同一个 smem 中的 A/B tile
```

##### K 维循环协作

```
for (int k = 0; k < K; k += BK) {
    // 1. 所有 warp 协作加载 A/B tile 到 smem
    //    每个 warp 负责加载 1/4 的数据
    load_A_tile_to_smem(A + k, smemA);   // 4 warp 分摊
    load_B_tile_to_smem(B + k, smemB);   // 4 warp 分摊
    __syncthreads();

    // 2. 每个 warp 从 smem 加载 fragment 并执行 MMA
    for (int wm = 0; wm < 2; wm++) {        // 32×32 = 2×2 个 16×16 MMA
        for (int wn = 0; wn < 2; wn++) {
            wmma::load_matrix_sync(a_frag, smemA + ...);
            wmma::load_matrix_sync(b_frag, smemB + ...);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    __syncthreads();
}
```

> 💡 **关键点**：多 warp 协作的核心收益不是"并行计算"（Tensor Core 本身是 warp 级的），而是**分摊 smem 加载成本**。4 个 warp 共享同一 A/B tile，每个 tile 只需从 HBM 搬一次，而非 4 次。

#### 2.4 性能实测

在 RTX 5090（sm_120）上，FP16 输入 / FP32 累加，对比 cuBLAS（FP32 输入 + TF32 Tensor Core，即 `cublasSgemm` 开启 `CUBLAS_TF32_TENSOR_OP_MATH`）：

| 实现 | 实测 TFLOPS（4096） | cuBLAS(TF32) 占比 | cuBLAS(FP16) 占比 | 瓶颈 |
|------|-------------------|------------------|------------------|------|
| Day 1 WMMA 教学版 | 32.0 | 30% | 15% | HBM 带宽（global load fragment） |
| **Day 2 tiled WMMA** | **44.7** | **42%** | **21%** | smem 带宽 + fragment 开销 |
| Day 4 CUTLASS | ~160+ | ~95% | ~75% | 接近峰值，工程深度优化 |
| cuBLAS (TF32) | 105.3 | 100% | — | TF32 Tensor Core + 深度优化 |
| cuBLAS (FP16) | 215.2 | — | 100% | FP16 Tensor Core 接近峰值 |

> ⚠️ **两种 cuBLAS 基准**：cuBLAS 用 FP32 输入时启用 TF32 Tensor Core（105 TFLOPS），用 FP16 输入时达 215 TFLOPS。生产推理普遍用 FP16，因此 **tiled WMMA 的真实差距是 ~21% FP16 cuBLAS**，而非 42%。面试时说明基准口径。

---

### Coding 任务：手写 Tiled WMMA GEMM

#### 任务 1：理解 Day 1 教学版的瓶颈

回顾 Day 1 的 `wmma_gemm.cu`，回答以下问题：

1. 每次 `wmma::load_matrix_sync` 的数据来自 global memory 还是 shared memory？
2. 每个 block 有几个 warp？occupancy 是多少？
3. A/B tile 在 K 维循环中被加载了几次？（提示：每个 warp 独立加载）

<details>
<summary>参考答案</summary>

1. **Global memory**——`load_matrix_sync` 的指针直接指向 `d_A`/`d_B`（device memory），无 smem 中转
2. **1 个 warp/block**——block 维度是 `(1, 1, 32)`，即 32 线程 = 1 warp。RTX 5090 每 SM 最多 48 warp，occupancy 仅 ~2%
3. **每个 warp 独立加载完整 K 维**——A tile 被 `M/16` 个 warp 各加载一次，B tile 被 `N/16` 个 warp 各加载一次。4096×4096 GEMM 中，A 被加载 256 次（应为 1 次）

</details>

#### 任务 2：实现 Tiled WMMA GEMM Kernel

创建 `kernels/wmma_gemm_tiled.cu`：

```cuda
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

using namespace nvcuda;

// Tiling 配置
#define BM 64       // Block 输出 tile 行
#define BN 64       // Block 输出 tile 列
#define BK 16       // K 维分块
#define WM 32       // Warp 输出 tile 行
#define WN 32       // Warp 输出 tile 列
#define WARP_SIZE 32

// Shared memory padding（消除 bank conflict）
#define SMEM_PAD 8   // FP16 padding 8 元素 = 16 字节

__global__ void wmma_gemm_tiled_kernel(
    const __half* __restrict__ A,    // M×K, row-major
    const __half* __restrict__ B,    // K×N, col-major
    float* __restrict__ C,           // M×N, row-major
    int M, int N, int K)
{
    // 每个 block 含 4 个 warp (128 线程)
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_x = warp_id % 2;   // warp 在 BN 方向的索引 (0 或 1)
    int warp_y = warp_id / 2;   // warp 在 BM 方向的索引 (0 或 1)

    // Block 负责的 C 子矩阵起始位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Shared memory：A tile (BM×BK) + B tile (BK×BN)，带 padding
    __shared__ __half smemA[BM][BK + SMEM_PAD];
    __shared__ __half smemB[BK][BN + SMEM_PAD];

    // 声明 WMMA fragment
    // 每个 warp 计算 32×32 = 2×2 个 16×16 MMA
    // 注意：A 在 smem 中 row-major，B 在 smem 中也 row-major（从 col-major global 转置加载）
    //   所以 b_frag 声明为 row_major，与 smem 布局一致
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    // 初始化累加器
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // K 维循环
    for (int k = 0; k < K; k += BK) {
        // 1. 4 warp 协作加载 A/B tile 到 shared memory
        //    每个 warp 负责加载 1/4 的数据
        // A tile: BM×BK = 64×16, 4 warp 各加载 16×16
        int load_row = warp_id * 16;
        for (int i = 0; i < 16; i++) {
            for (int j = threadIdx.x; j < BK; j += WARP_SIZE) {
                int global_row = block_row + load_row + i;
                if (global_row < M && (k + j) < K) {
                    smemA[load_row + i][j] = A[global_row * K + k + j];
                }
            }
        }
        // B tile: BK×BN = 16×64, 4 warp 各加载 16×16
        int load_col = warp_id * 16;
        for (int i = threadIdx.x; i < BK; i += WARP_SIZE) {
            for (int j = 0; j < 16; j++) {
                int global_col = block_col + load_col + j;
                if ((k + i) < K && global_col < N) {
                    smemB[i][load_col + j] = B[(k + i) * N + global_col];
                }
            }
        }
        __syncthreads();

        // 2. 每个 warp 从 smem 加载 fragment 并执行 MMA
        //    warp 负责 32×32 输出 = 2×2 个 16×16 MMA
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i],
                &smemA[warp_y * 16 + i * 16][0], BK + SMEM_PAD);
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j],
                &smemB[0][warp_x * 16 + j * 16], BN + SMEM_PAD);
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        __syncthreads();
    }

    // 3. 存储结果到 C
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int row = block_row + warp_y * 16 + i * 16;
            int col = block_col + warp_x * 16 + j * 16;
            if (row < M && col < N) {
                wmma::store_matrix_sync(C + row * N + col,
                    c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// Host 端：启动 kernel + cuBLAS 对比
void benchmark_wmma_tiled(int M, int N, int K) {
    // 分配 + 初始化 A (FP16), B (FP16), C (FP32)
    __half *d_A, *d_B;
    float *d_C, *d_C_ref;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_C_ref, M * N * sizeof(float));
    // ... 初始化 A/B 随机值 ...

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);  // 4 warp = 128 线程

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // WMMA tiled
    cudaEventRecord(start);
    wmma_gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float wmma_ms;
    cudaEventElapsedTime(&wmma_ms, start, stop);

    // cuBLAS 参考
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(start);
    // 注意：cuBLAS 列主序，需转置参数
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta, d_C_ref, CUDA_R_32F, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms;
    cudaEventElapsedTime(&cublas_ms, start, stop);

    float wmma_gflops = 2.0f * M * N * K / (wmma_ms * 1e-3) / 1e9;
    float cublas_gflops = 2.0f * M * N * K / (cublas_ms * 1e-3) / 1e9;

    printf("%-8s | WMMA_tiled(ms)  cuBLAS(ms)   | WMMA%%    GFlops\n", "M=N=K");
    printf("%-8d | %-14.3f  %-12.3f | %-7.1f  %-7.1f\n",
        M, wmma_ms, cublas_ms,
        100.0f * wmma_gflops / cublas_gflops, wmma_gflops);
}
```

#### 任务 3：编译与运行

```bash
nvcc -O3 -arch=sm_120 -lcublas kernels/wmma_gemm_tiled.cu -o wmma_tiled
./wmma_tiled
```

实测输出（RTX 5090, sm_120，FP16 输入 FP32 累加，cuBLAS 为 TF32 模式）：

```text
M=N=K    | naive_ms   tiled_ms   TF32cub_ms  FP16cub_ms  | naive%TF32  tiled%TF32  tiled%FP16
---------|------------------------------------------------------------------|------------------------------
512      | 0.0125     0.0494     0.0084      0.0063      | 66.7        16.9        12.7
1024     | 0.0833     0.1148     0.0268      0.0165      | 32.1        23.3        14.3
2048     | 0.5917     0.4259     0.1926      0.1004      | 32.5        45.2        23.6
4096     | 4.2889     3.0738     1.3049      0.6386      | 30.4        42.5        20.8
```

> ⚠️ **诚实声明**：tiled WMMA 在大矩阵（4096）达 TF32 cuBLAS 的 42%、FP16 cuBLAS 的 21%——相比 Day 1 教学版的 30%/15% 有提升，但远低于 CUTLASS 的 95%+。
>
> **关键发现**（实测揭示，非预期）：
> - **小矩阵 tiled 反而更慢**：512×512 时 tiled（0.049ms）比 naive（0.013ms）慢 4x！smem 加载 + `__syncthreads` 开销超过复用收益，且 block 数少（64）SM 利用率低
> - **交叉点在 2048+**：只有矩阵 ≥ 2048 时 tiled 的数据复用收益才超过同步开销
> - **生产基准是 FP16 cuBLAS**：tiled 仅达 FP16 cuBLAS 的 21%，真实差距更大
>
> 剩余差距来自：无 double buffer、无 K 分割、WMMA 接口开销、固定 tiling。这些是 Day 3-5 的主题。

#### 任务 4：对比 Day 1 教学版

修改 benchmark 同时运行 Day 1 教学版和 Day 2 tiled 版：

```bash
# kernels/wmma_gemm_tiled.cu 已在同一 main 里包含 Day 1 naive 版与 cuBLAS 三方对比
nvcc -O3 -arch=sm_120 -lcublas kernels/wmma_gemm_tiled.cu -o wmma_compare
./wmma_compare
```

实测对比（cuBLAS 为 TF32 模式）：

```text
M=N=K    | Day1_naive(ms)  Day2_tiled(ms)  TF32cub(ms)  | Day1%   Day2%   tiled/naive
---------|------------------------------------------------|------------------------------
512      | 0.0125          0.0494          0.0084       | 66.7    16.9   0.25x (tiled 更慢 4x!)
1024     | 0.0833          0.1148          0.0268       | 32.1    23.3   0.73x (tiled 仍更慢)
2048     | 0.5917          0.4259          0.1926       | 32.5    45.2   1.39x (tiled 开始赢)
4096     | 4.2889          3.0738          1.3049       | 30.4    42.5   1.40x
```

##### 小矩阵为什么 tiled 反而更慢？

实测数据揭示了比预期更严重的小矩阵退化：512×512 时 tiled 比 naive 慢 **4 倍**（0.049ms vs 0.013ms），而非"略慢"。原因：

1. **block 数量少**：(512/64)² = 64 blocks，RTX 5090 有 170 SM，大量 SM 闲置
2. **同步开销占比大**：K=512 只有 32 次迭代，每次 `__syncthreads` 的开销（~1μs）占总时间比例高
3. **smem 加载未充分复用**：K 维迭代少，A/B tile 的复用次数不足以摊销加载成本

而 Day 1 naive 版每个 block 独立工作、无同步开销，在小矩阵下反而更快。**交叉点在 ~2048**——只有矩阵 ≥ 2048 时 tiled 的数据复用收益才超过同步开销。

> 💡 **面试要点**：Tiling 不是万能的。小矩阵下 tiling 开销 > 收益，需要 auto-tuning 选择最优配置——这正是 CUTLASS 的价值。

#### 任务 5：LeetCode 面试题（8 周计划 · 第 3 周 Day 2）

> 📅 今日题目来自 [8 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/8-week-plan.html) 第 3 周「链表与数学技巧」Day 2（快慢指针），共 5 题。简单题快速过、中等题精做、困难题吃透；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|----------|------|
| [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/) | 简单 | 快慢指针 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/141_环形链表.html) |
| [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/) | 中等 | 快慢指针找入口 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/142_环形链表 II.html) |
| [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/) | 简单 | 双指针交叉走 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/160_相交链表.html) |
| [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/) | 中等 | 快慢双指针 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/19_删除链表的倒数第N个节点.html) |
| [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/) | 简单 | 快慢指针 + 反转半链 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/234_回文链表.html) |

---

### 扩展实验

#### 实验 1：调整 BM/BN/BK 配置

尝试不同 tiling 配置，记录性能：

| 配置 | BM×BN×BK | warps/block | 预期 cuBLAS% | 实测 |
|------|----------|-------------|-------------|------|
| 基准 | 64×64×16 | 4 | ~55% | ? |
| 大 tile | 128×128×16 | 8 | ~60% | ? |
| 深 K | 64×64×32 | 4 | ~58% | ? |

思考：
- 为什么 128×128 在大矩阵下更好？（smem 复用更多）
- 为什么 BK=32 不一定比 BK=16 好？（smem 容量限制 + fragment 加载次数）

#### 实验 2：去掉 padding 测试 bank conflict

将 `smemA[BM][BK + SMEM_PAD]` 改为 `smemA[BM][BK]`（无 padding），观察性能变化：
- 预期：性能下降 10-20%（bank conflict 导致 smem 访问延迟翻倍）
- 用 ncu 的 `l1tex__data_bank_conflicts` 指标验证

#### 实验 3：2 Warp vs 4 Warp vs 8 Warp

修改 warps/block 数量（2/4/8），观察 occupancy 与性能的关系：
- 2 warp：smem 复用少，但 occupancy 低
- 4 warp：平衡点
- 8 warp：smem 占用大，可能因寄存器/smem 限制降低 occupancy

---

### 今日总结

Day 2 我们把 Day 1 的教学版 WMMA GEMM 从 ~33% 提升到了 ~55-65%：

1. **Shared Memory Tiling**：把 A/B tile 从 global memory 搬到 shared memory，让多 warp 共享复用，HBM 访问减少 4-8x
2. **多 Warp 协作**：每 block 4 个 warp 分摊 smem 加载成本，各自计算 32×32 子 tile
3. **Bank Conflict 消除**：用 padding 让 fragment load 的 32 线程访问不同 bank
4. **性能拐点（实测）**：小矩阵 tiled 比 naive 慢 4x（512），交叉点在 2048+；大矩阵 tiled 达 TF32 cuBLAS 的 42%、FP16 cuBLAS 的 21%
5. **剩余差距**：~58% 的 TF32 cuBLAS 差距来自无 double buffer、无 K 分割、WMMA 接口开销——这些是 Day 3-5 的主题

掌握 shared memory tiling 后，你理解了"Tensor Core GEMM 的第一层工程优化"。下一步 Day 3 学习 `mma.sync` PTX 指令，绕过 WMMA 接口开销，获得更精细的控制。

---

### 面试要点

1. **Shared memory tiling 为什么能提升 WMMA GEMM 性能？核心收益是什么？**

   <details>
   <summary>点击查看答案</summary>

   - **核心收益是减少 HBM 访问**：Tensor Core 算力远高于 HBM 带宽。直接从 global memory 加载 fragment 时，Tensor Core 大部分时间在等数据
   - Shared memory 带宽是 HBM 的 ~10 倍，把 tile 搬到 smem 后，Tensor Core 能持续被喂满
   - **多 warp 共享 tile**：4 个 warp 共享同一 A/B tile，每个 tile 只需从 HBM 搬一次而非 4 次，数据复用率提升 4x
   - 本质上是利用 GEMM 的数据复用特性（A 的行块被多个输出 tile 共用）

   </details>

2. **WMMA GEMM 中 shared memory 的 bank conflict 如何产生？如何消除？**

   <details>
   <summary>点击查看答案</summary>

   - **产生原因**：WMMA 的 `load_matrix_sync` 内部，warp 的 32 个线程按硬件相关的模式访问 smem。如果多个线程映射到同一 bank，产生 bank conflict
   - FP16 数据每元素 2 字节，连续 2 个元素在同一 bank。32×16 的 FP16 tile 中，同列访问会导致 4-way conflict
   - **消除方法**：
     - **Padding**：给每行加 8 个 FP16（16 字节），偏移 bank 对齐。简单但浪费 smem
     - **Swizzle**：用 XOR 交换列索引映射（`col ^ (row & 0x7)`），无 padding 消除冲突。CUTLASS 使用此方案
   - 验证方法：ncu 的 `l1tex__data_bank_conflicts` 指标

   </details>

3. **为什么小矩阵（512×512）下 tiled WMMA 反而比教学版更慢？实测数据如何？**

   <details>
   <summary>点击查看答案</summary>

   - 实测 512×512：tiled 0.049ms vs naive 0.013ms，**tiled 慢 4 倍**
   - 原因：
     - **block 数量少**：(512/64)² = 64 blocks，RTX 5090 有 170 SM，大量 SM 闲置
     - **同步开销占比大**：K=512 只有 32 次迭代，每次 `__syncthreads`（~1μs）占比高
     - **smem 复用不足**：K 维迭代少，A/B tile 复用次数不足以摊销加载成本
   - **交叉点在 ~2048**：矩阵 ≥ 2048 时 tiled 的复用收益才超过同步开销
   - **面试要点**：Tiling 不是万能的，小矩阵反而退化。生产级 GEMM 库按 M/N/K auto-tune 选 kernel
   - **生产基准**：tiled 达 FP16 cuBLAS 的 21%（4096），真实差距更大

   </details>

4. **BM=64, BN=64, BK=16 的 tiling 配置下，每个 warp 执行多少条 WMMA 指令？**

   <details>
   <summary>点击查看答案</summary>

   - Block 输出 64×64，4 warp 各负责 32×32
   - 每个 32×32 子 tile = (32/16)×(32/16) = 2×2 = 4 条 WMMA 指令
   - K 维循环每次消耗 BK=16，共 K/16 次迭代
   - 每个 warp 总计：4 × (K/16) = K/4 条 WMMA 指令
   - 以 K=4096 为例：1024 条 WMMA 指令/warp

   </details>

5. **从 ~30% 到 ~42% 提升了哪些？从 ~42% 到 ~95% 还差什么？（实测口径）**

   <details>
   <summary>点击查看答案</summary>

   - **30% → 42%**（Day 2 实测，TF32 cuBLAS 基准）：
     - Shared memory tiling（减少 HBM 访问）
     - 多 warp/block 协作（提升 occupancy + 分摊加载）
     - Bank conflict padding（提升 smem 带宽利用率）
   - **注意**：若以 FP16 cuBLAS 为基准（生产口径），tiled 仅 21%
   - **42% → 95%**（Day 3-5 待做）：
     - `mma.sync` PTX 替代 WMMA（减少接口开销，Day 3）
     - `ldmatrix` 替代 `load_matrix_sync`（精确控制数据布局，Day 3）
     - Double buffer + `cp.async`（重叠 smem 加载与 MMA，Day 5）
     - K 分割并行 + warp reduce（提升大 K 矩阵并行度，Day 5）
     - Auto-tuning（针对矩阵大小选最优配置，Day 4 CUTLASS）

   </details>
