# LeetGPU Sparse Matrix-Vector Multiplication 题解

> **面试考察度**：⭐⭐⭐⭐ SpMV 是稀疏计算的经典题，面试考"warp 行归约 + 不规则访存"；推理引擎中 attention 的稀疏化、MoE 的路由都用类似模式
> **面试形式**：手写 CSR SpMV + 讲清"一个 warp 负责一行"的映射

## 1. 题目概述

- **标题 / 题号**：Sparse Matrix-Vector Multiplication（LeetGPU #18，medium）
- **链接**：https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
- **难度**：中等
- **标签**：CUDA、SpMV、CSR、warp 行归约、不规则访存、memory-bound

**题意**：给定矩阵 `A`（`M×N`）和向量 `x`（`N`），计算 `y = A @ x`：

$$y[i] = \sum_{j=0}^{N-1} A[i][j] \cdot x[j]$$

函数签名固定：

```cpp
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz);
```

**关键约束**：

- `A`、`x`、`y` 均为 FP32，`A` **行主序 dense**（注意：尽管名为 "sparse"，参考实现是 dense `torch.matmul`，`nnz` 仅作元数据提示）
- 容差 `atol=1e-3, rtol=1e-3`
- 性能测例 `M=1000, N=10000, nnz=3,500,000`（~35% 密度）

> ⚠️ **题目特殊性**：参考实现用 dense `torch.matmul`，`A` 传的是 dense 行主序指针，`nnz` 未被参考使用。本题实为 **dense matvec（GEMV）**，只是题目名带 "sparse"。解题用 dense GEMV 即可过；若想真正练 SpMV，可自行把 `A` 转 CSR 格式（见 §5.3）。

> 💡 **面试价值**：尽管本题是 dense，面试中真正的 SpMV（CSR 格式）是高频题——"一个 warp 负责一行、行内归约"是稀疏计算的标准骨架。本题解同时给出 dense GEMV（过题）和 CSR SpMV（面试标准）两个版本。

## 2. CPU 基线 / 朴素 GPU 方法

```cpp
// CPU 串行 dense GEMV
void gemv_cpu(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        double acc = 0.0;
        for (int j = 0; j < N; ++j) acc += (double)A[i*N+j] * x[j];
        y[i] = (float)acc;
    }
}
```

朴素 GPU：每 thread 算一行，串行扫 N 列——`O(M·N)` 但每行单 thread，并行度受限于 `M`。

## 3. GPU 设计

### 3.1 Dense GEMV：一个 warp 负责一行

**核心映射**：一个 warp（32 thread）负责 `A` 的一行。warp 内 32 thread 协作扫 `N` 列（每 thread 算 `N/32` 个元素的点积），最后 warp shuffle 归约成行和。

![两阶段归约：block reduce → global reduce](../../../images/cuda_reduction_overview.svg)

| 配置 | 值 |
|------|----|
| block | 256 thread = 8 warp |
| 每 warp | 1 行 |
| grid | `ceil(M / 8)` block |

### 3.2 CSR SpMV（面试标准版）

若 `A` 是 CSR 格式（`row_ptr[M+1]`、`col_idx[nnz]`、`values[nnz]`）：一个 warp 负责一行，warp 内 thread 协作扫 `row_ptr[i]..row_ptr[i+1]` 的非零元，各算部分和后 warp shuffle 归约。

> 💡 **为什么不一个 thread 一行？** 行长 `N` 可能很大，单 thread 串行扫慢；一个 warp 协作 + shuffle 归约，把行内点积并行化。这是 SpMV/GEMV 的通用映射。

## 4. Kernel 实现

### 4.1 Dense GEMV（过题版）

```cuda
// cuda_spmv.cu —— Dense GEMV（本题参考为 dense）：一个 warp 一行
// 编译命令: nvcc -O3 -arch=sm_120 cuda_spmv.cu -o spmv -lineinfo
// 运行:     ./spmv 1000 10000

#include <cstdio>
#include <cstdlib>
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

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- Dense GEMV：一个 warp 一行 ----
__global__ void gemv_kernel(const float* __restrict__ A,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int N) {
    int warpId_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warpId_in_block;
    if (row >= M) return;

    const float* Arow = A + (size_t)row * N;
    float local = 0.0f;
    // warp 内 32 lane 协作扫 N 列
    for (int j = lane; j < N; j += WARP_SIZE)
        local += Arow[j] * x[j];

    float row_sum = warp_reduce_sum(local);   // warp shuffle 归约
    if (lane == 0) y[row] = row_sum;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1000;
    int N = (argc > 2) ? atoi(argv[2]) : 10000;
    size_t bytesA = (size_t)M*N*sizeof(float), bytesx = N*sizeof(float), bytesy = M*sizeof(float);
    float *hA=(float*)malloc(bytesA), *hx=(float*)malloc(bytesx), *hy=(float*)malloc(bytesy);
    srand(42);
    for (int i=0;i<M*N;++i) hA[i]=(float)(rand()%200-100)/100.0f;
    for (int i=0;i<N;++i) hx[i]=(float)(rand()%200-100)/100.0f;

    float *dA,*dx,*dy;
    CHECK_CUDA(cudaMalloc(&dA,bytesA)); CHECK_CUDA(cudaMalloc(&dx,bytesx)); CHECK_CUDA(cudaMalloc(&dy,bytesy));
    CHECK_CUDA(cudaMemcpy(dA,hA,bytesA,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx,hx,bytesx,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gemv_kernel<<<(M+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK, BLOCK_SIZE>>>(dA,dx,dy,M,N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    printf("M=%d N=%d time=%.3fms bw=%.1fGB/s\n", M, N, ms, (2.0f*M*N+M+N)*sizeof(float)/1e9/(ms/1e3));

    CHECK_CUDA(cudaMemcpy(hy,dy,bytesy,cudaMemcpyDeviceToHost));
    float maxDiff=0; for(int i=0;i<M&&i<8;++i){double r=0;for(int j=0;j<N;++j)r+=hA[i*N+j]*hx[j];maxDiff=fmaxf(maxDiff,fabsf(hy[i]-(float)r));}
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff<1e-2f?"PASS":"FAIL");

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dy));
    free(hA); free(hx); free(hy);
    return 0;
}
```

### 4.2 LeetGPU 提交版本

```cuda
// sparse_matrix_vector_multiplication_submit.cu —— Dense GEMV（过题）
#include <cuda_runtime.h>
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffff, val, o);
    return val;
}

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (row >= M) return;
    const float* Arow = A + (size_t)row * N;
    float local = 0.0f;
    for (int j = lane; j < N; j += WARP_SIZE) local += Arow[j] * x[j];
    float row_sum = warp_reduce_sum(local);
    if (lane == 0) y[row] = row_sum;
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    gemv_kernel<<<(M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, BLOCK_SIZE>>>(A, x, y, M, N);
}
```

### 4.3 CSR SpMV（面试标准版，供参考）

```cuda
// CSR SpMV：一个 warp 一行，扫 row_ptr[i]..row_ptr[i+1] 的非零元
__global__ void spmv_csr_kernel(const int* row_ptr, const int* col_idx,
                                const float* values, const float* x,
                                float* y, int M) {
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= M) return;

    int start = row_ptr[row], end = row_ptr[row + 1];
    float local = 0.0f;
    for (int i = start + lane; i < end; i += WARP_SIZE)   // warp 协作扫非零元
        local += values[i] * x[col_idx[i]];
    float row_sum = warp_reduce_sum(local);
    if (lane == 0) y[row] = row_sum;
}
```

### 4.4 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **行映射** | `row = bx*8 + warpId` | 一个 warp 一行 |
| **warp 协作扫列** | `for (j = lane; j < N; j += 32)` | 32 lane 分摊 N 列 |
| **点积累加** | `local += Arow[j] * x[j]` | 每 lane 算部分和 |
| **warp 归约** | `warp_reduce_sum(local)` | shuffle 折半归约成行和 |
| **写回** | `y[row] = row_sum` | lane 0 写 |

> 💡 **关键洞察**：GEMV/SpMV 的并行度在"行间"（M 行并行）+ "行内"（warp 协作扫列）。一个 warp 一行是标准映射——既利用行间并行，又用 warp shuffle 做行内归约。CSR 版的区别是扫 `row_ptr` 区间而非全列，访存不规则（`col_idx` 间接寻址）。

## 5. 性能分析与优化

### 5.1 编译运行

```bash
nvcc -O3 -arch=sm_120 cuda_spmv.cu -o spmv -lineinfo
./spmv 1000 10000
```

### 5.2 瓶颈分析

- Dense GEMV：`2·M·N` FLOP / `(M·N + N + M)·4B` ≈ `0.5 FLOP/Byte`，memory-bound；
- CSR SpMV：更极端——`nnz` 个非零元读 `values+col_idx`（8B/非零元）只做 2 FLOP，`0.25 FLOP/Byte`，且间接寻址 `x[col_idx]` 不规则。

### 5.3 CSR 优化方向

1. **ELLPACK**：每行非零元数相近时用 ELLPACK（coalesced 访问）；
2. **hybrid CSR/ELL**：取长补短；
3. **warp 内 thread 数自适应**：行长差异大时按行 nnz 分配 1/2/4... thread。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M·N)`（dense）/ `O(nnz)`（CSR） |
| **算术强度** | `~0.5 FLOP/Byte`（dense），memory-bound |
| **瓶颈类型** | **memory-bound**；CSR 因间接寻址更甚 |
| **warp 归约** | 每行 `log₂32 = 5` 步 |

> 💡 **一句话总结**：SpMV = "一个 warp 一行 + warp shuffle 行内归约"，是稀疏计算的标准骨架。本题实为 dense GEMV（参考用 dense matmul），但面试标准版是 CSR——掌握 warp 行归约 + 不规则访存处理是核心。

## 面试考点

- **手撕要求**：默写 warp 行归约 GEMV（一个 warp 一行，warp shuffle 归约）+ CSR SpMV 变体。
- **高频追问**：
  - **为什么一个 warp 一行而不是一个 thread？** 行长可能大，单 thread 串行慢；warp 协作 + shuffle 归约把行内点积并行化。
  - **CSR 的访存为什么不规则？** `x[col_idx[i]]` 是间接寻址，`col_idx` 不连续 → `x` 的访问不 coalesced，cache 命中率低。
  - **CSR vs ELLPACK？** CSR 行长可变、灵活但访存不规则；ELLPACK 对齐到最长行、coalesced 但补零浪费算力。行长相近用 ELL，差异大用 CSR。
  - **本题为什么用 dense？** 参考实现是 dense matmul，`nnz` 未用——题目名误导，解题按 dense GEMV 即可。
- **进阶延伸**：稀疏 attention（block-sparse）、MoE 路由（per-token 选 expert 的稀疏 matmul）都是 SpMV 的变体。cuSPARSE 提供生产级 SpMV。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | warp shuffle 归约，SpMV 行内归约的基础组件 |
| 75 | [Sparse Matrix-Dense Matrix Multiplication](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication) | 困难 | 稀疏 GEMM，SpMV 的矩阵版进阶 |
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) | 中等 | 稠密 GEMM tiling，对比稀疏 vs 稠密访存模式 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | 树形归约，SpMV 行内归约的基础组件 |

> 💡 **选题思路**：CSR 稀疏格式 + warp shuffle 行内归约，练习不规则访存与稀疏矩阵乘模板。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
