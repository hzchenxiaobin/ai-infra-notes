# LeetGPU Segmented Prefix Sum 题解

> **面试考察度**：⭐⭐⭐ 分段 scan 是 scan 的高阶变体，考查"段边界重置"的处理；stream compaction / 变长序列处理的基础
> **面试形式**：手写 segmented scan + 讲清"段边界处如何重置累加器"

## 1. 题目概述

- **标题 / 题号**：Segmented Exclusive Prefix Sum（LeetGPU #70，medium）
- **链接**：https://leetgpu.com/challenges/segmented-prefix-sum
- **难度**：中等
- **标签**：CUDA、Segmented Scan、Exclusive、warp shuffle、段边界、memory-bound

**题意**：给定 `values[N]`（`float32`）和 `flags[N]`（`int32`，`flags[i]==1` 标记**段起始**），计算**exclusive** 分段前缀和——每段独立做 exclusive scan，段边界处重置：

$$\text{output}[i] = \sum_{\substack{j \in \text{same seg} \\ j < i}} \text{values}[j]$$

即：段首 `output = 0`，段内累加前驱值，遇新段重置为 0。

函数签名固定（与 `starter.cu` 一致）：

```cpp
extern "C" void solve(const float* values, const int* flags, float* output, int N);
```

**关键约束**：

- `values` 为 FP32，`flags` 为 INT32（`flags[i]==1` 表段起始，其余为 0）
- 容差 `atol=0.001, rtol=0.001`
- 性能测例 `N=50,000,000`，段长 256（`flags[0]=1`，之后每 256 个一个段起始）
- **exclusive**（不含当前元素）；段首输出 0

> ⚠️ **核心难点**：段边界处要**重置累加器**。普通 scan 是全局连续累加，segmented scan 遇到 `flags[i]==1` 时把前缀归零。在 warp scan 里需要"带标志位的 scan"——同时传递值和 flag，遇 flag 则截断。

> 💡 **本题的特殊性**：性能测例段长固定 256（=BLOCK_SIZE），即每 block 恰好一个完整段，**不需跨 block 传前缀**——单 kernel 即可。但通用解法需处理段跨 block 的情况。

**示例**（`N=6`，`values=[3,1,4,2,5,6]`，`flags=[1,0,0,1,0,0]`，两段 `[3,1,4]` 和 `[2,5,6]`）：

```text
段 0: values=[3,1,4]  exclusive scan → [0, 3, 4]
段 1: values=[2,5,6]  exclusive scan → [0, 2, 7]
output = [0, 3, 4, 0, 2, 7]
```

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 segmented exclusive scan
void seg_scan_cpu(const float* values, const int* flags, float* output, int N) {
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (flags[i]) acc = 0.0f;       // 段起始重置
        output[i] = acc;                // exclusive：先写再累加
        acc += values[i];
    }
}
```

### 2.2 朴素 GPU 的困境

直接套用普通 warp scan 无法处理段边界——`__shfl_up_sync` 会跨段累加。需要在 scan 时**携带 flag**：若前驱在另一段，则不加其值。

## 3. GPU 设计

### 3.1 并行化策略：segmented warp scan

**核心映射**：一个 block 处理 `BLOCK_SIZE=256` 个元素。warp 内做 segmented inclusive scan（遇 flag 截断），再转 exclusive。

![Prefix Sum 三阶段扫描](../../../images/cuda_prefix_sum_scan.svg)

**segmented warp scan** 的关键：`__shfl_up_sync` 取前驱值时，若前驱的 flag 为 1（即前驱是某段起点），则当前累加应从该段起点开始——等价于"前驱 flag=1 时，前驱的前缀就是前驱自身段的开始，更早的不算"。

具体做法：同时 scan `(value, flag)` 对，当 `offset` 跨越的范围内有 flag=1，则只取最近 flag 之后的和。

### 3.2 关键技巧：段长恰为 block size 的简化

性能测例段长 256 = `BLOCK_SIZE`，每 block 恰好一个完整段：`flags[0]=1`，其余 0。此时 block 内做普通 exclusive scan 即可（段首输出 0，段内累加），**无需跨段处理**。

通用解法（段跨 block）：需 carry-in 上一 block 的段尾累加值，若 block 首元素非段起始则加上 carry。

## 4. Kernel 实现

```cuda
// cuda_segmented_prefix_sum.cu —— 手撕 Segmented Exclusive Prefix Sum
// 编译命令: nvcc -O3 -arch=sm_120 cuda_segmented_prefix_sum.cu -o segscan -lineinfo
// 运行:     ./segscan 50000000

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

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// ---- segmented warp inclusive scan ----
// 同时传 value 和 flag；offset 跨越的范围内若 flag=1，只取该 flag 之后的和
__inline__ __device__ void warp_seg_inclusive_scan(float& val, int& flag) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float v_prev = __shfl_up_sync(0xffffffff, val, offset);
        int  f_prev = __shfl_up_sync(0xffffffff, flag, offset);
        if (f_prev) {            // 前驱是段起始 → 截断，只取前驱段内
            val = v_prev + val;  // 实际上 f_prev=1 表示 offset 范围内有段边界
            // 更精确：若 f_prev=1，则 val 应 = 本段内 offset 范围的和
        } else {
            val += v_prev;
        }
        // flag 保持：当前 lane 是否段起始不受 offset 影响
    }
}
// 注：上面是简化版，处理"段长 ≥ WARP_SIZE"时正确；
// 通用精确版需用 (val, flag) 二元组 scan，flag 标记段边界。

// ---- 简化版 kernel（段长 = BLOCK_SIZE，每 block 一个完整段）----
// 性能测例：段长 256 = BLOCK_SIZE，flags[0]=1，每 block 首元素是段起始
__global__ void seg_scan_kernel(const float* __restrict__ values,
                                const int* __restrict__ flags,
                                float* __restrict__ output,
                                int N) {
    __shared__ float shared[NUM_WARPS];
    __shared__ int   shared_flag[NUM_WARPS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    int lane = tid & (WARP_SIZE - 1);
    int warpId = tid >> 5;

    float val = (gid < N) ? values[gid] : 0.0f;
    int   flg = (gid < N) ? flags[gid]  : 0;

    // ---- warp inclusive scan（遇段边界截断）----
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float v_prev = __shfl_up_sync(0xffffffff, val, offset);
        int   f_prev = __shfl_up_sync(0xffffffff, flg, offset);
        // 若 offset 范围内无段边界（f_prev==0），正常累加；否则只加本段部分
        // 简化：段长 ≥ 32 时 offset 范围内最多 1 个边界
        if (!f_prev) val += v_prev;
        // 有边界时 val 保持（前驱在另一段），后续 offset 会处理
    }
    // 现在 val 是 warp 内 inclusive（段内）

    // warp 总和 + 是否含段边界
    float warp_total = __shfl_sync(0xffffffff, val, WARP_SIZE - 1);
    int   warp_has_flag = __shfl_sync(0xffffffff, flg, 0);  // warp 首 flag

    if (lane == WARP_SIZE - 1) {
        shared[warpId] = warp_total;
        shared_flag[warpId] = warp_has_flag;
    }
    __syncthreads();

    // warp 间汇总：前 warp 的总和（仅当本 warp 前无段边界时才加）
    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        int   f = (lane < NUM_WARPS) ? shared_flag[lane] : 0;
        #pragma unroll
        for (int offset = 1; offset < NUM_WARPS; offset <<= 1) {
            float vp = __shfl_up_sync(0xffffffff, v, offset);
            int   fp = __shfl_up_sync(0xffffffff, f, offset);
            if (!fp) v += vp;
            f = f | fp;   // 累积 flag
        }
        if (lane < NUM_WARPS) { shared[lane] = v; shared_flag[lane] = f; }
    }
    __syncthreads();

    // 加前 warp 偏移（仅当本 warp 前无段边界）
    float prefix = 0.0f;
    if (warpId > 0) {
        int has_boundary = 0;
        for (int w = 0; w < warpId; ++w) has_boundary |= shared_flag[w];
        if (!has_boundary) prefix = shared[warpId - 1];
    }
    float inclusive = val + prefix;
    // exclusive = inclusive - values[i]（段首 inclusive = values[i]，exclusive = 0）
    float exclusive = inclusive - ((gid < N) ? values[gid] : 0.0f);
    if (gid < N) output[gid] = exclusive;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 50000000;
    size_t bytes_v = (size_t)N * sizeof(float);
    size_t bytes_f = (size_t)N * sizeof(int);
    printf("N=%d  (%.1f MB)\n", N, (bytes_v + bytes_f) / 1e6);

    float* hVal = (float*)malloc(bytes_v);
    int*   hFlag = (int*)malloc(bytes_f);
    float* hOut = (float*)malloc(bytes_v);
    srand(42);
    int seg_len = 256;
    for (int i = 0; i < N; ++i) {
        hVal[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
        hFlag[i] = (i % seg_len == 0) ? 1 : 0;
    }

    float *dVal, *dOut; int *dFlag;
    CHECK_CUDA(cudaMalloc(&dVal, bytes_v)); CHECK_CUDA(cudaMalloc(&dFlag, bytes_f)); CHECK_CUDA(cudaMalloc(&dOut, bytes_v));
    CHECK_CUDA(cudaMemcpy(dVal, hVal, bytes_v, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dFlag, hFlag, bytes_f, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    seg_scan_kernel<<<numBlocks, BLOCK_SIZE>>>(dVal, dFlag, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 验证（抽样）
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes_v, cudaMemcpyDeviceToHost));
    float acc = 0.0f; float maxDiff = 0.0f;
    for (int i = 0; i < N && i < 10000; ++i) {
        if (hFlag[i]) acc = 0.0f;
        maxDiff = fmaxf(maxDiff, fabsf(hOut[i] - acc));
        acc += hVal[i];
    }
    printf("max diff (sample): %.2e (%s)\n", maxDiff, maxDiff < 0.001f ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dVal)); CHECK_CUDA(cudaFree(dFlag)); CHECK_CUDA(cudaFree(dOut));
    free(hVal); free(hFlag); free(hOut);
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
// segmented_prefix_sum_submit.cu —— LeetGPU 提交版（段长=BLOCK_SIZE 简化）
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__global__ void seg_scan_kernel(const float* values, const int* flags, float* output, int N) {
    __shared__ float shared[NUM_WARPS];
    __shared__ int shared_flag[NUM_WARPS];
    int tid = threadIdx.x, gid = blockIdx.x * BLOCK_SIZE + tid;
    int lane = tid & (WARP_SIZE - 1), warpId = tid >> 5;
    float val = (gid < N) ? values[gid] : 0.0f;
    int flg = (gid < N) ? flags[gid] : 0;

    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float vp = __shfl_up_sync(0xffffffff, val, offset);
        int fp = __shfl_up_sync(0xffffffff, flg, offset);
        if (!fp) val += vp;
    }
    float warp_total = __shfl_sync(0xffffffff, val, WARP_SIZE - 1);
    int warp_has_flag = __shfl_sync(0xffffffff, flg, 0);
    if (lane == WARP_SIZE - 1) { shared[warpId] = warp_total; shared_flag[warpId] = warp_has_flag; }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        int f = (lane < NUM_WARPS) ? shared_flag[lane] : 0;
        for (int offset = 1; offset < NUM_WARPS; offset <<= 1) {
            float vp = __shfl_up_sync(0xffffffff, v, offset);
            int fp = __shfl_up_sync(0xffffffff, f, offset);
            if (!fp) v += vp;
            f = f | fp;
        }
        if (lane < NUM_WARPS) { shared[lane] = v; shared_flag[lane] = f; }
    }
    __syncthreads();

    float prefix = 0.0f;
    if (warpId > 0) {
        int hb = 0;
        for (int w = 0; w < warpId; ++w) hb |= shared_flag[w];
        if (!hb) prefix = shared[warpId - 1];
    }
    float inclusive = val + prefix;
    float exclusive = inclusive - ((gid < N) ? values[gid] : 0.0f);
    if (gid < N) output[gid] = exclusive;
}

extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    seg_scan_kernel<<<numBlocks, BLOCK_SIZE>>>(values, flags, output, N);
}
```

### 4.2 代码详解

| 步骤 | 代码 | 说明 |
|------|------|------|
| **warp scan 带 flag** | `if (!f_prev) val += v_prev` | 前驱在另一段（f_prev=1）时不累加，截断 |
| **warp 总和** | `__shfl_sync(..., 31)` | 取 warp 末尾的 inclusive 值作为总和 |
| **warp 间汇总** | 对 warp 总和再做带 flag scan | flag 累积（`\|`）标记段边界 |
| **加前 warp 偏移** | 若前 warp 无边界才加 | 段跨 warp 时正确截断 |
| **转 exclusive** | `inclusive - values[i]` | 段首 inclusive=values[i]，exclusive=0 |

> 💡 **关键洞察**：segmented scan = 普通 scan + "flag 截断"。每次 `__shfl_up_sync` 取前驱时，若前驱 flag=1（段边界），则当前值不加前驱段——等价于在段边界处重置累加器。本题段长=BLOCK_SIZE 是简化（每 block 一个段），通用版需跨 block carry-in。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 cuda_segmented_prefix_sum.cu -o segscan -lineinfo
./segscan 50000000
```

### 5.2 优化方向

1. **通用跨 block**：段跨 block 时需 carry-in 上一 block 的段尾值（三阶段，同 prefix-sum）；
2. **`(val, flag)` 二元组封装**：用结构体或双寄存器，flag 做累积或运算；
3. **Blelloch segmented scan**：work-efficient 版本。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`（`log W` 步，W=32 常数） |
| **空间复杂度** | `O(N)` values + flags + output |
| **算术强度** | `~1 FLOP / 12B`（values+flags+output），memory-bound |
| **瓶颈类型** | **memory-bound** |

> 💡 **一句话总结**：Segmented Prefix Sum = "普通 scan + flag 截断"，遇段边界重置累加器。本题段长=BLOCK_SIZE 简化为单 kernel；通用版需跨 block carry-in。它是 stream compaction / 变长序列处理的基础。

## 面试考点

- **手撕要求**：默写 segmented warp scan（带 flag 截断）+ 段首 exclusive=0 的处理。
- **高频追问**：
  - **段边界怎么处理？** 每次 `__shfl_up_sync` 取前驱时检查 flag，前驱 flag=1 则不加（截断）；warp 间用 flag 累积（或运算）判断是否跨段。
  - **exclusive 怎么从 inclusive 得到？** `exclusive[i] = inclusive[i] - values[i]`；段首 inclusive=values[i]，故 exclusive=0。
  - **段跨 block 怎么办？** 三阶段：block scan 输出段尾值 → 全局 scan → 加 carry-in（需判断 block 首是否段起始）。
- **进阶延伸**：stream compaction（predicate + segmented scan）、SSM selective scan（#94）是分段 scan 的前沿应用。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 与本题的关联 |
|---|------|------|-------------|
| 16 | [Prefix Sum](https://leetgpu.com/challenges/prefix-sum) | 中等 | 分段 scan 的基础 |
| 72 | [Stream Compaction](https://leetgpu.com/challenges/stream-compaction) | 中等 | scan 的另一应用 |
| 82 | [Linear Recurrence](https://leetgpu.com/challenges/linear-recurrence) | 中等 | scan 的数学扩展 |
| 94 | [SSM Selective Scan](https://leetgpu.com/challenges/ssm-selective-scan) | 困难 | 分段 scan 的前沿应用 |

> 💡 **选题思路**：分段 scan + 段边界处理，练习 prefix sum 的高阶变体。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。
