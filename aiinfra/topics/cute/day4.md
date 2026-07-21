# Day 4（周四）：Copy 原语体系

> **今日目标**：掌握 `cute::copy` 的调度机制——它如何根据源/目标 Layout 自动选择向量化宽度、`cp.async`、TMA；能手动构建 3-stage 流水线
> **面试考察度**：⭐⭐⭐⭐ 实践级，能解释 copy 原语的自动调优策略

---

### 学习任务 1：copy 的调度机制（45 分钟）

#### copy 不是 memcpy

`cute::copy(src_tensor, dst_tensor)` 表面像 memcpy，实际是一个**编译期 auto-tuning** 的搬运调度器：

```cpp
// 源码：copy.hpp
template <typename SrcEngine, typename SrcLayout,
          typename DstEngine, typename DstLayout>
void copy(Tensor<SrcEngine, SrcLayout> const& src,
          Tensor<DstEngine, DstLayout> const& dst);
```

它会检查源/目标的 Layout 与 Engine，按以下优先级选择搬运策略：

| 源 Engine | 目标 Engine | 自动选择的策略 | 触发条件 |
|-----------|-------------|----------------|----------|
| gmem | smem | `cp.async`（Ampere+） | arch >= sm_80 |
| gmem | smem | TMA（Hopper+） | arch >= sm_90 且 Layout 匹配 TMA 要求 |
| smem | rmem | 向量化 `lds`（float4 等） | coalesce 后连续 |
| rmem | smem | 向量化 `sts` | 同上 |
| smem | smem | register 中转转置 | 源/目标布局不同 |

#### 向量化宽度的自动选择

`copy` 会根据 `coalesce` 后的连续长度选择最大向量化宽度：

```cpp
auto L = make_layout(make_shape(128), make_stride(1));       // 连续 128
auto L8 = make_layout(make_shape(8), make_stride(1));        // 连续 8
auto Ls = make_layout(make_shape(128), make_stride(2));      // stride=2，不连续

// copy 自动选择：
// L  → 一次 128-byte 向量化（4 个 float4）
// L8 → 一次 8-element（2 个 float4）
// Ls → 逐元素（无法向量化）
```

> 💡 **关键洞察**：`copy` 的性能完全取决于你给的 Layout——你不用写 `float4`/`cp.async`，只要 Layout 连续，它自动用最优指令。这也是为什么 Day 2 的 `coalesce` 这么重要：它是 `copy` 选择策略的依据。

### 学习任务 2：cp.async 与多阶段流水线（60 分钟）

#### 手动构建 3-stage 流水线

Ampere 的 `cp.async` 让 gmem→smem 搬运与计算重叠。CuTe 用 `copy` + barrier 封装：

```cpp
// 3-stage 流水线：3 个 smem buffer，2 个在做计算，1 个在加载
__shared__ float smem_A[3][128 * 64];   // 3 个 buffer
__shared__ cuda::barrier<cuda::thread_scope_block> bar[3];

// 初始化 barrier（每个 buffer 一个）
if (threadIdx.x == 0) {
    for (int i = 0; i < 3; ++i)
        init(&bar[i], blockDim.x);
}
__syncthreads();

// Prologue：先发起 2 次 cp.async
auto gA0 = make_tensor(gmem_ptr_A + 0 * 128 * 64, A_tile_layout);
auto sA0 = make_tensor(make_smem_ptr(smem_A[0]), A_smem_layout);
copy(gA0, sA0);
arrive(bar[0]);

auto gA1 = make_tensor(gmem_ptr_A + 1 * 128 * 64, A_tile_layout);
auto sA1 = make_tensor(make_smem_ptr(smem_A[1]), A_smem_layout);
copy(gA1, sA1);
arrive(bar[1]);

// 主循环：计算 buffer i，加载 buffer i+2
for (int k = 0; k < K_tiles; ++k) {
    int stage = k % 3;
    int load_stage = (k + 2) % 3;

    wait(bar[stage]);  // 等加载完成
    // ... 用 sA[stage] 做 MMA ...
    arrive(bar[stage]);  // 释放 buffer

    if (k + 2 < K_tiles) {
        auto gA_next = make_tensor(gmem_ptr_A + (k + 2) * 128 * 64, A_tile_layout);
        auto sA_next = make_tensor(make_smem_ptr(smem_A[load_stage]), A_smem_layout);
        copy(gA_next, sA_next);
        arrive(bar[load_stage]);
    }
}
```

> ⚠️ **注意**：上面是手写的 3-stage 流水线，只为理解原理。实际工程中用 `cute::TensorPipeline`（`tensor_pipeline.hpp`）或 CUTLASS 的 `MainloopPipeline` 封装，避免手写 barrier 管理出错。

### 学习任务 3：copy_if 与边界处理（30 分钟）

```cpp
// 带谓词的 copy：处理 K 不整除 tile 的尾段
copy_if(gA_tile, sA_tile, [&](auto coord) {
    return k_idx < K_total;  // 只拷有效元素
});

// 等价的朴素的 boundary check
for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
    if (i < remaining) smem[i] = gmem[i];
}
```

| 维度 | 朴素边界 | copy_if |
|------|----------|---------|
| 代码量 | 手写循环 + 分支 | 一行 |
| 性能 | 分支预测失败 | 向量化谓词 |
| 可读性 | 差 | 好 |

### 学习任务 4：动手实验（30 分钟）

创建 `kernels/cute_copy_pipeline.cu`，用 `copy` + barrier 实现 3-stage 流水线搬运，对比朴素逐元素搬运的带宽：

```cpp
// cute_copy_pipeline.cu —— copy 原语与 3-stage 流水线
// 编译: nvcc -o cute_copy_pipeline cute_copy_pipeline.cu \
//        -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17 -lcuda
#include <cute/tensor.hpp>
#include <cuda/barrier>
#include <cuda_runtime.h>
using namespace cute;

// TODO: 实现 3-stage 流水线搬运 128×4096 矩阵 tile
//       对比 copy() vs 朴素 for 循环的带宽
//       用 cudaEvent 计时，输出 GB/s

int main() {
    // 1. 分配 gmem 矩阵
    // 2. 跑朴素搬运，计时
    // 3. 跑 3-stage 流水线，计时
    // 4. 对比带宽
    return 0;
}
```

### 今日检查清单
- [ ] 能说出 `copy` 根据源/目标 Engine 自动选择的 5 种策略
- [ ] 能解释 `coalesce` 如何决定向量化宽度
- [ ] 能手写 3-stage 流水线（cp.async + barrier）
- [ ] 理解 `copy_if` 相比朴素边界处理的优势
- [ ] `cute_copy_pipeline.cu` 跑通，流水线版带宽显著高于朴素版

---

