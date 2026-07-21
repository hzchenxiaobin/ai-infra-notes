# Day 7（周日）：CuTe GEMM 实战与面试复盘

> **今日目标**：用本周学的 CuTe 原语（Layout + Tensor + copy + Swizzle + MMA）从零组装一个 GEMM，对标 `examples/cute/tutorial`，完成面试复盘
> **面试考察度**：⭐⭐⭐⭐⭐ 综合应用，能独立用 CuTe 原语写 kernel

---

### 学习任务 1：阅读官方 CuTe Tutorial GEMM（60 分钟）

CUTLASS 仓库的 `examples/cute/` 目录有一组递进式 GEMM 教程，是本周的对标目标：

| 示例 | 内容 | 本周对应 |
|------|------|----------|
| `00_hello_gemm` | 最朴素 CuTe GEMM | Day 1-3 综合 |
| `01_gemm_tiled` | 引入 tiling 与 partition | Day 3 |
| `02_gemm_swizzle` | 加 Swizzle 消除 bank conflict | Day 5 |
| `03_gemm_pipeline` | 加 cp.async 流水线 | Day 4 |
| `04_gemm_mma` | 用 MMA atom 替换朴素计算 | Day 3 rmem |
| `05_hopper_gemm` | TMA + WGMMA + warp specialization | Day 6 |

精读 `00_hello_gemm.cu`，对照本周学的原语：
- `make_layout` 构造 gmem/smem Layout
- `make_tensor` 绑定数据
- `local_partition` 切给 thread
- `copy` 搬运 gmem→smem
- `tensor(i,j)` 访问元素累加

### 学习任务 2：手写 CuTe GEMM（90 分钟）

参照 `00_hello_gemm`，用本周原语实现一个 FP16 GEMM，目标达到 cuBLAS 70%+：

```cpp
// cute_gemm.cu —— 用 CuTe 原语组装 GEMM（不用 CollectiveBuilder）
// 编译: nvcc -o cute_gemm cute_gemm.cu \
//        -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17 -lcublas
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

template <int TileM, int TileN, int TileK>
__global__ void cute_gemm_kernel(
    half_t const* A, half_t const* B, half_t* C,
    int M, int N, int K)
{
    // 1. 定义 Layout
    auto gmem_layout = make_layout(make_shape(M, N), make_stride(N, 1));
    auto smem_layout_A = make_layout(make_shape(TileM, TileK), make_stride(TileK, 1),
                                     Swizzle<3, 4, 3>{});
    // ... B, C 类似 ...

    // 2. 创建 Tensor
    auto gA = make_tensor(A + blockIdx.y * TileM * K, /* A tile layout */);
    auto sA = make_tensor(make_smem_ptr(smem_A), smem_layout_A);

    // 3. 用 copy 搬运 gmem → smem（自动 cp.async + swizzle）
    copy(gA, sA);

    // 4. 用 local_partition 切给 warp/thread
    auto tA = local_partition(sA, thread_layout, threadIdx.x);

    // 5. 用 MMA atom 计算
    //    auto mma = make_mma_atom<SM80_16x8x16F16F16F16F16_TN>();
    //    mma(tC_accum, tA, tB, tC_accum);

    // 6. 写回 gmem
    copy(gC, sC);
}
```

> ⚠️ **提示**：完整实现较长（~200 行），重点是体会"原语组装"的过程，不必追求极致性能。Day 6 的 TMA + WGMMA 是进阶，本周做到 cp.async + Ampere MMA 即可。

### 学习任务 3：性能对比与 ncu 分析（60 分钟）

```bash
# 对比：朴素 GEMM vs CuTe GEMM vs CUTLASS vs cuBLAS
ncu --set full --kernel-name "cute_gemm" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              dram__throughput.avg.pct_of_peak_sustained_elapsed,\
              smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./cute_gemm
```

| 指标 | 目标 | 含义 |
|------|------|------|
| `sm__throughput` | > 60% | SM 利用率 |
| `dram__throughput` | < 40% | GEMM 应计算 bound |
| `...tensor_op_hmma` | > 50% | Tensor Core 利用率 |
| `l1tex__data_bank_conflicts` | 接近 0 | Swizzle 是否生效 |

### 学习任务 4：面试题复盘（60 分钟）

#### CuTe 高频面试题

1. **CuTe 的 Layout 是什么？它和普通的 shape/stride 对有什么区别？**
   - Layout 是 `Coord → offset` 的整数函数，由 Shape + Stride（+ 可选 Swizzle）定义
   - 区别：CuTe Layout 支持嵌套、代数运算（coalesce/compose/tiled），swizzle 是 Layout 的一部分

2. **什么是** `coalesce`**？它有什么用？**
   - 把多维 Layout 化简为等价 1D，条件是相邻逻辑坐标对应相邻物理偏移
   - 用途：判断能否向量化加载，是 `copy` 选择搬运策略的依据

3. **CuTe 的 Tensor 为什么是双模板** `Tensor<Engine, Layout>`**？**
   - Engine 决定数据在哪（gmem/smem/rmem），Layout 决定怎么访问
   - 分离让同一 Layout 复用于不同存储层级，访问语法 `tensor(i,j)` 统一

4. `local_partition` **和** `slice` **的区别？**
   - `slice` 取固定坐标的子 Tensor（如某行某列）
   - `local_partition` 按线程/warp 的逻辑布局切分，返回"当前线程/warp 拿到的那一份"

5. **CuTe 的** `copy` **如何自动选择搬运策略？**
   - 根据源/目标 Engine（gmem/smem/rmem）和 Layout 的 coalesce 结果
   - gmem→smem 自动用 cp.async（sm_80+）或 TMA（sm_90+），向量化宽度由连续长度决定

6. **Swizzle 在 CuTe 中是什么？为什么说"Swizzle 就是 Layout"？**
   - Swizzle 是 smem 地址的 XOR 变换，消除 bank conflict
   - CuTe 把它作为 Layout 的第三参数 `Layout<Shape, Stride, Swizzle>`，搬运代码自动应用，无需手改

7. **Swizzle 是幂等的吗？这意味着什么？**
   - 是，`swizzle(swizzle(x)) = x`
   - 意味着写入时 swizzle 与读取时反 swizzle 是同一操作，`copy` 双向都能自动处理

8. **TMA 相比 cp.async 有什么优势？**
   - 块级粒度（一条指令搬整个 tile）、硬件地址计算、支持多播、不占 SM
   - 是 Hopper WGMMA 的前提——TMA 负责 gmem→smem，WGMMA 直接从 smem 读

9. **什么是 warp specialization？**
   - 不同 warp 干不同事：producer warp 发 TMA，consumer warpgroup 做 WGMMA
   - 把搬数据与算数据彻底分离，流水线深度可达 4-8 stage

10. **CuTe 如何表达 MMA fragment？**
    - 用 `Tensor<Engine=rmem, Layout>` 表示 register 数组上的 fragment
    - 与 gmem/smem Tensor 接口统一，可 slice/partition，CUTLASS 2.x 的 `wmma::fragment` 布局不透明

11. **CuTe 与 CUTLASS 2.x 的索引方式对比？**
    - 2.x：每种布局特化一个迭代器类，模板参数爆炸
    - CuTe：Layout 是一等公民，`tensor(i,j)` 通用，可任意嵌套/合成

12. **为什么 CuTe 能脱离 CUTLASS GEMM 模板独立使用？**
    - Layout/Tensor/Copy 三个抽象与 GEMM 无关，任何"分块+搬运+计算"的 kernel 都能用
    - CUTLASS 3.x GEMM 只是用 CuTe 原语拼出的一个特例

### 学习任务 5：总结与知识图谱（30 分钟）

#### 本周知识图谱

```
                  CuTe（CUTLASS Tensor Engine）
                       /          \
              Layout 代数        Tensor 引擎
              /  |  \             /    \
        coalesce compose   gmem/smem/rmem
              \  |  /             \    /
            嵌套 Shape          local_partition
                 |                   |
              Swizzle              copy 原语
              (Layout 第三参数)    /  |  \
                 |             cp.async TMA  向量化
              bank conflict       \  |  /
                 |              流水线
              MMA 友好              |
                 \            warp specialization
                  \              /
                   CuTe GEMM 实战
                        |
              对标 examples/cute/tutorial
```

#### 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [CuTe 源码](https://github.com/NVIDIA/cutlass/tree/main/include/cute) `include/cute/` | 源码 | ⭐ 必读 |
| [CuTe Tutorial 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) `examples/cute/` | 示例 | ⭐ 必读 |
| [CUTLASS 3.0 发布博客](https://developer.nvidia.com/blog/cutlass-3-0/) CuTe 一节 | 博客 | ⭐ 必读 |
| [GTC 2023 CuTe 演讲](https://developer.nvidia.com/gtc/2023/video/s40095) | 视频 | 📌 推荐 |
| [CuTe Slack #cutlass 频道](https://nvidia-ai-infra.slack.com/) | 社区 | 📌 推荐 |
| [CUTLASS 专题 Day 2](../cutlass/day2.md) CuTe 入门 | 教程 | 📎 复习前置 |

