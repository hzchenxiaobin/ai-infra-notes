# 第 1 周 · 第 3 天：学习 Tiling 与数据复用

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 1 周"GEMM 与 Ascend 基础"的第 3 天教程，可独立阅读，不依赖前序内容。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 解释 `BLOCK_M` / `BLOCK_N` / `BLOCK_K` 分别切 `D = A × B` 的哪个维度，并写出输出 Tile 数（`ceil(M/BLOCK_M) × ceil(N/BLOCK_N)`）和 K 维循环次数（`ceil(K/BLOCK_K)`）的计算公式；
2. 手工切分一个 `M=128、N=128、K=256` 的 GEMM，列出每个输出 Tile 需要读取的 A/B 数据区域和数量；
3. 推导"无复用总读取量"与"完美复用总读取量"的表达式，说明 A 的复用因子是 N 方向 Tile 数、B 的复用因子是 M 方向 Tile 数；
4. 计算单个输出 Tile 的算术强度 `2·BM·BN / ((BM+BN)·s)`，解释为什么 Tile 越大每字节数据支撑的计算越多、但 Tile 大小受 L0C/L1 容量硬约束；
5. 在仓库源码中定位 BLOCK 参数的定义与传递路径（`GemmConfig` → JIT 模板 → kernel → scheduler），并指出代码里体现"数据复用"的至少三处机制（L0C 累加器、L2 调度 swizzle 与 L2 hint、resident A/B）。

## 二、预备知识

- **第 1 天的内容**：`2MNK` 计算量、数据量与算术强度的定义；片上容量常量（L1 每核 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB，见 `deep_gemm/include/deep_gemm/common.hpp` 第 54～58 行）。3.1 节会简要回顾。
- **第 2 天的内容**：Major 与 stride；仓库以 `[N, K]` 视角描述 B（`nt` 约定）。今天说"B 的第 j 行块"时一律指 N 方向的行、沿 K 展开的数据。
- **Python 基础（可选）**：实践脚本只用循环和整除运算，CPU 即可运行；没有 Python 时可用纸笔完成等价演算。
- **不需要 Ascend 硬件**：今天全部内容（概念、公式、手工切分、源码阅读）都不依赖 NPU 环境。

背景一句话：第 1 天已经证明，真实 GEMM 的输入输出放不进片上存储，性能关键在于"让每份数据被尽可能多的计算复用"。今天回答"怎么复用"——把大矩阵切成小块（Tiling），让一个输出 Tile 所需的输入恰好装进片上缓冲，并通过 Tile 间的共享把 GM 访问降到最低。仓库 README 第 5 行说 DeepGEMM-Ascend 替使用者"隐藏 fractal 布局、对齐约束、地址计算等复杂性"，Tiling 正是这些被隐藏的复杂性中最核心的一环。

## 三、详细讲解

### 3.1 从"放不下"到"分块流式"：Tiling 的基本思想

回顾第 1 天的结论：以 `(4096, 7168, 16384)` BF16 为例，A+B+D 共 408 MiB，而每核 L1 只有 512 KiB（差数百倍），且还要切成多级流水 stage。所以 kernel 只能"流式"工作：

```text
每次只在片上放：A 的一小块 + B 的一小块 + 输出的一小块
算完一块输出 → 换下一块 → 直到整个 D 完成
```

"块"有三个方向的大小，就是 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`。它们在仓库里是**编译期常量**——kernel 模板参数（`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 第 12 行）和 host 侧配置（`GemmConfig` 的 `block_m/block_n/block_k`，`csrc/jit_kernels/gemm_config.hpp` 第 100 行）。编译期确定意味着 L1/L0 缓冲的布局、地址偏移都可以静态算好，这也是 JIT 为每个 shape 生成专用代码的原因之一（第 12 天展开）。

### 3.2 BLOCK_M / BLOCK_N / BLOCK_K 分别切什么

先看输出侧。D 是 `[M, N]`，把它按 `BLOCK_M × BLOCK_N` 切成网格：

```text
num_m_blocks = ceil(M / BLOCK_M)      // M 方向的行块数
num_n_blocks = ceil(N / BLOCK_N)      // N 方向的列块数
输出 Tile 总数 = num_m_blocks × num_n_blocks
```

第 `(i, j)` 个输出 Tile 负责计算 `D[i·BM : (i+1)·BM, j·BN : (j+1)·BN]`。**M/N 方向的 Tile 彼此独立**，可以分配给任何核并行；这正是调度器只按 M×N 网格发任务的原因（`scheduler.hpp` 第 93 行 `num_blocks = ceil_div(shape_m, BLOCK_M) * ceil_div(shape_n, BLOCK_N)`）。

再看收缩维。一个输出 Tile 的数学定义是沿 K 全长做内积：

```text
D_tile = A 行块 × B 行块（沿 K 全长）
```

但 K 全长的 A/B 行块同样可能太大（例如 `BLOCK_M=128、K=16384` 的 A 行块要 4 MiB），所以 K 也要切：kernel 把 K 循环切成 `num_k_blocks = ceil(K / BLOCK_K)` 步，每步只把 `BLOCK_M × BLOCK_K` 的 A 片段和 `BLOCK_N × BLOCK_K` 的 B 片段搬进 L1，**在 L0C 累加器里逐步累加**，走完 K 循环才得到最终 Tile。kernel 源码（`bf16_gemm.hpp` 第 96、112～114 行）：

```cpp
const uint32_t num_k_blocks = ceil_div(eff_shape_k, BLOCK_K);
...
for (uint32_t k_blk_idx = 0, s = 0; k_blk_idx < num_k_blocks; k_blk_idx++, s = ...) {
    const uint32_t k_idx = k_idx_base + k_blk_idx * BLOCK_K;
    const uint32_t actual_k = min(eff_shape_k - k_blk_idx * BLOCK_K, BLOCK_K);
```

**为什么 K 维不能像 M/N 一样分给不同核？** 因为 K 是累加维：`D[i][j] = Σ_p A[i][p]·B[p][j]` 的部分和必须累加在同一份累加器上。M/N 方向没有累加关系，所以才能任意并行。这个"K 必须串行累加、M/N 可以并行拆分"的不对称性，决定了 GEMM 调度的基本形态。

三个 BLOCK 的分工总结：

| 参数 | 切哪个维 | 决定什么 | 片上对应 |
| :---: | :---: | :--- | :--- |
| `BLOCK_M` | M（输出行） | 输出 Tile 的高度；每次搬的 A 行数 | L1 的 A 缓冲、L0A、L0C 行数 |
| `BLOCK_N` | N（输出列） | 输出 Tile 的宽度；每次搬的 B 行数 | L1 的 B 缓冲、L0B、L0C 列数 |
| `BLOCK_K` | K（收缩维） | K 循环每步的深度；每次搬运的 A/B 片段长度 | L1 每个 stage 的深度、K 循环次数 |

补充一点（第 12/22 天再深入）：BLOCK 是 L1 级的搬运与调度单位，真正喂给矩阵乘加单元的是更小的 `MAD_M/N/K` 子块——kernel 里 `BLOCK` 还会按 MAD 尺寸再切（`bf16_gemm.hpp` 第 123～153 行的三层循环），约束是 `BLOCK_K % MAD_K == 0`（第 70 行 `static_assert`）。今天先把 BLOCK 当作黑盒尺寸，MAD 留到第 3～4 周。

### 3.3 手工切分：M=128、N=128、K=256

取主例 `BLOCK_M=64、BLOCK_N=64、BLOCK_K=64`（数值均能整除，便于演算；真实 kernel 会选更大的块，见 3.7）：

```text
num_m_blocks = ceil(128/64) = 2
num_n_blocks = ceil(128/64) = 2
num_k_blocks = ceil(256/64) = 4
输出 Tile 数 = 2 × 2 = 4
```

输出 Tile 与所需输入的对应关系（B 采用仓库 `[N, K]` 视角，"B 行块 j"即 N 方向第 j 块、沿 K 展开的数据）：

| 输出 Tile | 计算的 D 区域 | 需要的 A（行块 × K 全长） | 需要的 B（行块 × K 全长） | K 循环步数 | 每 Tile 输入数据量（BF16） |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `(0, 0)` | `D[0:64, 0:64]` | `A[0:64, 0:256]`（64×256） | `B[0:64, 0:256]`（64×256） | 4 | 2×16384 元素 = 64 KiB |
| `(0, 1)` | `D[0:64, 64:128]` | `A[0:64, 0:256]` | `B[64:128, 0:256]` | 4 | 64 KiB |
| `(1, 0)` | `D[64:128, 0:64]` | `A[64:128, 0:256]` | `B[0:64, 0:256]` | 4 | 64 KiB |
| `(1, 1)` | `D[64:128, 64:128]` | `A[64:128, 0:256]` | `B[64:128, 0:256]` | 4 | 64 KiB |

每个 Tile 的 K 循环分 4 步，每步搬运 `64×64` 的 A 片段和 `64×64` 的 B 片段（共 8 KiB，很容易装进一个 L1 stage）：

```text
第 t 步（t = 0..3）：搬 A[行块, 64t : 64(t+1)] 和 B[行块, 64t : 64(t+1)] → L1 → L0 → MAD → 累加进 L0C
```

四步累加完成后，这个 Tile 的 L0C 里就是最终结果，写回 GM 的 D 对应区域。

### 3.4 复用在哪里：一个输出 Tile 的数据被谁共享

观察上表的关键事实：**A 行块 0 同时是 Tile (0,0) 和 Tile (0,1) 的输入；B 行块 0 同时是 Tile (0,0) 和 Tile (1,0) 的输入。** 一般规律：

```text
A 的某个行块被 num_n_blocks 个输出 Tile 共用（同 M 行的所有 Tile 都要它）
B 的某个行块被 num_m_blocks 个输出 Tile 共用（同 N 列的所有 Tile 都要它）
```

这就是 GEMM 数据复用的全部来源，它分布在四个层次（对应仓库的四个机制，源码导读逐一打卡）：

1. **Tile 内、K 循环间：L0C 累加器复用。** 一个 Tile 的所有 K 步累加到同一块 L0C，A/B 数据"用过即弃"，但输出累加器贯穿始终（`bf16_gemm.hpp` 第 149～151 行 `mad(..., ENABLE_KEEP, is_first)`）。输出数据完全不用重复读写。
2. **Tile 内、MAD 子块间：L0A/L0B 复用。** BLOCK 切成多个 MAD 子块时，同一份 L0A 数据可被多个 n 方向的 MAD 子块复用（`bf16_gemm.hpp` 第 142 行的搬运条件 `n_mad_idx == 0`——不是第一个 n 子块就跳过 L0A 搬运）。
3. **Tile 间：L2 复用。** 不同 Tile 共用的 A/B 行块，先被某个 Tile 从 GM 读入 L2，其他 Tile 命中 L2 而不再访问 GM。仓库的调度 swizzle 专门让 32 个核同时工作在相邻 Tile 上，最大化这种命中（`scheduler.hpp` 第 14～15 行注释 "GROUP_M swizzle + N-first snake for L2 locality"）；L2 hint 则决定"什么数据值得留在 L2"（`heuristics.hpp` 第 200～201 行）。
4. **核内跨 Tile：L1 驻留复用（resident A/B）。** 特殊场景下 A 或 B 整块驻留 L1/L0，连续两个任务共享同一行块时直接跳过搬运（`bf16_gemm.hpp` 第 92～93 行 `reuse_a` / `reuse_b`、第 109～118 行的条件搬运）。第 26 天专门学习。

今天重点算第 3 层的账——它回答"提高数据复用为什么能减少 GM 访问"。

### 3.5 复用的量化：无复用 vs 完美复用

设 dtype 每元素 s 字节。沿用 3.3 的记号：

```text
无任何跨 Tile 复用（每个 Tile 独立从 GM 读全部输入）：
    GM 读 = num_m_blocks × num_n_blocks × (BLOCK_M + BLOCK_N) × K × s
          = (num_n_blocks × M + num_m_blocks × N) × K × s

完美复用（每个 A/B 元素只从 GM 读一次，其余命中 L2/L1）：
    GM 读 = (M + N) × K × s

两者之比：
    A 的放大倍数 = num_n_blocks     （每个 A 元素被 N 方向的每个 Tile 各读一次）
    B 的放大倍数 = num_m_blocks
```

代入主例（BLOCK 64/64/64，BF16，s=2）：

| 量 | 计算 | 结果 |
| :--- | :--- | :--- |
| 无复用 GM 读 | `(2×128 + 2×128) × 256 × 2 B` | 256 KiB |
| 完美复用 GM 读 | `(128 + 128) × 256 × 2 B` | 128 KiB |
| 放大倍数 | `256 / 128` | **2 倍** |
| 每 Tile 算术强度 | `2×64×64 / ((64+64)×2)` | 32 FLOP/B |

对比不同 Tile 大小（同 shape，方块 Tile `BM = BN = b` 时算术强度 = `b/2` FLOP/B，与 M、N、K 都无关）：

| BLOCK (BM×BN×BK) | 输出 Tile 数 | K 步数 | 无复用/完美复用 | 每 Tile 算术强度 | 每 L1 stage 需 `(BM+BN)×BK×2` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 32×32×64 | 4×4 = 16 | 4 | 4 倍 | 16 FLOP/B | 8 KiB |
| 64×64×64 | 2×2 = 4 | 4 | 2 倍 | 32 FLOP/B | 16 KiB |
| 128×128×128 | 1×1 = 1 | 2 | 1 倍（无需复用） | 64 FLOP/B | 64 KiB |
| 128×128×256 | 1×1 = 1 | 1 | 1 倍 | 64 FLOP/B | 128 KiB |

规律一目了然：**Tile 越大，输入复用越高、GM 放大越小、每字节数据支撑的计算越多**——这正是第 1 天"大矩阵算力受限"结论的 Tile 级版本。

### 3.6 为什么复用是"必需品"而不是锦上添花

用仓库自己的性能模型做一次量级估算。`csrc/jit_kernels/heuristics.hpp` 第 52～71 行的 `get_hw_desc()` 写着 Ascend 950 的有效速率（这是仓库选配置用的成本模型参数，非官方规格书）：

```cpp
.gm_bw_gbps=3500, .l2_bw_gbps=4500,
.aic_clock_mhz=1650,
.bf16_flops_per_cycle=8192,  // 4096 multiply-adds; each counts as two FLOPs.
```

由此估算（32 个 AI Core，`common.hpp` 第 51 行 `kNumAICores = 32`）：

| 量 | 计算 | 结果 |
| :--- | :--- | :--- |
| 每核 BF16 算力 | `8192 FLOP/周期 × 1.65 GHz` | ≈ 13.5 TFLOPS |
| 全芯片算力 | × 32 | ≈ 432.5 TFLOPS（README 实测 431，吻合） |
| GM 带宽均分到每核 | `3500 / 32` | ≈ 109 GB/s |
| 单靠 GM 喂饱每核需要的算术强度 | `13.5e3 / 109` | **≈ 124 FLOP/B** |
| L2 带宽均分到每核 | `4500 / 32` | ≈ 141 GB/s |
| 单靠 L2 喂饱每核需要的算术强度 | `13.5e3 / 141` | ≈ 96 FLOP/B |

对照 3.5 的表：即使 128×128 的大 Tile，每 Tile 算术强度也只有 64 FLOP/B——**从单核视角看，无论 GM 还是 L2 都喂不满算力**。真正的出路是跨核共享：调度器让 32 个核同时算相邻的 Tile（一个 4×8 的邻域，见源码导读站点 3），这 32 个 Tile 只共享 4 个 A 行块和 8 个 B 行块——一份 B 数据从 GM 进入 L2 后被多核命中，等效于"每字节 GM 流量支撑了多倍计算"。再加上 L1 驻留与 L0C 累加，才把有效算术强度抬到算力需求之上。这就是 README 中 `(4096, 7168, 16384)` 能跑到硬件极限 99.8%、而 `(128, ...)` 只有 549 TFLOPS 且内存带宽打满 2456 GB/s 的根本原因：M 太小时 Tile 总数少、A/B 的跨 Tile 共享机会少，复用上不去，就退化为带宽受限（第 1 天 3.4 节的表可对照）。

### 3.7 Tile 为什么不能任意大：容量与对齐约束

"Tile 越大越好"止步于三道硬约束，全部能在源码里找到：

1. **L0C 容量**：输出 Tile 的累加器按 FP32 存放，`block_m × block_n × 4 ≤ L0CSizeBytes(256 KiB)`，即 `BM × BN ≤ 65536`。`make_gemm_candidate()` 的第一道检查（`heuristics.hpp` 第 101～102 行）：

   ```cpp
   const uint64_t block_bytes = static_cast<uint64_t>(block_m) * block_n * sizeof(float);
   if (block_bytes > L0CSizeBytes) return std::nullopt;
   ```

   128×128（64 KiB）轻松通过；256×256（256 KiB）恰好卡在上限。

2. **L1 容量与流水深度**：每个 L1 stage 存 `(BM + BN) × BK × s` 字节的 A/B 数据，还要留出 `num_l1_stages` 级（≥2，用于搬运与计算重叠）。`heuristics.hpp` 第 170～174 行按 `(L1SizeBytes - sf_bytes) / ab_bytes` 算出可用级数，不足 2 级直接否决。kernel 侧对应 `bf16_gemm.hpp` 第 66 行 `static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes, "L1 overflow")`。

3. **BLOCK_K 的上下限与对齐**：host 启发式把 BLOCK_K 限制在 `[min_block_k, 2×min_block_k]`——BF16 时 `min_block_k = 256 字节/周期 × 8 / 16 bit = 128`，即 128 或 256（`heuristics.hpp` 第 619～621 行，注释说明这是为了"整倍于 L1→L0 每周期搬运 256 字节，避免碎片搬运"）；另外 `block_k` 必须是对齐值（BF16 为 16，`get_mad_alignment_k`）的倍数（第 108～109 行）。M/N 方向搜索步长是 `FRAC_MN = 16`（`common.hpp` 第 63 行），上限 `block_cap`：BF16 为 512（第 604 行）。

所以选 Tile 是一个**带约束的搜索问题**：`select_gemm_config()`（`heuristics.hpp` 第 592 行起）在上述约束下从大到小枚举 `(block_m, block_n, block_k)`，用成本模型估算耗时，取最优。对 `(128, 128, 256)` 这样的中型 GEMM，合法配置很多，最终选哪个由成本模型决定——这正是第 12 天的主题，今天只要能算出"候选集合"即可。

### 3.8 边界 Tile：actual_m / actual_n

M、N、K 不被 BLOCK 整除时，边缘 Tile 被截断。调度器提供 `get_actual_m` / `get_actual_n`（`scheduler.hpp` 第 129～139 行）：

```cpp
__aicore__ uint32_t get_actual_m(uint32_t m_block_idx) const {
    ...
    return min(shape_m - m_block_idx * BLOCK_M, BLOCK_M);
}
```

例如 `M=130、BLOCK_M=64`：`num_m_blocks = ceil(130/64) = 3`，三个行块的实际行数是 `64、64、2`。kernel 搬运和计算都按 `actual_m/actual_n/actual_k` 截断（`bf16_gemm.hpp` 第 105～106、114 行），避免越界读 GM 和浪费算力。K 维同理（第 114 行 `actual_k = min(eff_shape_k - k_blk_idx * BLOCK_K, BLOCK_K)`）。测试里的 `unaligned` 场景（如 `tests/test_bf16_gemm.py unaligned-k`）覆盖的就是这类 shape。

## 四、源码导读

今天的导读主题是"**BLOCK 参数如何定义、如何约束、如何驱动调度与搬运**"，全程静态阅读、无需硬件。建议对照仓库依次打开以下文件（路径相对 `DeepGEMM-Ascend/`）。

### 站点 1：BLOCK 的 host 侧定义 —— `csrc/jit_kernels/gemm_config.hpp`

关键结构：`GemmConfig`（第 98 行起）。第 97 行的注释一句话点题：

```cpp
// Execution configuration: how to compute (tiling, pipeline, launch).
struct GemmConfig {
    // Tile sizes
    uint32_t block_m, block_n, block_k;
    uint32_t mad_m, mad_n, mad_k;
    // Pipeline depths
    uint32_t num_l1_stages;  // ...
```

`block_m/block_n/block_k`（第 100 行）就是今天的主角；`mad_*`（第 101 行）是更细的计算子块（今天只区分概念）；`num_l1_stages`（第 104 行）决定 L1 切几级轮转。第 163～172 行的 `std::formatter<GemmConfig>` 会把配置打印成 `GemmConfig(block=128x128x128, mad=..., ...)`——实践步骤里用它观察真实配置。

### 站点 2：BLOCK 的约束与搜索 —— `csrc/jit_kernels/heuristics.hpp`

关键函数：`make_gemm_candidate()`（第 92 行起）与 `select_gemm_config()`（第 592 行起）。按顺序读这几处：

- 第 100～102 行：L0C 容量检查（3.7 节约束 1）；
- 第 108～109 行：`block_k` 对齐检查；
- 第 170～174 行：`ab_bytes = (block_m + block_n) * block_k * elem_bits / 8`，除 L1 容量得 `num_l1_stages`（约束 2）；
- 第 200～201 行：L2 hint——A 的复用次数是 `num_n_blocks`，当 `num_n_blocks <= 2 && num_m_blocks > num_n_blocks`（A 几乎无复用）时把 A 标为 `NOTALLOC_KEEP`（加载时不占 L2），把 L2 让给高复用的一侧。这是"复用分析直接决定缓存策略"的落地；
- 第 604 行：`block_cap`（BF16 为 512）；第 619～621 行：`min_block_k` / `max_block_k`（BF16 为 128 / 256）；
- 第 626～628 行：`block_m`、`block_n` 的双层枚举循环（从大到小、按 16 步进），内层用成本模型打分。第 633～641 行的 `avg_block_bytes`、`min_memory_time_ns` 就是在算"每个 Tile 平均要搬多少字节"。

这一站回答的问题：**Tile 大小不是拍脑袋定的，而是在容量约束下搜索出来的。**

### 站点 3：Tile 网格与任务分发 —— `deep_gemm/include/deep_gemm/scheduler.hpp`

关键结构：`Scheduler`（第 49 行起）。文件头注释（第 10～15 行）直接给出了设计：每个核通过 `get_next_block()` 领取持久化任务编号 `(++current_iter) * kNumCores + core_id`（第 142 行），再映射为 Tile。重点读：

- 构造函数（第 87～97 行）：`num_m_blocks = ceil_div(shape_m, BLOCK_M)`、`num_n_blocks = ceil_div(shape_n, BLOCK_N)`、`num_blocks` 为两者乘积——3.2 节公式的源码出处；
- `get_global_idx()`（第 113～115 行）：`return block_idx * block_size;`——Tile 序号乘 BLOCK 得到全局元素偏移，`m_idx = m_block_idx × BLOCK_M`；
- `get_actual_m` / `get_actual_n`（第 129～139 行）：边界截断；
- `get_swizzled_block()`（第 202～248 行）：把线性任务编号映射成 `(m_block_idx, n_block_idx)` 的 swizzle。模板参数 `kGroupM = 4`（第 47 行），`kGroupN = 32 / kGroupM = 8`（第 219 行）：当 N 方向 Tile 足够多时，32 个连续任务恰好覆盖一个 **4×8 的 Tile 邻域**——32 个核共享 4 个 A 行块与 8 个 B 行块，这些数据从 GM 读入 L2（容量 128 MiB，`heuristics.hpp` 第 68 行 `l2_size_bytes`）后被反复命中；奇偶"蛇形"（`kSnake`，第 235 行）让相邻两批任务的 B 行块也衔接。**这是 3.4 节第 3 层复用（L2）的源码实体。**

### 站点 4：kernel 中的 BLOCK 与 K 循环 —— `deep_gemm/include/deep_gemm/bf16_gemm.hpp`

模板入口 `bf16_gemm_impl`（模板参数第 9～26 行，`BLOCK_M, BLOCK_N, BLOCK_K` 在第 12 行）。按顺序读：

- 第 59～64 行：片上缓冲声明——`l1a(BLOCK_M, BLOCK_K, 0)`、`l1b(BLOCK_N, BLOCK_K, ...)`、`l0a(MAD_M, MAD_K)`、`l0b(MAD_N, MAD_K)`、`l0c(MAD_M, MAD_N)`。**每个 L1 stage 的大小就是 `(BM 或 BN) × BLOCK_K`**，L0C 累加器按 MAD 尺寸切成 `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)` 块（第 51 行）；
- 第 66～70 行：容量 `static_assert`（L1/L0A/L0B/L0C overflow、`BLOCK_K % MAD_K == 0`）——3.7 节约束的设备侧版本；
- 第 89～96 行：调度主循环。`m_idx = scheduler.get_global_idx(BLOCK_M, m_block_idx)` 得到本 Tile 的全局行起点；`num_k_blocks = ceil_div(eff_shape_k, BLOCK_K)`；
- 第 92～93 行：`reuse_a = (m_idx == last_m_idx)`、`reuse_b = (n_idx == last_n_idx)`——**核内跨 Tile 复用检测**；配合第 109～118 行：resident A 时 A 变化才整体重搬（第 109～110 行）、非 resident A 每个 K 步都搬（第 117 行）、resident B 且 B 未变则跳过搬运（第 118 行 `if (!kResidentB || !reuse_b)`）；
- 第 112～118 行：K 循环主体。每步 `copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, ...)` 搬 `actual_m × actual_k` 的 A、`copy_gm_to_l1(l1b[s], ...)` 搬 `actual_n × actual_k` 的 B，`s` 在 `kNumL1Stages` 个 stage 间轮转；
- 第 123～153 行：MAD 两层循环（可先只看结构）：第 142～145 行的搬运条件（`n_mad_idx == 0` 时才搬 L0A）就是 3.4 节第 2 层的 L0 级复用；第 149～151 行 `mad(..., ENABLE_KEEP, is_first)` 是第 1 层的 L0C 累加复用。

### 站点 5：配置如何进入 kernel —— `csrc/jit_kernels/bf16_gemm.hpp`

关键函数：`launch_bf16_gemm()`（第 16 行起）。第 17 行 `select_gemm_config(desc)` 选出配置；第 48～53 行把 `config.block_m / block_n / block_k / mad_* / num_l1_stages / ...` 逐个填进 `std::format` 生成的模板实例化代码；第 71 行 `jit->compile` 编译、第 73 行 `jit->launch` 启动。这条链补全了今天的主线：

```text
GemmDesc（算什么）
→ select_gemm_config（选 BLOCK/stage）
→ JIT 生成 bf16_gemm_impl<..., BLOCK_M, BLOCK_N, BLOCK_K, ...> 实例
→ Scheduler 按 BLOCK 切 M×N 网格发任务
→ kernel K 循环按 BLOCK_K 搬数据、L0C 累加
```

**阅读检查点**（能回答再进入实践）：

1. `GemmConfig` 里 `block_*` 与 `mad_*` 的关系是什么？kernel 里的约束是哪一条？
2. 给定 `(M, N) = (128, 128)`、`BLOCK_M = 64`、`BLOCK_N = 32`，`Scheduler` 构造出的 `num_blocks` 是多少？
3. `select_gemm_config` 否决一个候选 `(block_m, block_n, block_k)` 的前两条硬性理由是什么？
4. kernel 第 142 行 `copy_l1_to_l0a` 的触发条件里为什么有 `n_mad_idx == 0`？
5. 32 个核领到的 32 个连续任务编号会被 swizzle 成什么样的 Tile 分布？为什么对 L2 友好？

## 五、实践步骤

以下步骤按顺序执行，**全部不需要 Ascend 硬件**；总时长约 2 小时。

### 步骤 1：纸笔切分主例（约 25 分钟）

不看 3.3 节的表，对 `M=128、N=128、K=256、BLOCK_M=64、BLOCK_N=64、BLOCK_K=64` 独立完成：

1. 写出 `num_m_blocks`、`num_n_blocks`、`num_k_blocks` 和输出 Tile 总数；
2. 画出 2×2 的输出 Tile 网格，为每个 Tile 标注它需要的 A 行块与 B 行块（编号即可）；
3. 计算每个 Tile 需要读取的 A/B 元素数与字节数（BF16）；
4. 计算无复用总 GM 读取量、完美复用读取量和放大倍数；
5. 追加一问：若 `M=130`（其他不变），`num_m_blocks` 变为多少？各行块的 `actual_m` 分别是多少？（答案：3；64、64、2。）

参考答案见第六节与 3.3～3.5 节。

### 步骤 2：用脚本枚举 Tile 配置（约 25 分钟）

在自己的练习目录新建 `tile_accounting.py`（不要改动仓库文件），写入：

```python
def cd(a, b):
    return -(-a // b)          # ceil 除法

def account(m, n, k, bm, bn, bk, elem_bytes=2):
    nmb, nnb, nkb = cd(m, bm), cd(n, bn), cd(k, bk)
    tiles = nmb * nnb
    no_reuse = (nnb * m + nmb * n) * k * elem_bytes      # 无跨 Tile 复用的 GM 读
    unique   = (m + n) * k * elem_bytes                   # 完美复用的 GM 读
    amp      = no_reuse / unique
    ai       = 2 * bm * bn / ((bm + bn) * elem_bytes)     # 每 Tile 算术强度
    stage_b  = (bm + bn) * bk * elem_bytes                # 每 L1 stage 的 A+B 字节
    stages   = min(8, 512 * 1024 // stage_b)              # kNumPipeEvents=8, L1=512KiB
    l0c_ok   = bm * bn * 4 <= 256 * 1024                  # L0C 容量检查（同 heuristics）
    return (f"BLOCK {bm:3d}x{bn:3d}x{bk:3d}  tiles={tiles:3d}  k_steps={nkb}  "
            f"amp={amp:.0f}x  AI={ai:5.0f} FLOP/B  L1_stage={stage_b//1024:3d}KiB  "
            f"L1_stages={stages}  L0C_ok={l0c_ok}")

M, N, K = 128, 128, 256
for bm, bn, bk in [(32, 32, 64), (64, 64, 64), (128, 128, 128),
                   (128, 128, 256), (256, 256, 128), (256, 512, 128)]:
    print(account(M, N, K, bm, bn, bk))
```

运行 `python tile_accounting.py`，预期输出（可逐项核对 3.5 节的表）：

```text
BLOCK  32x 32x 64  tiles= 16  k_steps=4  amp=4x  AI=   16 FLOP/B  L1_stage=  8KiB  L1_stages=8  L0C_ok=True
BLOCK  64x 64x 64  tiles=  4  k_steps=4  amp=2x  AI=   32 FLOP/B  L1_stage= 16KiB  L1_stages=8  L0C_ok=True
BLOCK 128x128x128  tiles=  1  k_steps=2  amp=1x  AI=   64 FLOP/B  L1_stage= 64KiB  L1_stages=8  L0C_ok=True
BLOCK 128x128x256  tiles=  1  k_steps=1  amp=1x  AI=   64 FLOP/B  L1_stage=128KiB  L1_stages=4  L0C_ok=True
BLOCK 256x256x128  tiles=  1  k_steps=2  amp=1x  AI=  128 FLOP/B  L1_stage=128KiB  L1_stages=4  L0C_ok=True
BLOCK 256x512x128  tiles=  1  k_steps=2  amp=1x  AI=  171 FLOP/B  L1_stage=192KiB  L1_stages=2  L0C_ok=False
```

观察两点：256×512 被 L0C 否决（`L0C_ok=False`，对应 `heuristics.hpp` 第 101～102 行）；BLOCK_K 从 128 加到 256 不改变复用（`amp` 不变），只减少 K 步数、但每个 L1 stage 更大、可轮转的级数更少——这正是 3.7 节"BLOCK_K 是搬运效率与流水深度的折中"的数值体现。

### 步骤 3：带宽需求量级估算（约 15 分钟）

用 3.6 节的模型参数（每核 `8192 FLOP/周期 × 1.65 GHz`，GM 3500 GB/s、L2 4500 GB/s，32 核）计算并回答：

1. 单靠 GM 喂饱一个核需要多少算术强度？（≈ 124 FLOP/B）
2. 步骤 2 里哪几组配置的每 Tile 算术强度达到或超过该值？它们靠什么满足 L0C/L1 约束、又付出了什么代价？
3. 32 个核共享一个 4×8 Tile 邻域时，一份 B 行块数据从 GM 读入 L2 后最多可被多少个核命中？（答案：8——同一 B 行块服务邻域内 4 个 m 方向 Tile，每个核各命中一次，加上最初读取共 8 次使用；A 行块同理是 4 核共享。）

### 步骤 4：源码导读打卡（约 30 分钟）

按第四节顺序依次打开 5 个站点文件，找到列出的行号并通读相应片段，每站用一句话记下它的职责。重点验证三处"复用"的源码实体：

- `bf16_gemm.hpp` 第 92～93 行的 `reuse_a` / `reuse_b` 与第 109～118 行的条件搬运；
- `scheduler.hpp` 第 14～15 行注释与第 202～248 行 swizzle（找出 `kGroupM = 4`、`kGroupN = 32 / kGroupM`）；
- `heuristics.hpp` 第 200～201 行的 L2 hint（对应"num_n_blocks 小 → A 无复用 → 不占 L2"）。

无硬件时的替代任务就是本步骤本身——纯静态阅读。

（可选，有 Ascend 环境时）在仓库根目录执行：

```bash
./develop.sh
export DG_PRINT_CONFIGS=1
python tests/test_bf16_gemm.py normal --skip-prof
```

输出中每个用例会先打印一行 `GemmDesc(m=..., n=..., k=..., ...) GemmConfig(block=MxNxK, mad=..., l1_stages=..., ...)`（`heuristics.hpp` 第 749～753 行的打印逻辑）。挑两三个 shape，把打印出的 `block=` 与你用 3.7 节约束推出的候选集合对照：真实配置是否落在合法区间内？`m=1` 的解码场景选了多大的 `block_m`（提示：搜索下限会截到 `align(m, 16)`，`heuristics.hpp` 第 608～609 行）？

## 六、当天产出

学习计划要求今天的产出是**一张输出 Tile 与输入 Tile 对应关系图**。以下参考答案以主例（`M=128、N=128、K=256`，`BLOCK 64/64/64`）绘制，建议先自己画草稿再对照：

```text
输入侧（GM，B 采用仓库 [N, K] 视角；K 按 BLOCK_K=64 切 4 步）

        K=256 ──────────────────────────────────────────▶
      ┌──────────┬──────────┬──────────┬──────────┐
 A    │ A00      │ A01      │ A02      │ A03      │  A 行块 0 = A[0:64,   0:256]
 M=128├──────────┼──────────┼──────────┼──────────┤
      │ A10      │ A11      │ A12      │ A13      │  A 行块 1 = A[64:128, 0:256]
      └──────────┴──────────┴──────────┴──────────┘
      ┌──────────┬──────────┬──────────┬──────────┐
 B    │ B00      │ B01      │ B02      │ B03      │  B 行块 0 = B[0:64,   0:256]
 N=128├──────────┼──────────┼──────────┼──────────┤
      │ B10      │ B11      │ B12      │ B13      │  B 行块 1 = B[64:128, 0:256]
      └──────────┴──────────┴──────────┴──────────┘
        每格 64×64 元素（Ait = A 行块 i 的第 t 个 K 片段，Bij 同理）

输出侧（D 的 2×2 Tile 网格；每格 = 64×64 输出元素）

              N 0..63                 N 64..127
         ┌─────────────────────┬─────────────────────┐
 M 0..63 │ D(0,0)              │ D(0,1)              │
         │  A 行块 0 × B 行块 0 │  A 行块 0 × B 行块 1 │
         │  = Σt A0t ⊗ B0t     │  = Σt A0t ⊗ B1t     │
         ├─────────────────────┼─────────────────────┤
 M64..127│ D(1,0)              │ D(1,1)              │
         │  A 行块 1 × B 行块 0 │  A 行块 1 × B 行块 1 │
         │  = Σt A1t ⊗ B0t     │  = Σt A1t ⊗ B1t     │
         └─────────────────────┴─────────────────────┘

复用关系（图中箭头的含义）：
  A 行块 0 ──被 D(0,0) 和 D(0,1) 共用──▶ A 复用因子 = num_n_blocks = 2
  B 行块 0 ──被 D(0,0) 和 D(1,0) 共用──▶ B 复用因子 = num_m_blocks = 2
  每个 Tile 内：4 个 K 步累加进同一块 L0C（输出不重复读写）
  无跨 Tile 复用时 GM 读 = 256 KiB；完美复用 = 128 KiB（放大 2 倍）
```

配套自查清单：

- 每个 Tile 需要的输入是不是"一个 A 行块 + 一个 B 行块、都沿 K 全长"？
- 放大倍数是否等于 `num_n_blocks`（A 侧）/ `num_m_blocks`（B 侧）？
- 图中是否标出了 K 分片（BLOCK_K）和边界截断（本例恰好整除，可另画一个 `M=130` 的版本练习 `actual_m`）？

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 分别切 `D = A × B` 的哪个维度？为什么 K 维的循环通常不能拆给不同核并行，而 M/N 方向的 Tile 可以？
2. `M=256、N=512、K=1024`，`BLOCK_M=128、BLOCK_N=128、BLOCK_K=256`：输出 Tile 数是多少？每个 Tile 的 K 循环走几步？每个 Tile 需要读取多少 A 元素和 B 元素？
3. 承上题（BF16）：无跨 Tile 复用时总 GM 读取量是多少字节？完美复用时是多少？比值是多少？
4. 一个 `64×64` 的 BF16 Tile 算术强度是多少？`256×256` 呢？为什么方块 Tile 的算术强度与 K 无关？
5. `make_gemm_candidate()` 否决候选配置的第一条容量检查是什么？`256×512` 的 BLOCK 会因为哪条约束被否决？
6. BF16 的 `block_k` 搜索范围是多少？下限由什么决定？
7. 调度器的 `kGroupM = 4`、`kGroupN = 32 / kGroupM` swizzle 想达到什么效果？它服务于哪一层存储的复用？
8. （思考题）把 `BLOCK_M` 和 `BLOCK_N` 同时加倍，GM 流量、L1 每 stage 容量、L0C 需求各往哪个方向变？这说明选 Tile 本质上是在权衡什么？

**参考答案：**

1. `BLOCK_M` 切 M（输出行，决定 A 行块大小）、`BLOCK_N` 切 N（输出列，决定 B 行块大小）、`BLOCK_K` 切 K（收缩维，决定每次搬运的深度与循环步数）。K 是累加维，部分和必须累加在同一累加器上（kernel 中同一个 `l0c_tile`，`bf16_gemm.hpp` 第 149～151 行 `ENABLE_KEEP`）；M/N 方向的 Tile 之间没有累加关系、彼此独立，所以可以任意分配（`scheduler.hpp` 的 `num_blocks = num_m_blocks × num_n_blocks`）。
2. Tile 数 `= ceil(256/128) × ceil(512/128) = 2 × 4 = 8`；K 步数 `= ceil(1024/256) = 4`；每 Tile 读 A `128×1024 = 131072` 个元素、B `128×1024 = 131072` 个元素（沿 K 全长）。
3. 无复用：`(num_n_blocks·M + num_m_blocks·N)·K·s = (4×256 + 2×512)×1024×2 = 4 MiB`；完美复用：`(256+512)×1024×2 = 1.5 MiB`；比值：A 侧 4 倍、B 侧 2 倍，总体 `4 MiB / 1.5 MiB ≈ 2.67`。
4. `64×64`：`2×64×64/((64+64)×2) = 32 FLOP/B`；`256×256`：`2×256×256/((256+256)×2) = 128 FLOP/B`。方块 `b×b` 时分子分母的 K（搬入量与计算量都随 K 线性增长）约掉了，得 `b/2` FLOP/B，故与 K 无关——K 只影响循环次数，不影响"每字节数据支撑多少计算"。
5. L0C 检查：`block_m × block_n × sizeof(float) ≤ L0CSizeBytes(256 KiB)`（`heuristics.hpp` 第 101～102 行）。`256×512` 的累加器需 `256×512×4 = 512 KiB > 256 KiB`，被否决（即实践步骤 2 输出的 `L0C_ok=False`）。
6. 128 或 256。下限 `min_block_k = l1_to_l0_bytes_per_cycle × 8 / elem_bits = 256×8/16 = 128`（`heuristics.hpp` 第 619～621 行），为的是整倍于 L1→L0 每周期 256 字节的搬运宽度，避免碎片传输。
7. 让 32 个核的 32 个连续任务落在一个 `4×8` 的 Tile 邻域：共享 4 个 A 行块与 8 个 B 行块，这些数据从 GM 进入 L2 后被多核命中。它服务于 **L2 层的跨 Tile 复用**（`scheduler.hpp` 第 10～15 行注释 "GROUP_M swizzle + N-first snake for L2 locality"，第 219 行 `kGroupN = 32 / kGroupM`）。
8. GM 流量下降（复用因子 `num_n_blocks`、`num_m_blocks` 各减半）、算术强度翻倍；但 L1 每 stage 需求约翻倍（`(BM+BN)×BK×s`）、L0C 累加器需求翻 4 倍（`BM×BN×4`）。本质权衡：**用有限的片上容量换 GM 访问次数的减少**——Tile 加倍受 L0C 的 `BM×BN ≤ 65536` 和 L1 的 stage 数量（≥2 级流水）双重封顶。

## 附录 A：本日概念速查

| 概念 | 一句话定义 | 仓库落点 |
| :--- | :--- | :--- |
| `BLOCK_M/N/K` | Tile 在 M/N/K 三个方向的尺寸 | `GemmConfig.block_*`、kernel 模板参数 |
| 输出 Tile 数 | `ceil(M/BM) × ceil(N/BN)` | `Scheduler` 构造函数（scheduler.hpp:91-93） |
| K 循环步数 | `ceil(K/BK)` | `bf16_gemm.hpp:96` `num_k_blocks` |
| 每 Tile 输入 | 一个 A 行块 + 一个 B 行块，均沿 K 全长 | kernel 第 112～118 行循环体 |
| 复用因子 | A 侧 = `num_n_blocks`；B 侧 = `num_m_blocks` | swizzle 与 L2 hint 的设计依据 |
| Tile 算术强度 | `2·BM·BN / ((BM+BN)·s)`，方块时 = `b/2` FLOP/B | 成本模型 `get_gemm_cost` 的输入 |
| `actual_m/n/k` | 边界 Tile 的有效尺寸 | `scheduler.hpp:129-139`、kernel:114 |
| L0C 累加复用 | 一个 Tile 的 K 循环累加进同一 L0C | `mad(..., ENABLE_KEEP, is_first)` |
| L2 swizzle | 32 个连续任务映射到 4×8 邻域 | `kGroupM=4`、`kGroupN=32/4` |
| L2 hint | 无复用的一侧不占 L2（`NOTALLOC_KEEP`） | `heuristics.hpp:200-201` |
| resident A/B | 行块不变时跳过重复搬运 | `reuse_a/reuse_b`（bf16_gemm.hpp:92-93） |

## 附录 B：遗留问题记录模板

```text
# 第 3 天遗留问题
1. （例如）MAD_M/N/K 与 BLOCK_M/N/K 的具体分工还只是概念性理解 —— 计划第 12/22 天深入
2. （例如）kGroupM=4 的 swizzle 映射细节（XOR、蛇形）没有逐行推演 —— 计划第 17 天对照 scheduler 逐个 block 手算
3. （例如）L1 多级 stage 如何与 K 循环配合实现搬运/计算重叠 —— 计划第 23 天流水线专题
4. ……
```
