# 第 4 周 · 第 1 天（总第 22 天）：拆解 K 维两层循环

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 4 周"BF16 Kernel——MAD、流水线与 Epilogue"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 45 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

第 3 周我们已经把 `bf16_gemm_impl` 的外围全部走通：模板参数（第 15 天）、片上内存规划（第 16 天）、输出块调度（第 17 天）、地址表达（第 18 天）、`GM → L1 → L0` 搬运（第 19 天），并在第 20 天把它们串成了一个输出 Tile 的完整演算——但当时我们刻意把 K 维循环当成一个黑盒，只数了圈数。今天打开这个黑盒：`deep_gemm/include/deep_gemm/bf16_gemm.hpp:112-168` 的 K 维其实有**两层**循环——外层 `BLOCK_K` 循环（`k_blk_idx`）管数据搬运的节拍，内层 `MAD_K` 循环（`k_mad_idx`）管矩阵乘指令的节拍。两层循环各带一个循环计数器、一个 stage 环下标（`s` 与 `s_ab`）和一套边界变量（`actual_k` 与 `eff_mad_k`），共同决定 `is_first` / `is_last` 两个标志——它们控制 L0C 累加器"何时清零、何时提交"。本周后几天（第 23～26 天）将依次展开同步、MAD、Epilogue 和特殊路径，今天先把 K 维的**执行顺序**彻底搞清楚，当天产出是一张 K 维循环展开图。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 K 维两层循环的**分工**：`BLOCK_K` 循环是 `GM → L1` 数据搬运与多级缓冲流水的节拍，`MAD_K` 循环是 `L0A/L0B → MAD → L0C` 单条矩阵乘指令的节拍，并能解释为什么一层循环不够（`BLOCK_K` 受 L1 容量与搬运带宽约束、`MAD_K` 受 L0A/L0B 容量与整除约束，两者量级不同）；
2. 写出外层循环的全部控制量的定义与来源：`num_k_blocks`、`eff_shape_k`、`k_idx_base`、`k_idx`、`actual_k`、L1 stage 环下标 `s`（`bf16_gemm.hpp:95-97, 112-114`），并说明 `kResidentA` 路径下读偏移的变化（第 143 行）；
3. 写出内层循环的全部控制量：`k_mad_idx`、`eff_mad_k`、L0 stage 环下标 `s_ab`（第 133-137 行），并解释 `as_mad_aligned`（`ascend/tile_ptr.hpp:52-55`）对 tile 形状的对齐作用与传给 `mad()` 的仍是原始 `eff_mad_k` 的区别；
4. 分析 `is_first`、`is_last` 对累加的影响：`is_first` 如何变成 `mad()` 的 `clear_C` 参数（`ascend/copy.hpp:371-381`）、`is_last` 如何在 `ENABLE_KEEP` / `ENABLE_UPDATE` 之间选择 unit_flag、为什么这两个标志是**对每个输出子块各自独立判定**的；
5. 给定 shape 与一组 Block/MAD 参数，**手工展开** K 维循环的完整执行顺序：每一步的 `(k_blk_idx, k_mad_idx, actual_k, eff_mad_k, s, s_ab, is_first, is_last)`，包括 K 不整除 `BLOCK_K` 时的尾部处理；
6. 说明 K-Grouped GEMM 中 `eff_shape_k` / `k_idx_base` 的来源（`scheduler.hpp:73, 124, 163-177`），以及空组为什么不产生任何循环迭代；
7. 有 Ascend 环境时，用 `DG_PRINT_CONFIGS=1` 打印真实选中的 `GemmConfig`，核对自己演算用的 `BLOCK_K / MAD_K / L1 stages / L0 stages`；无环境时完成纸面展开与源码对照。

**当天产出：K 维循环展开图**（模板见第六节，参考答案见 3.6 节）。

## 二、预备知识

- **第 3 天**：`BLOCK_M/N/K` 的 Tiling 概念，输出 Tile 与 K 维循环次数的计算。
- **第 12 天**：`GemmConfig` 中 `block_k`、`mad_k`、`num_l1_stages`、`num_l0_stages` 字段的含义；`BLOCK_K` 与 `MAD_K` 是两级不同的切分。
- **第 15 天**：`bf16_gemm_impl` 的 26 个模板参数分类，尤其是 Block 组（`BLOCK_M/N/K`）、MAD 组（`MAD_M/N/K`）和 Stage 组（`kNumL1Stages/kNumL0Stages`）。
- **第 16 天**：`l1a/l1b/l0a/l0b/l0c` 的 `constexpr` 声明与多阶段缓冲的 offset 分配（`bf16_gemm.hpp:59-69`）；BF16 下 L0A/L0B 各 64 KB、L0C 256 KB、L1 512 KB（`common.hpp:54-57`）。
- **第 17 天**：`Scheduler::get_next_block` 分发输出块；`actual_m` / `actual_n` 是 M/N 边界的"真实长度"。
- **第 19 天**：`copy_gm_to_l1` / `copy_l1_to_l0a/b` 的签名，`actual_m/n/k` 如何约束边界搬运的规模。
- **第 4 天**：BF16 输入、FP32 累加（`l0c` 是 `float` 累加器）。
- **C++ 语法**：`for` 循环的逗号分隔初始化/递增表达式（`for (uint32_t i = 0, s = 0; ...; i++, s = ...)`）、`min`、整除向上取整。能对照读懂即可。

不需要第 23 天的同步细节：本文提到 `asc_sync_wait/notify` 时只说明"它在等谁、通知谁"，配对关系与时序图留到明天。

## 三、详细讲解

### 3.1 全景：K 维循环在数据流的什么位置

先回忆整体数据流（第 5 天）：

```text
GM ──MTE2──> L1 ──MTE1──> L0A/L0B ──M──> L0C ──Epilogue──> GM
```

K 维两层循环正是这条流水线上两个"节拍器"：

- **外层 `BLOCK_K` 循环**：每圈把 A 的一个 `BLOCK_M × BLOCK_K` 片段和 B 的一个 `BLOCK_N × BLOCK_K` 片段从 GM 搬进 L1 的第 `s` 号 stage 缓冲（`bf16_gemm.hpp:116-119`）。L1 里共有 `kNumL1Stages` 份这样的缓冲，第 `i` 圈搬运可以和第 `i-1`、`i-2`… 圈的 L1→L0 搬运及计算重叠——这是 MTE2 与 MTE1/M 流水的基础。
- **内层 `MAD_K` 循环**：每圈从 L1 取出 `MAD_M × MAD_K`（A 侧）和 `MAD_N × MAD_K`（B 侧）搬进 L0A/L0B 的第 `s_ab` 号 stage（第 141-146 行），然后执行一次 `mad()`（第 149-151 行），把乘加结果累加进 L0C。L0A/L0B 里共有 `kNumL0Stages` 份缓冲，让 L1→L0 搬运和上一条 mmad 的计算重叠。

两层循环的完整嵌套位置（含 M/N 子块循环，箭头右侧是源码行号）：

```text
while (scheduler.get_next_block(m, n))                       // :89   输出块 BLOCK_M × BLOCK_N
│  num_k_blocks = ceil_div(eff_shape_k, BLOCK_K)             // :96
│  for k_blk_idx in [0, num_k_blocks):                       // :112  ┐ 外层：BLOCK_K 循环
│  │  k_idx    = k_idx_base + k_blk_idx * BLOCK_K            // :113  │ GM 上的 K 起点
│  │  actual_k = min(eff_shape_k - k_blk_idx*BLOCK_K,        // :114  │ 本块真实 K 长度
│  │                BLOCK_K)                                  //       │
│  │  [wait] GM → L1: A[m_idx.., k_idx..] → l1a[s]           // :116-119
│  │        GM → L1: B[n_idx.., k_idx..] → l1b[s]            //       s = L1 stage 环
│  │  for m_mad_idx in [0, actual_m) step MAD_M:              // :123  │ M 子块
│  │  │  for n_mad_idx in [0, actual_n) step MAD_N:          // :124  │ N 子块
│  │  │  │  [若 k_blk_idx==0] 等 epilogue stage 空闲          // :129-131
│  │  │  │  for k_mad_idx in [0, actual_k) step MAD_K:        // :133  ┐┐ 内层：MAD_K 循环
│  │  │  │  │  is_first  = (k_blk_idx==0 && k_mad_idx==0)     // :134 ││
│  │  │  │  │  is_last   = (k_blk_idx==num_k_blocks-1         // :135 ││
│  │  │  │  │              && k_mad_idx+MAD_K >= actual_k)    //      ││
│  │  │  │  │  eff_mad_k = min(actual_k - k_mad_idx, MAD_K)   // :137 ││
│  │  │  │  │  [wait] L1 → L0A/L0B (stage s_ab)               // :141-146 ┘│ s_ab = L0 stage 环
│  │  │  │  │  mad(l0c, l0a, l0b, m, n, eff_mad_k,            // :149-151   │
│  │  │  │  │      is_last ? UPDATE : KEEP, is_first)         //            │
│  │  │  │  [若最后一个 k_blk] L0C → epilogue / GM            // :155-163   │
│  │  │  [notify] 释放 L1 stage s                             // :167 ┘
```

三个容易忽略但很关键的结构事实：

1. **M/N 子块循环在 `k_blk` 循环之内**。也就是说，每搬来一个 K 块，就把所有 `(m_mad, n_mad)` 输出子块都推进一个 `BLOCK_K` 切片，然后才搬下一个 K 块。
2. **L0C 子块跨 `k_blk` 迭代持续累加**。每个输出子块在 `l0c[s_cd]` 里的值，是所有 `(k_blk, k_mad)` 步骤累加的总和——这正是 `is_first` 必须检查 `k_blk_idx == 0`、提交/写回必须等到最后一个 `k_blk` 的原因。
3. **两个 stage 环转速不同**。`s` 每个 `BLOCK_K` 圈转一格（模 `kNumL1Stages`），`s_ab` 每条 mmad 转一格（模 `kNumL0Stages`）。

### 3.2 两层循环的分工：为什么一层不够

理想情况下我们希望"一条循环、一次搬运、一次计算"，但有两级硬约束把 K 维切成了两层：

**`BLOCK_K` 受 L1 容量与搬运带宽约束。** 每个 L1 stage 要容纳 `(BLOCK_M + BLOCK_N) × BLOCK_K` 的 BF16 数据（`bf16_gemm.hpp:59-60` 的 `l1a`/`l1b` 声明，stage 份数由 `l1b.offset(kNumL1Stages) <= L1SizeBytes` 断言，第 66 行）。`BLOCK_K` 越大，单圈搬运的连续数据越多、循环开销越少，但 stage 份数变少、流水深度变浅。启发式给 BF16 的范围是（`csrc/jit_kernels/heuristics.hpp:618-621`）：

```text
min_block_k = l1_to_l0_bytes_per_cycle × 8 / elem_bits = 256 × 8 / 16 = 128
max_block_k = 2 × min_block_k = 256
```

即 BF16 的 `BLOCK_K` 典型取 **128 或 256**（注释：取 L1→L0 单周期搬运量的倍数，避免部分搬运浪费带宽）。

**`MAD_K` 受 L0A/L0B 容量与整除约束。** 单条 mmad 的操作数驻留在 L0A/L0B：`MAD_M × MAD_K × 2B × kNumL0Stages ≤ 64KB`（L0A 断言，第 67 行，L0B 对称）。`kNumL0Stages` 被启发式**固定为 2**（`heuristics.hpp:111-112`，注释"always 2 stage"）。于是给定 `block_m × block_n` 后，`MAD_K` 由一组公式推出（`heuristics.hpp:129-142`）：

```text
l0a_k = L0ASizeBytes × 8 / (num_l0_stages × mad_m × elem_bits)   // A 侧 K 上限
l0b_k = L0BSizeBytes × 8 / (num_l0_stages × mad_n × elem_bits)   // B 侧 K 上限
mad_k = min(l0a_k, l0b_k, max(k_alignment, block_k / num_l0_stages))
mad_k = 向下对齐到 k_alignment（BF16 为 16）
要求：block_k % mad_k == 0，且 mad_k ≥ min(block_k, mte1_copy_k) / num_l0_stages
```

例如 `block = 128×128×128` 时：`l0a_k = l0b_k = 128`，`mad_k = min(128, 128, max(16, 64)) = 64`，`128 % 64 == 0` 成立——所以 **MAD = 128×128×64**。

两级约束的量级不同（搬运粒度 128/256 vs 指令 K 深度 64/128），且仓库用一条 `static_assert` 把关系钉死（`bf16_gemm.hpp:70`）：

```cpp
static_assert(BLOCK_K % MAD_K == 0, "BLOCK_K must be a multiple of MAD_K");
```

这条断言带来一个重要推论：**满块（`actual_k == BLOCK_K`）时每条 mmad 的 K 深度恰好等于 `MAD_K`，不存在零头；只有最后一个 K 块（`actual_k < BLOCK_K`）的最后一次 mmad 可能出现 `eff_mad_k < MAD_K`**。

### 3.3 外层循环：`k_blk_idx` 与它的控制量

逐个看外层循环头（`bf16_gemm.hpp:112-114`）涉及的控制量：

| 变量 | 定义处 | 含义 |
|---|---|---|
| `eff_shape_k` | :95 | 本次 GEMM 的有效 K 长度。Normal/Batched 时就是 `shape_k`；K-Grouped 时是当前组的 `scheduler.current_shape_k`（见 3.7） |
| `num_k_blocks` | :96 | `ceil_div(eff_shape_k, BLOCK_K)`，外层循环总圈数 |
| `k_idx_base` | :97 | K 方向的全局起点偏移。Normal 时为 0；K-Grouped 时为 `scheduler.get_k_idx_base()`（前序各组 K 的累计起点） |
| `k_blk_idx` | :112 | 外层循环计数器，从 0 到 `num_k_blocks - 1` |
| `k_idx` | :113 | `k_idx_base + k_blk_idx * BLOCK_K`，本块数据在 GM 上的 K 起点元素偏移，供 `a.index(m_idx, k_idx)` / `b.index(n_idx, k_idx)` 计算 GM 地址 |
| `actual_k` | :114 | `min(eff_shape_k - k_blk_idx * BLOCK_K, BLOCK_K)`，本块**真实** K 长度；只有最后一个块可能小于 `BLOCK_K` |
| `s` | :112 | L1 stage 环下标，`k_blk_idx % kNumL1Stages`（用条件表达式手写取模，`:112` 的递增子句） |

三个要点：

**要点 1：`k_idx` 是 GM 地址的输入，`actual_k` 是搬运规模的输入。** 第 117-118 行的搬运 `copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, ...)` 中，`k_idx` 决定从 GM 的哪个 K 列开始读，`actual_k` 决定读多少列（尾部块只搬真实存在的列，不多搬）。这正是第 19 天"边界搬运"在 K 维的体现。

**要点 2：`s` 环实现 L1 多级缓冲。** 第 116 行先 `asc_sync_wait(PIPE_MTE1, PIPE_MTE2, s)`——等 MTE1 侧确认第 `s` 号 L1 缓冲已被消费完（第 167 行的 `asc_sync_notify(PIPE_MTE1, PIPE_MTE2, s)` 在本块所有 M/N 子块用完后发出）；随后发起 GM→L1 搬运，第 119 行通知 MTE1 数据就绪。配对细节是第 23 天的主题，今天只需记住：**进入第 `k_blk_idx` 圈时，`s` 号缓冲必须是空闲的；离开时它被标记为可复用**。首次运行前，第 83 行的 `set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>()` 把全部 stage 事件预置为"空闲"，保证第一圈不卡死。

**要点 3：`kResidentA` 特例。** 当 A 的整个 K 常驻 L1（单缓冲，第 59 行把 `l1a` 的 `shape_k` 定为 `aligned(SHAPE_K, 16)`，第 109-110 行在 K 循环外一次性搬完），内层读 L1 的偏移不再是 stage 内的 `k_mad_idx`，而是 `(kResidentA ? k_blk_idx * BLOCK_K : 0) + k_mad_idx`（第 143 行）——`k_blk_idx * BLOCK_K` 这一项正是"跨出当前 stage、在整条常驻缓冲里寻址"的补偿。这是第 26 天的主题，这里只需看出 `k_blk_idx` 在地址计算中的另一个用途。

### 3.4 内层循环：`k_mad_idx`、`eff_mad_k` 与 `s_ab`

内层循环头在第 133 行：

```cpp
for (uint32_t k_mad_idx = 0, s_ab = 0; k_mad_idx < actual_k;
     k_mad_idx += MAD_K, s_ab = s_ab == kNumL0Stages - 1 ? 0 : s_ab + 1) {
```

| 变量 | 定义处 | 含义 |
|---|---|---|
| `k_mad_idx` | :133 | 内层循环计数器，**在当前 K 块内**从 0 开始、步长 `MAD_K`、上界 `actual_k` |
| `s_ab` | :133 | L0A/L0B stage 环下标，`k_mad_idx / MAD_K` 对 `kNumL0Stages` 取模（典型为 2，双缓冲） |
| `eff_mad_k` | :137 | `min(actual_k - k_mad_idx, MAD_K)`，本条 mmad 的**真实** K 深度 |
| `l0a_tile` / `l0b_tile` | :138-139 | `l0a[s_ab].as_mad_aligned(eff_mad_m, eff_mad_k)` 等，本条 mmad 的 L0 操作数 tile |

四个要点：

**要点 1：`eff_mad_k` 是 `actual_k` 的镜像。** 外层的尾部处理（`actual_k < BLOCK_K`）传导到内层：满块时 `eff_mad_k` 恒等于 `MAD_K`；只有最后一个 K 块的最后一次迭代可能出现 `eff_mad_k = actual_k - k_mad_idx < MAD_K`。它与 M/N 侧的 `eff_mad_m = min(actual_m - m_mad_idx, MAD_M)`（:125）、`eff_mad_n`（:126，多一步按 UB dtype 的对齐）构成同一个"真实长度"模式。

**要点 2：`as_mad_aligned` 只对齐 tile 描述符，不改变传给 mmad 的 K。** `ascend/tile_ptr.hpp:52-55`：

```cpp
constexpr __aicore__ operand_ptr<...> as_mad_aligned(uint64_t shape_mn, uint64_t shape_k) const {
    constexpr auto FRAC_K = get_frac_k<dtype_t>();
    return operand_ptr<...>(aligned(shape_mn, FRAC_MN), aligned(shape_k, FRAC_K), addr);
}
```

BF16 下 `FRAC_MN = 16`（`common.hpp:63`）、`get_frac_k<bfloat16_t>() = 32 / 2 = 16`（`common.hpp:73-79`）。于是 `eff_mad_k = 42` 时 tile 的 `shape_k` 会被抬到 `aligned(42, 16) = 48`——这个对齐值决定 **L1→L0 搬运**读多少个 fractal 列（`copy_l1_to_l0a` 用 `dst.shape_k / FRAC_K` 计算搬运量，`ascend/copy.hpp:134-145`）；而第 149 行传给 `mad()` 的第三个形状参数仍是**原始的 `eff_mad_k`（42）**。"tile 形状对齐、计算深度取真值"这两件事并存，尾部多余的对齐列如何不污染结果属于硬件/instruction 细节，仓库用 `unaligned-k` 测试（3.6 节）守住正确性，我们不凭空猜测。

**要点 3：`s_ab` 环转速更快。** 外层 `s` 每圈转一格，内层 `s_ab` 每条 mmad 转一格。以 `BLOCK_K=128, MAD_K=64, kNumL0Stages=2` 为例：一个 K 块内 `s_ab` 走 `0 → 1`，下一个 K 块又从 0 重新开始（`s_ab` 在 `for` 初始化子句里归零，注意它是**每个输出子块的 K 遍历各自从 0 起**的）。第 141/146/148/152 行围绕每次 L1→L0 搬运和 mmad 的 `wait/notify` 同样留到第 23 天。

**要点 4：L0 搬运有条件复用。** 第 142-145 行：

```cpp
if (BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0)
    copy_l1_to_l0a(l0a_tile, l1a[kResidentA ? 0 : s], m_mad_idx, ...);
if ((!kResidentB || !reuse_b) && (BLOCK_M == MAD_M || BLOCK_N != MAD_N
    || BLOCK_K > MAD_K * kNumL0Stages || m_mad_idx == 0))
    copy_l1_to_l0b(l0b_tile, l1b[s], n_mad_idx, k_mad_idx);
```

当整个 K 块装得进 L0 的 stage 缓冲（`BLOCK_K <= MAD_K * kNumL0Stages`）且输出在 N（或 M）方向拆了多个子块时，同一份 L0A 可以服务所有 `n_mad_idx`——所以只在 `n_mad_idx == 0` 时搬一次，其余子块直接复用 L0 里现成的数据（L0B 对 `m_mad_idx` 对称）。这解释了"一个 K 块内 L1→L0 的实际搬运次数"可以少于"子块数 × MAD 步数"。条件里出现的 `BLOCK_K > MAD_K * kNumL0Stages` 正是判断"L0 装不下整个 K 块、无法跨子块复用"的情形。完整分析留到第 24 天。

### 3.5 `is_first` / `is_last`：L0C 累加器的生命周期

第 134-135 行定义、第 149-151 行消费，是今天最核心的两行：

```cpp
bool is_first = k_blk_idx == 0 && k_mad_idx == 0;
bool is_last  = k_blk_idx == num_k_blocks - 1 && k_mad_idx + MAD_K >= actual_k;
...
mad(l0c_tile, l0a_tile, l0b_tile, eff_mad_m, eff_mad_n, eff_mad_k,
    is_last ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
    is_first);
```

**`is_first` → `clear_C`：从零开始。** `mad()` 的封装在 `ascend/copy.hpp:370-381`：

```cpp
template <typename TA, typename TB, typename TC>
__aicore__ inline void mad(l0c_ptr<TC> l0c, l0a_ptr<TA> l0a, l0b_ptr<TB> l0b,
                           uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           asc_unit_flag_mode unit_flag, bool clear_C = false) {
    ...
    asc_mmad(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n,
             static_cast<uint8_t>(unit_flag), /*k_direction_align*/ true,
             /*c_matrix_source*/ false, /*c_matrix_init_val*/ clear_C);
}
```

注意两点：其一，仓库的 `mad()` 按 `(m, n, k)` 接收形状，内部调用 `asc_mmad` 时调换成 `(m, k, n)`；其二，`clear_C` 对应 CANN 指令参数 `c_matrix_init_val`（`asc_mmad` 等 cube 指令声明于 CANN 工具链的 `c_api` 头文件，在本仓库之外，经 `<c_api/asc_simd.h>` 引入）。`is_first = true` 时本条 mmad 用初始值 0 作为 L0C 的来源——这是必须的：`l0c[s_cd]` 这个物理缓冲刚存过**上一个输出块**的结果（第 16 天的 stage 复用），不清零就会把旧值累加进来。

**`is_last` → `ENABLE_UPDATE`：末步提交。** `asc_unit_flag_mode` 的取值（`DISABLE / ENABLE_KEEP / ENABLE_UPDATE`）同样定义在 CANN 的 `c_api` 头里。仓库内**所有** GEMM 主线都遵守同一个约定——中间步 `ENABLE_KEEP`、最后一步 `ENABLE_UPDATE`：

- `bf16_gemm.hpp:150`、`fp8_gemm.hpp:229`、`fp8_dequant_gemm.hpp:321`、`mega_moe.hpp:599`：`is_last ? ENABLE_UPDATE : ENABLE_KEEP`；
- 单发场景（一次 mmad 出结果，没有 K 循环）如 `mqa_logits.hpp:397`：直接 `ENABLE_UPDATE` 且 `clear_C = true`。

从这套用法可以读出的契约是：**只有 K 维最后一次 mmad 的结果才作为最终值写出到 L0C**，供紧随其后的第 155-163 行（`epilogue.aic_store` 或 `kDirectStore` 下的 `copy_l0c_to_gm`）消费；中间各步的部分和用 `ENABLE_KEEP` 保持在累加链中。寄存器级的确切行为属于 CANN 指令集文档，我们以仓库用法为准、不过度解读。

**两个标志为什么长这样？** 逐条拆解：

- `is_first` 要求 `k_blk_idx == 0 && k_mad_idx == 0` 两个条件同时成立。漏掉 `k_blk_idx == 0` 会怎样？第 2、3…个 K 块的第一条 mmad 也会 `clear_C`，把前面 K 块辛苦累加的部分和清掉，最终结果只剩最后一个 K 块的贡献——错误。漏掉 `k_mad_idx == 0` 则同一个 K 块内每条 mmad 都清零，只剩最后一条的部分和——同样错误。
- `is_last` 用 `k_mad_idx + MAD_K >= actual_k` 而不是 `==`。满块时最后一步 `k_mad_idx + MAD_K == actual_k`（如 64+64=128）；尾部块时 `k_mad_idx + MAD_K` 可能**超过** `actual_k`（如 `actual_k = 106` 时 64+64=128 > 106），此时该步的 `eff_mad_k = 42 < MAD_K`，但确实已是最后一次迭代。`>=` 把两种情形都覆盖了。
- **两个标志按输出子块独立判定。** `is_first`/`is_last` 在 `m_mad`/`n_mad` 循环之内计算（:134-135），所以对每个 `(m_mad_idx, n_mad_idx)` 子块，它的 K 遍历都各自"第一步清零、末步提交"——每个 `l0c[s_cd]` 子块是一条独立的累加链，`s_cd` 环大小 `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`（:51）。

把一个输出子块的 K 遍历（共 `T = Σ_{k_blk} ceil(actual_k / MAD_K)` 条 mmad）中标志的取值画出来：

```text
mmad 序号:   #1         #2    #3   ...   #T-1       #T
k_blk_idx:   0          0     1    ...   last       last
is_first:    ✔(clear_C) ✘     ✘    ...   ✘          ✘
is_last:     ✘          ✘     ✘    ...   ✘          ✔(UPDATE)
unit_flag:   KEEP       KEEP  KEEP ...   KEEP       UPDATE
clear_C:     true       false false ...  false      false
L0C 语义:    C=0+A·B    C+=A·B C+=A·B ... C+=A·B    C+=A·B 并提交
```

最后补一个衔接点（详细机制是第 25 天）：第 129-131 行在每个子块 K 遍历**开始前**（`k_blk_idx == 0` 时）调用 `epilogue.aic_wait_free(s_epi)` 等待 epilogue 的 UB stage 空闲；第 155-163 行在**最后一个 K 块**的 K 遍历结束后把 L0C 交出去。`s_epi` 环（:107, :122, :164, :169 的 `epi_base`/`s_epi` 接力）让多个输出子块的写回也能排队重叠。今天只需要知道：K 循环的首尾正是 Epilogue 的接入点。

### 3.6 手工展开示范（当天产出的参考答案）

**设定**：取 `unaligned-k` 测试里的真实 shape **M=128, N=128, K=234**（`tests/generators.py:519-541`，该组还含 K=4/17/15/4095 等极端值）。假设 JIT 选中如下配置（满足仓库全部静态约束，且 `MAD_K` 与 `heuristics.hpp:129-142` 对 128×128 block 的推导结果一致；真实配置由 `select_gemm_config` 的完整代价模型决定，有环境时用 `DG_PRINT_CONFIGS=1` 核对）：

```text
BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, MAD_M=128, MAD_N=128, MAD_K=64
kNumL1Stages=8, kNumL0Stages=2, num_mad_cd_stages=(128/128)×(128/128)=1
```

**先算三个总数**：

```text
num_k_blocks = ceil_div(234, 128) = 2
每个 K 块的 mmad 数 = ceil(actual_k / 64)：块 0 → 2 条；块 1 → 2 条
一个输出块的总 mmad 数 = 2 × (128/128) × (128/128) × 2 = 4 条（乘上 M/N 子块数 1×1）
```

**逐 步 展 开 表**（单核单输出块；`k_idx` 为 GM 上 K 起点偏移；搬运列数即 `actual_k`）：

| 步 | k_blk_idx | k_idx | actual_k | s (L1环) | k_mad_idx | eff_mad_k | s_ab (L0环) | is_first | is_last | unit_flag | clear_C | 本步动作 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 128 | 0 | – | – | – | – | – | – | – | GM→L1：A[0:128, 0:128]、B[0:128, 0:128] → l1a[0]/l1b[0] |
| 1 | 0 | 0 | 128 | 0 | 0 | 64 | 0 | **✔** | ✘ | KEEP | **true** | L1→L0(s_ab=0)；mmad：C = 0 + A[:,0:64]·B[:,0:64] |
| 2 | 0 | 0 | 128 | 0 | 64 | 64 | 1 | ✘ | ✘ | KEEP | false | L1→L0(s_ab=1)；mmad：C += A[:,64:128]·B[:,64:128] |
| 3 | 1 | 128 | **106** | 1 | – | – | – | – | – | – | – | GM→L1：A[0:128, 128:234]、B[0:128, 128:234] → l1a[1]/l1b[1]（只搬 106 列） |
| 4 | 1 | 128 | 106 | 1 | 0 | 64 | 0 | ✘ | ✘ | KEEP | false | L1→L0(s_ab=0)；mmad：C += A[:,128:192]·B[:,128:192] |
| 5 | 1 | 128 | 106 | 1 | 64 | **42** | 1 | ✘ | **✔** | **UPDATE** | false | L1→L0(s_ab=1，tile K 对齐到 48)；mmad：C += A[:,192:234]·B[:,192:234] 并提交 |
| 6 | 1 | 128 | 106 | 1 | – | – | – | – | – | – | – | 最后一个 k_blk：epilogue.aic_store / copy_l0c_to_gm 写回；释放 L1 stage 1 |

核对几个容易算错的格子：

- **步 5 的 `eff_mad_k`**：`min(106 - 64, 64) = 42`。`is_last` 判定：`k_mad_idx + MAD_K = 128 >= 106` ✔，且 `k_blk_idx == 1 == num_k_blocks-1` ✔。
- **步 4 不是 `is_last`**：`0 + 64 = 64 >= 106` 不成立。
- **步 2、4 的 `is_first`**：`k_blk_idx != 0`，即使 `k_mad_idx == 0` 也不清零——否则 K 块 0 的部分和会被丢掉。
- **`s_ab` 的取值**：每个输出子块的 K 遍历内 0→1，步 3 进入新 K 块后从 0 重新计（本例 M/N 子块数为 1，若拆成 2×2 个子块，则每个子块各自走一遍步 1-2、4-5 的 K 遍历）。
- **L0A/L0B 占用校验**（对照第 16 天）：`l0a` 每 stage `128×64×2B = 16KB`，2 stage 共 32KB ≤ 64KB ✔；`l1a` 每 stage `128×128×2B = 32KB`，8 stage 共 256KB，加 `l1b` 同样 256KB，共 512KB ≤ L1 512KB ✔。

**对齐情形对照**（第 20 天用过的 K=256）：若 `BLOCK_K=128, MAD_K=64`，则 2 块 × 2 步 = 4 条 mmad，全部 `eff_mad_k=64`；若按 `heuristics.hpp:648-651` 的加宽规则（K 是 256 的倍数时尝试 `max_block_k` 且 stage 数 ≥3 即采纳）选中 `BLOCK_K=256`，则推导出 `MAD_K=128`，1 块 × 2 步 = 2 条 mmad。**同样的 K，不同的配置，循环结构不同**——这正是"展开前必须先确定配置"的原因。

### 3.7 K-Grouped：`eff_shape_k` 与 `k_idx_base` 从哪来

K 维两层循环对 K-Grouped GEMM（MoE 场景，A/B 是各组 K 拼接的平铺缓冲）天然友好，因为外层循环的全部控制量都从 scheduler 取：

- `eff_shape_k = scheduler.current_shape_k`（:95）：当前组的 K 长度。`get_next_block` 推进组游标时更新它（`scheduler.hpp:163-177`：组边界由 `grouped_layout` 前缀和给出，`last_psum_k = aligned(current_psum_k, kAlignment)` 后 `current_shape_k = current_psum_k - last_psum_k`）。
- `k_idx_base = scheduler.get_k_idx_base()`（:97，即 `scheduler.hpp:124` 的 `last_psum_k`）：前序各组累计的 K 起点，使 `k_idx = k_idx_base + k_blk_idx * BLOCK_K` 直接落在平铺缓冲的正确位置。
- 空组（`current_shape_k == 0`）不消耗任何 block 编号（`scheduler.hpp:174-175` 的判断），`num_k_blocks = 0`，两层循环一次都不执行。

Normal GEMM 下二者退化为 `eff_shape_k = shape_k`、`k_idx_base = 0`。这个设计让**同一个两层循环体**无需分支就能服务 Normal 与 K-Grouped。

## 四、源码导读

建议按站点顺序阅读，总计约 45 分钟。所有行号以仓库当前版本为准。

### 站点 1：两层循环主体（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:112-168`，精读）

逐段拆解（本文第三节已覆盖语义，这里给阅读地图）：

- `:112-114` 外层循环头：`k_blk_idx`、`s` 环、`k_idx`、`actual_k`；
- `:116-119` 等 L1 stage 空闲 → GM→L1 双搬运（A/B 各一次）→ 通知就绪；
- `:121-127` 等搬运完成；`s_epi` 归位到 `epi_base`；M/N 子块双重循环与 `eff_mad_m/eff_mad_n`、`l0c[s_cd]` 子块选择；
- `:129-131` 首个 K 块时等 epilogue stage 空闲（非 `kDirectStore`）；
- `:133-139` 内层循环头：`k_mad_idx`、`s_ab` 环、`is_first/is_last/eff_mad_k`、两个 `as_mad_aligned` tile；
- `:141-146` 等 L0 stage → 条件化 L1→L0A/L0B 搬运（含 `kResidentA` 偏移与 L0 复用条件）→ 通知；
- `:148-152` 等 L0 数据就绪 → `mad(...)` 消费 `is_last/is_first` → 通知 L0 stage 可复用；
- `:155-163` 最后一个 K 块：`kDirectStore` 分支 `copy_l0c_to_gm`，否则 `epilogue.aic_store`；
- `:164-169` `s_epi` 环推进；`asc_sync_notify(PIPE_MTE1, PIPE_MTE2, s)` 释放 L1 stage；输出块间传递 `epi_base`。

### 站点 2：`mad()` 封装（`deep_gemm/include/deep_gemm/ascend/copy.hpp:370-381`，精读）

确认三件事：`(m, n, k) → asc_mmad(m, k, n)` 的参数顺序调换；`unit_flag` 直通指令；`clear_C`（默认 false）对应指令的 `c_matrix_init_val`。`sizeof(TA)==1 || sizeof(TB)==1` 时改走 `asc_mmad_mx`（FP8/FP4 带 scale 的变体，第 32 天再见）。

### 站点 3：tile 对齐（`deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp:52-55` + `deep_gemm/include/deep_gemm/common.hpp:63, 73-79`，浏览）

`as_mad_aligned` 把 `shape_mn` 对齐到 `FRAC_MN=16`、`shape_k` 对齐到 `get_frac_k<T>()`（BF16 为 16）。对照 `bf16_gemm.hpp:138-139` 与 `:149`：对齐值进 tile 描述符（影响 L1→L0 搬运量），原始 `eff_mad_k` 进 `mad()`。

### 站点 4：缓冲声明与断言（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:51, 59-70`，复习）

第 51 行 `num_mad_cd_stages`；第 59-60 行 `l1a/l1b`（注意 `kResidentA` 时 `l1a` 的 `shape_k` 换成整条 K）；第 66-69 行容量断言；**第 70 行 `BLOCK_K % MAD_K == 0`**——两层循环粒度关系的钉子。

### 站点 5：配置从哪来（`csrc/jit_kernels/heuristics.hpp`，选读）

- `:618-621`：BF16 的 `min_block_k=128 / max_block_k=256` 及其动机注释；
- `:111-112`：`num_l0_stages` 恒为 2；
- `:129-142`：`MAD_K` 推导公式与两条约束（整除 `block_k`、不小于 `min(block_k, mte1_copy_k)/num_l0_stages`，后者的注释解释了"MAD_K 太小会浪费 MTE1 带宽"）；
- `:648-651`：K 整除 `max_block_k` 时尝试加宽 `BLOCK_K` 的规则；
- `:749-753`：`DG_PRINT_CONFIGS=1` / `DG_JIT_DEBUG=1` 时打印 `GemmDesc` + `GemmConfig`（实践步骤 C 的观察工具）。

对照 `csrc/jit_kernels/gemm_config.hpp:98-121`：`GemmConfig` 的 `block_k/mad_k/num_l1_stages/num_l0_stages` 字段如何进入模板实参（`csrc/jit_kernels/bf16_gemm.hpp:48-56` 的 `std::format` 实参表）。

### 站点 6：K-Grouped 的控制量来源（`deep_gemm/include/deep_gemm/scheduler.hpp:73-76, 124, 163-177`，浏览）

`current_shape_k` 的更新、`get_k_idx_base()`（`last_psum_k`）、空组跳过逻辑。

### 站点 7：同一模式的其他实现（浏览，建立"模式识别"）

- `deep_gemm/include/deep_gemm/fp8_gemm.hpp:205-229`：FP8 版两层循环，`is_first/is_last/eff_mad_k` 完全同构，只是 K 先对齐到 64（`k64`）且多出 scale 因子的搬运；
- `deep_gemm/include/deep_gemm/mqa_logits.hpp:397`：单发 `mad(..., ENABLE_UPDATE, /*clear_C=*/true)`——没有 K 循环时的退化形态，反衬 `is_first/is_last` 在多步累加中的必要性。

### 站点 8：测试入口（`tests/test_bf16_gemm.py:31-32`、`tests/generators.py:519-541`，浏览）

`python tests/test_bf16_gemm.py unaligned-k --skip-prof` 的用例来源，含本文示范用的 `(128, 128, 234)`。

## 五、实践步骤

### A. 无 Ascend 环境：纸面展开（主任务，约 55 分钟）

**A-1 基础展开（必做）**。用 3.6 节的设定（M=128, N=128, K=234；BLOCK 128×128×128，MAD 128×128×64，L1 stages=8，L0 stages=2）**先不看参考答案**，自己填写完整展开表：

1. 计算 `num_k_blocks`、每个 K 块的 `actual_k`、总 mmad 数；
2. 列出 4 条 mmad 每条的 `(k_blk_idx, k_mad_idx, eff_mad_k, s, s_ab, is_first, is_last, unit_flag, clear_C)`；
3. 写出 2 次 GM→L1 搬运各自的 GM 数据区间（K 维）与目标 stage；
4. 对照 3.6 节的表订正。

**A-2 变体演算（必做）**。只改一个变量，重新展开并记录差异：

| 变体 | 改动 | 观察重点 |
|---|---|---|
| V1 | K=384（其余同 A-1） | 3 个 K 块；`s` 环走 0,1,2；总 mmad 数 6；`is_last` 落在第 6 条 |
| V2 | K=64（其余同 A-1） | 单块 `actual_k=64`；只有 1 条 mmad，它**同时**是 `is_first` 和 `is_last`（clear_C=true 且 UPDATE） |
| V3 | MAD_M=MAD_N=64（其余同 A-1，K=234） | M/N 子块数变为 2×2=4，`num_mad_cd_stages=4`；同一个 K 块内 4 个子块各自走 K 遍历；L0B 复用条件（`BLOCK_M != MAD_M` 且 `BLOCK_K <= MAD_K*kNumL0Stages`）使 L0B 只在 `m_mad_idx==0` 时搬运——数一数一个 K 块内 L1→L0A/L0B 各搬几次 |
| V4 | `kResidentA=true`，SHAPE_K=234 | 第 143 行读偏移变为 `k_blk_idx*BLOCK_K + k_mad_idx`；GM→L1 的 A 搬运移出 K 循环（第 109-110 行） |

**A-3 源码标注（必做）**。打开 `bf16_gemm.hpp:112-168`，给每一行标注它属于"外层控制量 / 外层搬运 / M/N 子块 / 内层控制量 / 内层搬运 / mad / 写回 / stage 环推进"中的哪一类，然后用三种颜色（或三种记号）分别标出 `s`、`s_ab`、`s_cd/s_epi` 的全部出现点，直观感受三个环的转速。

**A-4 判定题自验**。对下面每个陈述判断真假并给出行号依据（答案见第七节）：

1. `eff_mad_k` 有可能大于 `MAD_K`。（ ）
2. 当 `num_k_blocks == 1` 时，第一条 mmad 既是 `is_first` 又是 `is_last`。（ ）
3. `is_last` 的条件写成 `k_mad_idx + MAD_K == actual_k` 在所有 shape 下都正确。（ ）
4. M/N 拆成多个子块时，每个子块的 K 遍历都有自己的 `is_first`。（ ）

### B. 有 Ascend 环境：观察真实配置（可选，约 20 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py unaligned-k --skip-prof
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
```

1. 从输出里找到 `(m=128, n=128, k=234)` 一类的 `GemmDesc(...)` 行及其 `GemmConfig(block=...x...x..., mad=...x...x..., l1_stages=..., l0_stages=...)` 行（由 `heuristics.hpp:749-753` 打印）；
2. 把真实配置代入 A-1 的展开表重算，与自己假设的配置对比：`BLOCK_K`、`MAD_K` 是否如 3.2 节公式所推？
3. 用 K 为 256 倍数的 shape（如 normal 组里的 4096）观察 `BLOCK_K` 是否被加宽到 256；
4. 若无环境，改为在 `heuristics.hpp:618-651` 里静态追踪：对 K=234 与 K=256 分别判断加宽分支是否触发、`make_gemm_candidate` 返回的 `mad_k` 是多少。

## 六、当天产出

**K 维循环展开图**。把 A-1/A-2 的结果整理成一页笔记，至少包含：

1. **嵌套结构图**：仿照 3.1 节的 ASCII 图，标出两层 K 循环与 M/N 子块循环的包含关系、三个 stage 环（`s`/`s_ab`/`s_cd`）挂在哪一层；
2. **逐步展开表**：至少覆盖对齐（K=384 或 256）与尾部（K=234）两种 shape，列同 3.6 节；
3. **标志时间线**：一个输出子块从 `is_first`（clear_C）到 `is_last`（UPDATE）再到写回的顺序图；
4. **变体记录**：A-2 的 V1-V4 结论（各一两行即可）；
5. **遗留问题**：交给后续天（例如"`asc_sync_wait/notify` 的配对时序"交给第 23 天，"L0 复用条件的搬运次数账本"交给第 24 天）。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. `BLOCK_K` 循环和 `MAD_K` 循环各自负责哪段数据通路？为什么不能合并成一层？
2. 给定 `K=234, BLOCK_K=128, MAD_K=64`，写出所有 `(k_blk_idx, k_mad_idx, eff_mad_k)` 三元组，并指出哪次迭代 `is_last=true`。
3. `is_first` 传给 `mad()` 后最终落到 `asc_mmad` 的哪个语义位置？它为什么必须同时检查 `k_blk_idx == 0` 和 `k_mad_idx == 0`？
4. `is_last` 的判定为什么用 `k_mad_idx + MAD_K >= actual_k` 而不是 `==`？分别举一个 `==` 成立和 `>` 成立的数值例子。
5. 一个输出块里 `is_first` 总共出现几次（设 M/N 子块数为 P×Q、K 块数为 B）？`epilogue.aic_store` 被调用几次？
6. 若把 `is_first` 误写成 `k_mad_idx == 0`（漏掉 `k_blk_idx == 0`），K=234、BLOCK_K=128 的例子会算出什么？错在第几步？
7. K-Grouped GEMM 中 `num_k_blocks` 每个输出块都一样吗？`k_idx` 相比 Normal 多了哪一项？空组时两层循环执行几圈？

### 参考答案

1. 外层管 `GM → L1`（数据搬运/流水节拍，受 L1 容量与带宽约束，BF16 典型 128/256），内层管 `L1 → L0A/L0B → mmad → L0C`（指令/计算节拍，受 L0 容量约束且须整除 `BLOCK_K`，典型 64/128）。两者粒度不同且 `BLOCK_K` 通常是 `MAD_K` 的倍数（`bf16_gemm.hpp:70` 的 `static_assert`），合并成一层就无法让 L1 搬运与 L0 计算/搬运以各自的深度流水。
2. `(0,0,64)、(0,64,64)、(1,0,64)、(1,64,42)`；`is_last` 在 `(1,64)` 成立（`64+64=128 >= 106`）。
3. `clear_C` 参数 → `asc_mmad` 的 `c_matrix_init_val`（经 `copy.hpp:371-381`）。只查 `k_mad_idx` 会让后续 K 块的首条 mmad 也清零，丢掉前面块的累加；只查 `k_blk_idx` 无法区分块内第一条——二者都必须为 0 才是"该子块 K 遍历的第一条"。
4. 满块末步两者相等（如 `actual_k=128, k_mad_idx=64`：64+64=128）；尾部块末步左大右小（如 `actual_k=106, k_mad_idx=64`：128 > 106），此时 `eff_mad_k=42 < MAD_K`。`==` 会漏掉尾部情形，最后一步永远不提交。
5. `is_first` 出现 `P × Q` 次（每个输出子块的 K 遍历各一次）；`aic_store`（或 direct store）同样 `P × Q` 次——都在最后一个 K 块结束后触发（`:155-163`）。
6. 步 3（第 2 个 K 块的 `k_mad_idx=0` 步）会 `clear_C=true`，把 K 块 0 累加的 128 列部分和清掉，最终结果只等于 K∈[128,234) 的贡献，缺了前 128 列。
7. 不一定——`num_k_blocks = ceil_div(当前组 current_shape_k, BLOCK_K)`，各组 K 不同则不同（`:95-96`）；`k_idx = k_idx_base + k_blk_idx*BLOCK_K`，比 Normal 多了 `get_k_idx_base()`（前序组累计 K 起点，`scheduler.hpp:124`）；空组 `current_shape_k=0` → `num_k_blocks=0`，两层循环零次迭代。

**A-4 判定题答案**：

1. **假**。`eff_mad_k = min(actual_k - k_mad_idx, MAD_K)`（:137），`min` 的上界就是 `MAD_K`，至多相等。
2. **假**。反例：K=128、BLOCK_K=128、MAD_K=64 时 `num_k_blocks=1`，但有 2 条 mmad，第一条只是 `is_first`。只有再满足 `actual_k ≤ MAD_K`（单条 mmad，如 V2 的 K=64）时两个标志才同时落在第一条上。
3. **假**。尾部块末步 `k_mad_idx + MAD_K > actual_k`（如 128 > 106），`==` 版本永远不成立，最后一步不会提交。
4. **真**。`is_first`/`is_last` 在 `m_mad`/`n_mad` 循环之内计算（:134-135），每个子块的 K 遍历独立判定。

---

**明日预告（第 23 天）**：今天我们反复搁置的 `asc_sync_wait` / `asc_sync_notify` 终于登场——`ascend/sync.hpp` 与 MTE1/MTE2/M 三条流水线的配对关系、L1/L0 stage 事件环的时序，以及"同步过少产生竞争、过多降低并行度"的权衡。
