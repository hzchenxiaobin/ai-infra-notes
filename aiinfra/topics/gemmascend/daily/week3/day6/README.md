# 第 3 周 · 第 6 天（总第 20 天）：完整演算一个输出 Tile（复盘日）

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 3 周"BF16 Kernel——模板、调度与数据搬运"的第 6 天（复盘日）教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 30 分钟，完整演算约 70 分钟，源码核对约 30 分钟，实践与自评约 40 分钟。

本周前五天把 BF16 Kernel 拆成了五块：模板参数（第 15 天）、片上内存规划（第 16 天）、Normal 调度（第 17 天）、地址与 Tile 表达（第 18 天）、GM→L1→L0 搬运（第 19 天）。每天只看一块的好处是聚焦，代价是容易"只见零件不见机器"。今天不引入任何新内容，按学习计划的要求做一次**完整演算**：选择 `M=128、N=128、K=256`，假设一组 Block/MAD 参数，从"哪个核领任务"开始，到"D 写回 GM"结束，把这个 GEMM 唯一一个输出 Tile 的完整生命周期在纸上走一遍——输出 Tile 数、K 维循环次数、每一次搬运的数据区域、边界 Tile、片上缓冲的使用，最后逐项对照源码核对。这个 shape 的全部数据量只有 64 KiB + 64 KiB 输入和 32 KiB 输出，是能被"手工全trace"的最小真实规模。

**通过标准**：能把调度、地址计算和数据搬运串成一个完整过程。

复盘方法与前两周相同：判断掌握程度的唯一标准是**闭卷复述**。第三节是参考答案——请先自己算，再对照订正；每一步都标注了数字的代码出处，"对照源码检查自己的演算"就落实在这些出处上。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 对给定 shape 与假设配置，**闭卷算出**输出 Tile 数、wave 数、每个核领到的任务（`scheduler.hpp:91-93、142、190-195`），并解释本例为什么恰好只有一个核干活；
2. 算出 **K 维循环的完整结构**：外层 K 块循环次数（`num_k_blocks`，`bf16_gemm.hpp:96`）、每个 K 块内的 MAD 段数（`bf16_gemm.hpp:133`）、MAD 总次数，并用 FLOP 守恒验证；
3. 对**每一次搬运列出四要素**（源地址、目标地址、实际大小、stride）：4 次 GM→L1、8 次 L1→L0，地址全部由 `gm_ptr::index()`（`common.hpp:160`）与 `operand_ptr::offset()`（`tile_ptr.hpp:38`）推出；
4. 画出**片上缓冲使用图**：L1/L0A/L0B/L0C/UB 各级哪些 stage 被触碰、哪些空闲，并用 `static_assert`（`bf16_gemm.hpp:66-70`）验算容量；
5. 说明四次 `mad()` 调用各自的 `is_first/is_last`、`ENABLE_UPDATE/ENABLE_KEEP` 与 `clear_C` 取值（`bf16_gemm.hpp:134-135、149-151`），以及 L0C 从清零、累加到定稿的生命周期；
6. 描述 Epilogue 写回路径：`aic_store`（L0C→UB，FP32→BF16 量化）→ `aiv_drain`（UB→GM），并判断哪个 AIV 认领了本例的 Tile（`epilogue.hpp:42-47、49-87`）；
7. **标出边界 Tile**：说明本例为什么没有任何边界 Tile，并对 M=160、N=100、K=200 三个变体算出 `actual_*`、`as_mad_aligned` 后的形状与 `eff_mad_*`；
8. 把本周五天的结论串成一条时间线，指出每个数字来自哪个文件的哪一行，并列出仍不确定的问题（交给第 23～26 天）。

**当天产出：一个输出 Tile 的完整生命周期演算记录**（模板见第六节，参考答案见第三节）。

## 二、知识梳理（先复盘，再演算）

### 2.1 本周一览：五天各回答了什么问题

| 天 | 回答的问题 | 核心结论 | 在今天演算中的位置 |
| :--: | :-- | :-- | :-- |
| 15 | Kernel 长什么样？ | 26 个模板参数分七类；`SHAPE_M/N/K=0` 是"用运行期 shape"的哨兵（`bf16_gemm.hpp:47-49`）；`kDirectStore/kResidentA/B` 是路径开关 | 3.0 的假设配置表 |
| 16 | 数据放在片上哪里？ | `l1a/l1b/l0a/l0b/l0c` 五个 `constexpr` 指针对象（`bf16_gemm.hpp:59-64`），多级缓冲靠 `offset(n) = addr + n × size_per_stage()` 划分（`tile_ptr.hpp:37-38`），容量由 `static_assert` 把关（66-70 行） | 3.3 的目标地址、3.7 的缓冲使用图 |
| 17 | 谁算哪个 Tile？ | 持久化调度：第 t 次调用领全局块号 `t × kNumCores + core_id`（`scheduler.hpp:142`）；`get_global_idx` 把块号变成全局坐标（113-115 行）；`get_actual_m/n` 截断尾块（129-139 行） | 3.1 的任务分配 |
| 18 | 坐标怎么变地址？ | K-major 下 `index(mn, k)` 偏移 = `sizeof(T) × (mn × stride_outer + k)`（`common.hpp:160`）；`as_mad_aligned` 把形状向上对齐到 `FRAC_MN/FRAC_K`（`tile_ptr.hpp:52-55`）；L1 是 NZ 分形摆放，L0 永远是 MAD 要求的固定分形 | 3.3 的源地址、3.4 的分形坐标 |
| 19 | 数据怎么搬？ | GM→L1 用 `copy_gm_to_l1`（nd2nz，MTE2，`copy.hpp:120-124`）；L1→L0 用 `copy_l1_to_l0a/b`（按分形单元，MTE1，`copy.hpp:136-138`）；`actual_m/n/k` 三级裁剪链处理边界 | 3.3/3.4 的四要素表、3.8 的边界变体 |

### 2.2 演算前要带在手边的六个公式

演算中反复用到下面六个式子，全部来自本周读过的源码（独立阅读的读者也可只用这六个公式跟上本文）：

```text
① ceil_div(a, b)      = (a + b - 1) / b                      common.hpp:41
② aligned(a, b)       = (a + b - 1) / b * b                  common.hpp:48
③ GM 源偏移（K-major） = sizeof(T) × (mn × stride_outer + k)  common.hpp:160
④ size_per_stage      = shape_mn × shape_k × sizeof(T)        tile_ptr.hpp:37
   offset(n)          = addr + n × size_per_stage()           tile_ptr.hpp:38
⑤ actual_m/n          = min(shape − blk × BLOCK, BLOCK)       scheduler.hpp:134、138
   actual_k           = min(K − k_blk × BLOCK_K, BLOCK_K)     bf16_gemm.hpp:114
⑥ as_mad_aligned      = (aligned(mn, 16), aligned(k, 16))     tile_ptr.hpp:52-55
   （BF16：FRAC_MN = FRAC_K = 16，common.hpp:63、77）
```

### 2.3 一条时间线串起五天

把五天的知识按执行顺序排开，就是今天要走的六步：

```text
领任务(17) → 坐标与 actual(17) → GM→L1 搬运(18+19) → L1→L0 搬运(18+19)
           → mad 累加(16) → store/drain 写回(16)
```

每一步都会产生一组可核对的数字。下面进入正题。

## 三、详细讲解：完整演算 M=128、N=128、K=256

### 3.0 设定：问题与假设配置

**问题**：`D[128, 128] = A[128, 256] × Bᵀ`，即 `bf16_gemm_nt` 语义——A、B 都是 K-major 紧凑 BF16 张量（`stride_outer = K = 256`），D 是 `[M, N]` 行主序 BF16，无累加、无 bias、Identity epilogue。数据量：A = 128×256×2 = 65536 B，B 同，D = 128×128×2 = 32768 B；总 FLOP = 2×128×128×256 = 8388608。

**假设配置**（按学习计划"假设一组 Block/MAD 参数"；沿用第 19 天的假设以保持布局数字一致）：

| 参数 | 假设值 | 代码角色 | 源码锚点 |
| :-- | :-- | :-- | :-- |
| `BLOCK_M/N/K` | 128 / 128 / 128 | Tile 尺寸，L1 stage 按 BLOCK×BLOCK_K 规划 | `bf16_gemm.hpp:59-60` |
| `MAD_M/N/K` | 128 / 128 / 64 | MAD 单元一次消费的尺寸，L0 stage 按 MAD×MAD_K 规划 | `bf16_gemm.hpp:62-63` |
| `kNumL1Stages` | 3 | L1 stage 环深度 | `bf16_gemm.hpp:59-60` |
| `kNumL0Stages` | 2 | L0 stage 环深度（源码恒取 2） | `heuristics.hpp:111-112` |
| `kNumEpilogueStages` | 1 | 每个 AIV 的 UB stage 数（`kNumAIVs = 2`，`bf16_gemm.hpp:54`） | `epilogue.hpp:24` |
| `kNumCores` | 32 | 持久化调度的核数（950DT） | `common.hpp:51` |
| `kDirectStore / kResidentA / kResidentB` | 均为 false | 走普通 Epilogue 路径 | 第 26 天展开 |

**假设的合法性核对**（"对照源码检查"的第一批检查点，依据 `heuristics.hpp` 的 `make_gemm_candidate`）：

- **L0 级数**：源码固定 `c.num_l0_stages = 2`（`heuristics.hpp:111-112` 注释"always 2 stage"）——与假设一致；
- **MAD 搜索**：对 block 128×128×128、BF16、双 K-major：`k_alignment = get_mad_alignment_k(16) = 32×8/16 = 16`（`heuristics.hpp:42-44`）；首个 split `(1,1)` 即被接受——`mad_m×mad_n×4 = 65536 ≤ L0C/2 = 131072`（127 行），`l0a_k = l0b_k = 64K×8/(2×128×16) = 128`，`mad_k = min(128, 128, max(16, 64)) = 64`，能整除 `block_k` 且通过 142 行的下界检查——**MAD = 128×128×64 正是搜索结果**（124-147 行）；
- **Epilogue 级数**：`num_cd_stages = 1`，`epilogue_bytes = 128×128×2 = 32768`；`num_epilogue_stages = min(min(16, 256K/32K), ceil_div(num_blocks=1, 32)×1) = 1`（`heuristics.hpp:187-193`）——与假设一致；
- **L1 级数**：BF16 无 SF，公式给出 `min(num_pipe_events=8, 512K / ((128+128)×128×2)) = 8`（`heuristics.hpp:167-174`）。本文假设 3 级是为了与第 19 天的 L1 布局数字（`l1b` 基址 98304）保持一致；**级数不影响本例的任何搬运内容**——`num_k_blocks = 2 < 3`，无论配 3 级还是 8 级，被触碰的都只有 stage 0 和 1（见 3.2/3.7）。若真用 8 级，`l1b` 基址变为 `l1a.offset(8) = 262144`，`l1b.offset(8) = 524288` 恰好压线通过 66 行的容量断言。

> 真实 `select_gemm_config` 还要经过成本模型与收缩精修（`heuristics.hpp:626-712`），打印出的配置未必是 128×128×128——这正是实践步骤 A-1 要用 `DG_PRINT_CONFIGS=1` 验证的事。

### 3.1 第 1 步：输出 Tile 数与任务分配（调度）

代入公式 ①⑤（`scheduler.hpp:91-93`）：

```text
num_m_blocks = ceil_div(128, 128) = 1
num_n_blocks = ceil_div(128, 128) = 1
num_blocks   = 1 × 1 = 1            → 输出 Tile 数 = 1
num_waves    = ceil_div(1, 32) = 1
```

整个 GEMM 就是**一个 Tile 的生命周期**。任务分配（`scheduler.hpp:142、190-195`）：

| 核 | 第 1 次 `get_next_block` 的 block_idx | 结果 |
| :-- | :-- | :-- |
| 核 0 | `0 × 32 + 0 = 0 < 1` | 领到 Tile，进入 K 循环 |
| 核 1～31 | `0 × 32 + core_id = 1..31 ≥ 1` | 返回 `false`，立即退出 |

核 0 领到块号 0 后走 `get_swizzled_block`：`local_num_m_blocks = 1 < kGroupM = 4`，命中**快路径**（`scheduler.hpp:209-211`）——`m_block_idx = 0 / 1 = 0`，`n_block_idx = 0 % 1 = 0`，混洗逻辑（superrow/XOR/蛇形）根本不会执行。随后（`bf16_gemm.hpp:90-91、105-106`）：

```text
m_idx = get_global_idx(128, 0) = 0 × 128 = 0
n_idx = get_global_idx(128, 0) = 0
actual_m = min(128 − 0, 128) = 128        actual_n = min(128 − 0, 128) = 128
```

每个 block 是 1 个 AIC + 2 个 AIV（`__mix__(1, kDirectStore ? 0 : 2)`，`bf16_gemm.hpp:27`）。空 block 的 AIC 第一次领任务就退出；其 AIV 执行 `aiv_initialize()` 后同样在 `while (get_next_block)` 的第一次判断处退出（`bf16_gemm.hpp:177-180`）。**本例实际参与计算的只有 block 0 的 1 个 AIC 和 2 个 AIV（其中还有 1 个 AIV 全程空闲，见 3.6）**——这就是第 17 天说的"小矩阵浪费核"的极端形态。

### 3.2 第 2 步：K 维循环次数

`num_k_blocks = ceil_div(256, 128) = 2`（`bf16_gemm.hpp:96`）。循环变量与 stage 环（112 行的轮转式 `s = s == kNumL1Stages-1 ? 0 : s+1`）：

| k_blk_idx | k_idx | actual_k（114 行） | L1 stage `s` |
| :-- | :-- | :-- | :-- |
| 0 | 0 | min(256−0, 128) = 128 | 0 |
| 1 | 128 | min(256−128, 128) = 128 | 1 |

每个 K 块内的 MAD 段数 = `ceil_div(actual_k, MAD_K) = ceil_div(128, 64) = 2`（133 行）。M/N 方向的 MAD 循环（123-124 行）以 `actual_m/actual_n` 为界、步进 `MAD_M/MAD_N`——`128 = MAD_M = MAD_N`，各只有 1 轮。所以：

```text
MAD 总次数 = num_k_blocks × K 段数 × m_mad 轮数 × n_mad 轮数 = 2 × 2 × 1 × 1 = 4
FLOP 守恒：4 × (2×128×128×64) = 8388608 = 2×M×N×K ✓
```

**注意 stage 2 空闲**：`num_k_blocks = 2 < kNumL1Stages = 3`，L1 的第 3 个 stage 全程未被触碰——stage 环的深度是为"K 块更多时搬运与计算重叠"准备的，本例用不满（第 23 天会看到重叠条件）。

### 3.3 第 3 步：四次 GM→L1 搬运

每推进一个 K 块，A、B 各发生一次 GM→L1（`bf16_gemm.hpp:117-118`）。源地址用公式 ③（`common.hpp:160`，`stride_outer = K = 256`），目标地址用公式 ④（`tile_ptr.hpp:37-38`；`l1b` 基址 = `l1a.offset(3) = 3 × 32768 = 98304`，`bf16_gemm.hpp:60`）：

| # | k_blk (s) | 操作数 | GM 源（相对张量基址） | L1 目标 | 实际大小 | 源行距 | NZ `c0_stride` |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 0 (s=0) | A | `a.index(0, 0)` = **+0 B** | `l1a[0]` @ L1 **+0** | 128 行 × 128 列 = 32768 B | 512 B | 128 |
| 2 | 0 (s=0) | B | `b.index(0, 0)` = **+0 B** | `l1b[0]` @ L1 **+98304** | 32768 B | 512 B | 128 |
| 3 | 1 (s=1) | A | `a.index(0, 128)` = 2×(0×256+128) = **+256 B** | `l1a[1]` @ L1 **+32768** | 32768 B | 512 B | 128 |
| 4 | 1 (s=1) | B | `b.index(0, 128)` = **+256 B** | `l1b[1]` @ L1 **+131072** | 32768 B | 512 B | 128 |

三个观察：

1. **搬运 #3/#4 的源不是"紧跟在 #1/#2 之后"的连续 32 KiB**：`nd2nz` 按二维区域搬——第 r 行取 GM `[base + 256 + r×512, +256B)`，即每行 256 B、行距 512 B 的跨步块（`copy.hpp:122-124` 的 `n=actual_mn、d=actual_k` 与 `stride = src.stride_outer × sizeof(T)`）。A 的两个 K 块合起来恰好覆盖 A 的全部 64 KiB，每个元素只读一次——单 Tile 无重读。
2. 每次搬运后，L1 stage 内呈第 19 天反推出的 **NZ 摆放**：8 个 K-slab × 128 行 × 32 B = 32768 B；`set_mte2_nz(1, 1, dst.shape_mn=128, 0)`（`copy.hpp:121`、`14-21`）。
3. 本例 `actual_mn × actual_k = BLOCK × BLOCK_K`，**搬运量 = 缓冲容量**，不存在"未更新的旧数据"区域——这是选这个 shape 的好处之一（对照 3.8）。

### 3.4 第 4 步：八次 L1→L0 搬运

在每个 `(k_blk, k_mad)` 组合处，A、B 各发生一次 L1→L0（`bf16_gemm.hpp:143、145`）。六标量来自 `copy.hpp:136-138`：`x_start = m_mad_idx/16 = 0`，`y_start = k_mad_idx/16`，`x_step = 128/16 = 8`，`y_step = 64/16 = 4`，`src_stride = dst_stride = 128/16 = 8`。源分形的 stage 内字节起点 = `y_start × shape_mn × 32`（第 19 天的 NZ 布局公式）：

| k_blk | k_mad (s_ab) | A：L1 源 | A：L0 目标 | B 同构 | 分形数 (x×y) | 每次 B 数 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 0 (s_ab=0) | `l1a[0]` 分形 (0, **0**) = +0 B | `l0a[0]` @ **0** | `l1b[0]` (0,0) → `l0b[0]` | 8 × 4 | 16384 B |
| 0 | 64 (s_ab=1) | `l1a[0]` 分形 (0, **4**) = +16384 B | `l0a[1]` @ **+16384** | `l1b[0]` (0,4) → `l0b[1]` | 8 × 4 | 16384 B |
| 1 | 0 (s_ab=0) | `l1a[1]` 分形 (0, **0**) | `l0a[0]` | `l1b[1]` (0,0) → `l0b[0]` | 8 × 4 | 16384 B |
| 1 | 64 (s_ab=1) | `l1a[1]` 分形 (0, **4**) | `l0a[1]` | `l1b[1]` (0,4) → `l0b[1]` | 8 × 4 | 16384 B |

要点：

- **两级 stage 环独立轮转**：L1 的 `s` 随 `k_blk_idx` 走（0→1），L0 的 `s_ab` 随 `k_mad_idx` 走（0→1→0→1）——L0A/L0B 是真正的双缓冲：MAD 正在消费 `l0a[0]` 时，MTE1 可以预填 `l0a[1]`；
- **复用条件本例全部为真、一次不少地搬**：143 行的 `BLOCK_N == MAD_N`（128==128 ✓）让 A 每个 k_mad 段都搬；145 行的 `BLOCK_M == MAD_M`（✓）让 B 同样全搬。`if` 的省搬运效果要等 `BLOCK` 与 `MAD` 在 M/N 方向出现差异（多 MAD Tile）时才会体现——本例是它的退化情形；
- `l0a_tile/l0b_tile` 的形状字段经 `as_mad_aligned(128, 64)` 后不变（128、64 均已是 16 的倍数，公式 ⑥）；
- L1→L0 总流量 = 8 × 16384 = 131072 B = A、B 各 64 KiB——MN 方向没有 MAD 拆分，每个元素恰好跨过一次 L1→L0。

### 3.5 第 5 步：四次 MAD 与 L0C 的生命周期

每次搬运对（L0A+L0B 就绪）后发起一次 `mad`（`bf16_gemm.hpp:149-151`），规模 `128 × 128 × 64`。两个标志（134-135 行）决定 L0C 的行为：

| mad# | k_blk | k_mad | `is_first`（k_blk==0 && k_mad==0） | `is_last`（k_blk==末块 && k_mad+64≥actual_k） | unit_flag | `clear_C`（= is_first） | 语义 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 0 | 0 | ✓ | ✗（0 ≠ 1） | `ENABLE_KEEP` | `true` | 清零 L0C 后累加第 1 段 |
| 2 | 0 | 64 | ✗ | ✗ | `ENABLE_KEEP` | `false` | 继续累加 |
| 3 | 1 | 0 | ✗ | ✗ | `ENABLE_KEEP` | `false` | 继续累加 |
| 4 | 1 | 64 | ✗ | ✓（1==1 且 64+64=128 ≥ 128） | `ENABLE_UPDATE` | `false` | 末段，结果定稿到 L0C |

L0C 的生命周期一图流（1 个 Tile = 128×128 FP32 = 65536 B，`num_mad_cd_stages = (128/128)×(128/128) = 1`，`bf16_gemm.hpp:51、64`）：

```text
mad#1        mad#2        mad#3        mad#4
clear_C ──► KEEP累加 ──► KEEP累加 ──► UPDATE定稿
 L0C=0        +A₀B₀ᵀ        +A₁B₁ᵀ        +A₂B₂ᵀ        L0C = Σ AkBkᵀ（FP32）
 (k_blk0段0)  (k_blk0段1)   (k_blk1段0)   (k_blk1段1)
```

两个细节：① `mad` 封装的参数序是 `(shape_m, shape_n, shape_k)`，而底层 `asc_mmad` 收到的是 `(shape_m, shape_k, shape_n)`（`copy.hpp:371-381`，378 行）——读硬件文档时注意换序；② `ENABLE_UPDATE/ENABLE_KEEP` 的硬件级语义（何时对外可见、何时仅内部累加）留到第 24 天，今天只记住调用形态：**非末段 KEEP、末段 UPDATE、首段附带 clear_C**。

### 3.6 第 6 步：Epilogue 写回

mad#4 之后，`k_blk_idx == num_k_blocks - 1` 成立，进入收尾（`bf16_gemm.hpp:155-163`）。普通路径（`kDirectStore = false`）两跳：

**第一跳：AIC 侧 `epilogue.aic_store(l0c_tile, s_epi=0, 128, 128)`**（161 行 → `epilogue.hpp:42-47`）：

- `copy_l0c_to_ub(ub[0], l0c_tile, 128, 128, DUAL_DST_DISABLE, aiv_id=0)`——L0C 的 FP32 结果在这次搬运中量化为 BF16（默认 `quant_mode = infer_quant_mode<bfloat16_t, float>() = F322BF16`，`copy.hpp:236-238、266-285`）；
- 目标 `ub[0]`：`ub{MAD_M, MAD_N}` 的 stage 0，`stage_idx = s_epi / kNumAIVs = 0`，容量 128×128×2 = 32768 B（`epilogue.hpp:24-25`）；
- 随后 `set_intra_block(PIPE_FIX, 0, 0)` 通知 AIV（46 行）。

**第二跳：AIV 侧 `epilogue.aiv_drain(d, 0, 0, 128, 128, s_epi=0)`**（`bf16_gemm.hpp:190` → `epilogue.hpp:49-87`）：

- 两个 AIV 都会进入 drain 循环，但认领规则是 `aiv_id == 2 − kNumAIVs + s_epi % kNumAIVs`（43、57 行）——`s_epi = 0` → **AIV0 认领，AIV1 在 57 行 `continue` 直接跳过**。本例 AIV1 全程空闲；
- AIV0 等 `wait_intra_block(PIPE_MTE3, 0, 0)`（69 行）后，`copy_ub_to_gm(d.index(0, 0).ptr(), output_ub, 128, 128, 256, ...)`（81-82 行）：128 行 × 每行 256 B，目的行距 256 B（`d.stride_outer = N = 128` 个元素），把 D 的 `[0:128, 0:128]` 共 32768 B 写回 GM。

收尾状态：`s_epi` 推进到 1（164 行），`epi_base = 1`、`last_m_idx = last_n_idx = 0`（169-171 行）——本例没有下一个 Tile，这些状态不再被消费。AIC 主循环结束，核 0 的 `get_next_block` 返回 `false`，kernel 退出。

### 3.7 片上缓冲使用总览（含容量验算）

```text
L1（512 KiB，假设 3 级）                                L0A/L0B（各 64 KiB = 2 × 16 KiB）
  l1a  stage0 [0, 32768)      ← k_blk0 的 A   ★使用      l0a[0] ← 偶数 k_mad 段（0）  ★双缓冲轮转
       stage1 [32768, 65536)   ← k_blk1 的 A   ★使用      l0a[1] ← 奇数 k_mad 段（64） ★（s_ab 环）
       stage2 [65536, 98304)                  空          l0b[0]/l0b[1] 同构
  l1b  stage0 [98304, 131072)  ← k_blk0 的 B   ★使用
       stage1 [131072, 163840) ← k_blk1 的 B   ★使用    L0C（256 KiB）
       stage2 [163840, 196608)                空          1 × 128×128 FP32 = 64 KiB
                                                        （clear → KEEP×2 → UPDATE）

UB（256 KiB）
  stage0 = 32 KiB BF16，AIV0 排水；AIV1 空闲；kNumStages = 1×2 = 2（epilogue.hpp:24）
```

用 `static_assert`（`bf16_gemm.hpp:66-70`）验算假设配置：

| 断言 | 代入 | 结果 |
| :-- | :-- | :-- |
| `l1b.offset(3) ≤ L1SizeBytes` | 98304 + 3×32768 = 196608 ≤ 524288 | ✓（余量 327680 B） |
| `l0a.offset(2) ≤ L0ASizeBytes` | 2×16384 = 32768 ≤ 65536 | ✓（恰好一半） |
| `l0b.offset(2) ≤ L0BSizeBytes` | 同上 | ✓ |
| `l0c.offset(1) ≤ L0CSizeBytes` | 65536 ≤ 262144 | ✓ |
| `BLOCK_K % MAD_K == 0` | 128 % 64 = 0 | ✓ |

**流量对账**（守恒检查）：

| 量 | 算式 | 值 |
| :-- | :-- | :-- |
| GM 读 A | 2 × 32768 | 65536 B（= 128×256×2，每元素一次） |
| GM 读 B | 2 × 32768 | 65536 B |
| GM 写 D | 128×128×2 | 32768 B |
| L1→L0 流量 | 8 × 16384 | 131072 B（A、B 各一次过） |
| FLOP | 4 × 2×128×128×64 | 8388608 = 2×M×N×K ✓ |
| 算术强度 | 8388608 / 163840 | 51.2 FLOP/B |

### 3.8 边界 Tile：本例为何没有，以及三个变体

**本例没有任何边界 Tile**：`BLOCK_M=128 | M=128`、`BLOCK_N=128 | N=128`、`BLOCK_K=128 | K=256=2×128`，三个维度的 `actual_*` 恒等于 `BLOCK_*`，第 19 天的三级裁剪链全程不触发。这正是学习计划选这个 shape 做"完整首演"的原因——先把主路径走通，边界行为用变体练习。下表三个变体**只改一个维度**、其余假设不变（答案已按公式 ①⑤⑥ 核对）：

| 变体 | 网格/循环变化 | 尾块 actual | `as_mad_aligned` 后 | 传给 `mad` 的 eff | epilogue 写回 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **M=160** | `num_m_blocks=2`、`num_blocks=2`：核 0 领 (m0,n0)、核 1 领 (m1,n0)（仍走快路径，2 < kGroupM=4） | Tile(m1) 的 `actual_m = min(160−128, 128) = 32` | `shape_mn = aligned(32,16) = 32` → `x_step = 2` | `eff_mad_m = 32` | `min(32,128) = 32` 行（D 的第 128..159 行） |
| **N=100** | 网格不变（`num_n_blocks=1`） | `actual_n = min(100, 128) = 100` | `shape_mn = aligned(100,16) = 112` → `x_step = 7` | `eff_mad_n = 112`（对齐值，L0C→UB 行距要求） | `min(100,128) = 100` 列 |
| **K=200** | `num_k_blocks = 2`，k_blk1 的 `actual_k = 72` | `actual_k = 72` | 末段 `shape_k = aligned(8,16) = 16` → `y_step = 1` | k_mad 0 段 `eff_mad_k=64`；k_mad 64 段 `eff_mad_k=8` | — |

**K=200 的 K 尾块细看**（三级裁剪链完整回放，对照第 19 天 3.6 节）：

- 第一级（GM→L1 少搬）：k_blk1 只搬 128 行 × 72 列（每行 144 B），第 5 个 K-slab（覆盖 k 64..79）只有前 8 个槽位被更新；
- 第二级（L1→L0 整分形搬）：末段 `y_start = 64/16 = 4`、`y_step = 1`——整个 slab 4 被搬进 L0A，其中 k 72..79 的 8 个槽位是**旧数据**；
- 第三级（计算与写回按有效量）：`mad` 收到 `shape_k = eff_mad_k = 8`，旧数据被排除在计算之外；`is_last` 判定 `64 + 64 = 128 ≥ 72` ✓ 成立，末段直接 `ENABLE_UPDATE`。
- 源码里的同类测试：`tests/generators.py:519-541` 的 unaligned-k 列表（如 `(128, 128, 234)`，524 行）——234 = 128 + 106，尾块 `actual_k = 106`，走的正是同一条链。

### 3.9 时间线总图（当天产出核心）

把 3.1～3.6 串成一张图——这就是"一个输出 Tile 从领任务到写回 GM"的完整生命周期：

```text
核 0（wave 0，唯一干活的核）                          片上缓冲
═══════════════════════════════════════════════════════════════════════════
领任务 (m0, n0)：m_idx=0, n_idx=0, actual_m=actual_n=128   [scheduler.hpp:142,209]
│
├─ k_blk 0（s=0）                                       L1 stage 0
│   GM→L1  A[0:128,   0:128]（+0B，32KiB）──► l1a[0]     [copy_gm_to_l1, MTE2]
│   GM→L1  B[0:128,   0:128]（+0B，32KiB）──► l1b[0]
│   ├─ k_mad 0  (s_ab=0)：l1a[0]分形(0,0) ─► l0a[0]      [copy_l1_to_l0a/b, MTE1]
│   │                       l1b[0]分形(0,0) ─► l0b[0]      │ mad#1 KEEP + clear_C
│   └─ k_mad 64 (s_ab=1)：l1a[0]分形(0,4) ─► l0a[1]      │ mad#2 KEEP
│                          l1b[0]分形(0,4) ─► l0b[1]      ▼ L0C 累加
├─ k_blk 1（s=1）                                       L1 stage 1
│   GM→L1  A[0:128, 128:256]（+256B）────► l1a[1]
│   GM→L1  B[0:128, 128:256]（+256B）────► l1b[1]
│   ├─ k_mad 0  (s_ab=0)：l1a[1]分形(0,0) ─► l0a[0]        │ mad#3 KEEP
│   └─ k_mad 64 (s_ab=1)：l1a[1]分形(0,4) ─► l0a[1]        │ mad#4 UPDATE ← is_last
│                                                          ▼
└─ 收尾：aic_store L0C ─(F322BF16)─► UB[0] ─► AIV0 drain ─► GM 的 D[0:128, 0:128]
   （AIV1 空闲；核 1..31 的 AIC/AIV 已提前退出）

统计：4 次 GM→L1（128 KiB）+ 8 次 L1→L0（128 KiB）+ 4 次 mad（8.39 MFLOP）+ 1 次 L0C→UB + 1 次 UB→GM（32 KiB）
```

## 四、源码导读：把每个数字对回代码

今天源码导读的目标与往常相反：不是从代码里挖新知识，而是**给 3.1～3.7 的每个数字找出处**。每站 5～10 分钟，带着"核对什么"去读。

### 站点 1：`scheduler.hpp:87-108、141-142、190-211`（核对 3.1）

- 91-93 行：三个网格常量的初始化列表——代入 `128/128/128` 应得 `1×1=1`；
- 142 行：`block_idx = (++current_iter) * kNumCores + core_id`——核 0 首次得 0，核 1..31 得 1..31；
- 190-195 行：Normal 分支的 `block_idx >= num_blocks` 早退；
- 209-211 行：快路径 `local_num_m_blocks < kGroupM` 时的行主序映射——本例唯一走过的映射分支。

### 站点 2：`scheduler.hpp:113-139`（核对 3.1 的坐标与 actual）

- 113-115 行：`get_global_idx = block_idx × block_size`——`0 × 128 = 0`；
- 129-139 行：`get_actual_m/get_actual_n` 的 `min(...)` 公式——代入得 128/128。

### 站点 3：`bf16_gemm.hpp:95-97、112-114`（核对 3.2）

- 96 行：`num_k_blocks = ceil_div(eff_shape_k, BLOCK_K) = 2`；
- 112 行：`s` 的轮转式——0→1，stage 2 永不触达；
- 114 行：`actual_k = min(256 − k_blk×128, 128) = 128`。

### 站点 4：`bf16_gemm.hpp:51、59-70`（核对 3.7 的全部地址与断言）

- 59-60 行：`l1a(128, 128, 0)`、`l1b(128, 128, l1a.offset(3))`——`size_per_stage = 32768`、`l1b` 基址 98304 的出处；
- 62-64 行：`l0a(128, 64)`、`l0b(128, 64)`、`l0c(128, 128)`——L0 stage 16384 B、L0C Tile 65536 B；
- 51 行：`num_mad_cd_stages = (128/128)×(128/128) = 1`；
- 66-70 行：五条 `static_assert`——逐条代入 3.7 的验算表。

### 站点 5：`common.hpp:156-165` + `tile_ptr.hpp:37-43、52-55`（核对 3.3/3.4 的地址）

- 160 行：K-major 的 `index` 偏移公式——`a.index(0,128) = 2×(0×256+128) = +256 B`；
- 37-38 行：`size_per_stage()` 与 `offset()`——所有 stage 基址的来源；
- 52-55 行：`as_mad_aligned` 对齐到 `FRAC_MN=16 / FRAC_K=16`——本例 (128,64) 不变，变体里 (8→16、100→112) 会用到。

### 站点 6：`ascend/copy.hpp:14-21、101-131、133-159`（核对 3.3/3.4 的搬运参数）

- 120-124 行：K 分支的 `set_mte2_nz(1, 1, 128, 0)` 与 `nd2nz` 的 `n=128、d=128、stride=512B`；
- 136-138 行：六标量 `x_start/y_start/x_step/y_step/src_stride/dst_stride = 0/4/8/4/8/8`（k_mad=64 时）；
- 147-159 行：`copy_l1_to_l0b` 与 l0a 版逐行同构。

### 站点 7：`bf16_gemm.hpp:123-152`（核对 3.4/3.5）

- 123-124 行：m/n 方向 MAD 循环以 `actual_m/actual_n` 为界——本例各 1 轮；
- 133-135 行：`s_ab` 环、`is_first/is_last` 的定义；
- 137-139 行：`eff_mad_k = min(128−k_mad, 64) = 64`、两次 `as_mad_aligned`；
- 141-146 行：两条复用 `if`——代入 `BLOCK_N==MAD_N`、`BLOCK_M==MAD_M` 均真；
- 149-151 行：`mad(..., 128, 128, 64, is_last ? UPDATE : KEEP, is_first)`。

### 站点 8：`bf16_gemm.hpp:155-171` + `epilogue.hpp:24-25、42-47、49-87`（核对 3.6）

- 155-163 行：末 K 块的 store 分支（`kDirectStore=false` → `aic_store`）；
- `epilogue.hpp:24`：`kNumStages = kNumStagesPerAIV × kNumAIVs = 1×2 = 2`；
- 43 行：`aiv_id = 2 − kNumAIVs + s_epi % kNumAIVs = 0`——AIV0 认领的出处；
- 57 行：`continue`——AIV1 跳过；
- 69、81-82 行：`wait_intra_block` 与 `copy_ub_to_gm(…, 128, 128, 256, …)`。

### 站点 9：`csrc/jit_kernels/heuristics.hpp:111-112、124-147、167-174、187-193`（核对 3.0 的假设）

- 111-112 行：`num_l0_stages` 恒 2；
- 124-147 行：MAD 搜索——对 128×128×128 BF16 得 `mad = 128×128×64`；
- 167-174 行：L1 级数容量公式——允许至 8 级，本文假设 3；
- 187-193 行：epilogue 级数——`min(8, ceil_div(1,32)×1) = 1`。

**阅读检查点**（能回答再进入实践）：

1. `scheduler.hpp:91-93` 代入本例后三个常量各是多少？
2. `bf16_gemm.hpp:60` 中 `l1b` 的基址表达式是什么？代入得多少？
3. `copy.hpp:137-138` 的六个标量在 k_mad=64 时各是多少？
4. 四次 `mad` 中 `is_last` 在哪几次为真？判定式的两个条件分别是什么？
5. `epilogue.hpp:57` 的 `continue` 让哪个 AIV 跳过了本例的 Tile？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：用真实配置重演主例（约 20 分钟）。**

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python - <<'EOF' 2>&1 | tee /tmp/day20_run1.log
import torch, torch_npu  # noqa
import deep_gemm
a = torch.randn(128, 256, dtype=torch.bfloat16, device='npu')
b = torch.randn(128, 256, dtype=torch.bfloat16, device='npu')
d = torch.empty(128, 128, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d)
torch.npu.synchronize()
ref = a.float() @ b.float().T
assert torch.allclose(d.float(), ref, atol=1, rtol=1e-2)
print("M=128, N=128, K=256 OK")
EOF
grep -o 'GemmConfig([^)]*)' /tmp/day20_run1.log | head -2
```

记录打印的 `GemmConfig(block=…x…x…, mad=…x…x…, l1_stages=…, l0_stages=…, epi_stages=…)`，对照 3.0 的假设表逐项比较：哪些一致（`l0_stages=2`、`epi_stages=1` 大概率一致）、哪些不同（`block/mad/l1_stages` 可能因成本模型与收缩精修而不同）。**用真实配置把 3.3/3.4 的两张表重算一遍**，这是"对照源码检查自己的演算"的硬件版。

**步骤 2：跑三个边界变体（约 15 分钟）。** 把上面脚本中的 shape 依次换成 `(160, 128, 256)`、`(128, 100, 256)`、`(128, 128, 200)`（注意 b 与 d 的形状随之改为 `[N, K]` 与 `[M, N]`），每个都验证 `allclose`。对照 3.8 的变体表：三个变体分别触发 M 尾块、N 尾块、K 尾块的裁剪链。

**步骤 3：对照测试套件的边界用例（约 10 分钟）。**

```bash
python tests/test_bf16_gemm.py unaligned-k --skip-prof 2>&1 | tee /tmp/day20_run2.log
python tests/test_bf16_gemm.py strided --skip-prof 2>&1 | tee /tmp/day20_run3.log
```

`strided` 列表里有 `(128, 128, 128)`（`tests/generators.py:474`，最接近主例的官方用例）；`unaligned-k` 里的 `(128, 128, 234)`（524 行）与变体 K=200 同类。任选一个，手算它的尾块 `actual_*` 再核对测试通过。

**步骤 4（可选）：观察模板实例化（约 10 分钟）。** 加 `DG_JIT_PRINT_COMPILER_COMMAND=1` 重跑步骤 1，在打印的 `bf16_gemm_impl<...>` 实例化实参里找到 `BLOCK_M/N/K`、`MAD_M/N/K`、`kNumL1Stages` 等 26 个实参的位置——第 12 天生成的就是这份代码。

### B. 无 Ascend 环境：静态替代任务

**B-1：闭卷重算主例（约 20 分钟）。** 合上教程，只看 2.2 的六个公式，对 `M=128, N=128, K=256` 重算：输出 Tile 数与干活的核、`num_k_blocks`、4 次 GM→L1 与 8 次 L1→L0 的四要素、4 次 mad 的标志、L1/L0/L0C/UB 的使用清单。对照 3.1～3.7 订正。

**B-2：边界变体手算（约 20 分钟）。** 对 3.8 的三个变体各回答：(a) 网格与调度怎么变（几个核干活）；(b) 尾块的 `actual_*`、`as_mad_aligned` 后的形状、`eff_mad_*`；(c) K=200 还要指出 `is_last` 落在哪个 k_mad 段、末段 `y_step` 是多少。对照 3.8 的表订正。

**B-3：守恒对账（约 10 分钟）。** 只用乘法验证：GM 读 A/B 的字节数、L1→L0 总流量、D 写回字节数、FLOP 总数、算术强度（参考 3.7 的对账表，先算再对）。

**B-4：用纯 Python 复刻搬运序列（约 25 分钟）。** 补齐 3 处空格（对照注释里的源码行填，不要凭记忆），运行后应复现 3.9 时间线中核 0 的全部事件：

```python
# day20_sim.py —— 无任何依赖，python3 直接运行
M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128
MM, MN_, MK = 128, 128, 64                 # MAD_M/MAD_N/MAD_K
L1_STAGES, L0_STAGES, CORES = 3, 2, 32
ELEM, FRAC = 2, 16                         # sizeof(bfloat16_t)、FRAC_MN=FRAC_K=16

ceil_div = lambda a, b: (a + b - 1) // b
aligned = lambda a, b: (a + b - 1) // b * b
gm_index = lambda mn, k: ELEM * ____                                   # ① common.hpp:160（stride_outer=K）
l1a_stage = ____                                                       # ② tile_ptr.hpp:37
l1b_base = L1_STAGES * l1a_stage                                       #   bf16_gemm.hpp:60
l0_stage = MM * MK * ELEM
num_kb, mad_no = ceil_div(K, BK), 0
for kb in range(num_kb):
    k_idx, ak, s = kb * BK, min(K - kb * BK, BK), kb % L1_STAGES
    print(f"[k_blk {kb}] GM→L1 A: src=+{gm_index(0, k_idx)}B dst=l1a[{s}]@{s * l1a_stage} region=128x{ak}")
    print(f"[k_blk {kb}] GM→L1 B: src=+{gm_index(0, k_idx)}B dst=l1b[{s}]@{l1b_base + s * l1a_stage} region=128x{ak}")
    for km in range(0, ak, MK):
        sab, effk = (km // MK) % L0_STAGES, min(ak - km, MK)
        y_step = ____                                                   # ③ bf16_gemm.hpp:138 的 as_mad_aligned 后
        mad_no += 1
        is_first, is_last = kb == 0 and km == 0, kb == num_kb - 1 and km + MK >= ak
        print(f"  [k_mad {km}] L1→L0: l1?[{s}]+frac(0,{km // FRAC}) → l0?[{sab}]@{sab * l0_stage} "
              f"x_step={aligned(128, FRAC) // FRAC} y_step={y_step} "
              f"mad#{mad_no}: {'UPDATE' if is_last else 'KEEP'}{' +clear_C' if is_first else ''}")
print("写回：aic_store → UB[0] → AIV0 drain → D[0:128, 0:128]（32 KiB），共",
      mad_no, "次 mad")
```

（参考：① `mn * K + k`；② `BM * BK * ELEM`；③ `aligned(effk, FRAC) // FRAC`。扩展任务：把 `M/N/K` 改成 3.8 的三个变体，观察输出如何变化——注意变体还需给 `x_step` 换成 `aligned(actual_m, FRAC) // FRAC` 才完整。）

**B-5：闭卷画时间线总图（约 20 分钟，当天产出核心）。** 凭记忆重画 3.9 的总图，要求包含：两级 stage 环及轮转变量、每次搬运的源/目标、四次 mad 的标志、store/drain 两跳、空闲的资源（核 1..31、L1 stage 2、AIV1）。画完对照订正，订正后的图并入当天产出。

**B-6：调度扩展练习（约 10 分钟）。** 设 `M=512, N=128, K=256`（其余假设不变）：算 `num_blocks` 与干活的核数；此时 `num_m_blocks=4` 不再走快路径，用 `get_swizzled_block`（`scheduler.hpp:217-246`）算核 0..3 各领到哪个 Tile（提示：`num_n_blocks=1 < kGroupN=8`，`n_groups=0`，所有块落在 240-246 行的分支；答案：核 c 领 (m{c}, n0)）。再想想每个核的 L1/L0 使用是否与主例相同（是——每个 Tile 的片上行为独立且相同）。

## 六、当天产出

学习计划要求今天"完整演算一个输出 Tile"。按下述模板整理（第三、五节是参考答案）：

```text
# DeepGEMM-Ascend 第 20 天记录：完整演算一个输出 Tile

## A. 问题与配置
M=＿, N=＿, K=＿（本例 128/128/256）；BLOCK=＿×＿×＿, MAD=＿×＿×＿,
L1 stages=＿, L0 stages=＿, epi stages=＿, cores=＿；有环境时贴上 DG_PRINT_CONFIGS 的真实 GemmConfig

## B. 调度
输出 Tile 数=＿, waves=＿, 干活的核=＿；m_idx=＿, n_idx=＿, actual_m=＿, actual_n=＿

## C. K 维循环
num_k_blocks=＿；每块 MAD 段数=＿；mad 总次数=＿；FLOP 守恒：＿ == 2MNK ✓/✗

## D. 搬运清单（每行一次搬运）
级    #   源                      目标                    大小/分形数        stride/标志
GM→L1 1   a.index(＿,＿)=+＿B     l1a[＿]@＿             ＿行×＿列=＿B     行距＿B
...
L1→L0 1   l1a[＿]分形(＿,＿)      l0a[＿]@＿             x＿×y＿=＿B       src/dst stride=＿
...
mad   1   128×128×64             l0c                    KEEP/UPDATE       clear_C=＿
...

## E. 片上缓冲使用图
（贴 B-5 的图：哪些 stage 被触碰、哪些空闲；static_assert 验算表）

## F. 边界变体（三个各一行）
变体     网格/调度变化      actual   as_mad_aligned   eff     写回
M=160    ＿                ＿       ＿               ＿       ＿
N=100    ＿                ＿       ＿               ＿       ＿
K=200    ＿                ＿       ＿               ＿       —

## G. 时间线总图
（贴 B-5：领任务 → 搬运 → mad → 写回的完整序列）

## H. 实验记录（有环境时）
真实配置与假设的差异及原因；三个变体的运行结果

## I. 遗留问题（交给后续天）
- 同步配对与事件号的完整语义（第 23 天）
- ENABLE_UPDATE/KEEP 与 clear_C 的硬件语义（第 24 天）
- aic_store/aiv_drain 的多 Tile 轮转（第 25 天）
- kDirectStore/kResidentA/B 路径（第 26 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. 主例的输出 Tile 数、wave 数、干活的核数各是多少？核 1～31 在哪一行代码退出？
2. 整个 GEMM 一共发生多少次 GM→L1、多少次 L1→L0、多少次 `mad`？各是多少字节/FLOP？
3. k_blk=1 时 A 的 GM 源偏移是多少字节？写出计算式与出处。
4. L1 的 stage 2 本例被使用了吗？为什么？`l1b` 的基址由哪行代码决定、是多少？
5. 四次 `mad` 中 `is_last` 哪次为真？写出判定式的两个条件。
6. L0A 的两个 stage 按什么变量轮转？一个 stage 装多少字节、多少个分形单元？
7. 本例为什么没有边界 Tile？给出一个会产生 K 边界的 K 值，并写出尾块的 `actual_k`、末段 `eff_mad_k` 与 `y_step`。
8. D 的写回经过哪两跳？本例哪个 AIV 执行了排水、另一个为什么跳过（哪一行）？
9. 若把 `kNumL1Stages` 从 3 改成 8（heuristics 公式允许），演算中哪些数字会变、哪些不变？
10. 写出 FLOP 守恒与 GM 流量的算式（A、B、D、L1→L0 四项）。
11. `mad` 封装与 `asc_mmad` 的参数顺序有什么区别？在哪一行完成换序？
12. （思考）若 `BLOCK_M=64`（M、N、K 与其余假设不变），输出 Tile 数与调度怎么变？MAD 搜索会改选什么？每块的 mad 次数呢？

**参考答案：**

1. 1 个 Tile、1 个 wave、只有核 0 干活。核 1～31 在 `get_next_block` 的 `block_idx >= num_blocks` 判断处返回 `false`（`scheduler.hpp:191-192`；block_idx = core_id ≥ 1）。其 AIV 在 `bf16_gemm.hpp:180` 的同一判断处退出。
2. 4 次 GM→L1（每次 32768 B 有效数据）；8 次 L1→L0（每次 16384 B，A、B 各 4 次）；4 次 `mad`（每次 2×128×128×64 = 2097152 FLOP）。
3. `+256 B`：`sizeof(bfloat16_t) × (mn×stride_outer + k) = 2 × (0×256 + 128)`（`common.hpp:160`）。
4. 没有被使用：`num_k_blocks = 2 < kNumL1Stages = 3`，`s` 环只触达 0/1（`bf16_gemm.hpp:112`）。`l1b` 基址 = `l1a.offset(kNumL1Stages)`（60 行）= 3×32768 = 98304。
5. 第 4 次（k_blk=1、k_mad=64）。条件：`k_blk_idx == num_k_blocks - 1`（1==1）且 `k_mad_idx + MAD_K >= actual_k`（64+64=128 ≥ 128）（`bf16_gemm.hpp:135`）。
6. 按 `s_ab` 轮转（随 `k_mad_idx`，133 行）；一个 stage = 128×64×2 = 16384 B = (128/16)×(64/16) = 8×4 = 32 个分形单元（每单元 16 行 × 32 B = 512 B）。
7. 因为 128 整除 M、N，128 整除 K=256，`actual_*` 恒等于 `BLOCK_*`。例如 K=200：`num_k_blocks=2`，尾块 `actual_k = 72`；末段（k_mad=64）`eff_mad_k = min(72−64, 64) = 8`，`as_mad_aligned` 后 `shape_k = 16`，`y_step = 1`。
8. 两跳：`aic_store` 把 L0C 经 `copy_l0c_to_ub` 量化（F322BF16）送入 `ub[0]`（`epilogue.hpp:42-47`），`aiv_drain` 再用 `copy_ub_to_gm` 写 D（81-82 行）。AIV0 执行（`s_epi=0`，认领规则在 43 行）；AIV1 在 57 行 `continue` 跳过。
9. 变：`l1b` 基址 = `l1a.offset(8)` = 262144，`l1b.offset(8)` = 524288 恰好压线通过 66 行断言。不变：所有搬运的源/目标内容（stage 0/1 照旧）、L0/L0C/UB 的使用、GM 流量——`num_k_blocks=2` 用不满更深的环。
10. FLOP：4 × 2×128×128×64 = 8388608 = 2×M×N×K；GM：A 2×32768 = 65536 B、B 65536 B、D 32768 B；L1→L0：8×16384 = 131072 B。
11. 封装是 `(shape_m, shape_n, shape_k)`，`asc_mmad` 是 `(shape_m, shape_k, shape_n)`；换序发生在 `copy.hpp:378`。
12. `num_m_blocks=2`、`num_blocks=2`：核 0、核 1 各领一块（`num_m_blocks=2 < 4` 仍走快路径）。MAD 搜索改选 `mad_m = 64`（split (1,1)，`l0a_k = 64K×8/(2×64×16) = 256`、`l0b_k = 128` → `mad_k = 64`，即 MAD = 64×128×64，`heuristics.hpp:124-147`）。每块的 K 循环结构不变（2 块 × 2 段 = 4 次 mad），两块合计 8 次——FLOP 守恒不变；`l1a` stage 缩为 64×128×2 = 16 KiB，`l1b` 基址 = `l1a.offset(3)` = 49152。

## 八、通过标准自评

学习计划的通过标准是：**能把调度、地址计算和数据搬运串成一个完整过程**。逐项自评，全部达成即通过：

1. **闭卷演算**：不看第三节，能在 30 分钟内完成主例六步（调度 → K 循环 → GM→L1 → L1→L0 → mad → 写回），数字与 3.1～3.7 一致；
2. **数字溯源**：对演算中的任意数字，能立刻指出它的代码出处（文件:行号）——第四节的九个站点就是抽查清单；
3. **边界泛化**：能对任给的小 shape（如 M=96、N=200、K=300）算出网格、尾块 `actual_*` 与 `eff_mad_*`——用 3.8 的变体方法即可；
4. **守恒校验**：FLOP 与字节流量对账能一次算平；
5. **口头复述**：能向别人用 10 分钟讲清"一个输出 Tile 从领任务到写回 GM 的完整生命周期"，不看任何材料；
6. **遗留问题**：列出了同步、MAD 语义、Epilogue 轮转、特殊路径四类待解问题——它们正好是第 4 周（第 22～26 天）的入口；明天（第 21 天）机动日建议重读一遍 `bf16_gemm_impl`，跳过同步细节，为第 4 周热身。
