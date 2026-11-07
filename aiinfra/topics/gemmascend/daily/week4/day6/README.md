# 第 4 周 · 第 6 天（总第 27 天）：完整复述 BF16 GEMM（复盘日）

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 4 周"BF16 Kernel——MAD、流水线与 Epilogue"的第 6 天（复盘日）教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 30 分钟，闭卷复述与数字演算约 70 分钟，源码对照修正约 30 分钟，实践与自评约 40 分钟。

本周五天把 `bf16_gemm_impl` 的内部拆成了五块：K 维两层循环（第 22 天）、多级流水与同步（第 23 天）、MAD 与 L0C 累加（第 24 天）、AIC/AIV 分工与 Epilogue（第 25 天）、特殊执行路径（第 26 天）。加上第 3 周的外围（模板参数、片上内存、调度、地址、搬运）与第 2 周的 Host 调用链，所有零件都已在手上。今天按学习计划的要求做一次**完整复述**：**不看源码**，画出

```text
任务调度
→ GM/L1/L0 数据搬运
→ MAD 累加
→ AIC/AIV Epilogue
→ 写回 GM
```

然后对照 `bf16_gemm.hpp` 修正，并**记录仍不理解的三个问题**。复盘方法与前几周相同：判断掌握程度的唯一标准是闭卷复述。第三节是一份按此要求写出的参考答案——用 `M=256、N=128、K=512` 与一组刻意选择的假设配置，把一个输出 Tile 从"领任务"到"写回 GM"的全部状态变化走一遍；请先自己画、自己算，再对照订正。这组配置让本周每条主线（L1 环回绕、L0 复用条件、多 CD stage 的 Epilogue 轮转、AIC/AIV 借还）都在同一个例子里出现。

**通过标准**：能讲清楚一个输出 Tile 从调度到写回的完整生命周期。

## 一、学习目标

完成今天的学习后，你应当能够：

1. **闭卷画出五段流程图**：任务调度（`scheduler.hpp`）、GM/L1/L0 数据搬运（`bf16_gemm.hpp:112-121、141-146`）、MAD 累加（`:133-152`）、AIC/AIV Epilogue（`:129-131、155-163` + `epilogue.hpp:37-47`）、写回 GM（`:173-192` + `epilogue.hpp:49-87`），每段标注执行者（AIC/AIV）、指令队列（MTE2/MTE1/M/FIX/V/MTE3）与关键行号；
2. 对给定 shape 与配置，**算出**一个输出 Tile 的全部结构数字：Tile 网格与核分配、`num_k_blocks`、每子块与全 Tile 的 mmad 数、L1→L0 搬运次数（含 `:142/:144` 复用条件的取舍）、4 次交付的 `s_epi` 路由，并用 FLOP 与字节流量对账验证；
3. **默写五个环**（`s`、`s_ab`、`s_cd`、`s_epi`、`epi_base`）的定义、转速与行号，并解释"事件号 = 缓冲编号"如何让少量事件支撑任意长的循环；
4. **追一条事件链**：说明 k_blk 环绕时 `wait(MTE1→MTE2, s)`（`:116`）消费的是哪一次 `notify`（`:167` 或预置 `:83`），以及无消费者的"遗留事件"如何被下一个输出 Tile 的首次等待消费；
5. 复述 AIC/AIV 的**四步借还协议**（`aic_wait_free` → `aic_store` → `aiv_drain` 的 wait → 归还 arrive）与三方不通信保持 `s_epi` 一致的两条机制；
6. 说明特殊路径是主线的**条件裁剪**：`kDirectStore` 删掉哪两段、`kResidentA/B` 删掉哪几条搬运，各自的前提约束是什么；
7. 对照 `bf16_gemm.hpp` 逐段修正自己的图，把错误归因为"漏了某行/记错条件/概念混淆"，并**记录仍不理解的三个问题**（归类：CANN 指令级 / 仓库可查但未读 / 后续天的主题）；
8. 有 Ascend 环境时，用 `DG_PRINT_CONFIGS=1` 打印真实 `GemmConfig`，说明假设配置与真实配置的差异及原因；无环境时完成全部纸面任务。

**当天产出：闭卷五段流程图（订正版）+ 一个输出 Tile 的完整生命周期演算 + 三个仍不理解的问题**（模板见第六节，参考答案见第三节）。

## 二、知识梳理（先复盘，再复述）

本节把本周五天与前几周的结论压成三张"随身清单"，作为闭卷复述时唯一允许带的东西。独立阅读的读者也可以只用这三张清单跟上第三节。

### 2.1 本周一览：五天各回答了什么问题

| 天 | 回答的问题 | 核心结论 | 在今天复述中的位置 |
| :--: | :-- | :-- | :-- |
| 22 | K 循环怎么转？ | 两层循环：外层 `k_blk_idx`（GM→L1 节拍，`s` 环）、内层 `k_mad_idx`（mmad 节拍，`s_ab` 环）；`actual_k`/`eff_mad_k` 处理尾部；`is_first`/`is_last` 控制清零与提交 | 3.3 的循环结构、3.4 的标志表 |
| 23 | 谁等谁？ | 四组有方向的队列事件：L1[s] 空闲/就绪、L0[s_ab] 空闲/就绪；只有"空闲"要 `set_flags` 预置（`:83-84`），否则死锁；事件号 = stage 号，随环复用 | 3.3 的事件链与账本 |
| 24 | 结果怎么攒出来的？ | `mad()` 收 `(m,n,k)`、内部换序 `(m,k,n)`（`copy.hpp:378`）；每子块一条累加链：首条 `clear_C`、中间 `KEEP`、末条 `UPDATE`；`s_cd` 到子块的映射对所有块一致 | 3.4 的 L0C 生命周期 |
| 25 | 谁把结果交出去？ | `__mix__(1, 2)`：1 AIC + 2 AIV 执行同一份代码；UB 四步借还；`s_epi` round-robin 路由 `aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs`、`stage_idx = s_epi / kNumAIVs`；三方靠"相同序列 + 相同推进规则"锁步 | 3.5/3.6 的交付与写回 |
| 26 | 有没有近道？ | `kDirectStore`/`kResidentA/B` 都是主线代码的条件裁剪（`:27、82、109-110、117-118、142-145、156-159、174`）；约束链 = 设备 `static_assert`（`:38、45`）+ host 闸门（`heuristics.hpp:521-533、552-557`） | 3.9 的变体 V3/V4 |

第 3 周的背景（用到时标注）：模板参数分类（第 15 天）、片上布局与 `static_assert`（第 16 天）、持久化调度与 swizzle（第 17 天）、`gm_ptr::index`/`operand_ptr::offset` 地址计算（第 18 天）、`nd2nz` 与分形搬运（第 19 天）、第 20 天的单 Tile 全 trace。第 2 周的 Host 链条今天只在骨架图顶部出现一次。

### 2.2 随身清单一：五个环与全部同步点

五个环（今天必须能默写）：

| 环 | 含义 | 推进节奏 | 定义/推进行号 |
| :-- | :-- | :-- | :-- |
| `s` | L1 stage 下标（也是事件号） | 每个 k_blk +1，模 `kNumL1Stages` | `bf16_gemm.hpp:112` |
| `s_ab` | L0A/L0B stage 下标 | 每个 k_mad +1，模 `kNumL0Stages` | `:133` |
| `s_cd` | L0C 子块下标 | 每个子块 +1，模 `num_mad_cd_stages = (BLOCK_M/MAD_M)×(BLOCK_N/MAD_N)` | `:124`（`:51`） |
| `s_epi` | UB stage 总编号 | 每个子块循环体末尾 +1，模 `epilogue.kNumStages`；**每个 k_blk 开头重置为 `epi_base`** | `:122`、`:164` |
| `epi_base` | 跨输出 Tile 续接 | 每个 Tile 结束时 `epi_base = s_epi` | `:86/:107/:169` |

四组队列事件（AIC 内；`asc_sync_wait(源, 目标, e)` = 目标队列阻塞等源的事件，方向由"谁有资格宣布这件事"决定）：

| 含义 | 方向 | 循环内 wait | 循环内 notify | 预置 |
| :-- | :-- | :-- | :-- | :-- |
| L1[s] 空闲 | MTE1 → MTE2 | `:116` | `:167`（在 m/n 子块循环之后） | `:83` |
| L1[s] 就绪 | MTE2 → MTE1 | `:121` | `:119` | —— |
| L0[s_ab] 空闲 | M → MTE1 | `:141` | `:152` | `:84` |
| L0[s_ab] 就绪 | MTE1 → M | `:148` | `:146` | —— |

两组块内事件（AIC↔AIV，flag 编码 `((subblock_id & 1) << 4) | (flag_id & 0xf)`，`ascend/sync.hpp:30`）：

| 含义 | arrive（发出） | wait（等待） | 预置 |
| :-- | :-- | :-- | :-- |
| UB stage 已填好（FIX 队列，随队保序） | `epilogue.hpp:46` | `epilogue.hpp:69`（AIV，挂算子的 `kPipe`） | —— |
| UB stage 已搬完（MTE3 队列） | `epilogue.hpp:84` | `epilogue.hpp:38`（AIC 的 `aic_wait_free`，挂 FIX） | `aiv_initialize`（`epilogue.hpp:89-97`） |

### 2.3 随身清单二：两个标志、三个 eff 与边界裁剪

- **两个标志**（`:134-135` 定义，`:149-151` 消费）：`is_first = (k_blk_idx==0 && k_mad_idx==0)` → `mad()` 的 `clear_C`（累加从 0 开始，覆盖上一块同位置子块的残留）；`is_last = (k_blk_idx==num_k_blocks-1 && k_mad_idx+MAD_K>=actual_k)` → `ENABLE_UPDATE`（末步提交），中间步 `ENABLE_KEEP`。每个输出子块的 K 遍历各自独立判定。
- **三个 eff**：`eff_mad_m = min(actual_m - m_mad_idx, MAD_M)`（`:125`）；`eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N), get_frac_k<ub_dtype_t>())`（`:126`，BF16+Identity 时对齐 16，保证 L0C→UB 行 32B 对齐）；`eff_mad_k = min(actual_k - k_mad_idx, MAD_K)`（`:137`）。tile 形状经 `as_mad_aligned` 对齐（决定 L1→L0 搬多少），传给 `mad()` 的仍是真值（决定算多深）。
- **边界三级裁剪**（第 19/22 天）：GM→L1 按 `actual_m/n/k` 少搬；L1→L0 按对齐后的整分形搬；计算与写回按 `eff_*`/`min` 真值。
- **常量**（`common.hpp`）：`kNumAICores = 32`（:51）；L1 512KB、L0A/L0B 各 64KB、L0C 256KB、UB 256KB（:54-58）；`FRAC_MN = 16`（:63）；BF16 的 `get_frac_k = 16`（:73-79）。

### 2.4 五段流程骨架（复述的起点）

```text
┌─ Host（第 2 周复习，今天只画一次）
│  deep_gemm.bf16_gemm_nt（deep_gemm/__init__.py:61-67）
│  → C++ binding（csrc/python_api.cpp）→ get_gemm_desc → launch_bf16_gemm
│  → select_gemm_config → JIT compile/launch（csrc/jit_kernels/bf16_gemm.hpp:16,23）
▼
bf16_gemm_impl（deep_gemm/include/deep_gemm/bf16_gemm.hpp，__mix__(1, 2)：每 block 1 AIC + 2 AIV）
① 任务调度   Scheduler.get_next_block（:89）→ (m_blk, n_blk) → m_idx/n_idx/actual（:90-106）
② 数据搬运   GM→L1（MTE2，:117-118）→ L1→L0A/L0B（MTE1，:143/:145，条件复用）
③ MAD 累加   mad（M 队列，:149-151）：clear_C → KEEP…→ UPDATE，写 L0C[s_cd]
④ Epilogue   AIC：aic_wait_free（:129-131，FIX）+ aic_store（:161，L0C→UB[s_epi]，含 F322BF16）
⑤ 写回 GM    AIV：aiv_drain（:190）→ wait → copy_ub_to_gm（epilogue.hpp:81-82）→ 归还 arrive（:84）
```

复述时对每一段追问三件事：**谁执行（AIC/AIV）、排入哪条队列、等谁的通知**。能全部答出，第三节就是你的订正清单；答不出，就回到对应天重读。

## 三、综合练习：闭卷复述 BF16 GEMM（含参考答案）

### 3.0 练习规则

按学习计划执行三步：

1. **闭卷画图**（约 40 分钟）：不看源码、不看本文第三节，画出 2.4 骨架的展开版——每个环节写出关键函数、行号、队列与同步点；
2. **数字演算**（约 30 分钟）：用 3.1 的设定算出全部结构数字（调度、循环、搬运、交付、对账）；
3. **对照修正**（约 30 分钟）：逐段对照源码与本节订正，把每处错误归类（漏行/记错条件/概念混淆），并记录仍不理解的三个问题（候选见第六节）。

### 3.1 参考设定：shape 与假设配置

**问题**：`D[256, 128] = A[256, 512] × Bᵀ`，即 `bf16_gemm_nt` 语义——A、B 都是 K-major 紧凑 BF16（`stride_outer = K = 512`），D 是 `[M, N]` 行主序 BF16，无累加、无 bias、无 alpha、Identity epilogue。数据量：A = 256×512×2 = 262144 B，B = 128×512×2 = 131072 B，D = 256×128×2 = 65536 B；总 FLOP = 2×256×128×512 = 33554432。

**假设配置**（沿用第 24/25 天 3.6/3.8 的配置家族，把 K 加大到 512 使 `s` 环回绕）：

| 参数 | 假设值 | 角色 | 源码锚点 |
| :-- | :-- | :-- | :-- |
| `BLOCK_M/N/K` | 128 / 128 / 128 | Tile 尺寸，L1 stage 布局 | `bf16_gemm.hpp:59-60` |
| `MAD_M/N/K` | 64 / 64 / 64 | 单条 mmad 的尺寸；M/N 拆 2×2 子块 | `:62-64` |
| `kNumL1Stages` | 2 | L1 stage 环深（K=512 有 4 个 k_blk，环将回绕） | `:59-60、66` |
| `kNumL0Stages` | 2 | L0 双缓冲（源码恒取 2，`heuristics.hpp:111-112`） | `:67-68` |
| `kNumEpilogueStages` | 2（每 AIV） | `epilogue.kNumStages = 2×2 = 4` | `epilogue.hpp:24` |
| `kNumCores` | 32 | 持久化调度核数（950DT） | `common.hpp:51` |
| `kDirectStore/kResidentA/B` | 均为 false | 走普通 Epilogue 路径 | 第 26 天 |

**合法性验算**（复述时先过一遍，全部来自 `bf16_gemm.hpp:66-70` 与 `epilogue.hpp:28-35` 的 `static_assert`）：

```text
:66  l1b.offset(2)：l1a 每 stage 128×128×2 = 32768B；l1b 基址 = l1a.offset(2) = 65536，
     65536 + 2×32768 = 131072 ≤ 512KB ✔
:67/:68  l0a/l0b 每 stage 64×64×2 = 8192B；offset(2) = 16384 ≤ 64KB ✔
:69  num_mad_cd_stages = (128/64)×(128/64) = 4；l0c 每 stage 64×64×4 = 16384B，
     offset(4) = 65536 ≤ 256KB ✔
:70  BLOCK_K 128 % MAD_K 64 == 0 ✔
epilogue.hpp:33  kNumStages 4 ≥ num_mad_cd_stages 4 ✔；:32 每AIV 2 ∈ [1,16] ✔
epilogue.hpp:29  MAD_N 64 % 16 == 0 ✔；:30 MAD_M 64 % 16 == 0 ✔
epilogue.hpp:35  每 AIV UB = 2×64×64×2 = 16384B，Identity 无附加区 ≤ 256KB ✔
```

> **关于假设配置的诚实说明**（第 24 天 3.6 已有同款说明）：启发式对 128×128 的块按 `(1,1)→(1,2)→(2,1)` 顺序搜索 MAD 拆分（`heuristics.hpp:124`），`(1,1)` 即 MAD 128×128×64 会先被接受，因此 64×64 通常**不会**被真实选中；L1 级数公式（`heuristics.hpp:171-172`）对 block 128×128×128 给出的是 8 而不是 2。选这组假设是为了让多 CD stage、L0 复用条件、`s` 环回绕三条本周主线在同一个例子里同时出现；真实配置用 `DG_PRINT_CONFIGS=1` 核对（实践步骤 A-1）。

### 3.2 第一段：任务调度

代入网格公式（`scheduler.hpp:91-93`）：

```text
num_m_blocks = ceil_div(256, 128) = 2
num_n_blocks = ceil_div(128, 128) = 1
num_blocks   = 2 × 1 = 2          → 1 个 wave（ceil_div(2, 32)）
```

任务分发（`scheduler.hpp:142`：`block_idx = (++current_iter) × kNumCores + core_id`）：

| 核 | 第 1 次 block_idx | 结果 |
| :-- | :-- | :-- |
| 核 0 | `0×32+0 = 0 < 2` | 领到 tile，进入 K 循环 |
| 核 1 | `0×32+1 = 1 < 2` | 领到 tile |
| 核 2～31 | `2..31 ≥ 2` | `get_next_block` 返回 `false`（`:191-192`），AIC 立即退出；其 AIV 在 `aiv_initialize()` 后于 `:180` 的同一判断处退出 |

`num_m_blocks = 2 < kGroupM = 4`，映射走**快路径**（`scheduler.hpp:209-211`）：`m_block = block_idx / num_n_blocks`、`n_block = block_idx % num_n_blocks`——核 0 得 (m0, n0)，核 1 得 (m1, n0)；GROUP_M swizzle/XOR/蛇形（`:217-246`）本例不执行。随后（`bf16_gemm.hpp:90-106`）：

```text
核 0：m_idx = 0、n_idx = 0、actual_m = min(256-0, 128) = 128、actual_n = 128
核 1：m_idx = 128、n_idx = 0、actual_m = min(256-128, 128) = 128、actual_n = 128
```

Normal GEMM 下 `eff_shape_k = shape_k = 512`、`k_idx_base = 0`（`:95-97`）、`batch_idx = b_group_idx = 0`（`:99-103`）。每个 block 由 1 AIC + 2 AIV 三方执行同一份代码（`:27` 的 `__mix__(1, 2)`），三方各自构造同一个 `Scheduler`（`:73`，构造在 `if ASCEND_IS_AIC` 之前）并重放同一序列——这是第 25 天"三方锁步"的基石。

### 3.3 第二段：GM/L1/L0 数据搬运（两层循环 + 同步链）

**循环结构**（`:112-133`）：

```text
num_k_blocks = ceil_div(512, 128) = 4            // :96
每个 k_blk：actual_k = 128（无尾部）；s = 0,1,0,1（模 kNumL1Stages=2 → 环回绕）
子块：m_mad ∈ {0,64} × n_mad ∈ {0,64}，共 4 个；s_cd = 0,1,2,3（每 k_blk 从 0 连续扫描，:124）
每个 k_blk 内每子块：k_mad ∈ {0,64}，2 条 mmad；s_ab = 0,1（:133）
每子块 K 遍历 = 4 k_blk × 2 步 = 8 条 mmad；每 tile = 4 子块 × 8 = 32 条
全 GEMM = 2 tile × 32 = 64 条；FLOP = 64 × 2×64×64×64 = 33554432 = 2×M×N×K ✔
```

**GM→L1 搬运**（每 tile 4 k_blk × (A,B) = 8 次，每次 `128×128×2 = 32768 B`；源偏移 = `sizeof(T) × (mn × stride_outer + k)`，`common.hpp:160`；目标 = `l1a[s] @ s×32768`、`l1b[s] @ 65536 + s×32768`；`nd2nz` 按 32B 分形重摆，`ascend/copy.hpp:120-124`）：

| k_blk | k_idx | s | 核 0：A 源 / B 源（`a.index(0,k_idx)`） | l1a[s] / l1b[s] @ L1 |
| :-- | :-- | :-- | :-- | :-- |
| 0 | 0 | 0 | +0 B / +0 B | 0 / 65536 |
| 1 | 128 | 1 | +256 B / +256 B | 32768 / 98304 |
| 2 | 256 | 0 | +512 B / +512 B | 0 / 65536（**复用 stage 0**） |
| 3 | 384 | 1 | +768 B / +768 B | 32768 / 98304（**复用 stage 1**） |

核 1 的 A 源整体加 `2×(128×512) = 131072 B`（`a.index(128, k_idx)`），其余同构。

**L1 环回绕的事件链**——本例最值得复述的一段。L1 只有 2 份缓冲，k_blk 2 要写 `l1a[0]`，必须等 k_blk 0 的数据被**全部 4 个子块**读完后由 MTE1 归还（`:167` 在 m/n 子块循环之后）。把 L1 空闲事件（方向 MTE1→MTE2）的收支记成账本：

| 事件 | 发出方 | 消费方（wait `:116`） |
| :-- | :-- | :-- |
| e0 预置（`set_flags`，`:83`） | 启动时 | k_blk 0 |
| e1 预置 | 启动时 | k_blk 1 |
| e0（k_blk 0 末尾 `:167`） | MTE1 | **k_blk 2（环回绕）** |
| e1（k_blk 1 末尾 `:167`） | MTE1 | k_blk 3 |
| e0（k_blk 2 末尾 `:167`） | MTE1 | 本块内无消费者——**遗留**，若同核还有下一个 tile，被其首个 k_blk 消费 |
| e1（k_blk 3 末尾 `:167`） | MTE1 | 同上，遗留 |

每份缓冲恰好配一对"空闲 + 就绪"事件（`:116/:167` 与 `:121/:119`）：就绪事件当场配对消费，空闲事件跨 k_blk 流转——**事件号 = 缓冲编号**，2 个号支撑 4 圈甚至任意长 K 的循环（第 23 天）。

**L1→L0A/L0B 搬运与复用**（`:141-146`）。A 的数据只依赖 `(m_mad, k_mad)`，B 只依赖 `(n_mad, k_mad)`；复用条件据此不对称（`:142` 看 `n_mad_idx`、`:144` 看 `m_mad_idx`）。每个 k_blk 的 8 个 `(m_mad, n_mad, k_mad)` 组合：

| m_mad | n_mad | k_mad | s_ab | A（`:142`：`BLOCK_N==MAD_N` ✗ / `BLOCK_K>MAD_K×2` ✗ / `n_mad==0`） | B（`:144`：`BLOCK_N != MAD_N` ✔ 恒真） |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 0 | 0/64 | 0/1 | 搬（n_mad==0） | 搬 |
| 0 | 64 | 0/64 | 0/1 | **复用**（n_mad≠0） | 搬 |
| 64 | 0 | 0/64 | 0/1 | 搬 | 搬 |
| 64 | 64 | 0/64 | 0/1 | **复用** | 搬 |

- 每 k_blk：A 搬 4 次、B 搬 8 次，每次 `64×64×2 = 8192 B`（L0 tile 形状 `as_mad_aligned(64,64)` 不变）；
- **A 复用的原理**：子块 (m0,n0) 的 K 遍历结束后 `l0a[0]/[1]` 恰好还装着 (m0, k0:64)/(m0, k64:128) 的分形，(m0,n1) 依相同的 `s_ab` 序读到的还是同一份；
- **B 不可复用的原理**：n_mad 循环把 `l0b[s_ab]` 覆写成了 n=64 的切片，m_mad=64 的子块必须重搬（`:144` 的 `BLOCK_N != MAD_N` 项恒真，正对应"N 方向拆分使 L0B 缓存失效"）；
- 每条 mmad 的四个同步点（`:141/:146/:148/:152`）**不因复用而省略**——闸门模型不变（第 23 天）。以 (m_mad=64, n_mad=0, k_mad=0) 的 A 搬运为例，`copy_l1_to_l0a` 的六个标量（`copy.hpp:137`）：`x_start=64/16=4、y_start=0、x_step=64/16=4、y_step=4、src_stride=128/16=8、dst_stride=4`。

### 3.4 第三段：MAD 累加（一个子块的 L0C 生命周期）

每个输出子块独占一份 `l0c[s_cd]`（64×64 FP32 = 16 KiB），映射对所有 k_blk、所有 tile 一致（`s_cd = (m_mad/64)×2 + n_mad/64`）。子块 (m0,n0)（s_cd=0）的 8 条 mmad：

| # | k_blk | s | k_mad | s_ab | is_first | is_last | unit_flag | clear_C | L0C[0] 语义 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 0 | 0 | 0 | 0 | ✔ | ✘ | KEEP | **true** | C = 0 + A[0:64,0:64]·Bᵀ[0:64,0:64] |
| 2 | 0 | 0 | 64 | 1 | ✘ | ✘ | KEEP | false | C += A[0:64,64:128]·Bᵀ[0:64,64:128] |
| 3 | 1 | 1 | 0 | 0 | ✘ | ✘ | KEEP | false | C += A[0:64,128:192]·… |
| 4 | 1 | 1 | 64 | 1 | ✘ | ✘ | KEEP | false | C += A[0:64,192:256]·… |
| 5 | 2 | 0 | 0 | 0 | ✘ | ✘ | KEEP | false | C += A[0:64,256:320]·… |
| 6 | 2 | 0 | 64 | 1 | ✘ | ✘ | KEEP | false | C += A[0:64,320:384]·… |
| 7 | 3 | 1 | 0 | 0 | ✘ | ✘ | KEEP | false | C += A[0:64,384:448]·… |
| 8 | 3 | 1 | 64 | 1 | ✘ | **✔** | **UPDATE** | false | C += A[0:64,448:512]·…，提交 |

其余三个子块结构完全同构，`is_first`/`is_last` 各自独立判定；步 3、5、7 说明 `k_mad_idx == 0` 但 `k_blk_idx != 0` 时**绝不能清零**。`mad()` 内部把 `(m,n,k)` 换序成 `asc_mmad(m,k,n)`（`copy.hpp:378`）。同一条 M 队列内对同一累加器的连续读-改-写天然串行，因此这组事件守的是 L0A/L0B 而不是 L0C；跨 tile 复用 L0C 时，下一块同位置子块的首条 mmad 用 `clear_C` 覆盖旧值，不依赖旧值（第 24 天）。

### 3.5 第四段：AIC/AIV Epilogue（借与交付）

`s_epi` 的节奏先要记牢（最容易画漏的一处）：每个 k_blk 开头重置 `s_epi = epi_base`（`:122`），每个子块循环体末尾推进一格、模 `kNumStages` 回绕（`:164`）。于是：

- **k_blk 0**：四个子块各调一次 `aic_wait_free(s_epi)`（`:129-131`），依次等 `FIX·AIV0·UB0、FIX·AIV1·UB0、FIX·AIV0·UB1、FIX·AIV1·UB1`——本 tile 是各核唯一/首个 tile，四个等待全部由 `aiv_initialize` 的预置直接放行（`epilogue.hpp:89-97`）。该等待挂在 FIX 队列上，挡住的是之后发往同一 UB stage 的交付拷贝，不影响 M 队列的 mmad；
- **k_blk 1、2**：`s_epi` 被重置又推进，但既无 wait_free 也无 store，无人消费（无害）；
- **k_blk 3（交付窗口，`:155-163`）**：每子块 K 遍历结束立即交付一次，`aic_store`（`epilogue.hpp:42-47`）= `copy_l0c_to_ub(ub[stage_idx], l0c_tile, 64, 64, DUAL_DST_DISABLE, aiv_id)`（FIX 队列，`quant_mode = F322BF16` 在此完成 FP32→BF16）+ `set_intra_block(PIPE_FIX, aiv_id, stage_idx)`"已填好"（arrive 随 FIX 队列保序，AIV 收到时数据必已就绪）。

核 0（tile m0，`epi_base = 0`）的交付路由表：

| 交付序 | 子块 | s_cd | s_epi | aiv_id = s_epi%2 | stage_idx = s_epi/2 | k_blk 0 时等过的归还 | GM 区域（`d.index(m_idx+m_mad, n_idx+n_mad)`） |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | (m0,n0) | 0 | 0 | 0 | 0 | FIX·0·0 | D[0:64, 0:64]（+0 B） |
| 2 | (m0,n1) | 1 | 1 | 1 | 0 | FIX·1·0 | D[0:64, 64:128]（+128 B） |
| 3 | (m1,n0) | 2 | 2 | 0 | 1 | FIX·0·1 | D[64:128, 0:64]（+16384 B） |
| 4 | (m1,n1) | 3 | 3 | 1 | 1 | FIX·1·1 | D[64:128, 64:128]（+16512 B） |

四个 tile 结果 round-robin 平分给两个 AIV（`epilogue.hpp:23` 注释）。块尾 `s_epi` 回绕到 0，`epi_base = 0`；`last_m_idx = 0、last_n_idx = 0`（`:169-171`，供 `kResidentA/B` 的 `reuse_a/reuse_b` 判定）。核 1 同构，四个 GM 区域整体平移到 D 的第 128..255 行。

### 3.6 第五段：写回 GM（AIV 排水与归还）

AIV 分支（`:173-192`）：`kDirectStore` 时 `return`（`:174`）→（本例非 k-grouped，跳过空组清零）→ `aiv_initialize()`（预置；无累加不设原子加）→ 以**与 AIC 相同的 Scheduler 序列**进入 `aiv_drain(d, m_idx, n_idx, actual_m, actual_n, s_epi)`（`:190`，`s_epi` 引用传参）。

`aiv_drain`（`epilogue.hpp:49-87`）对核 0 的执行：

```text
AIV0（asc_get_sub_block_id() = 0，处理 s_epi = 0, 2）
  子块扫描 (m0,n0)→(m0,n1)→(m1,n0)→(m1,n1)，s_epi 每 tile 前进一格（:56，含被 continue 跳过的）
  (m0,n0)：路由 aiv_id == 2-2+0%2 == 0 ✔ → wait_intra_block(MTE3, 0, 0)   ← 等 aic_store 的 FIX arrive
           Identity：process_mad_block 为空（FIX 已转 BF16）；kPipe = MTE3，:76-80 的 V→MTE3 握手被裁掉
           copy_ub_to_gm(d.index(0,0).ptr(), ub, 64, 64, d.stride_outer×2 = 256B, kL2CtrlStoreCd)（:81-82）
           set_intra_block(PIPE_MTE3, 0, 0)                                 ← 归还 arrive
  (m0,n1)：s_epi=1，路由不归我 → continue（s_epi 照常推进到 2）
  (m1,n0)：wait(MTE3,0,1) → copy_ub_to_gm D[64:128,0:64] → 归还(MTE3,0,1)
  (m1,n1)：continue
AIV1 同构，处理 s_epi = 1, 3，写 D[0:64,64:128] 与 D[64:128,64:128]
```

四个归还 arrive 本块内没有消费者——它们与 L1 的遗留事件同理：若同核还有下一个 tile，其 k_blk 0 的 `aic_wait_free` 才会真正等 AIV 排水（不再是预置直通）。本例每核只有一个 tile，因此归还事件随 kernel 结束废弃。三方自始至终**不传递 `s_epi`**，靠"相同确定性序列 + 相同推进规则"锁步（第 25 天 3.7）。

### 3.7 时间线总图（当天产出核心）

```text
核 0（block 0 = 1 AIC + AIV0 + AIV1）：tile (m0,n0)，m_idx=0，actual 128×128
────────────────────────────────────────────────────────────────────────────────────────
[启动] set_flags：L1 空闲 e0/e1（:83）、L0 空闲 e0/e1（:84）；AIV0/AIV1 aiv_initialize
       预置各自 UB stage 0/1 空闲（epilogue.hpp:89-97）
│
├─ k_blk 0（s=0，k_idx=0）
│   [MTE2] wait(MTE1→MTE2,0)〔预置 e0〕→ GM→L1：A(+0B)、B(+0B) 各 32KiB → notify(MTE2→MTE1,0)
│   [MTE1] wait(MTE2→MTE1,0)〔就绪〕
│   ├─ 子块(m0,n0) s_cd=0：aic_wait_free(FIX·0·0)〔AIV0 预置〕
│   │    k_mad 0 (s_ab=0)：wait(M→MTE1,0)〔预置〕→ 搬A+搬B → notify(MTE1→M,0) → wait(MTE1→M,0)
│   │      mad#1 KEEP + clear_C → notify(M→MTE1,0)
│   │    k_mad 64 (s_ab=1)：… mad#2 KEEP
│   ├─ 子块(m0,n1) s_cd=1：aic_wait_free(FIX·1·0)〔AIV1 预置〕
│   │    k_mad 0/64：A 复用 + 搬B(n1) → mad#3 KEEP+clear_C / mad#4 KEEP
│   ├─ 子块(m1,n0) s_cd=2：aic_wait_free(FIX·0·1) → 搬A(m1)+搬B(n0) → mad#5/#6
│   └─ 子块(m1,n1) s_cd=3：aic_wait_free(FIX·1·1) → A 复用 + 搬B(n1) → mad#7/#8
│   [MTE1] notify(MTE1→MTE2,0) ← L1[0] 四个子块全部读毕（:167）
│
├─ k_blk 1（s=1）：同构，mad#9..#16 全 KEEP；末尾 notify 归还 e1
│
├─ k_blk 2（s=0）★ 环回绕
│   [MTE2] wait(MTE1→MTE2,0)〔等 k_blk 0 末尾的归还〕→ GM→L1：A(+512B)、B(+512B) → l1a/l1b[0]
│   mad#17..#24 全 KEEP；末尾 notify 归还 e0（遗留，供下一个 tile 的首 k_blk）
│
├─ k_blk 3（s=1）★ 交付窗口
│   [MTE2] wait(MTE1→MTE2,1)〔k_blk 1 的归还〕→ GM→L1：A(+768B)、B(+768B)
│   ├─ 子块(m0,n0)：mad#25 KEEP / mad#26 ★is_last → UPDATE
│   │    → aic_store(s_epi=0)：L0C[0] ─FIX(F322BF16)→ AIV0 UB[0]，arrive(FIX·0·0)
│   ├─ 子块(m0,n1)：mad#27/#28(UPDATE) → aic_store(s_epi=1) → AIV1 UB[0]
│   ├─ 子块(m1,n0)：mad#29/#30(UPDATE) → aic_store(s_epi=2) → AIV0 UB[1]
│   └─ 子块(m1,n1)：mad#31/#32(UPDATE) → aic_store(s_epi=3) → AIV1 UB[1]
│
└─ 块尾：s_epi 回绕 0，epi_base=0；last_m_idx=0、last_n_idx=0（:169-171）
   AIV0：wait(MTE3,0,0) → copy_ub_to_gm D[0:64,0:64] → arrive(MTE3,0,0)
         wait(MTE3,0,1) → copy_ub_to_gm D[64:128,0:64] → arrive(MTE3,0,1)
   AIV1：D[0:64,64:128]、D[64:128,64:128] 同构
   get_next_block 返回 false → 三方退出（核 1 同构，行整体 +128）
```

### 3.8 流量与 FLOP 对账（每 tile）

| 量 | 算式 | 每 tile | 全 GEMM（2 tile） |
| :-- | :-- | :-- | :-- |
| GM 读 A | 4 k_blk × 32768 B | 131072 B | 262144 B = 256×512×2 ✔ 每元素一次 |
| GM 读 B | 4 × 32768 | 131072 B | 131072 B = 128×512×2 ✔ |
| GM 写 D | 4 交付 × 8192 | 32768 B | 65536 B = 256×128×2 ✔ |
| L1→L0 A | 4 k_blk × 4 次 × 8192 | 131072 B | 262144 B（每元素一次） |
| L1→L0 B | 4 × 8 次 × 8192 | 262144 B | 524288 B（**每元素两次**：N 拆分使 L0B 缓存失效，`:144`） |
| mmad | 32 条 | 32 × 524288 FLOP | 64 × 524288 = 33554432 = 2×M×N×K ✔ |
| 算术强度 | FLOP / GM 字节 | 16777216 / 294912 ≈ 56.9 FLOP/B | — |

### 3.9 变体练习（各附答案）

**V1：每核两个 tile（M=4096, N=256, K=512，其余假设不变）。** `num_m_blocks=32、num_n_blocks=2、num_blocks=64 > 32`，出现第二波。核 5 第 1 波 `block_idx=5`：`blocks_per_superrow = kGroupM×num_n_blocks = 8`（`scheduler.hpp:221`），`superrow=0、within=5`；`num_n_blocks=2 < kGroupN=8` 使 `n_groups=0`、`full_blocks=0`，落入 `:239-246` 的 else 分支：`m_local=5&3=1、n_local=5/4=1` → tile (m1, n1)。第 2 波 `block_idx=32+5=37`：`superrow=4、within=5` → tile (m17, n1)，`m_idx=2176`。第二个 tile 的三个衔接点：`epi_base` 从上一块末尾续接（本例 4 交付恰好转满 `kNumStages=4` 的环，回 0）；首个 `aic_wait_free(FIX·0·0)` 等的是 AIV0 的**真实归还**（预置已被上一块消费）；首个 `wait(MTE1→MTE2,0)` 消费上一块遗留的 e0 归还。跨块协议与环的"遗留事件"设计在此全部兑现。

**V2：K=520（K 尾块）。** `num_k_blocks = ceil_div(520,128) = 5`；k_blk 4 的 `actual_k = min(520-512,128) = 8` → GM→L1 只搬 128 行 × 8 列；每子块该块只有 1 条 mmad：`eff_mad_k = min(8-0,64) = 8`、`is_last`（`0+64 ≥ 8` ✔）→ UPDATE；tile 形状 `as_mad_aligned` 把 K 对齐到 16 而 `mad()` 收真值 8（`:138-139` vs `:149`）；`s` 环走 0,1,0,1,0——L1[0] 被第三次复用，等待链更长。对照 `tests/generators.py:519-541` 的 unaligned-k 用例（如 K=234）。

**V3：`kDirectStore=true`。** 启动变 `__mix__(1, 0)`（`:27`），无 AIV 子块；`:82` 由 AIC 设原子加（本例无累加，不触发）；`:129-131` 借 UB 消失；交付改为 `copy_l0c_to_gm(d.index(...).ptr(), l0c_tile, 64, 64, d.stride_outer, kL2CtrlStoreCd)`（`:156-159`，FIX 队列直写 GM）；AIV 分支 `:174` 直接 `return`。流程图的第④段变成"L0C → FIX → GM"，第⑤段由 AIC 兼任；`s_epi`/`epi_base` 照常计算但无人消费。前提由 `can_direct_store()`（`csrc/jit_kernels/heuristics.hpp:521-533`）把关——本例 N=128 对齐，允许。

**V4：`kResidentA=true`（换成 M=128 才合法）。** 设备断言要求 Normal 且编译期 `SHAPE_K`（`:40-41`）；host 闸门 `d.m <= block_m`（`heuristics.hpp:552-555`）——本例 M=256 会被拒（跨两个 tile 后 `reuse_a` 不恒成立），改 M=128 后：`l1a` 变单 stage 宽 `aligned(512,16) = 512`（`:59`，128×512×2 = 131072 B），`l1b` 基址 = `l1a.offset(1)`（`:60`）；块首一次搬完 A 的整条 K（`:109-110`）；K 循环内 A 搬运消失（`:117`），L1→L0A 源固定 `l1a[0]`、读偏移加 `k_blk_idx × BLOCK_K`（`:143`）。对照：`kResidentB` 对本例（K=512）非法——设备断言 `SHAPE_K <= kNumL0Stages × MAD_K = 128`（`:45`）与 host `d.k <= mad_k × num_l0_stages`（`heuristics.hpp:556-557`）都拒绝。

## 四、源码导读：对照 `bf16_gemm.hpp` 修正

今天源码导读的目标与往常相反：不是从代码里挖新知识，而是**给闭卷复述的每一段找出处、抓错误**。建议合上第三节，按站点顺序重读，每站 3～5 分钟，边读边用红笔改自己的图。

### 站点 1：启动与 shape（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:27-49`）

核对：`__mix__(1, kDirectStore ? 0 : 2)`（`:27`）；`SHAPE_M/N/K = 0` 时用运行时 shape（`:47-49`）；`kDirectStore && kWithAccumulation` 限 float（`:38`）、`kResidentB` 的 `SHAPE_K/BLOCK_N` 断言（`:45`）——你的图是否把启动形态与三方执行画对了？

### 站点 2：片上布局与断言（`:51-74`）

核对：`num_mad_cd_stages = (BLOCK_M/MAD_M)×(BLOCK_N/MAD_N)`（`:51`）；`l1a/l1b/l0a/l0b/l0c` 五个 `constexpr` 对象与 offset 分配（`:59-64`，`ascend/tile_ptr.hpp:37-38`）；五条 `static_assert`（`:66-70`）与 3.1 的验算逐条对数。你图里的 `l1b` 基址 65536、L0 stage 8192 B 是否写对？

### 站点 3：AIC 入口与预置（`:81-87`）

核对：`:82` 直写原子加的特例；`:83-84` 两组 `set_flags`——你画图时是否漏了"只有空闲事件需要预置、就绪绝不能预置"？

### 站点 4：任务循环外层（`:89-111`）

核对：`get_next_block` / `get_global_idx` / `get_actual_m/n`（`scheduler.hpp:141-142、113-115、129-139`）；`eff_shape_k/num_k_blocks/k_idx_base`（`:95-97`）；`kResidentA` 块首搬运（`:109-110`）。3.2 的两个 tile 与快路径条件是否复述正确？

### 站点 5：GM→L1 与事件对 1/2（`:112-121`）

核对：`s` 环的轮转式（`:112`）、`k_idx/actual_k`（`:113-114`）、wait/notify 的方向与事件号（`:116/:119/:121`）。3.3 的账本里"预置 e0 被 k_blk 0 消费、k_blk 2 消费 k_blk 0 的归还"是否说清了？

### 站点 6：子块循环 + L1→L0 + mad（`:123-152`）

核对：`s_cd` 连续扫描（`:124`）；三个 eff（`:125-126、137`）；`aic_wait_free` 只在 `k_blk_idx==0`（`:129-131`）；`s_ab` 环与 `is_first/is_last`（`:133-135`）；复用条件（`:142/:144`）；四个同步点（`:141/:146/:148/:152`）与 `mad` 实参表（`:149-151`）。3.3 的搬运计数（A 4 次 / B 8 次）与 3.4 的标志表逐格核对。

### 站点 7：交付窗口与续接（`:155-171`）

核对：交付只在最后一个 k_blk（`:155`）；`aic_store` 与 `copy_l0c_to_gm` 两分支（`:156-161`）；`s_epi` 推进（`:164`）与 `epi_base/last_m_idx/last_n_idx` 续接（`:169-171`）。3.5 的路由表是否与 `epilogue.hpp:43-44` 的两个算式一致？

### 站点 8：AIV 分支（`:173-192`）

核对：`:174` 的 `return`；`:177` `aiv_initialize`；`:179` AIV 的 `s_epi` 从 0 起；`:180` 同一 Scheduler 序列；`:190` 引用传参的 `aiv_drain`。3.6 的"continue 不阻止 s_epi 推进"是否画了？

### 站点 9：Epilogue 两个方法与调度器（`deep_gemm/include/deep_gemm/epilogue.hpp:37-47、49-97` + `scheduler.hpp:202-247`）

核对：`aic_wait_free`/`aic_store` 的路由算式（`epilogue.hpp:38、43-44`）；`aiv_drain` 的等待/搬运/归还三步（`:69、81-82、84`）；`aiv_initialize` 的批量预置（`:89-97`）；`get_swizzled_block` 的三个分支（快路径/尾部/swizzle）——3.9 V1 的 (m1,n1)/(m17,n1) 推算是否与 `:217-246` 一致？

**阅读检查点**（能全部回答再进入实践）：

1. `:122` 与 `:164` 分别对 `s_epi` 做了什么？为什么 k_blk 1、2 内的推进"无害"？
2. `:142` 的三个析取式在本例各取何值？哪个使 A 复用成立？
3. `:167` 为什么必须排在 m/n 子块循环之后？
4. `epilogue.hpp:84` 的归还 arrive 挂在 MTE3 上、排在 `copy_ub_to_gm` 之后——这给 `aic_wait_free` 的通过赋予了什么额外含义？
5. `scheduler.hpp:209-211` 快路径的触发条件是什么？本例为什么不走 swizzle？

## 五、实践步骤

### A. 有 Ascend 环境（约 25 分钟）

**步骤 1：打印真实配置，核对假设（约 10 分钟）。**

```bash
cd DeepGEMM-Ascend
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day27_run1.log
```

从输出（`csrc/jit_kernels/heuristics.hpp:749-753` 打印）找几个 `GemmConfig(block=...x...x..., mad=...x...x..., l1_stages=..., l0_stages=..., epi_stages=...)` 行，对照 3.1：`l0_stages` 恒 2；`mad` 大概率是 128×128×64、`l1_stages` 大概率 8——与假设的差异正是 3.1"诚实说明"的实证。**选一个真实配置，把 3.3～3.5 的三张表重算一遍**（结构不变，数字随配置变）。

**步骤 2：验证主例正确性（约 8 分钟）。**

```bash
DG_PRINT_CONFIGS=1 python - <<'EOF' 2>&1 | tee /tmp/day27_run2.log
import torch, torch_npu  # noqa
import deep_gemm
a = torch.randn(256, 512, dtype=torch.bfloat16, device='npu')
b = torch.randn(128, 512, dtype=torch.bfloat16, device='npu')
d = torch.empty(256, 128, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d)
torch.npu.synchronize()
ref = a.float() @ b.float().T
assert torch.allclose(d.float(), ref, atol=1, rtol=1e-2)
print("M=256, N=128, K=512 OK")
EOF
```

**步骤 3：跑边界组，对应 V2（约 7 分钟）。** `python tests/test_bf16_gemm.py unaligned-k --skip-prof`（`tests/test_bf16_gemm.py:31-32`），在 `tests/generators.py:519-541` 的用例里挑一个（如 K=234），手算它的尾块 `actual_k`/`eff_mad_k`，与测试通过互为印证。

**步骤 4（可选）：** 加 `DG_JIT_PRINT_COMPILER_COMMAND=1` 重跑步骤 2，在实例化命令里找到 `bf16_gemm_impl<...>` 的 26 个模板实参位置（第 12 天生成的代码）。

### B. 无 Ascend 环境：静态复述与手工演算（主任务，约 55 分钟）

**B-1 闭卷画五段流程图（必做，约 20 分钟，当天产出核心）。** 合上一切材料，画出 2.4 骨架的展开版：每段标注执行者、队列、关键行号、四组队列事件与两组块内事件、五个环。画完对照 2.2/2.3 与 3.7 订正，把每处错误归因（漏了某行/记错条件/概念混淆）写在图边。

**B-2 数字演算（必做，约 15 分钟）。** 不看 3.2～3.8，对 `M=256, N=128, K=512` 重算：Tile 网格与核分配、`num_k_blocks`、每子块与全 tile 的 mmad 数、每 k_blk 的 A/B 搬运次数（代入 `:142/:144`）、4 次交付的路由表、FLOP 与流量对账。对照订正。

**B-3 事件链核对（必做，约 8 分钟）。** 默写 L1 空闲事件（e0/e1）从预置到遗留的完整收支账本（3.3 的表），并回答：若有第二个 tile，它的前两个 `wait` 各消费什么？UB 的 4 个归还 arrive 同样记一份账本。

**B-4 复述核对器（必做，约 15 分钟）。** 补齐 3 处空格（对照注释里的源码行填，不要凭记忆），运行后应复现 3.7 的时间线（核 0 部分逐行一致）：

```python
# day27_sim.py —— 无任何依赖，python3 直接运行
M, N, K = 256, 128, 512
BM, BN, BK = 128, 128, 128                 # BLOCK_M/N/K
MM, MN_, MK = 64, 64, 64                   # MAD_M/N/K
CORES, L1S, L0S = 32, 2, 2                 # kNumCores / kNumL1Stages / kNumL0Stages
EPI_STAGES, AIVS, ELEM = 2, 2, 2           # kNumEpilogueStages（每 AIV）/ AIV 数 / sizeof(bf16)
ceil_div = lambda a, b: (a + b - 1) // b

num_cd = (BM // MM) * (BN // MN_)          # num_mad_cd_stages（bf16_gemm.hpp:51）
kNS = EPI_STAGES * AIVS                    # epilogue.kNumStages（epilogue.hpp:24）
num_kb = ceil_div(K, BK)
num_mb, num_nb = ceil_div(M, BM), ceil_div(N, BN)
print(f"网格 {num_mb}x{num_nb}={num_mb * num_nb}，CD stages={num_cd}，"
      f"epi kNumStages={kNS}，mmad/tile={num_kb * num_cd * (BK // MK)}")

for core in range(min(num_mb * num_nb, CORES)):
    blk = core                             # 本例块数 ≤ 32：第 0 波 block_idx = core_id（scheduler.hpp:142）
    m_blk, n_blk = ____                    # ① scheduler.hpp:209-211 快路径（num_mb < kGroupM=4）
    m_idx, n_idx = m_blk * BM, n_blk * BN
    am, an = min(M - m_idx, BM), min(N - n_idx, BN)
    epi_base, mad_no = 0, 0
    print(f"\n== 核 {core}：tile (m{m_blk},n{n_blk})，m_idx={m_idx}，actual {am}x{an}")
    for kb in range(num_kb):
        s, k_idx, s_epi = kb % L1S, kb * BK, epi_base          # :112 / :113 / :122
        wait_src = "预置(:83)" if kb < L1S else f"k_blk {kb - L1S} 末尾的归还(:167)"
        print(f"[k_blk {kb}] s={s}：wait(MTE1→MTE2,{s})〔{wait_src}〕")
        print(f"   GM→L1：A +{ELEM * (m_idx * K + k_idx)}B、B +{ELEM * (n_idx * K + k_idx)}B → l1a/l1b[{s}]")
        for mm in range(0, am, MM):
            for nn in range(0, an, MN_):
                s_cd = ____                # ② bf16_gemm.hpp:124：s_cd 连续扫描
                if kb == 0:
                    print(f"   子块(m{mm},n{nn}) s_cd={s_cd}：aic_wait_free(FIX·AIV{s_epi % AIVS}·UB{s_epi // AIVS})")
                for km in range(0, BK, MK):
                    copy_a = ____          # ③ bf16_gemm.hpp:142 的三个析取式
                    copy_b = (BM == MM) or (BN != MN_) or (BK > MK * L0S) or (mm == 0)
                    mad_no += 1
                    first = kb == 0 and km == 0
                    last = kb == num_kb - 1 and km + MK >= BK
                    print(f"     k_mad {km} s_ab={(km // MK) % L0S}：{'搬A' if copy_a else 'A复用'} "
                          f"{'搬B' if copy_b else 'B复用'} → mad#{mad_no} "
                          f"{'UPDATE' if last else 'KEEP'}{' +clear_C' if first else ''}")
                if kb == num_kb - 1:
                    print(f"     交付：aic_store(s_epi={s_epi}) → AIV{s_epi % AIVS} UB[{s_epi // AIVS}]，"
                          f"写 D[{m_idx + mm}:{m_idx + mm + MM},{n_idx + nn}:{n_idx + nn + MN_}]")
                s_epi = 0 if s_epi == kNS - 1 else s_epi + 1   # :164 每个子块推进（含无交付的 k_blk）
        print(f"   notify(MTE1→MTE2,{s}) ← L1[{s}] 全部子块读取完毕（:167）")
    epi_base = s_epi
    print(f"   块尾：epi_base = {epi_base}（:169）")
```

（参考：① `divmod(blk, num_nb)`；② `(mm // MM) * (BN // MN_) + nn // MN_`；③ `(BN == MN_) or (BK > MK * L0S) or (nn == 0)`。扩展任务：把 `M, N, K` 改成 3.9 的 V1/V2，观察输出变化——V2 需把 `range(0, BK, MK)` 的上界换成 `actual_k`、`last` 的 `BK` 换成 `actual_k` 才完整。）

**B-5 变体练习（必做，约 7 分钟）。** 闭卷回答 3.9 的 V1～V4 各两个问题（V1：核 5 领哪两个 tile、第二个 tile 的三个衔接点；V2：尾块的 `eff_mad_k` 与 tile 对齐值；V3：流程图哪两段被裁、谁兼任写回；V4：为什么 M=256 非法、K=512 对 `kResidentB` 为什么非法），再对照 3.9 订正。

**B-6 判定题自验（必做，约 5 分钟）。** 判断真假并给出行号依据（答案见第七节末尾）：

1. 本例 4 个子块的首条 mmad（各自 k_blk=0、k_mad=0）都带 `clear_C`。（  ）
2. k_blk 2 的 `wait(MTE1→MTE2, 0)` 消费的是启动预置事件。（  ）
3. 本例 A 与 B 的 L1→L0 搬运次数相同。（  ）
4. 4 次 `aic_store` 都发生在最后一个 k_blk，4 次 `aic_wait_free` 都发生在 k_blk=0。（  ）
5. AIV1 写回的 GM 区域是 D[0:64,64:128] 与 D[64:128,64:128]。（  ）
6. 本例每个核只有一个 tile，因此 4 次 `aic_wait_free` 全部靠 `aiv_initialize` 的预置直接通过。（  ）
7. `s_epi` 在每个 k_blk 开头重置为 `epi_base`，并在每个子块循环体末尾推进一格。（  ）
8. 若删除 `:121` 的 wait，MTE1 可能读到尚未写完的 L1。（  ）
9. `kDirectStore=true` 时，4 次交付改由 AIC 的 `copy_l0c_to_gm` 直写 GM，AIV 与 UB 从图中消失。（  ）
10. `kResidentB` 对本例（K=512）是合法选择。（  ）

**B-7 口头讲解（选做但强烈建议，约 10 分钟）。** 对着 3.7 的时间线，不看任何材料向想象中的同事讲一遍"一个输出 Tile 从调度到写回的完整生命周期"——这是通过标准的直接预演。

## 六、当天产出

学习计划要求今天"完整复述 BF16 GEMM"。按下述模板整理（第三节是参考答案）：

```text
# DeepGEMM-Ascend 第 27 天记录：完整复述 BF16 GEMM

## A. 闭卷五段流程图（订正版）
（贴 B-1 的图：五段 × 执行者/队列/行号；四组队列事件 + 两组块内事件；五个环；
  红笔订正处标注错误归因：漏行 / 记错条件 / 概念混淆）

## B. 一个输出 Tile 的生命周期演算
shape/配置：M=＿ N=＿ K=＿；BLOCK=＿ MAD=＿ L1=＿ L0=＿ epi=＿
调度：num_blocks=＿，干活核=＿，m_idx/n_idx/actual=＿
K 循环：num_k_blocks=＿，每子块 mmad=＿，全 tile=＿，FLOP 守恒：＿ ✓
搬运：每 k_blk A＿次 / B＿次（复用条件）；GM→L1 源偏移与 stage 表
事件账本：L1 空闲 e0/e1 的发出-消费全表；UB 归还的账本
标志表：一个子块的 8 条 mmad（is_first/is_last/unit_flag/clear_C）
交付路由：4 行（s_cd → s_epi → aiv/stage → GM 区域）
对账：GM 读 A/B、写 D、L1→L0、FLOP、算术强度
（有环境时贴上 DG_PRINT_CONFIGS 的真实 GemmConfig 与差异说明）

## C. 变体记录
V1 第二个 tile 的衔接点 / V2 K 尾块 / V3 直写 / V4 常驻 —— 各两三行

## D. 仍不理解的三个问题（必填）
问题 1：＿＿（归类：CANN 指令级 / 仓库可查但未读 / 后续天主题；打算去哪核实）
问题 2：＿＿
问题 3：＿＿
```

**"三个问题"的候选示例**（从本周各天定位过的边界中挑，也可以写自己的）：

1. M 队列写 L0C 与 FIX 队列读 L0C 之间的跨队列次序：仓库内无显式事件对，可见线索只有 `ENABLE_UPDATE` 的"提交/消费"配对与 stage 环深度，确切语义在 `asc_mmad`/`asc_copy_l0c2ub` 的 CANN 文档（第 24/25 天已定位边界）——归类：CANN 指令级；
2. `ENABLE_KEEP`/`ENABLE_UPDATE` 的寄存器级行为：部分和保存在哪、UPDATE 何时对外可见——仓库只以"两端 UPDATE、中间 KEEP"的用法呈现（第 22/24 天）——归类：CANN 指令级；
3. `as_mad_aligned` 抬升后的尾部对齐列（如 `eff_mad_k=8` 时 tile 的 `shape_k=16`、`eff_mad_n` 对齐列）为何不污染结果——第 19/22 天以三级裁剪链描述，硬件行为未见——归类：CANN 指令级；
4. `kL2CtrlA/B/Cd` 各档 L2 模式（`NORMAL_FIRST_VICTIM/PERSISTENT/...`）的性能语义——归类：后续天主题（第 50～52 天实验）；
5. `__mix__` 子块调度与 `asc_get_sub_block_id` 的硬件细节——归类：CANN 文档。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. 五段流程各由谁执行、排入哪些队列？写出每段的关键函数与行号。
2. 本例有几个输出 Tile？哪些核干活？为什么映射走快路径而不走 swizzle？
3. 写出本例一个 tile 的 K 循环结构（`num_k_blocks`、每子块 mmad 数、全 tile mmad 数），并用 FLOP 守恒验证。
4. k_blk 2 的 `wait(MTE1→MTE2, 0)` 等的是谁发出的事件？预置的 0 号事件被谁消费了？若无下一个 tile，k_blk 2/3 末尾发出的归还事件去了哪里？
5. 为什么本例 A 的 L1→L0 搬运次数是 B 的一半？请从 `:142/:144` 的条件与数据依赖两方面回答。
6. 给出子块 (m1,n1)（s_cd=3）第 8 条 mmad 的全部标志与语义；它写入哪个 L0C stage？随后 `aic_store` 把它交给哪个 AIV 的哪格 UB？
7. 三方（AIC/AIV0/AIV1）如何不传递 `s_epi` 就保持一致？若 `kNumEpilogueStages=1`，哪条断言会在编译期拒绝？
8. V1 中核 5 的第二个 tile (m17,n1)：它的首个 `wait(MTE1→MTE2,0)` 与首个 `aic_wait_free(FIX·0·0)` 各消费什么事件？
9. （开放）你记录的三个"仍不理解"分别是什么？各归类到哪一层（CANN 指令级 / 仓库可查但未读 / 后续天主题）？

### 参考答案

1. ① 任务调度：AIC 与两个 AIV 各自执行 `Scheduler::get_next_block`（`bf16_gemm.hpp:89`，纯标量）；② GM→L1：AIC 的 MTE2 队列（`:117-118`）、L1→L0：MTE1 队列（`:143/:145`）；③ MAD：AIC 的 M 队列（`:149-151`）；④ Epilogue：AIC 的 FIX 队列（`aic_wait_free` `:129-131`、`aic_store` `:161` + `epilogue.hpp:42-47`）；⑤ 写回：AIV 的 `kPipe`（Identity 时即 MTE3）与 MTE3 队列（`epilogue.hpp:69、81-82`）。
2. 2 个 tile（`num_blocks = 2×1`，`scheduler.hpp:91-93`）；核 0 与核 1（`block_idx = core_id < 2`，`:142、191-195`）；核 2～31 首次即退出。快路径条件 `local_num_m_blocks < kGroupM = 4`（`:209-211`），本例 `num_m_blocks = 2 < 4`。
3. `num_k_blocks = ceil_div(512,128) = 4`；每子块 4 k_blk × 2 k_mad = 8 条；全 tile 4 子块 × 8 = 32 条。守恒：32 × 2×64×64×64 = 16777216 = 2×128×128×512 ✔。
4. 等 k_blk 0 末尾 `:167` 的归还事件（MTE1 发出，因为它是 L1[0] 的读者）。预置的 0 号事件（`:83`）已被 k_blk 0 的首次 wait 消费。k_blk 2/3 末尾的归还事件成为"遗留"——本例无下一个 tile，随 kernel 结束废弃；若有，则被下一个 tile 的首个 k_blk 消费。
5. A 的 L1→L0 数据只依赖 `(m_mad, k_mad)`：`:142` 的条件中 `n_mad_idx == 0` 使 A 只在 n_mad=0 的两个子块搬运（每 k_blk 4 次）；B 依赖 `(n_mad, k_mad)`，n_mad 循环覆写了 `l0b[s_ab]` 的内容，`:144` 的 `BLOCK_N != MAD_N` 项恒真，每个子块都要重搬（每 k_blk 8 次）。A 每元素跨 L1→L0 一次、B 两次（3.8 对账）。
6. 第 8 条 mmad：`k_blk_idx=3`、`k_mad_idx=64`，`is_last`（3==3 且 64+64≥128）→ `ENABLE_UPDATE`、`clear_C=false`；K 遍历首条（k_blk 0、k_mad 0）带过 `clear_C`。它累加进 `l0c[s_cd=3]`（64×64 FP32）；`aic_store(s_epi=3)` 把结果经 `copy_l0c_to_ub`（F322BF16）交给 **AIV1 的 UB[0]**（`aiv_id=1、stage_idx=0`，`epilogue.hpp:43-44`），写 D[64:128,64:128]。
7. 机制一：三方持有相同的 `Scheduler`、按相同顺序遍历相同块序列，块内 m/n 子块扫描同构（`scheduler.hpp:10-12`、`epilogue.hpp:55-56`）；机制二：相同的推进规则——每子块前进一格、`kNumStages` 回绕（AIC `:122/:164` + `epi_base :169`；AIV `epilogue.hpp:56` 引用传参，continue 不阻止推进）。`kNumEpilogueStages=1` 时 `kNumStages = 2 < num_mad_cd_stages = 4`，违反 `epilogue.hpp:33` 的断言，JIT 编译期报错。
8. 首 k_blk 的 `wait(MTE1→MTE2,0)` 消费上一 tile 遗留的 e0 归还（上一 tile k_blk 2 末尾 `:167` 发出）；首个 `aic_wait_free(FIX·0·0)` 等 AIV0 的真实归还 arrive（`epilogue.hpp:84`，上一 tile 的 D[0:64,0:64] 排水完成后发出；`aiv_initialize` 的预置已被上一 tile 消费）。
9. 开放题，对照第六节 D 的候选示例自查：每个问题都应能写出"问题一句话 + 归类 + 打算去哪核实"。

**B-6 判定题答案**：

1. **真**。`is_first` 在 m/n 子块循环之内计算（`:134`），每个子块的 K 遍历各自首条清零。
2. **假**。预置的 0 号已被 k_blk 0 消费；k_blk 2 等的是 k_blk 0 末尾 `:167` 的归还。
3. **假**。每 k_blk A 4 次、B 8 次（`:142` 的 `n_mad==0` vs `:144` 的 `BLOCK_N != MAD_N` 恒真）。
4. **真**。`:129` 的条件 `k_blk_idx == 0`、`:155` 的 `k_blk_idx == num_k_blocks - 1`。
5. **真**。s_epi 1、3 归 AIV1（`:43` 的 `s_epi % 2`），对应 `epilogue.hpp:57` 通过、`:81-82` 写出。
6. **真**。每核一个 tile，`aiv_initialize`（`epilogue.hpp:89-97`）预置的 4 个标志未被消费过。
7. **真**。`:122` 重置、`:164` 推进——k_blk 1、2 内的推进随后被重置覆盖，无害。
8. **真**。MTE1 会越过 GM→L1 的完成点读 L1[s]，读到旧值/半写数据——写-读竞争（第 23 天 3.7 反例 2）。
9. **真**。`:156-159` 的 `copy_l0c_to_gm`；`:174` 的 `return` 与 `__mix__(1, 0)`。
10. **假**。设备断言 `SHAPE_K <= kNumL0Stages × MAD_K = 128`（`:45`）；host `d.k <= mad_k × num_l0_stages`（`heuristics.hpp:556-557`）都拒绝 K=512。

## 八、通过标准自评

学习计划的通过标准是：**能讲清楚一个输出 Tile 从调度到写回的完整生命周期**。逐项自评，全部达成即通过：

1. **闭卷复述**：不看源码与第三节，能在 40 分钟内画出五段流程展开图（含执行者、队列、同步点、五个环），并在 10 分钟内口头讲完一遍（B-1/B-7）；
2. **数字自洽**：B-2 的全部表格能独立重算；换一个 shape（如 M=128, N=256, K=384）也能当场算出网格、循环结构、搬运计数与交付路由；
3. **事件链**：对任一 wait/notify/arrive，能立刻说出"谁发出、谁消费、守护哪份缓冲"——3.3 的账本与第四节的九个站点就是抽查清单；
4. **生命周期叙述**：能回答"一个输出 Tile 经历的全部状态变化"：领任务 → 4 个 k_blk 的搬运与 32 条 mmad（L0C 从 clear_C 到 UPDATE）→ 4 次交付（UB 借还）→ AIV 排水写回 → 归还与续接；
5. **错误归因**：订正时把每处错误归了类，知道该回哪一天补课；
6. **三个问题**：记录了三个仍不理解的问题并归了类——其中 CANN 指令级的问题本周已"定位边界"（知道以仓库用法为准、确切答案在文档），这正是第 24/25 天方法论的正确延续。

明天（第 28 天）是机动日：建议重新运行 BF16 测试（`python tests/test_bf16_gemm.py normal --skip-prof` 等），把测试输出与本周读到的 Kernel 行为一一对应——这是闭卷复述的硬件版。第 5 周将转向测试体系与低精度 GEMM：今天复述的主线结构（两层 K 循环、事件配对、L0C 累加链、Epilogue 借还）在 FP8/FP4 路径中几乎原样复用，多出来的只是 scale 因子与反量化。
