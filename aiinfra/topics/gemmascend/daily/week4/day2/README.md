# 第 4 周 · 第 2 天（总第 23 天）：理解多级流水与同步

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 4 周"BF16 Kernel——MAD、流水线与 Epilogue"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 45 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

昨天（第 22 天）我们拆解了 K 维两层循环，但刻意把 `bf16_gemm_impl` 里出现的每一个 `asc_sync_wait` / `asc_sync_notify` 当成黑盒——只记住"它在等谁、通知谁"。今天打开这个黑盒。全部同步原语集中在一个只有 87 行的头文件 `deep_gemm/include/deep_gemm/ascend/sync.hpp` 里；而它们的使用点，几乎都落在 `bf16_gemm.hpp:112-168` 这段 K 循环中。核心问题只有一个：**AIC 上的 MTE2（GM→L1 搬运）、MTE1（L1→L0 搬运）和 M（mmad 计算）是三条异步执行的指令队列，却共享同一批物理缓冲（L1 的各 stage、L0A/L0B 的各 stage）——怎样用最少的同步让它们互不踩踏，又尽量重叠执行？** 答案是"每份缓冲配两个有方向的 events：空闲 + 就绪"，再用 stage 环下标循环复用少量事件号。当天产出是一张 **MTE1/MTE2/M 流水线时序图**（参考答案见 3.6 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 解释 AIC 上"一份标量程序、多条异步指令队列"的执行模型：MTE2/MTE1/M 各自按队列内顺序异步执行，`copy_gm_to_l1`、`copy_l1_to_l0a/b`、`mad` 分别排入哪条队列；并说明为什么共享缓冲上的"写-读"和"读-写"会构成数据竞争；
2. 写出 `asc_sync_notify` / `asc_sync_wait` 的语义契约：第一个参数是事件的**发出方**（源 pipe），第二个参数是**接收方**（目标 pipe）；notify 作为指令排入**源**队列、在源队列之前的指令全部完成后才发出事件，wait 挡住**目标**队列中排在它之后的指令；并能用"否则内核必然出错"的方式论证 notify 为什么必须按序执行；
3. 默写 `bf16_gemm_impl` AIC 分支的**四组配对事件**（L1 stage 空闲/就绪、L0 stage 空闲/就绪）的方向、行号与含义（`bf16_gemm.hpp:83-84, 116, 119, 121, 141, 146, 148, 152, 167`），并解释为什么只有"空闲"事件需要启动时用 `set_flags` 预置、缺少预置会死锁；
4. 把第 22 天的 stage 环下标 `s` / `s_ab` 映射到事件号上，说明"事件号 = 缓冲编号、随环循环复用"，并给出 `kNumL1Stages` 的两重上限：每方向事件号数量 `kNumPipeEvents = 8`（`common.hpp:61`、`heuristics.hpp:66, 168-172`）与 L1 容量（`bf16_gemm.hpp:66` 的 `static_assert`），会用 `heuristics.hpp:171-172` 的公式算出 128×128×128 块的 stage 数；
5. **手工画出** `kNumL1Stages=2 / kNumL0Stages=2` 的稳态流水线时序图（含预热段和每个时刻的事件收发），并能改画出 `kNumL1Stages=1` 的串行化对照，说明 stage 数从 2 到 8 改变的是什么（领先量/抖动缓冲，而不是稳态节拍）；
6. 分别给出"删掉某个 wait 会产生什么竞争"（覆写正在被读/算的缓冲）与"加一个多余的全量 barrier 会损失什么"（跨块重叠消失）的具体例子，解释同步**数量与位置**的权衡；
7. 有 Ascend 环境时，用 `DG_PRINT_CONFIGS=1` 记录真实选中的 `l1_stages/l0_stages` 并与两个上限核对；无环境时完成纸面推演与源码对照。

**当天产出：MTE1/MTE2/M 流水线时序图**（模板见第六节，参考答案见 3.6 节）。

## 二、预备知识

- **第 5 天**：GM/L1/L0A/L0B/L0C 的分层职责，MTE1、MTE2、M 等流水线的大致概念（今天把它们精确化）。
- **第 12 天**：`GemmConfig` 中 `num_l1_stages`、`num_l0_stages` 字段的含义。
- **第 15 天**：`bf16_gemm_impl` 模板参数中的 Stage 组（`kNumL1Stages`、`kNumL0Stages`、`kNumEpilogueStages`）。
- **第 16 天**：`l1a/l1b/l0a/l0b` 的多阶段缓冲声明与 offset 分配（`bf16_gemm.hpp:59-64`），容量断言（`:66-69`），L1 512KB / L0A、L0B 各 64KB（`common.hpp:54-58`）。
- **第 19 天**：`copy_gm_to_l1`（GM→L1，MTE2 队列）与 `copy_l1_to_l0a/b`（L1→L0，MTE1 队列）的签名和规模参数。
- **第 22 天**：K 维两层循环的控制量（`k_blk_idx`、`k_mad_idx`、`actual_k`、`eff_mad_k`）与两个 stage 环下标 `s`、`s_ab` 的转速——今天的同步正是挂在这两个环上；`is_first`/`is_last` 与 L0C 累加链留给明天。
- **C++ 语法**：函数模板与模板参数默认值、`#pragma unroll`、条件表达式取模（`s == N-1 ? 0 : s+1`）。能对照读懂即可。

另外说明：`asc_sync_notify` / `asc_sync_wait` / `asc_sync_intra_arrive` 等底层函数声明在 CANN 工具链的 `c_api/asc_simd.h`（`sync.hpp:3` 引入），该头文件在**本仓库之外**。本文对指令级细节的表述以仓库内的用法能反推出的契约为准，不凭空补充硬件行为。

## 三、详细讲解

### 3.1 为什么需要同步：三条异步队列共享一批缓冲

先回忆整体数据流（第 5 天）与第 19 天的搬运分工：

```text
GM ──MTE2──> L1 ──MTE1──> L0A/L0B ──M──> L0C ──Epilogue──> GM
```

关键事实是：AIC 上运行的是**一份**标量程序（`bf16_gemm_impl` 的循环体），但它下发的搬运/计算指令进入**不同的硬件指令队列**，各队列异步、按各自内部顺序执行——MTE2 队列执行 GM→L1 搬运，MTE1 队列执行 L1→L0 搬运，M 队列执行 `asc_mmad`。标量程序只保证"下发顺序"，不保证"完成顺序"。

问题来了：L1 里只有 `kNumL1Stages` 份 `BLOCK_M×BLOCK_K + BLOCK_N×BLOCK_K` 的缓冲（`bf16_gemm.hpp:59-60` 的 `l1a/l1b`），K 循环每圈写一份、读一份。假设没有任何同步、`kNumL1Stages=2`、`s = k_blk_idx mod 2`：

```text
K 块:      i         i+1        i+2
MTE2:     写 L1[s]   写 L1[s']  写 L1[s]   ← 第 i+2 次写入与下面同一段物理缓冲
MTE1:     读 L1[s]   读 L1[s']  读 L1[s]
               └── 若 MTE1 还没读完，i+2 的写入就落进来 → 读到新旧混合的数据
```

这就是**写-读竞争**。对称地，L0A/L0B 里只有 `kNumL0Stages` 份缓冲，MTE1 写入与 M 队列的 mmad 读取之间也存在同样的竞争（读-写竞争，mmad 读到被覆写一半的操作数）。同步要解决的就是这两类竞争——而且**只**需要解决这两类：缓冲有几份，同步就配几对事件，多了是浪费，少了是踩踏。

### 3.2 `sync.hpp` 的三层接口

`deep_gemm/include/deep_gemm/ascend/sync.hpp` 全文 87 行，提供三层封装（从底到顶）：

**第一层：裸 intrinsics 的成对薄封装。** `set_flags` / `wait_flags`（`:63-73`）把"一次预置/等待 `size` 个连续事件号"打包成循环：

```cpp
template <pipe_t pipe, pipe_t tpipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void set_flags() {
    for (uint32_t i = 0; i < size; i++)
        asc_sync_notify(pipe, tpipe, static_cast<event_t>(offset + i));
}
```

注意 `set_flags` 的本质是**连发 notify**——启动时把 N 个"缓冲空闲"事件预先发出去，让第一圈循环不必等待根本不存在的"上一轮"。`wait_flags`（`:69-73`）对称地连发 wait，仓库里用在内核收尾的排空（见站点 6）。`set_intra_blocks` / `wait_intra_blocks`（`:75-85`）是块内同步的批量版本。

**第二层：块内（intra-block）同步。** `set_intra_block` / `wait_intra_block`（`:29-40`）封装 `asc_sync_intra_arrive` / `asc_sync_intra_wait`，参数是 `(pipe, subblock_id, flag_id)`，flag 编码为 `((subblock_id & 1) << 4) | (flag_id & 0xf)`（`:30`）——低 4 位是 16 个 flag 号，第 4 位选择对端子块。它连接的是**同一个 block 内的 AIC 与 AIV**（`bf16_gemm.hpp:27` 的 `__mix__(1, kDirectStore ? 0 : 2)`：一个 block 由 1 个 AIC 加 2 个 AIV 组成），与连接"AIC 内部各队列"的 `asc_sync_notify/wait` 是两套机制。AIV 侧等待时忽略 `subblock_id`（`:34-39` 的注释："AIV ignores subblock_id for intra-block waiting."）。本节先记住接口，用途见 3.8 与第 25 天。

**第三层：带类型的命名事件（新接口）。** `PipeEvent<kSrc, kDst>`（`:10-13`）与 `IntraBlockEvent<kSubblock>`（`:44-47`）配上 `notify` / `wait`（`:15-27, 49-61`）：把事件的**方向**编码进类型，"参数类型检查每次调用写出的 PIPE 方向"（`:7-9` 注释），并支持 `kNumStages` 一次展开多个 stage。文件头注明"existing kernels will migrate incrementally"——目前 `bf16_gemm.hpp` 等主线仍直接用裸 `asc_sync_*`，而 `mega_moe.hpp:62-96` 已经用上了命名事件（`kL1AFull`、`kL1AEmpty`、`kL0AFull`…，Full=就绪、Empty=空闲，命名与本文的术语一一对应）。读旧代码用第一层的心智模型即可。

### 3.3 事件的语义：方向 + 事件号 + 队列顺序

从仓库全部用法（结合 `mega_moe.hpp:62-92` 的 Full/Empty 命名）可以反推出 `asc_sync_notify` / `asc_sync_wait` 的契约：

```text
asc_sync_notify(源, 目标, 事件号)
  · 作为一条指令排入【源】pipe 的指令队列，按队列内顺序执行；
  · 执行到它时（源队列中排在它之前的指令已全部完成），向【目标】发出该事件号的事件。

asc_sync_wait(源, 目标, 事件号)
  · 作为一条指令排入【目标】pipe 的指令队列；
  · 目标队列执行到它时停住，直到【源】的同号事件到达才放行后续指令。
```

三个要点：

**要点 1：事件是有方向的，方向由两个 pipe 参数表达。** `asc_sync_wait(PIPE_MTE1, PIPE_MTE2, s)` 的意思是"等 MTE1 发给 MTE2 的第 `s` 号事件"，而不是"MTE1 在等"。判断方向有一个可靠办法：看**谁有资格宣布这件事**。"L1 stage 空闲"只能由读完它的 MTE1 宣布，所以方向是 MTE1→MTE2；"L1 stage 就绪"只能由写完它的 MTE2 宣布，方向是 MTE2→MTE1。`mega_moe.hpp:62-63` 的命名直接印证：`PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull`（Full=数据就绪，MTE2 产出）、`PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1AEmpty`（Empty=缓冲空闲，MTE1 归还）。

**要点 2：notify 在源队列中按序执行，是正确性的硬要求。** 看 `bf16_gemm.hpp:117-121`：

```cpp
copy_gm_to_l1(l1a[s], ...);                    // MTE2 指令
copy_gm_to_l1(l1b[s], ...);                    // MTE2 指令
asc_sync_notify(PIPE_MTE2, PIPE_MTE1, s);      // :119
...
asc_sync_wait(PIPE_MTE2, PIPE_MTE1, s);        // :121  下一节细讲
```

如果 `:119` 的 notify 在下发瞬间就触发（而不是排进 MTE2 队列、等两条搬运完成后触发），那么 `:121` 的 wait 会立即放行，MTE1 将去读**还没写完**的 L1——竞争、结果错误。所以"事件的含义 = 源队列中位于 notify 之前的所有指令已完成"不是可选项，而是这套写法能工作的前提。`transform_sf.hpp:582-583` 的 `notify(PIPE_MTE2, PIPE_V, …)` + `wait(PIPE_MTE2, PIPE_V, …)`（V 等 MTE2 把 UB 写完才开始转置）是同一论证的 AIV 版本。

**要点 3：相邻的 notify/wait 不是"自己发自己等"。** `:119` 与 `:121` 方向相同、事件号相同，看起来像一对自言自语——其实它们排入**两条不同的队列**：`:119` 进 MTE2 队列（发完就绪信号），`:121` 进 MTE1 队列（挡住后续 L1→L0 搬运）。标量源代码把它们写在相邻两行，只因为两件事在**下发顺序**上紧挨着。同理，`epilogue.hpp:78-79` 的 `notify(kPipe, PIPE_MTE3, …)` + `wait(kPipe, PIPE_MTE3, …)`：前者排入 V 队列宣布 UB 数据就绪，后者挡住 MTE3 的 `copy_ub_to_gm`。

同方向、同事件号的一次 notify 与一次 wait 构成一次**会合**（rendezvous）；stage 环每圈复用同一组事件号，所以这少量旗标能支撑任意长的 K 循环。

### 3.4 BF16 Kernel 的四组配对事件

现在把 `bf16_gemm.hpp:81-168`（AIC 分支）里所有同步点收进一张表。**下标 `s` 是 L1 stage 环（`k_blk_idx mod kNumL1Stages`），`s_ab` 是 L0 stage 环（`(k_mad_idx/MAD_K) mod kNumL0Stages`）**（第 22 天）；事件号直接用这两个下标。

| 组 | 事件含义 | 方向（源→目标） | 启动预置 | 循环内 wait | 循环内 notify | 守护的资源 |
|---|---|---|---|---|---|---|
| 1 | L1 stage `s` **空闲** | MTE1 → MTE2 | `set_flags`（`:83`，预置 `kNumL1Stages` 个） | `:116` | `:167` | `l1a[s]/l1b[s]`：写入前等读完 |
| 2 | L1 stage `s` **就绪** | MTE2 → MTE1 | 无 | `:121` | `:119` | 同上：读取前等写完 |
| 3 | L0 stage `s_ab` **空闲** | M → MTE1 | `set_flags`（`:84`，预置 `kNumL0Stages` 个） | `:141` | `:152` | `l0a[s_ab]/l0b[s_ab]`：写入前等算完 |
| 4 | L0 stage `s_ab` **就绪** | MTE1 → M | 无 | `:148` | `:146` | 同上：计算前等搬完 |

逐条说明：

**为什么只有"空闲"事件需要预置？** 第一圈 K 循环之前不存在"上一轮"——MTE1 还没读过任何 L1，无从宣布空闲；但 `:116` 的 wait 已经在等"stage 0 空闲"。`:83-84` 的两条 `set_flags` 在内核启动时把全部"空闲"事件预先发出，第一圈才能通过。**漏掉预置的后果不是竞争而是死锁**：第一个 wait 永远等不到事件。反过来，"就绪"事件绝不能预置——就绪必须由真实搬运产生，预置就绪等于宣布空缓冲里有数据。

**一次 k_blk 迭代的完整下发顺序**（单 M/N 子块、2 条 mmad 的情形；`↷ 队列` 表示该指令排入哪条队列）：

```text
 1. wait(MTE1→MTE2, s=0)        ↷ MTE2   等 L1[0] 空闲（启动预置已满足）     :116
 2. copy_gm_to_l1(l1a[0], …)    ↷ MTE2   GM→L1 写 A 片段                     :117
 3. copy_gm_to_l1(l1b[0], …)    ↷ MTE2   GM→L1 写 B 片段                     :118
 4. notify(MTE2→MTE1, 0)        ↷ MTE2   完成后宣布 L1[0] 就绪                :119
 5. wait(MTE2→MTE1, 0)          ↷ MTE1   等 L1[0] 就绪                        :121
 6. wait(M→MTE1, s_ab=0)        ↷ MTE1   等 L0[0] 空闲（启动预置已满足）      :141
 7. copy_l1_to_l0a/l0b(→L0[0])  ↷ MTE1   L1→L0 写 stage 0                     :143/:145
 8. notify(MTE1→M, 0)           ↷ MTE1   完成后宣布 L0[0] 就绪                :146
 9. wait(MTE1→M, 0)             ↷ M      等 L0[0] 就绪                        :148
10. mad(l0c, l0a[0], l0b[0])    ↷ M       mmad 消费 L0[0]                     :149-151
11. notify(M→MTE1, 0)           ↷ M       完成后宣布 L0[0] 空闲                :152
12. wait(M→MTE1, s_ab=1)        ↷ MTE1   等 L0[1] 空闲（预置已满足）          :141（第二圈）
13. copy_l1_to_l0a/l0b(→L0[1])  ↷ MTE1                                        :143/:145
14. notify(MTE1→M, 1)           ↷ MTE1                                        :146
15. wait(MTE1→M, 1)             ↷ M                                           :148
16. mad(l0c, l0a[1], l0b[1])    ↷ M                                           :149-151
17. notify(M→MTE1, 1)           ↷ M       宣布 L0[1] 空闲                     :152
18. notify(MTE1→MTE2, 0)        ↷ MTE1   两次读取都完成后宣布 L1[0] 空闲      :167
    （随后 k_blk=1 的下发从第 1 步开始；此时 2～18 中多数指令尚未执行完）
```

最后一行注释是**流水线的本质**：标量程序只管按序下发，指令在三条队列里各自排队、被事件闸门自动调节；下发到第 k_blk=1 的第 2 步时，第 k_blk=0 的 mmad 可能还没开始执行。队列里的 wait 闸门保证了"谁也不会提前消费未就绪的数据、谁也不会提前覆写未释放的缓冲"。

还要注意两个结构细节：其一，`:167` 的"L1[s] 空闲" notify 排在**整个 m/n 子块循环之后**（`:123-166`）——L1 stage 是被该 K 块的**所有** M/N 子块依次读取的，全部读完才能归还；其二，`:141-152` 的 L0 事件对在**每条** mmad 上一对，`s_ab` 转得比 `s` 快（第 22 天要点 3）。`:142-145` 在复用 L0 数据时可能跳过拷贝，但 wait/notify 对依然执行，闸门模型不变。

### 3.5 stage 环下标 = 事件号：复用与上限

第 22 天的两个环下标在今天获得第二重身份：**它们就是事件号**。

- `s = k_blk_idx mod kNumL1Stages`：第 `i` 圈 K 循环使用方向 (MTE1→MTE2) 和 (MTE2→MTE1) 的第 `s` 号事件。事件号只有 `kNumL1Stages` 个，环转一圈事件号复用一轮——**N 份缓冲恰好需要 N 个事件号，支撑无限长的 K 循环**。
- `s_ab = (k_mad_idx / MAD_K) mod kNumL0Stages`：同理，用 (M→MTE1) 和 (MTE1→M) 方向的第 `s_ab` 号事件。

事件号是稀缺资源：`common.hpp:61` 定义 `kNumPipeEvents = 8`，`heuristics.hpp:66` 把它写进硬件描述（`.num_pipe_events=kNumPipeEvents, .num_intra_block_events=16`）。于是有两个直接推论：

**推论一：`kNumL1Stages ≤ 8`，且与其他事件族共享这 8 个号。** `heuristics.hpp:166-174`：

```cpp
// Share L1 capacity and pipe events between operands and scale factors.
constexpr int kNumMaxL1StagesWithSF = 4;
const int num_max_l1_stages = dtype.has_sf
    ? std::min(kNumMaxL1StagesWithSF, hw.num_pipe_events - c.num_l1_sf_stages)
    : hw.num_pipe_events;                          // :168  BF16：上限就是 8
const int64_t ab_bytes = (block_m + block_n) * block_k * dtype.elem_bits / 8;  // :171
const int num_l1_stages = std::min<int64_t>(num_max_l1_stages,
                                            (L1SizeBytes - sf_bytes) / ab_bytes);  // :172
```

BF16（无 scale factor）的 `num_l1_stages = min(8, L1 容量 ÷ 每个 stage 的 A/B 字节数)`。以 block 128×128×128 为例：`ab_bytes = (128+128)×128×2B = 64KB`，`min(8, 512KB/64KB) = 8`——与第 22 天演算用的 `kNumL1Stages=8` 一致；若 `BLOCK_K=256`，`ab_bytes=128KB`，容量先卡住：`min(8, 4) = 4`。带 scale factor 的 FP8/FP4 路径还要从 8 个号里切一块给 SF stage（`:160` 的 `num_l1_sf_stages=2`，kernel 侧用**事件基址**分区：`fp8_gemm.hpp:108-109` 定义 `kL1EventBase = 0`、`kSFEventBase = kNumL1Stages`，`:112-113` 两段 `set_flags` 带不同 offset 预置）。

**推论二：事件号 = 资源编号，缓冲不分 stage 时就用固定号。** 对照 `mqa_logits.hpp`：它的 L0A/L0B 没有多级缓冲，于是不用环下标，而是固定的 `EVENT_ID1`（L0A）/`EVENT_ID0`（L0B）——`set_flags<PIPE_M, PIPE_MTE1, 2>()`（`:274`）预置两个空闲；`wait(M→MTE1, EVENT_ID1)`（`:316`，MTE1 等 L0A 空闲）→ `copy_l1_to_l0a`（`:322`）→ `notify(MTE1→M, EVENT_ID1)`（`:325`，就绪）→ `wait(MTE1→M, EVENT_ID1)`（`:342`，M 等）→ mads → `notify(M→MTE1, EVENT_ID1)`（`:411`，归还）。**四组事件的骨架与 bf16 完全同构，只是"有几个缓冲配几个事件"**。

### 3.6 参考时序图（当天产出的参考答案）

**设定**：Normal GEMM，单输出块、单 M/N 子块（`BLOCK_M=MAD_M`、`BLOCK_N=MAD_N`），K=512、BLOCK_K=128、MAD_K=64 → `num_k_blocks=4`、每块 2 条 mmad。`kNumL1Stages=2`、`kNumL0Stages=2`。为画图假设：一次 GM→L1（T2）4 拍、一次 L1→L0（T1）2 拍、一条 mmad（M）2 拍。`T2_i` = 第 i 块的 GM→L1；`T1_i,j` = 第 i 块第 j 步 L1→L0（写 L0[j mod 2]）；`M_i,j` = 第 i 块第 j 条 mmad（读 L0[j mod 2]）。

**每条指令的触发条件**（依赖规则）：

| 指令 | 需要的事件 | 事件何时发出 |
|---|---|---|
| `T2_i` | L1[i mod 2] 空闲 | MTE1 读完该 stage 的两次 `T1` 之后（`:167`）；i<2 由 `:83` 预置 |
| `T1_i,j` | L1[i mod 2] 就绪（`:121`）+ L0[j mod 2] 空闲（`:141`） | 就绪：`T2_i` 完成后（`:119`）；空闲：`M` 消费后（`:152`），j 首次由 `:84` 预置 |
| `M_i,j` | L0[j mod 2] 就绪（`:148`） | `T1_i,j` 完成后（`:146`） |

**拍级甘特图**（每格 2 拍；"—" 表示队列停在该格起点的事件上）：

| 拍区间 | 0–2 | 2–4 | 4–6 | 6–8 | 8–10 | 10–12 | 12–14 | 14–16 | 16–18 |
|---|---|---|---|---|---|---|---|---|---|
| MTE2（GM→L1） | T2_0 | （续） | T2_1 | （续） | T2_2 | （续） | T2_3 | （续） | … |
| MTE1（L1→L0） | — 等 L1[0] 就绪 | — | T1_0,0 | T1_0,1 | T1_1,0 | T1_1,1 | T1_2,0 | T1_2,1 | … |
| M（mmad） | — 等 L0[0] 就绪 | — | — | M_0,0 | M_0,1 | M_1,0 | M_1,1 | M_2,0 | … |

逐个核对边界时刻（这是自测的核心练习）：

- **t=4**：T2_0 完成 → `:119` 发"L1[0] 就绪" → MTE1 放行 T1_0,0；同时 MTE2 无须等待（L1[1] 由预置满足）连续下发 T2_1。
- **t=6**：T1_0,0 完成 → `:146` 发"L0[0] 就绪" → M 放行 M_0,0。
- **t=8**：T1_0,1 完成（MTE1 读完 L1[0]）→ `:167` 发"L1[0] 空闲" → MTE2 放行 T2_2（写 L1[0] 第二轮）；M_0,0 完成 → `:152` 发"L0[0] 空闲" → MTE1 放行 T1_1,0（写 L0[0]）；T2_1 完成 → L1[1] 就绪 → MTE1 放行；M 放行 M_0,1（读 L0[1]，t=8 时 T1_0,1 已完成）。
- **t≥8 进入稳态**：每 4 拍完成一个 K 块——MTE2 一块搬运（4 拍）= MTE1 两步搬运（2+2 拍）= M 两条 mmad（2+2 拍），**三条队列全部满载**。

**稳态的流水线偏移示意**（块粒度）：

```text
K 块:      0        1        2        3        4
MTE2:     [T2_0]   [T2_1]   [T2_2]   [T2_3]   [T2_4] ...   GM→L1 连续不断
MTE1:              [T1_0]   [T1_1]   [T1_2]   [T1_3] ...   晚一个块（等 L1 就绪）
M:                           [M_0]    [M_1]    [M_2]  ...   再晚 2 拍（等 L0 就绪）
           |←— 预热（t<8）—→|←———— 稳态：t≥8 全满载 ————→|
```

**`kNumL1Stages=1` 的对照**（其余设定不变）：L1[0] 空闲要到 MTE1 读完本块两步（t=8）才发出，于是 T2_1 只能 [8,12)、T2_2 只能 [16,20)——MTE2 每块空转 4 拍；MTE1 与 M 之间也出现空泡。**每块周期从 4 拍恶化到 8 拍，三条队列没有一条能满载**。同步的配对一点没变，变的只是缓冲份数——这就是"同步过少（缓冲太少）降低并行度"的最小例子。

**`kNumL1Stages=8` 呢？** 在本例的理想时序下稳态节拍不变（还是 4 拍一块）——前 8 个 K 块的 GM→L1 完全无需等待。多出来的份数买的是**领先量**：当 T2 的实际耗时抖动（L2 miss、GM 带宽波动）时，深队列能吸收抖动而不让 M 队列断粮；代价是预热/排空段更长、对短 K 的 K 循环收益递减。第 52 天做单变量实验时可以验证这一点。

### 3.7 同步的权衡：过少竞争，过多降并行

**同步过少/放错位置的四个反例**（都针对 `bf16_gemm.hpp`）：

| 反例 | 后果 | 类型 |
|---|---|---|
| 删掉 `:116` 的 wait（MTE1→MTE2, s） | 第 `kNumL1Stages+1` 块的 GM→L1 在 MTE1 还在读 L1[s] 时覆写它，mmad 读到新旧混合数据 | **数据竞争** |
| 删掉 `:121` 的 wait（MTE2→MTE1, s） | MTE1 在 GM→L1 完成前就去读 L1[s]，读到旧值/半写数据 | **数据竞争** |
| 删掉 `:148` 的 wait（MTE1→M, s_ab） | mmad 在 L1→L0 完成前启动，读到 L0A/L0B 中**上一轮同 stage 的残留**或半更新数据 | **数据竞争** |
| 删掉 `:83-84` 的 set_flags 预置 | 第一圈的 `:116`/`:141` 永远等不到事件 | **死锁** |

注意第三行的具体场景：`kNumL0Stages=2` 时，L0[0] 在上一轮装的是**上一个输出块**（或本块上上个 K 块）的操作数，`s_ab` 环正是靠"写前等空闲"保证旧数据已被 mmad 消费——删掉 wait 后 M 队列在序执行，mad 紧跟着上一条 notify 排入，完全可能在 MTE1 写完之前执行。

**同步过多/过粗的三种形式**：

1. **多余的 barrier**。假设在每个 k_blk 末尾插入"等三条队列全部排空再进入下一块"的全量同步（如无脑地对所有 stage 连发 `wait_flags`）：仍然正确（最保守），但 MTE2 的跨块预取被废掉，退化为 `kNumL1Stages=1` 的串行形态。正确性需要的同步数量由**缓冲份数**精确决定：每份缓冲两个事件（空闲+就绪），一个不多、一个不少。
2. **缓冲份数（= 事件对数）不足**。`kNumL1Stages=1` 已见；`heuristics.hpp:111-118` 的注释写明了仓库对"至少 2"的态度——`num_l0_stages` 恒为 2（"always 2 stage"），MAD 搜索要求"L0C 至少 2 个 stage 以重叠 mad 与 store、L0A/L0B 装得下以重叠 load 与 mad"。份数受两重上限（3.5 节）：容量断言（`bf16_gemm.hpp:66-69`）与事件号数量（8）。
3. **等待颗粒度错位**。把 `:167` 的"L1[s] 空闲"提前到 m/n 循环之前发出，等于宣布"还没读的缓冲已经空闲"——这不是保守而是错误（伪空闲→竞争）。同理把 `:121` 挪到 `:119` 之前是对同号事件"先等后发"，直接死锁。**wait 挂在消费指令前、notify 挂在生产/消费完成后**，位置本身就是语义。

一句话总结：**同步的数量由缓冲份数唯一确定，同步的位置由"谁生产、谁消费"唯一确定；性能的调节旋钮是缓冲份数（受容量与事件数限制），而不是同步本身**。

### 3.8 块内同步一瞥（第 25 天的引子）

`bf16_gemm.hpp:27` 的 `__mix__(1, kDirectStore ? 0 : 2)` 说明一个 block 内住着 1 个 AIC 和 2 个 AIV（`kDirectStore` 时无 AIV）。AIC 内部的队列间同步用 3.3 的机制；**AIC 与 AIV 之间**用 `sync.hpp:29-40` 的块内同步（`asc_sync_intra_arrive` / `asc_sync_intra_wait`），参数是 `(pipe, subblock_id, flag_id)`，flag 编码 `((subblock_id & 1) << 4) | (flag_id & 0xf)`，每侧 16 个 flag 号（`heuristics.hpp:66` 的 `num_intra_block_events=16`）。

在 BF16 GEMM 里它服务于 Epilogue 的 UB stage 交接（`epilogue.hpp`）：AIC 的 `aic_store` 把 L0C 搬进 UB（`copy_l0c_to_ub`）后 `set_intra_block(PIPE_FIX, aiv_id, stage_idx)`（`:45-46`，AIC 的 FIX 管线宣布"第 stage_idx 号 UB 已填好"）；AIV 的 `aiv_drain` 用 `wait_intra_block(kPipe, aiv_id, stage_idx)`（`:69`）等它，处理完并 `copy_ub_to_gm` 后 `set_intra_block(PIPE_MTE3, aiv_id, stage_idx)`（`:84`，宣布"已排空"）；AIC 侧 `aic_wait_free`（`:37-39`）等 AIV 归还；AIV 启动时 `aiv_initialize` 用 `set_intra_blocks` 预置全部 stage 空闲（`:89-97`）——"空闲/就绪 + 预置"的四组骨架与今天完全同构，只是通道从队列间事件换成了块内 arrive/wait。`kPipe` 取 `PIPE_MTE3`（Identity 算子，`operators.hpp:100`）或 `PIPE_V`（带 scale/bias 的算子，`:129, 165`）。细节（s_epi 环如何跨 AIV 分摊、`2 - kNumAIVs + s_epi % kNumAIVs` 的路由）留给第 25 天。

## 四、源码导读

建议按站点顺序阅读，总计约 45 分钟。所有行号以仓库当前版本为准。

### 站点 1：同步原语总库（`deep_gemm/include/deep_gemm/ascend/sync.hpp`，全文 87 行，精读）

带着 3.3 的契约逐段读：`:7-13` 新式 `PipeEvent` 的方向类型设计；`:15-27` notify/wait 的 stage 展开；`:29-40` 块内 arrive/wait 与 flag 编码、AIV 忽略 subblock_id 的注释；`:44-61` `IntraBlockEvent` 与其 notify/wait；`:63-73` `set_flags`/`wait_flags`（注意 set_flags 就是连发 notify）；`:75-85` 批量块内版本。读完应当能回答：为什么 `set_flags` 需要模板参数 `offset`（提示：fp8 的事件分区）。

### 站点 2：BF16 Kernel 的同步点（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:81-84, 112-168`，精读）

对照 3.4 的表与下发顺序清单逐行核对：`:83-84` 预置；`:116/119` L1 空闲/就绪；`:121` L1 就绪等待；`:141/146` L0 空闲等待/就绪通知；`:148/152` L0 就绪等待/空闲通知；`:167` L1 空闲通知（注意它在 m/n 循环外、k_blk 循环内的位置）。顺带确认 `:129-131` 的 `epilogue.aic_wait_free` 是块内同步的接入点（3.8）。

### 站点 3：事件数与 stage 上限（`deep_gemm/include/deep_gemm/common.hpp:61` + `csrc/jit_kernels/heuristics.hpp:55-70, 111-118, 166-174`，精读）

`kNumPipeEvents=8` 的定义；硬件描述表里 `num_pipe_events`/`num_intra_block_events=16`；`num_l0_stages` 恒为 2 及其注释；`num_l1_stages` 的 min(事件数, 容量) 公式（`:171-172`）。用 128×128×128 与 128×128×256 两组 block 尺寸各算一遍 `ab_bytes` 和结果 stage 数。

### 站点 4：FP8 的事件基址分区（`deep_gemm/include/deep_gemm/fp8_gemm.hpp:108-113, 164-254`，浏览）

`kL1EventBase=0`、`kSFEventBase=kNumL1Stages`（`:108-109`）；`:112-113` 两段带 offset 的 `set_flags` 预置；K 循环内 `kL1EventBase + l1_event_idx` / `kSFEventBase + l1_sf_event_idx` 的用法——同一对方向 (MTE1↔MTE2) 上两个缓冲族如何共享 8 个事件号。

### 站点 5：无多级缓冲的对照（`deep_gemm/include/deep_gemm/mqa_logits.hpp:272-276, 291-325, 342-412`，浏览）

L0A/L0B 用固定 `EVENT_ID1`/`EVENT_ID0` 的四组配对（3.5 推论二）；Q/KV 的 L1 stage 环（`kQEventBase`/`kKVEventBase`，`:272-276`）。看它与 bf16 的同构性，体会"事件号=资源编号"。

### 站点 6：AIV 侧的同模式流水线（`deep_gemm/include/deep_gemm/transform_sf.hpp:543-633`，浏览）

这是把今天的心智模型平移到 AIV 的最佳练习：`set_flags<PIPE_V, PIPE_MTE2>`/`set_flags<PIPE_MTE3, PIPE_V>` 预置（`:543-544`）；`:577` MTE2 等空闲；`:580` GM→UB；`:582-583` 就绪；`:584` 等 MTE3 排空 dst UB；`:596-612` V 计算；`:614` 归还 src UB；`:621-622` MTE3 等就绪；`:625` UB→GM；`:628` 归还 dst UB；`:632-633` `wait_flags` 收尾排空。三级流水 MTE2→V→MTE3 与 AIC 的 MTE2→MTE1→M 完全同构。

### 站点 7：块内同步的使用面（`deep_gemm/include/deep_gemm/epilogue.hpp:20, 24, 37-47, 69-90` + `epilogue/operators.hpp:96-100, 129, 165, 176-183`，浏览）

`kPipe` 的取值来源；`aic_wait_free`/`aic_store`/`aiv_drain`/`aiv_initialize` 四个方法里的 intra-block 调用（3.8）；`operators.hpp:176-183` 里 bias 加载用的 V↔MTE2 事件对（notify 后紧跟 wait 的又一例）。

### 站点 8：新式命名事件（`deep_gemm/include/deep_gemm/mega_moe.hpp:62-96, 379-381, 482`，浏览）

`PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull` 等 30 多个命名事件：Full=就绪、Empty=空闲，方向类型即 3.3 要点 1 的文档化。看 `wait_intra_block<PIPE_MTE3, 0>(kL1BEmpty, l1b_stage_idx)`（`:379-381`）如何把"事件对象 + stage 下标"组合出环。

## 五、实践步骤

### A. 无 Ascend 环境：纸面推演（主任务，约 55 分钟）

**A-1 配对表（必做）**。不看 3.4 的表，打开 `bf16_gemm.hpp:81-168`，把所有 `asc_sync_*` / `set_flags` 调用抄下来，按"方向（源→目标）/ 事件号来源 / wait 行号 / notify 行号 / 预置行号 / 含义"六列整理成自己的四组配对表，再对照订正。

**A-2 画时序图（必做，当天产出的核心）**。用 3.6 的设定（K=512、BLOCK_K=128、MAD_K=64、单 M/N 子块、kNumL1Stages=2、kNumL0Stages=2、T2=4 拍/T1=2 拍/mmad=2 拍）**先不看参考答案**：

1. 先列依赖规则表（每条指令等什么事件、事件何时发出）；
2. 再画拍级甘特图（0 到 18 拍），逐拍核对 t=4/6/8 三个边界时刻各发出了哪些事件、放行了哪些指令；
3. 标出预热段（t<8）与稳态段（t≥8），数一数稳态周期。

**A-3 变体重画（必做）**。每次只改一个变量，重画甘特图并记录差异：

| 变体 | 改动 | 观察重点 |
|---|---|---|
| V1 | `kNumL1Stages=1`（其余同 A-2） | 每块周期 4→8 拍；MTE2/MTE1 各出现 4 拍空泡；与 3.6 的对照段核对 |
| V2 | `kNumL1Stages=3` | 前 3 块 T2 无需等待；稳态节拍不变——多出的份数买的是什么？ |
| V3 | `MAD_K=128`（每块 1 条 mmad） | `s_ab` 恒为 0，L0 事件只剩 0 号在反复使用；每块 L1→L0 只有一次 |
| V4 | `kNumL0Stages=1` | MTE1 与 M 完全串行（T1 与 mmad 交替），但 MTE2 与 MTE1 的重叠仍在 |

**A-4 反例推演（必做）**。对 3.7 表格的四个反例，各写两三行"错误现场"：删掉该同步后，哪两条指令在哪个缓冲上竞争/谁在等谁永远等不到。要求给出具体的 `s`/`s_ab` 取值（例如 kNumL1Stages=2 时第 3 块的 T2_2 与 T1_0,*/T1_1,* 争 L1[0]）。

**A-5 判定题自验**。判断真假并给出行号依据（答案见第七节）：

1. `asc_sync_wait(PIPE_MTE1, PIPE_MTE2, s)` 的第一个参数是等待的一方。（ ）
2. `bf16_gemm.hpp:119` 的 notify 与 `:121` 的 wait 方向相同、事件号相同，因此它们构成一对会合。（ ）
3. `kNumL0Stages=2` 时，MTE1 可以在 M 尚未执行任何 mmad 之前把两个 L0 stage 都填好（各发一个就绪事件）。（ ）
4. 把 `:167` 的 notify 移到 m/n 子块循环之前不会出错，因为 `:116` 的 wait 还在。（ ）
5. `kNumL1Stages` 可以取 9，只要 L1 容量装得下。（ ）

### B. 有 Ascend 环境：核对真实配置（可选，约 20 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
```

1. 从输出里找 `GemmConfig(...)` 行（由 `heuristics.hpp:749-753` 打印），记录不同 shape 的 `l1_stages` / `l0_stages`；
2. 核对 `l0_stages` 是否恒为 2、`l1_stages` 是否 ≤ 8；对 block 128×128×128 验证 `l1_stages=8` 是否出现；
3. 找一个 `BLOCK_K=256` 的配置，验证容量约束是否把 `l1_stages` 压到 4（3.5 节公式）；
4. 若无环境，改为静态演算：对 `heuristics.hpp:168-172` 代入 block 128×128×128 与 128×128×256 两组数值，手算 `num_l1_stages`。

## 六、当天产出

**MTE1/MTE2/M 流水线时序图**。把 A-1～A-4 的结果整理成一页笔记，至少包含：

1. **事件配对表**：四组事件的方向、行号、预置位置与守护的资源（3.4 表的自绘版）；
2. **依赖规则 + 拍级甘特图**：覆盖 `kNumL1Stages=2` 与 `kNumL1Stages=1` 两种配置，标注预热段/稳态段与边界时刻（t=4/6/8）的事件收发；
3. **下发顺序清单**：仿照 3.4 的 18 步清单，注明每步进入哪条队列——体现"下发 ≠ 完成"；
4. **变体记录**：V1～V4 各两三行结论（周期、空泡、环下标变化）；
5. **遗留问题**：交给后续天（例如"块内 arrive/wait 的 flag 编码与 AIC↔AIV 路由"交给第 25 天，"`s_epi` 环与 Epilogue stage"交给第 25 天，"`epilogue.aic_wait_free` 为什么只在 `k_blk_idx==0` 时调用"交给第 25 天）。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. 写出四组配对事件的方向、wait/notify 行号和含义；哪两组需要启动预置？"就绪"事件为什么绝不能预置？
2. 论证 `bf16_gemm.hpp:119` 的 notify 为什么必须"在 GM→L1 搬运完成后才触发"：如果它在下发瞬间就触发，最坏会发生什么？
3. 删掉 `:83-84` 的 `set_flags` 预置，第一圈会发生什么？为什么后果是死锁而不是竞争？
4. `kNumL1Stages` 受哪两重上限约束？分别对应源码哪几行？用 `heuristics.hpp:171-172` 的公式算出 block 128×128×128（BF16）的 stage 数。
5. 删掉 `:148` 的 `wait(MTE1→M, s_ab)` 会产生什么竞争？描述一个包含具体 `s_ab` 取值的错误现场。
6. `mqa_logits.hpp` 的 L0A/L0B 为什么用固定的 `EVENT_ID0/1` 而不是环下标？这与"事件号 = 资源编号"是什么关系？
7. 把 `kNumL1Stages` 从 2 提到 8，稳态吞吐会变成 4 倍吗？stage 数的本质作用是什么？对什么样的 shape 收益最小？

### 参考答案

1. 组1：L1[s] 空闲，MTE1→MTE2，wait `:116` / notify `:167`；组2：L1[s] 就绪，MTE2→MTE1，notify `:119` / wait `:121`；组3：L0[s_ab] 空闲，M→MTE1，wait `:141` / notify `:152`；组4：L0[s_ab] 就绪，MTE1→M，notify `:146` / wait `:148`。组1（`:83`）与组3（`:84`）需要预置——第一圈之前不存在"上一轮"，无真实来源可发空闲事件；就绪必须由真实搬运产生，预置就绪等于宣布空缓冲有数据，MTE1/M 会读到垃圾。
2. 若 notify 立即触发，`:121` 的 wait 立即放行，MTE1 在 GM→L1 完成前就读 L1[s]，读到旧值或半写数据——写-读竞争。所以 notify 必须作为指令排入源（MTE2）队列、在两条搬运之后按序执行；事件的真实含义是"源队列中位于 notify 之前的所有指令已完成"（3.3 要点 2）。
3. 第一个 `:116` 的 wait 等待"L1[0] 空闲"，该事件唯一的循环内来源是 `:167`——而 `:167` 位于第一圈循环体的末尾、且排在它前面的 MTE1 指令又依赖 `:121` 放行……生产该事件的指令本身排在等待点之后，永远执行不到 → 死锁。竞争需要"至少执行了一次错误的访问"，而死锁是压根不动。
4. 事件号数量：每方向 `kNumPipeEvents=8`（`common.hpp:61`，`heuristics.hpp:66, 168`）；L1 容量：`static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes)`（`bf16_gemm.hpp:66`），公式见 `heuristics.hpp:171-172`。block 128×128×128：`ab_bytes=(128+128)×128×2B=64KB`，`min(8, 512KB/64KB)=8`。
5. M 队列在序执行，mad 会紧跟着上一条指令排入并可能在 L1→L0 完成前执行。现场：`kNumL0Stages=2`、每块 2 条 mmad 时，第 1 块第 1 条 mmad（`s_ab=0`）读到的是 L0A[0]/L0B[0] 中**上一个输出块**（或上一轮循环）残留的操作数；或 MTE1 正在写入而 mmad 读到一半新一半旧——两种都属于读-写竞争，结果错误但不一定卡死。
6. 它的 L0A/L0B 没有多级缓冲（每份资源只有一份），所以不需要环下标；有几份缓冲就配几个事件号——`EVENT_ID1` 对 L0A、`EVENT_ID0` 对 L0B（`mqa_logits.hpp:274, 316, 325, 342, 373, 411`）。事件号标识的是"哪份资源"，环下标只是"资源有序复用"时的编号方式。
7. 不会。稳态节拍由最慢的队列决定（3.6 中是每 4 拍一块），份数超过稳态所需的领先量后，买的是吸收抖动的余量（GM 带宽波动、L2 miss 时 M 队列不断粮）与更长的预取深度；代价是更长的预热/排空段和事件号预算。对 K 很小（`num_k_blocks` 与 stage 数同量级）的 shape 收益最小——循环还没进入稳态就结束了。

**A-5 判定题答案**：

1. **假**。第一个参数是事件的**发出方**（源 pipe），第二个参数才是接收/等待方（3.3 要点 1；`mega_moe.hpp:62-63` 的 Full/Empty 命名可证）。
2. **真**。同方向同事件号的一次 notify 与一次 wait 构成会合（`:119` 发、`:121` 收）；它们相邻只是下发顺序相邻，实际分属 MTE2 与 MTE1 两条队列（3.3 要点 3）。
3. **真**。`:84` 预置了 0、1 两个空闲事件，MTE1 依次写 L0[0]、L0[1] 并各发一个就绪（`:146`），期间 M 一条 mmad 都没执行也不会阻塞 MTE1。
4. **假**。那样会在 L1[s] 还没被 m/n 子块读取时就宣布它空闲，`:116` 放行 MTE2 覆写正在被读的缓冲——伪空闲导致竞争（3.7 形式 3）。
5. **假**。BF16 无 SF 时 `num_max_l1_stages = num_pipe_events = 8`（`heuristics.hpp:168`），事件号数量在容量之前就卡住；带 SF 的路径还要扣掉 SF 的事件号。

---

**明日预告（第 24 天）**：同步的脚手架搭好了，明天聚焦它守护的计算本身——`mad()` 调用前后的流水如何衔接 L0A/L0B 到 L0C 的乘加、`ENABLE_UPDATE` / `ENABLE_KEEP` 与首次累加（`is_first`/`is_last` 在第 22 天埋下的伏笔）如何决定 L0C 累加链的清零与提交，以及多个 MAD Tile 如何拼出一个 Block 的完整输出。
