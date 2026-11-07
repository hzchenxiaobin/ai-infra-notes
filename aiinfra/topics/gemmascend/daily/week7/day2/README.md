# 第 7 周 · 第 2 天（总第 44 天）：理解 MegaMoE 计算主线

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 7 周"MegaMoE、HC Prenorm 与扩展算子"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 30 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

昨天（第 43 天）我们看清了 MegaMoE 的"外壳"：三个 Python 符号、SymmBuffer 对称缓冲区、13 个视图和权重转换。今天钻进内核本身——`deep_gemm/include/deep_gemm/mega_moe.hpp` 的 `mega_moe_impl` 把 `Dispatch → Grouped GEMM 1 → SwiGLU → Grouped GEMM 2 → Combine` 五段全部装进**一个 Kernel**，用 32 个 AI Core Block 上的三类角色（AIC / AIV0 / AIV1）协同完成。今天的三个核心问题：**谁在算、谁在搬；中间张量长什么形状、放在哪个区域；两个 Grouped GEMM 与 SwiGLU 靠什么机制衔接**。跨 rank 通信的协议与计算/通信重叠的深层设计留到明天（第 45 天），今天只把"经 SymBuffer 读写远端"当作一种已知的远端内存访问能力。当天产出是**MegaMoE 张量形状变化表**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 `mega_moe_impl`（`deep_gemm/include/deep_gemm/mega_moe.hpp:1125-1219`）以 `__global__ __mix__(1, 2)` 启动后的角色分工：每 Block 1 个 AIC + 2 个 AIV，AIC 跑 `aic_gemm_pipeline`（两个 Grouped GEMM 的全部 MAD），AIV0 跑元数据构建与 FP4→FP8 NZ 权重流水，AIV1 跑 Dispatch 拉取、SwiGLU epilogue 与 Combine push，64 个 AIV 共同执行 `combine_reduce`；
2. 解释 `BlockPhase`（`scheduler/mega_moe.hpp:9`）五种任务与 `take_linear_task`（`:58-118`）的调度顺序：shared Linear1 → shared Linear2 → routed（Linear1 带一波 lookahead 的优先权），并说明 **AIC 与 AIV1 用相同参数重放同一任务序列**从而在每个任务上"会合"；
3. 用 `ExpertCursor`（`scheduler/mega_moe.hpp:19-54`）从 `expert_recv_count_sum` 出发推演出每个 M-block 的 `valid_m`，并解释 FRAC_MN=16 分组平衡为什么比简单 ceil 切分更优；
4. 追踪 Dispatch 主线：`build_dispatch_metadata`（`comm/mega_moe.hpp:117-240`）如何统计专家计数、把 `(weight_bits<<32 | token_topk_idx)` 发给专家持有者，AIV1 的 `advance_dispatch_pull`（`mega_moe.hpp:763-873`）如何按路由把远端 token 拉进 `l1_acts` 接收池；
5. 讲清两个衔接机制的本质差异：**GEMM1 → SwiGLU 走 Block 内 UB 交接**（`ub_accum` + `IntraBlockEvent<1>` 的 FIX→V 信用，`mega_moe.hpp:99-101, 612-636, 900-903`），而 **SwiGLU → GEMM2 走 GM 进度计数**（`kLinear1Ready` 原子累加，`:536-540, 980-984`），并能说出各自的等待者与期望计数；
6. 描述 `vf_swiglu_quant`（`simd/mega_moe.hpp:57-166`）的计算：clamp、`gate × up / (1 + exp(-gate))`、路由权重乘入、逐 32 通道 amax 取 2 的幂的 UE8M0 SF、FP8 量化与 K-major SF 存储；
7. 解释环形接收池（`layout/mega_moe.hpp:221-226, 257-261`）与四个 `ProgressStage`（`:27-33`）如何让 l1_acts / l2_acts / routed.output 三块缓冲被五段循环复用而不冲突；
8. 独立完成 MegaMoE 张量形状变化表（当天产出），对给定配置写出每段输入输出的形状、dtype、所在区域与生产者/消费者。

## 二、预备知识

- **第 15、22、24 天**：BF16 Kernel 的模板参数、K 维两层循环（`BLOCK_K` 外层 + `MAD_K` 内层）、`mad()` 的 `ENABLE_UPDATE/ENABLE_KEEP` 与首次累加——MegaMoE 的 AIC 主循环完全沿用这套结构。
- **第 25 天**：AIC/AIV 分工与 Epilogue——`__mix__(1, 2)` 混合编程模型、AIC 的 FIX 写出与 AIV drain 的配合。今天 `IntraBlockEvent`（`deep_gemm/include/deep_gemm/ascend/sync.hpp:45-53`，"kSubblock selects the AIC/AIV0 or AIC/AIV1 connection"）是同一机制的 typed 封装。
- **第 26 天**：`kDirectStore` 等特殊路径——今天 Linear2 的 `copy_l0c_to_gm`（L0C 直接写 GM）与 DirectStore 思路一致。
- **第 32 天**：FP8/FP4 Kernel 路径——MegaMoE 是 fp8 激活 × fp4 路由权重 + fp8 shared 权重。
- **第 33 天**：Scale 布局——`MX_SF_DIVISOR = 64`（`deep_gemm/include/deep_gemm/common.hpp:65`，每 64 个 K 元素一对 UE8M0 指数打包进 int16）；今天还要区分 token-major 与 M-block 内 K-major 两种 SF 布局（`layout/mega_moe.hpp:194`）。
- **第 36、37 天**：M-Grouped GEMM——"token 按 expert 排序拼接后分段做 GEMM"正是今天两个 Grouped GEMM 的形态，只是分段信息（`expert_recv_count_sum`）由 Kernel 内的 Dispatch 元数据现场生成。
- **第 43 天**：MegaMoE 对外接口——13 个视图、权重转换（gate/up 交织、FP4 SF 指数 +6）、`BLOCK_M/BLOCK_N = 256`（`layout/mega_moe.hpp:12`）。本文沿用第 43 天的贯穿配置 `R=2, E=8, S=0, T=128, k=2, H=I=512`。

硬件常量（`deep_gemm/include/deep_gemm/common.hpp:51-65`，本教程直接引用）：`kNumAICores = 32`、`kNumAIVs = 64`；L1 每核 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB（FP32 累加器）、UB 256 KiB；`FRAC_MN = 16`（NZ 分形行数）；`MX_SF_DIVISOR = 64`。

## 三、详细讲解

### 3.1 一个 Kernel、三类角色：`mega_moe_impl` 总览

`mega_moe_impl`（`mega_moe.hpp:1125-1219`）是 7 个模板参数（R、E、S、T_max、k、H、I）的 `__global__ __mix__(1, 2)` Kernel，由 `launch_mega_moe`（`csrc/jit_kernels/mega_moe.hpp:24-51`）以 `num_blocks = kNumAICores`（32）启动。`__mix__(1, 2)` 表示**每个 Block 由 1 个 AIC（cube）和 2 个 AIV（vector）组成**——这与 BF16 Kernel 的 `__mix__(1, kDirectStore ? 0 : 2)`（`bf16_gemm.hpp:27`）是同一混合编程模型。于是整个 Kernel 的角色格局是：

```text
Block 0: [ AIC_0 | AIV_0(subblock 0) | AIV_1(subblock 1) ]
Block 1: [ AIC_1 | AIV_0            | AIV_1            ]
  ...          ...        ...              ...
Block 31:[ AIC_31| AIV_0            | AIV_1            ]

aiv_idx = 2 * core_idx + subblock_idx   （mega_moe.hpp:1168-1169，全局 64 个 AIV）
```

入口处每个角色选择自己的主战场（`mega_moe.hpp:1160-1218`）：

| 角色 | 数量 | 入口 | 职责（对应五段） |
|---|---|---|---|
| AIC | 32 | `aic_gemm_pipeline`（`:1161-1162`，函数体 `:402-669`） | **Grouped GEMM 1 + Grouped GEMM 2 的全部 MAD 计算与结果写出**（shared 与 routed 都算）；结束后 `asc_sync_block_wait(PIPE_S, kCombineDoneID)` 等 AIV 收尾（`:1163`） |
| AIV0 | 32 | 元数据 + `aiv0_weight_pipeline`（`:1188-1209`） | 先以 512 线程 SIMT VF 调 `build_dispatch_metadata`（`:1196-1198`）构建 **Dispatch 元数据**，再 `wait_dispatch_metadata`（`:1203`）汇总 M-block 总数，随后把 **FP4 路由权重解包成 FP8 NZ 喂进 AIC 的 L1B**（`:1209`） |
| AIV1 | 32 | `aiv1_epilogue_comm`（`:1210-1212`，函数体 `:671-1123`） | **Dispatch 拉取**（把远端 token 写进 `l1_acts`）、**SwiGLU 量化 epilogue**（读 AIC 写好的 UB，写 `l2_acts`）、**Combine push**（把 `routed.output` 行发给源 rank） |
| AIV0+AIV1 | 64 | `combine_reduce`（`:1215`，函数体 `:251-295`） | **Combine reduce**：把 `combine_input` 中每个 token 的 k(+1) 份输入 FP32 求和、一次舍入成 BF16 写进 `y` |

两个入口级细节值得先记下：

1. **`kOverlapReduce` 是编译期开关**（`:1138-1140`）：`T_max × (k + (S>0)) × H × sizeof(bf16) > 32 MiB` 时为真——combine 的数据量足够大，值得让 reduce 与前面的计算重叠执行（AIV1 在主循环里穿插推进 reduce，AIV0 在权重流水空隙推进 reduce）；否则走"全部算完再一次性归约"的简单路径。注释写明这是"Measured 32 MiB combine-work threshold for tail reduction"（实测阈值）。reduce 的 token 划分也随开关变化（`:1171-1176`）：重叠模式按 `aiv_idx` 跨步轮询（round-robin），非重叠模式按连续区间切分。
2. **元数据就绪有两种通知通道**（`:1205-1207` 注释 "Notify AIV1 via GM and AIC via event"）：AIV0 完成元数据汇总后，对 AIC 用 Block 内事件 `notify_intra_block<PIPE_S, 0>(kMetadataFull)`，对 AIV1 写 GM 标志 `progress.metadata_ready`。AIC 在 `aic_gemm_pipeline` 里收到事件后读取 `num_m_blocks_certificate`（`:441-454`）；AIV1 在主循环里轮询 GM 字节（`:1096-1102`）。同一件事、两种通道——因为 AIC/AIV1 分属不同子块，与 AIV0 的连接不同。

在进入逐段拆解之前，先给出全流程鸟瞰图（细环节后文展开）：

```text
                    ┌────────────── 本 rank 对称缓冲区（layout::Workspace）──────────────┐
 用户/远端 rank ──►  x [T,H] fp8 ──Dispatch pull(AIV1)──► l1_acts [ring,H] fp8 ──┐
 topk_idx [T,k] ─►  元数据(AIV0 SIMT) ─► expert_recv_count_sum [E/R]              │ GEMM1(AIC)
                    l1' 权重 FP4 ──AIV0: FP4→FP8 NZ──► AIC 的 L1B                │ A=激活,B=权重
                                                                                    ▼
                    ub_accum [128,256] bf16 ×2 stages（AIC FIX 写，AIV1 读）◄── L0C FP32
                                                                                    │ SwiGLU+量化(AIV1)
                    l2_acts [ring,I] fp8 + SF(K-major) ◄─────────────────────────┘
                                                                                    │ GEMM2(AIC)
 l2' 权重 FP4 ────AIV0: FP4→FP8 NZ──► AIC 的 L1B；A = l2_acts                     ▼
                    routed.output [ring,H] bf16（L0C 直接写 GM）
                                                                                    │ Combine push(AIV1)
 远端 rank ◄─────── combine_input [(k+S>0)·T, H] bf16（按 slot 存放）◄────────────┘
                                                                                    │ reduce(64 AIV)
 用户 ◄──────────── y [T,H] bf16（FP32 累加一次舍入）◄─────────────────────────────┘
```

### 3.2 调度语言：`BlockPhase`、`take_linear_task` 与 `ExpertCursor`

五段流水里有三个"计算生产者"：AIC 的 GEMM、AIV1 的 SwiGLU、AIV1 的 Combine push。它们必须对"接下来处理哪个 M-block 的哪一段 N"达成一致——这就是调度器 `scheduler/mega_moe.hpp` 的职责。

**任务用 `BlockPhase` 枚举**（`:9`）：`kNone / kLinear1 / kLinear2 / kSharedLinear1 / kSharedLinear2`。一个任务 = （阶段，`task_idx`），而 `task_idx` 在消费侧统一解码为：

```cpp
const uint32_t pool_block_idx = task_idx / num_n_blocks;   // 池内第几个 M-block
const uint32_t n_block_idx    = task_idx % num_n_blocks;   // 该块的第几个 N 块
```

（`mega_moe.hpp:466-467`；AIV0/AIV1 侧同理解码，`:339-342, :897-898`。）其中 `num_n_blocks` 按阶段取值（`:461-463`）：

| 阶段 | N 块大小 | `num_n_blocks` | K 维（`num_k`） | 说明 |
|---|---|---|---|---|
| Linear1（routed） | BLOCK_N=256 行权重 → 256 输出列 | `I / 128` | H | 每 N 块 = 128 行 gate + 128 行 up（第 43 天的交织） |
| Linear2（routed） | 256 列 | `H / 256` | I | 输出回到 hidden 维 |
| SharedLinear1 | 256 | `S·I / 128` | H | N 维被 shared expert 拼宽 |
| SharedLinear2 | 256 | `H / 256` | S·I | K 维被 shared expert 拼宽 |

注意 Linear1 的"输出列"是 2I 宽中间结果的 256 列片段，SwiGLU 后收缩为 128 列写入 `l2_acts`——所以 `I / 128` 个 Linear1 任务恰好铺满 I 列的 `l2_acts`（见 3.5）。

**`take_linear_task`（`scheduler/mega_moe.hpp:58-118`）是纯函数式的"发牌器"**：输入任务基址状态 `LinearTaskState`（四个 task_base 游标，`:11-16`）与 `core_idx`，输出下一个任务并推进游标。三个特征：

1. **shared 先于 routed**（`:68-86`）：先发完 SharedLinear1 波次，再 SharedLinear2，最后 routed。shared 任务不依赖 Dispatch 元数据，可以先跑；同时 AIC 侧注释说明"Shared tasks keep one shape so their L1B stages retain the same physical stride"（`mega_moe.hpp:659`）——shared 的 MAD 形状按 `num_tokens` 统一选择，L1B 物理跨度保持一致。
2. **routed 需要 `metadata_ready`**（`:88-92`）：元数据未就绪时返回 `kNone` 但不表示"耗尽"，由调用方区分（AIC 在 `:441-454` 里等 `kMetadataFull` 后重入）。
3. **Linear1 带一波 lookahead 的优先权**（`:94-103`）：

```cpp
// Prepare the next Linear2 wave with one Linear1 wave of lookahead
while (state.l1_task_base < num_routed_tasks[0] or state.l2_task_base < num_routed_tasks[1]) {
    bool is_linear1 = state.l2_task_base >= num_routed_tasks[1];
    if (not is_linear1 and state.l1_task_base < num_routed_tasks[0]) {
        const uint32_t required_linear1_task_end =
            ceil_div(min(state.l2_task_base + kNumAICores, num_routed_tasks[1]), kNumLinear2NBlocks) *
            kNumLinear1NBlocks;
        is_linear1 = state.l1_task_base < required_linear1_task_end + kNumAICores;
    }
    ...
}
```

含义：默认发 Linear2；但只要"已发出的 Linear1 尚不足以覆盖下一波（32 个）Linear2 任务所需的 M-block，再加一整波余量"，就优先发 Linear1。这保证了 AIC 切到某个 M-block 的 Linear2 时，该块的 SwiGLU（依赖其全部 Linear1 N 块）早已在流水线上。

**关键设计：AIC 与 AIV1 各自维护 `LinearTaskState`，用相同参数重放 `take_linear_task`，于是每个核上看到完全相同的任务序列**（`scheduler/mega_moe.hpp:56-57` 注释 "Replay the same routed task order on AIC and AIVs"）。AIC 在核 c 上对任务 t 执行 GEMM，AIV1 在核 c 上对同一任务 t 执行对应的后处理（Linear1 → SwiGLU；Linear2 → 归还信用并发布进度）。两者并发行进，顺序对齐靠下一节的信用机制。AIV0 只需要 routed 任务的权重信息，它调用 `take_linear_task` 时把 `num_shared_tokens` 传 0 跳过 shared（`mega_moe.hpp:332-333` 注释 "AIC handles shared weights; zero shared tokens skips those tasks"）。

**`ExpertCursor`（`scheduler/mega_moe.hpp:19-54`）把"池内 M-block"翻译成"某个 expert 的某段行"**。它的状态只有三个数：`pool_block_end`（当前 expert 在池内的排他终点）、`local_expert_idx`、`num_expert_tokens`。`advance_to(buffer, pool_block_idx)`（`:46-53`）在 `expert_recv_count_sum`（所有源 rank 的接收计数之和，24 位计数值 + 8 位到达数打包在一个 uint32 里，`layout/mega_moe.hpp:22-24`）上逐 expert 前进，空 expert（0 个 token）自然被跳过。`get_m_block_range`（`:33-43`）计算块内有效行数：

```cpp
// Balance FRAC_MN groups; expert M is compact, pool blocks keep the BLOCK_M stride
const uint32_t num_m_groups = ceil_div(num_expert_tokens, FRAC_MN);
const uint32_t num_groups_per_block = num_m_groups / num_expert_m_blocks;
const uint32_t num_larger_m_blocks = num_m_groups % num_expert_m_blocks;
expert_m_begin = (expert_m_block_idx * num_groups_per_block + min(expert_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
const uint32_t aligned_valid_m = (num_groups_per_block + (expert_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
return min(aligned_valid_m, num_expert_tokens - expert_m_begin);
```

为什么不是"前 ceil 块各 256 行、最后一块装剩余"？因为 MAD 的 NZ 分形要求 M 维按 `FRAC_MN = 16` 对齐（`aligned_valid_m` 一定是 16 的倍数），把 16 行一组的"组"均匀分给各块，能让多块 expert（如 300 个 token 分 2 块）得到 160/140 这样的均衡切分，而不是 256/44 的头重脚轻——后者会让最后一块的 MAD 大量空转。第 5 节的 B 组任务会让你手算这个例子。

### 3.3 第一段 Dispatch：元数据先行，拉取式搬运

Dispatch 要解决的问题：本 rank 只持有 E/R 个 expert，但全局所有 rank 的 token 都可能路由过来。MegaMoE 采用**两步走**：

**第一步（AIV0，SIMT）：构建元数据。** `build_dispatch_metadata`（`comm/mega_moe.hpp:117-240`）由最多 32 个 AIV0 worker 并行执行，每个 worker 是一次 512 线程的 `asc_vf_call`（`mega_moe.hpp:1196-1198`；`kNumDispatchThreads = 512`，`layout/mega_moe.hpp:20`）。流程四拍：

1. **计数**（`:139-157`）：每个 worker 在自己的 UB 里为全部 E 个 expert 计数（`expert_idx < num_experts` 过滤掉 `-1` 无效路由，`:154-156`）；
2. **前缀拼接**（`:160-173`）：跨 worker 做前缀和，得到每个 expert 的全局计数 `total_expert_count` 与各 worker 的写偏移；
3. **暂存路由**（`:175-189`）：每个有效路由 `(token, topk_slot)` 按 expert 分桶写入 `staged_token_topk_idx`；
4. **发给持有者**（`:192-214`）：对每个 expert，把 `(weight_bits << 32) | token_topk_idx`（64 位打包：高 32 位是 fp32 路由权重的位模式，低 32 位是 `token_idx * k + slot_idx`）经 `SymBuffer::map` 写进**持有该 expert 的 rank** 的 `src_token_metadata`；随后（`:217-239`）把每 expert 计数以 `kCountArrivalBit | count` 的打包形式原子加到持有者的 `expert_recv_count_sum`——到达位凑满 R 个即表示所有源 rank 的计数到齐。

`wait_dispatch_metadata`（`:242-275`）在 AIV0 上收尾：等待每个本地 expert 的到达位凑满 R（`:250-259`）、给可选的 `cumulative_local_expert_recv_stats` 累加（`:261-262`，即第 43 天看到的跨 launch 统计）、把各 expert 的 `ceil_div(count, BLOCK_M)` 求和进 `num_m_blocks_certificate`（`:263-274`）。这个证书就是 `num_total_m_blocks`——routed 池的 M-block 总数，AIC 和 AIV1 都靠它知道任务总量。

**第二步（AIV1）：按路由拉取 token。** `advance_dispatch_pull`（`mega_moe.hpp:763-873`）对每个池内 M-block、每行执行：

1. 用 `ExpertCursor` 定位当前 expert 与 `valid_m`（`:787-807`）；对每个源 rank 维护 `[begin, end)` 行区间（读 `expert_recv_count` 前缀，`:787-797`），并用**互质步长置换**源行（`next_coprime`，`:801-802`；设计意图见 `:756` 注释 "Permute the concatenated source rows without changing their counts"）——让连续的拉取行均匀散布在不同源 rank 上，避免远端读扎堆；
2. 解析该行对应的源 token：读 `src_token_metadata` 得 `(token_topk_idx, weight_bits)`（`:831-836`），换算出 `src_token_idx` 与 `src_topk_idx`；
3. 经 `sym_buffer.map(buffer, src_rank_idx)` 把指针平移到源 rank 的对称区（`:828`；`SymBuffer::map` 只是加一个 rank 偏移，`layout/sym_buffer.hpp:36-46`），把 `x` 的一行 FP8 与 SF 拉进 UB（`:841-844`），再写入本 rank 的 `l1_acts` 环形行（`:848-855`，行号由 `get_ring_m_idx` 决定）；
4. 顺手记下三样东西：路由权重以 fp32 位模式存进 `routed.topk_weights[pool_m_idx]`（`:857-859`，SwiGLU 要用）；`src_metadata[pool_m_idx] = (src_rank_idx << 24) | combine_m_idx`（`:863-865`，注释 "Source metadata: rank [31:24], combine M [23:0]"，Combine push 要用）；重叠模式下还记录 `route_m_blocks`（该路由的产出在哪个生产块，供 early reduce 查询，`:861-862`）。

每个 AIV1 核处理 M-block 内行号 ≡ `core_idx (mod 32)` 的行（`pull.m_idx_in_block` 初始为 `core_idx`，每次加 `kNumAICores`，`:760, :812-815`），一块拉完发布一次 `kDispatched` 进度（`:772`）。AIC 在 Linear1 任务的 K=0 时等待该块的 `kDispatched` 凑满 32（`:535-540`）才开始加载激活。

**注意这个 Dispatch 是"拉"不是"推"**：源 rank 不主动发送 token 数据，只发出轻量的路由元数据；真正的数据搬运由 expert 持有者按需远端读。这就是为什么 `l1_acts` 被称为"接收池"——它既是 Dispatch 的目的地，也是 GEMM1 的 A 矩阵。

### 3.4 第二段 Grouped GEMM 1：AIC 计算、AIV0 喂权重

`aic_gemm_pipeline`（`mega_moe.hpp:402-669`）是五段流水的计算心脏，**一个循环同时服务四种任务**（Linear1/Linear2 × shared/routed），区别只在数据源与 epilogue 目的地。以 routed Linear1 为主线索：

**任务解码**（`:456-493`）：`is_shared`/`is_linear1` 两个布尔定下四个量——N 块数、K 维 `num_k`（Linear1 用 H、Linear2 用 I；shared 的 Linear1/2 分别把 N/K 换成 S·I 拼宽值，`:460-464`）、`valid_m`（shared 用 `min(num_tokens - 块起点, BLOCK_M)`，routed 用 `ExpertCursor::advance_to`，`:469-470`）、权重基址（shared 无 expert 维，routed 按 `local_expert_idx * num_n_blocks + n_block_idx` 寻址，`:474-477`）。输入侧同样四选一（`:486-493`）：

| 任务 | A（激活） | A 的 SF |
|---|---|---|
| Linear1 routed | `routed.l1_acts`（接收池，ring 行） | token-major `[行, H/64]` |
| Linear2 routed | `routed.l2_acts`（SwiGLU 产出，ring 行） | M-block 内 K-major |
| SharedLinear1 | `input.acts` = x（原始 token，不经过 ring） | token-major |
| SharedLinear2 | `shared_l2_acts`（shared SwiGLU 产出，按 pad(T,256) 行） | K-major |

**MAD 形状双档**（`MMAConfig`，`:28-48`）。Kernel 源码开头的注释就是配置表：

```text
// BLOCK_M  BLOCK_N  BLOCK_K  MAD_M  MAD_N  MAD_K
//     256      256      512    128    128    256
//     256      256      256    256    256    128
```

`MMAConfig<kMadM>` 从 L0 容量反推：`BLOCK_K = L0ASizeBytes / kMadM`（64 KiB / 128 = 512 或 / 256 = 256），`MAD_K = BLOCK_K / 2`（L0 双缓冲，`kNumL0Stages = 2`），`MAD_N = L0BSizeBytes / 2 / MAD_K`。**每个任务按 `valid_m` 现场选档**（`:659-663`）：`(is_shared ? num_tokens : valid_m) <= 128 ? 128 : 256`——小 M 用细粒度档（M=128、K=512），大 M 用粗粒度档（M=256、K=256），两者共享同一套 L1/L0 分区，只是 stage 物理跨度不同（`:50-59` 的布局注释）：

```text
// AIC layout (KiB); shared and routed B occupy the same L1 region
// Memory  Region       Offset  Stages x size
// L1      A                 0      3 x 64
//         routed B        192      4 x 64
//         shared B        192      2 x 128 (M=128), 3 x 64 (M=256)
//         A SF            448      2 x 16
//         B SF            480      2 x 16
// L0A     A                 0      2 x 32
// L0B     B                 0      2 x 32
// L0C     FP32 accum        0      2 x 64 (M=128), 1 x 256 (M=256)
```

**B 侧的两种供给方式**是今天最有趣的对比：

- **shared 权重是 FP8，AIC 自己搬**：`asc_copy_gm2l1_align` 直接 GM→L1（`:514-525`），布局是第 43 天 `_transform_fp8_nz` 产出的 NZ 顺序；
- **routed 权重是 FP4，AIV0 负责解包**：`aiv0_weight_pipeline`（`:298-399`）对每个 routed 任务执行三段小流水（UB 注释 `:307-317`：FP4 3×32 KiB → FP8 NZ 2×64.5 KiB → reduce 借剩余 31 KiB）：

```text
GM 的 FP4 权重块 [MAD_N, valid_k/2] int8
  ──copy_gm_to_ub──► ub_fp4（3 stages）          :364-368
  ──vf_fp4_to_fp8_nz──► ub_nz（2 stages）        :370-375
  ──asc_copy_ub2l1──► AIC 的 L1B（4 stages）     :378-381
```

`vf_fp4_to_fp8_nz`（`simd/mega_moe.hpp:12-55`）把两个 FP4 nibble 展开成 FP8 并散布进带一行动填充的 NZ 布局；文件头注释（`:10`）点明数值约定："FP4 -> FP8 NZ, scaled by 2^-6; weight SFs compensate with +6"——反量化值统一除以 64，由权重 SF 指数 +6 一次性补偿（正是第 43 天 `_scale_fp4_weight_sf` 做的事）。AIC 与 AIV0 之间用 `IntraBlockEvent<0>` 的 `kL1BFull/kL1BEmpty`（`mega_moe.hpp:96-97`）配对：AIV0 写满一个 L1B stage 就 `notify_intra_block<PIPE_MTE3, 0>(kL1BFull, ...)`，AIC 在 MAD 前 `wait_intra_block<PIPE_MTE1, 0>(kL1BFull, ...)`（`:569-570`），用完归还 Empty（`:645-646`）。shared 任务全部做完后，AIC 把 L1B 区域连同 4 个新信用**移交**给 AIV0（`:479-484`，注释 "Hand L1B over to AIV0 after shared tasks"）——因为 shared B 和 routed B 物理上复用同一段 L1。

**A 侧与主循环**：激活经 `copy_gm_to_l1` 进 L1A（3 stages，`:542-547`），SF 经 `copy_gm_to_l1_mx` 进专用 L1 区（`:527-562`；Linear1 的激活 SF 是 token-major，Linear2 的是 M-block 内 K-major，两种 `Major` 模板参数对应不同读取方式，`:550-557`）。K 维两层循环与 BF16 Kernel 同构（第 22 天）：

```cpp
for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {          // 外层：L1 级
    const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
    ...
    for (uint32_t k_mad_idx = 0; k_mad_idx < valid_k; k_mad_idx += MAD_K) {   // 内层：L0 级
        ...
        for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
            ...
            mad(l0c_mad, l0a_mad, l0b[l0b_stage_idx], aligned_valid_m, MAD_N, MAD_K,
                is_last_k ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                k_idx + k_mad_idx == 0);                              // :598-600
```

`ENABLE_KEEP` 中间累加、最后一个 K 块 `ENABLE_UPDATE` 写出、首个 K 块初始化 L0C——与第 24 天的语义一致。L0A 在 `BLOCK_N / MAD_N` 次乘法间复用（`:573` 注释 "Reuse L0A across the BLOCK_N / MAD_N multiplies"）。任务完成后 AIC 把每核计数器 `num_issued_gemms` 加一写回 GM（`:665`）——AIV1 的 epilogue 靠它知道 AIC 是否已发出该任务的 FIX 结果（见下节）。

**Linear2 的两点差异**：A 换成 `l2_acts`，加载前要等 `kLinear1Ready`（`:536-540`）；输出不再进 UB，而是 `copy_l0c_to_gm` 直接 L0C→GM 写 `routed.output`（或 shared 写 `combine_input` 的第 k 个 slot），并带 `kWriteThroughShare`（值 12 的 L2 写透模式，`:107, :625-632`）。若目标 ring 行块是回绕复用的，先等旧数据的 `kCombineSent`（`:607-610`）。

### 3.5 第三段 SwiGLU：AIC 与 AIV1 的 UB 交接

**这是"两个 GEMM 与 SwiGLU 如何衔接"的第一半答案，也是整个 Kernel 最精巧的握手。** Linear1 的 GEMM 结果不落 GM，而是留在 **AIC 与 AIV1 共同操作的 UB 缓冲 `ub_accum`** 上：AIC 侧声明 `constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N)`（`:412`），AIV1 侧声明了地址相同的另一个 `ub_accum`（`:698`）——`__mix__(1, 2)` 的 Block 内，AIC 的 FIX 管道把 L0C 转储进配对 AIV 的 UB，两边靠 `IntraBlockEvent<1>`（"AIC/AIV1: FIX -> V/S full, MTE3/S -> FIX empty"，`:99-101`）同步。

缓冲形状由两条常量决定（`:24-26`）：`kNumEpilogueStages = 2`、`STORE_BLOCK_M = BLOCK_M / 2 = 128`——每个 256 行的 M-block 拆成两个 128 行的 epilogue 段，双 stage 让 AIC 写第 2 段时 AIV1 还在处理第 1 段。AIC 的写出（`:612-636`）：

```cpp
const uint32_t fix_step_m = is_linear1 ? STORE_BLOCK_M : valid_m;
for (uint32_t m_idx_in_block = 0; m_idx_in_block < valid_m; m_idx_in_block += fix_step_m) {
    const uint32_t ub_stage_idx = m_idx_in_block / STORE_BLOCK_M;
    if (n_mad_idx == 0)
        wait_intra_block<PIPE_FIX, 1>(kEpilogueEmpty, ub_stage_idx);   // 等 AIV1 归还该段
    if (is_linear1) {
        // Store MAD_N gate/up columns with a BLOCK_N destination stride
        const ub_ptr<bfloat16_t> ub_dst(STORE_BLOCK_M, BLOCK_N,
                                        ub_accum[ub_stage_idx].addr + n_mad_idx * sizeof(bfloat16_t));
        copy_l0c_to_ub(ub_dst, l0c_mad.offset_nz_m(m_idx_in_block), ...);   // L0C→UB（FIX）
    } else {
        copy_l0c_to_gm(l2_output, l0c_mad, valid_m, MAD_N, config.hidden, kWriteThroughShare);
    }
    if (n_mad_idx + MAD_N == BLOCK_N)
        notify_intra_block<PIPE_FIX, 1>(kEpilogueFull, ub_stage_idx);  // 通知 AIV1 该段就绪
}
```

注意 `n_mad_idx` 作为**列偏移**：MAD_N=128 时第一个 n_mad 块的 gate 结果落在 `ub_accum` 第 0:128 列、第二个落在 128:256 列——这正是第 43 天 gate/up 交织权重的镜像：**一个 [128, 256] 的 UB 段里，前 128 列是 gate、后 128 列是 up**。

AIV1 侧的消费者是 `advance_epilogue`（`:887-987`），它按镜像的任务序列逐任务处理，三道闸门保证不超前也不落后：

1. **AIC 已发出**：`asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms`（`:892`，注释 "Issued tasks still need their FIX completion event"）；
2. **Linear2 任务**：只做信用归还得发布进度——`wait kEpilogueFull → notify kEpilogueEmpty`（`:900-903`），然后发布 `kLinear2Ready`（`:983-984`）。注释 `:99-101` 说明了这个复用："Linear1 and Linear2 share the same epilogue credits"；
3. **Linear1 任务**：等 `kEpilogueFull` 后执行 SwiGLU（`:932`）。

SwiGLU + 量化的数值核心在 `vf_swiglu_quant`（`simd/mega_moe.hpp:57-166`），每次处理 2 行、每行 128 对 gate/up：

```cpp
auto gate_ptr = input;                          // ub_accum 段的前 128 列
auto up_ptr = input + layout::BLOCK_N / 2;      // 后 128 列
...
// Clamp gate above and up on both sides        :105-112
asc_min_scalar(gate, gate, activation_clamp, ...);
asc_min_scalar(up, up, activation_clamp, ...);
asc_max_scalar(up, up, -activation_clamp, ...);
// SwiGLU: gate * up / (1 + exp(-gate))         :114-122
asc_add_scalar(value, value /*=exp(-gate)*/, 1.0f, ...);
asc_div(value, gate, value, ...);
asc_mul(value, value, up, ...);
if constexpr (kApplyRouteWeight)                // routed 才乘路由权重
    asc_mul(value, value, route_weight[...], ...);
```

三件事在量化前一次完成：clamp（gate 只上限、up 双侧）、SwiGLU、**路由权重乘入**（`kApplyRouteWeight=true` 仅 routed，`:943-945`；权重就是 Dispatch 拉取时存的 fp32 位模式，`:923-931` 从 GM 读回 UB）。然后是 MX 量化（`:124-163`）：对每 32 通道组求 amax，把 `amax/448` 向上取成 2 的幂作为 UE8M0 指数（一连串位操作 `:138-147`），乘倒数缩放后 `asc_float2e4m3_rn_sat` 舍入饱和成 FP8（`:151-155`），指数对打包进 int16（`:157-163`）。

最后落盘（`:949-971`）：FP8 数据写 `l2_acts`（routed 走 ring 行、shared 走 `shared_l2_acts`），SF 写 K-major 区——`dst_sf` 的下标 `output_m_idx * num_sf_pairs + n_block_idx * kNumOutputSFPairs * BLOCK_M + stage_m_begin`（`:964-965`）体现了 `[N 块][pair][块内行]` 的顺序，与 `layout/mega_moe.hpp:194` 的注释一致："Input SF is token-major; quantized Linear2 SF is K-major within each M-block"。写完 `notify kEpilogueEmpty` 归还 UB 段（`:971`），并发布 `kLinear1Ready`（`:980-984`）。

**衔接机制小结**：

| 交接 | 机制 | 缓冲 | 等待方 |
|---|---|---|---|
| GEMM1 → SwiGLU | Block 内 `IntraBlockEvent<1>`（FIX→V 信用，2 段双缓冲） | `ub_accum` [128, 256] bf16 ×2 | AIV1 等 `kEpilogueFull` |
| SwiGLU → GEMM2 | GM 原子进度 `kLinear1Ready`（发布需 DSB_DDR 屏障，`:732-744`） | `l2_acts` + SF | AIC 等计数凑满 `I/128`（`:536-540`） |

为什么一个用事件、一个用计数？`ub_accum` 在同一个 Block 内、两个核独占共享，硬件事件最便宜；`l2_acts` 在 GM 里、要跨 Block 被 AIC 的各个核读取，只能走内存可见性协议（`asc_sync_data_barrier(DSB_DDR)` + `asc_atomic_add`，`publish_block_progress`，`:732-744`）。

### 3.6 第四段 Grouped GEMM 2 与第五段 Combine

**GEMM2（Linear2）**在 3.4 已随主循环讲完，这里只总结差异：A 是 SwiGLU 产出的 `l2_acts`（K=I），N 按 H/256 分块，输出直接 `copy_l0c_to_gm` 进 `routed.output`（ring 行）或 `combine_input` 的 shared slot。注意 shared Linear2 的输出行号用 `get_combine_m_idx(config.num_topk, pool_block_idx * BLOCK_M)`（`:628-629`）——**shared 输出占 `combine_input` 的第 k 个 slot**，与 k 条路由的 slot 并列。

**Combine 分两步：push 与 reduce。**

**Push（AIV1，`advance_combine_push`，`:999-1053`）**把每个池内行的 BF16 输出发回它的源 token。每行两步：读 `src_metadata` 解出 `(dst_rank, combine_m)`（`:1019-1020`；其中 `combine_m = slot * T_max + src_token_idx`，`get_combine_m_idx`，`layout/mega_moe.hpp:242-244`），然后 `copy_gm_to_ub` 拉一行 `routed.output`、`copy_ub_to_gm` 写进目标 rank 的 `combine_input`（经 `sym_buffer.map` 平移，`:1024-1036`）。push 每步至多发 2 行就让出（`:1015` 注释 "Send at most two rows before yielding"），每 `kNumMBlocksPerWait = 32·32/256 = 4` 个 M-block 批量等待并发布一次 `kCombineSent`（`:1002, :1044-1050`），让 AIC 能复用 ring 行；`kNumPushStages = 4` 个 UB 行缓冲让远端写与本地读重叠。

**Reduce（64 个 AIV，`combine_reduce` + `advance_combine_reduce`，`:139-295`）**是最终的加权求和。对每个本地 token：

1. 收集 `kNumInputs = num_topk + (S>0)` 份输入（`make_reduce_buffer`，`:128-136`——UB 里开 2 个 reduce stage，每个能装下 `(k+1) × num_channels` 的 BF16）：逐 slot 检查 `topk_idx` 是否有效（`>= num_experts` 跳过，`:211-214`），从 `combine_input + slot*T_max*H + token*H` 拷进 UB，shared 的输出垫在最后（`:209-217`）；
2. `vf_combine_reduce`（`simd/mega_moe.hpp:169-200`）做 **FP32 累加、一次舍入成 BF16**（`:221` 注释 "Accumulate in FP32, then round once to BF16"）——这就是第 43 天测试阈值 `REFERENCE_DIFF_THRESHOLD = 5e-6` 特别严的原因：DeepEP 的 baseline 把各 rank 部分和按 BF16 逐次舍入，MegaMoE 全程 FP32；
3. 结果写 `y[token * H ...]`，带 `kWriteThroughShare`（`:226-231`）。

非重叠模式下 reduce 很简单：`rank_barrier` 等全体 rank 到齐后顺序处理（`:262`）。重叠模式（`kOverlapReduce`）则复杂得多：token 按 `aiv_idx` 轮询跨步、用 `reduced_token_mask` 位图跳过已完成的（`:162-173`）；每个 token 先检查其各条路由的**生产块**是否已由 peer 推送完毕——读 `route_m_blocks` 得知该路由的输出在哪个池块产生，再对比 peer 写来的 `combine_ready_prefix`（`:175-197`）；AIV1 的 `publish_combine_prefix`（`:1055-1077`）负责把自己已完成的块前缀发布给所有源 rank。AIV0/AIV1 在各自主循环的空隙里以 `num_steps` 为步长推进 reduce（`:389-391, :1081-1089`），只有收尾才进 `combine_reduce` 排干剩余 token。这条重叠时间线的完整分析是**第 45 天**的主题。

### 3.7 进度链与环形复用：五段如何共用一块接收池

五段流水能装进一个 Kernel 而不互相踩踏，靠的是 `layout/mega_moe.hpp:27-33` 的四个进度阶段加环形下标。先看进度表（计数器在 `progress.blocks`，每次 launch 前由 Host `aclrtMemsetAsync` 清零，`csrc/jit_kernels/mega_moe.hpp:44-46`；读取只看低字节，所以 N 块数必须 ≤ 255，`mega_moe.hpp:1144-1147`）：

| 阶段 | 语义 | 发布者（每槽 +1） | 等待者 | 凑满计数 |
|---|---|---|---|---|
| `kDispatched` | `l1_acts` 的某 M-block 行已拉齐 | AIV1 pull（`:772`） | AIC 加载 Linear1 A（`:536-540`） | `kNumAICores`（32 个 pull worker） |
| `kLinear1Ready` | `l2_acts` 的某 M-block 行已量化落盘 | AIV1 epilogue（`:980-982`） | AIC 加载 Linear2 A（`:536-540`）；pull 复用 ring 行（`:779-784`） | `kNumLinear1NBlocks` = I/128 |
| `kLinear2Ready` | `routed.output` 的某 M-block 行已写出 | AIV1 epilogue（`:983-984`） | combine push（`:1003-1006`）；SwiGLU 复用 l2_acts（`:905-910`） | `kNumLinear2NBlocks` = H/256 |
| `kCombineSent` | 某池块的发送完成，ring 行可复用 | AIV1 push（分批，`:1044-1050`） | AIC Linear2 写回绕行块（`:607-610`） | `kNumAICores` |

槽位下标统一偏移 `num_shared_m_blocks`（shared 块与 routed 块共用进度数组）。发布前必须 `notify/wait kStoresDone`（MTE3→S）+ `DSB_DDR`，保证 GM 存储可见后再累加计数（`publish_block_progress`，`:732-744`）。

**环形接收池**（`layout/mega_moe.hpp:221-226, 257-261`）：池逻辑上有 `num_max_m_blocks = ceil_div(R·T_max·k, 256) + E/R` 个块，但三块 routed 缓冲（`l1_acts`/`l2_acts`/`output`）只开 `num_ring_blocks = min(num_max_m_blocks, 窗口数 × 32)` 行（窗口数 = `H/256 ≤ 16 ? 4 : 1`，注释 `:222`："Measured overlap window; the deadlock-safe minimum can still lose throughput"）。当池块号 ≥ 环块数时，`get_ring_block_idx = pool_block_idx & (kNumRingBlocks - 1)` 回绕映射。回绕不是免费的——**写回绕行前必须等它的上一个生命周期结束**，三种缓冲各等各的：

```text
向 ring 槽 (b - ring) 写新数据之前：
  写 routed.output[b & mask]  ←  等 kCombineSent[b - ring]   （旧输出已全部 push 走）
  写 l2_acts[b & mask]        ←  等 kLinear2Ready[b - ring]  （旧输入已被 GEMM2 消费）
  写 l1_acts[b & mask]        ←  等 kLinear1Ready[b - ring]  （旧激活已被 GEMM1+SwiGLU 消费）
```

（分别对应 `:607-610`、`:905-910`、`:779-784`。）这张"生产者-消费者"网就是第 43 天说的"接收池是环形缓冲，计算消费掉一块后 Dispatch 才能复用这块"的代码实现。

最后用一张时序示意图把今天的五段串起来（单个 Block 的视角，省略流水细节）：

```text
AIC   : [shared L1][shared L2][L1(b0)][L1(b1)][L2(b0)][L1(b2)][L2(b1)] ...
AIV1  : [pull b0  ][pull b1][SwiGLU b0 ][SwiGLU b1 ][SwiGLU b2 ] ... [push][reduce]
         └─kDispatched─┘        └─kLinear1Ready─┘   └─kLinear2Ready─┘
AIV0  : [元数据][routed 权重 FP4→FP8 NZ 流水（与 AIC 的 routed 任务同步）]
                      └─kL1BFull/Empty（Block 内事件）─┘
```

### 3.8 当天产出·参考：MegaMoE 张量形状变化表

先给**符号版**（R=rank 数、E=expert 数、S=shared expert 数、T=token 数 ≤ T_max、k=top-k、H=hidden、I=intermediate；`pool_rows = num_max_m_blocks × 256`、`ring_rows = num_ring_blocks × 256`，见 `layout/mega_moe.hpp:121-130, 217-226`）：

| # | 阶段 | 张量 / 区域 | 形状 | dtype | 布局要点 | 生产者 → 消费者 |
|---|---|---|---|---|---|---|
| 0 | 输入 | `input.acts` (x) + sf | [T, H] + [T, H/64] | fp8 + int16 | token-major，SF 每 64 元素一对指数 | 用户 → Dispatch pull（跨 rank 读） |
| 0 | 输入 | `input.topk_idx` / `topk_weights` | [T, k] | int64 / fp32 | `-1` 为无效路由 | 用户 → 元数据 |
| 1 | 元数据 | `total_expert_count` | [E] | uint32 | 每 expert 全局路由数 | AIV0 → 各 rank 互发 |
| 1 | 元数据 | `src_token_metadata` | [E/R, R, T_max·k] | uint64 | `(weight_bits<<32) \| token_topk_idx` | 全体 rank → 本地 pull |
| 1 | 元数据 | `expert_recv_count_sum` | [E/R] | uint32 | 值[23:0] + 到达数[31:24] 打包 | 全体 rank → AIV0 汇总 → 证书 |
| 2 | Dispatch | `routed.l1_acts` + sf | [ring_rows, H] + [ring_rows, H/64] | fp8 + int16 | 环形；行 = 池行 & mask；expert 顺序拼接 | AIV1 pull → AIC GEMM1 的 A |
| 2 | Dispatch | `routed.topk_weights` | [pool_rows] | fp32（位模式） | 每池行一个路由权重 | AIV1 → SwiGLU |
| 2 | Dispatch | `routed.src_metadata` | [pool_rows] | uint32 | rank[31:24] \| combineM[23:0] | AIV1 → Combine push |
| 3 | GEMM1 | L0C | [MAD_M, MAD_N] ×(BLOCK_N/MAD_N) | fp32 | 每 K 块一块，KEEP 累加 | mad → FIX |
| 3 | GEMM1 | `ub_accum` | [128, 256] × 2 段 | bf16 | 列 0:128 gate、128:256 up | AIC FIX → AIV1 SwiGLU（Block 内共享） |
| 4 | SwiGLU | `routed.l2_acts` + sf | [ring_rows, I] + [ring_rows·I/64] | fp8 + int16 | SF 在 M-block 内 K-major | AIV1 → AIC GEMM2 的 A |
| 4 | SwiGLU | `shared_l2_acts` + sf | [pad(T,256), S·I] | fp8 + int16 | shared 专用，不经 ring | AIV1 → AIC shared GEMM2 |
| 5 | GEMM2 | `routed.output` | [ring_rows, H] | bf16 | L0C 直接写 GM，写透 L2 | AIC → Combine push |
| 5 | GEMM2 | （shared）`combine_input` 第 k slot | [pad(T,256), H] | bf16 | slot=k | AIC → reduce |
| 6 | Combine | `combine_input` | [(k + (S>0)) · T_max, H] | bf16 | slot = 路由序号或 shared | push（本/远端 rank 写）→ reduce |
| 7 | 输出 | `y` | [T, H] | bf16 | FP32 累加一次舍入 | reduce → 用户 |

再给**贯穿示例的数值版**（第 43 天同款配置：R=2, E=8, S=0, T=128, k=2, H=I=512，并假设路由均匀——每个 expert 恰好收到 2 rank × 64 = 128 条路由）：

```text
派生容量：num_max_m_blocks = ceil(2·128·2/256) + 8/2 = 2 + 4 = 6 → pool_rows = 1536
          ring: H/256 = 2 ≤ 16 → 4 窗口 → 128 块；min(6,128) = 6 → ring_rows = 1536（本例不回绕）
运行时量：每个本地 expert 128 token → 各占 1 个 M-block（valid_m=128）→ num_total_m_blocks = 4
          Linear1 任务 = 4 × (512/128) = 16；Linear2 任务 = 4 × (512/256) = 8
          kOverlapReduce: 128·2·512·2B = 256 KiB < 32 MiB → 关闭（最终一次性归约）
```

| 阶段 | 张量 | 数值形状 | 说明 |
|---|---|---|---|
| 输入 | x / x_sf | [128, 512] / [128, 8] | |
| 输入 | topk_idx / topk_weights | [128, 2] | |
| 元数据 | expert_recv_count_sum | [4] | 每 expert = 64+64 = 128 |
| 元数据 | src_token_metadata | [4, 2, 256] | 每 expert 每源 rank 最多 T·k = 256 槽 |
| Dispatch | l1_acts / sf | [1536, 512] / [1536, 8] | 768 KiB / 24 KiB，只用前 4×128 = 512 行 |
| Dispatch | topk_weights / src_metadata | [1536] / [1536] | 各 6 KiB |
| GEMM1 | 单任务 (pool=1, n=1) | A [128, 512] × B [256, 512] | valid_m=128 ≤ 128 → MAD 档 M=128：BLOCK_K=512、MAD_K=256、MAD_N=128；K 循环 1 次、内层 2×2 = 4 次 mad，每次 [128, 128]×K256 |
| GEMM1 | ub_accum | [128, 256] × 2 | gate 列 0:128、up 列 128:256 |
| SwiGLU | 单任务输出 | [128, 128] fp8 + 每行 2 个 SF 对 | 存 l2_acts 行 256:384、列 128:256 |
| SwiGLU | l2_acts / sf | [1536, 512] / [12288 个 int16] | K=I=512，SF 24 KiB |
| GEMM2 | 单任务 (pool=1, n=0) | A [128, 512] × B [256, 512] → 输出 [128, 256] bf16 | 写 routed.output 行 256:384、列 0:256 |
| GEMM2 | routed.output | [1536, 512] bf16 | 1.5 MiB |
| Combine | combine_input | [2, 128, 512] bf16 | 2 slot × T×H，共 256 KiB |
| 输出 | y | [128, 512] bf16 | 每 token 2 输入 FP32 求和 |

自画这张表时的自查点：pool_rows 与 ring_rows 是否分清、Linear1 的 N 块数为什么是 I/128 而不是 2I/256 之外的数（交织后每 256 行权重块恰好对应 SwiGLU 的 128 输出列）、`ub_accum` 为什么是 128×256（STORE_BLOCK_M × BLOCK_N）、combine_input 的 slot 数为什么是 k+(S>0)。**通信对象与时间重叠的展开属于第 45 天**，今天的表只需标注"跨 rank"于哪几行（pull 的源、push 的目标）。

## 四、源码导读

今天的主线文件与阅读站点（建议按表中顺序读）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `deep_gemm/include/deep_gemm/mega_moe.hpp` | `MMAConfig` `:28-48`、事件表 `:61-107`、`aic_gemm_pipeline` `:401-669`、`aiv0_weight_pipeline` `:297-399`、`aiv1_epilogue_comm` `:671-1123`、`advance_combine_reduce` `:139-249`、`combine_reduce` `:251-295`、`mega_moe_impl` `:1125-1219` | 今天的主战场，全部执行逻辑 |
| 2 | `deep_gemm/include/deep_gemm/scheduler/mega_moe.hpp` | `BlockPhase` `:9`、`LinearTaskState` `:11-16`、`ExpertCursor` `:19-54`、`take_linear_task` `:58-118` | 只有 120 行，但值得逐行读完 |
| 3 | `deep_gemm/include/deep_gemm/layout/mega_moe.hpp` | `ProgressStage` `:27-33`、`Config` `:36-47`、`Workspace` `:85-250`、`get_num_max_m_blocks` `:217-219`、`get_num_ring_blocks` `:221-226`、`get_combine_m_idx` `:242-244`、`get_ring_block_idx/m_idx` `:257-266` | 今天重点从"视图"视角换成"执行区域"视角 |
| 4 | `deep_gemm/include/deep_gemm/simd/mega_moe.hpp` | `vf_fp4_to_fp8_nz` `:12-55`、`vf_swiglu_quant` `:57-166`、`vf_combine_reduce` `:169-200` | 三段 SIMD 数值内核；文件头注释 `:10` 解释 FP4 缩放约定 |
| 5 | `deep_gemm/include/deep_gemm/comm/mega_moe.hpp` | `build_dispatch_metadata` `:117-240`、`wait_dispatch_metadata` `:242-275`、`is_block_ready` `:100-106` | 元数据四拍流程与打包计数；`rank_barrier`/`wait_until` 留到第 45 天 |
| 6 | `deep_gemm/include/deep_gemm/ascend/sync.hpp` | `IntraBlockEvent` `:45-53`、`PipeEvent` `:12-27` | 事件类型系统："kSubblock selects the AIC/AIV0 or AIC/AIV1 connection" |
| 7 | `deep_gemm/include/deep_gemm/common.hpp` | `:51-65` | 硬件常量：32 AIC / 64 AIV、L1/L0/UB 容量、FRAC_MN、MX_SF_DIVISOR |
| 8 | `csrc/jit_kernels/mega_moe.hpp` | `launch_mega_moe` `:24-51` | progress 区清零（`:44-46`）、32 Block 启动（`:48-50`） |
| 9 | `tests/test_mega_moe.py` | `check_correctness` `:166-200`、`benchmark` `:203-243`、`run_baseline` `:130-156` | baseline 的五段拼装与本日讲解的对应关系；FLOPS 口径 `flops = 6·H·I·(recv_tokens + T·shared)`（`:221`） |
| 10 | `README.md` | MegaMoE 节 `:160-173` | 基准口径（EP8、top-k=6、1 shared expert） |

**`mega_moe.hpp` 分函数导读**（行号均指该文件）：

- **`:20-107` 常量与事件**：流水线档位（L0 双缓冲、L1 A×3 / routed B×4 / SF×2、epilogue×2）；MMAConfig 双档表；L1/L0 布局注释；全部 `PipeEvent`/`IntraBlockEvent` 的声明与用途注释——**先读懂这张事件表，后面的 wait/notify 才不会迷路**。
- **`:401-669` `aic_gemm_pipeline`**：主循环四步——`take_linear_task` 取任务（`:441-454` 含元数据门）→ 解码（`:456-493`）→ `compute` lambda（`:495-658`：B 供给两种、SF 分块加载、`wait_for_block` 等 A 就绪、两层 K 循环 + MAD、epilogue 分叉）→ 计数（`:665`）。`is_first/is_last` 的 mad 语义在 `:598-600`。
- **`:297-399` `aiv0_weight_pipeline`**：任务镜像取权重块 → FP4/FP8/L1B 三段小流水（`:352-393`）→ 权重 L2 提示按 expert 大小分档（`:345-346`）。
- **`:671-1123` `aiv1_epilogue_comm`**：UB 布局注释 `:687-717`；`publish_block_progress` `:732-744`；`advance_dispatch_pull` `:763-873`；`advance_epilogue` `:887-987`；`advance_combine_push` `:999-1053`；`publish_combine_prefix` `:1055-1077`；主调度循环 `:1079-1123`（reduce 窗口、取任务、元数据轮询、epilogue/push/pull 交替，`prefer_push` 翻转避免饿死）。
- **`:139-295` reduce**：`CombineReduceState` `:109-125`（可恢复的断点状态——token/channel/stage 游标）；`make_reduce_buffer` `:128-136`；`advance_combine_reduce` 的分块累加循环 `:205-246`。

**`layout/mega_moe.hpp` 区域导读**：`metadata`（`:142-163`）今天重点看 `expert_recv_count/sum`、`num_m_blocks_certificate`、`src_token_metadata`；`routed`（`:187-203`）注意 `:194` 的 SF 布局注释；`route_m_blocks`（`:213-215`）只服务重叠 reduce。

## 五、实践步骤

### A 组：有 Ascend 环境（至少 2 卡；对照 baseline 另需 `deep_ep` 与 `tilelang`）

1. **小配置跑通并对照本讲流程**：

   ```bash
   python tests/test_mega_moe.py --num-processes 2 \
       --hidden 512 --intermediate-hidden 512 --num-experts 8 \
       --num-topk 2 --num-tokens 64 --num-shared-experts 1 --skip-prof
   ```

   通过后把 3.8 节的贯穿示例在脑中过一遍：本配置下 `num_total_m_blocks` 是多少（提示：`--num-tokens 64` 且容量按 64 注册，2 rank × 64 token × 2 路由均匀分给 4 个本地 expert，每 expert 32 条 → 4 个块、valid_m=32）。若有 `deep_ep`，去掉 `--skip-baseline` 观察正确性断言（`test_mega_moe.py:166-200`）。

2. **验证 Dispatch 元数据主线**：写一个小脚本（模仿 `test` 的 spawn + `init_dist`，`test_mega_moe.py:246-282`）：2 rank、`E=8, k=2, T=128, H=I=512`，随机路由，跑一次 `fp8_fp4_mega_moe` 后 `torch.npu.synchronize()`，读取 `buffer.expert_recv_count.cpu()`（形状 [2, 4]：各源 rank 对各本地 expert 的计数），并独立统计 `topk_idx` 的 bincount 后 `dist.all_reduce`（完全照抄 `check_correctness` 的做法，`test_mega_moe.py:189-194`），断言两者一致。再加 `invalid_routes`（`topk_idx[::13, -1] = -1`）重跑，确认无效路由不计数。

3. **空输入与空 expert 路径**：

   ```bash
   python tests/test_mega_moe.py --num-processes 2 --hidden 512 --intermediate-hidden 512 \
       --num-experts 8 --num-topk 2 --num-tokens 0 --num-shared-experts 1 --skip-prof
   ```

   对照源码推演：T=0 时 `num_shared_m_blocks = 0`、`num_total_m_blocks = 0`，AIC 的任务循环直接走到 `kNone`（`mega_moe.hpp:441-454`），reduce 也无 token——但 Kernel 仍需完整走完同步协议（`combine_reduce` 里"Empty ranks also wait for peer writes"，`:286-294`）。

4. **（进阶）观察确定性**：`--num-correctness-tests 3` 运行，对照 `check_correctness` 的逐位相等断言（`test_mega_moe.py:199`）体会"任务重放 + 固定握手顺序"为什么能保证输出逐位确定。

### B 组：无 Ascend 环境（静态任务）

1. **形状变化表手算**（当天产出，对应 3.8 节）：用配置 `R=2, E=8, S=0, T=128, k=2, H=I=512`、均匀路由，先算 `num_max_m_blocks`、`num_ring_blocks`、`num_total_m_blocks`、两类任务数，再逐行填表（含字节量）；然后换 `S=1` 重算——哪些行变化？（`combine_input` 多一个 slot、`shared_l2_acts` 出现、`kNumLinear1NBlocks` 对 shared 变为 `I/128` 不变但 K/N 拼宽……）最后对照 3.8 节修正。
2. **任务发牌推演**：沿用上题配置（Linear1 任务 16、Linear2 任务 8、32 核）。逐核调用 `take_linear_task` 三次，写出核 0、9、17 各自的 `(phase, task_idx)` 序列（预期：核 0 = L1 t0 → L2 t0 → None；核 9 = L1 t9 → None；核 17 = None）。再把 `take_linear_task` 的 lookahead 条件（`scheduler/mega_moe.hpp:97-104`）代入验证"为什么第二轮全部发 Linear2"。思考：若 `I=3072、H=7168`，每 M-block 有多少 Linear1/Linear2 任务？多少 M-block 以上才能填满 32 核？
3. **ExpertCursor 演算**：某 rank 的 `expert_recv_count_sum` 为 `[300, 20, 0, 77]`。手工推演：每个 expert 占哪些池块、`pool_block_end` 序列、`num_total_m_blocks`，以及 expert 0 两个块的 `expert_m_begin/valid_m`（预期 160/140——用 `get_m_block_range` 的 FRAC_MN 分组公式算，`scheduler/mega_moe.hpp:33-43`）。再算 `[20]` 与 `[77]` 的 `valid_m`（预期 20 与 77，注意 `aligned_valid_m` 是 32 和 80）。
4. **回绕与重叠阈值手算**：配置 `R=2, E=8, k=2, H=512, T_max=8192`：算 `num_max_m_blocks`（预期 132）、`num_ring_blocks`（预期 128）、池块 128-131 回绕到哪些 ring 槽（预期 0-3）；写出三块缓冲回绕写入前各要等哪个进度阶段的哪个槽。再判断该配置的 `kOverlapReduce`（预期 false，16 MiB < 32 MiB）；把 k 改成 6 呢？（预期 true。）
5. **衔接链标注**：合上教程，在纸上写出五处数据交接及其机制——①AIV0 的 L1B → AIC（事件？哪对 full/empty？）；②AIC 的 `ub_accum` → AIV1 SwiGLU；③AIV1 SwiGLU 的 `l2_acts` → AIC GEMM2；④AIC 的 `routed.output` → AIV1 push；⑤push 的 `combine_input` → reduce。每处标注：共享介质（UB/L1/GM）、同步原语（IntraBlockEvent / 进度计数 + DSB）、等待者期望计数。再对照 3.5 与 3.7 节修正。

## 六、当天产出

**MegaMoE 张量形状变化表**——以 3.8 节的两张参考表为答案，自己合上教程完整画一版，至少覆盖：从 `x`/`topk_idx` 到 `y` 的每个中间张量（含元数据区与 UB 上的 `ub_accum`）、每行的形状/dtype/布局要点（token-major vs K-major、ring vs pool、slot 结构）、生产者与消费者、以及五处衔接机制（哪几处跨 rank、哪几处 Block 内）。有环境的同学附上 A 组第 2 步的 `expert_recv_count` 对照记录。写完后把说不清的行标出来，回到第四节对应站点重读。

## 七、自测题

1. `mega_moe_impl` 的 `__mix__(1, 2)` 是什么含义？三类角色各自负责五段中的哪几段？
   > 每个 Block = 1 AIC + 2 AIV，共 32 个 Block（`mega_moe.hpp:1127, :1160-1213`）。AIC：两个 Grouped GEMM 的全部 MAD 与写出；AIV0：Dispatch 元数据 + FP4→FP8 NZ 权重流水；AIV1：Dispatch 拉取、SwiGLU epilogue、Combine push；64 个 AIV 共同做 Combine reduce。
2. `BlockPhase` 有哪几种？`take_linear_task` 的发放顺序是什么？为什么 Linear1 有优先权？
   > `kNone/kLinear1/kLinear2/kSharedLinear1/kSharedLinear2`（`scheduler/mega_moe.hpp:9`）。顺序：SharedLinear1 → SharedLinear2 → routed；routed 里 Linear1 优先，直到它领先到能覆盖下一波 Linear2 所需的 M-block 再加一整波余量（lookahead 条件 `:97-104`），因为每个 M-block 的 Linear2 依赖其全部 Linear1 N 块的 SwiGLU 产出。
3. AIC 和 AIV1 没有共享任务队列，它们如何对齐任务？
   > 各自维护 `LinearTaskState`，用相同参数重放纯函数 `take_linear_task`（`scheduler/mega_moe.hpp:56-57`），每个核看到相同序列；AIV1 用 AIC 写的 `num_issued_gemms`（`mega_moe.hpp:665`）确认对端已发出（`:892`），再用 epilogue 信用等 FIX 落 UB。
4. `ExpertCursor::get_m_block_range` 为什么用 FRAC_MN=16 的组来平衡，而不是前满后空地切？
   > MAD 的 NZ 分形要求 M 按 16 对齐（`aligned_valid_m` 为 16 的倍数）；把 16 行组均匀分给各块，300 token 分 2 块得 160/140 而非 256/44，避免尾块 MAD 空转（`scheduler/mega_moe.hpp:31-43`）。
5. GEMM1 的结果如何到达 SwiGLU？SwiGLU 的输出又如何到达 GEMM2？两个机制有何本质不同？
   > 前者：AIC 用 FIX 把 L0C 写进与 AIV1 共享的 `ub_accum`（[128,256] bf16 × 2 段），`IntraBlockEvent<1>` 的 FIX→V 信用配对（`mega_moe.hpp:612-636, 900-946`）——Block 内 UB 独占共享，硬件事件即可。后者：SwiGLU 量化后写 GM 的 `l2_acts`，发布 `kLinear1Ready` 原子计数，AIC 等它凑满 `I/128`（`:536-540, :980-984`）——GM 数据要跨 Block 消费，必须 DSB_DDR + 原子加的内存可见性协议。
6. `ub_accum` 一段 [128, 256] 里 gate 和 up 各在哪？这个安排和第 43 天的哪个变换互为镜像？
   > gate 在列 0:128、up 在 128:256（AIC 以 `n_mad_idx` 为列偏移写入，`:619-624`；SwiGLU 以 `up_ptr = input + BLOCK_N/2` 读取，`simd/mega_moe.hpp:79-81`）。镜像即 `_interleave_weights` 的 gate/up 交织：权重每 256 行块 = 128 行 gate + 128 行 up，输出列恰好按此排列。
7. 路由权重（topk_weights）在哪一步、以什么形式参与计算？
   > Dispatch 拉取时以 fp32 位模式存进 `routed.topk_weights[pool_m_idx]`（`mega_moe.hpp:857-859`）；SwiGLU 里先读回 UB（`:923-931`），在量化**之前**乘进 SwiGLU 结果（`kApplyRouteWeight=true` 仅 routed，`simd/mega_moe.hpp:120-121`）——只做一次舍入。
8. `combine_input` 有几个 slot？shared expert 的输出放在哪？reduce 的舍入策略与 baseline 有何不同？
   > `num_topk + (num_shared_experts > 0)` 个 slot，每 slot T_max 行 H 列（`layout/mega_moe.hpp:206-208`）；shared Linear2 直接写第 k 个 slot（`mega_moe.hpp:628-629`）。reduce 全程 FP32 累加、最后一次舍入成 BF16（`:221`、`simd/mega_moe.hpp:169-200`），而 DeepEP baseline 按 BF16 逐次舍入部分和——这就是测试阈值 5e-6 的由来（`test_mega_moe.py:18-19`）。
9. 三块 routed 缓冲为什么要做成环形？回绕写入前各要等什么？
   > 环形大小 `min(num_max_m_blocks, 窗口数·32)`（`layout/mega_moe.hpp:221-226`），让接收/中间/输出缓冲远小于最坏池容量，计算与搬运重叠复用。回绕时：写 `routed.output` 等 `kCombineSent`（`:607-610`）、写 `l2_acts` 等 `kLinear2Ready`（`:905-910`）、写 `l1_acts` 等 `kLinear1Ready`（`:779-784`），均针对 `pool_block_idx - kNumRingBlocks` 的旧块。
10. `kOverlapReduce` 由什么决定？它改变了 reduce 的哪些行为？
    > 编译期常量：`T_max·(k+(S>0))·H·2B > 32 MiB`（`:1138-1140`）。开启后 token 按 `aiv_idx` 轮询跨步、用位图跳过已完成 token，AIV0/AIV1 在主循环空隙分步推进 reduce（断点状态 `CombineReduceState`），并通过 `route_m_blocks`/`combine_ready_prefix` 检查 peer 的生产进度（`:162-197`）；关闭则 rank barrier 后按连续区间一次性归约（`:262, :1171-1176`）。

---

明天（第 45 天）进入 MegaMoE 的通信与融合：阅读 `comm/mega_moe.hpp` 的 `rank_barrier`/`wait_until` 与 `csrc/backends/hccl.hpp`，看 EP Dispatch/Combine 的通信对象、UBMEM 直访的细节，以及计算与通信如何 overlap——今天 3.6/3.7 节里一笔带过的 `kOverlapReduce` 时间线与 `publish_combine_prefix` 协议将全面展开。
