# 第 4 周 · 第 4 天（总第 25 天）：理解 AIC/AIV 和 Epilogue

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 4 周"BF16 Kernel——MAD、流水线与 Epilogue"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 45 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

前三天我们一直在 AIC（cube 核）的视角里工作：第 22 天拆了 K 维两层循环，第 23 天配齐了 MTE1/MTE2/M 的事件闸门，第 24 天看清了 `mad()` 如何把结果留在 L0C。但数据流图的最后一段——`L0C → 写回 GM`——我们每次都停在 `epilogue.aic_store(l0c_tile, s_epi, ...)`（`bf16_gemm.hpp:161`）这一行就收住了。今天打开这最后一扇门。首先要面对一个此前被刻意忽略的事实：`bf16_gemm_impl` 不是"一个核在跑"，它的启动声明是 `__global__ __mix__(1, kDirectStore ? 0 : 2)`（`bf16_gemm.hpp:27`）——**每个 block 由 1 个 AIC 和 2 个 AIV 三方共同执行同一份内核代码**，靠 `if ASCEND_IS_AIC`（`:81` / `:173`）分成两条完全不同的路径。AIC 负责"算出来"（搬运、MAD、把 FP32 结果倒进 AIV 的 UB），AIV 负责"交出去"（加 bias / 乘 alpha / 精度转换 / 写回 GM）。两方靠**块内（intra-block）事件**在 UB stage 上完成一次生产者-消费者握手。要回答四个问题：**`__mix__` 与 `ASCEND_IS_AIC` 如何划分两个分支的职责**？**`aic_store`（AIC 侧）与 `aiv_drain`（AIV 侧）如何靠四步借还协议配合**？**accumulation、bias、alpha 和输出转换分别发生在哪一层、哪条队列**？**两个 AIV 与 AIC 的 `s_epi` 为什么能不通信地保持一致**？当天产出是一张 **AIC/AIV 分工与 Epilogue 流程图**（参考答案见 3.9 节），并接住第 23/24 天留下的"M 写 L0C 与 FIX 读 L0C 跨队列次序"问题。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 解释 `__mix__(1, kDirectStore ? 0 : 2)` 的含义（一个 block = 1 个 cube 子块 + 2 个 vector 子块；`kDirectStore` 时没有 AIV），说明 `ASCEND_IS_AIC`、`ASCEND_IS_AIV` 与 `asc_get_sub_block_id()`（AIV 上返回 0/1）三者在 `bf16_gemm.hpp:81-192` 中各自标记的代码身份；
2. 列出两个分支的完整职责表：AIC 分支（调度、GM→L1→L0 搬运、MAD、交付）与 AIV 分支（`kDirectStore` 直接退出 → K-grouped 空组清零 → `aiv_initialize` → drain 循环），并说明为什么 AIV 分支也要构造同一个 `Scheduler`（`bf16_gemm.hpp:73`）并以相同顺序遍历；
3. 逐参数解读 `aic_store`（`epilogue.hpp:42-47`）：`copy_l0c_to_ub` 的 `dst_subblock_id = aiv_id` 如何把结果写进**目标 AIV 自己的 UB**、`quant_mode` 默认值 `infer_quant_mode<cd_dtype_t, float>()` 在这里完成 FP32→BF16 的输出转换、`set_intra_block(PIPE_FIX, ...)` 排在拷贝之后的队列含义；
4. 写出块内事件的编码 `((subblock_id & 1) << 4) | (flag_id & 0xf)`（`sync.hpp:30`），解释"AIC 用完整编码等待、AIV 忽略 subblock 位"（`sync.hpp:33-40` 注释）这一不对称设计的必要性，并把 UB stage 的借还整理成四次调用构成的闭环：`aic_wait_free` → `aic_store`（copy + arrive）→ `aiv_drain` 的 `wait` → 处理 + `release`；
5. 逐行走读 `aiv_drain`（`epilogue.hpp:49-87`）：round-robin 路由过滤（`:57`）、`initialize_aiv_drain` 在等待之前预取 bias（`:53-54`）、原地（in-place）处理与 `sizeof` 断言（`:28`、`:71-72`）、`kPipe != PIPE_MTE3` 时 V→MTE3 的 `store_event` 握手（`:76-80`）、`copy_ub_to_gm`、`postprocess_mad_block`、挂在 MTE3 上的归还 arrive（`:84`）；
6. 用一张"语义 → Python 输入 → host 判定 → device 位置"表说清 accumulation（原子加在 GM store）、bias（AIV 的 V 管道加）、alpha（AIV 的 V 管道乘）、输出转换（Identity 时在 AIC 的 FIX 拷贝，带算子时挪到 AIV）各自的位置，以及被禁止的组合（bias×alpha、bias×accumulation）及其源码依据；
7. 给定 `kNumEpilogueStages`、`MAD_M/N`、`cd_dtype_t` 和 epilogue 算子，计算**每个 AIV 自己的 UB** 占用（`kNumStagesPerAIV` 个 stage + 算子附加区），并用 `epilogue.hpp:28-35` 的断言和 `heuristics.hpp:185-194` 的 host 侧约束互相印证；
8. 独立画出当天产出：AIC/AIV 分工图 + 一个输出块内 store/drain 的三泳道时序图；有 Ascend 环境时用 `tests/test_bf16_gemm.py normal` 核对 bias/alpha/accumulation 变体的真实行为与配置差异。

**当天产出：AIC/AIV 分工与 Epilogue 流程图**（模板见第六节，参考答案见 3.9 节）。

## 二、预备知识

- **第 5 天**：AIC（矩阵计算）与 AIV（向量/存取）的分工；GM、L1、L0A/B/C、UB 的层次职责。
- **第 16 天**：`UBSizeBytes = 256 * 1024`（`common.hpp:58`）；UB 是 AIV 的本地缓冲——今天会明确"两个 AIV 各有一份 UB"。
- **第 17 天**：`Scheduler` 的 `get_next_block` 是**确定性**遍历——每个核独立重放同一序列（`scheduler.hpp:10-12` 注释 "Each cube core independently walks the global (m_block, n_block) grid"）。
- **第 18 天**：`gm_ptr` / `ub_ptr` / `result_ptr` 的地址表达；`index()`、`stride_outer`。
- **第 22 天**：K 维两层循环；`s_epi` 在交付后推进（`bf16_gemm.hpp:164`）。
- **第 23 天**：`asc_sync_notify(src, dst, e)` / `asc_sync_wait(src, dst, e)` 的方向约定（dst 阻塞等 src 的事件）；"事件号 = 资源编号"。
- **第 24 天**：`mad()` 与 L0C 生命周期；`num_mad_cd_stages`；`aic_store` 的接口契约（当时只读了两行的"接口"，今天展开时序）；`copy_l0c_to_ub` 的 `unit_flag` 默认 `ENABLE_UPDATE`。
- **Host 侧（第 10～12 天）**：`GemmDesc`、`epilogue_class`、JIT 模板实例化——今天会看到 `kWithAccumulation` 和 epilogue 算子类型如何作为模板参数进入内核。
- **C++ 语法**：模板模板参数（`template <typename> class epilogue_operator_t`）、`if constexpr`、CRTP 式继承（`operators.hpp` 的算子族）。能对照读懂即可。

另外说明：`__mix__`、`ASCEND_IS_AIC`、`asc_get_sub_block_id`、`asc_sync_intra_arrive/wait`、`asc_set_atomic_add_*` 都声明在 CANN 工具链的 `c_api/asc_simd.h`（仓库之外）。本文对它们只陈述**从仓库内全部用法可以反推出的契约**，并标注推断与引用的边界，不凭空补充硬件行为。设备级常量 `kNumAIVs = 64`（`common.hpp:52`）与今天反复出现的 **Epilogue 模板参数 `kNumAIVs`（取 1 或 2，`epilogue.hpp:31`）** 同名不同物，注意区分。

## 三、详细讲解

### 3.1 全景：一个 block，三方执行者

先看启动声明（`bf16_gemm.hpp:27`）：

```cpp
__global__ __mix__(1, kDirectStore ? 0 : 2) void bf16_gemm_impl(...)
```

从仓库内所有用法可以读出的契约是：这个内核的每个 block（对应一个 AI Core，`block_idx` 即核号）由 **1 个 cube 子块（AIC）**和 **`kDirectStore ? 0 : 2` 个 vector 子块（AIV）**混合执行。三方执行**同一份代码**，用两个编译器内置条件区分身份：

- `ASCEND_IS_AIC` 为真 → 执行 `:81` 分支（搬运 + MAD + 交付）；
- 其余（`ASCEND_IS_AIV`）→ 执行 `:173` 的 `else` 分支（收尾 + 写回），AIV 内部再用 `asc_get_sub_block_id()` 区分自己是 0 号还是 1 号（`epilogue.hpp:51`）；
- `kDirectStore` 时没有 AIV 子块，输出由 AIC 直接写 GM（第 26 天展开），`:174` 的 `return` 只是让混合启动下不存在的路径在语义上无害。

于是第 5 天的数据流图今天可以画完整了——注意最后两段的执行者和队列归属：

```text
                AIC（cube）                                    每个 AIV（vector）
GM ──MTE2──> L1 ──MTE1──> L0A/L0B ──M──> L0C ──FIX──> UB[stage] ──V/MTE3──> GM
   (搬运: MTE2/MTE1)      (计算: M)          (交付)     (epilogue 处理)      (写回)
```

把三个执行者的"领地"列成表（行号均属 `bf16_gemm.hpp`）：

| 执行者 | 队列资源 | 职责 | 关键代码 |
|---|---|---|---|
| AIC | MTE2、MTE1、M、FIX | 调度遍历；GM→L1、L1→L0 搬运；mmad 累加到 L0C；把最终结果 `copy_l0c_to_ub` 进目标 AIV 的 UB 并发出块内事件 | `:81-172`，交付 `:155-163` |
| AIV 0 / AIV 1 | 各自的 V、MTE3、MTE2；**各自的 UB** | （可选）空组清零；等 AIC 的"已填好"事件；加 bias / 乘 alpha / 转换 / 舍入 / 量化；`copy_ub_to_gm` 写回（accumulation 时为原子加）；发出"已搬完"事件 | `:173-192`，主体在 `epilogue.hpp:49-87` |
| （`kDirectStore`） | —— | 无 AIV：AIC 用 `copy_l0c_to_gm` 直写 GM | `:156-159`，`:174` |

**两个 AIV 各有一份独立的 UB**，这是今天最重要的空间事实，有三处源码证据：其一，`aic_store` 的 `copy_l0c_to_ub(..., dst_subblock_id = aiv_id)`（`epilogue.hpp:45`）——目标子块编号就是"写到哪个 AIV 的 UB"；其二，`ub_ptr` 的 stage 分配只覆盖 `kNumStagesPerAIV` 份（`epilogue.hpp:25-26`，注意不是 `kNumStages` 份——总量断言 `:35` 也只按每 AIV 检查），而 `aic_store` 用的 `stage_idx = s_epi / kNumAIVs ∈ [0, kNumStagesPerAIV)`；其三，`aiv_zero_empty_groups` 用 `block_idx * kNumAIVs + asc_get_sub_block_id()`（`:112`）给两个 AIV 编全局号，说明 sub_block_id 是 AIV 的本地身份。

### 3.2 对比两个分支：AIC"交付"，AIV"收尾"

`Scheduler` 在 `if ASCEND_IS_AIC` **之前**构造（`bf16_gemm.hpp:73-74`）——AIC 和两个 AIV 各自持有一份，各自独立遍历同一个 (m_block, n_block) 序列。第 17 天已经确认 `get_next_block` 是纯函数式的确定性遍历（`scheduler.hpp:10-12`），这是今天整个协议的基石：**AIV 不需要被告知"下一个块是什么"，它重放同一序列即可与 AIC 对齐**。

AIC 分支（`:81-172`）前四天已逐段读透，今天只回顾与 Epilogue 交互的两点：

```cpp
if (k_blk_idx == 0 && !kDirectStore) {
    epilogue.aic_wait_free(s_epi);                                  // :129-131  借：等 AIV 归还 UB stage
}
...  // 该子块的 K 遍历（MAD）
if (k_blk_idx == num_k_blocks - 1) {
    if constexpr (kDirectStore) { ... copy_l0c_to_gm(...); }        // :156-159  AIC 直写（无 AIV）
    else           { epilogue.aic_store(l0c_tile, s_epi, eff_mad_m, eff_mad_n); }  // :161  还给 epilogue
}
s_epi = s_epi == epilogue.kNumStages - 1 ? 0 : s_epi + 1;          // :164  推进到下一个 UB stage
```

注意 `aic_wait_free` 只在 `k_blk_idx == 0` 调用：`s_epi` 在一个子块的整个 K 遍历期间保持不变（`:107` 赋值后直到 `:164` 才推进），借一次就够；且这个 wait 排在 FIX 队列上（`epilogue.hpp:38` 传 `PIPE_FIX`），真正被它挡住的是**之后**发往同一 UB stage 的 FIX 拷贝，M 队列的 mmad 不受影响。

AIV 分支（`:173-192`）只有四件事，结构上比 AIC 分支简单得多：

```cpp
} else {
    if constexpr (kDirectStore) return;                                        // :174  没有 AIV 的路径
    if constexpr (is_k_grouped && !kWithAccumulation)
        epilogue.template aiv_zero_empty_groups<kNumCores, kAlignment>(...);    // :175-176  空组清零
    epilogue.aiv_initialize();                                                  // :177  置“全部 UB stage 空闲”+原子加模式

    uint32_t s_epi = 0;                                                         // :179
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {                // :180  与 AIC 相同的遍历
        ...
        epilogue.aiv_drain(d, m_idx, n_idx, actual_m, actual_n, s_epi);        // :190  逐块收尾
    }
}
```

三点值得停下来咀嚼：

**其一，`aiv_initialize` 兼任"UB 就绪"信号。** 它做两件事（`epilogue.hpp:89-97`）：把本 AIV 的 `kNumStagesPerAIV` 个块内事件**全部置位**（`set_intra_blocks<kPipe, kNumStagesPerAIV>(asc_get_sub_block_id())`，含义"我名下所有 UB stage 现在都空闲"），以及 `kWithAccumulation` 时设置原子加模式（3.6 节）。AIC 的第一次 `aic_wait_free` 等的就是这些标志——因此**AIC 第一次往某个 AIV 的 UB 里写数据，必然发生在该 AIV 完成初始化（包括空组清零）之后**。空组清零 `aiv_zero_empty_groups`（`epilogue.hpp:108-141`）会临时占用 UB 的 stage 0 放零值再广播写 GM，它结束处的 `MTE3→V` 握手（`:138-139`，注释 "Release the shared UB only after all zero stores have consumed it"）保证所有 MTE3 零值写都已消费完 UB，之后才轮到 `aiv_initialize` 置位放行。

**其二，AIV 的 `s_epi` 与 AIC 的对应物不同但等价。** AIC 用 `epi_base`（`:86/:107/:169`）让 `s_epi` 跨输出块连续轮转；AIV 直接把 `s_epi` 以**引用**传入 `aiv_drain`（`:190`，`epilogue.hpp:50` 的 `uint32_t& s_epi`），在子块循环里同样"每 tile 前进一格、到 `kNumStages` 回绕"（`epilogue.hpp:56`），跨块自然延续。三方从 0 出发、数同样的 tile 总数（相同的 scheduler 序列 × 相同的 m/n 扫描），于是在**每个子块边界**对"`s_epi` 当前指向哪个 UB stage"达成一致——3.7 节专门论证。

**其三，AIV 分支里没有 `m_mad_idx/n_mad_idx` 的 MAD、没有 K 循环**——`aiv_drain` 的双重循环只是**重放 AIC 的子块扫描顺序**（`epilogue.hpp:55-56` 与 `bf16_gemm.hpp:123-124` 同构），用来知道自己该在哪个位置等哪个 tile。AIV 完全不参与 K 维：它看到的只有"每个子块一份最终结果"。

### 3.3 AIC 侧：`aic_store` 把结果倒进目标 AIV 的 UB

`epilogue.hpp:42-47` 全文只有两行实质代码：

```cpp
__aicore__ void aic_store(L0C l0c_tile, uint32_t s_epi, uint32_t m_elems, uint32_t n_elems) const {
    const uint32_t aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs;      // :43  路由：交给哪个 AIV
    const uint32_t stage_idx = s_epi / kNumAIVs;                  // :44  该 AIV 自己的第几格 UB
    copy_l0c_to_ub(ub[stage_idx], l0c_tile, m_elems, n_elems,
                   asc_dual_dst_mode::DUAL_DST_DISABLE, aiv_id);  // :45  FIX 队列：L0C → 目标 AIV 的 UB
    set_intra_block(PIPE_FIX, aiv_id, stage_idx);                // :46  块内 arrive："这格已填好"
}
```

第 24 天只记了接口，今天逐参数展开 `copy_l0c_to_ub`（`copy.hpp:266-285` 的声明）：

| 实参 | 值 | 含义 |
|---|---|---|
| `dst` | `ub[stage_idx]`（`ub_ptr<ub_dtype_t>`，形状 `MAD_M × MAD_N`） | 目标 AIV 的 UB 中第 `stage_idx` 格 |
| `src` | `l0c_tile`（本子块的 `l0c_ptr<float>`） | L0C 里的 FP32 最终结果 |
| `m_elems` / `n_elems` | `eff_mad_m` / `eff_mad_n`（`bf16_gemm.hpp:161`） | 真实行列数（N 已按 `get_frac_k<ub_dtype_t>` 对齐，保证行 32B 对齐，`epilogue.hpp:29`） |
| `dual_dst_ctrl` | `DUAL_DST_DISABLE` | 单目标（不广播到两个 AIV；仓库内全部 GEMM 用例均为 DISABLE） |
| `dst_subblock_id` | `aiv_id` | **写进哪个 AIV 的 UB**——上一节"每 AIV 独立 UB"的落点 |
| `quant_mode` | 默认 `infer_quant_mode<cd_dtype_t, float>()`（`copy.hpp:272`） | **输出转换发生在这里**：BF16 输出即 `F322BF16`（FP32→BF16，`copy.hpp:237-238`），FP32 输出即 `NoQuant`（`:235-236`） |
| `unit_flag` | 默认 `ENABLE_UPDATE`（`copy.hpp:275`） | 第 24 天的"读取侧收尾"，对 L0C 累加器的消费语义 |

于是**输出转换有两个位置**，取决于选中的 epilogue 算子：

- **`Identity`（普通 `D = A@B`）**：算子的 `ub_dtype_t = cd_dtype_t`（`operators.hpp:98`），FIX 拷贝顺手把 FP32 转成 BF16，AIV 收到的已经是最终数据，`process_mad_block` 是空实现，AIV 只做搬运；
- **`ScaleByAlpha` / `AddBias` 等带计算的算子**：`ub_dtype_t = float`（`operators.hpp:128, 164`），FIX 拷贝 `NoQuant` 原样搬运 FP32，转换挪到 AIV 的 V 管道算完之后（`asc_float2bfloat16_rn`，`operators.hpp:151/215`）。

`:46` 的 `set_intra_block(PIPE_FIX, aiv_id, stage_idx)` 是整个协议最精妙的一行：块内 arrive 与第 23 天的 `asc_sync_notify` 一样**排入指定队列、在该队列内按序生效**——它挂在 FIX 队列上、紧跟在 `copy_l0c_to_ub` 之后，所以**当 AIV 观察到这个事件时，UB 里的数据必然已经写完**。跨队列次序（AIC 的 FIX 队列 → AIV 的 V/MTE3 队列）不需要显式事件对，被 arrive 的队列归属免费附送了。

### 3.4 块内同步：UB stage 的四步借还协议

两个分支之间唯一的通信通道是 `sync.hpp:29-40` 的两个薄封装：

```cpp
__aicore__ inline void set_intra_block(pipe_t pipe, uint8_t subblock_id, uint8_t flag_id) {
    asc_sync_intra_arrive(pipe, ((subblock_id & 1) << 4) | (flag_id & 0xf));      // :29-31
}
__aicore__ inline void wait_intra_block(pipe_t pipe, uint8_t subblock_id, uint8_t flag_id) {
    // AIV ignores subblock_id for intra-block waiting.
    if ASCEND_IS_AIV {
        asc_sync_intra_wait(pipe, flag_id & 0xf);                                // :35-36  只用低 4 位
    } else {
        asc_sync_intra_wait(pipe, ((subblock_id & 1) << 4) | (flag_id & 0xf));    // :37-39  完整 5 位编码
    }
}
```

从仓库内全部用法可以反推出以下可见契约：

1. **编码 5 位**：`(subblock_id & 1) << 4 | flag_id & 0xf`——第 4 位选中"AIV 0 还是 AIV 1"这一组事件，低 4 位是组内编号（0～15）。`epilogue.hpp:32` 把 `kNumStagesPerAIV` 限制在 `1..16`，正是因为 flag 只有 4 位（host 侧的镜像约束是 `heuristics.hpp:190` 的 `hw.num_intra_block_events`）。
2. **AIC 侧等待用完整编码**：`aic_wait_free` 要区分"哪个 AIV 归还的哪一格"，所以带 subblock 位。
3. **AIV 侧等待只给 flag 位**（源码注释明说 AIV 忽略 subblock 位）。为什么这样是安全的？考虑反例：若 AIV 侧也按完整编码匹配，AIV1 等待 `(0<<4)|0`（AIC 发给 AIV0 的事件）就可能错拿别人的信号；实际上每个 AIV 只在"路由给自己的 tile"上才调用 wait（`epilogue.hpp:57` 的 `continue` 过滤在前），它等待的 flag 编号必然对应发给自己那一组的事件——因此"每个 AIV 只在**自己那一组** flag 上等待"是从代码必然成立的最小解释，硬件如何自动关联自己的组属于 CANN 指令细节。
4. **arrive 的 `pipe` 参数决定事件何时生效**（排入该队列、按队内次序），**wait 的 `pipe` 参数决定哪条队列被阻塞**——与第 23 天 `asc_sync_wait(src, dst, e)` 阻塞 dst 的模型一致。仓库内有一条直接佐证：`mqa_logits.hpp:420-421` 的注释 "Release accum stage on PIPE_V so it is not serialized behind the GM store"——刻意把归还 arrive 放在 V 队列以避开 MTE3 的长尾，说明**放在哪个队列**是有语义的、可调节的选择。

现在把一个 UB stage 的一生整理成四步闭环（括号内是执行者与队列）：

```text
     AIC（FIX 队列）                                    AIV aiv_id（kPipe 队列 + MTE3 队列）
┌─────────────────────────────────────┐          ┌──────────────────────────────────────────┐
│ ① aic_wait_free(s_epi)               │          │ ④（上一轮）处理完、写回 GM 后：              │
│    wait_intra_block(PIPE_FIX,        │◄─────────│    set_intra_block(PIPE_MTE3,            │
│        aiv_id, stage_idx)            │  事件④    │        aiv_id, stage_idx)   :84          │
│    “这格 UB 我要用了，等它空闲”         │  “已搬完”  │    “这格我消费完了，可复用”                  │
├─────────────────────────────────────┤          ├──────────────────────────────────────────┤
│ ② aic_store：copy_l0c_to_ub(...,      │          │ ③ aiv_drain：                            │
│      dst_subblock_id=aiv_id)         │          │    wait_intra_block(kPipe,                │
│    set_intra_block(PIPE_FIX,         │─────────►│        aiv_id, stage_idx)   :69          │
│        aiv_id, stage_idx)   :46      │  事件②   │    “这格有新数据了”                        │
│    “已填好”（排在拷贝后，队内保序）      │  “已填好”  │    process → copy_ub_to_gm → postprocess  │
└─────────────────────────────────────┘          └──────────────────────────────────────────┘
```

- **事件②**（FIX 队列，`aic_store`）：AIC 交付。因为 arrive 在 FIX 队列内排在拷贝之后，AIV 收到时数据必已就绪。
- **事件④**（MTE3 队列，`aiv_drain:84`）：AIV 归还。因为 arrive 在 MTE3 队列内排在 `copy_ub_to_gm`（和 postprocess 的写回）之后，AIC 收到时这格 UB 真正空闲、GM 数据也已落盘——AIC 可以放心复用。
- **AIV 侧的等待（③）排在算子的工作队列 `kPipe` 上**：`Identity` 的 `kPipe = PIPE_MTE3`（`operators.hpp:100`），等待直接挡住后续的 UB→GM 搬运；带计算的算子 `kPipe = PIPE_V`（如 `operators.hpp:129`），等待挡住后续的向量计算。两种情况都精确匹配"谁消费 UB 谁等待"。
- **AIC 侧的等待（①）只在 `k_blk_idx == 0` 执行**（`bf16_gemm.hpp:129`）：它阻塞的是 FIX 队列，而一个子块内 FIX 只在最后 k_blk 动作一次，开头借一次就覆盖了整个 K 遍历。

这套协议与第 23 天的 L1/L0 借还是同一个"每份缓冲配一对事件"的思想，区别有二：其一，L1/L0 的事件在同一核的队列之间（`asc_sync_notify/wait`），UB 的事件在 **AIC 与 AIV 两个核之间**（`asc_sync_intra_*`）；其二，flag 编号要带上"哪个 AIV"这一位。

**顺带接住第 23/24 天的遗留问题**："M 队列写 L0C 与 FIX 队列读 L0C 之间的跨队列次序"。今天把边界描得更清楚了：UB 这一段有完整的四步闭环保护；而 L0C 仍然没有——`aic_store` 的 FIX 拷贝读 `l0c[s_cd]`，与下一个输出块同位置子块首条 mmad 的 `clear_C` 写（M 队列）之间，仓库内没有任何事件对。源码可见的线索有两条：mmad 末步与 L0C 读取两端的 `ENABLE_UPDATE` "提交/消费"配对（第 24 天的观察），以及 stage 环的天然深度（`s_cd` 固定映射 + `s_epi` 跨块续接，下一块同位置子块的 `clear_C` 之间隔着一整块的 M 队列工作量，且 AIC 若在 `aic_wait_free` 上受阻，FIX 队列的积压也会推迟下一块对 UB 的复用）。但这两条都不是正确性证明——**指令级语义在 `asc_mmad` / `asc_copy_l0c2ub` 的 CANN 文档里**，本仓库把它当作硬件契约使用。把这个结论记入笔记，"遗留问题"从"没看懂"升级为"已定位边界"。

### 3.5 AIV 侧：`aiv_drain` 消费一个输出块

`epilogue.hpp:49-87` 是 AIV 分支的全部实质工作。逐段走读：

**进入（`:51-54`）——先准备，后等待。**

```cpp
const uint32_t aiv_id = asc_get_sub_block_id();          // :51  我是 AIV 0 还是 1
const uint32_t drain_stage_idx = s_epi / kNumAIVs;       // :52  本块的 drain 起点
// Prepare the whole drain before waiting for its first MAD tile.
op.initialize_aiv_drain(args, drain_stage_idx, n_idx, actual_n);   // :54
```

注释点明了意图：**准备工作（如 AddBias 预取 bias 行）在第一次 `wait` 之前发出**，让 GM→UB 的搬运与"AIC 还在算"的时间重叠。`AddBias::initialize_aiv_drain`（`operators.hpp:175-183`）先用一对 `V↔MTE2` 事件确认 bias 缓冲可覆盖，再 `copy_gm_to_ub` 把本块覆盖的 bias 列（`n_idx` 起 `actual_n` 列，一行 `[1, N]`）预取到 `bias_ub[drain_stage_idx]`，再用第二对事件让 V 管道知道加载已完成。`drain_stage_idx` 随块轮转，天然形成 bias 的双缓冲（host 侧把这一收益写进了代价模型，`heuristics.hpp:480-483` 注释 "AddBias owns one BLOCK_N row per AIV drain; its hooks preserve the existing overlap"）。

**双重循环与路由过滤（`:55-58`）——重放 AIC 的扫描顺序。**

```cpp
for (uint32_t m_mad = 0; m_mad < actual_m; m_mad += MAD_M) {
    for (uint32_t n_mad = 0; n_mad < actual_n; n_mad += MAD_N,
             s_epi = s_epi == kNumStages - 1 ? 0 : s_epi + 1) {
        if (aiv_id != 2 - kNumAIVs + s_epi % kNumAIVs) continue;   // :57  不归我的 tile，跳过
        const uint32_t stage_idx = s_epi / kNumAIVs;                 // :58
```

循环结构与 `bf16_gemm.hpp:123-124` 的 m/n 子块扫描逐字同构，`s_epi` 的推进公式也与 AIC 的 `:164` 相同——**即使 tile 不归自己，`s_epi` 照样前进一格**（`continue` 只跳过循环体，不跳过递增子句），这是三方锁步的关键实现细节。`:57` 的路由判断与 `aic_store` 的 `:43` 用同一算式：`kNumAIVs = 2` 时即 `s_epi % 2`，偶数 tile 归 AIV0、奇数归 AIV1——`epilogue.hpp:23` 注释称之为 "Assign complete MAD tiles round-robin across available AIVs"。

**等待与原地处理（`:59-74`）。**

```cpp
const epilogue::operators::MadBlock block{
    .stage_idx = stage_idx, .drain_stage_idx = drain_stage_idx,
    .m_idx = m_idx + m_mad, .n_idx = n_idx + n_mad, .n_mad = n_mad,
    .num_rows = min(actual_m - m_mad, MAD_M), .num_cols = min(actual_n - n_mad, MAD_N),
    .batch_idx = batch_idx };                                  // :59-68  给算子的“任务单”
wait_intra_block(kPipe, aiv_id, stage_idx);                    // :69  等 AIC 的“已填好”

const auto ub_stage = ub[stage_idx];                            // :71  输入视图：ub_dtype_t（如 float）
const ub_ptr<cd_dtype_t> output_ub(MAD_M, MAD_N, ub_stage.addr); // :72  输出视图：同一地址！
op.process_mad_block(ub_stage.ptr(), output_ub.ptr(), args, block);  // :74  原地计算
```

`:71-72` 是**原地（in-place）处理**：输入视图（FP32）和输出视图（BF16）指向同一块 UB。这解释了 `epilogue.hpp:28` 的断言 `"In-place epilogue output must fit in its input UB"`——`sizeof(ub_dtype_t) >= sizeof(cd_dtype_t)`，BF16（2B）输出写在原 FP32（4B）缓冲的前半截，放得下。`MadBlock`（`operators.hpp:86-90`）是算子拿到的全部上下文：写到哪个 GM 偏移（`m_idx/n_idx`）、bias 从哪列开始（`n_mad`）、边界行列数（`num_rows/num_cols`）、批内偏移（`batch_idx`）。

**写回与归还（`:76-84`）。**

```cpp
if constexpr (kPipe != PIPE_MTE3) {
    const auto store_event = static_cast<event_t>(stage_idx % kNumPipeEvents);  // :77
    asc_sync_notify(kPipe, PIPE_MTE3, store_event);    // :78  V 宣布“本格算完了”
    asc_sync_wait(kPipe, PIPE_MTE3, store_event);      // :79  MTE3 阻塞等 V（下一行搬运必须在 V 之后）
}
copy_ub_to_gm(d.index(block.m_idx, block.n_idx).ptr(), output_ub, block.num_rows, block.num_cols,
              d.stride_outer * sizeof(cd_dtype_t), kL2CacheMode);   // :81-82  UB → GM
op.postprocess_mad_block(args, block);                               // :83  尾随写回（如 FP8 的 SF）
set_intra_block(PIPE_MTE3, aiv_id, stage_idx);                       // :84  归还：“这格可复用”
```

三点解读。**其一**，`Identity` 的 `kPipe = PIPE_MTE3`，等待与搬运同队列、天然保序，握手对被 `if constexpr` 裁掉；带计算的算子 `kPipe = PIPE_V`，V 写完的输出要由 MTE3 搬走，因此 `:78-79` 用一对 `V→MTE3` 事件补上跨队列次序——事件号取 `stage_idx % kNumPipeEvents`（`kNumPipeEvents = 8`，`common.hpp:61`），不同 stage 的搬运互不阻塞，仍是第 23 天"事件号 = 资源编号"。**其二**，`copy_ub_to_gm` 用 `MadBlock` 里的**真值**行列数（`num_rows/num_cols`），只写有效区域，UB 里 `eff_mad_n` 对齐出来的填充列不会写进 GM。**其三**，`:84` 的归还 arrive 挂在 MTE3 上、排在 `:81-82`（以及 postprocess 里可能存在的第二次写回，如 `QuantizeToFP8` 的 SF 写 GM，`operators.hpp:357-362`）之后——所以 AIC 的 `aic_wait_free` 通过时，这格 UB 的一切工作都真正结束了。对比 `mqa_logits.hpp:420` 刻意把归还提前到 V 队列（不等待 GM store 长尾），可以体会**归还位置是协议的调优旋钮**：提前归还提高并行度，代价是"归还 ≠ GM 落盘"，需要调用方自行保证不再踩踏。

最后是 `aiv_zero_empty_groups`（`epilogue.hpp:108-141`）一笔带过：K-grouped GEMM 中空组（`grouped_layout` 前缀和相等，`:119-121`）没有任何 AIC 工作量，其输出区间只能由 AIV 显式写零（`:111` 断言：不允许与 accumulation 共存、不允许 FP8 输出）。两个 AIV 按 `aiv_idx = block_idx * kNumAIVs + sub_block_id` 全局编号、以 `kNumCores * kNumAIVs` 为步长分摊空组的所有 MAD 粒度 tile（`:129-134`）。细节属于第 37/38 天，今天只需知道"它是 AIV 分支的守门职责之一，且排在 `aiv_initialize` 之前、靠初始化标志间接与 AIC 的首次 `aic_store` 串行化"。

### 3.6 accumulation、bias、alpha、输出转换：各在哪一层？

现在把 host 的语义选择和 device 的执行位置接成一张总表。**host 侧入口**（`csrc/apis/gemm.hpp:106-123`）：`c` 与 `d` 同一张量（别名）→ accumulation；`c` 是独立的 `[1, N]` 连续张量 → bias；两者互斥地决定 `desc.acc`（`:180`）与 `epilogue_class`；`alpha` 是独立参数，经 `resolve_epilogue_class`（`csrc/jit_kernels/epilogue_class.hpp:110-128`）三选一（`epilogue_class` / `alpha` / `sfd` 互斥断言 `:117-118`）。

| 语义 | Python 输入形态 | host 判定 / 类 | device 上的位置（核 · 队列） |
|---|---|---|---|
| `D = A@B` | 不传 `c`、不传 `alpha` | `IdentityEpilogue` → `epilogue::operators::Identity` | 无额外计算；BF16 的 FP32→BF16 转换在 **AIC 的 FIX 拷贝**（`aic_store` → `quant_mode = F322BF16`） |
| `D += A@B`（accumulation） | `c` 与 `d` 为同一张量 `[M, N]` | `desc.acc = true`（`gemm.hpp:180`） | **GM 原子加**：普通路径由 AIV 在 `aiv_initialize` 设置 `asc_set_atomic_add_bfloat/float`（`epilogue.hpp:91-96`），其后的 `copy_ub_to_gm`（`:81`）变成原子加；`kDirectStore` 路径由 AIC 设置（`bf16_gemm.hpp:82`），作用于 `copy_l0c_to_gm` |
| `D = A@B + bias` | `c` 为独立 `[1, N]` 连续张量 | `BiasEpilogue` → `AddBias`（`gemm.hpp:122-123`） | **AIV 的 V 管道**：`initialize_aiv_drain` 预取 bias 行 GM→UB（`operators.hpp:175-183`），`process_mad_block` 逐行 `asc_add` 后转换写回（`:185-221`） |
| `D = alpha·A@B` | `alpha=` 标量 | `AlphaEpilogue` → `ScaleByAlpha` | **AIV 的 V 管道**：`asc_mul_scalar` 乘 alpha，再 `asc_float2bfloat16_rn` 转换（`operators.hpp:131-157`） |
| 随机舍入 BF16 输出 | `epilogue=deep_gemm.epilogue.BF16StochasticRounding()` | `BF16StochasticRoundingEpilogue` → `StochasticRoundToBF16` | **AIV 的 V 管道**：对 FP32 值做哈希驱动的随机舍入再打包 BF16（`operators.hpp:235-334`；验证方法第 30 天展开） |
| FP8 量化输出 | `d` 为 E4M3 且配对 `sfd` | `FP8QuantizationEpilogue` → `QuantizeToFP8` | **AIV 的 V 管道**：BF16 中间值按 64 元素组求 amax 生成 SF、量化为 E4M3（`:353-446`），`postprocess_mad_block` 把 SF 写回 GM（`:357-362`） |
| FP32→BF16 输出转换 | `d` 的 dtype 决定 | —— | `Identity`：**AIC 的 FIX 拷贝**内完成；带 V 算子：FIX 拷贝 `NoQuant` 原样搬 FP32，转换在 **AIV 的 V 管道**算完后进行（`operators.hpp:151, 215`） |

README 对前两行的官方表述（`README.md:58-60`）："Full C uses atomic add. A contiguous `[1, N]` bias is loaded and added in the **AIV epilogue** before the output cast."——与我们从源码读出的完全一致。

**被禁止的组合及其依据**：

- **bias × alpha**：`gemm.hpp:115-116` 断言 "alpha and bias cannot be used together"（bias 分支里显式检查 `AlphaEpilogue`）；`resolve_epilogue_class` 的互斥断言（`epilogue_class.hpp:117-118`）在更上层也拦一道。测试里专门验证了这条报错路径（`tests/common.py:191-197`）。
- **bias × accumulation**：`c` 是 `[1, N]` 时 `acc` 必为 false（`gemm.hpp:180` 的 `!with_bias`）；device 侧再拦一道——`AddBias::is_supported()` 要求 `!P::kWithAccumulation`（`operators.hpp:171-173`），编译期由 `epilogue.hpp:34` 触发。
- **alpha × accumulation 是允许的**：`ScaleByAlpha` 未重写 `is_supported`，语义为 `D += alpha·(A@B)`——原子加发生在 GM store，alpha 发生在 store 之前，两者不冲突。

**每个 AIV 的 UB 布局**（`epilogue.hpp:25-26` 与各算子的 `ub_end()`）：

```text
一个 AIV 的 UB（≤ 256KB，:35 断言）：
┌───────────────────────────────────────────────┐
│ stage 0: MAD_M × MAD_N × sizeof(ub_dtype_t)    │  kNumStagesPerAIV 格，
│ stage 1:            …                         │  由 aic_store 写入、
│   …                                            │  aiv_drain 消费
│ stage kNumStagesPerAIV-1                       │
├───────────────────────────────────────────────┤ ← op 的基址 = ub.offset(kNumStagesPerAIV)
│ 算子附加区（ub_end() 之前）：                    │
│  AddBias:  kNumStagesPerAIV 份 bias 行          │  bias_ub{1, BLOCK_N}（:167, :169）
│  QuantizeToFP8: kNumStagesPerAIV 份 SF 缓冲     │  sf_ub{MAD_N/64, MAD_M}（:345, :347）
│  Identity / ScaleByAlpha / 随机舍入: 无          │  ub_end() == ub_base
└───────────────────────────────────────────────┘
```

host 的 `heuristics.hpp:185-194` 用同一套公式选 `num_epilogue_stages`：`epilogue_bytes = mad_m×mad_n×epilogue_elem_bits/8 +（bias ? block_n×cd 元素大小 : 0）+（fp8 输出 ? mad_m×mad_n/32 : 0）`（`:188-189`），上限 `min(16 个块内事件, UBSizeBytes / epilogue_bytes)`，再 clamp 到每核块数 × CD stage 数（`:193`），并否决 `num_epilogue_stages × num_aivs < num_cd_stages` 的候选（`:194`，与 `epilogue.hpp:33` 的断言互为镜像）。其中 `epilogue_elem_bits`（`:81-82`）只有"BF16 输出且 Identity"才是 16，其余（FP32 输出、带算子、FP8 输出）是 32——正是"Identity 才让 FIX 拷贝直接转 BF16"的 host 侧投影。

### 3.7 三方锁步：没有通信的 `s_epi` 一致性

AIC 与两个 AIV 之间**只传 UB 数据和两个 arrive 事件**，`s_epi` 从不同步。三方却要在每个子块边界对"当前 tile 用哪格 UB"达成一致，靠两条机制：

**机制一：相同的确定性序列。** 三方从各自 `Scheduler` 的同一起点出发，`get_next_block` 产出相同的 (m_block, n_block) 序列（第 17 天的确定性）；在每个输出块内，AIC 的 `m_mad_idx/n_mad_idx` 双循环（`bf16_gemm.hpp:123-124`）与 AIV 的 `m_mad/n_mad` 双循环（`epilogue.hpp:55-56`）同构，边界条件同为 `actual_m/actual_n`——于是三方数出的 **tile 总数逐块一致**。

**机制二：相同的推进规则。** 三方都以"每见一个 tile 前进一格、到 `kNumStages - 1` 回绕 0"推进 `s_epi`：AIC 在 `:164`（交付后），AIV 在 `epilogue.hpp:56` 的递增子句（包括被 `continue` 跳过的 tile）。AIC 的跨块续接用 `epi_base`（`bf16_gemm.hpp:86/:107/:169`），AIV 靠引用传参让 `s_epi` 自然延续——表达方式不同，计数完全相同。

推论：**路由正确性完全依赖两条纯算式**（`epilogue.hpp:43-44`）——AIC 用 `aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs` 选目标、`stage_idx = s_epi / kNumAIVs` 选格；AIV 用同一算式判断"这个 tile 是不是我的"（`:57-58`）。只要锁步成立，"AIC 写的格"与"AIV 等的格"就是同一格。反过来，这也解释了 `epilogue.hpp:33` 为什么把 `kNumStages ≥ num_mad_cd_stages` 定为**硬断言**：`kNumStages` 是三方共享的环长，小于 CD stage 数时环会在一个块内先回绕，协议虽不至于写错格（wait/arrive 仍然配对），但 `s_epi` 到子块的映射失去固定性，仓库选择在编译期直接拒绝这种退化配置。

顺带一个真实用例：`fp8_dequant_gemm.hpp:169-171` 实例化的是 `Epilogue<..., kNumEpilogueStages, 1, ...>`——**kNumAIVs = 1**。此时 `2 - kNumAIVs = 1`，全部 tile 路由给 1 号 AIV（0 号 AIV 忙于 FP4 反量化，`kDequantAIVId = 0`，`:10`）；BF16 与 FP8 主线（`bf16_gemm.hpp:54`、`fp8_gemm.hpp:75`）都是 2。host 侧 `heuristics.hpp:186` 的 `num_aivs = dtype.dequant_b ? 1 : 2` 与之对应。

### 3.8 手工推演示范：把第 24 天的 2×2 例子补上 AIV 时序

**设定**（沿用第 24 天 3.6）：Normal GEMM，单输出块，M=N=128, K=256；`BLOCK 128×128×128`、`MAD 64×64×64`、`kNumL1Stages=8`、`kNumL0Stages=2`、`kNumEpilogueStages=2`（→ `epilogue.kNumStages = 2×2 = 4`）、`kNumAIVs=2`，`Identity` + BF16 输出。先验算约束：

```text
epilogue.hpp:32  kNumStagesPerAIV=2 ∈ [1,16] ✔
epilogue.hpp:33  kNumStages 4 ≥ num_mad_cd_stages 4 ✔
epilogue.hpp:29  MAD_N 64 % get_frac_k<bfloat16_t>()=16 == 0 ✔
epilogue.hpp:35  每 AIV UB：2 × 64×64×2B = 16KB，Identity 无附加区 ≤ 256KB ✔
heuristics 镜像：epilogue_bytes = 64×64×16/8 = 8KB → num_epilogue_stages ≤ min(16, 256KB/8KB)=16，再 clamp
```

第 24 天已经数出：全块 4 个子块、每个子块 4 条 mmad、最后一个 k_blk 内 4 次交付。今天补全三方时序（`epi_base = 0`；事件记法 `⌈FIX·a·s⌉` = 在 FIX 队列上向 AIV-a 的 stage-s 发出 arrive）：

```text
AIC（M 队列算 MAD，FIX 队列交付）
  │ 子块(m0,n0)：k_blk=0 先 wait_free(⌈a=0,s=0⌉)——首块，AIV 的 aiv_initialize 已预置，直接通过
  │             …4 条 mmad（K 遍历，第 24 天的表）…
  │ 子块(m0,n0)：k_blk=1 末尾 aic_store(s_epi=0)：copy_l0c_to_ub→AIV0 的 UB[0]，⌈FIX·0·0⌉
  │ 子块(m0,n1)：wait_free(⌈a=1,s=0⌉)（已预置，通过）；…mmad…；aic_store(s_epi=1)：→AIV1 UB[0]，⌈FIX·1·0⌉
  │ 子块(m1,n0)：wait_free(⌈a=0,s=1⌉)；…mmad…；aic_store(s_epi=2)：→AIV0 UB[1]，⌈FIX·0·1⌉
  │ 子块(m1,n1)：wait_free(⌈a=1,s=1⌉)；…mmad…；aic_store(s_epi=3)：→AIV1 UB[1]，⌈FIX·1·1⌉
  │ 块结束：epi_base = s_epi = 0（环 4 恰好转满一圈）
  ▼
AIV0（处理 s_epi = 0, 2）
  │ drain：s_epi 起点 0 → drain_stage_idx = 0；Identity 无需预取
  │ tile(s_epi=0, m0n0)：wait(kPipe=MTE3, ⌈a=0,s=0⌉) → 无计算（Identity，FIX 已转 BF16）
  │                     → copy_ub_to_gm(D[0:64, 0:64]) → ⌈MTE3·0·0⌉（归还）
  │ tile(s_epi=1, m0n1)：s_epi 推进到 1，aiv_id≠0 → continue（s_epi 照常推进到 2）
  │ tile(s_epi=2, m1n0)：wait(⌈a=0,s=1⌉) → copy_ub_to_gm(D[64:128, 0:64]) → ⌈MTE3·0·1⌉
  ▼
AIV1（处理 s_epi = 1, 3）
  │ tile(s_epi=0)：continue；tile(s_epi=1, m0n1)：wait(⌈a=1,s=0⌉) → D[0:64,64:128] → ⌈MTE3·1·0⌉
  │ tile(s_epi=3, m1n1)：wait(⌈a=1,s=1⌉) → D[64:128,64:128] → ⌈MTE3·1·1⌉
```

几个推演时容易看漏的细节：

1. **AIC 的 4 次 `wait_free` 都发生在 k_blk=0**，而 4 次 `aic_store` 都发生在 k_blk=1——wait_free 挡住的是 FIX 队列，本块内它总是先于同 stage 的拷贝入队，顺序天然正确；它真正等到 AIV 归还的场景出现在**下一个输出块**（`epi_base=0`，下一块的 s_epi 又从 0 开始，此时 AIV0 的 `⌈MTE3·0·0⌉` 若未发生，AIC 就阻塞在 wait_free 上）。
2. **AIV 的两个 tile 使用不同的 UB stage**（s=0 和 s=1），AIC 的第二次写入必须等第一次的归还——stage 双缓冲让 AIC 的下一块交付与 AIV 的上一块写回可以重叠；若只有 1 格（`kNumStagesPerAIV=1`），`epilogue.hpp:33` 要求 4 ≤ 2 不成立，编译期直接拒绝。
3. **`Identity` 路径 AIC 的 FIX 拷贝已经在做 FP32→BF16 转换**，AIV 的"处理"是零长度的，`kPipe = PIPE_MTE3` 使 wait 直接挂在搬运队列上；换成 `ScaleByAlpha`，同样的时序里 AIV 的 wait 挂在 V 队列，且 `:76-80` 会多出一对 `V→MTE3` 事件，每格 UB 也变成 64×64×4B=16KB。
4. **三方在块边界后 `s_epi` 都是 0**——因为 4 个 tile 恰好转满 `kNumStages=4` 的环。若 `kNumEpilogueStages=4`（`kNumStages=8`），块尾 `s_epi=4`，下一块从 stage 4 起步，`epi_base` 实现的错位复用让相邻块不背靠背踩同一格 UB。

### 3.9 参考流程图（当天产出的参考答案）

**图一：AIC/AIV 分工与 Epilogue 全流程**（把当天所有要素收进一张图）：

```text
                        ┌─────────────────────────── Host（JIT 前） ───────────────────────────┐
                        │ c 与 d 同张量→acc=true      c=[1,N]独立→BiasEpilogue  alpha=→Alpha     │
                        │ epilogue_class.hpp:110-128 三选一 → 算子类型 + Args 进内核模板参数        │
                        └──────────────────────────────────────────────────────────────────┘
                                          ↓ launch（__mix__(1, 2)，一个 block 三方执行）
┌─────────────────────── AIC（if ASCEND_IS_AIC, bf16_gemm.hpp:81-172）───────────────────────┐
│ Scheduler.get_next_block ──► GM ─MTE2→ L1 ─MTE1→ L0A/B ─M→ L0C（K 遍历 + mmad，第 22~24 天）│
│   每个子块：k_blk=0 时 aic_wait_free(s_epi)          [借]  FIX 队列等 AIV 的归还事件           │
│   最后 k_blk 时 aic_store：                                                            │
│     aiv_id = s_epi % 2（kNumAIVs=2）；stage_idx = s_epi / 2                              │
│     copy_l0c_to_ub(ub[stage], l0c, m, n, DUAL_DST_DISABLE, aiv_id)  [FIX，含 F322BF16]     │
│     set_intra_block(PIPE_FIX, aiv_id, stage_idx)                    [还→借，arrive 随队保序]│
│   s_epi++（% kNumStages）；块间 epi_base 续接（:169）                                     │
└────────────────────────────────────┬────────────────────────────────────────────────────┘
                        块内事件②“已填好”（FIX 队列内生效）
┌────────────────────────────────────▼────────────────────────────────────────────────────┐
│ AIV 0 / AIV 1（else 分支, :173-192）                                                     │
│  （可选）aiv_zero_empty_groups：K-grouped 空组写零（:175-176）                            │
│  aiv_initialize：置全部块内事件“空闲”（初始借权）+ acc 时设原子加（epilogue.hpp:89-97）      │
│  与 AIC 相同的 Scheduler 遍历 + 同构的 m/n 子块扫描，s_epi 引用传参、每 tile 推进一格        │
│  aiv_drain（epilogue.hpp:49-87）：                                                       │
│    路由过滤 aiv_id != s_epi % 2 → continue（s_epi 照常推进）                               │
│    initialize_aiv_drain：等待前预取（AddBias 的 bias 行 GM→UB，双缓冲随块轮转）              │
│    wait_intra_block(kPipe, self, stage)                              [借]              │
│    process_mad_block：原地处理（V 管道：+bias / ×alpha / 随机舍入 / FP8 量化）              │
│    [kPipe≠MTE3：notify/wait(V→MTE3, stage % 8)——V 完成后搬运才放行]                          │
│    copy_ub_to_gm(D[子块区域]，真值行列数，L2 模板参数；acc 时为原子加)     [MTE3]            │
│    postprocess_mad_block（FP8：SF 写回 GM）                                                │
│    set_intra_block(PIPE_MTE3, self, stage)                           [还]              │
└────────────────────────────────────┬────────────────────────────────────────────────────┘
                        块内事件④“已搬完”（MTE3 队列内生效）→ AIC 的下一次 aic_wait_free 放行
```

**图二：单个 UB stage 的借还闭环**——3.4 节的四步图，此处不重复；**图三：3.8 的三泳道时序图**即单块推演的产出。

画图时的自查清单（五处最易遗漏）：`aic_wait_free` 只在 `k_blk_idx == 0`；`set_intra_block` 的两个队列归属（FIX / MTE3）及其"随队保序"含义；AIV 的 `continue` 不阻止 `s_epi` 推进；`initialize_aiv_drain` 在 `wait` **之前**；`kPipe != PIPE_MTE3` 时多出的 `V→MTE3` 握手对。

## 四、源码导读

建议按站点顺序阅读，总计约 45 分钟。所有行号以仓库当前版本为准。

### 站点 1：`deep_gemm/include/deep_gemm/epilogue.hpp`（全文 144 行，精读，今天的主战场）

按 3.3～3.7 的顺序重读：`:7-26` 模板参数与 UB 布局（`params_t`、`ub_dtype_t`/`kPipe`、`ub` + `op` 基址）；`:28-35` 六条断言逐条说出动机；`:37-47` `aic_wait_free` / `aic_store`（每行都已在 3.3/3.4 展开）；`:49-87` `aiv_drain` 逐行核销 3.5 的走读；`:89-97` `aiv_initialize`；`:99-141` 空组清零（浏览，注意首尾两对 `V↔MTE3` 握手各自保护什么）。

### 站点 2：两个分支的骨架（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:27-35, 51-57, 81-192`，精读）

`:27` 的 `__mix__` 声明；`:53-55` Epilogue 实例化（`kNumEpilogueStages` 与 `kNumAIVs=2` 两个实参）；AIC 分支只重读 `:129-131`、`:155-169`（借、交付、推进、续接四个点）；AIV 分支通读——体会"结构比 AIC 分支简单一个量级"的原因。

### 站点 3：块内同步原语（`deep_gemm/include/deep_gemm/ascend/sync.hpp:29-40, 75-85`，精读）

`set_intra_block` / `wait_intra_block` 的编码与 AIV 忽略 subblock 的注释；`set_intra_blocks`（复数版，`aiv_initialize` 用它一次性置位全部 stage）。

### 站点 4：epilogue 算子族（`deep_gemm/include/deep_gemm/epilogue/operators.hpp:57-222` 精读，`:224-447` 浏览）

`Params` / `MadBlock` / `Operator` 基类（`:92-117`：`ub_dtype_t`、`kPipe`、三个 hook 的空默认实现、`is_supported`、`ub_end`——注释 "Which type of data the epilogue operator want from MAD output" / "The epilogue acquires the MAD tile on this pipe"）；`Identity`（`:119-122`）、`ScaleByAlpha`（`:124-158`）、`AddBias`（`:160-222`，注意 `bias_ub` 布局、`ub_end`、两组 `V↔MTE2` 握手、逐行 unpack/add/pack）。`StochasticRoundToBF16` 与 `QuantizeToFP8` 今天只看 `is_supported` 的约束和 `postprocess_mad_block` 的"第二次写回"，内部算法分别留给第 30/32 天。

### 站点 5：L0C→UB / UB→GM 搬运（`deep_gemm/include/deep_gemm/ascend/copy.hpp:231-317`，精读）

`infer_quant_mode`（`:231-264`，`NoQuant` 与 `F322BF16` 两个分支）；`copy_l0c_to_ub`（`:266-285`，重点 `dst_subblock_id`、`quant_mode`、`unit_flag` 三个默认值）；`copy_ub_to_gm` 的 `ub_ptr` 重载（`:308-317`）。

### 站点 6：交叉佐证（浏览，建立模式识别）

- `deep_gemm/include/deep_gemm/mqa_logits.hpp:392-404`：单发 mmad 之后 `wait_intra_block(PIPE_FIX,...)` → `copy_l0c_to_ub` → `set_intra_block(PIPE_FIX,...)` 的微缩版协议；`:416-425` 的 "Release accum stage on PIPE_V so it is not serialized behind the GM store"——归还位置可调的实证；
- `deep_gemm/include/deep_gemm/fp8_gemm.hpp:74-76`：与 BF16 同款的 `Epilogue<..., 2>`；`fp8_dequant_gemm.hpp:169-171`：`kNumAIVs=1` 的真实用例（`:10` `kDequantAIVId = 0`——AIV0 忙于反量化，epilogue 全归 AIV1）；
- `deep_gemm/include/deep_gemm/mega_moe.hpp:612-636`：另一套手写的 UB 借还（`kEpilogueEmpty/kEpilogueFull`，`wait_intra_block<PIPE_FIX, 1>` / `notify_intra_block<PIPE_FIX, 1>`）——同一协议在不同算子里的变体写法。

### 站点 7：host 侧的选择与约束（`csrc/jit_kernels/epilogue_class.hpp` 精读，`csrc/apis/gemm.hpp:99-123, 180` 与 `csrc/jit_kernels/heuristics.hpp:75-90, 185-194, 474-493` 浏览）

`resolve_epilogue_class` 的三选一；`gemm.hpp` 中 `with_bias` / `acc` 的判定；`heuristics.hpp` 中 `epilogue_elem_bits`、`num_epilogue_stages` 的计算与 `get_gemm_output_cost`（`fix_ns / bias_ns / gm_ns / vector_ns` 四段代价——为什么 bias 的预取与双缓冲值得 host 精心建模）；`csrc/jit_kernels/gemm_config.hpp:106` 的注释 "UB epilogue stages per AIV (AIC -> AIV)"。

### 站点 8：测试如何驱动这些语义（`tests/common.py:49-84, 154-168, 186-221`，浏览）

`gemm_dg` 中 `with_bias` / `accumulate` 两种 `c` 的传法（`:53-58`：accumulation 时 `d.copy_(c)` 后把 `d` 自己当 `c` 传入——印证"c 与 d 同张量"的约定）；Normal 模式自动生成的 bias/alpha/acc/stochastic 变体（`:154-174`）与 alpha×bias 的拒绝测试（`:191-197`）。

## 五、实践步骤

### A. 无 Ascend 环境：纸面推演（主任务，约 55 分钟）

**A-1 分工对比表（必做，当天产出的核心之一）**。合上本文，凭源码印象填出下表并标注行号，写完对照 3.1/3.2 订正：

| 项目 | AIC 分支 | AIV 分支 |
|---|---|---|
| 入口条件 | | |
| 持有的队列 | | |
| 是否遍历 Scheduler | | |
| K 维参与 | | |
| 对 UB stage 做什么 | | |
| 发出的块内事件（队列、方向） | | |
| 等待的块内事件 | | |
| `s_epi` 的推进与跨块续接 | | |

**A-2 三泳道时序图（必做，当天产出的核心之二）**。用 3.8 的设定（M=N=128, K=256，BLOCK 128³，MAD 64³，`kNumEpilogueStages=2`），**先不看参考答案**，画出 AIC / AIV0 / AIV1 三条泳道：4 个子块的 wait_free 与 aic_store 位置、每个 arrive/wait 事件（标队列）、AIV 两次 `copy_ub_to_gm` 的目标区域、块尾三方的 `s_epi` 值。然后回答：若第二个输出块紧随其后，AIC 在 `wait_free(⌈a=0,s=0⌉)` 处会不会阻塞？阻塞的条件是 AIV0 的什么事件尚未发生？

**A-3 变体推演（必做）**。每次只改一个变量，记录差异：

| 变体 | 改动 | 观察重点 |
|---|---|---|
| V1 | `kNumEpilogueStages=1`（其余同 A-2） | `kNumStages=2 < 4` → `epilogue.hpp:33` 编译期报错；从协议角度说明为什么仓库把它定为硬约束（提示：`s_epi` 到子块映射的固定性、AIC 在 wait_free 上的阻塞频率） |
| V2 | 换 `ScaleByAlpha`（FP32 UB）+ BF16 输出 | 每 AIV UB 变为 `2×64×64×4B=32KB`；FIX 拷贝变成 `NoQuant`；AIV 的 wait 挂 V 队列；多出 `V→MTE3` 握手（事件号 `stage % 8`）；host 侧 `epilogue_elem_bits` 从 16 变 32（`heuristics.hpp:81-82`）→ UB 容量上限 `256KB/16KB=16` 恰好与"16 个块内事件"上限相等（`heuristics.hpp:190` 两个 `min` 分量都是 16）——体会 UB 账本翻倍后留给附加区的余量同步减半 |
| V3 | 换 `AddBias`（`BLOCK_N=128`，BF16 输出） | 每 AIV UB = `2×(64×64×4B) + 2×(1×128×2B)` = 32KB + 512B；`drain_stage_idx` 随块轮转形成 bias 双缓冲；`:171-173` 禁止与 accumulation 组合的断言链 |
| V4 | `kNumAIVs=1`（fp8_dequant 场景） | `aiv_id = 2-1+0 = 1` 恒定，全部 tile 归 AIV1；`kNumStages = kNumStagesPerAIV`；host 侧 `heuristics.hpp:186` 的 `num_aivs = dequant_b ? 1 : 2` |
| V5 | accumulation 数值演算 | 设用户先 `d.copy_(c)` 使 D 初值为 X、本块某子块结果为 Y（BF16）：普通路径写出后 D 该区域的值是什么？（原子加：`X + Y`，以 BF16 精度在 GM 中完成）`kDirectStore` 路径由谁做原子加（`:82`）？为什么 `aiv_zero_empty_groups` 不允许 accumulation（`:111`——空组写零会与"保留初值"矛盾） |

**A-4 判定题自验（必做）**。判断真假并给出行号依据（答案见第七节末尾）：

1. `aic_store` 里的块内 arrive 挂在 PIPE_MTE3 上。（  ）
2. AIV 归还 UB stage 的 arrive 排在 `copy_ub_to_gm` 之后，因此 AIC 的 `aic_wait_free` 通过时，该 stage 对应的 GM 区域已经写完。（  ）
3. bias 与 accumulation 可以同时使用（`c=[1,N]` 且 `D += A@B`）。（  ）
4. `kNumStagesPerAIV` 不能超过 16，是因为块内事件的 flag 编号只有 4 位。（  ）
5. 两个 AIV 各自持有独立 UB，`stage_idx` 在各自 UB 内部从 0 计数；AIC 的 `copy_l0c_to_ub` 用 `dst_subblock_id` 选择目标 AIV。（  ）
6. `epilogue.hpp:28` 的断言保证的是"算子的输出缓冲不小于输入缓冲"，因为 epilogue 是原地处理。（  ）

**A-5 边界走查（选做）**。设 `M=100, N=128`（其余同 A-2）：调度器只给一个输出块（`actual_m = 100`），m/n 扫描仍是 2×2 共 4 个子块，边界在 `m_mad=64` 行——`num_rows = min(100-64, 64) = 36`。写出 AIV1 第二个 tile（`s_epi=3`）的 `MadBlock` 全部 8 个字段值，并说明：`copy_ub_to_gm` 写出的 GM 区域是哪一块？UB 里 `eff_mad_n` 对齐出来的填充列为什么不会污染 GM（提示 `:65-66` 用的是 `min` 真值）？

### B. 有 Ascend 环境：验证语义变体（可选，约 25 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
```

1. 对照 `tests/common.py:154-174` 确认本轮测试自动覆盖的 epilogue 变体：`with_bias ∈ {False, True}`、无 bias 时的随机 `alpha`、`accumulate` 用例与 `BF16StochasticRounding` 变体；
2. 从 `DG_PRINT_CONFIGS` 的 `GemmConfig(...)` 输出（`heuristics.hpp:749-753` 打印）记录几个 shape 的 `num_epilogue_stages`，验证 `num_epilogue_stages × 2 ≥ (block_m/mad_m)×(block_n/mad_n)`（`epilogue.hpp:33` 的 host 镜像）与 UB 账本 `num_epilogue_stages × epilogue_bytes ≤ 256KB`；
3. 注意打印行的两段结构（`heuristics.hpp:751` 的 `std::format("{} {}", d, config)`）：`GemmDesc` 段含 `acc=` 与 `epilogue_type=`（`gemm_config.hpp:151`），`GemmConfig` 段含 `num_epilogue_stages`；性能行尾部的 `acc=1` 标记来自 `tests/utils.py` 的 `print_perf`（`tests/common.py:232`）——把"测试变体 → 配置差异"记入笔记；
4. 再跑 `DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py unaligned-mn --skip-prof`，把 `epilogue.hpp:65-66` 的 `min(actual_m - m_mad, MAD_M)` / `min(actual_n - n_mad, MAD_N)` 与边界块的 `MadBlock`、`copy_ub_to_gm` 的真实行列数对应起来（`unaligned-k` 影响的是 mmad 一侧的 `eff_mad_k`，Epilogue 只看 M/N 边界）。

## 六、当天产出

**AIC/AIV 分工与 Epilogue 流程图**。把 A-1～A-3 的结果整理成一页笔记，至少包含：

1. **分工总图**：仿 3.9 图一，覆盖 host 三选一、`__mix__(1,2)`、两个分支的职责、四步借还协议（每个事件标注发出方、队列、含义）；
2. **三泳道时序图**：A-2 的推演结果（至少 4 个子块的交付 + 两个 AIV 的消费 + 归还事件）；
3. **语义位置表**：3.6 的"语义 → Python 输入 → host 判定 → device 位置"表（自己默写一遍）；
4. **UB 布局图**：一个 AIV 的 UB——`kNumStagesPerAIV` 格 stage + 算子附加区，标出 Identity / ScaleByAlpha / AddBias 三种情况的具体字节数（用 A-2 的配置）；
5. **变体记录**：V1～V5 各两三行结论；
6. **遗留问题更新**：把"M 写 L0C 与 FIX 读 L0C 的跨队列次序"条目改写为 3.4 节的结论（边界 + 两条可见线索 + 属 CANN 文档），并新增一条本周问过的自查："如果让你把归还 arrive 从 MTE3 提前到 V（仿 `mqa_logits.hpp:420`），需要什么前提？"（第 26 天看 `kDirectStore` 后再回答）。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. `__global__ __mix__(1, kDirectStore ? 0 : 2)` 各数字是什么意思？`kDirectStore` 时 AIV 分支的代码还执行吗？`asc_get_sub_block_id()` 在 AIV 上返回什么？
2. `aic_store` 中 `copy_l0c_to_ub` 的 6 个实参各是什么？`dst_subblock_id` 为什么必须是 `aiv_id`？输出转换（FP32→BF16）在什么条件下发生在这条拷贝里、什么条件下被挪到 AIV？
3. 写出块内事件的 5 位编码。为什么 AIC 等待时用完整编码、AIV 等待时只用低 4 位（从"若 AIV 也按完整编码匹配会怎样"推一遍）？
4. 完整描述一个 UB stage 的借还闭环（四次调用、各自的执行者与队列、每次传递的语义），并说明：`aic_wait_free` 为什么只在 `k_blk_idx == 0` 调用？AIV 的归还 arrive 为什么排在 `copy_ub_to_gm` 之后？
5. accumulation、bias、alpha 分别发生在哪个核、哪条队列/阶段？哪些组合被禁止？分别在 host 的哪一行、device 的哪一行被拦下？
6. 给定 `kNumEpilogueStages=2`、`MAD_M=MAD_N=64`：BF16 输出 + `Identity` 时每个 AIV 的 UB 占用是多少？换 `ScaleByAlpha` 呢？再换 `AddBias`（`BLOCK_N=128`，BF16）呢？写出三笔账并指出对应的断言/host 公式。
7. AIC 与两个 AIV 之间不传任何 `s_epi`，三方为什么能对"当前 tile 用哪格 UB"保持一致？写出两条机制。`kNumAIVs=1` 的真实用例在哪个文件，此时 tile 归谁？

### 参考答案

1. 一个 block 由 1 个 cube 子块（AIC）和 `kDirectStore ? 0 : 2` 个 vector 子块（AIV）混合执行同一份内核代码；`kDirectStore` 时没有 AIV 子块，`:174` 的 `return` 让 else 分支在该形态下不产生副作用；AIV 上 `asc_get_sub_block_id()` 返回 0 或 1（`epilogue.hpp:51`），是 AIV 的本地身份。
2. 实参：`ub[stage_idx]`（目标 AIV 的第 stage_idx 格 UB）、`l0c_tile`（本子块 L0C 结果）、`m_elems/n_elems`（`eff_mad_m/eff_mad_n`，N 已按 `get_frac_k<ub_dtype_t>` 对齐保证行 32B）、`DUAL_DST_DISABLE`（单目标）、`aiv_id`（写入哪个 AIV 的 UB）。`Identity` 时 `ub_dtype_t = cd_dtype_t`，默认 `quant_mode = F322BF16`（BF16 输出，`copy.hpp:237-238`），转换在 FIX 拷贝完成；带 V 算子时 `ub_dtype_t = float`，FIX 拷贝 `NoQuant` 原样搬 FP32，转换在 AIV 的 `process_mad_block` 尾部（`asc_float2bfloat16_rn`）。
3. `((subblock_id & 1) << 4) | (flag_id & 0xf)`（`sync.hpp:30`）。若 AIV 也按完整编码匹配：AIC 发给 AIV0 的事件编码是 `(0<<4)|s`，AIV1 等待时无法用自己的身份参与匹配（它拿到的参数同样是 `s`），两个 AIV 会争抢同一组编号；由于路由过滤（`:57`）保证每个 AIV 只等待"发给自己的"事件，"AIV 等待自动关联自身那一组"是从代码成立的最小解释（`sync.hpp:34` 注释明说 AIV 忽略 subblock_id）。
4. 闭环：① AIC `aic_wait_free`——FIX 队列 `wait_intra_block(PIPE_FIX, aiv_id, stage_idx)`，等 AIV 归还；② AIC `aic_store`——FIX 队列 `copy_l0c_to_ub` + `set_intra_block(PIPE_FIX, aiv_id, stage_idx)`"已填好"（arrive 随队保序，AIV 收到时数据必已就绪）；③ AIV `wait_intra_block(kPipe, aiv_id, stage_idx)`——在算子工作队列上等"已填好"；④ AIV 处理 + `copy_ub_to_gm` + `set_intra_block(PIPE_MTE3, aiv_id, stage_idx)`"已搬完"。`wait_free` 只需借一次（`s_epi` 在子块的 K 遍历期间不变，`:107` 到 `:164`），且它阻塞的是 FIX 队列——本子块的 FIX 拷贝要到最后一个 k_blk 才发出，开头的一次等待覆盖全程；归还排在 MTE3 的拷贝之后，使 AIC 的"空闲"语义包含"GM 已写完、UB 可复用"（`epilogue.hpp:84` 与 `mqa_logits.hpp:420` 的对照说明这是有意选择）。
5. accumulation：GM 原子加——普通路径 AIV 在 `aiv_initialize` 设 `asc_set_atomic_add_*`（`epilogue.hpp:91-96`），作用在其后的 `copy_ub_to_gm`；direct store 时 AIC 设（`bf16_gemm.hpp:82`）。bias：AIV 的 V 管道（预取 `operators.hpp:175-183`，相加 `:185-221`）。alpha：AIV 的 V 管道（`operators.hpp:131-157`）。禁止组合：bias×alpha（host `gemm.hpp:115-116` 断言，测试 `common.py:191-197` 验证报错）；bias×accumulation（host `acc = c.has_value() && !with_bias`，`gemm.hpp:180`；device `AddBias::is_supported` 要求 `!kWithAccumulation`，`operators.hpp:171-173`，经 `epilogue.hpp:34` 编译期触发）。alpha×accumulation 允许：`D += alpha·(A@B)`。
6. Identity+BF16：`2 × 64×64×2B = 16KB`；ScaleByAlpha：`2 × 64×64×4B = 32KB`；AddBias（BF16 输出）：`2 × 64×64×4B + 2 × 128×2B = 32KB + 512B`。依据：`epilogue.hpp:25-26`、`operators.hpp:167, 169`、总量断言 `:35`；host 公式 `heuristics.hpp:188-190`（`epilogue_bytes` 含 bias 附加，`epilogue_elem_bits` 由 `:81-82` 决定）。
7. 机制一：三方持相同 `Scheduler`、以相同顺序遍历相同的块序列，块内 m/n 子块扫描同构（`scheduler.hpp:10-12`、`epilogue.hpp:55-56`）；机制二：相同的 `s_epi` 推进规则——每 tile 前进一格、`kNumStages` 回绕（AIC `bf16_gemm.hpp:164` + `epi_base :169`；AIV `epilogue.hpp:56` 引用传参延续，`continue` 不阻止推进）。`kNumAIVs=1` 的用例：`fp8_dequant_gemm.hpp:169-171`（`kDequantAIVId=0`，`aiv_id = 2-1 = 1`，全部 tile 归 AIV1，AIV0 做 FP4 反量化）。

**A-4 判定题答案**：

1. **假**。挂在 `PIPE_FIX`（`epilogue.hpp:46`），随 FIX 队列保序。
2. **真**。归还 arrive 在 MTE3 队列内排在 `copy_ub_to_gm`（及 postprocess 写回）之后（`epilogue.hpp:81-84`）。
3. **假**。`c=[1,N]` 判定为 bias，`acc = c.has_value() && !with_bias`（`gemm.hpp:180`）；device 侧 `AddBias::is_supported` 也要求 `!kWithAccumulation`（`operators.hpp:171-173`）。
4. **真**。编码低 4 位是 flag 编号（`sync.hpp:30`），`kNumStagesPerAIV ≤ 16`（`epilogue.hpp:32`）；host 镜像 `hw.num_intra_block_events`（`heuristics.hpp:190`）。
5. **真**。`stage_idx = s_epi / kNumAIVs ∈ [0, kNumStagesPerAIV)`，`dst_subblock_id = aiv_id` 选择目标 AIV 的 UB（`epilogue.hpp:43-45`）。
6. **真**。`sizeof(ub_dtype_t) >= sizeof(cd_dtype_t)`（`epilogue.hpp:28`）——原地处理时输出视图必须放得进输入缓冲。

---

**明日预告（第 26 天）**：今天我们把普通路径的 Epilogue 走到了 GM。明天看它的三条"近道"：`kDirectStore`（AIC 用 `copy_l0c_to_gm` 直写、完全没有 AIV 分支——今天 `:82` 的 AIC 侧原子加已经露过脸）、`kResidentA/B`（A 或 B 常驻 L1/L0 的数据复用）以及非对齐 shape 与空组的处理，整理一张普通路径与特殊路径的对比表，并回答今天 A-5 留下的"归还位置前移"问题。
