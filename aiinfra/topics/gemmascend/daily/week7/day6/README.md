# 第 7 周 · 第 6 天（总第 48 天）：扩展算子复盘

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 7 周"MegaMoE、HC Prenorm 与扩展算子"的第 6 天（复盘日）教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 55 分钟，源码导读约 20 分钟，综合练习约 55 分钟，通过标准自评与自测题约 15 分钟。

本周走完了本仓库全部三个"GEMM 主线之外"的算子：第 43～46 天把 MegaMoE 从接口、计算主线、通信融合读到测试与性能入口，第 47 天读了唯一的 TileLang 算子 HC Prenorm；再加上第 39～40 天的 MQA Logits 家族，扩展算子的版图就齐了。今天是复盘日，**不引入新内容**。学习计划只给了一个任务：**从 MQA、MegaMoE、HC Prenorm 中选择最感兴趣的一个**，不看笔记回答四个问题——①输入输出和核心计算是什么？②调度与寻址有什么特殊之处？③为什么普通 GEMM 接口不能直接满足需求？④性能瓶颈可能在哪里？通过标准只有一句话：**至少能完整讲解一个扩展算子的计算流程**。复盘方法论与第 6/13/27/34/41 天相同：**闭卷复述是唯一标准**——本文 3.3～3.5 节给出三个算子的四问参考答案（每个都落到文件与行号），请先自己写再看；3.2 节的三算子总对比表是热身，也是"没想好选谁"时的决策辅助。

## 一、今日定位与学习目标

完成今天的学习后，你应当能够：

1. 闭卷默写**三算子总对比表**的主干（数学核心、输入输出、调度与寻址特征、与 GEMM 主线的关系、性能画像、实现语言），并说出每个格子的至少一个源码锚点（参考版见 3.2 节）；
2. 为你选定的算子闭卷回答**四问**，每问都有公式/形状/机制/数据支撑，并给出文件与函数级位置（3.3～3.5 节）；
3. 对另外两个算子，至少能各讲清"**它为什么不是一个普通 GEMM**"的最强一条理由（3.3～3.5 节每节的问题③）；
4. 把三个算子的性能画像区分开：MQA 是 **FIX-pipe bound**（README：FIX 99%）、MegaMoE 小 T 时延/同步受限、大 T 计算受限、HC Prenorm **访存受限**（"nearly saturates the HBM write bandwidth"）——并能说出各自对应的验证手段（口径公式或 profiling 字段）；
5. 完成一份"拼装版代价"的量化论证：用具体数字说明若用普通 GEMM API + 辅助 kernel 拼装，各需要几个 kernel、中间张量多大或多读多少字节（综合练习 B4）；
6. 把本周遗留问题逐条归档，明确去向（第 49 天机动、第 8 周性能分析，或放弃）。

## 二、预备知识

今天的复盘覆盖第 39～47 天的扩展算子内容，不引入新知识；哪一条卡住，就按下面的索引回到对应天数补漏：

- **第 17～26 天**：Normal GEMM 的调度、搬运、MAD、流水线、AIC/AIV 与 Epilogue——三个算子全部复用这套主干，问题③的论证全靠它做参照。
- **第 32～33 天**：FP8/FP4 路径与 UE8M0/MX scale（`MX_SF_DIVISOR = 64`，`common.hpp:65`）——MQA 与 MegaMoE 的输入都是低精度 + SF 布局。
- **第 36～38 天**：M-Grouped GEMM 与 psum layout——MegaMoE 内部两个 GEMM 的直接原型，baseline 也直接调它。
- **第 39～40 天**：MQA Logits 公式、一维条带调度、压缩存储、sparse 选块、paged 页表、FIRST/REUSE 快照——3.3 节的全部素材。
- **第 41 天**：Grouped GEMM 与 MQA 的复盘——3.2 节总表可视为那张表的续篇（本周补上 MegaMoE 与 HC Prenorm 两列）。
- **第 43～46 天**：MegaMoE 的接口/主线/通信/测试——3.4 节的全部素材，贯穿配置 `R=2, E=8, S=0, T=128, k=2, H=I=512` 沿用。
- **第 47 天**：HC Prenorm 的融合收益、`get_config`、K-split 原子归约与确定性——3.5 节的全部素材。

硬件常量回顾（`deep_gemm/include/deep_gemm/common.hpp:50-65`，Ascend 950DT）：32 个 AI Core（AIC），64 个 AIV；L1 每核 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB、UB 256 KiB；`FRAC_MN = 16`（NZ 分形行数）。另注：第五节 A 组任务需要 Ascend 环境（MegaMoE 还需多卡与 `deep_ep`，见 `tests/test_mega_moe.py` 的依赖分叉）；无环境时完成 B 组静态任务即可达到当天目标。

## 三、知识梳理

### 3.1 本周五天一览

| 天 | 主题 | 一句话结论 | 当天产出 |
| :---: | :--- | :--- | :--- |
| 43 | MegaMoE 对外接口 | 三个 Python 符号（`SymmBuffer` / `transform_weights_for_mega_moe` / `fp8_fp4_mega_moe`，`deep_gemm/__init__.py:137-141`）牵出对称缓冲区、13 个视图与权重转换 | API 与数据准备流程图 |
| 44 | MegaMoE 计算主线 | 一个 `mega_moe_impl`（`mega_moe.hpp:1127`）装下五段，AIC/AIV0/AIV1 三类角色 + 环形接收池 + 四阶段进度计数 | 张量形状变化表 |
| 45 | MegaMoE 通信与融合 | HCCL UBMEM 直访 + 拉取式 Dispatch / 推回式 Combine + 计算通信重叠 | 计算/通信时序图 |
| 46 | MegaMoE 测试与性能 | 多 rank spawn 组织、三口径公式（flops/HBM/comm）、msprof PIPE trace | 依赖与指标清单 |
| 47 | HC Prenorm / TileLang | 一份搬运两种计算（GEMM + 行平方和）+ K-split 原子归约；唯一 TileLang 算子 | 两种开发方式对比笔记 |

### 3.2 三算子总对比表（产出①参考版）

先闭卷画你自己的版本（列 = MQA / MegaMoE / HC Prenorm，行自选至少 7 个），再对照下表：

| 维度 | MQA Logits（第 39～40 天） | MegaMoE（第 43～46 天） | HC Prenorm（第 47 天） |
| :--- | :--- | :--- | :--- |
| 数学核心 | `logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`，仅 `j ∈ [ks_q, ke_q)` 有效 | MoE 前馈整层：Dispatch → GEMM1 → SwiGLU → GEMM2 → Combine（`README.md:162`） | `D = float(A) @ B.T` **且** `sqr_sum = sum(float(A)**2, dim=-1)`，一个 kernel（`hc_prenorm.py:363-379`） |
| 输入 | Q `[Q,H,D]` FP8/FP4（H∈[4,64] 且 4\|H，D∈{64,128}）、KV `[KV,D]` 同 dtype、weights `[Q,H]` bf16、每行窗口 `cu_seq_len_k_start/end`（`csrc/apis/attention.hpp:248-256`） | 激活 `[T,H]` FP8+SF、`topk_idx/weights [T,k]`、FP4 路由权重 + FP8 shared 权重（需预先 transform）、`SymmBuffer`（`csrc/apis/mega_moe.hpp:128-204`） | a `[M,K]` BF16、b `[N,K]` FP32（典型 N=24、K≤28672，`tests/test_hyperconnection.py:14-16`） |
| 输出 | logits `[Q, max_seqlen_k]` bf16，行距 512B 对齐，每行只写有效前缀（`attention.hpp:23-27, :301-302`） | `[T,H]` bf16 + `cumulative_local_expert_recv_stats`（本 rank 各 expert 接收计数） | d `[M,N]` FP32、sqr_sum `[M]` FP32 |
| 调度 | 一维条带：核 `core_id` 出发、`+= kNumCores` 轮询 q_block（`scheduler/mqa_logits.hpp:36`），q_block 内按 `kSplitKV=512` 切 kv_split | `BlockPhase` 五种任务、shared 优先、routed Linear1 带一波 lookahead；**AIC 与 AIV1 重放同一任务序列**（`scheduler/mega_moe.hpp:56-118`） | K-split 切给多组 core（`get_config` 三候选，`hc_prenorm.py:33-42`），split 内 token block 交错认领（`:175`） |
| 寻址/布局特殊点 | paged 版 `block_table` 间接寻址 + 物理页乱序的 `reversed_pairs` 恢复；sparse 版选块并集装载 | 远端访问 = 本地指针 + (远端基址 − 本地基址)（`layout/sym_buffer.hpp` 地址代数，UBMEM 直访）；`ExpertCursor` 把现场生成的接收计数映射成 M-block | AIV 现场做 BF16→FP32 拓宽 + NZ 分形打包（`hc_prenorm.py:196-267`），`T.dual_copy` 跨单元 UB→L1 |
| 与主线 GEMM 的关系 | 复用搬运/MAD/FIX/AIV 主干；新增条带调度、head 归约、窗口压缩存储、页表 | 内部两个 GEMM 就是 M-Grouped 形态，但分段信息在 kernel 运行时才产生；SwiGLU/通信是全新段落 | 复用两层 K 循环/累加/原子加思想，但 K-split 多核共写输出是主线没有的；TileLang 实现 |
| 性能画像 | **FIX-pipe bound**（README:151，FIX 99%）；Decode 场景 2006 GB/s | 小 T（64）时延/同步受限（228.5 TFLOPS、通信仅 37.5 GB/s）；大 T（16384）计算受限（846.3 TFLOPS） | **访存受限**（README:177，"nearly saturates the HBM write bandwidth"；峰值 82 TFLOPS ≪ 432） |
| 实现语言/主文件 | C++ 模板：`mqa_logits.hpp` + 三个 jit_kernels 入口 | C++ 模板：`mega_moe.hpp`（1221 行，仓库最大算子） | TileLang：`hc_prenorm.py` 单文件 398 行 |
| 测试文件 | `tests/test_attention.py`（1301 行，normal/paged/sparse/paged_sparse） | `tests/test_mega_moe.py`（322 行，多 rank spawn + baseline 交叉验证） | `tests/test_hyperconnection.py`（160 行，正确性/契约/确定性三层） |

三点补充：

- **三个算子回答了三个不同方向的"扩展"**：MQA 把 GEMM 的输出语义换掉（逐元素非线性 + 输出侧缩减 + 变长窗口）；MegaMoE 把多个算子和**跨卡通信**融进一个 kernel；HC Prenorm 把 **prologue 侧的旁路统计**搭上 GEMM 的数据搬运。选谁深入，取决于你对哪个方向感兴趣（见 3.6）。
- **实现方式的比例**：三个算子里两个是手写 C++ 模板（MQA、MegaMoE），一个是 TileLang（HC Prenorm，`README.md:199`"The mHC kernel is backed by Tilelang"）——第 47 天 3.8 节的对比笔记今天可以直接复用。
- **测试都长在同一棵树上**：`calc_diff` 余弦差、参考结果交叉验证（MQA 用 `ref_mqa_logits`，`tests/test_attention.py:94`；MegaMoE 用 deep_ep baseline，`tests/test_mega_moe.py:130-156`；HC 用 `a.float() @ b.T`，`tests/test_hyperconnection.py:59-98`）——第 30 天的验证方法贯穿始终。

### 3.3 四问参考答案：MQA Logits

**问题①：输入输出和核心计算是什么？**

- 计算：对每个 query 行 q 与每个有效 KV 位置 j，`logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`。注意顺序：先算 head 内点积，**ReLU 在点积之后、head 归约之前**，最后乘 head 权重 `w[q,h]` 求和。
- MAD 三个维度的对应（`mqa_logits.hpp:214-217`）：M = QH 行（`MAD_M` 装 query×head，如 H=32 时 MAD_M=64 装 2 个 query）、N = KV token（`MAD_N = kSplitKV = 512`）、K = head_dim（`MAD_K = kHeadDim`，64 或 128）。**没有 K 维两层循环**——收缩维一次装下，每个 (tile, kv_split) 只调一次 `mad(..., ENABLE_UPDATE, true)`（`:396-397`）。
- 数据通路：AIC 搬 Q/KV 进 L1/L0（连同 MX scale，`_mx` 后缀的装载版本）→ 单次 `mad` 进 L0C → `copy_l0c_to_ub` 时 **ReLU 融合在 FIX 搬运里**（`:400-404`，`infer_quant_mode<bfloat16_t, float>()` + relu 参数）→ AIV 的 `vf_mqa_logits_reduce`（`:102`）做 head 加权归约 → `mqa_logits_store`（`:46`）按窗口压缩写 GM。
- 家族四条路径：normal（连续 KV + 运行时调度器）、sparse（top-k 选块 + 元数据 kernel 排产）、paged（页表 + fused 页 `[DATA | SF]`）、paged sparse（两者叠加 + FIRST/REUSE 快照，状态由 Python 侧 `metadata._paged_sparse_mqa_state` 管理，`deep_gemm/__init__.py:111, :133-134`）。

**问题②：调度与寻址有什么特殊之处？**

1. **一维条带调度 + 窗口聚合**：`MQALogitsScheduler::next_q_block`（`scheduler/mqa_logits.hpp:31-52`）让每核从 `core_id` 出发按 `kNumCores` 跨步认领 q_block；块内所有 query 的 `[ks, ke)` 取 min/max 聚合成块级窗口，`kv_token_base = start/8*8`，`num_kv_splits = ceil((end − base)/512)`。
2. **压缩存储**：每行只写 `[ks, ke)` 前缀，列号 `dst_col = store_begin − ks`（`mqa_logits.hpp:590-596`），`store_begin/end` 由 split 区间与窗口取交；行头未对齐部分经 32B 对齐 scratch 中转（MTE3 约束，`:46-68`）。
3. **tile 活性判断**：`is_tile_active`（`:328`）跳过与当前 split 无交叠的 QH tile，`is_tile_active` 与压缩存储配合避免了为无效位置写 `-inf`。
4. **paged 间接寻址与顺序恢复**：`get_physical_page`（`scheduler/paged_mqa_logits.hpp:119-123`）查 `block_table` 把逻辑页映射到物理页；`get_physical_page_batch`（`:125-134`）把相邻两页从较小物理页发起成对搬运，AIV 侧用 `reversed_pairs` 位图在存储时交换回来（`mqa_logits.hpp:545-554` + `vf_mqa_logits_store_bf16`，`:74`）。
5. **元数据 kernel 排产**：sparse 家族把"按代价把任务切给核"从 Host 搬到设备侧一次性完成（`sparse_mqa_logits_metadata.hpp` 等），主 kernel 只消费静态计划。

**问题③：为什么普通 GEMM 接口不能直接满足需求？**

- **输出语义不是"矩阵乘的部分和"**：ReLU 发生在逐元素乘积求和之后、head 归约之前；GEMM 的 epilogue 只能作用于输出元素本身，不能做跨行（head 维）的加权归约 `Σ_h ·w[q,h]`。
- **每行有效区间不同**：`[ks, ke)` 逐行变化，GEMM 假设规则矩阵；掩码写 `-inf` 浪费带宽，压缩前缀存储需要窗口感知的 store。
- **中间张量不可承受**：若用 GEMM 算 `[Q·H, KV]` 打分矩阵再后处理，Prefill 用例（Q=4096、H=32、KV=8192）的中间张量为 `4096×32×8192×2B ≈ 2.15 GB`，而最终输出只有 `4096×8192×2B ≈ 64 MB`；多出两个 kernel 还要把它读一遍再写一遍。
- **间接寻址**：paged/sparse 的 KV 来自页表和选块表，GEMM 的双 major 寻址（`gm_ptr::index`）表达不了；FIRST/REUSE 快照更是调用级状态。

**问题④：性能瓶颈可能在哪里？**

- **FIX-pipe bound**：README 明说该 kernel "is FIX-pipe bound rather than compute bound, saturating the FIX pipe at 99% utilization"（`README.md:151`）。瓶颈在 L0C→UB 的转换吞吐（每个 (tile, split) 都要一次带 ReLU 的 `copy_l0c_to_ub`），不在 cube 算力——"增大 MAD tile"对它无益，优化方向是减少 FIX 转换次数/提高其吞吐。
- Decode 场景（Q=256）TFLOPS 565、带宽 2006 GB/s（`README.md:157-158`）说明小 Q 时访存占比上升；head 归约在 AIV 的 V 管上，是第二瓶颈。验证手段：`bench_msprof` 的管道利用率字段（`KernelProfile`，`deep_gemm/testing/bench.py:27-71`）里 fix 利用率应显著高于 mad。

### 3.4 四问参考答案：MegaMoE

**问题①：输入输出和核心计算是什么？**

- 计算：一整个 MoE 前馈层，五段（`README.md:162`）：Dispatch（token 按 expert 归属拉到持有者）→ Grouped GEMM1（`[recv,H] × W1[e]ᵀ → [recv,2I]`，gate/up 拼接）→ SwiGLU（`up × silu(gate)`，带 clamp、乘 topk 路由权重、重新量化成 FP8）→ Grouped GEMM2（`[recv,I] × W2[e]ᵀ → [recv,H]`）→ Combine（按 topk_weights 加权送回源 rank 求和，shared expert 输出也在此合并）。
- 输入：本 rank 激活 `[T,H]` FP8+SF、`topk_idx/topk_weights [T,k]`、预转换的 FP4 路由权重与 FP8 shared 权重（`transform_weights_for_mega_moe`：gate/up 交织、BLOCK_N 分块、NZ 重排、FP4 SF 指数 +6 补偿，`deep_gemm/mega/__init__.py:84-126`）、`SymmBuffer`（各 rank 同段 HCCL 注册内存）。
- 输出：`[T,H]` bf16 与 `cumulative_local_expert_recv_stats`（每 rank 各 expert 的接收计数，测试据此复算 flops，`tests/test_mega_moe.py:219`）。
- 主体是 7 个模板参数（R/E/S/T_max/k/H/I）的 `__global__ __mix__(1, 2)` kernel `mega_moe_impl`（`mega_moe.hpp:1127-1219`），32 个 Block 启动，每 Block 1 AIC + 2 AIV。

**问题②：调度与寻址有什么特殊之处？**

1. **三类角色**（`:1161-1215`）：AIC 跑 `aic_gemm_pipeline`（`:403`，两个 GEMM 的全部 MAD）；AIV0 跑元数据构建 + FP4→FP8 NZ 权重流水（`aiv0_weight_pipeline`，`:299`）；AIV1 跑 Dispatch 拉取（`advance_dispatch_pull`，`:763`）、SwiGLU epilogue（`vf_swiglu_quant`，`simd/mega_moe.hpp:59`）与 Combine 推回（`advance_combine_push`，`:1000`）；64 个 AIV 共同执行 `combine_reduce`（`:252`）。
2. **任务序列重放**：`take_linear_task`（`scheduler/mega_moe.hpp:58-118`）以 per-core 的任务游标产出 shared 优先、routed Linear1 带一波 lookahead 的任务序列；**AIC 与 AIV1 用相同参数重放同一序列**（头注释 "Replay the same routed task order on AIC and AIVs"，`:56-57`），从而在每个任务上"会合"——这是五段能融进一个 kernel 的调度根基。
3. **现场生成的分组信息**：`ExpertCursor`（`:19-54`）从 Dispatch 元数据写下的 `expert_recv_count_sum` 出发推演每个 M-block 的 `valid_m`，并以 `FRAC_MN=16` 分组平衡切分——对照第 36 天，M-Grouped GEMM 的 psum 是 Host 预先给定的，这里它在 kernel 运行时才产生。
4. **远端寻址**：`SymBuffer` 的"一层地址代数"——远端访问 = 本地指针加 `(远端基址 − 本地基址)`，Device 里远端读写与普通 GM 访问走同一条 MTE2/MTE3 通路；底座是 `HCCLSymmetricBuffer`（`csrc/backends/hccl.hpp:29`）的 UBMEM 直访注册。
5. **环形接收池 + 进度计数**：l1_acts / l2_acts / routed.output 三块缓冲经 `get_ring_block_idx`（`layout/mega_moe.hpp:255-267`）取模复用；四个 `ProgressStage`（kDispatched / kLinear1Ready / kLinear2Ready / kCombineSent，`:27-33`）的计数器表达"生产到哪了"，计数以 `kCountArrivalBit | value` 打包发布（`:23-24`）。

**问题③：为什么普通 GEMM 接口不能直接满足需求？**

- **分段信息在运行时才产生**：M-Grouped GEMM（`m_grouped_fp8_fp4_gemm_nt_contiguous`）要求 Host 侧先拿到 psum layout；MegaMoE 的分段（各 expert 收到多少 token）是 Dispatch 通信的结果，只有融进 kernel 才能省掉"通信 → 同步回 Host → 再启动 GEMM"的往返。
- **五段之间的衔接需要 kernel 内同步**：GEMM1 → SwiGLU 走 Block 内 UB 交接（FIX→V 信用），SwiGLU → GEMM2 走 GM 进度计数（`kLinear1Ready` 原子累加）——普通 API 之间中间张量必须落 HBM。
- **baseline 的代价有代码为证**：`run_baseline`（`tests/test_mega_moe.py:130-156`）用 deep_ep dispatch/combine + 2 次 grouped GEMM + TileLang SwiGLU 拼装，benchmark 侧要统计**7 个 kernel** 的时间（`:207-209` 的 kernel 名单）；通信 kernel 与计算 kernel 无法重叠。
- **Combine 不是 GEMM**：跨 rank 加权归约 + shared expert 合并，输出写在源 rank 的 token 行上。

**问题④：性能瓶颈可能在哪里？**

- **小 T（如 64）时延/同步受限**：README 表中 T=64 仅 138.7 µs、228.5 TFLOPS，通信带宽仅 37.5 GB/s（`README.md:166`）——固定开销主导：Dispatch 元数据协议的多次 `grid_sync`、`rank_barrier`、以及 96 个 expert 权重的读取（HBM 1850 GB/s 说明访存里权重占大头）。此形态下算力大量空转。
- **大 T（16384）计算受限**：846.3 TFLOPS（`README.md:173`）已接近 M-Grouped FP8×FP4 的水平（第 41 天：约 860）——瓶颈回到 cube 算力与 HBM 带宽的常规权衡。
- **量化手段现成**：三口径公式 `flops = 6·H·I·(recv + T·S)`（`tests/test_mega_moe.py:221`）、逻辑 HBM 字节（`:222-228`）、远端通信字节 = 远端 token 数 × H × 3（`:229-231`）；时延取最慢 rank（amax）、吞吐用平均工作量（`:239` 注释）。`kOverlapReduce`（32 MiB combine 工作量阈值，`mega_moe.hpp:1139`）决定 combine 归约是否与大 T 的计算重叠。

### 3.5 四问参考答案：HC Prenorm

**问题①：输入输出和核心计算是什么？**

- 计算（`hc_prenorm.py:363-379` docstring）：`D = float(A) @ B.T` 与 `sqr_sum = sum(float(A) ** 2, dim=-1)` 在**一个 kernel** 内完成；`num_splits` 只接受 None 和 1，因为 K split 的归约在 kernel 内部就做完了（`:377-388`）。
- 形状：a `[M,K]` BF16、b `[N,K]` FP32、d `[M,N]` FP32、sqr_sum `[M]` FP32。典型 N = hc_mult×(hc_mult+2) = 24（hc_mult=4），K = hc_mult×hidden ≤ 28672——**小 N 大 K** 的极端形状。
- 精度：cube 走 HF32 模式（`T.set_hf32_mode('nearest_even')`，`:136`）——A 是 BF16 可精确表示，B 从 FP32 舍入到 HF32，这就是测试容差 1e-6 而非位级的原因。
- 数据通路：AIV 把 x 搬进 UB，在 `T.SimdVF` 段（`:196-267`）**一趟完成三件事**——BF16→FP32 位级拓宽、NZ 分形打包（cube 只吃分形布局）、逐行平方和；`T.dual_copy`（`:269`）把打包结果送 AIC 的 L1，AIC 做 GEMM，L0C 跨 hidden block 累加后经 FIX 写回；行平方和从 AIV 的 UB 经 MTE3 直接写回 GM。

**问题②：调度与寻址有什么特殊之处？**

1. **K-split 多核共写输出**：hidden 维切给 `num_k_splits` 组 core，各组算部分和——输出区域被多组 core 写，这在主线 GEMM 里从未出现（每个输出 tile 只有一个 core 负责）。三个 split 候选（workload / hidden / cores，`get_config` `:33-42`）再降到最大 2 的幂。
2. **两相位 store + 跨核屏障**：非 deterministic 模式 split 0 普通写、其余 split 原子加（`T.set_atomic('add', 'float32')`，`:135`），两个相位间用 `T.ascend_sync_inter_arrive/wait` 成对构成全核屏障（`:295-357`）；"最后一个相位的屏障只在还有下一个 chunk 时需要"，且被编码成 0/1-trip 的 `T.Serial` 循环——因为 AutoSchedule 拒绝 `T.PerCoreTask` 内条件控制流下的跨核等待（`:100-103` 注释）。
3. **token block 交错认领**：`token_block = token_block_base + token_stage × cores_per_split`（`:175`），各核填满自己的 16 个 L0C 槽（`out_mul_l0` 首维，`256KB/16KB`）。
4. **prologue 侧融合**：与 BF16 Kernel 的"AIV 干 epilogue"相反，这里 AIV 干 **prologue + 旁路输出**——拓宽和 NZ 打包本来就必须在进 cube 之前完成，行平方和搭"顺风车"。
5. **L2 提示**：x 的加载带 `l2_cache_ctrl='notalloc_keep'`（`:191`）——K-split 下每个 x 元素全局只读一次，进 L2 无收益。

**问题③：为什么普通 GEMM 接口不能直接满足需求？**

- **sqr_sum 是逐行统计量**：GEMM epilogue 只见输出 tile，无法在数据流过时对输入行做 K 维全长的平方和；不融合就要第二个 kernel **再读一遍 A**——M=8192 用例 A 读取 469.8 MB，占总流量 99%，不融合总流量约翻倍到 ~943 MB（第 47 天 3.1/3.2 的手算）。
- **形状在 GEMM 启发式里是低效形状**：N=24 无法用 M×N 网格吃满 32 核，必须 K-split——而带原子归约的 K-split 不在 `bf16_gemm` 的模板空间里。
- **dtype 与布局**：A 是 BF16、cube 需要 FP32/HF32 的 NZ 布局，主线由 `copy_gm_to_l1` 家族隐藏的转换在这里要 AIV 现场做。
- **确定性契约**：det 模式要求 split 数与 M 无关（workload 候选固定 16，`:33-35` 注释）+ 固定相位写回顺序，普通 GEMM 接口没有这种 batch invariance 契约。

**问题④：性能瓶颈可能在哪里？**

- **访存受限**：README 称其 "nearly saturates the HBM write bandwidth"（`README.md:177`）；峰值 82 TFLOPS，远低于 BF16 峰值 432——因为 N=24 使算术强度极低。小 M 时 B 读取占 79% 流量，大 M 时 A 读取占 99%（第 47 天 3.1 手算），优化目标是把总流量压到最小，融合正是为此。
- **验证手段**：性能行的 `mad%` 低、`mte2%` 高（`print_perf`，`tests/utils.py:106-115`）就是"访存受限"在 profiling 数据上的投影；带宽口径 `bytes = 2·M·K + 4·(N·K + M·N + M)`（`count_bytes`，`tests/utils.py:95-103`）。

### 3.6 怎么选，以及易错点

**选择建议**（没有标准答案，按兴趣）：

- 想深入**注意力/推理栈** → 选 MQA：变长窗口、页表、稀疏选块是推理 kernel 的通用词汇，FIRST/REUSE 是"调用级状态"的典型设计；
- 想深入**分布式/MoE 系统** → 选 MegaMoE：它是三者中唯一跨卡的，任务重放、进度计数、环形缓冲、计算通信重叠都是系统级套路；
- 想深入**编译器/DSL 视角** → 选 HC Prenorm：TileLang 与手写 C++ 的对偶（第 47 天 3.8 节）会让你重新理解"手写 kernel 到底在控制什么"。

**易错点盘点**：

1. **MQA 的"K"不是循环维**：head_dim 是 MAD 的收缩维且只算一次；`kSplitKV=512` 切的是 **N（KV token）方向**，不要与 GEMM 的 `BLOCK_K` 混淆。
2. **MQA 的 ReLU 位置**：在点积之后、head 归约之前——不是对最终 logits 取 ReLU。
3. **MegaMoE 的任务游标是 per-core 的**：`LinearTaskState` 在每个核各自推进，"重放"靠的是相同初值 + 相同推进规则，不是共享变量。
4. **MegaMoE 的分段信息是设备侧现场生成的**：不要按第 36 天的习惯以为 psum 来自 Host。
5. **HC Prenorm 的原子加只给非 0 号 split**：split 0 的普通写必须先落地（相位 0），否则会被后来的原子加覆盖——这是两相位结构存在的原因。
6. **三个性能画像别记混**：FIX-pipe bound（MQA）/ 小 T 同步受限 + 大 T 计算受限（MegaMoE）/ 访存受限（HC Prenorm）——它们对应三组完全不同的调优方向。
7. **`kDirectStore` 的原子加与 HC 的 K-split 原子加不是一回事**：前者是单核内 L0C→GM 直写的旁路（第 26 天），后者是多核共写同一输出区域的归约手段。

## 四、源码导读

复盘日的读码目标是"把四问的结论钉回源码"。建议按下表抽查（全部文件本周或第 6 周已通读过，只需重读标注片段）：

| # | 文件 | 关键函数/对象 | 复盘核对点 |
| :---: | :--- | :--- | :--- |
| 1 | `csrc/apis/attention.hpp` | `fp8_fp4_mqa_logits` `:228-309`（config `:265-274`）、sparse `:180-226`、paged sparse `:121-175`、FIRST/REUSE 注释 `:119-120` | 3.3①的输入输出契约与硬编码配置 |
| 2 | `deep_gemm/include/deep_gemm/mqa_logits.hpp` | `mqa_logits_core_impl` `:195-`（MAD 维度 `:214-217`、`is_tile_active` `:328`、mad+ReLU `:396-404`、`reversed_pairs` `:545-554`、store 窗口 `:580-601`） | 3.3①②：单次 MAD、FIX 融合 ReLU、压缩存储 |
| 3 | `deep_gemm/include/deep_gemm/scheduler/mqa_logits.hpp` | `next_q_block` `:31-52` | 3.3②：窗口聚合公式 |
| 4 | `deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.hpp` | `make_paged_mqa_request` `:18-31`、`get_physical_page(_batch)` `:119-137` | 3.3②：页表寻址链 |
| 5 | `deep_gemm/include/deep_gemm/mega_moe.hpp` | `mega_moe_impl` `:1127-1219`（角色分支 `:1161-1215`、`kOverlapReduce` 阈值）、`aic_gemm_pipeline` `:403`、`aiv1_epilogue_comm` `:673`、`advance_dispatch_pull` `:763`、`advance_combine_push` `:1000` | 3.4①②：五段与三类角色 |
| 6 | `deep_gemm/include/deep_gemm/scheduler/mega_moe.hpp` | `BlockPhase` `:9`、`ExpertCursor` `:19-54`、`take_linear_task` `:58-118` | 3.4②：任务重放与 FRAC_MN 平衡 |
| 7 | `deep_gemm/include/deep_gemm/layout/mega_moe.hpp` | `BLOCK_M/BLOCK_N` `:12`、`ProgressStage` `:27-33`、`Workspace` `:85-`、环形池 `:255-267` | 3.4②：进度计数与缓冲复用 |
| 8 | `csrc/backends/hccl.hpp` | `HCCLSymmetricBuffer` `:29-`（`HcclCommMemReg` `:112`、`peer_ptrs`、注册池注释 `:166-171`） | 3.4②：UBMEM 直访底座 |
| 9 | `deep_gemm/mega/__init__.py` | `SymmBuffer` `:9`、`transform_weights_for_mega_moe` `:129`、`fp8_fp4_mega_moe` `:159` | 3.4①：三个公开符号 |
| 10 | `deep_gemm/tilelang_ops/hc_prenorm.py` | `get_config` `:13-58`、kernel `:61-360`（SimdVF `:196-267`、dual_copy `:269`、PerCoreTask `:295-357`）、API `:363-398` | 3.5①②：融合、K-split、相位屏障 |
| 11 | `tests/test_mega_moe.py` | `run_baseline` `:130-156`、`benchmark` `:203-243` | 3.4③④：拼装版证据与三口径公式 |
| 12 | `tests/test_attention.py` / `tests/test_hyperconnection.py` | `ref_mqa_logits` `:94`、select 选项 `:1236`；三层验证 `:59-135` | 3.3/3.5 的验证侧证据 |

**逐段导读一：MQA 的"单次 MAD + 窗口存储"闭环**（`mqa_logits.hpp:385-411` + `:580-601`）。先读 AIC 内层：`is_tile_active` 过滤后 `mad(l0c_tile, l0a[qh_tile_idx], l0b_tile, num_m_rows, issued_n, MAD_K, ENABLE_UPDATE, true)`——注意 `MAD_K` 就是 `kHeadDim`，此后**没有下一个 K 步**，紧接着 `wait_intra_block` 后 `copy_l0c_to_ub(..., infer_quant_mode<bfloat16_t, float>(), true, ...)` 把 L0C 的 FP32 部分和转成 BF16 并顺手做 ReLU，再 `set_intra_block` 通知 AIV。然后切到 AIV 侧 `:580-601`：对 tile 内每个 query 取 `ks/ke`，`store_begin = max(kv_idx, ks)`、`store_end = min(kv_idx + actual_kv, ke)`，无交叠则整行跳过，否则 `dst_col = store_begin − ks` 调 `mqa_logits_store`。两段连读，"问题①的计算"与"问题②的压缩存储"就闭合了。

**逐段导读二：MegaMoE 的任务重放**（`scheduler/mega_moe.hpp:56-118`）。先看头注释 "Replay the same routed task order on AIC and AIVs"——`take_linear_task` 是纯函数式的：同输入（`num_shared_tokens`、`num_total_m_blocks`、`core_idx`、各自的 `LinearTaskState`）必然产出同任务序列。shared 段（`:69-86`）两层 `layer_idx` 循环先发 SharedLinear1 再 SharedLinear2；routed 段（`:97-114`）的 lookahead 判断（`:99-104`）决定下一波发 Linear1 还是 Linear2——判据是"为即将到来的 Linear2 波预备好 Linear1 输入"。AIC 用它决定算哪个 GEMM，AIV1 用它决定做哪段的 epilogue/通信，两边在每个任务上会合。配合 `ExpertCursor::advance_to`（`:46-53`，`while (pool_block_idx >= pool_block_end)` 前推游标并从 GM 现场读 `expert_recv_count_sum`），"运行时才有的分段信息"就有了消费者。

**逐段导读三：HC Prenorm 的相位与屏障**（`hc_prenorm.py:295-357`）。两个 `T.PerCoreTask` 分别处理 `out_mul`（L0C→GM，走 FIX 管道）与 `sqrsum`（UB→GM，走 MTE3 管道）；flag 编号 `(chunk × phases + phase) % 4` 与 `+4` 分开两组管道的编号空间；每相位末尾的屏障包在 `for _ in T.Serial(num_store_syncs)` 里，`num_store_syncs` 是 0 或 1（`:318-322`）。读的时候问自己三个问题：phase 0 为什么只能是 split 0 写？屏障保护的不变量是什么（原子加不被普通写覆盖 + 下一 chunk 复用槽位前所有核已写完）？为什么 det 模式 `num_store_phases = num_k_splits`（固定累加顺序 → 位级可复现）。

## 五、综合练习

### A 组：有 Ascend 环境

1. **跑你选定算子的测试**（约 20 分钟，按选择取其一）：

   ```bash
   cd DeepGEMM-Ascend && ./develop.sh   # 若尚未构建
   python tests/test_attention.py prefill --skip-prof          # 选 MQA
   python tests/test_hyperconnection.py default --skip-prof    # 选 HC Prenorm
   python tests/test_mega_moe.py --num-processes 8 --skip-baseline   # 选 MegaMoE（需多卡；卡不足可试 2）
   ```

   观察与源码的对应：MQA 的用例分组（prefill/paged/sparse/paged_sparse，`tests/test_attention.py:1236`）对应 3.3 的四条路径；HC 测试开头 `Compiling` 进度条跑两轮（det 与非 det 各一批）；MegaMoE 的 `--skip-baseline` 正是 3.4③ 说的依赖分叉（baseline 需要 `deep_ep`）。若失败，按"环境 / 参考结果 / 布局 / Kernel"分层定位（第 31 天方法）。

2. **验证性能瓶颈假设**（约 15 分钟）：去掉 `--skip-prof` 重跑选定算子的一个用例，把性能行（`us / TFLOPS / GB/s` 及管道利用率字段，`tests/utils.py:106-115`）与 3.3～3.5 的"问题④"逐条对照：MQA 应见 fix 利用率显著高于 mad；HC 应见 `mad%` 低、`mte2%` 高；MegaMoE 对照 README 表（`README.md:164-173`）核对三口径。把观察记录进当天产出。

3. **（可选）跨算子对比**：用同一个视角各跑一个用例（如 MQA Decode 与 HC M=8192），体会"FIX 受限 / 访存受限"两种画像在数字上的差异（TFLOPS 都不高，但瓶颈字段完全不同）。

### B 组：无 Ascend 环境（静态任务，全部手工可完成）

1. **闭卷四问（当天核心练习，约 30 分钟）**：选定算子，合上所有材料，写四问完整回答（问题①要含公式与形状/dtype；问题②至少 3 个机制；问题③至少 2 条理由；问题④要含验证手段），再对照 3.3～3.5 对应小节逐条标红差异。

2. **三算子对比表默写**（约 10 分钟）：不看 3.2，画 3 列 × 至少 7 行的总表，对照补漏。

3. **手算（与选定算子对应，约 15 分钟；参考答案见附录）**：
   - **MQA**：某 query 行 `ks=100、ke=1000`，`sparse_block_kv B=16`，capacity=8（`sparse_kv_block_indices.size(1)`）。求：实际使用的块数；第 i 个选中块的 token 覆盖区间通式与第 0 块的区间；块 ID 从几开始（提示：`ks <= i·B + ks%B`，`csrc/apis/attention.hpp:419-421` docstring）；输出的逻辑宽度。再问：若 `page_kv=64`，一个物理页装几个 sparse 块？
   - **MegaMoE**：沿用贯穿配置 `R=2, E=8, S=0, T=128, k=2, H=I=512`，假设路由完全均匀。求：每 rank 的 expert 数与接收 token 数（`recv`）；`num_total_m_blocks`；routed Linear1 / Linear2 的任务总数（`kNumLinear1NBlocks = I/(BLOCK_N/2)`、`kNumLinear2NBlocks = H/BLOCK_N`，`BLOCK_M=BLOCK_N=256`）；core 0、core 8、core 16 前两次 `take_linear_task` 各领到什么（task_idx 与 phase，可能为 kNone）；benchmark 口径下的 flops 与远端通信字节数（`flops = 6·H·I·(recv + T·S)`，comm = 远端 token × H × 3，远端比例按 1/R 估）。
   - **HC Prenorm**：对 (m, n, k) = (2048, 24, 16384) 与 (8192, 24, 7168)，32 核，分别算 det 与非 det 的 `num_k_splits`、`cores_per_split`、`hidden blocks/split`、`num_token_blocks_per_chunk`、chunks（`get_config` 配方见第 47 天 3.3 节；`block_num_tokens=128`、`block_hidden=256`、L0C 16 槽）。

4. **"拼装版代价"量化**（约 15 分钟）：三个算子各算一笔账——MQA Prefill（Q=4096、H=32、KV=8192）若先算 `[Q·H, KV]` 打分矩阵：中间张量多大？需要额外几个 kernel（ReLU / 加权归约 / 掩码或压缩）？HC Prenorm M=8192 不融合时 A 要多读多少字节（约 469.8 MB，见第 47 天 3.1）？MegaMoE baseline 需要哪 7 个 kernel（`tests/test_mega_moe.py:207-209`）？把三笔账写成一小节，这就是问题③的"量化版"。

5. **遗留问题清理**（约 10 分钟）：把本周记录的未解决问题逐条归档：

   | 疑问（示例） | 去向 |
   | :--- | :--- |
   | FIX-pipe bound 如何用 profiling 量化、调什么 | 第 50～52 天 |
   | MegaMoE 的 msprof PIPE trace 实际长什么样 | 第 51 天（有环境时） |
   | TileLang AutoSchedule 还有哪些"博弈"模式 | 第 49 天机动或课后 |
   | MQA 稀疏寻址的更多变体（如 MFA） | 课外 |
   | K-split 原子归约的浮点顺序问题在别处怎么处理 | 对照第 47 天 det 模式即可 |

   与计划一致：明天（第 49 天）是机动日，"可补充 MoE、Attention 或 TileLang 背景知识，也可以继续补做之前的实验"。

## 六、当天产出

按计划，今天的产出是一份**扩展算子深度复盘笔记**，核心是你选定算子的四问闭卷回答，建议结构：

```text
# 扩展算子复盘笔记（第 48 天）
1. 我选择的算子：____（一句话理由）
2. 四问闭卷回答（对照 3.3～3.5 订正后的版本，差异标红）
   ① 输入输出与核心计算（公式、形状、dtype、数据通路）
   ② 调度与寻址特殊之处（≥3 个机制，各带文件:行号）
   ③ 为什么普通 GEMM 接口不够（≥2 条理由 + 一笔量化账）
   ④ 性能瓶颈假设与验证手段（口径公式 / profiling 字段）
3. 三算子对比表（3.2 默写订正版）
4. 另外两个算子的"最强一条理由"各一段
5. 手算记录（综合练习 3 对应小题）
6. 遗留问题与去向（练习 5 的表）
7. 自评结论（通过标准五个检查点：___ 个"能"）
```

有环境的同学附上 A 组第 1、2 步的输出记录。

## 七、通过标准

学习计划给当天的通过标准是：**"至少能完整讲解一个扩展算子的计算流程"**。本文把它拆成五个检查点，逐条自评"能 / 勉强 / 不能"：

| # | 检查点 | 验证方式 |
| :---: | :--- | :--- |
| 1 | 选定算子：能闭卷讲清输入输出与核心计算（公式、形状/dtype、数据通路，含源码锚点） | 综合练习 1 的问题①部分 |
| 2 | 选定算子：能讲清调度与寻址的至少 3 个特殊机制（各带文件与函数级位置） | 综合练习 1 的问题②部分 |
| 3 | 选定算子：能论证普通 GEMM 接口为何不满足（≥2 条理由，最好含一笔量化账） | 综合练习 1 的问题③ + 练习 4 |
| 4 | 选定算子：能给出性能瓶颈假设及验证手段（口径公式或 profiling 字段） | 综合练习 1 的问题④ + A 组第 2 步 |
| 5 | 三算子对比表主干可默写（数学核心 / 调度寻址 / 性能画像 / 实现方式） | 综合练习 2 |

**评判与补救**：

- **五条全部"能"**：达标，进入第 49 天机动或休息；第 8 周开始性能分析与综合实践。
- **检查点 1 或 2 "不能"**：回到对应天数的"详细讲解"节重读（MQA → 第 39/40 天；MegaMoE → 第 44/45 天；HC → 第 47 天 3.2～3.6 节），重做练习 1。
- **检查点 3 "不能"**：重读 3.3～3.5 各节的问题③，并把练习 4 的三笔账亲手算一遍。
- **检查点 4 "不能"**：回到第 46 天（MegaMoE 口径公式）与第 47 天 3.1（HC 带宽手算）；有环境则补做 A 组第 2 步。
- **检查点 5 "不能"**：只补不会的行列——paged/sparse 行回第 40 天，MegaMoE 通信行回第 45 天，TileLang 行回第 47 天 3.8 节。

## 八、自测题

1. 写出三个扩展算子的核心计算（MQA 一条公式、MegaMoE 五段链、HC Prenorm 两条公式）。
2. MQA 的 MAD 三个维度各对应什么？为什么它没有 K 维两层循环？
3. MQA 输出为什么每行只写前缀？列号怎么算？MTE3 的 32B 对齐怎么解决？
4. MQA 为什么是 FIX-pipe bound 而不是 compute bound？证据是什么？这个画像决定了调优方向是什么？
5. MegaMoE 的三类角色各做什么？"任务序列重放"是谁和谁重放、靠什么保证一致？
6. `ExpertCursor` 解决什么问题？它的输入信息从哪来、为什么说"运行时才产生"？
7. MegaMoE 在 T=64 与 T=16384 时 TFLOPS 相差近 4 倍，各自的瓶颈是什么？用什么公式/字段验证？
8. HC Prenorm 的融合省下了什么？用 M=8192 的用例给出量化。
9. HC 的 det 模式靠什么同时做到位级可复现与 batch invariance？
10. 三个算子各自"普通 GEMM 接口不满足"的最强一条理由分别是什么？
11. （开放）为什么"输入输出 / 调度寻址 / 为什么不是现成接口 / 性能瓶颈"这四问是检验"是否真正理解一个算子"的好框架？每一问分别防住了哪种"假懂"？

<details>
<summary>参考答案（先自己做再展开）</summary>

1. MQA：`logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`，`j ∈ [ks,ke)`；MegaMoE：Dispatch → Grouped GEMM1 → SwiGLU → Grouped GEMM2 → Combine（`README.md:162`）；HC：`D = float(A) @ B.T` 且 `sqr_sum = sum(float(A)**2, dim=-1)`（`hc_prenorm.py:370`）。
2. M = QH 行、N = KV token（`MAD_N = kSplitKV = 512`）、K = head_dim（`mqa_logits.hpp:214-217`）。收缩维就是 head_dim（64/128），一次 `mad` 装得下，所以没有 BLOCK_K/MAD_K 两层循环；`kSplitKV` 切的是 N 方向。
3. 每行有效区间 `[ks,ke)` 不同，只写前缀避免为无效位置写 `-inf`；`dst_col = store_begin − ks`（`mqa_logits.hpp:596`）；行头未对齐部分先经对齐 scratch 中转，再分两笔 `asc_copy_ub2gm_align`（`:46-68`）。
4. README 明说 FIX 99% 利用率（`README.md:151`）：每个 (tile, split) 都要一次 L0C→UB 的 `copy_l0c_to_ub`（带 ReLU），转换吞吐先于 cube 算力饱和。调优方向是减少 FIX 转换量/提高其吞吐，增大 MAD tile 无益。
5. AIC 跑两个 GEMM 的全部 MAD（`aic_gemm_pipeline`）、AIV0 做元数据与权重流水、AIV1 做 Dispatch 拉取/SwiGLU epilogue/Combine 推回（`mega_moe.hpp:1161-1215`）。重放发生在 AIC 与 AIV1 之间：`take_linear_task` 是纯函数式的，同输入同任务序列（`scheduler/mega_moe.hpp:56-57` 注释），两边在每个任务上会合。
6. 把 M-block 映射到非空本地 expert 并算出 `valid_m`；输入是 Dispatch 元数据写下的 `expert_recv_count_sum`（kernel 运行时由跨 rank 通信产生），对照第 36 天 M-Grouped 的 psum 由 Host 预先给定。
7. T=64：固定开销主导（元数据协议的 grid_sync/rank_barrier、权重读取），TFLOPS 228.5、通信仅 37.5 GB/s（`README.md:166`）；T=16384：846.3 TFLOPS 接近 M-Grouped 上限，回到计算/带宽受限。用 `flops = 6·H·I·(recv+T·S)` 等三口径（`test_mega_moe.py:221-231`）与 msprof trace 验证。
8. 一份 A 搬运喂两种计算：sqr_sum 在 AIV 拓宽/打包的同一趟 SIMD 里完成（`hc_prenorm.py:196-267`）；不融合则第二个 kernel 再读一遍 A——M=8192 时约多 469.8 MB（总流量约翻倍）。
9. 可复现：`num_store_phases = num_k_splits`，每个 split 在自己的相位写、相位间全核屏障，GM 累加顺序固定；batch invariance：workload 候选固定 16，split 数与 M 无关（`:33-35` 注释、`:96, :298-300`）。
10. MQA：输出是"逐元素非线性后的 head 加权归约"且每行有效区间不同，epilogue 模型表达不了（中间 `[Q,H,KV]` 张量也大到不可承受，Prefill 约 2.15 GB）。MegaMoE：分段信息运行时才产生 + 五段衔接需要 kernel 内同步（baseline 要 7 个 kernel）。HC：sqr_sum 是输入侧逐行统计量，epilogue 够不着，且小 N 大 K 需要 K-split 原子归约。
11. 开放题。参考：①防"只会背公式不懂契约"（形状/dtype/通路是可执行的细节）；②防"只会调用不懂机制"（调度寻址是 kernel 的骨架）；③防"不知道它为什么存在"（理解设计动机才能举一反三）；④防"说不出所以然的泛泛而谈"（性能画像逼你把计算、访存、流水线的账算清楚）。
</details>

## 附：综合练习 3 参考答案

**MQA（sparse）**：

- 实际使用块数 = `min(capacity, ceil((ke−ks)/B)) = min(8, ceil(900/16)) = min(8, 57) = 8`；
- 第 i 块覆盖 `[i·B + ks%B, i·B + ks%B + B) = [16i + 4, 16i + 20)`；第 0 块（指选中的第一个块）覆盖 `[100, 116)`；
- 块 ID 需满足 `ks ≤ i·B + ks%B`，即 `16i + 4 ≥ 100` → `i ≥ 6`，所以 **ID 从 6 开始**，取 6..13；
- 输出逻辑宽度 = `capacity × B = 8 × 16 = 128`（即 `[Q, 128]` 的压缩输出）；
- `page_kv=64` 时一个物理页装 `64/16 = 4` 个 sparse 块（`num_max_sparse_blocks % (page_kv/B) == 0` 的断言即此约束，`attention.hpp:94`）。

**MegaMoE**（`R=2, E=8, S=0, T=128, k=2, H=I=512`，均匀路由）：

- 每 rank expert 数 = `E/R = 4`；接收 token `recv = T·k/R = 128`（每 expert 32 个）；
- `num_total_m_blocks`：每 expert `ceil(32/256) = 1`，共 4 个 M-block；
- `kNumLinear1NBlocks = 512/(256/2) = 4`，`kNumLinear2NBlocks = 512/256 = 2`；routed 任务数：Linear1 = 4×4 = **16**，Linear2 = 4×2 = **8**；
- `take_linear_task`（32 核）：**core 0** 第一次 Linear1 task 0，第二次 Linear2 task 0；**core 8** 第一次 Linear1 task 8，第二次 Linear2 task_idx = 8 ≥ 8 → kNone（该核只做 Linear1）；**core 16** 第一次 Linear1 task_idx = 16 ≥ 16 → 跳过、Linear2 task_idx = 16 ≥ 8 → 跳过 → kNone，此后游标已推过界，一直 kNone——**只有 16 个核做 Linear1、8 个核做 Linear2**，正是小 T 下算力空转的缩影（对照问题④：T=64 时 228.5 TFLOPS）；
- flops = `6·512·512·(128 + 0) ≈ 0.20 GFLOP`；远端 token ≈ `recv·(1 − 1/R) = 64`，comm bytes ≈ `64 × 512 × 3 = 98,304 B ≈ 96 KB`。

**HC Prenorm**：

- (2048, 24, 16384)：token_blocks = `ceil(2048/128) = 16`；hidden blocks = `16384/256 = 64`，by_hidden = `64 & −64 = 64`；by_cores = 32。**非 det**：workload = `max(1, 32×16//16) = 32` → split_limit = min(32, 64, 32) = 32 → **splits = 32**，cores/split = 1，hidden/split = 2，blocks/chunk = 1×16 = 16，chunks = 1。**det**：workload 固定 16 → splits = 16，cores/split = 2，hidden/split = 4，blocks/chunk = 2×16 = 32，chunks = 1。
- (8192, 24, 7168)：token_blocks = 64；hidden blocks = 28，by_hidden = `28 & −28 = 4`。**非 det**：workload = `32×16//64 = 8` → split_limit = min(8, 4, 32) = 4 → **splits = 4**。**det**：min(16, 4, 32) = 4 → **splits = 4（两种模式相同）**——by_hidden 成为瓶颈时 det 与非 det 一致，与第 47 天 3.3 的结论互相印证。

---

明天（第 49 天）是机动或休息日：可补充 MoE、Attention 或 TileLang 背景知识，也可以继续补做之前的实验。第 8 周我们将换上性能分析的眼镜（第 50 天起：TFLOPS/带宽/算术强度、冷热缓存、单变量实验）——今天问题④里练习的"瓶颈假设 + 验证手段"，正是那一周的预习。
