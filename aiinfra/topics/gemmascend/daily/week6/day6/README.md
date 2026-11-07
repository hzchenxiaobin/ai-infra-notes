# 第 6 周 · 第 6 天（总第 41 天）：复盘 Grouped GEMM 与 MQA

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 6 周"Grouped GEMM 与 MQA Logits"的第 6 天（复盘日）教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 50 分钟，源码导读约 20 分钟，综合练习约 55 分钟，通过标准自评与自测题约 15 分钟。

本周前五天走出了 GEMM 主线的"势力范围"：第 36 天把组放在行上（M-Grouped，MoE 前向），第 37 天把组放在收缩维上（K-Grouped，MoE 反向 wgrad），第 38 天手算验证了 group 从 layout 到输出区间的映射，第 39 天进入 MQA Logits（`logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`），第 40 天读完了 Sparse 与 Paged Sparse 两个生产变体。今天是复盘日，**不引入新内容**。学习计划给今天三个任务：①对比普通 GEMM、Grouped GEMM 和 MQA 的计算模式；②总结哪些代码复用了 GEMM 主线概念、哪些属于新的调度或寻址问题；③整理一张术语表（group、paged、sparse、context、KV cache）。复盘的方法论与第 6/27/34 天相同：**闭卷复述是唯一标准**——本文 3.2～3.5 节给出四份参考答案，请先自己写再看。

## 一、今日定位与学习目标

完成今天的学习后，你应当能够：

1. 不看材料写出**四类算子的计算模式对比表**：Normal GEMM、M-Grouped、K-Grouped、MQA Logits 的数学定义、A/B/D 形状与 major 约束、调度方式、K 维处理、Epilogue 形态与边界/空组处理（参考版见 3.2 节）；
2. 把 MQA 家族细分为 normal / sparse / paged sparse 三条路径，说出它们在 **KV 来源、有效区间、输出形状、调度信息**上的差异，并对应到 Prefill 与 Decode 的分工；
3. 举出至少 8 个"**复用 GEMM 主线**"的代码证据（persistent block index、GROUP_M swizzle、`gm_ptr::index/group`、`copy_gm_to_l1` 家族、`mad()`、多级 stage 事件、AIC/AIV 分工、FIX 管转换、容量 `static_assert`、MX SF），并给出文件与函数级位置（3.3 节）；
4. 举出至少 6 个"**新增调度或寻址问题**"的代码证据（psum 组游标、D 逐组 slab 与空组清零、一维条带调度与窗口、head 归约与掩码压缩存储、`block_table` 间接寻址、FIRST/REUSE 快照、设备侧元数据 kernel），同样落到文件与函数（3.4 节）；
5. 闭卷解释术语表五个核心词条（group、paged、sparse、context、KV cache）与至少五个扩展词条（psum、request、query_mask、capacity、kv split、FIRST/REUSE）（3.5 节）；
6. 把第 38 天的通过标准再复述一遍：给定一个 MoE 输入，说清 token、group、expert 权重和输出在**前向与反向两个方向**上的对应关系（综合练习 2）。

## 二、预备知识

今天的复盘覆盖第 36～40 天全部内容，不引入新知识；哪一条卡住，就按下面的索引回到对应天数补漏：

- **第 17～20 天**：Normal 调度（`get_next_block`、GROUP_M swizzle、`gm_ptr::index`、`copy_gm_to_l1`）——3.3 节"复用盘点"的证据全部来自这条主线。
- **第 22～26 天**：K 维两层循环、`mad()` 与 L0C 累加、AIC/AIV 分工与 Epilogue——对比"MQA 没有 K 循环"时全靠它们做参照。
- **第 32～33 天**：FP8/FP4 与 UE8M0/MX scale（`MX_SF_DIVISOR = 64`）——MQA 与低精度分组 GEMM 的输入都是 MX 布局。
- **第 36 天**：M-Grouped 的 psum 行布局、组游标、"只有 B 拿组偏移"、BLOCK_M 钉死 256。
- **第 37 天**：K-Grouped 的 psum-over-K、共享 M×N 网格、空组清零与 accumulate 语义。
- **第 38 天**：group → 输出区间的手算配方——综合练习 2 直接复用。
- **第 39 天**：MQA Logits 公式、AIC/AIV 分工、窗口与压缩存储——综合练习 3 直接复用。
- **第 40 天**：sparse 选块、paged 页表寻址、FIRST/REUSE 快照、Prefill/Decode 差异。

> 推荐先花 5 分钟把第 37 天末尾的"三种 GEMM 类型对比表"和第 39 天的 MQA 流程图默画一遍——今天 3.2 节的总表就是它们的合并扩展版。另注：第五节 A 组任务需要 Ascend 环境（Ascend 950 系列 + CANN 9.20，见仓库 `README.md` 的 Requirements）；无环境时完成 B 组静态任务即可达到当天目标。

## 三、知识梳理

### 3.1 本周五天一览

| 天 | 主题 | 一句话结论 | 当天产出 |
| :---: | :--- | :--- | :--- |
| 36 | M-Grouped GEMM | 组在行上：A/D 是平坦缓冲，B 逐组一份 `[G,N,K]`；调度器在 Normal 的 M×N 网格外套一层组游标 | 输入输出示意图 |
| 37 | K-Grouped GEMM | 组在收缩维上：A/B 是平坦的 K 段拼接（tn、MN-major），D 逐组一份 `[G,M,N]`；所有组共享 M×N 网格，只有 K 循环长度不同 | 三种 GEMM 对比表 |
| 38 | 运行或推演 | 手算 group → 输出区间映射；MoE 双向对应关系 | 测试记录 / 推演笔记 |
| 39 | 普通 MQA Logits | 一维条带调度 + FIX 管 ReLU + head 加权归约 + `[ks,ke)` 掩码压缩存储 | MQA 计算流程图 |
| 40 | Paged/Sparse MQA | 选块（sparse）+ 分页（paged）+ FIRST/REUSE 快照；元数据 kernel 在设备侧排产 | 元数据与数据流图 |

### 3.2 计算模式对比：普通 GEMM vs Grouped GEMM vs MQA

先闭卷画你自己的版本，再对照下表（这是今天产出 ① 的参考版）：

| 维度 | Normal GEMM | M-Grouped | K-Grouped | MQA Logits |
| :--- | :--- | :--- | :--- | :--- |
| 数学定义 | `D[m,n] = Σ_k A[m,k]·B[n,k]` | 同左，但行段属于不同组：`D[ms,:] = A[ms,:] · B[g]^T`（ms 为组 g 的有效行） | `D[g,m,n] = Σ_{k∈K_g} A[k,m]·B[k,n]`，每组各得一份输出 | `logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`，仅 `j ∈ [ks_q, ke_q)` 有效 |
| A（左操作数） | `[M,K]` 任意 major | `[M,K]` 平坦，**必须 K-major**（`csrc/apis/gemm.hpp:90`） | `[K,M]` 平坦（tn），**必须 MN-major**（`:94`） | Q `[Q,H,D]`（FP8/FP4），H∈[4,64] 且 4\|H，D∈{64,128}（`csrc/apis/attention.hpp:254-256`） |
| B（右操作数） | `[N,K]` 任意 major | `[G,N,K]`（nt）/`[G,K,N]`（nn），**逐组一份** | `[K,N]` 平坦（tn、MN-major） | KV `[KV,D]` 单头共享（MQA），同 dtype |
| D（输出） | `[M,N]` 平坦 | `[M,N]` 平坦 | `[G,M,N]` **逐组一份** | logits `[Q, max_seqlen_k]` bf16，行距 512B 对齐，每行只写有效前缀 |
| 分组/区间信息 | 无 | `grouped_layout [G]` int32（行 psum） | `grouped_layout [G]` int32（K psum）+ 可选 `ks_cpu` Host 提示 | `cu_seq_len_k_start/end [Q]`（普通/sparse）；`context_lens [Q]` + `indices [Q]`（分页） |
| 调度 | 二维 M×N 网格，persistent block index + GROUP_M swizzle + N-first snake（`scheduler.hpp:14-15`） | 同款网格套组游标：组内走同一 swizzle，`m_block_idx += last_psum_m / BLOCK_M` 平移到全局（`scheduler.hpp:159-161`） | 所有组共享同一个 `num_blocks` 网格，组游标只负责推进 K 循环参数（`:163-177`） | 一维条带：核 `core_id` 从 `current_q_block_idx` 出发、`+= kNumCores` 跨步轮询 q_block（`scheduler/mqa_logits.hpp:23, :35-36`），q_block 内按 `kSplitKV=512` 切 kv_split |
| K 维处理 | 两层循环 `BLOCK_K / MAD_K`，部分和累加进同一块 L0C | 与 Normal 完全相同 | 每组 `eff_shape_k = current_shape_k`，`k_idx = k_idx_base + k_blk_idx·BLOCK_K`（`bf16_gemm.hpp:95-97, :113`） | **没有 K 循环**：`MAD_K = kHeadDim`，一次 `mad` 完成（`mqa_logits.hpp:217, :396-397`） |
| 组/页偏移落点 | 无（batch 偏移除外） | 只有 B：`gm_b.group(b_group_idx, ...)`（`bf16_gemm.hpp:100-103`） | 只有 D：`gm_d.group(d_group_idx, shape_m, shape_n)`（`:157, :186-188`） | 无 group 概念；paged 版用 `block_table` 把逻辑页映射到物理页 |
| Epilogue | bias / alpha / accumulate + dtype 转换（bias 与 alpha 仅 Normal，`gemm.hpp:107-117`；accumulate 见下行 K-Grouped） | 不支持 bias/alpha/accumulate | 支持 accumulate：`D[g] = A_g @ B_g + C[g]`，AIV drain 变 atomic add（`epilogue.hpp:89-96`） | FIX 管 ReLU（`mqa_logits.hpp:400-404`）→ AIV head 加权归约（`:101-151`）→ 窗口掩码压缩存储（`:584-601`） |
| 边界与空组 | `actual_m/n/k` 截断 | `get_actual_m` 恒返回 BLOCK_M（`scheduler.hpp:127-136` 的 TODO），正确性靠 padding 语义 | 非对齐组尾由 `actual_k` 截断；**空组由 AIV 显式清零**（`epilogue.hpp:108-141`） | `is_tile_active` 跳过与窗口无交叠的 tile；`store_begin/end` 按窗口裁剪（`mqa_logits.hpp:590-594`） |
| 典型场景 | 稠密前向/反向 | MoE 前向（token×expert 权重） | MoE 反向 wgrad | DeepSeek Lightning Indexer（README"MQA Logits"节） |

三点补充说明：

- **轴对称关系**：M-Grouped 与 K-Grouped 是同一机制的两次实例化——"把不规整的逻辑维度（组）压进一根物理轴（M 或 K），用 psum 数组描述边界"。对偶地，M-Grouped 是"分组维在输出保留维上"（B 逐组），K-Grouped 是"分组维在收缩维上"（D 逐组）；两者的 major 断言因此恰好相反（K-major vs MN-major）。`GemmType` 枚举把这四种形态并列（`common.hpp:87-92`；`Batched` 由 einsum API 使用，见 `csrc/apis/einsum.hpp:22`，不在本周范围）。
- **MQA 不是 GEMM 的分组变体，而是"乘加 + 逐元素非线性 + 输出侧缩减"的复合算子**。它的 MAD 三个维度有明确对应：M = QH 行（`MAD_M` 装 query×head）、N = KV token（`MAD_N = kSplitKV`）、K = head_dim（`mqa_logits.hpp:214-217`）。README 特别指出该 kernel **是 FIX-pipe bound 而非 compute bound**（"saturating the FIX pipe at 99% utilization"）——这与 Dense GEMM 的算力受限（BF16 达 99.8% 硬件上限）是两种完全不同的性能画像，调优方向也不同。
- **MQA 家族三条路径**的差异（对应第 39/40 天）：

| 维度 | normal（`fp8_fp4_mqa_logits`） | sparse（`fp8_fp4_sparse_mqa_logits`） | paged sparse（`fp8_fp4_paged_sparse_mqa_logits`） |
| :--- | :--- | :--- | :--- |
| KV 来源 | 连续 token-major `(kv[KV,D], sf[KV,D/64])` | 同左，但按选中块 gather | paged cache `[num_pages, page_kv, 1, token_bytes]`，fused 页 `[page DATA \| page SF]` |
| 有效 token | 每行窗口 `[ks, ke)` | 前 `min(capacity, ceil((ke−ks)/B))` 个选中块（B = `sparse_block_kv` ∈ {8,16}） | 每请求 `context_lens` 内的选中块 |
| 输出形状 | `[Q, max_seqlen_k]`，行前缀压缩 | `[Q, capacity×B]` | `[Q, capacity×B]` |
| 调度信息 | 运行时 `MQALogitsScheduler` | 元数据 kernel 排产（`ScheduleBoundary`） | 元数据 kernel 排产（`SplitKVRange` + 静态 `SplitKV` 记录） |
| 组织单位 | q_block（固定 token 数） | QBlock（4 query） | 请求（`indices` 标识，QBlock 不跨请求切割） |
| 跨调用复用 | 无 | 无 | FIRST/REUSE 快照：prepare + FIRST + (L−1)×REUSE |
| 场景 | 通用 | Prefill | Decode（多层解码链路） |

### 3.3 代码复用盘点：哪些是 GEMM 主线的"老朋友"

这是今天产出 ② 的前半部分。判断"复用"的标准：**同一个函数/机制在主线 Kernel（`bf16_gemm.hpp`）里存在，本周算子直接调用或仅换参数**。

| # | 主线概念（锚点） | 在 Grouped GEMM 的落点 | 在 MQA 家族的落点 |
| :---: | :--- | :--- | :--- |
| 1 | persistent block index `(++current_iter)×kNumCores + core_id`（`scheduler.hpp:142`） | 原样使用；组游标"只前推不回退"的正确性正是靠它单调递增（注释 `scheduler.hpp:33-35`） | 改为 `current_q_block_idx += kNumCores`（`scheduler/mqa_logits.hpp:36`），思想相同：每核跨步轮询 |
| 2 | GROUP_M swizzle + N-first snake，`get_swizzled_block`（`scheduler.hpp:202-248`） | M-Grouped 在**组内**复用同一 swizzle（`:159-160`）；K-Grouped 直接复用（`:179`） | 无二维网格，不使用 |
| 3 | `gm_ptr::index()` 双 major 寻址（`common.hpp:156-165`） | K-Grouped 平坦 A/B 的 `k_idx` 平移全靠它（`bf16_gemm.hpp:113`） | Q/KV 的 `q_t.index(q_base×kNumHeads, 0)` 等（`mqa_logits.hpp:294, :307`） |
| 4 | `gm_ptr::group()/batch()` 组/批偏移（`common.hpp:167-172, :181-185`） | M-Grouped 的 B（`bf16_gemm.hpp:103`）、K-Grouped 的 D（`:157, :186-188`） | 无 group 概念 |
| 5 | `copy_gm_to_l1` / `copy_l1_to_l0a/b` 搬运主干（`ascend/copy.hpp`） | 原样 | 原样，另加 `_mx` 后缀的 scale 装载版本（`copy_gm_to_l1_mx` / `copy_l1_to_l0a/b_mx`，`mqa_logits.hpp:296, :322-323`） |
| 6 | `mad()` + L0C FP32 累加（`bf16_gemm.hpp:64, :149-151`） | 原样；两组各自 `is_first/is_last` 重新开始累加 | `mqa_logits.hpp:396-397`，单次 MAD、`ENABLE_UPDATE` |
| 7 | 多级 stage + `asc_sync_wait/notify` 满空事件（`ascend/sync.hpp`） | 原样 | Q/KV/accum/logits 各自多级（声明与容量检查 `mqa_logits.hpp:231-262`）；AIC↔AIV 用 `set/wait_intra_block` 跨子块握手 |
| 8 | AIC/AIV `__mix__(1, 2)` 分工 | 原样：AIC 主循环 + AIV drain | 同款 1 AIC + 2 AIV：AIC 搬运+MAD，AIV reduce+store；sparse 里 AIV 还承担 gather 与 NZ 转换 |
| 9 | FIX 管 `copy_l0c_to_ub` + 量化模式 | 原样（dtype 转换） | ReLU 融合进 FIX 搬运（`mqa_logits.hpp:400-404`，`infer_quant_mode<bfloat16_t, float>()` + relu=true） |
| 10 | 容量 `static_assert`（第 16 天的方法） | 原样 | `mqa_logits.hpp:254-262` 一排 L1/L0A/L0B/L0C/UB 断言 |
| 11 | MX SF（UE8M0 打包 int16、`MX_SF_DIVISOR = 64`，`common.hpp:65`） | fp8/fp4 分组 GEMM 原样 | Q/KV 的 sf 走 `_mx` 装载；paged 版 AIV 先 `vf_mqa_logits_pack_scales` 打包再送 L1（`mqa_logits.hpp:11-31, :470-487`） |
| 12 | 边界截断配方 `min/aligned/ceil_div`（`common.hpp:41-48`） | `actual_k` 截组尾（`bf16_gemm.hpp:114`） | `actual_kv`、`issued_n = aligned(actual_kv, 128)`（`mqa_logits.hpp:344-346`） |

一句话总结：**"搬运 → L0 → MAD → FIX → AIV 写回"这条数据流主干，以及支撑它的寻址、同步、容量检查设施，被本周所有算子原样继承**；第 39 天教程说的"MQA 仍然用 MMAD 做矩阵乘"在代码层面就体现为上表的第 3～9 行。

### 3.4 新增问题盘点：哪些是新的调度或寻址问题

今天产出 ② 的后半部分。判断"新增"的标准：**主线 GEMM 没有对应机制，必须新写调度状态、寻址换算或元数据**。

| # | 新问题 | 所属算子 | 关键源码 | 一句话本质 |
| :---: | :--- | :--- | :--- | :--- |
| 1 | psum 布局与组游标 | M/K-Grouped | `scheduler.hpp:62-77`（游标状态）、`:144-177`（两个分支） | 把组边界编码进一个 int32 数组，运行时用单调前推的游标解码出"当前 block 属于哪组、组内有多少 m-block / K 多长" |
| 2 | 逐组偏移与配置钉死 | M-Grouped | `bf16_gemm.hpp:100-103`；`csrc/jit_kernels/heuristics.hpp:104-106, :610-618`；`csrc/jit/jit.hpp:59` | B 的组偏移 + `BLOCK_M` 被布局对齐粒度 `kMkAlignment = 256` 钉死（搜索区间 `m_range = {256, 256}`） |
| 3 | 空组输出与累加语义 | K-Grouped | `epilogue.hpp:108-141`（`aiv_zero_empty_groups`）、`:89-96`（atomic add） | 空组领不到任何 block，输出必须由 AIV 显式清零；accumulate 时 AIV drain 变 atomic add，空组保持 `D[g] = C[g]` 恰好就是期望 |
| 4 | 一维条带调度与窗口 | MQA normal | `scheduler/mqa_logits.hpp:31-52`（`next_q_block`） | 任务从二维网格退化成 (q_block, kv_split) 条带；窗口 `[start, end)` 由块内所有 query 的 ks/ke 取 min/max 得到 |
| 5 | head 维归约 | MQA 全家族 | `mqa_logits.hpp:101-151`（`vf_mqa_logits_reduce`） | GEMM 没有的输出侧缩减：`Σ_h score·w`，在 AIV 的 V 管上做 |
| 6 | 掩码压缩存储 | MQA 全家族 | `mqa_logits.hpp:45-68`（`mqa_logits_store`）、`:584-601` | 每行只写有效前缀、列号 = 原位置 − ks；MTE3 要求 32B 对齐 UB 源，行头经对齐 scratch 中转 |
| 7 | 块选择与并集装载 | sparse 家族 | `csrc/apis/attention.hpp:59-80`；`sparse_mqa_logits_metadata.hpp` | 第 i 块覆盖 `[i·B + ks%B, ...+B)`；4 个 query 的有序块列表做 sorted union + `query_mask`，共享块只装载一次 |
| 8 | 分页间接寻址 | paged 家族 | `scheduler/paged_mqa_logits.hpp:119-137` | `block_table[query 行, 逻辑页]` → 物理页 → `token_start = 物理页×page_kv + 页内偏移`；物理页允许乱序 |
| 9 | 物理页乱序的顺序恢复 | paged（非 sparse） | `mqa_logits.hpp:545-554`（`reversed_pairs` 计算）、`:73-99`（`vf_mqa_logits_store_bf16`） | gather 按物理地址序成对搬运，AIV 存储时按 `reversed_pairs` 位图把成对页交换回来，恢复逻辑页序 |
| 10 | FIRST/REUSE 快照 | paged sparse | `paged_sparse_mqa_logits.hpp`；`deep_gemm/__init__.py:87-135` | FIRST 把 gather+NZ 转换后的 KV 物化进 metadata 的 compact 区；REUSE 直接整页 GM→L1；状态由 Python 侧挂载，改 KV 必须重建 metadata |
| 11 | 设备侧元数据/排产 kernel | sparse 家族 | `sparse_mqa_logits_metadata.hpp`、`paged_sparse_mqa_logits_metadata.hpp`、`scheduler/paged_mqa_logits_metadata.hpp` | 把"按代价把任务切给核"从 Host 搬到 Device 一次性完成，主 kernel 只消费静态计划 |

对照着看会发现一个规律：**Grouped GEMM 的新问题集中在"调度状态"（组游标、空组），MQA 的新问题集中在"寻址与输出语义"（页表、选块、窗口、归约、快照）**——这正好回答了计划任务②的后半句。

### 3.5 术语表：group、paged、sparse、context、KV cache

这是今天产出 ③ 的参考版。五个核心词条先给"标准表述"，扩展词条用速查表。

**group（组）**：Grouped GEMM 中"共用一份权重（M-Grouped）或共得一份输出（K-Grouped）"的元素集合；MoE 里一个 group 就是一个 expert。设备侧由 `grouped_layout`（psum 数组，`[G]` int32 contiguous）描述：`grouped_layout[i]` 是前 i+1 组**实际规模**的累积结尾，组 g 的有效区间为 `[align(psum[g-1], 256), psum[g])`、组 0 从 0 开始，总规模 `M = align(psum[G-1], 256)`。Kernel 通过 `scheduler.get_group_idx()` 拿到当前 block 的组号。锚点：`scheduler.hpp:17-31`（注释）、`tests/generators.py:212-220`（`generate_psum_layout`）。

**paged（分页）**：KV cache 不按 token 连续存放，而是切成 `page_kv ∈ {64, 128}` 的页；逻辑页号经 `block_table`（`[Q, max_pages]` int32，行数 = query 数）映射到物理页，物理页允许乱序（测试用 `randperm` 打乱）。fused 页布局为 `[page DATA | page SF]`（数据在前、scale 在后，使 DATA 段保持 `token_bytes` 等距）。寻址链：逻辑块 → 逻辑页 `block_idx / (page_kv/B)` → 物理页 `block_table[row, logical_page]` → 物理 token 起点 `物理页 × page_kv + 页内偏移`。锚点：`scheduler/paged_mqa_logits.hpp:119-137`、`csrc/apis/attention.hpp:363-368`。

**sparse（稀疏）**：每个 query 只计算自己选中的 top-k 个 KV 块（块大小 `sparse_block_kv` B ∈ {8, 16}），第 i 个块覆盖 token 区间 `[i·B + ks%B, i·B + ks%B + B)`；每个 query 取前 `min(capacity, ceil((ke−ks)/B))` 个块（capacity = `sparse_kv_block_indices.size(1)`）。输出压缩为 `[Q, capacity×B]`。相邻 query 的选择高度重叠（测试用 overlap 0.8/0.9 模拟），元数据 kernel 据此做 sorted union 合并装载。锚点：`csrc/apis/attention.hpp:414-429`（docstring）、`deep_gemm/include/deep_gemm/sparse_mqa_logits_metadata.hpp`。

**context（上下文）**：一个请求已积累的 KV 长度。Decode 侧是 `context_lens [Q]` int32；普通/sparse MQA 用等价的每行窗口 `cu_seq_len_k_start/end [Q]`（`[ks, ke)`）表达。分页调度里 `make_paged_mqa_request` 取 `context_lens[q + count - 1]`——同请求最后一个 query 的上下文——代表整个请求的 KV 范围。锚点：`scheduler/paged_mqa_logits.hpp:14-31`。

**KV cache**：推理时缓存的历史 K/V 张量。MQA 中 K/V 共享单头，kernel 只见到一个 `[KV, D]` 矩阵（normal/sparse，token-major，DATA 与 SF 分开）或一张分页 fused cache `[num_pages, page_kv, 1, token_bytes]`（paged，`token_bytes = data_bytes + 4`）。paged sparse 的 FIRST/REUSE 机制会在 metadata 里物化一份"快照 KV"（compact 区），REUSE 读快照而不读当前 cache。锚点：`csrc/apis/attention.hpp:331-410`（`fp8_fp4_paged_mqa_logits` 的 fused_kv_cache 契约）、README"MQA Logits"节。

扩展词条速查表：

| 词条 | 代码名 | 一句话 | 锚点 |
| :--- | :--- | :--- | :--- |
| psum | `grouped_layout` | 组规模的累积结尾数组，"align 到 256 后再拼下一组" | `generators.py:212-220` |
| request（请求） | `indices [Q]` | Decode 侧同 ID 连续的 query 归属的请求；QBlock/页表契约都以请求为单位 | `scheduler/paged_mqa_logits.hpp:18-31` |
| query_mask | `ub_query_masks` → `masks`/`control` | 4 位掩码，第 i 位 = QBlock 内 query i 是否选中当前并集块；一路传到 AIV 的选列/散射 | `sparse_mqa_logits_metadata.hpp:46-51` |
| capacity | `num_max_sparse_blocks` | 每 query 最多选中的块数；`capacity×B < 65536`（16 位输出偏移） | `csrc/jit_kernels/sparse_mqa_logits.hpp:23` |
| kv split | `kSplitKV = 512` | KV 维的分段粒度，MAD 的 N；每段一次装载一次 MAD | `mqa_logits.hpp:207, :216` |
| FIRST/REUSE | `metadata._paged_sparse_mqa_state` | 首次调用物化 KV 快照、之后复用的两层状态机 | `deep_gemm/__init__.py:87-135` |
| MQA | Multi-Query Attention | K/V 单头共享的多查询注意力；本仓库只算它的 logits（打分）一步 | README、`attention.hpp:228-309` |
| Lightning Indexer | — | DeepSeek 的轻量索引器，MQA Logits 的服务对象；Prefill/Decode 两类用例即来自它 | README"MQA Logits"表 |
| expert / MoE | `B[g]` | 混合专家；M-Grouped 的 group 前向对应 expert 权重，K-Grouped 反向对应权重梯度 | README"M-Grouped GEMM"表 |
| 排产（schedule） | `ScheduleBoundary` / `SplitKVRange` | 元数据 kernel 写下的"每个核从哪干到哪"，主 kernel 免运行时搜索 | `sparse_mqa_logits_metadata.hpp:276-289` |

### 3.6 易错点盘点

1. **分组轴别记反**：M-Grouped 分组在 M（B 逐组、A/D 平坦），K-Grouped 分组在 K（D 逐组、A/B 平坦）——"谁逐组"与"分组维在哪"恰好相反。
2. **psum 是累积结尾不是组大小**：`grouped_layout = [3, 259]`（对齐 256）表示组 0 有 3 行、组 1 有 3 行（从 256 起），组 1 的起点是 `align(3, 256) = 256` 而不是 3。
3. **M-Grouped 的 `get_actual_m` 恒返回 BLOCK_M**（`scheduler.hpp:127-136` 的 TODO），组尾 padding 靠调用方语义（`ensure_zero_padding`），与 Normal 的截断机制不同。
4. **major 断言方向相反**：M-Grouped 要求 A 为 K-major（`gemm.hpp:90`），K-Grouped 要求 A、B 都为 MN-major（`:94-95`）——根源是 tn 的"逻辑转置"技巧。
5. **MQA 的"K"不是循环维**：head_dim 是 MAD 的收缩维且只做一次；`kSplitKV=512` 切的是 **N（KV token）方向**，不要与 GEMM 的 `BLOCK_K` 混淆。
6. **压缩输出的列号是行内的**：`dst_col = 原位置 − ks`，每行各自从 0 开始；不同 query 的有效列不会互相挤占。
7. **`block_table` 的行数是 Q 不是请求数**：契约要求同请求 query 对引用页达成一致，Host 直接按 query 行展开（测试 `repeat_interleave`）。
8. **REUSE 只认快照**：FIRST 之后修改原始 KV 内容不影响 REUSE 结果；改 KV 必须重建 metadata（`attention.hpp:119-120` 注释、测试三条断言）。
9. **性能画像不同**：Dense/M-Grouped GEMM 是算力受限（README：BF16 99.8%、M-Grouped FP8×FP4 862 TFLOPS），MQA 是 FIX-pipe bound（99% FIX 利用率）——"多开 MAD"对 MQA 无益，瓶颈在 L0C→UB 的转换吞吐。
10. **bias/alpha/accumulate 是 Normal 独享，K-Grouped 额外享有 accumulate**（`gemm.hpp:107-117`、`tests/common.py:56`）；M-Grouped 两者皆无。

## 四、源码导读

复盘日的读码目标是"把 3.2～3.4 节的结论钉回源码"。建议按下表抽查（每行只需重读标注的片段，全部文件本周都已通读过）：

| # | 文件 | 关键函数/对象 | 复盘核对点 |
| :---: | :--- | :--- | :--- |
| 1 | `deep_gemm/include/deep_gemm/scheduler.hpp` | 头注释 `:10-41`、`get_next_block` `:141-196`、`get_swizzled_block` `:202-248` | 3.2 表"调度"行：三个分支的组游标差异 |
| 2 | `deep_gemm/include/deep_gemm/common.hpp` | `GemmType` `:87-92`、`gm_ptr::index/group/batch` `:156-185`、`MX_SF_DIVISOR` `:65` | 3.3 表第 3/4/11 行 |
| 3 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | AIC 分支 `:81-172`（K 循环 `:112-118`、mad `:149-151`）、组偏移 `:95-103`、AIV 分支 `:173-192` | 3.3 表第 5/6 行 + 3.4 表第 2/3 行 |
| 4 | `deep_gemm/include/deep_gemm/epilogue.hpp` | `aiv_initialize` `:89-96`、`aiv_zero_empty_groups` `:108-141` | 3.4 表第 3 行（空组与累加） |
| 5 | `csrc/apis/gemm.hpp` | 形状注释 `:45-58`、M/K 分支 `:88-96`、Normal 独享检查 `:107-117`、`estimated_m/k` `:157-163` | 3.2 表形状与 major 行 |
| 6 | `csrc/jit_kernels/heuristics.hpp` | `:104-106`（BLOCK_M 钉死）、`:610-618`（m-grouped 搜索区间） | 3.4 表第 2 行 |
| 7 | `deep_gemm/include/deep_gemm/mqa_logits.hpp` | `mqa_logits_core_impl` `:195-613`（AIC `:269-414`、AIV `:415-612`）、`vf_mqa_logits_reduce` `:101-151`、`mqa_logits_store` `:45-68`、两个入口 `:623-675` | 3.3 表第 5～12 行 + 3.4 表第 4/5/6/9 行 |
| 8 | `deep_gemm/include/deep_gemm/scheduler/mqa_logits.hpp` | `MQALogitsScheduler` `:7-74`（`next_q_block` `:31-52`） | 3.4 表第 4 行（窗口聚合） |
| 9 | `deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.hpp` | `make_paged_mqa_request` `:18-31`、`get_physical_page` `:119-123`、`get_physical_page_batch` `:125-134` | 3.4 表第 8 行（页表寻址） |
| 10 | `csrc/apis/attention.hpp` | `fp8_fp4_mqa_logits` `:228-309`（config `:265-274`）、`get_paged_mqa_logits_metadata` `:311-329`、sparse 两个入口 `:59-80, :180-226`、FIRST/REUSE 注释 `:119-120` | 3.2 MQA 细分表 |
| 11 | `deep_gemm/__init__.py` | `get_paged_sparse_mqa_logits_metadata` `:87-112`、`fp8_fp4_paged_sparse_mqa_logits` `:115-135` | 3.4 表第 10 行（状态机） |
| 12 | `tests/common.py` + `tests/generators.py` + `tests/test_attention.py` | `gemm_dg` `:49-84`、`generate_psum_layout` `:212-220`、`get_available_trans_ab` `:290-299`、`ref_mqa_logits` `:94`、`run_mqa_logits_case` `:138` | 3.2 表"典型场景"行的测试侧证据 |

**逐段导读一：`get_next_block` 的三个分支对照**（`scheduler.hpp:141-196`）。先读 Normal 分支（`:190-195`）：`block_idx` 越界即返回 false，否则 swizzle 映射。再看 M-Grouped 分支（`:144-162`）：`while (true)` 不断前推组游标（`current_group_idx`、`last_psum_m = aligned(current_psum_m, kAlignment)`、`current_m_block_cumsum += num_m_blocks`），直到 `block_idx` 落入当前组的 block 范围；然后**在组内**走同一个 `get_swizzled_block`，最后 `m_block_idx += last_psum_m / BLOCK_M` 把组内编号平移到全局。K-Grouped 分支（`:163-177`）结构相同，但所有组共享 `num_blocks`（M×N 网格不改写），游标推进的是 K 参数（`current_shape_k`、`last_psum_k`、`current_sf_k_cumsum`），且有 `current_shape_k > 0` 的空组守卫。三个分支共用同一个 `block_idx` 公式——这就是 3.3 表第 1 行说的"单调性论证"（`:33-35` 注释）的用武之地。

**逐段导读二：MQA 的 AIC 分支与 `bf16_gemm_impl` 的 AIC 分支逐行同构**（`mqa_logits.hpp:269-414` vs `bf16_gemm.hpp:81-172`）。对照着读：`asc_set_mmad_direction_n()` + `set_flags`（mqa `:270-280` ↔ bf16 `:82-84`）→ 外层循环领任务（`next_q_block` ↔ `get_next_block`）→ 搬 A 侧（Q 连同 MX scale 进 L1/L0A，mqa `:291-325` ↔ bf16 `:110, :117`）→ 内层循环搬 B 侧（KV 进 L1 再 L0B，mqa `:343-382` ↔ bf16 `:112-146`）→ `mad`（mqa `:396-397` ↔ bf16 `:149-151`）→ `copy_l0c_to_ub`（mqa `:400-404` ↔ bf16 经 epilogue 的 `aic_store`）。差异浓缩在三点：KV 循环下标是 `kv_split_idx` 而不是 `k_blk_idx`（N 方向分段，不是 K 累加）；每组 (tile, split) 只有一次 `mad`（MAD_K = head_dim）；FIX 搬运带 `relu=true`。把这两段并排读完，3.3 节的"复用"与 3.4 节的"新增"就不再是两张清单，而是同一段代码里的两种颜色。

**逐段导读三：Paged 寻址三件套与顺序恢复**（`scheduler/paged_mqa_logits.hpp:119-137` 与 `mqa_logits.hpp:545-554, :73-99`）。`get_physical_page`（`:119-123`）：逻辑块号换算逻辑页与页内偏移，查 `block_table[request.q_token_start × stride + logical_page_idx]` 得物理页。`get_physical_page_batch`（`:125-134`）：把相邻两个逻辑页打包成一次搬运——`first_page = min(page0, page1)`、`page_gap = |page0 − page1|`、`num_batch = min(2, ...)`，AIC 侧据此发起一次 `copy_gm_to_l1(..., num_batch, kPageSize, page_gap × kv_page_stride_bytes)`（`mqa_logits.hpp:353-362`）。物理序与逻辑序相反时（`page0 > page1`），AIV 在 reduce 前算出 `reversed_pairs` 位图（`:548-553`），`vf_mqa_logits_store_bf16<kPageSize>` 存储时按位交换两个半区/相邻向量（`:73-99`）——gather 赚到连续 burst，存储负责把逻辑顺序换回来，两头都不亏。

## 五、综合练习

### A 组：有 Ascend 环境

1. **复跑分组测试并核对配置钉死**：

   ```bash
   cd DeepGEMM-Ascend
   DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py mgrouped --skip-prof 2>&1 | grep -m5 GemmConfig
   DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py kgrouped --skip-prof 2>&1 | grep -m5 GemmConfig
   ```

   预期：mgrouped 打印的所有配置 `block_m` 恒为 256（3.4 表第 2 行），kgrouped 的 `compiled_dims` 为 `mn`（K 运行时可知）。任选一个 kgrouped shape，把它打印出的 `block_k/mad_k` 代入第 22 天的两层循环框架，心算一组的 K 步数。

2. **跑 attention 全家并标注路径**：

   ```bash
   cd tests
   python test_attention.py all --skip-prof
   ```

   对照 3.2 节 MQA 细分表，在心里给每个 case 标注走的是 normal / sparse / paged / paged_sparse 哪条路径、Prefill 还是 Decode 参数（可参考 `run_paged_sparse_mqa_logits_perf_case`，`tests/test_attention.py:451`）。

3. **（可选）验证两种性能画像**：去掉 `--skip-prof` 重跑一个 mgrouped case 与一个 MQA case，把 TFLOPS/GB/s 与 README 的"M-Grouped GEMM"表、"MQA Logits"表对比；MQA 的 pipe profile 里 `fix` 利用率应显著高于 `mad`（README：FIX 99%）。

### B 组：无 Ascend 环境（静态任务，全部手工可完成）

1. **闭卷默写 3.2 主表**（约 15 分钟）：不看正文，画一张 4 列（Normal / M-Grouped / K-Grouped / MQA）× 至少 8 行（数学定义、A/B/D、分组信息、调度、K 维、组偏移、Epilogue、边界空组）的对比表，再对照补漏。

2. **MoE 双向手算**（约 20 分钟，第 38 天通过标准的复演）：设 2 个 expert（G=2）、6 个 token，路由结果为 expert 0 ← {t0, t3, t5}、expert 1 ← {t1, t2, t4}（每组 3 个 token），对齐度 256、`expected_m = 3`。

   (1) 前向（M-Grouped）：写出 `grouped_layout`（psum）、总 M、两组的有效行区间与 padding 区间；若 `BLOCK_M = 256`，写出每个组占用的全局 m-block 编号区间（提示：组 1 的 `in_group` m-block 经 `m_block_idx += last_psum_m / BLOCK_M` 平移）；t5 的输出落在 D 的哪一行？它乘的是哪个 expert 的权重？
   (2) 反向 wgrad（K-Grouped）：沿用同一个 psum 描述 K 维，写出 A `[K, M]`、B `[K, N]`、D `[G, M, N]` 的语义与两组的有效 K 区间；组 1 的 `k_idx_base` 是多少？t5 的梯度贡献累加进 D 的哪个 slab？
   (3) 一句话回答：为什么前向"B 逐组、D 平坦"而反向"D 逐组、A/B 平坦"？

3. **MQA 调度手算**（约 15 分钟）：设 H=32、D=128、`kSplitKV=512`、`MAD_M=64`（则 `kNumQTokensPerBlock = 8×64/32 = 16`、`kNumQTokensPerTile = 64/32 = 2`），Q=20。第二个 q_block（q16～q19）的窗口分别为 q16 `[100,300)`、q17 `[100,600)`、q18 `[0,600)`、q19 `[200,500)`。求：该 block 的 `(start, end)`、`kv_token_base`、`num_kv_splits`、每个 split 的 `(kv_idx, actual_kv, issued_n)`（`kNumVecElems = 128`）；再对 4 个 query 分别列出两个 split 的存储列区间（`store_begin/store_end/dst_col`，空的写"无"）；最后指出 `is_tile_active` 对 tile 0（q16,q17）在 split 1 上是否放行。

4. **复用/新增标注**（约 15 分钟）：对下列代码对象，各标注 R（复用主线）/ N（新增）/ D（主线机制的变形），并写一句理由。答案附后。

   | 对象 | R/N/D | 理由（先自己写） |
   | :--- | :---: | :--- |
   | `get_swizzled_block`（`scheduler.hpp:202`） | | |
   | `current_k_block_cumsum`（`scheduler.hpp:77`） | | |
   | `copy_gm_to_l1_mx`（`mqa_logits.hpp:296`） | | |
   | `vf_mqa_logits_reduce`（`mqa_logits.hpp:101`） | | |
   | `aiv_zero_empty_groups`（`epilogue.hpp:108`） | | |
   | `aiv_initialize` 里的 atomic add（`epilogue.hpp:92-96`） | | |
   | `MQALogitsScheduler::next_q_block`（`scheduler/mqa_logits.hpp:31`） | | |
   | `get_physical_page_batch`（`scheduler/paged_mqa_logits.hpp:125`） | | |
   | `mqa_logits_store` 的 32B scratch（`mqa_logits.hpp:45-68`） | | |
   | `get_actual_m` 的 M-Grouped 分支（`scheduler.hpp:129-136`） | | |
   | `reversed_pairs` + `vf_mqa_logits_store_bf16`（`mqa_logits.hpp:73-99`） | | |
   | `mad(..., ENABLE_UPDATE, true)`（`mqa_logits.hpp:396`） | | |

5. **术语表闭卷**（约 10 分钟）：合上教程，写出 group、paged、sparse、context、KV cache 五个词条的定义（各 2～3 句，须含一个代码锚点），再从扩展表里任选五个复述。

6. **遗留问题清理**（约 10 分钟）：把本周记录的未解决问题逐条归档——已解决的写一句话答案划掉；未解决的填入下表：

   | 疑问（示例） | 去向 |
   | :--- | :--- |
   | MegaMoE 如何在 M-Grouped 之上组织两次 Grouped GEMM + SwiGLU | 第 44 天 |
   | EP dispatch/combine 的通信对象与计算通信融合 | 第 45 天 |
   | HC Prenorm / TileLang 与手写 kernel 的对比 | 第 47 天 |
   | FIX-pipe bound 如何用 profiling 量化、调什么参数 | 第 50～52 天 |
   | MQA 稀疏寻址的更多变体（如 MFA） | 第 48 天选讲或课后 |

   与计划一致：明天（第 42 天）是机动日，"如果对 MQA 不感兴趣，可用这一天提前开始 MegaMoE，不必逐行读完所有 MQA 变体"。

## 六、当天产出

按计划，今天应整理出三件套（参考版即正文）：

1. **计算模式对比总表**：3.2 节主表 + MQA 细分表（若与自己的版本有出入，把差异行标红）；
2. **复用与新增两张清单**：3.3、3.4 节（建议合并成一页两栏，左"老朋友"右"新问题"，每条带文件:行号）；
3. **术语表**：3.5 节五个核心词条 + 扩展速查表。

建议把它们合并成一页"第 6 周复盘笔记"，结构如下：

```text
# 第 6 周复盘笔记（Grouped GEMM 与 MQA）
1. 一句话主线
   - 本周把 GEMM 主线扩展到 ____（分组）和 ____（注意力打分）两个方向；
     共同点是 ____（复用搬运/MAD/流水主干），新问题是 ____（调度状态/寻址语义）
2. 计算模式对比表（闭卷默写后订正的版本）
3. 复用 / 新增清单（各 ≥8 / ≥6 条，带锚点）
4. 术语表（5 核心 + 5 扩展）
5. 手算记录（练习 2/3 的中间量）
6. 遗留问题与去向（练习 6 的表）
7. 自评结论（通过标准六条：___ 个"能"）
```

## 七、通过标准

学习计划未给当天单独的通过标准，本文按三大任务拟定如下，拆成六个检查点逐条自评"能 / 勉强 / 不能"：

| # | 检查点 | 验证方式 |
| :---: | :--- | :--- |
| 1 | 能闭卷写出 3.2 主表（≥8 行 × 4 列，含数学定义、形状与 major、调度、K 维、Epilogue、边界空组） | 综合练习 1 |
| 2 | 能说出 MQA 三变体在 KV 来源、有效区间、输出形状、调度信息上的差异，各举一个源码锚点 | 3.2 细分表 + 练习 5 |
| 3 | 能列出 ≥8 个复用点与 ≥6 个新增点，每条给出文件与函数级证据 | 综合练习 4 + 3.3/3.4 节 |
| 4 | 术语表 5 个核心词条 + ≥5 个扩展词条闭卷可释（含代码锚点） | 综合练习 5 |
| 5 | MoE 双向对应关系（第 38 天通过标准）仍能完整复述 | 综合练习 2 全对 |
| 6 | MQA 调度手算（窗口聚合、split 参数、存储列区间）正确 | 综合练习 3 全对 |

**评判与补救**：

- **六条全部"能"**：达标，进入第 42 天机动或休息；下周开始 MegaMoE（第 43 天起）。
- **检查点 1 或 2 "不能"**：回到第 37 天 3.8 节（三种 GEMM 对比表）与第 39/40 天的对比小节重读，重做练习 1。
- **检查点 3 "不能"**：重读第 39 天"对比 MQA Logits 与普通 GEMM"一节与 3.3/3.4 节，把 12+11 条逐条在源码里翻出来。
- **检查点 4 "不能"**：只补不会的词条（group/psum → 第 36 天；paged/block_table → 第 40 天 3.3 节；sparse/容量 → 第 40 天 3.1 节；context/窗口 → 第 39 天；KV cache/fused 页 → 第 40 天 3.3 节）。
- **检查点 5 "不能"**：回到第 38 天综合练习重做 MoE 双向题。
- **检查点 6 "不能"**：回到第 39 天的手算配方（`next_q_block` 与存储列区间）重推一遍。

## 八、自测题

1. 写出四类算子的数学定义。哪一类的输出不是"矩阵乘的部分和"而是"逐元素非线性后的加权和"？
2. M-Grouped 与 K-Grouped 各自的"A/B/D 谁平坦、谁逐组"？major 断言分别是什么？为什么恰好相反？
3. `grouped_layout = [3, 259]`（对齐 256）描述了什么？写出两组的有效区间与总 M。
4. K-Grouped 的空组如何得到零输出？为什么开启 accumulate 时不做空组清零？
5. MQA 为什么没有 K 维两层循环？它的 MAD 三个维度（M/N/K）分别对应什么？`kSplitKV` 切的是哪个方向？
6. MQA 输出为什么每行只写前缀？列号怎么算？`mqa_logits_store` 如何满足 MTE3 的 32B 对齐要求？
7. 写出 paged 寻址从"逻辑块"到"物理 token 起点"的换算链。`block_table` 的行数为什么等于 Q？
8. 物理页乱序时，gather 与 store 如何配合恢复逻辑页顺序？
9. FIRST 与 REUSE 各自从哪里读 KV？谁在管理状态？什么操作之后必须重建 metadata？dry-run 为什么不翻转状态？
10. 各列出 5 个"复用主线"与 4 个"新增调度/寻址"的证据（文件:函数）。

<details>
<summary>参考答案（先自己做再展开）</summary>

1. 见 3.2 表第一行；MQA Logits 的输出是 `Σ_h relu(⟨q,kv⟩)·w`——ReLU 发生在逐元素乘积求和之后、head 归约之前，不是线性部分和。
2. M-Grouped：B 逐组 `[G,N,K]`、A/D 平坦，A 必须 K-major（`gemm.hpp:90`）；K-Grouped：D 逐组 `[G,M,N]`、A/B 平坦（tn），A、B 都必须 MN-major（`gemm.hpp:94-95`）。相反的根源：分组维一在输出保留维（行）、一在收缩维（K），tn 的"逻辑转置"使后者的内层连续轴换到了 MN。
3. 行 psum：组 0 有 3 行、组 1 有 3 行；组 0 有效行 `[0,3)`、padding `[3,256)`，组 1 有效行 `[256,259)`、padding `[259,512)`；总 `M = align(259,256) = 512`。
4. 空组领不到任何 block（`current_shape_k = 0` 时游标直接跳过，`scheduler.hpp:167-174`），AIV 侧由 `aiv_zero_empty_groups` 用一块全零 UB 把 `D[g]` 清零（`epilogue.hpp:108-141`）。accumulate 时 AIV drain 是 atomic add，空组保持 `D[g] = C[g]` 不动恰好就是期望，所以清零只在 `!kWithAccumulation` 时执行（`bf16_gemm.hpp:175-176`）。
5. 因为收缩维就是 head_dim（D∈{64,128}），一次 `mad(l0c, l0a, l0b, num_m_rows, issued_n, MAD_K=kHeadDim, ...)` 即可装下（`mqa_logits.hpp:216-217, :396-397`）；M = QH 行、N = KV token、K = head_dim。`kSplitKV=512` 切的是 N（KV token）方向，是装载与 MAD 的分段粒度，不是 K 累加步数。
6. 每行只写 `[ks, ke)` 对应的前缀可避免为无效位置写 `-inf`；`dst_col = 原位置 − ks`（`mqa_logits.hpp:596`）。MTE3 要求 UB 源 32B 对齐，行头未对齐的部分先经 `vf_mqa_logits_stage_head` 搬进对齐 scratch，再分两笔 `asc_copy_ub2gm_align`（`:45-68`）。
7. `logical_page = block_idx / (page_kv/B)`；`physical_page = block_table[query 行, logical_page]`；`token_start = physical_page × page_kv + (block_idx % (page_kv/B)) × B`（`scheduler/paged_mqa_logits.hpp:119-123`）。行数 = Q 是因为契约要求同请求的 query 对引用页达成一致，Host 直接按 query 行展开页表（`attention.hpp:100`、测试 `repeat_interleave`）。
8. AIC 侧 `get_physical_page_batch` 让相邻两页从较小物理页发起成对搬运（`first_page = min`，`:125-134`）；AIV 在 reduce 前比较两个物理页号得到 `reversed_pairs` 位图（`mqa_logits.hpp:548-553`），`vf_mqa_logits_store_bf16` 存储时按位交换半区/相邻向量（`:73-99`），逻辑页序得以恢复。
9. FIRST：AIV 经页表从原始分页 GM gather + NZ 转换，双写进本核 L1 与 metadata 的 compact 区；REUSE：AIC 用 `asc_copy_gm2l1_align` 直接把 compact 页 GM→L1。状态由 Python 侧 `metadata._paged_sparse_mqa_state` 管理（`__init__.py:87-135`）；**修改 KV 内容后必须重建 metadata**，因为 REUSE 只认快照。dry-run 不分配真实输出也不产生有效结果，翻转状态会污染后续真实调用，所以 `if is_first and not dry_run` 才翻转（`__init__.py:133-134`）。
10. 复用示例：`get_swizzled_block`（scheduler.hpp:202）、`gm_ptr::index`（common.hpp:156）、`copy_gm_to_l1` 家族、`mad`（bf16_gemm.hpp:149 / mqa_logits.hpp:396）、多级 stage + `asc_sync_wait/notify`、`copy_l0c_to_ub` 的 FIX 转换、容量 `static_assert`、MX SF `_mx` 装载。新增示例：组游标（scheduler.hpp:144-177）、`aiv_zero_empty_groups`（epilogue.hpp:108）、`next_q_block` 窗口聚合（scheduler/mqa_logits.hpp:31-52）、`get_physical_page_batch`（scheduler/paged_mqa_logits.hpp:125）、FIRST/REUSE 快照（`__init__.py:87-135`）、设备侧排产 kernel（`sparse_mqa_logits_metadata.hpp`）。
</details>

## 附一：综合练习 2 参考答案（MoE 双向手算）

**(1) 前向（M-Grouped）**：

- psum：组 0 有 3 行 → `psum[0] = 3`；组 1 从 `align(3,256) = 256` 起、有 3 行 → `psum[1] = 259`；总 `M = align(259,256) = 512`；
- 组 0 有效行 `[0,3)`、padding `[3,256)`；组 1 有效行 `[256,259)`、padding `[259,512)`；
- `BLOCK_M=256`：组 0 的 `num_m_blocks = ceil(3/256) = 1`，占全局 m-block `{0}`；组 1 的 `num_m_blocks = ceil((259−256)/256) = 1`，组内 m-block 0 平移 `last_psum_m / BLOCK_M = 256/256 = 1`，占全局 m-block `{1}`；
- dispatch 排序后组内行序为 t0, t3, t5，故 **t5 的输出落在 D[2]**（组 0 的第 3 行），它乘的是 **B[0]**（expert 0 的权重）。

**(2) 反向 wgrad（K-Grouped）**：沿用同一 psum 描述 K 维：

- 组 0 有效 K `[0,3)`、组 1 有效 K `[256,259)`，`total_k = 512`；
- A `[512, M]` / B `[512, N]`：第 k 行是按 expert 分段拼接的 token 流（本例 `[t0,t3,t5 | pad | t1,t2,t4 | pad]` 的激活与上游梯度）；D `[2, M, N]`：`D[0] = Σ_{k∈[0,3)} A[k,m]·B[k,n]`（expert 0 的权重梯度），`D[1] = Σ_{k∈[256,259)}`（expert 1）；
- 组 1 的 `k_idx_base = last_psum_k = align(3,256) = 256`；
- **t5 的梯度贡献累加进 D[0]**（expert 0 的 slab）。

**(3)** 前向按"token 选了哪个 expert"分组，输出行保持 dispatch 后的顺序（D 平坦）、权重逐组（B 逐组）；反向要对每个 expert 的 token 求和得到该 expert 的权重梯度，求和维就是 token 轴（K），所以 D 逐组、A/B 平坦——分组维从输出保留侧移到了收缩侧。

## 附二：综合练习 3 参考答案（MQA 调度手算）

- `(start, end)`：start 取块内各行 ks 的最小值 = 0，end 取各行 ke 的最大值 = 600；
- `kv_token_base = 0/8*8 = 0`；`num_kv_splits = ceil((600−0)/512) = 2`；
- split 0：`kv_idx = 0`，`actual_kv = min(512, 600−0) = 512`，`issued_n = aligned(512,128) = 512`；
- split 1：`kv_idx = 512`，`actual_kv = min(512, 600−512) = 88`，`issued_n = aligned(88,128) = 128`；
- 存储列区间（`store_begin = max(kv_idx, ks)`、`store_end = min(kv_idx+actual_kv, ke)`、`dst_col = store_begin − ks`）：

| query | split 0 | split 1 |
| :---: | :--- | :--- |
| q16 `[100,300)` | `[100,300)` → 200 列、dst 0 | `[512,300)` → 无 |
| q17 `[100,600)` | `[100,512)` → 412 列、dst 0 | `[512,600)` → 88 列、dst 412 |
| q18 `[0,600)` | `[0,512)` → 512 列、dst 0 | `[512,600)` → 88 列、dst 512 |
| q19 `[200,500)` | `[200,500)` → 300 列、dst 0 | `[512,500)` → 无 |

- `is_tile_active`（tile 0 = q16,q17，split 1 区间 `[512,600)`）：q16 的 `ke=300 ≤ 512` 不交叠，但 q17 的 `ke=600 > 512` → **放行**；
- 输出行宽 `max_seqlen_k = max(ke−ks) = 600`（q18）；输出张量 `[20, 600]`，行 q16 只写前 200 列。

## 附三：综合练习 4 参考答案

| 对象 | R/N/D | 理由 |
| :--- | :---: | :--- |
| `get_swizzled_block` | R | Normal 调度的原件，M/K-Grouped 组内直接调用 |
| `current_k_block_cumsum` | N | 主线没有"组"概念，K-Grouped 新增的游标状态 |
| `copy_gm_to_l1_mx` | R | `copy.hpp` 搬运家族的 MX scale 变体，fp8 GEMM 主线同样使用 |
| `vf_mqa_logits_reduce` | N | head 维加权归约是 MQA 特有的输出侧缩减 |
| `aiv_zero_empty_groups` | N | 空组输出清零，主线无对应机制 |
| `aiv_initialize` 里的 atomic add | D | epilogue 框架是主线的，atomic add 语义为 K-Grouped accumulate 新开 |
| `MQALogitsScheduler::next_q_block` | D | "每核跨步领任务"思想同源，但条带结构、窗口聚合是新的 |
| `get_physical_page_batch` | N | 页表间接寻址 + 成对打包，主线无间接寻址 |
| `mqa_logits_store` 的 32B scratch | N | MTE3 对齐限制下的行头中转，GEMM 写回不需要 |
| `get_actual_m` 的 M-Grouped 分支 | D | 接口沿用，行为改为恒返回 BLOCK_M（TODO 注释） |
| `reversed_pairs` + `vf_mqa_logits_store_bf16` | N | 物理页乱序的顺序恢复，分页特有 |
| `mad(..., ENABLE_UPDATE, true)` | R | 与 bf16_gemm 的 mad 调用同款，只是无 K 循环、单次调用 |
