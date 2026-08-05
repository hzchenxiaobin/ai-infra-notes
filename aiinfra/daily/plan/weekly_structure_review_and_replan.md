# AI Infra 10 周教程 —— 周级结构合理性与教学节奏评审 + 重组方案

> 评审日期：2026-08-04
> 评审范围：`aiinfra/daily/` 全部 week1–week10 README + `plan/learning_plan_10week.md` + 顶层 `README.md` + `SKILL.md`
> 评审角度：**周级教学结构**（循序渐进 / 有节奏 / 系统性 / 是否散乱拼盘）
> 与既有文档的分工：
> - `tutorial_quality_evaluation.md`、`market_benchmark_quality_review.md`：聚焦事实准确性、代码落盘、市场缺口（已完成）
> - `remediation_plan.md`、`market_driven_remediation_plan.md`：聚焦 P0–P3 整改执行（已执行）
> - **本文：聚焦"周级结构 + 教学节奏"，是上述文档未覆盖的维度**，输出一份**重组后的 10 周路线**

---

## 一、评估标准

用户要求每周内容"循序渐进、有节奏、系统性、非散乱拼盘"。拆解为三个可判定维度：

| 维度 | 含义 | 判定信号 |
|------|------|---------|
| **循序渐进** | 周与周之间有清晰能力台阶，前置依赖单向不倒置 | 出现"下周内容提前在本周出现"或"本周需要的知识下周才讲"即违规 |
| **有节奏** | 每周 day1→day7 遵循一致模板 | README 声称的"Day1-2 核心/Day3-4 源码/Day5 项目/Day6 Profiler/Day7 缓冲"是否被各周实际遵循 |
| **系统性** | 每周是单一主题聚焦深入 | 周标题出现"X + Y"、单周塞 ≥3 个无内在递进的子主题即违规 |

---

## 二、总体结论

10 周主线**方向正确**（GPU 基础 → kernel 优化 → FA → 推理系统 → 调度 → 分布式 → 面试），与 2025–2026 JD 技能树吻合。但 `learning_plan_10week.md` 在把原 8 周重排为 10 周时，为"凑满 7 天/周"把若干大专题**切碎塞进过渡周**，产生 3 类结构性病理：

1. **多主题拼盘周**（系统性破坏）：W3 / W4 / W5 / W8 / W9 标题即"X + Y"双主题，单周容纳 2–3 个无递进关系的子主题
2. **核心专题碎片化**（循序渐进破坏）：FlashAttention 被切散到 W3 / W4 / W5 三周共 4 段，中间被 6 个无关 day 打断；学习者无法在一周内形成连贯 FA 心智模型
3. **节奏模板名存实亡**（有节奏破坏）：顶层 README 声明"Day7 = 弹性/缓冲/复盘"，实际 5/10 周的 Day7 是新内容或活动；"Day3-4 = 源码分析、Day5 = 项目推进"几乎无周遵循

**结构性病理 ≠ 事实错误**：事实硬伤已由 remediation_plan 修复，但即使所有数字都正确，当前周级结构仍然散乱。本文不重复事实层评审，只处理结构层。

---

## 三、逐周评审

### Week 1：GPU 执行模型与内存基础 — ✅ 合理（保留）

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★★★ | 单主题（GPU 执行模型 + 内存层级），7 天全部围绕"建立 GPU 性能直觉" |
| 前置依赖 | ★★★★★ | 无外部依赖，d1 SM/Warp → d2 Occupancy → d3 deviceQuery 实测 → d4 Memory → d5 Bank Conflict → d6 Profiling → d7 复盘，链条干净 |
| 周内节奏 | ★★★★☆ | 唯一遵循声明模板的周；d7 真正做复盘 |

**结论**：本周是全课程**节奏最好的样板**，无需调整。

---

### Week 2：CUDA Kernel 优化 + Tensor Core — ⚠️ 半合理

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★☆☆ | "FMA GEMM 优化"与"Tensor Core/WMMA"是两种计算范式，混在一周；Day3 Streams、Day4 Nsight 打断 GEMM 优化主线 |
| 前置依赖 | ★★★★☆ | 依赖 W1，干净 |
| 周内节奏 | ★★★☆☆ | Day7 既是"限时手撕"（README）又是"CUTLASS"（10 周计划），口径冲突；手撕/源码/新内容三件事挤在 d7 |

**结构问题**：
1. GEMM 七层优化（d5，FMA 路线）与 WMMA（d6，Tensor Core 路线）是**两种范式**，前者是"如何榨 FMA 上限"，后者是"换指令集突破上限"。把 WMMA 接在七层路径末尾，让人误以为 WMMA 只是"第八层"，混淆了"同范式优化"与"换范式"。
2. Day3 CUDA Streams 与 GEMM 优化无内在递进，是为凑满 7 天塞入的独立话题。
3. Day7 口径冲突（README 写手撕，10 周计划写 CUTLASS），学习者不知该做什么。

**建议**：W2 收敛为"CUDA Kernel 优化方法论（FMA 路线）"，把 WMMA/Tensor Core 移出与 CUTLASS 合并成独立的 W3。

---

### Week 3：手撕复盘 + Transformer 算子手写 — ❌ 最严重的拼盘周

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★☆☆☆☆ | 单周塞 5 个无递进子主题：FA 简化 / CUTLASS+CuTe / Transformer 推理流程 / Softmax+LayerNorm kernel / Triton / Attention IO 分析 |
| 前置依赖 | ★★☆☆☆ | **倒置**：d1 FA 简化需要 online softmax，但 online softmax 在 W4d4 才系统讲；d2 CUTLASS 依赖 W2d6 WMMA，逻辑上属于 W2 的延续却跳到这里 |
| 周内节奏 | ★★☆☆☆ | d1 FA → d2 CUTLASS → d3 推理流程，三天三个不相关主题，无任何"逐层深入" |

**结构问题（重点）**：
1. **标题即承认拼盘**："手撕复盘 + Transformer 算子手写"是两件事；实际内容还混入 FA、CUTLASS、Triton、IO 分析，共 5 件事。
2. **FA 提前暴露**：d1 FA 简化版抢在 W4 FA 专题之前，学习者还没学 online softmax 就先写 FA kernel，是"先做后懂"；而 d6 又回到 Attention IO 分析，FA 知识被切成 d1 / d6 两段夹在无关内容中间。
3. **CUTLASS 错位**：CUTLASS 是 GEMM 工具库，自然归属 W2 的 WMMA 之后，却被挪到"Transformer 算子"周，破坏 GEMM 工具链连贯性。
4. **Triton 一天过载**：Triton 是 2025 算子岗 70%+ JD 必考、近乎必备技能，一天内从 program 模型塞到 FA，无性能实测空间（`market_benchmark_quality_review.md` 已指出）。
5. **Transformer 推理流程（Prefill/Decode）属系统视角**，放在算子手写周里，与 Softmax kernel 手写不在同一抽象层。

**建议**：W3 拆解重组——CUTLASS 归入新 W3（Tensor Core 专题），Softmax/LayerNorm/Triton/手撕归入新 W4（Transformer 算子），FA 简化与 IO 分析归入新 W5（FA 专题），Transformer 推理流程归入新 W6（推理系统基础）。

---

### Week 4：算子集成 + FlashAttention（前半）— ⚠️ 双主题缝合

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★☆☆ | 前半（d1-d3）算子集成/源码分析，后半（d4-d7）FA 深挖，两个主题 |
| 前置依赖 | ★★★☆☆ | d4 FA 论文需要 W3d1 FA 简化作铺垫，但中间隔了 d1-d3 三个集成日，铺垫已冷 |
| 周内节奏 | ★★☆☆☆ | d3 Profiling/Fusion → d4 FA 论文是硬切换；d7 官方源码是新内容而非复盘 |

**结构问题**：
1. "算子集成"（C++ Extension）是 W3 收尾内容的延续，被挪到 W4 前半当"过渡"，导致 W4 双主题。
2. FA 前半（d4-d7）本身连贯，但被前三天的集成内容隔开了与 W3 FA 简化的联系。
3. Day7 = FA 官方源码（新内容），违反"Day7 = 复盘"。

**建议**：算子集成融入各项目推进日（d5），不单列；FA 内容整体移入新 W5 连续讲。

---

### Week 5：FlashAttention（后半）+ 推理系统基础 — ⚠️ 双主题缝合

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★☆☆ | 前半（d1-d4）FA-2/集成/对比/IO 方法论，后半（d5-d7）推理系统入门 |
| 前置依赖 | ★★★★☆ | FA→推理的桥（FA 是推理核心算子）合理，但跨度大 |
| 周内节奏 | ★★★☆☆ | d4 IO 方法论总结是 FA 收官，d5 立刻跳 Prefill/Decode，pivot 较硬；d7 vLLM 架构是新内容 |

**结构问题**：
1. FA 被切在 W4/W5 两周，前半在 W4d4-d7、后半在 W5d1-d4，中间隔着 W4d1-d3 的集成内容与 W5d5-d7 的推理内容——**FA 实际跨越 2 周共 8 天但被切成 2 段**。
2. 推理系统基础（Prefill/Decode/KV Cache/vLLM）是 W6 推理系统核心的前置，本应与之连续，却被 FA 后半隔开。

**建议**：FA 全部内容合并到新 W5 单周贯通；推理系统基础与 W6 核心合并为新 W6。

---

### Week 6：推理系统核心 — ✅ 基本合理（量化插入略突兀）

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★★☆ | 单主题（推理系统核心机制），但 d5 量化是独立技术插入 |
| 前置依赖 | ★★★★☆ | 依赖 W5 推理基础，干净；d6 Dynamic Batching 是 W7 Batching 的前置，跨周衔接合理 |
| 周内节奏 | ★★★★☆ | d3 Mini 引擎 v0（项目推进）、d4 Profiling，符合模板；d7 真正复盘 |

**结构问题**：
1. 量化（d5）与 PagedAttention/FlashDecoding/Mini 引擎无内在递进，是为补市场缺口塞入（`market_driven_remediation_plan` B4 已落地），显得突兀。
2. Dynamic Batching（d6）与 W7 Continuous Batching 是兄弟概念，分在两周边界，可接受但不最优。

**建议**：量化移到新 W8（推理加速技术），与 FP8/投机解码/CUDA Graph 同聚；Dynamic Batching 与 Continuous Batching 合并到新 W7。

---

### Week 7：Batching 与调度 — ✅ 合理（保留）

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★★★ | 单主题，7 天全部围绕 batching/调度 |
| 前置依赖 | ★★★★★ | 依赖 W6，d1 Continuous Batching 接 W6d6 Dynamic Batching |
| 周内节奏 | ★★★★☆ | d5 Mini 引擎 v1（项目推进）、d7 调度总结，符合模板 |

**结论**：本周是**第二好的样板**，保留。建议把 W6d6 Dynamic Batching 并入本周 d1 让 batching 主题完整。

---

### Week 8：系统整合与分布式并行 — ❌ 三主题拼盘

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★☆☆☆ | 三主题：系统整合（d1-d3）、投机解码（d4）、分布式并行（d5-d6）、代码重构（d7） |
| 前置依赖 | ★★★☆☆ | d5 TP/PP/DP 与 d1-d3 调度器无递进；d6 Ring Attention 需要 FA 知识（W5）但隔了 3 周 |
| 周内节奏 | ★★☆☆☆ | d7 代码重构是清理活动，非学习主题也非复盘 |

**结构问题**：
1. **分布式并行（TP/PP/DP/Ring Attention）是 75%+ JD 要求的大主题**，被压成 d5-d6 两天塞在"系统整合"周末，远不够深入。
2. 投机解码（d4）与分布式无关，是推理加速技术，应与量化/CUDA Graph 同聚。
3. 系统整合（完整调度器、多请求并发）本属 W7 调度收尾或 W9 项目整合，夹在中间当过渡。
4. d7 代码重构既不是新主题也不是复盘，是工程杂务，占用宝贵的 day7 复盘位。

**建议**：分布式单独成新 W9；投机解码移到新 W8（推理加速技术）；系统整合内容并入新 W10 项目整合周。

---

### Week 9：系统联调与项目打磨 — ⚠️ 新旧混杂

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★☆☆ | "项目打磨"周却塞入两个新主题：MoE+EP（d5）、CUDA Graph（d4） |
| 前置依赖 | ★★★☆☆ | d5 MoE+EP 需要分布式 EP 知识（W8d5），但 W8 分布式只讲了 2 天，铺垫不足 |
| 周内节奏 | ★★★☆☆ | d6 README、d7 架构图是文档活动，非学习/复盘 |

**结构问题**：
1. **MoE+EP 是 2026 最热主题**（DeepSeek 带动），压成单天塞在"项目打磨"周，与联调/文档并列，主题严重不匹配。
2. CUDA Graph 是独立的 launch overhead 优化技术，与"系统联调"无递进。
3. d6/d7 是文档活动，应并入 W10 项目整合周。

**建议**：MoE+EP 移到新 W9（分布式并行）与 EP 并行同聚；CUDA Graph 移到新 W8（推理加速技术）；联调/文档移到新 W10。

---

### Week 10：面试冲刺 — ✅ 合理（Ascend 插入略偏）

| 维度 | 评分 | 说明 |
|------|------|------|
| 主题聚焦 | ★★★★☆ | 单主题（面试冲刺），d2 Ascend 对比是知识点而非面试训练，略偏 |
| 前置依赖 | ★★★★★ | 全课程收官 |
| 周内节奏 | ★★★★☆ | d4 Mock、d6 诊断剧本+手撕、d7 最终复盘，节奏好 |

**结论**：保留。建议 Ascend 对比移到新 W9（分布式/多硬件）让多硬件对比与分布式同聚，W10 腾出空间给项目整合与 mock。

---

## 四、结构性问题汇总（跨周）

### 问题 1：FlashAttention 碎片化 —— 最伤"系统性"

FA 内容实际分布：

```
W3d1 FA简化 ──(CUTLASS/推理流程/Softmax/Triton 4天打断)── W3d6 IO分析
   ──(W4d1-d3 集成/Profiling 3天打断)── W4d4 论文 → W4d5 Forward → W4d6 Backward → W4d7 官方源码
   ──(周末边界)── W5d1 FA-2 → W5d2 集成 → W5d3 性能对比 → W5d4 IO方法论
```

- 跨 **3 周、10 天**，被 **7 个无关 day** 打断
- 学习者无法在一周内形成连贯 FA 心智模型，每次捡起都要回顾
- FA 是算子岗/推理岗**面试核心追问点**，碎片化直接削弱面试表现

### 问题 2：Week 3 是知识拼盘（5 子主题无递进）

| Day | 子主题 | 真正归属 |
|-----|--------|---------|
| d1 | FA 简化 | FA 专题 |
| d2 | CUTLASS + CuTe | Tensor Core/GEMM 工具 |
| d3 | Transformer 推理流程 | 推理系统 |
| d4 | Softmax/LayerNorm kernel | Transformer 算子 |
| d5 | Triton | 独立语言主题 |
| d6 | Attention IO 分析 | FA 理论 |
| d7 | 算子分类总结 | 收尾 |

5 个子主题分属 4 个不同抽象层（硬件指令/算子/系统/语言），硬塞一周是典型"散乱知识拼盘"。

### 问题 3：前置依赖倒置

| 倒置 | 位置 | 正确顺序 |
|------|------|---------|
| FA 简化先于 online softmax | W3d1 用 FA，W4d4 才讲 online softmax | online softmax → FA |
| CUTLASS 远离 WMMA | W2d6 WMMA → W3d2 CUTLASS（隔 3 天无关内容） | WMMA → CUTLASS 连续 |
| Ring Attention 远离 FA | W5 FA → W8d6 Ring Attention（隔 3 周） | FA → Ring Attention 紧邻 |
| MoE+EP 远离分布式 EP | W8d5 分布式 → W9d5 MoE+EP（隔一周） | 分布式 → MoE+EP 连续 |

### 问题 4：节奏模板名存实亡

顶层 `README.md` 声明的周节奏模板 vs 实际：

| 声明 | 实际遵循的周 | 违规的周 |
|------|------------|---------|
| Day 1-2 核心学习+Coding | 大部分 | — |
| Day 3-4 源码分析 | W4（FA 源码）、W7（Scheduler 源码） | W2（d3 Streams/d4 Nsight 非源码）、W3（d3 推理流程/d4 算子非源码）、W8（d3 调度器/d4 投机解码非源码） |
| Day 5 项目推进 | W6（Mini 引擎 v0）、W7（Mini 引擎 v1） | W2（d5 GEMM）、W3（d5 Triton）、W4（d5 FA Forward）、W5（d5 Prefill/Decode）、W8（d5 分布式）、W9（d5 MoE） |
| Day 6 Profiler+Debug | W1、W6 | 其余各周 d6 多为新内容 |
| Day 7 弹性/缓冲/复盘 | W1、W3、W6、W7、W10 | W2（手撕）、W4（官方源码）、W5（vLLM 架构）、W8（代码重构）、W9（架构图）—— 5/10 周违规 |

模板本身也欠合理：Day3-4 强制"源码分析"对很多周不适用（如 W1 内存基础周无源码可读）。应改为更通用的"Day3-4 进阶实现/源码"。

### 问题 5：大专题天数不足 vs 小专题占用整周

| 专题 | 市场权重 | 当前天数 | 合理性 |
|------|---------|---------|--------|
| FlashAttention | 算子/推理岗必考 | 10 天（碎片化） | 天数够但碎片化，应集中 |
| Tensor Core/WMMA/CUTLASS | 算子岗分层点 | 2 天（W2d6-d7） | 不足 |
| Triton | 70%+ JD 必考 | 1 天（W3d5） | 严重不足 |
| 分布式并行（TP/PP/DP/NCCL/Ring） | 75%+ JD 必考 | 2 天（W8d5-d6） | 严重不足 |
| MoE + EP | 2026 最热 | 1 天（W9d5） | 严重不足 |
| 量化（FP8/INT8/W8A16） | 70%+ JD | 1 天（W6d5） | 不足 |
| CUDA Streams | 中 | 1 天（W2d3） | 偏多（可并入 W2 优化日） |
| 算子集成 C++ Extension | 中 | 2 天（W4d2/W5d2） | 重复，可融为 1 天 |

---

## 五、重组方案：单一主题周 + 连续专题块

### 重组原则

1. **每周一个主题**，标题不含"+"
2. **大专题给连续多天**（FA/Tensor Core/分布式/MoE），不跨周切散
3. **前置依赖严格单向**，消除倒置
4. **恢复统一节奏模板**（见下），Day7 真正做复盘+手撕
5. **保留所有已落地的市场整改内容**（GQA/MLA、PD 分离、MoE、FP8、诊断剧本、Triton benchmark、SGLang 对比等），只重排位置

### 重组后的 10 周路线

| 周 | 主题（单一） | 核心聚焦 | 主要内容来源 |
|----|------------|---------|-------------|
| **W1** | GPU 执行模型与内存基础 | SM/Warp/Occupancy/Memory/Bank Conflict/Nsight | 原 W1（不变） |
| **W2** | CUDA Kernel 优化方法论 | Warp Shuffle / Register Blocking / float4 / GEMM 七层 / Streams / Nsight | 原 W2 d1-d5 |
| **W3** | Tensor Core 与 CUTLASS | WMMA / mma.sync / CUTLASS / CuTe / 混合精度 / FP8 入门 | 原 W2 d6-d7 + 原 W3 d2 + 新 FP8 |
| **W4** | Transformer 算子手写 + Triton | Softmax / LayerNorm / GEMM backward / 手撕限时 / Triton 三方 benchmark | 原 W3 d3-d5,d7 + 新 Triton 扩展 |
| **W5** | FlashAttention 全专题 | FA 简化 → 论文/Online Softmax → Forward → Backward → FA-2 → FA-3 → 官方源码 → 性能对比 → IO 方法论 | 原 W3 d1,d6 + 原 W4 d4-d7 + 原 W5 d1-d4 |
| **W6** | 推理系统基础与 KV Cache | Prefill/Decode / KV Cache+GQA/MQA/MLA / vLLM 架构 / PagedAttention / FlashDecoding | 原 W5 d5-d7 + 原 W6 d1-d2 |
| **W7** | Batching 与调度 | Dynamic/Continuous Batching / vLLM Scheduler / Chunked Prefill / Prefix Caching+RadixAttention / PD 分离 / SGLang 对比 | 原 W6 d6 + 原 W7 全 + 原 W6 d4b |
| **W8** | 推理加速技术 | 量化(W8A16/INT8 KV/FP8/FP4) / 投机解码(Medusa/EAGLE/MTP) / CUDA Graph / 采样 kernel(top-p/top-k) | 原 W6 d5 + 原 W7 d3 + 原 W8 d4 + 原 W9 d4 + 新 |
| **W9** | 分布式并行与多硬件 | TP/PP/DP / NCCL / 通信计算重叠 / Ring Attention / MoE+EP / Ascend 对比 | 原 W8 d5-d6 + 原 W9 d5 + 原 W10 d2 |
| **W10** | 项目整合与面试冲刺 | Mini 引擎真整合 / 全链路 Profiling / 项目文档+架构图 / 面试题库 / Mock / 诊断剧本 / 手撕清单 / 最终复盘 | 原 W8 d1-d3,d7 + 原 W9 d1-d3,d6-d7 + 原 W10 d1,d3-d7 |

### 统一的 day1→day7 节奏模板（修正版）

> 替代顶层 README 中不切实际的"Day3-4 源码分析/Day5 项目推进"模板。

| Day | 类型 | 核心动作 | 说明 |
|-----|------|---------|------|
| Day 1 | 🔬 理论 + 基础 kernel | 概念建模 + 最简实现 | 每周的"地基" |
| Day 2 | 🔬 理论 + 基础 kernel | 概念深化 + 基础优化 | 在 d1 之上加一层 |
| Day 3 | 📖 进阶实现 / 源码 | 进阶优化或开源源码导读 | 二选一，按主题适用性 |
| Day 4 | 📖 进阶实现 / 源码 | 同上，或变体/扩展 | |
| Day 5 | 🛠 项目推进 | 接入 Mini 引擎或 benchmark | 把本周所学整合进项目 |
| Day 6 | 📊 Profiling + 性能分析 | ncu/nsys 实测 + Roofline | 真实数据留档 |
| Day 7 | 🧘 复盘 + 限时手撕 + 面试要点 | 知识地图 / 手撕清单 / 面试 Q&A 收敛 | 不引入新内容 |

### 各周 day 详细安排（重组后）

#### W1：GPU 执行模型与内存基础（不变）
d1 GPU 执行模型 → d2 Occupancy → d3 deviceQuery → d4 Memory Hierarchy → d5 Bank Conflict → d6 Nsight → d7 复盘

#### W2：CUDA Kernel 优化方法论
d1 Warp Shuffle + Warp/Block Reduce → d2 Register Blocking + 2D Tiling → d3 float4 向量化 + GEMM 七层路径（前四层） → d4 GEMM 七层路径（后三层）+ cuBLAS 对比 → d5 CUDA Streams + 异步执行（接入 mini benchmark） → d6 Nsight Compute 性能分析（GEMM v4/v6 实测） → d7 复盘 + 限时手撕（Reduce 30min / GEMM tiling 60min）

> 变化：原 W2d6 WMMA、d7 CUTLASS 移到新 W3；GEMM 七层从 1 天扩到 2 天（d3-d4），解决"七层塞一天"的过载。

#### W3：Tensor Core 与 CUTLASS
d1 Tensor Core 架构 + WMMA fragment 基础 → d2 手写 WMMA GEMM（m16n16k16）+ cuBLAS 对比 → d3 `mma.sync` + ldmatrix + 对齐约束 → d4 CUTLASS 源码 + CuTe（Layout/Tensor/local_tile）+ 实例化调用 → d5 项目推进：WMMA GEMM 接入 mini benchmark + double buffering → d6 Profiling：Tensor Core 利用率（`sm__pipe_tensor_op_hmma_cycles_active`）+ WMMA vs FMA vs cuBLAS 三方对比 → d7 复盘 + 手撕（WMMA fragment 生命周期）+ 混合精度/FP8 入门面试题

> 变化：把原 W2d6-d7 + W3d2 合并为单周；新增 mma.sync/ldmatrix 深度（原缺失）；FP8 入门放在这里（为 W8 量化铺路）。

#### W4：Transformer 算子手写 + Triton
d1 Transformer 推理流程（Prefill/Decode 视角）+ 算子分类 → d2 手写 Softmax kernel（naive → online → Welford） → d3 手写 LayerNorm kernel + GEMM backward 数据流 → d4 Triton 语言（program 模型 + tl.load/store/reduce） → d5 项目推进：Triton 重写 Softmax/GEMM/FA 三方 benchmark（与 CUDA 版对比留档） → d6 Profiling：Triton vs CUDA vs PyTorch 性能对比 + autotune 配置搜索 → d7 复盘 + 限时手撕（Softmax 20min / LayerNorm 30min）+ "何时用 Triton 何时必须 CUDA"决策表

> 变化：把原 W3 的 Softmax/LayerNorm/Triton/手撕/推理流程合并成单周；Triton 从 1 天扩到 2 天（d4-d5），解决市场缺口；Transformer 推理流程作为 d1 引入视角（后面 W6 深入）。

#### W5：FlashAttention 全专题（核心周）
d1 FA 简化版 CUDA 实现 + Attention IO 分析（4N²+4Nd 口径） → d2 FA 论文精读 + Online Softmax 三公式推导 → d3 手写 FA Forward kernel（完整版）+ causal 变体 → d4 FA Backward + GEMM Backward（`L_i = m_i + log ℓ_i` 重计算） → d5 项目推进：FA 接入 Mini 引擎（C++ Extension）+ 标准 vs 手写 vs 官方性能对比 → d6 FA-2 / FA-3 演进 + 官方 CUDA 源码导读（带行号）+ IO 方法论总结 → d7 复盘 + 限时手撕（FA Forward 简化版 60min）+ FA 面试 Q&A 收敛

> 变化：**核心重组**——把原散在 W3d1/d6 + W4d4-d7 + W5d1-d4 的 FA 内容全部合并到单周 7 天，连续不打断；学习者一周内从简化版到 FA-3 形成完整心智模型。

#### W6：推理系统基础与 KV Cache
d1 推理流程深入（Prefill compute-bound vs Decode memory-bound + Roofline） → d2 KV Cache 实现（含 GQA/MQA/MLA 变体 + 显存口算） → d3 vLLM 整体架构 + V1 演进 → d4 vLLM Worker + PagedAttention kernel → d5 项目推进：Mini 引擎 v0（接入 FA + KV Cache） → d6 FlashDecoding（Decode 并行度突破）+ Profiling → d7 复盘 + KV Cache/PagedAttention 面试 Q&A

> 变化：把原 W5d5-d7 + W6d1-d4 合并；量化移出（去 W8）；Mini 引擎 v0 留在本周 d5。

#### W7：Batching 与调度（保留+微调）
d1 Dynamic Batching + Continuous Batching → d2 vLLM Scheduler 源码分析 → d3 框架对比（vLLM/TRT-LLM/SGLang/LightLLM） → d4 Chunked Prefill + Prefix Caching + RadixAttention → d5 项目推进：Mini 引擎 v1（多请求并发） → d6 PD 分离推理（Disaggregated）+ TTFT/TPOT 改善实测 → d7 复盘 + 调度策略总结 + 面试 Q&A

> 变化：把原 W6d6 Dynamic Batching 并入本周 d1；其余基本不变（原 W7 是好样板）。

#### W8：推理加速技术（新周）
d1 量化基础（对称/非对称、per-channel/per-token、weight-only vs weight+activation）+ W8A16 dequant kernel → d2 INT8 KV Cache 量化 + FP8（E4M3/E5M2）kernel + GPTQ vs AWQ vs SmoothQuant 对比 → d3 投机解码（接受率精确期望 + Medusa/EAGLE/MTP 三路线） → d4 CUDA Graph（静态捕获 + 动态 shape bucketing）+ 采样 kernel（top-p/top-k） → d5 项目推进：量化/投机解码/CUDA Graph 选一接入 Mini 引擎 → d6 Profiling：量化前后精度+性能对比、CUDA Graph launch gap 实测 → d7 复盘 + 面试 Q&A（量化/投机解码/CUDA Graph/采样）

> 变化：把原散在 W6d5/W7d3/W8d4/W9d4 的加速技术合并成单周；解决量化/投机解码/CUDA Graph 各 1 天的碎片化。

#### W9：分布式并行与多硬件
d1 Tensor Parallelism（column/row-parallel QKV + all-reduce） → d2 Pipeline Parallelism（1F1B + bubble ratio）+ DP → d3 NCCL collectives（all-reduce/all-gather/reduce-scatter 通信量 + ring/tree 拓扑） → d4 通信计算重叠（双 stream + CUDA Graph overlap）+ nsys 时间线 → d5 Ring Attention（长上下文分布式注意力）+ MoE + EP（Top-K 路由 + all-to-all + DeepEP/EPLB） → d6 项目推进：TP 推理 demo + MoE 路由模拟器 + Ascend CANN 对比（编程模型/工具链） → d7 复盘 + 分布式/MoE/多硬件面试 Q&A

> 变化：分布式从 2 天扩到 5 天（d1-d5），解决 75%+ JD 主题被压成 2 天的问题；MoE+EP 与 EP 并行同聚（前置依赖打通）；Ascend 对比从 W10d2 移来（多硬件与分布式同属"跨设备"主题）。

#### W10：项目整合与面试冲刺
d1 整合全部自定义 Kernel 到 Mini 引擎（真整合，替换 sleep 模拟） → d2 系统联调（六步分层验证）+ 全链路 Profiling → d3 项目文档（README）+ 架构图 + 数据流图 → d4 高频面试题基础篇 + 进阶篇（含系统设计题） → d5 Mock 面试 + STAR 项目话术 → d6 诊断流程实战剧本（低 MFU/OOM/hang 三案例）+ 限时手撕清单（10 项） → d7 最终复盘：10 周能力地图 + 查漏补缺

> 变化：把原 W8d1-d3,d7/W9d1-d3,d6-d7 的项目整合内容合并到本周前半；面试冲刺集中后半；Ascend 已移走腾出空间。

---

## 六、与现状的迁移对照（原 → 新）

| 原 W?d? | 原主题 | 新归属 | 理由 |
|---------|--------|--------|------|
| W2d3 CUDA Streams | 优化方法论 | 新 W2d5 | 与 GEMM 优化主线分离，作为独立优化技术 |
| W2d6 WMMA | 优化周 | 新 W3d1-d2 | 与 CUTLASS 合并成 Tensor Core 专题周 |
| W2d7 CUTLASS / 手撕 | 冲突 | 新 W3d4 / 新 W2d7 | 消除口径冲突 |
| W3d1 FA 简化 | 拼盘周 | 新 W5d1 | 归入 FA 专题周 |
| W3d2 CUTLASS+CuTe | 拼盘周 | 新 W3d4 | 归入 Tensor Core 专题周 |
| W3d3 Transformer 推理流程 | 拼盘周 | 新 W4d1 | 归入 Transformer 算子周（作引入视角） |
| W3d4 Softmax/LayerNorm | 拼盘周 | 新 W4d2-d3 | 归入 Transformer 算子周 |
| W3d5 Triton | 拼盘周 | 新 W4d4-d5 | 扩为 2 天 |
| W3d6 Attention IO | 拼盘周 | 新 W5d1 | 归入 FA 专题周 |
| W3d7 算子分类总结 | 拼盘周 | 新 W4d7 | 归入 Transformer 算子周复盘 |
| W4d1-d3 算子集成/Profiling | 双主题 | 新 W5d5 / 新 W10d2 | 集成融入 FA 项目日；Profiling 融入联调 |
| W4d4-d7 FA 前半 | 双主题 | 新 W5d2-d6 | 归入 FA 专题周 |
| W5d1-d4 FA 后半 | 双主题 | 新 W5d1-d6 | 与 FA 前半合并 |
| W5d5-d7 推理基础 | 双主题 | 新 W6d1-d3 | 归入推理系统基础周 |
| W6d1-d2 PagedAttention/FlashDecoding | 核心周 | 新 W6d4-d6 | 归入推理系统基础周 |
| W6d5 量化 | 核心周 | 新 W8d1-d2 | 归入推理加速技术周 |
| W6d6 Dynamic Batching | 核心周 | 新 W7d1 | 与 Continuous Batching 合并 |
| W7d3 投机解码 | 调度周 | 新 W8d3 | 归入推理加速技术周 |
| W8d1-d3 系统整合 | 三主题 | 新 W10d1-d2 | 归入项目整合周 |
| W8d4 SGLang/投机解码 | 三主题 | 新 W7d3 / 新 W8d3 | SGLang 归调度周，投机解码归加速周 |
| W8d5-d6 分布式 | 三主题 | 新 W9d1-d4 | 归入分布式专题周 |
| W8d7 代码重构 | 三主题 | 新 W10d3 | 归入项目整合周 |
| W9d4 CUDA Graph | 新旧混杂 | 新 W8d4 | 归入推理加速技术周 |
| W9d5 MoE+EP | 新旧混杂 | 新 W9d5 | 归入分布式周（与 EP 同聚） |
| W9d6-d7 文档/架构图 | 新旧混杂 | 新 W10d3 | 归入项目整合周 |
| W10d2 Ascend | 面试周 | 新 W9d6 | 归入分布式/多硬件周 |

---

## 七、重组带来的收益（对照三维度）

| 维度 | 现状 | 重组后 |
|------|------|--------|
| **系统性**（单主题周） | W3/W4/W5/W8/W9 五周为"X+Y"拼盘 | 全部 10 周单主题，标题无"+" |
| **循序渐进**（前置不倒置） | FA 碎片化 3 周、CUTLASS 远离 WMMA、Ring Attention 远离 FA、MoE 远离 EP | FA 单周贯通；CUTLASS 紧邻 WMMA；Ring Attention 紧邻 FA（W5→W9 隔一周但同属注意力族，可接受）；MoE 紧邻 EP |
| **有节奏**（模板一致） | 5/10 周 Day7 违规；Day3-4/Day5 模板多数不遵循 | 全部 10 周遵循统一模板；Day7 真正做复盘+手撕，不引入新内容 |

### 专题深度收益

| 专题 | 现状天数 | 重组后天数 | 收益 |
|------|---------|----------|------|
| FlashAttention | 10 天碎片化 | 7 天连续 | 形成连贯心智模型 |
| Tensor Core/CUTLASS | 2 天 | 7 天 | 补 mma.sync/ldmatrix/CuTe 深度 |
| Triton | 1 天 | 2 天 | 补性能实测空间 |
| 分布式并行 | 2 天 | 5 天 | 覆盖 TP/PP/DP/NCCL/Ring/MoE/EP |
| 量化 | 1 天 | 2 天 | 补 FP8/FP4/GPTQ vs AWQ |
| MoE+EP | 1 天 | 1 天（并入分布式周） | 与 EP 前置打通 |

---

## 八、风险与权衡

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 物理目录迁移工作量大（git mv + 交叉引用） | 高 | 一次性大改 | 分两步：先改 README/plan（逻辑层），再 git mv（物理层），单独一个 commit |
| W5 FA 一周 7 天过载 | 中 | 学习者疲劳 | FA 是面试核心，值得专注一周；FA-3/官方源码列为 d6 进阶，d7 复盘可弹性 |
| W9 分布式+MoE+Ascend 三合一仍然偏多 | 中 | 单周信息密度高 | Ascend 标注"概念对照-only"（已落地），MoE 聚焦 Top-K 路由+all-to-all 两件事 |
| 与已执行的市场整改内容冲突 | 低 | 重复劳动 | 重组**只挪位置不删内容**，所有 B1–B7/C1–C5 新增内容保留，迁移对照表已映射 |
| 原 week README 的"学前导读/前置要求"需全量重写 | 中 | 文档工作量 | 随逻辑层 commit 一起更新 |

---

## 九、执行建议

### 分两阶段执行

**阶段 1（逻辑层，低风险）**：
1. 更新顶层 `README.md` 的 10 周路线表为新结构
2. 更新 `plan/learning_plan_10week.md` 为新结构（标注"v2 重组版"）
3. 更新各 week README 的"本周学习地图"+"每日学习材料"表
4. 更新各 week README 的"前置要求"指向新前置周
5. 更新 `SKILL.md` 的 LeetGPU 映射表与新 day 对齐
6. 运行 `python3 build.py` + `python3 build/lint_md_code.py` 验收

**阶段 2（物理层，单独 commit）**：
1. `git mv` 移动 day 目录到新 week（保留 git 历史）
2. 全仓库批量更新交叉引用（`weekN/dayM/` → 新编号）
3. 更新各 day README 内部的"原编号"标注
4. 再次运行 build + lint 验收

### 与既有整改计划的关系

- `remediation_plan.md` Phase 1–4：**继续有效**，事实修复/代码落盘/bug 修复与结构重组正交，互不冲突
- `market_driven_remediation_plan.md`：**已执行完成**，其新增内容（GQA/MLA、PD 分离、MoE、FP8、诊断剧本、Triton benchmark、SGLang 对比、causal FA、top-p 采样）在重组方案中**全部保留**，仅改变所属周
- 本文是**第三维度的整改**（结构/节奏），与前两者叠加后教程达到"事实准确 + 市场对标 + 结构系统"三重达标

### 验收标准

- [ ] 全部 10 个 week README 标题不含"+"
- [ ] 每个 week README 的"前置要求"指向的新前置周与本文表一致
- [ ] `learning_plan_10week.md` 更新为 v2，含迁移对照表
- [ ] `SKILL.md` LeetGPU 映射表与新 day 编号一致
- [ ] 每周 Day7 主题为"复盘/手撕/面试要点"，不引入新内容（grep 验证）
- [ ] `python3 build.py` 构建成功
- [ ] `python3 build/lint_md_code.py` 0 errors
- [ ] 提交 commit：`refactor(daily): 周级结构重组 - 单主题周 + FA专题贯通 + 分布式/MoE专题独立`

---

## 十、底线结论

当前 10 周教程**主线方向正确、内容体量充足、市场缺口已补齐**，但 `learning_plan_10week.md` 的重排为"凑满 7 天/周"把大专题切碎塞进过渡周，造成 W3 拼盘、FA 碎片化、分布式/MoE 天数不足、节奏模板名存实亡四类结构性问题。

按本文方案重组后：
- **5 个拼盘周**（W3/W4/W5/W8/W9）消失，全部变为单主题周
- **FlashAttention 从 3 周 10 天碎片**收敛为 1 周 7 天贯通
- **分布式并行从 2 天**扩为 5 天独立周，MoE+EP 与 EP 前置打通
- **节奏模板**统一恢复，Day7 真正做复盘+手撕

重组**只挪位置不删内容**，与已执行的市场整改完全兼容，是让教程从"内容齐全但散乱"升级到"内容齐全且系统渐进"的最后一公里。
