# AI Infra 教程市场对标与质量评审（2026-08）

> 评审范围：`aiinfra/daily/` 下全部 Markdown（week1–week8 + plan/ + 顶层文档，共约 90 个文件）。
> 评审方式：分周全文通读 + 纯 Python 脚本实跑验证 + 对照 2025–2026 年约 20 份市场 JD（字节 Seed、月之暗面、腾讯、美团、MiniMax、天数智芯、NVIDIA、OpenAI、Anthropic 等）与面经高频考点做差距分析。
> 与既有文档的关系：本文是对 `tutorial_quality_evaluation.md` 的市场视角补充与复核，两份文档结论基本一致；`remediation_plan.md` 的 P0 整改已部分落地。

---

## 一、总体评价：7/10

这是同题材自学材料里**结构设计和面试闭环做得最好的一档**，主线（CUDA 基础 → kernel 优化 → FlashAttention → 推理系统 → 调度 → 分布式 → 面试）与市场技能树高度吻合。但两类问题拖累了它作为"求职硬通货"的可信度：

1. **大量可核对的事实错误和数字口径矛盾**——对一份"教人用精确数字打动面试官"的教程，恰好打在卖点上；
2. **市场对标缺口**——算子深度（Tensor Core/CUTLASS 实测）和 2025 新热点（MoE/EP、PD 分离、FP8 实操）不足。

### 各维度评分

| 维度 | 评分 | 说明 |
|---|---|---|
| 路线设计 | ★★★★★ | 8 周递进合理，学前导读/衔接清单执行到位 |
| 概念与推导 | ★★★★ | online softmax、NCCL 通信量、FlashDecoding 合并公式等推导扎实 |
| 面试 Q&A 体系 | ★★★★ | ~250+ 题带答案、自测脚本、mock 面试、STAR 模板，闭环完整 |
| 工程实战 | ★★★ | mini 引擎多为 sleep 模拟，"整合"叙事部分名不副实 |
| Profiling 实操 | ★★ | 大量"预期输出"是占位符或虚构假设数据 |
| 事实准确性 | ★★ | 详见第三节硬伤清单 |
| 市场对标（2025–26） | ★★ | 推理系统侧较好，算子侧和新热点缺口明显 |

### 分周评分

| 周 | 评分 | 要点 |
|---|---|---|
| week1 | 7/10 | occupancy 手算题优秀、有真实实测；多处硬伤（warp 数表、寄存器单位错 4 倍、Ridge Point 三版本）、承诺的 reduction/matmul 本地 kernel 未交付 |
| week2 | 7.5/10 | day3/day5/day6（实测+诚实报告负面结果）是上乘水准；day2/day4b/day6b 占位符式"预期输出"、多处事实错误 |
| week3 | 7/10 | 主线设计优秀、面试题密度高；Ridge Point 矛盾、KV Cache 计算错误、陈旧 Day 编号引用 |
| week4 | 7/10 | FA 推导扎实（day2b gradcheck 疑似真跑过）；164KB smem 等 A100 残留、cp.async 架构要求写错、day5 集成链路跑不通 |
| week5 | 7/10 | vLLM 调度器拆解深；KV 显存数字跨文件矛盾、扩展实验死循环、day5 核心验证名不副实 |
| week6 | 7/10 | day3 源码级拆解与 day6 benchmark（实测吻合）是亮点；day4b 预期输出全面过时、README 代码块缩进系统性损坏 |
| week7 | 7.5/10 | 理论深度好（NCCL/Ring Attention 推导）、脚本大多实测可跑；day4b KV Cache 漏乘层数（16GB vs 应 ~524GB） |
| week8 | 7.5/10 | 自测→mock→查漏→复盘闭环完整；Ridge Point/GEMM 增益数字矛盾、复制粘贴残留 |
| plan/ | 6.5/10 | 评估与整改两份文档堪称范本；但四层金字塔互相矛盾、expanded plan 是教程大部分错误的源头 |

---

## 二、对照市场需求的覆盖度分析

### 覆盖良好（教程强项）

- **CUDA 基础**（SM/Warp/Occupancy/Bank Conflict，≈95% JD 要求）：week1–2 覆盖充分，occupancy 手算 6 题是亮点
- **手写 kernel 高频题**（reduce、softmax、layernorm、transpose、GEMM、attention）：基本都有本地代码
- **Roofline 分析框架**：反复训练（但数字有错，见第三节）
- **推理系统核心**：FlashAttention 原理与 IO 推导、PagedAttention、Continuous Batching、Chunked Prefill、vLLM 调度器源码拆解、Prefill/Decode 与 TTFT/TPOT 指标体系——覆盖到位

### 覆盖不足或半吊子

- **Triton**（≈70% JD，2025 后近乎必备）：只有 `week3/day3b` 一天，从 program 模型到 FA 一天塞完，无性能实测对比
- **Tensor Core/WMMA/CUTLASS**（算子岗核心追问点）：`week2/day6b` 的 WMMA "达 cuBLAS 85%" 无任何实测（预期输出全是 `xx.x` 占位符，kernel 结构是最朴素的单 warp 无 tiling，数字可疑）；`day4b` CUTLASS 无验证输出、依赖 day6b 却排在前面。面试被追问"手撕 WMMA GEMM"会露馅
- **量化**（≈70% JD）：只有 `week5/day6b` 一天（W8A16/INT8 KV 概念），缺 FP8 E4M3/E5M2 实操、GPTQ vs AWQ vs SmoothQuant 深入对比、FP4
- **分布式**（≈75% JD）：`week7/day3b` TP/NCCL 推导质量好，但只有模拟无真实多卡实验；bubble ratio 推导写错
- **Profiling 工具**（≈60% JD）：方法论好，但 ncu 数据多为"假设输出"，面试缺真实定位案例素材

### 明显缺失（2025–2026 JD 直接关键词）

- **MoE / EP 并行**：DeepSeek 带动的最热方向，EP32/大 EP 部署、all-to-all、DeepEP/EPLB 已进 JD，教程零覆盖
- **PD 分离（Prefill-Decode Disaggregation）**：腾讯、天数智芯 JD 技能标签原文，教程未提及
- **MLA / GQA / MQA**：KV Cache 章节算了显存却不讲这三个直接改变公式的结构，口算题必考变体
- **SGLang / RadixAttention**：`week6/day4b` 手写了 block-hash prefix caching 却不提 RadixAttention 主流路线
- **投机解码深度**：`week7/day3` 只有简化模拟，市场面经已考到接受率调优、Medusa/EAGLE/MTP 实现细节
- **采样 kernel（top-k/top-p）**：引擎全程 argmax，恰是面试常考手撕题
- **诊断流程题**（低 MFU/OOM/hang 排查）：几乎所有公司都问，教程有方法论无实战剧本
- **训练侧**（FSDP/ZeRO/混合精度）：完全缺席——纯推理岗可接受，目标含训练 infra 则是大缺口
- **Ascend/国产芯片**：≈35% 国内 JD 提及且在增长，`week8/day3b` 只有概念对比表无可执行练习

---

## 三、质量问题：必须修的硬伤

### 3.1 数字口径系统性混乱（最伤）

- **Ridge Point 三个版本并存**：12.6（A100 残留）/ 25 / 58.45（RTX 5090 实测正确值），遍布 week1–week8 正文、面试答案、自测脚本（`week8/day6/kernels/knowledge_selftest.py` 内部自相矛盾）。`learning_plan_week8_expanded.md` 把错误的 12.6 放进"必须熟练背诵的关键参数"清单——学习者会在面试里背出错答案
- **KV Cache 显存多处矛盾**：`week3/day1` 256MB 误写 512MB；`week5/day2` 的 2GB（正确）vs `week5/day6b` 的 134/64MB（漏乘 n_heads）；`week7/day4b` 1M token 写 16GB（漏乘层数，应 ~524GB）
- **GEMM 优化增益三套数字**：40%→55%→70% vs 31%→64%→90% vs 实测 30.8%，week8 复习时读者无所适从
- **显存带宽两套数字**：1555（`week8/day1` benchmark_demo.py，还误标"HBM"）vs 1792 GB/s

### 3.2 事实性错误（面试中直接说错）

- "Blackwell 起必须用 `_sync` 版 shuffle"（实际 Volta/CUDA 9 起）——`week2/day1`
- "`cp.async` 需要 Blackwell+ CC 12.0"（实际 Ampere CC 8.0 引入）——`week4/day3`
- Tensor Core 代际标错（Blackwell 是第五代非第四代）——`week2/day6b`
- RTX 5090 "108 SM"（实际 170）——`week2/day6`；"RTX 5090 HBM 峰值"（5090 是 GDDR7）——`week8/day1`
- B200 行疑填 H100 参数（132 SM / 989 TFLOPS sparse）——`week1/day1`
- 164KB smem / 40MB L2 等 A100 参数残留——`week4/day1/day2/day3`、`week3/day4`
- `week1/day1` 寄存器单位换算错 4 倍（"64K 寄存器"误写"64 KB"）；`week1/day5` bank conflict 定义与实验自相矛盾

### 3.3 可执行性问题（照做会踩坑）

- **10 个 README 的内嵌 Python 代码块缩进被系统性剥成单空格**，复制即 `IndentationError`——疑似构建/生成流程 bug
- 大量"预期输出"是占位符或与实际不符：`week6/day4b` 预期输出全面过时（实跑 3.00x vs 文档 1.29x）；`week8/day7` 写 Ridge≈12.6 而脚本实际打印 58.45
- `week5/day3` 扩展实验 1 按文档操作会**死循环**（迭代中修改 waiting 列表导致 livelock）
- `week4/day5` 的 `mini_engine_fa.py` 链路跑不通（load_inline 找不到函数定义 + 非连续张量 bug）
- `week5/day5` 宣称的"with/without cache 逐 token 一致性验证"实际没执行（`gen_cache` 是死变量）
- 编译路径指令错误：`cd week1 && nvcc kernels/...`（实际在 `dayN/kernels/`）；nsys 命令缺 `daily/` 层级

### 3.4 结构性问题

- 陈旧引用遍布：旧编号 "Day 18/19"、`vllm/engine/scheduler.py` 错误路径（V0 实际为 `vllm/core/scheduler.py`）
- 周 README 漏收 7 个 dayNb 补充日（恰是内容最深的几天：WMMA、量化、TP、Ring Attention、CUDA Graph、Ascend）
- plan/ 四层金字塔（minimal/detailed/expanded/教程）互相矛盾且无版本标注；**expanded plan 是教程大部分错误的源头**（shuffle 代际、FA IO、"3N²+4Nd"等均复制自 plan）
- week5/6 承诺"Week 6 手写 CUDA Graph"未交付（实际挪到 week7/day6b，交叉引用未更新）
- `profiles/week1_profile_summary.md` 混入 week2–5 数据，范围污染

---

## 四、改进建议（按优先级）

### P0 —— 修复可信度（约 1–2 周，大部分已在 remediation 计划中）

1. 建"唯一事实源"硬件参数文件（如 `daily/reference/hardware_specs.md`），SKILL.md 强制引用而非各自抄写数字；Ridge Point / KV Cache 口算 / GEMM 增益收敛到一套有出处的值
2. 逐项修掉事实错误清单和陈旧引用（大多可 grep 验证）
3. 修 README Python 代码块缩进的生成 bug，在构建流程加"代码块可执行性"检查
4. 所有"预期输出"二选一：实跑填真实数据（像 `week2/day6` 那样连负面结果都诚实报告——这是全书最好的段落），或明确标注"未验证示意"；把 `profiling/weekN/` 已有真实 .ncu-rep 链接进正文

### P1 —— 补市场硬缺口（决定能否过 2026 年面试）

5. **做实 Tensor Core 专题**：WMMA GEMM 实测到 cuBLAS 85%+ 并留档数据；补 CuTe 基础后再读官方 FA 源码（现 `week4/day3` 对没接触 CuTe 的读者不可执行）
6. **Triton 扩到 2–3 天**：softmax/GEMM/FA 各一个，与 CUDA 版做性能对比实测
7. **量化专题扩展**：FP8 E4M3/E5M2 实操 kernel、GPTQ vs AWQ vs SmoothQuant、FP4 概念、KV Cache 量化风险
8. **新增 MoE + EP 专题**：Top-K 路由 kernel、all-to-all 通信量、DeepEP/EPLB 概念——2026 推理岗最热
9. **新增 PD 分离专题**：Mooncake/DistServe 设计动机 + mini 模拟（已有 continuous batching 模拟框架，扩展成本低）
10. **KV Cache 章节补 GQA/MQA/MLA 变体**及口算公式——改动小、面试收益大

### P2 —— 面试针对性强化

11. 加"诊断流程题"实战剧本：故障注入案例（低 MFU/OOM/hang），走一遍 nsys→ncu→NCCL 日志排查流程
12. 手撕题限时化（已有 day7 雏形），补 top-p/top-k 采样 kernel、causal 版 FA（现有手写版全是 non-causal）
13. 场景/系统设计题扩充（百万 QPS serving、成本分析）；补真实项目话术（STAR 模板已有）
14. 补 SGLang/RadixAttention 与 vLLM 对比、vLLM V1 架构更新

### P3 —— 仓库卫生

15. plan/ 四层文档收敛：标注 minimal/detailed 已过时或归档
16. 各周 README 收录全部 dayNb 补充日；`profiles/week1_profile_summary.md` 按周拆分
17. Ascend 专题（`week8/day3b`）要么补可执行练习，要么明确标注"概念对照-only"

---

## 五、底线结论

- **目标是推理引擎方向 AI Infra 岗**：修完 P0 数字硬伤后即具备很强竞争力，主线与 JD 重合度高
- **目标是算子/高性能 kernel 岗**：当前 Tensor Core/CUTLASS/Triton 实测深度不够——这三项正是 2026 年算子岗面试的分层点，优先做 P1 第 5、6 条
- **趋势提示**：AI 辅助 coding 轮正在普及（Meta 已试点），面试重心向"瓶颈分析 + 验证 AI 输出 + tradeoff 权衡"迁移。教程里"诚实报告负面结果""profile 驱动决策"的段落恰好是优势，值得放大为单独的方法论章节
