# 市场对标驱动的整改计划（2026-08）

> 基于文档：`market_benchmark_quality_review.md`（市场对标与质量评审）
> 配套文档：`remediation_plan.md`（既有整改计划，基于 `tutorial_quality_evaluation.md`）
> 制定日期：2026-08-04
> 总工期：约 8–11 周（4 个阶段）

---

## 本计划与 remediation_plan.md 的关系

本文是**增量计划，不替代** `remediation_plan.md`。两文档的分工：

- `remediation_plan.md` 的 Phase 1（P0：参数统一/事实修复/CUDA bug/代码落盘）**继续有效且正在执行**，本文不重复其任务，只补充市场评审**新发现**的问题；
- `remediation_plan.md` 的 Phase 2/3 中部分专题（Triton day3b、分布式 day3b、量化 day6b、CUDA Graph day6b、FA backward day2b、Ring Attention day4b、FA3、FlashDecoding day4b）**已以 dayNb 形式落地，但普遍缺实测数据**——本文将它们升级为"做实"任务；
- 市场评审独有、旧计划完全没有的内容（MoE/EP、PD 分离、MLA/GQA、SGLang、诊断剧本、采样 kernel、代码块缩进 bug、plan/ 收敛等）是本文的主体。

**状态对照表**（评审问题 → 归属）：

| 评审问题 | remediation_plan 是否覆盖 | 本文任务 |
|---|---|---|
| Ridge Point / 硬件参数三套矛盾 | 任务 1.1（进行中） | A1（防复发机制） |
| shuffle 代际、FA IO、KV Cache 1MB 等 15 项事实错误 | 任务 1.2 | A2（增量错误清单） |
| 9 处 CUDA bug | 任务 1.3 | A3（增量 bug：livelock、假验证、链路不通） |
| markdown 代码未落盘 | 任务 1.4（部分已落盘） | A4（预期输出全面核验） |
| README Python 缩进损坏 | **未覆盖** | A3.1 |
| WMMA/CUTLASS/Triton 专题 | 任务 2.1–2.3（已落地但无实测） | B1–B3（做实） |
| 量化专题 | 任务 3.2（day6b 已落地） | B4（FP8 实操扩展） |
| MoE/EP、PD 分离、GQA/MQA/MLA | **未覆盖** | B5–B7 |
| 诊断剧本、采样 kernel、系统设计题、投机解码深化 | 部分（任务 3.4/4.2） | C1–C5 |
| plan/ 四层收敛、周 README 漏收 dayNb、profiles 污染 | **未覆盖** | D1–D4 |

---

## 整改路线总览

```
Phase A (P0) 可信度修复          Phase B (P1) 市场硬缺口          Phase C (P2) 面试强化           Phase D (P3) 仓库卫生
┌───────────────────────┐   ┌───────────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐
│ A1 硬件参数唯一事实源   │   │ B1 WMMA 做实（85%+实测）    │   │ C1 诊断流程实战剧本      │   │ D1 plan/ 收敛     │
│ A2 增量事实错误修复     │   │ B2 CUTLASS+CuTe 铺垫       │   │ C2 手撕限时清单扩充      │   │ D2 周README收dayNb│
│ A3 可执行性修复         │──▶│ B3 Triton 扩展+性能对比    │──▶│ C3 投机解码深化          │──▶│ D3 profiles 拆分  │
│ A4 预期输出全面核验     │   │ B4 量化扩展（FP8 实操）     │   │ C4 系统设计题扩充        │   │ D4 Ascend 定位    │
│ A5 陈旧引用清理        │   │ B5 MoE+EP 专题（新）        │   │ C5 SGLang/vLLM V1 对比  │   │                  │
│ 工期：1-2 周           │   │ B6 PD 分离专题（新）        │   │ 工期：2-3 周             │   │ 工期：0.5-1 周    │
│                        │   │ B7 GQA/MQA/MLA（新）       │   │                          │   │                  │
│                        │   │ 工期：3-5 周               │   │                          │   │                  │
└───────────────────────┘   └───────────────────────────┘   └────────────────────────┘   └──────────────────┘
```

**验收原则**：与 `remediation_plan.md` 一致——每个任务完成后必须通过"验证标准"全部检查项。所有性能数字必须来自真实运行留档（数据表 + 原始输出），禁止 `x.xxx` 占位符和"假设输出"。

---

## Phase A：P0 可信度修复（工期 1–2 周）

> 目标：消除"照着做会踩坑、面试会背错"的问题。与 remediation Phase 1 并行或紧接其后执行，**必须在 Phase B 前完成**。

### 任务 A1：建立硬件参数唯一事实源（防复发机制）

**目标**：remediation 任务 1.1 用 grep 替换修参数，但没有防复发机制。本任务建立单一事实源并写入写作规范。

**操作步骤**：
1. 新建 `aiinfra/daily/reference/hardware_specs.md`，内容：RTX 5090 实测参数表（FP32 104.75 TFLOPS、GDDR7 1792 GB/s、Ridge Point 58.45、170 SM、128 cores/SM、1536 threads/SM、48 warps/SM、24 blocks/SM、100 KB smem/SM、96 MB L2、sm_120、32 GB GDDR7），附 A100/H100/B200 对比列（明确标注架构代际），每个数字标注来源（`week1/day3/exercise/my_gpu_info.md` 实测 / 官方规格书）
2. 新建 `aiinfra/daily/reference/key_numbers.md`：面试必背数字清单——Ridge Point 推导、KV Cache 口算公式（含 GQA/MQA/MLA 变体占位）、GEMM 优化增益（以 week2/day6 实测一套为准）、标准 Attention vs FA IO（统一 4N²+4Nd 口径、~50x 加速比）
3. 在 `SKILL.md` 写作规范中增加硬性条款："教程正文一律引用 `reference/hardware_specs.md` 与 `reference/key_numbers.md`，禁止在正文中新写硬件参数数字"；自检清单加对应 grep 检查
4. 全仓库数字向事实源收敛（承接 remediation 任务 1.1 的收尾）

**验证标准**：
- [ ] `reference/hardware_specs.md`、`reference/key_numbers.md` 存在且每个数字有来源标注
- [ ] `grep -rn "19\.5 TFLOP\|1\.55 TB/s\|12\.6" aiinfra/daily/week*/ aiinfra/daily/plan/learning_plan_week8_expanded.md` 仅出现在"错误示范/A100 对比"的显式标注语境
- [ ] `grep -rn "1555" aiinfra/daily/` 无结果（`week8/day1/kernels/benchmark_demo.py` 带宽改 1792）
- [ ] `SKILL.md` 自检清单含唯一事实源引用条款

**预估工时**：3–4 小时

---

### 任务 A2：增量事实错误修复（remediation 1.2 未覆盖项）

**目标**：修复市场评审新发现、旧评估未列入的错误。

| # | 文件 | 错误 | 修复方式 | 验证 |
|---|------|------|---------|------|
| 1 | `week7/day4b/README.md` 学前导读 | Llama-7B 1M token KV Cache 写 16GB（漏乘 n_layer=32） | 改为 ~524GB（2×32×32×128×2B×1M），后续表格 32K/1M/B=8 三行同步修正 | 用 `2×n_layer×n_head×d_head×N×2B` 手算复核全表 |
| 2 | `week5/day6b/README.md` §1.3/§1.5 | 同配置 KV 显存 134MB 与 64MB 自相矛盾（漏乘 n_heads），与 `week5/day2` 的 2GB 冲突 | 统一为 day2 的正确口径（~2GB/4096token），注明 per-token 公式含 n_heads | `grep -n "134\|64 ?MB" week5/day6b/README.md` 复核 |
| 3 | `week3/day1/README.md` §1.3 | `2×32×4096×512×2B = 512 MB`（应为 256MB） | 改 256MB 并复核同类算式 | 手算复核 |
| 4 | `week2/day6/README.md` L629 | "RTX 5090 有 108 个 SM" | 改 170，wave 分析（64 block 填不满）重算并核对结论方向 | 170 SM 下重算 wave 数 |
| 5 | `week2/day6b/README.md` L52-53 | Hopper/Blackwell 都标"第四代 Tensor Core" | Blackwell 改第五代 | — |
| 6 | `week4/day3/README.md` §3.3 + 面试题 3 | "cp.async 需要 Blackwell+ CC 12.0" | 改为"Ampere（CC 8.0）引入；FA 官方实现基于 sm_80" | `grep -rn "cp.async.*12.0\|12.0.*cp.async" week4/` 无结果 |
| 7 | `week8/day6/kernels/knowledge_selftest.py` L67-68 vs L142 | Ridge Point 答案 12.6 与 58.45 同文件矛盾 | 全部改 58.45，运行脚本验证 | `python3 week8/day6/kernels/knowledge_selftest.py` 两种模式答案一致 |
| 8 | `week8/day7/README.md` 任务 1 预期输出 | 写"RTX5090 ≈ 12.6"，脚本实际打印 58.45 | 文档与 `week8_summary.py` 对齐 | 运行脚本比对文档 |
| 9 | GEMM 增益三套数字（`week8/day3` 15/40/55/60/70、`week8/day6` 15/31/64/90、`week8/day5/day7` 70%） | 读者不知背哪套 | 以 week2/day6 实测为准统一为一套（含"理论 vs 实测"两列说明），其余各表引用 `reference/key_numbers.md` | `grep -rn "40%\|55%\|64%\|90%" week8/day3/README.md week8/day6/README.md` 口径一致 |
| 10 | `week1/day1/README.md` §1.5 | 寄存器单位错 4 倍（"1024 线程×64 寄存器=64KB"，实际 256KB） | 按 1024×64×4B=256KB 重写，连带修正"仍在 256KB 内"结论 | 手算复核 |
| 11 | `week1/day1/README.md` 实验 1 表格 | grid(2)×block(16) 行 warp 数写 1（应为 2，与同页公式矛盾） | 改 2 | 与同页 `ceil(16/32)×2` 公式一致 |
| 12 | `week7/day3b/README.md` 面试要点 #3 | bubble ratio 推导分子分母不自洽（2(P-1) vs (P-1)/(M+P-1)） | 重写推导：1F1B bubble = (P-1) 个 micro-batch 时间 / (M+P-1) | 公式代值复核 |
| 13 | `week7/day3/README.md` | 投机解码近似公式 kα+1 与模拟数据不吻合（2.28x vs 1.94x） | 改用精确期望 `(1-α^(k+1))/(1-α)` 或显式标注近似及误差来源 | 代值复核一致 |

**验证标准**：上表 13 项全部修复，每项附"验证"列操作留痕。

**预估工时**：4–6 小时

---

### 任务 A3：可执行性修复

**目标**：消除"照做必踩坑"的问题。

#### A3.1 README Python 代码块缩进系统性损坏（根因治理）

**现象**：week5/6 共 10 个 README 的内嵌 ` ```python ` 代码块缩进被剥成单个前导空格，复制即 `IndentationError`；`kernels/*.py` 落盘文件正常。

**操作步骤**：
1. 定位根因：检查 README 生成/排版流程（是否从 expanded plan 提取时丢失缩进、或某个格式化工具所致）
2. 用对应 `kernels/*.py` 的真实缩进回填 README 代码块（以落盘文件为准做对齐，不要手敲）
3. 在构建/自检流程加守卫：提取每个 README 的 python 代码块做 `ast.parse()`，失败即报错（可挂在 `build.py` 或独立脚本 `build/lint_md_code.py`）

**验证标准**：
- [ ] `grep -rln '```python' aiinfra/daily/week*/day*/README.md | xargs -I{} python3 -c "..."`（提取代码块 + `ast.parse`）全部通过
- [ ] lint 脚本接入 `build.py` 或 SKILL 自检清单

#### A3.2 运行时 bug 修复

| # | 文件 | 问题 | 修复方式 | 验证 |
|---|------|------|---------|------|
| 1 | `week5/day3/kernels/mini_vllm_scheduler.py` L125/L148 | 扩展实验 1（max_blocks=8）死循环：迭代 `self.waiting` 中被 `_try_preempt` insert | 迭代改为遍历快照 `for sg in list(self.waiting):`；重跑扩展实验 1，确认抢占触发后可收敛（或显式展示 livelock 并作为教学点给出退出条件） | `max_blocks=8` 运行 60s 内正常终止，README 预期输出同步更新 |
| 2 | `week5/day5/kernels/mini_engine_v0.py` L198-206 | "with/without cache 逐 token 一致"验证未执行（`gen_cache` 死变量） | 补 `assert gen_cache == gen_no_cache` 式逐 token 比较并打印结果 | 运行输出含一致性 PASS |
| 3 | `week4/day5/kernels/mini_engine_fa.py` | `load_inline(functions=["flash_attention_forward"])` 找不到定义；`flash_attention_ops.cpp` 不存在；q/k/v 非连续 | 补 host wrapper（或在 .cu 中补 `at::Tensor` 入口）；permute 后加 `.contiguous()`；README 任务 1 与磁盘文件对齐 | CUDA 环境下 `python3 mini_engine_fa.py` 通过且结果与 PyTorch 参考一致 |
| 4 | `week4/day2/kernels/flash_attention_v2.cu` | `main()` 硬编码 N=256，`week4/day6` 任务 3 传参无效 | main 解析 argv（`atoi(argv[1])`，默认值保留） | `./flash_attention_v2 1024` 生效 |
| 5 | `week5/day4/kernels/paged_attention.cu` | 隐含约束 d≤256 未声明（`o_local` 标量、`q_shm[256]` 固定） | 代码加注释 + host 侧断言；README 说明约束与改造方向 | d=128 时明确报错而非静默错误 |
| 6 | `week6/day1/kernels/dynamic_batcher.py` | 基于线程时序竞争，README 给了精确"预期输出"（6 batch/87.4ms）不可复现 | README 改为"输出因时序而异，以下为一次示例"，观察重点改为区间描述 | 连跑 3 次，README 描述与三次结果均兼容 |
| 7 | `week7/day2/kernels/full_scheduler.py` | 预期输出参数（MemoryBudget 16/32、aging 4s）与默认值（64、5.0s）不符 | README 预期输出用默认参数重跑生成 | 默认参数运行输出与文档一致 |
| 8 | 命令路径 | `week5/day6` nsys 命令缺 `daily/` 层级；week1 多天 `nvcc kernels/...` 相对路径错误（实际在 `dayN/kernels/`）；`week5/day3` 写 `vllm/engine/scheduler.py`（V0 实为 `vllm/core/scheduler.py`） | 逐一修正为可从仓库根执行的路径 | 每条命令实际执行通过 |
| 9 | 失效图片 | `week6/day4b/README.md` L44/L81 两张 SVG 不存在；`week6/day4` §4.5 表格无表头 | 补图或删引用；修表头 | 全仓库图片引用扫描 0 失效 |

#### A3.3 集成叙事诚实化

`week7` 里程碑"1000+ 请求不崩溃、性能优于 eager"与 day5 实际（500 请求、sleep 模拟）脱节。短期：里程碑改为与实际测试一致（500 请求、模拟引擎，注明局限）；长期：由 remediation 任务 3.1（Mini 引擎真整合）达成后再改回。

**验证标准**：
- [ ] A3.1 lint 守卫上线，10 个受损 README 修复
- [ ] A3.2 表 9 项全部修复并实跑验证
- [ ] `week7/README.md` 里程碑与实际测试能力一致

**预估工时**：1–2 天

---

### 任务 A4：预期输出全面核验（消灭占位符）

**目标**：所有"预期输出"二选一：实跑填真实数据，或显式标注"未验证示意"。

**操作步骤**：
1. 全仓库扫描占位符：`grep -rn "x\.xxx\|xx\.x" aiinfra/daily/week*/` 建立清单
2. 按有无 GPU 分两批：
   - 纯 Python（stdlib/numpy）：本机直接实跑回填（week5/6 的模拟器、week7/8 的自测脚本）
   - CUDA/torch：在 GPU 环境实跑回填（week2 day1/day2/day4b/day5/day6b、week3、week4 day2/day5/day6、week7 day3b/day4/day6b 等）
3. 无法实跑的标注 `> ⚠️ 以下为示意输出，未经实跑验证`
4. 把 `profiling/week1-3/` 已有真实 `.ncu-rep`/数据**链接进对应正文**（当前读者不知道其存在）
5. 补齐 `week4/day2b` gradcheck 式的"实跑留档"模式作为模板（它是全书唯一像真跑过的记录）

**验证标准**：
- [ ] `grep -rn "x\.xxx\|xx\.x" aiinfra/daily/week*/` 无结果（或每处带"示意"标注）
- [ ] 每个性能数字可追溯到实跑记录（notes/profiles 文件或 README 内嵌原始输出）
- [ ] 正文至少 3 处链接 `profiling/` 下的真实报告

**预估工时**：2–3 天（含 GPU 实跑）

---

### 任务 A5：陈旧引用与结构修复

| # | 问题 | 操作 |
|---|------|------|
| 1 | 旧编号引用（"Day 15/16/18/19/20"、"Week 2 Day 8/12" 等，散见 week3–week5） | 逐一改为 weekN/dayM 新编号；`grep -rn "Day 1[0-9]\|Day 2[0-9]" aiinfra/daily/week*/` 复核 |
| 2 | 各 day7 "本周目录结构"与实际不符（缺 `daily/` 层级、列出不存在的 `website/`、`week2/kernels/` 等） | 用 `tree` 实际输出重新生成，或删除该节 |
| 3 | 周 README 漏收 dayNb（week2: day4b/day6b；week5: day4b/day6b；week6: day4b；week7: day3b/day4b/day6b；week8: day3b） | 全部补入学习地图与每日材料表 |
| 4 | `profiles/week1_profile_summary.md` 混入 week2–5 数据 | 按周拆分为 `weekN_profile_summary.md`，week1 只留本周内容 |
| 5 | `week6/README.md`/`day6` "六大指标"只列 5 个（漏 GPU Utilization） | 补齐或改"五大" |
| 6 | week8 题数口径（README "43+" vs day4 "12 道" vs day7 "11 道"/"30 道"） | 统计真实题数，统一口径并注明统计范围 |
| 7 | week8 day3/4/5/7 LeetCode 区复制粘贴错位（编辑距离/课程表/最长有效括号/排序链表串天） | 与 remediation 任务 4.1 第 7/8 项合并处理 |

**验证标准**：
- [ ] `grep -rn "Day 1[5-9]\|Day 2[0-9]" aiinfra/daily/week*/` 无陈旧引用
- [ ] 5 个周 README 的每日材料表与磁盘目录一一对应（脚本化比对）
- [ ] `profiles/` 下文件按周归属正确

**预估工时**：3–4 小时

---

### Phase A 整体验收清单

- [ ] A1–A5 全部完成
- [ ] 唯一事实源建立并写入 SKILL.md
- [ ] 代码块 lint 守卫上线
- [ ] `grep` 占位符扫描清零（或全部标注示意）
- [ ] `python3 build.py` 构建成功
- [ ] 提交 commit：`fix(daily): P0 市场评审增量修复 - 事实源/可执行性/预期输出核验/陈旧引用`

---

## Phase B：P1 市场硬缺口（工期 3–5 周）

> 目标：补齐 2025–2026 JD 直接关键词。B1–B4 是"把已有专题做实"，B5–B7 是全新内容。算子岗优先 B1–B4，推理引擎岗优先 B5–B7。

### 任务 B1：WMMA/Tensor Core 做实（升级 remediation 2.1）

**现状**：`week2/day6b` 已存在，但预期输出全占位符；"85% cuBLAS"无实测支撑且 kernel 结构（每 block 1 warp 算 16×16、无 smem tiling）撑不起该数字；`day4b` 还把这个未证实数字当既定事实二次引用。

**操作步骤**：
1. 重写 `wmma_gemm.cu` 为有实际竞争力的结构：多 warp/block + shared memory staging + `m16n16k16` fragment + 至少一版 double buffering
2. 在 RTX 5090 上实测 1024²/2048²/4096² 三档，与 cuBLAS FP16(TF32 另列) 对比，数据留档 `week2/day6b/notes/`
3. ncu 抓 Tensor Core 利用率（`sm__pipe_tensor_op_hmma_cycles_active` 类指标）填入正文
4. 若达不到 85%，正文诚实写实际值 + 差距归因（这是 week2/day6 负面结果诚实传统的延续），并修正 day4b 的引用
5. 面试题补强：WMMA vs `mma.sync`、ldmatrix、FP16 累加精度策略、对齐约束

**验证标准**：
- [ ] `nvcc -O3 -arch=sm_120 -lcublas week2/day6b/kernels/wmma_gemm.cu && ./a.out` 通过，正确性校验 PASS
- [ ] 正文性能表为实测留档数据，无任何占位符
- [ ] `day4b` 引用的数字与 day6b 实测一致

**预估工时**：2–3 天

---

### 任务 B2：CUTLASS + CuTe 铺垫（升级 remediation 2.2）

**现状**：`week2/day4b` 有 CUTLASS 概念但无验证输出，且依赖 day6b 却排在前面；`week4/day3` 要求读官方 FA 源码（CuTe 风格）却无 CuTe 前置——两次能力跳跃。

**操作步骤**：
1. 在 `week2/day4b` 前补 CuTe 最小铺垫（Layout/Tensor/`local_tile` 概念半日量，或作为 day4b 的第 0 节）
2. `cutlass_gemm_example.cu` 在目标环境编译跑通，输出实测性能并与 cuBLAS/WMMA 版对比留档；写明 CUTLASS 版本与 sm_120 编译注意事项（ArchTag 选择、CUTLASS 2.x vs 3.x 对新架构的支持差异）
3. 调整引用顺序：day4b 学前导读不再前向引用 day6b 的数字（B1 完成后可回填实测值）
4. `week4/day3` 源码导读改为"带行号的导读任务"：给出 `flash_fwd_kernel.h` 关键结构的实际数值（kBlockM/kBlockN 等）供核对

**验证标准**：
- [ ] CUTLASS 示例编译运行通过并留档性能数据
- [ ] 周地图顺序与依赖关系一致（无前向引用未学内容）
- [ ] `week4/day3` 导读任务可独立完成（含核对答案）

**预估工时**：2–3 天

---

### 任务 B3：Triton 扩展 + 性能对比（升级 remediation 2.3）

**现状**：`week3/day3b` 一天内从 program 模型塞到 FA，无性能实测。

**操作步骤**：
1. 拆为两天或一天半：day3b（program 模型 + softmax/GEMM），新增 day3c 或并入 week4（Triton FA + autotune）
2. 每个 kernel 与对应 CUDA 手写版、PyTorch 原生做三方 benchmark（do_bench，warmup/rep 明确），数据留档
3. 补"什么时候用 Triton、什么时候必须 CUDA"的决策表（对应面经高频追问）
4. autotune 配置空间搜索过程留档（最佳 config 与直觉的差异是面试好素材）

**验证标准**：
- [ ] 三个 Triton kernel 均有与 CUDA 版的实测性能对比表
- [ ] 面试题含"CUDA vs Triton trade-off"标准答案（有数据支撑）

**预估工时**：2 天

---

### 任务 B4：量化扩展——FP8 实操（升级 remediation 3.2）

**现状**：`week5/day6b` 已有 W8A16/INT8 KV 概念与 kernel，缺 FP8 实操、FP4、GPTQ vs AWQ vs SmoothQuant 深入对比；且 §1.3/§1.5 数字矛盾（A2 修）。

**操作步骤**：
1. 新增 FP8 章节：E4M3/E5M2 格式细节（位布局、动态范围、为何 E4M3 权重/E5M2 梯度）、per-tensor vs per-block（DeepSeek 128×128 细粒度）量化策略
2. Coding：FP8 GEMM 或 FP8 dequant kernel（RTX 5090 sm_120 支持 FP8；可用 `__nv_fp8_e4m3` 或 Triton `tl.float8e4nv` 降低门槛），精度/性能对比留档
3. GPTQ vs AWQ vs SmoothQuant 三方对比表（原理、校准数据需求、精度、适用场景）+ KV Cache 量化风险分析（对 attention 长序列的误差累积）
4. FP4（NVFP4）概念一节：Blackwell 新特性、与 FP8 的取舍——2026 JD 已出现
5. 面试题 ≥5 道新增

**验证标准**：
- [ ] FP8 kernel 实测数据留档（精度 max_diff + 性能对比）
- [ ] 三方量化对比表完整
- [ ] `grep -rn "FP8" aiinfra/daily/week8/` 面试题库中可查

**预估工时**：2–3 天

---

### 任务 B5：新增 MoE + EP 并行专题（全新，市场最热缺口）

**落位建议**：`week7/day5b/`（分布式 day3b 之后）

**教程内容要求**：
- 理论 1：MoE 结构（Top-K 路由、load balancing loss、capacity factor）
- 理论 2：EP 并行的通信模式（all-to-all dispatch/combine）、大 EP 部署（EP32/EP144）、DeepEP/EPLB 概念
- 理论 3：MoE 推理的显存/通信权衡（为何 decode 阶段 EP 优于 TP）
- Coding 1：Top-K 路由 kernel（Triton 或 CUDA，含 softmax + top-k + 计数）
- Coding 2：all-to-all 通信量计算 + 单机模拟（torch.distributed 或 numpy）
- 面试题 ≥5 道（aux-loss-free 均衡、EP vs TP 选择、all-to-all 通信量推导、DeepSeek MTP/MoE 结构参数）

**验证标准**：
- [ ] Top-K 路由 kernel 正确性验证通过
- [ ] all-to-all 通信量公式推导完整（类比 day3b NCCL 推导的深度）
- [ ] 面试题 ≥5 道并纳入 week8 题库

**预估工时**：3–4 天

---

### 任务 B6：新增 PD 分离专题（全新，JD 直接关键词）

**落位建议**：`week6/day5b/`（chunked prefill 之后，mini 引擎 v1 之前）

**教程内容要求**：
- 理论 1：为什么分离（prefill compute-bound vs decode memory-bound 的资源错配、TTFT/TPOT SLO 矛盾）——直接复用 week3/day1 与 week5/day1 的 roofline 结论
- 理论 2：Mooncake / DistServe / vLLM disaggregated serving 的架构（KV 传输层、调度层）
- 理论 3：KV Cache 跨节点传输（RDMA/NVLink、传输量计算：KV bytes/token × 序列长度）
- Coding：在 `continuous_batcher.py`/`chunked_prefill_simulator.py` 基础上扩展一个双角色模拟（prefill 池 + decode 池 + KV 队列），量化 TTFT/TPOT 改善
- 面试题 ≥5 道（PD 分离动机、KV 传输开销怎么算、什么流量特征下不划算、与 chunked prefill 的关系）

**验证标准**：
- [ ] PD 分离模拟器可运行，有 colocated vs disaggregated 的 TTFT/TPOT 对比数据
- [ ] 面试题 ≥5 道

**预估工时**：2–3 天

---

### 任务 B7：KV Cache 章节补 GQA/MQA/MLA（全新，小改动大收益）

**落位**：扩展 `week5/day2/README.md` 与面试题库

**操作步骤**：
1. 在显存计算一节后加"注意力变体对 KV Cache 的影响"：MHA → GQA（n_kv_head 缩减）→ MQA → MLA（低秩压缩到 d_c 维度）的公式与口算示例（LLaMA-7B MHA vs GQA-8 vs MLA 每 token KV bytes 对比表）
2. 更新 `reference/key_numbers.md` 的口算公式为含 `n_kv_head` 的一般形式
3. week8 题库新增 3 道（GQA 显存口算、MLA 原理、为何 MLA 的 KV Cache 这么小）

**验证标准**：
- [ ] `week5/day2` 有变体对比表且数字经手算复核
- [ ] 面试题 ≥3 道入库

**预估工时**：0.5–1 天

---

### Phase B 整体验收清单

- [ ] B1–B7 全部完成，新增专题均有实测数据留档
- [ ] 新增 ≥25 道面试题（MoE/EP、PD 分离、FP8、GQA/MLA、WMMA）
- [ ] `python3 build.py` 构建成功
- [ ] 提交 commit：`feat(daily): P1 市场缺口补齐 - WMMA实测/Triton对比/FP8/MoE+EP/PD分离/GQA-MLA`

---

## Phase C：P2 面试针对性强化（工期 2–3 周）

### 任务 C1：诊断流程题实战剧本

**目标**：覆盖"低 MFU / OOM / hang 怎么排查"这类几乎所有公司都问的流程题，从方法论升级为实战剧本。

**落位**：扩展 `week7/day6/` 或新增 `week8/day6b/`

**操作步骤**：
1. 制作 3 个故障注入案例（在 mini 引擎或 GEMM 系列上）：
   - 案例 1：人为制造低 MFU（如 GEMM 换 naive 版/关 Tensor Core）→ 走 nsys 看 gap → ncu 看 SM%/stall → 定位
   - 案例 2：人为制造 OOM（KV Cache 无分页泄漏）→ torch.cuda.memory 快照 → 定位到 allocator 行为
   - 案例 3：人为制造 hang（多 stream 错误依赖/死锁，可复用 A3.2 修的 livelock 作为教学案例）→ cuda-gdb / NCCL 日志（`NCCL_DEBUG=INFO`）排查
2. 每个案例写成"现象 → 假设 → 工具 → 证据 → 结论"五段式剧本，作为面试口述模板
3. 面试题 ≥3 道（附 STAR 式回答框架）

**验证标准**：
- [ ] 3 个案例均可复现（含注入开关），排查过程有真实工具输出留档
- [ ] 面试题 ≥3 道

**预估工时**：2–3 天

---

### 任务 C2：手撕限时清单扩充

**目标**：补齐市场高频手撕题缺口并全部限时化。

**操作步骤**：
1. 新增采样 kernel 专题：`week5/day5` 扩展实验升级为正式内容——top-k / top-p（nucleus）采样 kernel（CUDA 或 Triton），含 temperature scaling；这是面经高频且与 mini 引擎 argmax 现状直接衔接（顺手把引擎 sampler 做成可插拔）
2. 手写 FA 增加 causal 变体（现有手写版全是 non-causal，面试默认 causal）：在 `week4/day2` kernel 上加 causal mask 分支 + 块级跳过优化（causal 下可省一半块计算）
3. 限时手撕清单整合：把 week2/day7（Reduce 30min/GEMM 60min）模式推广成一张全课程手撕清单（reduce/softmax/layernorm/transpose/GEMM tiling/online softmax/FA 简化版/top-p/调度循环），每项标注限时与通过标准，放入 week8/day5 mock 流程

**验证标准**：
- [ ] top-p 采样 kernel 正确性验证通过与性能留档
- [ ] causal FA kernel 正确性验证通过，且相对 non-causal 有加速比数据
- [ ] 手撕清单 ≥10 项，每项有限时与通过标准

**预估工时**：2–3 天

---

### 任务 C3：投机解码深化

**现状**：`week7/day3` 只有简化模拟，近似公式与数据不吻合（A2 第 13 项修公式）；市场面经已考接受率调优与 Medusa/EAGLE/MTP 细节。

**操作步骤**：
1. 理论加深：接受率的精确期望、Medusa（多头草稿）/EAGLE（特征层草稿）/DeepSeek MTP 三条路线对比
2. 模拟升级：`advanced_features.py` 的投机解码模拟加入"接受率 vs 加速比"扫描曲线（k=1..8, α=0.5..0.9），留档数据
3. 选做（依赖 remediation 3.1 真引擎）：draft+target 双模型真实集成
4. 面试题 ≥3 道（为何接受率决定上限、draft 模型怎么选、verify 的 kernel 实现要点）

**验证标准**：
- [ ] 接受率扫描曲线数据留档，公式与模拟吻合
- [ ] 面试题 ≥3 道

**预估工时**：1–2 天

---

### 任务 C4：系统设计与场景题扩充

**现状**：week8 只有 2 道场景题且答案较泛；无行为面/项目话术。

**操作步骤**：
1. 新增 4 道系统设计题（写入 `week8/day4` 或新章节）：
   - 百万 QPS LLM serving 系统设计（异构调度、PD 分离、成本优化——串联 B5/B6 新内容）
   - 给定 SLO（TTFT<200ms, TPOT<50ms）反推集群配置与并行策略
   - 万卡训练系统设计（若目标含训练岗；可标注选学）
   - 推理成本分析（$/1M tokens 的构成：GPU 时价 × 利用率 × 吞吐）
2. 每题给"澄清需求 → 估算 → 架构 → 权衡 → 延伸"五段式参考回答
3. 补项目话术模板：把 week2/day6 GEMM 优化、mini 引擎整合两个项目各写一份 STAR 叙述示例（含可被追问的 3 层细节与诚实局限声明——直接复用负面结果诚实报告的传统）

**验证标准**：
- [ ] 系统设计题总数 ≥6 道，均有五段式参考答案
- [ ] STAR 项目话术 ≥2 份

**预估工时**：1–2 天

---

### 任务 C5：SGLang/RadixAttention 对比 + vLLM V1 收尾

**操作步骤**：
1. 在 `week6/day4b`（已手写 block-hash prefix caching）后补 RadixAttention 一节：前缀树 vs block hash 两条路线对比、SGLang 与 vLLM 的 prefix caching 机制差异、何时 radix tree 更优（共享前缀多且长）
2. 扩展 `week6/day4` 框架对比表：加入 SGLang（并修复该表无表头的格式 bug，A3.2 第 9 项）
3. vLLM V1 内容按 remediation 任务 4.3 执行，此处只补市场视角：V1 的哪些变化是面试考点（默认开启 chunked prefill/prefix caching 的设计动机）

**验证标准**：
- [ ] RadixAttention 章节 ≥60 行 + ≥2 道面试题
- [ ] 框架对比表含 vLLM/TRT-LLM/SGLang/LightLLM 四列且格式正确

**预估工时**：1 天

---

### Phase C 整体验收清单

- [ ] C1–C5 全部完成
- [ ] 手撕清单 ≥10 项、诊断剧本 3 个、系统设计题 ≥6 道
- [ ] 新增 ≥15 道面试题
- [ ] 提交 commit：`feat(daily): P2 面试强化 - 诊断剧本/采样kernel/causal FA/系统设计题/SGLang对比`

---

## Phase D：P3 仓库卫生（工期 0.5–1 周）

| # | 任务 | 操作 | 验证 |
|---|------|------|------|
| D1 | plan/ 四层文档收敛 | `AI_Infra_8_week_plan.md`、`AI_Infra_8_week_plan_detailed.md` 头部加"⚠️ 已过时，以 weekN/ 教程与 expanded plan 为准"或移入 `plan/archive/`；expanded plan 头部加版本标注与"代码以落盘文件为准"声明；补建 `learning_plan_week1_expanded.md` 或在索引中说明缺失原因 | plan/ 内每个文件头部有状态标注 |
| D2 | SKILL.md 同步 | LeetGPU 映射表与实际教程对齐（修复 week7 连续 6 天 Matrix Transpose 违反自身规则的问题，与 remediation 4.1 第 5 项合并）；前置阅读清单补 week4–8 expanded plan；写入唯一事实源条款（A1）与 lint 守卫（A3.1） | SKILL 自检清单可执行 |
| D3 | profiles 目录规范 | 按周拆分（A5 第 4 项）+ 每个 profile 文件头部注明环境与日期；`profiling/README.md` 加索引 | 读者可从任一教程页跳到对应真实数据 |
| D4 | Ascend 专题定位 | `week8/day3b` 二选一：补一个可执行练习（有昇腾环境时），或头部明确标注"概念对照-only，无可执行验证"；`requirements.txt` 补 numpy（`ring_attention_sim.py` 依赖） | 定位标注明确；`pip install -r requirements.txt` 后 stdlib/numpy 脚本全可跑 |

**验证标准**：
- [ ] 4 项全部完成
- [ ] `python3 build.py` 构建成功
- [ ] 提交 commit：`chore(daily): P3 plan收敛/SKILL同步/profiles规范/Ascend定位`

---

## 整体里程碑与时间线

| 里程碑 | 内容 | 预计工期 | 累计 |
|--------|------|---------|------|
| M1 | Phase A 完成（可信度修复） | 1–2 周 | 2 周 |
| M2 | Phase B 完成（市场硬缺口） | 3–5 周 | 5–7 周 |
| M3 | Phase C 完成（面试强化） | 2–3 周 | 7–10 周 |
| M4 | Phase D 完成（仓库卫生） | 0.5–1 周 | 8–11 周 |

**关键路径**：A（尤其 A1 事实源、A3 可执行性）→ B1/B2（WMMA/CUTLASS 依赖 A4 实跑环境）→ C1（诊断剧本依赖 A3.2 修复后的案例）→ D。

**与 remediation_plan 的依赖**：本计划 A2/A3 与 remediation 任务 1.2/1.3 同性质，可合并执行；B1–B4 是 remediation 2.1/2.2/2.3/3.2 的"做实"升级，若对应任务尚未执行则直接按本文标准执行（含实测要求）；C3 选做项依赖 remediation 3.1。

---

## 整改后预期效果对照

| 维度 | 现状（评审结论） | 整改后目标 |
|------|----------------|-----------|
| 事实准确性 | ★★（Ridge Point 三版本、KV 显存多处矛盾） | ★★★★★（唯一事实源 + lint 守卫） |
| Profiling 实操 | ★★（占位符与假设数据） | ★★★★（全部实测留档或显式标注） |
| 市场对标覆盖 | ★★（缺 MoE/EP、PD 分离、FP8、MLA） | ★★★★（JD 直接关键词全覆盖） |
| 算子深度 | WMMA 85% 无实测、CUTLASS 无输出 | WMMA/CUTLASS/Triton 均有实测数据 |
| 面试题量与结构 | ~250 题但缺新热点、场景题 2 道 | +40 题新热点，场景题 ≥6 道，手撕清单 ≥10 项限时 |
| 岗位适配 | 推理引擎岗强、算子岗弱 | 双轨均可（算子岗 B1–B4，引擎岗 B5–B7） |

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| GPU 环境不可用（A4/B1/B3/B4 实跑受阻） | 中 | 实测任务延期 | 云 GPU（AutoDL 等）；未跑通的显式标注"待实测"，禁止回填编造数字 |
| MoE/EP 内容膨胀（B5 主题过大） | 中 | 工期超预期 | 聚焦 Top-K 路由 kernel + all-to-all 通信量两件事，集群部署细节只给阅读链接 |
| 与 remediation_plan 并行执行产生冲突编辑 | 中 | 合并冲突/重复劳动 | 同文件修改先查 remediation 任务状态；A2/A3 与任务 1.2/1.3 合并提交 |
| Triton FP8 在 sm_120 的支持差异 | 低 | B4 kernel 需降级方案 | 备用方案：CUDA `__nv_fp8_e4m3` 手动 dequant kernel |
| 新增专题打乱 8 周节奏 | 低 | 教程膨胀 | 一律以 dayNb 补充日插入并更新周 README（A5 已规范），主线不动 |

---

## 附录：任务全量清单（按优先级排序）

| 优先级 | 编号 | 任务 | 预估工时 | 依赖 |
|--------|------|------|---------|------|
| P0 | A1 | 硬件参数唯一事实源 + SKILL 条款 | 3–4h | — |
| P0 | A2 | 增量事实错误修复（13 项） | 4–6h | A1 |
| P0 | A3 | 可执行性修复（缩进 lint/运行时 bug/叙事诚实化） | 1–2d | — |
| P0 | A4 | 预期输出全面核验（消灭占位符） | 2–3d | A3 |
| P0 | A5 | 陈旧引用与结构修复（7 项） | 3–4h | — |
| P1 | B1 | WMMA/Tensor Core 做实 | 2–3d | A4 |
| P1 | B2 | CUTLASS + CuTe 铺垫 | 2–3d | B1 |
| P1 | B3 | Triton 扩展 + 性能对比 | 2d | A4 |
| P1 | B4 | 量化扩展（FP8 实操） | 2–3d | B1 |
| P1 | B5 | MoE + EP 专题（新） | 3–4d | — |
| P1 | B6 | PD 分离专题（新） | 2–3d | — |
| P1 | B7 | GQA/MQA/MLA（新） | 0.5–1d | A1 |
| P2 | C1 | 诊断流程实战剧本（3 案例） | 2–3d | A3 |
| P2 | C2 | 手撕限时清单（采样 kernel + causal FA） | 2–3d | B3 |
| P2 | C3 | 投机解码深化 | 1–2d | A2 |
| P2 | C4 | 系统设计与场景题扩充 | 1–2d | B5, B6 |
| P2 | C5 | SGLang/RadixAttention + vLLM V1 | 1d | — |
| P3 | D1 | plan/ 四层收敛 | 2–3h | — |
| P3 | D2 | SKILL.md 同步 | 2h | A1, A3 |
| P3 | D3 | profiles 目录规范 | 2h | A5 |
| P3 | D4 | Ascend 定位 + requirements 修复 | 1h | — |
