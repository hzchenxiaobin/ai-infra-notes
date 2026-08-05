# AI Infra 10 周课程 · 整改方案与执行计划

> 依据：`plan/course_route_review.md`（整体路线评审 + Week 1–10 逐日评审）。
> 制定日期：2026-08-05
> 原则：**先修正确性，再清迁移残留，后做结构重构，最后补增强**。每个工作流可独立交付、独立验收。

---

## 一、问题全景与整改工作流划分

评审发现的全部问题归入五个工作流（Workstream），按优先级排序：

| 工作流 | 性质 | 优先级 | 预估工作量 |
|--------|------|--------|-----------|
| WS-1 正确性红线 | 知识性错误 + 代码 bug，学了会学错 | **P0** | 1–2 天 |
| WS-2 迁移残留清理 | 旧 8 周/旧日号/悬空引用/标题重复，体验第一杀手 | **P1** | 2–3 天 |
| WS-3 主项目贯穿线 | Mini 引擎真整合、真实测 | **P2** | 2–3 天 |
| WS-4 W9 结构重构 + 分布式真代码 | 去重、归位、补第一段分布式真代码 | **P2** | 2–3 天 |
| WS-5 空缺补齐与增强 | FP8 实操、vLLM 真部署、工程基建、配套欠账 | **P3**（按 ROI 选做） | 3–5 天 |

---

## 二、WS-1：正确性红线（P0）

> 验收标准：改动处内容在技术上正确；受影响脚本实际运行通过。

### 1.1 知识性错误（6 项）

| # | 位置 | 问题 | 修法 |
|---|------|------|------|
| 1 | `week9/day2` §2.4 | interleaved 1F1B 公式 `(P-1)/(V·M+P-1)×V` 与表格自相矛盾（`7/39×2` 写成 18%），且 M→∞ 时退化为标准 1F1B | 改为 `(P-1)/(V·M+P-1)`，重算对照表，同步修正"今日总结"与面试要点 2 |
| 2 | `week9/day4` §4.2 | 示例中 `compute_stream.wait_stream(comm_stream)` 让 y2 等 y1 的 all-reduce，破坏本日要演示的重叠 | 删除该 wait 并加注释解释（y2 不依赖 y1）；把这段从"自相矛盾的注释"改为正式教学点 |
| 3 | `week2` 全周 | GEMM 性能四套口径并存（D2 32% / D3 63% / D4 46% / D7 65%+） | 以 D3 实测为唯一锚点，其余天的数字标注"预估/旧口径"或改写 |
| 4 | `week2/day6` | Ridge Point 写 25（应为 58.45，与 reference/hardware_specs.md 一致） | 改 58.45，并全文 grep `25` 口径残留 |
| 5 | `week5` D2/D3/D7 共 4 处 | SRAM 164KB（RTX 5090 实为 100KB smem/L1 共享口径） | 统一为 100KB，IO 口径加一句说明 |
| 6 | `week6/day6` 共 4 处 | 80 SM（RTX 5090 应为 170 SM） | 改 170，并核对引用该数字的推导 |
| 7 | `week1/day7` + `week1` 正文 | SRAM 容量三处矛盾（100KB vs 192KB） | 以 100KB 统一 |
| 8 | `week8` D1 vs D2 | FP4 规格互相矛盾（"3 指数 ±12" vs "E2M1 ±4"） | 统一为 NVFP4 = E2M1 + microscaling |
| 9 | `week1/day5` | 广播/bank conflict 表述自相矛盾 | 改写该段，明确 broadcast 不占 conflict |
| 10 | `week1/day4` | 0.14% vs 0.66% 数字打架；padding 未教先用 | 统一数字；padding 加前向引用或提前讲 |
| 11 | `week10/day6` | top-p 验收"与 `torch.top_p` 对比"——该函数不存在 | 改为"与手写参考实现/vLLM SamplingParams 行为对比" |

### 1.2 代码级 bug（4 项）

| # | 位置 | 问题 | 修法 |
|---|------|------|------|
| 1 | `week10/day2/kernels/mini_engine_v2.py:27` | 从 `../../day4/kernels` 导入 `custom_ops_module`（实际在 `day1/kernels`），ImportError 被静默 catch → **真整合静默失效** | 改路径为 `day1/kernels`；catch 分支加 `warnings.warn`；运行验证 custom kernel 确实被调用 |
| 2 | `week10/day2` docstring | 引用"week9/day2 的 v1""week9/day1"，周号过时 | v1 实际在 week7/day5，改注释 |
| 3 | `week3/day5` | double buffer 代码是假双缓冲（同步实现未重叠） | 修复为真双缓冲或显式标注"同步占位实现，需 cp.async"（与 W10D4 实测表的口径一致） |
| 4 | `week9/day2` 任务 1 骨架 | `timeline[s][-1][1]` 取的是 micro-batch 编号而非结束时间（应为 `[-1][3]`）；`simulate_1f1b` 只有 `pass` | 补完并落盘（见 WS-4.3） |

### 1.3 文档破损（1 项）

- `week10/day5`：LeetCode 表后混入孤儿代码块（`longestValidParentheses` 栈解法 + 无头尾围栏），对应题目不在本日题表——整段删除；今日总结第 8 条"最长有效括号"残留碎片同删。

---

## 三、WS-2：迁移残留清理（P1）

> 验收标准：全文不存在"Week 8 的第一天""8 周能力地图"等旧口径；不存在指向不存在文件的链接；无重复粘贴标题。

### 2.1 plan/ 三代计划归档（半天）

- 旧 8 周极简/详细版、8 周 expanded（Day8–56 连续编号）移入 `plan/archive/` 或删除；保留 10 周 v2 为唯一主线。
- v2 计划中 18 个"待开发"占位天与实际教程状态对账，更新或删除。

### 2.2 周号/日号漂移全局修正（1 天）

| 位置 | 问题 |
|------|------|
| `week10` 全周（最严重） | D3"Week 8 的第一天""Week 1-7"、D1 总结自称"Day 4"、D2 自称"Day 5"、D4 自称"Day 3"且预告"明天 Day 4 进入进阶篇"（进阶篇已在 `_supplementary/from_w10d3/`）、D7 全篇 8 周口径 |
| `week6/day7` | 自称"Week 5" |
| `week7` 全周 | 学前导读/总结的自指日号 +1 错位 |
| `week8/day4` | "Day 6 的全链路 Profiling""明天 Day 7…Week 7 系统整合收官"引用错误 |
| `week9/day1` | LeetCode 标注"第 7 周补充"口径混乱 |

做法：先 grep 旧模式（`Week 8`、`Day [1-7] 我们`、`8 周`）列清单，再逐处人工确认改写（日号引用不能批量替换，容易误伤）。

### 2.3 Day7 复盘错位重写（3 处，各半天）

1. **W4D7**：整篇是旧 W3 复盘——按当前 W4 D1–6 内容重写。
2. **W7D7**：复盘的"Dynamic Batching、`benchmark_engine_v1.py`"在当前周不存在——按当前周实际内容重写，补 Prefix Caching/PD 分离，删幽灵引用。
3. **W10D7**（工作量最大）：
   - 全篇改 10 周口径（标题/正文/目标/完成标准）；
   - "Week 8 知识地图"7 天表按当前 week10 实际内容重填；
   - **能力地图补 Week 8/9**：量化、分布式 TP/PP/EP、MoE、Ascend 从"待提升"移到"已掌握概念/待深入实战"档，强项分母重算；
   - 目录结构 `aiinfra/week8/` 改为实际路径；`week8_summary.py` 改名 `week10_summary.py`（或去周号）；
   - 后续路线 Month 3"分布式与生产"与 Week 9 重叠，升级为"多卡实操（torchrun + NCCL 实测）"。

### 2.4 悬空引用与路径修复（1 天）

- 全局跑链接检查（见"六、防回归"）：不存在的 kernel 文件、错误 weekN/dayM 路径、`exercise/leetgpu_week1_review.md`（创建或删引用）、W8D6 引用的 `bench_eager.py`/`bench_graph.py`（落盘或删引用）。
- `week1/day7`"本周目录结构"按实际布局重写；`week10/day7` 目录结构同上。

### 2.5 标题重复粘贴清理（0.5 天）

- 已知：`week9` day1/5/6、`week10` day2/3/6、`week8` day1/4。
- 做法：grep 正则 `^## (.*)\1` 类模式（同一标题串出现两遍）全局扫一遍，逐处删重。

### 2.6 LeetGPU/LeetCode 编排清理（0.5 天）

- LeetGPU 重复题去重：Matrix Transpose 出现 4 次（W8D4、W10D1/D3/D4）、W7 D1/D2 与 D3/D4 成对重复——保留首次，其余换题或标"复用"。
- W3 LeetCode 改回第 3 周"链表与数学"主题；各周 LeetCode 周号标注统一口径。

---

## 四、WS-3：主项目贯穿线（P2）

> 目标：把"整合风险全压 W10"摊平；W10 结束时 Mini 引擎的每个声明都有真实运行证据。

### 3.1 W8D5 真整合（1 天）

- 把 W8D4 的 `BucketedGraphRunner` 套到 W7D5 的 `mini_engine_v1.py` decode 路径，产出 `mini_engine_v1_graph.py`；
- 实测 4 请求并发端到端 TBT 改善并留档（哪怕 1.2x 也是真数字）；
- 同步修复 W8D6：bench 脚本落盘、量化对比表标来源、D6 测量对象改为"D5 集成后的引擎"。

### 3.2 W10D2 联调跑真引擎（1 天）

- 依赖 WS-1.2.1 修复完成后：`stability_test.py` 的六步验证直接在 `mini_engine_v2` 上跑（替换纯模拟引擎），500+ 请求的成功率/P50/P99/KV 归零成为真实证据；
- W10D5 的"吞吐 X tokens/s、TTFT Y ms"占位数字用实测回填。

### 3.3 周地图与声明对齐（0.5 天）

- W10 周地图"真整合，替换 sleep 模拟"改为"D1 完成 kernel 封装、D2 接入引擎并联调"（与实际分工一致）；
- W10D1 补 CPU fallback 模式的预期输出。

---

## 五、WS-4：W9 结构重构 + 分布式真代码（P2）

> 目标：消除 D1→D2/3/4 的跨天重复；让"分布式纯概念"变成"写过一次"。

### 4.1 D1 减负、内容归位（1 天）

- `week9/day1`：§3b.3（PP）→ D2、§3b.5（NCCL）→ D3、§3b.6（通信重叠）→ D4；D1 只保留"为什么需要分布式 + TP + DP 定位"；删 D1 扩展实验 3（改为"见 Day 4"）。
- D2 与 D1 §3b.3/§3b.4 去重后聚焦 1F1B/bubble/推理 PP 部署形态（补 vLLM `pipeline_parallel_size` 一段）。
- D3 补增量内容：NVLink/PCIe/IB 带宽常识（NVLink4 ≈ 900GB/s、CX-7 IB 400Gb/s）、α-β 通信模型（latency + size/BW），让"通信量→通信时间"可估算；有 GPU 时加 `nccl-tests` 实测。
- D4 补 sequence parallelism（Megatron 2205.05198，all-reduce 拆 reduce-scatter/all-gather 以便重叠）——工业界主流做法，目前全周缺失。

### 4.2 分布式最小真代码（0.5 天，ROI 极高）

- 新增 30 行 `torch.distributed` 单机双进程 all-reduce demo（gloo 可跑、有 GPU 用 nccl），`torchrun --nproc_per_node=2` 启动；
- 放在 D3（NCCL 日），与 ring 通信量推导互证。

### 4.3 模拟器补完落盘（0.5 天）

- `week9/day2/kernels/pipeline_schedule_sim.py`：修 tuple 索引 bug（WS-1.2.4）、补完 `simulate_1f1b`、加 interleaved V=2/4 扫描；
- `week9/day3/kernels/ring_allreduce_sim.py`：补完 `...` 骨架并落盘。

### 4.4 D5/D6/D7 收尾（0.5 天）

- D5：补 Ring Attention 导读 + 链接 `_supplementary/from_w8d6/`；TP 通信量写法统一为 `2 × tokens × hidden`；
- D6：README 与脚本表格去重（README 留 5 对核心映射 + 链接）；删 LeetCode 牵强类比列；可选补只读 Ascend C GEMM 骨架片段；
- D7：Q&A 与前六天去重（单点题改链接）；Q1 补 KV Cache 显存；加 2–3 个场景化动手任务（给模型/GPU/延迟约束，写并行方案 + 通信量估算）。

---

## 六、WS-5：空缺补齐与增强（P3，按 ROI 选做）

| # | 事项 | 价值 | 工作量 |
|---|------|------|--------|
| 1 | **W8D2 重做为"真 FP8 GEMM 日"**：`torch._scaled_mm` 或 `__nv_fp8_e4m3` 实测 vs FP16，呼应 D1 的 FAIL 声明；同时删 D2 与 D1 重复的 70% 内容 | 高（填补"FP8 只在嘴上说"） | 1 天 |
| 2 | **加一天"真实模型 vLLM 部署 + 压测"**：跑一个 Qwen/Llama，出 TTFT/TPOT/吞吐报告，让 W6/W7 的指标概念落地 | 高（就业评估短板） | 1 天 |
| 3 | W3 D2/3/5 关键 kernel 落盘 + 实测（连带统一性能口径："D1 30%/D2 42% 实测 TF32"为锚） | 中高 | 1 天 |
| 4 | W4D3 `layernorm_welford.cu`、W4D5 `benchmark_triton.py` 落盘；W4 D5/D6 向 D4 收敛去重 | 中 | 0.5 天 |
| 5 | 采样 kernel 衔接：W8D7 地图/Q&A 出现"采样"但本周未教——W10D6 已补 top-p，在 W8D5 或 W8D7 加前向链接 | 中 | 0.5 天 |
| 6 | Day7 ASCII 知识地图 → SVG（W8D7 等连续违规处） | 中 | 0.5 天 |
| 7 | Day0 工程基建前导（CMake/编译/nsys 安装验证/git 工作流），放 W1 前 | 中（通用 Infra 岗位） | 0.5 天 |
| 8 | LeetGPU 10+ 天"待定"题解补齐；W5 benchmark 表 nan 回填 | 低中 | 1 天 |
| 9 | W10D4 面试题库补 Week 8/9 题目 3–4 道（量化/CUDA Graph/TP/EP 通信量） | 中 | 0.5 天 |
| 10 | W10D6 三份 ncu/py-spy"模拟留档"证据：真实跑一遍替换，或显著标注"示意输出，未实测" | 中（与 D3 数字诚信标准一致） | 0.5 天 |

---

## 七、分阶段执行计划

### 阶段 0：整改基线（0.5 天）

1. 把本方案工作项转成可勾选清单（或 issue）；
2. 建立三个可重复跑的检查脚本，接入 `build/lint_md_code.py` 同级：
   - **标题重复检测**：`^## Day N：(.+)\1` 类模式；
   - **旧口径扫描**：grep `Week 8|8 周|week8_summary|aiinfra/week8` 等清单；
   - **悬空链接检测**：提取 md 中相对路径链接，验证文件存在性。

### 阶段 1：WS-1 正确性红线（1–2 天）

- 按 §二 表格逐项修；每修一项：改内容 → 跑受影响脚本 → 在清单勾选。
- **必须先做**：WS-1.2.1（mini_engine_v2 导入）是 WS-3.2 的前置。

### 阶段 2：WS-2 迁移清理（2–3 天）

- 顺序：2.1 归档 → 2.2 周号日号 → 2.3 三个 Day7 重写 → 2.4 悬空引用 → 2.5 标题 → 2.6 编排；
- 每完成一周，跑阶段 0 的三个脚本验证。

### 阶段 3：WS-3 + WS-4（4–6 天，可并行）

- WS-3 需要 GPU 环境实测，WS-4 以文档重构为主，两者无依赖可并行；
- W9 重构完成后需同步更新 `week9/README.md` 学习地图。

### 阶段 4：WS-5 增强（按 ROI 选做）

- 推荐最小集：#1（FP8）、#2（vLLM 压测）、#9（题库补 W8/9）、#10（证据留档）。

---

## 八、验收标准（整体）

1. **正确性**：WS-1 全部 16 项关闭；`mini_engine_v2.py`、`stability_test.py`、`pipeline_schedule_sim.py`、`ring_allreduce_sim.py` 实际运行通过。
2. **一致性**：阶段 0 三个检查脚本全绿；性能数字每周只有一套实测口径（其余标"预估/示意"）。
3. **主项目**：W10D5 的项目话术里每个数字都能指向一份留档输出（benchmark/stability/profile）。
4. **W9**：D1 只讲 TP+动机；D2/3/4 无重复内容且各有增量；分布式最小 demo 可运行。
5. **复盘对齐**：W4/W7/W10 三个 Day7 的内容与所在周实际教学内容逐条对上。

## 九、防回归机制

- 三个检查脚本挂到 `build.py` 的构建流程（或 CI），每次构建自动跑；
- 数字口径唯一事实源：硬件参数只引用 `reference/hardware_specs.md`，性能实测只引用留档文件，禁止手抄；
- 新增/迁移内容时执行"三查"：查周号日号自称、查交叉引用路径、查标题重复——写入 `aiinfra/daily/SKILL.md` 或 AGENTS.md 作为内容维护规约。
