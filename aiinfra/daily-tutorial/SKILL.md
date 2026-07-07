---
name: daily-tutorial
description: Use when writing or revising a per-day learning tutorial (dayN/README.md) in this AI Infra learning repo. Triggers on requests like "write week3 day1", "complete day N", "add a new day tutorial", "补全 dayN 教程", "写每日教程". Produces tutorials following the repo's fixed 8-section skeleton, Chinese-first style, with compilable CUDA code, Nsight profiling commands, and interview Q&A. Do NOT use for editing opencode's own config, agents, or non-tutorial markdown.
---

# 写每日教程 Skill

本工程(`ai-infra-notes`)的每日教程遵循一套固定的结构和写作规范。本 skill 描述如何产出符合仓库惯例的 `weekN/dayM/README.md`。

## 1. 计划先行(Plan → Execute 映射)

每日教程不是凭空写的,而是对 `docs/` 下计划文件的"执行展开":

| 层级 | 文件 | 作用 |
|------|------|------|
| 周级总览 | `docs/AI_Infra_8_week_plan.md` | 8 周极简路线 |
| 周级详细 | `docs/AI_Infra_8_week_plan_detailed.md` | 每周目标 + 每日"理论学习/Coding/Checklist"三段种子 |
| 单周深度展开 | `docs/learning_plan_weekN_expanded.md` | 逐日详写(含时间分配、考察度 ⭐、附录) |
| **执行教程** | `weekN/dayM/README.md` | 本 skill 产出的主体 |

**前置阅读要求**:
在动笔写每日教程之前,必须先完整阅读 `docs/` 目录下的全部文档:
- `docs/AI_Infra_8_week_plan.md`
- `docs/AI_Infra_8_week_plan_detailed.md`
- `docs/learning_plan_week2_expanded.md`
- `docs/learning_plan_week3_expanded.md`

这些文档提供了 8 周整体路线、每周目标与逐日种子内容,是教程写作的上下文基础。

**写作流程**:
1. 先确认 `docs/learning_plan_weekN_expanded.md` 是否存在对应 Day 的计划;若无,先写计划
2. 将计划中的"理论学习/Coding 任务/Checklist"三段,展开为完整的 8 段教程
3. 计划文件的 Checklist 条目转化为教程中的验证问题或练习要求

## 2. 文件落位

```
weekN/dayM/
├── README.md           # 教程主体(本 skill 产出)
├── kernels/*.cu        # 完整可编译代码(教程中引用的真实文件)
├── exercise/           # 练习题与验证程序
└── notes/              # 笔记/延伸阅读
weekN/website/images/*.svg  # SVG 图(语义化小写命名,如 warp_shuffle_primitives.svg)
```

- 教程中用相对路径引用本地文件:`[kernels/hello_gpu.cu](kernels/hello_gpu.cu)`
- SVG 引用:`![中文alt描述](../website/images/xxx.svg)`(从 dayM/ 出发,`../website/images/` 解析到 weekN/website/images/)

## 3. 教学日 8 段骨架(固定顺序)

所有"教学型 Day"(非总结日)严格遵循以下顺序:

```
## Day N：<主题>
### 🎯 目标
### 学前导读：<为什么需要 X>
### 理论学习
### Coding 任务：<具体任务名>
### 扩展实验
### 今日总结
### 面试要点
```

### 3.1 `## Day N：<主题>`
- 标题用"主题词 + 限定词",如 `Day 2：Occupancy 与资源约束`、`Day 5：FlashAttention CUDA 实现(简化版)`
- 冒号用中文全角 `：`

### 3.2 `### 🎯 目标`(必有,紧贴标题)
```markdown
### 🎯 目标

通过今天的学习，你将：

1. <动词开头的目标,如"理解..."><br>
2. ...
3. ...
4. ...
5. ...
6. ...

> 💡 **为什么重要**：<一句话点题,衔接前一日,说明本日内容在学习路径中的定位>

---
```
- 固定引导句 `通过今天的学习，你将：`
- **6 条**编号目标(动词开头:理解/掌握/学会/能/实现)
- 末尾固定 `> 💡 **为什么重要**：<...>` blockquote
- 用 `---` 与下一章隔开

### 3.3 `### 学前导读：<为什么需要 X>`(教学日必有)
- 1-2 段 + 对比表/代码块,回答"为什么要学这个"
- 衔接前一日知识,制造认知冲突或动机(如"Shared Memory 还不够快")
- 开篇日(Day1)可加长(背景铺垫,~80 行);其余通常 15-35 行
- 常以 `> 💡 **一句话总结**：<...>` 收束
- 命名格式:`### 学前导读：<动机点题>`

### 3.4 `### 理论学习`(教学日必有)
- 用 `#### N.1`、`#### N.2` 分小节
- 配 SVG 图,插在小节首行:`![<中文alt>](../website/images/<filename>.svg)`
- 表格化对比(延迟、带宽、容量等量化数据)
- 善用 `##### 五级标题` 做深入解释(如"为什么 X?")
- 形象类比(把 SM 比作教室、warp 比作班组等)

### 3.5 `### Coding 任务:<任务名>`(教学日必有)
- **4 个** `#### 任务 N:<动作描述>` 子任务,呈递进结构:
  - 任务 1:创建 `.cu` 文件(给完整参考代码)
  - 任务 2:编译与运行(给 nvcc 命令 + 预期输出)
  - 任务 3:验证 / Profiling / 检查指标
  - 任务 4:**LeetGPU 在线题目**(见下方说明)
- **参考代码要求**:
  - 包含**完整可编译代码**:`#include`、`__global__` kernel、`main()`、host 端 `cudaMalloc/Memcpy`、验证逻辑、`cudaFree`
  - 代码块标注 ` ```cuda `
  - 代码块首行带注释:`// xxx.cu —— <说明>` + `// 编译命令: nvcc ...`
- **编译命令**:单独用 ` ```bash ` 代码块
- **预期输出**:用 ` ```text ` 代码块
- **文件链接**:用相对路径 `[kernels/xxx.cu](kernels/xxx.cu)` 引用真实文件
- 可插入 `#### 为什么 <反直觉问题>?` 解释型小节

#### LeetGPU 在线题目(任务 4,必有)

每天 Coding 任务必须包含一道来自 **https://leetgpu.com/** 的经典 CUDA 在线题目,作为当日所学知识的实战检验。

**选题原则**:
- 题目应与当日主题强相关(如 Day 1 学 Warp Shuffle → 选 reduce/scan 类题;Day 2 学 GEMM → 选矩阵乘法题;Day 5 学 FlashAttention → 选 attention/softmax 题)
- 优先选中等难度;若当日主题偏基础可选简单,偏进阶可选困难
- 避免重复(记录已用题目,后续 Day 不重选)

**呈现格式**(参考 `leetgpu/leetgpu-prefix-sum-solution.md`):
```markdown
#### 任务 4：LeetGPU 在线题目 —— <题目名>

**题目链接**：<https://leetgpu.com/challenges/<slug>>

**题目概述**：
<1-2 句话描述题意，说明输入输出与约束>

**与今日知识的关联**：
<说明本题考察的当日知识点，如"本题核心是 block 内两级归约，直接用 Day 1 的 warpReduceSum + blockReduceSum 结构">

**解题思路**：
<3-5 句话点明并行化策略、存储层次使用、关键技巧>

**参考实现**：
\`\`\`cuda
<完整可提交的 kernel 实现，含编译提交说明>
\`\`\`

> 💡 提交后把通过截图/耗时记录到 `exercise/leetgpu_<题目slug>.md`，与官方排行榜对比性能。
```

**题目归档**:每道题的完整题解单独存为 `leetgpu/leetgpu-<题目slug>-solution.md`,并在教程中引用链接。题解文件结构参照已有 `leetgpu/leetgpu-prefix-sum-solution.md`:
1. 题目概述(标题/链接/难度/标签)
2. CPU 基线 / 朴素 GPU 方法
3. GPU 设计(并行化策略、存储层次)
4. Kernel 实现
5. 性能分析与优化
6. 复杂度分析

### 3.6 `### 扩展实验`(教学日必有)
- **3 个** `#### 实验 N:<描述>`,递进或对比
- 每个实验给出修改建议 + 思考问题

### 3.7 `### 今日总结`(必有)
```markdown
### 今日总结

Day N 我们<掌握了/深入理解了/完成了> <主题>：

1. **<概念>**：<一句话概括>
2. **<概念>**：<一句话概括>
...
```
- 固定开头 `Day N 我们<动词>...：`
- **5-7 条**加粗编号列表
- 教学日常有一句展望(如"掌握这些后,你就...")

### 3.8 `### 面试要点`(必有)
- **5 题**问答(个别 3-4 题)
- 格式:问题加粗,答案缩进展开
  ```markdown
  1. **<面试官可能问的问题>?**

     - <答案要点 1>
     - <答案要点 2>
  ```
- 答案可含代码块或子编号
- 验收日(如 week2/day7)的面试题可含"评分关键"

## 4. 总结日(Day7)变体

总结日不套用 8 段骨架,改用:

```
## Day 7：<总结/验收主题>
### 🎯 目标
### Week N 知识地图              ← 替代学前导读
### 核心概念串讲
### <决策树/方法论>
### 总结任务 / Coding 任务        ← 验收型有,纯总结无
### 面试准备框架
### 常见误区澄清                  ← 替代常见错误
### Week N → Week N+1 衔接
### 弹性安排
### 今日总结
### 面试要点
## 📁 本周目录结构
## 🔗 推荐资源
## ✅ Week N 完成标准
```

**验收型 Day7**(如 week2/day7)额外含:
- 限时手撕任务 + **评分标准表**(`| 项目 | 分值 | 评分要点 |`)
- 性能对比报告模板
- GitHub 仓库整理 Checklist

## 5. 写作规范

### 语言
- **中文为主**,概念加粗
- 善用 blockquote:`> 💡 **一句话总结**：<...>`、`> ⚠️ **注意**：<...>`
- 代码块标注语言:` ```cuda` / ` ```bash` / ` ```text` / ` ```cpp`

### 量化指标
- 教学日平均约 **550 行 / 14 个代码块**
- 开篇日(Day1)和含完整 kernel 实现的日代码块最多(~26)
- 总结日代码块最少(~6-8)

### 图片
- SVG 命名:全小写 + 下划线,语义化(如 `register_blocking_dataflow.svg`)
- 每个 Day 平均引用 2-4 张 SVG
- alt 文本用中文
- **风格统一为手绘 sketch 风**(Excalidraw-like),具体要求:
  - **线条**:手绘不均匀、略带抖动的线条,避免完美直线或圆润矢量边
  - **笔触**:粗糙、类似马克笔/铅笔的描边,线宽可略有变化
  - **配色**:极简,一般不超过 3-4 种柔和颜色(如蓝、橙、绿、红 accent),背景为白色或米白色
  - **形状**:简单几何块——矩形、网格、箭头、圆角框,不画复杂 3D 或写实元素
  - **标签**:手写体/草书字体,英文用 Bradley Hand / Comic Sans MS 等,CJK 用楷体(Kaiti SC)等相匹配的手写感字体
  - **整体感觉**:轻松白板涂鸦,标注随意、有轻微错位也无妨,优先可读性和直观性

### 交叉引用
- 引用本 Day 文件:相对路径 `(kernels/xxx.cu)`
- 引用周级文件:`(../notes/week1_notes.md)`、`(../tools/cuda_occupancy_calculator.py)`
- 引用其他 Day:少见,必要时用 `../dayM/`

## 6. 构建集成

写完 dayN/README.md 后,教程会被 `weekN/website/build.py` 自动读取并生成 `dayN.html`:
- `build.py` 遍历 `weekN/day*/README.md`,解析首行 `## Day N：<title>`
- 图片路径 `../website/images/` 会被重写为 `images/`(网站输出目录)
- `.md` 链接会被重写为 `.html`(GitHub Pages 部署)

**验证命令**:
```bash
python3 weekN/website/build.py    # 单周构建
python3 build.py                   # 组合构建(含 week1/week2/leetcode)
```

## 7. 检查清单(写完一个 Day 后自检)

- [ ] 首行是 `## Day N：<主题>`(中文全角冒号)
- [ ] `### 🎯 目标` 紧跟标题,含 6 条编号 + `> 💡 为什么重要`
- [ ] 有 `### 学前导读`(总结日除外)
- [ ] `### 理论学习` 用 `#### N.x` 分节,配 SVG
- [ ] `### Coding 任务` 含 4 个任务(含 1 道 LeetGPU 在线题目),代码完整可编译,带 nvcc 命令 + 预期输出
- [ ] LeetGPU 题目与当日主题强相关,题解归档到 `leetgpu/leetgpu-<slug>-solution.md`
- [ ] `### 扩展实验` 3 个
- [ ] `### 今日总结` 5-7 条加粗编号
- [ ] `### 面试要点` 5 题问答
- [ ] 所有文件链接用相对路径且指向真实文件
- [ ] SVG 引用格式 `![中文alt](../website/images/xxx.svg)`
- [ ] 运行 `python3 weekN/website/build.py` 成功生成 `dayN.html`
