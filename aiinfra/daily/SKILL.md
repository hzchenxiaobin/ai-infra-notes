---
name: daily-tutorial
description: Use when writing or revising a per-day learning tutorial (dayN/README.md) in this AI Infra learning repo. Triggers on requests like "write week3 day1", "complete day N", "add a new day tutorial", "补全 dayN 教程", "写每日教程". Produces tutorials following the repo's fixed 8-section skeleton, Chinese-first style, with compilable CUDA code, Nsight profiling commands, and interview Q&A. Do NOT use for editing opencode's own config, agents, or non-tutorial markdown.
---

# 写每日教程 Skill

本工程(`ai-infra-notes`)的每日教程遵循一套固定的结构和写作规范。本 skill 描述如何产出符合仓库惯例的 `weekN/dayM/README.md`。

## 1. 计划先行(Plan → Execute 映射)

每日教程不是凭空写的,而是对 `aiinfra/daily/plan/` 下计划文件的"执行展开":

| 层级 | 文件 | 作用 |
|------|------|------|
| 周级总览 | `plan/AI_Infra_8_week_plan.md` | 8 周极简路线 |
| 周级详细 | `plan/AI_Infra_8_week_plan_detailed.md` | 每周目标 + 每日"理论学习/Coding/Checklist"三段种子 |
| 单周深度展开 | `plan/learning_plan_weekN_expanded.md` | 逐日详写(含时间分配、考察度 ⭐、附录) |
| **执行教程** | `weekN/dayM/README.md` | 本 skill 产出的主体 |

**前置阅读要求**:
在动笔写每日教程之前,必须先完整阅读 `aiinfra/daily/plan/` 目录下的全部文档:
- `plan/AI_Infra_8_week_plan.md`
- `plan/AI_Infra_8_week_plan_detailed.md`
- `plan/learning_plan_week2_expanded.md`
- `plan/learning_plan_week3_expanded.md`

这些文档提供了 8 周整体路线、每周目标与逐日种子内容,是教程写作的上下文基础。

**写作流程**:
1. 先确认 `plan/learning_plan_weekN_expanded.md` 是否存在对应 Day 的计划;若无,先写计划
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
- **5 个** `#### 任务 N:<动作描述>` 子任务,呈递进结构:
  - 任务 1:创建 `.cu` 文件(给完整参考代码)
  - 任务 2:编译与运行(给 nvcc 命令 + 预期输出)
  - 任务 3:验证 / Profiling / 检查指标
  - 任务 4:**LeetGPU 在线题目**(见下方说明)
  - 任务 5:**LeetCode 面试题**(见下方说明)
- **参考代码要求**:
  - 包含**完整可编译代码**:`#include`、`__global__` kernel、`main()`、host 端 `cudaMalloc/Memcpy`、验证逻辑、`cudaFree`
  - 代码块标注 ` ```cuda `
  - 代码块首行带注释:`// xxx.cu —— <说明>` + `// 编译命令: nvcc ...`
- **编译命令**:单独用 ` ```bash ` 代码块
- **预期输出**:用 ` ```text ` 代码块
- **文件链接**:用相对路径 `[kernels/xxx.cu](kernels/xxx.cu)` 引用真实文件
- 可插入 `#### 为什么 <反直觉问题>?` 解释型小节

#### LeetGPU 在线题目(任务 4,必有)

每天 Coding 任务必须包含一道来自 **https://leetgpu.com/** 的经典 CUDA 在线题目,与当日主题强相关、避免重复。完整题解的写作规范(6 段结构、Kernel 代码、SVG 插图、ncu profiling)见 [`leetgpu/SKILL.md`](../../leetgpu/SKILL.md);教程"任务 4"只需给出题目链接 + 1-2 句与当日知识的关联,并用相对链接指向下表对应题解。

**已归档题解**(位于 `leetgpu/weekN/dayM/`):

| 教程 | 主题 | LeetGPU 题目 | 题解链接 |
|------|------|--------------|----------|
| Week1 Day1 | GPU 执行模型基础 | Vector Addition | [leetgpu-vector-addition-solution.md](../../leetgpu/week1/day1/leetgpu-vector-addition-solution.md) |
| Week1 Day2 | Occupancy 与资源约束 | ReLU | [leetgpu-relu-solution.md](../../leetgpu/week1/day2/leetgpu-relu-solution.md) |
| Week1 Day3 | 认识你的 GPU —— deviceQuery 与 Occupancy 计算 | Matrix Addition | [leetgpu-matrix-addition-solution.md](../../leetgpu/week1/day3/leetgpu-matrix-addition-solution.md) |
| Week1 Day4 | Memory Hierarchy 深入 | Matrix Transpose | [leetgpu-matrix-transpose-solution.md](../../leetgpu/week1/day4/leetgpu-matrix-transpose-solution.md) |
| Week1 Day5 | Bank Conflict 分析与实践 | Reduction | [leetgpu-reduction-solution.md](../../leetgpu/week1/day5/leetgpu-reduction-solution.md) |
| Week1 Day6 | Nsight Profiling 实战 | Matrix Multiplication | [leetgpu-matrix-multiplication-solution.md](../../leetgpu/week1/day6/leetgpu-matrix-multiplication-solution.md) |
| Week1 Day7 | 总结与复盘 | Matrix Addition | [leetgpu-matrix-addition-solution.md](../../leetgpu/week1/day7/leetgpu-matrix-addition-solution.md) |
| Week2 Day1 | Warp Shuffle 原语与 Warp/Block Reduce | Prefix Sum | [leetgpu-prefix-sum-solution.md](../../leetgpu/week2/day1/leetgpu-prefix-sum-solution.md) |
| Week2 Day2 | Register Blocking 与 2D Tiling | GEMM | [leetgpu-gemm-solution.md](../../leetgpu/week2/day2/leetgpu-gemm-solution.md) |
| Week2 Day3 | CUDA Streams 与异步执行 | 2D Convolution | [leetgpu-2d-convolution-solution.md](../../leetgpu/week2/day3/leetgpu-2d-convolution-solution.md) |
| Week2 Day4 | Nsight Compute 性能分析 | Softmax | [leetgpu-softmax-solution.md](../../leetgpu/week2/day4/leetgpu-softmax-solution.md) |
| Week2 Day5 | FlashAttention CUDA 实现（简化版） | Softmax Attention | [leetgpu-softmax-attention-solution.md](../../leetgpu/week2/day5/leetgpu-softmax-attention-solution.md) |
| Week2 Day6 | 整合优化到 cuBLAS 70%+ | Histogramming | [leetgpu-histogramming-solution.md](../../leetgpu/week2/day6/leetgpu-histogramming-solution.md) |
| Week2 Day7 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | Max Subarray Sum | [leetgpu-max-subarray-sum-solution.md](../../leetgpu/week2/day7/leetgpu-max-subarray-sum-solution.md) |
| Week3 Day1 | Trace Transformer 推理流程 | 1D Convolution | [leetgpu-1d-convolution-solution.md](../../leetgpu/week3/day1/leetgpu-1d-convolution-solution.md) |
| Week3 Day2 | 手写 Softmax 与 LayerNorm Kernel | Softmax | [leetgpu-softmax-solution.md](../../leetgpu/week3/day2/leetgpu-softmax-solution.md) |
| Week3 Day3 | 源码分析 —— PyTorch / FasterTransformer | Argmax | [leetgpu-argmax-solution.md](../../leetgpu/week3/day3/leetgpu-argmax-solution.md) |
| Week3 Day4 | Attention IO 分析 | Attention | [leetgpu-attention-solution.md](../../leetgpu/week3/day4/leetgpu-attention-solution.md) |
| Week3 Day5 | 算子接入 Mini 引擎 | Matrix Addition | [leetgpu-matrix-addition-solution.md](../../leetgpu/week3/day5/leetgpu-matrix-addition-solution.md) |
| Week3 Day6 | 端到端 Profiling 与 Kernel Fusion | RMS Normalization | [leetgpu-rms-normalization-solution.md](../../leetgpu/week3/day6/leetgpu-rms-normalization-solution.md) |
| Week3 Day7 | Transformer 算子分类与 Week 3 总结 | Causal Self-Attention | [leetgpu-causal-self-attention-solution.md](../../leetgpu/week3/day7/leetgpu-causal-self-attention-solution.md) |
| Week4 Day1 | FlashAttention 论文精读与 Online Softmax 完整推导 | Softmax Attention | [leetgpu-softmax-attention-solution.md](../../leetgpu/week4/day1/leetgpu-softmax-attention-solution.md) |
| Week4 Day2 | 手写完整 FlashAttention Forward Kernel | Attention | [leetgpu-attention-solution.md](../../leetgpu/week4/day2/leetgpu-attention-solution.md) |
| Week4 Day3 | FlashAttention 官方 CUDA 源码分析 | Dot Product | [leetgpu-dot-product-solution.md](../../leetgpu/week4/day3/leetgpu-dot-product-solution.md) |
| Week4 Day4 | FlashAttention-2 论文与源码差异 | Batched Matrix Multiplication | [leetgpu-batched-matrix-multiplication-solution.md](../../leetgpu/week4/day4/leetgpu-batched-matrix-multiplication-solution.md) |
| Week4 Day5 | 算子接入 Mini 引擎 —— FlashAttention 集成 | Matrix Copy | [leetgpu-matrix-copy-solution.md](../../leetgpu/week4/day5/leetgpu-matrix-copy-solution.md) |
| Week4 Day6 | 性能对比分析 —— 标准 vs 手写 vs 官方 | Multi-Head Attention | [leetgpu-multi-head-attention-solution.md](../../leetgpu/week4/day6/leetgpu-multi-head-attention-solution.md) |
| Week4 Day7 | IO 优化方法论总结与 Week 4 收官 | GPT-2 Transformer Block | [leetgpu-gpt-2-transformer-block-solution.md](../../leetgpu/week4/day7/leetgpu-gpt-2-transformer-block-solution.md) |
| Week5 Day1 | 推理流程 —— Prefill vs Decode | INT8 KV-Cache Attention | [leetgpu-int8-kv-cache-attention-solution.md](../../leetgpu/week5/day1/leetgpu-int8-kv-cache-attention-solution.md) |
| Week5 Day2 | 实现 KV Cache | Grouped Query Attention (GQA) | [leetgpu-grouped-query-attention-solution.md](../../leetgpu/week5/day2/leetgpu-grouped-query-attention-solution.md) |
| Week5 Day3 | vLLM 整体架构分析 | Speculative Decoding Verification | [leetgpu-speculative-decoding-verification-solution.md](../../leetgpu/week5/day3/leetgpu-speculative-decoding-verification-solution.md) |
| Week5 Day4 | vLLM Worker 与 PagedAttention | Causal Self-Attention | [leetgpu-causal-self-attention-solution.md](../../leetgpu/week5/day4/leetgpu-causal-self-attention-solution.md) |
| Week5 Day5 | 项目推进 —— Mini 推理引擎 v0 | Token Embedding Layer | [leetgpu-token-embedding-layer-solution.md](../../leetgpu/week5/day5/leetgpu-token-embedding-layer-solution.md) |
| Week5 Day6 | 端到端 Profiling | Weight Dequantization | [leetgpu-weight-dequantization-solution.md](../../leetgpu/week5/day6/leetgpu-weight-dequantization-solution.md) |
| Week5 Day7 | 推理系统核心问题总结与 Week 5 收官 | Simple Inference | [leetgpu-simple-inference-solution.md](../../leetgpu/week5/day7/leetgpu-simple-inference-solution.md) |
| Week6 Day1 | Dynamic Batching | Simple Inference | [leetgpu-simple-inference-solution.md](../../leetgpu/week6/day1/leetgpu-simple-inference-solution.md) |
| Week6 Day2 | Continuous Batching | Max Subarray Sum | [leetgpu-max-subarray-sum-solution.md](../../leetgpu/week6/day2/leetgpu-max-subarray-sum-solution.md) |
| Week6 Day3 | vLLM Scheduler 源码分析 | Stream Compaction | [leetgpu-stream-compaction-solution.md](../../leetgpu/week6/day3/leetgpu-stream-compaction-solution.md) |
| Week6 Day4 | TensorRT-LLM / LightLLM 调度对比 | Segmented Prefix Sum | [leetgpu-segmented-prefix-sum-solution.md](../../leetgpu/week6/day4/leetgpu-segmented-prefix-sum-solution.md) |
| Week6 Day5 | Mini 推理引擎 v1 | Batched Matrix Multiplication | [leetgpu-batched-matrix-multiplication-solution.md](../../leetgpu/week6/day5/leetgpu-batched-matrix-multiplication-solution.md) |
| Week6 Day6 | Latency / Throughput 测试 | Top K Selection | [leetgpu-top-k-selection-solution.md](../../leetgpu/week6/day6/leetgpu-top-k-selection-solution.md) |
| Week6 Day7 | 调度优化策略总结与 Week 6 收官 | Dot Product | [leetgpu-dot-product-solution.md](../../leetgpu/week6/day7/leetgpu-dot-product-solution.md) |
| Week7 Day1 | 多请求并发支持 | Matrix Copy | [leetgpu-matrix-copy-solution.md](../../leetgpu/week7/day1/leetgpu-matrix-copy-solution.md) |
| Week7 Day2 | 完整调度器 | Vector Reversal | [leetgpu-vector-reversal-solution.md](../../leetgpu/week7/day2/leetgpu-vector-reversal-solution.md) |
| Week7 Day3 | SGLang / LightLLM 高级特性 | Scalar Multiply | [leetgpu-scalar-multiply-solution.md](../../leetgpu/week7/day3/leetgpu-scalar-multiply-solution.md) |
| Week7 Day4 | 整合全部自定义 Kernel | Matrix Transpose | [leetgpu-matrix-transpose-solution.md](../../leetgpu/week7/day4/leetgpu-matrix-transpose-solution.md) |
| Week7 Day5 | 系统联调 | Element Reversal | [leetgpu-element-reversal-solution.md](../../leetgpu/week7/day5/leetgpu-element-reversal-solution.md) |
| Week7 Day6 | 全链路 Profiling | Reduction | [leetgpu-reduction-solution.md](../../leetgpu/week7/day6/leetgpu-reduction-solution.md) |
| Week7 Day7 | 代码重构与文档 | Matrix Addition | [leetgpu-matrix-addition-solution.md](../../leetgpu/week7/day7/leetgpu-matrix-addition-solution.md) |
| Week8 Day1 | 项目文档完善 | SiLU | [leetgpu-silu-solution.md](../../leetgpu/week8/day1/leetgpu-silu-solution.md) |
| Week8 Day2 | 架构图与数据流图 | Rotary Positional Embedding | [leetgpu-rope-embedding-solution.md](../../leetgpu/week8/day2/leetgpu-rope-embedding-solution.md) |
| Week8 Day3 | 高频面试题基础篇 | SwiGLU | [leetgpu-swiglu-solution.md](../../leetgpu/week8/day3/leetgpu-swiglu-solution.md) |
| Week8 Day4 | 高频面试题进阶篇 | Sliding Window Self-Attention | [leetgpu-sliding-window-self-attention-solution.md](../../leetgpu/week8/day4/leetgpu-sliding-window-self-attention-solution.md) |
| Week8 Day5 | Mock 面试 | LoRA Linear | [leetgpu-lora-linear-solution.md](../../leetgpu/week8/day5/leetgpu-lora-linear-solution.md) |
| Week8 Day6 | 查漏补缺 | Batch Normalization | [leetgpu-batch-normalization-solution.md](../../leetgpu/week8/day6/leetgpu-batch-normalization-solution.md) |
| Week8 Day7 | 最终复盘 | 1D Convolution | [leetgpu-1d-convolution-solution.md](../../leetgpu/week8/day7/leetgpu-1d-convolution-solution.md) |

> 💡 新增 Day 若上表无对应题解,按 [`leetgpu/SKILL.md`](../../leetgpu/SKILL.md) 在 `leetgpu/weekN/dayM/` 下新建 `leetgpu-<slug>-solution.md` 并补入上表。

#### LeetCode 面试题(任务 5,必有)

每天 Coding 任务额外包含一道来自 **https://leetcode.cn/** 的面试高频题,作为算法基本功的日常练习。完整题解的写作规范(6 段结构、C++/Python 参考代码、手绘 SVG、复杂度分析)见 [`leetcode/daily/SKILL.md`](../../leetcode/daily/SKILL.md);教程"任务 5"只需给出题目链接 + 1-2 句核心套路点题,并用相对链接指向下表对应题解。

**已归档题解**(位于 `leetcode/daily/weekN/dayM/`):

| 教程 | 主题 | LeetCode 题目 | 题解链接 |
|------|------|---------------|----------|
| Week1 Day1 | 数组 / 双指针 | 42. 接雨水 | [接雨水.md](../../leetcode/daily/week1/day1/接雨水.md) |
| Week1 Day2 | 动态规划 | 53. 最大子数组和 | [最大子数组和.md](../../leetcode/daily/week1/day2/最大子数组和.md) |
| Week1 Day3 | 字符串 / 滑窗 | 3. 无重复字符的最长子串 | [无重复字符的最长子串.md](../../leetcode/daily/week1/day3/无重复字符的最长子串.md) |
| Week1 Day4 | 链表 | 206. 反转链表 | [反转链表.md](../../leetcode/daily/week1/day4/反转链表.md) |
| Week1 Day5 | 树 / DFS | 236. 二叉树的最近公共祖先 | [二叉树的最近公共祖先.md](../../leetcode/daily/week1/day5/二叉树的最近公共祖先.md) |
| Week1 Day6 | 回溯 | 46. 全排列 | [全排列.md](../../leetcode/daily/week1/day6/全排列.md) |
| Week1 Day7 | 栈 / 困难 | 84. 柱状图中最大的矩形 | [柱状图中最大的矩形.md](../../leetcode/daily/week1/day7/柱状图中最大的矩形.md) |
| Week2 Day1 | 哈希表 | 1. 两数之和 | [两数之和.md](../../leetcode/daily/week2/day1/两数之和.md) |
| Week2 Day2 | 动态规划 | 70. 爬楼梯 | [爬楼梯.md](../../leetcode/daily/week2/day2/爬楼梯.md) |
| Week2 Day3 | 双指针 | 15. 三数之和 | [三数之和.md](../../leetcode/daily/week2/day3/三数之和.md) |
| Week2 Day4 | 链表 | 21. 合并两个有序链表 | [合并两个有序链表.md](../../leetcode/daily/week2/day4/合并两个有序链表.md) |
| Week2 Day5 | 树 / BFS | 102. 二叉树的层序遍历 | [二叉树的层序遍历.md](../../leetcode/daily/week2/day5/二叉树的层序遍历.md) |
| Week2 Day6 | 单调栈 | 739. 每日温度 | [每日温度.md](../../leetcode/daily/week2/day6/每日温度.md) |
| Week2 Day7 | 单调队列 | 239. 滑动窗口最大值 | [滑动窗口最大值.md](../../leetcode/daily/week2/day7/滑动窗口最大值.md) |
| Week3 Day1 | 双指针 | 11. 盛最多水的容器 | [盛最多水的容器.md](../../leetcode/daily/week3/day1/盛最多水的容器.md) |
| Week3 Day2 | 动态规划 | 198. 打家劫舍 | [打家劫舍.md](../../leetcode/daily/week3/day2/打家劫舍.md) |
| Week3 Day3 | 字符串 | 5. 最长回文子串 | [最长回文子串.md](../../leetcode/daily/week3/day3/最长回文子串.md) |
| Week3 Day4 | 链表 | 141. 环形链表 | [环形链表.md](../../leetcode/daily/week3/day4/环形链表.md) |
| Week3 Day5 | 树 | 98. 验证二叉搜索树 | [验证二叉搜索树.md](../../leetcode/daily/week3/day5/验证二叉搜索树.md) |
| Week3 Day6 | 回溯 | 78. 子集 | [子集.md](../../leetcode/daily/week3/day6/子集.md) |
| Week6 Day3 | 设计 / 哈希+双向链表 | 146. LRU 缓存 | [LRU缓存.md](../../leetcode/daily/week6/day3/LRU缓存.md) |
| Week6 Day4 | 图 / 拓扑排序 | 207. 课程表 | [课程表.md](../../leetcode/daily/week6/day4/课程表.md) |
| Week6 Day5 | 图 / DFS 连通分量 | 200. 岛屿数量 | [岛屿数量.md](../../leetcode/daily/week6/day5/岛屿数量.md) |
| Week6 Day6 | 数组 / 一次遍历 | 121. 买卖股票的最佳时机 | [买卖股票的最佳时机.md](../../leetcode/daily/week6/day6/买卖股票的最佳时机.md) |
| Week6 Day7 | 滑动窗口 / 困难 | 76. 最小覆盖子串 | [最小覆盖子串.md](../../leetcode/daily/week6/day7/最小覆盖子串.md) |
| Week7 Day1 | 哈希表 | 128. 最长连续序列 | [最长连续序列.md](../../leetcode/daily/week7/day1/最长连续序列.md) |
| Week7 Day2 | 贪心 | 621. 任务调度器 | [任务调度器.md](../../leetcode/daily/week7/day2/任务调度器.md) |
| Week7 Day3 | 动态规划 | 139. 单词拆分 | [单词拆分.md](../../leetcode/daily/week7/day3/单词拆分.md) |
| Week7 Day4 | 设计 / 树 | 208. 实现 Trie (前缀树) | [实现Trie.md](../../leetcode/daily/week7/day4/实现Trie.md) |
| Week7 Day5 | 堆 / 优先队列 | 23. 合并 K 个升序链表 | [合并K个升序链表.md](../../leetcode/daily/week7/day5/合并K个升序链表.md) |
| Week7 Day6 | 堆 / 设计 | 295. 数据流的中位数 | [数据流的中位数.md](../../leetcode/daily/week7/day6/数据流的中位数.md) |
| Week7 Day7 | 设计 / LRU | 146. LRU 缓存 | [LRU缓存.md](../../leetcode/daily/week7/day7/LRU缓存.md) |
| Week8 Day1 | 数组 / 排序 / 贪心 | 56. 合并区间 | [合并区间.md](../../leetcode/daily/week8/day1/合并区间.md) |
| Week8 Day2 | 图 / 多源 BFS | 994. 腐烂的橘子 | [腐烂的橘子.md](../../leetcode/daily/week8/day2/腐烂的橘子.md) |
| Week8 Day3 | 动态规划 / 完全背包 | 322. 零钱兑换 | [零钱兑换.md](../../leetcode/daily/week8/day3/零钱兑换.md) |
| Week8 Day4 | 图 / 拓扑排序 | 207. 课程表 | [课程表.md](../../leetcode/daily/week6/day4/课程表.md) |
| Week8 Day4 | 动态规划 / 二维 DP | 72. 编辑距离 | [编辑距离.md](../../leetcode/daily/week8/day4/编辑距离.md) |
| Week8 Day5 | 困难 / 栈 + DP | 32. 最长有效括号 | [最长有效括号.md](../../leetcode/daily/week8/day5/最长有效括号.md) |
| Week8 Day6 | 动态规划 / 二分 | 300. 最长递增子序列 | [最长递增子序列.md](../../leetcode/daily/week8/day6/最长递增子序列.md) |
| Week8 Day7 | 链表 / 归并排序 | 148. 排序链表 | [排序链表.md](../../leetcode/daily/week8/day7/排序链表.md) |

> 💡 新增 Day 若上表无对应题解,按 [`leetcode/daily/SKILL.md`](../../leetcode/daily/SKILL.md) 在 `leetcode/daily/weekN/dayM/` 下新建 `<题目名>.md` 并补入上表。

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

**提交与推送**:
构建验证通过后,将新增/修改的文件提交并推送到远程:
```bash
git add -A
git commit -m "docs(weekN/dayM): <主题>"
git push origin
```

## 7. 检查清单(写完一个 Day 后自检)

- [ ] 首行是 `## Day N：<主题>`(中文全角冒号)
- [ ] `### 🎯 目标` 紧跟标题,含 6 条编号 + `> 💡 为什么重要`
- [ ] 有 `### 学前导读`(总结日除外)
- [ ] `### 理论学习` 用 `#### N.x` 分节,配 SVG
- [ ] `### Coding 任务` 含 5 个任务(含 1 道 LeetGPU 在线题目 + 1 道 LeetCode 面试题),代码完整可编译,带 nvcc 命令 + 预期输出
- [ ] LeetGPU 题目与当日主题强相关,题解归档到 `leetgpu/leetgpu-<slug>-solution.md`
- [ ] LeetCode 题目为面试高频题,题解归档到 `leetcode/daily/weekN/dayM/<题目名>.md`
- [ ] `### 扩展实验` 3 个
- [ ] `### 今日总结` 5-7 条加粗编号
- [ ] `### 面试要点` 5 题问答
- [ ] 所有文件链接用相对路径且指向真实文件
- [ ] SVG 引用格式 `![中文alt](../website/images/xxx.svg)`
- [ ] 运行 `python3 weekN/website/build.py` 成功生成 `dayN.html`
- [ ] 提交并推送更改：`git add -A && git commit -m "docs(weekN/dayM): <主题>" && git push origin`
