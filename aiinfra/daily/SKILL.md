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
weekN/images/*.svg  # SVG 图(语义化小写命名,如 warp_shuffle_primitives.svg)
```

- 教程中用相对路径引用本地文件:`[kernels/hello_gpu.cu](kernels/hello_gpu.cu)`
- SVG 引用:`![中文alt描述](../images/xxx.svg)`(从 dayM/ 出发,`../images/` 解析到 weekN/images/)

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
- 配 SVG 图,插在小节首行:`![<中文alt>](../images/<filename>.svg)`
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

每天 Coding 任务必须包含一道来自 **https://leetgpu.com/** 的经典 CUDA 在线题目,与当日主题强相关、避免重复。完整题解的写作规范(6 段结构、Kernel 代码、SVG 插图、ncu profiling)见 [`leetgpu/SKILL.md`](https://github.com/hzchenxiaobin/leetgpu/blob/main/SKILL.md);教程"任务 4"只需给出题目链接 + 1-2 句与当日知识的关联,并用相对链接指向下表对应题解。

**已归档题解**(位于 [独立 leetgpu 仓库](https://github.com/hzchenxiaobin/leetgpu)的 `weekN/dayM/`):

| 教程 | 主题 | LeetGPU 题目 | 题解链接 |
|------|------|--------------|----------|
| Week1 Day1 | GPU 执行模型基础 | Vector Addition | [leetgpu-vector-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-vector-addition-solution.html) |
| Week1 Day2 | Occupancy 与资源约束 | ReLU | [leetgpu-relu-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-relu-solution.html) |
| Week1 Day3 | 认识你的 GPU —— deviceQuery 与 Occupancy 计算 | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week1 Day4 | Memory Hierarchy 深入 | Matrix Transpose | [leetgpu-matrix-transpose-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-transpose-solution.html) |
| Week1 Day5 | Bank Conflict 分析与实践 | Reduction | [leetgpu-reduction-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-reduction-solution.html) |
| Week1 Day6 | Nsight Profiling 实战 | Matrix Multiplication | [leetgpu-matrix-multiplication-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-multiplication-solution.html) |
| Week1 Day7 | 总结与复盘 | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week2 Day1 | Warp Shuffle 原语与 Warp/Block Reduce | Prefix Sum | [leetgpu-prefix-sum-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-prefix-sum-solution.html) |
| Week2 Day2 | Register Blocking 与 2D Tiling | GEMM | [leetgpu-gemm-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gemm-solution.html) |
| Week2 Day3 | CUDA Streams 与异步执行 | 2D Convolution | [leetgpu-2d-convolution-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-2d-convolution-solution.html) |
| Week2 Day4 | Nsight Compute 性能分析 | Softmax | [leetgpu-softmax-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-solution.html) |
| Week2 Day5 | FlashAttention CUDA 实现（简化版） | Softmax Attention | [leetgpu-softmax-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-attention-solution.html) |
| Week2 Day6 | 整合优化到 cuBLAS 70%+ | Histogramming | [leetgpu-histogramming-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-histogramming-solution.html) |
| Week2 Day7 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | Max Subarray Sum | [leetgpu-max-subarray-sum-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-max-subarray-sum-solution.html) |
| Week3 Day1 | Trace Transformer 推理流程 | 1D Convolution | [leetgpu-1d-convolution-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-1d-convolution-solution.html) |
| Week3 Day2 | 手写 Softmax 与 LayerNorm Kernel | Softmax | [leetgpu-softmax-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-solution.html) |
| Week3 Day3 | 源码分析 —— PyTorch / FasterTransformer | Argmax | [leetgpu-argmax-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-argmax-solution.html) |
| Week3 Day4 | Attention IO 分析 | Attention | [leetgpu-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-attention-solution.html) |
| Week3 Day5 | 算子接入 Mini 引擎 | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week3 Day6 | 端到端 Profiling 与 Kernel Fusion | RMS Normalization | [leetgpu-rms-normalization-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-rms-normalization-solution.html) |
| Week3 Day7 | Transformer 算子分类与 Week 3 总结 | Causal Self-Attention | [leetgpu-causal-self-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-causal-self-attention-solution.html) |
| Week4 Day1 | FlashAttention 论文精读与 Online Softmax 完整推导 | Softmax Attention | [leetgpu-softmax-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-attention-solution.html) |
| Week4 Day2 | 手写完整 FlashAttention Forward Kernel | Attention | [leetgpu-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-attention-solution.html) |
| Week4 Day3 | FlashAttention 官方 CUDA 源码分析 | Dot Product | [leetgpu-dot-product-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dot-product-solution.html) |
| Week4 Day4 | FlashAttention-2 论文与源码差异 | Batched Matrix Multiplication | [leetgpu-batched-matrix-multiplication-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batched-matrix-multiplication-solution.html) |
| Week4 Day5 | 算子接入 Mini 引擎 —— FlashAttention 集成 | Matrix Copy | [leetgpu-matrix-copy-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-copy-solution.html) |
| Week4 Day6 | 性能对比分析 —— 标准 vs 手写 vs 官方 | Multi-Head Attention | [leetgpu-multi-head-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-multi-head-attention-solution.html) |
| Week4 Day7 | IO 优化方法论总结与 Week 4 收官 | GPT-2 Transformer Block | [leetgpu-gpt-2-transformer-block-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gpt-2-transformer-block-solution.html) |
| Week5 Day1 | 推理流程 —— Prefill vs Decode | INT8 KV-Cache Attention | [leetgpu-int8-kv-cache-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int8-kv-cache-attention-solution.html) |
| Week5 Day2 | 实现 KV Cache | Grouped Query Attention (GQA) | [leetgpu-grouped-query-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-grouped-query-attention-solution.html) |
| Week5 Day3 | vLLM 整体架构分析 | Speculative Decoding Verification | [leetgpu-speculative-decoding-verification-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-speculative-decoding-verification-solution.html) |
| Week5 Day4 | vLLM Worker 与 PagedAttention | Causal Self-Attention | [leetgpu-causal-self-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-causal-self-attention-solution.html) |
| Week5 Day5 | 项目推进 —— Mini 推理引擎 v0 | Token Embedding Layer | [leetgpu-token-embedding-layer-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-token-embedding-layer-solution.html) |
| Week5 Day6 | 端到端 Profiling | Weight Dequantization | [leetgpu-weight-dequantization-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-weight-dequantization-solution.html) |
| Week5 Day7 | 推理系统核心问题总结与 Week 5 收官 | Simple Inference | [leetgpu-simple-inference-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-simple-inference-solution.html) |
| Week6 Day1 | Dynamic Batching | Simple Inference | [leetgpu-simple-inference-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-simple-inference-solution.html) |
| Week6 Day2 | Continuous Batching | Max Subarray Sum | [leetgpu-max-subarray-sum-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-max-subarray-sum-solution.html) |
| Week6 Day3 | vLLM Scheduler 源码分析 | Stream Compaction | [leetgpu-stream-compaction-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-stream-compaction-solution.html) |
| Week6 Day4 | TensorRT-LLM / LightLLM 调度对比 | Segmented Prefix Sum | [leetgpu-segmented-prefix-sum-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-segmented-prefix-sum-solution.html) |
| Week6 Day5 | Mini 推理引擎 v1 | Batched Matrix Multiplication | [leetgpu-batched-matrix-multiplication-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batched-matrix-multiplication-solution.html) |
| Week6 Day6 | Latency / Throughput 测试 | Top K Selection | [leetgpu-top-k-selection-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-top-k-selection-solution.html) |
| Week6 Day7 | 调度优化策略总结与 Week 6 收官 | Dot Product | [leetgpu-dot-product-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dot-product-solution.html) |
| Week7 Day1 | 多请求并发支持 | Matrix Copy | [leetgpu-matrix-copy-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-copy-solution.html) |
| Week7 Day2 | 完整调度器 | Vector Reversal | [leetgpu-vector-reversal-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-vector-reversal-solution.html) |
| Week7 Day3 | SGLang / LightLLM 高级特性 | Scalar Multiply | [leetgpu-scalar-multiply-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-scalar-multiply-solution.html) |
| Week7 Day4 | 整合全部自定义 Kernel | Matrix Transpose | [leetgpu-matrix-transpose-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-transpose-solution.html) |
| Week7 Day5 | 系统联调 | Element Reversal | [leetgpu-element-reversal-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-element-reversal-solution.html) |
| Week7 Day6 | 全链路 Profiling | Reduction | [leetgpu-reduction-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-reduction-solution.html) |
| Week7 Day7 | 代码重构与文档 | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week8 Day1 | 项目文档完善 | SiLU | [leetgpu-silu-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-silu-solution.html) |
| Week8 Day2 | 架构图与数据流图 | Rotary Positional Embedding | [leetgpu-rope-embedding-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-rope-embedding-solution.html) |
| Week8 Day3 | 高频面试题基础篇 | SwiGLU | [leetgpu-swiglu-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-swiglu-solution.html) |
| Week8 Day4 | 高频面试题进阶篇 | Sliding Window Self-Attention | [leetgpu-sliding-window-self-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sliding-window-self-attention-solution.html) |
| Week8 Day5 | Mock 面试 | LoRA Linear | [leetgpu-lora-linear-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-lora-linear-solution.html) |
| Week8 Day6 | 查漏补缺 | Batch Normalization | [leetgpu-batch-normalization-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batch-normalization-solution.html) |
| Week8 Day7 | 最终复盘 | 1D Convolution | [leetgpu-1d-convolution-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-1d-convolution-solution.html) |

> 💡 新增 Day 若上表无对应题解,按 [`leetgpu/SKILL.md`](https://github.com/hzchenxiaobin/leetgpu/blob/main/SKILL.md) 在独立仓库的 `weekN/dayM/` 下新建 `leetgpu-<slug>-solution.md` 并补入上表。

#### LeetCode 面试题(任务 5,必有)

每天 Coding 任务额外包含一道来自 **https://leetcode.cn/** 的面试高频题,作为算法基本功的日常练习。完整题解(6 段结构、C++/Python 参考代码、手绘 SVG、复杂度分析)已归档到独立的 [LeetCode 题解仓库](https://hzchenxiaobin.github.io/leetcode/);教程"任务 5"只需给出题目链接 + 1-2 句核心套路点题,并用相对链接指向下表对应题解。

**已归档题解**(位于 [独立 LeetCode 题解仓库](https://hzchenxiaobin.github.io/leetcode/)):

| 教程 | 主题 | LeetCode 题目 | 题解链接 |
|------|------|---------------|----------|
| Week1 Day1 | 数组 / 双指针 | 42. 接雨水 | [42_接雨水.md](https://hzchenxiaobin.github.io/leetcode/problems/42_接雨水.html) |
| Week1 Day2 | 动态规划 | 53. 最大子数组和 | [53_最大子数组和.md](https://hzchenxiaobin.github.io/leetcode/problems/53_最大子数组和.html) |
| Week1 Day3 | 字符串 / 滑窗 | 3. 无重复字符的最长子串 | [3_无重复字符的最长子串.md](https://hzchenxiaobin.github.io/leetcode/problems/3_无重复字符的最长子串.html) |
| Week1 Day4 | 链表 | 206. 反转链表 | [206_反转链表.md](https://hzchenxiaobin.github.io/leetcode/problems/206_反转链表.html) |
| Week1 Day5 | 树 / DFS | 236. 二叉树的最近公共祖先 | [236_二叉树的最近公共祖先.md](https://hzchenxiaobin.github.io/leetcode/problems/236_二叉树的最近公共祖先.html) |
| Week1 Day6 | 回溯 | 46. 全排列 | [46_全排列.md](https://hzchenxiaobin.github.io/leetcode/problems/46_全排列.html) |
| Week1 Day7 | 栈 / 困难 | 84. 柱状图中最大的矩形 | [84_柱状图中最大的矩形.md](https://hzchenxiaobin.github.io/leetcode/problems/84_柱状图中最大的矩形.html) |
| Week2 Day1 | 哈希表 | 1. 两数之和 | [1_两数之和.md](https://hzchenxiaobin.github.io/leetcode/problems/1_两数之和.html) |
| Week2 Day2 | 动态规划 | 70. 爬楼梯 | [70_爬楼梯.md](https://hzchenxiaobin.github.io/leetcode/problems/70_爬楼梯.html) |
| Week2 Day3 | 双指针 | 15. 三数之和 | [15_三数之和.md](https://hzchenxiaobin.github.io/leetcode/problems/15_三数之和.html) |
| Week2 Day4 | 链表 | 21. 合并两个有序链表 | [21_合并两个有序链表.md](https://hzchenxiaobin.github.io/leetcode/problems/21_合并两个有序链表.html) |
| Week2 Day5 | 树 / BFS | 102. 二叉树的层序遍历 | [102_二叉树的层序遍历.md](https://hzchenxiaobin.github.io/leetcode/problems/102_二叉树的层序遍历.html) |
| Week2 Day6 | 单调栈 | 739. 每日温度 | [739_每日温度.md](https://hzchenxiaobin.github.io/leetcode/problems/739_每日温度.html) |
| Week2 Day7 | 单调队列 | 239. 滑动窗口最大值 | [239_滑动窗口最大值.md](https://hzchenxiaobin.github.io/leetcode/problems/239_滑动窗口最大值.html) |
| Week3 Day1 | 双指针 | 11. 盛最多水的容器 | [11_盛最多水的容器.md](https://hzchenxiaobin.github.io/leetcode/problems/11_盛最多水的容器.html) |
| Week3 Day2 | 动态规划 | 198. 打家劫舍 | [198_打家劫舍.md](https://hzchenxiaobin.github.io/leetcode/problems/198_打家劫舍.html) |
| Week3 Day3 | 字符串 | 5. 最长回文子串 | [5_最长回文子串.md](https://hzchenxiaobin.github.io/leetcode/problems/5_最长回文子串.html) |
| Week3 Day4 | 链表 | 141. 环形链表 | [141_环形链表.md](https://hzchenxiaobin.github.io/leetcode/problems/141_环形链表.html) |
| Week3 Day5 | 树 | 98. 验证二叉搜索树 | [98_验证二叉搜索树.md](https://hzchenxiaobin.github.io/leetcode/problems/98_验证二叉搜索树.html) |
| Week3 Day6 | 回溯 | 78. 子集 | [78_子集.md](https://hzchenxiaobin.github.io/leetcode/problems/78_子集.html) |
| Week6 Day3 | 设计 / 哈希+双向链表 | 146. LRU 缓存 | [146_LRU缓存.md](https://hzchenxiaobin.github.io/leetcode/problems/146_LRU缓存.html) |
| Week6 Day4 | 图 / 拓扑排序 | 207. 课程表 | [207_课程表.md](https://hzchenxiaobin.github.io/leetcode/problems/207_课程表.html) |
| Week6 Day5 | 图 / DFS 连通分量 | 200. 岛屿数量 | [200_岛屿数量.md](https://hzchenxiaobin.github.io/leetcode/problems/200_岛屿数量.html) |
| Week6 Day6 | 数组 / 一次遍历 | 121. 买卖股票的最佳时机 | [121_买卖股票的最佳时机.md](https://hzchenxiaobin.github.io/leetcode/problems/121_买卖股票的最佳时机.html) |
| Week6 Day7 | 滑动窗口 / 困难 | 76. 最小覆盖子串 | [76_最小覆盖子串.md](https://hzchenxiaobin.github.io/leetcode/problems/76_最小覆盖子串.html) |
| Week7 Day1 | 哈希表 | 128. 最长连续序列 | [最长连续序列.md](https://hzchenxiaobin.github.io/leetcode/problems/128_最长连续序列.html) |
| Week7 Day2 | 树 / 递归 | 101. 对称二叉树 | [对称二叉树.md](https://hzchenxiaobin.github.io/leetcode/problems/101_对称二叉树.html) |
| Week7 Day3 | 动态规划 | 139. 单词拆分 | [单词拆分.md](https://hzchenxiaobin.github.io/leetcode/problems/139_单词拆分.html) |
| Week7 Day4 | 设计 / 树 | 208. 实现 Trie (前缀树) | [实现Trie.md](https://hzchenxiaobin.github.io/leetcode/problems/208_实现Trie.html) |
| Week7 Day5 | 堆 / 优先队列 | 23. 合并 K 个升序链表 | [合并K个升序链表.md](https://hzchenxiaobin.github.io/leetcode/problems/23_合并K个升序链表.html) |
| Week7 Day6 | 堆 / 设计 | 295. 数据流的中位数 | [数据流的中位数.md](https://hzchenxiaobin.github.io/leetcode/problems/295_数据流的中位数.html) |
| Week7 Day7 | 回溯 | 51. N 皇后 | [N 皇后.md](https://hzchenxiaobin.github.io/leetcode/problems/51_N皇后.html) |
| Week8 Day1 | 数组 / 排序 / 贪心 | 56. 合并区间 | [合并区间.md](https://hzchenxiaobin.github.io/leetcode/problems/56_合并区间.html) |
| Week8 Day2 | 图 / 多源 BFS | 994. 腐烂的橘子 | [腐烂的橘子.md](https://hzchenxiaobin.github.io/leetcode/problems/994_腐烂的橘子.html) |
| Week8 Day3 | 动态规划 / 完全背包 | 322. 零钱兑换 | [零钱兑换.md](https://hzchenxiaobin.github.io/leetcode/problems/322_零钱兑换.html) |
| Week8 Day4 | 图 / 拓扑排序 | 207. 课程表 | [课程表.md](https://hzchenxiaobin.github.io/leetcode/problems/207_课程表.html) |
| Week8 Day4 | 动态规划 / 二维 DP | 72. 编辑距离 | [编辑距离.md](https://hzchenxiaobin.github.io/leetcode/problems/72_编辑距离.html) |
| Week8 Day5 | 困难 / 栈 + DP | 32. 最长有效括号 | [最长有效括号.md](https://hzchenxiaobin.github.io/leetcode/problems/32_最长有效括号.html) |
| Week8 Day6 | 动态规划 / 二分 | 300. 最长递增子序列 | [最长递增子序列.md](https://hzchenxiaobin.github.io/leetcode/problems/300_最长递增子序列.html) |
| Week8 Day7 | 链表 / 归并排序 | 148. 排序链表 | [排序链表.md](https://hzchenxiaobin.github.io/leetcode/problems/148_排序链表.html) |

> 💡 新增 Day 若上表无对应题解,在 [独立 LeetCode 题解仓库](https://hzchenxiaobin.github.io/leetcode/) 的 `daily/weekN/dayM/` 下新建 `<题目名>.md` 并补入上表。

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

### 数学公式

- 行内公式用 `$...$`，块级公式用 `$$...$$`
- **禁止**用反引号 `` `...` `` 包裹数学公式，否则会被渲染为等宽代码，KaTeX 不会识别
- 公式内函数/运算符使用 LaTeX 命令：`\exp`、`\log`、`\sum`、`\max`、`\frac`、`\sqrt`，避免直接写 `exp`、`log`、`Σ`、`√`


### 量化指标
- 教学日平均约 **550 行 / 14 个代码块**
- 开篇日(Day1)和含完整 kernel 实现的日代码块最多(~26)
- 总结日代码块最少(~6-8)

### 图片
- **禁止 ASCII 图片**:所有示意图、流程图、架构图一律用 SVG,不要在 Markdown 中嵌入 ASCII 字符画(如用 `+---+`、`|   |` 拼成的表格或流程图)
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

写完 dayN/README.md 后,教程会被 `build/weeks.py` 自动读取并生成 `dayN.html`:
- `build.py` 遍历 `weekN/day*/README.md`,解析首行 `## Day N：<title>`
- 图片路径 `../images/` 会被重写为 `images/`(网站输出目录)
- `.md` 链接会被重写为 `.html`(GitHub Pages 部署)

**验证命令**:
```bash
python3 build.py                   # 组合构建(含 week1~week8/leetgpu/topics/paper)
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
- [ ] LeetCode 题目为面试高频题,题解归档到[独立 LeetCode 题解仓库](https://hzchenxiaobin.github.io/leetcode/)的 `daily/weekN/dayM/<题目名>.md`
- [ ] `### 扩展实验` 3 个
- [ ] `### 今日总结` 5-7 条加粗编号
- [ ] `### 面试要点` 5 题问答
- [ ] 所有文件链接用相对路径且指向真实文件
- [ ] SVG 引用格式 `![中文alt](../images/xxx.svg)`
- [ ] 运行 `python3 build.py` 成功生成 `dayN.html`
- [ ] 提交并推送更改：`git add -A && git commit -m "docs(weekN/dayM): <主题>" && git push origin`
