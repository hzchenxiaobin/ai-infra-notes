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

- 教程中用 GitHub 完整链接引用本地文件:`[kernels/hello_gpu.cu](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/weekN/dayM/kernels/hello_gpu.cu)`(GitHub Pages 上相对路径会 404)
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
- **文件链接**:用 GitHub 完整链接 `[kernels/xxx.cu](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/weekN/dayM/kernels/xxx.cu)` 引用真实文件(GitHub Pages 上相对路径会 404)
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
| Week2 Day3 | float4 向量化 + GEMM 七层路径（前四层） | GEMM | [leetgpu-gemm-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gemm-solution.html) |
| Week2 Day4 | GEMM 七层路径（后三层）+ cuBLAS 对比 | 待定 | — |
| Week2 Day5 | CUDA Streams 与异步执行 | 2D Convolution | [leetgpu-2d-convolution-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-2d-convolution-solution.html) |
| Week2 Day6 | Nsight Compute 性能分析 | Softmax | [leetgpu-softmax-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-solution.html) |
| Week2 Day7 | 限时 Kernel 手撕 + GitHub 整理 | Max Subarray Sum | [leetgpu-max-subarray-sum-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-max-subarray-sum-solution.html) |
| Week3 Day1 | Tensor Core 与 WMMA | Histogramming | [leetgpu-histogramming-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-histogramming-solution.html) |
| Week3 Day2 | 手写 WMMA GEMM 与 cuBLAS 性能对比 | 待定 | — |
| Week3 Day3 | mma.sync 指令与 ldmatrix | 待定 | — |
| Week3 Day4 | CUTLASS 源码分析 + CuTe 概念铺垫 | 待定 | — |
| Week3 Day5 | 项目推进 —— WMMA GEMM 接入 Benchmark | 待定 | — |
| Week3 Day6 | Profiling —— Tensor Core 利用率 | 待定 | — |
| Week3 Day7 | 复盘与手撕 —— Tensor Core/CUTLASS | 待定 | — |
| Week4 Day1 | Trace Transformer 推理流程（Prefill/Decode） | 1D Convolution | [leetgpu-1d-convolution-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-1d-convolution-solution.html) |
| Week4 Day2 | 手写 Softmax 与 LayerNorm Kernel | Softmax | [leetgpu-softmax-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-solution.html) |
| Week4 Day3 | LayerNorm 优化与 GEMM Backward 数据流 | RMS Normalization | [leetgpu-rms-normalization-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-rms-normalization-solution.html) |
| Week4 Day4 | Triton 语言专题 | Argmax | [leetgpu-argmax-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-argmax-solution.html) |
| Week4 Day5 | 项目推进 —— Triton 三方 Benchmark | 待定 | — |
| Week4 Day6 | Profiling —— Triton vs CUDA vs PyTorch | 待定 | — |
| Week4 Day7 | Transformer 算子分类与总结 | Causal Self-Attention | [leetgpu-causal-self-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-causal-self-attention-solution.html) |
| Week5 Day1 | FA CUDA 实现（简化版）+ Attention IO 分析 | Softmax Attention | [leetgpu-softmax-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-attention-solution.html) |
| Week5 Day2 | FA 论文精读与 Online Softmax 推导 | Attention | [leetgpu-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-attention-solution.html) |
| Week5 Day3 | 手写完整 FA Forward Kernel | Multi-Head Attention | [leetgpu-multi-head-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-multi-head-attention-solution.html) |
| Week5 Day4 | FA Backward 与 GEMM Backward | Dot Product | [leetgpu-dot-product-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dot-product-solution.html) |
| Week5 Day5 | 性能对比 + FA 接入 Mini 引擎 | Matrix Copy | [leetgpu-matrix-copy-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-copy-solution.html) |
| Week5 Day6 | FA-2 + 官方源码 + IO 方法论 | Batched Matrix Multiplication | [leetgpu-batched-matrix-multiplication-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batched-matrix-multiplication-solution.html) |
| Week5 Day7 | 复盘与手撕 —— FA 限时手写 | GPT-2 Transformer Block | [leetgpu-gpt-2-transformer-block-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gpt-2-transformer-block-solution.html) |
| Week6 Day1 | 推理流程 —— Prefill vs Decode | INT8 KV-Cache Attention | [leetgpu-int8-kv-cache-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int8-kv-cache-attention-solution.html) |
| Week6 Day2 | 实现 KV Cache（含 GQA/MQA/MLA） | Grouped Query Attention (GQA) | [leetgpu-grouped-query-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-grouped-query-attention-solution.html) |
| Week6 Day3 | vLLM 整体架构分析 | Speculative Decoding Verification | [leetgpu-speculative-decoding-verification-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-speculative-decoding-verification-solution.html) |
| Week6 Day4 | vLLM Worker 与 PagedAttention | Causal Self-Attention | [leetgpu-causal-self-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-causal-self-attention-solution.html) |
| Week6 Day5 | 项目推进 —— Mini 推理引擎 v0 | Token Embedding Layer | [leetgpu-token-embedding-layer-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-token-embedding-layer-solution.html) |
| Week6 Day6 | FlashDecoding | Weight Dequantization | [leetgpu-weight-dequantization-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-weight-dequantization-solution.html) |
| Week6 Day7 | 推理系统核心问题总结 | Simple Inference | [leetgpu-simple-inference-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-simple-inference-solution.html) |
| Week7 Day1 | Continuous Batching（含 Dynamic Batching） | Simple Inference | [leetgpu-simple-inference-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-simple-inference-solution.html) |
| Week7 Day2 | vLLM Scheduler 源码分析 | Stream Compaction | [leetgpu-stream-compaction-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-stream-compaction-solution.html) |
| Week7 Day3 | TRT-LLM / LightLLM / SGLang 调度对比 | Segmented Prefix Sum | [leetgpu-segmented-prefix-sum-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-segmented-prefix-sum-solution.html) |
| Week7 Day4 | Chunked Prefill 与 Prefix Caching | Batched Matrix Multiplication | [leetgpu-batched-matrix-multiplication-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batched-matrix-multiplication-solution.html) |
| Week7 Day5 | Mini 推理引擎 v1（多请求并发） | Top K Selection | [leetgpu-top-k-selection-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-top-k-selection-solution.html) |
| Week7 Day6 | PD 分离推理 | Dot Product | [leetgpu-dot-product-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dot-product-solution.html) |
| Week7 Day7 | 调度优化策略总结 | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week8 Day1 | 量化推理专题 —— W8A16/INT8 KV/FP8 | Weight Dequantization | [leetgpu-weight-dequantization-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-weight-dequantization-solution.html) |
| Week8 Day2 | FP8 量化深入 | 待定 | — |
| Week8 Day3 | SGLang / 投机解码 | Scalar Multiply | [leetgpu-scalar-multiply-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-scalar-multiply-solution.html) |
| Week8 Day4 | CUDA Graph 实操 | Matrix Transpose | [leetgpu-matrix-transpose-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-transpose-solution.html) |
| Week8 Day5 | 项目推进 —— 加速技术接入 | 待定 | — |
| Week8 Day6 | Profiling —— 量化/CUDA Graph | Reduction | [leetgpu-reduction-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-reduction-solution.html) |
| Week8 Day7 | 复盘与面试 Q&A | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week9 Day1 | 分布式推理 —— TP/PP/DP | Matrix Copy | [leetgpu-matrix-copy-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-copy-solution.html) |
| Week9 Day2 | Pipeline Parallelism 与 DP | Vector Reversal | [leetgpu-vector-reversal-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-vector-reversal-solution.html) |
| Week9 Day3 | NCCL Collectives | 待定 | — |
| Week9 Day4 | 通信计算重叠 | 待定 | — |
| Week9 Day5 | MoE + EP 并行专题 | 待定 | — |
| Week9 Day6 | 多硬件对比：CUDA vs Ascend | 待定 | — |
| Week9 Day7 | 复盘与面试 Q&A | Element Reversal | [leetgpu-element-reversal-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-element-reversal-solution.html) |
| Week10 Day1 | 整合全部自定义 Kernel | Matrix Transpose | [leetgpu-matrix-transpose-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-transpose-solution.html) |
| Week10 Day2 | 系统联调（六步分层验证） | Element Reversal | [leetgpu-element-reversal-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-element-reversal-solution.html) |
| Week10 Day3 | 项目文档完善（README） | Matrix Addition | [leetgpu-matrix-addition-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-addition-solution.html) |
| Week10 Day4 | 高频面试题基础篇 | SiLU | [leetgpu-silu-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-silu-solution.html) |
| Week10 Day5 | Mock 面试 | LoRA Linear | [leetgpu-lora-linear-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-lora-linear-solution.html) |
| Week10 Day6 | 诊断流程实战剧本 + 手撕清单 | Sliding Window Self-Attention | [leetgpu-sliding-window-self-attention-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sliding-window-self-attention-solution.html) |
| Week10 Day7 | 最终复盘 —— 10 周能力地图 | 1D Convolution | [leetgpu-1d-convolution-solution.md](https://hzchenxiaobin.github.io/leetgpu/leetgpu-1d-convolution-solution.html) |

> 💡 新增 Day 若上表无对应题解,按 [`leetgpu/SKILL.md`](https://github.com/hzchenxiaobin/leetgpu/blob/main/SKILL.md) 在独立仓库的 `weekN/dayM/` 下新建 `leetgpu-<slug>-solution.md` 并补入上表。

#### LeetCode 面试题(任务 5,必有)

每天 Coding 任务包含一组来自 **https://leetcode.cn/** 的面试高频题,作为算法基本功的日常练习。题目安排与独立的 [8 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/8-week-plan.html) 对齐——该计划把 Hot 100 / CodeTop / 面试经典 150 / 剑指 Offer 共 198 道高频题按类别编排为 8 周,与本教程 week1-week8 一一对应;完整题目清单另见 [高频算法面试题汇总](https://hzchenxiaobin.github.io/leetcode/problems/hot-interview.html)。

**安排规则**:

- **Day 1-6**:每天 3-5 题(第 2、5 周题量较少,末尾 1-2 天为机动补漏日,不新增题目)。教程中给出「题目 | 难度 | 核心套路 | 题解」表格 + 1-2 句刷题建议
- **Day 7**:本周 LeetCode 题目回顾(汇总表),重做本周错题、总结模板笔记

**每周主题与题量**(与 8 周计划一致):

| 周 | 主题 | 题量 |
|----|------|------|
| Week 1 | 数组、哈希与双指针(含手撕排序) | 26 |
| Week 2 | 字符串、滑动窗口与矩阵 | 20 |
| Week 3 | 链表与数学技巧 | 24 |
| Week 4 | 栈、队列、堆、设计与贪心区间 | 29 |
| Week 5 | 二叉树(上)——遍历、形态与 BST | 20 |
| Week 6 | 二叉树(下)+ 回溯 + 网格搜索 | 25 |
| Week 7 | 二分查找与动态规划基础 | 25 |
| Week 8 | 动态规划进阶与图论 | 29 |

完整题解(6 段结构、C++/Python 参考代码、手绘 SVG、复杂度分析)已归档到独立的 [LeetCode 题解仓库](https://hzchenxiaobin.github.io/leetcode/);教程只需给出题目链接 + 核心套路,并链接对应题解页面(`https://hzchenxiaobin.github.io/leetcode/problems/<题号>_<题名>.html`)。

> 💡 修改某天的 LeetCode 题目时,以 [8 周刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/8-week-plan.html) 为准;题解缺失的题目按题解仓库 `solution/SKILL.md` 1.3「优先补全清单」先在题解仓库补写。

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
- 验收日(如 week10/day3)的面试题可含"评分关键"

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

**验收型 Day7**(如 week10/day3)额外含:
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

### 硬件参数与必背数字（唯一事实源，硬性条款）

- **教程正文一律引用 `reference/hardware_specs.md` 与 `reference/key_numbers.md`，禁止在正文中新写硬件参数数字**（算力、带宽、SM 数、Ridge Point、KV Cache 口算结果、GEMM 优化增益、Attention IO 加速比等）。正文中需要时用一句话 + 链接引用，如"RTX 5090 峰值 FP32 见 [硬件参数事实源](../../reference/hardware_specs.md)"（从 `weekN/dayM/` 出发为 `../../reference/`）
- 确需在正文出现数字时（如推导中间步骤），必须与事实源**逐位一致**；事实源中标注 `[需核实]` 的数字禁止写入正文
- RTX 5090 基准三数：**104.75 TFLOPS / 1792 GB/s / Ridge Point 58.45**（实测，源自 `week1/day3/exercise/my_gpu_info.md`）；`19.5 TFLOPS`、`1.55 TB/s`、`12.6` 是 **A100** 的参数，只允许出现在"A100 对比 / 错误示范"的显式标注语境
- 发现正文数字与事实源冲突时，以事实源为准修正正文，并在提交信息中注明

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
- [ ] 硬件参数/必背数字引用唯一事实源，正文无新写硬件参数数字；grep 自检：
  ```bash
  # 新写/修改的 dayN 中出现的算力、带宽数字应与 reference/ 完全一致
  grep -n "TFLOPS\|GB/s\|Ridge Point" aiinfra/daily/weekN/dayM/README.md
  # A100 旧数字只允许出现在显式标注的对比/错误示范语境
  grep -rn "19\.5 TFLOP\|1\.55 TB/s\|12\.6" aiinfra/daily/weekN/dayM/README.md
  ```
- [ ] README 内嵌 ```python 代码块缩进无损坏（禁止"单前导空格"剥落）：
  ```bash
  # 提取所有 python 代码块做 ast.parse，IndentationError 即报错
  python3 build/lint_md_code.py
  # 期望输出：Checked N files, M code blocks, 0 errors
  ```
- [ ] 提交并推送更改：`git add -A && git commit -m "docs(weekN/dayM): <主题>" && git push origin`
