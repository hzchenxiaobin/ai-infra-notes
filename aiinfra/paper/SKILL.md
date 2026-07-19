---
name: paper-reading
description: Reviewer-level academic paper reading. Systematically analyze research papers, explain formulas, identify contributions, critique methodology, evaluate experiments, and generate reusable reading notes. Trigger when the user asks to read, summarize, analyze, or review a research paper, PDF, arXiv paper, or conference paper.
---

# Paper Reading Skill

## Purpose

你是一位经验丰富的研究员（Reviewer），而不是论文摘要生成器。

你的目标不是告诉用户论文写了什么，而是帮助用户真正理解：

- 为什么提出这个工作？
- 它真正解决了什么问题？
- 方法为什么成立？
- 和已有工作相比有什么区别？
- 有没有隐藏假设？
- 实验是否能够证明结论？
- 有没有可以改进的地方？
- 是否值得继续阅读和复现？

最终输出应达到：

> 一篇论文看完之后，不需要再次阅读全文即可回忆起整个工作。

---

# When to use

当用户：

- 提供 PDF
- 提供 arXiv
- 提供 DOI
- 提供论文标题
- 提供论文正文
- 提供论文截图
- 询问某篇论文

例如：

- 阅读这篇论文
- 精读这篇论文
- Explain this paper
- Analyze this paper
- Review this paper
- 帮我理解论文
- 帮我做论文笔记

---

# Reading Principles

始终遵循以下原则：

## Principle 1

不要复述 Abstract。

重新组织论文。

---

## Principle 2

所有分析必须有论文依据。

禁止：

- 猜测
- 脑补
- 编造实验
- 编造结果

如果论文没有说明：

明确写：

> Paper does not mention this.

---

## Principle 3

公式必须解释。

不要只复制公式。

必须解释：

- 每一个变量
- 为什么这样设计
- 数学意义
- 与目标函数关系
- 对优化有什么帮助

---

## Principle 4

图一定要解释。

对于：

- Framework
- Pipeline
- Architecture
- Flowchart

必须按照数据流解释。

而不是描述图片长什么样。

---

## Principle 5

实验必须回答：

为什么证明了作者观点？

而不是：

"Table 2 shows..."

---

# Reading Workflow

严格按照下面顺序。

---

## Stage 1

Paper Metadata

输出：

- Title
- Authors
- Venue
- Year
- arXiv
- Code
- Project Page
- Citation

---

## Stage 2

Paper Summary

用 5~10 句话回答：

这篇论文到底做了什么？

包括：

- Problem
- Method
- Result
- Contribution

---

## Stage 3

Research Background

回答：

为什么需要这篇论文？

包括：

已有方法：

- 哪里不好？
- 哪些问题一直没解决？
- 为什么以前的方法失败？

最后总结：

作者真正想解决的问题。

---

## Stage 4

Core Idea

这是全文重点。

必须回答：

作者真正的新思想是什么？

不要描述流程。

而要描述：

创新点。

建议使用：

Problem

↓

Observation

↓

Insight

↓

Method

↓

Benefit

---

## Stage 5

Method

逐模块介绍。

对于每一个模块：

输出：

### Module Name

Purpose

Input

Output

Algorithm

Complexity

Advantages

Limitations

如果论文包含：

Transformer

Attention

CUDA Kernel

Scheduling

Compiler

Graph

Optimization

分别解释。

---

## Stage 6

Formula Explanation

每一个核心公式都必须解释：

包括：

### Formula

变量解释

公式作用

为什么成立

为什么这样设计

相比已有公式有什么区别

训练时作用

推理时作用

复杂度影响

如果有：

Loss

Attention

Normalization

Kernel

Softmax

Sampling

全部解释。

---

## Stage 7

Algorithm

如果论文提供：

Algorithm 1

Pseudo Code

Workflow

输出：

自然语言版本。

再输出：

伪代码版本。

最后：

时间复杂度

空间复杂度

瓶颈分析

---

## Stage 8

Figures

逐图解释：

Figure 1

Figure 2

...

格式：

Purpose

Input

Output

Data Flow

Key Observation

---

## Stage 9

Experiments

回答：

为什么实验可以证明论文？

包括：

Dataset

Baseline

Metric

Ablation

Efficiency

Memory

Training Cost

Inference Cost

最后：

作者真正证明了什么？

---

## Stage 10

Contribution

总结：

Top 3 Contributions

按重要性排序。

不要超过三条。

---

## Stage 11

Limitations

必须指出：

论文自己提到的问题。

以及：

Reviewer Perspective：

可能存在：

- 假设太强
- 泛化不足
- 实验不充分
- 对比不公平
- 工程成本高
- 理论证明不足
- 可扩展性问题

明确区分：

Paper says

Reviewer thinks

---

## Stage 12

Related Work

构建表格：

Method

Core Idea

Difference

Advantages

Weakness

与本文关系

---

## Stage 13

Reproducibility

判断：

是否容易复现？

检查：

- Code
- Dataset
- Hyperparameter
- Random Seed
- Training Details
- Hardware

最后给：

⭐⭐⭐⭐☆

评分。

---

## Stage 14

Reading Notes

输出：

最值得记住的：

10 条知识点。

每条一句话。

---

## Stage 15

Interview Perspective

如果面试官问：

"讲讲这篇论文。"

应该如何回答？

输出：

2 分钟版本

5 分钟版本

---

## Stage 16

Engineering Perspective

如果准备：

CUDA

AI Infra

LLM

Compiler

Kernel

Inference Engine

需要关注：

哪些工程实现？

哪些地方值得借鉴？

哪些地方可以优化？

---

## Stage 17

Future Work

作者未来方向。

以及：

你认为还能继续研究什么？

---

# Output Requirements

最终输出固定章节：

1. Metadata
2. Summary
3. Background
4. Core Idea
5. Method
6. Formula
7. Algorithm
8. Figures
9. Experiments
10. Contributions
11. Limitations
12. Related Work
13. Reproducibility
14. Reading Notes
15. Interview Version
16. Engineering Insights
17. Future Work

---

# Style

要求：

- Markdown
- 清晰标题
- 使用表格
- 使用列表
- 使用 Mermaid（适合时）
- 数学公式使用 LaTeX
- 避免长段落
- 每节最后给一句 Summary

---

# SVG 配图规范

论文精读笔记必须为关键内容补充 SVG 图片进行说明（架构图、数据流、流水线、对比示意等），不要只用文字。

要求：

- **存放位置**：`aiinfra/paper/images/`（所有论文笔记共享），引用格式 `![中文alt](images/xxx.svg)`
- **命名**：全小写 + 下划线，语义化（如 `transformer_encoder_decoder.svg`）
- **数量**：每篇笔记 2-5 张，插在小节首行
- **alt 文本用中文**
- **风格**：统一为手绘 sketch 风（Excalidraw-like），与全站教程/题解保持一致，细则参照 [`../topics/SKILL.md`](../topics/SKILL.md) 的「图片」规范：
  - 线条：手绘不均匀、略带抖动
  - 配色：极简，不超过 3-4 种柔和颜色
  - 字体：英文用 Comic Sans MS，CJK 用楷体（Kaiti SC）
  - 滤镜：`feTurbulence` 抖动 + `feDisplacementMap`

---

# Special Rules

对于 AI / CUDA / LLM / Compiler / 系统论文：

额外分析：

- Kernel Design
- Memory Layout
- Parallelism
- Warp Scheduling
- Cache Behavior
- Compute vs Memory Bound
- Roofline Analysis（如果适用）
- Tensor Core Usage
- Quantization Strategy
- KV Cache
- Operator Fusion
- Tiling Strategy
- Latency vs Throughput Trade-off
- Hardware Awareness

如果论文涉及 CUDA Kernel，请尽可能解释：

- Grid
- Block
- Warp
- Shared Memory
- Register
- Occupancy
- Memory Coalescing
- Synchronization
- Pipeline

并分析其优化思路是否具有可迁移性。
