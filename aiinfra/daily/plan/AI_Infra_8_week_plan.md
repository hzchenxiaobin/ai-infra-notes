# AI Infra 8 周冲刺学习计划（工程实战版）

> 📌 本文件是**极简总览**（164 行），完整版见 [AI_Infra_8_week_plan_detailed.md](AI_Infra_8_week_plan_detailed.md)（883 行，含每日详细任务）。
> 每周概览见各周 [README.md](../week1/README.md)。

> 适合人群：具备 CUDA / 算子优化基础，希望转向 AI Infra（推理系统 /
> 分布式 / 内核优化）\
> 学习强度：每日 3～5 小时\
> 核心目标：从"会写 kernel"进阶到"能做系统优化"

------------------------------------------------------------------------

## 🧭 总体节奏

每周安排：

-   3 天：核心学习 + coding
-   2 天：源码分析
-   1 天：项目推进
-   1 天：总结 + profiler + debug

------------------------------------------------------------------------

## 🚀 Week 1：GPU 执行本质 + Profiling

### 🎯目标

GPU 性能 = memory + 并行度

### 📅 任务安排

-   Day 1-2：GPU 执行模型（warp / SM / occupancy）
-   Day 3-4：Memory hierarchy（shared memory / bank conflict）
-   Day 5：Nsight profiling
-   Day 6：benchmark + 瓶颈分析

------------------------------------------------------------------------

## 🔥 Week 2：GEMM & Kernel 优化

### 🎯目标

所有深度学习本质都是 matmul

### 📅 任务安排

-   Day 1-2：naive GEMM
-   Day 3-4：tiling + shared memory
-   Day 5：性能分析（对比 cuBLAS）
-   Day 6：FlashAttention 论文
-   Day 7：总结 checklist

------------------------------------------------------------------------

## ⚡ Week 3：Transformer 执行本质

### 🎯目标

理解执行而不是模型

### 📅 任务安排

-   trace transformer
-   实现 softmax / layernorm
-   分析 attention IO
-   阅读推理源码

------------------------------------------------------------------------

## 🚀 Week 4：FlashAttention 深挖

### 🎯目标

理解 IO 优化

### 📅 任务安排

-   论文精读
-   实现 block-wise attention
-   性能对比
-   总结优化策略

------------------------------------------------------------------------

## 🧠 Week 5：推理系统

### 🎯目标

进入 AI Infra 核心

### 📅 任务安排

-   推理流程（prefill / decode）
-   实现 KV cache
-   阅读 vLLM
-   mini 推理引擎 v0

------------------------------------------------------------------------

## ⚙️ Week 6：Batching & 调度

### 🎯目标

建立系统感

### 📅 任务安排

-   dynamic batching
-   continuous batching
-   latency / throughput 测试
-   调度优化

------------------------------------------------------------------------

## 🔥 Week 7：系统整合

### 🎯目标

完成 mini AI Infra 系统

### 📦 要求

-   多请求支持
-   batching
-   scheduler
-   latency 测试

------------------------------------------------------------------------

## 🏁 Week 8：项目打磨 + 面试准备

### 🎯目标

转化为面试能力

### 📅 任务安排

-   项目整理（README + 架构图）
-   高频问题准备
-   mock 面试

------------------------------------------------------------------------

## 📦 最终产出

### 项目

-   Mini 推理引擎
-   Attention kernel
-   GEMM 优化

### 能力

-   GPU 性能分析
-   推理系统优化
-   Kernel 优化方法论

------------------------------------------------------------------------

## 🎯 总结

构建完整 AI Infra 能力闭环：

-   Kernel
-   System
-   Profiling
-   Optimization
