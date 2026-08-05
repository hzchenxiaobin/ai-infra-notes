# AI Infra 10 周学习计划（工程实战版）

> 适合人群：具备 CUDA / 算子优化基础，希望转向 AI Infra（推理系统 / 分布式 / 内核优化）  
> 学习强度：每日 3～5 小时  
> 核心目标：从"会写 kernel"进阶到"能做系统优化"

---

## 🎯 课程概览

本仓库将 AI Infra 核心能力拆解为 **10 周** 的递进式学习路线，每周聚焦**一个主题**，从 GPU 执行本质、Kernel 优化、Tensor Core/CUTLASS、Transformer 算子、FlashAttention 全专题，一直延伸到推理系统、Batching / 调度、推理加速技术、分布式并行与面试冲刺。每天的学习材料都按 `weekN/dayM/` 组织，包含：

- `README.md`：当天教程（理论 + Coding 任务 + 面试要点）
- `kernels/`：可直接编译运行的 `.cu` / `.py` 示例
- `exercise/`：练习题与验证程序
- `notes/`：延伸阅读与笔记

完成 10 周后，你将拥有一个可运行的 **Mini 推理引擎**、一整套自定义 CUDA Kernel、以及完整的性能分析与面试知识体系。

---

## 🧭 总体节奏（每周循环）

| 天数 | 类型 | 核心动作 |
|------|------|----------|
| Day 1-2 | 🔬 理论 + 基础 kernel | 概念建模 + 最简实现 |
| Day 3-4 | 📖 进阶实现 / 源码 | 进阶优化或开源源码导读 |
| Day 5 | 🛠 项目推进 | 接入 Mini 引擎或 benchmark |
| Day 6 | 📊 Profiling + 性能分析 | ncu/nsys 实测 + Roofline |
| Day 7 | 🧘 复盘 + 限时手撕 + 面试要点 | 知识地图 / 手撕清单 / 面试 Q&A 收敛 |

---

## 🚀 10 周学习路线

| 周 | 主题 | 核心目标 | 关键产出 | 入口 |
|----|------|----------|----------|------|
| **Week 1** | GPU 执行模型与内存基础 | 建立 GPU 性能直觉 —— **性能 = Memory + 并行度** | 7 个 CUDA kernel、3+ Nsight 报告、GPU 架构与性能笔记 | [进入 Week 1](week1/README.md) |
| **Week 2** | CUDA Kernel 优化方法论 | 掌握 Warp Shuffle、Register Blocking、GEMM 七层路径、CUDA Streams | GEMM 优化达 cuBLAS 60%+、多流 pipeline 实测、限时手撕留档 | [进入 Week 2](week2/README.md) |
| **Week 3** | Tensor Core 与 CUTLASS | 掌握 WMMA/mma.sync、CUTLASS 三级 Tiling、CuTe 布局抽象 | WMMA GEMM kernel、CUTLASS 源码分析、混合精度策略 | [进入 Week 3](week3/README.md) |
| **Week 4** | Transformer 算子手写 + Triton | 手写 Softmax/LayerNorm/GEMM Backward、Triton 三方 benchmark | Softmax/LayerNorm kernel、Triton vs CUDA 性能对比表 | [进入 Week 4](week4/README.md) |
| **Week 5** | FlashAttention 全专题 | 从 FA 简化版到 FA-3 完整贯通：论文/Forward/Backward/官方源码/性能对比 | FA Forward/Backward kernel、性能对比报告、IO 方法论总结 | [进入 Week 5](week5/README.md) |
| **Week 6** | 推理系统基础与 KV Cache | Prefill/Decode、KV Cache（GQA/MQA/MLA）、vLLM、PagedAttention、FlashDecoding | KV Cache kernel、PagedAttention kernel、Mini 引擎 v0 | [进入 Week 6](week6/README.md) |
| **Week 7** | Batching 与调度 | Continuous Batching、vLLM Scheduler、Chunked Prefill、PD 分离、Mini 引擎 v1 | Mini 引擎 v1、PD 分离模拟器、Scheduler 复刻 | [进入 Week 7](week7/README.md) |
| **Week 8** | 推理加速技术 | 量化（W8A16/INT8 KV/FP8）、投机解码、CUDA Graph、采样 kernel | 量化 kernel、投机解码模拟器、CUDA Graph 集成 | [进入 Week 8](week8/README.md) |
| **Week 9** | 分布式并行与多硬件 | TP/PP/DP、NCCL、通信计算重叠、Ring Attention、MoE+EP、Ascend 对比 | TP demo、MoE 路由模拟器、分布式推理分析 | [进入 Week 9](week9/README.md) |
| **Week 10** | 项目整合与面试冲刺 | Mini 引擎真整合、全链路 Profiling、面试题库、Mock 面试、诊断剧本 | 全链路引擎、面试题库、诊断剧本、10 周能力地图 | [进入 Week 10](week10/README.md) |

> 更详细的每日任务与 Checklist 可参考 `aiinfra/daily/plan/learning_plan_10week.md`。

---

## 🗺️ 学习路径建议

1. 在本页了解整体节奏与每周目标。
2. 从 [Week 1](week1/README.md) 开始，按 Day 1 → Day 7 推进；每天先读教程，再跑 `kernels/` 中的代码。
3. 每个 kernel 配套 Nsight Profiling 任务，记录指标并对比理论预期。
4. 从 Week 5 起，将手写算子逐步接入 Mini 推理引擎，关注端到端正确性与性能。
5. 每天完成 LeetGPU 在线题目与 LeetCode 面试题，题解分别归档在 [独立 leetgpu 仓库](https://github.com/hzchenxiaobin/leetgpu) 与 `leetcode/`。
6. 每周末用 Day 7 复盘，整理笔记、补全未完成任务，并更新项目文档。

---

## 📦 最终产出

### 项目

- Mini 推理引擎（支持单请求 → 多请求 → Continuous Batching → PD 分离）
- Attention kernel（标准 Attention + FlashAttention + FlashDecoding）
- GEMM 优化（naive → tiled → register blocking → WMMA）
- MoE 路由 + EP 通信量模拟器

### 能力

- GPU 性能分析（Nsight Compute / Systems）
- 推理系统优化（KV Cache、Batching、调度、量化、PD 分离）
- 分布式并行（TP/PP/DP/EP、Ring Attention、MoE）
- Kernel 优化方法论（memory-bound / compute-bound、IO 优化、fusion）

---

## 📚 更多资源

- [LeetGPU 题解](https://hzchenxiaobin.github.io/leetgpu/) — CUDA 在线挑战题解
- [LeetCode 题解](../leetcode/README.md) — 面试高频算法题解
- [CUTLASS 专题](https://hzchenxiaobin.github.io/ai-infra-notes/cutlass/index.html) — 横向深挖 CUTLASS
- [Triton 专题](https://hzchenxiaobin.github.io/ai-infra-notes/triton/index.html) — 横向深挖 Triton
