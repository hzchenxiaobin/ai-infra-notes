# AI Infra 10 周学习计划（工程实战版）

> 适合人群：具备 CUDA / 算子优化基础，希望转向 AI Infra（推理系统 / 分布式 / 内核优化）  
> 学习强度：每日 3～5 小时  
> 核心目标：从"会写 kernel"进阶到"能做系统优化"

---

## 🎯 课程概览

本仓库将 AI Infra 核心能力拆解为 **10 周** 的递进式学习路线，每周聚焦一个主题，从 GPU 执行本质、Kernel 优化、Transformer 算子、FlashAttention，一直延伸到推理系统、Batching / 调度、分布式并行、系统整合与面试冲刺。每天的学习材料都按 `weekN/dayM/` 组织，包含：

- `README.md`：当天教程（理论 + Coding 任务 + 面试要点）
- `kernels/`：可直接编译运行的 `.cu` / `.py` 示例
- `exercise/`：练习题与验证程序
- `notes/`：延伸阅读与笔记

完成 10 周后，你将拥有一个可运行的 **Mini 推理引擎**、一整套自定义 CUDA Kernel、以及完整的性能分析与面试知识体系。

---

## 🧭 总体节奏（每周循环）

| 天数 | 类型 | 核心动作 |
|------|------|----------|
| Day 1-2 | 🔬 核心学习 + Coding | 理论学习 + 动手写 kernel / 基础实现 |
| Day 3-4 | 📖 源码分析 | 阅读开源代码（cuBLAS / vLLM / FlashAttention 等） |
| Day 5 | 🛠 项目推进 | 将本周所学整合到 Mini 推理引擎或 benchmark 中 |
| Day 6 | 📊 总结 + Profiler + Debug | 性能分析、瓶颈定位、文档整理 |
| Day 7 | 🧘 弹性/缓冲 | 补进度、复盘或深入某个未完成的点 |

---

## 🚀 10 周学习路线

| 周 | 主题 | 核心目标 | 关键产出 | 入口 |
|----|------|----------|----------|------|
| **Week 1** | GPU 执行模型与内存基础 | 建立 GPU 性能直觉 —— **性能 = Memory + 并行度** | 7 个 CUDA kernel、3+ Nsight Compute 报告、GPU 架构与性能笔记 | [进入 Week 1](week1/README.md) |
| **Week 2** | CUDA Kernel 优化 + Tensor Core | 掌握 Warp Shuffle、Register Blocking、GEMM 七层优化、WMMA、CUTLASS 源码 | GEMM 优化达 cuBLAS 60%+、WMMA 实测、CUTLASS 分析笔记 | [进入 Week 2](week2/README.md) |
| **Week 3** | 手撕复盘 + Transformer 算子 | 限时手写 kernel、Softmax/LayerNorm、Triton 三方 benchmark | 手撕留档、Softmax/LayerNorm kernel、Triton 性能对比表 | [进入 Week 3](week3/README.md) |
| **Week 4** | 算子集成 + FlashAttention（前半） | C++ Extension 集成、FA 论文精读、Forward/Backward kernel | FA Forward kernel、FA Backward 推导、profiling 报告 | [进入 Week 4](week4/README.md) |
| **Week 5** | FlashAttention（后半）+ 推理系统基础 | FA-2/性能对比/IO 方法论、Prefill/Decode、KV Cache、vLLM 架构 | FA 性能对比、KV Cache kernel、vLLM 架构分析 | [进入 Week 5](week5/README.md) |
| **Week 6** | 推理系统核心 | PagedAttention、FlashDecoding、Mini 引擎 v0、量化、Dynamic Batching | PagedAttention kernel、Mini 引擎 v0、量化 kernel | [进入 Week 6](week6/README.md) |
| **Week 7** | Batching 与调度 | Continuous Batching、vLLM Scheduler、Chunked Prefill、PD 分离、Mini 引擎 v1 | Mini 引擎 v1、PD 分离模拟器、Scheduler 复刻 | [进入 Week 7](week7/README.md) |
| **Week 8** | 系统整合与分布式并行 | 多请求并发、完整调度器、投机解码、TP/PP/DP、Ring Attention、MoE+EP | 完整调度器、MoE 路由模拟器、分布式推理分析 | [进入 Week 8](week8/README.md) |
| **Week 9** | 系统联调与项目打磨 | Kernel 集成、六步联调、CUDA Graph、全链路 Profiling、README/架构图 | 全链路引擎、CUDA Graph 集成、项目文档 | [进入 Week 9](week9/README.md) |
| **Week 10** | 面试冲刺 | 面试题库、Ascend 对比、Mock 面试、诊断剧本、最终复盘 | 21 道进阶题自测、Mock 面试记录、诊断剧本、能力地图 | [进入 Week 10](week10/README.md) |

> 更详细的每日任务与 Checklist 可参考 `aiinfra/daily/plan/learning_plan_10week.md`。

---

## 🗺️ 学习路径建议

1. 在本页了解整体节奏与每周目标。
2. 从 [Week 1](week1/README.md) 开始，按 Day 1 → Day 7 推进；每天先读教程，再跑 `kernels/` 中的代码。
3. 每个 kernel 配套 Nsight Profiling 任务，记录指标并对比理论预期。
4. 从 Week 4 起，将手写算子逐步接入 Mini 推理引擎，关注端到端正确性与性能。
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
