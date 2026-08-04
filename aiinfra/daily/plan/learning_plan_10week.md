# AI Infra 10 周学习计划（统一 day1-day7，无补充日）

> 制定日期：2026-08-04
> 基于：原 8 周计划（56 主线日 + 14 补充日 dayNb）重新统筹，扩展为 10 周 × 7 天 = 70 天
> 原则：**每周固定 day1-day7，不再使用 dayNb/dayNc 补充日**；补充日内容按主题归并到主线
> 映射：下表"原编号"列标注内容来源（如 `原W2d6b` 表示原 week2/day6b）

---

## 总览

| 周 | 主题 | 核心产出 |
|----|------|---------|
| Week 1 | GPU 执行模型与内存基础 | deviceQuery 实测、Occupancy 计算、Bank Conflict 分析、ncu/nsys 入门 |
| Week 2 | CUDA Kernel 优化 + Tensor Core | Reduce/GEMM tiling/float4、WMMA、CUTLASS 源码分析 |
| Week 3 | 手撕复盘 + Transformer 算子手写 | 限时手撕 reduce/GEMM、Softmax/LayerNorm kernel、Triton 入门 |
| Week 4 | 算子集成 + FlashAttention（前半） | C++ Extension 集成、FA 论文精读、FA Forward Kernel、FA Backward |
| Week 5 | FlashAttention（后半）+ 推理系统基础 | FA-2/性能对比/IO 方法论、Prefill/Decode、KV Cache、vLLM 架构 |
| Week 6 | 推理系统核心 | PagedAttention、FlashDecoding、Mini 引擎 v0、量化、Dynamic Batching |
| Week 7 | Batching 与调度 | Continuous Batching、vLLM Scheduler、Chunked Prefill、PD 分离、Mini 引擎 v1 |
| Week 8 | 系统整合与分布式并行 | 多请求并发、完整调度器、投机解码、TP/PP/DP、Ring Attention、MoE+EP |
| Week 9 | 系统联调与项目打磨 | Kernel 集成、六步联调、CUDA Graph、全链路 Profiling、README/架构图 |
| Week 10 | 面试冲刺 | 面试题库（基础+进阶）、Ascend 对比、Mock 面试、诊断剧本、最终复盘 |

---

## Week 1：GPU 执行模型与内存基础

> 原 week1，不变

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | GPU 执行模型基础（SM/Warp/Thread） | W1d1 |
| 2 | Occupancy 与资源约束 | W1d2 |
| 3 | 认识你的 GPU —— deviceQuery 与 Occupancy 计算 | W1d3 |
| 4 | Memory Hierarchy 深入 | W1d4 |
| 5 | Bank Conflict 分析与实践 | W1d5 |
| 6 | Nsight Profiling 实战（ncu + nsys） | W1d6 |
| 7 | 总结与复盘 | W1d7 |

---

## Week 2：CUDA Kernel 优化 + Tensor Core

> 原 week2 主线（d1-d4/d6）+ 补充日（d6b WMMA、d4b CUTLASS）；原 d5(FA 简化)、d7(手撕) 移至 Week 3

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | Warp Shuffle 原语与 Warp/Block Reduce | W2d1 |
| 2 | Register Blocking 与 2D Tiling | W2d2 |
| 3 | CUDA Streams 与异步执行 | W2d3 |
| 4 | Nsight Compute 性能分析 | W2d4 |
| 5 | 整合优化到 cuBLAS 70%+（GEMM 七层路径） | W2d6 |
| 6 | Tensor Core 与 WMMA —— 从 FMA 到 Tensor Core | W2d6b |
| 7 | CUTLASS 源码分析 + CuTe 概念铺垫 | W2d4b |

---

## Week 3：手撕复盘 + Transformer 算子手写

> 原 week2 d5/d7（FA 简化 + 手撕）+ 原 week3 d1-d4 + d3b（Triton）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | FlashAttention CUDA 实现（简化版） | W2d5 |
| 2 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | W2d7 |
| 3 | Trace Transformer 推理流程（Prefill/Decode） | W3d1 |
| 4 | 手写 Softmax 与 LayerNorm Kernel | W3d2 |
| 5 | 源码分析 —— PyTorch / FasterTransformer | W3d3 |
| 6 | Triton 语言专题 —— 用 Triton 重写 Softmax/GEMM/FA | W3d3b |
| 7 | Attention IO 分析（4N²+4Nd 口径） | W3d4 |

---

## Week 4：算子集成 + FlashAttention（前半）

> 原 week3 d5-d7（集成/Profiling/总结）+ 原 week4 d1-d3 + d2b（FA Backward）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | 算子接入 Mini 引擎（C++ Extension） | W3d5 |
| 2 | 端到端 Profiling 与 Kernel Fusion | W3d6 |
| 3 | Transformer 算子分类与总结 | W3d7 |
| 4 | FlashAttention 论文精读与 Online Softmax 推导 | W4d1 |
| 5 | 手写完整 FlashAttention Forward Kernel | W4d2 |
| 6 | FlashAttention Backward 与 GEMM Backward | W4d2b |
| 7 | FlashAttention 官方 CUDA 源码分析 | W4d3 |

---

## Week 5：FlashAttention（后半）+ 推理系统基础

> 原 week4 d4-d7（FA-2/集成/对比/总结）+ 原 week5 d1-d3（Prefill/KV/vLLM）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | FlashAttention-2 论文与源码差异 | W4d4 |
| 2 | 算子接入 Mini 引擎 —— FlashAttention 集成 | W4d5 |
| 3 | 性能对比分析 —— 标准 vs 手写 vs 官方 | W4d6 |
| 4 | IO 优化方法论总结与 Week 收官 | W4d7 |
| 5 | 推理流程 —— Prefill vs Decode | W5d1 |
| 6 | 实现 KV Cache（含 GQA/MQA/MLA 变体） | W5d2 |
| 7 | vLLM 整体架构分析 | W5d3 |

---

## Week 6：推理系统核心

> 原 week5 d4-d7 + d4b/d6b（FlashDecoding/量化）+ 原 week6 d1（Dynamic Batching 引入）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | vLLM Worker 与 PagedAttention | W5d4 |
| 2 | FlashDecoding —— Decode 阶段并行度突破 | W5d4b |
| 3 | 项目推进 —— Mini 推理引擎 v0 | W5d5 |
| 4 | 端到端 Profiling | W5d6 |
| 5 | 量化推理专题 —— W8A16/INT8 KV/FP8 | W5d6b |
| 6 | 推理系统核心问题总结 | W5d7 |
| 7 | Dynamic Batching | W6d1 |

---

## Week 7：Batching 与调度

> 原 week6 d2-d7 + d4b/d5b（Chunked Prefill/PD 分离）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | Continuous Batching | W6d2 |
| 2 | vLLM Scheduler 源码分析 | W6d3 |
| 3 | TensorRT-LLM / LightLLM / SGLang 调度对比 | W6d4 |
| 4 | Chunked Prefill 与 Prefix Caching 实操 | W6d4b |
| 5 | Mini 推理引擎 v1（多请求并发） | W6d5 |
| 6 | Prefill/Decode 分离推理（PD Disaggregated） | W6d5b |
| 7 | Latency / Throughput 测试 | W6d6 |

---

## Week 8：系统整合与分布式并行

> 原 week6 d7（调度总结）+ 原 week7 d1-d3 + d3b/d4b/d5b（分布式/Ring Attention/MoE）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | 调度优化策略总结 | W6d7 |
| 2 | 多请求并发支持 | W7d1 |
| 3 | 完整调度器（优先级/超时/抢占） | W7d2 |
| 4 | SGLang / 投机解码 | W7d3 |
| 5 | 分布式推理 —— TP/PP/DP 与通信计算重叠 | W7d3b |
| 6 | Ring Attention —— 长上下文分布式注意力 | W7d4b |
| 7 | MoE + EP 并行专题 | W7d5b |

---

## Week 9：系统联调与项目打磨

> 原 week7 d4-d7 + d6b（CUDA Graph）+ 原 week8 d1-d2（项目文档/架构图）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | 整合全部自定义 Kernel | W7d4 |
| 2 | 系统联调（六步分层验证） | W7d5 |
| 3 | 全链路 Profiling | W7d6 |
| 4 | CUDA Graph 实操 —— 消除 Launch Overhead | W7d6b |
| 5 | 代码重构与文档 | W7d7 |
| 6 | 项目文档完善（README） | W8d1 |
| 7 | 架构图与数据流图 | W8d2 |

---

## Week 10：面试冲刺

> 原 week8 d3-d7 + d3b/d6b（Ascend/诊断剧本）

| Day | 主题 | 原编号 |
|-----|------|--------|
| 1 | 高频面试题基础篇 | W8d3 |
| 2 | 多硬件对比：NVIDIA CUDA vs Ascend CANN | W8d3b |
| 3 | 高频面试题进阶篇（含系统设计题） | W8d4 |
| 4 | Mock 面试 | W8d5 |
| 5 | 查漏补缺 | W8d6 |
| 6 | 诊断流程实战剧本 + 手撕限时清单 | W8d6b |
| 7 | 最终复盘 —— 10 周能力地图 | W8d7 |

---

## 迁移说明

### 内容归并原则

1. **补充日按主题归并**：14 个 dayNb 补充日按主题归入相邻主线周，不再单独编号
2. **边界调整**：
   - 原 W2d5（FA 简化）+ W2d7（手撕）移至 W3，让 W2 聚焦"CUDA 优化 + Tensor Core"
   - 原 W3d5-d7（集成/Profiling/总结）移至 W4，与 FA 前半衔接
   - 原 W4d4-d7（FA 后半）移至 W5，与推理系统基础衔接
   - 原 W5d4b/d6b（FlashDecoding/量化）+ W6d1 归入 W6（推理系统核心）
   - 原 W7 的 4 个补充日（分布式/Ring Attention/MoE/CUDA Graph）拆分：3 个入 W8（分布式并行），1 个入 W9（CUDA Graph）
3. **周主题重命名**：部分周主题调整以反映归并后的内容（如 W3 = "手撕复盘 + Transformer 算子"）

### 与原 8 周计划的对照

| 原 week | 原天数 | 新归属 |
|---------|--------|--------|
| W1 (7天) | 7 | → 新 W1（不变） |
| W2 (9天) | 7主线 + d4b/d6b | → 新 W2 (7) + 新 W3 d1-d2 |
| W3 (8天) | 7主线 + d3b | → 新 W3 d3-d7 + 新 W4 d1-d3 |
| W4 (8天) | 7主线 + d2b | → 新 W4 d4-d7 + 新 W5 d1-d4 |
| W5 (9天) | 7主线 + d4b/d6b | → 新 W5 d5-d7 + 新 W6 d1-d6 |
| W6 (9天) | 7主线 + d4b/d5b | → 新 W6 d7 + 新 W7 d1-d7 |
| W7 (11天) | 7主线 + d3b/d4b/d5b/d6b | → 新 W8 d1-d7 + 新 W9 d1-d5 |
| W8 (9天) | 7主线 + d3b/d6b | → 新 W9 d6-d7 + 新 W10 d1-d7 |

### 目录迁移工作量

本计划为**逻辑重排**，物理目录迁移（`git mv` + 更新所有交叉引用）建议分步执行：
1. 先按上表创建新 weekN/dayM 目录结构
2. `git mv` 移动文件（保留 git 历史）
3. 全仓库批量更新交叉引用（`weekN/dayM/` → 新编号）
4. 更新各 week README 的学习地图与每日材料表
5. 更新 SKILL.md 的 LeetGPU 映射表
6. 运行 `python3 build.py` + `python3 build/lint_md_code.py` 验收

> ⚠️ 目录迁移涉及大量路径变更，建议单独一个 commit 完成，避免与内容修改混淆。
