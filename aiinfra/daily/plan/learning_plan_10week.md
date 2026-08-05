# AI Infra 10 周学习计划（v2 重组版）

> 制定日期：2026-08-04
> 版本：v2 —— 基于 `weekly_structure_review_and_replan.md` 周级结构重组方案
> 原则：**每周单一主题**（标题不含"+"）、**大专题连续多天不跨周切散**、**前置依赖严格单向**、**统一 day1→day7 节奏模板**
> 节奏模板：Day1-2 理论+基础kernel → Day3-4 进阶实现/源码 → Day5 项目推进 → Day6 Profiling → Day7 复盘+手撕+面试Q&A

---

## 总览

| 周 | 主题（单一） | 核心产出 |
|----|------------|---------|
| Week 1 | GPU 执行模型与内存基础 | deviceQuery 实测、Occupancy 计算、Bank Conflict 分析、ncu/nsys 入门 |
| Week 2 | CUDA Kernel 优化方法论 | Warp Shuffle、Register Blocking、GEMM 七层路径、CUDA Streams、限时手撕 |
| Week 3 | Tensor Core 与 CUTLASS | WMMA GEMM、mma.sync/ldmatrix、CUTLASS 源码、CuTe 布局、混合精度 |
| Week 4 | Transformer 算子手写 + Triton | Softmax/LayerNorm kernel、GEMM Backward、Triton 三方 benchmark |
| Week 5 | FlashAttention 全专题 | FA 简化→论文→Forward→Backward→FA-2/3→官方源码→性能对比→IO 方法论 |
| Week 6 | 推理系统基础与 KV Cache | Prefill/Decode、KV Cache(GQA/MQA/MLA)、vLLM、PagedAttention、FlashDecoding |
| Week 7 | Batching 与调度 | Dynamic/Continuous Batching、vLLM Scheduler、Chunked Prefill、PD 分离 |
| Week 8 | 推理加速技术 | 量化(W8A16/FP8)、投机解码、CUDA Graph、采样 kernel |
| Week 9 | 分布式并行与多硬件 | TP/PP/DP、NCCL、通信重叠、Ring Attention、MoE+EP、Ascend 对比 |
| Week 10 | 项目整合与面试冲刺 | Mini 引擎真整合、项目文档、面试题库、Mock、诊断剧本、能力地图 |

---

## Week 1：GPU 执行模型与内存基础（不变）

| Day | 主题 | 类型 |
|-----|------|------|
| 1 | GPU 执行模型基础（SM/Warp/Thread） | 理论+基础kernel |
| 2 | Occupancy 与资源约束 | 理论+基础kernel |
| 3 | 认识你的 GPU —— deviceQuery 与 Occupancy 计算 | 进阶实现 |
| 4 | Memory Hierarchy 深入 | 进阶实现 |
| 5 | Bank Conflict 分析与实践 | 项目推进 |
| 6 | Nsight Profiling 实战（ncu + nsys） | Profiling |
| 7 | 总结与复盘 | 复盘 |

---

## Week 2：CUDA Kernel 优化方法论

> 来源：原 W2 d1-d5,d7（d6 WMMA 移至新 W3）
> 变化：GEMM 七层从 1 天扩为 2 天（d3-d4）；CUDA Streams 移至 d5；Nsight 移至 d6

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | Warp Shuffle 原语与 Warp/Block Reduce | 理论+基础kernel | W2d1 |
| 2 | Register Blocking 与 2D Tiling | 理论+基础kernel | W2d2 |
| 3 | float4 向量化 + GEMM 七层路径（前四层） | 进阶实现 | W2d5（拆分） |
| 4 | GEMM 七层路径（后三层）+ cuBLAS 对比 | 进阶实现 | W2d5（拆分）+ 待开发 |
| 5 | CUDA Streams 与异步执行 | 项目推进 | W2d3 |
| 6 | Nsight Compute 性能分析 | Profiling | W2d4 |
| 7 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | 复盘+手撕 | W2d7 |

---

## Week 3：Tensor Core 与 CUTLASS

> 来源：原 W2d6（WMMA）+ 原 W3d2（CUTLASS）
> 变化：WMMA 与 CUTLASS 合并成单周；新增 mma.sync/ldmatrix/CuTe 深度天

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | Tensor Core 架构 + WMMA fragment 基础 | 理论+基础kernel | W2d6 |
| 2 | 手写 WMMA GEMM 与 cuBLAS 性能对比 | 进阶实现 | 待开发 |
| 3 | mma.sync 指令与 ldmatrix —— Tensor Core 底层编程 | 进阶实现 | 待开发 |
| 4 | CUTLASS 源码分析 + CuTe 概念铺垫 | 进阶实现/源码 | W3d2 |
| 5 | 项目推进 —— WMMA GEMM 接入 Benchmark 与 Double Buffering | 项目推进 | 待开发 |
| 6 | Profiling —— Tensor Core 利用率与 WMMA vs FMA 对比 | Profiling | 待开发 |
| 7 | 复盘与手撕 —— Tensor Core/CUTLASS 面试要点 | 复盘+手撕 | 待开发 |

---

## Week 4：Transformer 算子手写 + Triton

> 来源：原 W3 d3,d4,d5,d7
> 变化：Softmax/LayerNorm/Triton/手撕合并成单周；Triton 扩为 2 天

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | Trace Transformer 推理流程（Prefill/Decode） | 理论+基础kernel | W3d3 |
| 2 | 手写 Softmax 与 LayerNorm Kernel | 理论+基础kernel | W3d4 |
| 3 | LayerNorm 优化与 GEMM Backward 数据流 | 进阶实现 | 待开发 |
| 4 | Triton 语言专题 —— 用 Triton 重写 Softmax/GEMM/FA | 进阶实现 | W3d5 |
| 5 | 项目推进 —— Triton 三方 Benchmark 与 Autotune | 项目推进 | 待开发 |
| 6 | Profiling —— Triton vs CUDA vs PyTorch 性能对比 | Profiling | 待开发 |
| 7 | Transformer 算子分类与总结 | 复盘 | W3d7 |

---

## Week 5：FlashAttention 全专题（核心周）

> 来源：原 W3d1,d6 + 原 W4 d4,d5,d6,d7 + 原 W5 d1,d2,d3,d4
> 变化：**核心重组** —— FA 从跨 3 周碎片化收敛为单周 7 天贯通

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | FA CUDA 实现（简化版）+ Attention IO 分析 | 理论+基础kernel | W3d1 + W3d6(supp) |
| 2 | FA 论文精读与 Online Softmax 推导 | 理论+基础kernel | W4d4 |
| 3 | 手写完整 FA Forward Kernel | 进阶实现 | W4d5 |
| 4 | FA Backward 与 GEMM Backward | 进阶实现 | W4d6 |
| 5 | 性能对比 + FA 接入 Mini 引擎（C++ Extension） | 项目推进 | W5d3 + W4d2(supp) + W5d2(supp) |
| 6 | FA-2 论文与源码差异 + 官方源码 + IO 方法论 | 进阶实现/源码 | W5d1 + W4d7(supp) + W5d4(supp) |
| 7 | 复盘与手撕 —— FA 限时手写与面试 Q&A | 复盘+手撕 | 待开发 |

---

## Week 6：推理系统基础与 KV Cache

> 来源：原 W5 d5,d6,d7 + 原 W6 d1,d2,d3,d7
> 变化：推理基础与推理核心合并；量化移至 W8；Dynamic Batching 移至 W7

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | 推理流程 —— Prefill vs Decode | 理论+基础kernel | W5d5 |
| 2 | 实现 KV Cache（含 GQA/MQA/MLA 变体） | 理论+基础kernel | W5d6 |
| 3 | vLLM 整体架构分析 | 进阶实现/源码 | W5d7 |
| 4 | vLLM Worker 与 PagedAttention | 进阶实现 | W6d1 |
| 5 | 项目推进 —— Mini 推理引擎 v0 | 项目推进 | W6d3 |
| 6 | FlashDecoding —— Decode 阶段并行度突破 | Profiling | W6d2 |
| 7 | 推理系统核心问题总结 | 复盘 | W6d7 |

---

## Week 7：Batching 与调度

> 来源：原 W6d6（Dynamic Batching）+ 原 W7 全
> 变化：Dynamic Batching 并入 d1 与 Continuous Batching 合讲；其余不变

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | Continuous Batching（含 Dynamic Batching） | 理论+基础kernel | W7d1 + W6d6(supp) |
| 2 | vLLM Scheduler 源码分析 | 进阶实现/源码 | W7d2 |
| 3 | TensorRT-LLM / LightLLM / SGLang 调度对比 | 进阶实现 | W7d3 |
| 4 | Chunked Prefill 与 Prefix Caching 实操 | 进阶实现 | W7d4 |
| 5 | Mini 推理引擎 v1（多请求并发） | 项目推进 | W7d5 |
| 6 | Prefill/Decode 分离推理（PD Disaggregated） | Profiling | W7d6 |
| 7 | 调度优化策略总结 | 复盘 | W7d7 |

---

## Week 8：推理加速技术

> 来源：原 W6d5（量化）+ 原 W8d4（投机解码）+ 原 W9d4（CUDA Graph）
> 变化：量化/投机解码/CUDA Graph/采样合并成单周

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | 量化推理专题 —— W8A16/INT8 KV/FP8 | 理论+基础kernel | W6d5 |
| 2 | FP8 量化深入 —— E4M3/E5M2 kernel 与 GPTQ vs AWQ 对比 | 进阶实现 | 待开发 |
| 3 | SGLang / 投机解码 | 进阶实现 | W8d4 |
| 4 | CUDA Graph 实操 —— 消除 Launch Overhead | 进阶实现 | W9d4 |
| 5 | 项目推进 —— 量化/投机解码/CUDA Graph 接入 Mini 引擎 | 项目推进 | 待开发 |
| 6 | Profiling —— 量化前后精度性能对比与 CUDA Graph Launch Gap | Profiling | 待开发 |
| 7 | 复盘与面试 Q&A —— 量化/投机解码/CUDA Graph/采样 | 复盘+手撕 | 待开发 |

---

## Week 9：分布式并行与多硬件

> 来源：原 W8d5（TP/PP/DP）+ 原 W8d6（Ring Attention）+ 原 W9d5（MoE+EP）+ 原 W10d2（Ascend）
> 变化：分布式从 2 天扩为 5 天；MoE+EP 与 EP 前置打通；Ascend 从面试周移来

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | 分布式推理 —— TP/PP/DP 与通信计算重叠 | 理论+基础kernel | W8d5 |
| 2 | Pipeline Parallelism 与 DP —— 1F1B/bubble ratio/数据并行 | 进阶实现 | 待开发 |
| 3 | NCCL Collectives —— all-reduce/all-gather/reduce-scatter 通信量 | 进阶实现 | 待开发 |
| 4 | 通信计算重叠 —— 双 Stream + CUDA Graph Overlap | 进阶实现 | 待开发 |
| 5 | MoE + EP 并行专题 | 项目推进 | W9d5 + W8d6(supp) |
| 6 | 多硬件对比：NVIDIA CUDA vs Ascend CANN | Profiling | W10d2 |
| 7 | 复盘与面试 Q&A —— 分布式/MoE/多硬件 | 复盘 | 待开发 |

---

## Week 10：项目整合与面试冲刺

> 来源：原 W8 d1,d2,d3,d7 + 原 W9 d1,d2,d3,d6,d7 + 原 W10 d1,d3,d4,d5,d6,d7
> 变化：项目整合内容合并到本周前半；面试冲刺集中后半

| Day | 主题 | 类型 | 原编号 |
|-----|------|------|--------|
| 1 | 整合全部自定义 Kernel | 项目推进 | W9d1 + W8d2(supp) + W8d3(supp) |
| 2 | 系统联调（六步分层验证） | 项目推进 | W9d2 + W9d3(supp) + W4d3(supp) + W8d1(supp) |
| 3 | 项目文档完善（README）+ 架构图 | 项目推进 | W9d6 + W9d7(supp) + W8d7(supp) |
| 4 | 高频面试题基础篇 | 复盘+面试 | W10d1 + W10d3(supp) |
| 5 | Mock 面试 | 复盘+面试 | W10d4 |
| 6 | 诊断流程实战剧本 + 手撕限时清单 | 复盘+手撕 | W10d6 |
| 7 | 最终复盘 —— 10 周能力地图 + 查漏补缺 | 复盘 | W10d7 + W10d5(supp) |

---

## 迁移说明

### 重组原则

1. **每周一个主题**，标题不含"+"
2. **大专题给连续多天**（FA/Tensor Core/分布式/MoE），不跨周切散
3. **前置依赖严格单向**，消除倒置
4. **统一 day1→day7 节奏模板**，Day7 真正做复盘+手撕
5. **保留所有已落地的市场整改内容**（GQA/MLA、PD 分离、MoE、FP8、诊断剧本、Triton benchmark、SGLang 对比等），只重排位置

### 与 v1 计划的对照

| v1 周 | v1 天数 | v2 归属 |
|---------|--------|--------|
| W1 (7天) | 7 | → v2 W1（不变） |
| W2 (7天) | d1-d5,d7 → v2 W2(7)；d6 → v2 W3d1 | |
| W3 (7天) | d1 → v2 W5d1；d2 → v2 W3d4；d3 → v2 W4d1；d4 → v2 W4d2；d5 → v2 W4d4；d6 → v2 W5 supp；d7 → v2 W4d7 | |
| W4 (7天) | d1 → v2 W10 supp；d2 → v2 W5 supp；d3 → v2 W10 supp；d4 → v2 W5d2；d5 → v2 W5d3；d6 → v2 W5d4；d7 → v2 W5 supp | |
| W5 (7天) | d1 → v2 W5d6；d2 → v2 W5 supp；d3 → v2 W5d5；d4 → v2 W5 supp；d5 → v2 W6d1；d6 → v2 W6d2；d7 → v2 W6d3 | |
| W6 (7天) | d1 → v2 W6d4；d2 → v2 W6d6；d3 → v2 W6d5；d4 → v2 W10 supp；d5 → v2 W8d1；d6 → v2 W7 supp；d7 → v2 W6d7 | |
| W7 (7天) | → v2 W7（不变，d1 合入 W6d6 supp） | |
| W8 (7天) | d1-d3 → v2 W10 supp；d4 → v2 W8d3；d5 → v2 W9d1；d6 → v2 W9 supp；d7 → v2 W10 supp | |
| W9 (7天) | d1 → v2 W10d1；d2 → v2 W10d2；d3 → v2 W10 supp；d4 → v2 W8d4；d5 → v2 W9d5；d6 → v2 W10d3；d7 → v2 W10 supp | |
| W10 (7天) | d1 → v2 W10d4；d2 → v2 W9d6；d3 → v2 W10 supp；d4 → v2 W10d5；d5 → v2 W10 supp；d6 → v2 W10d6；d7 → v2 W10d7 | |

### 待开发内容

以下天的内容为新增，需后续开发（已创建占位 README）：

| 周 | 天 | 主题 |
|----|-----|------|
| W2 | d4 | GEMM 优化续篇 —— 后三层路径与 cuBLAS 对比 |
| W3 | d2 | 手写 WMMA GEMM 与 cuBLAS 性能对比 |
| W3 | d3 | mma.sync 指令与 ldmatrix |
| W3 | d5 | 项目推进 —— WMMA GEMM 接入 Benchmark |
| W3 | d6 | Profiling —— Tensor Core 利用率 |
| W3 | d7 | 复盘与手撕 —— Tensor Core/CUTLASS |
| W4 | d3 | LayerNorm 优化与 GEMM Backward |
| W4 | d5 | 项目推进 —— Triton 三方 Benchmark |
| W4 | d6 | Profiling —— Triton vs CUDA vs PyTorch |
| W5 | d7 | 复盘与手撕 —— FA 限时手写 |
| W8 | d2 | FP8 量化深入 |
| W8 | d5 | 项目推进 —— 加速技术接入 Mini 引擎 |
| W8 | d6 | Profiling —— 量化/CUDA Graph |
| W8 | d7 | 复盘与面试 Q&A |
| W9 | d2 | Pipeline Parallelism 与 DP |
| W9 | d3 | NCCL Collectives |
| W9 | d4 | 通信计算重叠 |
| W9 | d7 | 复盘与面试 Q&A |
