# Week 4：算子集成 + FlashAttention（前半）

> 核心目标：掌握 C++ Extension 集成、端到端 Profiling、FlashAttention 论文精读与 Forward/Backward kernel 实现

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| --------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 3，掌握 Transformer 算子手写、Triton 基础、Attention IO 分析　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　|
| 本周产出　 | C++ Extension 集成代码、ncu/nsys profiling 报告、FlashAttention Forward kernel、FA Backward 推导　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | 手写 FA Forward kernel 正确性 PASS，理解 online softmax 完整推导与 FA Backward 数据流　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: 算子接入 Mini 引擎（C++ Extension）
        ↓
Day 2: 端到端 Profiling 与 Kernel Fusion
        ↓
Day 3: Transformer 算子分类与总结
        ↓
Day 4: FlashAttention 论文精读与 Online Softmax 推导
        ↓
Day 5: 手写完整 FlashAttention Forward Kernel
        ↓
Day 6: FlashAttention Backward 与 GEMM Backward
        ↓
Day 7: FlashAttention 官方 CUDA 源码分析

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | 算子接入 Mini 引擎（C++ Extension） | [day1/](day1/README.md) |
| Day 2 | 端到端 Profiling 与 Kernel Fusion | [day2/](day2/README.md) |
| Day 3 | Transformer 算子分类与总结 | [day3/](day3/README.md) |
| Day 4 | FlashAttention 论文精读与 Online Softmax 推导 | [day4/](day4/README.md) |
| Day 5 | 手写完整 FlashAttention Forward Kernel | [day5/](day5/README.md) |
| Day 6 | FlashAttention Backward 与 GEMM Backward | [day6/](day6/README.md) |
| Day 7 | FlashAttention 官方 CUDA 源码分析 | [day7/](day7/README.md) |
