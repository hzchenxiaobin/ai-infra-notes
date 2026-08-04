# Week 5：FlashAttention（后半）+ 推理系统基础

> 核心目标：FA-2 改进、性能对比、IO 方法论、Prefill/Decode、KV Cache（含 GQA/MLA）、vLLM 架构

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| ------------------------------------------------------------|
| 前置要求　 | 已完成 Week 4，掌握 C++ Extension、FA Forward/Backward　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　|
| 本周产出　 | FA 性能对比报告、IO 方法论总结、KV Cache kernel、vLLM 架构分析　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | FA 端到端性能对比留档，KV Cache kernel PASS，理解 vLLM 三层架构　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: FlashAttention-2 论文与源码差异
        ↓
Day 2: 算子接入 Mini 引擎 —— FlashAttention 集成
        ↓
Day 3: 性能对比分析 —— 标准 vs 手写 vs 官方
        ↓
Day 4: IO 优化方法论总结与收官
        ↓
Day 5: 推理流程 —— Prefill vs Decode
        ↓
Day 6: 实现 KV Cache（含 GQA/MQA/MLA 变体）
        ↓
Day 7: vLLM 整体架构分析
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | FlashAttention-2 论文与源码差异 | [day1/](day1/README.md) |
| Day 2 | 算子接入 Mini 引擎 —— FlashAttention 集成 | [day2/](day2/README.md) |
| Day 3 | 性能对比分析 —— 标准 vs 手写 vs 官方 | [day3/](day3/README.md) |
| Day 4 | IO 优化方法论总结与收官 | [day4/](day4/README.md) |
| Day 5 | 推理流程 —— Prefill vs Decode | [day5/](day5/README.md) |
| Day 6 | 实现 KV Cache（含 GQA/MQA/MLA 变体） | [day6/](day6/README.md) |
| Day 7 | vLLM 整体架构分析 | [day7/](day7/README.md) |
