# Week 3：Transformer 执行本质与算子手写

> 核心目标：从 GPU 视角理解 Transformer 推理执行流程，手写 Softmax/LayerNorm/标准 Attention Kernel，完成算子 IO 分析与端到端 Profiling

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 2 学习，掌握 Warp Shuffle、Register Blocking GEMM、CUDA Streams、FlashAttention 简化版 Forward Kernel　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| 本周产出　 | Transformer forward 时间线、Softmax Kernel（warp shuffle reduce）、LayerNorm Kernel（两级 reduce）、标准 Attention Forward Kernel（含 IO 量化）、端到端 Profiling 报告、Transformer 算子分类表 |
| 周日里程碑 | 手写 Softmax/LayerNorm/标准 Attention 三个 memory-bound 算子，HBM 读写量计算与 ncu 实测一致（误差 < 15%），能用 arithmetic intensity 对算子分类　　　　　　　　　　　　　　　　　　　　　　　  |

---

## 🧭 本周学习地图

```
Day 1: Transformer 推理流程 → Prefill vs Decode + torch.profiler 时间线
  ↓
Day 2: Softmax + LayerNorm Kernel → safe softmax + 两级 reduce + warp shuffle
  ↓
Day 3: 源码分析 → PyTorch ATen / FasterTransformer 的优化手法
  ↓
Day 4: Attention IO 分析 → 标准 Attention HBM 读写量 + O(N²) 量化
  ↓
Day 5: 项目推进 → 算子接入 Mini 引擎 + 端到端正确性
  ↓
Day 6: 端到端 Profiling → 定位 memory-bound 算子 + fusion 机会
  ↓
Day 7: 算子分类 → arithmetic intensity 分类表 + 优化方向总结
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | Trace Transformer 推理流程 | [day1/](day1/README.md) |
| Day 2 | 手写 Softmax 与 LayerNorm Kernel | [day2/](day2/README.md) |
| Day 3 | 源码分析 —— PyTorch / FasterTransformer | [day3/](day3/README.md) |
| Day 4 | Attention IO 分析 | [day4/](day4/README.md) |
| Day 5 | 算子接入 Mini 引擎 | [day5/](day5/README.md) |
| Day 6 | 端到端 Profiling 与 Kernel Fusion | [day6/](day6/README.md) |
| Day 7 | Transformer 算子分类与 Week 3 总结 | [day7/](day7/README.md) |
