# Week 4：FlashAttention 深挖

> 核心目标：从算法原理到 CUDA 实现完整掌握 FlashAttention，理解 IO 优化的核心思想，能在 Mini 引擎中替换标准 Attention

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ------------| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 3 学习，掌握 Softmax/LayerNorm/标准 Attention Kernel、arithmetic intensity 算子分类、端到端 Profiling　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| 本周产出　 | FlashAttention 论文精读笔记、Online Softmax 完整推导、完整 Forward Kernel（flash_attention_v2.cu）、官方源码分析报告、FlashAttention-2 差异总结、Mini 引擎 Attention 替换版、性能对比报告、IO 优化方法论 checklist |
| 周日里程碑 | 手写 FlashAttention Forward Kernel 在 N=4096,d=64 时与标准 Attention 误差 < 1e-3 且加速 2x+，集成到 Mini 引擎端到端正确，能白板推导 online softmax 三公式　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |

---

## 🧭 本周学习地图

```
Day 1: FlashAttention 论文精读 → Online Softmax 三公式推导 + IO 复杂度对比
  ↓
Day 2: 手写完整 Forward Kernel → batch/multi-head + shared memory tiling
  ↓
Day 3: 官方 CUDA 源码分析 → flash_fwd_kernel.h / 分块策略 / warp 分配
  ↓
Day 4: FlashAttention-2 论文 → 减少 non-matmul FLOPs / work partitioning
  ↓
Day 5: 项目推进 → Mini 引擎集成 FlashAttention，替换标准 Attention
  ↓
Day 6: 性能对比 → 标准 vs 手写 vs 官方 / 不同 N/B/H 扫描 / HBM 验证
  ↓
Day 7: IO 优化方法论总结 → 提炼通用策略 + 面试复盘 + GitHub 整理
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | FlashAttention 论文精读与 Online Softmax 完整推导 | [day1/](day1/README.md) |
| Day 2 | 手写完整 FlashAttention Forward Kernel | [day2/](day2/README.md) |
| Day 3 | FlashAttention 官方 CUDA 源码分析 | [day3/](day3/README.md) |
| Day 4 | FlashAttention-2 论文与源码差异 | [day4/](day4/README.md) |
| Day 5 | 算子接入 Mini 引擎 —— FlashAttention 集成 | [day5/](day5/README.md) |
| Day 6 | 性能对比分析 —— 标准 vs 手写 vs 官方 | [day6/](day6/README.md) |
| Day 7 | IO 优化方法论总结与 Week 4 收官 | [day7/](day7/README.md) |
