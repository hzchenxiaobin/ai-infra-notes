# Week 4：FlashAttention 深挖

> 核心目标：从算法原理到 CUDA 实现完整掌握 FlashAttention，理解 IO 优化的核心思想，能在 Mini 引擎中替换标准 Attention

| 项目 | 说明 |
|------|------|
| **整体目标** | 深入理解 FlashAttention 的 tiling + online softmax 原理，手写支持 batch/multi-head 的完整 Forward Kernel，集成到 Mini 推理引擎，建立 IO 优化的系统方法论 |
| **核心产出** | ① FlashAttention 论文精读笔记 ② Online Softmax 完整推导 ③ 完整 Forward Kernel（`flash_attention_v2.cu`）④ 官方源码分析报告 ⑤ FlashAttention-2 差异总结 ⑥ Mini 引擎 Attention 替换版 ⑦ 性能对比报告 ⑧ IO 优化方法论 checklist |
| **验收标准** | ① 能白板推导 online softmax 三公式 ② 手写 Kernel 在 N=4096, d=64 时与标准 Attention 误差 < 1e-3 且加速 2x+ ③ 能解释 FlashAttention-2 比 FA1 快的 3 个原因 ④ 集成到 Mini 引擎后端到端正确 ⑤ 能用 ncu 验证 HBM 访问量随 N 线性增长 |
| **时间投入** | 工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h |

## 本周知识图谱

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

## 进入每日学习
