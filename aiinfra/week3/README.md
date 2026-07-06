# Week 3：Transformer 执行本质与算子手写

> 核心目标：从 GPU 视角理解 Transformer 推理执行流程，手写 Softmax/LayerNorm/标准 Attention Kernel，完成算子 IO 分析与端到端 Profiling

| 项目 | 说明 |
|------|------|
| **整体目标** | 理解 Transformer 推理的 Prefill/Decode 两阶段执行特征，手写 Softmax/LayerNorm/标准 Attention 三个 memory-bound 算子，用 arithmetic intensity 对算子分类 |
| **核心产出** | ① Transformer forward 的 torch.profiler 时间线 ② Softmax Kernel（warp shuffle reduce）③ LayerNorm Kernel（两级 reduce）④ 标准 Attention Forward Kernel（含 IO 量化）⑤ 端到端 Profiling 报告 ⑥ Transformer 算子分类表 |
| **验收标准** | ① Softmax/LayerNorm 与 PyTorch 误差 < 1e-5 ② 手动计算的标准 Attention HBM 读写量与 ncu 实测一致（误差 < 15%）③ 能画出 Transformer forward 算子时间线 ④ 能用 arithmetic intensity 判断每个算子是 compute-bound 还是 memory-bound |
| **时间投入** | 工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h |

## 本周知识图谱

```
Day 15: Transformer 推理流程 → Prefill vs Decode + torch.profiler 时间线
  ↓
Day 16: Softmax + LayerNorm Kernel → safe softmax + 两级 reduce + warp shuffle
  ↓
Day 17: 源码分析 → PyTorch ATen / FasterTransformer 的优化手法
  ↓
Day 18: Attention IO 分析 → 标准 Attention HBM 读写量 + O(N²) 量化
  ↓
Day 19: 项目推进 → 算子接入 Mini 引擎 + 端到端正确性
  ↓
Day 20: 端到端 Profiling → 定位 memory-bound 算子 + fusion 机会
  ↓
Day 21: 算子分类 → arithmetic intensity 分类表 + 优化方向总结
```

## 进入每日学习
