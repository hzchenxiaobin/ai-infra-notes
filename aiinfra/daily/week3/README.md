# Week 3：手撕复盘 + Transformer 算子手写

> 核心目标：限时手写 kernel 验证、CUTLASS 进阶、Transformer 推理 Prefill/Decode、Softmax/LayerNorm/Triton 手写

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| ------------------------------------------------------------|
| 前置要求　 | 已完成 Week 2，掌握 GEMM tiling、WMMA、手撕 reduce/GEMM　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　|
| 本周产出　 | 手撕留档、CUTLASS 分析、Softmax/LayerNorm kernel、Triton 三方 benchmark、Attention IO 分析　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | Triton GEMM 追平 cuBLAS，Softmax/LayerNorm kernel 正确性 PASS，算子分类表完成　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: FlashAttention CUDA 实现（简化版）
        ↓
Day 2: CUTLASS 源码分析 + CuTe 概念铺垫
        ↓
Day 3: Trace Transformer 推理流程（Prefill/Decode）
        ↓
Day 4: 手写 Softmax 与 LayerNorm Kernel
        ↓
Day 5: Triton 语言专题 —— 用 Triton 重写 Softmax/GEMM/FA
        ↓
Day 6: Attention IO 分析（4N²+4Nd 口径）
        ↓
Day 7: Transformer 算子分类与总结
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | FlashAttention CUDA 实现（简化版） | [day1/](day1/README.md) |
| Day 2 | CUTLASS 源码分析 + CuTe 概念铺垫 | [day2/](day2/README.md) |
| Day 3 | Trace Transformer 推理流程（Prefill/Decode） | [day3/](day3/README.md) |
| Day 4 | 手写 Softmax 与 LayerNorm Kernel | [day4/](day4/README.md) |
| Day 5 | Triton 语言专题 —— 用 Triton 重写 Softmax/GEMM/FA | [day5/](day5/README.md) |
| Day 6 | Attention IO 分析（4N²+4Nd 口径） | [day6/](day6/README.md) |
| Day 7 | Transformer 算子分类与总结 | [day7/](day7/README.md) |
