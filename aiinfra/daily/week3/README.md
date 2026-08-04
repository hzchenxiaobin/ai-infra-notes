# Week 3：手撕复盘 + Transformer 算子手写

> 核心目标：限时手写 Reduce/GEMM/Softmax/LayerNorm kernel，掌握 Transformer 推理的 Prefill/Decode 特征，用 Triton 重写核心算子

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| --------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 2，掌握 Warp Shuffle、Register Blocking、GEMM tiling、WMMA 基础　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　|
| 本周产出　 | 限时手撕 kernel 录音/留档、Softmax/LayerNorm CUDA kernel、Triton softmax/gemm/FA 三方 benchmark、Attention IO 分析　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | 30 分钟内手写 Reduce + 60 分钟手写 GEMM tiling，Triton GEMM 追平 cuBLAS　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: FlashAttention CUDA 实现（简化版）
        ↓
Day 2: 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告
        ↓
Day 3: Trace Transformer 推理流程（Prefill/Decode）
        ↓
Day 4: 手写 Softmax 与 LayerNorm Kernel
        ↓
Day 5: 源码分析 —— PyTorch / FasterTransformer
        ↓
Day 6: Triton 语言专题 —— 用 Triton 重写 Softmax/GEMM/FA
        ↓
Day 7: Attention IO 分析（4N²+4Nd 口径）

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | FlashAttention CUDA 实现（简化版） | [day1/](day1/README.md) |
| Day 2 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | [day2/](day2/README.md) |
| Day 3 | Trace Transformer 推理流程（Prefill/Decode） | [day3/](day3/README.md) |
| Day 4 | 手写 Softmax 与 LayerNorm Kernel | [day4/](day4/README.md) |
| Day 5 | 源码分析 —— PyTorch / FasterTransformer | [day5/](day5/README.md) |
| Day 6 | Triton 语言专题 —— 用 Triton 重写 Softmax/GEMM/FA | [day6/](day6/README.md) |
| Day 7 | Attention IO 分析（4N²+4Nd 口径） | [day7/](day7/README.md) |
