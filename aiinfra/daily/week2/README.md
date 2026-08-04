# Week 2：CUDA Kernel 优化 + Tensor Core

> 核心目标：掌握 Warp Shuffle、Register Blocking、float4 向量化、GEMM 七层优化路径，理解 Tensor Core（WMMA）与手撕检验

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| ------------------------------------------------------------|
| 前置要求　 | 已完成 Week 1，掌握 SM/Warp/Thread、Occupancy、Memory Hierarchy、ncu/nsys　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　|
| 本周产出　 | warp_reduce / register_blocking_gemm / wmma_gemm 等 kernel、GEMM 优化性能对比表、限时手撕留档　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | GEMM 优化达 cuBLAS 60%+（FMA 路线），30 分钟手写 Reduce + 60 分钟手写 GEMM tiling　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: Warp Shuffle 原语与 Warp/Block Reduce
        ↓
Day 2: Register Blocking 与 2D Tiling
        ↓
Day 3: CUDA Streams 与异步执行
        ↓
Day 4: Nsight Compute 性能分析
        ↓
Day 5: 整合优化到 cuBLAS 70%+（GEMM 七层路径）
        ↓
Day 6: Tensor Core 与 WMMA —— 从 FMA 到 Tensor Core
        ↓
Day 7: 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | Warp Shuffle 原语与 Warp/Block Reduce | [day1/](day1/README.md) |
| Day 2 | Register Blocking 与 2D Tiling | [day2/](day2/README.md) |
| Day 3 | CUDA Streams 与异步执行 | [day3/](day3/README.md) |
| Day 4 | Nsight Compute 性能分析 | [day4/](day4/README.md) |
| Day 5 | 整合优化到 cuBLAS 70%+（GEMM 七层路径） | [day5/](day5/README.md) |
| Day 6 | Tensor Core 与 WMMA —— 从 FMA 到 Tensor Core | [day6/](day6/README.md) |
| Day 7 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | [day7/](day7/README.md) |
