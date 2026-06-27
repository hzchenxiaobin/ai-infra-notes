# Week 2：CUDA 进阶优化与性能分析

> 核心目标：掌握 Warp Shuffle、Register Blocking、CUDA Stream 异步执行、Nsight 性能分析和 FlashAttention CUDA 实现

| 项目 | 说明 |
|------|------|
| 前置要求 | 已完成 Week 1 学习，掌握向量加法、Naive GEMM、Shared Memory Tiling GEMM、Softmax Kernel |
| 建议时长 | 工作日每天 2.5h，周末每天 6h，周计 24.5h |
| 本周产出 | Warp Reduce Kernel、Register Blocking GEMM（cuBLAS 40%+）、Multi-Stream 重叠执行、Nsight 分析报告、FlashAttention 简化版 Forward Kernel、整合优化 GEMM（cuBLAS 70%+） |
| 周日里程碑 | 手写优化 GEMM 达到 cuBLAS 70%+ 性能，完成简化版 FlashAttention Forward Kernel |

---

## 🧭 本周学习地图

```
Day 1: Warp Shuffle 原语 → Warp Reduce Kernel（两级归约）
        ↓
Day 2: Register Blocking + 2D Tiling → GEMM cuBLAS 40%+
        ↓
Day 3: CUDA Streams 异步 → H2D/Compute/D2H 重叠流水线
        ↓
Day 4: Nsight Compute → Register Blocking GEMM 瓶颈分析
        ↓
Day 5: FlashAttention → Online Softmax 推导 + Forward Kernel
        ↓
Day 6: 整合 Warp Shuffle + Register Blocking → GEMM cuBLAS 70%+
        ↓
Day 7: 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立文件，位于 [`days/`](days/) 目录：

| Day | 主题 | 文件 |
|-----|------|------|
| Day 1 | Warp Shuffle 原语与 Warp/Block Reduce | [days/day1.md](days/day1.md) |
| Day 2 | Register Blocking 与 2D Tiling | [days/day2.md](days/day2.md) |
| Day 3 | CUDA Streams 与异步执行 | [days/day3.md](days/day3.md) |
| Day 4 | Nsight Compute 性能分析 | [days/day4.md](days/day4.md) |
| Day 5 | FlashAttention CUDA 实现（简化版） | [days/day5.md](days/day5.md) |
| Day 6 | 整合优化到 cuBLAS 70%+ | [days/day6.md](days/day6.md) |
| Day 7 | 限时 Kernel 手撕 + GitHub 整理 + 性能对比报告 | [days/day7.md](days/day7.md) |
