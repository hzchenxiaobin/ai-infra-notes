# Week 1：GPU 执行模型与内存优化基础

> 核心目标：掌握 SM/Warp/SIMT 执行模型、Occupancy 资源约束、Memory Hierarchy、Coalescing/Bank Conflict 与 Nsight Profiling，建立 GPU 性能直觉

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 前置要求　 | 具备 C/C++ 基础，了解基本数据结构与算法（如双指针）；无需 CUDA 经验　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| 本周产出　 | hello_gpu / vector_add / relu / matrix_add / transpose / reduction / matmul 等 CUDA Kernel、deviceQuery 与 Occupancy 计算报告、Bank Conflict 对比实验、3+ Nsight 报告、Week 1 学习笔记与面试速查表 |
| 周日里程碑 | 系统复盘 Week 1，建立从"硬件执行模型"到"代码优化"的完整思路链，能用 ncu/nsys 定位 kernel 瓶颈类型（memory-bound / compute-bound / latency-bound）　　　　　　　　　　　 |

---

## 🧭 本周学习地图

```
Day 1: GPU 执行模型 → SM/Warp/SIMT + Grid/Block/Thread + 第一个 CUDA 程序 + Vector Add
        ↓
Day 2: Occupancy 与资源约束 → 寄存器/Shared Memory/Block 数上限 + Register Spilling + ReLU
        ↓
Day 3: 认识你的 GPU → deviceQuery + 峰值算力/显存带宽计算 + Matrix Addition
        ↓
Day 4: Memory Hierarchy 深入 → Coalesced Access + Shared Memory Tiling + Matrix Transpose
        ↓
Day 5: Bank Conflict 分析 → 32 bank 结构 + Padding 优化 + Reduction
        ↓
Day 6: Nsight Profiling 实战 → nsys/ncu + Roofline 模型 + Matrix Multiplication
        ↓
Day 7: 总结与复盘 → 知识地图 + 优化决策树 + 面试速查表 + 综合练习
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | GPU 执行模型基础 | [day1/](day1/README.md) |
| Day 2 | Occupancy 与资源约束 | [day2/](day2/README.md) |
| Day 3 | 认识你的 GPU —— deviceQuery 与 Occupancy 计算 | [day3/](day3/README.md) |
| Day 4 | Memory Hierarchy 深入 | [day4/](day4/README.md) |
| Day 5 | Bank Conflict 分析与实践 | [day5/](day5/README.md) |
| Day 6 | Nsight Profiling 实战 | [day6/](day6/README.md) |
| Day 7 | 总结与复盘 | [day7/](day7/README.md) |
