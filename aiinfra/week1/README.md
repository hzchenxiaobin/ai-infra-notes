# Week 1：GPU 执行本质 + Profiling

> 核心目标：建立 GPU 性能直觉 —— **性能 = Memory + 并行度**

| 项目 | 说明 |
|------|------|
| 前置要求 | 已安装 CUDA Toolkit 11.8+ / 12.x，Nsight Compute / Systems |
| 建议时长 | 每日 3~5 小时 |
| 本周产出 | 7 个 CUDA kernel、3+ Nsight Compute 报告、GPU 架构与性能笔记 |

---

## 🧭 本周学习地图

![Week 1 学习地图](website/images/week1_roadmap.svg)

```
Day 1: GPU 执行模型（SM / Warp / SIMT）
        ↓
Day 2: Occupancy 与资源约束（寄存器 / 共享内存）
        ↓
Day 3: 源码分析 —— deviceQuery / occupancyCalculator
        ↓
Day 4: Memory Hierarchy（Global / Shared / Cache / Coalescing）
        ↓
Day 5: Bank Conflict 分析与 Padding 技巧
        ↓
Day 6: Nsight Profiling 实战（Compute + Systems）
        ↓
Day 7: 总结与复盘
```

---

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`，包含该天的 kernels、exercise、notes：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | GPU 执行模型基础 | [day1/](day1/README.md) |
| Day 2 | Occupancy 与资源约束 | [day2/](day2/README.md) |
| Day 3 | 认识你的 GPU —— deviceQuery 与 Occupancy 计算 | [day3/](day3/README.md) |
| Day 4 | Memory Hierarchy 深入 | [day4/](day4/README.md) |
| Day 5 | Bank Conflict 分析与实践 | [day5/](day5/README.md) |
| Day 6 | Nsight Profiling 实战 | [day6/](day6/README.md) |
| Day 7 | 总结与复盘 | [day7/](day7/README.md) |

## 📂 周级资源

- [tools/](tools/) — Occupancy 计算器等辅助工具
- [notes/week1_notes.md](notes/week1_notes.md) — 学习笔记模板
- [profiles/week1_profile_summary.md](profiles/week1_profile_summary.md) — Profiling 报告汇总
