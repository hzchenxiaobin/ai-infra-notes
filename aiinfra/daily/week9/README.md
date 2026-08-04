# Week 9：系统联调与项目打磨

> 核心目标：整合全部自定义 Kernel、六步分层联调、CUDA Graph、全链路 Profiling、项目文档与架构图

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| --------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 8，掌握多请求并发、完整调度器、分布式并行、MoE+EP　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　|
| 本周产出　 | 自定义 Kernel 集成引擎、CUDA Graph 集成、全链路 profiling 报告、README + 架构图　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | Mini 引擎全链路可跑（自定义 kernel + CUDA Graph），README 10 分钟内可跑通　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: 整合全部自定义 Kernel
        ↓
Day 2: 系统联调（六步分层验证）
        ↓
Day 3: 全链路 Profiling
        ↓
Day 4: CUDA Graph 实操 —— 消除 Launch Overhead
        ↓
Day 5: 代码重构与文档
        ↓
Day 6: 项目文档完善（README）
        ↓
Day 7: 架构图与数据流图

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | 整合全部自定义 Kernel | [day1/](day1/README.md) |
| Day 2 | 系统联调（六步分层验证） | [day2/](day2/README.md) |
| Day 3 | 全链路 Profiling | [day3/](day3/README.md) |
| Day 4 | CUDA Graph 实操 —— 消除 Launch Overhead | [day4/](day4/README.md) |
| Day 5 | 代码重构与文档 | [day5/](day5/README.md) |
| Day 6 | 项目文档完善（README） | [day6/](day6/README.md) |
| Day 7 | 架构图与数据流图 | [day7/](day7/README.md) |
