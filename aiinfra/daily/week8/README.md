# Week 8：系统整合与分布式并行

> 核心目标：掌握多请求并发、完整调度器、投机解码、分布式并行（TP/PP/DP）、Ring Attention、MoE+EP

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| --------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 7，掌握 Continuous Batching、vLLM Scheduler、PD 分离、Mini 引擎 v1　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　|
| 本周产出　 | 完整调度器（优先级/超时/抢占）、投机解码模拟、分布式推理分析、MoE 路由模拟器　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | 完整调度器 500 请求不崩溃，理解 TP/PP/DP/EP 四维并行与 all-to-all/all-reduce 通信模式　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: 调度优化策略总结
        ↓
Day 2: 多请求并发支持
        ↓
Day 3: 完整调度器（优先级/超时/抢占）
        ↓
Day 4: SGLang / 投机解码
        ↓
Day 5: 分布式推理 —— TP/PP/DP 与通信计算重叠
        ↓
Day 6: Ring Attention —— 长上下文分布式注意力
        ↓
Day 7: MoE + EP 并行专题

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | 调度优化策略总结 | [day1/](day1/README.md) |
| Day 2 | 多请求并发支持 | [day2/](day2/README.md) |
| Day 3 | 完整调度器（优先级/超时/抢占） | [day3/](day3/README.md) |
| Day 4 | SGLang / 投机解码 | [day4/](day4/README.md) |
| Day 5 | 分布式推理 —— TP/PP/DP 与通信计算重叠 | [day5/](day5/README.md) |
| Day 6 | Ring Attention —— 长上下文分布式注意力 | [day6/](day6/README.md) |
| Day 7 | MoE + EP 并行专题 | [day7/](day7/README.md) |
