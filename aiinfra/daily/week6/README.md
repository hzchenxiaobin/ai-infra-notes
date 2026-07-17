# Week 6：Batching & 调度

> 核心目标：建立系统感，理解 Dynamic Batching 和 Continuous Batching 的原理与实现，阅读 vLLM/TensorRT-LLM 调度器源码，将 Mini 引擎升级到 v1 支持多请求并发

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| ------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 5 学习，掌握 KV Cache 实现、vLLM 架构与 PagedAttention、Mini 推理引擎 v0、Prefill/Decode Profiling　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| 本周产出　 | Dynamic Batching 实现、Continuous Batching 实现、vLLM Scheduler 源码分析、TensorRT-LLM Scheduler 对比、Mini 引擎 v1（多请求 + Scheduler）、Latency/Throughput benchmark、调度策略对比表 |
| 周日里程碑 | 实现 Dynamic/Continuous Batching，构建 Mini 引擎 v1 支持多请求并发，绘制 throughput-latency 曲线并识别饱和点　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |

---

## 🧭 本周学习地图

```
Day 1: Dynamic Batching → 请求聚合、padding、timeout、max batch size
  ↓
Day 2: Continuous Batching → iteration-level 调度、请求动态加入/退出
  ↓
Day 3: vLLM Scheduler 源码分析 → schedule() / SchedulingBudget / preemption
  ↓
Day 4: TensorRT-LLM / LightLLM 调度对比 → Inflight Batching / Chunked Prefill
  ↓
Day 5: Mini 推理引擎 v1 → Continuous Batching + Scheduler + 多请求并发
  ↓
Day 6: Latency / Throughput 测试 → 不同 batch size / 请求分布 / 饱和点
  ↓
Day 7: 调度优化策略总结 → 策略对比表 + 面试复盘 + GitHub 整理
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | Dynamic Batching | [day1/](day1/README.md) |
| Day 2 | Continuous Batching | [day2/](day2/README.md) |
| Day 3 | vLLM Scheduler 源码分析 | [day3/](day3/README.md) |
| Day 4 | TensorRT-LLM / LightLLM 调度对比 | [day4/](day4/README.md) |
| Day 5 | Mini 推理引擎 v1 | [day5/](day5/README.md) |
| Day 6 | Latency / Throughput 测试 | [day6/](day6/README.md) |
| Day 7 | 调度优化策略总结与 Week 6 收官 | [day7/](day7/README.md) |
