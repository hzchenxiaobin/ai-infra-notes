# Week 6：Batching & 调度

> 核心目标：建立系统感，理解 Dynamic Batching 和 Continuous Batching 的原理与实现，阅读 vLLM/TensorRT-LLM 调度器源码，将 Mini 引擎升级到 v1 支持多请求并发

| 项目 | 说明 |
|------|------|
| **整体目标** | 理解从单请求到多请求的推理系统调度，掌握 Dynamic Batching 和 Continuous Batching 的实现，构建支持多请求并发的 Mini 推理引擎 v1 |
| **核心产出** | ① Dynamic Batching 实现 ② Continuous Batching 实现 ③ vLLM Scheduler 源码分析 ④ TensorRT-LLM Scheduler 对比 ⑤ Mini 引擎 v1（多请求 + Scheduler）⑥ Latency/Throughput benchmark ⑦ 调度策略对比表 |
| **验收标准** | ① 能实现 Dynamic Batching，多个请求正确聚合 ② 能实现 Continuous Batching，新请求可任意 iteration 加入 ③ 能解释 vLLM Scheduler 的核心逻辑 ④ Mini 引擎 v1 能同时处理多个请求 ⑤ 能绘制 throughput-latency 曲线并识别饱和点 |
| **时间投入** | 工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h |

## 本周知识图谱

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

## 进入每日学习
