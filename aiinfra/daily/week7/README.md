# Week 7：系统整合

> 核心目标：将前六周所学整合为一个完整的 Mini AI Infra 系统，完成多请求并发、完整调度器、自定义 Kernel 接入、端到端联调和稳定性测试

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| ------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 6 学习，掌握 Dynamic/Continuous Batching、vLLM Scheduler、Mini 推理引擎 v1、Throughput/Latency 测试　　　　　　　　　　　　　　　　　　　　　 |
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| 本周产出　 | 多请求并发支持、完整调度器（优先级/超时/资源预算/抢占）、SGLang/LightLLM 高级特性分析、全部自定义 Kernel 整合、系统联调、全链路 Profiling、代码重构与文档 |
| 周日里程碑 | 整合为完整 Mini AI Infra 系统，支持多请求并发与完整调度器，连续处理 1000+ 请求不崩溃，端到端性能优于或等于 PyTorch eager　　　　　　　　　　　　　　　　  |

---

## 🧭 本周学习地图

```
Day 1: 多请求并发支持 → 线程安全队列、异步 Future、结果回调、请求生命周期
  ↓
Day 2: 完整调度器 → 优先级、超时、资源预算、抢占
  ↓
Day 3: SGLang / LightLLM 高级特性 → Speculative Decoding、Chunked Prefill、Prefix Caching
  ↓
Day 4: 整合全部自定义 Kernel → GEMM、FlashAttention、Softmax、LayerNorm 接入
  ↓
Day 5: 系统联调 → KV Cache + Batching + Scheduler + Kernel 端到端测试
  ↓
Day 6: 全链路 Profiling → 系统级瓶颈定位、与 vLLM 对比
  ↓
Day 7: 代码重构与文档 → README、架构图、接口统一、稳定性测试
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | 多请求并发支持 | [day1/](day1/README.md) |
| Day 2 | 完整调度器 | [day2/](day2/README.md) |
| Day 3 | SGLang / LightLLM 高级特性 | [day3/](day3/README.md) |
| Day 4 | 整合全部自定义 Kernel | [day4/](day4/README.md) |
| Day 5 | 系统联调 | [day5/](day5/README.md) |
| Day 6 | 全链路 Profiling | [day6/](day6/README.md) |
| Day 7 | 代码重构与文档 | [day7/](day7/README.md) |
