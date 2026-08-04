# Week 7：Batching 与调度

> 核心目标：掌握 Continuous Batching、vLLM Scheduler 源码、Chunked Prefill、PD 分离、Mini 引擎 v1、性能测试

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| --------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 6，掌握 PagedAttention、Mini 引擎 v0、量化、Dynamic Batching　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　|
| 本周产出　 | Continuous Batching 模拟器、vLLM Scheduler 复刻、Chunked Prefill simulator、PD 分离模拟器、Mini 引擎 v1　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | Mini 引擎 v1 多请求并发可跑，PD 分离模拟器验证 TTFT/TPOT 改善，理解 vLLM schedule() 5 步流程　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: Continuous Batching
        ↓
Day 2: vLLM Scheduler 源码分析
        ↓
Day 3: TensorRT-LLM / LightLLM / SGLang 调度对比
        ↓
Day 4: Chunked Prefill 与 Prefix Caching 实操
        ↓
Day 5: Mini 推理引擎 v1（多请求并发）
        ↓
Day 6: Prefill/Decode 分离推理（PD Disaggregated）
        ↓
Day 7: Latency / Throughput 测试

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | Continuous Batching | [day1/](day1/README.md) |
| Day 2 | vLLM Scheduler 源码分析 | [day2/](day2/README.md) |
| Day 3 | TensorRT-LLM / LightLLM / SGLang 调度对比 | [day3/](day3/README.md) |
| Day 4 | Chunked Prefill 与 Prefix Caching 实操 | [day4/](day4/README.md) |
| Day 5 | Mini 推理引擎 v1（多请求并发） | [day5/](day5/README.md) |
| Day 6 | Prefill/Decode 分离推理（PD Disaggregated） | [day6/](day6/README.md) |
| Day 7 | Latency / Throughput 测试 | [day7/](day7/README.md) |
