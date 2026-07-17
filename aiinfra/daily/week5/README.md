# Week 5：推理系统与 KV Cache

> 核心目标：进入 AI Infra 核心，理解 LLM 推理的 Prefill/Decode 两阶段，实现 KV Cache，阅读 vLLM 源码，构建第一个可运行的 Mini 推理引擎

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| ------------| -------------------------------------------------------------------------------------------------------------------------------------------------------|
| 前置要求　 | 已完成 Week 4 学习，掌握 FlashAttention Forward Kernel、Online Softmax 推导、IO 优化方法论、Mini 引擎 Attention 集成　　　　　　　　　　　　　　　　  |
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　  |
| 本周产出　 | Prefill/Decode 模拟脚本、KV Cache CUDA 实现（C++ 类）、vLLM 架构分析报告、PagedAttention 笔记、Mini 推理引擎 v0、Profiling 报告、推理系统核心问题清单 |
| 周日里程碑 | 实现 KV Cache（decode latency 降低 10x+），构建 Mini 推理引擎 v0 完成单请求推理，能画出 vLLM 架构图并测量 TTFT/per-token decode latency　　　　　　　 |

---

## 🧭 本周学习地图

```
Day 1: Prefill vs Decode → 两阶段特征对比 + PyTorch 模拟 + 算术强度分析
  ↓
Day 2: KV Cache 实现 → C++/CUDA 缓存分配、更新、查询、多轮对话
  ↓
Day 3: vLLM 整体架构 → LLMEngine / Scheduler / Worker / SequenceGroup
  ↓
Day 4: vLLM Worker + PagedAttention → BlockSpaceManager / Block Table / Copy-on-Write
  ↓
Day 5: Mini 推理引擎 v0 → 单请求 + KV Cache + Prefill/Decode 循环
  ↓
Day 6: 端到端 Profiling → TTFT / TBT / 阶段 latency / 瓶颈定位
  ↓
Day 7: 推理系统核心问题总结 → 内存管理、Batch 策略、Latency 隐藏、调度开销
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | 推理流程 —— Prefill vs Decode | [day1/](day1/README.md) |
| Day 2 | 实现 KV Cache | [day2/](day2/README.md) |
| Day 3 | vLLM 整体架构分析 | [day3/](day3/README.md) |
| Day 4 | vLLM Worker 与 PagedAttention | [day4/](day4/README.md) |
| Day 5 | 项目推进 —— Mini 推理引擎 v0 | [day5/](day5/README.md) |
| Day 6 | 端到端 Profiling | [day6/](day6/README.md) |
| Day 7 | 推理系统核心问题总结与 Week 5 收官 | [day7/](day7/README.md) |
