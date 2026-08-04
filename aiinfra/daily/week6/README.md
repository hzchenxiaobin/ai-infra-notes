# Week 6：推理系统核心

> 核心目标：PagedAttention、FlashDecoding、Mini 引擎 v0、量化推理、Dynamic Batching、推理系统总结

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| ------------------------------------------------------------|
| 前置要求　 | 已完成 Week 5，掌握 FA-2、KV Cache、vLLM 架构　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　|
| 本周产出　 | PagedAttention kernel、Mini 引擎 v0、量化 kernel、Dynamic Batching 模拟器　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | Mini 引擎 v0 端到端可跑，量化 kernel PASS，理解 PagedAttention block table　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: vLLM Worker 与 PagedAttention
        ↓
Day 2: FlashDecoding —— Decode 阶段并行度突破
        ↓
Day 3: 项目推进 —— Mini 推理引擎 v0
        ↓
Day 4: 端到端 Profiling
        ↓
Day 5: 量化推理专题 —— W8A16/INT8 KV/FP8
        ↓
Day 6: Dynamic Batching
        ↓
Day 7: 推理系统核心问题总结
```

---

## 📚 每日学习材料

每天的学习内容已拆分为独立目录 `dayN/`（含该天的 kernels、exercise、notes）：

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | vLLM Worker 与 PagedAttention | [day1/](day1/README.md) |
| Day 2 | FlashDecoding —— Decode 阶段并行度突破 | [day2/](day2/README.md) |
| Day 3 | 项目推进 —— Mini 推理引擎 v0 | [day3/](day3/README.md) |
| Day 4 | 端到端 Profiling | [day4/](day4/README.md) |
| Day 5 | 量化推理专题 —— W8A16/INT8 KV/FP8 | [day5/](day5/README.md) |
| Day 6 | Dynamic Batching | [day6/](day6/README.md) |
| Day 7 | 推理系统核心问题总结 | [day7/](day7/README.md) |
