# Week 5：推理系统与 KV Cache

> 核心目标：进入 AI Infra 核心，理解 LLM 推理的 Prefill/Decode 两阶段，实现 KV Cache，阅读 vLLM 源码，构建第一个可运行的 Mini 推理引擎

| 项目 | 说明 |
|------|------|
| **整体目标** | 理解 LLM 推理的 Prefill/Decode 两阶段本质差异，掌握 KV Cache 的设计与实现，阅读 vLLM 架构源码与 PagedAttention，构建支持单请求的 Mini 推理引擎 v0，完成端到端 Profiling |
| **核心产出** | ① Prefill/Decode 模拟脚本 ② KV Cache CUDA 实现（C++ 类）③ vLLM 架构分析报告 ④ PagedAttention 笔记 ⑤ Mini 推理引擎 v0 ⑥ Profiling 报告 ⑦ 推理系统核心问题清单 |
| **验收标准** | ① 能清晰区分 Prefill 和 Decode 的计算/内存特征 ② KV Cache 输出与无 cache 版本一致，decode latency 降低 10x+ ③ 能画出 vLLM 架构图并解释请求生命周期 ④ Mini 引擎 v0 能完成单条请求完整推理 ⑤ 能测量 TTFT 和 per-token decode latency |
| **时间投入** | 工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h |

## 本周知识图谱

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

## 进入每日学习
