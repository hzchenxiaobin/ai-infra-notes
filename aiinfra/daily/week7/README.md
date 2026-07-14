# Week 7：系统整合

> 核心目标：将前六周所学整合为一个完整的 Mini AI Infra 系统，完成多请求并发、完整调度器、自定义 Kernel 接入、端到端联调和稳定性测试

| 项目 | 说明 |
|------|------|
| **整体目标** | 整合 GEMM、FlashAttention、Softmax/LayerNorm、KV Cache、Continuous Batching、Scheduler，构建完整的 Mini AI Infra 系统 |
| **核心产出** | ① 多请求并发支持 ② 完整调度器 ③ SGLang/LightLLM 高级特性分析 ④ 全部自定义 Kernel 整合 ⑤ 系统联调 ⑥ 全链路 Profiling ⑦ 代码重构与文档 |
| **验收标准** | ① 支持并发提交多个请求且结果正确返回 ② 完整调度器支持优先级、超时、资源预算 ③ 系统能连续处理 1000+ 请求不崩溃 ④ 内存使用稳定，无持续增长 ⑤ 端到端性能优于或等于 PyTorch eager ⑥ 产出 README、架构图、性能报告 |
| **时间投入** | 工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h |

## 本周知识图谱

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

## 进入每日学习
