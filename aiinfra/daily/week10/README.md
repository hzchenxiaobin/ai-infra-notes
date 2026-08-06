# Week 10：项目整合与面试冲刺

> 核心目标：Mini 引擎真整合、全链路 Profiling、项目文档与架构图、面试题库、Mock 面试、诊断剧本与最终复盘

| 项目　　　 | 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------| ------------------------------------------------------------|
| 前置要求　 | 已完成 Week 9，掌握分布式并行、MoE+EP、多硬件对比　　　　　　　　　　　　　　　　　　　　　　　|
| 建议时长　 | 工作日每天 2.5h，周末每天 6h，周计 24.5h　　　　　　　　　　|
| 本周产出　 | 全链路引擎（真整合）、项目 README + 架构图、面试题库（基础+进阶）、Mock 面试记录、诊断剧本 3 案例、10 周能力地图　　　　　　　　　　　　　　　　　　　　　　　　　|
| 周日里程碑 | Mini 引擎全链路可跑，README 10 分钟内可跑通，面试题自测全 PASS，完成 10 周能力地图　　　　　　　　　　　　　　　　　　　　　　　|

---

## 🧭 本周学习地图

```
Day 1: 完成 custom kernel 封装（LayerNorm/Softmax/FlashAttention），接入 Mini 引擎
        ↓
Day 2: 接入引擎并联调（六步分层验证 + 全链路 Profiling，真引擎模式 `--real`）
        ↓
Day 3: 项目文档（README）+ 架构图 + 数据流图
        ↓
Day 4: 高频面试题基础篇 + 进阶篇（含系统设计题）
        ↓
Day 5: Mock 面试 + STAR 项目话术
        ↓
Day 6: 诊断流程实战剧本（低 MFU/OOM/hang 三案例）+ 限时手撕清单（10 项）
        ↓
Day 7: 最终复盘 —— 10 周能力地图 + 查漏补缺
```

---

## 📚 每日学习材料

| Day | 主题 | 目录 |
|-----|------|------|
| Day 1 | 整合全部自定义 Kernel | [day1/](day1/README.md) |
| Day 2 | 系统联调（六步分层验证） | [day2/](day2/README.md) |
| Day 3 | 项目文档完善（README） | [day3/](day3/README.md) |
| Day 4 | 高频面试题基础篇 | [day4/](day4/README.md) |
| Day 5 | Mock 面试 | [day5/](day5/README.md) |
| Day 6 | 诊断流程实战剧本 + 手撕限时清单 | [day6/](day6/README.md) |
| Day 7 | 最终复盘 —— 10 周能力地图 | [day7/](day7/README.md) |

> 📁 补充材料：`_supplementary/` 目录包含面试题进阶篇、查漏补缺、多请求并发、完整调度器、Latency/Throughput 测试、端到端 Profiling、源码分析、架构图等延伸内容。
