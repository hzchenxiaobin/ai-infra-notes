# Day 7 Profiling 总结

## 今日目标
整理本周所有 profiling 数据，形成可复用的性能分析文档。

## 总结任务

### 任务 1：填写 Week 1 性能汇总表

| Day | Kernel | Occupancy | Memory Throughput | Compute Throughput | Bank Conflicts | 瓶颈类型 |
|-----|--------|-----------|-------------------|-------------------|----------------|---------|
| 1 | hello_gpu | | | | N/A | |
| 2 | occupancy_test | | | | N/A | |
| 4 | transpose_naive | | | | N/A | |
| 4 | transpose_optimized | | | | N/A | |
| 5 | conflict_read | | | | | |
| 5 | no_conflict_read | | | | | |

### 任务 2：绘制/整理图表

1. GPU 内存层次延迟图
2. Occupancy 与资源关系图
3. Roofline 简图（标出本周各 kernel 大致位置）

### 任务 3：撰写 Week 1 Profiling 总结

参考 `profiles/week1_profile_summary.md` 模板，填写：

- 环境信息
- 各 kernel 详细分析
- Roofline 分析
- 主要发现
- 下一步优化方向

## 检查清单

- [ ] 所有 daily profiling 任务已完成
- [ ] 性能汇总表已填写
- [ ] `profiles/week1_profile_summary.md` 已更新
- [ ] 能用自己的话解释每个 kernel 的瓶颈

## 面试准备

1. 本周哪个 kernel 是 memory-bound？为什么？
2. 哪个 kernel 受 occupancy 影响最大？
3. 如果给你一个 kernel，你的 profiling 流程是什么？
