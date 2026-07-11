# Week 1 Profiling 报告汇总

> 本周 profiling 按天拆解，具体任务见各 day 的 `notes/` 目录。

## 每日 Profiling 任务索引

| 天数 | 主题 | 对应文件 | 核心代码 |
|------|------|---------|---------|
| Day 1 | GPU 执行模型与 kernel launch | [day1/notes/day1_hello_gpu.md](../day1/notes/day1_hello_gpu.md) | `kernels/hello_gpu.cu` |
| Day 2 | Occupancy 与寄存器约束 | [day2/notes/day2_occupancy.md](../day2/notes/day2_occupancy.md) | `kernels/occupancy_test.cu` |
| Day 3 | 设备属性与 Occupancy Calculator | [day3/notes/day3_device_query.md](../day3/notes/day3_device_query.md) | `deviceQuery` / `occupancyCalculator` |
| Day 4 | Memory Hierarchy 与矩阵转置 | [day4/notes/day4_transpose.md](../day4/notes/day4_transpose.md) | `kernels/transpose.cu` |
| Day 5 | Bank Conflict 分析 | [day5/notes/day5_bank_conflict.md](../day5/notes/day5_bank_conflict.md) | `kernels/bank_conflict.cu` |
| Day 6 | Nsight 综合 Profiling 实战 | [day6/notes/day6_nsight_profiling.md](../day6/notes/day6_nsight_profiling.md) | 全部 kernel |
| Day 7 | 总结与复盘 | [day7/notes/day7_summary.md](../day7/notes/day7_summary.md) | - |

---

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU 型号 | |
| CUDA Capability | |
| SM 数量 | |
| 每个 SM 最大 Warp 数 | |
| 理论显存带宽 | |
| CUDA 版本 | |
| Nsight Compute 版本 | |
| Nsight Systems 版本 | |

---

## 性能汇总表

请根据每日任务填写：

| Day | Kernel | Occupancy | Memory Throughput | Compute Throughput | Bank Conflicts | 瓶颈类型 |
|-----|--------|-----------|-------------------|-------------------|----------------|---------|
| 1 | hello_gpu | | | | N/A | |
| 2 | occupancy_test | | | | N/A | |
| 4 | transpose_naive | | | | N/A | |
| 4 | transpose_optimized | | | | N/A | |
| 5 | conflict_read | | | | | |
| 5 | no_conflict_read | | | | | |

---

## 各 Kernel 分析

### 1. hello_gpu

- **对应命令**：见 [`day1/notes/day1_hello_gpu.md`](../day1/notes/day1_hello_gpu.md)
- **关键指标**：
- **分析**：

### 2. occupancy_test

- **对应命令**：见 [`day2/notes/day2_occupancy.md`](../day2/notes/day2_occupancy.md)
- **关键指标**：
- **分析**：

### 3. transpose_naive vs transpose_optimized

- **对应命令**：见 [`day4/notes/day4_transpose.md`](../day4/notes/day4_transpose.md)
- **关键指标对比**：

| 版本 | Memory Throughput | 执行时间 | 备注 |
|------|-------------------|---------|------|
| naive | | | |
| optimized | | | |

- **分析**：

### 4. conflict_read vs no_conflict_read

- **对应命令**：见 [`day5/notes/day5_bank_conflict.md`](../day5/notes/day5_bank_conflict.md)
- **关键指标**：
- **分析**：

---

## Roofline 分析

- **Peak FLOP/s**：
- **Peak Bandwidth**：

在 Roofline 图上标出各 kernel 大致位置：

```
  FLOP/s
    │
Peak├───────────────┐
    │               │
    │     ○ optimized?    ○ occupancy_test?
    │               │
    │        ○ transpose_naive
    │               │
    │   ○ hello_gpu
    │               │
    └───────────────┴─────────────── Arithmetic Intensity
                Memory-bound → Compute-bound
```

---

## 总结

- **本周主要发现**：
- **下一步优化方向**：
