# Day 2 Profiling 任务：Occupancy 与资源约束

## 今日目标
通过 profiling 理解寄存器使用对 occupancy 的影响。

## 需要编译的代码

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1
nvcc -o kernels/occupancy_test kernels/occupancy_test.cu
```

## Profiling 任务

### 任务 1：使用 ncu 查看 occupancy 相关指标

```bash
ncu \
  --metrics \
    sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    launch__shared_mem_per_block_dynamic,\
    launch__shared_mem_per_block_static \
  ./kernels/occupancy_test
```

**观察重点**：
- `launch__registers_per_thread` 的值
- `sm__occupancy.avg.pct_of_peak_sustained_elapsed` 的值
- 理论 occupancy 与实际 occupancy 的差异

### 任务 2：对比不同寄存器用量下的 occupancy

修改 `kernels/occupancy_test.cu` 中 `#pragma unroll` 的参数或增加局部变量，重新编译并记录：

| 版本 | 寄存器/线程 | 理论 Occupancy | 实际 Occupancy |
|------|------------|---------------|---------------|
| 基础版 | | | |
| 增加局部变量 | | | |
| 使用 `__launch_bounds__` | | | |

## 数据记录

| 指标 | 基础版 | 高寄存器版 | launch_bounds 版 |
|------|--------|-----------|-----------------|
| Registers/thread | | | |
| Occupancy | | | |
| 执行时间 | | | |

## 思考题

1. 为什么增加局部变量会降低 occupancy？
2. `__launch_bounds__` 是如何强制编译器减少寄存器使用的？
3. 100% occupancy 一定比 50% occupancy 快吗？
