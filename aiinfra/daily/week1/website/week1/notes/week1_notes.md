# Week 1 学习笔记模板

## 1. GPU 执行模型

- SM：
- Warp：
- SIMT：
- Grid/Block/Thread：

## 2. Occupancy

- 定义：
- 影响因素：
- 当前 GPU 的理论最大 warp 数：

## 3. 内存层次

| 内存类型 | 典型延迟 | 容量 | 是否可编程 |
|---------|---------|------|-----------|
| Register | | | |
| Shared Memory | | | |
| L1 Cache | | | |
| L2 Cache | | | |
| Global Memory | | | |

## 4. Coalesced Access

- 定义：
- 如何写出 coalesced 代码：

## 5. Bank Conflict

- 结构：32 banks，每 bank 4 bytes
- 产生条件：
- 消除方法：

## 6. Nsight 常用命令

```bash
# 基础 profiling
ncu ./app

# 指定指标
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./app

# 生成报告
ncu --set full -o report ./app

# 系统级 trace
nsys profile -o timeline ./app
```

## 7. 本周实验记录

| Kernel | Occupancy | Memory Throughput | Compute Throughput | 瓶颈类型 |
|--------|-----------|-------------------|-------------------|---------|
| hello_gpu | | | | |
| occupancy_test | | | | |
| transpose_naive | | | | |
| transpose_optimized | | | | |
| conflict_read | | | | |
| no_conflict_read | | | | |

## 8. 面试问题自测

- Q: 什么是 SIMT？
- A:

- Q: Occupancy 是否越高越好？
- A:

- Q: 如何判断 kernel 是 memory-bound 还是 compute-bound？
- A:
