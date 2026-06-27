# Day 4 Profiling 任务：Memory Hierarchy 与矩阵转置

## 今日目标
通过矩阵转置对比 naive 版本和 shared memory 优化版的内存性能差异。

## 需要编译的代码

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1
nvcc -o kernels/transpose kernels/transpose.cu
```

## Profiling 任务

### 任务 1：使用 ncu 对比 memory throughput

```bash
# Naive 版本
ncu \
  --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    sm__cycles_elapsed.avg \
  ./kernels/transpose
```

> 注：当前 transpose.cu 包含两个 kernel，ncu 会分别采集。

### 任务 2：使用 nsys 查看完整时间线

```bash
nsys profile -o profiles/day4_transpose_timeline ./kernels/transpose
```

**观察重点**：
- `transpose_naive` 与 `transpose_optimized` 的耗时差异
- memory throughput 差异
- global memory read/write 数据量

## 数据记录

| 版本 | 执行时间 | Memory Throughput | Read Bytes | Write Bytes |
|------|---------|-------------------|-----------|------------|
| naive | | | | |
| optimized | | | | |

## 思考题

1. Naive 版本的瓶颈是什么？读还是写？
2. Shared memory 优化版如何做到 coalesced write？
3. 为什么 shared memory tile 要加 padding？（见 Day 5）
