# Day 5 Profiling 任务：Bank Conflict 分析

## 今日目标
使用 Nsight Compute 观察 bank conflict 对性能的影响。

## 需要编译的代码

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1
nvcc -o kernels/bank_conflict kernels/bank_conflict.cu
```

## Profiling 任务

### 任务 1：使用 ncu 观察 bank conflict 指标

```bash
ncu \
  --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    sm__cycles_elapsed.avg,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./kernels/bank_conflict
```

**观察重点**：
- `conflict_read` 与 `no_conflict_read` 的 bank conflict 数值差异
- 执行 cycle 数差异
- throughput 差异

### 任务 2：只跑单个 kernel 精确对比

```bash
# 仅运行 conflict 版本
ncu --kernel-name regex:conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./kernels/bank_conflict

# 仅运行无 conflict 版本
ncu --kernel-name regex:no_conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./kernels/bank_conflict
```

## 数据记录

| 版本 | Load Bank Conflicts | Store Bank Conflicts | 执行 Cycles | Throughput |
|------|---------------------|----------------------|------------|-----------|
| conflict_read | | | | |
| no_conflict_read | | | | |

## 思考题

1. 为什么 `tile[TILE_DIM][TILE_DIM]` 会产生 bank conflict？
2. Padding 后为什么能消除 conflict？代价是什么？
3. 除了 padding，还有什么方法可以避免 bank conflict？
