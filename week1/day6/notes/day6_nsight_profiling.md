# Day 6 Profiling 实战：综合Profiling

## 今日目标
对本周所有 kernel 进行综合 profiling，掌握 Roofline 分析和瓶颈定位。

## 需要编译的代码

```bash
cd /Users/chenbinbin/GitHub/aiinfra/week1
nvcc -o kernels/hello_gpu kernels/hello_gpu.cu
nvcc -o kernels/occupancy_test kernels/occupancy_test.cu
nvcc -o kernels/transpose kernels/transpose.cu
nvcc -o kernels/bank_conflict kernels/bank_conflict.cu
```

## Profiling 任务

### 任务 1：生成完整 Nsight Compute 报告

对每个 kernel 生成完整报告：

```bash
ncu --set full -o profiles/day6_hello_gpu ./kernels/hello_gpu
ncu --set full -o profiles/day6_occupancy_test ./kernels/occupancy_test
ncu --set full -o profiles/day6_transpose ./kernels/transpose
ncu --set full -o profiles/day6_bank_conflict ./kernels/bank_conflict
```

用 GUI 打开：

```bash
ncu-ui profiles/day6_transpose.ncu-rep
```

### 任务 2：使用 nsys 查看应用级时间线

```bash
nsys profile -o profiles/day6_full_timeline \
  --trace cuda,nvtx,osrt \
  ./kernels/transpose
```

### 任务 3：Roofline 分析

对每个 kernel 记录：

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `launch__occupancy`

判断瓶颈类型：

| Kernel | Compute Throughput | Memory Throughput | 瓶颈类型 |
|--------|-------------------|-------------------|---------|
| hello_gpu | | | |
| occupancy_test | | | |
| transpose_naive | | | |
| transpose_optimized | | | |
| conflict_read | | | |
| no_conflict_read | | | |

## 关键指标速查

| 指标 | 含义 |
|------|------|
| `sm__occupancy.avg.pct_of_peak_sustained_elapsed` | Occupancy |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 计算单元利用率 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 显存带宽利用率 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory load bank conflict |
| `sm__cycles_elapsed.avg` | 执行周期数 |

## 今日产出

- [ ] 至少 3 个完整 Nsight Compute 报告（`.ncu-rep`）
- [ ] 1 个 Nsight Systems 时间线报告（`.nsys-rep`）
- [ ] 1 张瓶颈类型判断表
