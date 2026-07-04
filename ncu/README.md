# Week 1 & Week 2 性能分析任务汇总

本目录汇总了 `week1` 和 `week2` 中所有需要使用 **ncu（Nsight Compute）** 和 **nsys（Nsight Systems）** 进行性能分析的任务，方便统一查阅和批量执行。

---

## 常用指标速查

| 指标 | 含义 | 健康参考 |
|------|------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算吞吐量占峰值比例 | > 60% 较好 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 内存吞吐量占峰值比例 | > 60% 较好 |
| `sm__occupancy.avg.pct_of_peak_sustained_elapsed` | 实际 occupancy | > 70% 较好 |
| `launch__registers_per_thread` | 每个线程使用的寄存器数 | 越少通常 occupancy 越高 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory load bank conflict 次数 | 越少越好 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | Shared memory store bank conflict 次数 | 越少越好 |
| `sm__cycles_elapsed.avg` | 平均执行 cycle 数 | 用于对比 |
| `smsp__average_warps_issue_stalled_long_scoreboard.pct` | Long Scoreboard stall 占比 | < 20% 较好 |

---

## Week 1

### Day 1 — GPU 执行模型与 `hello_gpu`

**目标**：理解 kernel launch 开销和线程/block 调度。

```bash
# 时间线分析
nsys profile -o profiles/day1_hello_gpu_timeline ./kernels/hello_gpu

# 基础硬件指标
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_elapsed ./kernels/hello_gpu
```

**观察重点**：`cudaLaunchKernel` CPU 时间、GPU 执行时间、block 并行度、active warp 比例。

---

### Day 2 — Occupancy 与寄存器约束

**目标**：理解寄存器用量如何限制 occupancy，以及如何减少寄存器压力。

```bash
ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread,\
  launch__shared_mem_per_block_dynamic,\
  launch__shared_mem_per_block_static \
  ./kernels/occupancy_test
```

**观察重点**：occupancy、active warps、registers/thread、dynamic/static shared memory。

**进阶**：手动修改 `#pragma unroll`、局部变量、`__launch_bounds__` 后重新编译并对比。

---

### Day 3 — Device Query / Occupancy Calculator

**无 explicit ncu/nsys 命令。**

内容聚焦硬件参数查询（`deviceQuery`、`cudaGetDeviceProperties`、峰值 FLOPs/BW）和 occupancy 计算，是后续 profiling 的前置知识。

---

### Day 4 — Memory Hierarchy / 矩阵转置

**目标**：对比 naive 与 shared-memory tiled transpose 的内存吞吐量和实际 DRAM 流量。

```bash
# 吞吐量与 cycle 对比
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
  sm__cycles_elapsed.avg \
  ./kernels/transpose

# 应用级时间线
nsys profile -o profiles/day4_transpose_timeline ./kernels/transpose

# L1/L2 与 DRAM 实际流量
ncu --metrics \
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
  dram__bytes_read.sum,\
  dram__bytes_write.sum \
  ./transpose
```

**观察重点**：DRAM throughput、global read/write bytes、elapsed cycles、真实 DRAM traffic。

---

### Day 5 — Bank Conflict

**目标**：观察 bank conflict 对性能和吞吐量的影响。

```bash
# 整体对比
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
  sm__cycles_elapsed.avg,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./kernels/bank_conflict

# 单独 kernel 精确对比
ncu --kernel-name regex:conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./kernels/bank_conflict

ncu --kernel-name regex:no_conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./kernels/bank_conflict
```

**观察重点**：load/store bank conflicts、cycles、throughput。

---

### Day 6 — Nsight 综合 Profiling 实战

**目标**：掌握完整 Nsight Compute 报告生成、GUI 查看、Nsight Systems 系统级 trace。

```bash
# 生成完整 ncu 报告
ncu --set full -o profiles/day6_hello_gpu ./kernels/hello_gpu
ncu --set full -o profiles/day6_occupancy_test ./kernels/occupancy_test
ncu --set full -o profiles/day6_transpose ./kernels/transpose
ncu --set full -o profiles/day6_bank_conflict ./kernels/bank_conflict

# GUI 打开
ncu-ui profiles/day6_transpose.ncu-rep

# 系统级 timeline
nsys profile -o profiles/day6_full_timeline --trace cuda,nvtx,osrt ./kernels/transpose
```

** Roofline / 瓶颈分类**：记录 `sm__throughput`、`dram__throughput`、`launch__occupancy`，判断 compute-bound 或 memory-bound。

---

### Day 7 — Week 1 Profiling 总结

**无新命令。**

任务：整理前 6 天数据，填写 `week1/profiles/week1_profile_summary.md`。

---

## Week 2

### Day 1 — Warp Shuffle / Block Reduce

**目标**：验证 Warp Shuffle 具有高 occupancy 和极低执行时间。

```bash
ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread \
  ./warp_reduce
```

**相关 LeetGPU**：Prefix Sum 建议用 ncu 对比不同参数性能差异。

---

### Day 2 — Register Blocking / 2D Tiling GEMM

**目标**：确认寄存器用量在限制内，并通过 ncu 调优 GEMM 参数。

```bash
# 编译时查看寄存器用量（非 ncu，但相关）
nvcc -Xptxas -v -o register_gemm kernels/register_blocking_gemm.cu -O3 -arch=sm_80 -lcublas
```

**相关 LeetGPU**：GEMM 建议用 ncu 对比不同参数性能差异。

---

### Day 3 — CUDA Streams / 异步执行

**目标**：验证多 stream 下 H2D/Compute/D2H 真正重叠。

```bash
nsys profile -o multi_stream_timeline ./multi_stream
```

**相关 LeetGPU**：Convolution 建议用 ncu 对比不同参数性能差异。

---

### Day 4 — Nsight Compute 性能分析

**目标**：掌握 ncu CLI、关键指标解读、定位瓶颈、CSV 导出、profile-optimize-verify 循环。

```bash
# 编译带 lineinfo
nvcc -o gemm_profile kernels/register_blocking_gemm.cu -O3 -arch=sm_80 -lcublas -g -lineinfo

# 详细 profile
ncu --kernel-name regex:gemmRegisterBlocking -o gemm_profile_report \
  --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
  smsp__warps_eligible.sum.per_cycle,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./gemm_profile 2>&1 | tee ncu_output.txt

# 导出 CSV
ncu --csv --page details -i gemm_profile_report.ncu-rep > gemm_profile.csv

# 优化后重新 profile
ncu --kernel-name regex:gemmRegisterBlocking -o gemm_profile_v2 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./gemm_profile_v2

# nsys 时间线（扩展）
nsys profile -o timeline_report ./gemm_profile
```

**关键 stall reason**：Long Scoreboard、Math Pipe Throttle、MIO Throttle、Wait、Barrier、LG Throttle。

**Softmax 参考命令**：

```bash
ncu --set full --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed \
  ./softmax
```

---

### Day 5 — FlashAttention 简化版

**目标**：判断 FlashAttention 是 compute-bound 还是 memory-bound，并与标准 attention 对比。

```bash
nvcc -o flash_attn_profile kernels/flash_attention.cu -O3 -arch=sm_80 -g -lineinfo

ncu --kernel-name regex:flashAttentionFwd \
  --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed \
  ./flash_attn_profile
```

**相关 LeetGPU**：Attention 建议用 ncu 对比不同参数性能差异。

---

### Day 6 — 整合优化 GEMM

**目标**：验证 `float4` + Warp Shuffle + coalesced write 后 SM throughput > 60%、Long Scoreboard < 20%。

```bash
nvcc -o gemm_profile integrated_gemm.cu -O3 -arch=sm_80 -lcublas -g -lineinfo

ncu --kernel-name regex:gemmIntegrated -o integrated_profile \
  --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./gemm_profile
```

**与 Day 2 Register Blocking 对比目标**：

| Metric | Day 2 Register Blocking | Day 6 Integrated target |
|--------|------------------------|------------------------|
| SM Throughput | ~45% | > 60% |
| Memory Throughput | ~78% | ~70–80% |
| Achieved Occupancy | ~56% | > 70% |
| Long Scoreboard Stall | ~35% | < 20% |

**相关 LeetGPU**：Histogram 建议用 ncu 分析 atomic 冲突、shared memory bank conflict、occupancy，对比 global atomic vs shared memory atomic。

---

### Day 7 — 限时手撕 + 性能报告

**无新命令。**

任务：撰写 `week2/day7/notes/performance_report.md`，用 ncu 指标解释每一层优化前后瓶颈的变化。

---

## 通用方法论

1. **先 nsys，后 ncu**：
   - `nsys` 找最耗时 kernel / 时间线问题。
   - `ncu` 深入分析 SM throughput、memory throughput、occupancy、stall reason。

2. **优化后重新验证**：
   - 修改代码 → 重新编译 → 重新 `ncu` → 对比指标。

3. **判断瓶颈**：
   - `sm__throughput` 高、`dram__throughput` 低 → compute-bound。
   - `dram__throughput` 高、`sm__throughput` 低 → memory-bound。
   - Roofline balance point（A100 约 25 FLOP/byte）可作参考。

---

## 快速命令模板

```bash
# 基础 profiling
ncu ./app

# 指定指标
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./app

# 生成完整报告
ncu --set full -o report ./app
ncu-ui report.ncu-rep

# 系统级 trace
nsys profile -o timeline ./app
nsys profile -o timeline --trace cuda,nvtx,osrt ./app

# 单独 kernel
ncu --kernel-name regex:<kernel_name> --metrics ... ./app

# CSV 导出
ncu --csv --page details -i report.ncu-rep > report.csv
```
