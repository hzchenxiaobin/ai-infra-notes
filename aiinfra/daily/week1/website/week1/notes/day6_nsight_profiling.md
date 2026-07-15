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

Roofline 模型是一个用来快速判断 kernel 是 **memory-bound** 还是 **compute-bound** 的可视化工具。

#### Roofline 模型坐标轴含义

| 坐标轴 | 含义 | 单位 |
|--------|------|------|
| **横轴（X 轴）** | **Arithmetic Intensity（算术强度）** | FLOPs / byte |
| **纵轴（Y 轴）** | **Attainable FLOP/s（可达到的算力）** | FLOP/s 或 GFLOP/s |

**横轴 Arithmetic Intensity** 表示每读取 1 字节数据能进行多少次浮点运算：

```text
AI = FLOPs（浮点运算次数） / Bytes（内存读写字节数）
```

- AI 越小 → 计算少、访存多 → 越容易是 **memory-bound**
- AI 越大 → 计算多、访存少 → 越容易是 **compute-bound**

**纵轴 Attainable FLOP/s** 表示该 kernel 在当前硬件上实际能达到的算力上限，由两个天花板共同决定：

1. **Memory Bandwidth ceiling**：`Achievable FLOP/s = AI × Peak Bandwidth`（斜线）
2. **Peak Compute ceiling**：`Achievable FLOP/s = Peak FLOP/s`（水平线）

#### Ridge Point（山脊点 / 平衡点）

两条线的交点叫 Ridge Point：

```text
Ridge Point = Peak FLOP/s / Peak Bandwidth
```

以 RTX 5090 为例：

```text
Peak FP32 算力 ≈ 19.5 TFLOP/s
Peak HBM 带宽 ≈ 1.55 TB/s
Ridge Point ≈ 19.5 / 1.55 ≈ 12.6 FLOP/Byte
```

含义：**每读 1 byte 数据要做约 12.6 次浮点运算，才能打满 GPU 算力**。

- `AI < Ridge Point` → **Memory-bound**（算力喂不饱，瓶颈在带宽）
- `AI > Ridge Point` → **Compute-bound**（数据充足，瓶颈在算力）

#### 记录指标并判断瓶颈

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
