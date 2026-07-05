# Week 1 & Week 2 性能分析任务汇总

本目录汇总了 `week1` 和 `week2` 中所有需要使用 **ncu（Nsight Compute）** 和 **nsys（Nsight Systems）** 进行性能分析的任务，并放置了可直接编译执行的 `.cu` 源码。

**目录结构**：

```text
profiling/
├── README.md                    # 本文件：总览
├── example_analysis.md          # ncu 结果分析实例
└── week1/
    ├── day1/hello_gpu.cu        # Day 1 kernel
    ├── day2/occupancy_test.cu   # Day 2 kernel
    ├── day4/                    # Day 4 kernels + README（原有）
    ├── day5/bank_conflict.cu    # Day 5 kernel
    └── day6/
        ├── README.md            # Day 6 综合任务指引
        └── matrix_multiplication/  # Day 6 LeetGPU 题目
```

> **Week 2 代码现状**：`week2/day*/kernels/` 目录目前为空，README 中引用的 `warp_reduce.cu`、`register_blocking_gemm.cu`、`flash_attention.cu` 等源文件尚未创建。因此 Week 2 部分当前只能查看命令模板，无法直接执行；等后续源码补齐后可按相同模式放入 `profiling/week2/`。
>
> **分析实例**：参见 [`example_analysis.md`](example_analysis.md)，以 `bank_conflict` 为例演示如何逐步分析 ncu 输出。

---

## 环境准备与常见故障

### WSL2 下 `ncu` 报 `ERR_NVGPUCTRPERM`

在 WSL2 中，CUDA 驱动由 Windows 宿主提供，因此 `/etc/modprobe.d/nvidia.conf` 中的 `NVreg_RestrictProfilingToAdminUsers=0` 对 `ncu` **无效**。若执行 `ncu` 时出现：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

需要在 **Windows 宿主** 上开放 GPU 性能计数器权限。

**方法 1：NVIDIA Control Panel（推荐）**

1. 在 Windows 中打开 **NVIDIA Control Panel**。
2. 菜单栏选择 `Desktop -> Enable Developer Settings`。
3. 左侧导航选择 `Developer -> Manage GPU Performance Counters`。
4. 勾选 `Allow access to the GPU performance counter to all users`，点击 `Apply`。

**方法 2：Windows 注册表**

若无法打开 NVIDIA Control Panel，可在管理员 PowerShell 中执行：

```powershell
$path = 'HKLM:\SOFTWARE\NVIDIA Corporation\Global\NVTweak'
if (!(Test-Path $path)) { New-Item -Path $path -Force | Out-Null }
New-ItemProperty -Path $path -Name 'RmProfilingAdminOnly' -Value 0 -PropertyType DWord -Force
```

**生效**

完成上述任一设置后，在 WSL2 中执行：

```bash
wsl --shutdown
```

或重启 WSL2 实例，然后重新运行 `ncu`。

> **验证**：`nsys profile` 通常可以正常工作；只有需要 GPU Performance Counters 的 `ncu` 会触发此权限错误。

---

## 快速开始

每个 `week1/day*/` 目录下都有 `Makefile`，进入目录后执行：

```bash
cd profiling/week1/day1
make
./hello_gpu
```

即可运行对应程序，再按下方命令进行 `ncu` / `nsys` 分析。

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

### Day 1 — GPU 执行模型与 `hello_gpu` + Vector Add block size 对比

**目录**：`profiling/week1/day1/`

#### hello_gpu（线程层次验证）

```bash
cd profiling/week1/day1
make hello_gpu
./hello_gpu

# nsys timeline
nsys profile -o hello_gpu_timeline ./hello_gpu

# ncu 基础指标
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_elapsed ./hello_gpu
```

**观察重点**：`cudaLaunchKernel` CPU 时间、GPU 执行时间、block 并行度、active warp 比例。

#### Vector Add：不同 block size 性能对比

> 对应 [Week 1 Day 1 LeetGPU Vector Add](../week1/day1/README.md)，用 ncu 对比 block_size = 32/64/128/256/512/1024 的性能差异。

```bash
make vector_add_blocksize
./vector_add_blocksize          # 打印各 block_size 的 time + bandwidth

make profile                    # ncu 分析各 block_size 的 DRAM/SM/occupancy
```

**观察重点**：
- `dram__throughput`：memory-bound kernel 的核心指标，应 > 60%（block_size ≥ 128 时）
- `sm__occupancy`：block_size=32 时低（1 warp/block），256 时高
- **关键结论**：block_size=256 通常最优；memory-bound kernel 不需 100% occupancy

详见 [`profiling/week1/day1/README.md`](week1/day1/README.md)。

---

### Day 2 — Occupancy 与寄存器约束

**目录**：`profiling/week1/day2/`

```bash
cd profiling/week1/day2
make
./occupancy_test

ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread,\
  launch__shared_mem_per_block_dynamic,\
  launch__shared_mem_per_block_static \
  ./occupancy_test
```

**观察重点**：occupancy、active warps、registers/thread、dynamic/static shared memory。

**进阶**：手动修改 `#pragma unroll`、局部变量、`__launch_bounds__` 后重新 `make` 并对比。

---

### Day 3 — Device Query / Occupancy Calculator

**无 explicit ncu/nsys 命令。**

内容聚焦硬件参数查询（`deviceQuery`、`cudaGetDeviceProperties`、峰值 FLOPs/BW）和 occupancy 计算，是后续 profiling 的前置知识。

---

### Day 4 — Memory Hierarchy / 矩阵转置

**目录**：`profiling/week1/day4/`（原有目录，包含 `transpose.cu`、`bandwidth.cu`、`transpose_tiles.cu`）

```bash
cd profiling/week1/day4
make
./transpose

# 吞吐量与 cycle 对比
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
  sm__cycles_elapsed.avg \
  ./transpose

# 应用级时间线
nsys profile -o transpose_timeline ./transpose

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

**目录**：`profiling/week1/day5/`

```bash
cd profiling/week1/day5
make
./bank_conflict

# 整体对比
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
  sm__cycles_elapsed.avg,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./bank_conflict

# 单独 kernel 精确对比
ncu --kernel-name regex:conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./bank_conflict

ncu --kernel-name regex:no_conflict_read \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__cycles_elapsed.avg \
  ./bank_conflict
```

**观察重点**：load/store bank conflicts、cycles、throughput。

---

### Day 6 — Nsight 综合 Profiling 实战

**目录**：`profiling/week1/day6/`

本目录不复制 kernel，而是复用前面几天的可执行文件，直接采集完整报告。

```bash
cd profiling/week1/day6
make -C ../day1
make -C ../day2
make -C ../day4
make -C ../day5

# 生成完整 ncu 报告
ncu --set full -o day6_hello_gpu ../day1/hello_gpu
ncu --set full -o day6_occupancy_test ../day2/occupancy_test
ncu --set full -o day6_transpose ../day4/transpose
ncu --set full -o day6_bank_conflict ../day5/bank_conflict

# GUI 打开
ncu-ui day6_transpose.ncu-rep

# 系统级 timeline
nsys profile -o day6_full_timeline --trace cuda,nvtx,osrt ../day4/transpose
```

** Roofline / 瓶颈分类**：记录 `sm__throughput`、`dram__throughput`、`launch__occupancy`，判断 compute-bound 或 memory-bound。

---

### Day 7 — Week 1 Profiling 总结

**无新命令。**

任务：整理前 6 天数据，填写 `week1/profiles/week1_profile_summary.md`。

---

## Week 2

### Day 1 — Warp Shuffle / Block Reduce

**目录**：`profiling/week2/day1/`

> 对应 [Week 2 Day 1 任务 3：使用 ncu 查看 Warp Shuffle 效率](../week2/day1/README.md)

**目标**：验证 Warp Shuffle 具有高 occupancy 和极低执行时间。

```bash
cd profiling/week2/day1
make
./warp_reduce          # 正确性验证 + 计时

make profile           # ncu 基础指标
make profile-full      # ncu 完整报告
make nsys              # nsys 时间线
```

**观察重点**：
- `sm__occupancy`：高（~80-100%），寄存器/smem 用量极少
- `sm__throughput`：中低（~20-40%），归约是 memory-bound
- `launch__registers_per_thread`：极少（~20），shuffle 不消耗额外寄存器
- `smsp__average_warps_issue_stalled_long_scoreboard.pct`：高（>30%），grid-stride loop 等 HBM

详见 [`profiling/week2/day1/README.md`](week2/day1/README.md)。

---

### Day 2 — Register Blocking / 2D Tiling GEMM

**目标**：确认寄存器用量在限制内，并通过 ncu 调优 GEMM 参数。

```bash
# 编译时查看寄存器用量（非 ncu，但相关）
nvcc -Xptxas -v -o register_gemm kernels/register_blocking_gemm.cu -O3 -arch=sm_120 -lcublas
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
nvcc -o gemm_profile kernels/register_blocking_gemm.cu -O3 -arch=sm_120 -lcublas -g -lineinfo

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

**目录**：`profiling/week2/day5/`

> 对应 [Week 2 Day 5 实验 3：用 ncu 分析 FlashAttention Kernel](../week2/day5/README.md)

**目标**：判断 FlashAttention 是 compute-bound 还是 memory-bound，并与标准 Attention 对比 HBM 读写量。

```bash
cd profiling/week2/day5
make
./flash_attention          # 运行 FlashAttention + Standard Attention 对比

make profile-flash         # ncu 分析 FlashAttention
make profile-standard      # ncu 分析 Standard Attention（对比）
make profile-hbm           # HBM 读写量对比（dram__bytes_read/write）
make profile-full          # ncu 完整报告
make nsys                  # nsys 时间线
```

**观察重点**：
- FlashAttention：SM Throughput > DRAM Throughput → **compute-bound**（IO 从 O(N²) 降到 O(Nd)）
- Standard Attention：DRAM Throughput >> SM Throughput → **memory-bound**（O(N²) 物化 S/P）
- `dram__bytes_read/write`：FlashAttention 远少于 Standard（核心加速来源）
- `sm__occupancy`：FlashAttention 较低（32KB smem/block），但整体更快

详见 [`profiling/week2/day5/README.md`](week2/day5/README.md)。

---

### Day 6 — 整合优化 GEMM + Histogram

**目录**：`profiling/week2/day6/`

> 对应 [Week 2 Day 6 任务 3（ncu 验证 GEMM）+ 任务 4（Histogram）](../week2/day6/README.md)

#### 整合版 GEMM

```bash
cd profiling/week2/day6
make integrated_gemm
./integrated_gemm          # 对比 cuBLAS，目标 70%+

make profile-gemm          # ncu 核心指标（SM/DRAM throughput, registers, stall）
make profile-gemm-full     # ncu 完整报告
make nsys-gemm             # nsys 时间线
```

**与 Day 2 Register Blocking 对比目标**：

| 指标 | Day 2 Register Blocking | Day 6 整合版 目标 |
|------|------------------------|-----------------|
| `sm__throughput` | ~45% | **> 60%** |
| `dram__throughput` | ~78% | ~70-80% |
| `sm__occupancy` | ~56% | **> 70%** |
| `smsp__...stalled_long_scoreboard.pct` | ~35% | **< 20%** |

#### Histogram（Global atomic vs Shared memory）

```bash
make histogram
./histogram                # 对比两个版本的 latency

make profile-hist-global   # Global atomic ncu 指标
make profile-hist-shared   # Shared memory ncu 指标
make profile-hist-hbm      # HBM 读写量对比
```

**观察重点**：shared 版 `dram__bytes_write` 远少于 global 版（smem privatization 减少 global atomic 写放大）。

详见 [`profiling/week2/day6/README.md`](week2/day6/README.md)。

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
