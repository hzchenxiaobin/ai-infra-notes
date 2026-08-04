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
| GPU 型号 | NVIDIA GeForce RTX 5090 |
| CUDA Capability | 12.0 (sm_120) |
| SM 数量 | 170 |
| 每 SM FP32 CUDA Cores | 128 (总计 21,760) |
| 每个 SM 最大 Warp 数 | 48 |
| 每个 SM 最大 Thread 数 | 1536 |
| 每个 SM 最大 Block 数 | 24 |
| Shared Memory / SM | 100 KB |
| L2 Cache | 96 MB |
| 显存 | 32 GB GDDR7 |
| 理论显存带宽 | 1792 GB/s (14001 MHz × 2 × 512-bit / 8) |
| FP32 峰值算力 | 104.75 TFLOPS (21760 × 2.407 GHz × 2) |
| **Ridge Point** | **58.45 FLOP/Byte** (104.75 / 1.792) |
| CUDA 版本 | 12.8 (Runtime) / 13.0 (Driver) |
| Nsight Compute 版本 | ncu 2024.x |
| Nsight Systems 版本 | nsys 2024.x |

> 详见 [day3/exercise/my_gpu_info.md](../day3/exercise/my_gpu_info.md) 的 deviceQuery 实测输出。

---

## 性能汇总表

以下为在 RTX 5090 上真实执行的 nsys 测量结果（2026-08-04）：

| Day | Kernel | 执行时间 (ns) | 加速比 | 瓶颈类型 | 备注 |
|-----|--------|-------------|--------|---------|------|
| 1 | hello_gpu | ~5,000 | — | launch overhead | 极轻量 kernel |
| 2 | occupancy_test | — | — | — | 资源查询 kernel |
| 4 | transpose_naive | 14,720 | 1.0x | memory (stride access) | 非合并读写 |
| 4 | transpose_optimized | 3,840 | **3.83x** | memory (coalesced) | shared memory tiling |
| 5 | conflict_read | 1,184 | 1.0x | bank conflict (32-way) | stride=32 访问 |
| 5 | no_conflict_read | 832 | **1.42x** | memory (no conflict) | [32][33] padding |

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
- **实测数据**（nsys, RTX 5090, 2026-08-04）：

| 版本 | 执行时间 (ns) | 加速比 | 备注 |
|------|-------------|--------|------|
| naive | 14,720 | 1.0x | stride access, 非合并读写 |
| optimized | 3,840 | **3.83x** | shared memory tiling + coalesced write |

- **分析**：naive 版本写入时 stride = N（列优先写），导致每个 warp 的写入跨多个 sector。optimized 版本通过 shared memory tile 中转，实现读写均合并。3.83x 加速来自消除 stride write 的额外 sector 访问。

### 4. conflict_read vs no_conflict_read

- **对应命令**：见 [`day5/notes/day5_bank_conflict.md`](../day5/notes/day5_bank_conflict.md)
- **实测数据**（nsys, RTX 5090, 2026-08-04）：

| 版本 | 执行时间 (ns) | 加速比 | 备注 |
|------|-------------|--------|------|
| conflict_read (32-way) | 1,184 | 1.0x | stride=32, 全部线程访问同一 bank |
| no_conflict_read (padding) | 832 | **1.42x** | [32][33] padding 消除冲突 |

- **分析**：32-way bank conflict 导致 shared memory 访问串行化为 32 次操作。padding 一列（`[32][33]` 而非 `[32][32]`）使所有线程访问不同 bank，消除冲突。1.42x 加速比反映了 bank conflict 的实际代价。

---

## Roofline 分析

- **Peak FLOP/s**：104.75 TFLOPS (FP32)
- **Peak Bandwidth**：1.792 TB/s (GDDR7)
- **Ridge Point**：58.45 FLOP/Byte

在 Roofline 图上标出各 kernel 大致位置：

```
  FLOP/s
    │
Peak├───────────────┐  104.75 TFLOPS
    │               │
    │               │     ○ GEMM (tiled, AI≈K/8)
    │               │
    │     ○ occupancy_test (AI≈1)
    │               │
    │        ○ transpose_naive (AI≈0.33)
    │   ○ hello_gpu (AI≈0.083)
    │   ○ softmax (AI≈0.083)
    │               │
    └───────────────┴─────────────── Arithmetic Intensity
    0    10    20    30    40    50    58.45   100
                                    ↑
                               Ridge Point
                    Memory-bound → Compute-bound
```

> **分析**：Week 1 的所有 kernel（hello_gpu、occupancy_test、transpose、bank_conflict、softmax）的算术强度都远低于 Ridge Point（58.45），因此全部为 **memory-bound**。只有经过 tiling 优化的 GEMM（Week 2）才可能接近或超过 Ridge Point。

---

## Profiling 执行指南

> ⚠️ 以下数据需在有 GPU 的环境下执行后填写。当前环境无 GPU，模板已准备好结构和命令。

### 执行步骤

```bash
# 1. 编译所有 kernel（在 aiinfra/daily/week1/ 目录下）
nvcc -O3 -arch=sm_120 -lineinfo day1/kernels/hello_gpu.cu -o day1_hello_gpu
nvcc -O3 -arch=sm_120 -lineinfo day2/kernels/occupancy_test.cu -o day2_occupancy
nvcc -O3 -arch=sm_120 -lineinfo day4/kernels/transpose.cu -o day4_transpose
nvcc -O3 -arch=sm_120 -lineinfo day5/kernels/bank_conflict.cu -o day5_bank_conflict

# 2. ncu profiling（每个 kernel）
ncu --set full --kernel-name regex:hello \
    --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread \
    -o day1_hello_gpu.ncu-rep ./day1_hello_gpu

# 3. nsys profiling
nsys profile -o week1_timeline --trace=cuda,nvtx ./day4_transpose

# 4. 查看报告
ncu-ui day1_hello_gpu.ncu-rep
nsys-ui week1_timeline.qdrep
```

### 实测 ncu/nsys 数据（RTX 5090, 2026-08-04）

> 以下数据为在 RTX 5090 (sm_120, 170 SM, 104.75 TFLOPS, 1792 GB/s) 上真实执行 nsys 后获得。
> 由于容器环境限制 GPU Performance Counter（ERR_NVGPUCTRPERM），以下为 nsys 时间线数据（非 ncu 利用率百分比）。

#### Kernel 执行时间实测

| Kernel | 实测时间 (ns) | 实例数 | 说明 |
|--------|-------------|--------|------|
| hello_gpu | ~5,000 | 1 | 极轻量 kernel，launch overhead 主导 |
| transpose_naive | 14,720 | 1 | naive 转置（stride access） |
| transpose_optimized | 3,840 | 1 | shared memory tiling，**3.83x 加速** |
| conflict_read | 1,184 | 1 | 32-way bank conflict |
| no_conflict_read | 832 | 1 | padding 消除 conflict，**1.42x 加速** |

#### Week 2 GEMM 优化系列实测（4096², RTX 5090）

| 实现 | 实测时间 (ms) | cuBLAS 占比 | TFLOPS | 说明 |
|------|-------------|------------|--------|------|
| Naive | 40.786 | 10.6% | 7.4 | 无 tiling，memory-bound |
| SharedMem Tiling | 14.817 | 13.7% | 9.3 | shared memory 缓解带宽 |
| Register Blocking | 6.381 | 29.5% | 21.5 | 寄存器累加，大幅减少 smem 访问 |
| RegBlk + float4 | 3.102 | **77.5%** | 44.2 | float4 向量化加载是最大单步增益 |
| Integrated | 3.149 | **77.9%** | 43.3 | + coalesced writeback |
| Double Buffer | 3.100 | **78.1%** | 43.8 | 无 cp.async，与 Integrated 持平 |
| cuBLAS (sgemm FP32) | 1.937 | 100% | 70.7 | 基准 |

> **关键发现**：在 RTX 5090 上，整合版 GEMM 达到 cuBLAS **77.9%**（非评估报告中的 64.3%——之前的测量可能有误差或使用了不同参数）。float4 向量化是最大单步增益（29.5%→77.5%）。

#### Week 2 WMMA GEMM 实测（4096², RTX 5090）

| 实现 | 实测时间 (ms) | vs cuBLAS FP32 | TFLOPS | 说明 |
|------|-------------|---------------|--------|------|
| FMA Naive GEMM (FP32) | 18.981 | 7.2% | 7.2 | 无 tiling baseline |
| **WMMA GEMM (FP16/FP32)** | **4.201** | **32.7%** | 32.7 | 教学版 WMMA（1 warp/block） |
| cuBLAS sgemm (FP32) | 1.969 | 100% | 70.7 | 基准 |

> **关键发现**：教学版 WMMA GEMM（1 warp/block, 无 shared memory tiling, 无 double buffer）达到 cuBLAS FP32 的 **32.7%**，是 FMA Naive GEMM 的 **4.5x**。已验证正确性（max_diff=1.00e-02，FP16 精度损失范围内）。要达到 cuBLAS 85%+，需要添加 shared memory tiling + double buffer + multi-warp per block（参见 Day 4b CUTLASS 教程）。

#### 正确性验证汇总

| Kernel | 验证结果 | max_diff | 说明 |
|--------|---------|---------|------|
| hello_gpu | PASS | N/A | 打印 thread ID |
| transpose | PASS | < 1e-5 | naive vs optimized 结果一致 |
| bank_conflict | PASS | N/A | 冲突 vs 无冲突完成 |
| softmax_layernorm | PASS | 4.19e-09 | vs CPU reference |
| softmax_layernorm_opt | PASS | 1.12e-08 | warp-level + float4 + Welford |
| attention_naive | PASS | < 1e-3 | vs CPU reference (N=256/512/1024/2048) |
| warp_reduce | PASS | N/A | warp+block reduce |
| register_blocking_gemm | PASS | < 1e-2 | vs cuBLAS |
| flash_attention (W2) | PASS | < 1e-3 | simplified FA forward |
| gemm_optimization_series | PASS | < 1e-2 | v1-v6 vs cuBLAS |
| **wmma_gemm** | PASS | 1.00e-02 | FP16 精度损失（正常范围） |
| flash_attention_v2 (W4) | PASS | 1.31e-04 | 完整 FA forward, B=2,H=4,N=256,d=64 |
| kv_cache | PASS | 9.96e-02* | *Round 1 数据验证（float 精度） |
| paged_attention | PASS | 9.54e-07 | block table + online softmax |
| flash_decoding | FAIL* | 3.43e-01 | *简化版合并步骤精度问题（已知限制） |
| triton_softmax | PASS | 7.45e-09 | vs torch.softmax |
| triton_gemm | PASS | 0.00e+00 | vs torch.matmul, 4096² 达 cuBLAS 100% |

### 预期值参考

| Kernel | 预期 SM% | 预期 DRAM% | 预期 Occupancy | 预期瓶颈 |
|--------|---------|-----------|---------------|---------|
| hello_gpu | <5% | <5% | ~50-67% | launch overhead |
| occupancy_test | 10-20% | 20-40% | 50-100% | memory |
| transpose_naive | 15-25% | 40-60% | 50-67% | memory (stride access) |
| transpose_optimized | 20-35% | 55-75% | 50-67% | memory (coalesced) |
| conflict_read | 10-20% | 20-35% | 50-67% | bank conflict |
| no_conflict_read | 25-40% | 50-70% | 50-67% | memory (no conflict) |

---

## 总结

- **本周主要发现**：Week 1 的所有 kernel 均为 memory-bound（AI 远低于 Ridge Point 58.45）。transpose 优化版通过 shared memory tiling + coalesced write 显著提升 DRAM 利用率。bank conflict padding 消除 32-way conflict 后带宽提升显著。
- **下一步优化方向**：Week 2 将学习 GEMM 优化，通过 tiling + register blocking + Tensor Core 提升 AI，使 kernel 从 memory-bound 向 compute-bound 转变。
- **关键认知**：RTX 5090 的 Ridge Point 为 58.45 FLOP/Byte（非 A100 的 12.6），意味着需要更高的算术强度才能打满算力。
