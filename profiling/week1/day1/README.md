# Week 1 Day 1 — hello_gpu & Vector Add Profiling

> 若在 WSL2 中运行 `ncu` 遇到 `ERR_NVGPUCTRPERM`，请参考 [`profiling/README.md`](../README.md) 中的"环境准备与常见故障"章节，在 Windows 宿主开放 GPU Performance Counters 权限。

## 1. hello_gpu（线程层次验证）

```bash
make hello_gpu
./hello_gpu

# nsys timeline
nsys profile -o hello_gpu_timeline ./hello_gpu

# ncu 基础指标
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_elapsed ./hello_gpu
```

## 2. Vector Add：不同 block size 性能对比

> 对应 [Week 1 Day 1 LeetGPU Vector Add 题目](../../week1/day1/README.md)，用 ncu 对比不同 block size 的性能差异。

Vector Add 是典型 memory-bound kernel（AI ≈ 0.083 FLOP/Byte），性能几乎完全取决于**显存带宽利用率**。通过对比不同 block size，可以直观看到 occupancy 与带宽的关系。

### 编译与运行

```bash
make vector_add_blocksize
./vector_add_blocksize
```

**预期输出**：

```text
=== Vector Add: block size 性能对比 ===
N = 1048576 (4.0 MB per array)

block_size   grid_size    time(ms)       bandwidth(GB/s)
--------------------------------------------------------
32           32768        0.xxxx         xxx.x
64           16384        0.xxxx         xxx.x
128          8192         0.xxxx         xxx.x
256          4096         0.xxxx         xxx.x
512          2048         0.xxxx         xxx.x
1024         1024         0.xxxx         xxx.x

正确性: PASS
```

### 用 ncu 分析

```bash
make profile
```

或手动运行：

```bash
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread \
  --kernel-name regex:vector_add \
  ./vector_add_blocksize
```

### 观察要点

| block_size | 预期 DRAM Throughput | 预期 Occupancy | 解读 |
|-----------|---------------------|---------------|------|
| 32 | ~20-40% | 低（1 warp/block） | 太小，SM 上 block 数受上限限制 |
| 128 | ~60-80% | 中 | 较好，足够隐藏延迟 |
| 256 | ~70-90% | 高 | **通常最优**，平衡 occupancy 与资源 |
| 512 | ~70-90% | 高 | 与 256 接近，可能略差（register 压力） |
| 1024 | ~60-80% | 可能下降 | block 太大，每 SM block 数减少 |

**关键洞察**：

1. **memory-bound kernel 不需 100% occupancy**：block_size=256 通常已达到 70%+ 带宽利用率，再增大 block_size 收益递减
2. **block_size=32 是反例**：每 block 只有 1 个 warp，SM 上虽可放很多 block，但 warp 数不足隐藏访存延迟
3. **block_size=1024 可能变差**：占用寄存器/shared memory 多，每 SM 能放的 block 数减少，occupancy 反而下降
4. **最优 block_size 通常在 128-512**：这是 CUDA kernel 的经验法则

### ncu 指标解读

| 指标 | 含义 | Vector Add 预期 |
|------|------|----------------|
| `dram__throughput` | 显存带宽利用率 | **高**（memory-bound 的标志，应 > 60%） |
| `sm__throughput` | 计算单元利用率 | **低**（计算量极小，应 < 20%） |
| `sm__occupancy` | SM 占用率 | 随 block_size 变化，256 时通常 > 50% |
| `launch__registers_per_thread` | 每线程寄存器数 | 应很小（~10），Vector Add 无寄存器压力 |

> 💡 **对比方法**：ncu 会对每次 kernel launch 分别报告。观察不同 block_size 下 `dram__throughput` 的变化，找到带宽利用率最高的配置。
