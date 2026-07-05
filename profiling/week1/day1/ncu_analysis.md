# NCU 分析报告：hello_gpu

> 报告文件：`hello_gpu.ncu-rep`  
> 设备：NVIDIA GeForce RTX 5090 (CC 12.0)  
> 分析日期：2026-07-05

## 1. 如何打开 NCU 报告

### 命令行查看摘要
```bash
ncu --import hello_gpu.ncu-rep --print-summary per-kernel
```

### 图形界面分析
```bash
ncu-ui hello_gpu.ncu-rep
```

## 2. 关键指标总览

### GPU Speed Of Light Throughput

| 指标 | 数值 | 含义 |
|---|---|---|
| Duration | 40.77 µs | Kernel 实际执行时间 |
| Elapsed Cycles | 103,788 | 总经过周期 |
| SM Frequency | 2.54 GHz | SM 时钟频率 |
| Compute (SM) Throughput | 0.04% | SM 计算利用率极低 |
| Memory Throughput | 6.21% | 整体内存利用率极低 |
| DRAM Throughput | 0.04% | 显存带宽利用率极低 |
| L1/TEX Throughput | 3.27% | L1 缓存利用率极低 |
| L2 Throughput | 0.06% | L2 缓存利用率极低 |

**结论**：该 Kernel 既不是计算瓶颈，也不是显存带宽瓶颈。主要问题是启动配置（Launch Configuration）导致 SM 严重空闲。

## 3. Launch Statistics 分析

| 指标 | 数值 | 含义 |
|---|---|---|
| Grid Size | 4 | 只启动了 4 个 Block |
| Block Size | 8 | 每个 Block 仅 8 个线程 |
| Threads | 32 | 总线程数 |
| # SMs | 170 | RTX 5090 共有 170 个 SM |
| Waves Per SM | 0.00 | 每个 SM 连 1 个 Wave 都分不到 |
| Registers Per Thread | 24 | 每个线程使用 24 个寄存器 |
| Dynamic Shared Memory | 0 byte | 未使用动态共享内存 |
| Static Shared Memory | 0 byte | 未使用静态共享内存 |

NCU 给出的优化建议：

1. **Block Size 不是 32 的倍数**：Warp 固定为 32 线程，8 线程意味着 Warp 中 24 个线程被 mask 掉，造成硬件资源浪费。
2. **Grid 太小**：4 个 Block 远小于 170 个 SM，大量 SM 完全未被利用。

## 4. Occupancy 占用率分析

| 指标 | 数值 | 含义 |
|---|---|---|
| Theoretical Occupancy | 50.00% | 理论最大占用率 |
| Achieved Occupancy | 2.08% | 实际占用率 |
| Achieved Active Warps Per SM | 1.00 | 每个 SM 平均只有 1 个 Active Warp |
| Theoretical Active Warps per SM | 24.00 | 理论每个 SM 最多 24 个 Active Warp |

NCU 提示：理论占用率 50% 受限原因是 **每个 SM 能容纳的 Block 数量太少**，而实际 2.08% 说明 Kernel 启动规模过小，调度器没有足够 Warp 隐藏延迟。

## 5. 代码行为推测

从指标反推，`hello_gpu` 是一个非常简单的演示 Kernel：

- 几乎不做数学运算（Compute Throughput 0.04%）
- 几乎不访问全局内存（DRAM 0.04%）
- Block Size 为 8，Grid 为 2×2=4

这种 Kernel 通常用于理解 CUDA 执行模型（Grid/Block/Thread 层次），而非性能测试。

## 6. 优化建议

### 6.1 调整启动配置

将 Block Size 改为 32 的倍数，并让 Grid 足够大以覆盖所有 SM：

```cpp
// 推荐配置：128 或 256 线程 per block
hello_gpu<<<170, 128>>>();

// 或根据数据量计算 grid
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
hello_gpu<<<blocksPerGrid, threadsPerBlock>>>();
```

### 6.2 增加每个线程的工作量

如果 Kernel 只是 `printf` 或简单赋值，即使线程数再多也测不出带宽或计算瓶颈。通常需要让：

- 每个线程处理足够的数据量
- 总数据量达到几十 MB 以上，才能有效评估显存带宽
- 引入足够计算（如浮点运算）才能评估 SM 算力

## 7. ncu-ui 界面重点

使用 `ncu-ui hello_gpu.ncu-rep` 打开后，重点关注以下页面：

1. **Summary**：顶部瓶颈提示（SOL Bottleneck）
2. **GPU Speed Of Light Throughput**：判断是 compute-bound 还是 memory-bound
3. **Memory Workload Analysis**：查看 L1/L2/DRAM 命中率和传输量
4. **Launch Statistics**：Grid/Block 配置、寄存器占用、共享内存
5. **Occupancy**：理论/实际占用率
6. **Source/PTX/SASS**：将指标反汇编到具体指令行

右上角的 **Roofline 图** 可以一眼看出 Kernel 落在 compute roof 还是 memory roof 上。

## 8. 总结

`hello_gpu` 这个 Kernel 没有任何计算或显存压力，NCU 报告的核心价值在于验证 **Launch Configuration**。对于真正的性能分析练习，建议使用计算密集型的 `matmul.ncu-rep` 或内存密集型的 `transpose_tiles.ncu-rep`，它们更能体现 compute vs memory 的权衡。
