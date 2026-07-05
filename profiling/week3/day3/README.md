# Week 3 Day 3 — 源码分析优化对比 Profiling

> 对应 [Week 3 Day 3 任务 3（ncu 对比优化前后）+ 实验 2（warp vs block D-scan）+ 实验 3（stall 分析）](../../week3/day3/README.md)

Day 3 的核心是读 PyTorch/FasterTransformer 源码后，对 Day 16 的 kernel 做优化（warp 级 Softmax + float4 LayerNorm），用 ncu 量化优化收益。

## 1. ncu 对比优化前后（任务 3）

### 编译与运行

```bash
make softmax_layernorm_opt
./softmax_layernorm_opt
```

该程序包含 4 个 kernel：优化版（`softmax_warp_kernel` + `layernorm_float4_kernel`）和基准版（`softmax_block_kernel` + `layernorm_scalar_kernel`），自动计时 + 正确性验证 + speedup。

### ncu 分析

```bash
make profile-opt       # 优化版 kernel 指标
make profile-base      # 基准版 kernel 指标
make profile-all       # 全部 4 个 kernel 一次性 profile
make profile-full      # ncu --set full 完整报告
make nsys              # nsys 时间线
```

### 预期对比

| Kernel | DRAM Throughput | SM Throughput | Time | 观察 |
|--------|-----------------|---------------|------|------|
| `softmax_block_kernel`（Day16） | ~50-60% | ~15-20% | 基准 | memory-bound，带宽未喂饱 |
| `softmax_warp_kernel`（优化） | ~60-75% | ~15-22% | 更快 | DRAM 利用率提升（省了同步开销） |
| `layernorm_scalar_kernel`（Day16） | ~45-55% | ~12-18% | 基准 | 逐元素加载，指令多 |
| `layernorm_float4_kernel`（优化） | ~65-80% | ~18-25% | 更快 | DRAM 利用率明显提升（向量化） |

**关键观察**：float4 优化后 DRAM Throughput 明显上升，但 SM Throughput 变化不大——memory-bound kernel 优化的特征是**提升带宽利用率，不是算力利用率**。

## 2. warp vs block D-scan（实验 2）

### 编译与运行

```bash
make warp_vs_block
./warp_vs_block
```

**预期输出**：

```text
=== Softmax: warp-level vs block-level D-scan ===
M=1024

D        Block(ms)       Warp(ms)        Speedup
--------------------------------------------------------
256      0.xxxx          0.xxxx          x.xx
512      0.xxxx          0.xxxx          x.xx
1024     0.xxxx          0.xxxx          x.xx
2048     0.xxxx          0.xxxx          x.xx
4096     0.xxxx          0.xxxx          x.xx
```

### ncu 分析

```bash
make profile-dscan
```

### 观察要点

| D 值 | warp 级表现 | block 级表现 | 原因 |
|------|-----------|-------------|------|
| 256 | **快** | 慢 | warp 级无 `__syncthreads`，延迟低 |
| 1024 | **快** | 接近 | D=1024 是 PyTorch 的 dispatch 分界 |
| 4096 | 可能慢 | 可能反超 | 每 lane 处理 128 元素，并行度下降 |

**关键洞察**：PyTorch 在 D≤1024 时选 warp 级（无同步开销），D>1024 时回退 block 级（更多线程协作）。这就是 dispatch 分界的工程依据。

## 3. ncu stall 原因分析（实验 3）

```bash
make profile-stall
```

### 观察要点

| Kernel | Long Scoreboard Stall 预期 | Membar Stall 预期 | 解读 |
|--------|--------------------------|-------------------|------|
| `layernorm_scalar_kernel` | 高 | 中 | 逐元素 load，等内存次数多 |
| `layernorm_float4_kernel` | 可能略升 | 低 | 单次 load 更大但次数少，用长延迟换少次数 |

**关键洞察**：float4 版本的 Long Scoreboard stall 占比可能略升（单次等待更久），但总执行时间下降（等待次数少）。这是"用少量长延迟换大量短延迟"的权衡。

## ncu 指标解读

| 指标 | 含义 | 优化版预期变化 |
|------|------|-------------|
| `dram__throughput` | 显存带宽利用率 | **上升**（float4 向量化提升带宽利用） |
| `sm__throughput` | 计算单元利用率 | 基本不变（计算量没变） |
| `sm__occupancy` | SM 占用率 | warp 版可能略降（blockDim=128 vs 256） |
| `smsp__...stalled_long_scoreboard.pct` | 等内存 stall | 可能略升但总时间下降 |
| `gpu__time_duration.sum` | kernel 执行时间 | **下降**（优化有效） |

## 优化收益量化

```
Softmax:  warp 级 vs block 级 → 1.1-1.5x（省 __syncthreads）
LayerNorm: float4 vs scalar  → 1.3-2x（向量化加载，指令数 ÷4）
```

> 💡 这些优化都是"工程细节级"的——不改变算法，只改变数据加载和同步方式。真正的算法级优化（online softmax、Welford 一次 reduce）需要 Day 17 源码分析的深入理解。
