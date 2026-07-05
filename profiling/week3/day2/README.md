# Week 3 Day 2 — Softmax + LayerNorm Profiling

> 对应 [Week 3 Day 2 任务 3（ncu 验证 memory-bound）+ 实验 1（D-scan 尺度律）](../../week3/day2/README.md)

## 1. ncu 验证 memory-bound（任务 3）

### 编译与运行

```bash
make softmax_layernorm
./softmax_layernorm          # 正确性验证 + 计时
```

### ncu 分析

```bash
make profile          # 核心指标（SM/DRAM throughput + time）
make profile-stall    # 含 stall reasons（Long Scoreboard 等）
make profile-full     # ncu --set full 完整报告
make nsys             # nsys 时间线
```

### 观察要点

| Kernel | 预期 DRAM Throughput | 预期 SM Throughput | 判定 |
|--------|---------------------|-------------------|------|
| `softmax_kernel` | 50-70% | 15-25% | **Memory-bound**（DRAM >> SM） |
| `layernorm_kernel` | 50-70% | 15-25% | **Memory-bound**（DRAM >> SM） |

**关键洞察**：
- DRAM Throughput >> SM Throughput → memory-bound（AI ≈ 0.375 FLOP/Byte）
- Long Scoreboard stall 占比最高（等 HBM 加载）
- 如果 DRAM 未达 80%+，说明带宽未喂饱 → Day 17 向量化加载的提升空间

## 2. D-scan 性能尺度律（实验 1）

### 编译与运行

```bash
make sl_dscan
./sl_dscan          # 扫描 D=256/512/768/1024/2048/4096
```

**预期输出**：

```text
=== Softmax + LayerNorm D-scan (memory-bound scale law) ===
M=128, threads=256

D        SM time(ms)   LN time(ms)   SM BW(GB/s)   LN BW(GB/s)
------------------------------------------------------------------------
256      0.xxxx        0.xxxx        xxx.x         xxx.x
512      0.xxxx        0.xxxx        xxx.x         xxx.x
1024     0.xxxx        0.xxxx        xxx.x         xxx.x
4096     0.xxxx        0.xxxx        xxx.x         xxx.x
```

### ncu 分析 D-scan

```bash
make profile-dscan   # 对所有 D 值的 kernel 做 ncu 分析
```

### 观察要点

| D 值 | 时间变化 | 带宽变化 | 解读 |
|------|---------|---------|------|
| 256 → 512 | ~2x | 稳定 | D 翻倍 → Bytes 翻倍 → 时间翻倍 |
| 1024 → 2048 | ~2x | 稳定 | memory-bound 的线性尺度律 |
| 2048 → 4096 | ~2x | 稳定 | 时间 ≈ Bytes / Bandwidth |

**关键洞察**：
1. **D 翻倍 → 时间翻倍**：memory-bound kernel 的耗时 ≈ Bytes / Bandwidth，与 D 成正比
2. **带宽利用率相对稳定**：受 DRAM 峰值带宽限制，不随 D 变化
3. **reduce 次数不变**：仍是 warp shuffle 的固定 5 步，与 D 无关
4. **小 D 时带宽偏低**：L2 cache 命中率高 + launch overhead 占比大

## 3. ncu 指标解读

| 指标 | 含义 | Softmax/LayerNorm 预期 |
|------|------|----------------------|
| `dram__throughput` | 显存带宽利用率 | **高**（>50%），memory-bound 的标志 |
| `sm__throughput` | 计算单元利用率 | **低**（<25%），计算量极小 |
| `sm__occupancy` | SM 占用率 | 中高（>50%），寄存器/smem 用量少 |
| `smsp__...stalled_long_scoreboard.pct` | 等内存 stall 占比 | **高**（>30%），三遍扫描每次从 HBM 读 |
| `gpu__time_duration.sum` | kernel 执行时间 | 随 D 线性增长 |

## 瓶颈判定

```
DRAM Throughput >> SM Throughput  → memory-bound
Long Scoreboard stall 高           → 等 HBM 加载（三遍扫描的代价）
D 翻倍 → 时间翻倍                  → 线性尺度律（memory-bound 特征）
```

> 💡 这正是 Day 17 优化的起点：向量化加载（float4）能提升 DRAM Throughput，online softmax 能减少扫描次数（3→2），Welford 能减少 LayerNorm 的 reduce 次数（2→1）。
