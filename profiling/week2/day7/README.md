# Week 2 Day 7 — 验收日 Profiling

> 对应 [Week 2 Day 7 任务 1（手撕 Reduce）+ 任务 2（手撕 GEMM）+ 任务 5（性能报告）](../../week2/day7/README.md)

Day 7 是验收日，通过限时手撕验证本周所学。本目录提供两个手撕 kernel 的可执行代码 + ncu profiling，用于验收时量化性能指标。

## 1. Block Reduce（任务 1：30 分钟手撕）

### 编译与运行

```bash
make block_reduce
./block_reduce
```

**预期输出**：

```text
=== Block Reduce (Warp Shuffle + 两级归约) ===
N = 4194304 (16.00 MB)

GPU Sum: 20973.xxxx
CPU Sum: 20973.xxxx
Diff:    0.00xxxx (PASS)
Time:    0.xxx ms (xx.xx GB/s bandwidth)
```

### ncu 分析

```bash
make profile-reduce       # 核心指标
make profile-reduce-full  # ncu --set full 完整报告
make nsys-reduce          # nsys 时间线
```

### 验收指标

| 指标 | 预期 | 解读 |
|------|------|------|
| `sm__occupancy` | **高**（>80%） | 寄存器/smem 用量极少 |
| `sm__throughput` | 中低（~20-40%） | 归约是 memory-bound |
| `dram__throughput` | **高**（>60%） | grid-stride loop 从 HBM 读 |
| `launch__registers_per_thread` | 少（~20） | shuffle 不消耗额外寄存器 |
| `smsp__...stalled_long_scoreboard.pct` | 高（>30%） | 等 HBM 加载 |

## 2. Register Blocking GEMM（任务 2：60 分钟手撕）

### 编译与运行

```bash
make gemm_timed
./gemm_timed
```

**预期输出**：

```text
=== Register Blocking GEMM (手撕验收) ===
BM=128 BN=128 BK=8 TM=8 TN=8 Threads=256

N        Our(ms)    cuBLAS(ms) GFLOPS    Percent
------------------------------------------------
512      0.xxx      0.xxx      xxxx.x    xx.x%  PASS
1024     0.xxx      0.xxx      xxxx.x    xx.x%  PASS
2048     x.xxx      x.xxx      xxxx.x    xx.x%  PASS
4096     xx.xxx     xx.xxx     xxxx.x    xx.x%  PASS
```

### ncu 分析

```bash
make profile-gemm       # 核心指标
make profile-gemm-full  # ncu --set full 完整报告
make nsys-gemm          # nsys 时间线
```

### 验收指标

| 指标 | 预期 | Day 6 整合版对比 | 解读 |
|------|------|-----------------|------|
| `sm__throughput` | ~40-50% | Day 6 >60% | 手撕版无 float4，SM 利用较低 |
| `dram__throughput` | ~70-80% | ~70-80% | 带宽利用接近 |
| `sm__occupancy` | ~50-60% | >70% | 手撕版寄存器用量未优化 |
| `smsp__...stalled_long_scoreboard.pct` | ~30% | <20% | 手撕版无向量化，等内存更多 |
| cuBLAS 百分比 | ~40-50% | ~70% | 手撕版与 Day 6 的差距即优化空间 |

### 性能差距分析（任务 5 报告素材）

手撕版 vs Day 6 整合版的差距来源：

| 优化点 | 手撕版 | Day 6 整合版 | 收益 |
|--------|--------|-------------|------|
| float4 向量化加载 | ❌ 逐元素 | ✅ 128-bit | ~1.5-2x |
| Coalesced 写回 | ❌ 逐元素 | ✅ float4 | ~1.2x |
| Warp Shuffle 辅助 | ❌ 无 | ✅ 写回优化 | ~1.1x |
| 总计 | ~45% cuBLAS | ~70% cuBLAS | ~1.5x |

> 💡 这正是 Day 7 验收的意义：手撕版暴露的差距就是 Day 3-6 优化的价值。用 ncu 量化每一项的指标差异，形成完整的"优化方法论闭环"。

## 3. 性能对比报告（任务 5）

用两个程序的输出 + ncu 指标，填充报告模板：

```markdown
| 版本 | 时间(ms) | GFLOPS | cuBLAS 百分比 | 关键优化点 |
|------|---------|--------|--------------|-----------|
| Naive (Day 2 基准) | | | ~1-3% | 无优化 |
| Register Blocking (手撕) | | | ~45% | + Register 累加器 |
| + float4 + Shuffle (Day 6) | | | ~70% | + 128-bit 加载 + 写回优化 |
| cuBLAS | | | 100% | NVIDIA 官方优化 |
```

ncu 瓶颈诊断记录：

```markdown
| 版本 | SM Throughput | DRAM Throughput | Long Scoreboard Stall | 瓶颈 |
|------|--------------|----------------|----------------------|------|
| 手撕 GEMM | ~45% | ~78% | ~35% | memory-bound (等 HBM) |
| Day 6 整合版 | >60% | ~75% | <20% | 趋向 compute-bound |
```
