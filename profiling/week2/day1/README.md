# Week 2 Day 1 — Warp Reduce Profiling

> 对应 [Week 2 Day 1 任务 3：使用 ncu 查看 Warp Shuffle 效率](../../week2/day1/README.md)

## 编译与运行

```bash
make
./warp_reduce
```

**预期输出**：

```text
=== Warp Shuffle Block Reduce ===
Array size: 4194304 (16.00 MB)
GPU Sum: 20973.xxxxxx
CPU Sum: 20973.xxxxxx
Diff:    0.00xxxx (PASS)
Time:    0.xxx ms (xx.xx GB/s bandwidth)

=== ncu 分析命令 ===
ncu --metrics ...
```

## ncu 分析

### 基础指标（任务 3）

```bash
make profile
```

等价于：

```bash
ncu --metrics \
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct \
  --kernel-name regex:blockReduceSum \
  ./warp_reduce
```

### 完整报告

```bash
make profile-full    # ncu --set full，生成完整报告
```

### nsys 时间线

```bash
make nsys            # nsys profile，观察 kernel 时间线和间隙
```

## 观察要点

| 指标 | 含义 | Warp Reduce 预期 |
|------|------|-----------------|
| `sm__occupancy` | SM 占用率 | **高**（~80-100%），寄存器/smem 用量极少 |
| `sm__throughput` | 计算单元利用率 | **中低**（~20-40%），归约是 memory-bound |
| `launch__registers_per_thread` | 每线程寄存器数 | **极少**（~20），shuffle 不消耗额外寄存器 |
| `smsp__average_warps_issue_stalled_long_scoreboard.pct` | 等内存 stall 占比 | **高**（>30%），grid-stride loop 等 HBM 加载 |

### 关键洞察

1. **Occupancy 高**：Warp Shuffle 不消耗 shared memory（仅 `warpSums[32]` = 128B），寄存器用量少 → occupancy 接近峰值
2. **SM Throughput 不高**：归约本质是 memory-bound（AI ≈ 0.25 FLOP/Byte），计算单元不是瓶颈
3. **Long Scoreboard Stall 高**：grid-stride loop 从 HBM 读数据时 warp 等待，这是 memory-bound 的典型特征
4. **Shuffle 本身不产生 stall**：`__shfl_down_sync` 延迟 ~1-2 cycles，在 ncu 中几乎看不到相关 stall

### 对比实验建议

修改 `threadsPerBlock` 参数（128/256/512），观察 occupancy 和 bandwidth 变化：

```bash
# 修改 main() 中的 launchReduce(d_input, d_temp, d_output, n, 128)
# 重新编译运行，对比不同 block size 的 ncu 指标
```
