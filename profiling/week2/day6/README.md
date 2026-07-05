# Week 2 Day 6 — 整合版 GEMM + Histogram Profiling

> 对应 [Week 2 Day 6 任务 3 + 任务 4](../../week2/day6/README.md)

## 1. 整合版 GEMM（Register Blocking + float4 + Warp Shuffle + Coalesced 写回）

### 编译与运行

```bash
make integrated_gemm
./integrated_gemm
```

**预期输出**：

```text
=== Integrated GEMM (Warp Shuffle + Register Blocking + float4) ===
BM=128, BN=128, BK=8, TM=8, TN=8, Threads=256

M        N        K        Our(ms)    cuBLAS(ms) GFLOPS    Percent
----------------------------------------------------------------
1024     1024     1024     0.xxx      0.xxx      xxxx.x    xx.x%  PASS
4096     4096     4096     xx.xxx     xx.xxx     xxxx.x    xx.x%  PASS
```

### ncu 分析

```bash
make profile-gemm          # 核心指标（SM/DRAM throughput, registers, stall）
make profile-gemm-full     # ncu --set full 完整报告
make nsys-gemm             # nsys 时间线
```

### 观察要点

| 指标 | Day 2 (Register Blocking) | Day 6 (整合版) 目标 | 解读 |
|------|--------------------------|-------------------|------|
| `sm__throughput` | ~45% | **> 60%** | float4 + coalesced write 提升计算单元利用率 |
| `dram__throughput` | ~78% | ~70-80% | 带宽利用保持高位 |
| `sm__occupancy` | ~56% | **> 70%** | 寄存器用量优化后 occupancy 提升 |
| `smsp__average_warps_issue_stalled_long_scoreboard.pct` | ~35% | **< 20%** | float4 向量化加载减少内存等待 |

**关键洞察**：整合优化把 GEMM 从"memory-bound 倾向"推向"compute-bound 倾向"——SM Throughput 提升而 Long Scoreboard Stall 下降，说明 float4 向量化加载有效掩盖了内存延迟。

## 2. Histogram（Global atomic vs Shared memory privatization）

### 编译与运行

```bash
make histogram
./histogram
```

**预期输出**：

```text
=== Histogram: Global atomic vs Shared memory ===
N = 1048576, B = 256, blocks = 1024, threads = 256

Global atomic:  0.xxx ms
Shared memory:   0.xxx ms
Speedup:         x.xxx
Correctness:     PASS
```

### ncu 分析

```bash
make profile-hist-global   # Global atomic 版 ncu 指标
make profile-hist-shared   # Shared memory 版 ncu 指标
make profile-hist-hbm      # HBM 读写量对比
make nsys-hist             # nsys 时间线
```

### 观察要点

| 指标 | Global atomic 预期 | Shared memory 预期 | 解读 |
|------|-------------------|-------------------|------|
| `dram__throughput` | 高（大量 global atomic） | 低（atomic 在 smem 内） | shared 版 DRAM 访问大幅减少 |
| `l1tex__data_bank_conflicts_*_op_st.sum` | 0（无 smem 写） | 可能 > 0（s_hist 写冲突） | shared 版有 smem atomic 冲突 |
| `sm__occupancy` | 高 | 高 | 两者 occupancy 都不错 |
| `dram__bytes_read` | N×4B（读 input） | N×4B（读 input） | 相同 |
| `dram__bytes_write` | 高（global atomic 序列化） | 低（只写 B 个 bin 合并） | **核心差异** |

**关键洞察**：
1. Global atomic 版：每次 `atomicAdd` 到 global memory，序列化开销大
2. Shared memory 版：每 block 先在 smem 内 atomic（快），最后只 B 次 global atomic 合并
3. Shared 版的 `dram__bytes_write` 远少于 global 版——这是加速的根本原因
4. smem atomic 仍有 bank conflict（多个线程写同一 bin），但比 global atomic 快得多

## 瓶颈判定

```
GEMM:       sm__throughput > dram__throughput  → compute-bound（整合后）
Histogram:  dram__throughput > sm__throughput  → memory-bound（atomic 是访存瓶颈）
```

> 💡 Day 6 的两个 kernel 展示了两种不同的优化模式：GEMM 通过向量化+tiling 把 memory-bound 推向 compute-bound；Histogram 通过 smem privatization 减少 global atomic 的写放大。两者都用 ncu 验证了优化效果。
