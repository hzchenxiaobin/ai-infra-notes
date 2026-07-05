# Week 2 Day 5 — FlashAttention Profiling

> 对应 [Week 2 Day 5 实验 3：用 ncu 分析 FlashAttention Kernel](../../week2/day5/README.md)

## 编译与运行

```bash
make
./flash_attention
```

**预期输出**：

```text
=== FlashAttention Simplified Forward ===
Config: N=256, D=64, batch=1, heads=1
SRAM usage per block: 32.00 KB
Grid: (4, 1, 1), Block: (64, 4)
FlashAttention GPU Time: 0.xxx ms
Result check: PASS
Standard Attention GPU Time: 0.xxx ms
Result check: PASS
Speedup: x.xxx

=== ncu 分析命令 ===
...
```

## ncu 分析

### FlashAttention 指标

```bash
make profile-flash
```

### Standard Attention 对比

```bash
make profile-standard
```

### HBM 读写量对比

```bash
make profile-hbm
```

### 完整报告 + nsys 时间线

```bash
make profile-full    # ncu --set full，生成 .ncu-rep
make nsys            # nsys 时间线
```

## 观察要点

### FlashAttention vs Standard Attention 指标对比

| 指标 | FlashAttention 预期 | Standard Attention 预期 | 解读 |
|------|---------------------|------------------------|------|
| `dram__throughput` | 中（~30-50%） | 高（~60-80%） | FlashAttention HBM 访问少（O(Nd)），Standard 多（O(N²)） |
| `sm__throughput` | 中高（~40-60%） | 低（~20-30%） | FlashAttention 计算占比更高（SRAM 内做 GEMM） |
| `sm__occupancy` | 中（~40-60%） | 高（~70-90%） | FlashAttention 用大量 shared memory（32KB/block），限制 block 数 |
| `dram__bytes_read` | 少（~Nd） | 多（~N²） | 核心差异：FlashAttention 不物化 S/P |
| `dram__bytes_write` | 少（~Nd） | 多（~N²） | Standard 写 S/P 到 HBM，FlashAttention 只写 O |

### 关键洞察

1. **FlashAttention 是 compute-bound**：SM Throughput > DRAM Throughput，因为计算从 HBM 搬到了 SRAM
2. **Standard Attention 是 memory-bound**：DRAM Throughput >> SM Throughput，O(N²) 读写主导
3. **HBM 读写量差异是加速的根本原因**：FlashAttention O(Nd) vs Standard O(N²)
4. **Occupancy 权衡**：FlashAttention 用 32KB shared memory/block 换取 HBM 访问减少，occupancy 降低但整体更快
5. **N 越大优势越明显**：N 翻倍时 Standard 的 HBM IO 变 4x，FlashAttention 只变 2x

### 瓶颈判定

```
FlashAttention:  sm__throughput > dram__throughput  → compute-bound
Standard Attn:   dram__throughput > sm__throughput  → memory-bound
```

> 💡 FlashAttention 把瓶颈从 memory-bound（Standard）转变为 compute-bound，这正是 IO-aware 优化的目标——让计算单元成为瓶颈而非内存带宽。
