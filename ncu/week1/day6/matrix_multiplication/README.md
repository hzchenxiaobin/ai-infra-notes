# Matrix Multiplication — ncu 分析

本题来自 LeetGPU [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)，是 Week 1 Day 6 的在线题目任务 5。这里把 naive 与 shared-memory tiling 两个版本抽出来，方便直接用 ncu 对比分析。

---

## 1. 编译运行

```bash
cd ncu/week1/day6/matrix_multiplication
make
./matmul
```

预期输出示例：

```text
Naive:  45.123 ms (  5.9 GFLOPS)
Tiled:   8.456 ms ( 31.7 GFLOPS)
Speedup: 5.34x
Tiled (no bank conflict): 7.234 ms ( 37.1 GFLOPS)
Speedup vs Tiled: 1.17x
```

---

## 2. ncu 采集关键指标

```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  sm__cycles_elapsed.avg \
  ./matmul
```

也可以生成完整报告：

```bash
ncu --set full -o matmul_report ./matmul
ncu-ui matmul_report.ncu-rep
```

---

## 3. 预期 ncu 结果趋势

| Kernel | cycles | SM throughput | DRAM throughput | occupancy | bank conflicts |
|--------|--------|---------------|-----------------|-----------|----------------|
| `matmul_naive` | 高 | 低（可能 < 30%） | 低（可能 < 40%） | 正常 | 0 |
| `matmul_tiled` | 低 | 较高（可能 40-60%） | 较高（可能 50-70%） | 正常 | 可能有 2-way |
| `matmul_tiled_nobc` | 更低 | 较高（可能 50-70%） | 较高 | 正常 | 接近 0 |

---

## 4. 如何分析

### 4.1 先看 cycle 和 GFLOPS

`matmul_tiled` 通常比 `matmul_naive` 快 3~8 倍。cycle 数会直观反映这个差距。

### 4.2 看 throughput 判断瓶颈大类

- `matmul_naive`：
  - `sm__throughput` 低 + `dram__throughput` 低 → 不是 compute 也不是 bandwidth 打满
  - 原因：大量重复 global memory 访问，每个线程从 global memory 读 A 的一整行和 B 的一整列，内存访问模式差，有效带宽利用率低

- `matmul_tiled`：
  - `sm__throughput` 和 `dram__throughput` 都明显提升
  - 原因：Shared Memory Tiling 把 A/B 的子矩阵缓存到 Shared Memory，实现 K 维度数据复用，减少 global memory 访问

### 4.3 看 bank conflict

`tiled` 版本在读取 `s_A[threadIdx.y][k]` 时容易出现 bank conflict。原因：`TILE_SIZE = 16` 时，相邻两行在 Shared Memory 中相距 `16 × 4 = 64` 字节，恰好是 32-bank 共享内存（每 bank 4 字节）的整数倍。因此同一列的偶数行落在 bank `k`，奇数行落在 bank `k + 16`，形成 **2-way bank conflict**。

如果 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 数值较大，可以改为 padding 版本：

```cuda
__shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
__shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];
```

行 stride 变为 `17 × 4 = 68` 字节，`68 / 4 = 17` 与 32 互质，相邻行会落到不同 bank，conflict 基本消失。

也可以直接对比三个 kernel 的 bank conflict 数量：

```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
  ./matmul
```

### 4.4 Roofline 定位

算术强度：

```text
AI = 2 * M * N * K / (M*K + K*N + M*N) / sizeof(float)
```

对于 M=N=K=512：

```text
AI ≈ 2 * 512³ / (3 * 512² * 4) ≈ 85 FLOP/Byte
```

A100 的 Ridge Point ≈ 12.6 FLOP/Byte，因此 `matmul_tiled` 更接近 **compute-bound**。但受限于简单实现（没有 register blocking、warp shuffle、float4 等），通常还远没打到峰值算力。

---

## 5. 进阶优化方向

1. **增大 TILE_SIZE**：尝试 32×32，观察 occupancy 和 bank conflict 变化
2. **Register Blocking**：每个线程计算多个输出元素，提高数据复用
3. **Thread Block 形状**：比如 16×8 而非 16×16，平衡 occupancy 和 register 压力
4. **float4 向量化加载**：一次加载 4 个 float，减少 load 指令数
5. **Warp Shuffle**：在 warp 内做局部归约/数据交换，减少对 Shared Memory 的依赖

每次修改后都用同样的 ncu 命令对比：

```bash
make clean && make
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__occupancy.avg.pct_of_peak_sustained_elapsed ./matmul
```
