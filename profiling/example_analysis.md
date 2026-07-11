# ncu 结果分析实例：以 `bank_conflict` 为例

下面以 `ncu/week1/day5/bank_conflict.cu` 为例，演示如何运行 `ncu`、拿到指标、并逐步分析出性能瓶颈。

---

## 1. 编译并运行

```bash
cd ncu/week1/day5
make
./bank_conflict
```

输出示例：

```text
Bank conflict kernels finished. Use ncu to compare metrics.
```

---

## 2. 采集指标

```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
  sm__cycles_elapsed.avg,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__occupancy.avg.pct_of_peak_sustained_elapsed \
  ./bank_conflict
```

> 如果提示没有 `ncu` 权限，可尝试加 `sudo`；或者把结果输出到文件：`ncu --metrics ... ./bank_conflict 2>&1 | tee ncu_log.txt`

---

## 3. 假设得到的 ncu 输出

由于实际数值依赖 GPU 型号和 CUDA 版本，下面给出一组**典型趋势**（基于 RTX 5090）：

| Kernel | `sm__cycles_elapsed.avg` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | `sm__throughput` | `dram__throughput` | `sm__occupancy` |
|--------|--------------------------|-----------------------------------------------------------|------------------|--------------------|-----------------|
| `conflict_read` | 12,800 | 1,048,576 | 18% | 45% | 75% |
| `no_conflict_read` | 3,200 | 0 | 55% | 48% | 75% |

---

## 4. 逐步分析

### 步骤 1：先看 cycle 数

`conflict_read` 跑了 **12,800 cycles**，而 `no_conflict_read` 只跑了 **3,200 cycles**。

> **4 倍慢**。这说明存在严重的性能问题。

### 步骤 2：看 occupancy

两者都是 **75%**。occupancy 不是瓶颈。

> 如果 occupancy 很低（比如 < 50%），才需要怀疑寄存器、shared memory 或 block size 配置。

### 步骤 3：看 throughput

- `conflict_read` 的 `sm__throughput` 只有 **18%**
- `no_conflict_read` 的 `sm__throughput` 达到 **55%**

`sm__throughput` 低说明 SM 计算单元大量空闲，没有在高效工作。

### 步骤 4：看 bank conflict 指标

- `conflict_read`：**1,048,576** 次 shared memory load bank conflict
- `no_conflict_read**：**0**

两者 `dram__throughput` 差不多（~48%），说明 global memory 访问不是主要瓶颈。真正的瓶颈在 **shared memory 的 bank conflict**。

### 步骤 5：得出结论

```text
conflict_read 慢 4 倍的原因：
→ 同一个 warp 内多个线程访问了同一个 bank 的不同地址
→ 导致 shared memory 访问被串行化
→ SM 大量时间花在等 shared memory
→ sm__throughput 大幅下降
```

---

## 5. 如何验证结论

修改源码，把 `tile[TILE_DIM][TILE_DIM]` 改成 `tile[TILE_DIM][TILE_DIM + 1]`（加 padding），重新编译并运行同样的 ncu 命令，预期看到：

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 变为 0 或接近 0
- `sm__cycles_elapsed.avg` 大幅下降
- `sm__throughput` 明显上升

如果指标确实这样变化，就说明 bank conflict 分析正确。

---

## 6. 分析流程总结

拿到 ncu 结果后，建议按这个顺序看：

1. **`sm__cycles_elapsed.avg`** —— 先确定谁慢、慢多少
2. **`sm__occupancy.avg.pct_of_peak_sustained_elapsed`** —— 排除 occupancy 问题
3. **`sm__throughput` vs `dram__throughput`** —— 判断是 compute-bound 还是 memory-bound
   - SM 高、DRAM 低 → compute-bound
   - SM 低、DRAM 高 → memory-bound
   - 两者都低 → 可能是 latency/stall/bank conflict/occupancy 问题
4. **看具体 stall / conflict 指标** —— 定位根因
   - bank conflict：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_*`
   - global memory 效率：`dram__bytes_read.sum`、`dram__bytes_write.sum`
   - warp stall：`smsp__average_warps_issue_stalled_*`
5. **修改代码 → 重新编译 → 重新 ncu → 对比指标** —— 验证优化效果

---

## 7. 另一个例子：transpose

```bash
cd ncu/week1/day4
make
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
  sm__cycles_elapsed.avg \
  ./transpose
```

**预期分析**：

| Kernel | cycles | DRAM throughput | global read bytes | global write bytes |
|--------|--------|-----------------|-------------------|--------------------|
| `transpose_naive` | 高 | 低 | 正常 | 正常 |
| `transpose_optimized` | 低 | 高 | 正常 | 正常 |

`transpose_naive` 按列写 global memory，造成 stride access，有效带宽低，因此 `dram__throughput` 低、cycle 高。`transpose_optimized` 通过 shared memory tile 把写操作转成 coalesced，带宽利用率提高，cycle 下降。

---

## 8. 常见误区

- **只盯着 cycle 数**：cycle 只是结果，要找到为什么 cycle 高。
- **看到 occupancy 低就加 block size**：occupancy 低可能是寄存器或 shared memory 限制，盲目加线程数不一定有效。
- **忽略 bank conflict**：shared memory 冲突会让 kernel 显著变慢，但初学者容易只关注 global memory。
- **不做 A/B 对比**：优化前后一定要跑同样的 ncu 命令，用数据说话。
