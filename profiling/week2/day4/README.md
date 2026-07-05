# Week 2 Day 4 — Nsight Compute Profiling

> 对应 [Week 2 Day 4 Coding 任务 1-5 + 扩展实验](../../week2/day4/README.md)

Day 4 的核心是学习 ncu 命令行用法、关键指标解读、瓶颈判定和 profile-optimize-verify 闭环。本目录包含两个 profiling 对象：Register Blocking GEMM（任务 1-4）和 Softmax（任务 5）。

## 1. Register Blocking GEMM Profiling（任务 1-4）

### 编译与运行

```bash
make register_gemm
./register_gemm
```

### ncu 分析

```bash
make profile-gemm          # 核心指标（任务 1）→ 输出到 ncu_output.txt
make profile-gemm-csv      # 导出 CSV（便于命令行查看）
make profile-gemm-full     # ncu --set full 完整报告
make nsys-gemm             # nsys 时间线（实验 1）
```

等价于手动运行：

```bash
ncu \
  --kernel-name regex:gemmRegisterBlocking \
  -o gemm_profile_report \
  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__warps_eligible.sum.per_cycle,\
smsp__average_warps_issue_stalled_long_scoreboard.pct \
  ./register_gemm 2>&1 | tee ncu_output.txt
```

### 分析任务清单（任务 2）

拿到 ncu 报告后，按以下清单分析：

1. **读取 SM Throughput 和 Memory Throughput**，判断是 compute-bound 还是 memory-bound
   - Memory Throughput >> SM Throughput → memory-bound
   - SM Throughput >> Memory Throughput → compute-bound
2. **读取 Achieved Occupancy**，判断 SM 利用率是否充分（目标 > 70%）
3. **查看 Warp Stall Reasons**，找出主要 stall 原因
4. **对比 L1/TEX Hit Rate**，判断缓存效率
5. **打开 ncu-ui → Source 视图**，定位最耗时的代码行

### 案例分析参考（任务 3）

| 指标 | 典型值 | 解读 |
|------|--------|------|
| SM Throughput | ~45% | 偏低 |
| Memory Throughput | ~78% | 高 → memory-bound |
| Achieved Occupancy | ~56% | 偏低，register 太多 |
| Long Scoreboard Stall | ~35% | 高！全局内存加载延迟是主要瓶颈 |
| Math Pipe Throttle | ~12% | FMA 依赖链 |

### 优化方向（任务 4）

| Profile 发现 | 尝试优化 | 预期效果 |
|------------|---------|---------|
| Memory Throughput 高，SM Throughput 低 | 增大 TM×TN（如 8×8→16×8） | 提升计算强度 |
| Long Scoreboard Stall 高 | 引入 Double Buffering | 掩盖内存延迟 |
| Achieved Occupancy 低 | 减少 register 使用（减小 TM 或 TN） | 提升 warp occupancy |
| L1 Hit Rate 低 | 检查 coalesced access 模式 | 提升缓存效率 |

## 2. Softmax Profiling（任务 5）

### 编译与运行

```bash
make softmax_profile
./softmax_profile
```

### ncu 分析

```bash
make profile-softmax       # 核心指标
make profile-softmax-full  # ncu --set full 完整报告
make nsys-softmax          # nsys 时间线
```

### 观察要点

| 指标 | 预期值 | 解读 |
|------|--------|------|
| `dram__throughput` | **高**（>60%） | memory-bound 的标志 |
| `sm__throughput` | **低**（<20%） | 计算量极小 |
| `sm__occupancy` | 中高（>50%） | 寄存器/smem 用量少 |
| `smsp__...stalled_long_scoreboard.pct` | **高**（>30%） | 三遍扫描每次从 HBM 读，等内存 |

**关键洞察**：Softmax 是纯 memory-bound（AI ≈ 0.375 FLOP/Byte），ncu 应显示 DRAM Throughput >> SM Throughput，Long Scoreboard Stall 高。优化方向：fusion（减少 HBM 读写次数）、online softmax（三遍→两遍）。

## 3. nsys Timeline 分析（实验 1）

```bash
make nsys-gemm        # GEMM 时间线
make nsys-softmax     # Softmax 时间线
```

用 Nsight Systems GUI 打开 `.nsys-rep`，观察：
- H2D memcpy、Kernel、D2H memcpy 的时间占比
- 多个 kernel 的执行顺序
- CPU 和 GPU 的并行情况

## Profile → 优化 → 验证 完整闭环

```
1. Profile（baseline）→ make profile-gemm
2. 识别瓶颈 → SM/DRAM throughput + stall reasons
3. 针对性优化 → 修改 kernel 代码
4. 重新 Profile → make profile-gemm（对比指标变化）
5. 确认优化有效 → SM throughput 提升 / Long Scoreboard 下降
```
