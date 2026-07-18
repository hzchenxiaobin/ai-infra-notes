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

> 报告对象：`register_gemm_full.ncu-rep` 中的 `gemmRegisterBlocking` kernel，矩阵规模 M=N=K=1024，grid=(8,8,1)=64 blocks，block=(256,1,1)，CC 12.0（Blackwell，170 SMs）。
> 复现：`make profile-gemm-full` 后用 `ncu --import register_gemm_full.ncu-rep --page details` 查看。

#### Step 1：先看 Speed Of Light Throughput 一张表

| 指标 | 实测值 | 含义 |
|------|--------|------|
| **Compute (SM) Throughput** | **5.25%** | SM 流水线占峰值百分比 |
| **Memory Throughput** | **5.98%** | 整个内存子系统（L1+L2+DRAM）综合占用 |
| DRAM Throughput | 0.96% | 仅 HBM 带宽占用（几乎没动） |
| L1/TEX Cache Throughput | 15.88% | L1/TEX 缓存带宽占用 |
| L2 Cache Throughput | 4.22% | L2 带宽占用 |
| Duration | 497.95 us | kernel 耗时 |

#### Step 2：用「SM Thp × Memory Thp」决策表判断瓶颈类型

只看这两个数容易误判。需按下面的决策树读：

| 模式 | SM Thp | Mem Thp | 瓶颈类型 | 典型对策 |
|------|--------|---------|---------|---------|
| A 计算瓶颈 | **高（>60%）** | 中/低 | compute-bound | 简化算式、上 Tensor Core |
| B 带宽瓶颈 | 中/低 | **高（>60%）** | memory-bandwidth-bound | 增大计算强度、fusion |
| C 双高 | 高 | 高 | 已接近峰值，看下一层细节 | 微调 tile / 流水线 |
| **D 双低** | **低（<15%）** | **低（<15%）** | **latency-bound / under-utilized** | 提 occupancy、增大 grid、掩盖延迟 |

**本例 SM=5.25%、Mem=5.98% → 属于模式 D：latency-bound / under-utilized**。SM 和内存都没在"忙"，硬件大量空转——瓶颈既不是计算也不是带宽，而是「warp 不够多 / 等不到」。

> ⚠️ 注意：不要因为 `Memory(5.98%) > SM(5.25%)` 就套用「Memory >> SM ⇒ memory-bound」。两个数都极低时，差距毫无意义，瓶颈在 latency/occupancy 而非带宽。判读必须配合 stall reasons 和 occupancy 一起看。

#### Step 3：在模式 D 下，定位「warp 在等什么」——Warp State Statistics

每个 issued instruction 平均间隔 14.97 个 cycle，分布如下：

| Stall 原因 | cycles/inst | 占比 | 含义 |
|------------|-------------|------|------|
| **long_scoreboard** | **10.375** | **69.3%** | 等 L1TEX（global/shared）load 完成依赖 |
| short_scoreboard | 0.705 | 4.7% | 等 MIO 短延迟 |
| dispatch_stall | 0.666 | 4.5% | 调度器内部资源 |
| not_selected | 0.588 | 3.9% | 本 warp 就绪但本轮没被选中 |
| barrier | 0.491 | 3.3% | `__syncthreads` |
| wait | 0.490 | 3.3% | 显式 wait |
| lg_throttle | 0.313 | 2.1% | load/store 单元满载 |
| mio_throttle | 0.139 | 0.9% | MIO pipe 拥塞 |
| **math_pipe_throttle** | **0.002** | **0.01%** | FMA pipe 满 ← 几乎为 0 |

结论：**真正卡住 warp 的是** `long_scoreboard`**（等内存依赖）**，`math_pipe_throttle` 几乎为 0 → 计算绝对不是瓶颈。
精确表述：**memory-latency-bound（被内存延迟卡住），且因驻留 warp 太少无法掩盖延迟，整体表现为 under-utilization**。

#### Step 4：为什么 warp 不够多？查 Launch Statistics 与 Occupancy

| 指标 | 实测值 | 说明 |
|------|--------|------|
| Grid Size | 64 blocks | 1024/128=8 → 8×8=64 |
| # SMs | 170 | Blackwell |
| **Waves Per SM** | **0.19** | 64/170 → **连一轮都没填满** |
| Registers Per Thread | 128 | `acc[8][8]` float + `r_A/r_B` 占满 |
| Block Limit Registers | 2 | 128 reg/thread → 每 SM 只能驻留 2 个 block |
| Theoretical Occupancy | 33.33% | 被 register 数量限制 |
| **Achieved Occupancy** | **16.62%** | 实际只有理论值的一半 |
| Active Warps / Scheduler | 2.0 / 12 | 硬件上限的 1/6 |
| Eligible Warps / Scheduler | 0.21 | 平均每周期几乎没有就绪 warp |
| No Eligible | 86.66% | 86% 的周期调度器空转 |

根因链：
1. `acc[8][8]` 让每线程用 128 reg → 每 SM 只能驻留 2 个 block → 理论 occupancy 仅 33%。
2. 矩阵只有 1024，grid 才 64 blocks，连 170 个 SM 都填不满（0.19 wave）。
3. 驻留 warp 太少（每 scheduler 2 个）→ 无法用其他 warp 掩盖 `long_scoreboard` 几百周期的延迟 → 86% 周期空转。

#### Step 5：用 ncu 的 OPT 规则交叉验证

| 规则 | 预估加速 | 对应结论 |
|------|---------|---------|
| `HighPipeUtilization` | 局部 94.75% | 所有计算 pipe 严重未用 → 同 Step 2 |
| `IssueSlotUtilization` | 局部 86.66% | 调度器 86% 时间空转 → 同 Step 4 |
| `CPIStall` | 全局 69.29% | `long_scoreboard` 占 69% → 同 Step 3 |
| `LaunchConfiguration` | 全局 62.35% | grid 64 < 170 SMs → 同 Step 4 |
| `TheoreticalOccupancy` | 全局 66.67% | register 限制 → 同 Step 4 |
| `UncoalescedGlobalAccess` | 10.32% | C 写回 4/32 byte/sector → 放大 long_scoreboard |
| `SharedMemoryConflicts` | 14.35% | smem 5-way bank conflict（40% wavefront）→ 放大 long_scoreboard |

四个独立角度（throughput / stall / launch / occupancy）指向同一根因，判读自洽。

#### Step 6：决策表——本例应该怎么优化

| 优先级 | 动作 | 预期受益指标 |
|--------|------|-------------|
| P0 | 跑更大的 M=N=K（如 4096 → grid=1024 blocks，6 waves） | Waves/SM、Achieved Occupancy ↑，整体 Thp ↑ |
| P0 | 减小 register（如 TM×TN 8×8 → 8×4，或拆 K 维 / split-K） | Theoretical Occupancy 33% → 50%+ |
| P1 | 修 C 写回的 coalescing（ncu 报 4/32 byte/sector） | L1/TEX Hit Rate 29% ↑，long_scoreboard ↓ |
| P1 | 修 smem bank conflict（5-way，40% wavefront） | long_scoreboard ↓ |
| P2 | 引入 double buffering 掩盖 load 延迟 | long_scoreboard ↓，eligible warps ↑ |

#### 一句话总结

> SM Thp 5.25% ≈ Mem Thp 5.98% 都极低 ⇒ **不是带宽/计算瓶颈，是 latency/occupancy 瓶颈**；再看 stall，`long_scoreboard` 占 69% ⇒ **被内存延迟卡住**；再看 occupancy，128 reg/thread 把理论 occupancy 压到 33%、实测 16.6%，加上 grid 没填满 SM ⇒ **没有足够 warp 掩盖内存延迟**。优化优先级：先增大问题规模/降寄存器提升 occupancy，再修内存访问模式降低延迟。

### Warp Stall Reasons 深度分析（任务 2.3 扩展）

> 仍以 `register_gemm_full.ncu-rep` 为例。通过 `ncu --import register_gemm_full.ncu-rep --page source --print-source sass --metrics smsp__pcsamp_warps_issue_stalled_<reason>` 可将 stall 采样数定位到 SASS 指令级，从而精确找到「warp 卡在哪一行代码」。

#### 什么是 Warp Stall？

GPU 每个 SMSP（SM 子分区）的 warp 调度器每周期尝试发射一条指令。若当前 warp 的下一条指令因为依赖未就绪而无法发射，就叫一次 **stall**。ncu 用 **PC 采样**统计：每隔固定周期采样一次各 warp slot 当前停在哪条指令、属于哪种 stall 原因。

核心指标 `Warp Cycles Per Issued Instruction = 14.97 cycle` 表示**平均每发射 1 条指令要等 15 个周期**，其中绝大部分时间在 stall。把这个 15 cycle 拆开看各 stall 原因的占比，就能定位瓶颈。

#### stall 原因总览（本例实测）

| Stall 原因 | cycles/inst | 占比 | 采样数 | 严重度 |
|------------|-------------|------|--------|--------|
| **long_scoreboard** | **10.375** | **69.3%** | **13,624** | 🔴 主瓶颈 |
| short_scoreboard | 0.705 | 4.7% | 929 | 🟡 次要 |
| not_selected | 0.588 | 3.9% | 779 | 🟡 |
| barrier | 0.491 | 3.3% | 662 | 🟡 |
| wait | 0.490 | 3.3% | 613 | 🟡 |
| lg_throttle | 0.313 | 2.1% | 420 | 🟢 |
| mio_throttle | 0.139 | 0.9% | 189 | 🟢 |
| branch_resolving | 0.112 | 0.7% | 149 | 🟢 |
| no_instruction | 0.064 | 0.4% | 77 | 🟢 |
| drain | 0.026 | 0.2% | 37 | ⚪ |
| **math_pipe_throttle** | **0.002** | **0.01%** | **2** | ⚪ 几乎为零 |
| membar/misc/sleeping/tex_throttle | 0 | 0% | 0 | ⚪ |

> 读法：cycles/inst 是「平均每条指令因该原因停顿的周期数」，加起来约等于总 CPI（14.97）。占比 = cycles/inst ÷ 14.97。

#### 逐项分析 + 源码定位 + 优化方法

##### 1. long_scoreboard（69.3%）—— 主瓶颈

**含义**：warp 等待一个 **L1TEX 长延迟操作**（global load / shared load / local load / texture）的 scoreboard 依赖解除。典型场景：一条 `LDG`（全局内存加载）发出后，后续需要用到该数据的指令要等数据从 HBM/L2 返回（数百 cycle）。

**本例源码定位**（SASS 级 stall 采样）：

| SASS 地址 | 指令 | stall 采样数 | 对应 CUDA 代码 |
|-----------|------|-------------|---------------|
| 0x110131c900 | `STS [R19+0x400], R16` | 1874 | `s_B[...][...] = B[...]`（B tile 加载） |
| 0x110131c480 | `STS [R3], R16` | 1750 | `s_A[...][...] = A[...]`（A tile 加载） |
| 0x110131c610 | `STS [R3+0x800], R16` | 1747 | `s_A[...][...] = A[...]`（A tile 加载） |
| 0x110131c540 | `STS [R3+0x400], R16` | 1732 | `s_A[...][...] = A[...]`（A tile 加载） |
| 0x110131c6e0 | `STS [R3+0xc00], R16` | 1724 | `s_A[...][...] = A[...]`（A tile 加载） |
| 0x110131c9f0 | `STS [R19+0x800], R16` | 1688 | `s_B[...][...] = B[...]`（B tile 加载） |
| 0x110131cae0 | `STS [R19+0xc00], R16` | 1617 | `s_B[...][...] = B[...]`（B tile 加载） |
| 0x110131c840 | `STS [R19], R16` | 1453 | `s_B[...][...] = B[...]`（B tile 加载） |
| **合计** | | **12,885 / 13,624** | **94.6% 集中在 tile 加载的 STS** |

**关键发现**：所有 long_scoreboard stall 都不在计算循环（FFMA 指令 stall = 0），而在 **tile 加载阶段的 STS（shared memory store）指令**上。SASS 模式为：

```asm
LDG.E.CONSTANT R16, desc[UR10][R16.64]   ; 从 global memory 加载到 R16（发出后不阻塞）
...                                       ; 编译器插入几条无关指令试图掩盖
STS [R3], R16                             ; 把 R16 存入 shared memory ← 卡在这里！
                                          ;   R16 还没从 HBM 返回，scoreboard 未解除
```

即 `s_A[loadRow][aCol] = A[(cRow + loadRow) * K + (bk + aCol)]` 这行 CUDA 代码：先 `LDG` 从 HBM 读数据到寄存器 R16，再 `STS` 写入 shared memory。`STS` 依赖 `LDG` 的结果，而 `LDG` 要等数百周期，编译器插入的无关指令不足以填满这个窗口 → `STS` 上报 long_scoreboard。

**优化方法**：

| 方法 | 原理 | 本例适用性 |
|------|------|-----------|
| **Double Buffering** | 用两块 shared memory buffer 交替，计算当前 tile 时预取下一个 tile，用 FFMA 计算掩盖 LDG 延迟 | ✅ 最直接，可将 long_scoreboard 大幅降低 |
| **增大 BK**（如 8→16/32） | 减少 K 维迭代次数 → 减少 LDG 次数；每次加载更多数据分摊延迟 | ✅ 但会增加 smem 用量，需平衡 occupancy |
| **Software Prefetch** | 手动提前发出 LDG，插入更多无关指令 | ⚠️ 编译器已做一定调度，手动效果有限 |
| **提升 Occupancy** | 更多驻留 warp → 调度器在当前 warp stall 时切换到其他 warp | ✅ 本例 occupancy 仅 16.6%，提升空间大 |
| **Vectorized Load**（LDG.128） | 用 128-bit 向量加载减少指令数 | ✅ 当前是标量 LDG.E.32，可优化 |

##### 2. short_scoreboard（4.7%）—— 次要

**含义**：等待 **L1TEX 短延迟操作**（如 shared memory load、constant load）的 scoreboard 依赖。比 long_scoreboard 延迟短（~20-30 cycle），通常是 shared memory 读后立即用。

**本例源码定位**：

| SASS 地址 | 指令 | 采样数 | 说明 |
|-----------|------|--------|------|
| 0x110131cc70 | `FFMA R114, R52, R20, R114` | 238 | 计算循环第一条 FFMA，等 `LDS.128 R52` |
| 0x110131cac0 | `IMAD.WIDE R16, R8, 0x4, R16` | 44 | 地址计算 |
| 0x110131c9d0 | `IMAD.WIDE R16, R7, 0x4, R16` | 61 | 地址计算 |

stall 集中在计算循环开头：`LDS.128` 从 shared memory 读出 `r_A`/`r_B` 后，紧跟的 `FFMA` 要等数据到达。编译器已尽量插入了其他 FFMA 来掩盖，但第一条 FFMA 无法被掩盖。

**优化方法**：编译器调度已较好，手动优化空间不大；增大 BK 让 LDS 数据被更多 FFMA 复用（摊薄 short_scoreboard 占比）。

##### 3. not_selected（3.9%）

**含义**：warp 本身 **已就绪（eligible）**，但调度器本轮选中了其他 warp 发射。这**不是真正的 stall**，而是正常的多 warp 轮转。只有当 eligible warp 很少时才值得关注。

**本例关联**：本例 `Eligible Warps Per Scheduler = 0.21`（极低），`One or More Eligible = 13.34%`。not_selected 只有 3.9% 恰好说明 eligible warp 太少——大部分周期连一个就绪 warp 都没有（86.66% No Eligible），谈不上「选了别的 warp」。

**优化方法**：不需要直接优化 not_selected；提升 occupancy 和减少 long_scoreboard 后，not_selected 会自然上升（好事，说明有更多 warp 可调度）。

##### 4. barrier（3.3%）

**含义**：warp 停在 `__syncthreads()` 等待其他 warp 到达同步点。

**本例源码定位**：

| SASS 地址 | 指令 | 采样数 | 说明 |
|-----------|------|--------|------|
| 0x110131cbf0 | `LDS.128 R52, [R10+UR5]` | 329 | `__syncthreads()` 后第一条 LDS |
| 0x110131ee00 | `BRA.U !UP0, ...` | 333 | 循环回跳处的分支 |

stall 采样落在 `BAR.SYNC` 之后的指令上，说明 warp 到达 `__syncthreads()` 后要等其他 warp 完成 tile 加载。根因仍是 long_scoreboard：部分 warp 的 LDG 慢，拖慢了整个 block 的同步。

**优化方法**：减少 barrier 前的 long_scoreboard（同上）；减少 `__syncthreads()` 次数（如增大 BK 减少 K 维迭代轮数）；考虑 `__syncwarp()` 替代部分场景。

##### 5. wait（3.3%）

**含义**：显式等待某些资源（如 `arrive/wait` barrier、async copy 完成等）。

**本例源码定位**：集中在 tile 加载阶段的地址计算和分支指令（`ISETP`、`BRA`），属于控制流等待。影响较小。

##### 6. lg_throttle（2.1%）

**含义**：Load/Global store unit 满载，新的 load/store 指令无法发射到 MIO pipe。

**本例源码定位**：

| SASS 地址 | 指令 | 采样数 | 说明 |
|-----------|------|--------|------|
| 0x110131f3a0 | `STG.E desc[UR10][R16.64+0x4], R72` | 19 | C 写回 global memory |
| 0x110131f220 | `STG.E desc[UR10][R12.64+0x4], R63` | 14 | C 写回 |
| ... | （共 ~20 条 STG.E） | 各 2-19 | C 矩阵写回阶段 |

stall 集中在 **C 矩阵写回阶段**的 `STG.E`（global store）指令。这正是 ncu 报告的 uncoalesced global store（4/32 byte/sector）所在位置——每个 thread 写 8×8=64 个 float，但 stride 导致 store 效率极低，MIO pipe 被大量 sector 请求填满。

**优化方法**：修复 C 写回的 coalescing（让相邻 thread 写相邻地址），或用 shared memory 暂存后批量 coalesced 写回。

##### 7. mio_throttle（0.9%）

**含义**：Memory I/O pipe 拥塞（L1TEX 的 LSU 管线满），load/store 指令无法进入 pipe。

**本例源码定位**：集中在计算循环的 `LDS.128`（shared memory load）指令上——8 条 `LDS.128` 连续发射后 MIO pipe 被填满。

**优化方法**：影响很小，不需要专门优化。

##### 8. math_pipe_throttle（0.01%）—— 反证计算不是瓶颈

**含义**：FMA/Math pipe 满载，新的 FFMA 无法发射。**这个值高才说明是 compute-bound**。

**本例**：仅 2 个采样、0.002 cycle/inst。计算循环中有数百条 `FFMA` 指令（`acc[8][8]` 的 64 个累加），但 math_pipe_throttle 几乎为零 → **FMA pipe 从未满载，计算绝对不是瓶颈**。这从另一个角度验证了 Step 2 的结论。

#### stall 原因 → 瓶颈类型 → 优化优先级 决策表

| 主导 stall 原因 | 瓶颈类型 | 典型场景 | 首选优化 |
|---------------|---------|---------|---------|
| **long_scoreboard** 高 | memory-latency-bound | global load 延迟未掩盖 | Double Buffer / 提 occupancy / vectorized load |
| **math_pipe_throttle** 高 | compute-bound | FMA/INT pipe 满载 | 简化算式 / 上 Tensor Core / 增大计算强度 |
| **lg_throttle / mio_throttle** 高 | memory-throughput-bound | LSU/MIO pipe 带宽满 | coalescing / 减少 memory 指令数 |
| **barrier** 高 | sync-bound | `__syncthreads` 等待 | 减少 barrier 次数 / 平衡 workload |
| **short_scoreboard** 高 | smem-latency-bound | shared load 立即用 | 编译器调度 / 增大 tile 复用 |
| **no_instruction** 高 | instruction-starvation | 取指瓶颈 / 代码膨胀 | 减小指令量 / I-cache 优化 |
| 各项都低但 CPI 高 | latency-bound（stall 分散） | occupancy 太低 | 提 occupancy / 增大 grid |

**本例诊断**：long_scoreboard 占 69.3% 且集中在 LDG→STS 的 tile 加载阶段，math_pipe_throttle ≈ 0 → **memory-latency-bound，需通过 double buffering + 提升 occupancy 来掩盖 global load 延迟**。

#### 如何复现本节的 stall 源码定位

```bash
# 查看总 stall 汇总表
ncu --import register_gemm_full.ncu-rep --page details --print-details all --section WarpStateStats

# 定位 long_scoreboard 集中在哪些 SASS 指令
ncu --import register_gemm_full.ncu-rep \
    --page source --print-source sass \
    --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard

# 同时查看多种 stall 原因的源码分布
ncu --import register_gemm_full.ncu-rep \
    --page source --print-source sass \
    --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard,\
smsp__pcsamp_warps_issue_stalled_short_scoreboard,\
smsp__pcsamp_warps_issue_stalled_lg_throttle,\
smsp__pcsamp_warps_issue_stalled_barrier
```

> 读 SASS stall 表的方法：每行是一条 SASS 指令，最右列是该指令上采样到的 stall 次数。**数字最大的几行就是瓶颈代码所在**。配合 SASS 指令类型（`LDG`=global load, `STS`=shared store, `LDS`=shared load, `FFMA`=float FMA, `STG`=global store, `BAR.SYNC`=__syncthreads）即可判断 stall 根因。

### 优化方向（任务 4）

| Profile 发现 | 尝试优化 | 预期效果 |
|------------|---------|---------|
| Memory Throughput 高，SM Throughput 低 | 增大 TM×TN（如 8×8→16×8） | 提升计算强度 |
| Long Scoreboard Stall 高 | 引入 Double Buffering | 掩盖内存延迟 |
| Achieved Occupancy 低 | 减少 register 使用（减小 TM 或 TN） | 提升 warp occupancy |
| L1 Hit Rate 低 | 检查 coalesced access 模式 | 提升缓存效率 |
| long_scoreboard 集中在 LDG→STS | Double Buffering + 增大 BK | 掩盖 global load 延迟 |
| lg_throttle 集中在 STG（C 写回） | 修复 coalesced store | 减少 sector 请求 |
| math_pipe_throttle ≈ 0 | 无需优化计算 | 验证非 compute-bound |

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
