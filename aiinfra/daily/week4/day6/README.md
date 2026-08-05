## Day 6：Profiling —— Triton vs CUDA vs PyTorch 性能对比

### 🎯 目标

通过今天的学习，你将：

1. 能用 ncu 和 `torch.profiler` 分析 Triton / CUDA / PyTorch 三方 kernel 的内部指标差异<br>
2. 理解 Triton 自动生成的 **shared memory tiling / Tensor Core / 向量化** 与手写 CUDA 的差异<br>
3. 能用 ncu 的关键指标解释"为什么 Triton GEMM 达 70% 而 CUDA 手写只有 30%"<br>
4. 掌握 `torch.profiler` 的算子时间线分析，能看到单次 forward 的算子级耗时分布<br>
5. 能根据 profiling 数据选择最优实现——"这个算子是 memory-bound，用 Triton 自动 tiling 够了"<br>

> 💡 **为什么重要**：Day 5 的 benchmark 给了"谁快谁慢"，今天用 ncu 打开看"为什么快为什么慢"。面试中被问"Triton 为什么比你的手写 CUDA 快 2x"，不能回答"可能是因为..."，必须用 ncu 数据说话。

---

### 学前导读：从计时到指标

Day 5 的 benchmark 只看了 wall time（ms）。今天用 ncu 看**内部指标**：

| 指标 | 回答的问题 |
|------|----------|
| `sm__pipe_tensor_op_hmma` | Tensor Core 是否被使用？利用率多少？ |
| `dram__throughput` | HBM 带宽是否瓶颈？ |
| `l1tex__throughput` | Shared memory 是否瓶颈？ |
| `sm__occupancy` | Occupancy 是否足够？ |
| `launch__registers_per_thread` | 寄存器用量是否限制 occupancy？ |

> 💡 **一句话总结**：Day 5 实测 Triton GEMM 大矩阵达 cuBLAS 97.5%，今天用 ncu 解释"Triton 用了 Tensor Core + autotune 选最优 tiling，而手写 CUDA 没用 Tensor Core + 固定 tiling"。

---

### 理论学习

#### 6.1 三方 Profiling 对比

##### GEMM 三方实测对比（RTX 5090, 4096×4096, FP16）

| Metric | Triton GEMM（实测 97.5% cuBLAS） | CUDA 手写 (Day 1, ~30% cuBLAS) | cuBLAS |
|--------|------------------------------|----------------|--------|
| cuBLAS 占比 | **97.5%**（实测） | **~30%**（实测） | 100% |
| `sm__pipe_tensor_op_hmma` | 60-75%（推理值） | 0% | 85-95% |
| `sm__throughput` | 65-80% | 45% | 85% |
| `dram__throughput` | 35-45% | 70% | 28% |
| `sm__occupancy` | 50-65% | 45% | 60% |
| `registers_per_thread` | ~64-80 | ~96-128 | ~80 |
| cuBLAS% | 70-80% | 30% | 100% |

##### 关键洞察

1. **Triton 用了 Tensor Core（60-75%）**：`tl.dot` 自动调 WMMA/mma.sync，而 Day 1 手写 CUDA 用 FMA（0%）
2. **Triton 的 occupancy 更高**：自动选 `num_warps=4/8`，寄存器用量更少（~64-80 vs ~96-128）
3. **Triton 的 dram 利用率更低**：自动 smem tiling + double buffer 减少了 HBM 访问
4. **cuBLAS 仍领先**：Tensor Core 85-95%（vs Triton 60-75%），因 swizzle / K 分割 / epilogue fusion

#### 6.2 torch.profiler 算子时间线

##### 使用方法

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # 运行一次 forward
    output = model(input)

# 打印算子时间线
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome trace（可用 chrome://tracing 可视化）
prof.export_chrome_trace("trace.json")
```

##### 预期时间线（Transformer 单层）

```text
Operator                           CUDA Time (μs)  占比
gemm (QKV projection)              120             45%
layernorm                          25              9%
gemm (QK^T)                        40              15%
softmax                            15              6%
gemm (PV)                          35              13%
layernorm                          25              9%
gelu                               8               3%
Total                              268             100%
```

##### 算子分类（Day 1 的回顾）

| 算子类型 | AI (FLOP/Byte) | Bound 类型 | 优化方向 |
|---------|---------------|-----------|---------|
| GEMM (QKV, FFN) | ~1365 | Compute | Tensor Core + tiling |
| Softmax | ~1 | Memory | Fusion + online softmax |
| LayerNorm | ~0.6 | Memory | Welford + fusion |
| GELU | ~2 | Memory | Fusion |

#### 6.3 ncu 分析 Triton 生成的 kernel

##### Triton kernel 的命名

Triton 编译后的 kernel 名为 `xxx_kernel_<hash>`，用 `--kernel-name regex:` 匹配：

```bash
ncu --kernel-name regex:gemm_kernel \
    --metrics sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed,... \
    python3 kernels/benchmark_triton.py
```

##### Triton 的自动优化验证

用 ncu 验证 Triton autotune 选的 config 是否真的最优：

```bash
# 手动指定 BLOCK_M=64 (非最优)
TRITON_KERNEL_CONFIG='BLOCK_M:64,BLOCK_N:64,BLOCK_K:32' python3 ...

# 对比 autotune 选的 config（如 BLOCK_M=128, BLOCK_N=256）
ncu --metrics sm__pipe_tensor_op_hmma,sm__occupancy ...
```

---

### Coding 任务

#### 任务 1：ncu 分析 Triton GEMM

```bash
ncu --set full --kernel-name regex:gemm_kernel \
    --metrics sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread \
    python3 kernels/benchmark_triton.py
```

预期输出（RTX 5090, 4096×4096）：

```text
gemm_kernel_<hash>, 4096 x 4096
  sm__pipe_tensor_op_hmma    68.5%    ← Triton 用了 Tensor Core!
  sm__throughput             72.0%
  dram__throughput           38.2%
  sm__occupancy              58.0%
  launch__registers_per_thread 72
```

##### 对比 Day 1 手写 CUDA

```bash
ncu --set full --kernel-name regex:wmma_naive \
    --metrics sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed,... \
    ./wmma_naive
```

```text
wmma_naive_kernel, 4096 x 4096
  sm__pipe_tensor_op_hmma    25.0%    ← 手写 CUDA Tensor Core 利用率低
  sm__throughput             45.0%
  dram__throughput           70.0%    ← 带宽 bound
  sm__occupancy              25.0%    ← 1 warp/block, occupancy 极低
  launch__registers_per_thread 96
```

##### 分析

| 指标 | Triton | 手写 CUDA | 差距原因 |
|------|--------|----------|---------|
| Tensor Core 利用率 | 68.5% | 25.0% | Triton 自动 smem tiling + double buffer |
| HBM 带宽 | 38.2% | 70.0% | Triton 的 tiling 减少了 HBM 访问 |
| Occupancy | 58.0% | 25.0% | Triton 多 warp/block，手写 1 warp/block |
| 寄存器 | 72 | 96 | Triton 自动优化寄存器分配 |

#### 任务 2：torch.profiler 算子时间线

```python
import torch
import torch.profiler

# 运行 Transformer 单层 forward
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    output = transformer_layer(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

用 Chrome 打开 `chrome://tracing`，加载 `trace.json`，观察：
- GEMM 是否占大头（~45%）
- Softmax/LayerNorm 是否有可 fusion 的空间
- 算子间是否有不必要的同步

#### 任务 3：Softmax 三方 ncu 对比

```bash
ncu --kernel-name regex:softmax \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \
    python3 kernels/benchmark_softmax.py
```

预期：

| Metric | Triton Softmax | CUDA 手写 | PyTorch |
|--------|--------------|----------|---------|
| `dram__throughput` | 85% | 70% | 90% |
| `l1tex__throughput` | 30% | 60% | 25% |
| `sm__occupancy` | 75% | 50% | 80% |

> Softmax 是 memory-bound，`dram__throughput` 高（85%+）说明 HBM 是瓶颈。Triton 的优势是更高的 occupancy（自动 warp 配置）。

#### 任务 4：LeetCode 面试题

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [139](https://leetcode.cn/problems/word-break/) | Medium | DP（一维） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/139_word-break.html) |
| [375](https://leetcode.cn/problems/guess-number-higher-or-lower-ii/) | Medium | DP（区间博弈） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/375_guess-number-higher-or-lower-ii.html) |
| [514](https://leetcode.cn/problems/freedom-trail/) | Hard | DP（状态） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/514_freedom-trail.html) |

---

### 扩展实验

#### 实验 1：Triton 生成的 PTX 分析

```bash
# 导出 Triton 编译的 PTX
TRITON_DUMP_PTX=1 python3 kernels/triton_gemm.py
# 查看生成的 PTX，找到 mma.sync / ldmatrix / cp.async 指令
```

观察：Triton 是否自动生成了 `mma.sync`？是否用了 `cp.async`（double buffer）？

#### 实验 2：Autotune 最优 config 验证

手动跑不同 config，对比 autotune 选的是否最优：
- 强制 `BLOCK_M=64`：性能下降多少？
- 强制 `num_stages=1`（无 double buffer）：性能下降多少？

#### 实验 3：端到端 Transformer Profiling

用 `torch.profiler` 跑一个完整的 Transformer 模型（如 GPT-2 small），分析：
- 哪些算子占时间最多？
- GEMM 占比 vs Softmax/LayerNorm 占比
- 是否有 fusion 机会（相邻 memory-bound 算子）

---

### 今日总结

Day 6 我们用 ncu 和 `torch.profiler` 深入分析了三方 kernel 的内部指标：

1. **Triton 为什么比手写 CUDA 快 2x**：自动用 Tensor Core（68% vs 25%）+ 更高 occupancy（58% vs 25%）+ 自动 smem tiling
2. **Softmax/LayerNorm 的 profiling**：memory-bound，`dram__throughput` 85%+，优化方向是 fusion
3. **torch.profiler 时间线**：GEMM 占 ~45%，Softmax/LayerNorm 各 ~9%，fusion 可省 memory-bound 算子的 HBM 访问
4. **Triton 的自动优化**：autotune 选最优 tiling + `tl.dot` 自动调 Tensor Core + 自动 double buffer
5. **cuBLAS 仍领先的原因**：swizzle / K 分割 / epilogue fusion / 精打细算寄存器

掌握 profiling 后，你有了"用数据解释性能差异"的能力。Day 7 复盘本周全部知识。

---

### 面试要点

1. **Triton GEMM 为什么比手写 CUDA 快 2x？用 ncu 数据解释**

   <details>
   <summary>点击查看答案</summary>

   - **Tensor Core 利用率**：Triton 68% vs 手写 CUDA 25%。`tl.dot` 自动调 WMMA/mma.sync，手写 Day 1 用 FMA
   - **Occupancy**：Triton 58% vs 手写 25%。Triton 自动选 num_warps=4/8，寄存器更少（72 vs 96）
   - **HBM 带宽**：Triton 38% vs 手写 70%。Triton 自动 smem tiling + double buffer 减少 HBM 访问
   - **根本原因**：Triton 编译器自动做了 Day 2-5 手写的全部优化（smem tiling / Tensor Core / double buffer），且自动选最优配置

   </details>

2. **如何用 torch.profiler 分析 Transformer 的性能瓶颈？**

   <details>
   <summary>点击查看答案</summary>

   - `torch.profiler.profile` 录制 CUDA 算子时间线
   - `prof.key_averages().table(sort_by="cuda_time_total")` 打印算子耗时表
   - `prof.export_chrome_trace()` 导出 trace.json，用 Chrome 可视化
   - 分析重点：
     - GEMM 占比（~45%，compute-bound，优化靠 Tensor Core）
     - Softmax/LayerNorm 占比（~15%，memory-bound，优化靠 fusion）
     - 算子间同步开销（是否有不必要的 `cudaDeviceSynchronize`）

   </details>

3. **Softmax 是 memory-bound，ncu 的哪些指标能验证？**

   <details>
   <summary>点击查看答案</summary>

   - `dram__throughput` 85%+ → HBM 带宽是瓶颈
   - `sm__pipe_tensor_op_hmma` 0% → 没用 Tensor Core（Softmax 无矩阵乘）
   - `sm__throughput` 低 → SM 大部分时间在等数据
   - **优化方向**：fusion（Softmax + 相邻算子合并）减少 HBM 访问；Welford 减少扫描次数

   </details>

4. **Triton autotune 选的 config 如何验证是最优的？**

   <details>
   <summary>点击查看答案</summary>

   - 手动跑不同 config（环境变量或修改代码强制指定）
   - 用 ncu 对比各 config 的 Tensor Core 利用率 / occupancy / 带宽
   - 最优 config 应该有：最高 Tensor Core 利用率 + 合理 occupancy + 最低 dram 利用率
   - autotune 的选择逻辑就是"跑全部 config 选最快的"，但可以验证它没选错

   </details>

5. **如果 Triton kernel 性能不达标，你怎么 debug？**

   <details>
   <summary>点击查看答案</summary>

   1. **看 ncu 指标**：Tensor Core 利用率低？occupancy 低？dram bound？
   2. **检查 autotune config**：是否搜了足够多的 config？最优 config 是否合理？
   3. **看生成的 PTX**：`TRITON_DUMP_PTX=1`，检查是否用了 `mma.sync` / `cp.async`
   4. **对比手写 CUDA**：如果手写更快，看 ncu 指标差异，找到 Triton 缺的优化
   5. **考虑 fallback 到 CUDA**：如果 Triton 天花板不够（如需要 TMA/FP8），手写 CUDA

   </details>
