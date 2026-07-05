# Week 3 Day 1 — Transformer 推理流程 Profiling

> 对应 [Week 3 Day 1 晚间编程任务 + 练习题 2/3](../../week3/day1/README.md)

Day 1 的 profiling 有三个层次：torch.profiler（算子级）→ nsys（系统级时间线）→ ncu（kernel 级指标）。

## 1. torch.profiler 分析（算子级时间分解）

```bash
make profile
# 或
python3 trace_transformer.py
```

**输出内容**：
- Prefill 阶段（N=1024）top 算子时间表 → GEMM 占 60%+（compute-bound）
- Decode 阶段（N=1）top 算子时间表 → GEMM 占比下降，softmax/layernorm 上升
- Latency 对比：Prefill 单 token vs Decode 单 token
- torch.compile 对比（kernel fusion 效果）
- Chrome trace 文件（`trace_prefill.json` / `trace_decode.json`）

### 分析任务清单

1. 找出 Prefill 阶段 CUDA 时间 top3 算子（预期 `aten::mm`）
2. 找出 Decode 阶段 CUDA 时间 top3 算子（GEMM 占比下降）
3. 计算 Prefill 单 token 时间 vs Decode 单 token 时间
4. 在 Chrome trace 中观察 kernel 间隙（gap = launch overhead）

## 2. nsys 系统级时间线（练习 2）

```bash
make nsys           # 采集时间线
make nsys-stats     # 查看 kernel 统计
```

或手动运行：

```bash
nsys profile -o transformer_trace python3 trace_transformer.py
nsys stats -t cuda_gpu_kern_sum transformer_trace.nsys-rep
```

### nsys 观察要点

| 观察项 | Prefill 预期 | Decode 预期 | 含义 |
|--------|-------------|-------------|------|
| GEMM kernel 总时间占比 | > 60% | < 40% | Prefill GEMM 主导 |
| Softmax/LayerNorm 占比 | 10-20% | 30-50% | Decode 下相对上升 |
| kernel 间隙（gap） | < 10% | > 20% | Decode launch overhead 大 |
| SM 利用率（GUI 绿色 bar） | 高 | 低 | Decode memory-bound 的直观表现 |

## 3. ncu kernel 级分析

```bash
make ncu            # 分析所有关键 kernel
make ncu-gemm       # 只分析 GEMM kernel
make ncu-softmax    # 只分析 softmax kernel
```

### ncu 观察要点

| Kernel | 阶段 | sm__throughput 预期 | dram__throughput 预期 | 瓶颈类型 |
|--------|------|---------------------|----------------------|---------|
| GEMM (mm) | Prefill (N=1024) | **高**（>60%） | 中 | compute-bound |
| GEMM (mm) | Decode (N=1) | **低**（<20%） | **高**（>60%） | memory-bound |
| Softmax | 两阶段 | 低（<20%） | **高**（>60%） | memory-bound |
| LayerNorm | 两阶段 | 低（<20%） | **高**（>60%） | memory-bound |

### 关键洞察

1. **同一 GEMM kernel 在 Prefill/Decode 下瓶颈类型切换**：
   - Prefill：大矩阵 → AI 高 → compute-bound
   - Decode：M=1 → AI 极低 → memory-bound
2. **Softmax/LayerNorm 始终是 memory-bound**：与 M 无关，AI ≈ 0.4
3. **Decode 的 launch overhead 占比更大**：kernel 小而多，nsys 时间线中 gap 更明显

## 4. torch.compile 对比（练习 3）

`trace_transformer.py` 已内置 `torch.compile(model, mode="reduce-overhead")` 对比：

```python
compiled_model = torch.compile(model, mode="reduce-overhead")
```

**预期**：
- `torch.compile` 会融合 LayerNorm + GEMM 等相邻算子
- kernel 数量减少 30-50%
- Decode 阶段提升更明显（launch overhead 减少占比更大）

## 三层 Profiling 流程

```
① torch.profiler  →  找 top3 算子 + 算子级时间分解
② nsys             →  系统级时间线 + SM 利用率 + kernel 间隙
③ ncu              →  kernel 级 SM/DRAM throughput + stall reasons
```

> 💡 这三层是递进关系：torch.profiler 找"哪个算子慢"→ nsys 找"慢在时间线的哪里"→ ncu 找"为什么这个 kernel 慢"。
