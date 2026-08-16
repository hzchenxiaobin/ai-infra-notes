## Day 4：通信计算重叠 —— 双 Stream + CUDA Graph Overlap

### 🎯 目标

通过今天的学习，你将：

1. 理解**通信与计算的串行执行**问题——TP 每层 all-reduce 时 GPU 算力闲置，NCCL 通信时 SM 空闲<br>
2. 掌握 **双 Stream 重叠**——compute stream + comm stream 并行执行，用 `torch.cuda.Stream` 实现<br>
3. 理解 **CUDA Graph 捕获双流 launch 序列**——减少 kernel launch overhead，固定调度<br>
4. 能实现 **TP 的通信计算重叠**——把一层的计算拆成两半，前半算完就开始 all-reduce 前半，后半同步计算<br>
5. 理解 **overlap 的收益边界**——通信量 vs 计算量的比例决定 overlap 收益<br>

> 💡 **为什么重要**：TP 每层都要 all-reduce，如果通信不与计算重叠，大模型推理的 latency 会被通信串行开销吃掉 20-40%。通信计算重叠是"分布式推理低延迟"的核心工程能力，面试常问"TP 的 all-reduce 怎么和计算重叠"。

---

### 学前导读：串行 vs 重叠

TP 每层 forward：
```
串行:  compute (GEMM) → all-reduce → compute (next GEMM) → all-reduce → ...
         ↑ SM 忙, NCCL 闲      ↑ SM 闲, NCCL 忙      ↑ SM 忙
```

双 Stream 重叠：
```
compute_stream:  compute_1a | compute_1b | compute_2a | ...
comm_stream:          | all_reduce_1a |       | all_reduce_2a | ...
                              ↑ 与 compute_1b 重叠
```

| 策略 | 每层时间 | SM 利用率 |
|------|---------|----------|
| 串行 | compute + comm | comm 时 0% |
| 重叠 | max(compute, comm) | 接近 100% |

> 💡 **一句话总结**：双 Stream 让 NCCL 通信与 CUDA 计算并行执行，把"串行加法"变成"取最大值"。

---

### 理论学习

#### 4.1 CUDA Stream 与并发执行

##### Stream 的基本概念

CUDA Stream 是 GPU 上的异步任务队列：
- 同一 Stream 内的 kernel 串行执行
- 不同 Stream 的 kernel 可并发执行（受资源约束）

```python
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    # compute kernel 提交到 compute_stream
    output = linear(input)

with torch.cuda.stream(comm_stream):
    # NCCL all-reduce 提交到 comm_stream
    dist.all_reduce(output)
```

##### 并发条件

两个 Stream 的 kernel 能真正并发需要：
1. GPU 有空闲 SM（compute kernel 没占满全部 SM）
2. NCCL 通信走 DMA（不占 SM，只占 PCIe/NVLink 带宽）
3. 无数据依赖（或依赖已通过 event wait 解决）

#### 4.2 TP 的通信计算重叠策略

##### 层切分（Layer Splitting）

把一层的 GEMM 按输出维度切两半：

```
Full layer: Y = X @ W  (W: hidden × hidden)

Split: Y1 = X @ W[:, :hidden/2]    (前半输出)
       Y2 = X @ W[:, hidden/2:]    (后半输出)

TP all-reduce Y1 → 重叠 Y2 计算
TP all-reduce Y2 → 重叠下一层前半计算
```

##### 调度时间线

```
compute: |Y1 calc| Y2 calc | next_Y1 calc | next_Y2 calc |
comm:    |       | AR(Y1)  |              | AR(Y2)       |
                  ↑ Y2 计算与 Y1 all-reduce 重叠
```

##### 代码结构

```python
def tp_layer_overlap(x, w1, w2, comm_stream, compute_stream):
    # 前半: compute on compute_stream
    with torch.cuda.stream(compute_stream):
        y1 = x @ w1  # w1 = W[:, :hidden/2]
    # 前半: all-reduce on comm_stream (等 compute 完成)
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_stream(compute_stream)
        dist.all_reduce(y1)
    # 后半: compute on compute_stream —— 与 y1 的 all-reduce 并行重叠
    # 注意：y2 = x @ w2 不依赖 y1，因此 compute_stream 不应 wait_stream(comm_stream)。
    # 若误加该 wait，y2 会被强制等 y1 的 all-reduce 完成，本日要演示的"通信计算重叠"
    # 就退化为串行执行。这正是初学者常踩的坑：把"两段计算共享输入 x"误读成"有数据依赖"。
    with torch.cuda.stream(compute_stream):
        y2 = x @ w2  # w2 = W[:, hidden/2:]
    # 后半: all-reduce on comm_stream
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_stream(compute_stream)
        dist.all_reduce(y2)
    return torch.cat([y1, y2], dim=-1)
```

#### 4.3 CUDA Graph 捕获双流序列

##### 为什么要用 CUDA Graph？

双 Stream 代码每层要提交多个 kernel launch（compute + comm + wait），launch overhead 累积：
- 每 kernel launch ~5-10μs
- 80 层 × 4 launch/层 = 320 launch × 7μs = 2.2ms 纯 launch 开销

CUDA Graph 把整个 launch 序列"录下来"，一次性重放：
```python
# 捕获
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for layer in model.layers:
        tp_layer_overlap(x, layer.w1, layer.w2, comm_stream, compute_stream)

# 重放（一次 launch）
graph.replay()
```

##### 收益

| 策略 | launch 次数 | launch 开销 |
|------|-----------|-----------|
| 无 Graph | 320+ | ~2.2ms |
| CUDA Graph | 1 | ~10μs |

> 💡 **生产实践**：vLLM/TensorRT-LLM 的推理引擎都用 CUDA Graph 捕获整个 forward 序列。但 Graph 要求静态 shape，shape 变化时需重新捕获或用 shape bucketing。

#### 4.4 Overlap 的收益边界

##### 收益公式

```
串行时间: T_compute + T_comm
重叠时间: max(T_compute, T_comm)
收益: T_compute + T_comm - max(T_compute, T_comm) = min(T_compute, T_comm)
```

##### 收益条件

| 情况 | T_compute vs T_comm | 重叠收益 |
|------|---------------------|---------|
| compute-bound | T_compute >> T_comm | 通信完全被遮盖，收益 ≈ T_comm |
| comm-bound | T_comm >> T_compute | 计算完全被遮盖，收益 ≈ T_compute |
| 均衡 | T_compute ≈ T_comm | 收益 ≈ T_compute = T_comm（最大相对收益） |

##### TP 的典型比例

N=8 TP, 70B 模型, batch=1, seq=2048, FP16:
```
T_compute/层 ≈ 5ms (GEMM)
T_comm/层 ≈ 2ms (all-reduce 28.7MB, NVLink 400GB/s → 0.07ms? 实际有 latency)
```

compute >> comm，重叠收益有限（comm 已被自然遮盖）。但在大 batch 或长 seq 下，comm 占比上升，重叠收益增大。

#### 4.5 Sequence Parallelism（Megatron 2205.05198）

##### 为什么需要 Sequence Parallelism？

标准 TP 的 all-reduce 通信量 = $2 \times \text{tokens} \times \text{hidden}$，其中 $\text{tokens} = \text{batch} \times \text{seq\_len}$。当序列很长（如 32K/128K context）时，activation 的 all-reduce 通信量与 seq_len 成正比，成为瓶颈。

Megatron-LM 的 Sequence Parallelism（SP）核心思路：**把 seq_len 维也切分到各卡**，每卡只持有 1/N 的 sequence，减少 activation 通信量。

##### SP 与 TP 的协作

SP 不替换 TP，而是**与 TP 配合**——在 TP 的非 Attention 部分（LayerNorm、Dropout、Residual）把 sequence 切分：

```
标准 TP（无 SP）：
  LayerNorm:  activation [B, S, H] → 每卡全量 → all-reduce 聚合
  Attention:  QKV column-parallel → 各 head 独立 → all-reduce 聚合
  → 通信量/层 = 2 × B × S × H × sizeof(dtype)  (S = 全序列)

SP + TP：
  LayerNorm:  activation [B, S/N, H] → 每卡 1/N 序列 → 无 all-reduce
  Attention:  all-gather 把 [B, S/N, H] → [B, S, H] → 正常 TP attention → reduce-scatter 回 [B, S/N, H]
  → LayerNorm 部分通信量 = 0；Attention 部分用 all-gather + reduce-scatter 替代 all-reduce
  → 总通信量不变（2 × B × S × H），但 LayerNorm 的 activation 显存降为 1/N
```

##### SP 的关键改进：all-reduce 拆成 reduce-scatter + all-gather

标准 TP 用 all-reduce（2(N-1)/N × V），SP 把它拆开：

| 阶段 | 标准 TP | SP |
|------|---------|-----|
| LayerNorm 后 | all-reduce [B,S,H] | reduce-scatter → [B, S/N, H] |
| Attention 前 | — | all-gather → [B, S, H] |
| Attention 后 | — | reduce-scatter → [B, S/N, H] |
| Output 投影后 | all-reduce [B,S,H] | all-gather → [B, S, H] |

**通信量相同**（all-reduce = reduce-scatter + all-gather），但拆开后可以**与计算重叠**——reduce-scatter/all-gather 各自只传一半数据，更容易被 GEMM 计算掩盖。这正是 §4.2 双流重叠策略的理想应用场景。

##### SP 的收益

| 维度 | 标准 TP | SP + TP |
|------|---------|---------|
| LayerNorm activation 显存 | $O(B \times S \times H)$ | $O(B \times S/N \times H)$（降 1/N） |
| 通信量 | 2V(N-1)/N | 相同（但拆成两半可重叠） |
| 重叠效率 | all-reduce 难重叠（整块传） | reduce-scatter/all-gather 易重叠（半块） |
| 适用场景 | 短序列 | 长序列（S > 4K 时收益显著） |

> 💡 **面试要点**：SP 是 Megatron-LM 的工业界主流做法，核心不是减通信量（不变），而是：(1) 减 LayerNorm 的 activation 显存（长序列时 LN 的 activation 占比不可忽略）；(2) 把 all-reduce 拆成 reduce-scatter + all-gather，便于通信计算重叠。vLLM/TensorRT-LLM 在长上下文场景已开始采用 SP。

> 📖 **延伸阅读**：Megatron-LM 论文 "Reducing Activation Recomputation in Large Transformer Models" (2205.05198)，Section 5 详细推导 SP 的通信量与显存。

---

### Coding 任务

#### 任务 1：双 Stream TP 重叠 Demo

```python
import torch
import torch.distributed as dist

def tp_forward_overlap(x, w1, w2, comm_stream, compute_stream):
    with torch.cuda.stream(compute_stream):
        y1 = x @ w1
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_stream(compute_stream)
        dist.all_reduce(y1)
    with torch.cuda.stream(compute_stream):
        y2 = x @ w2  # 与 y1 all-reduce 重叠
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_stream(compute_stream)
        dist.all_reduce(y2)
    compute_stream.wait_stream(comm_stream)
    return torch.cat([y1, y2], dim=-1)

# benchmark: 串行 vs 重叠
def bench_serial(x, w, iters=100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = x @ w
        dist.all_reduce(y)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def bench_overlap(x, w1, w2, iters=100):
    # ... 双 stream 版 ...
    pass
```

#### 任务 2：CUDA Graph 捕获

```python
def capture_graph(model, input):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(input)
    return graph

# benchmark: 无 Graph vs Graph
graph = capture_graph(model, sample_input)
torch.cuda.synchronize()
start.record()
for _ in range(100):
    graph.replay()
end.record()
```

#### 任务 3：LeetCode 面试题（10 周计划 · 第 9 周 Day 4 复盘）

> 📅 今日为 [10 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/10-week-plan.html) 第 9 周「动态规划进阶——子序列、区间与二维 DP」复盘日。重做本周错题、总结模板笔记；没做完的题目今天补上。

---

### 今日总结

1. **双 Stream 重叠**：compute stream + comm stream 并发，把串行 T_comp+T_comm 变为 max(T_comp, T_comm)
2. **TP 层切分**：GEMM 按输出分两半，前半 all-reduce 与后半计算重叠
3. **CUDA Graph**：捕获双流 launch 序列，消除 320+ kernel launch 的 ~2ms 开销
4. **收益边界**：compute-bound 时通信被自然遮盖，收益小；comm-bound 时收益大
5. **生产实践**：vLLM/TRT-LLM 用 CUDA Graph + 双 Stream 做推理低延迟

---

### 面试要点

1. **TP 的 all-reduce 怎么和计算重叠？**

   <details>
   <summary>答案</summary>

   - 把一层 GEMM 按输出维度切两半（Y1, Y2）
   - compute_stream 算 Y1 → comm_stream all-reduce Y1
   - compute_stream 同时算 Y2（与 Y1 的 all-reduce 重叠）
   - comm_stream all-reduce Y2
   - 收益：把 T_comp + T_comm 变为 max(T_comp, T_comm)

   </details>

2. **CUDA Graph 在分布式推理中起什么作用？**

   <details>
   <summary>答案</summary>

   - 捕获整个 forward 的 kernel launch 序列（含双 Stream 的 compute + comm + wait）
   - 重放时一次 launch，消除 320+ kernel 的 launch overhead（~2ms → ~10μs）
   - 代价：要求静态 shape，shape 变化需重新捕获或 shape bucketing
   - 生产：vLLM/TRT-LLM 的推理引擎都用 Graph

   </details>

3. **通信计算重叠的收益什么时候最大？**

   <details>
   <summary>答案</summary>

   - 收益 = min(T_compute, T_comm)
   - compute-bound（T_comp >> T_comm）：通信被自然遮盖，收益小
   - comm-bound（T_comm >> T_comp）：计算被遮盖，收益大
   - 均衡（T_comp ≈ T_comm）：相对收益最大（时间减半）
   - 大 batch / 长 seq 时 comm 占比上升，重叠收益增大

   </details>
