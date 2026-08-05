## Day 1：分布式推理 —— TP/PP/DP 与通信计算重叠分布式推理专题 —— TP/PP/DP 与通信计算重叠

### 🎯 目标

通过今天的学习，你将：

1. 理解**为什么单卡不够**——模型太大（70B FP16 ≈ 140GB，单卡 A100 80GB 放不下）、吞吐需求（单卡 tok/s 上限低）、延迟需求（单层 GEMM 算力受限）三大动机<br>
2. 掌握 **Tensor Parallelism (TP)**——column-parallel linear（QKV 投影按 head 切）、row-parallel linear（Output 投影按 input 切 + all-reduce）、TP=2/4/8 的通信/算力权衡<br>
3. 掌握 **Pipeline Parallelism (PP)**——GPipe vs 1F1B、micro-batching、bubble ratio 公式 $\text{bubble} = (P-1)/(M+P-1)$<br>
4. 理解 **Data Parallelism (DP)** 在推理中的定位——与 TP/PP 的区别，适合高吞吐多请求场景而非单大模型放置<br>
5. 掌握 **NCCL collectives**——all-reduce（ring 拓扑，$2(N-1)$ 步通信）、all-gather、reduce-scatter 的通信量对比<br>
6. 学会**通信计算重叠**——`torch.cuda.Stream` 双流（compute stream + comm stream）、CUDA Graph 捕获双流 launch 序列

> 💡 **为什么重要**：Day 3 分析了 SGLang/LightLLM 的高级特性（Speculative Decoding、Chunked Prefill、Prefix Caching），但它们都假设"模型能放进单卡"。当模型大到单卡放不下（如 70B+），或吞吐需求超过单卡上限时，就必须上**分布式推理**。TP/PP/DP 是分布式推理的三大基石，NCCL 通信与通信-计算重叠是把"分布式开销"压到最低的关键工程能力——这是面试"大模型分布式推理"的高频考点，也是 Mini 引擎从"单卡 demo"走向"多卡生产"的必经之路。

---

### 学前导读：为什么单卡不够

Day 1-3 的 Mini 引擎都跑在单卡上。但当模型规模和业务需求增长时，单卡会从三个维度撞墙：

```
单卡撞墙的三个维度：
  1. 显存墙：70B FP16 权重 ≈ 140GB，A100 80GB / H100 80GB 单卡放不下
     → 需要 TP/PP 把权重切到多卡
  2. 吞吐墙：单卡 decode tok/s 有上限（如 ~2000 tok/s），高 QPS 场景不够
     → 需要 DP 多副本并行处理请求
  3. 延迟墙：单层大 GEMM 受单卡算力限制，prefill 长 prompt 时延迟高
     → 需要 TP 把单层 GEMM 拆到多卡，降低单步延迟
```

| 维度 | 单卡瓶颈 | 分布式方案 | 代价 |
|------|---------|-----------|------|
| 显存 | 模型放不下 | TP（切权重）/ PP（切层） | 通信开销 |
| 吞吐 | tok/s 上限 | DP（切数据，多副本） | 显存翻倍 + grad all-reduce |
| 延迟 | 单层 GEMM 慢 | TP（单层算力 x N） | 每层 all-reduce |

> 💡 **一句话总结**：TP 解决"放不下 + 单层慢"，PP 解决"放不下"，DP 解决"吞吐不够"。实际系统常组合 DP+TP 或 DP+PP——外层 DP 切请求扩吞吐，内层 TP/PP 切模型放显存。

---

### 理论学习

#### 3b.1 为什么需要分布式推理

分布式推理的三大动机，对应三种不同的并行策略：

##### 动机 1：模型太大，单卡显存放不下

```
模型显存估算（FP16，2 bytes/param）：
  7B  → 14 GB   （单卡 A100 80GB 轻松）
  13B → 26 GB   （单卡够，但 KV Cache 空间紧张）
  70B → 140 GB  （单卡放不下，必须切）
  175B→ 350 GB  （需 4-8 卡 TP/PP）
额外显存：KV Cache（随 batch 和序列长度增长）+ activation
```

> ⚠️ **KV Cache 容易被忽略**：70B 模型即便权重切到 2 卡（每卡 70GB），KV Cache 在大 batch 长序列下可能再吃几十 GB，导致单卡仍 OOM。这就是为什么 TP=2 不一定够，常需 TP=4/8。

##### 动机 2：吞吐需求（throughput）

单卡 decode 吞吐有上限（受算力和带宽限制）。高 QPS 场景（如多人同时聊天）需要多副本并行——这就是 DP。每张卡跑一个完整模型副本，独立处理不同请求，吞吐近似线性扩展。

##### 动机 3：延迟需求（latency）

单层大 GEMM（如 70B 的 FFN，矩阵 [B, 8192]×[8192, 28672]）在单卡上耗时较长。TP 把这个 GEMM 切到 N 卡，每卡算 1/N，单步延迟近似降至 1/N（扣除通信开销后）。

#### 3b.2 Tensor Parallelism (TP)

![TP/PP/DP 三种分布式并行对比](../images/tp_pp_dp_overview.svg)

TP 的核心思想：**把每一层的权重矩阵切到多卡，各卡并行计算同一层的不同部分，通过通信聚合结果**。

##### Column-Parallel Linear（QKV 投影用）

权重 $W \in \mathbb{R}^{out \times in}$ 按 **output 维**切分，每个 rank 持有 $W_i \in \mathbb{R}^{out/N \times in}$：

```
输入 X[B, in]  → 各 rank 相同（broadcast 或各 rank 自有）
各 rank 计算 Y_i = X @ W_i^T  →  Y_i[B, out/N]
输出：各 rank 持有 [B, out/N]，按 head 天然并行
通信量：0（无需 all-reduce，下游 attention 按 head 消费）
```

> 💡 **为什么 QKV 用 column-parallel**：Attention 天然按 head 分组，column 切分正好让每个 rank 持有若干 head 的 Q/K/V，后续 attention 计算完全本地化，无需跨 rank 通信。

##### Row-Parallel Linear（Output 投影用）

权重 $W \in \mathbb{R}^{out \times in}$ 按 **input 维**切分，每个 rank 持有 $W_i \in \mathbb{R}^{out \times in/N}$：

```
输入 X_i[B, in/N]  → 各 rank 不同（来自上一层 column 切分）
各 rank 计算 Y_i = X_i @ W_i^T  →  Y_i[B, out]（部分和）
输出：all-reduce(sum) 聚合各 rank 部分和 → Y[B, out]
通信量：all-reduce [B, out] 元素
```

##### TP 的经典 pattern：Column + Row 配对

一个 Attention Block = ColumnParallel(QKV) + Attention + RowParallel(Output)，**全程只需 1 次 all-reduce**（在 Output 投影处）：

```
X → ColumnParallel(QKV) → [各 rank 算自己 head 的 QKV，无通信]
  → Attention（各 head 独立，无通信）
  → RowParallel(Output) → [各 rank 算部分和，1 次 all-reduce 聚合]
  → Y
```

##### TP=2/4/8 权衡

| TP size | 算力提升 | 每层通信量 | 显存节省 | 适用场景 |
|---------|---------|-----------|---------|---------|
| TP=2 | x2 | 1 次 all-reduce（小） | 权重 ÷2 | 13B-34B 模型 |
| TP=4 | x4 | 1 次 all-reduce（中） | 权重 ÷4 | 70B 模型 |
| TP=8 | x8 | 1 次 all-reduce（大） | 权重 ÷8 | 175B 模型 |

> ⚠️ **TP 不是越大越好**：TP 增大后，all-reduce 通信量不变（每层都全量 all-reduce 输出），但每卡算的 GEMM 变小（算力利用率下降），且通信占比升高。通常 TP≤8，再大用 PP 补充。

#### 3b.3 Pipeline Parallelism (PP)

![NCCL Ring All-Reduce 拓扑](../images/nccl_ring_topology.svg)

PP 的核心思想：**把模型的不同层切到不同卡（stage），数据像流水线一样流过各 stage**。

##### GPipe vs 1F1B

```
GPipe（朴素流水线）：
  - 一次性把整个 mini-batch forward 完所有 stage，再 backward
  - 空泡大：前 P-1 步只有部分 stage 在工作
  - 显存大：要存所有 micro-batch 的 activation

1F1B（One Forward One Backward）：
  - 交错执行 forward 和 backward：做完一个 micro-batch 的 forward 立刻 backward
  - 空泡一样大，但显存省（activation 数量稳定在 P）
  - 推理只需 forward，1F1B 的 backward 优势不体现，但 prefill 可借鉴 micro-batch 流水
```

##### Micro-batching 与 bubble ratio

把一个 mini-batch 切成 $M$ 个 micro-batch，依次注入 $P$ 个 stage 的流水线：

$$\text{bubble ratio} = \frac{P-1}{M+P-1}$$

| P (stage) | M (micro-batch) | bubble | 说明 |
|-----------|----------------|--------|------|
| 4 | 1 | 75% | 空泡占主导，效率低 |
| 4 | 4 | 43% | 仍较差 |
| 4 | 16 | 16% | 较好 |
| 8 | 64 | 10% | 大 M 下空泡可忽略 |

> 💡 **关键洞察**：增大 $M$（micro-batch 数）是降低 bubble 的主要手段。推理场景下，prefill 长 prompt 可切成多个 chunk 走流水线（与 Day 3 的 Chunked Prefill 思路一致），相当于增大 $M$。

##### 推理场景下的 PP

推理只有 forward，没有 backward，PP 的收益主要在**显存**（每卡只存部分层）。代价是 stage 间点对点通信（send/recv activation）和流水线空泡。PP 适合**超大模型**（TP=8 仍放不下时叠加 PP）。

#### 3b.4 Data Parallelism (DP)

DP 的核心思想：**每张卡持有完整模型副本，各卡处理不同的请求/batch**。

##### DP vs TP/PP 的区别

| 维度 | DP | TP / PP |
|------|----|---------|
| 切分对象 | 输入数据 | 模型权重 |
| 每卡模型 | 完整副本 | 部分权重 |
| 通信 | grad all-reduce（训练）/ 无或很少（推理） | 每层 all-reduce（TP）/ send-recv（PP） |
| 扩展性 | 吞吐近似线性 | 受通信限制 |

##### DP 在推理中的定位

- **训练**：DP 是主流，每步 backward 后 grad all-reduce
- **推理**：DP 更简单——各卡独立处理请求，**无需跨卡通信**（除了请求路由）。吞吐近似线性扩展，是高 QPS 场景的首选
- **推理 DP 的通信**：仅请求分发（CPU 层面），GPU 间几乎无通信

> 💡 **推理 DP 的优势**：零 GPU 通信、实现简单、吞吐线性扩展。劣势是每卡要放完整模型（显存不省）。所以"模型放得下单卡"时，DP 是推理扩吞吐的最佳选择。

#### 3b.5 NCCL collectives

NCCL（NVIDIA Collective Communications Library）是 GPU 间集合通信的标准库，PyTorch 通过 `torch.distributed` 调用。

##### all-reduce（TP/DP 最常用）

所有 rank 的张量做逐元素规约（sum/max/avg），结果分发到所有 rank。

**Ring all-reduce** 分两阶段，共 $2(N-1)$ 步：

```
阶段 1：reduce-scatter（N-1 步）
  - 每步：每卡 send 一块给右邻 + recv 左邻的块，累加
  - N-1 步后：每卡持有一个块的完整和
  - 通信量 = V × (N-1)/N（V = 张量大小）

阶段 2：all-gather（N-1 步）
  - 每步：每卡把自己那块完整和 send 给右邻 + recv
  - N-1 步后：每卡都有完整结果
  - 通信量 = V × (N-1)/N

总计：2 × V × (N-1)/N
```

##### all-gather / reduce-scatter

| collective | 作用 | 通信量 | 典型用途 |
|-----------|------|--------|---------|
| **all-reduce** | 全聚合 + 全分发 | $2V(N-1)/N$ | TP 输出聚合、DP 梯度同步 |
| **all-gather** | 各 rank 拼接不同块 | $V(N-1)/N$ | column-parallel 输出收集 |
| **reduce-scatter** | 全聚合 + 分散结果 | $V(N-1)/N$ | 拆分 all-reduce 的前半段 |
| **broadcast** | 单卡数据广播到所有 | $V$ | 初始化、参数同步 |
| **send/recv** | 点对点传输 | $V$ | PP stage 间传 activation |

> 💡 **为什么用 ring**：ring 拓扑每卡只与左右邻居通信，带宽利用率高、无拥塞点；通信量与 $N$ 无关（$2V(N-1)/N \approx 2V$），只步数随 $N$ 线性增长。NCCL 还支持 tree 拓扑（延迟更低，适合小数据）和 NVLink/InfiniBand 自适应。

##### 通信量对比示例（V = 1GB, N=8）

```
all-reduce:    2 × 1GB × 7/8 ≈ 1.75 GB  （每卡发送+接收）
all-gather:    1GB × 7/8    ≈ 0.875 GB
reduce-scatter: 1GB × 7/8   ≈ 0.875 GB
```

#### 3b.6 通信计算重叠

![通信-计算重叠：双 CUDA Stream](../images/comm_compute_overlap.svg)

分布式推理的通信开销如果不能被计算掩盖，TP/PP 的加速比会被严重吃掉。**通信-计算重叠**是核心优化手段。

##### 双流（dual stream）方案

```
compute_stream: 执行 GEMM / Attention 等前向计算
comm_stream:    执行 all-reduce / send-recv 等通信

不重叠（串行）：
  compute_stream: [==== GEMM ====]
  comm_stream:                            [== all-reduce ==]
  total = T_compute + T_comm

重叠（双流并行）：
  compute_stream: [==== GEMM ====]
  comm_stream:        [== all-reduce ==]   ← 与 GEMM 并行
  total ≈ max(T_compute, T_comm)
```

`torch.cuda.Stream` 实现要点：

```python
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    y = gemm(x)                      # compute on compute_stream
with torch.cuda.stream(comm_stream):
    torch.distributed.all_reduce(g)  # comm on comm_stream
# 两流并发执行，GPU 硬件调度（需有空闲 SM）
```

##### 重叠的前提条件

- **数据无依赖**：通信的数据与计算的数据不重叠（如本层 GEMM 与上一层的 all-reduce 可重叠）
- **GPU 资源足够**：两路 kernel 需有足够空闲 SM 并发，否则仍串行
- **流同步正确**：有依赖时用 `stream.wait_stream()` 建立顺序

##### CUDA Graph + 通信重叠

decode 阶段每步 shape 固定，可用 CUDA Graph 捕获整个 forward（含双流 launch 序列），消除 Python/CPU launch 开销：

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    with torch.cuda.stream(compute_stream):
        compute(...)
    with torch.cuda.stream(comm_stream):
        all_reduce(...)
# 每步 decode 只需 g.replay()，一次 launch 整个图
```

> ⚠️ **CUDA Graph 的限制**：要求 shape 静态（decode 每步 token 数固定，适合；prefill shape 变化，需多个 graph 或回退 eager）。vLLM/TensorRT-LLM 在 decode 路径广泛使用 CUDA Graph。

### Coding 任务：TP 推理 demo + 通信重叠

#### 任务 1：创建 tp_inference_demo.py

创建文件 [kernels/tp_inference_demo.py](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/week9/day1/kernels/tp_inference_demo.py)，用 `torch.chunk` 在单卡上模拟 2-GPU TP 推理：

```python
# tp_inference_demo.py —— 2-GPU Tensor Parallelism 推理 Demo（单卡模拟）
# 运行命令: python tp_inference_demo.py
# 依赖: torch（CPU 或单 GPU 均可）

class ColumnParallelLinear(nn.Module):
    """列并行：W 按 output dim 切分，各 rank 输出 [B, out/tp]，无需通信"""
    def forward(self, x):
        shards = [x @ w.t() + b for w, b in zip(self.weights, self.bias)]
        return torch.cat(shards, dim=-1)

class RowParallelLinear(nn.Module):
    """行并行：W 按 input dim 切分，各 rank 输出部分和，all-reduce 聚合"""
    def forward(self, x):
        x_shards = torch.chunk(x, self.tp_size, dim=-1)
        partial = [x_i @ w.t() for x_i, w in zip(x_shards, self.weights)]
        # 模拟 all-reduce(sum)：真实场景用 torch.distributed.all_reduce(y)
        return torch.stack(partial, dim=0).sum(dim=0) + self.bias

class TPAttentionBlock(nn.Module):
    """ColumnParallel(QKV) + Attention + RowParallel(Output)，1 次 all-reduce"""
    ...
```

完整代码见 [kernels/tp_inference_demo.py](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/week9/day1/kernels/tp_inference_demo.py)。

代码要点：
- `ColumnParallelLinear`：权重按 output dim 切成 `tp_size` 片，各片独立计算后 `torch.cat` 模拟拼接（真实 TP 下游按 head 消费，常无需 concat）
- `RowParallelLinear`：权重按 input dim 切，输入也按 input dim chunk，各 rank 算部分和后 `torch.stack().sum()` 模拟 all-reduce
- `TPAttentionBlock`：组合 Column(QKV) + Attention + Row(Output)，验证一个 block 仅需 1 次通信
- `single_gpu_forward`：用拼接后的完整权重做等价单卡 forward，用于正确性对比

#### 任务 2：运行 tp_inference_demo.py 并验证正确性

```bash
python kernels/tp_inference_demo.py
```

**预期输出**（节选）：

```text
================================================================
  正确性测试：TP=2 模拟 vs 单卡等价
================================================================
  output shape : TP=(2, 128, 512), single=(2, 128, 512)
  max abs diff : 2.53e-07
  result       : PASS

================================================================
  通信模式分析（TP=2，一个 Attention Block）
================================================================
  ColumnParallelLinear (QKV 投影):
    通信量: 0（无需 all-reduce，下游 attention 按 head 消费）
  RowParallelLinear (Output 投影):
    通信量: all-reduce(sum) [B, out] 元素，FP32 每元素 4B
  => 一个 Attention Block 共 1 次 all-reduce（在 output 投影处）

================================================================
  Demo 完成！TP 推理模拟 通过
```

##### 观察重点

1. **正确性**：TP=2 模拟输出与单卡等价输出 max diff < 1e-5（浮点累加顺序导致微小误差）
2. **通信 pattern**：一个 Attention Block 仅 1 次 all-reduce（Row(Output) 处），Column(QKV) 零通信
3. **单卡模拟局限**：无法体现真实多卡加速（GEMM 算力 x2），仅验证切分正确性与通信结构

#### 任务 3：创建并运行 comm_overlap_demo.py + nsys profiling

创建文件 [kernels/comm_overlap_demo.py](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/week9/day1/kernels/comm_overlap_demo.py)，演示双 CUDA Stream 通信-计算重叠：

```python
# comm_overlap_demo.py —— 通信-计算重叠 Demo（双 CUDA Stream）
# 运行命令: python comm_overlap_demo.py
# Profiling: nsys profile -o comm_overlap --trace=cuda python comm_overlap_demo.py

compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

def serial_step():
    dummy_gemm(compute_stream, SIZE)       # compute 先做完
    comm_stream.wait_stream(compute_stream) # 依赖：comm 等 compute
    dummy_comm(comm_stream, SIZE)

def overlap_step():
    dummy_gemm(compute_stream, SIZE)       # 两流无依赖，并行 launch
    dummy_comm(comm_stream, SIZE)

# 用 torch.cuda.Event 计时
serial_ms = measure(serial_step, ...)
overlap_ms = measure(overlap_step, ...)
```

完整代码见 [kernels/comm_overlap_demo.py](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/week9/day1/kernels/comm_overlap_demo.py)。

运行与 profiling：

```bash
# 运行（需 CUDA 环境）
python kernels/comm_overlap_demo.py

# nsys 采集时间线
nsys profile -o comm_overlap --trace=cuda,nvtx python kernels/comm_overlap_demo.py

# 查看报告
nsys-ui comm_overlap.nsys-rep
# 在 timeline 中应看到 compute_stream 与 comm_stream 的 kernel 交错或重叠
```

**预期输出**（节选，具体数值随 GPU 而变）：

```text
================================================================
  通信-计算重叠 Demo（双 CUDA Stream）
================================================================
  iters=10, size=2048x2048 FP16

  compute alone : 0.85 ms
  comm alone    : 0.62 ms

  [串行] total  : 1.47 ms  (compute + comm = 1.47)
  [重叠] total  : 0.91 ms  (max = 0.85)

  加速比        : 1.62x
  理论上限      : 1.73x (完全重叠)
```

##### 观察重点

1. **串行**：total ≈ compute + comm（两段串接）
2. **重叠**：total ≈ max(compute, comm)（两流并发，受限于较长者）
3. **加速比**：1.5-1.7x（取决于 GPU 空闲 SM 是否足够容纳两路 kernel）
4. **nsys timeline**：compute_stream 和 comm_stream 的 kernel 在时间轴上交错或重叠

> ⚠️ **单 GPU 重叠局限**：两路 kernel 能否真正并发取决于 GPU 是否有空闲 SM。大 GEMM 占满所有 SM 时，通信 kernel 会被迫排队。真实多卡场景下，通信走 NVLink/IB（独立硬件），与计算 SM 完全解耦，重叠效果更好。

#### 任务 4：LeetGPU 在线题目 —— Matrix Copy

**题目链接**：<https://leetgpu.com/challenges/matrix-copy>

**与今日知识的关联**：分布式推理的核心开销是**通信**（all-reduce / send-recv），而通信的本质是**数据在 GPU 间搬运**——与 Matrix Copy 同构：都是 bandwidth-bound 的纯数据搬移。Matrix Copy 练习的是如何高效搬运（coalesced 读写、避免 bank conflict、用满显存带宽），这正是 NCCL kernel 内部的优化目标。理解 Matrix Copy 的带宽利用率分析，就能估算 all-reduce 的通信下限：`T_comm = V / bandwidth`。做好这题说明你掌握了"数据搬运的性能上限"，是分析通信开销的基础。

> 💡 提交后在 [LeetGPU Matrix Copy](https://leetgpu.com/challenges/matrix-copy) 上记录通过耗时。完整题解见 [Matrix Copy 题解](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-copy-solution.html)。

#### 任务 5：LeetCode 面试题（8 周计划 · 第 7 周 补充）

> 📅 今日为分布式推理专题补充日，LeetCode 从 [8 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/8-week-plan.html) 第 7 周「二分查找与动态规划基础」中精选 4 道高频题（二分模板 + 一维 DP + 背包 DP），巩固本周算法基础。简单题快速过、中等题精做；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|----------|------|
| [704. 二分查找](https://leetcode.cn/problems/binary-search/) | 简单 | 二分模板（闭区间 / 左闭右开） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/704_二分查找.html) |
| [198. 打家劫舍](https://leetcode.cn/problems/house-robber/) | 中等 | 一维 DP（选/不选状态转移） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/198_打家劫舍.html) |
| [322. 零钱兑换](https://leetcode.cn/problems/coin-change/) | 中等 | 完全背包 DP（求最少硬币） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/322_零钱兑换.html) |
| [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/) | 中等 | 0-1 背包 DP（求能否装满） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/416_分割等和子集.html) |

> 💡 刷题建议：704 是二分模板，5 分钟默写一遍确保边界无错；198 是 DP 入门，理解"选/不选"状态转移；322 和 416 分别是完全背包与 0-1 背包，对比记忆"内外层循环顺序"的差异（完全背包正向、0-1 背包逆向）。

---

### 扩展实验

#### 实验 1：把 TP 模拟改成 TP=4

修改 `TP_SIZE = 4`（要求 `n_heads % 4 == 0`），重新运行正确性测试。观察：max diff 是否仍 < 1e-5？通信 pattern 分析中"每 block 1 次 all-reduce"是否不变？

> 思考：TP=4 时单卡 GEMM 算力 x4，但 all-reduce 通信量是否变化？（提示：不变，每层仍全量 all-reduce 输出 [B, out]。这正是 TP 通信占比随 N 升高的原因。）

#### 实验 2：用 torch.distributed 真实多卡跑 TP

若有 2+ GPU，把 `ColumnParallelLinear` / `RowParallelLinear` 中的 `torch.cat` / `torch.stack().sum()` 替换为真实的 `torch.distributed.all_reduce(y, op=ReduceOp.SUM)`。用 `torchrun --nproc_per_node=2 tp_inference_demo.py` 启动。对比真实多卡下的加速比与单卡模拟的差异。

> 思考：真实 2-GPU TP 的加速比为什么远不到 2x？（提示：all-reduce 通信开销 + kernel launch + 数据依赖。可通过 nsys 测量通信占比。）

#### 实验 3：实现"上一层 all-reduce 与本层 GEMM 重叠"

当前 `comm_overlap_demo.py` 演示的是无依赖的 GEMM + 通信重叠。修改为更贴近真实 TP 的场景：上一层的 all-reduce（comm_stream）与本层的 QKV GEMM（compute_stream）重叠——两者数据无依赖（all-reduce 的是上一层输出，GEMM 用的是本层权重 + 上一层已就绪的输入）。用 nsys 验证两流 kernel 是否真的交错。

> 思考：为什么"上一层通信"能与"本层计算"重叠，而"同一层的通信"不能？（提示：同层内 Output 投影的 all-reduce 依赖该投影的部分和，存在数据依赖；跨层则无。）

---

### 今日总结

Day 1 我们系统学习了分布式推理的三大并行策略与通信-计算重叠：

1. **三大动机**：显存墙（模型放不下）、吞吐墙（tok/s 不够）、延迟墙（单层 GEMM 慢），分别对应 TP/PP、DP、TP
2. **Tensor Parallelism**：column-parallel 切 QKV（按 head，零通信）+ row-parallel 切 Output（all-reduce 聚合），一个 Attention Block 仅 1 次通信
3. **Pipeline Parallelism**：按层切 stage，micro-batching 流水，bubble ratio = $(P-1)/(M+P-1)$，增大 $M$ 降空泡；1F1B 省 activation 显存
4. **Data Parallelism**：推理中各卡独立处理请求，零 GPU 通信，吞吐线性扩展，是高 QPS 首选
5. **NCCL collectives**：all-reduce = reduce-scatter + all-gather，ring 拓扑 $2V(N-1)/N$ 通信量；all-gather / reduce-scatter 各 $V(N-1)/N$
6. **通信-计算重叠**：双 CUDA Stream（compute + comm）让 total 从 $T_c+T_a$ 降到 $\max(T_c, T_a)$；CUDA Graph 捕获双流序列消除 launch 开销
7. **实测验证**：`tp_inference_demo.py` 验证 TP 切分正确性（max diff 2.5e-7）与通信 pattern；`comm_overlap_demo.py` 量化重叠加速比 1.5-1.7x

掌握这些后，你就有了分布式推理的理论基础——后续可结合 vLLM 的多卡支持（TP 后端）和 TensorRT-LLM 的 TP+PP 组合实践真实多卡部署。

---

### 面试要点

1. **TP 下 QKV 怎么切？为什么这样切？**（⭐⭐⭐⭐⭐ 必考）

<details>
<summary>点击查看答案</summary>

- **QKV 投影用 column-parallel**：权重 $W \in \mathbb{R}^{3d \times d}$ 按 **output 维**切分，每个 rank 持有 $W_i \in \mathbb{R}^{3d/N \times d}$
- **切 output 维的原因**：
  - Attention 天然按 head 分组，column 切分让每个 rank 持有若干完整 head 的 Q/K/V
  - 后续 attention 计算（$QK^T$、softmax、$\cdot V$）各 head 独立，完全本地化，**无需跨 rank 通信**
- **Output 投影用 row-parallel**：权重按 input 维切，各 rank 算部分和，**1 次 all-reduce 聚合**
- **整体通信**：一个 Attention Block 仅 1 次 all-reduce（在 Output 投影处），QKV 投影零通信
- **TP 经典配对**：Column + Row 配对是 TP 的标准 pattern，最小化通信次数

</details>


2. **all-reduce 的通信量是多少？ring 拓扑怎么工作？**（⭐⭐⭐⭐ 高频）

<details>
<summary>点击查看答案</summary>

- **通信量**：ring all-reduce 总通信量 = $2V(N-1)/N$（$V$ = 张量大小，$N$ = GPU 数）
  - 每卡发送 + 接收 = $2V(N-1)/N$，与 $N$ 近似无关（$N$ 大时趋近 $2V$）
- **ring 两阶段**：
  1. **reduce-scatter**（$N-1$ 步）：每步每卡 send 一块给右邻 + recv 左邻的块并累加，$N-1$ 步后每卡持有一个块的完整和
  2. **all-gather**（$N-1$ 步）：每步每卡 send 自己那块完整和给右邻，$N-1$ 步后每卡都有完整结果
- **为什么用 ring**：每卡只与左右邻居通信，带宽利用率高、无拥塞点；步数随 $N$ 线性增长但单步数据量小
- **对比**：朴素 all-reduce（全部发到一个卡再广播）通信量 $O(VN)$ 且中心卡拥塞，ring 是 $O(V)$

</details>


3. **1F1B 的 bubble ratio 怎么算？如何降低？**（⭐⭐⭐⭐ 高频）

<details>
<summary>点击查看答案</summary>

- **bubble ratio**：$\text{bubble} = (P-1)/(M+P-1)$
  - $P$ = pipeline stage 数（GPU 数）
  - $M$ = micro-batch 数
- **推导**：总时间 = $(M+P-1) \times t_{mb}$（从第一个 micro-batch 进入第一个 stage 到最后一个 micro-batch 离开最后一个 stage）。每个 stage 处理 $M$ 个 micro-batch，忙碌 $M \times t_{mb}$，空闲 $(P-1) \times t_{mb}$（启动填满 $P$ 个 stage 期间该 stage 在等待）。因此 bubble 占比 $= (P-1)/(M+P-1)$
- **降低方法**：
  1. **增大 $M$**（micro-batch 数）：$M \gg P$ 时 bubble → 0（主要手段）
  2. **减小 $P$**：但 $P$ 小则显存省得少，需权衡
  3. **interleaved schedule**（1F1B interleaved）：把每个 stage 再细分，进一步压空泡
- **推理场景**：prefill 长 prompt 可切 chunk 走流水线（类比 Chunked Prefill），相当于增大 $M$

</details>


4. **NCCL 的 ring 拓扑相比 tree 拓扑有什么优劣？**（⭐⭐⭐ 中频）

<details>
<summary>点击查看答案</summary>

- **ring 拓扑**：
  - 优势：带宽利用率高（每卡只与左右邻居通信，链路不争抢）、通信量与 $N$ 无关（$\approx 2V$）
  - 劣势：延迟随 $N$ 线性增长（$2(N-1)$ 步），小数据时步数开销显著
  - 适用：大数据 all-reduce（TP 输出聚合、DP 梯度同步）
- **tree 拓扑**：
  - 优势：延迟低（$\log N$ 步）
  - 劣势：根节点带宽争抢、通信量随 $N$ 增长
  - 适用：小数据 broadcast / reduce（参数同步、控制信息）
- **NCCL 自适应**：运行时根据 GPU 拓扑（NVLink、PCIe、InfiniBand）和数据大小自动选 ring/tree 或混合
- **实际**：TP/DP 大张量 all-reduce 几乎都用 ring；初始化小 broadcast 用 tree

</details>


5. **通信-计算重叠怎么实现？有什么前提和限制？**（⭐⭐⭐⭐⭐ 必考）

<details>
<summary>点击查看答案</summary>

- **实现**：双 CUDA Stream
  - `compute_stream` 跑 GEMM/Attention，`comm_stream` 跑 all-reduce
  - 无依赖时两流并发 launch，GPU 硬件调度并发执行
  - 有依赖时用 `stream.wait_stream()` 建立顺序
- **重叠前提**：
  1. **数据无依赖**：通信的数据与计算的数据不重叠（如上一层 all-reduce 与本层 GEMM 可重叠）
  2. **GPU 资源足够**：两路 kernel 需有空闲 SM 并发，否则单卡上仍串行
  3. **多卡场景更有效**：通信走 NVLink/IB（独立硬件），与计算 SM 完全解耦，重叠效果远好于单卡双流
- **收益**：total 从 $T_c + T_a$ 降到 $\max(T_c, T_a)$，典型加速 1.5-1.8x
- **CUDA Graph 进阶**：捕获双流 launch 序列，消除 Python/CPU launch 开销，decode 阶段 shape 固定时收益最大
- **限制**：prefill shape 动态变化，CUDA Graph 难以捕获，需多 graph 或回退 eager

</details>
