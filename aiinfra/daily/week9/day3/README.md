## Day 3：NCCL Collectives —— all-reduce/all-gather/reduce-scatter 通信量

### 🎯 目标

通过今天的学习，你将：

1. 掌握 NCCL 三大 collectives 的语义与通信量公式——all-reduce、all-gather、reduce-scatter<br>
2. 理解 **ring 拓扑**的通信步数推导——all-reduce 为什么是 `2(N-1)` 步<br>
3. 能对比 **ring vs tree** 拓扑的通信模式与适用场景<br>
4. 理解 TP 的 all-reduce、EP 的 all-to-all、ZeRO 的 reduce-scatter 各自的通信量<br>
5. 能为给定并行策略计算通信量，判断通信是否瓶颈<br>

> 💡 **为什么重要**：NCCL 是分布式训练/推理的通信底座。面试常问"all-reduce 通信量怎么算""ring 拓扑几步完成"。Day 1 给了 NCCL 概览，今天深化三大 collectives 的通信量推导。

---

### 学前导读：从"调 NCCL API"到"理解通信量"

Day 1 用 `torch.distributed.all_reduce` 做了 TP 的通信。但调用 API 不等于理解通信——面试官会追问"all-reduce 在 8 卡上要几步通信？每步传多少数据？"。

| Collective | 语义 | 通信量（每节点） | 步数（ring） |
|-----------|------|----------------|------------|
| all-reduce | 所有节点得到相同的 sum | 2×(N-1)/N × 数据量 | 2(N-1) |
| all-gather | 所有节点得到完整数据 | (N-1)/N × 数据量 | N-1 |
| reduce-scatter | 所有节点得到不同的分片 | (N-1)/N × 数据量 | N-1 |

> 💡 **一句话总结**：all-reduce = reduce-scatter + all-gather，所以是 2(N-1) 步。

---

### 理论学习

#### 3.1 Ring All-Reduce

##### 通信过程

N 个节点排成环，数据分成 N 块。All-reduce 分两个阶段：

**阶段 1：Reduce-Scatter（N-1 步）**
```
每步: 每节点把自己的第 i 块发给右邻, 接收左邻的第 i-1 块并累加
N-1 步后: 每节点拥有一个完整的 reduced 块（不同节点拥有不同块）
```

**阶段 2：All-Gather（N-1 步）**
```
每步: 每节点把自己的完整块发给右邻
N-1 步后: 每节点拥有所有 N 个完整块 = all-reduced 完整数据
```

##### 通信量

- 每节点每步发 1/N 数据量
- 总步数 2(N-1)
- 每节点总通信量 = 2(N-1) × (1/N) × 数据量 = **2(N-1)/N × 数据量**

##### 示例（N=4, 数据 4GB）

```
Reduce-Scatter (3 步):
  每步每节点发 1GB → 总发 3GB
All-Gather (3 步):
  每步每节点发 1GB → 总发 3GB
每节点总通信量: 6GB = 2×(4-1)/4 × 4GB = 6GB ✓
```

#### 3.2 All-Gather

##### 语义

每个节点有数据的一个分片，all-gather 后所有节点拥有完整数据。

##### Ring 实现

```
N-1 步: 每节点把自己的分片发给右邻, 接收左邻的分片
N-1 步后: 每节点拥有所有 N 个分片 = 完整数据
```

##### 通信量

- 每节点总通信量 = (N-1) × (1/N) × 数据量 = **(N-1)/N × 数据量**

##### 用途

- TP 的 column-parallel：各卡有自己 head 的结果，all-gather 拼成完整输出
- ZeRO-3 的参数收集

#### 3.3 Reduce-Scatter

##### 语义

每个节点有完整数据，reduce-scatter 后每个节点拥有 reduced 结果的一个分片。

##### Ring 实现

```
N-1 步: 每节点把一个块发给右邻, 接收左邻的块并累加
N-1 步后: 每节点拥有一个 reduced 分片
```

##### 通信量

- 每节点总通信量 = (N-1) × (1/N) × 数据量 = **(N-1)/N × 数据量**

##### 用途

- ZeRO 的梯度分片
- all-reduce 的前半段

#### 3.4 三者关系

```
all-reduce = reduce-scatter + all-gather
通信量: all-reduce (2(N-1)/N) = reduce-scatter ((N-1)/N) + all-gather ((N-1)/N)
```

| Collective | 通信量 | 步数 | 结果 |
|-----------|--------|------|------|
| reduce-scatter | (N-1)/N × D | N-1 | 每节点一个分片 |
| all-gather | (N-1)/N × D | N-1 | 每节点完整数据 |
| all-reduce | 2(N-1)/N × D | 2(N-1) | 每节点完整 reduced 数据 |

#### 3.5 Ring vs Tree 拓扑

##### Ring 拓扑

```
节点 0 → 节点 1 → 节点 2 → 节点 3 → 节点 0
```

- 优点：带宽利用率高（每步每节点都在发/收）
- 缺点：步数多（2(N-1)），大 N 时延迟累积
- 适用：N ≤ 8（单机）、大消息

##### Tree 拓扑

```
      节点 0
     /      \
  节点 1   节点 2
   /  \     /  \
  3    4   5    6
```

- 优点：步数少（`log N`）
- 缺点：根节点带宽瓶颈
- 适用：小消息、大 N

##### NCCL 的混合策略

NCCL 默认用 **ring**（大消息）+ **tree**（小消息）混合：
- 大消息：ring 带宽利用率高
- 小消息：tree 延迟低
- 阈值：通常 ~1KB-4KB

#### 3.6 各并行的通信量

##### Tensor Parallelism (TP)

每层 forward 结束做 all-reduce：
```
通信量/层 = 2(N-1)/N × activation_size
activation_size = batch × seq_len × hidden_dim × sizeof(dtype)
```

N=8, batch=1, seq=2048, hidden=4096, FP16:
```
= 2×7/8 × 1×2048×4096×2 = 28.7 MB/层
```

##### Pipeline Parallelism (PP)

每 stage 边界做 send/recv：
```
通信量/stage = activation_size
```
（比 TP 少一个 all-reduce，但有 bubble）

##### Expert Parallelism (EP)

每层 MoE 做 all-to-all dispatch + all-to-all combine：
```
通信量/层 = 2 × tokens × top_k × expert_dim × sizeof(dtype)
```

##### Data Parallelism (DP, 训练)

每步 all-reduce 梯度：
```
通信量/步 = 2(N-1)/N × model_size
```

---

### Coding 任务

#### 任务 1：通信量计算器

```python
def comm_cost(collective, data_size, N):
    if collective == "all_reduce":
        return 2 * (N - 1) / N * data_size, 2 * (N - 1)
    elif collective == "all_gather":
        return (N - 1) / N * data_size, N - 1
    elif collective == "reduce_scatter":
        return (N - 1) / N * data_size, N - 1

# TP all-reduce 通信量
for N in [2, 4, 8]:
    cost, steps = comm_cost("all_reduce", 1*2048*4096*2, N)  # batch×seq×hidden×FP16
    print(f"TP all-reduce N={N}: {cost/1e6:.2f} MB, {steps} steps")
```

#### 任务 2：NCCL Ring All-Reduce 模拟

```python
import numpy as np

def ring_all_reduce(data_chunks, N):
    """模拟 ring all-reduce, 返回每步通信"""
    # data_chunks: list of N arrays
    steps = []
    # Reduce-scatter phase
    for step in range(N - 1):
        # 每节点发 chunk[i], 收 chunk[(i-1) % N] 并累加
        ...
    # All-gather phase
    for step in range(N - 1):
        ...
    return steps

N = 4
chunks = [np.zeros(4) for _ in range(N)]
for i in range(N):
    chunks[i][i] = 1.0  # 每节点一个位置有 1
steps = ring_all_reduce(chunks, N)
print(f"Total steps: {len(steps)} (expect {2*(N-1)})")
```

#### 任务 3：LeetCode 面试题

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [374](https://leetcode.cn/problems/guess-number-higher-or-lower/) | Easy | 二分 | 暂无 |
| [278](https://leetcode.cn/problems/first-bad-version/) | Easy | 二分 | 暂无 |
| [153](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/) | Medium | 二分 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/153_寻找旋转排序数组中的最小值.html) |

---

### 今日总结

1. **all-reduce = reduce-scatter + all-gather**，通信量 2(N-1)/N × D，步数 2(N-1)
2. **all-gather / reduce-scatter** 各 (N-1)/N × D，步数 N-1
3. **Ring 拓扑**：大消息带宽利用率高，步数多；**Tree 拓扑**：小消息延迟低，步数少
4. **TP**：每层 all-reduce activation；**PP**：每 stage send/recv；**EP**：每层 all-to-all；**DP**：每步 all-reduce 梯度

---

### 面试要点

1. **Ring all-reduce 的通信量是多少？怎么推导？**

   <details>
   <summary>答案</summary>

   - 通信量 = 2(N-1)/N × 数据量
   - 推导：reduce-scatter (N-1 步, 每步发 1/N 数据) + all-gather (N-1 步, 同)
   - 每节点总发/收 = 2 × (N-1) × (1/N) × D = 2(N-1)/N × D
   - 步数 = 2(N-1)

   </details>

2. **all-reduce、all-gather、reduce-scatter 的关系是什么？**

   <details>
   <summary>答案</summary>

   - all-reduce = reduce-scatter + all-gather
   - reduce-scatter：完整数据 → 每节点一个 reduced 分片
   - all-gather：分片 → 每节点完整数据
   - 通信量：all-reduce (2(N-1)/N) = reduce-scatter ((N-1)/N) + all-gather ((N-1)/N)

   </details>

3. **Ring 和 tree 拓扑各有什么优缺点？NCCL 怎么选？**

   <details>
   <summary>答案</summary>

   - Ring：带宽利用率高（每步每节点都在发收），步数多 2(N-1)，适合大消息 + 小 N
   - Tree：步数少 log N，根节点带宽瓶颈，适合小消息 + 大 N
   - NCCL 混合：大消息用 ring，小消息用 tree，阈值 ~1-4KB

   </details>
