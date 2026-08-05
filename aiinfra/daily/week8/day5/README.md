## Day 5：项目推进 —— 量化/投机解码/CUDA Graph 接入 Mini 引擎

### 🎯 目标

通过今天的学习，你将：

1. 能把 Day 1-4 的加速技术（量化/投机解码/CUDA Graph）选一个接入 Mini 引擎<br>
2. 理解三种加速技术的**集成复杂度与收益对比**，能做选型决策<br>
3. 能实现 **CUDA Graph 集成**（最简单、收益最大）——捕获 Mini 引擎的 decode forward 序列<br>
4. 能验证集成前后的**性能提升**——decode latency 降低、throughput 提升<br>

> 💡 **为什么重要**：Day 1-4 分别学了量化/投机解码/CUDA Graph，但没接入引擎。今天是"从学到用"的关键一步——选一个加速技术真正集成到 Mini 引擎，用实测数据验证收益。

---

### 学前导读：三种加速技术的集成选型

| 技术 | 集成复杂度 | 预期收益 | 推荐优先级 |
|------|-----------|---------|----------|
| **CUDA Graph** | 低（~50 行） | decode latency -30-50% | ⭐⭐⭐ 首选 |
| INT8 KV Cache 量化 | 中（~200 行） | KV 显存 -50%, decode bandwidth -30% | ⭐⭐ |
| 投机解码 | 高（~500 行） | throughput +1.5-2x | ⭐ 可选 |

> 💡 **一句话总结**：CUDA Graph 是"低投入高回报"的首选——50 行代码就能消除 decode 路径 50% 的 launch overhead。今天重点做 CUDA Graph 集成。

---

### 理论学习

#### 5.1 CUDA Graph 集成到 Mini 引擎

##### 为什么 Decode 路径收益最大？

Decode 阶段 M=1，每个 kernel 只几 μs，但 launch overhead 5-10μs/kernel：
```
Decode forward: 30 个 kernel × 7μs launch = 210μs 纯 launch
               30 个 kernel × 3μs 执行 = 90μs 实际计算
总 latency: 300μs, launch 占 70%!
```

CUDA Graph 把 30 次 launch 压成 1 次：
```
Graph replay: 10μs launch + 90μs 计算 = 100μs
收益: 300μs → 100μs, -67%
```

##### 集成步骤

```python
class MiniEngineWithGraph:
    def __init__(self, model, max_seq_len):
        self.model = model
        self.graph = None
        self.static_input = None
        self.static_output = None

    def capture_decode_graph(self, seq_len):
        """捕获 decode forward 的 CUDA Graph"""
        # 1. 静态 buffer（地址必须固定）
        self.static_input = torch.zeros(1, seq_len, dtype=torch.long, device='cuda')
        self.static_position = torch.zeros(1, seq_len, dtype=torch.long, device='cuda')

        # 2. warmup（确保 cudnn/cublas 初始化）
        for _ in range(3):
            _ = self.model(self.static_input, self.static_position)

        # 3. 捕获
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input, self.static_position)

    def decode_step(self, input_ids, position_ids):
        """用 graph replay 做 decode"""
        # 1. 复制输入到静态 buffer
        self.static_input.copy_(input_ids)
        self.static_position.copy_(position_ids)
        # 2. replay
        self.graph.replay()
        # 3. 从静态 buffer 读输出
        return self.static_output.clone()
```

##### 静态 Buffer 的必要性

CUDA Graph 要求输入输出地址固定（捕获时录下的地址）。所以必须用**静态 buffer**：
- 捕获前分配固定地址的 buffer
- 每次调用时 `copy_` 输入到 buffer
- replay 后从 buffer 读输出

#### 5.2 Shape Bucketing

##### 动态 Shape 的问题

Decode 的 seq_len 每步 +1（KV Cache 增长），shape 变化导致 Graph 无法复用。

##### Bucketing 策略

按 seq_len 分桶，每桶预捕获一个 Graph：

```python
class BucketedGraphEngine:
    def __init__(self, model, bucket_sizes=None):
        if bucket_sizes is None:
            bucket_sizes = [128, 256, 512, 1024, 2048, 4096]
        self.graphs = {}
        for sz in bucket_sizes:
            self.capture_graph(sz)  # 每个桶捕获一个

    def decode_step(self, input_ids, seq_len):
        # 找最近的 bucket
        bucket = min(self.graphs.keys(), key=lambda s: abs(s - seq_len) if s >= seq_len else float('inf'))
        # pad 到 bucket 大小
        padded = F.pad(input_ids, (0, bucket - seq_len))
        # replay 对应 bucket 的 graph
        return self.graphs[bucket].replay(padded)
```

##### Bucket 数量权衡

- 太少：padding 浪费多
- 太多：捕获 + 显存开销大
- 实践：6-8 个 bucket 覆盖常见 seq_len

#### 5.3 INT8 KV Cache 量化集成（备选）

##### 集成步骤

1. 前向时把 KV Cache 量化为 INT8（per-token scale）
2. Attention kernel 内在线 dequant
3. 显存节省 50%，decode 带宽节省 30%

```python
class MiniEngineWithKVQuant:
    def store_kv(self, k, v):
        # 量化: per-token scale
        self.k_scale = k.abs().max(dim=-1, keepdim=True).values / 127
        self.k_int8 = (k / self.k_scale).to(torch.int8)
        # v 同理

    def attention(self, q):
        # 在线 dequant
        k_fp16 = self.k_int8.to(torch.float16) * self.k_scale
        # attention 计算
        ...
```

#### 5.4 投机解码集成（备选）

##### 集成步骤

1. 加载 draft model（小模型）
2. draft 生成 k 个候选 token
3. target model 一次 forward 验证
4. 接受匹配的 token

```python
class MiniEngineWithSpecDecode:
    def generate(self, prompt, max_tokens):
        while len(generated) < max_tokens:
            # 1. draft 生成 k 个候选
            draft_tokens = self.draft_model.generate(prompt, k=4)
            # 2. target 验证
            target_logits = self.target_model.forward(prompt + draft_tokens)
            target_tokens = target_logits.argmax(-1)
            # 3. 接受匹配的
            for i, (d, t) in enumerate(zip(draft_tokens, target_tokens)):
                if d == t:
                    generated.append(d)
                else:
                    generated.append(t)  # 用 target 的
                    break
```

---

### Coding 任务

#### 任务 1：CUDA Graph 集成

```python
import torch

class MiniEngineGraph:
    def __init__(self, model):
        self.model = model
        self.graphs = {}  # bucket_size -> CUDAGraph

    def capture(self, bucket_sizes=[128, 256, 512, 1024, 2048]):
        for sz in bucket_sizes:
            static_in = torch.zeros(1, sz, dtype=torch.long, device='cuda')
            static_pos = torch.arange(sz, device='cuda').unsqueeze(0)
            # warmup
            for _ in range(3):
                _ = self.model(static_in, static_pos)
            # capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = self.model(static_in, static_pos)
            self.graphs[sz] = (g, static_in, static_pos, static_out)

    def decode(self, input_ids, position_ids):
        seq_len = input_ids.shape[1]
        bucket = min(s for s in self.graphs if s >= seq_len)
        g, sin, spos, sout = self.graphs[bucket]
        sin[:, :seq_len].copy_(input_ids)
        spos[:, :seq_len].copy_(position_ids)
        g.replay()
        return sout[:, :seq_len].clone()

# benchmark: eager vs graph
engine = MiniEngineGraph(model)
engine.capture()
# ... 对比 eager decode vs graph decode 的 latency ...
```

#### 任务 2：验证收益

```python
def bench_eager(model, input_ids, iters=100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = model(input_ids)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def bench_graph(engine, input_ids, iters=100):
    # ... graph replay 版 ...
    pass

print(f"Eager: {bench_eager(model, input):.2f} ms")
print(f"Graph: {bench_graph(engine, input):.2f} ms")
```

#### 任务 3：LeetCode 面试题

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [215](https://leetcode.cn/problems/kth-largest-element-in-an-array/) | Medium | 快速选择/堆 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/215_kth-largest-element-in-an-array.html) |
| [347](https://leetcode.cn/problems/top-k-frequent-elements/) | Medium | 堆/桶排 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/347_top-k-frequent-elements.html) |
| [295](https://leetcode.cn/problems/find-median-from-data-stream/) | Hard | 双堆 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/295_find-median-from-data-stream.html) |

---

### 今日总结

1. **CUDA Graph 集成**：50 行代码，decode latency -30-50%，首选加速技术
2. **静态 Buffer**：Graph 要求地址固定，用 `copy_` + `clone` 中转
3. **Shape Bucketing**：按 seq_len 分桶预捕获，6-8 个 bucket 覆盖常见场景
4. **INT8 KV 量化**（备选）：显存 -50%，带宽 -30%，集成复杂度中
5. **投机解码**（备选）：throughput +1.5-2x，集成复杂度高

---

### 面试要点

1. **CUDA Graph 怎么集成到推理引擎？**

   <details>
   <summary>答案</summary>

   - 静态 buffer 分配（地址固定）
   - warmup 3 次（初始化 cudnn/cublas）
   - `torch.cuda.graph(g)` 捕获 forward 序列
   - 调用时 `copy_` 输入 → `replay()` → `clone()` 输出
   - 收益：30 个 kernel launch 压成 1 次，decode latency -30-50%

   </details>

2. **动态 shape 怎么用 CUDA Graph？**

   <details>
   <summary>答案</summary>

   - Shape bucketing：按 seq_len 分桶（如 128/256/512/...），每桶预捕获一个 Graph
   - 调用时找最近 bucket，pad 到 bucket 大小，replay 对应 Graph
   - 权衡：bucket 少 → padding 浪费多；bucket 多 → 捕获 + 显存开销大
   - 生产实践：6-8 个 bucket 覆盖常见 seq_len

   </details>
