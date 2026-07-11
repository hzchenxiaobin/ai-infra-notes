# 第6周深度展开：Batching & 调度（7天）

> **适用对象**：陈斌斌（已完成第5周学习，掌握 Prefill/Decode、KV Cache、vLLM 架构、Mini 推理引擎 v0）
> **本周目标**：建立系统感，理解 Dynamic Batching 和 Continuous Batching 的原理与实现，阅读 vLLM/TensorRT-LLM 调度器源码，将 Mini 引擎升级到 v1 支持多请求并发
> **时间投入**：工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h
> **周日里程碑**：Mini 推理引擎 v1 支持 Continuous Batching + 优先级调度 + 多请求并发，产出 throughput-latency 曲线

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 理解从单请求到多请求的推理系统调度，掌握 Dynamic Batching 和 Continuous Batching 的实现，构建支持多请求并发的 Mini 推理引擎 v1 |
| **核心产出** | ① Dynamic Batching 实现 ② Continuous Batching 实现 ③ vLLM Scheduler 源码分析 ④ TensorRT-LLM Scheduler 对比 ⑤ Mini 引擎 v1（多请求 + Scheduler）⑥ Latency/Throughput benchmark ⑦ 调度策略对比表 |
| **验收标准** | ① 能实现 Dynamic Batching，多个请求正确聚合 ② 能实现 Continuous Batching，新请求可任意 iteration 加入 ③ 能解释 vLLM Scheduler 的核心逻辑 ④ Mini 引擎 v1 能同时处理多个请求 ⑤ 能绘制 throughput-latency 曲线并识别饱和点 |
| **面试准备** | 积累12-15道调度专题面试题，覆盖 Dynamic/Continuous Batching、Scheduler、Preemption、Priority、Throughput-Latency Trade-off 五大主题 |

### 本周知识图谱

```
Day 36: Dynamic Batching → 请求聚合、padding、timeout、max batch size
 ↓
Day 37: Continuous Batching → iteration-level 调度、请求动态加入/退出
 ↓
Day 38: vLLM Scheduler 源码分析 → schedule() / SchedulingBudget / preemption
 ↓
Day 39: TensorRT-LLM / LightLLM 调度对比 → Inflight Batching / 不同实现思路
 ↓
Day 40: Mini 推理引擎 v1 → Continuous Batching + Scheduler + 多请求并发
 ↓
Day 41: Latency / Throughput 测试 → 不同 batch size / 请求分布 / 饱和点
 ↓
Day 42: 调度优化策略总结 → 策略对比表 + 面试复盘 + GitHub 整理
```

### 前置准备清单

#### 硬件/软件验证
- [ ] 已完成第5周所有 Coding 任务（Mini 推理引擎 v0、KV Cache）
- [ ] PyTorch 可用且 `torch.__version__ >= 2.0`
- [ ] vLLM 源码已下载（用于阅读 Scheduler）
- [ ] `nsys --version` 正常
- [ ] 理解 Week 5 Day 31 的 vLLM 架构笔记

#### 验证命令
```bash
# 验证 PyTorch + CUDA
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
# 预期输出：torch 2.x.x cuda 12.x available True

# 验证 vLLM 源码路径
ls vllm/engine/scheduler.py 2>/dev/null || echo "vLLM source not found"

# 验证 nsys
nsys --version
```

---

## Day 36（周一）：Dynamic Batching

> **今日目标**：理解 Dynamic Batching 的原理，实现请求队列 + 超时等待 + 最大 batch size 限制，验证多请求聚合的正确性和吞吐提升。
> **面试考察度**：⭐⭐⭐⭐ 高频，Dynamic Batching 是推理服务的基础能力

---

### 学习任务1：Dynamic Batching 原理（45分钟）

#### 阅读内容
- **论文/博客**：
 - "Batching: Dynamic vs Static in ML Serving"
 - vLLM 博客中关于 batching 的部分
- **重点**：
 - 为什么需要 batching
 - Dynamic Batching 的工作流程
 - Padding 的代价与优化
 - Timeout 与 max batch size 的 trade-off

#### 核心概念笔记

**1. 为什么需要 Batching？**

```
单个请求的 GEMM 通常是 memory-bound（尤其 decode 阶段 M=1）
合并多个请求后，M 增大，GEMM 更接近 compute-bound

效果：
 - Throughput 显著提升
 - 但单个请求的 latency 可能增加（需要等待其他请求）
```

**2. Dynamic Batching 工作流程**

```
请求队列: [R1, R2, R3, R4, R5, ...]

调度策略:
 1. 当队列中有请求时，启动 timer
 2. 等待一段时间（timeout）或凑够 max_batch_size
 3. 将当前队列中的请求聚合成一个 batch
 4. 对 batch 做 forward
 5. 返回每个请求的结果

参数：
 - max_batch_size: 最大 batch 大小
 - max_waiting_time: 最大等待时间
 - padding_strategy: 如何填充不同长度的请求
```

**3. Padding 的代价与优化**

```
问题：一个 batch 中不同请求长度不同
 R1: [a, b, c] len=3
 R2: [d, e, f, g, h] len=5
 R3: [i, j] len=2

Naive padding:
 pad 到 max_len=5:
 [a, b, c, 0, 0]
 [d, e, f, g, h]
 [i, j, 0, 0, 0]
 浪费：R1 和 R3 有很多 pad token 计算

Padding-free 优化：
 - 将不同长度的请求尽量分到不同 batch
 - 使用 attention mask 避免 pad token 参与计算
 - Continuous Batching 进一步减少 padding
```

**4. Timeout 与 Batch Size 的 Trade-off**

```
max_batch_size 大 + timeout 长：
 - 吞吐高（batch 更满）
 - 延迟高（请求等待时间长）

max_batch_size 小 + timeout 短：
 - 延迟低
 - 吞吐低（batch 不满，GPU 利用率低）

实际系统需要根据 SLA 调参。
```

### 学习任务2：Batching 的收益量化（30分钟）

#### 计算示例

```
单个 decode 请求：
 QKV GEMM: M=1, N=d, K=3d
 FLOPs ≈ 2 × 1 × d × 3d = 6d²
 但 M=1 无法充分利用 Tensor Core，实际 throughput 很低

Batch=4 的 decode 请求：
 QKV GEMM: M=4, N=d, K=3d
 FLOPs ≈ 2 × 4 × d × 3d = 24d²
 M=4 比 M=1 更容易利用 GPU 并行

理论吞吐提升：
 - 当 batch 从 1 增加到 B，throughput 通常近似线性增长（直到 compute-bound）
 - 但实际 latency 也会从 T 增加到 ~B×T 或更少
```

#### Throughput vs Latency 曲线

```
Latency
 │
 │ ╱
 │ ╱
 │ ╱
 │ ╱
 │ ╱
 │╱
 └──────────────► Throughput

曲线特征：
 - 低吞吐时延迟低
 - 随着 batch 增大，延迟增加
 - 达到饱和点后，延迟急剧上升
```

---

### 晚间编程任务：Dynamic Batcher（1小时）

#### 完整代码

```python
# dynamic_batcher.py —— Dynamic Batching 实现
# 运行命令: python dynamic_batcher.py

import time
import threading
from collections import deque
from typing import List, Callable, Optional
import torch

class Request:
 def __init__(self, request_id, data):
 self.request_id = request_id
 self.data = data # 例如 prompt tokens
 self.arrival_time = time.time()
 self.result = None
 self.done_event = threading.Event()

class DynamicBatcher:
 """Dynamic Batcher：请求队列 + 超时等待 + 最大 batch size"""
 def __init__(self, max_batch_size=4, max_wait_time=0.05):
 self.max_batch_size = max_batch_size
 self.max_wait_time = max_wait_time
 self.queue = deque()
 self.lock = threading.Lock()
 self.stop_event = threading.Event()
 self.worker_thread = threading.Thread(target=self._worker_loop)
 self.worker_thread.start()

 def submit(self, request: Request):
 """提交请求到队列"""
 with self.lock:
 self.queue.append(request)

 def _collect_batch(self) -> List[Request]:
 """收集一个 batch"""
 batch = []
 deadline = time.time() + self.max_wait_time

 while len(batch) < self.max_batch_size:
 remaining = deadline - time.time()
 if remaining <= 0 and len(batch) > 0:
 break

 with self.lock:
 if self.queue:
 batch.append(self.queue.popleft())
 elif len(batch) > 0:
 # 没有更多请求，但已有 batch，直接处理
 break

 if not batch:
 # 完全没有请求，短暂睡眠
 time.sleep(0.001)
 deadline = time.time() + self.max_wait_time

 return batch

 def _process_batch(self, batch: List[Request]):
 """处理一个 batch（这里用 sleep 模拟模型 forward）"""
 batch_size = len(batch)
 # 模拟：batch 越大，forward 时间越长，但 per-request 时间减少
 forward_time = 0.01 + 0.005 * batch_size
 time.sleep(forward_time)

 # 返回结果
 for i, req in enumerate(batch):
 req.result = f"result_for_{req.request_id}_batch_size_{batch_size}"
 req.done_event.set()

 def _worker_loop(self):
 while not self.stop_event.is_set():
 batch = self._collect_batch()
 if batch:
 self._process_batch(batch)

 def shutdown(self):
 self.stop_event.set()
 self.worker_thread.join()

def main():
 batcher = DynamicBatcher(max_batch_size=4, max_wait_time=0.05)

 # 模拟发送 10 个请求
 requests = []
 print("Submitting 10 requests...")
 for i in range(10):
 req = Request(request_id=i, data=f"prompt_{i}")
 batcher.submit(req)
 requests.append(req)
 time.sleep(0.02) # 模拟请求到达间隔

 # 等待所有请求完成
 start = time.time()
 for req in requests:
 req.done_event.wait()
 total_time = time.time() - start

 print(f"\nAll requests done in {total_time:.3f}s")
 print(f"Average latency: {total_time / len(requests):.3f}s")

 # 打印每个请求的结果
 for req in requests:
 print(f" Request {req.request_id}: {req.result}")

 batcher.shutdown()

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python dynamic_batcher.py

# 预期输出
# Submitting 10 requests...
# 
# All requests done in x.xxxs
# Average latency: x.xxxs
# Request 0: result_for_0_batch_size_x
# ...
```

#### 练习题

**练习1（基础）**：修改 `_process_batch` 使用真实的 PyTorch 模型 forward。

**练习2（进阶）**：实现 padding 策略，处理不同长度的输入序列。

**练习3（综合）**：测试不同 `max_batch_size` 和 `max_wait_time` 下的吞吐和延迟。

---

### 今日面试题

**面试题1**：什么是 Dynamic Batching？它的优缺点是什么？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Dynamic Batching**：将到达的请求暂时放入队列，等待一定时间或凑够一定数量后，聚合成一个 batch 一起执行
- **优点**：
 - 提高 GPU 利用率（尤其 decode 阶段 M 增大）
 - 提高 throughput
- **缺点**：
 - 引入等待延迟（request-level latency 增加）
 - 需要 padding，造成计算浪费
 - 一个长请求会阻塞整个 batch
- **适用场景**：吞吐优先、请求到达率高的服务

**面试题2**：Dynamic Batching 中的 padding 有什么问题？如何优化？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Padding 问题**：
 - 不同长度请求需要 pad 到同一长度
 - Pad token 也要参与 forward，浪费计算
 - 序列长度差异越大，浪费越严重
- **优化方法**：
 1. **长度分组**：将长度相近的请求分到同一个 batch
 2. **Attention mask**：让 pad token 不参与 attention
 3. **Padding-free / Pack sequence**：直接拼接序列，用 position ids 区分
 4. **Continuous Batching**：减少 padding，新请求可在任意 iteration 加入

---

### 今日自测清单

- [ ] 能解释 Dynamic Batching 的工作流程
- [ ] 能理解 batch size 和 timeout 的 trade-off
- [ ] 能说出 padding 的 3 种优化方法
- [ ] `dynamic_batcher.py` 运行成功，多请求被聚合处理
- [ ] 能计算 batching 对 throughput 的理论提升

---

## Day 37（周二）：Continuous Batching

> **今日目标**：理解 Continuous Batching（Inflight Batching）的原理，实现 iteration-level 调度，支持请求在 decode 阶段动态加入和退出。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，Continuous Batching 是现代 LLM 推理服务的核心

---

### 学习任务1：Continuous Batching 原理（45分钟）

#### 阅读内容
- **论文/博客**：
 - "Orca: A Distributed Serving System for Transformer-Based Generative Models" (Yu et al., OSDI 2022) — Continuous Batching 的奠基论文
 - vLLM 文档中关于 continuous batching 的部分
- **重点**：
 - Iteration-level scheduling vs Request-level scheduling
 - 请求动态加入和退出
 - 如何避免一个长请求阻塞整个 batch

#### 核心概念笔记

**1. Dynamic Batching 的问题**

```
Dynamic Batching 是 request-level 的：
 - 一个 batch 中的所有请求一起开始
 - 所有请求都完成后才释放 batch
 - 问题：一个长请求会阻塞短请求

例子：
 Batch = [R1(要生成10 tokens), R2(要生成100 tokens)]
 R1 在 10 个 iteration 后已经完成，但必须等待 R2 完成
 R1 占用的资源被浪费
```

**2. Continuous Batching 的核心思想**

```
Iteration-level scheduling：
 - 每个 iteration 都重新构建 batch
 - 新请求可以在任意 iteration 加入
 - 完成的请求可以在任意 iteration 退出
 - 不再存在"一个 batch 一起结束"的概念

效果：
 - GPU 始终满负荷运行
 - 短请求不会被长请求阻塞
 - Throughput 和 latency 都更好
```

**3. Continuous Batching 的工作流程**

```
Time 0: Batch = [R1_prefill, R2_prefill]
Time 1: R1 完成 prefill，进入 decode
 Batch = [R1_decode, R2_decode]
Time 2: 新请求 R3 到达
 Batch = [R1_decode, R2_decode, R3_prefill]
Time 3: R1 生成结束
 Batch = [R2_decode, R3_prefill]
Time 4: R2 生成结束，R4 到达
 Batch = [R3_decode, R4_prefill]
...
```

**4. Prefill + Decode 混合调度**

```
Continuous Batching 可以混合 prefill 和 decode：
 - 一个 iteration 中同时处理：
 - 新请求的 prefill
 - 正在生成请求的 decode
 - 这是 vLLM 和许多现代推理系统的标准做法

挑战：
 - Prefill 和 decode 的计算特征不同
 - Prefill 会"打断" decode 的 smooth latency
 - 需要 token budget 控制每轮的计算量
```

### 学习任务2：Scheduler 状态机（30分钟）

#### Sequence 状态转换

```
 ┌─────────────┐
 │ WAITING │
 └──────┬──────┘
 │ scheduler 选择
 ▼
 ┌─────────────┐
 │ RUNNING │
 └──────┬──────┘
 │
 ┌───────┼───────┐
 ▼ ▼ ▼
 FINISHED SWAPPED RUNNING (next iter)
```

#### 每轮调度决策

```
Scheduler 每轮需要决定：
 1. 继续运行哪些 RUNNING 请求
 2. 从 WAITING 中加入哪些新请求
 3. 是否需要抢占某些 RUNNING 请求
 4. 处理 FINISHED 请求（释放资源）

约束：
 - Token budget: 控制 prefill + decode 的总 token 数
 - 显存预算: KV Cache 不能超出限制
```

---

### 晚间编程任务：Continuous Batcher（1.5小时）

#### 完整代码

```python
# continuous_batcher.py —— Continuous Batching 实现
# 运行命令: python continuous_batcher.py

import time
import threading
from collections import deque
from enum import Enum
from typing import List, Dict, Optional

class SequenceStatus(Enum):
 WAITING = "waiting"
 RUNNING = "running"
 FINISHED = "finished"

class Sequence:
 def __init__(self, seq_id, prompt, max_new_tokens=20):
 self.seq_id = seq_id
 self.prompt = prompt
 self.tokens = list(prompt)
 self.max_new_tokens = max_new_tokens
 self.generated_count = 0
 self.status = SequenceStatus.WAITING
 self.result_event = threading.Event()

 def append_token(self, token):
 self.tokens.append(token)
 self.generated_count += 1
 if self.generated_count >= self.max_new_tokens:
 self.status = SequenceStatus.FINISHED
 self.result_event.set()

class ContinuousBatcher:
 """Continuous Batcher：每轮 iteration 重新构建 batch"""
 def __init__(self, max_token_budget=100):
 self.max_token_budget = max_token_budget
 self.waiting_queue = deque()
 self.running_sequences: Dict[int, Sequence] = {}
 self.lock = threading.Lock()
 self.stop_event = threading.Event()
 self.worker_thread = threading.Thread(target=self._worker_loop)
 self.worker_thread.start()

 def submit(self, seq: Sequence):
 """提交新请求"""
 with self.lock:
 self.waiting_queue.append(seq)

 def _schedule(self) -> List[Sequence]:
 """每轮调度：决定哪些 sequence 运行"""
 batch = []
 token_budget = self.max_token_budget

 with self.lock:
 # 1. 继续运行正在 decode 的序列
 finished_ids = []
 for seq_id, seq in self.running_sequences.items():
 if seq.status == SequenceStatus.FINISHED:
 finished_ids.append(seq_id)
 else:
 # decode 一步消耗 1 个 token budget
 if token_budget >= 1:
 batch.append(seq)
 token_budget -= 1

 # 释放完成的序列
 for seq_id in finished_ids:
 del self.running_sequences[seq_id]

 # 2. 从 waiting 中加入新请求做 prefill
 # 简化：每个 prefill 请求的 cost = prompt 长度
 new_waiting = deque()
 for seq in self.waiting_queue:
 prompt_cost = len(seq.prompt)
 if token_budget >= prompt_cost and len(batch) < 8:
 seq.status = SequenceStatus.RUNNING
 self.running_sequences[seq.seq_id] = seq
 batch.append(seq)
 token_budget -= prompt_cost
 else:
 new_waiting.append(seq)
 self.waiting_queue = new_waiting

 return batch

 def _run_iteration(self, batch: List[Sequence]):
 """运行一个 iteration"""
 # 模拟 forward
 time.sleep(0.005 * len(batch))

 for seq in batch:
 if seq.status == SequenceStatus.RUNNING:
 # 生成新 token
 new_token = seq.tokens[-1] + 1 # 简单模拟
 seq.append_token(new_token)

 def _worker_loop(self):
 iteration = 0
 while not self.stop_event.is_set():
 batch = self._schedule()
 if batch:
 self._run_iteration(batch)
 iteration += 1

 # 打印每轮状态
 running = sum(1 for s in batch if s.status == SequenceStatus.RUNNING)
 finished = sum(1 for s in batch if s.status == SequenceStatus.FINISHED)
 print(f"Iter {iteration}: batch_size={len(batch)}, running={running}, finished={finished}")
 else:
 time.sleep(0.001)

 def shutdown(self):
 self.stop_event.set()
 self.worker_thread.join()

def main():
 batcher = ContinuousBatcher(max_token_budget=50)

 # 模拟请求到达
 sequences = []
 print("Submitting sequences...")

 # R1: 短 prompt，生成 5 tokens
 s1 = Sequence(seq_id=1, prompt=[1, 2, 3], max_new_tokens=5)
 batcher.submit(s1)
 sequences.append(s1)

 time.sleep(0.01)

 # R2: 长 prompt，生成 10 tokens
 s2 = Sequence(seq_id=2, prompt=[10, 11, 12, 13, 14], max_new_tokens=10)
 batcher.submit(s2)
 sequences.append(s2)

 time.sleep(0.02)

 # R3: 中等 prompt，生成 3 tokens
 s3 = Sequence(seq_id=3, prompt=[20, 21], max_new_tokens=3)
 batcher.submit(s3)
 sequences.append(s3)

 # 等待所有请求完成
 for s in sequences:
 s.result_event.wait()

 print("\nAll sequences finished")
 for s in sequences:
 print(f" Seq {s.seq_id}: tokens={s.tokens}, generated={s.generated_count}")

 batcher.shutdown()

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python continuous_batcher.py

# 预期输出
# Submitting sequences...
# Iter 1: batch_size=1, running=1, finished=0
# Iter 2: batch_size=2, running=2, finished=0
# Iter 3: batch_size=3, running=3, finished=0
# Iter 4: batch_size=2, running=2, finished=0
# ...
# All sequences finished
# Seq 1: tokens=[1, 2, 3, 4, 5, 6, 7, 8], generated=5
# ...
```

#### 练习题

**练习1（基础）**：在 `_schedule` 中加入优先级逻辑，高优先级请求优先调度。

**练习2（进阶）**：实现 preemption：当显存/token budget 不足时，将某些 RUNNING 请求换出。

**练习3（综合）**：将 Continuous Batcher 与 Week 5 的 MiniEngineV0 结合，实现真正基于 transformer 的 continuous batching。

---

### 今日面试题

**面试题1**：Continuous Batching 和 Dynamic Batching 有什么区别？为什么 Continuous Batching 更适合 LLM 推理？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **Dynamic Batching**：request-level，请求一起开始一起结束，一个长请求会阻塞整个 batch
- **Continuous Batching**：iteration-level，每轮重新构建 batch，请求可动态加入和退出
- **为什么更适合 LLM**：
 - LLM 生成长度差异大，Dynamic Batching 下短请求要等长请求
 - Continuous Batching 让 GPU 始终满负荷运行
 - 吞吐和延迟都更好
 - 可以混合 prefill 和 decode

**面试题2**：Continuous Batching 中，如何混合 Prefill 和 Decode？有什么挑战？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **混合方式**：每轮 scheduler 同时选择：
 - 新请求做 prefill（一次性处理多个 prompt tokens）
 - 正在生成的请求做 decode（每次 1 个 token）
- **挑战**：
 1. **Token budget 分配**：prefill 消耗大量 token budget，可能影响 decode 的 smooth latency
 2. **内存需求**：prefill 需要临时 KV Cache 空间
 3. **Latency 抖动**：prefill 请求加入时，decode 请求的 latency 会突然增加
- **解决方案**：
 - 限制每轮 prefill 的 token 数（chunked prefill）
 - 优先保证 decode 请求的 iteration 节奏
 - 使用 token budget 动态平衡

---

### 今日自测清单

- [ ] 能解释 Continuous Batching 的核心思想
- [ ] 能对比 Continuous 和 Dynamic Batching
- [ ] 能画出 iteration-level 调度的时间线
- [ ] `continuous_batcher.py` 运行成功，请求动态加入/退出
- [ ] 理解 prefill + decode 混合调度的挑战

---

## Day 38（周三）：vLLM Scheduler 源码分析

> **今日目标**：深入阅读 vLLM `Scheduler.schedule()` 的实现，理解 SchedulingBudget、preemption、swapping 的具体逻辑。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，vLLM Scheduler 是推理调度的经典实现

---

### 学习任务1：Scheduler 源码结构（45分钟）

#### 阅读内容
- `vllm/engine/scheduler.py`
- `vllm/core/scheduler.py`（v0.3+ 版本可能重构）
- `vllm/sequence.py`：`SequenceStatus`、`SequenceGroup`

#### 核心概念笔记

**1. `Scheduler` 类关键方法**

```python
class Scheduler:
 def __init__(self, scheduler_config, cache_config, lora_config,
 pipeline_parallel_size, tensor_parallel_size,
 vocab_size):
 self.waiting = deque() # WAITING queue
 self.running = deque() # RUNNING queue
 self.swapped = deque() # SWAPPED queue
 self.block_manager = BlockSpaceManager(...) # 内存管理

 def schedule(self):
 # 返回 (seq_group_metadata_list, scheduler_outputs)
 ...
```

**2. `schedule()` 的 5 个步骤**

```python
def schedule(self):
 # 1. 处理已完成的 running 序列
 # 2. 尝试运行 running queue（continuous batching）
 # 3. 尝试从 waiting queue 加入新请求
 # 4. 如果显存不足，进行 preemption/swapping
 # 5. 构建 scheduler_outputs
```

**3. `SchedulingBudget` 类**

```python
@dataclass
class SchedulingBudget:
 token_budget: int
 max_num_seqs: int
 
 def can_schedule(self, num_new_tokens, num_new_seqs):
 return (self.num_batched_tokens + num_new_tokens <= self.token_budget and
 self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)
```

关键预算：
- `token_budget`：每轮最多处理的 token 数（prefill tokens + decode tokens）
- `max_num_seqs`：每轮最多并发的 sequence 数

**4. Preemption 策略**

```python
class PreemptionMode:
 RECOMPUTE = "recompute"
 SWAP = "swap"
```

```
Recompute（默认）：
 - 丢弃被抢占请求的 KV Cache
 - 之后重新从 prompt 计算
 - 优点：不需要 CPU 内存，通常更快
 - 缺点：重新计算有额外开销

Swap：
 - 将被抢占请求的 KV Cache 换出到 CPU
 - 之后从 CPU 换入
 - 优点：不重新计算
 - 缺点：需要 CPU 内存，PCIe 传输慢
```

---

### 学习任务2：关键源码追踪（45分钟）

#### 阅读任务

1. 找到 `_schedule_running()` 方法，理解如何处理 running queue
2. 找到 `_schedule_swapped()` 方法，理解如何处理 swapped queue
3. 找到 `_schedule_waiting()` 方法，理解如何从 waiting queue 加入请求
4. 找到 `_preempt()` 方法，理解 preemption 的具体实现

#### 核心逻辑

```python
def _schedule_running(self, budget):
 # 遍历 running queue
 # 对每个 sequence group:
 # - 检查是否能分配所需 block
 # - 如果能，加入本轮 batch
 # - 如果不能，进行 preemption
```

```python
def _preempt(self, seq_group, blocks_to_swap_out):
 # 根据 preemption_mode 选择 recompute 或 swap
 if self.scheduler_config.preemption_mode == PreemptionMode.RECOMPUTE:
 self._preempt_by_recompute(seq_group)
 else:
 self._preempt_by_swap(seq_group, blocks_to_swap_out)
```

---

### 晚间任务：源码笔记整理（1小时）

#### 笔记模板

```markdown
# vLLM Scheduler 源码分析

## 核心类
- Scheduler: 主调度器
- SchedulingBudget: 调度预算
- SequenceGroup: 请求组
- BlockSpaceManager: 内存管理

## schedule() 流程
1. 处理完成请求
2. 调度 running queue
3. 调度 waiting queue
4. Preemption / swapping
5. 输出 scheduler_outputs

## SchedulingBudget
- token_budget: 每轮最大 token 数
- max_num_seqs: 每轮最大 sequence 数

## Preemption
- Recompute: 丢弃 KV Cache，之后重算
- Swap: KV Cache 换出到 CPU
- 默认 Recompute，因为通常更快

## Day 39（周四）：TensorRT-LLM / LightLLM 调度对比

> **今日目标**：了解 TensorRT-LLM 的 Inflight Batching 和 LightLLM 的调度实现，对比不同推理框架的调度思路。
> **面试考察度**：⭐⭐⭐⭐ 高频，"不同推理框架的 batching 策略"是加分题

---

### 学习任务1：TensorRT-LLM Inflight Batching（45分钟）

#### 阅读内容
- **文档**：TensorRT-LLM User Guide → "Inflight Batching"
- **源码**：`tensorrt_llm/batch_manager/` 相关代码
- **重点**：
 - Inflight Batching 与 Continuous Batching 的关系
 - TensorRT-LLM 的调度器接口
 - 与 vLLM 的差异

#### 核心概念笔记

**1. Inflight Batching = Continuous Batching**

```
TensorRT-LLM 使用 "Inflight Batching" 术语，本质上就是 Continuous Batching：
 - 请求可以在 engine 运行过程中加入
 - 请求完成后可以立即退出
 - 每轮 iteration 重新构建 batch
```

**2. TensorRT-LLM 调度器特点**

```
特点：
 - 调度器在 C++ 层实现，性能更高
 - 与 TensorRT engine 深度集成
 - 支持多种调度策略（max_tokens_in_batch、max_requests_in_batch 等）
 - 支持 chunked prefill（将大 prefill 拆分）

与 vLLM 的差异：
 - vLLM Scheduler 在 Python 层，更灵活
 - TensorRT-LLM Scheduler 在 C++ 层，性能更好但灵活性较低
```

**3. Chunked Prefill**

```
问题：
 - 长 prompt 的 prefill 会占用大量 token budget
 - 导致 decode 请求的 latency 抖动

Chunked Prefill：
 - 将长 prefill 拆成多个 chunk
 - 每个 chunk 与 decode 请求一起执行
 - 平滑 latency，避免长 prompt 阻塞
```

---

### 学习任务2：LightLLM 调度特点（30分钟）

#### 阅读内容
- **仓库**：https://github.com/ModelTC/lightllm
- **文档**：LightLLM 的 batching 实现

#### 核心概念笔记

**LightLLM 特点**：
```
1. Token Attention：
 - 不预分配固定大小的 KV Cache
 - 用类似内存池的方式动态管理

1. 动态 split fuse：
 - 将 prefill 和 decode 动态组合
 - 优化 GPU 利用率

1. 高吞吐量：
 - 在多项 benchmark 中表现优秀
 - 特别适合高并发场景
```

---

### 学习任务3：框架对比（30分钟）

| 特性 | vLLM | TensorRT-LLM | LightLLM |
|------|------|-------------|----------|
| Batching | Continuous Batching | Inflight Batching | Dynamic Split Fuse |
| 调度器语言 | Python | C++ | Python/C++ |
| KV Cache 管理 | PagedAttention | PagedAttention | Token Attention |
| 易用性 | 高 | 中 | 中 |
| 性能 | 高 | 很高 | 很高 |
| 灵活性 | 高 | 中 | 中 |
| 主要优势 | 生态好、PagedAttention | NVIDIA 官方优化 | 高吞吐 |

### 晚间任务：对比分析报告（1小时）

#### 报告模板

```markdown
# LLM 推理框架调度策略对比

## vLLM
- Batching: Continuous Batching
- Scheduler: Python
- KV Cache: PagedAttention
- 优点: 灵活、生态好
- 缺点: Python scheduler overhead

## TensorRT-LLM
- Batching: Inflight Batching
- Scheduler: C++
- KV Cache: PagedAttention
- 优点: 性能高、NVIDIA 官方
- 缺点: 灵活性较低

## LightLLM
- Batching: Dynamic Split Fuse
- KV Cache: Token Attention
- 优点: 高吞吐
- 缺点: 生态较小

## Day 40（周五）：项目推进 —— Mini 推理引擎 v1

> **今日目标**：将 Mini 引擎升级到 v1，支持多请求并发、Continuous Batching、基础 Scheduler、优先级配置。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，"如何支持多请求并发"是推理系统面试核心

---

### 学习任务1：Mini 引擎 v1 架构设计（45分钟）

#### 设计目标

在 Mini 引擎 v0 基础上升级：
1. 支持多请求并发提交
2. 实现 Continuous Batching
3. 实现基础 Scheduler（token budget、max seqs、优先级）
4. 线程安全的请求队列
5. 异步返回结果（future/callback）

#### 架构图

```
┌─────────────────────────────────────────────┐
│ Mini Engine v1 │
│ │
│ ┌─────────────┐ │
│ │ Request Queue│ ← 多线程安全缓冲 │
│ └──────┬──────┘ │
│ │ │
│ ▼ │
│ ┌─────────────┐ │
│ │ Scheduler │ ← 每轮构建 batch │
│ │ - budget │ │
│ │ - priority │ │
│ └──────┬──────┘ │
│ │ │
│ ▼ │
│ ┌─────────────┐ │
│ │ Worker │ ← 执行 model forward │
│ │ - Prefill │ │
│ │ - Decode │ │
│ └──────┬──────┘ │
│ │ │
│ ▼ │
│ ┌─────────────┐ │
│ │ Sampler │ ← greedy / top-k / top-p │
│ └──────┬──────┘ │
│ │ │
│ ▼ │
│ ┌─────────────┐ │
│ │ Result Queue│ ← 异步返回 │
│ └─────────────┘ │
│ │
└─────────────────────────────────────────────┘
```

---

### 晚间编程任务：Mini 推理引擎 v1（1.5小时）

#### 完整代码

```python
# mini_engine_v1.py —— Mini 推理引擎 v1（多请求 + Continuous Batching + Scheduler）
# 运行命令: python mini_engine_v1.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import List, Dict, Optional
from mini_engine_v0 import MiniLLM, MiniTokenizer

class Request:
 def __init__(self, request_id, input_ids, max_new_tokens=20, priority=0):
 self.request_id = request_id
 self.input_ids = input_ids # List[int]
 self.max_new_tokens = max_new_tokens
 self.priority = priority
 self.generated_ids = []
 self.kv_cache = None # per-layer (k, v)
 self.status = "waiting" # waiting/running/finished
 self.future = Future()
 self.created_at = time.time()

class MiniScheduler:
 """基础 Scheduler：token budget + max seqs + 优先级"""
 def __init__(self, max_token_budget=100, max_num_seqs=8):
 self.max_token_budget = max_token_budget
 self.max_num_seqs = max_num_seqs

 def schedule(self, waiting: deque, running: Dict[int, Request]) -> List[Request]:
 batch = []
 token_budget = self.max_token_budget

 # 1. 继续运行 running 请求（decode）
 # 按优先级排序
 running_sorted = sorted(running.values(), key=lambda r: -r.priority)
 for req in running_sorted:
 if req.status == "running":
 if token_budget >= 1 and len(batch) < self.max_num_seqs:
 batch.append(req)
 token_budget -= 1

 # 2. 从 waiting 中加入新请求（prefill）
 waiting_sorted = sorted(waiting, key=lambda r: -r.priority)
 still_waiting = deque()
 for req in waiting_sorted:
 prompt_len = len(req.input_ids)
 if token_budget >= prompt_len and len(batch) < self.max_num_seqs:
 req.status = "running"
 batch.append(req)
 token_budget -= prompt_len
 else:
 still_waiting.append(req)

 return batch, still_waiting

class MiniEngineV1:
 """Mini 推理引擎 v1"""
 def __init__(self, model: MiniLLM, tokenizer: MiniTokenizer,
 max_token_budget=100, max_num_seqs=8, device="cuda"):
 self.model = model.to(device).eval()
 self.tokenizer = tokenizer
 self.device = device
 self.scheduler = MiniScheduler(max_token_budget, max_num_seqs)

 self.waiting_queue = deque()
 self.running_requests: Dict[int, Request] = {}
 self.lock = threading.Lock()
 self.stop_event = threading.Event()
 self.next_request_id = 0

 self.worker_thread = threading.Thread(target=self._worker_loop)
 self.worker_thread.start()

 def submit(self, prompt: str, max_new_tokens=20, priority=0) -> Future:
 """提交请求，返回 Future"""
 with self.lock:
 req_id = self.next_request_id
 self.next_request_id += 1

 input_ids = self.tokenizer.encode(prompt)
 req = Request(req_id, input_ids, max_new_tokens, priority)

 with self.lock:
 self.waiting_queue.append(req)

 return req.future

 @torch.no_grad()
 def _run_iteration(self, batch: List[Request]):
 if not batch:
 return

 # 构建 input tensors
 # 简化：每个请求单独 forward（不做真正的 batch 合并）
 # 实际应该按长度分组 padding 或使用 continuous batching 的 merge
 for req in batch:
 if req.status == "running":
 if req.kv_cache is None:
 # Prefill
 input_ids_tensor = torch.tensor([req.input_ids], device=self.device)
 logits, kv_cache = self.model(input_ids_tensor, use_cache=True)
 req.kv_cache = kv_cache
 next_token_logits = logits[:, -1, :]
 else:
 # Decode
 next_input_id = req.generated_ids[-1] if req.generated_ids else req.input_ids[-1]
 input_ids_tensor = torch.tensor([[next_input_id]], device=self.device)
 logits, kv_cache = self.model(input_ids_tensor, kv_cache=req.kv_cache, use_cache=True)
 req.kv_cache = kv_cache
 next_token_logits = logits[:, -1, :]

 # greedy sampling
 next_token = torch.argmax(next_token_logits, dim=-1).item()
 req.generated_ids.append(next_token)

 if len(req.generated_ids) >= req.max_new_tokens:
 req.status = "finished"

 def _worker_loop(self):
 while not self.stop_event.is_set():
 with self.lock:
 # 移除 finished 请求
 finished_ids = [rid for rid, req in self.running_requests.items() if req.status == "finished"]
 for rid in finished_ids:
 req = self.running_requests.pop(rid)
 output_text = self.tokenizer.decode(req.input_ids + req.generated_ids)
 req.future.set_result(output_text)

 # 调度
 batch, self.waiting_queue = self.scheduler.schedule(
 self.waiting_queue, self.running_requests
 )

 # 新加入 running 的请求
 for req in batch:
 if req.request_id not in self.running_requests:
 self.running_requests[req.request_id] = req

 if batch:
 self._run_iteration(batch)
 else:
 time.sleep(0.001)

 def shutdown(self):
 self.stop_event.set()
 self.worker_thread.join()

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}\n")

 model = MiniLLM(vocab_size=1000, d_model=512, n_heads=8, n_layers=4)
 tokenizer = MiniTokenizer(vocab_size=1000)
 engine = MiniEngineV1(model, tokenizer, max_token_budget=80, max_num_seqs=4, device=device)

 # 提交多个请求
 prompts = [
 "hello world",
 "this is a longer prompt for testing batching",
 "short",
 "another test prompt here",
 ]

 futures = []
 for i, prompt in enumerate(prompts):
 priority = 1 if i == 0 else 0
 future = engine.submit(prompt, max_new_tokens=5, priority=priority)
 futures.append((i, future))
 print(f"Submitted request {i}: '{prompt}'")

 # 等待结果
 print("\nWaiting for results...")
 for i, future in futures:
 result = future.result()
 print(f"Request {i} result: {result}")

 engine.shutdown()

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
# 确保 mini_engine_v0.py 在同一目录
python mini_engine_v1.py

# 预期输出
# Using device: cuda
# 
# Submitted request 0: 'hello world'
# ...
# 
# Waiting for results...
# Request 0 result: ...
# ...
```

#### 练习题

**练习1（基础）**：实现真正的 batch 合并 forward（将多个请求的 input 拼接到一个 tensor）。

**练习2（进阶）**：添加 timeout 机制，超过最大等待时间的请求返回错误。

**练习3（综合）**：实现请求取消 API：`cancel(request_id)`。

---

### 今日面试题

**面试题1**：如何将单请求推理引擎扩展为多请求并发？需要解决哪些问题？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. **请求队列**：线程安全的缓冲，支持异步提交
2. **调度器**：决定每轮运行哪些请求（Dynamic/Continuous Batching）
3. **KV Cache 管理**：每个请求独立维护 KV Cache，动态分配/释放
4. **结果返回**：使用 future/callback 异步返回结果
5. **资源隔离**：token budget、显存预算，防止一个请求耗尽资源
6. **生命周期管理**：waiting → running → finished，超时/取消处理

**面试题2**：推理引擎中，优先级调度有什么优缺点？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **优点**：
 - 高优先级请求（如付费用户、交互式请求）获得更快响应
 - 可以配置不同 SLA
- **缺点**：
 - 低优先级请求可能饥饿
 - 需要复杂的调度逻辑和预算管理
 - 优先级反转问题（高优先级等待低优先级的资源）
- **缓解**：
 - 设置最大等待时间
 - 动态调整优先级
 - 资源预留

---

### 今日自测清单

- [ ] 能说出多请求推理引擎的 6 个核心问题
- [ ] Mini 引擎 v1 运行成功，能处理多个并发请求
- [ ] 实现了 Continuous Batching 和基础 Scheduler
- [ ] 支持优先级配置
- [ ] 使用 Future 异步返回结果

---

## Day 41（周六）：Latency / Throughput 测试

> **今日目标**：测试 Mini 引擎 v1 在不同 batch size 和请求分布下的 latency 和 throughput，绘制 throughput-latency 曲线，识别系统饱和点。
> **时间分配**：6小时全天投入（benchmark 框架2h + 扫描测试2h + 曲线绘制2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"如何评估推理系统性能"是系统优化的核心

---

### 任务1：Benchmark 框架设计（2小时）

#### 测试场景

| 场景 | 描述 |
|------|------|
| 固定并发数 | 同时提交 N 个请求，观察吞吐和延迟 |
| 固定吞吐量 | 以固定 QPS 发送请求，观察 P99 延迟 |
| 短/长/混合请求 | 不同 prompt 长度和生成长度分布 |
| 优先级 | 高优先级和低优先级请求混合 |

#### 关键指标

| 指标 | 含义 |
|------|------|
| Throughput | 每秒生成的 tokens 数 |
| Avg Latency | 平均端到端延迟 |
| P50/P99 Latency | 中位数/99 分位延迟 |
| TTFT | 首 token 时间 |
| TPOT | 每输出 token 时间 |
| Batch Size Distribution | batch 大小分布 |

---

### 任务2：编写 Benchmark 脚本（2小时）

#### 完整代码

```python
# benchmark_engine_v1.py —— Mini 引擎 v1 性能测试
# 运行命令: python benchmark_engine_v1.py

import time
import threading
import statistics
import matplotlib.pyplot as plt
from concurrent.futures import wait
from mini_engine_v1 import MiniEngineV1
from mini_engine_v0 import MiniLLM, MiniTokenizer

class Benchmark:
 def __init__(self, engine, prompts, max_new_tokens=10):
 self.engine = engine
 self.prompts = prompts
 self.max_new_tokens = max_new_tokens
 self.results = []

 def run_fixed_concurrency(self, concurrency=4):
 """固定并发数测试"""
 self.results = []
 futures = []

 start_time = time.time()
 for i in range(concurrency):
 prompt = self.prompts[i % len(self.prompts)]
 future = self.engine.submit(prompt, self.max_new_tokens)
 futures.append(future)

 wait(futures)
 total_time = time.time() - start_time

 latencies = []
 for f in futures:
 result = f.result()
 # 这里简化：latency 用总时间近似
 latencies.append(total_time)

 total_tokens = concurrency * self.max_new_tokens
 throughput = total_tokens / total_time

 return {
 'concurrency': concurrency,
 'total_time': total_time,
 'throughput': throughput,
 'avg_latency': statistics.mean(latencies),
 'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)],
 }

 def run_qps_test(self, qps=10, duration=10):
 """固定 QPS 测试"""
 futures = []
 latencies = []
 start_time = time.time()
 next_send_time = start_time

 while time.time() - start_time < duration:
 if time.time() >= next_send_time:
 prompt = self.prompts[len(futures) % len(self.prompts)]
 submit_time = time.time()
 future = self.engine.submit(prompt, self.max_new_tokens)
 futures.append((submit_time, future))
 next_send_time += 1.0 / qps
 else:
 time.sleep(0.001)

 # 等待所有请求完成
 for submit_time, future in futures:
 future.result()
 latencies.append(time.time() - submit_time)

 total_tokens = len(futures) * self.max_new_tokens
 throughput = total_tokens / duration

 return {
 'qps': qps,
 'total_requests': len(futures),
 'throughput': throughput,
 'avg_latency': statistics.mean(latencies),
 'p50_latency': sorted(latencies)[len(latencies) // 2],
 'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)],
 }

def scan_concurrency(engine, prompts, max_new_tokens=10):
 """扫描不同并发数下的性能"""
 benchmark = Benchmark(engine, prompts, max_new_tokens)
 results = []

 for concurrency in [1, 2, 4, 8, 16, 32]:
 print(f"\nTesting concurrency={concurrency}...")
 result = benchmark.run_fixed_concurrency(concurrency)
 results.append(result)
 print(f" Throughput: {result['throughput']:.2f} tokens/s")
 print(f" Avg Latency: {result['avg_latency']:.3f}s")
 print(f" P99 Latency: {result['p99_latency']:.3f}s")

 return results

def plot_results(results, output_file="throughput_latency.png"):
 """绘制 throughput-latency 曲线"""
 throughputs = [r['throughput'] for r in results]
 latencies = [r['avg_latency'] for r in results]

 plt.figure(figsize=(10, 6))
 plt.plot(throughputs, latencies, 'o-', linewidth=2, markersize=8)
 plt.xlabel("Throughput (tokens/s)")
 plt.ylabel("Average Latency (s)")
 plt.title("Throughput vs Latency")
 plt.grid(True)
 plt.savefig(output_file)
 print(f"\nPlot saved to {output_file}")

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}\n")

 model = MiniLLM(vocab_size=1000, d_model=512, n_heads=8, n_layers=4)
 tokenizer = MiniTokenizer(vocab_size=1000)
 engine = MiniEngineV1(model, tokenizer, max_token_budget=200, max_num_seqs=8, device=device)

 prompts = [
 "hello world",
 "this is a test prompt",
 "another example prompt for testing batching and scheduling",
 "short",
 "a medium length prompt here",
 ]

 # 扫描并发
 results = scan_concurrency(engine, prompts, max_new_tokens=10)

 # 绘制曲线
 plot_results(results)

 # 固定 QPS 测试
 print("\n=== QPS Test ===")
 benchmark = Benchmark(engine, prompts, max_new_tokens=10)
 qps_result = benchmark.run_qps_test(qps=5, duration=10)
 print(f"QPS: {qps_result['qps']}")
 print(f"Total requests: {qps_result['total_requests']}")
 print(f"Throughput: {qps_result['throughput']:.2f} tokens/s")
 print(f"Avg Latency: {qps_result['avg_latency']:.3f}s")
 print(f"P99 Latency: {qps_result['p99_latency']:.3f}s")

 engine.shutdown()

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
# 需要安装 matplotlib: pip install matplotlib
python benchmark_engine_v1.py

# 预期输出
# Using device: cuda
# 
# Testing concurrency=1...
# Throughput: x.xx tokens/s
# Avg Latency: x.xxxs
# ...
# 
# Plot saved to throughput_latency.png
```

---

### 任务3：分析饱和点（2小时）

#### 饱和点识别

```
Throughput-Latency 曲线：
 - 低并发：throughput 随并发线性增长，latency 增长缓慢
 - 接近饱和：throughput 增长放缓，latency 开始快速上升
 - 饱和后：throughput 不再增长，latency 急剧上升

饱和点标志：
 - GPU 利用率接近 100%
 - Kernel 间隙很小
 - 队列开始堆积
```

#### 优化方向

| 瓶颈 | 表现 | 优化 |
|------|------|------|
| Compute-bound | throughput 不再增长，SM util 高 | 量化、更大 batch、模型优化 |
| Memory-bound | latency 随并发线性增长 | KV Cache 优化、PagedAttention |
| Launch overhead | kernel 间隙大 | CUDA Graph、kernel fusion |
| Scheduling overhead | scheduler 成为瓶颈 | C++ scheduler、预分配 buffer |

### 今日面试题

**面试题1**：如何做 LLM 推理系统的 throughput-latency benchmark？需要关注哪些指标？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **测试方法**：
 - 固定并发数：同时提交 N 个请求
 - 固定 QPS：以恒定速率发送请求
 - 混合请求：短/长/不同优先级
1. **关键指标**：
 - Throughput（tokens/s）
 - Avg/P50/P99 latency
 - TTFT、TPOT
 - Batch size distribution
 - GPU utilization
1. **分析方法**：
 - 绘制 throughput-latency 曲线
 - 找到饱和点
 - 用 nsys 分析瓶颈类型

**面试题2**：如何识别推理系统的饱和点？饱和后如何优化？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **识别饱和点**：
 - Throughput 不再随并发增加而增长
 - Latency 开始快速上升
 - GPU 利用率接近 100%
 - 请求队列开始堆积
- **饱和后优化**：
 - Compute-bound：量化、模型蒸馏、更大 batch
 - Memory-bound：KV Cache 量化、PagedAttention
 - Launch overhead：CUDA Graph、kernel fusion
 - Scheduling：C++ scheduler、负载均衡

---

### 今日自测清单

- [ ] 完成 throughput-latency benchmark 脚本
- [ ] 扫描至少 5 个不同并发数
- [ ] 能识别系统的饱和点
- [ ] 绘制 throughput-latency 曲线
- [ ] 完成固定 QPS 测试
- [ ] 能根据曲线判断瓶颈类型

---

## Day 42（周日）：调度优化策略总结

> **今日目标**：整理调度相关的优化策略，复盘本周面试题，整理 GitHub 仓库和性能报告。
> **时间分配**：6小时全天投入（策略总结2h + 面试复盘2h + GitHub 整理2h）
> **面试考察度**：⭐⭐⭐⭐⭐ 高频，"调度优化策略"是系统设计的总结性考点

---

### 任务1：调度策略对比表（2小时）

#### 策略对比

| 策略 | 原理 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| Static Batching | 固定 batch size | 简单 demo | 简单 | 吞吐低、延迟高 |
| Dynamic Batching | 请求聚合、超时等待 | 吞吐优先 | 提高 GPU 利用率 | 请求级阻塞、padding 浪费 |
| Continuous Batching | 每轮重新构建 batch | LLM 推理 | 吞吐+延迟兼顾 | 实现复杂 |
| Priority Scheduling | 高优先级优先 | 多租户 | SLA 保障 | 低优先级饥饿 |
| Preemption | 抢占低优先级请求 | 显存不足时 | 保证高优先级 | 额外 overhead |
| Chunked Prefill | 拆分长 prefill | 长短混合 | 平滑 latency | 调度复杂 |
| Speculative Decoding | 小模型预测、大模型验证 | 低延迟 | 降低 TBT | 需要 draft model |

#### 策略选择决策树

```
选择 batching 策略：
 1. 是否要求最低延迟？
 → 是：Static / 小 batch + 优先级
 → 否：继续
 1. 请求到达是否连续？
 → 是：Dynamic Batching
 → 否：Static Batching
 1. 是否是 LLM 自回归生成？
 → 是：Continuous Batching
 → 否：Dynamic Batching
 1. 是否多租户？
 → 是：+ Priority Scheduling
 1. 是否有长 prompt？
 → 是：+ Chunked Prefill
```

---

### 任务2：面试复盘（2小时）

#### 本周核心面试题回顾

1. Dynamic Batching 的原理和优缺点
2. Padding 问题和优化
3. Continuous Batching 与 Dynamic Batching 的区别
4. Continuous Batching 为什么适合 LLM
5. Prefill + Decode 混合调度的挑战
6. vLLM Scheduler 的 schedule() 流程
7. SchedulingBudget 的作用
8. Preemption 的两种模式
9. TensorRT-LLM Inflight Batching 与 vLLM 的差异
10. Chunked Prefill 原理
11. 多请求并发需要解决哪些问题
12. 优先级调度的优缺点
13. Throughput-latency benchmark 方法
14. 如何识别饱和点
15. 调度策略对比和选择

---

### 任务3：GitHub 整理与报告（2小时）

#### 仓库结构建议

```
week6-batching-scheduling/
├── day36-dynamic-batching/
│ ├── dynamic_batcher.py
│ └── README.md
├── day37-continuous-batching/
│ ├── continuous_batcher.py
│ └── README.md
├── day38-vllm-scheduler/
│ └── scheduler_analysis.md
├── day39-framework-comparison/
│ └── framework_comparison.md
├── day40-mini-engine-v1/
│ ├── mini_engine_v1.py
│ └── README.md
├── day41-benchmark/
│ ├── benchmark_engine_v1.py
│ ├── throughput_latency.png
│ └── benchmark_report.md
└── day42-summary/
 ├── scheduling_strategy_comparison.md
 └── week6_report.md
```

#### 性能报告模板

```markdown
# Week 6 Batching & Scheduling 报告

## 测试环境
- GPU: [型号]
- CUDA: 12.x
- PyTorch: 2.x

## Dynamic Batching
- max_batch_size=4, max_wait_time=0.05s
- 10 个请求平均聚合 batch size: x
- 平均延迟: x.xxxs

## Continuous Batching
- 3 个序列动态加入/退出
- GPU 利用率: xx%

## Mini 引擎 v1
- 支持 4 个并发请求
- Continuous Batching + 优先级调度

## Throughput-Latency 曲线
- 并发 1: throughput=x, latency=x
- 并发 4: throughput=x, latency=x
- 并发 8: throughput=x, latency=x
- 饱和点: 并发=x

## 调度策略对比
[见 scheduling_strategy_comparison.md]
```

---

### 今日面试题

**面试题1**：对比 Static Batching、Dynamic Batching、Continuous Batching，分别适用于什么场景？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **Static Batching**：固定 batch size，一起开始一起结束。适用于简单 demo 或请求长度完全相同的场景
- **Dynamic Batching**：请求级聚合，超时等待。适用于吞吐优先、请求到达连续但非 LLM 自回归的场景
- **Continuous Batching**：iteration-level 调度，请求动态加入/退出。适用于 LLM 自回归生成，因为生成长度差异大
- **选择**：LLM 推理服务通常用 Continuous Batching；传统 CV/NLP 服务常用 Dynamic Batching

**面试题2**：在 LLM 推理服务中，如何平衡 throughput 和 latency？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. **Continuous Batching**：基础，本身就在平衡吞吐和延迟
2. **Token budget 控制**：限制每轮处理的 token 数，避免 prefill 阻塞 decode
3. **Chunked Prefill**：拆分长 prefill，平滑 decode latency
4. **优先级调度**：保障关键请求的延迟
5. **Timeout 机制**：避免低优先级请求无限等待
6. **动态扩缩容**：根据负载调整实例数
7. **量化/压缩**：提高吞吐同时降低延迟
8. **关键**：根据 SLA 要求做 trade-off，没有绝对最优

---

### 今日自测清单

- [ ] 能对比 5 种调度策略的优缺点
- [ ] 能用决策树选择合适的 batching 策略
- [ ] 完成本周 15 道面试题的自问自答
- [ ] 整理 GitHub 仓库
- [ ] 生成 Week 6 性能报告和调度策略对比文档
- [ ] 能解释如何平衡 throughput 和 latency
- [ ] 规划 Week 7（系统整合）的学习重点

---

## 附录A：第6周面试题汇总

| 题号 | 题目 | 考察频率 | 相关天数 | 难度 |
|------|------|---------|---------|------|
| 1 | Dynamic Batching 的原理和优缺点？ | ⭐⭐⭐⭐ | Day 36 | 中 |
| 2 | Padding 有什么问题？如何优化？ | ⭐⭐⭐⭐ | Day 36 | 中 |
| 3 | Continuous vs Dynamic Batching？ | ⭐⭐⭐⭐⭐ | Day 37, 42 | 中 |
| 4 | Continuous Batching 为什么适合 LLM？ | ⭐⭐⭐⭐⭐ | Day 37 | 中 |
| 5 | Prefill + Decode 混合调度的挑战？ | ⭐⭐⭐⭐ | Day 37 | 中 |
| 6 | vLLM Scheduler 的 schedule() 流程？ | ⭐⭐⭐⭐⭐ | Day 38 | 中 |
| 7 | SchedulingBudget 的作用？ | ⭐⭐⭐⭐ | Day 38 | 中 |
| 8 | Preemption 的两种模式？ | ⭐⭐⭐⭐ | Day 38 | 中 |
| 9 | Inflight Batching 和 Continuous Batching 区别？ | ⭐⭐⭐⭐ | Day 39 | 中 |
| 10 | Chunked Prefill 是什么？ | ⭐⭐⭐⭐ | Day 39 | 中 |
| 11 | 多请求并发需要解决哪些问题？ | ⭐⭐⭐⭐⭐ | Day 40 | 中 |
| 12 | 优先级调度的优缺点？ | ⭐⭐⭐⭐ | Day 40 | 中 |
| 13 | 如何做 throughput-latency benchmark？ | ⭐⭐⭐⭐ | Day 41 | 中 |
| 14 | 如何识别饱和点？ | ⭐⭐⭐⭐ | Day 41 | 中 |
| 15 | 调度策略如何选择？ | ⭐⭐⭐⭐⭐ | Day 42 | 高 |

---

## 附录C：关键公式汇总

**1. Throughput**
```
Throughput (tokens/s) = total_generated_tokens / total_time
```

**2. Latency**
```
Avg Latency = sum(end_time - submit_time) / num_requests
P99 Latency = 99th percentile of latencies
```

**3. Batch 利用率**
```
GPU Utilization = actual_compute_time / total_time
 = 1 - kernel_gap_ratio
```

**4. Token Budget**
```
token_budget = prefill_tokens + decode_tokens
 = sum(len(prompt) for new requests) + num_running_decode_requests
```

---

## 附录D：调度优化速查表

| 问题 | 现象 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| 吞吐低 | GPU 利用率低 | nsys SM util | 增大 batch、continuous batching |
| 延迟高 | 请求等待时间长 | 记录 submit→end 时间 | 减小 batch、优先级调度 |
| 长请求阻塞短请求 | Dynamic batching | 观察请求完成顺序 | 改用 Continuous Batching |
| Prefill 阻塞 decode | Decode latency 抖动 | 按 iteration 分析 | Chunked Prefill、token budget |
| 低优先级饥饿 | 低优先级总不完成 | 记录等待时间 | 设置最大等待时间、动态优先级 |
| 显存不足 | OOM 或 preemption 频繁 | 监控显存 | PagedAttention、KV 量化 |
| 调度器 CPU 瓶颈 | 调度时间占比高 | profile Python 层 | C++ scheduler、预分配 buffer |
| Kernel 间隙大 | launch overhead | nsys timeline | CUDA Graph、kernel fusion |

---

> 💡 **Week 6 总结**：本周我们建立了"系统感"，从 Dynamic Batching 到 Continuous Batching，从 vLLM Scheduler 到 TensorRT-LLM Inflight Batching，最终构建出支持多请求并发的 Mini 推理引擎 v1，并完成了 throughput-latency benchmark。最核心的收获是：LLM 推理调度不是简单的 batch 聚合，而是要在 iteration 级别动态平衡 prefill/decode、吞吐/延迟、优先级/公平性。下周将进入系统整合，把前六周的所有组件（GEMM、FlashAttention、Softmax/LayerNorm、KV Cache、Batching、Scheduler）联调成一个完整的 Mini AI Infra 系统。
