# 第7周深度展开：系统整合（7天）

> **适用对象**：陈斌斌（已完成第6周学习，掌握 Dynamic/Continuous Batching、vLLM Scheduler、Mini 推理引擎 v1）
> **本周目标**：将前六周所学整合为一个完整的 Mini AI Infra 系统，完成多请求并发、完整调度器、全部自定义 Kernel 接入、端到端联调和稳定性测试
> **时间投入**：工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h
> **周日里程碑**：Mini AI Infra 系统能连续处理 1000+ 请求，内存稳定，产出系统级性能报告和完整项目文档

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 整合 GEMM、FlashAttention、Softmax/LayerNorm、KV Cache、Continuous Batching、Scheduler，构建完整的 Mini AI Infra 系统 |
| **核心产出** | ① 多请求并发支持 ② 完整调度器 ③ SGLang/LightLLM 高级特性分析 ④ 全部自定义 Kernel 整合 ⑤ 系统联调 ⑥ 全链路 Profiling ⑦ 代码重构与文档 |
| **验收标准** | ① 支持并发提交多个请求且结果正确返回 ② 完整调度器支持优先级、超时、资源预算 ③ 系统能连续处理 1000+ 请求不崩溃 ④ 内存使用稳定，无持续增长 ⑤ 端到端性能优于或等于 PyTorch eager ⑥ 产出 README、架构图、性能报告 |
| **面试准备** | 积累12-15道系统整合专题面试题，覆盖并发控制、调度器设计、Kernel 融合、系统联调、Profiling、性能优化六大主题 |

### 本周知识图谱

```
Day 43: 多请求并发支持 → 线程安全队列、异步 Future、结果回调
 ↓
Day 44: 完整调度器 → 优先级、超时、资源预算、抢占
 ↓
Day 45: SGLang / LightLLM 高级特性 → Speculative Decoding、Chunked Prefill、Prefix Caching
 ↓
Day 46: 整合全部自定义 Kernel → GEMM、FlashAttention、Softmax、LayerNorm 接入
 ↓
Day 47: 系统联调 → KV Cache + Batching + Scheduler + Kernel 端到端测试
 ↓
Day 48: 全链路 Profiling → 系统级瓶颈定位、与 vLLM 对比
 ↓
Day 49: 代码重构与文档 → README、架构图、接口统一、稳定性测试
```

### 前置准备清单

#### 硬件/软件验证
- [ ] 已完成第6周所有 Coding 任务（Mini 引擎 v1、Continuous Batching）
- [ ] Week 2-4 的自定义 Kernel 代码可用（warp_reduce、register_blocking_gemm、flash_attention_v2、softmax_layernorm）
- [ ] PyTorch 可用且 `torch.__version__ >= 2.0`
- [ ] `ncu` 和 `nsys` 可用
- [ ] 理解前六周的所有核心概念

#### 验证命令
```bash
# 验证前六周代码路径
ls week2/day13-integrated-gemm/integrated_gemm.cu 2>/dev/null || echo "integrated_gemm not found"
ls week4/day23-handwritten-kernel/flash_attention_v2.cu 2>/dev/null || echo "flash_attention_v2 not found"
ls week3/day16-kernels/softmax_layernorm.cu 2>/dev/null || echo "softmax_layernorm not found"

# 验证 PyTorch + nsys
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nsys --version
ncu --version
```

---

## Day 43（周一）：多请求并发支持

> **今日目标**：为 Mini 引擎添加线程安全的请求队列、异步处理机制和结果回调，支持并发提交多个请求。
> **面试考察度**：⭐⭐⭐⭐ 高频，"多请求并发"是推理系统从 demo 到产品的关键

---

### 学习任务1：并发系统设计（45分钟）

#### 核心组件

```
多请求并发推理系统需要：
 1. 线程安全的请求队列
 2. 异步处理线程池或事件循环
 3. 结果返回机制（Future / Callback / Streaming）
 4. 请求生命周期管理
 5. 错误处理与超时控制
```

#### 并发模型选择

```
模型1：单 Worker 线程 + 共享队列
 - 简单，适合教学
 - 调度器和模型执行串行

模型2：调度线程 + 执行线程分离
 - 调度器专门构建 batch
 - Worker 专门执行 forward
 - 适合更复杂的系统

模型3：线程池 + 任务队列
 - 多个 Worker 并行处理
 - 适合多 GPU 场景

Mini 系统选择模型2：调度线程 + 执行线程
```

#### 线程安全要点

```
1. 使用锁保护共享状态（waiting queue、running map、KV cache metadata）
2. 条件变量通知新请求到达
3. 避免在锁内执行耗时操作（如 CUDA forward）
4. 使用原子操作更新计数器
5. 注意 Python GIL 对多线程的限制
```

### 学习任务2：异步返回机制（30分钟）

#### Future vs Callback vs Streaming

```
Future：
 - 优点：接口简单，调用方可以阻塞等待
 - 缺点：大结果占用内存，不适合流式输出

Callback：
 - 优点：灵活，结果到达时触发
 - 缺点：代码复杂，嵌套回调

Streaming：
 - 优点：用户体验好，token 逐个返回
 - 缺点：需要维护流状态

实际系统通常同时支持三种。
```

---

### 晚间编程任务：并发请求处理模块（1小时）

#### 完整代码

```python
# request_queue.py —— 线程安全请求队列 + Future + Callback
# 运行命令: python request_queue.py

import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Callable, Optional

class InferenceRequest:
 def __init__(self, request_id, prompt, max_new_tokens=20,
 priority=0, timeout=None,
 callback: Optional[Callable] = None):
 self.request_id = request_id
 self.prompt = prompt
 self.max_new_tokens = max_new_tokens
 self.priority = priority
 self.timeout = timeout
 self.callback = callback
 self.future = Future()
 self.submit_time = time.time()
 self.start_time = None
 self.end_time = None

 def set_result(self, result):
 self.end_time = time.time()
 self.future.set_result(result)
 if self.callback:
 self.callback(self.request_id, result)

 def set_exception(self, exception):
 self.end_time = time.time()
 self.future.set_exception(exception)

class ThreadSafeRequestQueue:
 """线程安全请求队列，支持优先级"""
 def __init__(self):
 self._queue = deque()
 self._lock = threading.Lock()
 self._cond = threading.Condition(self._lock)

 def put(self, request: InferenceRequest):
 with self._cond:
 # 按优先级插入
 inserted = False
 for i, req in enumerate(self._queue):
 if request.priority > req.priority:
 self._queue.insert(i, request)
 inserted = True
 break
 if not inserted:
 self._queue.append(request)
 self._cond.notify()

 def get(self, timeout=None) -> Optional[InferenceRequest]:
 with self._cond:
 if not self._queue and timeout is not None:
 self._cond.wait(timeout)
 if self._queue:
 return self._queue.popleft()
 return None

 def get_batch(self, max_size=8, max_wait_time=0.05) -> list:
 """批量获取，带超时"""
 batch = []
 deadline = time.time() + max_wait_time

 with self._cond:
 while len(batch) < max_size:
 remaining = deadline - time.time()
 if remaining <= 0:
 break
 if not self._queue:
 self._cond.wait(remaining)
 if self._queue:
 batch.append(self._queue.popleft())

 return batch

 def __len__(self):
 with self._lock:
 return len(self._queue)

def main():
 queue = ThreadSafeRequestQueue()

 def callback(req_id, result):
 print(f"Callback: Request {req_id} finished with length {len(result)}")

 # 提交请求
 requests = []
 for i in range(5):
 req = InferenceRequest(
 request_id=i,
 prompt=f"prompt_{i}",
 max_new_tokens=10,
 priority=i % 2, # 偶数高优先级
 callback=callback
 )
 queue.put(req)
 requests.append(req)

 # 模拟 worker 处理
 print("Processing requests...")
 for _ in range(5):
 req = queue.get(timeout=0.1)
 if req:
 print(f"Processing request {req.request_id} (priority={req.priority})")
 req.set_result(f"result_for_{req.request_id}")

 # 等待所有 Future
 for req in requests:
 if not req.future.done():
 continue
 print(f"Request {req.request_id}: {req.future.result()}")

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python request_queue.py

# 预期输出
# Processing requests...
# Processing request 1 (priority=1)
# Callback: Request 1 finished with length 16
# ...
```

#### 练习题

**练习1（基础）**：为 `InferenceRequest` 添加 streaming 支持，每次生成 token 调用回调。

**练习2（进阶）**：实现请求超时自动取消机制。

**练习3（综合）**：将 `ThreadSafeRequestQueue` 集成到 Mini 引擎 v1 中。

---

### 今日面试题

**面试题1**：推理系统中如何实现多请求并发？需要注意哪些线程安全问题？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **线程安全队列**：使用锁和条件变量保护请求队列
2. **异步返回**：使用 Future、Callback 或 Streaming
3. **共享状态保护**：waiting queue、running map、KV cache metadata 都需要锁保护
4. **避免在锁内执行 CUDA 操作**：防止死锁和性能下降
5. **生命周期管理**：waiting → running → finished / timeout / cancel
6. **Python GIL**：注意多线程下 Python 代码的并发限制

**面试题2**：Future、Callback、Streaming 三种结果返回方式各有什么优缺点？（⭐⭐⭐ 中频）

**参考答案要点**：
- **Future**：接口简单，调用方阻塞等待；不适合大结果或流式输出
- **Callback**：灵活，结果到达时触发；嵌套回调复杂
- **Streaming**：用户体验好，token 实时返回；需要维护流状态
- **实际系统**：通常同时支持三种，根据场景选择

---

### 今日自测清单

- [ ] 能解释多请求并发系统的核心组件
- [ ] 能说出 5 个线程安全要点
- [ ] `request_queue.py` 运行成功，支持优先级
- [ ] 理解 Future/Callback/Streaming 的区别

---

## Day 44（周二）：完整调度器

> **今日目标**：实现完整的调度器，支持优先级调度、超时控制、资源预算（显存、batch size 上限），并支持抢占。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，完整调度器是推理系统的核心能力

---

### 学习任务1：调度器设计（1小时）

#### 完整调度器功能

```
一个生产级调度器需要支持：
 1. 优先级调度：高优先级请求优先
 2. 超时控制：请求最大等待时间和执行时间
 3. 资源预算：显存预算、batch size 上限、token budget
 4. 抢占：显存不足时抢占低优先级请求
 5. 公平性：避免低优先级请求饥饿
 6. 调度策略可配置
```

#### 优先级调度实现

```
数据结构：优先队列（堆）
 - 按优先级排序
 - 同优先级按 FIFO

预算检查：
 - 高优先级请求可以突破某些限制（预留资源）
 - 普通请求受限于常规预算
```

#### 超时控制

```
等待超时：
 - 请求在 waiting queue 中超过 max_waiting_time
 - 返回超时错误

执行超时：
 - 请求运行超过 max_execution_time
 - 强制取消，释放资源
```

#### 抢占策略

```
触发条件：
 - 高优先级请求到来但显存/预算不足
 - 显存碎片严重

被抢占对象选择：
 - 最低优先级
 - 最近最少使用
 - 剩余 token 最少（换出成本低）

抢占后处理：
 - Recompute：丢弃 KV Cache
 - Swap：换出到 CPU
```

### 学习任务2：调度器伪代码（30分钟）

```python
class FullScheduler:
 def __init__(self, token_budget, max_num_seqs, max_waiting_time,
 enable_preemption=True):
 self.token_budget = token_budget
 self.max_num_seqs = max_num_seqs
 self.max_waiting_time = max_waiting_time
 self.enable_preemption = enable_preemption
 
 def schedule(self, waiting, running, swapped, memory_budget):
 batch = []
 budget = self.token_budget
 
 # 1. 处理 running（continuous batching）
 for req in sorted(running.values(), key=lambda r: -r.priority):
 if budget >= 1 and len(batch) < self.max_num_seqs:
 batch.append(req)
 budget -= 1
 
 # 2. 处理 swapped
 for req in sorted(swapped, key=lambda r: -r.priority):
 if memory_budget.can_allocate(req):
 swapped.remove(req)
 batch.append(req)
 budget -= req.required_tokens
 
 # 3. 从 waiting 加入新请求
 for req in sorted(waiting, key=lambda r: -r.priority):
 # 检查超时
 if req.waiting_time > self.max_waiting_time:
 req.timeout()
 continue
 
 # 检查预算
 if budget >= req.required_tokens and len(batch) < self.max_num_seqs:
 if memory_budget.can_allocate(req):
 batch.append(req)
 budget -= req.required_tokens
 elif self.enable_preemption:
 # 抢占低优先级 running 请求
 victim = self.select_victim(running, req)
 if victim:
 self.preempt(victim)
 batch.append(req)
 
 return batch
```

---

### 晚间编程任务：完整调度器实现（1.5小时）

#### 完整代码

```python
# full_scheduler.py —— 完整调度器（优先级、超时、资源预算、抢占）
# 运行命令: python full_scheduler.py

import time
import heapq
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class RequestStatus(Enum):
 WAITING = "waiting"
 RUNNING = "running"
 SWAPPED = "swapped"
 FINISHED = "finished"
 TIMEOUT = "timeout"

@dataclass
class ScheduledRequest:
 req_id: int
 prompt_len: int
 priority: int
 max_new_tokens: int
 status: RequestStatus = RequestStatus.WAITING
 submit_time: float = 0.0
 start_time: Optional[float] = None
 required_kv_blocks: int = 0

class MemoryBudget:
 """显存预算模拟"""
 def __init__(self, total_blocks=64):
 self.total_blocks = total_blocks
 self.used_blocks = 0
 
 def can_allocate(self, blocks: int) -> bool:
 return self.used_blocks + blocks <= self.total_blocks
 
 def allocate(self, blocks: int):
 if not self.can_allocate(blocks):
 raise RuntimeError("Out of memory")
 self.used_blocks += blocks
 
 def free(self, blocks: int):
 self.used_blocks = max(0, self.used_blocks - blocks)
 
 def __repr__(self):
 return f"MemoryBudget({self.used_blocks}/{self.total_blocks})"

class FullScheduler:
 """完整调度器"""
 def __init__(self, token_budget=100, max_num_seqs=8,
 max_waiting_time=5.0, enable_preemption=True,
 reserved_blocks=8):
 self.token_budget = token_budget
 self.max_num_seqs = max_num_seqs
 self.max_waiting_time = max_waiting_time
 self.enable_preemption = enable_preemption
 self.reserved_blocks = reserved_blocks
 
 self.waiting: List[ScheduledRequest] = []
 self.running: Dict[int, ScheduledRequest] = {}
 self.swapped: List[ScheduledRequest] = []
 self.memory = MemoryBudget(total_blocks=64)
 self.time = 0.0
 
 def submit(self, req: ScheduledRequest):
 req.submit_time = self.time
 heapq.heappush(self.waiting, (-req.priority, req.submit_time, req))
 
 def schedule(self) -> List[ScheduledRequest]:
 self.time += 1.0
 batch = []
 remaining_tokens = self.token_budget
 
 # 1. 继续运行 running 请求（decode，每个消耗 1 token）
 for req_id in list(self.running.keys()):
 req = self.running[req_id]
 if req.status == RequestStatus.FINISHED:
 self.memory.free(req.required_kv_blocks)
 del self.running[req_id]
 elif remaining_tokens >= 1 and len(batch) < self.max_num_seqs:
 batch.append(req)
 remaining_tokens -= 1
 
 # 2. 从 waiting 中加入新请求
 still_waiting = []
 while self.waiting and remaining_tokens > 0 and len(batch) < self.max_num_seqs:
 neg_priority, submit_time, req = heapq.heappop(self.waiting)
 
 # 检查等待超时
 if self.time - submit_time > self.max_waiting_time:
 req.status = RequestStatus.TIMEOUT
 print(f"Request {req.req_id} timed out")
 continue
 
 # 检查 token budget
 if req.prompt_len > remaining_tokens:
 still_waiting.append((neg_priority, submit_time, req))
 continue
 
 # 检查显存预算（高优先级可突破预留）
 available_blocks = self.memory.total_blocks - self.memory.used_blocks
 if req.priority > 0:
 available_blocks += self.reserved_blocks
 
 if req.required_kv_blocks <= available_blocks:
 self.memory.allocate(req.required_kv_blocks)
 req.status = RequestStatus.RUNNING
 req.start_time = self.time
 self.running[req.req_id] = req
 batch.append(req)
 remaining_tokens -= req.prompt_len
 elif self.enable_preemption:
 # 尝试抢占低优先级 running 请求
 victim = self._select_victim(req)
 if victim:
 self._preempt(victim)
 self.memory.allocate(req.required_kv_blocks)
 req.status = RequestStatus.RUNNING
 self.running[req.req_id] = req
 batch.append(req)
 remaining_tokens -= req.prompt_len
 else:
 still_waiting.append((neg_priority, submit_time, req))
 else:
 still_waiting.append((neg_priority, submit_time, req))
 
 # 恢复仍在等待的请求
 for item in still_waiting:
 heapq.heappush(self.waiting, item)
 
 return batch
 
 def _select_victim(self, new_req: ScheduledRequest) -> Optional[ScheduledRequest]:
 """选择被抢占的最低优先级 running 请求"""
 victims = [req for req in self.running.values()
 if req.priority < new_req.priority]
 if not victims:
 return None
 return min(victims, key=lambda r: (r.priority, r.start_time or 0))
 
 def _preempt(self, victim: ScheduledRequest):
 """抢占请求（简化：直接释放资源）"""
 print(f"Preempting request {victim.req_id} (priority={victim.priority})")
 self.memory.free(victim.required_kv_blocks)
 victim.status = RequestStatus.SWAPPED
 self.swapped.append(victim)
 del self.running[victim.req_id]
 
 def finish_request(self, req_id: int):
 if req_id in self.running:
 self.running[req_id].status = RequestStatus.FINISHED

def main():
 scheduler = FullScheduler(token_budget=50, max_num_seqs=4,
 max_waiting_time=10.0, enable_preemption=True)
 
 # 提交 6 个请求，优先级不同
 for i in range(6):
 req = ScheduledRequest(
 req_id=i,
 prompt_len=10 + i * 2,
 priority=i % 3, # 0, 1, 2, 0, 1, 2
 max_new_tokens=5,
 required_kv_blocks=4 + i
 )
 scheduler.submit(req)
 
 # 模拟 5 轮调度
 for round in range(5):
 batch = scheduler.schedule()
 print(f"\nRound {round}: {scheduler.memory}")
 print(f" Batch: [{', '.join(f'R{req.req_id}(p={req.priority})' for req in batch)}]")
 
 # 模拟完成一些请求
 if round == 2:
 scheduler.finish_request(0)
 scheduler.finish_request(1)

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python full_scheduler.py

# 预期输出
# Round 0: MemoryBudget(...)
# Batch: [R5(p=2), R2(p=2), R4(p=1), R1(p=1)]
# ...
```

#### 练习题

**练习1（基础）**：修改抢占策略为 swap（不释放资源，而是标记为 swapped）。

**练习2（进阶）**：实现公平性机制：低优先级请求等待时间过长时提升优先级。

**练习3（综合）**：将 `FullScheduler` 集成到 Mini 引擎 v1 中。

---

### 今日面试题

**面试题1**：设计一个完整的 LLM 推理调度器，需要考虑哪些因素？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. **Batching 策略**：Dynamic 还是 Continuous
2. **优先级调度**：多租户 SLA 保障
3. **资源预算**：token budget、max seqs、显存预算
4. **超时控制**：等待超时、执行超时
5. **抢占与换出**：显存不足时如何处理
6. **公平性**：避免低优先级饥饿
7. **调度频率**：每轮 iteration 都调度还是固定间隔
8. **性能目标**：吞吐优先还是延迟优先

**面试题2**：调度器中的抢占策略如何选择被抢占的请求？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- 选择原则：
 1. **优先级最低**：优先保障高优先级请求
 2. **剩余工作最少**：换出成本低
 3. **最近最少使用**：避免频繁抢占同一个请求
 4. **资源占用最多**：释放更多资源
- 实际系统中通常是多因素综合打分
- 抢占后处理：recompute（丢弃 KV Cache）或 swap（换出到 CPU）

---

### 今日自测清单

- [ ] 能列出完整调度器的 6 个功能
- [ ] `full_scheduler.py` 运行成功，支持优先级和抢占
- [ ] 理解超时控制和资源预算的实现
- [ ] 能解释抢占策略的选择原则

---

## Day 45（周三）：SGLang / LightLLM 高级特性

> **今日目标**：了解 SGLang 的 speculative decoding、LightLLM 的 chunked prefill 和其他先进特性（prefix caching），评估哪些特性值得集成到 Mini 引擎。
> **面试考察度**：⭐⭐⭐⭐ 高频，高级特性是面试中的加分项

---

### 学习任务1：Speculative Decoding（45分钟）

#### 阅读内容
- **论文**："Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., NeurIPS 2022)
- **地址**：https://arxiv.org/abs/2211.17192
- **博客**：SGLang 文档中关于 speculative decoding 的部分
- **重点**：
 - 基本原理：小模型（draft）预测，大模型（target）验证
 - 为什么能加速
 - 适用场景和限制

#### 核心概念笔记

**1. 基本原理**

```
传统 Decode：
 每步：输入 1 个 token → 大模型 forward → 输出 1 个 token
 缺点：大模型每次只处理 1 个 token，无法充分利用 GPU

Speculative Decoding：
 1. 小模型（draft model）连续生成 k 个候选 tokens
 2. 大模型（target model）一次验证这 k 个 tokens
 3. 接受匹配的 tokens，从第一个不匹配处重新采样

效果：
 - 如果 draft 质量高，每步可接受多个 tokens
 - 大模型 batch 变大，利用率提高
 - 保持输出分布不变（与原始大模型一致）
```

**2. 加速原理**

```
假设：
 - draft model 生成 1 个 token 的时间 = t_d
 - target model 验证 k 个 token 的时间 ≈ T_target（基本固定）
 - 平均接受率 = α

传统每 token 时间 ≈ T_target
Speculative 每 token 时间 ≈ (k × t_d + T_target) / (k × α)

当 draft 很快且 α 高时，加速明显。
```

**3. 适用场景和限制**

```
适用：
 - 大模型 decode 是瓶颈
 - 有合适的 draft model（如小版自身、n-gram 模型）
 - 对延迟敏感

限制：
 - 需要额外内存放 draft model
 - draft 质量低时可能无加速甚至变慢
 - 实现复杂，需要仔细对齐分布
```

---

### 学习任务2：Chunked Prefill 与 Prefix Caching（45分钟）

#### Chunked Prefill

```
问题：
 - 长 prompt 的 prefill 一次性处理大量 tokens
 - 占用大量 token budget 和显存
 - 阻塞同 batch 的 decode 请求

Chunked Prefill：
 - 将长 prompt 分成多个 chunk
 - 每个 chunk 与 decode 请求一起执行
 - 逐步完成 prefill，同时不中断 decode

收益：
 - 平滑 decode latency
 - 提高 batch 利用率
```

#### Prefix Caching

```
问题：
 - 多个请求共享相同 prefix（如系统提示、多轮对话历史）
 - 每次都要重新计算 prefix 的 KV Cache

Prefix Caching：
 - 缓存常见 prefix 的 KV Cache
 - 新请求匹配到缓存 prefix 时，直接复用

收益：
 - 降低 TTFT
 - 减少重复计算
 - 特别适合多轮对话和模板化请求
```

---

### 学习任务3：特性收益评估（15分钟）

| 特性 | 收益 | 实现复杂度 | 是否值得集成 |
|------|------|-----------|------------|
| Speculative Decoding | 降低 TBT 2-3x | 高 | 高价值但复杂，可选 |
| Chunked Prefill | 平滑 latency | 中 | 推荐集成 |
| Prefix Caching | 降低 TTFT | 中 | 推荐集成 |
| 量化 KV Cache | 降低显存 | 中 | 推荐集成 |
| CUDA Graph | 降低 launch overhead | 中 | 推荐集成 |

### 晚间任务：可行性分析报告（1小时）

#### 报告模板

```markdown
# Mini 引擎高级特性评估

## Speculative Decoding
- 收益: 2-3x TBT 降低
- 复杂度: 高
- 是否集成: Phase 2

## Chunked Prefill
- 收益: 平滑 decode latency
- 复杂度: 中
- 是否集成: Phase 1

## Prefix Caching
- 收益: 降低 TTFT
- 复杂度: 中
- 是否集成: Phase 1

## 优先级
1. Chunked Prefill
2. Prefix Caching
3. CUDA Graph
4. Speculative Decoding
```

#### 练习题

**练习1（基础）**：解释 speculative decoding 为什么能保持输出分布不变。

**练习2（进阶）**：设计 prefix caching 的 key：如何快速判断两个 prompt 前缀是否相同？

**练习3（综合）**：评估在你的目标场景下，哪 2-3 个高级特性最值得优先实现。

---

### 今日面试题

**面试题1**：什么是 Speculative Decoding？它为什么能加速 LLM 推理？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **原理**：小模型（draft）快速生成 k 个候选 tokens，大模型（target）一次验证这 k 个 tokens
- **加速原因**：
 - 小模型生成速度快
 - 大模型一次验证多个 tokens，提高 batch 利用率
 - 如果 draft 质量高，每步可接受多个 tokens
- **保持分布不变**：通过特殊的接受/拒绝采样策略，确保最终输出分布与大模型自回归采样一致
- **适用场景**：decode 延迟敏感、有合适 draft model

**面试题2**：Chunked Prefill 和 Prefix Caching 分别解决了什么问题？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Chunked Prefill**：
 - 解决长 prompt prefill 阻塞 decode 的问题
 - 将长 prefill 拆分成多个 chunk，与 decode 交错执行
 - 效果：平滑 decode latency，避免单点延迟尖峰
- **Prefix Caching**：
 - 解决重复 prefix 的 KV Cache 重复计算问题
 - 缓存常见 prefix 的 KV Cache，新请求匹配时复用
 - 效果：降低 TTFT，特别适合系统提示、多轮对话

---

### 今日自测清单

- [ ] 理解 Speculative Decoding 的原理和保持分布不变的方法
- [ ] 理解 Chunked Prefill 的收益
- [ ] 理解 Prefix Caching 的应用场景
- [ ] 能评估各高级特性的收益和复杂度

---

## Day 46（周四）：整合全部自定义 Kernel

> **今日目标**：将 Week 2-4 手写的 GEMM、FlashAttention、Softmax、LayerNorm 接入 Mini 引擎，替换 PyTorch 对应算子。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，Kernel 集成是 Infra 工程师的核心能力

---

### 学习任务1：集成策略（1小时）

#### 替换清单

| PyTorch 算子 | 自定义 Kernel | 说明 |
|-------------|--------------|------|
| `torch.matmul` (QKV GEMM) | Register Blocking GEMM / cuBLAS | 大 GEMM 仍可用 cuBLAS |
| `F.softmax` | `softmax_kernel` | Row-wise softmax |
| `F.layer_norm` | `layernorm_kernel` | LayerNorm |
| `torch.matmul` (Attention) | `flashAttentionForward` | FlashAttention |

#### 集成方式

```
方式1：PyTorch C++ Extension
 - 最常用
 - 写 .cpp wrapper + .cu kernel
 - 用 load_inline 或 setup.py 编译

方式2：Triton
 - Python 写 kernel
 - torch.compile 自动集成
 - 适合快速迭代

Mini 系统选择方式1，与 Week 4 Day 26 一致。
```

#### 精度与性能

```
集成顺序：
 1. 先替换单个算子，验证正确性
 2. 再替换多个算子，验证端到端
 3. 最后做性能对比

注意点：
 - 精度：FP32 reduce 保持稳定
 - 性能：教学版 kernel 可能比 PyTorch 慢，需要逐步优化
 - 内存：自定义 kernel 的 tensor 分配要与 PyTorch 一致
```

### 学习任务2：Kernel 封装接口设计（30分钟）

```cpp
// ops.cpp
#include <torch/extension.h>

// GEMM: C = alpha * A * B + beta * C
at::Tensor gemm_forward(at::Tensor A, at::Tensor B, float alpha=1.0f, float beta=0.0f);

// FlashAttention
at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V);

// Softmax
at::Tensor softmax_forward(at::Tensor input);

// LayerNorm
at::Tensor layernorm_forward(at::Tensor input, at::Tensor gamma, at::Tensor beta, double eps=1e-5);
```

---

### 晚间编程任务：Kernel 集成模块（1.5小时）

#### 完整代码

```python
# custom_ops_module.py —— 自定义 Kernel 封装模块
# 运行命令: python custom_ops_module.py
# 依赖: 需要对应的 .cu 和 .cpp 文件

import torch
from torch.utils.cpp_extension import load_inline

# 读取 kernel 源文件
# 注意：实际项目中应将路径调整为对应周的位置
with open("softmax_layernorm.cu", "r") as f:
 softmax_layernorm_src = f.read()

with open("flash_attention_v2.cu", "r") as f:
 flash_attention_src = f.read()

cpp_src = """
#include <torch/extension.h>
at::Tensor softmax_forward(at::Tensor input);
at::Tensor layernorm_forward(at::Tensor input, at::Tensor gamma, at::Tensor beta, double eps);
at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V);
"""

# 动态编译
custom_ops = load_inline(
 name="custom_ops",
 cpp_sources=cpp_src,
 cuda_sources=softmax_layernorm_src + flash_attention_src,
 functions=["softmax_forward", "layernorm_forward", "flash_attention_forward"],
 verbose=True,
 extra_cuda_cflags=["-O3", "-arch=sm_120"],
)

class CustomKernelTransformerLayer(torch.nn.Module):
 """使用自定义 Kernel 的 Transformer Layer（简化版）"""
 def __init__(self, d_model=512, n_heads=8, d_ff=2048):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.qkv = torch.nn.Linear(d_model, 3 * d_model)
 self.out = torch.nn.Linear(d_model, d_model)
 self.norm1_weight = torch.nn.Parameter(torch.ones(d_model))
 self.norm1_bias = torch.nn.Parameter(torch.zeros(d_model))
 self.norm2_weight = torch.nn.Parameter(torch.ones(d_model))
 self.norm2_bias = torch.nn.Parameter(torch.zeros(d_model))
 self.ffn = torch.nn.Sequential(
 torch.nn.Linear(d_model, d_ff),
 torch.nn.GELU(),
 torch.nn.Linear(d_ff, d_model),
 )

 def forward(self, x, use_custom_ops=True):
 B, N, D = x.shape

 # LayerNorm1 + QKV
 if use_custom_ops:
 x_flat = x.reshape(B * N, D)
 x_norm = custom_ops.layernorm_forward(x_flat, self.norm1_weight, self.norm1_bias, 1e-5)
 x_norm = x_norm.reshape(B, N, D)
 else:
 x_norm = torch.nn.functional.layer_norm(x, (D,), self.norm1_weight, self.norm1_bias, 1e-5)

 qkv = self.qkv(x_norm)
 qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
 q, k, v = qkv[0], qkv[1], qkv[2]

 # FlashAttention
 if use_custom_ops:
 # 假设 q,k,v 是 FP32
 attn_out = custom_ops.flash_attention_forward(q, k, v)
 else:
 scale = self.d_head ** -0.5
 attn = torch.matmul(q, k.transpose(-2, -1)) * scale
 attn = torch.nn.functional.softmax(attn, dim=-1)
 attn_out = torch.matmul(attn, v)

 attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
 x = x + self.out(attn_out)

 # LayerNorm2 + FFN
 if use_custom_ops:
 x_flat = x.reshape(B * N, D)
 x_norm = custom_ops.layernorm_forward(x_flat, self.norm2_weight, self.norm2_bias, 1e-5)
 x_norm = x_norm.reshape(B, N, D)
 else:
 x_norm = torch.nn.functional.layer_norm(x, (D,), self.norm2_weight, self.norm2_bias, 1e-5)

 x = x + self.ffn(x_norm)
 return x

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}")

 layer = CustomKernelTransformerLayer(d_model=512, n_heads=8).to(device).eval()
 x = torch.randn(2, 128, 512, device=device)

 with torch.no_grad():
 out_custom = layer(x, use_custom_ops=True)
 out_pytorch = layer(x, use_custom_ops=False)

 max_diff = (out_custom - out_pytorch).abs().max().item()
 print(f"Max diff (custom vs pytorch): {max_diff:.2e}")
 print("PASS" if max_diff < 1e-3 else "FAIL")

if __name__ == "__main__":
 main()
```

#### 练习题

**练习1（基础）**：单独测试 `softmax_forward` 和 `layernorm_forward` 的正确性。

**练习2（进阶）**：将 register blocking GEMM 也接入，替换小尺寸的 QKV GEMM。

**练习3（综合）**：在 Mini 引擎 v1 中使用自定义 kernel 替换标准 Attention 和 LayerNorm。

---

### 今日面试题

**面试题1**：如何将自定义 CUDA kernel 集成到 PyTorch 推理引擎中？需要注意什么？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. 写 CUDA kernel（`.cu`）和 C++ wrapper（`.cpp`）
2. 使用 `torch.utils.cpp_extension.load_inline` 或 `setup.py` 编译
3. 在 wrapper 中：
 - 用 `at::Tensor` 接收张量
 - 用 `data_ptr<float>()` 获取裸指针
 - 用 `at::cuda::getCurrentCUDAStream()` 获取当前 stream
 - 用 `at::empty_like(input)` 分配输出
1. 注意点：
 - stream 一致性：确保 kernel 在正确 stream 上执行
 - 精度：FP32 reduce 保持稳定
 - 内存布局：与 PyTorch 对齐
 - 边界处理：非对齐尺寸

**面试题2**：自定义 Kernel 集成后，如何验证正确性和性能？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **正确性**：
 1. 单算子对比 PyTorch 实现，误差 < 阈值
 2. 多算子组合后对比端到端输出
 3. 不同尺寸和边界条件测试
- **性能**：
 1. 用 `cudaEvent` 或 `torch.cuda.Event` 测 latency
 2. 用 `nsys` 看时间线
 3. 用 `ncu` 分析 kernel 瓶颈
 4. 对比 throughput-latency 曲线

---

### 今日自测清单

- [ ] 能列出要替换的 PyTorch 算子和对应的自定义 kernel
- [ ] 理解 PyTorch C++ Extension 的集成流程
- [ ] 能设计 kernel 封装接口
- [ ] 理解集成时的 stream、精度、内存布局注意事项
- [ ] 能解释如何验证正确性和性能

---

## Day 47（周五）：系统联调

> **今日目标**：将 KV Cache、Batching、Scheduler、自定义 Kernel 全部联调，进行端到端多请求推理测试和长时间稳定性测试。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，系统联调能力是 Infra 工程师的分水岭

---

### 学习任务1：联调 checklist（1小时）

#### 联调顺序

```
Step 1: 单请求正确性
 - 使用 Mini 引擎 v0 验证基础 forward 正确

Step 2: 多请求并发正确性
 - 使用 Mini 引擎 v1 验证多请求结果正确
 - 检查请求生命周期管理

Step 3: KV Cache 一致性
 - with cache vs without cache 输出一致
 - 多轮对话 cache 复用正确

Step 4: Scheduler 正确性
 - 优先级调度正确
 - 超时取消正确
 - 资源预算不突破

Step 5: 自定义 Kernel 集成
 - 单算子替换正确
 - 多算子组合正确
 - 端到端性能对比

Step 6: 稳定性测试
 - 连续处理 1000+ 请求
 - 内存使用稳定
 - 异常情况处理（OOM、非法输入）
```

#### 常见问题与排查

| 问题 | 可能原因 | 排查方法 |
|------|---------|---------|
| 结果不一致 | Kernel 边界处理错误 | 小尺寸对比 |
| 内存泄漏 | KV Cache 未释放 | 监控显存 |
| 请求卡住 | Scheduler 死锁 | 打印调度日志 |
| 显存 OOM | batch 过大 | 检查 budget |
| 性能下降 | kernel 未优化 | ncu profiling |

### 学习任务2：稳定性测试脚本（30分钟）

```python
# stability_test.py —— Mini 系统稳定性测试
# 运行命令: python stability_test.py

import time
import torch
from mini_engine_v1 import MiniEngineV1
from mini_engine_v0 import MiniLLM, MiniTokenizer

def stability_test(engine, num_requests=1000, max_new_tokens=10):
 """连续处理大量请求，监控正确性和内存"""
 prompts = [
 "hello world",
 "this is a test prompt",
 "short",
 "another example prompt for testing",
 "a medium length prompt here for batching test",
 ]

 success_count = 0
 fail_count = 0
 start_time = time.time()

 for i in range(num_requests):
 prompt = prompts[i % len(prompts)]
 try:
 future = engine.submit(prompt, max_new_tokens)
 result = future.result(timeout=30)
 success_count += 1
 if i % 100 == 0:
 print(f"Processed {i} requests, success={success_count}")
 except Exception as e:
 fail_count += 1
 print(f"Request {i} failed: {e}")

 total_time = time.time() - start_time
 print(f"\n=== Stability Test Result ===")
 print(f"Total requests: {num_requests}")
 print(f"Success: {success_count}")
 print(f"Fail: {fail_count}")
 print(f"Total time: {total_time:.3f}s")
 print(f"Throughput: {num_requests / total_time:.2f} req/s")

 # 显存监控
 if torch.cuda.is_available():
 torch.cuda.synchronize()
 allocated = torch.cuda.memory_allocated() / 1024 / 1024
 reserved = torch.cuda.memory_reserved() / 1024 / 1024
 print(f"GPU memory allocated: {allocated:.2f} MB")
 print(f"GPU memory reserved: {reserved:.2f} MB")

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}\n")

 model = MiniLLM(vocab_size=1000, d_model=512, n_heads=8, n_layers=2)
 tokenizer = MiniTokenizer(vocab_size=1000)
 engine = MiniEngineV1(model, tokenizer, max_token_budget=200, max_num_seqs=8, device=device)

 stability_test(engine, num_requests=500, max_new_tokens=5)

 engine.shutdown()

if __name__ == "__main__":
 main()
```

#### 练习题

**练习1（基础）**：在稳定性测试中加入异常输入（空 prompt、超长 prompt）。

**练习2（进阶）**：监控每 100 个请求的显存增长，绘制内存使用曲线。

**练习3（综合）**：将自定义 kernel 接入后重新做稳定性测试，对比内存和性能。

---

### 今日面试题

**面试题1**：系统联调时，如何确保多请求并发的正确性？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. **分层验证**：
 - 单请求正确性
 - 多请求并发正确性
 - 带 KV Cache 的正确性
 - 带 Scheduler 的正确性
1. **关键点**：
 - 每个请求的 KV Cache 隔离
 - 请求生命周期状态正确转换
 - Scheduler 不丢失请求
 - 异步结果正确返回给对应请求
1. **测试方法**：
 - 与 PyTorch eager 版对比输出
 - 长时间稳定性测试
 - 边界条件和异常输入测试

**面试题2**：如何做推理系统的稳定性测试？需要关注哪些指标？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **测试规模**：连续处理 1000+、10000+ 请求
2. **关键指标**：
 - 成功率
 - 平均延迟 / P99 延迟
 - 吞吐
 - 显存使用是否持续增长（内存泄漏）
 - CPU 使用率
1. **异常情况**：
 - OOM 处理
 - 非法输入
 - 超时取消
 - 请求突然大量涌入
1. **工具**：nsys、ncu、torch profiler、自定义监控

---

### 今日自测清单

- [ ] 能列出系统联调的 6 个步骤
- [ ] 理解常见问题的排查方法
- [ ] 完成稳定性测试脚本
- [ ] 系统能连续处理 500+ 请求
- [ ] 显存使用稳定，无明显增长
- [ ] 能处理异常输入和超时

---

## Day 48（周六）：全链路 Profiling

> **今日目标**：对完整 Mini AI Infra 系统进行全链路 profiling，定位系统级瓶颈，与 vLLM 进行同条件对比。
> **时间分配**：6小时全天投入（nsys/ncu 采集2h + 瓶颈分析2h + vLLM 对比2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"系统级性能分析"是 Infra 优化的核心能力

---

### 任务1：系统级 Profiling 设计（2小时）

#### 分析维度

```
系统级瓶颈来源：
 1. 调度开销（Scheduler CPU 时间）
 2. 内存分配开销（KV Cache 分配、tensor 分配）
 3. Kernel launch overhead
 4. 计算瓶颈（GEMM、Attention、FFN）
 5. 内存瓶颈（KV Cache 读取、attention）
 6. Python GIL / 多线程竞争
```

#### Profiling 工具组合

```
Nsight Systems (nsys):
 - 看系统级时间线
 - 识别 kernel 间隙、CPU-GPU 同步、stream 利用率

Nsight Compute (ncu):
 - 看关键 kernel 的 SM/DRAM throughput
 - 识别 compute-bound 还是 memory-bound

PyTorch Profiler:
 - 看算子级时间分解
 - 方便定位 PyTorch 层的 overhead

自定义计时：
 - scheduler 时间
 - memory allocation 时间
 - result callback 时间
```

---

### 任务2：全链路 Profiling 脚本（2小时）

#### 完整代码

```python
# full_chain_profile.py —— Mini 系统全链路 Profiling
# 运行命令: nsys profile -o mini_system_profile python full_chain_profile.py

import time
import torch
from mini_engine_v1 import MiniEngineV1
from mini_engine_v0 import MiniLLM, MiniTokenizer

class ProfileMiniSystem:
 def __init__(self, engine):
 self.engine = engine
 self.metrics = {
 'submit_times': [],
 'schedule_times': [],
 'forward_times': [],
 'result_times': [],
 }

 def profile_single_request(self, prompt, max_new_tokens=10):
 t0 = time.perf_counter()
 future = self.engine.submit(prompt, max_new_tokens)
 t1 = time.perf_counter()
 result = future.result()
 t2 = time.perf_counter()

 self.metrics['submit_times'].append(t1 - t0)
 self.metrics['result_times'].append(t2 - t1)
 return result

 def run(self, num_requests=50, max_new_tokens=10):
 prompts = [
 "hello world",
 "this is a test prompt",
 "short",
 "another example prompt for testing batching and scheduling",
 ]

 # 预热
 for _ in range(3):
 self.profile_single_request(prompts[0], 3)

 torch.cuda.synchronize()
 start = time.perf_counter()

 futures = []
 for i in range(num_requests):
 prompt = prompts[i % len(prompts)]
 future = self.engine.submit(prompt, max_new_tokens)
 futures.append(future)

 # 等待所有完成
 results = []
 for future in futures:
 results.append(future.result())

 torch.cuda.synchronize()
 total_time = time.perf_counter() - start

 total_tokens = num_requests * max_new_tokens
 throughput = total_tokens / total_time

 print(f"=== Full Chain Profile ===")
 print(f"Total requests: {num_requests}")
 print(f"Total time: {total_time:.3f}s")
 print(f"Throughput: {throughput:.2f} tokens/s")
 print(f"Avg submit time: {sum(self.metrics['submit_times'])/len(self.metrics['submit_times'])*1000:.3f} ms")
 print(f"Avg result time: {sum(self.metrics['result_times'])/len(self.metrics['result_times'])*1000:.3f} ms")

 if torch.cuda.is_available():
 print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
 print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")

 return throughput, total_time

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}\n")

 model = MiniLLM(vocab_size=1000, d_model=512, n_heads=8, n_layers=4)
 tokenizer = MiniTokenizer(vocab_size=1000)
 engine = MiniEngineV1(model, tokenizer, max_token_budget=200, max_num_seqs=8, device=device)

 profiler = ProfileMiniSystem(engine)
 profiler.run(num_requests=50, max_new_tokens=10)

 engine.shutdown()

if __name__ == "__main__":
 main()
```

#### 采集命令

```bash
# Nsight Systems
nsys profile -o mini_system_profile --trace=cuda,nvtx python full_chain_profile.py

# 查看 kernel 统计
nsys stats -t cuda_gpu_kern_sum mini_system_profile.nsys-rep

# Nsight Compute（分析特定 kernel）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
 --kernel-name regex:"gemm|flash_attention|softmax|layernorm" \
 python full_chain_profile.py
```

---

### 任务3：瓶颈定位与 vLLM 对比（2小时）

#### 分析框架

```
Step 1: 用 nsys 看时间线
 - 找出 kernel 间隙最大的区域
 - 判断是调度 overhead 还是 launch overhead

Step 2: 用 ncu 分析 top kernel
 - 判断 compute-bound 还是 memory-bound
 - 与 PyTorch/cuBLAS 版本对比

Step 3: 用自定义计时拆分阶段
 - submit / schedule / forward / result 各占多少时间

Step 4: 与 vLLM 对比
 - 同模型、同 batch、同序列长度下
 - 对比 throughput 和 latency
```

#### 常见系统级瓶颈

| 瓶颈 | 表现 | 优化 |
|------|------|------|
| Python Scheduler CPU-bound | schedule 时间长 | C++ scheduler、简化调度逻辑 |
| Memory allocation | 频繁 cudaMalloc | 预分配 buffer、memory pool |
| Kernel launch overhead | kernel 间隙大 | CUDA Graph、kernel fusion |
| GIL contention | 多线程下 Python 层慢 | 减少 Python 层、使用多进程 |
| Copy between CPU-GPU | input/output 传输慢 | pinned memory、zero-copy |

### 今日面试题

**面试题1**：如何做 LLM 推理系统的全链路性能分析？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **系统级**：nsys 看时间线，找 kernel 间隙、stream 利用率、CPU-GPU 同步
2. **Kernel 级**：ncu 分析 top kernel 的 SM/DRAM throughput，判断 bound 类型
3. **算子级**：PyTorch profiler 看算子时间分解
4. **阶段拆分**：自定义计时测量 submit/schedule/forward/result
5. **对比分析**：与 vLLM/TensorRT-LLM 同条件对比

**面试题2**：你的 Mini 系统和 vLLM 的差距主要在哪里？如何缩小？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **常见差距**：
 1. **Kernel 优化**：vLLM 使用高度优化的 attention kernel（FlashAttention、PagedAttention），教学版 kernel 性能较低
 2. **Scheduler 性能**：vLLM 经过多年优化，调度逻辑更高效
 3. **Memory management**：vLLM 的 PagedAttention 内存利用率更高
 4. **CUDA Graph**：vLLM 支持 CUDA Graph 减少 launch overhead
 5. **Multi-GPU / TP / PP**：vLLM 支持分布式，Mini 系统是单 GPU
- **缩小差距的方法**：
 1. 使用官方 FlashAttention/PagedAttention kernel
 2. 将 scheduler 用 C++ 重写
 3. 实现 PagedAttention
 4. 引入 CUDA Graph
 5. 使用 torch.compile 自动优化

---

### 今日自测清单

- [ ] 完成全链路 profiling 脚本
- [ ] 用 nsys 采集系统时间线
- [ ] 用 ncu 分析关键 kernel
- [ ] 拆分 submit/schedule/forward/result 各阶段时间
- [ ] 识别系统 top3 瓶颈
- [ ] 能提出至少 3 个优化建议

---

## Day 49（周日）：代码重构与文档

> **今日目标**：整理代码结构，统一接口，添加 README 和架构图，完成 Week 7 总结。
> **时间分配**：6小时全天投入（代码重构2h + 文档编写2h + 面试复盘2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，项目文档和代码质量是面试考察点

---

### 任务1：代码重构（2小时）

#### 重构目标

```
1. 统一目录结构
2. 统一接口命名
3. 添加关键注释和 docstring
4. 消除重复代码
5. 增加类型注解
6. 添加基础单元测试
```

#### 建议目录结构

```
mini_ai_infra/
├── README.md
├── setup.py
├── requirements.txt
├── mini_infra/
│ ├── __init__.py
│ ├── engine.py # MiniEngineV1
│ ├── scheduler.py # FullScheduler
│ ├── request.py # InferenceRequest
│ ├── kv_cache.py # KVCacheManager
│ ├── model.py # MiniLLM
│ ├── tokenizer.py # MiniTokenizer
│ └── kernels/
│ ├── __init__.py
│ ├── gemm_kernel.py
│ ├── flash_attention_kernel.py
│ ├── softmax_kernel.py
│ └── layernorm_kernel.py
├── tests/
│ ├── test_engine.py
│ ├── test_scheduler.py
│ ├── test_kv_cache.py
│ └── test_kernels.py
├── benchmarks/
│ ├── benchmark_engine.py
│ └── benchmark_results/
└── docs/
 ├── architecture.md
 ├── performance_report.md
 └── api.md
```

#### 重构要点

```python
# 统一接口示例
class InferenceEngine:
 """Mini AI Infra 推理引擎"""
 
 def __init__(self, model_config, scheduler_config, device="cuda"):
 ...
 
 def submit(self, prompt: str, max_new_tokens: int = 20,
 priority: int = 0, timeout: Optional[float] = None) -> Future[str]:
 """提交请求，返回 Future"""
 ...
 
 def shutdown(self):
 """关闭引擎"""
 ...
```

---

### 任务2：文档编写（2小时）

#### README 模板

```markdown
# Mini AI Infra

一个用于学习 AI Infra 的迷你推理系统，支持：
- 单请求 / 多请求并发
- KV Cache 加速 Decode
- Continuous Batching
- 优先级调度
- 自定义 CUDA Kernel（GEMM、FlashAttention、Softmax、LayerNorm）

## 快速开始

```bash
pip install -r requirements.txt
python examples/single_request.py
python examples/multi_request.py
```

## 架构

[架构图]

## 性能

[性能报告链接]

## 测试

```bash
python -m pytest tests/
```
```

#### 架构文档

```markdown
# Mini AI Infra 架构

## 模块划分

1. **Engine**：对外接口，管理请求生命周期
2. **Scheduler**：Continuous Batching + 优先级调度
3. **Worker**：执行模型 forward
4. **KV Cache Manager**：管理 KV Cache 分配和复用
5. **Model**：Mini transformer 模型
6. **Kernel**：自定义 CUDA kernel

## 数据流

Request → Queue → Scheduler → Worker → Sampler → Result

## 关键设计决策

- 使用 Python 实现调度器，便于教学
- 使用 PyTorch C++ Extension 集成自定义 kernel
- 单 GPU 实现，后续可扩展多 GPU
```

---

### 任务3：面试复盘（2小时）

#### 本周核心面试题回顾

1. 多请求并发设计
2. Future/Callback/Streaming 区别
3. 完整调度器设计
4. 抢占策略选择
5. Speculative Decoding 原理
6. Chunked Prefill 和 Prefix Caching
7. 自定义 kernel 集成流程
8. 系统联调步骤
9. 稳定性测试方法
10. 全链路 profiling 方法
11. Mini 系统与 vLLM 的差距
12. 代码重构目标
13. 架构文档编写

---

### 今日面试题

**面试题1**：如何设计一个可维护的 LLM 推理系统代码结构？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **模块化**：Engine、Scheduler、Worker、KV Cache、Model、Kernel 分离
2. **统一接口**：明确的 submit/result/shutdown API
3. **配置驱动**：模型配置、调度配置独立
4. **测试覆盖**：单元测试 + 集成测试 + 性能测试
5. **文档完善**：README、架构图、API 文档、性能报告
6. **错误处理**：明确的异常类型和日志
7. **可观测性**：metrics、profiling 接口

**面试题2**：你的 Mini AI Infra 项目最大的技术难点是什么？如何解决？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **难点1：Continuous Batching 的正确性**
 - 每轮重新构建 batch，需要精确管理每个请求的状态和 KV Cache
 - 解决：清晰的状态机、单元测试、长时间稳定性测试
- **难点2：自定义 Kernel 与 PyTorch 的集成**
 - stream、内存布局、精度问题
 - 解决：逐步替换、充分对比验证
- **难点3：调度器性能**
 - Python scheduler 可能成为瓶颈
 - 解决：简化调度逻辑、预分配 buffer、后续可用 C++ 重写
- **难点4：系统联调**
 - 组件多，边界问题复杂
 - 解决：分层验证、逐步集成、充分测试

---

### 今日自测清单

- [ ] 代码重构完成，目录结构清晰
- [ ] 接口统一，有类型注解和 docstring
- [ ] 编写 README
- [ ] 编写架构文档
- [ ] 编写性能报告
- [ ] 添加基础测试
- [ ] 完成本周面试题复盘
- [ ] 规划 Week 8（项目打磨 + 面试准备）

---

## 附录A：第7周面试题汇总

| 题号 | 题目 | 考察频率 | 相关天数 | 难度 |
|------|------|---------|---------|------|
| 1 | 多请求并发如何实现？线程安全问题？ | ⭐⭐⭐⭐ | Day 43 | 中 |
| 2 | Future/Callback/Streaming 区别？ | ⭐⭐⭐ | Day 43 | 中 |
| 3 | 完整调度器设计需要考虑什么？ | ⭐⭐⭐⭐⭐ | Day 44 | 高 |
| 4 | 抢占策略如何选择被抢占请求？ | ⭐⭐⭐⭐ | Day 44 | 中 |
| 5 | Speculative Decoding 原理？ | ⭐⭐⭐⭐ | Day 45 | 高 |
| 6 | Chunked Prefill 和 Prefix Caching？ | ⭐⭐⭐⭐ | Day 45 | 中 |
| 7 | 自定义 kernel 如何集成到 PyTorch？ | ⭐⭐⭐⭐⭐ | Day 46 | 中 |
| 8 | 系统联调有哪些步骤？ | ⭐⭐⭐⭐ | Day 47 | 中 |
| 9 | 稳定性测试关注哪些指标？ | ⭐⭐⭐⭐ | Day 47 | 中 |
| 10 | 全链路 profiling 怎么做？ | ⭐⭐⭐⭐ | Day 48 | 中 |
| 11 | Mini 系统与 vLLM 的差距在哪？ | ⭐⭐⭐⭐⭐ | Day 48 | 高 |
| 12 | 如何设计可维护的推理系统代码？ | ⭐⭐⭐⭐ | Day 49 | 中 |
| 13 | 项目最大技术难点是什么？ | ⭐⭐⭐⭐⭐ | Day 49 | 高 |

---

## 附录C：关键公式汇总

**1. 系统吞吐**
```
System Throughput = total_generated_tokens / total_time
```

**2. 系统延迟**
```
Avg E2E Latency = sum(end_time - submit_time) / num_requests
```

**3. GPU 利用率**
```
GPU Utilization = compute_time / (compute_time + gap_time)
```

**4. Memory Efficiency**
```
Memory Efficiency = used_kv_cache / allocated_kv_cache
```

---

## 附录D：系统优化速查表

| 问题 | 现象 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| 调度器 CPU 瓶颈 | schedule 时间占比高 | 自定义计时 | C++ scheduler、简化逻辑 |
| 内存分配慢 | cudaMalloc 频繁 | nsys | memory pool、预分配 |
| Kernel 间隙大 | launch overhead | nsys timeline | CUDA Graph、kernel fusion |
| 单算子慢 | ncu 指标低 | ncu | 优化 kernel、用官方实现 |
| 多请求结果错误 | 状态机混乱 | 单元测试 | 清晰状态管理、隔离 KV Cache |
| 内存持续增长 | OOM | 监控显存 | 确保 finished 请求释放资源 |
| Python GIL | 多线程下 CPU 高 | profile Python | 减少 Python 层、C++ 扩展 |
| 与 vLLM 差距大 | 吞吐显著低 | 同条件对比 | 使用官方 kernel、PagedAttention、CUDA Graph |

---

> 💡 **Week 7 总结**：本周我们完成了 Mini AI Infra 系统的整合。从多请求并发、完整调度器，到自定义 Kernel 集成、系统联调和全链路 profiling，最终形成了一个可运行、可测试、有文档的完整项目。最核心的收获是：系统整合不是简单堆砌组件，而是要在接口、状态、资源、性能四个层面做统一设计。Week 8 将进入项目打磨和面试准备，把学习成果转化为面试能力。
