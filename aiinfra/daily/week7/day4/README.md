## Day 4：Chunked Prefill 与 Prefix Caching 实操

### 🎯 目标

通过今天的学习，你将：

1. 理解 Chunked Prefill 的核心思想：将长 prompt 分块处理，与 decode 迭代混合调度<br>
2. 掌握 Prefix Caching 的实现原理：基于 block hash 的 KV cache 复用与 LRU 淘汰<br>
3. 理解两者如何协同工作以同时改善 TTFT 和 TBT<br>
4. 实现一个带 Prefix Caching 的简化推理引擎<br>
5. 能用实测数据量化 prefix caching 对共享 system prompt 场景的加速<br>
6. 能在面试中清晰阐述 chunked prefill 和 prefix caching 的适用场景与 trade-off<br>

> 💡 **为什么重要**：Day 4 学习了 TRT-LLM 的 chunked prefill 和 vLLM 的 prefix caching 概念，但只是模拟器。今天把它们真实实现到一个简化引擎中，是从"知道概念"到"能讲实现细节"的关键一步。面试中"如何优化多轮对话的推理延迟"是高频场景题。

---

### 学前导读：长 Prompt 与共享 System Prompt 的两个痛点

Day 4 的 chunked prefill simulator 和 Day 2 的 continuous batcher 解决了调度层面的问题，但有两个实际场景的痛点尚未覆盖：

**痛点 1：长 Prompt 阻塞 Decode**

当一个 4096-token 的 prompt 进入 prefill 时，它会占用大量 token budget，导致已在 decode 的请求 TBT 飙升。chunked prefill 将 4096 拆成 8 个 512-token 的 chunk，每个 chunk 只占 512 budget，与 decode 混合调度。

**痛点 2：重复计算共享 System Prompt**

多个请求共享同一个 system prompt（如 "You are a helpful assistant..."），但每个请求都重新计算这段 prefix 的 KV cache。prefix caching 通过 block hash 匹配，直接复用已缓存的 KV block，跳过 prefill。

| 优化 | 解决的问题 | TTFT 改善 | TBT 改善 | 实现复杂度 |
|------|-----------|----------|---------|-----------|
| Chunked Prefill | 长 prompt 阻塞 decode | ↓ 30-50% | ↓ 20-40% | 中 |
| Prefix Caching | 重复计算共享 prefix | ↓ 50-90% (命中时) | 不变 | 中 |
| 两者联合 | 上述两个 | ↓ 50-90% | ↓ 20-40% | 高 |

> 💡 **一句话总结**：chunked prefill 解决"长 prompt 挤占 decode"，prefix caching 解决"重复计算 prefix"。两者正交，可同时启用。

---

### 理论学习

#### 1.1 Chunked Prefill 深入

Day 4 已介绍了 chunked prefill 的基本概念，今天深入实现细节。

##### chunk_size 选择

| chunk_size | 优点 | 缺点 | 适用场景 |
|-----------|------|------|---------|
| 128 | TBT 波动最小 | 长 prompt 完成慢 | 对 TBT 敏感的在线服务 |
| 512 | 平衡 TTFT 和 TBT | — | 通用场景（vLLM 默认） |
| 2048 | TTFT 最低 | TBT 波动大 | 对 TTFT 敏感的批处理 |
| ∞（不切分） | 最简单 | 长 prompt 阻塞 | 短 prompt 场景 |

**关键约束**：`chunk_tokens + decode_tokens ≤ max_num_batched_tokens`（token budget）

##### 与 Continuous Batching 的集成

```python
# 每个 iteration 的调度决策：
def schedule_iteration():
    # 1. 保留 running 请求的 decode（每请求 1 token）
    decode_tokens = len(running_seqs)
    
    # 2. 计算剩余 token budget
    remaining = max_budget - decode_tokens
    
    # 3. 如果有 waiting 的 prefill 请求，切分 chunk
    if waiting_seqs and remaining > 0:
        chunk_size = min(remaining, waiting_seqs[0].remaining_prefill)
        admit_chunk(waiting_seqs[0], chunk_size)
    
    # 4. 如果还有剩余 budget，继续 decode
    # → decode 和 prefill chunk 在同一 batch 中执行
```

#### 1.2 Prefix Caching 深入

##### Block Hash 机制

```python
# 每个 KV cache block 对应一段 token sequence
# 用 token sequence 的 hash 作为 block 的唯一标识
block_hash = hash(token_ids[start : start + block_size])

# 请求进来时，逐 block 计算 hash，匹配已缓存的 block
for block in request.token_blocks:
    h = hash(block.tokens)
    if h in cache_pool:
        # 命中！直接引用，不需要重新 prefill
        request.kv_blocks.append(cache_pool[h])
    else:
        # 未命中，需要 prefill
        break  # prefix 匹配到此为止
```

##### LRU 淘汰策略

```python
class PrefixCache:
    def __init__(self, max_blocks):
        self.cache = OrderedDict()  # hash -> block_data
        self.max_blocks = max_blocks
    
    def get(self, h):
        if h in self.cache:
            self.cache.move_to_end(h)  # LRU: 移到末尾（最近使用）
            return self.cache[h]
        return None
    
    def put(self, h, block_data):
        if h not in self.cache:
            self.cache[h] = block_data
            if len(self.cache) > self.max_blocks:
                self.cache.popitem(last=False)  # 淘汰最久未使用
```

##### 命中率分析

| 场景 | Prefix 长度 | 命中率 | 节省计算 |
|------|-----------|--------|---------|
| 单轮对话 | 0 | 0% | 0% |
| 多轮对话（同一 session） | 历史 tokens | 80-95% | 60-80% |
| 共享 system prompt | system prompt 长度 | 90-99% | 30-50% |
| Few-shot（相同 examples） | examples 长度 | 80-90% | 40-60% |

##### RadixAttention：前缀树替代 block-hash（SGLang 方案）

上述 block-hash 方案（vLLM）有一个局限：**block 对齐损失**。若共享前缀长度不是 block_size 的整数倍，尾部不对齐的部分无法复用。SGLang 的 **RadixAttention** 用基数树（radix tree / 前缀树）替代 block-hash，解决对齐损失。

**数据结构对比**：

```
block-hash（vLLM）:
  每 16 token 算一个 hash，命中需严格对齐到 block 边界
  请求: [sys_prompt(100 token) | user_input]
         ← 6 个完整 block 命中(96 token) → 4 token 丢弃

RadixAttention（SGLang）:
  前缀树存任意长度的前缀，命中无对齐要求
  请求: [sys_prompt(100 token) | user_input]
         ← 100 token 全部命中（前缀树精确匹配）→
```

| 维度 | block-hash（vLLM） | RadixAttention（SGLang） |
|------|-------------------|------------------------|
| 数据结构 | block 级哈希表 | 前缀树（radix tree） |
| 匹配粒度 | block（如 16 token） | 任意前缀长度 |
| 对齐损失 | 有（尾部不对齐丢弃） | 无（精确匹配） |
| 多轮对话 | 每轮重新哈希 | 前缀树自动增量 |
| 共享前缀多 | 命中率高但浪费尾部 | 命中率更高且无浪费 |
| 实现复杂度 | 简单（哈希表） | 中等（树 + 引用计数） |

> 💡 **何时 RadixAttention 更优**：共享前缀多且长时（多轮对话、few-shot batch、共享 system prompt）。SGLang 在这类场景的 KV Cache 命中率比 vLLM 高 5-15%（无对齐损失）。vLLM 0.5+ 也在改进 prefix caching 的对齐处理，但数据结构仍是 block-hash。

> 📖 品读论文：SGLang 论文（RadixAttention 的原论文），vLLM prefix caching RFC

#### 1.3 两者协同

chunked prefill 和 prefix caching 是正交的：

```
请求到达 → prefix caching 匹配 → 剩余未命中部分 → chunked prefill 切分
```

例如：system prompt 1024 tokens + user input 3072 tokens
- prefix caching 命中 1024 tokens（跳过 prefill）
- 剩余 3072 tokens 用 chunked prefill，chunk_size=512，分 6 个 chunk

#### 1.4 vLLM 中的实现

vLLM v0.5+ 默认启用 `--enable-prefix-caching`：
- `BlockPool` 管理物理 block，每个 block 有 `ref_count` 和 `hash`
- `SequenceGroup` 计算其 token sequence 的 block hash
- 调度时优先匹配已缓存的 block
- 使用 copy-on-write：多个请求共享同一物理 block，写时才复制

---

### Coding 任务：实现带 Prefix Caching 的推理引擎

#### 任务 1：创建 `prefix_cache_engine.py`

完整代码见 [kernels/prefix_cache_engine.py](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/week7/day4/kernels/prefix_cache_engine.py)。

代码实现：
- `BlockPool`：管理 KV cache blocks，支持 allocate/free/hash 匹配
- `PrefixCache`：基于 `OrderedDict` 的 LRU 缓存，block hash → physical block
- `Request`：含 token_ids、已匹配的 prefix blocks、待 prefill 的 tokens
- `PrefixCacheEngine`：调度器，每个请求先做 prefix 匹配，剩余部分 prefill

#### 任务 2：运行与验证

```bash
python3 kernels/prefix_cache_engine.py
```

预期输出：

```text
=== Prefix Caching Engine Demo ===
System prompt: 64 tokens, User input: 128 tokens

--- Scenario 1: No prefix caching ---
Request 1: prefill 192 tokens, latency = 19.2 ms
Request 2: prefill 192 tokens, latency = 19.2 ms
Request 3: prefill 192 tokens, latency = 19.2 ms
Total: 57.6 ms

--- Scenario 2: With prefix caching ---
Request 1: prefill 192 tokens (cache miss), latency = 19.2 ms
Request 2: prefill 128 tokens (64 prefix hit), latency = 12.8 ms
Request 3: prefill 128 tokens (64 prefix hit), latency = 12.8 ms
Total: 44.8 ms
Speedup: 1.29x

--- Scenario 3: Multi-turn dialogue ---
Turn 1: prefill 32 tokens, latency = 3.2 ms
Turn 2: prefill 8 tokens (56 prefix hit), latency = 0.8 ms
Turn 3: prefill 8 tokens (64 prefix hit), latency = 0.8 ms
Total: 4.8 ms
Without cache: 19.2 ms
Speedup: 4.00x
```

#### 任务 3：Profiling

```bash
# 用 cProfile 分析 prefix matching 的开销
python3 -m cProfile -s cumtime kernels/prefix_cache_engine.py | head -20
```

关注：
- prefix matching 耗时 vs prefill 耗时（应 << 1%）
- cache 命中率随请求数增长的趋势

#### 任务 4：LeetGPU 在线题目

[Segmented Prefix Sum](https://hzchenxiaobin.github.io/leetgpu/leetgpu-segmented-prefix-sum-solution.html)

prefix caching 的 block hash 计算类似于 segmented prefix sum 中按段累积——每个 block 是一个"段"，hash 是段的标识。

#### 任务 5：LeetCode 面试题

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [146](https://leetcode.cn/problems/lru-cache/) | Medium | LRU Cache（直接对应 prefix cache 淘汰） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/146_lru-cache.html) |
| [460](https://leetcode.cn/problems/lfu-cache/) | Hard | LFU Cache（扩展：频率感知淘汰） | — |
| [200](https://leetcode.cn/problems/number-of-islands/) | Medium | DFS/BFS（连通分量） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/200_number-of-islands.html) |
| [130](https://leetcode.cn/problems/surrounded-regions/) | Medium | DFS（边界 flood fill） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/130_surrounded-regions.html) |

---

### 扩展实验

#### 实验 1：调整 block_size 对命中率的影响

修改 `block_size` 为 8, 16, 32, 64，观察命中率变化：
- block_size 越小，匹配粒度越细，命中率越高，但 hash 计算开销越大
- block_size 越大，匹配粒度越粗，命中率越低，但管理开销越小

#### 实验 2：实现 Chunked Prefill + Prefix Caching 联合

在 `PrefixCacheEngine` 中添加 chunked prefill：
- prefix 匹配后，剩余 tokens 按 chunk_size 切分
- 每个 chunk 与 decode 迭代混合调度
- 测量 TTFT 和 TBT 改善

#### 实验 3：Copy-on-Write 实现

当多个请求共享同一 prefix block 时，某个请求需要 append 新 token 到该 block：
- 当前实现：直接复制 block（简单但浪费内存）
- COW 实现：`ref_count` 管理，写时才复制，读时共享

---

### 今日总结

Day 4 我们将 chunked prefill 和 prefix caching 从概念推进到了真实实现：

1. **Chunked Prefill**：长 prompt 分块与 decode 混合调度，chunk_size 选择影响 TTFT/TBT 权衡
2. **Prefix Caching**：block hash 匹配 + LRU 淘汰，共享 system prompt 场景命中率达 90%+
3. **两者协同**：正交互补，prefix caching 先跳过已知 prefix，chunked prefill 再切分剩余部分
4. **实测验证**：多轮对话场景 4x 加速，共享 system prompt 场景 1.3x 加速
5. **工程要点**：block hash 计算、LRU 淘汰、ref_count/COW、token budget 管理
6. **面试核心**：能讲清 prefix caching 的 block hash 机制和 LRU 淘汰策略，能量化命中场景的加速比

---

### 面试要点

1. **Prefix Caching 的 block hash 是怎么计算的？为什么不用 token sequence 直接比较？**

   <details>
   <summary>点击查看答案</summary>

   - **计算方式**：将每个 block（block_size 个 token）的 token sequence 做 hash（如 MD5/xxhash），得到一个固定长度的 hash 值作为 block 的唯一标识
   - **为什么不用直接比较**：
     1. **性能**：直接比较 token sequence 是 O(block_size)，而 hash 比较是 O(1)
     2. **存储**：cache pool 只需存 hash → physical block 映射，不需存完整 sequence
     3. **匹配效率**：请求进来时逐 block 计算 hash，O(1) 查表即可判断是否命中
   - **注意事项**：hash 碰撞概率极低（MD5/xxhash），但生产环境可加 sequence 验证作为兜底

   </details>

2. **Chunked Prefill 的 chunk_size 怎么选？太大和太小各有什么问题？**

   <details>
   <summary>点击查看答案</summary>

   - **太大**（如 2048）：
     - TTFT 低（长 prompt 快速完成 prefill）
     - TBT 波动大（大 chunk 占满 token budget，decode 被挤压）
     - 适合对 TTFT 敏感的批处理场景
   - **太小**（如 128）：
     - TBT 波动小（decode 不被挤压）
     - TTFT 高（长 prompt 需多个 iteration 才完成 prefill）
     - 适合对 TBT 敏感的在线服务
   - **vLLM 默认**：512（平衡 TTFT 和 TBT）
   - **关键约束**：`chunk_tokens + decode_tokens ≤ max_num_batched_tokens`

   </details>

3. **Prefix Caching 在什么场景下收益最大？什么场景下收益不明显？**

   <details>
   <summary>点击查看答案</summary>

   - **收益最大**：
     1. 多轮对话（历史 tokens 完全复用，命中率 80-95%）
     2. 共享 system prompt（多个请求用同一 system prompt，命中率 90-99%）
     3. Few-shot learning（相同的 examples 前缀）
   - **收益不明显**：
     1. 单轮对话（无历史可复用）
     2. 请求间无共享 prefix（每个请求的 prompt 完全不同）
     3. prefix 很短（即使命中也只省几个 token 的计算）
   - **面试技巧**：能说出"命中率取决于请求间的 prefix 重叠度"即可

   </details>

4. **Prefix Caching 的 LRU 淘汰策略有什么问题？有没有更好的策略？**

   <details>
   <summary>点击查看答案</summary>

   - **LRU 的问题**：
     1. 扫描场景：大量一次性请求会冲刷掉高频 prefix（cache pollution）
     2. 冷启动：cache 空时所有请求都 miss
   - **改进策略**：
     1. **LFU（Least Frequently Used）**：按访问频率淘汰，不受扫描影响
     2. **Segmented LRU**：分 hot/cold 两个区，新 block 先进 cold，被再次访问才升 hot
     3. **引用计数 + 延迟淘汰**：正在被请求引用的 block 不淘汰（vLLM 的 `ref_count` 机制）
   - **vLLM 的做法**：`ref_count` > 0 的 block 不可淘汰；`ref_count` = 0 的 block 按 LRU 淘汰

   </details>

5. **Chunked Prefill 和 Prefix Caching 能同时启用吗？它们是如何协同的？**

   <details>
   <summary>点击查看答案</summary>

   - **可以同时启用，且是正交的**：
     1. 请求到达 → 先做 prefix caching 匹配（跳过已知 prefix 的 prefill）
     2. 剩余未命中的 tokens → 用 chunked prefill 切分为 chunk
     3. 每个 chunk 与 decode 迭代混合调度
   - **协同效果**：
     - prefix caching 减少 prefill 总量（跳过已知部分）
     - chunked prefill 平滑剩余 prefill 对 decode 的影响
     - 两者叠加：既减少计算量，又平滑调度
   - **vLLM 实现**：`--enable-prefix-caching` + 默认 chunked prefill（v0.5+）

   </details>
