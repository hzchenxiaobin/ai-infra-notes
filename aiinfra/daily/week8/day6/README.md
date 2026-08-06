## Day 6：Profiling —— 量化前后精度性能对比与 CUDA Graph Launch Gap

### 🎯 目标

通过今天的学习，你将：

1. 能用 ncu/nsys 实测 **CUDA Graph 前后的 launch gap**——验证 launch overhead 占比<br>
2. 能做 **量化前后的精度对比**——W8A16/INT8 KV/FP8 的 perplexity 或 max_diff<br>
3. 能做 **量化前后的性能对比**——latency/throughput/显存的变化<br>
4. 能产出 **加速技术 ROI 表**——每种技术的集成成本 vs 性能收益<br>

> 💡 **为什么重要**：Day 5 集成了加速技术，今天用 profiling 验证收益。面试常问"CUDA Graph 收益多少""量化精度损失多少"，必须用数据回答。

---

### 理论学习

#### 6.1 CUDA Graph Launch Gap 实测

##### 两个层级的测量对象

- **微观演示**：`kernels/bench_eager.py`（10 层 Linear+LayerNorm 合成模型，kernel 粒度看 launch gap）
- **引擎级基准**：`kernels/bench_graph.py`（测量对象是 Day 5 真整合的 `MiniEngineV1Graph`，对比其 decode 路径 eager vs Graph 的 TBT 延迟）

##### nsys 时间线对比

```bash
# Eager 模式（微观演示模型）
nsys profile --trace cuda -o eager_profile python3 kernels/bench_eager.py

# Graph 模式（引擎级：MiniEngineV1Graph decode 路径）
nsys profile --trace cuda -o graph_profile python3 kernels/bench_graph.py
```

##### 预期时间线

**Eager**：
```
[CPU: launch kernel 1] [CPU: launch kernel 2] ... [CPU: launch kernel 30]
[GAP] [GPU: exec kernel 1] [GAP] [GPU: exec kernel 2] ... 
       ↑ 5-10μs gap per kernel
```

**Graph**：
```
[CPU: graph replay]  ← 一次 launch
[GPU: exec kernel 1 → 2 → 3 → ... → 30]  ← 无 gap
```

##### 实测数据（微观演示模型）

> 📏 **测量对象**：RTX 5090, **10 层 Linear+LayerNorm 合成模型**（`bench_eager.py` 的 `DecodeLikeModel`, seq=32 decode-like），**非引擎实测**。

```text
=== CUDA Graph Launch Gap (decode-like, 10 layers, seq=32) ===
Eager:  146.0 us
Graph:  67.7 us
Launch overhead: 78.3 us (53.7%)
Speedup: 2.16x
```

> ✅ **实测验证**：launch overhead 占 53.7%（超过一半！），CUDA Graph 后 speedup 2.16x。完全验证了"decode 路径 launch 占 50%+, Graph 后 -50%"的说法。

##### 引擎级 TBT 对比（MiniEngineV1Graph，待实测回填）

```bash
python3 kernels/bench_graph.py   # 4 请求并发，对比引擎 decode 路径 eager vs graph
```

> ⚠️ **数字诚信**：引擎级 TBT 数字需在 GPU 环境运行 `bench_graph.py` 后回填，禁止把上面合成模型的 146.0us/67.7us/2.16x 直接当作引擎收益。

```text
=== 测量对象：MiniEngineV1Graph decode 路径（Day 5 真整合引擎）===
  Eager decode: avg <待实测> ms/step
  Graph decode: avg <待实测> ms/step
  加速比: <待实测>x
```

##### 量化 launch overhead

```python
def measure_launch_overhead(model, input_ids):
    # Eager
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = model(input_ids)
    end.record()
    torch.cuda.synchronize()
    eager_ms = start.elapsed_time(end) / 100

    # Graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(input_ids)
    torch.cuda.synchronize()
    start.record()
    for _ in range(100):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    graph_ms = start.elapsed_time(end) / 100

    launch_overhead = eager_ms - graph_ms
    print(f"Eager: {eager_ms:.3f} ms, Graph: {graph_ms:.3f} ms")
    print(f"Launch overhead: {launch_overhead:.3f} ms ({launch_overhead/eager_ms*100:.1f}%)")
```

#### 6.2 量化精度对比

##### 精度指标

| 指标 | 含义 | 适合 |
|------|------|------|
| max_diff | 量化前后 logits 的最大差 | 简单验证 |
| perplexity | 语言模型困惑度 | 标准评估 |
| accuracy | 下游任务准确率 | 端到端 |

##### 精度对比脚本

```python
def compare_quantization(model_fp16, model_quant, test_prompts):
    for prompt in test_prompts:
        # FP16 参考输出
        with torch.no_grad():
            logits_fp16 = model_fp16(prompt).logits
        # 量化模型输出
        with torch.no_grad():
            logits_quant = model_quant(prompt).logits
        # 精度指标
        max_diff = (logits_fp16 - logits_quant).abs().max().item()
        cosine_sim = F.cosine_similarity(logits_fp16.flatten(), logits_quant.flatten(), dim=0)
        # 生成结果对比
        tokens_fp16 = logits_fp16.argmax(-1)
        tokens_quant = logits_quant.argmax(-1)
        token_match = (tokens_fp16 == tokens_quant).float().mean()
        print(f"max_diff={max_diff:.4f}, cosine={cosine_sim:.6f}, token_match={token_match:.2%}")
```

##### 预期精度损失

> 📊 **数据来源 / 口径**：下表为**量级参考值**，综合 vLLM、TensorRT-LLM、GPTQ、AWQ、SmoothQuant 等公开 benchmark 与社区典型结论；**非本仓库 Mini 引擎实测**。实际精度损失与模型、校准数据、per-channel/per-token 策略强相关，需用本节脚本在目标模型上实测后回填。

| 量化方案 | max_diff | perplexity 变化 | token_match |
|---------|---------|----------------|------------|
| W8A16 (INT8 权重) | ~0.1 | < 0.5% | > 99% |
| W4A16 (GPTQ) | ~0.5 | < 1% | > 98% |
| INT8 KV Cache | ~0.01 | < 0.1% | > 99.5% |
| FP8 E4M3 | ~0.05 | < 0.3% | > 99% |

> ⚠️ **数字诚信**：上表数值为**估算量级 / 第三方公开资料参考**，非本仓库实测。Mini 引擎的量化精度对比需在 GPU 环境运行 `compare_quantization` 实测后回填。

#### 6.3 量化性能对比

##### 性能指标

> 📊 **数据来源 / 口径**：
> - 显存两行（模型 / KV Cache）为**理论计算值**：FP16 权重 = 参数量 × 2 bytes，W8A16/FP8 ≈ 1 byte，INT8 KV Cache = FP16 KV 的 50%。
> - latency / throughput 三行为**7B 模型在数据中心级 GPU（A100/H100 口径）上的量级参考值**，来源为 vLLM、TensorRT-LLM 等公开 benchmark 与社区典型结论；**非本仓库 Mini 引擎实测**。

| 指标 | FP16 baseline | W8A16 | INT8 KV | FP8 |
|------|-------------|-------|---------|-----|
| 显存(模型) | 14GB(7B) | 7GB | 14GB | 7GB |
| 显存(KV Cache) | 2GB | 2GB | 1GB | 2GB |
| Prefill latency | 100ms | 90ms | 100ms | 60ms |
| Decode latency | 5ms | 4ms | 3.5ms | 3ms |
| Throughput | 200 tok/s | 250 tok/s | 280 tok/s | 350 tok/s |

> ⚠️ **数字诚信**：显存两行为**理论计算值**（参数量 × 字节数，见下方推导）；latency/throughput 三行为**第三方公开 benchmark 量级参考**，非本仓库实测。Mini 引擎的实际收益需在 GPU 环境用 `bench_graph.py` / 量化 benchmark 脚本实测后回填。

##### 显存节省计算

```
FP16 模型显存 = 参数量 × 2 bytes
W8A16 = 参数量 × 1 byte (INT8) + scale = ~50% FP16
W4A16 = 参数量 × 0.5 byte (INT4) + scale = ~25% FP16
FP8 = 参数量 × 1 byte = ~50% FP16

INT8 KV Cache = seq_len × hidden × 1 byte (vs FP16 2 byte) = 50%
```

#### 6.4 加速技术 ROI 表

| 技术 | 集成成本(行) | 显存收益 | latency 收益 | throughput 收益 | ROI |
|------|------------|---------|------------|----------------|-----|
| CUDA Graph | ~50 | 0 | -30-50% decode | +20-30% | ⭐⭐⭐ |
| INT8 KV Cache | ~200 | -50% KV | -30% decode | +15-20% | ⭐⭐ |
| W8A16 量化 | ~300 | -50% 模型 | -10% prefill | +10% | ⭐⭐ |
| FP8 GEMM | ~500(Hopper) | -50% 模型 | -40% | +50% | ⭐⭐ |
| 投机解码 | ~500 | 0 | 0 | +50-100% | ⭐ |

---

### Coding 任务

#### 任务 1：Launch Gap 实测

```bash
# nsys 对比 eager vs graph
nsys profile --trace cuda -o eager python3 kernels/bench_eager.py
nsys profile --trace cuda -o graph python3 kernels/bench_graph.py
nsys stats eager.nsys-rep --report cuda_gpu_kern  # 看 kernel 间 gap
nsys stats graph.nsys-rep --report cuda_gpu_kern
```

#### 任务 2：量化精度对比

```python
# 对比 FP16 vs W8A16 vs INT8 KV 的 logits
for quant in ['fp16', 'w8a16', 'int8_kv']:
    model = load_model(quant)
    logits = model(test_prompt)
    diff = (logits - logits_fp16).abs().max()
    print(f"{quant}: max_diff={diff:.4f}")
```

#### 任务 3：LeetCode 面试题

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [23](https://leetcode.cn/problems/merge-k-sorted-lists/) | Hard | 堆 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/23_合并K个升序链表.html) |
| [253](https://leetcode.cn/problems/meeting-rooms-ii/) | Medium | 堆/扫描线 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/253_会议室II.html) |
| [703](https://leetcode.cn/problems/kth-largest-element-in-a-stream/) | Easy | 堆 | 暂无 |

---

### 今日总结

1. **Launch Gap**：Eager 模式 30 个 kernel × 5-10μs = 150-300μs launch overhead，占 decode 50%+
2. **CUDA Graph 收益**：launch 压成 1 次，decode latency -30-50%
3. **量化精度**：W8A16 max_diff ~0.1, INT8 KV ~0.01, FP8 ~0.05，token_match > 98%
4. **量化性能**：W8A16 显存 -50%, INT8 KV 显存 -50%+带宽 -30%, FP8 latency -40%
5. **ROI 排序**：CUDA Graph > INT8 KV > W8A16 > FP8 > 投机解码（按集成成本/收益比）

---

### 面试要点

1. **CUDA Graph 的 launch gap 占 decode 的多少？怎么测？**

   <details>
   <summary>答案</summary>

   - Decode M=1 时 30 个 kernel × 5-10μs launch = 150-300μs，占总 latency 50%+
   - 测法：nsys profile 看 kernel 间 gap；或对比 eager vs graph 的 wall time 差
   - Graph 后：1 次 replay (~10μs) 替代 30 次 launch，latency -30-50%

   </details>

2. **量化前后精度损失多少？怎么评估？**

   <details>
   <summary>答案</summary>

   - 评估指标：max_diff（logits 最大差）、perplexity 变化、token_match（生成 token 一致率）
   - W8A16: max_diff ~0.1, perplexity < 0.5%, token_match > 99%
   - INT8 KV: max_diff ~0.01, token_match > 99.5%
   - FP8: max_diff ~0.05, token_match > 99%
   - 生产实践：perplexity 变化 < 1% 为可接受

   </details>

3. **各种加速技术的 ROI 怎么排序？**

   <details>
   <summary>答案</summary>

   按"集成成本/收益比"排序：
   1. CUDA Graph（50 行, -50% decode latency）
   2. INT8 KV Cache（200 行, -50% KV 显存 + -30% bandwidth）
   3. W8A16（300 行, -50% 模型显存）
   4. FP8（500 行 + Hopper, -40% latency）
   5. 投机解码（500 行, +50-100% throughput 但复杂度高）

   </details>
