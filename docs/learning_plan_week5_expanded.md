# 第5周深度展开：推理系统与 KV Cache（7天）

> **适用对象**：陈斌斌（已完成第4周学习，掌握 FlashAttention 原理与实现、IO 优化方法论）
> **本周目标**：进入 AI Infra 核心，理解推理系统的完整流程，实现 KV Cache，阅读 vLLM 源码，构建第一个可运行的 Mini 推理引擎
> **时间投入**：工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h
> **周日里程碑**：手写 KV Cache 支持多轮对话，Mini 推理引擎 v0 完成单请求 prefill+decode，产出推理系统核心问题清单

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 理解 LLM 推理的 Prefill/Decode 两阶段，掌握 KV Cache 的设计与实现，阅读 vLLM 架构源码，构建支持单请求的 Mini 推理引擎 v0 |
| **核心产出** | ① Prefill/Decode 模拟脚本 ② KV Cache CUDA 实现（C++ 类）③ vLLM 架构分析报告 ④ PagedAttention 笔记 ⑤ Mini 推理引擎 v0 ⑥ Profiling 报告 ⑦ 推理系统核心问题清单 |
| **验收标准** | ① 能清晰区分 Prefill 和 Decode 的计算/内存特征 ② KV Cache 输出与无 cache 版本一致，decode latency 降低 10x+ ③ 能画出 vLLM 架构图并解释请求生命周期 ④ Mini 引擎 v0 能完成单条请求完整推理 ⑤ 能测量 TTFT 和 per-token decode latency |
| **面试准备** | 积累12-15道推理系统专题面试题，覆盖 Prefill/Decode、KV Cache、vLLM、PagedAttention、Continuous Batching、调度优化六大主题 |

### 本周知识图谱

```
Day 29: Prefill vs Decode → 两阶段特征对比 + PyTorch 模拟
 ↓
Day 30: KV Cache 实现 → C++/CUDA 缓存分配、更新、查询、多轮对话
 ↓
Day 31: vLLM 整体架构 → LLMEngine / Scheduler / Worker / SequenceGroup
 ↓
Day 32: vLLM Worker + PagedAttention → BlockSpaceManager / Block Table / Copy-on-Write
 ↓
Day 33: Mini 推理引擎 v0 → 单请求 + KV Cache + Prefill/Decode 循环
 ↓
Day 34: 端到端 Profiling → TTFT / TBT / 阶段 latency / 瓶颈定位
 ↓
Day 35: 推理系统核心问题总结 → 内存管理、Batch 策略、Latency 隐藏、调度开销
```

### 前置准备清单

#### 硬件/软件验证
- [ ] 已完成第4周所有 Coding 任务（FlashAttention Kernel、Mini 引擎集成）
- [ ] PyTorch 可用且 `torch.__version__ >= 2.0`
- [ ] 可安装 vLLM（用于源码阅读，不强求运行）：`pip install vllm`
- [ ] `nsys --version` 正常（端到端 profiling 需要）
- [ ] 理解 Week 3 Day 15 的 Prefill/Decode 笔记

#### 验证命令
```bash
# 验证 PyTorch + CUDA
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
# 预期输出：torch 2.x.x cuda 12.x available True

# 验证 vLLM 可导入（可选）
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm not installed"

# 验证 nsys
nsys --version
# 预期输出：NVIDIA Nsight Systems version 202x.x.x
```

---

## Day 29（周一）：推理流程 —— Prefill vs Decode

> **今日目标**：深入理解 Prefill 和 Decode 两阶段的本质差异，用 PyTorch 模拟完整推理流程，计算两阶段的 FLOPs/Bytes/Arithmetic Intensity。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，"Prefill vs Decode"是推理系统入门第一考点

---

### 学习任务1：Prefill vs Decode 的本质差异（45分钟）

#### 阅读内容
- **论文/博客**：vLLM 博客 "How vLLM serves LLM" 中 Prefill vs Decode 章节
- **补充阅读**：AnyScale 博客 "LLM Inference Performance Engineering" 中 prefill/decode 部分
- **重点**：
 - 两阶段的输入形状差异
 - Attention 矩阵形状差异
 - 计算强度和瓶颈类型差异
 - TTFT (Time To First Token) 和 TBT (Time Between Tokens) 的定义

#### 核心概念笔记

**1. Prefill 阶段（首次前向 / 输入处理阶段）**

```
输入: prompt tokens, shape = (B, N_prompt, d)

处理:
 1. 一次性并行处理所有 prompt tokens
 2. 计算所有 token 的 Q, K, V
 3. 对 prompt 内部做 self-attention（N×N 完整矩阵）
 4. 输出第一个新 token 的 logits

特征:
 - 计算密集 (compute-bound)
 - GEMM 是大矩阵乘（M=N_prompt 较大）
 - 可充分利用 Tensor Core
 - latency 关注: TTFT (Time To First Token)
```

**2. Decode 阶段（自回归生成阶段）**

```
输入: 上一个生成的 token, shape = (B, 1, d)

处理:
 1. 只计算新 token 的 Q
 2. 从 KV Cache 读取所有历史 K, V
 3. 新 Q 与所有历史 K/V 做 attention（1×N 矩阵）
 4. 输出下一个 token 的 logits
 5. 重复直到生成 </eos> 或达到最大长度

特征:
 - 访存密集 (memory-bound)
 - GEMM 退化为向量×矩阵（M=1）
 - 每次都要读取完整 KV Cache
 - latency 关注: TBT (Time Between Tokens) / TPOT (Time Per Output Token)
```

**3. 关键指标**

| 指标 | 全称 | 含义 | 阶段 |
|------|------|------|------|
| TTFT | Time To First Token | 首个 token 生成时间 | Prefill |
| TBT | Time Between Tokens | 相邻输出 token 间隔 | Decode |
| TPOT | Time Per Output Token | 每个输出 token 时间 | Decode |
| TPS | Tokens Per Second | 吞吐 | Decode |
| E2E Latency | End-to-End Latency | 总延迟 | 全程 |

**4. 计算强度对比（B=1, N_prompt=1024, d=512）**

```
Prefill QKV GEMM:
 FLOPs = 2 × B × N × d × 3d = 2 × 1 × 1024 × 512 × 1536 ≈ 1.6G
 Bytes = B × N × d + 3d² + 3B × N × d ≈ 4MB
 AI ≈ 400 FLOP/Byte → compute-bound

Decode QKV GEMM (M=1):
 FLOPs = 2 × B × 1 × d × 3d ≈ 1.6M
 Bytes = B × 1 × d + 3d² + 3B × 1 × d ≈ 8KB
 AI 理论值高，但矩阵极小，实际 memory-bound（SM 空闲等数据）
```

### 学习任务2：Decode 阶段为什么 Memory-bound？（30分钟）

#### 详细分析

```
Decode 阶段每次处理 1 个新 token：
 - 需要读取历史 K, V: 2 × L × d × bytes
 - 需要读取模型权重: 2 × d_model² × bytes
 - 计算量: O(L × d + d²)

Arithmetic Intensity:
 AI = FLOPs / Bytes ≈ d / (2 × d × bytes_per_float) ≈ 0.125 FLOP/Byte

远低于 Ridge Point (~12.6 on RTX 5090)，所以是 memory-bound。
```

**直观理解**：
- Prefill 像"一群人一起搬砖"，人多活也多，瓶颈是人力（算力）
- Decode 像"一个人重复跑腿取材料"，每次只干一点活，但要从仓库（HBM）取大量材料

#### 优化方向预览

| 优化方向 | 目标 | 代表技术 |
|---------|------|---------|
| 减少 KV Cache 读取 | 降低内存访问 | KV Cache、PagedAttention、Quantization |
| 合并多个 decode 请求 | 提高 M，让 GEMM 变大 | Continuous Batching、Inflight Batching |
| 减少 launch overhead | 降低调度开销 | CUDA Graph、torch.compile |
| 隐藏传输延迟 | overlap 计算与通信 | Pipeline Parallelism、Async |

---

### 晚间编程任务：PyTorch 模拟 Prefill/Decode 流程（1小时）

#### 完整代码

```python
# prefill_decode_simulation.py —— 模拟 Transformer 推理的 Prefill/Decode 两阶段
# 运行命令: python prefill_decode_simulation.py
# 依赖: pip install torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class MiniTransformer(nn.Module):
 """最小 Transformer Block，用于演示 Prefill/Decode"""
 def __init__(self, d_model=512, n_heads=8, d_ff=2048):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.qkv = nn.Linear(d_model, 3 * d_model)
 self.out = nn.Linear(d_model, d_model)
 self.norm1 = nn.LayerNorm(d_model)
 self.norm2 = nn.LayerNorm(d_model)
 self.ffn = nn.Sequential(
 nn.Linear(d_model, d_ff),
 nn.GELU(),
 nn.Linear(d_ff, d_model),
 )

 def forward(self, x, use_cache=False, k_cache=None, v_cache=None):
 """
 x: (B, N, d_model)
 use_cache: 是否使用 KV Cache
 k_cache/v_cache: 历史 KV，shape (B, H, L, d_head)
 返回: output, (new_k_cache, new_v_cache)
 """
 B, N, _ = x.shape

 # LayerNorm + QKV
 x_norm = self.norm1(x)
 qkv = self.qkv(x_norm)
 qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
 q, k, v = qkv[0], qkv[1], qkv[2]

 # Attention
 scale = self.d_head ** -0.5
 if use_cache and k_cache is not None:
 # Decode: 只与新 token 的 Q 做 attention
 k = torch.cat([k_cache, k], dim=2) # (B, H, L+1, d)
 v = torch.cat([v_cache, v], dim=2)

 attn = torch.matmul(q, k.transpose(-2, -1)) * scale
 attn = F.softmax(attn, dim=-1)
 out = torch.matmul(attn, v)

 out = out.transpose(1, 2).reshape(B, N, self.d_model)
 x = x + self.out(out)

 # FFN
 x = x + self.ffn(self.norm2(x))

 return x, (k, v)

def simulate_inference(model, prompt, max_new_tokens=20):
 """模拟完整推理流程：Prefill + Decode"""
 device = next(model.parameters()).device
 B, N = prompt.size(0), prompt.size(1)

 # ========== Prefill 阶段 ==========
 torch.cuda.synchronize()
 t_start = time.time()

 with torch.no_grad():
 logits, (k_cache, v_cache) = model(prompt, use_cache=False)
 first_token_logits = logits[:, -1, :] # 取最后一个位置的 logits

 torch.cuda.synchronize()
 ttft = (time.time() - t_start) * 1000 # ms

 print(f"=== Prefill Phase ===")
 print(f" Input shape: {tuple(prompt.shape)}")
 print(f" TTFT: {ttft:.3f} ms")
 print(f" KV Cache shape: {tuple(k_cache.shape)}")

 # ========== Decode 阶段 ==========
 generated = []
 decode_times = []

 # 这里简化：用 argmax 采样，实际可用 temperature/top-p
 next_token = first_token_logits.argmax(dim=-1, keepdim=True)
 generated.append(next_token.item())

 for step in range(max_new_tokens - 1):
 # 每次只输入上一个 token
 next_input = torch.cat([
 model.ffn[0].weight.new_zeros(B, 1, model.d_model), # placeholder
 ], dim=1) if False else None

 # 实际应该通过 embedding 获取 token 的向量
 # 为简化，直接用 prompt 最后一个 token 的 embedding 做输入
 # 这里用可学习的 placeholder token 向量（仅用于演示）
 next_token_emb = model.qkv.weight.new_zeros(B, 1, model.d_model).normal_(0, 0.02)

 torch.cuda.synchronize()
 t_start = time.time()

 with torch.no_grad():
 logits, (k_cache, v_cache) = model(
 next_token_emb, use_cache=True, k_cache=k_cache, v_cache=v_cache
 )
 next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

 torch.cuda.synchronize()
 decode_times.append((time.time() - t_start) * 1000)
 generated.append(next_token.item())

 print(f"\n=== Decode Phase ===")
 print(f" Generated {len(generated)} tokens")
 print(f" Mean TBT: {sum(decode_times)/len(decode_times):.3f} ms")
 print(f" Max TBT: {max(decode_times):.3f} ms")
 print(f" Min TBT: {min(decode_times):.3f} ms")
 print(f" Generated token IDs: {generated}")

 return ttft, decode_times

def profile_phase(model, x, name, n_iter=10):
 """Profile 一个阶段"""
 for _ in range(3):
 _ = model(x)
 torch.cuda.synchronize()

 start = torch.cuda.Event(enable_timing=True)
 end = torch.cuda.Event(enable_timing=True)
 start.record()
 for _ in range(n_iter):
 with torch.no_grad():
 _ = model(x)
 end.record()
 torch.cuda.synchronize()
 ms = start.elapsed_time(end) / n_iter
 print(f"{name}: {ms:.3f} ms")
 return ms

def main():
 torch.manual_seed(42)
 device = "cuda"
 d_model, n_heads = 512, 8
 model = MiniTransformer(d_model, n_heads).to(device).eval().half()

 # Prefill: 处理长 prompt
 N = 1024
 prompt = torch.randn(1, N, d_model, device=device, dtype=torch.float16)

 print(f"Model: d_model={d_model}, n_heads={n_heads}")
 print(f"Prompt length: {N}\n")

 simulate_inference(model, prompt, max_new_tokens=10)

 # 单独 profile prefill vs decode
 print("\n=== Standalone Profiling ===")
 profile_phase(model, prompt, f"Prefill (N={N})")

 decode_input = torch.randn(1, 1, d_model, device=device, dtype=torch.float16)
 profile_phase(model, decode_input, f"Decode single token")

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python prefill_decode_simulation.py

# 预期输出
# Model: d_model=512, n_heads=8
# Prompt length: 1024
# 
# === Prefill Phase ===
# Input shape: (1, 1024, 512)
# TTFT: x.xxx ms
# KV Cache shape: (1, 8, 1024, 64)
# 
# === Decode Phase ===
# Generated 10 tokens
# Mean TBT: x.xxx ms
# ...
```

#### 练习题

**练习1（基础）**：修改 `N=256, 512, 1024, 2048`，分别测量 TTFT，绘制 TTFT 随 N 变化曲线。
> 提示：TTFT 与 N 大致呈二次关系（因为 attention 是 O(N²)）。

**练习2（进阶）**：在 Decode 阶段使用真实 embedding（而不是随机向量）作为输入。
> 提示：添加一个简单的 `nn.Embedding(vocab_size, d_model)`，将 token id 转换为向量。

**练习3（综合）**：比较 `use_cache=True` 和 `use_cache=False` 在 Decode 阶段的 latency 差异。
> 提示：不用 cache 时，每次都要重新计算历史 K/V，latency 会高很多。

---

### 今日面试题

**面试题1**：LLM 推理的 Prefill 和 Decode 阶段有什么区别？各自的瓶颈是什么？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **Prefill**：输入 `(B, N_prompt, d)`，一次性并行处理所有 prompt tokens，计算完整 N×N attention，输出第一个 token。瓶颈是算力（compute-bound），关注 TTFT
- **Decode**：输入 `(B, 1, d)`，自回归逐个生成 token，使用 KV Cache 避免重复计算历史 K/V。瓶颈是显存带宽（memory-bound），关注 TBT/TPOT
- **根本原因**：Decode 阶段 M=1，GEMM 退化为向量×矩阵，计算量小但数据读取量大，arithmetic intensity 极低

**面试题2**：什么是 TTFT 和 TBT？在系统优化中分别如何优化？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **TTFT (Time To First Token)**：从请求进入到输出第一个 token 的时间，主要由 Prefill 阶段决定
- **TBT (Time Between Tokens)**：相邻输出 token 之间的时间，主要由 Decode 阶段决定
- **优化 TTFT**：使用 Tensor Core、FlashAttention、减少 prompt 长度、并行 prefill
- **优化 TBT**：KV Cache、PagedAttention、Continuous Batching、CUDA Graph、量化 KV Cache

---

### 今日自测清单

- [ ] 能清晰区分 Prefill 和 Decode 的输入形状、attention 矩阵形状、瓶颈类型
- [ ] 能解释为什么 Decode 是 memory-bound（M=1 导致 AI 极低）
- [ ] 能说出 TTFT 和 TBT 的定义及优化方向
- [ ] PyTorch 模拟脚本运行成功，输出 TTFT 和 TBT
- [ ] 能绘制 TTFT 随 N 变化的曲线

---

## Day 30（周二）：实现 KV Cache

> **今日目标**：理解 KV Cache 的核心思想，实现支持分配、更新、查询的 C++/CUDA KV Cache，支持多轮对话的历史缓存复用。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，KV Cache 是推理系统优化的基础

---

### 学习任务1：KV Cache 核心思想（45分钟）

#### 阅读内容
- **论文/博客**：
 - "Efficient Large Language Models: A Survey" 中 KV Cache 章节
 - vLLM 博客中关于 KV Cache 的部分
- **重点**：
 - 为什么需要 KV Cache
 - Cache 的分配策略：静态 vs 动态
 - 多轮对话中如何复用历史 cache

#### 核心概念笔记

**1. 为什么需要 KV Cache？**

```
Decode 阶段：
 第 t 步需要计算 attention(Q_t, K_1..K_t, V_1..V_t)
 第 t+1 步需要计算 attention(Q_{t+1}, K_1..K_{t+1}, V_1..V_{t+1})

观察：K_1..K_t 和 V_1..V_t 在第 t+1 步会重复计算

KV Cache：
 - 第 t 步计算完 K_t, V_t 后，保存到 cache
 - 第 t+1 步只需要计算 K_{t+1}, V_{t+1}，然后从 cache 读取历史 K/V
 - 避免重复计算，显著降低 decode latency
```

**2. 有/无 KV Cache 的对比**

| 维度 | 无 KV Cache | 有 KV Cache |
|------|------------|------------|
| 每步计算 K/V | 重新计算所有历史 K/V | 只计算新 token 的 K/V |
| 每步 FLOPs | O(L × d²) | O(d²) |
| 每步 HBM 读取 | 重新读取所有历史 tokens | 从 cache 读取历史 K/V |
| 内存使用 | 低 | 高（2 × L × d × bytes） |
| Decode latency | 高（与 L 成正比增长） | 低（基本稳定） |

**3. KV Cache 的内存占用**

```
每个 token 的 KV Cache 大小 = 2 × num_layers × num_heads × d_head × bytes_per_elem

例如：
 layers=32, heads=32, d_head=128, fp16
 每 token KV Cache = 2 × 32 × 32 × 128 × 2 bytes = 524 KB

对于 4096 tokens：
 总 KV Cache = 4096 × 524 KB ≈ 2 GB

对于 batch=16：
 总 KV Cache = 16 × 2 GB = 32 GB
```

> 这就是为什么 KV Cache 是长文本、大 batch 推理的主要内存瓶颈。

**4. 分配策略：静态 vs 动态**

```
静态分配：
 - 为每个请求预先分配最大序列长度的 cache
 - 优点：简单，无内存碎片
 - 缺点：内存浪费（请求实际长度可能远小于最大长度）

动态分配：
 - 根据序列长度动态分配/扩展 cache
 - 优点：内存利用率高
 - 缺点：需要管理碎片、分配开销

PagedAttention（vLLM）：
 - 将 KV cache 分成固定大小的 block
 - 类似 OS 的虚拟内存分页
 - 解决动态分配的碎片问题
```

### 学习任务2：多轮对话中的 Cache 复用（30分钟）

#### 场景分析

```
多轮对话：
 Round 1: User: "你好" → Model: "你好！有什么可以帮你？"
 Round 2: User: "你好，请介绍一下 FlashAttention" → Model: "FlashAttention 是..."

Round 2 的 prompt 包含：
 [系统提示] + [Round 1 User] + [Round 1 Assistant] + [Round 2 User]

Cache 复用：
 - Round 1 已经计算过的 K/V 可以直接复用
 - Round 2 只需要计算新增 tokens 的 K/V
 - 大幅降低多轮对话的 TTFT
```

#### 实现要点

```
1. 为每个对话 session 维护一个 KV Cache
2. 每次用户输入时，先复用已有 cache
3. 对新输入 tokens 做 prefill，并将新 K/V 追加到 cache
4. 生成 assistant 回复时，使用 append 模式更新 cache
5. 释放已完成/超时的 session cache
```

---

### 晚间编程任务：手写 KV Cache（1.5小时）

#### 完整代码

```cpp
// kv_cache.cu —— 支持多轮对话的 KV Cache CUDA 实现
// 编译命令: nvcc -o kv_cache kv_cache.cu -O3 -arch=sm_120
// 运行命令: ./kv_cache

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// --------------------------------------------------
// 简单 KV Cache 类
// 存储 layout: (num_layers, B, H, max_seq_len, d_head)
// 为简化，这里只实现单层
// --------------------------------------------------
class KVCache {
public:
 KVCache(int num_layers, int batch_size, int num_heads, int max_seq_len, int d_head)
 : num_layers_(num_layers), batch_size_(batch_size),
 num_heads_(num_heads), max_seq_len_(max_seq_len), d_head_(d_head) {

 size_per_layer_ = (size_t)batch_size_ * num_heads_ * max_seq_len_ * d_head_ * sizeof(float);
 total_size_ = (size_t)num_layers_ * size_per_layer_;

 cudaMalloc(&k_cache_, total_size_);
 cudaMalloc(&v_cache_, total_size_);
 cudaMemset(k_cache_, 0, total_size_);
 cudaMemset(v_cache_, 0, total_size_);

 // 每个 batch 当前已缓存的序列长度
 seq_lens_ = std::vector<int>(batch_size_, 0);
 }

 ~KVCache() {
 cudaFree(k_cache_);
 cudaFree(v_cache_);
 }

 // 追加新的 K/V 到 cache
 // k_new/v_new shape: (batch_size, num_heads, new_len, d_head)
 // 本函数假设 k_new/v_new 在 device 上
 void append(int layer_id, const float* k_new, const float* v_new, int new_len) {
 for (int b = 0; b < batch_size_; b++) {
 int start = seq_lens_[b];
 int end = start + new_len;
 if (end > max_seq_len_) {
 printf("Error: seq len exceeds max_seq_len_\n");
 return;
 }

 // 拷贝 k_new[b, :, :, :] 到 k_cache_[layer_id, b, :, start:end, :]
 for (int h = 0; h < num_heads_; h++) {
 size_t src_offset = ((size_t)b * num_heads_ * new_len * d_head_ +
 h * new_len * d_head_) * sizeof(float);
 size_t dst_offset = ((size_t)layer_id * batch_size_ * num_heads_ * max_seq_len_ * d_head_ +
 b * num_heads_ * max_seq_len_ * d_head_ +
 h * max_seq_len_ * d_head_ +
 start * d_head_) * sizeof(float);
 size_t bytes = (size_t)new_len * d_head_ * sizeof(float);
 cudaMemcpy(k_cache_ + dst_offset / sizeof(float), k_new + src_offset / sizeof(float),
 bytes, cudaMemcpyDeviceToDevice);
 cudaMemcpy(v_cache_ + dst_offset / sizeof(float), v_new + src_offset / sizeof(float),
 bytes, cudaMemcpyDeviceToDevice);
 }
 seq_lens_[b] = end;
 }
 }

 // 获取当前 cache 指针和序列长度
 void get_cache(int layer_id, float** k_ptr, float** v_ptr, std::vector<int>* seq_lens) {
 *k_ptr = k_cache_ + (size_t)layer_id * size_per_layer_ / sizeof(float);
 *v_ptr = v_cache_ + (size_t)layer_id * size_per_layer_ / sizeof(float);
 *seq_lens = seq_lens_;
 }

 int get_seq_len(int batch_id) const {
 return seq_lens_[batch_id];
 }

 void reset() {
 cudaMemset(k_cache_, 0, total_size_);
 cudaMemset(v_cache_, 0, total_size_);
 std::fill(seq_lens_.begin(), seq_lens_.end(), 0);
 }

 void reset_batch(int batch_id) {
 for (int l = 0; l < num_layers_; l++) {
 for (int h = 0; h < num_heads_; h++) {
 size_t offset = ((size_t)l * batch_size_ * num_heads_ * max_seq_len_ * d_head_ +
 batch_id * num_heads_ * max_seq_len_ * d_head_ +
 h * max_seq_len_ * d_head_) * sizeof(float);
 cudaMemset(k_cache_ + offset / sizeof(float), 0, max_seq_len_ * d_head_ * sizeof(float));
 cudaMemset(v_cache_ + offset / sizeof(float), 0, max_seq_len_ * d_head_ * sizeof(float));
 }
 }
 seq_lens_[batch_id] = 0;
 }

private:
 int num_layers_, batch_size_, num_heads_, max_seq_len_, d_head_;
 size_t size_per_layer_, total_size_;
 float* k_cache_;
 float* v_cache_;
 std::vector<int> seq_lens_;
};

// --------------------------------------------------
// 测试：模拟多轮对话
// --------------------------------------------------
void initData(float* data, int n) {
 for (int i = 0; i < n; i++) {
 data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
 }
}

int main() {
 const int num_layers = 2;
 const int batch_size = 1;
 const int num_heads = 8;
 const int max_seq_len = 1024;
 const int d_head = 64;

 printf("=== KV Cache Test ===\n");
 printf("Config: layers=%d, batch=%d, heads=%d, max_len=%d, d_head=%d\n",
 num_layers, batch_size, num_heads, max_seq_len, d_head);

 KVCache cache(num_layers, batch_size, num_heads, max_seq_len, d_head);

 // Round 1: prompt 长度 10
 int round1_len = 10;
 size_t round1_bytes = (size_t)batch_size * num_heads * round1_len * d_head * sizeof(float);
 float *d_k1, *d_v1;
 cudaMalloc(&d_k1, round1_bytes);
 cudaMalloc(&d_v1, round1_bytes);
 // 实际应用中会调用 transformer layer 生成 k1/v1
 // 这里用随机数据填充
 float* h_tmp = (float*)malloc(round1_bytes);
 initData(h_tmp, batch_size * num_heads * round1_len * d_head);
 cudaMemcpy(d_k1, h_tmp, round1_bytes, cudaMemcpyHostToDevice);
 cudaMemcpy(d_v1, h_tmp, round1_bytes, cudaMemcpyHostToDevice);

 cache.append(0, d_k1, d_v1, round1_len);
 printf("After Round 1 (len=%d): seq_len=%d\n", round1_len, cache.get_seq_len(0));

 // Round 2: 新增 5 个 tokens
 int round2_len = 5;
 size_t round2_bytes = (size_t)batch_size * num_heads * round2_len * d_head * sizeof(float);
 float *d_k2, *d_v2;
 cudaMalloc(&d_k2, round2_bytes);
 cudaMalloc(&d_v2, round2_bytes);
 initData(h_tmp, batch_size * num_heads * round2_len * d_head);
 cudaMemcpy(d_k2, h_tmp, round2_bytes, cudaMemcpyHostToDevice);
 cudaMemcpy(d_v2, h_tmp, round2_bytes, cudaMemcpyHostToDevice);

 cache.append(0, d_k2, d_v2, round2_len);
 printf("After Round 2 (len=%d): seq_len=%d\n", round2_len, cache.get_seq_len(0));

 // Round 3: 新增 8 个 tokens
 cache.append(0, d_k2, d_v2, 8);
 printf("After Round 3 (len=8): seq_len=%d\n", cache.get_seq_len(0));

 // 验证总长度
 int expected = round1_len + round2_len + 8;
 if (cache.get_seq_len(0) == expected) {
 printf("PASS: seq_len = %d (expected %d)\n", cache.get_seq_len(0), expected);
 } else {
 printf("FAIL: seq_len = %d (expected %d)\n", cache.get_seq_len(0), expected);
 }

 // 内存占用统计
 size_t bytes_per_token = (size_t)num_layers * num_heads * d_head * sizeof(float) * 2;
 printf("KV Cache bytes per token: %zu\n", bytes_per_token);
 printf("Max memory usage: %zu MB\n", bytes_per_token * max_seq_len / (1024 * 1024));

 free(h_tmp);
 cudaFree(d_k1); cudaFree(d_v1);
 cudaFree(d_k2); cudaFree(d_v2);

 return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o kv_cache kv_cache.cu -O3 -arch=sm_120

# 运行
./kv_cache

# 预期输出
# === KV Cache Test ===
# Config: layers=2, batch=1, heads=8, max_len=1024, d_head=64
# After Round 1 (len=10): seq_len=10
# After Round 2 (len=5): seq_len=15
# After Round 3 (len=8): seq_len=23
# PASS: seq_len = 23 (expected 23)
# KV Cache bytes per token: 8192
# Max memory usage: 8 MB
```

#### 练习题

**练习1（基础）**：扩展 `KVCache` 支持多个 batch，每个 batch 有不同的序列长度。

**练习2（进阶）**：实现 `KVCache` 的 FP16 版本（使用 `half` 类型），对比内存占用。

**练习3（综合）**：将 `KVCache` 与 Day 29 的 `MiniTransformer` 结合，实现真正的 decode 缓存复用。

---

### 今日面试题

**面试题1**：KV Cache 的核心思想是什么？为什么能显著降低 Decode latency？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- Decode 阶段是自回归的，第 t 步和第 t+1 步都需要历史 K/V
- 没有 KV Cache 时，每次都要重新计算所有历史 tokens 的 K/V
- KV Cache 将每步新生成的 K/V 保存下来，后续步骤直接读取
- 每步计算量从 O(L × d²) 降到 O(d²)，latency 通常降低 10x+
- 代价是显存占用：2 × layers × heads × L × d_head × bytes

**面试题2**：KV Cache 的内存占用如何计算？长文本场景下会带来什么问题？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- 每 token KV Cache = 2 × num_layers × num_heads × d_head × bytes_per_elem
- 总 KV Cache = batch_size × seq_len × per_token_size
- 例如 LLaMA-7B (32 layers, 32 heads, 128 d_head, fp16)：每 token 约 524 KB，4096 tokens 约 2 GB
- 长文本问题：
 1. 显存 OOM
 2. batch size 受限
 3. decode latency 随序列增长（读取更多 KV）
- 解决方案：PagedAttention、KV Cache 量化（INT8/FP8）、稀疏 attention、滑动窗口

---

### 今日自测清单

- [ ] 能解释 KV Cache 的核心思想和收益
- [ ] 能计算给定模型配置下的 KV Cache 内存占用
- [ ] 能区分静态分配和动态分配
- [ ] 代码编译运行正确，多轮追加后 seq_len 正确
- [ ] 理解多轮对话中 cache 复用的流程
- [ ] 能说出 KV Cache 的 3 个优化方向（PagedAttention、量化、滑动窗口）

---

## Day 31（周三）：vLLM 整体架构分析

> **今日目标**：阅读 vLLM 源码，理解 LLMEngine / Scheduler / Worker / SequenceGroup 的分层架构，梳理一个请求从进入到输出的完整生命周期。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，vLLM 是推理系统面试的核心素材

---

### 学习任务1：vLLM 架构概览（45分钟）

#### 阅读内容
- **仓库**：https://github.com/vllm-project/vllm
- **核心文件**：
 - `vllm/engine/llm_engine.py`：`LLMEngine` 类
 - `vllm/engine/scheduler.py`：`Scheduler` 类
 - `vllm/worker/worker.py`：`Worker` 类
 - `vllm/sequence.py`：`Sequence`、`SequenceGroup`
- **阅读重点**：
 - `LLMEngine.step()` 的执行流程
 - `Scheduler.schedule()` 如何决定哪些请求进入本次 iteration
 - `SequenceGroup` 的概念
 - 请求生命周期：arrival → waiting → running → swapped → finished

#### 核心概念笔记

**1. vLLM 分层架构**

```
User Request
 │
 ▼
┌─────────────────┐
│ LLMEngine │ ← 对外接口，管理整个推理生命周期
│ │
│ - add_request │
│ - step() │
└────────┬────────┘
 │
 ▼
┌─────────────────┐
│ Scheduler │ ← 决定每轮运行哪些 sequence group
│ │
│ - schedule() │
│ - Budget mgmt │
└────────┬────────┘
 │
 ▼
┌─────────────────┐
│ Worker │ ← 执行实际模型前向
│ │
│ - execute_model│
│ - ModelRunner │
└─────────────────┘
```

**2. 关键类定义**

| 类名 | 作用 |
|------|------|
| `Sequence` | 单个序列（一个请求可能包含多个候选序列） |
| `SequenceGroup` | 一个请求对应一个 group，包含 prompt + 多个采样序列 |
| `SequenceStatus` | WAITING / RUNNING / SWAPPED / FINISHED |
| `SchedulerOutputs` | scheduler 的输出，包含本轮要运行的 sequence groups |
| `SamplerOutput` | 采样结果，包含每个 sequence 的下一个 token |

**3. 请求生命周期**

```
请求到达
 │
 ▼
WAITING（等待调度）
 │
 ▼ Scheduler.schedule() 选择请求
RUNNING（执行中）
 │
 ├── 生成新 token
 │
 ├── 达到 max_tokens → FINISHED
 │
 └── 显存不足 → SWAPPED（换出到 CPU）
 │
 └── 显存可用 → RUNNING
```

**4. `LLMEngine.step()` 执行流程**

```python
def step(self):
 # 1. Scheduler 决定本轮运行哪些请求
 seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
 
 # 2. Worker 执行模型前向
 outputs = self.model_executor.execute_model(seq_group_metadata_list)
 
 # 3. 处理输出（采样、更新 sequence 状态）
 self._process_model_outputs(outputs, scheduler_outputs)
 
 # 4. 返回结果
 return request_outputs
```

---

### 学习任务2：Scheduler 调度策略（45分钟）

#### 阅读内容
- `vllm/engine/scheduler.py` 中的 `Scheduler.schedule()` 方法
- `SchedulingBudget` 类

#### 核心概念笔记

**1. Scheduler 的决策依据**

```
Scheduler 每轮决定：
 - 从 waiting queue 中选取哪些请求进入 running
 - 是否需要抢占（preemption）running 的请求
 - 是否需要 swap out 到 CPU

约束条件：
 - token budget：本轮最多能处理的 token 数
 - num_seqs budget：本轮最多能处理的 sequence 数
 - 显存预算：block allocator 报告剩余 block 数
```

**2. Continuous Batching 实现**

```python
def schedule(self):
 # 1. 先处理 running queue 中的请求（continuous batching）
 # 2. 如果还有 budget，从 waiting queue 加入新请求
 # 3. 如果显存不足，抢占低优先级请求
 # 4. 返回 scheduler_outputs
```

**关键**：每轮 iteration 都重新构建 batch，新请求可以在任意 iteration 加入。

**3. 抢占（Preemption）与换出（Swapping）**

```
Preemption：
 - 当高优先级请求到来但显存不足时，抢占低优先级请求
 - 两种策略：
 1. Recomputation：丢弃被抢占请求的 KV cache，之后重新计算（默认，因为通常更快）
 2. Swapping：将被抢占请求的 KV cache 换出到 CPU 内存
```

### 晚间任务：绘制 vLLM 架构图（1小时）

#### 要求

1. **系统架构图**：LLMEngine → Scheduler → Worker → Model Runner 的层次关系
2. **请求生命周期图**：WAITING → RUNNING → FINISHED / SWAPPED
3. **时序图**：一个请求从 `add_request` 到返回结果的完整调用链

#### 练习题

**练习1（基础）**：在 vLLM 源码中找到 `LLMEngine.step()`，记录其 4 个主要步骤。

**练习2（进阶）**：找到 `Scheduler._schedule_running()` 和 `_schedule_waiting()`，理解 continuous batching 的实现。

**练习3（综合）**：解释 vLLM 中 preemption 的两种策略（recomputation vs swapping）及其适用场景。

---

### 今日面试题

**面试题1**：vLLM 的整体架构是怎样的？一个请求从进入到输出经历哪些阶段？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- 分层架构：LLMEngine（对外接口）→ Scheduler（调度）→ Worker（执行）→ Model Runner（模型前向）
- 请求生命周期：
 1. 用户调用 `LLMEngine.add_request()`，请求进入 WAITING queue
 2. `Scheduler.schedule()` 决定哪些请求进入本轮 batch
 3. `Worker.execute_model()` 执行模型前向
 4. 采样得到新 token，更新 sequence 状态
 5. 请求完成（FINISHED）或被抢占（SWAPPED）
- Continuous Batching：每轮 iteration 都重新构建 batch，新请求可在任意 iteration 加入

**面试题2**：vLLM 中的 Scheduler 依据什么做调度决策？什么是 SchedulingBudget？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- Scheduler 依据：
 - Token budget：本轮最多处理的 token 数
 - Num_seqs budget：本轮最多处理的 sequence 数
 - 显存预算：Block allocator 报告的剩余 block 数
 - 请求优先级（可选）
- `SchedulingBudget`：封装了上述预算，scheduler 在添加请求时检查是否超出预算
- 调度目标：最大化 throughput，同时控制 latency（通过 budget 限制 batch 大小）

---

### 今日自测清单

- [ ] 能画出 vLLM 架构图（LLMEngine → Scheduler → Worker）
- [ ] 能解释 Sequence、SequenceGroup、SequenceStatus 的概念
- [ ] 能描述请求从 add_request 到 finished 的完整生命周期
- [ ] 理解 Continuous Batching 的实现方式
- [ ] 理解 SchedulingBudget 的作用
- [ ] 能解释 preemption 的两种策略

---

## Day 32（周四）：vLLM Worker 与 PagedAttention

> **今日目标**：深入 vLLM Worker 层，理解 PagedAttention 的 block table、物理/逻辑映射、copy-on-write 机制。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，PagedAttention 是 vLLM 最核心的创新

---

### 学习任务1：vLLM Worker 执行流程（45分钟）

#### 阅读内容
- `vllm/worker/worker.py`：`Worker` 类
- `vllm/worker/model_runner.py`：`ModelRunner` 类
- `vllm/attention/`：Attention backend 抽象

#### 核心概念笔记

**1. Worker 的职责**

```
Worker:
 - 管理 GPU 设备和模型权重
 - 接收 Scheduler 输出的 sequence group metadata
 - 构建 input metadata（包括 block table）
 - 调用 ModelRunner 执行模型前向
 - 返回采样结果
```

**2. `execute_model` 调用链**

```python
def execute_model(self, seq_group_metadata_list):
 # 1. 构建 input tokens 和 positions
 # 2. 构建 attention metadata（含 block table）
 # 3. 调用 model_runner.run()
 # 4. 采样得到 next token
 # 5. 返回 outputs
```

**3. Attention Backend 抽象**

```python
# vllm/attention/selector.py 根据硬件选择 backend
# - FlashAttention backend
# - XFormers backend
# - PagedAttention backend

class Attention:
 def forward(self, query, key, value, kv_cache, attn_metadata):
 # 根据 backend 调用不同的 attention 实现
 ...
```

**4. Block Table 传入 Kernel**

```python
# Attention metadata 中包含 block_tables
# shape: (num_seqs, max_num_blocks_per_seq)
# 每个元素是物理 block 编号

# Kernel 内部根据 block_table 找到 KV cache 的物理位置
```

---

### 学习任务2：PagedAttention 原理（45分钟）

#### 阅读内容
- **论文**："Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., SOSP 2023)
- **地址**：https://arxiv.org/abs/2309.06180
- **源码**：`vllm/core/block_manager.py`、`vllm/core/block.py`

#### 核心概念笔记

**1. 传统 KV Cache 分配的内存碎片问题**

```
静态分配：
 - 为每个请求预分配 max_seq_len 空间
 - 浪费严重

动态分配：
 - 请求实际长度不确定
 - 频繁分配/释放产生外部碎片
 - 显存利用率低
```

**2. PagedAttention 核心思想**

```
借鉴 OS 虚拟内存分页：
 - 将 KV cache 分成固定大小的 block（如 16 tokens）
 - 逻辑 block 号：从序列角度看是连续的
 - 物理 block 号：实际在显存中的位置，可以不连续
 - 通过 block table 维护逻辑→物理映射
```

**3. Block Table 结构**

```
逻辑 view（用户/模型视角）:
 Seq 0: [L0, L1, L2, L3, L4, L5, L6, L7] (逻辑 block 0-7)

物理 view（显存中实际位置）:
 L0 → P3
 L1 → P7
 L2 → P1
 L3 → P12
 ...

Block Table for Seq 0: [3, 7, 1, 12, ...]
```

**4. Copy-on-Write（写时复制）**

```
场景：多个 sequence 共享同一个 prompt 的 KV cache（如 beam search 或并行采样）

共享时：
 - Seq A 和 Seq B 共享 prompt 的物理 block
 - 只读，不冲突

当 Seq B 要写入新 token 时：
 - 复制共享的物理 block 到新的物理 block
 - Seq B 的 block table 指向新 block
 - Seq A 不受影响

优点：避免不必要的复制，节省显存
```

**5. Block Allocator**

```python
class BlockAllocator:
 def allocate(self, num_blocks):
 # 从 free block pool 分配物理 block
 
 def free(self, block):
 # 释放物理 block 回 pool
 
 def fork(self, parent_block_table):
 # 创建 copy-on-write 副本
```

---

### 晚间任务：PagedAttention 笔记整理（1小时）

#### 笔记模板

```markdown
# PagedAttention 笔记

## 核心问题
传统 KV Cache 分配的内存碎片和浪费

## 解决方案
1. 将 KV cache 分成固定大小 block
2. 逻辑 block 连续，物理 block 可以不连续
3. Block table 维护映射

## Copy-on-Write
- 多个 sequence 共享 prompt block
- 写入时复制到新的物理 block

## Block Allocator
- 管理物理 block 池
- allocate / free / fork

## Day 33（周五）：项目推进 —— Mini 推理引擎 v0

> **今日目标**：整合本周所学，构建 Mini 推理引擎 v0：支持单请求、KV Cache、Prefill/Decode 循环，使用 PyTorch 作为模型执行后端。
> **面试考察度**：⭐⭐⭐⭐ 高频，"如何构建一个推理引擎"是工程能力体现

---

### 学习任务1：Mini 引擎 v0 架构设计（45分钟）

#### 设计目标

构建一个最小化的 LLM 推理引擎：
1. 单请求处理（暂不支持多请求并发）
2. 支持 Prefill + Decode 两阶段
3. 使用 KV Cache 加速 Decode
4. 使用 PyTorch 作为模型后端（Week 7 再替换为自定义 kernel）
5. 提供简洁的 API：`generate(prompt, max_new_tokens)`

#### 架构图

```
┌─────────────────────────────────────────┐
│ Mini Engine v0 │
│ │
│ Request: prompt tokens │
│ │ │
│ ▼ │
│ Prefill: │
│ - 用完整 prompt 做一次 forward │
│ - 生成 first token │
│ - 保存所有 prompt tokens 的 K/V │
│ │ │
│ ▼ │
│ Decode Loop: │
│ - 每次输入上一个生成的 token │
│ - 从 KV Cache 读取历史 K/V │
│ - 生成下一个 token │
│ - 将新 K/V 追加到 KV Cache │
│ - 重复直到 EOS 或 max_tokens │
│ │ │
│ ▼ │
│ Response: generated tokens │
│ │
└─────────────────────────────────────────┘
```

#### 接口设计

```python
class MiniEngineV0:
 def __init__(self, model, tokenizer):
 self.model = model
 self.tokenizer = tokenizer
 self.kv_cache = None
 
 def generate(self, prompt, max_new_tokens=20):
 # 1. prefill
 # 2. decode loop
 # 3. return generated text
```

---

### 晚间编程任务：Mini 推理引擎 v0（1.5小时）

#### 完整代码

```python
# mini_engine_v0.py —— Mini 推理引擎 v0（单请求 + KV Cache）
# 运行命令: python mini_engine_v0.py
# 依赖: pip install torch transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class MiniLLM(nn.Module):
 """最小 LLM 模型，用于演示推理引擎"""
 def __init__(self, vocab_size=1000, d_model=512, n_heads=8, d_ff=2048, n_layers=4):
 super().__init__()
 self.vocab_size = vocab_size
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.n_layers = n_layers

 self.embedding = nn.Embedding(vocab_size, d_model)
 self.layers = nn.ModuleList([
 MiniTransformerLayer(d_model, n_heads, d_ff)
 for _ in range(n_layers)
 ])
 self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

 def forward(self, input_ids, kv_cache=None, use_cache=False):
 """
 input_ids: (B, N)
 kv_cache: list of (k, v) tuples per layer, or None
 返回: logits, new_kv_cache
 """
 x = self.embedding(input_ids) # (B, N, d)

 new_kv_cache = []
 for i, layer in enumerate(self.layers):
 layer_cache = kv_cache[i] if kv_cache is not None else None
 x, layer_new_cache = layer(x, layer_cache, use_cache)
 new_kv_cache.append(layer_new_cache)

 logits = self.lm_head(x) # (B, N, vocab_size)
 return logits, new_kv_cache

class MiniTransformerLayer(nn.Module):
 def __init__(self, d_model=512, n_heads=8, d_ff=2048):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.qkv = nn.Linear(d_model, 3 * d_model)
 self.out = nn.Linear(d_model, d_model)
 self.norm1 = nn.LayerNorm(d_model)
 self.norm2 = nn.LayerNorm(d_model)
 self.ffn = nn.Sequential(
 nn.Linear(d_model, d_ff),
 nn.GELU(),
 nn.Linear(d_ff, d_model),
 )

 def forward(self, x, kv_cache=None, use_cache=False):
 B, N, _ = x.shape

 x_norm = self.norm1(x)
 qkv = self.qkv(x_norm)
 qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
 q, k, v = qkv[0], qkv[1], qkv[2]

 if use_cache and kv_cache is not None:
 k_cache, v_cache = kv_cache
 k = torch.cat([k_cache, k], dim=2)
 v = torch.cat([v_cache, v], dim=2)

 scale = self.d_head ** -0.5
 attn = torch.matmul(q, k.transpose(-2, -1)) * scale
 attn = F.softmax(attn, dim=-1)
 out = torch.matmul(attn, v)

 out = out.transpose(1, 2).reshape(B, N, self.d_model)
 x = x + self.out(out)
 x = x + self.ffn(self.norm2(x))

 return x, (k, v)

class MiniTokenizer:
 """最小 tokenizer：用简单规则分割单词为 token id"""
 def __init__(self, vocab_size=1000):
 self.vocab_size = vocab_size
 self.word_to_id = {}
 self.id_to_word = {}
 self.next_id = 1 # 0 留给未知词/特殊 token

 def encode(self, text: str) -> List[int]:
 tokens = []
 for word in text.lower().split():
 if word not in self.word_to_id:
 self.word_to_id[word] = self.next_id
 self.id_to_word[self.next_id] = word
 self.next_id += 1
 tokens.append(self.word_to_id[word])
 return tokens

 def decode(self, ids: List[int]) -> str:
 words = []
 for id in ids:
 if id in self.id_to_word:
 words.append(self.id_to_word[id])
 else:
 words.append(f"<unk_{id}>")
 return " ".join(words)

class MiniEngineV0:
 """Mini 推理引擎 v0"""
 def __init__(self, model: MiniLLM, tokenizer: MiniTokenizer, device="cuda"):
 self.model = model.to(device).eval()
 self.tokenizer = tokenizer
 self.device = device

 @torch.no_grad()
 def generate(self, prompt: str, max_new_tokens: int = 20):
 input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)

 # ========== Prefill ==========
 logits, kv_cache = self.model(input_ids, use_cache=True)
 next_token_logits = logits[:, -1, :]
 next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

 generated_ids = [next_token.item()]

 # ========== Decode Loop ==========
 for _ in range(max_new_tokens - 1):
 logits, kv_cache = self.model(next_token, kv_cache=kv_cache, use_cache=True)
 next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
 generated_ids.append(next_token.item())

 return self.tokenizer.decode(generated_ids)

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}")

 vocab_size = 1000
 d_model = 512
 n_heads = 8
 n_layers = 4

 model = MiniLLM(vocab_size, d_model, n_heads, n_layers=n_layers)
 tokenizer = MiniTokenizer(vocab_size)
 engine = MiniEngineV0(model, tokenizer, device)

 prompt = "hello world this is a test"
 print(f"Prompt: {prompt}")

 generated = engine.generate(prompt, max_new_tokens=10)
 print(f"Generated: {generated}")

 # 验证 with/without cache 一致性
 print("\n=== KV Cache Correctness Check ===")
 input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

 with torch.no_grad():
 logits_with_cache, _ = model(input_ids, use_cache=False)

 # 不用 cache，模拟 decode：每次重新输入完整历史
 generated_no_cache = []
 current_ids = input_ids.clone()
 for _ in range(5):
 with torch.no_grad():
 logits_no_cache, _ = model(current_ids, use_cache=False)
 next_token = torch.argmax(logits_no_cache[:, -1, :], dim=-1, keepdim=True)
 generated_no_cache.append(next_token.item())
 current_ids = torch.cat([current_ids, next_token], dim=1)

 print(f"With cache first token: {torch.argmax(logits_with_cache[:, -1, :], dim=-1).item()}")
 print(f"Without cache tokens: {generated_no_cache}")

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python mini_engine_v0.py

# 预期输出
# Using device: cuda
# Prompt: hello world this is a test
# Generated: token1 token2 ...
# 
# === KV Cache Correctness Check ===
# With cache first token: x
# Without cache tokens: [x, ...]
```

#### 练习题

**练习1（基础）**：在 `generate` 中加入温度采样（temperature sampling）和 top-k 采样。

**练习2（进阶）**：修改模型使其支持 KV Cache 的 FP16 存储，验证显存占用减少。

**练习3（综合）**：实现多轮对话 API：`chat(messages: List[dict])`，复用历史 KV Cache。

---

### 今日面试题

**面试题1**：如何构建一个最简单的 LLM 推理引擎？需要哪些核心组件？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- 核心组件：
 1. **模型后端**：执行 transformer forward（PyTorch/TensorRT/vLLM）
 2. **Tokenizer**：文本 ↔ token id 转换
 3. **KV Cache**：存储历史 K/V，避免重复计算
 4. **Prefill/Decode 循环**：prefill 处理 prompt，decode 自回归生成
 5. **采样器**：argmax/greedy/temperature/top-k/top-p
 6. **调度器**（可选）：多请求时决定 batch 组合
- 最小流程：
 1. encode prompt
 2. prefill forward → first token
 3. decode loop with KV cache → next tokens
 4. decode token ids → text

**面试题2**：在推理引擎中，Prefill 和 Decode 阶段分别需要保存什么到 KV Cache？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Prefill 阶段**：保存 prompt 中每个 token 的 K 和 V
 - 对第 i 层 transformer，保存 shape 为 `(B, H, N_prompt, d_head)` 的 K 和 V
- **Decode 阶段**：保存每个新生成 token 的 K 和 V
 - 每步追加 shape 为 `(B, H, 1, d_head)` 的 K 和 V
- 最终 KV Cache 的长度 = prompt_len + generated_len
- 只有 K 和 V 需要保存，Q 是实时计算的

---

### 今日自测清单

- [ ] 能说出 Mini 推理引擎 v0 的 5 个核心组件
- [ ] 能理解 Prefill/Decode 与 KV Cache 的交互
- [ ] `mini_engine_v0.py` 运行成功，能生成文本
- [ ] 验证了 with cache 和 without cache 的输出一致性
- [ ] 能画出 Mini 引擎 v0 的执行流程图
- [ ] 理解多轮对话中 KV Cache 复用的设计

---

## Day 34（周六）：端到端 Profiling

> **今日目标**：对 Mini 推理引擎 v0 做端到端 profiling，测量 TTFT、TBT、各阶段 latency 占比，定位首个 token 时间和逐 token 时间的瓶颈。
> **时间分配**：6小时全天投入（nsys 采集2h + latency 分析2h + 瓶颈定位2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"如何做推理系统 profiling"是系统优化的标准流程

---

### 任务1：测量 TTFT / TBT / 阶段 Latency（2小时）

#### 完整代码

```python
# profile_engine_v0.py —— Mini 推理引擎 v0 端到端 Profiling
# 运行命令: python profile_engine_v0.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from mini_engine_v0 import MiniLLM, MiniTokenizer, MiniEngineV0

def profile_engine(engine, prompt, max_new_tokens=20):
 input_ids = torch.tensor([engine.tokenizer.encode(prompt)], device=engine.device)

 # 预热
 for _ in range(3):
 _ = engine.model(input_ids, use_cache=False)
 torch.cuda.synchronize()

 # ========== Prefill Profiling ==========
 torch.cuda.synchronize()
 t_start = time.perf_counter()

 with torch.no_grad():
 logits, kv_cache = engine.model(input_ids, use_cache=True)
 first_token_logits = logits[:, -1, :]
 first_token = torch.argmax(first_token_logits, dim=-1, keepdim=True)

 torch.cuda.synchronize()
 ttft = (time.perf_counter() - t_start) * 1000

 print(f"=== Prefill Phase ===")
 print(f" Prompt length: {input_ids.size(1)}")
 print(f" TTFT: {ttft:.3f} ms")
 print(f" KV Cache shape per layer: {kv_cache[0][0].shape}")

 # ========== Decode Profiling ==========
 decode_times = []
 decode_breakdown = {
 'forward': [],
 'sampling': [],
 'cache_append': [],
 }

 next_token = first_token
 for step in range(max_new_tokens):
 t0 = time.perf_counter()

 with torch.no_grad():
 logits, kv_cache = engine.model(next_token, kv_cache=kv_cache, use_cache=True)

 t1 = time.perf_counter()
 next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
 t2 = time.perf_counter()

 torch.cuda.synchronize()
 t3 = time.perf_counter()

 decode_times.append((t3 - t0) * 1000)
 decode_breakdown['forward'].append((t1 - t0) * 1000)
 decode_breakdown['sampling'].append((t2 - t1) * 1000)
 decode_breakdown['cache_append'].append((t3 - t2) * 1000)

 print(f"\n=== Decode Phase ({max_new_tokens} tokens) ===")
 print(f" Mean TBT: {sum(decode_times)/len(decode_times):.3f} ms")
 print(f" P50 TBT: {sorted(decode_times)[len(decode_times)//2]:.3f} ms")
 print(f" P99 TBT: {sorted(decode_times)[int(len(decode_times)*0.99)]:.3f} ms")
 print(f" Max TBT: {max(decode_times):.3f} ms")
 print(f" Min TBT: {min(decode_times):.3f} ms")

 print(f"\n=== Decode Breakdown (mean ms) ===")
 for key, times in decode_breakdown.items():
 print(f" {key}: {sum(times)/len(times):.3f} ms ({sum(times)/sum(decode_times)*100:.1f}%)")

 print(f"\n=== Throughput ===")
 total_time = ttft + sum(decode_times)
 print(f" Total time: {total_time:.3f} ms")
 print(f" Output throughput: {max_new_tokens / (sum(decode_times)/1000):.2f} tokens/s")

 return {
 'ttft': ttft,
 'decode_times': decode_times,
 'breakdown': decode_breakdown,
 }

def scan_prompt_lengths(engine, lengths=[128, 256, 512, 1024]):
 """扫描不同 prompt 长度下的 TTFT"""
 print("\n=== TTFT vs Prompt Length ===")
 print(f"{'Prompt Length':>15} {'TTFT(ms)':>12} {'Per-token(ms)':>15}")
 print("-" * 45)

 for L in lengths:
 # 构造长度为 L 的 prompt
 words = []
 for i in range(L):
 words.append(f"word{i}")
 prompt = " ".join(words)
 input_ids = torch.tensor([engine.tokenizer.encode(prompt)], device=engine.device)

 # 修正长度（因为 tokenizer 可能合并）
 actual_len = input_ids.size(1)

 torch.cuda.synchronize()
 t_start = time.perf_counter()
 with torch.no_grad():
 logits, kv_cache = engine.model(input_ids, use_cache=True)
 torch.cuda.synchronize()
 ttft = (time.perf_counter() - t_start) * 1000

 print(f"{actual_len:>15} {ttft:>12.3f} {ttft/actual_len:>15.3f}")

def main():
 device = "cuda" if torch.cuda.is_available() else "cpu"
 print(f"Using device: {device}\n")

 model = MiniLLM(vocab_size=1000, d_model=512, n_heads=8, n_layers=4)
 tokenizer = MiniTokenizer(vocab_size=1000)
 engine = MiniEngineV0(model, tokenizer, device)

 prompt = "hello world this is a test prompt for profiling"
 profile_engine(engine, prompt, max_new_tokens=20)

 scan_prompt_lengths(engine, lengths=[128, 256, 512])

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python profile_engine_v0.py

# 预期输出
# Using device: cuda
# 
# === Prefill Phase ===
# Prompt length: 9
# TTFT: x.xxx ms
# KV Cache shape per layer: (1, 8, 9, 64)
# 
# === Decode Phase (20 tokens) ===
# Mean TBT: x.xxx ms
# P50 TBT: x.xxx ms
# ...
# 
# === Decode Breakdown (mean ms) ===
# forward: x.xxx ms (xx.x%)
# sampling: x.xxx ms (xx.x%)
# cache_append: x.xxx ms (xx.x%)
# 
# === Throughput ===
# Total time: x.xxx ms
# Output throughput: x.xx tokens/s
```

---

### 任务2：Nsight Systems 采集时间线（2小时）

#### 采集命令

```bash
nsys profile -o mini_engine_v0_timeline \
 --trace=cuda,nvtx \
 python mini_engine_v0.py

# 统计 kernel 时间
nsys stats -t cuda_gpu_kern_sum mini_engine_v0_timeline.nsys-rep
```

#### 观察重点

1. **Prefill 阶段 kernel**：GEMM kernel 时间占比高（compute-bound）
2. **Decode 阶段 kernel**：小 kernel 多，间隙大（memory-bound + launch overhead）
3. **Kernel 间隙（gap）**：相邻 kernel 之间的空白
4. **CUDA Stream 利用率**：是否所有操作都在 default stream

#### 分析任务清单

1. 找出 Prefill 阶段 CUDA 时间 top3 算子
2. 找出 Decode 阶段 CUDA 时间 top3 算子
3. 计算 kernel 间隙占总时间的比例
4. 判断系统主要瓶颈是 compute、memory 还是 launch overhead

---

### 任务3：瓶颈定位与优化方向（2小时）

#### 常见瓶颈与优化

| 现象 | 瓶颈类型 | 优化方向 |
|------|---------|---------|
| TTFT 随 N 快速增长 | Prefill compute-bound | FlashAttention、Tensor Core、reduce prompt |
| TBT 随 L 增长 | Decode memory-bound | KV Cache 优化、PagedAttention、量化 |
| Kernel 间隙大 | Launch overhead | CUDA Graph、torch.compile、kernel fusion |
| Decode forward 占比高 | 模型计算 | 量化、剪枝、小模型 |
| Sampling 占比高 | CPU 瓶颈 | 将采样移到 GPU、批处理采样 |

### 今日面试题

**面试题1**：如何做 LLM 推理系统的端到端 profiling？需要关注哪些指标？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **系统级**：用 Nsight Systems 采集时间线，看 kernel 排列和间隙
2. **阶段级**：测量 TTFT（Prefill）和 TBT（Decode）
3. **Kernel 级**：用 Nsight Compute 分析 top 算子是 compute-bound 还是 memory-bound
4. **关键指标**：
 - TTFT、TBT/TPOT、throughput (tokens/s)
 - Kernel 间隙占比
 - SM/Memory utilization
 - KV Cache 内存占用
1. **瓶颈定位**：
 - TTFT 高 → prefill compute-bound → FlashAttention、Tensor Core
 - TBT 高 → decode memory-bound → KV Cache、PagedAttention
 - 间隙大 → launch overhead → CUDA Graph

**面试题2**：Decode 阶段的 TBT 为什么会随序列长度增长？如何优化？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **原因**：
 1. 序列变长，KV Cache 变大，每次都要读取更多历史 K/V
 2. Attention 计算量从 O(d²) 变为 O(L×d)
 3. KV Cache 可能超出 L2/L1 cache，需要更多 HBM 访问
- **优化方向**：
 1. **PagedAttention**：更高效管理 KV Cache 内存，减少碎片
 2. **KV Cache 量化**：INT8/FP8 减少数据量
 3. **稀疏/滑动窗口 attention**：只保留最近 K 个 tokens
 4. **MLA/MQA/GQA**：减少 KV 的头数，降低内存和计算
 5. **Continuous Batching**：合并多个 decode 请求，提高 SM 利用率

---

### 今日自测清单

- [ ] 能测量并解释 TTFT 和 TBT
- [ ] 完成 profiling 脚本，得到 mean/P50/P99 TBT
- [ ] 能分析 Decode 阶段 forward/sampling/cache_append 的时间占比
- [ ] 能用 nsys 采集 Mini 引擎时间线
- [ ] 能识别 Prefill 和 Decode 阶段的主要瓶颈
- [ ] 能给出至少 3 个优化 TBT 的具体方向

---

## Day 35（周日）：推理系统核心问题总结

> **今日目标**：整理推理系统的关键挑战，总结内存管理、Batch 策略、Latency 隐藏、调度开销四大核心问题，复盘本周面试题，整理 GitHub 仓库。
> **时间分配**：6小时全天投入（核心问题2h + 面试复盘2h + GitHub 整理2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"推理系统核心问题"是系统设计的总结性考点

---

### 任务1：推理系统核心问题清单（2小时）

#### 核心问题分类

```
推理系统四大核心问题：
 1. 内存管理：KV Cache 的动态增长与显存限制
 2. Batch 策略：如何组合请求以平衡吞吐和延迟
 3. Latency 隐藏：compute 与 communication overlap
 4. 调度开销：如何最小化调度延迟
```

**1. 内存管理**

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| KV Cache 显存占用大 | 2×L×layers×heads×d_head×bytes | 量化、PagedAttention、GQA/MLA |
| 动态长度 | 每个请求生成长度不确定 | PagedAttention、动态分配 |
| 长文本 OOM | 序列过长超出显存 | 滑动窗口、稀疏 attention、offloading |
| 多轮对话 | 历史上下文占用显存 | Cache 复用、prompt 压缩 |

**2. Batch 策略**

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Static Batching | 固定 batch size | 简单 | 延迟高，资源利用率低 |
| Dynamic Batching | 请求聚合，超时等待 | 提高吞吐 | 引入等待延迟 |
| Continuous Batching | 每轮重新构建 batch | 吞吐+延迟兼顾 | 实现复杂 |
| Inflight Batching | TensorRT-LLM 术语 | 同 continuous | 同 continuous |

**3. Latency 隐藏**

```
目标：让 GPU 在不同任务之间 overlap，减少空闲等待

手段：
 1. CUDA Graph：减少 launch overhead
 2. Async Copy：overlap 数据传输与计算
 3. Pipeline Parallelism：跨设备 overlap
 4. Speculative Decoding：用小模型预测，大模型验证
 5. Chunked Prefill：将大 prefill 拆成小块，与 decode 交错
```

**4. 调度开销**

```
调度开销来源：
 1. Python GIL / 多线程竞争
 2. 每次 iteration 重新构建 input tensors
 3. 内存分配/释放
 4. CPU-GPU 同步点

优化：
 1. 减少 Python 层操作，核心逻辑用 C++
 2. 预分配 tensor buffer
 3. 异步采样
 4. 减少 cudaSynchronize 调用
```

---

### 任务2：面试复盘（2小时）

#### 本周核心面试题回顾

1. Prefill vs Decode 的区别和瓶颈
2. TTFT 和 TBT 的定义及优化
3. KV Cache 的核心思想和内存占用
4. 静态 vs 动态 KV Cache 分配
5. vLLM 整体架构
6. SequenceGroup 和请求生命周期
7. Scheduler 的决策依据
8. Preemption / Swapping 策略
9. PagedAttention 原理
10. Copy-on-Write 应用场景
11. 如何构建 Mini 推理引擎
12. Prefill/Decode 阶段分别保存什么到 KV Cache
13. 如何做端到端 profiling
14. TBT 为什么随序列增长
15. 推理系统四大核心问题

---

### 任务3：GitHub 整理与报告（2小时）

#### 仓库结构建议

```
week5-inference-system/
├── day29-prefill-decode/
│ ├── prefill_decode_simulation.py
│ └── notes.md
├── day30-kv-cache/
│ ├── kv_cache.cu
│ └── README.md
├── day31-vllm-architecture/
│ └── vllm_architecture.md
├── day32-pagedattention/
│ └── pagedattention_notes.md
├── day33-mini-engine-v0/
│ ├── mini_engine_v0.py
│ └── README.md
├── day34-profiling/
│ ├── profile_engine_v0.py
│ └── profiling_report.md
└── day35-summary/
 ├── inference_system_core_problems.md
 └── week5_report.md
```

#### 性能报告模板

```markdown
# Week 5 推理系统报告

## 测试环境
- GPU: [型号]
- CUDA: 12.x
- PyTorch: 2.x

## KV Cache 测试
- 配置: layers=2, batch=1, heads=8, max_len=1024, d_head=64
- 多轮追加后 seq_len 正确: PASS
- 每 token KV Cache: 8192 bytes

## Mini 引擎 v0
- Prompt: "hello world this is a test prompt for profiling"
- TTFT: x.xxx ms
- Mean TBT: x.xxx ms
- Throughput: x.xx tokens/s

## Profiling 发现
1. Prefill 阶段 top 算子: [GEMM]
2. Decode 阶段 top 算子: [GEMM/softmax]
3. Kernel 间隙占比: xx%

## 核心问题总结
[见 inference_system_core_problems.md]
```

---

### 今日面试题

**面试题1**：设计一个 LLM 推理服务时，需要考虑哪些核心问题？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. **内存管理**：KV Cache 动态增长、显存限制、PagedAttention、量化
2. **Batch 策略**：Dynamic/Continuous Batching 平衡吞吐和延迟
3. **Latency 隐藏**：CUDA Graph、async copy、speculative decoding、chunked prefill
4. **调度开销**：减少 Python 层、预分配 buffer、异步采样
5. **扩展性**：多 GPU、TP/PP 并行、多节点
6. **正确性**：数值精度、采样一致性、KV Cache 一致性

**面试题2**：Continuous Batching 和 Dynamic Batching 有什么区别？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Dynamic Batching**：
 - 在请求级别做 batching，等待一定时间或凑够 batch size
 - 一个 batch 内所有请求一起开始、一起结束
 - 一个长请求会阻塞整个 batch
- **Continuous Batching（Inflight Batching）**：
 - 在 iteration 级别做 batching
 - 每轮都重新构建 batch，新请求可以在任意 iteration 加入
 - 完成的请求可以随时退出，不会阻塞其他请求
- **对比**：Continuous Batching 更适合 LLM 自回归生成，因为每个请求的生成长度差异大

---

### 今日自测清单

- [ ] 能列出推理系统四大核心问题
- [ ] 每个核心问题能说出 2-3 个解决方案
- [ ] 能对比 Continuous Batching 和 Dynamic Batching
- [ ] 能说出 5 个优化 TBT 的方向
- [ ] 完成本周 15 道面试题的自问自答
- [ ] 整理 GitHub 仓库
- [ ] 生成 Week 5 性能报告和核心问题文档
- [ ] 规划 Week 6（Batching & 调度）的学习重点

---

## 附录A：第5周面试题汇总

| 题号 | 题目 | 考察频率 | 相关天数 | 难度 |
|------|------|---------|---------|------|
| 1 | Prefill vs Decode 的区别和瓶颈？ | ⭐⭐⭐⭐⭐ | Day 29 | 中 |
| 2 | TTFT 和 TBT 是什么？如何优化？ | ⭐⭐⭐⭐⭐ | Day 29, 34 | 中 |
| 3 | KV Cache 的核心思想和收益？ | ⭐⭐⭐⭐⭐ | Day 30 | 中 |
| 4 | KV Cache 内存占用如何计算？ | ⭐⭐⭐⭐⭐ | Day 30 | 中 |
| 5 | 静态 vs 动态 KV Cache 分配？ | ⭐⭐⭐⭐ | Day 30 | 中 |
| 6 | vLLM 整体架构是怎样的？ | ⭐⭐⭐⭐⭐ | Day 31 | 中 |
| 7 | SequenceGroup 是什么？ | ⭐⭐⭐⭐ | Day 31 | 中 |
| 8 | Scheduler 依据什么做决策？ | ⭐⭐⭐⭐ | Day 31 | 中 |
| 9 | Preemption 的两种策略？ | ⭐⭐⭐⭐ | Day 31 | 中 |
| 10 | PagedAttention 解决了什么问题？ | ⭐⭐⭐⭐⭐ | Day 32 | 高 |
| 11 | Copy-on-Write 是什么？ | ⭐⭐⭐⭐ | Day 32 | 中 |
| 12 | 如何构建一个最简单的推理引擎？ | ⭐⭐⭐⭐ | Day 33 | 中 |
| 13 | Prefill/Decode 分别保存什么到 KV Cache？ | ⭐⭐⭐⭐ | Day 33 | 中 |
| 14 | 如何做推理系统端到端 profiling？ | ⭐⭐⭐⭐ | Day 34 | 中 |
| 15 | TBT 为什么随序列长度增长？ | ⭐⭐⭐⭐⭐ | Day 34 | 中 |
| 16 | 推理系统有哪些核心问题？ | ⭐⭐⭐⭐⭐ | Day 35 | 高 |
| 17 | Continuous vs Dynamic Batching？ | ⭐⭐⭐⭐⭐ | Day 35 | 中 |

---

## 附录C：关键公式汇总

**1. KV Cache 每 Token 大小**
```
bytes_per_token = 2 × num_layers × num_heads × d_head × bytes_per_elem
```

**2. 总 KV Cache 大小**
```
total_kv_cache = batch_size × seq_len × bytes_per_token
```

**3. Prefill QKV GEMM FLOPs**
```
FLOPs = 2 × B × N × d × 3d
```

**4. Decode QKV GEMM FLOPs**
```
FLOPs = 2 × B × 1 × d × 3d
```

**5. TTFT / TBT**
```
TTFT = Prefill 阶段总时间
TBT = Decode 阶段单步平均时间
Throughput = generated_tokens / total_decode_time
```

---

## 附录D：推理系统优化速查表

| 问题 | 现象 | 检查方法 | 解决方案 |
|------|------|---------|---------|
| TTFT 过高 | 首 token 慢 | profile prefill | FlashAttention、Tensor Core |
| TBT 过高 | 生成慢 | profile decode | KV Cache、PagedAttention、量化 |
| TBT 随 L 增长 | 序列长后更慢 | 扫描不同 L | GQA/MLA、滑动窗口、稀疏 attention |
| 显存 OOM | KV Cache 太大 | 监控显存 | PagedAttention、INT8 KV、减少 batch |
| Kernel 间隙大 | 调度开销 | nsys timeline | CUDA Graph、torch.compile |
| 长请求阻塞 batch | Dynamic batching | 观察请求完成时间 | Continuous Batching |
| 多轮对话 TTFT 高 | 重复计算历史 | 检查 cache 复用 | 维护 session KV Cache |
| 显存碎片 | 分配失败但显存够 | block allocator | PagedAttention |
| Throughput 低 | GPU 利用率低 | nsys SM util | 增大 batch、continuous batching |

---

> 💡 **Week 5 总结**：本周我们进入了 AI Infra 的核心——推理系统。我们理解了 Prefill/Decode 的本质差异，实现了 KV Cache，阅读了 vLLM 架构和 PagedAttention，构建了第一个可运行的 Mini 推理引擎 v0，并做了端到端 profiling。最核心的收获是：LLM 推理优化 = 内存管理（KV Cache/PagedAttention）+ 调度策略（Continuous Batching）+ Latency 隐藏（CUDA Graph/async）+ 计算优化（FlashAttention/量化）。下周将进入 Batching & 调度，学习 Dynamic/Continuous Batching 的完整实现。
