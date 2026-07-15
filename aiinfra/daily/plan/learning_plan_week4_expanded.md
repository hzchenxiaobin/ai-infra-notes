# 第4周深度展开：FlashAttention 深挖与 IO 优化（7天）

> **适用对象**：陈斌斌（已完成第3周学习，掌握 Transformer 执行流程、手写 Softmax/LayerNorm、标准 Attention IO 分析）
> **本周目标**：从算法原理到 CUDA 实现完整掌握 FlashAttention，理解 IO 优化的核心思想，能在 Mini 引擎中替换标准 Attention
> **时间投入**：工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h
> **周日里程碑**：手写完整 FlashAttention Forward Kernel 支持 batch/multi-head，与标准 Attention 对比长序列（N=4096）加速 2x+，产出 IO 优化方法论报告

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 深入理解 FlashAttention 的 tiling + online softmax 原理，手写支持 batch/multi-head 的完整 Forward Kernel，集成到 Mini 推理引擎，建立 IO 优化的系统方法论 |
| **核心产出** | ① FlashAttention 论文精读笔记 ② Online Softmax 完整推导 ③ 完整 Forward Kernel（`flash_attention_v2.cu`）④ 官方源码分析报告 ⑤ FlashAttention-2 差异总结 ⑥ Mini 引擎 Attention 替换版 ⑦ 性能对比报告 ⑧ IO 优化方法论 checklist |
| **验收标准** | ① 能白板推导 online softmax 三公式 ② 手写 Kernel 在 N=4096, d=64 时与标准 Attention 误差 < 1e-3 且加速 2x+ ③ 能解释 FlashAttention-2 比 FA1 快的 3 个原因 ④ 集成到 Mini 引擎后端到端正确 ⑤ 能用 ncu 验证 HBM 访问量随 N 线性增长 |
| **面试准备** | 积累10-12道 FlashAttention 专题面试题，覆盖 online softmax、tiling、FA1/FA2 差异、IO 复杂度、工程集成五大主题 |

### 本周知识图谱

```
Day 22: FlashAttention 论文精读 → Online Softmax 三公式推导 + IO 复杂度对比
 ↓
Day 23: 手写完整 Forward Kernel → batch/multi-head + shared memory tiling
 ↓
Day 24: 官方 CUDA 源码分析 → flash_fwd_kernel.h / 分块策略 / warp 分配
 ↓
Day 25: FlashAttention-2 论文 → 减少 non-matmul FLOPs / work partitioning
 ↓
Day 26: 项目推进 → Mini 引擎集成 FlashAttention，替换标准 Attention
 ↓
Day 27: 性能对比 → 标准 vs 手写 vs 官方 / 不同 N/B/H 扫描 / HBM 验证
 ↓
Day 28: IO 优化方法论总结 → 提炼通用策略 + 面试复盘 + GitHub 整理
```

### 前置准备清单

#### 硬件/软件验证
- [ ] 已完成第3周所有 Coding 任务（标准 Attention IO 分析、Mini 引擎集成）
- [ ] `ncu --version` 正常（Week 2 Day 11 已验证）
- [ ] PyTorch 可用且 `torch.__version__ >= 2.0`
- [ ] 已阅读 Week 2 Day 12 的 FlashAttention 简化版笔记
- [ ] 已阅读 FlashAttention 论文 Section 1-3（至少一遍）

#### 验证命令
```bash
# 验证 PyTorch + CUDA
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
# 预期输出：torch 2.x.x cuda 12.x available True

# 验证 Week 3 代码路径存在
ls week3/day18-attention-io/attention_naive.cu
# 预期：文件存在

# 验证 ncu 可用
ncu --version
# 预期输出：NVIDIA Nsight Compute 202x.x.x
```

---

## Day 22（周一）：FlashAttention 论文精读与 Online Softmax 完整推导

> **今日目标**：精读 FlashAttention 论文，完整推导 online softmax 三公式，从理论上理解为什么 HBM 访问能从 O(N²) 降到 O(Nd)。
> **时间分配**：早间1.5h（论文精读1h + 公式推导30min）+ 晚间1h（编程实践）
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，FlashAttention 是推理系统面试第一考点

---

### 学习任务1：FlashAttention 论文核心思想（45分钟）

#### 阅读内容
- **论文**："FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., NeurIPS 2022)
- **地址**：https://arxiv.org/abs/2205.14135
- **阅读范围**：
 - Section 1: Introduction（标准 Attention 的内存问题）
 - Section 2: Background（Attention 定义、GPU 内存层次、IO 复杂度）
 - Section 3: FlashAttention（3.1 Tiling、3.2 Online Softmax、3.3 IO 复杂度分析）
- **辅助阅读**：
 - Princeton NLP 图解博客：https://princeton-nlp.github.io/flash-attention-blog/
 - Week 2 Day 12 的简化版 FlashAttention 笔记

#### 核心概念笔记

**1. 标准 Attention 的 IO 痛点**

```
标准 Attention：
 S = Q × K^T (N×N) → 写入 HBM
 P = softmax(S) (N×N) → 写入 HBM
 O = P × V (N×d) → 写入 HBM

HBM 访问量（当 N >> d 时）：
 读写 S: 2N²
 读写 P: 2N²
 读写 Q,K,V,O: O(Nd)
 总计: O(N²)
```

**关键洞察**：softmax 和第二个 GEMM 之间必须物化 P，因为：
- GEMM 库（cuBLAS）通常要求输入是连续内存中的矩阵
- softmax 算子和 GEMM 算子之间没有原生融合接口
- 结果导致 N×N 的 P 矩阵必须写回 HBM 再读出

**2. FlashAttention 的两个核心创新**

| 创新点 | 解决的问题 | 关键思想 |
|--------|-----------|---------|
| **Tiling** | SRAM 容量有限，放不下完整的 Q/K/V | 将 Q/K/V 分成小 tile，逐块加载到 SRAM |
| **Online Softmax** | 分块后无法得到全局 max | 维护 running max/sum/output，递推更新 |

**3. SRAM 容量约束决定分块大小**

```
每 Block 需要的 SRAM：
 Q tile: Br × d
 K tile: Bc × d
 V tile: Bc × d
 S tile: Br × Bc
 总计: Br×d + 2×Bc×d + Br×Bc ≤ SRAM_per_SM

以 RTX 5090 为例，shared memory 上限 164 KB/SM：
 d=64, Br=128, Bc=128:
 128×64 + 2×128×64 + 128×128 = 8192 + 16384 + 16384 = 40960 floats = 160 KB ✓
```

### 学习任务2：Online Softmax 三公式完整推导（45分钟）

#### 推导目标

标准 softmax：
```
yi = exp(xi - m) / l
where m = max(xj), l = Σ exp(xj - m)
```

分块计算时，每个 KV tile 只能看到部分 xj，无法直接得到全局 m 和 l。Online softmax 通过维护 running 状态解决。

#### 状态定义

- `m`：已处理所有块的 running maximum
- `l`：已处理所有块的 running sum（以 m 为参考点）
- `o`：已处理所有块的 running output（部分加权和）

初始状态：`m = -∞, l = 0, o = 0`

#### 推导过程

**处理新块前**：已有全局状态 `(m, l, o)`，旧参考点是 `m`。

**新块到来**：新块的 score 为 `xj`，value 为 `vj`。

**Step 1：计算新全局 max**
```
m_new = max(m, max(xj))
```

**Step 2：将旧 sum 缩放到新参考点**

旧 sum 以 `m` 为参考：`l = Σ exp(x_old - m)`

要转换到以 `m_new` 为参考：
```
exp(x_old - m_new) = exp(x_old - m) × exp(m - m_new)

所以新的旧部分和 = l × exp(m - m_new)
```

新块的和：
```
Σ exp(xj - m_new)
```

合并：
```
l_new = l × exp(m - m_new) + Σ exp(xj - m_new)
```

**Step 3：更新 running output**

旧的 output 是以旧概率分布加权的：
```
o = Σ [exp(x_old - m) / l] × v_old
```

新概率分布下，旧部分的权重应为：
```
exp(x_old - m_new) / l_new = [exp(x_old - m) × exp(m - m_new)] / l_new
 = [exp(x_old - m) / l] × [l × exp(m - m_new) / l_new]
```

所以旧 output 需要乘以缩放因子：
```
o_scale = l × exp(m - m_new) / l_new
```

新块的贡献：
```
new_contrib = Σ [exp(xj - m_new) / l_new] × vj
```

最终：
```
o_new = o × o_scale + new_contrib
 = o × (l × exp(m - m_new) / l_new) + Σ (exp(xj - m_new) / l_new) × vj
```

#### 三个公式汇总

```
公式1（Max 更新）: m_new = max(m, max(xj))

公式2（Sum 更新）: l_new = l × exp(m - m_new) + Σ exp(xj - m_new)

公式3（Output 更新）:
 o_new = o × (l × exp(m - m_new) / l_new) + Σ (exp(xj - m_new) / l_new) × vj
```

**最终输出**：所有 KV tile 处理完后，`O = o / l`？不对！注意我们的 `o` 定义已经是按最终 softmax 概率加权的结果（因为每次更新都除以了 `l_new`），所以最终直接 `O = o`。

> 有些版本会维护未归一化的 `o`（即不除以 l_new），最后做一次 `O = o / l`。两种写法等价，工程上常见的是每次归一化，最后直接输出。

#### 数值稳定性要点

- `exp(m - m_new)` 中 `m_new >= m`，所以指数 ≤ 0，不会溢出
- `m_new` 是全局 max，新块的 `exp(xj - m_new) ≤ 1`
- 即使 `m_new = m`（新块没有更大的值），`exp(0) = 1`，公式退化为简单累加

---

### 学习任务3：IO 复杂度对比（30分钟）

#### 标准 Attention vs FlashAttention

| 实现 | HBM 访问量 | N=4096, d=64, FP32 | N=8192, d=64 |
|------|-----------|-------------------|--------------|
| 标准 Attention | O(N² + Nd) | ~206 MB | ~805 MB |
| FlashAttention | O(Nd) | ~2 MB | ~4 MB |
| **加速比（IO 角度）** | | **~100x** | **~200x** |

> 实际 wall-clock 加速通常 2-8x，因为标准 Attention 的 GEMM 部分也是 compute-bound，不能完全被 IO 限制。

#### 为什么实际加速没有 IO 加速那么大？

```
标准 Attention 时间 = max(T_gemm, T_memory)
 - T_gemm: 由 Tensor Core 决定，与 FLOPs 成正比
 - T_memory: 由 HBM 带宽决定

FlashAttention 时间 ≈ T_gemm（因为 IO 不再是瓶颈）

如果原始 T_gemm ≈ T_memory，加速比 ≈ 2x
如果原始 T_memory >> T_gemm，加速比 ≈ 8x+
```

所以 FlashAttention 在长序列、小 head dim（d 较小）时收益最大。

---

### 晚间编程任务：PyTorch 标准 Attention vs FlashAttention 对比（1小时）

#### 完整代码

```python
# compare_attention_io.py —— 标准 Attention vs FlashAttention IO 与速度对比
# 运行命令: python compare_attention_io.py
# 依赖: pip install torch

import torch
import torch.nn.functional as F
import math
import time

def standard_attention(Q, K, V):
 """标准 Attention，物化 S 和 P"""
 d = Q.size(-1)
 S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
 P = F.softmax(S, dim=-1)
 O = torch.matmul(P, V)
 return O

def flash_attention_pytorch(Q, K, V, Br=64, Bc=64):
 """
 纯 PyTorch 实现的 FlashAttention 算法（教学版，不追求速度）
 用于验证 online softmax 正确性
 """
 N, d = Q.size(-2), Q.size(-1)
 scale = 1.0 / math.sqrt(d)
 O = torch.zeros_like(Q)
 L = torch.zeros(Q.size()[:-1] + (1,), device=Q.device, dtype=Q.dtype)

 for q_start in range(0, N, Br):
 q_end = min(q_start + Br, N)
 Qi = Q[..., q_start:q_end, :] * scale

 m = torch.full((Qi.size()[:-1] + (1,)), -1e30, device=Q.device, dtype=Q.dtype)
 l = torch.zeros(Qi.size()[:-1] + (1,), device=Q.device, dtype=Q.dtype)
 o = torch.zeros_like(Qi)

 for kv_start in range(0, N, Bc):
 kv_end = min(kv_start + Bc, N)
 Kj = K[..., kv_start:kv_end, :]
 Vj = V[..., kv_start:kv_end, :]

 Sij = torch.matmul(Qi, Kj.transpose(-2, -1))

 # Online softmax update
 mij = torch.max(Sij, dim=-1, keepdim=True).values
 m_new = torch.max(m, mij)

 # Scale old l and o
 l_scale = torch.exp(m - m_new)
 l_new = l * l_scale + torch.sum(torch.exp(Sij - m_new), dim=-1, keepdim=True)

 # Compute P weights for new block
 Pij = torch.exp(Sij - m_new) / l_new

 # Scale old o and add new contribution
 o = o * (l * l_scale / l_new) + torch.matmul(Pij, Vj)

 m = m_new
 l = l_new

 O[..., q_start:q_end, :] = o

 return O

def benchmark(func, Q, K, V, name, n_iter=10):
 # warmup
 for _ in range(3):
 _ = func(Q, K, V)
 torch.cuda.synchronize()

 start = torch.cuda.Event(enable_timing=True)
 end = torch.cuda.Event(enable_timing=True)
 start.record()
 for _ in range(n_iter):
 out = func(Q, K, V)
 end.record()
 torch.cuda.synchronize()
 ms = start.elapsed_time(end) / n_iter
 print(f"{name}: {ms:.3f} ms")
 return out

def main():
 torch.manual_seed(42)
 device = "cuda"
 dtype = torch.float32
 d = 64
 seq_lens = [512, 1024, 2048, 4096]

 print("=== Attention IO & Speed Comparison ===")
 print(f"head dim d={d}, FP32\n")

 for N in seq_lens:
 print(f"--- N={N} ---")
 Q = torch.randn(1, 1, N, d, device=device, dtype=dtype)
 K = torch.randn(1, 1, N, d, device=device, dtype=dtype)
 V = torch.randn(1, 1, N, d, device=device, dtype=dtype)

 # 正确性验证
 O_std = standard_attention(Q, K, V)
 O_fa = flash_attention_pytorch(Q, K, V)
 max_diff = (O_std - O_fa).abs().max().item()
 print(f"Max diff (standard vs flash): {max_diff:.2e}")

 # 速度对比
 benchmark(standard_attention, Q, K, V, "Standard Attention")
 benchmark(flash_attention_pytorch, Q, K, V, "FlashAttention (PyTorch)")

 # 理论 IO 对比
 bytes_per_elem = 4
 std_io = (3 * N * N + 4 * N * d) * bytes_per_elem / (1024 * 1024)
 fa_io = (4 * N * d) * bytes_per_elem / (1024 * 1024)
 print(f"Theoretical HBM IO: Standard={std_io:.2f} MB, FlashAttention={fa_io:.2f} MB, ratio={std_io/fa_io:.1f}x\n")

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python compare_attention_io.py

# 预期输出
# === Attention IO & Speed Comparison ===
# head dim d=64, FP32
# 
# --- N=512 ---
# Max diff (standard vs flash): x.xx e-06
# Standard Attention: x.xxx ms
# FlashAttention (PyTorch): x.xxx ms
# Theoretical HBM IO: Standard=3.06 MB, FlashAttention=0.50 MB, ratio=6.1x
# ...
```

#### 练习题

**练习1（基础）**：手动推导 online softmax：已处理块的 `m=1.0, l=2.0`，新块 score=`[2.0, 0.5, 3.0]`，value=`[[1,2],[3,4],[5,6]]`，计算新的 `m_new, l_new, o_new`（假设 o 初始为 0）。
> 提示：m_new=3.0, l_scale=exp(1-3)=0.135, l_new=2×0.135 + exp(2-3)+exp(0.5-3)+exp(3-3) = 0.27+0.368+0.082+1.0=1.72

**练习2（进阶）**：修改 `flash_attention_pytorch` 使用未归一化的 `o`（最后做 `O=o/l`），验证与归一化版本结果一致。

**练习3（综合）**：用 `torch.profiler` 对比两种实现的 `cuda_memory_usage`，观察标准 Attention 的 N×N 中间矩阵分配。

---

### 今日面试题

**面试题1**：FlashAttention 为什么快？请从 HBM 访问量的角度完整分析。（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- 标准 Attention 需要物化 S=QK^T 和 P=softmax(S) 两个 N×N 矩阵到 HBM，HBM 访问量为 O(N²)
- FlashAttention 通过 tiling 将 Q/K/V 分成小 tile，利用 online softmax 在 SRAM 中完成 softmax 和输出累加
- HBM 访问量降为 O(Nd)（只读 Q/K/V，只写 O）
- 速度来源不是减少 FLOPs，而是减少数据移动；符合"减少数据移动比减少计算更重要"的优化原则
- 长序列（N>2048）、小 head dim 时收益最大，因为此时 HBM 带宽是瓶颈

**面试题2**：完整推导 online softmax 的三个更新公式，并解释 `exp(m - m_new)` 的作用。（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**（要求白板推导）：
```
m_new = max(m, max(xj))
l_new = l × exp(m - m_new) + Σ exp(xj - m_new)
o_new = o × (l × exp(m - m_new) / l_new) + Σ (exp(xj - m_new) / l_new) × vj
```
- `exp(m - m_new)` 是统一参考点的缩放因子。因为 softmax 的分母需要以同一个 max 为参考，当全局 max 从 m 更新到 m_new 时，之前所有 exp 值都需要从"以 m 为参考"缩放到"以 m_new 为参考"
- 这个缩放因子保证递推过程中的概率分布始终一致
```

**面试题3**：FlashAttention 的实际 wall-clock 加速为什么通常只有 2-8x，而不是 IO 复杂度的 100x？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- 标准 Attention 的时间 = max(T_gemm, T_memory)。当 N 不够大时，GEMM 计算本身也占相当时间
- FlashAttention 消除了 O(N²) 的 HBM 读写，但 GEMM 的 FLOPs 没有减少
- 如果原始 T_gemm ≈ T_memory，加速比 ≈ 2x；如果 T_memory >> T_gemm，加速比 ≈ 8x+
- 所以长序列、小 d（GEMM 计算强度低）时收益最大

---

### 今日自测清单

- [ ] 能独立推导 online softmax 的三个更新公式
- [ ] 能解释 `exp(m - m_new)` 的作用（统一参考点）
- [ ] 能说出标准 Attention HBM 访问 O(N²) 的来源（物化 S 和 P）
- [ ] 能解释 FlashAttention HBM 访问为什么是 O(Nd)
- [ ] PyTorch 教学版实现编译运行正确，与标准 Attention 误差 < 1e-5
- [ ] 能解释为什么实际 wall-clock 加速通常是 2-8x 而不是 100x
- [ ] 能计算给定 Br/Bc/d 下的 SRAM 使用量

---

## Day 23（周二）：手写完整 FlashAttention Forward Kernel

> **今日目标**：在 Week 2 Day 12 简化版基础上，手写支持 batch、multi-head 的完整 FlashAttention Forward Kernel，正确处理边界和 online softmax。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，能手写 FlashAttention 是 Infra 面试的高区分度技能

---

### 学习任务1：完整 Forward Kernel 设计（1小时）

#### 设计要点

与 Week 2 Day 12 简化版相比，完整版需要：
1. **支持 batch 和 multi-head**：通过 `blockIdx.z` 区分 batch，`blockIdx.y` 区分 head
2. **正确处理边界**：seq_len N 不一定是 Br 的倍数
3. **更合理的线程分配**：每个 warp 负责一个 Q 行的 online softmax 更新
4. **使用 warp shuffle 加速 reduce**：比单线程循环更高效
5. **向量化加载**：用 `float4` 加载 Q/K/V tile

#### 线程配置设计

```
每个 Block 处理一个 Q tile（Br 行）
Block 维度: (Bc, Br/4) 或更优：每个 warp 负责一行 Q

推荐配置：
 Br = 64, Bc = 64, d = 64
 Block: (32, 4) = 128 threads
 - 每个 warp (32 threads) 负责 Br/4 = 16 行 Q 中的一行
 - 或者更简单地：每个 warp 负责一行 Q，需要 Br 个 warps

对于 d=64, Br=64:
 每个 Q 行有 64 个元素，一个 warp 的 32 线程每人处理 2 个元素
 Block 配置：(32, 8) = 256 threads，8 个 warp，每个 warp 负责 8 行 Q
```

#### Online Softmax 在 CUDA 中的实现

```
每个 Q 行独立维护 (m, l, acc[d])

对于第 i 个 Q 行：
 m = -inf, l = 0, acc[d] = 0
 对于每个 KV tile j:
 Sij = Qi · Kj^T (长度 Bc 的向量)
 mij = max(Sij) (warp reduce max)
 m_new = max(m, mij)
 
 # 缩放旧 l 和 acc
 scale_old = exp(m - m_new)
 l = l * scale_old
 acc *= scale_old
 
 # 处理新块
 for c in 0..Bc-1:
 p = exp(Sij[c] - m_new)
 l += p
 acc += p * Vj[c, :]
 
 m = m_new
 
 # 归一化输出
 O[i, :] = acc / l
```

> 注意：为了数值稳定性，常见做法是在每个 KV tile 内部先对 Sij 做 local softmax，再与全局状态合并。

---

### 晚间编程任务：完整 FlashAttention Forward Kernel（1.5小时）

#### 完整代码

```cpp
// flash_attention_v2.cu —— 完整 FlashAttention Forward Kernel
// 支持 batch, multi-head，正确处理边界
// 编译命令: nvcc -o flash_attention_v2 flash_attention_v2.cu -O3 -arch=sm_120
// 运行命令: ./flash_attention_v2

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// --------------------------------------------------
// 可调整参数
// --------------------------------------------------
constexpr int Br = 64; // Q tile 行数
constexpr int Bc = 64; // KV tile 行数
constexpr int D = 64;  // Head dimension

// Block 配置：Bc 列方向线程，每个 warp 负责一行 Q
// 推荐：Bc=64, warpsPerBlock=8 => 256 threads
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 256
static_assert(Br % WARPS_PER_BLOCK == 0, "Br must be divisible by WARPS_PER_BLOCK");
constexpr int ROWS_PER_WARP = Br / WARPS_PER_BLOCK; // 8

// --------------------------------------------------
// Warp 级 reduce 原语
// --------------------------------------------------
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --------------------------------------------------
// FlashAttention Forward Kernel
// 输入: Q[B,H,N,D], K[B,H,N,D], V[B,H,N,D]
// 输出: O[B,H,N,D]
// --------------------------------------------------
__global__ void flashAttentionForward(const float* __restrict__ Q, const float* __restrict__ K,
                                      const float* __restrict__ V, float* __restrict__ O, int B, int H, int N, int d) {

    // Shared memory 分配
    __shared__ float s_Q[Br][D];
    __shared__ float s_K[Bc][D];
    __shared__ float s_V[Bc][D];

    // 当前 Block 的 batch, head, Q tile 行起始
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qTileRow = blockIdx.x * Br;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warpId = tid / 32;

    // 当前 warp 负责的 Q 行范围
    int qRowStart = warpId * ROWS_PER_WARP;

    // 计算 base offset: Q/K/V/O 中 (batch, head) 对应的起始位置
    int bhOffset = ((batch * H + head) * N) * d;

// 每个线程负责 Q tile 中某些元素的加载
// 协作加载 Q tile: Br×d 个元素，由 THREADS_PER_BLOCK 个线程加载
    #pragma unroll
    for (int idx = tid; idx < Br * d; idx += THREADS_PER_BLOCK) {
        int r = idx / d;
        int c = idx % d;
        int globalRow = qTileRow + r;
        s_Q[r][c] = (globalRow < N) ? Q[bhOffset + globalRow * d + c] : 0.0f;
    }
    __syncthreads();

    // 每个 warp 维护 ROWS_PER_WARP 个 Q 行的 running 状态
    float m[ROWS_PER_WARP];
    float l[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP][D];

    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        m[i] = -1e30f;
        l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // 内层循环：遍历 KV tile
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
// 协作加载 K tile 和 V tile
        #pragma unroll
        for (int idx = tid; idx < Bc * d; idx += THREADS_PER_BLOCK) {
            int r = idx / d;
            int c = idx % d;
            int globalRow = kvStart + r;
            float kv_val = (globalRow < N) ? K[bhOffset + globalRow * d + c] : 0.0f;
            s_K[r][c] = kv_val;
            s_V[r][c] = (globalRow < N) ? V[bhOffset + globalRow * d + c] : 0.0f;
        }
        __syncthreads();

// 每个 warp 处理 ROWS_PER_WARP 个 Q 行
        #pragma unroll
        for (int localRow = 0; localRow < ROWS_PER_WARP; localRow++) {
            int qi = qRowStart + localRow;
            if (qi >= Br || (qTileRow + qi) >= N)
                continue;

            // Step 1: 计算 Sij[qi][:] = Qi[qi] · Kj[:]^T
            // 每个线程计算 Bc/32 个点积结果，然后用 warp shuffle 汇总
            float Sij[Bc / 32];
            #pragma unroll
            for (int c = lane; c < Bc; c += 32) {
                float dot = 0.0f;
                #pragma unroll
                for (int di = 0; di < d; di++) {
                    dot += s_Q[qi][di] * s_K[c][di];
                }
                Sij[c / 32] = dot;
            }

            // Step 2: 找出当前 KV tile 的局部 max（warp reduce）
            float localMax = -1e30f;
            #pragma unroll
            for (int i = 0; i < Bc / 32; i++) {
                localMax = fmaxf(localMax, Sij[i]);
            }
            localMax = warpReduceMax(localMax);

            // Step 3: online softmax update
            float m_prev = m[localRow];
            float m_new = fmaxf(m_prev, localMax);
            float scale_old = expf(m_prev - m_new);

            // 缩放旧状态
            m[localRow] = m_new;
            l[localRow] = l[localRow] * scale_old;
            #pragma unroll
            for (int di = 0; di < d; di++) {
                acc[localRow][di] *= scale_old;
            }

// 处理新块
            #pragma unroll
            for (int i = 0; i < Bc / 32; i++) {
                int c = lane + i * 32;
                bool valid = c < Bc && (kvStart + c) < N;
                float s_val = valid ? Sij[i] : -1e30f;
                float p_val = valid ? expf(s_val - m_new) : 0.0f;

                // 汇总 p_val 到 lane 0（warp sum）
                float p_sum = warpReduceSum(p_val);
                if (lane == 0) {
                    l[localRow] += p_sum;
                }

// 每个线程计算 p_val * Vj[c][di]，用 warp shuffle 汇总
                #pragma unroll
                for (int di = 0; di < d; di++) {
                    float contrib = valid ? p_val * s_V[c][di] : 0.0f;
                    float sum_contrib = warpReduceSum(contrib);
                    if (lane == 0) {
                        acc[localRow][di] += sum_contrib;
                    }
                }
            }

            // 广播 l 和 acc 到 warp 内所有线程（因为后面要写回需要所有线程参与）
            // 实际上我们在 lane 0 持有正确值，需要 broadcast
            l[localRow] = __shfl_sync(0xFFFFFFFF, l[localRow], 0);
            #pragma unroll
            for (int di = 0; di < d; di++) {
                acc[localRow][di] = __shfl_sync(0xFFFFFFFF, acc[localRow][di], 0);
            }
        }

        __syncthreads();
    }

// 写回 O，每个 warp 负责 ROWS_PER_WARP 行
    #pragma unroll
    for (int localRow = 0; localRow < ROWS_PER_WARP; localRow++) {
        int qi = qRowStart + localRow;
        int globalRow = qTileRow + qi;
        if (qi >= Br || globalRow >= N)
            continue;

        float inv_l = 1.0f / l[localRow];
        #pragma unroll
        for (int di = lane; di < d; di += 32) {
            O[bhOffset + globalRow * d + di] = acc[localRow][di] * inv_l;
        }
    }
}

// --------------------------------------------------
// CPU 参考实现（标准 Attention）
// --------------------------------------------------
void cpuAttention(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    float* S = (float*)malloc(N * N * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }

        float mx = S[i * N];
        for (int j = 1; j < N; j++)
            mx = fmaxf(mx, S[i * N + j]);
        float sm = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - mx);
            sm += S[i * N + j];
        }
        for (int j = 0; j < N; j++)
            S[i * N + j] /= sm;

        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += S[i * N + j] * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }

    free(S);
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) {
        maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    }
    bool ok = maxDiff < eps;
    printf(" maxDiff = %.2e (%s)\n", maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    int B = 2;
    int H = 4;
    int N = 256;
    int d = D;

    printf("=== FlashAttention v2 Forward Kernel ===\n");
    printf("Config: B=%d, H=%d, N=%d, d=%d\n", B, H, N, d);
    printf("Tile: Br=%d, Bc=%d, Threads=%d\n\n", Br, Bc, THREADS_PER_BLOCK);

    size_t totalElems = (size_t)B * H * N * d;
    size_t bytes = totalElems * sizeof(float);

    float* h_Q = (float*)malloc(bytes);
    float* h_K = (float*)malloc(bytes);
    float* h_V = (float*)malloc(bytes);
    float* h_O = (float*)malloc(bytes);
    float* h_O_CPU = (float*)malloc(bytes);

    initData(h_Q, totalElems);
    initData(h_K, totalElems);
    initData(h_V, totalElems);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    dim3 grid((N + Br - 1) / Br, H, B);
    dim3 block(THREADS_PER_BLOCK);

    // warmup
    flashAttentionForward<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, d);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    flashAttentionForward<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

    // CPU 验证（只验证第一个 head）
    cpuAttention(h_Q, h_K, h_V, h_O_CPU, N, d);
    printf("[B=0, H=0] First head check:\n");
    checkResult(h_O, h_O_CPU, N * d, 1e-3f);
    printf("GPU Time: %.3f ms\n", ms);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O_CPU);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o flash_attention_v2 flash_attention_v2.cu -O3 -arch=sm_120

# 运行
./flash_attention_v2

# 预期输出
# === FlashAttention v2 Forward Kernel ===
# Config: B=2, H=4, N=256, d=64
# Tile: Br=64, Bc=64, Threads=256
# 
# [B=0, H=0] First head check:
# maxDiff = x.xx e-04 (PASS)
# GPU Time: x.xxx ms
```

#### 练习题

**练习1（基础）**：修改 Br=128, Bc=128，重新编译运行，观察 SRAM 使用量和性能变化。
> 提示：SRAM = Br×d + 2×Bc×d + 寄存器占用，注意不要超出 shared memory 上限。

**练习2（进阶）**：实现向量化加载 Q/K/V tile（用 `float4`），对比性能提升。
> 提示：每个线程一次加载 4 个 float，加载循环次数减少为 1/4。

**练习3（综合）**：在 N=512, 1024, 2048, 4096 上测试，记录运行时间和最大误差。

---

### 今日面试题

**面试题1**：手写 FlashAttention Forward Kernel 时，线程如何分配？每个 warp 负责什么？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- 每个 Block 负责一个 Q tile（Br 行）
- 每个 warp 负责一行或多行 Q 的完整 online softmax 计算
- warp 内 32 个线程协作：
 - 分别计算 Sij 向量的不同部分
 - 用 `warpReduceMax` 求局部 max
 - 用 `warpReduceSum` 求 p 的局部和、求 p×V 的局部和
- 跨 warp 不需要通信，因为每个 Q 行的计算是独立的
- KV tile 通过 shared memory 共享给所有 warp

**面试题2**：FlashAttention Kernel 中为什么不需要 `__syncthreads()` 在 online softmax 内部？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- 在 warp 内部，`__shfl` 是硬件同步的，不需要 `__syncthreads()`
- 每个 Q 行的 online softmax 完全在一个 warp 内完成，不涉及跨 warp 数据共享
- `__syncthreads()` 只在两个地方需要：
 1. Q/K/V tile 加载到 shared memory 后，确保所有线程可见
 2. 切换到下一个 KV tile 前，确保当前 tile 计算完成
- 这种设计避免了频繁的 block 级同步，是 FA2 减少同步点的关键思路之一

---

### 今日自测清单

- [ ] 能设计 Block/线程配置：每个 Block 一个 Q tile，每个 warp 负责若干 Q 行
- [ ] 能解释 warp shuffle 在 Sij reduce、p_sum、p×V 汇总三个地方的作用
- [ ] 代码编译运行正确，N=256 时与 CPU 标准 Attention 误差 < 1e-3
- [ ] 支持 batch 和 multi-head，grid=(N/Br, H, B) 配置正确
- [ ] 能正确处理 N 不是 Br 倍数的边界情况
- [ ] 理解 online softmax 中 `__syncthreads()` 只需要在 tile 加载后使用

---

## Day 24（周三）：FlashAttention 官方 CUDA 源码分析

> **今日目标**：阅读 FlashAttention 官方 CUDA 源码，理解其分块策略、warp 分配、shared memory 使用，对比手写实现与官方实现的差距。
> **面试考察度**：⭐⭐⭐⭐ 高频，"看过哪些开源 kernel 实现"是加分题

---

### 学习任务1：官方源码结构（45分钟）

#### 阅读内容
- **仓库**：https://github.com/Dao-AILab/flash-attention
- **核心文件**：
 - `csrc/flash_attn/src/flash_fwd_kernel.h`：Forward Kernel 主文件
 - `csrc/flash_attn/src/kernel_traits.h`：分块参数和类型定义
 - `csrc/flash_attn/src/softmax.h`：online softmax 相关辅助
 - `csrc/flash_attn/src/utils.h`：通用工具函数
- **阅读重点**：
 - Kernel launch 参数（grid/block）如何确定
 - 一个 warp group 处理多少个 attention head
 - shared memory 的分配和复用
 - 如何处理不同 head dimension（d=64, 128 等）

#### 核心概念笔记

**1. Kernel Traits 设计**

官方代码使用模板参数 `Kernel_traits` 来组织所有分块参数：

```cpp
template <typename Kernel_traits> __global__ void flash_fwd_kernel(...)
```

常见参数：
- `kBlockM` = Br（Q tile 行数）
- `kBlockN` = Bc（KV tile 行数）
- `kHeadDim` = d
- `kNWarps` = 每个 block 的 warp 数
- `kNThreads` = 每个 block 的线程数

**2. 分块策略**

| 配置 | d=64 | d=128 | d=256 |
|------|------|-------|-------|
| Br | 128 | 128 | 128 |
| Bc | 128 | 64/128 | 64 |
| Warps/Block | 4 | 8 | 16 |

**关键设计**：
- d 越大，Bc 越小，因为 K/V tile 占用 shared memory 更多
- warps 数量随 d 增加，保证有足够的计算并行度

**3. Warp Group 与 Work Partitioning**

```
FlashAttention-2 的改进：
- 将 warps 分成若干 warp groups
- 每个 warp group 负责输出 tile 的一个子块
- 减少 warp 之间的同步，提高并行度
```

一个 Block 处理一个 Q tile（Br 行 × d 列），内部划分为 warp groups：
- 例如 4 warps 分成 4 个 warp groups，每个处理 Br/4 行
- 或者 8 warps 分成 4 个 groups，每个 group 2 warps

**4. Shared Memory 分配**

```cpp
// 伪代码
__shared__ float sQ[Br][d]; // Q tile
__shared__ float sK[Bc][d]; // K tile
__shared__ float sV[Bc][d]; // V tile（通常与 K 复用或分时复用）
```

**复用技巧**：
- K 和 V tile 可以分时复用同一块 shared memory（因为计算 Sij 时只需要 K，计算 output 时只需要 V）
- Q tile 常驻 shared memory
- 这种复用显著减少 shared memory 需求，允许更大的 Br/Bc

---

### 学习任务2：源码分析重点（45分钟）

#### 分析任务清单

1. **找到 `flash_fwd_kernel` 的入口**，追踪一个 thread 的完整数据流
2. **理解 `make_tiled copy` / `cp_async`**：官方使用 async copy（Blackwell+）从 global memory 到 shared memory，与我们的 `__syncthreads` 加载不同
3. **理解 `online_softmax` 函数模板**：如何将 online softmax 泛化为支持不同精度
4. **理解 `compute_attn_1rowblock` 或类似函数**：一个 Q tile 行 block 的计算逻辑

#### 关键差距分析

| 维度 | 手写教学版 | 官方实现 | 差距 |
|------|----------|---------|------|
| 数据加载 | 普通 global → shared memory | `cp_async` 异步拷贝 + 双缓冲 | 官方隐藏加载延迟 |
| 精度支持 | FP32 only | FP16/BF16 + FP32 accumulate | 官方带宽减半 |
| 分块参数 | 固定 Br/Bc/d | 模板参数，多配置 auto-tune | 官方适配更多场景 |
| Warp 分工 | 简单每个 warp 一行 | warp groups + 子块划分 | 官方并行度更高 |
| Shared mem | K/V 分开 | K/V 复用 | 官方更省内存 |
| Tensor Core | 未使用 | 可能使用 WMMA/mma | 官方峰值更高 |

---

### 晚间任务：源码阅读笔记整理（1小时）

#### 笔记模板

```markdown
# FlashAttention 官方源码分析笔记

## 文件结构
- flash_fwd_kernel.h: Forward kernel 主入口
- kernel_traits.h: 模板参数定义
- softmax.h: online softmax 辅助

## 关键发现
1. Br/Bc 随 d 变化：d 越大 Bc 越小
2. K/V 分时复用 shared memory
3. cp_async 异步拷贝

## 与手写版的差距
1. 异步拷贝隐藏延迟
2. 多精度支持
3. warp group 更细粒度并行
```

## Day 25（周四）：FlashAttention-2 论文与源码差异

> **今日目标**：理解 FlashAttention-2 相对于 FA1 的关键改进，分析其减少 non-matmul FLOPs 和 better work partitioning 的具体做法。
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，FA2 改进点是高频追问

---

### 学习任务1：FlashAttention-2 核心改进（1小时）

#### 阅读内容
- **论文**："FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)
- **地址**：https://arxiv.org/abs/2307.08691
- **阅读范围**：
 - Section 1: Introduction（FA1 的不足）
 - Section 2: Background（online softmax 回顾）
 - Section 3: FlashAttention-2 Algorithm（3.1 减少 non-matmul FLOPs，3.2 更好的 work partitioning）
- **辅助阅读**：官方仓库中 FA2 的源码差异

#### 核心概念笔记

**1. FA1 的不足**

```
FA1 的问题：
1. 不同 warp group 之间存在冗余的 softmax 统计量同步
2. 非 matmul 计算（online softmax 的 reduce/rescale）没有充分并行
3. Q tile 行 block 内部的 warp 分工不够细，导致部分 warp 空闲
```

**2. FA2 改进一：减少 Non-Matmul FLOPs**

```
FA1 中，每个线程块内的不同 warp 需要重复计算一些中间量。

FA2 改进：
- 让一个 warp group 负责输出 tile 的一个子块（sub-tile）
- 在 warp group 内部完成该子块的全部 online softmax 计算
- 避免跨 warp group 的同步和重复计算

具体效果：
- FA1: non-matmul FLOPs 与 matmul FLOPs 之比约为 1:10
- FA2: 降低到 1:20 或更少
```

**3. FA2 改进二：更好的 Work Partitioning**

```
FA1 的 work partitioning：
- 一个 Block 处理一个 Q tile
- Block 内 warps 共同完成整个 Q tile

FA2 的 work partitioning：
- 一个 Block 仍处理一个 Q tile
- 但将 Q tile 在行方向进一步划分给不同 warp groups
- 甚至可以在序列长度方向并行（seq_parallel）

两种并行维度：
1. Batch × Head 并行（z 和 y 维度）
2. Sequence length 并行（x 维度，一个 head 的序列分多个 block）
3. 内部 warp group 并行（一个 block 内）
```

**4. FA2 改进三：更好的 Occupancy**

```
通过减少 register 和 shared memory 使用，每个 SM 可以驻留更多 block。
- FA1 可能一个 SM 只能跑 1 个 block
- FA2 可以跑 2-3 个 block，提高 occupancy
```

#### FA1 vs FA2 关键差异表

| 维度 | FlashAttention-1 | FlashAttention-2 |
|------|------------------|------------------|
| Non-matmul 并行 | 不够充分 | warp group 内独立完成 |
| Work partitioning | 按 Q tile | 按 Q tile 子块 + seq 并行 |
| Warp 同步 | 较多 | 较少 |
| Occupancy | 较低 | 较高 |
| 反向传播 | 支持 | 更高效 |
| 长序列收益 | 好 | 更好 |

---

### 学习任务2：Seq 并行 vs Head 并行（30分钟）

#### 概念对比

```
Head 并行（Batch/Head 维度）：
 - 不同 head 在不同 block 上并行
 - 优点：自然，不需要同步
 - 缺点：当 head 数少时（如 8 头），并行度不够

Seq 并行（Sequence 长度维度）：
 - 同一个 head 的序列分成多个 block 并行
 - 优点：增加并行度，尤其适合长序列
 - 缺点：需要处理块间依赖（但 FlashAttention 的 tiling 天然支持）
```

#### 如何选择？

```
并行度优先级：
1. 先充分利用 Batch × Head 并行（gridDim.y × gridDim.z）
2. 如果 Batch×Head 不够大，再开启 Seq 并行
3. 长序列场景下，Seq 并行收益明显
```

---

### 晚间任务：基于 FA2 思想优化手写 Kernel（1小时）

#### 优化方向

1. **更细的 warp group 分工**：将 ROWS_PER_WARP 从 8 改为 4，增加 warp 间并行
2. **减少同步**：确保 warp 内计算不需要 block 级同步
3. **Seq 并行实验**：当 B×H 较小时，尝试在 x 维度使用更多 block

#### 练习题

**练习1（基础）**：列出 FA1 vs FA2 的至少 3 个关键差异。

**练习2（进阶）**：修改 Day 23 的 Kernel，让一个 warp group（2 warps）负责一个 Q 行子块，观察性能变化。

**练习3（综合）**：在 RTX 5090 上测试官方 FA1 和 FA2 的性能差异（如果环境允许安装 flash-attention 包），记录不同 seq_len 下的加速比。

---

### 今日面试题

**面试题1**：FlashAttention-2 相比 FlashAttention-1 有哪些关键改进？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
1. **减少 non-matmul FLOPs**：通过 warp group 子块划分，让 softmax/rescale 计算在 warp group 内独立完成，减少冗余
2. **更好的 work partitioning**：除了 batch/head 并行，还引入 sequence 长度方向并行，提高长序列下的并行度
3. **更高的 occupancy**：优化 register 和 shared memory 使用，每个 SM 可驻留更多 block
4. **更少的 warp 同步**：减少 block 级同步点
5. **反向传播更高效**

**面试题2**：FlashAttention-2 中，seq 并行和 head 并行有什么区别？什么时候用 seq 并行？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Head 并行**：不同 attention head 在不同 block 上并行，天然无依赖，是首选
- **Seq 并行**：同一个 head 的序列分成多个 block 并行，增加并行度
- **使用时机**：当 batch×head 数量不足以填满 GPU 时使用 seq 并行，尤其长序列场景
- **注意**：seq 并行需要处理 Q tile 的边界，但 FlashAttention 的 tiling 天然适合这种划分

---

### 今日自测清单

- [ ] 能说出 FA1 vs FA2 的 3 个关键差异
- [ ] 理解"减少 non-matmul FLOPs"的具体含义
- [ ] 理解 warp group 子块划分如何提高并行度
- [ ] 能解释 seq 并行与 head 并行的 trade-off
- [ ] 阅读 FA2 论文 Section 3.1 和 3.2

---

## Day 26（周五）：项目推进 —— 集成 FlashAttention 到 Mini 引擎

> **今日目标**：将手写 FlashAttention kernel 接入 Week 3 的 Mini 推理引擎，替换标准 Attention 路径，验证端到端正确性。
> **面试考察度**：⭐⭐⭐⭐ 高频，"如何把自定义 kernel 集成到推理框架"是工程能力体现

---

### 学习任务1：Mini 引擎 Attention 替换设计（45分钟）

#### 设计目标

在 Week 3 Day 19 的 Mini 引擎基础上：
1. 新增 `flash_attention_forward` 自定义 CUDA 算子
2. 在 `MiniAttention` 中使用该算子替换 `torch.matmul(Q,K^T) → softmax → torch.matmul(P,V)` 路径
3. 保持 GEMM（QKV/Output/FFN）仍使用 PyTorch/cuBLAS
4. 对比"标准 Attention" vs "FlashAttention"的正确性和速度

#### 架构图

```
Mini Transformer Engine v2:

Input x (B, N, d)
 │
 ├─► [LayerNorm1] ──► [QKV GEMM (cuBLAS)] ──► Q, K, V
 │ │
 │ ├─► [FlashAttention★] ──► O
 │ │ （替换 QK^T → softmax → PV）
 │ │
 │ └─► [Out GEMM]
 │
 ├─► [LayerNorm2]
 │
 └─► [FFN]

★ = 自定义 CUDA FlashAttention kernel
```

#### 接口设计

```cpp
// flash_attention_ops.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V);
// Q/K/V shape: (B, H, N, d)
// O shape: (B, H, N, d)
```

---

### 学习任务2：封装自定义算子（45分钟）

#### C++ Wrapper 代码

```cpp
// flash_attention_ops.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// 声明 Day 23 的 kernel
void launch_flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int B, int H, int N,
                                    int d, cudaStream_t stream);

at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V) {
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B,H,N,d)");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K shape must match Q");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V shape must match Q");

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), d = Q.size(3);
    auto O = at::empty_like(Q);

    launch_flash_attention_forward(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
                                   B, H, N, d, at::cuda::getCurrentCUDAStream());
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward, "FlashAttention forward (CUDA)");
}
```

对应的 `launch_flash_attention_forward` 需要在 `flash_attention_v2.cu` 中添加：

```cpp
// 在 flash_attention_v2.cu 末尾添加
void launch_flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int B, int H, int N,
                                    int d, cudaStream_t stream) {

    dim3 grid((N + Br - 1) / Br, H, B);
    dim3 block(THREADS_PER_BLOCK);

    flashAttentionForward<<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, d);
}
```

---

### 晚间编程任务：Mini 引擎 FlashAttention 版（1.5小时）

#### 完整代码

```python
# mini_engine_fa.py —— Mini Transformer 引擎（FlashAttention 版）
# 运行命令: python mini_engine_fa.py
# 依赖: 需要 flash_attention_v2.cu 和 flash_attention_ops.cpp

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline
import os

# 动态编译自定义 FlashAttention 算子
cuda_src = open("flash_attention_v2.cu").read()
cpp_src = """
#include <torch/extension.h>
at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V);
"""

fa_ops = load_inline(
 name="fa_ops",
 cpp_sources=cpp_src,
 cuda_sources=cuda_src,
 functions=["flash_attention_forward"],
 verbose=True,
 extra_cuda_cflags=["-O3", "-arch=sm_120"],
)

class MiniAttentionFA(nn.Module):
 """用自定义 FlashAttention 替换标准 Attention"""
 def __init__(self, d_model=512, n_heads=8):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.qkv = nn.Linear(d_model, 3 * d_model)
 self.out = nn.Linear(d_model, d_model)

 def forward(self, x):
 B, N, _ = x.shape
 qkv = self.qkv(x)
 qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
 qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, H, N, d)
 q, k, v = qkv[0], qkv[1], qkv[2]

 # ★ 自定义 FlashAttention
 out = fa_ops.flash_attention_forward(q, k, v)

 out = out.transpose(1, 2).reshape(B, N, self.d_model)
 return self.out(out)

class MiniAttentionStd(nn.Module):
 """标准 Attention（PyTorch 实现）"""
 def __init__(self, d_model=512, n_heads=8):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.qkv = nn.Linear(d_model, 3 * d_model)
 self.out = nn.Linear(d_model, d_model)

 def forward(self, x):
 B, N, _ = x.shape
 qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
 q, k, v = qkv[0], qkv[1], qkv[2]
 scale = self.d_head ** -0.5
 attn = torch.matmul(q, k.transpose(-2, -1)) * scale
 attn = F.softmax(attn, dim=-1)
 out = torch.matmul(attn, v)
 out = out.transpose(1, 2).reshape(B, N, self.d_model)
 return self.out(out)

class TransformerBlock(nn.Module):
 def __init__(self, d_model=512, n_heads=8, d_ff=2048, use_fa=True):
 super().__init__()
 attn_cls = MiniAttentionFA if use_fa else MiniAttentionStd
 self.attn = attn_cls(d_model, n_heads)
 self.norm1 = nn.LayerNorm(d_model)
 self.norm2 = nn.LayerNorm(d_model)
 self.ffn = nn.Sequential(
 nn.Linear(d_model, d_ff),
 nn.GELU(),
 nn.Linear(d_ff, d_model),
 )

 def forward(self, x):
 x = x + self.attn(self.norm1(x))
 x = x + self.ffn(self.norm2(x))
 return x

def benchmark(model, x, name, n_iter=20):
 for _ in range(3):
 _ = model(x)
 torch.cuda.synchronize()

 start = torch.cuda.Event(enable_timing=True)
 end = torch.cuda.Event(enable_timing=True)
 start.record()
 for _ in range(n_iter):
 _ = model(x)
 end.record()
 torch.cuda.synchronize()
 ms = start.elapsed_time(end) / n_iter
 print(f"{name}: {ms:.3f} ms / forward")
 return ms

def main():
 torch.manual_seed(42)
 d_model, n_heads = 512, 8

 # 测试不同序列长度
 for N in [512, 1024, 2048]:
 print(f"\n===== N={N} =====")
 x = torch.randn(1, N, d_model, device="cuda", dtype=torch.float32)

 model_std = TransformerBlock(d_model, n_heads, use_fa=False).cuda()
 model_fa = TransformerBlock(d_model, n_heads, use_fa=True).cuda()
 model_fa.load_state_dict(model_std.state_dict())

 with torch.no_grad():
 out_std = model_std(x)
 out_fa = model_fa(x)
 max_diff = (out_std - out_fa).abs().max().item()
 print(f"Max diff (Std vs FlashAttention): {max_diff:.2e}")

 with torch.no_grad():
 ms_std = benchmark(model_std, x, f"Standard Attention (N={N})")
 ms_fa = benchmark(model_fa, x, f"FlashAttention (N={N})")
 print(f"Speedup: {ms_std / ms_fa:.2f}x")

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
# 确保 flash_attention_v2.cu 和本文件在同一目录
python mini_engine_fa.py

# 预期输出
# ===== N=512 =====
# Max diff (Std vs FlashAttention): x.xx e-04
# Standard Attention (N=512): x.xxx ms / forward
# FlashAttention (N=512): x.xxx ms / forward
# Speedup: 0.8x ~ 1.2x
# 
# ===== N=2048 =====
# ...
# Speedup: 1.5x ~ 3.0x
```

> **注意**：N 较小时 FlashAttention 可能没有优势（甚至略慢），因为 kernel launch 和 shared memory 开销。N 越大优势越明显。

#### 练习题

**练习1（基础）**：修改 `MiniAttentionFA` 使其支持 FP16 输入（kernel 内部 cast 到 FP32 计算）。

**练习2（进阶）**：在 Mini 引擎中同时使用自定义 FlashAttention 和自定义 LayerNorm，对比全 PyTorch 版的速度。

**练习3（综合）**：用 `nsys profile` 采集 Mini 引擎 FlashAttention 版的时间线，观察 kernel 数量是否减少。

---

### 今日面试题

**面试题1**：如何把自定义 FlashAttention kernel 集成到 PyTorch 推理引擎中？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. 写 CUDA kernel（如 Day 23 的 `flashAttentionForward`）
2. 写 C++ wrapper 函数，接收 `at::Tensor`，调用 kernel
3. 使用 `torch.utils.cpp_extension.load_inline` 或 `setup.py` 编译成 Python 可调用的模块
4. 在 PyTorch `nn.Module` 中替换标准 Attention 路径
5. 注意：传递正确的 stream（`at::cuda::getCurrentCUDAStream()`），保证与 PyTorch 的 async 行为一致

**面试题2**：FlashAttention 在什么情况下可能比标准 Attention 慢？为什么？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **短序列（N 较小）**：FlashAttention 的 shared memory 设置、online softmax 递推有固定开销，可能比直接调用 cuBLAS + softmax 慢
- **head dim 较大（d=256+）**：tile 变小，计算强度降低，优势减弱
- **GPU 上 HBM 带宽不是瓶颈时**：例如 batch 很小但 head 很多，标准 Attention 的 GEMM 可能已经接近峰值
- **实现不够优化**：教学版缺少 async copy、双缓冲、向量化等优化
- 实际部署中需要 benchmark 决定是否启用

---

### 今日自测清单

- [ ] 能写出 `flash_attention_forward` 的 C++ wrapper
- [ ] 能用 `load_inline` 编译 FlashAttention 自定义算子
- [ ] Mini 引擎 FlashAttention 版编译运行成功
- [ ] 自定义版与标准 Attention 输出误差 < 1e-3
- [ ] 长序列（N=2048+）下观察到加速
- [ ] 能解释短序列下 FlashAttention 可能更慢的原因

---

## Day 27（周六）：性能对比分析

> **今日目标**：构建 benchmark 框架，系统对比标准 Attention、手写 FlashAttention、官方 FlashAttention 在不同 seq_len/batch/head 下的性能，并用 ncu 验证 HBM 访问量。
> **时间分配**：6小时全天投入（benchmark 框架2h + 扫描测试2h + ncu 验证2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"如何做性能对比"是工程能力的直接体现

---

### 学习任务1：Benchmark 框架设计（2小时）

#### 对比维度

| 维度 | 取值范围 |
|------|---------|
| seq_len N | 512, 1024, 2048, 4096, 8192 |
| batch size B | 1, 4, 16 |
| num heads H | 8, 16 |
| head dim d | 64, 128 |
| 实现 | Standard, Handwritten FA, Official FA |

#### 关键指标

- **Latency (ms)**：单次 forward 时间
- **Throughput (tokens/s)**：`B * N / latency`
- **HBM IO (MB)**：理论值 + ncu 实测值
- **Speedup**：相对于标准 Attention 的加速比
- **Max Diff**：与标准 Attention 的数值误差

---

### 学习任务2：编写 Benchmark 脚本（2小时）

#### 完整代码

```python
# benchmark_flash_attention.py —— FlashAttention 性能对比框架
# 运行命令: python benchmark_flash_attention.py

import torch
import torch.nn.functional as F
import math
import time
import json

try:
 from flash_attn import flash_attn_func # 官方 FlashAttention
 HAS_OFFICIAL = True
except ImportError:
 HAS_OFFICIAL = False
 print("Warning: official flash_attn not installed, skipping official benchmark")

def standard_attention(Q, K, V):
 d = Q.size(-1)
 S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
 P = F.softmax(S, dim=-1)
 O = torch.matmul(P, V)
 return O

def benchmark(func, Q, K, V, n_iter=10):
 # warmup
 for _ in range(3):
 _ = func(Q, K, V)
 torch.cuda.synchronize()

 start = torch.cuda.Event(enable_timing=True)
 end = torch.cuda.Event(enable_timing=True)
 start.record()
 for _ in range(n_iter):
 out = func(Q, K, V)
 end.record()
 torch.cuda.synchronize()
 ms = start.elapsed_time(end) / n_iter
 return ms

def theoretical_io(N, d, dtype_size=4):
 """理论 HBM IO（MB）"""
 std_io = (3 * N * N + 4 * N * d) * dtype_size / (1024 * 1024)
 fa_io = (4 * N * d) * dtype_size / (1024 * 1024)
 return std_io, fa_io

def main():
 torch.manual_seed(42)
 device = "cuda"
 dtype = torch.float32

 configs = [
 {"B": 1, "H": 8, "N": 512, "d": 64},
 {"B": 1, "H": 8, "N": 1024, "d": 64},
 {"B": 1, "H": 8, "N": 2048, "d": 64},
 {"B": 1, "H": 8, "N": 4096, "d": 64},
 {"B": 4, "H": 8, "N": 2048, "d": 64},
 {"B": 1, "H": 16, "N": 2048, "d": 128},
 ]

 results = []

 print("=== FlashAttention Performance Benchmark ===")
 print(f"{'B':>3} {'H':>3} {'N':>5} {'d':>4} | {'Std(ms)':>10} {'Hand(ms)':>10} {'Off(ms)':>10} | {'Hand-Spd':>10} {'Off-Spd':>10} | {'StdIO(MB)':>10} {'FAIO(MB)':>10}")
 print("-" * 100)

 for cfg in configs:
 B, H, N, d = cfg["B"], cfg["H"], cfg["N"], cfg["d"]

 Q = torch.randn(B, H, N, d, device=device, dtype=dtype)
 K = torch.randn(B, H, N, d, device=device, dtype=dtype)
 V = torch.randn(B, H, N, d, device=device, dtype=dtype)

 # 标准 Attention
 ms_std = benchmark(standard_attention, Q, K, V)

 # 手写 FlashAttention（需要已编译的 fa_ops）
 try:
 from mini_engine_fa import fa_ops
 ms_hand = benchmark(fa_ops.flash_attention_forward, Q, K, V)
 hand_speedup = ms_std / ms_hand
 except Exception as e:
 ms_hand = float('nan')
 hand_speedup = float('nan')

 # 官方 FlashAttention
 if HAS_OFFICIAL:
 ms_off = benchmark(flash_attn_func, Q, K, V)
 off_speedup = ms_std / ms_off
 else:
 ms_off = float('nan')
 off_speedup = float('nan')

 std_io, fa_io = theoretical_io(N, d)

 print(f"{B:>3} {H:>3} {N:>5} {d:>4} | {ms_std:>10.3f} {ms_hand:>10.3f} {ms_off:>10.3f} | {hand_speedup:>10.2f}x {off_speedup:>10.2f}x | {std_io:>10.2f} {fa_io:>10.2f}")

 results.append({
 "B": B, "H": H, "N": N, "d": d,
 "std_ms": ms_std,
 "hand_ms": ms_hand,
 "off_ms": ms_off,
 "hand_speedup": hand_speedup,
 "off_speedup": off_speedup,
 "std_io_mb": std_io,
 "fa_io_mb": fa_io,
 })

 # 保存结果
 with open("benchmark_results.json", "w") as f:
 json.dump(results, f, indent=2)
 print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
 main()
```

#### 运行步骤

```bash
python benchmark_flash_attention.py

# 预期输出
# === FlashAttention Performance Benchmark ===
# B H N d | Std(ms) Hand(ms) Off(ms) | Hand-Spd Off-Spd | StdIO(MB) FAIO(MB)
# ----------------------------------------------------------------------------------------------------
# 1 8 512 64 | x.xxx x.xxx x.xxx | x.xx x.xx | x.xx x.xx
# ...
```

---

### 学习任务3：ncu 验证 HBM 访问量（2小时）

#### 验证目标

验证手写 FlashAttention 的 HBM 访问量是否随 N 线性增长（而非 N²）。

#### ncu 命令

```bash
# 编译带 lineinfo 的可执行文件
nvcc -o flash_attention_v2 flash_attention_v2.cu -O3 -arch=sm_120 -g -lineinfo

# Profile HBM 读写量
ncu \
 --metrics \
 dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
 --kernel-name regex:flashAttentionForward \
 ./flash_attention_v2

# 注意：需要先修改 main 函数支持不同 N，或分别编译多个版本
```

#### 预期结果分析

```
N=512, d=64:
 理论 FA IO = 4 * 512 * 64 * 4 bytes = 524 KB
 实测 dram_read + dram_write 应接近这个量级

N=1024, d=64:
 理论 FA IO = 4 * 1024 * 64 * 4 bytes = 1 MB
 实测应约为 N=512 时的 2x

N=2048, d=64:
 理论 FA IO = 4 * 2048 * 64 * 4 bytes = 2 MB
 实测应约为 N=512 时的 4x
```

对比标准 Attention：N 翻倍时 IO 应接近 4x。

#### 练习题

**练习1（基础）**：手动计算 N=4096, d=64 时标准 Attention 和 FlashAttention 的理论 HBM IO。

**练习2（进阶）**：修改 benchmark 脚本，加入 memory bandwidth utilization 和 FLOPS 估算。

**练习3（综合）**：用 matplotlib 绘制 latency 随 N 变化的曲线，对比三种实现。

---

### 今日面试题

**面试题1**：如何设计一个 FlashAttention 的 benchmark？需要对比哪些指标？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **Latency**：单次 forward 时间（ms）
2. **Throughput**：tokens/s 或 queries/s
3. **HBM IO**：理论值 + ncu 实测值，验证 O(Nd) vs O(N²)
4. **Speedup**：相对标准 Attention 的加速比
5. **Correctness**：与标准 Attention 的数值误差
6. **扫描维度**：seq_len N、batch B、num_heads H、head_dim d
7. **对比对象**：标准 Attention、手写 FA、官方 FA、PyTorch `scaled_dot_product_attention`

**面试题2**：如何用 ncu 验证 FlashAttention 的 HBM 访问确实是 O(Nd) 而不是 O(N²)？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- 使用 `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum`
- 测试 N=512, 1024, 2048, 4096，固定 d
- 如果 HBM 访问量 ≈ N 的线性倍数（N 翻倍，IO 翻倍），则是 O(Nd)
- 如果 HBM 访问量 ≈ N² 的倍数（N 翻倍，IO 4x），则是 O(N²)
- 注意实测值会有 cache、padding 等额外开销，误差 20-30% 内正常

---

### 今日自测清单

- [ ] 完成 benchmark 框架，能自动扫描 N/B/H/d 组合
- [ ] 记录标准 Attention、手写 FA、官方 FA 的 latency
- [ ] 计算并对比理论 HBM IO
- [ ] 用 ncu 实测手写 FA 的 dram__bytes_read + dram__bytes_write
- [ ] 验证 N 翻倍时 FA 的 HBM IO 约翻倍（O(Nd)）
- [ ] 能解释不同配置下 speedup 差异的原因
- [ ] 保存 benchmark 结果到 JSON/CSV

---

## Day 28（周日）：IO 优化方法论总结

> **今日目标**：从 FlashAttention 中提炼通用 IO 优化方法论，复盘本周面试题，整理 GitHub 仓库和性能报告。
> **时间分配**：6小时全天投入（方法论2h + 面试复盘2h + GitHub 整理2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"IO 优化方法论"是系统优化的总结性考点

---

### 任务1：IO 优化方法论提炼（2小时）

#### 从 FlashAttention 提炼的通用策略

```
FlashAttention 的核心思想：
 减少 HBM 访问 = 在 fast memory（SRAM/Shared Memory）中完成尽可能多的计算
```

**方法论 Checklist**：

| 策略 | 含义 | 适用场景 | FlashAttention 中的应用 |
|------|------|---------|------------------------|
| **Tiling** | 将大数据集分块到 fast memory | 数据量 > fast memory 容量 | Q/K/V 分块加载到 shared memory |
| **Online Algorithm** | 避免全局同步，边算边更新 | 需要全局 reduce 的场景 | online softmax 递推 |
| **Kernel Fusion** | 合并相邻算子，避免中间结果写回 HBM | memory-bound 算子相邻时 | 将 QK^T、softmax、PV 融合为一个 kernel |
| **Recomputation** | 用计算换内存访问 | 重算代价 < 读写代价时 | backward 时重算 forward 中间值 |
| **Data Layout 优化** | 调整数据排布提高访问局部性 | 不规则访问模式 | Q/K/V 按 (B,H,N,d) 连续存储 |
| **Async Copy / 双缓冲** | 隐藏数据传输延迟 | 数据搬运与计算可重叠 | `cp_async`、double buffering |

#### 场景决策树

```
遇到一个 memory-bound 算子：
 1. 是否能用 tiling 放进 SRAM？
 → 是：用 tiling + online algorithm
 → 否：考虑量化/压缩减少数据量
 1. 是否有相邻的 memory-bound 算子？
 → 是：用 kernel fusion
 → 否：考虑向量化加载提升带宽利用率
 1. 中间结果是否可被重算？
 → 是：用 recomputation 减少 HBM 读写
 → 否：考虑更换算法
```

#### IO 优化与计算优化的关系

```
优化优先级（通常）：
 1. 减少不必要的数据移动（IO 优化）
 2. 融合 kernel 减少 launch overhead
 3. 提升计算吞吐量（Tensor Core、指令级优化）

原因：
 - 数据移动能耗和延迟通常远高于计算
 - "You can hide compute, but you can't hide memory"
```

---

### 任务2：面试复盘（2小时）

#### 本周核心面试题回顾

1. FlashAttention 为什么快？HBM 角度分析
2. 推导 online softmax 三公式
3. 实际 wall-clock 加速为什么只有 2-8x？
4. FlashAttention Kernel 线程如何分配？
5. FA1 vs FA2 的关键差异
6. seq 并行 vs head 并行
7. 官方实现中 d 越大 Bc 越小？
8. K/V 如何复用 shared memory？
9. 如何把自定义 FlashAttention 集成到 PyTorch？
10. FlashAttention 什么时候比标准 Attention 慢？
11. 如何设计 FlashAttention benchmark？
12. 如何用 ncu 验证 HBM 访问 O(Nd)？
13. IO 优化方法论有哪些？

#### 自问自答

建议每个问题限时 3 分钟口述，录音或文字记录，然后对照参考答案找差距。

---

### 任务3：GitHub 整理与性能报告（2小时）

#### 仓库结构建议

```
week4-flashattention/
├── day22-paper-reading/
│ ├── compare_attention_io.py
│ └── notes.md
├── day23-handwritten-kernel/
│ ├── flash_attention_v2.cu
│ └── README.md
├── day24-official-source/
│ └── source_analysis.md
├── day25-flashattention2/
│ └── fa2_notes.md
├── day26-mini-engine/
│ ├── mini_engine_fa.py
│ └── flash_attention_ops.cpp
├── day27-benchmark/
│ ├── benchmark_flash_attention.py
│ └── benchmark_results.json
└── day28-summary/
 ├── io_optimization_methodology.md
 └── performance_report.md
```

#### 性能报告模板（`performance_report.md`）

```markdown
# Week 4 FlashAttention 性能报告

## 测试环境
- GPU: NVIDIA GeForce RTX 5090
- CUDA Version: 12.4
- Driver: 550.54.15

## 正确性
| 配置 | 与标准 Attention 最大误差 | 结果 |
|------|------------------------|------|
| N=256, d=64 | x.xx e-04 | PASS |
| N=4096, d=64 | x.xx e-04 | PASS |

## 性能对比（B=1, H=8, d=64）

| N | Standard(ms) | Handwritten FA(ms) | Official FA(ms) | Hand Speedup | Official Speedup |
|---|-------------|-------------------|----------------|--------------|-----------------|
| 512 | x.xxx | x.xxx | x.xxx | x.xx | x.xx |
| 1024 | x.xxx | x.xxx | x.xxx | x.xx | x.xx |
| 2048 | x.xxx | x.xxx | x.xxx | x.xx | x.xx |
| 4096 | x.xxx | x.xxx | x.xxx | x.xx | x.xx |

## HBM IO 验证
| N | 理论标准 IO(MB) | 理论 FA IO(MB) | ncu 实测 FA IO(MB) |
|---|----------------|---------------|-------------------|
| 512 | x.xx | x.xx | x.xx |
| 2048 | x.xx | x.xx | x.xx |

## 结论
- 长序列（N>2048）下手写 FlashAttention 达到 x.x 倍加速
- HBM 访问量随 N 线性增长，验证 O(Nd) 复杂度
- 与官方实现差距主要来自 async copy、双缓冲、Tensor Core
```

#### IO 优化方法论文档（`io_optimization_methodology.md`）

```markdown
# 从 FlashAttention 提炼的 IO 优化方法论

## 核心原则
减少 HBM 访问，在 fast memory 中完成计算。

## 策略清单
1. Tiling
2. Online Algorithm
3. Kernel Fusion
4. Recomputation
5. Async Copy / 双缓冲
6. 数据布局优化

## 决策树
[见 Day 28 笔记]

## Transformer 中的应用
- FlashAttention: Tiling + Online Softmax + Fusion
- KV Cache: 减少重复计算
- PagedAttention: 内存布局优化
- CUDA Graph: 减少 launch overhead
```

---

### 今日面试题

**面试题1**：从 FlashAttention 中提炼出通用的 IO 优化方法论，并举一个 Transformer 外的应用例子。（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **Tiling**：把大矩阵/张量分块到 SRAM，例如卷积中的 im2col + 分块
- **Online Algorithm**：避免全局同步，例如流式计算中的 online mean/variance
- **Kernel Fusion**：合并相邻算子，例如 CNN 中的 conv + bn + relu 融合
- **Recomputation**：用计算换内存，例如 activation checkpointing 反向传播
- **例子**：CNN 中的 conv + bn + relu 融合。未融合时要写卷积结果到 HBM，BN 再读；融合后在 register 中直接传递，省去一次 HBM 读写

**面试题2**：在 AI Infra 中，IO 优化和计算优化哪个更优先？为什么？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **通常 IO 优化更优先**，原因：
 1. 数据移动能耗和延迟远高于计算
 2. 现代 GPU 算力增长快于内存带宽增长，memory wall 越来越严重
 3. 很多推理场景本来就是 memory-bound
- **不是绝对**：如果系统已经是 compute-bound，再优化 IO 收益很小，应该优化计算（Tensor Core、更好的 work partitioning）
- **正确做法**：先用 profiling 判断瓶颈类型，再针对性优化

---

### 今日自测清单

- [ ] 能列出 6 种 IO 优化策略并解释每种含义
- [ ] 能用决策树分析一个陌生算子是否适合 tiling/fusion/recomputation
- [ ] 能说出 FlashAttention 中每种策略的具体体现
- [ ] 完成本周 13 道面试题的自问自答
- [ ] 整理 GitHub 仓库，所有代码有对应 README
- [ ] 生成性能对比报告
- [ ] 生成 IO 优化方法论文档
- [ ] 能解释为什么 IO 优化通常优先于计算优化
- [ ] 规划 Week 5（推理系统）的学习重点

---

## 附录A：第4周面试题汇总

| 题号 | 题目 | 考察频率 | 相关天数 | 难度 |
|------|------|---------|---------|------|
| 1 | FlashAttention 为什么快？HBM 角度 | ⭐⭐⭐⭐⭐ | Day 22 | 高 |
| 2 | 推导 online softmax 三公式 | ⭐⭐⭐⭐⭐ | Day 22 | 高 |
| 3 | 实际 wall-clock 加速为什么只有 2-8x？ | ⭐⭐⭐⭐ | Day 22 | 中 |
| 4 | FlashAttention Kernel 线程如何分配？ | ⭐⭐⭐⭐⭐ | Day 23 | 高 |
| 5 | Kernel 中为什么不需要频繁 `__syncthreads`？ | ⭐⭐⭐⭐ | Day 23 | 中 |
| 6 | 官方实现中 d 越大 Bc 越小？ | ⭐⭐⭐⭐ | Day 24 | 中 |
| 7 | K/V 如何复用 shared memory？ | ⭐⭐⭐ | Day 24 | 中 |
| 8 | FA1 vs FA2 的关键差异？ | ⭐⭐⭐⭐⭐ | Day 25 | 高 |
| 9 | seq 并行 vs head 并行？ | ⭐⭐⭐⭐ | Day 25 | 中 |
| 10 | 如何把自定义 FlashAttention 集成到 PyTorch？ | ⭐⭐⭐⭐ | Day 26 | 中 |
| 11 | FlashAttention 什么时候比标准 Attention 慢？ | ⭐⭐⭐⭐ | Day 26 | 中 |
| 12 | 如何设计 FlashAttention benchmark？ | ⭐⭐⭐⭐ | Day 27 | 中 |
| 13 | 如何用 ncu 验证 HBM 访问 O(Nd)？ | ⭐⭐⭐⭐ | Day 27 | 中 |
| 14 | IO 优化方法论有哪些？ | ⭐⭐⭐⭐⭐ | Day 28 | 高 |
| 15 | IO 优化和计算优化哪个更优先？ | ⭐⭐⭐⭐ | Day 28 | 中 |

---

## 附录C：关键公式汇总

**1. Online Softmax 三公式**
```
m_new = max(m, max(xj))
l_new = l × exp(m - m_new) + Σ exp(xj - m_new)
o_new = o × (l × exp(m - m_new) / l_new) + Σ (exp(xj - m_new) / l_new) × vj
```

**2. SRAM 使用量**
```
SRAM_per_block = Br × d + 2 × Bc × d + Br × Bc （未复用 K/V）
SRAM_per_block = Br × d + Bc × d + Br × Bc （K/V 复用）
```

**3. HBM IO 复杂度**
```
标准 Attention: O(N² + Nd) ≈ O(N²) when N >> d
FlashAttention: O(Nd)
```

**4. 实际加速比上限**
```
speedup ≈ max(T_gemm, T_memory_std) / max(T_gemm, T_memory_fa)
 ≈ 2-8x for typical cases
```

**5. Arithmetic Intensity**
```
AI = FLOPs / Bytes
Memory-bound: AI < Ridge Point
Compute-bound: AI > Ridge Point
```

**6. Roofline Ridge Point（RTX 5090）**
```
Ridge Point = Peak FLOP/s / Peak Bandwidth
 = 19.5 TFLOP/s / 1.55 TB/s
 ≈ 12.6 FLOP/Byte
```

---

## 附录D：性能诊断速查表（Week 4 专用）

| 现象 | 可能原因 | 检查方法 | 解决方案 |
|------|---------|---------|---------|
| 手写 FA 结果误差大 | online softmax 参考点更新错误 | 与 CPU 标准 Attention 对比 | 检查 m_new 和 scale_old 计算 |
| 手写 FA 长序列更慢 | shared memory 不足导致 occupancy 低 | ncu 看 occupancy | 减小 Br/Bc 或 d |
| N 小时 FA 比标准慢 | kernel launch 和 shared mem 固定开销 | benchmark N=512/1024 | 短序列用标准 Attention |
| ncu HBM IO 非线性增长 | 中间结果仍写回 HBM | 检查 kernel 是否有额外 global 写 | 确保 O 是唯一输出 |
| 官方 FA 比手写快很多 | async copy / 双缓冲 / Tensor Core | 对比 ncu stall reasons | 学习官方实现 |
| 多 batch 下加速不明显 | batch/head 维度未充分并行 | 检查 gridDim | 使用 seq 并行 |
| 编译报错 shared memory 超限 | Br/Bc/d 配置过大 | 计算 SRAM 使用量 | 减小 tile 大小 |
| d=128 比 d=64 慢很多 | Bc 未随 d 调整 | 检查分块参数 | 减小 Bc，增加 warps |

---

> 💡 **Week 4 总结**：本周我们从"为什么 FlashAttention 快"出发，完整推导了 online softmax，手写了支持 batch/multi-head 的 Forward Kernel，分析了官方实现和 FA2 的改进，最终集成到 Mini 引擎并做了系统 benchmark。最核心的收获是建立 IO 优化方法论：当数据移动是瓶颈时， tiling + online algorithm + kernel fusion + recomputation 是通用的解决思路。下周将进入推理系统，学习 KV Cache、vLLM 和 Continuous Batching。
