# Transformer：从零开始理解现代深度学习的基石架构

> **适用对象**：有 Python + PyTorch 基础、了解线性代数与基本 ML 概念，但尚未系统学习 Transformer 架构的开发者；无需 GPU kernel 经验，本专题从模型层"从零"起步
> **本周目标**：从序列建模的演化动机出发，逐步理解 Self-Attention、Multi-Head Attention、Positional Encoding、Transformer Block 的数学原理与工程实现，最终用 PyTorch 从零手写一个可训练的 mini-GPT，打通"论文公式 → 代码实现 → 训练验证"全链路
> **时间投入**：工作日每天 2.5h（早间 1.5h + 晚间 1h），周末每天 5h，周计 22.5h
> **周日里程碑**：用纯 PyTorch 实现一个 6 层 Decoder-only mini-GPT（含 Token Embedding + RoPE + Multi-Head Attention + FFN + LayerNorm），在 Shakespeare 文本上完成训练并生成可读文本，产出架构图笔记与面试问答

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 Transformer 的核心组件（Self-Attention / Multi-Head / Positional Encoding / LayerNorm / FFN / Residual），理解 Encoder-Decoder / Decoder-only / Encoder-only 三种架构变体，能用 PyTorch 从零实现并训练 |
| **核心产出** | ① 手写 Self-Attention（单头 → 多头）② 三种位置编码实现对比（Sinusoidal / Learned / RoPE）③ 完整 Transformer Block ④ mini-GPT 训练 demo ⑤ 架构推导笔记 ⑥ 面试问答集 |
| **验收标准** | ① 能手写 $\text{Attention}(Q,K,V)$ 公式并解释每一步的 shape 变化 ② 能说出 Multi-Head 为什么要拆 head 而不是直接用大维度 ③ 能解释 LayerNorm 的 Pre-Norm vs Post-Norm 差异 ④ mini-GPT 训练 loss 持续下降并生成可读文本 ⑤ 能画出 Decoder-only 的前向数据流 |
| **面试准备** | 积累 10-12 道面试题，覆盖注意力计算复杂度、位置编码演化、LayerNorm vs RMSNorm、KV Cache 直觉、Pre/Post-Norm、与 RNN/CNN 对比 |

### 本专题与 [Week 3 Transformer 执行本质](../../daily/week3/README.md) / [Attention 论文精读](../../paper/attention_is_all_you_need/README.md) 的边界

| 维度 | Week 3（每日教程） | 论文精读 | 本 Transformer 专题 |
|------|---------------------|----------|----------------------|
| **视角** | kernel 层——Softmax/LayerNorm/Attention 的 CUDA 实现 | 论文层——逐节解读原论文的设计动机 | 模型层——从零组装完整 Transformer |
| **范围** | 单个算子的 GPU 优化 | 单篇论文的理论分析 | 完整架构：Embedding → Attention → FFN → 训练 |
| **深度** | 深入到 warp shuffle、shared memory | 深入到公式推导与消融实验 | 深入到每行 PyTorch 代码与 shape 推演 |
| **产出** | 可编译 CUDA kernel | 论文精读笔记 | 可训练 mini-GPT |
| **前置** | Week 1-2 CUDA 基础 | 基本深度学习知识 | Python + PyTorch 基础 |

> 💡 **一句话总结**：Week 3 教你"怎么把 Attention 写成高效 kernel"，论文精读教你"Transformer 为什么这样设计"，本专题教你"怎么从零拼出一个能跑的 Transformer"——三者互补，先读本专题建立模型层全貌，再读论文深挖动机，最后进 Week 3 看 kernel 实现，会如读散文。

### 前置准备清单

#### 软件/环境验证
- [ ] Python >= 3.8
- [ ] PyTorch >= 2.0（`python3 -c "import torch; print(torch.__version__)"`）
- [ ] NumPy / Matplotlib（可视化注意力权重）
- [ ] GPU 非必需（CPU 可跑 mini-GPT 的 small config），有 GPU 更佳

#### 验证命令
```bash
# 验证 PyTorch
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 验证 numpy + matplotlib
python3 -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
```

#### 必读资源（本周会反复用到）
- ⭐ [Attention Is All You Need 论文精读](../../paper/attention_is_all_you_need/README.md) — Transformer 原始论文，本专题的理论骨架
- ⭐ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar 的图解，建立直觉的最佳入门
- ⭐ [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP 的逐行代码注解，本专题 Day 7 的对标
- 📌 [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy 的视频教程，mini-GPT 的灵感来源
- 📌 [minGPT](https://github.com/karpathy/minGPT) / [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的参考实现
- 📌 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau 2014，Attention 机制的前身

---

## 为什么学 Transformer

Transformer 是现代大模型（GPT / LLaMA / DeepSeek / Qwen）的共同骨架。理解它的架构不是为了"多学一个模型"，而是为了：

| 场景 | 不懂 Transformer | 懂 Transformer |
|------|-------------------|-----------------|
| **写 Attention kernel** | 照抄公式，不知道为什么 Q·K 要除以 $\sqrt{d_k}$ | 知道是方差稳定化，数值不溢出才能 softmax |
| **做推理优化** | 不知道 KV Cache 为什么只缓存 K/V 不缓存 Q | 理解自回归生成时 Q 每步变、K/V 只增不减 |
| **读 vLLM / TensorRT-LLM 源码** | 看到 PagedAttention 一头雾水 | 知道它在分块管理 K/V，逻辑就是分页内存 |
| **调大模型超参** | 不知道 num_heads / head_dim 怎么选 | 知道 head 数影响子空间多样性，head_dim 影响单头表达力 |
| **面试** | 只能背"Transformer 是编码器-解码器" | 能从 RNN 的长程依赖缺陷推到 Attention 的解决方案 |

> 💡 **一句话总结**：Transformer 是 AI Infra 的"OS"——你不需要会写 OS 才能写程序，但理解 OS 能让你写出更好的程序。同理，理解 Transformer 架构能让你写出更好的 kernel、做出更好的系统优化、答好更多的面试题。

---

## 核心概念速览

### 1. 序列建模的演化：从 RNN 到 Attention

#### RNN 的痛点：长程依赖与串行计算

RNN/LSTM 通过隐藏状态 $h_t = f(h_{t-1}, x_t)$ 传递信息，存在两个根本问题：

- **长程依赖衰减**：信息经过多步传递后逐渐丢失，梯度也随时间步消失/爆炸
- **串行计算**：$h_t$ 依赖 $h_{t-1}$，无法并行，训练速度受限于序列长度

#### Attention 的破局：全局视野 + 并行计算

Attention 机制让每个位置直接"看到"序列中所有其他位置，一步计算全局依赖：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 维度 | RNN / LSTM | Self-Attention |
|------|------------|----------------|
| 依赖范围 | 相邻步传递，长程靠隐状态间接携带 | 任意两位置直接交互，一步到位 |
| 并行性 | 串行，$O(n)$ 时间步 | 全并行，$O(1)$ 时间步（矩阵乘法） |
| 复杂度 | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$（序列长时更贵） |
| 位置信息 | 隐含在计算顺序中 | 需额外注入（Positional Encoding） |

> ⚠️ **注意**：Attention 的 $O(n^2)$ 复杂度是后续 FlashAttention / Sparse Attention / Linear Attention 等优化的动机，但这些优化都是在"理解标准 Attention"之后的事。本专题先把标准版吃透。

### 2. Self-Attention：QKV 矩阵乘法的本质

#### 三个角色的分工

给定输入 $X \in \mathbb{R}^{n \times d}$（$n$ 个 token，每个 $d$ 维）：

| 角色 | 生成方式 | 语义 | Shape |
|------|----------|------|-------|
| **Query** $Q$ | $X W^Q$ | "我在找什么" | $(n, d_k)$ |
| **Key** $K$ | $X W^K$ | "我能提供什么" | $(n, d_k)$ |
| **Value** $V$ | $X W^V$ | "我实际携带的信息" | $(n, d_v)$ |

#### 计算流程（4 步）

1. **打分**：$S = QK^T$ — 每个 query 与所有 key 做点积，得到注意力分数矩阵 $(n, n)$
2. **缩放**：$S = S / \sqrt{d_k}$ — 防止点积过大导致 softmax 梯度消失
3. **归一化**：$A = \text{softmax}(S)$ — 每行归一化，得到注意力权重（概率分布）
4. **加权求和**：$O = AV$ — 用注意力权重对 value 加权求和，得到输出 $(n, d_v)$

> 💡 **为什么除以 $\sqrt{d_k}$？** 当 $d_k$ 较大时，$QK^T$ 的方差正比于 $d_k$，点积值会很大。softmax 对大值敏感（梯度趋近于 0），除以 $\sqrt{d_k}$ 将方差拉回 1，保持梯度稳定。

#### Shape 变化速查

```
X: (n, d)                    ← 输入序列
Q = X @ Wq: (n, d_k)         ← 查询
K = X @ Wk: (n, d_k)         ← 键
V = X @ Wv: (n, d_v)         ← 值
S = Q @ K^T: (n, n)          ← 注意力分数
A = softmax(S): (n, n)       ← 注意力权重
O = A @ V: (n, d_v)          ← 输出
```

### 3. Multi-Head Attention：子空间并行关注

#### 为什么需要多头

单头 Attention 用全部 $d$ 维做点积，所有"关注模式"混在一起。多头把 $d$ 维拆成 $h$ 个 $\frac{d}{h}$ 维子空间，每个 head 独立做 Attention，让模型同时关注不同类型的关系（如语法依赖、语义相似、位置模式）。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

#### 关键细节：总计算量不变

| 配置 | 单头 | 多头（$h$ 个 head） |
|------|------|----------------------|
| 每 head 维度 | $d$ | $d / h$ |
| head 数量 | 1 | $h$ |
| 单 head 点积复杂度 | $O(n^2 \cdot d)$ | $O(n^2 \cdot d/h)$ |
| 总复杂度 | $O(n^2 \cdot d)$ | $O(n^2 \cdot d)$（不变） |
| 参数量 | $3 \times d \times d$ | $3 \times d \times d$（不变） |

> 💡 **一句话总结**：多头不是"多算"，而是"换一种切分方式"——把 $d$ 维空间拆成 $h$ 个子空间，每个子空间独立学一种关注模式，最后拼回来。总 FLOPs 和参数量与单头相同。

### 4. 位置编码：给无序的注意力注入顺序

Self-Attention 是**置换等变**的——打乱输入顺序，输出只是对应打乱，内容不变。要让模型感知位置，必须额外注入位置信息。

#### 三种主流方案

| 方案 | 公式 | 特点 | 代表模型 |
|------|------|------|----------|
| **Sinusoidal** | $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$ | 固定不学习，可外推到更长序列 | 原始 Transformer |
| **Learned** | $PE_{pos} \in \mathbb{R}^d$ 为可学习参数 | 灵活但无法外推超过训练长度 | GPT-2 / BERT |
| **RoPE** | 对 $Q, K$ 旋转：$q'_m = q_m e^{im\theta}$ | 相对位置编码，外推性好 | LLaMA / Qwen / DeepSeek |

#### RoPE 直觉

RoPE 不直接给输入加位置，而是对 Query 和 Key 做**旋转**——位置 $m$ 的 query 旋转角度 $m\theta$，位置 $n$ 的 key 旋转角度 $n\theta$。点积 $q'_m \cdot k'_n$ 只依赖相对位置 $m - n$：

$$q'_m \cdot k'_n = q_m \cdot k_n \cdot \cos((m-n)\theta) + \text{cross terms}$$

> 💡 **为什么 RoPE 成为现代标配？** ① 相对位置比绝对位置更符合语言直觉 ② 可外推（配合 NTK-aware scaling）③ 不引入额外参数 ④ 对 KV Cache 友好（旋转可预计算）

### 5. Transformer Block：LayerNorm + FFN + 残差

一个完整的 Transformer Block 由三部分组成：

#### 残差连接

$$x_{\text{out}} = x + \text{SubLayer}(x)$$

残差连接让梯度可以"跳过"子层直接反传，缓解深层网络的梯度消失。GPT 系列用 **Pre-Norm**（先 LayerNorm 再进子层），原始 Transformer 用 **Post-Norm**（子层后再 LayerNorm）。

| 方案 | 公式 | 训练稳定性 | 代表模型 |
|------|------|------------|----------|
| **Post-Norm** | $x' = \text{LN}(x + \text{SubLayer}(x))$ | 不稳定，需 warmup | 原始 Transformer / BERT |
| **Pre-Norm** | $x' = x + \text{SubLayer}(\text{LN}(x))$ | 稳定，可省 warmup | GPT-2 / LLaMA / DeepSeek |

#### FFN（Feed-Forward Network）

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

FFN 是两层线性变换 + 激活，中间维度通常是 $4d$（如 $d=768$ 则 FFN 隐层 $3072$）。FFN 是 Transformer 的"记忆容量"来源——Attention 负责信息路由，FFN 负责信息变换。

| 激活函数 | 公式 | 特点 | 代表模型 |
|----------|------|------|----------|
| ReLU | $\max(0, x)$ | 简单，但有神经元死亡 | 原始 Transformer |
| GELU | $x \cdot \Phi(x)$ | 平滑，现代标配 | GPT-2 / BERT |
| SwiGLU | $\text{Swish}(xW_1) \otimes (xW_2)$ | GLU 变体，更强表达力 | LLaMA / DeepSeek |

> 💡 **为什么 FFN 中间维度是 4d？** 经验值。Attention 做的是"加权平均"（输出仍在 $d$ 维空间），FFN 做的是"非线性变换"，需要更大的中间维度提供足够的表达力。$4d$ 是大量实验后的 sweet spot。

---

## 最小可运行示例：手写 Self-Attention

```python
# self_attention.py —— 从零实现 Self-Attention
# 运行: python3 self_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """单头自注意力：Q/K/V 共享输入"""

    def __init__(self, embed_dim):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        B, T, C = x.shape

        qkv = self.qkv(x)                          # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)             # each: (B, T, C)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn = F.softmax(scores, dim=-1)                  # (B, T, T)
        out = attn @ v                                    # (B, T, C)

        out = self.proj(out)                              # (B, T, C)
        return out, attn


class MultiHeadAttention(nn.Module):
    """多头自注意力：拆 head → 各自 attention → 拼回"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)                                   # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)                      # each: (B, T, C)

        # 拆 head: (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale      # (B, num_heads, T, T)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                        # (B, num_heads, T, head_dim)

        # 拼回: (B, num_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out, attn


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, embed_dim, num_heads = 2, 10, 64, 8
    x = torch.randn(batch, seq_len, embed_dim)

    mha = MultiHeadAttention(embed_dim, num_heads)
    out, attn = mha(x)

    print(f"输入 shape:  {x.shape}")
    print(f"输出 shape:  {out.shape}")
    print(f"注意力权重 shape: {attn.shape}")
    print(f"每行注意力权重和: {attn[0, 0, 0, :].sum().item():.4f}")  # 应为 1.0
```

```bash
python3 self_attention.py
```

```text
输入 shape:  torch.Size([2, 10, 64])
输出 shape:  torch.Size([2, 10, 64])
注意力权重 shape: torch.Size([2, 8, 10, 10])
每行注意力权重和: 1.0000
```

---

## 本周学习计划

| 天数 | 主题 | 核心概念 | 核心产出 |
|------|------|----------|----------|
| Day 1 | 序列建模演化与 Attention 动机 | RNN 长程依赖、并行性瓶颈、Bahdanau Attention | RNN vs Attention 对比笔记 |
| Day 2 | Self-Attention 数学推导 | QKV 矩阵乘法、缩放因子、softmax 归一化 | 手写单头 Self-Attention |
| Day 3 | Multi-Head Attention | 子空间拆分、head 拼接、参数量分析 | 手写多头 Attention + 可视化 |
| Day 4 | 位置编码 | Sinusoidal / Learned / RoPE 三种方案对比 | 三种位置编码实现 |
| Day 5 | Transformer Block 组装 | LayerNorm、FFN、残差、Pre/Post-Norm | 完整 Transformer Block |
| Day 6 | 完整架构与变体 | Encoder-Decoder / Decoder-only / Encoder-only | 三种架构对比笔记 |
| Day 7 | 从零实现 mini-GPT + 训练 | Token Embedding、RoPE、生成式推理 | 可训练 mini-GPT demo |

---

## 面试要点

**Q：Self-Attention 为什么要除以 $\sqrt{d_k}$？**
> 当 $d_k$ 较大时，$QK^T$ 的方差正比于 $d_k$，点积值变大，softmax 进入饱和区（梯度趋近 0）。除以 $\sqrt{d_k}$ 将方差拉回 1，保持梯度稳定。直觉上就是"维度越高，点积越大，需要更强的缩放"。

**Q：Multi-Head Attention 的计算量和参数量与单头相比如何？**
> 完全相同。多头把 $d$ 维拆成 $h$ 个 $d/h$ 维子空间，每个 head 的 $W_i^Q, W_i^K, W_i^V$ 是 $d \times (d/h)$，所有 head 加起来还是 $d \times d$。计算量同理，$h$ 个 $O(n^2 \cdot d/h)$ 的 head 加起来还是 $O(n^2 \cdot d)$。多头改变的是"关注模式的多样性"，不是"计算量"。

**Q：为什么现代大模型都用 Pre-Norm 而不是 Post-Norm？**
> Pre-Norm 的残差路径不经过 LayerNorm，梯度可以直接从输出流回输入，训练更稳定，不需要 warmup。Post-Norm 的残差经过 LayerNorm，深层网络梯度容易消失，训练不稳定但最终效果可能更好（如果训得动）。大模型优先稳定性，所以选 Pre-Norm。

**Q：RoPE 相比 Sinusoidal 位置编码有什么优势？**
> ① RoPE 编码的是相对位置（$q_m \cdot k_n$ 只依赖 $m-n$），更符合语言直觉 ② 可外推到训练时未见的序列长度 ③ 不引入额外可学习参数 ④ 旋转矩阵可预计算，对 KV Cache 友好。Sinusoidal 虽然也可外推，但它是加在输入上的绝对位置编码，表达力不如相对位置。

**Q：Attention 的计算复杂度是多少？为什么这是个问题？**
> Self-Attention 的计算复杂度是 $O(n^2 \cdot d)$，其中 $n$ 是序列长度。$n^2$ 来自 $QK^T$ 这个 $n \times n$ 矩阵乘法。当 $n$ 很大时（如长文档 $n=128K$），$n^2$ 项爆炸，显存和计算都不可承受。这是 FlashAttention（分块计算不实例化 $n \times n$ 矩阵）、Sparse Attention（稀疏化注意力矩阵）、Linear Attention（用核函数近似 softmax 降低到 $O(n)$）等方法的动机。

**Q：KV Cache 的原理是什么？为什么推理时只需要缓存 K/V？**
> 自回归生成时，每生成一个 token，之前的 token 的 K/V 不变（因为 $W^K, W^V$ 不变，输入不变）。所以可以把已计算的 K/V 缓存起来，新 token 只需计算自己的 $q, k, v$，然后 $q_{\text{new}}$ 与缓存的 $K$ 做点积。不需要缓存 Q，因为 Q 每步都不同（只属于当前 token）。

**Q：LayerNorm 和 RMSNorm 有什么区别？**
> LayerNorm 减均值再除标准差：$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$。RMSNorm 只除 RMS（均方根），不减均值：$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}$。RMSNorm 少了减均值和 $\beta$，计算更快，实验证明效果与 LayerNorm 相当。LLaMA / DeepSeek 等现代模型用 RMSNorm。

**Q：FFN 的中间维度为什么通常是 4d？SwiGLU 为什么用 3 个矩阵？**
> 4d 是经验值，提供足够的非线性表达力。SwiGLU 公式是 $\text{Swish}(xW_1) \otimes (xW_2)$，需要两个"升维"矩阵 $W_1, W_2$（各 $d \times 4d$）加一个"降维"矩阵 $W_3$（$4d \times d$），共 3 个矩阵。标准 FFN 只有 2 个矩阵（$W_1: d \times 4d$, $W_2: 4d \times d$）。SwiGLU 多一个矩阵但表达力更强，为保持总参数量不变，中间维度从 $4d$ 降到 $\frac{8d}{3}$。

**Q：Encoder-Decoder、Decoder-only、Encoder-only 三种架构分别适合什么任务？**
> ① Encoder-Decoder（原始 Transformer / T5）：适合 seq2seq 任务（翻译、摘要），Encoder 双向看输入，Decoder 自回归生成输出 ② Decoder-only（GPT / LLaMA）：适合自回归生成（对话、续写），单向注意力 + KV Cache 推理高效 ③ Encoder-only（BERT）：适合理解任务（分类、NER），双向注意力看到完整上下文。现代大模型几乎都用 Decoder-only，因为生成任务通用性最强且 scaling law 最优。

---

## 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [Attention Is All You Need 论文精读](../../paper/attention_is_all_you_need/README.md) | 论文精读 | ⭐ 必读 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解博客 | ⭐ 必读 |
| [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) | 逐行代码 | ⭐ 必读 |
| [Let's build GPT (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 视频教程 | ⭐ 必读 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 参考实现 | ⭐ 必读 |
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | 论文 | 📌 推荐 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 论文 | 📌 推荐 |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | 论文 | 📌 推荐 |
| [Week 3 Transformer 执行本质](../../daily/week3/README.md) | 每日教程 | 📎 衔接 |
| [Week 4 FlashAttention 深挖](../../daily/week4/day1/README.md) | 每日教程 | 📎 衔接 |

---

## 目录结构

```
aiinfra/topics/transformer/
├── README.md                    # 本文件（专题概览 + 学习计划）
├── kernels/                     # 可运行代码示例
│   ├── self_attention.py        # Day 2-3: 手写 Self-Attention / MHA
│   ├── positional_encoding.py   # Day 4: 三种位置编码实现
│   ├── transformer_block.py     # Day 5: 完整 Transformer Block
│   └── mini_gpt.py              # Day 7: 从零实现 mini-GPT
├── notes/                       # 推导笔记与架构分析
│   ├── attention_math.md        # Day 2: Attention 数学推导
│   ├── positional_encoding.md   # Day 4: 位置编码演化
│   └── architecture_variants.md # Day 6: 三种架构变体对比
└── benchmark/                   # 性能对比
    └── attention_benchmark.py   # 各组件性能基准
```

> 💡 **后续延伸**：完成本专题后，建议进入 [Week 3 Transformer 执行本质](../../daily/week3/README.md) 看 Softmax / LayerNorm / Attention 的 CUDA kernel 实现——你会发现 kernel 优化的每一步都对应着你在这里学到的数学公式。再读 [Week 4 FlashAttention](../../daily/week4/day1/README.md) 时，"为什么 FlashAttention 要分块计算"会变得显而易见——因为它在解决你这里学到的 $O(n^2)$ 显存问题。配合 [FlashAttention 论文精读](../../paper/flashattention/README.md)，你能从"模型公式"一路打通到"GPU 内存层级优化"。
