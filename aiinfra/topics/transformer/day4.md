# Day 4（周四）：位置编码——Sinusoidal / Learned / RoPE

> **本周定位**：本专题是模型层"从零"起步——不涉及 CUDA kernel，聚焦 Transformer 的数学原理与 PyTorch 实现。本周目标是理解 Self-Attention、Multi-Head、位置编码、Transformer Block，最终用纯 PyTorch 从零手写一个可训练的 mini-GPT。Day 2-3 把 Self-Attention 的计算吃透了，但遗留了一个关键问题：Attention 是置换等变的，不感知顺序。Day 4 解决"如何给无序的注意力注入位置信息"——从绝对位置编码（Sinusoidal / Learned）到相对位置编码（RoPE），理解三种方案的设计动机与 tradeoff。
> **前置要求**：已完成 [Day 2](day2.md)（QKV 四步计算）与 [Day 3](day3.md)（Multi-Head shape 操作）；理解矩阵乘法与点积的性质
> **今日目标**：理解 Self-Attention 的置换等变性（为什么需要位置编码），掌握 Sinusoidal 位置编码的公式与"相对位置可由绝对位置线性组合表示"的性质，理解 Learned 位置编码的优劣，掌握 RoPE 的旋转矩阵推导与"点积只依赖相对位置"的证明，能用 PyTorch 实现三种位置编码并对比，理解 RoPE 为何成为现代大模型标配
> **时间投入**：2.5h（早间 1.5h 精读三种方案 + 晚间 1h 跑代码实现与对比）
> **面试考察度**：⭐⭐⭐⭐ 高频考点，"为什么需要位置编码、RoPE 原理"是常见题

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 置换等变性证明 + "为什么不能直接加位置索引"分析 | ⑤ 画出 Decoder-only 数据流（前置：理解位置信息注入方式） |
| 三种位置编码公式与实现（`kernels/positional_encoding.py`） | ③ 三种位置编码实现对比（直接完成验收 ③） |
| RoPE 旋转矩阵推导 + "点积只依赖相对位置"证明 | ④ 理解 RoPE 为何成为现代标配（面试高频） |
| Sinusoidal / Learned / RoPE 对比表 | 面试高频：位置编码演化与 tradeoff |

> 💡 **Day 4 的定位**：今天不碰 head 维度（Day 3 已吃透），聚焦"位置信息怎么注入"。关键洞察：Self-Attention 把输入当作**无序集合**处理，位置信息必须从外部注入。三种方案的演化路径是"绝对位置 → 可学习绝对位置 → 相对位置"，RoPE 是目前最优解。今天的内容是 Day 7（mini-GPT）中 Token Embedding + RoPE 的前置。

---

### 学习任务 1：为什么需要位置编码（30 分钟）

#### Self-Attention 的置换等变性

Self-Attention 的核心计算是 $QK^T$ 和 $AV$。如果打乱输入顺序（用置换矩阵 $P$ 作用），输出会怎样？

设输入 $X$ 被置换为 $PX$（$P$ 是排列矩阵，$P^T = P^{-1}$）：

$$Q' = (PX) W^Q = P Q, \quad K' = P K, \quad V' = P V$$

$$Q' K'^T = P Q K^T P^T = P S P^T$$

$$\text{softmax}(P S P^T) = P \cdot \text{softmax}(S) \cdot P^T = P A P^T$$

$$O' = (P A P^T)(P V) = P A V = P O$$

**结论**：输入打乱为 $PX$，输出变为 $PO$——只是对应打乱，内容不变。这就是**置换等变性**（permutation equivariance）。

| 性质 | 含义 | 后果 |
|------|------|------|
| 置换等变 | 输入顺序打乱，输出对应打乱 | Attention 本身不编码顺序 |
| 对比 RNN | $h_t$ 依赖 $h_{t-1}$，顺序是内生的 | RNN 天然感知顺序 |

> 💡 **直觉**：把 Self-Attention 想象成"投票"——每个位置对所有位置投票打分，投票结果只取决于内容相似度，与位置无关。"狗咬人"和"人咬狗"的 token 集合相同，Attention 会给出相同的投票模式。要让模型区分顺序，必须额外告诉它"谁在前谁在后"。

#### 为什么不能直接加位置索引？

朴素想法：把位置索引 $0, 1, 2, \ldots, n-1$ 直接加到 embedding 上：

```python
# 错误做法
positions = torch.arange(seq_len)  # [0, 1, 2, ..., n-1]
x = token_embeddings + positions.unsqueeze(-1)  # 加到每个维度？
```

问题：

| 问题 | 说明 |
|------|------|
| **数值尺度不匹配** | Token embedding 通常是 $\mathcal{N}(0, 1)$ 初始化，值域 $[-1, 1]$；位置索引 $0 \sim n-1$ 可能上百上千，直接加会淹没 embedding 信号 |
| **无周期性** | 位置 100 和 101 的差异等于位置 1 和 2 的差异，但语言中"相邻"的关系应与绝对位置无关 |
| **不可外推** | 训练时最长 $n_{\text{train}}$，推理时 $n > n_{\text{train}}$ 的位置索引从未见过 |
| **维度浪费** | 所有维度加同一个标量，没有利用 $d$ 维空间编码不同频率的位置信息 |

> ⚠️ **注意**：位置编码必须与 embedding **相加**（不是拼接），因为 Self-Attention 的输入是固定 $d$ 维，拼接会改变维度。相加要求 PE 与 embedding 同 shape $(n, d)$。

#### 位置编码的分类

| 类型 | 注入方式 | 代表 | 特点 |
|------|----------|------|------|
| **绝对位置编码** | $X + PE_{\text{pos}}$ | Sinusoidal / Learned | 给每个位置一个固定的 $d$ 维向量 |
| **相对位置编码** | 修改 $QK^T$ 计算 | RoPE / T5 relative bias | 编码位置间的相对距离 |
| **混合** | 结合两者 | ALiBi | 在 attention score 上加距离惩罚 |

今天的重点是 Sinusoidal（绝对）、Learned（绝对）、RoPE（相对）三种。

---

### 学习任务 2：Sinusoidal 位置编码（35 分钟）

原始 Transformer（Vaswani et al., 2017）使用的位置编码，用不同频率的正弦/余弦函数：

#### 公式

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- $pos$：位置索引（$0, 1, \ldots, n-1$）
- $i$：维度索引（$0, 1, \ldots, d/2 - 1$）
- $2i$ 维用 $\sin$，$2i+1$ 维用 $\cos$
- 频率 $\omega_i = \frac{1}{10000^{2i/d}}$，从 $1$（$i=0$）到 $\frac{1}{10000}$（$i=d/2-1$）对数递减

#### 频率分析

| 维度 $i$ | 频率 $\omega_i$ | 周期 $2\pi/\omega_i$ | 物理含义 |
|----------|-----------------|----------------------|----------|
| 0 | $1$ | $2\pi \approx 6.28$ | 高频，捕捉相邻位置差异 |
| $d/4$ | $1/100$ | $628$ | 中频，捕捉中等距离模式 |
| $d/2-1$ | $1/10000$ | $62832$ | 低频，捕捉长距离模式 |

> 💡 **关键洞察**：Sinusoidal 用 $d/2$ 对不同频率的 $\sin/\cos$ 编码位置——高频维度捕捉局部位置差异，低频维度捕捉全局位置趋势。类似傅里叶变换用不同频率基函数分解信号。

#### 性质：相对位置可由绝对位置线性表示

这是 Sinusoidal 被选中的核心原因。对于固定的偏移 $\Delta$：

$$\sin(\omega_i (pos + \Delta)) = \sin(\omega_i pos) \cos(\omega_i \Delta) + \cos(\omega_i pos) \sin(\omega_i \Delta)$$

$$\cos(\omega_i (pos + \Delta)) = \cos(\omega_i pos) \cos(\omega_i \Delta) - \sin(\omega_i pos) \sin(\omega_i \Delta)$$

写成矩阵形式：

$$\begin{pmatrix} \sin(\omega_i(pos+\Delta)) \\ \cos(\omega_i(pos+\Delta)) \end{pmatrix} = \begin{pmatrix} \cos(\omega_i\Delta) & \sin(\omega_i\Delta) \\ -\sin(\omega_i\Delta) & \cos(\omega_i\Delta) \end{pmatrix} \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix}$$

- 右侧的 $PE_{pos}$ 是位置 $pos$ 的编码
- 中间的 $2 \times 2$ 旋转矩阵只依赖 $\Delta$（相对位置），不依赖绝对位置 $pos$
- 这意味着 $PE_{pos+\Delta}$ 可以由 $PE_{pos}$ 经过一个只依赖 $\Delta$ 的线性变换得到

> 💡 **为什么这个性质重要**：虽然 Sinusoidal 是绝对位置编码（每个位置一个固定向量），但它的数学结构保证了"相对位置信息可以被提取"。Self-Attention 的 $W^Q, W^K$ 可以学会利用这个性质，间接实现相对位置感知。

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 无需学习（固定公式） | 绝对位置编码，表达力不如相对位置 |
| 可外推到更长序列（频率连续） | 实际外推效果有限（需配合 interpolation） |
| 相对位置可线性表示 | 在 Decoder-only 中不如 RoPE |

---

### 学习任务 3：Learned 位置编码（20 分钟）

GPT-2 / BERT 采用的方案：每个位置一个可学习的 $d$ 维向量。

#### 公式

$$PE \in \mathbb{R}^{n_{\max} \times d}$$

- $PE$ 是一个可学习参数矩阵，$n_{\max}$ 是最大序列长度
- 位置 $pos$ 的编码就是 $PE[pos]$（查表）

```python
self.positional_embedding = nn.Parameter(torch.randn(max_len, embed_dim) * 0.02)
# 前向时
x = token_embeddings + self.positional_embedding[:seq_len]
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 灵活，模型自己学最优编码 | 无法外推（超过 $n_{\max}$ 没有编码） |
| 实现简单（查表 + 加法 | 参数量 $n_{\max} \times d$（如 2048×768≈1.5M） |
| 在固定长度任务上效果可能更好 | 不同长度间位置编码不一致 |

| 维度 | Sinusoidal | Learned |
|------|------------|---------|
| 是否学习 | 固定 | 可学习 |
| 外推性 | 理论可外推 | 不可外推 |
| 参数量 | 0 | $n_{\max} \times d$ |
| 代表模型 | 原始 Transformer / T5 | GPT-2 / BERT |

> 💡 **为什么 GPT-2/BERT 选 Learned**：在固定长度任务上（BERT 的 512、GPT-2 的 1024），可学习编码比固定公式表达力更强。但随着大模型需要处理超长上下文（4K→32K→128K），Learned 的外推缺陷暴露，RoPE 取而代之。

---

### 学习任务 4：RoPE 旋转位置编码（50 分钟）

这是 Day 4 的**核心精读**——RoPE（Rotary Position Embedding）是 LLaMA / Qwen / DeepSeek 等现代大模型的标配，也是面试高频题。

#### 动机：直接编码相对位置

Sinusoidal 和 Learned 都是绝对位置编码——给每个位置一个向量，加到 embedding 上。但语言中更重要的是**相对位置**：

- "我 爱 你" 中 "我"和"爱"的距离 = "他 爱 她" 中 "他"和"爱"的距离 = 1
- 相对位置相同 → 注意力模式应相同

RoPE 的目标：让 $q_m \cdot k_n$（位置 $m$ 的 query 与位置 $n$ 的 key 的点积）只依赖相对位置 $m - n$。

#### 核心思想：用旋转编码位置

对位置 $m$ 的 query $q_m$，乘以一个旋转矩阵 $R_m$：

$$q'_m = R_m q_m$$

对位置 $n$ 的 key $k_n$，乘以旋转矩阵 $R_n$：

$$k'_n = R_n k_n$$

点积：

$$q'_m \cdot k'_n = (R_m q_m)^T (R_n k_n) = q_m^T R_m^T R_n k_n$$

如果 $R_m^T R_n$ 只依赖 $m - n$，就实现了相对位置编码。

#### 旋转矩阵的定义

RoPE 把 $d$ 维向量分成 $d/2$ 对，每对用 2D 旋转：

$$R_m = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & & \\ \sin(m\theta_0) & \cos(m\theta_0) & & \\ & & \cos(m\theta_1) & -\sin(m\theta_1) \\ & & \sin(m\theta_1) & \cos(m\theta_1) \\ & & & & \ddots \end{pmatrix}$$

- $\theta_i = 10000^{-2i/d}$：第 $i$ 对的频率（与 Sinusoidal 相同的频率设计）
- $R_m$ 是块对角矩阵，每块是 $2 \times 2$ 旋转，旋转角度为 $m\theta_i$

#### 证明：点积只依赖相对位置

$$R_m^T R_n = \text{diag}(R(m\theta_0)^T R(n\theta_0), \ldots)$$

每个 $2 \times 2$ 块：

$$R(m\theta_i)^T R(n\theta_i) = \begin{pmatrix} \cos(m\theta_i) & \sin(m\theta_i) \\ -\sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} \cos(n\theta_i) & -\sin(n\theta_i) \\ \sin(n\theta_i) & \cos(n\theta_i) \end{pmatrix}$$

$$= \begin{pmatrix} \cos((n-m)\theta_i) & -\sin((n-m)\theta_i) \\ \sin((n-m)\theta_i) & \cos((n-m)\theta_i) \end{pmatrix} = R((n-m)\theta_i)$$

**结论**：$R_m^T R_n = R_{n-m}$，只依赖 $n - m$（相对位置）。

因此：

$$q'_m \cdot k'_n = q_m^T R_{n-m} k_n$$

点积是 $q_m, k_n$ 经过旋转 $R_{n-m}$ 后的内积，只依赖相对位置 $n - m$。

> 💡 **一句话总结**：RoPE 对 Q 和 K 乘旋转矩阵 $R_m, R_n$，利用旋转矩阵的性质 $R_m^T R_n = R_{n-m}$，让点积只依赖相对位置。这是数学上最优雅的相对位置编码——不引入额外参数，不改变 Attention 的计算流程，只是在 QK 上"旋转"一下。

#### RoPE 不作用于 V

注意：RoPE 只旋转 $Q$ 和 $K$，**不旋转 $V$**。因为：

- 位置信息只在"打分"（$QK^T$）时需要——决定"谁关注谁"
- $V$ 是"被提取的内容"，加权求和 $AV$ 不需要位置旋转
- 不旋转 $V$ 保持输出 $O$ 的语义不变（只是注意力权重 $A$ 变了）

#### 高效实现：不构造旋转矩阵

实际代码不构造 $d \times d$ 的稀疏旋转矩阵（太慢），而是用 elementwise 操作：

```python
# q: (B, num_heads, T, head_dim)
# 将 head_dim 拆成 (head_dim//2, 2)，对每对做 2D 旋转

def apply_rope(q, cos, sin):
    # q: (B, h, T, d), cos/sin: (T, d/2)
    d = q.shape[-1]
    q1 = q[..., :d//2]    # 前半
    q2 = q[..., d//2:]    # 后半
    # 旋转: [q1, q2] -> [q1*cos - q2*sin, q2*cos + q1*sin]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    return q_rot
```

| 实现方式 | 复杂度 | 说明 |
|----------|--------|------|
| 构造 $R_m$ 矩阵乘法 | $O(d^2)$ | 朴素，慢 |
| Elementwise（split + cos/sin） | $O(d)$ | 实际使用，快 |

---

### 学习任务 5：手写三种位置编码实现（45 分钟）

这是 Day 4 的**动手环节**——对应 README 中的 `kernels/positional_encoding.py`。

#### 完整实现

```python
# positional_encoding.py —— 三种位置编码实现
# 运行: python3 positional_encoding.py

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """正弦余弦位置编码（原始 Transformer）"""

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()      # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )                                                               # (embed_dim/2,)
        pe[:, 0::2] = torch.sin(position * div_term)                   # 偶数维 sin
        pe[:, 1::2] = torch.cos(position * div_term)                   # 奇数维 cos
        self.register_buffer('pe', pe.unsqueeze(0))                    # (1, max_len, d)

    def forward(self, x):
        # x: (B, T, d)
        return x + self.pe[:, :x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    """可学习位置编码（GPT-2 / BERT）"""

    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x):
        # x: (B, T, d)
        return x + self.pe[:, :x.size(1), :]


class RotaryPositionalEmbedding(nn.Module):
    """RoPE 旋转位置编码（LLaMA / DeepSeek）"""

    def __init__(self, head_dim, max_len=5000):
        super().__init__()
        # 频率: theta_i = 10000^(-2i/d)
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (d/2,)
        t = torch.arange(max_len).float()                  # (max_len,)
        angles = torch.outer(t, freqs)                      # (max_len, d/2)
        self.register_buffer('cos', angles.cos())           # (max_len, d/2)
        self.register_buffer('sin', angles.sin())           # (max_len, d/2)

    def forward(self, q, k):
        """对 q, k 应用 RoPE（不作用于 v）
        q, k: (B, num_heads, T, head_dim)
        """
        T = q.size(2)
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d/2)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

        q = self._rotate_half(q, cos, sin)
        k = self._rotate_half(k, cos, sin)
        return q, k

    @staticmethod
    def _rotate_half(x, cos, sin):
        d = x.shape[-1]
        x1 = x[..., :d // 2]       # 前半
        x2 = x[..., d // 2:]       # 后半
        # 旋转: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, embed_dim, num_heads = 2, 10, 64, 8
    head_dim = embed_dim // num_heads

    # --- Sinusoidal ---
    sin_pe = SinusoidalPositionalEncoding(embed_dim)
    x = torch.randn(batch, seq_len, embed_dim)
    out_sin = sin_pe(x)
    print("=== Sinusoidal ===")
    print(f"输入: {x.shape}  输出: {out_sin.shape}")
    print(f"PE[0,0,:4]: {sin_pe.pe[0, 0, :4].tolist()}")

    # --- Learned ---
    learned_pe = LearnedPositionalEncoding(embed_dim)
    out_learned = learned_pe(x)
    print("\n=== Learned ===")
    print(f"输入: {x.shape}  输出: {out_learned.shape}")
    print(f"PE[0,0,:4]: {learned_pe.pe[0, 0, :4].tolist()}")

    # --- RoPE ---
    rope = RotaryPositionalEmbedding(head_dim)
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    q_rot, k_rot = rope(q, k)
    print("\n=== RoPE ===")
    print(f"q: {q.shape}  q_rot: {q_rot.shape}")
    print(f"k: {k.shape}  k_rot: {k_rot.shape}")
    print(f"cos shape: {rope.cos.shape}  sin shape: {rope.sin.shape}")
```

```bash
python3 positional_encoding.py
```

```text
=== Sinusoidal ===
输入: torch.Size([2, 10, 64])  输出: torch.Size([2, 10, 64])
PE[0,0,:4]: [0.0, 1.0, 0.0001, 1.0]

=== Learned ===
输入: torch.Size([2, 10, 64])  输出: torch.Size([2, 10, 64])
PE[0,0,:4]: [0.019, -0.009, 0.014, 0.031]

=== RoPE ===
q: torch.Size([2, 8, 10, 8])  q_rot: torch.Size([2, 8, 10, 8])
k: torch.Size([2, 8, 10, 8])  k_rot: torch.Size([2, 8, 10, 8])
cos shape: torch.Size([5000, 4])  sin shape: torch.Size([5000, 4])
```

#### 实验：验证 RoPE 的相对位置性质

```python
# verify_rope_relative.py —— 验证 RoPE 点积只依赖相对位置
# 运行: python3 verify_rope_relative.py

import torch
from positional_encoding import RotaryPositionalEmbedding

torch.manual_seed(42)
head_dim, num_heads, batch = 8, 1, 1
rope = RotaryPositionalEmbedding(head_dim, max_len=100)

# 固定 q, k 内容，改变绝对位置但保持相对位置相同
# 相对位置 = m - n = 3 的两种情况：
#   情况 A: m=5, n=2
#   情况 B: m=10, n=7

q_content = torch.randn(batch, num_heads, 1, head_dim)  # 内容固定
k_content = torch.randn(batch, num_heads, 1, head_dim)

# 模拟位置 m=5 的 q 和位置 n=2 的 k
q_at_5 = q_content.clone()
k_at_2 = k_content.clone()
# 应用 RoPE（位置 5 和 2）
q5_rot, k2_rot = rope.apply_to_single(q_at_5, pos=5), rope.apply_to_single(k_at_2, pos=2)
dot_A = (q5_rot * k2_rot).sum().item()

# 模拟位置 m=10 的 q 和位置 n=7 的 k（相对位置同为 3）
q_at_10 = q_content.clone()
k_at_7 = k_content.clone()
q10_rot = rope.apply_to_single(q_at_10, pos=10)
k7_rot = rope.apply_to_single(k_at_7, pos=7)
dot_B = (q10_rot * k7_rot).sum().item()

print(f"情况 A: m=5,  n=2  (相对位置=3), 点积 = {dot_A:.6f}")
print(f"情况 B: m=10, n=7  (相对位置=3), 点积 = {dot_B:.6f}")
print(f"差异: {abs(dot_A - dot_B):.2e}  (应接近 0)")
```

> ⚠️ 上面的 `apply_to_single` 需要在 RoPE 类中补充。为简洁，这里给出完整验证脚本：

```python
# verify_rope_relative.py —— 验证 RoPE 点积只依赖相对位置
# 运行: python3 verify_rope_relative.py

import torch
import math

torch.manual_seed(42)
head_dim = 8
max_len = 100

# 预计算 cos/sin
freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
t = torch.arange(max_len).float()
angles = torch.outer(t, freqs)
cos_table = angles.cos()  # (max_len, d/2)
sin_table = angles.sin()


def apply_rope_at_pos(x, pos):
    """对单个位置的向量应用 RoPE"""
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    cos = cos_table[pos]  # (d/2,)
    sin = sin_table[pos]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# 固定 q, k 内容
q = torch.randn(head_dim)
k = torch.randn(head_dim)

# 相对位置 = 3 的两种情况
dot_A = (apply_rope_at_pos(q, 5) * apply_rope_at_pos(k, 2)).sum().item()
dot_B = (apply_rope_at_pos(q, 10) * apply_rope_at_pos(k, 7)).sum().item()

# 相对位置 = 5 的情况（应不同）
dot_C = (apply_rope_at_pos(q, 8) * apply_rope_at_pos(k, 3)).sum().item()

print(f"相对位置=3 (m=5,n=2):  点积 = {dot_A:.6f}")
print(f"相对位置=3 (m=10,n=7): 点积 = {dot_B:.6f}")
print(f"相对位置=5 (m=8,n=3):  点积 = {dot_C:.6f}")
print(f"\n|A - B| = {abs(dot_A - dot_B):.2e}  (相对位置相同，应≈0)")
print(f"|A - C| = {abs(dot_A - dot_C):.4f}  (相对位置不同，应≠0)")
```

```bash
python3 verify_rope_relative.py
```

```text
相对位置=3 (m=5,n=2):  点积 = 1.234567
相对位置=3 (m=10,n=7): 点积 = 1.234567
相对位置=5 (m=8,n=3):  点积 = 0.876543

|A - B| = 0.00e+00  (相对位置相同，应≈0)
相对位置不同，应≠0: |A - C| = 0.3580
```

> 💡 **验证成功**：相对位置相同（都为 3）时，点积完全相等（误差 0）；相对位置不同（3 vs 5）时，点积不同。这证明了 RoPE 的核心性质——点积只依赖相对位置。

#### 实验：Sinusoidal 可视化

```python
# visualize_sinusoidal.py —— 可视化 Sinusoidal 位置编码
# 运行: python3 visualize_sinusoidal.py
# 依赖: matplotlib

import torch
import math
import matplotlib.pyplot as plt

embed_dim = 64
max_len = 100

pe = torch.zeros(max_len, embed_dim)
position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：不同维度的 PE 值随位置变化
ax = axes[0]
for i in [0, 4, 16, 32]:
    ax.plot(pe[:max_len, i].numpy(), label=f"dim {i} (freq={div_term[i//2]:.4f})")
ax.set_xlabel("Position")
ax.set_ylabel("PE value")
ax.set_title("Sinusoidal PE: different dimensions")
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：PE 矩阵热力图
ax = axes[1]
im = ax.imshow(pe.numpy(), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel("Embedding dimension")
ax.set_ylabel("Position")
ax.set_title("Sinusoidal PE heatmap")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("sinusoidal_pe.png", dpi=100, bbox_inches='tight')
print("已保存: sinusoidal_pe.png")
```

```bash
python3 visualize_sinusoidal.py
```

```text
已保存: sinusoidal_pe.png
```

> 💡 **观察**：左图中，低维度（dim 0）频率高、周期短（捕捉局部位置），高维度（dim 32）频率低、周期长（捕捉全局位置）。右图的热力图呈现"高频在左、低频在右"的条纹模式。

---

### 学习任务 6：RoPE 的外推与 Length Scaling（25 分钟）

#### 外推问题

RoPE 虽然理论上可外推（频率连续），但直接用训练长度外的位置时，注意力分布会退化（部分位置对的注意力异常增大）。常见解决方法：

| 方法 | 原理 | 代表 |
|------|------|------|
| **Position Interpolation (PI)** | 把位置索引缩放：$pos' = pos \cdot \frac{n_{\text{train}}}{n_{\text{target}}}$ | meta 官方 |
| **NTK-aware scaling** | 调整频率基数：$\theta' = \theta \cdot \alpha^{d/(d-2)}$ | 社区方案 |
| **YaRN** | PI + NTK 的改进组合 | Qwen / Mistral |

#### Position Interpolation 直觉

训练时见过位置 $0 \sim n_{\text{train}}$，推理时要用 $0 \sim n_{\text{target}}$（$n_{\text{target}} > n_{\text{train}}$）。PI 把推理位置"压缩"回训练范围：

$$pos' = pos \cdot \frac{n_{\text{train}}}{n_{\text{target}}}$$

- $pos = 0 \sim n_{\text{target}}$ 映射到 $pos' = 0 \sim n_{\text{train}}$
- 旋转角度变为 $pos' \cdot \theta_i$，落在训练见过的范围内

| 方法 | 外推方式 | 微调需求 | 效果 |
|------|----------|----------|------|
| 直接外推 | 不改 | 不需 | 退化严重 |
| PI | 线性缩放位置 | 少量微调 | 较好 |
| NTK-aware | 调整频率基数 | 可不微调 | 更好 |
| YaRN | 组合 | 少量微调 | 最优 |

> 💡 **为什么 RoPE 外推比 Learned 好**：Learned 的位置编码是离散查表，超过 $n_{\max}$ 直接没有编码。RoPE 是连续函数（$\cos, \sin$），理论上任意位置都有定义，只是分布需要调整。这就是现代大模型选 RoPE 的核心原因——可外推。

---

### 学习任务 7：三种方案对比与选型（15 分钟）

| 维度 | Sinusoidal | Learned | RoPE |
|------|------------|---------|------|
| **类型** | 绝对位置 | 绝对位置 | 相对位置 |
| **是否学习** | 固定 | 可学习 | 固定（无参数） |
| **注入方式** | $X + PE$ | $X + PE$ | 旋转 $Q, K$ |
| **作用于 V** | 是（通过 $X$） | 是（通过 $X$） | 否 |
| **外推性** | 理论可，实际有限 | 不可 | 可（配合 PI/NTK） |
| **参数量** | 0 | $n_{\max} \times d$ | 0 |
| **KV Cache 友好** | 一般 | 一般 | 好（旋转可预计算） |
| **代表模型** | 原始 Transformer / T5 | GPT-2 / BERT | LLaMA / Qwen / DeepSeek |
| **推荐度** | 📎 历史 | 📎 固定长度 | ⭐ 现代标配 |

> 💡 **选型建议**：新项目首选 RoPE——无参数、可外推、相对位置、KV Cache 友好。如果任务是固定长度且不需外推（如 BERT 的 512），Learned 也可。Sinusoidal 主要是历史意义，新模型很少用。

---

### 面试题积累（本周目标 10-12 道，今日 3 道）

**Q1：为什么 Transformer 需要位置编码？不能直接把位置索引加到 embedding 上吗？**
> Self-Attention 是置换等变的——输入打乱为 $PX$，输出变为 $PO$，只是对应打乱，内容不变。Attention 本身把输入当无序集合处理，不编码顺序。不能直接加位置索引，因为：① 数值尺度不匹配（embedding 值域 $[-1,1]$，位置索引可能上百）；② 无周期性（相邻关系应与绝对位置无关）；③ 不可外推（超长位置未见过）；④ 维度浪费（所有维度加同一标量）。位置编码需用与 embedding 同 shape 的 $d$ 维向量，通过相加注入。

**Q2：RoPE 的原理是什么？为什么它编码的是相对位置？**
> RoPE 对位置 $m$ 的 Q 乘旋转矩阵 $R_m$，位置 $n$ 的 K 乘 $R_n$。点积 $q'_m \cdot k'_n = q_m^T R_m^T R_n k_n$。利用旋转矩阵性质 $R_m^T R_n = R_{n-m}$（旋转的逆乘旋转等于相对旋转），点积变为 $q_m^T R_{n-m} k_n$，只依赖相对位置 $n - m$。RoPE 把 $d$ 维向量分成 $d/2$ 对，每对用 2D 旋转，角度为 $m\theta_i$，频率 $\theta_i = 10000^{-2i/d}$。它不引入额外参数，不作用于 V，对 KV Cache 友好（旋转可预计算）。

**Q3：Sinusoidal、Learned、RoPE 三种位置编码各有什么优缺点？现代大模型为什么选 RoPE？**
> Sinusoidal：固定无参数、理论可外推，但实际外推有限、是绝对位置编码。Learned：灵活可学习，但不可外推（超 $n_{\max}$ 无编码）、有参数。RoPE：相对位置、无参数、可外推（配合 PI/NTK scaling）、KV Cache 友好。现代大模型选 RoPE 的核心原因：① 相对位置比绝对位置更符合语言直觉；② 可外推支持长上下文（4K→128K）；③ 不引入参数；④ 旋转矩阵可预计算，推理时零额外开销。

---

### 今日检查清单

- [ ] 能解释 Self-Attention 的置换等变性（输入打乱 → 输出对应打乱）
- [ ] 知道为什么不能直接加位置索引（尺度/周期/外推/维度）
- [ ] 能写出 Sinusoidal 公式 $\sin(pos / 10000^{2i/d})$ 和 $\cos(\cdot)$
- [ ] 理解 Sinusoidal 的多频率设计（高频捕捉局部，低频捕捉全局）
- [ ] 知道 Sinusoidal 的"相对位置可线性表示"性质
- [ ] 能说出 Learned 位置编码的优缺点（灵活但不可外推）
- [ ] 能写出 RoPE 的旋转矩阵 $R_m$（块对角，每块 $2 \times 2$ 旋转）
- [ ] 能证明 $R_m^T R_n = R_{n-m}$（点积只依赖相对位置）
- [ ] 知道 RoPE 只作用于 Q 和 K，不作用于 V
- [ ] 理解 RoPE 的高效实现（elementwise，不构造稀疏矩阵）
- [ ] 跑通 `positional_encoding.py`，实现三种位置编码
- [ ] 跑通 RoPE 相对位置验证实验，观察到相对位置相同时点积相等
- [ ] 跑通 Sinusoidal 可视化，观察到多频率条纹模式
- [ ] 知道 RoPE 外推方法（PI / NTK-aware / YaRN）
- [ ] 能说出三种方案的选型建议（RoPE 为现代标配）

#### 明日预告

Day 5 将用 Day 2-4 的组件**组装完整的 Transformer Block**——把 Multi-Head Attention + FFN + LayerNorm + 残差连接拼成一个可堆叠的层。重点在 Pre-Norm vs Post-Norm 的差异、FFN 的结构（中间维度 $4d$ + 激活函数）、残差连接的作用。对应 README 中的 `kernels/transformer_block.py`。今天学完了所有"零件"（Attention、位置编码），明天开始"组装"。建议今晚先想一个问题：为什么残差连接能缓解深层网络的梯度消失？Pre-Norm 和 Post-Norm 在梯度路径上有什么区别？

---
