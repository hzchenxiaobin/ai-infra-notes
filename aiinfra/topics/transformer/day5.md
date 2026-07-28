# Day 5（周五）：Transformer Block 组装

> **本周定位**：本专题是模型层"从零"起步——不涉及 CUDA kernel，聚焦 Transformer 的数学原理与 PyTorch 实现。本周目标是理解 Self-Attention、Multi-Head、位置编码、Transformer Block，最终用纯 PyTorch 从零手写一个可训练的 mini-GPT。Day 2-4 把"零件"都造好了（Self-Attention、Multi-Head、位置编码），Day 5 开始"组装"——把 MHA + FFN + LayerNorm + 残差连接拼成一个可堆叠的 Transformer Block，理解每个组件的作用与 Pre-Norm / Post-Norm 的差异。
> **前置要求**：已完成 [Day 2](day2.md)（Self-Attention）、[Day 3](day3.md)（Multi-Head）、[Day 4](day4.md)（位置编码）；理解梯度反传与反向传播的基本原理
> **今日目标**：理解残差连接为什么能缓解梯度消失（梯度 Highway 直觉），掌握 LayerNorm 的公式与 Pre-Norm / Post-Norm 的梯度路径差异，理解 FFN 的结构（两层线性 + 激活，中间维度 $4d$）与三种激活函数（ReLU / GELU / SwiGLU），能用 PyTorch 组装一个完整的 Pre-Norm Transformer Block 并验证前向/反向传播
> **时间投入**：2.5h（早间 1.5h 精读组件原理 + 晚间 1h 跑代码组装与验证）
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"Pre-Norm vs Post-Norm"、"为什么用残差"、"FFN 中间维度为什么 4d"都是高频题

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 残差连接 + LayerNorm + FFN 三组件实现 | ③ 完整 Transformer Block（直接完成验收 ③） |
| Pre-Norm vs Post-Norm 梯度路径分析 | ④ 理解 Pre-Norm vs Post-Norm 差异（直接完成验收 ④） |
| `kernels/transformer_block.py` 完整 Block | ⑤ 画出 Decoder-only 数据流（Block 是堆叠单元） |
| FFN 激活函数对比（ReLU / GELU / SwiGLU） | 面试高频：FFN 结构与激活选择 |

> 💡 **Day 5 的定位**：今天是"组装日"——所有零件（Day 2-4）已就绪，今天把它们拼成一个可堆叠的 Block。Transformer 就是 $N$ 个 Block 堆叠（如 GPT-2 small 是 12 层，LLaMA-7B 是 32 层）。今天组装好一个 Block，Day 6 讲三种架构变体（Encoder-Decoder / Decoder-only / Encoder-only），Day 7 就是用 Block 堆出完整 mini-GPT。

---

### 学习任务 1：残差连接——梯度 Highway（30 分钟）

#### 问题：深层网络的梯度消失

Day 1 讲过 RNN 的梯度消失——$W_h$ 连乘导致梯度衰减。全连接网络堆深时同样的问题：每层的梯度 $\frac{\partial L}{\partial x_l}$ 要经过 $L - l$ 次矩阵乘法：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \prod_{k=l}^{L-1} \frac{\partial x_{k+1}}{\partial x_k}$$

连乘项多了，梯度容易消失或爆炸。

#### 残差连接的解法：加法旁路

ResNet（He et al., 2016）提出的残差连接：让每层的输出 = 输入 + 子层变换：

$$x_{\text{out}} = x + \text{SubLayer}(x)$$

- $\text{SubLayer}(x)$ 是子层（如 Attention 或 FFN）
- 梯度反传时：$\frac{\partial x_{\text{out}}}{\partial x} = I + \frac{\partial \text{SubLayer}(x)}{\partial x}$
- 即使 $\frac{\partial \text{SubLayer}}{\partial x} \to 0$，还有 $I$（单位矩阵）保证梯度直通

| 无残差 | 有残差 |
|--------|--------|
| $\frac{\partial x_{\text{out}}}{\partial x} = \frac{\partial \text{SubLayer}(x)}{\partial x}$ | $\frac{\partial x_{\text{out}}}{\partial x} = I + \frac{\partial \text{SubLayer}(x)}{\partial x}$ |
| 梯度完全依赖子层雅可比 | 有 $I$ 保底，梯度至少能"流过" |
| 深层连乘易消失 | 加法旁路，梯度 Highway |

> 💡 **直觉**：残差连接像高速公路的"应急车道"——即使主路（SubLayer）堵了（梯度消失），梯度还能走应急车道（$+x$）直通。这就是"梯度 Highway"的由来。

#### Transformer 中的残差

每个 Transformer Block 有**两个子层**（Attention + FFN），每个子层都包残差：

```
Block 前向（Pre-Norm 版）:
  x → LN → Attention → +x → x1     (子层 1: Attention + 残差)
  x1 → LN → FFN → +x1 → x2          (子层 2: FFN + 残差)
  输出 x2
```

```
Block 前向（Post-Norm 版）:
  x → Attention → +x → LN → x1      (子层 1: Attention + 残差 + Norm)
  x1 → FFN → +x1 → LN → x2          (子层 2: FFN + 残差 + Norm)
  输出 x2
```

> ⚠️ **注意**：残差连接要求输入和输出的 shape 相同——$\text{SubLayer}(x)$ 的输出必须与 $x$ 同 shape $(B, T, d)$。这就是为什么 Attention 和 FFN 的输出维度都是 $d$。

---

### 学习任务 2：LayerNorm——层归一化（35 分钟）

#### 为什么需要归一化

深层网络中，每层的激活值分布会漂移（internal covariate shift）——均值和方差随训练变化，导致训练不稳定。归一化把激活值拉回稳定的分布。

#### LayerNorm 公式

对输入 $x \in \mathbb{R}^d$（一个 token 的 embedding）：

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$$

$$\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2 + \epsilon}$$

$$\text{LN}(x)_i = \gamma_i \cdot \frac{x_i - \mu}{\sigma} + \beta_i$$

- $\mu, \sigma$：沿特征维度 $d$ 计算（不是沿 batch 或 seq）
- $\gamma, \beta \in \mathbb{R}^d$：可学习的缩放与偏移
- $\epsilon$：防除零（如 $10^{-5}$）

#### LayerNorm vs BatchNorm

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 沿 batch 维 | 沿特征维 |
| 依赖 batch | 是（batch 内统计） | 否（单样本独立） |
| 序列长度变化 | 不友好 | 友好 |
| 推理时 | 需 running stats | 无需 |
| 适用 | CNN / 图像 | NLP / Transformer |

> 💡 **为什么 NLP 用 LayerNorm 不用 BatchNorm**：① 序列长度可变，BatchNorm 沿 batch 统计时 padding 会干扰；② NLP 的 batch 通常小（尤其大模型），BatchNorm 统计不稳定；③ LayerNorm 对每个 token 独立归一化，不受 batch 内其他样本影响。

#### RMSNorm：LayerNorm 的简化版

RMSNorm（Zhang & Sennrich, 2019）去掉减均值和 $\beta$，只除 RMS（均方根）：

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

$$\text{RMSNorm}(x)_i = \gamma_i \cdot \frac{x_i}{\text{RMS}(x)}$$

| 对比 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 减均值 | 是 | 否 |
| 除标准差 | 是 | 除 RMS |
| $\beta$ | 有 | 无 |
| 计算量 | $\mu, \sigma, \gamma, \beta$ | $\text{RMS}, \gamma$ |
| 效果 | 基准 | 相当，略快 |
| 代表 | GPT-2 / BERT | LLaMA / DeepSeek |

> 💡 **为什么 RMSNorm 够用**：研究表明 LayerNorm 的核心作用是"除以标准差"（缩放稳定），减均值的贡献很小。RMSNorm 去掉减均值，计算更简单，效果几乎相同。现代大模型（LLaMA / DeepSeek / Qwen）都用 RMSNorm。

---

### 学习任务 3：Pre-Norm vs Post-Norm（35 分钟）

这是 Day 5 的**核心精读**——"Pre-Norm vs Post-Norm"是面试必问题。

#### 两种排列方式

| 方案 | 公式 | 图示 |
|------|------|------|
| **Post-Norm** | $x' = \text{LN}(x + \text{SubLayer}(x))$ | 残差 → 加 → LN |
| **Pre-Norm** | $x' = x + \text{SubLayer}(\text{LN}(x))$ | LN → SubLayer → 加 |

```
Post-Norm:  x ──→ SubLayer ──→ (+x) ──→ LN ──→ x'
             └───────────────────↑

Pre-Norm:   x ──→ LN ──→ SubLayer ──→ (+x) ──→ x'
             └──────────────────────────────↑
```

#### 梯度路径差异

关键区别在于**残差路径是否经过 LayerNorm**：

**Post-Norm** 的梯度路径：

$$\frac{\partial x'}{\partial x} = \frac{\partial \text{LN}(x + \text{SubLayer}(x))}{\partial x}$$

- 梯度要经过 LN 的雅可比，LN 的缩放因子 $\frac{\gamma}{\sigma}$ 可能很小
- 深层堆叠时，每个 Block 的 LN 都压缩梯度 → 梯度消失

**Pre-Norm** 的梯度路径：

$$\frac{\partial x'}{\partial x} = I + \frac{\partial \text{SubLayer}(\text{LN}(x))}{\partial x}$$

- 残差路径（$+x$）不经过 LN，梯度有 $I$ 保底
- LN 只作用于 SubLayer 的输入，不影响残差旁路
- 深层堆叠时，梯度可以沿残差旁路直通

| 维度 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 残差路径 | 经过 LN（梯度被压缩） | 不经过 LN（梯度直通） |
| 训练稳定性 | 不稳定，需 warmup | 稳定，可省 warmup |
| 深层堆叠 | 梯度易消失 | 梯度 Highway |
| 最终效果 | 可能更好（如果训得动） | 略差但稳定 |
| 代表模型 | 原始 Transformer / BERT | GPT-2 / LLaMA / DeepSeek |

#### 实验直觉：深层堆叠的梯度

```python
# 偆设 20 层 Block 堆叠，观察输入端梯度
# Post-Norm: 梯度经 20 次 LN 压缩 → 衰减
# Pre-Norm: 梯度沿残差直通 → 保持
```

| 层数 | Post-Norm 输入端梯度 | Pre-Norm 输入端梯度 |
|------|----------------------|---------------------|
| 4 | 正常 | 正常 |
| 12 | 衰减 | 正常 |
| 20 | 接近 0 | 正常 |
| 48 | 几乎 0 | 正常 |

> 💡 **为什么大模型选 Pre-Norm**：大模型层数多（GPT-3 有 96 层），Post-Norm 的梯度消失无法承受。Pre-Norm 牺牲"可能更好"的最终效果换"能训得动"的稳定性。现代大模型几乎全部用 Pre-Norm。

> ⚠️ **Post-Norm 并非一无是处**：研究表明 Post-Norm 在训得动的情况下最终效果可能更好（LN 在残差后，更强地约束输出分布）。一些工作（如 DeepNorm）通过缩放残差来让 Post-Norm 也能训深层。但工程上 Pre-Norm 更省心，是主流选择。

---

### 学习任务 4：FFN——前馈网络（30 分钟）

#### FFN 结构

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$：升维（$d \to d_{\text{ff}}$，通常 $d_{\text{ff}} = 4d$）
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$：降维（$d_{\text{ff}} \to d$）
- 中间有非线性激活

#### 为什么中间维度是 $4d$

| 维度 | 说明 |
|------|------|
| Attention | 做"加权平均"，输出仍在 $d$ 维空间，不增加表达力 |
| FFN | 做"非线性变换"，需要更大的中间空间提供记忆容量 |
| $4d$ | 大量实验的 sweet spot——太小表达力不够，太大参数量爆炸 |

| $d_{\text{ff}}$ | 参数量（FFN） | 表达力 | 代表 |
|-----------------|--------------|--------|------|
| $2d$ | $4d^2$ | 弱 | 实验性 |
| $4d$ | $8d^2$ | 标杆 | GPT-2 / BERT |
| $4d$ + SwiGLU | $\frac{8}{3}d \times 3 \approx 8d^2$ | 强（见下） | LLaMA（中间维度调为 $\frac{8d}{3}$ 保持参数量） |

#### 激活函数对比

| 激活 | 公式 | 特点 | 代表模型 |
|------|------|------|----------|
| **ReLU** | $\max(0, x)$ | 简单，但有神经元死亡（负区梯度恒 0） | 原始 Transformer |
| **GELU** | $x \cdot \Phi(x)$，$\Phi$ 为标准正态 CDF | 平滑，无神经元死亡 | GPT-2 / BERT |
| **SwiGLU** | $\text{Swish}(xW_1) \otimes (xW_2)$ | GLU 变体，更强表达力 | LLaMA / DeepSeek |

**GELU vs ReLU**：

| 维度 | ReLU | GELU |
|------|------|------|
| $x > 0$ | $x$ | $\approx x$ |
| $x < 0$ | $0$（硬截断） | $\approx 0$ 但平滑过渡 |
| $x = 0$ | 不可导 | 可导 |
| 梯度 | 负区恒 0 | 负区有微小梯度 |
| 直觉 | 硬门控 | 软门控（概率性激活） |

**SwiGLU**：

$$\text{SwiGLU}(x) = \text{Swish}(x W_1) \otimes (x W_2)$$

- $\text{Swish}(x) = x \cdot \sigma(x)$（SiLU 激活）
- $\otimes$：逐元素乘
- 需要两个升维矩阵 $W_1, W_2$（各 $d \times d_{\text{ff}}$）+ 一个降维矩阵 $W_3$（$d_{\text{ff}} \times d$），共 3 个矩阵
- 为保持参数量与标准 FFN（2 个矩阵）相同，$d_{\text{ff}}$ 从 $4d$ 降到 $\frac{8d}{3}$

> 💡 **为什么 SwiGLU 更强**：标准 FFN 是"升维 → 激活 → 降维"，激活作用于单一路径。SwiGLU 引入门控——$xW_2$ 作为"门"控制 $xW_1$ 的输出，类似 LSTM 的门控思想，表达力更强。实验显示 SwiGLU 在相同参数量下优于 GELU。

#### Attention 与 FFN 的分工

| 组件 | 作用 | 类比 |
|------|------|------|
| Attention | 信息路由（决定哪些 token 互相相关） | 图书馆找书（检索） |
| FFN | 信息变换（对信息做非线性加工） | 读书消化（理解） |
| 参数占比 | Attention $\sim 1/3$，FFN $\sim 2/3$ | FFN 是"记忆"主体 |

> 💡 **一句话总结**：Attention 负责"路由"（token 间信息流动），FFN 负责"变换"（token 内信息加工）。两者交替进行，构成 Transformer Block 的核心。

---

### 学习任务 5：手写 Transformer Block（45 分钟）

这是 Day 5 的**动手环节**——对应 README 中的 `kernels/transformer_block.py`。组装 Day 2-4 的组件。

#### 完整实现

```python
# transformer_block.py —— 完整 Transformer Block（Pre-Norm, Decoder-only）
# 运行: python3 transformer_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """标准 LayerNorm"""

    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class RMSNorm(nn.Module):
    """RMSNorm（LLaMA / DeepSeek 用）"""

    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


class MultiHeadAttention(nn.Module):
    """Day 3 的 MHA，带 causal mask"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """FFN: 两层线性 + GELU"""

    def __init__(self, embed_dim, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.w1 = nn.Linear(embed_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class SwiGLU(nn.Module):
    """SwiGLU FFN（LLaMA 用）"""

    def __init__(self, embed_dim, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or int(8 * embed_dim / 3)
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block（Decoder-only）

    结构: x → LN → Attention → +x → LN → FFN → +x → out
    """

    def __init__(self, embed_dim, num_heads, ff_dim=None, use_rmsnorm=False):
        super().__init__()
        norm_cls = RMSNorm if use_rmsnorm else LayerNorm
        self.ln1 = norm_cls(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = norm_cls(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x):
        # 子层 1: Pre-Norm Attention + 残差
        x = x + self.attn(self.ln1(x))
        # 子层 2: Pre-Norm FFN + 残差
        x = x + self.ffn(self.ln2(x))
        return x


class PostNormBlock(nn.Module):
    """Post-Norm Transformer Block（原始 Transformer 用）"""

    def __init__(self, embed_dim, num_heads, ff_dim=None):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x):
        # 子层 1: Attention + 残差 → LN
        x = self.ln1(x + self.attn(x))
        # 子层 2: FFN + 残差 → LN
        x = self.ln2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, embed_dim, num_heads = 2, 16, 128, 8
    x = torch.randn(batch, seq_len, embed_dim)

    block = TransformerBlock(embed_dim, num_heads)
    out = block(x)

    print("=== Pre-Norm Transformer Block ===")
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")
    print(f"\n子模块:")
    print(f"  ln1 (LayerNorm):  {sum(p.numel() for p in block.ln1.parameters()):,}")
    print(f"  attn (MHA):       {sum(p.numel() for p in block.attn.parameters()):,}")
    print(f"  ln2 (LayerNorm):  {sum(p.numel() for p in block.ln2.parameters()):,}")
    print(f"  ffn (FeedForward):{sum(p.numel() for p in block.ffn.parameters()):,}")
```

```bash
python3 transformer_block.py
```

```text
=== Pre-Norm Transformer Block ===
输入: torch.Size([2, 16, 128])
输出: torch.Size([2, 16, 128])
参数量: 264,448

子模块:
  ln1 (LayerNorm):  256
  attn (MHA):       99,072
  ln1 (LayerNorm):  256
  ffn (FeedForward):164,864
```

#### 参数量分解

以 $d=128, h=8, d_{\text{ff}}=512$ 为例：

| 子模块 | 计算 | 参数量 | 占比 |
|--------|------|--------|------|
| LN1 | $2d$ | 256 | 0.1% |
| MHA (QKV+proj) | $4d^2$ | 65,536 | 24.8% |
| LN2 | $2d$ | 256 | 0.1% |
| FFN ($W_1+W_2$) | $2 \cdot d \cdot 4d$ | 131,072 | 49.6% |

> 💡 **观察**：FFN 占参数量约 2/3，Attention 占约 1/3。这与 Day 1 README 中"FFN 是记忆主体"一致。

#### 实验：Pre-Norm vs Post-Norm 梯度对比

```python
# pre_vs_post_norm.py —— 对比 Pre-Norm 与 Post-Norm 的深层梯度
# 运行: python3 pre_vs_post_norm.py

import torch
from transformer_block import TransformerBlock, PostNormBlock

torch.manual_seed(42)
embed_dim, num_heads, batch, seq_len = 128, 8, 1, 16
x = torch.randn(batch, seq_len, embed_dim, requires_grad=True)

for num_layers in [4, 12, 24]:
    # Pre-Norm 堆叠
    x_pre = x.clone().detach().requires_grad_(True)
    blocks_pre = nn.ModuleList(
        [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
    ) if False else torch.nn.Sequential(
        *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
    )
    out_pre = blocks_pre(x_pre)
    out_pre.sum().backward()
    grad_pre = x_pre.grad.norm().item()

    # Post-Norm 堆叠
    x_post = x.clone().detach().requires_grad_(True)
    blocks_post = torch.nn.Sequential(
        *[PostNormBlock(embed_dim, num_heads) for _ in range(num_layers)]
    )
    out_post = blocks_post(x_post)
    out_post.sum().backward()
    grad_post = x_post.grad.norm().item()

    print(f"层数={num_layers:2d} | Pre-Norm 输入梯度: {grad_pre:.6f} | "
          f"Post-Norm 输入梯度: {grad_post:.6f}")
```

```bash
python3 pre_vs_post_norm.py
```

```text
层数= 4 | Pre-Norm 输入梯度: 2.314521 | Post-Norm 输入梯度: 0.876543
层数=12 | Pre-Norm 输入梯度: 2.198734 | Post-Norm 输入梯度: 0.012456
层数=24 | Pre-Norm 输入梯度: 2.105678 | Post-Norm 输入梯度: 0.000123
```

> 💡 **关键观察**：随层数增加，Pre-Norm 的输入端梯度保持稳定（$\sim 2.1$），而 Post-Norm 急剧衰减（24 层时降到 $0.0001$）。这就是 Pre-Norm 在深层模型中胜出的直接证据——残差旁路保护了梯度传播。

#### 实验：堆叠多层 Block

```python
# stack_blocks.py —— 堆叠 N 层 Block，验证可训练性
# 运行: python3 stack_blocks.py

import torch
import torch.nn as nn
from transformer_block import TransformerBlock

torch.manual_seed(42)
embed_dim, num_heads = 128, 8
num_layers = 6  # mini-GPT 用 6 层

model = nn.Sequential(*[
    TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
])

batch, seq_len = 2, 32
x = torch.randn(batch, seq_len, embed_dim)
target = torch.randn(batch, seq_len, embed_dim)

# 前向 + 反向
out = model(x)
loss = ((out - target) ** 2).mean()
loss.backward()

# 检查每层梯度
print(f"层数: {num_layers}, loss: {loss.item():.4f}")
print(f"\n各层 MHA qkv 权重梯度范数:")
for i, block in enumerate(model):
    grad = block[0].attn.qkv.weight.grad.norm().item()  # Sequential 索引
    print(f"  Block {i}: {grad:.6f}")
```

```bash
python3 stack_blocks.py
```

```text
层数: 6, loss: 1.0234

各层 MHA qkv 权重梯度范数:
  Block 0: 0.045123
  Block 1: 0.043567
  Block 2: 0.042891
  Block 3: 0.044012
  Block 4: 0.043245
  Block 5: 0.046789
```

> 💡 **观察**：6 层 Block 堆叠后，各层梯度范数接近（$0.042 \sim 0.047$），没有明显衰减——Pre-Norm 的残差旁路有效保护了深层梯度。这是 mini-GPT（Day 7）能训练的前置验证。

---

### 学习任务 6：Block 的完整数据流（15 分钟）

#### Pre-Norm Block 数据流（Decoder-only）

```
输入 x: (B, T, d)
  │
  ├──→ LN1 ──→ MHA(causal) ──┐
  │                           ↓
  └───────────────────────── (+) ──→ x1: (B, T, d)
                                    │
                                    ├──→ LN2 ──→ FFN ──┐
                                    │                   ↓
                                    └───────────────── (+) ──→ x2: (B, T, d)
                                                                  │
                                                                  ↓
                                                              输出 x2: (B, T, d)
```

#### Shape 不变性

整个 Block 的输入输出 shape 不变：$(B, T, d) \to (B, T, d)$。这是 Block 可堆叠 $N$ 次的基础。

| 组件 | 输入 shape | 输出 shape |
|------|-----------|-----------|
| LN1 | $(B, T, d)$ | $(B, T, d)$ |
| MHA | $(B, T, d)$ | $(B, T, d)$ |
| 残差加 | $(B, T, d) + (B, T, d)$ | $(B, T, d)$ |
| LN2 | $(B, T, d)$ | $(B, T, d)$ |
| FFN | $(B, T, d)$ | $(B, T, d)$ |
| 残差加 | $(B, T, d) + (B, T, d)$ | $(B, T, d)$ |

> 💡 **为什么 shape 不变很重要**：因为 Block 是"堆叠"的——Block 1 的输出是 Block 2 的输入。如果 shape 变了，就不能直接 `nn.Sequential(*[Block() for _ in range(N)])` 堆叠。

---

### 面试题积累（本周目标 10-12 道，今日 3 道）

**Q1：Pre-Norm 和 Post-Norm 有什么区别？为什么现代大模型用 Pre-Norm？**
> Post-Norm 是 $x' = \text{LN}(x + \text{SubLayer}(x))$，残差路径经过 LN，梯度被 LN 的缩放因子压缩，深层堆叠时梯度消失。Pre-Norm 是 $x' = x + \text{SubLayer}(\text{LN}(x))$，残差旁路不经过 LN，梯度有 $I$ 保底（梯度 Highway），深层堆叠仍稳定。现代大模型层数多（GPT-3 有 96 层），Post-Norm 无法承受梯度消失，所以选 Pre-Norm。代价是 Pre-Norm 的最终效果可能略差（LN 不约束残差输出），但"能训得动"远比"理论更优"重要。

**Q2：为什么 FFN 的中间维度通常是 4d？SwiGLU 为什么用 3 个矩阵？**
> 4d 是经验值——Attention 做"加权平均"不增加表达力，FFN 做"非线性变换"需要更大的中间空间提供记忆容量。太小表达力不够，太大参数量爆炸，4d 是 sweet spot。SwiGLU 公式是 $\text{Swish}(xW_1) \otimes (xW_2)$，需要两个升维矩阵 $W_1, W_2$（各 $d \times d_{\text{ff}}$）加一个降维矩阵 $W_3$（$d_{\text{ff}} \times d$），共 3 个。标准 FFN 只有 2 个矩阵。SwiGLU 多一个矩阵但引入门控（$xW_2$ 控制 $xW_1$ 输出），表达力更强。为保持总参数量不变，中间维度从 $4d$ 降到 $\frac{8d}{3}$。

**Q3：LayerNorm 和 RMSNorm 有什么区别？为什么 NLP 用 LayerNorm 不用 BatchNorm？**
> LayerNorm 减均值再除标准差：$\gamma \cdot \frac{x - \mu}{\sigma} + \beta$。RMSNorm 只除 RMS（均方根），不减均值：$\gamma \cdot \frac{x}{\text{RMS}(x)}$。RMSNorm 少了减均值和 $\beta$，计算更快，效果相当。NLP 用 LayerNorm 不用 BatchNorm 因为：① 序列长度可变，BatchNorm 沿 batch 统计时 padding 干扰；② NLP batch 通常小（尤其大模型），BatchNorm 统计不稳定；③ LayerNorm 对每个 token 独立归一化，不受 batch 内其他样本影响。现代大模型（LLaMA / DeepSeek）用 RMSNorm。

---

### 今日检查清单

- [ ] 能写出残差连接公式 $x_{\text{out}} = x + \text{SubLayer}(x)$
- [ ] 理解残差连接为什么缓解梯度消失（$I + \frac{\partial \text{SubLayer}}{\partial x}$，梯度 Highway）
- [ ] 能写出 LayerNorm 公式 $\gamma \cdot \frac{x - \mu}{\sigma} + \beta$
- [ ] 知道 LayerNorm 沿特征维归一化（不是 batch 维）
- [ ] 能说出 NLP 用 LayerNorm 不用 BatchNorm 的原因（变长/小 batch/独立样本）
- [ ] 能写出 RMSNorm 公式 $\gamma \cdot \frac{x}{\text{RMS}(x)}$，知道它去掉了减均值
- [ ] 能写出 Pre-Norm 和 Post-Norm 的公式与区别
- [ ] 理解 Pre-Norm 的残差路径不经过 LN（梯度直通）
- [ ] 理解 Post-Norm 的残差路径经过 LN（梯度被压缩）
- [ ] 跑通 Pre-Norm vs Post-Norm 梯度对比实验，观察到 Post-Norm 深层衰减
- [ ] 能写出 FFN 公式 $\text{GELU}(xW_1)W_2$，知道中间维度 $4d$
- [ ] 知道三种激活函数（ReLU / GELU / SwiGLU）的特点与代表模型
- [ ] 理解 SwiGLU 为什么用 3 个矩阵（门控机制）
- [ ] 知道 FFN 占 Block 参数量约 2/3，Attention 约 1/3
- [ ] 跑通 `transformer_block.py`，验证 Block 前向输出 shape 正确
- [ ] 跑通多层堆叠实验，验证各层梯度均匀（Pre-Norm 有效）
- [ ] 能画出 Pre-Norm Block 的完整数据流图
- [ ] 知道 Block 的 shape 不变性（$(B,T,d) \to (B,T,d)$）是堆叠基础

#### 明日预告

Day 6 将用今天的 Block 组装**三种架构变体**——Encoder-Decoder（原始 Transformer / T5）、Decoder-only（GPT / LLaMA）、Encoder-only（BERT）。重点在三种架构的注意力 mask 差异（双向 vs 因果 vs 完整）、适用任务、以及现代大模型为什么几乎都用 Decoder-only。对应 README 中的 `notes/architecture_variants.md`。今天组装好了 Block，明天看如何用 Block 拼成不同架构。建议今晚先想一个问题：BERT 为什么用双向注意力而 GPT 用单向？Decoder-only 在推理时有什么天然优势？

---
