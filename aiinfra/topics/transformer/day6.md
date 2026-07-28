# Day 6（周六）：完整架构与变体

> **本周定位**：本专题是模型层"从零"起步——不涉及 CUDA kernel，聚焦 Transformer 的数学原理与 PyTorch 实现。本周目标是理解 Self-Attention、Multi-Head、位置编码、Transformer Block，最终用纯 PyTorch 从零手写一个可训练的 mini-GPT。Day 5 把"零件"组装成了一个可堆叠的 Transformer Block，Day 6 解决"怎么用 Block 拼成不同架构"——同样一个 Block，配上不同的注意力 mask 与堆叠方式，就得到 Encoder-Decoder / Decoder-only / Encoder-only 三种架构变体。今天把三种架构的 mask 差异、适用任务、推理方式彻底理清，为 Day 7 用 Block 堆出 mini-GPT 做最后铺垫。
> **前置要求**：已完成 [Day 2](day2.md)（Self-Attention）、[Day 3](day3.md)（Multi-Head）、[Day 4](day4.md)（位置编码）、[Day 5](day5.md)（Transformer Block：残差 + LayerNorm + FFN + Pre/Post-Norm）；理解因果 mask 的作用（Day 5 的 `MultiHeadAttention` 已带 causal mask）
> **今日目标**：掌握三种架构变体（Encoder-Decoder / Decoder-only / Encoder-only）的结构与 Block 堆叠方式，理解注意力 mask 是区分三种架构的关键（双向 vs 因果 vs 交叉），搞清 Cross-Attention 的 Q/K/V 来源，建立 KV Cache 的直觉（为什么只缓存 K/V 不缓存 Q、内存开销如何），能说清现代大模型为什么几乎都用 Decoder-only（scaling law + 任务通用性 + 训练效率 + 推理友好），动手实现三种 mask 与三种架构的前向并对比
> **时间投入**：5h（早间 2h 精读三种架构 + Cross-Attention + KV Cache；下午 2h 跑代码实现三种 mask 与架构对比；晚间 1h 整理 `notes/architecture_variants.md` 笔记与面试题）
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"三种架构区别与适用任务"、"为什么都用 Decoder-only"、"Cross-Attention 的 QKV 来自哪"、"KV Cache 原理"都是高频题

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 三种架构变体对比表（结构 / mask / 适用任务 / 代表模型） | ⑤ 能画出 Decoder-only 的前向数据流（扩展到能画出三种架构的数据流） |
| 三种注意力 mask 实现（Full / Causal / Cross） | ① 理解 Attention 公式在不同架构中的用法（mask 是 Attention 的关键参数） |
| Cross-Attention 的 Q/K/V 来源推导 | ① 解释每步 shape 变化（Cross-Attention 的 Q 与 K/V 维度不同源） |
| KV Cache 直觉与内存开销分析 | 面试高频：Decoder-only 推理为什么高效 |
| "为什么选 Decoder-only"论证链 | 面试高频：三种架构 tradeoff + 现代趋势 |
| `notes/architecture_variants.md` 笔记 | 本周核心产出之一（架构变体对比） |

> 💡 **Day 6 的定位**：今天是"架构日"——Day 2-5 造好了"零件"（Self-Attention、Multi-Head、位置编码、Block），今天看怎么用同一批零件拼出三种不同的"整车"。关键洞察是：**区分三种架构的不是 Block 本身，而是注意力 mask**。同一个 Pre-Norm Block，给它双向 mask 就是 Encoder（BERT），给它因果 mask 就是 Decoder（GPT），再插一个 Cross-Attention 子层就是 Encoder-Decoder 的 Decoder。今天把这套"换 mask 换架构"的逻辑想透，Day 7 堆 mini-GPT 时你会知道"为什么 Decoder-only 就是一串 causal Block"。

---

### 学习任务 1：三种架构总览（25 分钟）

#### 一张表看清三种变体

原始 Transformer（Vaswani et al., 2017）本身是 Encoder-Decoder 结构。后续工作按"只用一半"或"调整训练目标"演化出三种架构：

| 架构 | Block 堆叠 | 注意力 mask | 预训练任务 | 适用任务 | 代表模型 |
|------|-----------|-------------|-----------|----------|----------|
| **Encoder-Decoder** | Encoder（双向）+ Decoder（因果 + Cross-Attn） | Encoder 全连接 / Decoder 因果 / Cross-Attn 无 mask | Seq2Seq（如翻译） | seq2seq：翻译、摘要、对话 | 原始 Transformer / T5 / BART / Whisper |
| **Decoder-only** | 单栈因果 Block | 因果 mask（下三角） | Next-token prediction | 生成：续写、对话、CoT | GPT 系列 / LLaMA / DeepSeek / Qwen |
| **Encoder-only** | 单栈双向 Block | 全连接（无 mask） | MLM（完形填空） | 理解：分类、NER、抽取 | BERT / RoBERTa / DeBERTa |

#### 区分三种架构的关键：注意力 mask

同一个 Block，**唯一改变的是注意力矩阵哪些位置被屏蔽**：

| mask 类型 | 注意力矩阵形状 | 允许的注意力 | 用于 |
|----------|---------------|-------------|------|
| **Full（双向）** | $(n, n)$ 全部可见 | 位置 $i$ 看所有位置（含未来） | Encoder / BERT |
| **Causal（因果）** | $(n, n)$ 下三角可见 | 位置 $i$ 只看 $\leq i$ 的位置 | Decoder 自注意力 / GPT |
| **Cross（交叉）** | $(n_{\text{dec}}, n_{\text{enc}})$ | Decoder 每个位置看 Encoder 所有位置 | Encoder-Decoder 的 Cross-Attention |

```
Full mask (Encoder/BERT):       Causal mask (Decoder/GPT):    Cross mask (Enc-Dec):
  1 1 1 1  ← 位置0看全部          1 0 0 0  ← 位置0只看自己       1 1 1 1  ← dec 每位看 enc 全部
  1 1 1 1                          1 1 0 0  ← 位置1看0,1          1 1 1 1
  1 1 1 1                          1 1 1 0                       1 1 1 1
  1 1 1 1                          1 1 1 1                       1 1 1 1
  (n_enc × n_enc)                  (n_dec × n_dec)               (n_dec × n_enc)
```

> 💡 **一句话总结**：三种架构 = 同一个 Block + 不同的 mask。Encoder 给双向 mask，Decoder 给因果 mask，Encoder-Decoder 的 Decoder 还多一个 Cross-Attention 子层（Q 来自 Decoder，K/V 来自 Encoder）。理解 mask，就理解了三种架构的本质。

> ⚠️ **注意**：还有第四种常见 mask——**Padding mask**，用于屏蔽 batch 中较短序列的 padding 位置。它通常叠加在上述 mask 之上（如 Encoder 的双向 mask + padding mask）。今天先聚焦三种"结构型 mask"，padding mask 是工程细节，Day 7 mini-GPT 会用到。

---

### 学习任务 2：Encoder-Decoder——原始 Transformer（50 分钟）

#### 整体结构

原始 Transformer 由**两个栈**组成：

```
源序列 src ──→ [Embedding + PE] ──→ Encoder栈 (×N)
                                      │
                                      ↓ encoder 输出 memory: (B, n_src, d)
                                      │
目标序列 tgt ──→ [Embedding + PE] ──→ Decoder栈 (×N) ──→ Linear + Softmax ──→ 输出概率
                                      ↑
                                      └── Cross-Attention 的 K/V 来自 memory
```

每个 **Encoder Block**（Day 5 的 Block 配双向 mask，两个残差子层）：

$$h = x + \text{MHA}_{\text{full}}(\text{LN}_1(x))$$

$$x_{\text{enc}} = h + \text{FFN}(\text{LN}_2(h))$$

每个 **Decoder Block**（三个子层）：

$$h = x + \text{MHA}_{\text{causal}}(\text{LN}_1(x))$$

$$h = h + \text{CrossAttn}(\text{LN}_2(h), \text{memory})$$

$$x_{\text{dec}} = h + \text{FFN}(\text{LN}_3(h))$$

- Decoder 比 Encoder 多一个子层：**Cross-Attention**
- 原始论文配置：$N=6$ 层，$d_{\text{model}}=512$，$h=8$，$d_{\text{ff}}=2048$

#### Cross-Attention：Q/K/V 来自哪里

这是 Encoder-Decoder 与其他两种架构的**本质区别**。Self-Attention 的 Q/K/V 都来自同一输入；Cross-Attention 的 Q 来自 Decoder，K/V 来自 Encoder：

| 角色 | 来源 | Shape |
|------|------|-------|
| **Query** $Q$ | Decoder 当前层输出 $h \cdot W^Q$ | $(B, n_{\text{dec}}, d)$ |
| **Key** $K$ | Encoder 输出 memory $\cdot W^K$ | $(B, n_{\text{src}}, d)$ |
| **Value** $V$ | Encoder 输出 memory $\cdot W^V$ | $(B, n_{\text{src}}, d)$ |
| 注意力分数 $S$ | $Q K^T$ | $(B, n_{\text{dec}}, n_{\text{src}})$ |
| 输出 $O$ | $\text{softmax}(S/\sqrt{d}) V$ | $(B, n_{\text{dec}}, d)$ |

> 💡 **直觉**：Cross-Attention 让 Decoder 的每个位置"查询" Encoder 的全部输出，决定"翻译/生成当前 token 时该关注源序列的哪些位置"。比如英译中，Decoder 生成"猫"时，Cross-Attention 的注意力权重会集中在 Encoder 中"cat"的位置。这就是 Bahdanau Attention（Day 1）在 Transformer 中的化身——只不过从 RNN 上下文向量变成了全局 K/V 矩阵。

#### 训练与推理的不对称

| 阶段 | Decoder 输入 | 注意 |
|------|-------------|------|
| **训练** | 完整目标序列（teacher forcing） | 因果 mask 保证位置 $i$ 只看 $\leq i$，可一次前向算出所有位置 loss |
| **推理** | 逐 token 自回归生成 | 每步把新 token 拼到已生成序列，重新前向（或用 KV Cache） |

> ⚠️ **注意**：训练时用 teacher forcing（喂完整目标序列 + 因果 mask），可并行计算所有位置的预测——这是 Transformer 比纯 RNN 训练快的根本原因（Day 1 讲过 RNN 的串行瓶颈）。推理时无法并行（每步依赖上一步输出），但 KV Cache 能避免重复计算。

#### 代表模型：T5

T5（Text-to-Text Transfer Transformer）把所有任务统一成"文本→文本"：

| 任务 | 输入 | 输出 |
|------|------|------|
| 翻译 | `translate English to French: Hello` | `Bonjour` |
| 摘要 | `summarize: <长文>` | `<摘要>` |
| 分类 | `classify: This movie is great` | `positive` |

T5 用统一的 Encoder-Decoder 处理所有任务，证明了 seq2seq 架构的通用性。但后续 GPT-3 证明 Decoder-only + prompting 也能统一所有任务，且 scaling 更优——这是架构演进的分水岭。

---

### 学习任务 3：Decoder-only——GPT / LLaMA（50 分钟）

#### 结构：一栈因果 Block

Decoder-only 砍掉 Encoder 与 Cross-Attention，只保留**因果自注意力 Block**：

```
token 序列 ──→ [Embedding + PE] ──→ Block(因果) ×N ──→ LN ──→ Linear(d→V) ──→ logits
```

每个 Block 就是 Day 5 写的 `TransformerBlock`（带 causal mask 的 MHA + FFN + Pre-Norm + 残差）。**没有 Cross-Attention 子层**，因为输入和输出是同一个序列（自回归）。

#### 因果 mask：只看过去

$$\text{mask}_{ij} = \begin{cases} 0, & j \leq i \\ -\infty, & j > i \end{cases}$$

位置 $i$ 的 query 只能与位置 $\leq i$ 的 key 做点积。这保证了"预测第 $i+1$ 个 token 时，只用到前 $i$ 个 token 的信息"——没有信息泄漏。

#### 预训练：Next-Token Prediction

给定序列 $(x_1, x_2, \ldots, x_n)$，模型预测每个位置的下一个 token：

$$\mathcal{L} = -\sum_{t=1}^{n-1} \log P(x_{t+1} \mid x_1, \ldots, x_t)$$

- 每个位置都是训练信号（不像 BERT 只预测 15% 的 mask 位）
- 无需人工标注，互联网文本即是无限训练数据
- 因果 mask 让一次前向同时算出所有位置的 loss（训练并行）

#### KV Cache：推理高效的关键

自回归生成时，第 $t$ 步生成 token $x_t$，需要计算 $x_t$ 与 $x_1, \ldots, x_t$ 的注意力。关键观察：

| 量 | 第 $t$ 步 vs 第 $t+1$ 步 | 是否变化 |
|----|------------------------|----------|
| $K_{1..t}, V_{1..t}$（前 $t$ 个 token 的 K/V） | 第 $t+1$ 步仍需要 | **不变**（输入与权重不变） |
| $q_{t+1}$（新 token 的 query） | 每步新生成 | 变化 |

所以可以把已计算的 $K, V$ **缓存**起来：

```
第 t 步:
  q_t @ K_cache^T → scores → softmax → @ V_cache → 输出
  K_cache.append(k_t), V_cache.append(v_t)   ← 只追加，不重算

第 t+1 步:
  只算 q_{t+1}, k_{t+1}, v_{t+1}
  K_cache.append(k_{t+1}), V_cache.append(v_{t+1})
  q_{t+1} @ K_cache^T → ...                   ← 复用缓存
```

| 维度 | 无 KV Cache | 有 KV Cache |
|------|-------------|-------------|
| 第 $t$ 步计算量 | $O(t \cdot d)$（重算所有 K/V 与注意力） | $O(d)$（只算新 token 的 q/k/v） |
| 第 $t$ 步显存 | $O(t \cdot d)$ | $O(t \cdot d)$（缓存本身占显存） |
| 总生成 $n$ token | $O(n^2 \cdot d)$ | $O(n \cdot d)$ |

> 💡 **为什么不缓存 Q**：Q 只属于"当前要生成的 token"，每步都不同，缓存无意义。K/V 属于"已生成的历史 token"，一旦算出就不再变化，所以值得缓存。这是 Day 5 面试题与 README 面试题的延伸——KV Cache 的本质是"利用自回归生成时 K/V 的单调累积性"。

> ⚠️ **KV Cache 的代价是显存**：每层缓存 $2 \times n \times d$（K 和 V），$L$ 层共 $2 L n d$。以 LLaMA-7B（$L=32, d=4096$，batch=1）为例，序列长 $n=2048$ 时 KV Cache 约 $2 \times 32 \times 2048 \times 4096 \times 2\text{B} \approx 1\text{GB}$。长序列 + 大 batch 时 KV Cache 显存成为瓶颈——这是 PagedAttention（vLLM）的动机：像操作系统分页管理内存一样分块管理 KV Cache。

#### 代表模型配置

| 模型 | $L$（层数） | $d$ | $h$ | $d_{\text{ff}}$ | 参数量 |
|------|-----------|-----|-----|----------------|--------|
| GPT-2 small | 12 | 768 | 12 | 3072 | 124M |
| GPT-2 XL | 48 | 1600 | 25 | 6400 | 1.5B |
| LLaMA-7B | 32 | 4096 | 32 | 11008 | 7B |
| LLaMA-70B | 80 | 8192 | 64 | 28672 | 70B |
| DeepSeek-V3 | 61 | 7168 | 128 | — | 671B（MoE） |

> 💡 **观察**：Decoder-only 的"长大"主要靠堆层数 $L$ 与加宽 $d$。Day 5 的 Block shape 不变性（$(B,T,d) \to (B,T,d)$）正是"无限堆叠"的基础。

---

### 学习任务 4：Encoder-only——BERT（45 分钟）

#### 结构：一栈双向 Block

BERT（Bidirectional Encoder Representations from Transformers）只用 Encoder——双向注意力的 Block 栈：

```
token 序列 ──→ [Embedding + PE] ──→ Block(双向) ×N ──→ ──→ 各位置的隐藏表示
                                                       ├── [CLS] 位置 → 分类头
                                                       └── 各 token 位置 → MLM 头
```

- 与 Decoder-only 唯一的结构差异：**没有因果 mask**，每个 token 能看到所有 token（含未来）
- 这让 BERT 的表示是"双向上下文"的——更适合"理解"而非"生成"

#### 预训练：MLM（Masked Language Model）

BERT 的预训练不是 next-token prediction，而是**完形填空**：

1. 随机选 15% 的 token 位置
2. 其中 80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持原样
3. 模型预测这些位置的原 token

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{mask}} \log P(x_i \mid x_{\text{未mask}})$$

| 维度 | MLM（BERT） | Next-Token（GPT） |
|------|-------------|-------------------|
| 预测目标 | 被 mask 的原 token | 下一个 token |
| 上下文 | 双向（看前后所有 token） | 单向（只看过去） |
| 训练信号密度 | 15% 的位置 | 100% 的位置 |
| 能否自回归生成 | 否（mask 位置随机，无法顺序生成） | 是 |
| 适合任务 | 理解（分类、NER、抽取） | 生成（续写、对话） |

#### 为什么 BERT 不能直接做生成

BERT 的双向注意力让每个位置看到**未来 token**——预测 $x_i$ 时已经"偷看"了 $x_{i+1}$。这在理解任务里是优势（更多上下文），但在自回归生成里是**信息泄漏**：

- 生成时未来的 token 还不存在，无法"双向看"
- 强行用 BERT 生成需要复杂的迭代解码（如逐步 mask + 采样），效果差且不自然

> 💡 **一句话总结**：BERT 与 GPT 的根本差异不是层数或参数，而是**注意力 mask**——BERT 双向（适合理解），GPT 单向（适合生成）。mask 决定了模型能"看到什么"，进而决定了能"做什么任务"。

#### 特殊 token：[CLS] 与 [MASK]

| token | 作用 | 用法 |
|-------|------|------|
| `[CLS]` | 序列级表示 | 放在句首，其最终隐藏向量用于分类任务（情感、NLI） |
| `[SEP]` | 句子分隔 | 双句任务（如 NLI）中分隔前提与假设 |
| `[MASK]` | 预训练占位 | MLM 预训练时标记被预测位置；微调/推理时不出现 |

> ⚠️ **注意**：`[MASK]` 只在预训练出现，下游微调时输入没有 `[MASK]`——这是 BERT 的 pretrain-finetune 范式与 GPT 的 prompting 范式的另一个差异。GPT 用同一套 next-token 目标预训练与生成，无需特殊 token 适配。

---

### 学习任务 5：动手实现三种 mask 与三种架构（70 分钟）

这是 Day 6 的**动手环节**——用一个 Block 类，配上三种 mask，前向三种架构。对应 README 的 `notes/architecture_variants.md`。

#### 完整实现

```python
# architecture_variants.py —— 三种 mask + 三种架构变体的前向对比
# 运行: python3 architecture_variants.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """通用 MHA：支持 self / cross / causal 三种模式"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _split(self, x, B, T):
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, kv=None, mask=None):
        # x: (B, T, C) 作为 Query 来源
        # kv: (B, S, C) 作为 Key/Value 来源；为 None 时是 self-attention
        B, T, C = x.shape
        if kv is None:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            S = T
        else:
            # Cross-attention: Q 来自 x，K/V 来自 kv
            qkv_q = self.qkv(x)
            q = qkv_q.chunk(3, dim=-1)[0]            # 只取 Q
            S = kv.shape[1]
            # 为简化演示，K/V 也走同一投影（实际实现会分开投影）
            _, k, v = self.qkv(kv).chunk(3, dim=-1)

        q = self._split(q, B, T)
        k = self._split(k, B, S)
        v = self._split(v, B, S)

        scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, T, S)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                     # (B, h, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), attn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.w1 = nn.Linear(embed_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    """Pre-Norm Block，可配置 self / causal / cross"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x, kv=None, mask=None):
        x = x + self.attn(self.ln1(x), kv=kv, mask=mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class EncoderOnly(nn.Module):
    """BERT 式：双向 mask 的 Block 栈"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, tokens):
        x = self.tok(tokens)
        B, T, _ = x.shape
        full_mask = torch.ones(T, T, device=x.device)       # 全 1：双向可见
        for blk in self.blocks:
            x = blk(x, mask=full_mask)
        return x


class DecoderOnly(nn.Module):
    """GPT 式：因果 mask 的 Block 栈"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, tokens):
        x = self.tok(tokens)
        B, T, _ = x.shape
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))  # 下三角：因果
        for blk in self.blocks:
            x = blk(x, mask=causal_mask)
        return x


class EncoderDecoder(nn.Module):
    """原始 Transformer：Encoder(双向) + Decoder(因果 + Cross-Attn)"""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.tok_enc = nn.Embedding(vocab_size, embed_dim)
        self.tok_dec = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )
        # Decoder Block 多一次 Cross-Attn：这里用第二个 Block 模拟
        self.decoder_self = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.cross_attn = nn.ModuleList(
            [MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.cross_ln = nn.ModuleList(
            [LayerNorm(embed_dim) for _ in range(num_layers)]
        )

    def encode(self, src_tokens):
        x = self.tok_enc(src_tokens)
        T = x.shape[1]
        full_mask = torch.ones(T, T, device=x.device)
        for blk in self.encoder:
            x = blk(x, mask=full_mask)
        return x                                     # memory: (B, n_src, d)

    def decode(self, tgt_tokens, memory):
        x = self.tok_dec(tgt_tokens)
        T = x.shape[1]
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        S = memory.shape[1]
        cross_mask = torch.ones(T, S, device=x.device)   # Decoder 每位看 Encoder 全部

        for self_blk, cattn, cln in zip(self.decoder_self, self.cross_attn, self.cross_ln):
            # 不调用 self_blk.forward（它含 self-attn+FFN），而是按子层拆开：
            x = x + self_blk.attn(self_blk.ln1(x), mask=causal_mask)[0]       # 子层1: causal self-attn
            x = x + cattn(cln(x), kv=memory, mask=cross_mask)[0]              # 子层2: cross-attn
            x = x + self_blk.ffn(self_blk.ln2(x))                              # 子层3: FFN
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    vocab, d, h, L = 1000, 64, 8, 2
    B, T_src, T_tgt = 2, 10, 8

    src = torch.randint(0, vocab, (B, T_src))
    tgt = torch.randint(0, vocab, (B, T_tgt))

    print("=== 三种架构前向对比 ===")
    enc = EncoderOnly(vocab, d, h, L)
    dec = DecoderOnly(vocab, d, h, L)
    enc_dec = EncoderDecoder(vocab, d, h, L)

    out_enc = enc(src)
    out_dec = dec(tgt)
    out_encdec = enc_dec.decode(tgt, enc_dec.encode(src))

    print(f"Encoder-only  输入:{src.shape}  输出:{out_enc.shape}")
    print(f"Decoder-only  输入:{tgt.shape}  输出:{out_dec.shape}")
    print(f"Enc-Dec       src:{src.shape} tgt:{tgt.shape}  输出:{out_encdec.shape}")

    print(f"\n参数量:")
    print(f"  Encoder-only:  {sum(p.numel() for p in enc.parameters()):,}")
    print(f"  Decoder-only:  {sum(p.numel() for p in dec.parameters()):,}")
    print(f"  Enc-Dec:       {sum(p.numel() for p in enc_dec.parameters()):,}")

    # 验证 mask 生效：Encoder 双向（全 > 0），Decoder 因果（上三角 = 0）
    print(f"\n=== mask 验证（结构事实，与随机种子无关）===")
    x = torch.randn(1, 4, d)
    _, attn_full = enc.blocks[0].attn(enc.blocks[0].ln1(x),
                                      mask=torch.ones(4, 4))
    _, attn_causal = dec.blocks[0].attn(dec.blocks[0].ln1(x),
                                        mask=torch.tril(torch.ones(4, 4)))
    print(f"Encoder(双向): 所有位置 > 0 ? {(attn_full[0, 0] > 0).all().item()}")
    print(f"Decoder(因果): 上三角(j>i) = 0 ? "
          f"{(torch.triu(attn_causal[0, 0], diagonal=1) == 0).all().item()}")
    tril_bool = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    print(f"Decoder(因果): 下三角(j<=i) > 0 ? "
          f"{(attn_causal[0, 0][tril_bool] > 0).all().item()}")
    print(f"\nDecoder 注意力矩阵结构（· 表示 >0，0 表示屏蔽）:")
    for row in attn_causal[0, 0]:
        print("  " + "  ".join("0" if v == 0 else "·" for v in row))
```

```bash
python3 architecture_variants.py
```

```text
=== 三种架构前向对比 ===
Encoder-only  输入:torch.Size([2, 10])  输出:torch.Size([2, 10, 64])
Decoder-only  输入:torch.Size([2, 8])  输出:torch.Size([2, 8, 64])
Enc-Dec       src:torch.Size([2, 10]) tgt:torch.Size([2, 8])  输出:torch.Size([2, 8, 64])

参数量:
  Encoder-only:  163,968
  Decoder-only:  163,968
  Enc-Dec:       361,472

=== mask 验证（结构事实，与随机种子无关）===
Encoder(双向): 所有位置 > 0 ? True
Decoder(因果): 上三角(j>i) = 0 ? True
Decoder(因果): 下三角(j<=i) > 0 ? True

Decoder 注意力矩阵结构（· 表示 >0，0 表示屏蔽）:
  ·  0  0  0
  ·  ·  0  0
  ·  ·  ·  0
  ·  ·  ·  ·
```

> 💡 **关键观察**：① Encoder-only 与 Decoder-only 参数量完全相同（同样 $L$ 层 Block），区别只在 mask ② Encoder-Decoder 参数量约为单栈的 $2.2$ 倍（两套 Block 栈 + Cross-Attn 子层）③ Encoder 注意力矩阵全 $> 0$（双向），Decoder 上三角严格 $= 0$（因果）——mask 的差异直接体现在注意力权重上，且这是确定的结构性质，与随机种子无关。

#### 三种 mask 的本质：一次数学看清

| mask | $\text{mask}_{ij}$ | softmax 后 $A_{ij}$ | 物理含义 |
|------|--------------------|---------------------|----------|
| Full | $1$（全部） | 全部 > 0 | 位置 $i$ 关注所有 $j$ |
| Causal | $1$ if $j \leq i$ else $0$ | $j > i$ 时 $= 0$ | 位置 $i$ 只看过去 |
| Cross | $1$（全部，shape 不同） | 全部 > 0 | Decoder 每位看 Encoder 全部 |

> 💡 **mask 的实现技巧**：用 `masked_fill(mask == 0, float('-inf'))` 而非 `masked_fill(mask == 0, 0)`——因为 softmax 前填 $-\infty$ 才能让对应位置权重精确为 0（$e^{-\infty} = 0$）。填 0 会让该位置仍参与 softmax（$e^0 = 1$），导致信息泄漏。

---

### 学习任务 6：为什么现代大模型几乎都用 Decoder-only（40 分钟）

这是 Day 6 的**核心论证**——面试高频题"为什么都用 Decoder-only"的完整答案。

#### 四个理由

| 理由 | 说明 |
|------|------|
| **① Scaling law 更优** | Kaplan et al. (2020) 显示 Decoder-only 的 loss 随参数/数据/算力幂律下降最干净。Encoder-Decoder 在同等算力下 scaling 效率略低 |
| **② 任务通用性** | Next-token prediction 是万能目标——翻译、摘要、问答、代码、CoT 都能统一成"续写"。Encoder-only 难做生成，Encoder-Decoder 需为每类任务设计输入格式 |
| **③ 训练效率** | 每个 token 都是训练信号（100% 密度），无 padding 浪费（序列即样本）。BERT 只 15% 位置有信号，Encoder-Decoder 的 Decoder 也有 padding |
| **④ 推理友好（KV Cache）** | 自回归生成 + KV Cache 让推理 $O(n \cdot d)$。涌现能力（in-context learning、CoT）在 Decoder-only 大模型上最显著 |

#### Scaling law 直觉

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

- $L$：loss，$N$：参数量，$N_c$：临界参数
- Decoder-only 的 $\alpha$ 与数据效率最优，意味着"加参数/数据，loss 下降最快"
- 这是 GPT-3 选择 Decoder-only 并放大到 175B 的理论依据

#### 任务统一：一个目标搞定一切

| 任务 | Decoder-only 的处理方式 |
|------|------------------------|
| 翻译 | `英文: Hello → 法文: Bonjour`（续写） |
| 摘要 | `文章: <长文> → 摘要: <续写>` |
| 问答 | `问题: ... → 答案: <续写>` |
| 代码 | `def fib(n):` → `<续写函数体>` |
| 推理 | `Q: ... Let's think step by step.` → `<CoT 续写>` |

无需为每类任务设计特殊头或输入格式——这是 prompting/instruction tuning 范式的基础。

#### 例外：Encoder-Decoder 仍有用武之地

并非所有场景都该用 Decoder-only：

| 场景 | 推荐架构 | 原因 |
|------|----------|------|
| 通用对话 / 续写 / CoT | Decoder-only | 任务通用 + scaling 优 |
| 语音识别（ASR） | Encoder-Decoder（Whisper） | 输入是连续音频帧，Encoder 提取声学特征，Decoder 生成文本 |
| 机器翻译（专用） | Encoder-Decoder（T5/NLLB） | 源语言与目标语言分离，Encoder 理解源、Decoder 生成目标，对齐更清晰 |
| 理解任务（嵌入/分类） | Encoder-only（BERT） | 双向上下文表示更强，嵌入质量高 |

> 💡 **一句话总结**：Decoder-only 在"通用生成 + 大规模 scaling"上胜出，成为大模型主流。但 Encoder-Decoder 在"输入输出异构"（如语音→文本）和 Encoder-only 在"理解/嵌入"场景仍有不可替代的优势。架构选择是 tradeoff，不是"哪个一定更好"。

---

### 学习任务 7：整理 architecture_variants 笔记（30 分钟）

完成 README 目录中约定的 `notes/architecture_variants.md`。建议结构：

```markdown
# Transformer 三种架构变体对比

## 1. 总览对比表
（本日学习任务 1 的总览表）

## 2. 注意力 mask：区分三种架构的关键
（Full / Causal / Cross 三种 mask 的定义与图示）

## 3. Encoder-Decoder
- 结构（Encoder + Cross-Attn + Decoder）
- Cross-Attention 的 Q/K/V 来源
- 训练（teacher forcing）vs 推理（自回归）
- 代表：原始 Transformer / T5 / BART

## 4. Decoder-only
- 结构（因果 Block 栈）
- KV Cache 原理与内存开销
- 代表：GPT / LLaMA / DeepSeek

## 5. Encoder-only
- 结构（双向 Block 栈）
- MLM 预训练
- 代表：BERT / RoBERTa

## 6. 为什么现代大模型选 Decoder-only
（学习任务 6 的四个理由）
```

> 💡 **为什么单独写 notes**：架构变体对比是"查得到、用得着"的参考资料——日后读 vLLM/TensorRT-LLM 源码或准备面试时，可直接翻 `notes/architecture_variants.md`。把 Day 6 的论证沉淀成笔记，是本周的核心产出之一。

---

### 面试题积累（本周目标 10-12 道，今日 4 道）

**Q1：Encoder-Decoder、Decoder-only、Encoder-only 三种架构分别适合什么任务？为什么现代大模型几乎都用 Decoder-only？**
> ① Encoder-Decoder（原始 Transformer / T5）：Encoder 双向看源序列，Decoder 自回归生成目标，适合 seq2seq（翻译、摘要）。② Decoder-only（GPT / LLaMA）：因果 mask + next-token prediction，适合生成（对话、续写、CoT）。③ Encoder-only（BERT）：双向 mask + MLM，适合理解（分类、NER、嵌入）。现代大模型几乎都用 Decoder-only，因为：① scaling law 最优（loss 随参数幂律下降最干净）② next-token prediction 是万能目标，prompting 统一所有任务 ③ 训练信号密度 100%，无 padding 浪费 ④ KV Cache 让自回归推理高效，且 in-context learning / CoT 等涌现能力最显著。但 Encoder-Decoder 在输入输出异构场景（如 Whisper 语音识别）、Encoder-only 在理解/嵌入场景仍有优势。

**Q2：Cross-Attention 和 Self-Attention 有什么区别？Encoder-Decoder 中 Cross-Attention 的 Q/K/V 分别来自哪里？**
> Self-Attention 的 Q/K/V 都来自同一输入 $X$（$Q=XW^Q, K=XW^K, V=XW^V$），注意力矩阵是方阵 $(n, n)$。Cross-Attention 的 Q 来自一个序列，K/V 来自另一个序列——在 Encoder-Decoder 中，Q 来自 Decoder 当前层输出 $h$，K/V 来自 Encoder 的输出 memory。注意力矩阵形状是 $(n_{\text{dec}}, n_{\text{enc}})$，让 Decoder 的每个位置"查询" Encoder 的全部输出，决定生成当前 token 时关注源序列的哪些位置。这正是 Bahdanau Attention 在 Transformer 中的化身。Cross-Attention 通常不加因果 mask（Decoder 每位都能看 Encoder 全部），但可能加 padding mask 屏蔽源序列的 padding 位。

**Q3：KV Cache 的原理是什么？为什么 Decoder-only 推理时只需缓存 K/V？KV Cache 的内存开销有多大？**
> 自回归生成时，第 $t$ 步的注意力需要 $x_1, \ldots, x_t$ 的 K/V。关键观察：已生成 token 的输入与权重不变，它们的 K/V 一旦算出就不再变化。所以把历史 K/V 缓存起来，新一步只算新 token 的 $q, k, v$，把新 $k, v$ 追加到缓存，再用 $q$ 与缓存 K 做点积。不缓存 Q 因为 Q 只属于当前 token，每步都不同，缓存无意义。内存开销：每层缓存 $K, V$ 各 $n \times d$，$L$ 层共 $2 L n d$（batch 维另乘）。以 LLaMA-7B（$L=32, d=4096$，batch=1，fp16）为例，$n=2048$ 时约 1GB。长序列 + 大 batch 时 KV Cache 显存成瓶颈，这是 vLLM PagedAttention（分块管理 KV Cache）的动机。

**Q4：BERT 为什么用双向注意力而 GPT 用单向？BERT 能直接做自回归生成吗，为什么？**
> BERT 用双向是因为它的任务是"理解"——分类、NER 等需要看完整上下文（前后 token），双向 mask 让每个位置都能看到所有 token，表示更丰富。GPT 用单向（因果 mask）是因为它的任务是"生成"——预测下一个 token 时未来的 token 还不存在，必须只能看过去。BERT 不能直接做自回归生成，因为双向注意力让每个位置"偷看"了未来 token（信息泄漏），生成时未来 token 不存在，无法用双向注意力。强行用 BERT 生成需要复杂的迭代解码（如逐步 mask + 采样），效果差且不自然。这就是"理解"与"生成"任务对 mask 的根本需求差异。

---

### 今日检查清单

- [ ] 能画出三种架构的总览对比表（结构 / mask / 任务 / 代表模型）
- [ ] 知道区分三种架构的关键是**注意力 mask**（Full / Causal / Cross），不是 Block 本身
- [ ] 能写出三种 mask 的定义（Full 全 1、Causal 下三角、Cross 形状不同但全 1）
- [ ] 理解 Encoder-Decoder 的结构：Encoder(双向) + Decoder(因果 + Cross-Attn + FFN)
- [ ] 能说清 Cross-Attention 的 Q/K/V 来源（Q 来自 Decoder，K/V 来自 Encoder memory）
- [ ] 能写出 Cross-Attention 的 shape 变化（$(n_{\text{dec}}, d) \times (d, n_{\text{enc}}) \to (n_{\text{dec}}, n_{\text{enc}})$）
- [ ] 理解 Decoder-only 就是一栈 causal Block（Day 5 的 `TransformerBlock`）
- [ ] 能写出因果 mask 的定义（$\text{mask}_{ij} = 0$ if $j \leq i$ else $-\infty$）
- [ ] 理解 next-token prediction 训练目标 $\mathcal{L} = -\sum \log P(x_{t+1} \mid x_{\leq t})$
- [ ] 能说清 KV Cache 原理：历史 K/V 不变，缓存复用，每步只算新 token 的 q/k/v
- [ ] 知道为什么不缓存 Q（Q 每步都变，缓存无意义）
- [ ] 能估算 KV Cache 内存开销（$2 L n d$，会算 LLaMA-7B 的例子）
- [ ] 知道 KV Cache 显存瓶颈是 PagedAttention 的动机
- [ ] 理解 Encoder-only（BERT）用双向 mask + MLM 预训练
- [ ] 能说出 MLM 与 next-token prediction 的差异（信号密度 15% vs 100%、双向 vs 单向）
- [ ] 知道 BERT 为什么不能直接做生成（双向信息泄漏）
- [ ] 知道 `[CLS]` / `[MASK]` / `[SEP]` 特殊 token 的作用
- [ ] 跑通 `architecture_variants.py`，验证三种架构前向 shape 正确
- [ ] 观察到 Encoder 注意力全 > 0、Decoder 上三角 = 0 的 mask 效果
- [ ] 能说出"为什么现代大模型选 Decoder-only"的四个理由
- [ ] 知道 Decoder-only 的例外场景（ASR 用 Enc-Dec、理解用 BERT）
- [ ] 完成 `notes/architecture_variants.md` 笔记
- [ ] 能画出 Decoder-only 的前向数据流（本周验收 ⑤ 的扩展）

#### 明日预告

Day 7 是本周的**收尾与里程碑**——用 Day 2-6 的全部零件从零拼出一个可训练的 mini-GPT（Decoder-only）。具体包括：Token Embedding + 位置编码（RoPE）+ $N$ 层 causal Block + LM Head，在 Shakespeare 文本上训练 next-token prediction，观察 loss 下降并用它生成可读文本。对应 README 中的 `kernels/mini_gpt.py`，也是本周验收标准 ④（mini-GPT 训练 loss 持续下降并生成可读文本）的最终交付。今天把三种架构理清后，明天你会看到 mini-GPT 就是 Day 6 的 Decoder-only 加上 Embedding 与 LM Head。建议今晚先想：mini-GPT 的前向数据流怎么画？训练时 loss 怎么算（每个位置都预测下一个 token）？生成时怎么从 logits 采样（greedy vs temperature vs top-k）？

---
