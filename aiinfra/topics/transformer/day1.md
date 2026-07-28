# Day 1（周一）：序列建模与注意力直觉

> **本周定位**：本专题是 [CUDA 专题](../cuda/README.md) / [Triton 专题](../triton/README.md) / [MoE 专题](../moe/README.md) 的**算法前置**——从 0 理解 Transformer 本身，后续专题再回答"怎么把它在 GPU 上跑飞快"。本周不用 `nn.Transformer`，纯手写一个字符级 GPT（~1M 参数）并在 Tiny Shakespeare 上训练。
> **前置要求**：有 Python 基础和最基本的 PyTorch 经验（会写 `nn.Module`、知道 `loss.backward()` 即可），**无需任何 CUDA/系统背景，CPU 即可**
> **今日目标**：理解"下一个 token 预测"这个任务设定；跑通 tokenization → embedding lookup；用"查词典"类比建立 Q/K/V 直觉，能用一句话向不懂的人解释"注意力就是每个词决定自己该多看哪些词"
> **时间投入**：2h（理论 1h + 动手 1h）
> **面试考察度**：⭐⭐⭐ 了解级，能说清 Transformer 解决了 RNN 什么痛点、embedding 为什么能承载语义、注意力为什么是"软检索"

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| "下一个 token 预测"任务设定 + RNN/Transformer 对比 | ④ 训练 loss 正常下降（理解训练目标的前提） |
| 字符级 tokenization → embedding lookup 流程 | ② 手写 MHA 的输入是 embedding（Day 2-3 前置） |
| Q/K/V "软检索"直觉 | ① 白板默写 attention 公式（Day 2 前置：先有直觉再推公式） |
| toy 注意力权重热力图 + "对顺序无感知"观察 | ③ 讲清位置编码为什么必需（Day 3 的引子） |

---

### 学习任务 1：序列建模与"下一个 token 预测"（25 分钟）

#### 序列建模的本质：从历史预测未来

序列建模（sequence modeling）的目标是：给定已观察到的序列 $x_1, \dots, x_{t-1}$，预测下一个元素 $x_t$。语言模型把"元素"定义为 token（字符/子词/词），任务就是：

$$P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

整段文本的概率由链式法则分解为各步条件概率的乘积：

$$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

- **训练目标**：最大化这个似然，等价于最小化交叉熵损失 $\mathcal{L} = -\frac{1}{T}\sum_t \log P(x_t \mid x_{<t})$
- **生成（推理）**：从模型采样 $x_t$，再把它拼回输入预测 $x_{t+1}$……如此自回归地 rolling forward——这就是 Day 6 `generate` 函数干的事

> 💡 **一句话总结**：所有 GPT 类模型都在做同一件事——**"猜下一个词"**。这个看似简单的任务，只要模型足够大、数据足够多，就能涌现出翻译、写代码、推理等能力（本周的小模型只负责生成"莎士比亚腔"）。

#### RNN 的痛点：为什么需要新架构

RNN/LSTM 是 Transformer 之前序列建模的主力，它逐步串行地传递隐状态：$h_t = f(h_{t-1}, x_t)$。与 Transformer 的对比最能说明它解决了什么痛点：

| 维度 | RNN / LSTM | Transformer |
|------|-----------|-------------|
| 序列依赖 | 逐步串行，$t$ 必须等 $t-1$ 算完 | 所有位置并行计算 |
| 长程依赖 | 梯度要穿过 $T$ 步，易消失 | 任意两位置直接相连，路径长度为 1 |
| 训练效率 | 无法并行，GPU 利用率低 | 矩阵乘法为主，吃满 GPU |
| 上下文长度 | 实际有效上下文短 | 轻松上千，现代模型达百万 |

> 💡 **一句话总结**：RNN 是"排队一个一个传话"，Transformer 是"所有人同时开会、任意两人直接对话"——并行性换来了可扩展性，这就是大模型时代的地基。今天先建立这个动机，Day 2 再看 Transformer 具体怎么"开会"。

### 学习任务 2：从字符到向量——tokenization 与 embedding（25 分钟）

Transformer 不能直接吃文本，必须先变成数值向量。这一步分两阶段：**tokenization**（文本 → 整数 id）和 **embedding lookup**（id → 稠密向量）。

#### tokenization：文本切分成 token

| 方案 | 做法 | 词表大小 | 本周选择 |
|------|------|---------|---------|
| 字符级 | 每个字符一个 token | ~100 | ✅ 简单、可训、足以演示 |
| 词级 | 按空格/标点切词 | 数万～数十万 | 词表大、稀疏 |
| 子词（BPE/WordPiece） | 高频词整体 + 低频词拆段 | 数千～数万 | GPT-2/LLaMA 实际方案 |

本周 Day 5 用字符级：Tiny Shakespeare 只有 ~65 个字符，模型小、训练快，但足以学到单词级连贯。今天的代码也是字符级。

#### 为什么 one-hot 不够

最朴素的"id → 向量"是 one-hot：把 id $i$ 表示成第 $i$ 位为 1、其余为 0 的 $V$ 维向量（$V$ = 词表大小）。它有三个致命缺陷：

1. **所有词两两正交**：任意两个不同 token 的余弦相似度恒为 0——"king"和"queen"与"king"和"apple"一样远，没有任何语义结构
2. **维度 = 词表大小**：GPT-2 词表 50257，每个词要 50257 维，又稀疏又巨大
3. **不可学习**：one-hot 是固定的，无法在训练中被调整

#### embedding：低维稠密、可学习的"语义坐标"

embedding 把每个 id 映射成一个 $d$ 维稠密向量（$d \ll V$，GPT-2 small 用 768），用一个可学习的矩阵 $E \in \mathbb{R}^{V \times d}$：

$$\text{embed}(i) = E[i] \quad\text{（取第 } i \text{ 行）}$$

关键洞察：**embedding lookup 在数学上等价于 one-hot 矩阵乘法**——

$$\text{onehot}(i) \cdot E = E[i]$$

所以 `nn.Embedding` 不是什么新魔法，就是一个"查表"，写成矩阵乘也完全等价（今天的实验 1 会验证这一点）。但它解决了 one-hot 的全部缺陷：

- 低维稠密（$d=768$ 而非 $V=50257$）
- **可学习**：$E$ 是模型参数，随训练更新
- **承载语义**：训练中出现在相似上下文里的词，embedding 会被推向相近的方向（distributional semantics——"一个词的意思由它的上下文决定"）

> 💡 **为什么 embedding 能承载语义**：语言模型的训练目标是"预测上下文"。要让"king"和"queen"都能被正确预测，它们在相似上下文（如 "The ___ ruled the kingdom"）里出现时承担相似角色，梯度自然把它们推向相近的向量。embedding 的语义结构是**训练涌现**的，不是人为指定的——今天的实验 2 用一个手构造的示意 embedding 直观展示这种结构。

### 学习任务 3：注意力直觉——从"查词典"到"软检索"（10 分钟）

这是 Day 1 的核心直觉，为 Day 2 手写公式做准备。把每个 token 想象成在图书馆查资料的人：

- **Query（查询）**："我现在需要什么样的信息？"
- **Key（键）**："我能提供什么样的信息？"——相当于每本书的索引卡片
- **Value（值）**："我实际携带的信息内容"

每个位置用自己的 Q 去和所有位置的 K 算相似度（点积），softmax 归一化成权重，再对所有 V 加权求和。**相似度决定"看谁"，V 决定"看到什么"**。

和查词典的唯一区别是：词典是**精确匹配**（hard retrieval，找到那一本），注意力是**可微的软匹配**（soft retrieval，每本都看一点、按相似度分配比重）——因此可以反向传播、可以学习。

> 💡 **验收一句话**：注意力就是**每个词决定自己该多看哪些词**——用自己的 Q 去和别人的 K 比相似度，按相似度把别人的 V 加权拿过来。今天的实验 3 会把这个直觉跑成一张热力图。

### 学习任务 4：动手实验——embedding 语义 + 注意力热力图（60 分钟）

完整文件：[kernels/attention_intuition.py](kernels/attention_intuition.py)（CPU 可跑，仅依赖 PyTorch）

#### 实验 1：字符级 tokenization + embedding lookup

```python
# attention_intuition.py（节选）—— 实验 1
import torch
import torch.nn.functional as F

text = "hello"
chars = sorted(set(text))                     # 建字符表
stoi = {c: i for i, c in enumerate(chars)}    # char -> id
ids = torch.tensor([stoi[c] for c in text])   # tokenize 成 id 序列

torch.manual_seed(0)
embed = torch.nn.Embedding(num_embeddings=len(chars), embedding_dim=4)
vectors = embed(ids)                          # (T, 4) embedding lookup

# embedding lookup ≡ one-hot @ 权重矩阵
onehot = F.one_hot(ids, num_classes=len(chars)).float()
print(torch.allclose(onehot @ embed.weight, vectors, atol=1e-6))  # True
```

#### 实验 2：embedding 的语义空间

用 4 维手构造 embedding（维度对应 [王权, 男性, 女性, 成年]，真实模型维度不可解释，这里为示意），观察余弦相似度与经典词向量类比：

```python
# 实验 2（节选）
vocab = ["king", "queen", "man", "woman", "boy", "girl"]
embed = torch.tensor([
    [1, 1, 0, 1],  # king
    [1, 0, 1, 1],  # queen
    [0, 1, 0, 1],  # man
    [0, 0, 1, 1],  # woman
    [0, 1, 0, 0],  # boy
    [0, 0, 1, 0],  # girl
], dtype=torch.float32)

cos = F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=-1)
# 经典类比：king - man + woman ≈ queen
analogy = embed[0] - embed[2] + embed[3]
sims = F.cosine_similarity(analogy, embed, dim=-1)
print(vocab[sims.argmax()], sims.max())       # queen 1.0
```

#### 实验 3：toy 注意力权重矩阵（核心 3 行）

今天先不缩放、不加掩码，把 embedding 同时当作 Q=K=V，看"谁看谁"的权重：

```python
# 实验 3（核心 3 行）
scores = embed @ embed.T                  # 1) QK^T：两两相似度
attn = F.softmax(scores, dim=-1)          # 2) softmax 归一化成权重
out = attn @ embed                        # 3) 加权求和 V
```

#### 完整运行与预期输出

```bash
python3 kernels/attention_intuition.py
```

```text
================================================================
实验 1：字符级 tokenization → embedding lookup
================================================================
文本: 'hello'
字符表: ['e', 'h', 'l', 'o']  →  ids: [1, 0, 2, 2, 3]
embedding 后形状: (5, 4)  (T=5, d=4)
'l' 的向量: [0.3223, -1.2633, 0.35, 0.3081]
one-hot @ E 是否等价于 embedding lookup: True
one-hot cos('h','e') = 0.000  ← 所有不同字符都正交，无语义信息

================================================================
实验 2：embedding 的语义空间（余弦相似度 + 类比）
================================================================
余弦相似度矩阵：
         king  queen    man  woman    boy   girl
 king  1.00  0.67  0.82  0.41  0.58  0.00
queen  0.67  1.00  0.41  0.82  0.00  0.58
  man  0.82  0.41  1.00  0.50  0.71  0.00
woman  0.41  0.82  0.50  1.00  0.00  0.71
  boy  0.58  0.00  0.71  0.00  1.00  0.00
 girl  0.00  0.58  0.00  0.71  0.00  1.00

cos(king, man)  = 0.816  (同为男性/成年 → 高)
cos(man, boy)   = 0.707  (同为男性 → 高)
cos(king, girl) = 0.000  (几乎无关 → 低)

类比 king - man + woman → 最接近 'queen' (cos=1.000)
→ embedding 空间的'方向'承载了语义关系，这是 one-hot 做不到的

================================================================
实验 3：toy 注意力权重矩阵（自注意力，Q=K=V=embedding）
================================================================
注意力权重矩阵（行=查询词，列=被看词），每行和为 1：
         king  queen    man  woman    boy   girl
 king  0.49  0.18  0.18  0.07  0.07  0.02
queen  0.18  0.49  0.07  0.18  0.02  0.07
  man  0.31  0.11  0.31  0.11  0.11  0.04
woman  0.11  0.31  0.11  0.31  0.04  0.11
  boy  0.24  0.09  0.24  0.09  0.24  0.09
 girl  0.09  0.24  0.09  0.24  0.09  0.24

king → queen/man 权重 (0.18/0.18) 远大于 king → girl (0.02)
→ 每个词'多看语义相关的词'，这就是注意力 = 软检索

文本热力图（█ 越多 = 权重越高）：
         king  queen    man  woman    boy   girl
 king  ██████████  ████  ████  █  █  
queen  ████  ██████████  █  ████    █
  man  ██████  ██  ██████  ██  ██  █
woman  ██  ██████  ██  ██████  █  ██
  boy  █████  ██  █████  ██  █████  ██
 girl  ██  █████  ██  █████  ██  █████

加权求和后 king 的新表示: [0.6652, 0.7311, 0.2689, 0.91]
⚠️ 这份权重完全基于语义相似度，对'顺序'毫无感知——
   打乱词序，每行权重分布不变。这正是 Day 3 要引入位置编码的原因。
```

#### 三个关键观察

| 实验 | 观察 | 对应"为什么" |
|------|------|------------|
| 1 | `one-hot @ E` 与 `embed(ids)` 完全一致 | embedding lookup = 查表 = one-hot 矩阵乘 |
| 1 | one-hot 任意两词余弦相似度 = 0 | one-hot 无语义，必须用可学习 embedding |
| 2 | king-man 相似度 0.82 远高于 king-girl 0.00 | embedding 空间方向承载语义 |
| 2 | king - man + woman ≈ queen（cos=1.0） | 语义关系可做向量算术（one-hot 做不到） |
| 3 | king 对 queen/man 权重 0.18 > 对 girl 0.02 | 注意力 = 按相似度"软检索"相关词 |
| 3 | 打乱词序权重不变 | 注意力对顺序无感知 → Day 3 位置编码的动机 |

> ⚠️ **今天故意省略的两件事**（Day 2 补上）：① **缩放因子** $\sqrt{d_k}$——今天 $d=4$ 数值小看不出来，$d$ 大时 softmax 会饱和；② **因果掩码**——今天是双向"开会"，decoder 生成时不许看未来。明天手写完整公式时会把这两点补齐。

### 面试题积累（本周目标 8-10 道，今日 3 道）

**Q1：为什么 one-hot 不适合表示词？embedding 为什么能承载语义？**
> 答：one-hot 维度等于词表大小、又稀疏又巨大，且任意两个不同 token 正交（余弦相似度恒为 0），"king"和"queen"与"king"和"apple"一样远，没有任何语义结构，也无法学习。embedding 把每个 id 映射成低维稠密向量（如 768 维），权重可学习；训练中出现在相似上下文里的词会被梯度推向相近方向（distributional semantics），于是 embedding 空间涌现出语义结构（如 king-man 相似度高于 king-girl，king-man+woman≈queen）。注意 embedding lookup 在数学上等价于 one-hot @ 权重矩阵——`nn.Embedding` 本质就是一张可学习的查表。

**Q2：用一句话解释注意力机制，Q/K/V 各是什么？**
> 答：注意力是"每个词决定自己该多看哪些词"——每个位置用 Query（"我需要什么信息"）去和所有位置的 Key（"我能提供什么信息"）算点积相似度，softmax 归一化成权重，再对 Value（"实际携带的内容"）加权求和。和查词典的区别是：词典是精确匹配（hard retrieval），注意力是可微的软匹配（soft retrieval）——因此可以反向传播、可以学习。**相似度决定"看谁"，V 决定"看到什么"**。

**Q3：RNN 和 Transformer 在处理序列时有什么本质区别？为什么 Transformer 能 scale 到那么大？**
> 答：RNN 逐步串行传递隐状态，$t$ 必须等 $t-1$ 算完，无法并行、长程梯度易消失；Transformer 让所有位置一次性两两交互（注意力），任意两位置路径长度为 1，且主体是矩阵乘法可吃满 GPU 并行算力。并行性带来了可扩展性——模型越大、数据越多越能受益，这是 Transformer 能 scale 到百亿千亿参数的地基。代价是注意力对序列长度 $O(T^2)$ 的复杂度，催生了 FlashAttention、KV Cache、稀疏/线性注意力等后续优化（Day 6-7）。

### 今日检查清单

- [ ] 能说出序列建模/语言模型的任务设定（"预测下一个 token"，链式法则分解）
- [ ] 能列出 RNN 相比 Transformer 的 3 个痛点（串行/长程梯度/无法并行）
- [ ] 跑通字符级 tokenization → embedding lookup，知道 `nn.Embedding` 本质是查表
- [ ] 能解释为什么 one-hot 不够（正交/高维稀疏/不可学习）
- [ ] 能解释 embedding 为什么能承载语义（训练涌现 + distributional semantics）
- [ ] 验证 `one-hot @ E` 与 `embed(ids)` 完全一致
- [ ] 能用 Q/K/V "查词典"类比解释注意力，说出"相似度决定看谁，V 决定看到什么"
- [ ] 能区分 hard retrieval（词典）与 soft retrieval（注意力，可微可学习）
- [ ] 跑通 toy 注意力热力图，观察到"语义相关的词权重更高"
- [ ] 发现"注意力对顺序无感知"——能说清这是 Day 3 位置编码的动机
- [ ] 知道今天省略了缩放因子和因果掩码，明天补齐

#### 明日预告

Day 2 将正式手写 **Scaled Dot-Product Attention** $\text{softmax}(QK^\top / \sqrt{d_k}) V$——把今天的"3 行 toy 版"补上缩放因子和因果掩码，并与 PyTorch 官方 `F.scaled_dot_product_attention` 对齐到 1e-6。今天建立了"注意力 = 软检索"的直觉，明天要回答两个"为什么"：为什么除以 $\sqrt{d_k}$（点积方差随 $d$ 线性增长，不缩放则 softmax 饱和、梯度消失），以及为什么因果掩码填 $-\infty$ 而不是 0（填 0 仍有 $\exp(0)=1$ 的权重、未来信息泄漏）。建议今晚先扫一眼 [kernels/attention_from_scratch.py](kernels/attention_from_scratch.py) 的核心 15 行，为明天的对齐实验做准备。

---
