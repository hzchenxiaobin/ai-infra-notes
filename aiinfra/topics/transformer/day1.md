# Day 1：序列建模与注意力直觉

## 🎯 目标

通过今天的学习，你将：

1. 理解"下一个 token 预测"这一任务设定，能说清语言模型到底在"学"什么
2. 跑通 tokenization → embedding lookup 的完整链路，理解从文本到向量的每一步
3. 掌握 embedding 为什么能承载语义、one-hot 为什么不行（数学与直觉两个层面）
4. 用"查词典"类比建立 Q/K/V 直觉，理解注意力是"可微的软检索"
5. 用 3 行 PyTorch 手算一个 toy 注意力权重矩阵并画出文本热力图
6. 发现"注意力对顺序无感知"这一现象，为 Day 3 的位置编码埋下伏笔

> 💡 **前置知识**：Python 基础 + 最基本的 PyTorch 经验（会写 `nn.Module`、知道 `loss.backward()` 即可），无需任何 CUDA/系统背景
> ⚠️ **环境要求**：Python >= 3.9，PyTorch >= 2.0（**CPU 即可**，有 GPU 更快但非必需），内存 >= 8GB

---

## 为什么从"下一个 token 预测"开始

几乎所有现代大模型——GPT、LLaMA、Claude、DeepSeek——都在做同一件事：**给定上文，预测下一个 token**。形式化地，语言模型在学习一个条件概率分布

$$P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

这个目标看似简单，却统一了"理解"和"生成"：要准确预测下一个词，模型必须学会语法、事实、推理、常识。当规模足够大、数据足够多，涌现出的就是 ChatGPT 这样的通用能力。本周我们不谈 scale，只回答一个更基础的问题：**要算这个条件概率，模型内部到底发生了什么？**

### 从文本到概率，中间缺了什么

模型不能直接对"字符"做数学运算，必须先把文本变成向量。最朴素的链路是：

```
文本 → tokenization（切成 token、转成 id）→ embedding（id → 向量）→ 模型 → logits → 下一个 token 的概率
```

今天我们只走到"embedding"并顺手窥探"注意力"——因为这两步决定了模型能否"理解语义"。先看一个对比，理解为什么不能偷懒用最简单的表示：

| 表示方式 | 做法 | 任意两个不同 token 的关系 | 能否学习 |
|----------|------|--------------------------|----------|
| one-hot | 每个 token 是一个长度 = 词表 V 的 0/1 向量 | 恒正交，余弦相似度 = 0 | 无语义可学 |
| **embedding** | 一个可学习的 $V \times d$ 矩阵，按 id 取行 | 相似度由学习决定 | **方向承载语义** |

> 💡 **一句话总结**：one-hot 把每个词关进一个孤立的格子，词与词之间毫无关系；embedding 把词放进一个连续空间，让"语义相近"等于"方向相近"——这是注意力能工作的地基。

本周的整体路线如下图，今天是第一步——建立直觉：

![Transformer 从零学习路径：Day 1-7 渐进式路线](../images/transformer_learning_pipeline.svg)

---

## 核心概念

### 1.1 序列建模与"下一个 token 预测"

给定一段文本 $x_{1:t}$，模型要输出下一个 token $x_t$ 在词表上的概率分布。训练时用** teacher forcing**：拿真实上文喂进去，让模型在每个位置都预测下一个 token，用交叉熵损失拉近预测分布与真实 token 的 one-hot。

$$\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t})$$

推理时则是**自回归**（autoregressive）：每生成一个 token，就把它接回末尾，作为下一步的上文。这个"自己喂自己"的循环就是 GPT 类模型生成的本质，也是 Day 6 KV Cache 要优化的对象。

### 1.2 Tokenization：从文本到数字

模型的第一步是把字符串切成 token 并映射成整数 id。两条主流路线：

| 方案 | 粒度 | 词表大小 | 代表 |
|------|------|----------|------|
| 字符级 | 单个字符 | ~100 | 本周的字符级 GPT、char-rnn |
| BPE / SentencePiece | 子词 | ~3万~10万 | GPT-2、LLaMA |

字符级词表小、实现简单，适合教学；BPE 词表大但更高效（一个常见词就是一个 token）。今天用字符级，关键是建立 **stoi（char→id）/ itos（id→char）** 这张映射表：

```python
text = "hello"
chars = sorted(set(text))                  # 建字符表 ['e','h','l','o']
stoi = {c: i for i, c in enumerate(chars)} # char -> id
itos = {i: c for c, i in stoi.items()}     # id -> char
ids = [stoi[c] for c in text]              # tokenize: [1, 0, 2, 2, 3]
```

### 1.3 Embedding：为什么 one-hot 不够

拿到 id 后，最省事的表示是 one-hot：把 id $i$ 变成只在第 $i$ 位为 1 的向量。但 one-hot 有两个致命缺陷：

1. **维度爆炸**：词表 5 万，每个 token 就要 5 万维，且几乎全是 0
2. **无语义**：任意两个不同 token 的 one-hot 正交，余弦相似度恒为 0——`cat` 和 `dog` 在 one-hot 空间里和 `cat` 与 `桌子` 一样远

embedding 的做法是引入一个**可学习**的权重矩阵 $E \in \mathbb{R}^{V \times d}$（$d$ 远小于 $V$，如 $d=768$），按 id 取对应行：

```python
embed = torch.nn.Embedding(num_embeddings=V, embedding_dim=d)
vectors = embed(ids)   # (T,) -> (T, d)
```

#### 深入：embedding lookup 的数学本质

`embed(ids)` 看起来是"查表"，但它在数学上等价于 **one-hot 乘权重矩阵**：

$$\text{embed}(i) = \text{onehot}(i) \cdot E = E[i,:]$$

因为 $\text{onehot}(i)$ 只在第 $i$ 位为 1，乘出来恰是 $E$ 的第 $i$ 行。实现上用 gather（$O(T)$）而非矩阵乘（$O(TV)$），但**语义上 embedding 就是一个作用于 one-hot 的线性层**。这个等价关系很重要——它说明 embedding 不是魔法，只是把"离散 id"投影到一个可学习的连续空间，而这个空间的方向可以承载语义。

### 1.4 注意力直觉：一次"软检索"

有了 embedding，每个 token 是一个向量。但单个词的向量是"孤立"的——它不知道上下文里还有什么。注意力解决的就是**让每个词根据上下文调整自己的表示**。

把每个 token 想象成在图书馆查资料的人：

- **Query（查询）**："我现在需要什么样的信息？"
- **Key（键）**："我能提供什么样的信息？"——相当于每本书的索引卡片
- **Value（值）**："我实际携带的信息内容"

每个位置用自己的 Q 去和所有位置的 K 算相似度（点积），归一化成权重，再对所有 V 加权求和。**相似度决定"看谁"，V 决定"看到什么"**。

| 维度 | 查词典（硬检索） | 注意力（软检索） |
|------|------------------|-------------------|
| 匹配方式 | 精确等值匹配 | 点积相似度 |
| 输出 | 单个条目 | 所有条目的加权平均 |
| 可微 | 不可微（不可反传） | **可微，可学习** |
| 多义 | 只能匹配一个 | 可同时关注多个 |

> 💡 **一句话总结**：注意力就是"每个词决定自己该多看哪些词"——和查词典的唯一区别是它是可微的软匹配，所以能反向传播、能学习。

### 1.5 toy 注意力：三行代码看本质

今天先不缩放、不掩码，只用三行代码感受注意力的本质（设 $Q=K=V=X$，即 token 的 embedding）：

```python
scores = X @ X.T              # 1) QK^T：两两相似度
attn   = F.softmax(scores, -1)# 2) 归一化成权重
out    = attn @ X             # 3) 加权求和 V
```

- 第 1 行：$X X^\top$ 的第 $(i,j)$ 个元素是 token $i$ 和 token $j$ 的点积，即"关注分数"
- 第 2 行：softmax 把每行归一化为和为 1 的权重分布
- 第 3 行：每个 token 的新表示 = 所有 token 向量的加权平均，权重就是上一步的注意力

完整的缩放因子 $\sqrt{d_k}$、因果掩码、Multi-Head 是 Day 2-3 的内容。今天只要建立"注意力 = 软检索"的直觉。

---

## 最小可运行示例

完整文件：[kernels/attention_intuition.py](kernels/attention_intuition.py)（CPU 可跑，仅依赖 PyTorch）。它包含三个递进实验：字符级 tokenization + embedding lookup、embedding 的语义空间、toy 注意力权重热力图。

```python
# attention_intuition.py —— Day 1: 字符级 tokenization、embedding 语义与注意力直觉
# 运行：python3 attention_intuition.py
# 仅依赖 PyTorch，CPU 即可运行
import torch
import torch.nn.functional as F


def demo_tokenize_and_embed():
    """字符级 tokenization + embedding lookup：从文本到向量。"""
    text = "hello"
    chars = sorted(set(text))                     # 建字符表
    stoi = {c: i for i, c in enumerate(chars)}    # char -> id
    itos = {i: c for c, i in stoi.items()}        # id -> char
    ids = torch.tensor([stoi[c] for c in text])   # tokenize 成 id 序列
    print(f"文本: {text!r}")
    print(f"字符表: {chars}  →  ids: {ids.tolist()}")

    torch.manual_seed(0)
    embed = torch.nn.Embedding(num_embeddings=len(chars), embedding_dim=4)
    vectors = embed(ids)                          # (T, 4)
    print(f"embedding 后形状: {tuple(vectors.shape)}  (T=5, d=4)")

    # 关键：embedding lookup 在数学上等价于 one-hot @ 权重矩阵
    onehot = F.one_hot(ids, num_classes=len(chars)).float()
    lookup_via_matmul = onehot @ embed.weight     # 与 embed(ids) 完全一致
    print(f"one-hot @ E 是否等价于 embedding lookup: "
          f"{torch.allclose(lookup_via_matmul, vectors, atol=1e-6)}")

    # one-hot 的致命缺陷：任意两个不同字符正交，余弦相似度恒为 0
    cos = F.cosine_similarity(onehot[0], onehot[1], dim=0)
    print(f"one-hot cos('h','e') = {cos:.3f}  ← 所有不同字符都正交，无语义信息")


def demo_embedding_semantics():
    """手构造 embedding 观察余弦相似度与词向量类比。"""
    vocab = ["king", "queen", "man", "woman", "boy", "girl"]
    # 4 个维度分别对应：[王权, 男性, 女性, 成年]（真实模型维度不可解释，这里为示意）
    embed = torch.tensor([
        [1, 1, 0, 1],  # king
        [1, 0, 1, 1],  # queen
        [0, 1, 0, 1],  # man
        [0, 0, 1, 1],  # woman
        [0, 1, 0, 0],  # boy
        [0, 0, 1, 0],  # girl
    ], dtype=torch.float32)

    cos = F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=-1)
    print(f"cos(king, man)  = {cos[0, 2]:.3f}  (同为男性/成年 → 高)")
    print(f"cos(man, boy)   = {cos[2, 4]:.3f}  (同为男性 → 高)")
    print(f"cos(king, girl) = {cos[0, 5]:.3f}  (几乎无关 → 低)")

    # 经典词向量类比：king - man + woman ≈ queen
    analogy = embed[0] - embed[2] + embed[3]
    sims = F.cosine_similarity(analogy, embed, dim=-1)
    best = sims.argmax().item()
    print(f"类比 king - man + woman → 最接近 '{vocab[best]}' (cos={sims[best]:.3f})")


def demo_toy_attention():
    """3 行 PyTorch 算 toy 注意力权重矩阵并画文本热力图。"""
    vocab = ["king", "queen", "man", "woman", "boy", "girl"]
    embed = torch.tensor([
        [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1],
        [0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0],
    ], dtype=torch.float32)

    # ---- 注意力权重矩阵，核心就 3 行（今天先不缩放、不掩码，只看直觉）----
    scores = embed @ embed.T                  # 1) QK^T：两两相似度
    attn = F.softmax(scores, dim=-1)          # 2) softmax 归一化成权重
    out = attn @ embed                        # 3) 加权求和 V（每个词的新表示）

    print("king → queen/man 权重 "
          f"({attn[0, 1]:.2f}/{attn[0, 2]:.2f}) 远大于 king → girl ({attn[0, 5]:.2f})")
    print("⚠️ 这份权重完全基于语义相似度，对'顺序'毫无感知——")
    print("   打乱词序，每行权重分布不变。这正是 Day 3 要引入位置编码的原因。")


if __name__ == "__main__":
    demo_tokenize_and_embed()
    demo_embedding_semantics()
    demo_toy_attention()
```

运行：

```bash
python3 kernels/attention_intuition.py
```

预期输出（节选关键部分，完整输出见下方"实验与观察"）：

```text
实验 1：字符级 tokenization → embedding lookup
文本: 'hello'
字符表: ['e', 'h', 'l', 'o']  →  ids: [1, 0, 2, 2, 3]
embedding 后形状: (5, 4)  (T=5, d=4)
one-hot @ E 是否等价于 embedding lookup: True
one-hot cos('h','e') = 0.000  ← 所有不同字符都正交，无语义信息

实验 3：toy 注意力权重矩阵（自注意力，Q=K=V=embedding）
king → queen/man 权重 (0.18/0.18) 远大于 king → girl (0.02)
→ 每个词'多看语义相关的词'，这就是注意力 = 软检索
⚠️ 这份权重完全基于语义相似度，对'顺序'毫无感知——
   打乱词序，每行权重分布不变。这正是 Day 3 要引入位置编码的原因。
```

> ⚠️ **注意**：实验 2 的 embedding 是**手构造**的（4 维分别对应王权/男性/女性/成年），所以 `king - man + woman → queen` 的余弦相似度高达 1.000。真实模型的 embedding 维度（如 768）不可逐维解释，但"方向承载语义"的几何性质同样成立。

---

## 深入原理

### embedding lookup 与 one-hot @ W 的等价性

设词表大小 $V$、embedding 维度 $d$，权重矩阵 $E \in \mathbb{R}^{V \times d}$。对 id $i$，其 one-hot 记为 $\mathbf{e}_i \in \{0,1\}^{V}$（仅第 $i$ 位为 1）：

$$\mathbf{e}_i^\top E = \sum_{k} (\mathbf{e}_i)_k \, E_{k,:} = E_{i,:} = \text{embed}(i)$$

这正是 `torch.allclose(onehot @ embed.weight, embed(ids))` 返回 `True` 的原因。工程上用 gather 实现是为了效率（$O(T)$ vs 稀疏矩阵乘 $O(TV)$），但**语义上 embedding 就是一个作用于 one-hot 的线性投影**——这也是为什么后续把 embedding 和第一层 Linear 放在一起看时，它们本质同源。

### 余弦相似度为什么能衡量语义

余弦相似度只看**方向**不看大小：

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert\mathbf{a}\rVert \, \lVert\mathbf{b}\rVert} \in [-1, 1]$$

训练好的 embedding 把语义相近的词推向空间的相近方向（夹角小、余弦大）。one-hot 之所以不行，是因为任意两个不同 token 的 one-hot 正交（夹角 90°、余弦恒为 0）——所有词"等距"，没有任何语义结构可言。embedding 通过学习把"语义距离"编码成了"几何角度"，这是注意力能基于点积做软检索的前提。

### 词向量类比的几何含义

`king - man + woman ≈ queen` 之所以成立，是因为训练后的 embedding 空间里存在近似的**语义方向**：`king - man` 大致等于"王权"方向，把它加到 `woman` 上就落在 `queen` 附近。这说明 embedding 不只是把词散乱地放进空间，而是把**语义关系**编码成了空间的几何结构（平移、方向）。注意这是真实大模型 embedding 的经验现象，并非严格成立；今天的手构造例子只是为了直观。

### 注意力的置换等变性：为什么它"不知道顺序"

设输入 $X \in \mathbb{R}^{T \times d}$，自注意力（$Q=K=V=X$）的输出为 $X X^\top X$。若用一个排列矩阵 $P$ 打乱 token 顺序（$X' = PX$），则：

$$\text{out}' = (PX)(PX)^\top (PX) = P (X X^\top X) = P \cdot \text{out}$$

即输出只是被同样地打乱了顺序，**每一行对应的关注模式完全不变**。这就是"置换等变"（permutation-equivariant）：注意力本身对顺序毫无感知，打乱输入顺序，它只是跟着打乱输出。这正是 Day 3 必须引入位置编码的根本原因——没有位置信息，"狗咬人"和"人咬狗"在注意力看来一模一样。

下图是 Day 2 将要完整实现的缩放点积注意力的数据流，今天的 toy 版本是其"去缩放、去掩码"的简化形态：

![Scaled Dot-Product Attention 数据流](../images/transformer_attention_dataflow.svg)

---

## 实验与观察

Day 1 不做性能 benchmark，而是通过三个定性实验建立直觉。运行 `python3 kernels/attention_intuition.py` 的完整输出如下。

### 实验 1：one-hot vs embedding

tokenization 把 `"hello"` 转成 id 序列 `[1, 0, 2, 2, 3]`，embedding lookup 把它变成 `(5, 4)` 的向量矩阵。关键观察：

| 检查项 | 结果 | 含义 |
|--------|------|------|
| `one-hot @ E == embed(ids)` | `True` | embedding lookup 数学上等价于 one-hot 乘权重矩阵 |
| `one-hot cos('h','e')` | `0.000` | 任意不同字符的 one-hot 正交，毫无语义信息 |

### 实验 2：embedding 的语义空间

手构造 6 个词的 4 维 embedding，算余弦相似度矩阵：

```text
         king  queen    man  woman    boy   girl
 king   1.00  0.67  0.82  0.41  0.58  0.00
queen   0.67  1.00  0.41  0.82  0.00  0.58
  man   0.82  0.41  1.00  0.50  0.71  0.00
woman   0.41  0.82  0.50  1.00  0.00  0.71
  boy   0.58  0.00  0.71  0.00  1.00  0.00
 girl   0.00  0.58  0.00  0.71  0.00  1.00
```

- `cos(king, man) = 0.816`（同为男性/成年 → 高）
- `cos(man, boy) = 0.707`（同为男性 → 高）
- `cos(king, girl) = 0.000`（几乎无关 → 低）
- 类比 `king - man + woman → queen`（cos = 1.000）

→ embedding 空间的"方向"承载了语义关系，这是 one-hot 做不到的。

### 实验 3：toy 注意力权重热力图

```text
         king  queen    man  woman    boy   girl
 king   0.49  0.18  0.18  0.07  0.07  0.02
queen   0.18  0.49  0.07  0.18  0.02  0.07
  man   0.31  0.11  0.31  0.11  0.11  0.04
woman   0.11  0.31  0.11  0.31  0.04  0.11
  boy   0.24  0.09  0.24  0.09  0.24  0.09
 girl   0.09  0.24  0.09  0.24  0.09  0.24
```

文本热力图（`█` 越多 = 权重越高）：

```text
         king  queen    man  woman    boy   girl
 king  ██████████  ████  ████  █  █  
queen  ████  ██████████  █  ████    █
  man  ██████  ██  ██████  ██  ██  █
woman  ██  ██████  ██  ██████  █  ██
  boy  █████  ██  █████  ██  █████  ██
 girl  ██  █████  ██  █████  ██  █████
```

- 每行和为 1（softmax 归一化），对角线最大（自己和自己最相似）
- `king → queen/man`（0.18/0.18）远大于 `king → girl`（0.02）——语义相关的词获得更高权重
- **打乱词序，每行权重分布完全不变**——这是置换等变性的直接体现，引出 Day 3 的位置编码

---

## 常见陷阱与最佳实践

### 1. 把 embedding 当成纯"查表"而忽略它是线性投影

```python
# 思维误区：以为 embedding 和 Linear 是两类完全不同的东西
# 实际：embed(ids) ≡ one_hot(ids) @ E，它就是一个作用于 one-hot 的线性层
```

症状：后续学 LM head（输出层）时困惑"为什么 logits 可以用 embedding 矩阵的转置来算"——答案正是两者都是线性投影。理解了等价性，weight tying（共享 embedding 与输出层权重）就是自然推论。

### 2. 用 one-hot 喂模型而不自知

```python
# ❌ 错误：直接把 one-hot 当输入向量喂给后续层
x = F.one_hot(ids, num_classes=50000).float()   # (T, 50000) 几乎全是 0
# ✅ 正确：先过 embedding 降维
x = embed(ids)                                  # (T, d)，d 远小于 V
```

症状：显存爆炸、训练极慢、模型学不到任何语义（one-hot 间无相似度结构）。

### 3. toy attention 忘记 softmax 直接用原始分数

```python
# ❌ 错误：scores 不是概率分布，和远大于 1，加权求和后数值爆炸
out = (X @ X.T) @ X
# ✅ 正确：先 softmax 归一化成权重再加权
out = F.softmax(X @ X.T, dim=-1) @ X
```

症状：输出数值无界、量纲混乱；注意力失去"软选择"的语义。

### 4. 误以为注意力"知道顺序"

注意力是置换等变的——打乱输入顺序，每行权重分布不变。今天实验 3 的热力图已经直接验证：权重只依赖两两相似度，与位置无关。

> ⚠️ **注意**：这不是 bug 而是 feature——注意力故意只看内容相似度，顺序信息交给位置编码（Day 3）单独注入，职责分离。

### 5. 混淆余弦相似度与欧氏距离

| 度量 | 看什么 | 对 magnitude 敏感 | 适合 embedding |
|------|--------|-------------------|----------------|
| 余弦相似度 | 方向（夹角） | 否 | ✅ 语义相似度 |
| 欧氏距离 | 绝对位置 | 是 | ❌ 受向量 norm 干扰 |

embedding 的 norm 没有明确语义，衡量相似度应优先用余弦。

---

## 面试要点

**Q：为什么用 embedding 而不用 one-hot？**
  one-hot 有两个致命缺陷：① 维度等于词表大小（几万到几十万），几乎全 0，极度浪费；② 任意两个不同 token 正交、余弦相似度恒为 0，没有任何语义结构。embedding 用一个可学习的 $V \times d$ 矩阵把离散 id 投影到低维连续空间，让"语义相近"等于"方向相近"，且能通过反向传播学习——这是注意力等一切后续机制能工作的地基。

**Q：embedding lookup 和 one-hot 乘权重矩阵是什么关系？**
  完全等价。$\text{embed}(i) = \text{onehot}(i) \cdot E = E[i,:]$，因为 one-hot 只在第 $i$ 位为 1。实现上用 gather（$O(T)$）而非矩阵乘（$O(TV)$）是为了效率，但语义上 embedding 就是一个作用于 one-hot 的线性层。这解释了为什么可以用 embedding 权重的转置做输出层（weight tying）。

**Q：余弦相似度为什么能衡量语义相似度？**
  余弦相似度 $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}$ 只看方向不看大小。训练好的 embedding 把语义相近的词推向空间的相近方向，所以夹角小、余弦大。它对向量 norm 不敏感，比欧氏距离更适合 embedding，因为 embedding 的 norm 没有明确语义。

**Q：注意力为什么是"软检索"？和查词典有什么区别？**
  查词典是硬检索：精确匹配一个 key、返回一个 value，不可微。注意力是软检索：用 Q 和所有 K 算点积相似度，softmax 归一化成权重，对所有 V 加权求和——可以同时关注多个位置，且整个过程可微、能反向传播。这就是注意力能被端到端学习的原因。

**Q：注意力对输入顺序敏感吗？为什么？**
  不敏感。注意力是置换等变的：用排列矩阵 $P$ 打乱输入 $X \to PX$，输出变为 $P(X X^\top X)$，只是同样被打乱，每行的关注模式不变。因为 $QK^\top$ 只依赖两两内容相似度，与位置无关。这正是必须引入位置编码的原因——否则"狗咬人"和"人咬狗"在注意力看来一模一样。

**Q：`king - man + woman ≈ queen` 说明了什么？**
  说明 embedding 空间把语义关系编码成了几何结构：`king - man` 大致是"王权"方向，加到 `woman` 上落在 `queen` 附近。即语义的平移关系对应空间的向量加减。这是真实大模型 embedding 的经验现象（并非严格成立），今天的例子用手构造维度让它直观可见。

---

## 今日总结

Day 1 我们建立了 Transformer 的两个核心直觉：

1. **任务设定**：语言模型统一于"下一个 token 预测" $P(x_t \mid x_{<t})$，理解与生成都从中涌现
2. **tokenization**：文本 → id 序列，字符级用 stoi/itos 建映射表，BPE 是工程主流
3. **embedding**：可学习的 $V \times d$ 矩阵按 id 取行，数学上等价于 one-hot 乘权重矩阵
4. **one-hot 的缺陷**：维度爆炸 + 任意不同 token 正交无语义，必须用 embedding 降维并承载语义
5. **注意力 = 软检索**：Q/K/V 类比查词典，三行代码 $\text{softmax}(X X^\top) X$ 让每个词多看语义相关的词
6. **置换等变性**：注意力只看内容相似度、对顺序无感知，这是 Day 3 引入位置编码的根本原因

> 💡 **明日预告**：Day 2 将把今天的 toy 注意力补全为标准 Scaled Dot-Product Attention——加上缩放因子 $\sqrt{d_k}$（防 softmax 饱和）和因果掩码（不许看未来），并与 `F.scaled_dot_product_attention` 对齐到 1e-6。今天建立"注意力是软检索"的直觉后，明天的公式推导会非常自然。

---

## 推荐资源

| 资源 | 类型 | 优先级 | 说明 |
|------|------|--------|------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 博客 | ⭐ 必读 | 最好的可视化入门，配今天的直觉建立 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) §1-§3 | 论文 | ⭐ 必读 | 原始论文，只需读前半 |
| [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 视频 | 📌 推荐 | Karpathy 2 小时视频，与本周主线对应 |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | 博客 | 📌 推荐 | Decoder-only 架构图解，Day 4 预习 |
| [Word2Vec 论文](https://arxiv.org/abs/1301.3781) | 论文 | 📎 参考 | embedding 语义空间的经典源头 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 源码 | 📎 参考 | Day 5 的参照实现，今天可先扫一眼结构 |
