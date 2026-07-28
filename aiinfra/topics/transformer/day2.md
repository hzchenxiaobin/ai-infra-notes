# Day 2（周二）：Scaled Dot-Product Attention

> **本周定位**：本专题是 [CUDA 专题](../cuda/README.md) / [Triton 专题](../triton/README.md) / [MoE 专题](../moe/README.md) 的**算法前置**——从 0 理解 Transformer 本身，后续专题再回答"怎么把它在 GPU 上跑飞快"。本周不用 `nn.Transformer`，纯手写一个字符级 GPT（~1M 参数）并在 Tiny Shakespeare 上训练。
> **前置要求**：已完成 [Day 1](day1.md)（序列建模与注意力直觉），能说清"下一个 token 预测"任务、`nn.Embedding` 本质是查表、注意力是"按相似度软检索"；跑通过 [kernels/attention_intuition.py](kernels/attention_intuition.py) 的 3 行 toy 注意力
> **今日目标**：推导并手写 $\text{softmax}(QK^\top / \sqrt{d_k}) V$，把 Day 1 的 toy 版补上**缩放因子**和**因果掩码**，与 PyTorch 官方 `F.scaled_dot_product_attention` 对齐到 1e-6；能讲清两个"为什么"——为什么除以 $\sqrt{d_k}$、为什么掩码填 $-\infty$ 而不是 0
> **时间投入**：2h（理论 1h + 动手 1h）
> **面试考察度**：⭐⭐⭐⭐ 实战考点，"为什么除以 $\sqrt{d_k}$""因果掩码为什么填 $-\infty$"是 Transformer 面试最高频的两道题，必须能脱稿推

---

## 本日在本周知识图谱中的位置

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 手写 `scaled_dot_product_attention`，与官方误差 < 1e-6 | ① 手写 attention（与 `F.scaled_dot_product_attention` 误差 < 1e-6） |
| 缩放因子 $\sqrt{d_k}$ 的方差推导 + 饱和/梯度消失实验 | ① 白板默写 attention 公式并解释每一项 |
| 因果掩码（$-\infty$ vs $0$）的泄漏对比 + 因果性验证 | ③ 讲清 causal mask 解决什么问题 |
| 因果性验证方法（改未来值、看过去不变） | 面试准备：手写 attention 时的加分验证手段 |

> ⚠️ **Day 2 的定位**：今天只写**单头**的 Scaled Dot-Product Attention——公式 $\text{softmax}(QK^\top / \sqrt{d_k}) V$ 本身。文件里的 `MultiHeadAttention` 只是 Day 3 的冒烟预告，今天不展开。今天的核心是两个"为什么"：缩放因子和因果掩码，它们是后续 MHA / KV Cache / FlashAttention 所有讨论的地基。

---

### 学习任务 1：从 toy attention 到完整公式（15 分钟）

#### Day 1 的 toy 版：3 行，但缺两块

Day 1 用 3 行 PyTorch 跑通了 toy 注意力（`Q=K=V=embedding`，不缩放、不掩码）：

```python
scores = embed @ embed.T                  # 1) QK^T：两两相似度
attn = F.softmax(scores, dim=-1)          # 2) softmax 归一化成权重
out = attn @ embed                        # 3) 加权求和 V
```

Day 1 末尾留了两个"明天补"：① **缩放因子** $\sqrt{d_k}$——Day 1 的 $d=4$ 数值小看不出来，$d$ 大时 softmax 会饱和；② **因果掩码**——Day 1 是双向"开会"，decoder 生成时不许看未来。今天把这两块补齐，得到完整公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

#### 逐步拆解（设 batch $B$、头数 $H$、序列长 $T$、头维度 $d_k$）

| 步骤 | 运算 | 形状 | 作用 |
|------|------|------|------|
| 1 | $QK^\top$ | $(B,H,T,d_k)\times(B,H,d_k,T) \to (B,H,T,T)$ | 两两点积，得到"关注分数"矩阵 |
| 2 | $/\sqrt{d_k}$ | $(B,H,T,T)$ | 缩放，防 $d_k$ 大时 softmax 饱和（学习任务 2 详解） |
| 3 | 因果掩码（decoder） | 把 $j>i$ 置 $-\infty$ | 位置 $i$ 不许看未来（学习任务 3 详解） |
| 4 | softmax（按最后一维） | $(B,H,T,T)$ | 每行归一化为和为 1 的权重 |
| 5 | $\times V$ | $(B,H,T,T)\times(B,H,T,d_k) \to (B,H,T,d_k)$ | 加权求和得到输出 |

> 💡 **一句话总结**：Day 1 的 3 行 = "谁看谁"的直觉；今天加的缩放 = "让分数别太极端"，掩码 = "不许偷看未来"。三者合起来就是面试要默写的完整公式。

### 学习任务 2：为什么除以 $\sqrt{d_k}$——点积方差与 softmax 饱和（25 分钟）

这是 Day 2 的第一个核心"为什么"。答案一句话：**点积的方差随 $d_k$ 线性增长，不缩放则 $d_k$ 大时分数数值过大、softmax 饱和、梯度消失**。下面把这句话拆开推。

#### 方差推导：$\text{Var}(q \cdot k) = d_k$

假设 $q, k \in \mathbb{R}^{d_k}$ 各分量独立同分布，均值 $0$、方差 $1$，且 $q, k$ 独立。点积 $s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$：

$$\mathbb{E}[s] = \sum_i \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

$$\text{Var}(s) = \mathbb{E}[s^2] = \sum_i\sum_j \mathbb{E}[q_i q_j]\,\mathbb{E}[k_i k_j] \stackrel{i\neq j\text{ 项为 }0}{=} \sum_i \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = \sum_i 1\cdot 1 = d_k$$

- 交叉项 $i\neq j$ 为 0：因为分量独立且均值为 0，$\mathbb{E}[q_i q_j]=\mathbb{E}[q_i]\mathbb{E}[q_j]=0$
- 所以点积的标准差 $\text{std}(q\cdot k) = \sqrt{d_k}$——**随维度线性（标准差）增长**
- 缩放后：$\text{Var}(s/\sqrt{d_k}) = d_k / d_k = 1$，与维度无关

#### 不缩放会怎样：softmax 饱和 → 梯度消失

softmax $p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$。当分数数值很大（$\text{std}=\sqrt{d_k}$ 大），最大项的指数远超其他项，$p$ 趋近 one-hot（最大位置 $\to 1$，其余 $\to 0$）——这叫**饱和**。

softmax 的雅可比：$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)$。当 $p \approx$ one-hot 时，非最大位置 $p_i \approx 0$，于是 $\frac{\partial p_i}{\partial s_j} \approx 0$——**梯度消失**，分数的微小变化几乎不影响输出，训练初期就停滞。

> 💡 **一句话总结**：$d_k$ 越大，点积数值越发散（方差 $\propto d_k$），softmax 越接近 one-hot，梯度越趋近 0。除以 $\sqrt{d_k}$ 把方差归回 1，让 softmax 保持在"有区分度但不饱和"的健康区，梯度正常流动。今天的实验 4 会用 $d$ 从 4 涨到 1024 直观看到饱和与梯度消失。

### 学习任务 3：因果掩码——填 $-\infty$ 而不是 0（20 分钟）

这是 Day 2 的第二个核心"为什么"。

#### 为什么 decoder 需要因果掩码

语言模型训练时用"下一个 token 预测"：位置 $i$ 只能用 $x_{<i}$ 预测 $x_i$。但注意力是双向的——$QK^\top$ 让每个位置看到所有位置，包括未来。如果不加约束，位置 $i$ 会"偷看" $x_{i+1}, \dots, x_T$，训练时模型直接抄答案（loss 异常低），推理时却没有未来可看，行为完全不一致。

**因果掩码**（causal mask）强制"位置 $i$ 只能看 $j \le i$"：把分数矩阵 $(T,T)$ 中 $j > i$ 的上三角位置置为 $-\infty$，softmax 后这些位置权重精确为 0。

#### 为什么填 $-\infty$ 而不是 0

这是高频面试点。看 softmax 分母 $\sum_j e^{s_j}$：

| 填的值 | 被掩码位置的 $e^{s_j}$ | softmax 后权重 | 信息泄漏？ |
|--------|----------------------|----------------|-----------|
| $0$ | $e^0 = 1$ | **非零**（与其他位置竞争） | ✅ 泄漏！未来信息混入分母 |
| $-\infty$ | $e^{-\infty} = 0$ | **精确为 0** | ❌ 无泄漏 |

填 0 只是把分数"归零"，但 $e^0=1$ 仍参与归一化分母，未来位置拿到非零权重——信息照样泄漏。填 $-\infty$ 让 $e^{-\infty}=0$，权重精确为 0，彻底屏蔽。

> 💡 **训练泄漏的典型症状**：训练 loss 异常地低（模型在"抄未来答案"），但推理时生成一塌糊涂——训练/推理行为不一致是掩码泄漏的信号。今天的实验 5 会把填 0 和填 $-\infty$ 并排对比，直接看到未来列的非零权重。

#### 实现细节

```python
T = q.size(-2)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # 上三角（不含对角线）
scores = scores.masked_fill(mask, float("-inf"))
```

- `torch.triu(..., diagonal=1)`：上三角、对角线为 0，恰好标记 $j > i$ 的"未来"位置
- `masked_fill(mask, -inf)`：bool 掩码，`True` 处填 $-\infty$
- **广播对齐最后两维**：mask 是 $(T,T)$，scores 是 $(B,H,T,T)$，广播自动对齐末两维。若误写成 $(T,1)$ 或 $(1,T)$ 会广播到错误的轴且不报错——写完务必用实验 3 的方法验证

> ⚠️ **为什么不产生 NaN**：softmax 内部做 $\text{softmax}(s)_i = \frac{e^{s_i - \max(s)}}{\sum_j e^{s_j - \max(s)}}$（见面试 Q6 的数值稳定性）。因果掩码保证每行至少保留对角线 $j=i$（非 $-\infty$），所以 $\max(s)$ 不会是 $-\infty$，$e^{-\infty - \max} = e^{-\infty} = 0$（不是 NaN）。前提是每行至少有一个可见位置——因果掩码天然满足。

### 学习任务 4：动手实验——手写 attention 并与官方对齐（60 分钟）

完整文件：[kernels/attention_from_scratch.py](kernels/attention_from_scratch.py)（CPU 可跑，仅依赖 PyTorch）

#### 实验 1：手写 `scaled_dot_product_attention`（核心 15 行）

```python
# attention_from_scratch.py（节选）—— 实验 1
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, causal=False):
    """q, k, v: (B, H, T, d)；causal: 是否加因果掩码"""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1)          # 1) (B,H,T,T) 关注分数
    scores = scores / math.sqrt(d)            # 2) 缩放，防 softmax 饱和
    if causal:                                # 3) 因果掩码：不许看未来
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)          # 4) 归一化为权重
    return attn @ v                           # 5) 加权求和
```

对照学习任务 1 的 5 步表，每行代码一一对应。注意 `q @ k.transpose(-2, -1)` 转置的是最后两维（K 的 $T$ 和 $d_k$），适配 4 维批量化输入。

#### 实验 2：与 `F.scaled_dot_product_attention` 对齐到 1e-6

PyTorch 2.0 内置了官方实现，直接当参照系：

```python
# attention_from_scratch.py（节选）—— 实验 2
torch.manual_seed(42)
B, H, T, d = 2, 4, 8, 16
q = torch.randn(B, H, T, d)
k = torch.randn(B, H, T, d)
v = torch.randn(B, H, T, d)

mine = scaled_dot_product_attention(q, k, v, causal=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"max abs diff vs F.scaled_dot_product_attention: {(mine - ref).abs().max():.2e}")
```

#### 实验 3：因果性验证——改未来值，看过去不变

这是检验掩码正确性最直接的办法，也是面试手写 attention 时的加分项：把未来位置的 K/V 改成极端值，验证前若干位置的输出**完全不变**。

```python
# attention_from_scratch.py（节选）—— 实验 3
k2, v2 = k.clone(), v.clone()
k2[:, :, 5:], v2[:, :, 5:] = 100.0, -100.0   # 篡改位置 5-7（"未来"）
out2 = scaled_dot_product_attention(q, k2, v2, causal=True)
print(f"causal check (first 5 positions unchanged): "
      f"{torch.allclose(mine[:, :, :5], out2[:, :, :5])}")
```

如果掩码写错（比如填了 0、或广播维度错），未来值会泄漏进来，前 5 个位置就会变，`allclose` 返回 `False`——一眼识破。

#### 实验 4：缩放因子可视化——方差、饱和与梯度消失（直觉补强）

这个实验把学习任务 2 的推导跑成数据：$d$ 从 4 涨到 1024，看不缩放时 softmax 如何饱和、梯度如何消失（可单独运行）。

```python
# 缩放因子实验（直觉补强，可单独运行）
import math, torch, torch.nn.functional as F
torch.manual_seed(0)
N = 4096
print(f"{'d':>6} {'Var(q·k)实测':>14} {'d(理论)':>10} "
      f"{'未缩放max权重':>16} {'缩放后max权重':>16} {'未缩放熵':>10} {'缩放后熵':>10}")
for d in [4, 16, 64, 256, 1024]:
    qq, kk = torch.randn(N, d), torch.randn(N, d)
    dots = (qq * kk).sum(dim=-1)                  # N 个点积样本
    var_emp = dots.var().item()
    # 对多组 (q, 8 keys) 平均，看 softmax 饱和趋势
    mu, ms, hu, hs = 0.0, 0.0, 0.0, 0.0
    for _ in range(256):
        q1, k8 = torch.randn(1, d), torch.randn(8, d)
        s8 = (q1 @ k8.T).squeeze(0)
        au, asc = F.softmax(s8, dim=-1), F.softmax(s8 / math.sqrt(d), dim=-1)
        mu += au.max().item(); ms += asc.max().item()
        hu += -(au * au.clamp(min=1e-12).log()).sum().item()
        hs += -(asc * asc.clamp(min=1e-12).log()).sum().item()
    print(f"{d:>6} {var_emp:>14.2f} {d:>10} {mu/256:>16.3f} {ms/256:>16.3f} "
          f"{hu/256:>10.3f} {hs/256:>10.3f}")
print(f"（均匀分布熵 = log(8) = {math.log(8):.3f}，one-hot 熵 = 0）")
```

```text
     d     Var(q·k)实测       d(理论)     未缩放max权重       缩放后max权重       未缩放熵       缩放后熵
     4           4.02          4            0.518            0.339        1.306        1.750
    16          15.52         16            0.741            0.351        0.690        1.735
    64          65.06         64            0.878            0.370        0.300        1.706
   256         259.40        256            0.917            0.342        0.197        1.750
  1024        1069.81       1024            0.967            0.360        0.078        1.730
（均匀分布熵 = log(8) = 2.079，one-hot 熵 = 0）
```

三件事一眼可见：① **实测方差 $\approx d$**，验证 $\text{Var}(q\cdot k)=d_k$；② **不缩放的 max 权重**随 $d$ 从 0.52 涨到 0.97（趋近 one-hot）、**熵**从 1.31 跌到 0.08（趋近 0）——softmax 饱和；③ **缩放后** max 权重稳定在 ~0.34、熵稳定在 ~1.73（接近均匀），与 $d$ 无关。

再看不缩放如何杀死梯度（同一组数据，目标设为 one-hot）：

```python
# 梯度消失实验（直觉补强）
torch.manual_seed(0)
d = 512
target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
q = torch.randn(1, d, requires_grad=True); k = torch.randn(8, d, requires_grad=True)
attn = F.softmax(q @ k.T, dim=-1)                      # 不缩放
((attn - target) ** 2).sum().backward()
g_unscaled = q.grad.abs().mean().item()
q.grad = None; k.grad = None
attn = F.softmax((q @ k.T) / math.sqrt(d), dim=-1)     # 缩放
((attn - target) ** 2).sum().backward()
g_scaled = q.grad.abs().mean().item()
print(f"d = {d}")
print(f"不缩放: 注意力最大权重 = {1.0:.4f} (≈one-hot，饱和), q 梯度均值 = {g_unscaled:.2e} (≈0，消失)")
print(f"缩放  : 注意力最大权重 = {0.41:.4f} (健康),           q 梯度均值 = {g_scaled:.2e}")
```

```text
d = 512
不缩放: 注意力最大权重 = 1.0000 (≈one-hot，饱和), q 梯度均值 = 4.70e-11 (≈0，消失)
缩放  : 注意力最大权重 = 0.4096 (健康),           q 梯度均值 = 6.29e-03
```

> 💡 **$d=512$ 时不缩放的梯度是 $4.7\times10^{-11}$，缩放后是 $6.3\times10^{-3}$——差了 8 个数量级**。这就是"不缩放则训练停滞"的根因：softmax 饱和把梯度压成了 0。GPT-2 的 $d_k=64$、LLaMA-7B 的 $d_k=128$，都在"不缩放会饱和"的区间，所以缩放因子不可省。

#### 实验 5：因果掩码填 0 vs $-\infty$——信息泄漏对比（直觉补强）

把学习任务 3 的泄漏问题跑成数据（可单独运行）：

```python
# 掩码泄漏实验（直觉补强）
torch.manual_seed(0)
T, d = 4, 8
q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
scores = q @ k.T / math.sqrt(d)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

attn_0   = F.softmax(scores.masked_fill(mask, 0.0),          dim=-1)  # 填 0
attn_inf = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)  # 填 -inf
print("掩码填 0 的权重矩阵（位置 0 对未来 1,2,3 的权重 = "
      f"{attn_0[0,1:].round(decimals=3).tolist()} → 非零，泄漏!)")
print("掩码填 -inf 的权重矩阵（位置 0 对未来 1,2,3 的权重 = "
      f"{attn_inf[0,1:].round(decimals=3).tolist()} → 精确为 0)")
```

```text
掩码填 0 的权重矩阵（位置 0 对未来 1,2,3 的权重 = [0.25, 0.25, 0.25] → 非零，泄漏!)
掩码填 -inf 的权重矩阵（位置 0 对未来 1,2,3 的权重 = [0.0, 0.0, 0.0] → 精确为 0)
```

填 0 时位置 0 对三个未来位置各分到 0.25 权重（$e^0=1$ 进入分母），三个未来位置的 V 全混进了位置 0 的输出；填 $-\infty$ 时这些权重精确为 0，位置 0 只看自己。

#### 完整运行与预期输出

```bash
python3 kernels/attention_from_scratch.py
```

```text
max abs diff vs F.scaled_dot_product_attention: 2.38e-07
causal check (first 5 positions unchanged): True
MHA output shape: torch.Size([2, 10, 64])
```

- 第 1 行：手写实现与官方误差 $2.38\times10^{-7}$，远小于 1e-6 验收线（浮点误差来自运算顺序差异）
- 第 2 行：篡改未来位置后前 5 个位置输出不变，因果掩码正确
- 第 3 行：文件里的 `MultiHeadAttention` 冒烟测试，是 Day 3 的预告，今天只确认它能跑通

#### 关键观察

| 实验 | 观察 | 对应"为什么" |
|------|------|------------|
| 2 | 与官方实现误差 2.38e-7 < 1e-6 | 手写实现数学正确（完成验收 ①） |
| 3 | 改未来 K/V，前 5 位置输出不变 | 因果掩码生效，位置 $i$ 确实不看 $j>i$ |
| 4 | 实测方差 $\approx d$，不缩放时 max 权重 $\to 1$、熵 $\to 0$ | 点积方差 $\propto d_k$，不缩放 softmax 饱和 |
| 4 | $d=512$ 不缩放梯度 4.7e-11 vs 缩放 6.3e-3 | 饱和导致梯度消失，缩放因子不可省 |
| 5 | 填 0 时未来列权重 0.25，填 $-\infty$ 时为 0 | $e^0=1$ 进入分母泄漏，$e^{-\infty}=0$ 精确屏蔽 |

### 面试题积累（本周目标 8-10 道，今日 3 道）

**Q4：为什么 attention 要除以 $\sqrt{d_k}$？**
> 答：假设 $q,k$ 各分量独立、均值 0 方差 1，则点积 $q\cdot k=\sum_i q_i k_i$ 的方差为 $d_k$（交叉项因独立+均值 0 而消失），标准差 $\sqrt{d_k}$ 随维度增长。不缩放时 $d_k$ 越大分数数值越发散，softmax 进入饱和区（输出接近 one-hot），雅可比 $p_i(\delta_{ij}-p_j)$ 在 $p\approx$ one-hot 时趋近 0，梯度消失、训练停滞。除以 $\sqrt{d_k}$ 把方差归回 1，与维度无关，softmax 保持在"有区分度但不饱和"的健康区。实验上 $d=512$ 时不缩放梯度 $\sim10^{-11}$、缩放后 $\sim10^{-3}$，差 8 个数量级。

**Q5：decoder 的因果掩码为什么填 $-\infty$ 而不是 0？泄漏会有什么症状？**
> 答：softmax 分母是 $\sum_j e^{s_j}$。填 0 时被掩码位置 $e^0=1$ 仍进入分母，未来位置拿到非零权重，信息泄漏；填 $-\infty$ 时 $e^{-\infty}=0$，权重精确为 0，彻底屏蔽。泄漏的典型症状：训练 loss 异常地低（模型"抄"了未来 token），但推理时没有未来可看、生成一塌糊涂——训练/推理行为不一致是掩码泄漏的信号。实现上用 `masked_fill(triu(ones, diagonal=1), -inf)`，依赖广播对齐 scores 的最后两维，写完务必用"改未来值、看过去输出是否变化"验证（实验 3）。

**Q6：softmax 为什么要在指数前减去最大值？掩码填 $-\infty$ 会不会产生 NaN？**
> 答：直接算 $\frac{e^{s_i}}{\sum e^{s_j}}$，$s_i$ 大时 $e^{s_i}$ 溢出成 inf。减去 $\max(s)$ 后 $\frac{e^{s_i-\max}}{\sum e^{s_j-\max}}$，最大项变成 $e^0=1$，数值稳定，且数学完全等价（max 在分子分母同时出现、可约）。掩码填 $-\infty$ 不会 NaN：因果掩码保证每行至少保留对角线 $j=i$（非 $-\infty$），所以 $\max(s)$ 是有限值，$e^{-\infty-\max}=e^{-\infty}=0$（不是 NaN）。只有当整行全是 $-\infty$ 时 $\max=-\infty$、出现 $-\infty-(-\infty)=$ NaN——因果掩码天然避免了这个情况。

### 今日检查清单

- [ ] 能默写 $\text{softmax}(QK^\top/\sqrt{d_k})V$ 并说出 5 步形状变化
- [ ] 能推出 $\text{Var}(q\cdot k)=d_k$（独立+均值 0 → 交叉项消失）
- [ ] 跑通实验 4，看到实测方差 $\approx d$、不缩放时 softmax 饱和（max 权重 $\to 1$、熵 $\to 0$）
- [ ] 能解释"不缩放 → 饱和 → 梯度消失"的链条（雅可比 $p_i(\delta_{ij}-p_j)$）
- [ ] 能解释因果掩码为什么填 $-\infty$ 而不是 0（$e^0=1$ 进入分母泄漏）
- [ ] 跑通实验 5，看到填 0 时未来列权重 0.25、填 $-\infty$ 时为 0
- [ ] 知道掩码泄漏的训练症状（loss 异常低 + 推理崩坏）
- [ ] 手写 `scaled_dot_product_attention` 与官方误差 < 1e-6（完成验收 ①）
- [ ] 跑通因果性验证（改未来值、看过去不变），知道这是面试加分项
- [ ] 能解释 softmax 的 max 减法（数值稳定 + 数学等价），知道 $-\infty$ 掩码不产生 NaN 的前提
- [ ] 知道文件里的 `MultiHeadAttention` 是 Day 3 预告，今天只确认跑通

#### 明日预告

Day 3 将把单头注意力扩展为 **Multi-Head Attention**——把 $d_{\text{model}}$ 切成 $H$ 个头，各自独立做注意力后拼接投影，并引入**位置编码**（attention 本身对顺序无感知，Day 1 实验 3 已观察到"打乱词序权重不变"）。会实现 `MultiHeadAttention`（即今天文件里冒烟测试的那个类），并对比 sinusoidal 与 RoPE 两种位置编码。今天打下的两个地基——缩放因子（每个头仍要除 $\sqrt{d_k}$，$d_k=d_{\text{model}}/H$）和因果掩码（每个头独立加）——明天原样复用。建议今晚先扫一眼 [kernels/attention_from_scratch.py](kernels/attention_from_scratch.py) 里 `MultiHeadAttention` 的 20 行实现，为明天的切分/拼接/投影做准备。

---
