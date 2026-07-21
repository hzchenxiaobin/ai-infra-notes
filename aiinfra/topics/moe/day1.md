# Day 1（周一）：MoE 总览与算法原理

> **今日目标**：理解 MoE 的定位与设计动机，掌握稀疏门控的数学形式，能区分 Token-Choice / Expert-Choice / 共享专家三类路由范式
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"MoE 是什么、为什么稀疏激活能省算力"必问

---

### 学习任务 1：MoE 是什么——从 Dense FFN 到稀疏专家（45 分钟）

#### 阅读内容
- **论文**：[Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) §1-3（Shazeer 2017）
- **论文**：[GShard](https://arxiv.org/abs/2006.16668) §2-3（Lepikhin 2021）
- **复习**：[DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) §5.3 DeepSeekMoE

#### 核心要点

MoE（Mixture-of-Experts）把 Transformer 的 FFN 层替换为"门控 + 多专家"结构：每个 token 只激活少数专家，**总参数量与每 token 计算量解耦**。

$$y(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x), \qquad g(x) = \mathrm{TopK}(\mathrm{softmax}(W_g \, x))$$

| 维度 | Dense FFN | 稠密 MoE | 稀疏 MoE（Top-K） |
|------|-----------|----------|-------------------|
| 每 token 激活参数 | 全部 | 全部 | $K/N$（如 Mixtral 2/8 = 25%） |
| 总参数量 | 固定 | 固定 | 可扩到百亿/千亿 |
| 计算量 | $\propto$ 参数量 | $\propto$ 参数量 | $\propto$ 激活参数（省算力） |
| 显存 | $\propto$ 参数量 | $\propto$ 参数量 | $\propto$ 总参数（显存仍贵） |
| 代表 | GPT-3 175B | — | Mixtral 8x7B、DeepSeek-V3 671B/37B |

> 💡 **一句话总结**：MoE 的核心交易是"用显存换算力"——把参数量扩到千亿但每 token 只算几十亿，训练/推理 FLOPs 大幅下降，代价是显存与通信。这也是为什么 MoE 几乎总是和专家并行、KV Cache 优化、量化绑定出现。

#### 三种路由范式

| 范式 | 决策者 | 代表 | 特点 |
|------|--------|------|------|
| **Token-Choice** | token 选 expert（top-k） | GShard / Switch / Mixtral / DeepSeek | 主流，可能负载不均 |
| **Expert-Choice** | expert 选 token（top-k 反向） | EC-MoE（Zhou 2022） | 负载天然均衡，但破坏因果性 |
| **Shared Expert** | 部分 expert 必过 + 其余 top-k | DeepSeekMoE | 缓解专家冗余，装通用知识 |

### 学习任务 2：负载均衡——为什么 MoE 训练容易崩（45 分钟）

#### 路由崩塌现象

不加约束时，门控会迅速收敛到"少数专家垄断所有 token"——其他专家收不到梯度，退化为死专家。

#### Auxiliary Loss（GShard / Switch）

$$L_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i = \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}[\text{token } t \text{ 选了专家 } i]$：每个专家收到的 token 比例（不可导的硬统计）
- $P_i = \frac{1}{T}\sum_{t=1}^{T} \text{softmax}(W_g x_t)_i$：每个专家的平均门控概率（可导）
- 乘积形式让不可导的 $f_i$ 通过可导的 $P_i$ 反传梯度

> ⚠️ **面试高频坑**：$f_i$ 和 $P_i$ 必须分开计算——$f_i$ 用 `argmax` 硬统计（stop gradient），$P_i$ 用 softmax 软概率。直接用 $\sum P_i^2$ 会让所有 token 挤向概率最高的专家，反而加剧不均衡。

#### 容量因子（Capacity Factor）

每个专家有一个容量上限 $C = \frac{T}{N} \times \text{cap\_factor}$，超出的 token 被丢弃。GShard 用 cap=1.25，Switch 用 cap=1.0-1.5。

| cap_factor | 训练效率 | 负载均衡 | 推理 |
|------------|----------|----------|------|
| < 1.0 | 高（省算力） | 差（丢 token 多） | 一般不用 |
| 1.0-1.5 | 中 | 训练常用 | 推理不丢 |
| > 2.0 | 低 | 几乎不丢 | 不推荐 |

### 学习任务 3：环境搭建与参考实现（30 分钟）

```bash
# 克隆 vLLM（本周 Day 5 精读其 fused_moe）
git clone https://github.com/vllm-project/vllm.git
# 关键文件：vllm/model_executor/layers/fused_moe/fused_moe.py

# 克隆 Megatron-LM（本周 Day 4 精读其 MoE 通信）
git clone https://github.com/NVIDIA/Megatron-LM.git
# 关键文件：megatron/core/transformer/moe/moe_layer.py
```

#### PyTorch 朴素 MoE 参考实现

```python
# naive_moe.py —— 朴素 MoE FFN（仅前向，无负载均衡）
import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts_w1 = nn.Parameter(torch.randn(num_experts, d_ff, d_model))
        self.experts_w2 = nn.Parameter(torch.randn(num_experts, d_model, d_ff))
        self.top_k = top_k

    def forward(self, x):                       # x: [T, d_model]
        logits = self.gate(x)                   # [T, num_experts]
        scores = F.softmax(logits, dim=-1)
        topk_val, topk_idx = scores.topk(self.top_k, dim=-1)   # [T, k]
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for k in range(self.top_k):
                e = topk_idx[t, k].item()
                h = F.gelu(x[t] @ self.experts_w1[e].T)
                out[t] += topk_val[t, k] * (h @ self.experts_w2[e].T)
        return out
```

> ⚠️ **注意**：上面的双重 for 循环只是教学用，实际不可用（无并行、无向量化）。本周 Day 2-3 会逐步把它改成 Triton kernel。

### 今日检查清单

- [ ] 能写出 MoE 前向公式 $y = \sum g_i(x) E_i(x)$ 并解释 Top-K 门控
- [ ] 能说出 Dense FFN / 稠密 MoE / 稀疏 MoE 三者的计算量与显存差异
- [ ] 能解释 auxiliary loss 为什么用 $f_i \cdot P_i$ 乘积形式（不可导 + 可导）
- [ ] 能说出容量因子的作用与典型取值
- [ ] 跑通 `naive_moe.py`，确认前向输出 shape 正确

---

