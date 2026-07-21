# Day 1（周一）：MoE 算法总览与朴素实现

> **本周定位**：本专题是 [CUTLASS 专题](../cutlass/README.md)（算子视角，Day 7 Group GEMM）之后的**系统视角**——把 Grouped GEMM、Top-K 路由、all-to-all 通信、负载均衡组装成一个完整的 MoE 层。本周目标是用 Triton 拼出一个 Top-2 路由的 MoE FFN 层，性能达到 Megatron-LM 参考实现 70%+，产出 ncu 性能报告。
> **前置要求**：已完成 CUTLASS Day 3（3.x GEMM）与 Day 7（Group GEMM），掌握 GEMM Tiling 与 CollectiveBuilder；建议读过 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) 对 DeepSeekMoE 有基本认知
> **今日目标**：理解 MoE 的算法动机（稀疏激活降训练/推理成本）、前向数据流（门控 → Top-K → 分派 → 专家 → combine）、路由算法演进（Top-1/2/K + 负载均衡）、DeepSeekMoE 三创新（细粒度专家 + 共享专家 + 设受限路由），用朴素 PyTorch 实现一个 MoE FFN 并观察其性能瓶颈，建立本周源码地图
> **时间投入**：2.5h（早间 1.5h 精读算法 + 晚间 1h 跑朴素实现）
> **面试考察度**：⭐⭐⭐ 了解级，能说清 MoE 是什么、为什么稀疏激活能省算力、DeepSeekMoE 的三个创新

---

## 本日在本周知识图谱中的位置

```
Day 1          Day 2           Day 3            Day 4           Day 5          Day 6        Day 7
 总览      →   Gating +    →   Grouped       →   Expert      →  vLLM         → 完整       →  调优
 算法动机      Top-K 融合      GEMM              Parallelism    fused_moe       Triton       ncu
 数据流        Triton         Triton/CUTLASS    all-to-all     源码精读        MoE FFN      报告
 路由算法      kernel          cuBLAS 对照      Megatron 通信                  性能对比
 朴素实现
  ↑
  你在这里（地基：不理解 MoE 数据流与瓶颈，后面 6 天都不知道在优化什么）
```

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| MoE 前向数据流图（门控 → Top-K → 分派 → 专家 → combine） | ① 能画出 MoE 前向数据流（完成验收 ①） |
| 路由算法演进表（Top-1/2/K + 负载均衡） | ④ 能解释 EP 的 all-to-all 时序（前置：先理解路由） |
| 朴素 PyTorch MoE 实现 + 性能瓶颈观察 | ② Gating+Top-K kernel 5x+ PyTorch（Day 2 的基准线） |
| 本周源码地图（vLLM/Megatron/Triton 三条线） | ⑤ ncu 定位 MoE 通信/计算占比（Day 7 的前置） |

---

### 学习任务 1：MoE 是什么——稀疏激活的经济学（30 分钟）

#### 从稠密 FFN 说起

标准 Transformer 的每个 block 是 `Attention + FFN`，FFN 占约 2/3 参数量与 FLOPs。FFN 的计算是两次线性变换 + 激活：

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x), \quad W_1 \in \mathbb{R}^{d_{\text{ff}} \times d},\ W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$$

- 每个 token 都要算完整的 $2 \cdot d \cdot d_{\text{ff}}$ FLOPs
- 模型越大（$d_{\text{ff}}$ 越大），每 token 成本线性增长

**痛点**：想要更强能力 → 加大模型 → 每 token 成本飙升 → 训练推理都贵。

#### MoE 的核心思想：稀疏激活

MoE（Mixture-of-Experts）把一个大 FFN 换成 $N$ 个小 FFN（"专家"），每个 token 只激活其中 $K$ 个（$K \ll N$）：

$$\text{MoE}(x) = \sum_{i \in \text{TopK}(g(x), K)} g_i(x) \cdot \text{FFN}_i(x)$$

- $g(x) = \text{Softmax}(W_g x)$：门控（router）输出 $N$ 个亲和力分数
- $\text{TopK}$：选亲和力最高的 $K$ 个专家
- 总参数 $= N \cdot (2 \cdot d \cdot d_{\text{ff}})$，但每 token 只算 $K \cdot (2 \cdot d \cdot d_{\text{ff}})$ FLOPs

> 💡 **一句话总结**：MoE 把"参数量"与"每 token 计算量"解耦——总参数可以做到几百 B，但每 token 只激活几 B，**用稀疏激活换经济性**。

#### 稀疏激活的经济学算账

| 模型 | 总参数 | 激活参数/ token | 训练成本 | 备注 |
|------|--------|-----------------|---------|------|
| GPT-3（稠密） | 175B | 175B | 基准 | 每 token 算全部参数 |
| Switch Transformer | 1.6T | 1.6B（Top-1） | ~1/4 | 2048 专家，每 token 选 1 |
| GShard | 600B | 96B（Top-2） | ~1/3 | 每 token 选 2 |
| Mixtral 8x7B | 47B | 13B（Top-2） | ~1/3 | 8 专家选 2，开源标杆 |
| **DeepSeek-V2** | **236B** | **21B（Top-6）** | **−42.5% vs 67B 稠密** | 160 路由 + 2 共享 |
| DeepSeek-V3 | 671B | 37B（Top-8） | 进一步降低 | FP8 + Mega MoE |

> ⚠️ **注意"激活参数"不等于"显存占用"**：MoE 的全部 $N$ 个专家权重都要存在显存里，只是每 token 只用 $K$ 个。所以 MoE 是"省算力不省显存"——这也是为什么 MoE 训练要用 Expert Parallelism 把不同专家放到不同 GPU（Day 4 讲）。

#### MoE 的工程挑战

稀疏激活带来三个新问题，本周后面 6 天都在解决它们：

| 挑战 | 说明 | 对应 Day |
|------|------|---------|
| **动态路由** | 每 token 选不同专家，GEMM 的 shape 动态变化 | Day 2-3（Gating + Grouped GEMM） |
| **负载不均** | 热门专家过载、冷门专家闲置，batch 内算力浪费 | Day 1（今天）+ Day 2 |
| **通信开销** | Expert Parallelism 的 all-to-all 通信可能吃掉稀疏省的算力 | Day 4-5 |

### 学习任务 2：MoE 前向数据流与路由算法演进（45 分钟）

这是 Day 1 的**核心精读**内容——理解数据流才能画图（验收 ①），理解路由演进才能读懂 DeepSeekMoE。

#### 前向数据流（验收 ① 的核心）

```
输入 x: [num_tokens, d]
    │
    ▼
┌──────────────┐
│   Gating     │  g = softmax(x @ W_g)   → [num_tokens, N]
│  (门控网络)   │  W_g: [d, N]
└──────────────┘
    │
    ▼
┌──────────────┐
│   Top-K      │  选每 token 亲和力最高的 K 个专家
│   + 重归一化  │  g_topk, idx_topk = topk(g, K)
│              │  g_topk = g_topk / g_topk.sum()   ← 对 K 个分数重归一化
└──────────────┘
    │
    ▼
┌──────────────┐
│  Dispatch    │  按 idx_topk 把 token 散到对应专家
│  (分派)      │  → 每 expert 收到一组 token（动态数量）
└──────────────┘
    │
    ▼
┌──────────────┐
│  Expert FFN  │  对每 expert: y_e = FFN_e(x_e)
│  (专家计算)   │  N 个专家并行（或 Grouped GEMM 单 kernel）
└──────────────┘
    │
    ▼
┌──────────────┐
│   Combine    │  按 idx_topk 把专家输出散回原位置
│  (合并)      │  y = Σ_{k=0}^{K-1} g_topk[:,k] * y_scattered[:,k]
│              │  → [num_tokens, d]
└──────────────┘
    │
    ▼
输出 y: [num_tokens, d]
```

> 💡 **关键洞察**：MoE 的"分派 → 专家 → 合并"本质是 **scatter-gather 模式**——token 按 expert 索引散开、各算各的、再按索引合并。这与稠密 GEMM 的"全员算同一矩阵"根本不同，是后面 Grouped GEMM（Day 3）与 EP all-to-all（Day 4）的算法基础。

#### 路由算法演进

| 论文 | 年份 | 路由 | 负载均衡 | 特点 |
|------|------|------|---------|------|
| **Shazeer 2017**（Sparsely-Gated MoE） | 2017 | Top-K（K=2-4） | 软负载均衡损失（$P \cdot f$） | 奠基：门控 + Top-K + 负载损失 |
| **GShard** | 2020 | Top-2 | 容量因子 + auxiliary loss | 专家并行 + all-to-all；容量因子 $C = \frac{T \cdot K}{N} \cdot \text{factor}$ |
| **Switch Transformer** | 2021 | Top-1 | 简化 auxiliary loss | 极简路由（每 token 1 专家）；训练稳定性研究 |
| **DeepSeekMoE** | 2024 | Top-K（V2: K=6） | 三级损失 + token 丢弃 | 细粒度专家 + 共享专家 + 设受限路由 |
| **Mixtral 8x7B** | 2024 | Top-2 | 简单 auxiliary loss | 开源 MoE 标杆，8 专家选 2 |

#### Top-1 vs Top-2 vs Top-K 的 tradeoff

| 维度 | Top-1（Switch） | Top-2（GShard/Mixtral） | Top-6（DeepSeek-V2） |
|------|----------------|------------------------|---------------------|
| 每 token 计算量 | 最低（1×FFN） | 中（2×FFN） | 高（6×FFN） |
| 路由稳定性 | 差（1 专家决定） | 中 | 好（6 个平均） |
| 负载均衡难度 | 高（更容易崩塌） | 中 | 低（多专家分摊） |
| 专家组合空间 | $N$ | $\binom{N}{2}$ | $\binom{N}{6}$ |
| 典型 $N$ | 128-2048 | 8-64 | 64-256 |

> 💡 **DeepSeekMoE 为什么用 Top-6**：细粒度切分（把 1 个大专家切成 $m$ 个小专家）后，单个专家容量小，必须激活更多专家才能达到同等激活参数量。DeepSeek-V2 把 8 个粗粒度专家切成 160 个细粒度专家，激活 6 个——组合空间 $\binom{160}{6} \approx 2.1 \times 10^{10}$，远大于 $\binom{8}{2} = 28$，专家可以更专精。

#### Top-K + 重归一化

读 GShard 的路由公式：

$$g_{i,t} = \begin{cases} \frac{s_{i,t}}{\sum_{j \in \text{TopK}} s_{j,t}} & \text{if } i \in \text{TopK}(s_{\cdot,t}, K) \\ 0 & \text{otherwise}\end{cases}$$

- $s_{i,t} = \text{Softmax}_i(u_t^\top e_i)$：token $t$ 对专家 $i$ 的亲和力（softmax 后）
- Top-K 后对选中的 $K$ 个分数**重归一化**（除以和）——让 $K$ 个权重加起来等于 1
- 重归一化的原因：原始 softmax 是对 $N$ 个专家归一化，Top-K 后只剩 $K$ 个，权重和不等于 1，需要重新归一化

### 学习任务 3：负载均衡与容量因子（45 分钟）

#### 路由崩塌问题

如果没有负载均衡，MoE 训练会出现**路由崩塌**（routing collapse）——所有 token 都涌向少数"明星专家"，其他专家闲置：

- 明星专家过载：batch 内 token 数 >> 算力预算，被迫丢弃 token
- 冷门专家闲置：收不到 token，梯度小，永远学不好
- 恶性循环：明星专家越训越强，冷门专家越来越弱

#### 负载均衡损失（auxiliary loss）

Shazeer 2017 提出的经典形式：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i$：专家 $i$ 收到的 token **占比**（硬统计，$\frac{\text{count}_i}{T}$）
- $P_i$：专家 $i$ 的**平均亲和力**（软统计，$\frac{1}{T}\sum_t s_{i,t}$）
- $N$：专家数（归一化因子）

**为什么用 $f_i \cdot P_i$ 而不是单独 $f_i$？**
- $f_i$ 不可导（Top-K 选择是离散的），$P_i$ 可导（softmax 输出）
- 乘积 $f_i \cdot P_i$ 让"负载高 + 亲和力高"的专家受惩罚——这正是崩塌的根源
- 当所有专家均匀时 $f_i = P_i = 1/N$，$\mathcal{L}_{\text{aux}} = N \cdot N \cdot \frac{1}{N^2} = 1$（最小值）

#### 容量因子（Capacity Factor）

GShard 引入容量因子防止专家过载：

$$\text{capacity}_e = \left\lfloor \frac{T \cdot K}{N} \cdot \text{cf} \right\rfloor$$

- $T$：batch 内 token 数
- $K$：每 token 激活专家数
- $N$：专家总数
- $\frac{T \cdot K}{N}$：**理想均匀分布下**每专家应收的 token 数
- $\text{cf}$：容量因子（典型 1.0-1.5），留 buffer 应对不均

| cf 值 | 效果 |
|-------|------|
| cf = 1.0 | 严格预算，超出丢弃；DeepSeek-V2 用此值 |
| cf = 1.25-1.5 | 留 buffer，少丢 token；GShard/Mixtral 常用 |
| cf = ∞ | 不丢弃，但负载可能严重不均 |

#### Token 丢弃（Token Dropping）

训练时，如果某 expert 收到的 token 数超过 capacity，按亲和力从低到高丢弃：

```python
# 伪代码
for expert in experts:
    if len(tokens[expert]) > capacity[expert]:
        # 按亲和力排序，丢弃最低的
        sorted_idx = argsort(affinity[expert], descending=True)
        keep_idx = sorted_idx[:capacity[expert]]
        tokens[expert] = tokens[expert][keep_idx]
```

> ⚠️ **训练丢、推理不丢**：训练时丢弃是为了算力均衡（让所有 expert 的 GEMM shape 相同），但评估/推理时不能丢——否则结果不确定。DeepSeek-V2 还保留约 10% 的训练序列永不丢弃，保持训练-推理一致性。

#### DeepSeek-V2 的三级负载均衡

读 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) §5.3，DeepSeekMoE 把单一 auxiliary loss 拆成三级：

| 级别 | 损失 | 目标 | $\alpha$ |
|------|------|------|---------|
| **专家级** $\mathcal{L}_{\text{ExpBal}}$ | 防单个专家崩塌 | $f_i$ 归一化因子 $N_r / K_r T$ | 0.003 |
| **设备级** $\mathcal{L}_{\text{DevBal}}$ | 每设备算力均衡 | 按专家组 $\mathcal{E}_i$ 聚合 | 0.05 |
| **通信级** $\mathcal{L}_{\text{CommBal}}$ | 每设备**接收**均衡 | $f_i''$ 以 $D/MT$ 归一 | 0.02 |

> 💡 **为什么需要通信级**：设备受限路由（$M=3$）限定了每 token 的**发送**目标 ≤3 台设备，但**接收**端可能不均——某些设备被很多 token 选中。$\mathcal{L}_{\text{CommBal}}$ 专门约束接收均衡，让 all-to-all 通信的接收侧也均衡。

### 学习任务 4：DeepSeekMoE 的三个创新（45 分钟）

读 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) §5.3，DeepSeekMoE 相比 GShard/Switch 的三个核心创新。

#### 创新 1：细粒度专家切分

传统 MoE（GShard）用少量粗粒度专家（如 8 个），DeepSeekMoE 把每个粗粒度专家切成 $m$ 个细粒度专家：

$$N_{\text{fine}} = m \times N_{\text{coarse}}$$

- DeepSeek-V2：$N_r = 160$ 路由专家（相当于把 8 个粗粒度各切成 20 份）
- 每个细粒度专家的中间维度更小（1536 vs 传统 4096+）
- 激活 $K_r = 6$ 个——组合空间 $\binom{160}{6} \approx 2.1 \times 10^{10}$

**为什么细粒度更好**：
- 同样激活参数量下，组合空间指数级增大
- 专家可以更专精（每个小专家学一个细分知识）
- 消融显示细粒度 + Top-6 比粗粒度 + Top-2 性能更好

#### 创新 2：共享专家（Shared Expert）

除了 $N_r$ 个路由专家，还有 $N_s$ 个**共享专家**——所有 token 必过，无门控：

$$\text{MoE}(x) = \sum_{s=1}^{N_s} \text{FFN}_s^{\text{shared}}(x) + \sum_{i \in \text{TopK}} g_i \cdot \text{FFN}_i^{\text{routed}}(x)$$

- DeepSeek-V2：$N_s = 2$ 共享专家
- 共享专家捕捉**通用知识**（如语法、常见模式）
- 路由专家只需学**专精知识**，避免冗余

> 💡 **共享专家的工程意义**：共享专家是稠密计算（所有 token 都算），可以与 all-to-all 通信 overlap——通信时算共享专家，通信完成算路由专家。这是 DeepSeek-V2 训练加速的关键 overlap 之一。

#### 创新 3：设备受限路由（Device-Limited Routing）

专家并行把 $N$ 个专家铺在 $D$ 台设备上，每 token 激活 $K$ 个。如果 $K$ 个专家分布在不同设备，all-to-all 通信涉及最多 $K$ 台设备。细粒度专家（$K=6$）让通信更频繁。

**设受限路由**：先选亲和力最高的 $M$ 台**设备**，再在其中做 Top-K：

```python
# 伪代码
device_scores = scores.reshape(num_tokens, num_devices, experts_per_device).sum(-1)  # [T, D]
top_devices = device_scores.topk(M, dim=-1).indices                                  # [T, M]
# 在 top_devices 范围内做 Top-K
masked_scores = scores.clone()
masked_scores[~in_top_devices] = -inf
topk_scores, topk_idx = masked_scores.topk(K, dim=-1)
```

- DeepSeek-V2：$D=8$ 设备，$M=3$——每 token 至多发往 3 台设备
- 通信量从 $O(K)$ 台降到 $O(M)$ 台（$M < K$）
- 论文实验：$M \geq 3$ 时性能与无限制 Top-K 基本持平——**通信限制几乎不花精度**

> ⚠️ **设受限路由的 tradeoff**：$M$ 太小（如 1）会限制专家选择空间，掉性能；$M$ 太大（如 $D$）退化为无限制，通信无节省。$M=3$ 是 DeepSeek-V2 的甜点。

#### DeepSeekMoE 前向数据流（完整版）

```
输入 x: [T, d]
    │
    ├──────────────────────┐
    │                      │
    ▼                      ▼
┌──────────┐         ┌──────────────┐
│ 共享专家  │         │   Gating     │
│ (N_s 个)  │         │  + 设受限路由 │
│ 无门控    │         │  + Top-K     │
└──────────┘         └──────────────┘
    │                      │
    │                      ▼
    │              ┌──────────────┐
    │              │   Dispatch   │  all-to-all（≤M 台设备）
    │              └──────────────┘
    │                      │
    │                      ▼
    │              ┌──────────────┐
    │              │  路由专家 FFN │  N_r 个，每 token 激活 K_r
    │              └──────────────┘
    │                      │
    │                      ▼
    │              ┌──────────────┐
    │              │   Combine    │  all-to-all 反向
    │              └──────────────┘
    │                      │
    └──────────┬───────────┘
               │
               ▼
    y = shared_out + routed_out
               │
               ▼
输出 y: [T, d]
```

### 学习任务 5：朴素 PyTorch MoE 实现（45 分钟）

这是 Day 1 的**动手环节**——用纯 PyTorch 实现一个 MoE FFN，观察它的性能瓶颈，为 Day 2-3 的 Triton 优化建立基准线。

#### 朴素实现（`kernels/naive_moe.py`）

```python
# naive_moe.py —— 朴素 PyTorch MoE FFN
import torch
import torch.nn.functional as F


class NaiveMoEFFN(torch.nn.Module):
    """Top-K 路由的 MoE FFN，纯 PyTorch 实现。

    用于 Day 1 建立 baseline，Day 2-3 会用 Triton 重写 Gating 与 Grouped GEMM。
    """
    def __init__(self, d: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()
        self.d = d
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = torch.nn.Linear(d, num_experts, bias=False)

        # 专家权重（堆叠成 [num_experts, d_ff, d] 与 [num_experts, d, d_ff]）
        self.w1 = torch.nn.Parameter(torch.randn(num_experts, d_ff, d) * 0.02)
        self.w2 = torch.nn.Parameter(torch.randn(num_experts, d, d_ff) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, d]
        num_tokens = x.size(0)

        # Step 1: Gating
        logits = self.gate(x)                          # [T, N]
        scores = F.softmax(logits, dim=-1)             # [T, N]

        # Step 2: Top-K
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)  # [T, K]
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # 重归一化

        # Step 3: Dispatch + Expert FFN（朴素：逐专家循环）
        output = torch.zeros_like(x)                   # [T, d]
        for expert in range(self.num_experts):
            # 找出哪些 token 选了这个专家
            mask = (topk_idx == expert).any(dim=-1)    # [T]
            if not mask.any():
                continue

            token_idx = mask.nonzero(as_tuple=True)[0]  # [num_selected]
            x_e = x[token_idx]                          # [num_selected, d]

            # 专家 FFN: SiLU(x @ W1.T) @ W2.T
            h = F.silu(x_e @ self.w1[expert].T)         # [num_selected, d_ff]
            y_e = h @ self.w2[expert].T                 # [num_selected, d]

            # Step 4: Combine（按 topk_score 加权累加）
            for k in range(self.top_k):
                k_mask = (topk_idx[token_idx, k] == expert)
                if k_mask.any():
                    weight = topk_scores[token_idx[k_mask], k].unsqueeze(-1)  # [n, 1]
                    output[token_idx[k_mask]] += weight * y_e[k_mask]

        return output
```

#### 性能观察

```python
# bench_naive_moe.py —— 性能对比
import torch, time
from naive_moe import NaiveMoEFFN

def bench(fn, num_warmups=5, num_tests=10):
    for _ in range(num_warmups): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_tests): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_tests / 1e3  # 秒

# 配置：模拟 DeepSeek-V2 的 MoE 层（缩小版）
d, d_ff, num_experts, top_k = 5120, 1536, 8, 2
num_tokens = 4096

moe = NaiveMoEFFN(d, d_ff, num_experts, top_k).cuda()
x = torch.randn(num_tokens, d, device='cuda')

# 稠密 FFN 对照
dense_w1 = torch.randn(d_ff, d, device='cuda')
dense_w2 = torch.randn(d, d_ff, device='cuda')
def dense_ffn():
    h = F.silu(x @ dense_w1.T)
    return h @ dense_w2.T

t_moe = bench(lambda: moe(x))
t_dense = bench(dense_ffn)
print(f'Naive MoE:   {t_moe*1e3:.1f} ms')
print(f'Dense FFN:   {t_dense*1e3:.1f} ms')
print(f'MoE/Dense:   {t_moe/t_dense:.2f}x')
```

```text
# 预期输出（A100，4096 tokens, 8 experts, Top-2）
Naive MoE:   3.2 ms
Dense FFN:   0.4 ms
MoE/Dense:   8.0x   ← 朴素 MoE 比稠密 FFN 慢 8 倍！
```

#### 朴素实现的 3 个瓶颈

| 瓶颈 | 原因 | 浪费 |
|------|------|------|
| **逐专家循环** | `for expert in range(N)` 串行调用 8 个小 GEMM | launch 开销 + GPU 利用率低 |
| **动态 shape** | 每 expert 的 `num_selected` 动态变化，无法用 cuBLAS 高效处理 | 小 GEMM 算不满 Tensor Core |
| **scatter-gather 开销** | `x[token_idx]` 与 `output[token_idx] +=` 是非连续访问 | 内存带宽浪费 |

> 💡 **关键洞察**：朴素 MoE 比稠密 FFN 慢 8 倍，但理论上 MoE 的计算量只是稠密的 $K/N \times \text{d\_ff\_ratio}$。瓶颈不在算力，而在**动态路由导致的串行 + 小 GEMM**。Day 3 的 Grouped GEMM 就是把这个串行循环换成单 kernel 的批量 GEMM。

#### cuBLAS 逐专家对照

```python
# 逐专家 cuBLAS（比朴素 PyTorch 快，但仍是串行）
def expert_cublas():
    output = torch.zeros_like(x)
    for expert in range(num_experts):
        mask = (topk_idx == expert).any(dim=-1)
        if not mask.any(): continue
        x_e = x[mask]
        h = F.silu(x_e @ w1[expert].T)
        y_e = h @ w2[expert].T
        # ... combine
    return output

t_cublas = bench(expert_cublas)
print(f'cuBLAS per-expert: {t_cublas*1e3:.1f} ms')
print(f'Grouped GEMM target: {t_cublas*0.9*1e3:.1f} ms  (90% of cuBLAS, 验收 ③)')
```

```text
cuBLAS per-expert: 1.8 ms
Grouped GEMM target: 1.6 ms  (90% of cuBLAS, 验收 ③)
```

- 逐专家 cuBLAS 比朴素 PyTorch 快（1.8ms vs 3.2ms），因为 cuBLAS 优化好
- 但仍是串行 8 个 GEMM，有 launch 开销
- **Day 3 的 Grouped GEMM 目标**：单 kernel 达到逐专家 cuBLAS 的 90%（验收 ③）

### 学习任务 6：MoE 系统地图与本周路径（30 分钟）

#### 三条工程实现路线

| 路线 | 代表 | 特点 | 本周对应 |
|------|------|------|---------|
| **Triton 手写** | 本专题 Day 2-3-6 | 用 Triton 写 Gating + Grouped GEMM + 完整 MoE | 学习用 |
| **vLLM fused_moe** | vLLM 生产级推理 | Triton 融合 kernel，支持 FP8 + 持久化调度 | Day 5 精读 |
| **Megatron MoE** | 训练框架 | PyTorch + NCCL all-to-all + cuBLAS GEMM | Day 4 通信 |
| **DeepGEMM** | DeepSeek 自研 | FP8/FP4 GEMM + Mega MoE 单 kernel 融合 | [DeepGEMM 专题](../deepgemm/README.md) |

#### 本周源码地图

```
本周会读/写的代码：
├── kernels/                        # 自己写
│   ├── naive_moe.py                # Day 1: 朴素 PyTorch MoE（今天）
│   ├── triton_gating.py            # Day 2: Triton Gating + Top-K 融合
│   ├── triton_grouped_gemm.py      # Day 3: Triton Grouped GEMM
│   ├── ep_demo.py                  # Day 4: 2 卡 Expert Parallelism demo
│   └── triton_moe.py               # Day 6: 完整 Triton MoE FFN
├── notes/                          # 读别人的
│   ├── vllm_fused_moe.md           # Day 5: vLLM fused_moe 精读
│   ├── megatron_moe.md             # Day 4: Megatron MoE 通信
│   └── deepseek_moe.md             # Day 1/7: DeepSeekMoE 算法笔记
└── benchmark/
    ├── compare_moe.py              # Day 6: MoE 性能对比脚本
    └── report.md                   # Day 7: 性能报告
```

#### MoE 系统瓶颈与本周对应

| 瓶颈 | 量化 | 优化手段 | Day |
|------|------|---------|-----|
| **Gating + Top-K** | 占 MoE 前向 5-10%，但 PyTorch 实现慢 5x+ | Triton 融合 kernel | Day 2 |
| **Grouped GEMM** | 占 MoE 前向 60-70%（主算力） | Triton/CUTLASS 批量 GEMM | Day 3 |
| **all-to-all 通信** | EP 训练时占 20-40% | 通信/计算 overlap + 设受限路由 | Day 4 |
| **负载不均** | 导致 last-wave 浪费 | 容量因子 + 均衡损失 | Day 1（今天） |
| **dispatch/combine scatter** | 占 10-15%，非连续访问 | 融合进 GEMM kernel | Day 5-6 |
| **小 batch 利用率低** | decode 时 expert 收到 1-2 token | 持久化调度 + tile 化 | Day 5-6 |

> 💡 **本周路径总结**：Day 1 建立算法与瓶颈认知 → Day 2-3 优化计算（Gating + Grouped GEMM）→ Day 4 优化通信（EP all-to-all）→ Day 5 读生产级实现（vLLM）→ Day 6 组装完整 MoE → Day 7 ncu 调优报告。每天解决一个瓶颈，最后拼出完整 MoE。

### 面试题积累（本周目标 10-12 道，今日 3 道）

本周逐步积累面试题，今日从"了解级"开始：

**Q1：MoE 为什么能降低训练成本？激活参数和总参数有什么区别？**
> 答：MoE 把一个大 FFN 换成 $N$ 个小专家 FFN，每 token 只激活 $K$ 个（$K \ll N$）。总参数 $= N \times \text{FFN}$（全部存在显存），激活参数 $= K \times \text{FFN}$（每 token 实际计算）。训练成本与激活参数成正比，所以 $N=160, K=6$ 的 DeepSeek-V2 总参数 236B 但每 token 只算 21B，训练成本比 67B 稠密模型低 42.5%。注意"激活参数不等于显存占用"——全部 $N$ 个专家权重都要存。

**Q2：Top-1、Top-2、Top-K 路由各有什么优缺点？DeepSeekMoE 为什么用 Top-6？**
> 答：Top-1（Switch）计算最省但路由稳定性差、负载均衡难度高；Top-2（GShard/Mixtral）是折中；Top-K（DeepSeek-V2 K=6）路由稳定性好、负载均衡容易，但每 token 计算量大。DeepSeekMoE 用 Top-6 是因为细粒度切分（160 个小专家）后单个专家容量小，必须激活更多才能达到同等激活参数；同时组合空间 $\binom{160}{6} \approx 2 \times 10^{10}$ 远大于 $\binom{8}{2} = 28$，专家可以更专精。Top-6 的代价是每 token 通信目标多，靠设受限路由（$M=3$）压制。

**Q3：DeepSeekMoE 的三个创新是什么？共享专家有什么作用？**
> 答：① **细粒度专家切分**——把粗粒度大专家切成 $m$ 份小专家，组合空间指数级增大，专家更专精；② **共享专家**——$N_s$ 个无门控专家，所有 token 必过，捕捉通用知识，避免路由专家间的知识冗余；共享专家是稠密计算，可与 all-to-all 通信 overlap；③ **设备受限路由**——先选亲和力最高的 $M$ 台设备再做 Top-K，把每 token 通信目标从 $O(K)$ 降到 $O(M)$（DeepSeek-V2 $M=3, K=6$），$M \geq 3$ 时性能几乎无损。三级负载均衡损失（专家级/设备级/通信级）配合 token 丢弃保证训练均衡。

### 今日检查清单

- [ ] 能说出 MoE 的核心思想（稀疏激活，参数量与每 token 计算量解耦）
- [ ] 能区分总参数与激活参数，知道 MoE"省算力不省显存"
- [ ] 能画出 MoE 前向数据流（输入 → 门控 → Top-K → 分派 → 专家 → 合并）
- [ ] 能说出 Top-1/Top-2/Top-K 的 tradeoff（计算量 vs 稳定性 vs 组合空间）
- [ ] 理解 Top-K 后重归一化的原因（让 K 个权重和为 1）
- [ ] 能写出负载均衡损失 $\mathcal{L}_{\text{aux}} = \alpha N \sum f_i P_i$ 并解释为什么用 $f_i \cdot P_i$
- [ ] 能解释路由崩塌（routing collapse）的成因与恶性循环
- [ ] 能说出容量因子的公式 $C = \frac{T \cdot K}{N} \cdot \text{cf}$ 与 cf 取值范围
- [ ] 理解训练丢 token、推理不丢的原因
- [ ] 能列出 DeepSeekMoE 的三个创新（细粒度 + 共享 + 设受限）
- [ ] 能解释共享专家的作用（通用知识 + 通信 overlap）
- [ ] 能解释设受限路由的 tradeoff（$M$ 太小掉性能，太大通信无节省）
- [ ] 能说出 DeepSeek-V2 的三级负载均衡（专家级/设备级/通信级）
- [ ] 跑通朴素 PyTorch MoE，观察到它比稠密 FFN 慢 5-8 倍
- [ ] 能列出朴素 MoE 的 3 个瓶颈（逐专家循环 / 动态 shape / scatter-gather）
- [ ] 浏览了 vLLM `fused_moe.py` 与 Megatron `transformer/moe.py` 的位置，标记了 Day 4-5 精读文件

#### 明日预告

Day 2 将深入 **Gating + Top-K 融合 kernel**——用 Triton 写一个融合的门控 + Top-K + 重归一化 kernel，达到纯 PyTorch 的 5x+（验收 ②）。今天理解了 MoE 的数据流与朴素实现的瓶颈，明天要针对第一个瓶颈（Gating/Top-K 的 PyTorch 多 kernel 开销）动手优化。建议今晚先扫一眼 Triton 的 `tl.argmax` / `tl.topk` 原语，以及 vLLM `fused_moe.py` 里的 `fused_experts` 函数签名，为明天做准备。

---
