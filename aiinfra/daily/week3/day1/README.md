## Day 1：Trace Transformer 推理流程

### 🎯 目标

通过今天的学习，你将：

1. 理解 Transformer 推理的 **Prefill / Decode 两阶段**执行特征及其对 GPU 性能的根本影响
2. 掌握 `torch.profiler` 的使用方法，能独立采集并分析一次 forward 的算子时间线
3. 能列出 Transformer 单层的 **6 类算子**及其执行顺序，理解哪些是 compute-bound、哪些是 memory-bound
4. 理解 Decode 阶段 M=1 的 GEMM 为什么退化为 memory-bound

> 💡 **为什么重要**：Prefill vs Decode 是推理系统入门必考题。不理解两阶段差异，就无法理解 KV Cache、PagedAttention、Continuous Batching 等推理优化的动机。今天的内容是 Week 3 全周的地基——后续手写 Softmax/LayerNorm Kernel、Attention IO 分析、端到端 Profiling 都建立在这套认知之上。

---

### 学前导读：Transformer 推理不是一次 forward 那么简单

在训练时，我们习惯把一整个 batch 的 token 喂给模型，一次 forward 得到所有位置的输出。但**推理场景完全不同**：

- 用户输入一条 prompt（可能几千个 token），模型需要**并行处理**整条 prompt → 这叫 **Prefill**
- 然后**逐个 token 生成**回答，每次只产出 1 个 token，直到遇到 EOS → 这叫 **Decode**

这两阶段虽然跑的是同一套 Transformer 层，但**算子形状截然不同**，导致性能特征天差地别。理解这个差异，是所有推理优化的起点——记住 Prefill / Decode 这两个词，今天的全部内容都围绕它们展开。

---

### 背景知识：从零认识 Transformer

> 📖 本节为零基础铺垫。如果你已经熟悉 Transformer 结构，可以直接跳到「理论学习」。

#### 0.1 Transformer 是什么

Transformer 是 Google 在 2017 年论文《Attention is All You Need》中提出的神经网络架构。在它之前，处理序列（文本、语音）靠 RNN/LSTM——逐 token 串行处理，第 t 步必须等第 t-1 步算完，无法并行。Transformer 用 **Self-Attention** 机制一次性并行处理整个序列，让序列里的每个 token 直接"看到"其他所有 token，彻底摆脱了串行依赖。

今天所有主流大模型——GPT 系列、LLaMA、Qwen、DeepSeek、Claude——都是 Transformer 的 **decoder-only** 变体（只有解码器半边）。它们的差异在规模、数据、训练方法上，计算结构完全一样。

##### 深入理解：Decoder 是什么

Decoder（解码器）来自 2017 年原始 Transformer 论文的**编码器-解码器（Encoder-Decoder）**结构。当时的 Transformer 为机器翻译设计（英文 → 法文），分两半：

- **Encoder（编码器）**：读入完整的源句子，用**双向** Self-Attention——每个 token 可以看前后所有 token，不带 mask。任务是把整句话"理解"成一组上下文表示，不生成文字
- **Decoder（解码器）**：**逐个生成**目标句子，包含两种 attention：
 1. **带 Causal Mask 的 Self-Attention**：只看已生成的部分（不许看未来，因为它就是生成方）
 2. **Cross-Attention**：以已生成内容为 Q，去查询 Encoder 的输出（K/V 来自编码器）——"写下一个词时回头看原文"

由此分出三个架构流派：

| 流派 | 结构 | 代表模型 | 擅长 |
|------|------|---------|------|
| Encoder-only | 只有编码器，双向 attention | BERT | 理解类任务（分类、NER、检索） |
| Encoder-Decoder | 完整两半 | 原始 Transformer、T5 | 序列到序列（翻译、摘要） |
| **Decoder-only** | 只有解码器，causal mask | **GPT、LLaMA、Qwen、DeepSeek** | 生成（通用 LLM） |

**Decoder-only 就是"把编码器扔掉"**：一叠带 Causal Mask 的 Transformer Block，没有 Cross-Attention（没有编码器可查），prompt 直接喂进去，用**下一个 token 预测**统一训练和推理。它胜出是因为训练目标极简（任何文本都能当训练数据，不需要翻译对/标注）、理解和生成统一在一个模型里、架构简单好扩展——BERT 那种 encoder-only 路线在大模型时代基本被这条路线吸收取代了。

> 💡 **一句话**：Decoder 是原始 Transformer 里"负责逐词生成输出"的那一半，特征是因果掩码 + 自回归；现代 LLM 只保留这一半，所以叫 decoder-only。

##### 深入理解："并行处理整个序列"到底指什么

这句话包含三个论断，逐个拆开：

**① "串行依赖"指什么——RNN 的痛点。** RNN/LSTM 的核心计算是 `h_t = f(h_{t-1}, x_t)`——第 t 步的隐状态依赖第 t-1 步，形成**链式递归**：序列长 N 就必须严格串行执行 N 步，GPU 有几万个核心也只能干等。这是**架构层面**的串行，不是工程优化能解决的。而且信息从第 1 个 token 传到第 N 个要走 N-1 步，路径越长衰减越严重（长程依赖、梯度消失）。

**② Self-Attention 为什么能并行。** Attention 里没有任何跨位置的递归状态，token i 的输出 `out_i = Σ_j softmax(q_i·k_j/√d)·v_j` 只依赖两样东西：**所有位置的输入向量**（计算前就已知）和**固定的权重矩阵**。于是：任意一对 (i, j) 的打分 `q_i·k_j` 互相独立 → N×N 个点积可以同时算；任意一个输出 `out_i` 也互相独立 → N 个位置可以同时算。矩阵化后就是 `S = Q·Kᵀ` 一次 GEMM——正是 GPU 最擅长的规则大规模并行计算。整层前向在序列维度上没有任何循环。

**③ "直接看到其他所有 token"。** 指任意两个 token 信息交互的**路径长度是 O(1)**：位置 1 和位置 N 交换信息只需一次点积 `q₁·k_N`，一步到位，不经过任何中间 token。

| | RNN/LSTM | Self-Attention |
|--|---------|----------------|
| 任意两 token 信息交互路径 | O(N)（逐步传递） | **O(1)**（一次点积） |
| 序列维并行度 | 无（h_t 依赖 h_{t-1}） | **完全并行**（一次 GEMM） |
| 长程依赖 | 随距离衰减 | 与距离无关 |
| 序列维计算量 | O(N·d²)，但串行 | O(N²·d)，但全并行 |

> ⚠️ **两个限定条件**，避免理解偏差：
>
> 1. **"摆脱串行"只针对训练和 Prefill**。自回归 Decode 仍是一个 token 一个 token 生成的——但那是**算法层面**的串行（算下一个需要上一个 token 作为输入，信息上不可避免），不是 RNN 那种架构层面的串行。而且 Decode 每一步内部依然并行：1 个 query 对所有历史 key 的打分是同时算的
> 2. **Causal Mask 是"并行"和"自回归"能共存的关键**。训练时整个目标序列已知，可以一次性并行算所有位置的输出；mask 把上三角置 -∞，保证位置 i 只用到 ≤ i 的信息——用掩码在并行计算中**模拟**出"只能看过去"的语义。没有 mask，并行训练就会偷看答案

> 💡 **代价**：并行不是免费的——S 矩阵是 N×N，计算量和显存随序列长度平方增长。这就是 Week 2 Day 5 FlashAttention 要解决的问题。本质上 Transformer 是**用计算量换并行度**，而 GPU 的架构恰好奖励这种交换。

> 💡 **对本课的意义**：Transformer 不是"很多种不同神经网络"的堆叠，而是**同一个 Block 重复 L 次**。只要彻底搞懂一层的算子构成（今天的任务），整个模型的执行特征就清楚了。

#### 0.2 从文字到向量：Token 与 Embedding

模型不认字，只认数字。输入文本先经过两步转换：

1. **Token 化**：文本被切成 token（大致是"子词"），每个 token 对应词表里的一个 id。例：「你好，世界」→ `[1024, 306, 998]`（示意）。词表大小（vocab_size）通常 3 万~15 万
2. **Embedding 查表**：每个 token id 在 embedding 矩阵（vocab_size × d_model）里查出对应的向量

经过这两步，一句话变成形状为 `(N, d_model)` 的矩阵——N 是序列长度（token 数），d_model 是每个 token 的向量维度（512 / 4096 / 8192 等）。**这个矩阵就是后面所有计算的输入**，d_model 也常简写为 d。

##### 深入理解："Embedding 查表"到底在做什么

**① Embedding 矩阵是什么。** 它是模型的**一个可学习参数**，形状 `(vocab_size, d_model)`——词表里每个 token 占一行，每行是一个 d 维向量。比如 vocab_size=5 万、d_model=4096，FP16 下约 400 MB。训练前随机初始化，训练中和 Linear 权重一样被梯度更新。直觉上，每行向量就是该 token 的"身份证"：训练后**语义相近的 token，向量在空间中也相近**（经典的 `king - man + woman ≈ queen` 就是这么来的）。

**② "查表"是怎么查的。** 输入是 token id 序列 `[1024, 306, 998]`，查表就是**用 id 当行号，把对应行取出来**：

```
EmbeddingLookup([1024, 306, 998]) = [ E[1024],   # 第 1024 行，d 维
                                      E[306],    # 第 306 行
                                      E[998]  ]  # → 输出 (N, d_model)
```

每个 token id 独立取一行，输出从 `(N,)` 的整数序列变成 `(N, d_model)` 的浮点矩阵。

**③ 为什么叫"查表"而不叫"矩阵乘"。** 数学上它**等价于** one-hot 乘矩阵：`one_hot(1024) @ E = E[1024]`。但 one-hot 向量长度 = vocab_size（5 万），只有一个位置是 1——真做 `(N, V) @ (V, d)` 的 GEMM 是 O(N·V·d)，其中 99.998% 的乘法都在乘 0。所以工程实现是 **gather（按索引收集）**：直接按行号读内存，O(N·d) 访存，一次乘法都不用。GPU 上对应 `aten::embedding` kernel：N 个 token 分给不同线程，各把自己那行的 d 个浮点数连续读出——行内是连续内存，天然 coalesced。它是典型的 **memory-bound** 算子（零计算、纯搬运），但数据量小，通常不是瓶颈。

**④ 和本课其他内容的关联：**

- **与 LM Head 对称**：输入侧"id → 向量"是查 E 的第 i 行；输出侧 LM Head"向量 → 词表 logits"是乘 E 的转置。很多模型两者**共享同一份权重**（weight tying），省一半参数
- **不含位置信息**：查表只看 token 内容不看位置——「狗咬人」和「人咬狗」查出来是一样的行。位置信息靠位置编码（Positional Embedding / RoPE）注入
- **Decode 阶段**：每步只查 1 行（新 token 那行），开销可忽略；Decode 的瓶颈在权重读取和 KV Cache，不在 embedding

> 💡 **一句话**：Embedding 层就是一张"token id → 向量"的查找表，训练时学出来，推理时按行号读内存——数学上等价 one-hot 矩阵乘，工程上是 gather，零计算、纯访存。

#### 0.3 Self-Attention：让 token 之间交换信息

Attention 的核心思想：**每个 token 根据"谁和自己相关"，从其他 token 那里加权收集信息，更新自己的表示**。比如「苹果发布了新手机」中的「苹果」，看到「手机」后就应该更偏向公司含义。

![Self-Attention QKV 流程](../images/self_attention_qkv.svg)

每个 token 的向量 x 通过**三个独立的 Linear（就是 GEMM）**投影成三个角色：

- **Query（查询）**："我想找什么信息"
- **Key（键）**："我有什么特征，可以被别人检索到"
- **Value（值）**："我实际提供的信息内容"

##### 深入理解：QKV 投影的具体过程

**① Linear 层是什么。** 一个 Linear（全连接层）就是 `y = x·W + b`：输入向量左乘一个**可学习的权重矩阵**，再加偏置。QKV 投影用三个**各自独立**的权重矩阵 W_Q、W_K、W_V（形状都是 `d_model × d_model`），它们是训练学出来的参数，互不共享。

**② 单个 token 视角。** token i 的向量 `x_i`（1×d）分别乘三个矩阵：

```
q_i = x_i · W_Q      # 1×d 乘 d×d → 1×d
k_i = x_i · W_K
v_i = x_i · W_V
```

`q_i` 的每个元素 = `x_i` 的全部 d 个元素按 W_Q 某一列加权求和。所谓"投影"，就是把同一个向量**线性变换到三个不同的空间**——变换的方式（权重）是模型从数据中学出来的。

**③ 矩阵视角（GEMM 的由来）。** 所有 token 堆成输入矩阵 `X`（N×d），三个投影就是三次矩阵乘：

```
Q = X · W_Q    # (N, d) @ (d, d) → (N, d)
K = X · W_K    # 同上
V = X · W_V    # 同上
```

Q 的第 i 行就是 token i 的 query 向量。**这正是"Linear 就是 GEMM"的含义**——N 个 token 的投影合成一次矩阵乘，GPU 一次 kernel 算完。工程上常把三个矩阵拼成一个大矩阵 `W_QKV`（d × 3d），**一次 GEMM** 算出 Q/K/V 再切开（本文 mini 代码的 `nn.Linear(d_model, 3*d_model)` 就是这种融合写法），少一次 kernel launch、权重读取也合并。

**④ 为什么要三个不同的投影，而不是直接用 x？** 如果 Q=K=V=x，会有两个问题：

- **打分对称**：`x_i·x_j = x_j·x_i`，"i 关注 j"和"j 关注 i"被迫相同，但语言里关系往往不对称（"猫追老鼠"中猫对老鼠的关注 ≠ 老鼠对猫的）
- **自相关主导**：`x_i·x_i` 通常是最大点积，每个 token 只顾看自己

用检索系统类比：Q 是"搜索词"，K 是"索引/标签"，V 是"正文内容"——三者职责不同，自然需要三套投影，让模型分别学习"怎么查""怎么被查到""提供什么"。

**⑤ 投影之后还要分头。** 得到的 q_i（d 维）会按头切分：h 个头各拿 d_head = d/h 维（如 d=512、h=8 → 每头 64 维），每个头独立做后续 attention（见 0.4）。

> 💡 **和 Decode 性能的联系**：这三个 GEMM 的权重 W_Q/W_K/W_V 大小固定（d×d），与 N 无关。Prefill 时 N 大，计算量摊得开，是 compute-bound；Decode 时 N=1，三次 GEMM 退化成向量×矩阵（GEMV），计算量极小但三个 d×d 权重照样要完整读一遍——这就是 1.3 节"M=1 时 GEMM 变 memory-bound"的具体实例。

然后分四步计算：

1. **打分**：`S = Q × Kᵀ / √d_head`。S 是 N×N 矩阵，S[i][j] 表示 token i 对 token j 的关注度（点积越大越相关）。除以 √d_head 防止点积过大导致 softmax 饱和
2. **Causal Mask**（decoder 专用）：把 S 的上三角（j > i）置为 -∞——第 i 个 token **不许偷看未来**，只能关注自己和历史。这保证了训练和推理行为一致
3. **Softmax**：对 S 的每一行做 softmax，得到概率矩阵 P，每行和为 1
4. **加权求和**：`out = P × V`。token i 的新表示 = 所有 token 的 V 按关注度加权混合

> 💡 **和 Week 2 的联系**：步骤 1 和 4 是两个 GEMM（QKᵀ 和 PV），步骤 3 就是 Week 2 Day 5 讲的 softmax——Attention 的 IO 瓶颈就藏在这三个算子里，FlashAttention 正是为消除 S/P 写回 HBM 而生。

#### 0.4 Multi-Head Attention：多角度同时看

单个 attention 只能学到一种"关注模式"。Multi-Head Attention 把 d_model 拆成 h 个头（如 8 头 × 64 维），**每个头独立做一遍 attention**，各自学习不同的关系（有的头关注语法、有的关注指代），最后把 h 个头的输出拼接，再过一个 Output Linear 混合。

工程视角：多头不是循环 h 次，而是把 Q/K/V reshape 成 `(B, h, N, d_head)` 做 **batched GEMM**——这就是 Attention kernel 的并行维度有 B×h 这么多的原因。

#### 0.5 一层 Transformer Block 的完整组装

一层 Block = 两个子层 + 两个残差连接：

```
x → LayerNorm → Multi-Head Attention → (⊕ 残差)
  → LayerNorm → FFN                 → (⊕ 残差)
```

- **LayerNorm**：把每个 token 的 d 维向量归一化（均值 0、方差 1，再缩放平移），稳定训练。本质是 element-wise + 两次 reduce
- **FFN**：`Linear2(GELU(Linear1(x)))`，把维度 d → d_ff → d（d_ff 通常是 4×d_model）。这是模型"知识存储"的主要场所，参数量占一层的大头
- **残差连接（⊕）**：让梯度可以跳过子层直接传播，是几十上百层网络能训练起来的关键

这是 **Pre-LN** 结构（LayerNorm 在子层之前），GPT-2 之后的现代模型几乎都用它。

#### 0.6 整体架构与自回归生成

![Transformer 整体架构与自回归生成](../images/transformer_overview.svg)

把上面的 Block **堆叠 L 层**（GPT-3 是 96 层，小模型 12~32 层），再接：

1. **Final LayerNorm**
2. **LM Head**：一个巨大的 Linear（d_model × vocab_size），把最后一层输出投影到词表维度
3. **Softmax**：得到下一个 token 在词表上的概率分布
4. **采样**：按概率选出 1 个新 token（贪心 / top-k / top-p）

关键特性——**自回归（autoregressive）**：新生成的 token 会被拼回输入序列末尾，再喂给模型生成下一个，如此循环直到生成结束符（EOS）。这个"一次只产一个 token"的循环，就是学前导读里 **Decode 阶段**的由来；而第一次并行处理整条 prompt，就是 **Prefill**。

> 💡 现在回头看学前导读的 Prefill/Decode 对比，应该能明白：两阶段跑的是**同一套 Block、同一批权重**，差别只在输入矩阵的 N 维度（N = prompt 长度 vs N = 1）。

#### 0.7 关键符号速查

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `N` | 序列长度（token 数） | Prefill 数百~数万；Decode 恒为 1 |
| `d` / `d_model` | 隐藏维度（每个 token 的向量长度） | 512（本课 mini 模型）~ 8192 |
| `h` | Attention 头数 | 8 ~ 128 |
| `d_head` | 每个头的维度 = d / h | 64 或 128 |
| `d_ff` | FFN 中间维度 | 通常 4 × d_model |
| `vocab_size` | 词表大小 | 3 万 ~ 15 万 |
| `B` | Batch size | 推理时通常 1 ~ 数百 |
| `L` | 层数 | 12 ~ 96+ |

---

### 理论学习

#### 1.1 Prefill vs Decode 执行特征对比

![Prefill vs Decode 执行特征对比](../images/prefill_vs_decode.svg)

上图直观展示了两阶段的核心差异。下面用表格精确对比：

| 维度 | Prefill 阶段 | Decode 阶段 |
|------|-------------|-------------|
| **输入形状** | `(B, N_prompt, d)`，N_prompt 可达数千 | `(B, 1, d)`，每次只处理 1 个 token |
| **Attention 矩阵** | N×N 完整矩阵 | 1×N（单 query 对所有历史 key） |
| **计算量** | 大（GEMM 是 M×N×K 的大矩阵乘） | 小（GEMM 退化为向量×矩阵） |
| **瓶颈类型** | 通常是 **Compute-bound**（GEMM 主导） | 通常是 **Memory-bound**（KV Cache 读取主导） |
| **GPU 利用率** | 高（SM 充分利用，60-85%） | 低（大量 SM 空闲，等显存，10-30%） |
| **典型优化** | Tensor Core、FlashAttention | KV Cache、PagedAttention、CUDA Graph |

> 💡 **一句话总结**：Prefill 是"一大堆数据一起算"，算力是瓶颈；Decode 是"一次只算一个 token，但要翻一遍历史"，访存是瓶颈。

#### 1.2 Transformer 单层数据流

![Transformer 单层数据流](../images/transformer_dataflow.svg)

上图展示了一个标准 Transformer Block 的完整数据流。按执行顺序：

![Transformer 单层前向数据流](../../images/week3_transformer_forward_flow.svg)

**关键观察**：Transformer 单层包含 **6 类主要算子**：

1. **QKV / Output / FFN Linear**（GEMM，本课 mini 模型共 4 个）：Prefill 时 compute-bound，Decode 时 memory-bound
2. **Attention QKᵀ**（GEMM）：同上，bound 类型随 M 切换
3. **Softmax**：element-wise + reduction，永远 **memory-bound**
4. **Attention PV**（GEMM）：同 QKᵀ
5. **LayerNorm**（2 次）：element-wise + reduction，永远 **memory-bound**
6. **GELU**（FFN 中间）：element-wise，永远 **memory-bound**

一句话记忆：**GEMM 的 bound 类型随 M 切换，其余算子永远是 memory-bound**——这就是面试题 2 的标准答案。

**算子执行顺序与依赖**：

![Transformer 单层算子执行流水线](../../images/week3_transformer_layer_pipeline.svg)

> 💡 **为什么重要**：理解算子顺序是后续 kernel fusion 的基础。例如 LayerNorm + QKV GEMM 可以融合成单个 kernel，省去中间结果写回 HBM。Day 6 会详细分析 fusion 机会。

#### 1.3 为什么 Decode 是 Memory-bound：M=1 的 GEMM

![Decode 为什么是 Memory-bound](../images/decode_memory_bound.svg)

上图用 QKV GEMM 为例，直观展示了 M 从 1024 变成 1 时，arithmetic intensity 的骤降：

**Prefill 阶段 QKV GEMM**：
- 矩阵形状：`(1024, 512) × (512, 1536)`
- FLOPs = 2×1024×512×1536 ≈ 1.6G
- Bytes ≈ 1.3M（读 x + W，写 QKV）
- **AI ≈ 384 FLOP/Byte >> Ridge Point(12.6) → Compute-bound**

**Decode 阶段 QKV GEMM**：
- 矩阵形状：`(1, 512) × (512, 1536)` — M=1，退化为向量×矩阵
- FLOPs = 2×1×512×1536 ≈ 1.6M（少了 1024 倍）
- Bytes ≈ 0.8M（W 的大小没变，还是要读完整权重）
- **AI ≈ 2 FLOP/Byte << Ridge Point(12.6) → Memory-bound**

**根本原因**：M=1 时计算量与 M 成正比骤降，但权重矩阵 W 的大小不变，读取量几乎没减。AI = FLOPs/Bytes 极低，数据喂不饱计算单元。

**Decode 阶段的 Attention 更严重**——每次生成 1 个 token，都要读取**整个 KV Cache**（所有历史 token 的 K 和 V）：

```
KV Cache 大小 = 2 × N_layers × N_past × d × dtype_size
 N_past=4096, d=512, 32层, FP16 → 2×32×4096×512×2 = 512 MB
 每生成 1 个 token 要读 512 MB → 纯访存瓶颈
```

**优化方向**：
- **KV Cache**：避免重算历史 K/V（空间换时间）
- **PagedAttention**（vLLM）：减少 KV 显存碎片
- **CUDA Graph**：减少 Decode 阶段 kernel launch overhead
- **Continuous Batching**：合并多个 decode 请求提高 M

---

### 工具准备：torch.profiler 使用方法

#### 核心 API

![torch.profiler 工作流](../images/torch_profiler_workflow.svg)

```python
import torch.profiler

with torch.profiler.profile(
 activities=[
 torch.profiler.ProfilerActivity.CPU, # 采集 CPU 端调度
 torch.profiler.ProfilerActivity.CUDA, # 采集 GPU 端 kernel
 ],
) as prof:
 for _ in range(5):
 out = model(x)

# 按 CUDA 时间排序，输出 top 算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# 导出 Chrome trace（可用 chrome://tracing 打开）
prof.export_chrome_trace("transformer_trace.json")
```

#### 关键指标解读

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `Self CUDA` | 该算子自身的 GPU 执行时间（不含子算子） | 排序依据，找 top3 |
| `Self CPU` | 该算子的 CPU 调度时间 | 判断 launch overhead |
| `CPU Mem` | CPU 端内存分配 | 判断是否有频繁分配 |
| `# Calls` | 调用次数 | 判断是否过度 launch |

**分析流程**：
1. 按 `Self CUDA` 排序找 top3 算子 → 定位耗时最重的计算
2. 看 `Self CPU` vs `Self CUDA` 比值 → 若 CPU 远大于 CUDA，说明 launch overhead 高
3. 在 Chrome trace 中观察 kernel 之间的空白（gap）→ gap = CPU 调度延迟
4. 对比 Prefill 和 Decode 的算子分布差异

---

### Coding 任务：Trace Transformer Forward

#### 任务 1：实现 Mini Transformer Block（`trace_transformer.py`）

```python
# trace_transformer.py —— 最小 Transformer Block + Prefill/Decode profiling
# 运行命令: python trace_transformer.py
# 依赖: pip install torch

import torch
import torch.nn as nn
import math

class MiniAttention(nn.Module):
 def __init__(self, d_model=512, n_heads=8):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_head = d_model // n_heads
 self.qkv = nn.Linear(d_model, 3 * d_model)
 self.out = nn.Linear(d_model, d_model)

 def forward(self, x):
 B, N, _ = x.shape
 qkv = self.qkv(x) # GEMM: B*N*d x d*3d
 qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
 qkv = qkv.permute(2, 0, 3, 1, 4) # 3, B, n_heads, N, d_head
 q, k, v = qkv[0], qkv[1], qkv[2]
 scale = self.d_head ** -0.5
 attn = torch.matmul(q, k.transpose(-2, -1)) * scale # GEMM: Q x K^T -> N x N
 attn = torch.softmax(attn, dim=-1) # softmax（memory-bound）
 out = torch.matmul(attn, v) # GEMM: attn x V -> N x d_head
 out = out.transpose(1, 2).reshape(B, N, self.d_model)
 return self.out(out) # GEMM: Output Linear

class TransformerBlock(nn.Module):
 def __init__(self, d_model=512, n_heads=8, d_ff=2048):
 super().__init__()
 self.attn = MiniAttention(d_model, n_heads)
 self.norm1 = nn.LayerNorm(d_model)
 self.norm2 = nn.LayerNorm(d_model)
 self.ffn = nn.Sequential(
 nn.Linear(d_model, d_ff),
 nn.GELU(),
 nn.Linear(d_ff, d_model),
 )

 def forward(self, x):
 x = x + self.attn(self.norm1(x)) # Attention + residual
 x = x + self.ffn(self.norm2(x)) # FFN + residual
 return x

def profile_phase(model, x, name, n_iter=5):
 """对一个阶段做 profiling 并输出 top 算子"""
 # warmup
 for _ in range(2):
 _ = model(x)
 torch.cuda.synchronize()

 with torch.profiler.profile(
 activities=[
 torch.profiler.ProfilerActivity.CPU,
 torch.profiler.ProfilerActivity.CUDA,
 ],
 ) as prof:
 for _ in range(n_iter):
 _ = model(x)
 torch.cuda.synchronize()

 print(f"\n===== {name} Phase (shape={tuple(x.shape)}) =====")
 print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
 prof.export_chrome_trace(f"trace_{name}.json")

def main():
 torch.manual_seed(42)
 d_model, n_heads = 512, 8
 model = TransformerBlock(d_model, n_heads).cuda().half()

 # Prefill: 处理长 prompt（N=1024）
 x_prefill = torch.randn(1, 1024, d_model, device="cuda", dtype=torch.float16)
 profile_phase(model, x_prefill, "prefill", n_iter=5)

 # Decode: 逐 token 生成（N=1）
 x_decode = torch.randn(1, 1, d_model, device="cuda", dtype=torch.float16)
 profile_phase(model, x_decode, "decode", n_iter=10)

 print("\n===== 观察要点 =====")
 print("1. Prefill 阶段：gemm 类算子 CUDA 时间占比最高（compute-bound）")
 print("2. Decode 阶段：总时间远小于 prefill，但单 token 时间占比不合理地高（memory-bound）")
 print("3. 对比 softmax/layernorm 在两阶段的绝对时间——decode 下它们可能占更大比例")
 print("4. 打开 trace_prefill.json（chrome://tracing）观察 kernel 顺序与间隙")

if __name__ == "__main__":
 main()
```

#### 任务 2：运行并采集 Prefill / Decode trace

```bash
# 运行（需 CUDA GPU）
python trace_transformer.py

# 打开 Chrome trace 可视化
# 1. 浏览器访问 chrome://tracing
# 2. Load trace_prefill.json 和 trace_decode.json
# 3. 观察 GPU kernel 的时间线排列
```

#### 任务 3：分析两阶段的算子差异

```
===== Prefill Phase (shape=(1, 1024, 512)) =====
--------------------------------- ... ---------------------------------
Name Self CUDA Calls ...
aten::_scaled_dot_product... xxx us 5
aten::mm xxx us 20 ← QKV/Out/FFN GEMM
aten::layer_norm xxx us 10
aten::softmax xxx us 5
...

===== Decode Phase (shape=(1, 1, 512)) =====
--------------------------------- ... ---------------------------------
Name Self CUDA Calls ...
aten::mm xxx us 20 ← GEMM 但矩阵极小
aten::layer_norm xxx us 10
aten::softmax xxx us 5
...
```

**分析任务清单**：

1. 找出 Prefill 阶段 CUDA 时间 top3 算子（预期是 mm/linear 类 GEMM）
2. 找出 Decode 阶段 CUDA 时间 top3 算子（预期 GEMM 占比下降，layernorm/softmax 占比上升）
3. 计算 Prefill 单 token 时间 vs Decode 单 token 时间（Prefill 快得多，因为并行度高）
4. 在 Chrome trace 中观察 kernel 之间的间隙（gap = launch overhead）

**预期发现**：
- **Prefill**：GEMM（`aten::mm`）占 CUDA 时间 60%+，是绝对主导 → compute-bound
- **Decode**：GEMM 矩阵极小（M=1），时间占比下降；softmax/layernorm 相对占比上升；kernel 间 gap 更明显（launch overhead 占比增大）→ memory-bound

#### 任务 4：LeetGPU 在线题目 —— Matrix Multiplication

**题目链接**：<https://leetgpu.com/challenges/matrix-multiplication>

**题目概述**：给定行主序 FP32 矩阵 `A`（`M×N`）、`B`（`N×K`），计算 `C = A @ B`：`C[i][j] = Σ_{k=0}^{N-1} A[i][k] * B[k][j]`，输出形状 `(M, K)`。无 α/β、无 FP16——纯 CUDA Core 的 tiled matmul。

**与今日知识的关联**：Matrix Multiplication 是 GEMM 的最纯粹形态——今天 profiling 揭示了 Prefill 阶段 `aten::mm` 占 CUDA 时间 60%+（compute-bound），本题就是手写这个主角：naive 版每 thread 独立算一个 `C` 元素，`A`/`B` 被重复读，算术强度仅 1/8 FLOP/Byte（memory-bound）；shared memory tiling 靠数据复用把 AI 拉高，转为 compute-bound。它是"用 ncu 判定 bound 类型"的最佳练习对象——同一份代码加 tiling 前后 `DRAM%` 与 `SM%` 的对比，就是今天 Roofline 分析的实战。

> 💡 完整题解见 [Matrix Multiplication 题解](../../../../aiinfra/topics/cuda/medium/gemm/matrix-multiplication.md)。

#### 任务 5：LeetCode 面试题 —— 盛最多水的容器

**题目链接**：[11. 盛最多水的容器](https://hzchenxiaobin.github.io/leetcode/problems/盛最多水的容器.html)

**题目概述**：给定 `n` 个非负整数 `a1, a2, ..., an`，每个代表坐标中的一个点 `(i, ai)`。找出两条线，使得它们与 x 轴构成的容器能容纳最多的水。

**与今日知识的关联**：盛最多水容器的**双指针贪心**与今日 profiling 中"缩小搜索范围"的思路同构——双指针从两端向中间逼近，每次移动较短的一边（因为移动较长的一边不可能得到更大面积），就像 profiling 中逐步缩小瓶颈范围。两者都是"通过排除不可能的候选来高效定位最优解"。

> 💡 完整题解见 [盛最多水的容器题解](https://hzchenxiaobin.github.io/leetcode/problems/盛最多水的容器.html)。

---

### 扩展实验

**实验1（基础）**：修改 `d_model=1024, n_heads=16`，重新 profile，观察 GEMM 时间变化。
> 提示：GEMM 计算量与 d_model 平方相关，layernorm/softmax 与 d_model 线性相关。

**实验2（进阶）**：用 `nsys profile -o transformer_trace python trace_transformer.py` 采集系统级时间线，在 Nsight Systems GUI 中对比 Prefill 和 Decode 的 SM 利用率。
> 提示：Decode 阶段 SM 利用率会很低（绿色 bar 很短），这就是 memory-bound 的直观表现。

**实验3（综合）**：在 TransformerBlock 中加一个 `forward_with_fusion` 方法，用 `torch.compile(model, mode="reduce-overhead")` 自动做 kernel fusion，对比 fused vs unfused 的 kernel 数量。
> 提示：`torch.compile` 会把 LayerNorm + GEMM 等相邻算子融合，kernel 数减少 30-50%。

---

### 面试要点

**面试题1**：Transformer 推理的 Prefill 和 Decode 阶段有什么区别？为什么 Decode 通常是 memory-bound？（⭐⭐⭐ 高频）

**参考答案要点**：
- **Prefill**：输入是 `(B, N_prompt, d)`，N_prompt 可达数千。所有 GEMM 是大矩阵乘，计算量大，GPU SM 充分利用 → **Compute-bound**
- **Decode**：输入是 `(B, 1, d)`，每次只生成 1 个 token。GEMM 退化为向量×矩阵（M=1），计算量极小，但每次都要读取整个 KV Cache（N 个历史 token） → **Memory-bound**
- **根本原因**：Decode 阶段计算强度（FLOP/Byte）极低。M=1 的 GEMM 每读 1 行 K/V 只做 d 次乘加，arithmetic intensity ≈ 2 FLOP/Byte，远低于 Ridge Point（~12.6）
- **优化方向**：KV Cache（避免重算 K/V）、PagedAttention（减少 KV 显存碎片）、CUDA Graph（减少 launch overhead）、Continuous Batching（合并多个 decode 请求提高 M）

**面试题2**：Transformer 单层包含哪些算子？哪些是 compute-bound，哪些是 memory-bound？（⭐⭐⭐ 高频）

**参考答案要点**：

| 算子 | 类型（Prefill） | 类型（Decode） | 原因 |
|------|----------------|----------------|------|
| QKV/Out/FFN Linear (GEMM) | Compute-bound | Memory-bound | Prefill M 大；Decode M=1 |
| Attention QK^T (GEMM) | Compute-bound | Memory-bound | 同上 |
| Attention Softmax | Memory-bound | Memory-bound | element-wise + reduction |
| Attention PV (GEMM) | Compute-bound | Memory-bound | 同上 |
| LayerNorm | Memory-bound | Memory-bound | element-wise + reduction |
| GELU | Memory-bound | Memory-bound | element-wise |

> 关键洞察：GEMM 在 Prefill 和 Decode 之间会切换 bound 类型，而 Softmax/LayerNorm/GELU 永远是 memory-bound（与 M 无关）。

---

### 验证 Checklist

- [ ] 能解释 Prefill 和 Decode 的输入形状差异及其对性能的影响
- [ ] 能列出 Transformer 单层的 6 类算子及其执行顺序
- [ ] torch.profiler 代码运行成功，输出 Prefill/Decode 的算子时间表
- [ ] 找出 Prefill 阶段 CUDA 时间 top3 算子
- [ ] 能解释为什么 Decode 阶段 GEMM 变成 memory-bound（M=1 导致计算强度低）
- [ ] 能用 chrome://tracing 打开 trace 文件并观察 kernel 间隙
