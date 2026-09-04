# FlashAttention IO 复杂度推导：从分块计算到 Θ(N²d²/M)

> **导读**：面试高频题——"FlashAttention 和标准 Attention 的 FLOPs 完全相同（~2N²d），加速全部来自 HBM 访问量的降低。请从分块计算的角度推导 Θ(N²d²/M) 这个 IO 界，说明什么时候退化成 O(Nd)，并估算 d=64、N=4096 时 IO 降低多少倍"。本文按 IO 模型 → 标准实现的 IO → 分块方案 → 逐步计数 → 下界最优性 → 退化条件 → 数值估算的顺序完整走一遍，数字与 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)（Theorem 2/3）一一对应。

---

## 一句话结论

**Θ(N²d²/M) 的来源是一句话：K、V 只需从 HBM 读一遍，但 Q 和 O 要为 K/V 的每个列块重新流式过一遍；列块数是 T_c = Θ(Nd/M)，每遍代价 Θ(Nd)，相乘即 Θ(N²d²/M)**。当 SRAM 大到能同时装下全部 K、V（M ≳ 2Nd）时，列块数退化为 1，复杂度落到 Θ(Nd) 线性。d=64、N=4096、M≈100KB（fp16）时，Θ 层面比值 = M/d² ≈ 12.5×；计入常数因子后约 4~9×（取决于 baseline 是否含 mask/dropout 的额外读写），与论文实测的 "up to 9× fewer HBM accesses" 一致。

---

## 一、问题设定与 IO 计算模型

### 1.1 两级存储模型

IO 复杂度分析（I/O complexity, Aggarwal & Vitter 1988）把 GPU 抽象成两级存储：

- **HBM**：容量大（几十 GB）、带宽相对低（A100 ~2TB/s），是所有数据的"家"
- **SRAM**：每个 SM 的共享内存，容量 M 很小、带宽极高（~19TB/s 量级），所有计算只能在 SRAM 里的数据上进行

**只统计 HBM ↔ SRAM 之间搬运的数据量（元素个数或字节数），计算本身免费**。这是合理的：A100 上 FP16 算力 ~312 TFLOPS，而 HBM 带宽换算成 FP16 元素只有 ~1 万亿个/秒——attention 这种每元素只做 O(1)~O(d) 次运算的算子是**带宽瓶颈（memory-bound）**，搬运量直接决定墙钟时间。

### 1.2 符号与任务

| 符号 | 含义 | 大小 |
|:---:|------|:---:|
| N | 序列长度 | — |
| d | head 维度 | — |
| M | SRAM 容量（按元素计） | A100 每 SM 约 100~164KB，H100 228KB |
| Q, K, V | 输入 | 各 N×d |
| S = QKᵀ | 注意力分数 | **N×N ← 问题根源** |
| P = softmax(S) | 注意力权重 | N×N |
| O = PV | 输出 | N×d |

任务：精确计算 O = softmax(QKᵀ)V。FLOPs 两种实现都是 Θ(N²d)，差别只在 HBM 搬运量。

---

## 二、标准 Attention 的 IO：为什么是 Θ(Nd + N²)

标准实现把 S 和 P **物化（materialize）到 HBM**，逐算子执行：

| 步骤 | 读 | 写 | HBM 访问量 |
|------|----|----|:---:|
| S = QKᵀ | Q (Nd), K (Nd) | S (N²) | N² + 2Nd |
| P = softmax(S) | S (N²) | P (N²) | 2N² |
| O = PV | P (N²), V (Nd) | O (Nd) | N² + 2Nd |
| **合计** | | | **4N² + 4Nd = Θ(N² + Nd)** |

根因很清楚：**N×N 的中间矩阵远大于 M**（N=4096 时 S 是 16.8M 个元素 ≈ 33.6MB fp16，而 SRAM 只有 ~0.1MB），根本放不进 SRAM，每算一步都得把中间结果写回 HBM、下一步再读回来。训练时若还带 attention mask 和 dropout，S/P 还要再多过两遍，变成 ~8N²。

IO 复杂度 Θ(Nd + N²)（论文 Theorem 2 前半）就是这么来的——**N² 项完全由中间矩阵的物化贡献**。

---

## 三、FlashAttention 的分块方案

FlashAttention 的核心是：**用 online softmax 把 softmax 增量式化，从而永远不需要完整的 S/P；再按 SRAM 容量把 Q/K/V/O 切成块，所有中间结果留在片上**。

### 3.1 块大小怎么定

SRAM 里需要同时驻留 4 个块（K_j、V_j、Q_i、O_i），所以块高取（论文 Algorithm 1）：

$$B_c = \left\lceil \frac{M}{4d} \right\rceil \quad (K,V \text{ 的列块行数}), \qquad B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil,\; d\right) \quad (Q,O \text{ 的行块行数})$$

为什么是 4：片上同时放 K_j、V_j（各 B_c×d）、Q_i、O_i（各 B_r×d）共 4 块 → 每块预算 M/4。附带一个漂亮的性质：分块分数矩阵 S_ij 的大小 B_r×B_c ≤ (M/4d)·d = M/4，**也放得下**——所有中间量都在片上。

### 3.2 循环结构

```
外层 for j = 1..T_c,  T_c = ⌈N / B_c⌉        # K/V 的列块
    从 HBM 读入 K_j, V_j                       # 每轮 2·B_c·d
    内层 for i = 1..T_r,  T_r = ⌈N / B_r⌉    # Q/O 的行块
        从 HBM 读入 Q_i, O_i（及统计量 l_i, m_i）
        片上算 S_ij = Q_i K_jᵀ                 # 不落盘
        online softmax：更新 running max m、归一化因子 l，得到 P̃_ij
        片上更新 O_i ← rescale(O_i) + P̃_ij V_j
        把 O_i, l_i, m_i 写回 HBM
最后一轮结束后对 O 做一次统一归一化（diag(l)^{-1}）
```

online softmax（Milakov & Gimelshein 2018 的 safe softmax 增量版）保证：**每行只需要 O(1) 个额外统计量（m, l）在 HBM 和 SRAM 之间往返，而不需要 N 长的分数行**。正确性与三遍 softmax 逐 bit 等价（数值稳定版），所以它是 exact attention，不是近似。

---

## 四、逐步计数：Θ(N²d²/M) 怎么来的

按上面的循环结构数 HBM 搬运量（以元素计）：

1. **K、V 的读取**：外层每轮读 K_j、V_j（2·B_c·d），T_c 轮恰好把 K、V 各读一遍 → 合计 **2Nd**，只发生一次
2. **Q 的读取**：内层每轮读全部 Q 块（T_r·B_r·d = Nd），而这样的内层扫描要执行 **T_c 次** → 合计 T_c·Nd
3. **O 的读写**：每轮内层对每个 O_i 一读一写 → 2Nd 每轮 → 合计 2·T_c·Nd
4. **统计量 l、m**：每行 O(1) 个，合计 O(N) 每轮，可忽略

总搬运量：

$$\underbrace{2Nd}_{K,V \text{ 一遍}} + \underbrace{3 \cdot T_c \cdot Nd}_{Q/O \text{ 每列块一遍}} = \Theta(T_c \cdot Nd)$$

代入列块数 $T_c = \lceil N/B_c \rceil = \Theta\!\left(\dfrac{N}{M/d}\right) = \Theta\!\left(\dfrac{Nd}{M}\right)$：

$$\boxed{\;\text{HBM 访问量} = \Theta\!\left(\frac{Nd}{M} \cdot Nd\right) = \Theta\!\left(\frac{N^2d^2}{M}\right)\;}$$

加上任何算法都免不了的"读 QKV、写 O"这个 Θ(Nd) 基底，完整形式是 **Θ(N²d²/M + Nd)**（题目中的写法）；当 M ≤ Nd 时第一项占主导，通常简写为 Θ(N²d²/M)。

**直觉记忆法**：FlashAttention 相当于对 attention 做了一层"K/V 维度的循环分块（loop tiling）"。K、V 流一遍就够，但 Q 和 O 要**为每个 K/V 列块重新从 HBM 流一遍**；SRAM 每 M/d 行 K/V 切出一个列块，共 Nd/M 个列块，每遍代价 Nd，乘起来就是 N²d²/M。**M 翻倍 → 列块数减半 → IO 减半**，这就是 IO 与 SRAM 成反比的原因。

与标准实现对比（Θ 层面）：

$$\frac{\Theta(N^2)}{\Theta(N^2d^2/M)} = \frac{M}{d^2}$$

**IO 降低的渐近倍数 = M/d²**——片上 SRAM 能装下多少个"完整 d 维向量"的平方量级。

---

## 五、为什么这是下界（最优性）

论文 Theorem 3 证明：**在 d ≤ M ≤ Nd 的整个区间内，任何精确计算 attention 的算法都需要 Ω(N²d²/M) 次 HBM 访问**——FlashAttention 在这个区间内渐近最优，没有算法能在 SRAM 大小不变时做得更好。

下界的直觉（通信复杂性论证的轮廓）：

- 输出第 i 行 O_i 依赖**全部** N 个 key/value（softmax 把整行耦合在一起），所以"完成"一行输出必须看过完整的 K、V（Θ(Nd) 数据）
- SRAM 里同时最多驻留 Θ(M/d) 行的 Q/O 状态（每行 d 个数），即**每流完一遍 K/V，最多只能"推进" Θ(M/d) 行输出**
- 要推进全部 N 行，需要 N/(M/d) = Nd/M 轮，每轮搬运 Ω(Nd) → 合计 Ω(N²d²/M)

**面试加分点**：论文的下界证明在"SRAM 能装下全部数据（M = Θ(Nd)）"的端点处是用反证法收的——若某个算法在该区间只需要 o(N²d²/M)，代入 M = Θ(Nd) 会得到 o(Nd)，但它连输入 QKV（Nd 大小）都读不完，矛盾。

再进一步：后续工作 [The I/O Complexity of Attention (Saha & Ye, 2024)](https://arxiv.org/abs/2402.07443) 把界收得更紧——精确界是 Θ(min(N²d/√M, N²d²/M))，交叉点在 **M = d²**：当 M ≥ d²（大 cache 区间，实际情况 M≈100KB ≫ d²=8KB 都在此区间）FlashAttention 最优；当 M < d² 时存在比 FlashAttention 更优的算法。也就是说 **FlashAttention 是"大 SRAM 区间"的最优解**，这个限定条件值得在面试中点出。

---

## 六、退化条件：什么时候变成 O(Nd) 线性

把 M 往上推，列块数 T_c = Nd/M 往下掉，当 **K、V 能整个放进 SRAM** 时：

$$2Nd \le M \;\Rightarrow\; T_c = 1 \;\Rightarrow\; \text{IO} = \underbrace{3Nd}_{\text{读一遍 } Q,K,V} + \underbrace{Nd}_{\text{写一遍 } O} = \Theta(Nd)$$

此时 Q 按行块流式过一遍（每块读入一次、写出 O_i 一次），K、V 全程驻留片上，**每个元素恰好进出 HBM 一次——这是任何算法的绝对下界，IO 与序列长度成线性**。

与公式衔接：把 M = Nd 代入 N²d²/M 恰好得 Nd——两个区域在边界连续过渡，不是突变。

现实检查（fp16，M ≈ 100KB = 51,200 元素，d = 64）：线性区间要求 N ≤ M/(2d) = 400。N = 4096 远在分块区（T_c ≈ 21 个列块），这就是为什么 **FA 的收益随 N 增长**——N 越大，标准实现的 N² 物化代价越惨，而 FA 只是列块数线性增加。

工程上这个退化条件解释了两个现象：

- **短序列 attention 不慢**：N 小时 M ≥ 2Nd 天然成立（或 T_c 只有 1~2），标准实现和 FA 的 IO 都是 Θ(Nd)，没什么可省的
- **滑窗/局部 attention 的 IO 视角**：窗口 w 固定时等价于每个 Q 块只看 w 个 key，有效"Nd"变小，IO 同样回到 Θ(Nd·w/d) 级别的线性

---

## 七、数值估算：d = 64，N = 4096

取 A100 典型的可用片上 SRAM **M ≈ 100KB**，fp16（2 字节/元素）→ M = 51,200 元素。

### 7.1 Θ 层面（按题目的复杂度式直接比）

$$\frac{\text{标准}}{\text{FA}} = \frac{N^2 + Nd}{N^2d^2/M + Nd} \approx \frac{M}{d^2} = \frac{51{,}200}{64^2} = \frac{51{,}200}{4{,}096} \approx \mathbf{12.5\times}$$

具体字节数：

| | 元素数 | fp16 字节数 |
|---|:---:|:---:|
| 标准 attention | 4N² + 4Nd ≈ 68M（S 写读、P 写读各一遍） | ≈ 136 MB |
| FlashAttention（Θ 级） | N²d²/M ≈ 1.34M | ≈ 2.7 MB |

### 7.2 计入常数因子（更贴近真实）

Θ 隐藏了两侧的常数，摊开数一遍：

- **FA 侧**：B_c = M/(4d) = 200 行 → T_c = ⌈4096/200⌉ = 21 个列块；每轮 Q/O 搬运 3Nd → 总 ≈ 3·21·Nd + 2Nd ≈ 17M 元素 ≈ **34 MB**（常数来自"4 块共享 M"和"O 每轮一读一写"）
- **标准侧（无 mask/dropout）**：4N² ≈ 134 MB → **比值 ≈ 4×**
- **标准侧（含 mask + dropout，训练常见配置）**：S/P 再多 4 遍 ≈ 8N² ≈ 268 MB → **比值 ≈ 8~9×**

这与论文 Figure 2 实测的 "up to **9×** fewer HBM accesses" 正好对上——论文的 baseline 是带 mask/dropout 的训练态标准实现。

### 7.3 结论与注意点

- **量级答案：约一个数量级（10× 上下）**。Θ 框架下 12.5×，仔细数常数 4~9×，都落在"一个数量级"上
- HBM 访问量降 ~10× **不等于**墙钟时间降 10×：FA 的 kernel 还有额外开销（重计算、rescale），实测墙钟加速 ~2~4×——因为它把算子从"纯带宽瓶颈"推回到接近算力瓶颈，带宽利用率做不到 100%
- M 的取值敏感：若按 A100 满配 164KB 算，Θ 比值是 20×；H100 228KB 则是 28×。面试时说清"M 按 100KB、fp16"的假设即可
- 附带红利（面试可提）：FA 的**显存占用**也从 Θ(N²) 降到 Θ(N)——4096 长度下 S/P 不再物化，省下的 67MB×层数×头数才是长序列训练能开起来的关键

---

## 小结

1. **模型**：两级存储，只数 HBM 搬运量；attention 是带宽瓶颈算子，IO 决定速度
2. **标准实现**：物化 S、P 两个 N×N 矩阵 → Θ(N² + Nd)
3. **FA 推导**：K/V 读一遍（2Nd）+ Q/O 为每个 K/V 列块流一遍（T_c·3Nd），T_c = Θ(Nd/M) → **Θ(N²d²/M + Nd)**；一句话——"Q/O 流的遍数 = SRAM 装不下的 K/V 列块数"
4. **最优性**：d ≤ M ≤ Nd 区间内这是下界（论文 Theorem 3）；更精细的分析（Saha & Ye 2024）给出交叉点 M = d²，FA 在大 cache 区间最优
5. **退化**：M ≥ 2Nd（K、V 驻留片上）→ T_c = 1 → **Θ(Nd) 线性**；fp16、M=100KB、d=64 时对应 N ≲ 400
6. **估算**：d=64、N=4096、M=100KB → M/d² ≈ 12.5×（Θ 级）；计入常数与 baseline 配置后 4~9×，与论文实测一致——**量级上约省一个数量级的 HBM 流量**

## 参考

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Theorem 2/3 与 Algorithm 1 的块大小定义；[ar5iv 版](https://ar5iv.labs.arxiv.org/html/2205.14135)可直接在线看推导）
- Saha & Ye, [The I/O Complexity of Attention, or How Optimal is FlashAttention?](https://arxiv.org/abs/2402.07443)（紧界 Θ(min(N²d/√M, N²d²/M))，交叉点 M=d²）
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（online softmax 出处）
- 相关笔记：同目录 [swizzle_mechanism.md](swizzle_mechanism.md)、[ldmatrix_bank_conflict.md](ldmatrix_bank_conflict.md)
