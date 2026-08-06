## Day 5：MoE + EP 并行专题

### 🎯 目标

通过今天的学习，你将：

1. 理解 **MoE 结构**——Top-K 路由、load balancing loss、capacity factor
2. 掌握 **EP 并行的通信模式**——all-to-all dispatch/combine、大 EP 部署（EP32/EP144）、DeepEP/EPLB 概念
3. 理解 **MoE 推理的显存/通信权衡**——为何 decode 阶段 EP 优于 TP
4. 手写 **Top-K 路由 kernel** 模拟 + **all-to-all 通信量推导**

> 💡 **为什么重要**：DeepSeek-V3/R1、Mixtral、GPT-4 等前沿模型都用 MoE。2024-2026 JD 中 MoE/EP 是推理引擎岗的核心关键词，面试几乎必问"EP vs TP 怎么选""all-to-all 通信量怎么算"。

---

### 1. MoE 结构基础

#### 1.1 MoE 是什么？

MoE（Mixture of Experts）用"稀疏激活"替代稠密 FFN：每个 token 只激活 top-k 个专家，总参数量大但单次 forward 只用一小部分。

```
token x
   │
   ▼
gate(x) = softmax(x @ W_gate)   ← 路由网络（小 GEMM）
   │
   ▼
top-k(gate)                      ← 选 k 个专家（如 k=2）
   │
   ▼
dispatch: x 发给 k 个专家         ← all-to-all（EP 时跨节点）
   │
   ▼
expert_i(x) = FFN_i(x)           ← k 个专家并行计算
   │
   ▼
combine: 加权求和 Σ w_i · expert_i(x)  ← all-to-all 回收
   │
   ▼
output
```

| 概念 | 说明 | 典型值 |
|------|------|--------|
| num_experts | 总专家数 | 8（Mixtral）/ 64（DeepSeek-V3）/ 128 |
| top_k | 每 token 激活专家数 | 2（Mixtral）/ 6（DeepSeek-V3） |
| 稀疏比 | top_k / num_experts | 1/4 ~ 1/10 |
| gate network | 路由网络（小线性层） | hidden → num_experts |

#### 1.2 Load Balancing Loss

**问题**：若不加约束，gate 会把大部分 token 路由给少数"热门"专家，导致：
- 热门专家过载（capacity 溢出，token 被丢弃）
- 冷门专家闲置（参数浪费）

**解法**：训练时加 aux loss，鼓励专家负载均匀：

```
aux_loss = α × num_experts × Σ (f_i × P_i)
  f_i = 实际分给专家 i 的 token 比例
  P_i = gate 给专家 i 的平均概率
```

**DeepSeek 的 aux-loss-free 策略**：不加 aux loss，而是给每个专家一个偏置项 `b_i`，动态调整 `b_i` 让负载均衡（推理时无 loss 开销）。这是 2024+ MoE 面试热点。

#### 1.3 Capacity Factor

每个专家预分配的 token 容量 = `capacity_factor × (num_tokens × top_k / num_experts)`。

- capacity_factor=1.0：正好装下均匀分布的 token；超出的丢弃
- capacity_factor=1.25：留 25% 余量，减少丢弃但浪费显存

> 💡 **面试要点**：capacity factor 是显存与丢弃率的权衡。Drop token 会损失信息，但过大 capacity 浪费资源——这是 MoE 工程化的关键调参点。

---

### 2. EP 并行的通信模式

#### 2.1 EP 是什么？

Expert Parallelism：把 `num_experts` 个专家分布到 `ep_size` 个节点上，每节点 `num_experts / ep_size` 个专家。

```
Node 0: Expert 0, 1        Node 1: Expert 2, 3
Node 2: Expert 4, 5        Node 3: Expert 6, 7
```

每个 token 选 top_k 个专家，这些专家可能跨节点 → 需要 **all-to-all** 通信。

#### 2.2 all-to-all 通信量推导

**Dispatch 阶段**（输入分发）：
```
每 token 发送量 = top_k × hidden_dim × dtype_bytes
  （token 的 hidden 向量发给 k 个专家）
总量 = num_tokens × top_k × hidden_dim × dtype_bytes
```

**Combine 阶段**（输出回收）：
```
每 token 回收量 = top_k × expert_hidden × dtype_bytes
  （k 个专家的输出向量回收）
总量 = num_tokens × top_k × expert_hidden × dtype_bytes
```

**跨节点流量**（均匀分布假设）：
```
远程比例 = 1 - 1/ep_size
  （top_k 中平均有 k × (1 - 1/EP) 个专家在远程）
跨节点总量 = (dispatch + combine) × (1 - 1/ep_size)
```

**LLaMA-MoE 示例**（Mixtral 8×7B：8 专家, top_k=2, hidden=4096, EP=4, fp16）：

| 阶段 | 每 token | 1024 tokens 总量 | 跨节点(EP4, 比例 0.75) |
|------|---------|-----------------|----------------------|
| Dispatch | 2 × 4096 × 2B = 16 KB | 16 MB | 12 MB |
| Combine | 2 × 4096 × 2B = 16 KB | 16 MB | 12 MB |
| **合计** | 32 KB | 32 MB | **24 MB** |

> 💡 **面试口述**：EP all-to-all 通信量 = `2 × num_tokens × top_k × hidden × dtype × (1 - 1/EP)`。EP 越大，远程比例越高（EP=32 时 97% 流量跨节点），对网络带宽要求极高——这就是 DeepSeek 提出 DeepEP（专用 EP 通信库）的原因。

---

### 3. Coding：Top-K 路由 + EP 通信量模拟

#### 任务 1：运行 MoE 路由模拟器

```bash
python kernels/moe_routing_simulator.py
```

**预期输出**（8 专家, top_k=2, 1024 tokens, EP=4, fp16）：

```text
配置: 8 专家, top_k=2, 1024 tokens, hidden=512, EP=4

===== 1. Top-K 路由（gate softmax + top-k）=====
  前 5 个 token 的路由:
    token 0: experts=[3 6], weights=['0.256', '0.177']
    token 1: experts=[3 4], weights=['0.312', '0.188']
    ...

===== 2. 负载均衡分析 =====
  专家负载（期望/token: 256.0）:
    Expert 0:  248  ...
    Expert 7:  274  ██████████████████████████████
  最大偏差: 7.0%  (均衡)

===== 3. EP all-to-all 通信量推导 =====
  Dispatch（输入分发）:
    每 token: 2 KB (top_k × hidden × 2B)
    总量(最坏): 2.1 MB
    总量(远程,比例0.75): 1.6 MB
  Combine（输出回收）:
    每 token: 2 KB
    总量(远程): 1.6 MB
  all-to-all 总跨节点流量: 3.1 MB

===== 4. EP vs TP 选择 =====
  EP4: 专家切到 4 节点, all-to-all 流量 3.1 MB
  TP4: 权重切到 4 节点, 每层 2 次 all-reduce
  → Decode 阶段 EP 优于 TP: decode batch 小, all-reduce 开销占比大
  → Prefill 阶段 TP 可能更优: batch 大, all-reduce 摊薄, 且无 all-to-all
```

##### 观察重点

1. **Top-K 路由**：gate softmax 后选 top-2，每 token 激活 2 个专家（稀疏激活）
2. **负载均衡**：随机 gate 下偏差 ~7%（1024 token 样本小）；真实训练需 aux-loss 或 bias 调整
3. **EP 通信量**：∝ num_tokens × top_k × hidden × (1 - 1/EP)，EP4 时 75% 流量跨节点
4. **EP vs TP**：decode 选 EP（all-to-all 小），prefill 可能选 TP（all-reduce 摊薄）

#### 任务 2：扫描 EP 规模，观察远程比例变化

修改 `MoEConfig.ep_size`，扫描 2/4/8/32，观察 `1 - 1/EP` 如何趋近 1（EP32 时 96.9% 流量跨节点）。

> 思考：为什么 DeepSeek 用 EP32 甚至 EP144？（提示：专家数多（256+），单节点放不下，必须大 EP；且 decode batch 小，EP 的 all-to-all 流量可控。）

---

### 4. 为什么 decode 阶段 EP 优于 TP？

| 维度 | EP（专家并行） | TP（张量并行） |
|------|--------------|--------------|
| 切分对象 | 专家（不同 token 去不同专家） | 权重（同一 token 的 GEMM 切到多卡） |
| 通信模式 | all-to-all（dispatch + combine） | all-reduce（每层 2 次） |
| 通信量（decode, M=1） | `2 × k × hidden × (1-1/EP)`（小，~KB） | `2 × tokens × hidden`（每层；tokens=M 时消息虽小，但 all-reduce 每层都做、延迟固定，无法被 batch 摊薄） |
| 通信量（prefill, M=N） | `2 × N × k × hidden × (1-1/EP)`（大） | `2 × tokens × hidden`（每层；tokens=N 时消息大，但可被大 batch 的计算摊薄） |
| decode 适合度 | ✅ all-to-all 随 batch 小而小 | ❌ all-reduce 不随 batch 减小，M=1 时开销占比大 |
| prefill 适合度 | ❌ all-to-all 大 | ✅ all-reduce 摊薄 |

> 💡 **面试核心结论**：
> - **decode 选 EP**：batch 小（M=1~8），EP 的 all-to-all 流量小；TP 的 all-reduce 通信量虽为 `2 × tokens × hidden`（每层），但每层都做、延迟固定，batch 小时无法被摊薄，开销占比大
> - **prefill 可能选 TP**：batch 大（M=N），TP 的 all-reduce 摊薄，且无 all-to-all
> - **DeepSeek 的选择**：prefill 用 TP+EP 混合，decode 用纯 EP（EP32/EP144）

---

### 5. 大 EP 部署：DeepEP / EPLB

#### 5.1 DeepEP

DeepSeek 开源的专用 EP 通信库，针对 MoE all-to-all 优化：
- **低延迟 dispatch/combine kernel**：RDMA + NVLink 混合拓扑感知
- **节点内/节点间分层**：节点内 NVLink（高带宽），节点间 RDMA（较低带宽）
- **支持 EP32/EP144**：DeepSeek-V3 生产部署规模

#### 5.2 EPLB（Expert Parallelism Load Balancer）

DeepSeek 的负载均衡策略：
- 训练时记录每个专家的负载
- 推理时动态调整专家到节点的映射，让每节点负载均匀
- 配合 aux-loss-free 的 bias 调整

> 📖 延伸阅读：DeepSeek-V3 技术报告 §3.3（MoE 架构）、DeepEP GitHub、EPLB 文档

---

### 6. 面试要点

1. **MoE 的 Top-K 路由是怎么工作的？**（⭐⭐⭐⭐⭐ 必考）
   - gate(x) = softmax(x @ W_gate) → top-k 选 k 个专家
   - dispatch: token 发给 k 个专家 → 专家并行计算 → combine 加权求和
   - 稀疏激活：总参数大但单次 forward 只用 top_k/num_experts

2. **EP all-to-all 通信量怎么算？**（⭐⭐⭐⭐⭐ 必考）
   - dispatch: `num_tokens × top_k × hidden × dtype`
   - combine: `num_tokens × top_k × expert_hidden × dtype`
   - 跨节点比例: `1 - 1/EP`（均匀分布假设）
   - 总量: `2 × num_tokens × top_k × hidden × dtype × (1 - 1/EP)`

3. **EP vs TP 怎么选？为什么 decode 用 EP？**（⭐⭐⭐⭐⭐ 必考）
   - decode batch 小（M=1~8）：EP all-to-all 流量小；TP all-reduce 每层都做、延迟固定，batch 小时无法被摊薄，开销占比大
   - prefill batch 大（M=N）：TP all-reduce 摊薄，EP all-to-all 大
   - DeepSeek：prefill TP+EP 混合，decode 纯 EP

4. **aux-loss-free 均衡是什么？**（⭐⭐⭐⭐ 高频）
   - 传统 MoE 加 aux loss 鼓励负载均衡，但推理时 loss 无意义
   - DeepSeek 给每个专家一个 bias 项，动态调整 bias 让负载均衡
   - 推理时无 loss 开销，且均衡效果不依赖训练数据

5. **DeepSeek-V3 的 MoE 结构参数？**（⭐⭐⭐ 中频）
   - 256 专家（ routed），top_k=6，共享专家 1 个
   - hidden=7168，expert_hidden=2048（细粒度专家，单专家小）
   - EP32~EP144 部署，用 DeepEP 通信库

6. **capacity factor 是什么？**（⭐⭐⭐ 中频）
   - 每专家预分配容量 = `capacity_factor × (N × top_k / num_experts)`
   - factor=1.0 刚好均匀，超出丢弃；1.25 留余量但浪费显存
   - drop token 损失信息，是 MoE 工程化调参点

---

### 今日总结

1. **MoE 结构**：gate + top-k 路由 + 稀疏激活，总参数大但单次 forward 只用一小部分
2. **EP 并行**：专家分布到多节点，dispatch/combine 用 all-to-all
3. **通信量公式**：`2 × num_tokens × top_k × hidden × dtype × (1 - 1/EP)`
4. **EP vs TP**：decode 选 EP（all-to-all 小），prefill 可能选 TP（all-reduce 摊薄）
5. **负载均衡**：aux-loss（训练）或 aux-loss-free bias（推理，DeepSeek）
6. **大 EP 部署**：DeepEP 通信库 + EPLB 负载均衡，支持 EP32/EP144

> 📖 延伸阅读：DeepSeek-V3 技术报告、Mixtral 论文、DeepEP GitHub、GShard（MoE 并行开创性论文）

##### Ring Attention 导读（长上下文分布式注意力）

Ring Attention 是处理超长上下文（100K+ tokens）的分布式 Attention 方案——KV 跨 GPU 环形流式传输，每个 GPU 持有一部分 Q，KV 在 GPU 间传递，本地 attention 计算与通信重叠。它与 FlashAttention 的关系：Ring Attention = FlashAttention + 分布式 KV 传输，online softmax 天然支持跨 GPU 合并。

> 📖 **Ring Attention 完整讲解**（含 NCCL send/recv 通信、双流重叠、load balancing、KV buffer 显存降至 1/N）见 [`_supplementary/from_w8d6/README.md`](../_supplementary/from_w8d6/README.md)。该内容原属 Week 8 Day 6 补充，现已归入 Week 9 分布式专题。
