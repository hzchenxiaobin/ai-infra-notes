# 大模型前沿技术一周学习计划

> 设计时间：2026 年 8 月
> 核心主线：DeepSeek（V3 → R1 → V3.2 → V4）与 Kimi（K1.5 → K2 → K3）两条开源技术线
> 覆盖方向：架构、训练、推理、Agent 四大前沿
> 建议投入：每天 4–6 小时，可按节奏压缩或拉长

---

## 第 0 天晚上：学习前准备（30 分钟）

- [ ] 注册/准备好 arXiv、Hugging Face 账号；准备一个能辅助精读论文的 AI 助手
- [ ] 建立笔记本（Notion / Obsidian / 备忘录均可），每天结尾写「3 句话总结」
- [ ] 收藏两个"论文源头"：
  - DeepSeek GitHub：https://github.com/deepseek-ai
  - Moonshot AI GitHub：https://github.com/MoonshotAI（所有 Kimi 技术报告都在这里）

---

## Day 1（周一）：打地基 —— 现代大模型架构总览

**目标**：看懂任何一篇技术报告的"模型结构"章节。

**核心概念**：Transformer 回顾 → MoE（混合专家）→ 注意力变体（MHA / GQA / MLA）

**阅读材料**：

- [ ] 《DeepSeek-V3 Technical Report》（arXiv:2412.19437）第 2 章 Architecture——重点看 **MLA（多头潜在注意力）** 和 **DeepSeekMoE** 的细粒度专家 + 共享专家设计
- [ ] 辅助：搜索"MLA 图解"，把 KV cache 压缩这件事搞懂

**动手任务**：

- [ ] 用笔算一遍：671B 总参数、37B 激活的 MoE，推理时每 token 实际用多少参数？为什么 MoE 能"参数多但算得便宜"？

**检验标准**：能向朋友解释清楚"MLA 为什么省显存、MoE 为什么省算力"。

📖 详细笔记：[day1.md](day1.md)

---

## Day 2（周二）：推理模型与强化学习 —— R1 的革命

**目标**：理解 2025 年最重要的范式转变——用 RL"训出"推理能力。

**核心概念**：GRPO 算法、RLVR（可验证奖励的强化学习）、长思维链（Long CoT）、Aha moment、蒸馏

**阅读材料**：

- [ ] 《DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning》（arXiv:2501.12948）——本周最重要的一篇，精读
- [ ] 《Kimi k1.5: Scaling Reinforcement Learning with LLMs》——对照阅读，看 Moonshot 对 o1 级推理的复现路线，尤其 long2short 和 partial rollout 技巧

**关键问题**（边读边答）：

- [ ] R1-Zero 纯 RL 不放 SFT，为什么能涌现反思能力？
- [ ] GRPO 相比 PPO 省掉了什么？为什么这对大模型训练重要？
- [ ] "蒸馏小模型"和"直接对小模型做 RL"哪个效果好，为什么？

**动手任务**：

- [ ] 用 DeepSeek-R1 或任意 reasoning 模型问一道数学题，观察它的思维链，找出"自我纠错"的片段

📖 详细笔记：[day2.md](day2.md)

---

## Day 3（周三）：训练效率与基础设施 —— 万卡集群怎么炼模型

**目标**：理解前沿实验室真正的护城河——训练工程和优化器。

**核心概念**：FP8 混合精度训练、DualPipe 流水线并行、专家并行（EP）、Muon / MuonClip 优化器

**阅读材料**：

- [ ] 《Muon is Scalable for LLM Training》（Moonlight 论文）——Muon 优化器的规模化验证
- [ ] 《Kimi K2: Open Agentic Intelligence》（arXiv:2507.20534）预训练章节——重点看 **MuonClip 的 QK-clip 技术**：解决万亿参数模型训练中的 attention logits 爆炸问题，让 K2 在 15.5T token 上预训练零 loss spike
- [ ] 回看 DeepSeek-V3 报告第 3 章 Infrastructures（DualPipe、跨节点 all-to-all 通信）

**关键问题**：

- [ ] 为什么 AdamW 统治了这么多年，Muon 凭什么挑战它？（提示：矩阵正交化、token 效率）
- [ ] FP8 训练的精度风险在哪？DeepSeek 怎么解决的？

**检验标准**：能画出"数据并行 + 流水线并行 + 专家并行"三者如何组合在一万亿参数模型上。

📖 详细笔记：[day3.md](day3.md)

---

## Day 4（周四）：长上下文与高效注意力 —— 稀疏注意力时代

**目标**：理解 2025–2026 年最热的架构方向：让百万 token 上下文变得便宜。

**核心概念**：稀疏注意力、线性注意力、闪电索引器（Lightning Indexer）、KV cache 压缩

**阅读材料**（三篇对照读，本周最精彩的一天）：

- [ ] 《DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention》——**DSA（DeepSeek 稀疏注意力）**，首次把稀疏注意力做到不损失性能
- [ ] 《Kimi Linear》——Moonshot 的混合线性注意力架构，号称在各种上下文下超越全注意力
- [ ] 《MoBA: Mixture of Block Attention》——把 MoE 思想用到注意力选择上的长文本方案

**背景知识**：DeepSeek-V3.2 正是靠 DSA 大幅压低长上下文成本，才有了后来 V4 默认 1M token 上下文的商业化可行性。

**动手任务**：

- [ ] 画一张对比表：Full Attention / MLA / DSA / 线性注意力，在"计算复杂度、KV cache 大小、长文本性能"三个维度上的取舍

📖 详细笔记：[day4.md](day4.md)

---

## Day 5（周五）：Agentic 智能 —— 从聊天模型到智能体

**目标**：理解 2026 年竞争的主战场——Agent 能力是怎么"训"出来的。

**核心概念**：工具调用、agentic 数据合成管线、环境交互 RL、端到端 Agent 训练

**阅读材料**：

- [ ] 《Kimi K2》后训练章节——大规模 **agentic 数据合成管线** + 真实/合成环境联合 RL，这是 K2 在 SWE-Bench Verified 拿到 65.8、Tau2-Bench 66.1 的关键
- [ ] 《DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models》（arXiv:2512.02556）——三大突破：DSA、可扩展 RL 框架（高算力版 V3.2-Speciale 拿下 IMO 和 IOI 金牌）、大规模 agentic 任务合成管线
- [ ] 泛读：《GLM-5: from vibe coding to agentic engineering》（arXiv:2602.15763），了解同代竞品思路

**动手任务**：

- [ ] 实际体验一次 agentic 工作流：让任意一个支持工具调用的模型完成一个多步任务（比如"查资料并生成一个表格文件"），记录它在哪一步出错、为什么

📖 详细笔记：[day5.md](day5.md)

---

## Day 6（周六）：最新前沿 —— Kimi K3 与下一代架构

**目标**：站上当前（2026 年中）最前沿，理解架构创新的下一步。

**阅读材料**：

- [ ] **Kimi K3 技术报告**（github.com/MoonshotAI）——2.8T 参数、全球首个开源 3T 级模型，核心两项新架构：
  - **KDA（Kimi Delta Attention）**：一种新型注意力机制
  - **Attention Residuals（AttnRes）**：替代传统残差连接的新组件，带来一致的 scaling 收益
  - 原生视觉能力 + 100 万 token 上下文窗口
- [ ] 《Attention Residuals》独立论文——可"即插即用"替换残差连接的通用创新，值得精读
- [ ] 泛读：NVIDIA《Nemotron 3 Super》报告——工业界另一条路线：Mamba-Transformer 混合架构 + MoE

**关键问题**：

- [ ] K3 为什么敢做到 2.8T？激活参数多少？靠哪些效率技术兜底？
- [ ] Attention Residuals 和 2016 年 ResNet 的残差连接本质区别是什么？

**补充视野**（时间充裕可看）：

- [ ] DeepSeek V4 系列（2026 年 4 月发布，V4-Pro 约 1.6T 总参 / 49B 激活，默认 1M 上下文）——注意 R2 至今未发布，网上流传的"R2 参数"都是谣言，学会辨别信息真伪本身就是前沿素养
- [ ] 效率综述类论文，如记忆压缩的率失真视角（arXiv:2607.08032）

📖 详细笔记：[day6.md](day6.md)

---

## Day 7（周日）：整合输出 —— 从输入到输出

**目标**：把一周的知识固化成自己的东西。**教是最好的学。**

**任务**（三选一或全做）：

- [ ] **写一篇技术博客**（推荐）：《2026 大模型前沿技术全景：从 DeepSeek-V3.2 到 Kimi K3》，梳理四条主线——MoE 架构、RL 推理、稀疏/线性注意力、Agentic 训练
- [ ] **做一场 15 分钟分享**：给自己/朋友/同事讲，准备 10 页 slides
- [ ] **画一张大图**：一张 A3 纸画出各模型技术演进树（V2→V3→R1→V3.2→V4；K1.5→K2→K2.5→K3），标注每个节点的核心创新

**自测题**（答不出就回炉对应那天）：

- [ ] MLA 和 DSA 分别解决什么问题？能叠加吗？
- [ ] MuonClip 的 qk-clip 解决什么训练不稳定问题？
- [ ] 为什么 V3.2 要用"可扩展 RL 框架"继续 scaling post-training，而不是只卷预训练？
- [ ] K3 的 KDA 和 AttnRes 分别属于架构的哪一层创新？

📖 详细笔记：[day7.md](day7.md)

---

## 学习建议

1. **论文读法**：第一遍只读摘要 + 图 + 结论（15 分钟），第二遍精读方法章节，公式不懂就跳过先抓直觉
2. **善用 AI 辅助**：把论文 PDF 丢给 AI 助手，让它逐章拆解、出理解测试题
3. **取舍原则**：时间紧的话，Day 2（R1）和 Day 4（稀疏注意力）最不能砍——分别是 2025 和 2026 年最重要的技术事件

---

## 论文清单速查

| 论文                               | 链接              | 对应天        |
| ---------------------------------- | ----------------- | ------------- |
| DeepSeek-V3 Technical Report       | arXiv:2412.19437  | Day 1 / Day 3 |
| DeepSeek-R1                        | arXiv:2501.12948  | Day 2         |
| Kimi k1.5                          | MoonshotAI GitHub | Day 2         |
| Muon is Scalable for LLM Training  | MoonshotAI GitHub | Day 3         |
| Kimi K2: Open Agentic Intelligence | arXiv:2507.20534  | Day 3 / Day 5 |
| DeepSeek-V3.2-Exp (DSA)            | arXiv             | Day 4         |
| Kimi Linear                        | MoonshotAI GitHub | Day 4         |
| MoBA                               | MoonshotAI GitHub | Day 4         |
| DeepSeek-V3.2                      | arXiv:2512.02556  | Day 5         |
| GLM-5                              | arXiv:2602.15763  | Day 5         |
| Kimi K3 技术报告                   | MoonshotAI GitHub | Day 6         |
| Attention Residuals                | MoonshotAI GitHub | Day 6         |
| Nemotron 3 Super                   | NVIDIA            | Day 6         |