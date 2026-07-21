---
name: paper-reading
description: 用于在 aiinfra/paper 下做论文精读并沉淀笔记。规定了目录组织、笔记文档结构（17 节骨架）、SVG 配图规范与网站部署集成。触发于"精读这篇论文"、"分析这篇论文"、"Explain this paper"、"Review this paper"、"帮我做论文笔记"等请求。
---

# 论文精读 Skill

本工程（`ai-infra-notes`）的论文精读追求 **Reviewer 视角**而非摘要复述：分析要有论文依据，公式要解释动机，实验要回答"为什么能证明观点"。

最终目标：

> 一篇论文看完之后，不需要再次阅读全文即可回忆起整个工作。

---

## 1. 目录组织

```
aiinfra/paper/
├── SKILL.md                        # 本文件（精读规范）
├── images/                         # 所有论文笔记共享的 SVG 插图
├── website/                        # 站点产物（build.py 生成，勿手改）
└── <paper-name>/                   # 每篇论文一个目录，全小写下划线
    ├── <paper-name>.pdf            # 原文 PDF（arXiv 下载）
    └── README.md                   # 精读笔记主体（按 §4 骨架）
```

- 论文目录名建议与 arXiv 短名一致，如 `attention_is_all_you_need`、`flashattention2`。
- **参考样板**：[`attention_is_all_you_need/README.md`](attention_is_all_you_need/README.md) 是当前唯一完整范例，新笔记的章节、详略、口吻应与它对齐。

## 2. 阅读原则

1. **不复述 Abstract**——用自己的逻辑重组论文，而非照抄原文顺序。
2. **所有分析必须有论文依据**。禁止猜测、脑补、编造实验与结果；论文未说明的，明确写 `Paper does not mention this.`。
3. **公式必须解释**：每个变量、为什么这样设计、数学意义、与目标的关系、对优化的帮助。不要只复制公式。
4. **图必须按数据流解释**：Framework / Pipeline / Architecture 图讲数据怎么流动，而不是"图里有什么"。
5. **实验必须回答"为什么证明了作者观点"**，而不是 `Table 2 shows...`。

## 3. 阅读流程

按 17 个阶段推进。简单阶段一句话带过，重点阶段（Core Idea、Method、Formula）写透。

| # | 阶段 | 关键问题 |
|---|------|----------|
| 1 | Metadata | 标题/作者/机构/Venue/年份/arXiv/代码链接/任务/关键词 |
| 2 | Summary | 5~10 句：Problem → Method → Result → Contribution |
| 3 | Background | 已有方法哪里不好？哪些问题一直没解决？作者真正想解决什么？ |
| 4 | Core Idea | 作者真正的新思想是什么？（全文重点，见 §3.1） |
| 5 | Method | 逐模块拆解（见 §3.2） |
| 6 | Formula Explanation | 每个核心公式的变量、动机、作用（见 §3.3） |
| 7 | Algorithm | 自然语言版 + 伪代码版 + 复杂度与瓶颈 |
| 8 | Figures | 逐图：Purpose / 数据流 / Key Observation |
| 9 | Experiments | 为什么这些实验能证明论文观点？作者真正证明了什么？ |
| 10 | Contributions | Top 3，按重要性排序，不超过三条 |
| 11 | Limitations | `Paper says`（作者自认）与 `Reviewer thinks`（你的质疑）分开写 |
| 12 | Related Work | 表格：方法 / 核心思想 / 与本文差异 / 优点 / 弱点 |
| 13 | Reproducibility | Code/Dataset/超参/训练细节/硬件逐项检查，给 ⭐1~5 评分 |
| 14 | Reading Notes | 最值得记住的 10 条知识点，每条一句话 |
| 15 | Interview Version | "讲讲这篇论文"的 2 分钟版 + 5 分钟版 |
| 16 | Engineering Insights | 面向 CUDA / AI Infra 的工程借鉴点（见 §6） |
| 17 | Future Work | 作者自己的方向 + 你认为后来真正发生的延续 |

### 3.1 Core Idea 链式图

不要描述流程，要描述创新点。统一用代码块画 Problem → Observation → Insight → Method → Benefit 的推理链：

```text
Problem     已有方法的根本瓶颈
   ↓
Observation 被忽视的关键事实
   ↓
Insight     由此产生的新想法
   ↓
Method      把想法落成方法
   ↓
Benefit     量化了什么收益
```

### 3.2 Method 模块模板

每个模块按固定字段写（不存在的字段省略，不要硬凑）：

- **Purpose**：这个模块解决什么
- **Input / Output**：形状与含义（张量维度写清，如 $n \times d_k$）
- **Algorithm**：核心计算式
- **Complexity**：时间与空间
- **Advantages / Limitations**：设计取舍

### 3.3 Formula 检查清单

每个核心公式回答：

- 变量表（符号 → 含义 → 维度）
- 公式在训练 / 推理时分别起什么作用
- 为什么成立、为什么这样设计（如 $1/\sqrt{d_k}$ 防 softmax 饱和）
- 相比已有公式的区别
- 复杂度影响

Loss、Attention、Normalization、Softmax、Kernel、Sampling 相关公式全部覆盖。

## 4. 输出文档结构

笔记统一写在 `<paper-name>/README.md`，章节固定如下（与样板一致）：

```markdown
# <论文标题> —— <一句话副标题>

> 原文 PDF：[<paper-name>.pdf](<paper-name>.pdf)

## 1. Metadata          ← 表格
## 2. Summary
## 3. Background        ← 3.1 已有方法及其问题 / 3.2 作者真正想解决的问题
## 4. Core Idea         ← 链式图代码块 + 一句话总结
## 5. Method            ← 5.x 分模块，配 SVG 图
## 6. Formula Explanation
## 7. Algorithm
## 8. Figures
## 9. Experiments       ← 9.x 设置/主结果/消融/作者真正证明了什么
## 10. Contributions
## 11. Limitations      ← Paper says / Reviewer thinks
## 12. Related Work     ← 表格
## 13. Reproducibility  ← 表格 + ⭐ 评分
## 14. Reading Notes
## 15. Interview Version
## 16. Engineering Insights
## 17. Future Work

> **一句话总结**：……
```

## 5. 写作风格

- **中文为主**，术语保留英文（Kernel、Warp、Softmax 等）。
- Markdown：多用表格与列表，避免长段落；代码块标注语言（` ```cpp ` / ` ```python ` / ` ```text `）。
- 数学公式用 `$...$` / `$$...$$`，**禁止**用反引号包裹公式；函数/运算符用 LaTeX 命令（`\log`、`\sum`、`\sqrt`）。
- 推理链、数据流等示意图优先用 ` ```text ` 代码块或 SVG；**站点不渲染 Mermaid，不要用 Mermaid**。
- 结尾给一句 `> **一句话总结**`。

## 6. Engineering Insights 专项

AI / CUDA / LLM / Compiler / 系统类论文，§16 额外分析（按相关性取舍）：

| 维度 | 关注点 |
|------|--------|
| Kernel Design | Grid / Block / Warp 划分，occupancy |
| Memory | Memory Layout、Shared Memory、Register、合并访存（coalescing） |
| Bound 分析 | Compute vs Memory Bound、Roofline（适用时） |
| 硬件特性 | Tensor Core、量化策略、TMA / Warp Specialization |
| 系统结构 | KV Cache、算子融合、Tiling Strategy、流水与同步 |
| 权衡 | Latency vs Throughput、Hardware Awareness |
| 可迁移性 | 该优化思路能否迁移到其他算子/硬件 |

## 7. SVG 配图规范

论文笔记必须为关键内容（架构、数据流、流水线、对比）补充 SVG，不要只用文字。

- **禁止 ASCII 图片**：所有示意图、流程图、架构图一律用 SVG，不要在 Markdown 中嵌入 ASCII 字符画（如用 `+---+`、`|   |` 拼成的表格或流程图）

- **存放位置**：`aiinfra/paper/images/`（所有笔记共享）
- **引用格式**（从 `<paper-name>/README.md` 出发，注意是 `../images/`）：

  ```markdown
  ![中文alt](../images/xxx.svg)
  ```

- **命名**：全小写 + 下划线，语义化，如 `transformer_architecture.svg`
- **数量**：每篇 2-5 张，插在小节首行；alt 文本用中文
- **风格**：统一手绘 sketch 风（`feTurbulence` 抖动滤镜 + Comic Sans / 楷体），细则参照 [`../topics/SKILL.md`](../topics/SKILL.md) 的「图片」规范

## 8. 网站构建与部署

笔记写完后由 `aiinfra/paper/website/build.py` 生成站点页面，并被根 `build.py` 复制到 `public/paper/`：

```bash
python3 aiinfra/paper/website/build.py   # 只重建 paper 站点
python3 build.py                          # 组合构建全站（含 paper）
```

- 笔记中的 `](../images/xxx.svg)`、PDF 相对链接会被构建器正确处理，无需手工改写。
- `aiinfra/paper/website/` 是构建产物，会被重建，**不要手改**。
- push 到 `main` 时 `deploy.yml` 监听 `aiinfra/paper/**` 自动部署到 GitHub Pages。

**自检清单**：

- [ ] 笔记位于 `aiinfra/paper/<paper-name>/README.md`，PDF 已放入同目录
- [ ] 17 节骨架完整（Core Idea / Limitations / Engineering Insights 不可省略）
- [ ] 公式用 `$...$`，无反引号包裹公式，无 Mermaid
- [ ] 含 2-5 张 SVG，引用路径为 `../images/xxx.svg`，alt 为中文
- [ ] `python3 build.py` 成功生成 `public/paper/<paper-name>/index.html`
- [ ] `git push origin main` 触发部署
