# SGLang：高性能 LLM 推理服务引擎

> **适用对象**：想系统上手 SGLang 的开发者——无需推理引擎先验经验（[Week 6](../../daily/week6/README.md) / [Week 7](../../daily/week7/README.md) 或 [vLLM 专题](../vllm/README.md) 学过更好，概念会更亲切）
> **本周目标**：从"会用 SGLang 跑推理"到"理解它的核心机制"——打通 认知 → 环境搭建 → API 使用 → 推理原理 → RadixAttention 等独门机制 → 性能压测 → 完整项目 的七步链路
> **时间投入**：每天 2～3 小时（理论 ≤ 40%，实践 ≥ 60%），周计约 19 小时
> **周日里程碑**：完成 Mini Project `mini-llm-service`——一个可复现的推理服务（OpenAI 兼容 API + 并发测试 + 性能报告），能回答"这台机器上这个模型，多少并发时性价比最高"
> **配套计划**：完整七天计划见 [SGLang 一周入门学习计划](notes/SGLang一周入门学习计划.md)（本专题的原始蓝本）

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 SGLang 的定位与选型逻辑，独立完成安装、启动、四种调用方式；理解 RadixAttention、零开销调度、Chunked Prefill 三大核心机制；会用 `bench_serving` 做规范压测并解读 |
| **核心产出** | ① 裸推理基线脚本与耗时记录 ② 环境/启动日志笔记（含 KV 池 `#tokens`）③ 可复用的 OpenAI 兼容客户端 ④ 开/关前缀缓存对照实验数据 ⑤ 并发阶梯 + 缓存对照的性能报告 ⑥ Mini Project 代码仓库 |
| **验收标准** | ① 能说清 SGLang 与 vLLM 在前缀缓存机制上的差异与选型结论 ② 不看笔记写出启动命令与 OpenAI 兼容客户端 ③ 能推导 KV 每 token 字节公式并与启动日志对账 ④ 按五段式讲清 RadixAttention ⑤ 性能报告有数据、有对比、有结论 |
| **面试准备** | 每日积累面试题，覆盖：推理引擎存在的意义、memory-bound 与 batch 免费午餐、KV Cache 显存公式、Continuous Batching 调度时机、radix tree vs block hash、TTFT/TPOT 权衡曲线 |

### 本专题与 Week 6/7、vLLM 专题的边界

| 维度 | Week 6/7（每日教程） | vLLM 专题 | 本 SGLang 专题 |
|------|---------------------|-----------|---------------|
| **视角** | 原理复现——手写 mini 引擎 | 工程层——vLLM 的使用、源码与部署 | 工程层——SGLang 的使用、机制与压测 |
| **核心机制** | KV Cache / 调度器的教学简化版 | PagedAttention / block 级前缀缓存 | RadixAttention / token 级前缀缓存 |
| **适合时机** | 先学（打原理地基） | 任选其一或两者对照 | 学完 Week 6/7 后任选其一或对照 |
| **产出** | mini 引擎 | 部署 + 源码走读笔记 | 部署 + 机制实验 + 性能报告 |

> 💡 **一句话总结**：SGLang 与 vLLM 共享同一套"通用内功"（Continuous Batching、分页 KV），差异在"独门招式"（前缀缓存的匹配机制）——两个专题互为对照，先学哪个都行。

### 前置准备清单

- [ ] Linux + NVIDIA GPU（显存 ≥ 8GB；示例统一用 `Qwen/Qwen3-0.6B` 小模型）
- [ ] Python 3.10+、CUDA 12.x/13.0、PyTorch 2.x
- [ ] `pip install "sglang[all]"`（或使用官方 Docker 镜像 `lmsysorg/sglang`）
- [ ] HuggingFace 访问（国内可设 `HF_ENDPOINT=https://hf-mirror.com`）

#### 必读资源（本周反复使用）

- ⭐ [SGLang 官方文档](https://docs.sglang.ai/) — 安装、API、特性的权威参考
- ⭐ [SGLang GitHub 仓库](https://github.com/sgl-project/sglang) — Quick Start 与 issue 排障
- 📌 [Week 6 推理系统基础与 KV Cache](../../daily/week6/README.md) — 前置原理（prefill/decode/KV Cache）
- 📌 [Week 7 Batching 与调度](../../daily/week7/README.md) — 前置原理（Continuous Batching、Prefix Caching）
- 📌 [vLLM 专题](../vllm/README.md) — 对照阅读（PagedAttention vs RadixAttention）

---

## 为什么学 SGLang

SGLang 是 LMSYS（UC Berkeley / 斯坦福 Sky Computing Lab）开源的高性能 LLM 推理服务框架（NeurIPS 2024 论文），DeepSeek 官方推荐推理方案之一，xAI（Grok）生产采用。它的价值不在"又一个推理框架"，而在把**前缀缓存做成了引擎的一等公民**：

| 创新 | 解决的问题 | 效果 |
|------|------------|------|
| **RadixAttention** | 多轮对话 / RAG / Agent 的共享前缀每个请求都从第 0 个 token 重算 | 基数树 token 级前缀缓存，默认开启零配置；高前缀重叠负载 TTFT 降 20%～40% |
| **零开销调度器** | CPU 调度耗时赶上 GPU 一步 decode，GPU 空转等 CPU | CPU 准备第 n+1 步与 GPU 跑第 n 步重叠，高并发下调度开销趋近于零 |
| **Chunked Prefill** | 长 prompt 的 prefill 独占 GPU，在线用户 decode 集体卡顿 | 长 prefill 切块与 decode 混编，TPOT 曲线平滑 |

| 场景 | 不懂 SGLang | 懂 SGLang |
|------|------------|-----------|
| **技术选型** | "大家都用 vLLM 就用 vLLM" | 前缀重叠率 > 60% 的负载优先试 SGLang，用 PoC 数据定案 |
| **性能调优** | 不知道为什么并发上去延迟暴涨 | 知道 KV 池容量、甜点区、抢占重算，会用 `/metrics` 和 bench_serving 定位 |
| **面试** | 只会背"SGLang 有 radix cache" | 能按五段式讲清机制、画出基数树、现场推 KV 显存公式 |

> 💡 **一句话总结**：模型是发动机，推理引擎是变速箱 + 底盘——SGLang 是一台**特别擅长复用重复计算**的高性能变速箱。

---

## 核心概念速览

### 1. 推理的两阶段与资源瓶颈

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算内容 | 一次并行算完整个 prompt | 每次前向只出 1 个 token |
| 瓶颈 | Compute Bound（算力） | Memory Bound（显存带宽） |
| 决定指标 | TTFT（首 token 延迟） | TPOT（每 token 间隔） |

decode 的 memory-bound 是一切故事的起点：每步搬全部权重却只算一点点 → **batch 是免费午餐**（搬运成本与 batch 无关），直到撞上 KV 显存容量或算力上限。

### 2. 引擎通用内功（与 vLLM 共享）

- **KV Cache**：缓存每层注意力的 K/V 投影，计算量从 O(n²) 降到 O(n)；代价是显存（每 token 字节 = $2 \times L \times H_{kv} \times d_{head} \times b$）
- **Continuous Batching**：每个 decode step 后重组 batch，完成即离开、等待即插入（Orca, OSDI 2022）
- **分页 KV 池**：启动时预分配显存池，运行时按需切分——`mem-fraction-static` 控制池子大小

### 3. SGLang 独门机制（本专题主角）

- **RadixAttention**：基数树 + 最长前缀匹配 + LRU 淘汰，token 级粒度、默认开启；省的全是共享前缀的 prefill
- **零开销调度器**：CPU 调度与 GPU 计算 overlap
- **Chunked Prefill**：长 prefill 切块与 decode 混合编批

---

## 最小可运行示例

```bash
# 启动 OpenAI 兼容服务
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000
# 等待日志出现 "The server is fired up and ready to roll!"
```

```python
# 用 OpenAI SDK 直接调用——只改 base_url
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "用一句话解释 RadixAttention。"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

> 💡 **观察点**：启动日志里的 `KV Cache is allocated. #tokens: N`——这个数字是 KV 池能容纳的 token 数，除以 context length 就是并发容量（Day 2 / Day 4 展开）。

---

## 本周学习计划

| 天数 | 主题 | 核心概念 | 核心产出 |
|------|------|----------|----------|
| Day 1 | 认识 SGLang | 推理引擎存在的意义、SGLang 定位与发展、vs vLLM 选型 | 裸推理基线脚本与耗时记录 |
| Day 2 | 环境搭建 | 安装路径取舍、启动日志六阶段、mem-fraction-static | 服务跑通 + 日志笔记（含 `#tokens`） |
| Day 3 | 基本使用 | `/generate` vs `/v1`、OpenAI 兼容、采样参数、离线 Engine | 可复用的 `chat_client.py` |
| Day 4 | 推理基础 | Prefill/Decode、KV Cache 显存公式、Continuous Batching | 两个观察实验 + 请求旅程图 |
| Day 5 | 核心机制 | RadixAttention 五段式、零开销调度、Chunked Prefill | 开/关前缀缓存对照实验数据 |
| Day 6 | 性能工程 | TTFT/TPOT/吞吐、benchmark 方法学、bench_serving | 并发阶梯 + 缓存对照性能报告 |
| Day 7 | Mini Project | 七天技能串联：部署 + 客户端 + 测试 + 报告 | `mini-llm-service` 代码仓库 |

### Day 1（周一）：认识 SGLang —— 它是什么、解决什么问题

- 理解 `model.generate()` 裸推理撑不起真实服务的四个原因（吞吐 / 显存 / 重复计算 / 非服务化）
- 掌握"batch 是免费午餐"的 memory-bound 直觉；建立 SGLang 五层位置图
- 对比 vLLM：核心差异在前缀缓存匹配机制（token 级基数树 vs block 级哈希）
- 实践：裸推理基线脚本 + GitHub/文档侦察 → [进入 Day 1](day1.md)

### Day 2（周二）：环境搭建 —— 安装并启动 SGLang

- uv/pip 与 Docker 两条安装路径的取舍；`launch_server` 启动与参数
- 逐行读懂启动日志：`#tokens` 是 Day 4 讲 KV Cache 的关键伏笔
- 实践：安装 → 启动 → `/health` 健康检查 → 排障速查 → [进入 Day 2](day2.md)

### Day 3（周三）：基本使用 —— API 调用与 OpenAI Compatible API

- `/generate` 原生端点与 `/v1/chat/completions` 的本质区别（chat template 由谁套）
- OpenAI SDK 对接三要素；流式输出体验 TTFT/TPOT 体感；离线 `sgl.Engine` 批量推理
- 实践：curl / SDK / 流式 / Engine 四种方式全打通 → [进入 Day 3](day3.md)

### Day 4（周四）：LLM 推理基础 —— 通用内功

- Prefill（compute-bound）与 Decode（memory-bound）性质相反的根源
- KV Cache 显存公式推导并与 Day 2 的 `#tokens` 对账；GQA 为什么体现在 KV 头数上
- Continuous Batching 在"每一步"调度：班车 vs 传送带
- 实践：TTFT 体检 + 长短混合并发观察 + 手绘请求旅程图 → [进入 Day 4](day4.md)

### Day 5（周五）：SGLang 核心机制 —— 独门武功

- RadixAttention 五段式：问题 → 传统方案 → 缺陷 → SGLang 方案 → 收益来源
- 零开销调度器与 Chunked Prefill 各自解决的时间矛盾与公平性 tradeoff
- 实践：多轮对话前缀复用实验 + `--disable-radix-cache` 对照 + `/metrics` → [进入 Day 5](day5.md)

### Day 6（周六）：性能与工程实践 —— 指标与 Benchmark

- TTFT / TPOT / 吞吐三大指标与延迟-吞吐权衡曲线（欠载 / 甜点 / 过载三区）
- benchmark 方法学五条纪律：固定分布、控制变量、预热、分位数、声明口径
- 实践：`bench_serving` 并发阶梯找拐点 + 共享前缀数据集量化缓存收益 → [进入 Day 6](day6.md)

### Day 7（周日）：Mini Project —— 部署一个带性能报告的推理服务

- 串联全部技能：启动脚本、封装客户端、冒烟/并发测试、自写 benchmark、RESULTS.md
- 三条验收：冒烟 / 并发 20×200 / 自写与官方工具量级一致
- 产出可写进简历的完整项目 `mini-llm-service` → [进入 Day 7](day7.md)

---

## 面试要点

**Q：SGLang 与 vLLM 的核心差异是什么？怎么选型？**
> 表层能力已趋同（Continuous Batching、分页 KV、FP8、OpenAI API），分水岭在**前缀缓存的匹配机制**：vLLM APC 按 block 哈希匹配（粒度粗，前缀边界不对齐时整块失配）；SGLang RadixAttention 用基数树做 token 级最长前缀匹配、默认开启。选型：前缀重叠率 > 60%（RAG、多轮 Agent、固定 system prompt）优先试 SGLang，实测吞吐可高 20-40%；追求模型覆盖广度与新硬件首发选 vLLM；最终用 PoC 数据定案，不要信仰站队。

**Q：RadixAttention 为什么快？收益和损失各在哪？**
> 基数树组织所有请求的 KV——每个节点是一段 token 序列，路径即前缀。新请求做最长前缀匹配：命中部分直接复用 KV、只 prefill 尾部；完成后新 KV 插回树，LRU 从最冷叶子往上淘汰。**省的全是共享前缀的 prefill 计算**（512 token 的 system prompt，1000 个并发只算 1 次），直接收益是 TTFT、间接收益是吞吐。多轮对话 = 老路径延长，天然全命中。无共享前缀时几乎零收益但也几乎零损失（匹配开销微秒级）——所以敢默认开启。

**Q：decode 为什么是 memory-bound？"batch 是免费午餐"怎么理解？**
> decode 每步前向都要把全部模型权重从显存搬到计算单元，却只算 1 个 token——搬运成本 ∝ 权重大小、与 batch 无关；加大 batch 只增加计算量，而计算单元本来就闲置。batch 1→64：单步耗时几乎不变、产出 ×64。免费到撞上天花板为止：KV 池容量（`#tokens` 上限）或算力——所以吞吐曲线会饱和甚至回落（过载触发抢占重算，干了活还得重干）。

**Q：KV Cache 每 token 占多少显存？启动日志的 `#tokens` 怎么来的？**
> $2 \times L \times H_{kv} \times d_{head} \times b$（K/V 两份 × 层数 × **KV 头数** × 头维 × 精度字节）——注意用 KV 头数不是注意力头数（GQA 的省显存就藏在这里）。`#tokens` = KV 池字节 ÷ 每 token 字节，除以 context length 就是并发容量上限。现场能推这个公式并和日志对上账，比背十个概念有说服力。

**Q：TTFT 和 TPOT 分别由什么决定？怎么诊断"服务慢"？**
> TTFT = 排队 + prefill（prompt 长度、前缀缓存命中率、并发排队）；TPOT = decode 每步间隔（batch 大小、算力、chunked prefill 干扰）。TTFT 差 → 查排队与缓存命中（`cache_hit_rate`）；TPOT 差 → 查 batch 是否过大、长 prefill 是否阻塞。SLO 绑分位数（P99）不绑均值——长尾才是最差 1% 用户的真实体验。

---

## 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [SGLang 官方文档](https://docs.sglang.ai/) | 官方文档 | ⭐ 必读 |
| [SGLang GitHub 仓库](https://github.com/sgl-project/sglang) | 源码 | ⭐ 必读 |
| [SGLang 论文](https://arxiv.org/abs/2312.07104)（NeurIPS 2024） | 论文 | ⭐ 必读（RadixAttention 设计动机） |
| [SGLang 一周入门学习计划](notes/SGLang一周入门学习计划.md) | 本专题蓝本 | ⭐ 必读 |
| [Week 6 推理系统基础与 KV Cache](../../daily/week6/README.md) | 每日教程 | 📌 推荐（前置） |
| [Week 7 Batching 与调度](../../daily/week7/README.md) | 每日教程 | 📌 推荐（前置） |
| [vLLM 专题](../vllm/README.md) | 姊妹专题 | 📌 推荐（对照阅读） |
| [Orca](../../paper/orca/orca.pdf)（OSDI 2022） | 论文 | 📎 参考（Continuous Batching 原始论文） |
| [Sarathi-Serve](../../paper/sarathi_serve/sarathi_serve.pdf)（OSDI 2024） | 论文 | 📎 参考（Chunked Prefill） |
| [Week 8 推理加速](../../daily/week8/README.md) | 每日教程 | 📎 衔接（投机解码、量化） |

---

## 目录结构

```
aiinfra/topics/sglang/
├── README.md                      # 本文件（专题概览 + 一周提纲）
├── day1.md                        # Day 1: 认识 SGLang（定位、选型、裸推理基线）
├── day2.md                        # Day 2: 环境搭建（安装、启动日志、显存切分）
├── day3.md                        # Day 3: 基本使用（原生/OpenAI API、流式、离线 Engine）
├── day4.md                        # Day 4: 推理基础（两阶段、KV Cache、Continuous Batching）
├── day5.md                        # Day 5: 核心机制（RadixAttention、零开销调度、Chunked Prefill）
├── day6.md                        # Day 6: 性能工程（三指标、方法学、bench_serving）
├── day7.md                        # Day 7: Mini Project（mini-llm-service 完整项目）
└── notes/
    └── SGLang一周入门学习计划.md    # 原始七天学习计划（本专题蓝本）
```

> 💡 **后续延伸**：完成本专题后，沿 [SGLang 一周入门学习计划](notes/SGLang一周入门学习计划.md)第四节的 4 周进阶路线继续：① 通用原理深化（对照 vLLM 精读 PagedAttention）② SGLang 源码走读（`http_server.py` → `scheduler.py` → `radix_cache.py`）③ 硬件与算子层（TP / FlashAttention / Triton）④ 优化专题 + 产出（FP8 / 投机解码 / PD 分离 / 生产化）。
