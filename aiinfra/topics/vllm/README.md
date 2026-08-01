# vLLM：一周入门高吞吐 LLM 推理引擎

> **适用对象**：完成 [Week 5 PagedAttention / Mini 引擎](../../daily/week5/README.md) 与 [Week 6 Continuous Batching](../../daily/week6/README.md)（或具备等价的 KV Cache / 调度器基础）、想系统上手 vLLM 的开发者
> **本周目标**：从"会用 vLLM 跑推理"到"读懂 vLLM 架构"，打通 快速上手 → PagedAttention → Continuous Batching → 源码走读 → 优化特性 → 分布式 → 部署压测 的完整链路
> **时间投入**：工作日每天 2.5h（早间 1.5h + 晚间 1h），周末每天 5h，周计 22.5h
> **周日里程碑**：能用 `vllm serve` 部署一个开源模型并完成压测（TTFT / TPOT / 吞吐），能画出 vLLM 分层架构图并讲清一次请求从进入到返回的完整生命周期，产出源码走读笔记与面试问答

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 vLLM 的两大核心创新（PagedAttention、Continuous Batching），读懂 LLMEngine → Scheduler → ModelRunner → Worker 的分层架构，会用主要优化特性（Chunked Prefill、Prefix Caching、CUDA Graph、量化），能独立完成部署与压测 |
| **核心产出** | ① 离线推理 + Server 部署 demo ② PagedAttention / block table 笔记 ③ 调度器状态机笔记 ④ 源码架构走读笔记 ⑤ benchmark 压测报告 ⑥ 面试问答集 |
| **验收标准** | ① 能讲清 PagedAttention 为什么能把 KV Cache 显存浪费从 60-80% 降到 ~4% ② 能画出 Scheduler 的 waiting/running/swapped 状态机 ③ 能说清 prefill（compute-bound）与 decode（memory-bound）的差异及对调度的影响 ④ 能解释 Chunked Prefill 如何降低 TTFT 与 ITL 的冲突 ⑤ 压测报告含 TTFT / TPOT / throughput 三组指标 |
| **面试准备** | 积累 8-10 道面试题，覆盖 PagedAttention 原理、Continuous Batching、prefill/decode 差异、vLLM 架构分层、KV Cache 量化、Speculative Decoding |

### 本专题与 [Week 5](../../daily/week5/README.md) / [Week 6](../../daily/week6/README.md) / [vLLM 论文精读](../../paper/vllm/README.md) 的边界

| 维度 | Week 5/6（每日教程） | vLLM 论文精读 | 本 vLLM 专题 |
|------|----------------------|----------------|---------------|
| **视角** | 原理复现——手写 mini 调度器 / PagedAttention kernel | 论文层——PagedAttention 的设计动机与实验 | 工程层——真实 vLLM 的使用、源码与部署 |
| **范围** | 单个机制的简化实现 | 单篇论文 | 完整引擎：API → 调度 → kernel → 分布式 |
| **深度** | 教学简化版（几百行 Python/CUDA） | 论文公式与图表 | 源码级（vLLM 主仓库） |
| **产出** | mini 引擎 v0/v1 | 论文精读笔记 | 可上线的部署 + 压测报告 |

> 💡 **一句话总结**：Week 5/6 教你"自己造一个简化版 vLLM"，论文精读教你"PagedAttention 为什么这样设计"，本专题教你"真实的 vLLM 怎么用、怎么读、怎么部署"——先复现原理再读源码，会比直接啃源码轻松一个数量级。

### 前置准备清单

#### 软件/环境验证
- [ ] Python >= 3.9
- [ ] CUDA >= 11.8 的 GPU（vLLM 官方要求 compute capability >= 7.0；量化/FP8 特性需要更新的卡）
- [ ] `pip install vllm`（建议虚拟环境）
- [ ] HuggingFace 模型访问（本专题示例用 `Qwen/Qwen2.5-0.5B-Instruct` 等小模型，显存 < 2GB 即可跑通）

#### 验证命令
```bash
# 验证 vLLM 安装与 GPU 可见性
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

#### 必读资源（本周会反复用到）
- ⭐ [vLLM 论文精读](../../paper/vllm/README.md) — PagedAttention 原始论文（SOSP 2023），Day 2 的理论骨架
- ⭐ [vLLM 官方文档](https://docs.vllm.ai/) — API、特性、配置的权威参考
- ⭐ [vLLM GitHub 仓库](https://github.com/vllm-project/vllm) — Day 4 源码走读对象
- 📌 [Week 5 Day 4 PagedAttention kernel](../../daily/week5/day4/README.md) — 手写版 PagedAttention，读源码前的热身
- 📌 [Week 6 Day 2 Continuous Batching](../../daily/week6/day2/README.md) — 手写版调度器，Day 3 的对照

---

## 为什么学 vLLM

vLLM 是当前开源 LLM 推理引擎的事实标准（vLLM / TensorRT-LLM / SGLang 三足鼎立中使用最广），也是 AI Infra 面试中推理方向的最高频考点。它的价值不在于"又一个推理框架"，而在于两个定义行业的创新：

| 创新 | 解决的问题 | 效果 |
|------|------------|------|
| **PagedAttention** | KV Cache 按最大长度预分配，显存碎片 + 内部浪费高达 60-80% | 借鉴 OS 虚拟内存分页，显存浪费降到 ~4%，batch size 提升 2-4 倍 |
| **Continuous Batching** | 静态 batching 要等整批最慢的序列结束，GPU 大量空转 | iteration 级调度，完成即换入新请求，吞吐提升一个数量级 |

| 场景 | 不懂 vLLM | 懂 vLLM |
|------|-----------|---------|
| **部署模型** | 只会 `model.generate()`，吞吐惨不忍睹 | 知道 `gpu_memory_utilization`、`max_num_seqs` 怎么调 |
| **推理优化** | 不知道为什么 decode 慢 | 知道 decode 是 memory-bound，瓶颈在 KV Cache 读带宽 |
| **读源码** | 打开仓库几十万行无从下手 | 按 LLMEngine → Scheduler → Worker 分层拆解 |
| **面试** | 只会背"PagedAttention 是分页" | 能讲清 block table、copy-on-write、调度抢占（preemption） |

> 💡 **一句话总结**：vLLM 是把 OS 内存管理思想搬进 LLM 推理的典范——学完它，你对"系统思维如何优化 AI 负载"会有具体而非抽象的理解。

---

## 核心概念速览

### 1. 推理的两阶段：Prefill 与 Decode

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算内容 | 并行处理全部输入 token | 自回归逐 token 生成 |
| 算术强度 | 高（compute-bound） | 低（memory-bound，瓶颈在 KV Cache 读带宽） |
| 关键指标 | TTFT（Time To First Token） | TPOT / ITL（每 token 间隔） |
| 优化方向 | 算力利用率、Chunked Prefill | KV Cache 管理、量化、Speculative Decoding |

### 2. PagedAttention：KV Cache 的分页管理

- **逻辑块 → 物理块**：每个序列的 KV Cache 切成定长 block（默认 16 token），通过 block table 映射到物理显存块，逻辑连续、物理离散
- **按需分配**：不再按 `max_model_len` 预分配，生成多少占多少
- **共享与 Copy-on-Write**：beam search / parallel sampling 时多个序列共享 prompt 的物理块，写时才复制

### 3. Continuous Batching：iteration 级调度

- 每个 decoding step 结束就重新组 batch：完成的请求退出、waiting 队列中的请求补入
- 三个队列状态机：`waiting` → `running` →（显存不足时）`swapped` / 抢占重算
- 与 PagedAttention 配合：分页让"换入换出"的成本足够低，调度才能激进

### 4. vLLM 分层架构

```
用户 API 层    LLM（离线）/ OpenAI API Server（在线）
引擎层        LLMEngine：接收请求、驱动 step 循环
调度层        Scheduler：waiting/running/swapped 队列 + 配额决策
块管理层      BlockSpaceManager：block table 与物理块分配
执行层        ModelRunner / Worker：构造输入、调用模型
算子层        PagedAttention kernel、量化 GEMM、CUDA Graph
```

> 💡 读源码时记住一条主线：**一次 `generate()` = `add_request` → 循环 `step()` → `scheduler.schedule()` 决定本步跑哪些请求 → `model_runner.execute_model()` 前向 → 采样 → 更新状态**。

---

## 最小可运行示例：离线批推理

```python
# quickstart.py —— vLLM 最小离线推理示例
# 运行: python3 quickstart.py

from vllm import LLM, SamplingParams

prompts = [
    "用一句话解释 PagedAttention：",
    "CUDA 中 shared memory 的作用是",
]

# gpu_memory_utilization 控制 KV Cache 显存占比，小显存机器可调低
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.6)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"prompt: {output.prompt!r}")
    print(f"generated: {output.outputs[0].text!r}")
```

```bash
python3 quickstart.py
```

Server 模式（Day 1 晚间任务）：

```bash
# 启动 OpenAI 兼容服务
vllm serve Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.6

# 另开终端验证
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": "你好", "max_tokens": 32}'
```

---

## 本周学习计划

| 天数 | 主题 | 核心概念 | 核心产出 |
|------|------|----------|----------|
| Day 1 | 快速上手与全景 | LLM 离线 API、OpenAI 兼容 Server、SamplingParams、引擎分层初识 | 跑通离线推理 + Server 部署，画出架构分层初稿 |
| Day 2 | PagedAttention 原理 | KV Cache 显存浪费、分页、block table、Copy-on-Write | block table 图解笔记，对照手写 kernel 读 CUDA 实现 |
| Day 3 | Continuous Batching 与调度器 | iteration 级调度、三队列状态机、抢占（recompute/swap）、配额参数 | 调度器状态机笔记，`max_num_seqs`/`max_num_batched_tokens` 调参实验 |
| Day 4 | 源码架构走读 | LLMEngine → Scheduler → BlockSpaceManager → ModelRunner → Worker 调用链 | 一次请求的完整生命周期走读笔记 |
| Day 5 | 吞吐优化特性 | Chunked Prefill、Prefix Caching、CUDA Graph、量化（GPTQ/AWQ/FP8） | 各特性开关的 A/B 对比实验记录 |
| Day 6 | 分布式与解码优化 | Tensor/Pipeline Parallel、Speculative Decoding、Structured Output、LoRA serving | 多卡 TP 部署 demo（或单机模拟），特性对比笔记 |
| Day 7 | 部署实战与 Benchmark | `vllm bench`、TTFT/TPOT/吞吐、Roofline 视角分析、与 mini 引擎对照复盘 | 完整压测报告 + 架构图定稿 + 面试问答集 |

### Day 1（周一）：快速上手与全景

- 安装 vLLM，跑通离线 `LLM.generate()` 与 `vllm serve` 两种模式
- 读官方文档的 Engine Args，理解 `gpu_memory_utilization`、`max_model_len`、`dtype` 三个最常用参数
- 晚间：浏览 vLLM 仓库目录结构，建立"引擎层/调度层/执行层"的初步分层印象

### Day 2（周二）：PagedAttention 原理

- 精读 [vLLM 论文](../../paper/vllm/README.md) 的 §3-§4：KV Cache 浪费分析、分页设计、block table
- 对照 [Week 5 Day 4 手写 PagedAttention kernel](../../daily/week5/day4/README.md)，找到 vLLM 源码中对应的 CUDA kernel（`csrc/attention/`）
- 产出：画出 logical block → block table → physical block 的映射图，标出 CoW 时机

### Day 3（周三）：Continuous Batching 与调度器

- 对照 [Week 6 Day 2 手写 Continuous Batcher](../../daily/week6/day2/README.md)，读 vLLM 的 `Scheduler`：waiting/running/swapped 三队列
- 理解抢占策略：显存不足时是 recompute 还是 swap，各适合什么场景
- 实验：固定模型，扫 `max_num_seqs` × `max_num_batched_tokens`，观察吞吐与 TTFT 变化

### Day 4（周四）：源码架构走读

- 主线走读：`LLM.generate()` → `LLMEngine.add_request()` → `step()` → `Scheduler.schedule()` → `ModelRunner.execute_model()` → Sampler
- 重点数据结构：`SequenceGroup`、`SequenceStatus`、`PhysicalTokenBlock`、scheduler outputs
- 产出：一张一次请求生命周期的时序图 + 关键类职责表（对照 [Week 8 Day 2 架构图方法](../../daily/week8/day2/README.md)）

### Day 5（周五）：吞吐优化特性

- **Chunked Prefill**：把长 prompt 切块与 decode 混跑，消除 prefill 对 ITL 的干扰（对照 [Week 6 Day 4 Chunked Prefill](../../daily/week6/day4/README.md)）
- **Prefix Caching**：system prompt / 多轮对话的 KV 复用
- **CUDA Graph**：消除小 batch decode 的 kernel launch 开销
- **量化**：GPTQ/AWQ/FP8 的支持方式（对照 [GPTQ 论文精读](../../paper/gptq/README.md)）
- 实验：每个特性单独开关，记录吞吐变化

### Day 6（周六）：分布式与解码优化

- Tensor Parallel：`--tensor-parallel-size N` 多卡部署，理解 TP 切分的是哪些权重
- Speculative Decoding：draft 模型 + verify 的加速原理与在 vLLM 中的开启方式
- Structured Output（guided decoding）、LoRA serving 简介
- 晚间：整理一周的实验数据

### Day 7（周日）：部署实战与 Benchmark

- 用 `vllm bench serve`（或 `benchmarks/` 脚本）对部署的服务压测：不同并发下的 TTFT / TPOT / throughput
- 用 [Week 8 Day 1 的 benchmark 方法论](../../daily/week8/day1/README.md) 分析瓶颈：哪个并发点开始 memory-bound 饱和
- 对照 [Week 6 Day 6 mini 引擎 benchmark](../../daily/week6/day6/README.md)，总结"教学版与工业版差在哪"
- 定稿架构图，整理面试问答集

---

## 面试要点

**Q：PagedAttention 解决了什么问题？原理是什么？**
> 传统推理框架按最大序列长度为每个请求预分配连续 KV Cache，内部碎片（没生成那么长）+ 外部碎片（大小不一难复用）导致 60-80% 显存浪费。PagedAttention 借鉴 OS 虚拟内存：把 KV Cache 切成定长 block（默认 16 token），序列持有逻辑块，通过 block table 映射到物理块，按需分配、逻辑连续物理离散。显存浪费降到 ~4%（仅最后一个 block 的内部碎片），同等显存下 batch size 可提升 2-4 倍。

**Q：Continuous Batching 相比 Static Batching 为什么能提升吞吐？**
> Static batching 要等整批请求全部生成完才能处理下一批，而各请求生成长度差异大，短请求结束后 GPU 槽位空转。Continuous batching（in-flight batching）在每个 decoding step 后重新组 batch：完成的请求立即退出，waiting 队列的请求立即补入，GPU 始终跑满。配合 PagedAttention 低开销的显存管理，吞吐可提升一个数量级（Orca/vLLM 论文数据）。

**Q：Prefill 和 Decode 的瓶颈有什么不同？对系统设计有什么影响？**
> Prefill 并行处理全部输入 token，算术强度高，是 compute-bound，关键指标是 TTFT；Decode 每步只算 1 个 token 但要读全部历史 KV Cache，算术强度极低，是 memory-bound（瓶颈在 HBM 读带宽），关键指标是 TPOT/ITL。影响：① 两阶段对硬件的需求不同，催生了 PD 分离（prefill/decode 部署在不同节点）② Chunked Prefill 把 prefill 切块与 decode 混跑，避免长 prompt 阻塞 decode 的 ITL ③ Decode 优化走 KV Cache 量化、Speculative Decoding 方向，Prefill 优化走算力利用率方向。

**Q：vLLM 的 Scheduler 在显存不足时怎么办？**
> 抢占（preemption）。两种策略：① **Recompute**：把被抢占序列的 KV Cache 块释放，下次调度时重新 prefill（默认，省显存、多算力）② **Swap**：把 KV Cache 换出到 CPU 内存，之后换回（省算力、占 CPU 内存且有 PCIe 传输开销）。长序列、共享前缀多的场景 swap 更划算；短序列重算更快。

**Q：Chunked Prefill 解决什么问题？**
> 长 prompt 的 prefill 一次占用整个 step，期间 decode 请求的 ITL 被拉长（用户看到流式输出卡顿）。Chunked Prefill 把 prefill 切成固定 token 预算的块，每个 step 内 decode 与一小块 prefill 混跑：TTFT 略增但 ITL 平滑，整体吞吐提升。本质上是用 `max_num_batched_tokens` 这个统一预算同时约束两类请求。

**Q：KV Cache 占多少显存？怎么估算？**
> 每 token 的 KV 字节数 = $2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype 字节数}$（2 是 K 和 V）。以 LLaMA-7B（32 层、32 头、$d_{\text{head}}=128$、FP16）为例：每 token ≈ 0.5MB，2048 token 的序列约 1GB。这也是为什么 vLLM 用 `gpu_memory_utilization` 把加载权重后的剩余显存几乎全划给 KV Cache 池。

**Q：Speculative Decoding 为什么能加速？vLLM 怎么支持？**
> Decode 是 memory-bound：每步读全部权重 + KV Cache 却只算 1 个 token，算力大量闲置。用小 draft 模型一次猜 $k$ 个 token，大模型一次前向并行 verify（verify 是 compute-bound 的活），接受其中匹配的前缀。单次前向产出多个 token，等效降低每 token 的显存读取成本。vLLM 通过 `--speculative-model` 等参数开启，也有 n-gram / Medusa 等免 draft 模型的方案。

---

## 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [vLLM 论文精读](../../paper/vllm/README.md)（SOSP 2023） | 论文精读 | ⭐ 必读 |
| [vLLM 官方文档](https://docs.vllm.ai/) | 官方文档 | ⭐ 必读 |
| [vLLM GitHub 仓库](https://github.com/vllm-project/vllm) | 源码 | ⭐ 必读 |
| [Week 5 PagedAttention / Mini 引擎](../../daily/week5/README.md) | 每日教程 | ⭐ 必读（前置） |
| [Week 6 Continuous Batching](../../daily/week6/README.md) | 每日教程 | ⭐ 必读（前置） |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)（OSDI 2022） | 论文 | 📌 推荐（Continuous Batching 原始论文） |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310)（OSDI 2024） | 论文 | 📌 推荐（Chunked Prefill） |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 论文 | 📌 推荐 |
| [GPTQ 论文精读](../../paper/gptq/README.md) | 论文精读 | 📎 衔接（量化） |
| [Week 8 面试准备](../../daily/week8/README.md) | 每日教程 | 📎 衔接 |

---

## 目录结构

```
aiinfra/topics/vllm/
├── README.md                    # 本文件（专题概览 + 一周提纲）
├── day1.md                      # Day 1: 快速上手与全景（两种模式、Engine Args、六层架构）
├── day2.md                      # Day 2: PagedAttention 原理（分页、block table、CoW）
├── day3.md                      # Day 3: Continuous Batching 与调度器（状态机、配额、抢占）
├── day4.md                      # Day 4: 源码架构走读（step 三段式、slot_mapping、V0/V1）
├── kernels/                     # 可运行示例（规划中）
│   └── quickstart.py            # Day 1: 最小离线推理示例
├── notes/                       # 走读笔记（规划中）
│   ├── paged_attention.md       # Day 2: block table 图解
│   ├── scheduler.md             # Day 3: 调度器状态机
│   └── source_walkthrough.md    # Day 4: 源码调用链
└── benchmark/                   # 压测脚本与结果（规划中）
    └── serving_bench.md         # Day 7: TTFT/TPOT/吞吐报告
```

> 💡 **后续延伸**：完成本专题后，可以沿着三条线继续深入：① **算子层**——回到 [Week 5](../../daily/week5/README.md) 把 PagedAttention kernel 的 CUDA 实现吃透；② **系统层**——读 PD 分离（prefill/decode disaggregation）与 KV Cache 跨节点传输（LMCache、Mooncake）；③ **对标框架**——对比 SGLang（RadixAttention）与 TensorRT-LLM，理解设计取舍。面试前过一遍 [Week 8 面试准备](../../daily/week8/README.md)，把本专题的产出浓缩成应答材料。
