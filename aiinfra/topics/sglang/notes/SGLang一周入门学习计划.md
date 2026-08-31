# SGLang 一周入门学习计划（2026 版）

> 适用版本：SGLang v0.5.x（撰写时最新稳定版为 v0.5.18，2026 年 8 月）
> 每天投入：2～3 小时 | 理论 ≤ 40%，实践 ≥ 60%
> 目标：建立完整认知框架 + 跑通 SGLang + 理解核心机制（而不是读完所有源码）

---

# 一、SGLang 学习知识地图

先建立全局视图，知道自己在学什么、学到哪一层：

```
LLM（大语言模型，如 Qwen / Llama / DeepSeek）
│
├── 训练（Training）—— 本计划不涉及
│
└── 推理（Inference）—— 用训练好的模型生成文本
    │
    │   核心矛盾：推理 = 每次只生成 1 个 token，GPU 算力大但"喂不饱"
    │   → 推理通常是 Memory Bound（受显存带宽限制），不是 Compute Bound
    │
    ├── 推理的两阶段
    │   ├── Prefill（预填充）：一次性处理整个 prompt，计算量大
    │   └── Decode（解码）：逐 token 生成，逐个字"蹦"出来
    │
    ├── 推理引擎（Inference Engine）—— 解决"怎么让 GPU 高效跑推理"
    │   │   通用技术：KV Cache、Continuous Batching、Paged KV 管理、调度器
    │   │
    │   ├── vLLM        —— 生态最广、模型支持最多（PagedAttention 发源地）
    │   ├── TensorRT-LLM —— NVIDIA 官方，极致性能但要编译
    │   ├── TGI / llama.cpp / Ollama 等
    │   └── SGLang ⭐ 本计划主角
    │       │
    │       ├── 定位：高性能 LLM 服务引擎 + 结构化生成框架
    │       ├── 出身：UC Berkeley / 斯坦福 LMSYS 团队（2024 年论文，NeurIPS 2024）
    │       ├── 生产采用：xAI（Grok）、DeepSeek 官方推荐、NVIDIA、AMD
    │       │
    │       └── 核心模块
    │           ├── RadixAttention     —— 基数树前缀缓存（跨请求 KV 复用，默认开启）
    │           ├── Continuous Batching —— 连续批处理 + 零开销调度器
    │           ├── Chunked Prefill     —— 长 prompt 分块，不阻塞 decode
    │           ├── KV Cache 管理       —— 分页显存池 + LRU 淘汰
    │           ├── OpenAI Compatible API —— 对外提供标准 HTTP 服务
    │           ├── 结构化输出（xgrammar）—— JSON/正则约束解码
    │           └── 分布式能力          —— TP（张量并行）、PD 分离、EP（专家并行）
    │
    └── 性能指标体系（衡量推理引擎好坏的尺子）
        ├── TTFT   —— 首 token 延迟（用户等多久看到第一个字）
        ├── TPOT   —— 每个输出 token 的间隔（生成流不流畅）
        └── Throughput —— 吞吐（整服务每秒处理多少 token）
```

**一句话理解 SGLang 的位置**：模型是发动机，推理引擎是变速箱+底盘。SGLang 是一台"特别擅长复用重复计算（前缀缓存）"的高性能变速箱。

---

# 二、7 天完整学习计划

## Day 1：认识 SGLang —— 它是什么、解决什么问题

### 1. 今日学习目标

学完今天，你应该能用自己的话回答："为什么直接用 PyTorch + Transformers 跑大模型推理不行？为什么需要推理引擎？SGLang 和 vLLM 有什么不同？"

### 2. 核心知识点

**① 什么是 SGLang**
- 是什么：一个开源的高性能 LLM 推理服务框架（serving framework），提供 HTTP API 服务 + 一套结构化生成的前端语言。
- 为什么需要它：直接用 HuggingFace `model.generate()` 跑推理，一次只处理一个请求、KV 显存浪费严重、吞吐量极低，无法用于真实服务。
- 解决什么问题：把"模型权重"变成"一个高并发、低延迟的在线推理服务"。
- 在 SGLang 中的作用：这就是本体。

**② SGLang 的发展背景**
- 是什么：2024 年初由 LMSYS Org（UC Berkeley / 斯坦福 Sky Computing Lab）发布，论文《SGLang: Efficient Execution of Structured Language Model Programs》被 NeurIPS 2024 接收。
- 为什么需要了解：它的设计动机是"真实 LLM 应用（Agent、多轮对话、RAG）中存在大量重复的 prompt 前缀，传统引擎每次都在重复计算"。
- 解决什么问题：从"单次问答"到"复杂 LLM 程序"的执行效率问题。
- 在 SGLang 中的作用：解释了 RadixAttention 为什么是它的一等公民特性。

**③ SGLang 解决什么问题（推理引擎存在的意义）**
- 是什么：GPU 很贵，模型推理时 GPU 经常"吃不饱"（利用率低）。推理引擎通过批处理、显存管理、调度把 GPU 利用率拉满。
- 它解决：吞吐低、延迟高、显存浪费三大问题。

**④ SGLang 在 LLM 推理系统中的位置**
- 是什么：位于"模型权重"之上、"应用代码"之下的中间层。应用只发 HTTP 请求，不用关心 GPU 细节。

**⑤ SGLang 与 vLLM 的关系和区别**
- 是什么：两者都是主流开源推理引擎，表层能力已趋同（都支持 Continuous Batching、分页 KV、FP8 量化、OpenAI API）。
- 区别核心：
  - vLLM：PagedAttention + block 级前缀缓存（APC，需 `--enable-prefix-caching` 开启），模型覆盖最广（400+ 架构），是"安全默认项"。
  - SGLang：RadixAttention 基数树 token 级前缀缓存（默认开启，零配置），高前缀重叠场景（RAG、多轮 Agent、固定 system prompt）吞吐可高出 20%～40%；结构化输出（xgrammar 约束解码）开销更低；MoE 大模型（DeepSeek 系）服务表现一流。
- 工程结论：prompt 前缀重叠率 > 60% 优先试 SGLang；追求模型覆盖广度和 Blackwell 新硬件首发支持选 vLLM。

### 3. 推荐学习顺序

1. 先理解问题：写一个最朴素的 PyTorch 推理循环，感受"逐 token 生成"有多慢（见实践任务）。
2. 再理解概念：推理引擎要解决什么（吞吐/延迟/显存）。
3. 再阅读文档：浏览 SGLang 官方文档首页和 GitHub README（docs.sglang.ai / github.com/sgl-project/sglang），只看概览，不深入。
4. 最后动手：跑一遍 SGLang 官方 README 里的 Quick Start（哪怕只是看懂命令含义）。

### 4. 动手实践任务

**任务 A：感受"裸推理"（30 分钟）**

在有 GPU 的机器上，用最原始的方式跑一次生成，为之后理解"引擎优化了什么"建立体感：

```python
# naive_generate.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "Qwen/Qwen3-0.6B"   # 显存紧张就用这个小模型
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="auto")

prompt = "用三句话解释什么是 KV Cache："
inputs = tok(prompt, return_tensors="pt").to("cuda")

t0 = time.time()
out = model.generate(**inputs, max_new_tokens=100)
t1 = time.time()

print(tok.decode(out[0], skip_special_tokens=True))
print(f"生成 100 个 token 耗时：{t1 - t0:.2f} 秒")
```

预期结果：能生成通顺文本，记录耗时（通常在数秒级）。**把这个耗时记下来，Day 6 做 Benchmark 时对比。**

**任务 B：浏览式侦察（30 分钟）**
- 打开 GitHub `sgl-project/sglang`，记录：star 数、最近 release 版本号、README 中的安装命令。
- 打开官方文档的 "Get Started" 页面，把启动命令抄到笔记里（明天要用）。

### 5. 学习时间安排（共 2.5 小时）

| 时长 | 内容 |
|---|---|
| 40 分钟 | 理论：推理引擎为什么存在、SGLang 背景 |
| 30 分钟 | 实践 A：裸推理脚本 |
| 50 分钟 | 阅读：SGLang README + 文档首页 + vLLM 对比 |
| 30 分钟 | 实践 B：侦察记录 + 写一页"我理解的 SGLang"总结 |

### 6. 今日必须掌握

1. SGLang 是推理引擎/服务框架，不是模型、不是训练框架。
2. 推理引擎存在的三大理由：吞吐、延迟、显存利用率。
3. SGLang 出身 LMSYS，论文核心贡献是 RadixAttention。
4. SGLang vs vLLM 的核心差异点：前缀缓存机制（radix tree vs block hash）与默认开启与否。

### 7. 今日检查题

1. （概念理解）为什么 `model.generate()` 循环生成 100 个 token 要执行 100 次前向传播？能不能一次生成全部？
2. （原理分析）"推理是 Memory Bound"这句话如果成立，那么"把 batch 开大"为什么能提高吞吐而不明显增加延迟？
3. （工程问题）你的应用是客服机器人，所有请求共享一段 2000 token 的系统提示词。仅从这一点判断，SGLang 和 vLLM 哪个更值得先试？为什么？
4. （工程问题）如果领导问你"SGLang 和模型微调有什么关系"，你怎么用一句话纠正这个误解？

**📦 今日产出**：理解 SGLang 在 LLM 推理系统中的位置 + 一段可运行的裸推理基线脚本及耗时记录。

---

## Day 2：环境搭建 —— 安装并启动 SGLang

### 1. 今日学习目标

独立完成 SGLang 安装，成功启动一个模型推理服务进程，并能看到服务正常监听端口。今天的成败标准只有一个：**服务跑起来**。

### 2. 核心知识点

**① GPU 环境要求**
- 是什么：SGLang 需要 NVIDIA GPU（官方推荐 8GB+ 显存起步，学习用 0.6B～8B 小模型即可；也支持 AMD ROCm、TPU，但学习阶段建议 NVIDIA）。需要 Python 3.10+（当前 v0.5.x 搭配 CUDA 12.x/13.0 + PyTorch 2.x）。
- 为什么需要：模型权重和 KV Cache 都要放显存；显存决定你能跑多大的模型。
- 解决什么问题：避免"装完发现跑不动"的浪费。
- 在 SGLang 中的作用：`--mem-fraction-static` 等参数直接和显存打交道。

**② 安装方式（pip / uv / Docker / 源码）**
- 是什么：四种安装路径。学习首选 pip 或官方 Docker 镜像。
- 为什么需要它：SGLang 依赖 FlashInfer 等 CUDA 算子库，环境配错是新手最大的坑。
- 在 SGLang 中的作用：无（基础设施），但决定了后面 6 天顺不顺。

**③ Docker 使用方式**
- 是什么：官方维护镜像 `lmsysorg/sglang`（标签如 `v0.5.x-cu130-runtime`），一行命令拉起带全部依赖的环境。
- 为什么需要它：你已有 Docker 基础——这是最不容易踩依赖坑的方式，也是生产部署的真实形态。
- 解决什么问题：CUDA / PyTorch / FlashInfer 版本地狱。

**④ 启动模型 / 启动推理服务**
- 是什么：`python -m sglang.launch_server --model-path <模型> --port 30000`，SGLang 默认端口 30000。
- 为什么需要它：SGLang 的"服务化"形态——加载权重 → 预分配 KV 显存池 → 起 HTTP 服务。
- 解决什么问题：把模型变成一个网络服务。
- 在 SGLang 中的作用：这是后续所有天的操作入口。

### 3. 推荐学习顺序

1. 先检查硬件：`nvidia-smi` 确认 GPU 型号和显存。
2. 再选安装方式：有干净 GPU 机器 → pip；环境乱/想隔离 → Docker。
3. 安装 → 验证安装（import 测试）。
4. 下载一个小模型并启动服务。
5. 读懂启动日志（重点看显存分配和 KV cache 大小的输出）。

### 4. 动手实践任务

**任务 A：环境检查（10 分钟）**

```bash
nvidia-smi            # 确认 GPU、显存、驱动
python3 --version     # 3.10+
docker --version      # 如走 Docker 路线
```

**任务 B：安装 SGLang（二选一，40 分钟）**

方式一：pip（推荐用 uv 更快）：

```bash
pip install uv
uv pip install "sglang[all]" --system
# 验证
python -c "import sglang; print(sglang.__version__)"
```

方式二：Docker：

```bash
docker pull lmsysorg/sglang:latest
docker run --gpus all -it --rm \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest bash
```

预期结果：`import sglang` 不报错，版本号为 v0.5.x。

**任务 C：启动你的第一个 SGLang 服务（40 分钟）**

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000
```

预期结果：日志依次出现模型加载、`KV Cache is allocated. #tokens: xxxxxx`、`The server is fired up and ready to roll!`。**把 `#tokens` 这个数字抄下来**——Day 4 讲 KV Cache 时会用到。

**任务 D：服务健康检查（10 分钟）**

另开一个终端：

```bash
curl http://localhost:30000/health
curl http://localhost:30000/get_model_info
```

预期结果：health 返回 200，model_info 返回模型 JSON 信息。

**排障提示**：FlashInfer 相关报错 → 检查 CUDA 版本是否匹配；显存不够 → 换更小模型或加 `--mem-fraction-static 0.7`；端口被占 → 换 `--port`。

### 5. 学习时间安排（共 2.5 小时）

| 时长 | 内容 |
|---|---|
| 20 分钟 | 理论：环境要求、安装方式对比 |
| 90 分钟 | 实践：检查 → 安装 → 启动 → 健康检查（含踩坑时间） |
| 20 分钟 | 阅读：启动日志逐行读懂 |
| 20 分钟 | 总结：把完整启动命令和日志关键行整理进笔记 |

### 6. 今日必须掌握

1. `python -m sglang.launch_server` 是启动服务的标准入口。
2. SGLang 默认端口 30000，`/health` 是健康检查端点。
3. 启动时 SGLang 会预分配一大块显存给 KV Cache 池。
4. pip 与 Docker 两种安装方式各自的适用场景。

### 7. 今日检查题

1. （概念理解）为什么 SGLang 启动时要"预分配"KV Cache 显存，而不是按需向 CUDA 申请？（提示：想想 cudaMalloc 的开销和碎片化）
2. （原理分析）`--mem-fraction-static 0.9` 和 `0.7` 分别意味着什么？调低它会牺牲什么、换来什么？
3. （工程问题）服务启动报 `CUDA out of memory`，但 `nvidia-smi` 显示显存还有空余，可能是什么原因？（至少说出两种）
4. （工程问题）如果让你在没有公网的内网机器上部署 SGLang，整个交付物应该包含哪些东西？

**📦 今日产出**：成功启动 SGLang 服务 + 一份记录启动命令、日志关键行（含 KV cache tokens 数）的环境笔记。

---

## Day 3：基本使用 —— API 调用与 OpenAI Compatible API

### 1. 今日学习目标

熟练使用三种方式调用 SGLang：curl 原生 API、OpenAI SDK、Python 离线 Engine。理解 OpenAI Compatible API 为什么成为行业标准。

### 2. 核心知识点

**① HTTP 推理 API（原生端点）**
- 是什么：SGLang 的原生端点 `/generate`，直接传 `text` + `sampling_params`。
- 为什么需要它：最贴近引擎能力的接口，参数最全（top_p、temperature、max_new_tokens 等）。
- 解决什么问题：程序如何"问"模型问题。
- 在 SGLang 中的作用：服务的主入口之一。

**② OpenAI Compatible API**
- 是什么：SGLang 在 `/v1` 下实现了与 OpenAI 官方完全兼容的接口：`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/v1/embeddings`、`/v1/responses`。
- 为什么需要它：行业标准。任何为 OpenAI 写的客户端代码（LangChain、各种 SDK），只需改 `base_url` 和 `api_key` 就能切到自部署模型。
- 解决什么问题：供应商锁定 + 迁移成本。
- 在 SGLang 中的作用：对外兼容层——这也是 SGLang 与 vLLM 可以"无缝互换"的原因。

**③ Sampling 参数**
- 是什么：`temperature`（随机性）、`top_p`（核采样）、`max_new_tokens`（生成长度上限）等。
- 为什么需要它：同一模型，不同参数 = 不同产品性格（客服要低 temperature，创意写作要高）。
- 在 SGLang 中的作用：每次请求都可独立指定。

**④ Python 离线 Engine（sglang.Engine）**
- 是什么：不起 HTTP 服务，直接在 Python 进程内 `sglang.Engine(model_path=...)` 批量推理。
- 为什么需要它：离线批处理（数据清洗、评测、批量生成）不需要服务化开销。
- 解决什么问题：离线场景 vs 在线服务的分工。
- 在 SGLang 中的作用：同一套引擎，两种使用形态。

### 3. 推荐学习顺序

1. 先概念：请求 → tokenize → 推理 → 返回，一次 API 调用经历了什么。
2. 再实践 curl：最接近"线上正在发生什么"。
3. 再 OpenAI SDK：最接近真实应用写法。
4. 最后 Engine：理解在线/离线两种形态。

### 4. 动手实践任务

**任务 A：curl 调用原生 API（20 分钟）**

```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "北京是中国的首都，上海是",
    "sampling_params": {"max_new_tokens": 32, "temperature": 0.7}
  }'
```

预期结果：返回 JSON，含续写文本和 `meta_info`（其中有 prompt/completion token 数——明天讲 TTFT 会用）。

**任务 B：OpenAI SDK 调用（30 分钟）**

```bash
pip install openai
```

```python
# chat_client.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",   # 与你启动时的 model-path 一致
    messages=[
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "用一句话解释什么是 Continuous Batching。"},
    ],
    temperature=0.7,
    max_tokens=128,
)
print(resp.choices[0].message.content)
print("用量：", resp.usage)
```

预期结果：正常返回回答 + token 用量统计。

**任务 C：流式输出（20 分钟）**

把任务 B 的调用加上 `stream=True`，逐 chunk 打印：

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "讲一个 50 字的小故事"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

预期结果：文字像打字机一样逐段出现。**留意"第一个字出现得很快、后续匀速"——这就是 TTFT + TPOT 的体感。**

**任务 D：离线 Engine 批量推理（30 分钟）**

```python
# offline_engine.py
import sglang as sgl

llm = sgl.Engine(model_path="Qwen/Qwen3-0.6B")

prompts = [
    "中国的首都是",
    "法国的货币是",
    "1 + 1 =",
] * 10   # 30 个请求，观察批量处理速度

outs = llm.generate(prompts, {"max_new_tokens": 16, "temperature": 0})
for o in outs[:3]:
    print(o["text"])
llm.shutdown()
```

预期结果：30 个请求几乎"同时"完成，比逐个调用快一个数量级。**这个加速来自 Continuous Batching——Day 4 的原理主角。**

### 5. 学习时间安排（共 2.5 小时）

| 时长 | 内容 |
|---|---|
| 30 分钟 | 理论：API 形态、OpenAI 兼容的意义、sampling 参数 |
| 100 分钟 | 实践：任务 A～D |
| 20 分钟 | 阅读：官方文档 "OpenAI Compatible API" 章节，看看还支持哪些端点 |

### 6. 今日必须掌握

1. `/generate` 原生端点与 `/v1/chat/completions` 的区别。
2. OpenAI SDK 对接自部署服务只需改 `base_url`。
3. `temperature` / `top_p` / `max_tokens` 三个参数的作用。
4. 在线服务（launch_server）vs 离线批处理（sgl.Engine）的选用。

### 7. 今日检查题

1. （概念理解）为什么流式输出时"第一个 token"和"后续 token"的等待感受完全不同？背后对应推理的哪两个阶段？
2. （原理分析）任务 D 中 30 个请求批量完成远快于逐个串行。如果 batch 无限大，吞吐会无限涨吗？瓶颈会在哪里出现？
3. （工程问题）线上服务要同时接"实时聊天"和"每晚批量生成日报"两种业务，你会怎么部署？（提示：几种形态、几份权重）
4. （工程问题）一个为 OpenAI GPT 写的应用想切换到本地模型，除了 `base_url` 还可能有哪些不兼容点？（提示：模型名、function calling、上下文长度）

**📦 今日产出**：成功通过 curl / OpenAI SDK / 流式 / 离线 Engine 四种方式调用 SGLang + 一份可复用的 `chat_client.py`。

---

## Day 4：LLM 推理基础 —— Prefill / Decode / KV Cache / Continuous Batching

### 1. 今日学习目标

理解推理引擎的"通用内功"：这些概念不属于 SGLang 独有，但不理解它们就无法理解 SGLang 的任何优化。学完后能画出一次推理请求的完整生命周期。

### 2. 核心知识点

**① Token**
- 是什么：模型处理文本的最小单位（约 0.5～1 个汉字 / 0.75 个英文单词）。
- 为什么需要它：模型不看"字"，看 token id 序列；计费和性能指标都按 token 算。

**② Prefill（预填充）**
- 是什么：收到 prompt 后，一次性并行计算 prompt 所有 token 的注意力，生成第一个输出 token。
- 特点：计算密集型（Compute Bound）——prompt 越长越耗时，直接决定 TTFT。

**③ Decode（解码）**
- 是什么：基于已有上下文，逐 token 自回归生成，一次前向只出 1 个 token。
- 特点：显存带宽密集型（Memory Bound）——每步都要把全部权重从显存搬一遍，却只算一点点。这是推理优化一切故事的起点。

**④ KV Cache**
- 是什么：注意力计算中每个 token 的 Key/Value 矩阵缓存起来，后续 token 直接复用，不用重算历史。
- 为什么需要它：没有它，生成第 n 个 token 要重算前 n-1 个的注意力，复杂度爆炸。
- 代价：占显存（这就是 Day 2 日志里 "KV Cache is allocated. #tokens" 的含义——这块池子能容纳多少个 token 的 KV）。

**⑤ Batch（批处理）**
- 是什么：把多个请求拼在一起过一遍 GPU。
- 为什么需要它：decode 阶段是 Memory Bound，搬一次权重算 1 个请求和算 64 个请求，耗时几乎一样——batch 是"免费"提升吞吐的手段。

**⑥ Continuous Batching（连续批处理）**
- 是什么：传统静态批处理要等整批全部生成完才能放新请求进来（最短的请求等最长的）；Continuous Batching 在**每一步 decode 之后**都重新调度——完成的请求立即离开，等待的请求立即插入。
- 类比：静态批处理像班车（人齐才发）；Continuous Batching 像流水线传送带（随来随上、完成即下）。
- 解决什么问题：GPU 空转 + 请求排队，吞吐可提升数倍到十几倍。

**⑦ Memory Bound vs Compute Bound**
- 是什么：程序瓶颈在"搬数据"（带宽）还是"算数据"（算力）。
- 在推理中的映射：prefill 偏 Compute Bound，decode 偏 Memory Bound——两个阶段性质相反，这是后来"PD 分离"等高级架构的动机（了解即可）。

### 3. 推荐学习顺序

1. 先理解问题：decode 一次只出一个 token，GPU 大部分时间在"搬权重"。
2. 再理解概念：KV Cache 省重算 → Batch 摊薄搬运成本 → Continuous Batching 消除等待。
3. 再阅读：找一张 prefill/decode 示意图（SGLang 文档或任何推理引擎科普），对着图复述。
4. 最后动手：用实验"看见"这些概念（见下）。

### 4. 动手实践任务

**任务 A：用日志"看见"prefill 和 decode（40 分钟）**

写一个脚本，分别发送"短 prompt + 长生成"和"长 prompt + 短生成"两种请求，对比耗时结构：

```python
# observe_phases.py
import time, requests

def timed(prompt, max_new_tokens, label):
    t0 = time.time()
    r = requests.post("http://localhost:30000/generate", json={
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }, stream=True)
    first_token_time = None
    for i, line in enumerate(r.iter_lines()):
        if line and first_token_time is None:
            first_token_time = time.time()
    t1 = time.time()
    meta = r.json() if False else None  # stream 模式下看行数据
    print(f"[{label}] TTFT≈{(first_token_time or t1)-t0:.3f}s, 总耗时≈{t1-t0:.3f}s")

timed("你好", 200, "短prompt+长生成")
timed("请阅读以下文本：" + "人工智能" * 500, 8, "长prompt+短生成")
```

预期结果：短 prompt 场景 TTFT 极小、总时长被 decode 主导；长 prompt 场景 TTFT 明显变大。**这就是 prefill 与 decode 成本结构的直接证据。**

**任务 B：观察 Continuous Batching（40 分钟）**

同时发 8 个"生成长度各不相同"的请求（max_new_tokens 分别为 16/32/64/128…），用 `asyncio` + `aiohttp` 并发发出，记录每个请求的完成时间：

```python
# observe_batching.py
import asyncio, aiohttp, time

async def one(session, n):
    t0 = time.time()
    async with session.post("http://localhost:30000/generate", json={
        "text": "写一首诗：",
        "sampling_params": {"max_new_tokens": n, "temperature": 0.8},
    }) as r:
        await r.read()
    return n, time.time() - t0

async def main():
    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*[one(s, n) for n in [16, 32, 64, 128, 16, 64, 32, 128]])
        for n, dt in results:
            print(f"max_new_tokens={n:4d}  耗时 {dt:.2f}s")

asyncio.run(main())
```

预期结果：16-token 的请求**不会**等 128-token 的请求做完才返回，各自按自身长度先后完成。如果是静态批处理，短请求会被拖到和最长请求一样慢。

**任务 C：估算你的 KV Cache 池（20 分钟）**

回看 Day 2 记录的 `#tokens`，用模型 context length 估算：`#tokens ÷ 4096 ≈ 同时容纳多少条满长对话`。写在笔记里。

### 5. 学习时间安排（共 3 小时）

| 时长 | 内容 |
|---|---|
| 60 分钟 | 理论：7 个核心概念（今天理论最重，但明天开始回落） |
| 100 分钟 | 实践：任务 A～C |
| 20 分钟 | 总结：手绘一张"一次请求在引擎内的旅程"图 |

### 6. 今日必须掌握

1. Prefill（并行、算力密集、决定 TTFT）与 Decode（串行、带宽密集、决定 TPOT）的区别。
2. KV Cache 缓存的是什么、为什么能省计算、代价是显存。
3. Continuous Batching 在"每一步"调度，而不是"每一批"调度。
4. Memory Bound 的含义及其对"batch 越大吞吐越高"的解释力。

### 7. 今日检查题

1. （概念理解）为什么 prefill 可以并行处理整个 prompt，而 decode 必须逐 token 串行？（提示：自回归依赖）
2. （原理分析）如果关掉 KV Cache，生成 100 个 token 的计算量大约变成原来的多少倍？（量级估算即可）
3. （原理分析）Continuous Batching 中，一个请求生成完毕后，它释放的显存和算力立刻给谁用？调度器需要知道哪些信息才能做插入决策？
4. （工程问题）你的服务白天是聊天（短 prompt 长生成），夜里是文档摘要（长 prompt 短生成）。两类负载的性能瓶颈分别在哪里？能用同一组引擎参数同时优化两者吗？

**📦 今日产出**：理解 KV Cache 和 Continuous Batching + 两个可运行的观察实验脚本及其结果记录。

---

## Day 5：SGLang 核心机制 —— RadixAttention 与调度优化

### 1. 今日学习目标

理解 SGLang 区别于其他引擎的"独门武功"：RadixAttention / Prefix Cache / Chunked Prefill / 零开销调度。每个机制按"问题 → 传统方案 → 问题 → SGLang 方案 → 收益来源"的链条理解。

### 2. 核心知识点

**① RadixAttention（今天的主角）**

- **问题是什么**：真实应用（多轮对话、RAG、Agent）中，大量请求共享相同前缀（system prompt、知识库文档、工具定义）。传统引擎每个请求都从第 0 个 token 重算。
- **传统方案怎么做**：不缓存（全量重算），或 vLLM APC 那样按固定 block 哈希匹配（block 未对齐就失配）。
- **存在什么问题**：重算浪费算力、拉高 TTFT；block 级匹配粒度粗，前缀差一个 token 就可能整块失配。
- **SGLang 如何解决**：用一棵**基数树（Radix Tree）**组织所有请求的 KV Cache——每个节点是一段 token 序列，从根到叶是一条完整前缀。新请求到来时在树上做**最长前缀匹配**：命中部分直接复用 KV，只 prefill 不命中的尾部；请求结束后把新 KV 插回树中，LRU 自动淘汰冷节点。token 级粒度、默认开启、零配置。
- **性能收益来自哪里**：省掉的是共享前缀的 prefill 计算。论文报告共享前缀负载下最高 6.4× 加速；工程实测中，前缀重叠率 60% 以上时 TTFT 降低 20%～40%，RAG / 多轮 Agent 场景吞吐提升 20%～40%。
- **关键直觉**：512 token 的 system prompt，1000 个并发请求只算 1 次。

**② Prefix Cache（前缀缓存）**
- 是什么：RadixAttention 的用户侧体现。`/metrics` 端点里有 `cache_hit_rate` 可以直接观测。
- 在 SGLang 中的作用：默认开启，可用 `--disable-radix-cache` 关闭做对照实验（今天的实践要用）。

**③ Continuous Batching 在 SGLang 中的实现：零开销调度器**
- 问题：调度器跑在 CPU 上，如果调度耗时赶上 GPU 一步 decode 的耗时，GPU 就要等 CPU——"调度开销"吃掉 batch 收益。
- SGLang 方案：把调度与 GPU 前向传播**重叠（overlap）**——GPU 跑第 n 步时，CPU 同时准备第 n+1 步的 batch。高并发下调度开销趋近于零。
- 收益来源：GPU 利用率接近 100%，高并发场景下相对其他引擎有稳定优势。

**④ Chunked Prefill（分块预填充）**
- 问题：一个 32K token 的长 prompt 进来，prefill 要算很久，期间 decode 全部卡住——所有在线用户都感觉到"卡顿"。
- 传统方案：要么忍受卡顿，要么 prefill 和 decode 分机器（PD 分离，成本高）。
- SGLang 方案：把长 prefill 切成若干 chunk，与 decode 请求**混合编批**：每一步既推进若干 decode，又消化一块 prefill。
- 收益来源：TPOT 不再被长 prompt 周期性打爆，延迟曲线变平滑。

**⑤ 调度机制 & KV Cache 管理总览**
- 调度器：决定"这一步跑哪些请求"，策略受 cache 命中影响（cache-aware——优先跑能命中前缀缓存的请求）。
- KV 管理：分页显存池 + 基数树索引 + 引用计数 + LRU 淘汰，多轮对话的历史 KV 在轮次间自然保留。

### 3. 推荐学习顺序

1. 先理解问题：看一段多轮对话的 prompt 结构，标出每轮重复的部分。
2. 再理解概念：基数树（前缀树）数据结构——画一棵只有 3 个请求的小树。
3. 再阅读：SGLang 博客/论文中 RadixAttention 的图（搜 "SGLang RadixAttention blog"）。
4. 最后动手：做对照实验亲眼看到缓存命中带来的差异。

### 4. 动手实践任务

**任务 A：多轮对话前缀复用实验（50 分钟）**

模拟 5 轮对话，每轮把完整历史发给服务，观察每轮 TTFT 的变化：

```python
# multi_turn_cache.py
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
history = [{"role": "system", "content": "你是一名历史老师。" + "请严谨作答。" * 100}]  # 故意加长前缀

questions = ["讲讲秦朝。", "它怎么灭亡的？", "和隋朝有什么共同点？", "对后世影响最大的是什么？", "用一句话总结。"]

for i, q in enumerate(questions, 1):
    history.append({"role": "user", "content": q})
    t0 = time.time()
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B", messages=history, max_tokens=64, temperature=0)
    t1 = time.time()
    history.append({"role": "assistant", "content": resp.choices[0].message.content})
    print(f"第{i}轮: prompt_tokens={resp.usage.prompt_tokens:5d}, 耗时={t1-t0:.2f}s")
```

预期结果：**prompt_tokens 每轮递增，但耗时并不同比变长**——因为历史前缀的 KV 全部命中了缓存。第 5 轮的 prompt 可能是第 1 轮的几倍长，耗时却接近。

**任务 B：开/关 Radix Cache 对照实验（40 分钟）**

1. 重启服务并关闭缓存：`python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --disable-radix-cache`
2. 重跑任务 A 的脚本，记录每轮耗时。
3. 恢复正常启动，再跑一遍。
4. 画一个两行对比表，计算缓存带来的加速比。

预期结果：关缓存后，后几轮耗时随历史变长显著上升；开缓存则基本持平。

**任务 C：看 metrics（20 分钟）**

```bash
curl http://localhost:30000/metrics | grep -i cache
```

预期结果：看到 `sglang_cache_hit_rate` 等指标。**学会用指标说话，是 Day 6 Benchmark 的基础。**

### 5. 学习时间安排（共 3 小时）

| 时长 | 内容 |
|---|---|
| 50 分钟 | 理论：RadixAttention 五段式 + Chunked Prefill + 调度器 |
| 110 分钟 | 实践：任务 A～C |
| 20 分钟 | 总结：用自己的话写一段"RadixAttention 为什么快"（不超过 200 字） |

### 6. 今日必须掌握

1. RadixAttention = 基数树 + 最长前缀匹配 + LRU 淘汰，token 级粒度、默认开启。
2. 它省的是**共享前缀的 prefill 计算**，直接收益是 TTFT，间接收益是吞吐。
3. Chunked Prefill 解决"长 prompt 阻塞 decode"的卡顿问题。
4. 零开销调度器 = CPU 调度与 GPU 计算重叠。
5. 会用 `--disable-radix-cache` 和 `/metrics` 做机制验证。

### 7. 今日检查题

1. （概念理解）为什么 Radix Tree 比"哈希表存固定 block"更适合多轮对话场景？考虑第 3 轮对话在历史末尾追加新内容的模式。
2. （原理分析）RadixAttention 的收益大小取决于什么负载特征？什么情况下它几乎零收益（但也几乎零损失）？
3. （原理分析）KV Cache 池满了会发生什么？LRU 淘汰树上节点时，为什么不能随便删中间节点？（提示：想想子节点依赖）
4. （工程问题）你的 RAG 系统每个请求都把检索到的文档放在 prompt 最前面、system prompt 放在最后，缓存命中率很低。怎么调整 prompt 结构来提升命中率？为什么？

**📦 今日产出**：理解 RadixAttention + 一份"开/关前缀缓存"的对照实验数据表。

---

## Day 6：性能与工程实践 —— 指标与 Benchmark

### 1. 今日学习目标

掌握推理服务的性能语言（TTFT / TPOT / Throughput），会用官方工具跑一个规范的 Benchmark，并理解 batch size、并发、前缀缓存三个变量对性能的影响。

### 2. 核心知识点

**① TTFT（Time To First Token）**
- 是什么：从发出请求到收到第一个 token 的时间。 = 排队时间 + prefill 时间。
- 为什么重要：直接决定用户"等多久开始有反应"。前缀缓存主要优化它。

**② TPOT（Time Per Output Token）**
- 是什么：decode 阶段平均每 token 的间隔。决定"打字机"流不流畅。
- 相关指标 ITL（Inter-Token Latency）：相邻 token 的间隔分布。

**③ Throughput（吞吐量）**
- 是什么：单位时间处理的 token 数，分 input/output 或合计（tok/s）。服务的"产能"。
- 为什么重要：GPU 是固定成本，吞吐越高 = 每 token 成本越低。

**④ 延迟 vs 吞吐的权衡**
- 是什么：batch 开大 → 吞吐升、单请求延迟也升。服务调优本质是在这条曲线上选工作点。

**⑤ Benchmark 方法学**
- 是什么：固定模型、固定数据集（输入/输出长度分布）、阶梯式并发，测量各指标的分位数（p50/p95/p99）。
- 为什么需要它："我感觉挺快"不是工程语言；"c=50 时 TTFT p95=340ms"才是。

### 3. 推荐学习顺序

1. 先概念：三个指标各自衡量什么、互相什么关系。
2. 再工具：学会 `sglang.bench_serving` 的参数。
3. 再实验：一次只改一个变量（并发 / 缓存 / 生成长度）。
4. 最后解读：把数字翻译成工程结论。

### 4. 动手实践任务

**任务 A：官方 Benchmark 工具初体验（40 分钟）**

```bash
python -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --model Qwen/Qwen3-0.6B \
  --host localhost --port 30000 \
  --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 100 \
  --request-rate 8
```

预期结果：输出一份报告，含 Successful requests、Mean/P99 TTFT、Mean TPOT、Output token throughput 等。**把整份报告存成文件。**

**任务 B：并发阶梯实验（50 分钟）**

把 `--request-rate` 分别设为 1 / 4 / 16 / 64，各跑一次，填表：

| 并发 | TTFT p50 | TPOT p50 | 总吞吐 (tok/s) |
|---|---|---|---|
| 1 | | | |
| 4 | | | |
| 16 | | | |
| 64 | | | |

预期结果与结论：吞吐随并发上升但趋于饱和；TTFT/TPOT 随并发恶化。**找到你这台机器的"拐点"。**

**任务 C：前缀缓存对 Benchmark 的影响（40 分钟）**

1. 用共享前缀数据集跑一遍：`--dataset-name generated-shared-prefix`（工具内置，模拟大量共享前缀的请求）。
2. 关 radix cache 重启服务，再跑一遍。
3. 对比两次的 TTFT 均值。

预期结果：开缓存时 TTFT 显著更低——这是 Day 5 理论在标准负载下的量化验证。

**任务 D（可选加分）：与 Day 1 裸推理对比（10 分钟）**

回看 Day 1 的 100-token 生成耗时，对比今天在服务上同模型的单请求生成速度，体会引擎优化的量级。

### 5. 学习时间安排（共 3 小时）

| 时长 | 内容 |
|---|---|
| 30 分钟 | 理论：指标体系 + benchmark 方法学 |
| 140 分钟 | 实践：任务 A～C（+D） |
| 10 分钟 | 总结：把三张表整理成一页"性能观察报告" |

### 6. 今日必须掌握

1. TTFT / TPOT / Throughput 的定义和各自的影响因素。
2. `sglang.bench_serving` 的基本用法。
3. 并发升高时"吞吐升、延迟升"的权衡曲线。
4. 前缀缓存命中如何体现在 TTFT 上。

### 7. 今日检查题

1. （概念理解）一个服务 TTFT 很好但 TPOT 很差，瓶颈更可能在哪里？反过来呢？
2. （原理分析）为什么吞吐随并发上升最终会饱和甚至下降？饱和点由什么资源决定？
3. （原理分析）前缀缓存命中为什么主要改善 TTFT 而不是 TPOT？
4. （工程问题）老板要求"P99 TTFT < 500ms，同时吞吐最大化"。你会如何设计实验找到满足条件的最大并发？

**📦 今日产出**：完成一次完整性能测试 + 一页含并发阶梯表和缓存对照表的性能观察报告。

---

## Day 7：Mini Project —— 部署一个带性能报告的推理服务

### 1. 今日学习目标

把前 6 天的所有技能串成一个完整项目：**用 SGLang 部署开源模型，提供 OpenAI Compatible API，带 Python 客户端、并发测试和性能报告。**

### 2. 项目说明

**项目名称**：`mini-llm-service` —— 一个最小但完整的 LLM 推理服务实践。

**需求清单**：
1. 环境准备（Docker 或 pip）
2. 模型启动（SGLang 服务）
3. API 调用（OpenAI 兼容）
4. Python Client（封装好的客户端类）
5. 多请求并发测试（asyncio 压测）
6. 简单性能测试（TTFT/TPOT/吞吐统计）
7. 结果分析（一页报告）

### 3. 项目目录结构

```
mini-llm-service/
├── README.md               # 项目说明：环境、启动方式、测试结果
├── requirements.txt        # sglang, openai, aiohttp
├── scripts/
│   └── start_server.sh     # 启动服务（含参数说明注释）
├── client/
│   ├── __init__.py
│   └── llm_client.py       # 封装的客户端类
├── tests/
│   ├── test_basic.py       # 基础调用冒烟测试
│   ├── test_concurrent.py  # 并发测试
│   └── benchmark.py        # 简易性能测试（自写，不依赖官方工具）
└── report/
    └── RESULTS.md          # 性能数据 + 分析结论
```

### 4. 推荐代码结构

**`client/llm_client.py`**：

```python
from openai import OpenAI

class LLMClient:
    def __init__(self, base_url="http://localhost:30000/v1",
                 model="Qwen/Qwen3-0.6B"):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model

    def chat(self, messages, temperature=0.7, max_tokens=256, stream=False):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=stream)
        if stream:
            return resp  # 生成器
        return resp.choices[0].message.content, resp.usage

    def multi_turn(self, session_history: list, user_input: str, **kw):
        """多轮对话：自动维护历史，触发前缀缓存"""
        session_history.append({"role": "user", "content": user_input})
        content, usage = self.chat(session_history, **kw)
        session_history.append({"role": "assistant", "content": content})
        return content, usage
```

**`tests/benchmark.py`**（自写简易版，测 TTFT/TPOT）：

```python
import asyncio, time, statistics
import aiohttp

URL = "http://localhost:30000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"

async def one_request(session, prompt, max_tokens=128):
    t0 = time.time()
    ttft, tokens = None, 0
    async with session.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": True,
    }) as r:
        async for line in r.content:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.time() - t0
                tokens += 1
    total = time.time() - t0
    tpot = (total - ttft) / max(tokens - 1, 1)
    return ttft, tpot, tokens, total

async def run(concurrency, n_requests):
    prompts = ["介绍一下西安。"] * n_requests   # 相同 prompt → 高前缀命中
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as s:
        async def bounded(p):
            async with sem:
                return await one_request(s, p)
        t0 = time.time()
        results = await asyncio.gather(*[bounded(p) for p in prompts])
        wall = time.time() - t0
    ttfts = [r[0] for r in results]
    tpots = [r[1] for r in results]
    total_out = sum(r[2] for r in results)
    print(f"并发={concurrency:3d} | TTFT p50={statistics.median(ttfts)*1000:.0f}ms "
          f"| TPOT p50={statistics.median(tpots)*1000:.0f}ms "
          f"| 输出吞吐={total_out/wall:.0f} tok/s")

if __name__ == "__main__":
    for c in [1, 4, 16, 64]:
        asyncio.run(run(c, n_requests=c * 4))
```

### 5. 实现步骤（按序执行）

1. **环境准备**（15 分钟）：建目录、写 `requirements.txt`、创建虚拟环境或进 Docker 容器。
2. **模型启动**（15 分钟）：写 `start_server.sh`，内容就是你 Day 2 的启动命令，加上 `--enable-metrics`；启动并用 `/health` 验证。
3. **基础调用**（20 分钟）：实现 `llm_client.py`，写 `test_basic.py` 验证单轮、多轮、流式三种调用。
4. **并发测试**（30 分钟）：写 `test_concurrent.py`，20 个并发请求，断言全部成功返回。
5. **性能测试**（40 分钟）：跑 `benchmark.py` 的并发阶梯，再跑一遍官方 `sglang.bench_serving` 交叉验证。
6. **结果分析**（30 分钟）：写 `report/RESULTS.md`，包含：环境信息（GPU 型号/显存）、并发阶梯表、前缀缓存命中观察、三条结论。
7. **收尾**（10 分钟）：写 `README.md`，让别人能 5 分钟复现你的项目。

### 6. 测试方法

- 冒烟测试：单请求返回非空、usage 字段完整。
- 并发测试：20 并发全部 200，无超时。
- 性能验收：自写 benchmark 与官方工具的量级一致（不要求数字相同，差 2 倍以内说明测量口径没问题）。

### 7. 最终成果

- 一个可 `git clone → 装依赖 → 跑脚本` 复现的完整项目。
- 一份 RESULTS.md，能回答："这台机器上这个模型，多少并发时性价比最高？"
- 一张贯穿 7 天的完整学习闭环证明。

### 8. 学习时间安排（共 3 小时）

全部实践，无新增理论。理论不够的环节回查 Day 2～6 笔记。

### 9. 今日必须掌握（项目验收标准）

1. 不看笔记写出服务启动命令。
2. 不看笔记写出 OpenAI 兼容客户端。
3. 能解释项目里每一处和前 6 天哪个知识点对应。
4. 性能报告有数据、有对比、有结论。

### 10. 今日检查题

1. （综合）如果项目要支持 100 个用户同时多轮对话，你现在的单进程部署最先撞上什么瓶颈？怎么验证你的判断？
2. （综合）为什么 `benchmark.py` 里所有请求用相同 prompt 会高估真实性能？怎么改更接近真实负载？
3. （工程）要把这个项目部署到生产，至少还缺什么？（提示：鉴权、限流、监控、日志、多副本）

**📦 今日产出**：完成 Mini Project——代码仓库 + RESULTS.md 性能报告。

---

# 三、一周后你应该掌握什么

### 已经掌握（能独立操作 + 能给别人讲清楚）

- SGLang 是什么、解决什么问题、与 vLLM 的核心差异
- 安装、启动、健康检查、参数调整（mem-fraction、port、model-path）
- curl / OpenAI SDK / 流式 / 离线 Engine 四种调用方式
- 用 bench_serving 和自写脚本做并发阶梯压测并解读结果

### 初步理解（懂原理、做过实验，但还没到源码级）

- Prefill / Decode 两阶段及其计算特征（Compute Bound vs Memory Bound）
- KV Cache 的机制与显存代价
- Continuous Batching 的调度思想
- RadixAttention / Prefix Cache 的原理、收益来源与适用负载
- Chunked Prefill、零开销调度器的存在意义
- TTFT / TPOT / Throughput 指标体系

### 暂时不需要深入（知道存在即可，留给进阶）

- SGLang 前端 DSL（`@sgl.function` 结构化编程）
- Tensor Parallel / Expert Parallel / PD 分离等多卡与分布式部署
- Speculative Decoding（投机解码）
- 量化部署细节（FP8/AWQ 的具体配置）
- CUDA Kernel 层实现（FlashAttention/FlashInfer 源码）
- SGLang Router、多节点集群、生产级可观测体系

---

# 四、下一阶段学习路线（4 周进阶计划）

> 目标：从"会用 SGLang"到"懂推理引擎"，为读源码和做推理优化打基础。
> 节奏：每天 2～3 小时，每周 5 天学习 + 2 天复盘缓冲。

## 第 1 周：推理引擎通用原理深化（以 vLLM 为对照）

| 天 | 内容 | 实践 |
|---|---|---|
| D1 | PagedAttention 论文精读：KV 分页像操作系统内存管理 | 画 block table 示意图，写一篇 500 字笔记 |
| D2 | 安装 vLLM，跑通与 SGLang 相同的模型 | 同模型双引擎启动，对比启动日志 |
| D3 | 双引擎 Benchmark 对比（相同负载） | 用 bench_serving 打两个引擎，填对比表 |
| D4 | 前缀缓存机制对比：radix tree vs block hash | 设计"前缀差 1 个 token"的边界用例，观察两引擎命中率 |
| D5 | 读 vLLM 的 V1 架构设计文档 | 画出 EngineCore / Scheduler / KVCacheManager 关系图 |

## 第 2 周：深入 SGLang 源码（自顶向下，不求全懂）

| 天 | 内容 | 实践 |
|---|---|---|
| D1 | 源码地图：`python/sglang/srt/` 目录结构，从 `entrypoints/http_server.py` 出发 | 给启动流程画调用链：HTTP → TokenizerManager → Scheduler |
| D2 | 调度器：`managers/scheduler.py` 的主循环 | 加一行日志打印每步 batch 大小，重启观察 |
| D3 | Radix Cache：`mem_cache/radix_cache.py` | 读 `match_prefix` 函数，写单测调用它验证前缀匹配 |
| D4 | Continuous Batching 实现：`get_new_batch_prefill` / `run_batch` | 用一个"长短混合"负载观察 batch 组成随时间变化 |
| D5 | 一周复盘 | 写一篇《SGLang 一次请求的内部旅程》技术博客 |

## 第 3 周：硬件与算子层（CUDA / FlashAttention / 并行）

| 天 | 内容 | 实践 |
|---|---|---|
| D1 | GPU 基础复习：SM、显存层次、带宽与算力的 Roofline 模型 | 查你 GPU 的带宽和 FLOPS，算 decode 理论速度上限 |
| D2 | FlashAttention 原理：为什么不实例化 N×N 注意力矩阵 | 读 FlashAttention 论文第 1～3 节 + 图解博客 |
| D3 | CUDA Kernel 初识：写一个简单的向量加法 kernel（Triton 即可） | 用 Triton 写 vector add 和 fused softmax，对比 PyTorch 原生 |
| D4 | Tensor Parallel：把模型切到多卡的数学原理（Megatron 风格） | 双卡环境用 `--tp 2` 启动 SGLang，观察显存与吞吐变化 |
| D5 | Attention backend 生态：FlashInfer / FlashAttention / Triton 在 SGLang 中的切换 | 用 `--attention-backend` 切换后端跑同一个小 benchmark |

## 第 4 周：推理性能优化专题 + 产出

| 天 | 内容 | 实践 |
|---|---|---|
| D1 | 量化：FP8 / INT8 / AWQ 的原理与精度-性能权衡 | 用 FP8 量化模型对比 BF16 的质量与速度 |
| D2 | Speculative Decoding：小模型起草、大模型验证 | 读 EAGLE 论文摘要，了解 SGLang 的 speculative 参数 |
| D3 | PD 分离（Prefill/Decode Disaggregation）：为什么两个阶段值得分开部署 | 读 DistServe/Mooncake 论文摘要，画架构图 |
| D4 | 生产化：SGLang Router、多副本、Prometheus + Grafana 监控 | 给 Mini Project 加上 metrics 采集和一张监控面板 |
| D5 | 综合产出 | 二选一：① 给 SGLang 提一个文档/小修复 PR；② 写一篇完整的《SGLang 核心机制源码解析》长文 |

**4 周后达到的状态**：能独立阅读推理引擎核心模块源码、能量化分析任何部署的性能瓶颈、具备向 AI Infra 工程师方向继续深入（kernel 开发 / 分布式调度 / 集群优化）的完整知识框架。

---

## 附：常见踩坑速查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `CUDA out of memory` 启动失败 | mem-fraction 太高 / 模型太大 | 降 `--mem-fraction-static`，换小模型 |
| FlashInfer 编译/加载报错 | CUDA 与 PyTorch 版本不匹配 | 用官方 Docker 镜像最省心 |
| 缓存命中率始终为 0 | prompt 前缀不一致（如开头有时间戳） | 把可变内容放 prompt 末尾 |
| 并发上去后延迟暴涨 | 超过显存能容纳的 batch | 看 `#tokens` 池大小，降并发或加卡 |
| TTFT 周期性尖刺 | 长 prompt 阻塞 decode | 确认 chunked prefill 开启，调 `--chunked-prefill-size` |

> 版本提示：SGLang 迭代极快（月均多个版本），执行本计划时以 `pip show sglang` 的实际版本和 docs.sglang.ai 当前文档为准；本文命令基于 v0.5.x 验证。
