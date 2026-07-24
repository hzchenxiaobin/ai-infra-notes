# AI Infra 面经与面试题整理

## 专题笔记

- [AI Infra 面试题与参考答案（北美面经篇）](notes/us_interview_qa.md)：知乎《AI infra 面试经验贴》全帖面试题整理，逐题附参考答案（GPU 基础 / 手撕 kernel / LeetCode / 系统设计）
- [AI Infra 社招面试实录与参考答案（面试官视角）](notes/social_interview_qa.md)：知乎《记AI-infra/大模型推理社招面试一兄弟的全过程》面试题整理，附参考答案（项目深挖 / 推理 / 量化 / CUDA / C++ 手撕）

> **来源**：知乎公开面经与面试题总结、牛客网面经帖（链接见文末参考资料），检索整理时间 2026-07
> **适用对象**：准备 AI Infra / 推理引擎 / 训练框架 / 高性能计算方向岗位的求职者
> **说明**：知乎页面有反爬保护，部分内容基于搜索摘要整理；牛客网面经为帖子原文摘录，细节请点原文链接核对

---

## 一、面试的一般形式

技术面典型流程（[AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231)）：

1. **自我介绍**（约 10 分钟）
2. **基础技术快问快答**（热身，也可能放在最后）
3. **手撕代码**（CUDA kernel 或 C++/算法题）

社招额外注意（[记 AI-infra/大模型推理社招面试全过程](https://zhuanlan.zhihu.com/p/1920946738270810330)）：

- 推理/量化/CUDA 的基础和进阶问题都会问
- 面试官对 **C++ 能力有明确期望**，最后常出 C++ 手撕题

牛客网面经普遍呈现类似流程（[快手一面](https://www.nowcoder.com/feed/main/detail/e9825f8b587d48b39cbd1f8971e09557)、[小厂实习](https://www.nowcoder.com/feed/main/detail/166e576d5afa4a298cf9492ed51bed04)）：**项目拷打 → 技术八股/场景题 → LeetCode 手撕**，部分岗位（如文远知行）还会拷打 K8s 等工程基础设施。

## 二、高频考点分类

### 1. CUDA / GPU 编程（权重最大）

知乎用户经验（[Micro 的想法](https://www.zhihu.com/pin/1745542997583015936)）：大模型相关岗位**手撕 CUDA 与 LeetCode 的比例约 4:1**，FlashAttention、FMHA 这类优化手段面试基本都会问。

常见问题（[大模型 AI Infra 方向面试常见问题](https://www.zhihu.com/question/1916645420085514580/answer/1928865811105322750)、[C++/CUDA/AI-infra 面试经验总结](https://zhuanlan.zhihu.com/p/2005325241803621742)）：

- 为什么 shared memory 可以优化 GEMM？什么情况需要 `__syncthreads` 同步？
- 对 FlashAttention 的理解；Hopper 架构有哪些新特性？
- 怎么定位性能热点？CUDA kernel 有哪些优化手段？
- GPU 和 CPU 的真正区别？SM / Warp / SIMD / SIMT 分别是什么？
- Warp 分化会带来什么性能坑？
- 算力与带宽：训练慢的时候通常卡在哪些环节？
- 现场手写 GEMM CUDA kernel，并追问优化方向（[AI Infra 实习面经（二）](https://zhuanlan.zhihu.com/p/1907536883430437857)）

牛客网补充高频题：

- Bank Conflict 是什么、怎么解决（[鹅厂一面](https://www.nowcoder.com/feed/main/detail/79f0a345a13a466a831a9ae5104f0c03)）
- 怎么减少 launch kernel overhead（鹅厂一面）
- Cutlass 2.0 怎么实现 grouped GEMM（[快手二面](https://www.nowcoder.com/feed/main/detail/490dbb6cdebd4d42ad2bc46757d433af)）
- `CUDA_DEVICE_MAX_CONNECTIONS` 能干什么、和 launch bound 什么关系（快手二面）
- Hopper TMA 的优点、调用方式、是否需要经过 L1（[小厂面经](https://www.nowcoder.com/feed/main/detail/166e576d5afa4a298cf9492ed51bed04)）
- Blackwell 和 Hopper 比有什么变化（快手二面）
- 聊一下你所找到的 CUDA GEMM 优化方法（[快手一面](https://www.nowcoder.com/feed/main/detail/e9825f8b587d48b39cbd1f8971e09557)）

**练习**：LeetGPU 刷题（本仓库 `leetgpu/` 目录即为配套练习）。

### 2. 模型推理（近年考察核心）

[AI infra 26 秋招面经](https://zhuanlan.zhihu.com/p/2017740483217081305)总结：**推理优化是绝对核心**——PD 分离、KV Cache 管理、Continuous Batching 等工程架构设计被反复考察。

高频题：

- LLM 推理为什么需要 KV Cache？怎么计算 KV Cache 大小？如何管理？
- PagedAttention：vLLM 分块管理 KV Cache，避免显存碎片
- Prefix Cache / RadixAttention、LRU 淘汰策略
- Continuous Batching 动态拼 batch；调度策略有哪些、何时触发调度
- Speculative Decoding：小模型出草稿、大模型并行验证
- 延迟 vs 吞吐怎么权衡，具体业务场景怎么选
- MQA / GQA / MLA 对 KV Cache 大小的影响
- SD（Stable Diffusion）有哪些加速策略，和 LLM serving 有什么区别
- 对 K8s、Ray 了解多少

牛客网补充高频题：

- **AF 分离**：解决什么问题？既然有 PD 分离了，为什么还要 AF 分离？（快手一面）
- **Chunk Prefill**：为了解决什么问题而提出？（鹅厂一面）
- **Flash Attention V2 vs V1** 的改进细节，及 V3/V4 版本（快手一面/二面）
- vLLM / SGLang 里的 continuous batching 介绍（小厂面经）

参考：[AI Infra 面试常考——vLLM](https://zhuanlan.zhihu.com/p/2011083570035319972)、[大模型推理优化面试题 v0.1](https://zhuanlan.zhihu.com/p/1894450748672161081)、[AI Infra 面试 QA 总结——大模型推理](https://zhuanlan.zhihu.com/p/2031444649823449112)

### 3. 模型训练 / 分布式

- 并行策略：TP / PP / DP / SP 的区别、使用场景、怎么选择和配比
- ZeRO 1/2/3 原理；FSDP 与 ZeRO 的关系
- 通信计算重叠；流水线并行 GPipe / 1F1B
- 显存计算：参数 / 梯度 / 优化器状态 / 激活值各占多少
- 混合精度训练、梯度累积
- Megatron 相关细节

牛客网补充高频题：

- **TP 切分细节**：Megatron 里 MLP 第一个矩阵和第二个矩阵分别是行切还是列切？通信分别是什么算子（all-reduce / reduce-scatter）？（小厂面经）
- 大模型一层有几个线性层？TP 的时候怎么切？有什么思路优化中间的 allreduce？（快手一面）
- 介绍流水线并行，说明 1F1B、DualPipe（小厂面经）
- reduce-scatter 和 all-to-all 通信的区别（鹅厂一面）
- 场景题：大集群中节点内有 NVLink、节点间部分机器有 RDMA，怎么设计分布式推理方案（鹅厂一面）

参考：[LLM 大模型训练框架岗面试题](https://www.zhihu.com/question/647498812)

### 4. C++ 八股

- 内存模型、STL 容器底层实现、并发、智能指针、模板
- 手撕 `shared_ptr` 是经典题（字节实习面真题）
- 避免循环引用（`weak_ptr`）

参考：[AI Infra 面试常考—C++ 八股](https://zhuanlan.zhihu.com/p/2017623155192115538)、[AI Infra 面试 QA 总结——C++](https://zhuanlan.zhihu.com/p/2032225037470671229)

> 本仓库 [C++ 面试专题](../cpp/README.md) 已系统覆盖这些考点。

### 5. 量化

- GPTQ、SmoothQuant 原理（字节某组全程问量化 + 数学推导，见[面经](https://www.zhihu.com/question/29906268)）
- SmoothQuant 中 smooth scale 如何求、α 取值
- 数学：Cholesky 分解、凸函数、泰勒展开

牛客网补充高频题（[智谱一面](https://www.nowcoder.com/feed/main/detail/846a09e34fea4fe9a7e14da2a88e3f72)、[美团一面](https://www.nowcoder.com/feed/main/detail/48166f26d9c2472eaa217bb94f7e88fa)）：

- 校准算法：MinMax 和 Percentile 有什么不同？还知道 KL、MSE 吗？代码实现
- 按量化粒度说明 SmoothQuant 是什么粒度？per-tensor / channel / group 哪个更细？
- AWQ、SmoothQuant、GPTQ 的原理、作用流程与对比
- NVFP4 的原理、怎么做缩放、在哪个维度缩放、保存格式
- 常用量化到什么格式（FP8、NVFP4）
- 剪枝稀疏化可以分为哪些种类

### 6. RL / 训练后（牛客网新增高频）

- 大模型 RL 全流程：涉及哪些模型？PPO / GRPO 有什么区别？（小厂面经）
- RL 里 rollout 耗时占比大概多少？policy MFU 大概多少？MFU 计算公式、6Nd 公式是什么？
- rollout 有哪些优化点（rollout 量化、异步 rollout 等）
- RL 中如何把预训练权重同步到推理引擎
- LLM 的知识蒸馏放在预训练做是否合适

### 7. 分布式系统 / 调度（牛客网新增高频）

- **K8s**：Pod / Deployment 从提交到拉起的全流程；Controller / Informer 原理；滚动更新时流量怎么切；容器如何做到 PID 隔离（[虾皮一面](https://www.nowcoder.com/feed/main/detail/e610f57cfd3548cd96a27d92e2f8b25e)、[百度一面](https://www.nowcoder.com/feed/main/detail/9c4940f8c9ac4f21b5107eef45ed98c1)、文远知行一面）
- **Docker**：实现原理、容器隔离机制（百度一面）
- **Ray**：底层实现与特性、调度器结构、单节点 OOM 怎么处理（快手一面、虾皮一面）
- Go 后端八股（百度 AI infra 偏后端方向）：tag 映射、反射、Slice/Map 扩容、GMP 模型；网络 TCP/UDP/HTTP/HTTPS、send/write/mmap/sendfile、用户态/内核态

> 牛客网友评论："为什么 ai infra 问的像后端啊"——部分公司的 AI Infra 岗实际偏平台/后端工程，K8s 与分布式调度是重点（[百度面经](https://www.nowcoder.com/feed/main/detail/9c4940f8c9ac4f21b5107eef45ed98c1)）。

### 8. 场景题 / 开放题

- "客户给你一个模型说速度太慢，从哪些方面着手解决？"（字节豆包 ML Sys 实习真题，见[26 届暑期实习面经](https://www.zhihu.com/question/1890702342850053964/answer/1892522266945897426)）
- 面试官给具体场景让你出解决方案，并指出你项目中的可优化点（[商汤面经](https://www.zhihu.com/question/634549091/answer/3390948311)）
- 大集群中节点内有 NVLink、节点间部分有 RDMA，设计分布式推理方案（鹅厂一面）

## 三、公司风格差异

| 公司 | 特点 |
|------|------|
| 字节 | 基础 + 手撕代码，重编程能力；豆包 ML Sys 岗考过"模型太慢怎么优化"开放题 |
| 快手 | 偏底层算子与架构：Flash Attention V2/V4、TP 切分、Ray、Cutlass、Blackwell；FA3/FA4、hang 排查 |
| 腾讯 | 推理架构与通信：Chunk Prefill、reduce-scatter/all-to-all、bank conflict、分布式推理方案设计 |
| 百度 | 偏后端/平台工程：Go 八股、网络、K8s/Docker（网友反映"像后端面"） |
| 美团 | 量化深挖：AWQ/SmoothQuant/GPTQ 原理与优化路径、剪枝稀疏化、模型架构对比 |
| 智谱 | 量化全流程：校准算法、粒度对比、NVFP4、代码实现 |
| 虾皮 | 工程基础设施：K8s 全流程、Ray 调度、B+树/索引 |
| 文远知行（车企） | K8s 拷打、网络通信；要求高、问得细，最新技术都要掌握 |
| 华为海思 | C/C++ 基础 + AI 算子开发 |
| 小红书 | 一面代码面、一面业务面，流程快，一面两道代码题 |
| 商汤 | 面试官 AI Infra 功底深，会深挖项目并给场景题 |
| 米哈游 | 一面聊业务，二面可能画风突转深挖预训练细节 |
| 零一万物 | 拷打项目 + 并行计算优化点，代码题不难（两数之和） |
| 量化相关组 | 可能全程量化算法 + 数学推导 |

> 牛客网友观察（[搞AI Infra才是主流？](https://www.nowcoder.com/feed/main/detail/12617f22ddf14346bb6e950f1838d6dd)）：底层 infra 岗**招的几乎全是 infra 不是 researcher**，**几乎不招初级员工**，中位数工作经验 12.2 年，仅 13% 有博士学位。

## 四、准备建议（综合各面经）

1. **CUDA 是硬通货**：能手写 GEMM、reduce、softmax、FlashAttention 级别的 kernel；LeetGPU 刷题
2. **吃透 vLLM / SGLang 源码**：简历上写"改动过 vLLM 内部机制"（如基于 PagedAttention 做 KV Cache 定制优化）比看面经更能拉开差距（[知乎专栏](https://zhuanlan.zhihu.com/p/2056491157132203202)）
3. **推理 > 训练**（当前校招趋势）：PD 分离、KV Cache、Continuous Batching、投机解码必考；AF 分离、Chunk Prefill、FA V3/V4 等新进展要跟
4. **C++ 不能放**：智能指针、STL、模板、并发；社招尤其看重
5. **项目要能经得起深挖**：面试官普遍"拷打项目"，会指出可优化点并追问解决方案
6. **技术迭代快，别只背八股**：车企等要求"最新技术都要掌握"，牛客网友反映"光背八股面经不太行"（[AI infra 应届春招](https://www.nowcoder.com/feed/main/detail/b198bb96bade40768bfaaa94cb946256)）
7. **大厂偏底层，中厂偏工程**：大厂考 CUDA/算子/并行，部分公司（百度/虾皮/文远）重 K8s/Ray/分布式调度
8. **学习资源**：[chenzomi12/AISystem](https://github.com/chenzomi12/AISystem)（中文 AI 系统全栈课）、[HuaizhengZhang/AI-Infra-from-Zero-to-Hero](https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero)（论文+工业实践）、[CalvinXKY/InfraTech](https://github.com/CalvinXKY/InfraTech)

## 参考资料

### 知乎面经类

- [AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231)
- [AI infra 26 秋招面经](https://zhuanlan.zhihu.com/p/2017740483217081305)
- [AI Infra 实习面经（二）](https://zhuanlan.zhihu.com/p/1907536883430437857)
- [AI Infra & 投机解码方向实习面经](https://zhuanlan.zhihu.com/p/1998061370952929925)
- [C++/CUDA/AI-infra 面试经验总结](https://zhuanlan.zhihu.com/p/2005325241803621742)
- [记 AI-infra/大模型推理社招面试一兄弟的全过程](https://zhuanlan.zhihu.com/p/1920946738270810330)
- [大模型算法方向实习会经常提问哪些问题](https://www.zhihu.com/question/634549091/answer/3390948311)
- [26 届暑期实习面经（字节豆包 ML Sys）](https://www.zhihu.com/question/1890702342850053964/answer/1892522266945897426)

### 知乎面试题整理类

- [大模型 AI Infra 方向面试会有哪些经常提问的问题？](https://www.zhihu.com/question/1916645420085514580/answer/1928865811105322750)
- [AI Infra 面试问题 QA 总结——大模型推理](https://zhuanlan.zhihu.com/p/2031444649823449112)
- [AI Infra 面试问题 QA 总结——C++](https://zhuanlan.zhihu.com/p/2032225037470671229)
- [AI Infra 面试常考——vLLM 大模型推理框架](https://zhuanlan.zhihu.com/p/2011083570035319972)
- [AI Infra 面试常考—C++ 八股](https://zhuanlan.zhihu.com/p/2017623155192115538)
- [大模型推理优化面试题 v0.1](https://zhuanlan.zhihu.com/p/1894450748672161081)
- [LLM 大模型训练框架岗有哪些面试题？](https://www.zhihu.com/question/647498812)
- [AI Infra 面试真正拉开差距的不是你看了多少面经](https://zhuanlan.zhihu.com/p/2056491157132203202)

### 牛客网面经类

- [快手 Ai infra 一面拷打](https://www.nowcoder.com/feed/main/detail/e9825f8b587d48b39cbd1f8971e09557)
- [快手 ai infra 二面](https://www.nowcoder.com/feed/main/detail/490dbb6cdebd4d42ad2bc46757d433af)
- [百度 AI infra 面经 好难](https://www.nowcoder.com/feed/main/detail/9c4940f8c9ac4f21b5107eef45ed98c1)
- [鹅厂实习一面 Ai infra](https://www.nowcoder.com/feed/main/detail/79f0a345a13a466a831a9ae5104f0c03)
- [美团实习 ai infra 一面分享](https://www.nowcoder.com/feed/main/detail/48166f26d9c2472eaa217bb94f7e88fa)
- [智谱 Ai infra 一面面经](https://www.nowcoder.com/feed/main/detail/846a09e34fea4fe9a7e14da2a88e3f72)
- [虾皮 ai infra 研发实习一面](https://www.nowcoder.com/feed/main/detail/e610f57cfd3548cd96a27d92e2f8b25e)
- [数坤科技 AI infra 实习一面](https://www.nowcoder.com/feed/main/detail/f6b2716a6e1b4c0f96564ca06af3609b)
- [AI infra 实习面经（小厂）](https://www.nowcoder.com/feed/main/detail/166e576d5afa4a298cf9492ed51bed04)
- [AI infra 应届春招（京东一面）](https://www.nowcoder.com/feed/main/detail/5e217371048d4c35b033f13ab277dd45)
- [AI infra 应届春招（文远知行一面）](https://www.nowcoder.com/feed/main/detail/9ec1d6e590a04b16b9ea40ade8d180bd)
- [AI Infra 面经 攒人品版](https://www.nowcoder.com/feed/main/detail/d695614a06424c148c04586ac3a66e78)
- [AI 岗位高频面试题整理: AI infra 方向](https://www.nowcoder.com/feed/main/detail/6ce0a48780564d378ebf559f4a90553c)
- [搞 AI Infra 才是主流？](https://www.nowcoder.com/feed/main/detail/12617f22ddf14346bb6e950f1838d6dd)
