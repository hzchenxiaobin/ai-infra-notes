# AI Infra 面经与面试题整理（知乎）

## 专题笔记

- [AI Infra 面试题与参考答案（北美面经篇）](notes/us_interview_qa.md)：知乎《AI infra 面试经验贴》全帖面试题整理，逐题附参考答案（GPU 基础 / 手撕 kernel / LeetCode / 系统设计）
- [AI Infra 社招面试实录与参考答案（面试官视角）](notes/social_interview_qa.md)：知乎《记AI-infra/大模型推理社招面试一兄弟的全过程》面试题整理，附参考答案（项目深挖 / 推理 / 量化 / CUDA / C++ 手撕）

> **来源**：知乎公开面经与面试题总结（链接见文末参考资料），检索整理时间 2026-07
> **适用对象**：准备 AI Infra / 推理引擎 / 训练框架 / 高性能计算方向岗位的求职者
> **说明**：知乎页面有反爬保护，以下内容基于搜索摘要整理，细节请点原文链接核对

---

## 一、面试的一般形式

技术面典型流程（[AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231)）：

1. **自我介绍**（约 10 分钟）
2. **基础技术快问快答**（热身，也可能放在最后）
3. **手撕代码**（CUDA kernel 或 C++/算法题）

社招额外注意（[记 AI-infra/大模型推理社招面试全过程](https://zhuanlan.zhihu.com/p/1920946738270810330)）：

- 推理/量化/CUDA 的基础和进阶问题都会问
- 面试官对 **C++ 能力有明确期望**，最后常出 C++ 手撕题

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

参考：[AI Infra 面试常考——vLLM](https://zhuanlan.zhihu.com/p/2011083570035319972)、[大模型推理优化面试题 v0.1](https://zhuanlan.zhihu.com/p/1894450748672161081)、[AI Infra 面试 QA 总结——大模型推理](https://zhuanlan.zhihu.com/p/2031444649823449112)

### 3. 模型训练 / 分布式

- 并行策略：TP / PP / DP / SP 的区别、使用场景、怎么选择和配比
- ZeRO 1/2/3 原理；FSDP 与 ZeRO 的关系
- 通信计算重叠；流水线并行 GPipe / 1F1B
- 显存计算：参数 / 梯度 / 优化器状态 / 激活值各占多少
- 混合精度训练、梯度累积
- Megatron 相关细节

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

### 6. 场景题 / 开放题

- "客户给你一个模型说速度太慢，从哪些方面着手解决？"（字节豆包 ML Sys 实习真题，见[26 届暑期实习面经](https://www.zhihu.com/question/1890702342850053964/answer/1892522266945897426)）
- 面试官给具体场景让你出解决方案，并指出你项目中的可优化点（[商汤面经](https://www.zhihu.com/question/634549091/answer/3390948311)）

## 三、公司风格差异

| 公司 | 特点 |
|------|------|
| 字节 | 基础 + 手撕代码，重编程能力；豆包 ML Sys 岗考过"模型太慢怎么优化"开放题 |
| 小红书 | 一面代码面、一面业务面，流程快，一面两道代码题 |
| 商汤 | 面试官 AI Infra 功底深，会深挖项目并给场景题 |
| 米哈游 | 一面聊业务，二面可能画风突转深挖预训练细节 |
| 零一万物 | 拷打项目 + 并行计算优化点，代码题不难（两数之和） |
| 量化相关组 | 可能全程量化算法 + 数学推导 |

## 四、准备建议（综合各面经）

1. **CUDA 是硬通货**：能手写 GEMM、reduce、softmax、FlashAttention 级别的 kernel；LeetGPU 刷题
2. **吃透 vLLM / SGLang 源码**：简历上写"改动过 vLLM 内部机制"（如基于 PagedAttention 做 KV Cache 定制优化）比看面经更能拉开差距（[知乎专栏](https://zhuanlan.zhihu.com/p/2056491157132203202)）
3. **推理 > 训练**（当前校招趋势）：PD 分离、KV Cache、Continuous Batching、投机解码必考
4. **C++ 不能放**：智能指针、STL、模板、并发；社招尤其看重
5. **项目要能经得起深挖**：面试官普遍"拷打项目"，会指出可优化点并追问解决方案

## 参考资料

### 面经类

- [AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231)
- [AI infra 26 秋招面经](https://zhuanlan.zhihu.com/p/2017740483217081305)
- [AI Infra 实习面经（二）](https://zhuanlan.zhihu.com/p/1907536883430437857)
- [AI Infra & 投机解码方向实习面经](https://zhuanlan.zhihu.com/p/1998061370952929925)
- [C++/CUDA/AI-infra 面试经验总结](https://zhuanlan.zhihu.com/p/2005325241803621742)
- [记 AI-infra/大模型推理社招面试一兄弟的全过程](https://zhuanlan.zhihu.com/p/1920946738270810330)
- [大模型算法方向实习会经常提问哪些问题](https://www.zhihu.com/question/634549091/answer/3390948311)
- [26 届暑期实习面经（字节豆包 ML Sys）](https://www.zhihu.com/question/1890702342850053964/answer/1892522266945897426)

### 面试题整理类

- [大模型 AI Infra 方向面试会有哪些经常提问的问题？](https://www.zhihu.com/question/1916645420085514580/answer/1928865811105322750)
- [AI Infra 面试问题 QA 总结——大模型推理](https://zhuanlan.zhihu.com/p/2031444649823449112)
- [AI Infra 面试问题 QA 总结——C++](https://zhuanlan.zhihu.com/p/2032225037470671229)
- [AI Infra 面试常考——vLLM 大模型推理框架](https://zhuanlan.zhihu.com/p/2011083570035319972)
- [AI Infra 面试常考—C++ 八股](https://zhuanlan.zhihu.com/p/2017623155192115538)
- [大模型推理优化面试题 v0.1](https://zhuanlan.zhihu.com/p/1894450748672161081)
- [LLM 大模型训练框架岗有哪些面试题？](https://www.zhihu.com/question/647498812)
- [AI Infra 面试真正拉开差距的不是你看了多少面经](https://zhuanlan.zhihu.com/p/2056491157132203202)
