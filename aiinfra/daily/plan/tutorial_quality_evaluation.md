# AI Infra 8 周教程质量评估与改进建议

> 评估日期：2026-08-04
> 评估范围：`aiinfra/daily/` 目录下全部文件（images/ 下 svg 除外）
> 评估目的：以"面试 + 找工作"为目标，结合 2025 年 AI Infra 高性能/算子工程师岗位市场要求，评价教程质量并提出改进点
> 评估方法：8 个子任务并行通读 8 周全部 README/kernel/notes（~40,000+ 行），交叉验证关键事实

---

## 一、总体评估

本教程是一套**结构完整、教学逻辑清晰、内容体量可观**（8 周 56 天，~40,000+ 行）的 AI Infra 工程实战学习材料。从 GPU 执行本质 → Kernel 优化 → Transformer 算子 → FlashAttention → 推理系统 → Batching/调度 → 系统整合 → 面试准备的递进路线**设计合理**，8 段式日级骨架（`SKILL.md` 规范）执行一致性高，每日配套 LeetGPU + LeetCode 形成组合训练。作为面试/求职导向的教程，**概念教学与面试 Q&A 是其最强项**（~250+ 道面试题，online softmax 三公式推导、occupancy 手算、GEMM 优化层次路径等均为白板级）。

**但在"工程实战"与"对标市场要求"两个维度上存在系统性短板**：核心算子/系统代码大量停留在 markdown 代码块（未落盘）、性能 profiling 几乎从未真正执行、关键硬件参数三套数据互相矛盾、若干 CUDA 内核存在真实 bug、且对 2025 年 AI Infra 高性能/算子工程师岗位的高频考点（Tensor Core/CUTLASS/Triton/分布式/量化/FA3/FlashDecoding）覆盖严重不足。

### 评估摘要表

| 维度 | 评级 | 说明 |
|------|------|------|
| 路线设计 | ★★★★★ | 8 周递进、每周主题清晰、前后衔接紧密 |
| 概念教学深度 | ★★★★☆ | online softmax 推导、occupancy 手算、Welford 等为白板级 |
| 面试 Q&A 体系 | ★★★★☆ | ~250+ 题、`<details>` 折叠、频率星级、覆盖面广 |
| 代码工程实战 | ★★★☆☆ | 少数真测量（GEMM v1-v6、多流），大量未落盘 + 模拟 |
| Profiling 实操 | ★★☆☆☆ | 方法论好但执行率 < 10%，数据表几乎全空白 |
| 事实准确性 | ★★☆☆☆ | 硬件参数三套矛盾、多处事实错误、若干 CUDA bug |
| 市场对标覆盖 | ★★☆☆☆ | 缺 Tensor Core/CUTLASS/Triton/分布式/量化/FA3 等高频考点 |
| 仓库卫生 | ★★★☆☆ | 硬编码路径、broken SVG、重复表格、目标虚标 |

---

## 二、教程的显著优点（值得保留）

### 2.1 概念教学深度高

- **Week1**：occupancy 手算 4 步法 + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API 验证（`week1/day3/exercise/occupancy_verify.cu`，217 行，运行时实测 vs 手算对比）；coalesced access 的 transaction/sector 机制深挖；bank conflict padding 推导
- **Week2 Day5**：online softmax 三公式 LaTeX 推导（含"逐步归一化"与"FA 论文末端归一化"等价性证明）
- **Week3 Day3**：Welford 并行合并公式与 SASS 级 LDG.128 vs LDG.32 分析（`softmax_layernorm_opt.cu`，560 行，5 个 kernel）
- **Week6 Day3**：vLLM 5 步 `schedule()` 源码级复刻（`vllm_scheduler_analyzer.py`，520 行，含 RECOMPUTE/SWAP 两种抢占模式，实测迭代数验证）

这些都是**真正能达到面试白板要求**的内容。

### 2.2 8 段骨架 + 双题库组合训练

`SKILL.md` 定义的"目标/学前导读/理论学习/Coding 任务(含 LeetGPU+LeetCode)/扩展实验/今日总结/面试要点"结构在 56 天中执行率接近 100%，LeetGPU 与当日主题强相关（Prefix Sum↔Warp Reduce、GEMM↔Register Blocking、Softmax Attention↔FlashAttention），形成"学-练-面"闭环。

### 2.3 有真实测量数据

- Week2 Day6 的 GEMM v1→v6 优化系列在 RTX 5090 上实测（v1 10.6% → v4 64.3% cuBLAS 占比，0 spill，`gemm_optimization_series.cu` 728 行）
- Week2 Day3 的多流 pipeline 实测 2.43x 加速（`multi_stream_pipeline.cu`，含 nsys 时间线截图）
- Week3 Day3 的 Welford vs baseline 实测对比

这些是**少数真正落地**的工程数据，弥足珍贵。

### 2.4 工程诚实性

- Week2 Day6 明确承认"整合版 kernel 实际未使用 shuffle 写回"（给出 3 条理由）、"double buffer 不用 cp.async 不会提速"
- Week3 Day5 诚实记录自定义 kernel 比 PyTorch 慢 0.8-0.95x
- Week5 Day4 的 `paged_attention.cu` 明确标注"简化：单 head、单 sequence、decode 场景"

这种诚实态度对面试叙事有帮助。

### 2.5 面试 Q&A 体系化

每日 5 题 + 周末复盘 10-15 题，Week8 集中整理 ~43 道去重题，覆盖 GPU 基础/Kernel 优化/Attention/推理系统/vLLM/系统设计/行为题，且用 `<details>` 折叠答案 + 频率星级标注，适合自测。

---

## 三、关键质量问题（直接影响面试/求职可信度）

### 3.1 硬件参数三套数据互相矛盾 —— 最严重问题

整个仓库中"RTX 5090"参数存在**三套互相打架的数据**，且只有一套正确：

| 来源 | FP32 算力 | 带宽 | Ridge Point | threads/SM | warps/SM | blocks/SM | smem |
|------|----------|------|-------------|-----------|---------|-----------|------|
| `week1/day1/README.md` L108-169 | 989 TFLOPS | 141 GB HBM3e | — | 2048 | 64 | 32 | 228KB |
| `week1/day4/README.md` L68-71, `day6`, `day7`, `week8` 全部 | **19.5 TFLOPS** | **1.55 TB/s HBM** | **12.6** | — | — | — | — |
| `week1/day3/exercise/my_gpu_info.md`（实测 deviceQuery） | **104.75 TFLOPS** | **1792 GB/s GDDR7** | **58.45** | **1536** | **48** | **24** | 100KB |

- **第一套**是 **B200 数据中心卡**规格误标为 RTX 5090（989 TFLOPS、HBM3e）
- **第二套**是 **A100 40GB SXM4** 规格误标为 RTX 5090（19.5 TFLOPS、1.55 TB/s HBM）—— **这是被引用最广的错误数据**，导致全仓库所有 Roofline 分析、occupancy 手算、bandwidth 利用率计算全部基于错误 ridge point（12.6 vs 正确 58.45）
- **第三套**才是**正确的 RTX 5090**（104.75 TFLOPS、1792 GB/s GDDR7、sm_120）

**直接后果**：

- 按 README 教学路径，学习者会算出错误的 occupancy（50% vs 正确 66.7%）
- 算子 bound 分类推理过程错（AI=12 算 memory-bound，但实际 ridge=58.45，结论碰巧对但推理错）
- `week1/tools/cuda_occupancy_calculator.py` 的 sm_120 表项也是 B200 数据
- **面试中被问"你用的卡 ridge point 多少"时答 12.6 会直接暴露问题**

### 3.2 核心知识点存在事实性错误

| 错误 | 位置 | 正确事实 |
|------|------|---------|
| "Warp Shuffle 从 Blackwell(CC 12.0)起引入" | `week2/day1/README.md` L31, L84 | Kepler sm_30 (2012) 引入；`_sync` 后缀 + 无 mask 版本弃用始于 **Volta/CUDA 9 (2017)** 的 Independent Thread Scheduling |
| FlashAttention IO = O(Nd) | `week4/day1` L173, `day6` L73, `day7` L290 | 严格界为 **Θ(N²d²/M)**（M 为 SRAM 大小）。仓库自己的 `paper/flashattention/README.md` 都写对了，daily notes 反而简化错 |
| 标准 Attention IO = "3N²+4Nd" | `week4/day1` L54, `day6` L68 | 逐项求和实际为 4N²+4Nd（day6）或 5N²+4Nd（day1），**N² 系数少算 1-2** |
| FA 在 N=4096 时 IO "~2 MB" | `week4/day1` L172 | 应为 ~4 MB（4Nd·4B）；同一天的 Python 脚本算出 4MB，**同日内自相矛盾** |
| 同一 N=4096 的 IO 加速比出现三处不同值 | `week4/day1` L173(~100x) / L338(51.6x) / `day6` L81(48.8x) | 应统一 |
| `week8/day4` L124 KV Cache "LLaMA2-7B 1 MB/token (FP16)" | `week8/day4` | 正确为 ~524 KB/token（`week8/day6` L124 算对了），**同周内 2x 矛盾** |
| GEMM 优化"八层"vs"九层" | `week8/day3` 八层 / `day6` 九层 / `day6` 表格 7 行 | 实际 8 层（Naive→Tiling→RegBlock→float4→Shuffle→DblBuf→TensorCore→Auto-tuning） |
| Register Blocking 占 cuBLAS "40%" vs "45%" | `week8/day3` 40% / `day6` 45% | 同一优化步骤两个数 |
| `nvcc-- default - stream per - thread` | `week8/day3` L149, `interview_basics.py` L102 | 应为 `nvcc --default-stream per-thread`（空格丢失） |
| "L2 Cache 40MB" (RTX 5090) | `week1/day7` L88 | 实际 96 MB |
| "32B sector since Blackwell" | `week1/day4` L178 | 32-byte sector 早于 Blackwell 多代 |
| "warp shuffle 从 CC 12.0 起" | `week1/day3/notes` L172 | 同上，Kepler 起 |

### 3.3 CUDA 内核存在真实 bug

| 文件 | bug | 影响 |
|------|-----|------|
| `week5/day2/kernels/kv_cache.cu` L156 | `cache.append(0, d_k2, d_v2, 8)` 从 5 token 缓冲读 8 token | 越界读 3 token 未初始化内存；因验证只查 Round 1 而被掩盖 |
| `week7/day4/kernels/custom_ops_module.py` L47 | `atomicMax((int*)&s_max, __float_as_int(max_val))` | int 重解释 trick **仅对非负 float 保序**，softmax 输入可负 → max 错 → softmax 错 |
| 同上 L152-163 | online softmax 更新混用两种归一化 | 输出幅度在 key 循环中漂移，数值不正确 |
| 同上 L181/191/206 | kernel launch 未传 stream | 直接违反同 README 4.3 ① 强制的 `getCurrentCUDAStream()` |
| 同上 L133 | `out_local[256]` 硬编码 | D>256 时静默栈损坏，无 `TORCH_CHECK` |
| 同上 L176-210 | 三个 wrapper 无 `TORCH_CHECK` 形状/dtype/连续性检查 | 违反同 README 4.3 ⑤ |
| `week3/day3/kernels/softmax_layernorm_opt.cu` | print 字符串 "Day16" | 与 week3/dayN 目录编号体系不一致 |
| `week7/day2/kernels/full_scheduler.py` L192-193 | `else: break` 应为 `continue` | 一个 running 请求放不下就退出循环，跳过后续请求 |
| `week3/day6/kernels/profiling_targets.cu` | naive GEMM（无 tiling，AI≈0.25）被标为 "compute-bound" | 理论上为 memory-bound，教学误导 |

### 3.4 代码"只在 markdown 里"，未落盘 —— 违反自有 SKILL 规范

`SKILL.md` L40/L117 明确要求 `kernels/*.cu` 为"完整可编译代码(教程中引用的真实文件)"，并用 GitHub 链接引用。但实际：

- **Week2**：`warp_reduce.cu`、`register_blocking_gemm.cu`、`flash_attention.cu`、`integrated_gemm.cu` 仅存在于 markdown 代码块，磁盘上没有（Day7 的"本周目录结构"和"GitHub 整理 Checklist"声称它们存在 —— 文档与现实不符）
- **Week4**：`flash_attention_v2.cu`、`mini_engine_fa.py`、`benchmark_flash_attention.py`、`flash_attention_fa2.cu` **全部不存在**；`mini_engine_fa.py` 中 `open("flash_attention_v2.cu").read()` 会直接崩溃
- 后果：学习者无法直接编译运行；面试时声称"我实现了 FlashAttention kernel"但仓库里找不到文件

### 3.5 Profiling 几乎从未真正执行

- **Week1**：`profiles/week1_profile_summary.md` 与所有 `dayN/notes/*.md` 数据表**全部空白**；唯一具体的 ncu 数据（day6 L227-230）是虚构的"假设值"
- **Week6 Day6**：benchmark 用 `SimulatedEngine`（解析模型）而非真实的 `MiniEngineV1`；无真实 nsys/ncu 数据
- **Week7 Day6**：`full_chain_profile.py` 全程 `time.sleep` 模拟，**从未真正调用 nsys/ncu**，只把命令 print 出来当"指南"
- 全仓库真正落地测量的只有 Week2 Day6 GEMM 系列和 Day3 多流 —— **8 周教程号称"Profiling 是核心能力"，实际执行率 < 10%**

### 3.6 文档/仓库卫生问题

- `week1/day*/notes/*.md` 全部硬编码 macOS 路径 `/Users/chenbinbin/GitHub/aiinfra/week1`
- Week1 有 6 处 SVG 引用路径错误（`../../images/` 应为 `../images/`）且文件不存在
- `week1/day7` L509-522 与 L526-539 LeetCode 表格**完全重复**；`week2/day7` L449-461 与 L465-477 同样重复
- `week4/day6` LeetGPU 题目与 `day2` 重复（都是 Multi-Head Attention）
- `week7` 6 天 LeetGPU 全是 Matrix Transpose（Day1/2/3/4/5/7 重复 6 次）—— 严重违反 `SKILL.md`"避免重复"要求
- 性能目标虚标：`week2/README.md` 承诺"cuBLAS 70%+"，实测 64.3%，Day7 完成标准悄悄退到"65%+"
- `week8` 多处 LeetCode 总结表格放错题目（Day3 出现零钱兑换、Day4 出现课程表、Day5 出现最长有效括号、Day7 出现排序链表，均为 copy-paste 错误）
- `week8/day5` mock 面试声称"30 分钟"，实际各段时间加和 2040 秒 = 34 分钟

### 3.7 面试题数量虚标

`week8/README.md` L9 与 `day7/README.md` L73 声称"50+ 高频面试题库"，实际去重后约 43-45 道。要凑到"50+"需把填空子项和参数训练也单独计数，或计入跨文件重复题。

---

## 四、对标市场要求的差距分析

结合 2025 年国内 AI Infra 高性能工程师与算子工程师岗位的常见 JD（华为/字节/阿里/腾讯/DeepSeek/商汤/地平线等），本教程的覆盖情况如下：

### 4.1 算子工程师岗位核心要求 vs 教程覆盖

| 核心要求 | 教程覆盖 | 差距严重度 |
|---------|---------|-----------|
| CUDA kernel 手写（GEMM/Softmax/LayerNorm/Attention） | ✅ 强（Week1-4） | — |
| **Tensor Core / WMMA / `mma.sync`** | ❌ 仅作为"vs cuBLAS 差距"提及，无任何 hands-on | **致命** —— 算子岗必考，cuBLAS 默认用 WMMA，不掌握手写 GEMM 永远卡在 65% |
| **CUTLASS 源码分析** | ❌ 提及 3 次作为黑盒，零源码分析，无 `cutlass::gemm::device::Gemm` 实例 | **致命** —— 大厂算子岗 JD 明确要求"CUTLASS 熟悉" |
| **Triton 语言** | ❌ 完全缺失（仅在 Week3 提一句"替代 C++ Extension"） | **致命** —— 2025 年算子岗几乎必考 Triton（OpenAI Triton），且是写 FlashAttention/FlashDecoding 的事实标准 |
| 混合精度（FP16/BF16/FP8） | ⚠️ 仅 `integrated_gemm_leetgpu.cu` 用了 `__half`；无 BF16、无 FP8、无 Tensor Core 累加策略 | **高** |
| FlashAttention-3 | ❌ 仅 Day4 Q5 推测性讨论；仓库有 `paper/flashattention3/` 但 daily notes 从未链接 | **高** —— FA3（Hopper/FP8/async pipeline）是当前热点 |
| FlashDecoding / FlashDecoding++ | ❌ 完全缺失 | **高** —— 推理算子岗必问 |
| 反向传播 kernel | ❌ 全部为 forward；`L_i = m_i + log ℓ_i` 从未出现 | **高** —— 算子岗会问 backward |
| MQA/GQA 实现 | ❌ 概念提及，无 kernel 实现（`paged_attention.cu` 单 head） | **中高** |
| Auto-tuning | ⚠️ 方法论提及，无实现 | **中** |
| 架构特性（TMA/cp.async/warp specialization） | ⚠️ 提及但无实现 | **中高** —— Hopper/Blackwell 算子优化核心 |
| Operator fusion 实操 | ⚠️ Week3 Day6 讨论，无 fused kernel 实现 | **中** |
| Sliding window attention | ❌ 缺失 | **中**（Mistral 等） |
| 多硬件（Ascend NPU） | ❌ 零提及 | **高** —— 国内市场 Ascend 普及，华为/昇腾生态岗位明确要求 |

### 4.2 AI Infra 高性能工程师岗位核心要求 vs 教程覆盖

| 核心要求 | 教程覆盖 | 差距严重度 |
|---------|---------|-----------|
| 推理系统（vLLM/PagedAttention/Continuous Batching） | ✅ 强（Week5-6） | — |
| KV Cache / Prefill vs Decode | ✅ 强 | — |
| Nsight profiling 方法论 | ✅ 概念强，❌ 实操几乎为零 | **高** —— 面试会问"你实际 profile 过什么" |
| **分布式推理（TP/PP/DP）** | ❌ 完全缺失（Week7 Day7 自标"待提升"） | **致命** —— 大厂 Infra 岗必问 TP sharding、all-reduce、1F1B |
| **NCCL / 通信库 / 通信计算重叠** | ❌ 全仓库未出现 "NCCL" | **致命** |
| **量化（GPTQ/AWQ/INT4/FP8/weight-only）** | ❌ 仅"INT8 KV Cache"提及 | **高** —— 2025 年量化是降本核心 |
| **CUDA Graph** | ⚠️ 反复提及为优化手段，但**从未实现**（静态捕获、动态 shape 处理） | **高** —— 推理服务必用 |
| Speculative Decoding 实操 | ⚠️ Week7 Day3 模拟，无真实集成 | **中高** |
| Chunked Prefill 实操 | ⚠️ Week6 Day4 模拟器，无真实集成 | **中高** |
| Prefix Caching 实操 | ❌ 仅提及，无实现 | **中高** |
| Disaggregated Prefill/Decode 服务 | ❌ 完全缺失 | **高** —— DeepSeek/DistServe/Splitwise 是当前趋势 |
| torch.compile / 图优化 | ❌ 多处提及，无实操 | **中** |
| Triton Server / 部署 / 服务化 | ❌ 缺失 | **中** |
| 成本分析（$/token、GPU 经济学） | ❌ 缺失 | **中** |
| vLLM V1 架构（AsyncLLMEngine） | ❌ 描述的是 SOSP 2023 原版 | **中** —— vLLM 已演进 |
| MoE / 多模态推理 | ❌ 仅在未来路线图 | **低**（进阶） |
| Ring attention / 长上下文分布式注意力 | ❌ 缺失 | **中** |
| LoRA / 多模型服务（S-LoRA、Punica） | ❌ 缺失 | **中** |

### 4.3 "Mini 推理引擎"作为项目叙事的真实性

Week7 号称"整合为完整 Mini AI Infra 系统"，但实际：

- 7 个 `.py` 文件是**独立 demo，并非集成系统**：`concurrent_engine.py`（Day1）、`full_scheduler.py`（Day2）、`custom_ops_module.py`（Day4）互不调用
- **Day4 的真实 CUDA kernel 从未被任何 engine 调用**：Day5 的"自定义 kernel 集成测试"是假的（只用 0.8x 速度乘子），Day6 的 profiling 也是 `time.sleep` 模拟
- KV Cache 全程是 Python `dict`，非真实 GPU 内存；无真实 PagedAttention block table 接入
- `mini_engine_v1.py`（Week6 Day5）的 `_run_iteration` 逐请求循环 forward，**未做真正的 batched GEMM**（留作"实验 1"）
- 面试中若被追问"你的引擎里 custom kernel 实际接进去了吗"，**如实回答会暴露整合度不足**

---

## 五、各周质量速览

| 周 | 主题 | 行数(约) | 真实代码 | 实测数据 | 面试题 | 主要问题 |
|----|------|---------|---------|---------|--------|---------|
| W1 | GPU 执行本质 + Profiling | ~6,900 | 4 个 .cu（可编译） | ❌ 全空白 | ~46 | 硬件参数三套矛盾；6 处 broken SVG；硬编码 macOS 路径 |
| W2 | GEMM & Kernel 优化 | ~4,400 | 3 个 .cu（可编译） | ✅ GEMM v1-v6、多流 | ~38 | Warp Shuffle 历史错；4 个 kernel 仅在 markdown；目标 70% 实测 64.3% |
| W3 | Transformer 执行本质 | ~3,600 | 5 个 .cu（可编译）+ 2 .py | ✅ Welford 对比 | ~33 | GEMM compute-bound 标注错；mini_engine 不接 Day3 优化 |
| W4 | FlashAttention 深挖 | ~3,200 | 0 个文件落盘 | ❌ benchmark 全是占位符 | ~31 | FA IO 简化为 O(Nd) 不严谨；N² 系数错；4 个文件不存在 |
| W5 | 推理系统与 KV Cache | ~4,900 | 2 .cu + 5 .py | ❌ 模拟为主 | ~36 | kv_cache.cu 越界 bug；mini_engine 不接 CUDA kernel；vLLM 源码无引用 |
| W6 | Batching & 调度 | ~5,260 | 7 .py | ❌ benchmark 用 SimulatedEngine | ~35 | LightLLM 覆盖浅；无真 batched forward；无真 nsys/ncu |
| W7 | 系统整合 | ~5,780 | 1 .py（真 CUDA via load_inline） | ❌ 全 time.sleep 模拟 | ~34 | custom_ops 有 5 个 bug；kernel 未被 engine 调用；LeetGPU 6 次重复 |
| W8 | 项目打磨 + 面试准备 | ~4,300 | 7 .py（工具类） | N/A | ~43 | "50+"虚标；KV Cache 2x 错；GEMM 层数 8/9 矛盾；mock 面试只是计时器 |

---

## 六、改进建议（按优先级）

### P0 —— 必须修复（否则面试时会被当场打脸）

1. **统一硬件参数**。以 `week1/day3/exercise/my_gpu_info.md` 的实测数据为唯一基准，全仓库搜索替换：
   - 19.5 TFLOPS → 104.75 TFLOPS
   - 1.55 TB/s HBM → 1.792 TB/s GDDR7
   - Ridge 12.6 → 58.45
   - 2048 threads/64 warps/32 blocks → 1536/48/24
   - 修复 `cuda_occupancy_calculator.py` 的 sm_120 表项
   - 重算所有 occupancy 手算题、Roofline 判定、bandwidth 利用率

2. **修复事实错误**：
   - Week2 Day1: Warp Shuffle 改为"Kepler(sm_30, 2012)引入；`_sync` 后缀 + mask 要求始于 Volta(sm_70, 2017) Independent Thread Scheduling"
   - Week4: FA IO 严格界改为 Θ(N²d²/M)，标注"O(Nd) 是 M=Θ(Nd) 时的特例"；统一 N² 系数（4N²+4Nd）；统一 N=4096 的 IO（4MB）与加速比（~50x）
   - Week8 Day4: KV Cache 改为 524 KB/token（与 Day6 对齐）
   - 统一 GEMM 优化层数（8 层）、Register Blocking 占比（40% 或 45% 二选一，建议用 Week2 Day6 实测的 30.8%）

3. **修复 CUDA bug**：
   - `week5/day2/kv_cache.cu` L156: 改 `new_len=5` 或扩大 `d_k2` 分配
   - `week7/day4/custom_ops_module.py`: atomicMax 改用 shared memory + warp shuffle block reduce max；修正 online softmax 更新公式；加 `at::cuda::getCurrentCUDAStream()`；加 `TORCH_CHECK(D<=256)` 与形状/连续性/dtype 检查
   - `week7/day2/full_scheduler.py` L192: `break` 改 `continue`

4. **把 markdown 里的代码落盘**。Week2 的 4 个 `.cu`、Week4 的 4 个文件全部创建为真实文件，确保 `nvcc` 能编译、`python` 能跑；删除 Day7 中"本周目录结构"与现实的矛盾。

### P1 —— 高优先级（补齐面试高频考点）

5. **新增 Tensor Core / WMMA 专项日**。建议在 Week2 插入或扩展 Day6：手写 `nvcuda::wmma` GEMM（FP16 输入/FP32 累加），对比 FMA GEMM，让 GEMM 占比从 64% 冲到 85%+。这是"手撕 GEMM 到 cuBLAS 90%"问题的标准答案。

6. **新增 CUTLASS 源码分析日**。Week2 Day3/Day4 当前只提了 CUTLASS 名字，应真正读 `cutlass/gemm/device/gemm.h`，理解 `ThreadblockShape/WarpShape/InstructionShape` 三级 tiling，并实例化一个 `cutlass::gemm::device::Gemm` 调用。

7. **新增 Triton 语言专题**。至少 1 天：用 Triton 重写 Softmax/GEMM/FlashAttention forward，对比 CUDA 实现的行数与性能。2025 年算子岗几乎必考。

8. **新增分布式推理专题**（建议 Week7 扩展或 Week8 前置）：
   - Tensor Parallelism（column-parallel / row-parallel QKV，all-reduce）
   - Pipeline Parallelism（1F1B，micro-batching）
   - NCCL collectives 与通信计算重叠（`torch.cuda.Stream` 双流）
   - 至少 5 道面试题

9. **真正执行一次 Profiling 并记录数据**。选 Week2 Day6 的 GEMM 或 Week4 Day2 的 FlashAttention，实际跑 `ncu --set full` 与 `nsys profile`，把报告截图、关键指标表填进 `profiles/` 和 `notes/`。空表是最大败笔。

10. **补齐 FlashAttention-3 / FlashDecoding**。Week4 Day4 的 FA2 之后加 FA3（Hopper async pipeline、FP8、warp specialization）；Week5 加 FlashDecoding（KV 跨 SM 切分，解决 decode 阶段单 query 并行度不足）。链接仓库已有的 `paper/flashattention3/`。

### P2 —— 中优先级（提升项目可信度与覆盖面）

11. **真正整合 Mini 引擎**。让 `custom_ops_module.py` 的 kernel 被 `mini_engine_v1` 调用；用真实 `torch.cuda.Event` 替换 `time.sleep`；实现真正的 batched forward（pad + merge）；把 `paged_attention.cu` 接入而非用 `torch.cat`。这样面试叙事才站得住。

12. **量化专题**。新增 1-2 天：W8A16/W4A16 weight-only 量化、INT8 KV Cache、FP8（Hopper/Blackwell）、AWQ vs GPTQ 对比，配套 dequant kernel。

13. **CUDA Graph 实操日**。静态捕获 + 动态 shape 处理（shape bucketing / replay），实测 launch overhead 降低。

14. **Speculative Decoding / Chunked Prefill / Prefix Caching 实操**。当前都是模拟，应至少有一个真实集成进 mini engine。

15. **多硬件对比专题**（针对国内市场）。至少 1 天对比 NVIDIA CUDA vs Ascend CANN/Ascend C++ vs Triton 的编程模型差异（grid/block 映射、同步原语、memory hierarchy）。这对投华为/昇腾岗位的候选人尤其重要。

16. **补 backward pass / 反向传播**。至少 1 天：FlashAttention backward（含 `L_i = m_i + log ℓ_i` 重计算 trick）、GEMM backward。算子岗会问。

17. **补 Ring Attention / 长上下文分布式注意力**。长上下文推理（百万 token）岗位热点。

### P3 —— 低优先级（打磨）

18. 修复所有硬编码 macOS 路径、broken SVG 引用、重复表格、`nvcc` 空格丢失
19. Mock 面试工具升级为 LLM 驱动的交互式面试官（当前只是带 prompt 的计时器）
20. `knowledge_selftest.py` 的 exact-match 评分改为归一化匹配
21. Week7 的 LeetGPU 题目去重（6 次 Matrix Transpose 改为不同题）
22. 补 vLLM V1 架构（AsyncLLMEngine、V1 scheduler）以反映 2025 年现状
23. 修复 Week8 LeetCode 总结表格的 copy-paste 错误（零钱兑换/课程表/最长有效括号/排序链表放错位置）
24. 修正 mock 面试时长声明（34 分钟而非 30 分钟）
25. 面试题数量"50+"改为"43+"或补齐到真实 50+

---

## 七、结论

这套教程**作为"概念入门 + 面试理论准备"材料质量上乘**，8 周路线设计、每日骨架、面试 Q&A 体系、online softmax 推导、occupancy 手算等都是亮点，适合"已经会写 kernel，想系统补 AI Infra 全景"的学习者。

但**作为"工程实战"和"求职硬通货"则存在系统性短板**：

- 硬件参数三套矛盾
- 核心算子代码未落盘
- profiling 几乎未执行
- Mini 引擎整合度不足
- 对 2025 年岗位高频考点（Tensor Core/CUTLASS/Triton/分布式/量化/FA3/FlashDecoding）覆盖严重不足

一名只学完这套教程的候选人，在面对以下追问时会比较被动：

- "手撕一个用 WMMA 的 GEMM"
- "你的引擎里 custom kernel 真的接进去了吗，profile 数据呢"
- "TP 下 QKV 怎么切，all-reduce 在哪个 stream"
- "你用的卡 ridge point 多少"（答 12.6 会暴露）
- "FlashAttention 的 IO 严格界是多少"（答 O(Nd) 不够）
- "用 Triton 写一个 softmax"

**建议优先级**：先修 P0（参数/事实/bug/落盘），再补 P1（Tensor Core/CUTLASS/Triton/分布式/真 profiling），即可将教程从"概念良好、工程不足"提升到"面试硬通货"水平。

---

## 附录：各周详细评估速查

### Week 1 —— GPU 执行本质 + Profiling
- **亮点**：occupancy 4 步手算法、`occupancy_verify.cu` 实测对比、bank conflict padding、`register_spill.cu` A/B demo、Day7 十题自测
- **问题**：硬件参数三套矛盾（B200/A100/真实 5090）；6 处 broken SVG；硬编码 macOS 路径；profiling 数据表全空白；Day5 "2-way conflict" 示例技术错误
- **面试准备度**：★★★★☆（概念强）／工程执行度：★★☆☆☆

### Week 2 —— GEMM & Kernel 优化
- **亮点**：GEMM v1-v6 实测（10.6%→64.3%）、多流 2.43x 实测、online softmax 推导、sector/cache-line 深挖、float4 向量化
- **问题**：Warp Shuffle 历史错（Blackwell→应 Kepler/Volta）；4 个 kernel 未落盘；目标 70% 实测 64.3%；Day2 预测 45.8% vs Day6 实测 30.8% 不一致
- **面试准备度**：★★★★☆ ／工程执行度：★★★☆☆

### Week 3 —— Transformer 执行本质
- **亮点**：Welford 并行合并、float4 + SASS LDG.128 分析、源码级 PyTorch/FasterTransformer、mini_engine.py 真集成
- **问题**：naive GEMM 错标 compute-bound；mini_engine 用 Day2 未优化 kernel；FP32 vs FP16 不一致；"Day16" 编号
- **面试准备度**：★★★★☆ ／工程执行度：★★★☆☆

### Week 4 —— FlashAttention 深挖
- **亮点**：online softmax 三公式完整推导、完整 forward kernel（markdown 内 305 行）、FA1/2/3 对比
- **问题**：4 个文件未落盘；IO 简化为 O(Nd) 不严谨；N² 系数错；同日 IO 2MB vs 4MB 矛盾；加速比 100x/51.6x/48.8x 三值；LeetGPU 与 Day2 重复
- **面试准备度**：★★★☆☆（forward 强，backward/IO 严格性弱）／工程执行度：★★☆☆☆

### Week 5 —— 推理系统与 KV Cache
- **亮点**：Prefill/Decode 量化分析、kv_cache.cu 真实现、paged_attention.cu 真实现（含 block table + online softmax）、vLLM 5 步 schedule 复刻
- **问题**：kv_cache.cu 越界 bug；mini_engine 不接 CUDA kernel；vLLM 源码无实际引用；多轮复用声称但未实现
- **面试准备度**：★★★★☆ ／工程执行度：★★★☆☆

### Week 6 —— Batching & 调度
- **亮点**：vLLM scheduler 520 行复刻（含两种抢占）、chunked prefill 模拟器、4 框架对比
- **问题**：benchmark 用 SimulatedEngine 非真引擎；无真 nsys/ncu；LightLLM 覆盖浅；无真 batched forward
- **面试准备度**：★★★★☆ ／工程执行度：★★☆☆☆

### Week 7 —— 系统整合
- **亮点**：custom_ops_module.py 真编译 CUDA via load_inline、concurrent_engine 真并发、full_scheduler 真调度逻辑
- **问题**：custom_ops 5 个 bug（atomicMax/online softmax/stream/硬编码/无 check）；kernel 未被任何 engine 调用；Day5 假集成；Day6 假 profiling；LeetGPU 6 次重复
- **面试准备度**：★★★☆☆ ／工程执行度：★★☆☆☆

### Week 8 —— 项目打磨 + 面试准备
- **亮点**：knowledge_selftest.py 自动判分、STAR 模板、系统设计题、8 周能力地图
- **问题**："50+"虚标；KV Cache 2x 错；GEMM 层数 8/9 矛盾；mock 面试只是计时器；LeetCode 表格 copy-paste 错；RTX 5090 用 A100 参数
- **面试准备度**：★★★★☆ ／工程执行度：★★★☆☆
