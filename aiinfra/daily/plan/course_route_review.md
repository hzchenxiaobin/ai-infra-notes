# AI Infra 10 周课程 · 整体路线评审

> 评审范围：`aiinfra/daily/` 下全部 10 周 markdown 课程（week1–week10 每天教程 + 周 README + plan/ 全部计划文档，不含 .svg)。
> 评审视角：资深 AI Infra 技术专家 / 大厂面试官 / 课程设计者。
> 评审约束：只评整体路线，不逐日抠细节（个别日级问题仅作为路线级证据引用）。
> 评审日期：2026-08-05

## 一、整体路线判断（结论先行）

**路线骨架是合理的，方向感清晰，但执行完成度拖了后腿。**

v2 版 10 周结构（W1 GPU 基础 → W2 Kernel 优化方法论 → W3 Tensor Core/CUTLASS → W4 算子+Triton → W5 FlashAttention 专题 → W6 推理系统/KV Cache → W7 Batching/调度 → W8 加速技术 → W9 分布式/MoE → W10 整合+面试）是一条教科书级的 LLM 推理方向成长链：

- **依赖链基本正确**：GEMM 优化(W2)→ WMMA/Tensor Core(W3)→ Softmax/LayerNorm 等融合算子(W4)→ Attention(W5)→ 系统层(W6/W7)→ 加速手段(W8)→ 规模化(W9)，每一环都用到上一环的原语（warpReduce、online softmax、tiling 反复复用），没有明显的前置倒置。
- **单主题周的设计好于旧 8 周版**：plan/ 里 `weekly_structure_review_and_replan.md` 对旧版"FA 跨 3 周碎片化、CUTLASS 远离 WMMA"的批评是对的，v2 的重组修掉了这些结构病。
- **"每日 LeetGPU 手撕 + LeetCode 算法 + Day7 复盘 + 限时手撕"的双轨闭环**是很好的设计，直接对标面试场景。

## 二、路线级的五个真实问题

### 1. 三代计划并存，迁移未完成 —— 最大的路线执行风险

plan/ 下同时存在：旧 8 周极简/详细版（自己标注"已过时"）、8 周体系 expanded(Day8–56 连续编号）、10 周 v2 计划。教程正文是旧周内容迁移来的，残留大量错位：

- W4 Day7 整篇是旧 W3 复盘，与本周内容对不上；W7 Day7 复盘的"Dynamic Batching、benchmark_engine_v1.py"在当前周不存在；W6 Day7 自称"Week 5"。
- 多处悬空引用（指向不存在的 kernel 文件、错误的 weekN/dayM 路径）；W3 的 Day2/3/5 关键 kernel 只内嵌在 README、未落盘。
- v2 计划自己列了 18 个"待开发"占位天，与实际教程状态的对应关系已无人维护。

**后果**：学习者按路线走会反复撞上"这里说的东西不存在"。路线图画对了，但路没修完。

### 2. 主项目贯穿性弱，整合风险全部压到最后一周

Mini 推理引擎本应是 10 周的脊椎，实际状态是：W6 v0 / W7 v1 是纯 Python 模拟器（部分用 sleep 模拟），W8 的"量化/投机解码/CUDA Graph 三选一接入"只有约 50 行伪代码，W9 整周不碰引擎，直到 W10 才"真整合 + 六步联调"。一个能写进简历的"真 GPU 推理引擎"在最后一周才拼装，稳定性测试用的还是模拟引擎——**项目烂尾风险高度集中**，而简历恰恰最需要这个项目是真的。

### 3. 覆盖面偏科：全押推理，训练侧和工程基建缺席

- 作为"LLM 推理优化工程师"课程覆盖很好；作为"AI Infra 工程师"课程，**训练侧几乎为零**（无 Megatron/FSDP/训练视角 3D 并行、无 optimizer/grad 通信）；W9 分布式全是概念 + 单卡 Python 模拟，没有任何一段真实 NCCL/多进程代码。
- **工程基建空白**：全程假设"会 C/C++"，但 CMake、gdb、Docker、git 工作流、CI 一天都没有；PyTorch 内部机制（autograd engine、dispatcher、caching allocator）只在 W5/W10 顺带用到，没有系统教学。

### 4. GPU 手感曲线断裂，难度前重后轻

W2–W5 是高密度手写 CUDA（GEMM 七层、FA fwd+bwd，连续 4 周高强度）；W7、W9 整周零 GPU 代码（纯模拟）；W8 只有一天有 kernel。学员在最需要保持手感的后半程反而 3 周不碰 nvcc。同时每周 24.5h 的强度对在职学习者偏紧，W2+W5 两座高峰连在一起，中途掉队概率高。

### 5. 配套资源有欠账

LeetGPU 映射表有 10+ 天"待定"无题解，且同一题跨多天重复（如 Matrix Transpose 出现 4 次）；W5 的 benchmark 表留了 nan 待补；全程单卡 RTX 5090，Ascend 对比只是打印表格的概念对照。

## 三、面向就业评估

| 目标岗位 | 匹配度 | 说明 |
|---|---|---|
| CUDA Kernel 工程师 | **高** | GEMM 七层、WMMA/CUTLASS、FA fwd/bwd、量化 kernel，手撕训练充足 |
| LLM 推理优化工程师 | **中高** | 知识链完整，但缺真实模型端到端部署与压测（vLLM 只是读代码，没真跑过一个 Qwen/Llama 出 TTFT/TPOT/吞吐报告） |
| 通用 AI Infra 工程师 | **中** | 训练侧、多卡实操、工程基建是短板 |

## 四、优先修改的 5 件事（按 ROI 排序）

1. **完成迁移清理**：归档/删除旧 8 周 plan 与 expanded，逐周修复 Day7 复盘错位和悬空引用。这是学习者体验的第一杀手，工作量可控。
2. **修复主项目贯穿线**：Mini 引擎每周五固定推进、每周产出真实可跑增量（W8 真接入 CUDA Graph/量化、W9 至少双进程演示），把整合风险从 W10 摊平到全程。
3. **W9 加一段真代码**：单机双进程 `torch.distributed`/NCCL allreduce 最小 demo，30 行就够，把"分布式纯概念"变成"写过一次"。
4. **补工程基建轨**：W1 前加 Day0(CMake/编译/nsys 安装验证/git 工作流），W4 或 W10 补一节 PyTorch Extension 构建链。
5. **清配套欠账**：补齐 LeetGPU 待定题解并去重；加一天"真实模型 vLLM 部署 + 压测报告"，让 W6/W7 的指标概念落地一次。

## 五、评分

**整体：78 / 100** —— 骨架 90 分，完成度 65 分。

| 维度 | 评分 |
|---|---|
| 体系完整性 | 75（推理链完整，训练/基建缺失） |
| 就业匹配度 | 80（kernel/推理岗位高） |
| 实践价值 | 72（前期强，后期模拟化） |
| 技术深度 | 85 |
| 学习难度（合理性） | 70（偏紧、前重后轻） |
| 项目质量 | 65（主线贯穿弱、整合靠后） |

## 六、一句话总结

这条路线的"地图"是对的，问题在"路况"——把迁移残留清干净、让 Mini 引擎每周都真实长大、给分布式补上第一段真代码，这套课程就能从"很好的自学笔记"变成"能交付结果的培养方案"。

---

# Week 1 逐日评审（GPU 执行模型与内存基础）

> 评审范围：week1/day1–day7 README 全文 + 各日 kernels/exercise/notes 实际文件核对。
> 评审日期：2026-08-05

**整体判断**：作为开篇周，选题和顺序（执行模型 → Occupancy → 认识硬件 → 内存层次 → Bank Conflict → Profiling → 复盘）是合理的，知识链完整。主要问题是：① LeetGPU 一道 Matrix Transpose 连做 4 天；② Day2/Day3 的 Occupancy 内容大面积重复；③ 关键数字互相打架（Ridge Point 58.45 vs 25、SRAM 100KB vs 192KB）；④ 多处 README 内嵌代码与仓库实际文件不一致。

## Day 1：GPU 执行模型基础（736 行）

1. **当前价值**：SM/Warp/SIMT/divergence/Grid-Block-Thread 一节课讲透，概念对应表和代际对比表（5090/A100/B200）质量高，是全课程的地基。
2. **是否必要**：**必要**，且作为 Day 1 无可替代。
3. **难度等级**：简单~中等（纯概念 + 20 行 hello_gpu）。
4. **建议修改**：
   - 修复标题重复字串；统一 `kernels/hello_gpu.cu`(8,1,1）与正文代码（4,2,1）的 block 配置，删掉 `exercise/` 下的近重复文件。
   - `__ballot_sync`、Ridge Point 等超前概念只保留"详见 Day N"的一句话指针，不展开。
   - 核实 B200 SM 数（文中 132，公开资料多为 148)。
5. **推荐补充资料**：NVIDIA CUDA C Programming Guide 第 2 章（Programming Model）；《CUDA C 编程权威指南》第 1-2 章。

## Day 2：Occupancy 与资源约束（699 行）

1. **当前价值**：Occupancy 三大约束、register spilling 检测（`-Xptxas -v`)、`__launch_bounds__`、`#pragma unroll` 寄存器权衡表——这是面试和实际调参最常用的一组知识，讲得有深度。
2. **是否必要**：**必要**，但需要与 Day 3 重新切分边界（见下）。
3. **难度等级**：中等（四步手算 + spilling 实验有一定门槛）。
4. **建议修改**：
   - 明确分工：Day 2 讲"occupancy 原理 + 资源约束 + 调参手段"，Day 3 只做"deviceQuery + 手算/API 验证"，把 Occupancy Calculator、延迟表、50-75% 经验法则从 Day 3 删掉（都在 Day 2 讲过）。
   - 修复路径混乱：实验 3 引用的 `occupancy_test_b.cu` 实际在 `exercise/` 不在 `kernels/`；三处基准路径写法统一。
   - "CUDA Occupancy Calculator Excel"说法过时（CUDA 12 起已并入 Nsight Compute)，改为教 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API(Day 3 已有，可提前合并）。
   - LeetGPU 换掉 Matrix Transpose，改用与 occupancy 更相关的题（如 ReLU，题解已归档）。
5. **推荐补充资料**：CUDA C Programming Guide 第 5 章（Performance Guidelines）的 occupancy 小节；Nsight Compute 官方 Occupancy 页面文档。

## Day 3：认识你的 GPU —— deviceQuery 与 Occupancy 计算（593 行）

1. **当前价值**：跑官方 deviceQuery、手算峰值算力/带宽/Ridge Point、手写 mini deviceQuery——"认识自己的硬件"这堂课很受用，`my_gpu_info.md` 实测留档是好习惯，也是全课程硬件数字的事实源。
2. **是否必要**：**必要**，但当前约一半内容（occupancy 理论）是 Day 2 的复读，需要瘦身。
3. **难度等级**：简单（跑命令 + 套公式）。
4. **建议修改**：
   - **修复数字硬伤**：面试要点 #3 的"平衡点约 25"必须改为 58.45；面试要点 #1 带宽公式补上 ×2；3.4 节 Excel 示例的 50% 与 3.5 手算 66.7% 二选一（以手算为准）。
   - 删重后补入更有价值的内容：显存 ECC、`multiProcessorCount` 与实际调度的关系，或把 Day 4 的 Ridge Point 应用提前到这里形成"算完就用"的闭环。
   - 修正 `--cc 8.0` 示例为 12.0；`maxBlocksPerMultiProcessor=24` 在 deviceQuery 输出里拿不到，注明来源（CUDA Occupancy API 或文档）。
   - LeetGPU 换掉 Matrix Transpose（如 Matrix Addition，题解已归档）。
5. **推荐补充资料**：NVIDIA 官方 CUDA Samples 的 deviceQuery 源码；TechPowerUp / NVIDIA 官方白皮书核对本卡参数。

## Day 4：Memory Hierarchy 深入（688 行）

1. **当前价值**：本周信息密度最高的一天——内存层次延迟表、sector/cache line 机制、coalesced vs stride 量化对比、AI 与 Ridge Point 应用、tiling 思想。把"为什么内存比计算重要"讲透了。
2. **是否必要**：**必要**，核心日。
3. **难度等级**：中等~困难（sector 映射 + tiling 对初学者偏陡，smem `__shared__`/`__syncthreads` 当天首现即用）。
4. **建议修改**：
   - 修复 0.14% vs 0.66% 的自相矛盾（以 0.14% 为准）；LayerNorm AI 的字节数核算（读 4+写 4=8B）重算。
   - tiled 转置用到的 `+1` padding 要么给一句自洽解释，要么把 tiled 版整体移到 Day 5(padding 当天现学现用，更顺）。
   - 统一 README 内嵌代码与仓库 `transpose.cu`（两个 kernel + rand 填充）的版本；删掉 4.1 节重复的内存层次图。
   - LeetGPU 保留 Matrix Transpose（这天用最贴切），但明确这是该题唯一一次布置。
5. **推荐补充资料**：《CUDA C 编程权威指南》第 4 章（内存模型）；Nsight Compute Memory Workload Analysis 官方文档。

## Day 5：Bank Conflict 分析与实践（404 行）

1. **当前价值**：32 bank 结构、四种访问模式、padding 推导（33%32=1）讲得清楚；是 Week 2 GEMM tiling 的必要前置。
2. **是否必要**：**必要**，但体量和深度可以微调。
3. **难度等级**：中等。
4. **建议修改**：
   - **修复理论自相矛盾**：正文 5.2 说 `tile[tid % 2]` 是广播不算 conflict，扩展实验 1 却标注为"2-way conflict"——以正文为准改实验。
   - 任务 1 的 main 补上 D2H 拷回和结果校验（现在 `h_out` 分配后从未使用）；grid(1,1) 单 tile 的规模能否稳定复现 Checklist 要求的"慢 2x"，要么放大规模要么放宽标准。
   - 承接 Day 4：把 tiled 转置移入本日任务（padding 现学现用），LeetGPU Reduction 的 warp shuffle 部分给 3 行最小铺垫或标注"Week 2 Day 1 详讲"。
5. **推荐补充资料**：CUDA C Programming Guide 中 shared memory bank 小节；经典博文 "Using Shared Memory in CUDA"(NVIDIA Developer Blog)。

## Day 6：Nsight Profiling 实战（465 行）

1. **当前价值**：ncu/nsys 分工、瓶颈三分类、Roofline、bank_conflict 五步走分析实例——把前 5 天的 kernel 全部用工具重新量一遍，闭环设计好。
2. **是否必要**：**必要**，方法论日。
3. **难度等级**：中等（命令多但均为复用已有 kernel）。
4. **建议修改**：
   - 修复标题重复、"本 week's kernel"中英混写；任务 2-4 的二进制路径 `./kernels/xxx` 全部改为实际路径（`day2/kernels/...` 等）并补编译命令。
   - 6.6 节"假设数据"的 1,048,576 次 conflict 与 grid(1,1) 小 kernel 量级不符，换成 Day 5 真实采集的数据。
   - 与 Day 5 任务 3 的 ncu 指标几乎重复——本日重点应放在"跨 kernel 横向对比 + Roofline 定位"，而不是重测一遍。
   - 实验 3 的 CUDA Graph 标注"Week 8 详讲"；链接未引用的 `notes/day6_nsight_profiling.md`。
5. **推荐补充资料**：Nsight Compute / Nsight Systems 官方 Getting Started；NVIDIA GTC 演讲 "CUDA Kernel Optimization"(Roofline 实战类）。

## Day 7：总结与复盘（741 行）

1. **当前价值**：知识地图 + 优化决策树 + Q1-Q10 自测 + 完成标准清单，复盘框架完整；"补实验、手绘 6 张图"的弹性安排务实。
2. **是否必要**：**必要**，符合课程 Day7 模板。
3. **难度等级**：简单（无新代码，纯复盘）。
4. **建议修改**：
   - 清理"8 周计划"遗留（2 处）改为 10 周；修复 SRAM 容量三处矛盾（100KB vs 192KB，以 100KB smem/L1 共享口径统一）。
   - "本周目录结构"声称的 `week1/kernels/` 不存在，按实际布局（各 day 自己的 kernels/）重写。
   - 复盘表如实反映 LeetGPU 实际布置（改完 Day2/3 题目后是 6 天 4 题）；`exercise/leetgpu_week1_review.md` 要么创建要么删引用。
   - 综合练习的 GEMM tiling 要求超出 Week 1 已教内容，降级为"naive GEMM + ncu 测 memory-bound"或推迟为 Week 2 预习。
5. **推荐补充资料**：CUDA C Programming Guide 第 5 章全文（作为本周收口阅读）；自己整理的一页纸"优化决策树"（已含在任务中，保留）。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | 执行模型 | 必要 | 简单~中 | 标题/代码文件不一致 |
| 2 | Occupancy | 必要 | 中 | 与 Day3 重复、路径混乱 |
| 3 | 认识 GPU | 必要（需瘦身） | 简单 | 数字硬伤（25 vs 58.45) |
| 4 | 内存层次 | 必要 | 中~难 | 0.14% vs 0.66%、padding 未教先用 |
| 5 | Bank Conflict | 必要 | 中 | 广播/conflict 自相矛盾 |
| 6 | Profiling | 必要 | 中 | 路径全错、假设数据失真 |
| 7 | 复盘 | 必要 | 简单 | 8 周遗留、目录结构失真 |

---

# Week 2 逐日评审（CUDA Kernel 优化方法论）

> 评审范围：week2/day1–day7 README 全文精读 + 各日 kernels/ 实际文件与跨日引用核对。
> 评审日期：2026-08-05

**整体判断**：本周是全课程技术含量最高的一周，Day1(Warp Shuffle)、Day2(Register Blocking)、Day3(float4 + v1–v6 全系列实测）三天构成了一条非常扎实的 GEMM 优化主线，尤其 Day3 的"诚实实测 + 每层收益归因"是整套课程的标杆写法。但**本周也是数字矛盾最严重的一周**：Day2 实测 32%、Day3 实测 63%、Day4 又称 46%/20-30%、Day7 验收标准写 65%+——四套口径并存；Day4 的"后三层"叙事与 Day3 实际内容（已讲到整合版）脱节；Day7 要求口述 FlashAttention，而本周根本没教（旧 8 周版迁移遗留）。

## Day 1：Warp Shuffle 原语与 Warp/Block Reduce（557 行）

1. **当前价值**：四个 shuffle 原语、mask/width 参数、down vs xor 的区别、两级归约结构——讲得干净准确，代码可编译且带 CPU 验证（PASS 留档，247 GB/s)。"第二级为什么由 Warp 0 做"这类追问式设计很好。LeetGPU Prefix Sum(`__shfl_up_sync`）与当日内容对称，选题精准。
2. **是否必要**：**必要**。shuffle 是 Week2 GEMM 写回、Week4 Softmax/LayerNorm、Week5 FA 的公共原语，放在本周开篇正确。
3. **难度等级**：中等（概念新但代码量小，实测 0.068ms 的即时反馈对建立信心有帮助）。
4. **建议修改**：
   - `🎯 目标`只有 5 条，补齐为模板要求的 6 条。
   - 正文 `warpReduceSumXor` 在 kernel 中未被调用，加一句"留作实验 1 对比用"避免读者疑惑。
   - 面试要点 1 说"从 Blackwell 架构开始必须使用 `_sync` 版本"——应为 Volta(CC 7.0)，正文 1.2 节写对了，面试答案写错了，统一。
5. **推荐补充资料**：NVIDIA Developer Blog《Using CUDA Warp-Level Primitives》；CUDA C Programming Guide 的 Warp Shuffle 附录。

## Day 2：Register Blocking 与 2D Tiling（484 行）

1. **当前价值**：GEMM 优化的"分水岭日"——三级数据复用、寄存器账本（~88 regs)、线程二维映射讲得清楚；2.5 节 Double Buffering 与 Occupancy 隐藏延迟的对比表是全文亮点（把两种延迟隐藏机制的正交关系讲透了）。代码含 cuBLAS 对比 + 正确性校验，完整。
2. **是否必要**：**必要**，本周核心日之一。
3. **难度等级**：困难（从"每线程算 1 元素"跳到"每线程 8×8 子块 + 协作加载"，是全课程第一道真正的门槛）。
4. **建议修改**：
   - **修复目标与实测的矛盾**：目标/Checklist 写"达 cuBLAS 40%+(4096)"，而预期输出里 4096 实测只有 32.3%——要么改目标为"30%+（以 Day3 整合版 63% 为终态）"，要么如实说明"2048 达 39%,4096 受 wave 数限制"。
   - 学前导读说"Week 1 中我们学习了 Shared Memory Tiling GEMM"——Week1 只教过转置 tiling，没教过 GEMM tiling；补一个 5 行的 naive GEMM + smem tiling 回顾（或直接引用 Day3 系列的 v2）消除断层。
   - 性能分层表（Naive 1-3% → RegBlk 40-60%）与 Day3 实测（v3=30.8%）不一致，以 Day3 的 v1–v6 实测表为唯一口径。
5. **推荐补充资料**：Simon Boehm《How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance》(siboehm.com，与本日完全同构的经典长文）；NVIDIA DL Performance Guide 的 Matrix Multiplication 章节。

## Day 3：float4 向量化 + 整合版 GEMM（814 行）

1. **当前价值**：**本周质量最高的一天，也是全套课程的范文**。cache line/sector 的定量讲解（32B sector 利用率账）、"float4 不减字节数、减的是指令数"的三收益归因、shuffle 写回的 64 条指令成本核算、"为什么整合版最终没用 shuffle"的诚实说明、v1–v6 全系列实测 + 寄存器/smem 用量表 + 每层增益归因——这是真正工程师视角的写法，面试能直接背。
2. **是否必要**：**必要**，本周核心日。
3. **难度等级**：困难（814 行信息密度极高，建议拆出阅读节奏提示）。
4. **建议修改**：
   - 修复标题重复字串；小节编号 6.1/6.2/6.3 改为 3.x;"今日总结"自称"Day 6"改为 Day 3。
   - **与 Day4 重新划界**：本日实际已讲完整合版（含双缓冲 v6)，按现有分工 Day4 无米下锅。建议把"v6 双缓冲 + cp.async"整块移到 Day4，本日收在 v5 整合版，使"前四层/后三层"的分工名副其实。
   - 任务 3 指标表列名"Day 2 / Day 6（整合版）"改为"Day 2 / 本日整合版"；LeetGPU Histogramming 与 SKILL 映射表（GEMM）不一致，同步更新映射表。
   - "LeetCode 机动补漏日"安排合理，保留。
5. **推荐补充资料**：继续 siboehm GEMM 系列（第 5-7 篇：vectorized loads / double buffering）；CUTLASS 的 `pipeline` 文档（为 Day4/Week3 铺垫）。

## Day 4：GEMM 优化续篇 —— 后三层路径与 cuBLAS 对比（358 行）

1. **当前价值**：唯一的增量价值是 **cuBLAS 三基准口径**(FP32 68 / TF32 89 / FP16 210 TFLOPS）和"面试时如何报数字"的话术——这很重要。但其余内容（shuffle 写回、double buffer、整合版）与 Day3 大面积重复，且全是伪代码片段，没有可编译的新产出。
2. **是否必要**：**部分必要，当前形态需要重做**。三基准口径值得保留，"后三层"叙事已被 Day3 掏空。
3. **难度等级**：中等（概念都在 Day3 见过，无新代码）。
4. **建议修改**（问题最多的一天）：
   - **数字必须统一**：本日"整合版 4096 实测 ~4.4ms / FP32 46%"与 Day3 的"3.178ms / 63.4%"直接冲突；任务 3 的"7-Layer Benchmark"(dbuf 6.555ms / 20.3% TF32）是第三套数字；今日总结写"60-80%"、面试要点表格写"L7 累计 82%"——四处互相矛盾。以 Day3 的实测系列为唯一事实源重写本日。
   - 任务 1/2 的 shuffle 写回、cp.async 双缓冲目前是**伪代码**，落盘为真实可编译的 `day4/kernels/gemm_cp_async.cu`(`#include <cuda_pipeline.h>`，呼应 Day3"真双缓冲需要 cp.async"的诚实结论），否则本日没有产出。
   - **LeetCode 与 Week1 Day2 完全重复**(283/11/15/42 四道双指针原题照搬），按 8 周计划第 2 周补正确题目（矩阵主题 Day5 的 73/54/48/240 实际从未被布置，可移到这里）。
   - 补 LeetGPU 题（映射表标"待定"，建议 2D Convolution 或复用 GEMM 做 cp.async 改造）。
   - "Day 3 的 Register Blocking"应为"Day 2"。
5. **推荐补充资料**：CUDA C Programming Guide 的 `cp.async` / `cuda::pipeline` 章节；CUTLASS 3.x 的 multistage pipeline 设计文档。

## Day 5：CUDA Streams 与异步执行（589 行）

1. **当前价值**：Default Stream 隐式同步的坑、Pinned Memory 与 DMA 的机制解释（含"为什么 DMA 处理不了 page fault")、多流 H2D/Compute/D2H 重叠——理论清晰，且**实测 2.43x 加速 + nsys timeline 真实截图**留档，教学闭环完整。"为了看清 kernel 条把循环调到 10000 次"的调试备注是难得的工程细节。
2. **是否必要**：**必要**。Stream 是 Week8 CUDA Graph、Week9 通信重叠的前置，位置合理。
3. **难度等级**：中等（概念直观，代码模式固定）。
4. **建议修改**：
   - "今日总结"自称"Day 3"改为 Day 5；小节编号 3.x 改为 5.x。
   - LeetGPU 用的是 Matrix Multiplication，但 Week1 Day6 已布置过同题——按 SKILL 映射表改回 **2D Convolution**（题解已归档）。
   - 与 GEMM 主线略脱节：可加一个 10 行的小实验"用 2 个 stream 跑 Day3 的 GEMM 前后两半"，把 stream 接回本周主线（也为 Week9 通信计算重叠埋钩子）。
5. **推荐补充资料**：NVIDIA Developer Blog《GPU Pro Tip: CUDA 7 Streams Simplify Concurrency》；CUDA C Programming Guide 的 Asynchronous Concurrent Execution 章节。

## Day 6：Nsight Compute 性能分析（387 行）

1. **当前价值**：Warp Stall Reasons 分类表、指标正常范围表、"Profile→优化→再 Profile"闭环——比 Week1 Day6 更深一层（stall 归因是新增量）。用 Day2 的 GEMM 做分析对象，承接主线。
2. **是否必要**：**必要，但需瘦身**。ncu/nsys 分工、Roofline、基本命令在 Week1 Day6 已完整教过，当前约 40% 是复读。
3. **难度等级**：中等。
4. **建议修改**：
   - **修复硬伤**：4.5 节写"RTX 5090 约 25 FLOP/byte"，与事实源（58.45）和本文件面试要点 5(58.45）自相矛盾——这是 reference 硬性条款明确禁止的，必须改。
   - 删掉与 Week1 Day6 重复的工具介绍，本日聚焦三个新增量：Stall Reasons 归因、Source View(`-g -lineinfo`)、优化闭环实操；开头加一句"Week1 Day6 已教基础命令，今天深入归因"。
   - 任务 3 的"假设 ncu 输出"换成对 `register_blocking_gemm.cu` 的真实采集数据（Day3 有实测传统，别在这里断掉）。
   - "今日总结"自称"Day 4"改为 Day 6。
5. **推荐补充资料**：Nsight Compute Kernel Profiling Guide（官方 stall reason 定义）；GTC 演讲《CUDA Performance Analysis with Nsight Compute》。

## Day 7：限时 Kernel 手撕 + GitHub 整理 + 性能对比报告（741 行）

1. **当前价值**：限时手撕（30min Reduce + 60min GEMM）带**评分标准表**和**易错点复盘表**，是面试训练的正确形态；"warp 1 能不能做第二级归约"的知识补充质量很高。手撕这一核心机制必须保留。
2. **是否必要**：**必要**，验收日。但当前混入了大量本周没教的内容，需要手术。
3. **难度等级**：困难（限时手写本来就是高压设计，合理）。
4. **建议修改**：
   - **删除/替换 FlashAttention 内容**：任务 3 的 FA 口述、Online Softmax 三公式默写、完成标准里的"FA 简化版测试通过"、常见误区表里的 FA 条目——本周 Day1-6 没有任何 FA 内容（旧 8 周版迁移遗留）。替换为"60 分钟手撕 float4 整合版 GEMM"或"口述 GEMM 七层优化路径"（与面试要点 4 呼应）。
   - **修复目录结构**：任务 5 和"📁 本周目录结构"里的文件映射全部错位（multi_stream 在 day5 不在 day3;`day5/flash_attention.cu`、`cutlass_gemm_example.cu`、`wmma_gemm.cu` 均不存在；gemm 三文件在 day3 不在 day6)，按实际布局重写，删掉"（CUTLASS 已移至…)"之类的迁移批注残片。
   - 任务 7（LeetCode 回顾）**整段出现了两次**，删一；任务 6 报告路径 `week10/day3/notes/` 改为本日 `day7/notes/`。
   - "Week 2 → Week 3 衔接"把 Week3 描述为"Transformer 算子手写"——实际 Week3 是 Tensor Core/CUTLASS，重写衔接段；完成标准的"65%+"与 Day3 实测 63% 对齐。
   - 补 LeetCode Day5（矩阵 4 题：73/54/48/240）的归属（建议移到 Day4，见上）。
5. **推荐补充资料**：限时手撕就用本周自己的 kernel 做题库；额外可读 FlashAttention 论文 Section 3 作为 **Week 5 预习**（而不是假装本周教过）。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | Warp Shuffle | 必要 | 中 | 质量高，仅小瑕疵（Volta 误写 Blackwell) |
| 2 | Register Blocking | 必要 | 难 | 目标 40%+ vs 实测 32%；GEMM tiling 前置断层 |
| 3 | float4 + 整合版 | 必要 | 难 | 课程标杆；与 Day4 分工需重划 |
| 4 | 后三层 + 三基准 | 部分必要 | 中 | **四套性能数字互斥；伪代码无产出；LeetCode 抄了 Week1** |
| 5 | CUDA Streams | 必要 | 中 | 实测截图优秀；LeetGPU 与 Week1 重复 |
| 6 | Nsight Compute | 必要（需瘦身） | 中 | **Ridge Point 25 vs 58.45 硬伤**；40% 复读 Week1 |
| 7 | 限时手撕验收 | 必要 | 难 | **FA 内容系迁移遗留（本周未教）；目录结构全错位；任务 7 重复两次** |

**本周最优先修复的三件事**：① Day4 的性能数字统一（四套口径→Day3 实测一套）；② Day7 的 FA 遗留内容替换；③ Day6 的 Ridge Point 25→58.45。

---

# Week 3 逐日评审（Tensor Core 与 CUTLASS）

> 评审范围：week3/day1–day7 README 全文精读，结合仓库文件存在性核对（week3 仅 day1/day4 有落盘 kernel)。
> 评审日期：2026-08-05

**整体判断**：本周主线（WMMA 教学版 → smem tiling → mma.sync/ldmatrix → CUTLASS → double buffer → profiling → 复盘）是一条干净的能力爬坡链，"诚实标注实测 vs 预估"的口径贯彻全周，值得肯定。三个系统性问题：① **性能数字再次多套口径**——Day2 实测 tiled 42.5%，Day3/Day5 引用同一 kernel 时却写 55-65%，Day6 Roofline 表又冒出 110/125/170 TFLOPS 的第四套数字；② **Day2/3/5 的核心 kernel 全部不在仓库**（只在 README 内嵌，Day2 host 端还有 `...` 省略），而 Week1/2 建立的规矩是"完整可编译落盘"；③ **LeetCode 全周跑偏**——8 周计划第 3 周主题是"链表与数学"，实际布置的全是 DP 题（DP 是第 7/8 周的主题），且 283/240 等题重复出现。

## Day 1：Tensor Core 与 WMMA（553 行）

1. **当前价值**：Tensor Core 架构演进表、fragment 生命周期、混合精度策略讲得清楚；教学版实测 30.4%(TF32)/14.9%(FP16)+ 差距归因（无 smem tiling、1 warp/block）的诚实声明很好；ncu 任务给了 Tensor Core 专属指标（`sm__pipe_tensor_op_hmma`)。
2. **是否必要**：**必要**，本周地基。
3. **难度等级**：中等（概念多但代码是单 warp 的简化版，反而降低了门槛——合理的教学设计）。
4. **建议修改**：
   - 修复标题重复字串；"调用流程总结"代码块**重复出现两次且内容不同**（一版有 cuBLAS 对比、一版没有），删一。
   - **消除内部矛盾**：目标 #4 写"实测 ~33%"，但今日总结 #4 写"手写 WMMA GEMM 达到 cuBLAS 85%+"、面试要点 3 标题也写"达到 85%"——全部统一为实测口径（TF32 30% / FP16 15%)。
   - 学前导读"Day 6 的整合版 GEMM 64.3%"是 Week2 旧编号，改为"Week 2 Day 3"；且 64.3% 是 FP32 FMA 口径，与本日的 TF32 口径不同，注明。
   - LeetGPU 任务链接应指向 leetgpu.com 题目页 + 题解页两个链接（现在只有题解链接）；与 SKILL 映射表（Histogramming）不一致，二选一同步。
5. **推荐补充资料**：CUDA C++ Programming Guide 的 WMMA 章节；NVIDIA 博客《Programming Tensor Cores in CUDA 9》（虽老但概念准确）。

## Day 2：手写 WMMA GEMM 与 cuBLAS 性能对比（554 行）

1. **当前价值**：**本周最有教学发现的一天**——"小矩阵 tiled 反而比 naive 慢 4x、交叉点在 2048"是实测得出的反直觉结论，直接通向 auto-tuning 的必要性，面试可用。bank conflict padding vs swizzle 的引入自然。实测 42.5%/20.8% 口径诚实。
2. **是否必要**：**必要**，从"能跑"到"能用"的关键一步。
3. **难度等级**：困难（多 warp 协作 + fragment 数组 + padding，本周第一个陡坡）。
4. **建议修改**：
   - **代码必须落盘**：`kernels/wmma_gemm_tiled.cu` 在仓库不存在，且 README 内嵌代码 host 端有 `...` 省略（不可直接编译）——违反课程"完整可编译"规矩，补齐完整 main + 正确性验证并落盘。
   - **目标与实测对齐**：目标 #5 写"实测 ~50-65%"、今日总结写"55-65%"，实测是 42.5%(4096 TF32)——统一为 42%/21% 口径。
   - 任务 4 引用的 `kernels/wmma_gemm_compare.cu` 也不存在，落盘或改为"在同一 main 里调用两个 kernel"。
   - 补 LeetGPU 任务（映射表标"待定"；建议与 tiling 相关的 Matrix Transpose 变体或直接复用 GEMM 题做 tiled 改造）。
5. **推荐补充资料**：CUTLASS 文档的 GEMM Pipeline 章节；siboehm GEMM 系列的 smem tiling 篇（kernel 3-4)。

## Day 3：mma.sync 指令与 ldmatrix（482 行）

1. **当前价值**：fragment 线程-数据映射表、ldmatrix 变体（.x4/.x2/.trans)、对齐约束、"为什么 FA 源码直接用 mma.sync"的面试题——把 Week5 读 FA 源码需要的底层词汇一次性备齐，定位精准。
2. **是否必要**：**必要**，但它是"读懂源码"导向的一天，动手产出 weakest。
3. **难度等级**：困难（PTX 内联汇编 + 寄存器布局，全课程概念密度最高点之一）。
4. **建议修改**：
   - **代码落盘并实测**：`kernels/mma_sync_gemm.cu` 不存在，正文 kernel 是"概念示意"（写回部分有 `...` 省略）;3.6 节和任务 2 的性能表（68.9% 等）全是"预期"而非实测——与 Day1/2 建立的实测传统断裂。要么落盘实测，要么把数字明确标注"预估"并降低精确度（不写 78.0 这种假精确值）。
   - 引用 Day2 数据时用实测值（42.5%)，不要另起一套（55.2%)。
   - 任务 1 的参考答案先说"地址被忽略"再自我纠正为"4 个独立 8×8 矩阵分组"——保留正确的第二段即可，错误的第一段会误导。
   - 补 LeetGPU 任务（映射表"待定")。
5. **推荐补充资料**：NVIDIA PTX ISA 文档的 `mma.sync` / `ldmatrix` 条目（权威定义）;GTC 演讲《Developing CUDA Kernels for GEMM with Tensor Cores》。

## Day 4：CUTLASS 源码分析 + CuTe 概念铺垫（447 行）

1. **当前价值**：三级 tiling 抽象、模板参数详解表、Epilogue Fusion、NumStages——是"读懂工业级库"的正确切入点；CuTe 最小铺垫（Shape+Stride=Layout）为 Week5 读 FA 源码埋伏笔，克制而必要。
2. **是否必要**：**必要**（面试刚需），但当前的"源码分析"名不副实。
3. **难度等级**：中等（概念为主，代码是模板实例化）。
4. **建议修改**：
   - 标题重复字串修复；学前导读"Day 4 的 WMMA 教学版"应为"Day 1"；任务 3"对比 Day 4 的手写 WMMA"应为 Day 1/2。
   - **"源码分析"要落到实处**：当前只讲了模板接口，没有带读者打开过一个真实源码文件。增加 20 行导读：`cutlass/include/cutlass/gemm/device/gemm.h` 的类注释 + `threadblock` 目录结构，或明确把标题改为"CUTLASS 概念与调用"。
   - 预期输出整表是 `0.0xx` 占位（已诚实标注未实跑）——尽快补真实数据，否则本周唯一一次"手写 vs CUTLASS vs cuBLAS 三方对比"落空；sm_120 用 CUTLASS 2.x 的 ArchTag=Sm80 兼容性需要在正文给结论而不是挂在"B2 任务"悬空引用上。
   - LeetGPU 与 Day 1 重复（同为 Batched MatMul)，换题；LeetCode 的 283/240 是重复题。
5. **推荐补充资料**：CUTLASS GitHub 仓库的 `examples/00_basic_gemm`;CuTe 官方教程（`media/docs/cute/00_quickstart.md`)。

## Day 5：Double Buffering + Benchmark 框架（559 行）

1. **当前价值**：cp.async vs memcpy 的路径对比、2/3/4-stage 的 smem 容量账、"小矩阵 double buffer 反而变慢"的收益条件分析——理论部分扎实；任务 3 把 Day1/2 实测与 Day3/5 预估分列标注，口径诚实。
2. **是否必要**：**必要**(double buffer 是 FA K/V 循环的标准写法，Week5 直接用）。
3. **难度等级**：困难。
4. **建议修改**：
   - **修复任务 1 代码的正确性问题**：内嵌 kernel 里 `producer_acquire()` 包的是普通赋值拷贝（`smemA[...] = A[...]`)，这不是异步拷贝——`cuda::pipeline` 只有配合 `cuda::memcpy_async` 才会真正重叠。当前代码实际是"串行拷贝 + 无效 pipeline 注解"，跑起来不会有任何 double buffer 收益。改用 `cuda::memcpy_async` 或 `cp.async` PTX，并落盘 `kernels/wmma_gemm_dbuf.cu`。
   - **b_frag 布局前后不一致**：Day 2 用 `row_major`（注释说明 smem 转置加载），Day 5 同样的 smem 加载代码却声明 `col_major`——二者必有一错，实测验证后统一（Day7 手撕模板沿用了 Day5 的写法，会连带出错）。
   - 任务 2"预期输出"里 Day2 列（2.810ms/55.2%）与 Day2 实测（3.0738ms/42.5%）不符，改为引用 Day2 实测值；dbuf 列标注"预估"。
   - 补 LeetGPU 任务（映射表"待定")。
5. **推荐补充资料**：libcu++ 的 `cuda::pipeline` 官方文档；CUTLASS `mainloop` 源码中 multistage 的实现（`mma_multistage.h`)。

## Day 6：Profiling —— Tensor Core 利用率（412 行）

1. **当前价值**：把 Day1-5 的性能数字逐个用 ncu 指标"打开"（瓶颈转移：HBM→smem→compute)，"推理值 vs 实测"的限制声明（`ERR_NVGPUCTRPERM`）写得负责任；面试 Q&A 的"四步定位法"可直接复用。
2. **是否必要**：**必要**，方法论闭环日。
3. **难度等级**：中等。
4. **建议修改**：
   - **任务可执行性**：任务 1-3 依赖 `./wmma_tiled`、`./wmma_dbuf`、`./cublas_bench` 三个二进制，但对应源码均未落盘（见 Day2/5)——补齐源码，否则本日任务全部无法执行。
   - 任务输出的精确数字（48.2%、62.8%、91.3%）是推理值，在"预期输出"块上标注"推理示意，非实测"，与 6.3 节的声明对齐。
   - **Roofline 表数字冲突**：WMMA tiled "Achieved 110 TFLOPS"与 Day2 实测 44.7 TFLOPS 差 2.5 倍，cuBLAS 170 与 Day1 实测 215.2 不符——重算或删除该表。
   - AI 公式排版错误："AI = 2×4096³/(3×4096²)×2"应为"÷(3×4096²×2)"（结果 1365 是对的，式子写错）。
5. **推荐补充资料**：Nsight Compute Kernel Profiling Guide;Roofline 原始论文（Williams et al., 2009）或其综述。

## Day 7：复盘与手撕（473 行）

1. **当前价值**：三个手撕模板（WMMA 骨架/fragment 生命周期/double buffer 结构）粒度合适，"面试官可能追问"的设计好；混合精度/FP8 对比表是本周最有沉淀价值的加餐，直接服务 Week8。
2. **是否必要**：**必要**。
3. **难度等级**：中等。
4. **建议修改**：
   - **知识地图是 ASCII 图**，违反课程"禁止 ASCII 图、一律 SVG"的硬性规范，重绘为 `week3/images/` 下的手绘风 SVG。
   - **性能演进总表口径错误**："FMA GEMM (W2) ~64%"被放进 TF32 列——W2 的 63.4% 是 FP32 FMA cuBLAS 口径，对 TF32 口径应是 ~41%。修正并加一行口径说明，顺手解决全课程"同一个 GEMM 三个百分比"的混乱。
   - 手撕 1 模板的 `b_frag` 声明（`col_major`）需与 Day2/Day5 修正后的正确写法一致。
   - 缺"今日总结"小节（模板必有）；补 LeetCode 本周回顾表（当前 7 天无一处回顾，且本周题目整体跑偏为 DP——见下）。
   - **修正本周 LeetCode 编排**：按 8 周计划第 3 周"链表与数学"(24 题）重新分配 Day1-6，把现有 DP 题归还给第 7/8 周。
5. **推荐补充资料**：FP8 格式论文《FP8 Formats for Deep Learning》(Micikevicius et al.);CUTLASS 的 FMHA 示例（为 Week5 预习）。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | Tensor Core + WMMA | 必要 | 中 | 总结写 85% vs 实测 30% 自相矛盾；标题/流程块重复 |
| 2 | smem tiling WMMA | 必要 | 难 | **kernel 未落盘且代码不完整**；目标 55-65% vs 实测 42% |
| 3 | mma.sync + ldmatrix | 必要 | 难 | 性能数字全是"预期"；kernel 未落盘 |
| 4 | CUTLASS + CuTe | 必要 | 中 | 无真实源码导读；输出全占位；LeetGPU 与 Day1 重复 |
| 5 | double buffer | 必要 | 难 | **内嵌代码实为同步拷贝，pipeline 注解无效**;b_frag 布局存疑 |
| 6 | ncu profiling | 必要 | 中 | 任务依赖的三个二进制无源码；Roofline 表数字冲突 |
| 7 | 复盘手撕 | 必要 | 中 | ASCII 知识地图违规；FMA 口径错误；缺今日总结 |

**本周最优先修复的三件事**：① Day2/3/5 的 kernel 落盘并实测（连带修复 Day5 的假 double buffer 代码）；② 统一性能口径（全周以"Day1 30%/Day2 42% 实测 TF32"为锚，其余标预估）；③ LeetCode 改回第 3 周"链表与数学"主题。

---

# Week 4 逐日评审（Transformer 算子手写 + Triton）

> 评审范围：week4/day1–day7 README 全文精读，结合仓库文件存在性核对（week4 落盘代码：day2 `softmax_layernorm.cu` + day4 三个 Triton 文件）。
> 评审日期：2026-08-05

**整体判断**：本周的"四日核心"(D1 trace → D2 手写 Softmax/LayerNorm → D3 Welford/Backward → D4 Triton）内容质量是全课程最好的区段之一——D2 的 block reduce 工程化、D4 的 Triton 三方实测（含"Triton 小矩阵反而慢"的诚实数据）都达到范文水准。但本周有两个严重的结构问题：① **Day7 是旧 Week3 复盘的整体搬运**，它回顾的 Attention IO、Mini 引擎、源码分析等内容在当前 Week4 的 Day1-6 根本不存在，目录树还写着 `week3/`；② **Day5/Day6 与 Day4 大面积重复**(autotune 机制讲两遍、决策表讲三遍、FA benchmark 重复），且 Day5 同一文件内两套 GEMM 数字互斥（97.5% vs 72%)。另有 LeetCode 编排半周链表半周 DP 的分裂（D1/D2 用第 3 周链表主题，D3-D6 又用 DP)。

## Day 1：Trace Transformer 推理流程（Prefill/Decode)（376 行）

1. **当前价值**：把 Prefill/Decode、M=1 GEMM 退化、KV Cache 读取账（256MB/token)、6 类算子 bound 分类一次讲清，AI 计算（384 vs 2 FLOP/Byte）与 Ridge Point 挂钩规范；`trace_transformer.py` 内嵌完整可跑，torch.profiler 工具教学恰到好处。这是 Week6-7 推理系统周的认知地基。
2. **是否必要**：**必要**，本周开篇位置正确。
3. **难度等级**：简单~中等（概念为主，代码是调库）。
4. **建议修改**：
   - 修复标题重复字串；"为什么重要"里"Week 3 全周的地基"改为"Week 4"；目标只有 4 条，补到 6 条；**补"今日总结"小节**（当前缺失，模板必有）。
   - 任务 3 的预期输出是 `xxx us` 占位——跑一次真实 profile 填入实际数字（本周 D4 已有实测传统，别在开篇断掉）。
   - LeetGPU Matrix Multiplication 已是第三次布置（W1D6、W2D5)，换题或与 SKILL 映射表（1D Convolution）对齐。
   - "Day 6 会详细分析 fusion 机会"指向不准（fusion 在 D3/D7),Day6 实际是三方 profiling。
5. **推荐补充资料**：vLLM 博客《How vLLM serves LLMs》；HuggingFace 的 LLM Inference 优化系列（prefill/decode 部分）。

## Day 2：手写 Softmax 与 LayerNorm Kernel（587 行）

1. **当前价值**：**本周最扎实的一天**。safe softmax 数学等价证明、三遍扫描的 HBM 账、两级 block reduce 模板复用 Week2 原语（依赖链真实成立）、kernel 落盘且 PASS 留档（maxDiff 1e-9/1e-6)、ncu 验证任务完整。"一行一个 block"的并行映射讲解清晰，是 Week5 FA 行块处理的直接前置。
2. **是否必要**：**必要**，本周核心日。
3. **难度等级**：中等（有 Week2 原语铺垫，坡度设计合理）。
4. **建议修改**：
   - 修正前瞻引用："分块版留到 Week 4 FlashAttention"应为 **Week 5**;"Day 4 的 Attention IO 分析"在当前 Week4 不存在（在 Week5 Day1)，改为正确指针。
   - LeetGPU 用了 GroupNorm，与 SKILL 映射表（Softmax）不一致——二者都对，但需同步映射表；注意 Day3 的 RMSNorm 只在扩展实验出现，未落到 LeetGPU 任务（映射表 W4D3 = RMS Normalization)。
   - 目标 5 条补到 6 条。
5. **推荐补充资料**：PyTorch ATen `SoftMax.cu`(warp 级 vs block 级 dispatch);OneFlow 的 softmax 优化博客（中文，deep dive 向量化）。

## Day 3：LayerNorm 优化与 GEMM Backward 数据流（514 行）

1. **当前价值**：Welford 递推 + 合并公式 + 数值稳定性对比（catastrophic cancellation）讲得完整；GEMM backward 数据流（dA=dC@B^T/dB=A^T@dC）用有限差分验证，是 Week5 FA backward 的必要前置；RMSNorm 扩展实验贴近生产（LLaMA/Qwen)。
2. **是否必要**：**必要**。Welford + backward 两块内容都不可替代，但一天装两块略挤。
3. **难度等级**：中等~困难（Welford 并行合并公式是本周数学密度最高点）。
4. **建议修改**：
   - **kernel 落盘**：`layernorm_welford.cu` 不存在，且内嵌代码的跨 warp 合并部分是 `...` 省略——补齐并落盘（Day2 已有完整 blockReduce 模板可复用）。预期输出已标"量级预估"，落盘后替换为实测。
   - 补 LeetGPU 任务（映射表 W4D3 = **RMS Normalization**，与实验 3 正好呼应，把实验 3 升级为任务 4)。
   - LeetCode 改回本周的链表主题（D1/D2 用的是链表，本日突然变成 DP 152/918/410)，保持周内一致。
   - Day2 预告的"Day 3 读 FasterTransformer `generalLayerNorm`"实际只有一句话提及——要么在 3.2 节后加 15 行 FT 源码导读，要么修正 Day2 的预告。
5. **推荐补充资料**：FasterTransformer `layernorm_kernels.cu`(Welford 生产实现）；CS231n backprop notes（反向数据流）。

## Day 4：Triton 语言专题（694 行）

1. **当前价值**：**本周信息价值最高的一天**。block-level 抽象与 CUDA 概念对照表、`tl.dot` 自动 Tensor Core、autotune 机制、三方实测数据齐全且诚实（"Triton 小矩阵 0.27x 反而慢");Triton FA 从 naive 到 8x 加速的实测梯度漂亮。三个 kernel 全部落盘。
2. **是否必要**：**必要**(Triton 是算子岗 JD 高频要求），但 FA 部分与 Week5 有前置倒置。
3. **难度等级**：中等~困难（新语言 + 一天三个算子，体量 694 行偏饱）。
4. **建议修改**：
   - 修复标题重复；任务 1 的 kernels/ 链接误指向 `week10/day7`；正文中"Week 3 Day 1-3 我们手写了 CUDA softmax"应为"Week 4 Day 2-3";"Week 2 Day 5 手写 CUDA FlashAttention"不存在（那是 Week5 内容）——FA 相关引用全部修正。
   - **前置倒置处理**:Triton FA 用了 online softmax 三件套，但 online softmax 要到 Week5 Day2 才正式推导。两个选择：a) 本日 FA 改为"黑盒调用 + 预告 Week5 拆开";b) 在 1c 节前加 10 行 online softmax 最小推导。推荐 b)，成本最低。
   - 实验 2 引用的"Day 1 手写 CUDA GEMM(~30%)"表述混乱——手写 CUDA GEMM 是 Week2/3 的产出，注明确切出处（Week3 Day1 WMMA 教学版 30%)。
5. **推荐补充资料**：Triton 官方 tutorials(01-vector-add → 03-matrix-multiplication → 06-fused-attention，与本日结构完全同构）。

## Day 5：项目推进 —— Triton 三方 Benchmark（380 行）

1. **当前价值**：决策表（何时 Triton 何时 CUDA）和"benchmark 框架统一接口"有独立价值；5.3 的实测发现（Triton 大矩阵 GEMM 97.5%、softmax 追平 torch）如果属实是很好的素材。
2. **是否必要**：**部分必要**——与 Day4 重复面太大（autotune 机制、三方对比表、FA 数字、决策表全部在 Day4 出现过）。建议合并或重定位。
3. **难度等级**：中等。
4. **建议修改**：
   - **同一文件数字互斥必须解决**:5.3 节 GEMM 表（4096 Triton 0.661ms / 97.5%）与任务 2 预期输出（1.810ms / 72%）是两套数字；今日总结又说"70-80%"。保留一套实测，其余删除。注意这两套与 Day4 的 0.6391ms/1.00x 又是第三套——全周统一到一个 benchmark 脚本产出。
   - **落盘 `kernels/benchmark_triton.py`**（不存在，Day6 任务也依赖它）;FA 对比表的"官方 FA ~0.4ms"等数字注明来源（实跑还是引用）。
   - 与 Day4 去重：autotune 机制不再重讲，本日聚焦"框架 + 决策表 + 边界 case（小矩阵退化）";LeetCode 回到链表主题；补 LeetGPU（映射表"待定"，建议 Argmax 或复用 GEMM 做 autotune 练习）。
   - "比 Day 1 手写 CUDA(30%）快 2-3x"的对比基准写错出处（应为 Week3 Day1)。
5. **推荐补充资料**：triton-bench(Meta 的 Triton benchmark 套件）;`torch.compile` 的 Inductor 文档（理解 autotune 的生产形态）。

## Day 6：Profiling —— Triton vs CUDA vs PyTorch（334 行）

1. **当前价值**："用 ncu 解释 Triton 为什么快"(Tensor Core 利用率 68% vs 25%）的叙事方向正确；调试决策树（面试题 5）实用。
2. **是否必要**：**边缘必要**——是 Week1 Day6、Week2 Day6、Week3 Day6 之后的第四个 profiling 日，新增量最薄（约 30% 新内容），且依赖的产出物（benchmark 脚本、wmma_naive 二进制）都不存在。
3. **难度等级**：中等。
4. **建议修改**：
   - 与 Day5 合并或改为"半天任务"：本日真正的新增量只有"Triton kernel 的 ncu 指标长什么样"+"Triton PTX 导出分析"（实验 1)，可把实验 1 升级为任务主线，其余删减。
   - 预期输出标注口径：ncu 指标表是推理值还是实测需明示（Week3 Day6 已建立标注先例）。
   - 修复依赖：任务 1/3 引用 `kernels/benchmark_triton.py`、`benchmark_softmax.py`（均不存在），对比对象"Day 1 手写 CUDA/wmma_naive"指向错误且不存在的二进制。
   - torch.profiler 教学与 Day1 重复，保留一处即可；LeetCode 回链表主题；补 LeetGPU。
5. **推荐补充资料**：Nsight Compute 的 "Kernel Profiling Guide";Triton 的 `TRITON_DUMP_PTX` / `MLIR_ENABLE_DUMP` 调试文档。

## Day 7：Transformer 算子分类与总结（493 行）

1. **当前价值**：**内容本身是优秀的复盘**——算子分类决策树、Prefill/Decode 两张 AI 分类表、五步诊断法、常见误区表，都是高质量的面试资产。问题是它复盘的不是这一周。
2. **是否必要**：**必要，但必须重写**。当前文本回顾的 Day3 源码分析、Day4 Attention IO(`attention_naive.cu`)、Day5 Mini 引擎（`mini_engine.py`)、Day6 profiling 报告在当前 Week4 的 Day1-6 全部不存在；衔接段说"Week 4 将深入 FlashAttention"（实为 Week5);"Week 2 Day 5 已学 FA 简化版"为假；目录结构写的是 `week3/` 树。
3. **难度等级**：简单（复盘日）。
4. **建议修改**：
   - **按当前 Day1-6 实际内容重写复盘主线**:trace 方法 → 手写 reduce 算子 → Welford/backward → Triton 三算子 → 三方 benchmark → profiling。保留算子分类表（它本来就是按 Day1 的框架写的，可以保留）和五步诊断法。
   - 任务 2 产出清单重写：`trace_transformer.py`、`softmax_layernorm.cu`、`layernorm_welford.cu`（待落盘）、三个 Triton kernel、benchmark 表——删掉 `attention_naive.cu`/`mini_engine.py`/`profiling_report.md`。
   - 任务 3 的 LeetGPU/LeetCode 回顾表与实际布置全对不上（表列 Day3 Reduction/Day4 Softmax Attention/Day6 RMS，实际是 Day1 MatMul/Day2 GroupNorm/Day4 MatMul;LeetCode 表列链表题而 Day3-6 实际布置 DP)——按修正后的实际布置重列。
   - 衔接段改为 Week 4→**Week 5**(FlashAttention 专题），删掉"Week 2 Day 5 已学 FA"的假引用，改为"online softmax 将在 Week5 Day2 正式推导，本周 Day4 已在 Triton 里用过黑盒版"。
   - 位置错误的"任务 4 LeetGPU"浮在推荐资源之后，移回任务区。
5. **推荐补充资料**：FlashAttention 论文 Section 2-3(Week5 预习）；其余沿用现有推荐资源表（质量不错）。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | Trace Prefill/Decode | 必要 | 简单~中 | 预期输出全占位；缺今日总结；LeetGPU 第三次重复 |
| 2 | Softmax/LayerNorm | 必要 | 中 | 质量范文级；前瞻引用（FA/IO）指向错误周 |
| 3 | Welford + Backward | 必要 | 中~难 | kernel 未落盘、跨 warp 合并省略；LeetCode 跑偏 DP |
| 4 | Triton 专题 | 必要 | 中~难 | 链接错指 week10;FA 前置倒置（online softmax 未教先用） |
| 5 | 三方 Benchmark | 部分必要 | 中 | **与 Day4 大重复；同文件两套 GEMM 数字互斥**;benchmark 脚本未落盘 |
| 6 | 三方 Profiling | 边缘必要 | 中 | 第四个 profiling 日，新增量最薄；依赖的二进制不存在 |
| 7 | 算子分类复盘 | 必要（需重写） | 简单 | **整篇是旧 Week3 复盘，复盘的课程本周没上过**；目录树写 week3/ |

**本周最优先修复的三件事**：① 按当前 Day1-6 重写 Day7 复盘；② Day5/Day6 向 Day4 收敛去重并统一 GEMM 数字（一周一套实测）;③ Day3 的 `layernorm_welford.cu` 与 Day5 的 `benchmark_triton.py` 落盘。

---

# Week 5 逐日评审（FlashAttention 专题）

> 评审范围：week5/day1–day7 README 全文精读，结合仓库文件存在性核对（落盘：day1 `flash_attention.cu`、day3 `flash_attention_v2.cu`、day4 `gemm_backward.cu` + `flash_attention_backward.py`、day5 benchmark 脚本）。
> 评审日期：2026-08-05

**整体判断**：本周是课程的"皇冠周"，选题和深度都对得起这个定位——D1 的 online softmax 完整推导（含每步归一化 vs 末尾归一化两种变体）、D3 的完整 kernel、D4 的 backward(logsumexp 推导 + gradcheck 实测 PASS）是高质量的。三个系统问题：① **D1 与 D2 大面积重复**（三公式推导、IO 分析、tiling 各讲两遍）;② **SRAM 数字三连错**——D2/D3/D7 都写"RTX 5090 164 KB/SM"(164 KB 是 A100，5090 是 100 KB，D1 自己写的是对的）;③ **旧日编号全面漂移**(D1 总结自称 Day 5、D2 自称 Day 1、D3 自称 Day 2、D5 自称 Day 6、D6 自称 Day 4)。另有两处内嵌 Python 缩进损坏（D2/D5),D5 的 benchmark 手写/官方两列全是 nan（核心交付落空）。

## Day 1：FA CUDA 简化版 + Attention IO 分析（896 行）

1. **当前价值**：单日信息量全周最大——Attention 基础七问（√d 推导、causal mask)+ online softmax 逐步推导（汇率类比 + 两种归一化变体等价性）+ 论文 Algorithm 1 对照 + 完整可编译 kernel(PASS 留档）+ SRAM 账目正确（48KB 静态上限、5090 100KB/SM)。"省略 1/√d 并显式声明"的教学处理诚实。
2. **是否必要**：**必要**，但体积过大（896 行），且与 D2 重复约 50%。
3. **难度等级**：困难（内容密度全课程之最，建议拆节奏提示）。
4. **建议修改**：
   - "今日总结"自称"Day 5"、任务 4"对应 Day 5 的主题"——改回 Day 1。
   - 与 D2 去重：三公式推导保留在本日（写得更完整）,D2 改为纯"论文精读 + IO 严格界 + 教学版 PyTorch 验证"，删掉重复的推导节。
   - 0.x 节的 Attention 基础与 `topics/transformer` 专题重复，可收缩为速查表 + 链接（W4D1 已有此先例）。
   - 目标 5 条补到 6 条；LeetCode 用了"第 2 周 Day 5 矩阵"（本周应为第 5 周二叉树主题，且矩阵题在 W2 已被跳过/待补）——按 8 周计划重新对齐。
5. **推荐补充资料**：FlashAttention 论文（2205.14135)Section 1-2;Tri Dao 的 Stanford CS25 演讲视频。

## Day 2：论文精读与 Online Softmax 推导（511 行）

1. **当前价值**：独立增量只剩三块——IO 严格界 Θ(N²d²/M)、wall-clock 只有 2-8x 的解释（T_gemm vs T_memory)、PyTorch 教学版验证。其余（三公式推导、tiling、SRAM 约束）是 D1 的复读。
2. **是否必要**：**部分必要**。按"论文精读"重定位后值得保留；当前形态与 D1 重复过半。
3. **难度等级**：中等（若按重定位后）。
4. **建议修改**：
   - **修复内嵌 Python 缩进损坏**：`flash_attention_pytorch` 整个函数嵌进了 `standard_attention` 里、`return O` 缩进错、main 嵌套——按现状不可运行，必修。
   - **SRAM 数字改错**："RTX 5090 上限 164 KB/SM"出现两次（1.2 节、面试要点 4)，改为 100 KB（事实源/D1 口径）;Br=Bc=128 的 96 KB 案例需注明必须动态 smem + `cudaFuncSetAttribute`（超 48KB 静态上限）。
   - 修正引用："Week 3 Day 4 我们用 ncu 实测"不存在；"今天是 Week 4 全周的理论基石"应为 Week 5；"今日总结"自称 Day 1。
   - PyTorch 教学版"比 standard 还慢"的教学点保留（好素材），可加事件计时拆解 launch overhead 归因。
   - LeetGPU 与映射表（Attention vs Causal Self-Attention）二选一同步。
5. **推荐补充资料**：FA 论文 Theorem 2(IO 严格界证明）；论文的 Table 2(benchmark 数据）。

## Day 3：手写完整 FA Forward Kernel（640 行）

1. **当前价值**：从 D1"一线程一行"升级到"warp 管 8 行 + warp shuffle 归约"的完整 kernel,batch/head grid 设计、寄存器压力分析（acc[8][64]≈120 regs）讲得实在；kernel 落盘且 PASS 留档（maxDiff 1.31e-4)。本周工程核心日。
2. **是否必要**：**必要**。
3. **难度等级**：困难（本周动手最高峰）。
4. **建议修改**：
   - "今日总结"自称"Day 2"；学前导读"Day 1 我们用纯 PyTorch 实现"应为 **Day 2**(PyTorch 教学版在 D2)。
   - "48 KB ≤ 164 KB (RTX 5090 上限）"改为 100 KB；实验 1 的 Br=Bc=128(96KB）需注明动态 smem opt-in，否则编译即失败。
   - 面试要点 3 的寄存器账（"120 reg/thread → 每 SM ~544 线程"）建议与 Week1 Day2 的 occupancy 四步法呼应一句。
   - 说明 CPU 验证只覆盖第一个 head(B=0,H=0)，多 head 正确性靠结构对称性——或补一个全量对比的开关。
5. **推荐补充资料**：FA 官方 CUDA 源码的 `flash_fwd_kernel.h`（对照找差距，为 D6 铺垫）;LeetGPU MHA 题解。

## Day 4：FA Backward 与 GEMM Backward（541 行）

1. **当前价值**：**本周质量最高的一天**。logsumexp"无损压缩"推导（$L_i = m_i + \log l_i$)、$D_i = O_i \cdot dO_i$ 化简、Algorithm 2 分块重算，全部落到可验证代码：`gemm_backward.cu` 有限差分 PASS、`flash_attention_backward.py` gradcheck 1e-15 量级 PASS、saved-tensor 内存对比表（32.9x）实测留档。LeetCode 选题与当日主题的类比（栈=recomputation）有巧思。
2. **是否必要**：**必要**，是"读过论文"和"能落地"的分水岭日。
3. **难度等级**：困难（本周数学最高峰，但推导链条完整，坡度可接受）。
4. **建议修改**：
   - 修正引用：今日总结"Day 3 读官方源码"在当前周不存在（源码导读在 `_supplementary/` 与 D6)，改为正确指针。
   - 任务 2 预期输出里 `GPU Time: 0.0xx ms` 仍是占位，补实测。
   - 实验 1 的"复用 Week 2 Day 2 GEMM 模板达 cuBLAS 50%+"与 W2 实测（RegBlock 32-39%）口径冲突，改为"达 W2D3 整合版水平（~63% FP32 口径）"或放宽为方向性目标。
   - 与 W4D3 的 GEMM backward 有~15%重复（dA/dB 公式），可加一句"已在 W4D3 学过，这里快速过"并聚焦 FA 特有部分。
5. **推荐补充资料**：FA 论文 Appendix B(backward 推导）;PyTorch `autograd.Function` 官方教程（自定义 backward 工程范式）。

## Day 5：性能对比 + 接入 Mini 引擎（420 行）

1. **当前价值**：benchmark 维度设计（N/B/H/d 四维度）和"ncu 验证 IO 随 N 线性增长"的方法论正确；Std 列实测数据（0.053→9.611ms）留档。
2. **是否必要**：**必要，但当前交付是空心的**——Hand/Off 两列全 nan，本周唯一一次"手写 vs 官方"的定量对比没有发生。
3. **难度等级**：中等。
4. **建议修改**：
   - **修复内嵌 Python 缩进损坏**（同 D2，整段不可运行）。
   - **补上缺失的两列**：`mini_engine_fa`（在 `_supplementary/from_w5d2`）接入或直接用 D3 的 v2 kernel 包一个 PyTorch wrapper 补 Hand 列；`flash-attn` 装不上就明确写"环境未就绪"并给替代（PyTorch SDPA 的 FA 后端）——别让验收表停在 nan。
   - 修复路径：任务 3 引用的 `day2/kernels/flash_attention_v2.cu` 应为 `day3/kernels/`。
   - 标题与内容不符：week README/目标提到"接入 Mini 引擎"，集成内容已在 `_supplementary`，要么链接过去要么改 week README。
   - 今日总结自称"Day 6"；面试要点 5 末尾有三行无关的孤儿 bullet；LeetGPU MHA 与 D3 重复（映射表本日是 Matrix Copy)。
5. **推荐补充资料**：flash-attn 官方 benchmark 脚本（`benchmarks/flash_attention.py`);Nsight Compute 的 `dram__bytes` 指标文档。

## Day 6：FA-2 论文与官方源码（452 行）

1. **当前价值**：FA2 三改进（non-matmul 减半、seq 并行、occupancy）讲得清楚；FA3 部分（TMA、warp specialization pingpong、FP8 布局手术、incoherent processing）是全课程少见的"下一代硬件"深潜，质量高，且链接到 paper 笔记。
2. **是否必要**：**必要**(FA1/2/3 演进是面试必考），但动手任务落空。
3. **难度等级**：中等。
4. **建议修改**：
   - "今日总结"自称"Day 4"；学前导读"Day 3 我们读了官方源码"为悬空引用——源码导读实际在 `_supplementary/from_w4d7`，要么链接要么改述。
   - 任务 2/3 依赖的 `flash_attention_fa2.cu` 未落盘，且引用路径 `day2/kernels/flash_attention_v2.cu` 错误（应为 day3)——把 FA2 风格改造（ROWS_PER_WARP 8→4）做成真实可跑的对比实验并留档，否则本日零产出。
   - "明天把 FA 集成到 Mini 引擎"——明天是 Day 7 复盘，改为正确预告。
   - LeetCode 标注"第 4 周 Day 4"与 D4 重复（同一天计划被布置两次），顺延到 Day 5。
5. **推荐补充资料**：FA2 论文（2307.08691);FA3 论文（2407.08608）+ 课程 paper 笔记；Colfax 的 CUTLASS Hopper 系列（TMA/warp specialization 扩展阅读）。

## Day 7：复盘与手撕（423 行）

1. **当前价值**:10 道 Q&A 覆盖全周追问链、限时手撕三题（60+15+15）设计合理；IO 复杂度表把 forward/backward 统一起来。
2. **是否必要**：**必要**，但手撕模板有正确性问题。
3. **难度等级**：中等。
4. **建议修改**：
   - **知识地图是 ASCII 框图**，违反"禁止 ASCII 图"规范（与 W3D7 同样问题），重绘 SVG。
   - **手撕 1 模板不可作为手写参考**：每线程声明 `S[Br][Bc]`、`acc[Br][d]`(Br=Bc=64 时是每线程 4096+ 个 float 的寄存器数组），且循环结构是"每线程遍历所有行"——既编译不出也教错并行模型。改用 D3 v2 kernel 的 warp-per-rows 结构做模板骨架。
   - **IO 口径统一**：本日用 O(Nd²) 严格界，而 D1/D2/D5 用 O(Nd) 简化——补一行换算说明（Θ(N²d²/M),M=Θ(Nd) 时 O(Nd)),Q3 的推导结论（O(Nd)）与表格（O(Nd²)）自相矛盾也需修。
   - "性能对比总表（Day 5 实测口径）"的数字（手写 ~1.8ms、官方 ~0.4ms 等）并非来自 D5 实测（D5 两列是 nan)——要么标注"示意"，要么等 D5 修复后回填。
   - Q7 的"RTX 5090 164 KB/SM"改为 100 KB；补 LeetGPU/LeetCode 本周回顾表（映射表 W5D7 = GPT-2 Transformer Block)。
5. **推荐补充资料**：本周 D1-D6 自身即是最佳材料；额外可读 Tri Dao 的 FA2 博客。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | FA 简化版 + IO | 必要 | 难 | 896 行过载；与 D2 重复 50%；旧编号 |
| 2 | 论文精读 | 部分必要 | 中 | **内嵌 Python 缩进损坏**;164KB 错误；与 D1 重复 |
| 3 | 完整 FA Forward | 必要 | 难 | 质量好；引用错位（PyTorch 版在 D2);164KB 错误 |
| 4 | FA Backward | 必要 | 难 | **本周最佳**；仅引用与口径小问题 |
| 5 | 性能对比 | 必要（需修） | 中 | **Hand/Off 两列 nan，核心交付落空**；脚本缩进损坏 |
| 6 | FA2/3 演进 | 必要 | 中 | FA2 改造实验未落盘，本日零产出；路径错 |
| 7 | 复盘手撕 | 必要 | 中 | **手撕模板并行模型错误**;ASCII 图违规；164KB 错误 |

**本周最优先修复的三件事**：① D5 benchmark 补上手写/官方两列（含脚本缩进修复）;② D7 手撕模板换成 D3 的正确并行结构；③ 全周 SRAM 数字 164KB→100KB(D2/D3/D7 共 4 处）+ IO 口径统一说明。

---

# Week 6 逐日评审（推理系统基础与 KV Cache）

> 评审范围：week6/day1–day7 README 全文精读，结合仓库文件存在性核对（落盘：day1 `prefill_decode_simulation.py`、day2 `kv_cache.cu`、day3 `mini_vllm_scheduler.py`、day4 `paged_attention.cu`、day5 `mini_engine_v0.py`、day6 `flash_decoding.cu`、day7 `week5_summary.py`)。
> 评审日期：2026-08-05

**整体判断**：本周是迁移质量最好的一周——知识链（Prefill/Decode → KV Cache → vLLM 架构 → PagedAttention → Mini 引擎 → FlashDecoding）顺序正确、环环相扣，D2/D3/D4 三个核心代码全部落盘且带 PASS 验证，D3 的 mini 调度器甚至附了抢占/livelock 的实测记录。主要问题：① **D1 与 Week4 Day1 重复约 50%**(Prefill/Decode 在 W4 已完整教过）;② **Day7 又是旧周复盘搬运**（全文自称"Week 5"，目录树写 `week5/`，且它回顾的 Day6=profiling 日、profile_engine_v0.py 在当前周不存在——本周实际没有 profiling 日，打破了"D6=profiling"的周模板）;③ 多处内嵌 Python 缩进损坏（D1/D5，落盘文件本身正常）;④ D6 多处写"80 个 SM"(RTX 5090 是 170)。

## Day 1：推理流程 —— Prefill vs Decode（536 行）

1. **当前价值**:TTFT/TBT/TPOT 指标体系、Decode 四大优化方向表、"TTFT 与 TBT 优化手段几乎不重叠"的洞察是新增量；模拟脚本落盘且有实测输出。Ridge Point 58.45 引用规范。
2. **是否必要**：**必要，但需瘦身**——Prefill/Decode 对比表、M=1 退化、Roofline 分析、torch.profiler 教学在 W4D1 已完整教过，本日约一半是复读。
3. **难度等级**：简单~中等。
4. **建议修改**：
   - 与 W4D1 去重：本日聚焦三个增量——时延指标体系（TTFT/TBT/TPOT/E2E)、KV Cache 收益直觉、四大优化方向预告；两阶段对比只留一张速查表 + 链接 W4D1。
   - 修正引用："Week 4 我们把 FlashAttention 吃透"应为 Week 5;"从 Week 5 开始进入推理系统"应为 Week 6;"Week 3 Day 1 第一次画过 Prefill/Decode"应为 Week 4 Day 1。
   - 修复内嵌 Python 缩进损坏（整段函数嵌套进 `forward`)；落盘文件正常，保持两者同步。
   - 预期输出里 TTFT 115.7ms 与 standalone Prefill 0.132ms 差近千倍——这是首次调用 warmup 未排除所致，加一句解释或修正测量（否则与"Prefill compute-bound"的叙事自相矛盾）。
5. **推荐补充资料**：AnyScale 的 Continuous Batching 博客；vLLM 论文 Section 1-2（动机部分）。

## Day 2：实现 KV Cache（含 GQA/MQA/MLA 变体）(494 行）

1. **当前价值**：本周地基日，质量范文级——5D 布局、三种分配策略演进逻辑（浪费→碎片→分页）、GQA/MQA/MLA 对比表 + 口算示例（MHA 524KB→GQA 131KB→MQA 16.4KB→MLA 65KB)、KVCache 类落盘 + 多轮追加验证 PASS。MLA 的讲解（低秩潜在向量）覆盖了 DeepSeek 热点，面试价值高。
2. **是否必要**：**必要**，本周核心日。
3. **难度等级**：中等。
4. **建议修改**：
   - 修复标题重复字串（"实现 KV Cache"出现两次）。
   - 任务 3 的 ncu 命令 profile 的是 `cudaMemcpy`（不是 kernel),`--kernel-name regex:memcpy` 抓不到——memcpy 不走 SM kernel，应用 nsys 而非 ncu，或改成直接实现实验 3 的 `append_kernel` 再 profile（顺便把实验 3 升级为正式任务，教学版 append 的低效是现成素材）。
   - "KV Cache bytes per token: 8192"与文中 524 KB 的口径（教学配置 vs LLaMA-7B）加一句衔接说明，避免读者混淆。
5. **推荐补充资料**：GQA 论文（2305.13245)、MLA 出处 DeepSeek-V2 论文（2405.04434);vLLM 的 `BlockSpaceManager` 源码（预习 D4)。

## Day 3：vLLM 整体架构分析（700 行）

1. **当前价值**：本周最丰满的一天——三层架构、SequenceGroup 设计动机（beam search 共享 prompt)、三重预算、Recomputation vs Swapping 的代价分析；mini 调度器 310 行纯标准库可跑，实验 1 附**真实实测记录**(req1 被抢占 6 次、14 轮收敛）并点出 livelock 边界；V0→V1 演进附录是跟踪工业界的好示范。
2. **是否必要**：**必要**。
3. **难度等级**：中等（系统概念多但有模拟器可玩，坡度合理）。
4. **建议修改**：
   - 内嵌代码末尾多出一个空 ``` 围栏；面试要点 5 末尾有一行孤儿 bullet（"Continuous Batching…跨平台通用")，清理。
   - `SchedulingBudget.can_add` 里 `self.num_seqs` 等字段先使用后声明（dataclass 字段顺序）值得一句说明，或直接给可运行版。
   - 任务 3 的源码对照要求 `pip install vllm`——给一个无安装的替代路径（直接打开 GitHub 指定文件链接 + 行号），否则无 GPU/无环境学员这一天任务 3 无法完成。
   - LeetGPU Top-P Sampling 与 SKILL 映射表（Speculative Decoding Verification）不一致，二选一同步。
5. **推荐补充资料**：vLLM SOSP 2023 论文；vLLM V1 架构博客（vllm.ai);Orca 论文（Continuous Batching 原始出处，OSDI 2022)。

## Day 4：vLLM Worker 与 PagedAttention（459 行）

1. **当前价值**：OS 分页类比表、block table 间接寻址代码、CoW refcount 机制、block_size=16 的权衡分析全部到位；kernel 落盘且用**故意打乱的物理 block(7,1,12,3）验证间接寻址正确性**（max_diff=0)，测试设计有说服力；与 W5 FA 的衔接（online softmax 复用）真实成立。
2. **是否必要**：**必要**，本周核心日。
3. **难度等级**：中等~困难（间接寻址 + online softmax 融合是全周最硬的 kernel)。
4. **建议修改**：
   - 清理面试要点 5 末尾两行孤儿 bullet。
   - kernel 是逐 token 串行扫描的教学版——补一句性能定位说明（"本 kernel 验证寻址正确性，生产版按 FlashDecoding 切 KV，见 Day 6")，把 D4→D6 的钩子在正文挂上。
   - LeetGPU Causal Self-Attention 与 W4D7/W5D2 重复，按映射表换题或说明复用理由。
   - 实验 3（连续 vs paged 性能对比）建议给出实测参考值——"间接寻址开销 <1%"是面试好数字，值得跑一次留档。
5. **推荐补充资料**：vLLM 论文 Section 3-4(PagedAttention 机制）;vLLM 源码 `csrc/attention/` 的 paged attention kernel。

## Day 5：项目推进 —— Mini 推理引擎 v0（487 行）

1. **当前价值**：课程主项目线的第一个里程碑——5 组件拆解表与 vLLM 角色对照、"with/without cache 逐 token 一致"的正确性验证设计好；随机权重"乱码生成"的预期管理写得到位。
2. **是否必要**：**必要**，项目主线日。
3. **难度等级**：中等。
4. **建议修改**：
   - 修复内嵌 Python 缩进损坏（整段嵌套）;落盘文件正常，保持同步。
   - 修正自指："Week6 再加 session 管理"应为 Week7（本周就是 Week6)。
   - **验证 depth 不足**：目前只验证"cache 一致"，建议补实验 2 的实测数据（with/without cache 的 TBT 曲线）——这是本日最能产出"简历数字"的实验，不该留白。
   - LeetGPU GPT-2 Transformer Block 与映射表（Token Embedding）不一致，且 D7 又用了同一题（周内重复）——二选一。
5. **推荐补充资料**：nanoGPT 的 `generate()`（最简推理循环参考）;HuggingFace `GenerationMixin` 的 `use_cache` 实现。

## Day 6：FlashDecoding（422 行）

1. **当前价值**："Q 切不了就切 KV"的破题叙事漂亮；跨 block 合并的数学证明（rescale 回全局 max）完整；FlashDecoding++ 的两个改进（预估 max、定长 chunk）覆盖到位；与 PagedAttention 的正交关系（存储层 vs 计算层）讲清了。
2. **是否必要**：**必要**，但注意它顶掉了本周的 profiling 日。
3. **难度等级**：中等~困难。
4. **建议修改**：
   - 修复标题重复；**"80 个 SM"出现 4 处，全部改为 170**(RTX 5090)——利用率 ~1.25%、实验 2 的"80 SM × 64 token = 5120"连带重算。
   - 修正引用："复用 Week 4 的三公式"应为 Week 5;"Day 5 把它们整合进 Mini 引擎时"语态修正（D5 已过）。
   - LeetGPU INT8 KV-Cache 与 D1 重复（映射表本日是 Weight Dequantization)——换题。
   - 正文 kernel 是骨架（省略号），完整版在落盘文件——正文加一句指向；任务 3 的 ncu 对比需要 standard decode 基线 kernel（实验 2 才创建），调整任务顺序或合并。
5. **推荐补充资料**：FlashDecoding 原始博客（PyTorch 官方 blog "Accelerating Generative AI with PyTorch: FlashDecoding");FlashDecoding++ 论文（2311.01282)。

## Day 7：推理系统核心问题总结（391 行）

1. **当前价值**：四大核心问题地图 + 现象→检查→解决速查表是本周最有沉淀价值的产物，误区澄清质量高；自测脚本（week5_summary.py）落盘可跑。
2. **是否必要**：**必要，但需全面改编号**——整篇是旧 Week5 复盘：标题自称 Week 5、目录树写 `aiinfra/daily/week5/`、衔接段写"Week 5→Week 6"、迁移批注残片（"FlashDecoding 已移至 Week 6 Day 2"）未清理。
3. **难度等级**：简单。
4. **建议修改**：
   - 全篇 Week 5→Week 6、Week 6→Week 7 替换（包括知识地图标题、衔接表、完成标准、脚本文件名 `week5_summary.py` 建议改为 `week6_summary.py`)。
   - **修复 Day6 错位**：知识地图表把 Day6 记为"端到端 Profiling / profile_engine_v0.py"——当前周 Day6 是 FlashDecoding，该 profiling 日和脚本都不存在。要么补一个 profiling 任务（对 mini_engine_v0 测 TTFT/TBT，呼应 D5 实验 2)，要么把表和速查表里的"Day6 决策树"引用改为 FlashDecoding。
   - 目录结构按实际重写（含 `day6/kernels/flash_decoding.cu`；删两行迁移批注）。
   - LeetGPU GPT-2 Transformer Block 与 D5 重复（映射表 W6D7 = Simple Inference)——换题。
   - 衔接表内容（Continuous Batching 深入、CUDA Graph、引擎 v1）与 Week7 实际内容吻合，保留但改编号。
5. **推荐补充资料**：DistServe 论文（PD 分离，W7D6 预习）；其余沿用现有资源表。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | Prefill vs Decode | 必要（需瘦身） | 简单~中 | 与 W4D1 重复 50%；脚本缩进损坏；引用错周 |
| 2 | KV Cache 实现 | 必要 | 中 | 范文级；ncu profile memcpy 方法错误 |
| 3 | vLLM 架构 | 必要 | 中 | 质量好；抢占实测留档；源码任务缺无安装替代 |
| 4 | PagedAttention | 必要 | 中~难 | 范文级；LeetGPU 重复 |
| 5 | Mini 引擎 v0 | 必要 | 中 | 脚本缩进损坏；缺 TBT 对比实测 |
| 6 | FlashDecoding | 必要 | 中~难 | **80 SM 错误 ×4**；顶掉了 profiling 日；LeetGPU 重复 |
| 7 | 周复盘 | 必要（需改编号） | 简单 | **全篇自称 Week 5；回顾的 profiling 日不存在**；目录树错 |

**本周最优先修复的三件事**：① D7 全篇改编号 + 修复 Day6 错位（补 profiling 任务或改引用）;② D6 的 80 SM→170 SM(4 处）;③ D1 与 W4D1 去重。

---

# Week 7 逐日评审（Batching 与调度）

> 评审范围：week7/day1–day7 README 全文精读，结合仓库文件存在性核对（落盘：day1 `continuous_batcher.py`、day2 `vllm_scheduler_analyzer.py`、day3 `chunked_prefill_simulator.py`、day4 `prefix_cache_engine.py`、day5 `mini_engine_v1.py`、day6 `pd_disaggregated_simulator.py`、day7 `week6_summary.py`)。
> 评审日期：2026-08-05

**整体判断**：本周的知识链（Continuous Batching → vLLM Scheduler 深潜 → 框架对比 → Chunked/Prefix 实操 → Mini 引擎 v1 → PD 分离）是推理系统部分的合理进阶，D2 的 Scheduler 复刻（含 RECOMPUTE/SWAP 双 demo 实测时间线）是本周高光。主要问题：① **旧编号残留为全课程之最**——几乎每天的学前导读/总结都自称错位的日号（D1 正文引用"Day 1 的 Dynamic Batcher"，而 Dynamic Batching 已被移到 `_supplementary`，对比基线悬空）;② **D7 再次整篇是旧周复盘**（自称 Week 6，回顾的 Day1 Dynamic、Day6 benchmark 在当前周不存在，当前 D6 的 PD 分离反而只字未提）;③ **LeetGPU 周内成对重复**(D1/D2 同为 Prefix Sum,D3/D4 同为 Segmented Prefix Sum);④ **全周零 GPU 代码**（纯 Python 模拟器），加上 W9，学员连续三周不碰 nvcc;⑤ D3 说 vLLM 默认 chunk_size=2048、D4 面试答案说 512，同周互相矛盾。

## Day 1：Continuous Batching（419 行）

1. **当前价值**：Dynamic vs Continuous 的对比表、iteration 时间线、混合调度挑战引入自然；ContinuousBatcher 带线程 + token budget，可跑。
2. **是否必要**：**必要**，但开篇基线悬空——它对比的"Day 1 的 Dynamic Batcher"不在本周（在 `_supplementary/from_w6d6`)，读者没有上下文。
3. **难度等级**：中等。
4. **建议修改**：
   - 修复内嵌 Python 缩进损坏（类定义层层嵌套）；落盘文件正常，保持同步。
   - 把 Dynamic Batching 的 10 行最小实现/对比表从 `_supplementary` 提回本日学前导读（否则"长请求阻塞"的对比无的放矢）。
   - 修正全部旧编号："Week 5 Day 3 我们读过"应为 Week 6 Day 3；"明天 Day 3 深入 vLLM Scheduler"应为 Day 2；今日总结自称"Day 2"。
   - LeetGPU Prefix Sum 与 D2 重复，本日换 Simple Inference（映射表 W7D1）或明确 D1/D2 二选一。
5. **推荐补充资料**：Orca 论文（OSDI 2022,iteration-level scheduling 原始出处）;AnyScale 的 Continuous Batching 博客。

## Day 2：vLLM Scheduler 源码分析（1016 行）

1. **当前价值**：**本周最佳，全课程的系统课范文**。schedule() 5 步流程、双预算、防饿死的 `if self.swapped: return`、RECOMPUTE vs SWAP 的量化对比（重算几 ms vs PCIe 几十 ms)；复刻实现 520 行带三个真实可跑 demo(S1 被抢占 g 归零 vs SWAP 保留进度的时间线），教学闭环完整；V1 演进附录克制准确。
2. **是否必要**：**必要**，但 1016 行是单日体量上限的 2 倍，建议拆。
3. **难度等级**：困难（体量大；概念本身因 D1/W6D3 铺垫而不算陡）。
4. **建议修改**：
   - 修正旧编号（学前导读"Day 2 的 Continuous Batcher"应为 Day 1；今日总结自称"Day 3";"明天 Day 4"应为 Day 3)。
   - 体量拆分：把"附录 V1 演进"和任务 3 的对比分析移入 `notes/`，正文收在 ~600 行；或在开头加"阅读节奏"指引（理论 3.1-3.4 必读，3.5 源码追踪可选）。
   - LeetGPU Prefix Sum 与 D1 重复（映射表 W7D2 = Stream Compaction，更合适）——换题。
   - 与 W6D3 的关系需在开头一句说清（W6D3 讲架构全景 + 简化版调度器，本日专讲 schedule() internals)，否则学员会疑惑两遍 vLLM 调度器。
5. **推荐补充资料**：vLLM 源码 `vllm/core/scheduler.py`(V0）与 V1 的 `vllm/v1/core/scheduler.py` 对照；vLLM V1 设计文档。

## Day 3：TRT-LLM / LightLLM / SGLang 调度对比（695 行）

1. **当前价值**："Inflight = Continuous"的术语解魅、四框架对比表、chunked prefill 模拟器实测（尖峰 2.0→1.2ms，降 40%）都是面试直接可用的素材；RadixAttention vs block-hash 的对齐损失分析到位。
2. **是否必要**：**必要**，但与 D4 有内容撞车（chunked prefill 概念、RadixAttention 对比表两边都讲）。
3. **难度等级**：中等。
4. **建议修改**：
   - 修复标题重复字串；修正旧编号（"Day 3 的 _schedule_waiting"应为 Day 2；"Day2/Day3"注释应为 Day1/Day2；今日总结自称"Day 4")。
   - **与 D4 划界**：本日保留"概念 + 框架对比 + 模拟器",RadixAttention 对比表移到 D4（本日只留一句预告）;chunked prefill 的调度伪代码去重。
   - **数字矛盾**：本日写"vLLM 默认 chunk_size=2048",D4 面试答案写"vLLM 默认 512"——核实后统一（vLLM V1 默认 2048)。
   - LeetCode 的 200/130 与 D4 重复，顺延去重。
5. **推荐补充资料**：TensorRT-LLM 的 Inflight Batching 官方文档；SGLang 论文（RadixAttention)。

## Day 4：Chunked Prefill 与 Prefix Caching 实操（372 行）

1. **当前价值**：prefix caching 的 block hash + LRU + 命中率分析表（多轮 80-95%、共享 system prompt 90-99%）是新增量；引擎模拟有实测加速比（多轮 4x);LeetCode 选了 LRU/LFU 对应淘汰策略，呼应好。
2. **是否必要**：**必要**(prefix caching 是本周唯一的新机制），但 chunked 部分与 D3 重复。
3. **难度等级**：中等。
4. **建议修改**：
   - 修正自指（"Day 4 学习了…概念"应为 Day 3;"Day 4 的 chunked prefill simulator 和 Day 2 的 continuous batcher"应为 Day 3/Day 1)。
   - 删与 D3 重复的 chunked prefill 概念节，保留"联合调度"和 prefix caching 主体；修复 chunk_size 默认值矛盾（见 D3)。
   - LeetGPU 与 D3 同为 Segmented Prefix Sum——本日换 Top K Selection（映射表 W7D5 题）或保留 D3 一题。
   - 任务 4 只给了题解链接没给题目链接（leetgpu.com)，补上。
5. **推荐补充资料**：vLLM prefix caching RFC;SGLang RadixAttention 论文（与 D3 共享）。

## Day 5：Mini 推理引擎 v1（多请求并发）(435 行）

1. **当前价值**：课程主项目线的第二个里程碑——Future 异步、锁粒度设计（锁内队列/锁外 forward)、优先级调度、实测时间线（batch 4→3→2→1）都讲得清楚；与 vLLM 组件的对照表保持了系统视角。
2. **是否必要**：**必要**，项目主线日。
3. **难度等级**：中等~困难（线程 + 调度 + 模型三者首次合体）。
4. **建议修改**：
   - 修复标题重复；修正跨周引用（"Week 5 Day 5 的 Mini 引擎 v0"应为 Week 6 Day 5;"Week 6 Day 1-4"应为 Week 7 Day 1-4)。
   - **"并发收益 2.9x"要加口径说明**：当前 `_run_iteration` 仍是逐请求单独 forward（实验 1 才做真 batch 合并），所以 2.9x 只是调度层（23 轮→8 轮）的模拟收益，不是 GPU 吞吐收益——建议在数字旁注明，并把实验 1（真 batch 合并 forward）升级为任务 3.5，这是本日离"真实收益"最近的一步。
   - "明天 Day 6 做 throughput-latency benchmark"——实际 D6 是 PD 分离，该 benchmark 内容在 `week10/_supplementary`，修正预告或补回。
   - LeetGPU INT8 MatMul 与"多请求调度"主题关联牵强（且量化是 Week8 主题），换 Top K Selection（映射表）。
5. **推荐补充资料**：vLLM 的 `AsyncLLMEngine`（异步接口原型）;Python `concurrent.futures.Future` 文档。

## Day 6：PD 分离推理（204 行）

1. **当前价值**：本周新概念密度最高的一天——资源错配/SLO 矛盾的破题、KV 传输量口算（32K prompt=16GB=160ms RDMA)、Mooncake/DistServe/vLLM V1 三系统对比、与 chunked prefill 的关系表，全部面试高频；模拟器有"教学模型"的诚实声明。
2. **是否必要**：**必要**（本日内容是 2024-2026 面试必考，也是 W9 分布式的引子），体量反而偏轻。
3. **难度等级**：中等。
4. **建议修改**：
   - 修复标题重复；2.1 节的架构图是 ASCII 框图，违反"禁止 ASCII 图"规范，重绘 SVG。
   - 模拟器输出标"RTX 5090 模拟参数"——但数值是示意模型产出，改为"教学模拟，参数示意"避免误读为实测。
   - 修正引用："回顾 Day 1 与 Week3/Day1 的 Roofline 分析"——应为 Week 6 Day 1（与 Week1 Day3)。
   - 本日无 LeetGPU/LeetCode 任务（五天结构里缺任务 4/5)，按模板补齐（映射表 W7D6 = Dot Product;LeetCode 用第 6 周回溯进阶 5 题——正好补上 D7 回顾表里列了但没人布置的那组）。
   - 可适当扩容：加一个"KV 传输用 NCCL 还是 RDMA"的对比小节，为 W9 NCCL 周埋钩子。
5. **推荐补充资料**：Mooncake 论文（FAST '25);DistServe 论文（OSDI '24);vLLM disaggregated serving RFC。

## Day 7：调度优化策略总结（357 行）

1. **当前价值**:7 策略对比表 + 决策树 + 6 个误区澄清（"RECOMPUTE 默认非因快""饱和点非仅 util=100%"）都是高质量的面试资产；自测脚本落盘。
2. **是否必要**：**必要，但需按当前周重写**——整篇是旧 Week6 复盘：知识地图 Day1=Dynamic Batching、Day6=Benchmark（两者都不在当前周，产出物 `dynamic_batcher.py`/`benchmark_engine_v1.py` 分别在两个 `_supplementary` 目录）；本周实际的 D4 prefix caching、D6 PD 分离在复盘中完全缺席；目录树写 `week6/`、衔接写"Week 6→Week 7"、结尾还在说"8 周学习的收官"。
3. **难度等级**：简单。
4. **建议修改**：
   - 知识地图按当前周重写：D1 Continuous → D2 vLLM Scheduler → D3 框架对比 → D4 Chunked+Prefix → D5 引擎 v1 → D6 PD 分离；策略对比表加 Prefix Caching 和 PD 分离两行；误区澄清加一条"PD 分离不是更快，是 SLO 解耦"。
   - 目录结构改 `week7/` 并按实际文件重写；删迁移批注残片；脚本改名 `week7_summary.py`；衔接段改为"Week 7→Week 8（量化/投机解码/CUDA Graph)"——当前衔接表里"Week 7 系统整合/8 周收官"全是旧框架。
   - LeetCode 回顾表与实际布置对不上（表列 Day1=路径问题 112/113/129/222/437，实际 D1 布置的是 LCA 组；Day6 回溯进阶无人布置）——按实际重列，并把回溯进阶组落到 D6（见上）。
   - LeetGPU Reduction 与 W1D5/W2D7 重复（映射表 W7D7 = Matrix Addition)，换题或标注复用。
5. **推荐补充资料**：沿用现有推荐资源表（质量好，Orca 论文已含）。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | Continuous Batching | 必要 | 中 | Dynamic 基线悬空（在 _supplementary)；旧编号 |
| 2 | vLLM Scheduler | 必要 | 难 | **本周最佳**;1016 行过载；旧编号 |
| 3 | 框架对比 | 必要 | 中 | 与 D4 撞车（chunked/RadixAttention);chunk 默认值矛盾 |
| 4 | Chunked+Prefix | 必要 | 中 | LeetGPU 与 D3 重复；自指错位 |
| 5 | Mini 引擎 v1 | 必要 | 中~难 | "2.9x"只是调度层模拟；benchmark 预告落空 |
| 6 | PD 分离 | 必要 | 中 | ASCII 架构图违规；缺 LeetGPU/LeetCode 任务 |
| 7 | 周复盘 | 必要（需重写） | 简单 | **旧 Week6 复盘；本周 D4/D6 内容缺席；目录树错** |

**本周最优先修复的三件事**：① D7 按当前周实际内容重写（补 Prefix Caching/PD 分离，删 Dynamic/Benchmark 幽灵）;② 全周旧编号统一（学前导读/总结的自指日号 +1 错位）;③ LeetGPU 成对重复去重（D1/D2、D3/D4)。

---

# Week 8 逐日评审（推理加速技术）

> 评审范围：week8/day1–day7 README 全文精读，结合仓库文件存在性核对（落盘：day1 三个 CUDA kernel、day3 `advanced_features.py`、day4 `cuda_graph_capture.py` + `shape_bucketing.py`)。
> 评审日期：2026-08-05

**整体判断**：本周两头硬、中间空。D1（量化三层武器 + 三个真实 kernel + FP8 软件模拟"诚实 FAIL"声明）和 D4(CUDA Graph,1.89x 实测 + bucketing 正确性 PASS）是全课程后半段最好的两天。但：① **D2 与 D1 大面积重复**(E4M3/E5M2、GPTQ/AWQ/SmoothQuant、FP4 全部讲两遍，且 FP4 格式描述两天互相矛盾）;② **D3 的 chunked prefill/prefix caching 是第三次讲**(W7D3/D4 已教），真正的增量只有投机解码（这部分反而写得最好——精确期望公式 + Medusa/EAGLE/MTP 对比）;③ **D5 项目日全是伪代码**，加速技术"接入 Mini 引擎"没有真实发生——整体路线评审里"主项目贯穿性弱"的判断在此坐实；④ D7 知识地图又是 ASCII 框图，且"采样 kernel"出现在地图和 Q&A 里却本周没有任何一天教过。

## Day 1：量化推理专题 —— W8A16/INT8 KV/FP8（653 行）

1. **当前价值**：本周最佳候选——对称/非对称、三种 scale 粒度、"scale 提到点积外"的数学推导、AWQ vs GPTQ vs SmoothQuant 三方对比、KV 量化误差累积分档、FP4/MXFP8 前沿，全部有据；三个 kernel 落盘且实测留档（W8A16 快 1.5x);FP8 kernel 正确性 FAIL 的诚实声明（软件模拟、比 FP32 还慢）是教学诚信的典范。
2. **是否必要**：**必要**，本周地基。
3. **难度等级**：困难（理论密度 + 三个 kernel，一天偏满）。
4. **建议修改**：
   - 修复标题重复；修正悬空引用（"Day 6 的 profiling/瓶颈决策树"指向已被移走的 benchmark 日，"B1 任务（WMMA 做实）"是旧标签）。
   - FP8 软件模拟 FAIL 之后给一条出路：补一个"真 FP8"的最小示例（`__nv_fp8_e4m3` + torch._scaled_mm）或明确指向 Week8 D2 应该承担这个角色（见 D2 评审）。
   - "7B 权重 14 GB、每步读 14 GB"的口径与 reference/key_numbers.md 对齐检查（数字本身自洽，建议链接事实源）。
   - LeetCode 用"第 5 周回顾"——按 8 周计划第 8 周（DP 进阶与图论）重新对齐。
5. **推荐补充资料**：AWQ 论文（2306.00978)、GPTQ 论文（2210.17323)、SmoothQuant 论文；DeepSeek-V3 技术报告的 FP8 章节。

## Day 2：FP8 量化深入（340 行）

1. **当前价值**:E4M3/E5M2 位级细节表（偏置、最小正数）比 D1 略细；GPTQ/AWQ 的 PyTorch 伪代码模拟方向对。
2. **是否必要**：**当前形态不必要**——约 70% 内容（FP8 格式表、FP8 vs INT8、GPTQ/AWQ/SmoothQuant 对比、FP4）与 D1 重复，且无 kernel 落盘、无 LeetGPU、无可运行代码。建议重做。
3. **难度等级**：中等。
4. **建议修改**：
   - **重做为"真 FP8 实操日"**：承接 D1 的 FAIL 声明——用 `torch._scaled_mm`(Hopper/Blackwell FP8 GEMM）或 `__nv_fp8_e4m3` 写一个真实 FP8 GEMM，实测 vs FP16 的 2x 算力；这样 D1（软件模拟）→D2（真硬件）形成完整故事线，也填补了全课程"FP8 只在嘴上说"的空缺。
   - 若保留现有结构：删掉与 D1 重复的格式表和三方对比（只留链接）;FP4 规格矛盾必须修正（D1 写"3 指数+隐含尾数 ±12"、本日写"E2M1 ±4"——NVFP4 实为 E2M1 + microscaling，统一）；算力倍数核对（FP4 = FP16 的 4x ≈ 836 TFLOPS 与 D1 的"FP8 的 2x"自洽，但需一处口径）。
   - GPTQ/AWQ 的 PyTorch 模拟落盘为可运行脚本（当前是伪代码，且 `weight[:, col+1:] -= ...` 的 in-place 写法会破坏 Hessian 的前提，需注释说明简化）。
   - LeetCode 的 240 已是第四次出现；补 LeetGPU（映射表"待定"，建议 Weight Dequantization 的 FP8 变体）。
5. **推荐补充资料**：torchao 仓库（PyTorch 官方量化库，FP8 支持）;NVIDIA Transformer Engine 的 FP8 教程。

## Day 3：SGLang / 投机解码（528 行）

1. **当前价值**：投机解码部分是全课程独一份的好内容——"kα+1 是近似上界、精确期望是等比级数 (1-α^(k+1))/(1-α)"的纠偏、α 扫描表、Medusa/EAGLE/MTP 三路线对比（含 DeepSeek MTP)、α=0.5+k=8 变慢的实测模拟。
2. **是否必要**：**必要，但要砍掉一半**——3.2 Chunked Prefill、3.3 Prefix Caching 是 W7D3/D4 的第三次重复；标题承诺的"SGLang"正文完全没有（SGLang 在 W7D3 讲过）。
3. **难度等级**：中等。
4. **建议修改**：
   - 删掉 3.2/3.3，各留三行速查 + 链接 W7D3/D4；标题改为"投机解码：原理与三路线（Medusa/EAGLE/MTP)"。
   - 修正"Day 2 的 FullScheduler"（无此物）为"Week 7 的调度器";chunk_size"推荐 512(vLLM 默认）"与 W7D3 的 2048 矛盾，统一。
   - 把 C3 补充的三路线内容并入主体（目前在文末像补丁）；面试题 6-8 并入面试要点。
   - LeetGPU Scalar Multiply 与投机解码的关联过于牵强，换 Speculative Decoding Verification(W6D3 用过）或按映射表标注复用。
5. **推荐补充资料**：Medusa 论文（2401.10774)、EAGLE 论文（2401.15077)、DeepSeek-V3 的 MTP 章节；speculative decoding 原始论文（2302.01318)。

## Day 4：CUDA Graph 实操（534 行）

1. **当前价值**：本周另一个高峰——launch overhead 的 41% 占比测算、capture/replay 三步法、三个必踩的坑（warmup/静态 buffer/新建 tensor)、bucketing 三种方案与回退策略；两个脚本落盘 + 实测（1.89x、max_diff=0);"vLLM decode 用 graph、prefill 用 eager"的生产经验点题准。
2. **是否必要**：**必要**。
3. **难度等级**：中等（概念直观，API 模式固定）。
4. **建议修改**：
   - 修复标题重复；修正引用（"Day 6 的全链路 Profiling"悬空——该日在 `_supplementary`；"明天 Day 7 代码重构...Week 7 系统整合收官"应为 Week 8 的 Day5 项目日）。
   - 与 W2D5(Streams）加一句衔接（graph 是 stream 语义的延伸），并把"实验 3 的 cudaGraphExecUpdate"标注为进阶（PyTorch 未暴露）。
   - LeetGPU Vector Addition 与主题关联靠"launch-bound"类比，可用；但映射表 W8D4 = Matrix Transpose，二选一同步。
   - LeetCode"第 7 周补充"的标注方式与全周不统一（D3 用第 7 周 Day 3)，统一编排口径。
5. **推荐补充资料**：CUDA C++ Programming Guide 的 CUDA Graphs 章节；PyTorch blog《Accelerating PyTorch with CUDA Graphs》。

## Day 5：项目推进 —— 加速技术接入 Mini 引擎（283 行）

1. **当前价值**：选型表（复杂度 vs 收益）和"先接 CUDA Graph"的优先级判断正确。
2. **是否必要**：**必要，但当前交付不合格**——三个集成方案全是伪代码片段，没有可运行文件、没有实测数字；任务 2 的 benchmark 函数是 `pass`。这是主项目线的又一次空心推进（同 W5D5)。
3. **难度等级**：中等。
4. **建议修改**：
   - **落盘真实集成**：把 D4 的 `BucketedGraphRunner` 套到 W7D5 的 `mini_engine_v1.py` decode 路径上，产出 `mini_engine_v1_graph.py`，实测 4 请求并发的端到端 TBT 改善（哪怕 1.2x 也是真数字）——这是本周最该有的一次"真整合"。
   - 与 D4 去重（bucketing 伪代码、静态 buffer 讲解都是 D4 的）；本日聚焦"工程集成 + 实测对比"。
   - 目标只 4 条、面试要点只 2 题，补到模板规格；补 LeetGPU（映射表"待定")。
5. **推荐补充资料**：vLLM 的 `cudagraph` 捕获实现（`vllm/worker/model_runner.py` 的 capture 逻辑）。

## Day 6：Profiling —— 量化/CUDA Graph 验证（245 行）

1. **当前价值**:launch gap 实测（146.0→67.7μs,overhead 53.7%,speedup 2.16x）是本日真实亮点；ROI 表把本周技术排了序。
2. **是否必要**：**必要**，但体量偏轻、一半任务不可执行。
3. **难度等级**：中等。
4. **建议修改**：
   - 落盘 `bench_eager.py`/`bench_graph.py`（当前引用但不存在）；注明实测配置已写但脚本缺失。
   - 6.2/6.3 的量化精度/性能表（W8A16 max_diff ~0.1、throughput 350 tok/s 等）无来源标注——标"预期值/文献口径"或用 D1 的 kernel 实测替换部分行。
   - 与 D5 联动：D6 的量化对比任务应测的是"D5 集成后的引擎"，当前两者脱节（D5 没集成成功，D6 就没有东西可 profile)——先修 D5。
   - 补 LeetGPU（映射表 W8D6 = Reduction）和 LeetCode 主题标注（堆主题与第 8 周 DP 不符）。
5. **推荐补充资料**：沿用 Nsight 文档；加一篇 MLPerf Inference 的量化精度报告作参考口径。

## Day 7：复盘与面试 Q&A（207 行）

1. **当前价值**:ROI 总表 + 10 道 Q&A 收敛得紧凑；接受率公式、FP4 挑战等复用本周最好的素材。
2. **是否必要**：**必要**，本周第一个"复盘内容与本周实际一致"的 Day7（难得）。
3. **难度等级**：简单。
4. **建议修改**：
   - 知识地图是 ASCII 框图——重绘 SVG（连续第三个违规的 D7，建议全课程统一补图）。
   - **采样（top-p/top-k/temperature）从未在本周任何一天教过**，却出现在知识地图、ROI 表、Q7、Q10——要么在 D5/D6 补一个采样 kernel 小节（LeetGPU Top-P 现成），要么从复盘里删掉。
   - "下周预告"写"Week 10"——跳过了 Week 9（分布式），修正。
   - Q10 的"Mini 引擎 decode 5ms→3ms、throughput 200→300"无实测支撑（D5 未集成）——标"示意"或等 D5 修复后回填。
   - 补 LeetGPU/LeetCode 本周回顾表（模板要求）；目标只 3 条，补到规格。
5. **推荐补充资料**：本周 D1/D3/D4 自身即可；加 NVIDIA Blackwell 白皮书（FP8/FP4 章节）。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | 量化专题 | 必要 | 难 | 本周最佳；悬空引用（B1/Day6);FP8 FAIL 缺下文 |
| 2 | FP8 深入 | **当前形态不必要** | 中 | 与 D1 重复 70%；无落盘代码；FP4 规格与 D1 矛盾 |
| 3 | 投机解码 | 必要（需砍半） | 中 | 投机解码写得好；chunked/prefix 第三次重复；标题 SGLang 无内容 |
| 4 | CUDA Graph | 必要 | 中 | 范文级；悬空"Day 6 profiling"引用 |
| 5 | 接入引擎 | 必要（需重做） | 中 | **全伪代码，集成没有真实发生** |
| 6 | Profiling | 必要 | 中 | 有实测亮点；引用的 bench 脚本不存在；量化表无来源 |
| 7 | 复盘 | 必要 | 简单 | ASCII 图违规；采样内容无源；预告跳过 Week 9 |

**本周最优先修复的三件事**：① D5 真实落地（Graph 接入 mini_engine_v1 + 实测）;② D2 重做为"真 FP8 GEMM 日"（呼应 D1 的 FAIL 声明）;③ D3 删除 chunked/prefix 重复内容、改名聚焦投机解码。

---

# Week 9 逐日评审（分布式并行与多硬件）

> 评审范围：week9/README + day1–day7 README 全文精读，结合仓库文件存在性核对（落盘：day1 `tp_inference_demo.py` + `comm_overlap_demo.py`、day5 `moe_routing_simulator.py`、day6 `cuda_vs_ascend_comparison.py`、`_supplementary/from_w8d6/ring_attention_sim.py`；day2/3/4 无 kernel 落盘）。
> 评审日期：2026-08-05

**整体判断**：本周知识选题全部命中面试高频点（TP/PP/DP/EP 四维并行、NCCL 通信量、bubble ratio、通信计算重叠、MoE/EP、CUDA vs Ascend），D5（MoE+EP）是质量最高的一天。但结构病明显：① **D1 一天讲完了 D2/D3/D4 的全部主题**（TP+PP+DP+NCCL+overlap 五大块塞进 2.5h 工作日），导致后三天 60%+ 内容是对 D1 的重复深化；② **D2 的 interleaved 1F1B 公式/表格自相矛盾**（知识性错误）；③ **D4 双流苏 overlap 示例的 `wait_stream` 写反**，恰恰破坏了它要演示的重叠；④ day2/3/4 的模拟器代码只有骨架（含 `pass`/`...`），未落盘；⑤ day1/day5/day6 标题重复粘贴（`CANNNVIDIA...`）；⑥ 周学习地图承诺 Day 5 含 Ring Attention，正文只字未提（材料在 `_supplementary` 但无链接）。此外 D6 整周不碰 Mini 引擎的判断在整体评审中已坐实，本周（W9）同样完全脱离项目主线。

## Day 1：分布式推理 —— TP/PP/DP 与通信计算重叠（573 行）

1. **当前价值**：TP 的 column/row-parallel 切分讲得最好，"一个 Attention Block 仅 1 次 all-reduce"是核心记忆点；`tp_inference_demo.py` 单卡模拟 TP 可运行、有正确性验证（max diff 2.5e-7），教学闭环完整。
2. **是否必要**：**必要，但严重超载**——TP+PP+DP+NCCL+通信重叠五个主题 + 2 个 demo + LeetGPU + 4 道 LeetCode，2.5h 装不下，实际需 6–8h。
3. **难度等级**：困难（因信息量而非单点难度）。
4. **建议修改**：
   - 把 §3b.3（PP）、§3b.5（NCCL）、§3b.6（通信重叠）整体移至 Day 2/3/4，D1 只保留"为什么需要分布式 + TP + DP 定位"——这是修复全周冗余的关键一步。
   - 修正标题重复（`...通信计算重叠分布式推理专题 —— TP/PP/DP...`）。
   - `comm_overlap_demo.py` 依赖 CUDA 环境，与 tp demo（CPU 可跑）门槛不同，需显式标注。
   - LeetCode 标注"8 周计划第 7 周补充"，与本周（第 9 周）编排口径混乱，统一或说明。
5. **推荐补充资料**：Megatron-LM 论文（1909.08053，TP 原始出处）；Google《How to Scale Your Model》（jax-ml.github.io/scaling-book，TP/通信量推导最清晰的现代资料）。

## Day 2：Pipeline Parallelism 与 DP（341 行）

1. **当前价值**：GPipe vs 1F1B 的显存对比（O(M) vs O(P)）和 bubble ratio 计算是面试高频点；"推理只有 forward、PP 主要为省显存"的定位准确。
2. **是否必要**：**必要**——但前提是 D1 减负；当前状态下与 D1 §3b.3/§3b.4 重复约 60%。
3. **难度等级**：中等（公式推导为主）。
4. **建议修改**：
   - **修正 interleaved 1F1B 公式错误**：`(P-1)/(V·M+P-1)×V` 与表格数值自相矛盾（`7/39×2` 写成 18%，实为 36%），且 M→∞ 时退化为 `(P-1)/M`，与标准 1F1B 相同，失去降 bubble 的意义；应为 `(P-1)/(V·M+P-1)`。
   - 任务 1 的 `simulate_gpipe` 骨架有逻辑问题（`timeline[s][-1][1]` 取的是 tuple 第 2 个元素即 micro-batch 编号而非结束时间，应为 `[-1][3]`），且 `simulate_1f1b` 只有 `pass`；补完并落盘 `kernels/pipeline_schedule_sim.py`。
   - 补一段"推理 PP 实际部署形态"（vLLM `pipeline_parallel_size` 行为），否则全以训练视角（backward/1F1B）展开，与前 8 周的推理主线脱节。
5. **推荐补充资料**：Megatron-LM v3（2104.04473，interleaved 1F1B 出处）；GPipe（1811.06965）；PipeDream（1806.03377，进阶选读）。

## Day 3：NCCL Collectives —— 通信量推导（295 行）

1. **当前价值**：三大 collectives 通信量公式表 + ring 两阶段推导是本周最核心的硬知识；N=4/4GB 算例直观。
2. **是否必要**：**必要**，但与 D1 §3b.5 高度重复（公式、ring/tree 对比表几乎一样）；单独一天只有公式表 + 两个小练习，撑不满 2.5h。
3. **难度等级**：简单（纯公式推导，无新概念）。
4. **建议修改**：
   - 增加 D1 没有的内容：NVLink/PCIe/IB 带宽实测常识（NVLink4 ≈ 900GB/s、CX-7 IB 400Gb/s）、α-β 通信模型（latency + size/bandwidth），让"通信量→通信时间"能落地估算。
   - 任务 2 的 `ring_all_reduce` 模拟只有骨架（`...`），补完并落盘。
   - 有 GPU 环境时加 `nccl-tests`（all_reduce_perf）实测练习。
5. **推荐补充资料**：NCCL 官方文档（docs.nvidia.com/deeplearning/nccl）；nvidia/nccl-tests 仓库。

## Day 4：通信计算重叠 —— 双 Stream + CUDA Graph Overlap（292 行）

1. **当前价值**：层切分重叠（GEMM 分两半、前半 all-reduce 与后半计算重叠）是比 D1"无依赖双流 demo"更接近真实 TP 的进阶内容；收益边界 `min(T_comp, T_comm)` 的分析有工程价值。
2. **是否必要**：**必要，但需与 D1 去重**——D1 扩展实验 3（上一层 all-reduce 与本层 GEMM 重叠）与本日主题撞车。
3. **难度等级**：困难（双流同步关系易错，需 GPU 验证）。
4. **建议修改**：
   - **修正 4.2 示例代码的同步错误**：`compute_stream.wait_stream(comm_stream)` 让 y2 计算等待 y1 的 all-reduce 完成，恰恰破坏了本日要演示的重叠——注释里自问自答却没删这行。删掉并解释原因（y2 不依赖 y1），这是现成的最佳教学点。
   - 与 D1 分工：D1 保留"双流概念 + demo"，层切分重叠独占放 D4，删 D1 扩展实验 3（改为"见 Day 4"）。
   - 任务代码依赖 `torch.distributed` 多卡初始化但无任何 init 说明，单卡跑不起来；提供 gloo 单进程 mock 或明确标注需 2 GPU + torchrun。
   - **补 sequence parallelism**（Megatron，2205.05198）：把 TP 的 all-reduce 拆成 reduce-scatter/all-gather 以便重叠——工业界主流做法，目前全周缺失。
5. **推荐补充资料**：Sequence Parallelism 论文（2205.05198）；vLLM/TRT-LLM 的 overlap 实现。

## Day 5：MoE + EP 并行专题（265 行）

1. **当前价值**：**本周质量最高的一天**——EP all-to-all 通信量推导、"decode 选 EP / prefill 选 TP"结论表、aux-loss-free bias、DeepEP/EPLB，全部是 2024+ 面试热点；`moe_routing_simulator.py` 落盘可跑。
2. **是否必要**：**必要且及时**（DeepSeek 系模型让 EP 成为必考）。
3. **难度等级**：中等。
4. **建议修改**：
   - 修正标题重复（`MoE + EP 并行专题MoE + EP 并行专题（...）`）。
   - **把 Ring Attention 真正纳入**：周地图承诺了 Day 5 含 Ring Attention，正文只字未提；加导读 + 链接 `_supplementary/from_w8d6/`，或并入正文。
   - 缺 LeetCode/LeetGPU 任务（其他天都有），若有意减负应在 README 说明，保持格式一致。
   - 第 4 节 TP 通信量写成 `2 × hidden × EP` 易误导（维度不含 tokens/batch），统一写成 `2 × tokens × hidden` 便于与 EP 对比。
5. **推荐补充资料**：DeepSeek-V3 技术报告（2412.19437，MoE 架构 + EP 部署细节）；deepseek-ai/DeepEP、deepseek-ai/EPLB；Switch Transformer（2101.03961）/GShard（2006.16668）。

## Day 6：多硬件对比 —— NVIDIA CUDA vs Ascend CANN（476 行）

1. **当前价值**："概念映射五对"（smem↔UB、Tensor Core↔Cube、warp 切换↔三单元流水）是应对"多硬件适配"面试题的实用框架；14 维速查表 + 可打印脚本设计合理；开头"概念对照-only"的定位声明诚实。
2. **是否必要**：**中等必要**——纯对照表没有一行 Ascend 代码，学习者只是背映射；目标昇腾岗位则必要，否则可压缩为半天附录。
3. **难度等级**：简单（纯概念记忆）。
4. **建议修改**：
   - 修正标题重复（`Ascend CANNNVIDIA CUDA vs Ascend CANN`）。
   - README 表格与 `cuda_vs_ascend_comparison.py` 内容 100% 重复，维护双份易腐坏；README 只留核心 5 对映射 + 链接，详细表只放脚本。
   - LeetCode 四题的"与今日主题的类比"列（编辑距离=迁移最小代价、克隆图=跨硬件复刻）牵强，稀释主题，删类比列或整段移除。
   - 补一个只读性 Ascend C 代码片段（GEMM 的 Copy→Compute→Copy 骨架，不要求运行），让映射从表格落到代码形态。
5. **推荐补充资料**：昇腾社区 Ascend C 算子开发指南（hiascend.com）；torch_npu 仓库；CANN msprof 使用手册。

## Day 7：复盘与面试 Q&A（239 行）

1. **当前价值**：知识地图 + 并行策略决策表 + 10 道 Q&A 结构清晰；"70B 怎么部署 8 卡"这类综合题正好串起全周；Checklist 可自测。
2. **是否必要**：**必要**，周末收尾合理。
3. **难度等级**：简单（以回顾为主）。
4. **建议修改**：
   - Q&A 与 D1–6 各自"面试要点"重复（ring all-reduce 通信量在 D1/D3/D7 出现三遍）；D7 只保留综合型题目（Q1/Q2/Q10），单点知识题改为链接指回对应天。
   - 决策表偏理想化：Q1 答案只算权重 17.5GB/卡，未提 KV Cache——与 D1 自己强调的"KV Cache 容易被忽略"矛盾，补充。
   - 加动手收尾任务：给 2–3 个场景（如"34B、2×A100、TTFT<500ms"），写并行方案 + 通信量估算 + 预计占比，代替纯勾选 checklist。
5. **推荐补充资料**：vLLM 文档 Parallelism and Scaling 章节；SGLang 多节点部署文档。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | TP/PP/DP + 通信重叠 | 必要（严重超载） | 难 | 一天讲完 D2/3/4 全部主题，全周冗余的源头 |
| 2 | PP 与 DP | 必要（依赖 D1 减负） | 中 | **interleaved bubble 公式错误**；模拟器代码未落盘且有 bug |
| 3 | NCCL 通信量 | 必要 | 简单 | 与 D1 重复；缺带宽常识与 α-β 模型，撑不满一天 |
| 4 | 通信计算重叠 | 必要（需去重） | 难 | **示例 wait_stream 写反破坏重叠**；缺 sequence parallelism |
| 5 | MoE + EP | 必要（本周最佳） | 中 | Ring Attention 只在补充材料无链接；TP 通信量写法误导 |
| 6 | CUDA vs Ascend | 中等必要 | 简单 | 无一行 Ascend 代码；README 与脚本双份维护；LeetCode 类比牵强 |
| 7 | 复盘 Q&A | 必要 | 简单 | 与前六天面试要点重复；Q1 漏 KV Cache；缺动手收尾任务 |

**本周最优先修复的三件事**：① D2 interleaved bubble 公式 + D4 wait_stream 两处知识性错误；② D1 减负、PP/NCCL/overlap 归位 D2/3/4，消除跨天重复；③ D2/D3 模拟器代码补完落盘，D5 补 Ring Attention 链接。

---

# Week 10 逐日评审（项目整合与面试冲刺）

> 评审范围：week10/README + day1–day7 README 全文精读 + `_supplementary/` 十个补充目录归属核对，结合仓库文件存在性核对（落盘：day1 `custom_ops_module.py`、day2 `mini_engine_v2.py` + `stability_test.py`、day3 `benchmark_demo.py`、day4 `interview_basics.py`、day5 `mock_interview.py`、day7 `week8_summary.py`；day6 无落盘）。另抽查 `mini_engine_v2.py` 实际代码。
> 评审日期：2026-08-05

**整体判断**：本周是旧"8 周版 Week 8"的整体迁移，但**迁移只做了一半**——D3/4/5/7 正文仍自称"Week 8 的第一天""8 周能力地图""8 周完成标准"，D7 目录结构写 `aiinfra/week8/`、脚本名 `week8_summary.py`，周编号体系全面过时；迁移造成的 Day 编号漂移未同步（D1 总结自称"Day 4 我们……"、D2 自称"Day 5"、D4 自称"Day 3"且预告"明天 Day 4 进入进阶篇"——进阶篇实际已移入 `_supplementary/from_w10d3/`）；day2/3/6 标题重复粘贴。内容本身：D1（kernel 集成）、D2（六步联调）、D6（诊断剧本三案例）是全课程最有工程味的三天；D3 的实测留档（RTX 5090 SiLU 60% 带宽）是数字诚信标杆；D4 的 GEMM 八层"理论+实测"双口径表有独特面试价值。但：① **`mini_engine_v2.py:27` 从 `day4/kernels` 导入 `custom_ops_module`（实际在 `day1/kernels`），ImportError 被静默 catch → 引擎静默回退 PyTorch 原生算子，"真整合"名存实亡**；② D7 能力地图没有 Week 8/9 内容（量化、分布式列为"待提升"，而这正是刚教完的）；③ 周地图承诺的架构图/进阶篇/查漏补缺都在 `_supplementary` 而正文无链接；④ D6 三份 ncu/py-spy"证据留档"是编造输出，与 D3 的实测诚信标准冲突。

## Day 1：整合全部自定义 Kernel（433 行）

1. **当前价值**：本周最实的一天——`load_inline` 编译流水线、C++ Wrapper 三板斧（`empty_like`/`data_ptr`/`size`）、六大注意事项（stream 一致性/布局/边界/TORCH_CHECK）是面试必考"怎么把 CUDA kernel 接进 PyTorch"的标准答案；`custom_ops_module.py` 落盘，"教学版可能比 PyTorch 慢（0.81x）"的诚实预期姿态正确。
2. **是否必要**：**必要**——把 Week 2–5 的 kernel 资产变现为"项目"的关键一步。
3. **难度等级**：中等（模式固定但细节多）。
4. **建议修改**：
   - 修正过时引用：学前导读"Day 3 分析了高级特性"、LeetGPU 段"Week 7 自定义 kernel 集成中……"、今日总结自称"Day 4 我们……"。
   - 周地图承诺"真整合，替换 sleep 模拟"，但本日只建了独立 `TransformerLayer`，未接进 Mini 引擎——地图措辞改为"完成 kernel 封装，Day 2 接入引擎"。
   - 补一份无 CUDA（CPU fallback）模式的预期输出，与现有"有 CUDA 环境"版并列。
5. **推荐补充资料**：PyTorch 官方 Custom C++/CUDA Extensions 教程；vLLM 的 `csrc/` + `setup.py` 组织方式（生产对照）。

## Day 2：系统联调（六步分层验证）（399 行）

1. **当前价值**：六步分层验证（单请求→并发→KV 隔离→Scheduler→Kernel→稳定性）与"组件正确 ≠ 系统正确"的五类跨组件 bug（串台/泄漏/死锁/超时传播/竞态）是全课程最有工程味的内容之一；`stability_test.py` 落盘可跑；`mini_engine_v2.py` 是 v1 模拟器到"真 timing + batched forward"的真实升级。
2. **是否必要**：**必要**——主项目线的收官动作。
3. **难度等级**：困难（并发正确性本身就难）。
4. **建议修改**：
   - **修复 `mini_engine_v2.py:27` 导入路径**（`day4/kernels` → `day1/kernels`）——当前静默 fallback，自定义 kernel 根本没被调用；catch 时至少打 warning。
   - `mini_engine_v2.py` docstring 引用"week9/day2 的 v1""week9/day1"——周号过时（v1 实际在 week7/day5）。
   - 修正标题重复；六步表格中"+ Queue（Day1）""+ Scheduler（Day2）""+ Custom Kernel（Day4）"对应旧 Week7 日序，改为当前实际来源（week7/day5、week10/day1）。
   - `stability_test.py` 仍是纯模拟引擎（"仅标准库，无需 GPU"），而真正的 v2 引擎在同目录却没有对应联调套件——稳定性测试应直接在 `mini_engine_v2` 上跑，否则"500+ 请求稳定"的成果证据仍是模拟的。
5. **推荐补充资料**：py-spy（hang 排查，与 D6 案例 3 呼应）；`torch.cuda.memory_snapshot()` 官方文档（OOM 排查）。

## Day 3：项目文档完善（README）（453 行）

1. **当前价值**：README 六段结构"每段对应一个面试问题"的框架、benchmark 六坑（warmup/计时含 memcpy/忘 sync/不算基线比例）是高价值软技能；`benchmark_demo.py` 有真实留档数据（RTX 5090 SiLU 60% 带宽），"实测留档 + 声明环境"是全课程数字诚信的标杆。
2. **是否必要**：**必要**——求职包装课，排在 D1/D2 之后合理。
3. **难度等级**：简单。
4. **建议修改**：
   - 全文"Week 8 的第一天""Week 1-7 我们积累了""Week 7 结束时"等旧周号改成 10 周口径；修正标题重复。
   - 地图承诺的"架构图 + 数据流图"不在正文（在 `_supplementary/from_w9d7/`），链接或并回。
   - 任务 3 表格里 GEMM/FlashAttention 两行数字（1.8ms/72%、4.0ms/2.1x）无留档来源，与 SiLU 行的实测标准不一致——标"示意"或回填真实测量。
5. **推荐补充资料**：vLLM/SGLang 的 README 作对标样本；`torch.utils.benchmark`（比手写 Event 循环更稳）。

## Day 4：高频面试题基础篇（566 行）

1. **当前价值**：12 题自测系统（`interview_basics.py` 落盘、可交互抽题）形式好；GEMM 八层"理论阶梯 + RTX 5090 实测占比"双口径是本日最大亮点——实测列的反直觉数据（float4 是最大单步收益 30.8%→64.3%、合并写回收益在噪声内、Double Buffer 同步实现未重叠）比任何理论表都有面试价值。
2. **是否必要**：**必要**——与 Week 1–2 是复习关系，定位清晰。
3. **难度等级**：简单（纯复习）。
4. **建议修改**：
   - 今日总结自称"Day 3 我们……"、预告"明天 Day 4 进入进阶篇"——进阶篇实际在 `_supplementary/from_w10d3/`，改链接；"零钱兑换"一句是旧文残留碎片（本日 LeetCode 是二维 DP），删。
   - 12 题覆盖面偏窄：Week 8（量化/投机解码/CUDA Graph）和 Week 9（TP/EP/通信量）一道题都没有——补 3–4 题，否则冲刺周反而丢了最近学的内容。
   - LeetGPU 又是 Matrix Transpose（D1/D3 已连续用过两次），换一道或标注复用。
5. **推荐补充资料**：本日自带实测数据已够；可加 GPU MODE 的 CUDA MODE 系列讲座（基础题口述素材）。

## Day 5：Mock 面试（369 行）

1. **当前价值**：STAR 法 + 7 环节计时框架 + follow-up 预判表（"为什么不用 vLLM""差距多少""显存不够怎么办"）是求职冲刺的核心交付；`mock_interview.py` 落盘。
2. **是否必要**：**必要**。
3. **难度等级**：简单（难在执行不在内容）。
4. **建议修改**：
   - **修复损坏的 markdown**：LeetCode 表后混入孤儿代码块（`# 栈解法…longestValidParentheses…` + 无头尾围栏），对应的 32 题不在本日题表——整段删除；今日总结第 8 条"最长有效括号……同构"同为残留碎片，删。
   - 量化数字仍是占位（"吞吐 X tokens/s、TTFT Y ms"）——到 Week 10 第五天应被 D2 联调/D3 benchmark 的真实数字替换，README 留 X/Y 会被面试官当场戳穿。
   - Mock 流程补 Week 9 位：分布式/MoE 是 75%+ JD 要求（D7 自己说的），但 7 个环节里没有分布式技术难点的讲解位。
5. **推荐补充资料**：Alex Xu《System Design Interview》（场景设计题框架）；一亩三分地/牛客的大厂 AI Infra 面经（校准 follow-up 命中率）。

## Day 6：诊断流程实战剧本 + 手撕限时清单（202 行）

1. **当前价值**：三个诊断案例（低 MFU/OOM/hang）的"现象→假设→工具→证据→结论"五段式是全课程独有内容；案例 2（`_free_finished_seq_groups` 漏释放）和案例 3（遍历时 insert 导致 livelock）明显来自真实踩坑；手撕 10 项限时清单把散落各周的 kernel 题收拢成一张表，收官设计好。
2. **是否必要**：**必要**——但作为"C1+C2 补充"合编日，体量偏轻且缺模板件（无 Coding 任务结构、无 LeetGPU/LeetCode）。
3. **难度等级**：中等。
4. **建议修改**：
   - 三份"证据（模拟留档）"里的 ncu/py-spy 输出是**编造的**（`sm__throughput 12%`、py-spy 栈帧写得像真的）——真实跑一遍留档（推荐），或显著标注"示意输出，未实测"，否则与 D3 建立的数字诚信标准冲突。
   - 手撕清单"来源"列错乱：`W8A16 dequant GEMV → week10/day2`（D2 是联调日，dequant 在 week8/day1）、`FA 简化版 → week10/day7`（D7 是复盘日）——逐行核对修正。
   - 修正标题重复（`……手撕限时清单诊断流程实战剧本 + 手撕限时清单（C1 + C2 补充）`）。
   - top-p 验收标准"与 `torch.top_p` 对比"——torch 没有 `top_p` 函数，改为"与手写参考实现/vLLM SamplingParams 行为对比"。
5. **推荐补充资料**：py-spy 文档；`NCCL_DEBUG=INFO`/`TORCH_NCCL_DEBUG` 排障指南（衔接 Week 9 分布式 hang 诊断）；CUB DeviceRadixSort/Scan 文档（top-p CUDA 化）。

## Day 7：最终复盘 —— 10 周能力地图（512 行）

1. **当前价值**：能力地图 checklist + 强弱项呈现策略（"待提升诚实承认 + 改进计划"）+ 后续 6 个月路线，是整套课程的正确收尾；"常见误区澄清"六条质量高（尤其"手写 kernel 不一定比 PyTorch 快""待提升别藏起来"）。
2. **是否必要**：**必要，但当前是错位的复盘**——盘的是旧 8 周，不是现在的 10 周。
3. **难度等级**：简单。
4. **建议修改**（本日是全周问题最集中的）：
   - **全面改写为 10 周口径**：标题/正文/目标全换；"Week 8 知识地图"的 7 天表与当前 week10 完全对不上（表里的"Day 2 架构图""Day 6 查漏补缺"已在 `_supplementary`）。
   - **能力地图补 Week 8/9 内容**：量化（Week 8 教过却列"待提升"）、分布式 TP/PP/EP（Week 9 教过却列"待提升 [ ]"）、MoE、Ascend 对比——至少移到"已掌握概念/待深入实战"档；强项 18/24 的分母重算。
   - 目录结构 `aiinfra/week8/`、`week8_summary.py` 文件名、"Week 8 完成标准"清单——全部改名/改路径。
   - 后续路线 Month 3"分布式与生产"与 Week 9 已学内容重叠，升级为"多卡实操（torchrun + NCCL 实测）"这类增量目标。
5. **推荐补充资料**：沿用文末推荐资源清单（CUTLASS/FA2/vLLM/SGLang 选得准）；加 DeepSeek-V3 技术报告作为"分布式+量化+MoE"三线延伸读物。

## 逐日汇总表

| Day | 主题 | 必要性 | 难度 | 核心问题 |
|---|---|---|---|---|
| 1 | Kernel 集成 | 必要 | 中 | 内容扎实；"Day 3/Day 4/Week 7"等引用过时 |
| 2 | 系统联调 | 必要 | 难 | **`mini_engine_v2.py` 导入路径错误 → 静默 fallback，真整合失效**；stability_test 仍是模拟引擎 |
| 3 | 项目文档 | 必要 | 简单 | 实测留档是标杆；"Week 8 的第一天"等周号过时；GEMM/FA 表数字无来源 |
| 4 | 面试题基础篇 | 必要 | 简单 | GEMM 八层双口径是亮点；缺 Week 8/9 题目；进阶篇引用指向错误 |
| 5 | Mock 面试 | 必要 | 简单 | **孤儿代码块破坏 markdown**；量化数字仍是 X/Y 占位；Mock 流程无 Week 9 位 |
| 6 | 诊断剧本+手撕清单 | 必要 | 中 | 五段式内容独有；**三份 ncu/py-spy 证据是编造输出**；手撕清单来源列错乱 |
| 7 | 最终复盘 | 必要（需重做） | 简单 | **盘的是旧 8 周**：能力地图无量化/分布式/MoE，目录写 `aiinfra/week8/` |

**本周最优先修复的三件事**：① `mini_engine_v2.py:27` 导入路径（真整合静默失效，代码级 bug）；② D7 全面改写为 10 周口径、能力地图补 Week 8/9；③ 全文周号/日号漂移 + day2/3/6 标题重复 + D5 孤儿代码块清理。
