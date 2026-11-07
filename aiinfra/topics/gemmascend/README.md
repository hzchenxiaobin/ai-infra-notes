# DeepGEMM-Ascend 逐日学习计划

建议用 **8 周**完成，每天学习 2～3 小时。每周安排 5 天新内容、1 天复盘、1 天机动或休息；如果某一天的内容没有掌握，可以顺延，不必赶进度。

以下源码路径均相对于 `DeepGEMM-Ascend/`。没有 Ascend 硬件时，可以先完成源码阅读、调用链和手工演算；正确性测试与性能实验需要在项目支持的 Ascend 环境中完成。

## 第 1 周：GEMM 与 Ascend 基础

### 第 1 天：理解 GEMM 基本问题

- 掌握 M/N/K 的含义以及 `D = A × B` 的维度关系。
- 理解计算量 `2 × M × N × K` 和输入输出数据量。
- 用一个 `4 × 3` 与 `3 × 2` 的矩阵手算结果。
- 思考：为什么大矩阵不能简单地把所有数据一次性放进片上内存？

**当天产出：** 一页 GEMM 基础笔记。

### 第 2 天：理解矩阵布局

- 学习 Row-major、Column-major、stride 和 contiguous。
- 理解 NT、NN、TN、TT 中 N/T 的含义。
- 用 PyTorch 创建张量，观察 `transpose()` 前后的 `shape`、`stride()` 和 `is_contiguous()`。
- 明确“逻辑转置”和“实际搬移数据”的区别。

**当天产出：** 一张四种 GEMM 布局的 A/B 形状对照表。

### 第 3 天：学习 Tiling 与数据复用

- 理解 `BLOCK_M/N/K` 的作用。
- 手工切分一个 `M=128、N=128、K=256` 的 GEMM。
- 计算输出 Tile 数以及每个 Tile 需要读取的 A/B 数据。
- 理解提高数据复用为什么能减少 GM 访问。

**当天产出：** 一张输出 Tile 与输入 Tile 对应关系图。

### 第 4 天：学习数值类型与量化基础

- 理解 BF16 和 FP32 累加。
- 了解 FP8、FP4 的表示范围和精度折中。
- 理解 Scaling Factor 和量化粒度的基本概念。
- 思考低精度计算为什么通常仍需要高精度累加。

**当天产出：** BF16、FP8、FP4 的优缺点对比表。

### 第 5 天：建立 Ascend 硬件与流水线概念

- 学习 GM、L1、L0A、L0B、L0C、UB 的大致职责。
- 理解 AIC 与 AIV 的分工。
- 了解 MTE1、MTE2、M 等流水线。
- 初步理解下面的数据流：

```text
GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM
```

**当天产出：** 一张 Ascend GEMM 数据流图。

### 第 6 天：复盘与基础练习

- 用 Python 或纸笔实现一个小矩阵分块计算。
- 不看笔记解释 Tiling、stride、MAD 和 Scale。
- 补齐前五天的疑问，暂不深究不理解的硬件指令细节。

**通过标准：** 能解释一个输出 Tile 需要哪些输入数据，以及数据为什么要分层搬运。

### 第 7 天：机动或休息

未完成的内容顺延到这一天；已完成则休息。

## 第 2 周：项目结构与 JIT 调用链

### 第 8 天：阅读项目说明并建立代码地图

- 阅读 `README.md` 的项目介绍、Requirements 和 Interfaces。
- 浏览 `deep_gemm/`、`csrc/`、`tests/` 和 `scripts/`。
- 区分 Python API、C++ Host、JIT、Device Kernel 和测试代码。
- 记录项目支持的 Dense GEMM、Grouped GEMM、MQA、MegaMoE 和 HC Prenorm。

**当天产出：** 一张仓库目录与职责对照表。

### 第 9 天：构建并运行最小测试

有环境时，在仓库根目录执行：

```bash
./develop.sh
python tests/test_bf16_gemm.py normal --skip-prof
```

- 阅读 `tests/test_bf16_gemm.py` 的命令行参数。
- 理解 `normal` 和 `--skip-prof` 的含义。
- 没有环境时，改为静态追踪该命令会执行哪些函数。

**当天产出：** 环境信息、运行结果或环境阻塞项记录。

### 第 10 天：追踪 Python 公开接口

- 阅读 `deep_gemm/__init__.py`。
- 找到 `bf16_gemm_nn`、`fp8_fp4_gemm_nn` 等接口的来源。
- 阅读 `csrc/python_api.cpp` 中对应的 binding。
- 理解 Python 参数如何进入 C++。

**当天产出：** `deep_gemm.bf16_gemm_nn` 的入口调用链。

### 第 11 天：理解 GemmDesc

- 阅读 `csrc/apis/gemm.hpp` 和 `csrc/apis/layout.hpp`。
- 重点分析 `get_gemm_desc()`。
- 记录 M/N/K、major、stride、dtype、group 信息如何写入 `GemmDesc`。
- 理解 Normal、M-Grouped、K-Grouped 的 shape 检查差异。

**当天产出：** 一张 `GemmDesc` 字段来源表。

### 第 12 天：理解 JIT 配置、编译与启动

- 阅读 `csrc/jit_kernels/bf16_gemm.hpp`。
- 阅读 `csrc/jit_kernels/gemm_config.hpp` 和 `heuristics.hpp`。
- 理解 `select_gemm_config()`、模板代码生成、`jit->compile()` 和 `jit->launch()`。
- 区分 `BLOCK_M/N/K` 与 `MAD_M/N/K`。
- 阅读 `csrc/jit/jit.hpp`，了解 Kernel 缓存的基本作用。

**当天产出：** 从 `GemmDesc` 到 Kernel launch 的流程图。

### 第 13 天：串起完整 Host 调用链

不看之前的笔记，独立画出：

```text
Python API
→ C++ binding
→ get_gemm_desc
→ launch_bf16_gemm
→ select_gemm_config
→ JIT compile
→ jit launch
→ bf16_gemm_impl
```

标注每一层的输入、输出和关键职责，再对照源码修正。

**通过标准：** 能说明 `GemmDesc` 与 `GemmConfig` 的区别，以及 JIT 为什么需要生成模板实例化代码。

### 第 14 天：机动或休息

可以补跑测试，或继续研究本周尚未理解的 JIT 细节。

## 第 3 周：BF16 Kernel——模板、调度与数据搬运

### 第 15 天：阅读 Kernel 模板参数

主读 `deep_gemm/include/deep_gemm/bf16_gemm.hpp` 的模板声明和函数入口。

- 分类整理 Shape、Block、MAD、stage、Core、L2 和 Epilogue 参数。
- 理解编译期 shape 与运行时 shape 的关系。
- 理解 `kDirectStore`、`kResidentA/B` 的初步用途。

**当天产出：** 模板参数分类表。

### 第 16 天：分析片上内存规划

- 找到 L1、L0A、L0B、L0C 的声明。
- 理解多阶段缓冲如何通过 offset 分配。
- 阅读与容量、对齐有关的 `static_assert`。
- 手工估算一组配置的 L1/L0 使用量。

**当天产出：** 一张 Kernel 片上内存布局图。

### 第 17 天：理解 Normal GEMM 调度

- 阅读 `deep_gemm/include/deep_gemm/scheduler.hpp`。
- 理解 M/N 维如何切块并分配给 AI Core。
- 分析 `get_next_block()`、`get_global_idx()` 和边界处理。
- 用一个具体 shape 手工计算每个 block 的任务。

**当天产出：** 一张 M/N Tile 调度示意图。

### 第 18 天：理解地址与 Tile 表达

- 阅读 `ascend/tile_ptr.hpp`。
- 理解 `gm_ptr`、`l1_ptr`、`l0a_ptr` 等类型表达的对象。
- 追踪 `index()`、`batch()`、`group()` 和 stride 的作用。
- 重点思考 MN-major 与 K-major 对地址计算的影响。

**当天产出：** 一组地址计算示例。

### 第 19 天：追踪 GM 到 L1/L0 的搬运

- 阅读 `ascend/copy.hpp`。
- 分析 `copy_gm_to_l1()` 和 `copy_l1_to_l0a/b()`。
- 对一次搬运记录源地址、目标地址、实际大小和 stride。
- 理解 `actual_m/n/k` 对边界搬运的作用。

**当天产出：** `GM → L1 → L0` 数据搬运图。

### 第 20 天：完整演算一个输出 Tile

选择 `M=128、N=128、K=256`，假设一组 Block/MAD 参数：

- 计算输出 Tile 数和 K 维循环次数。
- 列出每次搬运的数据区域。
- 标出边界 Tile 和片上缓冲的使用。
- 对照源码检查自己的演算。

**通过标准：** 能把调度、地址计算和数据搬运串成一个完整过程。

### 第 21 天：机动或休息

建议重读 `bf16_gemm_impl`，但暂时跳过同步细节，留到下一周。

## 第 4 周：BF16 Kernel——MAD、流水线与 Epilogue

### 第 22 天：拆解 K 维两层循环

- 区分 `BLOCK_K` 循环和 `MAD_K` 循环。
- 理解 `k_blk_idx`、`k_mad_idx` 和 `eff_mad_k`。
- 分析 `is_first`、`is_last` 对累加的影响。
- 手工展开一个小 shape 的循环执行顺序。

**当天产出：** K 维循环展开图。

### 第 23 天：理解多级流水与同步

- 阅读 `ascend/sync.hpp`。
- 分析 `asc_sync_wait()` 和 `asc_sync_notify()` 的配对关系。
- 理解 L1 stages、L0 stages 和 stage 下标循环。
- 解释为什么同步过少会产生竞争，过多又可能降低并行度。

**当天产出：** MTE1/MTE2/M 流水线时序图。

### 第 24 天：理解 MAD 计算与结果暂存

- 阅读 `mad()` 调用及其前后同步。
- 理解 L0A/L0B 到 L0C 的计算过程。
- 分析 `ENABLE_UPDATE`、`ENABLE_KEEP` 和首次累加。
- 理解多个 MAD Tile 如何组成一个 Block 输出。

**当天产出：** 一段 MAD 执行过程的伪代码。

### 第 25 天：理解 AIC/AIV 和 Epilogue

- 阅读 `deep_gemm/include/deep_gemm/epilogue.hpp`。
- 对比 Kernel 中 `ASCEND_IS_AIC` 的两个分支。
- 理解 accumulation、bias、alpha 和输出转换的位置。
- 分析 AIC store 与 AIV drain 如何配合。

**当天产出：** AIC/AIV 分工与 Epilogue 流程图。

### 第 26 天：学习特殊执行路径

- 对比 `kDirectStore` 与普通 Epilogue 路径。
- 理解 `kResidentA/B` 的数据复用方式。
- 查看非对齐 shape 和空组处理。
- 思考这些路径分别适用于什么场景，以及有哪些约束。

**当天产出：** 普通路径与特殊路径对比表。

### 第 27 天：完整复述 BF16 GEMM

不看源码，画出：

```text
任务调度
→ GM/L1/L0 数据搬运
→ MAD 累加
→ AIC/AIV Epilogue
→ 写回 GM
```

然后对照 `bf16_gemm.hpp` 修正，并记录仍不理解的三个问题。

**通过标准：** 能讲清楚一个输出 Tile 从调度到写回的完整生命周期。

### 第 28 天：机动或休息

可以重新运行 BF16 测试，将测试输出与本周读到的 Kernel 行为对应起来。

## 第 5 周：测试体系与低精度 GEMM

### 第 29 天：理解测试数据生成

- 阅读 `tests/generators.py`。
- 理解 shape、major、recipe、grouped layout 如何组合。
- 找到非对齐和 strided 场景的生成逻辑。
- 跟踪 `GemmTestDesc.generate()` 的输出。

**当天产出：** 测试用例字段说明表。

### 第 30 天：理解正确性验证流程

- 阅读 `tests/common.py` 和 `tests/utils.py`。
- 分析参考结果、ACLNN 交叉验证和误差比较。
- 理解 accumulation、bias、alpha 的语义。
- 理解随机舍入为什么需要特殊比较方法。
- 了解 meta tensor 并行编译的目的。

**当天产出：** 一条完整测试流程图。

### 第 31 天：运行并分析 BF16 测试

有环境时选择部分用例运行：

```bash
python tests/test_bf16_gemm.py normal --skip-prof
python tests/test_bf16_gemm.py unaligned-k --skip-prof
```

- 将测试场景与源码中的边界处理对应起来。
- 如果失败，先判断是环境、参考结果、布局还是 Kernel 问题。
- 没有环境时，选取一个测试用例做静态推演。

**当天产出：** 测试记录或静态推演笔记。

### 第 32 天：学习 FP8/FP4 Kernel 路径

- 阅读 `tests/test_fp8_gemm.py` 和 `tests/test_fp4_gemm.py`。
- 阅读 `csrc/jit_kernels/fp8_gemm.hpp`。
- 阅读 `deep_gemm/include/deep_gemm/fp8_gemm.hpp` 和 `fp8_dequant_gemm.hpp`。
- 对比 FP8×FP8、FP4×FP4、FP8×FP4 的执行路径。
- 找到反量化和高精度累加的位置。

**当天产出：** BF16 与低精度 GEMM 的主要差异表。

### 第 33 天：理解 Scale 布局转换

- 阅读 `csrc/apis/layout.hpp` 和 `csrc/utils/layout.hpp`。
- 阅读 `deep_gemm/include/deep_gemm/transform_sf.hpp`。
- 阅读 `tests/test_layout.py`。
- 理解 `recipe`、UE8M0 打包和 MN-major Scale 布局。
- 解释为什么不能总是直接使用用户输入的 Scale 布局。

**当天产出：** Scale 从输入到 Kernel 布局的转换图。

### 第 34 天：串联低精度调用链

选择 `fp8_fp4_gemm_nn`，追踪：

```text
输入与 Scale
→ recipe 检查
→ Scale 布局转换
→ JIT 配置
→ 低精度 Kernel
→ 结果验证
```

有环境时可运行 `python tests/test_fp8_gemm.py normal --skip-prof`。

**通过标准：** 能解释 Scaling Factor 在哪里转换、在哪里参与计算。

### 第 35 天：机动或休息

补充低精度数值基础，或复盘 BF16 Kernel 中仍然不清楚的部分。

## 第 6 周：Grouped GEMM 与 MQA Logits

### 第 36 天：学习 M-Grouped GEMM

- 阅读 `csrc/apis/gemm.hpp` 的 M-Grouped 分支。
- 阅读 `scheduler.hpp` 中相关调度逻辑。
- 阅读 `tests/common.py` 中 M-Grouped 调用。
- 理解 A、B、D 和 `grouped_layout` 的形状及对应关系。
- 将 token 分组与 MoE expert 计算联系起来。

**当天产出：** M-Grouped 输入输出示意图。

### 第 37 天：学习 K-Grouped GEMM

- 阅读 K-Grouped 的 API 和 Scheduler 分支。
- 理解 K 维分组、每组输出和累加语义。
- 对比 Normal、M-Grouped、K-Grouped。
- 分析空组和非对齐组的处理。

**当天产出：** 三种 GEMM 类型对比表。

### 第 38 天：运行或推演 Grouped GEMM

有环境时运行：

```bash
python tests/test_bf16_gemm.py mgrouped --skip-prof
python tests/test_bf16_gemm.py kgrouped --skip-prof
```

- 跟踪一个 group 从 layout 到输出区间的映射。
- 没有环境时，手工构造两个不等长 group 并推演调度过程。

**通过标准：** 给定一个 MoE 输入，能说明 token、group、expert 权重和输出的对应关系。

### 第 39 天：学习普通 MQA Logits

- 阅读 `tests/test_attention.py`。
- 阅读 `csrc/apis/attention.hpp` 和 `csrc/jit_kernels/mqa_logits.hpp`。
- 阅读 `deep_gemm/include/deep_gemm/mqa_logits.hpp`。
- 理解 Q、KV、weights 的形状和 logits 计算过程。
- 对比 MQA Logits 与普通 GEMM 的调度及 Epilogue。

**当天产出：** MQA Logits 计算流程图。

### 第 40 天：学习 Paged/Sparse MQA

- 阅读 `sparse_mqa_logits.hpp`、`paged_sparse_mqa_logits.hpp`。
- 阅读对应 metadata 和 scheduler 文件。
- 理解 Paged KV 寻址和 Sparse block 选择。
- 理解 FIRST/REUSE 状态以及 KV 快照复用。
- 梳理 Prefill 与 Decode 的主要差异。

**当天产出：** Paged/Sparse MQA 元数据与数据流图。

### 第 41 天：复盘 Grouped GEMM 与 MQA

- 对比普通 GEMM、Grouped GEMM 和 MQA 的计算模式。
- 总结哪些代码复用了 GEMM 主线概念，哪些属于新的调度或寻址问题。
- 整理一张术语表：group、paged、sparse、context、KV cache。

### 第 42 天：机动或休息

如果对 MQA 不感兴趣，可用这一天提前开始 MegaMoE，不必逐行读完所有 MQA 变体。

## 第 7 周：MegaMoE、HC Prenorm 与扩展算子

### 第 43 天：理解 MegaMoE 对外接口

- 阅读 `tests/test_mega_moe.py` 和 `deep_gemm/mega/__init__.py`。
- 阅读 `csrc/apis/mega_moe.hpp`。
- 理解权重转换、对称缓冲区和主要输入输出。
- 找到 EP、top-k、expert 和 shared expert 等概念的位置。

**当天产出：** MegaMoE API 与数据准备流程图。

### 第 44 天：理解 MegaMoE 计算主线

- 阅读 `deep_gemm/include/deep_gemm/mega_moe.hpp`。
- 阅读 `scheduler/mega_moe.hpp` 和 `layout/mega_moe.hpp`。
- 理解两个 Grouped GEMM 与 SwiGLU 如何衔接。
- 将中间张量的形状变化记录下来。

```text
Dispatch → Grouped GEMM 1 → SwiGLU → Grouped GEMM 2 → Combine
```

**当天产出：** MegaMoE 张量形状变化表。

### 第 45 天：理解 MegaMoE 通信与融合

- 阅读 `comm/mega_moe.hpp` 和 `csrc/backends/hccl.hpp`。
- 理解 EP Dispatch 与 Combine 的通信对象。
- 分析计算和通信为什么需要融合。
- 初步判断不同 token 数下可能存在的计算或通信瓶颈。

**当天产出：** MegaMoE 计算/通信时序图。

### 第 46 天：学习 MegaMoE 测试与性能入口

- 阅读 `scripts/profiling/msprof_mega_moe.py`。
- 查看 `tests/test_mega_moe.py` 如何组织多 rank 测试。
- 理解性能结果中的计算带宽、内存带宽和通信带宽。
- 没有多卡环境时，以静态分析为主，不强行搭建环境。

**当天产出：** MegaMoE 测试依赖和性能指标清单。

### 第 47 天：学习 HC Prenorm / TileLang

- 阅读 `deep_gemm/tilelang_ops/hc_prenorm.py`。
- 阅读 `tests/test_hyperconnection.py`。
- 理解 Prenorm 与 GEMM 的融合。
- 比较手写 Ascend Kernel 与 TileLang Kernel 的代码结构和开发方式。

**当天产出：** 两种 Kernel 开发方式对比笔记。

### 第 48 天：扩展算子复盘

从 MQA、MegaMoE、HC Prenorm 中选择最感兴趣的一个，回答：

- 输入输出和核心计算是什么？
- 调度与寻址有什么特殊之处？
- 为什么普通 GEMM 接口不能直接满足需求？
- 性能瓶颈可能在哪里？

**通过标准：** 至少能完整讲解一个扩展算子的计算流程。

### 第 49 天：机动或休息

可补充 MoE、Attention 或 TileLang 背景知识，也可以继续补做之前的实验。

## 第 8 周：性能分析与综合实践

### 第 50 天：建立性能分析基础

- 阅读 `deep_gemm/testing/bench.py` 和 `tests/utils.py`。
- 理解 TFLOPS、带宽和算术强度。
- 区分计算受限、访存受限和流水线受限。
- 理解冷缓存、热缓存以及 JIT 编译耗时与执行耗时的区别。
- 阅读项目 README 的 Performance 部分。

**当天产出：** 一份性能指标公式与解释笔记。

### 第 51 天：观察 JIT 配置与 Profiling 输出

有环境时尝试：

```bash
export DG_PRINT_CONFIGS=1
export DG_JIT_PRINT_COMPILER_COMMAND=1
```

- 运行一个较小的 GEMM 测试并记录所选配置。
- 去掉 `--skip-prof`，观察性能输出。
- 将 shape、配置、耗时、TFLOPS 和带宽放在同一张记录表中。
- 没有环境时，分析 README 中已有性能数据。

**当天产出：** 一组基线性能记录。

### 第 52 天：完成单变量性能实验

固定一个 shape，每次只改变一个因素：

- Block 大小；
- L1/L0 stage 数；
- resident A/B；
- direct store；
- AI Core 数量。

每次实验都先验证正确性，再比较性能，避免同时修改多个参数。

**通过标准：** 能根据结果提出并验证一个性能假设，而不是只给出一组调参结果。

### 第 53 天：选择并实施综合小项目

从下面选择一项：

1. 新增一个非对齐或 strided 测试场景。
2. 针对一个 shape 修改启发式配置并比较结果。
3. 增加一个简单的 Epilogue 变体。
4. 不写代码，完成一份 BF16 Kernel 的详细源码讲解。

先列出修改范围、预期结果和验证方法，再开始实施。

### 第 54 天：验证与修正

- 运行与改动相关的正确性测试。
- 有性能改动时，与第 51 天的基线比较。
- 检查新增测试是否真正覆盖目标场景。
- 记录失败实验，不要为了得到好结果而省略不利数据。

**当天产出：** 修改说明、测试结果和性能对比。

### 第 55 天：最终复盘与分享

准备一份 15 分钟分享或 2～3 页总结，至少包含：

- 仓库整体架构；
- Python 到 Device Kernel 的调用链；
- BF16 GEMM 数据流和流水线；
- 测试与正确性验证方法；
- 一个扩展算子；
- 一次性能分析或综合实践。

最后检查自己是否能够：

- 独立追踪公开 API 到 Device Kernel；
- 解释一个输出 Tile 的完整生命周期；
- 判断测试失败的大致层级；
- 说明低精度与 Grouped GEMM 的主要差异；
- 对一个 shape 做有依据的性能分析。

### 第 56 天：休息与后续规划

整理笔记和代码，保留仍未解决的问题。后续可以按兴趣继续深入 Ascend 指令级优化、MegaMoE 多卡通信、MQA 稀疏寻址或 TileLang。

## 核心完成标准

不一定要逐行读完所有文件，但完成本计划后应重点掌握：

1. **BF16 GEMM 主线**：调度、搬运、MAD、流水线和 Epilogue。
2. **JIT 调用链**：参数检查、配置选择、代码生成、编译缓存和启动。
3. **测试体系**：参考结果、边界场景、布局和低精度误差验证。
4. **一个扩展方向**：MQA、MegaMoE 或 HC Prenorm 至少掌握一个。
5. **独立实践能力**：能完成一次小修改，并验证正确性和性能。
