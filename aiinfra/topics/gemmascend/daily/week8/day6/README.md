# 第 8 周 · 第 6 天（总第 55 天）：最终复盘与分享

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 8 周"性能分析与综合实践"的第 6 天（最终复盘日）教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 55 分钟，源码导读约 15 分钟，综合练习（含试讲/写作）约 60 分钟，通过标准自评与自测题约 20 分钟。

八周计划今天到达终点站。前 54 天的回答可以排成一条线：**代码在哪里、怎么组织**（第 1～2 周）→ **BF16 Kernel 内部怎么算**（第 3～4 周）→ **怎么验证算得对、低精度怎么算**（第 5 周）→ **分组与扩展算子**（第 6～7 周）→ **算得快不快、为什么**（第 8 周）。学习计划给最后一天的任务只有一句话：**准备一份 15 分钟分享或 2～3 页总结**，内容至少覆盖六个板块——仓库整体架构、Python 到 Device Kernel 的调用链、BF16 GEMM 数据流和流水线、测试与正确性验证方法、一个扩展算子、一次性能分析或综合实践；随后**自查五项能力**：独立追踪公开 API 到 Device Kernel、解释一个输出 Tile 的完整生命周期、判断测试失败的大致层级、说明低精度与 Grouped GEMM 的主要差异、对一个 shape 做有依据的性能分析。复盘方法论与前七次复盘日相同：**闭卷复述是唯一标准**——第三节给出六个板块的参考骨架（每条结论都锚定文件与行号），请先自己写再对照；第七节把五项能力拆成可打勾的检查点。明天（第 56 天）是休息与后续规划，今天把"还欠什么"归档好即可收工。

## 一、今日定位与学习目标

完成今天的学习后，你应当能够：

1. 闭卷写出**六板块总结**的骨架：每个板块一页纸内的主干结论 + 至少三个 `文件:行号` 级锚点，且能用一句话向没有读过本仓库的同事说明每个板块"它回答什么问题"；
2. 按 15 分钟的时间预算**组织一次分享**（建议：架构 2 分钟、调用链 3 分钟、数据流 4 分钟、测试 2 分钟、扩展算子 2 分钟、性能实践 2 分钟），并说出哪两个板块最值得多给时间、为什么；
3. 通过五项能力自查：**① 从 `deep_gemm.bf16_gemm_nt` 出发独立追踪到 `bf16_gemm_impl` 的 26 个模板参数与 9 个运行期参数；② 讲清一个输出 Tile 从领任务到写回 GM 的完整生命周期（含五个环、四组队列事件、AIC/AIV 借还协议）；③ 对一次假想失败按"编译期 / Host / 结果层"三层分诊；④ 默写 BF16、低精度、M-Grouped、K-Grouped 四条路径的差异表；⑤ 对一个给定 shape 手算 TFLOPS/GB/s/AI 并判定受限类型**；
4. 把八周积累的遗留问题逐条归档，标注去向（第 56 天起的四个深入方向：Ascend 指令级优化、MegaMoE 多卡通信、MQA 稀疏寻址、TileLang）；
5. 区分"已能独立复述"与"看过但依赖笔记"的知识，后者列出回读计划——复盘的目的不是宣布全懂，而是知道自己站在哪里。

## 二、预备知识

今天的复盘覆盖全部八周，不引入新知识；哪一条卡住，按索引回到对应天数补漏：

| 周 | 板块 | 复习入口 |
| :---: | :--- | :--- |
| 1 | GEMM 基础、布局、Tiling、数值类型、Ascend 存储层次 | 第 1～6 天；第 6 天复盘（通过标准：输出 Tile 需要哪些输入、数据为什么分层搬运） |
| 2 | 项目结构、JIT 调用链 | 第 13 天复盘：八层链全景图、`GemmDesc` vs `GemmConfig`、模板实例化三问 |
| 3 | Kernel 模板、片上内存、调度、地址、搬运 | 第 20 天复盘：`M=128, N=128, K=256` 单 Tile 全演算 |
| 4 | K 循环、流水线、MAD、AIC/AIV、Epilogue、特殊路径 | 第 27 天复盘：五段流程 + `M=256, N=128, K=512` 生命周期演算 |
| 5 | 测试体系、FP8/FP4 路径、Scale 布局 | 第 34 天复盘：低精度调用链串联 |
| 6 | M/K-Grouped、MQA 家族 | 第 41 天复盘：三种 GEMM 对比 + MQA 流程 |
| 7 | MegaMoE、HC Prenorm | 第 48 天复盘：三算子四问参考答案 |
| 8 | 性能指标、基线、单变量实验、综合项目、验证修正 | 第 50～54 天；昨天的三件套报告（修改说明、测试结果、性能对比）是今天"综合实践"板块的现成素材 |

硬件常量速查（`deep_gemm/include/deep_gemm/common.hpp:51-65`，Ascend 950DT）：32 个 AIC、64 个 AIV；L1 每核 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB、UB 256 KiB；`FRAC_MN = 16`（NZ 分形行数）、`MX_SF_DIVISOR = 64`（每个 SF pair 覆盖的 K 元素数）。**环境说明**：第五节 A 组任务需要 Ascend 环境（CANN 9.20 + torch_npu，已按第 9 天 `./develop.sh` 构建）；无环境时完成 B 组静态任务即可达到当天目标——今天是复盘日，核心产出（分享/总结）本来就不依赖硬件。

## 三、知识梳理

### 3.1 八周一览：每天回答了什么问题

| 天 | 问题 | 一句话结论 |
| :---: | :--- | :--- |
| 1～5 | GEMM 是什么、在什么硬件上算 | `D = A×B`，计算量 `2·M·N·K`；四种布局由 stride 决定；Tiling 换复用；BF16/FP8/FP4 是精度折中；`GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` |
| 8～12 | 代码在哪里、怎么启动 | 五类代码一图（见 3.2）；`GemmDesc`（算什么）→ `GemmConfig`（怎么算）→ 模板实例化代码 → Bisheng 编译（缓存 `~/.dj`）→ 携 9 个运行期参数 launch |
| 15～20 | Kernel 内部怎么分工 | 26 模板参数分七类；片上缓冲 `constexpr` 定址 + `static_assert` 查容量；持久化调度 + swizzle；`gm_ptr::index` 地址代数；ND2NZ 分形搬运与三级边界裁剪 |
| 22～26 | 计算怎么流水起来 | K 两层循环（`s`/`s_ab` 两个环）；四组队列事件 + 两组块内事件；`clear_C → KEEP → UPDATE` 累加链；AIC store 与 AIV drain 的 UB 借还；`kDirectStore`/`kResidentA/B` 是主线的条件裁剪 |
| 29～34 | 怎么知道算对了 | 两阶段测试（meta 预编译 + 五类断言）；`calc_diff` 相对误差与 `1e-5` 阈值；SF 转换成 UE8M0 打包 MN-major；反量化在 MAD 内部、累加在 L0C |
| 36～40 | 分组与注意力 | M-Grouped 用 psum 前缀和（BLOCK_M 钉死 256）；K-Grouped 共享 M×N 网格、逐组 K 段；MQA 是"非线性 + 窗口 + 归约"的新输出语义，FIX-pipe bound |
| 43～47 | 融合算子 | MegaMoE 五段一 kernel（任务重放 + 进度计数 + 环形池）；HC Prenorm 用 TileLang 做 K-split 原子归约，一份搬运两种计算 |
| 50～54 | 快不快、为什么 | TFLOPS/GB/s/AI 三公式 + Roofline 判别；冷 L2 与设备侧计时；单变量实验纪律；综合小项目与验证修正 |

### 3.2 板块一：仓库整体架构

**五类代码，一张地图**（第 8 天的产出订正版）：

| 目录 | 职责 | 关键文件 |
| :--- | :--- | :--- |
| `deep_gemm/`（Python） | 门面与 re-export、MegaMoE Python 侧、TileLang 算子、测试工具 | `__init__.py`（`:26-81` 从 `_C` re-export 全部接口，`:159` `_C.init` 初始化 JIT）、`mega/__init__.py`、`tilelang_ops/hc_prenorm.py`、`testing/bench.py` |
| `csrc/`（C++ Host） | binding、接口层、配置与启动、JIT 运行时封装 | `python_api.cpp`（唯一翻译单元，`:30-40` 的 `PYBIND11_MODULE` 依次注册 config/epilogue/gemm/attention/einsum/layout/aclnn/mega_moe 八组 API）、`apis/`、`jit_kernels/`、`jit/jit.hpp`（`:16` 全局 `jit` 对象、`:34-49` Runtime 的核数与 dry-run、`:59` `kMkAlignment = 256`）、`utils/`、`backends/hccl.hpp` |
| `deep_gemm/include/deep_gemm/`（设备侧模板） | Device Kernel 头文件，**随包分发、运行时才由 Bisheng 编译** | `bf16_gemm.hpp`、`fp8_gemm.hpp`、`fp8_dequant_gemm.hpp`、`scheduler.hpp`、`epilogue.hpp`、`ascend/`（`copy.hpp`/`sync.hpp`/`tile_ptr.hpp`）、`mqa_logits.hpp`、`mega_moe.hpp`、`transform_sf.hpp` |
| `tests/` | 测试套件：选择器 + 编排 + 生成器 + 比较器 | `test_bf16_gemm.py` 等 10 个脚本、`common.py`（`run_gemm_test` `:131`）、`generators.py`、`utils.py` |
| `scripts/`、`third-party/` | profiling 与 pyi 生成；第三方 TileLang 算子 | `scripts/profiling/msprof_mega_moe.py` |

**算子版图**（README "Interfaces" 与 `__init__.py` 对应）：Dense GEMM（BF16 / FP8 / FP4 / 混合 FP8×FP4，各 nt/nn/tn/tt 四布局）、M-Grouped 与 K-Grouped contiguous、MQA Logits（normal / sparse / paged / paged sparse 四条路径）、MegaMoE（`SymmBuffer` / `transform_weights_for_mega_moe` / `fp8_fp4_mega_moe` 三个符号）、HC Prenorm（`tf32_hc_prenorm_gemm`）、Einsum；工具函数 `set_num_sms`、`transform_sf_into_required_layout`、ACLNN 参考 GEMM 等。README 的 Performance 表给出硬件上限锚点：FP4 = 1730、FP8 = 865、BF16 = 432 TFLOPS（`README.md:108-113`）。

分享时的一句定位：**这是一个"Python 门面 + C++ Host 策略层 + 设备模板 Kernel 三段式"的 JIT 库——接口与 DeepGEMM 上游完全兼容，性能靠把 tiling/流水/对齐焊进编译期常量逼近硬件极限**（`README.md:5` 的 "lightweight abstraction over the Ascend MAD primitives"）。

### 3.3 板块二：Python 到 Device Kernel 的调用链

八层链（第 13 天复盘的浓缩版，行号已对照当前源码核实）：

```text
① deep_gemm.bf16_gemm_nt(a, b, d)                     deep_gemm/__init__.py:64（re-export）
② bf16_gemm<false,false> binding                      csrc/apis/gemm.hpp:425-448（注册 :632-637，
                                                       经 csrc/python_api.cpp:35 进 _C）
③ get_gemm_desc() → optional<GemmDesc>                csrc/apis/gemm.hpp:60-197
④ launch_bf16_gemm(desc)                              csrc/jit_kernels/bf16_gemm.hpp:16-78
   ⑤ select_gemm_config(desc) → GemmConfig            csrc/jit_kernels/heuristics.hpp:592-755
   ⑥ std::format 渲染 26 实参 → jit->compile           csrc/jit_kernels/bf16_gemm.hpp:18-71
   ⑦ jit->launch(kernel, launch_options, 9 参数)       csrc/jit_kernels/bf16_gemm.hpp:73-77
⑧ bf16_gemm_impl<...26 模板参数>(...9 运行期参数)      deep_gemm/include/deep_gemm/bf16_gemm.hpp:27-34
```

一句话版：**用户张量经 binding 归一（逻辑转置 + dtype 断言 + epilogue 归一）后由 `get_gemm_desc` 收敛成 `GemmDesc`（算什么）；`launch_bf16_gemm` 用它选出 `GemmConfig`（怎么算），把二者渲染成一份 `bf16_gemm_impl<...>` 显式实例化源码，交给 DeepJIT/Bisheng 编译（可缓存于 `~/.dj`）并携带 9 个运行期参数启动到当前 NPU stream 上**（stream 来自 `csrc/python_api.cpp:21` 的 `get_current_npu_stream`）。

必须随身携带的三个判别点：

1. **`GemmDesc` vs `GemmConfig`**：定义都在 `csrc/jit_kernels/gemm_config.hpp`，注释原文分别是 "Problem descriptor: **what to compute**" 与 "Execution configuration: **how to compute**"；desc 由 `get_gemm_desc` 从用户张量提取校验，config 由 `select_gemm_config` 启发式搜索——同一 desc 可配不同 config（第 52 天单变量实验正是人为换配置）。
2. **为什么 JIT 要生成模板实例化代码**：`bf16_gemm_impl` 是 26 参函数模板，不实例化就没有机器码；把 block/mad/stages 焊成编译期常量，kernel 才能用 `constexpr` 规划片上地址、用 `static_assert`（`bf16_gemm.hpp:66-70`）在编译期暴露容量错误。
3. **代价与三层对策**：二进制缓存（`DG_JIT_CACHE_DIR`，默认 `~/.dj`）、`compiled_dims`（nt/nn 默认 `"nk"`，M 留运行期）、dry-run 预热（`compile_without_load`，`csrc/jit_kernels/bf16_gemm.hpp:67-70`；`deep_gemm/testing/par_compile.py:6-13` 的 `set_dry_run(True)`）。

### 3.4 板块三：BF16 GEMM 数据流与流水线

**五段主叙事**（每段一句话 + 锚点，完整版见第 27 天复盘）：

```text
bf16_gemm_impl（__mix__(1, kDirectStore ? 0 : 2)：每 block 1 AIC + 2 AIV，bf16_gemm.hpp:27）
① 调度   Scheduler::get_next_block（:89 → scheduler.hpp:141）：block_idx = (++iter)×kNumCores + core_id
          持久化认领输出 Tile；m_idx/n_idx/actual_m/n 由 get_global_idx（scheduler.hpp:113）与
          get_actual_m/n（scheduler.hpp:129/:137）给出
② 搬运   GM→L1（MTE2，:117-118 → copy.hpp:102）：按 actual_m/n/k 少搬，nd2nz 分形重摆；
          L1→L0A/L0B（MTE1，:143/:145 → copy.hpp:134/:148）：按对齐后整分形搬，条件复用
③ MAD    mad()（M 队列，:149-151 → copy.hpp:371）：首条 clear_C、中间 KEEP、末条 UPDATE，
          部分和攒在 L0C[s_cd]（fp32）
④ 交付   aic_wait_free（:129-131）+ aic_store（:161 → epilogue.hpp:42-47）：
          L0C →FIX（F322BF16）→ UB[s_epi]，round-robin 平分给两个 AIV
⑤ 写回   aiv_drain（:190 → epilogue.hpp:49-87）：wait → copy_ub_to_gm（epilogue.hpp:81）→ 归还 arrive；
          累加时 aiv_initialize 设原子加（epilogue.hpp:89-97）
```

**一个输出 Tile 的生命周期**（自查能力②，第 27 天演算的骨架）：核 c 领到 (m_blk, n_blk) → `m_idx = m_blk·BLOCK_M`、`actual_m = min(shape_m − m_idx, BLOCK_M)` → `num_k_blocks = ceil_div(shape_k, BLOCK_K)`（`:96`）→ 每个 k_blk 走一遍"wait L1[s] 空闲（`:116`）→ GM→L1 → notify 就绪（`:119`）→ 四个子块循环（`:123-124`）→ 每 k_mad 四个同步点（`:141/:146/:148/:152`）→ notify L1[s] 归还（`:167`）"→ 最后一个 k_blk 里每子块 K 遍历结束即 `aic_store` → 块尾 `epi_base = s_epi` 续接（`:169-171`）→ AIV 按同一 Scheduler 序列排水写 GM → `get_next_block` 返回 false 则三方退出。

**五个环**（必须能默写）：`s`（L1 stage，模 `kNumL1Stages`，`:112`）、`s_ab`（L0 stage，模 `kNumL0Stages`，`:133`）、`s_cd`（L0C 子块，模 `num_mad_cd_stages = (BLOCK_M/MAD_M)×(BLOCK_N/MAD_N)`，`:124/:51`）、`s_epi`（UB 总编号，每 k_blk 开头重置为 `epi_base`、每子块末尾 +1，`:122/:164`）、`epi_base`（跨 Tile 续接，`:169`）。**事件号 = 缓冲编号**是"少量事件支撑任意长循环"的关键。

**变体是主线的条件裁剪**（第 26 天）：`kDirectStore` 删掉 UB 借还与 AIV 分支（`:27/:156-159/:174`，约束在 `heuristics.hpp:521-533` 的 `can_direct_store`）；`kResidentA/B` 删掉循环内搬运（`:109-110/:117/:143-145`，设备断言 `:40/:45` + host 闸门 `heuristics.hpp:552-557`）。

### 3.5 板块四：测试与正确性验证方法

**两阶段编排**（`tests/common.py:131-237` 的 `run_gemm_test`）：阶段 1 用 meta 张量 + `par_compile` 在 dry-run 下并发预编译全部 case（`:152-178`）；阶段 2 逐 case 运行五类验证（`:180-236`）。

**五类断言**（每类一句）：

| # | 断言 | 验证什么 | 锚点 |
| :---: | :--- | :--- | :--- |
| ① | dg 内核 vs `d_ref`（FF/FT/TF 三组合） | 内核忠实复现 `alpha·(A@Bᵀ)+c` 参考公式 | `tests/common.py:199-201` |
| ② | 直接输出 vs fp32 累加（仅 k-grouped 非累加） | 直写路径的内部 cast 与"fp32 累加后 host 转"逐位一致 | `tests/utils.py:61-67`、`common.py:202-206` |
| ③ | ACLNN 交叉验证（仅 Normal） | 第二个独立实现同样要对 | `common.py:87-128`、`csrc/apis/aclnn.hpp` |
| ④ | 随机舍入复刻 | host 用 numpy 复刻内核的确定性 quartet 哈希，逐位比较 | `utils.py:70-92`、`common.py:215-221` |
| ⑤ | 性能测量 | `bench_msprof` + `print_perf` | `common.py:223-236` |

**误差口径**：`calc_diff = ‖x−y‖²/(‖x‖²+‖y‖²)`（`tests/utils.py:36-43`），尺度不变、允许少数元素差 1～2 ULP；BF16 用 `threshold=1e-5`（cube 与参考的 K 累加顺序差，`utils.py:49-51` 注释），`threshold=0` 走 `torch.equal` 逐位比较（FP8 MX 路径）。

**三个护栏**：`d` 预填 `randn` 垃圾（k-grouped 非累加填 NaN，`common.py:190`）让漏写显形；`disable_cpu_fallback()`（`common.py:142` → `deep_gemm/testing/disable_fallback.py:44-73`）把静默 CPU 回退变成响亮报错；确定性种子（`generators.py:9-12` 的 `reset_seed`）保证失败可精确复现。

**失败三层分诊**（自查能力③）：

| 层 | 在哪暴露 | 典型信号 |
| :--- | :--- | :--- |
| 编译期 | JIT 编译（Compiling 进度条阶段） | 设备 `static_assert`：L1/L0A/L0B/L0C overflow（`bf16_gemm.hpp:66-70`） |
| Host 侧 | launch 前，Python `RuntimeError` | `DJ_HOST_ASSERT` 文本，如 "Bias must be contiguous"（`csrc/apis/gemm.hpp:114`）、"alpha and bias cannot be used together"（`:116`） |
| 结果层 | 断言处 | `assert_equal` 报错带行区间与 diff 值（`tests/utils.py:55-58`）；注意有"预期报错"的合法测试（`common.py:191-198`） |

再叠一层"环境"判别（第 31 天）：`ModuleNotFoundError`、Bisheng 不存在 → 环境；`dg:` 与 `aclnn:` 同时失败 → 优先怀疑参考；报错含 "Ambiguous major axis" 等 → 布局；仅 `dg:` 失败且 diff 超阈值 → Kernel。

### 3.6 板块五：一个扩展算子（三选一讲深）

分享只需讲透一个，另两个各给"为什么不是普通 GEMM"的最强一条理由。三算子速查（细节见第 48 天复盘）：

| 维度 | MQA Logits | MegaMoE | HC Prenorm |
| :--- | :--- | :--- | :--- |
| 数学核心 | `logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩)·w[q,h]`，仅 `j ∈ [ks,ke)` 有效 | MoE 前馈整层：Dispatch → GEMM1 → SwiGLU → GEMM2 → Combine | `D = float(A)@B.T` **且** `sqr_sum = Σ(float(A)²)`，一个 kernel |
| 主文件 | `mqa_logits.hpp`（core `:195`、impl `:623`、paged `:650`） | `mega_moe.hpp`（`mega_moe_impl` `:1127`，仓库最大算子） | `hc_prenorm.py`（`get_config` `:13`、kernel `:61`、API `:363`，唯一 TileLang） |
| 最强理由 | 输出是"逐元素非线性后的 head 加权归约"且每行有效区间不同，epilogue 模型表达不了 | 分段信息（各 expert 收多少 token）在 kernel 运行时才由通信产生，五段衔接需要 kernel 内同步 | `sqr_sum` 是输入侧逐行统计量，epilogue 够不着；不融合要多读一遍 A |
| 性能画像 | FIX-pipe bound（README.md:151，FIX 99%） | 小 T 同步受限 / 大 T 计算受限（846 TFLOPS） | 访存受限（"nearly saturates the HBM write bandwidth"，README.md:177） |

选择建议（第 48 天 3.6）：想深入推理栈选 MQA（页表/稀疏选块是通用词汇）；想深入分布式系统选 MegaMoE（任务重放、进度计数、计算通信重叠）；想深入 DSL 视角选 HC Prenorm（TileLang 与手写 C++ 的对偶）。讲深的标准是四问：输入输出与核心计算、调度与寻址特殊之处、为什么普通 GEMM 接口不够、性能瓶颈在哪——四问各自的参考答案都在第 48 天复盘 3.3～3.5 节。

### 3.7 板块六：一次性能分析或综合实践

**三公式**（自查能力⑤的工具箱）：

```text
FLOPs = 2·M·N·K                    （tests/common.py:233）
bytes = Σ numel × element_size      （count_bytes，tests/utils.py:95-103，每张量只计一次）
TFLOPS = FLOPs / dur_ns / 1000      （KernelProfile.tflops，bench.py:51-52）
GB/s   = bytes / dur_ns             （KernelProfile.gbps，bench.py:54-55）
AI     = FLOPs / bytes              → 与机器平衡点（峰值TFLOPS/峰值带宽）比较定方向
```

**测量纪律**：`bench_msprof` 默认 `num_warmups=10, num_tests=30, flush_l2=True`（`bench.py:288-290`）——每次迭代前清 2×L2 缓冲（`:263-268`）强制冷缓存，耗时取 30 次设备侧 PMU 均值；JIT 编译在测量前由 meta 并行预编译完成，不入账。

**一个可整段搬进分享的手算算例**（第 50 天 3.2 的复核过版本）——HC Prenorm `8192×24×28672`（`README.md:184`）：

```text
FLOPs = 2×8192×24×28672 ≈ 11.27 GFLOP
bytes = 8192×28672×2 (a,bf16) + 24×28672×4 (b,fp32)
      + 8192×24×4 (d,fp32) + 8192×4 (sqr_sum) ≈ 473.3 MB
AI    ≈ 23.8 FLOP/B ≪ BF16 平衡点估算 ≈123 → 访存受限
按 136.7 us 复核：TFLOPS ≈ 82.5 ✓（README：82）、GB/s ≈ 3462 ✓（README：3463）
```

对照组：Dense BF16 `4096×7168×16384` 的 AI ≈ 2249 ≫ 123 → 计算受限，431/432 = 99.8%（`README.md:113`）。**判别流程**：① 算 AI 定方向 → ② 看流水线利用率（`KernelProfile` 的 `aic_mad`/`aic_mte2`/`aic_fixpipe` 字段，`bench.py:28-45`；`print_perf` 直接打 mad%/mte2%，`tests/utils.py:106-115`）→ ③ 都不满查并行度（Tile 数 vs 32 核）。

**综合实践**（第 53～54 天）：四个项目 A（strided 测试接线）/ B（启发式门控）/ C（ReLU epilogue）/ D（BF16 源码讲解），验证遵循"正确性回归 → 覆盖三问法 → 同口径性能对比 → 失败如实记录"；产出三件套 = 修改说明 + 测试结果 + 性能对比。分享里最有说服力的一段通常是：**一个假设 → 一次单变量改动 → 一组同口径数据 → 证实/证伪结论**——证伪同样是有效产出，它修正的是对机制的理解。

### 3.8 差异速查：低精度与 Grouped GEMM（自查能力④）

| 维度 | BF16（基线） | 低精度 FP8/FP4 | M-Grouped | K-Grouped |
| :--- | :--- | :--- | :--- | :--- |
| 输入语义 | 数值本身 | **量化码 + SF**：`D = (A⊙sfA)@(B⊙sfB)ᵀ`，SF 按 recipe 分块共享 | A 平坦 `[M,K]`（K-major）、B `[G,N,K]`、`grouped_layout` psum 前缀和 | A `[K,M]`/B `[K,N]`（MN-major）、D `[G,M,N]`、每组 K 段 |
| Host 侧新增工作 | — | `transform_sf_into_required_layout` 规整成 **UE8M0 打包 int16、MN-major**（`gemm.hpp:128-154` 调用，实现在 `csrc/apis/layout.hpp` 与设备侧 `transform_sf.hpp`） | psum 检查、`expected_m = m/g`（`gemm.hpp:158-159`） | `expected_k` 三副面孔（`heuristics.hpp:597-599`） |
| Kernel 侧差异 | — | SF 独立 L1 缓冲与独立事件区间（`fp8_gemm.hpp:82-83`）；**反量化在 Cube MAD 内部**（`asc_mmad_mx`，`copy.hpp:370-381`），**FP32 累加在 L0C**；K 尾 `k64 = aligned(actual_k, 64)` + `fill_tail_block` 显式补零（`fp8_gemm.hpp:167-170`）；FP8×FP4 多一条 AIV0 解码流水（`fp8_dequant_gemm.hpp:10-93`） | 调度器分组游标（`scheduler.hpp:144-162`）；**BLOCK_M 钉死 256**（`heuristics.hpp:106`）；组边界按 256 对齐 | 共享 M×N 网格 + 逐组 `current_shape_k`（`scheduler.hpp:163-177`）；空组 `aiv_zero_empty_groups` 写零（`bf16_gemm.hpp:175-176`）；**支持 accumulate**（原子加） |
| 派发 | — | `dispatch_fp8_fp4_gemm` 三分（`gemm.hpp:22-33`）：fp8×fp8 与 fp4×fp4 共用 `fp8_gemm_impl` 模板、fp8×fp4 独立 dequant kernel | — | sfa/sfb 走 `transform_k_grouped_sf_into_required_layout` |
| 允许的 epilogue | acc/bias/alpha 均可（Normal） | 同左（Normal） | **无 bias/acc/alpha**（只对 Normal 开放） | **支持 acc**；bias/alpha 不支持 |

一句话总结：**低精度改变的是"数值语义"（码 + SF、反量化位置、SF 布局转换），Grouped 改变的是"调度与寻址"（分组元数据、组游标、输出形态）；两者正交，可以叠加成 `m_grouped_fp8_fp4_gemm_nt_contiguous`**。

### 3.9 易错点盘点（八周高频错误）

1. **逻辑链 ≠ 调用栈**：⑤⑥⑦ 是 `launch_bf16_gemm` 一个函数体内的三条语句（`csrc/jit_kernels/bf16_gemm.hpp:17/71/73`）。
2. **major 由 stride 推断**，不由用户声明（`csrc/utils/layout.hpp:44-47`）；binding 的转置是零拷贝逻辑视图。
3. **`SHAPE_* = 0` 是"运行期取值"哨兵**（`bf16_gemm.hpp:47-49`），不是维度为 0。
4. **`is_first` 在子块循环内判定**：每个输出子块的 K 遍历各自首条清零，`k_blk_idx==0 && k_mad_idx==0` 才 `clear_C`（`:134`）。
5. **`aic_wait_free` 只在 k_blk 0、`aic_store` 只在最后一个 k_blk**（`:129/:155`）；`s_epi` 每 k_blk 开头重置为 `epi_base`（`:122`）。
6. **BF16 无 `fill_tail_block`**：K 尾靠三级 `min` 截断 + `as_mad_aligned` 对齐 + ND2NZ 隐式零填充；显式补零（`static_assert(sizeof(T)==1)`）只属于 fp8/fp4。
7. **"测试通过"≠"覆盖了目标场景"**：要过三问法（跑到了吗 / 数据有属性吗 / 出错会发现吗，第 54 天）。
8. **性能对比必须同口径**：冷 L2、30 次迭代、设备侧计时、配置行同记；改了 kernel 行为才与基线对照，测试基建改动是新数据采集。
9. **MQA 的 "K" 不是循环维**：head_dim 是收缩维且一次装下，`kSplitKV=512` 切的是 N（KV token）方向。
10. **HC 的 K-split 原子加与 `kDirectStore` 的原子加不是一回事**：前者是多核共写归约，后者是单核内 L0C→GM 直写旁路。

## 四、源码导读

复盘日的读码目标是"把六板块的结论钉回源码"。按下表抽查（全部文件此前已通读过，只重读标注片段），每站 2～3 分钟：

| # | 文件 | 关键符号 | 复盘核对点（所属板块） |
| :---: | :--- | :--- | :--- |
| 1 | `deep_gemm/__init__.py` | re-export `:26-81`、`_C.init` `:159` | 板块一：算子版图与门面 |
| 2 | `csrc/python_api.cpp` | `PYBIND11_MODULE` `:30-40`、stream `:21` | 板块一/二：唯一翻译单元与八组注册 |
| 3 | `csrc/apis/gemm.hpp` | `get_gemm_desc` `:60-197`、binding `:425-448`、注册 `:632-637`、断言群 `:106-123` | 板块二：③的七阶段与 ② 的归一 |
| 4 | `csrc/jit_kernels/bf16_gemm.hpp` | `launch_bf16_gemm` `:16-78` | 板块二：⑤⑥⑦ 三步与 9 个 launch 参数 |
| 5 | `csrc/jit_kernels/heuristics.hpp` | `select_gemm_config` `:592-755`、打印块 `:749-753` | 板块二/六：配置指纹与硬约束（m-grouped BLOCK_M=256 `:106`） |
| 6 | `csrc/jit/jit.hpp` | `jit` `:16`、Runtime `:34-49`、`kMkAlignment` `:59` | 板块二：缓存与 dry-run 的底座 |
| 7 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | `bf16_gemm_impl` `:27-193` | 板块三：AIC 主循环 `:89-172`、AIV 分支 `:173-192` |
| 8 | `deep_gemm/include/deep_gemm/scheduler.hpp` | `Scheduler` `:49-`、`get_next_block` `:141`、M 分支 `:144-162`、K 分支 `:163-177`、psum 注释 `:17-23` | 板块三/差异表：调度与分组 |
| 9 | `deep_gemm/include/deep_gemm/epilogue.hpp` | `aic_wait_free` `:37`、`aic_store` `:42-47`、`aiv_drain` `:49-87`、`aiv_initialize` `:89-97`、`aiv_zero_empty_groups` `:109-141` | 板块三：借还协议；差异表：空组清零 |
| 10 | `deep_gemm/include/deep_gemm/ascend/copy.hpp` | `copy_gm_to_l1` `:102`、`copy_l1_to_l0a/b` `:134/:148`、`_mx` 变体 `:162/:216`、`mad` `:371` | 板块三/差异表：分形搬运与 `asc_mmad_mx` |
| 11 | `deep_gemm/include/deep_gemm/ascend/sync.hpp` | `set_intra_block` `:29`、`wait_intra_block` `:33`、`set_flags` `:64` | 板块三：事件方向与预置 |
| 12 | `tests/common.py` + `tests/utils.py` + `tests/generators.py` | `run_gemm_test` `:131`、`gemm_dg` `:49`、`gemm_aclnn` `:87`；`calc_diff` `:36`、`assert_equal` `:46`、`count_bytes` `:95`、`print_perf` `:106`；`reset_seed` `:9-12` | 板块四：五类断言与三个护栏 |
| 13 | `deep_gemm/testing/bench.py` + `par_compile.py` + `disable_fallback.py` | `KernelProfile` `:28-71`、`bench_msprof` `:284-`、L2 flush `:263-268` | 板块六：口径与设备侧计时 |
| 14 | `deep_gemm/include/deep_gemm/fp8_gemm.hpp` + `fp8_dequant_gemm.hpp` + `transform_sf.hpp` | `gm_sfa/sfb` `:65-66`、SF 缓冲 `:82-83`、`fill_tail_block` 调用 `:167-170`；dequant `:10-93`；`transform_sf` `:455` | 差异表：SF 布局与反量化位置 |
| 15 | `mqa_logits.hpp` / `mega_moe.hpp` / `hc_prenorm.py` | core `:195` / `mega_moe_impl` `:1127` / `get_config` `:13` | 板块五：选定算子的四问锚点 |

**重点重读三段**：

- **`bf16_gemm.hpp:89-172`（AIC 主循环）**：闭卷讲 Tile 生命周期前的最后校准——五个环的推进语句（`:112/:124/:133/:122/:164/:169`）、四组 wait/notify 的方向、复用条件（`:142/:144`）。读完自问：k_blk 2 的 `wait(MTE1→MTE2, 0)` 消费的是哪一次 notify？
- **`heuristics.hpp:749-753`（打印块）+ 一行真实配置**：5 行读完——整行去重、`std::puts`、dry-run 也执行。这是"配置指纹"与分享中"我看过真实配置"的证据链。
- **`tests/common.py:180-236`（运行段）**：对照板块四的五类断言逐行编号，确认每类断言的调用位置与条件分支。

## 五、综合练习

### A 组：有 Ascend 环境

1. **跑一个最小闭环并留档**（约 15 分钟）：

   ```bash
   cd DeepGEMM-Ascend && ./develop.sh   # 若尚未构建
   python tests/test_bf16_gemm.py normal --skip-prof     # 正确性
   DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee day55_configs.log
   ```

   从日志抄一行 `GemmDesc(...) GemmConfig(...)`，给每个字段标注来源层（②③⑤）——这行直接进分享的"调用链"板块当实物证据。

2. **去掉 `--skip-prof` 重跑一个用例**（约 10 分钟）：把 perf 行（us / TFLOPS / GB/s / mad% / mte2%）与 3.7 的公式手工复核一遍，任选一行与 README 对数；把"AI → 受限类型 → 利用率字段"的推理写进分享的"性能分析"板块。

3. **试讲并计时**（约 25 分钟）：按第六节大纲对空计时讲 15 分钟；若第 53～54 天有未完成项，可用今天的运行记录补"综合实践"板块的最后一块证据。

### B 组：无 Ascend 环境（静态任务，全部手工可完成）

1. **闭卷八层链**（约 10 分钟）：合上材料画出 3.3 的调用链，标注每层文件与关键行号、③ 的输入输出、④ 内含 ⑤⑥⑦；对照订正，≤3 处为掌握。

2. **闭卷讲 Tile 生命周期**（约 15 分钟）：对 `M=256、N=128、K=512`（BLOCK/MAD = 128/64、L1/L0 双缓冲，第 27 天的设定）复述五段流程；必答三问：k_blk 2 的 L1 空闲事件由谁发？（k_blk 0 末尾 `:167`）每 k_blk A 搬 4 次 B 搬 8 次为什么？（`:142/:144` 的复用条件）4 次交付各路由到哪个 AIV 的哪格 UB？（`s_epi` 的 round-robin）

3. **失败分诊五连**（约 10 分钟，第 13 天练习 2 的全周扩展版）——写出每例的层（编译期/Host/结果层/环境/参考）与一句话依据：

   ```text
   (a) Compiling 进度条阶段报 "L0C overflow"
   (b) RuntimeError: Bias must be contiguous
   (c) dg: (rows slice(0, 128, None), diff=2.31e-01) 与 aclnn: 同一用例同样失败
   (d) RuntimeError: operator 'aten::mm' has no NPU kernel and would fall back to the CPU
   (e) 仅 dg: diff=3.2e-6 失败（threshold=1e-5），同一用例 aclnn 通过
   ```

   参考答案：(a) 编译期——设备 `static_assert`（L0C 容量超限，`bf16_gemm.hpp:69`），在 Compiling 进度条阶段爆发；(b) Host 侧——`DJ_HOST_ASSERT` 文本原样抛出（`csrc/apis/gemm.hpp:114`）；(c) 参考层——dg 与 aclnn 两个独立实现同时失败，优先怀疑参考结果或测试数据本身（第 31 天四层分诊）；(d) 环境与护栏——`disable_fallback` 注册的抛错 kernel 被触发（`deep_gemm/testing/disable_fallback.py:33-39`），说明某个参考计算用到了无 NPU 实现的算子；(e) Kernel 层——仅 dg 失败且 diff 超阈值而 aclnn 通过，指向本仓库 kernel 的数值/边界问题；注意 diff=3.2e-6 只比 1e-5 阈值高一个量级以内，应先复跑排除噪声、再按"特定 shape/major/trans_ab"缩小范围。

4. **差异表默写**（约 10 分钟）：不看 3.8，画 BF16/低精度/M-Grouped/K-Grouped 四列对比（行自选至少 5 个），对照补漏；再闭卷回答：为什么 SF 必须在进 Kernel 前被规整成 UE8M0 打包 MN-major？（硬件按指数加法应用 scale，`asc_mmad_mx` 的前提；GM 布局固定才能让 `gm_sfa` 的 `stride_outer` 简单）

5. **性能手算**（约 15 分钟）：对 FP8 推理 `128×7168×16384`（`README.md:123`：54.8 us、549 TFLOPS、2456 GB/s）手算 FLOPs = 2×128×7168×16384 ≈ 30.06 GFLOP，用延迟复核 TFLOPS ≈ 549 ✓；再用表内两列相除得 AI ≈ 223.6 FLOP/B（第 50 天 3.7 的方法），与 FP8 平衡点估算 ≈247 比较判定受限类型并给出调优方向（M=128 的 Tile 数少、并行度不足是两头都不满的根因）；再对 Dense BF16 `4096×7168×16384` 复算 AI ≈ 2249 并说明为何 mad% 应接近满。

6. **写总结初稿**（约 20 分钟，主任务）：按第六节模板写出 2～3 页总结，或列 15 分钟分享的逐段讲稿要点；每个板块至少留三个 `文件:行号` 锚点。

### 双组通用：遗留问题归档（约 10 分钟）

把八周笔记末尾的未解决问题合并成一张表，标注去向：

| 疑问（示例） | 去向（第 56 天起的四个方向） |
| :--- | :--- |
| `ENABLE_KEEP/UPDATE` 的寄存器级语义、M 与 FIX 队列间的次序保证 | Ascend 指令级优化（CANN 文档 + `asc_mmad`） |
| `ProgressStage` 计数的内存一致性、UBMEM 直访的权限模型 | MegaMoE 多卡通信 |
| FIRST/REUSE 快照的更多变体、页表批量搬运的调度 | MQA 稀疏寻址 |
| TileLang AutoSchedule 与手写循环的等价性、`T.Serial` 的 0/1-trip 技巧 | TileLang |
| 成本模型的具体公式（`get_gemm_cost` 内部） | 性能分析进阶（可回读 `heuristics.hpp`） |

## 六、当天产出

按计划，今天的产出是**一份 15 分钟分享或 2～3 页总结**（二选一或各一份精简版），建议结构：

```text
# DeepGEMM-Ascend 八周学习总结（第 55 天）

1. 仓库整体架构（一张五类代码地图 + 算子版图一句话）          ～0.5 页 / 2 分钟
2. Python 到 Device Kernel 的调用链（八层图 + GemmDesc/GemmConfig
   一句话 + 为什么 JIT 实例化 + 三层对策）                      ～0.5 页 / 3 分钟
3. BF16 GEMM 数据流与流水线（五段叙事 + 一个 Tile 的生命周期，
   含五个环与 AIC/AIV 借还）                                    ～1 页 / 4 分钟
4. 测试与正确性验证（两阶段 + 五类断言 + 误差口径 + 失败三层）   ～0.5 页 / 2 分钟
5. 一个扩展算子（选定算子的四问；另两个各一句"为什么不是普通 GEMM"）～0.5 页 / 2 分钟
6. 性能分析与综合实践（三公式 + 一个手算算例 + 判别流程；
   若做了第 53～54 天项目：假设-改动-数据-结论一段）            ～0.5 页 / 2 分钟
附录：自查五项的"能/勉强/不能"记录 + 遗留问题表（含去向）
```

写作自查两问：**不看笔记能否只凭这份总结重讲一遍？** 每条关键结论是否都有可以翻给质疑者看的 `文件:行号` 或数据？

## 七、通过标准

学习计划第 55 天的自查清单共五项能力，逐条自评"能 / 勉强 / 不能"：

| # | 检查点 | 验证方式 | 未达标的补救 |
| :---: | :--- | :--- | :--- |
| 1 | **独立追踪公开 API 到 Device Kernel** | 综合练习 A/B-1：闭卷八层链，标注层、文件、行号、③④ 的输入输出；`GemmDesc` vs `GemmConfig` 各一句话 | 回读第 13 天复盘 2.2～2.5 |
| 2 | **解释一个输出 Tile 的完整生命周期** | 综合练习 A/B-2：五段叙事 + 五个环 + 三问全对（事件由谁发 / 搬运为何不对称 / 交付路由） | 回读第 27 天复盘 3.2～3.7 |
| 3 | **判断测试失败的大致层级** | 综合练习 A/B-3：五连分诊至少四对；能说出每层的典型报错文本 | 回读第 31 天分诊表 + 第 54 天 3.2 |
| 4 | **说明低精度与 Grouped GEMM 的主要差异** | 综合练习 B-4：四列差异表默写无重大遗漏；能答"SF 为何必须预规整" | 回读第 32/36/37 天 + 第 34 天复盘 |
| 5 | **对一个 shape 做有依据的性能分析** | 综合练习 A-2/B-5：手算三指标与 README 对数、AI 定方向、利用率字段定位瓶颈、给一句调优方向 | 回读第 50 天 3.2～3.4 |

**评判**：五项全部"能"——达标，八周计划完成，明天进入休息与后续规划；某项"勉强"——按补救列回读后对该项重做闭卷；某项"不能"——把它写进遗留问题表作为后续深入方向的第一个条目，不视为计划失败（计划明言"不一定要逐行读完所有文件"）。**核心完成标准**（学习计划末节）的优先级：BF16 GEMM 主线 > JIT 调用链 > 测试体系 > 一个扩展方向 > 独立实践能力——前两项是地基，务必"能"。

## 八、自测题

先闭卷作答，再对照参考答案与源码订正。

1. 用一句话分别定位 `GemmDesc` 和 `GemmConfig`，并说出二者的产生函数与文件。
2. 为什么 `bf16_gemm_impl` 必须做成模板并由 JIT 现场实例化？编译期常量换来了哪两个设备侧好处？
3. 一个输出 Tile 的 K 遍历中，"首条清零、中间 KEEP、末条 UPDATE"分别由哪两个标志控制？它们在哪一行定义、哪一行消费？
4. L1 只有 2 级缓冲却有 4 个 k_blk，为什么不死锁？"事件号 = 缓冲编号"如何支撑任意长的 K？
5. `s_epi` 在每个 k_blk 开头和结尾各发生什么？`epi_base` 解决什么问题？
6. 测试的 `d` 预填策略（randn / NaN）各服务于什么缺陷检测？`disable_cpu_fallback` 防住什么？
7. 三层失败各在哪里暴露？各举一个真实报错文本。
8. BF16 的 K 尾和 FP8 的 K 尾处理差在哪？为什么 FP8 需要 `fill_tail_block` 而 BF16 不需要？
9. M-Grouped 的 `grouped_layout` 存的是什么？组边界按多少对齐？为什么 BLOCK_M 被钉死为 256？
10. SF 进入 Kernel 前被规整成什么布局？反量化和高精度累加分别发生在哪里？
11. K-Grouped 为什么支持 accumulate 而 M-Grouped 不支持？空组的输出由谁、在什么条件下写零？
12. 给定 BF16 shape `M=4096, N=7168, K=16384`：写出 FLOPs、`count_bytes` 口径的 bytes、AI，判定受限类型并引用 README 数据佐证。
13. （开放）六板块中你认为哪一块是"理解本仓库的钥匙"？哪一块对你当前工作最有用？为什么？

<details>
<summary>参考答案（先自己做再展开）</summary>

1. `GemmDesc` 是 "Problem descriptor: what to compute"（算什么），由 `get_gemm_desc`（`csrc/apis/gemm.hpp:60-197`）从用户张量提取校验；`GemmConfig` 是 "Execution configuration: how to compute"（怎么算），由 `select_gemm_config`（`csrc/jit_kernels/heuristics.hpp:592-755`）启发式搜索；二者都定义在 `csrc/jit_kernels/gemm_config.hpp`。
2. 26 参函数模板不实例化就没有机器码，组合空间近乎无限只能用到哪个现场实例化哪个（JIT）。编译期常量让 kernel 用 `constexpr` 规划片上地址（`bf16_gemm.hpp:59-64`）并用 `static_assert` 在编译期暴露容量错误（`:66-70`）。
3. `is_first = (k_blk_idx==0 && k_mad_idx==0)`、`is_last = (k_blk_idx==num_k_blocks-1 && k_mad_idx+MAD_K>=actual_k)`，定义在 `:134-135`，消费在 `:149-151`（`clear_C=is_first`，`is_last ? ENABLE_UPDATE : ENABLE_KEEP`）。
4. 缓冲归还事件跨 k_blk 流转：k_blk 0/1 的首次 wait 消费启动预置（`:83`），k_blk 2/3 消费 k_blk 0/1 末尾 `:167` 的归还。事件号 = stage 号，2 个号支撑任意长循环。
5. 开头 `s_epi = epi_base`（`:122`），每个子块循环体末尾 +1 并模 `epilogue.kNumStages` 回绕（`:164`）；`epi_base = s_epi`（`:169`）让下一个 Tile 的交付编号跨块续接，AIC 与 AIV 无需通信即可锁步。
6. randn 垃圾（Normal，`common.py:190`）让"漏写任何输出区域"的 diff 爆炸；NaN（k-grouped 非累加）额外让"空组该写零却没写"显形（NaN 参与任何比较都失败）。`disable_cpu_fallback`（`common.py:142`）给无 NPU 实现的算子注册抛错 kernel，防止参考计算静默回退 CPU 而"假通过"。
7. 编译期：JIT 编译阶段，如 "L1 overflow"（`bf16_gemm.hpp:66-70`）；Host 侧：launch 前 Python `RuntimeError`，如 "Bias must be contiguous"（`gemm.hpp:114`）；结果层：`assert_equal` 报错带行区间与 diff（`tests/utils.py:55-58`）。
8. BF16 靠三级 `min` 截断 + `as_mad_aligned` 对齐 + ND2NZ 隐式零填充（补零列乘出 0 不影响累加）；FP8 的 SF 粒度是 64（`MX_SF_DIVISOR`），MAD 的 K 上界抬到 `k64 = aligned(actual_k, 64)`，搬运引擎只补到 32 字节（= 32 个 fp8），缺口由 `fill_tail_block` 显式补零（`fp8_gemm.hpp:167-170`，`static_assert(sizeof(T)==1)` 所以 BF16 用不了）。
9. psum 前缀和：`grouped_layout[i]` 是前 i+1 组实际行数的累积结尾（`scheduler.hpp:17-23`）；组边界按 256 对齐（组 g 有效行 `[align(psum[g-1],256), psum[g])`）；BLOCK_M=256 与 psum 布局的组边界对齐（`heuristics.hpp:106` 的硬约束），也是 `static_assert(kAlignment == BLOCK_M)` 成立的原因。
10. UE8M0 打包的 int16、MN-major（每 pair 覆盖 64 个 K 元素）；反量化发生在 Cube 的 MAD 内部（`asc_mmad_mx`，`copy.hpp:370-381`，scale 乘法被设计成指数加法），FP32 累加发生在 L0C。
11. K-Grouped 每组输出是干净的 `[M,N]`，`D[g] = A_g@B_g + C[g]` 语义清晰（空组保持 `D[g]=C[g]` 恰好正确）；M-Grouped 的输出含 padding 行，与 atomic add 语义冲突。空组由 AIV 的 `aiv_zero_empty_groups` 清零（`bf16_gemm.hpp:175-176` → `epilogue.hpp:109-141`），且只在 `!kWithAccumulation` 时执行。
12. FLOPs = 2×4096×7168×16384 ≈ 962.07 GFLOP；bytes = (4096×16384 + 16384×7168 + 4096×7168)×2 = 427,819,008 B；AI ≈ 2249 ≫ BF16 平衡点 ≈123 → 计算受限；README.md:113：2229.9 us / 431 TFLOPS / 432 上限 = 99.8%。
13. 开放题。参考：钥匙通常是"调用链 + Tile 生命周期"（它解释了其余一切代码为什么长这样）；性能板块对工程工作最有用（方法论可迁移到任何 kernel）。
</details>

---

至此八周计划收官。明天（第 56 天）休息与后续规划：整理笔记和代码、保留仍未解决的问题，再按兴趣从四个方向挑一个深入——**Ascend 指令级优化**（`ENABLE_KEEP/UPDATE`、FIX/M 队列次序、L2 控制模式）、**MegaMoE 多卡通信**（HCCL UBMEM 直访、进度计数一致性）、**MQA 稀疏寻址**（页表、选块、FIRST/REUSE 快照）或 **TileLang**（DSL 与手写 kernel 的对偶）。今天自查表上每一个"勉强"，都是那张后续路线图上现成的第一站。
