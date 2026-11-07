# 第 8 周 · 第 2 天（总第 51 天）：观察 JIT 配置与 Profiling 输出

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 8 周"性能分析与综合实践"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

昨天（第 50 天）建立了性能分析的"度量衡"：TFLOPS、有效带宽、算术强度、受限类型，以及 `bench_msprof` 的设备侧计时方法。今天打开仓库自带的两个**观察窗口**，把抽象的方法变成可见的数据：`DG_PRINT_CONFIGS` 让我们看到启发式为每个 shape **实际选中**的 `GemmConfig`（BLOCK/MAD/stages/direct_store/resident/launch 全套参数），`DG_JIT_PRINT_COMPILER_COMMAND` 让我们看到 JIT 真正执行的编译命令；再配合去掉 `--skip-prof` 后的 perf 输出，把 **shape → 配置 → 耗时 → TFLOPS → 带宽** 汇成同一张记录表。这张表就是**基线性能记录**——第 52 天的每个单变量实验都要回到它来做对照。没有 Ascend 环境时，今天同样有完整的替代任务：用 `get_hw_desc()` 的硬件模型手推 README 里的峰值算力、手工验证配置搜索的硬约束、并把 README 已发表的性能数据整理成同格式的基线表。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 正确设置并解释两个观察开关的语义与生效时机：`DG_PRINT_CONFIGS`（打印每个 shape 选中的配置，`README.md:89`）与 `DG_JIT_PRINT_COMPILER_COMMAND`（打印编译器与反汇编命令，`README.md:93`），理解超集开关 `DG_JIT_DEBUG`（`README.md:88`）、`DG_JIT_*` 回退到 `DJ_JIT_*` 且在 JIT 运行时首次创建时快照（`README.md:85`）这两条规则，以及为什么变量必须在**进程启动前**导出；
2. 逐字段读懂 `select_gemm_config` 打印的一行 `GemmDesc(...) GemmConfig(...)`：两个 formatter 分别定义在 `csrc/jit_kernels/gemm_config.hpp:145-161` 与 `:163-172`，能说出 block/mad/stages/l2_ctrl/direct_store/resident/launch_options 每个字段的含义；
3. 说明打印出的每个配置字段如何被写进 JIT 生成的模板实例化代码（`csrc/jit_kernels/bf16_gemm.hpp:18-66` 的 `std::format` 代码生成），把今天的观察与第 12 天的 JIT 调用链重新接上；
4. 用源码里的硬约束手工验证一个配置的合法性：`block_cap`（`heuristics.hpp:604`）、BLOCK_M×BLOCK_N 的 L0C 容量约束（`:100-102`）、BLOCK_K 的上下界（`:620-621`）、L1 stage 数公式（`:168-174`）、`num_l0_stages` 恒为 2（`:111-112`）；
5. 用 `get_hw_desc()`（`heuristics.hpp:52-71`）的数字手推 Ascend 950 的峰值算力：BF16 432 / FP8 865 / FP4 1730 TFLOPS，并与 README Hardware limit 列（`README.md:108-113`）对数；
6. 有环境时：运行一个自写的小型 GEMM 脚本与官方测试，采集配置行与 perf 行，交叉验证 README 数据，完成一张包含 shape、配置、耗时、TFLOPS、带宽、mad%/mte2% 的基线记录表；
7. 无环境时：从 README 已有性能数据构建同格式基线表，并完成配置约束的手工演算。

## 二、预备知识

- **第 12 天**：JIT 配置、编译与启动——`select_gemm_config()` 选出 `GemmConfig`，模板代码生成 → `jit->compile()` → `jit->launch()`。今天观察的正是这条链路上"配置"环节的真实输出。
- **第 15～16 天**：Kernel 模板参数分类与片上内存规划。`GemmConfig` 的 BLOCK/MAD/stages 字段就是当时那些模板参数的来源；L1/L0A/L0B/L0C/UB 容量常量在 `deep_gemm/include/deep_gemm/common.hpp:54-58`。
- **第 26 天**：`kDirectStore`、`kResidentA/B` 特殊执行路径——它们在配置行里对应 `direct_store=` 与 `resident=` 字段。
- **第 50 天**：`bench_msprof` / `KernelProfile` 的测量流程、TFLOPS/带宽/AI 公式、冷缓存口径、`print_perf` 的输出格式。今天的 perf 观察完全建立在这些内容之上。
- **环境说明**：第五节 A 组任务需要 Ascend 环境（CANN 9.20 + torch_npu，且已按第 9 天完成 `./develop.sh` 构建）；无环境时完成 B 组静态任务（峰值推导、约束演算、README 基线表）即可达到当天目标。

## 三、详细讲解

### 3.1 两个观察开关：控制什么、何时生效

仓库在 README 的 Environment Variables 一节（`README.md:81-98`）登记了所有开关。今天用到三个，按"作用范围从小到大"排列：

| 开关 | 作用 | 实现位置 | 默认 |
|---|---|---|---|
| `DG_PRINT_CONFIGS` | 打印每个 shape 选中的配置 | 本仓库 `csrc/jit_kernels/heuristics.hpp:749-753` | 0 |
| `DG_JIT_PRINT_COMPILER_COMMAND` | 打印编译器与反汇编命令 | DeepJIT 运行时（`third-party/deep-jit` 子模块），本仓库 README 记录其行为（`README.md:93`） | 0 |
| `DG_JIT_DEBUG` | JIT 调试功能超集：包含上两者再加 kernel 加载耗时报告（`DG_JIT_PRINT_LOAD_TIME` 的内容） | DeepJIT + `heuristics.hpp:749` | 0 |

配置打印的本仓库侧实现只有 5 行，值得整段读（`heuristics.hpp:749-753`，位于 `select_gemm_config` 末尾）：

```cpp
if (deep_jit::get_env<int>("DG_JIT_DEBUG") || deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
    static std::unordered_set<std::string> printed;
    const auto line = std::format("{} {}", d, config);
    if (printed.insert(line).second) std::puts(line.c_str());
}
```

三个细节直接决定你看到什么：

1. **打印的是 `GemmDesc` + `GemmConfig` 拼成的一行**——既是"问题描述"（shape、布局、累加、epilogue）也是"执行方案"（tiling、流水线深度、L2 提示、launch 块数）。
2. **按整行字符串去重**：`static std::unordered_set` 是进程级单例，同一个 (desc, config) 组合整个进程只打印一次。所以重复调用同一个 GEMM 不会刷屏；反过来，**若两次调用打印了不同配置，说明 desc 有差异**（布局、bias、alpha、compiled_dims 任一不同都会改 desc）。
3. **它在 `select_gemm_config` 里，而该函数在 dry-run 阶段也会被调用**——`launch_bf16_gemm` 先选配置、再判断 dry-run（`csrc/jit_kernels/bf16_gemm.hpp:16-17` vs `:67-70`），而测试的预编译阶段 `par_compile` 正是在 dry-run 下跑的（`deep_gemm/testing/par_compile.py:6-13`）。因此**配置行会在测试开头的 `Compiling` 进度条阶段就成批出现**，perf 阶段不再重复打印。

**生效时机是今天最容易踩的坑。** README 明确写道："Each `DG_JIT_*` variable falls back to the corresponding global `DJ_JIT_*` variable when unset, and is snapshotted when the JIT runtime is first created"（`README.md:85`）。也就是说 `DG_JIT_PRINT_COMPILER_COMMAND` 这类 `DG_JIT_*` 变量在 JIT 运行时第一次创建时被快照，**进程内再改 `os.environ` 不会生效**——必须在启动 Python 之前 `export`：

```bash
export DG_PRINT_CONFIGS=1
export DG_JIT_PRINT_COMPILER_COMMAND=1
python my_baseline.py        # 顺序不能反
```

另有两个相关事实：

- **编译缓存**：`DG_JIT_CACHE_DIR` 指定编译产物缓存目录，默认 `$HOME/.dj`（`README.md:91`）。想观察"编译命令"就需要真正发生编译（缓存未命中）；把它指到一个空目录可以强制重编译：`export DG_JIT_CACHE_DIR=/tmp/dj_fresh`。CANN 9.20 提供的编译工具是 `bin/bisheng` 与 `bin/ld.lld`（`README.md:24`），打印出的命令就是在调它们。
- **构建期快照陷阱**：`setup.py` 会把构建时设置过的部分 `DG_JIT_*` 变量写进包里的 `envs.py`，导入时自动恢复（`setup.py:17-25`、`setup.py:131-140`、`deep_gemm/__init__.py:7-13`）。如果哪天你用 `pip install .` 时终端里还挂着 `DG_JIT_PRINT_COMPILER_COMMAND=1`，装出来的包会**永远**打印编译命令。构建前记得 unset。

最后一个实用提示：配置行走的是 C++ 的 `std::puts`（C stdout），perf 行走的是 Python 的 `print`，两者缓冲不同，重定向到文件时顺序可能交错。建议直接看终端输出，或用 `stdbuf -oL python ...` 统一行缓冲。

### 3.2 配置行解剖：GemmDesc 与 GemmConfig

两个 formatter 的格式串是配置行的"语法"，先看 `GemmDesc`（`gemm_config.hpp:146-158`）：

```text
GemmDesc(m={}, n={}, k={}, cd_dtype={}, major_a={}, major_b={}, acc={}, epilogue_type={}, compiled_dims='{}', gemm_type={}
```

（M-Grouped 追加 `num_groups/expected_m`，K-Grouped 追加 `num_groups/expected_k`，Batched 追加 `batch`，`:152-157`，然后补一个右括号。）再看 `GemmConfig`（`gemm_config.hpp:165-170`）：

```text
GemmConfig(block={}x{}x{}, mad={}x{}x{}, l1_stages={}, l0_stages={}, epi_stages={}, l1_sf_stages={}, sf_k_blocks={}, l2_ctrl={}/{}/{}, direct_store={}, resident={}/{}/{}, dual_aiv_dequant={}, launch_options=LaunchOptions(num_blocks={}))
```

合起来，一次 `deep_gemm.bf16_gemm_nt(a, b, d)`（`a` 为 `[m,k]`、`b` 为 `[n,k]` 的连续张量，输出 bf16、无 bias/alpha）的打印行形如（`…` 为运行时才能确定的值）：

```text
GemmDesc(m=4096, n=7168, k=2048, cd_dtype=bfloat16_t, major_a=K, major_b=K, acc=false, epilogue_type=epilogue::operators::Identity, compiled_dims='nk', gemm_type=Normal) GemmConfig(block=…x…x…, mad=…x…x…, l1_stages=…, l0_stages=…, epi_stages=…, l1_sf_stages=…, sf_k_blocks=…, l2_ctrl=…/…/…, direct_store=…, resident=…/…/…, dual_aiv_dequant=…, launch_options=LaunchOptions(num_blocks=…))
```

这行的前半部分全部可以从调用推导：`bf16_gemm_nt` 的 `compiled_dims` 默认是 `"nk"`（`csrc/apis/gemm.hpp:632-637`，即 N、K 编译为常量、M 运行时传入）；连续的 `[m,k]`/`[n,k]` 张量使 `major_a=major_b=K`；不带 `c` 时 epilogue 被 `resolve_epilogue_class` 解析为 `Identity`（`epilogue_class.hpp:124-125`，其打印名见 `:29-34`）、`acc=false`（`apis/gemm.hpp:180`）。逐字段含义：

| 字段 | 含义 | 决定它的代码 |
|---|---|---|
| `m/n/k, cd_dtype, major_a/b, acc, epilogue_type, compiled_dims, gemm_type` | 问题本身：shape、输出类型、A/B 主方向、是否累加、epilogue 种类、哪些维度编译期常量、GEMM 类型 | `get_gemm_desc()`（`csrc/apis/gemm.hpp:60-197`），第 11 天已读过 |
| `block=MxNxK` | 输出 Tile 的 BLOCK_M/N 与 K 分块 BLOCK_K | `select_gemm_config` 的双层搜索（`heuristics.hpp:626-657`） |
| `mad=MxNxK` | 一次 MAD 指令的矩阵分块（MAD_M/N/K），第 22 天的两层 K 循环由此展开 | `make_gemm_candidate` 的 split 搜索（`:124-147`） |
| `l1_stages` | L1 缓冲阶段数（MTE2↔MTE1），第 16/23 天的多级流水深度 | L1 容量公式（`:168-174`） |
| `l0_stages` | L0 A/B 缓冲阶段数（MTE1↔Cube） | 候选生成时恒为 2（`:111-112`） |
| `epi_stages` | 每个 AIV 的 UB epilogue 阶段数（AIC→AIV） | `:186-194` |
| `l1_sf_stages / sf_k_blocks` | L1 SF 缓冲阶段数 / 每 SF 块覆盖的 K-block 数（BF16 恒为 0/0） | `:156-164`、`:177-183` |
| `l2_ctrl=A/B/store` | 三个 L2 cache 提示（枚举名如 `NORMAL_FIRST_VICTIM`、`NOTALLOC_KEEP`、`NOTALLOC_CLEAN`，`common.hpp:120-134`） | `:196-201`、`:660-667` |
| `direct_store` | 是否走 AIC 直接写回 GM 的路径（第 26 天） | `can_direct_store` + `refine_gemm_pipeline`（`:521-533`、`:535-543`） |
| `resident=A/B/SF` | A/B/SF 是否常驻 L1 复用（第 26 天） | `refine_gemm_pipeline`（`:544-565`） |
| `dual_aiv_dequant` | FP4 反量化是否用双 AIV 解码 | `:567-571`、`:727-737` |
| `launch_options.num_blocks` | kernel 启动的 block 数 = 使用的 AI Core 数 | 候选生成时直接取 `d.num_cores`（`:98`），而 desc 的 `num_cores` 来自 `runtime->get_num_sms()`（`csrc/apis/gemm.hpp:176`） |

最后一条值得展开：`num_blocks` 打印出来就是**这个 kernel 实际占用的 AI Core 数**，它来自 `get_num_sms()`（默认查设备，可被 `deep_gemm.set_num_sms()` 覆盖，`csrc/jit/jit.hpp:34-41`）。在记录表里核对 `num_blocks == deep_gemm.get_num_sms()` 是检验你理解 launch 链路的好办法，也是第 52 天"改变 AI Core 数量"实验的挂钩点。

**这行配置不只是日志，它就是 kernel 的"基因"。** `launch_bf16_gemm` 紧接着用 `std::format` 把这些值填进模板实例化代码（`bf16_gemm.hpp:18-42`）：`config.block_m/n/k`、`mad_m/n/k`、三个 stage 数、`num_blocks`、三个 L2 提示、`direct_store`、`resident_a/b` 逐个成为 `bf16_gemm_impl<...>` 的模板实参，再 `jit->compile()` 编译、`jit->launch()` 启动（`:71-77`）。所以"打印的配置"="编译进 kernel 的参数"，不存在打印与实际不一致的问题。

### 3.3 配置从哪里来：搜索空间与硬约束速览

第 12 天读过 `select_gemm_config` 的骨架，今天从"验证打印结果"的角度重看它的**硬约束**（全部在 `heuristics.hpp`）：

| 约束 | 公式（BF16，elem_bits=16，无 SF） | 源码 |
|---|---|---|
| BLOCK_M/N 上限 `block_cap` | 512（FP4 或 FP8×FP4 反量化路径为 256） | `:604` |
| BLOCK_K 上下界 | `min_block_k = 256×8/16 = 128`，`max_block_k = 256`（即 l1→l0 单周期搬运量决定） | `:620-621` |
| L0C 容量 | `BLOCK_M × BLOCK_N × 4B ≤ 256KB`，即 `BLOCK_M×BLOCK_N ≤ 65536` | `:100-102`，`L0CSizeBytes` 见 `common.hpp:57` |
| K 对齐 | `block_k % k_alignment == 0`，BF16 的 `k_alignment = 32×8/16 = 16` | `:108-109` |
| L0 阶段数 | `num_l0_stages = 2`（候选阶段恒定） | `:111-112` |
| L1 阶段数 | `num_l1_stages = min(8, (512KB − SF字节) / ab_bytes)`，`ab_bytes = (BLOCK_M+BLOCK_N) × BLOCK_K × 2B`，且 ≥ 2 否则候选非法 | `:170-174` |
| 搜索方向 | BLOCK_M 从大到小、BLOCK_N 从大到小（"大块先建立 cost 上界，同 cost 偏好小 M/N"，`:625-626` 注释），代价模型 `get_gemm_cost` 剪枝 | `:626-657` |

一个可以完全手工复核的算例（也是 B 组核心练习）：**BF16、block=512×128、block_k=128 的 L1 阶段数**。

```text
ab_bytes = (512 + 128) × 128 × 2 B = 163,840 B
num_l1_stages = min(8, 524,288 / 163,840) = min(8, 3) = 3      （整数除法）
验证：3 × 163,840 = 491,520 ≤ 524,288 ✓   4 × 163,840 = 655,360 > 512KB ✗
```

同一公式还能反推"为什么想用更宽的 block_k 会失败"：`block=512×128、block_k=256` 时 `ab_bytes = 640×256×2 = 327,680`，`524,288/327,680 = 1 < 2`，候选直接非法（`:173`）——这就是 `:648-651` 里"更宽 K 候选还需 `num_l1_stages ≥ 3` 才采纳"的物理原因。

选中主配置后还有两轮"精修"也会反映在打印行里：`refine_gemm_pipeline`（`:535-574`）在代价模型允许时尝试打开 `direct_store`（`can_direct_store`，`:521-533`，要求 Normal + Identity epilogue + 地址/stride 对齐等）和 `resident_a/b`（`:544-565`，要求 K 编译为常量且矩阵能装进 L1）；`:672-694` 再比较不同的 BLOCK_K。这些分支的输入条件（`compiled_dims` 里有没有 'k'、M/N 是否单 block）都能从 `GemmDesc` 打印字段直接读出——**看日志反推分支走向**是今天的隐藏练习。

### 3.4 硬件模型 HWDesc：用源码数字复算 README 峰值

启发式不查设备表，而是内置一份 Ascend 950 的硬件模型 `get_hw_desc()`（`heuristics.hpp:52-71`）。关键数字：`aic_clock_mhz=1650`、`bf16_flops_per_cycle=8192`（注释：4096 次乘加，每次记 2 FLOP）、`num_ai_cores=32`、`gm_bw_gbps=3500`、`l2_bw_gbps=4500`。

第 50 天我们只能从 README 读到硬件上限；今天可以从源码**推导**它：

```text
BF16 峰值 = 8192 FLOP/cycle/core × 1650 MHz × 32 cores = 432.5 TFLOPS   → README：432 ✓
FP8  峰值 = get_mad_flops_per_cycle(8) = 8192 × 16/8 = 16384 → 865.0 TFLOPS → README：865 ✓
FP4  峰值 = get_mad_flops_per_cycle(4) = 8192 × 16/4 = 32768 → 1730.1 TFLOPS → README：1730 ✓
```

（`get_mad_flops_per_cycle` 在 `heuristics.hpp:46-48`：按操作数位宽等比放大 BF16 的每周期 FLOP。）三个数与 README Dense GEMM 表的 Hardware limit 列（`README.md:108-113`）完全吻合——**README 的"硬件极限"就是这份 host 侧模型算出来的**。同时 `gm_bw_gbps=3500` 也印证了第 50 天用 HC Prenorm 数据反推的 ≈3.5 TB/s HBM 带宽基准。

顺带记住 `num_ai_cores=32` 与 `common.hpp:51` 的 `kNumAICores = 32`：配置行里 `num_blocks` 的默认值 32 即来源于此。

### 3.5 性能输出：print_perf 一行该记什么

去掉 `--skip-prof`（`tests/test_bf16_gemm.py:26`："Skip performance measurements and just test correctness"）后，`run_gemm_test` 在每个用例的正确性断言之后进入 perf 段（`tests/common.py:223-236`）：

```python
if not skip_prof and test.profile and k:
    trans_ab = test.get_available_trans_ab()[0]
    prof = bench_msprof(lambda: gemm_dg(test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab),
                        kernel_names='gemm_impl')
print_perf(m, n, k, f'layout={layout}, {out_opt}, acc={int(test.accumulate)}, {test.tag}',
           2.0 * m * n * k, count_bytes(a, b, c, d), prof)
```

`print_perf`（`tests/utils.py:106-115`）的输出格式：

```text
 > Perf (m=  4096, n=  7168, k= 16384, layout=NT, bf16, acc=0,   fwd): 2229.900 us |  431 TFLOPS |  192 GB/s | mad= 99.x% | mte2= 12.x%
      │shape 与口径标签                                    │          │            │             │            └── 两条关键流水线利用率
      │                                                    │          │            └── 有效带宽（count_bytes 口径）
      │                                                    │          └── TFLOPS（2·M·N·K / dur）
      │                                                    └── 设备侧时长（30 次迭代均值）
```

（上行格式按 `print_perf` 的格式串构造，数值取自 README Dense BF16 行 `README.md:113` 的期望值，mad/mte2 以实际运行为准。）一行同时给出第 50 天的三个指标中的两个和两条流水线利用率——**mad 与 mte2 的对比就是受限类型的现场判据**：mad 接近 100% → 计算受限；mte2 接近 100% → 访存受限；fixpipe 高 → 回写受限（print_perf 不显示 fixpipe，完整字段在 `KernelProfile.__repr__`，`bench.py:57-71`，可用 `print(prof)` 直接看）。

注意两点口径：

1. perf 段只用每个用例的**第一种** trans 布局（`get_available_trans_ab()[0]`），所以 54 个 shape 各产出一条 perf 行（`normal` 选择的用例数见 `tests/generators.py:544-576`：3 个前向 M × 9 组 (N,K) + 1 个反向 M × 9 × 3 种变体）；
2. 指标分母是 `dur_ns`（30 次迭代的 PMU 均值）、分子 `2.0*m*n*k` 与 `count_bytes` 都是**逻辑口径**（第 50 天 3.2 节）。

### 3.6 基线记录表：把观察变成第 52 天的"对照组"

今天的产出是一组基线性能记录。建议的表结构（列名即采集顺序）：

| # | shape (M×N×K) | dtype / 布局 | 选中配置（block / mad / l1_stages / direct_store / resident / num_blocks） | dur (us) | TFLOPS | GB/s | mad % | mte2 % | AI (FLOP/B) | 受限判断 |
|---|---|---|---|---|---|---|---|---|---|---|

三条纪律，保证第 52 天可比：

1. **口径写进表头**：设备型号、CANN 版本、`bench_msprof` 默认参数（冷 L2、30 次迭代、设备侧计时）、`deep_gemm.get_num_sms()` 的值。第 52 天改一个变量回来对照时，其他条件必须与今天逐项相同。
2. **AI 列当场算**：`AI = 2MNK / count_bytes 字节数`（或直接 `TFLOPS / GB/s × 1000`，两者同除 dur，比值与 dur 无关），配合 mad/mte2 给出受限判断——这一列是"提出性能假设"的原材料。
3. **记录"离峰距离"**：`TFLOPS / 硬件上限`（432/865/1730，见 3.4 节）。README Dense 表的 Utilization 列就是这么算的（98.3%～99.8%，`README.md:108-113`）；离峰远的行（如 FP8 推理 M=128 行约 63%）才是后续优化实验的候选对象。

没有环境的 B 组同样产出这张表，只是数据来源换成 README（见第五节 B 组第 3 步）。

## 四、源码导读

| # | 文件 | 关键符号 | 职责 |
|---|---|---|---|
| 1 | `csrc/jit_kernels/heuristics.hpp` | `select_gemm_config` `:592-755`、打印块 `:749-753`、`make_gemm_candidate` `:92-204`、`get_hw_desc` `:52-71`、`refine_gemm_pipeline` `:535-574`、`can_direct_store` `:521-533` | 今天的主战场：配置搜索、硬约束、打印 |
| 2 | `csrc/jit_kernels/gemm_config.hpp` | `GemmDesc` `:39-95`、`GemmConfig` `:97-121`、两个 formatter `:145-172`、`get_compiled_dim` `:16-21` | 配置行的数据结构与"语法" |
| 3 | `csrc/jit_kernels/bf16_gemm.hpp` | `launch_bf16_gemm` `:16-78`（代码生成 `:18-42`、dry-run `:67-70`、launch `:73-77`） | 配置 → 模板实例化 → 编译启动 |
| 4 | `csrc/jit_kernels/fp8_gemm.hpp` | `launch_fp8_gemm` `:11-79` | 同一 `select_gemm_config`，低精度多 SF 字段 |
| 5 | `csrc/apis/gemm.hpp` | `get_gemm_desc` `:60-197`（`num_cores` `:176`）、`bf16_gemm_nt` 绑定 `:632-637` | desc 字段来源与 `compiled_dims` 默认值 |
| 6 | `csrc/jit/jit.hpp` | `get_num_sms`/`set_num_sms` `:34-41`、`init_jit` `:78-92` | num_blocks 的上游；JIT 运行时创建（env 快照发生在此时） |
| 7 | `deep_gemm/include/deep_gemm/common.hpp` | 容量常量 `:51-65`、L2 模式枚举 `:118-135` | 约束公式里的全部常数 |
| 8 | `tests/common.py` / `tests/utils.py` / `deep_gemm/testing/bench.py` | perf 段 `common.py:223-236`、`print_perf` `utils.py:106-115`、`KernelProfile` `bench.py:27-71` | 性能输出口径（第 50 天已精读） |
| 9 | `README.md` | 环境变量 `:81-98`、Performance `:100-185` | 开关语义与权威性能数据 |
| 10 | `setup.py` / `deep_gemm/__init__.py` | `persistent_env_names` `setup.py:17-25`、envs 恢复 `__init__.py:7-13` | 构建期 env 快照陷阱 |

**`heuristics.hpp` 今天重点读四段**（全文 757 行，第 12 天读过骨架，今天带着"验证打印"的目的读）：

- **`:749-753`（打印块）**：见 3.1 节。注意 `static` 去重集合是函数级静态量，进程内持久；`std::puts` 走 C stdout。
- **`:592-657`（`select_gemm_config` 前半）**：`:604` 的 `block_cap`、`:620-621` 的 BLOCK_K 界、`:626-628` 从大到小的双层循环、`:635-641` 的代价剪枝、`:648-651` 的更宽 K 候选。对照 3.3 节的约束表逐行确认。
- **`:92-204`（`make_gemm_candidate`）**：把一组 (block_m, block_n, block_k) 变成合法 `GemmConfig` 或 `nullopt` 的全部规则：`:100-102` L0C、`:106` M-Grouped 固定 BLOCK_M=对齐值、`:111-112` l0_stages=2、`:124-147` MAD split 搜索、`:168-174` L1 阶段数。B 组的约束演算全部以此为准。
- **`:52-71`（`get_hw_desc`）**：3.4 节的硬件模型。每个字段的注释都值得抄进笔记（如 `bf16_flops_per_cycle=8192` 旁的 "4096 multiply-adds; each counts as two FLOPs"）。

**`gemm_config.hpp` 读两个层面**：数据结构（`GemmDesc` `:39-95`：问题描述，含 grouped/batched/stride 扩展字段；`GemmConfig` `:97-121`：执行方案，字段注释即流水线语义，如 `num_l1_stages` 旁的 "MTE2 <-> MTE1"）与 formatter（`:145-172`：打印行的语法，`LaunchOptions` 也有自己的 formatter `:133-143`）。`get_compiled_dim`（`:16-21`）解释了 `compiled_dims='nk'` 如何把 N/K 变成编译期常量——它同时是 `resident`/`direct_store` 精修分支的开关条件（`heuristics.hpp:544`、`:672`）。

**`bf16_gemm.hpp` 全文仅 80 行**，今天精读 `:18-42` 的 `std::format` 模板：把打印行里的 `block/mad/stages/num_blocks/l2_ctrl/direct_store/resident` 与 `desc` 的 `major/compiled_dims/acc/epilogue` 逐一对应到模板实参表——这张"打印字段 → 模板实参"的映射表是 3.2 节结论的直接证据。`:67-70` 的 dry-run 分支（`compile_without_load`）解释了为什么预编译阶段也会触发配置打印。

**`jit.hpp`** 短小：`:34-41` 的 `get_num_sms/set_num_sms`（`num_blocks` 上游）；`:78-92` 的 `init_jit` 用 `"DG"` 前缀和 include 路径构造 `AscendJIT`——README 说的"env 快照发生在 JIT 运行时首次创建"就发生在这之后的 DeepJIT 内部（`third-party/deep-jit` 子模块，本仓库未内联其源码，行为以 `README.md:81-98` 为准）。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **设置开关并准备干净缓存**（顺序不能反）：

   ```bash
   cd DeepGEMM-Ascend
   ./develop.sh          # 若第 9 天之后未再构建
   export DG_PRINT_CONFIGS=1
   export DG_JIT_PRINT_COMPILER_COMMAND=1
   export DG_JIT_CACHE_DIR=/tmp/dj_day51      # 空目录，强制观察编译命令；之后可换回默认
   ```

2. **写一个"小型 GEMM 测试"**（`normal` 全量要跑 50+ 个 shape，先自选 4 个有 README 对照的 shape），保存为 `week8_day2_baseline.py`：

   ```python
   import torch, deep_gemm
   from deep_gemm.testing import bench_msprof

   SHAPES = [(128, 7168, 2048), (128, 7168, 16384), (4096, 4096, 7168), (4096, 7168, 16384)]
   print('num_sms =', deep_gemm.get_num_sms())

   for m, n, k in SHAPES:
       a = torch.randn(m, k, dtype=torch.bfloat16, device='npu')
       b = torch.randn(n, k, dtype=torch.bfloat16, device='npu')
       d = torch.empty(m, n, dtype=torch.bfloat16, device='npu')
       deep_gemm.bf16_gemm_nt(a, b, d)          # 首调触发编译；此时打印 GemmDesc/GemmConfig 行
       torch.npu.synchronize()
       prof = bench_msprof(lambda: deep_gemm.bf16_gemm_nt(a, b, d), kernel_names='gemm_impl')
       flops, moved = 2.0 * m * n * k, (a.numel() + b.numel() + d.numel()) * 2
       print(f'{m}x{n}x{k}: {prof.dur_us:.1f} us | {prof.tflops(flops):.0f} TFLOPS | '
             f'{prof.gbps(moved):.0f} GB/s | mad={prof.aic_mad*100:.1f}% | mte2={prof.aic_mte2*100:.1f}%')
   ```

   运行 `stdbuf -oL python -u week8_day2_baseline.py | tee day51_run.log`，然后检查：
   - 每个新 shape 出现**一条**配置行（同 shape 重复调用不再打印——去重集合）；四个 shape 的 `block=` 是否不同？M=128 与 M=4096 的差异最有观察价值；
   - `launch_options.num_blocks` 是否等于开头打印的 `num_sms`；
   - `DG_JIT_PRINT_COMPILER_COMMAND` 输出的命令里能否找到 `bisheng` 与 `ld.lld`（`README.md:24`）；保持 `DG_JIT_CACHE_DIR=/tmp/dj_day51` 再跑一遍同一脚本（缓存已填充），对比前后输出的差异，体会"编译一次、缓存复用"。

3. **跑官方测试并对比两种模式**（较重，可选做或只跑前几个用例）：

   ```bash
   python tests/test_bf16_gemm.py normal --skip-prof    # 观察配置行在 Compiling 进度条阶段成批出现
   python tests/test_bf16_gemm.py normal                # 去 --skip-prof，观察 > Perf 行
   ```

   体会 3.1 节的第 3 个细节：配置行集中在 `par_compile` 阶段打印（`tests/common.py:152-176` 的 meta 预编译 + dry-run），perf 阶段只出 `> Perf` 行；`--skip-prof` 时 perf 行末尾是 `OK`（`tests/utils.py:113-114`）。

4. **交叉验证 README**：脚本里 `4096×7168×16384` 这一行应接近 README Dense BF16 的 2229.9 us / 431 TFLOPS / 192 GB / 99.8% 利用率（`README.md:113`，机器与 CANN 版本相同时）。手算该行 AI（第 50 天已算：≈2249 FLOP/B），并核对 `TFLOPS/432` 与 mad% 是否互相印证。

5. **填基线表**：按 3.6 节的表结构整理第 2～4 步的数据（含配置列：block / mad / l1_stages / direct_store / resident / num_blocks），加上表头口径说明，存为 `week8_day2_baseline.md`。这张表同时服务第 52 天（对照）与第 54 天（回归验证）。

### B 组：无 Ascend 环境（静态任务）

1. **峰值推导**（3.4 节的完整版）：从 `get_hw_desc()`（`heuristics.hpp:52-71`）出发，手算 BF16/FP8/FP4 峰值 TFLOPS 与 GM 带宽，填成下表并与 README 对数：

   | dtype | FLOP/cycle/core | 峰值 TFLOPS（计算过程） | README Hardware limit |
   |---|---|---|---|
   | BF16 | 8192 | 8192×1650e6×32 = 432.5 | 432（`README.md:113`） |
   | FP8 | 16384 | …（自己算） | 865（`:112`） |
   | FP4 | 32768 | …（自己算） | 1730（`:110`） |

2. **配置约束演算**（3.3 节方法，全部手工）：对 BF16 NT GEMM（K-major A/B，无 SF）：
   - (a) 推导 `k_alignment=16`、`min_block_k=128`、`block_cap=512`（引用 `heuristics.hpp:108/:620/:604`、`common.hpp:63` 的 `FRAC_MN` 与 `heuristics.hpp:69` 的 `fractal_row_bytes=32`）；
   - (b) 证明 `block=256×256` 与 `block=512×128` 都满足 L0C 约束（`BLOCK_M×BLOCK_N ≤ 65536`）；
   - (c) 计算 `block=256×256×256` 的 `ab_bytes` 与 `num_l1_stages`（答案：`512×256×2 = 262,144 B`，`524,288/262,144 = 2`），并与 `512×128×128` 的 3 stage 对比，解释"大 M/N 换窄 K、或小 M/N 换宽 K"是 L1 容量下的权衡；
   - (d) 推导 M=1 的 shape（如 `1×7168×2048`）的 BLOCK_M 只能是 16：`m_limit = align(1,16) = 16`，`m_range = {16, min(512,16)}`（`heuristics.hpp:608-609`）。
3. **构建 README 基线表**（3.6 节格式，数据来自 README）：
   - Dense GEMM 四行（`README.md:110-113`）：补算 AI（BF16 行可完整字节演算：A+B+D 共 427,819,008 B，AI ≈ 2249；FP8/FP4 行的 SF 字节模型含 recipe 假设，改用 `AI = TFLOPS/GB/s × 1000` 从表内两列相除得到有效 AI）与 Utilization 复核（1701/1730、861/865、431/432）；
   - FP8 推理表挑 4 行（建议 `README.md:119/123/126/130`）：记录 dur/TFLOPS/GB/s，算有效 AI 与离峰距离 `TFLOPS/865`，标注受限判断（对照第 50 天 3.7 节的结论）；
   - 这张表就是你的"第 51 天基线记录"，第 52 天的有环境同学做单变量实验时，你可以用同一张表推演"预期方向"。
4. **配置行语法练习**：合上教程，默写 `GemmDesc(...)` 与 `GemmConfig(...)` 两个格式串（以 `gemm_config.hpp:145-172` 为准），并给每个字段标注"由哪段代码决定"（对照 3.2 节的表）。再回答：如果调用换成 `bf16_gemm_nt(a, b, d, c=bias)`，打印行的哪些字段会变？（提示：`apis/gemm.hpp:107-123` 的 with_bias 判定与 `epilogue_class.hpp:51-64` 的 `AddBias`。）
5. **写基线笔记**（当天产出）：把 1～3 的结果整理成一份带口径说明的记录文件。

## 六、当天产出

**一组基线性能记录**——建议一个文件（`week8_day2_baseline.md`）+ 一张主表，至少包含：

1. **口径头**：设备/CANN 版本（或注明"数据来自 README，测量口径为 Ascend 950DT + CANN 9.20 + `bench_msprof` 冷 L2"，`README.md:102`）、AI Core 数（`num_blocks`）、迭代次数与设备侧计时的说明；
2. **主表**（3.6 节结构）：≥4 行数据，每行含 shape、选中配置（block/mad/l1_stages/direct_store/resident/num_blocks）、dur、TFLOPS、GB/s、mad%、mte2%、AI、受限判断；
3. **交叉验证段**：至少一行与 README 对数（如 Dense BF16 431 TFLOPS / 99.8%），至少一处手工复算（AI 或 num_l1_stages）；
4. **配置观察笔记**：不同 shape 的配置差异及你的解释（例如 M=1 与 M=4096 的 BLOCK_M 差异、direct_store 何时打开、resident 何时打开——引用 3.3 节的分支条件）。

写完后自查：不看教程，能否从一条配置行说出这个 kernel 用了多大的 Tile、几级 L1 流水、多少个 AI Core？能否从一条 perf 行说出受限类型与离峰距离？

## 七、自测题

1. `DG_PRINT_CONFIGS`、`DG_JIT_PRINT_COMPILER_COMMAND`、`DG_JIT_DEBUG` 三者什么关系？为什么后两个必须在进程启动前导出？
   > 前者只打印选中的配置（本仓库实现，`heuristics.hpp:749-753`）；第二个打印编译/反汇编命令（DeepJIT 实现，`README.md:93`）；`DG_JIT_DEBUG` 是超集，还包含加载耗时报告（`README.md:88`）。`DG_JIT_*` 变量在 JIT 运行时首次创建时被快照、且可回退到 `DJ_JIT_*`（`README.md:85`），进程内事后修改无效，须启动前 export。
2. 打印行里 `block=` 与 `mad=` 有什么区别？
   > `block` 是调度与搬运的输出 Tile（BLOCK_M/N/K），决定 Tile 数和 GM→L1→L0 的数据量；`mad` 是单条 MAD 指令的矩阵分块（MAD_M/N/K），由 block 按 split (1,1)/(1,2)/(2,1) 切分得到（`heuristics.hpp:124-147`），决定 L0A/L0B 到 L0C 的计算粒度（第 22 天两层 K 循环）。
3. BF16 下 `block=512×128×128` 的 `l1_stages` 是多少？写出计算。
   > 3。`ab_bytes=(512+128)×128×2=163,840 B`；`num_l1_stages = min(8, 524,288/163,840) = 3`（整数除法，`heuristics.hpp:170-174`）；3×163,840=491,520 ≤ 512KB 而 4×超容量。
4. `launch_options.num_blocks` 从哪里来？怎么改变它？
   > 候选生成时取 `d.num_cores`（`heuristics.hpp:98`），desc 的 `num_cores` 来自 `runtime->get_num_sms()`（`apis/gemm.hpp:176`），默认查设备（32），可用 `deep_gemm.set_num_sms()` 覆盖（`jit.hpp:38-41`）——这是第 52 天"AI Core 数量"实验的入口。
5. 为什么配置行会在测试的 `Compiling` 阶段就大量出现，而 perf 阶段不再打印？
   > `select_gemm_config` 在 dry-run 判断之前执行（`bf16_gemm.hpp:16-17` vs `:67-70`），而 `par_compile` 在 dry-run 下预编译全部用例（`par_compile.py:6-13`）；加上按整行字符串的进程级去重集合（`heuristics.hpp:750-752`），每个唯一 (desc, config) 只在首次出现时打印一次。
6. 一条 perf 行显示 `mad=98.7% | mte2=31.0%`，这说明什么？若显示 `mad=45% | mte2=42%` 呢？
   > 前者：Cube MAD 近饱和，计算受限（对应 Dense GEMM 99%+ 利用率的形态，`README.md:108-113`）。后者：两条主流水线都不满，瓶颈在并行度/调度/尾部效应（如 M 很小时 Tile 数不足），应检查 Tile 数与 AI Core 数的关系——参考第 50 天 3.3 节判别流程第③步。
7. 用 `get_hw_desc()` 的数字推导 FP8 峰值算力。
   > `get_mad_flops_per_cycle(8) = 8192×16/8 = 16384 FLOP/cycle/core`（`heuristics.hpp:46-48`）；峰值 = 16384×1650 MHz×32 = 865.0 TFLOPS，与 README Hardware limit 865 一致（`README.md:112`）。
8. 为什么基线表必须把冷 L2、迭代次数、设备侧计时写进口径头？
   > 第 52 天的单变量实验要与今天的基线**同口径**对比：冷/热 L2 改变带宽观感（`bench_msprof` 默认每次迭代前清 2×L2 缓冲，`bench.py:263-268`、`:344-346`）；迭代均值与设备侧 PMU 计时排除 JIT 编译与 Host 噪声（第 50 天 3.1/3.6 节）。口径不同，对照无效。

---

明天（第 52 天）在今天的基线上做**单变量性能实验**：固定一个 shape，每次只改一个因素——Block 大小、L1/L0 stage 数、resident A/B、direct store、AI Core 数量（`set_num_sms`）——每次先验证正确性再比较性能，并练习"提出假设 → 验证假设"，而不是只给一组调参结果。
