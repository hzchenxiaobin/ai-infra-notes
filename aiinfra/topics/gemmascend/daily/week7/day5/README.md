# 第 7 周 · 第 5 天（总第 47 天）：学习 HC Prenorm / TileLang

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 7 周"MegaMoE、HC Prenorm 与扩展算子"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 25 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前四天我们把 MegaMoE 的接口、计算主线、通信融合和测试性能入口读完了。今天换一个视角：仓库里**唯一一个不用手写 C++ 模板实现的算子**——HC Prenorm（mHC 模块的 prenorm 融合 GEMM），它用 [TileLang](https://github.com/tile-ai/tilelang) 这个 Python DSL 编写（README "The mHC kernel is backed by Tilelang"，`README.md:199`；`tilelang` 是包依赖，`README.md:28`、`setup.py:149`）。只读两个文件：`deep_gemm/tilelang_ops/hc_prenorm.py`（398 行，配置 + Kernel + API 全在一个 Python 文件里）和 `tests/test_hyperconnection.py`（160 行，正确性 / 确定性 / 性能三层验证）。今天的四个核心问题：**HC Prenorm 算什么、为什么能与 GEMM 融合；TileLang Kernel 如何表达我们前六周熟悉的那套硬件概念（分块、多级缓冲、AIC/AIV、NZ 布局、K-split 原子归约）；deterministic 模式如何做到位级可复现与 batch invariance；手写 C++ Kernel 与 TileLang Kernel 两种开发方式在代码结构和工作流上有何差异**。当天产出是**两种 Kernel 开发方式对比笔记**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 HC Prenorm 的计算契约（`hc_prenorm.py:363-379` docstring）：`D = float(A) @ B.T` 与 `sqr_sum = sum(float(A) ** 2, dim=-1)` 在**一个 Kernel** 内完成；说出 a/b/d/sqr_sum 的形状与 dtype（`[M, K]` BF16 / `[N, K]` FP32 / `[M, N]` FP32 / `[M]` FP32），并解释 `num_splits` 只接受 None 和 1 的原因（K split 在 Kernel 内部归约，`:377-388`）；
2. 解释测试用例形状的语义（`test_hyperconnection.py:14-16` 注释）：m 是 token 数，`n = hc_mult * (hc_mult + 2)` 是 mix 输出列数（hc_mult=4 → n=24），`k = hc_mult * hidden`（k=28672 → hidden=7168），以及 k=7680/7168 两个形状为何存在（覆盖 K block 数不是 2 的幂的情况）；
3. 手工推演 `get_config()`（`hc_prenorm.py:13-58`）：固定 tile（BLOCK_M=128、BLOCK_K=256、MAD_K=64）、L0B/L0C 容量整除出 stage 数、K-split 的三个候选（workload / hidden / cores）与"降到最大 2 的幂"，并解释 deterministic 模式把 workload split 固定为 16 的原因（`:33-35` 注释：M 相关的 split 选择会改变浮点归约分组、破坏 batch invariance）；
4. 对照列出 Kernel 的片上缓冲规划（`:112-132`）：`x_ub/x_nz_ub/sqrsum_ub`（UB）、`x_l1/fn_l1`（L1）、`x_l0`（L0A）、`fn_l0`（L0B）、`out_mul_l0`（L0C），用 `T.annotate_buffer_versions` 的版本数手工核验四条容量不等式（L1 ≤ 512KB、L0A/L0B ≤ 64KB、L0C ≤ 256KB、UB ≤ 256KB，硬件常量见 `deep_gemm/include/deep_gemm/common.hpp:50-58`）；
5. 追踪一条 x 数据的完整通路（`:185-293`）：GM→`x_ub`（`pad_value=0` 边界补零）→ `T.SimdVF` 内 BF16→FP32 位级拓宽 + NZ 分形打包 + 逐行平方和 → `T.dual_copy` 从 AIV 的 UB 送 AIC 的 L1 → `x_l0` → `T.gemm`（`clear_accum` / `unit_flag_ctrl`）→ `out_mul_l0` → FIX 管道写回 GM；并能复述 m=137 时第二个 128 行块只有 9 行有效、AIV1 半区被整体跳过的边界处理；
6. 讲清跨核同步与原子归约的设计（`:96-103` 注释 + `:134-136` + `:295-357`）：非 deterministic 模式"split 0 普通写、其余 split 原子加"的两相位结构、`PIPE_FIX`/`PIPE_MTE3` 两组 flag 编号、最后一步屏障为何只在还有下一个 chunk 时需要、以及为什么用 0/1-trip 的 `T.Serial` 循环编码可选屏障（AutoSchedule 拒绝 `T.PerCoreTask` 内条件控制流下的跨核等待）；
7. 描述测试的三层验证（`test_hyperconnection.py`）：正确性（与 `a.float() @ b.T` / `a.float().square().sum(-1)` 比，余弦差 < 1e-6）、split 契约（`num_splits=1` 时写 `d[0]`）、确定性（10 次位级重复 + 对第 0、m//2、m-1 行做单行 batch invariance 检查），以及 `compile_hc_prenorm_gemm` 如何用 dry-run 把两种确定性变体并行编译（`:31-56`、`hc_prenorm.py:394-397`、`deep_gemm/testing/par_compile.py`）；
8. 完成当天产出：一张手写 C++ 模板 Kernel（`bf16_gemm.hpp` 路线）与 TileLang Kernel（`hc_prenorm.py` 路线）的对比表，至少覆盖语言与文件组织、Host 调用链、shape 特化、内存规划、流水线表达、AIC/AIV 表达、NZ 布局处理、跨核同步、编译缓存九个维度，并能默写 8 对同构概念的对偶关系。

## 二、预备知识

- **第 5 天**：GM/L1/L0A/L0B/L0C/UB 分层与 AIC/AIV 分工——今天所有缓冲分配都建立在这张图上。
- **第 12 天**：`select_gemm_config()` 与 BLOCK_M/N/K、MAD_M/N/K 的区别——今天会看到它们的 TileLang 对应物（`block_num_tokens=128`、`block_hidden=256`、`mad_hidden=64`，`hc_prenorm.py:76-78`）。
- **第 15～16 天**：模板参数分类、片上内存 offset 规划与容量 `static_assert`——对照今天 `T.alloc_*` + `get_config` 整除计算。
- **第 22 天**：K 维两层循环（BLOCK_K 循环 × MAD_K 循环）与 `is_first` 累加——对照今天 hidden block 循环 × `hidden_mad` 循环与 `clear_accum`。
- **第 23 天**：MTE1/MTE2/M/FIX 管道与 `asc_sync_wait/notify` 配对——今天的 `T.ascend_sync_inter_arrive/wait` 是**跨核**同步，注意区分。
- **第 25 天**：`__mix__(1, 2)` 混合 Kernel、`ASCEND_IS_AIC` 分支、AIC store 与 AIV drain——对照今天 `T.MixedKernel` + `(core_id, aiv_id)` + `T.PerCoreTask`。
- **第 26 天**：`kDirectStore` 路径的 `asc_set_atomic_add_float()`（`bf16_gemm.hpp:69`）——与今天 `T.set_atomic('add', 'float32')` 同一思想。
- **第 30～31 天**：`calc_diff` 余弦差（`tests/utils.py:36-43`）、参考结果交叉验证、并行编译（dry run + meta 思路）。
- **第 46 天**：`bench_msprof`、`KernelProfile` 的管道利用率字段（`deep_gemm/testing/bench.py:27-71`）——今天性能行里的 `mad=%`/`mte2=%` 直接来自它。
- **背景概念**：mHC（Manifold-Constrained Hyper-Connections）是 DeepSeek 模型的模块，本算子是其归一化前（"prenorm"）的一部分：对每 token 的 `hc_mult × hidden` 扩展隐藏态算一次小 N 大 K 的混合矩阵乘，同时产出逐行平方和统计量。模型层细节超出本仓库范围，掌握上面两个公式即可。

硬件常量回顾（`common.hpp:50-58`，Ascend 950DT）：32 个 AI Core、L1 512KB、L0A/L0B 各 64KB、L0C 256KB、UB 256KB。`hc_prenorm.py:17-18` 硬编码的 `l0b_size_bytes = 64KB`、`l0c_size_bytes = 256KB` 与之完全一致。

## 三、详细讲解

### 3.1 HC Prenorm 算什么：算子契约、用例形状与性能特征

对外 API 只有一个函数（经 `deep_gemm/__init__.py:157` 从 `tilelang_ops` 导出）：

```python
deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None)
# D = float(A) @ B.T，且 sqr_sum = sum(float(A) ** 2, dim=-1)，一个 kernel 完成
# a: [M, K] BF16 输入；b: [N, K] FP32 权重
# d: [M, N] FP32 输出；sqr_sum: [M] FP32 输出
# num_splits: 只接受 None 或 1 —— 本后端在 kernel 内部完成 K split 的归约
```

（`hc_prenorm.py:363-398`。）`num_splits` 的契约值得注意：接口保留了 `[num_splits, M, N]` 形状的输出以兼容上游 API，但本后端在 Kernel 内部就把各 split 的部分和加完，所以外部只需要一份结果（`:386-388` 注释）。空输入 `m == 0` 直接返回（`:390-391`）。

Kernel 内部的形参名与 API 不同，对应关系是 `a→x`、`b→fn`、`d→out_mul`、`sqr_sum→sqrsum`（`:91-94`）。名字里的 `tf32` 来自 `T.set_hf32_mode('nearest_even')`（`:136`）：Ascend cube 单元的 HF32 模式是 TF32 的同类物（截断尾数位宽的 FP32 乘加），舍入模式取 nearest-even；A 是 BF16（尾数位宽小于 HF32，可精确表示），B 是 FP32，会被舍入到 HF32 参与计算——这也是测试容差不能为 0 的原因（见 3.7）。

测试用例的形状语义（`test_hyperconnection.py:14-16` 注释）：

| 符号 | 测试值 | Kernel 参数 | 含义 |
| :---: | :---: | :---: | :--- |
| m | 13 / 137 / 512 / 4096 / 8192（peak 另有 1~32768） | `num_tokens`（`T.dynamic`，`:87`） | token 数 |
| n = hc_mult·(hc_mult+2) | 24（hc_mult=4） | `mhc_mult3`（`T.const`，`:88`） | mix 输出列数 |
| k = hc_mult·hidden | 28672/20480/16384/7680/7168 | `mhc_hidden_size`（`T.const`，`:89`） | 扩展隐藏态宽度（hidden = k/4 = 7168/5120/4096/1920/1792） |

注释里说得很清楚：k=16384/20480/28672 是生产形状（hidden 4096/5120/7168），**7680 和 7168 是为了覆盖"K block 数不是 2 的幂"**——按 256 一块，它们分别有 30 和 28 个 hidden block（28672 有 112 个），这个差异会直接影响 3.3 的 K-split 钳制。

性能特征：这是"小 N 大 K"的极端形状——N 只有 24，K 却达 28672。一个 `(128, 32)` 的输出 tile 要读 `(128, 256)` 的 A 块和 `(32, 256)` 的 B 块，读写流量远大于计算量。README 的性能表（`README.md:175-185`）称其 "nearly saturates the HBM write bandwidth"。用测试的口径手工验证两行（`flops = 2·m·n·k`，`bytes = count_bytes(a, b, d, sqr_sum)`，即 a 每元素 2B、b/d/sqr_sum 每元素 4B，`test_hyperconnection.py:92-96`）：

| M | N | K | FLOP | 字节（a/b/d/sqr） | README 时延 | TFLOPS | GB/s |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 13 | 24 | 28672 | 17.9M | 745,472 / 2,752,512 / 1,248 / 52 ≈ 3.50MB | 6.7 µs | 2.7→**3** | 522→**519** |
| 8192 | 24 | 28672 | 11.27G | 469.76M / 2.75M / 0.79M / 0.03M ≈ 473.3MB | 136.7 µs | 82.5→**82** | 3462→**3463** |

两个结论：小 M 时 **B 的读取占 79%**（2.75/3.50MB），大 M 时 **A 的读取占 99%**（469.8/473.3MB）；计算利用率最高只有 82 TFLOPS，远低于 cube 峰值——这是一个**访存受限**的算子，优化目标就是把总流量压到最小。这正是"融合"的动机。

### 3.2 融合的收益：一份搬运，两种计算

如果不融合，需要两个 Kernel：GEMM 读一遍 A，`square().sum(-1)` 再读一遍 A——对 M=8192 的用例，主导流量（A 读取 469.8MB）直接翻倍到约 943MB。融合方案让**同一份 A 数据的搬运同时喂饱两种计算单元**：

```text
                        ┌──────────── AIV（每 core 2 个，各管 128 行 tile 的 64 行）────────────┐
                        │                                                                      │
GM[x] ──MTE2──▶ x_ub ──SimdVF──┬──▶ x_nz_ub（NZ 分形）──dual_copy──▶ x_l1 ──MTE1──▶ x_l0(L0A) ─┐
    （pad_value=0 补零）        │                                                │             │ T.gemm（HF32）
                               └──▶ sqrsum_ub ──MTE3──▶ GM[sqrsum]               ▼             ▼
GM[fn] ──MTE2──▶ fn_l1 ──MTE1──▶ fn_l0(L0B) ─────────────────────────────▶ out_mul_l0(L0C) ──FIX──▶ GM[out_mul]
```

- **AIV（向量核）**：把 x 的半个 tile 从 GM 搬进 UB；在把 BF16 **位级拓宽**成 FP32、打包成 cube 需要的 **NZ 分形布局**的同时，用 SIMD 顺手算出每一行的平方和（3.5 详解）。行平方和从 AIV 的 UB 经 MTE3 直接写回 GM。
- **AIC（矩阵核）**：`T.dual_copy` 把 AIV 打包好的数据送进 L1，再进 L0A/L0B，用 cube 做 GEMM；结果在 L0C 里跨 hidden block 累加，最后经 FIX 管道写回 GM。

对比第 25 天 BF16 Kernel 的分工：那边 AIV 干的是** epilogue**（bias/alpha/输出转换），这边 AIV 干的是 **prologue + 旁路输出**——拓宽和 NZ 打包本来就必须在进 cube 之前完成，行平方和搭了这趟"顺风车"，一行额外的 GM 读取都没有。这就是"Prenorm 与 GEMM 的融合"的本质：**归一化统计量与 GEMM 共用同一份数据搬运**。

### 3.3 `get_config`：住在 Python 里的启发式

`get_config(mhc_hidden_size, mhc_mult3, num_tokens, deterministic)`（`:13-58`）是纯 Python 的配置启发式——对比第 12 天住在 `csrc/jit_kernels/heuristics.hpp` 里的 C++ 版本，位置变化的根本原因是整个 Kernel 都在 Python 里。逐步拆解（设 n=24、950DT 的 32 个 AI Core）：

1. **固定 tile**（`:15-16`）：`block_num_tokens=128`（即 BLOCK_M）、`block_hidden=256`（即 BLOCK_K）。shape 门槛在 `:20-24`：`mhc_mult3` 必须 8 对齐且 ≤64，`mhc_hidden_size` 必须是 256 的倍数。
2. **物理 N 对齐**（`:27`）：`block_mhc_mult3 = (mhc_mult3 + 15) // 16 * 16`——n=24 向上取到 32。物理缓冲按 32 列分配，但数据搬运和计算区域仍按逻辑的 24 列截取（Kernel `:159-160` 注释："Keep physical buffers block_mhc_mult3-aligned, but restrict data movement and MAD regions to the logical mhc_mult3"）。
3. **stage 数 = 容量整除**（`:30-31`）：`num_l0b_stages = 64KB // (32 × 256 × 4) = 2`（B tile `[32, 256]` FP32 每个 32KB，L0B 放 2 份）；`num_l0c_stages = 256KB // (128 × 32 × 4) = 16`（输出 tile `[128, 32]` FP32 每个 16KB，L0C 放 16 份）。
4. **K-split 的三个候选**（`:33-42`）。先看源码注释原文（`:33-35`）：

   > K partitioning defines the per-row floating-point reduction. M-dependent split selection improves L0C residency/B reuse, but changes the reduction grouping and breaks batch invariance.

   K 维（hidden 方向）切给多组 core 并行，各组算部分和再原子相加——切法决定了每行浮点累加的分组顺序。三个候选：

   - `num_k_splits_by_workload = 16 if deterministic else max(1, 32 × 16 // num_token_blocks)`——**与 M 相关**！deterministic 模式强制取 16，正是为了切断与 M 的依赖（batch invariance）；
   - `num_k_splits_by_hidden = num_hidden_blocks & -num_hidden_blocks`——取 hidden block 数的最低置位比特（最大 2 幂因子），保证每组分到整数个 block：k=28672（112 块）→ 16，k=20480（80 块）→ 16，k=16384（64 块）→ 64，k=7680（30 块）→ 2，k=7168（28 块）→ 4；
   - `num_k_splits_by_cores = 32 & -32 = 32`——保证每组整数个 core。

   `split_limit = min(三者)`，再 `num_k_splits = 1 << (split_limit.bit_length() - 1)` 降到不超过它的最大 2 的幂（workload 项可能不是 2 的幂，如 `512 // 3 = 170`）。
5. **派生量**（`:44-47`）：`active_aic_cores = num_k_splits × min(32 // num_k_splits, num_token_blocks)`（token block 太少时不激活空转的 core）；`cores_per_split`、`num_hidden_blocks_per_split`、`num_token_blocks_per_chunk = cores_per_split × num_l0c_stages`（每个 core 有 16 个 L0C 槽，一组 `cores_per_split` 个 core 一个 chunk 消化 `cores_per_split × 16` 个 token block，多了就分多个 chunk）。

对 k=28672（112 blocks、by_hidden=16）把 default 用例全部手算一遍（cores=32）：

| m | token_blocks | 非 det workload | splits | cores/split | hidden/split | blocks/chunk | chunks | det splits |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 13 | 1 | 512 | 16 | 1 | 7 | 16 | 1 | 16 |
| 137 | 2 | 256 | 16 | 2 | 7 | 32 | 1 | 16 |
| 512 | 4 | 128 | 16 | 2 | 7 | 32 | 1 | 16 |
| 4096 | 32 | 16 | 16 | 2 | 7 | 32 | 1 | 16 |
| 8192 | 64 | 8 | **8** | 4 | 14 | 64 | 1 | **16**（chunks=2） |

注意 m=8192 这一行：非 det 选 8 个 split（每个 split 有 4 个 core、14 个 hidden block、64 个 token block 一个 chunk 装下），det 固定 16 个 split（每 split 2 core、7 个 hidden block、每 chunk 32 个 token block、共 2 个 chunk）——**这是 default 用例中两种模式配置不同的唯一 m 值**（k=7680/7168 时 by_hidden 只有 2/4，两种模式都被钳到相同值）。

### 3.4 Kernel 主体：TileLang 里的同一套硬件概念

`@tilelang.jit` 装饰的 `tf32_hc_prenorm_gemm_kernel`（`:60-360`）用 Python 语法写 Kernel，`T.*` 原语在编译期被 TileLang 降低成设备代码。先看声明区（`:76-94`）：常量 `block_num_tokens=128`、`block_hidden=256`、`mad_hidden=64`、`num_aivs=2`（对照 `bf16_gemm.hpp:25` 的 `__mix__(1, 2)`）；`T.dynamic('num_tokens')` 与 `T.const('mhc_mult3'/'mhc_hidden_size')` 区分运行期/编译期 shape——正是第 15 天 C++ 模板里 `SHAPE_M != 0 ? SHAPE_M : shape_m_runtime` 约定的另一种写法。

**片上缓冲规划**（`:112-132`）——把 `T.alloc_*` 的位置、`T.annotate_buffer_versions` 的版本数列成表（n=24 的配置）：

| 缓冲 | 分配 | 形状 / dtype | 版本数（流水级数） | 单版本大小 | 位置 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `x_ub` | `T.alloc_shared` | (64, 256) BF16 | 3（`num_load_stages`） | 32 KB | UB（AIV） |
| `x_nz_ub` | `T.alloc_shared` + NZ 布局标注 | (65, 256) FP32 | 2（`num_cast_stages`） | 65 KB | UB（AIV） |
| `sqrsum_ub` | `T.alloc_shared` | (16, 64) FP32 | 1（首维即 16 个 stage 槽） | 4 KB（整块） | UB（AIV） |
| `x_l1` | `T.alloc_l1` | (128, 256) FP32 | 3（`num_l1_x_stages`） | 128 KB | L1（AIC） |
| `fn_l1` | `T.alloc_l1` | (32, 256) FP32 | 2（`num_l0b_stages`） | 32 KB | L1（AIC） |
| `x_l0` | `T.alloc_l0a` | (128, 64) FP32 | 2（`num_mad_stages`） | 32 KB | L0A |
| `fn_l0` | `T.alloc_l0b` | (4, 32, 64) FP32 | 2（`num_l0b_stages`） | 32 KB | L0B |
| `out_mul_l0` | `T.alloc_l0c` | (16, 128, 32) FP32 | 1（首维 16 个 token stage 槽） | 16 KB/槽 | L0C |

手工核验容量（对应 C++ 侧 `static_assert(l1b.offset(...) <= L1SizeBytes)` 那组检查，`bf16_gemm.hpp:66-69`）：

- L1：`3×128KB + 2×32KB = 448KB ≤ 512KB`；
- L0A：`2×32KB = 64KB`，L0B：`2×32KB = 64KB`——**都恰好用满**；
- L0C：`16×16KB = 256KB`——**恰好用满**；
- UB：`3×32KB + 2×65KB + 4KB = 230KB ≤ 256KB`。

`T.annotate_buffer_versions`（`:123-132`）的语义就是多版本缓冲（multi-buffering）供流水轮转，概念上对应 C++ 的 `kNumL1Stages/kNumL0Stages`——只是那边靠手写 `s = s == N-1 ? 0 : s+1` 的下标轮转（`bf16_gemm.hpp:112`），这边声明出来交给 TileLang 的调度器。

**执行骨架**（对照源码行号）：

```text
with T.MixedKernel(active_aic_cores):                      # :106，(core_id, aiv_id)
  split_id = core_id // cores_per_split                    # :107
  for chunk in T.Serial(num_token_chunks):                 # :137
    for hidden_block in T.Pipelined(hidden_blocks_of_split, num_stages=2):   # :155
      装 fn：GM → fn_l1 → fn_l0（4 段 mad_hidden）          # :161-169
      for token_stage in T.Serial(num_valid_token_stages): # :171，含 multi_buffer_eligible 标注
        AIV：x GM → x_ub（pad_value=0）                     # :185-193
        AIV：SimdVF 拓宽 + NZ 打包 + 行平方和 → sqrsum_ub   # :196-267
        AIV→AIC：T.dual_copy(x_nz_ub → x_l1)               # :269-272
        AIC：for hidden_mad in 4: L1→L0 + T.gemm           # :274-293
    PerCoreTask：按相位写回 out_mul（FIX 管道）+ 跨核屏障   # :295-325
    PerCoreTask：按相位写回 sqrsum（MTE3 管道）+ 跨核屏障   # :327-357
```

几个关键对应关系：

- **token block 的分配**：`token_block = token_block_base + token_stage × cores_per_split`（`:175`）——同 split 内各 core 交错认领 token block，各自填满自己的 16 个 L0C 槽（`out_mul_l0` 的首维），这就是 `num_token_blocks_per_chunk = cores_per_split × num_l0c_stages` 的来历。
- **K 两层循环再现**：外层 `T.Pipelined(num_hidden_blocks_per_split, num_stages=2)` 是 BLOCK_K 级（256 列一块、双缓冲预取下一块），内层 `for hidden_mad in T.Serial(num_mads_per_block)`（`256/64 = 4` 次）是 MAD_K 级——第 22 天的结构原样出现。
- **首次累加**：`clear_accum=(local_hidden_block == 0 and hidden_mad == 0)`（`:287`）——正是 C++ 里的 `is_first`（`bf16_gemm.hpp:134`），甚至名字都一样。`unit_flag_ctrl` 在非最后一步传 2、最后一个 MAD 传 3（`:288-292`），控制 L0C→FIX 管道的 tile 释放标志：只有累加完成的 tile 才允许被 FIX 取走（写回的 `T.copy` 同样传 3，`:316`；具体编码由 TileLang Ascend 后端定义）。
- **边界处理**：`actual_num_tokens = T.min(num_tokens - token_begin, 128)`（`:177`）；`num_valid_token_stages` 把 stage 数钳到剩余 block 数（`:139-142`）；AIV 半区级判断 `if aiv_token_begin < num_tokens:` 整段跳过（`:180`）；x 的加载带 `pad_value=0`（`:192`），不足一行补零——零行对 GEMM 和平方和都无贡献。以 m=137 为例：第二个 128 行块只有 9 行有效，AIV0 处理第 128~136 行（补零到 64 行），AIV1 的起始行 192 ≥ 137，整段跳过；AIC 侧 `T.gemm` 与 L1→L0 搬运都只取 `actual_num_tokens=9` 行（`:277-285`）。

### 3.5 AIV 的 SIMD 段：拓宽、NZ 打包与行平方和

`:196-267` 的 `with T.SimdVF(latency=640):` 是一段显式 SIMD 向量代码（latency 参数为调度器提供时延信息）。逐块拆解：

**（1）BF16→FP32 位级拓宽**（`:221-225`）：每行 256 个 BF16 分两个 pass，每 pass `vld` 加载 128 个；`vintlv(zero_bf16, packed)` 把 0 半字与 BF16 元素交错，再 `T.reinterpret(..., 'float32x64')` 重释成 64-lane 的 FP32 向量 `lo/hi`——BF16 恰好是 FP32 的高 16 位，所以"补零交错 + 重释"就是一次免费的类型拓宽，一条指令都不用算。

**（2）行平方和**（`:227-235, :252-262`）：`row_acc += lo·lo + hi·hi`（`vadd/vmul`）把每行 256 个平方累进 `row_acc` 的 64 个 lane；行末 `vcadd` 跨 lane 规约成标量 `row_sum`，`vdupv` 广播回去，`vcmp(row_indices, current_row, 'eq')` 生成"第 `current_row` 行"的 lane 掩码，`vsel` 把 `row_sum` 写进 `sqr_sums` 的对应 lane。循环 64 行后，**`sqr_sums` 的第 i 个 lane 就是第 i 行的平方和**。跨 hidden block 累加（`:264-267`）：`local_hidden_block != 0` 时先 `vld` 读出上一块的结果相加再写回 `sqrsum_ub[token_stage]`。

**（3）NZ 分形打包**（`:237-250`）：`vsstb` 是带步长的散射写：`nz_stride = (nz_stage_rows << 16) | (8 * nz_stage_rows)` 把 NZ 布局需要的两级步长编码进一个 int32，`lo/hi` 直接落到 `x_nz_ub` 的分形位置上；`x_nz_ub` 的形状 (65, 256)（`nz_stage_rows = 64 + 1`，`:104`）与 `T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})`（`:114`）共同定义了这个紧凑 NZ 布局。

为什么要自己打包？**cube 的 L0A/L0B 只吃分形（NZ）布局**——README 开篇说的 "fractal layouts" 复杂度，在 C++ 库里被 `copy_gm_to_l1/copy_l1_to_l0a/b` 原语隐藏（`ascend/copy.hpp`，第 19 天）；但这里数据是 AIV 现场生成的 FP32，必须由 AIV 亲手摆成分形形状。这是 TileLang 路线一个真实的额外负担，也是这段 SIMD 代码存在的原因。

**（4）跨核交接**（`:269-272`）：`T.dual_copy(x_nz_ub, x_l1)` 完成 AIV 的 UB → AIC 的 L1 搬运——两个 AIV 各把自己那 64 行 NZ 数据送进 `x_l1`，拼成 cube 侧完整的 128 行 tile。普通 `T.copy` 表达不了这种跨计算单元的生产者-消费者关系，`dual` 的名字即由此来。

回到 3.2 的图：这一段 SIMD 代码就是整个融合的核心——**拓宽（GEMM 需要 FP32）、NZ 打包（cube 需要）、行平方和（prenorm 需要）三件事在数据流过 UB 的瞬间一次完成**。

### 3.6 跨核同步、原子加与确定性（当天难点）

K-split 之后，多组 core 写**同一个输出区域**，这在前六周的主线 GEMM 里从未出现过（每个输出 tile 只有一个 core 负责）。解决方案分三层：

**（1）原子加**（`:134-136`）：`if split_id != 0: T.set_atomic('add', 'float32')`——非 0 号 split 的所有 GM 写（`out_mul` 和 `sqrsum`）变成浮点原子加；Kernel 结束时 `T.set_atomic_none()`（`:359-360`）。对比 `kDirectStore` 路径的 `asc_set_atomic_add_float()`（`bf16_gemm.hpp:69`），同一思想。

**（2）store 相位与屏障**（`:96-103` 注释 + `:295-357`）。非 deterministic 模式（`num_k_splits > 1` 时 `num_store_phases = 2`）：

- **phase 0**：只有 split 0 写（普通写，非原子）——先把一份"干净的"部分和落进 GM；
- **phase 1**：其余 split 原子加上自己的部分和。

如果不加屏障，split 2 的原子加可能落在 split 0 的普通写**之前**，然后被普通写覆盖——结果就错了。所以两个相位之间必须有全核屏障。两个 `T.PerCoreTask` 块分别处理 `out_mul`（L0C→GM，走 FIX 管道，flag 编号 `(chunk × phases + phase) % 4`）和 `sqrsum`（UB→GM，走 MTE3 管道，flag 编号 `4 + (...) % 4`），每个相位末尾 `T.ascend_sync_inter_arrive('PIPE_FIX'/'PIPE_MTE3', flag)` + `ascend_sync_inter_wait(...)` 成对出现，构成 32 个核之间的屏障（`:323-325, :355-357`）。

"最后一步屏障只在还有下一个 chunk 时需要"（`:96-98` 注释）：下一个 chunk 会复用 L0C/UB 的 stage 槽和循环使用的 flag 编号，必须等所有核写完才能开新 chunk；如果后面没有 chunk 了，这笔屏障就是纯开销。注意这套同步是**跨核**（inter-core）的，与第 23 天 AIC 内部管道间的 `asc_sync_wait/notify`（MTE1/MTE2/M 配对）不是一回事。

**（3）deterministic 模式**（`:96`：`num_store_phases = num_k_splits`；`:298-300`：`store_this_phase = (split_id == store_phase)`）：每个 split 在**自己的相位**里写，相位间有屏障——GM 上的累加顺序被固定为 split 0 → 1 → 2 → …，于是同一输入必然产生位级相同的结果。再配合 3.3 里"det 模式 split 数与 M 无关"，单行 batch 与整行 batch 的归约分组也完全一致——这就是 batch invariance 的两块拼图。

最后是那段著名的注释（`:100-103`）：

> Encode that optional barrier as a uniform 0/1-trip loop because AutoSchedule rejects inter-core waits under conditional control flow inside T.PerCoreTask.

"最后一个相位是否需要屏障"是个运行期条件，但 TileLang 的 AutoSchedule 拒绝在 `T.PerCoreTask` 内的条件控制流下放置跨核等待。作者的办法是把屏障包进 `for _ in T.Serial(num_store_syncs)`（`:323-324, :355-356`），其中 `num_store_syncs` 是一个 0 或 1 的循环次数（`:318-322`）——**用均匀的循环把条件编码掉**。这是读 TileLang 代码时很典型的"与调度器博弈"的痕迹：C++ 路线里你直接写 `if`，TileLang 路线里你得顺着 AutoSchedule 的规矩来。

**（4）L2 提示**：x 的加载带 `l2_cache_ctrl='notalloc_keep'`（`:191`）——K-split 下每个 x 元素全局只读一次，分配进 L2 没有收益；sqrsum 的写回带 `'normal_fv'`（`:348`）。对照 C++ 的 `asc_load_l2_cache_mode::NOTALLOC_KEEP` / `asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM`（`common.hpp:120-131`）——同一组硬件提示，两种拼写。

### 3.7 测试：三层验证与两批并行编译

`tests/test_hyperconnection.py` 的结构：

**（1）用例集**（`:17-27`）：`default` = m∈{13, 137, 512, 4096, 8192} × (n,k)∈{(24, 28672), (24, 20480), (24, 16384), (24, 7680), (24, 7168)}，共 25 个；`peak` = m∈{1, 13, 128, 4096, 8192, 16384, 32768} × (24, 28672)。m=13 测不足一个 block，m=137 测跨 block 边界（128+9）。

**（2）并行编译**（`:31-56`，`compile_hc_prenorm_gemm`）：先用 `torch.zeros` 造齐所有 shape，把调用包成 `partial`；然后 `for deterministic in (False, True): par_compile(kernels)`——**两批**，因为确定性开关改变 `get_config`，每个 shape 要编译两个 Kernel 变体。`par_compile`（`deep_gemm/testing/par_compile.py:16-35`）在线程池里逐个执行 callable，并用 `dry_run()` 上下文（`:6-13`）置位 `_C.set_dry_run(True)`；Kernel 侧的对应逻辑在 `hc_prenorm.py:394-397`：dry run 时只调 `.compile(...)` 不 launch——这就是第 30 天"meta tensor 并行编译"在本仓库的实现形态（配置、缓存和 launch 全部走通，只差最后一步执行）。

**（3）主测试**（`:59-98`，`test_hc_prenorm_gemm`）三层验证：

1. **正确性**：参考结果 `ref_d = a.float() @ b.T`、`ref_sqr_sum = a.float().square().sum(-1)`，`calc_diff`（余弦差，`tests/utils.py:36-43`）阈值 **1e-6**——比 BF16 GEMM 的容差紧得多，因为输出是 FP32 且 a 本身是 BF16（在 HF32 中精确）；但 B 从 FP32 舍入到 HF32 引入的误差使它不能要求位级相等；
2. **split 契约**：用 `(1, m, n)` 形状的 `d_split` 调 `num_splits=1`，验证 `d_split[0]` 与参考一致（`:79-84`）——保证上游按 split 形状调用时语义正确；
3. **性能**（`--skip-prof` 时不做）：`bench_msprof(..., kernel_names='tf32_hc_prenorm_gemm')` 按名字匹配 TileLang 生成的 Kernel，`print_perf` 打印 `us / TFLOPS / GB/s / mad% / mte2%`（`tests/utils.py:106-115`），flops 与 bytes 的口径即 3.1 的两个公式。`mad%` 低、`mte2%` 高正是"访存受限"论断在 profiling 数据上的投影。

**（4）确定性测试**（`:101-135`，`test_hc_prenorm_gemm_deterministic`）：先按 1e-6 验证正确性，然后——

- **位级可复现**：保存首次结果，再跑 10 次，`torch.equal(d.view(torch.int32), ...)` 要求位级相同（`:121-124`）；
- **batch invariance**：取第 0、m//2、m-1 行（`sorted({0, m // 2, m - 1})`）单独作为单行 batch 重算，要求与整批结果的对应行**位级一致**（`:126-132`）。

这两个检查分别对应 3.6 的两块设计：固定相位的写回顺序（可复现）与 M 无关的 split 数（batch invariance）。测试开头的 `disable_cpu_fallback()`（`deep_gemm/testing/disable_fallback.py`）给所有会回退 CPU 的算子注册抛异常的 NPU kernel，防止参考计算悄悄落到 CPU 上拖慢测试、掩盖问题。

`__main__`（`:140-160`）：`select ∈ {all, default, peak}`、`--skip-prof`；固定种子后先 `compile_hc_prenorm_gemm` 再对每个 slice 跑主测试和确定性测试。`python tests/test_hyperconnection.py default` 不加 `--skip-prof` 时，k=28672 的 5 个用例的性能行正好对应 README 表的 5 行。

### 3.8 当天产出·参考：两种 Kernel 开发方式对比笔记

**主对比表**（对照物：手写 C++ 路线的 `deep_gemm/include/deep_gemm/bf16_gemm.hpp` + `csrc/` Host 链；TileLang 路线的 `deep_gemm/tilelang_ops/hc_prenorm.py`）：

| 维度 | 手写 C++ 模板（DeepJIT 路线） | TileLang（Python DSL 路线） |
| :--- | :--- | :--- |
| 代表文件 | `deep_gemm/include/deep_gemm/*.hpp`（device）+ `csrc/apis/`、`csrc/jit_kernels/`（host） | `deep_gemm/tilelang_ops/hc_prenorm.py` 单文件（398 行，含配置与 API） |
| Kernel 语言 | C++20 模板，`__global__ __mix__` 函数 | Python DSL，`@tilelang.jit` 装饰的函数 + `T.*` 原语 |
| Host 调用链 | python_api.cpp → `get_gemm_desc` → `launch_bf16_gemm` → `select_gemm_config`（heuristics.hpp）→ 生成模板实例化代码 → `jit->compile/launch`（第 13 天） | 直接调用装饰后的函数；配置启发式 `get_config` 就在同一个 Python 文件里（`:13-58`） |
| shape 特化 | 模板实参 + `SHAPE_M != 0 ? SHAPE_M : shape_m_runtime` 回退 | `T.const`（编译期）/ `T.dynamic`（运行期，`:87-89`） |
| 片上内存 | `constexpr l1_ptr/l0a_ptr/l0b_ptr/l0c_ptr` 按 offset 手工排布 + `static_assert` 容量检查（`bf16_gemm.hpp:59-69`） | `T.alloc_shared/l1/l0a/l0b/l0c` + `T.annotate_buffer_versions`；容量由 `get_config` 的整除计算保证（`:30-31`） |
| 流水线 | 手写 stage 下标轮转 + `asc_sync_wait/notify` 配对（`bf16_gemm.hpp:112, 116-121`） | `T.Pipelined` / `annotate_buffer_versions` / `multi_buffer_eligible`，重叠交给 AutoSchedule 安排 |
| AIC/AIV | `__mix__(1, 2)` 属性 + `ASCEND_IS_AIC` 分支 + `epilogue.hpp`（第 25 天） | `T.MixedKernel` + `(core_id, aiv_id)` + `T.PerCoreTask` + `T.dual_copy`（`:106, 269-272, 295, 327`） |
| NZ/分形布局 | `copy_gm_to_l1/copy_l1_to_l0a/b` 原语内部隐藏（`ascend/copy.hpp`） | 显式：`make_ascend_compact_nz_layout` + SIMD `vsstb` 散射打包（`:114, 237-250`） |
| 跨核同步 | 主线 GEMM 不需要；`kDirectStore` 用原子加（`bf16_gemm.hpp:69`） | `T.ascend_sync_inter_arrive/wait` + flag 编号；K-split 必须跨核（`:323-325, 355-357`） |
| L2 提示 | `asc_load/store_l2_cache_mode` 枚举、模板参 `kL2CtrlA/B/StoreCd` | 字符串 `'notalloc_keep'` / `'normal_fv'`（`:191, 348`） |
| 编译与缓存 | DeepJIT：`DG_JIT_*` 环境变量、`~/.dj` 缓存、bisheng 编译（第 12 天） | TileLang 自带 JIT；`pip` 依赖 `tilelang`（`setup.py:149`）；dry run 走 `.compile()`（`:394-397`） |
| 仓库内占比 | 主线全部（BF16/FP8/FP4、MQA、MegaMoE） | 唯一：mHC prenorm |
| 测试集成 | `tests/` + `bench_msprof` + `par_compile` | **完全相同的一套**（`test_hyperconnection.py` 与其他测试同构） |

**同构概念对偶表**（两边说的是同一件事）：

| 手写 C++ 侧 | TileLang 侧 |
| :--- | :--- |
| `BLOCK_K` 外层循环 × `MAD_K` 内层循环 | hidden block 循环（`T.Pipelined`，256 列）× `hidden_mad` 循环（64 列，`:155, 274`） |
| `is_first`（`bf16_gemm.hpp:134`） | `clear_accum=(local_hidden_block == 0 and hidden_mad == 0)`（`:287`） |
| `kNumL1Stages` 手动轮转下标 | `T.annotate_buffer_versions` 版本数 / `T.Pipelined`（`:123-132`） |
| `actual_m/n/k` 边界钳制 | `actual_num_tokens` / `num_valid_token_stages` / `pad_value=0`（`:139-153, 177-193`） |
| `__mix__(1, 2)` + `ASCEND_IS_AIC` | `T.MixedKernel` + `T.PerCoreTask` + `T.dual_copy` |
| `asc_set_atomic_add_float()` | `T.set_atomic('add', 'float32')`（`:135`） |
| `static_assert(l1b.offset(...) <= L1SizeBytes)` | `get_config` 的 `L0B/L0C 容量 // tile 大小`（`:30-31`） |
| `select_gemm_config`（heuristics.hpp） | `get_config`（`:13-58`） |

**一句话总结**：两种方式共享同一套硬件心智模型（分块、多级缓冲、两级 K 循环、边界钳制、原子归约、L2 提示），差异在**控制权的位置**——C++ 路线把每一步的调度、同步、地址都握在自己手里（代价是模板复杂度和一整条 C++ Host 链）；TileLang 路线把这些声明出来交给 AutoSchedule（代价是偶尔要顺着调度器的规矩改写代码，如 0/1-trip 循环，以及 NZ 打包这类本可由原语隐藏的细节要自己动手）。本仓库用实例给出了取舍：追求峰值与定制调度的主线 GEMM 家族全部手写；唯一需要"AIV 旁路融合"的 mHC 算子选了 TileLang。

## 四、源码导读

今天的主线文件与阅读站点（建议按顺序读，第一个建议逐行）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `deep_gemm/tilelang_ops/hc_prenorm.py` | `get_config` `:13-58`、`tf32_hc_prenorm_gemm_kernel` `:60-360`、`tf32_hc_prenorm_gemm` `:363-398` | batch invariance 注释 `:33-35`、相位/屏障注释 `:96-103`、缓冲分配 `:112-132`、原子+HF32 `:134-136`、SIMD 段 `:196-267`、dual_copy `:269-272`、`T.gemm` `:282-293`、两个 `PerCoreTask` `:295-357`、dry run `:394-397` |
| 2 | `tests/test_hyperconnection.py` | `CASE_SLICES` `:17-27`、`compile_hc_prenorm_gemm` `:31-56`、`test_hc_prenorm_gemm` `:59-98`、`test_hc_prenorm_gemm_deterministic` `:101-135`、`__main__` `:140-160` | 形状语义注释 `:14-16`、1e-6 阈值 `:76-77`、split 契约 `:79-84`、位级重复 `:121-124`、batch invariance 采样 `:126-132` |
| 3 | `deep_gemm/__init__.py`、`deep_gemm/tilelang_ops/__init__.py` | 导出 `:156-157` / `:1` | `tf32_hc_prenorm_gemm` 如何进入 `deep_gemm` 命名空间；`hc_prenorm.py:10` 的 `from .. import _C` 说明它仍复用 C++ 扩展的 `get_num_sms/get_deterministic_algorithms/get_dry_run` |
| 4 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | 模板声明 `:9-27`、缓冲规划 `:59-69`、`set_flags` `:83-84`、K 循环与同步 `:112-121`、`is_first` `:134` | 3.8 对比表的"另一边"；重点比对 `__mix__(1, 2)`、offset+`static_assert`、手动 stage 轮转 |
| 5 | `deep_gemm/include/deep_gemm/common.hpp` | `kNumAICores/L1/L0A/L0B/L0C/UBSizeBytes` `:50-58`、L2 枚举 `:120-131` | 硬件常量与 `hc_prenorm.py:17-18` 的一致性；`NOTALLOC_KEEP/NORMAL_FIRST_VICTIM` 与 TileLang 字符串的对应 |
| 6 | `deep_gemm/testing/par_compile.py`、`csrc/apis/runtime.hpp` | `dry_run` `par_compile.py:6-35`、`set/get_dry_run` `runtime.hpp:46-54` | dry run 的完整闭环 |
| 7 | `tests/utils.py` | `calc_diff` `:36-43`、`count_bytes` `:95-103`、`print_perf` `:106-115` | 3.1 口径公式与 3.7 阈值的出处 |
| 8 | `deep_gemm/testing/bench.py` | `KernelProfile` `:27-71`、`bench_msprof` `:284-400` | 第 46 天已读；今天只看 `kernel_names` 子串匹配与 `mad/mte2` 字段 |
| 9 | `README.md` | HC Prenorm 节 `:175-185`、依赖 `:28`、致谢 `:199` | 性能表（3.1 手算的验证对象）、"The mHC kernel is backed by Tilelang" |

**`hc_prenorm.py` 按执行顺序分块导读**：① `get_config`（`:13-58`，先自己做一遍 3.3 的手算再读）→ ② 声明区与缓冲（`:76-132`，对着 3.4 的表核验容量）→ ③ 主循环（`:137-293`，对着 3.4 的骨架和 3.5 的 SIMD 拆解）→ ④ 两个 `PerCoreTask`（`:295-357`，对着 3.6 的相位表，逐行读 flag 计算 `:302, :334` 和 `num_store_syncs` `:318-322`）→ ⑤ API（`:363-398`，注意 `num_splits` 断言与 dry run 分支）。**`test_hyperconnection.py`**：先 `:14-27` 理解形状，再 `:59-98` 主测试，然后 `:101-135` 确定性测试，最后回头看 `:31-56` 编译批次为什么要跑两轮。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **正确性与边界**（约 20 分钟）：

   ```bash
   cd DeepGEMM-Ascend && ./develop.sh   # 若尚未构建
   python tests/test_hyperconnection.py default --skip-prof
   ```

   观察三个现象：开头的 `Compiling` 进度条跑**两轮**（deterministic=False/True 各一批，`:49-51`）；m=13/137 的用例通过（不足一块 / 跨块边界，对应 3.4 的边界处理）；每个用例先后输出普通测试、split 契约和 `deterministic: ... > OK` 三段结果。若失败，按"环境 / 参考结果 / 配置 / Kernel"的顺序分层定位（第 31 天的方法）。

2. **性能对照 README**（约 15 分钟）：去掉 `--skip-prof` 重跑 default，抄下 k=28672 的 5 行 `us / TFLOPS / GB/s / mad% / mte2%`，与 README 表（`README.md:181-185`）和 3.1 的手算三方对照；用 `mad%`/`mte2%` 验证"访存受限"的判断。再跑 `python tests/test_hyperconnection.py peak` 观察 m=1 和 m=32768 两个极端（单 token / 多 chunk）。

3. **（进阶）单变量配置实验**（约 20 分钟）：`get_config` 只读 `_C.get_num_sms()`，可以不启动 Kernel 直接观察配置变化：

   ```python
   import deep_gemm
   from deep_gemm.tilelang_ops.hc_prenorm import get_config
   for cores in (32, 16, 8):
       deep_gemm.set_num_sms(cores)
       print(cores, get_config(28672, 24, 4096))
   ```

   解释 `num_k_splits` 随核数的变化（提示：`by_cores = cores & -cores`，16 核时 `min(16, 16, 16) = 16` 不变，8 核时被钳到 8）；再对 m=8192 比较两种确定性模式的配置差异，与 3.3 的表格互相印证。

4. **（可选）确定性观察**：把 `test_hc_prenorm_gemm_deterministic` 中 10 次重复改成打印 `d.view(torch.int32)[0, :4]`（临时改测试脚本，做完还原），确认每次位级相同；再临时把 `use_deterministic_algorithms(True)` 换成 False，观察非 det 模式下 m=8192、k=28672 的 split 数变为 8 后，单行 batch invariance 检查是否仍然通过（预期：普通写+原子加的顺序不受控，位级一致性无保证——这解释了 det 模式存在的意义）。

### B 组：无 Ascend 环境（静态任务）

1. **`get_config` 手算（当天核心练习）**：对 (m, n, k) = (13, 24, 28672)、(4096, 24, 28672)、(8192, 24, 7168)，分别算出 det 与非 det 两种模式的全部 9 个返回字段，与 3.3 的表格和源码 `:49-58` 对照。追加两问：k=7680 时 `num_k_splits_by_hidden` 是多少（30 & −30 = 2，两种模式都被钳到 2）？default 用例中哪些 (m, k) 组合的 det 与非 det 配置不同（m=8192 且 k∈{28672, 20480, 16384}）？
2. **容量对账**：n=24 时默写 3.4 的缓冲表（位置/形状/版本数/大小），核验 L1 ≤ 512KB、L0A/L0B ≤ 64KB、L0C ≤ 256KB、UB ≤ 256KB 四条不等式；再对 n=8（`block_mhc_mult3=16`）重算 `num_l0b_stages`（4）和 `num_l0c_stages`（32，`256KB // (128×16×4)`），体会 N 变小后槽位变多的权衡。
3. **相位/屏障演算**：设 `num_k_splits=4`、`num_token_chunks=2`、每个核 `num_valid_token_stages=3`。分别画出非 det（`num_store_phases=2`）与 det（`num_store_phases=4`）两个 chunk 的相位时间线：每个相位谁在写、写 `out_mul` 还是 `sqrsum`、FIX/MTE3 的 flag 各是多少（`(chunk × phases + phase) % 4` 与 `+4`）、屏障各发生几次（注意非 det 最后一个相位：`num_store_syncs = min(chunks - 1 - chunk, 1)`，chunk 1 时为 0）。对照 `:296-325` 检查。
4. **带宽口径手算**：复算 README 的 M=13 与 M=8192 两行（3.1 的两个算式）；然后回答：若不融合（sqrsum 单独一个 Kernel 再读一遍 A），M=8192 的总字节和带宽口径变成多少（约 943MB，同样时延下带宽口径翻倍——或同样带宽下时延翻倍），说明融合的量化收益。
5. **概念对偶默写**：合上 3.8 的对偶表，从 C++ 侧的 8 个概念（两层 K 循环、is_first、stage 轮转、actual_m/n/k、__mix__、原子加、容量断言、select_gemm_config）默写 TileLang 侧对应物并标注行号，再对照修正。
6. **写对比笔记（当天产出）**：以 3.8 为参考答案独立完成，并补充至少一条自己的观察（例如：TileLang 路线的 `get_config` 与 Kernel 同文件，改配置不用重新编译 C++ 扩展；或 NZ 打包负担从 copy 原语转移到了用户代码）。

## 六、当天产出

**两种 Kernel 开发方式对比笔记**——以 3.8 的两张表为参考答案，合上教程独立完成。主对比表至少覆盖九个维度（语言与文件组织 / Host 调用链 / shape 特化 / 内存规划 / 流水线 / AIC-AIV / NZ 布局 / 跨核同步 / 编译缓存），每个维度给出**两侧的代码证据**（文件:行号）；对偶表至少 8 对，能从任一侧推向另一侧。最后用三五句话回答取舍问题：为什么本仓库主线 GEMM 手写而 mHC 用 TileLang（提示：从"谁需要定制调度"和"谁需要声明式表达旁路融合"两个角度）。有环境的同学附上 A 组第 1、2 步的输出记录。

## 七、自测题

1. 写出 HC Prenorm 的两个输出公式和四个张量的形状/dtype；`num_splits` 为什么只接受 None 和 1？
   > `D = float(A) @ B.T`、`sqr_sum = sum(float(A)**2, dim=-1)`（`hc_prenorm.py:370`）；a `[M,K]` BF16、b `[N,K]` FP32、d `[M,N]` FP32、sqr_sum `[M]` FP32。本后端在 Kernel 内部完成 K split 的归约，外部只需一份结果（`:377-388`）。
2. sqr_sum 的计算为什么没有给 A 增加任何额外的 GM 读流量？
   > 它在 AIV 把 x 从 GM 搬进 UB 之后、于拓宽/打包的同一趟 SIMD 里完成（`:196-267`）；不融合则需要第二个 Kernel 再读一遍 A（M=8192 时约多 469.8MB）。
3. `get_config` 的三个 K-split 候选各依赖什么？deterministic 模式为什么把 workload 项固定为 16？
   > 依赖 token block 数（M）、hidden block 数的最低置位比特（K）、核数的最低置位比特（`:36-40`）。M 相关的 split 选择会改变每行浮点归约分组、破坏 batch invariance（`:33-35` 注释），det 模式固定 16 使 split 数只依赖 K 和核数。
4. n=24 时物理缓冲为什么按 32 列分配？Kernel 如何保证逻辑上只用 24 列？
   > `block_mhc_mult3 = ceil16(n)` 用于对齐物理缓冲（`:27`）；数据搬运与 MAD 区域仍按逻辑 `mhc_mult3` 截取（`:159-160` 注释、`:162-168, 284-285`）。
5. `out_mul_l0` 的 16 个槽是什么？token block 在同 split 内如何分配给各核？
   > L0C 里的 16 个 `[128, 32]` 输出 tile 槽（`256KB/16KB`，`:31, :121`）；`token_block = token_block_base + token_stage × cores_per_split`（`:175`），各核交错认领，一个 chunk 共 `cores_per_split × 16` 个 block（`:47`）。
6. `T.dual_copy` 与普通 `T.copy` 的区别是什么？
   > dual_copy 表达 AIV 的 UB → AIC 的 L1 这种跨计算单元的生产者-消费者搬运（`:269-272`）；普通 `T.copy` 描述同一处理单元视角下的搬运（如 GM→UB、L1→L0）。
7. 非 det 模式的两个 store 相位各做什么？屏障为什么必要？为什么最后一个相位的屏障是可选的？0/1-trip 循环解决什么问题？
   > phase 0 = split 0 普通写，phase 1 = 其余 split 原子加（`:96-98, :297-301`）；不加屏障，原子加可能被 phase 0 的普通写覆盖（`:323-325`）。下一个 chunk 复用 L0C/UB 槽和 flag，故仅其后需要屏障（`:98-99, :318-322`）。AutoSchedule 拒绝 `T.PerCoreTask` 内条件控制流下的跨核等待，所以把屏障包进次数为 0/1 的 `T.Serial` 循环（`:100-103`）。
8. det 模式靠什么同时做到位级可复现和 batch invariance？
   > 可复现：`num_store_phases = num_k_splits`，每个 split 在自己的相位写、相位间全核屏障，GM 累加顺序固定为 split 0,1,2,…（`:96, :298-300`）；batch invariance：split 数与 M 无关（workload 项固定 16，`:36-37`），单行与整批的归约分组一致。
9. 确定性测试如何采样 batch invariance？为什么用 `view(torch.int32)` 比较？
   > 取第 0、m//2、m-1 行单独重算，要求与整批结果位级一致（`:126-132`）；位级比较不能直接 `torch.equal` 浮点（-0.0/NaN 的表示问题），按位视图比较才是严格的位级相等（`:123-124` 同理）。
10. 默写五对本日讲过的概念对偶（C++ 侧 ↔ TileLang 侧）。
    > 例：`is_first` ↔ `clear_accum`；`kNumL1Stages` 手动轮转 ↔ `annotate_buffer_versions`/`T.Pipelined`；`__mix__(1,2)` ↔ `T.MixedKernel`+`PerCoreTask`+`dual_copy`；`asc_set_atomic_add_float` ↔ `T.set_atomic('add','float32')`；容量 `static_assert` ↔ `get_config` 整除计算（完整表见 3.8）。

---

明天（第 48 天）是扩展算子复盘日：从 MQA、MegaMoE、HC Prenorm 中选一个你最感兴趣的，不看笔记回答四个问题——输入输出和核心计算是什么、调度与寻址有什么特殊之处、为什么普通 GEMM 接口不能直接满足需求、性能瓶颈可能在哪里。今天读过的 `tf32_hc_prenorm_gemm`（一份搬运两种计算 + K-split 原子归约）和第 46 天的测量口径，都会是你分析"性能瓶颈在哪"时的现成工具。
