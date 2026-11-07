# 第 2 周 · 第 4 天（总第 11 天）：理解 GemmDesc

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 2 周"项目结构与 JIT 调用链"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 55 分钟，实践步骤约 45 分钟，自测题约 15 分钟。

第 10 天我们把 `deep_gemm.bf16_gemm_nn` 追到了 `launch_bf16_gemm` 的门口，并留下一句话：**这一步之后，Python 传来的信息全部收敛在 `GemmDesc` 一个结构体里**。今天就来打开这个结构体：逐字段分析 `csrc/apis/gemm.hpp` 的 `get_gemm_desc()`，弄清 M/N/K、major、stride、dtype、group 信息分别从哪里来、写进哪个字段、又被谁消费；同时对比 Normal、M-Grouped、K-Grouped 三种 GEMM 在 shape 检查上的差异。`GemmDesc` 之后的 `select_gemm_config`、模板代码生成与 JIT 编译启动属于第 12 天；SF 布局转换的细节属于第 33 天。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 `GemmDesc` 与 `GemmConfig` 的分工——"算什么"（问题描述）与"怎么算"（执行配置）——以及二者各自的定义位置和 `GemmArgs` 的作用；
2. 按 `get_gemm_desc()`（`csrc/apis/gemm.hpp:60-197`）的执行顺序复述七个阶段：major 判定 → shape 提取 → D/grouped_layout 检查 → `c` 语义与空问题早退 → SF 转换 → expected_m/k 估算 → 构造 `GemmDesc`；
3. 解释 `get_shape_by_spec` 的标注机制，并写出 Normal、Batched、M-Grouped、K-Grouped 四种类型各自的 `g/m/n/k` 分别来自哪些张量的哪些轴；
4. 解释 major 如何由 stride 推断（不是用户声明的）、`outer_stride_a/b/d` 的含义，以及为什么 M-Grouped 强制 A 为 K-major、K-Grouped 强制 A/B 为 MN-major；
5. 区分 `d_dtype` 与 `cd_dtype`、`acc` 与 bias、`expected_m` 与 `expected_k`、`grouped_layout` 在 m-grouped 与 k-grouped 下的不同语义（psum 前缀和）；
6. 说明 SF 在 `get_gemm_desc` 内部的转换入口（`csrc/apis/layout.hpp` 的 `get_recipe` / `transform_sf_into_required_layout`）以及 `GemmDesc::sfa/sfb` 持有张量引用的作用；
7. 用 `DG_PRINT_CONFIGS=1` 在真实运行中观察打印出来的 `GemmDesc`，或用手工演算在纸面上填出一份 `GemmDesc`，并与字段来源表互相验证。

**当天产出：一张 `GemmDesc` 字段来源表**（模板见第六节，参考答案见 3.10 节）。

## 二、预备知识

- **第 10 天的调用链**：binding 的五步（逻辑转置 → dtype 断言 → epilogue 归一 → `get_gemm_desc` → `launch_bf16_gemm`）；`nt/nn/tn/tt` 后缀与 `get_transposed` 的零拷贝转置；`c` 的三种语义（`None` / 与 `d` 同指针 / 独立 `[1,N]`）。今天在这些结论之上展开 `get_gemm_desc` 内部。
- **第 8～9 天**：`_C` 扩展由 `csrc/python_api.cpp` 编译；`csrc/apis/`、`csrc/utils/`、`csrc/jit_kernels/` 都是头文件；`DJ_HOST_ASSERT`/`DJ_PANIC` 失败在 Python 侧表现为 `RuntimeError`。
- **第 1 周概念**：M/N/K 与 `D = A × B`（第 1 天）；stride、Row-major/Column-major、逻辑转置不搬数据（第 2 天）；BF16/FP8/FP4 与 Scaling Factor（第 4 天）；`GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` 数据流（第 5 天）。
- **C++ 语法**：指定初始化器（`.field = value`）、`std::optional`、`std::tie`、`switch` 于 `enum class`。能对照读懂即可。

## 三、详细讲解

### 3.1 `GemmDesc` 与 `GemmConfig`：一对结构体

两个结构体都定义在 `csrc/jit_kernels/gemm_config.hpp`：

- **`GemmDesc`（第 39～95 行）**——**问题描述：算什么**。维度、dtype、指针、布局（major/stride）、分组信息、累加与 epilogue。它由 `get_gemm_desc()` 从用户张量**提取并校验**得到，不含任何"如何切 Tile、几级流水"的决策。
- **`GemmConfig`（第 97～121 行）**——**执行配置：怎么算**。`block_m/n/k`、`mad_m/n/k`、`num_l1_stages` 等流水深度、L2 控制和 launch 选项。它由 `select_gemm_config(desc)`（`csrc/jit_kernels/heuristics.hpp`）**根据 desc 启发式推导**得到——第 12 天的主角。
- `GemmArgs`（第 124～127 行）只是 `GemmDesc desc + GemmConfig config` 的打包。

调用链中的位置：

```text
binding (bf16_gemm / fp8_fp4_gemm / *_grouped_* / einsum)
    │  逻辑转置、dtype 断言、epilogue 归一
    ▼
get_gemm_desc()  ──►  std::optional<GemmDesc>     ← 今天
    │  （nullopt = 空问题，直接返回）
    ├─ dispatch_fp8_fp4_gemm：按 a_dtype/b_dtype 选 kernel 链   ── 第 12 天
    └─ launch_bf16_gemm → select_gemm_config(desc) → GemmConfig
                          → 生成 bf16_gemm_impl<...> 模板代码 → jit->compile/launch
```

`GemmDesc` 的字段可分六组（行号见 3.10 的来源表）：**指针与 dtype**（`a_ptr/b_ptr/d_ptr`、`a_dtype/b_dtype/d_dtype/cd_dtype`、`sfa_ptr/sfb_ptr`）、**形状**（`m/n/k`、`num_cores`）、**布局**（`major_a/major_b`、`outer_stride_a/b/d`、`stride_batch_a/b/d`）、**累加与 epilogue**（`acc`、`epilogue_class`、`compiled_dims`）、**分组扩展**（`gemm_type`、`grouped_layout`、`num_groups`、`expected_m/expected_k`）、**生命周期**（`sfa/sfb` 持有转换后的 SF 张量）。

### 3.2 `get_gemm_desc` 的七个阶段

`get_gemm_desc()`（`csrc/apis/gemm.hpp:60-197`）是所有 GEMM binding 共用的校验与描述构造函数，返回 `std::optional<GemmDesc>`。按执行顺序：

| 阶段 | 行号 | 做什么 |
| :--- | :--- | :--- |
| ① major 判定 | 79-80 | `get_major_type_ab(a/b)` 从 stride 推断 `Major::K` / `Major::MN` |
| ② shape 提取 | 81-97 | 按 `gemm_type` 走 `switch`，用 `get_shape_by_spec` 交叉校验出 `g/m/n/k`，并附带各类型的 major 断言 |
| ③ 布局检查 | 99-103 | `check_major_type_cd(d)`（D 必须 N-major）；`grouped_layout` 必须连续且为 `kInt` |
| ④ `c` 语义与早退 | 105-123 | 判定累加 / bias / 无；`gemm_early_return` 处理空问题；bias 时包装 `BiasEpilogue` |
| ⑤ SF 转换 | 125-154 | `layout::get_recipe` 合并粒度参数；`transform_sf_into_required_layout` 把 SF 重排成 kernel 需要的布局 |
| ⑥ 估算提示 | 156-163 | `cd = c.value_or(d)`；m-grouped 估 `estimated_m = m/g`，k-grouped 估 `estimated_k = k/g` |
| ⑦ 构造 | 164-196 | 用指定初始化器逐字段填出 `GemmDesc` |

两个全局性事实先记住：

- **nullopt = 没事可做**。`gemm_early_return` 对 `m==0`、`n==0`、`k==0` 的空问题直接放行返回，binding 拿到空 optional 就 `return`（`gemm.hpp:249/319/382/446/494/548`），不构造 desc、不触发 JIT。
- **所有 shape/major/stride 检查都集中在这里**，六个 binding（bf16、fp8 家族、m-grouped、k-grouped）+ einsum 复用同一套逻辑——这是"薄 Python、厚 `get_gemm_desc`"设计的延续（第 10 天思考题）。

### 3.3 M/N/K 从哪里来：`get_shape_by_spec` 与四种 gemm_type

机制回顾（第 10 天 3.4 节初见）：`csrc/utils/layout.hpp:140-219` 的 `get_shape_by_spec<"输出字母">("标注"_sp = 张量, ...)` 给每个张量的每一维标注逻辑含义，再按字母**跨张量交叉校验**同名维度：同一字母在任何两个张量上取值不同就 `DJ_PANIC("Conflicting shape dimension ...")`（`utils/layout.hpp:195-203`）。函数开头的 doc 注释（`gemm.hpp:45-58`）给出四种类型的形状表：

```text
 *                A                 B                CD
 *
 * Normal:       [M, K]            [N, K]            [M, N]
 * MGrouped:     [M, K]         [G, N, K]            [M, N]
 * KGrouped(*):  [M, K]            [N, K]         [G, M, N]
 * Batched:   [B, M, K]         [B, N, K]         [B, M, N]
```

`switch` 的四个分支（`gemm.hpp:81-97`）：

| gemm_type | 模板参数 | 各张量标注 | g 的来源 | m / n / k 的来源 |
| :--- | :--- | :--- | :--- | :--- |
| Normal | `<"mnk">` | A=`"mk"`、B=`"nk"`、D=`"mn"` | 无（g=0） | m：A[0]、D[0]；n：B[0]、D[1]；k：A[1]、B[1] |
| Batched | `<"bmnk">` | A=`"bmk"`、B=`"bnk"`、D=`"bmn"` | A[0]、B[0]、D[0] | m：A[1]、D[1]；n：B[1]、D[2]；k：A[2]、B[2] |
| MGrouped | `<"gmnk">` | A=`"mk"`、B=`"gnk"`、D=`"mn"`、layout=`"g"` | B[0] 与 layout 的长度 | m：A[0]、D[0]；n：B[1]、D[1]；k：A[1]、B[2] |
| KGrouped | `<"gmnk">` | A=`"mk"`、B=`"nk"`、D=`"gmn"`、layout=`"g"` | D[0] 与 layout 的长度 | m：A[0]、D[1]；n：B[0]、D[2]；k：A[1]、B[1] |

三个要点：

1. **`grouped_layout` 本身参与 shape 校验**。它以 `"g"` 标注进入 spec，其长度必须与 B[0]（m-grouped）或 D[0]（k-grouped）一致——组数不是单独传参，而是从张量形状**推导并核对**出来的。
2. **KGrouped 的 `(*)`：进函数前先"假转置"**。k-grouped 的 binding（`gemm.hpp:366-367、529-530`）先对 `[K, M]`/`[K, N]` 的用户张量做 `transpose(0, 1)`，注释写明"只为拿到正确的 shape/stride，不搬数据"。于是 `get_gemm_desc` 里 A/B 一律是规范形式 `[M, K]`/`[N, K]`，四种类型共用同一套 spec 逻辑。
3. **FP4 打包的尺寸翻倍**。两个 FP4 挤一个字节（`kPackedFP4 = torch::kInt8`，`csrc/utils/dtype.hpp:9`），`get_shape_by_spec` 在扫描时对 FP4 张量的 **major 轴**尺寸乘 2（`utils/layout.hpp:185-187`）——逻辑元素数与物理存储量的换算就藏在这一行。

### 3.4 major 与 outer stride：从 stride 推断布局

**major 不是用户声明的，是推断的**（第 10 天已见，今天补全去消费侧的链路）：

- `get_major_type_ab`（`utils/layout.hpp:44-47`）：`stride(-1)==1 → Major::K`（规范形式 `[MN, K]` 的 K 轴连续）；否则 `stride(-2)==1 → Major::MN`（MN 轴连续）。两轴都连续会被 `major_check`（第 37-42 行）判为"ambiguous major"并 panic。
- `check_major_type_cd`（第 49-52 行）：**D 永远要求 N-major**（`stride(-1)==1`，且 `dim>=2`）——分组/批量 D（`[G,M,N]`、`[B,M,N]`）也只看最后两维。

**outer stride 是"外层轴的行距"**，由 `get_outer_stride`（`gemm_config.hpp:27-34`）写入：

- K-major 操作数：断言 `stride(-1)==1`，取 `stride(-2)`（沿 MN 的行距）；
- MN-major 操作数：断言 `stride(-2)==1`，取 `stride(-1)`（沿 K 的行距）；
- D 固定按 `Major::K` 取 `stride(-2)`（沿 M 的行距）。

单位是**元素数**。紧凑输入下它们分别等于 `k/k/n`（A/B/D），从而复现旧的硬编码行为；strided 或 permute 过的操作数则携带真实 stride（`gemm_config.hpp:68-75` 注释）。这些值最终在 `launch_bf16_gemm` 里被装进 `gm_ptr`（`bf16_gemm.hpp:73-77`）：`gm_ptr<uint8_t>{desc.outer_stride_a, a_ptr, stride_batch_a}`——设备侧 `gm_ptr::index()`（`deep_gemm/include/deep_gemm/common.hpp:156-165`）用 `mn_idx * stride_outer + k_idx`（K-major）或 `k_idx * stride_outer + mn_idx`（MN-major）完成寻址。**K-major 与 MN-major 的差异最终体现为这一句地址计算的不同**。

**四种 gemm_type 的 major 断言差异**（`gemm.hpp:90、94-95`）：

| gemm_type | 对 A 的要求 | 对 B 的要求 |
| :--- | :--- | :--- |
| Normal / Batched | K 或 MN 均可 | K 或 MN 均可 |
| MGrouped | **必须 `Major::K`** | K 或 MN 均可 |
| KGrouped | **必须 `Major::MN`** | **必须 `Major::MN`** |

直觉（严格论证留到第 17～19 天的 scheduler/tile_ptr/copy）：M-grouped 的 A 是变长行段的拼接，行内 K 连续（K-major）才能按行高效搬运；k-grouped 沿 K 分组、每组输出独立 `[M,N]`，要求 K 是外层步进轴（MN 轴连续）。今天只需记住：**这些断言会在 binding 层直接拒绝违规输入**（测试侧对应 `tests/generators.py:332、372-373` 的生成约束）。

### 3.5 dtype 字段：`a_dtype/b_dtype/d_dtype/cd_dtype`

四个 dtype 字段的来源与分工：

| 字段 | 来源 | 消费者 |
| :--- | :--- | :--- |
| `a_dtype` / `b_dtype` | `a/b.scalar_type()`（`gemm.hpp:168-169`） | `dispatch_fp8_fp4_gemm`（`gemm.hpp:22-33`）按 `(a_dtype, b_dtype)` 三路分派：fp8×fp8 → `launch_fp8_gemm`；fp4×fp4 → `launch_fp4_gemm`；fp8×fp4 → `launch_fp8_dequant_gemm` |
| `d_dtype` | `cd.scalar_type()`，其中 `cd = c.value_or(d)`（`gemm.hpp:156、170`） | 记录参与输出的 C/D 数据 dtype |
| `cd_dtype` | `d.scalar_type()`（`gemm.hpp:177`） | kernel 模板的输出类型：`launch_bf16_gemm` 用 `GemmDesc::dtype_to_cd_kernel_type(desc.cd_dtype)` 生成 `"bfloat16_t"/"float"`（`gemm_config.hpp:87-94`、`bf16_gemm.hpp:63`） |

两个细节：

- 累加/bias 时 `gemm_early_return` 已断言 `c` 与 `d` 同 dtype（`utils/layout.hpp:71`），所以 `d_dtype` 与 `cd_dtype` 实际总是相等；二者并存是把"数据是什么"和"kernel 模板实例化成什么"分开表达。
- `dtype_to_kernel_type`（`gemm_config.hpp:79-85`）把 `kFloat8_e4m3fn`/`kPackedFP4` 映射为设备侧类型名 `"float8_e4m3_t"`/`"float4_e2m1x2_t"`——dtype 到 JIT 模板参数的桥梁。

### 3.6 group 信息：`gemm_type`、`grouped_layout`、`num_groups`、`expected_m/k`

**`gemm_type`** 是 `enum class GemmType : uint8_t { Normal, MGroupedContiguousWithPsumLayout, KGroupedContiguousWithPsumLayout, Batched }`（`deep_gemm/include/deep_gemm/common.hpp:87-92`），配套谓词 `is_m_grouped/is_k_grouped/is_grouped/is_batched`（第 105-116 行）。它由 binding 指定（不是推断的）：Normal 的 binding 传 `GemmType::Normal`，m-grouped 传 `MGroupedContiguousWithPsumLayout`，以此类推。设备侧 scheduler、host 侧 heuristics 都靠它分支。

**`grouped_layout`：psum 前缀和**。字段注释（`gemm_config.hpp:56-57`）写得很清楚：`int32[num_groups]`，m-grouped 存**每组的结束行号**（prefix-sum-of-rows），k-grouped 存**每组累计结束的 K**。它描述的是"组边界在哪里"，而非"每组多长"。测试侧的生成逻辑（`tests/generators.py:212-220`）：

```python
psum = 0
for group_idx in range(num_groups):
    psum_layout.append(psum + size)          # 第 i 组的结束位置（未对齐）
    psum = aligned(psum + size, alignment)   # 下一组从对齐位置开始
```

组间对齐为 **256**（`Runtime::kMkAlignment`，`csrc/jit/jit.hpp:59`）。这意味着两件事：

1. **`m`（m-grouped）和 `k`（k-grouped）是含 padding 的总量**。m-grouped 的 A 形状是 `[total_m, K]`（`generators.py:328、334`），有效行区间由 `grouped_layout` 描述，组间是 padding；`ensure_zero_padding=True` 时输入 padding 为零则输出 padding 也为零（`gemm.hpp:286-287` doc 注释）。
2. **D 的形状不同**：m-grouped 的 D 是 `[M, N]`（各组输出拼在行轴上）；k-grouped 的 D 是 `[G, M, N]`（每组一个输出）。

**`num_groups`** 就是 spec 提取出的 `g`；Batched 类型复用它存**批数**（`gemm_config.hpp:62-66` 注释），并配套 `stride_batch_a/b/d`（仅 Batched 时取 `a/b/d.stride(0)`，否则 0，`gemm.hpp:188-190`）——因为 `.permute()` 之后批维 stride 不再是 `M*K/N*K/M*N`，必须在运行期携带。

**`expected_m` / `expected_k`：给启发式的"预估提示"**，不参与正确性：

- 来源：`expected_m.value_or(estimated_m)`、`expected_k.value_or(estimated_k)`（`gemm.hpp:186-187`）。`estimated_m` 在 m-grouped 时取 `m/g`、`estimated_k` 在 k-grouped 时取 `k/g`（第 157-163 行）。
- k-grouped binding 若拿到 `ks_cpu`（每组真实 K 的 CPU 列表），会先经 `expected_k_from_ks`（`gemm.hpp:35-43`）取**平均值**（下限 1）作为提示（第 373、537 行）——因为含 padding 的 `k` 会高估真实计算量。
- 消费：heuristics 多处用 `expected_k` 估算 k-grouped 的成本与 resident 判定（如 `heuristics.hpp:158、545`）；`expected_m` 目前主要出现在 `DG_PRINT_CONFIGS` 的打印输出里（`gemm_config.hpp:153`）。注意两种 m-grouped binding 的透传方式略有差异：bf16 版直接透传 optional（未提示时落到 `m/g`），fp8 版用 `value_or(0u)`（未提示时记 0）。

### 3.7 `c` 的三种语义与空问题早退（第 10 天的回顾 + k-grouped 补充）

`gemm.hpp:105-123` 与 `utils/layout.hpp:54-86`：

- **`is_cd_same`**（第 106 行）：`c` 与 `d` 同指针（meta/空张量用 `is_alias_of` 比较存储身份——测试的并行编译路径需要它）→ 累加模式。
- **`with_bias`**（第 107 行）：仅 `gemm_type == Normal` 且 `c` 与 `d` 不同指针；随后断言 `c` 是二维、形状 `[1, N]`（bias）或 `[M, N]`（与 d 同指针的累加）、与 `d` 同 device（第 108-112 行），且 bias 必须连续、不能与 alpha 同用（第 113-117 行）→ 包装成 `BiasEpilogue`（第 122-123 行）。
- **`acc = c.has_value() && !with_bias`**（第 180 行）：k-grouped 的 `c` 是 `[G, M, N]`，与 `d` 同指针时 `acc=true`，语义 `D[g] += A_g @ B_g`；m-grouped 的 binding 干脆不传 `c`（`gemm.hpp:313、488` 都给 `std::nullopt`），测试也断言 m-grouped 不支持累加（`tests/common.py:56`、`generators.py:330`）。
- **`gemm_early_return`**（`utils/layout.hpp:54-86`）：`m==0/n==0` 直接放行；`k==0` 时把 D 清零（或从 C 拷贝）后返回；"legacy 累加路径"在此处先把 `d.copy_(c)`（第 83-84 行）——注意这个副作用发生在 `get_gemm_desc` 内部，容易忽略。

### 3.8 SF 的转换：`get_gemm_desc` 内嵌的 layout 环节

低精度 GEMM 的 `(数据, SF)` 对里，SF 要先变成 kernel 需要的布局（MN-major、两个 ue8m0 沿 K 打包进一个 `int16`，见仓库 README 第 56 行的 NOTE）。这个转换就发生在 `get_gemm_desc` 内（`gemm.hpp:125-154`），入口在 `csrc/apis/layout.hpp`：

- **`get_recipe`**（`apis/layout.hpp:113-128`）：`recipe`（`(gran_m, gran_n, gran_k)` 三元组）与 `recipe_a`/`recipe_b`（各 `(gran_mn, gran_k)`）互斥；缺省 `recipe_a = recipe_b = (1, 32)`；两 operand 的 `gran_k` 必须相同。归一成统一三元组。
- **`transform_sf_into_required_layout`**（第 68-92 行）/ **`transform_k_grouped_sf_into_required_layout`**（第 94-111 行）：先用 `check_sf_layout`（第 37-65 行）判断输入是否已经是所需布局（short、`gran_mn==1`、MN-major 连续、形状 `[..., ceil(mn/gran_mn), ceil(k/gran_k/2)]`，且 `gran_k` 必须为 32）——**是则原样返回，零开销**；否则调 `launch_transform_sf` 起 kernel 重排。fp32 SF 必须允许转 ue8m0（`disable_ue8m0_cast=true` 时断言失败，第 87 行）。
- sfa 用 `(m, k)` 与 A 对应、sfb 用 `(n, k)` 与 B 对应（`gemm.hpp:132-138、145-152`）；k-grouped 走专用变体并携带 `grouped_layout`。

转换结果写进 `sfa_ptr/sfb_ptr`（`gemm.hpp:171-172`），同时**整张张量保存在 `GemmDesc::sfa/sfb` 里**（第 194-195 行）——注释（`gemm_config.hpp:77-78`）说明原因：防止 kernel 启动前张量被释放。这是"裸指针 + 保活引用"的标准配对。细节留到第 33 天。

### 3.9 Normal / M-Grouped / K-Grouped 的 shape 检查差异总表

| 检查项 | Normal | M-Grouped | K-Grouped | Batched |
| :--- | :--- | :--- | :--- | :--- |
| A 进入时的形状 | `[M, K]` | `[M, K]`（M 含组间 padding） | `[M, K]`（由 `[K, M]` 逻辑转置而来） | `[B, M, K]` |
| B 进入时的形状 | `[N, K]` | `[G, N, K]` | `[N, K]`（由 `[K, N]` 逻辑转置而来） | `[B, N, K]` |
| D 形状 | `[M, N]` | `[M, N]` | `[G, M, N]` | `[B, M, N]` |
| `grouped_layout` | 不需要 | 必需 `[G]` int32（连续） | 必需 `[G]` int32（连续） | 不需要 |
| g 的交叉校验 | — | `B[0]` vs layout 长度 | `D[0]` vs layout 长度 | `A[0]/B[0]/D[0]` 互相 |
| major 硬约束 | 无 | `major_a == K` | `major_a == MN` 且 `major_b == MN` | 无 |
| `c` 的支持 | 无 / 累加 / bias | 不支持（binding 传 nullopt） | 无 / 累加（`[G,M,N]`） | 无 / 累加 |
| `stride_batch_*` | 0 | 0 | 0 | `a/b/d.stride(0)` |
| `expected_m/k` 默认 | `m` / `k` | `m/g` / `k` | `m` / 平均 `ks_cpu` 或 `k/g` | `m` / `k` |
| binding 默认 `compiled_dims` | nn/nt `"nk"`，tn/tt `"mn"` | `"nk"` | `"mn"` | — |

一张图记住两种 grouped 的几何含义（数字取自第五节 B-2/B-3 演算）：

```text
M-grouped（G=2，sizes=[300,512]，对齐 256）——组切在 M 轴上，输出拼回 [M,N]：
 A 的行轴 total_m=1024：
 |<- g0: 300 行 ->|<- pad 212 ->|<- g1: 512 行 ->|
 0               300            512               1024
 grouped_layout = [300, 1024]（各组结束行号）

K-grouped（G=2，ks=[128,300]，对齐 256）——组切在 K 轴上，每组独立输出 [M,N]：
 K 轴 total_k=768：
 |<- g0: 128 ->|<- pad 128 ->|<- g1: 300 ->|<- pad 212 ->|
 0            128            256             556          768
 grouped_layout = [128, 556]（各组累计结束 K）；D = [G, M, N]
```

### 3.10 `GemmDesc` 字段来源表（当天产出·参考答案）

| 字段（类型） | 写入位置 | 值来源 | 说明 / 主要消费者 |
| :--- | :--- | :--- | :--- |
| `a_ptr`、`b_ptr`、`d_ptr`（void*） | gemm.hpp:165-167 | `a/b/d.data_ptr()` | GM 数据裸指针；launch 时装进 `gm_ptr`（bf16_gemm.hpp:73-77） |
| `a_dtype`、`b_dtype` | :168-169 | `a/b.scalar_type()` | `dispatch_fp8_fp4_gemm` 三路分派依据（gemm.hpp:22-33） |
| `d_dtype` | :170 | `cd.scalar_type()`（`cd = c.value_or(d)`，:156） | C/D 数据 dtype（与 `cd_dtype` 实际相等） |
| `sfa_ptr`、`sfb_ptr`（void*） | :171-172 | 转换后 SF 的 `data_ptr()`，无 SF 则 `nullptr` | kernel 读取的 SF 地址；BF16 恒为 `nullptr` |
| `m`、`n`、`k`（uint32_t） | :173-175 | `get_shape_by_spec` 交叉校验（:81-97） | 逻辑问题规模；m-grouped 的 m、k-grouped 的 k 含 padding |
| `num_cores`（uint32_t） | :176 | `runtime->get_num_sms()`（jit.hpp:34-36） | 可用 AI Core 数（沿用上游 sms 命名）；heuristics 估并行度 |
| `cd_dtype` | :177 | `d.scalar_type()` | kernel 输出类型：`dtype_to_cd_kernel_type` → `"bfloat16_t"/"float"` |
| `major_a`、`major_b`（Major） | :178-179 | `get_major_type_ab(a/b)`（由 stride 推断，utils/layout.hpp:44-47） | 决定设备侧 `gm_ptr::index()` 的寻址公式 |
| `acc`（bool） | :180 | `c.has_value() && !with_bias` | 累加 `D += A@B^T`；launch 生成模板参数 `desc.acc`（bf16_gemm.hpp:61） |
| `epilogue_class` | :181 | `resolve_epilogue_class` + bias 时换 `BiasEpilogue`（:122-123） | 提供设备侧算子名与运行期参数（alpha/bias 指针） |
| `compiled_dims`（string） | :182 | binding 透传 | 哪些维度编译期常量化：`get_compiled_dim`（gemm_config.hpp:17-21） |
| `gemm_type`（GemmType） | :183 | binding 指定 | Normal/MGrouped/KGrouped/Batched；scheduler 与 heuristics 的总开关 |
| `grouped_layout`（void*） | :184 | `grouped_layout->data_ptr()`，无则 `nullptr` | psum 前缀和：m-grouped=各组结束行号，k-grouped=各组累计结束 K |
| `num_groups`（uint32_t） | :185 | spec 提取的 `g` | 组数；Batched 时为批数 |
| `expected_m`（uint32_t） | :186 | 提示参数或 `estimated_m`（m-grouped 为 `m/g`） | 动态分组 M 提示；目前主要见于 `DG_PRINT_CONFIGS` 输出 |
| `expected_k`（uint32_t） | :187 | 提示参数或 `estimated_k`（k-grouped 为平均 ks 或 `k/g`） | heuristics 估算 k-grouped 成本（heuristics.hpp:158、545 等） |
| `stride_batch_a/b/d`（uint64_t） | :188-190 | 仅 Batched：`a/b/d.stride(0)`，否则 0 | 批间跳步（元素数）；permute 后不等于 M*K 等 |
| `outer_stride_a/b`（uint64_t） | :191-192 | `get_outer_stride(a/b, major)`（gemm_config.hpp:27-34） | 外层轴行距：K-major 取 `stride(-2)`，MN-major 取 `stride(-1)` |
| `outer_stride_d`（uint64_t） | :193 | `get_outer_stride(d, Major::K)` | D 恒 N-major，取 `stride(-2)`（沿 M 行距） |
| `sfa`、`sfb`（optional\<Tensor\>） | :194-195 | 转换后的 SF 张量本体 | 保活引用，防止 kernel 启动前被释放（gemm_config.hpp:77-78） |

打印验证：`std::formatter<GemmDesc>`（`gemm_config.hpp:145-161`）在 `DG_PRINT_CONFIGS=1` 或 `DG_JIT_DEBUG=1` 时由 `select_gemm_config` 末尾输出（`heuristics.hpp:749-753`，按行去重）：

```text
GemmDesc(m=128, n=64, k=256, cd_dtype=bfloat16_t, major_a=Major::K, major_b=Major::K,
         acc=false, epilogue_type=epilogue::operators::Identity, compiled_dims='nk',
         gemm_type=GemmType::Normal) GemmConfig(block=...)
```

（枚举以 `Major::K` 这类 `类型::名字` 形式打印，来自 `csrc/utils/format.hpp:11-20` 的 magic_enum formatter。）

## 四、源码导读

今天的目标是**把 3.10 表的每一行落到真实代码上**。每站 5～10 分钟。

### 站点 1：`csrc/jit_kernels/gemm_config.hpp`（174 行，先读）

- 第 36-38 行注释点题："Problem descriptor: what to compute"；
- `get_compiled_dim`（17-21）：`compiled_dims` 里有该字母则返回维度值，否则 0 表示运行期取值；
- `get_outer_stride`（27-34）：两种 major 各取哪根 stride、各断言什么；
- `GemmDesc`（39-95）逐字段读注释：`sfa_ptr` "FP8 only; nullptr for BF16"（44-45）、`acc`（50）、分组扩展（54-60）、batch 扩展（62-66）、outer stride（68-75）、`sfa/sfb` 保活（77-78）；
- `GemmConfig`（97-121）与 `GemmArgs`（124-127）今天只认脸，明天精读；
- 两个 `std::formatter`（145-172）：打印格式与 3.10 的示例对应。

### 站点 2：`csrc/apis/gemm.hpp` 的 `get_gemm_desc`（60-197，主战场）

按 3.2 的七阶段分段读：

1. 第 45-58 行 doc 注释的四类型形状表 + `(*)` 假转置说明；
2. 第 78-80 行：`g/m/n/k` 初始化与两次 `get_major_type_ab`；
3. 第 81-97 行 `switch`：对照 3.3 表逐分支写出每个字母的来源，并找到三条 major 断言；
4. 第 99-103 行：`check_major_type_cd` 与 grouped_layout 的连续性/int32 检查；
5. 第 105-123 行：`is_cd_same`（meta/空张量的 `is_alias_of` 分支）、`with_bias`、c 形状检查、`gemm_early_return` 调用、`BiasEpilogue` 包装；
6. 第 125-154 行：`get_recipe` 与 sfa/sfb 的两条转换路径（普通 / k-grouped 专用）；
7. 第 156-163 行：`cd`、`estimated_m/estimated_k`；
8. 第 164-196 行：指定初始化器列表——**逐行对照 3.10 表**，数一数你能不看表格说出几个字段的来源。

再快速扫四个调用方，确认"同一函数、四种 gemm_type"：`fp8_fp4_gemm`（242-248，Normal）、`m_grouped_fp8_fp4_gemm_contiguous`（311-318）、`k_grouped_fp8_gemm_tn_contiguous`（374-381，注意 366-368 的假转置与提前的 shape 提取）、`bf16_gemm`（440-445）与 `k_grouped_bf16_gemm_tn_contiguous`（538-547）。

### 站点 3：`csrc/utils/layout.hpp`（221 行）

- `get_shape_by_spec`（140-219）：`main_axis`（142-148）找 stride=1 的轴；`check_spec`（151-161）校验维数与 major 轴存在性；`get_one_dim`（169-212）的冲突检测与 FP4 翻倍（185-187）；
- `major_check`/`get_major_type_ab`/`check_major_type_cd`（37-52）；
- `gemm_early_return`（54-86）：注意第 83-84 行的 `d.copy_(c)` 副作用。

### 站点 4：`deep_gemm/include/deep_gemm/common.hpp`（188 行，host/device 共享）

- `Major`（82）与 `GemmType`（87-92）的定义位置——**这是 host 与 device 共用的头**，所以 desc 的枚举能直接进设备侧模板参数；
- `gm_ptr`（137-186）：`index()`（156-165）两种 major 的地址公式、`group()`（167-172）按 `group_idx * shape_mn * shape_k` 跳组、`batch()`（181-185）用 `stride_batch`——`outer_stride_*` 与 `stride_batch_*` 的最终归宿。

### 站点 5：`csrc/apis/layout.hpp`（142 行，只读 SF 入口）

- `check_sf_layout`（37-65）：所需布局的判据（dtype、维数、major、形状、`gran_k == 32`）；
- `transform_sf_into_required_layout`（68-92）：已是所需布局则原样返回；fp32 与 `disable_ue8m0_cast` 的互斥断言（87）；
- `get_recipe`（113-128）：recipe 合并与缺省值。
- 今天不看 `csrc/jit_kernels/transform_sf.hpp`（第 33 天）。

### 站点 6：消费者视角 `csrc/jit_kernels/bf16_gemm.hpp`（78 行，只读 16-78）

- 第 17 行 `select_gemm_config(desc)`：desc 驱动配置选择（明天展开）；
- 第 43-66 行 `std::format` 参数表：`desc.major_a/major_b`、`get_compiled_dim(desc.m/n/k, ...)`、`desc.acc`、`desc.gemm_type`、`dtype_to_cd_kernel_type(desc.cd_dtype)`、`desc.epilogue_class->get_epilogue_operator_type()`——**desc 的哪些字段进了模板参数**；
- 第 73-77 行 `jit->launch`：三个 `gm_ptr{outer_stride, ptr, stride_batch}` 与 `m/n/k/grouped_layout/num_groups`——**哪些字段进了运行期参数**。

### 站点 7：`csrc/jit/jit.hpp`（96 行，只看 Runtime）

- `kMkAlignment = 256`（59）与 `get_mk_alignment_for_contiguous_layout`（61-63）——组间对齐常量的出处；
- `runtime` 全局（76）与 `get_num_sms`（34-36）——`num_cores` 字段的来源。

### 站点 8：测试视角（psum 与形状的官方示范）

- `tests/generators.py:212-220` `generate_psum_layout`：前缀和与对齐的生成逻辑；
- `_generate_m_grouped_contiguous`（326-337）：A `[total_m, K]`（K-major 断言 332）、B `[G, N, K]`；
- `_generate_k_grouped_contiguous`（361-377）：A/B `[total_k, M/N]`（MN-major 断言 372-373）；
- `tests/common.py:49-84` `gemm_dg`：m-grouped 分支（67-73，B 在 1/2 维转置）与 k-grouped 分支（74-82，传 `ks_cpu` 与 `g_layout`）。

**阅读检查点**（能回答再进入实践）：

1. `GemmDesc` 有哪些字段只对 grouped 类型非默认？哪些只对 Batched 非默认？
2. M-grouped 分支里 `g` 的两个来源是什么？不一致时走哪一行 panic？
3. `get_outer_stride` 对 MN-major 操作数断言什么、返回什么？D 为什么固定按 `Major::K` 取？
4. `estimated_m` 在什么条件下等于 `m/g`？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：用 `DG_PRINT_CONFIGS=1` 观察真实 `GemmDesc`（约 15 分钟）**。

```bash
DG_PRINT_CONFIGS=1 python - <<'EOF'
import torch, torch_npu  # noqa
import deep_gemm

M, N, K = 128, 64, 256
a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')
b = torch.randn(N, K, dtype=torch.bfloat16, device='npu')
d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')

deep_gemm.bf16_gemm_nt(a, b, d)                 # ① 无 c
torch.npu.synchronize()
d.copy_(torch.randn(M, N, dtype=torch.bfloat16, device='npu'))
deep_gemm.bf16_gemm_nt(a, b, d, c=d)            # ② c=d 同指针 → acc=true
torch.npu.synchronize()
bias = torch.randn(1, N, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d, c=bias)         # ③ [1,N] → AddBias
torch.npu.synchronize()
EOF
```

预期输出三行 `GemmDesc(...) GemmConfig(...)`（同一 desc+config 组合只打印一次，`heuristics.hpp:750-752`）。逐字段核对：① `acc=false, epilogue_type=...Identity, gemm_type=GemmType::Normal`；② `acc=true`；③ `acc=false, epilogue_type=...AddBias`。把打印行抄进第六节记录。

**步骤 2：打印 m-grouped 与 k-grouped 的 desc（约 15 分钟）**。手工复刻 psum 生成（等价于 `generators.py:212-220`）：

```python
import torch, torch_npu  # noqa
import deep_gemm

def gen_psum(sizes, alignment=256):
    layout, psum = [], 0
    for s in sizes:
        layout.append(psum + s)
        psum = (psum + s + alignment - 1) // alignment * alignment
    return psum, torch.tensor(layout, dtype=torch.int32, device='npu')

G, N, K = 2, 64, 128
M, g_layout = gen_psum([256, 512])                    # M=768, layout=[256,768]
a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')     # [M,K] K-major
b = torch.randn(G, N, K, dtype=torch.bfloat16, device='npu')  # [G,N,K]
d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, g_layout)
torch.npu.synchronize()

G, M, N = 2, 64, 64
ks = [128, 256]
Kt, g_layout = gen_psum(ks)                           # Kt=512, layout=[128,512]
a = torch.randn(Kt, M, dtype=torch.bfloat16, device='npu')    # [K,M] m 轴连续
b = torch.randn(Kt, N, dtype=torch.bfloat16, device='npu')    # [K,N]
d = torch.empty(G, M, N, dtype=torch.bfloat16, device='npu')
deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, g_layout)
torch.npu.synchronize()
```

预期：m-grouped 行含 `gemm_type=GemmType::MGroupedContiguousWithPsumLayout, num_groups=2, expected_m=384`（=768/2）；k-grouped 行含 `gemm_type=GemmType::KGroupedContiguousWithPsumLayout, num_groups=2, expected_k=192`（=(128+256)/2）。核对 `major_a`：前者 `Major::K`，后者 `Major::MN`。

**步骤 3：违规输入实验（约 10 分钟）**。沿用步骤 2 的 `gen_psum`，重新构造张量，逐条触发并记录异常消息，对应回源码行：

```python
M, N, K = 128, 64, 256
a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')   # [M,K] 行主序 → K-major
b = torch.randn(N, K, dtype=torch.bfloat16, device='npu')
d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')

deep_gemm.bf16_gemm_nt(a, b, torch.empty(M, N + 1, dtype=torch.bfloat16, device='npu'))
# 预期：Conflicting shape dimension 'n'（utils/layout.hpp:195-203）

G = 2
_, g_layout = gen_psum([256, 256])
b_g = torch.randn(G, N, K, dtype=torch.bfloat16, device='npu')       # [G,N,K]
a_mn = torch.randn(K, M, dtype=torch.bfloat16, device='npu').t()     # [M,K] strides (1,K) → MN-major
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_mn, b_g, d, g_layout)
# 预期：major_a == Major::K 断言失败（apis/gemm.hpp:90）

ks = [128, 256]
Kt, g_layout_k = gen_psum(ks)
b_k = torch.randn(Kt, N, dtype=torch.bfloat16, device='npu')          # [K,N] n 轴连续 → 转置后 MN-major
a_bad = torch.randn(M, Kt, dtype=torch.bfloat16, device='npu').t()    # [K,M] strides (1,Kt) → 转置后 K-major
deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a_bad, b_k,
        torch.empty(G, M, N, dtype=torch.bfloat16, device='npu'), ks, g_layout_k)
# 预期：major_a == Major::MN 断言失败（apis/gemm.hpp:94）
```

注意违规方向的构造方式：要让"假转置后"的视图变成违规 major，得从反方向 `.t()` 出来（第 2 天"逻辑转置只换 stride 元数据"的复习）。

**步骤 4（可选）：meta + dry_run 走 host 链路（约 5 分钟）**。`device='meta'` 的张量 + `deep_gemm._C.set_dry_run(True)`（第 9 天同款路径）下重复步骤 1，`DG_PRINT_CONFIGS` 依然会打印 desc——不需要 NPU 存储就能验证字段推导。

### B. 无 Ascend 环境：静态替代任务

**B-1：Normal 手工演算（约 10 分钟）**。设 `M=128, N=64, K=256`，`nt` 调用：A `[128,256]` strides `(256,1)`、B `[64,256]` strides `(256,1)`、D `[128,64]` strides `(64,1)`，均 bf16。填出：`m/n/k`、`major_a/major_b`、`outer_stride_a/b/d`、`gemm_type`、`num_groups`、`grouped_layout`、`expected_m/k`、`acc`、`compiled_dims`。
（答案：m=128（A[0]、D[0]）、n=64（B[0]、D[1]）、k=256（A[1]、B[1]）；major 均为 `Major::K`；outer stride 分别 256/256/64；`GemmType::Normal`、`num_groups=0`、`grouped_layout=nullptr`、`expected_m=128`、`expected_k=256`、`acc=false`、`compiled_dims="nk"`。）

**B-2：M-grouped 手工演算（约 15 分钟）**。`G=2`，每组行数 `[300, 512]`，对齐 256，`K=128, N=64`，`nt` 调用。按 `generate_psum_layout` 逻辑先算 `grouped_layout` 与 `total_m`，再假设 A `[total_m, K]` strides `(128,1)`、B `[2, 64, 128]` strides `(8192,128,1)`、D `[total_m, 64]` strides `(64,1)`，填出完整 desc。
（答案：`grouped_layout=[300, 1024]`、`total_m=1024`；g=2（B[0] 与 layout 长度）、m=1024、n=64、k=128；`major_a=major_b=Major::K`（断言通过）；`outer_stride_a=128`、`outer_stride_b=128`（取 `stride(-2)`）、`outer_stride_d=64`；`expected_m=512`（=m/g）；`stride_batch_*=0`；`gemm_type=MGroupedContiguousWithPsumLayout`。）

**B-3：K-grouped 手工演算（约 15 分钟）**。`G=2`，每组 K `[128, 300]`，对齐 256，`M=64, N=64`，`tn` 调用。物理张量 A/B 均为 `[total_k, 64]` strides `(64,1)`（mn 轴连续）。先算 `grouped_layout` 与 `total_k`，再写出 binding 假转置后 A/B 的 shape/stride，最后填完整 desc。
（答案：`grouped_layout=[128, 556]`、`total_k=768`；转置后 A `[64,768]` strides `(1,64)`、B 同；g=2（D[0] 与 layout 长度）、m=64（A[0]、D[1]）、n=64（B[0]、D[2]）、k=768（A[1]、B[1]）；`major_a=major_b=Major::MN`（断言通过）；`outer_stride_a=outer_stride_b=64`（取 `stride(-1)`）；`outer_stride_d=64`；`expected_k=214`（=(128+300)/2，若传 `ks_cpu`）、`expected_m=64`；`compiled_dims="mn"`；`gemm_type=KGroupedContiguousWithPsumLayout`。）

**B-4：闭卷填字段来源表（约 20 分钟）**。合上教程，凭源码（`gemm.hpp:164-196` + `gemm_config.hpp:39-95`）默写 3.10 表的"字段 → 来源行号 → 一句话说明"三列，重点检验：`d_dtype` 与 `cd_dtype`、`acc`、`grouped_layout`、`expected_m/k`、三个 outer stride。然后对照订正——这张订正后的表就是当天产出。

**B-5：switch 分支标注（约 10 分钟）**。对 `gemm.hpp:81-97` 的四个 case，各写一行"每个输出字母来自哪些张量的哪些轴"，并与 3.3 表核对；再指出 KGrouped 两条 major 断言与 MGrouped 一条断言的行号。

## 六、当天产出

学习计划要求今天的产出是**一张 `GemmDesc` 字段来源表**。按下述模板填写（B-1～B-3 的演算过程附在后面）：

```text
# DeepGEMM-Ascend 第 11 天记录：理解 GemmDesc

## A. GemmDesc 字段来源表（闭卷填写后对照 3.10 订正）
| 字段 | 来源（文件:行） | 值/规则 | 消费者 |
|------|----------------|---------|--------|
| a_ptr/b_ptr/d_ptr | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| a_dtype/b_dtype   | ＿＿＿ | ＿＿＿ | dispatch_fp8_fp4_gemm |
| d_dtype / cd_dtype| ＿＿＿ | ＿＿＿ | ＿＿＿ |
| sfa_ptr/sfb_ptr、sfa/sfb | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| m/n/k             | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| num_cores         | ＿＿＿ | ＿＿＿ | heuristics |
| major_a/major_b   | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| outer_stride_a/b/d| ＿＿＿ | ＿＿＿ | gm_ptr::index |
| acc               | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| epilogue_class    | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| gemm_type/grouped_layout/num_groups | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| expected_m/expected_k | ＿＿＿ | ＿＿＿ | ＿＿＿ |
| stride_batch_a/b/d| ＿＿＿ | ＿＿＿ | ＿＿＿ |

## B. 三种 gemm_type 的 shape 检查差异表
- g 的来源：Normal=＿＿、MGrouped=＿＿、KGrouped=＿＿
- major 断言：MGrouped=＿＿、KGrouped=＿＿
- grouped_layout 语义：m-grouped=＿＿、k-grouped=＿＿（对齐=＿＿）

## C. 手工演算记录（B-1～B-3）
- Normal：m=＿ n=＿ k=＿ major=＿/＿ outer_stride=＿/＿/＿
- M-grouped：layout=[＿,＿] total_m=＿ expected_m=＿
- K-grouped：layout=[＿,＿] total_k=＿ expected_k=＿ major=＿/＿

## D. 实验记录（有环境时）
- DG_PRINT_CONFIGS 抄录的三行 GemmDesc：＿＿＿
- 违规输入的三条异常消息与对应源码行：＿＿＿

## E. 遗留问题（交给第 12～33 天）
- select_gemm_config 如何用 desc 选 GemmConfig（第 12 天）
- gm_ptr 在设备侧的完整寻址与搬运（第 18～19 天）
- SF 布局转换的 kernel 细节（第 33 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `GemmDesc` 与 `GemmConfig` 各描述什么？分别定义在哪个文件？`GemmArgs` 是什么？
2. `get_gemm_desc` 为什么返回 `std::optional<GemmDesc>`？什么时候返回空？binding 此时做什么？
3. Normal 的 `m/n/k` 各来自哪些张量的哪些轴？M-grouped 的 `g` 来自哪里？两处不一致会发生什么？
4. k-grouped binding 为什么在调用 `get_gemm_desc` 前先对 A、B 做 `transpose(0, 1)`？数据被搬动了吗？
5. `major_a` 是用户指定的还是推断的？M-grouped 与 k-grouped 各自对 major 有什么硬性要求？违反时在哪一行被拒绝？
6. 对 K-major 与 MN-major 的操作数，`outer_stride` 分别取哪一轴的 stride？D 的 outer stride 为什么固定按 `Major::K` 取？
7. `d_dtype` 与 `cd_dtype` 有什么区别？`cd` 是什么？
8. `grouped_layout` 里存的是什么？m-grouped 与 k-grouped 的语义有何不同？组间对齐是多少？`m`（m-grouped）和 `k`（k-grouped）为什么含 padding？
9. `expected_m`/`expected_k` 从哪里来？`ks_cpu` 如何影响 `expected_k`？谁在消费它们？
10. `acc` 何时为 true？bias 时 `epilogue_class` 变成什么？为什么 m-grouped 的 binding 不传 `c`，而 k-grouped 可以？
11. （思考题）为什么所有 shape/major/stride 检查集中在 `get_gemm_desc` 一处而不是分散到各 binding？为什么 `GemmDesc` 除了 `sfa_ptr/sfb_ptr` 还要持有 `sfa/sfb` 张量本体？

**参考答案：**

1. `GemmDesc` 是问题描述（算什么：维度、dtype、指针、布局、分组、累加），`GemmConfig` 是执行配置（怎么算：tile、流水、L2、launch 选项）；都在 `csrc/jit_kernels/gemm_config.hpp`（39-95、97-121）。`GemmArgs`（124-127）是二者打包。二者的分工即"从用户张量提取校验"与"由 desc 启发式推导"。
2. 因为空问题（`m==0` 或 `n==0`；`k==0` 时先清零/拷贝 D）不需要构造描述、更不需要 JIT，`gemm_early_return`（`utils/layout.hpp:54-86`）返回 true 即整体返回 `nullopt`；binding 拿到空值直接 `return`（如 `gemm.hpp:446`）。
3. Normal：m 来自 A[0] 与 D[0]，n 来自 B[0] 与 D[1]，k 来自 A[1] 与 B[1]。M-grouped 的 g 来自 B[0] 与 `grouped_layout` 的长度（以 `"g"` 标注进 spec）。不一致走 `get_one_dim` 的 `DJ_PANIC("Conflicting shape dimension ...")`（`utils/layout.hpp:195-203`），Python 侧抛 `RuntimeError`。
4. 让 A/B 统一成规范形式 `[M,K]/[N,K]`，四种 gemm_type 共用同一套 spec 与 major 逻辑（`gemm.hpp:363-365` 注释）。`transpose` 只交换 shape/stride 元数据，零拷贝（第 2/10 天的概念）。
5. 推断的：`get_major_type_ab` 按 `stride(-1)==1 → K`、`stride(-2)==1 → MN`（`utils/layout.hpp:44-47`）。M-grouped 要求 `major_a == Major::K`（`gemm.hpp:90`）；k-grouped 要求 A、B 都是 `Major::MN`（`gemm.hpp:94-95`）。
6. K-major 断言 `stride(-1)==1`、取 `stride(-2)`（沿 MN 行距）；MN-major 断言 `stride(-2)==1`、取 `stride(-1)`（沿 K 行距）（`gemm_config.hpp:27-34`）。D 恒为 N-major（`check_major_type_cd` 只允许 `stride(-1)==1`），所以固定按 K-major 规则取 `stride(-2)`（沿 M 行距）。
7. `cd = c.value_or(d)`（`gemm.hpp:156`）——累加/bias 时的 C 或普通时的 D。`d_dtype` 是 `cd` 的 dtype（第 170 行），`cd_dtype` 是 `d` 的 dtype（第 177 行）并经 `dtype_to_cd_kernel_type` 决定 kernel 模板输出类型；由于 `gemm_early_return` 断言 C/D 同 dtype（`utils/layout.hpp:71`），二者实际相等。
8. `int32[G]` 的 psum 前缀和：m-grouped 存每组结束行号，k-grouped 存每组累计结束 K（`gemm_config.hpp:56-57` 注释）。组间对齐 256（`Runtime::kMkAlignment`，`jit.hpp:59`）。`total_m/total_k` 按"上一组结束位置对齐后作为下一组起点"累计（`generators.py:212-220`），所以 `m`、`k` 含组间 padding；有效区间由 layout 描述。
9. binding 参数（`expected_m_for_psum_layout`、`expected_k`）或估算值：m-grouped 落到 `m/g`，k-grouped 落到 `k/g`（`gemm.hpp:157-163、186-187`）。k-grouped binding 用 `expected_k_from_ks` 取 `ks_cpu` 平均值（`gemm.hpp:35-43`）——含 padding 的 `k` 会高估真实计算量。heuristics 用 `expected_k` 估 k-grouped 成本（`heuristics.hpp:158、545` 等）；`expected_m` 目前主要出现在 `DG_PRINT_CONFIGS` 打印（`gemm_config.hpp:153`）。
10. `acc = c.has_value() && !with_bias`（`gemm.hpp:180`），即 `c` 与 `d` 同指针同形的累加（k-grouped 的 `c` 是 `[G,M,N]`）。bias 时 `epilogue_class` 换成 `BiasEpilogue`（`gemm.hpp:122-123`，设备算子 `epilogue::operators::AddBias`）。m-grouped binding 把 `c` 写死为 `std::nullopt`（`gemm.hpp:313、488`），测试也断言其不支持累加（`common.py:56`、`generators.py:330`）——变长行段拼接下"先读旧值再累加"的语义没有实现。
11. 开放题，参考方向：集中校验让四个 binding + einsum 复用同一套逻辑，规则不会漂移，`GemmDesc` 成为唯一可信的问题描述；后续 heuristics、代码生成、launch 全部只依赖这个结构体。持有 `sfa/sfb` 本体是因为转换后的 SF 张量可能是 `launch_transform_sf` 新分配的临时对象，裸指针不能保活——`shared_ptr`/Tensor 引用保证 kernel 异步启动前存储不被释放（`gemm_config.hpp:77-78` 注释）。

---

今天结束时，你应该能把任何一个 Python 调用的张量参数，逐字段地"翻译"成一份 `GemmDesc`。明天（第 12 天）看它的搭档：`select_gemm_config` 如何从这份描述推导 `GemmConfig`，以及模板代码生成、`jit->compile()` 与 `jit->launch()`。
