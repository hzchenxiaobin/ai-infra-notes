# 第 8 周 · 第 4 天（总第 53 天）：选择并实施综合小项目

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 8 周"性能分析与综合实践"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 45 分钟，源码导读约 25 分钟，实践步骤约 90 分钟（今天是项目日，动手时间最长），自测题约 15 分钟。

前三周把"砖"备齐了：第 50 天有了度量衡（TFLOPS、带宽、算术强度），第 51 天有了观察窗口和基线记录表（`DG_PRINT_CONFIGS`），第 52 天有了单变量实验方法（假设 → 预测 → 改动 → 正确性 → 测量 → 判定）。今天把这些工具变成**一次真实的综合实践**：从四个方向中选一项实施——① 新增一个非对齐或 strided 测试场景；② 针对一个 shape 修改启发式配置并比较结果；③ 增加一个简单的 Epilogue 变体；④ 不写代码，完成一份 BF16 Kernel 的详细源码讲解。学习计划对本天的纪律要求只有一句话，但它是整个项目的骨架：**"先列出修改范围、预期结果和验证方法，再开始实施"**。今天的完成定义是"立项文档 + 实施 + 首轮验证"；**完整回归测试与基线对比留给明天（第 54 天"验证与修正"）**，所以今天结束时改动可以保留在工作区。四个项目都有不依赖 Ascend 硬件的静态替代版本，无环境同样能达到当天目标。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 按立项模板为所选项目写清三件事：**修改范围**（精确到文件、函数、插入点）、**预期结果**（可观察、可证伪）、**验证方法**（具体命令 + 通过标准），并说明为什么必须"先写预期再动手"；
2. 说清四个候选项目各自的改动面、硬件依赖和主要风险，并根据自身条件（有无硬件、时间、兴趣）做出有理由的选择；
3. 对所选项目完成实施与首轮验证，如实记录实验现象与预期的偏差——包括失败的实验；
4. 追踪一个自定义扩展在调用链上的全部贯通点（以测试场景与 Epilogue 为例：测试生成 → Python API → `GemmDesc` → 启发式决策 → JIT 代码生成 → 设备模板实例化）；
5. 区分失败的三种层级——编译期 `static_assert`（响亮）、host 侧 `DJ_HOST_ASSERT`（响亮）、正确性 diff 超阈值或性能回归（数据问题）——并在失败发生时判断它属于哪一层；
6. 无 Ascend 环境时，完成等价的静态产出：设计文档、接线图、UB 布局或调度波形的手工推演。

## 二、预备知识

- **第 51 天**：`DG_PRINT_CONFIGS` 配置行语法与基线记录表——项目 B 的对照基准。
- **第 52 天**：五个旋钮的代码地图、`heuristics.hpp` 的插入点选择原则（块大小类改搜索范围、标志类改最终配置）、"改写链"清单、单变量实验闭环——项目 B 直接复用这套方法。
- **第 29～31 天**：测试体系——`GemmTestDesc` → `enumerate_gemm_tests` → `run_gemm_test` 的数据流，`assert_equal` / `calc_diff` 的误差口径——项目 A 的地基。
- **第 11～13 天**：`GemmDesc` 字段来源与 JIT 调用链——所有项目都要用到"参数如何进入 C++"这条链。
- **第 22～26 天**：K 维循环、流水线同步、MAD、AIC/AIV 分工与 Epilogue——项目 C 和 D 的素材。
- **工具**：`git diff` / `git checkout -- <file>` 管理改动；今天的修改**不应提交**，是否保留到明天由项目性质决定（见第五节）。
- **环境说明**：需要编译或运行的项目（B、C，以及 A 的运行验证）需要 Ascend 环境（CANN 9.20 + torch_npu，已按第 9 天完成 `./develop.sh` 构建）；项目 A 的设计、项目 D 的全部内容、以及各项目的静态推演均不需要硬件。

## 三、详细讲解

### 3.1 先立项、后动手：今天的纪律

为什么学习计划强调先写"预期结果"再实施？原因和第 52 天的"假设闭环"同源：**没有事先写下的预期，任何结果都可以在事后被解释成"符合预期"**，项目就退化成"改了些代码、跑了些测试"。立项文档的作用是把项目变成一个可证伪的实验：预期写错了，项目会诚实地告诉你。

今天所有项目共用一个立项模板（存为 `week8_day4_plan.md`）：

| 字段 | 内容要求 | 反例（不合格） |
|---|---|---|
| 项目目标 | 一句话说清做什么、为什么值得做 | "优化一下性能" |
| 修改范围 | 文件 + 函数 + 插入点（行号），注明新增/修改 | "改 heuristics 相关代码" |
| 预期结果 | 可观察的现象：配置行长什么样、输出什么、误差多少、性能方向 | "应该会更好" |
| 验证方法 | 具体命令 + 通过标准（阈值/对比基准） | "跑一下看看" |
| 风险与回退 | 哪些联动可能让预期落空；如何还原 | 略过不写 |

立项后是实施，然后是**首轮验证**（至少一条正确性证据），最后把"实际现象 vs 预期"写进记录。失败的三种层级都要认识：

1. **编译期拦截**：设备侧 `static_assert`（如 `deep_gemm/include/deep_gemm/bf16_gemm.hpp:66-70` 的 "L1 overflow"）——非法配置响亮地失败，记录即可；
2. **Host 侧拦截**：`DJ_HOST_ASSERT`（如 `csrc/apis/gemm.hpp:114` 的 "Bias must be contiguous"）——参数违规在 launch 前被拦下；
3. **结果层失败**：正确性 diff 超阈值或性能回归——最难的一类，明天第 54 天专门处理，今天先如实记录。

### 3.2 四个选项总览与选择

| 项目 | 一句话描述 | 主要改动文件 | 硬件需求 | 难度 | 主要风险 |
|---|---|---|---|---|---|
| A：测试场景 | 为仓库补一个真正被覆盖的边界场景 | `tests/generators.py`、`tests/common.py`（纯 Python） | 运行需硬件；设计+推演可无硬件 | 低～中 | 接线点遗漏导致"看起来在测、实际没测" |
| B：启发式配置 | 对一个 shape 持久化地改进配置选择并完整比较 | `csrc/jit_kernels/heuristics.hpp`（需 `./develop.sh` 重建） | 性能对比强依赖硬件 | 中 | 改写链让实际配置偏离意图；性能回归 |
| C：Epilogue 变体 | 新增一个自定义输出变换算子并贯通全链路 | 4 个文件横跨 device/host/binding | 运行强依赖硬件 | 中～高 | 命名/类型不匹配导致 JIT 实例化失败；语义设计错误（见 3.5） |
| D：源码讲解 | 不写代码，产出一份 BF16 Kernel 详细讲解文档 | 无（产出 Markdown） | 零硬件 | 难在"讲清楚" | 复述而非解释；缺少可核对的事实出处 |

选择建议：

- **无 Ascend 环境** → 项目 D 是完全体；项目 A/B/C 的静态版本（设计文档 + 手工推演）也全部可行，且立项模板一模一样。
- **有环境、时间紧** → 项目 A（其中"加一个非对齐 shape"的 A1 子任务约 10 分钟，可作为任何人的热身）。
- **有环境、想练完整工程链路** → 项目 C：它是四个项目里唯一贯穿 Python binding → host → JIT 代码生成 → 设备模板四层的，做完后对整个仓库的理解会上一个台阶。
- **有环境、想深挖性能** → 项目 B：直接把第 52 天的实验升级为持久化改进。

一天只做一个项目。下面 3.3～3.6 节对四个项目分别给出"改动面 + 立项示范 + 验证方法"，**你只需要精读自己选中的那一节**，其余浏览即可。

### 3.3 项目 A：新增非对齐或 strided 测试场景

**改动面：测试体系。** 先用一张图回顾数据流（第 29～31 天的产出）：

```text
tests/test_bf16_gemm.py（CLI 选择器）
  → run_gemm_test(...)                        tests/common.py:131
  → enumerate_gemm_tests(...)                 tests/generators.py:652
      → _enumerate_normal / _unaligned_mn / _unaligned_k / _strided
  → 逐个 GemmTestDesc：
      ① generate(device='meta') + par_compile   （并行预编译，common.py:157-175）
      ② generate() → gemm_dg → deep_gemm API     （真实执行，common.py:199）
      ③ assert_equal(d, d_ref)                   （参考比对，common.py:201）
      ④ gemm_aclnn 交叉验证（可选，common.py:210-213）
      ⑤ bench_msprof 性能（可选，common.py:226-228）
```

本项目有三个由浅入深的子任务。

**A1（热身，约 10 分钟）：给非对齐场景加一个 shape。** `_enumerate_normal_unaligned_k`（`tests/generators.py:519-541`）维护了一个非对齐 K 的 shape 列表（`k=17/15/234/4095` 等）。在列表（`:520-525`）中追加一项，例如 `(256, 512, 1000)`——K=1000 不是 16 的倍数（BF16 候选要求 `block_k` 对齐 16，`heuristics.hpp:108-109` 的 `k_alignment`，于是 `k % block_k != 0` 必然产生尾部块，触发 `actual_k` 边界路径），也不是 64 的倍数（FP8 的 SF 粒度 `MX_SF_DIVISOR`，`common.hpp:65`），对两类路径都构成真实非对齐。注意 FP4 的奇偶 K 修正逻辑在 `:532-533`，K=1000 是偶数不受影响。预期：`python tests/test_bf16_gemm.py unaligned-k --skip-prof` 打印的 `Num tests` 增加 4（每个 shape × 4 种 major 组合，`generators.py:479` 的 `product`），输出中出现 `m=  256, n=  512, k= 1000` 的行且全部通过。

**A2（主菜）：让 `strided` 选择器真正测试 stride。** 先陈述两个可核对的事实（以当前仓库 Public Release 2026/10 为准，动手前请用 `rg -n "make_strided" tests/` 重新确认）：

1. `make_strided`（`tests/generators.py:238-246`）在当前测试代码里**没有任何调用方**；
2. `_enumerate_normal_strided`（`tests/generators.py:466-488`）把 `strided=True` 写进 `GemmTestDesc`（字段声明在 `:261`），但 `_generate_normal`（`:300-324`）和 `gemm_dg`（`tests/common.py:49-84`）都**不读取** `test.strided`。

两条合起来的结论：`python tests/test_bf16_gemm.py strided` 目前实际测试的是"这些 shape + `accumulate=True` 的**连续**张量"，`[strided]` 标签（`common.py:22-23`）只是标题。这是一个标准的"为已有特性补测试"场景：kernel 侧对非紧凑外层 stride 的支持是真实存在的——`GemmDesc` 专门设计了 `outer_stride_a/b/d` 字段（`csrc/jit_kernels/gemm_config.hpp:68-75`），host 侧 `get_outer_stride`（`gemm_config.hpp:27-34`）从张量读取真实 stride，设备侧 `gm_ptr::index` 用 `stride_outer` 参与地址计算（`deep_gemm/include/deep_gemm/common.hpp:156-165`）——但普通 GEMM 测试没有覆盖它。

还有第三个事实：**直接调用现成的 `make_strided` 也不行**。它把 `shape[dim]` 和 `stride[dim]` 同时放大，再试图把原张量 `copy_` 进放大后的视图——源形状与目标形状不匹配，必然抛错。所以需要自己写一个正确的构造：**保持逻辑形状不变，只放大外层 stride，内层轴保持单位 stride**。内层必须连续不是随便说的，而是 host 断言的要求：`get_major_type_ab` / `major_check`（`csrc/utils/layout.hpp:41-47`）与 `check_major_type_cd`（`:49-52`）只要求最后一个维或倒数第二个维之一是单位 stride；`get_outer_stride`（`gemm_config.hpp:29/:32`）据此取另一个维的 stride 作为外层 stride。唯一强制连续的是 bias（`gemm.hpp:114`）——所以 **c 永远不要做 stride**。

正确的构造函数（放进 `tests/common.py` 或 `tests/generators.py`，二选一即可）：

```python
def pad_outer_stride(t: torch.Tensor, pad: int = 64) -> torch.Tensor:
    """Keep the logical shape; only enlarge the outer-dim stride (inner axis stays unit-stride)."""
    shape, stride = list(t.shape), list(t.stride())
    if t.dim() == 2 and t.stride(-2) == 1 and t.stride(-1) != 1:
        # MN-major [MN, K]：外层是 K（dim -1），stride = (1, MN + pad)
        stride[-1] = shape[-2] + pad
        buf = torch.empty((shape[-1] - 1) * stride[-1] + shape[-2], dtype=t.dtype, device=t.device)
    else:
        # K-major [MN, K] 与输出 D [M, N]：外层是 MN（dim -2），stride = (K + pad, 1)
        assert t.stride(-1) == 1
        stride[-2] = shape[-1] + pad
        buf = torch.empty((shape[-2] - 1) * stride[-2] + shape[-1], dtype=t.dtype, device=t.device)
    view = buf.as_strided(shape, stride)
    view.copy_(t)
    return view
```

接线点选在 `run_gemm_test` 的**运行阶段**（两处输出缓冲 `d` 创建之后），而不是 `gemm_dg` 内部——原因有二：① `gemm_dg` 也被 `par_compile` 用 meta 张量调用（`common.py:157-159`），在运行阶段接线可以让编译阶段保持原样（stride 不是 `GemmConfig` 的一部分，只是运行时 launch 参数，`csrc/jit_kernels/bf16_gemm.hpp:74-76`，所以同一个编译产物即可覆盖）；② 输出 `d` 必须以"带 padding 的视图"形式存在，kernel 写进视图后 `assert_equal(d, d_ref)` 才能比较到结果。

```python
# tests/common.py，插在 common.py:190（d = torch.randn_like(d_ref) 这行之后）
# 与 common.py:209（第二处 d = torch.randn_like(d_ref) 之后），各一份：
if test.strided and not isinstance(a, tuple):          # 先只覆盖 BF16（无 SF 元组）
    a, b, d = pad_outer_stride(a), pad_outer_stride(b), pad_outer_stride(d)
```

最后补一个跳过：`gemm_aclnn` 顶部（`tests/common.py:88-91` 现有 `if test.gemm_type != GemmType.Normal: return False` 的先例）加 `if test.strided: return False`——ACLNN 交叉验证不必为 strided 背书，主验证是 `d_ref` 比对。这样立项三要素齐了：

- **修改范围**：`tests/common.py` 新增 `pad_outer_stride` + 两处接线 + `gemm_aclnn` 一处跳过（纯 Python，**不需要** `./develop.sh`）；
- **预期结果**：`python tests/test_bf16_gemm.py strided --skip-prof` 全部通过；`d.stride()` 显示非紧凑（如 `(K+64, 1)`）；perf 行（`:226-228` 的 `bench_msprof` 拿到的正是 strided 张量）正常打印；
- **验证方法**：上条命令 + 一个最小直调脚本（见第五节 A 组步骤 3）；通过标准是 `calc_diff < 1e-5`（与 `test_bf16_gemm.py:17` 的阈值同口径）。

**A3（进阶展望）**：把 strided 推广到 FP8/FP4（`a` 是 `(data, sf)` 元组的情形，`generators.py:124-166`）。SF 布局有自己的对齐约束（第 33 天），难度明显更高，本天不做。

### 3.4 项目 B：针对一个 shape 修改启发式配置并比较

**改动面：`csrc/jit_kernels/heuristics.hpp` 一处 + 重建。** 它与第 52 天的关系是"实验 → 改进"：第 52 天用环境变量门控做**临时**实验、用完即还原；今天把验证过的判断固化为 **shape 门控的持久改动**，并做完整比较。第 52 天的插入点原则原样适用：**块大小类旋钮改搜索范围（早），标志类旋钮改最终配置（晚）**。

示范立项（以第 52 天示范 2 的推演为基础）：对 `128×7168×2048` BF16 NT，把 block 搜索钉到 `128×224`——32 个 N 块恰好铺满 32 个核、单 wave、无尾部浪费：

```cpp
// ===== [第 53 天项目] 插在 heuristics.hpp:609 之后（m_range/n_range 定义之后）=====
if (d.gemm_type == GemmType::Normal && d.m == 128 && d.n == 7168 && d.k == 2048 &&
    d.a_dtype == at::kBFloat16 && d.b_dtype == at::kBFloat16) {
    m_range = {128, 128};
    n_range = {224, 224};
}
```

（这段是你自己添加的项目代码，不属于仓库原有接口；`GemmDesc.a_dtype` 的类型是 `at::ScalarType`，`gemm_config.hpp:43`。）

立项三要素：

- **修改范围**：`heuristics.hpp:609` 之后一处插入；`./develop.sh` 重建。
- **预期结果**（写进文档，可证伪）：`DG_PRINT_CONFIGS=1` 下该 shape 的配置行 `block=128x224x128`。这里有一处值得手推的细节——K=2048 满足 `k % max_block_k == 0`（`max_block_k = 2 × min_block_k = 256`，`:620-621`），`:648-651` 会尝试更宽的 `block_k=256`，但其 `ab_bytes = (128+224)×256×2 = 180,224 B`，L1 只装得下 `524,288/180,224 = 2` 级，不满足 `≥ 3` 级的替换条件，所以**预测 block_k 保持 128**（`128×224×128` 时 `ab_bytes = 90,112 B`，5 级）。性能预测沿用第 52 天的波形表：`BLOCK_N=224` 时 32 tile / 1 wave，手推下界约 9.1 µs（`128×7168×2048`，计算 8.70 µs / 访存 9.06 µs 取 max）。同时要预判**改写链**：bias 变体可能触发 `:697-711` 的双缓冲搜索，K 精修 `:672-694` 可能改 `block_k`——最终以打印行为准。
- **验证方法**：① 正确性：`python tests/test_bf16_gemm.py normal --skip-prof`（门控只命中目标 shape，其余 shape 走默认路径，回归既全面又聚焦）；② 配置核对：目标 shape 的配置行符合预期，其余 shape 与第 51 天基线一致；③ 性能：最小脚本 + `bench_msprof`，与第 51 天基线表对比，填第 52 天的日志表格式。

其他可选题目（同一方法换旋钮）：钉死 `block_k` 并门控 `:620-621` 与 `:672`；对 `128×7168×512` 这类小 K shape 尝试翻转 resident A 资格（第 52 天源码导读第 2 条已走查过：K=512 时 `a_bytes = 128×512×2 = 131KB < 512KB` 可启用）。

**无硬件的静态版本**：完整立项文档 + 手推预测表——枚举 `128×7168×2048` 在 BF16 NT 下的合法候选（`m_limit=128`，`block_cap=512`，L0C 约束，N 对齐 16），按 `get_gemm_compute_cost` 的 `num_waves × 单 Tile 面积`（`heuristics.hpp:219/:231`）排出 BLOCK_N ∈ {128, 224, 256, 512} 的预测，并写出"我预测启发式默认会选哪个、我的改动后选哪个"，将来有环境时用 `DG_PRINT_CONFIGS` 逐格验证。

### 3.5 项目 C：增加一个简单的 Epilogue 变体

**改动面：4 个文件，横跨设备算子、host 类、Python 绑定。** 这是四个项目里链路最长的，也最能检验前七周的积累。

**第一步：盘点现有家族。** 设备侧算子在 `deep_gemm/include/deep_gemm/epilogue/operators.hpp`：`Operator` 基类（`:92-117`）+ 五个实现——`Identity`（`:120`）、`ScaleByAlpha`（`:125-158`）、`AddBias`（`:161-222`）、`StochasticRoundToBF16`（`:225-335`）、`QuantizeToFP8`（`:338-447`）。Host 侧类在 `csrc/jit_kernels/epilogue_class.hpp`：基类 `EpilogueClass`（`:14-27`）+ 一一对应的五个类（`:29-107`）。Python 绑定在 `csrc/apis/epilogue_class.hpp:10-21`，暴露了 `Identity`、`Alpha`、`FP8Quantization`、`BF16StochasticRounding` 四个——**`Bias` 不暴露**，它由 `get_gemm_desc` 在检测到 `[1, N]` 的 `c` 时自动构造（`csrc/apis/gemm.hpp:122-123`）。

**第二步：吃透自定义 epilogue 的贯通链。** 一条自上而下的完整链路（项目立项文档里应有这张图）：

```text
deep_gemm.bf16_gemm_nt(a, b, d, epilogue=deep_gemm.epilogue.XXX())
  → resolve_epilogue_class            csrc/jit_kernels/epilogue_class.hpp:110-128
      （与 alpha / sfd 互斥：:117；然后调 check(c, d)）
  → GemmDesc.epilogue_class           csrc/jit_kernels/gemm_config.hpp:51
  → 启发式用 dynamic_cast 探测类型：
      get_gemm_dtype:81  with_transform = 非 Identity → epilogue_elem_bits = 32
      can_direct_store:526 只允许 Identity → 自定义 epilogue 一定走 AIV 路径
  → get_epilogue_operator_type() 的返回字符串被 std::format 写进模板实参
                                     csrc/jit_kernels/bf16_gemm.hpp:64
  → 设备模板参数 template <typename> class epilogue_operator_t
                                     deep_gemm/include/deep_gemm/bf16_gemm.hpp:23
  → Epilogue<cd_dtype_t, op_t, ...> 实例化框架
                                     deep_gemm/include/deep_gemm/epilogue.hpp:15-55
  → AIC：aic_store 把 L0C(fp32) 拷进 UB（宽度由 ub_dtype_t 决定）   epilogue.hpp:41-47
  → AIV：aiv_drain → op.initialize/process/postprocess_mad_block    epilogue.hpp:54/74/83
  → copy_ub_to_gm 写回 GM                                           epilogue.hpp:81-82
```

三个必须理解的机制（每个都能在源码里指出处）：

1. **就地收缩**：`aiv_drain` 把**同一地址**的 UB 交给算子两次——`values`（`ub_dtype_t*`，如 float）和 `output`（`cd_dtype_t*`，如 bfloat16）（`epilogue.hpp:71-74`）。`static_assert(sizeof(ub_dtype_t) >= sizeof(cd_dtype_t))`（`:28`）保证原地写窄是安全的。`Identity`+BF16 时 `ub_dtype_t = bfloat16_t`（无变换，`epilogue_elem_bits=16`）；非 Identity 时 `ub_dtype_t = float`，算子读 fp32、写窄类型——这正是 `get_gemm_dtype`（`heuristics.hpp:81-82`）里 `with_transform` 的含义。
2. **流水线联动**：`epilogue_elem_bits` 从 16 变 32 会让 UB 每 stage 占用翻倍，`make_gemm_candidate` 的 `num_epilogue_stages` 相应缩小（`heuristics.hpp:188-190`），还受 wave 数封顶（`:193`）。所以**新增非 Identity epilogue 后配置行的 `epi_stages` 变小是预期内的联动**，不是 bug。
3. **语义边界**：epilogue 作用于**本块刚算出的乘积**，而累加 `c` 是 AIV 在 `copy_ub_to_gm` 时用 atomic add 加到 GM 的 `d` 上（`aiv_initialize` 里 `asc_set_atomic_add_*`，`epilogue.hpp:91-96`）。测试侧印证：`_generate_normal` 的参考是 `alpha*(A@B) + c`（`tests/generators.py:320-322`）——alpha 乘在乘积上、c 在 GM 侧相加。**自定义算子同理：它看不到 c**。设计 `check()` 时应据此排除歧义（见下）。

**第三步：示范实现——ReLU epilogue（零参数算子，最适合作为第一个）**。四个修改点：

① 设备算子，`operators.hpp` 中 `ScaleByAlpha` 之后新增（骨架照抄 `:125-158`，核心一行是 `asc_max_scalar`——它在仓库里有 float 用法先例：`deep_gemm/include/deep_gemm/simd/mega_moe.hpp:108-110` 的 SwiGLU clamp）：

```cpp
template <ParamsType P>
struct ReLU : Operator<P> {
    using Operator<P>::Operator;
    using cd_dtype_t = typename P::cd_dtype_t;
    using ub_dtype_t = float;
    static constexpr pipe_t kPipe = PIPE_V;

    __aicore__ void process_mad_block(__ubuf__ ub_dtype_t* values, __ubuf__ cd_dtype_t* output,
                                      Args args, const MadBlock& block) const {
        apply_values<P::MAD_M, P::MAD_N>(values, output);
    }

private:
    template <uint32_t kNumRows, uint32_t kNumCols>
    static __simd_vf__ void apply_values(__ubuf__ float* values, __ubuf__ cd_dtype_t* output) {
        constexpr uint32_t kNumFP32PerVector = asc_get_vf_len() / sizeof(float);
        static_assert(kNumRows * kNumCols % kNumFP32PerVector == 0);
        const auto mask = asc_create_mask_b32(PAT_ALL);
        for (uint32_t value_idx = 0; value_idx < kNumRows * kNumCols; value_idx += kNumFP32PerVector) {
            vector_float value;
            asc_loadalign(value, values + value_idx);
            asc_max_scalar(value, value, 0.0f, mask);        // ReLU：负值清零
            if constexpr (std::is_same_v<cd_dtype_t, float>) {
                asc_storealign(output + value_idx, value, mask);
            } else {
                vector_bfloat16_t converted;
                vector_uint16_t packed;
                asc_float2bfloat16_rn(converted, value, mask, ASC_POSITION_EVEN);
                asc_pack_to_low(packed, reinterpret_cast<vector_uint32_t&>(converted));
                asc_storealign(output + value_idx, reinterpret_cast<vector_bfloat16_t&>(packed),
                               asc_create_mask_b16(PAT_VL64));
            }
        }
    }
};
```

② Host 类，`csrc/jit_kernels/epilogue_class.hpp` 新增：

```cpp
class ReluEpilogue final : public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::ReLU";
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DJ_HOST_ASSERT(not c.has_value(), "ReLU requires a direct output without accumulation");
    }
};
```

注意返回字符串**必须与设备结构体名逐字符一致**（含命名空间）——它会被 format 进 JIT 生成代码作为模板实参（`csrc/jit_kernels/bf16_gemm.hpp:64`），拼错即编译失败。`check` 排除 `c`，就是上面"语义边界"的落地：ReLU 只作用于新算出的乘积，`relu(A@B + c)` 与 `relu(A@B) + c` 不等价，不让用户踩这个坑（`BF16StochasticRoundingEpilogue::check` 在 `:72-75` 有同样的先例）。

③ 绑定，`csrc/apis/epilogue_class.hpp` 的 `register_apis` 里加：

```cpp
    pybind11::class_<ReluEpilogue, EpilogueClass, std::shared_ptr<ReluEpilogue>>(submodule, "ReLU")
        .def(pybind11::init<>());
```

④ 重建 + 验证：`./develop.sh`，然后 `deep_gemm.bf16_gemm_nt(a, b, d, epilogue=deep_gemm.epilogue.ReLU())`，参考结果 `torch.relu(a.float() @ b.float().t()).to(torch.bfloat16)`（epilogue 看不到 c，见上）。

立项三要素：**修改范围**＝上述四点（新文件零个，全部是"在既有模式旁加一个实例"）；**预期结果**＝ReLU 正确（diff < 1e-5，BF16 输出阈值同 `test_bf16_gemm.py:17`）、配置行 `epilogue_type=epilogue::operators::ReLU` 且 `epi_stages` 相比 Identity 变小、`direct_store=0`；**验证方法**＝最小脚本 + 回归 `python tests/test_bf16_gemm.py normal --skip-prof`（确认默认路径未破坏）+ `DG_PRINT_CONFIGS=1` 核对联动。工程提醒：动了 `deep_gemm/include/` 下的设备头文件，把 `DG_JIT_CACHE_DIR` 指到新目录以免读到旧缓存 kernel（第 52 天 3.3 节工程细节 2）。

**无硬件的静态版本**：完成四点改动的完整设计文档（代码如上，但重点是能解释每一行为什么这么写）+ 两项手推：① UB 布局对比——设 MAD Tile 128×256，Identity+BF16 每 stage 占 `128×256×2 = 64KB`，ReLU（fp32 变换）占 `128×256×4 = 128KB`，按 UB 256KB（`common.hpp:58`）与 `num_intra_block_events=16`（`heuristics.hpp:66`）推 `epi_stages` 的变化方向；② 用第 25 天的 AIC/AIV 流程图标出 ReLU 的插入位置与 `kPipe=PIPE_V` 带来的同步差异（`epilogue.hpp:76-80`：非 MTE3 管道在 store 前要 notify/wait 一次 MTE3——框架已自动处理，但要能指出在哪）。

### 3.6 项目 D：不写代码，完成一份 BF16 Kernel 详细源码讲解

**为什么它是一个"项目"而不是读书笔记**：产出要有明确的读者与用途（组内分享 / 新人导读），质量标准是"读者不看源码也能理解"，且**每个事实都能在源码里指出处**——这迫使你把第 3、4 周的被动阅读变成主动解释。

建议的文档骨架（8 节，每节配一张图或一张表）：

| # | 节 | 核心素材（真实出处） |
|---|---|---|
| 1 | 问题与接口 | `GemmDesc` 字段表（`gemm_config.hpp:39-95`；第 11 天的产出可直接复用） |
| 2 | 编译期与运行期 | 模板参数 `SHAPE_M/N/K` vs 运行时 shape（设备 `bf16_gemm.hpp:11/:47-49`）、`get_compiled_dim`（`gemm_config.hpp:17-21`）、`compiled_dims="nk"` 的来源（`csrc/apis/gemm.hpp:635`） |
| 3 | 片上内存规划 | `l1a/l1b/l0a/l0b/l0c` 声明与 offset 级联（设备 `bf16_gemm.hpp:59-64`）、容量 `static_assert`（`:66-70`）、容量常数（`common.hpp:54-58`） |
| 4 | 调度 | `Scheduler` 构造与 `get_next_block/get_global_idx`（设备 `:72-74`、`scheduler.hpp:49+`）、三种 `GemmType` 的分支（`:76-79`） |
| 5 | AIC 主循环逐步走 | resident A 装载（`:109-110`）、GM→L1（`:116-119`）、L1→L0（`:141-146`）、`mad` 与 `is_first/is_last`（`:133-152`）、K 尾块写回（`:155-163`） |
| 6 | 同步与流水 | `set_flags`（`:83-84`）、每一对 `asc_sync_wait/notify` 的流水含义（第 23 天）、stage 下标循环（`:112/:124/:133`） |
| 7 | AIV 分支 | `aiv_zero_empty_groups`（`:175-176`）、`aiv_initialize` 的 atomic（`epilogue.hpp:91-96`）、`aiv_drain` 的 stage 轮转（`epilogue.hpp:49-87`） |
| 8 | 特殊路径与约束 | direct store（`:37-39/:56-57/:156-159/:174`）、resident A/B（`:40/:45/:109-118/:142-145`） |

**质量标准**（自查三问）：① 不看源码，能否讲清一个输出 Tile 从调度到写回的完整生命周期，精确到同步事件级？（这是第 27 天通过标准的升级版。）② 每个论断旁是否有 `文件:行号`？③ 文档结尾是否诚实列出"三个我仍未理解的问题"？（第 27 天就要求过记录三个问题，这次要带着答案或带着更精确的问题离开。）

## 四、源码导读

今天按项目分组读，只精读自己选的方向（行号以当前仓库为准）：

| # | 文件 | 关键符号 | 与今天的关系 |
|---|---|---|---|
| 1 | `tests/generators.py` | `GemmTestDesc` `:249-276`、`_generate_normal` `:300-324`、`_enumerate_normal_strided` `:466-488`、`_enumerate_normal_unaligned_k` `:519-541`、`make_strided` `:238-246`（无调用方） | 项目 A 的主战场 |
| 2 | `tests/common.py` | `run_gemm_test` `:131-237`（meta 编译 `:157-175`、运行 `:189-213`、perf `:226-228`）、`gemm_dg` `:49-84`、`gemm_aclnn` `:87-128`（跳过先例 `:90-122`） | 项目 A 的接线层 |
| 3 | `tests/utils.py` | `calc_diff` `:36-43`、`assert_equal` `:46-58`、`print_perf` `:106-114` | 验证口径 |
| 4 | `csrc/jit_kernels/heuristics.hpp` | `select_gemm_config` `:592-755`（范围 `:604-621`、搜索 `:626-657`）、改写链（更宽 K `:648-651`、K 精修 `:672-694`、bias 双缓冲 `:697-711`）、`get_gemm_dtype` `:80-90`、`can_direct_store` `:521-533`、打印块 `:749-753` | 项目 B 主战场；项目 C 的联动来源 |
| 5 | `csrc/jit_kernels/gemm_config.hpp` | `GemmDesc` `:39-95`（`outer_stride_*` `:75`、`epilogue_class` `:51`）、`get_outer_stride` `:27-34`、`get_compiled_dim` `:17-21`、`GemmConfig` `:97-121` | A/B/C 共用的描述层 |
| 6 | `csrc/jit_kernels/bf16_gemm.hpp` | `launch_bf16_gemm` `:16-78`（epilogue 类型进代码 `:64`、launch 传 stride `:74-77`） | B/C 的代码生成层 |
| 7 | `csrc/jit_kernels/epilogue_class.hpp` | `EpilogueClass` `:14-27`、五个实现 `:29-107`、`resolve_epilogue_class` `:110-128` | 项目 C 的 host 层 |
| 8 | `csrc/apis/epilogue_class.hpp` | `register_apis` `:10-21` | 项目 C 的绑定层 |
| 9 | `csrc/apis/gemm.hpp` | `bf16_gemm` `:425-448`、bias→`BiasEpilogue` `:122-123`、`num_cores` `:176`、绑定 `epilogue` 形参 `:632-637` | 项目 C 入口；A 的 stride 入口 `:191-193` |
| 10 | `csrc/utils/layout.hpp` | `major_check` `:41-42`、`get_major_type_ab` `:44-47`、`check_major_type_cd` `:49-52` | 项目 A 的合法性依据 |
| 11 | `deep_gemm/include/deep_gemm/epilogue.hpp` | `Epilogue` `:15-142`（`aic_store` `:41-47`、`aiv_drain` `:49-87`、就地收缩 `:71-74`、`static_assert` `:28-35`、atomic `:91-96`） | 项目 C 的设备框架 |
| 12 | `deep_gemm/include/deep_gemm/epilogue/operators.hpp` | `Operator` 基类 `:92-117`、`Params` `:59-68`、`MadBlock` `:86-90`、`Identity` `:120`、`ScaleByAlpha` `:125-158`、`AddBias` `:161-222`、`StochasticRoundToBF16` `:225-335`、`QuantizeToFP8` `:338-447` | 项目 C 的算子层（D 的第 7 节素材） |
| 13 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp`（设备） | 模板 `:9-26`、`static_assert` `:37-70`、AIC 主循环 `:81-172`、AIV `:173-192` | 项目 D 全文；C 的宿主 |
| 14 | `deep_gemm/include/deep_gemm/common.hpp` | `epilogue::Args` `:96-103`、容量常数 `:54-65`、`gm_ptr::index` `:156-165` | A 的地址计算；C 的参数结构 |
| 15 | `deep_gemm/include/deep_gemm/simd/mega_moe.hpp` | `asc_max_scalar` float 用法 `:108-110`（参数声明 `:61`） | 项目 C 的 intrinsic 先例 |
| 16 | `develop.sh` / `README.md` | 重建流程、`DG_JIT_CACHE_DIR` `README.md:91`、性能表 `:100-186` | 工程与对照数据 |

**重点读三段**（按项目）：

- **项目 A**：`tests/common.py:131-237` 完整走一遍 `run_gemm_test` 的五个阶段（meta 编译 → 运行 → 比对 → aclnn → perf），在纸上标出你的两处接线点各影响哪些阶段；再用 `rg -n "strided" tests/` 复核 3.3 节的两个事实。
- **项目 B**：`heuristics.hpp:592-657`（选择主循环）+ `:672-711`（K 精修与 bias 双缓冲）——确认你的门控之后还有哪些阶段可能改写 block；对照第 52 天读过的 `:535-574`（`refine_gemm_pipeline`）复习 resident/direct store 的启用条件。
- **项目 C**：`operators.hpp:92-158` 逐行读（基类的五个钩子默认实现 + `ScaleByAlpha` 全文——你的 ReLU 就是它的变体），然后 `epilogue.hpp:41-96`（`aic_store` 与 `aiv_drain`，注意 `:71-74` 同一地址两种类型视图、`:76-80` 的 PIPE_V→MTE3 同步）。
- **项目 D**：设备 `bf16_gemm.hpp` 全文 195 行 + 第 4 周读过的 `scheduler.hpp` / `ascend/copy.hpp` / `ascend/sync.hpp`。

## 五、实践步骤

### 第 0 步：立项（所有项目、所有组别必做，约 15 分钟）

按 3.1 节模板写 `week8_day4_plan.md`，五个字段填全。特别把"预期结果"写成可证伪的形式（配置行、数值、方向）。**写完再打开编辑器。**

### A 组：有 Ascend 环境

**项目 A（strided 接线）**

1. 热身 A1：按 3.3 节给 `_enumerate_normal_unaligned_k` 加 `(256, 512, 1000)`，运行 `python tests/test_bf16_gemm.py unaligned-k --skip-prof`，记录 `Num tests` 增量与新 case 输出（纯 Python 改动，无需重建）。
2. 实施 A2：按 3.3 节完成 `pad_outer_stride` + 两处接线 + `gemm_aclnn` 跳过；`git diff` 检查改动只落在计划范围。
3. 最小直调验证，存为 `week8_day4_strided.py`（与 `tests/` 同目录运行）：

   ```python
   import torch, deep_gemm

   M, N, K, PAD = 128, 512, 1024, 64
   torch.manual_seed(0)
   a = torch.randn(M, K, dtype=torch.float32, device='npu').to(torch.bfloat16)
   b = torch.randn(N, K, dtype=torch.float32, device='npu').to(torch.bfloat16)

   def pad_outer(t, pad):
       assert t.dim() == 2 and t.stride(-1) == 1
       stride = (t.shape[1] + pad, 1)
       buf = torch.empty((t.shape[0] - 1) * stride[0] + t.shape[1], dtype=t.dtype, device=t.device)
       view = buf.as_strided(t.shape, stride)
       view.copy_(t)
       return view

   a_s, b_s = pad_outer(a, PAD), pad_outer(b, PAD)
   d_s = pad_outer(torch.empty(M, N, dtype=torch.bfloat16, device='npu'), PAD)
   print('strides:', a_s.stride(), b_s.stride(), d_s.stride())   # 期望 (1088, 1) (1088, 1) (576, 1)

   deep_gemm.bf16_gemm_nt(a_s, b_s, d_s)
   torch.npu.synchronize()
   x, y = d_s.float(), a_s.float() @ b_s.float().t()
   diff = 1 - 2 * (x * y).sum() / (x * x + y * y).sum()
   print('diff:', float(diff))
   assert diff < 1e-5, diff
   print('strided GEMM OK')
   ```

4. 全量验证：`python tests/test_bf16_gemm.py strided --skip-prof`；对照立项预期（全部通过、stride 打印符合、perf 行正常）。
5. 保留改动到明天（第 54 天将检查"新增测试是否真正覆盖目标场景"——这个问题的答案就是 `week8_day4_strided.py` 的输出）。

**项目 B（启发式改进）**

1. 立项文档（含 3.4 节的手推预测：最终 `block_k` 为何保持 128）。
2. 打补丁（`heuristics.hpp:609` 之后）→ `./develop.sh` → `export DG_PRINT_CONFIGS=1` 跑一次目标 shape 的最小脚本，核对配置行与预测逐字段一致；若被改写链改写，记录实际值并修正预期。
3. 正确性回归：`python tests/test_bf16_gemm.py normal --skip-prof`（其余 shape 配置行应与第 51 天基线完全一致——把它们也抄进记录，作为"门控无泄漏"的证据）。
4. 性能：最小脚本 + `bench_msprof`（口径同第 51 天：冷 L2），与基线对比，按第 52 天 3.6 节日志表记录；判定三档（证实/证伪/无法判定）。
5. **保留补丁与构建产物到明天**（第 54 天将与基线做完整对比后决定去留）。

**项目 C（ReLU epilogue）**

1. 立项文档（含 3.5 节的语义边界说明：ReLU 看不到 c）。
2. 按 3.5 节完成四点改动 → `DG_JIT_CACHE_DIR=/tmp/dg_day4_cache ./develop.sh`（设备头文件改动，换缓存目录）。
3. 验证脚本 `week8_day4_relu.py`：随机 BF16 NT（如 `128×512×1024`），比较 `d` 与 `torch.relu(a.float() @ b.float().t()).to(torch.bfloat16)`，阈值 1e-5；再用 `DG_PRINT_CONFIGS=1` 记录配置行（`epilogue_type=epilogue::operators::ReLU`、`epi_stages` 变化、`direct_store=0`）。
4. 回归：`python tests/test_bf16_gemm.py normal --skip-prof` 确认默认路径未破坏；顺手跑一个 alpha 变体确认互斥检查仍生效（`resolve_epilogue_class:117`）。
5. 保留改动到明天；第 54 天将补一个"负值占比"的针对性用例（ReLU 只有负值存在时才可观测）。

**项目 D（源码讲解）**

1. 按 3.6 节骨架起草 → 逐节填充（每条论断带 `文件:行号`）→ 每节配图/表。
2. 自查三问；找一位"读者"试讲 15 分钟（或自述录音回听），记录对方没听懂的地方并修订。
3. 产出即最终形态，明天作为"分享材料"复核。

### B 组：无 Ascend 环境（静态任务）

1. **第 0 步照做**：立项文档与 A 组完全相同（这是静态版的全部框架）。
2. **项目 A 静态版**：写出 `pad_outer_stride` 与两处接线的设计（3.3 节已给出，逐行确认自己能解释）；然后手工演算一个 padded 寻址——设 A 为 `[4, 8]` BF16、`pad=4`（stride=(12,1)），按 `gm_ptr::index`（`common.hpp:160`：`addr += sizeof(dtype_t) * (mn_idx * stride_outer + k_idx / pack_size)`，BF16 的 `pack_size=1`）计算 `index(2, 3)` 的字节偏移 = `2 × 2×(12) + 3 × 2 = 54 B`；再算"若 kernel 误用紧凑 stride"的偏移 = `2 × 2×8 + 3 × 2 = 38 B`——两个地址的差异就是外层 stride 测试要捕获的错误。
3. **项目 B 静态版**：完成 3.4 节的候选枚举与预测表（BLOCK_N ∈ {128, 224, 256, 512} 的 waves/单 Tile 面积/手推时长），写出"预测启发式默认选择 vs 门控后选择"，标注将来用 `DG_PRINT_CONFIGS` 验证的核对清单。
4. **项目 C 静态版**：完成 3.5 节的两项手推（UB 布局与 `epi_stages` 变化方向；`kPipe=PIPE_V` 的同步位置 `epilogue.hpp:76-80`）+ 四点改动设计文档。
5. **项目 D**：本身就是静态项目，按 A 组项目 D 的步骤执行即可。
6. 所有静态产出都归档进 `week8_day4_plan.md` 的"预期结果"栏——它们就是未来有环境时的第一份待验证清单。

## 六、当天产出

**一份综合小项目记录**（`week8_day4_plan.md` + 实施产物），至少包含：

1. **立项文档**：3.1 节模板五字段，其中"预期结果"可证伪、"验证方法"含具体命令与通过标准——这是学习计划"先列出修改范围、预期结果和验证方法"的直接证据；
2. **实施产物**（按项目）：A＝`git diff` + 测试输出；B＝补丁 + 配置行对比 + 实验日志表；C＝四点改动 + 验证脚本输出 + 配置行；D＝讲解文档本身；
3. **首轮验证记录**：至少一条正确性证据，以及"实际 vs 预期"的逐项对照——预期落空的地方用一句话归因（改写链？门控没命中？语义理解错？），不求今天解决；
4. **给明天的交接**：改动说明、待跑测试清单、与第 51 天基线的对照项、已知风险；
5. **B 组替代产出**：立项文档 + 对应项目的静态推演（地址演算 / 波形预测表 / UB 布局手推 / 讲解文档）。

## 七、自测题

1. 为什么必须先写下"预期结果"再动手实施？
   > 没有事前预期，任何结果都能被事后合理化，项目退化为"改了代码、跑了测试"。先写预期把项目变成可证伪的实验，与第 52 天"假设闭环"同源——学习计划第 53 天原文即"先列出修改范围、预期结果和验证方法，再开始实施"。
2. `python tests/test_bf16_gemm.py strided` 目前真的在测 strided 输入吗？依据是什么？
   > 没有。两个代码事实：`make_strided`（`generators.py:238-246`）无任何调用方；`GemmTestDesc.strided`（`:261`）被 `_enumerate_normal_strided` 置为 True（`:486`）但 `_generate_normal` 和 `gemm_dg` 都不读取它。实际跑的是"这些 shape + accumulate=True 的连续张量"。
3. 为 strided 测试构造张量时，为什么只能放大外层 stride、内层轴必须保持单位 stride？bias 为什么不能做 stride？
   > Host 侧 `get_major_type_ab`/`major_check`（`csrc/utils/layout.hpp:41-47`）与 `get_outer_stride`（`gemm_config.hpp:27-34`）只认"内层轴单位 stride、外层 stride 任意"的布局；bias 有单独的连续性断言（`csrc/apis/gemm.hpp:114`）。
4. `GemmDesc.outer_stride_*` 从哪里来、到哪里被使用？
   > 来源：`get_gemm_desc` 里 `get_outer_stride(a/b, major)`（`csrc/apis/gemm.hpp:191-193`）；使用：launch 时构造 `gm_ptr` 传入 kernel（`csrc/jit_kernels/bf16_gemm.hpp:74-76`），设备侧 `gm_ptr::index` 用它计算地址（`common.hpp:156-165`）。
5. `get_epilogue_operator_type()` 返回的字符串在流程中扮演什么角色？为什么必须与设备结构体名逐字符一致？
   > 它被 `std::format` 写进 JIT 生成的模板实例化代码（`csrc/jit_kernels/bf16_gemm.hpp:64`），成为设备模板参数 `epilogue_operator_t` 的实参（设备 `bf16_gemm.hpp:23`）。拼错即 JIT 编译失败——这是"响亮的失败"。
6. 自定义非 Identity epilogue 为什么一定走 AIV 路径？ReLU 作用于累加 c 之前还是之后？
   > `can_direct_store` 要求 Identity（`heuristics.hpp:526`），设备侧也有 `static_assert`（设备 `bf16_gemm.hpp:56-57`）。epilogue 只处理本块乘积（L0C→UB 之后），累加 c 是 AIV 写回 GM 时的 atomic add（`epilogue.hpp:91-96`）——所以 ReLU 根本看不到 c，这也是 `ReluEpilogue::check` 排除 c 的原因。
7. 新增非 Identity epilogue 后，配置行的哪个字段预期会变小？为什么？
   > `epi_stages`。`get_gemm_dtype`（`heuristics.hpp:81-82`）里 `with_transform` 使 `epilogue_elem_bits` 从 16 变 32，UB 每 stage 占用翻倍，`make_gemm_candidate`（`:188-190`）相应减少 `num_epilogue_stages`。
8. 今天与明天的分工是什么？
   > 今天：立项 + 实施 + 首轮验证（至少一条正确性证据）+ 预期对照与归因；明天（第 54 天）：完整正确性回归、与第 51 天基线的性能对比、检查新增测试是否真正覆盖目标场景、如实记录失败实验。今天的改动保留在工作区，不提交。

---

明天（第 54 天）进入**验证与修正**：跑与改动相关的全部正确性测试、与基线比较性能、检验新测试的覆盖是否名副其实——今天立项文档里每一栏，明天都要有对应的验证结果。
