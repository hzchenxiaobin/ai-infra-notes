# 第 5 周 · 第 2 天（总第 30 天）：理解正确性验证流程

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 5 周"测试体系与低精度 GEMM"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

昨天我们读懂了测试数据从哪里来：`tests/generators.py` 把 shape × major × recipe × grouped layout 组合成 `GemmTestDesc`，`generate()` 吐出六元组 `(shape, a, b, c, d_ref, (g_layout, cmp_spans))`。今天沿着消费侧走——**这些数据如何被判定为"对"或"错"**。主战场是两个不大的文件：`tests/common.py`（237 行，测试编排）和 `tests/utils.py`（115 行，比较器与打印），交叉佐证散落在 `csrc/apis/aclnn.hpp`（ACLNN 参考实现）、`csrc/jit_kernels/epilogue_class.hpp` 与 `deep_gemm/include/deep_gemm/epilogue/operators.hpp`（随机舍入内核）。要回答四个问题：**一条断言背后有几道验证、各验什么？**（dg 内核 vs 参考结果、ACLNN 交叉验证、直接输出 vs fp32 累加、随机舍入复刻）**accumulation、bias、alpha 的语义在测试端与 host 端如何保持一致？**（`c` 的指针判同与 `[1, N]` bias、互斥检查）**随机舍入为什么不能直接比、必须复刻哈希？**（确定性伪随机 + 逐位比较）**meta 张量并行编译在省什么？**（把 JIT 编译从运行阶段剥离并并发化）。当天产出是**一条完整测试流程图**（参考答案见 3.9 节）。它直接服务明天：第 31 天要实际运行并分析 BF16 测试，判断失败属于哪个层级。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 画出 `run_gemm_test`（`tests/common.py:131-237`）两阶段的完整结构，并按顺序说出阶段 2 的**五类验证**——① dg 内核 vs `d_ref`（`(use_alpha, with_bias)` 的 FF/FT/TF 三次 `assert_equal`）、② 直接输出 vs fp32 累加（k-grouped 专属）、③ ACLNN 交叉验证、④ 随机舍入复刻、⑤ 性能测量——以及 alpha+bias 组合的拒绝检查（`:191-197`）；
2. 解释 **accumulation / bias / alpha 的语义闭环**：`gemm_dg` 端 `with_bias` 优先于 `accumulate`、累加走 `d.copy_(c)` 后原地 `c=d`（`common.py:53-58`）；host 端 `get_gemm_desc` 用**指针判同**区分累加与 bias、要求 bias 连续且与 alpha 互斥、把 bias 包装成 `BiasEpilogue`（`csrc/apis/gemm.hpp:105-123`）；参考公式 `alpha * (a_repr @ b_repr.T) + c` 与之一致（`generators.py:320-322`）；
3. 推导 `calc_diff`（`tests/utils.py:36-43`）的闭式——**diff = ‖x−y‖² / (‖x‖²+‖y‖²)**，即相对平方欧氏距离；解释 `assert_equal` 的双模式（`threshold==0` 逐位 `torch.equal`，`>0` 余弦相似度，`:46-58`）与 spans 的作用；说明为什么 BF16 用 1e-5 阈值（cube 内核与参考的 K 维累加顺序差 ~1 ULP），以及 docstring 里"FP8 MX 可逐位相等"的条件；
4. 列出 `gemm_aclnn`（`common.py:87-128`）的**全部跳过条件**并解释原因（grouped 无 aclnn 对应、K 不是 64 倍数、gran_mn ≠ 1 被拒、MN-major 单 K-pair 退化为 k≤64、fp8×fp4 布局受限）；解释其返回值语义（False=跳过，None=已运行，配合 `has_ref is not False`，`:212`）；说出 C++ 侧 `aclnn_bf16_gemm` 无 c 用 `aclnnMatmul`、有 c 用 `aclnnInplaceAddmm` 的二选一（`csrc/apis/aclnn.hpp:228-254`）；
5. 解释**随机舍入**的动机（fp32→bf16 丢弃 16 位尾数，就近舍入有偏、随机舍入期望无偏）与内核的确定性 quartet 哈希实现（`deep_gemm/include/deep_gemm/epilogue/operators.hpp:224-335`：四个乘法常数、列号混入、NaN 规范化为 0x7fc0、低 16 位补码比较加一）；解释 `assert_stochastic_bf16_matches_fp32`（`utils.py:70-92`）为什么必须在 host 用 numpy 复刻同一哈希、拿 fp32 直出结果当输入、最后**逐位**比较（`common.py:215-221` 的两次运行）；
6. 解释 `assert_direct_output_matches_fp32_accumulation`（`utils.py:61-67`，仅 k-grouped 非累加触发，`common.py:202-206`）验证什么：内核直写路径的内部 cast 与"fp32 累加后 host 端 `.to()`"必须逐位一致，以及 `d` 的 NaN 预填充如何配合空组写零（`common.py:190`）；
7. 解释 **meta 张量并行编译**的机制与目的：`generate(device='meta')` 只产生 shape/stride/dtype（`generators.py:176-186`），`par_compile` 在 `dry_run()` 上下文里用线程池并发执行这些 launch（`deep_gemm/testing/par_compile.py:6-36`），C++ 侧命中 `runtime->get_dry_run()` 分支只调 `jit->compile_without_load` 编译入缓存、不加载不启动（`csrc/jit_kernels/bf16_gemm.hpp:67-70`，开关在 `csrc/jit/jit.hpp:43-49`）；说明 trans_ab 与 alpha 数值为何不进入生成的代码串（第 29 天 3.3 的抵消转置 + epilogue 只烙类型名，`bf16_gemm.hpp:64`），从而大量变体被 JIT 缓存去重；
8. 解释 `disable_cpu_fallback`（`deep_gemm/testing/disable_fallback.py:44-73`）的护栏作用——给所有"没有 NPU 内核、会静默回落 CPU"的算子注册抛 `RuntimeError` 的 PrivateUse1 实现，保证参考计算真的在 NPU 上跑；
9. 读懂 `print_perf` / `count_bytes`（`utils.py:95-115`）与 `bench_msprof`（`deep_gemm/testing/bench.py:284-400`，10 次 warmup + 30 次迭代 + L2 刷新）的输出字段（us / TFLOPS / GB/s / mad% / mte2%），为第 50 天的性能分析打底。

**当天产出：一条完整测试流程图**（模板见第六节，参考答案见 3.9 节）。

## 二、预备知识

- **第 29 天**：六元组的含义、`generate()` 的确定性（每次重播种）、`cmp_spans` 的来源、`d` 的 NaN/随机预填充——今天全部要接上消费方。
- **第 10～12 天**：Python binding → `get_gemm_desc` → `select_gemm_config` → JIT 编译与启动的调用链；`GemmDesc` 与 `GemmConfig` 的区别；JIT 内核缓存的作用——今天的 meta 并行编译建立在这条链上。
- **第 24～25 天**：MAD 累加发生在 L0C（fp32），结果经 Epilogue（cast/bias/alpha）写回——今天讲的"直写 vs 累加""随机舍入 vs 就近舍入"都发生在这一段。
- **第 26 天**：`direct_store` 路径及其约束（`can_direct_store` 要求 `!acc || cd_dtype == fp32`，`csrc/jit_kernels/heuristics.hpp:528`）；k-grouped 空组由 AIV 写零——`assert_direct_output_matches_fp32_accumulation` 与 NaN 预填充分别对应这两点。
- **第 4 天**：BF16 有 8 位尾数，fp32→bf16 舍入丢弃 16 位——随机舍入的出发点。
- **Python 语法**：`functools.partial`、`contextmanager`、`concurrent.futures.ThreadPoolExecutor`、`dataclasses.replace`；numpy 的 `uint32` 溢出回绕语义（今天手算哈希会用到）。

另注：本文只讲**验证侧**。数据如何生成（含 `d_ref` 的构造细节）是第 29 天的内容；SF 布局转换（`transform_sf_into_required_layout`，ACLNN 交叉验证里 `convert_sf_for_aclnn` 的近亲）是第 33 天的内容；性能指标的含义展开是第 50 天的内容。

## 三、详细讲解

### 3.1 全景：一次测试运行的验证序列

先建立整体图景。`tests/test_bf16_gemm.py` 等 CLI 脚本只是选择器（第 29 天），一切编排都在 `common.run_gemm_test(a_dtype, b_dtype, gemm_type, …, threshold=1e-5)`（`common.py:131-141`）里。四个 GEMM 脚本（bf16/fp8/fp4/fp8_dequant）都传 `threshold=1e-5`（如 `test_bf16_gemm.py:17`）。函数体分两阶段：

```text
阶段 1 · meta 并行预编译（common.py:152-178）
  for test in tests:                          # 每个 desc
    for with_bias in (False, True) if Normal else (False,):
      compile_test.generate(device='meta')    # :157 只含 shape/stride/dtype
      for trans_ab in 可用组合:
        收集 partial(gemm_dg, …)              # :159 普通
        Normal 无 bias 再加 alpha 变体        # :160-164
        k-grouped 非累加再加 fp32 累加变体    # :165-168
        同 dtype 非累加 bf16 输出再加 SR 与 fp32 直出变体   # :169-174
  par_compile(kernels)                        # :175 dry_run 下线程池并发触发 JIT
  del + gc.collect + npu.empty_cache          # :176-178

阶段 2 · 运行与验证（common.py:180-236）
  for test in tests:
    for trans_ab in 可用组合:                  # Normal 有 4 种
      for (use_alpha, with_bias) in FF/FT/TF/TT:
        generate() → d 预填充 → [TT: 拒绝检查] / [其余: gemm_dg → sync → ① assert_equal]
        [k-grouped 非累加: ② direct vs fp32 累加]
      再 generate() 一次 + d=randn:
        ③ gemm_aclnn 交叉验证（仅 Normal 且未被跳过条件拦截）
        [同 dtype 非累加 bf16 输出: ④ 随机舍入验证]
    [skip_prof=False 且 test.profile 且 k>0: bench_msprof]
    print_perf(...)                            # 每个 desc 一行输出
```

三类**前置护栏**值得先记住：`disable_cpu_fallback()`（`:142`）防止参考计算静默回落 CPU；`d` 的预填充策略（`:190`，k-grouped 非累加填 NaN、其余填 `randn` 垃圾）保证内核必须**完整覆写**输出（漏写任何区域、包括空组写零，都会被抓住）；`torch.npu.synchronize()`（`:200, :211`）保证断言前内核已执行完毕。

一个 Normal 用例（如 bf16 `normal` 的某个 desc）在阶段 2 的验证次数可以精确数出来：4 种 trans_ab × [3 次 dg 断言（FF/FT/TF）+ 1 次 TT 拒绝检查 + 1 次 ACLNN 断言 + 1 次随机舍入断言（条件满足时）] = 每个 desc 最多 20 次断言、20 次 `generate()`（第 29 天 A-5 的口径）。

### 3.2 参考结果的语义：accumulation、bias 与 alpha

验证的基准是 `d_ref`，它的公式在生成侧（第 29 天）：

```python
d = (alpha_value * (a_repr.float() @ b_repr.float().T)
     + (c if c is not None else 0)).to(self.cd_dtype)      # generators.py:320-322
```

三种 Epilogue 语义全部体现在这一行里。**关键在于测试端与 host 端对 `c` 的解释必须一致**。先看测试端 `gemm_dg`（`common.py:49-84`）如何处理 `c`：

```python
if test.with_bias:
    extra_args['c'] = c            # bias：c 形状 (1, n)，原样传入
elif test.accumulate:
    assert test.gemm_type == GemmType.Normal or ... == KGrouped...
    d.copy_(c)                     # 先把 c 拷进 d
    extra_args['c'] = d            # 再把 d 自己作为 c 传入 → 原地累加
```

（`common.py:53-58`。）注意两点：**`with_bias` 优先于 `accumulate`**——`(F, T)` 组合即使 desc 带 `accumulate=True`，测的也是 bias（此时 `generate` 因 `with_bias=True` 把 c 造 成 `(1, n)`，`generators.py:304-308`）；**累加是"原地"约定**——`c` 与 `d` 必须是同一个张量。k-grouped 的参考公式相应为 `d[i] = (c[i] if acc else 0) + (A[ks]ᵀ·B[ks]).to(cd_dtype)`（`generators.py:396-397`），乘积先 cast 再加 c，与"直写 vs fp32 累加"两种内核路径的语义对应（3.6 节）。

再看 host 端 `get_gemm_desc` 如何反过来推断语义（`csrc/apis/gemm.hpp:105-123`）：

```cpp
// Meta and empty tensors have no data address; use storage identity for them.
const bool is_cd_same = c.has_value() &&
    ((d.is_meta() || d.numel() == 0) ? c->is_alias_of(d) : c->data_ptr() == d.data_ptr());
const bool with_bias = gemm_type == GemmType::Normal && c.has_value() && !is_cd_same;
if (gemm_type == GemmType::Normal && c.has_value()) {
    DJ_HOST_ASSERT(c->dim() == 2 && c->size(0) == (with_bias ? 1 : m) && c->size(1) == n, ...);
    DJ_HOST_ASSERT(c->device() == d.device(), ...);
}
if (with_bias) {
    DJ_HOST_ASSERT(c->is_contiguous(), "Bias must be contiguous");
    DJ_HOST_ASSERT(dynamic_cast<const AlphaEpilogue*>(epilogue_class.get()) == nullptr,
                   "alpha and bias cannot be used together");
}
if (with_bias)
    epilogue_class = std::make_shared<BiasEpilogue>(c.value());   // :122-123
```

三个要点：**指针判同**决定累加（`c` 就是 `d`）与 bias（`c` 是独立的 `[1, N]`）的分界——meta 张量没有数据地址，改用 `is_alias_of` 判存储同一性（这正 是 3.7 节 meta 编译能走通的前提）；**alpha 与 bias 互斥**，测试端用 `(T, T)` 组合专门验证这条拒绝路径（`common.py:191-197` 断言报错信息含 `alpha and bias cannot be used together`）；bias 在这里被包装成 `BiasEpilogue`。`alpha` 只在 Normal 的 `gemm_dg` 里暴露（`common.py:59-60`），host 端由 `resolve_epilogue_class` 解析成 `AlphaEpilogue`（`csrc/jit_kernels/epilogue_class.hpp:110-128`，epilogue 类实例 / alpha / sfd 三者互斥）。

五个 Epilogue 类（`epilogue_class.hpp`）与内核操作符的对应，是理解验证覆盖面的钥匙：

| Python 侧 | C++ 类 | 内核操作符类型名（烙进代码） | 验证位置 |
|---|---|---|---|
| （缺省） | `IdentityEpilogue`（`:29`） | `epilogue::operators::Identity` | dg 断言 ① |
| `alpha=…` | `AlphaEpilogue`（`:36`） | `ScaleByAlpha` | dg 断言 ① 的 TF 组合 |
| `c=(1,N)` | `BiasEpilogue`（`:51`） | `AddBias` | dg 断言 ① 的 FT 组合 |
| `epilogue.BF16StochasticRounding()` | `BF16StochasticRoundingEpilogue`（`:66`） | `StochasticRoundToBF16` | 断言 ④ |
| `(d, sfd)` 对 | `FP8QuantizationEpilogue`（`:78`） | `QuantizeToFP8` | GEMM 主测试不覆盖，einsum/其他路径用 |

另外留意退化 shape 的早退：`gemm_early_return`（`csrc/utils/layout.hpp:54-86`）在 `m==0 || n==0` 时直接返回，`k==0` 时把 `d` 置为 `c`（或清零）——"空 GEMM 也有确定的输出语义"。

### 3.3 误差比较：calc_diff 与 assert_equal 的双模式

`assert_equal`（`utils.py:46-58`）是比较的入口，它有两种截然不同的模式：

```python
def assert_equal(out, ref, msg, slices=None, threshold=0.0):
    """... threshold == 0 demands bit-for-bit equality (FP8 MX: identical cube +
    cast path). threshold > 0 uses calc_diff (BF16: the cube kernel and aclnn differ
    by up to ~1 ULP from different K-accumulation, so an exact match is not expected)."""
    spans = slices if slices is not None else [slice(0, out.size(0))]
    for s in spans:
        if threshold == 0.0:
            assert torch.equal(out[s], ref[s]), f'{msg} (rows {s})'
        else:
            diff = calc_diff(out[s].float(), ref[s].float())
            assert diff < threshold, f'{msg} (rows {s}, diff={diff:.2e})'
```

**模式一（threshold==0，函数默认值）**：`torch.equal` 逐位相等。docstring 说明其适用场景是"FP8 MX：相同的 cube + cast 路径"——当参考结果与内核走的是同一套数值路径（码字精确解码 + 2 的幂 SF），理论上可以位级一致。注意**四个 GEMM CLI 脚本全都传 `threshold=1e-5`**（`test_bf16_gemm.py:17` 等），`assert_equal` 的默认值 0 只在其他调用方生效；einsum 测试用自己的阈值（`test_einsum.py:67`：fp8 用 1.5e-3、bf16 用 1.5e-4，注释说明 FP8 参考是手工反量化路径、BF16 与 fp32 矩阵乘差几个 ULP）。

**模式二（threshold>0）**：`calc_diff` 的余弦相似度差异。`calc_diff`（`utils.py:36-43`）定义为 `diff = 1 − 2⟨x,y⟩/(⟨x,x⟩+⟨y,y⟩)`，分母为零（双全零）返回 0。对它做一个代数化简，含义立刻清晰：

```text
diff = (⟨x,x⟩+⟨y,y⟩−2⟨x,y⟩) / (⟨x,x⟩+⟨y,y⟩)
     = ‖x−y‖² / (‖x‖²+‖y‖²)          ← 相对平方欧氏距离
```

这是一个**尺度不变**的相对误差：整体乘上常数不改变 diff，单个元素的 1 ULP 误差贡献被全体元素平方和摊薄。手算两个例子（bf16 在 1.0 附近的 ULP 是 2⁻⁷ = 0.0078125）：

- 单元素 `x=[1.0]`、`y=[1.0078125]`（差 1 ULP）：diff = (0.0078125²)/(1+1.01569) ≈ **3.03e-5 > 1e-5**；
- 16384 个元素中 1 个差 1 ULP：diff ≈ 6.1e-5/(2×16384) ≈ **1.9e-9 ≪ 1e-5**。

所以 1e-5 的阈值允许"少数元素差 1～2 ULP"、拒绝"系统性偏差"——这正是 BF16 场景需要的：cube 内核按 BLOCK_K/MAD_K 分块累加，参考 `a_repr.float() @ b_repr.float().T` 是 torch 的另一套累加顺序，浮点加法不满足结合律，两者天然差几个 ULP；而任何真实的布局/调度/cast 错误都会产生远超阈值的差异。

`spans` 参数（即 `cmp_spans`）把比较收紧到各组有效行——m-grouped 在 `ensure_zero_padding=False` 时只比有效区间（第 29 天 3.8），dg 与 aclnn 两次断言都带上它（`common.py:201, :213`）。

### 3.4 ACLNN 交叉验证：第二参考实现

断言 ① 只证明"dg 内核 == torch 参考公式"。`gemm_aclnn`（`common.py:87-128`）引入**第二个独立实现**——CANN 官方算子库 ACLNN，它同样要与 `d_ref` 一致：

```python
def gemm_aclnn(test, a, b, c, d, g_layout=None, trans_ab='nt'):
    # ACLNN only provides a reference for the dense (Normal) GEMM; m-/k-grouped ops have
    # no aclnn counterpart, so skip the cross-check for them.
    if test.gemm_type != GemmType.Normal:
        return False
    if test.accumulate:
        d.copy_(c)                                  # 与 dg 相同的原地累加约定
    if test.gemm_type == GemmType.Normal:
        a = a if trans_ab[0] == 'n' else get_transposed(a, 0, 1)
        b = b if trans_ab[1] == 't' else get_transposed(b, 0, 1)
        if test.has_sf:
            ...若干跳过条件...
            getattr(deep_gemm, f'aclnn_fp8_fp4_gemm_{trans_ab}')(a, b, d,
                d if test.accumulate else None, recipe_a=test.a_recipe, recipe_b=test.b_recipe)
        else:
            # ACLNN follows the same in-place accumulation convention for `c`.
            getattr(deep_gemm, f'aclnn_bf16_gemm_{trans_ab}')(a, b, d, d if test.accumulate else None)
```

调用侧（`common.py:208-213`）先 `d = torch.randn_like(d_ref)` 填垃圾，再 `has_ref = gemm_aclnn(...)`，然后 `if has_ref is not False: assert_equal(d, d_ref, f'aclnn: …')`。**返回值语义是个小机关**：函数没有显式的 `return True`，成功路径的返回值是 `None`；`is not False` 让 None（已运行）通过、False（跳过）不比较。累加时 `d.copy_(c)` 先行（`:92-93`），与 `gemm_dg` 的 `d.copy_(c)` 镜像——两条实现遵守同一套原地累加约定。

**跳过条件全表**（fp8/fp4，`common.py:97-122`）：

| 条件 | 行号 | 原因（源码注释） |
|---|---|---|
| `gemm_type != Normal` | `:90-91` | m-/k-grouped 没有 aclnn 对应算子 |
| `test.k % 64 != 0` | `:98-99` | "ACLNN requires K to be a multiple of 64" |
| `test.a_recipe[0] != 1 or test.b_recipe[0] != 1` | `:104-105` | aclnnQuantMatmulV5 的 MX (ue8m0) 路径只支持逐行 1×32 标度，gran_mn ≠ 1 报 "Unsupported groupSize" |
| `ceil(k/32) <= 2` 且任一操作数 MN-major | `:111-112` | aclnn 把 MN-major 单 K-pair 的 scale `[k/64, rows, 2]` 的前导单例误读为 n=1（即 k ≤ 64 的退化形状） |
| fp8×fp4 且不满足"fp8 K-major × fp4 K-major + bf16/fp16 输出" | `:113-122` | 见 aclnn 支持矩阵（`aclnn.hpp:96-103`）：fp4 只能做 x2、fp8-MN-major×fp4 不支持等 |

被跳过时 dg 结果仍由断言 ① 对着 `d_ref` 验证（注释明确说明），交叉验证只是"锦上添花的第二意见"。bf16 路径无额外条件。

**C++ 侧的实现**（`csrc/apis/aclnn.hpp`）：

- `aclnn_bf16_gemm<kTransA, kTransB>`（`:208-255`）：无 `c` 时用 `aclnnMatmul`；有 `c`（即测试的累加场景，`c` 就是 `d`）时用 `aclnnInplaceAddmm`，α=β=1、KEEP_DTYPE，原地算 `D = D + A·Bᵀ`（`:229-241, :250-254`）。
- `aclnn_fp8_fp4_gemm`（`:121-190`）：走 `aclnnQuantMatmulV5`，操作数按**自然方向**传入（`transposeX1=false, transposeX2=true`，aclnn 尊重真实 stride，`:171`）；`K % 64 == 0` 有硬断言（`:140`）；scale 由 `convert_sf_for_aclnn`（`:54-83`）转成 `[mn, K/64, 2]`（MN-major 操作数则转置为 `[K/64, mn, 2]`）的 ue8m0 布局——这是第 33 天 `transform_sf_into_required_layout` 的近亲，今天知道"ACLNN 有自己的一套 scale 布局要求"即可；无原地累加变体，故先算到临时 `mm_out` 再 `d.copy_(c); d.add_(mm_out)`（`:154, :186-189`）。

第 29 天 3.3 埋的伏笔在这里兑现：`aclnn_bf16_gemm` 对 B 的规范化方向与 dg 相反——`get_transposed<!kTransB>`（`aclnn.hpp:216`）而 `bf16_gemm` 绑定用 `get_transposed<kTransB>`（`gemm.hpp:435`）。因此 dg 的四种 trans_ab 最终进内核的视图相同，而 **ACLNN 的四种 trans_ab 确实走不同的规范化路径**——交叉验证在"绑定别名的正确性"上比 dg 断言覆盖更广。

### 3.5 随机舍入：为什么需要特殊的比较方法

**问题背景**：fp32 有 24 位有效尾数，bf16 只有 8 位，fp32→bf16 舍入要丢 16 位。默认的就近舍入（round-to-nearest-even，如 `AddBias` 路径的 `asc_float2bfloat16_rn`，`operators.hpp:215`）是**有偏**的——大量小数值会被系统性舍为零，训练中梯度累加因此失真。**随机舍入（SR）**以正比于被丢弃小数部分的概率向上舍入，期望意义下无偏。项目把它做成了一个 epilogue：`deep_gemm.epilogue.BF16StochasticRounding()`（`epilogue_class.hpp:66-76`，要求非累加且 bf16 直出）。

**难点**：SR 是"随机"的——同一个输入两次运行结果可以不同，断言 ① 的两种模式都不适用（位级必败，余弦阈值说明不了舍入行为对不对）。项目的解法分两步。

第一步，**让随机性变成确定性的**。内核 `StochasticRoundToBF16`（`deep_gemm/include/deep_gemm/epilogue/operators.hpp:224-335`）的随机数不是计数器或熵源，而是一个**由输入位模式和位置决定的哈希**：每次处理 4 个连续列（quartet），

```text
h  = v0·0x5671d42b + v1·0x9995e499 + v2·0xace1b8a5 + v3·0xe153538d + quartet_index   (mod 2³²)
NaN 输入先参与哈希、再规范化为 0x7fc00000                                            (:277-296)
h  = (h ^ h>>23) · 0x7feb352d;  h ^= h>>16                                           (:309-313)
h1 = h·0x846ca68b ^ 自身>>11;   h2 = h·0xd35a2d97 ^ 自身>>11                          (:314-320)
每个成员分得 16 位随机数 R ∈ {h1 低, h1 高, h2 低, h2 高}                              (:322-328)
bf16 结果 = 高 16 位 + (低 16 位的补码 < R ? 1 : 0)                                   (:323-325)
```

`v0..v3` 是四个 fp32 的位模式（按 uint32 读）。向上舍入的条件等价于 `low16 + R > 0xffff`，即 **P(向上) ≈ low16/2¹⁶ = 被丢弃的小数部分**——这正是 SR 的定义。同一位置、同一位模式 → 同一 R → 同一结果。

第二步，**host 复刻这个哈希做精确验证**。`assert_stochastic_bf16_matches_fp32`（`utils.py:70-92`）的流程：

```text
① common.py:215-217  gemm_dg(..., epilogue=BF16StochasticRounding()) → 内核输出 bf16 的 d
② common.py:218-220  再跑一次 fp32 直出：Normal 直接把 d 换成 fp32 buffer；
                     k-grouped 用 accumulate=True + cd_dtype=fp32 的变体（累加到清零的 buffer 上）
③ utils.py:72-91     拿 fp32 值在 CPU 上用 numpy 复刻同一套哈希（同一组常数、同一混叠、
                     同一 NaN 规范化、pad 到 4 的倍数保证越界列是确定性零），
                     算出期望的 bf16 位模式
④ utils.py:92        assert torch.equal(output.cpu().view(torch.int16), expected...) —— 逐位相等
```

于是验证从"统计上接近"升级为"**每一位都和正确实现的 SR 一致**"：哈希常数抄错、列号没混入、NaN 处理遗漏、越界列未清零，任何一处偏差都会导致位级不匹配。注释 `# CUDA's quartet hash, with deterministic zeros for out-of-bounds columns`（`utils.py:74`）点明这套哈希与上游 CUDA 版 DeepGEMM 同源。还要注意 `torch.empty_strided(output.shape, output.stride(), …)`（`utils.py:72`）——fp32 参考与 bf16 输出保持相同形状**和跨步**，跨步/分组输出也能对齐。

**触发条件**（`common.py:215`）：`a_dtype == b_dtype and not test.accumulate and test.cd_dtype == torch.bfloat16`。容易忽略的是它**不限 bf16 输入**——fp8×fp8、fp4×fp4 的同 dtype 非累加 bf16 输出用例同样触发（累加在 L0C 中仍是 fp32，出 L0C 才舍入）。阶段 1 为此预编译了 SR 变体和 fp32 直出变体（`common.py:169-174`）。

### 3.6 k-grouped 专属：直接输出 vs fp32 累加

断言 ②（`common.py:202-206` → `utils.py:61-67`）只在 k-grouped 非累加时触发：

```python
def assert_direct_output_matches_fp32_accumulation(output, launch, label):
    assert output.dtype in (torch.float32, torch.bfloat16)
    fp32 = torch.zeros_like(output, dtype=torch.float32)
    launch(fp32, fp32)                       # 跑 fp32 累加变体：D = 0 + A·B（fp32 原地）
    expected = fp32.to(output.dtype)         # host 端 cast
    bits_dtype = torch.int16 if output.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(output.view(bits_dtype), expected.view(bits_dtype)), label
```

它验证的是**内核直写路径的 cast 语义**：直写（direct store）时内核在内部完成"L0C fp32 → cd_dtype"的转换并直接写 GM（第 26 天），此输出必须与"fp32 累加后在 host 用 `.to()` 转换"逐位一致——两者都只做一次舍入，舍入方式必须相同。`common.py:202-206` 传入的 `launch` 闭包把 `fp32_test`（`accumulate=True, cd_dtype=torch.float32` 的变体，阶段 1 已在 `:165-168` 预编译）跑在清零的累加器上。结合 `d` 的 NaN 预填充（`:190`）：k-grouped 非累加时 `d` 先填 NaN，内核若漏写空组区域（`aiv_zero_empty_groups` 的写零路径，第 26 天），断言 ① 立刻失败——两个检查一个管"写没写"、一个管"cast 对不对"。

### 3.7 meta 张量并行编译

**要解决的问题**：JIT 是这套测试最大的固定开销——每个 (desc 配置, epilogue 类型, cd_dtype) 组合都要生成一份模板实例化代码、调用编译器编译、加载 so（第 12 天）。如果这些发生在阶段 2，后果有二：NPU 在编译期间空转；单次编译动辄数秒，串行编译让整个测试慢到不可用。

**机制**（三条腿）：

1. **元张量**：`compile_test.generate(device='meta')`（`common.py:157`）走 `generators.py:176-186` 的 meta 分支——`torch.empty(storage_shape, …, device='meta')` + 转置，只有 shape/stride/dtype，**不占任何 NPU 存储**。host 路径（`get_gemm_desc` 读形状推断 major、`select_gemm_config` 选配置、生成代码）只依赖这些元信息；`d.is_meta()` 时指针判同退化为 `is_alias_of`（`gemm.hpp:105-106`），累加路径的 `d.copy_(c)`（`common.py:57`）在 meta 张量上是合法的形状检查。
2. **dry_run**：`par_compile`（`deep_gemm/testing/par_compile.py:16-36`）把所有 `partial(gemm_dg, …)` 放进 `dry_run()` 上下文（`:6-13`：`set_dry_run(True)` … `set_dry_run(False)`，binding 在 `csrc/apis/runtime.hpp:46-50`）。C++ 侧 `launch_bf16_gemm` 走到分叉点：

   ```cpp
   if (runtime->get_dry_run()) {
       jit->compile_without_load("bf16_gemm", code);   // 只编译入缓存
       return;
   }
   const auto kernel = jit->compile("bf16_gemm", code);
   jit->launch(kernel, config.launch_options, …);      // 正常路径：编译+加载+启动
   ```
   （`csrc/jit_kernels/bf16_gemm.hpp:67-77`；开关存取在 `csrc/jit/jit.hpp:43-49`。）即 dry_run 下**完整走一遍 host 路径**（参数检查、配置选择、代码生成），只把最后的"加载 + 启动"换成"编译入缓存"。
3. **并行**：`ThreadPoolExecutor`（`max_workers` 取 `DG_TEST_MAX_WORKERS` 或 CPU 数，`par_compile.py:27`）并发执行这些 partial，tqdm 显示进度。用线程而非进程是因为 JIT 运行时与内核缓存是**进程内全局**（`jit` 的 `LazyInit`，`jit.hpp:16`），线程天然共享；编译本身是外部编译器子进程，不是纯 Python 计算。

**代码串去重**是它便宜的关键。生成的代码串只包含类型与配置（`bf16_gemm.hpp:18-66`）：`major_a/major_b`、编译期维度、`GemmConfig` 各字段、`acc`、cd 类型、**epilogue 操作符的类型名**（`:64`，如 `"epilogue::operators::Identity"`）。由此推出两条：第 29 天已证四种 trans_ab 经两次抵消转置后 desc 完全相同 → **同一 epilogue 类型的 4 种 trans_ab 生成同一份代码**；alpha 的**数值**是 launch 参数（`AlphaEpilogue::make_epilogue_operator_args` 返回 `.alpha`，`epilogue_class.hpp:46-48`），不进代码串 → 普通/alpha 变体只差类型名。以 unaligned-k 的一个 desc 为例：阶段 1 收集 12 个 partial（4 trans × {普通, alpha, bias}，第 29 天 A-5），但只有 **3 份不同代码**（Identity / ScaleByAlpha / AddBias），其余全部命中 JIT 缓存（第 12 天）。' fwd' 类 desc（非累加 bf16 输出）的 20 个 partial 对应 5 份代码（再加 StochasticRoundToBF16 与 Identity+fp32）。

编译完成后 `del kernels, a, b, c, d_ref` + `gc.collect()` + `torch.npu.empty_cache()`（`common.py:176-178`）释放元张量与编译期内存，阶段 2 的每次 launch 都命中缓存——**运行阶段的耗时与断言才真正反映内核行为**，不被编译噪声污染（这也是 `--skip-prof` 只影响 bench、不影响编译策略的原因）。

### 3.8 护栏与输出：disable_cpu_fallback、bench 与 print_perf

**`disable_cpu_fallback`**（`deep_gemm/testing/disable_fallback.py:44-73`）：`run_gemm_test` 的第一行（`common.py:142`）。它遍历 torch 全部算子名，找出在四个"真实现"dispatch key（PrivateUse1、Composite* 等，`:4-9`）下都没有内核的算子，给它们注册一个**抛异常**的 PrivateUse1 实现：

```python
raise RuntimeError(
    f"[npu_no_fallback] operator '{op_name}' has no NPU kernel and "
    f"would fall back to the CPU. inputs=…")
```

（`:29-41`。）没有它，torch-npu 对个别算子会**静默把张量搬到 CPU 计算**——测试照样能过，但参考结果不再是"NPU 上的独立计算"，测试的意义被掏空，还拖慢速度。装上后任何回落立即变成显式失败。`allow` 参数可豁免个别算子；注册结果用模块级 `_libs` 保活防止 GC 撤销（`:12-13`）。

**性能侧**（今天只认识字段，展开留给第 50 天）：`skip_prof=False` 且 `test.profile` 且 k>0 时（`common.py:226-228`），`bench_msprof(lambda: gemm_dg(...), kernel_names='gemm_impl')`（`deep_gemm/testing/bench.py:284-400`）默认 10 次 warmup + 30 次迭代、每次迭代前清一块 2×L2 大小的 buffer 刷缓存（`:263-268, :344-346`），从 PMU 采集每个流水线的占用率。`print_perf`（`utils.py:106-115`）打印：

```text
 > Perf (m=  128, n=  128, k=  256, layout=NT, bf16, acc=0,  fwd):   12.345 us |  512 TFLOPS |  1024 GB/s | mad= 88.1% | mte2= 45.2%
```

（示意格式）其中 TFLOPS = 2mnk/耗时、GB/s = `count_bytes(a,b,c,d)`/耗时（`utils.py:95-103` 会展开 (data, sf) 二元组逐张量计字节）；`--skip-prof` 时该行以 `OK` 结尾。`kernel_names='gemm_impl'` 按子串**唯一匹配**内核名，匹配到多个会报错（`bench.py:271-281`）。

### 3.9 参考答案：完整测试流程图（当天产出）

```text
python tests/test_bf16_gemm.py normal --skip-prof
   │  test_bf16_gemm(GemmType.Normal, skip_prof=True, threshold=1e-5)（test_bf16_gemm.py:38,17）
   ▼
common.run_gemm_test(bf16, bf16, Normal, …)（common.py:131）
   │  ⓪ disable_cpu_fallback()（:142 → disable_fallback.py:44）
   │     护栏：参考计算中任何算子若回落 CPU 立即 RuntimeError
   │  tests = list(enumerate_gemm_tests(...))（:149，第 29 天：desc 枚举）
   ▼
┌─ 阶段 1 · meta 并行预编译（common.py:152-178）──────────────────────────────┐
│ per desc × with_bias(Normal 叉乘 False/True) ：                              │
│   compile_test.generate(device='meta')（:157）                               │
│     → 只有 shape/stride/dtype 的元张量（generators.py:176-186，无 NPU 存储）│
│   per trans_ab：收集 partial(gemm_dg, …)（:159）                            │
│     + alpha 变体（:160-164） + k-grouped fp32 累加变体（:165-168）          │
│     + SR 变体 / fp32 直出变体（:169-174）                                   │
│ par_compile(kernels)（:175 → par_compile.py:16）                            │
│   dry_run()：set_dry_run(True)（par_compile.py:6-13）                       │
│   ThreadPoolExecutor 并发执行全部 partial → C++ 完整走 host 路径            │
│   launch_bf16_gemm 命中 dry_run 分支：jit->compile_without_load             │
│   （bf16_gemm.hpp:67-70：只编译入缓存，不加载不启动）                       │
│   代码串只含类型/配置 → trans_ab 与 alpha 数值不进代码 → JIT 缓存去重      │
│ del + gc.collect() + npu.empty_cache()（:176-178）                          │
└──────────────────────────────────────────────────────────────────────────────┘
   ▼
┌─ 阶段 2 · 运行与验证（common.py:180-236）──────────────────────────────────┐
│ per desc：                                                                   │
│  per trans_ab（Normal 4 种 / M-grouped nt / K-grouped tn）：                │
│   per (use_alpha, with_bias) ∈ {FF, FT, TF, TT}（Normal；grouped 仅 FF）： │
│    alpha_test.generate()（:189，确定性种子 → 数据与 desc 参数完全对应）     │
│    d 预填充：k-grouped 非累加 → NaN；否则 randn 垃圾（:190）               │
│    [TT] 期望 RuntimeError('alpha and bias cannot be used together')         │
│         （:191-197 ↔ gemm.hpp:113-116 互斥检查）                            │
│    [FF/FT/TF] gemm_dg(...)（:199）→ torch.npu.synchronize()（:200）→        │
│      ① assert_equal(d, d_ref, cmp_spans, threshold=1e-5)（:201）           │
│         threshold>0 → calc_diff = ‖x−y‖²/(‖x‖²+‖y‖²) < 1e-5               │
│         （utils.py:36-58；bf16 容 K 累加顺序差 ~1 ULP，拒系统性偏差）      │
│    [k-grouped 非累加]                                                       │
│      ② assert_direct_output_matches_fp32_accumulation（:202-206 →          │
│         utils.py:61-67：直写 cast vs fp32 累加后 host cast，逐位相等）     │
│   再 test.generate() 一次（:208）+ d = randn（:209）：                      │
│    ③ gemm_aclnn（:210-213，仅 Normal 且未被跳过条件拦截）：                │
│         ACLNN 官方库独立实现 → assert_equal(d, d_ref, threshold)           │
│         （跳过条件：grouped / K%64 / gran_mn≠1 / MN-major 单 K-pair /      │
│           fp8×fp4 布局受限，common.py:90-122；has_ref is not False）       │
│    [a_dtype==b_dtype 且非累加且 bf16 输出]                                  │
│      ④ 随机舍入（:215-221）：                                               │
│         gemm_dg(epilogue=BF16StochasticRounding()) → bf16 的 d             │
│         再跑 fp32 直出（Normal 换 fp32 buffer / k-grouped 累加变体）       │
│         → host numpy 复刻 quartet 哈希算期望（utils.py:70-92）             │
│         → torch.equal 逐位比较（utils.py:92）                               │
│  [skip_prof=False 且 profile 且 k>0]                                        │
│    ⑤ bench_msprof(..., kernel_names='gemm_impl')（:226-228）               │
│  print_perf（:230-236）：us / TFLOPS / GB/s / mad% / mte2% 或 OK            │
└──────────────────────────────────────────────────────────────────────────────┘
   ▼
全部断言通过 → 每个 desc 打印一行 ' > Perf (...) OK'/性能行
任一断言失败 → AssertionError，消息含 'dg: {desc} {trans}' / 'aclnn: …' /
              'direct output: …' / 'stochastic rounding: …' 定位信息
```

## 四、源码导读

建议按站点顺序阅读，总计约 40 分钟。所有行号以仓库当前版本为准。

### 站点 1：比较器全家（`tests/utils.py` 全文，115 行）

`calc_diff`（`:36-43`，对着 3.3 的闭式推导读）、`assert_equal`（`:46-58`，双模式与 spans）、`assert_direct_output_matches_fp32_accumulation`（`:61-67`）、`assert_stochastic_bf16_matches_fp32`（`:70-92`，逐行对照 3.5 的四步：pad→哈希→混叠→舍入→NaN→比较）、`count_bytes`（`:95-103`）、`print_perf`（`:106-115`）。

### 站点 2：dg 启动器（`tests/common.py:42-84`）

`get_transposed`（`:42-46`，对 (data, sf) 二元组同时转置）、`gemm_dg` 三种类型的 API 选择：Normal 的 alpha/recipe/trans 处理（`:59-66`）、m-grouped 的 `nt` 约束（`:71-72`）、k-grouped 的单一 `recipe` 与 `host_ks_cpu`（`:74-82`，注释解释了为何等价）。

### 站点 3：ACLNN 交叉验证（`tests/common.py:87-128`）

对照 3.4 的跳过条件表逐条读注释——每条都写明了上游 aclnn 的限制原因；注意 `:92-93` 的 `d.copy_(c)` 与返回值语义（无显式 `return True`）。

### 站点 4：编排主体（`tests/common.py:131-237`）

阶段 1 的变体收集（`:153-174`，五种 partial 各自的条件）与清理（`:176-178`）；阶段 2 的验证序列（`:185-221`，重点 `:190` 预填充、`:191-197` 拒绝检查、`:201/:213` 两次 assert_equal 的标签差异、`:215-221` SR 的两次运行）、性能与打印（`:223-236`）。

### 站点 5：ACLNN 参考实现（`csrc/apis/aclnn.hpp`）

`aclnn_bf16_gemm`（`:208-255`）：matmul/addmm 二选一、`get_transposed<!kTransB>` 的反向规范化（`:216`）；`aclnn_fp8_fp4_gemm`（`:121-190`）：`K%64` 断言（`:140`）、dtype/major 支持矩阵注释（`:96-103`）、`convert_sf_for_aclnn` 的 `[mn, K/64, 2]` 布局（`:54-83`，浏览即可，细节留第 33 天）、临时缓冲累加（`:186-189`）；binding 的 `c = std::nullopt` 缺省（`:285-288`）。

### 站点 6：Epilogue 类与 SR 内核（两份文件对照）

`csrc/jit_kernels/epilogue_class.hpp`：五个类的 `get_epilogue_operator_type` 与 `check`（`:29-107`）、`resolve_epilogue_class` 的互斥解析（`:110-128`）。`deep_gemm/include/deep_gemm/epilogue/operators.hpp:224-335`：`StochasticRoundToBF16`——quartet 加载与解交织（`:271-274`）、四个哈希常数与列号（`:277-287`）、NaN 规范化（`:288-296`）、高低半拆分与补码（`:297-308`）、混叠（`:309-320`）、16 位比较加一（`:322-328`）、交织写回（`:329-330`）。读完核对：内核的 `asc_gt(hash, ~low)` 与 host 的 `low > 0xffff - R` 是同一条件。

### 站点 7：dry_run 链路（三个文件串起来）

`deep_gemm/testing/par_compile.py`（`dry_run` 上下文 + 线程池）→ `csrc/jit_kernels/bf16_gemm.hpp:16-78`（`launch_bf16_gemm` 的 dry_run 分叉 `:67-70`、代码串内容 `:18-66`）→ `csrc/jit/jit.hpp:43-49`（开关存取）。再对照 `csrc/apis/gemm.hpp:105-106` 的 `is_meta()` 分支，理解元张量为何能走通指针判同。

### 站点 8：护栏与性能（浏览）

`deep_gemm/testing/disable_fallback.py`（`_would_fallback` 的 dispatch key 判断 `:17-26`、`_make_raiser` `:29-41`、注册循环 `:53-72`）；`deep_gemm/testing/bench.py` 的 `KernelProfile` 字段（`:27-71`）与 `bench_msprof` 签名（`:284-317`，默认参数与 kernel_names 语义）——为第 50 天留个钩子。

## 五、实践步骤

### A. 无 Ascend 环境：静态阅读与手工演算（主任务，约 50 分钟）

**A-1 calc_diff 手算（必做）。** (a) 证明闭式 `1 − 2⟨x,y⟩/(⟨x,x⟩+⟨y,y⟩) = ‖x−y‖²/(‖x‖²+‖y‖²)`。(b) 手算三个 diff：`x=[1.0], y=[1.0078125]`；`x=[3.0,4.0], y=[3.0,4.03125]`；`x=16384 个 1.0, y` 同 `x` 但第 100 个元素为 1.0078125。哪些超过 1e-5？这如何同时解释"容 1 ULP"与"拒系统性偏差"？(c) 思考：两个全零张量比较结果是什么（看 `utils.py:41-42`）？

**A-2 验证序列静态追踪（必做，当天产出的核心）。** 对 `python tests/test_bf16_gemm.py unaligned-k --skip-prof` 中 shape 为 `(1024, 2048, 17)`、双 K-major 的 desc（`generators.py:519-541`）：写出阶段 1 收集的 partial 数（提示：`accumulate=True` → 无 SR/fp32 变体；4 trans × {普通, alpha}（无 bias 时）+ 4 trans × {bias} = 12）与去重后的代码份数（3）；阶段 2 每个 trans_ab 的验证序列（FF/FT/TF 三次 dg 断言 + TT 拒绝 + 一次 generate + ACLNN 断言；该 desc 有没有 SR 验证——查 `common.py:215` 三条件）；最后对照 3.9 的流程图订正。

**A-3 ACLNN 跳过条件判定（必做）。** 对下列五个 Normal fp8 用例判断 `gemm_aclnn` 是否执行交叉验证，并注明命中的条件行号（答案见第七节末尾）：

1. `(m=128, n=128, k=128)`，recipe `(1,1,32)`，A/B 均 K-major；
2. `(m=128, n=128, k=17)`，recipe `(1,1,32)`；
3. `(m=4096, n=4096, k=4096)`，recipe `(32,32,32)`；
4. `(m=64, n=64, k=64)`，recipe `(1,1,32)`，A MN-major、B K-major；
5. fp8×fp4：A fp8 MN-major、B fp4 K-major，k=4096，输出 bf16。

**A-4 SR 机制演算（必做）。** (a) 概念题：给定某 fp32 值的低 16 位 `low=0x8000`（正中间点），R 分别为 `0x7fff`、`0x8000`、`0xffff` 时是否向上舍入？向上舍入的概率是多少？与就近舍入（round-to-nearest-even）在 `low=0x8000` 时的行为有何不同？(b) 用下面的纯 numpy 脚本（不依赖 torch/NPU，可在任何机器运行）复刻 `utils.py:70-92`，并对 `(1.0, 2.0, 3.0, 4.0, 1.00390625)` 解释输出——前四个值为何必然不变，最后一个为何在这里向上舍入（它的 low16 恰为 0x8000，RN-even 会舍到 1.0，SR 以约 1/2 概率舍到 1.0078125）：

   ```bash
   python3 - <<'EOF'
   import numpy as np, struct

   def expected_sr_bf16(row):          # 复刻 tests/utils.py:70-92（单行版本）
       values = np.asarray(row, dtype=np.uint32).copy()
       n = values.shape[-1]
       values = np.pad(values, [(0, -n % 4)]).reshape(-1, 4)
       h = (values[:, 0] * np.uint32(0x5671d42b) + values[:, 1] * np.uint32(0x9995e499) +
            values[:, 2] * np.uint32(0xace1b8a5) + values[:, 3] * np.uint32(0xe153538d) +
            np.arange(values.shape[0], dtype=np.uint32))
       h = (h ^ (h >> 23)) * np.uint32(0x7feb352d); h ^= h >> 16
       h1, h2 = h * np.uint32(0x846ca68b), h * np.uint32(0xd35a2d97)
       h1 ^= h1 >> 11; h2 ^= h2 >> 11
       R = np.stack((h1 & 0xffff, h1 >> 16, h2 & 0xffff, h2 >> 16), -1).reshape(-1)
       flat = values.reshape(-1)
       out = (flat >> 16) + ((flat & 0xffff) > (0xffff - R))
       out = np.where((flat & 0x7fffffff) > 0x7f800000, 0x7fc0, out).astype(np.uint16)
       return out[:n], R[:n]

   bits = [struct.unpack('<I', struct.pack('<f', v))[0]
           for v in (1.0, 2.0, 3.0, 4.0, 1.00390625)]
   out, R = expected_sr_bf16(bits)
   for v, b, r in zip((1.0, 2.0, 3.0, 4.0, 1.00390625), out, R):
       print('fp32 %-12r bits=0x%08x  R=0x%04x  -> bf16 0x%04x' % (v, struct.unpack("<I", struct.pack("<f", v))[0], r, b))
   EOF
   ```

   (c) 思考：为什么脚本要 pad 到 4 的倍数、最后又只取前 n 个（提示：`utils.py:74` 的注释与内核按 quartet 处理越界列的方式）。

**A-5 meta 生成体验（必做）。** 用第 29 天 A-4 同款的 monkeypatch（绕开 `torch.npu.manual_seed`），在无 NPU 的机器上生成一个元张量六元组，验证"编译只需要元信息"：

   ```bash
   cd DeepGEMM-Ascend/tests
   python3 - <<'EOF'
   import sys; sys.path.insert(0, '.')
   import torch, generators
   if not hasattr(torch, 'float4_e2m1fn_x2'):
       torch.float4_e2m1fn_x2 = object()   # 唯一哨兵占位（不能等于任何真实 dtype，
   # 否则 fp8 会被误判成 fp4 而走 storage 减半分支，generators.py:179-181）
   generators.reset_seed = lambda *a, **k: None      # 绕开 torch.npu 依赖
   from generators import GemmTestDesc, GemmType, Major
   t = GemmTestDesc(GemmType.Normal, torch.bfloat16, torch.bfloat16, torch.bfloat16,
                    (64, 32, 48), Major.K, Major.K, accumulate=False)
   shape, a, b, c, d_ref, _ = t.generate(device='meta')
   print(shape, a.shape, a.stride(), a.device)   # 期望：(64,32,48) [64,48] (48,1) meta
   # fp8 操作数在 meta 下是 (data, sf) 二元组（generators.py:185-186）
   t2 = GemmTestDesc(GemmType.Normal, torch.float8_e4m3fn, torch.float8_e4m3fn,
                     torch.bfloat16, (64, 32, 48), Major.K, Major.K, recipe=(1, 1, 32))
   _, a2, b2, _, _, _ = t2.generate(device='meta')
   print(type(a2), a2[0].shape, a2[0].stride(), a2[1].shape)
   # 期望：tuple，data [64,48] stride (48,1)，sf [64,2]（ceil(48/32)=2）
   EOF
   ```

   说明：真实 `par_compile` 还需要 `deep_gemm._C` 与 NPU stream（`par_compile.py:18-22`），无硬件时到此为止——重点是确认元张量的 shape/stride 与第 29 天真实生成完全一致。

**A-6 判定题自验（必做）。** 判断真假并给出行号依据（答案见第七节末尾）：

1. 四个 GEMM CLI 脚本（bf16/fp8/fp4/fp8_dequant）传给 `run_gemm_test` 的 threshold 都是 1e-5，因此 `assert_equal` 的逐位相等模式在这些脚本中不会被走到。（  ）
2. ACLNN 交叉验证比较的是"dg 内核输出"与"ACLNN 输出"。（  ）
3. `gemm_dg` 中 `with_bias=True` 会覆盖 `accumulate=True` 的语义，此时 `c` 的形状是 `(1, n)`。（  ）
4. 随机舍入验证只对 bf16 输入的用例触发。（  ）
5. 阶段 1 中同一个 desc 的 4 种 trans_ab 会生成 4 份不同的 JIT 代码。（  ）
6. `disable_cpu_fallback` 注册的内核会让"没有 NPU 实现的算子"抛 RuntimeError，而不是让它悄悄在 CPU 上算。（  ）

### B. 有 Ascend 环境：运行与观察（可选，约 30 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
python tests/test_bf16_gemm.py normal --skip-prof
python tests/test_bf16_gemm.py unaligned-k --skip-prof
```

1. 观察开头的 `Compiling` 进度条（`par_compile` 的 tqdm）与 `Num tests:`（第 29 天 A-4：54 / 20）；
2. 观察 ` > Perf (m=…, layout=…, acc=…, …): OK` 行：把 20 行 unaligned-k 输出与 5 个 shape × 4 major 对上；acc 恒为 1（`accumulate=True`）——解释为什么这批用例没有随机舍入验证；
3. 去掉 `--skip-prof` 重跑 normal，把性能行的 mad%/mte2% 记下来（第 50 天的素材）；
4. 故意制造一次失败（可选）：临时把某次调用的 `threshold` 改成 `1e-30` 重跑，观察 `AssertionError: dg: … (rows …, diff=…)` 的定位信息格式，再还原。

## 六、当天产出

**一条完整测试流程图**。把 A-1～A-5 的结果整理成一页笔记，至少包含：

1. **主流程图**：以 3.9 为模板默写（两阶段、五类验证、三道护栏的位置与行号），写完对照订正；
2. **比较方法对照表**：断言 ①～④ 各自的比较对象、比较方式（余弦阈值 / 逐位 / 逐位复刻）、触发条件、容错理由；
3. **ACLNN 跳过条件表**：A-3 的判定结果与依据行号；
4. **SR 演算记录**：A-4 的三个 R 值判定、概率结论、numpy 脚本输出与解释；
5. **验证序列卡片**：A-2 的静态追踪结果（partial 数、代码份数、每 trans 的断言序列）；
6. **遗留问题清单**（给第 31/33/50 天）：至少记两条，例如"测试失败时如何区分环境/参考/布局/内核四类问题（明天主题）""ACLNN 的 `[mn, K/64, 2]` scale 布局与内核内部布局如何互转（第 33 天 `transform_sf_into_required_layout`）""mad%/mte2% 多少算好、瓶颈怎么判断（第 50 天）"。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. 阶段 2 中一个 Normal desc（非累加、bf16 输出、同 dtype）的每个 trans_ab 依次发生哪些事？`(T, T)` 组合为什么不运行内核？
2. `get_gemm_desc` 如何区分"累加"和"bias"？meta 张量下这个判断有什么特殊处理？为什么 alpha 与 bias 必须互斥？
3. `calc_diff` 的值域与含义是什么？为什么 BF16 测试用 1e-5 阈值而不是逐位相等？`assert_equal` 的 `threshold==0` 模式在什么前提下才成立？
4. `gemm_aclnn` 有哪些跳过条件？返回值 False 与 None 分别代表什么，`has_ref is not False` 为什么这样写？
5. 随机舍入验证为什么要跑两次内核？host 侧复刻的哈希以什么为输入、以什么为输出？为什么最后必须逐位比较而不能用阈值？
6. `assert_direct_output_matches_fp32_accumulation` 只对哪类用例触发？它和断言 ① 的 NaN 预填充分别在保护什么？
7. meta 并行编译里，"meta 张量""dry_run""线程池"各自解决什么问题？为什么 12 个 partial 可能只对应 3 份 JIT 代码？

### 参考答案

1. FF/FT/TF 三次：generate → `d` 预填充 randn → `gemm_dg` → sync → `assert_equal(d, d_ref, threshold)`；然后一次独立 generate + `d=randn` → ACLNN 交叉验证 → 随机舍入（SR 输出 + fp32 直出 + host 复刻比较）。`(T, T)` 是 alpha+bias 互斥的拒绝路径：期望 `RuntimeError('alpha and bias cannot be used together')`（`common.py:191-197` ↔ `gemm.hpp:113-116`），不产生输出也就无从比较。
2. 指针判同：`c->data_ptr() == d.data_ptr()` 即累加（原地），否则 Normal 下视为 `[1, N]` bias（`gemm.hpp:105-110`）；meta/空张量没有数据地址，改用 `c->is_alias_of(d)`（`:105-106`）。互斥因为两者都在 epilogue 阶段作用于同一份输出（alpha 缩放 vs 加 bias），`resolve_epilogue_class` 本就要求 epilogue 类/alpha/sfd 至多其一（`epilogue_class.hpp:117-118`），bias 再叠加 alpha 会产生未定义的组合语义。
3. `diff = ‖x−y‖²/(‖x‖²+‖y‖²) ∈ [0, 2]`，尺度不变的相对平方欧氏距离。BF16 下 cube 内核与 torch 参考的 K 维累加顺序不同，差可达 ~1 ULP（`utils.py:50-51` 注释），位级比较必然失败；1e-5 容纳零星 ULP 差（16384 个元素 1 个 ULP ≈ 1.9e-9）而拒绝系统性偏差。`threshold==0` 要求参考与内核走相同数值路径（如 FP8 MX 的码字精确解码 + 2 的幂 SF，cast 路径一致）。
4. 五条：grouped 无 aclnn 对应（`:90-91`）；`k % 64 != 0`（`:98-99`）；`gran_mn != 1` 被 aclnnQuantMatmulV5 拒绝（`:104-105`）；MN-major 且 `ceil(k/32) <= 2` 的退化 scale（`:111-112`）；fp8×fp4 仅 fp8(K)×fp4(K)+bf16/fp16 输出（`:113-122`）。False 表示跳过（不比较），None 表示已运行（函数无 `return True`）；`is not False` 让 None 通过、False 拦下（`:212`）。
5. 第一次带 `BF16StochasticRounding()` epilogue 得到 bf16 输出；第二次用 fp32 输出 buffer（k-grouped 用累加变体）拿到舍入前的精确 fp32 值。哈希输入是 4 个连续 fp32 的位模式加 quartet 列号（4 个常数乘加 + 混叠），输出是每个成员 16 位随机数 R，据此决定"高 16 位 + 是否进一"。逐位比较是因为 SR 的正确性定义在"每一位的舍入决策"上——哈希常数、列号、NaN 规范化、越界清零任何一处错都只体现为个别位翻转，阈值比较发现不了。
6. 仅 k-grouped 非累加（`common.py:202-206`）。它保护直写路径的 cast 语义（内核内部 fp32→cd_dtype 与 host `.to()` 逐位一致）；NaN 预填充（`:190`）保护"写完整性"——空组的 `aiv_zero_empty_groups` 写零或任何漏写都会让 NaN 残留并被断言 ① 抓住。
7. meta 张量让编译不占 NPU 存储（host 路径只读元信息）；dry_run 让 C++ 走完整 host 路径但停在 `compile_without_load`（编译入缓存、不加载不启动，`bf16_gemm.hpp:67-70`）；线程池并发提交编译、共享进程内 JIT 缓存（`par_compile.py:16-36`）。4 种 trans_ab 经抵消转置后 desc 相同、alpha 数值不进代码串（只烙 epilogue 类型名），12 个 partial = 4 trans × {Identity, ScaleByAlpha, AddBias} → 3 份代码，重复由缓存吸收。

**A-3 判定答案**：1. 执行（无命中）；2. 跳过（k=17 非 64 倍数，`:98-99`）；3. 跳过（recipe `(32,32,32)` 的 gran_mn=32≠1，`:104-105`）；4. 跳过（k=64 → `ceil(64/32)=2` 且 A MN-major，`:111-112`）；5. 跳过（fp8 必须 K-major 作 x1，`:116-122` ↔ `aclnn.hpp:96-103` 支持矩阵）。

**A-6 判定答案**：1. **真**（四个脚本均 `threshold=1e-5`，如 `test_bf16_gemm.py:17`；`assert_equal` 默认 0 只在其他调用方生效）。2. **假**（两者分别与 `d_ref` 比较，`common.py:201/:213`——ACLNN 是第二参考实现，不是 dg 的对照组）。3. **真**（`common.py:53-58` 的 if/elif 顺序；`generators.py:304-308` 按 with_bias 造 `(1, n)`）。4. **假**（条件是 `a_dtype == b_dtype`，fp8×fp8、fp4×fp4 同样触发，`common.py:215`）。5. **假**（两次抵消转置后 desc 相同 → 同一代码串，第 29 天 3.3；重复被 JIT 缓存去重）。6. **真**（`disable_fallback.py:29-41` 的 `_make_raiser`）。

---

**明日预告（第 31 天）**：运行并分析 BF16 测试。今天画好了流程图，明天让它真的跑起来——`normal` 与 `unaligned-k` 的输出如何与源码中的边界处理对应、`Compiling` 进度条与 `Num tests` 如何对上枚举器，以及测试失败时如何按"环境 / 参考结果 / 布局 / 内核"四个层级定位。没有环境时则选取一个测试用例做完整的静态推演。
