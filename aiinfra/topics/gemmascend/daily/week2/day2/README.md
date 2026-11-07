# 第 2 周 · 第 2 天（总第 9 天）：构建并运行最小测试

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 2 周"项目结构与 JIT 调用链"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 55 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

昨天我们建立了仓库地图，但整条链路还没有真正"转"过一次。今天的任务只有一个：**把最小测试跑起来**——有 Ascend 环境就实际运行并记录结果；没有环境，就沿同一条命令做静态追踪，弄清它会执行哪些函数。这一天是本周所有深入阅读的"地基"：从第 10 天起逐层拆调用链时，你会不断回想起今天看到的测试输出里那些 `fwd/dgrad/wgrad`、`layout=NT` 到底是什么。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出两条命令 `./develop.sh` 和 `python tests/test_bf16_gemm.py normal --skip-prof` 各自做什么、为什么顺序不能颠倒；
2. 逐项解释 `tests/test_bf16_gemm.py` 的命令行参数：位置参数 `select` 的 7 个取值与开关 `--skip-prof` 的确切语义；
3. 画出 `run_gemm_test` 的三个阶段（并行预编译 → 正确性验证 → 性能测量）以及各阶段调用的关键函数；
4. 说明 `normal` 用例集的构成——fwd/dgrad/wgrad 三类、共 54 个用例——以及参考结果 `d_ref` 是如何产生的；
5. 解释正确性判据：FP32 参考 + 余弦相似度阈值 1e-5、ACLNN 交叉验证、`alpha + bias` 组合的拒绝测试、BF16 随机舍入检验；
6. 说出 `--skip-prof` 具体跳过了什么（`bench_msprof` 与 TFLOPS/带宽/流水线占用输出）；
7. 留下一份记录：环境信息 + 运行结果，或者（无环境时）环境阻塞项 + 静态推演笔记。

**当天产出：环境信息、运行结果或环境阻塞项记录**（模板见第六节）。

## 二、预备知识

- **第 8 天的代码地图**：五类代码（Python API / C++ Host / JIT 启动 / Device Kernel / 测试）的位置，以及 `bf16_gemm_nn` 的调用链草图——今天会在真实运行中"踩"一遍这条链。
- **第 1 周概念**：M/N/K 与 `D = A × B`（第 1 天）、NT/NN 布局与 major（第 2 天）、BF16 计算与 FP32 累加（第 4 天）。测试输出里的 `layout=NT`、`acc=1`、`bf16/fp32` 都直接用到这些概念。
- **Python 基础**：`argparse`（命令行解析）、`functools.partial`（延迟绑定参数的回调）、`dataclass`（`GemmTestDesc`）、`concurrent.futures.ThreadPoolExecutor`（并行编译用到的线程池）。不需要写过，能读懂即可。
- **Ascend 环境（可选）**：命令实操需要 NPU + CANN + `torch_npu`（README Requirements：Ascend 950 系列、CANN 9.20、Python ≥ 3.10）。没有硬件时，本教程第五节 B 部分提供完整的静态替代任务。

## 三、详细讲解

### 3.1 两条命令的分工

学习计划第 9 天给出的是一组动作：先构建，再运行：

```bash
./develop.sh                                        # ① 构建并链接 C++ 扩展
python tests/test_bf16_gemm.py normal --skip-prof   # ② 跑 BF16 GEMM 的 normal 用例集，只测正确性
```

- **命令 ① 是"把 Host 侧准备好"**：把 `csrc/python_api.cpp` 编译成 `deep_gemm._C` 扩展并链接进包目录。Device kernel 此时**不会**被编译——它们是模板头文件，要等命令 ② 运行时由 JIT 按需编译（第 8 天已确认这一架构）。
- **命令 ② 是"让全链路转起来"**：从 Python 测试出发，穿过 `_C` 扩展、`get_gemm_desc`、`launch_bf16_gemm`、JIT 编译与启动，最终在 NPU 上执行 `bf16_gemm_impl`，再回到 Python 校验结果。今天不拆解中间每一层（第 10～12 天的任务），但要能说出"经过了哪几站"。

顺序不能颠倒：没有 ①，`import deep_gemm` 时 `from . import _C`（`deep_gemm/__init__.py` 第 25 行）就会失败，命令 ② 第 3 行 `from common import run_gemm_test` 便无法执行（`tests/common.py` 第 3 行 `import deep_gemm`）。

### 3.2 `develop.sh`：开发构建的四个动作

脚本总共 27 行，逐段看：

| 行号 | 动作 | 说明 |
| :--- | :--- | :--- |
| 第 5～7 行 | `cd` 到脚本所在目录（仓库根） | 保证 `setup.py`、`deep_gemm/` 的相对路径正确，结束后再 `cd` 回原目录（第 27 行） |
| 第 10 行 | `rm -rf build dist` | 清掉旧构建产物，保证本次构建是干净的 |
| 第 13 行 | `DG_USE_LOCAL_VERSION=0 python3 setup.py build` | 真正的构建：编译 `deep_gemm._C` |
| 第 16～20 行 | 在 `build/` 里找到 `.so`，`ln -sf` 链接进 `deep_gemm/`；同时链接 `../stubs/_C.pyi` | 让包目录里的 `from . import _C` 直接可用；成功时打印 `Linked: ... -> deep_gemm/`，找不到 `.so` 则报错退出（第 21～24 行的 else 分支） |

两个值得注意的细节：

- **`DG_USE_LOCAL_VERSION=0`**：`setup.py` 第 60～79 行的 `get_package_version()` 在 `DG_USE_LOCAL_VERSION=1`（默认）时会检查 git 工作区是否干净、并在版本号后追加 `+<git 短哈希>`；`develop.sh` 显式关掉它，所以**本地有未提交修改也能构建**，版本号就是 `deep_gemm/__init__.py` 第 1 行的 `__version__`（当前 `0.1.1`）。
- **被编译的源文件只有 `csrc/python_api.cpp`**（`setup.py` 第 43 行 `sources = ['csrc/python_api.cpp']`），`csrc/apis/` 等头文件都被它 `#include`，属于同一翻译单元（第 8 天站点 2 已确认）。

正式安装路线是 `pip install . --no-build-isolation` 或 `./build.sh`（README 第 43～47 行），区别在于 develop.sh 只做本地构建 + 符号链接，不生成 wheel。

### 3.3 测试脚本的命令行：`select` 与 `--skip-prof`

`tests/test_bf16_gemm.py` 只有 50 行，命令行解析集中在 `__main__`（第 21～30 行）：

```python
parser = ArgumentParser()
parser.add_argument('select', nargs='?', default='all',
    choices=['all', 'unaligned-k', 'unaligned-mn', 'strided', 'normal', 'mgrouped', 'kgrouped'])
parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true',
    help='Skip performance measurements and just test correctness')
```

**位置参数 `select`**（可省略，默认 `all`）选择用例集，第 31～50 行的 if/elif 把它映射到 `test_bf16_gemm(...)` 的不同实参：

| 取值 | 映射（`tests/test_bf16_gemm.py` 行号） | 用例来源（`tests/generators.py`） | 覆盖场景 |
| :--- | :--- | :--- | :--- |
| `normal` | `GemmType.Normal`（第 37～38 行） | `_enumerate_normal`（第 544 行） | 对齐、连续、无分组的常规 Dense GEMM（今天的主角） |
| `unaligned-k` | `Normal, unaligned_k=True`（第 31～32 行） | `_enumerate_normal_unaligned_k`（第 519 行） | K 不是块对齐倍数，如 k=17、234、4095 |
| `unaligned-mn` | `Normal, unaligned_mn=True`（第 33～34 行） | `_enumerate_normal_unaligned_mn`（第 491 行） | M/N 非对齐，如 17、15、130×257 |
| `strided` | `Normal, strided=True`（第 35～36 行） | `_enumerate_normal_strided`（第 466 行），tag 标记 `strided` | 带标记的 strided 场景（配套工具 `make_strided`，第 238 行） |
| `mgrouped` | `MGroupedContiguousWithPsumLayout`（第 39～40 行） | `_enumerate_m_grouped_contiguous`（第 579 行） | M 维分组（MoE expert 场景） |
| `kgrouped` | `KGroupedContiguousWithPsumLayout`（第 41～42 行） | `_enumerate_k_grouped_contiguous`（第 616 行） | K 维分组 |
| `all`（默认） | 依次调用上述 6 组（第 43～50 行） | 全部 | 全量回归 |

`GemmType` 与 `Major` 枚举定义在 `tests/generators.py` 第 15～23 行，与 C++ 侧 `deep_gemm/include/deep_gemm/common.hpp` 的同名枚举一一对应（第 8 天站点 7）。

**开关 `--skip-prof`**（别名 `--no-prof`，`action='store_true'`，默认 False）：help 文字写得很直白——"跳过性能测量，只测正确性"。它一路透传到 `run_gemm_test(..., skip_prof=skip_prof)`，具体跳过了什么见 3.7 节。

入口函数 `test_bf16_gemm`（第 6～18 行）把命令行选择翻译成一次完整测试：

```python
def test_bf16_gemm(gemm_type=GemmType.Normal, unaligned_mn=False, unaligned_k=False,
                   strided=False, skip_prof=False):
    return run_gemm_test(
        torch.bfloat16, torch.bfloat16,      # A、B 都是 bf16
        gemm_type, unaligned_mn, unaligned_k, strided,
        skip_prof=skip_prof,
        threshold=1e-5,                       # 正确性阈值（余弦相似度口径，见 3.6 节）
    )
```

### 3.4 `run_gemm_test` 的三个阶段

`tests/common.py` 第 131～237 行的 `run_gemm_test` 是所有 GEMM 测试（BF16/FP8/FP4）共用的总驱动。`python tests/test_bf16_gemm.py normal --skip-prof` 执行时，它的工作分为三个阶段：

```text
run_gemm_test(bf16, bf16, Normal, ..., skip_prof=True, threshold=1e-5)
│
├─ 阶段 0：disable_cpu_fallback()                ← common.py:142
│    给所有"没有 NPU 实现、会回退 CPU"的算子注册抛异常的内核，
│    保证测试真的在 NPU 上跑，而不是悄悄落到 CPU
│
├─ 阶段 1：并行预编译（common.py:152～178）
│    enumerate_gemm_tests() → 54 个 GemmTestDesc
│    对每个用例 × {无 bias, 有 bias} × {nt, tn, nn, tt} × {普通, alpha, ...}
│      用 device='meta' 生成"只有形状没有存储"的张量
│      包装成 partial(gemm_dg, ...)
│    par_compile(kernels)：线程池 + dry_run 并行触发 JIT 编译（只编译不加载）
│
├─ 阶段 2：正确性验证（common.py:180～221）
│    对每个用例重新生成真实数据（reset_seed 后在 NPU 上）
│    对每个 trans_ab × {无选项, bias, alpha}：
│      gemm_dg(...) 调 deep_gemm.bf16_gemm_{nt,tn,nn,tt}
│      torch.npu.synchronize() → assert_equal(d, d_ref, threshold=1e-5)
│    alpha+bias 组合 → 断言抛出 RuntimeError（拒绝测试）
│    gemm_aclnn(...) → ACLNN 参考实现交叉验证
│    BF16 输出用例 → 随机舍入（BF16StochasticRounding）一致性检验
│
└─ 阶段 3：性能测量（common.py:225～236）—— skip_prof=True 时整段跳过
     bench_msprof(lambda: gemm_dg(...), kernel_names='gemm_impl')
     print_perf(...)  ← prof=None 时只打印 "OK"
```

三个阶段的设计动机值得体会：

- **为什么先集中编译？** JIT 编译是整个流程中最慢的部分（调用 CANN 的 `bisheng` 编译器），且相互独立、可以并行。阶段 1 用 `par_compile`（`deep_gemm/testing/par_compile.py` 第 16 行）开线程池并发触发，`dry_run()` 上下文（第 7 行）调用 `_C.set_dry_run(True)`，使 C++ 侧走 `jit->compile_without_load`（`csrc/jit_kernels/bf16_gemm.hpp` 第 67～70 行）——只编译不加载。线程数由环境变量 `DG_TEST_MAX_WORKERS` 控制，默认取 CPU 核数（第 27 行）。编译产物缓存在 `$HOME/.dj`（`DG_JIT_CACHE_DIR` 默认值，README 第 91 行），所以**第二次运行会明显变快**。
- **为什么用 meta 张量？** `compile_test.generate(device='meta')`（`common.py` 第 157 行）只产生形状/dtype/stride 元数据、不占 NPU 存储；C++ 侧 `get_gemm_desc` 专门处理了 meta 张量（`csrc/apis/gemm.hpp` 第 105～106 行注释："Meta and empty tensors have no data address; use storage identity for them"）。这正是学习计划第 30 天"meta tensor 并行编译"的伏笔。
- **为什么禁用 CPU 回退？** `disable_cpu_fallback()`（`deep_gemm/testing/disable_fallback.py` 第 44 行）遍历所有 dispatch 算子，凡是只注册了 CPU 实现的，都挂上一个抛 `RuntimeError` 的 PrivateUse1 内核（第 29～39 行的 `_make_raiser`）。这样一旦某个操作意外落到 CPU，测试会立刻报 `[npu_no_fallback] operator ... has no NPU kernel`，而不是慢吞吞地用 CPU 算出"看起来对"的结果。

### 3.5 用例从哪来：`normal` 集合的 54 个用例

`_enumerate_normal`（`tests/generators.py` 第 544～576 行）按"前向 / 反向"组织用例，shape 来自 DeepSeek 模型的真实负载（与 README Performance 表中的 shape 一致）：

```python
fp32_output_nk = [(256, 7168), (129280, 7168)]                  # 2 组：FP32 输出
bf16_output_nk = [(2112, 7168), (576, 7168), (24576, 1536),
                  (32768, 512), (7168, 16384), (4096, 7168),
                  (7168, 2048)]                                  # 7 组：BF16 输出
m_fwd_list, m_bwd_list = [1, 128, 4096], [4096]
```

| 类别 | 生成逻辑（行号） | 数量 | A/B major → `layout` 标签 | 输出 |
| :--- | :--- | :--- | :--- | :--- |
| `fwd` | `for m in [1,128,4096]: for i in range(9)`（第 566～570 行） | 3×9 = 27 | K,K → `NT` | i<7 为 bf16（21 个），i≥7 为 fp32（6 个） |
| `dgrad` | `for m in [4096]: for (n,k) in 9 组`（第 572～574 行） | 9 | K,MN → `NN` | bf16 |
| `wgrad`（累加） | 同上（第 575 行） | 9 | MN,MN → `TN` | fp32，`accumulate=True` |
| `wgrad`（不累加） | 同上（第 576 行） | 9 | MN,MN → `TN` | bf16 |

合计 **27 + 27 = 54** 个 `GemmTestDesc`——这就是运行后 `Num tests:  54` 的来源。`layout` 标签由 `major_opt`（第 750 行）从 major 对拼出：`('N' if a==K else 'T') + ('T' if b==K else 'N')`，即第 1 周学过的 NT/NN/TN 记法。fwd/dgrad/wgrad 恰好覆盖三种布局组合，这个对应关系建议记进笔记。

每个 `GemmTestDesc`（第 250～297 行）是一个 dataclass，字段包括 `gemm_type`、两个 dtype、`shape_mnk`、两个 major、`accumulate`、`recipe`、`alpha`、`with_bias`、`tag` 等；`generate()`（第 429 行）按 gemm_type 分发，Normal 走 `_generate_normal`（第 300 行）。

### 3.6 正确性如何判定

**参考结果 `d_ref` 怎么来？** `_generate_normal`（`tests/generators.py` 第 310～322 行）先用 `generate_operand_with_major` 生成指定 major 的 A、B（bf16 分支：`torch.randn(...)` 转 bf16，再 `transpose→contiguous→transpose` 保证目标维连续），然后用**FP32 矩阵乘**算参考值：

```python
d = (alpha_value * (a_repr.float() @ b_repr.float().T)
     + (c if c is not None else 0)).to(self.cd_dtype)
```

即"参考 = 先升 FP32 精确计算，再舍入到目标 dtype"——这正是第 4 天"低精度计算需要高精度累加"的落地。

**怎么比？** `assert_equal`（`tests/utils.py` 第 46 行）有两种口径：

- `threshold == 0`：`torch.equal` 逐位相等（FP8 MX 用，因为 cube + cast 路径可复现）；
- `threshold > 0`：`calc_diff`（第 36 行）基于余弦相似度，`diff = 1 - 2⟨x,y⟩/(‖x‖²+‖y‖²)`，要求 `diff < threshold`。第 50～51 行注释解释了原因：**BF16 下 kernel 与参考的 K 维累加次序不同，最多差约 1 ULP，不指望逐位相等**。`test_bf16_gemm` 传入的 `threshold=1e-5`（`tests/test_bf16_gemm.py` 第 17 行）。

**kernel 怎么被调到？** `gemm_dg`（`tests/common.py` 第 49～84 行）是测试与被测接口之间的适配层：Normal 分支（第 59～66 行）根据 `trans_ab` 决定是否逻辑转置 A/B，然后 `getattr(deep_gemm, f'{gemm_name}_gemm_{trans_ab}')(a, b, d, **extra_args)` 动态选中 `bf16_gemm_nt/tn/nn/tt`；`extra_args` 里按需装 `c`（bias 或累加）和 `alpha`。注意内部规范形是 A[M,K]、B[N,K]：`trans_ab='nt'` 时 B 原样传入，`'nn'` 时才逻辑转置成 [K,N]（第 64～65 行）。

**三重交叉验证**：

1. **主验证**：`gemm_dg` 的输出 vs FP32 参考 `d_ref`（第 199～201 行，先 `torch.npu.synchronize()` 再断言）；
2. **ACLNN 交叉验证**：`gemm_aclnn`（第 87～128 行）调用 `deep_gemm.aclnn_bf16_gemm_{nt,tn,nn,tt}`（CANN 官方算子库的参考实现，`csrc/apis/aclnn.hpp`），同样与 `d_ref` 比（第 210～213 行）；只覆盖 Normal 类型（第 90～91 行注释：m-/k-grouped 没有 aclnn 对应物）；
3. **参数语义测试**：`alpha` 与 bias 同时出现时，断言接口抛出包含 `'alpha and bias cannot be used together'` 的 `RuntimeError`（第 191～198 行）——这个检查点在 C++ 侧 `csrc/apis/gemm.hpp` 第 116 行。

**随机舍入检验**（第 215～221 行）：对 bf16 输出、无累加的用例，用 `deep_gemm.epilogue.BF16StochasticRounding()` 再算一次，并用 `assert_stochastic_bf16_matches_fp32`（`tests/utils.py` 第 70 行）以同一种哈希在宿主侧复现期望的舍入位，逐位比对。今天只需知道"有这么一类特殊比较"，原理留到第 30 天。

### 3.7 `--skip-prof` 跳过了什么

看 `run_gemm_test` 末尾（`tests/common.py` 第 225～236 行）：

```python
prof = None
if not skip_prof and test.profile and k:
    trans_ab = test.get_available_trans_ab()[0]
    prof = bench_msprof(lambda: gemm_dg(test, a, b, c, d, g_layout=g_layout, trans_ab=trans_ab),
                        kernel_names='gemm_impl')
print_perf(m, n, k, f'layout={layout}, {out_opt}, acc={int(test.accumulate)}, {test.tag}',
           2.0 * m * n * k, count_bytes(a, b, c, d), prof)
```

`--skip-prof` 让 `prof` 保持 `None`，于是 `print_perf`（`tests/utils.py` 第 106 行）只打印一行 `... : OK`。去掉它，则会：

- 用 `bench_msprof`（`deep_gemm/testing/bench.py` 第 284 行）测量：默认 10 次 warmup + 30 次迭代、每次迭代前清 L2（`flush_l2=True`，第 289 行）、`backend='persistent'` 常驻 PMU 会话（第 294 行）；`kernel_names='gemm_impl'` 按子串匹配唯一的 GEMM kernel；
- 产出 `KernelProfile`（第 28 行）：除耗时（`dur_us`）外还有各流水线占用率——AIC 侧 `mad`（矩阵乘加）、`mte1`（L1→L0）、`mte2`（GM→L1）、`fixpipe`（L0C→输出），AIV 侧 `vec`、`mte2/mte3`；
- `print_perf` 换成完整行：`耗时 us | TFLOPS | GB/s | mad=xx% | mte2=xx%`。TFLOPS 用 `2.0*m*n*k / dur` 算，字节数用 `count_bytes` 数——正是第 1 天学过的计算量与数据量公式。

今天的正确性验证不需要这些，所以 `--skip-prof` 是合理的"最小测试"形态；性能观察留到第 31、51 天。

## 四、源码导读

今天的导读目标是**把第三节的内容落到真实代码行上**。每站 5～10 分钟，重点核对"函数名 + 行号 + 它在测试流程中的位置"。

### 站点 1：`tests/test_bf16_gemm.py`（50 行，今天的主入口）

- 第 6～18 行 `test_bf16_gemm`：参数即命令行选择；`threshold=1e-5` 是 BF16 的余弦相似度阈值；
- 第 25 行 `select`：`nargs='?'` 说明可省略、`default='all'`、7 个 choices——**这就是 3.3 节表格的出处**；
- 第 26 行 `--skip-prof`/`--no-prof`：`dest='skip_prof'`，两个写法是同一个开关；
- 第 37～38 行：`normal` → `test_bf16_gemm(GemmType.Normal, skip_prof=skip_prof)`——一句话版本的全部映射逻辑；
- 第 43～50 行：`all` 分支依次跑 6 组调用，顺序为 normal → unaligned-mn → unaligned-k → strided → mgrouped → kgrouped。

### 站点 2：`tests/common.py`（237 行，总驱动）

按 3.4 节的三阶段顺序精读三段：

1. **第 152～178 行（编译阶段）**：注意第 154 行 `for with_bias in (False, True)` 只对 Normal 展开（grouped 只测无 bias）；第 157 行 `device='meta'`；第 159～174 行向 `kernels` 列表追加 partial——数一数每类用例各追加几个（第五节 B 的手工演算题）；
2. **第 180～221 行（正确性阶段）**：第 186 行 `product((False, True), repeat=2)` 产生 `(use_alpha, with_bias)` 四种组合；第 189 行重新 `generate()` 出真实数据（内部先 `reset_seed()`，`generators.py` 第 9～12 行，种子固定为 0，保证可复现）；第 199～201 行调用 + 同步 + 断言；第 191～198 行 alpha+bias 拒绝测试；第 208～213 行 ACLNN；第 215～221 行随机舍入；
3. **第 225～236 行（性能阶段）**：3.7 节已逐行摘录。

再看两个适配函数：`gemm_dg`（第 49 行）如何拼 `extra_args` 并 `getattr` 到四个接口名；`gemm_aclnn`（第 87 行）为什么对 grouped 直接 `return False`。

### 站点 3：`tests/generators.py`（825 行，只读今天需要的部分）

- 第 15～23 行：`Major`、`GemmType` 枚举（与 C++ `common.hpp` 镜像）；
- 第 250～297 行：`GemmTestDesc` 全部字段 + `get_available_trans_ab`（Normal 返回 `['nt','tn','nn','tt']`）；
- 第 300～324 行：`_generate_normal`——重点看第 310～315 行 A、B 的生成（形状分别是 `(m,k)`、`(n,k)`！）和第 320～322 行的 FP32 参考计算；
- 第 544～576 行：`_enumerate_normal`——对照 3.5 节表格逐行验证数量与 major 组合；
- 第 652～679 行：`enumerate_gemm_tests` 的分发逻辑（normal → `_enumerate_normal`）。

FP8/FP4 相关的 `quantize_with_major`、`generate_random_quantized_with_major` 今天**跳过**（第 29、32 天再读）。

### 站点 4：`tests/utils.py`（115 行）

- 第 36～43 行 `calc_diff`：余弦相似度公式，注释说明"与 DeepGEMM 口径一致"；
- 第 46～58 行 `assert_equal`：`threshold` 两种分支；`slices` 参数为 grouped 场景按组比较预留（Normal 传 `None`，整张比较）；
- 第 106～115 行 `print_perf`：对照 3.7 节，`prof is None` 时只打 `'OK'`——**你运行 `--skip-prof` 时看到的每一行 OK 都出自这里**。

### 站点 5：`deep_gemm/testing/par_compile.py`（36 行）与 `disable_fallback.py`（73 行）

- `par_compile`（第 16 行）：`ThreadPoolExecutor` 并发执行所有 partial；`dry_run` 上下文（第 7～13 行）包住整个池，进入时 `set_dry_run(True)`、退出时恢复——与 `csrc/jit_kernels/bf16_gemm.hpp` 第 67～70 行的 `compile_without_load` 分支遥相呼应；
- `disable_cpu_fallback`（`disable_fallback.py` 第 44 行）：注意第 17～26 行 `_would_fallback` 的判断——只在算子**完全没有** NPU/Composite 实现时才动手；第 29～39 行 `_make_raiser` 生成的报错信息格式是 `[npu_no_fallback] operator 'xxx' has no NPU kernel ...`，排查问题时认准它。

### 站点 6：`deep_gemm/testing/bench.py`（965 行，只看骨架）

- 第 28～71 行 `KernelProfile`：字段与 `__repr__`——将来读性能输出时的"字段说明书"；
- 第 284～317 行 `bench_msprof` 签名与 docstring：`num_warmups=10`、`num_tests=30`、`flush_l2=True`、三种 backend（`slow`/`fast`/`persistent`）；
- 其余（PMU 记录解析、常驻会话）今天全部跳过，第 50～51 天回来。

### 站点 7：命令踩过的深层调用链（一瞥）

把 `python tests/test_bf16_gemm.py normal --skip-prof` 中**一次 kernel 调用**经过的站点串起来（只到"能说出每站名字"的深度，逐站精读是第 10～12 天）：

```text
tests/common.py:gemm_dg (第 66 行 getattr)
  → deep_gemm/__init__.py:64 (re-export _C.bf16_gemm_nt)
  → csrc/apis/gemm.hpp:424 bf16_gemm<false,false> → 第 440 行 get_gemm_desc (第 60 行定义)
  → csrc/jit_kernels/bf16_gemm.hpp:16 launch_bf16_gemm
      ├─ 第 17 行 select_gemm_config   (csrc/jit_kernels/heuristics.hpp:592)
      ├─ 第 18～42 行 std::format 生成 bf16_gemm_impl<...> 实例化代码
      ├─ dry_run? → 第 68 行 jit->compile_without_load     ← 编译阶段走这里
      └─ 第 71 行 jit->compile / 第 73 行 jit->launch       ← 正确性阶段走这里
  → deep_gemm/include/deep_gemm/bf16_gemm.hpp:27 bf16_gemm_impl (device)
  → 回到 tests/common.py:200 torch.npu.synchronize() → :201 assert_equal
```

顺带记一个伏笔：`m.def("bf16_gemm_nt", ...)` 的默认 `compiled_dims="nk"`（`csrc/apis/gemm.hpp` 第 635 行；tn/tt 是 `"mn"`，第 647、653 行）——N、K 编进模板、M 留作运行时参数，所以 fwd 里 `m=1/128/4096` 共用同一 (n,k) 时，nt 方向生成的代码是同一份、JIT 缓存直接命中。细节留到第 11～12 天。

**阅读检查点**（能回答再进入实践）：

1. `select` 省略不填会怎样？`--no-prof` 和 `--skip-prof` 是什么关系？
2. `run_gemm_test` 的第 157 行为什么传 `device='meta'`？dry_run 模式下 C++ 侧执行到哪一行就返回了？
3. `d_ref` 是用什么精度算出来的？`threshold=1e-5` 走的是 `assert_equal` 的哪个分支？
4. `gemm_dg` 中 `trans_ab='nt'` 和 `'nn'` 对 B 张量的处理有什么不同？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：环境盘点（约 15 分钟）**。逐条执行并把输出抄进第六节模板：

```bash
npu-smi info                                   # 芯片型号与状态
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
ls "$ASCEND_HOME_PATH/bin/bisheng" 2>/dev/null || ls /usr/local/Ascend/ascend-toolkit/latest/bin/bisheng
python --version
git submodule status                           # 确认 third-party/ 子模块非空
```

任何一条不满足（无 NPU、缺 CANN/`bisheng`、缺 `torch_npu`、子模块为空），都记入"阻塞项"，转 B 部分完成静态任务。

**步骤 2：构建（约 10 分钟）**。在仓库根目录：

```bash
./develop.sh
```

预期最后一行输出 `Linked: build/.../deep_gemm/_C...so -> deep_gemm/`。若报 `Error: No SO file found`，向上翻编译器报错（常见原因：`third-party/` 子模块未克隆，执行 `git submodule update --init --recursive` 后重试；或编译器不支持 C++20 `<format>`）。

**步骤 3：冒烟检查（约 5 分钟）**：

```bash
python -c "import deep_gemm; print(deep_gemm.__version__, deep_gemm.get_num_sms())"
```

预期打印版本号（如 `0.1.1`）与设备 AI 核数。若报 `ModuleNotFoundError: deep_gemm`，说明本地包不在搜索路径：在仓库根目录 `export PYTHONPATH=$PWD`（`python tests/test_bf16_gemm.py` 只会把 `tests/` 加入 `sys.path`，`deep_gemm/` 需要安装或 PYTHONPATH 提供）。

**步骤 4：运行最小测试（约 40 分钟，视机器而定）**：

```bash
python tests/test_bf16_gemm.py normal --skip-prof
```

对照下面的"预期输出形态"检查（耗时不具参考性，首次包含大量 JIT 编译）：

```text
Testing bf16 x bf16 gemm: Normal
Num tests:  54
Compiling: 100%|███| 960/960 [xx:xx<xx:xx,  x.xxit/s]      ← 并行预编译进度条
 > Perf (m=     1, n=  2112, k=  7168, layout=NT, bf16, acc=0,   fwd): OK
 > Perf (m=     1, n=   576, k=  7168, layout=NT, bf16, acc=0,   fwd): OK
 ...（fwd 共 27 行：m=1/128/4096 各 9 行；随后 dgrad/wgrad 共 27 行）
 > Perf (m=  4096, n=  7168, k=  2112, layout=NN, bf16, acc=0, dgrad): OK
 > Perf (m=  2112, n=  4096, k=  7168, layout=TN, fp32, acc=1, wgrad): OK
 ...（共 54 行，全部以 OK 结尾；任何一行抛断言异常即为失败）
```

记录要点：进度条总数（编译任务数，见 B-3 的推导）、54 行结果、总耗时、`$HOME/.dj` 缓存目录是否生成。再次运行一遍，体会缓存带来的提速，把两次耗时都记下来。

**步骤 5：失败时的分层定位（如有）**。按顺序判断故障属于哪一层，记入产出：

| 现象 | 层级 | 下一步 |
| :--- | :--- | :--- |
| `develop.sh` 编译报错 | 构建 | 看编译器输出：子模块 / C++20 / torch_npu 头文件路径 |
| `import deep_gemm` 失败 | 导入 | PYTHONPATH 或改用 `pip install . --no-build-isolation` |
| `[npu_no_fallback] operator ...` | 测试数据生成 | 某个 torch 算子缺 NPU 实现，记录算子名 |
| `assert_equal` / `calc_diff` 断言失败 | Kernel 或参考 | 记录用例 shape、layout、断言信息（深入分析是第 31 天任务） |
| kernel 启动后挂起/超时 | 启动 | 尝试 `ASCEND_LAUNCH_BLOCKING=1` 复现（README 第 98 行） |

### B. 无 Ascend 环境：静态替代任务

**B-1：命令行映射表（约 10 分钟）**。打开 `tests/test_bf16_gemm.py`，合上本教程，独立写出 `select` 7 个取值到 `test_bf16_gemm(...)` 实参的映射，再对照 3.3 节表格订正。

**B-2：手数用例（约 15 分钟）**。只看 `_enumerate_normal`（`generators.py` 第 544～576 行），在纸上回答：

1. `nk_list` 一共几组 (n,k)？bf16 输出与 fp32 输出各几组？
2. fwd 产生多少用例？dgrad/wgrad 呢？总数是多少？（答案：27、27、54，推导见 3.5 节）
3. 任选一个 (n,k)，写出它对应的 dgrad 用例三元组 `(m, n, k)` 和 layout 标签。（例如 (n,k)=(2112,7168) → dgrad 是 `(4096, 7168, 2112)`、`layout=NN`）

**B-3：手数编译任务（进阶，约 20 分钟）**。只看 `common.py` 第 152～178 行，推导编译阶段 `kernels` 列表的长度：

1. 每个用例、`with_bias=False` 时，每个 `trans_ab` 追加几个 partial？（普通 + alpha = 2 个）
2. `with_bias=True` 时呢？（只普通 1 个）→ 每用例基础变体 = 4×2 + 4×1 = **12 个**，54 用例共 **648**；
3. 第 169～174 行：bf16 输出且无累加的用例，`with_bias=False` 时每个 trans_ab 再追加随机舍入、FP32 输出两个变体（+8/用例）。数一数满足条件的用例（fwd 的 bf16 输出 21 个 + dgrad 9 个 + wgrad 不累加 9 个 = 39 个）；
4. 总数 = 39×20 + 15×12 = **960**——这就是进度条的分母。（由于 JIT 按代码内容缓存、且 `compiled_dims` 使部分实例重复，实际编译次数可能少于 960，但任务数不变。）

**B-4：单用例全链路推演（约 25 分钟）**。选 fwd 最小用例 `m=1, n=2112, k=7168`，沿 3.4～3.6 节与站点 7 的调用链，写一份"静态执行记录"：

| 步骤 | 位置 | 输入 → 输出 |
| :--- | :--- | :--- |
| 1. 构造用例 | `generators.py:548` `create_testcase` | `GemmTestDesc(Normal, bf16, bf16, bf16, (1,2112,7168), K, K, acc=False, tag='  fwd')` |
| 2. 编译期生成 | `common.py:157` | meta 张量：a=[1,7168]、b=[2112,7168]、d=[1,2112] |
| 3. 正确期生成 | `generators.py:300` | NPU 实数据；`d_ref = (a.float() @ b.float().T).to(bf16)` |
| 4. 调用（nt） | `common.py:66` | `deep_gemm.bf16_gemm_nt(a, b, d)` |
| 5. C++ 检查 | `apis/gemm.hpp:60` | `GemmDesc(m=1, n=2112, k=7168, major_a=K, major_b=K, ...)` |
| 6. JIT | `jit_kernels/bf16_gemm.hpp:16` | 选配置 → 生成 `bf16_gemm_impl<...>` 代码 → compile/launch |
| 7. 校验 | `common.py:200～201` | synchronize → `assert_equal(d, d_ref, threshold=1e-5)` |

补齐 `tn/nn/tt` 三个方向第 4 步的差异（提示：`common.py` 第 64～65 行对 A/B 的逻辑转置规则）。

**B-5：记录阻塞项（约 10 分钟）**。按第六节模板 C 填写：缺什么（NPU/CANN/torch_npu/…）、卡在哪条命令、替代完成了哪些静态任务。

## 六、当天产出

学习计划要求今天的产出是**环境信息、运行结果或环境阻塞项记录**。按实际情况选 A 或 C 填写（B 的推演笔记附在后面）：

```text
# DeepGEMM-Ascend 第 9 天记录：构建并运行最小测试

## A. 环境信息（有环境必填）
- 日期 / 仓库版本（git rev-parse --short HEAD）：＿＿＿
- NPU 型号（npu-smi info）：＿＿＿；AI 核数（get_num_sms）：＿＿
- CANN：$ASCEND_HOME_PATH = ＿＿＿；bisheng 存在：是/否
- torch / torch_npu / Python：＿＿ / ＿＿ / ＿＿
- deep_gemm.__version__：＿＿；构建方式：develop.sh / pip install

## B. 运行结果（有环境必填）
- 命令：python tests/test_bf16_gemm.py normal --skip-prof
- 构建结果（Linked 行）：＿＿＿
- Num tests = ＿＿；Compiling 进度条总数 = ＿＿（对比推导值 960）
- 首次运行耗时 ＿＿；二次运行耗时 ＿＿（JIT 缓存 $HOME/.dj）
- 54 行结果是否全部 OK：是/否；失败用例（shape/layout/断言信息）：＿＿＿

## C. 环境阻塞项（无环境必填）
- 阻塞项：□ 无 NPU  □ 无 CANN/bisheng  □ 无 torch_npu  □ 子模块未克隆  □ 其他：＿＿
- 卡住的命令与报错原文：＿＿＿
- 已完成的静态替代任务：□ B-1 映射表  □ B-2 数用例  □ B-3 数编译任务  □ B-4 全链路推演

## D. 静态推演笔记（附 B-2/B-3/B-4 的推导过程与结论）

## E. 遗留问题（交给第 10～13 天）
- ＿＿＿
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `./develop.sh` 做了哪几件事？为什么运行测试前必须先执行它？device kernel 是在这一步编译的吗？
2. `select` 参数有哪些取值？`normal` 和默认值 `all` 分别会执行哪些 `test_bf16_gemm` 调用？
3. `--skip-prof` 的确切语义是什么？加上与去掉它，`print_perf` 的输出有什么区别？
4. `run_gemm_test` 分哪三个阶段？每个阶段最关键的一个函数是什么？
5. 为什么要用 meta 张量 + `dry_run` 做并行预编译？`dry_run` 在 C++ 侧对应哪个分支？
6. `disable_cpu_fallback()` 解决什么问题？如果某个 torch 算子悄悄回退到 CPU，测试会以什么形式暴露？
7. 参考结果 `d_ref` 如何计算？`threshold=1e-5` 采用什么度量？为什么 BF16 不要求逐位相等？
8. `normal` 用例集共多少个？fwd/dgrad/wgrad 各对应什么 `layout` 标签、什么输出 dtype？
9. `alpha` 与 bias 同时传入时代码预期发生什么？Python 侧在哪几行、C++ 侧在哪一行检查？
10.（思考题）54 个用例为什么会产生 960 个编译任务？既然 `compiled_dims="nk"` 让不同 M 共享同一份 nt 方向代码，为什么还要为每个用例单独提交编译任务？

**参考答案：**

1. 四件事：`cd` 到仓库根；`rm -rf build dist` 清理；`DG_USE_LOCAL_VERSION=0 python3 setup.py build` 编译 `deep_gemm._C`（唯一源文件 `csrc/python_api.cpp`，`setup.py` 第 43 行）；把 `.so` 与 `stubs/_C.pyi` 符号链接进 `deep_gemm/`（`develop.sh` 第 5～20 行）。不先执行它，`import deep_gemm` 的 `from . import _C` 就会失败。device kernel **不是**这一步编译的——它们是随包分发的模板头文件，运行时由 JIT 编译（第 8 天 3.3 节）。
2. `all/normal/unaligned-k/unaligned-mn/strided/mgrouped/kgrouped`（`test_bf16_gemm.py` 第 25 行）。`normal` 只调 `test_bf16_gemm(GemmType.Normal, skip_prof=...)`（第 37～38 行）；`all` 依次调 6 组：normal、unaligned-mn、unaligned-k、strided、mgrouped、kgrouped（第 43～50 行）。
3. 跳过性能测量、只测正确性（第 26 行 help）。加上它时 `prof=None`，`print_perf` 每行以 `OK` 结尾（`utils.py` 第 113～114 行）；去掉后 `bench_msprof` 测 10 次预热 + 30 次迭代（清 L2），每行打印 `耗时 us | TFLOPS | GB/s | mad% | mte2%`（`common.py` 第 225～236 行、`utils.py` 第 109～112 行）。
4. ①并行预编译（`par_compile`，`common.py` 第 175 行）；②正确性验证（`gemm_dg` + `assert_equal`，第 199～201 行）；③性能测量（`bench_msprof`，第 228 行）。此外阶段 0 还有 `disable_cpu_fallback()`（第 142 行）。
5. JIT 编译最慢且任务相互独立：先在 `dry_run` 下用线程池并发触发，`set_dry_run(True)` 使 C++ 侧走 `jit->compile_without_load`（`jit_kernels/bf16_gemm.hpp` 第 67～70 行）——只编译不加载、不占 NPU；meta 张量（`common.py` 第 157 行）只提供形状元数据而无存储，`get_gemm_desc` 对 meta 有专门处理（`apis/gemm.hpp` 第 105～106 行）。
6. 防止某个算子静默回退 CPU 导致"测的不是 NPU"。它会为所有没有 NPU/Composite 实现的算子注册抛异常的 PrivateUse1 内核（`disable_fallback.py` 第 44～72 行），一旦回退立即报 `[npu_no_fallback] operator '...' has no NPU kernel ...`。
7. `d_ref` 由 A、B 升 FP32 后做矩阵乘（可加 alpha/bias）再舍入到目标 dtype（`generators.py` 第 320～322 行）。`threshold=1e-5 > 0` 走 `calc_diff` 余弦相似度分支（`utils.py` 第 54～58 行）；不要求逐位相等是因为 kernel 与参考的 K 维累加次序不同，BF16 下最多差约 1 ULP（`utils.py` 第 50～51 行注释）。
8. 54 个。fwd：K,K → `layout=NT`，输出 bf16（21 个）或 fp32（6 个）；dgrad：K,MN → `NN`，bf16；wgrad：MN,MN → `TN`，fp32 累加（`acc=1`）与 bf16 不累加各 9 个（`generators.py` 第 566～576 行，标签由 `major_opt` 第 750 行拼出）。
9. 预期抛出包含 `alpha and bias cannot be used together` 的 `RuntimeError`：Python 侧 `common.py` 第 191～198 行捕获并断言消息；C++ 侧检查点在 `apis/gemm.hpp` 第 116 行。
10. 每个用例在编译阶段展开为多种变体：4 种转置 × {普通、alpha}（无 bias）+ 4 种转置 × {bias}，共 12 个基础变体；bf16 输出且无累加的 39 个用例每个再加随机舍入、FP32 输出 8 个变体——39×20 + 15×12 = 960（`common.py` 第 152～178 行）。`compiled_dims="nk"` 只让**生成出的代码内容**相同从而命中 JIT 缓存，测试侧仍按用例逐个提交编译任务（partial 列表长度不变），只是重复内容的编译被缓存吸收。
