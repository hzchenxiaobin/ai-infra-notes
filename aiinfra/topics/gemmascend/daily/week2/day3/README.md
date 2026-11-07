# 第 2 周 · 第 3 天（总第 10 天）：追踪 Python 公开接口

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 2 周"项目结构与 JIT 调用链"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 50 分钟，源码导读约 60 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前两天我们建好了仓库地图（第 8 天）、又让整条链路真实地跑了一遍（第 9 天）。今天开始逐层拆解调用链，而**第一站**就是离用户最近的一层：当你在 Python 里写下 `deep_gemm.bf16_gemm_nn(a, b, d)` 时，这个调用到底从哪里来、参数怎么穿过 Python/C++ 边界、进入 C++ 后最先发生什么。今天只追到 `launch_bf16_gemm` 的门口——它内部的配置选择与 JIT 编译是第 11、12 天的内容；`GemmDesc` 的逐字段分析是第 11 天的内容。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 通读 `deep_gemm/__init__.py`，把它的内容分成四类（环境准备、`_C` 的 re-export、Python 层包装函数、子模块 re-export），并说出 `bf16_gemm_nn`、`fp8_fp4_gemm_nn` 等 37 个接口名真正的定义位置；
2. 解释 `csrc/python_api.cpp` 的"接线板"结构：`PYBIND11_MODULE` 如何把 `_C` 扩展与各 `register_apis` 连接起来；
3. 说出后缀 `nt/nn/tn/tt` 的确切含义：它描述的是**数学运算**（原样/转置），对应模板参数 `<kTransA, kTransB>` 与用户传入张量的形状，并能解释为什么 `nn` 的第二个模板参数是 `true`；
4. 复述 `bf16_gemm<kTransA, kTransB>` 进入 C++ 后的五步：逻辑转置 → dtype 断言 → epilogue 归一 → `get_gemm_desc` → `launch_bf16_gemm`；
5. 画出 pybind11 的参数类型映射：`torch.Tensor`、`None`、`(Tensor, Tensor)` 元组、`float`、Epilogue 对象分别变成什么 C++ 类型；
6. 说明 `fp8_fp4_gemm_nn` 相对 `bf16_gemm_nn` 的三点差异：`(数据, SF)` 二元组、recipe 参数、按 dtype 的三路 dispatch；
7. 独立写出 `deep_gemm.bf16_gemm_nn` 从 Python 到 `launch_bf16_gemm` 的入口调用链，每一站标注文件、函数和职责。

**当天产出：`deep_gemm.bf16_gemm_nn` 的入口调用链**（模板见第六节，参考答案见 3.7 节）。

## 二、预备知识

- **第 8 天的代码地图**：五类代码的位置；`_C` 是由 `csrc/python_api.cpp` 编译出的 pybind11 扩展；Device kernel 是运行时 JIT 编译的模板头文件。
- **第 9 天的运行体验**：`./develop.sh` 把 `.so` 链接进 `deep_gemm/`；`tests/common.py` 的 `gemm_dg` 用 `getattr(deep_gemm, f'bf16_gemm_{trans_ab}')` 拼出接口名；`DJ_HOST_ASSERT` 失败在 Python 侧表现为 `RuntimeError`。
- **第 1 周概念**：M/N/K 与 `D = A × B`（第 1 天）；Row-major/Column-major、stride、**逻辑转置不搬数据**（第 2 天）；BF16/FP8/FP4 与 Scaling Factor（第 4 天）。今天的 `get_transposed`、`Major::K/MN` 全部建立在这些概念上。
- **Python/C++ 基础**：pybind11 的 `m.def`/`py::arg`、C++ 函数模板与 `inline constexpr` 变量模板、`std::optional`/`std::pair`/`std::tuple`/`std::shared_ptr`。不需要写过，能对照读懂即可。

## 三、详细讲解

### 3.1 从 `import deep_gemm` 开始：`__init__.py` 的门面职责

`deep_gemm/__init__.py` 全文 159 行，是包的唯一 Python 门面。按行号分四段读：

| 行号 | 内容 | 职责 |
| :--- | :--- | :--- |
| 1～13 | `__version__`；`from .envs import persistent_envs` 并写回 `os.environ` | `envs.py` **不在仓库里**，是构建时由 `setup.py` 的 `generate_default_envs()`（第 131～140 行）生成的：把构建机器上的 `DG_JIT_CACHE_DIR`、`DG_JIT_DEBUG` 等 7 个变量（`setup.py` 第 17～25 行）快照下来，用户未显式设置时恢复默认 |
| 16～22 | `ASCEND_HOME` 探测；`get_library_root_path()` | 定位 Ascend 工具链；返回包根路径（`include/` 所在处）——它稍后交给 `_C.init`，JIT 编译 kernel 时用它找头文件 |
| 25～81 | `from . import _C` + 三批 `from ._C import ...` | **全部核心接口都来自 `_C` 扩展**：第 26～36 行配置函数（`set_num_sms` 等）；第 39～42 行 `cublaslt_*` 兼容别名；第 45～81 行 37 个 kernel 接口（fp8_fp4、fp8、aclnn、bf16、einsum、attention、layout） |
| 83～135 | `get_paged_sparse_mqa_logits_metadata`、`fp8_fp4_paged_sparse_mqa_logits` | **仅有的两个 Python 层包装**：内部调用 `_C._get_paged_sparse_mqa_logits_metadata` 等下划线绑定，但额外管理 FIRST/REUSE 状态（挂在 `metadata._paged_sparse_mqa_state` 上）与配置一致性检查 |
| 137～157 | `from .mega import ...`、`from ._C import epilogue`、locality、tilelang | 子模块 re-export：MegaMoE 接口、`deep_gemm.epilogue` 子模块、局部化辅助、TileLang 算子 |
| 159 | `_C.init(__version__, get_library_root_path())` | 收尾动作：初始化 JIT 配置（见 3.2 节） |

三个要点：

1. **`bf16_gemm_nn` 没有任何 Python 逻辑**。`__init__.py` 第 64 行 `from ._C import (bf16_gemm_nt, bf16_gemm_nn, ...)` 只是改名再导出。在 Python 里查 `inspect.getsource(deep_gemm.bf16_gemm_nn)` 会失败——它是 pybind11 生成的内置函数。
2. **别名有两处，机制不同**：`fp8_gemm_nn` 与 `m_grouped_fp8_gemm_nn_contiguous` 并没有单独的 C++ 函数，是 `csrc/apis/gemm.hpp` 第 624～629 行用 `m.attr("fp8_gemm_nn") = m.attr("fp8_fp4_gemm_nn")` 在模块对象上做的属性别名；`cublaslt_gemm_nn` 同理是 `csrc/apis/aclnn.hpp` 第 290～294 行给 `aclnn_bf16_gemm_nn` 起的别名，目的是兼容上游 cuBLASLt 风格的 API 名。
3. **为什么 paged sparse MQA 要包一层 Python**：见 `__init__.py` 第 83～86 行的注释——公开入口要同时管理"配置"与"FIRST/REUSE 状态"，直接暴露 `_C` 的下划线绑定会绕过状态管理，导致元数据与调用不匹配。这是"哪些接口值得留在 Python 层"的一个判断范例。

### 3.2 接口的源头：`_C` 扩展与 `PYBIND11_MODULE`

`_C` 是一个编译好的共享库（`.so`）。第 9 天已经知道：`develop.sh` 用 `python3 setup.py build` 编译并把 `.so` 链接进 `deep_gemm/`；正式安装则用 `pip install . --no-build-isolation`。今天要看清"编译了什么"：

- `setup.py` 第 43 行：`sources = ['csrc/python_api.cpp']`——**整个 C++ 侧只有一个翻译单元**；第 86～92 行把它编成名为 `deep_gemm._C` 的 `CppExtension`。`csrc/apis/`、`csrc/utils/`、`csrc/jit_kernels/` 全部是头文件，被 `python_api.cpp` 逐个 `#include`（第 7～14 行）。
- `csrc/python_api.cpp` 第 25～27 行的兜底 `#ifndef TORCH_EXTENSION_NAME / #define TORCH_EXTENSION_NAME _C`：用 torch 的构建系统时该宏由构建系统注入（值为 `deep_gemm._C`）；`CMakeLists.txt` 第 31 行的 `pybind11_add_module(_C csrc/python_api.cpp)` 是给 IDE 索引用的另一条构建路径，模块名同样叫 `_C`。
- 第 30～41 行是全文核心：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM-Ascend C++ library";
    deep_gemm::config::register_apis(m);         // csrc/apis/runtime.hpp
    deep_gemm::epilogue_class::register_apis(m);  // csrc/apis/epilogue_class.hpp
    deep_gemm::gemm_api::register_apis(m);        // csrc/apis/gemm.hpp
    deep_gemm::attention_api::register_apis(m);   // csrc/apis/attention.hpp
    deep_gemm::einsum::register_apis(m);          // csrc/apis/einsum.hpp
    deep_gemm::layout::register_apis(m);          // csrc/apis/layout.hpp
    deep_gemm::aclnn_api::register_apis(m);       // csrc/apis/aclnn.hpp
    deep_gemm::mega_moe::register_apis(m);        // csrc/apis/mega_moe.hpp
    deep_jit::register_python_api(m, deep_gemm::jit);
}
```

`python_api.cpp` 自己**不定义任何 GEMM 接口**，它只做三件事：把各 `apis/*.hpp` 的 `static void register_apis(pybind11::module_& m)` 接线（每个头文件自带注册函数，新增接口只改对应头文件即可）；定义 `deep_gemm::get_current_npu_stream()`（第 21～23 行，把 torch_npu 的当前流桥接给 C++ 侧）；第 1 行先 `#include <hccl/hccl.h>` 再包含 torch_npu 头，避免两个 HCCL 声明打架。

还有一个与"接口从哪来"直接相关的副产品：`setup.py` 第 120 行 `generate_pyi_file(name='_C', root='./csrc/apis', output_dir='./stubs')` 会从 `csrc/apis/` 的 `m.def` 调用生成 `stubs/_C.pyi` 类型桩（`develop.sh` 第 19 行把它也链接进包目录）。所以 IDE 里 `deep_gemm.bf16_gemm_nn(a, b, d)` 的参数提示，源头同样是 `csrc/apis/gemm.hpp`。

最后是 `__init__.py` 第 159 行那句 `_C.init(__version__, get_library_root_path())`：它进入 `csrc/apis/runtime.hpp` 第 41～44 行的 `m.def("init", ...)`，再调 `csrc/jit/jit.hpp` 第 78～92 行的 `init_jit`——记录版本号和包根路径，把 `include/` 目录登记为 JIT 编译 kernel 时的头文件搜索路径，并以 **LazyInit** 方式配置 `jit` 与 `runtime` 两个全局对象（`jit.hpp` 第 76 行）。注意 `runtime.hpp` 第 42 行的注释："Keep imports free of NPU runtime initialization so callers can fork"——import 与 init 都不触碰 NPU 设备，真正的初始化延迟到第一次使用，这保证 `import deep_gemm` 之后进程还能安全地 fork（测试的并行编译正是利用了这一点，见第 9 天 3.4 节）。

### 3.3 后缀 nt/nn/tn/tt：一种规范形式，四个入口

DeepGEMM 的 kernel 内部只有一种规范形式：

```text
D[M, N] = A_eff[M, K] @ B_eff[N, K]^T
```

无论用户用什么布局调用，binding 层都把它**零拷贝地**规范化到这个形式，再交给后面完全相同的 `get_gemm_desc` → `launch_*` 链路。四个入口的后缀描述的是**数学运算**：`n` 表示该操作数原样参与 `D = op(A) × op(B)`，`t` 表示参与的是它的转置。由此推出用户张量形状与 binding 动作（与 `csrc/apis/gemm.hpp` 第 415～419 行的 doc 注释一致）：

| 接口 | 数学含义 | 用户传 A | 用户传 B | 模板参数 | binding 对 A | binding 对 B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bf16_gemm_nt` | `A @ B^T` | `[M, K]` | `[N, K]` | `<false, false>` | 原样 | 原样 |
| `bf16_gemm_nn` | `A @ B` | `[M, K]` | `[K, N]` | `<false, true>` | 原样 | `transpose(0,1)` |
| `bf16_gemm_tn` | `A^T @ B` | `[K, M]` | `[K, N]` | `<true, true>` | `transpose(0,1)` | `transpose(0,1)` |
| `bf16_gemm_tt` | `A^T @ B^T` | `[K, M]` | `[N, K]` | `<true, false>` | `transpose(0,1)` | 原样 |

两个容易搞反的地方：

- **A 与 B 的规律相反**。对 A：后缀 `n`（原样）→ 用户张量就是规范形式 `[M,K]`，不需要转置（模板 `false`）；后缀 `t` → 用户给 `[K,M]`，binding 转回 `[M,K]`（模板 `true`）。对 B：后缀 `t`（数学上用 `B^T`）→ 用户张量 `[N,K]` 恰好就是 `B_eff`，不需要转置（模板 `false`）；后缀 `n`（原样）→ `B_eff` 必须等于 `user_B^T`，用户给 `[K,N]`，binding 转成 `[N,K]`（模板 `true`）。
- **转置是纯逻辑的**。binding 里的 `get_transposed`（`csrc/utils/layout.hpp` 第 88～99 行）只调 `tensor.transpose(0, 1)`——交换 shape 和 stride 的元数据，一个字节都不搬（第 2 天的概念）。README 第 61 行因此可以直白地写："`bf16_gemm_nn` and `fp8_fp4_gemm_nn` interfaces take `A[M, K]` and `B[K, N]`"。

测试侧可以交叉验证：`tests/common.py` 第 64～65 行，`a = a if trans_ab[0] == 'n' else get_transposed(a, 0, 1)`、`b = b if trans_ab[1] == 't' else get_transposed(b, 0, 1)`——测试生成器内部保存的是规范形式 A `[M,K]`、B `[N,K]`，于是对 `nn` 调用先把 B 转成 `[K,N]` 交给接口，binding 再转回来。`get_available_trans_ab()`（`tests/generators.py` 第 290 行）显示 Normal 类型会遍历全部四种方向。

### 3.4 `bf16_gemm_nn` 进入 C++ 后的五步

Python 侧的名字经 `__init__.py` 第 64 行 re-export 后，直接命中 `csrc/apis/gemm.hpp` 第 638～643 行的 binding：

```cpp
m.def("bf16_gemm_nn", bf16_gemm_nn,
      py::arg("a"), py::arg("b"), py::arg("d"),
      py::arg("c") = std::nullopt,
      py::arg("compiled_dims") = "nk",
      py::arg("alpha") = std::nullopt,
      py::arg("epilogue") = nullptr);
```

`bf16_gemm_nn` 在第 452 行被定义为函数模板实例：`inline constexpr auto bf16_gemm_nn = bf16_gemm<false, true>;`。`bf16_gemm` 模板本体（第 424～448 行）只做五件事：

```cpp
template <bool kTransA, bool kTransB>
static void bf16_gemm(const torch::Tensor& a_, const torch::Tensor& b_,
                      const torch::Tensor& d, const std::optional<torch::Tensor>& c,
                      const std::string& compiled_dims, const std::optional<float>& alpha,
                      const std::shared_ptr<EpilogueClass>& epilogue_class) {
    auto a = get_transposed<kTransA>(a_, 0, 1);   // ① nn: A 原样
    auto b = get_transposed<kTransB>(b_, 0, 1);   // ① nn: B [K,N] → [N,K]（零拷贝）
    DJ_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    auto maybe_desc = get_gemm_desc(
        GemmType::Normal,
        a, b, d, c, resolve_epilogue_class(epilogue_class, c, d, alpha),  // ② ③
        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, compiled_dims);                                    // bf16 无 SF/recipe
    if (!maybe_desc.has_value()) return;                                 // ④ 空问题提前返回
    deep_gemm::launch_bf16_gemm(maybe_desc.value());                     // ⑤ 交给下一层
}
```

**① 逻辑转置**：如 3.3 节。`get_transposed` 是 `if constexpr` 分发（`csrc/utils/layout.hpp` 第 88～99 行）：`kEnable=false` 直接原样返回；对 `torch::Tensor` 调 `transpose`；对 `std::pair<Tensor, Tensor>`（低精度接口的 `(数据, SF)` 对）则两个都转。

**② dtype 断言**：A、B 必须是 BF16；D 可以是 BF16 或 FP32（FP32 输出即"直接高精度写出"）。`DJ_HOST_ASSERT` 失败时向 Python 抛 `RuntimeError`——第 9 天见过的 `alpha and bias cannot be used together` 拒绝测试就是这个机制。

**③ epilogue 归一**：`resolve_epilogue_class`（`csrc/jit_kernels/epilogue_class.hpp` 第 110～128 行）把三种互斥输入归一成一个 `EpilogueClass` 对象：传了 `alpha` → `AlphaEpilogue`；传了 `epilogue` 对象 → 原样；两者都没有 → `IdentityEpilogue`。第 117 行断言三者（epilogue/alpha/sfd）至多一个。可用的 epilogue 类有四个（`csrc/apis/epilogue_class.hpp` 第 12～20 行注册到 `deep_gemm.epilogue` 子模块）：`Identity`、`Alpha(alpha)`、`FP8Quantization(sfd)`、`BF16StochasticRounding()`。每个类知道自己对应的设备侧算子名，如 `AlphaEpilogue::get_epilogue_operator_type()` 返回 `"epilogue::operators::ScaleByAlpha"`。

**④ `get_gemm_desc`（今天只看骨架，逐字段分析是第 11 天）**：`csrc/apis/gemm.hpp` 第 60～197 行，做四件事——

- **抽形状并交叉校验**：`std::tie(m, n, k) = get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d);`（第 83 行）。这套 `csrc/utils/layout.hpp` 第 140～219 行的机制给每个张量标注维度含义（A 标 "mk"、B 标 "nk"、D 标 "mn"），再按字母扫描所有张量收集同名维度的尺寸：`m` 来自 A 的第 0 维和 D 的第 0 维，两者必须相等；`k` 来自 A 的第 1 维和 B 的第 1 维；以此类推。任何冲突都会 `DJ_PANIC("Conflicting shape dimension ...")`。FP4 有个细节：打包存储（两个 FP4 挤一个字节）时 major 轴的尺寸要乘 2（第 185～187 行）。
- **判 major 与 stride**：`get_major_type_ab`（`csrc/utils/layout.hpp` 第 44～47 行）——`stride(-1)==1` 为 `Major::K`，`stride(-2)==1` 为 `Major::MN`；两者都是 1 则报"ambiguous major"。D 永远要求 `stride(-1)==1`（N-major，第 49～52 行）。
- **处理 `c` 的三种语义**（第 106～123 行）：`c=None` → 无累加；`c` 与 `d` **同一块存储且同形**（`data_ptr()` 相等，meta/空张量用 `is_alias_of`）→ 累加模式，`GemmDesc.acc=true`，语义 `D += A@B^T`；`c` 是独立的连续 `[1, N]` → bias 模式，包装成 `BiasEpilogue`，且断言不能与 alpha 同用（第 115～117 行）。这正是 README 第 58～60 行"C 形状 `[M, N]` 做累加、`[1, N]` 做 bias"的实现处。
- **空问题提前返回**：`gemm_early_return`（`csrc/utils/layout.hpp` 第 54～86 行）：`m==0` 或 `n==0` 直接返回；`k==0` 时把 D 清零（或从 C 拷贝）后返回——都不构造 `GemmDesc`、不触发 JIT。

通过后返回填好的 `GemmDesc`（结构定义在 `csrc/jit_kernels/gemm_config.hpp` 第 39～95 行）：裸指针 `a_ptr/b_ptr/d_ptr`、dtype、`m/n/k`、`major_a/major_b`、`acc`、`gemm_type`（枚举定义在 `deep_gemm/include/deep_gemm/common.hpp` 第 87～92 行，Normal 的分支连 `grouped_layout` 都是 `nullptr`）、外层 stride、epilogue 对象。**它是"问题是什么"的全部描述**，与第 12 天的 `GemmConfig`（"怎么算"的执行配置）是一对。

**⑤ `launch_bf16_gemm`**：`csrc/jit_kernels/bf16_gemm.hpp` 第 16～78 行——选配置、`std::format` 生成 `bf16_gemm_impl<...>` 模板实例化代码、`jit->compile`、`jit->launch`。今天到此为止，只记住一句话：**这一步之后，Python 传来的信息全部收敛在 `GemmDesc` 一个结构体里**。

### 3.5 `fp8_fp4_gemm_nn`：同一模式上的三个扩展

`fp8_fp4_gemm_nn` 的 binding（`csrc/apis/gemm.hpp` 第 562～569 行）与 bf16 版共享同一套模式（模板 `fp8_fp4_gemm<false, true>`，第 225～251 行；别名第 255 行），差异有三处：

1. **`a`/`b` 是 `(数据, SF)` 二元组**。C++ 签名是 `const std::pair<torch::Tensor, torch::Tensor>&`，第 207～213 行的 doc 注释写明：数据 `[M, K]`（nn 方向），SF 是 `[ceil(M/gran_m), ceil(K/gran_k)]`、FP32 或打包 ue8m0 的 short。`get_transposed` 对 pair 的处理让 SF 与数据同步转置。
2. **多出量化粒度参数**：`recipe`（三元组 `(gran_m, gran_n, gran_k)`）或 `recipe_a`/`recipe_b`（各 `(gran_mn, gran_k)`，二者 `gran_k` 必须相同），二选一互斥。合并逻辑在 `layout::get_recipe`（`csrc/apis/layout.hpp` 第 113～128 行）；缺省 `recipe_a = recipe_b = (1, 32)`。在 `get_gemm_desc` 内部（`csrc/apis/gemm.hpp` 第 125～154 行），SF 会被 `transform_sf_into_required_layout` 转成 kernel 需要的 MN-major 打包 ue8m0 布局——细节留到第 33 天，今天只需知道"SF 的重排发生在 `get_gemm_desc` 里，转换结果由 `GemmDesc::sfa/sfb` 持有引用，防止 kernel 启动前张量被释放"（`csrc/jit_kernels/gemm_config.hpp` 第 77～78 行注释）。
3. **按 dtype 三路 dispatch**：`dispatch_fp8_fp4_gemm`（`csrc/apis/gemm.hpp` 第 22～33 行）看 `(a_dtype, b_dtype)` 组合：`fp8×fp8 → launch_fp8_gemm`；`fp4×fp4 → launch_fp4_gemm`；`fp8×fp4 → launch_fp8_dequant_gemm`；其余组合断言失败（`fp4×fp8` 已不再支持）。也就是说 `fp8_fp4_gemm_nn` 是一个"家族入口"，内部再分派到三条 kernel 链。

另外注意：`fp8_gemm_nn` 只是 `fp8_fp4_gemm_nn` 的属性别名（第 625 行），`tests/common.py` 第 51 行按 `has_sf` 拼接口名 `'fp8_fp4'` 或 `'bf16'`，两组接口在 Python 侧的调用形态因此完全一致。

### 3.6 Python 参数如何进入 C++：pybind11 的类型桥梁

pybind11 的 `m.def` 用 `py::arg` 同时定义了**关键字名**和**默认值**。以 `bf16_gemm_nn` 为例，Python 侧每个实参到 C++ 形参的映射：

| Python 实参 | C++ 形参 | 转换规则 | 缺省 |
| :--- | :--- | :--- | :--- |
| `a`、`b`、`d`：`torch.Tensor` | `const torch::Tensor&` | pybind11 + `torch/python.h` 的 ATen 注册器，零拷贝借用 | 必填 |
| `c=None` 或 Tensor | `const std::optional<torch::Tensor>&` | `None → std::nullopt` | `nullopt` |
| `compiled_dims="nk"`：str | `const std::string&` | 字符串转换；决定哪些维度编成编译期常量（`get_compiled_dim`，`csrc/jit_kernels/gemm_config.hpp` 第 17～21 行；nn/nt 默认 `"nk"`，tn/tt 默认 `"mn"`） | `"nk"` |
| `alpha=None` 或 float | `const std::optional<float>&` | `None → nullopt`，float 直接装入 | `nullopt` |
| `epilogue=None` 或 `deep_gemm.epilogue.Alpha(0.5)` 等实例 | `const std::shared_ptr<EpilogueClass>&` | 注册过的类实例按 `shared_ptr` 传入；`None → nullptr` | `nullptr` |

对 `fp8_fp4_gemm_nn` 再加三种：`(Tensor, Tensor)` 元组 → `std::pair`；`(1, 128, 32)` 元组 → `std::tuple<int,int,int>`（recipe）；`(1, 32)` → `std::tuple<int,int>`；`disable_ue8m0_cast` 布尔 → `bool`。

由此可以直接读懂两种真实调用。README 第 64 行的示例用关键字风格：`deep_gemm.bf16_gemm_nn(a, b, c=bias, d=d)`——bias 走 `c` 形参、输出走 `d` 形参。`tests/common.py` 第 49～66 行的 `gemm_dg` 则是仓库内的标准消费者：先按测试语义组装 `extra_args`（bias 时 `c` 是 `[1,N]`；累加时 `d.copy_(c)` 后 `c=d`；Normal 加 `alpha`、`recipe_a/recipe_b`），再 `getattr(deep_gemm, f'{gemm_name}_gemm_{trans_ab}')(a, b, d, **extra_args)` 一行打尽四种后缀 × 两种 dtype 的所有组合——**接口形状的一致性让测试可以用字符串拼接口名**，这本身是这个 API 设计的一个优点。

### 3.7 完整入口调用链（当天产出）

把前面所有结论串起来，就是今天要交的图（`launch_bf16_gemm` 之后留到第 11～12 天补全）：

```text
Python: deep_gemm.bf16_gemm_nn(a, b, d, c=None, compiled_dims="nk", alpha=None, epilogue=None)
│
├─ deep_gemm/__init__.py:64        from ._C import bf16_gemm_nn（纯 re-export，无 Python 逻辑）
│
├─ csrc/python_api.cpp:30,35      PYBIND11_MODULE(_C) → gemm_api::register_apis(m)
│
├─ csrc/apis/gemm.hpp:638         m.def("bf16_gemm_nn", bf16_gemm_nn, py::arg(...)...)
│                                  ← 参数名、类型转换、默认值全部在此定死（3.6 节映射表）
│
├─ csrc/apis/gemm.hpp:452,424     bf16_gemm_nn = bf16_gemm<false, true>，五步：
│   ├─ :434-435  get_transposed    A 原样 [M,K]；B [K,N]→[N,K]（utils/layout.hpp:88，零拷贝）
│   ├─ :436-438  dtype 断言         A/B=bf16；D=bf16 或 fp32
│   ├─ :442      resolve_epilogue_class  alpha/epilogue/None → Alpha/…/IdentityEpilogue
│   │                               （jit_kernels/epilogue_class.hpp:110，并做 check）
│   ├─ :440-445  get_gemm_desc(GemmType::Normal, ...)
│   │                               （apis/gemm.hpp:60）
│   │     ├─ get_shape_by_spec<"mnk">  utils/layout.hpp:141（m/n/k 交叉校验，冲突即 panic）
│   │     ├─ get_major_type_ab         utils/layout.hpp:44（stride → Major::K/MN）
│   │     ├─ c: 与 d 同指针→acc=true；独立 [1,N]→BiasEpilogue；None→都不加
│   │     ├─ gemm_early_return         utils/layout.hpp:54（m/n=0 或 k=0 直接返回）
│   │     └─ 构造 GemmDesc             jit_kernels/gemm_config.hpp:39（问题描述结构体）
│   └─ :446      maybe_desc 为空（早退）则直接返回
│
└─ csrc/apis/gemm.hpp:447         launch_bf16_gemm(desc)
                                   └─ csrc/jit_kernels/bf16_gemm.hpp:16
                                      select_gemm_config → 生成 bf16_gemm_impl<...> 代码
                                      → jit->compile → jit->launch
                                      ※ 从这里开始属于第 11～12 天
```

## 四、源码导读

今天的导读目标是**把 3.7 节的每一站落到真实代码行上**。每站 5～10 分钟，重点核对"文件 + 行号 + 关键函数 + 它在调用链中的位置"。

### 站点 1：`deep_gemm/__init__.py`（159 行，今天的第一个精读对象）

按 3.1 节表格逐段读，重点确认三处：

- 第 25 行 `from . import _C`——所有接口的物理来源；对照第 26～81 行三批 import，把 37 个接口名分成 config / cublaslt 别名 / kernel 三组；
- 第 87～135 行两个 Python 包装函数：读 `get_paged_sparse_mqa_logits_metadata` 的 docstring（第 88～106 行，描述了调用方必须保证的数据约束）和 `fp8_fp4_paged_sparse_mqa_logits` 里的状态检查（第 122～134 行，`metadata` 必须来自配对函数、配置必须一致、首调后把 FIRST 翻成 REUSE）——体会"哪些逻辑必须放在 Python"；
- 第 159 行 `_C.init(...)`——结合站点 7 读。

### 站点 2：`setup.py`（161 行，只读构建接线）

- 第 17～25 行 `persistent_env_names`：7 个 `DG_JIT_*` 变量会在构建时快照（生成 `envs.py`），与 `__init__.py` 第 7～13 行遥相呼应；
- 第 43～53 行：唯一源文件 + include 路径（`deep_gemm/include`、`third-party/deep-jit`、`magic_enum`、Ascend 与 torch_npu 头文件）；
- 第 86～92 行 `CppExtension(name='deep_gemm._C', ...)`；
- 第 119～127 行 `.pyi` 桩生成：`generate_pyi_file(root='./csrc/apis')`——**IDE 类型提示的源头就是 `csrc/apis/` 的 binding**。

### 站点 3：`csrc/python_api.cpp`（42 行，全文精读）

- 第 1 行的 include 顺序注释（HCCL 头先于 torch_npu）；
- 第 21～23 行 `get_current_npu_stream()`——torch_npu 流桥接，后面 launch 都用它；
- 第 25～27 行 `TORCH_EXTENSION_NAME` 兜底；
- 第 30～41 行九个 `register_apis` 调用，对照 3.2 节表格；`deep_jit::register_python_api(m, deep_gemm::jit)` 注册的是 JIT 子系统的通用接口。

### 站点 4：`csrc/apis/gemm.hpp`（680 行，今天的主战场）

按"binding → 模板 → 辅助函数"顺序读：

1. 第 552～678 行 `register_apis`：先只看 `bf16_gemm_nn`（第 638～643 行）与 `fp8_fp4_gemm_nn`（第 562～569 行）两个 `m.def`，核对每个 `py::arg` 与默认值；再看第 624～629 行的 `m.attr` 属性别名；其余 def 今天扫一眼即可；
2. 第 413～453 行：`bf16_gemm` 的 doc 注释（形状约定）+ 模板本体五步 + 四个别名（第 450～453 行）；
3. 第 199～256 行：`fp8_fp4_gemm` 的 doc 注释（**仓库内最详细的接口文档**，含 SF 形状与 ue8m0 约定）+ 模板本体 + 别名；
4. 第 22～43 行：`dispatch_fp8_fp4_gemm` 与 `expected_k_from_ks`；
5. 第 45～58 行：`get_gemm_desc` 顶部的形状表注释（Normal/MGrouped/KGrouped/Batched 四类，`(*)` 处解释 KGrouped 为何要"假转置"——只为了拿 shape/stride 信息，不搬数据，与 3.3 节同一思想）；
6. 第 60～197 行 `get_gemm_desc`：今天只跟 Normal 路径，重点读第 81～97 行的 shape 提取 switch、第 106～123 行 `c` 的三种语义、第 164～196 行 `GemmDesc` 的构造列表——数一数你能在源码里指认几个字段。

### 站点 5：`csrc/utils/layout.hpp`（221 行，"逻辑布局"工具箱）

- 第 88～99 行 `get_transposed`：`if constexpr` 三分支（Tensor / pair / 禁用）——低精度接口的 `(数据, SF)` 同步转置就靠它；
- 第 37～52 行 `major_check` / `get_major_type_ab` / `check_major_type_cd`：**major 不是用户声明的，是从 stride 推断的**——同一个 `nn` 接口，行主序的 B 转置后是 MN-major，`t()` 出来的 B 转置后是 K-major，两种都合法；
- 第 101～138 行 `str` / `ShapeSpec` / `operator""_sp`：`"mk"_sp = a` 这类"字符串字面量绑定张量"的常量魔法，读个大概即可；
- 第 140～219 行 `get_shape_by_spec`：看 `get_one_dim`（第 169～212 行）如何跨张量交叉校验同名维度、FP4 打包时 major 轴乘 2（第 185～187 行）、冲突时的 `DJ_PANIC` 消息格式（第 196～203 行）；
- 第 54～86 行 `gemm_early_return`：`m/n=0` 与 `k=0` 的空问题处理。

### 站点 6：`csrc/jit_kernels/epilogue_class.hpp`（130 行）与 `csrc/apis/epilogue_class.hpp`（23 行）

- `EpilogueClass` 基类（第 14～27 行）的三个虚函数：`get_epilogue_operator_type()`（给 JIT 代码生成提供设备侧算子名）、`make_epilogue_operator_args()`（运行期参数，如 alpha 值、bias 指针）、`check()`（输出 dtype 校验）；
- 四个派生类（第 29～107 行）：`Identity` / `Alpha` / `Bias`（注意 `BiasEpilogue` 不在 Python 侧注册——它由 `get_gemm_desc` 在识别出 `[1,N]` bias 时内部构造，第 122～123 行）/ `BF16StochasticRounding` / `FP8Quantization`；
- `resolve_epilogue_class`（第 110～128 行）：互斥断言 + 缺省 Identity + `check`；
- `csrc/apis/epilogue_class.hpp` 第 10～21 行：`m.def_submodule("epilogue", ...)` 注册四个类——`deep_gemm.epilogue.Alpha(0.5)` 就从这里来。

### 站点 7：`csrc/apis/runtime.hpp`（82 行）与 `csrc/jit/jit.hpp`（96 行）

- `runtime.hpp` 第 41～44 行：`_C.init` 的 binding；第 46～54 行 `set_dry_run`/`get_dry_run`（注意 `set_dry_run(True)` 会 `jit.get()` 强制创建 JIT 运行时——第 9 天测试的并行编译用到过）；第 13～39 行是 `__init__.py` 第 26～36 行那批 config 函数的来源；
- `jit.hpp` 第 76 行 `runtime` 的 LazyInit；第 78～92 行 `init_jit`：包根路径 → `include/` 搜索路径、`"DG"` 前缀、版本号——**JIT 编译 kernel 模板时找的就是随包分发的 `deep_gemm/include/`**。

### 站点 8：消费者视角回看 `tests/common.py`（只读第 49～85 行）

`gemm_dg` 是接口的"官方用法示范"：第 51 行拼接口名、第 53～58 行组装 `c`（bias 直接传 `[1,N]`；累加先 `d.copy_(c)` 再传 `c=d`）、第 64～65 行按 `trans_ab` 决定 A/B 是否预先转置、第 66 行 `getattr` + `**extra_args`。读完它，3.6 节的参数映射表就有了活例证。

**阅读检查点**（能回答再进入实践）：

1. `deep_gemm.bf16_gemm_nn` 在 Python 侧能找到源码吗？它的 `m.def` 在哪个文件第几行？
2. `bf16_gemm<false, true>` 的两个布尔分别控制什么？`nn` 的用户 B 形状是什么？
3. `get_gemm_desc` 里 `c` 与 `d` 同指针、`c` 为独立 `[1,N]`、`c=None` 三种情况各发生什么？
4. `_C.init` 之后 NPU 初始化了吗？`runtime`/`jit` 两个全局对象何时才真正构造？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：验证接口的"出身"（约 10 分钟）**。在仓库根目录启动 `python`：

```python
import deep_gemm, inspect
print(deep_gemm.__version__)                    # 0.1.1（__init__.py 第 1 行）
print(type(deep_gemm.bf16_gemm_nn))             # 内置函数，不是 Python 函数
print(deep_gemm.bf16_gemm_nn.__module__)        # deep_gemm._C —— 定义在 _C 扩展里
try:
    inspect.getsource(deep_gemm.bf16_gemm_nn)   # 预期失败：无 Python 源码
except (OSError, TypeError) as e:
    print('no python source:', e)
print(deep_gemm.bf16_gemm_nn.__doc__)           # pybind11 按 py::arg 生成的签名文档
print(deep_gemm.fp8_gemm_nn is deep_gemm.fp8_fp4_gemm_nn)   # True：属性别名
```

对照 `csrc/apis/gemm.hpp` 第 638～643 行核对 `__doc__` 里出现的每个参数名和默认值。

**步骤 2：跑通一个最小 `nn` 调用并验证三种 `c` 语义（约 20 分钟）**。

```python
import torch, torch_npu  # noqa
import deep_gemm

M, N, K = 128, 64, 256
a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')   # A [M, K]
b = torch.randn(K, N, dtype=torch.bfloat16, device='npu')   # B [K, N]（nn 后缀）
d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')   # D [M, N]

deep_gemm.bf16_gemm_nn(a, b, d)                 # ① c=None
torch.npu.synchronize()
ref = (a.float() @ b.float()).to(torch.bfloat16)
print('no-c max diff:', (d.float() - ref.float()).abs().max().item())

bias = torch.randn(1, N, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nn(a, b, d, c=bias)         # ② c=[1,N] → bias
torch.npu.synchronize()
print('bias max diff:', (d.float() - (ref.float() + bias.float())).abs().max().item())

d.copy_(ref)                                     # ③ c=d 同指针 → 累加
deep_gemm.bf16_gemm_nn(a, b, d, c=d)
torch.npu.synchronize()
print('acc max diff:', (d.float() - 2 * ref.float()).abs().max().item())
```

预期三次误差都在 BF16 舍入量级（1e-2 以内）。接着体验 epilogue 与互斥约束：

```python
deep_gemm.bf16_gemm_nn(a, b, d, alpha=0.5)                  # AlphaEpilogue
torch.npu.synchronize()
print('alpha max diff:', (d.float() - 0.5 * ref.float()).abs().max().item())

deep_gemm.bf16_gemm_nn(a, b, d, epilogue=deep_gemm.epilogue.Identity())  # 显式 epilogue

try:
    deep_gemm.bf16_gemm_nn(a, b, d, c=bias, alpha=0.5)      # 预期 RuntimeError
except RuntimeError as e:
    print('rejected:', 'alpha and bias cannot be used together' in str(e))
```

最后一条与 `tests/common.py` 第 191～198 行的拒绝测试相互印证：异常消息来自 `csrc/apis/gemm.hpp` 第 115～117 行。

**步骤 3：观察 major 由 stride 决定（约 10 分钟）**。

```python
b_row = torch.randn(K, N, dtype=torch.bfloat16, device='npu')   # 行主序 [K,N]
b_col = torch.randn(N, K, dtype=torch.bfloat16, device='npu').t()  # 逻辑转置后 shape [K,N]
print(b_row.shape, b_row.stride(), b_col.shape, b_col.stride())
# 两者都喂给 nn 接口，binding 内部都变成 [N,K]，但 major 不同（MN-major vs K-major），都能正确计算：
for b_ in (b_row, b_col):
    deep_gemm.bf16_gemm_nn(a, b_, d)
    torch.npu.synchronize()
    print('ok, max diff:', (d.float() - (a.float() @ b_.float())).abs().max().item())
```

把两组 `shape/stride` 记下来，对应 `get_major_type_ab` 的两种输出。

**步骤 4（可选）：不占 NPU 存储地走一遍链路（约 10 分钟）**。用 meta 张量 + dry_run 触发"只编译不加载"（即第 9 天并行预编译的同款路径，`csrc/jit_kernels/bf16_gemm.hpp` 第 67～70 行分支）：

```python
a_m = torch.randn(M, K, dtype=torch.bfloat16, device='meta')
b_m = torch.randn(K, N, dtype=torch.bfloat16, device='meta')
d_m = torch.empty(M, N, dtype=torch.bfloat16, device='meta')
deep_gemm._C.set_dry_run(True)
deep_gemm.bf16_gemm_nn(a_m, b_m, d_m)    # 走完整 host 链路，JIT 只编译不 launch
deep_gemm._C.set_dry_run(False)
```

若想看 JIT 生成的代码，可先 `export DG_JIT_PRINT_COMPILER_COMMAND=1` 再执行（README 第 93 行）。

### B. 无 Ascend 环境：静态替代任务

**B-1：接口清单盘点（约 15 分钟）**。只用 grep 完成下面三张小表（命令供参考）：

```bash
grep -n "from ._C import\|from .mega import\|from .tilelang_ops import" DeepGEMM-Ascend/deep_gemm/__init__.py
grep -n "m.def(\"bf16_gemm_nn\"\|m.def(\"fp8_fp4_gemm_nn\"\|m.attr(\"fp8_gemm_nn\")" DeepGEMM-Ascend/csrc/apis/gemm.hpp
grep -n "register_apis" DeepGEMM-Ascend/csrc/python_api.cpp
```

1. `__init__.py` 中哪些名字是直接 re-export、哪些是 Python 包装、哪些来自子模块？
2. `bf16_gemm_nn` 与 `fp8_fp4_gemm_nn` 的 `m.def` 各在第几行？默认值列表分别是什么？
3. `python_api.cpp` 的每个 `register_apis` 对应 `csrc/apis/` 里哪个头文件？

**B-2：手工演算 `get_shape_by_spec` 与 major 判定（约 20 分钟）**。假设用户执行：

```python
a = torch.randn(128, 256)          # shape (128,256), strides (256,1)
b = torch.randn(256, 64)           # shape (256,64),  strides (64,1)
d = torch.empty(128, 64)           # shape (128,64),  strides (64,1)
deep_gemm.bf16_gemm_nn(a, b, d)    # dtype 都当 bf16
```

在纸上依次回答（不看答案）：

1. `get_transposed<true>(b, 0, 1)` 之后 b 的 shape 和 stride 是多少？（答案：`(64, 256)` / `(1, 64)`）
2. `get_shape_by_spec<"mnk">("mk"=a, "nk"=b, "mn"=d)` 逐字母推导 m/n/k，每个字母写出它来自哪些张量的哪些轴。（答案：m：a[0]=128 与 d[0]=128；n：b[0]=64 与 d[1]=64；k：a[1]=256 与 b[1]=256）
3. `major_a`、`major_b` 各是什么？`outer_stride_a/b/d` 各是多少？（答案：a 的 `stride(-1)=1` → `Major::K`；b 的 `stride(-1)=64≠1`、`stride(-2)=1` → `Major::MN`；outer stride 分别为 256、64、64）
4. 若把 d 换成 `(128, 63)`，会发生什么？异常消息的格式长什么样？（答案：`n` 冲突——b 给 64、d 给 63，`DJ_PANIC("Conflicting shape dimension 'n' ...")`，见 `utils/layout.hpp` 第 195～203 行）
5. 若 b 的两个 stride 都是 1（例如广播后的特殊视图），会触发哪条检查？（答案：`major_check` 的 "Ambiguous major axis" panic，第 38～40 行）

**B-3：四后缀规则表（约 15 分钟）**。合上教程，凭 `csrc/apis/gemm.hpp` 第 450～453 行的四个别名和第 418～419 行 doc 注释，重写 3.3 节的表格（每个后缀：数学含义、用户 A/B 形状、模板参数、binding 是否转置），再用 `tests/common.py` 第 64～65 行检验。

**B-4：参数映射表 + 一次 bias 调用的完整追踪（约 25 分钟）**。

1. 为 `bf16_gemm_nn` 与 `fp8_fp4_gemm_nn` 各写一张"Python 实参 → C++ 形参 → 默认值"表（来源：binding 行，3.6 节）；
2. 追踪 `tests/common.py` 第 53～54 行的 bias 路径：测试的 `c` 形状是 `[1, N]`（`generators.py` 第 306 行）→ `extra_args['c'] = c` → C++ 里 `with_bias=true` → 换成 `BiasEpilogue`（`apis/gemm.hpp` 第 122～123 行）→ `GemmDesc.acc=false`。写出这条链上每一步的文件与行号；
3. 再追踪累加路径：`d.copy_(c)` 后 `c=d` → `is_cd_same=true` → `acc=true`（第 106～107、180 行）。两条路径各产出一个"输入 → GemmDesc 关键字段"的小表。

**B-5：闭卷画调用链（约 15 分钟）**。不看 3.7 节，独立画出 `deep_gemm.bf16_gemm_nn` 从 Python 到 `launch_bf16_gemm` 的完整链路，标注每一站的文件、函数、行号和职责，然后对照订正——这张图就是今天的产出，誊抄进第六节模板。

## 六、当天产出

学习计划要求今天的产出是 **`deep_gemm.bf16_gemm_nn` 的入口调用链**。按下述模板填写（B-2～B-5 的演算过程附在后面）：

```text
# DeepGEMM-Ascend 第 10 天记录：追踪 Python 公开接口

## A. 入口调用链（闭卷画出后对照订正）
deep_gemm.bf16_gemm_nn(a, b, d, ...)
  → ＿＿＿（re-export，__init__.py 第 ＿ 行）
  → ＿＿＿（PYBIND11_MODULE，python_api.cpp 第 ＿ 行 → register_apis）
  → ＿＿＿（m.def，apis/gemm.hpp 第 ＿ 行；参数名/默认值在此定死）
  → ＿＿＿（模板实例 bf16_gemm<＿＿>，第 ＿ 行）
      ① ＿＿＿  ② ＿＿＿  ③ ＿＿＿  ④ ＿＿＿  ⑤ ＿＿＿
  → launch_bf16_gemm（jit_kernels/bf16_gemm.hpp 第 ＿ 行，第 11～12 天继续）

## B. 接口盘点结论
- 直接 re-export 的接口数：＿＿；Python 包装的接口：＿＿＿＿（为什么它需要 Python 层）
- fp8_gemm_nn 与 fp8_fp4_gemm_nn 的关系：＿＿＿＿；cublaslt_gemm_nn 的真实身份：＿＿＿＿

## C. 手工演算记录（B-2～B-4）
- shape/stride → (m, n, k)、major、outer stride 的推导：＿＿＿
- 四后缀规则表：＿＿＿
- bias 与累加两条路径的字段表：＿＿＿

## D. 实验记录（有环境时）
- 最小 nn 调用三种 c 的最大误差：＿＿ / ＿＿ / ＿＿
- alpha+bias 拒绝消息验证：是/否；major 观察记录：＿＿＿
- dry_run + meta 张量现象：＿＿＿

## E. 遗留问题（交给第 11～12 天）
- GemmDesc 里 ＿＿ 字段还不明白用途
- launch_bf16_gemm 里的配置与代码生成待展开
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `deep_gemm.bf16_gemm_nn` 有 Python 源码吗？给出它从符号到 C++ 函数的完整"身份链"。
2. `csrc/python_api.cpp` 里有哪些 `register_apis`？为什么这个文件可以只有 42 行？
3. `bf16_gemm_nn = bf16_gemm<false, true>`：第二个 `true` 对应后缀的哪个字母？binding 内部因此对 B 做了什么？发生数据搬运吗？
4. 四个后缀的用户 A/B 形状各是什么？`tn` 的数学含义是什么？
5. `get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d)` 中 `"mk"` 等标注的含义是什么？某个维度冲突时代码走到哪一行、Python 侧看到什么？
6. `c` 的三种传法（`None` / 与 `d` 同指针 / 独立 `[1,N]`）分别产生什么效果？`GemmDesc.acc` 各是多少？为什么 alpha 不能与 bias 同用？
7. `resolve_epilogue_class` 的"三选一"指什么？什么都不传时得到什么对象？
8. `fp8_fp4_gemm_nn` 的 `a` 参数在 Python 和 C++ 侧分别是什么结构？`dispatch_fp8_fp4_gemm` 依据什么选择 kernel？`fp8_gemm_nn` 是独立函数吗？
9. `major_a`/`major_b` 是用户指定的吗？由什么决定？同一个 `[K,N]` 的 B，行主序和 `t()` 出来的两种张量经过 `nn` 接口后 major 相同吗？
10. `_C.init(__version__, get_library_root_path())` 做了什么？为什么 import 阶段不初始化 NPU？`runtime`/`jit` 何时才真正构造？
11.（思考题）接口为什么在 Python 层完全不做形状检查，而把断言放到 C++ 的 `get_gemm_desc` 里？这种"薄 Python、厚 binding"的设计有什么代价与好处？

**参考答案：**

1. 没有。身份链：`__init__.py` 第 64 行 `from ._C import bf16_gemm_nn`（re-export）→ `_C` 扩展由 `csrc/python_api.cpp` 编译（`setup.py` 第 43、86 行）→ 第 35 行 `gemm_api::register_apis` → `csrc/apis/gemm.hpp` 第 638 行 `m.def("bf16_gemm_nn", bf16_gemm_nn, ...)` → 函数本体是第 452 行的模板实例 `bf16_gemm<false, true>`（第 424 行定义）。
2. 九个：config（runtime.hpp）、epilogue_class、gemm_api、attention_api、einsum、layout、aclnn_api、mega_moe，另有 `deep_jit::register_python_api`。它短是因为**所有接口都定义在 `csrc/apis/*.hpp` 内**，每个头文件自带 `register_apis`，`python_api.cpp` 只负责接线（加上流桥接与 HCCL include 顺序处理），新增接口不需要改这个文件。
3. 对应 `nn` 的第二个字母 `n`（数学上 B 原样参与 `D = A × B`）。用户传 B `[K,N]`，binding 调 `b.transpose(0,1)` 得到规范形式 `B_eff[N,K]`。`transpose` 只交换 shape/stride 元数据，零拷贝（`utils/layout.hpp` 第 88～99 行）。
4. 见 3.3 节表格：`nt`：A `[M,K]`、B `[N,K]`；`nn`：A `[M,K]`、B `[K,N]`；`tn`：A `[K,M]`、B `[K,N]`；`tt`：A `[K,M]`、B `[N,K]`。`tn` 表示 `D = A^T @ B`。
5. 标注声明"该张量的第 0/1 维分别是什么逻辑维度"，供同名维度交叉校验：A 标 "mk"（第 0 维是 m、第 1 维是 k）、B 标 "nk"、D 标 "mn"。冲突时走 `get_one_dim` 的 `DJ_PANIC("Conflicting shape dimension ...")`（`utils/layout.hpp` 第 195～203 行），Python 侧抛 `RuntimeError`。另外若某张量没有任何 stride=1 的轴，第 158 行报 "doesn't have major axis"。
6. `None`：无累加无 bias，直接算 `D = A@B^T`；与 `d` 同指针同形：`is_cd_same=true` → `acc=true`，语义 `D += A@B^T`（`k=0` 时早退且不清零 D）；独立连续 `[1,N]`：`with_bias=true` → 包装成 `BiasEpilogue`，`acc=false`（`apis/gemm.hpp` 第 106～123、180 行）。alpha 与 bias 互斥因为第 115～117 行断言 `AlphaEpilogue` 与 bias 不能共存——测试 `tests/common.py` 第 191～198 行专门验证该 `RuntimeError`。
7. `epilogue_class`、`alpha`、`sfd`（FP8 输出量化）三者互斥，至多一个（第 117 行断言）；`alpha` → `AlphaEpilogue`；都没传 → `IdentityEpilogue`（`jit_kernels/epilogue_class.hpp` 第 110～128 行），最后统一 `check(c, d)`。
8. Python 侧是 `(数据张量, SF 张量)` 二元组，C++ 侧是 `const std::pair<torch::Tensor, torch::Tensor>&`。`dispatch_fp8_fp4_gemm`（`apis/gemm.hpp` 第 22～33 行）按 `(a_dtype, b_dtype)` 分派：fp8×fp8 → `launch_fp8_gemm`、fp4×fp4 → `launch_fp4_gemm`、fp8×fp4 → `launch_fp8_dequant_gemm`。`fp8_gemm_nn` 不是独立函数，是第 625 行的属性别名。
9. 不是用户指定的，由 `get_major_type_ab` 从 stride 推断：`stride(-1)==1 → Major::K`，`stride(-2)==1 → Major::MN`。两种 B 转置后的 major 不同：行主序 `[K,N]`（strides `(N,1)`）→ 转置后 `stride(-2)=1` 是 MN-major；`t()` 出来的 `[K,N]`（strides `(1,K)`）→ 转置后 `stride(-1)=1` 是 K-major。两种都合法，kernel 按 `major_b` 走不同寻址。
10. `init` → `init_jit`（`csrc/jit/jit.hpp` 第 78～92 行）：登记版本号、包根路径与 `include/` 头文件搜索路径，为 JIT 准备配置；`runtime`/`jit` 都是 `LazyInit`（第 76 行），首次使用才构造。import 与 init 都不触碰 NPU 是为了 import 后还能 fork（`runtime.hpp` 第 42 行注释）——测试的并行预编译依赖这一点。
11. 开放题，参考方向：好处——调用链最短、类型/默认值/断言集中在一处（`m.def` + `get_gemm_desc`），Python/C++ 行为不会漂移；37 个接口靠 `getattr` 字符串拼接即可统一驱动（`tests/common.py` 第 66 行）；metadata-only 的 meta 张量路径也能复用同一 C++ 校验。代价——错误在运行到 C++ 时才暴露，IDE 只能靠生成的 `.pyi` 提供提示；因此仓库用 `setup.py` 生成类型桩弥补（第 120 行）。
