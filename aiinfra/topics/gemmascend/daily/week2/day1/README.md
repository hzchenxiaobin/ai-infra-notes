# 第 2 周 · 第 1 天（总第 8 天）：阅读项目说明并建立代码地图

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 2 周"项目结构与 JIT 调用链"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 50 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

第 1 周我们建立了 GEMM 与 Ascend 硬件的概念底座；从本周开始进入真实仓库。本周的主线是**从 Python 调用一路追到 Device Kernel**：今天先建立整体地图（结构），第 9 天跑通最小测试，第 10～13 天再逐层深入调用链。今天**不深读任何一段实现**，只回答三个问题：这个项目是什么？代码放在哪里、分几层？支持哪些算子？

## 一、学习目标

完成今天的学习后，你应当能够：

1. 用两三句话说出 DeepGEMM-Ascend 的项目定位，以及它与上游 DeepGEMM 的关系；
2. 画出仓库顶层目录树，说出 `deep_gemm/`、`csrc/`、`tests/`、`scripts/`、`third-party/` 各自的职责；
3. 区分 **Python API、C++ Host、JIT 启动、Device Kernel、测试**五类代码，每类各举出至少一个真实文件；
4. 列出项目支持的算子家族——Dense GEMM、Grouped GEMM、MQA Logits、MegaMoE、HC Prenorm——以及各自的 Python 入口和源码落点；
5. 说出构建方式的分工（`develop.sh`、`setup.py`、`CMakeLists.txt`），并解释"device kernel 为什么是头文件、为什么随 Python 包分发"。

**当天产出：一张仓库目录与职责对照表**（模板见第六节）。

## 二、预备知识

- **第 1 周内容**：第 1 天（M/N/K 与 shape）、第 5 天（GM/L1/L0A/L0B/L0C/UB 分层、AIC 与 AIV 分工、MTE 流水线）与今天关系最大；第 2～4 天（布局、Tiling、数值类型）有助于理解目录里 `scheduler`、`epilogue`、`transform_sf` 这些命名的含义。忘了也没关系，遇到术语回查即可。
- **C++ 阅读能力**：能看懂模板声明（`template <...> void f(...)`）和 `#include` 即可，不需要会写。pybind11 只需知道它是"C++ 函数注册给 Python 调用"的桥梁。
- **Python**：`import` / re-export 机制。
- **git 子模块（submodule）概念**：仓库依赖几个放在 `third-party/` 下的外部项目。
- **不需要 Ascend 硬件**：今天全部内容都是读 README 和静态浏览代码。有环境时的可选项见实践步骤第 6 步。

## 三、详细讲解

### 3.1 项目定位：从 README 读起

仓库 `README.md` 开头（第 3 行英文、第 9 行中文）给出了定位：

> DeepGEMM Ascend 是华为昇腾平台上的 DeepGEMM 实现。它完全兼容 DeepGEMM 的 API，支持 BF16、FP8、FP4 GEMM、MQA logits 和 MegaMoE 算子。DeepGEMM Ascend 和 DeepGEMM 使用相同的包名，用户只需安装当前包，即可沿用其他平台上 DeepGEMM 的 API 和开发流程。

提炼出三个关键事实：

| 关键事实 | 说明 | 出处 |
| :--- | :--- | :--- |
| API 兼容移植 | 与上游 DeepGEMM 同包名 `deep_gemm`，接口语义一致 | README 第 3、9 行 |
| 算子覆盖 | BF16/FP8/FP4 GEMM、（Paged/Sparse）MQA logits、MegaMoE，另有 HC Prenorm（见 Performance 小节） | README 第 3 行、第 175～185 行 |
| 目标硬件 | Ascend 950 系列；2026.09.30 首次发布 | README 第 17 行（News） |

**Requirements（第 21～29 行）**值得逐条记录，它解释了仓库里许多设计的来历：

| 依赖 | 仓库中的落点 |
| :--- | :--- |
| HUAWEI Ascend NPU（950 系列验证） | 硬件容量常量写在 `deep_gemm/include/deep_gemm/common.hpp`（第 50～61 行） |
| CANN 9.20，提供 `bin/bisheng`、`bin/ld.lld` | `bisheng` 是昇腾设备端编译器，JIT 运行时用它把 device kernel 编译成可加载产物 |
| `torch_npu` | Python 侧 NPU 后端；`setup.py` 链接 `torch_npu` 库（第 53 行） |
| Python ≥ 3.10 | — |
| C++20 `<format>` 支持 | Host 代码大量使用 `std::format`（如 `csrc/jit_kernels/bf16_gemm.hpp` 第 18 行） |
| `tilelang` | HC Prenorm kernel 用 TileLang 编写（`deep_gemm/tilelang_ops/hc_prenorm.py`），并声明为包依赖（`setup.py` 第 149 行） |
| `tree-sitter` / `tree-sitter-cpp` | 从源码生成 Python 类型桩 `_C.pyi`（`scripts/generate_pyi.py`） |

**Interfaces（第 49～79 行）**分三块：

1. **Kernel 接口**：文档直接指向上游 DeepGEMM 的接口说明（README 第 53 行），本仓库只补充 Ascend 特有差异——最重要的一条是 Scaling Factor 布局（第 56 行）：沿 K 维的每对 UE8M0 缩放因子打包成一个 `int16`，且按 MN-major 顺序存放。另外（第 58～60 行）：Dense GEMM 的 `c` 既可以是与 `d` 同形状的累加矩阵（用原子加实现 `D += A@B`），也可以是 `[1, N]` 的 contiguous bias（在 AIV epilogue 中、输出转换之前加上）。示例调用（第 63～65 行）：`deep_gemm.bf16_gemm_nn(a, b, c=bias, d=d)`。
2. **工具函数**（第 71～79 行）：`set_num_sms`/`get_num_sms`（可用 AI 核数）、`set_npu_arch`/`get_npu_arch`、M/K 对齐查询与设置、`use_deterministic_algorithms`、`transform_sf_into_required_layout`、`transform_k_grouped_sf_into_required_layout`、`get_paged_mqa_logits_metadata`，以及一组 `aclnn_*_gemm_*` 参考实现（测试中用于交叉验证）。
3. **环境变量**（第 81～98 行）：`ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME` 定位 CANN 工具包；`DG_JIT_DEBUG`、`DG_PRINT_CONFIGS`、`DG_JIT_CACHE_DIR`（JIT 编译缓存目录，默认 `$HOME/.dj`）、`DG_JIT_PRINT_COMPILER_COMMAND`、`DG_JIT_LAUNCH_TIMEOUT`、`ASCEND_LAUNCH_BLOCKING` 等。今天只需记住"有这批开关"，第 51 天会真正用它们。

**Performance（第 100～185 行）**是 README 用数据自证的地方：Dense GEMM 在 `M=4096, N=7168, K=16384` 下达到硬件极限的 98.3%～99.8%（按 FP4/FP8×FP4/FP8×FP8/BF16 类型，第 106～113 行）；MQA Logits 是 FIX-pipe 受限、占用率 99%（第 151 行）；HC Prenorm 接近写满 HBM 带宽（第 177 行）。测试环境为 Ascend 950DT + CANN 9.20、`bench_msprof`、冷 L2（第 102 行）。这些表同时是一份"仓库支持哪些算子"的清单——每个性能小节都对应一组源码和测试。

### 3.2 仓库顶层布局

```text
DeepGEMM-Ascend/
├── README.md                  # 项目说明（本日主读）
├── LICENSE / .clang-tidy / .gitignore / .gitmodules
├── setup.py                   # 构建脚本：把 csrc/python_api.cpp 编译成 deep_gemm._C 扩展
├── develop.sh                 # 开发构建：setup.py build + 把 .so 与 _C.pyi 符号链接进 deep_gemm/
├── build.sh                   # 打包构建：python3 setup.py bdist_wheel
├── CMakeLists.txt             # 仅供 IDE 索引调试，不参与正式构建（见文件第 1 行注释）
│
├── csrc/                      # ② C++ Host 代码 + ③ JIT 启动层
│   ├── python_api.cpp         #   唯一被编译的翻译单元；pybind11 模块入口
│   ├── apis/                  #   按功能域拆分的 8 个 Python API 实现头
│   ├── jit_kernels/           #   JIT 启动层：选配置 → 生成代码 → compile/launch
│   ├── jit/jit.hpp            #   DeepJIT 运行时的封装与全局 Runtime 状态
│   ├── utils/                 #   形状/布局/dtype 检查等 Host 工具
│   ├── backends/hccl.hpp      #   HCCL 对称缓冲（MegaMoE 通信）
│   └── indexing/main.cpp      #   仅供 CLion 索引 device 头文件（非生产代码）
│
├── deep_gemm/                 # ① Python 包 + ④ Device Kernel 模板
│   ├── __init__.py            #   公开 API 汇总 re-export
│   ├── include/deep_gemm/     #   Device Kernel 模板头文件（JIT 编译时才被 include）
│   ├── mega/__init__.py       #   MegaMoE Python 侧（SymmBuffer、权重转换、入口函数）
│   ├── tilelang_ops/          #   TileLang 实现（HC Prenorm）
│   ├── testing/               #   基准与测试辅助库（bench_msprof 等，随包分发）
│   └── locality_domain.py     #   CUDA 兼容的 no-op 桩（当前恒返回单一逻辑域）
│
├── tests/                     # ⑤ 测试代码（正确性 + 性能）
├── scripts/                   # generate_pyi.py；profiling/msprof_mega_moe.py
└── third-party/               # git 子模块：deep-jit、clangd-ascend、magic_enum；tilelang_ops
```

两个容易误判的点，先标记出来：

- **`deep_gemm/` 身兼两职**。它既是 Python 包（`.py` 文件 + 编译出的 `_C` 扩展），又通过 `deep_gemm/include/` 存放 device kernel 模板。`setup.py` 第 150～153 行把 `'include/**/*'` 声明为 package_data 打进 wheel；`deep_gemm/__init__.py` 第 159 行的 `_C.init(__version__, get_library_root_path())` 把包目录传给 C++ 侧，`csrc/jit/jit.hpp` 第 80 行据此把 `<包目录>/include` 设为 JIT 编译的 include 目录。**kernel 源码是随包分发的，运行时按需编译**——这是本仓库最核心的架构决定（3.3 节展开）。
- **`CMakeLists.txt` 不参与正式构建**。文件第 1 行注释写明"this CMake is only for debugging/IDE indexing; for setup, please use Torch extension"。它还定义了一个 `indexing` 目标（`csrc/indexing/main.cpp`），因为 device 源码只被 JIT 生成的代码 include，普通 IDE 索引不到，需要一个真实翻译单元把它们挂上。

### 3.3 五类代码：本仓库的分层

学习计划要求区分的五类代码，在仓库里的落点如下：

| # | 代码类别 | 所在位置 | 代表文件 | 一句话职责 |
| :---: | :--- | :--- | :--- | :--- |
| ① | Python API | `deep_gemm/*.py` | `__init__.py`、`mega/__init__.py`、`tilelang_ops/hc_prenorm.py` | 对外公开接口：re-export `_C` 符号、少量手写包装 |
| ② | C++ Host | `csrc/` | `python_api.cpp`、`apis/gemm.hpp`、`utils/layout.hpp` | 接收 Python 参数，做检查、组装描述符，调用启动层 |
| ③ | JIT 启动 | `csrc/jit_kernels/`、`csrc/jit/jit.hpp` | `jit_kernels/bf16_gemm.hpp`、`jit/jit.hpp` | 选执行配置、生成模板实例化代码、编译并启动 |
| ④ | Device Kernel | `deep_gemm/include/deep_gemm/` | `bf16_gemm.hpp`、`scheduler.hpp`、`ascend/copy.hpp` | 真正在 NPU 上运行的模板代码（AIC/AIV） |
| ⑤ | 测试 | `tests/` | `test_bf16_gemm.py`、`common.py`、`generators.py` | 生成用例、交叉验证、测性能 |

各层之间的依赖方向：

```text
        Python 调用方（模型代码 / tests/）
                    │ import deep_gemm
        ① Python API ▼
deep_gemm/__init__.py（re-export）· mega/ · tilelang_ops/
                    │ deep_gemm._C（pybind11 扩展，由 setup.py 编译）
        ② C++ Host  ▼
csrc/python_api.cpp → csrc/apis/*（参数检查、构造 GemmDesc）
                    │
        ③ JIT 启动  ▼
csrc/jit_kernels/*（选配置、生成实例化代码）──依赖──▶ csrc/jit/jit.hpp
                    │ jit->compile(...) / jit->launch(...)
        ④ Device    ▼
deep_gemm/include/deep_gemm/*.hpp（模板 kernel，被 JIT 生成的代码 #include）
```

⑤ 测试代码横跨①～④：`tests/` 只通过 Python API 驱动全部能力，同时引用 `deep_gemm/testing/` 里的基准工具。

**为什么 device kernel 是头文件，而不是提前编译好的库？** 看一眼 `csrc/jit_kernels/bf16_gemm.hpp` 第 16～78 行的 `launch_bf16_gemm` 就能明白（今天只看骨架）：

```cpp
static void launch_bf16_gemm(const GemmDesc& desc) {
    auto config = select_gemm_config(desc);                       // ③ 选执行配置
    const auto code = std::format(R"(                              // ③ 生成一段 C++ 代码
#include <deep_gemm/bf16_gemm.hpp>                                 // ④ include kernel 模板
namespace deep_gemm {{
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&bf16_gemm_impl<              //    显式实例化模板
        ..., {}, {}, {}, ...>);
}}
}})", /* 二十多个模板实参：major、shape、block/mad、stage 数、... */);
    const auto kernel = jit->compile("bf16_gemm", code);           // ③ 编译（带缓存）
    jit->launch(kernel, config.launch_options, /* 运行时参数 */);  // ③ 启动
}
```

`bf16_gemm_impl` 是一个有二十多个模板参数的函数模板（`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 第 9～27 行），shape、分块、stage 数等都是**编译期常量**。组合空间太大，不可能预先编译全部实例；于是每次遇到新的（shape, 配置）组合，就现场生成上面的实例化代码、调用 CANN 的 `bisheng` 编译、按内容缓存（`DG_JIT_CACHE_DIR`，默认 `$HOME/.dj`），然后加载启动。这就是"JIT 调用链"四个字的含义——本周第 10～13 天会把这条链逐站走通，今天只需在地图上标注出这些站点。

另一个值得注意的共享点：`deep_gemm/include/deep_gemm/common.hpp` 第 13～15 行注释强调它是 host 与 device **共用**的头——`Major`、`GemmType` 枚举、`gm_ptr` 指针类型、片上容量常量都定义在这里，`csrc/`（host）和 kernel（device）都 include 它。所以"五类代码"不是五个孤立目录，②③④之间有公共定义。

### 3.4 算子能力地图

学习计划要求记录五大家族算子。下表把"Python 入口 → 各层文件 → 测试"串成一列，全部为仓库内真实路径：

| 算子家族 | Python 入口（`deep_gemm.*`） | ② C++ Host | ③ JIT 启动 | ④ Device Kernel | ⑤ 测试 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dense BF16 GEMM | `bf16_gemm_nt/nn/tn/tt` | `csrc/apis/gemm.hpp` | `csrc/jit_kernels/bf16_gemm.hpp` | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | `tests/test_bf16_gemm.py` |
| Dense FP8/FP4/混合 GEMM | `fp8_fp4_gemm_nt/nn/tn/tt`（`fp8_gemm_*` 为别名） | `csrc/apis/gemm.hpp`（`dispatch_fp8_fp4_gemm`，第 22～33 行） | `csrc/jit_kernels/fp8_gemm.hpp`（FP8×FP8、FP4×FP4）；`fp8_dequant_gemm.hpp`（FP8×FP4） | `fp8_gemm.hpp`；`fp8_dequant_gemm.hpp`（同目录） | `tests/test_fp8_gemm.py`、`test_fp4_gemm.py`、`test_fp8_dequant_gemm.py` |
| M/K-Grouped GEMM | `m_grouped_bf16_gemm_{nt,nn}_contiguous`、`m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous`、`k_grouped_bf16_gemm_tn_contiguous`、`k_grouped_fp8_gemm_{nt,tn}_contiguous` | `csrc/apis/gemm.hpp`（`get_gemm_desc` 的 MGrouped/KGrouped 分支） | 复用 dense 的 launch（同一 `launch_bf16_gemm`/`launch_fp8_gemm`，靠 `desc.gemm_type` 区分） | 复用 dense kernel（`GemmType` 作模板参数，见 `common.hpp` 第 87～92 行） | `tests/test_bf16_gemm.py` 选 `mgrouped`/`kgrouped`；`tests/generators.py` |
| MQA Logits | `fp8_fp4_mqa_logits`、`fp8_fp4_paged_mqa_logits`、`fp8_fp4_sparse_mqa_logits`、`fp8_fp4_paged_sparse_mqa_logits` 及配套 `get_*_metadata` | `csrc/apis/attention.hpp` | `csrc/jit_kernels/mqa_logits.hpp`、`sparse_mqa_logits.hpp`、`paged_sparse_mqa_logits.hpp` | `deep_gemm/include/deep_gemm/` 下同名三份 + `scheduler/`、`layout/`、`*_metadata.hpp` | `tests/test_attention.py` |
| MegaMoE | `mega` 子包：`fp8_fp4_mega_moe`、`SymmBuffer`、`transform_weights_for_mega_moe` | `csrc/apis/mega_moe.hpp`（+ `csrc/backends/hccl.hpp`） | `csrc/jit_kernels/mega_moe.hpp` | `deep_gemm/include/deep_gemm/mega_moe.hpp` + `scheduler/mega_moe.hpp`、`layout/mega_moe.hpp`、`comm/mega_moe.hpp`、`simd/mega_moe.hpp` | `tests/test_mega_moe.py`；`scripts/profiling/msprof_mega_moe.py` |
| HC Prenorm | `tf32_hc_prenorm_gemm` | —（不经 `_C`） | —（TileLang 自带编译流程） | `deep_gemm/tilelang_ops/hc_prenorm.py`（TileLang kernel，纯 Python 包内定义） | `tests/test_hyperconnection.py` |

配套的工具/辅助能力：

| 能力 | 入口 | 落点 |
| :--- | :--- | :--- |
| Scale 布局转换 | `transform_sf_into_required_layout`、`transform_k_grouped_sf_into_required_layout` | `csrc/apis/layout.hpp` → `csrc/jit_kernels/transform_sf.hpp` → `deep_gemm/include/deep_gemm/transform_sf.hpp`；测试 `tests/test_layout.py` |
| Einsum / 批量 GEMM | `einsum`、`fp8_einsum` | `csrc/apis/einsum.hpp`（`fp8_bmm`，复用 `gemm.hpp` 的 `get_gemm_desc`，`GemmType::Batched`）；测试 `tests/test_einsum.py` |
| ACLNN 参考实现 | `aclnn_bf16_gemm_{nt,nn,tn,tt}`、`aclnn_fp8_fp4_gemm_*`、`cublaslt_gemm_*`（别名） | `csrc/apis/aclnn.hpp`（直接调 CANN aclnn 算子库，不走 JIT）；在 `tests/common.py` 的 `gemm_aclnn`（第 87 行）中做交叉验证 |
| Epilogue 类 | `deep_gemm.epilogue.Identity/Alpha/FP8Quantization/BF16StochasticRounding` | `csrc/apis/epilogue_class.hpp`、`csrc/jit_kernels/epilogue_class.hpp` |
| 运行时配置 | `set_num_sms`/`get_num_sms`、`set_npu_arch`、`use_deterministic_algorithms`、`npu_sleep` | `csrc/apis/runtime.hpp` → `csrc/jit/jit.hpp` 的 `Runtime` 类 |
| Locality domain | `get_num_locality_domains` 等 | `deep_gemm/locality_domain.py`（CUDA 兼容 no-op 桩，当前恒为单一逻辑域） |

读表时注意三个"覆盖差异"，它们说明 API 兼容不等于全部实现：

- `fp8_gemm_nt` 等名字只是 `fp8_fp4_gemm_nt` 的别名（`csrc/apis/gemm.hpp` 第 624～629 行用 `m.attr` 注册）；
- `m_grouped_fp8_fp4_gemm_nt_masked` 在本后端**未实现**（`csrc/apis/gemm.hpp` 第 606、630 行注释 "not implemented in this backend"）；
- grouped 接口的转置组合不全：m-grouped 只有 nt/nn，k-grouped BF16 只有 tn，k-grouped FP8 有 nt/tn（同文件第 586～676 行的 `m.def` 列表）。

### 3.5 构建与开发方式

| 入口 | 做什么 | 关键细节 |
| :--- | :--- | :--- |
| `./develop.sh` | 开发构建 | 第 13 行 `DG_USE_LOCAL_VERSION=0 python3 setup.py build`；第 16～19 行把 `build/` 里找到的 `.so` 和 `../stubs/_C.pyi` 符号链接进 `deep_gemm/`，使包目录里的 Python 代码能直接 `from . import _C` |
| `./build.sh` 或 `pip install . --no-build-isolation` | 打包发布 | `setup.py bdist_wheel`；`CustomBuildPy` 还会生成 `envs.py`（固化构建时的 `DG_JIT_*` 默认值）、`_C.pyi`，并把 `csrc/` 拷进 wheel 便于错误定位 |
| `CMakeLists.txt` | 仅 IDE 索引 | `pybind11_add_module(_C csrc/python_api.cpp)` 与 `indexing` 目标（见 3.2 节） |

`setup.py` 里与本日地图相关的三个事实：被编译的源文件**只有一个**——`sources = ['csrc/python_api.cpp']`（第 43 行）；include 路径包含 `deep_gemm/include`、`third-party/deep-jit/include`、`third-party/magic_enum/include`（第 44～52 行）；链接 `ascendcl`、`hccl_fwk`、`torch_npu`、`opapi` 等（第 53 行）。

`third-party/` 由 `.gitmodules` 声明三个子模块：**deep-jit**（JIT 运行时，`csrc/jit/jit.hpp` include 的 `deep_jit/*` 头来自它）、**clangd-ascend**（让 clangd 能理解昇腾设备端语法，服务 IDE）、**magic_enum**（枚举转字符串等工具）。另有 `third-party/tilelang_ops/`（SwiGLU 等辅助）。克隆时需要 `--recursive`（README Quick Start 第 35 行）；如果本地子模块目录是空的，说明没有递归克隆，仓库自身的阅读不受影响，只是读不到 DeepJIT 内部实现。

## 四、源码导读

今天的导读目标是"**验证并充实第三节的地图**"，每站停留 5～10 分钟，只看骨架不看实现细节。全程静态阅读、无需硬件。

### 站点 1：Python 包入口 —— `deep_gemm/__init__.py`（159 行）

通读一遍，注意它由四类内容构成：

1. **纯 re-export**：第 26～81 行从 `._C` 导入全部接口，分组注释清晰（FP8 FP4 GEMMs / FP8 GEMMs / ACLNN reference / BF16 GEMMs / Einsum / Attention / Layout）；第 39～42 行的 `cublaslt_gemm_*` 注明是"Upstream cuBLASLt API aliases backed by ACLNN"。
2. **手写包装函数**：`get_paged_sparse_mqa_logits_metadata`（第 87 行）与 `fp8_fp4_paged_sparse_mqa_logits`（第 115 行）。读第 83～85 行的模块级注释：分页稀疏 MQA 的配置与 FIRST/REUSE（KV 快照首次缓存、后续复用）状态必须由公开入口统一管理，所以这两个函数留在 Python 层而不是直接暴露 `_C` 绑定。
3. **子包**：`mega`（第 137～141 行：`SymmBuffer`、`transform_weights_for_mega_moe`、`fp8_fp4_mega_moe`）、`tilelang_ops`（第 157 行：`tf32_hc_prenorm_gemm`）、`locality_domain`（第 147～154 行）。
4. **初始化**：第 159 行 `_C.init(__version__, get_library_root_path())`——把版本号和包根路径传进 C++，供 JIT 定位 `include/`（呼应 3.2 节"kernel 随包分发"）。

### 站点 2：pybind11 总入口 —— `csrc/python_api.cpp`（42 行）

整个仓库唯一被编译的 `.cpp`。第 30 行 `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 之内只有 9 行注册（第 33～41 行），每行对应 `csrc/apis/` 的一个头文件：

```cpp
deep_gemm::config::register_apis(m);         // ← csrc/apis/runtime.hpp
deep_gemm::epilogue_class::register_apis(m); // ← csrc/apis/epilogue_class.hpp
deep_gemm::gemm_api::register_apis(m);       // ← csrc/apis/gemm.hpp
deep_gemm::attention_api::register_apis(m);  // ← csrc/apis/attention.hpp
deep_gemm::einsum::register_apis(m);         // ← csrc/apis/einsum.hpp
deep_gemm::layout::register_apis(m);         // ← csrc/apis/layout.hpp
deep_gemm::aclnn_api::register_apis(m);      // ← csrc/apis/aclnn.hpp
deep_gemm::mega_moe::register_apis(m);       // ← csrc/apis/mega_moe.hpp
deep_jit::register_python_api(m, deep_gemm::jit); // ← JIT 运行时自带的调试接口
```

这就是"Host 层按功能域拆分"的直接证据：**8 个 `register_apis` 对应 `apis/` 下 8 个头**。

### 站点 3：GEMM API 域 —— `csrc/apis/gemm.hpp`（680 行，本站只看三处）

1. 第 45～59 行的形状总表注释（Normal/MGrouped/KGrouped/Batched 下 A、B、CD 的形状）——第 1 天已经用过，今天从"地图"角度再确认：M/K-Grouped 的差异全部收敛在 `GemmType` 上。
2. 第 60 行 `get_gemm_desc()`：所有 GEMM 类接口共用的"参数 → 描述符"函数，第 81～97 行按 `gemm_type` switch 解形状。它产出的 `GemmDesc` 定义在 `csrc/jit_kernels/gemm_config.hpp` 第 39 行（"what to compute"），与第 98 行的 `GemmConfig`（"how to compute"）相对。
3. 第 413～453 行的 BF16 入口族：模板函数 `bf16_gemm<kTransA, kTransB>`（第 424 行）+ 四个别名（第 450～453 行，`bf16_gemm_nn = bf16_gemm<false, true>`），第 447 行调用 `launch_bf16_gemm` 进入 JIT 层；第 632～676 行把这些名字 `m.def` 给 Python。

### 站点 4：JIT 启动层 —— `csrc/jit_kernels/bf16_gemm.hpp`（80 行）

3.3 节已摘录骨架。今天补三个观察：

- `select_gemm_config(desc)`（第 17 行，实现在 `csrc/jit_kernels/heuristics.hpp` 第 592 行）从 `GemmDesc` 推出 `GemmConfig`——启发式选配置的细节留到第 12 天；
- 生成的代码通过 `#include <deep_gemm/bf16_gemm.hpp>`（第 19 行）把 device 模板拉进来，`bf16_gemm_impl<...>` 的实参列表（第 23～39 行）就是 `GemmConfig` 与 `GemmDesc` 字段的拼接；
- 第 67～70 行有个 `dry_run` 分支（`compile_without_load`）：只编译不加载，测试用它在无执行的情况下并行预编译 kernel（`deep_gemm/testing/par_compile.py`）。

### 站点 5：JIT 运行时封装 —— `csrc/jit/jit.hpp`（96 行）

- 第 14 行 `using AscendJIT = deep_jit::Runtime<deep_jit::Ascend>;`、第 16 行全局 `jit` 对象——真正的 JIT 运行时来自 `third-party/deep-jit` 子模块（外部仓库 DeepJIT，README 致谢部分注明）；
- 第 18～74 行 `Runtime` 类保存全局状态：可用核数 `num_sms`、NPU 架构 `arch`、`dry_run`、`deterministic_algorithms`、以及固定为 256 的 contiguous layout 对齐常量（第 59 行）——对应 README Utilities 里那组 get/set；
- 第 78～92 行 `init_jit()`：`include_dir = library_root / "include"`，即包目录下的 `deep_gemm/include/`（呼应站点 1 的 `_C.init` 调用）。

### 站点 6：Device Kernel 模板 —— `deep_gemm/include/deep_gemm/bf16_gemm.hpp`（195 行）

今天只看"外形"，不读循环体：

- 第 9～26 行：二十多个模板参数（`kMajorA/kMajorB`、`SHAPE_M/N/K`、`BLOCK_M/N/K`、`MAD_M/N/K`、各级 stage 数、L2 策略、`kDirectStore/kResidentA/kResidentB`）——这就是 JIT 要实例化的东西；
- 第 27 行入口 `bf16_gemm_impl`，带 `__global__ __mix__(1, kDirectStore ? 0 : 2)`：**混合核**，同一个 kernel 同时包含 AIC（矩阵乘加）与 AIV（向量）两套代码；
- 第 81 行 `if ASCEND_IS_AIC` 把函数分成两支：AIC 分支（第 81～172 行）做搬运与 MAD，另一支（第 173～192 行）做 epilogue 排空（`aiv_drain`）——这正是第 1 周第 5 天"AIC/AIV 分工"的落点；
- 第 66～70 行四个 `static_assert` 检查 L1/L0A/L0B/L0C 容量——容量常量来自 `common.hpp`。

目录层面再认一下公共设施：`scheduler.hpp`（输出 Tile 调度）、`epilogue.hpp`（+ `epilogue/operators.hpp`）、`ascend.hpp`（第 6～10 行注释写明按 `tile_ptr` / `copy` / `sync` 三层拆分的设备原语，对应 `ascend/` 下三个头）。

### 站点 7：Host/Device 共享定义 —— `deep_gemm/include/deep_gemm/common.hpp`（188 行）

第 13～15 行注释点明身份："Shared definitions used by BOTH host and device code"。确认四个成员的位置：容量/拓扑常量（第 50～61 行，`kNumAICores = 32`、`L1SizeBytes = 512 * 1024` 等）、`Major` 枚举（第 82 行）、`GemmType` 枚举（第 87～92 行，四个值）、`gm_ptr` 模板（第 137 行）。**Python 测试里还有一份镜像**：`tests/generators.py` 第 15 行 `Major`、第 20 行 `GemmType`——测试与 C++ 用同名的枚举描述同一件事。

### 站点 8：测试体系 —— `tests/`

- `tests/test_bf16_gemm.py`（50 行）：命令行 `select` 参数有 `all/normal/unaligned-mn/unaligned-k/strided/mgrouped/kgrouped` 七个选项（第 25 行），`--skip-prof` 表示只测正确性不测性能（第 26 行）——第 9 天的主角色；
- `tests/common.py`：`gemm_dg`（第 49 行）按 gemm_type/转置组合分发到 `deep_gemm` 的各个接口；`gemm_aclnn`（第 87 行）用 ACLNN 参考实现交叉验证（只覆盖 Normal）；`run_gemm_test`（第 131 行）是正确性 + 性能的总驱动；
- `tests/generators.py`：`GemmTestDesc`（第 250 行）描述一个测试用例，`enumerate_gemm_tests`（第 652 行）枚举 shape × major × recipe × gemm_type 的组合，还有 FP4/FP8 量化与 psum layout 的构造工具；
- `tests/utils.py`：`calc_diff`（第 36 行，基于余弦相似度的误差度量，口径与 DeepGEMM 一致）、`print_perf`（第 106 行）等；
- `tests/huge1g.cpp`：一个临时编译的小扩展，用 `ACL_MEM_MALLOC_HUGE1G_ONLY` 分配 1 GiB 大页 KV cache（供 `DG_TEST_HUGE1G=1` 时注意力测试使用）；
- 其余 `test_*.py` 与 3.4 节能力地图一一对应；`test_lazy_init.py` 验证 `import deep_gemm` 不触发 NPU 运行时初始化（保证可 fork，对应 `csrc/apis/runtime.hpp` 第 42 行注释）。

**阅读检查点**（能回答再进入实践）：

1. `deep_gemm/__init__.py` 里哪些接口是纯 re-export，哪两个是手写包装？为什么它们必须是 Python 代码？
2. `csrc/python_api.cpp` 的 9 行注册分别对应 `apis/` 下哪些文件？
3. `launch_bf16_gemm` 生成的代码字符串 include 了哪个头？该头文件物理上位于仓库哪个目录？JIT 编译时从哪里找到它？
4. `common.hpp` 为什么强调 host 和 device 共用？举出两个定义在其中的名字。

## 五、实践步骤

以下步骤按顺序执行，**全部不需要 Ascend 硬件**；总时长约 2 小时。所有命令都在仓库根目录 `DeepGEMM-Ascend/` 下运行，且均为只读操作，不会改动仓库。

### 步骤 1：通读 README 并摘录关键信息（约 25 分钟）

按顺序读完：项目介绍（中英两段）、News、Quick Start、Requirements、Interfaces（Kernel Interface + Utilities + Environment Variables）、Performance 各小节的标题和表头。在笔记里摘录：

- 目标硬件与首次发布时间；
- CANN 版本，以及它提供的两个关键工具名（`bisheng`、`ld.lld`）；
- Python 版本要求、两个 Python 依赖（`tilelang`、`tree-sitter*`）各自的用途；
- 6 个以上 Utilities 函数名；
- `DG_JIT_CACHE_DIR` 的默认值；
- 五个 Performance 小节分别对应什么算子。

### 步骤 2：清点目录（约 15 分钟）

```bash
ls
find deep_gemm csrc tests scripts -type f | sort
wc -l csrc/python_api.cpp deep_gemm/__init__.py csrc/apis/gemm.hpp
```

对照 3.2 节的目录树逐项确认：`csrc/apis/` 是否恰好 8 个头？`csrc/jit_kernels/` 有哪些文件？`deep_gemm/include/deep_gemm/` 的子目录（`ascend/`、`scheduler/`、`layout/`、`comm/`、`simd/`、`epilogue/`、`utils/`）各有哪些文件？`tests/` 下有几个 `test_*.py`？把文件数记进笔记。

### 步骤 3：五类代码代表文件打卡（约 30 分钟）

对 3.3 节表格的每一类，打开一个代表文件，记录三件事：总行数、你第一个能说出用途的函数/类名、它 include/import 了哪些**本仓库**路径。建议选择：

| 类别 | 文件 |
| :--- | :--- |
| ① Python API | `deep_gemm/__init__.py` |
| ② C++ Host | `csrc/apis/runtime.hpp`（82 行，最短的一个域） |
| ③ JIT 启动 | `csrc/jit_kernels/bf16_gemm.hpp` |
| ④ Device Kernel | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` |
| ⑤ 测试 | `tests/test_bf16_gemm.py` |

### 步骤 4：静态追踪两个入口（约 35 分钟）

**4a. 追踪 `bf16_gemm_nn`**（验证 3.3 节调用链的每一站）：

```bash
grep -n "bf16_gemm_nn" deep_gemm/__init__.py
# 预期：第 64 行，来自 ._C 的 re-export

grep -n "bf16_gemm_nn\|launch_bf16_gemm" csrc/apis/gemm.hpp
# 预期：第 447 行调用 launch_bf16_gemm；第 452 行别名定义；第 638 行 m.def 注册

grep -n "bf16_gemm_impl" csrc/jit_kernels/bf16_gemm.hpp deep_gemm/include/deep_gemm/bf16_gemm.hpp
# 预期：jit_kernels 第 23 行（生成代码中的实例化）、include 第 27 行（模板定义）
```

把三组行号抄进笔记，连成你自己的第一张调用链草图（第 13 天会闭卷重画）。

**4b. 追踪 `tf32_hc_prenorm_gemm`**（看一条不走 `_C` 的路线）：

```bash
grep -n "tf32_hc_prenorm_gemm" deep_gemm/tilelang_ops/__init__.py deep_gemm/tilelang_ops/hc_prenorm.py
grep -n "get_num_sms\|_C\." deep_gemm/tilelang_ops/hc_prenorm.py | head
```

确认：kernel 定义与公开入口都在 `hc_prenorm.py` 内（第 61、363 行），它不经过 pybind11/JIT 主线，但仍借用 `_C.get_num_sms()` 查询核数（第 26 行）。对照 `tests/test_hyperconnection.py` 的 import，说明这条路线的测试同样只依赖 Python API。

### 步骤 5：核对算子能力地图（约 15 分钟）

对照 3.4 节两张表逐行验证文件真实存在：

```bash
ls csrc/apis csrc/jit_kernels deep_gemm/include/deep_gemm tests scripts/profiling
grep -rn "not implemented" csrc/apis/gemm.hpp
```

你应该能找到 `m_grouped_fp8_fp4_gemm_nt_masked` 未实现的两处注释（第 606、630 行）。在能力地图上用记号标出"本后端未实现/组合不全"的接口。

### 步骤 6（可选，需要 Ascend 环境）：构建与导入检查（约 20 分钟）

在仓库根目录执行：

```bash
./develop.sh
python -c "import deep_gemm; print(deep_gemm.__version__); print(deep_gemm.get_num_sms())"
```

预期：`develop.sh` 输出 `Linked: .../deep_gemm/_C....so`；第二条命令打印版本号（如 `0.1.1+...`）和设备 AI 核数。这只是验证"构建 + import + 设备发现"这一段，不运行任何 kernel（跑最小测试是第 9 天的任务）。没有环境就跳过——今天的学习目标全部不依赖它。

### 步骤 7：组装当天产出（约 20 分钟）

按第六节模板独立填写对照表。建议**合上本教程**，只借助于你在步骤 1～5 留下的笔记和终端历史填写，填完再回来核对。

## 六、当天产出

学习计划要求今天的产出是**一张仓库目录与职责对照表**。建议按下面的模板整理（Markdown 或纸笔均可）：

```text
# DeepGEMM-Ascend 仓库代码地图（第 2 周 · 第 1 天）

## 1. 顶层目录与文件
| 路径 | 类别 | 职责（一句话） |
| README.md        | 文档   | （自己概括） |
| setup.py         | 构建   | … |
| develop.sh       | 构建   | … |
| CMakeLists.txt   | 构建   | … |
| csrc/            | ？     | … |
| deep_gemm/       | ？     | …（提示：两重身份） |
| tests/           | ？     | … |
| scripts/         | ？     | … |
| third-party/     | 依赖   | 列出三个子模块及用途 |

## 2. 五类代码
| 类别 | 位置 | 代表文件 | 一句话职责 |
| ① Python API | … | … | … |
| ② C++ Host   | … | … | … |
| ③ JIT 启动   | … | … | … |
| ④ Device Kernel | … | … | … |
| ⑤ 测试       | … | … | … |

## 3. 算子能力地图
| 算子 | Python 入口 | Host 文件 | JIT 文件 | Device 文件 | 测试文件 |
| Dense BF16 | … | … | … | … | … |
| Dense FP8/FP4 | … | … | … | … | … |
| M/K-Grouped | … | … | … | … | … |
| MQA Logits | … | … | … | … | … |
| MegaMoE | … | … | … | … | … |
| HC Prenorm | … | … | … | … | … |

## 4. 关键事实备忘
- 被编译的唯一一个 C++ 源文件：＿＿＿
- JIT 编译的 include 目录：＿＿＿（哪行代码设置的）
- JIT 缓存默认目录：＿＿＿
- 本后端未实现的接口：＿＿＿
- 目标硬件 / CANN 版本：＿＿＿

## 5. 遗留问题
- （记下今天没想明白的问题，第 9～13 天逐步解决）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. DeepGEMM-Ascend 与上游 DeepGEMM 是什么关系？对使用者意味着什么？
2. 仓库中哪五个位置分别对应五类代码？`deep_gemm/` 目录为什么算"两重身份"？
3. 被 `setup.py` 编译成 `deep_gemm._C` 扩展的源文件是哪个？`csrc/apis/` 下的头文件为什么不参与这次编译？device kernel 又是什么时候编译的？
4. `csrc/python_api.cpp` 里的注册行与 `csrc/apis/` 下的文件是什么对应关系？
5. 以 `bf16_gemm_nn` 为例，从 Python 调用到 device kernel 被"用上"，中间依次经过哪些文件/函数？（说出文件名和函数名即可）
6. 项目支持哪五大家族算子？各自的 Python 入口是什么？
7. HC Prenorm 的实现路线与其余算子有什么本质不同？
8. `fp8_gemm_nt` 与 `fp8_fp4_gemm_nt` 是什么关系？`m_grouped_fp8_fp4_gemm_nt_masked` 呢？
9. （思考题）如果把 `deep_gemm/include/` 从 wheel 的 package_data 中去掉，会发生什么？

**参考答案：**

1. 它是 DeepGEMM 在昇腾平台上的移植，完全兼容上游 API、共用 `deep_gemm` 包名；对使用者意味着：安装本包后，原先调用 DeepGEMM 的代码无需修改接口即可在昇腾上运行（README 第 3、9 行）。
2. ① `deep_gemm/*.py`；② `csrc/`（`python_api.cpp`、`apis/`、`utils/`、`backends/`）；③ `csrc/jit_kernels/` 与 `csrc/jit/jit.hpp`；④ `deep_gemm/include/deep_gemm/`；⑤ `tests/`。`deep_gemm/` 既是 Python 包（`__init__.py`、`mega/`、`tilelang_ops/`、`testing/`），又通过 `deep_gemm/include/` 存放 device kernel 模板头，后者被 `setup.py` 的 package_data 打进 wheel（第 150～153 行），供 JIT 运行时编译使用。
3. 只有 `csrc/python_api.cpp`（`setup.py` 第 43 行）。`apis/` 等头文件以 `#include` 方式被它拉入，属于同一翻译单元，无需单独编译；device kernel 模板则在**运行时**由 JIT 层生成实例化代码、调用 CANN 的 `bisheng` 编译并缓存（`csrc/jit_kernels/bf16_gemm.hpp` 第 18～42、71 行）。
4. 一一对应：`python_api.cpp` 第 33～40 行的 8 个 `register_apis` 分别来自 `apis/runtime.hpp`、`apis/epilogue_class.hpp`、`apis/gemm.hpp`、`apis/attention.hpp`、`apis/einsum.hpp`、`apis/layout.hpp`、`apis/aclnn.hpp`、`apis/mega_moe.hpp`。
5. `deep_gemm/__init__.py`（re-export `_C.bf16_gemm_nn`，第 64 行）→ `csrc/apis/gemm.hpp` 的 `bf16_gemm<false, true>`（第 425、452 行，经 `get_gemm_desc` 构造 `GemmDesc`）→ `launch_bf16_gemm`（`csrc/jit_kernels/bf16_gemm.hpp` 第 16 行，经 `select_gemm_config` 选 `GemmConfig`、生成并 `jit->compile` 实例化代码）→ `jit->launch` 启动 → device 侧执行 `bf16_gemm_impl`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 第 27 行）。
6. Dense GEMM（`bf16_gemm_*`、`fp8_fp4_gemm_*`）、Grouped GEMM（`m_grouped_*_contiguous`、`k_grouped_*_contiguous`）、MQA Logits（`fp8_fp4_mqa_logits` 及 paged/sparse/paged_sparse 变体）、MegaMoE（`deep_gemm.mega.fp8_fp4_mega_moe` 等）、HC Prenorm（`tf32_hc_prenorm_gemm`）。
7. HC Prenorm 用 TileLang 编写，kernel 与公开入口都在 Python 包内的 `deep_gemm/tilelang_ops/hc_prenorm.py`，不经过 pybind11 `_C` 扩展，也不走本仓库的 DeepJIT 编译链路（TileLang 自带编译流程）；其余算子都走"② Host → ③ JIT → ④ 模板 kernel"主线。
8. `fp8_gemm_nt` 是 `fp8_fp4_gemm_nt` 的属性别名（`csrc/apis/gemm.hpp` 第 624～629 行，`m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt")`），两者是同一个函数；`m_grouped_fp8_fp4_gemm_nt_masked` 在本后端未实现（同文件第 606、630 行注释），调用会失败。
9. JIT 编译生成的实例化代码 `#include <deep_gemm/bf16_gemm.hpp>` 等头将找不到文件，任何 kernel 都无法在运行时编译。`init_jit`（`csrc/jit/jit.hpp` 第 80 行）把 `<包目录>/include` 设为 include 目录，而该目录正是随包分发的（`setup.py` package_data `'include/**/*'`）——去掉它等于去掉了全部 device kernel 源码。
