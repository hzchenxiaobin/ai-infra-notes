# 第 2 周 · 第 6 天（总第 13 天）：串起完整 Host 调用链（复盘日）

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 2 周"项目结构与 JIT 调用链"的第 6 天（复盘日）教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：知识梳理约 50 分钟，综合练习约 70 分钟，通过标准自评约 20 分钟。

本周五天我们把 `deep_gemm.bf16_gemm_nt(a, b, d)` 从 Python 一路拆到了 `jit->launch` 的门口，每天都只看一层。今天是第 2 周复盘日，**不引入新内容**。学习计划给今天的任务是：**不看之前的笔记，独立画出从 Python API 到 `bf16_gemm_impl` 的完整 Host 调用链，标注每一层的输入、输出和关键职责，再对照源码修正**。

**通过标准**：能说明 `GemmDesc` 与 `GemmConfig` 的区别，以及 JIT 为什么需要生成模板实例化代码。

复盘方法与第 1 周相同：判断掌握程度的唯一标准是**闭卷复述**。第 2.2、2.3 节是"对照源码修正"环节的参考答案——请先做练习 1 再看它们；第 2.4、2.5 两节是通过标准两问的参考答案，先闭卷写自己的版本（练习 3、5），再回来订正。

## 一、今日定位与学习目标

完成今天的学习后，你应当能够：

1. 闭卷画出八层调用链——Python API → C++ binding → `get_gemm_desc` → `launch_bf16_gemm` → `select_gemm_config` → JIT compile → jit launch → `bf16_gemm_impl`——每层标注函数名、所在文件与关键行号；
2. 逐层说出每层的**输入是什么、输出是什么、关键职责是什么**（如 binding 层输入 pybind11 转换后的张量、输出一次 `launch_bf16_gemm(desc)` 调用）；
3. 说明 `GemmDesc` 与 `GemmConfig` 的区别：定义位置、由谁产生、字段分组、一对多关系，以及二者在"模板参数 + 运行期参数"两处分别被谁消费；
4. 解释 JIT 为什么需要生成模板实例化代码：C++ 模板的实例化规则、编译期常量对 kernel 的意义、由此带来的代价与仓库的三层对策（二进制缓存、`compiled_dims`、dry-run 预热）；
5. 追踪一个跨层的值（如 `set_num_sms` 的核数、变化的 M）如何穿过整条链，并判断它会不会触发重新编译；
6. 把本周五天的"日报"产物（代码地图、运行记录、入口调用链、字段来源表、launch 流程图）合并成一张可长期复用的架构图。

## 二、知识梳理

### 2.1 本周一览：五天各回答了什么问题

| 天 | 回答的问题 | 核心结论 |
| :---: | :--- | :--- |
| 8 | 代码在哪里、分几层？ | 五类代码：Python API（`deep_gemm/__init__.py`）、C++ Host（`csrc/apis`、`csrc/utils`）、JIT 启动（`csrc/jit_kernels` + `csrc/jit/jit.hpp`）、Device Kernel（`deep_gemm/include/deep_gemm/` 的模板头文件，随包分发、运行时才编译）、测试（`tests/`）。`_C` 扩展由唯一翻译单元 `csrc/python_api.cpp` 编译 |
| 9 | 链路怎么跑起来？ | `./develop.sh` 编译 `_C`；`python tests/test_bf16_gemm.py normal --skip-prof` 触发"并行预编译 → 正确性验证 → 性能测量"三阶段；`--skip-prof` 只测正确性 |
| 10 | 调用从哪里来？ | `bf16_gemm_nt` 等接口是 `_C` 内置函数（`__init__.py` 纯 re-export）；binding 五步：逻辑转置 → dtype 断言 → epilogue 归一 → `get_gemm_desc` → `launch_bf16_gemm`；nt/nn 默认 `compiled_dims="nk"`，tn/tt 默认 `"mn"` |
| 11 | 算什么？ | `get_gemm_desc()`（`csrc/apis/gemm.hpp:60-197`）七阶段把用户张量提取校验成 `GemmDesc`；返回 `nullopt` 表示空问题（m/n=0 或 k=0），不是错误 |
| 12 | 怎么算？ | `launch_bf16_gemm` 三步：`select_gemm_config` 推导 `GemmConfig`（硬约束 + 成本模型 + 精修）→ `std::format` 生成 26 实参的模板实例化源码 → `jit->compile`（缓存于 `~/.dj`）+ `jit->launch`；BLOCK 与 MAD 是两级切分；dry-run 只编译不启动 |

### 2.2 参考答案：完整 Host 调用链全景图

先做练习 1（闭卷画图），再对照本图修正。图中标注的是**物理调用栈**：② 和 ③ 是一次调用，④～⑦ 全部发生在 `launch_bf16_gemm` 这同一个函数体内（`csrc/jit_kernels/bf16_gemm.hpp:17/71/73`），⑥⑦ 是 DeepJIT 运行时的方法——计划里那条八箭头链是**逻辑顺序**，不要画成八个顺序嵌套的进程。

```text
┌─ ① Python API ──────────────────────────────────────────────────────────────┐
│ deep_gemm.bf16_gemm_nt(a, b, d, c=None, compiled_dims="nk", alpha=None)     │
│   deep_gemm/__init__.py:64（from ._C import，纯 re-export，无 Python 逻辑） │
│                     :159（import 时 _C.init(version, 包根路径) → init_jit） │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                ▼ pybind11 参数转换（Tensor/optional/pair/float）
┌─ ② C++ binding ─────────────────────────────────────────────────────────────┐
│ bf16_gemm<false,false>（csrc/apis/gemm.hpp:424-448；m.def 注册于 632-637，  │
│ 经 csrc/python_api.cpp:35 的 register_apis 接进 _C 模块）                    │
│   · get_transposed：按 nt/nn/tn/tt 把 A/B 统一成 [M,K]/[N,K] 视角（零拷贝） │
│   · dtype 断言：A/B 必须 BF16，D ∈ {BF16, FP32}（gemm.hpp:436-438）          │
│   · resolve_epilogue_class（csrc/jit_kernels/epilogue_class.hpp:110-128）：  │
│     epilogue 对象 / alpha 互斥，缺省补 IdentityEpilogue，再 check(c, d)      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                ▼
┌─ ③ get_gemm_desc（csrc/apis/gemm.hpp:60-197）── 输出 std::optional<GemmDesc> ┐
│   · major 判定：由 stride 推断（csrc/utils/layout.hpp:44-47，非用户声明）    │
│   · m/n/k：get_shape_by_spec 按轴标注交叉校验（utils/layout.hpp:140-219）    │
│   · c 语义：与 D 同指针 → acc；独立 [1,N] contiguous → bias → BiasEpilogue   │
│   · gemm_early_return（utils/layout.hpp:54-86）：m/n=0 或 k=0 → 返回         │
│     nullopt（k=0 时先把 D 清零或拷贝 C），② 层静默 return，不启动 kernel     │
│   · outer_stride_a/b/d（gemm_config.hpp:27-34）、num_cores（gemm.hpp:176，    │
│     = runtime->get_num_sms()）、stride_batch（仅 Batched）                   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                ▼ GemmDesc（①～③ 完成"算什么"的描述）
┌─ ④ launch_bf16_gemm（csrc/jit_kernels/bf16_gemm.hpp:16-78）─────────────────┐
│ ⑤ select_gemm_config(desc)（csrc/jit_kernels/heuristics.hpp:592-755）        │
│      → GemmConfig：搜索范围 → make_gemm_candidate 硬约束（heuristics.hpp:    │
│      92-204，含 c.launch_options.num_blocks = d.num_cores 于 :98）→ 成本    │
│      模型剪枝 → direct_store / resident_a/b / K 细化精修                    │
│      [DG_PRINT_CONFIGS] 打印 "GemmDesc(...) GemmConfig(...)"（749-753）      │
│ ⑥ 代码生成 + JIT compile：std::format 渲染 26 个模板实参（bf16_gemm.hpp:      │
│      18-66）→ 一段 &bf16_gemm_impl<...> 取地址的实例化源码；                 │
│      dry_run ? jit->compile_without_load（67-70）                            │
│              : jit->compile("bf16_gemm", code)（71）→ kernel 句柄             │
│      （jit 全局对象：csrc/jit/jit.hpp:14-16；Bisheng 编译，初始化于 init_jit   │
│      78-92；内容相同的 code 命中 DG_JIT_CACHE_DIR 缓存，默认 ~/.dj）           │
│ ⑦ jit->launch(kernel, config.launch_options, 9 个运行期参数)（73-77）：     │
│      gm_ptr×3（outer_stride, ptr, stride_batch）、m、n、k、grouped_layout、  │
│      num_groups、make_epilogue_operator_args(desc.n)；                       │
│      在当前 NPU stream 上执行（get_current_npu_stream，csrc/python_api.cpp:21）│
└───────────────────────────────┬─────────────────────────────────────────────┘
                                ▼
┌─ ⑧ bf16_gemm_impl（deep_gemm/include/deep_gemm/bf16_gemm.hpp:27-34）─────────┐
│   26 个模板参数（:9-26，编译期）+ 9 个运行期形参（gm_a/gm_b/gm_d、           │
│   shape_m/n/k_runtime、grouped_layout、num_groups、epilogue_args）一一对应    │
│   → Scheduler 分 Tile → GM→L1→L0 搬运 → MAD 累加 → Epilogue → 写回 GM       │
│   （设备侧内部 = 第 3～4 周的内容，本周止步于签名）                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

一句话版：**用户张量经 binding 归一后由 `get_gemm_desc` 收敛成 `GemmDesc`（算什么）；`launch_bf16_gemm` 用它选出 `GemmConfig`（怎么算），把二者渲染成一份 `bf16_gemm_impl<...>` 显式实例化源码，交给 DeepJIT/Bisheng 编译（可缓存）并携带 9 个运行期参数启动到当前 NPU stream 上。**

### 2.3 逐层标注：输入、输出与关键职责

| 层 | 源码位置 | 输入 | 输出 | 关键职责 |
| :---: | :--- | :--- | :--- | :--- |
| ① Python API | `deep_gemm/__init__.py:64/159` | 用户张量 a/b/d（+可选 c、compiled_dims、alpha、epilogue） | 对 `_C.bf16_gemm_nt` 的调用 | 门面与 re-export；import 时用包根路径初始化 JIT 运行时（`init_jit` 定位 `include/` 与 CANN 工具链） |
| ② C++ binding | `csrc/apis/gemm.hpp:424-448`（注册 632-637） | pybind11 转换后的 `torch::Tensor`、`std::optional`、`float`、Epilogue 对象 | 一次 `launch_bf16_gemm(desc)` 调用 | 逻辑转置（零拷贝）、dtype 断言、epilogue 归一（互斥检查 + 缺省 Identity） |
| ③ get_gemm_desc | `csrc/apis/gemm.hpp:60-197` | gemm_type、a/b/d/c、epilogue_class、grouped_layout、compiled_dims 等 | `std::optional<GemmDesc>` | major/shape/stride 提取与校验；c 的 acc/bias 语义；空问题早退（nullopt）；SF 布局转换（低精度，本周跳过） |
| ④ launch_bf16_gemm | `csrc/jit_kernels/bf16_gemm.hpp:16-78` | `GemmDesc` | 一次设备 kernel 启动 | "怎么算"的调度台：⑤⑥⑦ 三步串联 + dry-run 分支 |
| ⑤ select_gemm_config | `csrc/jit_kernels/heuristics.hpp:592-755` | `GemmDesc` | `GemmConfig` | 硬约束筛候选（L0C/L1/L0/UB 容量、对齐、m-grouped 固定 BLOCK_M=256）、成本模型剪枝、direct_store/resident/K 细化；无缓存（593 行 TODO） |
| ⑥ JIT compile | `jit->compile("bf16_gemm", code)`；`csrc/jit/jit.hpp:14-16/78-92` | kernel 名 + 生成的源码字符串 | kernel 句柄 | Bisheng 编译设备代码；内容相同 → 命中 `~/.dj` 缓存不再编译 |
| ⑦ jit launch | `csrc/jit_kernels/bf16_gemm.hpp:73-77`；`csrc/python_api.cpp:21-23` | 句柄 + `LaunchOptions`（num_blocks=num_cores 等）+ 9 个运行期参数 | 在当前 NPU stream 上排队执行的 kernel | 参数打包与流管理；`gm_ptr<uint8_t>{outer_stride, ptr, stride_batch}` 三元组与设备侧类型化形参一一对应 |
| ⑧ bf16_gemm_impl | `deep_gemm/include/deep_gemm/bf16_gemm.hpp:27-34` | 26 个模板参数（编译期）+ 9 个运行期参数 | 写回 GM 的 D | 调度、搬运、MAD、Epilogue（第 3～4 周展开） |

### 2.4 通过标准之一：`GemmDesc` 与 `GemmConfig` 的区别

两个结构体都定义在 `csrc/jit_kernels/gemm_config.hpp`，注释各一句话点题（第 36、97 行）：

| | `GemmDesc`（39-95 行） | `GemmConfig`（97-121 行） |
| :--- | :--- | :--- |
| 注释原文 | "Problem descriptor: **what to compute** (dimensions, dtypes, pointers, layouts)" | "Execution configuration: **how to compute** (tiling, pipeline, launch)" |
| 中文定位 | 算什么——问题的唯一描述 | 怎么算——解的一种选择 |
| 产生方式 | `get_gemm_desc()` 从用户张量**提取并校验**（第 11 天） | `select_gemm_config(desc)` 按同一份 desc **启发式搜索**（第 12 天） |
| 字段分组 | 指针/dtype、形状、布局（major + outer/batch stride）、acc + epilogue + compiled_dims、分组扩展（gemm_type/grouped_layout/expected_m/k）、生命周期（sfa/sfb 持有引用）六组 | Tile（block/mad）、流水深度（l1/l0/epi/sf stages）、路径开关（direct_store/resident/dual_aiv）、L2 控制、LaunchOptions 五组 |
| 唯一性 | 用户给定的问题唯一确定 | 同一 desc 理论上可配不同 config——heuristics 只是选出成本模型最优的一种（第 52 天的单变量实验正是人为换配置） |
| 被谁消费 | ③ 的产出、⑤ 的输入；26 个模板实参中的 majors/shape/acc/gemm_type/cd_dtype/epilogue/对齐来源；9 个 launch 参数**全部**来自 desc | 其余模板实参（block/mad/stages/L2/开关/num_blocks）+ `launch_options` |

三个"证据"帮助记忆：

1. `GemmArgs`（gemm_config.hpp:124-127）只是 `desc + config` 的打包——二者必须成对才能执行，但来源与语义完全不同；
2. `DG_PRINT_CONFIGS=1` 打印的每一行就是二者并列的样子：`GemmDesc(m=..., n=..., k=..., ...) GemmConfig(block=...x...x..., mad=..., l1_stages=..., ...)`（格式化器在 gemm_config.hpp:145-172，打印点在 heuristics.hpp:749-753）；
3. 分界线的另一种表述：**desc 里"每次调用都可能变的"（地址、运行期 shape、组信息、epilogue 参数）走 launch 参数；"决定生成哪份代码的"（majors、acc、gemm_type、dtype、epilogue 类型、compiled_dims 命中的维度）与 config 一起进模板参数。**

### 2.5 通过标准之二：JIT 为什么需要生成模板实例化代码

这一问的完整答案分四步，每步都有源码支撑：

**（a）C++ 模板不实例化就没有机器码。** `bf16_gemm_impl` 是一个 **26 参的函数模板**（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:9-26`）。函数模板只是"生成函数的配方"，编译器不会为它生成任何机器码，只有显式实例化（类型与常量全部确定）才产出一个可调用的函数。26 个参数的组合空间近乎无限，**不可能预先编译所有组合**，所以只能"用到哪个组合，现场实例化哪一个"——这正是 JIT 的含义。

**（b）为什么这些信息必须做成编译期常量，而不是运行期参数？** 因为设备 kernel 要用它们做**编译期资源规划**（`bf16_gemm.hpp:59-70`）：

```cpp
constexpr l1_ptr<bfloat16_t, kMajorA> l1a(BLOCK_M, ..., 0);   // L1 缓冲的地址/尺寸编译期定死
constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);
static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes, "L1 overflow");   // 容量错误编译期暴露
static_assert(BLOCK_K % MAD_K == 0, ...);
```

片上存储（L1/L0A/L0B/L0C）的布局、多级 stage 的偏移、容量断言全部依赖编译期常量；若改成运行期参数，这些地址只能运行时再算，`static_assert` 也无从谈起。这是接近硬件极限的必要条件，也是 host 侧 heuristics 精心推导出的 config 值得"焊死"进代码的原因。

**（c）生成手段。** host 侧用 `std::format` 把 desc + config 渲染成一段独立源码（`csrc/jit_kernels/bf16_gemm.hpp:18-42`）：

```cpp
static void __instantiate() {
    auto p = reinterpret_cast<void*>(&bf16_gemm_impl</*26 个实参*/>);
}
```

`__instantiate()` 没人调用，唯一的作用是**对模板函数取地址，迫使编译器实例化它**；`reinterpret_cast<void*>` 抹掉函数类型，让所有 kernel 的实例化代码形状统一。26 个实参 = desc 来源（majors、compiled_dims 命中的 shape、acc、gemm_type、cd_dtype、epilogue 类型）+ config 来源（block/mad、三级 stages、num_blocks、L2 提示、direct_store/resident）+ 全局常量（`kAlignment=256`，`csrc/jit/jit.hpp:59`）。

**（d）代价与对策。** 每换一组模板参数 = 一次 Bisheng 编译（秒级）。仓库的三层对策：

| 对策 | 机制 | 源码 |
| :--- | :--- | :--- |
| 二进制缓存 | 内容相同的生成代码命中缓存不再编译，只加载 | DeepJIT，`DG_JIT_CACHE_DIR`（默认 `~/.dj`，多路径首个命中） |
| `compiled_dims` | 把变化频繁的维度（如推理时的 M）留运行期（模板里填 0），只常量化稳定维度（N/K），减少特化数量 | `get_compiled_dim`（gemm_config.hpp:17-21）；nt/nn 默认 `"nk"`（gemm.hpp:635/641） |
| dry-run 预热 | `set_dry_run(True)` 后只编译不加载不启动，线程池并发把所有特化灌进缓存 | `compile_without_load`（bf16_gemm.hpp:67-70）；`deep_gemm/testing/par_compile.py` |

一句话版答案：**因为 kernel 是 26 参函数模板，只有按本次 desc+config 显式实例化才有机器码可启动；把 block/mad/stages 等做成编译期常量，kernel 才能用 `constexpr` 规划片上地址、用 `static_assert` 在编译期暴露容量错误——这是运行期参数做不到的。代价是每换一组参数要重新编译，所以要有缓存、`compiled_dims` 与 dry-run 预热来摊薄。**

### 2.6 易错点盘点

复盘时逐条自查，这些都是本周最容易混淆的地方：

1. **逻辑链 ≠ 调用栈**：⑤⑥⑦ 不是三个独立函数被顺序调用，而是 ④ `launch_bf16_gemm` 一个函数体里的三条语句（`bf16_gemm.hpp:17/71/73`）；③ 也是被 ② 的函数体调用后返回再继续。画图时把 ④ 画成"包含 ⑤⑥⑦ 的框"才对。
2. **`nullopt` 不是错误**：`get_gemm_desc` 返回 `nullopt` 表示空问题（m/n=0 或 k=0，后者先把 D 清零或拷贝 C，`utils/layout.hpp:75-80`），binding 静默 return；参数不合法才会走 `DJ_HOST_ASSERT`/`DJ_PANIC`，在 Python 侧表现为 `RuntimeError`。
3. **major 由 stride 推断，不由用户声明**：`get_major_type_ab`（`utils/layout.hpp:44-47`）看 `stride(-1)` 是否为 1；binding 的转置只是 shape/stride 的逻辑视图交换（`get_transposed`），`data_ptr` 不变。
4. **`SHAPE_* = 0` 的含义**：`get_compiled_dim` 未命中 `compiled_dims` 时返回 0，设备侧用 `SHAPE_M != 0 ? SHAPE_M : shape_m_runtime` 择一（`bf16_gemm.hpp:47-49`）——0 是"运行期取值"的哨兵，不是维度为 0。
5. **两个"缓存"**：kernel 二进制**有**缓存（DeepJIT，`~/.dj`）；配置选择**没有**缓存——`select_gemm_config` 每次完整重算（`heuristics.hpp:593` 的 TODO），因为它只是整数运算，远比编译便宜。
6. **dry-run 跳过的只是最后一步**：select 照跑（`DG_PRINT_CONFIGS` 仍打印）、代码照生成、编译照做（`compile_without_load`），只有"加载 + 启动"被跳过。
7. **epilogue 的双重身份**：`get_epilogue_operator_type()` 的字符串进模板参数（决定生成哪份代码），`make_epilogue_operator_args(n)` 的返回值进第 9 个 launch 参数（alpha、bias 指针、sfd 等）——同一个对象横跨两条通道。
8. **`num_cores` 一个值穿三层**：`runtime->get_num_sms()` → `desc.num_cores`（`gemm.hpp:176`）→ `config.launch_options.num_blocks`（`heuristics.hpp:98`）→ 模板参数 `kNumCores`（生成代码第 15 实参，`bf16_gemm.hpp:57`）；同一值又随 `launch_options` 传给 `jit->launch`——**模板参数给设备 Scheduler 静态分块，launch option 决定实际启动的 block 数**。

## 三、综合练习

以下练习全部**不需要 Ascend 硬件**；有环境的可选任务在练习 7。

### 练习 1：闭卷画链（主任务，约 20 分钟）

合上本教程与本周全部笔记，在白纸上画出计划要求的八层链：

```text
Python API → C++ binding → get_gemm_desc → launch_bf16_gemm
→ select_gemm_config → JIT compile → jit launch → bf16_gemm_impl
```

要求：每层标出函数/模块名与所在文件；③④⑧ 三层额外标出**输入和输出**（什么进来、什么出去）；④ 用框把 ⑤⑥⑦ 装进去。画完对照 2.2/2.3 用红笔订正。自评：**订正 ≤ 3 处 = 掌握；4～6 处 = 回读对应层当天的源码导读；> 6 处 = 本周内容需要整体重走**。

### 练习 2：故障定位——把报错放回正确的层（约 15 分钟）

以下是五次假想运行中出现的现象，写出各发生在哪一层（①～⑧）与哪个文件哪个函数，一句话说明判断依据。先独立作答，再看练习 2 答案（附在本节末）。

1. `RuntimeError: Shape mismatch: tensor #1, annotation="nk"`；
2. `RuntimeError: K-major operand requires unit stride on the inner (K) axis`；
3. `RuntimeError: Ambiguous major axis, the tensor has both last two dimensions contiguous`；
4. `Assertion failed: a.scalar_type() == torch::kBFloat16`；
5. 第二次运行同一测试，日志里不再出现 Bisheng 编译命令，但 `DG_JIT_PRINT_LOAD_TIME=1` 显示有 kernel 加载耗时。

### 练习 3：`GemmDesc` vs `GemmConfig` 闭卷辨析（约 15 分钟，通过标准第一问）

**Part A**：不看 2.4 节，闭卷填写下表，再对照订正。

```text
                GemmDesc                    GemmConfig
一句话           ＿＿＿＿（算什么/怎么算）      ＿＿＿＿
定义位置         ＿＿＿＿                      ＿＿＿＿
由谁产生         ＿＿＿＿                      ＿＿＿＿
字段分几组        ＿＿＿＿                      ＿＿＿＿
同一问题可有几种   ＿＿＿＿                      ＿＿＿＿
```

**Part B**：判断对错并写一行理由（答案见本节末）：

1. 同一个 `GemmDesc` 只可能对应一个合理的 `GemmConfig`；
2. 生成代码的 26 个模板实参全部来自 `GemmConfig`；
3. 用户可以通过 Python API 直接指定 `BLOCK_M`；
4. `D += A@B^T` 的累加语义写在 desc 里，最终变成模板参数 `kWithAccumulation`；
5. m-grouped 时 BLOCK_M 固定为 256，这条约束写在 desc 的构造阶段而不是 config 的推导阶段。

### 练习 4：一个值的跨层旅行（约 15 分钟，串起整条链）

**Part A：核数。** 在 Python 里执行 `deep_gemm.set_num_sms(16)` 后调用 `bf16_gemm_nt`：

1. 写出 16 依次经过哪些结构体的哪个字段，最终到达设备 kernel 的哪两个位置；
2. 追问：相比默认 32 核，这会不会触发一次新的 JIT 编译？为什么？

**Part B：变化的 M。** `bf16_gemm_nt` 调用 `M=128` 与 `M=4096`（N/K 不变）：

1. 模板参数 `SHAPE_M` 在两次调用中分别是多少？（提示：nt 的默认 `compiled_dims`）
2. 那么两次调用会共用同一份编译产物吗？写出完整的推理链；
3. 追问：如果两次恰好选出了相同的 `GemmConfig`，编译还会发生吗？此时两次调用的差异体现在哪里？

### 练习 5：手工渲染一次代码生成（约 15 分钟，通过标准第二问）

沿用第 12 天静态演算过的例子：BF16 Normal，`M=128, N=64, K=256`，`nt` 调用（A/B 均为 K-major，无 c/alpha/epilogue，`num_cores=32`），heuristics 推导出的 config 为 `block=128×512×128`、`mad=128×256×64`、`l1_stages=3`、`l0_stages=2`、`epi_stages=2`、`direct_store=false`、`resident_a/b=false`。

1. 写出 26 个模板实参的**值或来源类别**（desc / desc×compiled_dims / config / 全局常量；L2 三项标注"config，由 heuristics 推导"即可）；
2. 回答：`SHAPE_M` 为什么是 0？`bfloat16_t` 与 `epilogue::operators::Identity` 两个字符串分别由哪个函数产生？`__instantiate()` 为什么必须有？
3. 这段生成源码随后被交给谁？以什么为键命中缓存？

参考答案见附录 A。

### 练习 6：补齐本周疑问（约 10 分钟）

把本周每天笔记末尾的"遗留问题"翻出来逐条处理：已解决的写一句话答案划掉；未解决的填进下表（示例行供参考），明天（第 14 天机动日）或后续周按天消化。

| 疑问 | 后续主题 |
| :--- | :--- |
| 成本模型的具体公式（`get_gemm_cost` 内部） | 第 50 天 |
| `Scheduler` 如何把 block_idx 映射成输出 Tile | 第 17 天 |
| `copy_gm_to_l1` / `copy_l1_to_l0a/b` 的参数与边界处理 | 第 19 天 |
| 多级 stage 与 `asc_sync_wait/notify` 的配对时序 | 第 23 天 |
| `ENABLE_UPDATE/KEEP` 与首次累加 | 第 24 天 |
| AIC/AIV 分工与 `__mix__` | 第 25 天 |
| `kDirectStore` / `kResidentA/B` 的设备侧行为 | 第 26 天 |
| SF 布局转换（`transform_sf_into_required_layout`） | 第 33 天 |
| FP8/FP4 路径与 `dispatch_fp8_fp4_gemm` | 第 32 天 |
| 测试体系（参考结果、meta 张量、par_compile） | 第 29～30 天 |

### 练习 7（可选，需 Ascend 环境）

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day13_run.log
```

从日志里抄一行 `GemmDesc(...) GemmConfig(...)`（每种组合只打一次），给**每个字段**标注来源层：哪些来自 ②（epilogue 归一）、哪些来自 ③（major/shape/stride/num_cores）、哪些来自 ⑤（block/mad/stages/开关）。再删掉 `~/.dj` 缓存目录重跑一次，记录"冷/热"两次在 ⑥⑦ 两层的差异（有无编译命令、加载耗时）。无硬件时，练习 1～6 已是完整的静态替代任务。

### 练习 2 参考答案

1. **③**，`get_shape_by_spec` 的 `check_spec`（`csrc/utils/layout.hpp:154`）——张量维数与轴标注不符；
2. **③**，`get_outer_stride`（`csrc/jit_kernels/gemm_config.hpp:29`）——K-major 却内层轴 stride 不为 1；
3. **③**，`major_check`（`csrc/utils/layout.hpp:39`，被 `get_major_type_ab` 调用）——末两维都连续导致 major 二义；
4. **②**，`bf16_gemm` 的 dtype 断言（`csrc/apis/gemm.hpp:436`）——binding 层在进 `get_gemm_desc` 之前先验操作数类型；
5. **⑥**，`jit->compile` 命中 `~/.dj` 二进制缓存：相同生成代码不再编译，但 kernel 仍需加载进进程。

### 练习 3 Part B 参考答案

1. **错**。config 是 heuristics 在成本模型下"选一种"，同一 desc 可配别的 config（第 52 天的单变量实验就是人为换配置再验证正确性）；
2. **错**。26 个实参中还有 desc 来源（majors/shape/acc/gemm_type/cd_dtype/epilogue）与全局常量（kAlignment=256）；
3. **错**。Python API 只描述问题（张量、compiled_dims、alpha、epilogue），不透出任何 tiling/pipeline 参数；
4. **对**。`desc.acc`（`gemm.hpp:180`）→ 生成代码第 19 实参（`bf16_gemm.hpp:61`）→ 设备模板参数 `kWithAccumulation`（`bf16_gemm.hpp:20`）；
5. **错**。desc 只带 gemm_type/grouped_layout；"BLOCK_M 必须等于 256"是 ⑤ `make_gemm_candidate` 的硬约束（`heuristics.hpp:106`，与 psum 布局的组边界对齐有关），属于 config 推导阶段。

### 练习 4 参考答案

**Part A：**

1. `set_num_sms(16)` 写入全局 Runtime 的 `num_sms`（`csrc/jit/jit.hpp:38-41`，Python 侧入口 `csrc/apis/runtime.hpp:17-19`）→ 构造 desc 时 `num_cores = runtime->get_num_sms()`（`csrc/apis/gemm.hpp:176`）→ `make_gemm_candidate` 里 `launch_options.num_blocks = d.num_cores`（`heuristics.hpp:98`）→ 从这里**分成两条通道**：(a) 模板实参第 15 个 `config.launch_options.num_blocks.value()`（`csrc/jit_kernels/bf16_gemm.hpp:57`）变成设备模板参数 `kNumCores`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:16`），设备侧 `Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumCores, kAlignment>` 用它把 `block_idx` 映射成输出 Tile；(b) 同一个 `launch_options` 整体传给 `jit->launch`（`bf16_gemm.hpp:73`），决定实际启动的 block 数。
2. **会**。num_blocks 是模板实参之一，16 ≠ 32 → 生成的源码字符串不同 → 缓存键不同 → 触发一次新的 Bisheng 编译（32 核的旧缓存条目仍在，切回 32 核时直接命中）。

**Part B：**

1. 两次都是 0——`nt` 默认 `compiled_dims="nk"` 不含 `m`，M 永远走运行期；
2. **通常不能共用**：M 虽不进 SHAPE_M，但它是 `select_gemm_config` 的输入——M=128（单 wave，常触发 resident/direct_store、较小 block_m）与 M=4096（多 wave，更大 block_m）通常选出不同 config（第 12 天实践步骤 3 的观察点）→ block/stages/开关等模板参数不同 → 生成代码不同 → 各自首次编译；
3. 若两次恰好选出同一 config：生成代码逐字符相同 → 命中缓存不再编译，两次调用的差异只体现在 launch 参数上（第 4 个运行期参数 `desc.m`：128 vs 4096）。结论：**是否重编译由"整份生成代码是否变化"决定；`compiled_dims` 只决定某个 shape 维度自己进不进模板，而 config 也是生成代码的一部分**——这也是练习 3 Part B 第 1 题的深层原因。

## 四、通过标准

### 4.1 自评清单

学习计划的通过标准是"**能说明 `GemmDesc` 与 `GemmConfig` 的区别，以及 JIT 为什么需要生成模板实例化代码**"，加上当天任务本身，拆成六个检查点，逐条自评"能 / 勉强 / 不能"：

| # | 检查点 | 验证方式 |
| :---: | :--- | :--- |
| 1 | 闭卷画出八层链，顺序、文件、函数名基本正确 | 练习 1，订正 ≤ 3 处 |
| 2 | 每层能说出输入、输出、关键职责；能区分"逻辑链"与"物理调用栈" | 练习 1 的 ③④⑧ 标注 + 2.3 对照；易错点 1 |
| 3 | `GemmDesc` vs `GemmConfig`：一句话定位、产生方式、字段分组、一对多关系 | 练习 3 闭卷填表无重大遗漏 |
| 4 | JIT 模板实例化：为什么不实例化没机器码、编译期常量的好处、代价与三层对策 | 练习 5 + 2.5 对照；能不看材料说出 2.5 的一句话版 |
| 5 | 能追踪跨层的值（num_cores、变化的 M）并判断是否触发重编译 | 练习 4 两个 Part 全对 |
| 6 | 报错能定位到层 | 练习 2 五个场景至少对四个 |

### 4.2 评判与补救

- **六条全部"能"**：达标。明天（第 14 天）机动或休息，可以补跑测试或继续研究 JIT 细节；第 3 周开始进入设备 kernel 内部（模板参数、片上内存、调度）。
- **检查点 1/2 "不能"**：重走第 10 天 3.7 节（入口调用链）与第 12 天 3.7 节（launch 流程图），把两张图拼起来再画一次。
- **检查点 3 "不能"**：重读第 11 天 3.1 节 + 第 12 天 3.2 节（两个结构体各自的"一句话"），以及本日 2.4。
- **检查点 4 "不能"**：重读第 12 天 3.5～3.6 节（代码生成与 JIT 缓存），以及本日 2.5。
- **检查点 5 "不能"**：重读第 12 天 3.5 节（26 实参来源表）的"desc × compiled_dims"一行，以及本日易错点 8。
- **检查点 6 "不能"**：把练习 2 的五个报错按"②③⑤⑥"分四层重做，每层再各找一个源码断言作例子。

### 4.3 本周明确不要求掌握的内容

与计划的复盘定位一致，以下内容**不算**本周未达标项，各有专题日：设备 kernel 内部（Scheduler 第 17 天、地址与 Tile 第 18 天、搬运第 19 天、MAD/流水线/Epilogue 第 22～26 天）、成本模型公式（第 50 天）、SF 布局转换（第 33 天）、FP8/FP4 与 dequant 路径（第 32 天）、测试体系细节（第 29～31 天）、MegaMoE/MQA（第 6～7 周）。本周只要能把它们**放回 2.2 全景图的正确位置**即可。

## 附录 A：练习 5 参考答案

26 个实参（顺序即生成模板串 `bf16_gemm.hpp:24-39` 中 `{}` 的出现顺序）：

| # | 实参 | 来源类别 |
| :---: | :--- | :--- |
| 1-2 | `Major::K, Major::K` | desc（A/B 均按 stride 判为 K-major） |
| 3-5 | `0, 64, 256` | desc × compiled_dims（"nk" 命中 N、K，M 填 0 = 运行期取值） |
| 6-8 | `128, 512, 128` | config（block_m/n/k） |
| 9-11 | `128, 256, 64` | config（mad_m/n/k） |
| 12-14 | `3, 2, 2` | config（l1/l0/epi stages） |
| 15 | `32` | config（`launch_options.num_blocks` = desc.num_cores = 32） |
| 16-18 | `⟨l2_ctrl_a⟩, ⟨l2_ctrl_b⟩, ⟨l2_ctrl_store_cd⟩` | config（heuristics 按方向剪枝推导，本练习不演算具体枚举值） |
| 19 | `false` | desc（无 c → acc=false） |
| 20 | `GemmType::Normal` | desc |
| 21 | `bfloat16_t` | desc（`dtype_to_cd_kernel_type`，gemm_config.hpp:87-94） |
| 22 | `epilogue::operators::Identity` | desc（无 alpha/bias → IdentityEpilogue，epilogue_class.hpp:32） |
| 23 | `256` | 全局常量（`Runtime::get_mk_alignment_for_contiguous_layout()`，jit.hpp:59） |
| 24-26 | `false, false, false` | config（direct_store / resident_a / resident_b） |

问答：

1. `SHAPE_M=0` 因为 `compiled_dims="nk"` 不含 `m`，`get_compiled_dim` 返回 0；设备侧 `shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime`（`bf16_gemm.hpp:47`），运行期值来自 launch 第 4 个参数 `desc.m`。
2. `bfloat16_t` 来自 `GemmDesc::dtype_to_cd_kernel_type(desc.cd_dtype)`（生成代码处 `bf16_gemm.hpp:63` 调用）；`epilogue::operators::Identity` 来自 `desc.epilogue_class->get_epilogue_operator_type()`（`csrc/jit_kernels/epilogue_class.hpp:29-34`）。
3. `__instantiate()` 无人调用，唯一作用是对 `bf16_gemm_impl<...>` 取地址、迫使 Bisheng 为该特化生成机器码；`reinterpret_cast<void*>` 抹掉类型，让各 kernel 的实例化代码形状统一。
4. 交给 DeepJIT 运行时（`jit->compile("bf16_gemm", code)`），由 Bisheng 编译成可加载产物；**以生成源码的内容**（即 26 个实参的组合）为键缓存于 `DG_JIT_CACHE_DIR`（默认 `~/.dj`），下次相同组合只加载不编译。

## 附录 B：第 2 周复盘笔记模板（当天产出）

```text
# 第 2 周复盘笔记：完整 Host 调用链

1. 八层链（闭卷默写后对照 2.2 订正）
   -（贴图或文字版；红笔标出订正处）

2. 逐层一句话
   - Python API：＿＿；binding：＿＿；get_gemm_desc：＿＿
   - launch_bf16_gemm：＿＿；select_gemm_config：＿＿
   - JIT compile：＿＿；jit launch：＿＿；bf16_gemm_impl：＿＿

3. 通过标准两问（闭卷作答后对照 2.4/2.5）
   - GemmDesc vs GemmConfig：＿＿
   - JIT 为什么生成模板实例化代码：＿＿

4. 跨层追踪记录（练习 4）
   - set_num_sms(16) 的路径：＿＿；是否重编译：＿＿
   - M=128 vs M=4096：＿＿

5. 本周遗留问题（练习 6 处理后的清单）

6. 自评结论
   - 六个检查点：＿＿个"能"；未达标项与补救计划：＿＿
```

---

今天结束时，你应该能不看任何材料讲清：**一次 `deep_gemm.bf16_gemm_nt(a, b, d)` 从 Python 门面、binding 归一、desc 提取，到 config 选择、模板实例化代码生成、Bisheng 编译缓存、携带 9 个运行期参数启动的全过程**。明天（第 14 天）机动或休息；第 3 周我们将穿过 `jit->launch` 这条线，进入设备 kernel `bf16_gemm_impl` 的内部——模板参数、片上内存规划与任务调度。
