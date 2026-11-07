# 第 3 周 · 第 1 天（总第 15 天）：阅读 Kernel 模板参数

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 3 周"BF16 Kernel——模板、调度与数据搬运"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 45 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

第 2 周结束时，调用链停在 `jit->launch`：一份用 `std::format` 渲染出的 `bf16_gemm_impl<...>` 显式实例化代码被 Bisheng 编译成设备 kernel，9 个运行期参数被送上 NPU stream。从今天起我们打开设备侧的黑盒。本周的主文件只有一个：`deep_gemm/include/deep_gemm/bf16_gemm.hpp`（195 行），全部内容是一个函数模板 `bf16_gemm_impl`。今天**只读它的开头**——模板声明（第 9-26 行）和函数入口（第 27-34 行）：把 26 个模板参数按 Shape、Block、MAD、Stage、Core、L2、Epilogue 等类别整理成分类表（当天产出），弄清"编译期 shape 与运行期 shape"的边界在哪里，并初步认识三个带默认值的路径开关 `kDirectStore`、`kResidentA/kResidentB`。函数体里的调度、搬运、地址计算与同步，留给第 17～26 天逐个展开。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 `bf16_gemm_impl`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:27`）模板声明里 26 个模板参数的**分类**（布局 / Shape / Block / MAD / Stage / Core / L2 / Epilogue 与语义 / 对齐 / 路径开关）、各自的 Host 侧来源，以及在函数体内至少一个消费点的行号；
2. 解释 `SHAPE_M/N/K = 0` 哨兵的含义：设备侧第 47-49 行如何择一，`get_compiled_dim`（`csrc/jit_kernels/gemm_config.hpp:17-21`）如何决定填 0 还是填维度值，`bf16_gemm_nt` 默认 `compiled_dims="nk"` 的动机；
3. 说明**编译期 shape 的额外权限**：为什么 `kResidentA/kResidentB` 要求 `kGemmType == Normal && SHAPE_K != 0`（第 40 行断言），以及第 59 行的 L1 规划如何直接使用 `SHAPE_K`；
4. 读懂第 59-64 行的 `constexpr` 片上缓冲声明：`l1a/l1b/l0a/l0b/l0c` 的形状各由哪些模板参数决定、`l1b` 的起始地址为什么是 `l1a.offset(kNumL1Stages)`，并用 `size_per_stage/offset`（`ascend/tile_ptr.hpp:37-38`）手算一组配置的用量；
5. 列出函数体里的 11 条 `static_assert`（第 38、40、42、43、45、56、66-70 行）各自守护什么约束，体会"Host 硬约束 ↔ Device 断言互为镜像"（第 12 天）；
6. 说出函数入口 9 个运行期形参与 `jit->launch` 的 9 个实参（`csrc/jit_kernels/bf16_gemm.hpp:74-77`）的一一对应，解释"哪些信息进模板参数、哪些走 launch 参数"的分界原则；
7. 初步说明 `kDirectStore` 的用途（L0C 直接写回 GM、跳过 UB epilogue、AIV 数量归零、累加需原子加）与 `kResidentA/kResidentB` 的用途（A 常驻 L1 / B 常驻 L0B，跨输出 Tile 复用），以及它们各自的形状约束；
8. 有环境时用 `DG_PRINT_CONFIGS=1` 把打印出的 `GemmDesc/GemmConfig` 行手工翻译成 26 个模板实参；无环境时在纸面上完成同样的翻译与片上用量演算。

**当天产出：模板参数分类表**（模板见第六节，参考答案见 3.2 节）。

## 二、预备知识

- **第 12 天（最重要）**：`launch_bf16_gemm` 三步（选配置 → 生成代码 → 编译启动）；`GemmConfig` 五组字段；`BLOCK_M/N/K` 与 `MAD_M/N/K` 的两级切分；26 个模板实参的三类来源（desc / desc×compiled_dims / config）；`SHAPE=0` 表示运行期取值。
- **第 11 天**：`GemmDesc` 的字段与 `get_gemm_desc` 七阶段；`compiled_dims` 字段的含义。
- **第 13 天**：从 Python API 到 `jit->launch` 的完整 Host 调用链闭卷图。
- **第 5 天**：`GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` 数据流；MTE1（L1→L0）、MTE2（GM→L1）、M（计算）等流水线的分工；AIC（Cube，做矩阵乘）与 AIV（Vector，做向量后处理）的分工。
- **第 3 天**：`BLOCK_M/N/K` 的 Tiling 与数据复用。
- **第 4 天**：BF16 输入、FP32 累加（`l0c` 固定 `float` 的原因）。
- **C++ 语法**：函数模板与显式实例化、非类型模板参数（`uint32_t`、`bool`、枚举）、类型参数（`typename cd_dtype_t`）、模板模板参数（`template <typename> class epilogue_operator_t`）、`constexpr` 对象、`static_assert`、`if constexpr`。能对照读懂即可。

## 三、详细讲解

### 3.1 全景：这个文件在调用链的什么位置

`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 是**设备头文件**：它只被 JIT 生成代码 `#include`（`csrc/jit_kernels/bf16_gemm.hpp:19` 的 `#include <deep_gemm/bf16_gemm.hpp>`），由 Bisheng 编译成 NPU 上跑的 kernel。整个文件只有一个函数模板：

```cpp
template <                      // 第 9-26 行：26 个模板参数
    Major kMajorA, Major kMajorB,
    uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t MAD_M, uint32_t MAD_N, uint32_t MAD_K,
    uint32_t kNumL1Stages, uint32_t kNumL0Stages,
    uint32_t kNumEpilogueStages,
    uint32_t kNumCores,
    asc_load_l2_cache_mode kL2CtrlA,
    asc_load_l2_cache_mode kL2CtrlB,
    asc_store_l2_cache_mode kL2CtrlStoreCd,
    bool kWithAccumulation,
    GemmType kGemmType,
    typename cd_dtype_t,
    template <typename> class epilogue_operator_t,
    uint32_t kAlignment,
    bool kDirectStore = false, bool kResidentA = false, bool kResidentB = false
>
__global__ __mix__(1, kDirectStore ? 0 : 2) void bf16_gemm_impl(   // 第 27 行
    /* 9 个运行期形参，见 3.5 节 */
)
```

在完整调用链中的位置：

```text
Python bf16_gemm_nt(a, b, d, compiled_dims="nk")
    │  binding：逻辑转置 / 断言 / epilogue 归一        （第 10 天）
    ▼
get_gemm_desc ──► GemmDesc                              （第 11 天）
    ▼
launch_bf16_gemm                                        （第 12 天）
    ├─ select_gemm_config ──► GemmConfig
    ├─ std::format ──► bf16_gemm_impl<26 个模板实参> 的实例化源码
    └─ jit->compile + jit->launch(9 个运行期参数)
           │
           ▼  ═══════════ 以下进入设备侧（本周）═══════════
bf16_gemm_impl（deep_gemm/include/deep_gemm/bf16_gemm.hpp）
    ├─ 第 35-79 行：asc_init + 择 shape + 片上缓冲 constexpr 规划 + 断言 + Scheduler/Epilogue 构造
    ├─ 第 81-172 行：AIC 分支——调度 → GM→L1 → L1→L0A/L0B → MAD → L0C →（写回）
    └─ 第 173-192 行：AIV 分支——UB epilogue → 写回 GM（kDirectStore 时直接 return）
```

两个语法点先扫清：

- **`__global__ __mix__(1, kDirectStore ? 0 : 2)`**（第 27 行）：混合核的 AIC/AIV 配比——每个 block 由 1 个 AIC 加 N 个 AIV 组成；普通路径 N=2（AIV 负责 UB epilogue），`kDirectStore` 时 N=0（不需要向量核）。注意第 53-55 行实例化 `Epilogue` 时硬编码的第 9 个实参 `2` 正是 `kNumAIVs`，与普通路径的配比一致（AIC/AIV 分工细节第 25 天）。
- **`block_idx`**（第 73 行）：设备内置变量，即当前核（block）的编号，作为 `core_id` 传给 `Scheduler`。

### 3.2 26 个模板参数分类总表（当天产出·参考答案）

按下表逐个参数整理。行号未注明文件时均指 `deep_gemm/include/deep_gemm/bf16_gemm.hpp`。

| # | 参数 | 类别 | 设备侧含义 | Host 侧来源 | 函数体内消费点 |
| :--: | :-- | :-- | :-- | :-- | :-- |
| 1 | `kMajorA` | 布局 | A 的主轴（`Major::K` 或 `Major::MN`），决定 GM/L1 地址沿哪个方向走 | `desc.major_a` | `gm_a` 类型（28）、`l1a` 类型（59） |
| 2 | `kMajorB` | 布局 | B 的主轴 | `desc.major_b` | `gm_b`（29）、`l1b`（60） |
| 3 | `SHAPE_M` | Shape | M 的编译期值；**0 = 运行期取** | `get_compiled_dim(desc.m,'m',compiled_dims)` | 择一（47） |
| 4 | `SHAPE_N` | Shape | N 的编译期值；0 同上 | `get_compiled_dim(desc.n,'n',…)` | 择一（48） |
| 5 | `SHAPE_K` | Shape | K 的编译期值；0 同上；resident 路径**必需非 0** | `get_compiled_dim(desc.k,'k',…)` | 择一（49）、L1 规划（59）、断言（40、45） |
| 6 | `BLOCK_M` | Block | L1 级输出 Tile 的 M 尺寸 | `config.block_m` | `l1a` 形状（59）、`Scheduler`（73）、行号 90 |
| 7 | `BLOCK_N` | Block | L1 级 Tile 的 N 尺寸 | `config.block_n` | `l1b` 形状（60）、`Scheduler`（73）、91 |
| 8 | `BLOCK_K` | Block | K 维一次 GM→L1 搬运的深度 | `config.block_k` | `l1a/l1b` 深度（59-60）、`num_k_blocks`（96）、`actual_k`（114）、断言（70） |
| 9 | `MAD_M` | MAD | 一次 MAD 指令的 M 尺寸 | `config.mad_m` | `l0a/l0c` 形状（62、64）、`m_mad_idx` 步长（123） |
| 10 | `MAD_N` | MAD | 一次 MAD 的 N 尺寸 | `config.mad_n` | `l0b/l0c`（63-64）、`n_mad_idx` 步长（124）、对齐（126） |
| 11 | `MAD_K` | MAD | 一次 MAD 的 K 深度 | `config.mad_k` | `l0a/l0b` 深度（62-63）、`k_mad_idx` 步长（133）、断言（45、70） |
| 12 | `kNumL1Stages` | Stage | L1 中 A/B 缓冲级数（MTE2↔MTE1 之间） | `config.num_l1_stages` | `l1b` 基址（60）、`set_flags`（83）、stage 轮转（112）、断言（66） |
| 13 | `kNumL0Stages` | Stage | L0A/L0B 缓冲级数（MTE1↔Cube 之间） | `config.num_l0_stages` | `set_flags`（84）、stage 轮转（133）、断言（45、67、68） |
| 14 | `kNumEpilogueStages` | Stage | 每 AIV 的 UB epilogue 级数（AIC→AIV） | `config.num_epilogue_stages` | `Epilogue` 实参（54）→ `epilogue.kNumStages`（164） |
| 15 | `kNumCores` | Core | 参与持久化调度的 AI Core 数 | `config.launch_options.num_blocks` = `desc.num_cores` = `runtime->get_num_sms()`（heuristics.hpp:98、apis/gemm.hpp:176） | `Scheduler`（73）、`aiv_zero_empty_groups`（176） |
| 16 | `kL2CtrlA` | L2 | A 的 GM→L1 装载的 L2 缓存提示 | `config.l2_ctrl_a` | `copy_gm_to_l1`（110、117） |
| 17 | `kL2CtrlB` | L2 | B 的装载提示 | `config.l2_ctrl_b` | `copy_gm_to_l1`（118） |
| 18 | `kL2CtrlStoreCd` | L2 | D 写回 GM 的 L2 提示 | `config.l2_ctrl_store_cd` | `Epilogue` 实参（55）→ `copy_ub_to_gm`；direct 路径 `copy_l0c_to_gm`（159） |
| 19 | `kWithAccumulation` | Epilogue/语义 | `D += A@B` 累加语义 | `desc.acc` | `Epilogue`（53）、断言（38）、原子加（82） |
| 20 | `kGemmType` | Epilogue/语义 | Normal / M-Grouped / K-Grouped / Batched | `desc.gemm_type` | `Scheduler`（73）、派生标志（76-78）、分支（95、99-100、157、175-176） |
| 21 | `cd_dtype_t` | Epilogue/语义 | D 的元素类型（`bfloat16_t`/`float`/…） | `dtype_to_cd_kernel_type(desc.cd_dtype)` | `gm_d`（30）、`Epilogue`（53）、写回（157-158） |
| 22 | `epilogue_operator_t` | Epilogue/语义 | epilogue 算子（模板模板参数） | `epilogue_class->get_epilogue_operator_type()` | `Epilogue`（53-54）、断言（56-57） |
| 23 | `kAlignment` | 对齐 | 组边界（psum 布局）的对齐常数 | `Runtime::get_mk_alignment_for_contiguous_layout()` = 256（jit.hpp:59） | `Scheduler`（73）；m-grouped 断言 scheduler.hpp:146、k-grouped 对齐 scheduler.hpp:171 |
| 24 | `kDirectStore` | 路径开关 | L0C 直接写回 GM，跳过 UB epilogue | `config.direct_store` | `__mix__`（27）、原子加（82）、跳过 wait_free（129）、`copy_l0c_to_gm`（156-159）、AIV return（174）；断言（38、56） |
| 25 | `kResidentA` | 路径开关 | A 常驻 L1（整条 K 一次搬入，跨 Tile 复用） | `config.resident_a` | L1 规划（59-60）、装载一次（109-110）、K 循环跳过（117）、L1→L0A 偏移（143）；断言（40） |
| 26 | `kResidentB` | 路径开关 | B 常驻 L0B（同 n 的 Tile 间复用） | `config.resident_b` | `reuse_b` 逻辑（93、118、144-145）；断言（40、45） |

读表时的四个观察：

1. **只有最后三个参数有默认值**（`= false`，第 25 行）——它们是"可选优化路径"的开关，关掉时走最通用的路径；其余 23 个参数没有默认值，JIT 生成代码总是显式填满 26 个实参。
2. **参数类型不止整数**：有枚举（`Major`、`GemmType`、两个 L2 模式）、类型（`cd_dtype_t`）、模板模板参数（`epilogue_operator_t`）——所以枚举值能以 `GemmType::Normal` 这样的字符串直接渲染进生成代码（`csrc/utils/format.hpp` 用 magic_enum 注册了 formatter）。
3. **每个参数都能在函数体内找到消费点**。模板参数表就是 kernel 的"编译期配置面"：换任何一个值，生成的就是另一份机器码（`~/.dj` 缓存按代码内容区分）。
4. **三个 Stage 参数对应三段流水衔接**（第 12 天的 `GemmConfig` 注释）：`kNumL1Stages` 衔接 MTE2↔MTE1、`kNumL0Stages` 衔接 MTE1↔Cube、`kNumEpilogueStages` 衔接 AIC→AIV——它们共同决定第 5 天数据流图里每一段"飞行中的数据"有几份。

### 3.3 编译期 shape 与运行期 shape 的关系

三个 Shape 参数的取值机制（Host 侧，`csrc/jit_kernels/gemm_config.hpp:17-21`）：

```cpp
// Return dim value if name is in compiled_dims, else 0 (meaning "use runtime value").
static uint32_t get_compiled_dim(uint32_t dim, char name, const std::string& compiled_dims) {
    for (const char& c : compiled_dims)
        if (c == name) return dim;
    return 0;
}
```

设备侧的"择一"（第 47-49 行）：

```cpp
uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime;
uint32_t shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n_runtime;
uint32_t shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k_runtime;
```

要点逐条展开：

1. **运行期参数总是传**。`jit->launch` 无条件传 `desc.m/n/k`（`csrc/jit_kernels/bf16_gemm.hpp:77`）；编译期常量化时设备侧直接忽略对应的 runtime 值。也就是说 `SHAPE_M=0` 的 kernel 二进制可以服务任意 M，而 `SHAPE_M=128` 的实例只对 M=128 正确——Host 保证只在维度匹配时才生成该实例（生成代码字符串不同，缓存天然分开）。
2. **默认策略跟随"稳定维度"**。`bf16_gemm_nt/nn` 默认 `compiled_dims="nk"`、`bf16_gemm_tn/tt` 默认 `"mn"`（`csrc/apis/gemm.hpp:635/641/647/653`）。推理场景里 N/K 是固定权重形状、M 是变化的 token 数：常量化 N/K 让特化收益落在稳定维度上，避免每个 M 都触发重编译。tn/tt 的 A 逻辑上是 `[K,M]`，稳定轴换成 M/N，故默认 `"mn"`。用户可在 Python 侧显式传 `compiled_dims` 改变这一选择。
3. **编译期 K 解锁特殊路径**。第 40 行断言：

   ```cpp
   static_assert(!(kResidentA || kResidentB) || (kGemmType == GemmType::Normal && SHAPE_K));
   ```

   resident 路径不仅要求 Normal，还要求 `SHAPE_K != 0`。原因在第 59 行：resident A 要为**整条 K** 规划 L1 缓冲（`aligned(SHAPE_K, get_frac_k<bfloat16_t>())`），这个大小必须是编译期常量才能做 `constexpr` 地址规划。Host 侧同样只在 K 编译期常量化时才尝试 resident（`csrc/jit_kernels/heuristics.hpp:544-546` 的 `compiled_k` 条件）。第 45 行的 `SHAPE_K <= kNumL0Stages * MAD_K` 也是同理——用编译期 K 与 L0B 容量做静态检查。
4. **同一个 GEMM、不同的 `compiled_dims` = 不同的 kernel 实例**。以 M=128、N=64、K=256 的 `bf16_gemm_nt` 为例：

| `compiled_dims` | SHAPE_M/N/K | 效果 |
| :-- | :-- | :-- |
| `"nk"`（默认） | 0, 64, 256 | M 运行期；resident 可用（K 已常量化） |
| `"mnk"` | 128, 64, 256 | 全常量；M 变化即重编译 |
| `""`（空串） | 0, 0, 0 | 全运行期；resident 路径不可用（第 40 行断言会拒绝 resident 实例化） |

一句话总结：**编译期 shape 买到的是"特化 + 静态检查 + 特殊路径资格"，运行期 shape 买到的是"免重编译的灵活性"，`compiled_dims` 是用户在这两者之间的旋钮。**

### 3.4 模板参数如何被消费：从参数表到片上规划

第 47-79 行是模板参数的"集中消费区"。逐段看：

**（a）派生常量（第 51 行）**

```cpp
constexpr uint32_t num_mad_cd_stages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);
```

一个 BLOCK Tile 内含多少个 MAD 结果块——L0C 要按这个数量开槽轮转，让"算下一个 MAD tile"与"写出上一个 tile"重叠（第 12 天 3.3 节的两级切分）。

**（b）Epilogue 对象（第 53-57 行）**

```cpp
const Epilogue<cd_dtype_t, epilogue_operator_t, kWithAccumulation,
               BLOCK_M, BLOCK_N, MAD_M, MAD_N, kNumEpilogueStages, 2,
               kL2CtrlStoreCd> epilogue{epilogue_args};
static_assert(!kDirectStore || std::is_same_v<typename decltype(epilogue)::operator_t,
                                              epilogue::operators::Identity<...>>);
```

注意参数的"打包复用"：kernel 模板里的 7 个参数（`cd_dtype_t`、`epilogue_operator_t`、`kWithAccumulation`、`BLOCK_M/N`、`MAD_M/N`、`kNumEpilogueStages`、`kL2CtrlStoreCd`）原样转发给 `Epilogue`（`deep_gemm/include/deep_gemm/epilogue.hpp:7-15`），后者再打包成 `epilogue::Params` 传给算子（`epilogue/operators.hpp:59-68`）。算子（`Identity`/`ScaleByAlpha`/`AddBias`/`StochasticRoundToBF16`/`QuantizeToFP8`，operators.hpp:120/125/161/225/338）由此拿到 BLOCK/MAD 尺寸规划自己的 UB 布局。第 56-57 行断言：direct store 只允许 Identity 算子（L0C 直写 GM，没有 UB 后处理的机会）。

**（c）片上缓冲的 `constexpr` 规划（第 59-64 行，今天的重点预览，第 16 天展开）**

```cpp
constexpr l1_ptr<bfloat16_t, kMajorA> l1a(BLOCK_M, kResidentA ? aligned(SHAPE_K, get_frac_k<bfloat16_t>()) : BLOCK_K, 0);
constexpr l1_ptr<bfloat16_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kResidentA ? 1 : kNumL1Stages));

constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);
constexpr l0b_ptr<bfloat16_t> l0b(MAD_N, MAD_K);
constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
```

这五行就是"为什么这些参数必须是模板参数"的直接答案：

- `l1_ptr` 等是**带形状的片上指针**（`ascend/tile_ptr.hpp:93-97`）：`operand_ptr` 存 `addr` 与逻辑形状 `(shape_mn, shape_k)`，`size_per_stage() = shape_mn × shape_k × sizeof(dtype)`（tile_ptr.hpp:37），`offset(n)` 前进 n 级（38）。声明成 `constexpr` 意味着**地址算术在编译期完成**——片上存储基址固定，BLOCK/MAD/stages 全是常量，地址可直接折叠成立即数。
- L1 被划成"A 区 + B 区"：A 区是 `kNumL1Stages` 级、每级 `BLOCK_M × BLOCK_K × 2` 字节；B 区紧跟其后（`l1a.offset(kNumL1Stages)`），同样 `kNumL1Stages` 级、每级 `BLOCK_N × BLOCK_K × 2` 字节。`kResidentA` 时 A 区变成**一整个大槽**：深度从 `BLOCK_K` 换成 `aligned(SHAPE_K, 16)`（`get_frac_k<bfloat16_t>() = 32/sizeof(bfloat16_t) = 16`，common.hpp:73-79），且 B 区基址改用 `l1a.offset(1)`。
- L0A/L0B 按 MAD 尺寸开 `kNumL0Stages` 级；L0C 按 `(MAD_M, MAD_N)` 开 `num_mad_cd_stages` 级，元素类型固定 `float`（FP32 累加，第 4 天）。

```text
L1 (512KB)                              L0A (64KB)    L0B (64KB)    L0C (256KB)
┌───────────────┬───────────────┐      ┌─────────┐   ┌─────────┐   ┌─────────┐
│ l1a           │ l1b           │      │ l0a     │   │ l0b     │   │ l0c     │
│ ×kNumL1Stages │ ×kNumL1Stages │      │×kNumL0  │   │×kNumL0  │   │×num_mad │
│ (BLOCK_M×     │ (BLOCK_N×     │      │ (MAD_M× │   │ (MAD_N× │   │_cd_stag │
│  BLOCK_K)     │  BLOCK_K)     │      │  MAD_K) │   │  MAD_K) │   │(MAD_M×  │
└───────┬───────┴───────────────┘      │         │   │         │   │ MAD_N)  │
        └─ l1b 基址 = l1a.offset(kNumL1Stages)    └─────────┘   └─────────┘   └─────────┘
```

**（d）11 条 `static_assert`：编译期守门员**

| 行号 | 断言 | 守护的约束 |
| :-- | :-- | :-- |
| 38 | direct + 累加 → `cd_dtype_t` 必须是 `float` | 直写 + 原子加只支持 FP32 |
| 40 | resident → Normal 且 `SHAPE_K != 0` | resident 的 L1/L0 规划需要编译期 K |
| 42、43 | `kMajorA/B ∈ {K, MN}` | 布局枚举合法性 |
| 45 | `kResidentB` → `SHAPE_K ≤ kNumL0Stages × MAD_K` 且 `BLOCK_N == MAD_N` | 整个 B Tile 必须装进 L0B 的多级缓冲 |
| 56-57 | direct → epilogue 算子必须是 Identity | 直写没有 UB 后处理 |
| 66 | `l1b.offset(kNumL1Stages) ≤ 512KB` | L1 不溢出 |
| 67、68 | `l0a/l0b.offset(kNumL0Stages) ≤ 64KB` | L0A/L0B 不溢出 |
| 69 | `l0c.offset(num_mad_cd_stages) ≤ 256KB` | L0C 不溢出 |
| 70 | `BLOCK_K % MAD_K == 0` | K 块能被 MAD 整除切分 |

它们与第 12 天 heuristics 的硬约束（`make_gemm_candidate` 的 8 处 `return std::nullopt`）互为镜像：Host 保证不生成非法配置，Device 断言兜底——即使有人手写实例化，非法组合也在编译期爆掉而不是跑飞。

**（e）Scheduler 与派生标志（第 72-79 行）**

```cpp
Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumCores, kAlignment> scheduler(block_idx, shape_m, shape_n,
                                                                        grouped_layout, num_groups);
constexpr bool is_m_grouped = kGemmType == GemmType::MGroupedContiguousWithPsumLayout;
constexpr bool is_k_grouped = ...;  constexpr bool is_batched = ...;  constexpr bool is_grouped = ...;
```

`kGemmType/BLOCK_M/N/kNumCores/kAlignment` 五个参数转发给调度器（`deep_gemm/include/deep_gemm/scheduler.hpp:42-48`），后者用 `kNumCores` 把 `block_idx` 映射成输出 Tile（scheduler.hpp:142 的 `(++current_iter) * kNumCores + core_id`）。`kGemmType` 再派生出四个编译期标志，供函数体各处 `if constexpr` 消除分支。

### 3.5 函数入口：9 个运行期参数

模板参数之后是 9 个运行期形参（第 28-33 行），与 `jit->launch` 的 9 个实参一一对应：

| # | 设备形参 | 类型 | Host 实参（`csrc/jit_kernels/bf16_gemm.hpp:74-77`） | 含义 |
| :--: | :-- | :-- | :-- | :-- |
| 1 | `gm_a` | `gm_ptr<bfloat16_t, kMajorA>` | `gm_ptr<uint8_t>{outer_stride_a, a_ptr, stride_batch_a}` | A 的地址三元组 |
| 2 | `gm_b` | `gm_ptr<bfloat16_t, kMajorB>` | `gm_ptr<uint8_t>{outer_stride_b, b_ptr, stride_batch_b}` | B 的地址三元组 |
| 3 | `gm_d` | `gm_ptr<cd_dtype_t>` | `gm_ptr<uint8_t>{outer_stride_d, d_ptr, stride_batch_d}` | D 的地址三元组 |
| 4-6 | `shape_m/n/k_runtime` | `uint32_t` | `desc.m/n/k` | 运行期 shape（`SHAPE≠0` 时被忽略） |
| 7 | `grouped_layout` | `__gm__ int32_t*` | `desc.grouped_layout` | 组前缀和数组（grouped/batched 用） |
| 8 | `num_groups` | `uint32_t` | `desc.num_groups` | 组数 / batch 数 |
| 9 | `epilogue_args` | `epilogue::Args` | `epilogue_class->make_epilogue_operator_args(desc.n)` | alpha / bias 指针等运行期后处理状态 |

两个说明：

- `gm_ptr` 只装 `(addr, stride_outer, stride_batch)` 三个字（common.hpp:138-148），Host 侧用 `uint8_t` 实例化、设备侧按自己的 dtype/major 实例化，成员布局相同所以按位传递即可；`index()/group()/batch()` 等地址计算发生在设备侧（第 18 天精读）。
- `epilogue::Args`（common.hpp:95-103）是纯运行期状态：`alpha`（浮点乘子）、`bias`（偏置指针）、`sfd/sfd_stride/shape_n`（FP8 输出的 scale 因子）。BF16 Identity 场景下全是默认值。

**分界原则**（第 12 天已给出，今天在设备侧看到实体）：影响"生成哪份代码"的进模板参数（tiling、流水、路径开关、编译期 shape、类型）；每次调用都可能不同的（地址、运行期 shape、组信息、后处理数值）走 launch 参数。**模板参数创造特化，运行期参数创造复用。**

### 3.6 `kDirectStore` 的初步用途

普通路径的写回是三跳：`L0C →(AIC, FIX)→ UB →(AIV, MTE3)→ GM`，AIV 在 UB 里做后处理（转 BF16、bias、量化……）。`kDirectStore` 把它压缩成一跳：

```cpp
if (k_blk_idx == num_k_blocks - 1) {
    if constexpr (kDirectStore) {
        const auto d = gm_d.batch(batch_idx).group(...);
        copy_l0c_to_gm(d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(),
                       l0c_tile, eff_mad_m, eff_mad_n, d.stride_outer, kL2CtrlStoreCd);  // 156-159
    } else {
        epilogue.aic_store(l0c_tile, s_epi, eff_mad_m, eff_mad_n);                      // 161
    }
}
```

打开它之后 kernel 的四点变化：

1. **`__mix__(1, 0)`**（第 27 行）：不再需要 AIV，每个 block 只有 1 个 AIC；
2. **AIV 分支开头直接 `return`**（第 174 行）：向量核空转；
3. **跳过 `epilogue.aic_wait_free`**（第 129 行的 `!kDirectStore`）：没有 UB stage 就没有"AIC 等 AIV 腾槽"的同步；
4. **累加语义改用原子加**（第 82 行 `asc_set_atomic_add_float()`；第 37-38 行断言累加时 `cd_dtype_t` 必须是 `float`）：没有 UB 里的读-改-写，`D += ...` 只能靠写回指令的原子加完成。

前置条件由 Host 的 `can_direct_store`（heuristics.hpp:521-533）把关：Normal、Identity epilogue、M/N 对齐、D 指针及行距 32B 对齐、Tile 数 ≤ 核数（或行长满足 L2 sector 对齐）等一串条件，且要成本模型点头（`refine_gemm_pipeline`，heuristics.hpp:539-543）。**适用直觉**：输出小、Tile 少、无需后处理时，省掉 L0C→UB→GM 的中转更划算。深入对比留到第 26 天。

### 3.7 `kResidentA` / `kResidentB` 的初步用途

普通路径里，每个输出 Tile 的 K 循环每前进一个 `BLOCK_K` 都要重新搬 A/B（第 117-118 行），换一个输出 Tile 又从头再来。当矩阵某一维很小时（比如 M 很小的 decode 场景、或 N 只有一两个 Tile），这些搬运是纯粹的重复劳动——resident 开关让操作数"住下来"：

**`kResidentA`：A 常驻 L1。**

- L1 规划变化（第 59-60 行）：`l1a` 深度从 `BLOCK_K` 变为 `aligned(SHAPE_K, 16)`——**一整个槽装下 A 的全部 K**；`l1b` 基址相应前移到 `l1a.offset(1)`；
- 装载时机（第 109-110 行）：进入 K 循环**之前**一次性 `copy_gm_to_l1(l1a[0], a.index(m_idx, 0), actual_m, shape_k, ...)`；
- K 循环内（第 117 行）：`if constexpr (!kResidentA) copy_gm_to_l1(l1a[s], ...)`——这句被 `if constexpr` **编译期删除**；
- L1→L0A 的读取偏移（第 143 行）：`copy_l1_to_l0a(l0a_tile, l1a[kResidentA ? 0 : s], m_mad_idx, (kResidentA ? k_blk_idx * BLOCK_K : 0) + k_mad_idx)`——从常驻槽 `l1a[0]` 里按 K 位置取数；
- 跨 Tile 复用（第 92、109 行）：`reuse_a = (m_idx == last_m_idx)`——同一核连续分到相同 m_idx 的 Tile 时，连最初那次装载也省掉。

**`kResidentB`：B 常驻 L0B。** 约束更紧（第 45 行）：`SHAPE_K ≤ kNumL0Stages × MAD_K` 且 `BLOCK_N == MAD_N`——即整个 `[BLOCK_N, K]` 的 B Tile 恰好装进 L0B 的多级缓冲。满足时（第 93、118、144-145 行）：`reuse_b = (n_idx == last_n_idx)`，同一核连续处理相同 n_idx 的 Tile 时跳过 B 的 GM→L1 与 L1→L0B 搬运，B 留在 L0B 里被反复乘。

**Host 侧何时打开**（heuristics.hpp:546-565，BF16 分支）：仅当 Normal 且 K 编译期常量化（`compiled_k`）；`resident_a` 要求整块 A 装得进 L1 且 `M ≤ BLOCK_M`（M 方向只有一个 Tile，A 可被所有 N-Tile 复用）；`resident_b` 要求 N 只有一个 Tile 块、多 wave、`BLOCK_N == MAD_N`、`K ≤ MAD_K × num_l0_stages`。每次尝试都要成本模型比较后才保留。

**与第 40 行断言的关系再强调一次**：resident 的本质是"用编译期已知的 K 范围换静态 L1/L0 规划"，所以 `SHAPE_K=0`（K 运行期）时该路径根本不存在。这就是 3.3 节"编译期 shape 的额外权限"的具体例子。设备行为细节留到第 26 天。

## 四、源码导读

今天的目标是**把 3.2 分类表的每一行落到真实代码上**。每站 5～10 分钟，行号未注明文件时均指 `deep_gemm/include/deep_gemm/bf16_gemm.hpp`。

### 站点 1：模板声明与函数入口（第 9-34 行，精读）

- 逐个抄下 26 个参数名，按 3.2 的类别画竖线分组；注意第 25 行是**唯一有默认值**的一行；
- 第 27 行：数一数 `__mix__` 的两个实参，思考 `kDirectStore ? 0 : 2` 里 2 的去向（提示：第 54 行 `Epilogue` 的第 9 个实参）；
- 第 28-33 行：9 个形参与 3.5 的表格逐行对号。

### 站点 2：`constexpr` 规划与断言（第 35-79 行，精读）

- 第 47-49 行：择一表达式；第 51 行：`num_mad_cd_stages`；
- 第 53-57 行：`Epilogue` 的 10 个模板实参各来自哪个 kernel 模板参数；
- 第 59-64 行：五个片上指针——写出每个的 `(shape_mn, shape_k)` 与 `size_per_stage` 表达式（B-2 演算的公式来源）；
- 3.4（d）的断言表逐条在源码里找到；问自己：第 66 行为什么是 `l1b.offset(kNumL1Stages)` 而不是 `l1a.offset(...)`？（A 区 + B 区的总大小。）

### 站点 3：函数体的参数消费点（第 81-193 行，扫读，只找参数）

今天**不**读懂逻辑，只做"寻宝"：在函数体里找出以下模板参数的全部出现位置——`kNumL1Stages`（60、66、83、112）、`kNumL0Stages`（45、67、68、84、133）、`kResidentA`（40、59、60、109、117、第 143 行内 2 处）、`kResidentB`（40、45、118、144）、`kDirectStore`（82、129、156、174）、`kL2CtrlA/B`（110、117、118）、`kGemmType` 的派生标志（95、99-100、157、175-176）、`epilogue.kNumStages`（164，即 `kNumEpilogueStages × 2`，epilogue.hpp:24）。每个出现处写一行"它在过滤/计算什么"。

### 站点 4：`deep_gemm/include/deep_gemm/common.hpp`（枚举与常量）

- 第 82 行 `Major`、87-92 行 `GemmType`、95-103 行 `epilogue::Args`——模板参数与运行期参数用到的全部枚举/结构；
- 第 118-135 行：Host 侧镜像的 `asc_load/store_l2_cache_mode` 枚举（NORMAL_FIRST_VICTIM / NOTALLOC_KEEP / NOTALLOC_CLEAN…）——控制该次访存是否在 L2 分配缓存行及脏行处理方式，具体取值含义第 50 天结合性能再谈；
- 第 51、54-63 行：`kNumAICores=32`、L1/L0A/L0B/L0C/UB 容量、`kNumPipeEvents=8`、`FRAC_MN=16`——断言的分母。

### 站点 5：`deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp`（形状化指针）

- 第 18-56 行 `operand_ptr`：`size_per_stage()`（37）与 `offset(n)`（38）——第 59-60 行 L1 规划的全部算术；
- 第 93-97 行：`l1_ptr/l0a_ptr/l0b_ptr/l0c_ptr/ub_ptr` 别名——不同地址空间（`__cbuf__`/`__ca__`/`__cb__`/`__cc__`/`__ubuf__`）对应不同片上存储；
- 第 58-91 行 `result_ptr`：L0C/UB 用的二维结果指针。

### 站点 6：`deep_gemm/include/deep_gemm/epilogue.hpp`（Epilogue 的参数表）

- 第 7-15 行：`Epilogue` 的 10 个模板参数——与站点 2 第 53-55 行的实参对号；第 24 行 `kNumStages = kNumStagesPerAIV × kNumAIVs`；
- 第 28-35 行：又一组 `static_assert`（UB 容量、MAD_N 对齐、stage 数下限）——Epilogue 侧的守门员；
- `deep_gemm/include/deep_gemm/epilogue/operators.hpp:119-447`：五个算子的名字与 `ub_dtype_t/kPipe` 差异，今天只看结构。

### 站点 7：`csrc/jit_kernels/bf16_gemm.hpp`（生成侧，验证对号）

- 第 18-42 行模板串 + 43-66 行实参表：26 个实参与 3.2 表格的"Host 侧来源"列逐一对号——**这是验证分类表正确性的天然练习**；
- 第 73-77 行：launch 的 9 个实参与 3.5 表格对号。

### 站点 8：`csrc/apis/gemm.hpp` 与 `csrc/jit/jit.hpp`（两个来源补全）

- apis/gemm.hpp:176：`.num_cores = runtime->get_num_sms()`——`kNumCores` 的源头（查询设备 AI Core 数；`_C.set_num_sms` 可覆盖，第 52 天的实验因子）；
- apis/gemm.hpp:632-635：`bf16_gemm_nt` 的 `py::arg("compiled_dims") = "nk"`；
- jit.hpp:59：`kMkAlignment = 256`——`kAlignment` 的源头。

**阅读检查点**（能回答再进入实践）：

1. 26 个参数里哪三个有默认值？为什么恰好是它们？
2. `l1b` 的起始地址表达式是什么？`kResidentA` 打开时它怎么变？
3. 第 40 行与第 45 行断言分别对哪个路径开关设卡？各自要求什么？
4. `kNumEpilogueStages` 与 `epilogue.kNumStages` 是什么关系？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：把打印的配置翻译成模板实参（约 20 分钟）**。

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day15_run1.log
```

从输出中抄下**一行** `GemmDesc(m=..., n=..., k=..., ..., gemm_type=Normal) GemmConfig(block=...x...x..., mad=...x...x..., l1_stages=..., l0_stages=..., epi_stages=..., ..., direct_store=..., resident=.../.../..., ...)`，然后：按 `csrc/jit_kernels/bf16_gemm.hpp:43-66` 的顺序把它翻译成 26 个模板实参的列表（Major 来自 GemmDesc 的 major_a/major_b，epilogue 算子来自 GemmDesc 的 epilogue_type，`kAlignment=256`，`kNumCores` 来自 launch），填进第六节的分类表。逐项核对：`l1_stages ≤ 8`？`block_m×block_n×4 ≤ 256KB`？

**步骤 2：`compiled_dims` 与 kernel 实例（约 15 分钟）**。

```bash
DG_PRINT_CONFIGS=1 DG_JIT_PRINT_COMPILER_COMMAND=1 python - <<'EOF' 2>&1 | tee /tmp/day15_run2.log
import torch, torch_npu  # noqa
import deep_gemm

M, N, K = 128, 4096, 256
a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')
b = torch.randn(N, K, dtype=torch.bfloat16, device='npu')
d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')

for cd in ("nk", "mnk", ""):
    deep_gemm.bf16_gemm_nt(a, b, d, compiled_dims=cd)
torch.npu.synchronize()
EOF
```

观察三次调用的行为：`"nk"` 与 `""` 的 `GemmConfig` 是否相同？Bisheng 编译命令出现几次（提示：代码字符串不同才会重编译；`"nk"` 与 `""` 的 SHAPE 实参不同，`"mnk"` 又不同）？对照 3.3 的表格解释。

**步骤 3：寻找 resident 配置（约 10 分钟）**。用步骤 2 的 shape（M=128、K=256 编译期常量化）观察打印的 `resident=.../.../...` 是否出现 `1/0/0`（resident_a）。若没有，按 3.7 的条件清单推测原因（整块 A 是否装得进 L1？成本模型是否否决？），并把 `M` 改成 256、`K` 改成 512 再试一次记录差异。

**步骤 4（可选）：换 epilogue 算子（约 5 分钟）**。给 `bf16_gemm_nt` 传 `alpha=0.5` 再跑一次：`epilogue_operator_t` 变成 `epilogue::operators::ScaleByAlpha`，又是一份新 kernel。体会"类型也是模板参数"。

### B. 无 Ascend 环境：静态替代任务

**B-1：手工翻译 26 个模板实参（约 15 分钟）。** 沿用第 12 天 B-1 的设定：BF16 Normal，`M=128, N=64, K=256`，`bf16_gemm_nt`（`compiled_dims="nk"`，A/B 均 K-major，Identity epilogue，无累加），`num_cores=32`；**假设**启发式最终选中候选 `block=128×512×128`、`mad=128×256×64`、`l1_stages=3`、`l0_stages=2`、`epi_stages=2`、`l2: A/B=NORMAL_FIRST_VICTIM、store=NOTALLOC_CLEAN`、三个开关全关。写出完整的实例化：

```cpp
bf16_gemm_impl<
    Major::K, Major::K,
    0, 64, 256,
    128, 512, 128,
    128, 256, 64,
    3, 2, 2, 32,
    asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    asc_store_l2_cache_mode::NOTALLOC_CLEAN,
    false, GemmType::Normal, bfloat16_t, epilogue::operators::Identity,
    256, false, false, false>
```

逐个标注每个实参的类别（3.2 的表）与来源（desc / desc×compiled_dims / config / 全局常量）。

**B-2：片上缓冲演算（约 15 分钟）。** 用 B-1 的配置核对第 59-70 行（公式：`size_per_stage = shape_mn × shape_k × sizeof(dtype)`，BF16 占 2 字节，`offset(n) = 基址 + n × size_per_stage`）：

| 缓冲 | 形状 | 每级字节 | 级数 | 断言 | 计算 | 容量 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `l1a` | (128, 128) | 32768 | 3 | — | A 区 = 3×32768 = 98304 | — |
| `l1b` | (512, 128) | 131072 | 3 | 66 | 98304 + 3×131072 = **491520** | ≤ 524288 ✓ |
| `l0a` | (128, 64) | 16384 | 2 | 67 | 2×16384 = **32768** | ≤ 65536 ✓ |
| `l0b` | (256, 64) | 32768 | 2 | 68 | 2×32768 = **65536** | ≤ 65536 ✓（压线） |
| `l0c` | (128, 256) | 131072（float×4B） | 2 | 69 | 2×131072 = **262144** | ≤ 262144 ✓（压线） |

其中 `l0c` 的级数 = `num_mad_cd_stages = (128/128)×(512/256) = 2`。另核对断言 70：`128 % 64 == 0` ✓。（与第 12 天 B-1 第 7 步的数字完全一致——那边的启发式演算与这边的设备侧断言在本例中严丝合缝。）

**B-3：resident 变体演算（约 15 分钟）。** 同一 shape，仍用 `compiled_dims="nk"`（`SHAPE_K=256`），**假设** `refine_gemm_pipeline` 打开了 `resident_a`（对照 heuristics.hpp:548-555：`a_bytes = 128×align(256,16)×2 = 65536 < 512KB`，重算 `num_l1_stages = min(8, (524288−65536)/131072) = 3 > 2`，且 `M=128 ≤ block_m=128`，条件全满足）：

1. 重写第 59-60 行：`l1a` 形状变为 (128, **256**)，`size_per_stage = 65536`，只占 **1 级**；`l1b` 基址 = `l1a.offset(1)` = 65536；
2. 核对断言 66：`65536 + 3×131072 = 458752 ≤ 524288` ✓——resident A 用一个 64KB 的大槽换掉了 3 个 32KB 的小槽，B 区多出约 32KB 余量；
3. K 循环（`num_k_blocks = ceil(256/128) = 2`）：A 在循环前搬一次（第 109-110 行），循环内第 117 行被编译掉，L1→L0A 从 `l1a[0]` 取、k 偏移 = `k_blk_idx×128 + k_mad_idx`（第 143 行）；
4. 检查断言 45 能否让 `kResidentB` 也打开：`SHAPE_K=256 ≤ kNumL0Stages×MAD_K = 2×64 = 128`？**否**（且 `BLOCK_N=512 ≠ MAD_N=256`）——解释为什么：整个 B Tile 要装进 L0B 双缓冲，K=256 超出了这组 MAD 尺寸下的容量。

**B-4：闭卷填写分类表（约 15 分钟）。** 合上教程，凭记忆重画 3.2 的表（类别 + 每类参数名 + 默认值有无），再对照订正——订正后的表就是当天产出。

**B-5：源码寻宝（约 10 分钟）。** 不看 3.4/3.7 的行号提示，在 `bf16_gemm.hpp` 里独立找出并数一数：`kNumL1Stages`（全文 5 处 = 模板声明 1 + 函数体 4：60、66、83、112）、`kNumL0Stages`（全文 6 处 = 声明 1 + 函数体 5：45、67、68、84、133）、`kResidentA`（函数体 7 处：40、59、60、109、117、143 行内 2 处）、`kDirectStore` 在 AIV 分支的唯一一处（第 174 行）。数错就重读站点 3。

## 六、当天产出

学习计划要求今天的产出是**模板参数分类表**。按下述模板填写（B-1/B-2/B-3 的演算附在后面）：

```text
# DeepGEMM-Ascend 第 15 天记录：阅读 Kernel 模板参数

## A. 模板参数分类表（闭卷填写后对照 3.2 订正）
布局（2）：＿＿、＿＿
Shape（3）：＿＿、＿＿、＿＿ —— 0 的含义：＿＿
Block（3）：＿＿ —— 所在层级：＿＿，一次 ＿＿→＿＿ 搬运的粒度
MAD（3）：＿＿ —— 所在层级：＿＿，一次 ＿＿ 指令的计算块
Stage（3）：＿＿、＿＿、＿＿ —— 分别衔接 ＿＿↔＿＿、＿＿↔＿＿、＿＿→＿＿
Core（1）：＿＿ —— Host 来源：＿＿，设备侧消费点：＿＿
L2（3）：＿＿、＿＿、＿＿ —— 分别挂在哪次访存上
Epilogue/语义（4）：＿＿、＿＿、＿＿、＿＿
对齐（1）：＿＿ —— 值：＿＿，来源：＿＿
路径开关（3，均有默认值 ＿＿）：＿＿、＿＿、＿＿

## B. 消费点地图
- l1b 基址表达式：＿＿；kResidentA 打开后变为：＿＿
- num_mad_cd_stages = ＿＿；l0c 元素类型固定为 ＿＿（为什么：＿＿）
- 11 条 static_assert 中，容量类 4 条守护 ＿＿/＿＿/＿＿/＿＿；
  resident 类 2 条要求 ＿＿、＿＿

## C. 手工演算记录（B-1～B-3）
- 26 个实参逐个翻译（附类别标注）
- 普通配置片上用量：L1=＿＿ L0A=＿＿ L0B=＿＿ L0C=＿＿
- resident_a 变体：l1a 形状=＿＿ l1b 基址=＿＿ 总用量=＿＿
- kResidentB 为何不可用：＿＿

## D. 实验记录（有环境时）
- DG_PRINT_CONFIGS 抄录行 + 翻译出的 26 实参
- 三种 compiled_dims 下编译命令出现次数与原因
- resident 出现/未出现的原因分析

## E. 遗留问题（交给后续天）
- Scheduler 的 swizzle 与 get_next_block（第 17 天）
- gm_ptr/l1_ptr 的地址计算细节（第 18 天）
- set_flags/asc_sync 流水线同步（第 23 天）
- direct_store / resident 的完整行为与约束（第 26 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. 26 个模板参数分为哪几类？每类各几个参数？哪三个有默认值？
2. `SHAPE_M=0` 是什么意思？设备侧第 47-49 行如何处理？`bf16_gemm_nt` 默认把哪些维度常量化，为什么？
3. 为什么 `kResidentA/kResidentB` 要求 `kGemmType == Normal && SHAPE_K != 0`？给出断言行号和一条源码证据（提示：第 59 行的 L1 规划）。
4. `kResidentB` 受哪两个形状约束（第 45 行）？各自的物理原因是什么？
5. `kDirectStore` 打开后 kernel 有哪四点变化？
6. `l1b` 的起始地址由哪两个模板参数决定？写出表达式。`l0c` 的级数由什么决定？
7. `num_mad_cd_stages` 等于什么？`epilogue.kNumStages` 又等于什么（`kNumEpilogueStages` 与它的关系）？
8. `kNumCores` 的值从哪里来（追踪到 Python 可影响的接口）？设备侧哪两处消费它？
9. 9 个运行期参数分别是什么？"分界原则"是什么？
10. （思考）如果把 `BLOCK_K` 从模板参数改成运行期参数，会损失什么？`compiled_dims` 又是在什么之间做权衡？
11. `kAlignment` 的值和来源是什么？它与 m-grouped 的 `BLOCK_M` 有什么静态关系（提示：scheduler.hpp:146）？

**参考答案：**

1. 布局 2（kMajorA/B）、Shape 3（SHAPE_M/N/K）、Block 3（BLOCK_M/N/K）、MAD 3（MAD_M/N/K）、Stage 3（kNumL1Stages/kNumL0Stages/kNumEpilogueStages）、Core 1（kNumCores）、L2 3（kL2CtrlA/B、kL2CtrlStoreCd）、Epilogue/语义 4（kWithAccumulation、kGemmType、cd_dtype_t、epilogue_operator_t）、对齐 1（kAlignment）、路径开关 3（kDirectStore、kResidentA、kResidentB）。有默认值的是最后三个（`= false`，第 25 行）——它们是可关闭的优化路径开关。
2. 0 表示该维度留给运行期：`shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime`（47-49 行），`shape_m_runtime` 来自 launch 参数 `desc.m`。`bf16_gemm_nt` 默认 `compiled_dims="nk"`（apis/gemm.hpp:635）：推理中 N/K 是固定权重形状、M 是变化的 token 数，常量化 N/K 让特化收益落在稳定维度、避免每个 M 都重编译；`SHAPE_M=0` 的实例可服务任意 M。
3. 断言在第 40 行：`!(kResidentA || kResidentB) || (kGemmType == GemmType::Normal && SHAPE_K)`。源码证据：第 59 行 resident A 的 L1 槽深度直接用 `aligned(SHAPE_K, get_frac_k<bfloat16_t>())`——要为整条 K 做 `constexpr` 地址规划，K 必须是编译期常量；Host 侧 heuristics.hpp:544-546 同样只在 `compiled_k` 非零时尝试 resident。
4. `SHAPE_K <= kNumL0Stages * MAD_K` 且 `BLOCK_N == MAD_N`（第 45 行）。物理原因：整个 `[BLOCK_N, K]` 的 B Tile 必须恰好装进 L0B 的多级缓冲（`kNumL0Stages` 级、每级 `MAD_N × MAD_K`），`BLOCK_N == MAD_N` 保证 N 方向不拆分，`K ≤ 级数×MAD_K` 保证 K 方向装得下。
5. ① `__mix__(1, 0)`：不再配 AIV（第 27 行）；② AIV 分支开头 `return`（174）；③ 跳过 `epilogue.aic_wait_free`（129 的 `!kDirectStore`）；④ 累加时改用 `asc_set_atomic_add_float()`（82；38 行断言累加必须 `cd_dtype_t=float`）。写回路径变为 `copy_l0c_to_gm` 直写（156-159），且只允许 Identity 算子（56-57）。
6. `l1b` 基址 = `l1a.offset(kResidentA ? 1 : kNumL1Stages)`（第 60 行），即由 `kNumL1Stages`（或 resident 时的 1）与 `l1a` 的每级大小（`BLOCK_M × (kResidentA ? aligned(SHAPE_K,16) : BLOCK_K) × 2`）决定。`l0c` 级数 = `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`（第 51 行）。
7. `num_mad_cd_stages` = `(BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`，即一个 BLOCK 内 MAD 结果块的个数，L0C 按它开槽轮转。`epilogue.kNumStages = kNumStagesPerAIV × kNumAIVs = kNumEpilogueStages × 2`（epilogue.hpp:24，`kNumAIVs` 即第 54 行硬编码的 2）。
8. `desc.num_cores = runtime->get_num_sms()`（apis/gemm.hpp:176），再经 `make_gemm_candidate` 写入 `launch_options.num_blocks`（heuristics.hpp:98）成为第 15 个模板实参；Python 侧 `_C.set_num_sms(n)` 可覆盖（apis/runtime.hpp:17-19）。设备侧消费点：`Scheduler` 构造（73，`block_idx` 映射）与 `aiv_zero_empty_groups<kNumCores, kAlignment>`（176）。
9. 三个 `gm_ptr`（A/B/D 的地址三元组）、`shape_m/n/k_runtime`、`grouped_layout`、`num_groups`、`epilogue::Args`。分界原则：影响"生成哪份代码"的进模板参数（tiling/流水/路径开关/编译期 shape/类型），每次调用都可能不同的走 launch 参数（地址/运行期 shape/组信息/后处理数值）。
10. 开放题，参考方向：损失 `constexpr` 片上地址规划（第 59-64 行折叠成立即数的算术变成运行期计算）、`static_assert` 容量检查（66-70 行无法写）、`if constexpr` 分支消除（如第 117 行）；循环边界常量化带来的展开与调度优化也随之消失。`compiled_dims` 在"特化收益 + 特殊路径资格（如 resident 需要 SHAPE_K）"与"免重编译的灵活性"之间权衡：常量化稳定维度（N/K），把易变维度（M）留运行期。
11. 256；`Runtime::get_mk_alignment_for_contiguous_layout()`（jit.hpp:59 的 `kMkAlignment`）。m-grouped 时 `static_assert(kAlignment == BLOCK_M)`（scheduler.hpp:146）——psum 组边界按 256 对齐（第 11 天），BLOCK_M 必须与之相等，这正是 heuristics 里 m-grouped 固定 `block_m=256` 的原因（heuristics.hpp:106）。

---

今天结束时，你的手里应该有一张"26 个模板参数 → 类别 → 来源 → 消费点"的完整分类表，并能解释编译期 shape 与运行期 shape 的边界。明天（第 16 天）深入 3.4（c）预览过的五行 `constexpr`：L1、L0A、L0B、L0C 如何通过 offset 分配、多级缓冲如何排布、容量 `static_assert` 如何守护它们。
