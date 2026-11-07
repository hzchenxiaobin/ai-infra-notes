# 第 2 周 · 第 5 天（总第 12 天）：理解 JIT 配置、编译与启动

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 2 周"项目结构与 JIT 调用链"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 50 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

第 11 天我们把用户张量逐字段翻译成了一份 `GemmDesc`——"算什么"已经说清楚了。今天回答"怎么算"：`launch_bf16_gemm`（`csrc/jit_kernels/bf16_gemm.hpp`）拿到 desc 后只做三件事——`select_gemm_config()` 推导执行配置 `GemmConfig`、用 `std::format` 生成一份模板实例化源码、交给 `jit->compile()` 编译并 `jit->launch()` 启动。我们将逐个拆开这三步：重点区分 `BLOCK_M/N/K` 与 `MAD_M/N/K` 两级切分，弄清 heuristics 的"候选生成 + 硬约束 + 成本模型 + 精修"框架，看懂生成的代码长什么样、哪些信息进了模板参数、哪些留作运行期参数，最后理解 Kernel 缓存与 dry-run 的作用。设备 kernel 内部的调度、搬运与流水线属于第 3～4 周。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 按 `launch_bf16_gemm`（`csrc/jit_kernels/bf16_gemm.hpp:16-78`）的顺序复述三步：`select_gemm_config(desc)` → `std::format` 生成 `bf16_gemm_impl<...>` 实例化代码 → `jit->compile()` + `jit->launch()`，并指出 dry-run 分支的位置与用途；
2. 按"Tile / 流水深度 / 路径开关 / L2 控制 / Launch 选项"五组说出 `GemmConfig`（`csrc/jit_kernels/gemm_config.hpp:97-121`）的全部字段含义；
3. **区分 `BLOCK_M/N/K` 与 `MAD_M/N/K`**：前者是 L1 级 Tile（GM→L1 一次搬运、K 维循环的步进），后者是一次 MAD 指令的 L0A/L0B→L0C 计算块；能写出约束 `BLOCK_K % MAD_K == 0` 与 `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`；
4. 描述 `select_gemm_config`（`csrc/jit_kernels/heuristics.hpp:592-755`）的框架：搜索范围如何确定（`m_range/n_range`、`min/max_block_k`）、`make_gemm_candidate` 负责哪些硬约束、成本模型如何剪枝、`refine_gemm_pipeline` 与 K 细化做什么；
5. 手工追踪 `make_gemm_candidate` 对一个具体 (block_m, block_n, block_k) 的推导：MAD 搜索、L1/L0/UB 各级 stage 数，并对照设备侧 `static_assert`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:66-70`）验证片上内存用量；
6. 解释模板代码生成：26 个实参分别来自 desc 还是 config，`compiled_dims` 如何通过 `get_compiled_dim` 变成 `SHAPE_M/N/K`（0 = 运行期取值），`__instantiate()` 取函数地址的技巧，`__mix__(1, kDirectStore ? 0 : 2)` 的 AIC/AIV 配比；
7. 说明 `jit->compile(name, code)` / `jit->launch(kernel, options, args...)` 的输入输出、`LaunchOptions` 里有什么、Kernel 缓存目录（`DG_JIT_CACHE_DIR`，默认 `$HOME/.dj`）的行为，以及配置选择本身**没有**缓存这一点；
8. 用 `DG_PRINT_CONFIGS=1` + `DG_JIT_PRINT_COMPILER_COMMAND=1` 观察一次真实的选择与编译，或在纸面上完成候选配置演算与代码生成填空。

**当天产出：从 `GemmDesc` 到 Kernel launch 的流程图**（模板见第六节，参考答案见 3.7 节）。

## 二、预备知识

- **第 11 天**：`GemmDesc` 的字段与来源（`get_gemm_desc` 七阶段）；`GemmDesc` 与 `GemmConfig` 的分工（"算什么" vs "怎么算"）；`compiled_dims` 字段的含义。
- **第 10 天**：binding 五步（逻辑转置 → dtype 断言 → epilogue 归一 → `get_gemm_desc` → `launch_bf16_gemm`）；`bf16_gemm_nt` 默认 `compiled_dims="nk"`、`bf16_gemm_tn/tt` 默认 `"mn"`（`csrc/apis/gemm.hpp:635/641/647/653`）。
- **第 8～9 天**：`_C` 扩展与 `csrc/` 各目录职责；`DJ_HOST_ASSERT`/`DJ_PANIC` 失败在 Python 侧是 `RuntimeError`；`develop.sh` 构建方式。
- **第 1 周概念**：`BLOCK_M/N/K` 的 Tiling 与数据复用（第 3 天）；`GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` 数据流与 MTE1/MTE2/M 流水线（第 5 天）；BF16 输入、FP32 累加（第 4 天）。
- **C++ 语法**：函数模板显式实例化（取地址强制实例化）、`std::format` 的 `{}` 占位与 `{{`/`}}` 转义、指定初始化器。能对照读懂即可。

## 三、详细讲解

### 3.1 全景：`launch_bf16_gemm` 的三步

整个文件只有 80 行（`csrc/jit_kernels/bf16_gemm.hpp`），核心是：

```cpp
static void launch_bf16_gemm(const GemmDesc& desc) {
    auto config = select_gemm_config(desc);                    // ① 选配置
    const auto code = std::format(R"(...模板...)", /*26 个实参*/); // ② 生成代码
    if (runtime->get_dry_run()) {
        jit->compile_without_load("bf16_gemm", code);          //    dry-run：只编译不加载
        return;
    }
    const auto kernel = jit->compile("bf16_gemm", code);       // ③ 编译（带缓存）
    jit->launch(kernel, config.launch_options, /*9 个运行期参数*/); // ④ 启动
}
```

文件头注释（第 13-15 行）说明它同时服务 Normal / m-grouped / k-grouped——由 `desc.gemm_type` 区分，计算语义是 NT（`D = A @ B^T`，`desc.acc` 时 `D += A@B^T`）。也就是说**一种 GEMM 类型一个 launch 函数，布局与分组信息全部藏在 desc 里**。

在完整调用链中的位置（第 13 天要闭卷画的那张图的后半段）：

```text
Python bf16_gemm_nt(a, b, d)
    │  csrc/python_api.cpp binding：逻辑转置、断言、epilogue 归一
    ▼
get_gemm_desc() ──► GemmDesc          （第 11 天）
    ▼
launch_bf16_gemm(desc)                （今天）
    ├─① select_gemm_config(desc) ──► GemmConfig
    ├─② std::format ──► 一段 C++ 源码（bf16_gemm_impl 的显式实例化）
    └─③④ jit->compile / jit->launch ──► 设备上的 bf16_gemm_impl 执行
```

### 3.2 `GemmConfig`：执行配置的字段分组

`GemmConfig` 定义在 `csrc/jit_kernels/gemm_config.hpp:97-121`，注释一句话点题："Execution configuration: how to compute (tiling, pipeline, launch)"。按用途分五组：

| 组 | 字段 | 含义 | 来源/约束（heuristics） |
| :-- | :-- | :-- | :-- |
| Tile 大小 | `block_m/n/k` | L1 级 Tile：一个输出 Tile 的 M/N 尺寸、K 维一次搬运的深度 | 主搜索循环（heuristics.hpp:626-657）；`block_m×block_n×4B ≤ L0C 256KB`（101-102） |
| Tile 大小 | `mad_m/n/k` | 一次 MAD 指令的计算块尺寸 | `make_gemm_candidate` 内搜索（122-147） |
| 流水深度 | `num_l1_stages` | L1 数据缓冲级数（MTE2↔MTE1，GM→L1 与 L1→L0 的重叠） | 由 L1 容量 512KB 除以单级 AB 字节数（166-174），≥2 |
| 流水深度 | `num_l0_stages` | L0A/L0B 缓冲级数（MTE1↔Cube） | 固定为 2（112） |
| 流水深度 | `num_epilogue_stages` | 每 AIV 的 UB epilogue 级数（AIC→AIV） | UB 容量与输出 Tile 数共同限制（185-194） |
| 流水深度 | `num_l1_sf_stages`、`sf_k_blocks` | SF（缩放因子）缓冲，**BF16 恒为 0** | 仅低精度路径（156-164） |
| 路径开关 | `direct_store` | L0C 直接写回 GM，跳过 UB epilogue | `can_direct_store` + 成本比较（521-543） |
| 路径开关 | `resident_a/b/sf` | 操作数常驻 L1/L0，跨输出 Tile 复用 | `refine_gemm_pipeline`（544-565） |
| 路径开关 | `dual_aiv_dequant` | 双 AIV 解码（仅 FP8×FP4 dequant 路径） | `refine_gemm_pipeline`（566-571）、`select_gemm_config`（726-737） |
| L2 控制 | `l2_ctrl_a/b`（load）、`l2_ctrl_store_cd`（store） | 各操作数/输出的 L2 缓存提示 | 196-201、660-667 |
| Launch | `launch_options`（`deep_jit::ascend::LaunchOptions`） | `num_blocks`（=使用的 AI Core 数）、`num_ubuf_bytes`、`num_launch_timeout_secs` | `num_blocks = d.num_cores`（98） |

第 163-172 行的 `std::formatter<GemmConfig>` 定义了 `DG_PRINT_CONFIGS` 时的打印格式：`GemmConfig(block=MxNxK, mad=MxNxK, l1_stages=…, l0_stages=…, epi_stages=…, …)`——今天实践环节会用到。

与 `GemmDesc` 的关系再强调一次：desc 由用户张量**提取校验**而来（第 11 天），config 由 desc **启发式推导**而来（今天）；`GemmArgs`（gemm_config.hpp:124-127）只是二者打包。**desc 是"问题的唯一描述"，config 是"解的一种选择"**——同一个 desc 理论上可以配不同的 config（第 52 天的单变量实验正是这么做的）。

### 3.3 `BLOCK_M/N/K` 与 `MAD_M/N/K`：两级切分（本日重点）

Ascend 的矩阵乘发生在 Cube 上，数据要经过 `GM → L1 → L0A/L0B →(MAD)→ L0C`。这个硬件数据流天然分成两级，配置也相应分成两级：

```text
                    BLOCK 级（L1 世界）                 MAD 级（L0 世界）
输出 Tile:  ┌──────────────────────┐
  BLOCK_M × │   ┌────────┐          │   一次 MAD 指令：
  BLOCK_N   │   │ MAD_M ×│ MAD_N    │   L0A 里的 (MAD_M × MAD_K)
            │   └────────┘          │   乘 L0B 里的 (MAD_N × MAD_K)
            │   （沿 N 再切一段）     │   累加到 L0C 的 (MAD_M × MAD_N)
            └──────────────────────┘
  K 维:     |<--- BLOCK_K --->|     |<- MAD_K ->|（BLOCK_K 里再分若干段）
            GM→L1 一次搬这么多      L1→L0A/L0B 一次搬这么多
```

| | `BLOCK_M/N/K` | `MAD_M/N/K` |
| :-- | :-- | :-- |
| 所在层级 | L1（及调度） | L0A/L0B/L0C |
| 物理含义 | GM→L1 一次搬运的粒度；K 维循环 `k_blk_idx` 的步长；scheduler 分配的输出 Tile 尺寸 | 一次 MAD 指令的计算块；L1→L0A/L0B 一次搬运的粒度；`k_mad_idx` 循环的步长 |
| 主要容量约束 | `block_m×block_n×4B ≤ L0CSizeBytes`（256KB，heuristics.hpp:101-102）——整个 BLOCK 的累加结果最终要装进 L0C 的多个槽 | `mad_m×mad_n×4B ≤ L0C/2`（127，保证 L0C 至少 2 级以重叠计算与写出）；L0A/L0B 各 64KB 限制 `mad_k`（133-135） |
| 对齐约束 | `block_k % k_alignment == 0`（108-109，BF16 的 `k_alignment = 16`） | 搜索时保证 `block_k % mad_k == 0`（136-137） |

设备侧把它们串起来的两行（`deep_gemm/include/deep_gemm/bf16_gemm.hpp`）：

```cpp
constexpr uint32_t num_mad_cd_stages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);  // 第 51 行
...
static_assert(BLOCK_K % MAD_K == 0, "BLOCK_K must be a multiple of MAD_K");   // 第 70 行
```

即一个 BLOCK 的 K 维要跑 `BLOCK_K / MAD_K` 次 MAD；M/N 维要跑 `BLOCK_M/MAD_M × BLOCK_N/MAD_N` 次 MAD，每次结果占 L0C 的一个槽（`num_mad_cd_stages` 个槽轮转，供"计算下一个 MAD tile"与"把上一个 tile 写出去"重叠）。Kernel 里对应三层循环：`k_blk_idx`（步长 BLOCK_K，112 行）→ `m_mad_idx/n_mad_idx`（步长 MAD_M/MAD_N，123-124 行）→ `k_mad_idx`（步长 MAD_K，133 行）。

为什么 MAD 要比 BLOCK 小？直觉：L0A/L0B/L0C 远小于 L1（64/64/256KB vs 512KB），把整个 BLOCK 一次塞进 L0 既放不下也无法构造"装载下一块与计算当前块"的双缓冲；把 BLOCK 拆成多个 MAD 槽，L0 才能以 2 级（`num_l0_stages=2`）流水起来。3.5 节的例子会给出精确数字。

### 3.4 `select_gemm_config`：候选 → 硬约束 → 成本模型 → 精修

`select_gemm_config(const GemmDesc& d)`（heuristics.hpp:592-755）是纯 host 侧函数，返回一个 `GemmConfig`。第一行就有条 TODO（593）："add a memorization cache"——**配置选择每次调用都重算，没有缓存**（有缓存的是编译出的 kernel 二进制，见 3.6）。整体分四段：

**（a）确定搜索范围（605-621）**

- dtype 归一：`get_gemm_dtype(d)`（80-90）把 desc 的 dtype 组合折成 `elem_bits`（操作数位宽）、`epilogue_elem_bits`、`has_sf` 等。BF16×BF16 → `elem_bits=16, has_sf=false`。
- `block_cap`：BF16 为 512，FP4 或 dequant 为 256（604）。
- 对齐步进 `m_alignment/n_alignment`：K-major 操作数按 `FRAC_MN=16` 步进；MN-major 按 K 向对齐（`get_mad_alignment_k`，BF16 也是 16）。
- **m-grouped 特例（610-616）**：`block_m` 强制等于 `Runtime::get_mk_alignment_for_contiguous_layout()` = 256（`csrc/jit/jit.hpp:59`），`block_n` 也收窄到单值——因为 psum 布局的组边界以 256 对齐（第 11 天），BLOCK_M 必须与之匹配。
- `min_block_k = l1_to_l0_bytes_per_cycle × 8 / elem_bits`（620）：BF16 为 256×8/16 = **128**——正好是 L1→L0 每周期能搬的元素数，避免半次搬运浪费带宽；`max_block_k = 2 × min_block_k`（621）。

**（b）主搜索循环（626-657）**

```cpp
for (block_m = 大 → 小, 步长 m_alignment)
  for (block_n = 大 → 小, 步长 n_alignment) {
      // 剪枝 1：纯访存下界已经超过当前最优 → 跳过
      // 剪枝 2：计算时间 + 首载下界超过当前最优 → 跳过
      auto fitted = make_gemm_candidate(d, dtype, block_m, block_n, min_block_k);  // 硬约束
      if (!fitted) continue;
      if (k % max_block_k == 0) { /* 试更宽的 block_k，要求 ≥3 级 L1 */ }
      const auto cost = get_gemm_cost(d, dtype, *fitted, compute_cost, best_cost);  // 成本模型
      if (cost > best_cost) continue;
      best_cost = cost; config = *fitted;
  }
```

从大 Tile 开始搜是有意的（625 行注释）：大 Tile 早早确立成本上界，后面大量候选被剪枝。注释还说明同等成本时偏好更小的 M/N。

**（c）`make_gemm_candidate`（92-204）：硬约束清单**

给定 (block_m, block_n, block_k)，造一个"合法"候选，任何一条不满足就返回 `nullopt`：

| 约束 | 行号 | 内容 |
| :-- | :-- | :-- |
| L0C 容量 | 101-102 | `block_m × block_n × 4B ≤ 256KB` |
| m-grouped 固定 BLOCK_M | 106 | 必须等于 256 |
| K 对齐 | 108-109 | `block_k % k_alignment == 0` |
| L0 级数 | 112 | `num_l0_stages = 2` |
| MAD 搜索 | 122-147 | 依序试 `(split_m, split_n) ∈ {(1,1),(1,2),(2,1)}`：`mad = block/split`；要求 `mad_m×mad_n×4B ≤ L0C/2`；`mad_k = min(L0A 容量, L0B 容量, max(k_alignment, block_k/2))` 再向下对齐到能整除 block_k；且不能小于 `min(block_k, mte1_copy_k)/2`（太小会浪费 MTE1 带宽，139-142） |
| L1 级数 | 166-174 | `num_l1_stages = min(num_max_l1_stages, L1 剩余容量 / 单级 AB 字节数)`，必须 ≥2；`num_max_l1_stages` 受 `num_pipe_events=8` 限制 |
| epilogue 级数 | 185-194 | UB 容量限制 + `≥ ceil(总 Tile 数 / num_cores) × num_cd_stages` + `num_epilogue_stages × num_aivs ≥ num_cd_stages` |
| L2 读提示 | 196-201 | N 方向块数 ≤2 且 M 方向更多 → A 用 `NOTALLOC_KEEP`（不占 L2，B 复用 L2）；否则 `NORMAL_FIRST_VICTIM`；B 对称 |

其中 `num_pipe_events = 8`（`common.hpp:61`）限制了多级流水的 event 数上限——这就是 BF16 的 L1 级数最多 8 的原因。

**（d）成本模型与精修（206-594、658-755）**

成本模型一族（不必逐行推导，抓结构即可）：

- `get_gemm_compute_cost`（206-254）：MAD 峰值算力估计算时间——`compute_mflops = get_mad_flops_per_cycle × aic_clock_mhz`，即 8192 FLOP/周期 × 1650 MHz（BF16）；
- `get_gemm_cost`（256-468）：**访存侧**——按 L2 sector（128B）放大后的 GM/L2 装载字节、wave 数（`ceil(总 Tile 数 / num_cores)`）、首次装载不可与计算重叠的部分、L1→L0 传输时间，最终 `max(compute_bound, memory_bound)`；
- `get_gemm_output_cost`（474-494）与 `get_gemm_refinement_cost`（496-519）：输出侧（L0C→UB→GM、bias）与综合成本；
- `get_gemm_k_cost`（576-590）：在主搜索之后比较不同 `block_k` 时使用，额外计入 MAD issue 与 L0 换入换出的开销。

硬件常量全部来自 `get_hw_desc()`（52-71，注释标明 Ascend 950）：GM 带宽 3500 GB/s、L2 4500 GB/s、AIC 1650 MHz、BF16 8192 FLOP/周期、L1→L0 256 B/周期、32 个 AI Core、L2 128MB 等。

选出主配置后还有四步精修（658-747），BF16 主线只需关注前两步：

1. **L2 store 提示**（660-667）：默认 `NOTALLOC_CLEAN`；累加且 wave≥4 且 K 块数 ≥16 时改 `NORMAL_FIRST_VICTIM`（让被反复读加的 D 留在 L2）。
2. **`refine_gemm_pipeline`**（669，实现在 535-574）：若 `can_direct_store`（521-533：Normal + Identity epilogue + 对齐/D 指针 32B 对齐等一串条件）且输出成本不升，打开 `direct_store`；若 K 是编译期常量且 shape 合适，尝试 `resident_a`（A 整块常驻 L1，M ≤ BLOCK_M 时）或 `resident_b`（B 常驻 L0，N 只有一块时）——每次尝试都要成本模型点头才保留。
3. K 细化（671-694）：固定 M/N，按 2 的幂试不同 `block_k`（仅 Normal 且 K 编译期常量化时）。
4. bias / FP4 特例（696-747）：输出受限的 bias 场景换更小 Block 换双缓冲；FP4 全 MAD、双 AIV 解码、L2 提示微调——低精度相关，第 32 天再见。

最后（749-753）：若开了 `DG_PRINT_CONFIGS` 或 `DG_JIT_DEBUG`，用 `std::format("{} {}", d, config)` 打印 desc+config（静态集合去重，同一组合每进程只打一次）。

### 3.5 模板代码生成：26 个实参与 `__instantiate`

第②步把 desc + config"渲染"成一段独立的 C++ 源码（`bf16_gemm.hpp:18-42` 的模板串 + 43-66 的实参表）：

```cpp
#include <deep_gemm/bf16_gemm.hpp>

namespace deep_gemm {
static void __instantiate() {
    auto p = reinterpret_cast<void*>(&bf16_gemm_impl<
        /*1*/{}, /*2*/{},          // kMajorA, kMajorB
        /*3-5*/{}, {}, {},         // SHAPE_M, SHAPE_N, SHAPE_K
        /*6-8*/{}, {}, {},         // BLOCK_M, BLOCK_N, BLOCK_K
        /*9-11*/{}, {}, {},        // MAD_M, MAD_N, MAD_K
        /*12-13*/{}, {},           // kNumL1Stages, kNumL0Stages
        /*14*/{},                  // kNumEpilogueStages
        /*15*/{},                  // kNumCores = launch_options.num_blocks
        /*16-18*/{}, {}, {},       // kL2CtrlA, kL2CtrlB, kL2CtrlStoreCd
        /*19*/{},                  // kWithAccumulation
        /*20*/{},                  // kGemmType
        /*21*/{},                  // cd_dtype_t
        /*22*/{},                  // epilogue_operator_t
        /*23*/{},                  // kAlignment
        /*24-26*/{}, {}, {}>);     // kDirectStore, kResidentA, kResidentB
}
} // namespace deep_gemm
```

26 个实参的来源分三类（对照 43-66 行的实参列表）：

| 类别 | 实参 | 来源 |
| :-- | :-- | :-- |
| **来自 desc** | major_a/major_b（1-2）、acc（19）、gemm_type（20）、`dtype_to_cd_kernel_type(cd_dtype)`（21，BF16 输出 → `"bfloat16_t"`）、`epilogue_class->get_epilogue_operator_type()`（22，如 `"epilogue::operators::Identity"`） | 第 11 天的 desc 字段直译 |
| **desc × compiled_dims** | `get_compiled_dim(desc.m/n/k, 'm'/'n'/'k', compiled_dims)`（3-5）：字母在 `compiled_dims` 里就填维度值，否则填 **0**（`gemm_config.hpp:17-21`） | 0 的含义见下 |
| **来自 config** | block/mad（6-11）、三级 stages（12-14）、`launch_options.num_blocks.value()`（15，= num_cores）、三个 L2 提示（16-18）、direct_store/resident_a/b（24-26） | `select_gemm_config` 的产出 |
| **全局常量** | `Runtime::get_mk_alignment_for_contiguous_layout()`（23）= 256 | `csrc/jit/jit.hpp:59` |

枚举实参（`Major::K`、`GemmType::Normal`、`asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM` 等）能直接以 `类型::名字` 形式进字符串，靠的是 `csrc/utils/format.hpp:11-20` 用 magic_enum 给所有 enum 注册的 `std::formatter`。

**`SHAPE_M = 0` 表示"运行期取值"。** 设备 kernel 开头（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:47-49`）：

```cpp
uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime;
```

而 `shape_m/n/k_runtime` 正是 launch 传进来的 `desc.m/n/k`（3.6 节）。binding 默认 `bf16_gemm_nt/nn` 用 `"nk"`、`tn/tt` 用 `"mn"`（`apis/gemm.hpp:635-653`）——**推理场景里 M（token 数）随时变化、N/K（权重形状）固定**，把 N/K 编成常量、M 留给运行期， kernel 特化与重编译次数得以平衡（tn/tt 的 A 是 `[K,M]`，编译期固定的轴换成了 M/N，故默认 `"mn"`）。用户也可以在 Python 侧显式传 `compiled_dims` 改变这一选择。

**为什么要 JIT 生成这段代码？** 因为 `bf16_gemm_impl` 是一个 26 参的函数模板，只有显式实例化才生成机器码；而把 block/mad/stages 等做成**模板参数（编译期常量）**后，kernel 内部才能写 `constexpr l1a/l1b/l0a/l0b/l0c` 的片上地址规划、以及一组 `static_assert` 容量检查（`bf16_gemm.hpp:59-70`）——这些在第 16 天是主角。`__instantiate()` 这个没人调用的 static 函数唯一的作用是**取模板函数的地址，迫使编译器实例化它**；`reinterpret_cast<void*>` 只是抹掉类型方便编译单元统一处理。

**`__mix__(1, kDirectStore ? 0 : 2)`**（第 27 行）：kernel 的 AIC/AIV 配比——每个 block 1 个 AIC；普通路径再配 2 个 AIV 走 UB epilogue，`direct_store` 时 0 个 AIV（L0C 直接写 GM，不需要向量核）。AIC/AIV 分工细节留到第 25 天。

**运行期参数只有 9 个**（`bf16_gemm.hpp:73-77`）：三个 `gm_ptr<uint8_t>{outer_stride, ptr, stride_batch}`（A/B/D 的地址三元组，第 11 天 3.4 节）、`m/n/k`、`grouped_layout`、`num_groups`、`make_epilogue_operator_args(desc.n)`（如 bias 指针、alpha）。它们与 kernel 签名（28-34 行）的 9 个形参一一对应。**分界线很清晰：影响"生成哪份代码"的进模板参数，每次调用都可能不同的（地址、运行期 shape、组信息）走 launch 参数。**

### 3.6 `jit->compile` / `jit->launch` 与 Kernel 缓存

第③④步的 `jit` 是 `csrc/jit/jit.hpp` 里的全局对象：

```cpp
using AscendJIT = deep_jit::Runtime<deep_jit::Ascend>;   // 第 14 行
inline deep_jit::LazyInit<AscendJIT> jit(nullptr);       // 第 16 行，init_jit 时才真正构造
```

`init_jit`（78-92 行）由 `_C.init(__version__, get_library_root_path())` 触发（`deep_gemm/__init__.py:159`，`csrc/apis/runtime.hpp:41-44`）：

```cpp
const auto include_dir = library_root / "include";
const deep_jit::Config config(library_root, "DG", version, {include_dir}, {"deep_gemm/"});
jit = deep_jit::LazyInit<AscendJIT>([config] {
    auto jit_runtime = std::make_shared<AscendJIT>(config);
    jit_runtime->default_compiler_options.extra_bisheng_flags.emplace_back("--cce-disable-vf-stack-reserved-ubuf");
    jit_runtime->default_compiler_options.extra_bisheng_flags.emplace_back("-std=c++20");
    ...
});
```

要点：前缀 `"DG"` 对应 `DG_JIT_*` 环境变量；include 目录指向 `deep_gemm/include/`——所以生成的代码 `#include <deep_gemm/bf16_gemm.hpp>` 与仓库里的设备头是同一份；编译器是 CANN 的 **Bisheng**（`bin/bisheng`，README Requirements 第 24 行），默认附加 `--cce-disable-vf-stack-reserved-ubuf`（不保留 vf 栈的 UB，把 UB 让给 epilogue 缓冲）和 `-std=c++20`。

从本仓库能看到的 DeepJIT 接口只有三个（其实现位于独立仓库 [DeepJIT](https://github.com/deepseek-ai/DeepJIT)，`third-party/deep-jit` 为未内联源码的子模块，README 致谢第 199 行）：

| 调用 | 输入 | 输出/行为 |
| :-- | :-- | :-- |
| `jit->compile(name, code)` | kernel 名 + 生成的源码 | 编译并加载，返回 kernel 句柄；**内容相同的 code 命中缓存不再编译** |
| `jit->launch(kernel, launch_options, args...)` | 句柄 + LaunchOptions + 9 个参数 | 在当前 NPU stream 上启动（`get_current_npu_stream`，jit.hpp:94） |
| `jit->compile_without_load(name, code)` | 同 compile | 只编译不加载，配合 dry-run |

**Kernel 缓存**（README"Environment Variables"，83-98 行）：

- `DG_JIT_CACHE_DIR`：缓存目录，默认 `$HOME/.dj`；支持 `:` 分隔的多路径，查找从前到后**首个命中生效**，未命中时编译产物写入第一个路径；
- `DG_JIT_*` 变量在 **JIT runtime 首次创建时快照**（85 行），之后改环境变量不再生效；
- `DG_JIT_PRINT_COMPILER_COMMAND=1` 打印编译器命令（观察是否真的在编译）；`DG_JIT_PRINT_LOAD_TIME=1` 打印加载耗时；`DG_JIT_LAUNCH_TIMEOUT` 启动超时（默认 300 秒）。

两个容易混淆的"缓存"要分清：

1. **Kernel 二进制缓存**（DeepJIT，`~/.dj`）：同一份生成代码（同一组模板参数）第二次 `compile` 不再调 Bisheng，只是加载。冷启动慢、热启动快的根源（第 50 天会量化）。
2. **配置选择缓存**：**不存在**——`select_gemm_config` 每次调用都完整重算（heuristics.hpp:593 的 TODO 即此）。因为它只是整数运算，开销远小于编译，可以接受。

**dry-run 与并行编译。** `runtime->get_dry_run()` 为真时，launch 函数走到 `compile_without_load` 就返回——不加载、不启动（`bf16_gemm.hpp:67-70`）。用途见 `deep_gemm/testing/par_compile.py`：`par_compile` 用线程池并发执行一组调用，全程 `set_dry_run(True)`，**只为把所有需要的 kernel 特化编译进缓存**，之后正式运行全部命中缓存。测试体系用 meta 张量 + 这条路径做并行预热（第 30 天）。另一个小例子是 `csrc/apis/runtime.hpp:64-79` 的 `npu_sleep`：同样走"生成 → compile → launch"三步，是理解该模式的最小样本。

### 3.7 从 `GemmDesc` 到 Kernel launch 的流程图（当天产出·参考答案）

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Python: deep_gemm.bf16_gemm_nt(a, b, d, c=None, compiled_dims="nk")       │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               ▼  pybind11（csrc/python_api.cpp → apis/gemm.hpp）
                    逻辑转置 / dtype 断言 / epilogue 归一
                               ▼
                    get_gemm_desc()（apis/gemm.hpp:60-197）
                               │  GemmDesc：m,n,k / major / stride / dtype /
                               │  gemm_type / grouped_layout / epilogue_class / acc …
                               ▼
┌───────────────── launch_bf16_gemm(desc)（csrc/jit_kernels/bf16_gemm.hpp:16）─────────────────┐
│                                                                                              │
│  ① select_gemm_config(desc)（heuristics.hpp:592）                                            │
│     ├─ get_gemm_dtype：elem_bits=16、has_sf=false（BF16）                                      │
│     ├─ 搜索范围：m_range/n_range（m-grouped 固定 BLOCK_M=256）、min/max_block_k=128/256        │
│     ├─ 主循环（block_m ↓ × block_n ↓）                                                        │
│     │    ├─ make_gemm_candidate：硬约束（L0C/L0A/L0B/L1/UB 容量、对齐、MAD 搜索、级数≥2）       │
│     │    └─ get_gemm_cost：成本模型剪枝（GM/L2 装载 × wave × 算力，取 compute/memory bound 最大）│
│     ├─ L2 store 提示 → refine_gemm_pipeline（direct_store / resident_a/b，均需成本点头）        │
│     ├─ K 细化（仅 Normal + K 编译期常量）＋ bias/FP4 特例                                      │
│     └─ [DG_PRINT_CONFIGS] 打印 "GemmDesc(...) GemmConfig(...)"                                │
│           │  GemmConfig：block/mad、l1/l0/epi stages、direct_store、resident、l2_ctrl、         │
│           │             launch_options.num_blocks = num_cores                                  │
│           ▼                                                                                  │
│  ② std::format 生成源码（bf16_gemm.hpp:18-66）                                                │
│     #include <deep_gemm/bf16_gemm.hpp>                                                       │
│     static void __instantiate() {                                                            │
│         auto p = reinterpret_cast<void*>(&bf16_gemm_impl<                                     │
│             Major::K, Major::K,            ← desc.major_a/b                                  │
│             0, 64, 256,                    ← get_compiled_dim(m/n/k,"nk")：M 运行期          │
│             128, 512, 128,                 ← config.block_m/n/k                              │
│             128, 256, 64,                  ← config.mad_m/n/k                                │
│             3, 2, 2, 32,                   ← l1/l0/epi stages、num_blocks（=AI Core 数）      │
│             …, false, GemmType::Normal, bfloat16_t, epilogue::operators::Identity,           │
│             256, false, false, false>);   ← 对齐、acc、gemm_type、dtype、epilogue、路径开关    │
│     }                                                                                        │
│           │                                                                                  │
│           ├─ dry_run？──是──► jit->compile_without_load("bf16_gemm", code)；return            │
│           ▼                                                                                  │
│  ③ jit->compile("bf16_gemm", code)（csrc/jit/jit.hpp：DeepJIT + Bisheng）                    │
│     缓存目录 DG_JIT_CACHE_DIR（默认 ~/.dj，多路径首个命中），代码相同 → 不再编译                │
│           │  kernel 句柄                                                                     │
│           ▼                                                                                  │
│  ④ jit->launch(kernel, config.launch_options,                                                │
│                gm_ptr{stride_a, a_ptr, batch_a},   ← 9 个运行期参数：                         │
│                gm_ptr{stride_b, b_ptr, batch_b},     地址三元组 ×3、m/n/k、                    │
│                gm_ptr{stride_d, d_ptr, batch_d},     grouped_layout、num_groups、             │
│                m, n, k, grouped_layout, num_groups,  epilogue args（bias/alpha…）             │
│                epilogue_args)                                                                │
└──────────────────────────────┬───────────────────────────────────────────────────────────────┘
                               ▼  当前 NPU stream
              设备：bf16_gemm_impl（deep_gemm/include/deep_gemm/bf16_gemm.hpp）
              Scheduler 分 Tile → GM→L1→L0 → MAD 累加 → Epilogue → GM
              （第 3～4 周展开；第 13 天闭卷复画本图）
```

（图中数值是一个具体示例的填充结果，推导见第五节 B-1。）

## 四、源码导读

今天的目标是**把 3.7 流程图的每个箭头落到真实代码上**。每站 5～10 分钟。

### 站点 1：`csrc/jit_kernels/bf16_gemm.hpp`（80 行，主线，精读）

- 第 13-15 行注释：一个函数服务三种 gemm_type、NT 语义、acc 语义；
- 第 17 行：`select_gemm_config` 一行完成配置选择；
- 第 18-42 行：模板串——数一数 26 个 `{}`，注意 `{{`/`}}` 是字面大括号转义；
- 第 43-66 行：实参表——对照 3.5 的三类来源，逐个标出"desc / desc×compiled_dims / config / 全局"；
- 第 67-70 行：dry-run 分支；
- 第 71 行与 73-77 行：`compile` 与 `launch`——数一数 launch 的 9 个参数，与设备 kernel 签名（站点 5 的 28-34 行）一一连线。

### 站点 2：`csrc/jit_kernels/gemm_config.hpp` 的 `GemmConfig`（97-121）

- 逐字段读注释：`num_l1_stages` "MTE2 <-> MTE1"、`num_l0_stages` "MTE1 <-> Cube"、`num_epilogue_stages` "per AIV (AIC -> AIV)"——三个流水深度各自衔接哪两条流水线；
- `num_l1_sf_stages`/`sf_k_blocks` 注明 "0 for BF16"；
- 第 133-143 行 `std::formatter<LaunchOptions>`：`num_blocks`/`num_ubuf_bytes`/`num_launch_timeout_secs` 三个字段的出处；
- 第 163-172 行 `std::formatter<GemmConfig>`：`DG_PRINT_CONFIGS` 输出格式的定义。

### 站点 3：`csrc/jit_kernels/heuristics.hpp`（757 行，挑读，本日主战场）

按 3.4 的顺序读：

1. `get_hw_desc`（52-71）：Ascend 950 常量表——GM/L2 带宽、时钟、每周期算力、各级每周期搬运字节、32 核、L2 128MB；
2. `get_gemm_dtype`（80-90）：dtype 组合 → `elem_bits/has_sf/dequant_b`；
3. `make_gemm_candidate`（92-204）：对照 3.4（c）的约束清单，**在源码里找到每条 `return std::nullopt` 并注明原因**；重点读 122-147 的 MAD 搜索——三种 split、L0C/2、`l0a_k/l0b_k`、`mte1_copy_k` 下限；
4. `select_gemm_config`（592-755）：593 的 TODO（无记忆化）、605-616 的范围确定（m-grouped 固定 256）、620-621 的 block_k、626-657 主循环（两处剪枝 + wider 尝试）、658 断言、660-667 L2 store、669 refine、749-753 打印；
5. 成本模型（206-594）今天只读函数头与返回值：`get_gemm_compute_cost` 返回 `(mflops, time_ns, num_tail_blocks)`，`get_gemm_cost` 返回 `max(compute_bound, memory_bound)` 的纳秒数——知道每个函数"估什么"即可，公式留到第 50 天；
6. `refine_gemm_pipeline`（535-574）与 `can_direct_store`（521-533）：direct_store 的条件清单（Normal、Identity、对齐、D 指针 32B 对齐、Tile 数 ≤ 核数或行长对齐 128B…）。

### 站点 4：`csrc/jit/jit.hpp`（96 行）

- 第 14-16 行：`AscendJIT` 别名与 `jit` 全局（LazyInit，初始为 nullptr，`init_jit` 才赋真身）；
- `Runtime` 类（18-74）：`num_sms/arch/dry_run/deterministic` 四个状态，`set_npu_arch` 会写 `jit->default_compiler_options.arch`（第 31 行）——`set_npu_arch` Python API 影响的就是 JIT 编译目标架构；`kMkAlignment=256`（59）；
- `init_jit`（78-92）：Config 三要素（根目录/前缀 "DG"/版本）、include 目录、两条 Bisheng 旗标、`DG_JIT_WITH_LINEINFO`；
- 对照 README 83-98 行的环境变量表，标出每个变量影响本文件的哪一步。

### 站点 5：`deep_gemm/include/deep_gemm/bf16_gemm.hpp`（195 行，只读模板与骨架）

- 模板参数表（9-26）：与站点 1 生成的 26 个实参逐一对号——**这是验证"代码生成正确性"的天然练习**；
- 第 27 行：`__global__ __mix__(1, kDirectStore ? 0 : 2)`；
- 第 47-49 行：SHAPE 与 runtime shape 的择一；第 51 行：`num_mad_cd_stages`；
- 第 59-70 行：`constexpr` 片上缓冲声明与五个 `static_assert`——config 的每个容量约束在设备侧的镜像；
- 第 81 行起 AIC 分支的三层循环骨架（112/123/133 行的三个循环变量步长），173 行起 AIV 分支——今天只看结构，语义留到第 3～4 周。

### 站点 6：`deep_gemm/include/deep_gemm/common.hpp`（54-65、82-135）

- 容量常量（54-58）：L1 512KB、L0A/L0B 64KB、L0C 256KB、UB 256KB；`kNumPipeEvents=8`（61）、`FRAC_MN=16`（63）——heuristics 所有容量计算的分母；
- `Major`/`GemmType`（82-92）与 host 侧镜像的 `asc_load/store_l2_cache_mode`（118-135，注释"without Bisheng headers"）——模板参数里的枚举字符串为何能被 host 格式化。

### 站点 7：dry-run 与初始化的旁证

- `csrc/apis/runtime.hpp`：`_C.init` → `init_jit`（41-44，注释"Keep imports free of NPU runtime initialization so callers can fork"）；`set_dry_run` 先 `jit.get()` 强制初始化（46-50）；`npu_sleep`（64-79）一个 30 行的完整"生成-编译-启动"最小样本；
- `deep_gemm/__init__.py:159`：`_C.init(__version__, get_library_root_path())` 的调用点；
- `deep_gemm/testing/par_compile.py`：`dry_run` 上下文管理器 + 线程池并行编译（7-35）。

**阅读检查点**（能回答再进入实践）：

1. `launch_bf16_gemm` 里 `runtime->get_dry_run()` 为真时，三步里哪两步还会执行？哪一步被跳过？
2. `make_gemm_candidate` 里 `num_l0_stages` 是多少？`num_l1_stages` 的上限受哪个常量约束？
3. 生成代码的第 3-5 个 `{}` 由哪个函数填充？填 0 意味着什么？
4. `select_gemm_config` 有没有缓存？Kernel 二进制呢？各在何处？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：观察配置选择与编译命令（约 15 分钟）**。

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 DG_JIT_PRINT_COMPILER_COMMAND=1 \
    python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day12_run1.log
```

在输出中找到形如 `GemmDesc(m=..., n=..., k=..., ...) GemmConfig(block=...x...x..., mad=...x...x..., l1_stages=..., ...)` 的行（每组合只打一次，heuristics.hpp:750-752），以及 Bisheng 编译命令（含 `--cce-disable-vf-stack-reserved-ubuf`、`-std=c++20`）。抄录 2～3 行到第六节记录。

**步骤 2：验证 Kernel 缓存（约 10 分钟）**。

```bash
ls ~/.dj                                   # 观察缓存目录结构
DG_JIT_PRINT_COMPILER_COMMAND=1 DG_JIT_PRINT_LOAD_TIME=1 \
    python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day12_run2.log
```

对比两次日志：第二次**不应再有 Bisheng 编译命令**（相同代码命中 `~/.dj` 缓存），但有 kernel 加载耗时。再试 `DG_JIT_CACHE_DIR=/tmp/day12_fresh_cache` 强制空缓存重跑，确认编译命令回来了。把"冷/热"两次的差异写进记录——这是第 50 天"JIT 编译耗时与执行耗时"的第一次感性认识。

**步骤 3：shape 对配置的影响（约 15 分钟）**。

```bash
DG_PRINT_CONFIGS=1 python - <<'EOF'
import torch, torch_npu  # noqa
import deep_gemm

for M in (128, 4096):
    a = torch.randn(M, 7168, dtype=torch.bfloat16, device='npu')
    b = torch.randn(7168, 7168, dtype=torch.bfloat16, device='npu')
    d = torch.empty(M, 7168, dtype=torch.bfloat16, device='npu')
    deep_gemm.bf16_gemm_nn(a, b, d)
    torch.npu.synchronize()
EOF
```

对比两行 `GemmConfig`：M=128（单 wave、可能 resident/direct_store）与 M=4096（多 wave、更大 block_m）的差异；核对 `l1_stages ≤ 8`、`block_m×block_n×4 ≤ 256KB` 等约束是否全部满足。

**步骤 4（可选）：dry-run 预热（约 5 分钟）**。`deep_gemm._C.set_dry_run(True)` 后重复步骤 3 的调用——`DG_PRINT_CONFIGS` 仍打印（select 照跑），但无 kernel 启动；随后 `set_dry_run(False)` 重跑，观察加载时间是否因缓存而下降。这正是 `par_compile` 的原理。

### B. 无 Ascend 环境：静态替代任务

**B-1：手工追踪一个候选配置（约 20 分钟）。** 设 BF16 Normal `M=128, N=64, K=256`，`nt` 调用（`compiled_dims="nk"`，A/B 均 K-major，identity epilogue，`num_cores=32`）。按 `select_gemm_config` → `make_gemm_candidate(128, 512, 128)` 推导（这就是 3.7 图中数值的来历）：

1. 范围：`block_cap=512`；`m_alignment=n_alignment=16`（K-major → `FRAC_MN`）；`m_range={16,128}`、`n_range={16,512}`；`min_block_k=256×8/16=128`、`max_block_k=256`。
2. L0C 检查：`128×512×4B = 262144 ≤ 256KB`（压线通过）。
3. K 对齐：`k_alignment = fractal_row_bytes×8/16 = 16`，`128 % 16 == 0` ✓。
4. MAD 搜索（`num_l0_stages=2`）：
   - `(1,1)`：mad 128×512 → `128×512×4 = 262144 > L0C/2 = 131072` ✗；
   - `(1,2)`：mad 128×256 → `131072 ≤ 131072` ✓；`l0a_k = 64KB×8/(2×128×16) = 128`，`l0b_k = 64KB×8/(2×256×16) = 64`，`mad_k = min(128, 64, max(16, 128/2)) = 64`；`128 % 64 == 0` ✓；`mte1_copy_k = 128`，`64 < min(128,128)/2 = 64` 不成立 ✓ → **mad = 128×256×64**。
5. L1 级数：单级 AB 字节 = `(128+512)×128×2 = 163840`；`num_l1_stages = min(8, 512KB/163840 = 3) = 3`。
6. epilogue 级数：`num_cd_stages = (128/128)×(512/256) = 2`；每级 `128×256×2 = 65536B`；`min(16, 256KB/65536=4) = 4`，再 `min(4, ceil(1/32)×2 = 2) = 2`（总 Tile 数 = `ceil(128/128)×ceil(64/512) = 1`）。
7. 片上用量核对（设备侧 `static_assert`，`size_per_stage = mn×k×sizeof`，`ascend/tile_ptr.hpp:37`）：L0A `2×128×64×2 = 32768 ≤ 64KB`；L0B `2×256×64×2 = 65536 ≤ 64KB`（压线）；L0C `2×128×256×4 = 262144 ≤ 256KB`（压线）；L1 `3×(128+512)×128×2 = 491520 ≤ 512KB`。
8. wider 尝试（`k % max_block_k == 0`）：`make_gemm_candidate(128, 512, 256)` → 单级 AB = `(128+512)×256×2 = 327680`，`num_l1_stages = 512KB/327680 = 1 < 2` → **被拒**——大 BLOCK_K 会把 L1 级数挤没。

注意：这只是主循环实际评估的**第一个候选**；完整搜索会遍历 `block_n = 512, 496, …, 16` 并由成本模型选出最终者（可在硬件上用 `DG_PRINT_CONFIGS` 验证，不必手算到底）。

**B-2：手工生成模板代码（约 15 分钟）。** 用 B-1 的 desc 与候选 config，把 3.5 的 26 个占位符全部填出（答案即 3.7 图中②框内的代码：`Major::K, Major::K, 0, 64, 256, 128, 512, 128, 128, 256, 64, 3, 2, 2, 32, …, false, GemmType::Normal, bfloat16_t, epilogue::operators::Identity, 256, false, false, false`）。然后回答：SHAPE_M 为什么是 0？第 15 个实参 32 从哪来（提示：`launch_options.num_blocks = d.num_cores`，heuristics.hpp:98）？

**B-3：闭卷画流程图（约 15 分钟）。** 合上教程，凭源码画出 3.7 的图（从 `launch_bf16_gemm` 到 `jit->launch`，含 ①②③④ 四步与 dry-run 分支），标注每步的输入/输出结构体。对照订正——订正后的图就是当天产出。

**B-4：约束清单标注（约 10 分钟）。** 在 `heuristics.hpp:92-204` 里数出所有 `return std::nullopt`（共 8 处），每处写一行"违反了什么资源/规则"；再写出对应的设备侧 `static_assert`（`bf16_gemm.hpp:66-70`）——体会"host 检查与 device 断言互为镜像"。

## 六、当天产出

学习计划要求今天的产出是**从 `GemmDesc` 到 Kernel launch 的流程图**。按下述模板填写（B-1～B-2 的演算过程附在后面）：

```text
# DeepGEMM-Ascend 第 12 天记录：理解 JIT 配置、编译与启动

## A. 流程图（闭卷绘制后对照 3.7 订正）
launch_bf16_gemm
  ① select_gemm_config：输入=＿＿ 输出=＿＿；四个阶段：＿＿/＿＿/＿＿/＿＿
  ② 代码生成：26 个实参中来自 desc 的有＿＿个、来自 config 的有＿＿个；
     SHAPE_M=0 表示＿＿；__instantiate 的作用是＿＿
  ③ jit->compile：缓存目录=＿＿（默认），相同代码第二次调用会＿＿
  ④ jit->launch：9 个运行期参数分别是＿＿

## B. BLOCK vs MAD 对照（闭卷填写）
- BLOCK_M/N/K 是＿＿级 Tile，受＿＿容量约束；MAD_M/N/K 是＿＿级计算块，受＿＿约束
- num_mad_cd_stages = ＿＿；一个 BLOCK_K 内要执行＿＿次 MAD
- num_l0_stages 固定为＿＿；num_l1_stages 上限=＿＿（受＿＿限制）

## C. 手工演算记录（B-1/B-2）
- 候选 (128,512,128)：mad=＿＿×＿＿×＿＿ l1=＿＿ l0=＿＿ epi=＿＿
- 片上用量：L0A=＿＿ L0B=＿＿ L0C=＿＿ L1=＿＿（各与容量比较）
- 生成代码第 3-5 个实参：＿＿（为什么？）
- 被拒的 wider 候选 (128,512,256)：拒绝原因=＿＿

## D. 实验记录（有环境时）
- DG_PRINT_CONFIGS 抄录的 GemmConfig 行：＿＿
- 冷/热两次运行的差异（编译命令/加载时间）：＿＿
- M=128 与 M=4096 的配置差异：＿＿

## E. 遗留问题（交给后续天）
- 成本模型的公式细节（第 50 天）
- direct_store / resident 的设备侧行为（第 26 天）
- __mix__ 与 AIC/AIV 分工（第 25 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `launch_bf16_gemm` 依次做哪三件事？dry-run 时行为有何不同？dry-run 的实际用途是什么？
2. `GemmConfig` 的三个流水深度字段 `num_l1_stages`/`num_l0_stages`/`num_epilogue_stages` 各自衔接哪两条流水线？各自的确定方式是什么？
3. `BLOCK_M/N/K` 与 `MAD_M/N/K` 分别描述哪一级的切分？`BLOCK_K` 与 `MAD_K` 必须满足什么关系？为什么 MAD tile 不能太大（两个容量约束）？
4. `num_mad_cd_stages` 等于什么？它在 L0C 里起什么作用？
5. `select_gemm_config` 主循环为什么从大 Tile 开始搜？`make_gemm_candidate` 返回 `nullopt` 意味着什么？m-grouped 时 `block_m` 为什么被固定为 256？
6. BF16 的 `min_block_k` 是多少？它是怎么算出来的？这个值与哪条流水线的带宽有关？
7. 生成代码里 `SHAPE_M=0` 是什么意思？设备 kernel 如何处理？`bf16_gemm_nt` 默认把哪些维度编译期常量化，为什么？
8. `__instantiate()` 函数为什么没人调用却必须有？`reinterpret_cast<void*>` 的作用是什么？
9. `jit->compile` 与 `jit->launch` 各接收什么？Kernel 缓存目录如何配置、默认在哪里、多路径时如何查找？"配置选择"有缓存吗？
10. `launch_options.num_blocks` 装的是什么值？它同时出现在哪里（提示：模板参数表第 15 个）？
11. （思考题）为什么把 block/mad/stages 做成模板参数（编译期常量）而不是运行期参数？代价是什么？`compiled_dims` 如何缓解这个代价？

**参考答案：**

1. ①`select_gemm_config(desc)` 得到 `GemmConfig`；②`std::format` 生成 `bf16_gemm_impl<...>` 显式实例化源码；③`jit->compile("bf16_gemm", code)` 编译并 `jit->launch(kernel, launch_options, 9 个参数)` 启动（`bf16_gemm.hpp:16-78`）。dry-run 时走 `jit->compile_without_load`：只编译不加载、直接 return，不启动 kernel（67-70）。用途：`par_compile`（`deep_gemm/testing/par_compile.py`）在线程池里并发预热编译所有 kernel 特化，让正式运行全部命中缓存；meta 张量测试也走这条路。
2. `num_l1_stages`：L1 数据缓冲级数，衔接 MTE2（GM→L1）与 MTE1（L1→L0），由 L1 512KB 容量除以单级 AB 字节数得出、至少 2（heuristics.hpp:172-174）；`num_l0_stages`：L0A/L0B 缓冲级数，衔接 MTE1 与 Cube（MAD），固定为 2（112）；`num_epilogue_stages`：每 AIV 的 UB epilogue 级数（AIC→AIV），由 UB 容量与输出 Tile 数共同限制（185-194）。字段注释见 `gemm_config.hpp:104-106`。
3. BLOCK 是 L1 级 Tile（GM→L1 一次搬运、K 维循环步长、scheduler 的输出 Tile），约束 `block_m×block_n×4B ≤ L0C 256KB`（101-102，累加结果要放进 L0C）；MAD 是一次 MAD 指令的 L0A/L0B→L0C 计算块（L1→L0 搬运粒度、`k_mad_idx` 步长）。必须 `BLOCK_K % MAD_K == 0`（设备侧 static_assert，`bf16_gemm.hpp:70`；host 侧在搜索时保证，136-137）。MAD 不能太大：`mad_m×mad_n×4B ≤ L0C/2`（127，留出至少 2 级 L0C 槽重叠计算与写出）；L0A/L0B 各 64KB 通过 `l0a_k/l0b_k` 限制 `mad_k`（133-135）。
4. `(BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`（`bf16_gemm.hpp:51`）。一个 BLOCK 内所有 MAD tile 的 FP32 结果在 L0C 里占这么多个槽、轮转使用，让"算下一个 tile"与"写 out 上一个 tile"重叠。
5. 大 Tile 早早确立成本上界，后续候选被 `min_memory_time_ns` 和计算下界剪枝，搜索快（625 行注释）；同等成本偏好更小的 M/N。`nullopt` 表示该 (block_m, block_n, block_k) 违反了某条硬约束（L0C 容量、对齐、MAD 搜索失败、L1/epilogue 级数 <2 等），不是合法候选。m-grouped 的 psum 布局组边界按 256 对齐（`Runtime::kMkAlignment`，第 11 天），BLOCK_M 必须等于 256 才能与组边界对齐（heuristics.hpp:106）。
6. 128。`min_block_k = l1_to_l0_bytes_per_cycle × 8 / elem_bits = 256×8/16`（620）——正好是 MTE1（L1→L0）每周期能搬运的元素数，BLOCK_K 取它的整数倍才能避免"半次搬运"浪费 MTE1 带宽（619 行注释）；`max_block_k = 256`。
7. 0 表示该维度留给运行期：设备 kernel 里 `shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime`（`bf16_gemm.hpp:47-49`），`shape_m_runtime` 来自 launch 参数 `desc.m`。`bf16_gemm_nt/nn` 默认 `compiled_dims="nk"`（`apis/gemm.hpp:635/641`）：推理里 N/K 是固定权重形状、M 是变化的 token 数，常量化 N/K 让 kernel 特化的收益落在稳定的维度上、避免每个 M 都触发重编译。
8. `__instantiate()` 唯一的作用是对模板函数取地址（`&bf16_gemm_impl<...>`），迫使编译器实例化该特化并生成符号；`reinterpret_cast<void*>` 抹掉函数类型，让所有 kernel 的实例化代码形状统一，方便 DeepJIT 按"名字 + 源码"统一处理。
9. `compile` 接收 kernel 名与生成的源码字符串，返回 kernel 句柄；`launch` 接收句柄、`LaunchOptions` 和 9 个运行期参数。缓存目录 `DG_JIT_CACHE_DIR`，默认 `$HOME/.dj`；冒号分隔多路径时从前到后查找、首个命中生效，未命中时产物写入第一个路径（README 90-91）。配置选择**没有**缓存——`select_gemm_config` 每次完整重算（heuristics.hpp:593 的 TODO）。
10. `d.num_cores`（可用 AI Core 数，`make_gemm_candidate` 里 `c.launch_options.num_blocks = d.num_cores`，98 行）。它同时作为模板参数 `kNumCores` 传给 kernel（生成代码的第 15 个实参，`bf16_gemm.hpp:57`），设备侧 Scheduler 用它把 `block_idx` 映射到输出 Tile。
11. 开放题，参考方向：编译期常量让 kernel 能用 `constexpr` 规划 L1/L0A/L0B/L0C 地址、用 `static_assert` 在编译期暴露容量错误、循环完全展开——这些是接近硬件极限的前提（`bf16_gemm.hpp:59-70`）。代价是每换一组参数就要编译一个新实例（Bisheng 编译秒级），所以要有 `~/.dj` 二进制缓存与 dry-run 预热；`compiled_dims` 把"变化频繁的维度"（如 M）排除在模板参数外、只常量化稳定维度，在特化收益与重编译次数之间取得平衡。

---

今天结束时，你应该能把"一份 `GemmDesc` + 一份 `GemmConfig`"渲染成一段可编译的 `bf16_gemm_impl<...>` 实例化代码，并说清它从生成、缓存到启动的每一步。明天（第 13 天）是第 2 周复盘：不看笔记，独立串起从 Python API 到 `bf16_gemm_impl` 的完整 Host 调用链。
