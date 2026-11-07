# 第 3 周 · 第 2 天（总第 16 天）：分析片上内存规划

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 3 周"BF16 Kernel——模板、调度与数据搬运"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

昨天（第 15 天）我们读完了 `bf16_gemm_impl` 的 26 个模板参数，并在 3.4（c）节预览了第 59-64 行的五行 `constexpr`。今天就把这五行读透：**L1、L0A、L0B、L0C 这四块片上存储是如何被"规划"出来的**——每个缓冲的形状由哪些模板参数决定、多级（multi-stage）缓冲如何用 `offset()` 一段接一段地分配地址、容量与对齐的 `static_assert` 如何在编译期守门，最后像 heuristics 那样**手工估算一组配置的片上用量**。kernel 里没有一行 `malloc`：片上内存规划完全是编译期算术，这正是"这些参数必须是模板参数"的回报。函数体里的调度、地址计算、搬运与同步，分别留给第 17～26 天。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 在 `deep_gemm/include/deep_gemm/bf16_gemm.hpp` 中指出 `l1a/l1b/l0a/l0b/l0c` 五个片上缓冲的声明位置（第 59-64 行）及各自所在的地址空间（`__cbuf__`/`__ca__`/`__cb__`/`__cc__`，来自 `ascend/tile_ptr.hpp:93-96` 的别名定义）；
2. 解释 `operand_ptr` / `result_ptr`（`ascend/tile_ptr.hpp:18-56、58-91`）的三个成员（`addr`、`shape_mn/shape_m`、`shape_k/shape_n`）和两个核心方法：`size_per_stage()`（tile_ptr.hpp:37、73）与 `offset(n)`（tile_ptr.hpp:38、74），并写出"第 i 级缓冲的地址 = 基址 + i × size_per_stage"；
3. 推导 L1 的总占用公式：`kNumL1Stages × BLOCK_M × depth_a × 2 + kNumL1Stages × BLOCK_N × BLOCK_K × 2`（`depth_a` 普通路径为 `BLOCK_K`，`kResidentA` 时为 `aligned(SHAPE_K, 16)`），并解释 `l1b` 的基址为什么是 `l1a.offset(kResidentA ? 1 : kNumL1Stages)`（第 60 行）；
4. 说明 L0A/L0B 按 `(MAD_M/MAD_N, MAD_K)` 开 `kNumL0Stages` 级、L0C 按 `(MAD_M, MAD_N)` 开 `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)` 级（第 51、62-64 行），以及 `l0c` 元素类型固定为 `float` 的原因（FP32 累加）；
5. 逐条列出容量类 `static_assert`（第 66-69 行）与对齐类断言（第 70 行、`epilogue.hpp:29-30`），并把它们与 Host 侧 `make_gemm_candidate`（`csrc/jit_kernels/heuristics.hpp:92-204`）中的容量检查一一对应；
6. 用一组具体配置（如 `block=128×512×128、mad=128×256×64、l1_stages=3、l0_stages=2`）手工算出 L1/L0A/L0B/L0C 的字节数、起止 offset 和利用率，并核对全部断言；
7. 说出四个 stage 游标 `s`（第 112 行）、`s_ab`（第 133 行）、`s_cd`（第 124 行）、`s_epi`（第 107、164 行）各自轮转哪块缓冲的哪些槽位，以及 `set_flags`（第 83-84 行）在流水开始前把所有槽位标记为"空闲"的作用；
8. （延伸）描述 `Epilogue` 对 UB 的 constexpr 规划（`epilogue.hpp:24-26`：前 `kNumStagesPerAIV` 级做 L0C→UB 中转，算子 scratch 紧随其后）及其容量断言 `op.ub_end() <= UBSizeBytes`（epilogue.hpp:35）。

**当天产出：一张 Kernel 片上内存布局图**（模板见第六节，参考答案见 3.8 节）。

## 二、预备知识

- **第 15 天（最重要）**：26 个模板参数的分类表；`BLOCK_M/N/K` 与 `MAD_M/N/K` 的两级切分；`kNumL1Stages/kNumL0Stages/kNumEpilogueStages` 三个 Stage 参数；`kResidentA/kResidentB` 的初步用途；第 59-64 行五行 `constexpr` 的预览。
- **第 12 天**：`GemmConfig` 五组字段；`select_gemm_config()` 的候选搜索；"Host 硬约束 ↔ Device 断言互为镜像"。
- **第 5 天**：`GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` 数据流；MTE2（GM→L1）、MTE1（L1→L0）、M（计算）等流水线的分工；各存储层的职责。
- **第 4 天**：BF16 输入、FP32 累加——`l0c` 固定 `float` 的原因。
- **第 3 天**：`BLOCK_M/N/K` 的 Tiling；一个输出 Tile 需要读哪些 A/B 数据。
- **C++ 语法**：`constexpr` 对象与常量求值、`static_assert`、带默认值的构造函数参数、类型别名（`using`）。能对照读懂即可。

## 三、详细讲解

### 3.1 全景：五行 constexpr 就是全部的"内存分配器"

先回顾容量常量（`deep_gemm/include/deep_gemm/common.hpp:50-63`，注释标明是 Ascend 950DT 的核拓扑）：

| 存储层 | 常量 | 容量 | 归属/用途（第 5 天） |
| :-- | :-- | :-- | :-- |
| L1 | `L1SizeBytes` | 512 KB | AIC 私有；GM→L0 的中转站（NZ 分形布局） |
| L0A | `L0ASizeBytes` | 64 KB | A 矩阵专用，MAD 的左操作数 |
| L0B | `L0BSizeBytes` | 64 KB | B 矩阵专用，MAD 的右操作数 |
| L0C | `L0CSizeBytes` | 256 KB | FP32 累加器，MAD 的输出 |
| UB | `UBSizeBytes` | 256 KB | AIV 侧统一缓冲，epilogue 后处理 |

BF16 kernel 对前四块的完整规划就是 `bf16_gemm.hpp` 第 59-64 行：

```cpp
constexpr l1_ptr<bfloat16_t, kMajorA> l1a(BLOCK_M, kResidentA ? aligned(SHAPE_K, get_frac_k<bfloat16_t>()) : BLOCK_K, 0);
constexpr l1_ptr<bfloat16_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kResidentA ? 1 : kNumL1Stages));

constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);
constexpr l0b_ptr<bfloat16_t> l0b(MAD_N, MAD_K);
constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
```

读法逐行拆解：

1. **`l1a`**：形状 `(BLOCK_M, BLOCK_K)`——一个 L1 级 A Tile 的逻辑尺寸；基址显式传 `0`（L1 地址空间从 0 开始）。第三个模板参数 `kMajorA` 只影响搬运方向和 L1→L0 是否转置（`ascend/copy.hpp:134-145` 按 major 分支），**不影响逻辑大小**：`size_per_stage = shape_mn × shape_k × sizeof(dtype)` 对两种 major 都成立。`kResidentA` 打开时深度从 `BLOCK_K` 换成 `aligned(SHAPE_K, get_frac_k<bfloat16_t>())`（`get_frac_k` 对 BF16 返回 `32/2 = 16`，common.hpp:72-79）——一个槽装下整条 K。
2. **`l1b`**：形状 `(BLOCK_N, BLOCK_K)`；**基址 = `l1a.offset(...)`**——B 区紧跟在 A 区后面，这就是"offset 分配"的第一现场。普通路径 A 区占 `kNumL1Stages` 级，resident A 时只占 1 级。
3. **`l0a/l0b/l0c`**：只传形状、不传基址（构造函数默认 `addr = 0`，tile_ptr.hpp:29、65）——L0A、L0B、L0C 是三个独立地址空间，各自从 0 开始，不需要像 L1 那样在内部再分 A/B 区。

紧接着的 5 条 `static_assert`（第 66-70 行）给这套规划上保险（3.7 节展开）。整个"分配器"没有运行期开销：所有地址在编译期折叠成常量。这也回答了昨天的思考题——**为什么 BLOCK/MAD/stages 必须是模板参数**：换成运行期参数，这五行就没法写。

### 3.2 带形状的片上指针：`size_per_stage` 与 `offset`

`ascend/tile_ptr.hpp` 提供了两套"形状化指针"。**操作数指针** `operand_ptr`（第 18-56 行，L1/L0A/L0B 用）：

```cpp
template <typename ptr_t, typename elem_t, Major MAJOR = Major::K>
struct operand_ptr {
    uintptr_t addr;
    const uint64_t shape_mn, shape_k;   // 逻辑形状（与布局无关）
    constexpr __aicore__ uint64_t size_per_stage() const { return shape_mn * shape_k / k_per_slot * sizeof(dtype_t); }  // 37
    constexpr __aicore__ uintptr_t offset(uint32_t n) const { return addr + n * size_per_stage(); }                     // 38
    constexpr __aicore__ operand_ptr<...> operator[](uint32_t i) const { /* r.addr = offset(i) */ }                     // 39-43
};
```

**结果指针** `result_ptr`（第 58-91 行，L0C/UB 用）结构相同，只是两个维度叫 `shape_m/shape_n`（输出没有"操作数主轴"的概念）。五个别名（第 93-97 行）把地址空间和缓冲类型绑定：

| 别名 | 地址空间限定符 | 物理存储 | 本周 kernel 中的实例 |
| :-- | :-- | :-- | :--- |
| `l1_ptr<T, M>` | `__cbuf__` | L1 | `l1a`、`l1b`（第 59-60 行） |
| `l0a_ptr<T>` | `__ca__` | L0A | `l0a`（第 62 行） |
| `l0b_ptr<T>` | `__cb__` | L0B | `l0b`（第 63 行） |
| `l0c_ptr<T>` | `__cc__` | L0C | `l0c`（第 64 行） |
| `ub_ptr<T>` | `__ubuf__` | UB | `Epilogue::ub`（epilogue.hpp:25） |

三个关键方法，一句话一个：

- **`size_per_stage()`**：一级（一个 stage 槽）的字节数 = 行数 × 列数 × 元素宽度。BF16 下 `l1a` 每级 = `BLOCK_M × BLOCK_K × 2`。
- **`offset(n)`**：从当前 `addr` 前进 n 级的字节偏移 = `addr + n × size_per_stage()`。它**只做加法**，这就是多级缓冲的分配原语。
- **`operator[](i)`**：返回一个 `addr = offset(i)` 的新指针——"第 i 级缓冲"。函数体里 `l1a[s]`、`l0a[s_ab]`、`l0c[s_cd]` 都是这个下标（第 110、117-118、138-139、127 行）。

于是"多阶段缓冲通过 offset 分配"可以精确表述成一个两层结构：

```text
层 1（区域分配，写死在声明里）：
    l1b.addr = l1a.offset(kNumL1Stages)     // B 区基址 = A 区末尾
层 2（槽位轮转，运行期游标）：
    l1a[s]、l1b[s]   s    ∈ [0, kNumL1Stages)        // 第 112 行
    l0a[s_ab]、l0b[s_ab]  s_ab ∈ [0, kNumL0Stages)   // 第 133 行
    l0c[s_cd]        s_cd ∈ [0, num_mad_cd_stages)   // 第 124 行
    ub[s_epi / kNumAIVs]（经 aic_store，epilogue.hpp:44-45）
                     s_epi ∈ [0, epilogue.kNumStages) // 第 164 行
```

这个"offset 链"是全仓库的通用惯用法：FP8 kernel 在 L1 里再串两段 scale factor 缓冲（`fp8_gemm.hpp:80-83`：`l1a → l1b → l1_sfa → l1_sfb`，每段的基址都是上一段的 `offset(...)`）；`fp8_dequant_gemm.hpp:175-178`、`sparse_mqa_logits.hpp:139-142`、`transform_sf.hpp:518-523` 也是同样的写法。学会读这一段，等于学会读所有 kernel 的内存规划。

### 3.3 L1 的规划：A 区 + B 区，`kNumL1Stages` 级流水

L1 被划成两个区域，每区域 `kNumL1Stages` 级（resident A 例外）：

```text
L1（__cbuf__，512KB，基址 0）
┌────────────────────────────── 0x00000  l1a[0]   ┐
│         A 区                  l1a[1]   │ 每级 BLOCK_M × depth_a × 2 B
│  (MTE2 写 → MTE1 读)          l1a[2]   ┘ s 轮转
├────────────────────────────── l1a.offset(kNumL1Stages) = l1b 基址
│         B 区                  l1b[0]   ┐
│  (MTE2 写 → MTE1 读)          l1b[1]   │ 每级 BLOCK_N × BLOCK_K × 2 B
│                               l1b[2]   ┘ s 轮转
├────────────────────────────── l1b.offset(kNumL1Stages)   ← 断言 66 检查的点
│         空闲（若有）
└────────────────────────────── 0x80000（512KB）
```

四个要点：

1. **为什么 A、B 同用一个 stage 数**：第 117-118 行，`copy_gm_to_l1(l1a[s], ...)` 和 `copy_gm_to_l1(l1b[s], ...)` 用**同一个 `s`**——第 `s` 级 A Tile 和第 `s` 级 B Tile 属于同一个 `BLOCK_K` 切片，一起被 MTE2 搬入、一起被 MTE1 消费，所以两区级数天然相同。Host 侧公式的 `(block_m + block_n) * block_k` 一项（heuristics.hpp:171）正是把两区合并计算。
2. **多级的意义**：第 5 天讲过 MTE2（GM→L1）与 MTE1（L1→L0）是两条可并行的流水。`kNumL1Stages` 级缓冲意味着"飞行中"的 `BLOCK_K` 切片有几份：MTE2 往第 `s+1` 级装下一个切片时，MTE1 还在从第 `s` 级取数。级数越多，两级流水越不容易互相等待，但每级都要吃 L1 容量——这是 Stage 参数的本质权衡。
3. **resident A 的变化**：A 区从 "`kNumL1Stages` 个 `BLOCK_K` 深的小槽"变成"**1 个 `aligned(SHAPE_K, 16)` 深的大槽**"（第 59 行），B 区基址相应前移到 `l1a.offset(1)`（第 60 行）。K 循环里 `if constexpr (!kResidentA)` 把逐块装载编译掉（第 117 行），A 只在进入 K 循环前搬一次（第 109-110 行），L1→L0A 从常驻槽 `l1a[0]` 按 K 偏移取数（第 143 行）。为什么深度要对齐到 16：`get_frac_k<bfloat16_t>() = 32 字节 / 2 字节 = 16`，L1 物理上按 32 字节的 C0 块存放（3.5 节）。
4. **L1 的物理布局是 NZ 分形**：`copy_gm_to_l1` 走 `asc_copy_gm2l1_nd2nz`（copy.hpp:122、127），把 GM 的 ND（自然行）布局转成片上 NZ 分形布局——以 16 行 × 32 字节为单位的小块矩阵。这解释了为什么后面所有对齐都围绕 16 和 32B 展开（3.5、3.7 节）。布局细节留到第 18-19 天。

### 3.4 L0A/L0B 的规划：MAD 尺寸 × `kNumL0Stages` 级

```cpp
constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);   // 每级 MAD_M × MAD_K × 2 B，共 kNumL0Stages 级
constexpr l0b_ptr<bfloat16_t> l0b(MAD_N, MAD_K);   // 每级 MAD_N × MAD_K × 2 B，共 kNumL0Stages 级
```

L0A/L0B 直接按 **MAD 尺寸**（一条矩阵乘指令的操作数尺寸）开槽，而不是 BLOCK 尺寸：`mad()`（copy.hpp:371-381）每次消费的就是一个 `(MAD_M, MAD_K)` 的 `l0a_tile` 和一个 `(MAD_N, MAD_K)` 的 `l0b_tile`（第 138-139 行的 `l0a[s_ab].as_mad_aligned(...)`）。BLOCK Tile 是被**一串 MAD 指令**逐块算掉的（第 123-124 行 `m_mad_idx/n_mad_idx` 双层循环）。

- **`s_ab` 游标**（第 133 行）：内层 `k_mad_idx` 循环每前进一个 `MAD_K`，`s_ab` 轮转一格。MTE1（L1→L0，第 143-145 行）与 M/Cube（`mad()`，第 149 行）这两条流水靠 `kNumL0Stages` 级缓冲重叠。
- **Host 侧永远选 2**：`c.num_l0_stages = 2`（heuristics.hpp:111-112，注释 "always 2 stage"）——当前启发式固定双缓冲。所以模板参数虽然存在，实际实例几乎都是 2。
- **MAD_K 被 L0B 容量卡住**：候选搜索时（heuristics.hpp:133-137）先算两个"能装下的最大 K"：

  ```cpp
  const int l0a_k = L0ASizeBytes * 8 / (num_l0_stages * mad_m * dtype.elem_bits);   // 133
  const int l0b_k = L0BSizeBytes * 8 / (num_l0_stages * mad_n * dtype.elem_bits);   // 134
  int mad_k = std::min({l0a_k, l0b_k, std::max(k_alignment, block_k / num_l0_stages)});  // 135
  ```

  再向下对齐到 `k_alignment` 并保证整除 `block_k`（第 136-137 行）。主例配置里 `l0a_k = 128`、`l0b_k = 64`，所以 `MAD_K = 64`——**不是随手选的，是被 L0B 容量精确卡住**。
- **resident B 的容量断言**（第 45 行）：`SHAPE_K <= kNumL0Stages * MAD_K && BLOCK_N == MAD_N`——整个 `[BLOCK_N, K]` 的 B Tile 要装进 L0B 的多级缓冲并常驻（N 方向不拆分、K 方向装得下），此时同一核连续处理相同 `n_idx` 的 Tile 就跳过 B 的搬运（第 93、118、144-145 行）。

### 3.5 L0C 的规划：`num_mad_cd_stages` 个 FP32 结果槽

```cpp
constexpr uint32_t num_mad_cd_stages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);  // 第 51 行
constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);                                    // 每级 MAD_M × MAD_N × 4 B
```

三个问题依次回答：

1. **级数为什么是 `(BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`？** 一个 BLOCK Tile 的输出由这么多块 `MAD_M × MAD_N` 的结果拼成（第 15 天 3.4（a））。`s_cd` 游标在 `(m_mad_idx, n_mad_idx)` 双层循环里轮转（第 124 行），于是**每个 MAD 结果块在整个 K 循环期间独占一个 L0C 槽**：`is_first`（`k_blk_idx==0 && k_mad_idx==0`）时清零（`mad` 的 `clear_C`，第 149-151 行），此后 `ENABLE_KEEP` 逐块累加，直到最后一个 `BLOCK_K` 才被 `copy_l0c_to_gm`（direct，第 158 行）或 `epilogue.aic_store`（普通，第 161 行）送出去。槽位轮转让"算第 i 块"与"送出第 i-1 块"重叠——Host 侧的对应约束是 `mad_m * mad_n * sizeof(float) <= L0CSizeBytes / 2`（heuristics.hpp:127，注释 "at least 2 stage in L0C, to overlap mad and store"）。
2. **元素类型为什么固定 `float`？** BF16 输入、FP32 累加（第 4 天）：MAD 在 L0C 里做宽精度累加，精度损失只发生在最后转回 BF16 输出的那一刻。所以每级字节数 = `MAD_M × MAD_N × 4`，比同尺寸的 BF16 缓冲大一倍——这就是为什么 BLOCK 尺寸的上界检查用 `block_m * block_n * sizeof(float) <= L0CSizeBytes`（heuristics.hpp:101-102）。
3. **BLOCK 与 L0C 的关系**：注意 L0C 里同时只放**一个 BLOCK Tile 的**结果（`num_mad_cd_stages` 块），不是多个 BLOCK。输出 Tile 之间靠 `s_cd` 从 `epi_base` 重新起算衔接（第 86、107、169 行——细节第 24-25 天）。

**对齐视角看四块缓冲**。L1/L0A/L0B/L0C 的物理布局都遵循 NZ 分形：`FRAC_MN = 16`（common.hpp:63，"NZ fractal M/N rows"）和 `FRAC_K = 32字节 / sizeof(T)`（BF16 为 16）分别是 M/N 方向和 K 方向的最小单位。直接证据是 `copy_l1_to_l0a/l0b` 的下标换算（copy.hpp:137-138）：

```cpp
asc_copy_l12l0a(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K, ...);
```

下标做的是**整除**——L1→L0 的搬运以"16 行 × 32 字节"的分形块为单位整体搬。所以 MAD_M 必须是 16 的倍数、MAD_K 必须是 16（BF16）的倍数，否则 `m_mad_idx`、`k_mad_idx` 走到边界 tile 时除不尽。运行期还有一个"软对齐"：`as_mad_aligned`（tile_ptr.hpp:52-55、88-90）把边界 tile 的实际尺寸向上对齐到 `FRAC_MN/FRAC_K` 再当作子 tile 用（第 126-127、138-139 行），配合 `eff_mad_n = aligned(min(...), get_frac_k<ub_dtype_t>())`（第 126 行）保证 L0C→UB 的行步长 32 字节对齐（epilogue.hpp:29 的断言原文）。

### 3.6（延伸）UB 的规划：Epilogue 自己的一段 offset 链

普通路径的最后一跳是 `L0C →(AIC, FIX)→ UB →(AIV)→ GM`，UB 的规划不写在 `bf16_gemm.hpp` 里，而在 `Epilogue`（`deep_gemm/include/deep_gemm/epilogue.hpp:24-26`）：

```cpp
static constexpr uint32_t kNumStages = kNumStagesPerAIV * kNumAIVs;   // = kNumEpilogueStages × 2
static constexpr ub_ptr<ub_dtype_t> ub{MAD_M, MAD_N};                 // L0C→UB 中转区
static constexpr operator_t op{ub.offset(kNumStagesPerAIV)};          // 算子 scratch 紧随其后
```

- **中转区**：`kNumStagesPerAIV` 级、每级 `MAD_M × MAD_N × sizeof(ub_dtype_t)`。`aic_store`（epilogue.hpp:41-47）把 `l0c_tile` 拷到 `ub[stage_idx]`，其中 `aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs`、`stage_idx = s_epi / kNumAIVs`——MAD 结果块在两个 AIV 间轮转分配（`copy_l0c_to_ub` 的 `dst_subblock_id` 参数指定接收方，epilogue.hpp:45）；每个 AIV 只 drain 分给自己的块（epilogue.hpp:57 的 `continue`）。
- **算子 scratch**：`Identity` 不用额外 UB（`ub_end() = ub_base`，operators.hpp:110）；`AddBias` 在中转区后面放一段 `bias_ub{1, BLOCK_N}`（operators.hpp:167）。断言 `op.ub_end() <= UBSizeBytes`（epilogue.hpp:35）守总容量。
- **Host 镜像**：`epilogue_bytes = mad_m*mad_n*epilogue_elem_bits/8 (+ bias/fp8 附加)`，`num_epilogue_stages = min(num_intra_block_events=16, UBSizeBytes / epilogue_bytes)`，再被"每核 wave 数 × num_cd_stages"封顶（heuristics.hpp:186-194）。注意 `epilogue.hpp:33` 还要求 `kNumStages >= num_cd_stages`——**stage 数不只受容量约束，还受事件数（16）和调度进度约束**，这是"容量是必要条件而非充分条件"的活例子。

今天只需记住布局结构；`aic_wait_free/aic_store/aiv_drain` 的同步配合留到第 25 天。

### 3.7 编译期守门员：容量与对齐的 `static_assert`

设备侧（`bf16_gemm.hpp`，行号见括号）与 Host 侧（`heuristics.hpp`）的镜像对照：

| 设备断言 | 行号 | 守护的约束 | Host 侧镜像（heuristics.hpp） |
| :-- | :-- | :-- | :-- |
| `l1b.offset(kNumL1Stages) <= L1SizeBytes` | 66 | L1 不溢出（A 区 + B 区总大小） | `num_l1_stages = min(num_max_l1_stages, (L1SizeBytes - sf_bytes) / ab_bytes)`（172-174），`ab_bytes = (block_m+block_n)*block_k*elem_bits/8`（171）；BF16 时 `num_max_l1_stages = num_pipe_events = 8`（168，事件数上界） |
| `l0a.offset(kNumL0Stages) <= L0ASizeBytes` | 67 | L0A 不溢出 | `l0a_k = L0ASizeBytes*8/(num_l0_stages*mad_m*elem_bits)` 反解出装得下的 `mad_k`（133） |
| `l0b.offset(kNumL0Stages) <= L0BSizeBytes` | 68 | L0B 不溢出 | `l0b_k = ...`（134）；resident B 时 `SHAPE_K <= num_l0_stages*mad_k`（设备第 45 行） |
| `l0c.offset(num_mad_cd_stages) <= L0CSizeBytes` | 69 | L0C 不溢出 | `block_m*block_n*sizeof(float) <= L0CSizeBytes`（101-102）+ `mad_m*mad_n*4 <= L0CSizeBytes/2`（127） |
| `BLOCK_K % MAD_K == 0` | 70 | K 块能被 MAD 整除切分 | `while (mad_k && block_k % mad_k) mad_k -= k_alignment;`（137） |
| `MAD_N % get_frac_k<ub_dtype_t>() == 0` | epilogue.hpp:29 | L0C→UB 行步长 32B 对齐（断言原文） | `align_n` 含 bias/fp8/major 三种来源的向量寄存器宽度约束（121） |
| `MAD_M % FRAC_MN == 0` | epilogue.hpp:30 | 分形行对齐除尽 MAD tile | `align_m = mn_major ? k_alignment : FRAC_MN`（120），搜索时 `block_m % (split_m*align_m)` 检查（125） |

三个观察：

1. **`l1b.offset(kNumL1Stages)` 为什么不是 `l1a.offset(...)`？** 因为 `l1b` 的基址已经站在 A 区末尾，`l1b.offset(kNumL1Stages)` = A 区末尾 + B 区大小 = **总占用**。而 `l1a.offset(kNumL1Stages)` 只是 A 区末尾（即 `l1b` 基址本身）。断言要检查的是总账。
2. **Host 是"先算后选"，Device 是"最后验收"**：heuristics 从容量**反解**出合法的 stage/MAD 组合并做成本比较；设备断言只做最终校验。即使有人绕过 heuristics 手写实例化，非法配置也会在**编译期**爆掉（JIT 编译 Bisheng 时）而不是在 NPU 上跑飞。
3. **`s` 转成 `event_t` 的隐含上界**：`asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(s))`（第 116 行）里 `s ∈ [0, kNumL1Stages)`，事件编号因此不能超过 `kNumPipeEvents = 8`（common.hpp:61）——BF16 下 L1 级数的上界 8 来自这里（heuristics.hpp:168），而不是容量。`set_flags<...>(第 83-84 行)` 在流水开始前把 `kNumL1Stages`/`kNumL0Stages` 个事件全部置位（sync.hpp:63-67），等价于"所有槽位初始空闲"。

### 3.8 完整演算：一组配置的片上内存布局（当天产出·参考答案）

沿用第 15 天 B-1 的设定：BF16 Normal，`M=128, N=64, K=256`，`bf16_gemm_nt`（`compiled_dims="nk"`，Identity，无累加），**假设**启发式选中 `block=128×512×128`、`mad=128×256×64`、`l1_stages=3`、`l0_stages=2`、`epi_stages=2`。BF16 每元素 2 字节，`l0c` 为 float 每元素 4 字节。

**第一步：每个缓冲的每级大小与总占用。**

| 缓冲 | 形状（shape_mn, shape_k） | 每级字节 = mn×k×elem | 级数 | 总占用 | 容量 | 利用率 | 断言 |
| :-- | :-- | :-- | :-- | :-- | :-- | :--- | :-- |
| `l1a` | (128, 128) | 128×128×2 = 32768 | 3 | 98304 | — | — | — |
| `l1b` | (512, 128) | 512×128×2 = 131072 | 3 | 393216 | — | — | — |
| **L1 合计** | | | | **491520** | 524288 | **93.75%** | 66 ✓ |
| `l0a` | (128, 64) | 128×64×2 = 16384 | 2 | **32768** | 65536 | 50% | 67 ✓ |
| `l0b` | (256, 64) | 256×64×2 = 32768 | 2 | **65536** | 65536 | **100%（压线）** | 68 ✓ |
| `l0c` | (128, 256) float | 128×256×4 = 131072 | 2 = num_mad_cd_stages | **262144** | 262144 | **100%（压线）** | 69 ✓ |

其中 `num_mad_cd_stages = (128/128) × (512/256) = 2`（第 51 行）。UB（每个 AIV）：中转区 = `kNumStagesPerAIV × 128×256×2 = 2×65536 = 131072`，Identity 无算子 scratch，利用率 50%（断言 epilogue.hpp:35 ✓）。

**第二步：画出带 offset 的布局图。**

```text
L1（512KB）                      L0A（64KB）  L0B（64KB）   L0C（256KB）
0x00000 ┬ l1a[0] 32KB            0x00000 ┬ l0a[0] 16KB  0x00000 ┬ l0b[0] 32KB  0x00000 ┬ l0c[0] 128KB
0x08000 ┤ l1a[1]                 0x04000 ┤ l0a[1]       0x08000 ┤ l0b[1]       0x20000 ┤ l0c[1]
0x10000 ┤ l1a[2]                         └ 0x08000(=64K)        └ 0x10000(=64K)         └ 0x40000(=256K)
0x18000 ┼ l1b[0] 128KB  ← l1a.offset(3)    ↑ s_ab∈{0,1}          ↑ s_ab∈{0,1}    ↑ s_cd∈{0,1}
0x38000 ┤ l1b[1]                    (空闲 32KB)           (压线)             (压线)
0x58000 ┤ l1b[2]
0x78000 ┼ 空闲 32KB      ← 断言 66 检查到 491520 ≤ 524288
0x80000 ┴ (=512KB)       ↑ s∈{0,1,2}
```

**第三步：核对对齐断言。** `BLOCK_K % MAD_K = 128 % 64 = 0` ✓（第 70 行）；`MAD_N % 16 = 256 % 16 = 0`、`MAD_M % 16 = 128 % 16 = 0` ✓（epilogue.hpp:29-30）；`epilogue.kNumStages = 2×2 = 4 ≥ num_mad_cd_stages = 2` ✓（epilogue.hpp:33）。

**第四步：验证 Host 侧会选出同样的数字。**

- `num_l1_stages = min(8, 524288 / ((128+512)×128×2)) = min(8, 524288/163840) = min(8, 3.2→3) = 3` ✓（heuristics.hpp:171-174，整数除法向下取整）；
- `l0a_k = 65536×8/(2×128×16) = 128`，`l0b_k = 65536×8/(2×256×16) = 64`，`mad_k = min(128, 64, max(16, 128/2)) = 64` ✓（第 133-135 行）——L0B 是 MAD_K 的绑定约束；
- `epilogue_bytes = 128×256×16/8 = 65536`，`num_epilogue_stages = min(16, 262144/65536 = 4)`，再被 `ceil_div(num_blocks=1, 32) × num_cd_stages=2 = 2` 封顶 → 2 ✓（第 188-193 行）。

**第五步：resident_a 变体**（`SHAPE_K=256`，对照第 15 天 B-3）：`l1a` 形状变为 (128, **256**)、每级 65536、只占 **1 级**；`l1b` 基址 = `l1a.offset(1)` = 0x10000；总占用 = 65536 + 3×131072 = **458752 ≤ 524288** ✓（余量从 32KB 增至 64KB）；K 循环 `num_k_blocks = ceil(256/128) = 2`，A 只在循环前搬一次（第 109-110 行）。`kResidentB` 仍不可用：`SHAPE_K=256 > kNumL0Stages×MAD_K = 128`（且 `BLOCK_N=512 ≠ MAD_N=256`，第 45 行）。

## 四、源码导读

今天的目标是**把 3.8 的每一行数字的公式来源落到真实代码上**。每站 5～10 分钟，未注明文件时均指 `deep_gemm/include/deep_gemm/bf16_gemm.hpp`。

### 站点 1：五行声明与容量断言（第 51、59-70 行，精读）

- 第 59 行：写出 `l1a` 的 `size_per_stage` 表达式，注意 resident 分支里 `aligned(SHAPE_K, get_frac_k<bfloat16_t>())` 的两个函数各在 common.hpp 的哪一行（48、72-79）；
- 第 60 行：`l1b` 基址表达式；问自己：普通路径与 resident 路径下 `l1b.offset(kNumL1Stages)` 各等于多少（用 3.8 的数字验算：491520 / 458752）；
- 第 66-70 行：逐条抄下 5 条断言，标注每条检查的是"总账"还是"单区"。

### 站点 2：`ascend/tile_ptr.hpp`（分配原语，精读）

- 第 18-56 行 `operand_ptr`：`size_per_stage()`（37）、`offset(n)`（38）、`operator[]`（39-43）、`as_mad_aligned`（52-55）——注意 55 行把 `shape_mn` 对齐到 `FRAC_MN`、`shape_k` 对齐到 `FRAC_K`；
- 第 58-91 行 `result_ptr`：对照找出同名方法的行号（73、74、88-90）；
- 第 93-97 行：五个别名与地址空间限定符的对应表。

### 站点 3：函数体里的槽位消费（第 81-172 行，扫读，只找下标）

只做"寻宝"，不读逻辑：`l1a[...]`（109-110、117、143）、`l1b[...]`（118、145）、`l0a[...]`/`l0b[...]`（138-139）、`l0c[...]`（127）。每个出现处写一行"哪个游标、哪条流水在读写"。再看第 83-84 行 `set_flags` 的模板实参（`kNumL1Stages`/`kNumL0Stages`），结合 `ascend/sync.hpp:63-67` 理解"流水开始前把所有事件置位 = 所有槽位空闲"。

### 站点 4：`deep_gemm/include/deep_gemm/common.hpp`（容量与分形常量）

- 第 50-63 行：`kNumAICores=32`、五个容量常量、`kNumPipeEvents=8`、`FRAC_MN=16`——断言的分母全在这；
- 第 47-48 行 `aligned` 与第 72-79 行 `get_frac_k`：`get_frac_k<bfloat16_t>()` 现场算一遍（32/2=16）。

### 站点 5：`deep_gemm/include/deep_gemm/epilogue.hpp` + `epilogue/operators.hpp`（UB 规划）

- epilogue.hpp:24-26：`kNumStages = kNumStagesPerAIV × kNumAIVs`、`ub{MAD_M, MAD_N}`、`op{ub.offset(kNumStagesPerAIV)}`——又一条 offset 链；
- epilogue.hpp:28-35：8 条断言里挑出容量类（35）与对齐类（29、30）、调度类（33）各一条，说明各自守护什么；
- operators.hpp:110（Identity 的 `ub_end`）与 167-169（AddBias 的 `bias_ub`/`ub_end`）：算子 scratch 如何接在中转区后面。

### 站点 6：`csrc/jit_kernels/heuristics.hpp`（Host 镜像，验证对号）

- 第 101-102 行（BLOCK 进 L0C）、111-112 行（l0_stages 恒 2）、127 行（MAD tile ≤ L0C/2）；
- 第 133-137 行：`l0a_k/l0b_k/mad_k`——用 3.8 第四步的数字验算；
- 第 167-174 行：`num_l1_stages`——注意 168 行的上界是 `num_pipe_events`（8）而不是容量；
- 第 186-194 行：`num_epilogue_stages`——注意 190 行的上界 16 来自 `num_intra_block_events`。

### 站点 7：`deep_gemm/include/deep_gemm/fp8_gemm.hpp:80-94`（同惯用法，对照）

- 看 `l1a → l1b → l1_sfa → l1_sfb` 的四段 offset 链和末尾的 `l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes`：多两个 scale factor 区后，"总账断言"检查的是**链上最后一个缓冲的末端**；
- 第 93-94 行多出的 `BLOCK_K % MAD_K`、`MAD_K % 64`：FP8 路径额外的对齐约束。

**阅读检查点**（能回答再进入实践）：

1. `l0a` 和 `l1a` 都是操作数缓冲，为什么一个按 MAD 尺寸开槽、一个按 BLOCK 尺寸开槽？
2. `l1b.offset(kNumL1Stages)` 和 `l1a.offset(kNumL1Stages)` 各代表什么地址？
3. `num_mad_cd_stages` 与 `kNumL0Stages` 是不是同一个概念？各自轮转的循环在哪一行？
4. Host 侧 `num_l1_stages` 的上界 8 由什么决定？`num_epilogue_stages` 的上界 16 呢？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：打印配置并手工核算（约 25 分钟）。**

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day16_run1.log
grep GemmConfig /tmp/day16_run1.log | head -20
```

每行配置形如 `GemmConfig(block=MxNxK, mad=mxnxk, l1_stages=.., l0_stages=.., epi_stages=.., ...)`（格式定义在 `csrc/jit_kernels/gemm_config.hpp:166-170`）。挑 **3 个不同 shape** 的配置，对每个配置：按 3.8 的表格算出 L1/L0A/L0B/L0C 总占用与利用率，核对 `l1b.offset(l1_stages) ≤ 512KB` 等四条断言全部成立。特别注意寻找 **L0B 压线**（`l0_stages × mad_n × mad_k × 2 = 65536`）的行——`mad_k` 应恰好不超过 `l0b_k = 65536×8/(2×mad_n×16)`。

**步骤 2：观察 `l1_stages` 随 shape 的变化（约 15 分钟）。**

```bash
DG_PRINT_CONFIGS=1 python - <<'EOF' 2>&1 | tee /tmp/day16_run2.log
import torch, torch_npu  # noqa
import deep_gemm

for N in (64, 512, 2048):
    a = torch.randn(128, 256, dtype=torch.bfloat16, device='npu')
    b = torch.randn(N, 256, dtype=torch.bfloat16, device='npu')
    d = torch.empty(128, N, dtype=torch.bfloat16, device='npu')
    deep_gemm.bf16_gemm_nt(a, b, d)
torch.npu.synchronize()
EOF
```

对每个 N 记录 `block_n` 与 `l1_stages`，用 `num_l1_stages = min(8, 524288 / ((block_m+block_n)×block_k×2))` 验证打印值。思考：为什么 `block_n` 变大时 `l1_stages` 倾向于变小？

**步骤 3（可选）：bias epilogue 的 UB（约 10 分钟）。** 给 `bf16_gemm_nt` 传 `bias` 再跑一次并打印配置：`epi_stages` 是否变化？对照 heuristics.hpp:188-189（`epilogue_bytes` 加上 `block_n × elementSize`）解释。

### B. 无 Ascend 环境：静态替代任务

**B-1：闭卷核算主例（约 20 分钟）。** 合上教程，只看 `bf16_gemm.hpp` 第 51、59-70 行，对配置 `block=128×512×128、mad=128×256×64、l1_stages=3、l0_stages=2` 重算 3.8 的完整表格（六个总占用数字 + 四条断言 + 各缓冲利用率），再对照订正。

**B-2：判断新配置的合法性（约 15 分钟）。** 对下面两组配置，只依据第 66-70 行与 epilogue.hpp:29-35 判断能否通过全部断言；不能的指出第一条失败的断言：

- 配置 X：`block=128×256×128、mad=128×128×64、l1_stages=4、l0_stages=2、epi_stages=2`；
- 配置 Y：`block=128×512×128、mad=128×256×128、l1_stages=3、l0_stages=2、epi_stages=2`。

（参考：X 全部通过——L1 = 4×32768 + 4×65536 = 393216 ≤ 524288，L0A = 32768，L0B = 32768，L0C = 2×65536 = 131072，128%64=0；Y 的 L0B = 2×256×128×2 = 131072 > 65536，断言 68 失败——Host 侧 `l0b_k = 65536×8/(2×256×16) = 64` 本来就会把 `mad_k` 压回 64。）

**B-3：像 heuristics 一样反解（约 15 分钟）。** 设 `block=128×512×128`、`mad_m=128`、`mad_n=256`、BF16、`num_l0_stages=2`：按 heuristics.hpp:133-135 算 `l0a_k`、`l0b_k`、`mad_k`；再按 171-174 算 `num_l1_stages`；最后按 188-193 算 `num_epilogue_stages`（取 `M=128, N=64`，`num_cores=32`）。与 3.8 第四步对照——你刚刚手工复现了 `select_gemm_config` 的容量侧。

**B-4：resident 变体与压线分析（约 10 分钟）。** 同一配置打开 `resident_a`（`SHAPE_K=256`）：重写 `l1a` 形状与 `l1b` 基址，重算 L1 总占用与余量（对照 3.8 第五步：458752，余 65536）。然后回答：若 `SHAPE_K=1024`，`l1a` 大槽 = 128×1024×2 = 262144，B 区还能留几级？（`(524288-262144)/131072 = 2` 级——但 resident 要求 `num_l1_stages > 2`（heuristics.hpp:548-555 的条件，第 15 天 3.7），这组 K 会被 Host 否决。）

**B-5：闭卷画布局图（约 15 分钟）。** 凭记忆重画 3.8 第二步的四缓冲布局图（含每个槽的 offset、游标取值、断言检查点），对照订正——订正后的图就是当天产出。

## 六、当天产出

学习计划要求今天的产出是**一张 Kernel 片上内存布局图**。按下述模板填写（B-1～B-4 的演算附在后面）：

```text
# DeepGEMM-Ascend 第 16 天记录：分析片上内存规划

## A. 片上内存布局图（闭卷填写后对照 3.8 订正）
配置：block=＿×＿×＿，mad=＿×＿×＿，l1_stages=＿，l0_stages=＿，epi_stages=＿
L1（基址 0）：A 区 [0x＿＿, 0x＿＿) 共＿级×＿B；B 区 [0x＿＿, 0x＿＿) 共＿级×＿B；总占用＿B / 512KB（＿%）
L0A：＿级×＿B = ＿B / 64KB（＿%）；游标＿＿轮转，范围 [0, ＿)
L0B：＿级×＿B = ＿B / 64KB（＿%）
L0C：num_mad_cd_stages = ＿×＿ = ＿；每级＿B；总占用＿B / 256KB（＿%）
UB（每 AIV）：中转区＿级×＿B + 算子 scratch＿B = ＿B / 256KB
resident_a 变体：l1a 形状=（＿,＿) 占＿级；l1b 基址=＿；总占用＿B

## B. 公式与断言核对
- l1b 基址表达式：＿＿＿；L1 总账断言：＿＿＿ ≤ ＿＿＿（第＿行）
- l0c 级数 = ＿＿＿（第＿行）；l0c 元素类型 = ＿＿，因为＿＿＿
- 对齐断言三连：BLOCK_K % ＿＿ == 0；MAD_N % ＿＿ == 0；MAD_M % ＿＿ == 0
- Host 镜像公式：num_l1_stages = ＿＿＿（上界＿来自＿＿＿）；mad_k = min(＿＿, ＿＿, ＿＿)

## C. 手工演算记录（B-1～B-4）
- 主例六个占用数字与四条断言核对结果
- 配置 X / Y 的合法性判断与第一条失败断言
- l0a_k / l0b_k / mad_k / num_l1_stages / num_epilogue_stages 的反解过程

## D. 实验记录（有环境时）
- 3 个 GemmConfig 行的核算表（含利用率、L0B 是否压线）
- l1_stages 随 block_n 变化的观察与解释

## E. 遗留问题（交给后续天）
- s/s_ab/s_cd/s_epi 四个游标的同步配对（第 23 天）
- gm_ptr/l1_ptr 的完整地址计算（第 18 天）
- copy_gm_to_l1 的 NZ 布局参数（第 19 天）
- aic_store/aiv_drain 的 UB 消费（第 25 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `l1a/l1b/l0a/l0b/l0c` 的声明各在哪一行？各自通过哪个别名绑定到哪个地址空间？
2. 写出 `operand_ptr::size_per_stage()` 与 `offset(n)` 的公式。`l1a[2]` 的地址等于什么？
3. `l1b` 的基址表达式是什么？容量断言为什么写 `l1b.offset(kNumL1Stages)` 而不是 `l1a.offset(...)`？
4. `kResidentA` 打开后，`l1a` 的形状、级数和 `l1b` 的基址分别怎么变？为什么深度要对齐到 `get_frac_k<bfloat16_t>()`？
5. `l0c` 的级数由什么公式决定（给出行号）？为什么 `l0c` 的元素类型固定是 `float`？Host 侧哪一行保证了"L0C 至少两级"？
6. Host 侧 `num_l0_stages` 固定为多少？`MAD_K` 是如何被 L0A/L0B 容量约束的（写出 `l0a_k/l0b_k` 公式）？
7. `num_l1_stages` 的 Host 公式是什么？它的上界 8 来自容量还是别的东西？
8. `FRAC_MN` 和 `get_frac_k<bfloat16_t>()` 各等于多少？`copy_l1_to_l0a` 中哪一行代码说明片上搬运以这两个粒度为单位？
9. `Epilogue` 的 UB 是怎么规划的？`ub.offset(kNumStagesPerAIV)` 之后放的是什么？Identity 与 AddBias 的 UB 占用有何差别？
10. （思考）为什么整套规划能写成 `constexpr`？如果 `BLOCK_K` 变成运行期参数，3.8 的表格里哪些数字无法在编译期确定、哪些断言无法再写？
11. 配置 `block=128×512×128、mad=128×256×128、l1_stages=3、l0_stages=2` 会先挂在哪条断言上？Host 侧对应哪一行检查？

**参考答案：**

1. `bf16_gemm.hpp:59-60`（l1a/l1b）、62（l0a）、63（l0b）、64（l0c）。`l1_ptr→__cbuf__`（L1）、`l0a_ptr→__ca__`、`l0b_ptr→__cb__`、`l0c_ptr→__cc__`、`ub_ptr→__ubuf__`，别名定义在 `ascend/tile_ptr.hpp:93-97`。
2. `size_per_stage = shape_mn × shape_k / k_per_slot × sizeof(dtype)`（tile_ptr.hpp:37，BF16 无打包时 k_per_slot=1）；`offset(n) = addr + n × size_per_stage`（38）。`l1a[2].addr = 0 + 2 × BLOCK_M × BLOCK_K × 2`（operator[]，39-43）。
3. `l1b.addr = l1a.offset(kResidentA ? 1 : kNumL1Stages)`（第 60 行）。`l1b` 基址已站在 A 区末尾，`l1b.offset(kNumL1Stages)` = A 区 + B 区总大小，才是要和 `L1SizeBytes` 比较的总账；`l1a.offset(kNumL1Stages)` 只是 A 区末尾。
4. `l1a` 形状变为 `(BLOCK_M, aligned(SHAPE_K, 16))`、只占 1 级（第 59 行）；`l1b` 基址变为 `l1a.offset(1)`（第 60 行）。对齐到 16 是因为 `get_frac_k<bfloat16_t>() = 32字节/2字节 = 16`：L1 按 32 字节 C0 块存放，深度必须是 16 的倍数才能整块分配。
5. `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`（第 51 行）。`float` 是因为 BF16 输入需要 FP32 累加（MAD 在 L0C 里宽精度累加，第 4 天）。Host 侧 `mad_m*mad_n*sizeof(float) <= L0CSizeBytes/2`（heuristics.hpp:127）保证单个 MAD tile 至多占 L0C 一半，即至少两级。
6. 恒为 2（heuristics.hpp:111-112）。`l0a_k = L0ASizeBytes×8/(num_l0_stages×mad_m×elem_bits)`、`l0b_k` 同理把 `mad_m` 换 `mad_n`（133-134）；`mad_k = min(l0a_k, l0b_k, max(k_alignment, block_k/num_l0_stages))` 再向下对齐（135-137）。主例中 `l0b_k = 64` 是绑定约束。
7. `num_l1_stages = min(num_max_l1_stages, (L1SizeBytes - sf_bytes) / ab_bytes)`，BF16 时 `sf_bytes=0`、`ab_bytes = (block_m+block_n)×block_k×2`（heuristics.hpp:171-174）。上界 8 来自 `num_max_l1_stages = num_pipe_events`（168）——stage 游标 `s` 被转成 `event_t` 用于流水同步（第 116-121 行），事件编号上限 `kNumPipeEvents = 8`（common.hpp:61），不是容量。
8. `FRAC_MN = 16`（common.hpp:63）；`get_frac_k<bfloat16_t>() = 16`（common.hpp:73-79）。`copy_l1_to_l0a` 里 `asc_copy_l12l0a(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K, ...)`（copy.hpp:137）——下标做整除，说明搬运以 16 行 × 32 字节的分形块为单位。
9. `ub{MAD_M, MAD_N}` 的前 `kNumStagesPerAIV` 级做 L0C→UB 中转，`op{ub.offset(kNumStagesPerAIV)}` 把算子 scratch 接在中转区之后（epilogue.hpp:24-26）；`kNumStages = kNumStagesPerAIV × kNumAIVs` 个逻辑槽按 `s_epi % kNumAIVs` 轮转分配给 AIV。Identity 的 `ub_end() = ub_base`（无 scratch，operators.hpp:110）；AddBias 额外占 `bias_ub{1, BLOCK_N}` 的 `kNumStagesPerAIV` 级（operators.hpp:167-169）。
10. 因为五个缓冲的形状全部由模板参数（BLOCK/MAD/stages/SHAPE_K）决定、基址是常量 0 或上一段的 `offset()`，全部算术都是编译期常量折叠。若 `BLOCK_K` 运行期化：`l1a/l1b/l0a/l0b` 的 `size_per_stage`、所有 offset、`l1b` 基址都变成运行期值；第 66-70 行的容量/整除断言无法再写；`if constexpr (!kResidentA)` 这类分支消除也随之失效（`l1a[2]` 之类的常量地址不存在了）。
11. 先挂第 68 行 `l0b.offset(kNumL0Stages) <= L0BSizeBytes`：2 × 256×128×2 = 131072 > 65536。Host 侧对应 heuristics.hpp:134 的 `l0b_k = 64 < 128`——搜索时 `mad_k` 会被压到 64，根本不会生成这个实例。

---

今天结束时，你的手里应该有一张带 offset、级数、游标和断言检查点的四缓冲（外加 UB）布局图，并且能从模板参数一路手算到"某级缓冲的起始地址"。明天（第 17 天）把镜头从存储搬到任务：阅读 `scheduler.hpp`，看 M/N 维如何切块并分配给 32 个 AI Core。
