# 第 4 周 · 第 3 天（总第 24 天）：理解 MAD 计算与结果暂存

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 4 周"BF16 Kernel——MAD、流水线与 Epilogue"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 45 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

前两天我们搭好了脚手架：第 22 天拆开 K 维两层循环，认识了 `is_first` / `is_last` 两个标志；第 23 天配齐了 MTE1/MTE2/M 三条队列的四组同步事件。今天聚焦这些同步所守护的计算本身——`bf16_gemm.hpp:149-151` 那一行 `mad()` 调用。要回答四个问题：**一条 mmad 指令到底做什么**（L0A/L0B 的 BF16 操作数如何变成 L0C 里的 FP32 部分和）？**`ENABLE_UPDATE`、`ENABLE_KEEP` 和首次累加（`clear_C`）如何决定 L0C 累加器的生命周期**？**`mad()` 前后的同步为什么只守 L0A/L0B、不守 L0C**？**多个 MAD Tile 如何拼出一个 BLOCK_M × BLOCK_N 的完整输出**？当天产出是一段 **MAD 执行过程的伪代码**（参考答案见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 `mad()` 封装的签名与全部参数含义（`ascend/copy.hpp:370-381`），解释形状参数 `(m, n, k)` 到 `asc_mmad(m, k, n)` 的顺序调换、`unit_flag` 与 `clear_C` 两个控制参数在指令中的落点，以及 `sizeof(TA) == 1 || sizeof(TB) == 1` 时改走 `asc_mmad_mx` 的分支条件；
2. 画出 `mad()` 调用前后的同步序列（`bf16_gemm.hpp:141-152`：wait 空闲 → 条件搬运 → notify 就绪 → wait 就绪 → **mad** → notify 空闲），说明这组事件守护的是 L0A/L0B stage 而非 L0C，并解释为什么；
3. 描述 L0C 累加器的三段生命周期：首条 mmad `clear_C = true`（从 0 开始，覆盖上一输出块的残留）→ 中间步 `ENABLE_KEEP`（部分和保持在累加链中）→ 末条 `ENABLE_UPDATE`（提交，供写回消费）；并指出读取侧 `copy_l0c_to_ub` / `copy_l0c_to_gm` 的 `unit_flag` 默认值也是 `ENABLE_UPDATE`（`copy.hpp:275, 296`）；
4. 给定 BLOCK/MAD 参数，计算 `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)`（`bf16_gemm.hpp:51`）、每个输出子块固定映射到的 `s_cd` 编号和 L0C 容量占用，并用内核断言（`:69`）与启发式的半容量检查（`heuristics.hpp:127`）验证；
5. 手工推演多个 MAD Tile 组成一个 Block 输出的完整过程：2×2 子块的 `s_cd` 扫描顺序、跨 `k_blk` 的持续累加、最后一个 `k_blk` 结束时每个子块一次的 `aic_store` / `copy_l0c_to_gm`，以及 `s_epi` 如何轮转并路由到两个 AIV（`epilogue.hpp:43-44`）；
6. 对比三种同构用法：多步累加形态（`bf16_gemm.hpp:150`、`fp8_gemm.hpp:229`、`fp8_dequant_gemm.hpp:321`、`mega_moe.hpp:599`）、单发退化形态（`mqa_logits.hpp:396-397` 等：`UPDATE` 与 `clear_C = true` 同时成立）和读取形态（`copy.hpp:275, 296`），说明"何时清零、何时提交"是仓库级统一契约；
7. 独立写出当天产出：一段覆盖同步、标志、tile 选择与写回的 MAD 执行过程伪代码；有 Ascend 环境时用 `DG_PRINT_CONFIGS=1` 核对真实 `mad` 配置并验证自己的容量账本。

**当天产出：一段 MAD 执行过程的伪代码**（模板见第六节，参考答案见 3.8 节）。

## 二、预备知识

- **第 4 天**：BF16 输入、FP32 累加——`common.hpp:57` 把 L0C 标注为 `fp32 accumulator`，内核里 `l0c` 的类型就是 `l0c_ptr<float>`（`bf16_gemm.hpp:64`）。
- **第 15 天**：模板参数中的 Block 组（`BLOCK_M/N/K`）与 MAD 组（`MAD_M/N/K`）。
- **第 16 天**：`l1a/l1b/l0a/l0b/l0c` 的 `constexpr` 声明与多阶段缓冲 offset（`bf16_gemm.hpp:59-64`）、容量断言（`:66-69`）：L1 512KB、L0A/L0B 各 64KB、L0C 256KB（`common.hpp:54-57`）。
- **第 18 天**：`operand_ptr` / `result_ptr` 的地址表达；B 侧 tile 形状是 `MAD_N × MAD_K`（B 矩阵按"输出列 = 行"存放，NT 布局）。
- **第 22 天**：K 维两层循环；`is_first` / `is_last` 的定义（`:134-135`）；`eff_mad_m/n/k`；`as_mad_aligned` 对齐 tile 而 `mad()` 取真值；3.5 节已给出 `mad()` 封装的初览（今天展开）。
- **第 23 天**：四组配对事件的方向与队列模型（MTE2/MTE1/M）；"事件号 = 资源编号"；块内（intra-block）同步的接口。
- **C++ 语法**：函数模板、默认参数（`bool clear_C = false`）、`if constexpr` 分支。能对照读懂即可。

另外说明：`asc_mmad` / `asc_mmad_mx` / `asc_unit_flag_mode` 声明在 CANN 工具链的 `c_api/asc_simd.h`（`copy.hpp:3` 引入），该头文件在**本仓库之外**。本文对指令级语义的表述以仓库内全部用法能反推出的契约为准，不凭空补充硬件行为。

## 三、详细讲解

### 3.1 全景：一条 mmad 在数据流中的位置

先回忆整体数据流（第 5 天）：

```text
GM ──MTE2──> L1 ──MTE1──> L0A/L0B ──M──> L0C ──Epilogue──> GM
```

今天的主角是中间那段：**L0A/L0B → M 队列 → L0C**。调用点在 `bf16_gemm.hpp:149-151`：

```cpp
mad(l0c_tile, l0a_tile, l0b_tile, eff_mad_m, eff_mad_n, eff_mad_k,
    is_last ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
    is_first);
```

一条 mmad 语义上完成的事：把 A 侧 tile（`eff_mad_m × eff_mad_k`，BF16）与 B 侧 tile（`eff_mad_n × eff_mad_k`，BF16）相乘，把 `eff_mad_m × eff_mad_n` 个输出元素的 K 维点积累加进 L0C 的 FP32 累加器。三个"从哪来、到哪去、谁来管"：

```text
            ┌── 本条 mmad 的操作数（L0 双缓冲的当前 stage）──┐
            │                                              │
  L1[s] ─MTE1─> L0A[s_ab] ──┐
  L1[s] ─MTE1─> L0B[s_ab] ──┴─ M 队列: asc_mmad ──> L0C[s_cd]（FP32 累加器，跨 k_blk 持续累加）
                                (m, k, n)              │
                                                       │ 最后一个 k_blk 的 K 遍历结束后
                                                       ▼
                         kDirectStore ? L0C ──FIX──> GM（:156-159）
                                     :   L0C ──FIX──> UB[s_epi] ──AIV──> GM（:161）
```

三个 stage 环各管一段：`s_ab` 标识**本条 mmad 读哪份** L0A/L0B；`s_cd` 标识**写到哪份** L0C 累加器；`s_epi` 标识结果**倒进哪个** UB stage（交给哪个 AIV）。`s_ab` 每 `MAD_K` 步转一格（第 22 天），`s_cd` 每换一个输出子块转一格，`s_epi` 每次写回转一格——三个环转速互不相同，这是今天要建立的空间感。

### 3.2 `mad()` 封装逐参数解读

`ascend/copy.hpp:370-381` 全文只有 11 行：

```cpp
template <typename TA, typename TB, typename TC>
__aicore__ inline void mad(l0c_ptr<TC> l0c, l0a_ptr<TA> l0a, l0b_ptr<TB> l0b,
                           uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           asc_unit_flag_mode unit_flag, bool clear_C = false) {
    if constexpr (sizeof(TA) == 1 || sizeof(TB) == 1) {
        asc_mmad_mx(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n, static_cast<uint8_t>(unit_flag), true,
                    false, clear_C);
    } else {
        asc_mmad(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n, static_cast<uint8_t>(unit_flag), true,
                 false, clear_C);
    }
}
```

BF16 路径下三个 tile 参数的实例化类型（`bf16_gemm.hpp:62-64` 与 `tile_ptr.hpp:93-97`）：

| 参数 | 类型 | 地址空间 | 来源（bf16_gemm.hpp） |
|---|---|---|---|
| `l0c` | `l0c_ptr<float>` = `result_ptr<__cc__ float*, float>` | `__cc__`（L0C） | `:127` `l0c[s_cd].as_mad_aligned(eff_mad_m, eff_mad_n)` |
| `l0a` | `l0a_ptr<bfloat16_t>` = `operand_ptr<__ca__ ...>` | `__ca__`（L0A） | `:138` `l0a[s_ab].as_mad_aligned(eff_mad_m, eff_mad_k)` |
| `l0b` | `l0b_ptr<bfloat16_t>` = `operand_ptr<__cb__ ...>` | `__cb__`（L0B） | `:139` `l0b[s_ab].as_mad_aligned(eff_mad_n, eff_mad_k)` |

地址空间限定符（`__ca__` / `__cb__` / `__cc__`）本身就是"这块数据驻留在哪个物理缓冲"的类型级表达——第 18 天的 `tile_ptr` 体系在 mmad 这里收口。三个形状参数与两个控制参数：

| 参数 | BF16 路径的取值 | 含义 |
|---|---|---|
| `shape_m` | `eff_mad_m = min(actual_m - m_mad_idx, MAD_M)`（`:125`，不对齐） | 本条 mmad 的 M 维真实行数 |
| `shape_n` | `eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N), get_frac_k<ub_dtype_t>())`（`:126`） | N 维列数（已按 epilogue 的 UB dtype 对齐，见下） |
| `shape_k` | `eff_mad_k = min(actual_k - k_mad_idx, MAD_K)`（`:137`，不对齐） | K 维真实深度（尾部块可能小于 `MAD_K`） |
| `unit_flag` | `is_last ? ENABLE_UPDATE : ENABLE_KEEP` | 累加链控制：末步提交 / 中间保持（3.4 节） |
| `clear_C` | `is_first`（默认 `false`） | 首步清零：累加起点为 0（3.4 节） |

四个值得停下咀嚼的细节：

**其一，`(m, n, k) → (m, k, n)` 的顺序调换。** `mad()` 按 GEMM 的习惯 `(m, n, k)` 接收形状，内部调 `asc_mmad` 时变成 `(shape_m, shape_k, shape_n)`——K 和 N 互换了位置。追踪 `asc_mmad` 参数时务必记住这一点。

**其二，`eff_mad_n` 的对齐量随 epilogue 算子变化。** `:126` 把 N 对齐到 `get_frac_k<ub_dtype_t>()`：`Identity` 算子的 `ub_dtype_t = cd_dtype_t`（`epilogue/operators.hpp:98`），输出 BF16 时为 16；输出 FP32 或使用 `ScaleByAlpha` / `AddBias`（`ub_dtype_t = float`，`operators.hpp:128, 164`）时为 8（`common.hpp:73-79`）。动机写在 `epilogue.hpp:29` 的断言里：`"L0C-to-UB row stride must be 32-byte aligned"`——`MAD_N % get_frac_k<ub_dtype_t>() == 0` 保证 L0C→UB 每行搬运 32 字节对齐。对比第 22 天：操作数侧的 `as_mad_aligned` 把 K 对齐到 `get_frac_k<T>()`，结果侧（`result_ptr::as_mad_aligned`，`tile_ptr.hpp:88-90`）则把 M、N **都**对齐到 `FRAC_MN = 16`——同名函数、两个类、对齐规则不同。

**其三，tile 形状对齐、计算形状取真值，两件事并存。** 传给 `mad()` 的 `eff_mad_m` / `eff_mad_k` 是原始真值（如 42），而 tile 描述符的形状已被 `as_mad_aligned` 抬到对齐值（如 48）——对齐值决定 L1→L0 搬多少（`copy_l1_to_l0a` 用 `dst.shape_k / FRAC_K` 计算搬运量，`copy.hpp:134-145`），真值决定 mmad 算多深。第 22 天在 K 维看过的事实，在 M/N 维同样成立。

**其四，`asc_mmad_mx` 分支。** `sizeof(TA) == 1 || sizeof(TB) == 1`（FP8/FP4 是 1 字节类型）时走带 scale 的矩阵乘变体，三个控制参数（`unit_flag`、两个固定字面量、`clear_C`）完全相同。也就是说，今天建立的 `unit_flag` / `clear_C` 契约对低精度路径同样适用（第 32 天展开）。

关于两个固定字面量 `true, false`：沿用第 22 天的解读，它们对应 `asc_mmad` 声明中的 `k_direction_align` 与 `c_matrix_source` 位置参数，`clear_C` 对应 `c_matrix_init_val`。这三个名字来自 CANN 的 `c_api` 头文件（仓库之外），仓库所有调用点传的都是同一组值，本文不展开指令级细节。

### 3.3 mad 前后的同步：守门的是 L0A/L0B，不是 L0C

把 `bf16_gemm.hpp:141-152` 的六行摊开（同步方向沿用第 23 天的"源 → 目标"记法）：

```cpp
asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(s_ab));       // :141  MTE1 等 M 归还 L0[s_ab]
if (BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0)
    copy_l1_to_l0a(l0a_tile, l1a[kResidentA ? 0 : s], m_mad_idx, ...); // :143  MTE1: L1 → L0A[s_ab]
if ((!kResidentB || !reuse_b) && (BLOCK_M == MAD_M || ... || m_mad_idx == 0))
    copy_l1_to_l0b(l0b_tile, l1b[s], n_mad_idx, k_mad_idx);           // :145  MTE1: L1 → L0B[s_ab]
asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(s_ab));      // :146  宣布 L0[s_ab] 就绪
asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(s_ab));        // :148  M 等 L0[s_ab] 就绪
mad(l0c_tile, l0a_tile, l0b_tile, ...);                              // :149-151 M 队列: 消费 L0[s_ab]
asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(s_ab));      // :152  宣布 L0[s_ab] 空闲
```

围绕每条 mmad 的是一个完整的"借还"循环：写 L0 前等 M 队列归还（`:141`），写完宣布就绪（`:146`），mmad 开算前等就绪（`:148`），算完归还（`:152`）。`:142-145` 的条件搬运（L0 复用）可能跳过拷贝，但 wait/notify 对**永远执行**——闸门模型不因复用而改变（第 23 天 3.4 节末尾的提醒）。

**为什么这组事件不守护 L0C？** 分两层看：

- **同一子块的 K 遍历内**：所有写 `l0c[s_cd]` 的 mmad 都排入**同一条 M 队列**，队列内按序执行——对同一份累加器的连续读-改-写天然串行，无需事件。
- **跨输出块复用时**：下一个输出块的同位置子块会用 `clear_C = true` 覆盖这份 `l0c[s_cd]`（3.4 节）。仓库内**可见**的保护是 epilogue 的 stage 协议——`:129-131` 在首个 `k_blk` 调 `epilogue.aic_wait_free(s_epi)`（FIX 队列上等 AIV 归还 UB stage，`epilogue.hpp:37-39`）。但 M 队列（mmad 写 L0C）与 FIX 队列（`copy_l0c_to_ub` 读 L0C）之间**没有显式的事件对**；"上一块的 FIX 读取是否一定先于下一块的首条 mmad"这一跨队列次序，源码中无可见同步，属于 `asc_mmad` / `asc_copy_l0c2ub` 的指令级语义（CANN 文档 territory）。这是静态分析的边界——把它记入遗留问题，第 25 天画出 AIC/AIV 完整时序后再回头看。

### 3.4 L0C 累加器的三段生命周期：clear_C → KEEP → UPDATE

L0C 的声明只有一行（`bf16_gemm.hpp:64`）：

```cpp
constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
```

单个 stage 是 `MAD_M × MAD_N` 的 FP32 累加器（4 字节/元素），份数由 `:51` 决定：

```cpp
constexpr uint32_t num_mad_cd_stages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);
```

容量由 `:69` 钉死：`l0c.offset(num_mad_cd_stages) <= L0CSizeBytes`（256KB，`common.hpp:57`）。启发式侧还有一道半容量检查——`heuristics.hpp:127`：`if (mad_m * mad_n * sizeof(float) > L0CSizeBytes / 2) continue;`，其上方 `:117-118` 的注释写明动机："**at least 2 stage in L0C, to overlap mad and store**"（L0C 至少留 2 个 stage 的空间，让一个子块的 store 与另一个子块的 mad 重叠）、"fit l0a and l0b, to overlap load and mad"。

现在把第 22 天埋下的两个标志接到 L0C 的生命周期上。对一个输出子块，它的 K 遍历共 `T` 条 mmad：

```text
mmad 序号:     #1              #2          ...      #T-1            #T
k_blk_idx:     0               0           ...      last            last
is_first:      ✔ (clear_C)     ✘           ...      ✘               ✘
is_last:       ✘               ✘           ...      ✘               ✔ (UPDATE)
unit_flag:     KEEP            KEEP        ...      KEEP            UPDATE
clear_C:       true            false       ...      false           false
L0C[s_cd] 语义: C = 0 + A·B     C += A·B    ...      C += A·B        C += A·B 并提交
                                                                        │
                                     最后一个 k_blk 的 K 遍历结束后       ▼
                                     :155-163 消费（aic_store / copy_l0c_to_gm）
                                                                        │
下一个输出块的同位置子块:  #1' clear_C=true，从 0 开始覆盖旧值 ──────────┘
```

**为什么首条必须 `clear_C`？** `s_cd` 到子块的映射对所有输出块相同（3.5 节），所以这份 `l0c[s_cd]` 物理上还存着**上一个输出块同位置子块的最终结果**（内核启动后的第一个块则是未初始化残留）。`clear_C = true` 让累加起点为 0，语义是 `C = 0 + A·B` 而不是 `C = C_旧 + A·B`。反例推演（第 22 天自测 6 已数值演算过一种）：若 `is_first` 漏判，本块结果会叠加上一个块的完整结果——数值直接翻倍级别的错误。

**KEEP / UPDATE 的仓库级契约。** `asc_unit_flag_mode` 的取值（`DISABLE / ENABLE_KEEP / ENABLE_UPDATE`）定义在 CANN 的 `c_api` 头里；仓库内所有 GEMM 主线遵守同一约定——**中间步 KEEP、K 维末步 UPDATE**。从用法可以读出的契约是：只有最后一次 mmad 的结果才作为最终值供 `:155-163` 消费；中间各步的部分和保持在累加链中。寄存器级确切行为属于 CANN 指令集文档，我们以仓库用法为准。

**读取侧也带 UPDATE。** 一个容易忽略的事实：`copy_l0c_to_ub` 和 `copy_l0c_to_gm` 的 `unit_flag` 参数**默认值就是 `ENABLE_UPDATE`**（`copy.hpp:275, 296`）：

```cpp
template <typename T, typename ST>
__aicore__ inline void copy_l0c_to_ub(
    ub_ptr<T> dst, l0c_ptr<ST> src, ..., asc_unit_flag_mode unit_flag = asc_unit_flag_mode::ENABLE_UPDATE) {
```

把仓库内 `ENABLE_KEEP` / `ENABLE_UPDATE` 的全部出现点放在一起（3.7 节的表），会看到 UPDATE 恰好出现在**累加链的两端**——最后一条 mmad（产出侧收尾）和 L0C 的读出（消费侧收尾），KEEP 只出现在中间步。这个"两端 UPDATE、中间 KEEP"的形状就是仓库对这对标志的全部用法。

**单发退化形态。** 没有 K 循环的场景（一次 mmad 出结果），first 与 last 合并为同一条指令——`UPDATE` 与 `clear_C = true` 同时出现：

- `mqa_logits.hpp:396-397`：`mad(l0c_tile, l0a[qh_tile_idx], l0b_tile, num_m_rows, issued_n, MAD_K, asc_unit_flag_mode::ENABLE_UPDATE, true);`
- `sparse_mqa_logits.hpp:223-224`、`paged_sparse_mqa_logits.hpp:340`：同样的 `(ENABLE_UPDATE, true)` 组合。

反过来看：**多步累加时绝不能把 UPDATE 提前、也绝不能漏掉 clear_C**——单发形态正是这两个约束在"只有一步"时的自然合并。

### 3.5 多个 MAD Tile 如何组成一个 Block 输出

一个 `BLOCK_M × BLOCK_N` 的输出块被切成 `(BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)` 个 MAD Tile，**每个子块独占一份 L0C 累加器**。以 2×2 为例：

```text
BLOCK 128×128，MAD 64×64 → 4 个 MAD Tile，num_mad_cd_stages = 4

            n_mad_idx=0        n_mad_idx=64
        ┌────────────────┬────────────────┐
 m=0    │  (m0,n0)       │  (m0,n1)       │
        │  l0c[s_cd=0]   │  l0c[s_cd=1]   │
        ├────────────────┼────────────────┤
 m=64   │  (m1,n0)       │  (m1,n1)       │
        │  l0c[s_cd=2]   │  l0c[s_cd=3]   │
        └────────────────┴────────────────┘
  s_cd 扫描顺序（每个 k_blk 内一趟）: 0 → 1 → 2 → 3 →（回绕）0 → …
```

循环结构（`bf16_gemm.hpp:123-124`）有一个容易看错的细节：

```cpp
for (uint32_t m_mad_idx = 0, s_cd = 0; m_mad_idx < actual_m; m_mad_idx += MAD_M) {
    for (uint32_t n_mad_idx = 0; n_mad_idx < actual_n; n_mad_idx += MAD_N,
             s_cd = s_cd == num_mad_cd_stages - 1 ? 0 : s_cd + 1) {
```

`s_cd = 0` 在 **m_mad 循环的初始化子句**里，每个 `k_blk` 执行一趟时归零一次；`s_cd` 的推进在 **n_mad 循环的递增子句**里，跨 m 行**连续扫描**、到 `num_mad_cd_stages` 回绕。于是每个子块 `(i, j)` 固定映射到 `s_cd = i × (BLOCK_N/MAD_N) + j`——**对所有 k_blk、所有输出块都一致**。这个固定映射是 3.4 节"clear_C 覆盖的是上一个块同位置子块"的前提。

三个结构性事实把 MAD Tile 拼成 Block 输出：

**事实一：M/N 子块循环在 k_blk 循环之内，L0C 跨 k_blk 持续累加。** 每搬来一个 K 块，就把所有子块各推进一个 `BLOCK_K` 切片（第 22 天结构事实 1）。因此子块 `(i,j)` 的 `l0c[s_cd]` 保存的是它在**所有** `(k_blk, k_mad)` 步骤上的部分和之和——这正是 `is_first` 必须检查 `k_blk_idx == 0` 的原因。

**事实二：边界块少扫几个 s_cd，映射依然一致。** 若 `actual_m ≤ 64`（边界输出块），m_mad 只有一行，本趟只扫 `s_cd = 0, 1`；未被使用的 `s_cd = 2, 3` **无需清零**——它们将来作为某个子块 K 遍历的第一条 mmad 时自然会被 `clear_C` 覆盖。

**事实三：交付窗口在最后一个 k_blk。** `:155-163` 由 `k_blk_idx == num_k_blocks - 1` 守护，每个子块在其 K 遍历结束后立即交付一次：

```cpp
if (k_blk_idx == num_k_blocks - 1) {
    if constexpr (kDirectStore) {
        const auto d = gm_d.batch(batch_idx).group(...);
        copy_l0c_to_gm(d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(),
                       l0c_tile, eff_mad_m, eff_mad_n, d.stride_outer, kL2CtrlStoreCd);   // :158-159  L0C → GM
    } else {
        epilogue.aic_store(l0c_tile, s_epi, eff_mad_m, eff_mad_n);                        // :161       L0C → UB[s_epi]
    }
}
s_epi = s_epi == epilogue.kNumStages - 1 ? 0 : s_epi + 1;                                 // :164
```

非 `kDirectStore` 路径走 `epilogue.aic_store`（`epilogue.hpp:42-47`），两行：`copy_l0c_to_ub(ub[stage_idx], l0c_tile, ...)`（FIX 队列，L0C→UB，含 BF16/FP32 转换 `quant_mode`）+ `set_intra_block(PIPE_FIX, aiv_id, stage_idx)`（块内 arrive，通知 AIV "UB stage 已填好"）。`s_epi` 到 AIV 的路由是纯算术（`epilogue.hpp:43-44`）：

```cpp
const uint32_t aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs;   // 2 个 AIV 时：s_epi % 2
const uint32_t stage_idx = s_epi / kNumAIVs;                // 每个 AIV 自己的 UB stage 编号
```

`epilogue.hpp:23` 的注释点名了这套分配："**Assign complete MAD tiles round-robin across available AIVs**"。`kNumStages = kNumStagesPerAIV × kNumAIVs`（`:24`；BF16 内核把模板参数 `kNumEpilogueStages` 传作 `kNumStagesPerAIV`、AIV 数固定 2，`bf16_gemm.hpp:53-55`），并被 `:33` 的断言约束：`kNumStages >= (BLOCK_M/MAD_M) * (BLOCK_N/MAD_N)`——**UB stage 总数不得少于 CD stage 数**，否则轮转不开。启发式在 host 侧计算同一约束（`heuristics.hpp:186-194`：`num_epilogue_stages ≤ num_intra_block_events(16)`、`≤ UBSizeBytes / epilogue_bytes`，且 `num_epilogue_stages × num_aivs < num_cd_stages` 则整个候选被否决）。

AIV 侧的 `aiv_drain`（`epilogue.hpp:49-87`）以相同的 m/n 扫描顺序 `wait_intra_block(kPipe, aiv_id, stage_idx)` 等对应 tile，处理后 `copy_ub_to_gm` 写回 GM，再 `set_intra_block(PIPE_MTE3, aiv_id, stage_idx)` 归还 UB stage——AIC 的 `aic_wait_free` 等的就是这个归还。完整时序是第 25 天的主题，今天只需记住：**每个 MAD Tile 的最终结果经 `s_epi` 轮转进入某个 AIV 的某个 UB stage，由该 AIV 写回 GM**。

最后一块拼图是跨输出块的续接：`epi_base = s_epi`（`:169`）让下一个输出块从本轮结束的 `s_epi` 继续轮转——相邻块错开使用 UB stage，避免同一 stage 被两个块背靠背占用。

### 3.6 手工推演示范

**设定**：Normal GEMM，单输出块，M=128, N=128, K=256。假设 JIT 配置为：

```text
BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, MAD_M=64, MAD_N=64, MAD_K=64
kNumL1Stages=8, kNumL0Stages=2, kNumEpilogueStages=2（→ epilogue.kNumStages = 2×2 = 4）
```

> 说明：这是一组满足内核全部 `static_assert` 的**假设配置**（与第 22 天 A-2 变体 V3 同款）。启发式对 128×128 的 block 按 `(1,1) → (1,2) → (2,1)` 顺序搜索拆分（`heuristics.hpp:124`），`(1,1)` 即 MAD 128×128 会先被接受，因此 64×64 通常不会被真实选中；真实触发多 CD stage 的例子是宽 N 块——如 BLOCK_N=512 时 `(1,1)` 的 MAD 128×512 因超过 L0C 一半（`heuristics.hpp:127`）被拒，落到 `(1,2)` 得 MAD 128×256、2 个 CD stage。真实配置用 `DG_PRINT_CONFIGS=1` 核对（实践步骤 B）。

**先验算约束**（养成习惯，每次推演前过一遍）：

```text
:69  l0c: 4 stage × 64×64×4B = 64KB ≤ 256KB ✔        num_mad_cd_stages = 4
:67  l0a: 2 stage × 64×64×2B = 16KB ≤ 64KB ✔（l0b 对称）
:70  BLOCK_K 128 % MAD_K 64 == 0 ✔
     epilogue.hpp:33: kNumStages 4 ≥ num_mad_cd_stages 4 ✔
     epilogue.hpp:29: MAD_N 64 % 16 == 0 ✔（BF16 输出）
```

**总量**：`num_k_blocks = ceil(256/128) = 2`；每子块每 k_blk 有 `128/64 = 2` 条 mmad；每子块共 `2×2 = 4` 条；全块 `4 子块 × 4 = 16` 条 mmad、最后一次 k_blk 内 4 次交付。

**子块 (m0,n0)（s_cd=0）的完整 mmad 序列**——其余三个子块结构完全同构，`is_first` / `is_last` 各自独立判定：

| 步 | k_blk_idx | s (L1环) | k_mad_idx | eff_mad_k | s_ab (L0环) | is_first | is_last | unit_flag | clear_C | L0C[0] 语义 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 64 | 0 | **✔** | ✘ | KEEP | **true** | C = 0 + A[0:64, 0:64]·B[0:64, 0:64] |
| 2 | 0 | 0 | 64 | 64 | 1 | ✘ | ✘ | KEEP | false | C += A[0:64, 64:128]·B[0:64, 64:128] |
| 3 | 1 | 1 | 0 | 64 | 0 | ✘ | ✘ | KEEP | false | C += A[0:64, 128:192]·B[0:64, 128:192] |
| 4 | 1 | 1 | 64 | 64 | 1 | ✘ | **✔** | **UPDATE** | false | C += A[0:64, 192:256]·B[0:64, 192:256]，提交 |

注意步 3：`k_mad_idx == 0` 但 `k_blk_idx == 1`，`is_first` 为假——K 块 0 的 128 列部分和尚在 L0C[0] 中，绝不能清。`s_ab` 在每个子块的 K 遍历内各自从 0 起（第 22 天要点 3）。

**交付阶段**（k_blk 1 的 m/n 扫描中，每个子块 K 遍历结束即交付；非 `kDirectStore` 路径走 `aic_store`，`epi_base = 0`）：

| 交付序 | 子块 | s_cd | s_epi | aiv_id = s_epi%2 | stage_idx = s_epi/2 | 输出区域（GM） |
|---|---|---|---|---|---|---|
| 1 | (m0,n0) | 0 | 0 | 0 | 0 | D[0:64, 0:64] |
| 2 | (m0,n1) | 1 | 1 | 1 | 0 | D[0:64, 64:128] |
| 3 | (m1,n0) | 2 | 2 | 0 | 1 | D[64:128, 0:64] |
| 4 | (m1,n1) | 3 | 3 | 1 | 1 | D[64:128, 64:128] |

四份结果被 round-robin 平分给两个 AIV（AIV0 拿 s_epi 0/2，AIV1 拿 1/3），各自使用独立的 UB stage（stage_idx 0/1）。交付完毕后 `s_epi` 回绕到 0，`epi_base = s_epi = 0`（`:169`）——本例中 kNumStages 恰好等于 CD stage 数，环刚好转满一圈；若 kNumStages 更大（如 8），下一个输出块会从 s_epi=4 起步，错开 UB stage 的使用。

**16 条 mmad 的下发与执行**：全部按 `:141-152` 的模式交替排入 MTE1（搬运）与 M（计算）两条队列，执行受第 23 天的事件闸门调节——今天的表格只关心"每条 mmad 读写哪份缓冲、带什么标志"。

### 3.7 仓库级契约的一致性：全部用法一张表

把仓库内 `mad()` 的 `unit_flag` / `clear_C` 用法全部列出（`grep` 可复核）：

| 文件:行 | 形态 | unit_flag | clear_C |
|---|---|---|---|
| `bf16_gemm.hpp:150-151` | 多步累加 | `is_last ? UPDATE : KEEP` | `is_first` |
| `fp8_gemm.hpp:229-230` | 多步累加（K 对齐到 64 后） | 同上 | `is_first` |
| `fp8_dequant_gemm.hpp:321-322` | 多步累加（反量化路径） | 同上 | `is_first` |
| `mega_moe.hpp:599-600` | 多步累加（K 循环在最内层之外，A 跨 n_mad 复用） | `is_last_k ? UPDATE : KEEP` | `k_idx + k_mad_idx == 0` |
| `mqa_logits.hpp:397` | 单发（无 K 循环） | `UPDATE` | `true` |
| `sparse_mqa_logits.hpp:224` | 单发 | `UPDATE` | `true` |
| `paged_sparse_mqa_logits.hpp:340` | 单发 | `UPDATE` | `true` |
| `copy.hpp:275, 296`（默认值） | **读取** L0C（`copy_l0c_to_ub/gm`） | `UPDATE` | —— |

三条归纳：

1. **多步累加必是"两端 UPDATE、中间 KEEP"**：`is_last ? UPDATE : KEEP` 在四个 GEMM 实现中逐字相同；
2. **单发必是 `(UPDATE, true)`**：first 与 last 合并；
3. **clear_C 的判定表达式两种写法等价**：`k_blk_idx == 0 && k_mad_idx == 0`（bf16 系）与 `k_idx + k_mad_idx == 0`（`mega_moe.hpp:600`，把两层循环计数合成一个全局 K 偏移）。

`mega_moe.hpp:573-605` 值得单独看一眼：它的循环次序是 `k_mad` 在外、`n_mad` 在内，`l0a_mad` 被所有 `n_mad_idx` 复用（`:573` 注释 "Reuse L0A across the BLOCK_N / MAD_N multiplies"）——**循环怎么排、操作数怎么复用可以不同，但 L0C 累加链的"首步清零、末步提交"契约一个字不变**。这就是"模式识别"：读懂 bf16 这一份，其余实现可以按契约快速核对。

### 3.8 参考伪代码（当天产出的参考答案）

下面是一个输出块内 MAD 执行过程的完整伪代码（融合第 22～24 天；`↷ 队列` 表示指令排入哪条队列）。先自己写，再对照：

```text
# 常量：BLOCK_M/N/K, MAD_M/N/K, kNumL1Stages, kNumL0Stages,
#       num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N), kNumEpiStages = epilogue.kNumStages
# 输入：输出块位置 (m_idx, n_idx)，真实规模 actual_m/actual_n，有效 K eff_shape_k
s_epi = epi_base                                            # 本块 UB stage 轮转起点（:107）

for k_blk_idx in [0, num_k_blocks), s = k_blk_idx % kNumL1Stages:      # ── 第 22 天 ──
    k_idx    = k_idx_base + k_blk_idx * BLOCK_K
    actual_k = min(eff_shape_k - k_blk_idx*BLOCK_K, BLOCK_K)
    等 MTE1 归还 L1[s]                                       # :116  ↷ MTE2
    GM → L1[s]（A 片段、B 片段各一次）                        # :117-118 ↷ MTE2
    通知 MTE1：L1[s] 就绪                                    # :119  ↷ MTE2
    等 L1[s] 就绪                                            # :121  ↷ MTE1

    for m_mad_idx in [0, actual_m) step MAD_M:                # ── M 子块 ──
        for n_mad_idx in [0, actual_n) step MAD_N:            # ── N 子块，s_cd 连续扫描 ──
            eff_mad_m = min(actual_m - m_mad_idx, MAD_M)       # :125
            eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N),
                                frac_k(epilogue.ub_dtype))    # :126  保证 L0C→UB 行 32B 对齐
            l0c_tile = l0c[s_cd] 的 (eff_mad_m, eff_mad_n) 对齐到 16×16   # :127

            if k_blk_idx == 0 且 非 kDirectStore:
                等 AIV 归还 UB stage s_epi                    # :129-131  ↷ FIX（块内同步）

            for k_mad_idx in [0, actual_k) step MAD_K,
                s_ab = (k_mad_idx/MAD_K) % kNumL0Stages:      # ── 第 22 天 ──
                is_first  = (k_blk_idx == 0 && k_mad_idx == 0)          # :134
                is_last   = (k_blk_idx == num_k_blocks-1
                             && k_mad_idx + MAD_K >= actual_k)          # :135
                eff_mad_k = min(actual_k - k_mad_idx, MAD_K)            # :137
                l0a_tile = l0a[s_ab] 的 (eff_mad_m, eff_mad_k) 对齐     # :138
                l0b_tile = l0b[s_ab] 的 (eff_mad_n, eff_mad_k) 对齐     # :139

                等 M 归还 L0[s_ab]                             # :141  ↷ MTE1
                （复用条件不满足时）L1[s] → L0A[s_ab] / L0B[s_ab]  # :142-145 ↷ MTE1
                通知 M：L0[s_ab] 就绪                          # :146  ↷ MTE1
                等 L0[s_ab] 就绪                               # :148  ↷ M

                ─── mad(l0c_tile, l0a_tile, l0b_tile,          # :149-151 ↷ M
                │        eff_mad_m, eff_mad_n, eff_mad_k,      #   asc_mmad(m, k, n)
                │        is_last ? UPDATE : KEEP,              #   累加链：末步提交，中间保持
                │        is_first)                             #   首步清零：起点 0，覆盖旧块残留
                通知 MTE1：L0[s_ab] 空闲                        # :152  ↷ M

            if k_blk_idx == num_k_blocks - 1:                  # ── 交付窗口 ──
                kDirectStore ? L0C_tile → GM 直写               # :156-159 ↷ FIX
                             : L0C_tile → UB[s_epi]（含精度转换），
                               通知 AIV aiv_id = s_epi%2：      # :161     ↷ FIX（块内 arrive）
                               "stage s_epi/2 已填好"
                s_epi = (s_epi + 1) % kNumEpiStages            # :164

    通知 MTE2：L1[s] 空闲                                     # :167  ↷ MTE1
epi_base = s_epi                                              # :169  跨输出块续接轮转
```

写伪代码时最容易漏掉的五处（自查清单）：`eff_mad_n` 的对齐（`:126`）、`aic_wait_free` 只在 `k_blk_idx == 0` 时执行（`:129-131`）、交付只在最后一个 k_blk（`:155`）、`s_epi` 每次交付后推进（`:164`）、`epi_base` 的跨块续接（`:169`）。

## 四、源码导读

建议按站点顺序阅读，总计约 45 分钟。所有行号以仓库当前版本为准。

### 站点 1：MAD 主体与交付（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:123-169`，精读）

以"每条 mmad 读写哪份缓冲、带什么标志"为线索重读这段代码：`:125-127` 三个 eff 形状与 `l0c[s_cd]` tile 的选择；`:129-131` 交付前的 UB 等待；`:133-139` 内层控制量；`:141-152` 借还循环与 `mad()`；`:155-163` 交付的两条路径；`:164/:169` 两个续接点。对照 3.8 的伪代码逐行核销。

### 站点 2：`mad()` 封装与 L0C 读取（`deep_gemm/include/deep_gemm/ascend/copy.hpp:370-381` 精读，`:266-306` 浏览）

`mad()` 的参数调换与两个分支；`copy_l0c_to_ub` / `copy_l0c_to_gm` 的默认 `unit_flag = ENABLE_UPDATE`、默认 `quant_mode = infer_quant_mode<T, ST>()`（BF16 输出即 `F322BF16` 转换，`copy.hpp:231-264`）——今天只确认默认值，转换与 `dual_dst` 留给第 25 天。

### 站点 3：tile 类型与对齐（`deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp:58-97` + `deep_gemm/include/deep_gemm/common.hpp:54-79`，精读）

`result_ptr::as_mad_aligned`（`:88-90`，M/N 都对齐 `FRAC_MN=16`）与 `operand_ptr::as_mad_aligned`（`:52-55`，K 对齐 `get_frac_k<T>()`）的差异；`l0a_ptr/l0b_ptr/l0c_ptr/ub_ptr` 四个别名（`:93-97`）对应的地址空间；`get_frac_k` 对 BF16/FP32 的取值（`common.hpp:73-79`）。

### 站点 4：声明与断言（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:51, 59-70`，复习）

第 16 天读过的声明，今天换成 L0C 视角：`:64` `l0c(MAD_M, MAD_N)` 的形状为什么是 MAD 而不是 BLOCK；`:51` `num_mad_cd_stages`；`:69` 容量断言如何与 `heuristics.hpp:127` 的半容量检查呼应。

### 站点 5：MAD 拆分与 stage 约束（`csrc/jit_kernels/heuristics.hpp:111-148, 186-194`，精读）

`:111-112` `num_l0_stages` 恒为 2；`:116-127` MAD 搜索的顺序 `(1,1)/(1,2)/(2,1)`、两条注释（L0C 至少 2 stage 重叠 mad 与 store；L0A/L0B 装得下重叠 load 与 mad）与半容量拒绝；`:129-142` `MAD_K` 推导（第 22 天已读，今天看它如何反过来限制 `mad_m/mad_n` 的候选）；`:186-194` `num_cd_stages`、`num_epilogue_stages` 的计算与 `num_epilogue_stages × num_aivs ≥ num_cd_stages` 的否决条件。

### 站点 6：Epilogue 侧的接入（`deep_gemm/include/deep_gemm/epilogue.hpp:23-47`，精读）

`kNumStages = kNumStagesPerAIV × kNumAIVs`（`:24`）与 round-robin 注释（`:23`）；`:28-35` 的五条断言（尤其 `:29` 行对齐与 `:33` stage 数下限）；`aic_wait_free`（`:37-39`）与 `aic_store`（`:42-47`）两个方法各只有两行——今天读它们的"接口契约"，时序留给第 25 天。

### 站点 7：同构用法对照（浏览，建立模式识别）

- `deep_gemm/include/deep_gemm/mqa_logits.hpp:392-404`：单发 `mad(..., ENABLE_UPDATE, true)` 之后紧跟 `copy_l0c_to_ub`（也带 `ENABLE_UPDATE`）——产出与读取背靠背；
- `deep_gemm/include/deep_gemm/sparse_mqa_logits.hpp:219-231`：单发 + `l0c_stage_idx` 环轮转（`kNumL0CStages`）；
- `deep_gemm/include/deep_gemm/mega_moe.hpp:573-605`：`is_last_k ? UPDATE : KEEP` 与 `k_idx + k_mad_idx == 0`，A 跨 n_mad 复用的另一种循环排布；
- `deep_gemm/include/deep_gemm/fp8_gemm.hpp:205-231` / `fp8_dequant_gemm.hpp:298-323`：低精度路径的同一契约（K 先对齐到 64）。

### 站点 8：配置如何进入内核（`csrc/jit_kernels/bf16_gemm.hpp:43-66`，浏览）

`std::format` 实参表里 `config.mad_m/mad_n/mad_k/num_epilogue_stages` 的位置——host 选出的 MAD 拆分就是今天推演的模板参数来源；`tests/test_bf16_gemm.py:25-44` 是实践步骤 B 的运行入口。

## 五、实践步骤

### A. 无 Ascend 环境：纸面推演（主任务，约 55 分钟）

**A-1 写伪代码（必做，当天产出的核心）**。合上源码和本文，独立写出 3.8 节那样的伪代码，必须覆盖：L1/L0 两级借还同步、三个 eff 形状（含 `eff_mad_n` 的对齐）、`is_first` / `is_last` 的判定与消费、`s_cd` / `s_ab` / `s_epi` 三个环的推进、交付窗口的两条路径。写完对照 3.8 与源码订正，按"自查清单"五处逐项打勾。

**A-2 完整推演 2×2 场景（必做）**。用 3.6 的设定（M=128, N=128, K=256；BLOCK 128×128×128，MAD 64×64×64，kNumEpilogueStages=2），**先不看参考答案**：

1. 验算四条 `static_assert` 与 `epilogue.hpp:33` 的约束；
2. 列出子块 (m1,n1)（`s_cd=3`）的全部 4 条 mmad（`k_blk_idx / k_mad_idx / eff_mad_k / s_ab / is_first / is_last / unit_flag / clear_C`）；
3. 写出 4 次交付的 `(子块, s_cd, s_epi, aiv_id, stage_idx, GM 输出区域)` 表；
4. 数出全块 mmad 总数、`copy_l1_to_l0a` / `copy_l1_to_l0b` 的调用次数（提示：`BLOCK_K=128 ≤ MAD_K×kNumL0Stages=128` 且 `BLOCK_N != MAD_N`，代入 `:142/:144` 的复用条件数一数哪些步会跳过拷贝）；
5. 对照 3.6 订正。

**A-3 变体推演（必做）**。每次只改一个变量，重新推演并记录差异：

| 变体 | 改动 | 观察重点 |
|---|---|---|
| V1 | K=130（其余同 A-2） | k_blk 1 的 `actual_k=2`：每子块只有 1 条 `eff_mad_k=2` 的 mmad，它同时是 `is_last`；L0 tile 的 shape_k 对齐到 16 而 mmad 取真值 2；全块 mmad 数 12 |
| V2 | MAD_M=MAD_N=128（即 3.6 的块用 `(1,1)` 拆分） | `num_mad_cd_stages=1`、`s_cd` 恒 0、全块只有一条累加链（4 条 mmad）；每块 1 次交付，`epi_base` 按 0→1→2→3 跨块轮转——对比 3.6 的"块内轮转" |
| V3 | `kNumEpilogueStages=1`（其余同 A-2） | `kNumStages=2 < 4` → `epilogue.hpp:33` 断言失败，JIT 编译期报错——体会 UB stage 数与 CD stage 数的绑定关系 |
| V4 | 把 `is_first` 恒置为 `false`（其余同 A-2） | 每个子块的 `l0c[s_cd]` 从上一个输出块同位置子块的最终值开始累加：设上块该子块结果为 X、本块正确结果为 Y，写出实际得到的结果 |

**A-4 判定题自验（必做）**。判断真假并给出行号依据（答案见第七节末尾）：

1. 同一输出块内，子块 (m0,n1) 的第一条 mmad（`k_blk_idx=0, k_mad_idx=0`）的 `clear_C` 为 `true`。（ ）
2. `mad()` 传给 `asc_mmad` 的形状参数顺序是 `(m, n, k)`。（ ）
3. 启发式会拒绝 `mad_m × mad_n × 4B > 128KB` 的拆分，动机是给 L0C 至少留 2 个 stage 的空间。（ ）
4. `copy_l0c_to_ub` 的 `unit_flag` 默认值是 `ENABLE_KEEP`。（ ）
5. M/N 拆成 2×2 子块时，一个输出块内 `aic_store` 共被调用 4 次，且全部发生在最后一个 `k_blk`。（ ）

**A-5 L0C 容量账本（选做）**。对四组 MAD 拆分各算"单个 CD stage 字节数 / 256KB 可容纳份数 / 会被 `heuristics.hpp:127` 拒绝吗"：128×128、64×64、128×256、256×128。再回答：128×256 和 256×128 的账本相同，但启发式搜索顺序 `(1,1)/(1,2)/(2,1)` 意味着哪种更先被尝试？

### B. 有 Ascend 环境：核对真实配置（可选，约 20 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py unaligned-k --skip-prof
```

1. 从输出里找 `GemmConfig(...)` 行（由 `heuristics.hpp:749-753` 打印），记录若干 shape 的 `mad=...x...x...` 与 `num_epilogue_stages`；
2. 对每个配置计算 `num_mad_cd_stages = (block_m/mad_m) × (block_n/mad_n)`，验证 `num_mad_cd_stages × mad_m × mad_n × 4B ≤ 256KB`（`:69`）与 `num_epilogue_stages × 2 ≥ num_mad_cd_stages`（`epilogue.hpp:33`）；
3. 找一个 N 较大的 shape（如 n=4096 一类），看是否出现 `block_n=512 → mad=128x256x...` 的 2-CD-stage 配置，与 3.6 的说明互相印证；
4. 跑 `unaligned-k` 感受尾部 K（如 K=234 的 `eff_mad_k=42`）——正确性由测试体系守护（第 30 天展开），今天只把测试场景与 `:137` 的 `min` 对应起来。

## 六、当天产出

**一段 MAD 执行过程的伪代码**。把 A-1～A-3 的结果整理成一页笔记，至少包含：

1. **伪代码主体**：仿照 3.8，覆盖同步、标志、tile 选择、交付与三个 stage 环的推进，每行标注源码行号；
2. **L0C 生命周期图**：一个子块从 `clear_C`（首条）经 `KEEP`（中间）到 `UPDATE`（末条）再到被 `aic_store` / `copy_l0c_to_gm` 消费、最后被下一块 `clear_C` 覆盖的时间线（仿 3.4 的图）；
3. **2×2 推演表**：A-2 的 mmad 表与交付表（至少 (m1,n1) 一个子块完整 + 全部 4 次交付）；
4. **变体记录**：V1～V4 各两三行结论；
5. **遗留问题**：至少记录两条交给后续天（例如"M 队列写 L0C 与 FIX 队列读 L0C 之间的跨队列次序如何保证"交给第 25 天结合 epilogue 时序分析；"`copy_l0c_to_ub` 的 `quant_mode` / `dual_dst` 参数"交给第 25 天）。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. 写出 `mad()` 的 8 个参数及含义；`(shape_m, shape_n, shape_k)` 如何映射到 `asc_mmad` 的参数顺序？
2. `mad()` 前后四个同步点（`:141/:146/:148/:152`）各自的方向、事件号和守护对象是什么？为什么这组事件不守护 L0C——同一子块的 K 遍历内不需要、跨输出块时靠什么？
3. 首条 mmad 为什么必须 `clear_C = true`？`l0c[s_cd]` 里的"旧值"具体是什么？若某边界块只用到 `s_cd = 0,1`，未被使用的 `s_cd = 2,3` 需要事先清零吗？为什么？
4. `ENABLE_KEEP` / `ENABLE_UPDATE` 在仓库中的分工契约是什么？`copy_l0c_to_ub` / `copy_l0c_to_gm` 的 `unit_flag` 默认值是什么？这个默认值与"两端 UPDATE、中间 KEEP"的观察如何互相印证？
5. 给定 BLOCK 128×128、MAD 64×64、K=256、BLOCK_K=128：`num_mad_cd_stages` 是多少？每个子块执行几条 mmad？全块共几条、几次 `aic_store`？`epilogue.hpp:33` 对 `kNumEpilogueStages` 提出了什么要求？
6. `mqa_logits.hpp:397` 为什么 `ENABLE_UPDATE` 与 `clear_C = true` 同时出现？这对应两层 K 循环的什么退化情形？
7. `s_epi` 如何决定一个 MAD Tile 交给哪个 AIV 的哪个 UB stage（写出两个算式）？`epi_base`（`:169`）的作用是什么？

### 参考答案

1. `l0c`（目标累加器 tile，`l0c[s_cd]`）、`l0a` / `l0b`（操作数 tile，`l0a/l0b[s_ab]`）、`shape_m/n/k`（`eff_mad_m/eff_mad_n/eff_mad_k`，真实规模，N 已按 UB dtype 对齐）、`unit_flag`（`is_last ? UPDATE : KEEP`）、`clear_C`（`is_first`，默认 false）。映射为 `asc_mmad(l0c, l0a, l0b, shape_m, shape_k, shape_n, ...)`——K 与 N 互换（`copy.hpp:378`）。
2. `:141` wait(M→MTE1, s_ab)：MTE1 等 M 归还 L0[s_ab]（写前）；`:146` notify(MTE1→M, s_ab)：L0[s_ab] 就绪；`:148` wait(MTE1→M, s_ab)：M 等就绪（算前）；`:152` notify(M→MTE1, s_ab)：M 归还 L0[s_ab]。不守 L0C 的原因：同一子块 K 遍历内所有 mmad 在同一条 M 队列内按序执行，对同一累加器的读-改-写天然串行；跨输出块时靠 `clear_C`（语义上不依赖旧值）加 epilogue 的 stage 协议（`aic_wait_free` 等 AIV 归还 UB），M 与 FIX 队列之间关于 L0C 的显式次序在仓库内不可见（3.3 节的边界讨论）。
3. 因为 `s_cd` 到子块的映射对所有输出块相同（`:123-124` 的扫描方式），这份 `l0c[s_cd]` 存着上一个输出块同位置子块的最终结果（首个块则是未初始化残留）；不清零则本块结果叠加旧块结果。未使用的 `s_cd` 无需事先清零：它们将来必然作为某个子块 K 遍历的第一条 mmad 被 `clear_C` 覆盖。
4. 契约：中间步 `KEEP` 保持部分和在累加链中，K 维末步 `UPDATE` 提交最终值供 `:155-163` 消费（四个 GEMM 实现逐字相同的三元表达式）。读取侧默认 `ENABLE_UPDATE`（`copy.hpp:275, 296`）——UPDATE 出现在累加链的两端（最后一条 mmad 与读出），KEEP 只在中间，两处用法互相印证"UPDATE = 链的收尾"。
5. `num_mad_cd_stages = 2×2 = 4`；每子块 `2 k_blk × 2 步 = 4` 条；全块 16 条、4 次 `aic_store`（每个子块在最后一个 k_blk 各一次）。要求 `kNumStages = kNumEpilogueStages × 2 ≥ 4`，即 `kNumEpilogueStages ≥ 2`。
6. MQA 的单条 mmad 就产出最终 logits（`kHeadDim` 一次算完，没有 K 分块循环），first 与 last 合并为同一条：既要清零（clear_C）又要提交（UPDATE）。对应两层循环退化为 `num_k_blocks = 1` 且 `actual_k ≤ MAD_K`（只迭代一次）的情形。
7. `aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs`（2 AIV 时即 `s_epi % 2`），`stage_idx = s_epi / kNumAIVs`（`epilogue.hpp:43-44`）。`epi_base = s_epi`（`:169`）让下一个输出块从本轮结束位置继续轮转 UB stage，相邻块错开使用，避免同一 stage 被背靠背占用。

**A-4 判定题答案**：

1. **真**。`is_first` 在 m/n 子块循环之内计算（`:134`），每个子块的 K 遍历各自首条清零。
2. **假**。`asc_mmad` 收到的是 `(shape_m, shape_k, shape_n)`（`copy.hpp:378`）。
3. **真**。`heuristics.hpp:127` 的检查与 `:117` 注释 "at least 2 stage in L0C, to overlap mad and store"。
4. **假**。默认 `ENABLE_UPDATE`（`copy.hpp:275, 296`）。
5. **真**。交付由 `k_blk_idx == num_k_blocks - 1` 守护（`:155`），m/n 扫描中每子块一次（`:161`）。

---

**明日预告（第 25 天）**：今天我们把结果交到了 `epilogue.aic_store` 手里就停住了。明天读完另一半——`deep_gemm/include/deep_gemm/epilogue.hpp` 的 `aiv_drain` 与 `epilogue/operators.hpp` 的算子族：AIC 与 AIV 如何分工（`__mix__(1, 2)` 的含义）、accumulation / bias / alpha / 输出转换各发生在哪一层、AIC store 与 AIV drain 如何用块内同步配合，以及今天留下的"M 写 L0C 与 FIX 读 L0C 的跨队列次序"问题。
