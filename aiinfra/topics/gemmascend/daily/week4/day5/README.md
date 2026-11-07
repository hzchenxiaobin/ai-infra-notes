# 第 4 周 · 第 5 天（总第 26 天）：学习特殊执行路径

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 4 周"BF16 Kernel——MAD、流水线与 Epilogue"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 55 分钟，自测题约 15 分钟。

本周前四天我们把 `bf16_gemm_impl` 的**主线路径**走通了：K 维两层循环（第 22 天）、MTE1/MTE2/M 流水线同步（第 23 天）、MAD 与 L0C 累加（第 24 天）、AIC/AIV 分工与 Epilogue（第 25 天）。但主线路径并不是唯一形态——内核末尾还挂着三个默认为 `false` 的模板参数（`bf16_gemm.hpp:25`）：`kDirectStore`（AIC 从 L0C 直写 GM，完全不要 AIV 和 UB）、`kResidentA` / `kResidentB`（让 A 或 B 常驻片上，砍掉重复搬运）。它们都不是独立的第二套代码，而是**同一份主线代码的条件裁剪**：`if constexpr` 与运行时 `if` 在关键搬运点剪掉几行，数据流图就短了一截。今天逐一打开这三条"近道"，再看两类边界场景——非对齐 shape 与空组——分别被谁兜住。要回答四个问题：**直写路径删掉了什么、凭什么能删？**（约束链：设备断言 + host 闸门）**`kResidentA/B` 各自复用什么、省掉哪几条搬运指令？**（槽位稳定性）**非对齐 shape 为什么普通路径能容忍、直写路径不能？**（`eff_mad_n` 向上取整 vs 真值写回）**空组的输出由谁负责写零？**（AIC 跳过 + AIV 兜底）。当天产出是一张**普通路径与特殊路径对比表**（参考答案见 3.8 节），并接住第 25 天留下的"归还 arrive 能否从 MTE3 提前到 V"的问题——`kDirectStore` 正是它的极限答案。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出三个布尔模板参数的声明位置（`bf16_gemm.hpp:25`）与 host 侧对应字段（`csrc/jit_kernels/gemm_config.hpp:110-112`）、JIT 实例化时的传参位置（`csrc/jit_kernels/bf16_gemm.hpp:66`），并解释"特殊路径 = 主线代码的条件裁剪"这一设计——三条路径没有一个独立的函数体；
2. 对比 `kDirectStore` 与普通 Epilogue 路径：启动形态（`__mix__(1, kDirectStore ? 0 : 2)`，`:27`）、数据通路（`L0C → FIX → GM` vs `L0C → FIX → UB → V → MTE3 → GM`）、被裁剪的代码点（`:82`、`:129`、`:156-159`、`:174`）、以及 UB 与块内事件整体消失的事实；
3. 逐条解读 `can_direct_store()`（`csrc/jit_kernels/heuristics.hpp:521-533`）的六个条件，指出哪些有设备侧断言镜像（`:38`、`:56-57`）、哪些纯靠 host 把关，并用 `eff_mad_n = aligned(min(...), frac)`（`:126`）与 AIV 真值写回（`epilogue.hpp:65-66`）的差异解释**为什么直写要求 `d.n` 对齐而普通路径不要求**；
4. 解释 `kResidentA` 的四处代码改动（`:59-60`、`:109-110`、`:117`、`:143`）、L1 布局的变化（单 stage 宽 `aligned(SHAPE_K, 16)`）、`reuse_a = m_idx == last_m_idx` 的判定（`:92`），以及 host 条件 `d.m <= block_m` 等（`heuristics.hpp:552-555`）为什么恰好让复用恒成立；
5. 解释 `kResidentB` 的两个裁剪点（`:118`、`:144`）与设备断言 `SHAPE_K <= kNumL0Stages * MAD_K && BLOCK_N == MAD_N`（`:45`），并完成**槽位稳定性论证**：`s_ab` 在每个子块从 0 重置（`:133`）导致 B 的第 j 个 K 块恒落 `l0b[j % 环长]`，跨输出块复用还要求 `num_k_blocks == 1`——这条由 host 的 `d.k <= mad_k * num_l0_stages`（`heuristics.hpp:556-557`）配合 `make_gemm_candidate` 的 `mad_k ≤ block_k/num_l0_stages`（`:135`）间接保证；
6. 说明非对齐 shape 的三层边界机制（`actual_m/n/k` → `eff_mad_m/n/k` → `as_mad_aligned` tile）、`m/n == 0` 与 `k == 0` 的 host 早退（`csrc/utils/layout.hpp:54-86`），以及 m-grouped 组尾 padding 的现状（`scheduler.hpp:130-132` 的 TODO，细节留第 36 天）；
7. 说明空组在 AIC 侧如何被调度器跳过（`scheduler.hpp:167, 174-175`）、在 AIV 侧由 `aiv_zero_empty_groups`（`epilogue.hpp:108-141`）如何按 MAD 粒度分摊写零、与 `aiv_initialize` 的顺序关系，以及 accumulation 时空组"D 保持初值"的语义；
8. 有 Ascend 环境时，用 `DG_PRINT_CONFIGS=1` 观察真实 `GemmConfig` 中 `direct_store={}` 与 `resident={}/{}/{}` 的取值，并与 shape、major、acc 对应；无环境时完成纸面判定与推演。

**当天产出：普通路径与特殊路径对比表**（模板见第六节，参考答案见 3.8 节）。

## 二、预备知识

- **第 5 天**：GM/L1/L0A/L0B/L0C/UB 的层次职责；AIC 与 AIV 分工。
- **第 12 天**：`GemmConfig` 的字段（`block_m/n/k`、`mad_m/n/k`、`num_l1_stages`、`num_l0_stages`）；`select_gemm_config()` 的搜索框架。
- **第 15 天**：模板参数分类——今天聚焦末尾三个带默认值的布尔参数（`:25`）与编译期 shape（`SHAPE_M/N/K` 为 0 表示运行时传入，`:47-49`）。
- **第 16 天**：`l1a/l1b/l0a/l0b/l0c` 的 `constexpr` 声明与 offset 分配（`:59-69`）；L1 512KB、L0A/L0B 各 64KB、L0C 256KB、UB 256KB（`common.hpp:54-58`）。
- **第 17 天**：`Scheduler::get_next_block` 的确定性遍历与 swizzle；`actual_m/actual_n` 的含义。
- **第 19 天**：`copy_gm_to_l1` / `copy_l1_to_l0a/b` 的签名；`actual_m/n/k` 对边界搬运规模的约束。
- **第 22 天**：K 维两层循环；`kResidentA` 下 L1→L0A 读偏移的预告（day1 教程 3.3 要点 3）；空组不产生 K 循环迭代。
- **第 23 天**：`asc_sync_wait/notify` 的方向约定；`set_flags` 预置事件；L1/L0 stage 环。
- **第 24 天**：`mad()` 与 `ENABLE_UPDATE/KEEP`；L0C stage 的消费语义。
- **第 25 天**：`aic_store` / `aiv_drain` 的四步借还协议；`__mix__(1, 2)` 的三方执行；`aiv_zero_empty_groups` 与 `aiv_initialize` 的顺序（当时一笔带过，今天展开）；当天 A-5 留下的"归还 arrive 前移"问题。
- **C++ 语法**：`if constexpr`（编译期裁剪）、非 constexpr 的运行时 `if`（`kDirectStore` 等 constexpr 值会被编译器折叠，效果等同）。

另注：`__mix__`、`asc_set_atomic_add_float`、`asc_copy_l0c2gm` 等设备级原语声明在 CANN 工具链的 `c_api/asc_simd.h`（仓库之外）。本文只陈述从仓库内用法可以反推出的契约，推断与引用的边界会随文标注；`kMkAlignment = 256` 来自 `csrc/jit/jit.hpp:59`。

## 三、详细讲解

### 3.1 全景：三个布尔参数与它们的 host 开关

先看模板声明的最后一行（`bf16_gemm.hpp:25`）：

```cpp
    uint32_t kAlignment,
    bool kDirectStore = false, bool kResidentA = false, bool kResidentB = false
>
```

三个参数都有默认值 `false`——这就是"主线路径"的含义：不传它们时，前四天读到的代码就是全部。host 侧的开关在 `GemmConfig`（`gemm_config.hpp:110-112`）：

```cpp
    bool direct_store = false;
    bool resident_a = false, resident_b = false, resident_sf = false;
    bool dual_aiv_dequant = false;
```

JIT 实例化时把这三个值接成模板实参（`csrc/jit_kernels/bf16_gemm.hpp:66` 的 `config.direct_store, config.resident_a, config.resident_b`——注意 `resident_sf` 只属于 FP8 路径，BF16 不传）。值的来源是 `select_gemm_config()` 末段的 `refine_gemm_pipeline()`（`heuristics.hpp:535-574`）：主搜索先定出 BLOCK/MAD/stages，然后**试穿**两件"外套"——直写与常驻——用代价模型比较后再决定是否保留（3.2.4、3.3、3.4 分别展开）。

三条路径在内核里的"占地面积"极小，值得先列出全部裁剪点，后面各节逐一展开：

| 裁剪点 | 行号 | `kDirectStore` | `kResidentA` | `kResidentB` |
|---|---|---|---|---|
| 启动形态 `__mix__(1, ?)` | `:27` | AIV 子块数 0 | — | — |
| 累加原子模式 | `:82` | AIC 设 `asc_set_atomic_add_float` | — | — |
| L1 布局声明 | `:59-60` | — | `l1a` 宽度换成整条 K、单 stage | — |
| 块首整块搬运 | `:109-110` | — | `!reuse_a` 时一次搬完 A 的全部 K | — |
| K 循环内 GM→L1 | `:117-118` | — | A 的搬运 `if constexpr` 消失 | `!reuse_b` 时跳过 B 的搬运 |
| L1→L0A 源与偏移 | `:142-143` | — | 源固定 `l1a[0]`，偏移加 `k_blk_idx * BLOCK_K` | — |
| L1→L0B | `:144-145` | — | — | `!reuse_b` 时整条跳过 |
| 借 UB | `:129-131` | 不借 | — | — |
| 交付 | `:155-163` | `copy_l0c_to_gm` 直写 | — | — |
| AIV 分支 | `:173-177` | `:174` 直接 return | — | — |

一个贯穿全天的观察：**裁剪只删指令、不改协议**。流水线事件对（`:116/:119/:121/:148/:152/:167`）一个不少，`s`、`s_ab`、`s_cd`、`s_epi` 照常推进——特殊路径的省钱之处在于少搬数据、少一层握手，而不是换一套同步逻辑。这也是它们能作为"默认关闭的选项"安全存在的原因。

### 3.2 kDirectStore：删掉 AIV、UB 与整层块内同步

#### 3.2.1 启动形态与被裁剪的代码

第 25 天已经见过 `__global__ __mix__(1, kDirectStore ? 0 : 2)`（`:27`）：普通路径一个 block 由 1 个 AIC 加 2 个 AIV 三方执行；直写路径**没有 AIV 子块**，同一份代码里 `else` 分支对不存在的执行者来说不可达，`:174` 的 `return` 只是让语义明确无害：

```cpp
} else {
    if constexpr (kDirectStore) return;                                   // :174  没有 AIV 的路径
    if constexpr (is_k_grouped && !kWithAccumulation)
        epilogue.template aiv_zero_empty_groups<kNumCores, kAlignment>(...);  // :175-176
    epilogue.aiv_initialize();                                            // :177
```

AIC 分支里被裁剪的是第 25 天四步借还协议的"借"与"还"两端：

```cpp
if (k_blk_idx == 0 && !kDirectStore) {          // :129  借：普通路径才等 AIV 归还 UB stage
    epilogue.aic_wait_free(s_epi);
}
...
if (k_blk_idx == num_k_blocks - 1) {
    if constexpr (kDirectStore) {                // :156  还：AIC 自己写 GM，不经过 UB
        const auto d = gm_d.batch(batch_idx).group(is_k_grouped ? scheduler.get_group_idx() : 0, shape_m, shape_n);
        copy_l0c_to_gm(d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(),
                       l0c_tile, eff_mad_m, eff_mad_n, d.stride_outer, kL2CtrlStoreCd);   // :158-159
    } else {
        epilogue.aic_store(l0c_tile, s_epi, eff_mad_m, eff_mad_n);                // :161
    }
}
```

两条路径的数据通路对比：

```text
普通路径（__mix__(1, 2)：1 AIC + 2 AIV）：
  L0C ──FIX──> UB[stage] ──V(算子)──> ──MTE3──> GM
        ▲                        │
        └────── 块内事件借还 ×4 次握手/格 ──────┘

直写路径（__mix__(1, 0)：仅 1 AIC）：
  L0C ──FIX──> GM      （无 UB 中转、无 AIV、无块内事件）
```

注意 `epilogue` 对象仍然构造（`:53-55`），它的 `constexpr` 布局与全部 `static_assert`（`epilogue.hpp:28-35`）照常参与编译——只是运行时没有任何一方再调用它的方法；`s_epi`/`epi_base`（`:107/:122/:164/:169`）照常计算但无人消费。配置里的 `num_epilogue_stages` 也仍然存在（`heuristics.hpp:190-194` 的计算不因直写而跳过），属于"编译期在场、运行时闲置"。

还有一处前置差异：accumulation 的原子加模式由谁设置。普通路径是 AIV 在 `aiv_initialize` 里设（按 `cd_dtype_t` 选 bfloat 或 float，`epilogue.hpp:91-96`）；直写路径没有 AIV，改由 AIC 在进入主循环前设置（`:82`），且**只有 float 版本**：

```cpp
if ASCEND_IS_AIC {
    if constexpr (kDirectStore && kWithAccumulation) asc_set_atomic_add_float();   // :82
```

配合设备断言 `:37-39`：

```cpp
if constexpr (kDirectStore) {
    static_assert(!kWithAccumulation || std::is_same_v<cd_dtype_t, float>);        // :38
}
```

即**直写 + 累加只允许 FP32 输出**。为什么不允许 BF16？仓库没有注释说明；可见的事实是普通路径同时支持两种原子加而直写只选了 float——从第 25 天的表格知道，原子加发生在 GM store 一侧，直写路径的 store 是 `copy_l0c_to_gm`（内含 `F322BF16` 转换，见下节），"转换后再原子加 BF16"与"FP32 原子加"的数值与硬件代价不同，仓库选择了后者。把这条记为"约束 + 可见事实"，不猜测指令级行为。

#### 3.2.2 `copy_l0c_to_gm`：一条 FIX 队列上的 L0C → GM

直写的全部新增工作就是 `:158-159` 这一次拷贝。看它的实现（`ascend/copy.hpp:287-306`）：

```cpp
template <typename T, typename ST>
__aicore__ inline void copy_l0c_to_gm(
    __gm__ T* dst, l0c_ptr<ST> src,
    uint32_t m_elems, uint32_t n_elems,
    uint64_t dst_stride_elems,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
    asc_quant_mode quant_mode = infer_quant_mode<T, ST>(),
    bool relu = false,
    bool transpose_out = false,
    asc_unit_flag_mode unit_flag = asc_unit_flag_mode::ENABLE_UPDATE
) {
    if (transpose_out) { ... } else {
        asc_set_l0c2gm_nz2nd(1, 0, 0);                    // :301  NZ → ND 格式转换参数
    }
    asc_copy_l0c2gm(dst, src.ptr(), n_elems, m_elems, dst_stride_elems, src.shape_m, l2_ctrl, unit_flag, quant_mode, ...);
}
```

对照内核调用逐参数记录：

| 实参 | 值 | 含义 |
|---|---|---|
| `dst` | `d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr()` | 输出子块左上角的 GM 地址（`gm_ptr::index` 按 major 计算偏移，`common.hpp:156-165`） |
| `src` | `l0c_tile = l0c[s_cd].as_mad_aligned(eff_mad_m, eff_mad_n)` | 本子块的 L0C 累加结果（`:127`） |
| `m_elems` / `n_elems` | `eff_mad_m` / `eff_mad_n` | 真实行数；**列数是向上对齐后的值**（`:125-126`，3.5 节的关键） |
| `dst_stride_elems` | `d.stride_outer` | D 的行跨度（元素数），来自 host 的 `outer_stride_d`（`gemm.hpp:193`） |
| `l2_ctrl` | `kL2CtrlStoreCd`（模板参数） | host 默认选 `NOTALLOC_CLEAN`（`heuristics.hpp:664`） |
| `quant_mode` | 默认 `infer_quant_mode<cd_dtype_t, float>()` | BF16 输出即 `F322BF16`、FP32 即 `NoQuant`（`copy.hpp:235-238`）——与 `copy_l0c_to_ub` 的默认值**完全同款**（`copy.hpp:272`） |
| `unit_flag` | 默认 `ENABLE_UPDATE` | 第 24 天的"读取侧收尾"，对 L0C 累加器的消费语义 |

也就是说，第 25 天"输出转换发生在 AIC 的 FIX 拷贝里"的结论对直写同样成立：`Identity` 算子下 FP32→BF16 的转换就折叠在这条 FIX 队列的拷贝中，只是目的地从某个 AIV 的 UB 换成了 GM，`nz2nd` 的 NZ→行主序格式转换也一并在拷贝里完成（`:301`）。这正是设备断言 `:56-57` 只允许 `Identity` 的原因——任何需要 V 管道计算的算子（bias、alpha、随机舍入、FP8 量化）都没有了执行者。

#### 3.2.3 约束链：设备断言与 host 闸门

直写不是随便可开的。设备侧两条断言（`:38`、`:56-57`）之外，真正的把关在 host 的 `can_direct_store()`（`heuristics.hpp:521-533`）：

```cpp
static bool can_direct_store(const GemmDesc& d, const GemmDtype& dtype, const GemmConfig& c) {
    const auto hw = get_hw_desc();
    const auto stride_d = d.outer_stride_d ? d.outer_stride_d : d.n;
    const int align_m = is_mn_major(d.major_a) ? hw.get_mad_alignment_k(dtype.elem_bits) : 1;
    const int align_n = is_mn_major(d.major_b) ? hw.get_mad_alignment_k(dtype.elem_bits) : FRAC_MN;
    return d.gemm_type == GemmType::Normal && dynamic_cast<const IdentityEpilogue*>(d.epilogue_class.get()) &&
           (dtype.elem_bits != 4 || (is_k_major(d.major_a) && is_k_major(d.major_b))) &&
           (!d.acc || d.cd_dtype == at::kFloat) && d.m % align_m == 0 && d.n % align_n == 0 &&
           reinterpret_cast<uintptr_t>(d.d_ptr) % hw.ub_bank_bytes == 0 &&
           stride_d * dtype.epilogue_elem_bits / 8 % hw.ub_bank_bytes == 0 &&
           (int64_t(ceil_div(d.m, c.block_m)) * ceil_div(d.n, c.block_n) <= d.num_cores ||
            (d.m <= c.block_m && c.mad_n * dtype.epilogue_elem_bits / 8 % hw.l2_sector_bytes == 0));
}
```

逐条解读（BF16 下 `elem_bits=16`，`get_mad_alignment_k(16)=16`，`FRAC_MN=16`，`ub_bank_bytes=32`，`l2_sector_bytes=128`）：

| 条件 | 含义 | 依据与备注 |
|---|---|---|
| `gemm_type == Normal` | 只允许普通 GEMM | 设备侧 `:157` 防御性地写了 k-grouped 的 D 寻址分支，但 host 只放 Normal 通过；M/K-Grouped、Batched 的 D 与 epilogue 语义不同 |
| `IdentityEpilogue` | 无 bias/alpha/随机舍入/FP8 量化 | 设备断言 `:56-57` 的镜像；这些算子都需要 AIV 的 V 管道 |
| `elem_bits != 4 \|\| 双 K-major` | FP4 限制（BF16 恒过） | 属 FP8/FP4 路径的约束 |
| `!acc \|\| FP32` | 累加仅 FP32 | 设备断言 `:38` + `:82` 只设 float 原子加（3.2.1） |
| `d.m % align_m == 0` | M 对齐（K-major A 时 `align_m=1` 恒真；MN-major 时 16） | 与 `make_gemm_candidate` 中 MN-major 的 `align_m` 同源（`heuristics.hpp:120`）；普通路径容忍任意 M 靠的是 AIV 按真值行数写回 |
| `d.n % align_n == 0` | **N 必须对齐（K/MN-major 都是 16）** | 3.5 节演算：`eff_mad_n` 向上取整后直接决定 GM 写出宽度 |
| `d_ptr % 32 == 0` | D 基址 32B 对齐 | 与 `epilogue.hpp:29`"L0C-to-UB row stride must be 32-byte aligned"同族的 L0C 出口对齐要求，目的地从 UB 换成 GM（从两处代码的平行性推断） |
| `行跨度字节 % 32 == 0` | D 每行 32B 对齐 | 同上；`stride_d` 取 `outer_stride_d ?: n`，strided D 也按真实跨度检查 |
| 单 wave 或（单 m 块 且 `mad_n` 行字节 % 128 == 0） | 输出网格不超过一个 wave；或 M 装进一个 block 且每次写出的 `mad_n` 宽行是 128B L2 sector 的整数倍 | 仓库未注释此条动机；从形式看它约束的是 L0C→GM 直写的 L2 访问粒度与多 wave 写回模式，具体硬件原因需进一步实验，**此处只列可见约束** |

一个值得体会的分工：设备断言只拦"编译期就知道错"的组合（Identity、float 累加），而 shape 对齐、地址对齐、wave/sector 这些**依赖运行时指针与 shape 的条件**全部由 host 承担——设备代码里找不到它们的影子。3.5 节会看到为什么 shape 条件必须是 host 责任：设备路径本身没有越界检查，靠的是"对齐保证了不会越界"这一前提。

#### 3.2.4 host 何时选择直写：代价模型

`refine_gemm_pipeline`（`heuristics.hpp:535-543`）的试穿逻辑只有四行：

```cpp
if (can_direct_store(d, dtype, config)) {
    auto direct = config;
    direct.direct_store = true;
    if (get_gemm_output_cost(d, dtype, direct) <= get_gemm_output_cost(d, dtype, config)) config = direct;
}
```

`get_gemm_output_cost`（`:474-494`）对输出段的建模：

```cpp
if (with_bias && c.num_epilogue_stages >= 2 && num_waves >= 2)
    return num_waves * std::max(fix_ns, bias_ns + gm_ns + vector_ns);
return num_waves * (c.direct_store ? std::max(fix_ns, gm_ns)            // :493  直写：单流
                                   : std::max(fix_ns, bias_ns) + gm_ns + vector_ns);  // 普通：两段串行
```

其中 `fix_ns` 是 L0C 出口的搬运时间（`:486`，用 `l0c_to_ub_bytes_per_cycle` 速率），`gm_ns` 是 GM 写带宽时间（`:487`，`gm_bytes` 在累加时乘 `(1 + d.acc)`——原子加的读改写使 GM 流量翻倍，`:484-485`），`vector_ns` 是 AIV 的向量计算时间。对 `Identity`（`bias_ns = vector_ns = 0`）：普通路径记 `fix + gm`（两跳串行），直写记 `max(fix, gm)`（单流取瓶颈）——**`max ≤ sum` 恒成立，所以只要 `can_direct_store` 通过，直写在该模型下不会更差**。真正的门是资格条件；代价比较只是形式上的复核。

从条件反推适用场景：`单 wave`（输出块总数 ≤ 核数）或 `d.m ≤ block_m`（M 装进一个 block，block_m 上限 512，`heuristics.hpp:604`）。前者是小规模 GEMM，后者是**解码/前向的瘦 M 形状**（M=1、128 一类）；再叠加"必须 Identity、最好别累加（累加还得 FP32）"，直写瞄准的是"没有 epilogue 故事、输出又窄"的纯 GEMM。

#### 3.2.5 回答第 25 天的遗留问题

第 25 天 A-5 问："如果让你把归还 arrive 从 MTE3 提前到 V（仿 `mqa_logits.hpp:420`），需要什么前提？"现在可以给出完整答案：**归还提前的前提是缓冲的下一个生产者只依赖'消费者已处理完缓冲内容'，而不依赖'排在后面的尾部操作'（GM 写回）是否完成**。`mqa_logits` 把归还放在 V 队列，让下一轮 FIX 拷贝不被 MTE3 的 GM 长尾阻塞，代价是"归还 ≠ GM 落盘"，调用方要自己保证不再踩踏。`kDirectStore` 是同一思路的极限：**把 UB 这一层整体删掉，借还协议随之消失**——生产者（mmad 写 L0C）与消费者（`copy_l0c_to_gm` 读 L0C）都在 AIC 的队列体系内，跨队列次序由队列语义与 `ENABLE_UPDATE/KEEP`（第 24 天）承担。它成立的前提正是 3.2.3 的约束链：无 epilogue 计算、对齐满足、Normal——条件不满足时，UB 这级缓冲与 AIV 的灵活性是必需的，"提前归还"最多到 V 为止。

### 3.3 kResidentA：A 的整条 K 常驻 L1

第 22 天已经预告过 `:143` 的读偏移。现在把 resident A 的四处改动连起来看。

**改动一：L1 布局（`:59-60`）。**

```cpp
constexpr l1_ptr<bfloat16_t, kMajorA> l1a(BLOCK_M, kResidentA ? aligned(SHAPE_K, get_frac_k<bfloat16_t>()) : BLOCK_K, 0);
constexpr l1_ptr<bfloat16_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kResidentA ? 1 : kNumL1Stages));
```

普通路径下 `l1a` 是 `kNumL1Stages` 个 `BLOCK_M × BLOCK_K` 的 stage 环；resident A 把它换成**一个** `BLOCK_M × aligned(SHAPE_K, 16)` 的大缓冲（BF16 的 `get_frac_k` 是 16，`common.hpp:73-79`），`l1b` 紧随其后从 `l1a.offset(1)` 开始。容量断言 `:66`（`l1b.offset(kNumL1Stages) <= L1SizeBytes`）自动覆盖新布局。注意这一行是 `constexpr`——**SHAPE_K 必须是编译期常量**，这就是设备断言 `:40` 的由来：

```cpp
static_assert(!(kResidentA || kResidentB) || (kGemmType == GemmType::Normal && SHAPE_K));   // :40
```

resident 只允许 Normal GEMM，且 K 必须编译期已知（`SHAPE_K != 0`；`:47-49` 里 0 表示用运行时值）。host 侧镜像在 `heuristics.hpp:546`：`d.gemm_type == GemmType::Normal && compiled_k`。

**改动二：块首一次性整块搬运（`:109-110`）。**

```cpp
if (kResidentA && !reuse_a)
    copy_gm_to_l1(l1a[0], a.index(m_idx, 0), actual_m, shape_k, kL2CtrlA);
```

位置在 K 循环**之外**、输出块循环之内：本块的 A 行面板（`actual_m × shape_k`，整条 K）一次搬进 `l1a[0]`。是否搬运由 `reuse_a` 决定（`:92`）：

```cpp
const bool reuse_a = m_idx == last_m_idx;    // 上一个块与本块同属一个 m 行块
```

`last_m_idx` 初始化为 `shape_m`（`:87`，不可能的值）保证第一个块必然装载。

**改动三：K 循环内的 A 搬运消失（`:117`）。**

```cpp
if constexpr (!kResidentA) copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, kL2CtrlA);
```

`if constexpr` 编译期裁剪——resident A 下 K 循环里只剩 B 的 GM→L1（`:118`）。流水线事件（`:116/:119/:121`）原样保留，`s` 环照常转动，只是它现在只服务于 `l1b`。

**改动四：L1→L0A 的源与偏移（`:143`）。**

```cpp
copy_l1_to_l0a(l0a_tile, l1a[kResidentA ? 0 : s], m_mad_idx, (kResidentA ? k_blk_idx * BLOCK_K : 0) + k_mad_idx);
```

L1→L0A **照常发生**（每个子块的每个 `k_mad` 一次，L0A 环照常轮转）——resident A 复用的是 L1 这一级，不是 L0。区别只有两点：源 stage 固定 `l1a[0]`；K 偏移从"stage 内的 `k_mad_idx`"变成"整条缓冲里的 `k_blk_idx * BLOCK_K + k_mad_idx`"（第 22 天的"跨出当前 stage 的补偿"）。

**host 侧的资格与账本（`heuristics.hpp:548-555`）。**

```cpp
const int64_t a_bytes = int64_t(config.block_m) * align(d.k, FRAC_MN) * sizeof(uint16_t);
const int64_t b_bytes = int64_t(config.block_n) * config.block_k * sizeof(uint16_t);
const int num_l1_stages = a_bytes < L1SizeBytes ? std::min<int64_t>(hw.num_pipe_events, (L1SizeBytes - a_bytes) / b_bytes) : 0;
if (d.m <= config.block_m && num_l1_stages > config.num_l0_stages) {
    resident.resident_a = true;
    resident.num_l1_stages = num_l1_stages;
}
```

三个条件各有所指：`a_bytes < L1SizeBytes`——A 的整条行面板放得进 L1（512KB）；`num_l1_stages > num_l0_stages`——塞进 A 之后 B 的流水深度不能比原来浅；`d.m <= block_m`——**M 装进一个 block**。最后这条与 `reuse_a` 的语义严丝合缝：`num_m_blocks == 1` 时所有块的 `m_idx` 都是 0，除每核第一个块外 `reuse_a` 恒真。若 M 需要多个 block，同一核的连续块经 swizzle 后通常换到不同的 m 行块（N-first 遍历），复用率低，host 干脆不放行。代价模型对 resident A 的三处记账：装载次数按 `min(num_blocks, num_cores)` 计（`:351`，每核一次）；K-major 时整行按 sector 对齐计费（`:305-306`）；首块装载"不能与计算重叠"（`:405` 注释与 `:411`），以及 m < block_m 时按每次装载 64 cycle 的发射开销校准（`:499-505`，`kGmLoadIssueCycles`）。

### 3.4 kResidentB：B 常驻 L0B

resident B 的裁剪点是两条搬运（`:118` 与 `:144-145`）：

```cpp
if (!kResidentB || !reuse_b) copy_gm_to_l1(l1b[s], b.index(n_idx, k_idx), actual_n, actual_k, kL2CtrlB);   // :118
...
if ((!kResidentB || !reuse_b) && (BLOCK_M == MAD_M || BLOCK_N != MAD_N || BLOCK_K > MAD_K * kNumL0Stages || m_mad_idx == 0))
    copy_l1_to_l0b(l0b_tile, l1b[s], n_mad_idx, k_mad_idx);                                                // :144-145
```

`reuse_b = n_idx == last_n_idx`（`:93`）——上一个块与本块同属一个 n 列块。与 resident A 的一个重要差别：`:118` 是**运行时 `if`**（同一拷贝形状，只是跳过），而 resident A 的 `:117` 是 `if constexpr`（拷贝形状本身不同）。

**先理清 `:144` 后半段本来的含义**（第 22/24 天见过，这里换个角度）：这是"块内跨子块复用 L0B"的条件。B 的 L0B tile 只随 `n_mad_idx` 变化；当 `BLOCK_N == MAD_N`（每块只有一个 n 子块）且环不回绕（`BLOCK_K <= MAD_K * kNumL0Stages`）且有多行子块（`BLOCK_M != MAD_M`）时，B 在 `m_mad_idx == 0` 装进 L0B 后，同一块内其余 `m_mad` 全部免装。host 模型把它记作 `num_l0b_loads = 1`（`heuristics.hpp:438`）。**resident B 是把这条块内复用扩展到跨输出块**：`n_idx` 不变时连 `m_mad_idx == 0` 的那次也不装了。这也解释了设备断言为什么要求 `BLOCK_N == MAD_N`：

```cpp
static_assert(!kResidentB || (SHAPE_K <= kNumL0Stages * MAD_K && BLOCK_N == MAD_N));   // :45
```

**槽位稳定性论证**（本节核心）。`s_ab` 在 `k_mad` 循环的 init 子句里声明（`:133`），因此**每个子块、每个 k_blk 都从 0 重新计数**——B 的第 j 个 K 块恒写入 `l0b[j % kNumL0Stages]`，与 k_blk 无关。跨输出块复用 B 要求每个槽位的内容在新块里仍然正确，这需要两个条件：

1. **全部 K 的 chunk 数 ≤ 环长**：`SHAPE_K <= kNumL0Stages * MAD_K`（断言 `:45`）。此时第 j 个 chunk 落在 `l0b[j]`（`j < 环长`，取模是恒等），槽位与 chunk 一一对应。
2. **只有一个 k_blk**（`num_k_blocks == 1`）。若有多个 k_blk，第 0 个和第 1 个 k_blk 的第 j 个 chunk 会先后写进同一个 `l0b[j]`（s_ab 每 k_blk 重置），装载块结束时槽里留的是**最后一个** k_blk 的数据，复用块却期望第一个——除非两者相同，否则出错。

有意思的是，断言 `:45` 并**不**直接包含 `num_k_blocks == 1`；这条由 host 的配置生成规则补全：`make_gemm_candidate` 里 `mad_k <= max(k_alignment, block_k / num_l0_stages)`（`heuristics.hpp:135`），BF16 下 `k_alignment = 16 < block_k/2`（`block_k ≥ min_block_k = 128`，`:620`），所以 `mad_k ≤ block_k/2`，于是 resident 资格里的 `d.k <= mad_k * num_l0_stages`（`:556-557`）蕴含 `d.k <= block_k`，即 `num_k_blocks == 1`。**设备断言与 host 闸门合力才钉死正确性**——这是"为什么 host 条件不能随便删"的最好例证，也提醒我们读断言时要问"它没查什么"。

顺带一个对照：断言 `:45` 在运行时 K（`SHAPE_K = 0`）下会退化成 `0 ≤ x` 恒真——所以 `:40` 同时要求 `SHAPE_K` 编译期非零，host 的 `compiled_k` 闸门（`:546`）与之配套。

**host 资格（`:556-557`）**：`num_n_blocks == 1 && num_waves > 1 && block_n == mad_n && d.k <= mad_k * num_l0_stages`。`num_n_blocks == 1`（N 装进一个 block，配合 `reuse_b` 恒真）、`num_waves > 1`（每核至少两个块，复用真的会发生）、`block_n == mad_n` 与 `d.k` 上限对应设备断言。代价模型：B 的 GM 装载块数按 `min(num_blocks, num_cores)` 计（`:352`），L1→L0B 的字节按 1 倍而不是 `num_waves` 倍计（`:441`）。适用场景是**N 小、K 小、M 大**的 Normal GEMM（比如 N、K 都装得进 block 而 M 需要很多块的多波形状）。resident A 与 resident B 可以同时开（`:547-557` 里两个条件独立判定）：M、N 都装进一个 block 时，每核只有第一个块需要从 GM 装载 A 和 B，后续块零装载。

最后看一眼隔壁的实现方式，避免概念混淆：FP8/FP4 的 `fp8_gemm_impl` 没有 `kResidentA/B` 模板参数，它用的是 `kSingleMBlock`（`fp8_gemm.hpp:34, 97`，把调度器的 shape_m 钳到 BLOCK_M）加一套 **L1 stage 环回卷**的复用（`:124-150` 的 `can_reuse_a/b` 与 `l1a_base_stage_idx` 回退，资格是 `num_k_blocks <= kNumL1Stages`，`:141`）——同为"操作数复用"，BF16 与 FP8 的机制不同；今天只讲 BF16。

### 3.5 非对齐 shape：普通路径的容错与直写路径的收紧

先回顾普通路径的三层边界机制（第 17/19/22 天）：

1. **块级**：`actual_m = min(shape_m - m_idx, BLOCK_M)`、`actual_n = min(shape_n - n_idx, BLOCK_N)`（`scheduler.hpp:129-139`）、`actual_k = min(eff_shape_k - k_blk_idx*BLOCK_K, BLOCK_K)`（`:114`）——真实长度进循环边界与搬运规模。
2. **子块级**：`eff_mad_m = min(actual_m - m_mad_idx, MAD_M)`（真值）、`eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N), 16)`（**向上对齐**）、`eff_mad_k = min(...)`（真值）（`:125-126, 137`）；tile 经 `as_mad_aligned` 把形状向上取整到 fractal 粒度（`:127`、`tile_ptr.hpp:52-55`）。
3. **写回级**：AIV 的 `copy_ub_to_gm` 用 `MadBlock.num_rows/num_cols`——两个 `min` 真值（`epilogue.hpp:65-66`、`:81-82`）——只写有效区域，UB 里对齐出来的填充列不落 GM。

`eff_mad_n` 为什么要向上对齐？第 25 天讲过：FIX 拷贝（L0C→UB）的行必须 32B 对齐（`epilogue.hpp:29`），尾块的真值列数不是 16 的倍数时补到 16，填充列留在 UB 里等 AIV 用真值裁掉。**直写路径没有 AIV 这把裁刀**——`copy_l0c_to_gm` 的 `n_elems` 直接用 `eff_mad_n`（`:158-159`），向上对齐的宽度原样写进 GM。这就是 `d.n % 16 == 0` 的由来，用一个小演算看清后果：

> **演算：N=100 为何被直写拒绝。** 设 `BLOCK_N = MAD_N = 64`、`BLOCK_M = 64`、K 对齐。`num_n_blocks = ceil(100/64) = 2`，第二个 n 块 `actual_n = 100 - 64 = 36`；其唯一子块 `min(36, 64) = 36`，`eff_mad_n = aligned(36, 16) = 48`。普通路径：AIV 写 `num_cols = 36`，只覆盖 D 的第 64..99 列。直写路径：`copy_l0c_to_gm` 写 48 列——第 64..111 列。D 每行只有 100 列，第 100..111 列要么踩进下一行的前 12 个元素（行主序、stride = 100 时），要么在最后一行越出张量。因此 host 要求 `d.n % 16 == 0`：此时尾块真值列数是 16 的倍数（`n_idx`、`MAD_N` 都是 16 的倍数），`aligned()` 退化为恒等，写出宽度 == 真值宽度。M 侧没有这个问题（`eff_mad_m` 是 `min` 真值，行数不越界），所以 K-major A 时 `align_m = 1`；MN-major A 时 M 映射到操作数的 fractal K 轴，host 沿用 16 的对齐（与 `heuristics.hpp:120` 的 `align_m` 同源）。

普通路径对非对齐 shape 的**宽容**是系统性的：M/N/K 尾块靠 `min` 真值 + AIV 裁剪；K=17、15、234、4095、4 这类 shape（`tests/generators.py:519-541` 的 `unaligned-k` 列表）与 m/n=17、15、130、257、513（`:491-500` 的 `unaligned-mn` 列表）在四种 major 组合下全部通过（`tests/test_bf16_gemm.py:25-47` 的选择器）。另外两类"退化 shape"由 host 直接消化（`csrc/utils/layout.hpp:54-86` 的 `gemm_early_return`）：

- `m == 0 || n == 0`：直接返回，不启动内核（`:57-58`）；
- `k == 0`：无累加语义下 `d.zero_()`（有独立 `c` 时先 `d.copy_(c)`）、FP8 输出时 `sfd->zero_()`，然后返回（`:75-80`）——**Normal 的空 K 根本不进内核**；
- 累加且 `c` 与 `d` 不别名时，host 先 `d.copy_(c)` 完成初值化再启动（`:83-84`，"Legacy accumulation paths initialize D before launch"）。

最后是 m-grouped 的组尾（今天只立个牌子，细节第 36 天）：m-grouped 要求 `kAlignment == BLOCK_M = 256`（`scheduler.hpp:146`；`jit.hpp:59`），组起点按 256 对齐，但 `get_actual_m` 目前对 m-grouped **恒返回 BLOCK_M**（`:130-132`，带 TODO）——组尾的 padding 行会被照常计算，测试通过 `ensure_zero_padding` 把 A 的 psum 间隙清零来保证 padding 输出为零（`generators.py:338-340`），比较时也只比对有效行（`:357`）。

### 3.6 空组：AIC 跳过、AIV 兜底

K-grouped 的 `grouped_layout` 是前缀和数组：`grouped_layout[i]` = 第 i 组的累计结束 K，组 g 的有效 K 区间是 `[aligned(psum[g-1], 256), psum[g])`（`scheduler.hpp:25-31` 注释）。**空组**即区间长度为 0（`current_shape_k == 0`）。

**AIC 侧：调度器直接跳过**（第 22 天已见，这里补上 cumsum 细节）：

```cpp
while (true) {
    if (current_shape_k > 0 and block_idx < current_k_block_cumsum + num_blocks)   // :167  空组连一个块都没有
        break;
    if (++current_group_idx == num_groups) return false;
    last_psum_k = aligned(current_psum_k, kAlignment);
    current_psum_k = static_cast<uint32_t>(grouped_layout[current_group_idx]);
    current_sf_k_cumsum = last_psum_k / MX_SF_DIVISOR;
    if (current_shape_k > 0)          // :174  空组不消耗块配额，cumsum 不前进
        current_k_block_cumsum += num_blocks;
    current_shape_k = current_psum_k - last_psum_k;
}
```

空组不产出任何 (m_block, n_block)，K 循环零次迭代——AIC 对空组的 D 区域**一个字节都不写**。

**AIV 侧：`aiv_zero_empty_groups` 兜底写零**（`epilogue.hpp:108-141`）。调用位置在 AIV 分支开头、`aiv_initialize` 之前（`:175-177`），条件 `is_k_grouped && !kWithAccumulation`。逐段走读：

```cpp
static_assert(!kWithAccumulation && !std::is_same_v<cd_dtype_t, float8_e4m3_t>);        // :111  禁止 acc 与 FP8 输出
const uint32_t aiv_idx = block_idx * kNumAIVs + asc_get_sub_block_id() - (2 - kNumAIVs); // :112  AIV 全局编号
const uint32_t num_n_tiles = ceil_div(shape_n, MAD_N);
const uint32_t num_tiles = ceil_div(shape_m, MAD_M) * num_n_tiles;                       // :113-114  D 一组的 MAD 粒度 tile 数
const ub_ptr<cd_dtype_t> output_ub(MAD_M, MAD_N, ub.addr);                               // :115  借 UB 的 stage 0 放零值
bool initialized = false;
uint32_t k_begin = 0;
for (uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    const uint32_t k_end = static_cast<uint32_t>(grouped_layout[group_idx]);
    const bool is_empty = k_begin == k_end;                                              // :119-121  与调度器同式的空组判定
    k_begin = aligned(k_end, kAlignment);
    if (!is_empty || aiv_idx >= num_tiles) continue;                                     // :122
    if (!initialized) {                                                                  // :123-128  懒初始化：写一次零 tile
        zero_values(output_ub.template ptr<uint32_t>());                                 //   （V 管道填零，:99-106）
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);                                    //   等 V 写完，MTE3 才能搬
        initialized = true;
    }
    for (uint32_t tile_idx = aiv_idx; tile_idx < num_tiles; tile_idx += kNumCores * kNumAIVs) {  // :129  全体 AIV 分摊
        const uint32_t m_idx = tile_idx / num_n_tiles * MAD_M;
        const uint32_t n_idx = tile_idx % num_n_tiles * MAD_N;
        copy_ub_to_gm(d.group(group_idx, shape_m, shape_n).index(m_idx, n_idx).ptr(), output_ub,
                      min(shape_m - m_idx, MAD_M), min(shape_n - n_idx, MAD_N), ...);    // :132-133  真值行列写零
    }
}
if (initialized) {                                                                       // :136-139  归还共享 UB
    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);                                         //   所有零写消费完 stage 0 才放行
}
```

五个要点：其一，空组判定 `k_begin == k_end` 与调度器的 `current_shape_k == 0` 用同一套前缀和与 256 对齐，两侧必然一致。其二，`aiv_idx` 给两个 AIV 编全局号（`block_idx * kNumAIVs + sub_block_id`），所有空组的 tile 以 `kNumCores * kNumAIVs` 为步长分摊——注意这里的 `kNumAIVs` 是 Epilogue 的模板参数（取 2），不是设备级的 `kNumAIVs = 64`（`common.hpp:52`），两者同名不同物（第 25 天踩过的坑）。其三，零值只写一次 UB（懒初始化），`V→MTE3` 的握手保证搬运看到完整的零。其四，写零也用 `min` 真值行列，不越界。其五，结尾的 `MTE3→V` 握手（注释 "Release the shared UB only after all zero stores have consumed it"）把 UB 的 stage 0 还回来——之后 `aiv_initialize`（`:177`）才置位"全部 UB stage 空闲"的块内事件，而 AIC 的首次 `aic_wait_free` 等的正是这些事件（第 25 天 3.2 的结论）：**空组清零与 AIC 的首次交付由此间接串行化**。

**accumulation 语义**：`:175` 只在 `!kWithAccumulation` 时清零，`:111` 的断言再拦一道。累加时空组的正确行为是"D 保持初值"（`D[g] += 0`）：初值由调用方 `d.copy_(c)`（`tests/common.py:56-58`）或 host 的 `d.copy_(c)`（`layout.hpp:83-84`）提前放好，参考实现同样是 `d[i] = c[i] + 0`（`generators.py:394-397`）。若清零就会把初值抹掉——这就是断言的动机。**全部组都空**的极端情况由 host 早退接住：total k 为 0 时 `gemm_early_return` 直接 `d.zero_()` / `d.copy_(c)` 返回（3.5 节）。

测试侧，`kgrouped` 选择器跑三种变体（`generators.py:606-613`）：`normal`（全实）、`empty`（约 25% 随机置空，且首组必空、末组必实）、`single`（只有中间一组非空），并交叉四种 accumulate/输出 dtype 组合（`:629`）。

### 3.7 手工推演示范

**演算一：直写资格判定（M=128 前向形状）。** 设 Normal BF16 GEMM、nt 布局（A/B 均 K-major）、`M=128, N=7168, K=7168`、不传 `c`/`alpha`、BF16 输出、`compiled_dims="nk"`（`nt`/`nn` 的默认，`gemm.hpp:635`）。假设主搜索选中 `BLOCK 128×512×128`、`MAD 128×256×64`（L0C 恰好 128×512×4B=256KB；MAD 由 split (1,2) 得到）。逐条过 `can_direct_store`：Normal ✓；Identity ✓；`elem_bits=16≠4` ✓；不累加 ✓；`align_m=1`（K-major A）→ `128%1=0` ✓；`align_n=16` → `7168%16=0` ✓；D 基址 32B 对齐（torch 分配通常满足）✓；`stride_d=7168`，`7168×2B=14336B % 32 = 0` ✓；块数 `1×14=14 ≤ 32` 核 → **单 wave 条件通过**。结论：资格成立；再按 3.2.4 的代价比较（`max(fix,gm) ≤ fix+gm` 恒真）采纳直写。对照：同 shape 换 `M=4096`，块数 `8×14=112 > 32` 且 `4096 > block_m` → 两条分支都不通，直写被拒；换 `N=4097` → `4097%16≠0`，直接被对齐条件拦下（3.5 的演算说明了拦它的理由）。

**演算二：resident A 的 L1 账本与复用轨迹。** 设 `M=64, N=8192, K=512`（K 编译期）、`BLOCK 64×128×128`、`MAD 64×64×64`、`kNumL1Stages=8`、`kNumL0Stages=2`、32 核。L1 布局：`l1a` = 1 × (64×512×2B) = **64KB**（宽度 `aligned(512,16)=512`）；`l1b` 从 64KB 起、8 × (128×128×2B=32KB) = **256KB**；合计 320KB ≤ 512KB，`:66` 断言通过。块调度：`num_m_blocks=1`（< kGroupM=4，走快速 row-major 路径，`scheduler.hpp:209-211`），`num_n_blocks=64`，共 64 块、2 wave；核 0 分到块 0 与块 32，即 `(m=0, n=0)` 与 `(m=0, n=32×128)`——`m_idx` 都是 0，第二块 `reuse_a=true`，**A 的 GM→L1 一次都不做**。K 循环：`num_k_blocks = 512/128 = 4`，每块内 B 按 `s=0..3` 装进 `l1b`（环长 8，不重叠）；L1→L0A 从 `l1a[0]` 以偏移 `k_blk_idx*128 + k_mad_idx` 取数。流量对比：不 resident 时 A 的 GM→L1 总量 = 64 块 × 4 k_blk × (64×128×2B=16KB) = 4MB；resident 时 = 32 核 × 64KB = 2MB——host 模型 `a_load_blocks = min(64, 32) = 32`（`:351`）正是这笔账。host 资格复核：`d.m=64 ≤ block_m=64` ✓；`a_bytes=64KB < 512KB`，`num_l1_stages = min(8, (512-64)/32) = 8 > 2` ✓。

**演算三：resident B 的槽位与节省。** 设 `M=8192, N=64, K=128`（K 编译期）、`BLOCK 128×64×128`、`MAD 64×64×64`、`kNumL0Stages=2`、32 核。断言 `:45`：`SHAPE_K=128 ≤ 2×64=128` ✓、`BLOCK_N=64=MAD_N` ✓。host 资格：`num_n_blocks=1` ✓、`num_m_blocks=64` → waves=2 > 1 ✓、`d.k=128 ≤ 64×2` ✓。L0B 预算：2 × (64×64×2B=8KB) = 16KB ≤ 64KB ✓。槽位：`num_k_blocks = ceil(128/128) = 1`，唯一 k_blk 内 chunk 数 = 128/64 = 2 = 环长，chunk 0→`l0b[0]`、chunk 1→`l0b[1]`，无回绕、一一对应。块内轨迹（装载块）：子块 `(m_mad=0, n_mad=0)` 搬两个 chunk；子块 `(m_mad=64, n_mad=0)` 检查 `:144`——`BLOCK_M≠MAD_M`、`BLOCK_N==MAD_N`、`BLOCK_K(128) > 64×2(128)` 不成立、`m_mad_idx≠0` → 全假，**块内免装**（普通机制）。复用块（`reuse_b=true`，如核 0 的第二个块）：`:118` 与 `:144` 都被 `!kResidentB || !reuse_b` 关掉，B 的 GM→L1（16KB）与 L1→L0B（2×8KB）全部为零。全局节省 ≈ 32 个复用块 × 32KB = 1MB；host 模型 `:352`（`b_load_blocks = min(64, 32×1) = 32`）与 `:441`（L0 字节按 1 倍而非 2 倍计）对应。

**演算四：空组走查。** 设 k-grouped、`ks = [0, 128, 0, 256]`、`kAlignment=256`、`M=N=128`、`BLOCK 64×64`、`MAD 64×64`、32 核。按 `generate_psum_layout` 的规则（`generators.py:212-220`）得 `grouped_layout = [0, 128, 256, 512]`。调度器视角：组 0 区间 `[0,0)` 空、组 1 `[0,128)` 有 2×2=4 块、组 2 `[256,256)` 空、组 3 `[256,512)` 有 4 块——全局有效块共 8 个，`current_k_block_cumsum` 在空组处不前进（`:174-175`）。核 0 的两个持久块号 0 与 32：块 0 落组 1（`k_idx_base=0`），块 32 落组 3（cumsum=4，`k_idx_base=256`）。AIC 对组 0、2 的 D 区域零写入。AIV 视角（`kNumAIVs=2`）：`num_tiles = 2×2 = 4`；`aiv_idx = block_idx×2 + sub_id` 取 0..63，仅 `aiv_idx ∈ {0,1,2,3}` 的四个 AIV 各分到一个 tile（步长 64，`:129`）；空组判定 `k_begin == k_end` 在组 0（`0==0`）与组 2（`256==256`）成立，`d.group(0/2, ...)` 的 4 个 tile 被写真值零；若本例改为 `accumulate=True`，`:175` 的条件直接不调用清零，组 0/2 的 D 保持 host/调用方放好的初值。

### 3.8 参考对比表（当天产出的参考答案）

**表一：普通路径 vs `kDirectStore`。**

| 维度 | 普通路径（`kDirectStore=false`） | 直写路径（`kDirectStore=true`） |
|---|---|---|
| 启动形态（`:27`） | `__mix__(1, 2)`：1 AIC + 2 AIV | `__mix__(1, 0)`：仅 1 AIC |
| 数据通路 | `L0C →(FIX) UB →(V) →(MTE3) GM` | `L0C →(FIX) GM`（`copy_l0c_to_gm`，`:158-159`） |
| 输出转换 | Identity 在 FIX 拷贝；带算子在 AIV 的 V 管道 | FIX 拷贝内（`quant_mode` 默认推断，`copy.hpp:293`） |
| epilogue 算子 | Identity / Alpha / Bias / 随机舍入 / FP8 量化 | 仅 Identity（`:56-57`） |
| accumulation | AIV 设原子加，bfloat/float 皆可（`epilogue.hpp:91-96`） | AIC 设 `asc_set_atomic_add_float`（`:82`），仅 FP32（`:38`） |
| gemm 类型 | Normal / M-Grouped / K-Grouped / Batched | host 仅 Normal（`heuristics.hpp:526`） |
| 对齐要求 | 任意 M/N/K（`min` 真值 + AIV 裁剪） | `d.m/d.n` 对齐、D 基址与行跨度 32B、单 wave 或单 m 块 + sector 整除（`:526-532`） |
| UB 与块内事件 | `kNumEpilogueStages` 格 + 四步借还 | 运行时完全不用（`:129` 不借、`:156` 不还、`:174` AIV 退出） |
| host 代价模型（`:493`） | `waves × (max(fix, bias) + gm + vector)` | `waves × max(fix, gm)` |
| 典型场景 | 任意，尤其带 epilogue 计算 | 单 wave 小规模，或 M ≤ block_m 的瘦 M 前向/解码，Identity、无（或 FP32）累加 |

**表二：`kResidentA` vs `kResidentB`。**

| 维度 | `kResidentA` | `kResidentB` |
|---|---|---|
| 常驻位置 | L1 单 stage，宽 `aligned(SHAPE_K, 16)`（`:59`） | L0B stage 环，跨输出块存活（`:144`） |
| 复用条件 | `reuse_a = m_idx == last_m_idx`（`:92`） | `reuse_b = n_idx == last_n_idx`（`:93`） |
| 省掉的搬运 | K 循环内 A 的 GM→L1（`:117`，`if constexpr`） | `n_idx` 不变时 B 的 GM→L1（`:118`）与 L1→L0B（`:144`） |
| 仍发生的搬运 | L1→L0A 每 k_mad 照常（`:143`，源 `l1a[0]`、偏移 `k_blk_idx*BLOCK_K + k_mad_idx`） | 复用块内无 |
| 设备约束 | Normal + 编译期 SHAPE_K（`:40`）；L1 容量（`:66`） | 另加 `SHAPE_K ≤ kNumL0Stages×MAD_K && BLOCK_N == MAD_N`（`:45`） |
| host 资格 | `d.m ≤ block_m` 且 L1 塞下 A 后 B 的 stage 数不缩水（`:552-555`） | `num_n_blocks==1 && waves>1 && block_n==mad_n && d.k ≤ mad_k×num_l0_stages`（`:556-557`） |
| 隐含正确性条件 | — | `num_k_blocks == 1`（由 host 的 mad_k/block_k 关系间接保证，3.4） |
| 典型场景 | M 小（单 m 块）、K 大 | N 小（单 n 块）、K 小、M 大多 wave |

**表三：边界与退化场景一览。**

| 场景 | 谁来处理 | 关键代码 |
|---|---|---|
| M/N 尾块非对齐 | `actual_m/n` + `eff_mad_m`（min）/`eff_mad_n`（aligned）+ AIV 真值写回 | `scheduler.hpp:129-139`、`bf16_gemm.hpp:125-126`、`epilogue.hpp:65-66, 81-82` |
| K 尾块非对齐 | `actual_k` + `eff_mad_k`（min） | `bf16_gemm.hpp:114, 137` |
| `m==0 / n==0` | host 早退，不启动 | `layout.hpp:57-58` |
| `k==0`（Normal 全空） | host `d.zero_()` / `d.copy_(c)` 后返回 | `layout.hpp:75-80` |
| K-grouped 部分组空 | AIC 调度跳过；AIV `aiv_zero_empty_groups` 写零（!acc 时） | `scheduler.hpp:167, 174-175`、`epilogue.hpp:108-141` |
| K-grouped 空组 + acc | 不清零，D 保持初值（`+= 0`） | `bf16_gemm.hpp:175`、`epilogue.hpp:111` |
| m-grouped 组尾 padding | 现状：`actual_m` 恒 BLOCK_M + 测试清零 psum 间隙（TODO） | `scheduler.hpp:130-132`、`generators.py:338-340` |
| strided D | 真实行跨度进 `outer_stride_d`；直写按 32B 检查 | `gemm.hpp:193`、`heuristics.hpp:523, 529-530` |

## 四、源码导读

建议按站点顺序阅读，总计约 40 分钟。所有行号以仓库当前版本为准。

### 站点 1：`deep_gemm/include/deep_gemm/bf16_gemm.hpp`（精读，今天的主战场）

按 3.1 的裁剪点表逐个过：`:25`（三个默认参数）、`:27`（`__mix__`）、`:37-45`（三条设备断言——说出每条拦的是什么组合）、`:47-60`（编译期/运行时 shape 合流；resident 的 L1 布局）、`:82`、`:86-93`（`epi_base` 与 `last_m/n_idx` 的初始化——为什么是 `shape_m/shape_n`）、`:105-110`、`:116-119`（注意 `:117` 是 `if constexpr` 而 `:118` 是运行时 `if`）、`:142-145`（两个搬运条件的前后两个合取项分别属于 resident 与块内 L0 复用）、`:155-163`、`:173-177`。最后通读一遍 AIC 主循环，验证"裁剪只删指令、不改协议"。

### 站点 2：直写的拷贝（`deep_gemm/include/deep_gemm/ascend/copy.hpp:287-306`）

`copy_l0c_to_gm` 全文；对照 `:266-285` 的 `copy_l0c_to_ub`——签名几乎同款（`dst` 类型与 stride 单位不同），默认 `quant_mode` 都来自 `infer_quant_mode`（`:231-264`，`NoQuant`/`F322BF16` 两个分支）。体会"同一套 FIX 转换语义、两个目的地"。

### 站点 3：空组清零（`deep_gemm/include/deep_gemm/epilogue.hpp:99-141`）

`zero_values`（`:99-106`，`__simd_vf__` 的 V 管道填零）与 `aiv_zero_empty_groups` 全文；重点核对 `:112` 的全局编号、`:119-121` 的空组判定与调度器的一致性、`:123-128` 与 `:136-139` 两对握手各自保护什么。

### 站点 4：调度器的跳过与边界（`deep_gemm/include/deep_gemm/scheduler.hpp:127-139, 141-196, 202-248`）

`get_actual_m/n`（m-grouped 的 TODO）；`get_next_block` 的 k-grouped 分支（`:167` 的 `current_shape_k > 0` 与 `:174-175` 的 cumsum 不前进）；`get_swizzled_block` 的 row-major 尾区（`:212-216`，注释"reuses A across cores"——L2 层面的复用，与 resident 的 L1/L0 复用区分）。

### 站点 5：host 闸门与代价（`csrc/jit_kernels/heuristics.hpp`）

精读 `:474-494`（`get_gemm_output_cost`，直写分支在 `:493`）与 `:521-574`（`can_direct_store` + `refine_gemm_pipeline` 的两段试穿）。浏览四处记账：`:305-306`、`:351-352`、`:405-411`（"First fill includes the full resident-A load before the K loop; it cannot overlap compute"）、`:437-441`（`num_l0a/b_loads` 与 resident B 的 1 倍字节）、`:499-505`（`kGmLoadIssueCycles`）。再核对 `:135`（`mad_k` 的上界——3.4 论证 `num_k_blocks==1` 的原料）与 `:604, 620`（`block_cap`、`min_block_k`）。

### 站点 6：host→device 的参数传递（`csrc/jit_kernels/gemm_config.hpp:110-112, 163-171` 与 `csrc/jit_kernels/bf16_gemm.hpp:39, 66`）

`GemmConfig` 的布尔字段、`std::formatter` 打印的 `direct_store={}` / `resident={}/{}/{}`（`DG_PRINT_CONFIGS` 输出的就是它）、模板实参表的最后三格。

### 站点 7：早退与零维（`csrc/utils/layout.hpp:54-86`）

`gemm_early_return`：`m/n==0` 返回、`k==0` 的 `zero_/copy_`、累加初值的提前拷贝（`:83-84`）。与 `csrc/apis/gemm.hpp:99-123` 的 shape/`c` 检查连起来看（第 11 天的延续）。

### 站点 8：测试如何驱动这些场景（浏览）

`tests/test_bf16_gemm.py:25-47`（`unaligned-k/mn`、`strided`、`kgrouped` 选择器）；`tests/generators.py:491-541`（非对齐 shape 列表）、`:606-649`（`empty`/`single` 变体与四种 acc/dtype 组合）、`:212-220`（`generate_psum_layout` 的前缀和规则）、`:361-399`（k-grouped 参考结果的空组语义）；`tests/common.py:49-84`（`gemm_dg` 对 k-grouped 的调用与累加时 `d.copy_(c)`）。

### 站点 9：交叉佐证（浏览，建立模式识别）

- `deep_gemm/include/deep_gemm/fp8_gemm.hpp:34, 49-51, 77-78, 105, 235, 268`：与 BF16 同款的直写骨架；`:124-150` 的 `kSingleMBlock` + L1 环回卷复用——**另一种**操作数复用实现，避免与 `kResidentA/B` 混淆；
- `deep_gemm/include/deep_gemm/mqa_logits.hpp:416-425`："Release accum stage on PIPE_V so it is not serialized behind the GM store"——3.2.5 遗留问题答案的出处。

## 五、实践步骤

### A. 无 Ascend 环境：纸面推演（主任务，约 55 分钟）

**A-1 对比表（必做，当天产出的核心）。** 合上本文，凭源码印象默写 3.8 的三张表，标注行号；写完对照订正。重点自查五处最易出错的细节：直写时 AIV 分支是否执行（`:174`）；直写的原子加由谁、在哪一行设置；`eff_mad_n` 在两条路径里分别进到哪里；resident A 与 resident B 各自"省掉哪一级搬运"；空组清零与 `aiv_initialize` 的先后及其保障机制。

**A-2 直写资格判定（必做）。** 对下面四个 shape 逐条评估 `can_direct_store`（`heuristics.hpp:526-532`），写出每条的通过/失败原因：

| 设定 | 预期结论 |
|---|---|
| `M=1, N=7168, K=7168`，nt，BF16 输出，不累加 | 资格通过（瘦 M：`d.m ≤ block_m` 分支） |
| `M=128, N=7168, K=7168`，nt，BF16 输出，不累加 | 资格通过（单 wave 分支；即 3.7 演算一） |
| `M=4096, N=7168, K=7168`，nt，BF16 输出，不累加 | 拒绝（两条分支都不通） |
| `M=64, N=100, K=64`，nt，BF16 输出，累加 | 双重拒绝（`N=100` 不满足 `%16`；BF16 累加不满足 FP32）——并用 3.5 的演算说明 N 这条拦住的是什么事故 |

**A-3 resident A 推演（必做）。** 用 3.7 演算二的设定：先自行算出 `l1a/l1b` 的 L1 布局与容量断言，再列出核 0 两个块的 `reuse_a` 取值与被裁剪的搬运，最后算出节省的 GM→L1 字节并与 host 模型 `a_load_blocks`（`:351`）对账。变体：把 K 改成 4096（其余不变），`l1a` 变为 64×4096×2B = 512KB——`a_bytes < L1SizeBytes` 还成立吗？host 会怎么处理（`:551` 的 `num_l1_stages=0` → `:552` 不设 resident_a）？

**A-4 resident B 推演（必做）。** 用 3.7 演算三的设定：验证 `:45` 断言与 `:144` 的四个析取项在装载块两个子块上的取值；然后回答两个思考题：(a) 若 `BLOCK_K=256`（其余不变、K 仍 128），`num_k_blocks` 与槽位映射如何变化？`d.k ≤ mad_k×num_l0_stages` 还成立吗？(b) 把 `BLOCK_M` 改成 64（= MAD_M），`:144` 的条件在 `m_mad_idx>0` 时如何取值？这会改变 resident B 的收益吗？

**A-5 空组推演（必做）。** 用 3.7 演算四的 `grouped_layout=[0,128,256,512]`：(a) 手算调度器对块号 0..7 的组归属与 `k_idx_base`；(b) 列出 `aiv_zero_empty_groups` 中 `aiv_idx=0..5` 各自写了哪些 tile；(c) 改为 `accumulate=True` 后，空组区域的最终值由什么决定？写出从 Python 侧到内核的完整链条（`common.py:56-58` → `layout.hpp:83-84` → `bf16_gemm.hpp:175`）。

**A-6 判定题自验（必做）。** 判断真假并给出行号依据（答案见第七节末尾）：

1. `kDirectStore` 时内核不再构造 `Epilogue` 对象，相关 `static_assert` 全部失效。（  ）
2. 直写路径的 `copy_l0c_to_gm` 写出的列数是 `eff_mad_n`（向上对齐后），而普通路径 AIV 写出的是 `min` 真值。（  ）
3. `kResidentA` 生效时，L1→L0A 的搬运也一并被裁剪，A 直接从 L1 进 MAD。（  ）
4. `kResidentB` 的设备断言 `SHAPE_K <= kNumL0Stages * MAD_K` 在 `SHAPE_K=0`（运行时 K）时恒真，因此必须由 host 的 `compiled_k` 闸门配合才有意义。（  ）
5. K-grouped 的空组在 `accumulate=True` 时也由 AIV 写零，以保证输出确定。（  ）
6. `reuse_a/reuse_b` 在 `kResidentA/B` 均为 false 时也会被计算，只是没有消费者。（  ）

### B. 有 Ascend 环境：观察配置与边界（可选，约 25 分钟）

```bash
cd DeepGEMM-Ascend
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py unaligned-mn --skip-prof
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py kgrouped --skip-prof
```

1. 从 `GemmConfig(...)` 打印行（`heuristics.hpp:749-753`；格式见 `gemm_config.hpp:163-171`）收集 `direct_store` / `resident` 的取值，按 shape 分组：哪些 shape 开了直写？与 A-2 的判定对照；
2. `unaligned-mn` 一组应全部 `direct_store=0`——用 `generators.py:491-500` 的 shape 列表解释为什么（`%16` 条件）；
3. `kgrouped` 一组应全部 `direct_store=0, resident=0/0/0`——对应 `can_direct_store` 的 Normal 条件与 `:40` 断言；同时观察 `empty`/`single` 变体的正确性验证过程；
4. 选一个开了直写的 shape 去掉 `--skip-prof`，记录 TFLOPS 作为第 51 天基线实验的素材。

## 六、当天产出

**普通路径与特殊路径对比表**。把 A-1～A-5 的结果整理成一页笔记，至少包含：

1. **表一～表三**（3.8 的三张表，自己默写后订正）；
2. **直写资格判定记录**：A-2 四个 shape 的逐条结论；
3. **两条演算图**：resident A 的 L1 布局图（标注 `l1a` 单 stage 与 `l1b` 环的 offset）、resident B 的 L0B 槽位映射图（chunk j → `l0b[j]`，跨块不变）；
4. **空组链条**：A-5(c) 的"Python 初值 → host 拷贝 → 调度跳过 → 不清零"链条；
5. **遗留问题清账**：把第 25 天的"归还 arrive 前移"条目改写为 3.2.5 的结论；新增一条自查："设备断言 `:45` 没有检查 `num_k_blocks == 1`，这条正确性前提由谁、在哪一行保证？"（答案在 3.4，能独立复述才算过关）；
6. **给第 27 天复盘的问题清单**：本周从 K 循环到特殊路径至少积累三个"讲不清楚就是没懂"的问题，例如"`eff_mad_n` 为什么对齐而 `eff_mad_m` 不对齐""直写删掉 UB 后 L0C stage 的复用靠什么保障"。

## 七、自测题

先闭卷作答，再对照源码与本文订正。

1. `kDirectStore` 分别裁掉了第 25 天四步借还协议的哪几步？剩下未裁的是什么？`s_epi` 还会被计算吗？
2. `can_direct_store` 的条件里，哪些有设备侧断言镜像、哪些纯靠 host？各举两例并给行号。
3. 为什么直写要求 `d.n % 16 == 0` 而普通路径能处理 `N=100`？从 `eff_mad_n` 的计算（`:126`）与两条路径的写回参数（`:158-159` vs `epilogue.hpp:81-82`）各写一句话。
4. 写出 `kResidentA` 的四处代码改动（行号 + 一句话）。它要求 `d.m <= block_m`（host），这和 `reuse_a = m_idx == last_m_idx` 是什么关系？
5. `kResidentB` 的设备断言内容是什么？为什么跨块复用还需要 `num_k_blocks == 1`，而这条件由 host 的哪两行代码间接保证？
6. K-grouped 空组：AIC 侧哪一行让空组不产生任何块？AIV 侧清零的 tile 粒度、分摊步长、行列真值分别是什么？`accumulate=True` 时空组的输出值从哪里来？
7. `m==0`、`n==0`、`k==0` 分别在哪里被处理、各做什么？K-grouped 全部组为空时走哪条路？

### 参考答案

1. 裁掉"借"（`:129` 的 `aic_wait_free` 不执行）与"还"（`:161` 的 `aic_store` 换成 `:158-159` 的 `copy_l0c_to_gm`，AIC 自写 GM）；AIV 侧 `:174` 直接 return，"处理 + 归还"两步不存在。未裁的是 K/M/MTE1/MTE2 流水线事件（`:116/:119/:121/:148/:152/:167`）——裁剪只删指令、不改协议。`s_epi`/`epi_base` 仍按 `:122/:164/:169` 计算与推进，只是无消费者。
2. 有镜像的：Identity（host `heuristics.hpp:526` ↔ 设备 `:56-57`）、FP32 累加（host `:528` ↔ 设备 `:38`）。纯 host 的：`d.n % align_n`、D 基址/行跨度 32B 对齐、单 wave 或单 m 块 + sector 整除（`heuristics.hpp:528-532`）——设备路径没有越界检查，这些是"保证不出事"的前提条件。
3. `eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N), 16)`（`:126`）为满足 FIX 出口 32B 行对齐把尾块列数向上补到 16；直写把 `eff_mad_n` 原样作为 `copy_l0c_to_gm` 的 `n_elems`（`:158-159`），写出宽度可能超过真值；普通路径 AIV 用 `MadBlock.num_cols = min(...)` 真值（`epilogue.hpp:66, 81-82`）把填充列裁在 UB 里。`N=100` 时直写会写出 48 列踩进下一行（3.5 演算），故被 host 拒绝。
4. `:59`——`l1a` 宽度换成 `aligned(SHAPE_K, 16)`、单 stage，`l1b` 起点 `l1a.offset(1)`；`:109-110`——块首 `!reuse_a` 时一次搬 `actual_m × shape_k` 整条 K；`:117`——`if constexpr (!kResidentA)` 裁掉 K 循环内 A 的 GM→L1；`:143`——L1→L0A 源固定 `l1a[0]`、K 偏移加 `k_blk_idx * BLOCK_K`。`d.m <= block_m` 使 `num_m_blocks == 1`，所有块 `m_idx == 0`，故除每核首块外 `reuse_a` 恒真——host 条件让复用率成为 100%。
5. `SHAPE_K <= kNumL0Stages * MAD_K && BLOCK_N == MAD_N`（`:45`）。`s_ab` 每个子块从 0 重置（`:133`），第 j 个 K chunk 恒落 `l0b[j % 环长]`；多个 k_blk 时不同 k_blk 的同位 chunk 互相覆盖槽位，装载块结束时槽里是最后一个 k_blk 的数据，复用块读到的却是错的——所以必须单 k_blk。由 `heuristics.hpp:556-557` 的 `d.k <= mad_k * num_l0_stages` 与 `:135` 的 `mad_k <= max(k_alignment, block_k/num_l0_stages)`（BF16 下即 `mad_k ≤ block_k/2`）合起来蕴含 `d.k ≤ block_k`，即 `num_k_blocks == 1`。
6. AIC 侧：`scheduler.hpp:167` 的 `current_shape_k > 0` 闸门（空组连一个 (m,n) 块都不产出，`:174-175` 的 cumsum 也不前进）。AIV 侧：MAD 粒度 tile（`num_tiles = ceil(shape_m/MAD_M) × ceil(shape_n/MAD_N)`，`epilogue.hpp:113-114`），按 `kNumCores * kNumAIVs` 步长分摊（`:129`），写出行列取 `min` 真值（`:132-133`）。累加时不清零（`bf16_gemm.hpp:175` 的 `!kWithAccumulation`；`epilogue.hpp:111` 断言），空组 D 保持初值——初值来自调用方 `d.copy_(c)`（`common.py:56-58`）或 host 的提前拷贝（`layout.hpp:83-84`）。
7. `m==0 || n==0`：`gemm_early_return` 直接返回不启动（`layout.hpp:57-58`）；`k==0`：host `d.zero_()`（或先 `d.copy_(c)`）+ `sfd->zero_()` 后返回（`:75-80`）。K-grouped 全空时 total k 为 0，同样走 `k==0` 早退——不进内核。

**A-6 判定题答案**：

1. **假**。`epilogue` 仍构造（`:53-55`），`epilogue.hpp:28-35` 的断言照常参与编译；只是运行时无人调用其方法。
2. **真**。`:158-159` 传 `eff_mad_n`；`epilogue.hpp:81-82` 用 `block.num_cols`（`:66` 的 `min` 真值）。
3. **假**。L1→L0A 照常（`:143`），只是源 stage 固定 0、偏移加 `k_blk_idx * BLOCK_K`；resident A 复用的是 L1 级。
4. **真**。`SHAPE_K=0` 时 `0 <= x` 恒真（`:45`），断言失效；`heuristics.hpp:546` 的 `compiled_k` 闸门（配合 `:40` 的 `SHAPE_K` 断言）把运行时 K 的 resident 挡在门外。
5. **假**。累加时不调用 `aiv_zero_empty_groups`（`:175`），空组 D 保持初值（`D += 0`）；清零反而会破坏累加语义（`epilogue.hpp:111` 的断言动机）。
6. **真**。`:92-93` 无条件计算（`:87` 的初始化保证首块不复用），非 resident 配置下没有 `if` 消费它们。

---

**明日预告（第 27 天）**：本周收官复盘。不看源码，独立画出"任务调度 → GM/L1/L0 数据搬运 → MAD 累加 → AIC/AIV Epilogue → 写回 GM"的完整链路，并把今天的特殊路径作为分支标注在图上（直写旁路、resident 裁剪点、空组兜底）；然后对照 `bf16_gemm.hpp` 修正，记录仍讲不清楚的三个问题。通过标准：能讲清楚一个输出 Tile 从调度到写回的完整生命周期——包括它走的是普通路径还是某条特殊路径。
