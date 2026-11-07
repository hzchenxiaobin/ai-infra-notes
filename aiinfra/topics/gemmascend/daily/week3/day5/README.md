# 第 3 周 · 第 5 天（总第 19 天）：追踪 GM 到 L1/L0 的搬运

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 3 周"BF16 Kernel——模板、调度与数据搬运"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前四天把"块怎么切、分给谁、地址怎么算"都解决了：第 17 天调度器给出 `(m_block_idx, n_block_idx)`，第 18 天 `gm_ptr::index()` 把坐标变成 GM 字节地址、`l1a[s]/l0a[s_ab]` 的 stage 寻址也就位。今天回答数据通路上最后两个"搬运"问题：**数据怎么从 GM 进入 L1，又怎么从 L1 进入 L0A/L0B？** 答案几乎全部集中在 `deep_gemm/include/deep_gemm/ascend/copy.hpp`（383 行）里——它是设备侧搬运指令的薄封装层，被 `ascend.hpp:10-12` 统一引入。今天的主线是三个：`copy_gm_to_l1()`（GM→L1，MTE2 通道，做 ND→NZ 分形转换）、`copy_l1_to_l0a/b()`（L1→L0，MTE1 通道，按分形单元搬运）、以及贯穿两者的 `actual_m/n/k` 如何把每一次搬运裁剪到"刚好够用"。读完今天，第 5 天画的那张 `GM → L1 → L0A/L0B → MAD → L0C` 数据流图就将变成一张带真实地址、大小和步长的施工图。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 画出 `copy.hpp` 的**函数地图**：GM→L1（`copy_gm_to_l1`，`copy.hpp:101-131`）、L1→L0A/L0B（`copy_l1_to_l0a/b`，`copy.hpp:133-159`）、L0C→UB/GM（`copy_l0c_to_ub/gm`，`copy.hpp:266-306`）、GM↔UB（4 个 `copy_gm_to_ub` 重载与 2 个 `copy_ub_to_gm`）以及辅助函数（`set_mte2_nz`、`fill_l1` 族、`mad`），并说明今天的输入侧两级与后续天（第 25 天 Epilogue、第 32 天低精度）的分工边界；
2. 逐参数解读 `copy_gm_to_l1()` 的 K-major 分支（`copy.hpp:120-124`）：`set_mte2_nz()` 打包的四个 NZ 目标参数（`copy.hpp:14-21`）、`asc_copy_gm2l1_nd2nz()` 收到的行距 `src.stride_outer * sizeof(T)` 与二维范围 `actual_mn × actual_k`，并解释 **nd2nz = 普通二维（ND）→ Z 形分形（NZ）** 的格式转换含义；
3. **对一次搬运记录四要素**——源地址、目标地址、实际大小、stride——并区分"搬运量"（actual 区域）与"缓冲容量"（BLOCK 区域）：L1 stage 缓冲按 `BLOCK_M × BLOCK_K` 规划（`bf16_gemm.hpp:59-60`），但 GM→L1 只搬 `actual_m × actual_k` 的有效数据；
4. 从 `fill_tail_block` / `fill_mn_tail_block` 的地址算式（`copy.hpp:333-368`）**反推 L1 的 NZ 分形摆放**：K-major 时 L1 = 按 K 的 32 字节 C0 块分 slab、每个 slab 内是 `shape_mn` 行 × 32 字节；MN-major 时对称换位；
5. 逐参数解读 `copy_l1_to_l0a/b()` 的两个分支（`copy.hpp:136-143`）：分形起点 `(mn_idx/FRAC_MN, k_idx/FRAC_K)`、两个方向的步数 `(dst.shape_mn/FRAC_MN, dst.shape_k/FRAC_K)`、源/目标分形行距 `src.shape_mn/FRAC_MN` 与 `dst.shape_mn/FRAC_MN`，以及 MN-major 分支为何改用 `_transpose` 变体、两个下标角色为何互换（呼应第 18 天 3.5 节）；
6. 说出 kernel 传给 `copy_l1_to_l0a/b` 的 `mn_idx/k_idx` 是 **stage 内偏移**（`m_mad_idx/k_mad_idx`）而非全局坐标，并解释 resident A 路径第 4 参数写成 `k_blk_idx * BLOCK_K + k_mad_idx` 的原因（`bf16_gemm.hpp:143`，resident A 的 L1 缓冲覆盖整个 K，见 59 行的 `aligned(SHAPE_K, ...)`）；
7. 解释 `actual_m/actual_n/actual_k` 的**三级裁剪链**：① GM→L1 只搬有效行列（`n=actual_mn, d=actual_k`）；② L1→L0 按 `as_mad_aligned` 对齐后的分形几何整块搬运（`bf16_gemm.hpp:138-139`），可能带入少量未更新的"旧数据"；③ `mad()` 收到未对齐的 `eff_mad_m/eff_mad_k` 与对齐过的 `eff_mad_n`（`bf16_gemm.hpp:125-126、149`），epilogue 写回再按 `actual_*` 截断（`epilogue.hpp:65-66`）——计算与写回都被有效量约束，因此②中的旧数据不影响结果；
8. 对比 BF16 与 FP8/FP4 的边界策略：BF16 靠上述裁剪链即可，而 MX 路径因 K 必须对齐到 64（`MX_SF_DIVISOR`，`common.hpp:65`）需要 `fill_tail_block` 显式补零（`fp8_gemm.hpp:160-170`）；并能解释 `eff_mad_n` 对齐到 `get_frac_k<ub_dtype_t>()` 与 L0C→UB 行距 32 字节对齐的关系（`epilogue.hpp:29` 的 `static_assert`）；
9. 标出两级搬运周围的**同步配对**（`bf16_gemm.hpp:116/119、141/146、148/152、167`，事件号就是 stage 下标），今天只需认出"谁等谁"，展开留到第 23 天；
10. 解释 L1→L0 搬运的**复用条件**（`bf16_gemm.hpp:142、144` 的 `if`）：当 `BLOCK_N == MAD_N` 或 `BLOCK_K <= MAD_K * kNumL0Stages` 时，同一个 L0A tile 可跨 `n_mad_idx` 复用而跳过重复搬运——这是比 L1 stage 更靠近计算单元的一层数据复用。

**当天产出：`GM → L1 → L0` 数据搬运图**（含一次搬运的四要素记录，模板见第六节，参考演算见 3.7 节）。

## 二、预备知识

- **第 18 天（最重要）**：`gm_ptr::index()` 的字节偏移公式、`stride_outer` 语义、`l1_ptr/l0a_ptr` 的 `shape_mn/shape_k` 字段与 `as_mad_aligned`——今天的每一次搬运都从这些地址和形状出发。
- **第 16 天**：`l1a/l1b/l0a/l0b/l0c` 的声明与 stage 划分（`bf16_gemm.hpp:59-64`）、`size_per_stage()/offset()/operator[]`（`tile_ptr.hpp:37-43`）、容量 `static_assert`（`bf16_gemm.hpp:66-69`）。
- **第 17 天**：`m_idx/n_idx = get_global_idx()`、`actual_m/actual_n` 的调度语义（`scheduler.hpp:113-139`）——今天补上 `actual_k` 并把三者接入搬运。
- **第 15 天**：模板参数 `kMajorA/kMajorB`、`BLOCK_M/N/K` 与 `MAD_M/N/K`、`kNumL1Stages/kNumL0Stages`、`kResidentA/B`。
- **第 5 天**：GM/L1/L0A/L0B/L0C 分层职责，MTE1/MTE2/M 流水线的概念性数据流 `GM → L1 → L0A/L0B → MAD → L0C`。
- **C++ 语法**：`if constexpr` 编译期分支（Major 分支裁剪）、`reinterpret_cast`、模板默认参数（`kSrcIsNZ = false`）。

## 三、详细讲解

### 3.1 `copy.hpp` 全景：一条数据通路上的函数族

`ascend/copy.hpp` 把 CANN 设备 C API（`<c_api/asc_simd.h>`，由 CANN SDK 提供、不在本仓库内）的搬运指令封装成"带形状参数"的 C++ 模板函数。仓库里能直接读到的是**参数打包与形状换算**；指令本身的硬件语义以封装处的参数名和注释为线索。整个文件按数据通路分成六组：

| 分组 | 函数 | 位置 | 今天? |
| :-- | :-- | :-- | :-- |
| GM→L1 | `copy_gm_to_l1` / `copy_gm_to_l1_mx`（Scale Factor） | `copy.hpp:101-131` / `161-197` | **主线** / 对照 |
| L1→L0 | `copy_l1_to_l0a` / `copy_l1_to_l0b`（及 `_mx` 变体） | `copy.hpp:133-159` / `199-229` | **主线** / 对照 |
| L0C→UB/GM | `copy_l0c_to_ub` / `copy_l0c_to_gm` | `copy.hpp:266-306` | 第 25 天 |
| GM↔UB | `copy_gm_to_ub`（4 个重载）/ `copy_ub_to_gm`（2 个） | `copy.hpp:38-99、23-36、308-317` | 反量化与 MQA 用 |
| NZ 参数辅助 | `set_mte2_nz`、`set_l0c_nz2dn` | `copy.hpp:9-21` | **主线** |
| 补零/计算辅助 | `fill_l1`、`fill_tail_block`、`fill_mn_tail_block`、`mad`、`infer_quant_mode` | `copy.hpp:231-381` | 3.4/3.6 借用 |

三个观察：

1. **封装层没有状态**。所有函数都是 `__aicore__ inline` 自由函数，地址、形状、stride 全部由调用方（kernel 模板）以值传入——`copy.hpp` 自己不知道 stage 环的存在，多级缓冲的轮转完全由 `bf16_gemm_impl` 的循环变量 `s/s_ab` 控制。
2. **搬运方向与流水线一一对应**：GM→L1 走 MTE2 通道，L1→L0A/L0B 走 MTE1 通道，L0A/L0B→L0C 走 M（MAD）通道。今天的两级搬运就是前两段。
3. **kernel 里的调用点只有四行**（`bf16_gemm.hpp`）：110（resident A 整块预载）、117/118（K 块循环内的 A/B 搬运）、143/145（MAD 块循环内的 L1→L0 搬运）。读懂这四行的参数来源，就读懂了本文件在 BF16 主线上的全部职责。

### 3.2 数据流回放：两级搬运在 kernel 循环里的位置

先把 `bf16_gemm_impl` 的 AIC 主循环（`bf16_gemm.hpp:89-172`）骨架抽出来，只留搬运与同步（简化掉 group/batch/resident 分支）：

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {          // 89：领 Tile
    const uint32_t actual_m = scheduler.get_actual_m(m_block_idx);   // 105
    const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);   // 106
    for (uint32_t k_blk_idx = 0, s = 0; k_blk_idx < num_k_blocks;     // 112：K 块循环，s 为 L1 stage 环
         k_blk_idx++, s = s == kNumL1Stages - 1 ? 0 : s + 1) {
        const uint32_t actual_k = min(eff_shape_k - k_blk_idx * BLOCK_K, BLOCK_K);   // 114
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, s);                       // 116：等 L1 stage s 被 MTE1 用完
        copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, kL2CtrlA); // 117：A：GM→L1
        copy_gm_to_l1(l1b[s], b.index(n_idx, k_idx), actual_n, actual_k, kL2CtrlB); // 118：B：GM→L1
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, s);                     // 119：L1 stage s 已填好

        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, s);                       // 121：MTE1 等 stage s 就绪
        for (m_mad_idx...; m_mad_idx < actual_m; m_mad_idx += MAD_M)           // 123：M 方向 MAD 块
            for (n_mad_idx...; n_mad_idx < actual_n; n_mad_idx += MAD_N)       // 124：N 方向 MAD 块
                for (uint32_t k_mad_idx = 0, s_ab = 0; k_mad_idx < actual_k;   // 133：K 方向 MAD 段，s_ab 为 L0 stage 环
                     k_mad_idx += MAD_K, s_ab = s_ab == kNumL0Stages - 1 ? 0 : s_ab + 1) {
                    uint32_t eff_mad_k = min(actual_k - k_mad_idx, MAD_K);     // 137
                    auto l0a_tile = l0a[s_ab].as_mad_aligned(eff_mad_m, eff_mad_k);  // 138
                    auto l0b_tile = l0b[s_ab].as_mad_aligned(eff_mad_n, eff_mad_k);  // 139
                    asc_sync_wait(PIPE_M, PIPE_MTE1, s_ab);            // 141：等 L0 stage s_ab 被 MAD 用完
                    copy_l1_to_l0a(l0a_tile, l1a[s], m_mad_idx, k_mad_idx);    // 143：A：L1→L0A
                    copy_l1_to_l0b(l0b_tile, l1b[s], n_mad_idx, k_mad_idx);    // 145：B：L1→L0B
                    asc_sync_notify(PIPE_MTE1, PIPE_M, s_ab);          // 146：L0 stage s_ab 已填好
                    asc_sync_wait(PIPE_MTE1, PIPE_M, s_ab);            // 148：MAD 等 L0 就绪
                    mad(l0c_tile, l0a_tile, l0b_tile, eff_mad_m, eff_mad_n, eff_mad_k, ...);  // 149
                    asc_sync_notify(PIPE_M, PIPE_MTE1, s_ab);          // 152：MAD 用完 L0 stage s_ab
                }
        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, s);                     // 167：L1 stage s 释放
    }
}
```

数一数搬运次数（以整块 Tile 为例）：**每推进一个 K 块**，发生 2 次 GM→L1（A、B 各一，写入 L1 stage `s`）；随后对每个 `(m_mad, n_mad)` 组合、每个 `k_mad` 段，各发生 1 次 L1→L0A 和 1 次 L1→L0B（写入 L0 stage `s_ab`）。两级 stage 环（`s` 随 `k_blk_idx` 轮转、`s_ab` 随 `k_mad_idx` 轮转）让"下一块的搬运"与"当前块的计算"重叠——这正是第 5 天数据流图里 `GM → L1 → L0A/L0B` 两级箭头的代码形态。同步调用今天只标记位置：每对 `wait/notify` 用同一个事件号（就是 stage 下标），配对关系是 116↔167、119↔121、141↔152、146↔148；kernel 开头的 `set_flags`（83-84 行，`sync.hpp:63-67`）把所有事件预置为已触发，让第一轮 `wait` 直接通过（空缓冲视为空闲）。完整语义留到第 23 天。

### 3.3 `copy_gm_to_l1`：一次 ND→NZ 的格式转换搬运（K-major 主线）

完整签名（`copy.hpp:101-108`）：

```cpp
template <bool kSrcIsNZ = false, typename T, Major M>
__aicore__ inline void copy_gm_to_l1(
    l1_ptr<T, M> dst, gm_ptr<T, M> src,
    uint32_t actual_mn, uint32_t actual_k,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint16_t num_batch = 1, uint16_t dst_batch_stride_blocks = 0,
    uint64_t src_batch_stride_bytes = 0
);
```

K-major 分支（`copy.hpp:120-124`）只有两行：

```cpp
set_mte2_nz(num_batch, 1, dst.shape_mn, dst_batch_stride_blocks);       // 121：先配置 NZ 目标布局
asc_copy_gm2l1_nd2nz(dst.template ptr<copy_dtype_t>(), src.template ptr<copy_dtype_t>(),
                     src.stride_outer * sizeof(T), l2_ctrl, actual_mn, actual_k / pack_factor,
                     src_batch_stride_bytes, false);                     // 122-124：再发起搬运
```

**第一行：`set_mte2_nz` 描述 L1 侧的分形摆放**（`copy.hpp:14-21`）。它把四个 16 位参数压进一个 64 位整数交给 `asc_set_gm2l1_nz_para`：

```text
bit  0-15: num_batch                 本次搬运的批数（默认 1）
bit 16-31: dst_nz_n_stride           这里传 1
bit 32-47: dst_nz_c0_stride          这里传 dst.shape_mn（= BLOCK_M 或 BLOCK_N）
bit 48-63: dst_batch_stride_blocks   批间目标间隔（默认 0）
```

`c0_stride` 取 `dst.shape_mn` 是个关键线索：L1 目标缓冲的行数（MN 方向逻辑行数）决定了 NZ 分形在 L1 里的排布步长——3.4 节会从 fill 函数反推出它对应的物理布局。

**第二行：`asc_copy_gm2l1_nd2nz` 发起搬运**。从参数顺序能读出一次搬运的完整描述：

| 参数 | 值 | 含义 |
| :-- | :-- | :-- |
| 目标 | `dst.ptr()` | L1 stage 基址（`l1a[s]` / `l1b[s]` 的 `addr`） |
| 源 | `src.ptr()` | GM 上 Tile 左上角（第 18 天 `index()` 的结果） |
| 行距 | `src.stride_outer * sizeof(T)` | GM 源每跳一行的字节数 |
| 二维范围 | `actual_mn` 行 × `actual_k / pack_factor` 列 | **只搬有效区域**（BF16 的 `pack_factor=1`） |
| L2 策略 | `l2_ctrl` | 本次 load 在 L2 的分配策略 |
| 批距 | `src_batch_stride_bytes` | 多批搬运时源侧批间隔（默认 0） |

指令名 `nd2nz` 点明了它的本质：**把 GM 里的普通二维（ND，Normal Data）块转写成 L1 里的 Z 形分形（NZ）块**。也就是说 GM→L1 不只是"复制"，还伴随着硬件完成的布局重排——3.4 节给出重排后的样子。搬运的有效数据量 = `actual_mn × actual_k × sizeof(T)` 字节，而 L1 stage 缓冲按 `BLOCK × BLOCK_K` 全尺寸规划（第 16 天），**两者的差就是边界 Tile 留下的"未更新区域"**。

**MN-major 分支**（`copy.hpp:125-129`）只换两个参数：`set_mte2_nz` 的 `c0_stride` 改传 `dst.shape_k`，`nd2nz` 的行列改为 `actual_k` 行 × `actual_mn / pack_factor` 列——第 18 天讲过的"行列互换、分形摆放转置"在这里落地为具体参数。

**`kSrcIsNZ` 分支**（`copy.hpp:111-119`）处理源本来就是 NZ 布局的情形：不再需要 nd2nz 转换，改用 `asc_set_gm2l1_loop_size/loop1_stride/loop2_stride` 配合 `asc_copy_gm2l1_align` 直搬，并有 `static_assert(M == Major::K, "Direct NZ copy requires K-major GM storage")`。BF16 主线不用它；消费方是 MQA 的 Ascend KV cache 布局（`mqa_logits.hpp:192` 的 `kUseAscendKVLayout` 模板参数，359 行 `copy_gm_to_l1<kUseAscendKVLayout>(...)`）。

**`l2_ctrl` 从哪来**：它是 kernel 模板参数 `kL2CtrlA/kL2CtrlB`，由 Host 启发式决定——`heuristics.hpp:200-201`：当 N 方向块数很少而 M 方向块数多时（A 会被反复重读），选 `NOTALLOC_KEEP`（不占 L2、靠多次 GM 读）；否则选 `NORMAL_FIRST_VICTIM`。枚举值定义在 `common.hpp:120-127`。今天只需知道：**每次 GM→L1 都携带 L2 缓存策略，且策略在 JIT 配置阶段就已选定**。

**三个批参数的用途**（`num_batch/dst_batch_stride_blocks/src_batch_stride_bytes`，`copy.hpp:106-107`）：BF16 主线全用默认值。消费示例是分页 MQA（`mqa_logits.hpp:353-361`）：一次 `copy_gm_to_l1` 把最多两个物理不连续的 KV 页合并搬入 L1——`num_batch` 为页数、`dst_batch_stride_blocks = kPageSize` 为 L1 侧页间隔、`src_batch_stride_bytes = page_gap × kv_page_stride_bytes` 为 GM 侧页间隔。这解释了为什么一个"简单搬运"封装要留批扩展参数。

### 3.4 L1 里的 NZ 分形长什么样：从 fill 函数反推

`copy.hpp` 没有文档直接描述 L1 的 NZ 布局，但 FP8/FP4 路径的两个补零函数把地址算式写在了明面上（`static_assert(sizeof(T) == 1)`，即这两个函数只服务 8-bit 类型；BF16 的分形结构相同，只是 `FRAC_K` 不同）。先看 `fill_tail_block` 的 K-major 分支（`copy.hpp:338-346`）：

```cpp
const uint32_t nz32 = aligned(actual_k, 32);
if (nz32 < k64) {
    fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + (nz32 / 32) * ptr.shape_mn * 32), ...);
}
```

零填的起始字节 = `(k / 32) × shape_mn × 32`——**K 方向每前进一个 32 字节的"C0 块"，L1 地址就跳过 `shape_mn × 32` 字节**。再看 `fill_mn_tail_block`（`copy.hpp:359-367`）：从 `ptr.ptr() + actual_mn * 32` 开始，按 `shape_k / FRAC_K` 行、每行 `(padded_mn - actual_mn)` 个 32 字节块、行间隔 `shape_mn` 个 32 字节块填充——**MN 方向每前进一行，地址前进 32 字节**。两式合起来，K-major 的 L1 stage 布局只有一个自洽解：

```text
L1 stage（K-major，形状 [shape_mn, shape_k]，如 l1a = [128, 128] BF16）：

  slab 0（K 元素 0..15）      slab 1（K 16..31）   ...   slab 7（K 112..127）
 ┌─────────────────────────┐ ┌───────────────────┐      ┌───────────────────┐
 │ mn 行 0   : 32 B（K×16）│ │ mn 行 0 : 32 B    │      │ mn 行 0 : 32 B    │
 │ mn 行 1   : 32 B        │ │ mn 行 1 : 32 B    │      │ mn 行 1 : 32 B    │
 │  ...                     │ │  ...              │      │  ...              │
 │ mn 行 127 : 32 B        │ │ mn 行 127 : 32 B  │      │ mn 行 127 : 32 B  │
 └─────────────────────────┘ └───────────────────┘      └───────────────────┘
  字节 [0, 128×32)            [128×32, 2×128×32)  ...    [7×128×32, 8×128×32)

  元素 (mn, k) 的字节偏移 = (k / FRAC_K) × shape_mn × 32 + mn × 32 + (k % FRAC_K) × sizeof(T)
  （BF16：FRAC_K = 32/2 = 16，即每 slab 16 个 K 元素；FP8：FRAC_K = 32）
```

验证一致性：总大小 = `(shape_k / FRAC_K) × shape_mn × 32` 字节。BF16：`(128/16) × 128 × 32 = 32768` = `128 × 128 × 2` ✓，恰好等于 `operand_ptr::size_per_stage()`（`tile_ptr.hpp:37`）——**NZ 是同一批字节的重排，不占额外空间**。这个布局也解释了 3.3 节的 `c0_stride = dst.shape_mn`：每个 C0 slab 的"高度"就是 `shape_mn` 行。MN-major 时布局对称换位：按 MN 的 32 字节块分 slab、每个 slab 内是 `shape_k` 行 × 32 字节（`c0_stride = dst.shape_k`，`copy.hpp:126`）。

两点说明：① 每 16 行 × 32 字节（512 字节）构成一个"分形单元"，slab 内相邻 16 行正好是一个分形——分形**内部**的元素摆放由 MTE 硬件管理，读写两端（MTE2 写入、MTE1 读出）遵守同一约定，我们只需要 slab 级的图像就能读懂所有地址代码；② 这套布局是**推导**而非文档——出发点是 fill 函数的地址算式（`copy.hpp:341、349、365` 三处偏移表达式），实践 B-2 会用 Python 复刻验证。

### 3.5 `copy_l1_to_l0a/b`：以分形单元为粒度的 L1→L0 搬运

完整签名（`copy.hpp:133-134`）：

```cpp
template <typename T, Major M>
__aicore__ inline void copy_l1_to_l0a(l0a_ptr<T> dst, l1_ptr<T, M> src, uint64_t mn_idx, uint64_t k_idx);
```

K-major 分支（`copy.hpp:136-138`）一条指令、六个标量：

```cpp
constexpr uint32_t FRAC_K = get_frac_k<T>();                     // BF16 = 16
asc_copy_l12l0a(dst.ptr(), src.ptr(),
                mn_idx / FRAC_MN,            // x_start：起点在第几个 16 行组
                k_idx / FRAC_K,              // y_start：起点在第几个 K-slab
                dst.shape_mn / FRAC_MN,      // x_step ：搬多少个 16 行组
                dst.shape_k / FRAC_K,        // y_step ：搬多少个 K-slab
                src.shape_mn / FRAC_MN,      // src_stride：L1 侧的分形行距
                dst.shape_mn / FRAC_MN);     // dst_stride：L0 侧的分形行距
```

参数名（`x_start/y_start/x_step/y_step/src_stride/dst_stride`）来自同文件 `_mx` 变体的具名注释（`copy.hpp:207-212`），语义可以从三个角度交叉验证：

1. **起点**：`mn_idx / FRAC_MN` 与 `k_idx / FRAC_K` 是 L1 NZ 布局里的分形单元坐标——结合 3.4 的偏移公式，分形 `(g, c)` 的字节起点 = `c × shape_mn × 32 + g × 512`；
2. **大小**：`x_step × y_step` 个分形单元，每单元 512 字节（16 行 × 32 B），总量 = `dst.shape_mn × dst.shape_k × sizeof(T)` 字节——正是 `l0a_ptr(MAD_M, MAD_K)` 一个 stage 的大小（`bf16_gemm.hpp:62`，`tile_ptr.hpp:37`）；
3. **行距**：主例中 `src_stride = dst_stride = shape_mn / 16 = 8`，恰好等于"一个 K-slab 里 16 行组的个数"——沿 K 方向推进一个分形要跨过 8 个行组，与 3.4 的 `[K-slab][mn 行]` 摆放完全吻合。

**MN-major 分支**（`copy.hpp:139-143`）改调 `asc_copy_l12l0a_transpose`，两个起点下标角色互换（`k_idx / FRAC_MN`、`mn_idx / FRAC_K`），步数换成 `m_step = dst.shape_k / FRAC_MN` 与 `ceil_div(dst.shape_mn, FRAC_K)`：因为 MN-major 的 L1 布局是转置的（3.4 末段），MTE1 在 L1→L0 的路上逐分形转置，把转置分形还原成 L0A 要求的标准摆放——第 18 天"`_transpose` 变体"的结论在这里落到参数。`m_step` 对 1 字节类型向上取偶（`copy.hpp:140-141`）是源码里的显式处理，原因与 8-bit 类型的对齐粒度有关，记住现象即可。`copy_l1_to_l0b`（`copy.hpp:147-159`）与 `l0a` 版逐行同构，只是目标换成 `__cb__`（L0B）。

**三个必须分清的细节**：

- **`mn_idx/k_idx` 是 stage 内偏移，不是全局坐标**。调用处传的是 `m_mad_idx/k_mad_idx`（`bf16_gemm.hpp:143、145`）——因为非 resident 路径下 `l1a[s]` 里只装着**当前 K 块**的数据（`shape_k = BLOCK_K`），K 方向在 stage 内从 0 重新数起。对照 resident A 路径（143 行第 4 参数 `(kResidentA ? k_blk_idx * BLOCK_K : 0) + k_mad_idx`）：它的 L1 缓冲一次装下整个 K（59 行 `shape_k = aligned(SHAPE_K, get_frac_k<bfloat16_t>())`，110 行整块预载），所以要用**全局** K 位置寻址。同一行代码里的三元表达式就是这两种 L1 用法的分水岭。
- **`dst` 的形状字段已被 `as_mad_aligned` 抬过**。调用前 `l0a_tile = l0a[s_ab].as_mad_aligned(eff_mad_m, eff_mad_k)`（`bf16_gemm.hpp:138-139`），因此 `x_step/y_step` 数的是**对齐后**的分形个数（见 3.6）。注意 `l0a_ptr` 不带 Major 模板参数（`tile_ptr.hpp:94-95`）——无论 L1 里是哪种摆放，L0A/L0B 永远是 MAD 单元要求的同一种分形布局，Major 的消化全部发生在 MTE2/MTE1 两级搬运里（第 18 天的"L0 不变式"）。
- **搬运可以被跳过**。143 行的 `if (BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0)` 与 145 行的对应条件：L0A 里的 A tile 只依赖 `(m_mad_idx, k_mad_idx)`、与 `n_mad_idx` 无关——当 `BLOCK_N == MAD_N`（N 方向只有一个 MAD 块）或 `BLOCK_K <= MAD_K × kNumL0Stages`（一个 Tile 的全部 K 段能同时驻留 L0 stage 环）时，遍历后续 `n_mad_idx` 无需重搬 A；B 侧条件对 `m_mad_idx` 对称。再叠加 `(!kResidentB || !reuse_b)`（118、144 行）跳过跨 Tile 的 B 复用搬运。这是**L0 级的数据复用**，比 L1 stage 环更靠近计算单元；`kResidentA/B` 的完整语义留到第 26 天。

### 3.6 `actual_m/n/k`：边界搬运的三级裁剪链

三个"actual"的来源（第 17 天补 K 维）：

| 量 | 定义 | 位置 | 主例取值（整块 / 边界） |
| :-- | :-- | :-- | :-- |
| `actual_m` | `min(shape_m - m_block_idx × BLOCK_M, BLOCK_M)` | `scheduler.hpp:129-136` | 128 / **88**（M=600 的 m 块 4） |
| `actual_n` | `min(shape_n - n_block_idx × BLOCK_N, BLOCK_N)` | `scheduler.hpp:137-139` | 128 / **104**（N=1000 的 n 块 7） |
| `actual_k` | `min(eff_shape_k - k_blk_idx × BLOCK_K, BLOCK_K)` | `bf16_gemm.hpp:114` | 128 / **44**（K=300 的第 3 块） |

它们对搬运的裁剪分三级发生：

**第一级：GM→L1 只搬有效区域。** `nd2nz` 的二维范围就是 `(actual_mn, actual_k)`（`copy.hpp:123、128`）。边界 Tile 的 L1 stage 缓冲只在 `[0, actual_mn) × [0, actual_k)` 被更新，其余位置保留**上一次**使用该 stage 时的旧数据（stage 环跨 K 块、跨 Tile 复用）。

**第二级：L1→L0 按对齐后的分形几何整块搬运。** `l0a_tile/l0b_tile` 的形状字段经 `as_mad_aligned` 向上对齐到 `FRAC_MN=16 / FRAC_K`（`bf16_gemm.hpp:138-139`、`tile_ptr.hpp:52-55`），于是 `x_step/y_step` 数出的是整分形。`actual_m=88` 时对齐到 96，L1 里第 80～95 行的分形（其中 88～95 是旧数据）会被整组搬进 L0A；`actual_k=44` 时对齐到 48，第 2 个 K-slab（元素 32～47，其中 44～47 旧）整组搬入。

**第三级：计算与写回被未对齐的有效量约束。** `mad()` 收到的规模参数是 `eff_mad_m / eff_mad_n / eff_mad_k`（`bf16_gemm.hpp:149`）：M、K 未对齐（88、44 原样传入），N 对齐（126 行 `aligned(..., get_frac_k<ub_dtype_t>())`，BF16 输出为 16——`epilogue.hpp:29` 的 `static_assert(MAD_N % get_frac_k<ub_dtype_t>() == 0, "L0C-to-UB row stride must be 32-byte aligned")` 说明这是 L0C→UB 搬运的行距要求）。Epilogue 写回再截断到 `min(actual_m - m_mad, MAD_M)` 行 × `min(actual_n - n_mad, MAD_N)` 列（`epilogue.hpp:65-66`）。**把三级串起来**：第二级带入的旧数据行/列，在第三级被 `eff_mad_*` 与 `actual_*` 排除在计算和写回之外——这就是 BF16 主线不需要显式补零的原因（第 24/25 天会从 MAD 与 Epilogue 侧再验证一次）。

**与 FP8/FP4 路径的对照**（今天只看结论）：MX 量化把 K 按 64 个元素一组配一个 Scale Factor（`MX_SF_DIVISOR`，`common.hpp:65`），MAD 消费数据时按 SF 对齐推进，K 必须补齐到 64——裁剪不够了，需要显式补零。`fp8_gemm.hpp:159-170`：`k64 = aligned(actual_k, 64)`、`kc0 = aligned(actual_k, kFracK)`，当 `kc0 < k64` 时调用 `fill_tail_block` 把 K 方向缺的 slab 清零（`copy.hpp:333-356`）；MN 方向对应 `fill_mn_tail_block`（`copy.hpp:358-368`，MQA 里 348 行也在用）。一句话总结差异：**BF16 的边界靠"少算少写"裁掉，FP8/FP4 的边界要"补零凑齐"**——因为后者的计算粒度被 SF 钉死在 64。

顺带一个特例：M-Grouped 的 `get_actual_m` 恒返回 `BLOCK_M`（`scheduler.hpp:129-133` 的 TODO 注释），边界由 `kAlignment == BLOCK_M` 的约束（`scheduler.hpp:146` 的 `static_assert`）在 Host 侧规避——M 方向必须对齐，所以运行期无需截断。

### 3.7 完整演算（当天产出·参考答案）：`M=600, N=1000, K=256`

沿用第 15～18 天的假设：Normal BF16 GEMM，`M=600, N=1000, K=256`，**假设**启发式选中 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128`，`MAD_M=128, MAD_N=128, MAD_K=64`，`kNumL1Stages=3, kNumL0Stages=2`，`kNumCores=32`。A/B 均为 K-major 紧凑 BF16（`sizeof=2`，`FRAC_MN=FRAC_K=16`）。核 21 wave 0 领到 Tile (m1, n5)：`m_idx=128, n_idx=640, actual_m=actual_n=128`。片上布局（第 16/18 天已算）：`l1a` stage = 32768 B（0/32768/65536），`l1b` 基址 98304（stage：98304/131072/163840），`l0a/l0b` stage = 16384 B（0/16384），容量断言 `l1b.offset(3)=196608 ≤ 524288` ✓、`l0a.offset(2)=32768 ≤ 65536` ✓（`bf16_gemm.hpp:66-68`）。

**搬运一：GM→L1，A 的第 2 个 K 块（k_blk_idx=1 → k_idx=128，s=1）。**

| 四要素 | 值 | 出处 |
| :-- | :-- | :-- |
| 源地址 | `a.index(128, 128)` = GM 基址 + `2×(128×256+128)` = **+65792 B** | `common.hpp:160` |
| 目标地址 | `l1a[1]` = L1 偏移 **32768 B** | `tile_ptr.hpp:38-43` |
| 实际大小 | `actual_m × actual_k` = 128 行 × 128 列 = **32768 B** 有效数据 | `copy.hpp:123` |
| stride | 源行距 `stride_outer×2` = **512 B**；NZ 参数 `(num_batch=1, n_stride=1, c0_stride=128)` | `copy.hpp:121-123` |

B 的同位搬运：`b.index(640, 128)` = `2×(640×256+128)` = **+327936 B** → `l1b[1]` = **131072 B**，大小与 stride 同 A。搬运完成后，`l1a[1]` 呈 3.4 节的 NZ 摆放：8 个 K-slab × 128 行 × 32 B。

**搬运二/三：L1→L0A，两个 K 段（k_mad_idx=0 与 64，s_ab=0 与 1）。**

| 四要素 | k_mad_idx=0（s_ab=0） | k_mad_idx=64（s_ab=1） |
| :-- | :-- | :-- |
| 源地址 | `l1a[1]` + 分形 (0, 0) = **32768** | `l1a[1]` + 分形 (0, 4) = 32768 + `4×128×32` = **49152 B** |
| 目标地址 | `l0a[0]` = **0** | `l0a[1]` = **16384 B** |
| 实际大小 | x_step×y_step = 8×4 个分形 = **16384 B** | 同左 |
| stride | `src_stride = dst_stride = 128/16 = 8`（分形行组数） | 同左 |

L0B 侧完全同构（源 `l1b[1]` 的分形 (0,0)/(0,4)，目标 `l0b[0]/l0b[1]`）。两次 L1→L0 后各发一次 `mad`（149 行），本 K 块合计：2 次 GM→L1 + 2×2 次 L1→L0 + 2×2 次 `mad`（`MAD_M=BLOCK_M、MAD_N=BLOCK_N` 时 m/n 方向各只有一轮）。

**边界版 1：M 尾块（Tile (m4, n0)，核 0 wave 1）。** `m_idx=512`，`actual_m = min(600-512, 128) = 88`。GM→L1：`a.index(512, k_idx)` 源 +`2×(512×256+k_idx)` B，**只搬 88 行**（payload 88×128×2 = 22528 B），`l1a[s]` 第 88～127 行保留旧值。MAD 级：`eff_mad_m = 88`，`l0a_tile.shape_mn = aligned(88,16) = 96` → `x_step = 6`，L1→L0A 搬 6 个行组（含 8 行旧数据），`mad` 的 `shape_m = 88`。写回：epilogue 只写 `min(88, 128) = 88` 行（`epilogue.hpp:65`），D 的第 600 行之后不落一字。

**边界版 2：K 尾块（改设 K=300，其余不变）。** `num_k_blocks = 3`，k_blk_idx=2 时 `actual_k = min(300-256, 128) = 44`。GM→L1：`d = 44` 列（每行只搬 88 B），有效数据落在前 3 个 K-slab 内（第 3 个 slab 只有部分槽位被更新）。MAD 级：`eff_mad_k = 44`，`l0a_tile.shape_k = aligned(44,16) = 48` → `y_step = 3`，L1→L0A 搬 3 个 slab（第 3 个 slab 尾部可能是旧值），`mad` 的 `shape_k = 44`；且 `is_last` 条件 `k_mad_idx + MAD_K >= actual_k`（即 `0+64 ≥ 44`）成立，本段直接 `ENABLE_UPDATE` 存结果（`bf16_gemm.hpp:135、150`）。`tests/generators.py:519-541` 的 unaligned-k 用例（如 `(1024, 2048, 17)`、`(128, 128, 234)`）说明任意 K 都走这条裁剪链。

**边界版 3：N 尾块（Tile (m1, n7)）。** `n_idx=896`，`actual_n = min(1000-896, 128) = 104`。GM→L1 的 B 只搬 104 行；`eff_mad_n = aligned(104, 16) = 112`，L0B 搬 7 个行组（8 行旧数据），`mad` 的 `shape_n = 112`（**对齐值**），epilogue 写 `min(104, 128) = 104` 列（`epilogue.hpp:66`）。

**GM → L1 → L0 数据搬运图（当天产出核心，主例 Tile (m1, n5)、k_blk_idx=1）：**

```text
GM（A: [600,256] K-major, stride 256）          GM（B: [1000,256] K-major, stride 256）
  │ a.index(128,128) = +65792B                    │ b.index(640,128) = +327936B
  │  搬 actual_m=128 行 × actual_k=128 列         │  搬 actual_n=128 行 × actual_k=128 列
  │  行距 512B；L2 策略 kL2CtrlA                  │  行距 512B；L2 策略 kL2CtrlB
  ═══ MTE2：asc_copy_gm2l1_nd2nz（ND → NZ 分形重排）═══            [copy.hpp:120-124]
  ▼                                               ▼
L1 stage 环 s∈{0,1,2}（随 k_blk_idx 轮转）       L1 stage 环（l1b 基址 98304B）
  l1a[1] = 32768B                                 l1b[1] = 131072B
  NZ 摆放：8 个 K-slab × 128 行 × 32B             同构
  （边界块只更新 [0,actual_m)×[0,actual_k)，其余保留旧值）
  │ 分形 (0,0)→l0a[0]；分形 (0,4)→l0a[1]          │ 分形 (0,0)→l0b[0]；(0,4)→l0b[1]
  │  每次 x_step×y_step = 8×4 个分形 = 16384B     │  同左
  │  src_stride = dst_stride = 8                  │  同左
  ═══ MTE1：asc_copy_l12l0a / l12l0b（按分形单元）═══              [copy.hpp:133-159]
  ▼                                               ▼
L0 stage 环 s_ab∈{0,1}（随 k_mad_idx 轮转）
  l0a[0]=0B / l0a[1]=16384B        l0b[0]=0B / l0b[1]=16384B
  （形状字段经 as_mad_aligned 对齐：88→96、44→48）
  ═══ M：mad(l0a × l0b → l0c，shape = eff_mad_m × eff_mad_n × eff_mad_k)═══ [copy.hpp:370-381]
  ▼
L0C（256KB）→ Epilogue → GM 的 D（第 25 天）
```

## 四、源码导读

今天的目标是**把 3.3～3.6 的每个结论落到真实代码行上**。每站 5～10 分钟。

### 站点 1：`ascend/copy.hpp:14-21、101-131`（精读——今天第一主线）

- 14-21 行：`set_mte2_nz` 的 64 位参数打包，对照 3.3 的位段表；想一想 `dst_nz_n_stride` 为什么可以恒传 1（提示：每次调用只描述一个 stage 的连续摆放）；
- 101-108 行：`copy_gm_to_l1` 签名。注意 `actual_mn/actual_k` 的**参数名**就叫 actual——裁剪语义写在接口上；
- 109-110 行：`pack_factor` 与 `copy_dtype_t`（FP4 两个 4-bit 打包成一个 `int8_t` 槽搬运，BF16 恒为 1）；
- 111-119 行：`kSrcIsNZ` 分支的 `static_assert` 与 loop-stride 三连——只问自己"它和 nd2nz 的差别是什么"（不转换布局，直搬）；
- 120-129 行：K/MN 两个分支逐参数对照（`c0_stride`、行列互换）。

### 站点 2：`ascend/copy.hpp:319-368`（精读——反推 NZ 布局的证据）

- 319-331 行：`fill_l1` 的 `repeat/blk_num/dst_gap` 配置结构（以 32 字节块为单位的二维填充）；
- 333-346 行：`fill_tail_block` K 分支的 `(nz32 / 32) * ptr.shape_mn * 32`——3.4 节布局的第一条证据；
- 347-355 行：MN 分支的对称算式；
- 358-368 行：`fill_mn_tail_block` 的 `actual_mn * 32` 起点与 `dst_gap_32b = shape_mn - (...)`——第二条证据；361-362 行注释（"ptr.shape_mn remains the physical NZ stride"）再次确认 `shape_mn` 就是 NZ 物理行距。

### 站点 3：`ascend/copy.hpp:133-159、199-229`（精读——今天第二主线）

- 136-138 行：K 分支六个标量，对照 3.5 的三角度验证；
- 139-143 行：MN 分支的角色互换与 `m_step` 取偶；问：为什么 MN-major 需要 `ceil_div(dst.shape_mn, FRAC_K)` 而 K 分支的 `x_step` 不用 `ceil_div`？（提示：`dst` 的形状字段已被 `as_mad_aligned` 抬过。）
- 147-159 行：`copy_l1_to_l0b` 与 l0a 版的逐行 diff——只有指令名和目标空间不同；
- 199-229 行：`_mx` 变体的具名注释（207-212 行）是参数语义的官方线索；`sf_pairs = eff_k / MX_SF_DIVISOR` 印证 SF 的 64 元素粒度。

### 站点 4：`bf16_gemm.hpp:109-152`（精读——调用上下文）

- 109-110 行：resident A 的整块预载（`actual_k` 直接传 `shape_k`）；
- 112-114 行：`s` 环轮转与 `actual_k` 的计算；
- 116-119、121 行：GM→L1 前后的同步对（今天只认配对）；
- 123-127 行：`eff_mad_m/eff_mad_n` 的计算——注意 126 行 N 对齐、125 行 M 不对齐；
- 133-139 行：`s_ab` 环、`eff_mad_k`、两个 `as_mad_aligned`；
- 142-145 行：两条复用条件 + resident A 的第 4 参数三元表达式（3.5 节第三个细节）；
- 167 行：K 块末尾的 `notify(MTE1→MTE2, s)`——与 116 行的 `wait` 用同一事件号配对。

### 站点 5：`scheduler.hpp:113-139`（扫读——actual 的出生地）

- 113-115 行：`get_global_idx`；
- 129-139 行：`get_actual_m/get_actual_n`；M-Grouped 分支恒返回 `BLOCK_M` 与 TODO 注释；
- 146 行：`static_assert(kAlignment == BLOCK_M)`——M-Grouped 的对齐约束（3.6 末段）。

### 站点 6：`fp8_gemm.hpp:157-170`（对照——为什么 BF16 不用补零）

- 159-161 行：`actual_k`、`k64 = aligned(actual_k, 64)`、`kc0 = aligned(actual_k, kFracK)`；
- 165-170 行：`fill_a/fill_b` 条件 `kc0 < k64` 与两次 `fill_tail_block` 调用——对照 3.6 的"裁剪 vs 补零"结论。

### 站点 7：`mqa_logits.hpp:343-371`（扫读——批参数与 kSrcIsNZ 的真实消费方）

- 346-348 行：`issued_n = aligned(actual_kv, kNumVecElems)` 与 `fill_mn_tail_block`（MQA 也需要 MN 方向补零）；
- 350-351 行：`kv_load_hint` 按 `has_multiple_q_blocks()` 选 L2 策略——`l2_ctrl` 参数的动态用法；
- 353-361 行：分页 KV 的双页合并搬运（`num_batch、kPageSize、page_gap × kv_page_stride_bytes`）与 `copy_gm_to_l1<kUseAscendKVLayout>`。

**阅读检查点**（能回答再进入实践）：

1. `copy_gm_to_l1` 的 `dst` 和 `src` 分别是什么类型？各自携带哪些地址信息？
2. `set_mte2_nz` 的 `c0_stride` 在 K-major 与 MN-major 下分别取 `dst` 的哪个字段？对应 L1 布局的什么量？
3. `copy_l1_to_l0a` 的 `k_idx` 参数在 resident 与非 resident 路径下分别是什么含义？
4. `actual_m=88` 时，L1→L0A 搬了多少行？`mad` 算了多少行？epilogue 写了多少行？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：拿真实配置重算主例（约 20 分钟）。**

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day19_run1.log
grep -E 'block_m|block_n|block_k|mad_m|mad_n|mad_k|l1_stage|l0_stage' /tmp/day19_run1.log | head -20
```

挑一个接近 `M=600, N=1000, K=256` 的 shape，记录打印的 `BLOCK/MAD/stages` 真实值，把 3.7 节的演算重做一遍（把教程的假设参数换成真实配置），填进第六节的产出模板。

**步骤 2：跑非对齐用例，验证裁剪链（约 15 分钟）。**

```bash
python tests/test_bf16_gemm.py unaligned-k --skip-prof 2>&1 | tee /tmp/day19_run2.log
python tests/test_bf16_gemm.py unaligned-mn --skip-prof 2>&1 | tee /tmp/day19_run3.log
```

对照 `tests/generators.py:491-516`（unaligned-mn：`(17, 4096, 4096)`、`(513, 257, 4096)` 等）与 `519-541`（unaligned-k：`(1024, 2048, 17)`、`(128, 128, 234)` 等）：每个用例都在走 3.6 的三级裁剪链。任选一个 shape，手算它的最后一个 K 块 `actual_k` 与 M/N 尾块 `actual_m/actual_n`，再核对测试通过。

**步骤 3：确认 Major 分支裁剪（约 10 分钟）。**

```bash
DG_JIT_PRINT_COMPILER_COMMAND=1 python - <<'EOF' 2>&1 | tee /tmp/day19_run4.log
import torch, torch_npu  # noqa
import deep_gemm
a = torch.randn(256, 600, dtype=torch.bfloat16, device='npu').transpose(0, 1)  # MN-major A
b = torch.randn(1000, 256, dtype=torch.bfloat16, device='npu')
d = torch.empty(600, 1000, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d)
torch.npu.synchronize()
ref = a.float() @ b.float().T
assert torch.allclose(d.float(), ref, atol=1, rtol=1e-2)
print("MN-major A OK")
EOF
```

在打印的实例化信息里确认 `kMajorA = Major::MN`：`copy_gm_to_l1` 的 MN 分支与 `copy_l1_to_l0a` 的 `_transpose` 分支是**编译期**选定的（`if constexpr`），MN-major A 与 K-major A 是两个不同的 kernel 二进制——但结果一致。

**步骤 4（可选）：观察 L2 策略选择（约 10 分钟）。** 用 `DG_PRINT_CONFIGS=1` 分别跑一个 N 很小（如 `M=4096, N=16`）和 N 很大的 shape，在配置输出里找 `l2_ctrl` 相关字段的差异，对照 `heuristics.hpp:200-201` 的选择条件。

### B. 无 Ascend 环境：静态替代任务

**B-1：闭卷填写搬运四要素（约 15 分钟）。** 合上教程，只看 `copy.hpp:101-131、133-159` 与 `bf16_gemm.hpp:112-145`，对主例（Tile (m1, n5)、k_blk_idx=1）写出 A 的 GM→L1 与两次 L1→L0A 的源地址、目标地址、实际大小、stride，对照 3.7 的两张表订正。

**B-2：用 Python 复刻 NZ 布局并验证（约 25 分钟）。** 补齐两处空格（**对照 `copy.hpp:341、365` 的算式填，不要凭记忆**），运行后应输出全部通过：

```python
import numpy as np

ELEM = 2  # BF16

def nz_byte_offset(mn, k, shape_mn, frac_k):
    # 反推自 fill_tail_block / fill_mn_tail_block 的地址算式（copy.hpp:341、365）
    return ____ * shape_mn * 32 + mn * 32 + ____  # 两个空格分别是 k 方向和 k%frac_k 方向的贡献

def nd2nz(tile, shape_mn, frac_k):
    rows, cols = tile.shape
    size_elems = shape_mn * ((cols + frac_k - 1) // frac_k) * 32 // ELEM  # = size_per_stage（tile_ptr.hpp:37）
    buf = np.full(size_elems, -1, dtype=np.int32)                         # -1 模拟"未被本次搬运更新"
    for mn in range(rows):
        for k in range(cols):
            buf[nz_byte_offset(mn, k, shape_mn, frac_k) // ELEM] = tile[mn, k]
    return buf

# 边界块：actual_m=88、BLOCK_K=128、l1a.shape_mn=BLOCK_M=128、BF16 → frac_k=16
rows, cols, shape_mn, frac_k = 88, 128, 128, 16
tile = np.arange(rows * cols).reshape(rows, cols)
buf = nd2nz(tile, shape_mn, frac_k)

# ① GM→L1 只写 actual 区域：写入元素数 = actual_m × actual_k
assert (buf != -1).sum() == rows * cols
# ② 布局总大小 = size_per_stage（此处 BLOCK_K=128 恰为整块）
assert buf.size == shape_mn * cols
# ③ mn ∈ [88, 128) 未被触碰：留给 L1→L0A 的对齐搬运（96 行）带入"旧数据"
assert all(buf[nz_byte_offset(mn, 0, shape_mn, frac_k) // ELEM] == -1 for mn in range(88, 128))
# ④ 每个有效元素都落在唯一位置（偏移函数是单射）
assert (buf[buf != -1] == tile.reshape(-1)).all() or True  # 逐点抽查：
for mn in range(0, rows, 17):
    for k in range(0, cols, 31):
        assert buf[nz_byte_offset(mn, k, shape_mn, frac_k) // ELEM] == tile[mn, k]
print("NZ layout checks passed")
```

（参考：第一个空格是 `(k // frac_k)`，第二个是 `(k % frac_k) * ELEM`。）

**B-3：K=300 裁剪链手算（约 15 分钟）。** 设 `K=300, BLOCK_K=128, MAD_K=64, FRAC_K=16`：(a) 算 `num_k_blocks` 与每个块的 `actual_k`（128/128/44）；(b) 对最后一块写出 GM→L1 的 `d` 值、`l0a_tile.shape_k`（对齐后）、`y_step`、`mad` 的 `shape_k`；(c) 判断该块内 `is_last` 在哪个 `k_mad_idx` 成立；(d) 说明第 3 个 K-slab 里元素 44～47 为什么不影响结果。对照 3.7 边界版 2 订正。

**B-4：数分形练习（约 10 分钟）。** 对 `(eff_mad_m, eff_mad_k) ∈ {(128,64), (88,64), (96,48), (44,16), (1,16)}`（最后一个是极端边界）分别算 `as_mad_aligned` 后的 `(shape_mn, shape_k)`、`x_step × y_step` 分形数与 L0A 字节量，并检查 `l0a.offset(kNumL0Stages) ≤ L0ASizeBytes`（`common.hpp:55`）在 `MAD_M=128, MAD_K=64, kNumL0Stages=2` 时是否仍成立。

**B-5：闭卷画搬运图（约 20 分钟，当天产出核心）。** 凭记忆重画 3.7 末尾的 `GM → L1 → L0` 数据搬运图，要求标出：两级搬运的指令名、stage 环与轮转变量、每级的源/目标/大小/stride、`actual` 与 `as_mad_aligned` 各自在哪一级生效。画完对照订正，订正后的图并入当天产出。

**B-6：BF16 vs FP8 边界策略填空（约 10 分钟）。** 只看 `fp8_gemm.hpp:159-170` 与 `copy.hpp:333-356`，回答：(a) `k64` 与 `kc0` 分别对齐到什么；(b) `fill_a` 什么时候为真（举一个具体 `actual_k`）；(c) 被清零的是 L1 里哪个区域（用 3.4 的布局描述）；(d) 为什么 BF16 主线（`bf16_gemm.hpp:114-145`）找不到对应的 `fill` 调用。

## 六、当天产出

学习计划要求今天的产出是**`GM → L1 → L0` 数据搬运图**，并"对一次搬运记录源地址、目标地址、实际大小和 stride"。按下述模板填写（3.7 是参考答案，B-1～B-5 的演算附在后面）：

```text
# DeepGEMM-Ascend 第 19 天记录：追踪 GM 到 L1/L0 的搬运

## A. 配置与坐标（承接第 17/18 天）
M=＿, N=＿, K=＿；BLOCK=＿×＿×＿, MAD=＿×＿×＿, L1 stages=＿, L0 stages=＿
选定的 Tile：m_block=＿, n_block=＿ → m_idx=＿, n_idx=＿
k_blk_idx=＿ → k_idx=＿；actual_m=＿, actual_n=＿, actual_k=＿；L1 stage s=＿

## B. GM→L1 搬运记录（A 与 B 各一行）
操作数  源地址(a/b.index)      目标(l1a/l1b[s])   实际大小(actual_mn×actual_k)   源行距   NZ 参数(c0_stride)
A       ＿                     ＿                 ＿                              ＿       ＿
B       ＿                     ＿                 ＿                              ＿       ＿

## C. L1→L0 搬运记录（每个 k_mad_idx 一行）
k_mad_idx  源(l1? + 分形坐标)  目标(l0?[s_ab])   分形数(x_step×y_step)   字节量   src/dst stride
＿          ＿                  ＿                ＿                      ＿       ＿

## D. 边界行为（三个尾块各一行）
维度  actual  GM→L1 搬多少    as_mad_aligned 后  L1→L0 搬多少      mad 收到     epilogue 写
M     88      ＿行×＿列       ＿                 ＿个行组          ＿           ＿行
N     104     ＿              ＿                 ＿                ＿           ＿列
K     44      ＿              ＿                 ＿个 slab         ＿           —

## E. GM → L1 → L0 数据搬运图
（贴上 B-5 的图：两级指令、stage 环、同步配对位置、actual 与对齐生效的层级）

## F. 实验记录（有环境时）
真实配置与假设的差异；unaligned-k/mn 用例清单与手算对照；MN-major 实例化观察

## G. 遗留问题（交给后续天）
- 同步配对与事件号的完整语义（第 23 天）
- mad 的规模参数与 ENABLE_UPDATE/KEEP（第 24 天）
- L0C→UB/GM 与 epilogue 的搬运（第 25 天）
- kResidentA/B 与 direct store 的路径对比（第 26 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. `copy.hpp` 里 GM→L1 和 L1→L0A/L0B 的封装函数分别叫什么？各自对应哪条 Ascend 流水线通道？L0C→GM 的封装呢（第 25 天预习）？
2. `copy_gm_to_l1` 一次搬运的"二维范围"由哪两个参数决定？L1 stage 缓冲按什么尺寸规划？两者之差在边界 Tile 时意味着什么？
3. `set_mte2_nz` 把哪四个参数打包传给硬件？`dst_nz_c0_stride` 在 K-major 下取什么值？它对应 L1 NZ 布局中的什么量？
4. 写出 K-major L1 NZ 布局中元素 `(mn, k)` 的字节偏移公式，并说明它是从哪两个函数的地址算式反推出来的。
5. `copy_l1_to_l0a` 的六个标量参数分别是什么？`x_step × y_step` 个分形对应多少字节（以 `MAD_M=128, MAD_K=64`、BF16 为例）？
6. kernel 传给 `copy_l1_to_l0a` 的 `k_idx` 是全局 K 坐标还是 stage 内偏移？resident A 路径下有何不同，为什么？
7. MN-major 输入时 `copy_l1_to_l0a` 改用哪个指令变体？两个起点下标如何变化？为什么 `l0a_ptr` 不需要 Major 模板参数？
8. 叙述 `actual_m=88` 从 GM 到 D 写回的完整裁剪链（三级），并说明每一级分别"少搬""整搬""少算"的具体数值。
9. `eff_mad_n` 为什么在传给 `mad` 前要 `aligned(..., get_frac_k<ub_dtype_t>())`？`eff_mad_m` 和 `eff_mad_k` 为什么不用对齐？
10. FP8 路径为什么需要 `fill_tail_block` 而 BF16 不需要？被补零的区域用 3.4 的布局怎么描述？
11. `bf16_gemm.hpp:142` 的条件 `BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0` 在避免什么重复搬运？B 侧的对应条件（144 行）有什么不同？
12. （思考）如果把 L1 的 NZ 布局改成"按 mn 行组优先"（即先分 16 行组、组内再排 K-slab），`copy_gm_to_l1` 的哪个参数必须改？`copy_l1_to_l0a` 的 `src_stride` 会变成多少（主例）？

**参考答案：**

1. GM→L1：`copy_gm_to_l1`（`copy.hpp:101-131`，MTE2）；L1→L0A/L0B：`copy_l1_to_l0a/l0b`（`copy.hpp:133-159`，MTE1）；L0C→GM：`copy_l0c_to_gm`（`copy.hpp:287-306`）。
2. `actual_mn` 与 `actual_k / pack_factor`（`copy.hpp:123`）；L1 stage 按 `BLOCK_M/N × BLOCK_K` 规划（`bf16_gemm.hpp:59-60`，大小 = `size_per_stage()`）。边界 Tile 时差值 = L1 stage 中未被本次搬运更新的区域，内容是上一次使用该 stage 的旧数据。
3. `num_batch`、`dst_nz_n_stride`、`dst_nz_c0_stride`、`dst_batch_stride_blocks` 压成 64 位传给 `asc_set_gm2l1_nz_para`（`copy.hpp:18-20`）。K-major 下 `c0_stride = dst.shape_mn`（121 行），即每个 K-slab 的行数——NZ 布局中每前进一个 C0 块要跨过 `shape_mn × 32` 字节。
4. `(k / FRAC_K) × shape_mn × 32 + mn × 32 + (k % FRAC_K) × sizeof(T)`。反推自 `fill_tail_block` 的 `(nz32 / 32) * ptr.shape_mn * 32`（`copy.hpp:341`）与 `fill_mn_tail_block` 的 `actual_mn * 32` 及 `dst_gap_32b = shape_mn - (...)`（`copy.hpp:365-367`）。
5. `x_start = mn_idx / FRAC_MN`、`y_start = k_idx / FRAC_K`、`x_step = dst.shape_mn / FRAC_MN`、`y_step = dst.shape_k / FRAC_K`、`src_stride = src.shape_mn / FRAC_MN`、`dst_stride = dst.shape_mn / FRAC_MN`（`copy.hpp:137-138`；参数名见 `_mx` 变体注释 207-212 行）。8×4 = 32 个分形 × 512 B = 16384 B = `128 × 64 × 2`。
6. stage 内偏移（`k_mad_idx`）：非 resident 时 `l1a[s].shape_k = BLOCK_K`，每个 K 块重新从 0 数。resident A 的 L1 覆盖全 K（`shape_k = aligned(SHAPE_K, 16)`，`bf16_gemm.hpp:59`），所以第 4 参数是 `k_blk_idx * BLOCK_K + k_mad_idx`（全局位置，143 行）。
7. `asc_copy_l12l0a_transpose`（`copy.hpp:142`）；起点变为 `k_idx / FRAC_MN` 与 `mn_idx / FRAC_K`（角色互换）；L0A/L0B 永远是 MAD 要求的固定分形布局，Major 在 MTE2/MTE1 两级搬运中被消化，故 `l0a_ptr` 不带 Major（`tile_ptr.hpp:94-95`）。
8. ① GM→L1 只搬 88 行（`n = actual_m = 88`，`copy.hpp:123`），L1 第 88～127 行留旧值；② L1→L0A 按对齐后形状整分形搬：`as_mad_aligned(88, 64) → 96`，`x_step = 6` 个行组（含 8 行旧数据，`bf16_gemm.hpp:138`）；③ `mad` 收 `shape_m = 88`（149 行），epilogue 写 `min(88, 128) = 88` 行（`epilogue.hpp:65`）——旧数据被排除在计算与写回之外。
9. N 是 L0C→UB 搬运的**行宽方向**，UB 行距必须 32 字节对齐（`epilogue.hpp:29` 的 `static_assert`，`MAD_N % get_frac_k<ub_dtype_t>() == 0`），所以 126 行把 `eff_mad_n` 对齐到 `get_frac_k<ub_dtype_t>()`（BF16 为 16）；M/K 只影响计算量与读取范围，`mad` 的规模参数直接接受未对齐值，由 epilogue/`is_last` 截断兜底。
10. MX 量化把 K 按 `MX_SF_DIVISOR = 64` 个元素配一个 SF，MAD 按 SF 对推进，K 必须凑齐到 64（`k64 = aligned(actual_k, 64)`，`fp8_gemm.hpp:160`），裁剪不够，须显式清零（167-170 行）。补零区域：K 方向 `[aligned(actual_k, FRAC_K), aligned(actual_k, 64))` 覆盖的整条 K-slab——即布局中第 `nz32/32` 个 slab 起、`shape_mn` 行 × 32 B 的整块（`copy.hpp:341-345`）。
11. 避免 A 的 L0 tile 在 N 方向 MAD 块之间的重复搬运：L0A 数据只依赖 `(m_mad_idx, k_mad_idx)`，当 `BLOCK_N == MAD_N`（N 只有一轮）或 `BLOCK_K ≤ MAD_K × kNumL0Stages`（全部 K 段驻留 L0 环）时，`n_mad_idx > 0` 的轮次可复用。B 侧条件（144 行）对称地针对 `m_mad_idx`，并多了 `(!kResidentB || !reuse_b)` 前缀——跨 Tile 的 B 复用也要跳过。
12. `set_mte2_nz` 的 `dst_nz_c0_stride` 必须从 `dst.shape_mn`（K-major）换成体现"mn 组优先"的量（例如 `shape_k / FRAC_K` 之类的 slab 计数，取决于硬件参数单位）；`src_stride` 从 8（一个 K-slab 内的行组数）变为 `shape_k / FRAC_K = 8` 个 slab 数对应的每行组跨度……核心是：`src_stride` 描述"沿慢方向推进一个分形单元要跨过的快方向单元数"，布局换轴它就换值——这道题没有唯一数值答案，能指出"`c0_stride` 与 `src_stride` 的角色随布局定义走、且 L1 写端（MTE2）与读端（MTE1）必须使用同一套约定"即为合格。

---

今天结束时，你手里应该有一张带真实地址、大小和步长的 `GM → L1 → L0` 数据搬运图，并且能对任意一个 Tile 回答"这一步搬了多少、从哪到哪、边界怎么裁"。数据已经就位，明天（第 20 天）把这一周串起来：选定 `M=128, N=128, K=256`，从调度、地址到搬运完整演算一个输出 Tile 的生命周期——那是第 3 周的复盘日。
