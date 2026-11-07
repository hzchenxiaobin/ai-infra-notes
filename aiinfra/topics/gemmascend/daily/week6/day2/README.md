# 第 6 周 · 第 2 天（总第 37 天）：学习 K-Grouped GEMM

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 6 周"Grouped GEMM 与 MQA Logits"的第 2 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

昨天我们把分组轴放在 M 维：token 按 expert 排好序，A/D 是平坦缓冲，只有 B 逐组取权重。今天把分组轴**换到 K（收缩）维**：**K-Grouped GEMM 中 A `[total_k, M]` 与 B `[total_k, N]` 仍是两块平坦缓冲，但拼接的是每组各自的 K 段；输出变成每组分一份的 `D [G, M, N]`；所有组共享同一个 M×N block 网格，差别只在每组的 K 循环长度（`current_shape_k`）**。它对应 MoE 反向的 wgrad 场景：收缩维是按 expert 分组的 token 流，每组输出该 expert 的权重梯度。当天产出是**三种 GEMM 类型对比表**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 K-Grouped 的形状契约：A `[K, M]`（tn）、B `[K, N]`（tn）、D `[G, M, N]`、`ks_cpu` 可选 `[G]` CPU int 列表、`grouped_layout` `[G]` int32 且 contiguous；并指出成文处——`csrc/apis/gemm.hpp:45-58` 的形状注释与 `k_grouped_bf16_gemm_tn_contiguous` 的 docstring（`:501-514`）；
2. 解释 **psum-over-K 布局**：`grouped_layout[i]` 是前 i+1 组实际 K 的累积结尾；组 g 的有效 K 区间为 `[align(psum[g-1], 256), psum[g])`，组 0 从 K=0 开始；`total_k = align(psum[G-1], 256)`（`scheduler.hpp:25-31` 注释、`tests/generators.py:212-220`）；
3. 讲清 **"tn + 逻辑转置"技巧**：API 接收 `[K, M]`/`[K, N]`，Host 用 `get_transposed<true>(a_, 0, 1)` 只改 shape/stride/major 书面信息、不动数据（`gemm.hpp:363-365, :526-528` 注释），使 `get_gemm_desc` 无需单独处理转置分支；由此推出 **A、B 都必须 MN-major**（`gemm.hpp:94-95` 的断言），并与昨天"A 必须 K-major"对照出轴对称关系；
4. 说明 `ks_cpu` 与 `grouped_layout` 的分工：前者是**纯 Host 提示**，只进 `expected_k_from_ks()`（`gemm.hpp:35-43`，取均值、下限 1，缺省 `k/g`）供启发式估容量；后者是**设备侧唯一权威**，Kernel 从 GM 直读；
5. 逐行讲解 `Scheduler::get_next_block` 的 K-Grouped 分支（`scheduler.hpp:163-177`）：共享 M×N 网格（`num_blocks` 不改写）、组游标 `current_k_block_cumsum / last_psum_k / current_psum_k / current_sf_k_cumsum`、空组的 `current_shape_k > 0` 双重守卫，以及 `get_k_idx_base()/get_sf_idx_base()`（`:124-125`）的用途；
6. 分析 Kernel 层的三处 K-Grouped 逻辑：`eff_shape_k = scheduler.current_shape_k` 决定每组 K 循环长度、`k_idx = k_idx_base + k_blk_idx × BLOCK_K` 给**平坦的 A/B** 加组偏移（`bf16_gemm.hpp:95-97, :112-118`）、`gm_d.group(d_group_idx, shape_m, shape_n)` 给**逐组的 D** 加偏移（`:157, :186-188`）；并解释非对齐组尾由 `actual_k` 截断（`:114`）而空组输出由 `aiv_zero_empty_groups` 清零（`:175-176`、`epilogue.hpp:108-141`）；
7. 描述 **累加语义**：K-Grouped 支持 `D[g] = A_g @ B_g + C[g]`（`gemm.hpp:347, :512` 的 docstring；`tests/common.py:56` 允许 Normal 与 K-Grouped accumulate）；`kWithAccumulation` 使 AIV drain 变 atomic add（`epilogue.hpp:91-96`），空组保持 `D[g] = C[g]` 不动恰好就是期望结果，因此清零只在 `!kWithAccumulation` 时执行；
8. 总结配置层对 `expected_k` 的三副面孔：`compute_k = expected_k`（每块的计算量按组内 K 估）、`k = num_groups × expected_k`（A/B/SF 总流量按总 K 估）、`resident_k = expected_k`（fp8 常驻缓冲按组内 K 估）（`heuristics.hpp:597-599, :545`）；并指出 K-Grouped **没有** M-Grouped 的 BLOCK_M=256 钉死规则（`:104-106` 只过滤 m-grouped），`compiled_dims` 默认 `"mn"`（`gemm.hpp:612, :674`）——K 每组不同，只能运行时；
9. 独立完成**三种 GEMM 类型对比表**（当天产出），覆盖分组轴、形状、major、调度网格、空组、非对齐组尾、accumulate 与典型场景。

## 二、预备知识

- **第 3 天**：Tiling 与数据复用——今天每组共享同一个 M×N 网格，K 循环长度逐组不同。
- **第 11 天**：`GemmDesc` 与 `get_gemm_desc`——今天精读 `KGroupedContiguousWithPsumLayout` 分支。
- **第 18 天**：`gm_ptr` 的 `index()/batch()` 与 stride——今天 MN-major 的 `index(mn, k)` 地址公式是 A/B 组偏移的载体，`group()` 是 D 组偏移的载体。
- **第 20～22 天**：K 维两层循环（`BLOCK_K`/`MAD_K`）、`actual_k` 边界截断——今天的"非对齐组尾"就是同一机制在组边界上的复用。
- **第 24 天**：MAD 与首次/末次累加（`is_first`/`is_last`）——每组 K 循环各自独立地从 L0C 零开始。
- **第 25 天**：AIC/AIV 分工与 Epilogue——今天新增 `aiv_zero_empty_groups` 这一 AIV 专属职责。
- **第 26 天**：特殊执行路径——direct store 仅限 Normal（`heuristics.hpp:526`），K-Grouped 恒走 AIC store + AIV drain。
- **第 29、30 天**：`generators.py` 与 `common.py`——今天消费它们的 k-grouped 分支与 NaN 初始化技巧。
- **第 33 天**：Scale 布局转换——K-Grouped 的 sfa/sfb 都走 `transform_k_grouped_sf_into_required_layout`（`gemm.hpp:129-143`），细节当天已讲。
- **第 36 天（昨天）**：M-Grouped 的 psum 布局、分组游标、"只有 B 拿组偏移"——今天是同一套思想的镜像，请先确认昨天的对比表还能默画。

另注：第五节 A 组任务需要 Ascend 环境；无环境时完成 B 组静态任务（psum 手算、调度推演、地址与 K-block 演算、对比表）即可达到当天目标。

## 三、详细讲解

### 3.1 从 MoE 到 K-Grouped：为什么分组轴会在 K 上

昨天的 M-Grouped 对应 MoE **前向**：`y = W_g · x`，M 是 token、组在行上。MoE **反向求权重梯度**（wgrad）时形状反过来：

```text
dW_g = (dY_g)^T @ Act_g      dY_g: [tokens_g, M]，Act_g: [tokens_g, N]
     = 收缩维 = tokens（恰好按 expert 分组！）
```

Dispatch 阶段本来就把 token 按 expert 排好序拼接，于是"全部 token"构成一个平坦的收缩维，expert 边界把它切成 G 段——这正是 K-Grouped 的输入形态：**A/B 是沿 K 拼接的平坦缓冲，每组输出一份 `[M, N]` 的权重梯度**。测试用例的标注印证了这一场景（`generators.py:620-626`）：`(4, 4096, 7168, 8192)`、`(8, …, 4096)`、`(16, …, 2048)` 三组配置分别注释为 EP64/EP32/EP16——DeepSeek 类模型 256 个 expert，EP 度数越高每 rank 分到的 expert（= 组数 G）越多、每组 token 越少，而总 K 恒为 `4×8192 = 8×4096 = 16×2048 = 32768`；M/N 则是 `hidden(7168)` 与 `2×inter(4096)`/`inter(2048)` 的组合，对应 gate/up 与 down 两层投影的 wgrad。scheduler 的类注释把这一算子直接称为 "MoE k-grouped"（`scheduler.hpp:25`）。

两个自然的问题：

1. **为什么不循环调用 Normal GEMM？** 与昨天同理：组数多、每组 K 不均匀且动态，逐组 launch 开销大、小 K 组利用率低；且"按 expert 拼接的 token 流"本来就是现成的平坦缓冲。
2. **为什么不用 M-Grouped？** wgrad 的输出是**每 expert 一份权重**，输出天然带组维 `[G, M, N]`；M-Grouped 的输出是平坦的 `[M, N]`（组只是一种行划分）。输出形态决定了必须用 K 分组。

还有一个容易混淆的点：类型名里的 `WithPsumLayout` 指的是 `grouped_layout` 采用**前缀和（prefix-sum）编码**这一布局约定（昨天已建立），**不是**输出会自动跨组求和。K-Grouped 各组输出彼此独立，跨组聚合（如 wgrad 的最终 reduce）由上层完成。

### 3.2 输入输出契约：psum-over-K 布局

`get_gemm_desc` 上方的形状注释（`csrc/apis/gemm.hpp:45-58`）：

```text
 *                A                 B                CD
 *
 * Normal:       [M, K]            [N, K]            [M, N]
 * MGrouped:     [M, K]         [G, N, K]            [M, N]
 * KGrouped(*):  [M, K]            [N, K]         [G, M, N]
 * Batched:   [B, M, K]         [B, N, K]         [B, M, N]
 *
 * (*) for KGrouped, the caller should call A.transpose(0, 1) and B.transpose(0, 1)
 *     before passing to this function, ...
```

注意表中的 `[M, K]`/`[N, K]` 是**逻辑形状**：调用方（或 API 内部）先做 `transpose(0, 1)`，把物理上的 `[K, M]`/`[K, N]` 在元信息层面翻成 `[M, K]`/`[N, K]`，数据不动。`k_grouped_bf16_gemm_tn_contiguous` 的 docstring（`:501-514`）给出物理契约：`a: [K, M] (tn), bf16`、`b: [K, N] (tn)`、`d: [G, M, N]`、`ks_cpu: optional [G], int (on CPU). Per-group K sizes`、`grouped_layout: [G], int32. Per-group layout describing the contiguous K partition`、`c: optional [G, M, N] ... computes D = A @ B + C`。

**psum 布局的精确语义**（`scheduler.hpp:25-31` 注释）：

- A `[total_k, M]`、B `[total_k, N]` 是平坦缓冲，各组 K 段依次拼接；`grouped_layout[i]` = 前 i+1 组**实际 K** 的累积结尾（psum 约定）；
- 组 g 的有效 K 区间 = `[align(psum[g-1], 256), psum[g])`，组 0 从 K=0 开始；`[psum[g], align(psum[g], 256))` 是组间 padding；
- `total_k = align(psum[G-1], 256)`；D 是 `[G, M, N]` 逐组紧凑，无 padding；
- 类注释写"each group's K aligned to BLOCK_K"，代码实际对齐到 `kAlignment`（= 256，launch 传入，`csrc/jit_kernels/bf16_gemm.hpp:65`）；由于组尾有 `actual_k` 截断兜底（3.5 节），BLOCK_K 是否整除 256 都不影响正确性。

**手算贯穿全天的例子**：设 `G = 3`，三组实际 K 为 `[500, 256, 1000]`，对齐粒度 256。按 `generate_psum_layout`（`generators.py:212-220`，`sizes` 参数即逐组实际 K）：

```text
组 0：psum=0   → layout[0] = 0+500 = 500；下一组起点 = align(500,256) = 512
组 1：psum=512 → layout[1] = 512+256 = 768；下一组起点 = align(768,256) = 768
组 2：psum=768 → layout[2] = 768+1000 = 1768；total_k = align(1768,256) = 1792

grouped_layout = [500, 768, 1768]      # int32, 长度 3
A: [1792, M]（MN-major）  B: [1792, N]（MN-major）  D: [3, M, N]
```

全局 K 空间的划分（`▓` 有效，`·` padding）：

```text
K 轴（total_k = 1792）
 0────────500   512────────768   768──────────────────1768   1792
 ├─ 组 0 ──┤··· ├── 组 1 ──┤     ├───── 组 2 ────────┤ ···· │
 │  K=500  │212 │   K=256  │     │       K=1000       │ 124  │
 ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
 组起点 last_psum_k：0（组0）  512（组1）  768（组2）        组起点必为 256 倍数
 SF 基址 sf_idx_base = last_psum_k/64：0    8     12         组尾 psum[g] 不必对齐
```

注意与昨天对照的两个坑：

1. `grouped_layout[i]` 仍是**累积结尾**不是组大小；组 g 的 K 长度 = `psum[g] - align(psum[g-1], 256)`。
2. "K"出现在两个层面：`GemmDesc.k` 是**含 padding 的总 K**（本例 1792，launch 时作为 `shape_k_runtime` 传入，`csrc/jit_kernels/bf16_gemm.hpp:77`），而每组实际参与计算的是 `current_shape_k`（500/256/1000）。

### 3.3 API 层：`k_grouped_*_tn_contiguous` 与 expected_k

Python 侧入口是 `deep_gemm.k_grouped_bf16_gemm_tn_contiguous`（`deep_gemm/__init__.py:68`）与 `k_grouped_fp8_gemm_tn_contiguous`（`:57`；`:56` 的 `_nt_` 已废弃）。它们来自 `register_apis` 的 pybind 绑定（`csrc/apis/gemm.hpp:670-676, :607-614`），`compiled_dims` 默认 `"mn"`。

BF16 版本的 Host 实现（`gemm.hpp:515-550`）：

```cpp
static void k_grouped_bf16_gemm_tn_contiguous(
        const torch::Tensor& a_, const torch::Tensor& b_,
        const torch::Tensor& d,
        const std::optional<std::vector<int>>& ks_cpu,
        const torch::Tensor& grouped_layout,
        const std::optional<torch::Tensor>& c, ...) {
    // here we use transpose to make the input tensors at [m, k] shape
    // this is only for the purpose of getting the correct shape, ...
    auto a = get_transposed<true>(a_, 0, 1);        // [K, M] → 逻辑 [M, K]
    auto b = get_transposed<true>(b_, 0, 1);        // [K, N] → 逻辑 [N, K]
    auto [g, m, n, k] = get_shape_by_spec<"gmnk">("mk"_sp = a, "nk"_sp = b,
                                                  "gmn"_sp = d, "g"_sp = grouped_layout);
    DJ_HOST_ASSERT(use_psum_layout == true);
    auto expected_k = expected_k_from_ks(ks_cpu, k / g);
    auto maybe_desc = get_gemm_desc(
        GemmType::KGroupedContiguousWithPsumLayout,
        a, b, d, c, ..., grouped_layout, compiled_dims, false, std::nullopt, expected_k);
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_bf16_gemm(maybe_desc.value());
}
```

七个要点：

1. **逻辑转置**：`get_transposed<true>` 只交换 shape/stride 元信息（对 `(data, sf)` 元组会同时转置两者，见 `csrc/utils/layout.hpp` 的实现与 `tests/common.py:42-46` 的同名逻辑），数据与 major 不变。`[K, M]` 行-major 张量转置后即是"MN-major 的 `[M, K]`"——这就是 `get_gemm_desc` 中 `DJ_HOST_ASSERT(major_a == Major::MN)`（`gemm.hpp:94-95`）的由来：**K-Grouped 的 A、B 都必须 MN-major**。直观理由：组边界切在 K 维，MN-major（M/N 连续、K 为外层维）时每个 k 行是连续存储，组边界恰好落在行的整数倍处；若 K-major，组边界会切断行内连续段，"组"不再对应连续存储。这与昨天"A 必须 K-major"（组边界在 M 维）完全轴对称。
2. **形状推导**：K-Grouped 分支（`gemm.hpp:92-96`）四个张量标注 `"mk"/"nk"/"gmn"/"g"`——D 是三维 `"gmn"`，所以 **k 只能来自 A/B**（且是含 padding 的总 K），g 来自 D 的第 0 维与 layout 长度，m 来自 A/D，n 来自 B/D。
3. **`ks_cpu` 是纯 Host 提示**：`expected_k_from_ks`（`:35-43`）对非空列表取整数均值（下限 1），空/缺省回退到调用方给的 `k / g`。它只写入 `GemmDesc.expected_k`（`:187`）供启发式使用，**不进 Kernel**——设备侧唯一权威是 `grouped_layout`（Kernel 参数，`csrc/jit_kernels/bf16_gemm.hpp:77`）。
4. **accumulate 是支持的**：`c` 是可选的 `[G, M, N]`、与 D 同 dtype，语义 `D = A @ B + C`（docstring `:347, :512`）。`GemmDesc.acc = c.has_value() && !with_bias`（`:180`）；测试侧先 `d.copy_(c)` 再传 `c=d`（`tests/common.py:56-58`），Kernel 用 `kWithAccumulation` 模板处理（3.5 节）。对照昨天：M-Grouped 明确不支持（`generators.py:330` 断言），因为其输出的 padding 行语义与 atomic add 冲突；K-Grouped 每组输出是干净的 `[M, N]`，无此问题。bias/alpha 仍只对 Normal 开放（`:107`）。
5. **废弃的 nt 变体**：`k_grouped_fp8_gemm_nt_contiguous`（`:386-411`）接收 `[M, K]`/`[N, K]`，转置后转发给 tn 版本，并发出 `DeprecationWarning`；BF16 连 nt 变体都没有。测试因此对 k-grouped 只测 `tn`（`generators.py:295-296`）。
6. **低精度分发**：fp8 版 API 名字里只有 "fp8"，实际按 dtype 分发到 fp8×fp8 / fp4×fp4 / fp8×fp4 三个 kernel（`dispatch_fp8_fp4_gemm`，`gemm.hpp:22-33`）；sfa/sfb 都走 `transform_k_grouped_sf_into_required_layout`（`:129-143`，实现在 `csrc/apis/layout.hpp:94-111`，细节见第 33 天）。
7. **`use_psum_layout` 必须为 true**（`:535`），与 M-Grouped 一样是为兼容上游 API 保留的开关。

### 3.4 调度层：共享 M×N 网格 + K 组游标

`Scheduler`（`scheduler.hpp:42-249`）的 persistent block index 机制四种 GemmType 通用：`block_idx = (++current_iter) * kNumCores + core_id`（`:142`），`kNumCores = 32`（`common.hpp:51`）。

K-Grouped 与 M-Grouped 的调度差异**源于输出形态**：

- M-Grouped：D 平坦、每组行数不同 → 每组有**自己的** `num_m_blocks × num_n_blocks` 网格，组间块数不等；
- K-Grouped：D 逐组、M/N 对所有组相同 → **所有组共享同一个 M×N 网格**（构造函数不改写 `num_blocks`，`:91-93`），每个非空组恰好消耗 `num_blocks` 个任务。

K-Grouped 专属游标（`:72-77`）：

| 成员 | 含义 | 初始值（构造函数 `:103-107`） |
|---|---|---|
| `current_group_idx` | 最近返回的 block 所属的组 | 0 |
| `last_psum_k` | 当前组起始 K = `align(psum[g-1], 256)` | 0 |
| `current_psum_k` | 当前组有效结尾 K = `psum[g]` | `grouped_layout[0]` |
| `current_shape_k` | 当前组 K 长度 = 两者之差 | `grouped_layout[0]` |
| `current_sf_k_cumsum` | SF 的 K 基址 = `last_psum_k / 64` | 0 |
| `current_k_block_cumsum` | 之前的组**已消耗的 block 总数**（按整份网格累计） | 0 |

`get_next_block` 的 K-Grouped 分支（`:163-177`）逐行解读：

```cpp
while (true) {
    if (current_shape_k > 0 and block_idx < current_k_block_cumsum + num_blocks)
        break;                                            // ① 落在当前组（非空且未越界）
    if (++current_group_idx == num_groups)
        return false;                                     // ② 越过最后一组：任务耗尽
    last_psum_k = aligned(current_psum_k, kAlignment);    // ③ 新组起始 K（256 对齐）
    current_psum_k = grouped_layout[current_group_idx];   //    新组有效结尾
    current_sf_k_cumsum = last_psum_k / MX_SF_DIVISOR;    // ④ SF 基址（K/64，MX_SF_DIVISOR 见 common.hpp:65）
    if (current_shape_k > 0)                              // ⑤ 旧组非空才累计块数
        current_k_block_cumsum += num_blocks;
    current_shape_k = current_psum_k - last_psum_k;       // ⑥ 新组 K 长度
}
uint32_t in_group_idx = block_idx - current_k_block_cumsum;          // ⑦ 组内线性下标
get_swizzled_block(in_group_idx, num_m_blocks, m_block_idx, n_block_idx);  // ⑧ 全网格 swizzle
return true;
```

- **①⑤ 的空组守卫**：空组（`k_begin == k_end` → `current_shape_k == 0`）让 break 条件立即失败，于是 ③④⑤⑥ 推进到下一组；由于旧组 `current_shape_k == 0`，⑤ 不累计 `current_k_block_cumsum`——**空组不消耗任何 block**，后续组的组内下标仍从整份网格边界起算。对比昨天：M-Grouped 的空组占 0 行也被调度跳过，但 K-Grouped 的空组**仍占一份 `[M, N]` 输出**，需要 AIV 清零（3.5 节）。
- **⑦⑧**：`in_group_idx` 归零到组内坐标系后，走与 Normal **完全相同**的全网格 swizzle（`get_swizzled_block`，`:202-248`；注意此处 `num_m_blocks` 是全网格值，不像 M-Grouped 传组内值），**不需要任何平移**——M/N 坐标在所有组里含义一致。
- **游标只进不退**：单核 block_idx 单调递增，从缓存状态向前推进 cumsum 一定正确（`:33-35` 注释）。

**推演 3.2 的例子**（补 `M=4096, N=7168, BLOCK_M=128, BLOCK_N=256` → `num_m_blocks=32, num_n_blocks=28, num_blocks=896`；三组均非空 → 总 block = `3 × 896 = 2688`，`kNumCores=32` 时每核恰好 84 块）。追踪 **core 0** 的 block_idx 序列 `0, 32, 64, …`：

| block_idx | 组判定过程 | 落点 | `get_group_idx()` |
|---|---|---|---|
| 0 | `0 < 0+896` ✓ 组 0 | in_group=0，swizzle 后 (m,n) 组内块 0 | 0 |
| 864 | `864 < 896` ✓ 组 0 | 组 0 的最后一块 | 0 |
| 896 | `896 < 896` ✗ → 推进：`last_psum_k=512, current_psum_k=768, sf_cumsum=8, cumsum=896, shape_k=256`；`896 < 896+896` ✓ 组 1 | in_group=0 | 1 |
| 1792 | 推进：`last_psum_k=768, current_psum_k=1768, sf_cumsum=12, cumsum=1792, shape_k=1000`；`1792 < 1792+896` ✓ 组 2 | in_group=0 | 2 |
| 2656 | `2656 < 2688` ✓ 组 2 | core 0 的最后一块（第 84 块） | 2 |
| 2688 | 推进时 `++current_group_idx == 3 == num_groups` → **return false** | — | — |

同一个核的任务会**跨组**：core 0 依次在组 0 算 28 块、组 1 算 28 块、组 2 算 28 块。**换组意味着 K 循环长度、A/B/SF/D 的基址全部切换**——这正是 Kernel 层每次 `get_next_block` 后都要重新取 `current_shape_k / k_idx_base / sf_idx_base / group_idx` 的原因。

**空组变体**再推一步：若三组实际 K 改为 `[0, 256, 1000]`，则 `grouped_layout = [0, 256, 1256]`、`total_k = 1280`。构造时 `current_shape_k = 0`，第一次 `get_next_block` 即在 ① 失败并推进：`last_psum_k = align(0,256) = 0, current_psum_k = 256, sf_cumsum = 0`，⑤ 因旧组空而**不累计** → 组 1 的 block 从 `in_group = block_idx - 0` 开始。组 0 的 `D[0]` 没有任何 block 会写——留给 AIV 清零。

### 3.5 Kernel 层：K 偏移给 A/B，组偏移给 D

进入 `bf16_gemm_impl`（`bf16_gemm.hpp:27-193`），K-Grouped 专属逻辑集中在 AIC 主循环开头（`:95-97`）与 K 循环（`:112-118`）：

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    const uint32_t m_idx = scheduler.get_global_idx(BLOCK_M, m_block_idx);
    const uint32_t n_idx = scheduler.get_global_idx(BLOCK_N, n_block_idx);
    const uint32_t eff_shape_k = is_k_grouped ? scheduler.current_shape_k : shape_k;  // ★ 每组 K 长度
    const uint32_t num_k_blocks = ceil_div(eff_shape_k, BLOCK_K);                    // ★ 组内 K 块数
    const uint32_t k_idx_base = scheduler.get_k_idx_base();                          // ★ 组起始 K
    ...
    const auto b = gm_b.batch(batch_idx).group(b_group_idx, shape_n, shape_k);        // b_group_idx ≡ 0（k-grouped）
    ...
    for (uint32_t k_blk_idx = 0, s = 0; k_blk_idx < num_k_blocks; k_blk_idx++, ...) {
        const uint32_t k_idx = k_idx_base + k_blk_idx * BLOCK_K;                      // ★ 组偏移 + 组内偏移
        const uint32_t actual_k = min(eff_shape_k - k_blk_idx * BLOCK_K, BLOCK_K);    // ★ 组尾截断
        ...
        copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, kL2CtrlA);   // A：平坦，靠 k_idx 平移
        copy_gm_to_l1(l1b[s], b.index(n_idx, k_idx), actual_n, actual_k, kL2CtrlB);   // B：同上
```

四个关键点：

1. **A/B 的组偏移走 `index()` 而非 `group()`**：与昨天正好相反。A/B 是沿 K 拼接的**单块平坦缓冲**，`gm_ptr::group()`（`common.hpp:167-172`）的 `g × N × K` 公式对它们没有意义（`b_group_idx = 0`，`:100`）；组偏移体现在 `k_idx_base` 上。以 MN-major、`stride_outer = M`（紧存储 `[K, M]`）为例，`a.index(m_idx, k_idx)` 的地址 = 基址 + `sizeof(T) × (k_idx × M + m_idx)`（`common.hpp:156-165`），于是 `k_idx_base × M × sizeof(T)` 就是组 g 在 A 中的字节偏移。fp8 路径同理且把基址取出复用（`fp8_gemm.hpp:137-140`：`a = gm_a.batch(batch_idx).index(m_idx, k_idx_base)`，循环内再 `a.index(0, k_idx)`）。
2. **SF 的组偏移是 `sf_idx_base`**：SF 是 MN-major 的 `[mn, ceil(k/64)]` int16，`gm_sfa.index(m_idx, sf_idx_base)`（`fp8_gemm.hpp:139-140`）把列下标平移 `last_psum_k / 64`（`MX_SF_DIVISOR = 64`，`common.hpp:65`）——这就是调度器维护 `current_sf_k_cumsum` 的目的。
3. **D 的组偏移走 `group()`**：D 是逐组紧凑的 `[G, M, N]`，`d_group_idx = scheduler.get_group_idx()`，`gm_d.batch(batch_idx).group(d_group_idx, shape_m, shape_n)`（direct store 路径 `:157`，AIV drain 路径 `:186-188`；fp8 同款 `fp8_gemm.hpp:236, :281-282`）→ 地址加 `g × M × N × sizeof(dtype)`。fp8_dequant 路径模式相同（`fp8_dequant_gemm.hpp:328, :436`）。
4. **非对齐组尾**：`actual_k = min(eff_shape_k - k_blk_idx × BLOCK_K, BLOCK_K)`（`:114`）把组内最后一个 K 块截断到真实长度，组间 padding 的 K 行**不参与计算**（`copy_gm_to_l1` 的 MN-major 分支按 `actual_k` 行搬运，`ascend/copy.hpp:125-130`）。fp8 路径里 SF 按 64 对齐的缺口由 `fill_tail_block` 在 L1 内补零而不读 GM（`fp8_gemm.hpp:160-170`）。测试生成器仍把组间 padding 数据清零并注释 "k-group is always zero-padded between groups"（`generators.py:379-382`）——这是对调用方的契约要求与对齐读的双重保险。对照昨天：M-Grouped 的组尾 m-block 是**连 padding 整块算**（`get_actual_m` 恒 `BLOCK_M`），K-Grouped 的组尾 k-block 是**截断**——两种策略都正确，差别在 padding 行/列是否进入输出语义。

**空组输出清零：`aiv_zero_empty_groups`**（`epilogue.hpp:108-141`）。AIV 分支在 drain 循环之前调用（仅当 `is_k_grouped && !kWithAccumulation`，`bf16_gemm.hpp:175-176`；fp8/fp8_dequant 同款 `:269-270`、`:425-426`）：

```cpp
const uint32_t aiv_idx = block_idx * kNumAIVs + asc_get_sub_block_id() - (2 - kNumAIVs);
const uint32_t num_n_tiles = ceil_div(shape_n, MAD_N);
const uint32_t num_tiles = ceil_div(shape_m, MAD_M) * num_n_tiles;   // 每组 [M,N] 的 MAD tile 数
bool initialized = false; uint32_t k_begin = 0;
for (uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    const uint32_t k_end = grouped_layout[group_idx];
    const bool is_empty = k_begin == k_end;                          // 空组判定
    k_begin = aligned(k_end, kAlignment);
    if (!is_empty || aiv_idx >= num_tiles) continue;
    if (!initialized) { zero_values(...); ...; initialized = true; }  // UB 里备一份全零 tile
    for (uint32_t tile_idx = aiv_idx; tile_idx < num_tiles; tile_idx += kNumCores * kNumAIVs)
        copy_ub_to_gm(d.group(group_idx, shape_m, shape_n).index(...), output_ub, ...);  // 整组铺零
}
```

要点：全零 tile 在 UB 中只构造一次（`zero_values`，`:99-106`），随后所有 AIV 以 `kNumCores × kNumAIVs` 为步长把每个空组的整个 `[M, N]` 铺零；FIX/MTE3/VECTOR 之间用事件同步保护共享 UB（`:136-140`）。测试用 NaN 预填 D 来验证这条路径真的执行了（3.7 节）。

**累加语义**：`kWithAccumulation = desc.acc` 进入模板（`csrc/jit_kernels/bf16_gemm.hpp:61`）。AIV 的 `aiv_initialize` 在累加模式下设 atomic add（bf16 用 `asc_set_atomic_add_bfloat`，fp32 用 float，`epilogue.hpp:91-96`），drain 的 `copy_ub_to_gm` 于是变成"加到 D 上"而不是覆盖——调用方须先把 C 拷进 D（`tests/common.py:56-58`）。空组在累加模式下**不需要**任何处理：没有任何 block 会碰 `D[g]`，它保持 `C[g]`，恰是 `0 @ B + C` 的期望结果——这就是清零调用带 `!kWithAccumulation` 守卫的原因。

### 3.6 配置层：expected_k 的三副面孔

启发式（`csrc/jit_kernels/heuristics.hpp`）在 K-Grouped 下用两个不同的"K"建模：

- **`compute_k = expected_k`（组内 K）**：单个 block 的计算量、L1 常驻容量、fp4 反量化字节数都按"一个组"估（`:598`、`:214`、`:512`、`:545`）；
- **`k = num_groups × expected_k`（总 K）**：SF 相关的容量/流量按"所有组"估（`:597`、`:158`、`:178`）；
- **`num_batches = num_groups`**：block 总数按"每份网格 × G"计（`:599`、`:191-193`、`:218-219`），M/N 网格本身不变。

这些全部依赖 `expected_k` 这个 **Host 侧估计值**（来自 `ks_cpu` 均值或 `k/g`）——因为逐组真实 K 只存在于设备侧的 `grouped_layout` 里，Host 选配置时拿不到。估计偏差只影响配置优劣，不影响正确性。

与昨天对比的三条差异：

1. **没有 BLOCK_M 钉死**：`block_m != 256` 的过滤只作用于 m-grouped（`:104-106`），m_range 固定也只发生在 m-grouped（`:610-616`）。K-Grouped 的分组轴不在 M 上，BLOCK_M 可以自由搜索——组边界在 K 维，与 M/N block 划分正交。
2. **`compiled_dims` 默认 `"mn"`**（绑定 `gemm.hpp:612, :674`）：M/N 烘成编译期常量，K 留运行时——总 K 与逐组 K 都随调用变化，且 Kernel 内部以 `eff_shape_k` 为准，编译期钉死任何一个 K 都是错的。
3. **direct store 仅限 Normal**（`can_direct_store`，`:526`）：K-Grouped 恒走 AIC store + AIV drain（epilogue 需要处理空组清零与累加）。resident 路径：BF16 kernel 有 `static_assert(!(kResidentA || kResidentB) || (kGemmType == GemmType::Normal && SHAPE_K))`（`bf16_gemm.hpp:40`），BF16 的常驻仅限 Normal；fp8 的常驻对 k-grouped 开放，容量按 `resident_k = expected_k` 估（`heuristics.hpp:545-546, :558-562`）——单个 block 在一个组的 K 循环内把 A 或 B 整份留在 L1。

`GemmConfig` 打印时对 k-grouped 额外输出 `num_groups` 与 `expected_k`（`gemm_config.hpp:154-155`），可用 `DG_PRINT_CONFIGS=1` 观察（README:89）。

### 3.7 测试视角：三种变体与三档 Host 提示

**用例空间**（`generators.py:616-649`）：

```python
k_grouped_list = [
    ( 32, 4, 4, 1),                                        # 最小/边界
    (128, 4, 4, 1), ( 4, 4096, 7168, 1),
    ( 4, 4096, 7168, 8192), ( 4, 7168, 2048, 8192),        # EP64
    ( 8, 4096, 7168, 4096), ( 8, 7168, 2048, 4096),        # EP32
    (16, 4096, 7168, 2048), (16, 7168, 2048, 2048),        # EP16
]
cd_options = [(True, torch.float32), (False, torch.float32), (False, torch.bfloat16), (True, torch.bfloat16)]
```

每个 `(G, M, N, expected_k)` 先随机化逐组实际 K（`uniform(0.7, 1.3) × expected_k`，`:632`），再经 `enumerate_k_grouped_contiguous_test_variants`（`:606-613`）派生三种变体：

- **normal**：原样（可能含随机小扰动）；
- **empty**：约 25% 的组置零，且**强制首组为空、末组非空**（`empty_ks_cpu[0], empty_ks_cpu[-1] = 0, real_ks_cpu[-1]`）——专门打击 `aiv_zero_empty_groups` 与调度空组守卫；
- **single**：只有中间一组非空，其余全空——极端空组比例。

再与 4 种 (accumulate, cd_dtype) × 3 档 `host_ks_cpu` 组合：`'list'`（逐组**对齐后**的 K 列表）、`'none'`（None）、`'empty'`（[]），验证 `expected_k_from_ks` 的全部分支。合计 `8 × 3 × 4 × 3 = 288` 个 `GemmTestDesc`；只有 `normal` + `list` 变体会做性能测量（`:648`）。

**数据生成**（`_generate_k_grouped_contiguous`，`:361-399`）七步：

1. `generate_psum_layout(expected_k, G, 256, device, self.ks_cpu)`：以 `ks_cpu` 为逐组实际 K 构造 psum（`:363`）；
2. accumulate 时生成 `c [G, M, N]`（`:365-370`）；
3. 断言 `a_major == Major.MN`、`b_major == Major.MN`（`:372-373`，与 API 对称）；
4. 生成 A `(total_k, m)`、B `(total_k, n)`，`major_dim=1`（MN 连续），recipe 按 `(gran_k, gran_m)` 反转传入（`:374-377`）；
5. **组间 padding 清零**（`:379-382`，"k-group is always zero-padded between groups"）；
6. SF 的 padding 填有限随机垃圾（`:384-389`，`iter_psum_sep` 按 `gran_k=32` 切）；
7. 参考输出：逐组 `d[i] = (c[i] if accumulate else 0) + (a_repr[ks].float().T @ b_repr[ks].float()).to(cd_dtype)`（`:391-397`），`cmp_spans = None` 全量比较（`:399`）。

**调用与验证**（`tests/common.py`）：

- `gemm_dg` 的 k-grouped 分支只允许 `tn`，统一传单个 `recipe`，API 名按有无 SF 选 `fp8`/`bf16`（`:74-82`）；
- 无 ACLNN 对照（`:87-90`，分组算子无官方参考），正确性全靠 `d_ref`；
- **NaN 预填**：非累加用例 `d = torch.full_like(d_ref, float('nan'))`（`:190`）——若空组清零缺失，NaN 残留即断言失败；
- **fp32 直出对照**：每个非累加用例额外跑一遍 `accumulate=True, cd_dtype=float32` 的变体，`assert_direct_output_matches_fp32_accumulation`（`utils.py:61-67`）要求 bf16 直出与"fp32 累加再 cast"**逐位相等**（`:165-168, :202-206`）；随机舍入用例同样借用 fp32 累加变体（`:218`）；
- 性能行的 K 用 `sum(test.ks_cpu)`（真实总计算量，`:223-224`）。

## 四、源码导读

今天的主线文件与阅读站点（建议按表中顺序读）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `csrc/apis/gemm.hpp` | 形状注释 `:45-58`、`expected_k_from_ks` `:35-43`、K-Grouped case `:92-96`、SF 变换 `:129-143`、`estimated_k` `:161-163`、GemmDesc 填充 `:164-196`（expected_k `:187`、acc `:180`）、`k_grouped_fp8_gemm_tn_contiguous` `:351-384`（docstring `:326-350`）、废弃 nt `:386-411`、`k_grouped_bf16_gemm_tn_contiguous` `:515-550`（docstring `:501-514`）、绑定 `:607-622, :670-676` | 今天的主线 Host 逻辑 |
| 2 | `deep_gemm/include/deep_gemm/scheduler.hpp` | 类注释 `:25-35`、游标成员 `:72-77`、构造函数 `:103-107`、`get_k_idx_base/get_sf_idx_base` `:124-125`、`get_next_block` k-grouped 分支 `:163-177`、`get_swizzled_block` `:202-248` | 调度核心，见下方逐段导读 |
| 3 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | resident 断言 `:40`、`eff_shape_k/k_idx_base` `:95-97`、K 循环 `:112-118`、direct store D `:157`、`aiv_zero_empty_groups` 调用 `:175-176`、AIV drain `:180-191` | K 偏移与组偏移各自落到谁头上 |
| 4 | `deep_gemm/include/deep_gemm/fp8_gemm.hpp` | `gm_sfa/gm_sfb` `:63-66`、`eff_shape_k/k_idx_base/sf_idx_base` `:130-133`、取址 `:137-140`、`k64/fill_tail_block` `:159-170`、D group `:236, :281-282`、空组清零 `:269-270` | SF 的组偏移与非对齐补零 |
| 5 | `deep_gemm/include/deep_gemm/epilogue.hpp` | `aiv_initialize`（atomic add）`:89-97`、`zero_values` `:99-106`、`aiv_zero_empty_groups` `:108-141` | 空组清零与累加的实现 |
| 6 | `deep_gemm/include/deep_gemm/common.hpp` | `MX_SF_DIVISOR` `:65`、`GemmType` `:87-92`、`is_k_grouped` `:108-110`、`gm_ptr::index` `:156-165`、`group` `:167-172` | 地址公式与常量 |
| 7 | `csrc/jit_kernels/heuristics.hpp` | m-grouped 过滤 `:104-106`（k-grouped 无此规则）、`k=num_groups×expected_k` `:158, :178, :597`、`compute_k` `:598`、`num_batches` `:191-193, :599`、compute cost `:214-219`、dequant `:508-512`、direct store `:526`、resident `:544-562` | expected_k 的三副面孔 |
| 8 | `csrc/jit_kernels/gemm_config.hpp` | `grouped_layout/expected_k` 字段 `:56-60`、formatter `:154-155` | 描述符与打印 |
| 9 | `csrc/jit_kernels/bf16_gemm.hpp` | kAlignment `:65`、launch 传参 `:73-77`（k=总 K）、compiled dims `:45-47` | 256 对齐与运行时 K 如何进 Kernel |
| 10 | `tests/generators.py` | `generate_psum_layout` `:212-220`、`iter_psum_sep` `:230-235`、`get_available_trans_ab` `:295-296`、`_generate_k_grouped_contiguous` `:361-399`、变体 `:606-613`、用例枚举 `:616-649` | 三变体 × 三档提示的用例空间 |
| 11 | `tests/common.py` | `gemm_dg` k-grouped 分支 `:74-82`、accumulate 断言 `:56`、NaN 预填 `:190`、fp32 对照 `:165-168, :202-206`、perf 的 k `:223-224` | 验证差异 |
| 12 | `tests/utils.py` | `assert_direct_output_matches_fp32_accumulation` `:61-67` | 逐位相等断言 |
| 13 | `tests/test_bf16_gemm.py` / `test_fp8_gemm.py` | CLI `:25/:29`、kgrouped `:41-42`/`:45-46` | 选择子 |
| 14 | `csrc/apis/layout.hpp` | `transform_k_grouped_sf_into_required_layout` `:94-111` | SF 布局变换入口（细节见第 33 天） |
| 15 | `deep_gemm/include/deep_gemm/fp8_dequant_gemm.hpp` | `:190, :226, :328, :373, :425, :436` | 第三条 kernel 路径的同一模式 |

**`Scheduler` K-Grouped 相关逐段导读**（行号见 `deep_gemm/include/deep_gemm/scheduler.hpp`）：

- **`:25-31`**：权威语义注释——A/B 平坦、D 逐组、psum 约定、共享 M×N 网格、组间差异只有 `current_shape_k`、A/B 用 `get_k_idx_base()`、D 用 `get_group_idx() × shape_m × shape_n`、SF 用 `get_sf_idx_base()`。
- **`:33-35`**：block_idx 单调递增 ⇒ cumsum 游标只需前进。
- **`:72-77`**：五个 k-grouped 专属成员（3.4 节表格）。
- **`:103-107`**：构造时只初始化组 0 的 `current_psum_k/current_shape_k`；`num_blocks`/`num_m_blocks`/`num_n_blocks` 保持全网格值（`:91-93`）——与 m-grouped 改写 `num_m_blocks`（`:102`）形成对照。
- **`:124-125`**：`get_k_idx_base()` 返回 `last_psum_k`（元素偏移基数）；`get_sf_idx_base()` 返回 `current_sf_k_cumsum`（SF 对数偏移基数）。
- **`:163-177`**：`get_next_block` k-grouped 分支（3.4 节逐行讲解）。
- **`:202-248`**：`get_swizzled_block` 与 Normal 共用；k-grouped 传入的 `local_num_m_blocks` 就是全网格 `num_m_blocks`。

**`_generate_k_grouped_contiguous` 逐段导读**（`tests/generators.py:361-399`）：

- `:363`：psum 用 `ks_cpu`（**未对齐**的实际 K）构造——对齐发生在布局里而不是 ks 里；
- `:374-377`：A/B 形状 `(total_k, m/n)`、`major_dim=1`、`recipe[::-1]`（形状的轴序与逻辑 `[M, K]` 相反，gran 也随之反转）；
- `:395-397`：参考输出 `a_repr[ks].T @ b_repr[ks]`——`a_repr[ks]` 是 `[k_g, M]`，转置后正是 `A_g^T B_g`；
- `:399`：返回的 shape 是 `(m, n, total_k)`——第三维是**总 K**，与 `GemmDesc.k` 一致。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通 K-Grouped BF16 正确性测试**：

   ```bash
   python tests/test_bf16_gemm.py kgrouped --skip-prof
   ```

   确认 `Num tests: 288`（8 组形状 × 3 变体 × 4 输出选项 × 3 档 ks 提示）。随机挑 3 行输出，说出它的 (G, M, N, expected_k)、变体（normal/empty/single）与 `ks-cpu` 提示档位；对照 3.7 节解释该用例重点考察什么（如 empty 变体 → 空组清零）。

2. **观察配置与 expected_k**：

   ```bash
   export DG_PRINT_CONFIGS=1
   python tests/test_bf16_gemm.py kgrouped --skip-prof
   ```

   核对两点：k-grouped 的 `GemmDesc` 打印里带 `num_groups, expected_k`（`gemm_config.hpp:154-155`）；`GemmConfig` 的 `block_m` **不再恒为 256**（与昨天 mgrouped 的输出对比）。再观察同一形状下 `ks-cpu=list/none/empty` 三档提示选出的配置是否一致，体会"提示只影响配置选择"。

3. **Python 实验：不均匀分组 + 空组 + 累加**：

   ```python
   import torch, torch_npu
   import deep_gemm

   M, N = 512, 256
   ks = [500, 256, 1000]                       # 3 组不均匀
   layout, psum = [], 0
   for k in ks:
       layout.append(psum + k)
       psum = (psum + k + 255) // 256 * 256
   total_k = psum                               # 1792
   g_layout = torch.tensor(layout, dtype=torch.int32, device='npu')   # [500, 768, 1768]

   a = torch.randn(total_k, M, dtype=torch.bfloat16, device='npu')    # [K, M]，天然 MN-major
   b = torch.randn(total_k, N, dtype=torch.bfloat16, device='npu')
   a[500:512] = 0; a[1768:1792] = 0            # 组间 padding 清零（API 契约）
   b[500:512] = 0; b[1768:1792] = 0
   d = torch.full((3, M, N), float('nan'), dtype=torch.bfloat16, device='npu')  # NaN 抓未写区域

   deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks, g_layout)
   torch.npu.synchronize()

   spans = [(0, 500), (512, 768), (768, 1768)]
   ref = torch.zeros(3, M, N, dtype=torch.bfloat16, device='npu')
   for g, (lo, hi) in enumerate(spans):
       ref[g] = a[lo:hi].T @ b[lo:hi]
   print(torch.equal(d, ref))                   # 随机舍入等差异极小，可再用 calc_diff

   # 累加：D = A@B + C
   c = torch.randn(3, M, N, dtype=torch.bfloat16, device='npu')
   d2 = c.clone()
   deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a, b, d2, ks, g_layout, c=d2)
   torch.npu.synchronize()
   ref2 = ref + c
   print(torch.equal(d2, ref2))
   ```

   再把 `ks` 改成 `[0, 256, 1000]` 重跑第一段（`grouped_layout=[0,256,1256]`），验证空组 `d[0]` 被清零而非残留 NaN；把 `c=d2` 换成独立的 `c`（不 alias d）观察 Host 断言行为。

4. **（可选）低精度路径**：`python tests/test_fp8_gemm.py kgrouped --skip-prof`，对照 `fp8_gemm.hpp:130-140` 确认 SF 组偏移（`sf_idx_base`）与 A/B 共用同一调度。

### B 组：无 Ascend 环境（静态任务）

1. **psum 手算**：设 `G=4`、各组实际 K `[100, 200, 0, 613]`、对齐 256。写出 `grouped_layout`（应为 `[100, 456, 512, 1125]`——注意空组的处理）、`total_k`（1280）、每组有效/padding 区间、每组的 `last_psum_k/current_psum_k/current_shape_k/sf_idx_base`（组 3 应为 512/1125/613/8）。对照 `generate_psum_layout` 核对。
2. **调度推演**：接 3.2 的例子（`grouped_layout=[500,768,1768]`，`M=4096, N=7168, BLOCK_M=128, BLOCK_N=256`，`kNumCores=32`）。不看 3.4 节的表格，独立推演 **core 1** 的 block_idx 序列（1, 33, …, 2657，共 84 块）各属于哪一组、组游标何时推进；再推演空组变体 `grouped_layout=[0,256,1256]` 下 core 0 的前三次 `get_next_block`（应全部落在组 1，`current_k_block_cumsum` 保持 0）。
3. **K-block 演算**：组 0 的 `current_shape_k=500`、`BLOCK_K=256`。写出 `num_k_blocks=2`、两次迭代的 `k_idx`（0、256）与 `actual_k`（256、244），确认读取范围恰为 `[0,500)` 不触碰 padding；再算 fp8 情形下最后一个块的 `k64 = aligned(244, 64) = 256` 与 `fill_tail_block` 补零的范围（`[244, 256)`，对照 `fp8_gemm.hpp:159-170`）。
4. **地址计算**：bf16、`M=4096, N=7168, total_k=1792`、组 2（`last_psum_k=768`）。算：(a) D 的组偏移 `group(2, 4096, 7168)` = `2×4096×7168×2 = 117,440,512` 字节；(b) A 的组偏移 = `k_idx_base × stride_outer × sizeof` = `768 × 4096 × 2 = 6,291,456` 字节（MN-major，`stride_outer = M`）；(c) SF 的组偏移 = `sf_idx_base × shape_m × sizeof(int16)` = `12 × 4096 × 2 = 98,304` 字节（`shape_sf_k = ceil(1792/64) = 28`）。
5. **代码对照**：通读 `scheduler.hpp:163-177` 与 `bf16_gemm.hpp:95-118, :157, :175-188`，在纸上标出分组信息的流转：`grouped_layout`（GM 上的 int32 数组）→ Scheduler 游标 → `current_shape_k/k_idx_base/get_group_idx()/get_sf_idx_base()` → `index()` 平移 A/B/SF、`group()` 平移 D、`aiv_zero_empty_groups` 清零空组。数一数组号分别"转手"几次，与昨天的"只有 B 一次"对比。
6. **画对比表**：不看 3.8 节，独立完成**三种 GEMM 类型对比表**（当天产出），再对照修正。

## 六、当天产出

**三种 GEMM 类型对比表**——以 3.8 节的参考版为答案，自己合上教程画一版，至少覆盖：分组轴与典型场景、四种张量的形状与 major 约束、`grouped_layout` 的语义（行前缀和 vs K 前缀和）、调度网格的差异（每组独立网格 vs 共享网格复制 G 次）、组偏移落在谁头上（B 的 `group()` vs A/B 的 `index()` 平移 + D 的 `group()`）、空组与非对齐组的处理、accumulate/bias 支持差异、配置约束（BLOCK_M 钉死与否、compiled_dims）。写完后把画不出的格子标出来，回到第四节对应站点重读。有环境的同学附上 A 组第 2、3 步的配置打印与空组/累加实验记录。

## 七、自测题

1. K-Grouped 的四个张量形状是什么？A/B 的 major 约束是什么，为什么？
   > A `[K, M]`（tn）、B `[K, N]`（tn）、D `[G, M, N]`、`grouped_layout` `[G]` int32 contiguous，另有可选 `ks_cpu`（Host int 列表）与 `c [G, M, N]`。A/B 都必须 MN-major（`gemm.hpp:94-95`）：组边界在 K 维，MN-major 时 K 是外层维、每个 k 行连续，组边界落在行的整数倍处；K-major 会切断行内连续段（与 m-grouped 要求 A K-major 轴对称）。
2. `ks_cpu` 和 `grouped_layout` 各自的作用是什么？Kernel 用哪一个？
   > `ks_cpu` 是纯 Host 提示，经 `expected_k_from_ks`（均值、下限 1，缺省 `k/g`）写入 `GemmDesc.expected_k`，只影响启发式选配置；`grouped_layout` 是设备侧唯一权威，Kernel 从 GM 直读（`gemm.hpp:35-43, :373`；`csrc/jit_kernels/bf16_gemm.hpp:77`）。
3. 描述 `get_next_block` K-Grouped 分支的循环：什么条件下推进组？推进时更新哪五个量？空组如何被跳过？
   > `current_shape_k == 0` 或 `block_idx ≥ current_k_block_cumsum + num_blocks` 时推进；更新 `current_group_idx`（越界返回 false）、`last_psum_k = aligned(current_psum_k, 256)`、`current_psum_k = grouped_layout[新组]`、`current_sf_k_cumsum = last_psum_k / 64`、`current_k_block_cumsum += num_blocks`（仅当旧组非空）、`current_shape_k = 两者之差`。空组 `current_shape_k == 0` 使 break 与累计都被守卫跳过，不消耗任何 block（`scheduler.hpp:163-177`）。
4. 所有组共享什么？每组各自不同的是什么？组内 block 坐标需要平移吗？
   > 共享同一个 M×N block 网格（`num_blocks` 不改写，构造函数只初始化组 0 的 K 游标）；不同的是 K 循环长度 `current_shape_k` 与 A/B/SF/D 的基址。组内 `in_group_idx = block_idx - current_k_block_cumsum` 后直接走全网格 swizzle，**不平移**——M/N 坐标在所有组含义一致（`scheduler.hpp:178-179`）。
5. Kernel 里组信息分别如何影响 A、B、SF、D 的寻址？
   > A/B：`index(m_idx, k_idx_base + k_blk_idx×BLOCK_K)` 平移（MN-major 下即 `k_idx_base × stride_outer` 元素偏移，`bf16_gemm.hpp:113, :117-118`）；SF：`index(m_idx, sf_idx_base)`（`fp8_gemm.hpp:139-140`）；D：`group(d_group_idx, shape_m, shape_n)` 加 `g×M×N×sizeof`（`bf16_gemm.hpp:157, :186-188`）。A/B 不走 `group()`（`b_group_idx ≡ 0`），与 m-grouped 恰好相反。
6. 非对齐组尾怎么处理？和 M-Grouped 的组尾策略有何不同？
   > 组内最后一个 K 块由 `actual_k = min(eff_shape_k - k_blk_idx×BLOCK_K, BLOCK_K)` 截断，padding 的 K 行不参与计算；fp8 下 SF 按 64 对齐的缺口由 `fill_tail_block` 在 L1 补零（`bf16_gemm.hpp:114`、`fp8_gemm.hpp:159-170`）。M-Grouped 的组尾 m-block 则连 padding 整块计算（`get_actual_m` 恒 BLOCK_M），靠零 padding 保证输出正确——一个截断、一个整算。
7. 空组的输出谁来写？为什么累加模式下不需要这一步？
   > 非累加时 AIV 的 `aiv_zero_empty_groups` 把空组整个 `[M, N]` 铺零（`epilogue.hpp:108-141`，调用点 `bf16_gemm.hpp:175-176`）；测试用 NaN 预填 D 验证。累加模式下 `kWithAccumulation` 使 drain 变 atomic add，调用方预先 `d.copy_(c)`，空组的 `D[g]` 保持 `C[g]` 恰是 `0@B + C`，因此清零带 `!kWithAccumulation` 守卫。
8. K-Grouped 支持 accumulate 吗？M-Grouped 呢？为什么有这个差别？
   > 支持：`c [G, M, N]` 同 dtype，`D[g] = A_g@B_g + C[g]`（`gemm.hpp:347, :512`、`tests/common.py:56-58`）。M-Grouped 不支持（`generators.py:330`）：其输出的 padding 行会被整块计算，atomic add 会污染这些行；K-Grouped 每组输出是干净 `[M, N]`，且 wgrad 的分块累加正是刚需。
9. `expected_k` 在启发式里有哪三副面孔？为什么 Host 只能用估计值？
   > `compute_k = expected_k`（每块计算量/L1 常驻/反量化字节）、`k = num_groups × expected_k`（SF 总流量）、`num_batches = num_groups`（block 总数按整份网格 × G）（`heuristics.hpp:597-599, :545`）。逐组真实 K 只在设备侧 `grouped_layout` 里，Host 选配置时不可见；估计只影响配置优劣，不影响正确性。
10. K-Grouped 的 `compiled_dims` 为什么默认 `"mn"` 而不是 `"nk"`？BLOCK_M 有 M-Grouped 那样的钉死规则吗？
   > K（总 K 与逐组 K）随调用动态变化且 Kernel 以 `eff_shape_k` 为准，编译期钉死任何 K 都是错的，所以只有 M/N 进模板常量（`gemm.hpp:612, :674`）。没有钉死规则：`block_m != 256` 的过滤与 m_range 固定都只作用于 m-grouped（`heuristics.hpp:104-106, :610-616`），因为 K-Grouped 的组边界在 K 维，与 M/N block 划分正交。

## 附：3.8 三种 GEMM 类型对比表（当天产出·参考版）

| 维度 | Normal | M-Grouped | K-Grouped |
|---|---|---|---|
| 分组轴 / 典型场景 | 无 / dense 投影 | M（token 按 expert 排序）/ MoE 前向、dgrad | K（收缩维 = token 流按 expert 分组）/ MoE 反向 wgrad（EP） |
| A 形状（major） | `[M, K]`（任意） | `[M, K]`（**必须 K-major**） | `[K, M]` tn（**必须 MN-major**） |
| B 形状（major） | `[N, K]`（任意） | `[G, N, K]` nt / `[G, K, N]` nn（任意） | `[K, N]` tn（**必须 MN-major**） |
| D 形状 | `[M, N]` | `[M, N]`（与 A 同排列，含组间 padding 行） | `[G, M, N]`（逐组紧凑，无 padding） |
| `grouped_layout` 语义 | — | 行前缀和，`M = align(psum[G-1], 256)` | K 前缀和，`total_k = align(psum[G-1], 256)` |
| 调度网格 | 单个 M×N | 每组独立的 `M_g × N` 网格（块数随组变化） | 所有组**共享**同一 M×N 网格，每个非空组消耗 `num_blocks` 个任务 |
| 组间差异 | — | 行数 → `num_m_blocks` | K 长度 → `current_shape_k`（K 循环与 A/B/SF/D 基址） |
| 组偏移落点 | — | 只有 B（和 SFB）：`group(g, N, K)` | A/B：`index(·, k_idx_base)` 平移；SF：`sf_idx_base`；D：`group(g, M, N)` |
| 空组处理 | — | 占 0 行，调度自动跳过，无需清零 | 不消耗 block，但占一份 `[M, N]` 输出；非累加时 AIV `aiv_zero_empty_groups` 清零，累加时保持 `C[g]` |
| 非对齐组尾 | M/N 尾用 `actual_m/n` | 组尾 m-block **整块算**（`actual_m ≡ BLOCK_M`，padding 参与计算） | 组尾 k-block **截断**（`actual_k`，padding 不计算；fp8 的 SF 缺口 L1 内补零） |
| bias / alpha | 支持 | 不支持 | 不支持 |
| accumulate（D += …） | 支持 | 不支持 | **支持**（`c [G, M, N]`，AIV atomic add） |
| direct store | 可选（Normal only） | 否 | 否 |
| resident A/B | BF16 仅 Normal；fp8 可 | BF16 否；fp8 可 | BF16 否；fp8 可（`resident_k = expected_k`） |
| `compiled_dims` 默认 | nt/nn `"nk"`，tn/tt `"mn"` | `"nk"` | `"mn"`（K 只能运行时） |
| BLOCK_M 搜索 | 自由 | **钉死 256**（= kMkAlignment） | 自由（组边界与 M/N 正交） |
| ACLNN 对照 | 有 | 无 | 无 |
| Host 提示字段 | — | `expected_m = m / g` | `expected_k = 均值(ks_cpu)` 或 `k / g` |
| 测试专属手段 | — | `ensure_zero_padding` 可选 | NaN 预填 + fp32 累加逐位对照 + empty/single 变体 |

---

明天（第 38 天）动手运行（或手工推演）Grouped GEMM：跟踪一个 group 从 layout 到输出区间的映射，手工构造两个不等长 group 推演调度过程，并回答"给定一个 MoE 输入，token、group、expert 权重和输出如何对应"。
