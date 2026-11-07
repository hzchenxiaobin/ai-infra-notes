# 第 6 周 · 第 1 天（总第 36 天）：学习 M-Grouped GEMM

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 6 周"Grouped GEMM 与 MQA Logits"的第 1 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前五周我们把 Normal GEMM 的调度、搬运、MAD、流水线、Epilogue 和测试体系走了一遍，并在第 11 天见过 `GemmDesc` 里 `GemmType` 的四种取值。今天正式进入分组世界：**M-Grouped（M 维分组）GEMM 把"行属于哪个组"的信息压缩进一个长度为 G 的 int32 数组 `grouped_layout`，A 和 D 仍是两块平坦缓冲，B 是每组一份的 `[G, N, K]` 堆叠；调度器在 Normal 的 (m_block, n_block) 网格外面套一层"分组游标"，每领一个 block 顺带回答"它属于哪个组"，Kernel 再用这个组号给 B 加偏移**。它对应的最典型场景就是 MoE：token 已按 expert 排好序，A 的连续行段乘上对应 expert 的权重。当天产出是**M-Grouped 输入输出示意图**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 M-Grouped 的形状契约：A `[M, K]`（必须 K-major）、B `[G, N, K]`（nt）或 `[G, K, N]`（nn）、D `[M, N]`、`grouped_layout` `[G]` int32 且 contiguous；并指出"成文处"——`csrc/apis/gemm.hpp:45-58` 的形状注释与 `m_grouped_bf16_gemm_contiguous` 的 docstring（`:455-469`）；
2. 解释 **psum（前缀和）布局**：`grouped_layout[i]` 是前 i+1 组**实际行数**的累积结尾（不是组大小）；组 g 的有效行区间为 `[align(psum[g-1], 256), psum[g])`，组 0 从第 0 行开始；组间存储按 256 对齐补 padding，总行数 `M = align(psum[G-1], 256)`（`scheduler.hpp:17-23` 的注释、`tests/generators.py:212-220` 的 `generate_psum_layout`）；
3. 画出 `get_gemm_desc` 的 M-Grouped 分支（`gemm.hpp:88-91`）：`get_shape_by_spec<"gmnk">` 用四个张量标注 `"mk"/"gnk"/"mn"/"g"` 推出 g/m/n/k，`major_a == Major::K` 断言，`estimated_m = m / g` 写入 `GemmDesc.expected_m`（`:158-159`），以及 M-Grouped 不支持 bias / accumulate / alpha 的原因（这些分支只对 `GemmType::Normal` 开放，`:107`、`tests/common.py:56`）；
4. 逐行讲解 `Scheduler::get_next_block` 的 M-Grouped 分支（`scheduler.hpp:144-162`）：persistent block index `(++current_iter) * kNumCores + core_id`（`:10-12, :142`）、分组游标三件套 `current_m_block_cumsum / last_psum_m / current_psum_m`（`:67-70`）、组内 swizzle 复用（`get_swizzled_block`，`:202-248`）、`m_block_idx += last_psum_m / BLOCK_M` 的组内到全局平移，以及 `static_assert(kAlignment == BLOCK_M)`（`:146`）为什么成立；
5. 说明 Kernel 层只有 **B 享受组偏移**：`b_group_idx = scheduler.get_group_idx()`（`bf16_gemm.hpp:100`）经 `gm_ptr::group()`（`common.hpp:167-172`）加 `g × N × K × sizeof(dtype)` 字节；A 与 D 始终平坦（`bf16_gemm.hpp:102, :157, :186-188`）；`get_actual_m` 对 M-Grouped 恒返回 `BLOCK_M`（`scheduler.hpp:127-136` 的 TODO），正确性靠调用方保证 padding 行语义（`ensure_zero_padding`）；
6. 解释**配置层为什么把 BLOCK_M 钉死为 256**：`kMkAlignment = 256`（`csrc/jit/jit.hpp:59`）是布局约定的对齐粒度，`select_gemm_config` 直接拒绝 `block_m != 256` 的候选（`csrc/jit_kernels/heuristics.hpp:104-106`），m-grouped 的搜索区间被固定为 `m_range = {256, 256}`、`block_n ≤ L0C / (4 × 256) = 256`（`:610-618`），且"M-Grouped GEMM always uses BLOCK_M, rather than Expected M"（`:225-227`）；
7. 描述测试侧的构造与调用：`generate_psum_layout` 以 `uniform(0.7, 1.3) × expected_m` 随机化每组行数（`generators.py:212-220`）；`_generate_m_grouped_contiguous`（`:326-359`）零化 A 的 padding、给 SFA 的 padding 填**有限随机数**、逐组生成参考输出 `d[ms, :] = a_repr[ms, :] @ b_repr[i].T`；`gemm_dg` 的 M-Grouped 分支只允许 `nt` 布局（`tests/common.py:67-73`）且没有 ACLNN 交叉验证（`:87-90`）；
8. 回答计划的核心问题——**token 分组如何与 MoE expert 计算联系起来**：Dispatch 按 expert 排列 token 得到 A，`B[g]` 是 expert g 的权重，输出保持排列后的顺序由上层 Combine 逆排列；并独立画出 M-Grouped 输入输出示意图（当天产出）。

## 二、预备知识

- **第 3 天**：Tiling 与数据复用——今天 m-block 的划分沿 M 维按组进行，`BLOCK_M = 256` 是复用与对齐的折中。
- **第 11 天**：`GemmDesc` 与 `get_gemm_desc`——今天精读其中 `MGroupedContiguousWithPsumLayout` 分支；四种 GemmType 的 shape 检查差异在当天已埋下伏笔。
- **第 12 天**：`select_gemm_config` 与 JIT 配置——今天看它对 m-grouped 的两条特殊规则（BLOCK_M 钉死、compute_m 用 BLOCK_M）。
- **第 15、16 天**：模板参数与 L0A/L0B/L0C 容量——`BLOCK_M=256` 的容量依据是 `L0CSizeBytes = 256 KB`（`common.hpp:57`）。
- **第 17 天**：Normal 调度（`get_next_block`、`get_global_idx`、GROUP_M swizzle）——今天是同一套骨架的"分组扩展版"，请先确认自己还记得 persistent block index 的含义。
- **第 18 天**：`gm_ptr` 的 `index()/batch()` 与 stride——今天新增 `group()`（`common.hpp:167-172`）。
- **第 24、26 天**：`actual_m` 边界处理与特殊路径——今天 M-Grouped 的 `get_actual_m` 恒返回 `BLOCK_M`（带 TODO），是"特殊路径"的又一种形态。
- **第 29、30 天**：`generators.py` 的测试数据生成与 `common.py` 的验证流程——今天消费它们的 m-grouped 分支。
- **第 33 天**：Scale 布局转换——m-grouped 的 sfa 走 `MGroupedContiguousWithPsumLayout` 分支（组尾清零）、sfb 走 Batched 分支（`gemm.hpp:128-154`），今天只需知道入口，细节已在第 33 天讲过。

另注：第五节 A 组任务需要 Ascend 环境；无环境时完成 B 组静态任务（psum 手算、调度推演、地址计算、配置验证、画图）即可达到当天目标。

## 三、详细讲解

### 3.1 从 MoE 到 M-Grouped GEMM：为什么需要它

MoE（Mixture of Experts）层的前馈计算可以概括为：每个 token 被路由到若干 expert，每个 expert 是一个小型两层 FFN。设 expert 数为 G、某层权重为 `W_g ∈ [N, K]`，则该层需要计算

```text
y_i = W_{g(i)} · x_i        （x_i 是路由到 expert g(i) 的 token）
```

把所有 token 的 x 堆成矩阵，最自然的做法是**先把 token 按 expert 重排**（Dispatch 阶段的输出正是这种排列），于是：

- A `[M, K]`：M 是全部 token 数（更准确地说是 token·副本数，top-k > 1 时同一 token 会出现在多个组里），**组 g 的 token 恰好占据一段连续行**；
- B `[G, N, K]`：G 个 expert 的权重堆叠；
- D `[M, N]`：输出保持与 A 相同的行排列，由上层 Combine 按 token 原始位置gather 回去。

为什么不直接调 G 次 Normal GEMM？两个原因：

1. **每组 M 不相等且不可预测**：token 路由是动态的，有的 expert 分到几千行、有的只有几十行。逐 expert 启动 kernel，小 M 的那次调用利用率极低，且 G 次 launch 的固定开销不可忽视（DeepSeek 类模型 G 通常上百）。
2. **每个 expert 独立的 `[M_g, K]` 输入需要先拼装**：而"按 expert 排序后拼接"的 A 本来就是 Dispatch 的自然产物，直接交给一个 kernel 反而省去拆装。

也不适合用 Batched（`D[b] = A[b] @ B[b]^T`，`scheduler.hpp:37-41`）：Batched 要求每个 batch 的 M/N/K 完全相同，而每组 token 数不同；且 Batched 的 A 是 `[B, M, K]` 三维，M-Grouped 的 A 是拼接后的一块二维平坦缓冲，组间除对齐 padding 外没有空洞要求，内存利用率更高。

于是 M-Grouped 的核心设计问题变成：**如何用最少的元数据告诉 kernel"A 的哪些行属于哪个组"**？答案是前缀和数组 `grouped_layout`：长度 G、int32，第 i 个元素是前 i+1 组行数的累积结尾。G 个数就能完整描述任意不均匀的切分。项目 README 对该算子的定位是"Grouped GEMM for DeepSeek MoE experts"，`#Groups` 是 expert 数、"M per group" 是每个 expert 平均接收的 token 数（`README.md:134-136`）。

### 3.2 输入输出契约：A、B、D 与 grouped_layout

API 的"成文处"有两处。`get_gemm_desc` 上方的形状注释（`csrc/apis/gemm.hpp:45-58`）给出四种 GemmType 的对照：

```text
 *                A                 B                CD
 *
 *  Normal:       [M, K]            [N, K]            [M, N]
 *  MGrouped:     [M, K]         [G, N, K]            [M, N]
 *  KGrouped(*):  [M, K]            [N, K]         [G, M, N]
 *  Batched:   [B, M, K]         [B, N, K]         [B, M, N]
```

`m_grouped_bf16_gemm_contiguous` 的 docstring（`gemm.hpp:455-469`）补充语义细节：A 的行被划分为 G 组**连续存储**，B 是 `[G, N, K]` 的逐组堆叠，组边界由 `grouped_layout` 描述——"Per-group row counts describing the contiguous partition of A"。

**psum 布局的精确语义**（`scheduler.hpp:17-23` 注释是权威表述）：

- `grouped_layout` 是"prefix-sum-of-actual-rows"数组，长度 `num_groups`；
- 组 g 的**有效行**占据全局 M 空间的 `[align(psum[g-1], BLOCK_M), psum[g])`，组 0 从 0 开始；
- 其中 `BLOCK_M` 在本仓库恒为 256（3.6 节），即组间存储按 256 对齐；`[psum[g], align(psum[g], 256))` 是组 g 的**padding 行**；
- A 的总行数 `M = align(psum[G-1], 256)`。

注意两个容易踩的坑：

1. `grouped_layout[i]` 是**累积结尾**而不是组大小。组 g 的有效行数 = `psum[g] - align(psum[g-1], 256)`；组 0 的有效行数 = `psum[0]`。
2. "M"出现在两个层面：`GemmDesc.m`（A 的物理总行数，含 padding）和 `expected_m`（Host 估算的每组平均行数 `m / g`，`gemm.hpp:158-159`，用于启发式的成本模型）。名字都叫 m，含义完全不同。

**手算一个贯穿全天的例子**：设 `G = 3`，三组的实际行数分别为 `[300, 128, 512]`，对齐粒度 256。按 `generate_psum_layout`（`tests/generators.py:212-220`）的构造规则：

```text
组 0：psum=0 → layout[0] = 0+300 = 300；下一组起点 = align(300,256) = 512
组 1：psum=512 → layout[1] = 512+128 = 640；下一组起点 = align(640,256) = 768
组 2：psum=768 → layout[2] = 768+512 = 1280；总行数 M = align(1280,256) = 1280

grouped_layout = [300, 640, 1280]     # int32, 长度 3
A: [1280, K]   D: [1280, N]   B: [3, N, K]
```

全局行空间的划分（`▓` 有效，`·` padding）：

```text
行号      0────────300   512────────640   768────────────────1280
组 0      ▓▓▓▓▓▓▓▓▓▓···· ▓▓▓▓▓▓▓▓▓▓▓▓▓···· ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
          <---- 组 0 ----> <---- 组 1 ----> <-------- 组 2 -------->
有效区间   [0, 300)         [512, 640)       [768, 1280)
padding   [300, 512)       [640, 768)       （无，恰好对齐）
m-block   行 [0,256) [256,512) 行 [512,768)  行 [768,1024) [1024,1280)
          全局 m_block_idx: 0, 1           2                 3, 4
```

每个组内的 m-block 数 = `ceil(有效行数 / 256)`：组 0 有 2 块、组 1 有 1 块、组 2 有 2 块。注意组 0 的第二个 m-block `[256, 512)` 中只有 `[256, 300)` 是有效行，其余 212 行是 padding——但这个 block 仍然整块参与计算（3.5 节）。

### 3.3 API 层：`get_gemm_desc` 的 M-Grouped 分支

Python 侧入口是 `deep_gemm.m_grouped_bf16_gemm_nt_contiguous` / `_nn_`（`deep_gemm/__init__.py:66-67`）以及 fp8/fp4 版本 `m_grouped_fp8_fp4_gemm_nt_contiguous` 等（`:49-55`）。它们来自 `register_apis` 的 pybind 绑定（`csrc/apis/gemm.hpp:586-605, :656-669`；`m_grouped_fp8_gemm_*` 是 `m_grouped_fp8_fp4_gemm_*` 的别名，`:628-629`）。

BF16 版本的 Host 实现（`gemm.hpp:470-499`）：

```cpp
template <bool kTransB>
static void m_grouped_bf16_gemm_contiguous(
        const torch::Tensor& a, const torch::Tensor& b_,
        const torch::Tensor& d, const torch::Tensor& grouped_layout, ...) {
    auto b = get_transposed<kTransB>(b_, 1, 2);          // nn: [G, K, N] → 逻辑 [G, N, K]
    DJ_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(use_psum_layout == true);             // 目前唯一支持的布局
    auto maybe_desc = get_gemm_desc(
        GemmType::MGroupedContiguousWithPsumLayout,
        a, b, d, std::nullopt, ...,                       // 注意：c 恒为 nullopt
        grouped_layout, compiled_dims, true, expected_m_for_psum_layout);
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_bf16_gemm(maybe_desc.value());
}
```

四个要点：

1. **转置处理**：A 只有 `[M, K]` 形态（模板只对 `kTransB` 参数化，`:470`）；nn 变体把 B 的后两维逻辑转置（`get_transposed<kTransB>(b_, 1, 2)`，`:480`），fp8 版本则连 A 一起参数化但注册的也只有 nt/nn 两种（`:323-324`）。
2. **形状推导**：M-Grouped 分支（`:88-91`）用 `get_shape_by_spec<"gmnk">` 把四个张量的事实拼起来——A 标注 `"mk"`、B 标注 `"gnk"`、D 标注 `"mn"`、`grouped_layout` 标注 `"g"`（`csrc/utils/layout.hpp:140` 起的 `_sp` 机制，按 stride=1 找主轴并交叉校验同名维度一致）。因此 g 来自 B 的第 0 维和 layout 长度，m 来自 A/D，n 来自 B/D，k 来自 A/B，任何一处不一致都会在 Host 断言失败。
3. **major 约束**：`DJ_HOST_ASSERT(major_a == Major::K)`（`:90`）——A 必须 K-major。原因与第 33 天的 SF 契约同源：分组边界在行（MN）维度，K-major 的 A 每行连续，组边界恰好切在行的整数倍处；若 MN-major，"组"的概念在存储上不再连续。
4. **layout 本身的检查**（`:100-103`）：`grouped_layout->is_contiguous()` 且 `scalar_type() == torch::kInt`（int32）。

`GemmDesc` 的填充（`:164-196`）中与分组直接相关的字段：`gemm_type`、`grouped_layout`（数据指针）、`num_groups = g`、`expected_m = expected_m_for_psum_layout.value_or(m / g)`。`stride_batch_*` 恒为 0（那是 Batched 专用），`outer_stride_a/b/d` 仍按第 18 天的语义取。

**为什么没有 bias / accumulate / alpha**：`with_bias` 只在 `gemm_type == GemmType::Normal` 时成立（`:107-108`）；测试侧同样断言"accumulate 只允许 Normal 或 K-Grouped"（`tests/common.py:56`）并注明"grouped matmul doesn't support accumulation yet"（`generators.py:330`）。MoE 场景也不需要：每个输出行只属于一个组，不存在跨组累加的语义；bias 属于 Normal GEMM 的 epilogue 概念，m-grouped 的 epilogue 走通用 `epilogue_class`（如 `BF16StochasticRounding`，`common.py:169-171` 对 m-grouped bf16 同样会测）。

`use_psum_layout` 目前必须为 true（`:484`、绑定默认值 `py::arg("use_psum_layout") = true`，`:659`）——它选择"partial-sum 前缀和布局"这条唯一实现路径，参数保留是为了与 CUDA 版 DeepGEMM 的 API 兼容。`expected_m_for_psum_layout` 是可选的提示值，绕过 `m/g` 的粗略估算。

SF（低精度）路径补充一句（细节在第 33 天）：sfa 走 `transform_sf_into_required_layout` 时传入 `grouped_layout`（按组清零 padding 的 SF 行，`gemm.hpp:132-138`）；sfb 因为 B 天然带组维，走 Batched 形态的转换（`:145-152`）。

### 3.4 调度层：`Scheduler` 的分组游标

`Scheduler`（`deep_gemm/include/deep_gemm/scheduler.hpp:42-249`）是四种 GemmType 共用的任务分配器。Normal 模式下每个核独立地领 block：`block_idx = (++current_iter) * kNumCores + core_id`（`:142`），把 block_idx 映射到 (m_block, n_block) 并用 GROUP_M swizzle + N 向蛇形提升 L2 局部性。`kNumCores` 来自 `config.launch_options.num_blocks = d.num_cores`（`csrc/jit_kernels/heuristics.hpp:98`），即 AI Core 数（`common.hpp:51`：`kNumAICores = 32`）。

M-Grouped 在这套骨架外套了**三个分组游标**（`:67-70`）：

| 成员 | 含义 | 初始值（构造函数 `:99-102`） |
|---|---|---|
| `current_group_idx` | 最近一次 `get_next_block` 返回的 block 所属的组 | 0 |
| `last_psum_m` | 当前组的起始行 = `align(psum[g-1], 256)` | 0 |
| `current_psum_m` | 当前组的有效结尾行 = `psum[g]` | `grouped_layout[0]` |
| `current_m_block_cumsum` | 之前的组**已经消耗掉的 m-block 总数** | 0 |

构造时 `num_m_blocks` 被改写为组 0 的块数：`ceil_div(grouped_layout[0], BLOCK_M)`（`:102`）——而不是整个 M 的块数。

`get_next_block` 的 M-Grouped 分支（`:144-162`）逐行解读：

```cpp
static_assert(kAlignment == BLOCK_M);                    // ①
while (true) {
    if (block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
        break;                                           // ② block 落在当前组内
    if (++current_group_idx == num_groups)
        return false;                                    // ③ 越过最后一组：任务耗尽
    last_psum_m = aligned(current_psum_m, kAlignment);   // ④ 新组起始行（256 对齐）
    current_psum_m = grouped_layout[current_group_idx];  //    新组有效结尾
    current_m_block_cumsum += num_m_blocks;              // ⑤ 累加旧组的块数
    num_m_blocks = ceil_div(current_psum_m - last_psum_m, BLOCK_M);  // 新组块数
}
uint32_t in_group_idx = block_idx - current_m_block_cumsum * num_n_blocks;  // ⑥ 组内线性下标
get_swizzled_block(in_group_idx, num_m_blocks, m_block_idx, n_block_idx);   // ⑦ 组内 swizzle
m_block_idx += last_psum_m / BLOCK_M;                    // ⑧ 平移到全局 m-block 下标
return true;
```

- **①** `kAlignment`（256，由 launch 传入，`csrc/jit_kernels/bf16_gemm.hpp:65`）必须等于 `BLOCK_M`。这是整个布局的基石：组起始行是 256 的倍数、block 高度也是 256，两者相等保证**任何 m-block 都不会跨越两个组**，且 `last_psum_m / BLOCK_M`（⑧）是精确整除。
- **②** 当前组覆盖的全局 block 区间是 `[current_m_block_cumsum, current_m_block_cumsum + num_m_blocks) × num_n_blocks`。块数为 0 的空组会让条件立即失败从而在 ③④⑤ 中被跳过——空组在 M 维不占据任何行，天然无需处理（对比 K-Grouped：空组仍占 `[M, N]` 输出，需要 `aiv_zero_empty_groups` 清零，`bf16_gemm.hpp:175-176` 只对 k-grouped 启用）。
- **⑥⑦⑧** 组内先把 block_idx 归零到组内坐标系，做与 Normal 完全相同的 swizzle（`get_swizzled_block`，`:202-248`，含 `local_num_m_blocks < kGroupM(4)` 时的 row-major 快路径），最后把 m-block 加上组起始行对应的块偏移。n 维没有组的概念，直接沿用。

**为什么游标可以"只进不退"**：`block_idx` 在单个核内随 `current_iter` 单调递增，因此从缓存状态继续推进 cumsum 一定是正确的（`scheduler.hpp:33-35` 对 K-Grouped 的注释说明了同一原理）。这也是为什么游标可以做成普通成员变量而不需要二分查找——顺序到达的 block_idx 天然有序。

**用手算例子推演调度**（接 3.2 的例子，补上 `N = 7168, BLOCK_N = 256` → `num_n_blocks = 28`；组块数 2/1/2 → 总 block 数 `(2+1+2) × 28 = 140`；`kNumCores = 32`）。追踪 **core 0** 的 block_idx 序列 `0, 32, 64, 96, 128`：

| block_idx | 组判定过程 | 落点 | `get_group_idx()` |
|---|---|---|---|
| 0 | `0 < (0+2)×28=56` ✓ 组 0 | in_group=0 → (m=0,n=0)；全局 m_block=0+0=0 | 0 |
| 32 | `32 < 56` ✓ 组 0 | in_group=32 → (m=1,n=4)；全局 m_block=1 | 0 |
| 64 | `64 < 56` ✗ → 推进：`last_psum_m=512, current_psum_m=640, cumsum=2, num_m_blocks=ceil(128/256)=1`；`64 < (2+1)×28=84` ✓ 组 1 | in_group=64−56=8 → (m=0,n=8)；全局 m_block=0+512/256=**2** | 1 |
| 96 | `96 < 84` ✗ → 推进：`last_psum_m=768, current_psum_m=1280, cumsum=3, num_m_blocks=2`；`96 < (3+2)×28=140` ✓ 组 2 | in_group=96−84=12 → (m=0,n=12)；全局 m_block=0+768/256=**3** | 2 |
| 128 | `128 < 140` ✓ 组 2 | in_group=128−84=44 → (m=1,n=16)；全局 m_block=1+3=4 | 2 |
| 160 | 推进时 `++current_group_idx == 3 == num_groups` → **return false**，core 0 结束（共 5 块） | — | — |

140 个 block 分给 32 个核：core 0～11 各领 5 块、core 12～31 各领 4 块（`12×5 + 20×4 = 140`）。**同一个核领到的 block 会跨组**（core 0 依次经过组 0、1、2），这正是"分组"只是元数据层面的事——计算、搬运、Epilogue 的所有机制与 Normal 完全一致。

### 3.5 Kernel 层：只有 B 拿到组偏移

进入 `bf16_gemm_impl`（`deep_gemm/include/deep_gemm/bf16_gemm.hpp:27-193`），M-Grouped 专属的逻辑只有寥寥数行：

```cpp
Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumCores, kAlignment> scheduler(
    block_idx, shape_m, shape_n, grouped_layout, num_groups);        // :73-74
constexpr bool is_m_grouped = kGemmType == GemmType::MGroupedContiguousWithPsumLayout;  // :76
...
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {         // :89
    const uint32_t m_idx = scheduler.get_global_idx(BLOCK_M, m_block_idx);   // 全局行起点
    const uint32_t n_idx = scheduler.get_global_idx(BLOCK_N, n_block_idx);
    const uint32_t b_group_idx = is_m_grouped ? scheduler.get_group_idx() : 0;  // :100 ★
    const auto a = gm_a.batch(0);                                    // :102 A：平坦，无组偏移
    const auto b = gm_b.batch(0).group(b_group_idx, shape_n, shape_k);          // :103 ★ B：加组偏移
    const uint32_t actual_m = scheduler.get_actual_m(m_block_idx);   // :105 恒为 BLOCK_M（m-grouped）
    ...
}
```

`gm_ptr::group()`（`common.hpp:167-172`）的地址公式：

```cpp
constexpr __aicore__ gm_ptr group(uint64_t group_idx, uint64_t shape_mn, uint64_t shape_k) const {
    constexpr uint64_t pack_size = is_fp4<dtype_t>() ? 2 : 1;
    gm_ptr r = *this;
    r.addr += group_idx * shape_mn * shape_k / pack_size * sizeof(dtype_t);
    return r;
}
```

即组 g 的 B 起始地址 = 基址 + `g × N × K × sizeof(dtype)`（fp4 按 2 元素/字节折半）。这就是"B 是 `[G, N, K]` 紧凑堆叠"在地址层面的全部体现——**没有三维 stride，只有一段乘法**。随后 `b.index(n_idx, k_idx)` 照常按第 18 天的公式定位 block 的 `(n, k)` 起点。fp8 路径同理（`fp8_gemm.hpp:97-100, :136-137`），且 `gm_sfb` 也套同一个 `group()`。

**A 与 D 为什么不需要组偏移**：A 的组本来就拼接在同一段 `[M, K]` 里，`m_idx`（全局行号）已经落在正确的组段内；D 同理是平坦的 `[M, N]`，`gm_d.batch(0).group(0, ...)`（direct store 路径 `:157`、AIV drain 路径 `:186-188`）组号恒为 0。**分组信息只影响"B 从哪份权重取数"**，输出位置完全由全局 `m_idx/n_idx` 决定——这保证了写回与 Normal 完全同构。

**`actual_m` 的 TODO 语义**（`scheduler.hpp:127-136`）：

```cpp
__aicore__ uint32_t get_actual_m(uint32_t m_block_idx) const {
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
        // TODO: remove this and add tail handling for ensure_zero_padding=True
        return BLOCK_M;
    } else {
        return min(shape_m - m_block_idx * BLOCK_M, BLOCK_M);
    }
}
```

Normal 模式下最后一个 m-block 会被截断到剩余行数；M-Grouped 模式**恒返回 BLOCK_M**，即组尾的最后一个 m-block 连同 padding 行一起整块计算。正确性由两条路径保证（`gemm.hpp:286-288` 的参数注释："the kernel itself also accepts non-zero padding"）：

- `ensure_zero_padding = true`（默认）：调用方保证 A 的 padding 行为 0（测试在 `generators.py:338-340` 用 `zero_operand_slices_` + `iter_psum_sep` 零化），于是 D 的 padding 行 = `0 @ B[g]^T = 0`，全量比较安全（`cmp_spans = None`，`:357`）。此时 SFA 的 padding 可以填**有限**随机数（`:343-346`）——0 乘任何有限 scale 仍为 0，垃圾不会传播进有效结果；
- `ensure_zero_padding = false`：A 的 padding 行任意，D 的 padding 行随之是垃圾，测试只比较有效区间（`cmp_spans = iter_psum_slice(...)`，`:357`）。

有趣的对照：K-Grouped 的空组需要 `aiv_zero_empty_groups` 主动清零输出（`epilogue.hpp:108-135`），M-Grouped 的空组占 0 行、连清零都省了——这是"组在 M 维 vs K 维"带来的结构性差异，第 37 天会展开。

### 3.6 配置层：BLOCK_M 为什么被钉死为 256

M-Grouped 的 `GemmConfig` 有一条其他模式没有的硬规则。`select_gemm_candidate`（`csrc/jit_kernels/heuristics.hpp:100-106`）：

```cpp
// Check BLOCK_M, BLOCK_N: must fit L0C
const uint64_t block_bytes = static_cast<uint64_t>(block_m) * block_n * sizeof(float);
if (block_bytes > L0CSizeBytes) return std::nullopt;
// Check MGroupedLayout: must use the fixed BLOCK_M for contiguous layout
// TODO: pad zero manually in output output buffer for MGroupedLayout
if (is_m_grouped(d.gemm_type) && block_m != Runtime::get_mk_alignment_for_contiguous_layout()) return std::nullopt;
```

链条是这样的：

1. **布局约定端**：`kMkAlignment = 256`（`csrc/jit/jit.hpp:59`，`set_mk_alignment_for_contiguous_layout` 甚至断言不允许改成别的值，`:65-68`）。psum 布局的组间对齐粒度就是 256，`get_gemm_desc` 之后的 SF 变换也按 256 对齐（第 33 天）。
2. **调度端**：`static_assert(kAlignment == BLOCK_M)`（`scheduler.hpp:146`）要求 block 高度等于对齐粒度，否则组边界会切进 block 内部。
3. **配置端**：heuristics 直接过滤掉 `block_m != 256` 的候选（上面的 `return std::nullopt`），并在搜索区间里写死 `m_range = {256, 256}`（`heuristics.hpp:610-618`）；`BLOCK_N` 由 L0C 容量给出上界 `L0CSizeBytes / (sizeof(float) × 256) = 256 KB / 1 KB = 256`。

容量算账（第 16 天的知识）：`L0CSizeBytes = 256 × 1024`（`common.hpp:57`），fp32 累加器下 `BLOCK_M × BLOCK_N × 4B ≤ 256KB`。`256 × 256 × 4B = 256KB` 恰好占满 L0C——这是 m-grouped 在"组边界必须 256 对齐"约束下能用的最大方阵。

成本模型里还有一条注释（`heuristics.hpp:225-227`）：

```cpp
// Derive Actual MatMul Shape
// M-Grouped GEMM always uses BLOCK_M, rather than Expected M
// TODO: pad zero manually in output buffer for MGroupedLayout
const int compute_m = is_m_grouped(d.gemm_type) ? block_m : align(std::min(m, block_m), align_m);
```

估算 FLOPS 时，Normal 用"min(M, BLOCK_M) 对齐到 MAD 粒度"，而 M-Grouped **永远按整块 BLOCK_M 计**——因为组尾 block 连 padding 一起算，实际算的行数就是 BLOCK_M。`expected_m`（`m/g` 或用户提示值）只影响其他成本项的估算（如 `load_m = num_m_blocks * block_m`，`:300`），不改变"每块算满 256 行"的事实。

### 3.7 测试视角：generators 与 common.py 的 M-Grouped 分支

**用例空间**（`tests/generators.py:579-603`）：

```python
yield create_test(128, 1, 4, 4, Major.K)      # 最小用例：128 组、每组期望 1 行

m_group_list = [(4, 1), (4, 8192), (8, 4096)]          # (num_groups, expected_m)
n_k_list = [(6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048)]
```

这组 (G, expected_m, N, K) 与 README 性能表的 M-Grouped 条目（`README.md:138-145`）一一对应——测试用例就是 MoE 基准形状的正确性版本。`a_major` 恒为 `Major.K`（与 API 断言一致），`major_b` 遍历 K/MN 两种。`GemmTestDesc.alignment` 默认 256（`:260`），正是 `kMkAlignment`。

**数据生成**（`_generate_m_grouped_contiguous`，`:326-359`）按顺序做六件事：

1. `generate_psum_layout(expected_m, G, 256)`（`:212-220`）：每组行数 = `max(int(uniform(0.7, 1.3) × expected_m), 0)`——**刻意不均匀**（最小用例 128 组 × 期望 1 行会产出大量 0～1 行的组和空组），返回 `(total_m, psum_layout, int32 张量)`；
2. 断言不支持 accumulate（`:330`）、A 必须 K-major（`:332`）；
3. 生成 A `(total_m, k)`（K-major）与 B `(G, n, k)`；
4. `ensure_zero_padding=True` 时用 `iter_psum_sep`（`:230-235`）把 A 的 padding 行区间清零（`:338-340`）——`iter_psum_sep` 产出的正是 3.2 节手算的 `[ceil(psum[g]/1), align(psum[g],256))` 区间（gran=1 时即 padding 行本身）；
5. SFA 的 padding 填 `randn_like` 垃圾（`:343-346`，3.5 节解释了为什么无害）；
6. 参考输出：`d` 零初始化，逐组 `d[ms, :] = a_repr[ms, :] @ b_repr[i].T`，其中 `ms = iter_psum_slice(...)`（`:223-227`）产出有效区间（`:351-353`）；`cmp_spans` 按是否零 padding 决定全量或有效区间比较（`:357`）。

**调用侧**（`tests/common.py:49-84` 的 `gemm_dg`）：

```python
elif test.gemm_type == GemmType.MGroupedContiguousWithPsumLayout:
    if test.has_sf:
        extra_args['recipe_a'] = test.a_recipe
        extra_args['recipe_b'] = test.b_recipe
    assert trans_ab[0] == 'n'                       # A 只能 [M, K]
    b = b if trans_ab[1] == 't' else get_transposed(b, 1, 2)
    getattr(deep_gemm, f'm_grouped_{gemm_name}_gemm_{trans_ab}_contiguous')(
        a, b, d, g_layout, **extra_args)
```

`get_available_trans_ab` 对 m-grouped 只返回 `['nt']`（`generators.py:293-294`）——nt/nn 的差异在 `gemm_dg` 里体现为是否把 B 后两维逻辑转置。外层 `run_gemm_test` 中，m-grouped 不参与 bias/alpha 循环（`:154`、`:160-164` 与 `:186` 的条件都限定 Normal），但 bf16 同 dtype 时会测 `BF16StochasticRounding` epilogue 与 fp32 直出（`:169-174`）。

**为什么没有 ACLNN 对照**（`common.py:87-90`）：注释写明"ACLNN only provides a reference for the dense (Normal) GEMM; m-/k-grouped ops have no aclnn counterpart"——分组算子没有官方参考实现，正确性完全依赖 `d_ref`（PyTorch 逐组矩阵乘）。这也是第 30 天"参考结果 vs ACLNN 交叉验证"两条腿在分组场景只剩一条的原因。

### 3.8 当天产出·参考图：M-Grouped 输入输出示意图

```text
                              M-Grouped GEMM（psum 布局）
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  输入                                                                        │
 │                                                                             │
 │  grouped_layout [G] int32（前缀和，非组大小）                                  │
 │    g=0        g=1        g=2            g=G-1                               │
 │    [300,      640,       1280,  ......  psum[G-1]]                           │
 │                                                                             │
 │  A [M, K]（K-major，M = align(psum[G-1], 256)）                               │
 │    ┌─────────┬────────┬─────────┬─────┰┄┄┄┄┄┄┬─────────┐                    │
 │    │ 组0有效  │ 组0pad │ 组1有效  │ ... │ 组g pad │ 组G-1有效│                │
 │    │ [0,300) │[300,512)│[512,640)│     │         │         │                │
 │    └─────────┴────────┴─────────┴─────┰┄┄┄┄┄┄┴─────────┘                    │
 │         行号 = 全局 m_idx，组 g 有效区间 [align(psum[g-1],256), psum[g])       │
 │                                                                             │
 │  B [G, N, K]（紧凑堆叠；每 expert 一份权重）                                    │
 │    ┌────────┐ ┌────────┐      ┌────────┐                                    │
 │    │ B[0]   │ │ B[1]   │ ...  │ B[G-1] │   group() 偏移：                    │
 │    │ [N,K]  │ │ [N,K]  │      │ [N,K]  │   addr += g·N·K·sizeof(dtype)      │
 │    └────────┘ └────────┘      └────────┘                                    │
 └─────────────────────────────────────────────────────────────────────────────┘
                    │ get_gemm_desc: g/m/n/k + major_a==K + int32 layout
                    │ heuristics: BLOCK_M=256(=kMkAlignment), BLOCK_N≤256(L0C)
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  调度（Scheduler, 每核 persistent）                                            │
 │    block_idx = iter·kNumCores + core_id                                     │
 │    ┌─ 组游标推进：cumsum / last_psum_m / current_psum_m ─────────────────┐    │
 │    │  while block_idx ≥ (cumsum+num_m_blocks)·num_n_blocks: 推进组       │    │
 │    │  in_group_idx = block_idx − cumsum·num_n_blocks                     │    │
 │    │  get_swizzled_block(in_group_idx, 组内num_m_blocks, m, n)            │    │
 │    │  m_block_idx += last_psum_m / BLOCK_M   （组内 → 全局，精确整除）      │    │
 │    └────────────────────────────────────────────────────────────────────┘    │
 │    get_group_idx() → b_group_idx（只给 B 用）                                  │
 └─────────────────────────────────────────────────────────────────────────────┘
                    │ Kernel: b = gm_b.group(b_group_idx, N, K)
                    │         a/d 平坦无组偏移；actual_m 恒为 BLOCK_M
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  输出                                                                        │
 │                                                                             │
 │  D [M, N]（与 A 同排列）                                                      │
 │    ┌─────────┬────────┬─────────┬─────┬─────────┐                          │
 │    │ 组0有效  │ 组0pad │ 组1有效  │ ... │ 组G-1   │  pad 行 = 0@Bᵀ = 0       │
 │    │ A[0:300]│ (为零) │A[512:640]│     │         │  （ensure_zero_padding）  │
 │    │ ×B[0]ᵀ  │        │ ×B[1]ᵀ   │     │ ×B[G-1]ᵀ│                          │
 │    └─────────┴────────┴─────────┴─────┴─────────┘                          │
 │    MoE 语义：token 行 × expert g 权重 = 该 token 在 expert g 的 FFN 输出       │
 └─────────────────────────────────────────────────────────────────────────────┘
```

MoE 全链路的对应关系（回答计划的最后一问）：**Dispatch**（按 expert 排列 token，top-k 复制）→ 得到 A 与 `grouped_layout` → **M-Grouped GEMM**（本算子；FFN 第一层 `gate/up projection`，N = moe_intermediate×因子）→ 中间激活 → 第二层 M-Grouped GEMM（`down projection`，N = hidden）→ **Combine**（按 token 原位置 gather、加权求和）。第 43 天起的 MegaMoE 会把这条链连同通信一起融合。

## 四、源码导读

今天的主线文件与阅读站点（建议按表中顺序读）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `csrc/apis/gemm.hpp` | 形状注释 `:45-58`、`get_gemm_desc` 的 M-Grouped case `:88-91`、layout 检查 `:100-103`、`estimated_m` `:158-159`、GemmDesc 填充 `:164-196`、`m_grouped_bf16_gemm_contiguous` `:470-499`、`m_grouped_fp8_fp4_gemm_contiguous` `:289-321`、绑定 `:586-605, :656-669` | 今天的主线 Host 逻辑；docstring `:455-469` 是接口文档 |
| 2 | `deep_gemm/include/deep_gemm/scheduler.hpp` | 类注释 `:10-41`、游标成员 `:62-77`、构造函数 `:87-108`、`get_group_idx` `:118`、`get_actual_m` `:127-136`、`get_next_block` `:141-196`、`get_swizzled_block` `:202-248` | 调度核心，见下方逐段导读 |
| 3 | `deep_gemm/include/deep_gemm/bf16_gemm.hpp` | Scheduler 构造 `:73-74`、`b_group_idx` `:100`、a/b 取址 `:102-103`、direct store 的 D `:157`、AIV drain `:180-191` | 分组信息如何（只）落到 B |
| 4 | `deep_gemm/include/deep_gemm/common.hpp` | `GemmType` `:87-92`、`is_m_grouped` `:105-107`、`kNumAICores` `:51`、`L0CSizeBytes` `:57`、`gm_ptr::group` `:167-172` | 常量、枚举与组偏移地址公式 |
| 5 | `csrc/jit_kernels/heuristics.hpp` | `make_gemm_candidate` 的 m-grouped 过滤 `:104-106`、`compute_m` `:225-227`、`load_m` `:300`、m_range/n_range `:610-618` | BLOCK_M=256 的三条依据 |
| 6 | `csrc/jit/jit.hpp` | `kMkAlignment` `:59-68` | 对齐粒度常量与"不许改"断言 |
| 7 | `csrc/jit_kernels/bf16_gemm.hpp` | kAlignment 实参 `:65`、launch 传参 `:73-77` | `kAlignment` 与 `grouped_layout/num_groups` 如何传进 Kernel |
| 8 | `tests/generators.py` | `GemmType` `:20-24`、`generate_psum_layout` `:212-220`、`iter_psum_slice` `:223-227`、`iter_psum_sep` `:230-235`、`GemmTestDesc.alignment` `:260`、`get_available_trans_ab` `:293-294`、`_generate_m_grouped_contiguous` `:326-359`、`_enumerate_m_grouped_contiguous` `:579-603` | 用例空间与 psum 构造 |
| 9 | `tests/common.py` | `gemm_dg` m-grouped 分支 `:67-73`、accumulate 断言 `:56`、ACLNN 跳过 `:87-90`、bias/alpha 循环条件 `:154, :160-164, :186` | 调用与验证差异 |
| 10 | `tests/test_bf16_gemm.py` | CLI `:25, :39-40` | `mgrouped` 选择子 |
| 11 | `deep_gemm/include/deep_gemm/fp8_gemm.hpp` | `:97-100, :136-137` | 低精度路径的同一模式（含 sfb 的 group） |
| 12 | `deep_gemm/include/deep_gemm/epilogue.hpp` | `aiv_zero_empty_groups` `:108-135` | 反向对照：k-grouped 空组清零，m-grouped 不需要 |

**`Scheduler` M-Grouped 相关逐段导读**（行号见 `deep_gemm/include/deep_gemm/scheduler.hpp`）：

- **`:10-12`**：调度模型总述——每个核通过 `get_next_block` 独立走全局网格，persistent block index 按 `kNumCores` 跨步。
- **`:17-23`**：M-Grouped 的权威语义注释：A/D 平坦拼接、组 g 有效行 `[align(psum[g-1], BLOCK_M), psum[g])`、组 0 从 0 起、B/SFB 逐组、组号经 `get_group_idx()` 暴露、组尾 block 由 actual_m 截断（带 TODO，当前实现恒 BLOCK_M）。
- **`:42-50`**：模板参数——`kGemmType`、`BLOCK_M/BLOCK_N`、`kNumCores`、`kAlignment=256`、swizzle 参数 `kGroupM=4`。
- **`:62-77`**：状态成员。`grouped_layout` 是 `__gm__ int32_t*`（直读 GM，不搬 UB）；m-grouped 专用三游标 + `current_group_idx`。
- **`:87-108`**：构造函数。m-grouped 分支把 `num_m_blocks` 初始化为**组 0** 的块数（`:102`），其余分组信息惰性推进。
- **`:113-115`**：`get_global_idx` 只是 `block_idx * block_size`——组偏移刻意不在这里，由 Kernel 对 B 单独施加（`:110-112` 注释）。
- **`:118`**：`get_group_idx()` 返回最近一次 `get_next_block` 所属的组。
- **`:127-136`**：`get_actual_m` 的 m-grouped TODO（3.5 节）。
- **`:144-162`**：`get_next_block` 的 m-grouped 分支（3.4 节逐行讲解）。
- **`:202-248`**：`get_swizzled_block`——`local_num_m_blocks < kGroupM` 走 row-major 快路径（小 group 常态）；否则 GROUP_M swizzle + 蛇形 + 尾部 row-major。**逐组复用同一份 Normal swizzle 代码**，local 参数是组内块数。

**`_generate_m_grouped_contiguous` 逐段导读**（`tests/generators.py:326-359`，3.7 节已列六步，补两个细节）：

- `iter_psum_slice`（`:223-227`）与 `iter_psum_sep`（`:230-235`）是同一份 psum 的两种切法：前者产出**有效**区间 `[last, ptr)` 并把游标推进到 `aligned(ptr, alignment)`，后者产出**padding**区间 `[ceil(ptr/gran), aligned(ptr, alignment)/gran)`（gran=1 时即行区间，sf 场景除以 gran_k）。参考输出与清零分别消费两者。
- 返回值 `(total_m, n, k), a, b, None, d, (g_layout, cmp_spans)`——shape 的 m 是**总行数**（含 padding），与 `GemmDesc.m` 一致。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通 M-Grouped BF16 正确性测试**：

   ```bash
   python tests/test_bf16_gemm.py mgrouped --skip-prof
   ```

   确认 `Num tests:` 打印的数量（1 个最小用例 + 3×4×2 = 24 个组合，共 25 个 `GemmTestDesc`，每个还会因 epilogue/fp32 变体派生多个 kernel）。随机挑 3 行输出，对照 3.7 节说出它的 (G, expected_m, N, K) 与 major_b。

2. **观察被钉死的配置**：

   ```bash
   export DG_PRINT_CONFIGS=1
   python tests/test_bf16_gemm.py mgrouped --skip-prof
   ```

   在配置打印中核对：所有 m-grouped kernel 的 `BLOCK_M` 恒为 256、`BLOCK_N ≤ 256`；与 `normal` 模式的配置对比，体会同一 N/K 下 Normal 可以选小 BLOCK_M 而 m-grouped 不行。

3. **Python 实验：不均匀分组 + padding 语义**：

   ```python
   import torch, torch_npu
   import deep_gemm

   K, N = 512, 256
   sizes = [300, 128, 512]                      # 3 组不均匀
   layout, psum = [], 0
   for s in sizes:
       layout.append(psum + s)
       psum = (psum + s + 255) // 256 * 256
   M = psum
   g_layout = torch.tensor(layout, dtype=torch.int32, device='npu')

   a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')
   a[300:512] = 0; a[640:768] = 0               # 组 0/1 的 padding 清零
   b = torch.randn(3, N, K, dtype=torch.bfloat16, device='npu')
   d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')

   deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, g_layout)
   torch.npu.synchronize()

   ref = torch.zeros(M, N, dtype=torch.bfloat16, device='npu')
   for g, (lo, hi) in enumerate([(0, 300), (512, 640), (768, 1280)]):
       ref[lo:hi] = a[lo:hi] @ b[g].T
   print(torch.equal(d[300:512], torch.zeros_like(d[300:512])))   # padding 行为零
   diff = 1 - 2 * (d.float() * ref.float()).sum() / (d.float()**2 + ref.float()**2).sum()
   print(diff.item() < 1e-5)                                      # 与测试同款 calc_diff（utils.py:36-43）
   ```

   再把 `a[300:512] = 0` 那行删掉（不保证零 padding），重跑并只比较 `[0,300)` 等有效区间——验证 3.5 节"kernel 也接受非零 padding"的语义。

4. **（可选）低精度路径**：`python tests/test_fp8_gemm.py mgrouped --skip-prof`，对照第 32 天的知识确认 m-grouped 的 fp8 kernel 与 bf16 共用同一套调度（`fp8_gemm.hpp:97-100`）。

### B 组：无 Ascend 环境（静态任务）

1. **psum 手算**：设 `G=4`、各组实际行数 `[100, 200, 0, 613]`、对齐 256。写出 `grouped_layout`（应为 `[100, 356, 356, 1224]`——注意空组）、总行数 M（1280）、每组的有效/padding 区间、每组的 m-block 数（1, 1, 0, 3）与全局 m_block_idx 起点（0, 1, —, 2）。对照 `generate_psum_layout`（`generators.py:212-220`）核对你的推导。
2. **调度推演**：接 3.2 的例子（`grouped_layout=[300,640,1280]`，N=7168，BLOCK_N=256，kNumCores=32），不看 3.4 节的表格，独立推演 **core 5** 的 block_idx 序列（5, 37, 69, 101, 133）各自落在哪一组、组游标如何推进、`m_block_idx/n_block_idx` 与 `get_group_idx()` 的返回值。再算 core 31 领几块（4 块：31, 63, 95, 127）。
3. **地址计算**：bf16、`N=7168, K=4096`。用 `gm_ptr::group` 公式（`common.hpp:167-172`）算 `b_group_idx=2` 相对基址的字节偏移（`2 × 7168 × 4096 × 2 = 117,440,512` 字节，即两个 expert 权重各 56 MiB）；再推 fp4（pack_size=2）时同一偏移。
4. **配置验证**：打开 `heuristics.hpp`，找到并抄录三条证据：`:104-106`（候选过滤）、`:225-227`（compute_m）、`:610-618`（m_range）；结合 `jit.hpp:59` 与 `common.hpp:57`，用一句话向自己解释"为什么 BLOCK_M=256 是布局、调度、容量三方共同决定的"。
5. **代码对照**：通读 `scheduler.hpp:144-162` 与 `bf16_gemm.hpp:89-106`，在纸上标出：分组信息从 `grouped_layout`（GM 上的 int32 数组）→ Scheduler 游标 → `get_group_idx()` → `gm_ptr::group()` → B 的实际 load 地址，一共经过几次"转手"；A 和 D 在哪一环都不出现组号。
6. **画示意图**：不看 3.8 节，独立画出"M-Grouped 输入输出示意图"（当天产出），再对照修正；重点自查 psum 的前缀和语义、组间 256 padding、B 的堆叠与 group 偏移、D 与 A 的同排列关系、以及 MoE 的 token/expert 对应。

## 六、当天产出

**M-Grouped 输入输出示意图**——以 3.8 节的参考版为答案，自己合上教程画一版，至少覆盖：`grouped_layout` 的前缀和语义与空组行为、A 的组段/padding 划分（含 M = align(psum[G-1], 256)）、B 的 `[G, N, K]` 堆叠与 `group()` 偏移公式、调度器分组游标的推进循环、Kernel 中只有 B 消费组号、D 与 A 同排列及 padding 行的语义、以及从 token/expert 到 A/B/D 的 MoE 映射。写完后把画不出的环节标出来，回到第四节对应站点重读。有环境的同学附上 A 组第 2、3 步的配置与 padding 实验记录。

## 七、自测题

1. M-Grouped 的四个张量形状是什么？A 的 major 约束是什么，为什么？
   > A `[M, K]`（必须 K-major）、B `[G, N, K]`（nt）/`[G, K, N]`（nn）、D `[M, N]`、`grouped_layout` `[G]` int32 contiguous。分组边界在行维，K-major 的 A 每行连续、组边界恰为行的整数倍；MN-major 时组段在存储上不再连续（`gemm.hpp:88-91` 的断言、`generators.py:332`）。
2. `grouped_layout[i]` 存的是组大小还是别的什么？组 g 的有效行区间怎么算？M 是多少？
   > 存前 i+1 组实际行数的**累积结尾**（prefix sum）。组 g 有效区间 = `[align(psum[g-1], 256), psum[g])`，组 0 从 0 起；`M = align(psum[G-1], 256)`（`scheduler.hpp:17-23`、`generate_psum_layout`）。
3. 描述 `get_next_block` M-Grouped 分支的循环：什么条件下推进组？推进时更新哪四个量？
   > `block_idx ≥ (current_m_block_cumsum + num_m_blocks) * num_n_blocks` 时推进；更新 `current_group_idx`（越界即任务耗尽返回 false）、`last_psum_m = aligned(current_psum_m, 256)`、`current_psum_m = grouped_layout[新组]`、`current_m_block_cumsum += num_m_blocks`、`num_m_blocks = ceil_div(current_psum_m - last_psum_m, BLOCK_M)`（`scheduler.hpp:144-162`）。游标只需前进是因为单核的 block_idx 单调递增。
4. 组内 block 坐标如何变成全局坐标？`static_assert(kAlignment == BLOCK_M)` 保证什么？
   > 组内 `get_swizzled_block(in_group_idx, 组内num_m_blocks, m, n)` 得到局部 (m, n)，随后 `m_block_idx += last_psum_m / BLOCK_M` 平移；n 不平移。断言保证组边界（256 对齐）与 block 边界重合，block 永不跨组、整除精确（`scheduler.hpp:146, :159-161`）。
5. Kernel 里组号影响了哪些张量的寻址？A 和 D 为什么不受影响？
   > 只有 B（fp8 路径还有 SFB）：`gm_b.group(b_group_idx, shape_n, shape_k)` 加 `g·N·K·sizeof(dtype)`（`bf16_gemm.hpp:100-103`、`common.hpp:167-172`）。A 的组本就拼接在平坦缓冲里，全局 `m_idx` 已落在正确组段；D 与 A 同排列，输出位置由全局 m/n 决定。
6. `get_actual_m` 在 M-Grouped 下返回什么？正确性靠什么保证？
   > 恒返回 `BLOCK_M`（TODO 待实现组尾截断，`scheduler.hpp:127-136`）。`ensure_zero_padding=true` 时调用方清零 A 的 padding 行 → D 的 padding 行为 0（此时 SFA 的 padding 可填有限垃圾，因为 0×有限 scale 仍为 0）；`false` 时只比较有效区间（`generators.py:338-357`）。
7. 为什么 BLOCK_M 被钉死为 256？涉及哪三处代码？
   > 布局约定 `kMkAlignment=256`（`jit.hpp:59`，组间对齐粒度且不许改）+ 调度断言 `kAlignment == BLOCK_M`（`scheduler.hpp:146`）+ heuristics 过滤 `block_m != 256` 的候选并固定搜索区间（`heuristics.hpp:104-106, :610-618`）；BLOCK_N 上界由 L0C 容量给出（256×256×4B=256KB 恰好占满）。
8. 空组在 M-Grouped 和 K-Grouped 下分别怎么处理？MoE 场景里这个算子对应哪两步？
   > M-Grouped：空组 `num_m_blocks=0`，while 循环直接跳过，占 0 行无需清零；K-Grouped：空组仍占 `[M, N]` 输出，需要 AIV 的 `aiv_zero_empty_groups` 清零（`scheduler.hpp:144-156`、`bf16_gemm.hpp:175-176`、`epilogue.hpp:108-135`）。MoE 里对应 Dispatch（按 expert 排列 token 得 A + grouped_layout）之后的 FFN 前后两层投影，输出由 Combine 逆排列回收（3.8 节）。

---

明天（第 37 天）把分组轴换到 K 维：阅读 K-Grouped 的 API 与 Scheduler 分支，理解"每组输出、共享 M×N 网格、K 长度不同"的另一种分组形态，并对比 Normal、M-Grouped、K-Grouped 三种 GEMM 的异同。
