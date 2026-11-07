# 第 5 周 · 第 5 天（总第 33 天）：理解 Scale 布局转换

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 5 周"测试体系与低精度 GEMM"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

昨天（第 32 天）我们留下了一个"黑盒"：不管用户给什么形态的 Scaling Factor（fp32 还是 int16、K-major 还是 MN-major），`get_gemm_desc` 都会调用 `transform_sf_into_required_layout`，把它变成 Kernel 约定的布局。今天打开这个黑盒。主线只有一句话：**Kernel 与 SF 之间存在一份固定"契约"——int16 打包的 UE8M0、`[MN, ceil(K/64)]`、MN-major 紧凑、按 `gran_mn` 广播、分组 padding 清零——Host 先用 `check_sf_layout` 探测输入是否已经满足契约（满足则零拷贝返回），否则 JIT 启动一个跑在向量核（AIV）上的 `transform_sf` Kernel，在 MTE2/V/MTE3 三级流水里完成"抽指数 → 打包 → 转置 → 广播 → 清零"。** 当天产出是**Scale 从输入到 Kernel 布局的转换图**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 Kernel 侧 SF 目标布局的三要素：**int16**（每个 int16 是一个 ue8m0x2 "SF pair"，低/高字节各覆盖 32 个连续 K 元素，合计 `MX_SF_DIVISOR = 64`，`deep_gemm/include/deep_gemm/common.hpp:65`）；**逻辑形状 `[..., MN, ceil(K/64)]` 且物理 MN-major 紧凑**（`stride(-2) == 1`、`stride(-1) == MN`，`csrc/apis/layout.hpp:36` 的返回值注释）；**内容上沿 MN 已按 `gran_mn` 广播、分组 padding 为 0**；并指出这份契约的"成文处"：`GemmDesc` 的注释（`csrc/jit_kernels/gemm_config.hpp:36-38, :62-65`）与 `fp8_gemm_impl` 里 `gm_sfa/gm_sfb` 的构造（`deep_gemm/include/deep_gemm/fp8_gemm.hpp:63-66`）；
2. 解释 `recipe` 的解析规则：`get_recipe`（`csrc/apis/layout.hpp:113-128`）合并三元组 `(gran_m, gran_n, gran_k)` 或两个二元组（缺省 `(1, 32)`），两侧 `gran_k` 必须一致，且 `check_sf_layout` 钉死 `gran_k == 32`（`apis/layout.hpp:55`）；变换时 A 用 `gran_m`、B 用 `gran_n` 作为 `gran_mn`（`:79-81`）；
3. 逐条说出 `check_sf_layout`（`apis/layout.hpp:37-65`）的断言（dtype、维度、连续性、形状、`gran_k`），以及**零拷贝快速路径**的触发条件（short + `gran_mn == 1` + 紧凑 MN-major + 稠密 batch stride，`:57-62`）；
4. 画出 Host 调用链：`get_gemm_desc` 中 sfa/sfb 的调用点（`csrc/apis/gemm.hpp:125-154`）→ `launch_transform_sf`（`csrc/jit_kernels/transform_sf.hpp:15-105`）的五个步骤（输出张量构造、tiling 选择、L2 hint、JIT 代码生成、launch），并说明四种 GemmType 下 sfa/sfb 走的差异分支（M-grouped 的 sfb 走 Batched、K-grouped 两者都走 `transform_k_grouped_sf_into_required_layout`）；
5. 讲清设备 Kernel `transform_sf`（`deep_gemm/include/deep_gemm/transform_sf.hpp:447-639`）的结构：`__global__ __vector__`（AIV Kernel）、按 `2 × num_sms` 个向量核切分块任务、**四条计算路径**（fp32/int16 × K-major/MN-major）各自做什么、`repeat_simd` 如何沿 MN 广播 `gran_mn` 倍、`prepare_grouped_input` 的两种 padding 处理机制（行清零 vs 掩码 AND）；
6. 手算 UE8M0 打包的位级过程：`>> 23` 抽指数字节、两个相邻 K 粒度的指数拼成一个 int16（小端低字节在前）、`check_validation` 为什么断言符号位与尾数全 0（`transform_sf.hpp:13-21, :635-638`），以及分组掩码为什么必须在汇入校验累加器**之前** AND（`:269-275`）；
7. 描述转换 Kernel 的三级流水：MTE2（GM→UB）/ V（计算）/ MTE3（UB→GM）按 stage 轮转，`notify/wait` 四组配对（`:577-628`），级数由 UB 容量除以每级字节数得出、上限 8、下限 2（`:505-512`）；
8. 回答计划的核心问题——**为什么不能总是直接使用用户输入的 Scale 布局**（格式、方向、粒度、padding、批次紧凑性五条原因，加上"什么时候可以"的反例），并独立画出 Scale 从输入到 Kernel 布局的转换图（当天产出）。

## 二、预备知识

- **第 4 天**：BF16/FP8/FP4、Scaling Factor 与量化粒度——今天的 `recipe`、UE8M0 都是它的落地。
- **第 10～11 天**：Python binding 与 `GemmDesc`——今天关注 `sfa_ptr/sfb_ptr` 的来源，以及 `GemmDesc.sfa/sfb` 为什么要持有变换后的张量防止提前释放（`gemm_config.hpp:77-78`）。
- **第 12 天**：JIT 模板代码生成、`jit->compile()/launch()`——`launch_transform_sf` 的 `std::format` 实例化直接复用这套机制。
- **第 18 天**：`gm_ptr` 的 `index()/batch()` 与 stride 语义——今天 MN-major 的地址公式（`common.hpp:156-165`）是理解"为什么必须转置"的钥匙。
- **第 19、23 天**：`copy_gm_to_ub/copy_ub_to_gm`、多级流水与 `asc_sync_notify/wait` 配对——`transform_sf` 是同一套模式在 AIV（MTE2/V/MTE3）上的版本。
- **第 29～30 天**：generators 的 psum 布局（`iter_psum_sep`）、`par_compile` 与 bench——今天直接消费。
- **第 32 天**：FP8/FP4 路径与 SF 消费链（`copy_gm_to_l1_mx` → `copy_l1_to_l0a/b_mx` → `asc_mmad_mx`）——今天从"消费端"反推"生产端"的契约。

另注：第五节 A 组任务需要 Ascend 环境；无环境时完成 B 组静态任务（打包手算、形状推演、快速路径判定、分组填充推演、路径分派表、转换图）即可达到当天目标。

## 三、详细讲解

### 3.1 目标布局：Kernel 与 SF 的"契约"

先站在消费端。`fp8_gemm_impl` 对 SF 的全部假设浓缩在三行代码里（`fp8_gemm.hpp:63-66`）：

```cpp
const auto shape_sf_k = ceil_div(shape_k, MX_SF_DIVISOR);   // int16 列数 = ceil(K/64)
const gm_ptr<int16_t, Major::MN> gm_sfa(shape_m, gmem_sfa, static_cast<uint64_t>(shape_m) * shape_sf_k);
const gm_ptr<int16_t, Major::MN> gm_sfb(shape_n, gmem_sfb, static_cast<uint64_t>(shape_n) * shape_sf_k);
```

`gm_ptr` 构造参数是 `(stride_outer, addr, stride_batch)`（`common.hpp:147-148`）。对 MN-major，`index(mn_idx, k_idx)` 的地址公式是 `addr + sizeof(T) * (k_idx * stride_outer + mn_idx)`（`common.hpp:161-163`）——即**同一个 k 对的所有 MN 值在内存中相邻，k 对轴的步长恰为 MN**。翻译成 PyTorch 语言：逻辑 `[MN, ceil(K/64)]`、`stride(-2) == 1`、`stride(-1) == MN`。`GemmDesc` 的注释把这份契约定为仓库级规范："D is always N-major and SF is always MN-major"（`gemm_config.hpp:36-38`），Batched 情况下 "SFA/SFB are host-normalized to a dense `[B, mn, k/64]` layout, so their per-batch block is `mn*(k/64)` (no extra stride needed)"（`:62-65`）。

契约三要素与消费链的对应：

| 要素 | 内容 | 消费端为什么需要 |
|---|---|---|
| 类型 | int16 = ue8m0x2 pair：低字节覆盖 K∈[64p, 64p+32)、高字节覆盖 [64p+32, 64p+64) | `asc_mmad_mx` 在 Cube 内按 64 元素组应用 UE8M0（第 32 天 3.4 节），`copy_l1_to_l0a/b_mx` 按 pair 搬运（`ascend/copy.hpp:199-229`） |
| 形状/方向 | `[..., MN, ceil(K/64)]`，物理 MN-major 紧凑 | `copy_gm_to_l1_mx` 的 nd2nz 把"每个 k 对的一行 MN int16"整段突发进 L1 的 NZ 分形（`copy.hpp:161-197`）；MN 不连续就无法整段传输 |
| 内容 | 沿 MN 已按 `gran_mn` 广播到每一行；分组 padding 为 0 | `copy_l1_to_l0a_mx` 的 `x_start = m_idx / FRAC_MN`、`x_step = shape_mn / FRAC_MN`（`copy.hpp:207-209`）按 16 行分形**统一步进**寻址，要求每个 MN 行都有自己的 int16 |

第三条值得展开：如果 SF 保持"每 `gran_mn` 行一个值"的压缩形态，那么一个 16 行分形内部的行可能映射到不同的 scale 行（`m / gran_mn` 不是线性 stride），L1→L0 的搬运就必须逐行 Gather。把广播**实际写出到内存**（`repeat_simd`）之后，SF 与数据可以使用完全相同的分形寻址——这是用一次廉价变换换取 Kernel 内循环零开销的典型取舍。

### 3.2 recipe：量化粒度的表达与解析

`recipe` 回答"多少个元素共享一个 scale"。Host 侧统一由 `get_recipe` 归一（`apis/layout.hpp:113-128`）：

| 用户传法 | `gran_mn`（A / B） | `gran_k` |
|---|---|---|
| `recipe = (gran_m, gran_n, gran_k)` | `gran_m` / `gran_n` | `gran_k` |
| `recipe_a = (gran, gran_k)`（可缺省） | `gran`（缺省 1） | `gran_k`（缺省 32） |
| `recipe_b` 同上 | `gran`（缺省 1） | 必须与 `recipe_a` 相同（`:125`） |

两种传法互斥（`:118`）。随后的 `transform_sf_into_required_layout` 用 `is_sfa` 决定取哪个分量（`:79-81`）：A 的 SF 用 `gran_m`、B 的用 `gran_n`，`gran_k` 共用。最后 `check_sf_layout` 把 `gran_k` 钉死为 32（`:55`）——这正是 MX 标准粒度，也是"一个 int16 pair 覆盖 64 个 K"的来源。测试直接用对称 recipe `(gran_mn, gran_mn, 32)` 遍历 `gran_mn ∈ {1, 32, 128}`（`tests/test_layout.py:99`）。

### 3.3 输入空间与零拷贝快速路径

与目标布局的"一种"相对，用户输入是一个**组合空间**：

| 维度 | 取值 | 检查点（`apis/layout.hpp`） |
|---|---|---|
| dtype | fp32（每 32 个 K 一个 `2^e`，尚未打包）或 int16（已打包 ue8m0x2，每 64 个 K 一个） | `:43` |
| major | K-major（`stride(-1) == 1`）或 MN-major（`stride(-2) == 1`） | `:45-46` |
| 形状 | fp32：`[ceil(MN/gran_mn), ceil(K/32)]`；short：`[ceil(MN/gran_mn), ceil(K/64)]` | `:51-54`（`sf_pack_k = 2` 只对 short 生效） |
| 维度 | 2D；Batched 时 3D 且 `size(-3) == num_groups` | `:44, :48-49` |

`check_sf_layout` 的最后一段（`:57-62`）是快速路径：

```cpp
if (sf.scalar_type() == at::kShort and gran_mn == 1 and sf.stride(-2) == 1 and sf.stride(-1) == sf.size(-2)) {
    if (num_groups.has_value())
        return sf.stride(-3) == sf.size(-2) * sf.size(-1);   // batch 还必须稠密
    else
        return true;
}
```

四个条件恰好是 3.1 契约在"无需任何变换"时的特例：已是 int16（无需打包）、`gran_mn == 1`（无需广播）、紧凑 MN-major（无需转置）、稠密 batch（无需规整 stride）。命中时 `transform_sf_into_required_layout` 原样返回张量（`:83-84`），**一次拷贝都不发生**——测试用 `output.data_ptr() == sf.data_ptr()` 探测这条路径（`test_layout.py:149-150`）。任何一条不满足（最常见的就是用户给 fp32 或 K-major）都会走完整的 Kernel 变换。

另外注意 `disable_ue8m0_cast`（`:86-87`）：输入是 fp32 且该开关为 true 时直接断言失败。它用于禁止"用户以为传了精确 scale、实际被静默 cast 成 2 的幂"的隐患——语义上等价于"你必须自己保证 fp32 输入是合法 UE8M0"。

### 3.4 Host 链路：从 `get_gemm_desc` 到 `launch_transform_sf`

调用点全景（`csrc/apis/gemm.hpp:125-154`）：

```text
fp8_fp4_gemm_* / m_grouped_* / k_grouped_*（Python API）
  └→ get_gemm_desc（m/n/k 由 get_shape_by_spec 从张量标注推出，:81-97；
      注意 nt/tt 等逻辑转置会同时作用于 (A, sfA) 二元组，:239-240）
      ├→ layout::get_recipe(recipe, recipe_a, recipe_b)                    (:127)
      ├→ sfa = transform_sf_into_required_layout(sfA, m, k, recipe, …)     (:128-140)
      └→ sfb = transform_sf_into_required_layout(sfB, n, k, recipe, …)     (:141-154)
```

四种 GemmType 下两个 SF 走的分支并不相同：

| GemmType | sfa（A 的 SF，mn=m） | sfb（B 的 SF，mn=n） |
|---|---|---|
| Normal | 无组信息 | 无组信息 |
| Batched | `num_groups = g` | `num_groups = g` |
| M-Grouped（psum 布局） | `psum_layout = grouped_layout` | `num_groups = g`——B 是 `[G, N, K]` 堆叠，SF 天然带组维，按 Batched 处理（`:147-148`） |
| K-Grouped | `transform_k_grouped_sf_into_required_layout(…, psum)`（`:130`） | 同左（`:143`） |

`launch_transform_sf`（`csrc/jit_kernels/transform_sf.hpp:15-105`）做五件事：

1. **规整维度与探测输入形态**：2D 输入 `unsqueeze(0)`（`:23-24`）；`major = sf.stride(-1) == 1 ? K : MN`（`:30`）；`gran_mn` 必须是 `[1, 128]` 内的 2 的幂（`:25-26`）。
2. **构造输出张量**（`:45-53`）：

   ```cpp
   const uint32_t src_shape_m = sf.size(-2);                 // ceil(MN / gran_mn) 个 scale 行
   const uint32_t src_shape_k = sf.size(-1);                 // ceil(K/32)（float）或 ceil(K/64)（short）
   const uint32_t dst_shape_m = shape_m;                     // MN（广播后的行数）
   const uint32_t dst_shape_k = ceil_div(src_shape_k, is_float_input ? 2 : 1);  // int16 对数
   auto physical_sizes = sf.sizes().vec();
   physical_sizes[physical_sizes.size() - 2] = dst_shape_k;
   physical_sizes[physical_sizes.size() - 1] = dst_shape_m;
   auto output = torch::empty(physical_sizes, sf.options().dtype(at::kShort)).transpose(-2, -1);
   ```

   物理 `[..., ceil(K/64), MN]` 的稠密 int16，再 `transpose(-2, -1)` 得到逻辑 `[..., MN, ceil(K/64)]` 且 `stride(-2) == 1`——正是 3.1 的契约布局。
3. **选择 tiling**（`:57-65`，单位：`src_block_m` 是 scale 行数；`src_block_k` 对 float 是 fp32 元素数、对 short 是 int16 对数；目标块 `kDstBlockM = kSrcBlockM × kGranMN`、`kDstBlockK = kSrcBlockK / 2`（float））：

   | 输入 major | gran_mn | src_block_m | src_block_k |
   |---|---|---|---|
   | MN | 1 | `min(512, 8192)` = 512 | 32 |
   | MN | 32 | 256 | 8（float）/ 4（short） |
   | MN | 128 | 64 | 8（float）/ 4（short） |
   | K | 1 | 128 | 128 |
   | K | 32 | 8 | 128 |
   | K | 128 | 2 | 128 |

   规律：MN-major 的目标块 MN 宽度恒为 8192（`8192 / gran_mn`），K-major 的 K 宽度恒为 128——都是为了让 src+dst 的 UB 占用落在 2～8 级流水内（见 3.7）。
4. **决定 L2 hint**（`:68-71`）：输入外层 stride 折算字节 > 32 时默认 `NOTALLOC_KEEP`（跨行访问稀疏，不污染 L2），否则 `NORMAL_FIRST_VICTIM`；可用环境变量 `DG_TRANSFORM_SF_LD_CACHE_HINT` / `DG_TRANSFORM_SF_ST_CACHE_HINT` 覆盖。
5. **JIT 代码生成与启动**（`:72-102`）：`std::format` 拼出 `transform_sf<kFloat, kMajor, src_block_m, src_block_k, gemm_type, num_vector_cores, 256, gran_mn, l2…>` 的实例化代码，`jit->compile` 后以 `num_blocks = num_vector_cores = 2 × num_sms`（950DT 上为 64 个向量核）launch。传入的源 `gm_ptr` 以元素为单位携带外层 stride（K-major 用 `stride(-2)`、MN-major 用 `stride(-1)`，`:68`）与 batch stride（`stride(-3)`），Kernel 侧 `index()` 再乘 `sizeof(InputT)`（`common.hpp:156-165`）——因此用户的 strided/permuted 输入无需先 contiguous。

### 3.5 设备 Kernel：四条路径 + 广播 + 分组清零

`transform_sf`（`deep_gemm/include/deep_gemm/transform_sf.hpp:447-639`）是 `__global__ __vector__` Kernel——跑在 AIV 上，与 GEMM 主 Kernel 的 AIC 无关。任务划分：`ceil(M/srcBlockM) × ceil(K/srcBlockK) × num_batches` 个块均摊到向量核（`:530-537`）；块内枚举顺序保证分组轴单调推进（M-grouped 时 m 为慢轴、K-grouped 时 k 为慢轴，`:550-551`），这是 `prepare_grouped_input` 能顺序遍历 `grouped_layout` 的前提。

每个块的处理流程（省略同步，见 3.7）：

```text
GM → UB（copy_gm_to_ub，actual 之外补 0，asc_set_copy_pad_val(0)，:472）
  → [分组] prepare_grouped_input：确定组尾 padding 并清零/建掩码（:586-593）
  → 计算路径四选一（:595-612）：
     ① fp32 + K-major ：pack_validate_simd → packed_ub；transpose_simd → dst_ub
     ② fp32 + MN-major：pack_validate_simd 直写 dst_ub（打包即落位）
     ③ int16 + K-major：transpose_simd（仅转置）
     ④ int16 + MN-major：asc_copy_ub2ub（仅拷贝）
  → [gran_mn > 1] repeat_simd 在 dst_ub 原地把每列值沿 MN 广播 gran_mn 倍（:616-619）
  → UB → GM（copy_ub_to_gm，行 stride = dst_shape_m × 2B，:624-626）
```

| 路径 | 需要的动作 | 用到的缓冲 |
|---|---|---|
| ① fp32 + K | 抽指数、两行打包、再转置 | src_ub（u32）+ packed_ub（u16）+ dst_ub |
| ② fp32 + MN | 抽指数、相邻 K 行配对，一步落位（输入已转置） | src_ub + dst_ub（无 packed） |
| ③ short + K | 仅转置（bit 原样） | src_ub + dst_ub |
| ④ short + MN | 仅拷贝（值与相对位置都不变） | src_ub + dst_ub |

`kHasPackedBuffer = kFloat and kMajor == Major::K`（`:499`）——只有路径①需要中间的 packed 缓冲。

**广播**由 `repeat_simd`（`:181-201`）完成：目标宽度 256 走 `repeat_to_256_simd_impl`（`asc_loadalign_brc_elem2datablock` 把单个元素广播成 32 字节 datablock，再用 `asc_intlv` 蝶形逐级翻倍，`:89-121`），1024 的倍数走 `repeat_to_1024x_simd_impl`（逆序按组处理，`:123-179`）。两个实现都**逆序遍历行/组**（`:193, :146`）——因为扩张是原地进行的，倒着写才不会覆盖尚未读取的数据。

**分组 padding** 是 M-Grouped/K-Grouped 特有的问题：`grouped_layout` 是各组**真实结束位置**的前缀和，而矩阵按 256 对齐（`Runtime::get_mk_alignment_for_contiguous_layout()`，`csrc/jit/jit.hpp:59`），组尾到对齐边之间是 padding。`prepare_grouped_input`（`:374-445`）以 `kGranGroup` 为单位（M-grouped 用 `kGranMN`，K-grouped 用 32（float）/ 64（short），`:494`）把 `grouped_layout` 换算成"scale 元素"区间的组尾，然后二选一：

- **行粒度清零**（`kHasGroupedFlag == false`，即 padding 落在行维度：M-grouped+K-major、K-grouped+MN-major）：`clear_group_tail_simd` 直接把 src_ub 的尾行清 0（`:350-364`）；
- **列方向掩码**（`kHasGroupedFlag == true`，即 padding 落在连续维度：M-grouped+MN-major、K-grouped+K-major，`:500`）：在标量管上维护 `grouped_flag`（有效位置全 1、尾位置 0），打包时与数值 AND（`pack_validate_*` 的 `:234-240, :269-273`）。

一个重要的代码事实：**掩码只在两条 fp32 打包路径里被消费**；short 的两条纯搬运路径（③④）不消费掩码。因此测试对 int16 输入直接把 padding 填 0（`invalid_value = 0 if dtype == torch.int16 else -1`，`test_layout.py:91`），而对 fp32 填 -1.0——后者带符号位，专门用来验证清零/掩码确实生效：如果不处理，它会污染校验累加器导致 Kernel 误报（见 3.6）。

### 3.6 UE8M0 打包与校验的位级细节

两个相邻 K 粒度的 fp32 scale 打包成一个 int16：

```text
fp32 scale₀ / scale₁（各覆盖 32 个 K）              打包后的 int16（SF pair）
┌──┬──────────┬───────────────┐                  ┌───────────┬───────────┐
│s │ e₇ … e₀  │  23 位尾数     │    >>23 / >>15   │ e₁（高字节）│ e₀（低字节）│
└──┴──────────┴───────────────┘                  └───────────┴───────────┘
scale₀ = 2^e₀，覆盖 k ∈ [64p, 64p+32)              值 = (e₀, e₁)，覆盖 64 个 K
scale₁ = 2^e₁，覆盖 k ∈ [64p+32, 64p+64)           （小端：e₀ 在低字节）
```

- **K-major**（`pack_validate_k_major_2row_simd_impl`，`:216-248`）：一次处理两行 scale；每行 `asc_shiftright_scalar(sf, sf, 23)` 把指数字节挪到低位，`asc_storealign_pack_quarter_postupdate` 把 4 个 u32 压成 4 个字节——行内的相邻指数自然配成 pair；
- **MN-major**（`pack_validate_mn_major_simd_impl`，`:250-281`）：`sf0 >> 23`（偶数 K 行的指数落 bit 0-7）、`sf1 >> 15`（奇数 K 行的指数落 bit 8-15），`asc_or` 合成一个 u32 后 `asc_storealign_pack_postupdate` 压成 u16——**抽指数与配对一步完成**，且输入已是转置形态，无需再 transpose。

**校验**是 fp32 路径的独有防线：打包前把原始 fp32 位 OR 进 `block_acc0/1`（注意分组掩码的 AND 发生在 OR **之前**，`:241-242, :274-275`——padding 里的 -1.0 不能进校验），逐块 merge 进 `validation_ub`（`:203-214`），Kernel 末尾 `check_validation` 断言 `(combined & 0x80000000) == 0 and (combined & 0x007fffff) == 0`（`:13-21, :635-638`）——即所有 scale 非负且无尾数，否则以 "Float32 SF must be non-negative and can't have non-zero mantissa" 终止。

这也解释了一个初看奇怪的操作：fp32 路径在 Kernel 开头把**整个 UB 清零**（`clear_ub_simd<UBSizeBytes>`，`:525-528`）。打包按整向量宽度读取（`:230, :265`），块内 actual 之外的列不会被 GM→UB 拷贝覆盖，却仍会被 OR 进校验累加器——只有保证这些位置自始至终是 0，校验才不会误报。short 路径不清零：超出 actual 的转置/拷贝结果只留在 UB，不会被 `copy_ub_to_gm` 写回（它只写 `dst_actual_k × dst_actual_m`）。同理，奇数行会破坏"两行一组"的打包，故 `num_rows % 2 != 0` 时把补齐行整行清零（`:318-322`）。

**手算例子**（`gran_mn = 2`，M=4，K=128，fp32 输入）：

```text
用户 sf（fp32，[2, 4]）：                    输出（int16，逻辑 [4, 2]）
行 0（→ M 行 0,1）：2^1 2^2 2^3 2^4          out[0] = [0x0201, 0x0403]
行 1（→ M 行 2,3）：2^5 2^6 2^7 2^8          out[1] = [0x0201, 0x0403]
                                             out[2] = [0x0605, 0x0807]
                                             out[3] = [0x0605, 0x0807]
物理内存（[K 对][MN]，MN 连续）：
  p=0: 0x0201 0x0201 0x0605 0x0605        p=1: 0x0403 0x0403 0x0807 0x0807
```

`0x0201` = 低字节 0x01（e₀=1，覆盖 k∈[0,32)）+ 高字节 0x02（e₁=2，覆盖 k∈[32,64)）。测试的 `reference`（`test_layout.py:39-50`）就是这段语义的 PyTorch 版：`// (1<<23) & 0xff` ↔ 抽指数、奇数补零 ↔ pair 对齐、`view(torch.int16)`（小端低字节在前）↔ 打包次序、`repeat_interleave(gran_mn)` ↔ `repeat_simd`、最后 `transpose` ↔ MN-major。

### 3.7 AIV 三级流水：MTE2 / V / MTE3

`transform_sf` 的块循环是一个教科书式的多级流水（第 23 天 AIC 版本的同构）：`kNumStages = min(8, kValidUBBytes / kBytesPerStage)`，且 `static_assert(kNumStages >= 2)`（`:509-512`）。每级 = src_ub（+packed_ub）+ dst_ub 的字节数；UB 总量 256KB 中还要预留 8KB 谓词寄存器溢出区与 64×u32 校验区（`:505-508`）。例如 fp32+K-major+gran=1：src 128×128×4B=64KB + packed 128×64×2B=16KB + dst 64×128×2B=16KB ≈ 96KB/级 → 2 级；gran=32 时缩到约 42KB/级 → 5 级。dst 的行距在 K-major 时额外 +16 个 u16 以避免转置存取的 bank 冲突（`:489` 注释）。

stage 事件配对（`asc_sync_notify(源, 目的, e)` / `asc_sync_wait(源, 目的, e)`，参数序为"信号方→等待方"）：

| 事件 | notify | wait | 语义 |
|---|---|---|---|
| V → MTE2 | `:614` | 下一轮 `:577` | src 本级已被消费，可覆写 |
| MTE2 → V | `:582` | `:583` | src 数据已到 UB |
| V → MTE3 | `:621` | `:622` | dst 数据就绪 |
| MTE3 → V | `:628` | 下一轮 `:584` | dst 本级已写回，可覆写 |

开跑前 `set_flags` 预置全部"src/dst 空闲"事件（`:543-544`），收尾 `wait_flags` 排空残留事件（`:632-633`）。与 GEMM 主 Kernel 的 MTE2/MTE1/M 流水对照：搬运与计算的角色一一对应，只是"消费者"从 Cube 换成了向量管 V。

### 3.8 为什么不能总是直接使用用户输入的 Scale 布局（含当天产出·参考图）

汇总五条原因（每条都能在源码里找到"硬依据"）：

1. **数值格式不匹配**：Cube 的 `asc_mmad_mx` 只消费 UE8M0（指数加法代替浮点乘）；fp32 必须抽指数并两两打包成 int16。这是**位级重排**，任何视图/stride 都无法表达。
2. **方向不匹配**：Kernel 端 `gm_sfa/gm_sfb` 固定 MN-major（`fp8_gemm.hpp:65-66`）；`copy_gm_to_l1_mx` 的 nd2nz 要求每个 k 对的 MN 段连续才能整段突发。用户恰恰常给 K-major——沿 K 生成 scale 的自然布局就是 K-major。
3. **粒度展开**：`gran_mn > 1` 时用户每 `gran_mn` 行一个值，而 Kernel 按"每 MN 行一个 int16"寻址（`copy.hpp:207-209` 的 16 行分形步进）。行号到 scale 行的映射是 `m / gran_mn`——**整数除法不是线性 stride**，只能实际复制（`repeat_simd`）。
4. **分组 padding**：M/K-grouped 的组尾填充必须规范化为 0；fp32 的填充若带符号/尾数还会触发校验误报（3.6 节），所以清零/掩码必须在打包路径上强制执行。
5. **批次紧凑性**：Batched/Grouped 的输出被 `GemmDesc` 约定为稠密 `[B, MN, K/64]`、Kernel 不携带 SF 的外层 stride（`gemm_config.hpp:62-65`）——用户任意 permute/stride 的输入必须先规整。

反向的例外同样重要：输入恰好满足全部约定时（short、`gran_mn == 1`、紧凑 MN-major、稠密 batch），`check_sf_layout` 原样返回——**能省则省，不能省才启动 Kernel**。

**Scale 从输入到 Kernel 布局的转换图（当天产出·参考版）**：

```text
┌───────────────────── 用户输入 SF ─────────────────────┐
│ dtype: fp32 [ceil(MN/gran), ceil(K/32)]               │
│      或 int16 [ceil(MN/gran), ceil(K/64)]（已打包）   │
│ major: K-major 或 MN-major；可带 batch/组维、可 strided│
└──────────────────────────┬────────────────────────────┘
                           ▼
        check_sf_layout（apis/layout.hpp:37-65）
        short ∧ gran_mn==1 ∧ 紧凑 MN-major ∧ 稠密 batch？
              │是                          │否
              ▼                            ▼
        原样返回（零拷贝）      launch_transform_sf（jit_kernels/transform_sf.hpp）
              │                  ├─ 输出：物理 [.., ceil(K/64), MN] int16 → transpose
              │                  ├─ tiling：按 major/gran_mn 选块（:57-65）
              │                  └─ JIT 实例化 transform_sf<…> → launch（2×num_sms 个 AIV）
              │                            ▼
              │        ┌────── AIV Kernel transform_sf（三级流水，多 stage）──────┐
              │        │ GM→UB（MTE2，pad 0）                                     │
              │        │ ① fp32+K ：抽指数>>23 → 两行打包 → 16 行蝶形转置          │
              │        │ ② fp32+MN：>>23 / >>15 → OR 配对 → 直写 dst（免转置）     │
              │        │ ③ int16+K ：仅转置     ④ int16+MN：仅拷贝                 │
              │        │ （分组）行清零 或 grouped_flag 掩码 AND（仅 fp32 路径）    │
              │        │ （gran_mn>1）repeat_simd 沿 MN 广播 gran_mn 倍（原地逆序）│
              │        │ UB→GM（MTE3，行 stride = MN×2B）                          │
              │        └──────────────────────────┬───────────────────────────────┘
              ▼                                     ▼
┌──────────────────── 目标布局（契约）─────────────────────┐
│ int16（ue8m0x2：低/高字节各覆盖 32 个 K，合计 64）        │
│ 逻辑 [.., MN, ceil(K/64)]；物理 MN-major：stride(-2)=1， │
│ stride(-1)=MN；MN 已广播；分组 padding = 0               │
└──────────────────────────┬───────────────────────────────┘
                           ▼
GemmDesc.sfa/sfb（持有防释放，gemm_config.hpp:77-78）
  → fp8_gemm_impl: gm_sfa(MN-major) → copy_gm_to_l1_mx → l1_sfa
  → copy_l1_to_l0a/b_mx → asc_mmad_mx（UE8M0 在 Cube 内应用，L0C fp32 累加）
```

## 四、源码导读

今天的主线文件与阅读站点（建议按表中顺序读）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `csrc/apis/layout.hpp` | `check_sf_layout`（:37-65）、`transform_sf_into_required_layout`（:68-92）、`transform_k_grouped_sf_into_required_layout`（:94-111）、`get_recipe`（:113-128）、`register_apis`（:130-140） | 今天的主线 Host 逻辑；顶部注释（:19-36）是接口文档 |
| 2 | `csrc/jit_kernels/transform_sf.hpp` | `launch_transform_sf`（:15-105） | 输出构造、tiling、L2 hint、代码生成、launch 参数 |
| 3 | `deep_gemm/include/deep_gemm/transform_sf.hpp` | `transform_sf`（:447-639）与 SIMD 族 | 设备侧主战场，见下方逐段导读 |
| 4 | `tests/test_layout.py` | 全文件 | 测试组织与参考实现，见下方导读 |
| 5 | `tests/generators.py` | `enumerate_sf_layout`（:682-690）、`enumerate_k_grouped_psum_sf_layout`（:716-722）、`enumerate_sf_layout_performance_tests`（:725-747）、`iter_psum_sep`（:230-235） | 用例空间与 psum 区间换算 |
| 6 | `csrc/apis/gemm.hpp` | `get_gemm_desc` 的 SF 段（:125-154）、`get_transposed`（:239-240） | 调用点与四种 GemmType 的分支差异 |
| 7 | `csrc/utils/layout.hpp` | `major_check`（:37-42）、`get_major_type_ab`（:44-47）、`get_shape_by_spec`（:140-219） | 变换所依赖的 m/n/k 与 major 事实从哪来；`_sp` 字面量标注机制 |
| 8 | `deep_gemm/include/deep_gemm/fp8_gemm.hpp` | :63-66（gm_sfa/gm_sfb）、:82-83（l1_sfa/sfb）、:178-185（chunk 搬运）、:209-224（sf_y 与 L0 装载） | 消费端契约的反向印证 |
| 9 | `deep_gemm/include/deep_gemm/ascend/copy.hpp` | `copy_gm_to_l1_mx`（:161-197）、`copy_l1_to_l0a/b_mx`（:199-229） | MN-major 紧凑布局的搬运学依据 |
| 10 | `deep_gemm/include/deep_gemm/common.hpp` | `MX_SF_DIVISOR`（:65）、`gm_ptr`（:137-186） | 常量与地址公式 |
| 11 | `csrc/jit_kernels/gemm_config.hpp` | `GemmDesc` 注释（:36-38, :62-65）、`sfa/sfb` 持有（:77-78） | 契约的"成文处" |
| 12 | `csrc/python_api.cpp` / `deep_gemm/__init__.py` | :38 注册 / :79-80 导出 | Python 入口：`deep_gemm.transform_sf_into_required_layout` 等 |

**`transform_sf` 逐段导读**（行号见 `deep_gemm/include/deep_gemm/transform_sf.hpp`）：

- **`:447-464` 签名与模板**：`__global__ __vector__`；模板依次是 `kFloat`（输入 fp32?）、`kMajor`（输入方向）、`kSrcBlockM/kSrcBlockK`（源 tiling）、`kGemmType`、`kNumCores`（默认 64，实际由 launch 的 `2 × num_sms` 覆盖）、`kAlignment`（256）、`kGranMN`、两个 L2 模式。
- **`:468-496` 派生常量**：`kNumElemsPerPackedSF = kFloat ? 2 : 1`；`kDstBlockM = kSrcBlockM × kGranMN`、`kDstBlockK = kSrcBlockK / kNumElemsPerPackedSF`；UB 的行/列按输入 major 互换（K-major 时行=MN、MN-major 时行=K）；`kDstUBStride = kDstUBCols + (K-major ? 16 : 0)`（:489 防 bank 冲突）；`kGranGroup` 与 `kHasGroupedFlag`（:494, :500）决定分组处理路线。
- **`:499-523` 级数与 UB 布局**：`kReservedUBBytes = 8KB`（谓词寄存器溢出区）、`kValidationElems = 64`（仅 fp32）；`kNumStages = min(8, 容量/每级)`、`static_assert >= 2`；`ub_ptr` 依 offset 串联 src/packed/dst/validation/grouped_flag/tail，末尾 `static_assert` 防 UB 溢出。
- **`:525-528`**：fp32 路径先清零整个 UB（3.6 节解释了为什么）。
- **`:530-545` 块调度**：块数 = `ceil(M/srcM) × ceil(K/srcK) × batches`，均摊给 `kNumCores`；`set_flags` 预置两组初始事件。
- **`:546-575` 块循环头**：分解 batch/m/k 下标（分组轴慢速推进，:550-551）；`src/dst_actual_*` 截断边界；选中本级的三个 UB stage。
- **`:577-584` GM→UB**：`copy_gm_to_ub` 搬 actual 区域，前后两组 wait/notify 构成流水握手。
- **`:586-593` 分组准备**：`prepare_grouped_input` 维护 `current_group_idx/grouped_tail`，仅当块下标变化时重算。
- **`:595-612` 四条计算路径**：3.5 节的表格逐行对应；注意 ③ 用的是 `kSrcUBRows/kSrcUBCols`（short 无 packed 中转）。
- **`:614-619`**：释放 src；`kGranMN > 1` 时 `repeat_simd` 原地广播。
- **`:621-630` UB→GM**：`dst_offset = (batch × dst_shape_k + dst_k_idx) × dst_shape_m + dst_m_idx`（:624，即契约布局的线性下标）；store 后推进 stage。
- **`:632-638` 收尾**：排空事件；fp32 路径 `asc_sync_pipe(PIPE_ALL)` 后执行 `check_validation`。

**SIMD 辅助函数**（同文件，按需精读）：`transpose_16rows_simd_impl`（:23-71，`asc_intlv` 蝶形网络，`kSrcCols` 只允许 32/64/128）、`repeat_to_256/1024x_simd_impl`（:89-179，元素广播 + 翻倍）、`pack_validate_k_major_2row/mn_major_simd_impl`（:216-281，3.6 节位级细节）、`fill_bytes/clear_group_tail/clear_ub`（:283-372，清零族）、`merge_validation`（:203-214）。

**`tests/test_layout.py` 逐函数导读**：

- `make_sf`（:24-28）：fp32 用 `randint(1,255) × (1<<23)` view 成 float——只会生成合法 `2^e`；short 用随机 int16——任意 bit 模式都是合法 ue8m0x2。
- `with_major`（:31-37）：`transpose → contiguous → transpose` 得到指定 major 的物理布局，并断言 stride。
- `reference`（:39-50）：3.6 节手算例子的通用版——纯 PyTorch 的"正确答案"。
- `check_layout`（:53-58）：五重断言（dtype/shape/`stride(-2)==1`/`stride(-1)==size(-2)`/逐元素相等）——契约的可执行版本。
- `make_layout_case`（:75-109）：按 case 元组构造输入、填充分组 padding（expected 填 0、输入填 invalid）、组装 `run`（普通/K-grouped 两个入口）与 `expected`。
- `test_sf_layout`（:112-179）：用例 = `enumerate_sf_layout`（324 个 Normal/Batched）+ `enumerate_k_grouped_psum_sf_layout` 派生的 M/K-grouped（192 个）+ 性能用例（162 个），合计 678 个；按 `(dtype, gemm_type, major)` 分组执行；先用 `par_compile` 并行编译去重后的 48 个模板实例（4 GemmType × 2 dtype × 2 major × 3 gran_mn，:167-175）；性能模式用 `bench_msprof(kernel_names='transform_sfI', flush_l2=True)` 输出耗时、（输入+输出字节）/时间的带宽、vec/scalar/mte2/mte3 四个管道占用率（:152-163）；快速路径命中时改测 Host 开销（:149-150）。
- CLI（:182-196）：`python tests/test_layout.py [all|correctness|performance] [--skip-prof]`。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通正确性测试**：

   ```bash
   python tests/test_layout.py correctness
   ```

   确认 `Num tests: 678` 且每行 `OK`；对照 3.3/3.5 节，随机挑 5 行用例，手工说出它走快速路径还是四条计算路径中的哪一条。

2. **观察性能与流水占用**：

   ```bash
   python tests/test_layout.py performance
   ```

   记录 `mn=32768` 的性能行，做两组对比：`dtype=FP32` vs `dtype=short`（fp32 输入每 32 个 K 读 4 字节，short 每 64 个 K 读 2 字节——输入字节数差 4 倍，带宽列应有体现）；`major=K` vs `major=MN`（K-major 需要转置，观察 vec 列与 mte2 列的占比差异）。再找出**没有 TB/s 列**的行（快速路径命中，测的是 Host 开销），核对它们是否都是 `dtype=short、major=MN-major、gran_mn=1`。

3. **Python 实验：快速路径与四条路径**：

   ```python
   import torch, torch_npu
   import deep_gemm

   def ceil_div(x, y): return (x + y - 1) // y
   M, K, recipe = 256, 128, (1, 1, 32)

   # ① 目标布局：物理 [K对, MN] → 逻辑 [MN, K对]，stride(-2)=1
   sf = torch.randint(-32768, 32768, (ceil_div(K, 64), M), dtype=torch.int16).npu().transpose(-2, -1)
   out = deep_gemm.transform_sf_into_required_layout(sf, M, K, recipe, is_sfa=True)
   print(out.data_ptr() == sf.data_ptr())        # True：零拷贝

   # ② K-major 输入 → 触发 Kernel（路径③）
   sf_k = sf.transpose(-2, -1).contiguous()
   out2 = deep_gemm.transform_sf_into_required_layout(sf_k, M, K, recipe, is_sfa=True)
   print(torch.equal(out2, sf))                  # True：内容一致（仅转置）

   # ③ fp32 输入 → 打包 + 校验（路径①）
   e = torch.randint(1, 255, (M, ceil_div(K, 32)), dtype=torch.int32)
   sf_f = (e * (1 << 23)).view(torch.float32).npu()
   out3 = deep_gemm.transform_sf_into_required_layout(sf_f, M, K, recipe, is_sfa=True)
   print(out3.dtype, tuple(out3.shape), out3.stride())   # int16, (256, 2), (1, 256)

   # ④ 非法 fp32（1.5 带尾数）→ 设备侧断言
   bad = sf_f.clone(); bad[0, 0] = 1.5
   deep_gemm.transform_sf_into_required_layout(bad, M, K, recipe, is_sfa=True)
   # → Kernel 终止并打印 "Float32 SF must be non-negative and can't have non-zero mantissa"
   ```

4. **strided 输入与 L2 hint**：构造外层 stride 很大的 MN-major 输入（如 `big = torch.randint(..., (K//64, M + 512), dtype=torch.int16).npu()`，取 `sf = big.t()` 使 `stride(-1) = M + 512`），分别在不设置与设置 `export DG_TRANSFORM_SF_LD_CACHE_HINT=0`（`NORMAL_FIRST_VICTIM`）的情况下重跑第 3 步 ② 并用 `bench_msprof` 或性能模式对比耗时，验证 3.4 节第 4 点的 hint 启发式。

5. **验证 K-grouped 的 padding 清零**（仿 `make_layout_case`）：

   ```python
   M, K = 64, 512
   psum = torch.tensor([100, 356, 512], dtype=torch.int32, device='npu')
   sf = (torch.randint(1, 255, (M, K // 32), dtype=torch.int32) * (1 << 23)).view(torch.float32).npu()
   sf[:, 4:8] = -1.0    # 组 0 尾部：32 粒度区间 [ceil(100/32), 256/32)
   sf[:, 12:16] = -1.0  # 组 1 尾部：[ceil(356/32), 512/32)
   out = deep_gemm.transform_k_grouped_sf_into_required_layout(sf, M, K, (1, 1, 32), True, psum)
   print((out[:, 2:4] == 0).all().item(), (out[:, 6:8] == 0).all().item())  # True True
   print((out[:, :2] != 0).all().item())                                    # True
   ```

   再把输入换成 int16 并在 padding 列填非零垃圾重跑，对照 3.5 节解释观察到的差异（掩码只在 fp32 打包路径被消费）。

### B 组：无 Ascend 环境（静态任务）

1. **打包手算**：完成 3.6 节的手算例子（`gran_mn=2`、M=4、K=128），再自行取 `gran_mn=32`、M=128、K=64 重做一遍：写出用户 fp32 形状 `[4, 2]` 与输出 int16 逻辑形状 `[128, 1]`，标出哪些输出行共享同一个输入行。
2. **形状推演**：`M=4097、K=7296、gran_mn=128`。写出：fp32 输入形状 `[33, 228]`、输出形状 `[4097, 114]` 与输出字节数；short 输入形状 `[33, 114]`、输出形状与 fp32 情形相同。逐条核对 `check_sf_layout` 的两条 shape 断言（`:53-54`）与 `ceil(ceil(K/32)/2) == ceil(K/64)`。
3. **快速路径判定表**：对 `dtype ∈ {fp32, int16} × major ∈ {K, MN} × gran_mn ∈ {1, 32, 128}` 共 18 种组合，判定 `check_sf_layout` 能否返回 true；对不能的组合各写一句"卡在哪一条"。
4. **分组填充推演**：M-grouped、`psum_layout=[100, 356, 1024]`、`alignment=256`、`gran_mn=32`。用 `iter_psum_sep`（`generators.py:230-235`）算出需清零的 scale 行区间（应为 `[4, 8)` 与 `[12, 16)`），再对照 Kernel 的 `kGranGroup = kGranMN`、`kGroupAlignment = 256/32 = 8`（`transform_sf.hpp:494, :589`）验证一致；说明该区间内的 M 行范围。
5. **路径分派表**：合上教程，重写 3.5 节的四路径表（路径 × 需要的动作 × 用到的 UB 缓冲 × 是否消费 grouped_flag），并回答：为什么只有 fp32+K-major 需要 packed_ub？
6. **画转换图**：不看 3.8 节，独立画出"Scale 从输入到 Kernel 布局的转换图"（当天产出），再对照修正；重点自查快速路径分支、四条计算路径、repeat 与分组清零的位置、以及消费链的五个环节。

## 六、当天产出

**Scale 从输入到 Kernel 布局的转换图**——以 3.8 节的参考版为答案，自己合上教程画一版，至少覆盖：输入形态空间（dtype/major/维度/分组）、`check_sf_layout` 的快速路径分支、`launch_transform_sf` 的关键决策（输出构造、tiling、L2 hint）、Kernel 内四条计算路径与 repeat/清零的先后关系、目标布局三要素、以及从 `GemmDesc.sfa/sfb` 到 `asc_mmad_mx` 的消费链。写完后把画不出的环节标出来，回到第四节对应站点重读。有环境的同学附上 A 组第 2 步的性能记录（含 fast-path 行）。

## 七、自测题

1. Kernel 侧 SF 目标布局的三要素是什么？这份契约在哪里"成文"？
   > int16（ue8m0x2 pair）；逻辑 `[..., MN, ceil(K/64)]`、物理 MN-major 紧凑（`stride(-2)==1`、`stride(-1)==MN`）；按 `gran_mn` 广播、分组 padding 为 0。成文于 `GemmDesc` 注释（`gemm_config.hpp:36-38, :62-65`）与 `fp8_gemm.hpp:65-66` 的 `gm_sfa/gm_sfb` 构造；`check_sf_layout:57-62` 是其镜像。
2. 一个 int16 SF pair 的两个字节各覆盖哪些 K？`gran_k` 是多少？
   > 低字节覆盖 `[64p, 64p+32)`、高字节覆盖 `[64p+32, 64p+64)`（小端，对照 `reference` 的 `view(torch.int16)`）；`gran_k` 恒为 32（`apis/layout.hpp:55`），`MX_SF_DIVISOR = 64`（`common.hpp:65`）。
3. 什么输入可以零拷贝通过？为什么 `gran_mn > 1` 永远不行？
   > short + `gran_mn == 1` + `stride(-2)==1` 且 `stride(-1)==size(-2)`（Batched 还需稠密 batch stride，`:57-62`）。`gran_mn > 1` 时输出必须是"每 MN 行一个 int16"，而行号到 scale 行的映射 `m / gran_mn` 不是线性 stride，广播只能实际复制（`repeat_simd`）。
4. fp32 SF 的合法性约束是什么？在哪里检查？`disable_ue8m0_cast` 起什么作用？
   > 必须非负且尾数为 0（即 `2^e`）。设备侧：打包时把原始位 OR 进校验累加器，Kernel 末尾 `check_validation` 断言 `0x80000000` 与 `0x007fffff` 均为 0（`transform_sf.hpp:13-21, :635-638`）；分组掩码在 OR 之前 AND，padding 里的 -1.0 不会误报（`:269-275`）。Host 侧：fp32 输入 + `disable_ue8m0_cast=true` 直接断言失败（`apis/layout.hpp:86-87`），用于禁止隐式 cast。
5. 四条计算路径分别做什么？哪条需要中间 packed 缓冲？
   > fp32+K：抽指数两行打包 → 转置（需 packed_ub）；fp32+MN：抽指数配对直写 dst；short+K：仅转置；short+MN：仅拷贝（`transform_sf.hpp:595-612`，`kHasPackedBuffer = kFloat && K-major`，`:499`）。
6. M-grouped 下 sfb 为什么走 Batched 类型的转换？K-grouped 呢？
   > M-grouped 的 B 是 `[G, N, K]` 堆叠，SF 天然带组维，`gemm.hpp:145-152` 给 sfb 传 `num_groups=g` → `GemmType::Batched`；A 是连续 psum 布局，走 `MGroupedContiguousWithPsumLayout` + `psum_layout` 清组尾。K-grouped 的 A/B 都连续、组在 K 维，两者都走 `transform_k_grouped_sf_into_required_layout`（`:130, :143`）。
7. 分组 padding 的两种处理机制是什么？各用于哪些组合？
   > 行粒度清零（`clear_group_tail_simd` 直接清 src_ub 行）用于 padding 落在行维度的组合（M-grouped+K-major、K-grouped+MN-major）；列方向掩码（`grouped_flag` 全 1/0，打包时 AND）用于 padding 落在连续维度的组合（M-grouped+MN、K-grouped+K），但只在 fp32 打包路径被消费——short 的纯搬运路径不消费掩码，测试因此把 int16 的 padding 直接填 0（`test_layout.py:91`）。
8. 转换 Kernel 用哪三条 pipe？级数由什么决定？为什么 fp32 路径开头要清零整个 UB？
   > MTE2（GM→UB）、V（打包/转置/广播/清零）、MTE3（UB→GM）；`kNumStages = min(8, kValidUBBytes / kBytesPerStage)` 且 ≥2（`:509-512`），每级含 src（+packed）+dst，并预留 8KB 谓词溢出区与 64×u32 校验区。清零整个 UB 是因为打包按整向量宽度读取并 OR 进校验累加器，actual 之外未被搬运覆盖的位置必须恒为 0 才不会误报（`:525-528`，short 路径无校验故不清零）。

---

明天（第 34 天）把这些环节串成完整调用链：选择 `fp8_fp4_gemm_nn`，从"输入与 Scale"一路追到"结果验证"，并回答 Scaling Factor 在哪里转换、在哪里参与计算。
