# 第 5 周 · 第 4 天（总第 32 天）：学习 FP8/FP4 Kernel 路径

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 5 周"测试体系与低精度 GEMM"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前三天我们搭好了测试体系并跑通了 BF16 链路。今天进入低精度世界：**FP8×FP8、FP4×FP4、FP8×FP4 三条执行路径分别怎么走？反量化（Scale 参与计算）和高精度累加分别发生在哪里？** 答案的主线只有一句话：**输入是"量化码 + 块缩放因子（SF）"，Host 把 SF 规整成 UE8M0 打包布局，Cube 阵列用 `asc_mmad_mx` 在 MAD 内部应用缩放并在 L0C 里做 FP32 累加**；FP8×FP4 额外多出一条"AIV 向量核查表解码 FP4→FP8"的前置流水。当天产出是**BF16 与低精度 GEMM 的主要差异表**（参考版本见 3.7 节）。Scale 布局转换的细节留给明天（第 33 天），今天只需要知道它在 `get_gemm_desc` 里被调用、产出什么布局。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出低精度 GEMM 的数学语义：`D = (A_code ⊙ sfA) @ (B_code ⊙ sfB)^T`，其中 `sfA/sfB` 按 `recipe = (gran_m, gran_n, gran_k)`（测试默认 `(1, 1, 32)`，`tests/generators.py:556`）分块共享；说出 UE8M0 的含义（8 位纯指数、值为 2^e），以及为什么 fp32 形式的 SF 只允许 8 位指数、23 位尾数必须为 0（`csrc/apis/gemm.hpp:209-215` 的文档约束，`deep_gemm/include/deep_gemm/transform_sf.hpp:19-20` 的设备侧断言）；
2. 背出 dtype 派发表：`dispatch_fp8_fp4_gemm`（`csrc/apis/gemm.hpp:22-33`）按 `(a_dtype, b_dtype)` 三分——fp8×fp8 → `launch_fp8_gemm`，fp4×fp4 → `launch_fp4_gemm`，fp8×fp4 → `launch_fp8_dequant_gemm`，其余组合（含 fp4×fp8）直接断言失败；
3. 说出 Kernel 侧 SF 的三个关键常量及来源：`MX_SF_DIVISOR = 64`（一个 int16 "SF pair" 覆盖的 K 元素数，`deep_gemm/include/deep_gemm/common.hpp:65`）、`kSFPairsPerKBlock = BLOCK_K / 64`（`fp8_gemm.hpp:68`）、`kFracK`（fp8 为 32、fp4 为 64，`common.hpp:73-79`）；并能解释 SF 在 GM 中**总是 MN-major**（`fp8_gemm.hpp:13` 的注释与 `:65-66` 的 `gm_sfa/gm_sfb` 构造）；
4. 追踪 SF 的完整搬运与消费链路：GM → `copy_gm_to_l1_mx`（`ascend/copy.hpp:161-197`）→ L1 中**独立的 SF 多级缓冲**（`l1_sfa/l1_sfb`，事件区间 `kSFEventBase = kNumL1Stages`，`fp8_gemm.hpp:82-83, :108-113`）→ `copy_l1_to_l0a_mx / copy_l1_to_l0b_mx`（`copy.hpp:199-229`）把 UE8M0 与数据一同送进 L0A/L0B → `mad()` 在 8 位操作数时改用 `asc_mmad_mx`（`copy.hpp:370-381`）——**反量化发生在 Cube 的 MAD 内部，FP32 累加发生在 L0C**（`l0c_ptr<float> l0c`，`fp8_gemm.hpp:87`）；
5. 解释低精度 K 尾处理与 BF16 的差异：MAD 的 K 循环上界是 `k64 = aligned(actual_k, 64)` 而非 `actual_k`（`fp8_gemm.hpp:160, :205`），搬运引擎把数据补零到 32 字节（= fp8 的 32 元素 / fp4 的 64 元素），剩下的 `[kc0, k64)` 缺口由 `fill_tail_block` 显式补零（`fp8_gemm.hpp:167-170`，`copy.hpp:333-356`）——fp4 因 `kFracK = 64` 恰好无缺口，fp8 必须补最多 32 个元素；
6. 讲清 FP8×FP4 为什么要单独一个 Kernel：B 是打包 FP4，先由 **AIV0**（`kDequantAIVId = 0`，`fp8_dequant_gemm.hpp:10`）执行 `vf_build_lut`（E2M1→E4M3 查找表，`:12-40`）+ `vf_deq_nd2nz`（`:42-93`）+ `asc_copy_ub2l1`（`:408-413`）解码成 FP8 NZ 格式写入 L1B，AIC 用 `wait_intra_block / set_intra_block`（`:283-284, :339-341`）与它同步；**Cube 路径与 FP8×FP8 完全相同**（`:175-182` 的 L1/L0 布局、`:320-322` 的 `mad`）；此时只剩 1 个 AIV 做 Epilogue drain（`Epilogue<..., 1>`，`:169-171`；启发式 `heuristics.hpp:186`），或走 `dual_aiv_dequant + direct_store` 让两个 AIV 都解码、AIC 直写 D（`:129-131` 的 static_assert、`heuristics.hpp:727-736`）；
7. 对照测试说明"低精度为什么能用 1e-5 阈值"：`generate_random_quantized_with_major`（`generators.py:124-166`）直接随机生成量化码和 SF，再用**精确解码值**（`_scale_quantized_reference`，`:105-121`）构造参考结果——比的是"解码 + fp32 乘加"的忠实度，不是量化误差；同时说出 ACLNN 交叉验证在低精度下的限制（K 必须 64 倍数、MX 路径只支持 1×32 逐行缩放、fp8×fp4 只支持双 K-major 且 bf16/fp16 输出，`tests/common.py:98-122`）；
8. 独立写出**BF16 与低精度 GEMM 的主要差异表**（当天产出，参考版本见 3.7 节），覆盖 API、Host、派发、片上缓冲、MAD、尾部处理、AIV 分工七个层面。

## 二、预备知识

- **第 4 天**：BF16/FP32 与 FP8/FP4 的表示范围、Scaling Factor 与量化粒度——今天的 `recipe`、UE8M0 都是它的落地。
- **第 11 天**：`GemmDesc` 的字段来源——今天关注新增的 `sfa_ptr/sfb_ptr`（`csrc/jit_kernels/gemm_config.hpp:44-45`）与 `dtype_to_kernel_type` 对 fp4 的映射（`:79-85`）。
- **第 12 天**：`select_gemm_config`、模板代码生成、`jit->compile()/launch()`——今天读 `launch_fp8_gemm` 的 27 个模板实参时要直接复用。
- **第 15～16 天**：Kernel 模板参数分类与片上内存 offset 分配——`fp8_gemm_impl` 的模板只比 BF16 多出 `kSFKBlocks/kNumL1SFStages` 和 dtype 相关项。
- **第 19、22 天**：`copy_gm_to_l1 / copy_l1_to_l0a/b`、`actual_k`、`eff_mad_k`——今天的 `_mx` 变体和 `k64` 是它们的低精度延伸。
- **第 24～25 天**：L0C 累加（`ENABLE_UPDATE/KEEP`、`is_first/is_last`）与 AIC store / AIV drain——低精度路径原样复用，唯一变化是 dequant Kernel 的 AIV 分工。
- **第 29～30 天**：`GemmTestDesc.generate()` 与 `run_gemm_test` 的两阶段编排、五类断言——今天直接消费它们。

另注：今天的"运行"部分需要 Ascend 环境；无环境时完成第五节 B 组的静态任务（LUT 手算、MX 反量化手算、SF 布局推演、模板实参对照、AIC/AIV 时序图）即可达到当天目标。Scale 布局转换（`transform_sf_into_required_layout` 的内部实现）是第 33 天的主题，今天只看它的调用点与产出。

## 三、详细讲解

### 3.1 低精度 GEMM 在算什么：量化码 + 块缩放

BF16 GEMM 的输入就是数值本身；低精度 GEMM 的输入是**两部分**：量化码张量（fp8_e4m3 或打包的 fp4_e2m1x2）和缩放因子张量（SF）。数学上：

```text
D[m, n] = Σ_k ( A_code[m, k] × sfA[m, k] ) × ( B_code[n, k] × sfB[n, k] )
```

`sfA/sfB` 不是逐元素的：它们按 `recipe` 分块共享。`recipe = (gran_m, gran_n, gran_k)` 表示 A 沿 M 每 `gran_m` 行、沿 K 每 `gran_k` 列共享一个 scale（B 同理用 `gran_n/gran_k`）。测试的默认 recipe 是 `(1, 1, 32)`（`generators.py:556`）——即沿 K 每 32 个元素一个 scale、M/N 方向逐行/列，这正是 MX（microscaling）标准粒度；`test_recipes` 场景额外覆盖 `(1,32,32)、(32,1,32)、(32,32,32)`（`generators.py:441-462`）。

**UE8M0** 是 scale 的存储格式：8 位无符号整数 e，值为 2^e，没有尾数。用户可以传 fp32 的 scale，但只允许"8 位指数 + 尾数全 0"（`gemm.hpp:209-215` 文档、`transform_sf.hpp:19-20` 的 `check_validation` 断言），Host 会把它转换（cast）成 UE8M0 打包布局。为什么强制 2 的幂？因为硬件在 MAD 里对 scale 的乘法被设计成**指数加法**，这比通用浮点乘便宜得多，也是 `asc_mmad_mx` 存在的前提。

测试侧的参考结果因此不是"原始浮点矩阵 × 量化误差"，而是**精确解码**：`generate_random_quantized_with_major`（`generators.py:124-166`）先随机生成量化码（fp8 覆盖全部非 NaN 的 E4M3 编码，`:149-152`；fp4 用随机字节拆成两个 nibble，`:155-164`）和取自小集合的 SF（fp8 ∈ {2^-9, 2^-8, 2^-7}，fp4 ∈ {0.25, 0.5, 1.0}，`:137-144`），再用 `_scale_quantized_reference`（`:105-121`）把"码值 × SF"按块展开成 fp32 参考矩阵。这就是 `tests/test_fp8_gemm.py:21`、`test_fp4_gemm.py:20` 敢用 `threshold=1e-5` 的原因：Kernel 只需要忠实复现"解码 + fp32 乘加"，量化本身不引入比较误差。

### 3.2 三条执行路径：一个派发函数看全貌

Python 侧入口是 `fp8_fp4_gemm_{nt,nn,tn,tt}`（`deep_gemm/__init__.py:47-48`；`fp8_gemm_*` 是同名别名，`csrc/apis/gemm.hpp:624-627`）。以 `nt` 为例（`gemm.hpp:225-256`）：

```text
deep_gemm.fp8_fp4_gemm_nt((A, sfA), (B, sfB), D, recipe_a=..., recipe_b=...)
  └→ fp8_fp4_gemm<false,false>                       (gemm.hpp:226)
     ├→ get_gemm_desc(Normal, a, b, d, c, …, sfa_in, sfb_in, recipe, …)   (:242-248)
     │    ├→ layout::get_recipe(recipe, recipe_a, recipe_b)               (:127)
     │    ├→ sfa = transform_sf_into_required_layout(sfA, m, k, recipe, …) (:128-140)
     │    └→ sfb = transform_sf_into_required_layout(sfB, n, k, recipe, …) (:141-154)
     └→ dispatch_fp8_fp4_gemm(desc)                    (:250)
          ├─ (fp8,  fp8 ) → launch_fp8_gemm(desc)          (gemm.hpp:24-25)
          ├─ (fp4,  fp4 ) → launch_fp4_gemm(desc)          (gemm.hpp:26-27)
          ├─ (fp8,  fp4 ) → launch_fp8_dequant_gemm(desc)  (gemm.hpp:28-29)
          └─ 其他          → DJ_HOST_ASSERT(false)          (gemm.hpp:30-31)
```

三个要点：

1. **SF 在进入 Kernel 前必被规整**。不管用户给的是 fp32 还是 int16、K-major 还是 MN-major 的 SF，`transform_sf_into_required_layout`（实现在 `csrc/utils/layout.hpp`，设备侧算子在 `deep_gemm/include/deep_gemm/transform_sf.hpp`）都把它变成 Kernel 约定的布局：**UE8M0 打包的 int16、MN-major**。转换后的张量由 `GemmDesc.sfa/sfb` 持有防止提前释放（`gemm_config.hpp:78`，`.sfa_ptr/.sfb_ptr` 在 `:171-172` 填入）。细节明天展开。
2. **FP4×FP4 与 FP8×FP8 共用同一个 Kernel 模板**。`launch_fp4_gemm`（`csrc/jit_kernels/fp8_gemm.hpp:83-151`）与 `launch_fp8_gemm`（`:11-79`）生成的代码都 `#include <deep_gemm/fp8_gemm.hpp>` 并实例化 `fp8_gemm_impl`，唯一的实质区别是 `ab_dtype_t` 模板实参：`dtype_to_kernel_type` 把 `kPackedFP4` 映射成 `float4_e2m1x2_t`（`gemm_config.hpp:79-85`）——一个字节打包两个 E2M1 值。JIT 缓存键分别为 `"fp8_gemm"` 和 `"fp4_gemm"`（`:71, :143`）。
3. **FP8×FP4 是独立的第三个 Kernel**。注释（`csrc/jit_kernels/fp8_dequant_gemm.hpp:12-18`）说得很清楚：A 是硬件 fp8，B 是打包 fp4，需要在 AIV 向量核上反量化成 fp8；"cube datapath is identical to fp8_gemm"。注意方向性：只有 fp8×fp4，fp4×fp8 不支持（`gemm.hpp:202-203, :30-31`）——因为反量化设计在 B 侧。

### 3.3 SF 的存储与搬运：int16 打包、独立流水

Kernel 里 SF 的一等公民地位体现在常量、布局和流水三个层面。

**粒度常量**（`common.hpp:63-65`）：

```cpp
inline constexpr uint32_t FRAC_MN = 16;       // NZ 分形的 M/N 行数
inline constexpr uint32_t MX_ADDR_DIV = 16;   // MX 目标地址编码除数
inline constexpr uint32_t MX_SF_DIVISOR = 64; // 每个 SF pair 覆盖的 K 元素数
```

一个 int16 是一个 "SF pair"：两个 UE8M0 字节各覆盖 32 个连续 K 元素，合计 64——这正好对应 recipe 的 `gran_k = 32`。由此推出 Kernel 侧 SF 矩阵形状（`fp8_gemm.hpp:60-66`）：

```cpp
const auto shape_sf_k = ceil_div(shape_k, MX_SF_DIVISOR);          // int16 列数
const gm_ptr<int16_t, Major::MN> gm_sfa(shape_m, gmem_sfa, shape_m * shape_sf_k);
const gm_ptr<int16_t, Major::MN> gm_sfb(shape_n, gmem_sfb, shape_n * shape_sf_k);
```

即 SFA 逻辑上是 `[M, ceil(K/64)]` 的 int16 矩阵，物理上 MN 维连续（MN-major）——与 A/B 可以任意 major 不同，**SF 的 GM 布局是固定的**，这依赖 Host 侧的转换保证。

**L1 布局与独立事件**（`fp8_gemm.hpp:80-83, :108-113`）：

```cpp
constexpr l1_ptr<ab_dtype_t, kMajorA> l1a(BLOCK_M, BLOCK_K, 0);
constexpr l1_ptr<ab_dtype_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1Stages));
constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kSFPairsPerChunk, l1b.offset(kNumL1Stages));
constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kSFPairsPerChunk, l1_sfa.offset(kNumL1SFStages));
```

SF 缓冲排在 A/B 之后，**级数是 `kNumL1SFStages`（启发式固定为 2，`heuristics.hpp:160`）而不是 `kNumL1Stages`**；事件也分家：A/B 用 `[0, kNumL1Stages)` 区间，SF 用 `kSFEventBase = kNumL1Stages` 起始的独立区间。原因在粒度：SF 每 64 个 K 才一个 int16，数据量小得多，没必要与 A/B 同频换级。

**按 chunk 搬运**（`fp8_gemm.hpp:173-187, :253-259`）：SF 不是每个 k-block 都搬，而是每 `kSFKBlocks` 个 k-block 搬一次"chunk"：

```cpp
if (k_within_chunk == 0) {
    uint32_t chunk_k_blocks = min(num_k_blocks - k_block_idx, kSFKBlocks);
    uint32_t chunk_k_end = min(k_idx + chunk_k_blocks * BLOCK_K, eff_shape_k);
    uint32_t chunk_sf_pairs = ceil_div(chunk_k_end - k_idx, MX_SF_DIVISOR);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, kSFEventBase + l1_sf_event_idx);
    if (!reuse_sfa) copy_gm_to_l1_mx(l1_sfa[l1_sfa_stage_idx],
                                     sfa.index(0, k_block_idx * kSFPairsPerKBlock),
                                     actual_m, chunk_sf_pairs, kL2CtrlA);
    ...  // sfb 同理；asc_sync_notify
}
```

其中 `kSFPairsPerChunk = kSFPairsPerKBlock * kSFKBlocks`（`:69`）。`kSFKBlocks` 由启发式选择：初值 `min(4, ceil_div(k, block_k))`，再按 L1 容量和（dequant 情况下）`ceil_div(num_k_blocks, num_l1_sf_stages)` 收紧（`heuristics.hpp:156-164, :176-183`）。代价是 SF 分走了 L1：`num_l1_stages` 的上限从 BF16 的 8 降到 4（`kNumMaxL1StagesWithSF = 4`，`heuristics.hpp:167-168`）。

`copy_gm_to_l1_mx`（`copy.hpp:161-197`）本体是一个 `set_mte2_nz` + `asc_copy_gm2l1_dn2nz/nd2nz` 的组合，把 GM 的 int16 行变换成 L1 的 NZ 分形布局——和数据的 nd2nz 是同一套搬运引擎，只是元素类型换成 int16。

### 3.4 反量化与高精度累加的位置：`asc_mmad_mx`

这是今天最核心的一条链。MAD 内层循环（`fp8_gemm.hpp:205-232`）每次迭代做四件事：

```cpp
const auto sf_y = static_cast<uint16_t>(k_within_chunk * kSFPairsPerKBlock
                                        + k_mad_idx / MX_SF_DIVISOR);   // 本次 MAD_K 对应的 SF 列
auto l0a_tile = l0a[s_l0_ab].as_mad_aligned(mad_m, eff_mad_k);
auto l0b_tile = l0b[s_l0_ab].as_mad_aligned(mad_n, eff_mad_k);

asc_sync_wait(PIPE_M, PIPE_MTE1, s_l0_ab);
copy_l1_to_l0a(l0a_tile, l1a[l1a_stage_idx], m_mad_idx, k_mad_idx);     // 数据
...
copy_l1_to_l0a_mx(l0a_tile, l1_sfa[l1_sfa_stage_idx], m_mad_idx, sf_y, eff_mad_k);  // ← SF 一同进 L0A
copy_l1_to_l0b_mx(l0b_tile, l1_sfb[l1_sfb_stage_idx], n_mad_idx, sf_y, eff_mad_k);  // ← SF 一同进 L0B
asc_sync_notify(PIPE_MTE1, PIPE_M, s_l0_ab);

asc_sync_wait(PIPE_MTE1, PIPE_M, s_l0_ab);
mad(l0c_tile, l0a_tile, l0b_tile, mad_m, mad_n, eff_mad_k,
    is_last ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
    is_first);
asc_sync_notify(PIPE_M, PIPE_MTE1, s_l0_ab);
```

`copy_l1_to_l0a_mx`（`copy.hpp:199-213`）把 L1 里的 UE8M0 按 `sf_y` 列对齐搬进 L0A（目的地址用 `l0a.addr / MX_ADDR_DIV` 编码），`sf_pairs = eff_k / MX_SF_DIVISOR`。之后 `mad()` 的分派（`copy.hpp:370-381`）：

```cpp
if constexpr (sizeof(TA) == 1 || sizeof(TB) == 1) {
    asc_mmad_mx(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n,
                unit_flag, true, false, clear_C);      // ← fp8 / 打包 fp4 走 MX 变体
} else {
    asc_mmad(...);                                      // ← bf16 走普通 MMAD
}
```

`sizeof(TA) == 1` 覆盖 fp8（1 字节）和 `float4_e2m1x2_t`（1 字节打包两个 fp4）。`asc_mmad_mx` 是硬件的 MX 矩阵乘指令：**它在 Cube 内部把 L0A/L0B 里随数据一起进来的 UE8M0 作用到对应 64 元素组上，等价于先反量化再相乘**；乘加结果以 FP32 累加进 L0C（`l0c_ptr<float> l0c(MAD_M, MAD_N)`，`fp8_gemm.hpp:87`）。所以：

- **反量化的位置**：L1→L0 的 `_mx` 搬运把 SF 送进 L0A/L0B，真正"乘 scale"发生在 `asc_mmad_mx` 的 Cube 数据通路上——不在 GM 侧、不在 AIV 侧（fp8×fp8 / fp4×fp4 路径）；
- **高精度累加的位置**：L0C 的 fp32 累加器，`is_first` 清零、`ENABLE_KEEP` 中间保持、`is_last` 才 `ENABLE_UPDATE`——与 BF16 第 24 天讲的机制完全一致，只是数据通路换成了 `_mx`。

这也解释了 `asc_set_mmad_direction_n()`（`fp8_gemm.hpp:106`）与 SF 复用判断 `can_reuse_sf = can_reuse && num_k_blocks <= kSFKBlocks`（`:142`）的存在：前者设定 NZ 方向，后者让常驻 B 的 SF 一并驻留（`GemmConfig.resident_sf`，`gemm_config.hpp:111`）。

### 3.5 K 尾处理：`k64`、`kc0` 与 `fill_tail_block`

低精度的 K 尾比 BF16 多一层约束。看 `fp8_gemm.hpp:157-171`：

```cpp
const uint32_t actual_k = k_block_idx < eff_shape_k / BLOCK_K ? BLOCK_K : eff_shape_k % BLOCK_K;
const uint32_t k64 = aligned(actual_k, MX_SF_DIVISOR);   // ① 对齐到 64（SF pair 粒度）
const uint32_t kc0 = aligned(actual_k, kFracK);          // ② 对齐到 32(fp8)/64(fp4)（分形粒度）

asc_sync_wait(PIPE_MTE1, PIPE_MTE2, kL1EventBase + l1_event_idx);
if (!reuse_a) copy_gm_to_l1(l1a[l1a_stage_idx], a.index(0, k_idx), actual_m, actual_k, kL2CtrlA);
if (!reuse_b) copy_gm_to_l1(l1b[l1b_stage_idx], b.index(0, k_idx), actual_n, actual_k, kL2CtrlB);
const bool fill_a = (kMajorA == Major::K ? kc0 : actual_k) < k64;
const bool fill_b = (kMajorB == Major::K ? kc0 : actual_k) < k64;
if (fill_a && !reuse_a) fill_tail_block(l1a[l1a_stage_idx], actual_k);   // ③ 显式补零
if (fill_b && !reuse_b) fill_tail_block(l1b[l1b_stage_idx], actual_k);
```

三层逻辑：

1. **MAD 必须算到 `k64`**：`asc_mmad_mx` 的 SF 以 64 元素组为最小单位，内层循环上界是 `k_mad_idx < k64`（`:205`）、`eff_mad_k = min(MAD_K, k64 - k_mad_idx)`（`:208`）。`k64` 之外（到 BLOCK_K）的旧数据不会被消费，无需处理。
2. **搬运引擎只补到 `kc0`**：ND2NZ 搬运的零填充粒度是 32 字节——fp8 正好 32 个元素（`kFracK = 32`），fp4 是 64 个元素（`kFracK = 64`，`common.hpp:73-79`）。
3. **`fill_tail_block` 补 `[kc0, k64)` 的缺口**（`copy.hpp:333-356`）：K-major 时补第 `nz32/32` 个 32 字节块行（`k64 - nz32` 恰为 0 或 32）；MN-major 时补行区间 `[actual_k, k64)`。由于 `static_assert(sizeof(T) == 1)`（`:335`），这个函数**只属于 fp8/fp4**——BF16 完全依赖隐式补零（第 3 天 3.2 节的事实 1）。fp4 因为 `kFracK == MX_SF_DIVISOR == 64`，`kc0 == k64`，K-major 时永不触发；fp8 则最多需要补 32 个元素。

对照一个具体例子（实践步骤 B6 会让你完整推演）：K=70、fp8、K-major：`k64 = 128`、`kc0 = 96`，ND2NZ 补 `[70, 96)`，`fill_tail_block` 补 `[96, 128)`，MAD 消费 `[0, 128)`。

另外注意 MN-major 的对齐差异：`mad_m = kMajorA == MN ? aligned(eff_mad_m, kFracK) : eff_mad_m`（`:198-199`）——transpose 搬运要求 M/N 对齐到 `kFracK`，这正是 `static_assert(kMajorA == K || MAD_M % 32 == 0)`（`:55-58`）以及 fp4 MN-major 要求 `MAD_M/N >= 64` 的来源。

### 3.6 FP8×FP4：AIV 上的 LUT 反量化流水

`fp8_dequant_gemm_impl`（`deep_gemm/include/deep_gemm/fp8_dequant_gemm.hpp:113`）的 AIC 分支与 `fp8_gemm_impl` 几乎逐行相同（L1/L0 布局 `:175-187`、SF chunk `:269-279`、MAD 循环 `:287-337`），差别集中在两处：B 不再由 MTE2 从 GM 搬入 L1，而是等 AIV 解码；多了一组 intra-block 同步。新东西全在 AIV 侧。

**解码算法**分两步。第一步 `vf_build_lut`（`:12-40`）在 UB 里构造 16 项查找表：E2M1 的 8 个幅值码 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 映射到 E4M3 字节——码 0 → `0x00`、码 1 → `0x30`、码 2..7 → `0x30 + (code << 2)`（即 `0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C`），符号位（bit 3）左移 4 位变成 E4M3 的符号位（bit 7）。这张表在 Kernel 开头构造一次（`:364`）。第二步 `vf_deq_nd2nz`（`:42-93`）是流式解码：

```text
每个 256B 的 UB 读入 = 512 个 FP4 值
  ├─ asc_shiftright_scalar(high, bytes, 4)      高 nibble → odd 列
  ├─ asc_gather(low_fp8,  table, bytes)         低 nibble 查表 → E4M3 字节
  ├─ asc_gather(high_fp8, table, high)          高 nibble 查表
  ├─ asc_intlv(out0, out1, low_fp8, high_fp8)   交织恢复 [even0, odd0, even1, odd1, …]
  └─ asc_storealign_postupdate(…)               按 NZ 列组间距写入 ub_nzpad
```

也就是**查表 + 交织**完成 `FP4 nibble → FP8 字节` 的同时完成了 nd2nz 重排。`static_assert(col_bytes == 128 || col_bytes == 256)`（`:44`）限制了 B tile 的行字节数，对应启发式里 `row_elements != 256 && row_elements != 512` 即淘汰该候选（`heuristics.hpp:150-154`）。

**AIV0 的流水**（`:362-422`）：

```text
对每个输出块、每个 k-block：
  wait_intra_block(PIPE_MTE3, kDequantAIVId, l1_event_idx)   ← 等 AIC 释放 L1B 级（:390 或 :406-407）
  对每段行（num_b_rows_per_stage = num_b_rows/2 行）：
    copy_gm_to_ub(ub_fp4[s], b.index(...), actual_rows, b_valid_cols, …)   GM→UB（打包 fp4）
    vf_deq_nd2nz<…>(ub_fp4[s], ub_lut, ub_nzpad[s])                        UB 内解码
    asc_copy_ub2l1(l1b[l1b_stage] + row_idx*C0, ub_nzpad[s],
                   num_b_cols_grp, actual_rows, src_pad, dst_pad)           UB→L1（FP8 NZ）
  set_intra_block(PIPE_MTE3, dequant_aiv_idx, l1_event_idx)   ← 通知 AIC 本级解码完成
```

UB 缓冲规划（`:150-167`）：`ub_lut`（256B 表）+ `ub_fp4`（打包 fp4 源，多级）+ `ub_nzpad`（解码后的 NZ 目的，多级），级数由 `(UBSizeBytes - asc_get_vf_len()) / 每级字节数` 推出、上限 8。

**AIC↔AIV 同步**（AIC 侧 `:210-211, :283-284, :339-341`）：开跑前 `set_intra_blocks<PIPE_MTE1, kNumL1Stages>(kDequantAIVId)` 初始化事件；每个 k-block 消费 L1B 前先 `wait_intra_block` 等 AIV0 的"本级解码完成"，用完后再 `set_intra_block` 通知"本级空闲"。这构成一条以 `l1_event_idx` 为序号的跨核生产者-消费者协议，深度为 `kNumL1Stages`。

**AIV 分工的三种形态**：

| 形态 | 条件 | AIV0 | AIV1 | 输出路径 |
|---|---|---|---|---|
| 单 AIV 解码 | 默认 | 解码 B | Epilogue drain | `aic_store` → `aiv_drain`（`Epilogue<…, 1>`，`:169-171`） |
| 双 AIV 解码 | `dual_aiv_dequant`（启发式 `heuristics.hpp:727-736` 开启） | 解码 B 的前半行 | 解码 B 的后半行 | 必须 `direct_store`（`:129-131` static_assert），AIC 直接 `copy_l0c_to_gm` |
| （对照）fp8/fp4 | — | drain | drain | `Epilogue<…, 2>`，两个 AIV 轮流 drain |

启发式为解码单独建了代价模型：`fp4_decode_bytes_per_cycle = 64`（`heuristics.hpp:63`）、解码量摊到 `num_aivs` 个 AIV（`:506-514`），并在 `dtype.dequant_b` 时把 epilogue 可用 AIV 数当作 1（`:186`）。`dual_aiv_dequant` 的开启条件（`:727-736`）包括 direct_store、Normal 类型、以及"两倍 block_m 的解码带宽仍低于 MAD 吞吐"的判断。

### 3.7 BF16 与低精度 GEMM 的主要差异表（当天产出·参考版）

| 维度 | BF16（第 3～4 周） | FP8×FP8 / FP4×FP4 | FP8×FP4（dequant） |
|---|---|---|---|
| 输入 | 单张量 `A, B` | `(A, sfA)`、`(B, sfB)` 二元组 + `recipe`（`gemm.hpp:199-224`） | 同左，但 A 必须 fp8、B 必须 fp4 |
| API | `bf16_gemm_nt/nn/tn/tt` | `fp8_fp4_gemm_*`（`fp8_gemm_*` 为别名） | 同一 API，dtype 决定路径 |
| Host 派发 | `launch_bf16_gemm` | `dispatch_fp8_fp4_gemm` → `launch_fp8_gemm` / `launch_fp4_gemm`（`gemm.hpp:22-33`） | → `launch_fp8_dequant_gemm` |
| Host 新增工作 | 无 | SF 规整：`transform_sf_into_required_layout` → UE8M0 打包 int16、MN-major（`gemm.hpp:127-154`） | 同左 |
| Kernel | `bf16_gemm_impl` | 同一个 `fp8_gemm_impl`，`ab_dtype_t` = `float8_e4m3_t` / `float4_e2m1x2_t` | `fp8_dequant_gemm_impl`（Cube 路径相同） |
| L1 缓冲 | A、B 两种 | A、B + **SFA、SFB**（独立 `kNumL1SFStages=2` 级、独立事件区间，`fp8_gemm.hpp:80-83`） | 同左（L1B 由 AIV 写入） |
| L1 级数上限 | 8（`num_pipe_events`） | 4（`kNumMaxL1StagesWithSF`，`heuristics.hpp:167-168`） | 同左 |
| L0 装载 | `copy_l1_to_l0a/b`（仅数据） | 额外 `copy_l1_to_l0a/b_mx` 把 UE8M0 随数据送入 L0A/L0B（`copy.hpp:199-229`） | 同左（B 已是解码后的 fp8） |
| MAD 指令 | `asc_mmad` | `asc_mmad_mx`（硬件应用 UE8M0，`copy.hpp:374-376`） | 同左 |
| 累加 | L0C fp32 | L0C fp32（不变） | L0C fp32（不变） |
| K 对齐/尾部 | 16 元素粒度、隐式补零 | `k64 = aligned(k, 64)` 上界 + `fill_tail_block` 显式补 `[kc0, k64)`（3.5 节） | 同左；AIC 同样补 L1A/L1B 尾（`:263-266`） |
| BLOCK_K 约束 | 16 的倍数 | 64 的倍数（`k_alignment = sf_elements_per_pair`，`heuristics.hpp:108`） | 还需 ≤ 512 且 B 行元素 256/512（`:150-154`） |
| BLOCK 上限 | 512 | fp4/dequant 时 256（`heuristics.hpp:604`） | 同左 |
| AIV 分工 | 2×drain | 2×drain | 1×解码 + 1×drain，或 2×解码 + AIC direct store |
| Kernel 专属配置 | — | `sf_k_blocks`、`num_l1_sf_stages`、`resident_sf` | 另有 `dual_aiv_dequant`（`gemm_config.hpp:107-112`） |
| 测试阈值含义 | 舍入误差 | 解码忠实度（参考值即精确解码，3.1 节） | 同左 |

## 四、源码导读

今天的主线文件与阅读站点（建议按表中顺序读）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `tests/test_fp8_gemm.py` | `test_fp8_gemm`（:8-22）、CLI（:25-54） | 与 `test_bf16_gemm.py` 逐行对照：只差 dtype 与 `run_gemm_test` 参数 |
| 2 | `tests/test_fp4_gemm.py` | `test_fp4_gemm`（:8-21） | 同上，dtype 换成 `float4_e2m1fn_x2` |
| 3 | `tests/test_fp8_dequant_gemm.py` | `test_fp8fp4_dequant_gemm`（:11-24） | fp8×fp4 组合的入口，学习计划未列但它是第三条路径的测试 |
| 4 | `tests/generators.py` | `generate_random_quantized_with_major`（:124-166）、`_scale_quantized_reference`（:105-121）、`_enumerate_recipe`（:441-462） | 量化码/SF 的生成与精确解码参考 |
| 5 | `tests/common.py` | `has_sf`（:38-39）、`gemm_dg`（:49-66）、`gemm_aclnn`（:87-128） | `gemm_name` 如何切到 `fp8_fp4_gemm_*`；ACLNN 的低精度限制 |
| 6 | `csrc/apis/gemm.hpp` | `dispatch_fp8_fp4_gemm`（:22-33）、`get_gemm_desc` 的 SF 段（:125-154）、`fp8_fp4_gemm`（:225-256） | 派发表、SF 变换调用点、API 文档 |
| 7 | `csrc/jit_kernels/gemm_config.hpp` | `GemmDesc`（:39-95）、`GemmConfig`（:98-121） | `sfa_ptr/sfb_ptr`、`dtype_to_kernel_type`、`sf_k_blocks/num_l1_sf_stages/dual_aiv_dequant` |
| 8 | `csrc/jit_kernels/fp8_gemm.hpp` | `launch_fp8_gemm`（:11-79）、`launch_fp4_gemm`（:83-151） | 27 个模板实参、两个函数的唯二差异（缓存键与 ab_dtype） |
| 9 | `csrc/jit_kernels/fp8_dequant_gemm.hpp` | `launch_fp8_dequant_gemm`（:24-95） | 末尾多出的 `direct_store, dual_aiv_dequant, single_m_block` 三个实参 |
| 10 | `csrc/jit_kernels/heuristics.hpp` | `get_gemm_dtype`（:80-90）、SF 配置（:156-183）、解码代价（:506-514）、dual 开关（:727-736） | `has_sf/dequant_b` 如何改变候选搜索 |
| 11 | `deep_gemm/include/deep_gemm/fp8_gemm.hpp` | `fp8_gemm_impl`（:36-287） | 今天的主战场，见下方逐段导读 |
| 12 | `deep_gemm/include/deep_gemm/fp8_dequant_gemm.hpp` | `vf_build_lut`（:12-40）、`vf_deq_nd2nz`（:42-93）、`fp8_dequant_gemm_impl`（:113-443） | LUT、解码、AIC/AIV 协同 |
| 13 | `deep_gemm/include/deep_gemm/ascend/copy.hpp` | `copy_gm_to_l1_mx`（:161-197）、`copy_l1_to_l0a/b_mx`（:199-229）、`fill_tail_block`（:333-356）、`mad`（:370-381） | `_mx` 搬运族与 `asc_mmad_mx` 分派 |
| 14 | `deep_gemm/include/deep_gemm/common.hpp` | 常量（:63-65）、`get_frac_k`（:67-80）、`gm_ptr::index/group`（:156-172） | `MX_SF_DIVISOR`、fp4 的 `pack_size=2` 寻址 |

**`fp8_gemm_impl` 逐段导读**（行号见 `deep_gemm/include/deep_gemm/fp8_gemm.hpp`）：

- **`:9-17` 头注释**：一句话总纲——"Computes D = A @ B^T (logical NT shape)"，GM→L1 的 nd2nz 后 L1 恒为 NZ 分形，major 只影响 nd2nz 方向与 L1→L0 是否 transpose；"SF is always MN-major. D is always N-major."
- **`:18-45` 模板与签名**：对比第 15 天的 BF16 模板，新增 `kSFKBlocks/kNumL1SFStages`（:24）、`ab_dtype_t` 成为泛型（:30）、`__mix__(1, kDirectStore ? 0 : 2)`（:36）——直写时连 AIV 都不要。入参 `gm_a/gm_b` 带 major（:37-38），`gmem_sfa/gmem_sfb` 是裸 `__gm__ int16_t*`（:39-40），在 `:65-66` 才包成 MN-major `gm_ptr`。
- **`:60-72` 运行期形状与 SF 派生量**：`shape_sf_k`、`gm_sfa/gm_sfb`、`kSFPairsPerKBlock/kSFPairsPerChunk`、`kFracK`——3.3 节的常量都从这里来。
- **`:80-94` 片上布局与断言**：四块 L1（a/b/sfa/sfb）+ 三块 L0；`static_assert(l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes)`（:89）把 SF 纳入容量检查；`:93-94` 的 `BLOCK_K % MAD_K == 0`、`MAD_K % 64 == 0` 是 MX 粒度的硬约束。
- **`:104-125` AIC 准备**：`asc_set_atomic_add_float`（直写+累加时，:105）、`asc_set_mmad_direction_n`（:106）、事件初始化（:111-113）、复用判断（:124-125）。
- **`:127-155` 块级循环头**：与 BF16 相同的 `get_next_block` 调度，多出 `sf_idx_base = scheduler.get_sf_idx_base()`（:133，实现在 `scheduler.hpp:125`——k-grouped 时 SF 的组偏移）与 `sfa/sfb` 地址计算（:139-140）。
- **`:157-171` k-block 循环前半**：3.5 节的 `k64/kc0/fill` 三件套。
- **`:173-187` SF chunk 搬运**：3.3 节的按 chunk 搬运。
- **`:193-232` MAD 双层循环**：3.4 节的核心；`sf_y`（:209）把 MAD_K 切片映射到 SF 列。
- **`:234-242` 收尾**：最后一个 k-block 时直写 `copy_l0c_to_gm` 或 `epilogue.aic_store`。
- **`:247-259` 级推进**：A/B 与 SF 各自推进事件下标；`k_within_chunk >= kSFKBlocks` 时推进 SF 级（:253-258）。
- **`:267-286` AIV 分支**：直写直接 return（:268）；k-grouped 空组写零（:270）；`aiv_drain`（:284）与 BF16 完全同构。

**`fp8_dequant_gemm_impl` 只读三处差异**：`UB 缓冲规划`（:150-167）、`AIC 侧 intra-block 同步`（:210-211, :283-284, :339-341）、`AIV 解码分支`（:362-422，见 3.6 节流水图）；其余与 `fp8_gemm_impl` 逐行对照即可，`l1b` 的写入者从 MTE2 换成了 AIV（AIC 侧的 `copy_gm_to_l1` 只搬 A，`:261`；L1B 的尾补零仍在 AIC 侧，`:265-266`）。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通三条路径的正确性测试**（正确性断言与 JIT 编译照常执行，`--skip-prof` 只跳过性能测量）：

   ```bash
   python tests/test_fp8_gemm.py normal --skip-prof
   python tests/test_fp4_gemm.py normal --skip-prof
   python tests/test_fp8_dequant_gemm.py normal --skip-prof
   ```

   对照 3.1 节确认：标题行分别是 `fp8 x fp8`、`fp4 x fp4`、`fp8 x fp4 gemm: Normal`（`common.py:146`）；`Num tests` 为 42（`normal` 的 desc 枚举与 bf16 共用 `_enumerate_normal`，但 fp32 输出的两个 (n,k) 只在 `a_dtype == bfloat16` 时加入，`generators.py:563-564`，所以比 bf16 的 54 少 12）。

2. **跑非对齐与 recipe 场景**：

   ```bash
   python tests/test_fp8_gemm.py unaligned-k --skip-prof
   ```

   把 `unaligned-k` 的 5 个 shape（`generators.py:519-541`）与 3.5 节的 `k64/kc0/fill` 对上：哪些 shape 会触发 `fill_tail_block`？（提示：k=4/15/17/234/4095 对 64 取余。）recipe 场景没有独立 CLI 选项，只能通过 `all` 分支的最后一项（`test_fp8_gemm.py:55` 的 `test_recipes=True`）或在 Python 里直接调用 `test_fp8_gemm(test_recipes=True)` 触发。

3. **观察低精度配置**：

   ```bash
   export DG_PRINT_CONFIGS=1
   python tests/test_fp8_gemm.py normal --skip-prof 2>&1 | grep -m5 GemmConfig
   ```

   记录同一 shape 下 `l1_sf_stages`、`sf_k_blocks`、`l1_stages`（应 ≤ 4）与 BF16 的差异；再跑 `test_fp8_dequant_gemm.py` 看 `dual_aiv_dequant` 何时为 true。

4. **核对 ACLNN 交叉验证的范围**：对照 `common.py:98-122` 静态列出 `normal` 场景中会被 ACLNN 跳过的 desc 及原因（K 非 64 倍数 / recipe 的 `gran_mn != 1` / MN-major 且 `ceil(k/32) <= 2` / fp8×fp4 布局或输出 dtype 不符）——被跳过时测试不会有任何提示，`has_ref` 静默返回 False，这正是需要静态分析的原因。

### B 组：无 Ascend 环境（静态任务）

1. **LUT 手算**：根据 `vf_build_lut`（`fp8_dequant_gemm.hpp:24-38`）写出 16 项表的完整映射（幅值码 × 符号），并用 E4M3 的 `[s][eeee][mmm]` 位域逐项验证 `0x30 + (code << 2)` 对 code 2..7 的正确性（应得到 1.0、1.5、2.0、3.0、4.0、6.0）；解释为什么 code 0 和 code 1 需要 `asc_select` 特判。
2. **MX 反量化手算**：设 `A[1×64]` 的前 32 个元素共享 `sfA = 2^-2`、后 32 个共享 `2^-1`（正好打包成一个 int16 的两个字节），`B[1×64]` 全程共享 `sfB = 2^0`。取 A 的 4 个码 `{0b0110, 0b1110, 0b0001, 0b1000}`（+4、-4、+0.5、-0）与 B 的对应码 `{0b0001, 0b0010, 0b0101, 0b0111}`（+0.5、+1、+3、+6），按 `D = Σ (a×sfA)×(b×sfB)` 算出这 4 项的部分和，并写出如果硬件忘记应用 sfA 会得到什么（错多少倍）。
3. **SF 布局推演**：`M=4, K=128, recipe=(1,1,32)`。写出：用户侧 `sfA` 的形状（fp32，`[4, 4]`）；Kernel 侧 GM SF 的形状（`[4, ceil(128/64)=2]` int16，MN-major）；`BLOCK_K=128` 时的 `kSFPairsPerKBlock`（=2）与 `kSFKBlocks ≤ 4` 时一个 SF chunk 最多覆盖的 K 范围（512）。
4. **模板实参对照**：打开 `csrc/jit_kernels/fp8_gemm.hpp:38-66`，把 27 个 `std::format` 实参按"major(2) / shape(3) / block(3) / mad(3) / 流水(2+2+1) / launch(1) / L2(3) / 语义(4) / dtype(2) / epilogue(1) / 对齐(1) / 直写与单块(2)"分组，各写一行来源说明；再对照 `launch_fp8_dequant_gemm.hpp:55-83` 找出多出的实参。
5. **AIC/AIV 时序图**：以 `l1_event_idx ∈ {0, 1}`（`kNumL1Stages=2`）为横轴，画 AIC、AIV0（解码）、AIV1（drain）三条泳道，标注 `set_intra_block / wait_intra_block` 的配对（`fp8_dequant_gemm.hpp:283-284, :339-341, :406-407, :417`）与 L1B 级的占用区间。
6. **K 尾演算**：对 fp8 与 fp4 分别取 `actual_k = 70`，计算 `k64`、`kc0`、`fill_tail_block` 是否触发及补零区间；再取 `actual_k = 33` 重复一遍。

## 六、当天产出

**BF16 与低精度 GEMM 的主要差异表**——以 3.7 节为参考答案，自己合上教程写一版，至少覆盖：输入与 API、Host 侧 SF 处理、dtype 派发、Kernel 选择、L1/L0 缓冲差异、MAD 指令与反量化位置、累加精度、K 尾处理、AIV 分工、配置字段。写完后把说不清的行标出来，回到第四节对应站点重读。有环境的同学附上 A 组第 3 步的配置记录。

## 七、自测题

1. `dispatch_fp8_fp4_gemm` 收到 `(fp4, fp8)` 会发生什么？
   > 落入 `else` 分支，`DJ_HOST_ASSERT(false, "Unsupported operand dtypes")`（`gemm.hpp:30-31`）；只有 fp8×fp8、fp4×fp4、fp8×fp4 三种组合。
2. `MX_SF_DIVISOR` 是多少？一个 int16 的 SF 值覆盖多少个 K 元素、内部怎么组织？
   > 64（`common.hpp:65`）。一个 int16 = 一个 "SF pair" = 两个 UE8M0 字节，各覆盖 32 个连续 K 元素，对应 recipe 的 `gran_k=32`。
3. UE8M0 是什么？用户传 fp32 的 SF 有什么约束，为什么？
   > 8 位纯指数格式，值为 2^e。fp32 形式的 SF 只允许 8 位指数且尾数必须为 0（`gemm.hpp:209-215`，设备侧 `transform_sf.hpp:19-20` 断言），因为硬件 `asc_mmad_mx` 把 scale 乘法实现成指数加法。
4. FP8×FP8 路径上，反量化发生在哪里？高精度累加发生在哪里？
   > 反量化在 Cube 的 `asc_mmad_mx` 内部（`copy.hpp:374-376`）——SF 经 `copy_l1_to_l0a/b_mx` 随数据进入 L0A/L0B 后由硬件应用于对应 64 元素组；累加在 L0C 的 fp32 累加器（`fp8_gemm.hpp:87`），`is_first/ENABLE_KEEP/ENABLE_UPDATE` 机制与 BF16 相同。
5. FP4×FP4 为什么可以复用 `fp8_gemm_impl`？FP8×FP4 为什么不行？
   > fp4 以 `float4_e2m1x2_t`（1 字节两个 E2M1）作为 `ab_dtype_t` 实例化同一模板（`fp8_gemm.hpp:83-151`），Cube 的 `asc_mmad_mx` 原生接受 8 位打包操作数。FP8×FP4 的 B 需要先在 AIV 上 LUT 解码成 FP8 写入 L1B（`fp8_dequant_gemm.hpp:12-93`），AIV 流水不同，所以要独立的 `fp8_dequant_gemm_impl`，但其 Cube 路径与 fp8 完全相同。
6. fp8 的 K 尾处理比 BF16 多了什么？fp4 为什么常常不需要？
   > MAD 的 K 上界是对齐到 64 的 `k64`（SF 粒度），而搬运补零只到 `kc0 = aligned(k, kFracK)`；缺口 `[kc0, k64)` 由 `fill_tail_block` 显式补零（`fp8_gemm.hpp:167-170`，`copy.hpp:333-356`）。fp4 的 `kFracK = 64` 恰等于 `MX_SF_DIVISOR`，K-major 时 `kc0 == k64`，无需补；fp8 最多补 32 个元素。
7. `kNumL1SFStages` 为什么不等于 `kNumL1Stages`？SF 的搬运频率和 A/B 有何不同？
   > SF 每 64 个 K 才一个 int16，数据量小，按 `kSFKBlocks` 个 k-block 组成的 chunk 搬运（`fp8_gemm.hpp:173-187`），级数固定为 2（`heuristics.hpp:160`），且用 `kSFEventBase` 起始的独立事件区间（`:108-113`）；A/B 每个 k-block 都搬、级数更多但上限被 SF 挤占为 4（`heuristics.hpp:167-168`）。
8. `dual_aiv_dequant` 是什么？它强制要求什么条件？
   > 让两个 AIV 一起解码 B（各负责一半行，`fp8_dequant_gemm.hpp:362-363, :391-392`），由启发式在 direct_store、Normal 类型且解码带宽不是瓶颈时开启（`heuristics.hpp:727-736`）；它 static_assert 要求 `kDirectStore`（`:129-131`），此时没有 AIV 剩余做 drain，D 由 AIC 直写。

---

明天（第 33 天）我们钻进 `transform_sf_into_required_layout`：recipe 如何解析、UE8M0 如何打包、为什么不能总用用户给的 SF 布局。
