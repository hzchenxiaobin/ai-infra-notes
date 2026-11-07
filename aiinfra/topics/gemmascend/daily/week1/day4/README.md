# 第 1 周 · 第 4 天：学习数值类型与量化基础

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 1 周"GEMM 与 Ascend 基础"的第 4 天教程，可独立阅读，不依赖前序内容。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 画出 FP32、BF16、FP8（E4M3）、FP4（E2M1）和 UE8M0 的位布局，说明"指数位决定动态范围、尾数位决定相对精度"；
2. 用步长（spacing）演算证明：BF16 累加 1.0 会在 256 处"卡死"，而 FP32 能得到精确结果——从而解释**低精度计算为什么必须搭配高精度累加**；
3. 写出带 Scaling Factor（SF）的量化/反量化公式 `x ≈ round(x/sf) × sf`，并解释仓库为何把 `sf` 取成"不小于 `amax / 448`（E4M3）或 `amax / 6`（E2M1）的最小 2 的幂"（UE8M0）；
4. 解释量化粒度 `recipe = (gran_m, gran_n, gran_k)` 的含义、仓库默认的 `(1, 1, 32)` 粒度，以及"K 方向每 32 个元素一个 UE8M0、相邻两个打包进一个 int16"的物理布局；
5. 在源码中定位三类证据：量化参考实现（`tests/generators.py`）、SF 布局契约与转换（`csrc/apis/layout.hpp`、`deep_gemm/include/deep_gemm/transform_sf.hpp`）、kernel 侧的 FP32 累加器与 SF 进 L0 的通路（`fp8_gemm.hpp`、`ascend/copy.hpp`）；
6. 独立完成 BF16、FP8、FP4 的优缺点对比表（当天产出）。

## 二、预备知识

- **二进制浮点表示**：知道"符号 + 指数 + 尾数"的记数法，以及 `1.011₂ × 2^e` 这类写法。不需要记忆 IEEE 754 的全部细节，3.1 节会从零讲。
- **第 1～3 天的内容**：`2MNK` 计算量、GEMM 数据量与 Tiling 的概念。今天用第 1 天的数据量公式算"降精度省多少字节"，其余不强依赖。
- **Python 基础（可选）**：实践脚本只用标准库 `struct` 做位操作，CPU 即可运行；没有 Python 时用纸笔完成等价演算。
- **不需要 Ascend 硬件**：今天全部内容（位布局、量化演算、源码阅读）都不依赖 NPU 环境；带硬件的可选实验放在实践步骤末尾。

背景一句话：仓库 README 第 3 行说 DeepGEMM-Ascend "支持 BF16、FP8、FP4 GEMM"，同一颗 Ascend 950DT 上三种类型的实测算力大约是 431 → 861 → 1701 TFLOPS（README 性能表，详见 3.3）。算力翻倍、带宽减半的代价就是**每个数只剩 4～8 个比特**。今天回答的问题是：这么少的比特怎么装得下神经网络里的数？装不下怎么办？以及——为什么"乘"可以低精度、"加"却不行？

## 三、详细讲解

### 3.1 五种格式的位布局：从 32 位到 4 位

今天出场的是 4 种数值类型加 1 种缩放因子格式。按占用比特从大到小：

```text
FP32（4 字节）:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM    1 + 8 + 23
BF16（2 字节）:  S EEEEEEEE MMMMMMM                     1 + 8 + 7   （截断 FP32 的尾数）
E4M3（1 字节）:  S EEEE MMM                             1 + 4 + 3   （FP8）
E2M1（半字节）:  S EE M                                  1 + 2 + 1   （FP4，两个打包进 1 字节）
UE8M0（1 字节）: EEEEEEEE                                8（纯指数，无符号，仅作缩放因子）
```

逐个说明（记"尾数位 n 位（不含隐含 1）"）：

| 格式 | 仓库中的名字 | 动态范围（正规数） | 尾数位 n | 相对步长上限（机器 ε = 2^-n） | 舍入误差上限（ε/2） |
| :--- | :--- | :--- | :---: | :---: | :---: |
| FP32 | `float` / `at::kFloat` | 约 2^-126 ～ 2^128 | 23 | 2^-23 ≈ 1.2e-7 | 2^-24 |
| BF16 | `bfloat16` / `torch::kBFloat16` | 与 FP32 相同（指数位一致） | 7 | 2^-7 ≈ 0.78% | 2^-8 ≈ 0.39% |
| E4M3 | `torch.float8_e4m3fn`（kernel 类型 `float8_e4m3_t`，见 `csrc/jit_kernels/gemm_config.hpp` 第 81 行） | 最小次正规 2^-9，最大 448 | 3 | 2^-3 = 12.5% | 2^-4 = 6.25% |
| E2M1 | `torch.float4_e2m1fn_x2`（kernel 类型 `float4_e2m1x2_t`） | 0.5 ～ 6 | 1 | 2^-1 = 50% | 2^-2 = 25% |
| UE8M0 | `fp8_e8m0_t`（打包态为 int16） | 2^-126 ～ 2^127（见下） | 0 | 纯 2 的幂 | — |

三条注解：

- **E4M3 的 448 与 E2M1 的 16 个码点**不是教科书背诵题，它们就写在仓库里：`tests/generators.py` 第 80 行 `sf_div = 448.0 if quant_dtype == torch.float8_e4m3fn else 6.0` 用 448/6.0 作量化分母（即各自的最大幅值）；第 98～101 行 `_FP4_VALUES` 列出 E2M1 全部 16 个码点 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` 及其负值——FP4 总共只有 16 种取值，这是今天一切折中的根源。
- **FP4 的物理存储**是"两个值挤一个字节"：仓库把两个 E2M1 半字节（nibble）打包进一个 `int8`（`csrc/utils/dtype.hpp` 第 9 行 `kPackedFP4 = torch::kInt8`，注释 "Canonical storage dtype for two packed FP4 values"；地址计算里的 `pack_size = 2` 见 `deep_gemm/include/deep_gemm/common.hpp` 第 157 行）。所以 `K` 方向的 FP4 数据字节数是元素数的一半。
- **UE8M0 是"只有指数的浮点数"**：一个字节存 FP32 的 8 位指数，数值恒为 2 的幂。它不是用来存数据的，只用来存**缩放因子**。仓库的校验逻辑要求用户提供的 FP32 形式 SF 必须尾数为 0：`csrc/apis/gemm.hpp` 第 209～210 行的接口注释 "when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic"，设备侧的对应检查在 `transform_sf.hpp` 第 19 行（"Float32 SF must be non-negative and can't have non-zero mantissa"）。

### 3.2 指数与尾数的分工：动态范围 vs 相对精度

一个数 `x = ±1.m × 2^e`，两种比特各管一件事：

- **指数位决定"能表示多大、多小"**（动态范围）。BF16 砍掉 FP32 一半的比特却保留全部 8 位指数，所以它俩范围相同——BF16 只是"粗一点的 FP32"。E4M3 只有 4 位指数（范围 2^-9 ～ 448，跨度约 2×10^5），E2M1 只有 2 位（0.5 ～ 6，跨度 12 倍）。
- **尾数位决定"相邻两个数差多少"**（相对精度）。区间 `[2^p, 2^(p+1))` 内可表示值的步长恰为 `2^(p-n)`。据此手算几个例子（实践步骤 1 会让你全部亲手算一遍）：

| 区间 | FP32 (n=23) | BF16 (n=7) | E4M3 (n=3) | E2M1 (n=1) |
| :--- | :---: | :---: | :---: | :---: |
| `[1, 2)` | 2^-23 | 2^-7 ≈ 0.0078 | 0.125 | 0.5 |
| `[64, 128)` | 2^-17 | 2^0 = 1 | 2^3 = 8 | （超出范围） |
| `[256, 512)` | 2^-15 | 2^1 = 2 | 8 | （超出范围） |
| `[4096, 8192)` | 2^-11 ≈ 0.00049 | 2^5 = 32 | （超出 448） | （超出范围） |

这张表是今天最重要的直觉：**数越大、类型越窄，相邻可表示值之间隔得越远**。在 BF16 里 4096 附近的数彼此相距 32，`4096 + 0.5` 会被舍回 4096；在 E2M1 里 4 和 6 之间没有任何数。后面的累加问题（3.6）和量化问题（3.4）都是这张表的推论。

### 3.3 GEMM 为什么降精度：数据量减半、算力翻倍

用第 1 天的工具算账。取仓库 README 性能表的 shape `M=4096, N=7168, K=16384`，A+B 共 `(4096+7168) × 16384` 个元素，SF 开销按默认粒度 `(1, 1, 32)`（每行每 32 个 K 元素一个 UE8M0，见 3.5）计算：

| 类型 | 数据字节 | SF 字节 | 合计 | 相对 BF16 | 实测算力（README） | 硬件极限 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BF16×BF16 | 352 MiB | — | 352 MiB | 100% | 431 TFLOPS | 432（99.8%） |
| FP8×FP8 | 176 MiB | 5.5 MiB | 181.5 MiB | 52% | 861 TFLOPS | 865（99.5%） |
| FP4×FP4 | 88 MiB | 5.5 MiB | 93.5 MiB | 27% | 1701 TFLOPS | 1730（98.3%） |
| FP8×FP4 | 132 MiB | 5.5 MiB | 137.5 MiB | 39% | 861 TFLOPS | 865（99.5%） |

（算力数据来自 `README.md` 第 108～113 行的性能表，实测环境 Ascend 950DT；SF 字节数 = `(4096+7168) × (16384/32) = 5,767,168` 个 UE8M0，每个 1 字节。）

三个观察：

1. **类型每降一档，数据量减半、cube 算力翻倍**：431 → 861 → 1701，几乎精确的 2 倍阶梯。对第 1 天结论"大 GEMM 是算力受限"的形状，降类型直接翻倍 TFLOPS；对"带宽受限"的形状，数据量减半直接提高有效带宽。
2. **FP8×FP4 的算力与 FP8×FP8 相同（861）**：因为 FP4 那一路要先反量化成 FP8 再进矩阵乘加单元（3.7）。混合精度的收益在带宽（132 MiB < 176 MiB），不在算力。
3. **SF 开销很小**：FP8 下约 3.1%、FP4 下约 6.25%（3.5 节详细算）。这就是"缩放因子"方案的卖点——用 0.25 比特/元素的开销，把 8/4 比特的小类型撑回完整的动态范围。

### 3.4 Scaling Factor：让小类型装下大范围的数

E4M3 的正规范围只有 2^-9 ～ 448，而模型里一层的激活值范围千差万别。**分块缩放**是标准解法：把张量按粒度分组，每组算一个缩放因子，整组除以它再量化，用时乘回来：

```text
量化:   sf = 按组选取的缩放因子（2 的幂，UE8M0）
        x_q = round_to_dtype(x / sf)          # 落进 E4M3/E2M1 的码点
反量化: x̂ = x_q × sf
数学上: D[i][j] = Σ_p (a_q × sfa) × (b_q × sfb) —— SF 在乘法里可以提出来
```

仓库的参考实现是 `tests/generators.py` 的 `quantize_with_major()`（第 61～95 行），四步：

1. **分组**：按 `gran` 把张量 view 成 `[组数..., 组内元素]`（第 74～78 行）；
2. **求组内最大幅值并选 SF**（第 80～81 行）：

   ```python
   sf_div = 448.0 if quant_dtype == torch.float8_e4m3fn else 6.0
   sf_float = ceil_to_ue8m0(sf_view.abs().amax(dim=-1).clamp(1e-4) / sf_div)
   ```

   除以 `448`（或 `6.0`）保证组内最大值除以 SF 后不超出类型上限；`ceil_to_ue8m0`（第 27～31 行）把结果**向上取整到 2 的幂**：取 FP32 的指数位，尾数非零就进一位，再夹到 `[1, 254]`（对应 SF 值 `2^-126 ～ 2^127`）。向上取整保证 `amax / sf ≤ 448` 恒成立，量化**永不饱和**，代价是最多浪费 2 倍的头部空间（例如 `amax/448 = 0.51` 时 SF 取 1.0）。
3. **量化**：`x / sf_full` 后转成 E4M3（第 89～90 行调用 `_quantize_to_fp8_with_major`）或 E2M1（第 91～92 行调用 `_quantize_to_fp4_with_major`，第 36～45 行给出 E2M1 的中点舍入边界 0.25/0.75/…/5.0 和 nibble 打包）；
4. **精确参考**：返回 `(x_quant, sf)` 与反量化后的 FP32 参考 `f32_repr * sf_full`（第 95 行）。测试就用这个参考与 kernel 输出比对（第 6 天不动它，第 29 天细读）。

手算一个例子（实践步骤 2 会用脚本验证）。设某组 `amax = 6.5`：

- **E4M3**：`6.5 / 448 = 0.01451`，向上取 2 的幂得 `sf = 2^-6 = 0.015625`。组内 `x = 0.2`：`0.2 / 0.015625 = 12.8`，落在步长为 1 的区间 `[8, 16)`，舍入到 13；反量化 `13 × 0.015625 = 0.203125`，相对误差 1.6%。而 `amax` 自己：`6.5 / 0.015625 = 416`，恰好是 E4M3 精确码点，误差 0。
- **E2M1**：`6.5 / 6 = 1.083`，向上取 2 的幂得 `sf = 2`。同一个 `x = 0.2`：`0.2 / 2 = 0.1`，最近的 E2M1 码点是 0——**这个值被彻底丢了**。`6.5 / 2 = 3.25` 舍入到 3，反量化 6，误差 7.7%。

结论：**SF 只能撑"范围"，撑不了"分辨率"**。E4M3 有 3 位尾数 + 4 位指数，粗粒度也够用；E2M1 只有 16 个码点，粒度必须更细（或接受丢值），这正是下一节的主题。

### 3.5 量化粒度与 SF 的物理布局：recipe、ue8m0x2 与"32 的由来"

**粒度（recipe）** 描述"多少个元素共享一个 SF"。仓库接口（`csrc/apis/gemm.hpp` 第 205～224 行的 `fp8_fp4_gemm` 注释）把它写成三元组 `recipe = (gran_m, gran_n, gran_k)`：

```text
A 的 SF 形状: [ceil(M / gran_m), ceil(K / gran_k)]
B 的 SF 形状: [ceil(N / gran_n), ceil(K / gran_k)]
```

- `gran_m = gran_n = 1`：**每行一个 SF**（gran_k 方向再切）；
- `gran_m = 32`：A 的 32 行共享一个 SF（"块缩放"）；
- `gran_k = 32`：K 方向每 32 个元素一个 SF。

三个仓库事实值得记录：

1. **默认粒度是 `(1, 1, 32)`**：`csrc/apis/layout.hpp` 第 122～123 行 `get_recipe()` 的缺省值 `recipe_a/recipe_b = (1, 32)`；测试描述符的缺省 `recipe: tuple = (1, 1, 32)`（`tests/generators.py` 第 263 行）。也就是"每行 × 每 32 个 K 元素"。
2. **`gran_k` 被硬性锁定为 32**：`csrc/apis/layout.hpp` 第 55 行 `DJ_HOST_ASSERT(gran_k == 32, "gran_k must be 32")`。为什么是 32？硬件视角的解释在 `common.hpp` 的 `get_frac_k()`（第 73～79 行，注释 "32 bytes per C0"）：L1→L0 的搬运以 32 字节为一个 C0 行——FP8 一行恰好 32 个元素，正好一个 UE8M0；FP4 一行 64 个元素，恰好**一对** UE8M0。也就是说，`gran_k = 32` 让"一个 SF 管辖的数据"与"一次 L0 搬运的数据"严格对齐。
3. **两个相邻 K 的 SF 打包进一个 int16**：README 第 56 行 "each pair of UE8M0 scaling factors along the K dimension is packed into an `int16`, and the packed values are stored in MN-major order"。常量 `MX_SF_DIVISOR = 64`（`common.hpp` 第 65 行，注释 "K-elements per SF pair"）就是这对 SF 覆盖的 K 元素数。因此转换后的 SF 张量形状是 `[.., MN, ceil(K / gran_k / 2)]`（`csrc/apis/layout.hpp` 第 36 行），且**总是 MN-major**（第 22 行注释）。

**SF 的布局转换**是用户输入与 kernel 要求之间的桥梁。用户可以给任意 major、FP32 或已打包 int16 的 SF；`transform_sf_into_required_layout()`（`csrc/apis/layout.hpp` 第 68～92 行）先用 `check_sf_layout()`（第 37～65 行）判断"是否恰好已是目标布局"（第 57～62 行：int16 + `gran_mn=1` + MN-major 则原样返回），否则调用 `launch_transform_sf()` 起一个设备端 kernel `transform_sf`（`deep_gemm/include/deep_gemm/transform_sf.hpp` 第 455 行起）完成三件事：**校验**（FP32 SF 尾数必须为 0，第 14～21 行 `check_validation`）、**打包**（`asc_shiftright_scalar(sf, sf, 23)` 取出指数字节，两个 K 相邻 SF 拼进一个 int16，第 217～281 行的 `pack_validate_*_simd_impl`）、**按 `gran_mn` 沿 MN 复制**（第 89～121 行 `repeat_to_256_simd_impl`，支持 2～128 的 2 的幂）。第 33 天会专门精读这个 kernel。

**粒度的权衡**用开销公式一眼看清（SF 个数 = `ceil(MN/gran_mn) × ceil(K/32)`，打包后每 SF 1 字节）：

| 粒度 | SF 字节（`4096×7168×16384`） | 相对 FP8 数据（176 MiB） | 相对 FP4 数据（88 MiB） | 精度 |
| :--- | :---: | :---: | :---: | :--- |
| `(1, 1, 32)` 每行 | 5.5 MiB | 3.1% | 6.3% | 每行独立适配范围 |
| `(32, 32, 32)` 块 | 176 KiB | 0.1% | 0.2% | 32 行共享，离群值伤害全块 |

仓库两边都测：`test_recipes` 覆盖 `(1,32,32)/(32,1,32)/(32,32,32)`（`tests/generators.py` 第 448～452 行），SF 布局测试覆盖 `gran_mn ∈ (1, 32, 128)`（第 683 行 `enumerate_sf_layout`）。3.4 的例子已经说明伤害的形式：与离群值同组的小值会被整体吞掉。

### 3.6 低精度计算为什么仍要 FP32 累加

现在回答计划里的思考题。先看**如果用 BF16 当累加器会发生什么**——不需要硬件，算一遍就知道（实践步骤 2 的脚本给出了机器验证）：

```text
用 BF16 累加 1.0 + 1.0 + ...（4096 个）：
  部分和到达 256 后，区间 [256, 512) 的步长是 2，
  256 + 1.0 = 257 落在 256 与 258 正中间，舍入到偶数 → 还是 256。
  此后每加一个 1.0 都被原样舍回 —— 结果永远卡在 256。
用 FP32 累加同样的 4096 个 1.0：结果精确等于 4096。
```

问题不在于"1.0 表示不了"（它在所有格式里都精确），而在于**大数吃小数**：部分和越大，累加器的步长越大，小于半步长的乘积被整段丢弃。GEMM 的 K 高达 16384（README 性能表的 shape），丢掉的不是一个 0.5，而是成千上万个乘积。

**为什么 FP32 够、而且是"恰好够"**？关键观察是：**低精度格式的乘积在 FP32 里是精确的**——

```text
E4M3 × E4M3：尾数各 4 位（含隐含 1）→ 乘积至多 8 位尾数   ≪ FP32 的 24 位
E2M1 × E2M1：尾数各 2 位 → 乘积至多 4 位                  ≪ 24 位
BF16 × BF16：尾数各 8 位 → 乘积至多 16 位                 ≤ 24 位
UE8M0 缩放：× 2^e 只改指数，不产生任何舍入
```

所以"低精度乘、FP32 加"的流水线里，**唯一的舍入发生在加法本身**，每步相对误差不超过 2^-24，累加 16384 步的误差上界也在 1e-4 量级，远小于量化本身 6.25%（E4M3）/25%（E2M1）的误差——累加误差不污染量化预算，这个设计才算成立。反过来若用 BF16 累加，单步舍入就是 2^-8，而且有吞项风险，量化做得再好也白搭。

**仓库的三处直接证据**：

1. **L0C 累加器永远是 FP32**。BF16 kernel：`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 第 64 行 `constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);`；FP8/FP4 kernel：`fp8_gemm.hpp` 第 87 行同一句声明；FP8×FP4 kernel：`fp8_dequant_gemm.hpp` 第 182 行。容量常量的注释也写明了：`common.hpp` 第 57 行 `L0CSizeBytes = 256 * 1024;  // L0C (fp32 accumulator)`。第 3 天见过的 L0C 容量检查 `block_m × block_n × sizeof(float) ≤ 256 KiB`，那个 `sizeof(float)` 的出处就是这里。
2. **乘加语义**。`mad()`（`ascend/copy.hpp` 第 370～381 行）对 1 字节操作数走 `asc_mmad_mx`——矩阵乘加指令在**乘的同时应用两侧的 UE8M0 缩放**（反量化融合进 cube 单元），结果累加进 FP32 的 L0C；对 2 字节操作数走 `asc_mmad`。K 循环里 `is_first` 控制首次清零累加器、`ENABLE_KEEP` 控制后续累加进同一块 L0C（`bf16_gemm.hpp` 第 149～151 行、`fp8_gemm.hpp` 第 228～230 行 `mad(..., is_last ? ENABLE_UPDATE : ENABLE_KEEP, is_first)`）。
3. **输出时才降精度，且有舍入策略**。L0C 的 FP32 结果经 `copy_l0c_to_ub`（自带 `F322BF16` 转换模式，`ascend/copy.hpp` 第 236～238 行的 `infer_quant_mode`）或 epilogue 算子转成 BF16：普通路径是 `asc_float2bfloat16_rn`（round-to-nearest，`epilogue/operators.hpp` 第 151、215 行）；另有 `StochasticRoundToBF16`（第 225 行起）把被丢弃的低位按哈希概率进位，消除系统性偏差，测试用 `assert_stochastic_bf16_matches_fp32`（`tests/utils.py` 第 70 行起）逐位复现同一套哈希来验证。

一个补充观察：`tests/utils.py` 第 49～51 行的注释说 BF16 kernel 与 ACLNN 参考会"差最多 ~1 ULP，因为 K 累加顺序不同"——连**FP32**累加都只是"顺序不同则结果不同"，可见累加精度是何等敏感的东西；而低精度测试（`tests/test_fp8_gemm.py` 第 21 行、`tests/test_fp4_gemm.py` 第 20 行）用 `threshold=1e-5` 的余弦相似度比较，正因为参考值是"反量化后的精确值在 FP32 里乘加"（`tests/generators.py` 第 320～322 行 `d = (alpha_value * (a_repr.float() @ b_repr.float().T) + c).to(cd_dtype)`），kernel 只要同样用 FP32 累加，差异就只剩累加顺序效应，量级远小于阈值。

### 3.7 三条低精度路径与一张总账

低精度 GEMM 的操作数组合在 host 侧就分派完毕（`csrc/apis/gemm.hpp` 第 22～33 行 `dispatch_fp8_fp4_gemm`）：

| 操作数 | 执行路径 | cube 通路 | SF 通路 | 出处 |
| :--- | :--- | :--- | :--- | :--- |
| FP8 × FP8 | `launch_fp8_gemm` → `fp8_gemm_impl` | E4M3 数据 + UE8M0 | GM → L1(`l1_sfa/l1_sfb`) → L0(`fp8_e8m0_t`) → `asc_mmad_mx` | `csrc/jit_kernels/fp8_gemm.hpp` 第 11 行 |
| FP4 × FP4 | `launch_fp4_gemm` → **同一个** `fp8_gemm_impl`，只是实例化为 `float4_e2m1x2_t` | 打包 nibble + UE8M0 | 同上 | 同文件第 81～83 行注释 |
| FP8 × FP4 | `launch_fp8_dequant_gemm` → `fp8_dequant_gemm_impl` | B（FP4）先在 **AIV** 上查表反量化为 FP8（"build LUT → vf_deq_nd2nz → copy_ub_to_l1"），再走与 FP8 相同的 `mad_mx` | 同上 | `csrc/jit_kernels/fp8_dequant_gemm.hpp` 第 12～18 行 |
| FP4 × FP8 | **不支持**（`DJ_HOST_ASSERT(false, "Unsupported operand dtypes")`） | — | — | `csrc/apis/gemm.hpp` 第 28～31 行 |
| BF16 × BF16 | `launch_bf16_gemm` → `bf16_gemm_impl` | `asc_mmad`，无 SF（`GemmDesc.sfa_ptr/sfb_ptr` 为 nullptr，`gemm_config.hpp` 第 44～45 行注释 "FP8 only; nullptr for BF16"） | — | `bf16_gemm.hpp` |

SF 在 kernel 内的数据流（`fp8_gemm.hpp`）与 A/B 数据完全平行，只是更小、更晚：

```text
GM:  A/B 数据 (fp8/fp4)         SF (int16, MN-major, ue8m0x2 打包)
      │ copy_gm_to_l1              │ copy_gm_to_l1_mx（第 178～185 行；按 kSFKBlocks
      ▼                            ▼           个 K 块合并成一次搬运）
     L1: l1a/l1b 多级 stage        L1: l1_sfa/l1_sfb（第 82～83 行，级数 = kNumL1SFStages）
      │ copy_l1_to_l0a/b           │ copy_l1_to_l0a/b_mx（第 221～224 行，
      ▼                            ▼   sf 以 fp8_e8m0_t 进入 L0；第 218 行注释
     L0A/L0B ────────┐             │   "AB can enter L0 while the first SF request
                      ▼            ▼    is still in flight"）
              mad / asc_mmad_mx：×sfa ×sfb 融合，FP32 累加进 L0C
                      ▼
              Epilogue：L0C(FP32) → UB → BF16/FP32/FP8 输出 → GM
```

把今天全部内容记成一句话：**数据类型决定"每个数占几位、能表示哪些值"，Scaling Factor 决定"哪段范围被搬进小类型"，而 FP32 累加器决定"这些近似乘积加起来之后还算不算数"**。

## 四、源码导读

今天的导读主题是"**数值契约从 Python 测试到 device kernel 的完整传递**"，全程静态阅读、无需硬件。建议对照仓库依次打开以下文件（路径相对 `DeepGEMM-Ascend/`）。

### 站点 1：量化的参考实现 —— `tests/generators.py`

- `ceil_to_ue8m0()`（第 27～31 行）：`exp = ((bits // 2**23) % 256) + (bits % 2**23).bool().int()`——FP32 位模式取出指数、尾数非零则进位，`clamp(1, 254)` 后只留指数。这是 UE8M0 "向上取 2 的幂"的权威定义，实践步骤 2 的脚本逐行复刻了它。
- `_quantize_to_fp4_with_major()`（第 34～50 行）：第 36～37 行的注释列出 E2M1 的 8 个非负值与 7 个舍入中点；第 47 行 `(fp4_repr[..., ::2] & 0x0f) | ((fp4_repr[..., 1::2] & 0x0f) << 4)` 就是"两个 nibble 打包进一字节"。
- `_FP4_VALUES`（第 98～101 行）与 `_FP4_BYTE_LUT`（第 102 行）：全部 16 个码点及 256 种字节组合的解码表。
- `quantize_with_major()`（第 61～95 行）：3.4 节逐步讲解的四步流程；注意第 80～81 行的 `sf_div = 448.0 / 6.0` 是 E4M3/E2M1 的最大幅值，第 95 行返回的反量化参考就是测试的"标准答案"。
- `generate_random_quantized_with_major()`（第 124～166 行）：另一种测试策略——直接生成随机码点与随机 SF（FP8 从 `{2^-9, 2^-8, 2^-7}` 取、FP4 从 `{0.25, 0.5, 1.0}` 取，第 138～143 行），参考值 = 解码值 × SF。它覆盖了"非真实分布但覆盖全部码点"的极端情况。
- `GemmTestDesc`（第 249 行起）：`recipe` 缺省 `(1, 1, 32)`（第 263 行）、`has_sf` 属性（第 287～288 行，任一操作数非 BF16 即有 SF）。

### 站点 2：公开 API 的数值契约 —— `csrc/apis/gemm.hpp`

- `fp8_fp4_gemm` 的注释块（第 199～224 行）是低精度接口的"合同书"：A/B 的 dtype 组合、`sfA: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major`（第 208 行）、"fp32 SF 尾数必须为 0 否则 kernel panic"（第 209～210 行）、`d: shape [M, N], bf16 or fp32`（第 216 行）、`recipe / recipe_a / recipe_b / disable_ue8m0_cast`（第 218～223 行）。
- `dispatch_fp8_fp4_gemm()`（第 22～33 行）：按 `(a_dtype, b_dtype)` 元组分派三条路径，FP4×FP8 直接断言失败。
- `get_gemm_desc()`（第 60 行起）内对 SF 的处理（第 125～154 行）：对每个操作数调用 `transform_sf_into_required_layout`，转换后的张量指针写进 `GemmDesc.sfa_ptr/sfb_ptr`（第 171～172 行）。第 11 天会精读整个函数，今天只看 SF 这一段。

### 站点 3：SF 的形状与"免转换"判定 —— `csrc/apis/layout.hpp`

- 文件头注释（第 19～36 行）：输入可以是 FP32 或 int16（packed ue8m0x2），输出**永远是** short、MN-major、按 `gran_mn` 复制，形状 `[.., MN, ceil(K / gran_k / 2)]`。
- `check_sf_layout()`（第 37～65 行）：dtype 白名单（第 43 行）；形状断言（第 53～54 行）——FP32 输入按 `ceil(k / gran_k)` 列数检查，short 输入按 `ceil(k / gran_k * 2)`（打包对折）；第 55 行 `gran_k must be 32`；第 57～62 行的免转换快路径（已是 int16 + `gran_mn=1` + MN-major）。
- `get_recipe()`（第 113～128 行）：`recipe` 与 `recipe_a/recipe_b` 互斥（第 118 行），缺省 `(1, 1, 32)`（第 122～123 行），两侧 `gran_k` 必须相等（第 125 行）。
- `transform_sf_into_required_layout()`（第 68～92 行）：先 `check_sf_layout`，再处理 `disable_ue8m0_cast` 断言（第 86～87 行："Float32 SF must be converted into ue8m0"），最后 `launch_transform_sf`。

### 站点 4：SF 打包与校验的 device kernel —— `deep_gemm/include/deep_gemm/transform_sf.hpp`

- `check_validation()`（第 14～21 行）：把所有已搬运 SF 的位模式 OR 起来做一次检查——`ascendc_assert` 要求符号位为 0 且低 23 位（尾数）为 0。**用户传错 SF 时 kernel 在设备端 panic 的那句话就来自这里。**
- `pack_validate_k_major_2row_simd_impl()`（第 217～248 行）：`asc_shiftright_scalar(sf0, sf0, 23)`（第 243 行）把 FP32 SF 右移 23 位取出指数字节，再 `asc_storealign_pack_quarter` 打包——两行代码完成"FP32 → UE8M0 字节"。
- `pack_validate_mn_major_simd_impl()`（第 251～281 行）：MN-major 输入时，相邻两个 K 的指数分别移 23 位（第 276 行）和 15 位（第 277 行）后 `asc_or` 进同一个 32 位单元——**ue8m0x2 打包的源码实体**。
- `repeat_to_256_simd_impl()`（第 89～121 行）：`gran_mn > 1` 时沿 MN 方向复制 SF（`static_assert` 限定 2～128 的 2 的幂，第 94 行），服务于 `(32, ·, 32)` 这类块粒度。
- kernel `transform_sf`（第 455 行起）：纯向量核（`__global__ __vector__`），按 M×K 块网格分派到 64 个核，多级 UB 流水。今天只看它调用了上面三个函数的位置（第 595～612 行的四个分支），完整流水线第 33 天再啃。

### 站点 5：FP32 累加器与 SF 进 L0 —— `deep_gemm/include/deep_gemm/fp8_gemm.hpp` + `ascend/copy.hpp`

- `fp8_gemm.hpp` 第 87 行 `l0c_ptr<float> l0c(MAD_M, MAD_N)`：FP32 累加器声明（对照 `bf16_gemm.hpp` 第 64 行、`fp8_dequant_gemm.hpp` 第 182 行——三个 kernel 一模一样）。
- 第 63 行 `shape_sf_k = ceil_div(shape_k, MX_SF_DIVISOR)`、第 68 行 `kSFPairsPerKBlock = BLOCK_K / MX_SF_DIVISOR`：SF 的 K 向长度按"64 元素一对"折算。
- 第 82～83 行 `l1_sfa/l1_sfb` 的声明与第 89 行 `static_assert(l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes)`：SF 也要占 L1 流水（这是 `GemmConfig.num_l1_sf_stages` 的用途，`csrc/jit_kernels/gemm_config.hpp` 第 107 行注释 "0 for BF16"）。
- 第 173～187 行：`copy_gm_to_l1_mx` 按 chunk 搬 SF（`kSFKBlocks` 个 K 块的 SF 合并成一次搬运，第 174～176 行算 chunk 大小）。
- 第 219～224 行：SF 从 L1 进 L0——`copy_l1_to_l0a_mx / copy_l1_to_l0b_mx`，实现在 `ascend/copy.hpp` 第 200～229 行，其中 `sf_l1.ptr<fp8_e8m0_t>()`（第 206、222 行）表明 SF 以 e8m0 形式进入 L0。
- `mad()`（`ascend/copy.hpp` 第 370～381 行）：`sizeof(TA) == 1 || sizeof(TB) == 1` 时走 `asc_mmad_mx`（UE8M0 缩放融合进乘加），否则 `asc_mmad`；`clear_C` 参数就是 `is_first`。
- `common.hpp` 第 65 行 `MX_SF_DIVISOR = 64;  // K-elements per SF pair`——3.5 节"打包对"常量的出处。

### 站点 6：输出侧的精度转换 —— `deep_gemm/include/deep_gemm/epilogue/operators.hpp` + `tests/utils.py`

- `asc_float2bfloat16_rn`：出现在 `ScaleByAlpha`（第 151 行）和 `AddBias`（第 215 行）里——FP32 结果转 BF16 的常规路径。
- `StochasticRoundToBF16`（第 225 行起）：对 FP32 值的四个一组（quartet）做哈希（第 277～287 行的乘法常数 0x5671d42b 等），用被丢弃的低位与随机阈值比较决定是否进位。配套测试 `assert_stochastic_bf16_matches_fp32`（`tests/utils.py` 第 70～92 行）在 CPU 上用 NumPy 复现同一套哈希逐位验证。
- `QuantizeToFP8`（第 338 行起）与 `get_ue8m0_sf_exp`（第 8～24 行）/ `get_ue8m0_sf_inv`（第 26～35 行）：epilogue 里"现场量化回 FP8"的算子——对 BF16 值求组内 amax、算 UE8M0 指数（第 19 行注释给出了位运算公式），倒数 scale `2^-e` 用 BF16 指数减法实现（第 31～34 行）。这是 3.4 节量化流程在 device 侧的镜像，说明"量化不止发生在输入侧"。
- `infer_quant_mode`（`ascend/copy.hpp` 第 232～249 行）：`bfloat16_t → asc_quant_mode::F322BF16`——L0C→UB/GM 的搬运指令自带 FP32 到目标 dtype 的转换，这是"累加器 FP32、输出可低精度"在指令层面的落点。

**阅读检查点**（能回答再进入实践）：

1. `ceil_to_ue8m0` 为什么对指数"向上取整"？这保证了量化后的什么性质？
2. 用户给的 FP32 形式 SF 有哪两条合法性要求？分别在哪个文件检查？
3. 打包后的 SF 张量一个 int16 覆盖多少个 K 元素？`gran_k` 为什么必须是 32？
4. BF16、FP8、FP8×FP4 三个 kernel 的 L0C 累加器类型分别是什么？
5. SF 从 GM 到参与计算，依次经过哪些缓冲？以什么 dtype 进入 L0？
6. `mad()` 的 `clear_C` 参数和 `ENABLE_KEEP` 分别控制什么？

## 五、实践步骤

以下步骤按顺序执行，**全部不需要 Ascend 硬件**；总时长约 2 小时。

### 步骤 1：纸笔演算位布局与步长（约 20 分钟）

不看 3.2 节的表，独立完成：

1. 画出五种格式的位布局（位数自己推：E4M3 共 8 位、E2M1 共 4 位，减去符号和指数位剩下的就是尾数位）；
2. 用步长公式 `spacing = 2^(p-n)`（区间 `[2^p, 2^(p+1))`，n 为尾数位）填出 3.2 节那张表；
3. 手算三个具体值：BF16 表示 `4096 + 0.5` 得到什么？E4M3 表示 `100`（落在 `[64, 128)`）最近的两个码点是什么、舍到哪个？E2M1 表示 `4.7` 舍到哪个码点、相对误差多少？（答案：4096；96 和 104——100 恰在两者中点，按"偶"舍到 96（1.5×2^6 的尾数为偶）；4.7 舍到 4，相对误差约 14.9%。）
4. 用 3.6 节的开头例子推演"BF16 累加 1.0 卡死在 256"：写出 255 + 1 和 256 + 1 两次加法各自的舍入过程。

### 步骤 2：用脚本验证量化与累加行为（约 30 分钟）

在自己的练习目录新建 `quant_probe.py`（不要改动仓库文件），写入：

```python
import struct

def f32_bits(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def bits_f32(b): return struct.unpack('<f', struct.pack('<I', b))[0]

def bf16_round(x):
    """把 FP32 舍入到 BF16（round-to-nearest-even，即丢弃低 16 位的标准语义）"""
    b = f32_bits(x)
    base, low = b & 0xFFFF0000, b & 0xFFFF
    if low > 0x8000 or (low == 0x8000 and (base >> 16) & 1):
        base += 0x10000
    return bits_f32(base)

def accumulate_bf16(k):
    s = 0.0
    for _ in range(k):
        s = bf16_round(s + 1.0)
    return s

# E4M3 正值表（e=15,m=7 为 NaN，排除；最大值 448，与 tests/generators.py 的 sf_div 一致）
E4M3 = sorted({(m * 2.0 ** -9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
               for e in range(16) for m in range(8)} - {0.0})
# E2M1 全部码点，同 tests/generators.py 的 _FP4_VALUES
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def nearest(x, table):
    return min(table, key=lambda v: (abs(v - x), v))

def ceil_ue8m0(x):
    """等价于 tests/generators.py 第 27~31 行的 ceil_to_ue8m0"""
    b = f32_bits(x)
    exp = min(254, max(1, ((b >> 23) & 0xFF) + (b & 0x7FFFFF != 0)))
    return 2.0 ** (exp - 127)

def quantize_group(vals, table, dtype_max):
    amax = max(abs(v) for v in vals)
    sf = ceil_ue8m0(max(amax, 1e-4) / dtype_max)
    return [nearest(v / sf, table) * sf for v in vals], sf

def rel_err(orig, deq):
    return max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(orig, deq))

print("== 1) BF16 累加 1.0 x 4096 次 vs FP32 ==")
print(f"BF16 累加结果 = {accumulate_bf16(4096)}   (卡死在 256)")
print(f"FP32 累加结果 = {float(sum([1.0] * 4096))}")
print(f"BF16 在 [4096,8192) 的步长 = {bf16_round(4096 + 17) - 4096}"
      f"，所以 4096 + 0.5 -> {bf16_round(4096 + 0.5)}")

row_small = [0.9, 1.0, 1.1, 0.95, 1.05, 1.2, 0.85, 1.0] * 4   # 32 个元素, amax=1.2
row_outlier = [300.0] + [1.0] * 31                             # amax=300

for name, table, dmax in (("E4M3(FP8)", E4M3, 448.0), ("E2M1(FP4)", E2M1, 6.0)):
    print(f"\n== 2) {name}: 逐组(1x32) SF vs 整矩阵单 SF ==")
    deq, sf = quantize_group(row_small, table, dmax)
    print(f"小值行   逐组 SF: sf={sf:<10.4g} 最大相对误差 = {rel_err(row_small, deq):7.1%}")
    deq, sf = quantize_group(row_outlier, table, dmax)
    print(f"离群行   逐组 SF: sf={sf:<10.4g} 最大相对误差 = {rel_err(row_outlier, deq):7.1%}")
    tensor_sf = ceil_ue8m0(300.0 / dmax)     # 整矩阵 SF 由离群值决定
    deq = [nearest(v / tensor_sf, table) * tensor_sf for v in row_small]
    print(f"小值行 整矩阵 SF: sf={tensor_sf:<10.4g} 最大相对误差 = {rel_err(row_small, deq):7.1%}")

print("\n== 3) ceil_ue8m0 抽查 ==")
for x in (6.5 / 448, 6.5 / 6, 300 / 6, 1e-4 / 448):
    print(f"ceil_ue8m0({x:.6g}) = {ceil_ue8m0(x)}")
```

运行 `python quant_probe.py`，预期输出：

```text
== 1) BF16 累加 1.0 x 4096 次 vs FP32 ==
BF16 累加结果 = 256.0   (卡死在 256)
FP32 累加结果 = 4096.0
BF16 在 [4096,8192) 的步长 = 32.0，所以 4096 + 0.5 -> 4096.0

== 2) E4M3(FP8): 逐组(1x32) SF vs 整矩阵单 SF ==
小值行   逐组 SF: sf=0.003906    最大相对误差 =    4.8%
离群行   逐组 SF: sf=1           最大相对误差 =    4.0%
小值行 整矩阵 SF: sf=1           最大相对误差 =    4.8%

== 2) E2M1(FP4): 逐组(1x32) SF vs 整矩阵单 SF ==
小值行   逐组 SF: sf=0.25        最大相对误差 =   16.7%
离群行   逐组 SF: sf=64          最大相对误差 =  100.0%
小值行 整矩阵 SF: sf=64          最大相对误差 =  100.0%

== 3) ceil_ue8m0 抽查 ==
ceil_ue8m0(0.0145089) = 0.015625
ceil_ue8m0(1.08333) = 2.0
ceil_ue8m0(50) = 64.0
ceil_ue8m0(2.23214e-07) = 2.384185791015625e-07
```

对照输出逐条回答：

1. **BF16 累加卡死在 256、FP32 精确到 4096**——这就是 3.6 节的纸笔结论，也是"低精度必须高精度累加"的最直接证据；
2. **E4M3 对粒度不敏感**（本例逐组与整矩阵同为 4.8%）：4 位指数让它自带较宽的范围；**E2M1 对粒度极敏感**：逐组 SF 把"小值行"从 100% 误差救回到 16.7%，但**与离群值同组的 1.0 仍然全军覆没**（离群行 100%）——与离群值共享 SF 的小值无药可救，这正是 3.5 节"粒度权衡"的具体形态；
3. **`ceil_ue8m0` 的行为**与 `tests/generators.py` 一致：只升不降、只在 2 的幂上取值。第 3 项 `ceil_ue8m0(50) = 64` 说明向上取整最多浪费一倍头部空间。

### 步骤 3：手算 SF 存储开销（约 15 分钟）

对 `M=4096, N=7168, K=16384`（SF 字节数 = SF 个数 × 1 字节，打包不改变总字节数）：

1. 算出 `(1, 1, 32)` 粒度下 A 与 B 的 SF 总字节数（答案：`(4096 + 7168) × 512 = 5.5 MiB`）；
2. 分别除以 FP8、FP4 的数据字节数，得到开销百分比（答案：3.1%、6.25%）；
3. 换成 `(32, 32, 32)` 粒度重算（答案：SF 仅 176 KiB，开销降到 0.1%/0.2%）；
4. 思考：既然块粒度省 32 倍 SF 存储，为什么默认还是每行 `(1, 1, 32)`？（结合步骤 2 中 E2M1 的离群值实验回答。）

### 步骤 4：源码导读打卡（约 35 分钟）

按第四节顺序依次打开 6 个站点文件，找到列出的行号并通读相应片段，每站用一句话记下它的职责。重点验证四处"数值契约"的源码实体：

- `tests/generators.py` 第 80～81 行的 `sf_div = 448.0 / 6.0`；
- `csrc/apis/layout.hpp` 第 55 行的 `gran_k == 32` 断言与第 36 行的返回形状；
- `deep_gemm/include/deep_gemm/transform_sf.hpp` 第 19 行的尾数检查、第 243 行的右移 23、第 276～278 行的两字节拼装；
- `fp8_gemm.hpp` 第 87 行与 `bf16_gemm.hpp` 第 64 行的 `l0c_ptr<float>`。

无硬件时的替代任务就是本步骤本身——纯静态阅读。

（可选，有 Ascend 环境时）在仓库根目录执行：

```bash
./develop.sh
python tests/test_fp8_gemm.py normal --skip-prof
python tests/test_fp4_gemm.py normal --skip-prof
```

观察每个用例打印的 `OK`（正确性通过与 3.4 节的精确参考比对，阈值为余弦相似度 1e-5）；再去掉 `--skip-prof` 跑一个大 shape，对照 README 性能表感受 861 vs 1701 TFLOPS 的算力阶梯。

## 六、当天产出

学习计划要求今天的产出是**一张 BF16、FP8、FP4 的优缺点对比表**。以下参考答案建议先自己填一遍再对照：

| 维度 | BF16 | FP8 (E4M3) | FP4 (E2M1) |
| :--- | :--- | :--- | :--- |
| 位宽 / 存储 | 16 位（2 字节/元素） | 8 位（1 字节/元素） | 4 位（两值打包 1 字节，`kPackedFP4`） |
| 位布局 | 1+8+7 | 1+4+3 | 1+2+1 |
| 动态范围 | 与 FP32 相同（约 2^-126～2^128） | 2^-9 ～ 448（跨度约 2×10^5） | 0.5 ～ 6（跨度 12 倍） |
| 尾数位 / 舍入误差上限 | 7 位 / 2^-8 ≈ 0.39% | 3 位 / 2^-4 = 6.25% | 1 位 / 2^-2 = 25% |
| 是否需要 SF | 否（`sfa_ptr/sfb_ptr = nullptr`） | 是（默认每行 1×32） | 是，且比 FP8 更依赖细粒度 |
| SF 存储开销 | — | 约 3.1%（1×32 粒度） | 约 6.25%（1×32 粒度） |
| 数据量（相对 BF16） | 100% | 50% | 25% |
| 实测算力（README，同 shape） | 431 TFLOPS | 861 TFLOPS | 1701 TFLOPS |
| 乘积在 FP32 中是否精确 | 是（16 位 ≤ 24 位） | 是（8 位 ≤ 24 位） | 是（4 位 ≤ 24 位） |
| 优点 | 范围宽、无需量化、接口最简单、可作输出类型 | 范围与精度折中好；训练推理通吃；对粗粒度不敏感 | 带宽和算力都翻倍；极限推理首选 |
| 缺点 | 带宽与算力最贵（是 FP4 的 1/4 算力） | 需要量化管线与 SF 布局转换；输出仍需 BF16/FP32 | 仅 16 个码点；必须细粒度 SF；离群值敏感；布局约束多（打包维须为偶数等） |
| 仓库中的输出角色 | `d` 的默认 dtype（与 FP32 二选一） | 输入为主；epilogue 可再量化回 FP8（`QuantizeToFP8`） | 仅作输入（B 侧可 AIV 反量化后当 FP8 用） |

配套"选型判断"小表：

| 场景 | 推荐 | 理由 |
| :--- | :--- | :--- |
| 输出、累加入口、精度敏感层 | BF16 或 FP32 | L0C 之后的一切（bias、alpha、写回）都在 BF16/FP32 世界里 |
| 大多数训练/推理权重与激活 | FP8 | 算力翻倍且误差 6.25% 可控，4 位指数对离群值宽容 |
| 极限带宽/算力推理 | FP4 | 再翻倍的算力，代价是必须接受 25% 级量化误差与更细的 SF 粒度 |

自查清单：

- 是否写清了每种格式的"范围、精度、带宽、算力"四笔账？
- 是否注明 FP4 的"两值一字节"存储与 `kPackedFP4`？
- 是否写出了"乘积在 FP32 精确、所以累加用 FP32 就够"这条主线？
- SF 开销那两列（3.1%/6.25%）自己手算过吗（步骤 3）？

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. E4M3 和 BF16 都是"8 位级"的类型（1 字节 vs 2 字节），为什么 E4M3 的动态范围比 BF16 小、相对精度也比 BF16 差？各自少了哪几位？
2. 写出 UE8M0 的含义，并说明 `ceil_to_ue8m0` 为什么"只升不降"。`amax = 6.5` 时 E4M3 的 SF 是多少？头部空间浪费了多少倍？
3. 仓库的默认量化粒度是多少？`gran_k` 被断言为 32 的依据是什么（提示：`common.hpp` 的 `get_frac_k` 注释）？
4. 打包后的 SF 张量形状是什么？一个 int16 覆盖多少个 K 元素？SF 在 GM 中是什么 major？
5. 用 BF16 从 0 开始累加 4096 个 1.0，结果是多少？用 FP32 呢？这个实验说明 GEMM kernel 的哪条设计原则？
6. 为什么说"低精度乘积在 FP32 中精确"？E4M3×E4M3、E2M1×E2M1、BF16×BF16 的乘积尾数各至多多少位？由此推出整条流水线里唯一的舍入发生在哪？
7. FP8×FP4 GEMM 的算力为什么与 FP8×FP8 相同（861 TFLOPS）？FP4 那一路数据在进入 cube 单元前发生了什么？依据仓库哪个文件的注释？
8. （思考题）如果把 `quantize_with_major` 的 `sf_div` 从 448 改成 448×2，量化结果会怎么变？把它从 448 改成 224 呢？

**参考答案：**

1. E4M3 比 BF16 少 4 位指数、4 位尾数：指数 4 位（bias 7）→ 正规范围只有 2^-9（次正规）～448；尾数 3 位 → 相邻值相对间距 12.5%。BF16 完整保留 FP32 的 8 位指数，故范围相同，尾数 7 位 → 间距 0.78%。
2. UE8M0 是"只有 8 位指数、无符号无尾数"的缩放因子格式，数值恒为 2 的幂（`fp8_e8m0_t`）。"只升不降"保证 `amax / sf ≤ 448` 恒成立，量化值永不越界饱和（`ceil_to_ue8m0` 定义在 `tests/generators.py` 第 27～31 行）。`6.5 / 448 = 0.0145`，向上取 2 的幂得 `sf = 2^-6 = 0.015625`；实际 amax 除以 sf 为 416，距 448 的头部空间约 7%，加上取整本身最多 2 倍浪费（如 `amax/448 = 0.51` 时 sf = 1.0）。
3. 默认 `(1, 1, 32)`（`csrc/apis/layout.hpp` 第 122～123 行、`tests/generators.py` 第 263 行）。依据：L1→L0 以 32 字节为一个 C0 行搬运（`common.hpp` 第 77 行注释 "32 bytes per C0"），FP8 一行恰 32 个元素 = 一个 UE8M0 的管辖范围，FP4 一行 64 个元素 = 一对打包 SF，`gran_k=32` 让 SF 管辖边界与硬件搬运边界对齐。
4. `[.., MN, ceil(K / gran_k / 2)]`（`csrc/apis/layout.hpp` 第 36 行），dtype 为 short（int16）；一个 int16 装两个 UE8M0，覆盖 64 个 K 元素（`MX_SF_DIVISOR = 64`，`common.hpp` 第 65 行）；SF 总是 MN-major（README 第 56 行、`gemm_config.hpp` 第 38 行注释）。
5. BF16 卡在 256（区间 `[256,512)` 步长 2，`256+1` 恰为中点、舍偶回到 256），FP32 精确得 4096。说明：GEMM 的 K 维累加必须用远宽于操作数的累加器——仓库所有 kernel 的 L0C 都是 `l0c_ptr<float>`（`bf16_gemm.hpp` 第 64 行、`fp8_gemm.hpp` 第 87 行）。
6. n 位尾数（含隐含 1）的乘积尾数至多 2n 位：E4M3 4 位 → 8 位；E2M1 2 位 → 4 位；BF16 8 位 → 16 位，均 ≤ FP32 的 24 位；UE8M0 只改指数不舍入。因此量化、缩放、乘法全程精确，**唯一舍入发生在 FP32 加法本身**，每步误差 ≤ 2^-24，16384 步累加仍远小于量化误差（6.25%/25%），设计才成立。
7. 因为 FP4 的 B 在进入 cube 前先被反量化成 FP8：`csrc/jit_kernels/fp8_dequant_gemm.hpp` 第 12～18 行注释说明 B 在 AIV 上 "build LUT → vf_deq_nd2nz → copy_ub_to_l1" 后走与 FP8 相同的 `mad_mx` cube 路径，所以算力同 FP8×FP8（861 TFLOPS），收益是数据量（132 MiB < 176 MiB）。
8. `sf_div` 翻倍 → SF 大一档（2 的幂），所有量化值缩到原来一半的范围，头部空间更宽但**相对精度不变**（尾数位数没变），只是白浪费一倍范围；`sf_div` 减半 → `amax/sf` 可能超过 448，量化会**饱和**到 ±448，大值被削平——这正是 `sf_div` 必须等于类型最大幅值的原因。

## 附录 A：本日概念速查

| 概念 | 一句话定义 | 仓库落点 |
| :--- | :--- | :--- |
| E4M3 (FP8) | 1+4+3 位浮点，最大 448，舍入误差上限 6.25% | `torch.float8_e4m3fn`、`float8_e4m3_t` |
| E2M1 (FP4) | 1+2+1 位浮点，16 个码点，最大 6.0 | `torch.float4_e2m1fn_x2`、`_FP4_VALUES` |
| 打包 FP4 | 两个 E2M1 nibble 存一个 int8 | `kPackedFP4`（csrc/utils/dtype.hpp:9） |
| UE8M0 | 8 位纯指数缩放因子，恒为 2 的幂 | `fp8_e8m0_t`、`ceil_to_ue8m0` |
| Scaling Factor | 每组一个的缩放因子，撑起小类型的范围 | `quantize_with_major` |
| recipe / 粒度 | `(gran_m, gran_n, gran_k)`，默认 (1,1,32) | `get_recipe`、`GemmTestDesc.recipe` |
| gran_k=32 | K 向每 32 元素一个 SF，与 32 字节 C0 行对齐 | `check_sf_layout`、`get_frac_k` |
| ue8m0x2 打包 | 相邻两个 K 的 SF 拼一个 int16，覆盖 64 元素 | `MX_SF_DIVISOR=64`、transform_sf.hpp:276-278 |
| MN-major SF | kernel 要求的 SF 布局，由 transform kernel 保证 | `transform_sf_into_required_layout` |
| FP32 累加 | L0C 恒为 `l0c_ptr<float>`，乘积在 FP32 中精确 | bf16_gemm.hpp:64、fp8_gemm.hpp:87 |
| mad_mx | UE8M0 缩放融合进矩阵乘加的指令路径 | `asc_mmad_mx`（ascend/copy.hpp:374-376） |
| L0C→BF16 转换 | 输出时才降精度：RNE 或随机舍入 | `asc_float2bfloat16_rn`、`StochasticRoundToBF16` |

## 附录 B：遗留问题记录模板

```text
# 第 4 天遗留问题
1. （例如）asc_mmad_mx 内部如何逐元素应用 e8m0 缩放、乘加宽度是多少 —— 计划第 24 天 MAD 专题
2. （例如）transform_sf kernel 的多级 UB 流水与分组清零（prepare_grouped_input）没细读 —— 计划第 33 天
3. （例如）StochasticRoundToBF16 的 quartet 哈希为何能消除统计偏差 —— 计划第 30 天测试比对专题
4. ……
```
