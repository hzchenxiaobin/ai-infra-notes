# 第 1 周 · 第 1 天：理解 GEMM 基本问题

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 1 周"GEMM 与 Ascend 基础"的第 1 天教程，可独立阅读，不依赖前序内容。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 M、N、K 在 `D = A × B` 中的含义，并根据 A、B 的形状推断 D 的形状；
2. 解释计算量为什么是 `2 × M × N × K`，并计算给定 shape 的输入输出数据量与算术强度；
3. 不借助工具，手算一个 `4 × 3` 与 `3 × 2` 矩阵的乘法；
4. 从"容量"和"数据复用"两个角度，说明为什么大矩阵不能一次性放进片上内存——这是后续学习 Tiling（第 3 天）和 Ascend 存储层次（第 5 天）的动机。

## 二、预备知识

- **线性代数初步**：向量内积（点积）的定义。矩阵乘法即使忘了也没关系，3.2 节会从内积重新讲起。
- **Python 基础（可选）**：实践步骤中的验证代码只用列表和三重循环，不需要第三方库；另有一段可选的 PyTorch 代码，在 CPU 上即可运行。
- **不需要 Ascend 硬件**：今天全部内容（手算、公式、源码阅读）都不依赖 NPU 环境。

背景一句话：本系列教程围绕 **DeepGEMM-Ascend** 仓库展开。它是 DeepSeek 开源的 GEMM 库 DeepGEMM 在华为昇腾平台上的移植（见仓库 `README.md`），支持 BF16、FP8、FP4 GEMM、MQA logits 和 MegaMoE 等算子。今天我们只关心"GEMM 这个计算问题本身长什么样"，完全不涉及硬件指令和优化细节。

## 三、详细讲解

### 3.1 GEMM 是什么

GEMM = GEneral Matrix Multiply（通用矩阵乘法），即：

```text
D = A × B (+ C)
```

计算矩阵乘积 A×B，可选地再加上一个与 D 同形状的矩阵 C（累加或偏置）。它是深度学习中计算量最大的算子之一：Transformer 的线性层（QKV 投影、FFN）、注意力中的批量内积、MoE 的 expert 计算，最终都会落到一次或多次 GEMM 上。DeepGEMM-Ascend 的核心目标就是把各种形态的 GEMM 在昇腾硬件上做到接近极限性能——仓库 README 的 Performance 一节给出 BF16 GEMM 达到硬件极限 99.8% 利用率的实测数据。

### 3.2 M、N、K 与维度关系

先回顾内积：两个**长度相同**的向量对应元素相乘再求和，得到一个标量。例如 `[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32`。

矩阵乘法就是"行点列"：D 的第 i 行第 j 列元素，等于 A 的第 i 行与 B 的第 j 列的内积。能这样做的前提是 **A 的列数等于 B 的行数**，这个公共长度记作 K，称为收缩维（contraction dimension）：

```text
              K                          N
        ┌───────────┐              ┌───────────┐
        │           │              │           │
     M  │     A     │      ×    K  │     B     │     =     D
        │           │              │           │          [M × N]
        └───────────┘              └───────────┘
```

| 矩阵 | 形状 | 含义 |
| :---: | :---: | :--- |
| A | `[M, K]` | 左矩阵：M 个长度为 K 的行 |
| B | `[K, N]` | 右矩阵：N 个长度为 K 的列 |
| D | `[M, N]` | 输出：M×N 个内积结果 |

逐元素写作：

```text
D[i][j] = Σ (p = 0 … K-1)  A[i][p] × B[p][j]
```

以 LLM 线性层 `y = x·W` 为例：M 是 token 数，K 是输入维度（如隐藏维 7168），N 是输出维度。DeepSeek 系列模型的典型形状（7168、2112、16384 等）在仓库测试与性能表中随处可见（README 说明这些 shape 覆盖 DeepSeek 系列的推理和训练负载）。

**本仓库的约定**（先知道即可，细节留到第 2 天"矩阵布局"）：仓库 `csrc/apis/gemm.hpp` 第 45～59 行的注释给出了各 GEMM 类型的形状总表，其中 Normal 类型为 `A [M, K]`、`B [N, K]`、`CD [M, N]`，即 B 以 `[N, K]` 的逻辑形状传入，实际计算 `D = A @ B^T`（`nt` 系列接口，见同文件第 413 行起的注释 "Dense bfloat16 GEMM: D = A @ B^T (+ C)"）；`nn` 系列接口则接受 `[K, N]` 的 B（README："The `bf16_gemm_nn` and `fp8_fp4_gemm_nn` interfaces take `A[M, K]` and `B[K, N]`"）。两种视角本质相同：**都是沿长度为 K 的维度做内积收缩**。本教程统一使用数学约定（B 为 `[K, N]`）。

特例：M=1 时 GEMM 退化为向量×矩阵（GEMV）。仓库测试 `tests/generators.py` 第 561 行 `m_fwd_list = [1, 128, 4096]` 就包含 M=1 的解码场景。

### 3.3 计算量：为什么是 2 × M × N × K

每个输出元素 `D[i][j]` 需要 K 次乘法 + (K-1) 次加法。D 共有 M×N 个元素：

- 乘法次数 = M·N·K
- 加法次数 = M·N·(K-1) ≈ M·N·K（K 大时差 1 可忽略）

总计算量习惯上记为（一次乘法和一次加法各计一次浮点操作）：

```text
FLOPs = 2 × M × N × K
```

顺带把下表要用的单位说清楚。**FLOP**（FLoating-point OPeration，浮点运算）是"计算量"的计数单位：做一次浮点加法或一次浮点乘法都记为 1 FLOP；上文写作 `FLOPs`（小写 s）时只是复数，指浮点运算总次数，与快慢无关。数值大了按国际单位制词头分级：

| 单位 | 词头与数量级 | 本文档中的例子 |
| :---: | :--- | :--- |
| GFLOP | giga，10^9 | 2×128×7168×16384 ≈ 30.06 GFLOP |
| TFLOP | tera，10^12 | 2×4096×7168×16384 ≈ 0.962 TFLOP |

需要特别区分**差一个 S 的两种写法**，下表会同时出现：

- **TFLOP**（不带 S）：计算量单位。1 TFLOP = 10^12 次浮点运算。"这次 GEMM 要做 0.962 TFLOP"描述的是工作量，是任务自身的属性，硬件跑得快慢都改变不了它。
- **TFLOPS**（TeraFLOPS，结尾的 S = Per Second）：性能（速度）单位。1 TFLOPS = 每秒完成 10^12 次浮点运算，衡量硬件有多快。
- 两者的关系就是"路程 ÷ 时间 = 速度"：**计算量（TFLOP）÷ 耗时（s）= 性能（TFLOPS）**。这正是下表"2MNK ÷ 延迟"一列在做的事；第 8 周性能分析（第 50 天）会反复使用这一对单位。

用仓库自己的实测数据验证这个公式（数据来自 `README.md` Performance 一节的表格）：

| 类型 | M | N | K | 2MNK | 实测延迟 | 2MNK ÷ 延迟 | README 报告值 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BF16×BF16 | 4096 | 7168 | 16384 | ≈ 0.962 TFLOP | 2229.9 us | ≈ 431 TFLOPS | 431 TFLOPS |
| FP8×FP8 | 4096 | 7168 | 16384 | ≈ 0.962 TFLOP | 1117.6 us | ≈ 861 TFLOPS | 861 TFLOPS |
| FP8×FP8 | 128 | 7168 | 16384 | ≈ 30.06 GFLOP | 54.8 us | ≈ 549 TFLOPS | 549 TFLOPS |

"2MNK ÷ 延迟"一列的结果与 README 报告的计算性能完全一致——说明仓库的性能统计用的正是 `2MNK ÷ 时延` 这一口径。这给了一个可复算的交叉验证：只要相信时延数字，就能用 2MNK 复现表中的 TFLOPS。

### 3.4 数据量与算术强度

设 A、B 每个元素占 s_a、s_b 字节，D 占 s_d 字节（BF16 = 2，FP8 = 1，FP32 = 4）：

```text
读入：  bytes(A) = M·K·s_a          bytes(B) = N·K·s_b
写出：  bytes(D) = M·N·s_d
总量 ≈ M·K·s_a + N·K·s_b + M·N·s_d
```

（这是"逻辑数据量"，即每个元素恰好读写一次；实际还受缓存命中、缩放因子等因素影响，留到第 8 周性能分析再展开。）

**算术强度（arithmetic intensity）= 计算量 ÷ 数据量**，单位 FLOP/Byte，含义是"平均每读入一个字节数据，能支撑多少次浮点运算"。

以 README 中的旗舰 shape `(4096, 7168, 16384)`、BF16 为例：

| 项 | 计算 | 结果 |
| :--- | :--- | :--- |
| 计算量 | 2×4096×7168×16384 | ≈ 9.62×10^11 FLOP |
| A 数据量 | 4096×16384×2 B | 128 MiB |
| B 数据量 | 7168×16384×2 B | 224 MiB |
| D 数据量 | 4096×7168×2 B | 56 MiB |
| 合计 | — | 408 MiB（427,819,008 B） |
| 算术强度 | 9.62×10^11 ÷ 4.28×10^8 | ≈ 2249 FLOP/B |

一个可记的规律：边长为 n 的同 dtype 方阵（M=N=K=n），算术强度 ≈ 2n/(3s)，BF16（s=2）时即 **n/3**——n=4096 时约 1365 FLOP/B。矩阵越大，每字节数据能支撑的计算越多，这就是"大矩阵算力受限、小矩阵带宽受限"的根源。README 的 FP8 推理表恰好展示了这一对比（同为 N=7168、K=16384，FP8 输入、BF16 输出，忽略缩放因子）：

| M | 算术强度（估算） | 实测计算性能 | 实测内存带宽 | 状态 |
| :---: | :---: | :---: | :---: | :--- |
| 128 | ≈ 248 FLOP/B | 549 TFLOPS | 2456 GB/s | 偏带宽受限：数据搬运频繁 |
| 4096 | ≈ 3955 FLOP/B | 861 TFLOPS（极限的 99.5%） | 186 GB/s | 计算受限：数据被充分复用 |

结论：**GEMM 的性能关键不只是"算得快"，更是"让每字节数据被尽可能多的计算复用"**。这是理解一切 GEMM 优化（包括本仓库的 Tiling 与多级流水）的钥匙。

### 3.5 手算示例：4×3 乘 3×2

设：

```text
      ┌           ┐          ┌       ┐
      │  1  2  3  │          │  1  4 │
A  =  │  4  5  6  │     B  = │  2  5 │
      │  7  8  9  │          │  3  6 │
      │ 10 11 12  │          └       ┘
      └           ┘
```

A 的列数（3）等于 B 的行数（3），可以相乘，D 是 4×2。逐元素"行点列"：

- `D[0][0] = 1×1 + 2×2 + 3×3 = 14`，`D[0][1] = 1×4 + 2×5 + 3×6 = 32`
- `D[1][0] = 4×1 + 5×2 + 6×3 = 32`，`D[1][1] = 4×4 + 5×5 + 6×6 = 77`
- `D[2][0] = 7×1 + 8×2 + 9×3 = 50`，`D[2][1] = 7×4 + 8×5 + 9×6 = 122`
- `D[3][0] = 10×1 + 11×2 + 12×3 = 68`，`D[3][1] = 10×4 + 11×5 + 12×6 = 167`

```text
      ┌        ┐
      │  14  32 │
D  =  │  32  77 │
      │  50 122 │
      │  68 167 │
      └        ┘
```

操作数统计：8 个输出元素，每个 3 次乘法 + 2 次加法，共 24 次乘 + 16 次加；按 2MNK 口径 = 2×4×2×3 = 48。数据量统计（假如是 BF16）：A 12 个 + B 6 个 + D 8 个 = 26 个元素 × 2 B = 52 B——小到任何存储都放得下。真实模型的 GEMM 则完全是另一个量级（见 3.6）。

### 3.6 为什么大矩阵不能一次性放进片上内存

"片上内存"指计算单元旁边的私有高速存储。昇腾的存储层次第 5 天才展开，但容量的数量级今天就能算清楚。仓库把目标硬件（Ascend 950DT）的片上容量写成了编译期常量，位于 `deep_gemm/include/deep_gemm/common.hpp` 第 50～58 行：

```cpp
// Ascend 950DT core topology
inline constexpr uint32_t kNumAICores = 32;
inline constexpr uint32_t L1SizeBytes = 512 * 1024;   // L1 per core
inline constexpr uint32_t L0ASizeBytes = 64 * 1024;   // L0A
inline constexpr uint32_t L0BSizeBytes = 64 * 1024;   // L0B
inline constexpr uint32_t L0CSizeBytes = 256 * 1024;  // L0C (fp32 accumulator)
inline constexpr uint32_t UBSizeBytes = 256 * 1024;   // UB
```

把 3.4 节的旗舰 shape（BF16，A+B+D = 408 MiB）与这些容量放在一起：

| 存储 | 容量 | 与 408 MiB 的差距 |
| :--- | :--- | :--- |
| L1（每核） | 512 KiB | 单是 A（128 MiB）就是它的 256 倍 |
| L0A + L0B（每核） | 128 KiB | 更小，仅供矩阵乘加单元直接消费 |
| 全部 32 核的 L1 合计 | 16 MiB | 差约 26 倍，且**每核只能用自己的 L1** |
| 全部片上缓冲（L1+L0+UB，32 核） | ≈ 36 MiB | 仍差一个数量级 |

**理由一：容量根本不够。** 即使把所有核的私有存储全部加起来（这本来就不现实——它们彼此独立），也放不下一个中等规模 GEMM 的输入输出。

**理由二：片上缓冲还要"一分为多"。** 为了让数据搬运和计算重叠（流水线），同一段 L1/L0 要切成多个 stage 轮换使用，单块可用容量进一步缩小。kernel 模板里有直接证据——`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 第 66～69 行：

```cpp
static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes,  "L1 overflow");
static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes, "L0A overflow");
static_assert(l0b.offset(kNumL0Stages) <= L0BSizeBytes, "L0B overflow");
static_assert(l0c.offset(num_mad_cd_stages) <= L0CSizeBytes, "L0C overflow");
```

配置一旦超出容量，编译期直接报错。也就是说，**kernel 每次只能在片上持有矩阵的一个小块（Tile）**。

**理由三（反方向思考）：就算放得下，也不该这么做。** 3.4 节指出大 GEMM 的算术强度高达数千 FLOP/B，性能关键是让每份数据被大量复用。硬件的天然分工是：大而慢的 GM（Global Memory，设备主存）存放完整矩阵，小而快的片上存储反复装载小块数据、配合计算单元流式工作。正确的策略不是"把所有数据塞进片上"，而是**分块（Tiling）+ 分层搬运**：

```text
GM（A、B、D 完整存放）
  │  按 BLOCK_M/N/K 切成小块，分批搬运、充分复用
  ▼
L1（512 KiB/核）──▶ L0A/L0B ──▶ MAD 乘加 ──▶ L0C ──▶ … ──▶ 写回 GM
```

这正是第 3 天（Tiling 与数据复用）和第 5 天（Ascend 存储层次与流水线）的主题。今天只需建立这个核心印象：

> **GEMM kernel = 用小的片上缓冲，通过分块和复用，流式地完成一个放不下的大乘法。**

## 四、源码导读

今天的导读主题是"**M/N/K 在仓库里如何被定义、检查和传递**"，全程静态阅读、无需硬件。建议对照仓库依次打开以下文件（路径相对 `DeepGEMM-Ascend/`）。

### 站点 1：M/N/K 从哪里来 —— `csrc/apis/gemm.hpp`

关键函数：`get_gemm_desc()`（第 60 行起）。所有 GEMM 接口（`bf16_gemm_nt/nn/tn/tt`、`fp8_fp4_gemm_*` 等）最终都经过它。第 45～59 行的注释给出形状总表：

```text
 *                A                 B                CD
 *
 * Normal:       [M, K]            [N, K]            [M, N]
 * MGrouped:     [M, K]         [G, N, K]            [M, N]
 * KGrouped(*):  [M, K]            [N, K]         [G, M, N]
 * Batched:   [B, M, K]         [B, N, K]         [B, M, N]
```

Normal 分支（第 82～84 行）从 a、b、d 三个张量交叉解出 M/N/K：

```cpp
case GemmType::Normal:
    std::tie(m, n, k) = get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d);
    break;
```

解读：A 标注为 `"mk"`（形状 `[M, K]`）、B 标注为 `"nk"`、D 标注为 `"mn"`；同一个字母在多个张量中出现时必须相等，否则报错。这就是"A 的列数等于 B 的行数"在代码里的落点。

### 站点 2：形状一致性如何检查 —— `csrc/utils/layout.hpp`

关键函数：`get_shape_by_spec()`（第 140 行起）。其内部的 `get_one_dim`（第 169 行起）按维度名扫描每个张量的每一轴，一旦同一维度名出现冲突就 `DJ_PANIC`（第 195～203 行）——例如 A 是 `[M, K]` 而 D 的第二维与 B 的 N 不一致时直接终止。

同文件还有 `gemm_early_return()`（第 54 行），处理退化 shape：

- `m == 0 or n == 0`：没有输出元素，直接返回，不启动 kernel；
- `k == 0`：收缩维为空，没有乘法可做，把 D 清零（或拷贝 C）后返回（第 74～80 行）。

这从代码层面印证了 K 的语义：**K 是唯一"被消耗"的维度，M/N 是输出保留的维度**。

### 站点 3：M/N/K 存进描述符 —— `csrc/jit_kernels/gemm_config.hpp`

关键结构：`GemmDesc`（第 39 行起，字段 `m, n, k` 在第 46 行），由 `get_gemm_desc()` 在 `csrc/apis/gemm.hpp` 第 164～196 行填充（`.m = m, .n = n, .k = k` 见第 173～175 行）。其注释写得很清楚：

> Problem descriptor: what to compute (dimensions, dtypes, pointers, layouts).

紧随其后的 `GemmConfig`（第 98 行起）则是另一回事：

> Execution configuration: how to compute (tiling, pipeline, launch).

它装的是 `block_m/block_n/block_k`（第 100 行）等**怎么切块**的信息——今天先"认识一下"，第 12 天系统学习。记住一句话即可：**GemmDesc 描述"算什么"，GemmConfig 描述"怎么算"**。

### 站点 4：kernel 眼中的 shape —— `deep_gemm/include/deep_gemm/bf16_gemm.hpp`

BF16 kernel 的模板入口 `bf16_gemm_impl`（模板参数第 9～26 行，函数签名第 27 行起）。模板参数里同时出现两套维度：

```cpp
uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,   // 编译期 shape
uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,   // 切块大小
```

函数还接受运行时 shape（第 31 行 `shape_m_runtime` 等），并在第 47～49 行做选择——编译期为 0 时用运行时值：

```cpp
uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime;
```

第 66～69 行的四个 `static_assert`（3.6 节已引用）保证多级缓冲不超出 `common.hpp` 的容量常量。今天不必读 kernel 实现，只要能在文件里定位这几行即可。

### 站点 5：真实 shape 长什么样 —— `tests/generators.py` 与 `README.md`

- `tests/generators.py` 第 250 行的 `GemmTestDesc` 用一个 `shape_mnk` 字段描述测试用例（`m/n/k` 属性在第 271～276 行）。
- `_enumerate_normal()`（第 544 行起）列出真实测试 shape：`nk_list` 含 (7168, 16384)、(24576, 1536)、(2112, 7168) 等，`m_fwd_list = [1, 128, 4096]`（第 559～561 行）。
- `README.md` Performance 一节的 Dense GEMM / FP8 推理表把这些 shape 与实测时延、TFLOPS、带宽放在一起，正是 3.3、3.4 节所有数字的出处。

**阅读检查点**（能回答再进入实践）：

1. 给定 a、b、d 三个张量，`get_gemm_desc` 如何推出 m、n、k？形状矛盾时会发生什么？
2. `k == 0` 时程序行为是什么？为什么合理？
3. `GemmDesc` 和 `GemmConfig` 分别存放什么信息？

## 五、实践步骤

以下步骤按顺序执行，全部**不需要 Ascend 硬件**；总时长约 2 小时。

### 步骤 1：纸笔手算（约 25 分钟）

不看 3.5 节的例题，自己计算下面两个矩阵的乘积（先写出 D 的形状，再逐元素计算）：

```text
A = [ 2 0 1 ]        B = [ 1 2 ]
    [ 1 3 0 ]            [ 0 1 ]
    [ 0 1 2 ]            [ 2 0 ]
    [ 3 2 1 ]
```

完成后：数一数做了多少次乘法和加法，与 `2×M×N×K = 48` 对比。参考答案在附录 A。

### 步骤 2：用 Python 验证手算（约 15 分钟）

在你自己的练习目录新建 `gemm_basic.py`（不要改动仓库文件），写入：

```python
def gemm(a, b):
    m, k, n = len(a), len(b), len(b[0])
    assert len(a[0]) == k, "A 的列数必须等于 B 的行数"
    d = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0
            for p in range(k):
                s += a[i][p] * b[p][j]
            d[i][j] = s
    return d

a = [[2, 0, 1], [1, 3, 0], [0, 1, 2], [3, 2, 1]]
b = [[1, 2], [0, 1], [2, 0]]

for row in gemm(a, b):
    print(row)
```

运行 `python gemm_basic.py`，与手算结果对照。这段三重循环就是 3.2 节定义的逐字翻译：外两层遍历 M×N 个输出元素，最内层做长度为 K 的内积。

可选：若本机装有 PyTorch（CPU 即可），再用 `torch.matmul` 验证一次并观察形状：

```python
import torch
a = torch.tensor([[2., 0, 1], [1, 3, 0], [0, 1, 2], [3, 2, 1]])
b = torch.tensor([[1., 2], [0, 1], [2, 0]])
print(a.shape, b.shape, (a @ b).shape)
print(a @ b)
```

### 步骤 3：数量级演算（约 25 分钟）

用计算器或 Python 填写下表（BF16，s = 2 字节），并与后表核对：

| shape (M, N, K) | 2MNK (FLOP) | bytes(A+B+D) | 算术强度 (FLOP/B) |
| :--- | :--- | :--- | :--- |
| (4096, 7168, 16384) | ? | ? | ? |
| (128, 2112, 7168) | ? | ? | ? |
| (4, 2, 3)（步骤 1 的小例） | 48 | 52 B | ≈ 0.92 |

参考值：

| shape (M, N, K) | 2MNK | bytes | 算术强度 |
| :--- | :--- | :--- | :--- |
| (4096, 7168, 16384) | ≈ 9.62×10^11 | 427,819,008 B（408 MiB） | ≈ 2249 |
| (128, 2112, 7168) | ≈ 3.88×10^9 | 32,653,312 B（约 31 MiB） | ≈ 119 |

再用 README 的实测时延复算 TFLOPS：0.962×10^12 ÷ 2229.9×10^-6 s ≈ 431 TFLOPS（BF16 表）；3.88×10^9 ÷ 15.0×10^-6 s ≈ 258 TFLOPS（FP8 表中 M=128、N=2112、K=7168 行，README 报告 258 TFLOPS）。

### 步骤 4：容量对比演算（约 15 分钟）

用 `common.hpp` 的常量（32 核；L1 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB、UB 256 KiB）计算并回答：

1. 单核 L1 能装下 `(4096, 16384)` 的 BF16 矩阵 A 吗？差多少倍？（答案：128 MiB ÷ 512 KiB = 256 倍）
2. 32 核 L1 总量是多少？够放下 A+B+D（408 MiB）吗？（答案：16 MiB，差约 26 倍，且每核私有）
3. 打开 `deep_gemm/include/deep_gemm/bf16_gemm.hpp`，找到第 66～69 行的四个 `static_assert`，用自己的话解释它们防的是什么。

### 步骤 5：源码导读打卡（约 40 分钟）

按第四节顺序依次打开 5 个站点文件，找到列出的行号并通读相应片段，每站用一句话记下它的职责。无硬件时的替代任务就是本步骤本身——纯静态阅读。

（可选，有 Ascend 环境时）在仓库根目录执行 `./develop.sh`，再运行 `python tests/test_bf16_gemm.py normal --skip-prof`，观察输出中滚过的 shape 列表（如 1×7168×16384、4096×7168×16384），并心算每个 shape 的 2MNK。没有环境完全不影响今天的学习。

## 六、当天产出

学习计划要求今天的产出是**一页 GEMM 基础笔记**。建议按下面的模板整理（Markdown 或纸笔均可）：

```text
# GEMM 基础笔记（第 1 天）

1. 定义与维度
   - D = A × B：A [M,K] × B [K,N] → D [M,N]；D[i][j] = Σ A[i][p]·B[p][j]
   - 仓库约定：B 传 [N,K]，算 D = A @ B^T（nt 接口）；nn 接口接受 [K,N]
   -（画一个 M/N/K 维度示意图）

2. 计算量与数据量
   - FLOPs = 2MNK；Bytes = MK·s_a + NK·s_b + MN·s_d；AI = FLOPs/Bytes
   - 我的例子：(自选 shape，写出三项计算过程)

3. 手算实例
   -（贴上步骤 1 的 A、B、D 与操作计数）

4. 为什么放不进片上内存
   - 容量：L1 512 KiB/核 vs A 128 MiB（256 倍）；32 核合计 16 MiB vs 408 MiB
   - 复用：大 GEMM 的 AI 上千 FLOP/B，关键在数据复用而非一次装下
   - 结论：Tiling + 分层搬运（→ 第 3、5 天）

5. 源码锚点
   - get_gemm_desc（csrc/apis/gemm.hpp:60）：从 a/b/d 解出 m/n/k
   - GemmDesc（csrc/jit_kernels/gemm_config.hpp:39）："算什么"
   - 容量常量（deep_gemm/include/deep_gemm/common.hpp:50-58）

6. 遗留问题
   -（记下今天没想明白的问题，留到第 6 天复盘）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. A 的形状是 `[512, 4096]`，B 的形状是 `[4096, 1024]`。写出 D 的形状、M/N/K 的值和 2MNK。
2. 为什么计算量是 2MNK 而不是 MNK？
3. 仓库 `bf16_gemm_nt` 接口的 a、b、d 逻辑形状分别是什么？它计算的数学表达式是什么？
4. `get_gemm_desc` 处理过程中如果 K 为 0，会发生什么？为什么这样设计是合理的？
5. 从两个角度解释：为什么大矩阵不能简单地一次性放进片上内存？
6. `GemmDesc` 与 `GemmConfig` 的区别是什么？
7. （思考题）M=128 与 M=4096 的两个 GEMM（N、K 相同、同 dtype），哪个更可能受内存带宽限制？为什么？

**参考答案：**

1. D 是 `[512, 1024]`；M=512、N=1024、K=4096；2MNK = 2×512×1024×4096 = 4,294,967,296 ≈ 4.29 GFLOP。
2. 每个输出元素需要 K 次乘法 + (K-1) 次加法，合计约 2K 次浮点操作，共 M×N 个输出，故 2MNK。
3. a: `[M, K]`、b: `[N, K]`、d: `[M, N]`；计算 `D = A @ B^T (+ C)`（见 `csrc/apis/gemm.hpp` 第 413 行起的注释）。
4. `gemm_early_return`（`csrc/utils/layout.hpp:54`）把 D 清零（或拷贝 C）并直接返回，不启动乘法 kernel。K 是收缩维，K=0 意味着没有任何内积要做，输出退化为零/偏置，启动 kernel 反而是浪费。
5. 容量角度：单核 L1 只有 512 KiB，一块 128 MiB 的 A 就超出 256 倍，32 核私有存储合计也远不够，还要切出多级流水 stage；复用角度：大 GEMM 算术强度高达数千 FLOP/B，性能靠数据复用，正确做法是分块搬运 + 流水复用，而不是一次装下。
6. GemmDesc 是"算什么"（形状、dtype、指针、major 等问题定义），GemmConfig 是"怎么算"（block/mad 切块、stage 数、L2 策略、launch 选项等执行配置）。
7. M=128 更可能带宽受限。M 越小，与 M 无关的 B 数据量（N·K·s）占比越大，算术强度越低（本例约 248 vs 3955 FLOP/B），每字节数据支撑的计算少，时间容易耗在搬运上；README 的 FP8 表中 M=128 行实测内存带宽 2456 GB/s、仅 549 TFLOPS，M=4096 行 186 GB/s、861 TFLOPS（极限的 99.5%），与该分析一致。

## 附录 A：步骤 1 参考答案

D 是 4×2（A 的列数 3 = B 的行数 3，D = 4×2）：

```text
D[0][0] = 2×1 + 0×0 + 1×2 = 4      D[0][1] = 2×2 + 0×1 + 1×0 = 4
D[1][0] = 1×1 + 3×0 + 0×2 = 1      D[1][1] = 1×2 + 3×1 + 0×0 = 5
D[2][0] = 0×1 + 1×0 + 2×2 = 4      D[2][1] = 0×2 + 1×1 + 2×0 = 1
D[3][0] = 3×1 + 2×0 + 1×2 = 5      D[3][1] = 3×2 + 2×1 + 1×0 = 8
```

```text
      ┌     ┐
      │ 4  4 │
D  =  │ 1  5 │
      │ 4  1 │
      │ 5  8 │
      └     ┘
```

乘法 8×3 = 24 次，加法 8×2 = 16 次；按 2MNK 口径 = 2×4×2×3 = 48，与习惯计数一致。
