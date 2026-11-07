# 第 1 周 · 第 2 天：理解矩阵布局

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 1 周"GEMM 与 Ascend 基础"的第 2 天教程，可独立阅读，不依赖前序内容。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 解释 Row-major（行优先）和 Column-major（列优先）的内存排列规则，写出任意元素 `(i, j)` 的线性偏移公式；
2. 解释 stride（步长）和 contiguous（连续）的含义，判断一个张量是否 contiguous，并理解 stride 不仅能表达"行优先/列优先"，还能表达切片、行间空洞等非紧凑存储；
3. 说明 `nt` / `nn` / `tn` / `tt` 后缀中 N 和 T 的含义，写出四种布局下 A、B 的逻辑形状与典型 stride，并解释它们为什么都计算同一个 `D = A × B`；
4. 区分"逻辑转置"（`transpose()` 视图，只改元数据、零拷贝）和"实际搬移数据"（`contiguous()` / 拷贝，真实移动全部元素），并在仓库源码中找到这两类操作对应的位置；
5. 理解仓库如何用 `Major::K` / `Major::MN` 加上 stride 来描述操作数布局（而不是只用形状），为第 3 天的 Tiling 学习打基础。

## 二、预备知识

- **第 1 天的内容**：M/N/K 的含义与 `D = A × B` 的维度关系（A `[M, K]`、B `[K, N]`、D `[M, N]`）。本教程 3.5 节会简要回顾。
- **PyTorch 基础（可选）**：会创建张量、查看 `shape` 即可。实践步骤中的验证脚本在 CPU 上即可运行；没有 PyTorch 时，可用纸笔完成等价的手工演算任务。
- **不需要 Ascend 硬件**：今天全部内容（概念、源码阅读、手工演算）都不依赖 NPU 环境。

背景一句话：GEMM 库的接口首先要回答一个问题——"你给我的矩阵，在内存里是怎么摆的？"同一个数学乘法，操作数摆放方式不同，kernel 的取数地址、对齐要求甚至最优切块策略都会不同。仓库 README 第 5 行的话可以佐证：DeepGEMM-Ascend 之所以存在，就是为了替使用者"隐藏 fractal 布局、对齐约束、地址计算等复杂性"。今天我们学习布局本身；仓库如何处理布局，放在源码导读一节。

## 三、详细讲解

### 3.1 内存是一维的：布局问题的由来

内存（或任何存储）本质上是一个**一维**的字节序列，地址从 0 开始递增。而矩阵是**二维**的逻辑对象，我们说 `A[i][j]` 时用的是"第 i 行第 j 列"这种二维坐标。于是必须有一个约定，把二维坐标 `(i, j)` 映射到一维偏移：

```text
布局 = 一条规则：  (行号 i, 列号 j)  ──映射──▶  线性偏移 offset(i, j)
```

这条规则就是**矩阵布局（layout）**。规则不同，同一组数值在内存里的排列顺序就不同，读写同一个 `A[i][j]` 的地址也不同。今天要学的全部概念——Row-major、Column-major、stride、contiguous、N/T——都是围绕这条映射规则展开的。

### 3.2 Row-major 与 Column-major

设 A 是 `[M, K]` 矩阵（M 行 K 列），每个元素占 s 字节（BF16 为 2，FP32 为 4）。

**Row-major（行优先，行主序）**：先存第 0 行的所有元素，再存第 1 行，依此类推。C/C++/PyTorch/绝大多数深度学习框架默认使用它。

```text
A = │ 10 11 12 │         Row-major 内存（一维）：
    │ 20 21 22 │         ┌───┬───┬───┬───┬───┬───┐
    │ 30 31 32 │         │10 │11 │12 │20 │21 │22 │ 30 31 32 40 41 42
    │ 40 41 42 │         └───┴───┴───┴───┴───┴───┘
                          └─ 第 0 行 ─┘└─ 第 1 行 ─┘

地址公式：  addr(i, j) = base + (i × K + j) × s
```

**Column-major（列优先，列主序）**：先存第 0 列的所有元素，再存第 1 列。Fortran、BLAS（如 OpenBLAS、cuBLAS 的部分接口）内部常用它。

```text
A = │ 10 11 12 │         Column-major 内存（一维）：
    │ 20 21 22 │         ┌───┬───┬───┬───┬───┬───┐
    │ 30 31 32 │         │10 │20 │30 │40 │11 │21 │ 31 41 12 22 32 42
    │ 40 41 42 │         └───┴───┴───┴───┴───┴───┘
                          └── 第 0 列 ──┘└ 第 1 列 ┘

地址公式：  addr(i, j) = base + (j × M + i) × s
```

一个关键观察：**同一块物理内存，换一套读取规则，看到的矩阵就变成了它的转置**。把上面 Row-major 的内存序列按 `[K, M]` 行优先的规则去读（每 4 个数一行），得到的是 `A^T`：

```text
Row-major 存储的 A [4, 3]:    10 11 12 20 21 22 30 31 32 40 41 42
按 [3, 4] 行优先读取：        │ 10 11 12 20 │   =  A^T
                              │ 21 22 30 31 │
                              │ 32 40 41 42 │
```

这就是"转置可以不搬数据"的物理基础——转置前后只是**解释同一块内存的方式**变了。这一点 3.4 和 3.7 节会反复用到。

### 3.3 stride 与 contiguous

Row/Column-major 只是特例。一般的描述工具是 **stride（步长）**：沿每个维度走一步，地址要跳过多少个**元素**（注意 PyTorch 的 stride 以元素为单位，不是字节）：

```text
addr(i, j) = base + (i × stride[0] + j × stride[1]) × s
```

| 存储方式 | 逻辑形状 | stride | 说明 |
| :--- | :---: | :---: | :--- |
| Row-major | `[M, K]` | `(K, 1)` | 行优先；沿列走一步跳 1 个元素 |
| Column-major | `[M, K]` | `(1, M)` | 列优先；沿行走一步跳 1 个元素 |
| 列切片 `x[:, :3]`（原矩阵 K=5） | `[M, 3]` | `(5, 1)` | 每行只取前 3 列，行间距仍是 5 |
| 带行间隙的存储 | `[M, K]` | `(K+pad, 1)` | 行与行之间有 pad 个元素空洞 |

可见 stride 是"形状"之外的独立信息：**形状说有多少个元素，stride 说这些元素散布在哪里**。

**contiguous（连续）**：PyTorch 中 `t.is_contiguous()` 为真，当且仅当张量的 stride 恰好等于"紧凑排列"的 stride——即按行优先顺序遍历各维时，元素在内存中一个紧挨一个、无重叠无空洞。对二维 `[M, K]` 而言就是 `stride == (K, 1)`。上表只有第一种是 contiguous；column-major `[M, K]`（stride `(1, M)`）不是——虽然它也是一种"整齐"的排列，只是方向不同。

三个容易混淆的点提前澄清：

1. **非 contiguous ≠ 布局错误**。column-major、切片、转置视图都非 contiguous，但都是完全合法、甚至常常更高效的数据组织。
2. **contiguous 是"行优先意义下的紧凑"**。说一个张量"是 column-major"，等价于说"它的转置视图是（行优先）contiguous 的"。
3. **stride 可以大于形状需要的值**（上表第 4 行）。仓库测试就有专门的 strided 场景：`tests/generators.py` 第 238～246 行的 `make_strided()` 把某些维度的 size 和 stride 同时加大，制造出行间空洞，用来验证 kernel 能用 stride 正确寻址（而不是假设数据紧凑）。

### 3.4 逻辑转置：`transpose()` 前后发生了什么

PyTorch 的 `transpose(dim0, dim1)` 是**纯元数据操作**：交换 `shape` 中两个维度的长度，同时交换 `stride` 中对应的步长，`data_ptr`（首元素地址）和底层 storage **完全不动**，时间复杂度 O(1)。用一个 4×3 张量实验（完整脚本见实践步骤 1）：

```python
import torch
a = torch.arange(12).reshape(4, 3).float()   # 值 0..11
at = a.transpose(0, 1)
```

| 张量 | shape | stride | is_contiguous | data_ptr |
| :--- | :---: | :---: | :---: | :---: |
| `a` | `(4, 3)` | `(3, 1)` | True | p |
| `at = a.transpose(0, 1)` | `(3, 4)` | `(1, 3)` | False | **p（相同）** |
| `atc = at.contiguous()` | `(3, 4)` | `(4, 1)` | True | q（**新分配**） |

逐条解读：

- `a` 是 row-major `[4, 3]`，`addr(i, j) = base + 3i + j`。
- `at` 与 `a` 共享同一块内存（`data_ptr` 相同），只是 stride 从 `(3, 1)` 变成 `(1, 3)`——恰好是 column-major `[3, 4]` 的 stride。验证：`a[1, 2]` 的偏移 = `1×3 + 2 = 5`；`at[2, 1]` 的偏移 = `2×1 + 1×3 = 5`。同一个地址、同一个值（实验中两者都是 5.0）。
- `at` 的 `is_contiguous()` 变为 False——按 `[3, 4]` 行优先的紧凑要求 stride 应为 `(4, 1)`，而它是 `(1, 3)`。
- `at.contiguous()` 才真正分配新内存并**搬移数据**（新地址 q），把值按 `[3, 4]` 行优先重新排布。此后 `at` 与 `atc` 的数值相同（`torch.equal(at, atc)` 为 True），但物理排列不同。

所以 "transpose" 有两种含义，务必区分：

| | 逻辑转置 | 实际搬移数据 |
| :--- | :--- | :--- |
| 典型操作 | `t.transpose(0, 1)` | `t.contiguous()`、`t.clone()`、按新布局重写 |
| 本质 | 改 shape/stride 元数据，重解释同一块内存 | 移动全部 M×K 个元素到新位置 |
| 代价 | O(1)，零内存流量 | 分配新存储 + 读写共 2×M×K×s 字节 |
| 结果张量 | 非 contiguous 视图 | contiguous（或目标布局） |

这与 3.2 节的观察呼应：**"转置"作为数学关系，不需要搬数据；只有当硬件或下游接口要求特定的物理排列时，才需要搬**。仓库源码对这一点有非常直接的注释，见 3.7 节和源码导读。

顺带一提，仓库测试里构造特定 major 数据的惯用法 `x.transpose(0, 1).contiguous().transpose(0, 1)`（`tests/generators.py` 第 64 行，BF16 分支）正是两件事的组合：中间的 `.contiguous()` 搬一次数据（把存储变成转置后的紧凑排列），两头的 `transpose` 是零拷贝视图——最终得到逻辑形状不变、但 stride 变为 `(1, M)` 的 column-major 张量。

### 3.5 GEMM 的四种布局：NT / NN / TN / TT

先回顾第 1 天的设定：数学上 `D = A × B`，A 是 `[M, K]`、B 是 `[K, N]`、D 是 `[M, N]`，沿长度为 K 的维度做内积收缩。

GEMM 库（BLAS、cuBLAS、DeepGEMM）的接口都会给每个操作数一个标记：**N（no transpose，非转置）或 T（transpose，转置）**，说明"调用者传入的张量，相对它的数学形状是否被转置过"。仓库 `csrc/apis/gemm.hpp` 第 413～419 行的接口注释给出了精确定义（以 BF16 为例，FP8/FP4 的 `fp8_fp4_gemm_*` 在第 199～216 行，约定相同）：

```text
// Dense bfloat16 GEMM: D = A @ B^T (+ C).
// The `nt` / `nn` / `tn` / `tt` binding suffix selects the logical shape of the two operands.
//     a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
//     b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
//     d: [M, N], bf16 or fp32.
```

规则拆开读（第一个字母管 A，第二个字母管 B）：

- **A 的字母**：`n` → 传入 `[M, K]`（数学形状原样）；`t` → 传入 `[K, M]`（数学形状的转置）。
- **B 的字母**：`n` → 传入 `[K, N]`（数学形状原样）；`t` → 传入 `[N, K]`（数学形状的转置）。

四种组合（下表的"数学语义"列描述每种绑定实际计算的乘积，全部等价于同一个 `D = A_math × B_math`）：

| 后缀 | a 传入形状 | b 传入形状 | 数学语义 | 典型 stride（每个传入张量自身为 row-major 紧凑存储时） |
| :---: | :---: | :---: | :---: | :--- |
| `nt` | `[M, K]` | `[N, K]` | `D = a × b^T` | a `(K, 1)`；b `(K, 1)` |
| `nn` | `[M, K]` | `[K, N]` | `D = a × b` | a `(K, 1)`；b `(N, 1)` |
| `tn` | `[K, M]` | `[K, N]` | `D = a^T × b` | a `(M, 1)`；b `(N, 1)` |
| `tt` | `[K, M]` | `[N, K]` | `D = a^T × b^T` | a `(M, 1)`；b `(K, 1)` |

三点澄清：

1. **N/T 说的是"传入张量与数学形状的关系"，不是"数据是否被搬过"**。`bf16_gemm_nt` 的 b 传入 `[N, K]`，意味着调用者手里的张量（逻辑上）就是数学 B 的转置；这块内存可能本来就以这种形态存在（比如训练框架保存的权重），完全不需要谁执行过一次搬移。
2. **四种后缀计算的是同一个 D**。用 M=4、N=5、K=6 实测（实践步骤 1 的脚本包含此验证）：四种绑定算出的 D 与参考 `A_math @ B_math` 逐元素相等。
3. **D 没有字母**。输出永远是 `[M, N]`，且仓库要求它是 N-major（行优先紧凑），见 3.6 节。

为什么深度学习里 **NT 最常见**？`nn.Linear` 的权重 `weight` 形状是 `[N, K]`（输出维在前），且 row-major（stride `(K, 1)`）；激活 x 是 `[M, K]` row-major。前向计算 `y = x @ weight^T` 恰好是"a 为 `[M, K]` 的 n 形态 + b 为 `[N, K]` 的 t 形态"——即 NT。这也解释了为什么仓库把 `nt` 作为内部规范视图：`get_gemm_desc()` 统一接收 `[M, K]` / `[N, K]` 形状的 A/B（`csrc/apis/gemm.hpp` 第 50 行形状总表），四种后缀的接口在进入它之前先各自做一次**视图**转置（源码导读站点 3）。而 `nn` 接口是仓库为数学形状直接传入的场景提供的便利接口——README 第 61 行明确写着 "The `bf16_gemm_nn` and `fp8_fp4_gemm_nn` interfaces take `A[M, K]` and `B[K, N]`"。

这一约定与 BLAS/cuBLASLt 的 `op(A)` 记号一致——仓库还保留了 `cublaslt_gemm_nt/nn/tn/tt` 作为 ACLNN 后端的别名（`deep_gemm/__init__.py` 第 39～42 行）。

### 3.6 物理布局与 Major：仓库如何描述布局

"传入形状"（N/T）只是**逻辑**约定；同一逻辑形状还可以有不同的**物理**连续方向。看两种逻辑上都是 `[M, K]` 的张量：

```text
K-major（row-major）:  stride = (K, 1)   内存里连续的是"同一行的 K 个元素"
MN-major（col-major）: stride = (1, M)   内存里连续的是"同一列的 M 个元素"
```

仓库不用形状区分这两者（形状都是 `[M, K]`），而是用一个专门的枚举——`deep_gemm/include/deep_gemm/common.hpp` 第 82 行：

```cpp
enum class Major : uint8_t { K, MN };
```

host 侧从 stride 推断 major 的逻辑在 `csrc/utils/layout.hpp` 第 37～47 行：`stride(-1) == 1`（最后一维连续）→ `Major::K`；`stride(-2) == 1`（倒数第二维连续）→ `Major::MN`；两者同时为 1 视为歧义，直接 panic；两者都不为 1 则不合法。对 A/B 要求"最后两维必须有一维连续"，对输出 D 则固定要求 `stride(-1) == 1`（第 49～52 行 `check_major_type_cd`），即 D 恒为 N-major——`GemmDesc` 的注释总结为 "D is always N-major and SF is always MN-major"（`csrc/jit_kernels/gemm_config.hpp` 第 37～38 行）。

现在把逻辑和物理两个层面接起来。假设存储都是紧凑的，则 major 与 N/T 存在一一对应（这正是仓库测试给布局打标签的依据，`tests/generators.py` 第 750～752 行的 `major_opt()`）：

| a 的 major | b 的 major | 布局标签 | 理解 |
| :---: | :---: | :---: | :--- |
| K | K | `NT` | a `[M,K]` 行优先 = A 的 n 形态；b `[N,K]` 行优先 = B 的 t 形态 |
| K | MN | `NN` | b `[N,K]` 列优先 ≡ 数学 B `[K,N]` 行优先 = B 的 n 形态 |
| MN | K | `TT` | a `[M,K]` 列优先 ≡ `[K,M]` 行优先 = A 的 t 形态 |
| MN | MN | `TN` | 两个操作数都处于"数学形状的行优先"形态 |

```python
# tests/generators.py:750-752（原文）
def major_opt(major_a: Major, major_b: Major) -> str:
    """DeepGEMM-style NT/NN/TN/TT label for a (major_a, major_b) pair."""
    return ('N' if major_a == Major.K else 'T') + ('T' if major_b == Major.K else 'N')
```

注意 A、B 两个字母的判定方向**相反**（A 用 K→N，B 用 K→T）：因为收缩维 K 是 A 的**最后**一维、却是数学 B（`[K, N]`）的**第一**维。"K 连续"对 A 意味着数学形状原样（n），对 B 意味着转置形态（t）。

为什么仓库坚持用 major + stride 而不是"要求调用者转置成统一形状"？因为 major 直接决定 kernel 的取数地址公式（源码导读站点 1 的 `gm_ptr::index()` 按 major 走两个分支），还影响对齐要求和配置选择——例如 `csrc/jit_kernels/heuristics.hpp` 第 120～121 行会按 `is_mn_major(d.major_a)` 选取不同的对齐参数（细节留到第 12 天）。此外部分场景对 major 有硬性约束：`get_gemm_desc()` 在 M-Grouped 分支断言 `major_a == Major::K`（`csrc/apis/gemm.hpp` 第 90 行），K-Grouped 分支断言 A、B 都必须是 `Major::MN`（第 94～95 行）。

最后两个与布局相关的补充（今天知道即可）：

- **FP4 打包**：仓库用 `torch::kInt8` 一个字节存两个 FP4 值（`csrc/utils/dtype.hpp` 第 9 行 `kPackedFP4`），此时"存储元素数"与"逻辑元素数"相差 2 倍，`get_shape_by_spec` 会把 major 轴上的尺寸乘回 2（`csrc/utils/layout.hpp` 第 185～187 行）。这是"逻辑形状 ≠ 存储形状"的又一实例。
- **缩放因子（SF）布局**：低精度 GEMM 的 SF 要求 MN-major 的特殊打包布局，用户输入若不满足，会由 `transform_sf_into_required_layout()` 启动专门 kernel 重排（`csrc/apis/layout.hpp` 第 68～92 行）。这是"必须实际搬移数据"的典型场景，第 33 天专门学习。

### 3.7 逻辑转置 vs 实际搬移：仓库的设计选择

把 3.4～3.6 串起来，可以看清仓库在布局上的核心设计：**能视图转置的地方一律视图转置，物理数据一个字节都不动**。

四个后缀接口的实现是同一个模板函数（以 BF16 为例，`csrc/apis/gemm.hpp` 第 424～453 行）：

```cpp
template <bool kTransA, bool kTransB>
static void bf16_gemm(const torch::Tensor& a_, const torch::Tensor& b_, ...) {
    auto a = get_transposed<kTransA>(a_, 0, 1);   // 视图转置（或原样返回）
    auto b = get_transposed<kTransB>(b_, 0, 1);
    ...  // 统一以 [M, K] / [N, K] 视图进入 get_gemm_desc()
}
inline constexpr auto bf16_gemm_nt = bf16_gemm<false, false>;
inline constexpr auto bf16_gemm_tn = bf16_gemm<true, true>;
inline constexpr auto bf16_gemm_nn = bf16_gemm<false, true>;
inline constexpr auto bf16_gemm_tt = bf16_gemm<true, false>;
```

`get_transposed<kEnable>()`（`csrc/utils/layout.hpp` 第 88～99 行）在 `kEnable` 为真时调用 ATen 的 `tensor.transpose(dims...)`——如 3.4 节所证，这是零拷贝的元数据操作。K-Grouped 接口的注释把这层意图写得最明白（`csrc/apis/gemm.hpp` 第 526～528 行，`get_gemm_desc` 的文档注释第 55～57 行也有同样的说法）：

```cpp
// here we use transpose to make the input tensors at [m, k] shape
// this is only for the purpose of getting the correct shape, the actual data layout is not changed
// this will make get_gemm_desc much simpler, as we don't need to handle the transposed case separately
```

译：这里用 transpose 把输入张量变成 `[m, k]` 形状，只是为了拿到正确的形状信息，**实际数据布局没有任何改变**；这样 `get_gemm_desc` 就不必单独处理转置情形。

收益是量化的：一次 `[M, K]` BF16 矩阵的物理转置要读写 `2 × M × K × 2` 字节（读一遍 + 写一遍）。以第 1 天旗舰 shape 的 A（`4096 × 16384`）为例是 512 MiB 的内存流量——如果每个 GEMM 调用都先转置一次数据，仅此一项就可能吞掉全部收益。因此"用视图 + 让 kernel 直接支持两种 major"是高性能 GEMM 库的标准做法。

反过来说，**什么时候必须搬**？当目标格式无法用 stride 表达时。典型例子就是 3.6 节末尾的 SF 重排：硬件要求打包成对的 MN-major 布局，这不是"换个 stride 就能看出来的视图"，必须启动 `transform_sf` kernel 逐元素重写。判断口诀：**视图能表达的（转置、切片、加间隙）用视图；视图表达不了的（打包、分形重排）才搬数据**。

## 四、源码导读

今天的导读主题是"**布局信息在仓库里如何被表达、推断和传递**"，全程静态阅读、无需硬件。建议对照仓库依次打开以下文件（路径相对 `DeepGEMM-Ascend/`）。

### 站点 1：布局的设备侧定义 —— `deep_gemm/include/deep_gemm/common.hpp`

关键代码：`Major` 枚举（第 82 行）与 `gm_ptr` 结构（第 137 行起）。`gm_ptr` 是 kernel 侧操作数指针的封装，其 `index()` 方法（第 156～165 行）就是 3.2 节地址公式的落地：

```cpp
constexpr __aicore__ gm_ptr index(uint64_t mn_idx, uint64_t k_idx) const {
    gm_ptr r = *this;
    if constexpr (kMajor == Major::K) {
        r.addr += sizeof(dtype_t) * (mn_idx * stride_outer + k_idx / pack_size);
    } else {
        r.addr += sizeof(dtype_t) * (k_idx * stride_outer + mn_idx / pack_size);
    }
    return r;
}
```

K-major 时 `(mn_idx, k_idx)` 映射到 `mn_idx * stride_outer + k_idx`（行优先公式，stride_outer 即行距）；MN-major 时两个下标换位（列优先公式）。**同一个逻辑元素，major 不同，线性地址完全不同**——这就是 kernel 必须知道 major 的原因。

### 站点 2：从 stride 推断 major —— `csrc/utils/layout.hpp`

关键函数：`major_check()`（第 37～42 行）、`get_major_type_ab()`（第 44～47 行）、`check_major_type_cd()`（第 49～52 行）。

```cpp
static Major get_major_type_ab(const at::Tensor& t) {
    major_check(t);
    return t.stride(-1) == 1 ? Major::K : Major::MN;
}
```

三个函数分别处理：A/B 必须最后两维之一连续（否则 panic，两维都连续也算歧义 panic，例如 `[M, 1]` 的 stride 是 `(1, 1)`）；D 固定要求 `stride(-1) == 1`。`get_gemm_desc()` 第 79～80 行对 a、b 各调用一次 `get_major_type_ab`，结果写入描述符（第 178～179 行 `.major_a` / `.major_b`）。

### 站点 3：四种后缀与视图转置 —— `csrc/apis/gemm.hpp` + `csrc/utils/layout.hpp`

关键代码：`get_transposed()` 模板（`utils/layout.hpp` 第 88～99 行，对 `torch::Tensor` 或 `(Tensor, Tensor)` pair 调用 `transpose(dims...)`，SF 张量跟随数据一起转视图）；`bf16_gemm<kTransA, kTransB>`（`apis/gemm.hpp` 第 424～448 行）与第 450～453 行的四个别名（注意模板实参的对应关系：`nn = <false, true>`——因为 `[K, N]` 的 b 需要 transpose 视图才能变成统一的 `[N, K]` 视图，而 `nt = <false, false>` 什么都不用做）；`fp8_fp4_gemm` 及其四个别名（第 225～256 行，约定相同）。再读第 45～59 行 `get_gemm_desc` 的形状总表注释和第 55～57 行"not a real transpose"的说明，以及 `get_gemm_desc` Normal 分支（第 82～84 行）如何用 `"mk" / "nk" / "mn"` 标注从三个张量解出 M/N/K。

这一站回答的问题：**四种后缀如何归一化？**——全部通过零拷贝视图转成 `[M, K]` / `[N, K]`，major 和 stride 原样保留。

### 站点 4：stride 如何进入 kernel —— `csrc/jit_kernels/gemm_config.hpp`

关键函数：`get_outer_stride()`（第 23～34 行）与 `GemmDesc` 的 stride 字段（第 66、75 行）。注释写得非常清楚（第 23～26 行）：

```cpp
// Physical dim0 (non-contiguous axis) row stride in ELEMENTS, for the kernel's gm_ptr.
// K-major: inner axis is K (stride(-1) must be 1), outer stride is along MN = stride(-2).
// MN-major: inner axis is MN (stride(-2) must be 1), outer stride is along K = stride(-1).
```

即 kernel 只需要**内层轴连续 + 一个外层 stride** 这两个信息即可正确寻址，正好对应 `gm_ptr` 的 `index()` 公式。`get_gemm_desc()` 在 `csrc/apis/gemm.hpp` 第 191～193 行把 a、b、d 的 outer stride 写入 `GemmDesc`；Batched 场景还额外携带 `stride(0)`（第 188～190 行），因为 `.permute()` 之后 batch 维的间距不再是 `M*K` 等紧凑值。tight（紧凑）输入时这些值恰好等于 k / k / n，strided 输入则携带真实步长——`make_strided()` 制造的行间空洞就是这样被 kernel 正确跨过的。

### 站点 5：测试侧的布局机器 —— `tests/generators.py` 与 `tests/common.py`

- `Major` 枚举（`generators.py` 第 15～17 行，与 C++ 侧对应）。
- `quantize_with_major()` 的 BF16 分支（第 62～65 行）：`transpose(major_dim, -1).contiguous().transpose(major_dim, -1)` 惯用法——3.4 节末尾已拆解，"一次搬移 + 两次视图"构造出指定 major 的数据。
- `GemmTestDesc.get_available_trans_ab()`（第 290～298 行）：Normal/Batched 枚举 `['nt', 'tn', 'nn', 'tt']` 四种绑定，M-Grouped 只有 `['nt']`、K-Grouped 只有 `['tn']`。
- `_generate_normal()`（第 310～315 行）：`major_dim=0 if self.a_major == Major.MN else 1`——注意生成的 a、b **逻辑形状固定为 `[M, K]` / `[N, K]`**，major 只影响 stride。
- `major_opt()`（第 750～752 行）：major 对 → NT/NN/TN/TT 标签（3.6 节的对应表即源于此）。
- `tests/common.py` 的 `gemm_dg()`（第 49～66 行）：按 `trans_ab` 选择接口，并用视图转置把生成的张量适配成该后缀要求的形状（第 64～65 行，`get_transposed()` 是纯 Python 的 view 转置，第 42～46 行）；`run_gemm_test()` 第 182 行用 `major_opt` 打出 `layout=NT` 之类的标签（`print_perf` 第 230～236 行）。
- （可选）`tests/test_layout.py` 的 `with_major()`（第 31～37 行，同一惯用法构造两种 major 的 SF）与 `check_layout()`（第 53～58 行，断言输出 `stride(-2) == 1` 且 `stride(-1) == size(-2)`，即 MN-major）。

**阅读检查点**（能回答再进入实践）：

1. `get_major_type_ab` 如何区分 K-major 和 MN-major？什么情况会 panic？
2. `bf16_gemm_nn` 为什么对应模板实参 `<false, true>`？
3. kernel 侧寻址只需要哪两个布局信息？（内层轴连续 + outer stride）
4. 测试生成的 a、b 逻辑形状固定是什么？major 体现在哪里？

## 五、实践步骤

以下步骤按顺序执行，**全部不需要 Ascend 硬件**；总时长约 2 小时。

### 步骤 1：用 PyTorch 观察 transpose（约 20 分钟）

在你自己的练习目录新建 `layout_probe.py`（不要改动仓库文件），写入：

```python
import torch

torch.manual_seed(0)

# 实验 1：transpose 前后的 shape / stride / is_contiguous / data_ptr
a = torch.arange(12).reshape(4, 3).float()
at = a.transpose(0, 1)
atc = at.contiguous()
print('a   :', tuple(a.shape),  a.stride(),  a.is_contiguous(), hex(a.data_ptr()))
print('at  :', tuple(at.shape), at.stride(), at.is_contiguous(), hex(at.data_ptr()))
print('atc :', tuple(atc.shape),atc.stride(),atc.is_contiguous(), hex(atc.data_ptr()))
print('a 与 at 同一块内存:', a.data_ptr() == at.data_ptr())
print('a 与 atc 同一块内存:', a.data_ptr() == atc.data_ptr())
print('a[1,2] =', a[1, 2].item(), ' at[2,1] =', at[2, 1].item())
print('torch.equal(at, atc):', torch.equal(at, atc))

# 实验 2：切片造成非紧凑 stride
s = torch.randn(4, 5)[:, :3]
print('s   :', tuple(s.shape), s.stride(), s.is_contiguous())

# 实验 3：构造 MN-major 张量（仓库惯用法）
x = torch.randn(4, 3)
mn = x.transpose(0, 1).contiguous().transpose(0, 1)
print('mn  :', tuple(mn.shape), mn.stride(), mn.is_contiguous())
print('mn.transpose(0,1) 是否 contiguous:', mn.transpose(0, 1).is_contiguous())

# 实验 4：四种绑定算同一个 D
m, n, k = 4, 5, 6
A_math, B_math = torch.randn(m, k), torch.randn(k, n)
D_ref = A_math @ B_math
cases = {
    'nt': (A_math,                 B_math.transpose(0, 1)),
    'nn': (A_math,                 B_math),
    'tn': (A_math.transpose(0, 1), B_math),
    'tt': (A_math.transpose(0, 1), B_math.transpose(0, 1)),
}
for name, (ta, tb) in cases.items():
    if name == 'nt':   D = ta @ tb.transpose(0, 1)
    elif name == 'nn': D = ta @ tb
    elif name == 'tn': D = ta.transpose(0, 1) @ tb
    else:              D = ta.transpose(0, 1) @ tb.transpose(0, 1)
    print(f'{name}: a{tuple(ta.shape)} stride{ta.stride()}  '
          f'b{tuple(tb.shape)} stride{tb.stride()}  D==D_ref: {torch.equal(D, D_ref)}')
```

运行 `python layout_probe.py`，对照预期输出（实测值，float 的 data_ptr 每次不同）：

```text
a   : (4, 3) (3, 1) True 0x...
at  : (3, 4) (1, 3) False 0x...        # data_ptr 与 a 相同
atc : (3, 4) (4, 1) True 0x...         # data_ptr 与 a 不同
a 与 at 同一块内存: True
a 与 atc 同一块内存: False
a[1,2] = 5.0  at[2,1] = 5.0            # 同一地址、同一值
torch.equal(at, atc): True
s   : (4, 3) (5, 1) False              # 行间距仍为 5
mn  : (4, 3) (1, 4) False              # MN-major：stride(-2)==1
mn.transpose(0,1) 是否 contiguous: True  # 等价于 [3,4] row-major 存储
nt: a(4, 6) stride(6, 1)  b(5, 6) stride(1, 5)  D==D_ref: True
nn: a(4, 6) stride(6, 1)  b(6, 5) stride(5, 1)  D==D_ref: True
tn: a(6, 4) stride(1, 6)  b(6, 5) stride(5, 1)  D==D_ref: True
tt: a(6, 4) stride(1, 6)  b(5, 6) stride(1, 5)  D==D_ref: True
```

无 PyTorch 的替代任务：跳过本步骤，把步骤 2、3 全部用纸笔完成（它们是等价的手工演算）。

### 步骤 2：手工地址演算（约 25 分钟）

设 `A` 为 `[4, 3]` 矩阵，值如下，按 row-major 以 float32（s=4 字节）存储：

```text
A = │ 10 11 12 │
    │ 20 21 22 │
    │ 30 31 32 │
    │ 40 41 42 │
```

1. 写出 A 的 stride 与 12 个元素在内存中的排列顺序；
2. 写出 `At = A.transpose(0, 1)` 的 shape 与 stride，并说明为什么不需要搬任何数据；
3. 分别计算 `A[1][2]` 与 `At[2][1]` 的字节偏移（相对 base），验证两者相等（答案：`1×3+2 = 5` 与 `2×1+1×3 = 5`，均为 5 × 4 = 20 字节）；
4. 假设 `A2` 是**同样数值**的 `[4, 3]` column-major 存储，写出它的内存排列顺序和 `A2[i][j]` 的偏移公式，并回答：`A2` 与哪个矩阵共享"内存排列相同"的性质？（答案：`A2` 的内存序列与 `A^T` 按 `[3, 4]` row-major 存储时相同）

### 步骤 3：填写四种布局对照表（约 30 分钟，当天产出）

取 `M=128、N=64、K=32`、BF16（s=2）。假设 A_math、B_math 各自以"数学形状的 row-major 紧凑存储"存在，按四种绑定填表（形状、stride、是否 contiguous）。模板与参考答案见第六节。填完后自查两点：

- 每行的数学语义是否都化简为 `D = A_math × B_math`；
- 注意 `nt`/`tt` 的 b（`[N, K]` row-major）与 `nn`/`tn` 的 b（`[K, N]` row-major）的 stride 差别。

### 步骤 4：major 标签练习（约 15 分钟）

不看 3.6 节的表，独立完成：

1. 对 `(a_major, b_major)` 的四种组合 `(K,K)`、`(K,MN)`、`(MN,K)`、`(MN,MN)`，写出 `major_opt()` 会给出的标签（答案见自测题 6）；
2. 打开 `csrc/apis/gemm.hpp` 第 88～96 行，抄下 M-Grouped 与 K-Grouped 分支对 major 的断言，并从 `tests/generators.py` 第 332～337、372～377 行确认测试确实按此约束生成数据（M-Grouped 的 a 用 `major_dim=1` 即 K-major；K-Grouped 的 a、b 都用 `major_dim=1` 作用于 `[total_k, m/n]` 形状，即 MN-major）。

### 步骤 5：源码导读打卡（约 30 分钟）

按第四节顺序依次打开 5 个站点文件，找到列出的行号并通读相应片段，每站用一句话记下它的职责。

（可选，有 Ascend 环境时）在仓库根目录执行 `./develop.sh`，再运行 `python tests/test_bf16_gemm.py normal --skip-prof`，观察每个用例输出中的 `layout=NT` / `layout=NN` 等标签——它们正是 `major_opt(test.a_major, test.b_major)` 的返回值（`tests/common.py` 第 182、230～232 行）。没有环境完全不影响今天的学习，静态阅读即替代任务。

## 六、当天产出

学习计划要求今天的产出是**一张四种 GEMM 布局的 A/B 形状对照表**。参考模板（可直接抄入笔记填写，参考答案在其后）：

```text
# 四种 GEMM 布局对照表（第 2 天）   M=128, N=64, K=32, BF16

| 后缀 | a 传入形状 | a stride | a contig | b 传入形状 | b stride | b contig | D 形状/stride | 数学语义   |
| ---- | ---------- | -------- | -------- | ---------- | -------- | -------- | ------------- | ---------- |
| nt   | ?          | ?        | ?        | ?          | ?        | ?        | ?             | D = a×bᵀ   |
| nn   | ?          | ?        | ?        | ?          | ?        | ?        | ?             | D = a×b    |
| tn   | ?          | ?        | ?        | ?          | ?        | ?        | ?             | D = aᵀ×b   |
| tt   | ?          | ?        | ?        | ?          | ?        | ?        | ?             | D = aᵀ×bᵀ  |

补充：N/T 的含义：a 的字母 = A([M,K]) 是否转置传入；b 的字母 = B([K,N]) 是否转置传入。
     major 与标签：a K-major→'N'/MN-major→'T'；b K-major→'T'/MN-major→'N'。
```

**参考答案**（前提：每个传入张量**本身**都以 row-major 紧凑存储——例如 `nt` 的 b 就像 `nn.Linear` 的权重，天然以 `[N, K]` 形态存放，而不是谁的转置视图）：

| 后缀 | a 传入形状 | a stride | a contig | b 传入形状 | b stride | b contig | D 形状/stride | 数学语义 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| nt | `[128, 32]` | `(32, 1)` | True | `[64, 32]` | `(32, 1)` | True | `[128, 64]` `(64, 1)` | `D = a × bᵀ` |
| nn | `[128, 32]` | `(32, 1)` | True | `[32, 64]` | `(64, 1)` | True | `[128, 64]` `(64, 1)` | `D = a × b` |
| tn | `[32, 128]` | `(128, 1)` | True | `[32, 64]` | `(64, 1)` | True | `[128, 64]` `(64, 1)` | `D = aᵀ × b` |
| tt | `[32, 128]` | `(128, 1)` | True | `[64, 32]` | `(32, 1)` | True | `[128, 64]` `(64, 1)` | `D = aᵀ × bᵀ` |

说明：本表假设的是"每个传入张量自身紧凑"的情形，因此四行全部 contiguous。同样合法的是"视图"形态：例如 `nt` 的 b 可以是 `[32, 64]` row-major 存储的转置视图（形状 `[64, 32]`、stride `(1, 32)`、非 contiguous，即实践步骤 1 实验 4 的构造），也可以是逻辑 `[64, 32]`、stride `(1, 64)` 的 MN-major 张量——三种物理形态的计算结果完全相同，kernel 只是走不同的 major 寻址分支。这正是"逻辑形状、物理 major、连续性是三个独立信息"的体现。

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. 一个 `[512, 256]` 的张量，row-major 和 column-major 的 stride 分别是什么？
2. `t = x.transpose(0, 1)` 执行后，哪些东西变了（shape/stride/data_ptr/数据内容/is_contiguous），哪些没变？
3. 什么时候必须调用 `.contiguous()`？对 `[M, K]` BF16 张量，它的代价是多少字节内存流量？
4. `bf16_gemm_tt` 的 a、b 逻辑形状分别是什么？它计算的数学表达式是什么？
5. 调用者手里有一个 `[64, 32]`、stride 为 `(32, 1)` 的 b（row-major）。它作为 B 操作数对应哪种后缀？如果同一个逻辑张量 stride 是 `(1, 64)` 呢？
6. `(a_major, b_major)` 为 `(K, K)`、`(K, MN)`、`(MN, K)`、`(MN, MN)` 时，`major_opt()` 分别给出什么标签？
7. 仓库如何从张量推断 A/B 的 major？输出 D 的 major 有什么限制？
8. （思考题）`get_gemm_desc` 统一以 `[M, K]` / `[N, K]` 视图接收 A/B。四种后缀接口靠什么操作做到这一点？为什么这样设计是"免费"的？

**参考答案：**

1. row-major：`(256, 1)`；column-major：`(1, 512)`。
2. 变了：shape（两维互换）、stride（两个步长互换）、`is_contiguous()`（通常由 True 变 False）。没变：`data_ptr`、底层 storage、数据内容（每个下标组合对应的值不变，`t[j][i] == x[i][j]`）。
3. 当下游接口要求紧凑（或特定）物理布局、而当前张量只是视图时。代价：读一遍 + 写一遍 = `2 × M × K × 2` 字节（BF16）。
4. a 是 `[K, M]`，b 是 `[N, K]`；计算 `D = aᵀ × bᵀ`，即 `D[m][n] = Σ a[k][m]·b[n][k]`（见 `csrc/apis/gemm.hpp` 第 418～419 行注释）。
5. `[64, 32]` row-major 的 b 是 B_math（`[32, 64]`）的转置存储 → 't' 形态，可调用 `nt` 或 `tt`。stride 为 `(1, 64)` 时是 MN-major，等价于 `[32, 64]` row-major 的视图 → 'n' 形态，可调用 `nn` 或 `tn`。两种情况下数学结果相同，只是 kernel 走不同的 major 寻址分支。
6. 依次为 `NT`、`NN`、`TT`、`TN`（`tests/generators.py` 第 750～752 行）。
7. `get_major_type_ab()`：`stride(-1) == 1` → `Major::K`，`stride(-2) == 1` → `Major::MN`，两者同时为 1 视为歧义 panic，都不为 1 则断言失败（`csrc/utils/layout.hpp` 第 37～47 行）。D 由 `check_major_type_cd()` 限定 `stride(-1) == 1`，即恒为 N-major（第 49～52 行）。
8. 靠 `get_transposed<kTransA/kTransB>()` 中的视图 `transpose(0, 1)`（零拷贝元数据操作）。免费的原因：transpose 只交换 shape/stride，不搬数据（源码注释原话 "the actual data layout is not changed"，`csrc/apis/gemm.hpp` 第 55～57 行）；真实的 major 与 stride 信息原样传给 `get_gemm_desc`，kernel 再按 major 分支寻址。

## 附录 A：本日概念速查

| 概念 | 一句话定义 | 仓库落点 |
| :--- | :--- | :--- |
| Row-major | 先存第 0 行再存第 1 行；`addr(i,j) = (i×K+j)×s` | PyTorch 默认；D 的要求 |
| Column-major | 先存第 0 列；`addr(i,j) = (j×M+i)×s` | A/B 的 MN-major |
| stride | 沿各维走一步跳过的元素数 | `GemmDesc.outer_stride_a/b/d` |
| contiguous | stride 恰为行优先紧凑值 | `get_outer_stride` 的断言 |
| N/T 后缀 | 传入张量相对数学形状是否转置 | `bf16_gemm_nt/nn/tn/tt` |
| Major | 内层连续轴是 K 还是 MN | `enum class Major { K, MN }` |
| 逻辑转置 | 换 shape/stride 元数据，零拷贝 | `get_transposed`、`get_gemm_desc` 注释 |
| 实际搬移 | 移动全部元素到新布局 | `.contiguous()`、`transform_sf` |

## 附录 B：遗留问题记录模板

```text
# 第 2 天遗留问题
1. （例如）FP4 打包时 stride 与逻辑元素数的换算还没完全弄清 —— 计划第 4/32 天回顾
2. （例如）为什么 K-Grouped 强制 A、B 都 MN-major？—— 计划第 37 天对照 scheduler 分析
3. ……
```
