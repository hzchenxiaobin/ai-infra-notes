# 第 3 周 · 第 4 天（总第 18 天）：理解地址与 Tile 表达

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 3 周"BF16 Kernel——模板、调度与数据搬运"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

前两天解决了"缓冲怎么规划"（第 16 天：`l1a/l1b/l0a/l0b/l0c` 的形状、基址与级数）和"Tile 分给谁"（第 17 天：调度器把全局块号映射成 `(m_block_idx, n_block_idx)`，再由 `get_global_idx()` 换算成元素坐标 `m_idx/n_idx`）。今天回答最后一个"定位"问题：**这些坐标如何变成真实地址？** 答案分布在两个小文件里：`deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp`（102 行，片上指针族）和它指向的 `deep_gemm/include/deep_gemm/common.hpp`（`gm_ptr`，第 137-186 行，全局内存指针）。注意一个容易踩的坑：学习计划说"阅读 `ascend/tile_ptr.hpp`"，但 `gm_ptr` 其实**不在**这个文件里——tile_ptr.hpp 第 99-100 行的注释写明"gm_ptr lives in common.hpp so the host can build one and pass it as a kernel launch argument"。今天的主线是：`gm_ptr` 的 `index()/batch()/group()` 与两条 stride、MN-major 与 K-major 对地址计算的三点影响、以及把第 17 天的坐标一路串到 GM/L1/L0 的具体字节地址。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 画出**三族"形状化指针"**的全景：`gm_ptr`（`common.hpp:137-186`，全局内存，Host/Device 共享）、`operand_ptr`（`tile_ptr.hpp:18-56`，A/B 型操作数）与 `result_ptr`（`tile_ptr.hpp:58-91`，输出型），并说出 `l1_ptr/l0a_ptr/l0b_ptr/l0c_ptr/ub_ptr` 五个别名各自绑定的地址空间限定符（`tile_ptr.hpp:93-97`）；
2. 解释 `gm_ptr` 为什么必须放在 `common.hpp`：Host 侧要构造它并作为 kernel 启动参数传递，而 `__aicore__/__gm__` 在非设备编译器上被宏抹掉（`common.hpp:23-26`）；由此说明 `index()/group()/ptr()` 三个方法被 `#ifdef __CCE__` 包住、只在设备侧可用（`common.hpp:150-173`），`operator+/batch()` 则两侧都可编译；
3. **手写 `index()` 的两个分支**（`common.hpp:156-165`）：K-major 时 `addr += sizeof(T) × (mn_idx × stride_outer + k_idx / pack_size)`，MN-major 时 `addr += sizeof(T) × (k_idx × stride_outer + mn_idx / pack_size)`，并解释 `pack_size` 只作用于连续轴（FP4 沿连续轴两元素打包，BF16 恒为 1）；
4. 写出 `batch()` 与 `group()` 的字节偏移公式（`common.hpp:167-172、181-185`），说出它们各自服务哪种 `GemmType`（Batched 的批偏移；M-Grouped 的 B、K-Grouped 的 D 的组偏移），并在 `bf16_gemm.hpp:102-103、157、188` 找到调用点；
5. 追踪 **stride 的完整来源链**：PyTorch 张量的 `stride()` → `get_major_type_ab()`（`csrc/utils/layout.hpp:44-47`，`stride(-1)==1` 判 K-major）→ `get_outer_stride()`（`csrc/jit_kernels/gemm_config.hpp:27-34`）→ `GemmDesc.outer_stride_a/b/d`（`csrc/apis/gemm.hpp:191-193`）→ launch 处构造 `gm_ptr<uint8_t>{...}`（`csrc/jit_kernels/bf16_gemm.hpp:74-76`）→ kernel 参数 `gm_ptr<bfloat16_t, kMajorA>` → `copy_gm_to_l1` 的行距参数（`copy.hpp:123、127`）与 epilogue 写回的 burst 间距（`epilogue.hpp:81-82`）；
6. 解释输出矩阵的 Major 语义：`gm_d` 声明为 `gm_ptr<cd_dtype_t>`（默认 `Major::K`，`bf16_gemm.hpp:30`），此时 index 的"k_idx 槽位"装的是 N 列号——即"D 恒为 N-major（行主序）"，与 `GemmDesc` 注释（`gemm_config.hpp:74`）和 `check_major_type_cd`（`utils/layout.hpp:49-52`）对应；
7. 说明 **MN-major 与 K-major 的三点差异**：`index()` 公式换位；`copy_gm_to_l1` 的二维搬运行列互换且 NZ 目标参数从 `dst.shape_mn` 换成 `dst.shape_k`（`copy.hpp:120-129`）；`copy_l1_to_l0a/b` 改用 `_transpose` 变体、分形下标角色互换（`copy.hpp:139-143、153-157`）——同时说明 **L0A/L0B 布局与 Major 无关**（`l0a_ptr/l0b_ptr` 别名不带 Major 模板参数，`tile_ptr.hpp:94-95`）；
8. 说明 `as_mad_aligned`（`tile_ptr.hpp:52-55、88-90`）对齐的是 tile 句柄的**形状字段**（决定分形搬运几何与 L0C 的 NZ 行距），而 `mad()` 与 epilogue 仍接收未对齐的 `eff_*` 有效行数（`bf16_gemm.hpp:149-151、161`），两者分工不同；
9. 把第 17 天的演算延伸到地址：对给定 `(m_idx, n_idx, k_idx)` 与布局，手算 `a.index/b.index/d.index` 的字节偏移、`l1a[s]/l1b[s]` 的 stage 地址、`copy_l1_to_l0a` 的分形单元下标，以及边界 Tile 的 `as_mad_aligned` 效果。

**当天产出：一组地址计算示例**（模板见第六节，参考答案见 3.7 节）。

## 二、预备知识

- **第 17 天（最重要）**：`m_idx/n_idx = get_global_idx(BLOCK_*, block_idx)`（`scheduler.hpp:113-115`）、`k_idx = k_idx_base + k_blk_idx × BLOCK_K`（`bf16_gemm.hpp:113`）、`actual_m/actual_n` 的边界语义；今天的所有地址都从这些坐标出发。
- **第 16 天**：`operand_ptr/result_ptr` 的 `size_per_stage()/offset()/operator[]`——stage 下标寻址；今天补齐 `cvt_pointee`、`offset_bytes`、`offset_nz_m`、`as_mad_aligned` 的细节并把重心移到 `gm_ptr`。
- **第 2 天**：Row-major/Column-major、stride、逻辑转置与实际搬移的区别——今天就是"stride 进入 kernel"的代码落地。
- **第 11 天**：`GemmDesc` 的字段（m/n/k、major、group）；今天补上 `outer_stride_*` 与 `stride_batch_*` 两组字段。
- **第 5 天**：GM/L1/L0A/L0B/L0C/UB 的分层职责；地址空间限定符 `__gm__/__cbuf__/__ca__/__cb__/__cc__/__ubuf__` 是它们在 CCE 语法里的名字。
- **C++ 语法**：模板别名（`using l1_ptr = operand_ptr<...>`）、`reinterpret_cast`、`constexpr` 成员函数、特化与偏特化（`cvt_pointee`）。能对照读懂即可。

## 三、详细讲解

### 3.1 全景：三族"形状化指针"与六个地址空间

Ascend 的 CCE 设备编译器用**地址空间限定符**区分存储层级：`__gm__`（Global Memory）、`__cbuf__`（L1）、`__ca__`（L0A）、`__cb__`（L0B）、`__cc__`（L0C）、`__ubuf__`（UB）。DeepGEMM 在这些裸指针之上封装了三族"带形状的指针"——**地址 + 逻辑形状 + 布局信息**打包成一个值类型，让 kernel 代码用下标而不是裸字节算地址：

| 指针族 | 定义位置 | 地址空间 | 字段 | 谁构造 |
| :-- | :-- | :-- | :-- | :-- |
| `gm_ptr<T, kMajor>` | `common.hpp:137-186` | `__gm__` | `addr`、`stride_outer`、`stride_batch` | **Host**（launch 参数）与设备 |
| `operand_ptr<ptr_t, elem_t, MAJOR>` | `tile_ptr.hpp:18-56` | 任意（由 `ptr_t` 决定） | `addr`、`shape_mn`、`shape_k` | 仅设备（`constexpr`） |
| `result_ptr<ptr_t, elem_t>` | `tile_ptr.hpp:58-91` | 任意（由 `ptr_t` 决定） | `addr`、`shape_m`、`shape_n` | 仅设备（`constexpr`） |

五个片上别名（`tile_ptr.hpp:93-97`）：

```cpp
template <typename T, Major M = Major::K> using l1_ptr  = operand_ptr<__cbuf__ T*, T, M>;  // L1，保留 Major
template <typename T>                    using l0a_ptr = operand_ptr<__ca__   T*, T>;      // L0A，无 Major
template <typename T>                    using l0b_ptr = operand_ptr<__cb__   T*, T>;      // L0B，无 Major
template <typename T>                    using l0c_ptr = result_ptr<__cc__    T*, T>;      // L0C
template <typename T>                    using ub_ptr  = result_ptr<__ubuf__  T*, T>;      // UB
```

三个观察：

1. **`gm_ptr` 是唯一携带 stride 的指针**。片上缓冲的布局由 kernel 自己规划（第 16 天），形状规则、无需 stride；GM 里的张量可能是 strided/转置的，行距必须随指针携带。`gm_ptr` 也是唯一需要 Major 模板参数参与**地址计算**的类型——`l1_ptr` 保留 Major 只是为了让搬运函数（3.5 节）知道 L1 里摆的是哪种分形，`l0a_ptr/l0b_ptr` 干脆不带 Major，因为 L0 布局是硬件固定的。
2. **`operand_ptr` 与 `result_ptr` 的区别**是"第二维语义"：操作数是 `[shape_mn, shape_k]`（MN 轴 × K 轴，A/B 通用，`mn` 对 A 是 M、对 B 是 N），输出是 `[shape_m, shape_n]`。`result_ptr` 不需要 `k_per_slot`（FP4 打包只发生在输入）也不需要 Major。
3. **`cvt_pointee`（`tile_ptr.hpp:8-16`）解决"换元素类型、不丢地址空间"**：`ptr<uint8_t>()` 把 `__cbuf__ bfloat16_t*` 变成 `__cbuf__ uint8_t*` 而不是裸 `uint8_t*`——六个偏特化逐一保留 `__gm__/__cbuf__/__ca__/__cb__/__cc__/__ubuf__`。`copy_gm_to_l1` 里 `dst.template ptr<uint8_t>()`（`copy.hpp:118`）就用它按字节访问 L1。

**为什么 `gm_ptr` 在 `common.hpp`？** 因为它要跨 Host/Device：`csrc/jit_kernels/bf16_gemm.hpp:74-76` 在 Host 上构造三个 `gm_ptr` 传给 `jit->launch`。设备编译器（bisheng）定义 `__CCE__` 并提供 `__aicore__/__gm__` 关键字；Host 编译器没有，于是 `common.hpp:23-26` 把它们宏抹掉，让同一个结构体在两侧都能编译。代价是 `gm_ptr` 里依赖设备类型（`is_fp4` 用到 `float4_e2m1x2_t`，`common.hpp:67-70`）的方法必须 `#ifdef __CCE__` 整体屏蔽——所以 `index()/group()/ptr()` 只在设备侧存在，Host 侧的 `gm_ptr` 只是个"待填写的 POD"（三个整数）。`common.hpp` 头部注释"pure C++: no kernel_operator.h, no intrinsics"说的就是这条纪律。

### 3.2 `gm_ptr`：唯一携带 stride 的指针

```cpp
template <typename T, Major kMajor = Major::K>
struct gm_ptr {
    uintptr_t addr;             // 字节地址
    uint64_t stride_outer;      // “外层”行距，单位：元素（不是字节）
    uint64_t stride_batch;      // Batched 的批距，单位：元素；非 Batched 为 0
    ...
};
```

三个方法做三种"平移"，全部**返回新值、不修改自身**（值语义，kernel 里 `const auto a = gm_a.batch(...)` 靠这个特性层层派生句柄）：

**`index(mn_idx, k_idx)`（`common.hpp:156-165`）——定位到一个 Tile 的左上角元素：**

```cpp
constexpr __aicore__ gm_ptr index(uint64_t mn_idx, uint64_t k_idx) const {
    constexpr uint64_t pack_size = is_fp4<dtype_t>() ? 2 : 1;
    gm_ptr r = *this;
    if constexpr (kMajor == Major::K) {
        r.addr += sizeof(dtype_t) * (mn_idx * stride_outer + k_idx / pack_size);
    } else {
        r.addr += sizeof(dtype_t) * (k_idx * stride_outer + mn_idx / pack_size);
    }
    return r;
}
```

两个分支是同一句话：**外层下标 × 外层行距 + 内层下标（按打包除）**。K-major 时 MN 是外层、K 连续；MN-major 时 K 是外层、MN 连续。`pack_size` 只除在连续轴上——FP4（`float4_e2m1x2_t`）把两个 4-bit 值打包进一个字节，且打包方向沿连续轴（K-major 打包 K、MN-major 打包 MN，`tests/generators.py:503-507` 的注释"fp4 packs two int4 values into one byte along the MN(major) dim"印证了这一点）。BF16 的 `pack_size=1`，公式退化为纯乘加。

**`batch(idx)`（`common.hpp:181-185`）——Batched 的批偏移：**

```cpp
r.addr += idx * stride_batch * sizeof(dtype_t);
```

`stride_batch` 来自 `a.stride(0)`（`apis/gemm.hpp:188-190`，仅在 `GemmType::Batched` 时非零）。注意它不一定是 `M×K`——注释（`gemm_config.hpp:62-65`）说明 `.permute()` 之后批维 stride 会变，所以运行期携带。

**`group(group_idx, shape_mn, shape_k)`（`common.hpp:167-172`）——Grouped 的组偏移：**

```cpp
r.addr += group_idx * shape_mn * shape_k / pack_size * sizeof(dtype_t);
```

组按**紧凑排列**计算：第 g 组基址 = g × 每组元素数。两个消费场景（`bf16_gemm.hpp:103、157、188`）：

- **M-Grouped 的 B**：B 是按组存储的 `[G, N, K]`，`gm_b.batch(b).group(b_group_idx, shape_n, shape_k)`；
- **K-Grouped 的 D**：D 是 `[G, M, N]`，`gm_d.batch(b).group(d_group_idx, shape_m, shape_n)`。

而 M-Grouped 的 A/D、K-Grouped 的 A/B 都是**扁平大缓冲**（组内行直接拼接，`scheduler.hpp:17-31` 的注释），不需要 `group()`——组信息通过坐标本身（m_idx 落在哪个组段、`k_idx_base` 偏移）表达。Normal/BF16 主线下 `batch(0)`、`group(0, ...)` 都是恒等操作（`batch_idx/b_group_idx` 恒 0，`bf16_gemm.hpp:99-100`），但统一写出来让四种 `GemmType` 共享同一份 kernel。

**输出 D 的 Major 语义（易混点）**。`gm_d` 声明为 `gm_ptr<cd_dtype_t>`（`bf16_gemm.hpp:30`），默认 `Major::K`。代入 `index()` 的 K 分支：`addr += sizeof(T) × (m × stride_outer + n)`——第一参数槽位 `mn_idx` 装**行号 m**，第二参数槽位 `k_idx` 装**列号 n**。也就是说 `gm_ptr` 的"Major::K"含义是"**最后一个维度连续**"；对 `[M, N]` 的 D 而言最后一维是 N，所以 D 是行主序（N-major）。这与三处源码互相印证：`GemmDesc` 注释"outer_stride_d: D's stride along M (D is always N-major)"（`gemm_config.hpp:74`）、`get_outer_stride(d, Major::K)` 返回 `d.stride(-2)`（M 方向行距，`gemm_config.hpp:28-30`）、`check_major_type_cd` 断言 `d.stride(-1) == 1`（`utils/layout.hpp:49-52`）。同理，"A 是 K-major"="A 的最后一维（K）连续"。

### 3.3 stride 从哪来：Host 链路

以 `bf16_gemm_nt(a, b, d)` 为例（`apis/gemm.hpp:413-453`，`nt/nn/tn/tt` 后缀只选择**逻辑形状**：nt 的 a 是 `[M, K]`、b 是 `[N, K]`；`get_transposed` 只是逻辑转置不搬数据，`apis/gemm.hpp:434-435` → `utils/layout.hpp:88-99`）：

```text
PyTorch 张量 (shape, stride)
  │  get_major_type_ab(a)                    utils/layout.hpp:44-47
  │    major_check: stride(-2)、stride(-1) 恰好一个为 1   (37-42，两个都为 1 报“歧义”)
  │    stride(-1)==1 → Major::K，否则 → Major::MN
  ▼
get_outer_stride(a, major_a)                 gemm_config.hpp:27-34
  │    K-major  → stride(-2)（沿 MN 的行距）；断言 stride(-1)==1
  │    MN-major → stride(-1)（沿 K 的行距）；  断言 stride(-2)==1
  ▼
GemmDesc{ outer_stride_a/b/d, stride_batch_a/b/d, a_ptr/b_ptr/d_ptr, ... }   apis/gemm.hpp:188-193
  ▼
jit->launch(kernel, options,
    gm_ptr<uint8_t>{desc.outer_stride_a, (uintptr_t)desc.a_ptr, desc.stride_batch_a}, ...)   jit_kernels/bf16_gemm.hpp:74-76
  ▼
kernel 参数 gm_ptr<bfloat16_t, kMajorA> gm_a   (bf16_gemm.hpp:28)
```

三个要点：

1. **单位约定是"元素"**。`GemmDesc` 注释写明 outer/batch stride 均以元素为单位（`gemm_config.hpp:68-71`），乘 `sizeof(dtype_t)` 才变字节（`index()/batch()` 里就是这么做的）。Host 构造时用 `gm_ptr<uint8_t>` 只是把 `{addr, stride_outer, stride_batch}` 三个整数按 POD 传过 ABI；kernel 侧参数类型 `gm_ptr<bfloat16_t, kMajorA>` 与之布局相同，按 BF16 元素解释同一组数。
2. **紧凑张量的 stride 恰好等于形状**：行主序 `[M, K]` 的 `stride(-2)=K`，所以 `outer_stride_a = K`——这正是"紧凑输入复现旧的硬编码行为"（`gemm_config.hpp:69-70` 注释）。strided/permuted 输入则携带真实 stride，**kernel 不做任何整理搬运就能消化**。
3. **接口后缀与 Major 独立**。`[K, M]` 的连续张量喂给 `tn` 接口，内部 `.transpose(0,1)` 后 stride 变 `(1, M)` → MN-major；`[M, K]` 连续张量喂给 `nt` → K-major。测试生成器正是靠 `generate_operand_with_major(..., major_dim=...)`（`tests/generators.py:169-191`）枚举四种 Major 组合 × 四种后缀。

### 3.4 kernel 里的消费点：从坐标到地址

`bf16_gemm_impl` 的 AIC 主循环（`bf16_gemm.hpp:89-172`）里，地址计算集中在循环头部：

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    const uint32_t m_idx = scheduler.get_global_idx(BLOCK_M, m_block_idx);   // 90
    const uint32_t n_idx = scheduler.get_global_idx(BLOCK_N, n_block_idx);   // 91
    ...
    const uint32_t k_idx_base = scheduler.get_k_idx_base();                  // 97（K-Grouped 的组内 K 起点）
    const uint32_t batch_idx  = is_batched    ? scheduler.get_batch_idx()  : 0;   // 99
    const uint32_t b_group_idx = is_m_grouped ? scheduler.get_group_idx()  : 0;   // 100

    const auto a = gm_a.batch(batch_idx);                                    // 102：批句柄
    const auto b = gm_b.batch(batch_idx).group(b_group_idx, shape_n, shape_k); // 103：批 + 组句柄
    ...
    for (uint32_t k_blk_idx = 0, s = 0; k_blk_idx < num_k_blocks; ...) {     // 112
        const uint32_t k_idx = k_idx_base + k_blk_idx * BLOCK_K;             // 113
        ...
        copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, kL2CtrlA);  // 117
        copy_gm_to_l1(l1b[s], b.index(n_idx, k_idx), actual_n, actual_k, kL2CtrlB);  // 118
```

`a.index(m_idx, k_idx)` 返回的仍是 `gm_ptr`——它随后作为 `copy_gm_to_l1` 的**源**，其 `stride_outer` 在函数内部变成二维搬运的行距（`copy.hpp:123`：`src.stride_outer * sizeof(T)`）。也就是说 `index()` 只定位到 Tile 左上角，"每行跳多少字节"由随指针携带的 stride 在搬运时展开。**resident A 路径**是个好对照：整块 A 一次性搬入（`a.index(m_idx, 0)`，第 110 行），此后 K 维推进全部发生在 L1 寻址里（`copy_l1_to_l0a` 的第 4 参数 `(kResidentA ? k_blk_idx * BLOCK_K : 0) + k_mad_idx`，第 143 行）——GM 地址不动，L1 内偏移动。

输出侧两条路都用 `d.index`：

- **direct store**（`kDirectStore`）：L0C 直接写 GM，`copy_l0c_to_gm(d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(), l0c_tile, eff_mad_m, eff_mad_n, d.stride_outer, ...)`（`bf16_gemm.hpp:157-159`）——`.ptr()` 把 `gm_ptr` 转成 `__gm__ T*` 裸指针，`d.stride_outer` 作为写回的行间距；
- **普通 epilogue**：AIV 侧 `epilogue.aiv_drain(d, m_idx, n_idx, ...)`（第 190 行），内部 `copy_ub_to_gm(d.index(block.m_idx, block.n_idx).ptr(), output_ub, block.num_rows, block.num_cols, d.stride_outer * sizeof(cd_dtype_t), ...)`（`epilogue.hpp:81-82`）——**同一条 stride 从输入端（搬运行距）绕到输出端（写回 burst 间距）**。K-Grouped 的空组清零也走 `d.group(group_idx, shape_m, shape_n).index(m_idx, n_idx)`（`epilogue.hpp:132`）。

全链路一图：

```text
(m_block, n_block) ──get_global_idx──▶ m_idx / n_idx
k_blk_idx ──× BLOCK_K + k_idx_base──▶ k_idx
     │
     ▼  gm_ptr::index(mn_idx, k_idx)      [common.hpp:156]   + batch()/group() 平移
GM 字节地址 = base + sizeof(T) × (外层下标 × stride_outer + 内层下标 / pack)
     │
     ▼  copy_gm_to_l1(l1a[s], src, actual_mn, actual_k)      [copy.hpp:101]
L1 stage 地址 = l1a 基址 + s × size_per_stage               [tile_ptr.hpp:38-43]（第 16 天）
     │
     ▼  copy_l1_to_l0a(l0a_tile, l1a[s], m_mad_idx, k_mad_idx)   [copy.hpp:134]
L0 分形单元 = (m_mad_idx / FRAC_MN, k_mad_idx / FRAC_K)，tile 形状经 as_mad_aligned 对齐
     │
     ▼  mad(l0c[s_cd], l0a_tile, l0b_tile, eff_m, eff_n, eff_k)  [bf16_gemm.hpp:149]
L0C stage 地址 = l0c 基址 + s_cd × MAD_M × MAD_N × sizeof(float)  [tile_ptr.hpp:73-79]
     │
     ▼  epilogue / direct store：d.index(m_idx + m_mad_idx, n_idx + n_mad_idx)
GM 输出地址（行距 = d.stride_outer）
```

### 3.5 MN-major 与 K-major：同一元素的两种地址（今天的重点）

设 A 逻辑形状 `[M, K]`，同一个元素 `A[m][k]` 在两种物理存储下的线性位置：

```text
K-major（[M, K] 行主序，stride_outer = 行距，沿 MN）：
    元素偏移 = m × stride_outer + k          → index(m, k) 走 mn_idx × stride + k_idx
MN-major（[K, M] 行主序再逻辑转置，stride_outer = 沿 K 的行距）：
    元素偏移 = k × stride_outer + m          → index(k 槽, m 槽) 走 k_idx × stride + mn_idx
```

两个公式互为转置——**Major 决定"谁是外层下标"**。由此产生三处代码差异：

**(1) `index()` 换位**（3.2 节的公式）。`pack_size` 也跟着连续轴走：K-major 除在 `k_idx` 上，MN-major 除在 `mn_idx` 上。

**(2) `copy_gm_to_l1` 行列互换**（`copy.hpp:120-129`）：

```cpp
} else if constexpr (M == Major::K) {
    set_mte2_nz(num_batch, 1, dst.shape_mn, dst_batch_stride_blocks);          // 121
    asc_copy_gm2l1_nd2nz(..., actual_mn, actual_k / pack_factor, ...);         // 123：[mn 行, k 列]
} else {
    set_mte2_nz(num_batch, 1, dst.shape_k, dst_batch_stride_blocks);           // 126
    asc_copy_gm2l1_nd2nz(..., actual_k, actual_mn / pack_factor, ...);         // 128：[k 行, mn 列]
}
```

MTE2 硬件的 `nd2nz`（普通二维 → NZ 分形）搬运总是"逐行搬连续段"：K-major 时一行 = 某 MN 行的 K 段；MN-major 时一行 = 某 K 位置的 MN 段。NZ 目标参数 `dst_nz_c0_stride` 也从 `dst.shape_mn` 换成 `dst.shape_k`——**L1 里的分形摆放是转置的**。行距参数两侧相同（都是 `src.stride_outer * sizeof(T)`），但含义一个是"MN 行距"、一个是"K 行距"。

**(3) `copy_l1_to_l0a/b` 改用 transpose 变体**（`copy.hpp:133-159`）：

```cpp
if constexpr (M == Major::K) {
    asc_copy_l12l0a(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K,
                    dst.shape_mn / FRAC_MN, dst.shape_k / FRAC_K, ...);        // 137-138
} else {
    asc_copy_l12l0a_transpose(dst.ptr(), src.ptr(), k_idx / FRAC_MN, mn_idx / FRAC_K,
                              m_step, ceil_div(dst.shape_mn, FRAC_K), ...);    // 142-143
}
```

K-major：L1 内偏移 `(mn_idx / 16, k_idx / 16)`——MN 按 16 行分形、K 按 16 元素（32 字节 C0）分块。MN-major：**两个角色互换**——`k_idx` 除以 `FRAC_MN`、`mn_idx` 除以 `FRAC_K`，且调用 `asc_copy_l12l0a_transpose`：MTE1 硬件在 L1→L0 的路上完成转置，把转置分形还原成标准分形。

**L0 不变式**：无论 GM/L1 是哪种 Major，`l0a_ptr/l0b_ptr` 都不带 Major 模板参数（`tile_ptr.hpp:94-95`），L0A/L0B 里永远是 MAD 单元要求的同一种 Z 分形——Major 的消化全部发生在 MTE2（GM→L1，nd2nz）与 MTE1（L1→L0，可选 transpose）两级搬运里。这就是"逻辑转置不需要实际搬移数据"（第 2 天）在 kernel 里的精确边界：**用户侧的 `.transpose()` 免费，但转置的代价转移到了搬运指令的选择上**。

顺带两个佐证：`copy_gm_to_ub` 的 Major 重载同样做 rows/cols 互换（`copy.hpp:58-59`，FP8 反量化与 MQA 路径使用）；K-Grouped GEMM 直接断言 `major_a == Major::MN && major_b == Major::MN`（`apis/gemm.hpp:94-95`，因为 A/B 是沿 K 拼接的扁平 `[total_k, M/N]` 缓冲，K 必须是外层），而 M-Grouped 断言 `major_a == Major::K`（第 90 行）。

### 3.6 片上导航补课：`as_mad_aligned`、`offset_bytes`、`offset_nz_m`

第 16 天已算过 `size_per_stage()/offset()/operator[]`（stage 寻址），今天补三个导航细节：

**`as_mad_aligned(shape_mn, shape_k)`（`tile_ptr.hpp:52-55`）**：返回一个**同地址、形状向上对齐**的新 tile 句柄——`aligned(shape_mn, FRAC_MN=16)` 与 `aligned(shape_k, FRAC_K)`（BF16 的 `FRAC_K = 32字节/2字节 = 16`，`common.hpp:73-79`）。kernel 在三个地方用它处理边界 Tile（`bf16_gemm.hpp:127、138-139`）：

```cpp
uint32_t eff_mad_m = min(actual_m - m_mad_idx, MAD_M);                    // 125：有效行数（未对齐）
uint32_t eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N), get_frac_k<...>());  // 126
auto l0c_tile = l0c[s_cd].as_mad_aligned(eff_mad_m, eff_mad_n);           // 127：对齐后的形状字段
...
auto l0a_tile = l0a[s_ab].as_mad_aligned(eff_mad_m, eff_mad_k);           // 138
```

分工要分清：**对齐后的形状字段决定"搬运/寻址的物理几何"**（`copy_l1_to_l0a` 用 `dst.shape_mn / FRAC_MN` 数分形个数；L0C→UB 用 `src.shape_m` 作 NZ 行距），而 `mad()` 与 epilogue 收到的仍是未对齐的 `eff_mad_m`（第 149、161 行）——计算量和写回行数按有效行截断。`fill_mn_tail_block` 的注释（`copy.hpp:361-362`）把这件事说得很直白："ptr.shape_mn remains the physical NZ stride; padded_mn only bounds the region consumed by the following L1→L0 transfer"（该函数本身是 FP8 路径的显式补零，BF16 靠边界截断）。深入的边界行为留到第 19～20 天。

**`offset_bytes(bytes)`（`tile_ptr.hpp:45-50`）**：字节粒度平移，注释说明用于 NZ 分形布局（元素/stage 步长不适用处）。消费例子：稀疏 MQA 里 `l1kv[kv_stage_idx].offset_bytes(...)`（`sparse_mqa_logits.hpp:331`）、`paged_mqa_logits.hpp:221`。

**`offset_nz_m(m)`（`tile_ptr.hpp:81-86`）**：仅限 `__cc__`（L0C），`addr += m × FRAC_MN × sizeof(dtype_t)`——沿 L0C NZ 布局的 M 方向平移。消费例子：MegaMoE 的 `l0c_mad.offset_nz_m(m_idx_in_block)`（`mega_moe.hpp:622`）。

**`operator+(elems)`（`common.hpp:175-179`）**：`gm_ptr` 的元素粒度平移（`addr += elems × sizeof(dtype_t)`），分页 MQA 用于定位物理页：`kv_t + scheduler.get_kv_page_offset_bytes(first_page)`（`mqa_logits.hpp:356`，`kv_t` 是 `gm_ptr<qk_dtype_t, Major::K>`，267 行）。

### 3.7 完整演算：`M=600, N=1000, K=256`（当天产出·参考答案）

沿用第 15～17 天的假设：Normal BF16 GEMM，`M=600, N=1000, K=256`，**假设**启发式选中 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128`，`MAD_M=128, MAD_N=128, MAD_K=64`，`kNumL1Stages=3, kNumL0Stages=2`，`kNumCores=32`。核 21 在 wave 0 领到 Tile `(m1, n5)`（第 17 天 3.6 节）：`m_idx = 128`，`n_idx = 640`。A/B 均为 BF16（`sizeof = 2`，`pack_size = 1`）。

**第一步：输入 GM 地址（`a.index(m_idx, k_idx)`，k_blk_idx=1 → k_idx=128）。**

| A 的物理布局 | `stride_outer` | `a.index(128, 128)` 字节偏移 | 说明 |
| :-- | :-- | :-- | :--- |
| K-major 紧凑（`[600,256]` 行主序） | 256 | `2×(128×256 + 128) = 65792` | 行主序，行距 256 元素 |
| K-major strided（`[600,320]` 切出 `[:, :256]`） | 320 | `2×(128×320 + 128) = 82176` | 同一逻辑元素，行距变大 |
| MN-major（`[256,600]` 行主序再 `.T`） | 600 | `2×(128×600 + 128) = 153856` | 公式换位：`k×stride + m` |

三行算的都是**同一个逻辑元素 `A[128][128]`**——它落在各自存储的第 `128×256+128 = 32896`、`128×320+128 = 41088`、`128×600+128 = 76928` 个元素上。B 为 `[1000,256]` K-major：`b.index(640, 128) = 2×(640×256 + 128) = 327936` 字节。

**第二步：L1 stage 地址（第 16 天公式的数字版）。**

`l1a = l1_ptr<bfloat16_t, kMajorA>(128, 128, 0)`：`size_per_stage = 128×128×2 = 32768` 字节。

| stage s | `l1a[s]` | `l1b[s]`（基址 = `l1a.offset(3)` = 98304） |
| :-- | :--- | :--- |
| 0 | 0 | 98304 |
| 1 | 32768 | 131072 |
| 2 | 65536 | 163840 |

容量复核（`bf16_gemm.hpp:66`）：`l1b.offset(3) = 196608 ≤ L1SizeBytes = 524288` ✓。

**第三步：L0 分形单元（`copy_l1_to_l0a` 的下标整除）。**

Tile (m1, n5) 内 `m_mad_idx = n_mad_idx = 0`（`MAD_M = BLOCK_M`），`k_mad_idx ∈ {0, 64}`（`MAD_K=64`，`actual_k=128` 两段）。K-major A：

```text
k_mad_idx=0  → 分形单元 (mn 0/16 = 0, k 0/16 = 0)
k_mad_idx=64 → 分形单元 (mn 0/16 = 0, k 64/16 = 4)
l0a_tile 形状 (128, 64) → 8 × 4 个分形单元，占 L0A 2 级共 2×128×64×2 = 32768 ≤ 65536 ✓
```

若 A 是 MN-major，同样的 `(m_mad_idx, k_mad_idx)` 变成 `(k 0/16, mn 0/16)` 与 `(k 64/16, mn 0/16)`，且走 `_transpose` 变体（`copy.hpp:142-143`）——L0 里落点相同。

**第四步：边界 Tile 的 `as_mad_aligned`。**

m 块 4（核 0 的 wave-1 任务）：`m_idx = 512`，`actual_m = min(600−512, 128) = 88`。`eff_mad_m = 88`；`l0c_tile/l0a_tile` 的形状字段被 `as_mad_aligned` 抬到 `aligned(88, 16) = 96`——L1→L0 搬 96 行（6 个分形单元），`mad()` 仍按 88 行计算，epilogue 只写 `min(88, MAD_M) = 88` 行。

**第五步：输出 GM 地址。**

D 为 `[600,1000]` 行主序，`gm_d` 是 `Major::K` 语义（N 连续），`stride_outer = 1000`：

- 普通路径：AIV drain 写 Tile (m1, n5) 的 MAD 块 `(m_idx + 0, n_idx + 0)`：`d.index(128, 640) = 2×(128×1000 + 640) = 257280` 字节，写 `128 行 × 128 列`（该 Tile 的 `actual_m = min(600−128, 128) = 128`、`actual_n = min(1000−640, 128) = 128`，均为整块）；
- 核 0 的 wave-1 Tile (m4, n0)：`d.index(512, 0) = 2×(512×1000 + 0) = 1024000` 字节，写边界块 `88 行 × 128 列`（对照第四步的 `as_mad_aligned`）。

**第六步：group/batch 偏移小例（供对照，Normal 下为恒等）。**

- M-Grouped 的 B（`[G,1000,256]` 紧凑）：`b.group(2, 1000, 256)` → `+2×1000×256×2 = +1024000` 字节；
- K-Grouped 的 D（`[G,600,1000]`）：`d.group(1, 600, 1000)` → `+1×600×1000×2 = +1200000` 字节；
- Batched 的 A（`[B,600,256]` 紧凑，`stride(0) = 153600`）：`a.batch(3)` → `+3×153600×2 = +921600` 字节。

## 四、源码导读

今天的目标是**把 3.2～3.5 的每个公式落到真实代码行上**。每站 5～10 分钟。

### 站点 1：`ascend/tile_ptr.hpp` 全文（102 行，精读）

- 8-16 行：`cvt_pointee` 六个偏特化——逐一确认每个地址空间限定符都被保留；
- 18-56 行：`operand_ptr`。重点：21 行"logical, layout-invariant"注释（形状是逻辑形状，与 L1 物理分形摆放无关）、27 行 `k_per_slot`、33-35 行 `ptr<dst_dtype_t>()`、37-43 行 stage 寻址三件套（第 16 天回顾）、45-50 行 `offset_bytes`、52-55 行 `as_mad_aligned`；
- 58-91 行：`result_ptr`。对比 73 行与 37 行的 `size_per_stage` 差异（无 `k_per_slot`）、81-86 行 `offset_nz_m` 的 `static_assert(std::is_same_v<ptr_t, __cc__ elem_t*>)`；
- 93-97 行：五个别名；99-100 行：`gm_ptr` 在 common.hpp 的指路注释——**今天最容易漏读的一行**。

### 站点 2：`common.hpp` 的 `Major` 与 `gm_ptr`（82-85、137-186 行，精读）

- 82-85 行：`enum class Major : uint8_t { K, MN }` 与两个判定函数；
- 137-186 行：`gm_ptr` 全文。逐行核对 3.2 的公式；注意 150-173 行的 `#ifdef __CCE__` 边界（`ptr/index/group` 设备专用）与 175-185 行（`operator+/batch` 两侧可用）；思考：为什么 `group()` 需要 `shape_mn/shape_k` 两个参数而 `batch()` 只需 idx？（提示：批距随指针携带，组距按"紧凑排列"现场计算。）

### 站点 3：kernel 消费点（`bf16_gemm.hpp`，扫读）

- 28-30 行：三个 `gm_ptr` 参数的 Major 模板实参（`kMajorA/kMajorB`/默认 K）；
- 59-64 行：五个片上句柄声明（第 16 天回顾，注意 `l1_ptr` 的 Major实参就是 `kMajorA/kMajorB`——3.5 节差异(2)(3) 的开关键）；
- 89-103 行：坐标 → `batch()/group()` 句柄派生；
- 109-118 行：`a.index(m_idx, k_idx)` / `b.index(n_idx, k_idx)`（resident A 的 `index(m_idx, 0)` 对照）；
- 127、138-139 行：`as_mad_aligned` 三连；
- 143、145 行：`copy_l1_to_l0a/b` 的 `(m_mad_idx, k_mad_idx)` 与 resident A 的 K 偏移表达式；
- 157-159 行：direct store 的 `d.index(...).ptr()` 与 `d.stride_outer`；
- 188-190 行：AIV 侧同款派生。

### 站点 4：`ascend/copy.hpp` 的 Major 分支（51-64、101-159 行，精读——明天主线，今天只看地址相关参数）

- 58-59 行：`copy_gm_to_ub` 的 rows/cols 互换（同型模式的第二个实例）；
- 101-131 行：`copy_gm_to_l1`。只追三个地址相关量：`src.stride_outer * sizeof(T)`（行距）、`actual_mn/actual_k`（行列互换）、`set_mte2_nz` 的 `dst.shape_mn` vs `dst.shape_k`；
- 133-159 行：`copy_l1_to_l0a/b` 的两个分支——分形下标整除与 `_transpose` 变体；
- 359-368 行：`fill_mn_tail_block` 注释（"physical NZ stride" vs "bounds the region consumed"）——`as_mad_aligned` 语义的官方说法。

### 站点 5：Host 链路（三个文件，精读）

- `csrc/utils/layout.hpp:37-52`：`major_check`（歧义报错）/`get_major_type_ab`/`check_major_type_cd`；
- `csrc/apis/gemm.hpp:79-80、88-96`：Major 判定与两种 Grouped 的 Major 断言；`188-193`：stride 写入 `GemmDesc`；`413-453`：`bf16_gemm` 与四个后缀常量（`nt/tn/nn/tt`，450-453 行）；
- `csrc/jit_kernels/gemm_config.hpp:23-34`：`get_outer_stride`（两侧各断言哪根轴必须连续？）；`62-75`：两组 stride 字段的注释；
- `csrc/jit_kernels/bf16_gemm.hpp:71-77`：`jit->launch` 处三个 `gm_ptr<uint8_t>{...}` 的构造。

### 站点 6：`epilogue.hpp` 写回侧（49-52、59-87、109-141 行，扫读）

- 49-50 行：`aiv_drain(gm_ptr<cd_dtype_t, Major::K> d, ...)` 签名——D 的 Major 在类型上写死；
- 81-82 行：`d.index(block.m_idx, block.n_idx).ptr()` 与 `d.stride_outer * sizeof(cd_dtype_t)`；
- 132 行：K-Grouped 空组清零的 `d.group(...).index(...)` 链式调用。

### 站点 7：Host 镜像（`csrc/jit_kernels/heuristics.hpp:283-304`，验证对号）

`gm_load_bytes` 闭包把 `outer_stride` 翻译成 GM 流量：`stride_bytes = outer_stride ? outer_stride × max(elem_bits,8)/8 : row_bytes`（288 行）——紧凑时按行宽，strided 时按行距；再按 L2 sector（128B）对齐估算每行实际搬运字节（290-294 行）。这说明 **strided 输入不额外搬数据，但会按行距放大 GM 流量**，启发式据此选配置（303-304 行 `wa/wb_block_bytes`）。第 8 周性能分析会回来用这条公式。

**阅读检查点**（能回答再进入实践）：

1. `gm_ptr` 为什么不能放进 `tile_ptr.hpp`？它哪三个方法在 Host 上不存在？
2. `l1_ptr` 带 Major、`l0a_ptr` 不带——各自的理由？
3. `get_outer_stride` 在 K-major/MN-major 下分别返回 `stride(-2)` 还是 `stride(-1)`？另一根轴被什么断言约束？
4. 主例中 MN-major A 的 `a.index(128, 128)` 与 K-major A 的差多少字节？各自对应存储里的第几个元素？

## 五、实践步骤

### A. 有 Ascend 环境

**步骤 1：打印配置并手算地址（约 20 分钟）。**

```bash
cd DeepGEMM-Ascend
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof 2>&1 | tee /tmp/day18_run1.log
grep -E 'GemmDesc|GemmConfig' /tmp/day18_run1.log | head -10
```

挑一个 shape（最好接近 `M=600, N=1000, K=256`），记录打印的 `block_m/n/k、mad_m/n/k`，按 3.7 重算一遍六步（把教程里的假设参数换成真实配置）。若想精确复现主例，可直接运行步骤 2 的脚本。

**步骤 2：三种 A 布局实测（约 20 分钟）。**

```bash
python - <<'EOF' 2>&1 | tee /tmp/day18_run2.log
import torch, torch_npu  # noqa
import deep_gemm

M, N, K = 600, 1000, 256

def check(a, b, d, tag):
    deep_gemm.bf16_gemm_nt(a, b, d)
    torch.npu.synchronize()
    ref = a.float() @ b.float().T
    assert torch.allclose(d.float(), ref, atol=1, rtol=1e-2), tag
    print(f"{tag}: outer_stride_a={a.stride(-2) if a.stride(-1)==1 else a.stride(-1)}, OK")

# ① K-major 紧凑
a = torch.randn(M, K, dtype=torch.bfloat16, device='npu')
# ② K-major strided：从 [M, K+64] 里切出前 K 列，行距变成 K+64
a_big = torch.randn(M, K + 64, dtype=torch.bfloat16, device='npu')
a_strided = a_big[:, :K]
# ③ MN-major：[K, M] 行主序再逻辑转置，K 成为外层
a_mn = torch.randn(K, M, dtype=torch.bfloat16, device='npu').transpose(0, 1)

b = torch.randn(N, K, dtype=torch.bfloat16, device='npu')
for x, tag in ((a, 'K-major'), (a_strided, 'K-major strided'), (a_mn, 'MN-major')):
    print(tag, 'strides =', x.stride())
    d = torch.empty(M, N, dtype=torch.bfloat16, device='npu')
    check(x, b, d, tag)
print("all layouts OK")
EOF
```

预期：三种布局全部通过；打印的 stride 分别为 `(256, 1)`、`(320, 1)`、`(1, 600)`。对照 3.7 第一步的表格——kernel 对三种布局算出的源地址不同，但结果一致。再补一个 **strided D**（`check` 里把 `d` 换成 `torch.empty(M, N+32, ...)[:, :N]`，行距 1032）验证 `outer_stride_d` 路径（`check_major_type_cd` 只要求 `stride(-1)==1`）。

**步骤 3：跑通四种后缀 × Major 组合（约 10 分钟）。**

```bash
python tests/test_bf16_gemm.py normal --skip-prof
```

测试生成器（`tests/generators.py:466-538` 的 normal/strided/unaligned 枚举）自动覆盖 `[Major.K, Major.MN] × [Major.K, Major.MN]`。注意一个**源码事实**：`strided` 场景目前只影响用例标签——`GemmTestDesc.strided` 字段（`generators.py:261`）在 `_generate_normal` 里没有被消费，`make_strided`（238-246 行）是定义了但尚未接线的工具；所以真正的 strided 验证要靠步骤 2 手工构造。

**步骤 4（可选）：观察模板实参（约 10 分钟）。**

```bash
DG_JIT_PRINT_COMPILER_COMMAND=1 python - <<'EOF'
import torch, torch_npu  # noqa
import deep_gemm
a = torch.randn(256, 600, dtype=torch.bfloat16, device='npu').transpose(0, 1)
b = torch.randn(1000, 256, dtype=torch.bfloat16, device='npu')
d = torch.empty(600, 1000, dtype=torch.bfloat16, device='npu')
deep_gemm.bf16_gemm_nt(a, b, d)
EOF
```

在打印的编译命令/实例化代码里找到 `bf16_gemm_impl<...>` 的前两个模板实参（`kMajorA/kMajorB`，`jit_kernels/bf16_gemm.hpp:43-44`），确认 MN-major A 实例化为 `Major::MN`——不同的 Major 生成不同的 kernel 二进制（`copy_l1_to_l0a` 的 `if constexpr` 分支在编译期裁剪）。

### B. 无 Ascend 环境：静态替代任务

**B-1：闭卷重算主例（约 20 分钟）。** 合上教程，只看 `common.hpp:156-172、181-185` 与 `bf16_gemm.hpp:89-118`，对 `M=600, N=1000, K=256`、Tile (m1, n5)、k_blk_idx=1 重算 3.7 的第一、二、五步（GM 输入/输出地址与 L1 stage 地址），对照订正。

**B-2：用 Python 复刻地址计算器并验证（约 25 分钟）。** 补齐三处空格（**对照源码填，不要凭记忆**），然后用"值 = 存储线性偏移"的技巧自检：用 `arange` 构造张量，让每个元素的值恰好等于它在底层存储里的元素偏移，公式算出的偏移应该能直接取到同一元素。

```python
import torch

def index_byte_offset(major, stride_outer, mn_idx, k_idx, elem_bytes=2, pack=1):
    if major == 'K':
        elems = mn_idx * stride_outer + ____            # common.hpp:160
    else:
        elems = ____                                    # common.hpp:162
    return elems * elem_bytes

def group_byte_offset(g, shape_mn, shape_k, elem_bytes=2, pack=1):
    return ____ * elem_bytes                            # common.hpp:170

def batch_byte_offset(idx, stride_batch, elem_bytes=2):
    return ____ * elem_bytes                            # common.hpp:183

# 验证：值即偏移
M, N, K = 600, 1000, 256
base = torch.arange(M * (K + 64)).reshape(M, K + 64)     # K-major strided 存储
a_strided = base[:, :K]                                  # shape (600,256), stride (320,1)
assert index_byte_offset('K', 320, 128, 128) // 2 == a_strided[128, 128].item()

store = torch.arange(K * M).reshape(K, M)                # MN-major 存储
a_mn = store.T                                           # shape (600,256), stride (1,600)
assert index_byte_offset('MN', 600, 128, 128) // 2 == a_mn[128, 128].item()

assert group_byte_offset(2, N, K) == 2 * N * K * 2       # M-Grouped 的 B，第 2 组
assert batch_byte_offset(3, M * K) == 3 * M * K * 2      # Batched 的 A，第 3 批
print("all address checks passed")
```

**B-3：MN vs K 对照推导（约 15 分钟）。** 对 `A[300][77]`（M=600, K=256）分别写出两种布局下的元素偏移表达式并算出数值（K-major stride=256：`300×256+77 = 76877`；MN-major stride=600：`77×600+300 = 46500`）。再用 B-2 的验证技巧各构造一个张量确认。最后回答：为什么两个数值不同却指向同一逻辑元素？

**B-4：strided 演算（约 10 分钟）。** A 由 `[600, 320]` 切片而来（`outer_stride=320`）。算 `a.index(512, 0)`（wave-1 核 0 的 Tile）：`2×(512×320+0) = 327680` 字节；对照紧凑布局的 `2×(512×256) = 262144`。问：多出的字节数意味着什么？（提示：站点 7 的 `gm_load_bytes`——strided 按行距放大 GM 流量，但不改变读到的逻辑数据。）

**B-5：边界 Tile 与分形下标（约 15 分钟）。** 设 `M=600, BLOCK_M=128, MAD_M=128, MAD_K=64, FRAC_MN=16, FRAC_K=16`：(a) 算 m 块 4 的 `actual_m/eff_mad_m/as_mad_aligned` 后的形状字段（88 → 96）；(b) 写出 `copy_l1_to_l0a` 在 `k_mad_idx = 0、64` 时的两个分形单元下标（K-major 与 MN-major 各写一遍，注意 3.5 节的角色互换）；(c) 说明 `mad()` 与 epilogue 分别收到 88 还是 96，为什么最终 D 的第 512～599 行之外不会被写。

**B-6：闭卷画地址链路图（约 15 分钟）。** 凭记忆重画 3.4 节末尾的全链路图（坐标 → index → GM 地址 → stage → 分形单元 → L0C → d.index），并标注每一步的"步长来源"（stride_outer / size_per_stage / FRAC_MN、FRAC_K / d.stride_outer）——订正后的图并入当天产出。

## 六、当天产出

学习计划要求今天的产出是**一组地址计算示例**。按下述模板填写（B-1～B-5 的演算附在后面）：

```text
# DeepGEMM-Ascend 第 18 天记录：理解地址与 Tile 表达

## A. 配置与坐标（承接第 17 天）
M=＿, N=＿, K=＿；BLOCK=＿×＿×＿, MAD=＿×＿×＿, L1 stages=＿, L0 stages=＿
选定的 Tile：m_block=＿, n_block=＿ → m_idx=＿, n_idx=＿（哪个核的哪一跳？）
k_blk_idx=＿ → k_idx=＿；actual_m=＿, actual_n=＿, actual_k=＿

## B. GM 输入地址（a.index / b.index）
A 布局        stride_outer   a.index(m_idx, k_idx) 字节偏移
K-major       ＿             ＿
K-strided     ＿             ＿
MN-major      ＿             ＿
b.index(n_idx, k_idx) = ＿（B 布局：＿＿）

## C. L1 stage 地址
size_per_stage(l1a) = ＿；l1a[＿] = ＿；l1b 基址 = ＿；l1b[＿] = ＿
容量断言：l1b.offset(＿) = ＿ ≤ L1SizeBytes ＿ ✓/✗

## D. L0 分形单元与边界
copy_l1_to_l0a 的分形单元：k_mad_idx=＿ → (＿, ＿)；k_mad_idx=＿ → (＿, ＿)
边界 Tile：eff_mad_m=＿ → as_mad_aligned → ＿；mad 收到 ＿，epilogue 写 ＿ 行

## E. GM 输出地址
d.index(m_idx, n_idx) = ＿（stride_outer_d = ＿）
direct store 与 AIV drain 各自的调用行号：＿＿、＿＿

## F. group/batch（Normal 下恒等，各举一例）
b.group(＿, ＿, ＿) → +＿ 字节；d.group(＿, ＿, ＿) → +＿ 字节；a.batch(＿) → +＿ 字节

## G. 实验记录（有环境时）
三种 A 布局的 stride 打印与正确性结果；strided D 的结果；模板实参观察

## H. 遗留问题（交给后续天）
- copy_gm_to_l1/copy_l1_to_l0a 的指令参数细节（第 19 天）
- actual_m/n/k 与边界搬运、补零的完整行为（第 19～20 天）
- FP8/FP4 的 pack_size 与 SF 寻址（第 32～33 天）
- Grouped/Batched 的组游标与 k_idx_base（第 36～37 天）
```

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. 三族形状化指针分别定义在哪里？`gm_ptr` 为什么必须放在 `common.hpp`？它的哪三个方法在 Host 上不存在，为什么？
2. `gm_ptr` 有哪三个字段？`stride_outer` 的单位是什么？在 K-major 与 MN-major 下各代表沿哪根轴的行距？
3. 写出 `index(mn_idx, k_idx)` 两个分支的字节偏移公式。`pack_size` 何时为 2，它为什么只除在连续轴上？
4. `gm_d` 的模板参数是 `Major::K`，为什么说"D 是 N-major"？`check_major_type_cd` 断言了什么？
5. 完整说出 `outer_stride_a` 从 PyTorch 张量到 `copy_gm_to_l1` 行距参数的链路（至少四个环节、带文件与行号）。
6. `group()` 与 `batch()` 的字节偏移公式是什么？在 `bf16_gemm.hpp` 中找出它们的全部调用点，并说明各服务哪种 `GemmType`。
7. MN-major 与 K-major 对地址计算/搬运的三点差异是什么？为什么 `l0a_ptr` 不带 Major 模板参数？
8. 主例（`M=600, N=1000, K=256`，K-major A、`stride=256`）中，核 21 wave 0 的 Tile (m1, n5) 在 `k_blk_idx=1` 时 `a.index` 的字节偏移是多少？同一个逻辑元素在 MN-major（`stride=600`）存储下偏移是多少？
9. `as_mad_aligned` 修改 tile 句柄的什么、不修改什么？`mad()` 与 epilogue 分别接收对齐前还是对齐后的值？
10. `offset_bytes` 与 `offset_nz_m` 各用于什么场景（各举一个消费文件）？`result_ptr` 为什么不需要 `k_per_slot`？
11. （思考）strided 输入不需要先整理成紧凑布局就能被 kernel 消化——这个设计的代价转移到了哪里？Host 启发式的 `gm_load_bytes`（heuristics.hpp:283-296）如何体现这一点？如果改成"Host 先 `.contiguous()` 再传入"，多出来的开销是什么性质？

**参考答案：**

1. `gm_ptr`（`common.hpp:137-186`，`__gm__`）、`operand_ptr`（`tile_ptr.hpp:18-56`）、`result_ptr`（`tile_ptr.hpp:58-91`）。`gm_ptr` 要在 Host 上构造并作为 launch 参数传递，而 `common.hpp` 是 Host/Device 共享的头（"pure C++, no intrinsics"），`__aicore__/__gm__` 在非 `__CCE__` 编译器上被宏抹掉（`common.hpp:23-26`）。`ptr()/index()/group()` 被 `#ifdef __CCE__` 包住（150-173 行）——它们依赖设备专有的 `is_fp4`（`float4_e2m1x2_t` 只在 bisheng 上定义，67-70 行）与 `__gm__` 指针转型；`operator+/batch()` 不依赖，两侧可编译。
2. `addr`（字节地址）、`stride_outer`、`stride_batch`（后两者以**元素**为单位）。K-major：沿 MN 的行距（= `stride(-2)`）；MN-major：沿 K 的行距（= `stride(-1)`）。
3. K-major：`sizeof(T) × (mn_idx × stride_outer + k_idx / pack_size)`；MN-major：`sizeof(T) × (k_idx × stride_outer + mn_idx / pack_size)`（`common.hpp:159-164`）。`pack_size=2` 当且仅当 `dtype_t` 是 `float4_e2m1x2_t`（FP4 两个 4-bit 打包进一个字节槽）；打包沿连续轴进行，所以只对连续轴下标整除。
4. `Major::K` 的语义是"最后一维连续"；对 `[M, N]` 的 D，最后一维是 N，故 D 是 N-major（行主序）。`index` 的 `k_idx` 槽位装的是列号 n（`bf16_gemm.hpp:158` 的 `d.index(m_idx + m_mad_idx, n_idx + n_mad_idx)`）。`check_major_type_cd` 断言 `d.stride(-1) == 1`（`utils/layout.hpp:49-52`）。
5. `a.stride()` → `get_major_type_ab(a)`（`utils/layout.hpp:44-47`，`stride(-1)==1` 判 K-major，`major_check` 保证恰一根轴连续）→ `get_outer_stride(a, major_a)`（`gemm_config.hpp:27-34`，K-major 取 `stride(-2)` 并断言 `stride(-1)==1`）→ `GemmDesc.outer_stride_a`（`apis/gemm.hpp:191`）→ launch 构造 `gm_ptr<uint8_t>{desc.outer_stride_a, (uintptr_t)desc.a_ptr, ...}`（`jit_kernels/bf16_gemm.hpp:74`）→ kernel 参数 `gm_ptr<bfloat16_t, kMajorA>` → `copy_gm_to_l1` 的 `src.stride_outer * sizeof(T)`（`copy.hpp:123/127`）。
6. `group()`：`group_idx × shape_mn × shape_k / pack_size × sizeof(T)`（`common.hpp:170`）；`batch()`：`idx × stride_batch × sizeof(T)`（183 行）。调用点：`bf16_gemm.hpp:103`（M-Grouped 的 B，配 `b_group_idx`）、157/188（K-Grouped 的 D，配 `d_group_idx`）、102/157/188（Batched 的 `batch(batch_idx)`，三种类型统一调用、Normal 下恒等）；另见 `epilogue.hpp:132`（K-Grouped 空组清零）。
7. ① `index()` 公式换位（外层下标与 stride 的角色互换）；② `copy_gm_to_l1` 的二维搬运行列互换（`actual_mn/actual_k` 对调）且 `set_mte2_nz` 的 NZ 目标参数从 `dst.shape_mn` 换成 `dst.shape_k`（`copy.hpp:120-129`）——L1 分形摆放转置；③ `copy_l1_to_l0a/b` 改用 `_transpose` 变体、分形下标角色互换（`k_idx/FRAC_MN`、`mn_idx/FRAC_K`，`copy.hpp:142-143、156-157`）。L0A/L0B 必须是 MAD 单元要求的固定 Z 分形，Major 在 MTE2/MTE1 两级搬运中被消化，所以 `l0a_ptr/l0b_ptr` 不带 Major（`tile_ptr.hpp:94-95`）。
8. `a.index(128, 128)`：K-major `2×(128×256+128) = 65792` 字节；MN-major `2×(128×600+128) = 153856` 字节。两者都指向逻辑元素 `A[128][128]`，但分别落在各自存储的第 32896 与 76928 个元素上。
9. `as_mad_aligned` 只把 tile 句柄的**形状字段**向上对齐到 `FRAC_MN=16 / FRAC_K`（BF16 为 16），**不改变 `addr`**（`tile_ptr.hpp:52-55`）。对齐后的形状决定分形搬运个数与 L0C 的 NZ 行距等物理几何；`mad()` 收到未对齐的 `eff_mad_m/eff_n/eff_k`（`bf16_gemm.hpp:149`），epilogue 的 `aic_store/aiv_drain` 也按有效行数截断（161 行、`epilogue.hpp:65-66`）。
10. `offset_bytes`：NZ 分形布局下的字节粒度平移（`tile_ptr.hpp:45-50`；消费如 `sparse_mqa_logits.hpp:331`、`paged_mqa_logits.hpp:221`）。`offset_nz_m`：L0C 沿 NZ 布局 M 方向平移 `m × FRAC_MN × sizeof(T)`（81-86 行；消费如 `mega_moe.hpp:622`）。`result_ptr` 描述输出，不存在 FP4 打包（打包只发生在输入的连续 K/MN 轴），故无 `k_per_slot`。
11. 代价从"数据整理搬运"转移到"搬运指令的寻址参数"：同一份 kernel 代码靠随指针携带的 `stride_outer` 支持任意行距（MTE2 的 nd2nz 本来就按 (行数, 行宽, 行距) 描述搬运）。但 strided 会按行距放大 GM 流量与 L2 sector 覆盖——`gm_load_bytes` 里 `stride_bytes = outer_stride × max(elem_bits,8)/8`（heuristics.hpp:288），紧凑时退化成 `row_bytes`，行距与行宽的公共同余决定边界共享（290-294 行）。若 Host 先 `.contiguous()`：多出一次全量 GM 读+写（O(M×K) 数据搬移与额外显存），而 kernel 内按 stride 搬运只多读行距内的"洞"，通常便宜得多；这正是把布局决策留给搬运参数而非预处理的原因。

---

今天结束时，你手里应该有一组覆盖 GM 输入（三种布局）、L1 stage、L0 分形单元、边界对齐与 GM 输出的地址计算示例，并且能从"核 21 领到 Tile (m1, n5)"一路手算到"它正在读 A 的第 65792 字节、写 D 的第 257280 字节"。坐标和地址都齐了，明天（第 19 天）读 `ascend/copy.hpp` 的主线：`copy_gm_to_l1()` 与 `copy_l1_to_l0a/b()` 的指令参数、`actual_m/n/k` 如何裁剪每一次搬运的大小。
