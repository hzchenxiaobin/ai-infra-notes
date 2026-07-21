# Day 5（周五）：Swizzle 作为 Layout

> **今日目标**：从"swizzle 是个黑盒优化"升级到"swizzle 就是 Layout 的一部分"，能手推 XOR swizzle 映射，理解 MMA swizzle pattern
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，Swizzle 是 CuTe 区别于其他框架的标志性设计

---

### 学习任务 1：Swizzle 的本质——它就是 Layout（45 分钟）

#### 传统理解 vs CuTe 理解

| 传统理解 | CuTe 理解 |
|----------|-----------|
| Swizzle 是 smem 地址变换函数 | Swizzle **就是** Layout 的一部分 |
| 与 Layout 分离，独立应用 | `Layout<Shape, Stride, Swizzle>` 三元组 |
| 改 swizzle 要改搬运代码 | 改 swizzle 只改 Layout，搬运代码不变 |

```cpp
// CuTe 的 Swizzle 作为 Layout 第三参数
using Swizzle = Swizzle<3, 4, 3>;  // <M, S, B> 三参数
// 含义：把 (coord >> B) XOR (coord >> (B+S)) 的低 M 位混入地址

auto layout = make_layout(
    make_shape(_128{}, _8{}),
    make_stride(_8{}, _1{}),
    Swizzle{}                           // ← Swizzle 作为 Layout 的一部分
);
```

#### Swizzle 的数学定义

CuTe 的 `Swizzle<M, S, B>` 定义为：

$$\mathrm{swizzle}(\mathrm{offset}) = \mathrm{offset} \oplus ((\mathrm{offset} \gg B) \& ((1 \ll M) - 1) \ll (B+S-M))$$

直观理解：取 offset 的 `[B, B+S)` 位（共 S 位），取其低 M 位，XOR 回 offset 的 `[B+S-M, B+S)` 位。

```cpp
// Swizzle<3, 4, 3>: M=3, S=4, B=3
// 取 offset 的 [3, 7) 位（4 位），取低 3 位，XOR 到 [4, 7) 位
// 例：offset = 0b10101 (21)
//   [3,7) 位 = 0b10 (2)
//   低 3 位 = 0b010
//   XOR 到 [4,7): 21 ^ (0b010 << 4) = 21 ^ 32 = ... 
//   （具体手算留给实验）
```

> 💡 **关键洞察**：Swizzle 是**确定性**的——给定 offset，swizzle 后的 offset 唯一。它不是随机化，而是精心设计的 XOR 映射，目标是让相邻 thread 访问的 smem 元素落在不同 bank，消除 bank conflict。

### 学习任务 2：Swizzle 消除 bank conflict 的原理（60 分钟）

#### 问题：MMA 对 smem 的访问模式

Ampere `mma.m16n8k16` 指令要求 warp 从 smem 读取 16×16 的 A fragment。32 个 thread 的访问模式是固定的（由 PTX 规定），如果 smem 用朴素行主序布局，某些 thread 会访问同一 bank → bank conflict。

![CuTe Swizzle 消除 Bank Conflict 原理](../images/cute_swizzle_bank_conflict.svg)

> **图：** 左侧朴素布局下，多个 thread 访问同一列（同 bank）→ 串行化。右侧 Swizzle 把 thread→地址的映射"打乱"，使相邻 thread 落在不同 bank。关键：Swizzle 是 Layout 的一部分，搬运代码（`copy`）自动应用，无需手改。

#### 实践：手算一个 8×8 矩阵的 swizzle

```cpp
// 8×8 smem 矩阵，每元素 4B，共 32 bank（每 bank 4B）
// 朴素布局：row i, col j → offset = i*8 + j
//   thread (i, j) 和 thread (i, j+8) 访问同一 bank（如果有的话）

// Swizzle<3, 3, 3>: M=3, S=3, B=3
//   swizzle(i*8 + j) = (i*8 + j) ^ (((i*8 + j) >> 3) & 0b111 << 3)
//   简化：因为 i*8 是 8 的倍数，(i*8+j)>>3 = i + (j>>3) = i (j<8)
//   所以 swizzle(i*8 + j) = (i*8 + j) ^ (i << 3) = (i ^ i)*8 + j = j
//   → 把行主序变成列主序！
```

这就是 Swizzle 的魔力——一个 XOR 操作把行主序变成列主序（或更复杂的混合布局），让 MMA 的固定访问模式不再冲突。

### 学习任务 3：MMA Swizzle Pattern（45 分钟）

不同 MMA 指令对 fragment 布局要求不同，CuTe 预定义了对应 Swizzle：

| MMA 指令 | 推荐 Swizzle | 说明 |
|----------|--------------|------|
| `mma.m16n8k16` (Ampere) | `Swizzle<3, 4, 3>` | 8×8 子块内 XOR |
| `mma.m16n8k32` (INT8) | `Swizzle<3, 4, 3>` | 同上 |
| `wgmma.m64n16k16` (Hopper) | `Swizzle<3, 4, 3>` 或 `Swizzle<3, 3, 3>` | 依赖 N 维度 |
| TMA 加载 | `Swizzle<3, 4, 3>` 配合 `cp.async` | TMA descriptor 内嵌 swizzle |

```cpp
// 实际用法：在 smem Layout 中嵌入 Swizzle
auto smem_layout = make_layout(
    make_shape(_128{}, _8{}),
    make_stride(_8{}, _1{}),
    Swizzle<3, 4, 3>{}                   // ← MMA 友好的 swizzle
);

// copy 自动应用 swizzle
auto gA = make_tensor(gmem_ptr, gmem_layout);
auto sA = make_tensor(make_smem_ptr(smem), smem_layout);
copy(gA, sA);   // ← 写入时自动 swizzle，读取时自动反 swizzle

// MMA wrapper 期望的 smem Layout 就是带 swizzle 的
// 所以 copy + MMA 无缝衔接，无需手动转换
```

> ⚠️ **注意**：Swizzle 是**幂等**的——`swizzle(swizzle(x)) = x`。所以"写入时 swizzle"和"读取时反 swizzle"是同一个操作，这就是为什么 `copy` 双向都能自动处理。

### 学习任务 4：动手实验（30 分钟）

创建 `kernels/cute_swizzle_demo.cu`，验证 Swizzle 消除 bank conflict：

```cpp
// cute_swizzle_demo.cu —— Swizzle 作为 Layout 实验
// 编译: nvcc -o cute_swizzle_demo cute_swizzle_demo.cu \
//        -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // 1. 朴素 Layout vs Swizzle Layout
    auto plain  = make_layout(make_shape(_8{}, _8{}), make_stride(_8{}, _1{}));
    auto swizzled = make_layout(make_shape(_8{}, _8{}), make_stride(_8{}, _1{}),
                                Swizzle<3, 3, 3>{});

    // 2. 打印每个 (i,j) 的物理 offset
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("(%d,%d)->plain=%2d swizzled=%2d  ",
                   i, j, (int)plain(i, j), (int)swizzled(i, j));
        }
        printf("\n");
    }

    // 3. 验证：swizzled 的 bank（offset % 32）分布更均匀
    return 0;
}
```

### 今日检查清单
- [ ] 能说出 CuTe 中 Swizzle 是 Layout 的第三参数（`Layout<Shape, Stride, Swizzle>`）
- [ ] 能手算 `Swizzle<M, S, B>` 对某个 offset 的变换结果
- [ ] 能解释 Swizzle 消除 bank conflict 的原理（XOR 打散同 bank 访问）
- [ ] 知道 Swizzle 是幂等的（`swizzle(swizzle(x)) = x`），所以 copy 双向自动处理
- [ ] `cute_swizzle_demo.cu` 编译运行，验证 bank 分布

---

