# Day 2（周二）：Layout 代数深入

> **今日目标**：从"会调 `make_layout`"升级到"能手算嵌套 Layout 偏移、理解 Layout 的代数运算（coalesce/zipped/composed/partition）"
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，CuTe 面试必问"Layout 的代数性质"

---

### 学习任务 1：Layout 作为整数函数（45 分钟）

#### Layout 的数学定义

Layout $L$ 是一个从逻辑坐标 $\mathbf{c}$ 到物理偏移 $o$ 的函数：

$$o = L(\mathbf{c}) = \sum_i c_i \cdot \mathrm{stride}_i, \qquad \mathbf{c} \in [0, \mathrm{shape}_i)$$

由 **Shape**（每维元素数）和 **Stride**（每维步长）共同定义。关键是：Shape 和 Stride **都可以是嵌套的**（`Shape<Shape<_4,_2>, _8>`）。

```cpp
// 基础 Layout：4×4 列主序
auto L = make_layout(make_shape(4, 4), make_stride(1, 4));
// L(2, 3) = 2*1 + 3*4 = 14

// 嵌套 Layout：把 4×4 看成 (2,2) × 4 的分块
auto Ln = make_layout(
    make_shape(make_shape(_2{}, _2{}), _4{}),           // ((2,2), 4)
    make_stride(make_stride(_1{}, _2{}), _4{})          // ((1,2), 4)
);
// Ln(make_coord(make_coord(0, 1), 3)) = 0*1 + 1*2 + 3*4 = 14（同上）
```

> 💡 **关键洞察**：嵌套 Layout **不改变物理偏移**，只改变逻辑视图。同一个内存，你可以用平铺 Layout 看成 4×4，也可以用嵌套 Layout 看成 2×2×4——后者天然表达"分块"语义，是 CuTe 适配 GEMM tiling 的根本机制。

### 学习任务 2：Layout 的代数运算（60 分钟）

这是 CuTe Layout 区别于普通 shape/stride 对的核心。源码在 `layout.hpp`，关键函数：

| 运算 | 函数 | 含义 | 用途 |
|------|------|------|------|
| **coalesce** | `coalesce(L)` | 把多维 Layout 化简为等价的一维 | 判断能否向量化加载 |
| **zipped** | `zipped(L)` | 把嵌套结构压平为 (inner, outer) | smem 访问优化 |
| **tiled** | `tiled(L, tile)` | 在 L 上重复 tile | threadblock tiling |
| **flatten** | `flatten(L)` | 完全压平为 1D | 调试输出 |
| **slice** | `slice(L, coord)` | 取固定坐标的子 Layout | 取某行/某列 |
| **compose** | `compose(L1, L2)` | 函数复合 $L_1 \circ L_2$ | 自定义坐标映射 |

#### coalesce 的手算示例

```cpp
auto L = make_layout(make_shape(4, 8), make_stride(8, 1));
// 行主序，行内连续 → coalesce 后变成 (32,) 一维
auto Lc = coalesce(L);
// Lc == make_layout(make_shape(32), make_stride(1))
// → 说明可以一次性 copy 32 个连续元素（向量化）

auto L2 = make_layout(make_shape(4, 8), make_stride(1, 4));
// 列主序，行内 stride=1 也连续 → coalesce 后也是 (32,)
auto L2c = coalesce(L2);

auto L3 = make_layout(make_shape(4, 8), make_stride(2, 8));
// stride=2 不连续 → coalesce 不动，保持 (4, 8)
// → 不能直接向量化，需逐元素访问
```

> ⚠️ **面试高频坑**：`coalesce` 只在"相邻逻辑坐标对应相邻物理偏移"时才化简。判断条件是 stride 沿某维为 1 且该维连续。手算时画出 (coord → offset) 表，看 offset 是否连续。

### 学习任务 3：composed Layout——函数复合（45 分钟）

`compose(L1, L2)` 把两个 Layout 复合为 $L_1 \circ L_2$，即 $L_1(L_2(\mathbf{c}))$。这是 CuTe 表达"自定义坐标映射"的核心机制：

```cpp
// L1: 8×8 行主序矩阵
auto L1 = make_layout(make_shape(8, 8), make_stride(8, 1));
// L2: 把 64 个线程映射到 (thread_id // 8, thread_id % 8)
auto L2 = make_layout(make_shape(8, 8), make_stride(8, 1));

auto Lc = compose(L1, L2);
// Lc(thread_id) = L1(L2(thread_id)) = 直接得到 thread_id 对应的物理偏移
// → 这就是 CuTe 把"线程映射到数据"做成 Layout 复合的方式
```

#### 实践：用 composed Layout 表达 GEMM 的 warp 划分

GEMM 中一个 threadblock 处理 `128×128` 的 tile，内部 4 个 warp 各处理 `64×64`。用 CuTe：

```cpp
// tile 布局：128×128 行主序
auto tile_layout = make_layout(make_shape(_128{}, _128{}), make_stride(_128{}, _1{}));
// warp 划分：(4, 4) 个 warp，每个 32×32
auto warp_partition = make_layout(
    make_shape(make_shape (_4{}, _32{}), make_shape(_4{}, _32{})),
    make_stride(make_stride(_32{}, _128{}), make_stride(_8{}, _1{}))  // 每个 warp 跳 32×32
);
auto warp_tile = compose(tile_layout, warp_partition);
// → warp_tile(warp_id_m, warp_id_n, thread_m, thread_n) 直接给物理偏移
```

> 💡 **一句话总结**：CuTe 的 Layout 复合让你把"全局 tile → warp tile → thread tile"层层映射表达为函数复合，不用手算任何 `offset = warp_id * 32 * 128 + ...`。CUTLASS 3.x 的 `collective/mainloop.hpp` 里全是这种 `compose`。

### 学习任务 4：动手实验（30 分钟）

在 `kernels/` 下创建 `cute_layout_algebra.cu`：

```cpp
// cute_layout_algebra.cu —— Layout 代数实验
// 编译: nvcc -o cute_layout_algebra cute_layout_algebra.cu \
//        -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
#include <cute/tensor.hpp>
#include <iostream>
using namespace cute;

int main() {
    // 1. 基础 Layout 与 coalesce
    auto L = make_layout(make_shape(4, 8), make_stride(8, 1));
    std::cout << "L         = " << L << "\n";
    std::cout << "coalesce  = " << coalesce(L) << "\n";   // 应为 (32,):(1,)

    // 2. 嵌套 Layout：手算 (1,1) 的偏移
    auto Ln = make_layout(
        make_shape(make_shape(_2{}, _2{}), _4{}),
        make_stride(make_stride(_1{}, _2{}), _4{})
    );
    std::cout << "Ln        = " << Ln << "\n";
    std::cout << "Ln((1,1),2) = " << Ln(make_coord(make_coord(1,1), 2)) << "\n";  // 1+2+8=11

    // 3. composed Layout
    auto L1 = make_layout(make_shape(8, 8), make_stride(8, 1));
    auto L2 = make_layout(make_shape(4, 4), make_stride(2, 2));  // 取每隔一个
    auto Lc = compose(L1, L2);
    std::cout << "compose(L1,L2)(0,0) = " << Lc(0, 0) << "\n";   // L1(0,0)=0
    std::cout << "compose(L1,L2)(1,1) = " << Lc(1, 1) << "\n";   // L1(2,2)=18

    return 0;
}
```

### 今日检查清单
- [ ] 能写出 Layout 的数学定义（Coord → offset 的整数函数）
- [ ] 能手算嵌套 Layout `((2,2),4)` 在坐标 `((1,1),2)` 的偏移
- [ ] 能解释 `coalesce` 何时化简、何时不化简（连续 stride 条件）
- [ ] 能用 `compose` 表达"warp → thread tile"的坐标映射
- [ ] `cute_layout_algebra.cu` 编译运行，输出与手算一致

---

