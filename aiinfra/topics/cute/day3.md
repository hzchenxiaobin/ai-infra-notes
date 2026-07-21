# Day 3（周三）：Tensor 引擎分层

> **今日目标**：理解 Tensor = Engine + Layout 的双模板设计，掌握 gmem/smem/rmem 三种 engine 的区别与创建方式，能用切片/分区操作 Tensor
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，Tensor engine 是 CuTe 性能优化的基础

---

### 学习任务 1：Tensor 的双模板结构（45 分钟）

#### Tensor = Engine + Layout

CuTe 的 `Tensor` 是一个双参数模板：

```cpp
template <typename Engine, typename Layout>
class Tensor;
```

- **Engine**：负责"数据在哪"——指针、smem 句柄、register 数组
- **Layout**：负责"怎么访问"——Shape + Stride

这种分离让同一个 Layout 能用于不同存储层级：

| Engine 类型 | 创建方式 | 数据位置 | 典型用途 |
|------------|----------|----------|----------|
| **gmem** | `make_tensor(ptr, layout)` | Global Memory | 输入输出 |
| **smem** | `make_tensor(make_smem_ptr(ptr), layout)` | Shared Memory | tile 缓存 |
| **rmem** | `make_tensor(reg_array, layout)` | Register | MMA 输入输出 |
| **view** | `make_tensor(tensor, sub_layout)` | 同源 | 切片/分区 |

![CuTe Tensor 引擎分层：gmem/smem/rmem 三级](../images/cute_tensor_binding.svg)

> **图：** Tensor 的 Engine 决定数据驻留在哪一层存储。gmem→smem→rmem 是数据搬运的标准路径，每一层都用相同 Layout 接口访问，copy 原语自动处理层间差异。

#### 源码精读：`tensor.hpp` 中的 make_tensor 重载

```cpp
// 重载 1：裸指针（gmem 或 host）
template <typename T, typename Layout>
auto make_tensor(T* ptr, Layout layout);

// 重载 2：smem 指针（带 cuda::aligned 静态断言）
template <typename T, typename Layout>
auto make_tensor(cute::smem_ptr<T> ptr, Layout layout);

// 重载 3：register 数组（用于 MMA fragment）
template <typename RegArray, typename Layout>
auto make_tensor(RegArray&& reg_arr, Layout layout);  // RegArray 通常是 float[4] 等

// 重载 4：从已有 Tensor 切片（view，不拷贝）
template <typename Engine, typename Layout, typename... Coords>
auto make_tensor(Tensor<Engine, Layout> const& t, Coords... coords);
```

> 💡 **关键洞察**：所有 `make_tensor` 重载返回的 `Tensor<Engine, Layout>` 类型不同（Engine 不同），但访问语法 `tensor(i, j)` 完全相同。这就是 CuTe 让"gmem/smem/rmem 用同一套代码"的根本——类型不同，接口统一。

### 学习任务 2：Tensor 切片与分区（60 分钟）

#### 切片（slice）：取子 Tensor

```cpp
auto A = make_tensor(ptr, make_layout(make_shape(128, 64), make_stride(64, 1)));

// 取第 5 行（保留第 1 维，第 0 维固定为 5）
auto row5 = A(make_coord(5, _));
// row5 是 Tensor<Engine, Layout<Shape<64>, Stride<1>>>，指向同一内存

// 取第 3 列
auto col3 = A(make_coord(_, 3));

// 取一个 16×16 的子块（range 切片）
auto block = A(make_range(0, 16), make_range(0, 16));
```

#### 分区（partition）：把 tile 切给 warp/thread

这是 GEMM 中最常用的操作——把一个 threadblock tile 切成每个 warp 各自处理的子 tile：

```cpp
// threadblock 处理 128×64 的 A tile
auto A_tb = make_tensor(smem_ptr_A, make_layout(make_shape(_128{}, _64{}), ...));

// 4 个 warp，每个处理 32×64
auto warp_layout = make_layout(make_shape(_4{}, _1{}), make_stride(_32{}, _1{}));  // 4 warps 沿 M
// 用 local_partition 把 A_tb 按 warp_layout 切给当前 warp
auto A_w = local_partition(A_tb, warp_layout, warp_id);
// A_w 的 shape 变成 (32, 64)，访问 A_w(m, n) 自动定位到当前 warp 的子 tile

// 再切给 thread
auto thread_layout = make_layout(make_shape(_32{}, _1{}), make_stride(_1{}, _1{}));  // 32 threads 沿 M
auto A_t = local_partition(A_w, thread_layout, thread_id_in_warp);
```

> ⚠️ **注意**：`local_partition` 与 `slice` 的区别——`slice` 取固定坐标的子集，`local_partition` 按"线程/warp 的逻辑布局"切分，返回的是"当前线程/warp 拿到的那一份"。它是 CuTe 把硬件层级映射到数据切片的核心 API。

### 学习任务 3：registered Tensor 与 MMA fragment（45 分钟）

MMA 指令要求输入在 register 中，且布局必须匹配指令的 fragment 形状（如 `m16n8k16` 要求 A fragment 是 16×16）。CuTe 用 `make_tensor(reg_array, layout)` 直接表达：

```cpp
// Ampere mma.m16n8k16 的 A fragment
// 每 warp 32 个 thread，每个 thread 持有 8 个 FP16 元素（2×4 排布）
float A_frag[8];  // 实际是 cutlass::half_t[8]
auto rA = make_tensor(A_frag, make_layout(make_shape(_2{}, _4{}), make_stride(_4{}, _1{})));
// 现在 rA(i, j) 直接访问 fragment 的第 (i,j) 元素
// CuTe 的 MMA wrapper 会把这个 Tensor 直接喂给 mma.sync 指令
```

#### 与 CUTLASS 2.x 的 fragment 对比

```cpp
// CUTLASS 2.x：用 wmma::fragment，布局不透明
wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, row_major> a_frag;
// 你不知道哪个 thread 持有哪个元素，只能整体 load/store/compute

// CuTe：fragment 就是 Tensor<Engine=rmem, Layout>
// 每个 element 的位置透明，可以任意 slice/partition
```

> 💡 **一句话总结**：CuTe 把 MMA fragment 也统一为 `Tensor<rmem, Layout>`，让"warp 级 MMA"与"threadblock 级 tile"用同一套 slice/partition 接口。这是 CUTLASS 3.x 能用 CuTe 拼出 WGMMA 流水线的关键。

### 学习任务 4：动手实验（30 分钟）

创建 `kernels/cute_tensor_engines.cu`，验证 gmem/smem/rmem 三种 Tensor 的创建与切片：

```cpp
// cute_tensor_engines.cu —— Tensor engine 分层实验
// 编译: nvcc -o cute_tensor_engines cute_tensor_engines.cu \
//        -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
using namespace cute;

__global__ void test_engines(float* gmem_ptr) {
    // 1. gmem Tensor
    auto G = make_tensor(gmem_ptr, make_layout(make_shape(8, 8), make_stride(8, 1)));

    // 2. smem Tensor
    __shared__ float smem[64];
    auto S = make_tensor(make_smem_ptr(smem), make_layout(make_shape(8, 8), make_stride(8, 1)));

    // 3. rmem Tensor（register 数组）
    float reg[4];
    auto R = make_tensor(reg, make_layout(make_shape(2, 2), make_stride(2, 1)));

    // 4. 切片：取 G 的第 0 行
    auto G_row0 = G(make_coord(0, _));
    printf("G_row0 shape: ");
    print(G_row0.layout());
    printf("\n");

    // 5. 把 G 的数据 copy 到 S（同 Layout，自动向量化）
    copy(G, S);
}

int main() {
    float* d;
    cudaMalloc(&d, 64 * sizeof(float));
    test_engines<<<1, 1>>>(d);
    cudaDeviceSynchronize();
    return 0;
}
```

### 今日检查清单
- [ ] 能说出 `Tensor<Engine, Layout>` 的双模板结构与 Engine 的四种类型
- [ ] 能解释 `make_tensor` 的四个重载分别对应什么场景
- [ ] 能用 `local_partition` 把 threadblock tile 切给 warp/thread
- [ ] 理解 CuTe 把 MMA fragment 统一为 `Tensor<rmem, Layout>` 的意义
- [ ] `cute_tensor_engines.cu` 编译运行通过

---

