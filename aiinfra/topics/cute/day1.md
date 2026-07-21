# Day 1（周一）：CuTe 总览与独立编译环境

> **今日目标**：理解 CuTe 作为独立 kernel 组装框架的定位，脱离 CUTLASS GEMM 模板编译第一个纯 CuTe 程序，建立 `include/cute/` 源码地图
> **面试考察度**：⭐⭐⭐ 了解级，能说清 CuTe 与 CUTLASS 的关系、为什么 CuTe 能脱离 GEMM 独立使用

---

### 学习任务 1：CuTe 是什么——从" CUTLASS 的子模块"到"独立 kernel 框架"（45 分钟）

#### 阅读内容
- **官方定位**：[CUTLASS 3.0 发布博客](https://developer.nvidia.com/blog/cutlass-3-0/) 中 "CuTe: A New Programming Model" 一节
- **源码入口**：`include/cute/README.md`（CuTe 自带的简短说明）
- **对比阅读**：回顾 [CUTLASS 专题 Day 2](../cutlass/day2.md) 的 CuTe 入门内容

#### 核心要点

CuTe（CUTLASS Tensor Engine）在 CUTLASS 3.0 中引入，但它**不是 GEMM 专用**的——它是一套通用的 GPU kernel 组装原语，核心是三个解耦的抽象：

| 抽象 | 本质 | 解决的问题 |
|------|------|-----------|
| **Layout** | `Coord → offset` 的整数函数 | 索引计算（"第 i 行第 j 列在内存哪里"） |
| **Tensor** | `指针 + Layout` 的绑定 | 数据访问（"用逻辑坐标读写数据"） |
| **Copy** | 根据源/目标 Layout 自动选择最优搬运策略 | 数据搬运（gmem↔smem↔rmem 的向量化与异步） |

> 💡 **关键洞察**：这三个抽象与 GEMM 完全无关——你可以用 CuTe 写 softmax、reduction、attention、conv，任何需要"分块 + 搬运 + 计算"的 kernel。CUTLASS 3.x 的 GEMM 只是用 CuTe 原语拼出来的一个特例。这就是为什么本专题能脱离 CUTLASS GEMM 模板独立学习。

#### CuTe 与 CUTLASS 2.x 索引方式对比

```cpp
// CUTLASS 2.x 风格：手写索引，每种布局都要特化
template <typename LayoutA>
struct PredicatedTileAccessIterator;
// RowMajor / ColumnMajor / AffineRankN 各写一个，模板参数爆炸

// CuTe 风格：Layout 是一等公民，索引统一为 layout(coord)
auto A = make_tensor(ptr_A, make_layout(shape, stride));
float v = A(i, k);  // 不管什么布局，访问代码完全一样
```

| 维度 | CUTLASS 2.x 索引 | CuTe 索引 |
|------|------------------|-----------|
| 抽象层级 | 布局特化迭代器 | Layout 函数 + Tensor 重载 |
| 代码量 | 每种布局一个迭代器类（数百行） | 一套 `tensor(i,j)` 通用 |
| 可组合性 | 差（嵌套要重写） | 强（Layout 可任意嵌套/合成） |
| Swizzle | 散落在各迭代器 | Swizzle **就是** Layout 的一部分 |

### 学习任务 2：独立编译环境（45 分钟）

CuTe 是 header-only，只需 include `cute/tensor.hpp` 即可，**不依赖** `cutlass/gemm/`。

```bash
# 验证 CuTe 可独立编译（不引入 cutlass/gemm）
cd ${CUTLASS_ROOT}

cat > /tmp/cute_standalone.cu << 'EOF'
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(4, 4), make_stride(1, 4));
    printf("layout(2,3) = %d\n", (int)layout(2, 3));  // 14
    return 0;
}
EOF

nvcc -o /tmp/cute_standalone /tmp/cute_standalone.cu \
     -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
/tmp/cute_standalone
# 预期输出：layout(2,3) = 14
```

> ⚠️ **架构参数**：Day 5/6 的 TMA 与 WGMMA 内容必须用 `-arch=sm_90a`（Hopper）或更高。Ampere（`sm_80`）可跑 Day 1-4，但 TMA 相关编译会报错。建议全程用 `sm_90a`。

### 学习任务 3：建立 `include/cute/` 源码地图（30 分钟）

```
include/cute/
├── tensor.hpp              # ★ Tensor 抽象：make_tensor、operator()、切片
├── layout.hpp              # ★ Layout 抽象：make_layout、Shape、Stride
├── swizzle_layout.hpp      # ★ Swizzle 作为 Layout（Day 5 核心）
├── copy.hpp                # ★ copy/copy_if 原语（Day 4 核心）
├── copy_sms.hpp            #   smem→smem copy（register 转置等）
├── tensor_pipeline.hpp     #   流水线封装（double/triple buffer）
├── arch/                   #   硬件指令抽象
│   ├── copy_sm80.hpp       #     cp.async（Ampere）
│   ├── copy_sm90_tma.hpp   #     TMA（Hopper，Day 6 核心）
│   ├── mma_sm80.hpp        #     Ampere MMA
│   └── mma_sm90_gmma.hpp   #     WGMMA（Hopper）
├── atom/                   #   MMA atom：最小可复用计算单元
└── pointer.hpp             #   指针抽象（int4/float4 vectorized）
```

#### 本周精读文件优先级

| 文件 | Day | 优先级 | 内容 |
|------|-----|--------|------|
| `layout.hpp` | 2 | ⭐ 必读 | Layout 代数：coalesce、zipped、composed |
| `tensor.hpp` | 3 | ⭐ 必读 | Tensor engine 分层、切片 |
| `copy.hpp` | 4 | ⭐ 必读 | copy 原语与向量化选择 |
| `swizzle_layout.hpp` | 5 | ⭐ 必读 | Swizzle 作为 Layout |
| `arch/copy_sm90_tma.hpp` | 6 | 📌 推荐 | TMA descriptor |

### 今日检查清单
- [ ] 能说出 CuTe 三大抽象（Layout/Tensor/Copy）各自解决的问题
- [ ] 能解释 CuTe 为何能脱离 CUTLASS GEMM 模板独立使用
- [ ] `/tmp/cute_standalone.cu` 不依赖 `cutlass/gemm/` 编译运行通过
- [ ] 浏览了 `include/cute/` 目录，标记了本周精读文件
- [ ] 把 `examples/cute/` 目录的 00-05 示例名抄到笔记里（Day 7 对标用）

---

