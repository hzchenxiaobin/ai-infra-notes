# Day 6（周六）：TMA 与 Hopper 异步流水线

> **今日目标**：理解 TMA（Tensor Memory Accelerator）的硬件原理，能用 CuTe 构建 TMA descriptor 并用 `cute::TmaCopy` 搬运数据，了解 warp specialization 模式
> **面试考察度**：⭐⭐⭐⭐ 实践级，TMA 是 Hopper+ 的核心特性
> ⚠️ **本日需要 Hopper（sm_90a）及以上硬件**，Ampere 可读但无法运行 TMA 代码

---

### 学习任务 1：TMA 是什么——从 cp.async 到硬件搬运（45 分钟）

#### cp.async 的局限

Ampere 的 `cp.async` 让 gmem→smem 异步，但它仍是**线程级**指令——每个 thread 发一个 load，地址由 thread 计算。当 tile 变大（如 128×128），需要大量 thread 算地址、发指令，地址计算本身成为瓶颈。

| 维度 | cp.async（Ampere） | TMA（Hopper） |
|------|--------------------|---------------|
| 粒度 | 线程级（每 thread 一条指令） | 块级（一条指令搬整个 tile） |
| 地址计算 | thread 软件计算 | 硬件 descriptor 内嵌 |
| 多播 | 无 | 支持（一次广播到多个 SM） |
| 异步 | `cp.async.commit_group` | `cp.async.bulk.tensor` + barrier |
| 占用 SM | 是 | 否（TMA 单元独立） |

#### TMA descriptor 的构建

TMA 需要预先构建一个 **descriptor**（描述符），把源 tensor 的形状/步长/swizzle 编码成一个 128 字节的硬件结构：

```cpp
// CuTe 封装：make_tma_copy
#include <cute/arch/copy_sm90_tma.hpp>

// 1. 源 gmem Tensor
auto gmem_layout = make_layout(make_shape(M, K), make_stride(K, 1));
auto gA = make_tensor(gmem_ptr, gmem_layout);

// 2. 目标 smem Layout（含 swizzle）
auto smem_layout = make_layout(
    make_shape(_128{}, _64{}),
    make_stride(_64{}, _1{}),
    Swizzle<3, 4, 3>{}
);

// 3. 构建 TMA descriptor（编译期 + 运行期混合）
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
// tma_load 内部调用 cuTensorMapEncodeTiled 把 layout 编码成硬件 descriptor

// 4. 用 TMA 搬运（一条指令搬整个 128×64 tile）
auto sA = make_tensor(make_smem_ptr(smem), smem_layout);
copy(tma_load, gA_tile, sA);
// → 硬件自动算地址、应用 swizzle、发起异步搬运
```

> 💡 **关键洞察**：TMA 把"地址计算 + 搬运"从 thread 卸载到硬件。一个 threadblock 只需一条 TMA 指令就能搬整个 tile，腾出 thread 做计算。这是 Hopper 能跑 WGMMA（warp group 级矩阵乘）的前提——WGMMA 直接从 smem 读，TMA 负责 gmem→smem，计算与搬运完全解耦。

### 学习任务 2：TMA 流水线与 barrier（60 分钟）

TMA 搬运是异步的，需要 `mbarrier` 同步。CuTe 封装了完整的流水线：

```cpp
#include <cute/arch/copy_sm90_desc.hpp>
#include <cuda/barrier>

// 1. 创建 mbarrier（每 stage 一个）
__shared__ uint64_t mbar[3];   // 3-stage
if (threadIdx.x == 0) {
    for (int i = 0; i < 3; ++i)
        cute::initialize_barrier(mbar[i], 1);   // arrival count = 1（TMA 单线程发起）
}
__syncthreads();

// 2. 发起 TMA load + 设置 barrier
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout);
if (threadIdx.x == 0) {   // 只需 1 个 thread 发起 TMA
    auto sA = make_tensor(make_smem_ptr(smem_buf[0]), smem_layout);
    cute::copy(tma_load, gA_tile_k0, sA);
    cute::set_barrier_transaction_bytes(mbar[0], expected_bytes);  // 告知 barrier 字节数
    cute::arrive_and_wait(mbar[0]);   // TMA 完成后 barrier 释放
}

// 3. 主循环：计算 stage i，发起 stage i+2
for (int k = 0; k < K_tiles; ++k) {
    int stage = k % 3;
    cute::wait_barrier(mbar[stage]);   // 等 TMA 完成
    // ... WGMMA(sA[stage], sB[stage]) ...
    cute::arrive_barrier(mbar[stage]); // 释放 buffer

    if (k + 2 < K_tiles) {
        if (threadIdx.x == 0) {
            auto sA_next = make_tensor(make_smem_ptr(smem_buf[(k+2)%3]), smem_layout);
            cute::copy(tma_load, gA_tile_kplus2, sA_next);
            cute::arrive_barrier(mbar[(k+2)%3]);
        }
    }
}
```

### 学习任务 3：Warp Specialization 简介（30 分钟）

Hopper 引入 **warp specialization**——不同 warp 干不同事：

| Warp | 角色 | 代码 |
|------|------|------|
| **Producer warp** | 发起 TMA load | `copy(tma_load, ...); arrive(bar)` |
| **Consumer warpgroup** | 做 WGMMA 计算 | `wait(bar); wgmma(...)` |
| **DMA warp** | 专门搬数据 | 与计算 warp 完全分离 |

```cpp
// 伪代码：warp specialization 结构
if (warp_id == 0) {
    // Producer: 专职发起 TMA
    for (int k = 0; k < K_tiles; ++k) {
        cute::copy(tma_load, gA_k, sA[k%3]);
        cute::arrive(mbar[k%3]);
    }
} else {
    // Consumer: 专职计算
    for (int k = 0; k < K_tiles; ++k) {
        cute::wait(mbar[k%3]);
        wgmma(sA[k%3], sB[k%3], accum);
    }
}
```

> 💡 **一句话总结**：Warp specialization 把"搬数据"和"算数据"彻底分离到不同 warp，每个 warp 只干一件事，流水线深度可做到 4-8 stage。这是 Hopper GEMM 能达到峰值带宽的核心调度模式，CuTe 的 `TensorPipeline` + `collective/mainloop.hpp` 是其标准实现。

### 今日检查清单
- [ ] 能说出 TMA 与 cp.async 的 4 点核心区别
- [ ] 能用 `make_tma_copy` 构建 TMA descriptor
- [ ] 能用 mbarrier 实现 TMA 3-stage 流水线
- [ ] 理解 warp specialization 的 producer/consumer 分工
- [ ] （Hopper 硬件）跑通一个 TMA 搬运示例

---

