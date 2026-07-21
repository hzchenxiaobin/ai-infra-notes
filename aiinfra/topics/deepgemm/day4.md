# Day 4（周四）：Warp Specialization、寄存器重配与持久化调度

> **今日目标**：理解 DeepGEMM 的 warp specialization 分工（TMA warpgroup vs Math warpgroup）、`warpgroup_reg_dealloc/alloc` 寄存器重配、持久化调度器与 TMA multicast
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，warp specialization + 持久化调度是 Hopper kernel 的标志性设计

---

### 学习任务 1：Warp Specialization 角色划分（30 分钟）

DeepGEMM 的 threadblock 分为两个角色（见 `sm90_fp8_gemm_1d1d.cuh:170,246`）：

| 角色 | 线程数 | Warp ID | 职责 | 关键指令 |
|------|--------|---------|------|----------|
| **TMA warpgroup** | 128 | `>= kNumMathThreads/32` | 发起 TMA load + 管理屏障 | `tma::copy`, `arrive_and_expect_tx` |
| **Math warpgroup(s)** | 128-256 | `< kNumMathThreads/32` | 发射 WGMMA + scale 乘法 | `wgmma`, `ld_shared` |

```cpp
// 判断角色
const uint32_t warp_idx = threadIdx.x / 32;

if (warp_idx >= kNumMathThreads / 32) {
    // TMA warpgroup：让出寄存器
    cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
    // ... TMA load 循环 ...
} else {
    // Math warpgroup：申请更多寄存器
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
    // ... WGMMA 循环 ...
}
```

> 💡 **一句话总结**：Warp specialization 的本质是"专职化"——TMA warpgroup 只搬数据不碰 Tensor Core，Math warpgroup 只算不碰 TMA。两者通过 mbarrier 异步握手，谁不等谁。这与 CUTLASS 2.x 的"所有 warp 干一样的事"是根本转变。

### 学习任务 2：寄存器重配（warpgroup_reg_dealloc/alloc）（30 分钟）

读 `sm90_fp8_gemm_1d1d.cuh:153-155`：

```cpp
// 寄存器重配（更多 math 寄存器在有 unroll 时才需要）
constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);
```

| Warp | 寄存器上限 | 理由 |
|------|-----------|------|
| TMA | 24-40 | TMA 只需地址寄存器，几乎不占通用寄存器 |
| Math | 232-240 | WGMMA 累加器（64×N/128 个 FP32）+ scale buffer |

Hopper 用 `cutlass::arch::warpgroup_reg_dealloc<N>` / `warpgroup_reg_alloc<N>` 在运行时切换寄存器上限（底层是 `setmaxnreg` PTX 指令）。Ampere 的寄存器上限是编译期固定的，做不到这种动态重配。

> ⚠️ **注意**：寄存器重配必须在所有 warp 同步执行（`.sync.aligned`），且在 threadblock 入口处尽早做。DeepGEMM 在 TMA/Math 分支入口的第一件事就是重配寄存器。

### 学习任务 3：持久化调度器（45 分钟）

读 `scheduler/gemm.cuh`。DeepGEMM 用**持久化 kernel**——一个 threadblock 串行处理多个 tile，而不是一个 tile 一个 block。

```cpp
// 持久化调度核心循环
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // 处理一个 (m_block, n_block) tile
    // ... TMA load + WGMMA + epilogue ...
}
```

#### 调度器支持的 GemmType

读 `common/types.cuh`：

| GemmType | 含义 | 典型场景 |
|----------|------|----------|
| `Normal` | 标准 GEMM | 密集计算 |
| `MGroupedContiguous` | M 轴分组，token 连续 | MoE 前向/prefill |
| `MGroupedMasked` | M 轴分组，mask 标记 | MoE decode（CUDA graph） |
| `KGroupedContiguous` | K 轴分组 | MoE 权重梯度 |
| `MGroupedContiguousWithPsumLayout` | M 分组 + psum 布局 | MoE 反向 |
| `KGroupedContiguousWithPsumLayout` | K 分组 + psum 布局 | MoE 反向 |
| `Batched` | 批量 GEMM | — |

#### Stream-K 风格

调度器还支持 Stream-K——把 K 维也切分，让多个 threadblock 协作算同一个 (M, N) tile，最后用 `SM90_TMA_REDUCE_ADD_2D` 归约（见 `sm90_fp8_gemm_1d1d.cuh:341`）。这在小 M×N tile 数 < SM 数时消除长尾。

### 学习任务 4：TMA Multicast 与 Cluster（30 分钟）

DeepGEMM 支持 TMA multicast——一条 TMA 指令把数据广播到 cluster 内的多个 CTA：

```cpp
template <... uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA, ...>
// kNumTMAMulticast: multicast 到几个 CTA（1 或 2）
// kIsTMAMulticastOnA: A 还是 B 做 multicast
```

```cpp
const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a, &full_barrier, smem_a[stage], k_idx, m_idx, num_tma_multicast_a);
```

> 💡 **关键洞察**：TMA multicast 让相邻 CTA 共享同一份 A（或 B）数据——例如两个 CTA 算同一 M 行不同 N 列，A tile 只需从 gmem 搬一次。这是 Hopper cluster 级优化的核心，CUTLASS 3.x 也用同样机制。

### 学习任务 5：SMEM Swizzle 与 WGMMA desc（30 分钟）

读 `mma/sm90.cuh:194-279`，WGMMA 的操作数用 `GmmaDescriptor` 描述 SMEM 布局：

```cpp
// 构造 SMEM descriptor（WGMMA 操作数）
template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode, typename dtype_t>
cute::GmmaDescriptor make_gmma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx);
```

Swizzle 模式（`kSwizzleMode`）对应 GMMA LayoutType：

| kSwizzleMode | LayoutType | 适用 |
|--------------|------------|------|
| 0 / 16 | INTERLEAVE | 无 swizzle / 交织 |
| 32 | B32 | 32B 粒度 |
| 64 | B64 | 64B 粒度 |
| 128 | B128 | 128B 粒度（FP8 常用） |

> ⚠️ **注意**：DeepGEMM 的 swizzle 在 TMA descriptor 里指定（`kSwizzleAMode`），TMA 搬入 SMEM 时硬件自动应用，WGMMA 读取时硬件自动反 swizzle。K-major 布局要求 `kSwizzleMode == BLOCK_K * sizeof(dtype)`——FP8 即 128B。

### 今日检查清单

- [ ] 能列出 TMA warpgroup / Math warpgroup 的指令分工与寄存器需求差异
- [ ] 能解释 `warpgroup_reg_dealloc/alloc` 为什么是 Hopper 才有的
- [ ] 能说出持久化调度与 Stream-K 的关系
- [ ] 能解释 TMA multicast 如何让相邻 CTA 共享数据
- [ ] 理解 `make_gmma_desc` 如何编码 SMEM 地址 + swizzle 模式

---

