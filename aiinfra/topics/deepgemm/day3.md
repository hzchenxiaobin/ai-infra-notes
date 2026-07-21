# Day 3（周三）：SM90 FP8 GEMM Kernel 源码精读（上）

> **本周定位**：本专题是 [CUTLASS 专题](../cutlass/README.md)（库视角）与 [CuTe 专题](../cute/README.md)（原语视角）之后的**单点深钻**——拆开一个生产级 FP8/FP4 GEMM kernel 看每一行 PTX 怎么写。
> **前置要求**：已完成 Day 1（JIT 环境与源码地图）+ Day 2（FP8/FP4 scaling 数据流），理解 per-128-channel scaling 的软件 promote 流程
> **今日目标**：精读 `sm90_fp8_gemm_1d1d.cuh` 的**数据搬运与计算流水**——TMA descriptor 构造、WGMMA async 发射、mbarrier 双 barrier 同步、warp specialization 的两个分支
> **时间投入**：2.5h（早间 1.5h 精读 kernel 主体 + 晚间 1h 读 PTX 封装）
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"DeepGEMM 的 TMA 和 WGMMA 怎么 overlap"必问

---

## 本日在本周知识图谱中的位置

```
Day 1          Day 2           Day 3-4            Day 5           Day 6          Day 7
 总览      →   FP8/FP4     →   SM90 Kernel   →   Grouped      →  SM100/Mega  →  调优
 JIT 环境      Scaling         源码精读           GEMM for MoE     MoE            ncu
 源码地图      per-128-ch      TMA+WGMMA          contiguous/      TCgen05        报告
               UE8M0           持久化调度          masked/k-group   EP 融合
                                  ↑
                                  你在这里（Day 3 = 上半场：TMA/WGMMA/barrier；Day 4 = 下半场：调度器/epilogue/cluster）
```

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| TMA warpgroup + Math warpgroup 分支结构 | ① 能画出时序图并标注 barrier 握手点（Day 4 完善） |
| WGMMA 发射与 scale promote 代码 | ① 同上（理解 Math 分支才能画时序图） |
| SMEM 布局图 | ① 同上（时序图要标 SMEM buffer 状态） |

> ⚠️ **Day 3 与 Day 4 的分工**：今天读 kernel 主体（`sm90_fp8_gemm_1d1d.cuh` 的 TMA 分支 `:170-245` + Math 分支 `:246-348`），重点在**单 tile 内的流水线**。明天读持久化调度器（`scheduler/gemm.cuh`）、寄存器重配、cluster multicast、epilogue 的 TMA store——即**跨 tile 的调度**与**输出回写**。

---

### 学习任务 1：Hopper 异步三件套回顾（30 分钟）

复习 [CuTe 专题 Day 6](../cute/README.md) 与 [FlashAttention-3 论文精读](../../paper/flashattention3/README.md) §3，Hopper 的三个独立执行单元：

| 单元 | 指令 | 异步性 | 占 SM？ | DeepGEMM 封装 |
|------|------|--------|---------|---------------|
| **TMA** | `cp.async.bulk.tensor` | 硬件异步，1 thread 发射 | 否 | `tma::copy`（`common/tma_copy.cuh`） |
| **WGMMA** | `wgmma.mma_async.sync.aligned.m64nNk32` | 异步，warpgroup 发射后立即返回 | 是（Tensor Core） | `WGMMA::wgmma`（`mma/sm90.cuh:22`） |
| **CUDA core / SFU** | `add`/`mul`/`ld.shared`/`st.shared` | 同步 | 是 | 裸 PTX（`ptx/ld_st.cuh`） |

> 💡 **关键洞察**：三者可同时工作——TMA 搬数时 Tensor Core 在算上一个 tile，CUDA core 在做 scale 乘法。DeepGEMM 的 kernel 设计目标就是让三者不互相等待。warp specialization 是实现这一目标的标准手段：TMA warpgroup 专管搬运，Math warpgroup 专管计算，两者用 mbarrier 解耦。

#### WGMMA 三件套 PTX

读 `ptx/wgmma.cuh`（仅 25 行，全库最精简的文件之一）：

```cpp
// ptx/wgmma.cuh:7-9  —— fence：确保之前的 smem 写对后续 WGMMA 可见
CUTLASS_DEVICE void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// :11-13  —— commit：把 fence 后发射的所有 wgmma.mma_async 打包成一个 group
CUTLASS_DEVICE void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// :19-23  —— wait：等待最多 N 个未完成 group（N=0 表示等全部完成）
template <int N>
CUTLASS_DEVICE void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// :15-17  —— fence_operand：防止编译器把寄存器读写重排到 WGMMA 两侧
CUTLASS_DEVICE void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}
```

**时序关系**（`sm90_fp8_gemm_1d1d.cuh:291-306`）：

```
warpgroup_arrive()          ← wgmma.fence（确保 smem 数据就绪）
  ├─ wgmma.mma_async #0     ← k==0, ScaleOut::Zero（覆盖 accum）
  ├─ wgmma.mma_async #1     ← k==1, ScaleOut::One（累加进 accum）
  ├─ wgmma.mma_async #2     ← k==2, ScaleOut::One
  └─ wgmma.mma_async #3     ← k==3, ScaleOut::One
warpgroup_commit_batch()    ← commit_group（4 次 MMA 打包成一个 group）
warpgroup_fence_operand ×N  ← 防止编译器重排寄存器
warpgroup_wait<0>()         ← wait_group 0（等这个 group 完成）
```

> ⚠️ **为什么 fence_operand 要在 commit 后、wait 前各做一次？** 见 `:292-294`（commit 前）和 `:303-305`（wait 前）。编译器可能把 `accum` 的读写重排到 WGMMA 两侧——`fence_operand` 用空 asm + `"+f"` 约束强制编译器认为寄存器被修改了，阻止重排。

#### TMA 与 mbarrier 的 PTX

读 `ptx/tma.cuh:35-61`，TMA 加载用 mbarrier 做 completion signal：

```cpp
// :41-45  —— TMA 发射后立即 arrive，并告知期望传输的字节数
CUTLASS_DEVICE void mbarrier_arrive_and_set_tx(
    cutlass::arch::ClusterTransactionBarrier* ptr, const uint32_t& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

// :47-61  —— 等待 mbarrier 翻转 phase（自旋 try_wait）
CUTLASS_DEVICE void mbarrier_wait_and_flip_phase(
    cutlass::arch::ClusterTransactionBarrier* ptr, uint32_t& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"  // %2 = 超时(0x989680≈10ms)
        "@P1 bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}" :: "r"(...), "r"(phase), "r"(0x989680));
    phase ^= 1;   // phase 翻转
}
```

> 💡 **mbarrier 的工作方式**：TMA 硬件在传输完成后自动 arrive 并减去传输字节数；当 `expect_tx` 的字节数被减到 0 时，barrier 翻转 phase，等待方 `try_wait.parity` 成功返回。这是**纯硬件异步**——TMA 发射线程不需要轮询，也不需要 SM 周期。

### 学习任务 2：Kernel 模板参数与 SMEM 布局（45 分钟）

读 `sm90_fp8_gemm_1d1d.cuh:30-126`，kernel 是一个大模板：

```cpp
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_impl(...)
```

#### 模板参数分类

| 分类 | 参数 | 含义 | 谁决定 |
|------|------|------|--------|
| **Shape** | `SHAPE_M/N/K` | M/N/K 维度（0 表示运行时传入） | JIT 编译期（`get_compiled_dim`，受 `compiled_dims` 控制） |
| **Tile** | `BLOCK_M/N/K` | tile 大小，K 固定 128 | heuristics 选 |
| **Swizzle** | `kSwizzleAMode/BMode` | SMEM swizzle 模式（0/16/32/64/128） | heuristics 选，断言 `== block_k`（`:102-103`） |
| **Pipeline** | `kNumStages` | 流水线深度（典型 3-4） | heuristics 选（受 smem_capacity=232448 限制） |
| **线程** | `kNumTMAThreads` / `kNumMathThreads` | TMA/Math warpgroup 线程数 | heuristics 选 |
| **Cluster** | `kNumTMAMulticast` / `kIsTMAMulticastOnA` | TMA multicast（1 或 2）、multicast 作用在 A 还是 B | heuristics 选 |
| **调度** | `kNumSMs` | 使用的 SM 数 | `set_num_sms` |
| **类型** | `kGemmType` / `cd_dtype_t` | GEMM 类型 / 输出 dtype | API 决定 |

#### 关键断言

```cpp
DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");
DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");
DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");
```

- TMA warpgroup 固定 128 线程（1 warpgroup = 4 warps）
- Math warpgroup 是 128 的倍数（1-2 warpgroups，典型 128 或 256）
- BLOCK_K 固定 128（per-128-channel scaling，Day 2 已解释）
- 累加只在 FP32（1D1D 是 wgrad，需要 FP32 累加）
- BLOCK_M 必须是 WGMMA::M（64）的倍数——所以 BLOCK_M ∈ {64, 128}

#### SMEM 分配

读 `:68-125`，SMEM 是一块连续的 `extern __shared__` buffer，按以下顺序分配：

```
SMEM 布局（kNumStages = 3, BLOCK_M=128, BLOCK_N=128 为例）：
┌────────────────────────────────────────────────────────────────────┐
│ Tensor Maps (仅 KGrouped 模式，2 × TmaDescriptor = 256B)            │  ← :69
├────────────────────────────────────────────────────────────────────┤
│ D: BLOCK_M × BLOCK_N × FP32 = 128×128×4 = 65536B (64KB)            │  ← :70, :103
├────────────────────────────────────────────────────────────────────┤
│ A: [stage 0][stage 1][stage 2] × BLOCK_M×BLOCK_K×FP8              │  ← :104-106
│    = 3 × 128×128×1 = 49152B (48KB)                                 │
├────────────────────────────────────────────────────────────────────┤
│ B: [stage 0][stage 1][stage 2] × BLOCK_N×BLOCK_K×FP8              │  ← :107-109
│    = 3 × 128×128×1 = 49152B (48KB)                                 │
├────────────────────────────────────────────────────────────────────┤
│ SFA: [stage 0][stage 1][stage 2] × BLOCK_M×FP32                    │  ← :111-113
│     = 3 × 128×4 = 1536B                                             │
├────────────────────────────────────────────────────────────────────┤
│ SFB: [stage 0][stage 1][stage 2] × BLOCK_N×FP32 (aligned 128B)    │  ← :114-116
│     = 3 × 512B = 1536B                                              │
├────────────────────────────────────────────────────────────────────┤
│ full_barriers[3] + empty_barriers[3] (6 × ClusterTransactionBarrier)│  ← :120-124
└────────────────────────────────────────────────────────────────────┘
总计 ≈ 64 + 48 + 48 + 1.5 + 1.5 + barrier ≈ 163KB（< 232KB smem 容量）
```

> 💡 **关键设计**：DeepGEMM 把 **scale factor 也放进 SMEM 流水线**——每个 stage 不仅有 A/B 数据，还有对应的 SFA/SFB。这样 scale 读取与 WGMMA 计算完全 overlap，不会因为 scale 加载引入额外延迟。
>
> **PatternVisitor**（`:104-116`，定义在 `common/utils.cuh:11-22`）：SMEM 的 stage 索引通过 lambda 计算——`smem_a[stage_idx]` 返回对应 stage 的指针。这是一种零开销的"虚拟数组"，实际是 base + offset 的计算。

#### 对齐约束

```cpp
extern __shared__ __align__(1024) uint8_t smem_buffer[];                    // :93
DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "...must be aligned to 1024 bytes"); // :94
DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment"); // :76
```

- 整体 buffer 对齐 1024B（swizzle-128B 要求）
- D 的 size 必须是 1024 的倍数（`BLOCK_M * BLOCK_N * 4 % 1024 == 0`）
- SFA 的 size 必须是 128 的倍数（`BLOCK_M * 4 % 128 == 0` → `BLOCK_M % 32 == 0`）

### 学习任务 3：TMA Descriptor 构造（45 分钟）

TMA 是 Hopper 的硬件异步搬运单元，需要先在 host 端构造 `CUtensorMap`（TMA descriptor），再传给 kernel。

#### Host 端：make_tma_*_desc

读 `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:105-121`（JIT 胶水层）与 `csrc/jit_kernels/impls/runtime_utils.hpp:113-150`（`make_tma_2d_desc`）：

```cpp
// runtime_utils.hpp:113-150
static CUtensorMap make_tma_2d_desc(const torch::Tensor& t,
                                    int gmem_inner_dim, int gmem_outer_dim,
                                    int smem_inner_dim, int smem_outer_dim, ...) {
    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {gmem_inner_dim, gmem_outer_dim};
    const cuuint32_t smem_dims[2] = {smem_inner_dim, smem_outer_dim};
    const cuuint64_t gmem_strides[1] = {gmem_outer_stride * elem_size};
    const cuuint32_t elem_strides[2] = {1, 1};
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), ...),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, ...),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}
```

5 个 TMA descriptor（`sm90_fp8_gemm_1d1d.hpp:105-121`）：

| Descriptor | 数据 | gmem 布局 | smem 布局 | swizzle |
|------------|------|-----------|-----------|---------|
| `tensor_map_a` | A `[M, K]` | K-major（inner=K, outer=M） | `load_block_m × block_k` | `swizzle_a_mode`（== block_k=128） |
| `tensor_map_b` | B `[N, K]` | K-major（inner=K, outer=N） | `load_block_n × block_k` | `swizzle_b_mode`（== block_k=128） |
| `tensor_map_sfa` | SFA `[M, K/128]` | MN-major（inner=M, outer=K/128） | `block_m × block_k` | 0（无 swizzle） |
| `tensor_map_sfb` | SFB `[N, K/128]` | MN-major（inner=N, outer=K/128） | `block_n × block_k` | 0（无 swizzle） |
| `tensor_map_cd` | D `[M, N]` | N-major（inner=N, outer=M） | `store_block_m × store_block_n` | 由 config 决定 |

> 💡 **关键约束**（`:102-103`）：`swizzle_a_mode == block_k` 且 `swizzle_b_mode == block_k`——即 A/B 的 swizzle 模式必须等于 BLOCK_K。对 FP8（1 字节），`swizzle_mode=128` 意味着 `CU_TENSOR_MAP_SWIZZLE_128B`，每个 smem atom 是 128 字节（128 个 FP8）。

#### Device 端：tma::copy

读 `common/tma_copy.cuh:16-90`，`tma::copy` 是对 CuTe `SM90_TMA_LOAD_2D` 的封装：

```cpp
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, ...>
CUTLASS_DEVICE void copy(void const* desc_ptr, ClusterTransactionBarrier* barrier_ptr,
                         dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
                         const uint32_t& num_tma_multicast = 1, ...) {
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();
    if (num_tma_multicast == 1) {
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
            cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                         cache_hint, smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                         inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
        }
    } else {
        // SM90 multicast：只 leader CTA 发射，bitmask = (1<<num_tma_multicast)-1
        if (cute::block_rank_in_cluster() == 0) {
            cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, ..., (1 << num_tma_multicast) - 1, ...);
        }
    }
}
```

- `BLOCK_INNER_ATOM = kSwizzleMode / sizeof(dtype_t)`：swizzle 128B + FP8（1B）→ atom=128 元素
- 如果 `BLOCK_INNER > atom`，拆成多次 TMA load（如 BLOCK_K=128, atom=128 → 1 次）
- multicast 模式：cluster 内的 leader CTA（rank 0）发射，其他 CTA 被动接收

#### kernel 内的 TMA 发射

读 `sm90_fp8_gemm_1d1d.cuh:229-233`（TMA warpgroup 分支内）：

```cpp
tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx, num_tma_multicast_a);
tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx, num_tma_multicast_b);
tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a_ptr, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tensor_map_b_ptr, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE);
```

**关键细节**：
1. **SFA/SFB 的 inner/outer 与 A/B 不同**：SFA 是 `tma::copy<BLOCK_M, BLOCK_K, 0>`（inner=M, outer=K/128），A 是 `tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>`（inner=K, outer=M）。因为 SFA 是 MN-major 而 A 是 K-major。
2. **4 个 TMA 共享一个 barrier**：`full_barrier` 收到所有 4 个 TMA 的 completion signal，`expect_tx` 是 4 个 buffer 的总字节数。
3. **swizzle 仅作用于 A/B**：SFA/SFB 的 swizzle=0（无 swizzle），因为 scale 数据量小、不需要避免 bank conflict。

> ⚠️ **TMA descriptor 的 `__grid_constant__` 修饰**（`:44-48`）：5 个 TMA descriptor 都声明为 `const __grid_constant__`，这意味着它们存在 constant memory 的 grid-constant 参数区，所有 threadblock 共享一份、不需要每次 launch 拷贝。这对 TMA 性能至关重要——TMA descriptor 是 128 字节，如果不加 `__grid_constant__`，每个 threadblock 都要从 global memory 加载一份。

### 学习任务 4：mbarrier 双 barrier 设计（45 分钟）

这是 Day 3 的**核心精读**内容——理解双 barrier 才能画出 TMA + Math 的时序图。

#### Barrier 初始化

读 `:127-148`：

```cpp
if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
    // Initialize barriers
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        full_barriers[i]->init(1);                                          // 生产者=TMA（1 个 thread arrive）
        empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);  // 生产者=Math（N 个 warp arrive）
    }
    cutlass::arch::fence_barrier_init();    // 让 barrier 可见于 async proxy
}
(kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();
```

| Barrier | `init` 参数 | 含义 |
|---------|-------------|------|
| `full_barriers[i]` | `1` | 1 个生产者（TMA 的 1 个 elect thread）arrive 后翻转 |
| `empty_barriers[i]` | `kNumTMAMulticast * kNumMathThreads / 32` | N 个 warp（math warpgroup 的每个 warp）各 arrive 一次 |

> 💡 **为什么 empty_barrier 要 N 个 warp arrive？** Math warpgroup 有 `kNumMathThreads / 32` 个 warp（如 256/32=8），每个 warp 独立消费 smem 数据后都需要 signal "我用完了"。multicast 模式下，cluster 内的 peer CTA 也要 arrive（`kNumTMAMulticast *`）。

#### 双 barrier 握手时序

```
时间 →
TMA:   load stage0 → arrive(full[0], tx=bytes) → load stage1 → arrive(full[1]) → wait(empty[0]) → load stage0' → ...
Math:                              wait(full[0]) → WGMMA → arrive(empty[0]) → wait(full[1]) → ...
                                                    ↑ 两者在不同 stage 上并行
```

| Barrier | 生产者 | 消费者 | 含义 |
|---------|--------|--------|------|
| `full_barriers[stage]` | TMA `arrive_and_expect_tx` | Math `wait` | "数据已满，可算" |
| `empty_barriers[stage]` | Math `arrive` | TMA `wait` | "数据用完，可覆盖" |

**TMA 侧**（`:219-233`）：

```cpp
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
    // 1. 等 consumer 释放 buffer
    empty_barriers[stage_idx]->wait(phase ^ 1);

    // 2. 发射 4 个 TMA
    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], ...);
    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], ...);
    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(..., smem_a[stage_idx], ...);
    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(..., smem_b[stage_idx], ...);

    // 3. 告知 barrier 期望字节数
    full_barrier.arrive_and_expect_tx(SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB);
}
```

**Math 侧**（`:276-309`）：

```cpp
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
    // 1. 等 TMA 搬完
    full_barriers[stage_idx]->wait(phase);

    // 2. 读 scale + 发射 WGMMA（见 Day 2 学习任务 3）
    ...
    // 3. 释放 buffer（通知 TMA 可以覆盖）
    empty_barrier_arrive(stage_idx);
}
```

#### Pipeline stage 与 phase 的计算

读 `:165-168`：

```cpp
const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
    return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; // Pipeline stage and phase
};
uint32_t iter_idx = 0;
```

- `stage_idx = iter_idx % kNumStages`：循环复用 kNumStages 个 buffer
- `phase = (iter_idx / kNumStages) & 1`：每走过一轮 kNumStages，phase 翻转一次

> 💡 **为什么需要 phase？** mbarrier 用 phase parity（偶/奇）区分"本轮"和"上一轮"。TMA 的 `wait(phase ^ 1)` 表示等上一轮的 empty（buffer 被释放），Math 的 `wait(phase)` 表示等本轮的 full（数据就绪）。phase 机制让 mbarrier 可以被无限复用而不需要重新 init。

> 💡 **关键洞察**：双 barrier 是无锁流水线的标准模式——TMA 不需要知道 Math 何时算完，只需等 `empty` 信号；Math 不需要知道 TMA 何时搬完，只需等 `full` 信号。两者完全解耦，流水线深度由 `kNumStages` 决定（典型 3-4 stage）。

### 学习任务 5：WGMMA 指令选择与 smem desc（30 分钟）

#### FP8MMASelector

读 `mma/sm90.cuh:14-75`，FP8 的 WGMMA 是 `m64nNk32`（M=64 固定，N 从 8 到 256，K=32）：

```cpp
template <int N_, typename MMA>
struct FP8MMA {
    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;   // 每个 lane 持有的 FP32 accum 数
};

template <int N>
struct FP8MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8)   return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_SS_TN();
        // ... N 从 8 到 256，步长 8 ...
    }
    using type = decltype(FP8MMA<N, decltype(select_mma())>());
};
```

- `M=64, K=32` 是 FP8 WGMMA 的固定 shape
- `kNumAccum = 64 * N / 128`：128 个 lane 共享 `64×N` 个 FP32 accum，每个 lane 持有 `N/2` 个（如 N=128 → 64 个）
- `scale_d ? ScaleOut::One : ScaleOut::Zero`：第一个 MMA（k==0）覆盖 accum，后续累加

> ⚠️ **注意**：FP8 WGMMA 的 K=32（4 个 FP8 pack 成 32 字节），BF16 WGMMA 的 K=16。`BLOCK_K=128` 对应 FP8 的 4 次 WGMMA（`128/32`）、BF16 的 8 次（`128/16`）。

#### smem desc 构造

读 `mma/sm90.cuh:194-209`，WGMMA 不直接接受 smem 指针，而是接受一个 64 位的 `GmmaDescriptor`：

```cpp
template <class PointerType>
CUTLASS_DEVICE cute::GmmaDescriptor
make_smem_desc(PointerType smem_ptr, const int& layout_type,
               const uint32_t& leading_byte_offset = 0,
               const uint32_t& stride_byte_offset = 1024) {
    cute::GmmaDescriptor desc;
    const auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;           // smem 地址（右移 4 位，16B 对齐）
    desc.bitfield.layout_type_ = layout_type;               // swizzle 模式
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;  // LBO（atom 间 stride）
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;    // SBO（行间 stride）
    desc.bitfield.base_offset_ = 0;
    return desc;
}
```

kernel 内的调用（`:298-299`）：

```cpp
auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
```

- `layout_type=1` 表示 K-major（`cute::SM90::GMMA::Major::K`，FP8 要求 K-major）
- `math_wg_idx * WGMMA::M * BLOCK_K`：如果有 2 个 math warpgroup（BLOCK_M=128），每个 warpgroup 负责自己的 64 行
- `k * WGMMA::K`：在 BLOCK_K=128 内部按 K=32 步进

> 💡 **`make_gmma_desc` vs `make_smem_desc`**：`make_smem_desc`（`:194-209`）是简化版，固定 LBO=0、SBO=1024（默认 K-major）；`make_gmma_desc`（`:237-279`）是完整版，根据 major-ness 和 swizzle mode 计算 LBO/SBO。1D1D kernel 用简化版（因为 A/B 都是 K-major 且 swizzle=128B），1D2D kernel 用完整版（SFB 不是 K-major）。

### 学习任务 6：两个 warpgroup 分支的完整代码结构（30 分钟）

读 `:170-348`，理解 kernel 的整体控制流。

#### 分支结构

```
warp_idx >= kNumMathThreads / 32 ?
│
├─ YES → TMA warpgroup 分支（:170-245）
│        ├─ warpgroup_reg_dealloc<kNumTMARegisters>(24 或 40)
│        ├─ elect_one_sync()（只有 1 个 thread 发射 TMA）
│        └─ while (scheduler.get_next_block(...))     ← 持久化调度
│             ├─ （KGrouped）更新 tensormap
│             └─ for k_block_idx in 0..num_k_blocks
│                  ├─ wait(empty_barriers[stage])
│                  ├─ tma::copy × 4（SFA, SFB, A, B）
│                  └─ full_barrier.arrive_and_expect_tx(bytes)
│
└─ NO → Math warpgroup 分支（:246-348）
         ├─ warpgroup_reg_alloc<kNumMathRegisters>(232 或 240)
         ├─ 计算 lane 到 accum 的映射（r_0, r_1, row_idx, col_idx）
         └─ while (scheduler.get_next_block(...))     ← 同样的持久化调度
              ├─ for k_block_idx in 0..num_k_blocks
              │    ├─ wait(full_barriers[stage])
              │    ├─ ld_shared(scale_a, scale_b)
              │    ├─ warpgroup_arrive + wgmma × 4 + commit + wait
              │    ├─ empty_barrier_arrive(stage)
              │    └─ promote with scales（4 次 FFMA）
              ├─ tma_store_wait<0>()（等上一个 tile 的 store 完成）
              ├─ st_shared(final_accum → smem_d)
              └─ TMA_REDUCE_ADD_2D（写回 gmem）
```

#### 寄存器重配

读 `:150-155`：

```cpp
constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);
constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);
```

| 模式 | kNumPipelineUnrolls | TMA 寄存器 | Math 寄存器 |
|------|---------------------|-----------|-------------|
| Normal（有 pipeline unroll） | kNumStages（3-4） | 24 | 240 |
| KGrouped（无 unroll） | 0 | 40 | 232 |

- `warpgroup_reg_dealloc`（`:172`）：TMA warpgroup 释放寄存器给 Math warpgroup
- `warpgroup_reg_alloc`（`:248`）：Math warpgroup 抢占更多寄存器
- Hopper 的 warpgroup reg reconfig 机制：SM 内两个 warpgroup 可以**不均分**寄存器（默认各 128 个，重配后 TMA 只留 24-40，Math 拿到 232-240）

> 💡 **为什么 KGrouped 用 40/232 而 Normal 用 24/240？** KGrouped 模式下 `kNumPipelineUnrolls=0`（`#pragma unroll 0` = 不展开 k 循环），TMA warpgroup 需要更多寄存器处理 tensormap 替换逻辑（`:191-215`），所以多分配一些。Normal 模式展开 k 循环，Math 需要更多寄存器存放展开的 accum。

#### `#pragma unroll kNumPipelineUnrolls` 的作用

读 `:217` 和 `:275`：

```cpp
#pragma unroll kNumPipelineUnrolls
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) { ... }
```

- Normal 模式：`kNumPipelineUnrolls = kNumStages`（如 3），编译器展开前 3 次 k 循环——让 TMA 和 WGMMA 的 stage 流水线在编译期就排好
- KGrouped 模式：`kNumPipelineUnrolls = 0`，不展开（因为 K 是运行时变量，无法编译期展开）

> 💡 **pipeline unroll 的本质**：展开 k 循环让编译器在同一个指令流里同时看到多个 stage 的 TMA 发射和 WGMMA，从而把它们交错排列——TMA[stage0] → WGMMA[stage0 前一个] → TMA[stage1] → ... 这才是真正的**软件 pipelining**，而不只是循环展开。

#### TMA warpgroup 的 elect_one_sync

读 `:175`：

```cpp
if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
```

- TMA warpgroup 有 128 个线程，但**只有 1 个 thread** 实际发射 TMA（`elect_one_sync` 选 lane 0）
- 其他 127 个 thread 进入 `warpgroup_reg_dealloc` 后什么都不做（释放寄存器后休眠）
- 这是 Hopper warp specialization 的极致：TMA 是单 thread 指令，不需要整个 warpgroup

#### Math warpgroup 的 lane 映射

读 `:251-253`：

```cpp
const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;
```

- `math_wg_idx`：0 或 1（BLOCK_M=128 时有 2 个 math warpgroup，各负责 64 行）
- `row_idx = lane_idx / 4`：每个 lane 负责 2 行（`r_0` 和 `r_0+8`）
- `col_idx = lane_idx % 4`：每个 lane 负责若干列（取决于 BLOCK_N）

> 💡 **`__shfl_sync` 的用意**（注释 `:250`）："use `__shfl_sync` to encourage NVCC to use unified registers"——用 warp shuffle 而非整数除法，让编译器把 `math_wg_idx` 放进 uniform register，所有 lane 共享一份。

### 面试题积累（本周目标 10-12 道，今日 3 道）

**Q8：DeepGEMM 的 warp specialization 是怎么做的？TMA 和 Math warpgroup 怎么分工？**
> 答：kernel 有两类 warpgroup——TMA warpgroup（128 线程，但只有 1 个 elect thread 发射 TMA）和 Math warpgroup（128 或 256 线程，跑 WGMMA）。两者用 mbarrier 双 barrier 解耦：`full_barrier` 由 TMA arrive（数据就绪）、Math wait；`empty_barrier` 由 Math arrive（数据用完）、TMA wait。寄存器通过 `warpgroup_reg_dealloc` / `warpgroup_reg_alloc` 不均分——TMA 只留 24-40 个，Math 拿到 232-240 个。TMA 是单 thread 指令，不需要整个 warpgroup；Math 需要大量寄存器存放展开的 accum。

**Q9：mbarrier 的双 barrier 设计是什么？为什么需要 phase？**
> 答：双 barrier = `full_barriers[kNumStages]` + `empty_barriers[kNumStages]`。full 表示"数据已满可算"（TMA 生产、Math 消费），empty 表示"数据用完可覆盖"（Math 生产、TMA 消费）。phase 用 parity（偶/奇）区分本轮和上一轮——TMA `wait(phase^1)` 等上一轮的 empty，Math `wait(phase)` 等本轮的 full。`stage_idx = iter_idx % kNumStages`，`phase = (iter_idx / kNumStages) & 1`，每走过一轮 stage 翻转一次 phase。这让 mbarrier 可以被无限复用而不需要重新 init。

**Q10：WGMMA 的 fence / commit / wait 三件套各做什么？为什么 fence_operand 要调两次？**
> 答：① `wgmma.fence`（`warpgroup_arrive`）：确保之前的 smem 写对后续 WGMMA 可见，标记 WGMMA 序列开始；② `wgmma.commit_group`（`warpgroup_commit_batch`）：把 fence 后发射的所有 `wgmma.mma_async` 打包成一个 group；③ `wgmma.wait_group 0`（`warpgroup_wait<0>`）：等待所有 group 完成。`warpgroup_fence_operand` 防止编译器把 accum 寄存器的读写重排到 WGMMA 两侧——commit 前调一次（确保 accum 是"干净"的输入），wait 前再调一次（确保 WGMMA 写完才读 accum）。空 asm + `"+f"` 约束强制编译器认为寄存器被修改了。

### 今日检查清单

- [ ] 能说出 kernel 的关键模板参数分类（Shape / Tile / Swizzle / Pipeline / 线程 / Cluster / 调度 / 类型）
- [ ] 能画出 SMEM 布局（D / A×stage / B×stage / SFA×stage / SFB×stage / barriers）并说出大小
- [ ] 能解释 5 个 TMA descriptor 的 gmem/smem 布局与 swizzle 差异（A/B 有 swizzle，SFA/SFB 无）
- [ ] 理解 `__grid_constant__` 对 TMA descriptor 的意义（constant memory，全 grid 共享）
- [ ] 能解释 `full_barriers` / `empty_barriers` 的 `init` 参数（1 vs N warps）和握手时序
- [ ] 能写出 `stage_idx = iter_idx % kNumStages` / `phase = (iter_idx / kNumStages) & 1` 并解释 phase 的作用
- [ ] 能说出 WGMMA 的 `m64nNk32` shape，BLOCK_K=128 对应 4 次 WGMMA，`ScaleOut::Zero/One` 的含义
- [ ] 能解释 `warpgroup_fence_operand` 为什么在 commit 前后各调一次
- [ ] 读懂 TMA 分支（`:170-245`）与 Math 分支（`:246-348`）的代码结构
- [ ] 理解寄存器重配（TMA 24-40 / Math 232-240）与 `kNumPipelineUnrolls` 的关系
- [ ] 读完 `ptx/wgmma.cuh`（25 行）和 `ptx/tma.cuh` 的 mbarrier 部分

#### 明日预告

Day 4 将完成 `sm90_fp8_gemm_1d1d.cuh` 的下半场——精读 `scheduler/gemm.cuh` 的持久化调度与 Stream-K、cluster multicast 的 TMA 多播机制、epilogue 的 TMA store（`SM90_TMA_REDUCE_ADD_2D`）回写流程。今天理解了单 tile 内的 TMA+Math 流水线，明天要把它扩展到跨 tile 的持久化调度。建议今晚先扫一眼 `scheduler/gemm.cuh` 的 `get_next_block` 接口，理解 threadblock 如何串行处理多个 tile。

---
