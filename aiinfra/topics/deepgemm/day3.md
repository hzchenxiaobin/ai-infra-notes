# Day 3（周三）：SM90 FP8 GEMM Kernel 源码精读（上）

> **今日目标**：精读 `sm90_fp8_gemm_1d1d.cuh` 的数据搬运与计算流水——TMA descriptor、WGMMA async 发射、mbarrier 同步、持久化调度
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"DeepGEMM 的 TMA 和 WGMMA 怎么 overlap"必问

---

### 学习任务 1：Hopper 异步三件套回顾（30 分钟）

复习 [CuTe 专题 Day 6](../cute/README.md) 与 [FlashAttention-3 论文精读](../../paper/flashattention3/README.md) §3，Hopper 的三个独立执行单元：

| 单元 | 指令 | 异步性 | 占 SM？ |
|------|------|--------|---------|
| **TMA** | `cp.async.bulk.tensor` | 硬件异步，1 thread 发射 | 否 |
| **WGMMA** | `wgmma.mma_async.sync.aligned.m64nNk32` | 异步，warpgroup 发射后立即返回 | 是（Tensor Core） |
| **CUDA core / SFU** | `add`/`mul`/`exp` | 同步 | 是 |

> 💡 **关键洞察**：三者可同时工作——TMA 搬数时 Tensor Core 在算上一个 tile，CUDA core 在做 scale 乘法。DeepGEMM 的 kernel 设计目标就是让三者不互相等待。

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

#### 关键断言

```cpp
DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
// C/D type: only FP32 with accumulation
DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");
```

- TMA warpgroup 固定 128 线程（1 warpgroup）
- Math warpgroup 是 128 的倍数（1-2 warpgroups）
- BLOCK_K 固定 128（per-128-channel scaling）
- 累加只在 FP32

#### SMEM 分配

```
SMEM 布局（kNumStages = 3 为例）：
┌─────────────────────────────────────────────────┐
│ Tensor Maps (KGrouped 模式才用)                  │
├─────────────────────────────────────────────────┤
│ D: BLOCK_M × BLOCK_N × FP32                     │ ← 输出 tile
├─────────────────────────────────────────────────┤
│ A: [stage 0][stage 1][stage 2] × BLOCK_M×128×FP8│ ← 3-stage A buffer
├─────────────────────────────────────────────────┤
│ B: [stage 0][stage 1][stage 2] × BLOCK_N×128×FP8│ ← 3-stage B buffer
├─────────────────────────────────────────────────┤
│ SFA: [stage 0][stage 1][stage 2] × BLOCK_M×FP32 │ ← A 的 scale
├─────────────────────────────────────────────────┤
│ SFB: [stage 0][stage 1][stage 2] × BLOCK_N×FP32 │ ← B 的 scale
├─────────────────────────────────────────────────┤
│ full_barriers[3]  +  empty_barriers[3]          │ ← mbarrier 对
└─────────────────────────────────────────────────┘
```

> 💡 **关键洞察**：DeepGEMM 把 **scale factor 也放进 SMEM 流水线**——每个 stage 不仅有 A/B 数据，还有对应的 SFA/SFB。这样 scale 读取与 WGMMA 计算完全 overlap，不会因为 scale 加载引入额外延迟。

### 学习任务 3：TMA 加载与 mbarrier 握手（45 分钟）

读 `sm90_fp8_gemm_1d1d.cuh:170-245`（TMA warpgroup 分支）：

```cpp
// TMA warpgroup：1 个 thread 发起所有 TMA
if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        // 持久化调度：一个 threadblock 串行处理多个 tile
        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            // 1. 等 consumer 释放 buffer
            empty_barriers[stage_idx]->wait(phase ^ 1);

            // 2. 发起 4 个 TMA：SFA, SFB, A, B
            tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage], m_idx, sf_k_idx, num_multicast);
            tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage], n_idx, sf_k_idx, num_multicast);
            tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(tensor_map_a, &full_barrier, smem_a[stage], k_idx, m_idx, num_multicast);
            tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(tensor_map_b, &full_barrier, smem_b[stage], k_idx, n_idx, num_multicast);

            // 3. 告知 barrier 期望的字节数，TMA 完成后自动 release
            full_barrier.arrive_and_expect_tx(SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB);
        }
    }
}
```

#### mbarrier 的双 barrier 设计

DeepGEMM 用**成对 barrier**——`full_barriers` 和 `empty_barriers` 各 `kNumStages` 个：

| Barrier | 生产者 | 消费者 | 含义 |
|---------|--------|--------|------|
| `full_barriers[stage]` | TMA `arrive_and_expect_tx` | Math `wait` | "数据已满，可算" |
| `empty_barriers[stage]` | Math `arrive` | TMA `wait` | "数据用完，可覆盖" |

```
时间 →
TMA:   load stage0 → arrive(full[0]) → load stage1 → arrive(full[1]) → wait(empty[0]) → load stage0' → ...
Math:                              wait(full[0]) → WGMMA → arrive(empty[0]) → wait(full[1]) → ...
                                                       ↑ 两者在不同 stage 上并行
```

> 💡 **关键洞察**：双 barrier 是无锁流水线的标准模式——TMA 不需要知道 Math 何时算完，只需等 `empty` 信号；Math 不需要知道 TMA 何时搬完，只需等 `full` 信号。两者完全解耦，流水线深度由 `kNumStages` 决定（典型 3-4 stage）。

### 学习任务 4：WGMMA 发射与 scale 乘法（30 分钟）

读 `sm90_fp8_gemm_1d1d.cuh:246-321`（Math warpgroup 分支）：

```cpp
// Math warpgroup
cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
    float2 scales_b[WGMMA::kNumAccum / 4];

    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        // 1. 等 TMA 搬完
        full_barriers[stage_idx]->wait(phase);

        // 2. 读 scale（必须在 warpgroup_arrive 前读完，避免下一 tile 污染）
        auto scale_a_0 = ptx::ld_shared(smem_sfa[stage] + r_0);
        auto scale_a_1 = ptx::ld_shared(smem_sfa[stage] + r_1);
        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
            scales_b[i] = ptx::ld_shared(...smem_sfb[stage]...);

        // 3. 发射 WGMMA（BLOCK_K / WGMMA::K = 128/32 = 4 次）
        ptx::warpgroup_arrive();                         // wgmma.fence
        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
            auto desc_a = mma::sm90::make_smem_desc(smem_a[stage] + ..., 1);
            auto desc_b = mma::sm90::make_smem_desc(smem_b[stage] + ..., 1);
            WGMMA::wgmma(desc_a, desc_b, accum, k);      // wgmma.mma_async
        }
        ptx::warpgroup_commit_batch();                    // wgmma.commit_group
        ptx::warpgroup_wait<0>();                         // wgmma.wait_group 0

        // 4. 释放 buffer
        empty_barrier_arrive(stage_idx);

        // 5. Scale 乘法（在下一轮 TMA 搬运的影子里）
        for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
            final_accum[i*4+0] += scale_a_0 * scale_b_0 * accum[i*4+0];
            final_accum[i*4+1] += scale_a_0 * scale_b_1 * accum[i*4+1];
            final_accum[i*4+2] += scale_a_1 * scale_b_0 * accum[i*4+2];
            final_accum[i*4+3] += scale_a_1 * scale_b_1 * accum[i*4+3];
        }
    }
    // ... epilogue: 写回 SMEM → TMA store ...
}
```

#### WGMMA 指令选择

读 `mma/sm90.cuh`，FP8 的 WGMMA 是 `m64nNk32`（M=64 固定，N 从 8 到 256，K=32）：

```cpp
template <int N>
struct FP8MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8)   return MMA_64x8x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_SS_TN();
        if constexpr (N == 256) return MMA_64x256x32_F32E4M3E4M3_SS_TN();
        // ... N 从 8 到 256，步长 8 ...
    }
};
```

> ⚠️ **注意**：FP8 WGMMA 的 K=32（4 个 FP8 pack 成 32 字节），BF16 WGMMA 的 K=16。`BLOCK_K=128` 对应 FP8 的 4 次 WGMMA、BF16 的 8 次。

### 今日检查清单

- [ ] 能说出 kernel 的关键模板参数（BLOCK_K=128、kNumTMAThreads=128）
- [ ] 能画出 SMEM 布局（D / A×stage / B×stage / SFA×stage / SFB×stage / barriers）
- [ ] 能解释 `full_barriers` / `empty_barriers` 双 barrier 的握手时序
- [ ] 能说出 FP8 WGMMA 是 `m64nNk32`，BLOCK_K=128 对应 4 次 WGMMA
- [ ] 读懂 TMA 分支与 Math 分支的代码结构

---

