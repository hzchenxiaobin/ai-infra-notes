# Day 5（周五）：Grouped GEMM for MoE——三种布局与调度

> **本周定位**：本专题是 [CUTLASS 专题](../cutlass/README.md)（库视角）与 [CuTe 专题](../cute/README.md)（原语视角）之后的**单点深钻**——拆开一个生产级 FP8/FP4 GEMM kernel 看每一行 PTX 怎么写。
> **前置要求**：已完成 Day 3-4（SM90 FP8 GEMM 源码精读），理解持久化调度、TMA multicast、K-grouped 的 `tensormap.replace` 机制
> **今日目标**：精读 DeepGEMM 为 MoE 设计的三种 Grouped GEMM 布局——`MGroupedContiguous`（prefill）、`MGroupedMasked`（decode + CUDA graph）、`KGroupedContiguous`（权重梯度反向），搞清 `grouped_layout` tensor 在三种布局下的不同含义、调度器分支、psum layout 优化、`mk_alignment` 对齐契约
> **时间投入**：2.5h（早间 1.5h 精读调度器三种分支 + 晚间 1h 跑 grouped benchmark）
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"DeepGEMM 的 Grouped GEMM 为什么只分组 M 轴"必问

---

## 本日在本周知识图谱中的位置

```
Day 1          Day 2           Day 3-4            Day 5           Day 6          Day 7
 总览      →   FP8/FP4     →   SM90 Kernel   →   Grouped      →  SM100/Mega  →  调优
 JIT 环境      Scaling         源码精读           GEMM for MoE     MoE            ncu
 源码地图      per-128-ch      TMA+WGMMA          contiguous/      TCgen05        报告
                UE8M0           持久化调度          masked/k-group   EP 融合
                                                     ↑
                                                     你在这里（MoE 场景的三种 Grouped GEMM 布局）
```

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| 三种 Grouped GEMM 布局的 `grouped_layout` 含义 | ④ 能说出 DeepGEMM 的 Grouped GEMM 只分组 M 轴的设计原因 |
| `MGroupedContiguous` / `MGroupedMasked` / `KGroupedContiguous` 调度器分支 | ④ 同上（调度器是设计的核心） |
| psum layout 优化与 `mk_alignment` 契约 | ④ 同上（psum 是 contiguous 的内存优化） |
| Grouped GEMM benchmark 数据 | ③ Grouped GEMM for MoE benchmark（验收 ③ 的一部分） |

---

### 学习任务 1：MoE 场景与三种 Grouped GEMM 的动机（30 分钟）

#### MoE 为什么需要 Grouped GEMM

DeepSeek-V3 是 671B 参数的 MoE 模型，每个 token 激活 8 个 expert（共 256 个）。一次前向相当于 **8 个独立 GEMM**（每个 expert 处理一批 token），但每个 expert 收到的 token 数**动态变化**——可能是 0 到几千。

| 方案 | 问题 |
|------|------|
| 启动 256 个独立 GEMM kernel | kernel launch 开销爆炸；小 expert 算不满 SM |
| 用最大 batch padding 到 [256, max_m, K] | padding 浪费算力（很多 expert 收 0 个 token） |
| **Grouped GEMM**（单 kernel 跨 expert 调度） | ✓ 一次 launch，持久化调度让 SM 自己分配 expert 与 tile |

#### DeepGEMM 的三种 Grouped GEMM 布局

读 `common/types.cuh:20-44`，6 种 `GemmType` 里 4 种是 grouped：

| GemmType | API | `grouped_layout` 含义 | 典型场景 |
|----------|-----|----------------------|---------|
| `MGroupedContiguous` | `m_grouped_fp8_gemm_nt_contiguous` | `[M]` per-row group id（-1 = padding） | MoE 前向 prefill |
| `MGroupedContiguousWithPsumLayout` | 同上 + `use_psum_layout=True` | `[num_groups]` 累积 m 偏移 | MoE 前向 prefill（内存优化） |
| `MGroupedMasked` | `m_grouped_fp8_gemm_nt_masked` | `[num_groups]` 每 group 有效 m | MoE decode + CUDA graph |
| `KGroupedContiguous` | `k_grouped_fp8_gemm_nt_contiguous` | `[num_groups]` 每 group 的 K 长度 | MoE 权重梯度反向 wgrad |
| `KGroupedContiguousWithPsumLayout` | 同上 + `use_psum_layout=True` | `[num_groups]` 累积 k 偏移 | MoE wgrad（SM100 内存优化） |
| `Normal` / `Batched` | `fp8_gemm_*` / BMM | 无 | 普通 GEMM / 批量 GEMM |

> 💡 **一句话总结**：DeepGEMM 的 Grouped GEMM **只分组 M 轴或 K 轴**（N/K 对 M-grouped 固定，M/N 对 K-grouped 固定），因为 MoE 的 expert 共享同一形状的权重 `[N, K]`。这避开了 CUTLASS `GemmGroup` 那种"每 group 不同 M/N/K"的通用但低效的设计——**专用化换性能**。

#### 三种布局的数据形状对比

| 布局 | A 形状 | B 形状 | D 形状 | `grouped_layout` |
|------|--------|--------|--------|------------------|
| `MGroupedContiguous` | `[M, K]`（拼接所有 group） | `[num_groups, N, K]` | `[M, N]` | `[M]` int32，每行 group id |
| `MGroupedMasked` | `[num_groups, max_m, K]` | `[num_groups, N, K]` | `[num_groups, max_m, N]` | `[num_groups]` int，每 group 有效 m |
| `KGroupedContiguous` | `[sum_k, M]` | `[sum_k, N]` | `[num_groups, M, N]` | `[num_groups]` int32，每 group 的 K |

> ⚠️ **注意 K-grouped 的 A/B 形状**：A 是 `[sum_k, M]` 而非 `[M, sum_k]`——因为 wgrad 里 A 是梯度（沿 K 拼接），M/N 是固定的输出维度。这与 M-grouped 的 `[M, K]` 朝向相反。

### 学习任务 2：MGroupedContiguous——MoE 前向 Prefill（45 分钟）

这是 MoE **前向**（forward GEMM: `D = A @ B`）最常用的布局。prefill 阶段 CPU 知道每个 expert 收到多少 token，把所有 expert 的 token **拼接成连续张量** `[M, K]`。

#### 数据布局与 `grouped_layout` 含义

读 `tests/generators.py:329-368`（`generate_m_grouped_contiguous`），理解 contiguous 布局的构造：

```python
# generators.py:334-357
actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
m = sum(aligned_ms)

a = torch.randn((m, k), ...)                          # [M, K] 拼接所有 group
b = torch.randn((num_groups, n, k), ...)              # [num_groups, N, K] 每 expert 一份
grouped_layout = torch.empty(m, ..., dtype=torch.int32)  # [M] per-row group id

start = 0
for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
    actual_end = start + actual_m
    aligned_end = start + aligned_m
    grouped_layout[start: actual_end] = i              # 有效行标记为 group i
    grouped_layout[actual_end: aligned_end] = -1       # padding 行标记为 -1
    a[actual_end: aligned_end] = 0                     # padding 行清零
    start = aligned_end
```

- **A**：`[M, K]`，M = 所有 group 的对齐后 m 之和。group 0 的 token 在 `[0, aligned_m_0)`，group 1 在 `[aligned_m_0, aligned_m_0 + aligned_m_1)`，以此类推
- **B**：`[num_groups, N, K]`，每个 expert 一个独立的权重矩阵
- **`grouped_layout`**：`[M]` 的 int32，每行存该行属于哪个 group（padding 行是 -1）

> 💡 **为什么 padding 行标 -1？** 因为 group id 是非负的，-1 是哨兵值。kernel 内 `is_computation_valid` 检查 `grouped_layout[m_offset] >= 0` 决定是否跳过该 tile 的计算（`scheduler/gemm.cuh:314-316`）。

#### TMA descriptor 的构造差异

读 `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:185-201`（M-grouped 的 TMA 构造）：

```cpp
// A: 普通 2D，num_groups=1（A 是拼接的，TMA 不感知 group 边界）
const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k, ..., 1, swizzle_a_mode);

// B: num_groups 折叠到 outer 维度（TMA 通过 group_idx 切换 expert）
const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k, ..., num_groups, swizzle_b_mode);

// D: 普通 2D，num_groups=1（D 也是拼接的）
const auto tensor_map_d = make_tma_cd_desc(d, m, n, ..., 1, swizzle_cd_mode);
```

- A 和 D 的 TMA `num_groups=1`——它们是拼接的连续张量，TMA 不需要感知 group 边界
- B 的 TMA `num_groups=num_groups`——`make_tma_b_desc` 把 group 维度折叠到 outer dim（`runtime_utils.hpp:222-223`：`gmem_outer_dim * num_groups`），kernel 内用 `current_group_idx` 作为 outer_idx 切换 expert

#### `get_global_idx` 的 group offset

读 `scheduler/gemm.cuh:155-186`，理解 `kWithGroupOffset` 模板参数的作用：

```cpp
} else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
    const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
    return offset * shape_dim + block_idx * block_size;
}
```

- `kWithGroupOffset=true`：给 B 加 `group_id * shape_n` 偏移——`group_id` 从 `grouped_layout[m_block_idx * BLOCK_M]` 读取（取 tile 第 0 行的 group id）
- `kWithGroupOffset=false`：A 和 D 不加偏移（它们是连续拼接的，地址就是 `block_idx * block_size`）

> 💡 **关键设计**：M-grouped 的 A/D 是**物理连续**的，TMA 用线性地址访问；只有 B 需要"按 group 跳跃"——所以 `kWithGroupOffset` 只对 B 为 true。`cute::max(0, ...)` 处理 padding 行（-1 会被 clamp 到 0，但 `is_computation_valid` 会跳过这些 tile）。

#### `is_computation_valid` 跳过 padding tile

读 `scheduler/gemm.cuh:311-324`：

```cpp
CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
    if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
    }
    // ...
}
```

kernel 内（`sm90_fp8_gemm_1d2d.cuh:274`）：

```cpp
if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
    // 正常 WGMMA + promote
} else {
    // padding tile：只 wait barrier + arrive empty，不计算
    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
        full_barriers[stage_idx]->wait(phase);
        empty_barrier_arrive();
    }
}
```

- padding tile 仍要走流水线（TMA 依然加载数据，只是 A 的 padding 行是 0），但跳过 WGMMA 与 promote
- 这样设计是为了**不破坏流水线节拍**——如果 padding tile 直接 skip，TMA warpgroup 会因为 empty barrier 不被 arrive 而死锁

#### Multicast on B 的 group 一致性检查

读 `scheduler/gemm.cuh:298-305`（Day 4 已讲，这里强调 M-grouped 的特殊性）：

```cpp
if constexpr (kIsMulticastOnA) {
    return true;
} else {
    // multicast on B：要求 partner CTA 的 M 行属于同一 group
    const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
    const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
    return group_idx == peer_group_idx;
}
```

> ⚠️ **M-grouped + multicast on B 的独特约束**：两个 CTA 共享 B 块时，B 是某个 expert 的权重——如果两个 CTA 的 M 行属于不同 group，它们需要不同的 B，multicast 没意义。代码用 `m_block_idx ^ 1` 找 partner CTA 的 m_block_idx，对比两者的 `grouped_layout` 值。其他 GemmType 默认 multicast 有效，因为 B 与 group 无关（Normal）或每个 group 独立（masked/k-grouped）。

### 学习任务 3：MGroupedMasked——MoE Decode + CUDA Graph（45 分钟）

这是 MoE **decode** 阶段的布局。decode 时 batch_size 小（如 1），CUDA graph 捕获固定 shape 的 kernel，但每个 expert 收到的 token 数动态变化。

#### 数据布局与 `masked_m` 含义

读 `tests/generators.py:382-410`（`generate_m_grouped_masked`）：

```python
a = torch.randn((num_groups, max_m, k), ...)          # [G, max_m, K] 每个 group 独立
b = torch.randn((num_groups, n, k), ...)              # [G, N, K]
d = torch.empty((num_groups, max_m, n), ...)          # [G, max_m, N]

masked_m = torch.empty((num_groups,), dtype=torch.int)
for j in range(num_groups):
    masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))  # 每 group 有效 m
```

- A/B/D 都是 `[num_groups, max_m, *]` 三维——每个 group 独立的 padded 块
- `masked_m[g]` = group g 的有效 token 数（≤ max_m），超出的 `max_m - masked_m[g]` 行是 padding
- `grouped_layout` 就是 `masked_m` 本身（`[num_groups]` int）

#### 为什么 decode 用 masked 而非 contiguous

| 维度 | Contiguous（prefill） | Masked（decode） |
|------|----------------------|------------------|
| A 形状 | `[M, K]`（动态拼接） | `[G, max_m, K]`（固定 shape） |
| M 总量 | 动态（每 batch 不同） | 固定 `G * max_m` |
| CUDA graph | ✗（shape 变化） | ✓（shape 固定） |
| padding 浪费 | 无（精确拼接） | 有（每 group padded 到 max_m） |
| CPU 知道 token 分布 | ✓ | ✗（only GPU 知道 `masked_m`） |

> 💡 **关键洞察**：decode 用 masked 是为了 **CUDA graph 兼容**。CUDA graph 要求 kernel 的 grid/block shape 固定，但 decode 时每 expert 的 token 数动态变化——contiguous 布局的 M 会变，破坏 graph。masked 布局把 A padded 到 `[G, max_m, K]`，shape 固定，`masked_m` 作为 kernel 参数动态传入（GPU 端读取），graph 可重放。

#### 调度器的跨 group 累积分配

读 `scheduler/gemm.cuh:200-216`（MGroupedMasked 的 `get_next_block` 分支）：

```cpp
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    while (true) {
        // End of the task
        if (current_group_idx == kNumGroups)
            return false;

        // Within current group
        num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
        const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
        if (next_block_idx < current_m_block_cumsum * num_n_blocks)
            break;

        // Move to check the next group
        current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
    }
    get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
}
```

- `grouped_layout[current_group_idx]` = `masked_m[g]`，动态读取每 group 的有效 m
- `num_m_blocks` 每 group 不同（`ceil_div(masked_m[g], BLOCK_M)`），动态计算
- `current_m_block_cumsum` 跨 group 累积——threadblock 用线性 `next_block_idx` 跨 group 分配 tile
- 超出当前 group 范围则 `current_group_idx++` 推进到下一个 group

#### `get_global_idx` 的 group offset

读 `scheduler/gemm.cuh:163-165`：

```cpp
} else if constexpr (kGemmType == GemmType::MGroupedMasked or ...) {
    const auto offset = kWithGroupOffset ? current_group_idx : 0;
    return offset * shape_dim + block_idx * block_size;
}
```

- `kWithGroupOffset=true`：A/SFA/D 都加 `current_group_idx * shape_dim` 偏移——因为它们是 `[G, max_m, *]`，group 维度是 outer
- `current_group_idx` 由调度器在 `get_next_block` 内更新

读 `sm90_fp8_gemm_1d2d.cuh:191-199`，TMA 发射时对 A/SFA 用 `kWithGroupOffsetA=true`：

```cpp
constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, ...>(&tensor_map_a, &full_barrier,
        smem_a[stage_idx], k_idx,
        scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
        num_tma_multicast_a, batch_idx);
```

- Normal/Contiguous：`kWithGroupOffsetA=false`（A 是连续拼接，无 group 偏移）
- Masked：`kWithGroupOffsetA=true`（A 是 `[G, max_m, K]`，需 group 偏移）

#### `is_computation_valid` 检查有效行

读 `scheduler/gemm.cuh:316-318`：

```cpp
} else if constexpr (kGemmType == GemmType::MGroupedMasked) {
    return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
}
```

- 检查 tile 的起始行 `m_offset + m_block_idx * BLOCK_M` 是否小于当前 group 的 `masked_m`
- 超出的 tile 跳过 WGMMA（与 contiguous 的 padding 处理一致）

#### 配合 DeepEP 低延迟 kernel

读 README "Grouped GEMMs (masked layout)" 一节：

> During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions. ... an example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

- DeepEP 的低延迟 EP（Expert Parallelism）dispatch kernel 把 token 散到 `[G, max_m, K]` 布局，并返回 `masked_m`
- DeepGEMM 的 masked GEMM 直接消费这个布局，无需 reshape
- 这是 DeepSeek 推理栈的"算子级接力"——EP 通信 + GEMM 融合在数据布局上

### 学习任务 4：KGroupedContiguous——MoE 权重梯度反向（45 分钟）

这是 MoE **反向 wgrad**（`D = A.T @ B`，A 是前向激活梯度，B 是前向输入）的布局。沿 **K 轴**分组，每个 group 的 K 长度不同（对应不同 expert 的 token 数）。

#### 为什么 wgrad 沿 K 分组

前向是 `Y = X @ W`（M-grouped，X 沿 M 拼接）。反向 wgrad 是 `dW = X.T @ dY`：
- `X.T` 的形状 `[K, M]`——K 维拼接不同 expert 的 token（每 expert 的 token 数不同 → K 不同）
- `dY` 的形状 `[K, N]`——K 维同样拼接
- `dW` 的形状 `[num_groups, M, N]`——每个 expert 独立的权重梯度

所以 K-grouped 的"分组轴"是 K，每个 group 的 K 长度 = 该 expert 收到的 token 数。

#### 数据布局与 `ks_cpu` 含义

读 `tests/generators.py:438-479`（`generate_k_grouped_contiguous`）：

```python
k = sum(ks_cpu)
grouped_layout = torch.tensor(ks_cpu, ..., dtype=torch.int32)  # [num_groups] 每 group 的 K

a = torch.randn((k, m), ...)    # [sum_k, M] A.T 沿 K 拼接
b = torch.randn((k, n), ...)    # [sum_k, N] B 沿 K 拼接
c = torch.randn((num_groups, m, n), ...) * 32   # [G, M, N] 每 group 独立 C
d = c                            # D = C（with_accumulation）

start = 0
for i, group_k in enumerate(ks_cpu):
    end = start + group_k
    ref_d[i] = c[i] + (a[start:end].T @ b[start:end])   # 每 group 独立 GEMM
    start = end
```

- A/B 沿 K 拼接成 `[sum_k, *]`，`ks_cpu[g]` 是 group g 的 K 长度
- D 是 `[num_groups, M, N]`——每 group 独立输出（与 M-grouped 的拼接 D 不同）
- C 提供累积初值（wgrad 通常 `D = C + A.T @ B`）

#### SM90 的 `k_grouped_fp8_gemm_nt_contiguous`

读 `csrc/apis/gemm.hpp:348-400`（SM90 NT 路径）：

```cpp
static void k_grouped_fp8_gemm_nt_contiguous(...) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    // No psum on FP8 NT
    DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());
    // ...
    // Allocate tensormap buffer
    const auto num_sms = device_runtime->get_num_sms();
    const auto tensor_map_buffer = torch::empty({num_sms * 4 * sizeof(CUtensorMap)}, ...);
    // Dispatch
    if (arch_major == 9) {
        sm90_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n,
                                      ks_cpu.value(), grouped_layout, tensor_map_buffer, ...);
    }
}
```

- SM90 K-grouped 用 **1D1D kernel**（FP32 accumulate，wgrad 需要高精度累加）
- 强制 `recipe = (1, 1, 128)`：per-token A + per-channel B + gran_k=128
- 分配 `tensor_map_buffer`（每 SM 一份，用于 `tensormap.replace`，Day 4 已讲）
- SM90 NT 路径不支持 psum layout（注释 `:361`）

#### 调度器的跨 group 分配

读 `scheduler/gemm.cuh:238-261`（KGroupedContiguous 的 `get_next_block` 分支）：

```cpp
} else if constexpr (is_k_grouped_contiguous(kGemmType)) {
    while (true) {
        if (current_group_idx == kNumGroups)
            return false;

        // Within current group
        if (next_block_idx < (current_num_valid_groups + 1) * num_blocks)
            break;

        // Move to check the next group
        current_sf_k_cumsum += math::ceil_div(current_shape_k, kSFKSpan);
        current_num_valid_groups ++;
        if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            get_next_psum_k_group(++ current_group_idx, current_shape_k, current_k_start, current_k_end);
        } else {
            current_k_cumsum += current_shape_k;
            current_group_idx = next_group_idx ++;
            current_shape_k = next_shape_k;
            get_next_k_group(next_group_idx, next_shape_k);
        }
    }
    get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_blocks, m_block_idx, n_block_idx);
}
```

- `current_num_valid_groups`：已处理的 group 数
- `current_shape_k`：当前 group 的 K 长度（每 group 不同）
- `current_k_cumsum`：K 维累积偏移（用于 A/B 的 gmem 地址）
- `current_sf_k_cumsum`：SF K 维累积偏移（`ceil_div(shape_k, kSFKSpan)`，`kSFKSpan=128` 是 SF 的 K 跨度）
- `next_group_idx` / `next_shape_k`：预取下一个 group 的信息（避免 `get_next_k_group` 的循环开销）

#### `get_next_k_group` 跳过空 group

读 `scheduler/gemm.cuh:66-72`：

```cpp
CUTLASS_DEVICE void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
    for (; group_idx < kNumGroups; ++ group_idx) {
        shape_k = grouped_layout[group_idx];
        if (shape_k > 0)
            break;
    }
}
```

- 遍历 `grouped_layout`（ks_cpu）找下一个 `shape_k > 0` 的 group
- 空 group（`ks_cpu[g] == 0`，某 expert 收到 0 个 token）被跳过——不消耗 tile 配额

#### `get_global_idx` 的 K 偏移

读 `scheduler/gemm.cuh:166-180`：

```cpp
} else if constexpr (is_k_grouped_contiguous(kGemmType)) {
    auto offset = 0;
    if constexpr (kWithGroupOffset) {
        if constexpr (kIndexType == IndexType::MN) {
            offset = current_group_idx * shape_dim;        // D: [G, M, N]，group 维是 outer
        } else if constexpr (kIndexType == IndexType::K) {
            offset = current_k_cumsum;                      // A/B: 沿 K 拼接
        } else if constexpr (kIndexType == IndexType::SF_K) {
            offset = current_sf_k_cumsum;                   // SFA/SFB: SF 沿 K 拼接
        }
    }
    return offset + block_idx * block_size;
}
```

- `IndexType::MN`（D）：加 `current_group_idx * shape_dim`——D 是 `[G, M, N]`，每 group 独立
- `IndexType::K`（A/B）：加 `current_k_cumsum`——A/B 沿 K 拼接
- `IndexType::SF_K`（SFA/SFB）：加 `current_sf_k_cumsum`——SF 也沿 K 拼接，但跨度是 `ceil_div(shape_k, 128)`（每 128 K 一个 SF）

#### Day 4 的 `tensormap.replace` 在此登场

K-grouped 每 group 的 K 长度不同，TMA descriptor 的 K 维度与 gmem 基址需要动态更新。这正是 Day 4 学习任务 4 讲的 `tensormap.replace` 流程（`sm90_fp8_gemm_1d1d.cuh:191-215`）：

```cpp
if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
    last_group_idx = scheduler.current_group_idx;
    const uint64_t current_k_offset = scheduler.current_k_cumsum;
    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m);
    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n);
    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a, scheduler.current_shape_k, ...);
    // ... 7 步流程（见 Day 4）
}
```

> 💡 **Day 4 与 Day 5 的衔接**：Day 4 讲了 `tensormap.replace` 的 PTX 机制，Day 5 把它放进 MoE wgrad 的完整数据流——每 group 切换时更新 A/B 的 TMA descriptor，让 TMA 加载正确的 K 范围。这是"算子级 shape 变化"在 device 端的原地处理，无需 host 重新构造 descriptor。

### 学习任务 5：Psum Layout——减少 padding 浪费（30 分钟）

`MGroupedContiguousWithPsumLayout` 和 `KGroupedContiguousWithPsumLayout` 是 v2.6+ 引入的内存优化布局，把 masked 的 padded 块**折叠成连续的 psum（prefix-sum）布局**。

#### Psum layout 的核心思想

以 M-grouped psum 为例，对比：

```
普通 contiguous:          psum layout:
group 0: [0, m_0)         group 0: [0, m_0)
group 1: [m_0, m_0+m_1)   group 1: [align(m_0), align(m_0)+m_1)   ← 对齐后起点
group 2: [...]            group 2: [align(align(m_0)+m_1), ...)
```

- 普通 contiguous：每 group 起点对齐到 `mk_alignment`，padding 行填 0
- psum layout：`grouped_layout[g]` 存**累积的 end offset**（`m_0`, `m_0+m_1`, ...），起点是 `align(prev_end)`

读 `scheduler/gemm.cuh:74-85`（`get_next_psum_k_group`，K-grouped psum 版本）：

```cpp
CUTLASS_DEVICE void get_next_psum_k_group(uint32_t &group_idx, uint32_t &shape_k,
                                           uint32_t &k_start, uint32_t &k_end) const {
    for (; group_idx < kNumGroups; ++ group_idx) {
        const auto next_k_end = static_cast<uint32_t>(grouped_layout[group_idx]);
        k_start = math::align(k_end, kKAlignment);           // 起点对齐到 kKAlignment
        shape_k = next_k_end - k_start;                      // 有效 K = end - aligned_start
        k_end = next_k_end;
        if (shape_k > 0)
            break;
    }
}
```

- `grouped_layout[g]` = group g 的累积 end offset（psum）
- `k_start = align(prev_k_end, kKAlignment)`——起点对齐
- `shape_k = k_end - k_start`——有效长度（可能 < kKAlignment，因为是 psum 而非 aligned）

#### Psum layout 的优势

| 维度 | 普通 contiguous | psum layout |
|------|----------------|-------------|
| padding 行 | 每 group 末尾 padding 到 `mk_alignment` | 只在 group **起点**对齐，end 不对齐 |
| 总 M（或 K） | `sum(align(m_g))` | `align(last_end)`，通常更小 |
| `grouped_layout` | per-row group id（M-grouped）或 per-group K（K-grouped） | per-group 累积 end offset |
| 内存浪费 | 每 group 都有 padding | 仅 group 间对齐 gap |
| 支持 | SM90 + SM100 | SM100 为主（SM90 M-grouped psum 也支持） |

> 💡 **关键洞察**：psum layout 把 padding 从"每 group 末尾"挪到"group 间对齐 gap"，总 padding 量从 `num_groups * (align(m) - m)` 降到 `num_groups * kKAlignment`。对于 group 数多、每 group m 小的场景（如 EP16 的 16 个 expert），节省显著。

#### `ensure_zero_padding` 契约

读 `csrc/apis/gemm.hpp:176-177` 与 `tests/utils.py:6-14`：

```python
def assert_psum_zero_padding(a, d, grouped_layout, dtype_label):
    for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
        aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
        if current_m < aligned_m:
            a_padding = a_data[current_m: aligned_m]
            d_padding = d[current_m: aligned_m]
            assert torch.equal(a_padding, torch.zeros_like(a_padding)), f'nonzero {dtype_label} input padding'
            assert torch.equal(d_padding, torch.zeros_like(d_padding)), f'nonzero {dtype_label} output padding'
```

- `ensure_zero_padding=True`（默认）：要求用户保证 padding 行的 A 与 D 是 0
- 原因：psum layout 下 SFA 的 packing kernel 会跳过 gap 行，但 A/D 的 padding 必须是 0 才能保证数值正确（否则 WGMMA 会算进非零 padding）
- SM100 才需要这个契约（SM90 的 1D2D kernel 不依赖此假设）

#### `get_aligned_effective_m_in_block`：处理最后一块

读 `scheduler/gemm.cuh:189-195`：

```cpp
CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
    constexpr uint32_t UMMA_STEP_N = 16;
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
        return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ?
                           current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
    return BLOCK_M;
}
```

- psum layout 的最后一个 block 可能不满 `BLOCK_M`（因为 end 不对齐）
- `not kEnsureZeroPadding` 时：返回对齐到 `UMMA_STEP_N=16` 的有效 m（避免 WGMMA 越界）
- `kEnsureZeroPadding=true` 时：直接返回 `BLOCK_M`（padding 是 0，安全）

### 学习任务 6：mk_alignment 与 Heuristics（30 分钟）

#### `mk_alignment_for_contiguous_layout` 契约

contiguous layout 要求每 group 的 M（M-grouped）或 K（K-grouped）**起点对齐到 `mk_alignment`**。读 `csrc/jit_kernels/heuristics/runtime.hpp:39-57`：

```cpp
void set_mk_alignment_for_contiguous_layout(const int& new_value) {
    mk_alignment_for_contiguous_layout = new_value;
}

static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
    if (device_runtime->get_arch_major() != 10)
        return kLegacyMKAlignmentForContiguousLayout;      // SM90: 固定 128

    int block_m = 224, mma_step = 32;
    if (expected_m.has_value()) {
        for (; block_m > 32 and block_m - mma_step >= expected_m.value(); block_m -= mma_step);
    }
    return block_m;
}
```

- SM90：固定 128（`kLegacyMKAlignmentForContiguousLayout`）
- SM100：动态 32~224，根据 `expected_m` 选最小的能覆盖 m 的 block_m
- 用户调用 `set_mk_alignment_for_contiguous_layout(alignment)` 设置，必须 ≥ theoretical 值

#### Heuristics 对 grouped 的特殊处理

读 `csrc/jit_kernels/heuristics/sm90.hpp:16-36`，M-grouped contiguous 的 block_m 候选：

```cpp
if (desc.gemm_type == GemmType::MGroupedContiguous or
    desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
    block_m_candidates = std::vector{heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
}
```

- M-grouped 的 `block_m` **必须等于 `mk_alignment`**——因为 group 边界要对齐到 block_m 倍数
- 这限制了 block_m 候选只有 1 个（SM90 是 128，SM100 是 32~224 动态）
- 普通 GEMM 的 block_m 候选是 `{64, 128}`（甚至 256），M-grouped 牺牲灵活性换 group 边界对齐

> ⚠️ **为什么 block_m 必须等于 mk_alignment？** 如果 block_m < mk_alignment，group 边界可能落在 block 中间，一个 tile 跨两个 group——但 B 只能加载一个 group 的权重，跨 group 的 tile 语义错误。如果 block_m > mk_alignment，group 起点不在 block 边界，同样问题。所以 `block_m == mk_alignment` 是 group 边界对齐的必要条件。

#### Test 里的 alignment 选择

读 `tests/test_fp8_fp4.py:76-77`：

```python
alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
```

- 先调 `get_theoretical_mk_alignment_for_contiguous_layout()` 获取理论最小值
- 再 `set_mk_alignment_for_contiguous_layout(alignment)` 设置
- 这是用户的典型用法——用理论最小值，让 block_m 尽可能小（更多并行度）

### 学习任务 7：三种布局完整对照与 Dispatch 流程（30 分钟）

#### 三种 Grouped GEMM 完整对照表

| 维度 | MGroupedContiguous | MGroupedMasked | KGroupedContiguous |
|------|-------------------|----------------|---------------------|
| **场景** | MoE 前向 prefill | MoE decode + CUDA graph | MoE 反向 wgrad |
| **分组轴** | M | M | K |
| **A 形状** | `[M, K]` 拼接 | `[G, max_m, K]` | `[sum_k, M]` |
| **B 形状** | `[G, N, K]` | `[G, N, K]` | `[sum_k, N]` |
| **D 形状** | `[M, N]` 拼接 | `[G, max_m, N]` | `[G, M, N]` |
| **`grouped_layout`** | `[M]` per-row group id | `[G]` 每 group 有效 m | `[G]` 每 group 的 K |
| **TMA A** | 2D，无 group offset | 2D/3D，group offset | 2D + `tensormap.replace` |
| **TMA B** | 2D，group 折叠到 outer | 2D，group 折叠到 outer | 2D + `tensormap.replace` |
| **kernel** | 1D2D（SM90）/ 1D1D（SM100） | 1D2D / 1D1D | 1D1D only |
| **输出 dtype** | BF16（不累加） | BF16（不累加） | FP32（累加） |
| **psum layout** | ✓（MGroupedContiguousWithPsumLayout） | ✗（SM90）/ ✓（SM100 转 contiguous） | ✓（KGroupedContiguousWithPsumLayout） |
| **multicast on B 约束** | 两 CTA 同 group | 无（每 group 独立） | 无（每 group 独立） |
| **空 group 处理** | padding 行标 -1 | `masked_m[g]=0` 跳过 | `ks_cpu[g]=0` 跳过 |

#### Dispatch 决策树

读 `csrc/apis/gemm.hpp:166-232`（M-grouped）与 `:299-400`（K-grouped），完整的 dispatch 流程：

```
Python: m_grouped_fp8_gemm_nt_contiguous(a, b, d, grouped_layout, ...)
    │
    ▼
csrc/apis/gemm.hpp:166  m_grouped_fp8_fp4_gemm_nt_contiguous()
    │  检查 major_a == K, major_b == K
    │  transform_sf_pair_into_required_layout() 转换 SF 布局
    │
    ├─ arch_major == 9 + sfa is FP32:
    │   └─ sm90_m_grouped_fp8_gemm_contiguous_1d2d()  [1D2D, BF16 输出]
    │       │  GemmDesc{gemm_type = MGroupedContiguous or ...WithPsumLayout}
    │       │  get_best_config<SM90ArchSpec>(desc)
    │       │  block_m = mk_alignment（强制）
    │       │  make_tma_*_desc(B 的 num_groups = num_groups)
    │       └─ JIT 编译 + launch
    │
    └─ arch_major == 10 + sfa is INT (UE8M0):
        └─ sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d()  [1D1D, BF16 输出]
            └─ 类似，但用 SM100 kernel + UE8M0 scale

Python: m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m, ...)
    │
    ▼
csrc/apis/gemm.hpp:250  m_grouped_fp8_fp4_gemm_nt_masked()
    │  检查 major_a == K, major_b == K
    │  transform_sf_pair_into_required_layout()
    ├─ arch_major == 9: sm90_m_grouped_fp8_gemm_masked_1d2d()
    └─ arch_major == 10: sm100_m_grouped_fp8_fp4_gemm_masked_1d1d()

Python: k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks_cpu, grouped_layout, c, ...)
    │
    ▼
csrc/apis/gemm.hpp:348  k_grouped_fp8_gemm_nt_contiguous()  [SM90 only]
    │  recipe = (1, 1, 128) 强制
    │  不支持 psum layout
    │  分配 tensor_map_buffer
    └─ sm90_k_grouped_fp8_gemm_1d1d()  [1D1D, FP32 累加]
        └─ kernel 内 tensormap.replace 切换 group
```

#### 运行 Grouped GEMM Benchmark

```bash
cd DeepGEMM
DG_PRINT_CONFIGS=1 python3 tests/test_fp8_fp4.py
```

```text
# 预期输出（H800，截取 grouped 部分）
Testing m-grouped contiguous GEMM:
 > Perf (num_groups=4, m=32768, n= 6144, k=7168, 1D2D, layout=NT, psum=False, zero_pad=False):
   850 us | 1240 TFLOPS | 1200 GB/s
 > Perf (num_groups=8, m=32768, n= 7168, k=3072, 1D2D, layout=NT, psum=True, zero_pad=True):
   420 us | 1380 TFLOPS | 1450 GB/s

Testing m-grouped masked GEMM:
 > Perf (num_groups=32, expected_m_per_group= 192, n=6144, k=7168, 1D2D, psum=0):
   180 us (max: 220 us) | 850 TFLOPS | 900 GB/s

Testing k-grouped contiguous GEMM:
 > Perf (num_groups=4, m=4096, n=7168, k=8192, gran_k=128, k_alignment=128, psum=0):
   720 us | 1310 TFLOPS | 1100 GB/s
```

> 💡 **性能观察**：① contiguous 比 masked 快 20-30%（无 padding 浪费）；② psum layout 比普通 contiguous 快 5-10%（padding 更少）；③ K-grouped 的 wgrad 性能接近 forward（FP32 累加的开销被 TMA overlap 掩盖）；④ num_groups 越多，单 group m 越小，last-wave 利用率越低。

### 面试题积累（本周目标 10-12 道，今日 4 道）

**Q15：DeepGEMM 的 Grouped GEMM 为什么只分组 M 轴（或 K 轴），不像 CUTLASS GemmGroup 支持每 group 不同 M/N/K？**
> 答：因为 MoE 的 expert 共享同一形状的权重 `[N, K]`——M-grouped 的 N/K 固定，K-grouped 的 M/N 固定。专用化换性能：① TMA descriptor 只需 1 个（B 的 num_groups 折叠到 outer 维），无需每 group 重新构造；② 调度器用线性 `next_block_idx` 跨 group 分配，无需 host 端预排序；③ heuristics 只需对一个 (N, K) 选 config，block_m 强制等于 mk_alignment 保证 group 边界对齐。CUTLASS GemmGroup 的通用性带来每 group 独立 TMA descriptor、独立 config 选择、host 端排序的开销，在 MoE 场景不划算。

**Q16：MGroupedContiguous 和 MGroupedMasked 的区别是什么？什么时候用哪个？**
> 答：① 数据形状：contiguous 的 A 是 `[M, K]` 拼接所有 group，masked 的 A 是 `[G, max_m, K]` 每 group 独立 padded；② `grouped_layout`：contiguous 是 `[M]` per-row group id（-1 = padding），masked 是 `[G]` 每 group 有效 m；③ CUDA graph：contiguous 的 M 动态变化不兼容 graph，masked 的 shape 固定兼容；④ padding：contiguous 无浪费，masked 每 group padded 到 max_m。prefill 用 contiguous（CPU 知道 token 分布，精确拼接），decode 用 masked（shape 固定，CUDA graph 可重放，配合 DeepEP 低延迟 kernel）。

**Q17：K-grouped GEMM 为什么用 1D1D kernel 而 M-grouped forward 用 1D2D？**
> 答：因为 wgrad 需要高精度累加。① 1D1D 输出 FP32 并跨 K-block 累加（`final_accum += scale * accum`），wgrad 的梯度对精度敏感；1D2D 输出 BF16 不累加，forward 的激活对精度容忍度高。② 1D1D 的 B 是 per-channel scale（1D），wgrad 里 B 是激活（per-token scale，等价 per-channel）；1D2D 的 B 是 per-block scale（2D），forward 里 B 是权重（静态，per-block 够用）。③ K-grouped 还需要 `tensormap.replace` 动态切换 group，1D1D kernel 已有这套机制（Day 4 讲的 7 步流程）。

**Q18：psum layout 解决什么问题？`ensure_zero_padding` 契约是什么？**
> 答：psum layout 解决 masked/contiguous 的 padding 浪费——把"每 group 末尾 padding 到 mk_alignment"改成"group 间对齐 gap"，总 padding 从 `num_groups * (align(m) - m)` 降到 `num_groups * kKAlignment`。`grouped_layout[g]` 存累积 end offset，起点是 `align(prev_end, kKAlignment)`。`ensure_zero_padding=True` 要求用户保证 padding 行的 A 与 D 是 0——因为 psum layout 下 SFA packing kernel 跳过 gap 行，但 A/D 的 padding 必须是 0 才能保证 WGMMA 数值正确（否则非零 padding 被算进结果）。SM100 才需要此契约（SM90 的 1D2D 不依赖）。

### 今日检查清单

- [ ] 能说出 MoE 场景需要 Grouped GEMM 的 3 个原因（launch 开销 / padding 浪费 / 持久化调度）
- [ ] 能列出三种 Grouped GEMM 的分组轴、A/B/D 形状、`grouped_layout` 含义
- [ ] 能解释 MGroupedContiguous 的 `grouped_layout` 为何是 `[M]` per-row group id，padding 行标 -1 的作用
- [ ] 能说出 MGroupedContiguous 的 TMA B 为何把 `num_groups` 折叠到 outer 维（A/D 是连续拼接）
- [ ] 能解释 `kWithGroupOffset` 模板参数对不同 GemmType 的不同行为
- [ ] 能写出 MGroupedMasked 的 `get_next_block` 跨 group 累积分配逻辑
- [ ] 能说出 decode 用 masked 而非 contiguous 的原因（CUDA graph 兼容）
- [ ] 能解释 KGroupedContiguous 为什么沿 K 分组（wgrad 的 `dW = X.T @ dY`，X 沿 K 拼接）
- [ ] 能写出 K-grouped 的 `get_global_idx` 三种 IndexType（MN/K/SF_K）的 offset 计算
- [ ] 能说出 `get_next_k_group` 如何跳过空 group（`shape_k > 0` 判断）
- [ ] 能解释 psum layout 与普通 contiguous 的 padding 差异（末尾 padding vs group 间 gap）
- [ ] 能说出 `ensure_zero_padding` 契约的作用与适用架构（SM100）
- [ ] 能解释 `block_m == mk_alignment` 的必要性（group 边界对齐）
- [ ] 能说出 SM90 的 `mk_alignment` 固定 128，SM100 动态 32~224
- [ ] 读完 `scheduler/gemm.cuh:197-324`（三种 grouped 分支）、`csrc/apis/gemm.hpp:166-400`（dispatch）、`tests/generators.py:329-532`（数据生成）
- [ ] 跑通 `test_m_grouped_gemm_contiguous` / `test_m_grouped_gemm_masked` / `test_k_grouped_gemm_contiguous`，记录 TFLOPS

#### 明日预告

Day 6 将转向 **SM100（Blackwell）与 Mega MoE**——精读 `sm100_fp8_fp4_gemm_1d1d.cuh`（538 行，TCgen05 + TMEM + UTCCP）与 `sm100_fp8_fp4_mega_moe.cuh`（1460 行，全库最大的 kernel，单 kernel 融合 EP dispatch + 2×GEMM + SwiGLU + EP combine）。今天理解了 Grouped GEMM 的三种布局，明天要看到 Mega MoE 如何把这些布局**融合进单个 mega-kernel**——用 symmetric memory + NVLink barrier 让通信与计算 overlap。建议今晚先扫一眼 `deep_gemm/mega/__init__.py`（SymmBuffer 与 weight transform）与 `csrc/apis/mega.hpp`（symm buffer 分配逻辑），理解 Mega MoE 的内存模型。

---
