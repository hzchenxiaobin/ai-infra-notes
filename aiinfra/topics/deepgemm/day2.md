# Day 2（周二）：FP8/FP4 数据类型与 Per-128-Channel Scaling

> **本周定位**：本专题是 [CUTLASS 专题](../cutlass/README.md)（库视角）与 [CuTe 专题](../cute/README.md)（原语视角）之后的**单点深钻**——拆开一个生产级 FP8/FP4 GEMM kernel 看每一行 PTX 怎么写。
> **前置要求**：已完成 Day 1（DeepGEMM 总览与 JIT 环境），跑通 `test_fp8_fp4.py`，了解两层架构与源码地图
> **今日目标**：理解 FP8（e4m3/e5m2）与 FP4 的数值分布，掌握 DeepGEMM 的 per-128-channel scaling（SM90 FP32 scale / SM100 UE8M0 packed scale），搞清 KernelType 1D1D vs 1D2D 的区别，串通从用户 quant 到 kernel 内 scale 乘法的完整数据流
> **时间投入**：2.5h（早间 1.5h 精读 + 晚间 1h 跑 demo）
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，FP8 精度策略是 DeepGEMM 的立身之本

---

## 本日在本周知识图谱中的位置

```
Day 1          Day 2           Day 3-4            Day 5           Day 6          Day 7
 总览      →   FP8/FP4     →   SM90 Kernel   →   Grouped      →  SM100/Mega  →  调优
 JIT 环境      Scaling         源码精读           GEMM for MoE     MoE            ncu
 源码地图      per-128-ch      TMA+WGMMA          contiguous/      TCgen05        报告
               UE8M0           持久化调度          masked/k-group   EP 融合
                  ↑
                  你在这里（精度策略：不理解 scaling，读不懂 kernel 里的 promote 循环）
```

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| per-128-channel scaling 的数学形式 | ② 能解释 per-128-channel scaling 的数学形式与 SM100 UE8M0 的差异 |
| SM90 vs SM100 scale 处理对比 | ② 同上 |
| Scale 从用户 quant 到 kernel 乘法的数据流 | ① Day 3-4 画 TMA + Math 时序图的前提（scale 也在流水线里） |

---

### 学习任务 1：FP8 与 FP4 格式（45 分钟）

#### e4m3 vs e5m2 vs FP4

| 格式 | 指数位 | 尾数位 | 最大值 | 表示点 | 用途 |
|------|--------|--------|--------|--------|------|
| **e4m3** | 4 | 3 | ±448 | 448 个非零值 | 前向权重/激活 |
| **e5m2** | 5 | 2 | ±57344 | 128 个非零值，大动态范围 | 反向梯度 |
| **FP4 (E2M1)** | 2 | 1 | ±6.0 | 仅 8 个非零值：`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` | 极致压缩（Blackwell Mega MoE 权重） |

> 💡 DeepGEMM 只用 **e4m3**（不用 e5m2，反向也走 e4m3，靠 per-128-channel scaling 保精度）。FP4 是 E2M1，代码里以 `cutlass::float_e2m1_t` 表示，存储时两个 nibble pack 进一个 `int8`（`csrc/utils/math.hpp:11` 定义 `kPackedFP4 = torch::kInt8`）。

#### FP4 量化代码实证

读 `deep_gemm/utils/math.py:84-122`，FP4 量化用查找表实现（非硬件指令），代码非常直观：

```python
# deep_gemm/utils/math.py:84-93
def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    # {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)
```

- 阈值对应 8 个 FP4 值的**中点**，逐次累加得到 0-7 的码字
- 符号位放 bit 3，所以一个 nibble 是 `[sign | value_idx(3bit)]`
- 两个 nibble pack 成一个 int8：`(codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)`（`math.py:111`）
- FP4 scale 除以 **6.0**（FP4 max），FP8 scale 除以 **448.0**（e4m3 max）：`sf = x_amax / 6.0`（`math.py:106`）

#### Hopper vs Blackwell 算力

| 精度 | H100/H800 峰值 | B200 峰值 | 相对 FP16 |
|------|----------------|-----------|-----------|
| FP16/BF16 | 989 TFLOPS | ~2.2 PFLOPS | 1x |
| **FP8** | **1979 TFLOPS** | ~4.5 PFLOPS | **2x** |
| **FP4** | —（SM90 不支持） | ~9 PFLOPS | **4x**（SM100 独有） |

#### MmaKind 与 BLOCK_K 的耦合

读 `common/types.cuh:7-18`，DeepGEMM 用 `MmaKind` 区分指令族：

```cpp
enum class MmaKind {
    BF16        = 0,   // element_size = 2 字节
    MXFP8FP4    = 1,   // element_size = 1 字节（FP4 也按 1 字节算，pack 在 kernel 内处理）
};
```

`get_element_size` 直接决定 `BLOCK_K`（`csrc/jit_kernels/heuristics/sm90.hpp:60`）：

```cpp
const int block_k = 128 / get_element_size(desc.get_mma_kind());   // BF16→64, FP8/FP4→128
```

> 💡 **设计要点**：BF16 的 `BLOCK_K=64`（因为 WGMMA BF16 的 K=16，64/16=4 次累加）；FP8/FP4 的 `BLOCK_K=128`（WGMMA FP8 的 K=32，128/32=4 次累加）。两者都让 BLOCK_K 恰好是 4 次 MMA，这是 pipeline 设计的 sweet spot。

> 💡 **一句话总结**：DeepGEMM 在 SM90 主打 FP8（2x FP16），在 SM100 引入 FP4（4x FP16）用于 Mega MoE 的权重。FP4 的精度更脆弱，靠 UE8M0 共享指数 scaling 与 MX（Microscaling）格式补救。

### 学习任务 2：Per-128-Channel Scaling 的数学形式（45 分钟）

#### 从 per-tensor 到 per-128-channel

DeepGEMM 当前版本（v2.6+）的 FP8 GEMM 使用 **per-128-channel scaling**：K 维度按 128 长度分块，每块独立 scale。SM90 源码硬编码 `BLOCK_K == 128`（见 `sm90_fp8_gemm_1d1d.cuh:52`）：

```cpp
DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
```

数学形式：

$$D_{m,n} = \sum_{b=0}^{K/128-1} \left( \mathrm{FP8}(A_{m,\, b\cdot128:(b+1)\cdot128}) \cdot \mathrm{FP8}(B_{b\cdot128:(b+1)\cdot128,\, n}) \cdot s_a^{(m,b)} \cdot s_b^{(n,b)} \right)$$

- $s_a \in \mathbb{R}^{M \times K/128}$：A 的 scale，每行每 128 列一个
- $s_b \in \mathbb{R}^{N \times K/128}$：B 的 scale，每列每 128 行一个
- WGMMA 累加在 FP32，**每 128 步乘一次** $s_a^{(m,b)} \cdot s_b^{(n,b)}$

> ⚠️ **关键点**：scale 不是每 tensor 一个，也不是每 128×128 块一个，而是**沿 K 维每 128 通道一个**——这正好对齐 WGMMA 的 K=32（FP8）tile，4 次 WGMMA 累加后乘一次 scale，scale 乘法可融入 epilogue 不额外占 cycle。
>
> 数学合理性：$(sA_k \cdot sB_k) \cdot (A_k @ B_k)$ 对 k 求和等于 $(A \cdot sA) @ (B \cdot sB)^T$，即 per-channel scaling 等价于先量化再 GEMM——但分离 scale 让 WGMMA 用未 scale 的 FP8 输入，精度更好。

#### SMEM 中 scale 的占用

读 `sm90_fp8_gemm_1d1d.cuh:73-76`，每 pipeline stage 的 scale smem 占用极小：

```cpp
static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);        // 如 BLOCK_M=128 → 512B
static constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = BLOCK_N * sizeof(float);        // 如 BLOCK_N=128 → 512B
static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFB_SIZE_PER_STAGE, 128u);
DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 128 == 0, "Invalid TMA alignment");      // → BLOCK_M 必须是 32 的倍数
```

> 💡 **对比数据 smem**：FP8 数据每 stage 占 `BLOCK_M*128 + BLOCK_N*128` 字节（如 128×128 → 32KB），而 scale 只占 `BLOCK_M + BLOCK_N` 字节（256B）——**scale 的 smem 开销不到 1%**，所以可以放心地给 scale 也开 pipeline stage。

#### KernelType：1D1D vs 1D2D vs NoSF

读 `common/types.cuh:46-50` 与 `tests/generators.py:17-29`，DeepGEMM 把 scaling 分为三种 kernel 类型：

```cpp
enum class KernelType {
    Kernel1D1D = 0,   // A/B 的 scale 都是 1D（沿 K/128，即 per-token A + per-channel B）
    Kernel1D2D = 1,   // A 的 scale 1D，B 的 scale 2D（per-128×128-block）
    KernelNoSF = 2    // 无 scale（BF16 GEMM）
};
```

| KernelType | A 的 scale | B 的 scale | SM90 输出 dtype | 累加 | 典型用途 |
|------------|-----------|-----------|-----------------|------|----------|
| `Kernel1D1D` | per-token (1D) | per-channel (1D) | **FP32** | ✓ | 权重梯度 wgrad（B 是激活） |
| `Kernel1D2D` | per-token (1D) | per-block (2D) | **BF16** | ✗ | 前向 GEMM（B 是权重） |
| `KernelNoSF` | 无 | 无 | BF16/FP32 | 可选 | BF16 GEMM |

测试代码（`generators.py:94-98`）确认了这个分配：

```python
def get_kernel_types(dtype: torch.dtype) -> tuple:
    if dtype == torch.bfloat16:
        return (KernelType.KernelNoSF, )
    return (KernelType.Kernel1D2D, ) if get_arch_major() == 9 else (KernelType.Kernel1D1D, )
```

> ⚠️ **注意**：SM100 上**只有 1D1D** 一种 FP8/FP4 kernel（`sm100_fp8_fp4_gemm_1d1d.cuh`），靠 UE8M0 + 硬件 block_scale 解决精度问题，不再需要 1D2D。1D2D 是 SM90 的"折衷方案"——前向 B 用 per-block scale 是因为权重静态、per-block 够用，省 scale 存储。

### 学习任务 3：SM90 FP32 scale 的软件 promote 全流程（60 分钟）

这是 Day 2 的**核心精读**内容。读 `sm90_fp8_gemm_1d1d.cuh:229-320`，理解 scale 如何从 gmem 流到 accumulator。

#### Step 1：TMA 加载 scale 到 smem（TMA warpgroup）

`sm90_fp8_gemm_1d1d.cuh:229-230`，scale 和数据一起被 TMA 搬进 smem：

```cpp
// 每个 k-block（128 K）发射一次
tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, &full_barrier, smem_sfa[stage_idx], m_idx, sf_k_idx, num_tma_multicast_a);
tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, &full_barrier, smem_sfb[stage_idx], n_idx, sf_k_idx, num_tma_multicast_b);
```

- `tensor_map_sfa` / `tensor_map_sfb` 是 `__grid_constant__` TMA descriptor（kernel 参数，见 `:46-47`）
- 每个 stage 加载 `BLOCK_M` 个 float（SFA）+ `BLOCK_N` 个 float（SFB）
- `full_barrier.arrive_and_expect_tx(...)`（`:233`）把 scale 字节数也算进预期到达量

#### Step 2：Math warpgroup 读 scale 到寄存器

`sm90_fp8_gemm_1d1d.cuh:281-289`，在每个 k-block 的 MMA 之前，math warpgroup 从 smem 读 scale：

```cpp
// 读 A scales：每个 lane 读自己负责的两行（r_0 和 r_0+8）
auto scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);   // r_0 = warp_idx*16 + lane_idx/4
auto scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);   // r_1 = r_0 + 8

// 读 B scales：每个 lane 读自己负责的两列（float2 = 两个相邻 N-col）
#pragma unroll
for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
    scales_b[i] = ptx::ld_shared(reinterpret_cast<float2*>(smem_sfb[stage_idx] + i * 8 + col_idx * 2));
```

> 💡 **lane 到 scale 的映射**：WGMMA 的 accumulator `64×BLOCK_N` 分布在 128 个 lane 上，每个 lane 持有 `kNumAccum = 64*BLOCK_N/128` 个 FP32。这些 accumulator 按 2×2 子块组织——`scale_a_0/scale_a_1` 对应 2 行，`scales_b[i].x/.y` 对应 2 列，正好形成 4 个 accumulator 元素的外积。
>
> 关键约束（代码注释 `:282`）："all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results"——读 scale 必须在 WGMMA arrive 之前，否则下个 stage 的 TMA 可能覆盖 smem。

#### Step 3：WGMMA 计算（不带 scale）

`sm90_fp8_gemm_1d1d.cuh:295-306`，MMA 累加在 FP32 寄存器，**不乘 scale**：

```cpp
ptx::warpgroup_arrive();
#pragma unroll
for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {       // 128/32 = 4 次
    auto desc_a = mma::sm90::make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
    auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
    WGMMA::wgmma(desc_a, desc_b, accum, k);                // k==0 时 ScaleOut::Zero，否则 One
}
ptx::warpgroup_commit_batch();
ptx::warpgroup_wait<0>();                                   // 等 WGMMA 完成
```

- `ScaleOut::Zero`（k==0）覆盖 accum；`ScaleOut::One`（k>0）累加——见 `mma/sm90.cuh:22` 的 `scale_d` 参数
- 4 次 WGMMA 累加得到一个 k-block（128 K）的 partial accum

#### Step 4：Promote with scales（核心！）

`sm90_fp8_gemm_1d1d.cuh:312-320`，WGMMA 完成后立即乘 scale 累加到 `final_accum`：

```cpp
// Promote with scales
#pragma unroll
for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
    const float &scale_b_0 = scales_b[i].x;
    const float &scale_b_1 = scales_b[i].y;
    final_accum[i * 4 + 0] += scale_a_0 * scale_b_0 * accum[i * 4 + 0];
    final_accum[i * 4 + 1] += scale_a_0 * scale_b_1 * accum[i * 4 + 1];
    final_accum[i * 4 + 2] += scale_a_1 * scale_b_0 * accum[i * 4 + 2];
    final_accum[i * 4 + 3] += scale_a_1 * scale_b_1 * accum[i * 4 + 3];
}
```

> 💡 **数据流总结**：
> - `accum[...]`：单个 k-block 的 WGMMA 输出（FP32 寄存器，每 k-block 覆盖）
> - `final_accum[...]`：跨所有 k-block 的累加和（FP32 寄存器，最终输出）
> - 每 k-block 做 4 次标量乘加：`final_accum += (sA * sB) * accum`
> - 这是 4 次标量乘加，**可被编译器融合成 FFMA**，开销可忽略
>
> ⚠️ **为什么不在 WGMMA 之前乘 scale？** 因为 WGMMA 指令不接受 FP8 输入 + FP32 scale 的混合形式——WGMMA 是 `FP8 × FP8 → FP32`，scale 只能在输出后乘。SM100 的 `tcgen05.mma.block_scale` 才原生支持硬件 block scale（见学习任务 5）。

#### 1D2D 的差异：B 的 scale 走 normal global load

读 `sm90_fp8_gemm_1d2d.cuh`，1D2D 与 1D1D 的关键差异在 B 的 scale 加载方式：

| 维度 | 1D1D (`sm90_fp8_gemm_1d1d`) | 1D2D (`sm90_fp8_gemm_1d2d`) |
|------|------------------------------|------------------------------|
| SFA 加载 | TMA，per-stage | TMA，per-stage（同） |
| SFB 加载 | **TMA**，per-stage | **Math warps 从 gmem 直接 ld**（`:241-250`），单 buffer 全 k-block 共享 |
| SFB 布局 | 1D：`[N, K/128]`，每 N-col 一个 | 2D：`[N/128, K/128]`，每 128×128 块一个 |
| SFB smem | per-stage `BLOCK_N*4` 字节 | 单 buffer `shape_k_scales * (1 or 2) * 4` 字节 |
| 输出 | FP32（累加） | BF16（不累加） |
| B 的 scale 粒度 | per-channel（细） | per-block（粗，省存储） |

1D2D 的 SFB 读取（`:288-291`）：

```cpp
float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx), scale_b_1;
if constexpr (not kMustUseUniformedScaleB)                // BLOCK_N 跨两个 128-N 块时才读第二个
    scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);
```

`kMustUseUniformedScaleB`（`BLOCK_K % BLOCK_N == 0`）：当 BLOCK_N ≤ 128 时一个 scale 覆盖整个 BLOCK_N，只需读一次；BLOCK_N > 128 时可能跨两个 128-N 块，需要两个 scale。

> 💡 **为什么 1D2D 让 math warps 读 SFB？** 1D2D 的 SFB 是 per-block（每 128-N 一个），单个 BLOCK_N 只需 1-2 个 scale 值，TMA 启动开销不划算——直接让 math warps 用 `ld_shared` 几条指令就搞定，还能省 per-stage smem。

### 学习任务 4：SM100 UE8M0 与硬件 Block Scale（60 分钟）

#### SM90 FP32 scale vs SM100 UE8M0

读 README "Notices" 与 `csrc/apis/layout.hpp`，SM90 与 SM100 的 scaling factor 格式完全不同：

| 架构 | scale 格式 | 大小 | 含义 | 谁乘 scale |
|------|-----------|------|------|-----------|
| **SM90** | FP32 | 4 字节/scale | 完整浮点 scale | **软件**（promote loop） |
| **SM100** | **UE8M0 packed** | 1 字节/scale（4 个 pack 成 1 个 `torch.int`） | 仅 8 位指数，尾数隐含 1.0 | **硬件**（`tcgen05.mma.block_scale`） |

#### UE8M0 是什么

UE8M0（Unsigned Exponent 8-bit, Mantissa 0-bit）是 Blackwell 的 MX（Microscaling）格式——scale 只存 8 位无符号指数，尾数隐含为 1.0，所以 scale 值只能是 $2^e$（e 是 8 位整数）。4 个 UE8M0 pack 成一个 `torch.int`（32 位）。

读 `deep_gemm/utils/math.py:13-23` 看 UE8M0 的纯 Python 实现：

```python
def ceil_to_ue8m0(x: torch.Tensor):
    bits = x.abs().float().view(torch.int)                    # 重解释为 int32
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()  # 取指数位，尾数非零则进位
    return (exp.clamp(1, 254) << 23).view(torch.float)        # 重组回 FP32（尾数为 0）

def pack_ue8m0_to_int(x: torch.Tensor):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    x_int = x.view(torch.int)
    assert (x_int >= 0).all() and (x_int & 0x7fffff == 0).all()   # 必须是 UE8M0（尾数 0、非负）
    return (x_int >> 23).to(torch.uint8).view(torch.int)          # 取高 8 位，4 个字节 pack 成 1 个 int
```

- `ceil_to_ue8m0`：把任意 FP32 scale 向上取整到最近的 $2^e$（尾数非零就指数+1）
- `pack_ue8m0_to_int`：4 个 UE8M0 的指数（各 8 位）pack 成一个 int32
- 约束：scale 必须是非负、尾数为 0（即 $2^e$ 形式）

#### SMEM 里的 scale 是 uint32_t

读 `sm100_fp8_fp4_gemm_1d1d.cuh:88-89`，SM100 的 scale smem 类型与 SM90 完全不同：

```cpp
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);   // 注意是 uint32_t 不是 float！
constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
```

一个 `uint32_t` 装 4 个 UE8M0 指数，所以一个 packed word 覆盖 4 个 k-group 的 scale。

#### gran_k：scale 的 K 粒度（32 或 128）

SM100 引入了 SM90 没有的灵活性——scale 的 K 粒度可以是 32 或 128（`sm100_fp8_fp4_gemm_1d1d.cuh:67-68`）：

```cpp
DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
```

| gran_k | 含义 | 1 packed word 覆盖 | SF TMA 频率 |
|--------|------|-------------------|------------|
| 128 | MXFP8-128（同 SM90 per-128-channel） | 4×128=512 K = 4 个 BLOCK_K | 每 4 个 k-block 发一次 |
| 32 | MXFP8-32（更细粒度） | 4×32=128 K = 1 个 BLOCK_K | 每个 k-block 发一次 |

代码（`:65-66`）：

```cpp
constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
```

`gran_k=32` 时 SF TMA 频率 4 倍，但精度更高——适用于 FP4（精度脆弱，需要更细的 scale）。

#### 硬件 block_scale 指令

读 `ptx/tcgen05.cuh:40-58`，SM100 的 MMA 指令**原生支持 block scale**：

```cpp
struct SM100_MMA_MXF8F6F4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a, uint64_t const& desc_b,
        uint32_t const& tmem_c, uint32_t const& scale_c,
        uint64_t const& desc, uint32_t const& tmem_sfa, uint32_t const& tmem_sfb) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};
```

关键点：
- `tmem_c`：accumulator 在 **TMEM（Tensor Memory）** 里，不在寄存器——这是 SM100 与 SM90 的根本区别
- `tmem_sfa` / `tmem_sfb`：scale 也在 TMEM 里，MMA 硬件直接读
- `scale_c`：`p = (scale_c != 0)`，决定是累加（`p=true`）还是覆盖（`p=false`）——等价于 SM90 的 `ScaleOut`
- `kind::mxf8f6f4`：支持 FP8/FP6/FP4 全系 MX 格式
- `block_scale` 后缀：启用硬件 block scale，scale 是 UE8M0 packed

#### Scale 从 SMEM 到 TMEM：UTCCP

Scale 必须先从 SMEM 搬到 TMEM 才能被 MMA 用。读 `sm100_fp8_fp4_gemm_1d1d.cuh:350-369`：

```cpp
using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
    cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
if (sfa_stage_in_group_idx == 0) {                           // 每 kNumSFAStagesPerLoad 个 k-block 搬一次
    #pragma unroll
    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
        auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);   // 搬到 TMEM 的指定列
    }
}
```

- UTCCP = Tensor Memory Copy Collective，专用 SMEM→TMEM 搬运指令
- `kNumSFAStagesPerLoad` 决定搬一次 TMEM 能服务几个 k-block（gran_k=128 时 4 个，gran_k=32 时 1 个）

#### sfa_id / sfb_id：从 packed word 选哪个 exponent

读 `sm100_fp8_fp4_gemm_1d1d.cuh:376-377`：

```cpp
const uint32_t sfa_id = (kGranKA == 32 ? kUMMAKIdx : sfa_stage_in_group_idx);
const uint32_t sfb_id = (kGranKB == 32 ? kUMMAKIdx : sfb_stage_in_group_idx);
```

- gran_k=32：`sfa_id = kUMMAKIdx`（0..3，每个 UMMA_K=32 步用一个 exponent）
- gran_k=128：`sfa_id = sfa_stage_in_group_idx`（0..3，每个 k-block 用一个 exponent）

`sfa_id` 被嵌入指令描述符（`mma/sm100.cuh:151-155`）：

```cpp
CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}
```

> 💡 **SM90 vs SM100 scale 处理对比**：
>
> | 维度 | SM90 | SM100 |
> |------|------|-------|
> | Scale 格式 | FP32（4B/scale） | UE8M0 packed（1B/scale，4 个 pack 成 int32） |
> | Scale 存储 | SMEM | SMEM → **TMEM**（UTCCP 搬运） |
> | Scale 乘法 | **软件** promote loop（4 次 FFMA） | **硬件** `tcgen05.mma.block_scale`（零开销） |
> | Accumulator | FP32 寄存器 | TMEM |
> | gran_k | 固定 128 | 32 或 128 |
> | FP4 支持 | ✗ | ✓（`kind::mxf8f6f4` 原生支持） |

### 学习任务 5：Recipe 系统与 Scale 布局转换（45 分钟）

#### Recipe 参数

DeepGEMM 用 "recipe" 描述 scale 的粒度。读 `csrc/apis/layout.hpp:14-90`，有两种形式：

| 参数 | 形式 | 含义 |
|------|------|------|
| `recipe` | `(gran_mn_a, gran_mn_b, gran_k)` | A、B 共用同一个 gran_k |
| `recipe_a` + `recipe_b` | `(gran_mn, gran_k)` × 2 | A、B 独立 gran_k |

- `gran_mn`：MN 维粒度。`1` = per-token/per-channel（每行/列一个）；`128` = per-block（每 128 行/列一个）
- `gran_k`：K 维粒度。`32` 或 `128`

**默认 recipe**（`csrc/utils/layout.hpp:74-87`）：

```cpp
static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        return {1, 128, 128};                    // SM90: per-token A, per-block B, gran_k=128
    } else if (arch_major == 10) {
        return sfb_dtype == torch::kFloat ?
            std::make_tuple(1, 128, 128):        // SM100 legacy: per-token A, per-block B
            std::make_tuple(1,   1, 128);        // SM100 1D1D: per-token A, per-channel B
    }
}
```

> 💡 **为什么 SM100 默认 per-channel B？** 因为 SM100 有硬件 block_scale，per-channel 不再有性能损失（SM90 的 1D2D 就是为绕开 per-channel 的软件开销而设计）。SM100 统一用 1D1D，代码更简单、精度更高。

#### 测试中的 QuantConfig

读 `tests/generators.py:43-79`，测试用 `QuantConfig` 封装 quant 粒度：

```python
class QuantConfig:
    _legacy_quant_config = (128, 128, False, False)       # SM90 legacy: gran_k_a=128, gran_k_b=128, FP8×FP8

    def __init__(self, value: Tuple[int, int, bool, bool] = _legacy_quant_config):
        self.gran_k_a, self.gran_k_b, self.is_fp4_a, self.is_fp4_b = value
```

`get_recipes` 方法（`generators.py:56-63`）把 `QuantConfig` 翻译成 API 接受的 recipe：

```python
def get_recipes(self, is_wgrad: bool = False) -> Tuple[Tuple, Tuple, Tuple]:
    recipe, recipe_a, recipe_b = None, None, None
    if self.is_legacy():
        recipe = (1, 1, 128) if is_wgrad else None        # wgrad: per-channel B → 1D1D；fwd: 默认 → 1D2D
    else:
        recipe_a = (1, self.gran_k_a)                      # A: per-token
        recipe_b = (1, self.gran_k_b) if self.is_fp4_b or is_wgrad else (self.gran_k_b, self.gran_k_b)
    return recipe, recipe_a, recipe_b
```

容差（`generators.py:65-70`）：

```python
def max_diff(self) -> float:
    if self.is_fp4_a and self.is_fp4_b: return 0.02        # FP4×FP4
    if self.is_fp4_a or self.is_fp4_b: return 0.01         # FP8×FP4 混合
    return 0.001                                            # FP8×FP8
```

#### Scale 布局转换的 4 条路径

读 `csrc/apis/layout.hpp:40-60`，`transform_sf_into_required_layout` 根据 (dtype, gran_mn, gran_k, arch) 分 4 条路径：

| # | 输入 | 架构 | gran_mn, gran_k | 动作 | 用的 kernel |
|---|------|------|-----------------|------|------------|
| 1 | FP32 | SM90（或 SM100+`disable_ue8m0_cast`） | 1, 128 | 转置成 MN-major + TMA 对齐 | SM90 SFA、SFB（1D1D） |
| 2 | FP32 | SM90（或 SM100+`disable_ue8m0_cast`） | 128, 128 | 仅校验 2D 布局（不转置） | SM90 SFB（1D2D） |
| 3 | FP32 | SM100 | 任意, 32/128 | **广播 + pack 成 UE8M0 + TMA 对齐** | SM100 SFA/SFB |
| 4 | INT（已 pack） | SM100 | 1, 32/128 | 仅校验布局（用户已 pack） | SM100 SFA/SFB |

路径 3 的核心 kernel 在 `impls/smxx_layout.cuh:54-144`（`transpose_and_pack_fp32_into_ue8m0`），packing 逻辑（`:135-139`）：

```cpp
uint32_t packed = 0;
packed |= (values[0] >> 23u);   // FP32 #0 的指数 → bits 0-7
packed |= (values[1] >> 15u);   // FP32 #1 的指数 → bits 8-15
packed |= (values[2] >>  7u);   // FP32 #2 的指数 → bits 16-23
packed |= (values[3] <<  1u);   // FP32 #3 的指数 → bits 24-31
```

并断言尾数和符号为 0（`:131`）：`DG_DEVICE_ASSERT((values[j] & 0x807fffffu) == 0)`——即输入必须是 UE8M0 形式（`ceil_to_ue8m0` 处理过）。

#### 用户侧 quant 工具

读 `deep_gemm/utils/math.py`，DeepGEMM 提供完整的 quant 工具链：

| 函数 | 用途 | 关键点 |
|------|------|--------|
| `per_token_cast_to_fp8(x, use_ue8m0, gran_k=128)` | per-token FP8 quant | amax 沿最后一维降采样到 `K/gran_k`，scale = amax/448.0 |
| `per_channel_cast_to_fp8(x, use_ue8m0, gran_k=128)` | per-channel FP8 quant | amax 沿 dim 0 降采样，要求 `x.size(0) % gran_k == 0` |
| `per_block_cast_to_fp8(x, use_ue8m0, gran_k=128)` | per-block FP8 quant | amax 按 `(gran_k, gran_k)` 块降采样，pad 到 gran_k 倍数 |
| `per_token_cast_to_fp4(x, use_ue8m0, gran_k=128)` | per-token FP4 quant | scale = amax/6.0，pack 两个 nibble 成 int8 |
| `ceil_to_ue8m0(x)` | 向上取整到 $2^e$ | 取 FP32 指数位，尾数非零则进位 |
| `pack_ue8m0_to_int(x)` | 4 个 UE8M0 pack 成 int32 | 取高 8 位，4 字节 → 1 int |
| `transpose_packed_fp4(a)` | 转置 packed FP4 | 解包 nibble → 转置 → 重 pack |

> ⚠️ **注意**：输入转置、FP8 cast、scale 布局转换等**不在 GEMM kernel 内做**——DeepGEMM 专注 GEMM 本身，这些预处理由用户自行融合到前一层的 epilogue。README 提供了简单的 PyTorch 工具函数但性能不是最优。

### 学习任务 6：dispatch 全流程（30 分钟）

读 `csrc/apis/gemm.hpp:73-124`，把前几个学习任务串起来——从 Python 调用到 kernel 选择的完整 dispatch：

```cpp
static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a, ...)
{
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {                            // SM90 强制 K-major
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    const auto arch_major = device_runtime->get_arch_major();
    const auto [m, k]  = check_ab_fp8_fp4(a.first, major_a, arch_major);   // FP4 会自动 ×2
    const auto [n, k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);

    // 转 scale 布局，返回 (sfa, sfb, gran_k_a, gran_k_b)
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(...);

    // 根据 arch + sfa dtype dispatch
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
        if (gran_n == 1)
            sm90_fp8_gemm_1d1d(...);                         // SM90 + per-channel B → 1D1D（wgrad）
        else
            sm90_fp8_gemm_1d2d(...);                         // SM90 + per-block B → 1D2D（fwd）
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_fp4_gemm_1d1d(..., gran_k_a, gran_k_b, ...);   // SM100（统一 1D1D + UE8M0）
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}
```

> 💡 **dispatch 决策树**：
> 1. **架构**（SM90 vs SM100）：决定 scale 格式（FP32 vs UE8M0）和是否强制 K-major
> 2. **scale dtype**（FP32 vs INT）：决定走 SM90 还是 SM100 kernel
> 3. **gran_n**（B 的 MN 粒度）：仅 SM90 用，决定 1D1D（per-channel）还是 1D2D（per-block）
> 4. **gran_k**（仅 SM100）：作为模板参数 `kGranKA`/`kGranKB` 编进 kernel
> 5. **a/b dtype**（FP8 vs FP4）：仅 SM100 用，作为模板参数 `a_dtype_t`/`b_dtype_t` 编进 kernel

`fp8_requires_k_major()`（`csrc/utils/layout.hpp:32-34`）：

```cpp
static bool fp8_requires_k_major() {
    return device_runtime->get_arch_major() == 9;            // SM90 要求 K-major，SM100 无限制
}
```

原因是 SM90 的 WGMMA FP8 指令是 `MMA_64x{N}x32_F32E4M3E4M3_SS_TN`（`_TN` = transpose-N），只支持 K-major 操作数；SM100 的 UMMA 支持任意 major 组合。

### 面试题积累（本周目标 10-12 道，今日 4 道）

**Q4：DeepGEMM 的 per-128-channel scaling 数学形式是什么？为什么选 128？**
> 答：$D_{m,n} = \sum_{b=0}^{K/128-1} (A_{m,b\cdot128:(b+1)\cdot128} \cdot B_{b\cdot128:(b+1)\cdot128,n}) \cdot s_a^{(m,b)} \cdot s_b^{(n,b)}$。选 128 是因为它对齐 WGMMA FP8 的 K=32 tile（128/32=4 次 WGMMA 累加后乘一次 scale），scale 乘法可融入 epilogue 不额外占 cycle。数学上 $(sA_k \cdot sB_k) \cdot (A_k @ B_k)$ 对 k 求和等价于先量化再 GEMM，但分离 scale 让 WGMMA 用未 scale 的 FP8 输入，精度更好。

**Q5：SM90 的 1D1D 和 1D2D kernel 有什么区别？什么时候用哪个？**
> 答：① SFB 加载方式：1D1D 用 TMA per-stage 加载，1D2D 让 math warps 从 gmem 直接 ld；② SFB 粒度：1D1D 是 per-channel（每 N-col 一个），1D2D 是 per-block（每 128×128 块一个）；③ 输出 dtype：1D1D 是 FP32+累加（wgrad），1D2D 是 BF16 不累加（fwd）；④ 1D1D 的 B 是激活（per-token scale），1D2D 的 B 是权重（per-block scale 够用，省存储）。SM100 统一用 1D1D，靠硬件 block_scale 消除 per-channel 的软件开销。

**Q6：UE8M0 是什么？为什么 SM100 用它代替 FP32 scale？**
> 答：UE8M0 是 Unsigned Exponent 8-bit Mantissa 0-bit，scale 只存 8 位指数（值只能是 $2^e$），4 个 pack 成一个 int32。用它的原因：① 压缩 4 倍（4B→1B），K 大时 scale 数组不再是带宽瓶颈；② SM100 的 `tcgen05.mma.kind::mxf8f6f4.block_scale` 指令原生支持 UE8M0，硬件零开销乘 scale（SM90 需要软件 promote loop）；③ 是 FP4 能 work 的前提——FP4 精度太低必须配细粒度 scale，而细粒度 scale 又不能太大，UE8M0 是两者的折衷。`ceil_to_ue8m0` 把任意 FP32 向上取整到最近的 $2^e$（尾数非零就指数+1）。

**Q7：SM90 和 SM100 的 scale 处理有什么本质区别？**
> 答：六个维度——① 格式：FP32 vs UE8M0 packed；② 存储：SMEM vs SMEM→TMEM（UTCCP 搬运）；③ 乘法：软件 promote loop（4 次 FFMA）vs 硬件 `tcgen05.mma.block_scale`（零开销）；④ accumulator：FP32 寄存器 vs TMEM；⑤ gran_k：固定 128 vs 32/128 可选；⑥ FP4：不支持 vs 原生支持。根本原因是 SM100 有专用的 TMEM 和 block_scale 硬件路径，scale 不再占用 CUDA core 周期。

### 今日检查清单

- [ ] 能说出 e4m3 / e5m2 / FP4 的最大值与典型用途，知道 FP4 只有 8 个非零值
- [ ] 能写出 per-128-channel scaling 的数学形式，解释为什么分离 scale 不影响数学等价性
- [ ] 理解为什么 SM90 硬编码 `BLOCK_K == 128`（对齐 WGMMA K=32，4 次累加）
- [ ] 能画出 SM90 的 scale 数据流：TMA → smem → ld_shared 寄存器 → promote with scales
- [ ] 能说出 1D1D vs 1D2D 的 5 个差异（SFB 加载方式、输出 dtype、累加、粒度、用途）
- [ ] 能解释 UE8M0 是什么，为什么 4 个 pack 成 int32，`ceil_to_ue8m0` 怎么工作
- [ ] 能说出 SM90 vs SM100 scale 处理的 6 个差异（格式/存储/乘法/accumulator/gran_k/FP4 支持）
- [ ] 理解 `gran_k` 32 vs 128 的意义，`sfa_id`/`sfb_id` 如何从 packed word 选 exponent
- [ ] 能说出 recipe 的两种形式（`recipe` vs `recipe_a`+`recipe_b`），默认 recipe 在 SM90/SM100 各是什么
- [ ] 能画出 `fp8_fp4_gemm_nt` 的 dispatch 决策树
- [ ] 读完 `sm90_fp8_gemm_1d1d.cuh:281-320`、`sm100_fp8_fp4_gemm_1d1d.cuh:350-392`、`ptx/tcgen05.cuh:40-58`

#### 明日预告

Day 3 将精读 `sm90_fp8_gemm_1d1d.cuh` 的数据搬运与计算流水——TMA descriptor 构造、WGMMA async 发射、mbarrier 双 barrier 同步、持久化调度。今天理解了 scale 的数据流，明天要把它放进完整的 pipeline 时序图里。建议今晚先扫一眼 `ptx/wgmma.cuh`（只有 25 行）和 `ptx/tma.cuh`，熟悉 `wgmma.fence` / `commit_group` / `wait_group` 三件套。

---
