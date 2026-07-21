# Day 2（周二）：FP8/FP4 数据类型与 Per-128-Channel Scaling

> **今日目标**：理解 FP8（e4m3/e5m2）与 FP4 的数值分布，掌握 DeepGEMM 的 per-128-channel scaling（SM90 FP32 scale / SM100 UE8M0 packed scale），搞清 KernelType 1D1D vs 1D2D 的区别
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，FP8 精度策略是 DeepGEMM 的立身之本

---

### 学习任务 1：FP8 与 FP4 格式（45 分钟）

#### e4m3 vs e5m2 vs FP4

| 格式 | 指数位 | 尾数位 | 最大值 | 用途 |
|------|--------|--------|--------|------|
| **e4m3** | 4 | 3 | ±448 | 前向权重/激活 |
| **e5m2** | 5 | 2 | ±57344 | 反向梯度（大动态范围） |
| **FP4 (E2M1)** | 2 | 1 | ±6.0 | 极致压缩（Blackwell Mega MoE 权重） |

#### Hopper vs Blackwell 算力

| 精度 | H100/H800 峰值 | B200 峰值 | 相对 FP16 |
|------|----------------|-----------|-----------|
| FP16/BF16 | 989 TFLOPS | ~2.2 PFLOPS | 1x |
| **FP8** | **1979 TFLOPS** | ~4.5 PFLOPS | **2x** |
| **FP4** | —（SM90 不支持） | ~9 PFLOPS | **4x**（SM100 独有） |

> 💡 **一句话总结**：DeepGEMM 在 SM90 主打 FP8（2x FP16），在 SM100 引入 FP4（4x FP16）用于 Mega MoE 的权重。FP4 的精度更脆弱，靠 UE8M0 共享指数 scaling 与 MX（Microscaling）格式补救。

### 学习任务 2：Per-128-Channel Scaling 的数学形式（45 分钟）

#### 从 per-tensor 到 per-128-channel

DeepGEMM 当前版本（v2.6+）的 FP8 GEMM 使用 **per-128-channel scaling**：K 维度按 128 长度分块，每块独立 scale。源码硬编码 `BLOCK_K == 128`（见 `sm90_fp8_gemm_1d1d.cuh:52`）：

```cpp
DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
```

数学形式：

$$D_{m,n} = \sum_{b=0}^{K/128} \left( \mathrm{FP8}(A_{m,\, b\cdot128:(b+1)\cdot128}) \cdot \mathrm{FP8}(B_{b\cdot128:(b+1)\cdot128,\, n}) \cdot s_a^{(b)} \cdot s_b^{(b)} \right)$$

- $s_a \in \mathbb{R}^{M \times K/128}$：A 的 scale，每 128 列一个（per-channel）
- $s_b \in \mathbb{R}^{N \times K/128}$：B 的 scale，每 128 行一个
- WGMMA 累加在 FP32，每 128 步乘一次 $s_a^{(b)} \cdot s_b^{(b)}$

> ⚠️ **关键点**：scale 不是每 tensor 一个，也不是每 128×128 块一个，而是**沿 K 维每 128 通道一个**——这正好对齐 WGMMA 的 K=32（FP8）tile，4 次 WGMMA 累加后乘一次 scale，scale 乘法可融入 epilogue 不额外占 cycle。

#### KernelType：1D1D vs 1D2D vs NoSF

读 `common/types.cuh`，DeepGEMM 把 scaling 分为三种 kernel 类型：

| KernelType | scale 布局 | 适用 | 特点 |
|------------|-----------|------|------|
| `Kernel1D1D` | A/B 的 scale 都是 1D（沿 K/128） | 标准 FP8 GEMM | 最快，scale 与数据同方向 |
| `Kernel1D2D` | A 的 scale 1D，B 的 scale 2D | 某些 MoE 权重 | 兼容非标准 scale 布局 |
| `KernelNoSF` | 无 scale | BF16 GEMM | 不做 scaling |

### 学习任务 3：SM90 FP32 scale vs SM100 UE8M0（45 分钟）

读 README 的 "Notices" 一节，SM90 与 SM100 的 scaling factor 格式不同：

| 架构 | scale 格式 | 大小 | 含义 |
|------|-----------|------|------|
| **SM90** | FP32 | 4 字节/scale | 直接存浮点 scale 值 |
| **SM100** | **UE8M0 packed** | 1 字节/scale（4 个 pack 成 1 个 `torch.int`） | 仅 8 位指数，无尾数（MX 格式） |

#### UE8M0 是什么

UE8M0（Unsigned Exponent 8-bit, Mantissa 0-bit）是 Blackwell 的 MX（Microscaling）格式——scale 只存 8 位指数，尾数隐含为 1.0。4 个 UE8M0 pack 成一个 `torch.int`（32 位）。

```python
# SM100 的 scale 准备（伪代码）
# FP32 scale -> UE8M0 packed
scales_fp32 = ...  # [M, K/128]
scales_ue8m0_packed = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(scales_fp32)
# 内部：取 log2，量化到 8 位指数，4 个 pack 成一个 int
```

> 💡 **关键洞察**：UE8M0 把 scale 从 4 字节压到 1 字节——当 K 很大时，scale 数组本身会成为带宽瓶颈。SM100 的 MX 格式是 FP4 能 work 的前提：FP4 精度太低，必须配合细粒度 scale，而细粒度 scale 又不能太大，UE8M0 是两者的折衷。

### 学习任务 4：Scale 的 TMA 布局要求（30 分钟）

README 明确："The LHS scaling factor is required to have a TMA-aligned and transposed layout."

DeepGEMM 把 scale 也用 TMA 搬运（见 `sm90_fp8_gemm_1d1d.cuh:229-230`），因此 scale tensor 必须满足 TMA 对齐要求。用户提供工具函数：

```python
# 把用户 scale 转成 TMA 友好布局
sfa_aligned = deep_gemm.get_mn_major_tma_aligned_tensor(sfa)  # SM90 FP32
sfb_aligned = deep_gemm.get_mn_major_tma_aligned_tensor(sfb)

# SM100 的 UE8M0 packed 版本
sfa_packed = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
```

> ⚠️ **注意**：输入转置、FP8 cast、scale 布局转换等**不在 GEMM kernel 内做**——DeepGEMM 专注 GEMM 本身，这些预处理由用户自行融合到前一层的 epilogue。README 提供了简单的 PyTorch 工具函数但性能不是最优。

### 今日检查清单

- [ ] 能说出 e4m3 / e5m2 / FP4 的最大值与典型用途
- [ ] 能写出 per-128-channel scaling 的数学形式
- [ ] 理解为什么 `BLOCK_K == 128` 是硬编码（对齐 WGMMA K=32）
- [ ] 能说出 SM90 FP32 scale 与 SM100 UE8M0 packed 的差异
- [ ] 读完 `common/types.cuh`，能解释 `GemmType` 枚举的 7 个值

---

