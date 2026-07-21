# Day 5（周五）：Grouped GEMM for MoE

> **今日目标**：掌握 DeepGEMM 的 Grouped GEMM 三种变体（contiguous / masked / k-grouped），理解它为什么只分组 M 轴，对比 [CUTLASS 专题 Day 7](../cutlass/day7.md) 与 [MoE 专题](../moe/README.md) 的路径
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"DeepGEMM 的 Grouped GEMM 怎么为 MoE 设计"是 DeepSeek 系面试高频题

---

### 学习任务 1：DeepGEMM Grouped GEMM 的设计取舍（45 分钟）

读 README 的 "Grouped GEMMs (contiguous layout)" 一节：

> "Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape."

| 维度 | CUTLASS GemmGroup | DeepGEMM M-grouped |
|------|-------------------|---------------------|
| 分组轴 | 任意（每 group 的 M/N/K 都可变） | **仅 M 轴**（N/K 固定） |
| 问题尺寸 | 每 group 任意 | M 变长，N/K 所有 group 相同 |
| 调度复杂度 | 高（每 group 独立 tile） | 低（N/K 固定，tile 统一） |
| 适用场景 | 通用不等大 GEMM | MoE（专家共享 shape） |

> 💡 **关键洞察**：MoE 的所有专家有相同的 `[d_model, d_ff]` 权重 shape，只有每专家收到的 token 数不同——这正是"M 轴变长、N/K 固定"。DeepGEMM 砍掉 CUTLASS 的通用性，换来更简单的调度与更高的 SM 利用率。

### 学习任务 2：三种 Grouped GEMM 变体（45 分钟）

#### M-grouped contiguous（前向 / prefill）

```python
# 所有 token 按 expert 连续排列，grouped_layout 是前缀和
# grouped_layout: [num_experts]，grouped_layout[i] = 前 i+1 个 expert 的 token 总数
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    a,               # (A_fp8, SFA): [total_tokens, K]
    b,               # (B_fp8, SFB): [num_experts, N, K]
    d,               # [total_tokens, N]
    grouped_layout,  # [num_experts]，前缀和
)
```

- `total_tokens = sum(tokens_per_expert)`
- 每 expert 的 token 数在 `grouped_layout` 里
- 每 expert 段必须对齐到 `get_mk_alignment_for_contiguous_layout()`

#### M-grouped masked（decode / CUDA graph）

```python
# decode 阶段：CUDA graph 开启时 CPU 不知道每 expert 收多少 token
# 用 mask tensor 标记有效部分
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    a,      # [max_tokens, K]
    b,      # [num_experts, N, K]
    d,      # [max_tokens, N]
    mask,   # [num_experts, max_tokens]，bool
)
```

- 配合 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的低延迟 EP kernel 使用
- mask 标记每个 token 去哪个 expert，kernel 只算有效部分

#### K-grouped contiguous（MoE 权重梯度）

```python
# MoE 反向：M/N 固定，K 轴按 group 切分（对应前向的多个 expert）
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
    a,              # [M, total_K]
    b,              # [num_experts, total_K, N]
    d,              # [M, N]
    grouped_layout, # [num_experts]，K 维前缀和
)
```

- 用于 MoE 的 weight gradient（反向传播）
- M/N 固定，K 轴变长

### 学习任务 3：Psum Layout（MoE 反向优化）（30 分钟）

`GemmType` 里有 `MGroupedContiguousWithPsumLayout` 和 `KGroupedContiguousWithPsumLayout`——这是 DeepGEMM 为 MoE 反向做的特殊布局：

```python
# use_psum_layout=True 时，grouped_layout 存的是前缀和而非每 group 长度
# 允许 group 间有 padding（对齐到 kKAlignment=128）
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    use_psum_layout=True,
    ensure_zero_padding=True,  # 断言 padding 区为零
)
```

> ⚠️ **注意**：psum layout 是 2025.05 的权重梯度 kernel（[#95](https://github.com/deepseek-ai/DeepGEMM/pull/95)）引入的，让 MoE 反向的 K 分组可以有 128 对齐的 padding 而不影响正确性。读 `scheduler/gemm.cuh` 的 `get_next_psum_k_group` 可看实现。

### 学习任务 4：动手实践（30 分钟）

```python
# benchmark/grouped_gemm_demo.py
import torch, deep_gemm
from deep_gemm.testing import bench_kineto

# 模拟 DeepSeek-V3 的 MoE FFN（256 专家）
num_experts = 256
tokens_per_expert = torch.randint(4, 32, (num_experts,))
total_tokens = tokens_per_expert.sum().item()
K, N = 7168, 4096

# 生成 FP8 数据 + scale
a_fp8 = torch.randn(total_tokens, K, device='cuda').to(torch.float8_e4m3fn)
b_fp8 = torch.randn(num_experts, N, K, device='cuda').to(torch.float8_e4m3fn)
# ... 准备 SFA/SFB（需 TMA 对齐）...
grouped_layout = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32).cuda()

d = torch.empty(total_tokens, N, device='cuda', dtype=torch.bfloat16)

# DeepGEMM grouped
t = bench_kineto(
    lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a_fp8, sfa), (b_fp8, sfb), d, grouped_layout),
    'gemm_'
)
print(f'Grouped: {t*1e6:.0f} us | {2*total_tokens*N*K/t/1e12:.0f} TFLOPS')
```

### 今日检查清单

- [ ] 能说出 DeepGEMM 只分 M 轴的设计原因（MoE 专家共享 shape）
- [ ] 能区分 contiguous / masked / k-grouped 三种变体的适用场景
- [ ] 理解 psum layout 为什么用于 MoE 反向
- [ ] `grouped_gemm_demo.py` 跑通，记录 TFLOPS
- [ ] 用 `DG_PRINT_CONFIGS=1` 观察 JIT 选中的 tile 配置

---

