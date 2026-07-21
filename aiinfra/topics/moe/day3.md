# Day 3（周三）：Grouped GEMM 与 Token Dispatch

> **今日目标**：实现 MoE 的核心计算——把分派后的 token 喂给各专家做 FFN，掌握 Grouped GEMM 与 Scatter-Gather 两种实现路径
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"MoE 的专家计算怎么高效做"必问

---

### 学习任务 1：两种实现路径对比（30 分钟）

![MoE 专家计算两条路径：Scatter-Gather 逐专家 vs Grouped GEMM 批量](../images/moe_grouped_gemm_paths.svg)

#### 路径 A：Scatter-Gather + 逐专家 GEMM

```python
# 把 token 按 expert 分组，每个专家单独 GEMM
for e in range(num_experts):
    mask = (topk_idx == e).any(dim=-1)         # 哪些 token 去了专家 e
    tokens_e = x[mask]                          # gather
    h = F.gelu(tokens_e @ w1[e].T)              # 专家 e 的 FFN
    out[mask] += (h @ w2[e].T) * scores_e       # scatter 回去
```

- 优点：实现简单，每个 GEMM 是标准 cuBLAS 调用
- 缺点：`num_experts` 次 kernel launch；专家负载不均时小 GEMM 浪费 SM

#### 路径 B：Grouped GEMM（一次 launch）

```python
# 把所有专家的 GEMM 打包成一次调用
# 输入：[total_tokens, d_model]，total_tokens = T * K（每 token 去 K 个专家）
# 输出：[total_tokens, d_model]
out = grouped_gemm(x_dispatched, w1_all, num_groups=num_experts, group_sizes=...)
```

- 优点：1 次 kernel launch，SM 均衡分配
- 缺点：需要变长 group 支持（各专家 token 数不同）

| 维度 | Scatter-Gather | Grouped GEMM |
|------|----------------|--------------|
| kernel launch | $O(N)$ 次 | 1 次 |
| 负载均衡 | 差（小专家浪费 SM） | 好（SM 跨专家调度） |
| 实现复杂度 | 低 | 中-高 |
| 代表实现 | 早期 Megatron | vLLM `fused_moe`、CUTLASS `GemmGroup` |

> 💡 **一句话总结**：MoE 专家计算的工程核心就是把"逐专家 GEMM"换成"Grouped GEMM"——这是 vLLM 推理比朴素实现快数倍的关键。

### 学习任务 2：CUTLASS Grouped GEMM 回顾（30 分钟）

复习 [CUTLASS 专题 Day 7](../cutlass/day7.md) 的 Group GEMM：

```cpp
// CUTLASS 3.x 的 Group GEMM：一次 launch 处理多个不同尺寸的 GEMM
using GroupGemm = cutlass::gemm::device::GemmGroup<
    /* MmaType, AccType, Layout... */
>;

// host 端传入 problem_size 数组（每个专家的 token 数不同）
std::vector<cutlass::gemm::GemmCoord> problem_sizes = {
    {532, 4096, 4096},   // 专家 0 收到 532 个 token
    {498, 4096, 4096},   // 专家 1 收到 498 个 token
    {511, 4096, 4096},   // 专家 2 收到 511 个 token
    // ...
};
```

> ⚠️ **注意**：CUTLASS Group GEMM 适合"专家数少、每专家 token 多"的场景。DeepSeek-V3 的 256 专家 + 每 expert 平均 8 token 的情况下，CUTLASS Group GEMM 的 per-group 开销会摊薄收益——此时更适合用 Triton 的 `m_grouped` GEMM 或变长 tile 调度。

### 学习任务 3：Triton 实现 Grouped GEMM（60 分钟）

Triton 的 `tl.dot` + `pid` 重映射可以实现 Grouped GEMM。核心思路：把所有 expert 的 GEMM 拼成一个大 GEMM，用 group offset 数组定位每个 expert 的起点。

```python
# triton_grouped_gemm.py —— Triton 变长 Grouped GEMM
import triton
import triton.language as tl

@triton.jit
def grouped_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    group_offsets_ptr,        # [num_experts + 1]，前缀和，专家 i 的 token 范围
    TOTAL_TILES: tl.constexpr,
    D: tl.constexpr, K_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # 通过 pid 反查属于哪个 expert + expert 内的 tile id
    # （实际实现用 binary search 或预计算的 tile_to_expert 表）
    expert_id, tile_m = lookup_tile(group_offsets_ptr, pid, ...)

    # 加载该 expert 的权重
    w = tl.load(w_ptr + expert_id * K_DIM * D + ...)
    # 标准 GEMM tile
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for k in range(0, K_DIM, BLOCK_K):
        a = tl.load(...)
        b = tl.load(...)
        acc += tl.dot(a, b)
    tl.store(out_ptr + ..., acc)
```

完整的 Grouped GEMM 实现可参考 vLLM 的 [`fused_moe.py`](https://github.com/vllm-project/vllm) 中的 `fused_experts` 函数——它把 dispatch + GEMM + combine 融合成一组 Triton kernel，是本周 Day 5 的精读目标。

### 学习任务 4：Token Dispatch 的索引计算（30 分钟）

Grouped GEMM 要求输入是"按 expert 连续排列"的 token，因此需要一个 dispatch 步骤：

```python
# 给定 topk_idx: [T, K]，构造 dispatch 后的 token 顺序
# 例：T=4, K=2, num_experts=4
# topk_idx = [[0, 2], [1, 2], [0, 3], [2, 3]]
# 展开后 8 个 (token, expert) 对，按 expert 排序：
#   expert 0: token 0, token 2
#   expert 1: token 1
#   expert 2: token 0, token 1, token 3
#   expert 3: token 2, token 3

def dispatch_tokens(x, topk_idx, topk_val):
    T, K = topk_idx.shape
    # 展开成 [T*K, 1]
    flat_idx = topk_idx.reshape(-1)
    flat_val = topk_val.reshape(-1)
    # 每个 token 重复 K 次
    token_ids = torch.arange(T, device=x.device).repeat_interleave(K)
    # 按 expert 排序，得到 dispatch 顺序
    sort_order = flat_idx.argsort()
    dispatched_x = x[token_ids[sort_order]]            # [T*K, d_model]
    dispatched_w = flat_val[sort_order]
    # 计算 group_offsets（前缀和）
    counts = flat_idx.bincount(minlength=num_experts)
    group_offsets = torch.cumsum(counts, dim=0)        # [num_experts]
    return dispatched_x, dispatched_w, group_offsets
```

> 💡 **关键洞察**：dispatch 的本质是一次 `argsort` + `gather`——把 `[T, K]` 的稀疏路由展开成 `[T*K]` 的连续 token 序列。这步在训练里是 gather 操作，在 EP 推理里则是 all-to-all 通信（Day 4）。

### 今日检查清单

- [ ] 能说出 Scatter-Gather 与 Grouped GEMM 两条路径的优劣
- [ ] 能解释为什么 DeepSeek-V3（256 专家、每专家少 token）对 Grouped GEMM 不友好
- [ ] `triton_grouped_gemm.py` 跑通，结果与逐专家 cuBLAS 一致
- [ ] 能手写 `dispatch_tokens` 函数，正确构造 group_offsets
- [ ] 用 `nsys` 对比逐专家 GEMM vs Grouped GEMM 的 kernel launch 数

---

