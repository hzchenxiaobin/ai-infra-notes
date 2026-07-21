# Day 2（周二）：Gating 与 Top-K 路由算子

> **今日目标**：用 Triton 实现融合的 Gating + Softmax + Top-K kernel，理解 Top-K 选择的高效实现
> **面试考察度**：⭐⭐⭐⭐ 实践级，"MoE 的门控 kernel 怎么写"是高频题

---

### 学习任务 1：Gating kernel 的数据流（30 分钟）

![MoE 门控数据流：Logits → Softmax → Top-K → Scatter](../images/moe_gating_dataflow.svg)

#### 朴素实现的低效

```python
# 朴素三步：3 次 kernel launch，2 次中间 tensor 落盘 HBM
logits = x @ gate_weight.T        # [T, N]    —— GEMM
scores = softmax(logits, dim=-1)  # [T, N]    —— reduction
topk_val, topk_idx = scores.topk(k, dim=-1)  # [T, k]  —— 逐行 topk
```

每步都要把 `[T, N]` 写回 HBM 再读出。当 `T=4096, N=8` 时，中间 tensor 虽小，但 kernel launch 开销与 HBM 往返在小 batch 下占比极高。

#### 融合策略

| 融合层级 | kernel 数 | 适用场景 |
|----------|-----------|----------|
| 三步分离 | 3 | 调试用 |
| GEMM + Softmax 融合 | 2 | Triton `tl.dot` + `tl.softmax` |
| 全融合（GEMM+Softmax+TopK） | 1 | 极致优化，Top-K 在寄存器内做 |

> 💡 **一句话总结**：MoE 门控的 GEMM 很小（`[T, d_model] × [d_model, N]`，N 通常 8-64），瓶颈不是算力而是 kernel launch + HBM 往返——所以融合的收益远大于优化 GEMM 本身。

### 学习任务 2：Triton 实现 Gating + Softmax 融合（60 分钟）

```python
# triton_gating.py —— Gating + Softmax 融合 kernel
# 运行: python3 triton_gating.py
import triton
import triton.language as tl
import torch

@triton.jit
def gating_softmax_kernel(
    x_ptr, w_ptr, out_ptr,
    T, D: tl.constexpr, N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)                      # 沿 T 维分块
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, N)

    mask_t = offs_t < T
    # 加载 x: [BLOCK_T, D]
    x = tl.load(x_ptr + offs_t[:, None] * D + offs_d[None, :], mask=mask_t[:, None], other=0.0)
    # 加载 w: [N, D]（完整加载到寄存器，N 通常很小）
    w = tl.load(w_ptr + offs_n[:, None] * D + offs_d[None, :])
    # GEMM: [BLOCK_T, D] @ [D, N] -> [BLOCK_T, N]
    logits = tl.dot(x, tl.trans(w))
    # 行内 softmax（数值稳定）
    m = tl.max(logits, axis=1, keep_dims=True)
    e = tl.exp(logits - m)
    s = e / tl.sum(e, axis=1, keep_dims=True)
    # 写回 [BLOCK_T, N]
    tl.store(out_ptr + offs_t[:, None] * N + offs_n[None, :],
             s, mask=mask_t[:, None])

def triton_gating(x, w):
    T, D = x.shape
    N = w.shape[0]
    out = torch.empty(T, N, device=x.device, dtype=torch.float32)
    BLOCK_T = 64
    gating_softmax_kernel[(triton.cdiv(T, BLOCK_T),)](
        x, w, out, T, D=D, N=N, BLOCK_T=BLOCK_T)
    return out
```

### 学习任务 3：Top-K 选择的高效实现（45 分钟）

Top-K 在 GPU 上的难点：`[T, N]` 每行选 K 个最大值。当 N 较小（8-64）时，**寄存器内排序**比 heap 快。

```python
# triton_topk.py —— 小 N 的寄存器内 Top-K
@triton.jit
def topk_kernel(scores_ptr, val_ptr, idx_ptr,
                T, N: tl.constexpr, K: tl.constexpr,
                BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = tl.arange(0, N)
    mask_t = offs_t < T

    scores = tl.load(scores_ptr + offs_t[:, None] * N + offs_n[None, :],
                     mask=mask_t[:, None], other=-float('inf'))
    # 当 N 小（<=64）时，逐轮选最大值：K 轮，每轮 argmax + 屏蔽
    for k in tl.static_range(K):
        m = tl.max(scores, axis=1)                          # [BLOCK_T]
        idx = tl.argmax(scores, axis=1)                     # [BLOCK_T]
        # 写回第 k 个结果
        tl.store(val_ptr + offs_t * K + k, m, mask=mask_t)
        tl.store(idx_ptr + offs_t * K + k, idx, mask=mask_t)
        # 屏蔽已选位置
        scores = tl.where(offs_n[None, :] == idx[:, None], -float('inf'), scores)
```

| N 大小 | 推荐 Top-K 策略 | 复杂度 |
|--------|-----------------|--------|
| N <= 32 | 寄存器内 K 轮 argmax | $O(KN)$ |
| 32 < N <= 256 | bitonic sort 后取前 K | $O(N \log^2 N)$ |
| N > 256 | radix select / libcudart `topK` | $O(N)$ |

> ⚠️ **注意**：DeepSeek-V3 的 N=256（细粒度专家），Top-K=8，用寄存器内 K 轮 argmax 仍可行（8×256=2048 次比较/行）。但 Mixtral 的 N=8 K=2 用此法极快。

### 学习任务 4：动手实践（30 分钟）

在 `kernels/` 下创建 `triton_gating.py`（合并上面两段），并写一个 benchmark：

```python
# benchmark：对比 朴素 PyTorch vs Triton 融合
T, D, N, K = 4096, 4096, 8, 2
x = torch.randn(T, D, device='cuda')
w = torch.randn(N, D, device='cuda')

# 朴素：3 次 launch
def naive(x, w):
    logits = x @ w.T
    scores = logits.softmax(dim=-1)
    return scores.topk(K, dim=-1)

# Triton 融合 + 寄存器 topk
def fused(x, w):
    scores = triton_gating(x, w)
    return triton_topk(scores, K)

# 用 torch.utils.benchmark.Timer 对比
```

### 今日检查清单

- [ ] 能解释为什么 Gating kernel 的瓶颈是 kernel launch 而非 GEMM
- [ ] `triton_gating.py` 编译运行通过，结果与 PyTorch 一致
- [ ] 能说出 N<=32 / N>256 时 Top-K 的不同策略
- [ ] benchmark 显示 Triton 融合版比朴素 PyTorch 快 3x+
- [ ] 记录了融合前后 kernel launch 数量（用 `nsys stats`）

---

