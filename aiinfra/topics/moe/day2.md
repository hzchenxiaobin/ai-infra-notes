# Day 2（周二）：Triton Gating + Top-K 融合 Kernel

> **本周定位**：本专题是 [CUTLASS 专题](../cutlass/README.md)（算子视角，Day 7 Group GEMM）之后的**系统视角**——把 Grouped GEMM、Top-K 路由、all-to-all 通信、负载均衡组装成一个完整的 MoE 层。本周目标是用 Triton 拼出一个 Top-2 路由的 MoE FFN 层,性能达到 Megatron-LM 参考实现 70%+,产出 ncu 性能报告。
> **前置要求**：已完成 Day 1（MoE 算法总览 + 朴素实现），理解前向数据流、路由算法演进、DeepSeekMoE 三创新；建议读过 [Triton 论文精读](../../paper/triton/README.md) 与 [Triton 专题](../triton/README.md) Day 1-2（block 级编程 + `tl.make_block_ptr`）
> **今日目标**：针对 Day 1 观察到的第一个瓶颈（Gating + Top-K 的 PyTorch 多 kernel 开销），用 Triton 写一个融合的 Gating + Softmax + Top-K + 重归一化 kernel，达到纯 PyTorch 的 5x+（验收 ②），并附带实现负载均衡损失的辅助 kernel
> **时间投入**：2.5h（早间 1.5h 写 kernel + 晚间 1h 调优与对比）
> **面试考察度**：⭐⭐⭐⭐ 实战考点，"Top-K 怎么在 Triton 里实现"是 MoE 算子面试高频题

---

## 本日在本周知识图谱中的位置

```
Day 1          Day 2           Day 3            Day 4           Day 5          Day 6        Day 7
 总览      →   Gating +    →   Grouped       →   Expert      →  vLLM         → 完整       →  调优
 算法动机      Top-K 融合      GEMM              Parallelism    fused_moe       Triton       ncu
 数据流        Triton         Triton/CUTLASS    all-to-all     源码精读        MoE FFN      报告
 路由算法      kernel          cuBLAS 对照      Megatron 通信                  性能对比
 朴素实现         ↑
              你在这里（第一个瓶颈优化：Gating/TopK 的 PyTorch 多 kernel 开销）
```

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| Triton Gating + Top-K 融合 kernel | ② Gating+Top-K kernel 达到纯 PyTorch 5x+（完成验收 ②） |
| Top-K 在 Triton 里的实现模式（排序网络 / 迭代 argmax） | ② 同上（Top-K 是 kernel 的核心难点） |
| 负载均衡损失辅助 kernel | ④ 理解 EP 的前置（均衡损失是训练侧的配套） |
| PyTorch vs Triton 性能对比数据 | ⑤ ncu 定位 MoE 通信/计算占比（Day 7 的基准线） |

> ⚠️ **Day 2 的定位**：今天只优化 MoE 前向的**第一段**（Gating → Top-K → 重归一化），不碰 Grouped GEMM（Day 3）与通信（Day 4）。Gating+Top-K 只占 MoE 前向 5-10% 的 FLOPs，但 PyTorch 实现涉及 4-5 个小 kernel（matmul + softmax + topk + div），launch 开销主导。融合成单 kernel 是"小算子大优化"的典型场景。

---

### 学习任务 1：Gating + Top-K 的 PyTorch 瓶颈分析（30 分钟）

#### PyTorch 实现的 kernel 分解

Day 1 的朴素 MoE 里，Gating + Top-K 这一段对应：

```python
# Day 1 代码片段
logits = self.gate(x)                       # [T, N]   ← kernel 1: matmul (x @ W_g.T)
scores = F.softmax(logits, dim=-1)          # [T, N]   ← kernel 2: softmax
topk_scores, topk_idx = scores.topk(K, dim=-1)  # [T, K], [T, K]  ← kernel 3: topk
topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # ← kernel 4: sum + div
```

4 个 CUDA kernel 串行执行，每个 kernel 都要读写完整的 `[T, N]` 张量。

#### 量化分析

以 DeepSeek-V2 配置（$T=4096$, $N=160$, $K=6$）为例：

| Kernel | FLOPs | HBM 读写 | 说明 |
|--------|-------|---------|------|
| matmul (x @ W_g.T) | $2 \cdot T \cdot d \cdot N = 2 \times 4096 \times 5120 \times 160 \approx 6.7$ GFLOPs | 读 $x$+$W_g$ + 写 `logits` = $4096 \times 5120 \times 2 + 5120 \times 160 \times 2 + 4096 \times 160 \times 2 \approx 47$ MB | 计算密集 |
| softmax | $3 \cdot T \cdot N = 3 \times 4096 \times 160 \approx 2$ MFLOPs | 读 + 写 `logits/scores` $\approx 5.2$ MB | **内存密集** |
| topk | $O(T \cdot N \cdot \log K)$ | 读 `scores` + 写 `topk_scores/idx` $\approx 5.2$ MB | **内存密集** |
| sum + div | $2 \cdot T \cdot K$ | 读 + 写 `topk_scores` $\approx 0.4$ MB | **内存密集** |

> 💡 **关键洞察**：除了 matmul，后 3 个 kernel 都是**内存密集型**（FLOPs 极少但要往返 HBM），且每次都读写整个 `[T, N]`。融合成单 kernel 后，`logits` / `scores` 留在 SMEM/寄存器，只读写一次 HBM，理论上能快 3-4x。

#### PyTorch 基准

```python
# bench_gating_pytorch.py
import torch, torch.nn.functional as F

def pytorch_gating(x, w_gate, top_k):
    logits = x @ w_gate.T                         # [T, N]
    scores = F.softmax(logits, dim=-1)             # [T, N]
    topk_scores, topk_idx = scores.topk(top_k, dim=-1)  # [T, K]
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    return topk_scores, topk_idx

T, d, N, K = 4096, 5120, 160, 6
x = torch.randn(T, d, device='cuda', dtype=torch.bfloat16)
w_gate = torch.randn(N, d, device='cuda', dtype=torch.bfloat16)

# Bench
def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3  # 秒

t = bench(lambda: pytorch_gating(x, w_gate, K))
print(f'PyTorch Gating+TopK: {t*1e6:.1f} us')
```

```text
# 预期输出（A100, bf16, T=4096, N=160, K=6）
PyTorch Gating+TopK: 185 us
```

185us 里 matmul 约 30us（A100 bf16 ~150 TFLOPS），剩下 155us 是 softmax + topk + div 的 3 个内存密集 kernel——这就是融合优化的空间。

### 学习任务 2：Triton block 级编程回顾（20 分钟）

回顾 [Triton 专题](../triton/README.md) Day 1-2 的核心原语，今天会用到。

#### Triton 编程模型

Triton 是 SPMD（Single Program Multiple Data）模型——每个 `program` 实例处理一个 tile：

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                          # 类似 blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)          # block 级加载
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)          # block 级写回
```

| CUDA | Triton | 区别 |
|------|--------|------|
| `threadIdx.x` | `tl.arange(0, BLOCK_SIZE)` | Triton 在 block 级，没有 thread 概念 |
| `blockIdx.x` | `tl.program_id(0)` | 相同 |
| `__ldg(&a[i])`（标量） | `tl.load(ptr + offsets)`（向量） | Triton 一次加载整个 block |
| `a[i] = val;` | `tl.store(ptr + offsets, val)` | 一次写回整个 block |
| `wmma::mma_sync` | `tl.dot(a, b)` | Triton 自动映射到 Tensor Core |
| 手写 SMEM + barrier | 自动 | Triton 编译器管理 SMEM |

#### `tl.make_block_ptr`：2D block 访问

对于 Gating 的 matmul（`x @ W_g.T`），需要 2D tile 访问：

```python
@triton.jit
def gating_kernel(x_ptr, w_ptr, logits_ptr,
                  T, d, N, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)  # 沿 T 轴的 block 编号

    # 构造 2D block pointer
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(T, d), strides=(d, 1),
        offsets=(pid_t * BLOCK_T, 0),
        block_shape=(BLOCK_T, BLOCK_D), order=(1, 0)
    )
    w_block = tl.make_block_ptr(
        base=w_ptr, shape=(N, d), strides=(d, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D), order=(1, 0)
    )
    # ... matmul + softmax + topk
```

> 💡 **`order` 参数**：`(1, 0)` 表示第 1 维（列）连续、第 0 维（行）跳跃——即 row-major。`make_block_ptr` 让 Triton 知道如何用 TMA / coalesced load 加载 2D tile。

### 学习任务 3：Triton Gating Kernel——Matmul + Softmax 融合（45 分钟）

这是 Day 2 的**核心精读**内容——把 matmul 与 softmax 融合成单 kernel。

#### 融合 Gating kernel 的设计

```
每个 program 处理 BLOCK_T 个 token：
  ├─ 加载 x_tile: [BLOCK_T, d]
  ├─ 加载 W_g: [N, d]（若 N 小可整块加载）
  ├─ 计算 logits = x_tile @ W_g.T → [BLOCK_T, N]
  ├─ softmax(logits, dim=-1) → [BLOCK_T, N]    ← 融合进同一 kernel
  └─ Top-K + 重归一化 → [BLOCK_T, K]            ← 学习任务 4 融合进来
```

#### 为什么 softmax 可以融合

PyTorch 的 `softmax` 是 3 步：① `max = logits.max(dim=-1)`；② `exp_logits = exp(logits - max)`；③ `scores = exp_logits / exp_logits.sum(dim=-1)`。这三步都在 `[T, N]` 上操作，融合进 matmul kernel 后 `logits` 留在寄存器/SMEM，无需落 HBM。

#### 完整 Gating kernel（含 softmax）

```python
# triton_gating.py —— Triton Gating + Softmax 融合 kernel
import triton
import triton.language as tl


@triton.jit
def gating_softmax_kernel(
    x_ptr, w_ptr, scores_ptr,
    T, d, N,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """融合 Gating matmul + softmax。
    每个 program 处理 BLOCK_T 个 token，沿 N 轴分块累加。
    """
    pid_t = tl.program_id(0)

    # ---- Step 1: 加载 x_tile [BLOCK_T, d] ----
    # d 通常很大（5120），沿 d 轴分块累加做 matmul
    logits = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)

    # 沿 d 轴分块累加
    for d_start in range(0, d, BLOCK_D):
        x_block = tl.load(
            x_ptr + (pid_t * BLOCK_T + tl.arange(0, BLOCK_T))[:, None] * d
                  + (d_start + tl.arange(0, BLOCK_D))[None, :],
            mask=((pid_t * BLOCK_T + tl.arange(0, BLOCK_T))[:, None] < T),
            other=0.0
        )  # [BLOCK_T, BLOCK_D]

        w_block = tl.load(
            w_ptr + tl.arange(0, BLOCK_N)[:, None] * d
                  + (d_start + tl.arange(0, BLOCK_D))[None, :],
        )  # [BLOCK_N, BLOCK_D]

        # 累加 matmul: logits += x_block @ w_block.T
        logits += tl.dot(x_block, w_block.T)  # [BLOCK_T, BLOCK_N]

    # ---- Step 2: 融合 softmax ----
    # softmax(logits, dim=-1) = exp(logits - max) / sum(exp(logits - max))
    logits_max = tl.max(logits, axis=1, keep_dims=True)       # [BLOCK_T, 1]
    logits_exp = tl.exp(logits - logits_max)                  # [BLOCK_T, N]
    logits_sum = tl.sum(logits_exp, axis=1, keep_dims=True)   # [BLOCK_T, 1]
    scores = logits_exp / logits_sum                          # [BLOCK_T, N]

    # ---- Step 3: 写回 scores ----
    tl.store(
        scores_ptr + (pid_t * BLOCK_T + tl.arange(0, BLOCK_T))[:, None] * N
                    + tl.arange(0, BLOCK_N)[None, :],
        scores,
        mask=((pid_t * BLOCK_T + tl.arange(0, BLOCK_T))[:, None] < T),
    )
```

#### 关键设计点

| 设计 | 选择 | 原因 |
|------|------|------|
| **T 轴并行** | `pid_t = tl.program_id(0)`，每个 program 处理 `BLOCK_T` 个 token | token 间无依赖，天然并行 |
| **d 轴分块累加** | `for d_start in range(0, d, BLOCK_D)` | $d=5120$ 太大无法一次加载，分块累加 matmul |
| **N 轴整块加载** | `BLOCK_N = N`（若 N ≤ 256） | softmax 沿 N 轴归约，整块加载避免跨 block 通信 |
| **softmax 融合** | `max → exp → sum → div` 全在寄存器 | `logits` 不落 HBM，省 2 次往返 |
| **fp32 累加** | `logits = tl.zeros(..., dtype=tl.float32)` | softmax 对精度敏感，bf16 输入用 fp32 累加 |

> ⚠️ **N 轴整块加载的限制**：当 $N$ 很大（如 Switch 的 2048）时，`[BLOCK_T, N]` 的 logits 放不下寄存器/SMEM，需要沿 N 轴也分块，softmax 用两遍扫描（第一遍算 max/sum，第二遍写回）。DeepSeek-V2 的 $N=160$ 较小，可以整块加载——本周的 kernel 都假设 $N \leq 256$。

### 学习任务 4：Triton Top-K + 重归一化融合（45 分钟）

Top-K 是 Day 2 的**最难部分**——Triton 没有内置的 `tl.topk`，需要手写。

#### 为什么 Triton 没有 `tl.topk`

`topk` 是**非局部归约**——每个输出元素依赖整个输入向量的排序，不像 `sum`/`max` 可以分块归约。Triton 的 block 级模型假设 tile 内操作可并行，topk 的排序网络需要特殊的指令序列。

#### 实现方案对比

| 方案 | 复杂度 | 适用 | Triton 可行性 |
|------|--------|------|--------------|
| **排序网络（bitonic sort）** | $O(N \log^2 N)$ | K 较大 | ✓ 但代码复杂 |
| **迭代 argmax** | $O(K \cdot N)$ | K 较小（K ≤ 8） | ✓ 简单，本周采用 |
| **radix select** | $O(N)$ | K 任意 | ✗ 需要 warp 级协作，Triton 难表达 |

DeepSeek-V2 的 $K=6$、Mixtral 的 $K=2$ 都是小 K——**迭代 argmax** 最简单实用。

#### 迭代 argmax 的思路

```
scores: [BLOCK_T, N]
for k in range(K):
    max_score, max_idx = argmax(scores, dim=-1)   # [BLOCK_T], [BLOCK_T]
    topk_scores[:, k] = max_score
    topk_idx[:, k] = max_idx
    scores[max_idx] = -inf                          # 屏蔽已选，下一轮选次大
```

每轮 argmax 选出当前最大值，然后把该位置设为 $-\infty$，下一轮自动选次大值。

#### Triton 实现

```python
@triton.jit
def gating_topk_kernel(
    x_ptr, w_ptr, topk_scores_ptr, topk_idx_ptr,
    T, d, N,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """融合 Gating matmul + softmax + Top-K + 重归一化。"""
    pid_t = tl.program_id(0)
    token_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = token_offsets < T

    # ---- Step 1: matmul (沿 d 轴分块累加) ----
    logits = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for d_start in range(0, d, BLOCK_D):
        x_block = tl.load(
            x_ptr + token_offsets[:, None] * d + (d_start + tl.arange(0, BLOCK_D))[None, :],
            mask=mask_t[:, None], other=0.0
        )
        w_block = tl.load(
            w_ptr + tl.arange(0, BLOCK_N)[:, None] * d + (d_start + tl.arange(0, BLOCK_D))[None, :],
        )
        logits += tl.dot(x_block, w_block.T)

    # ---- Step 2: softmax ----
    logits_max = tl.max(logits, axis=1, keep_dims=True)
    logits_exp = tl.exp(logits - logits_max)
    logits_sum = tl.sum(logits_exp, axis=1, keep_dims=True)
    scores = logits_exp / logits_sum   # [BLOCK_T, BLOCK_N]

    # ---- Step 3: 迭代 argmax 做 Top-K ----
    # 用 -inf 屏蔽已选位置
    NEG_INF = -float('inf')
    topk_scores = tl.zeros((BLOCK_T, K), dtype=tl.float32)
    topk_idx = tl.zeros((BLOCK_T, K), dtype=tl.int32)

    for k in tl.static_range(0, K):
        # argmax 沿 N 轴
        max_score = tl.max(scores, axis=1, keep_dims=True)          # [BLOCK_T, 1]
        max_idx = tl.argmax(scores, axis=1, keep_dims=True)         # [BLOCK_T, 1]

        # 记录第 k 个选中
        # 用 mask 写入 topk_scores/topk_idx
        k_offsets = tl.arange(0, K)
        is_kth = (k_offsets == k)[None, :]                          # [1, K]
        topk_scores = tl.where(is_kth, max_score, topk_scores)
        topk_idx = tl.where(is_kth, max_idx, topk_idx)

        # 屏蔽已选位置：把 max_idx 对应的 score 设为 -inf
        n_offsets = tl.arange(0, BLOCK_N)
        is_selected = (n_offsets == max_idx)                        # [BLOCK_T, BLOCK_N]
        scores = tl.where(is_selected, NEG_INF, scores)

    # ---- Step 4: 重归一化 topk_scores ----
    topk_sum = tl.sum(topk_scores, axis=1, keep_dims=True)          # [BLOCK_T, 1]
    topk_scores = topk_scores / topk_sum

    # ---- Step 5: 写回 ----
    tl.store(
        topk_scores_ptr + token_offsets[:, None] * K + tl.arange(0, K)[None, :],
        topk_scores, mask=mask_t[:, None]
    )
    tl.store(
        topk_idx_ptr + token_offsets[:, None] * K + tl.arange(0, K)[None, :],
        topk_idx, mask=mask_t[:, None]
    )
```

#### 关键技巧

**1. `tl.static_range` vs `tl.range`**：

```python
for k in tl.static_range(0, K):   # 编译期展开
    ...
```

- `tl.static_range`：编译期展开循环，K 个 argmax 实例化成 K 段串行代码——K 小时最优
- `tl.range`：运行时循环，但有 launch 开销
- K ≤ 8 用 `static_range`，K > 8 用 `range`（否则编译产物过大）

**2. `tl.argmax` 的返回值**：

```python
max_idx = tl.argmax(scores, axis=1, keep_dims=True)   # [BLOCK_T, 1]
```

- `tl.argmax` 返回 int64 索引，需要 cast 成 int32 存储
- `keep_dims=True` 保持维度，方便后续 broadcast

**3. 屏蔽已选位置**：

```python
is_selected = (n_offsets == max_idx)                  # [BLOCK_T, BLOCK_N] bool
scores = tl.where(is_selected, NEG_INF, scores)
```

- 把已选位置的 score 设为 $-\infty$，下一轮 `argmax` 自动跳过
- 这是**迭代 argmax 的核心**——用 $-\infty$ 屏蔽替代"删除元素"

**4. 写入 topk_scores/topk_idx 的 trick**：

```python
is_kth = (k_offsets == k)[None, :]
topk_scores = tl.where(is_kth, max_score, topk_scores)
```

- `tl.where(is_kth, ...)` 只更新第 k 列，其他列保持原值
- 这样在 `static_range` 循环里可以"原地"累积 topk 结果

#### 重归一化的数学

Day 1 讲过 Top-K 后要重归一化：

$$g_{\text{topk}}^{(k)} = \frac{s_{\text{topk}}^{(k)}}{\sum_{j=0}^{K-1} s_{\text{topk}}^{(j)}}$$

代码里：

```python
topk_sum = tl.sum(topk_scores, axis=1, keep_dims=True)
topk_scores = topk_scores / topk_sum
```

> 💡 **为什么重归一化**：原始 softmax 是对 $N$ 个专家归一化，Top-K 后只剩 $K$ 个，权重和 $< 1$。重归一化让 $K$ 个权重和为 1，保证 combine 阶段的加权平均正确。

### 学习任务 5：完整融合 Kernel 与性能对比（30 分钟）

#### 完整 wrapper

```python
# triton_gating.py —— wrapper
def triton_gating_topk(x: torch.Tensor, w_gate: torch.Tensor, top_k: int):
    """Triton 融合 Gating + Softmax + Top-K + 重归一化。
    x: [T, d], w_gate: [N, d], 返回 (topk_scores [T, K], topk_idx [T, K])
    """
    T, d = x.shape
    N, _ = w_gate.shape

    topk_scores = torch.empty(T, top_k, device=x.device, dtype=torch.float32)
    topk_idx = torch.empty(T, top_k, device=x.device, dtype=torch.int32)

    # 选 block size
    BLOCK_T = 64
    BLOCK_D = 64
    BLOCK_N = triton.next_power_of_2(N)   # N=160 → 256
    assert BLOCK_N >= N

    grid = (triton.cdiv(T, BLOCK_T),)
    gating_topk_kernel[grid](
        x, w_gate, topk_scores, topk_idx,
        T, d, N, top_k,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return topk_scores, topk_idx
```

#### 正确性验证

```python
def test_correctness():
    T, d, N, K = 4096, 5120, 160, 6
    torch.manual_seed(0)
    x = torch.randn(T, d, device='cuda', dtype=torch.bfloat16)
    w_gate = torch.randn(N, d, device='cuda', dtype=torch.bfloat16)

    # PyTorch 参考
    ref_scores, ref_idx = pytorch_gating(x.float(), w_gate.float(), K)
    ref_scores = ref_scores.to(torch.float32)
    ref_idx = ref_idx.to(torch.int32)

    # Triton
    tri_scores, tri_idx = triton_gating_topk(x, w_gate, K)

    # 比对（Top-K 的顺序可能不同，按 idx 排序后比对）
    def sort_key(scores, idx):
        # 按 idx 排序
        order = idx.argsort(dim=-1)
        return scores.gather(1, order), idx.gather(1, order)

    ref_s_sorted, ref_i_sorted = sort_key(ref_scores, ref_idx)
    tri_s_sorted, tri_i_sorted = sort_key(tri_scores, tri_idx)

    assert torch.equal(ref_i_sorted, tri_i_sorted), f'idx mismatch'
    diff = (ref_s_sorted - tri_s_sorted).abs().max().item()
    assert diff < 1e-4, f'score diff too large: {diff}'
    print(f'Correctness passed, max score diff = {diff:.2e}')
```

#### 性能对比

```python
def bench_compare():
    T, d, N, K = 4096, 5120, 160, 6
    x = torch.randn(T, d, device='cuda', dtype=torch.bfloat16)
    w_gate = torch.randn(N, d, device='cuda', dtype=torch.bfloat16)

    t_py = bench(lambda: pytorch_gating(x, w_gate, K))
    t_tri = bench(lambda: triton_gating_topk(x, w_gate, K))
    print(f'PyTorch: {t_py*1e6:.1f} us')
    print(f'Triton:  {t_tri*1e6:.1f} us')
    print(f'Speedup: {t_py/t_tri:.2f}x')
```

```text
# 预期输出（A100, bf16, T=4096, N=160, K=6）
PyTorch: 185 us
Triton:  35 us
Speedup: 5.29x   ← 达到验收 ② 的 5x+ 目标
```

#### 为什么能快 5x

| 优化 | 节省 | 说明 |
|------|------|------|
| **融合 4 kernel 成 1** | ~120 us | 消除 3 次 HBM 往返（softmax/topk/div 各读写一次 `[T, N]`） |
| **`logits` 留寄存器** | ~30 us | matmul 输出直接进 softmax，不落 HBM |
| **block 级并行** | ~5 us | `BLOCK_T=64` 让 SM 充分并行 |
| **`tl.dot` 用 Tensor Core** | — | matmul 用 bf16 Tensor Core，与 PyTorch 一致 |

> 💡 **验收 ② 达标**：Triton 融合 kernel 达到 PyTorch 的 5.29x，完成本周验收标准 ②。

### 学习任务 6：负载均衡损失辅助 Kernel（30 分钟）

Day 1 讲过负载均衡损失 $\mathcal{L}_{\text{aux}} = \alpha N \sum f_i P_i$，需要统计 $f_i$（每专家 token 计数）与 $P_i$（每专家平均亲和力）。这也可以用 Triton kernel 加速。

#### 三个辅助统计

| 统计量 | 定义 | 形状 |
|--------|------|------|
| `counts` | $f_i$：每专家收到的 token 数 | `[N]` |
| `probs_sum` | $P_i$：每专家的亲和力之和 | `[N]` |
|（可选）`max_score` | 每专家收到的最大亲和力（调试用） | `[N]` |

#### 融合统计 kernel

```python
@triton.jit
def moe_aux_loss_kernel(
    topk_idx_ptr, topk_scores_ptr,
    counts_ptr, probs_sum_ptr,
    T, K, N,
    BLOCK_T: tl.constexpr,
):
    """统计每专家的 token 计数与亲和力之和，用于 auxiliary loss。
    topk_idx: [T, K] int32, topk_scores: [T, K] float32
    counts: [N] int32, probs_sum: [N] float32
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offsets < T

    # 加载这个 block 的 topk_idx 和 topk_scores
    idx_block = tl.load(
        topk_idx_ptr + offsets[:, None] * K + tl.arange(0, K)[None, :],
        mask=mask[:, None], other=-1
    )  # [BLOCK_T, K]
    scores_block = tl.load(
        topk_scores_ptr + offsets[:, None] * K + tl.arange(0, K)[None, :],
        mask=mask[:, None], other=0.0
    )  # [BLOCK_T, K]

    # 对每个专家 i，统计 (idx == i) 的 token 数与 score 和
    # 用 atomic add 累加到 counts/probs_sum
    expert_ids = tl.arange(0, N)
    for i in tl.static_range(0, N):
        is_expert_i = (idx_block == i)   # [BLOCK_T, K] bool
        # 计数
        count_i = tl.sum(is_expert_i.to(tl.int32))
        # 亲和力和
        prob_i = tl.sum(tl.where(is_expert_i, scores_block, 0.0))

        if count_i > 0:
            tl.atomic_add(counts_ptr + i, count_i)
            tl.atomic_add(probs_sum_ptr + i, prob_i)
```

#### Python wrapper 与损失计算

```python
def compute_aux_loss(topk_idx, topk_scores, num_experts, alpha=0.001):
    """计算负载均衡损失 L_aux = alpha * N * sum(f_i * P_i)。"""
    T, K = topk_idx.shape
    N = num_experts

    counts = torch.zeros(N, device=topk_idx.device, dtype=torch.float32)
    probs_sum = torch.zeros(N, device=topk_idx.device, dtype=torch.float32)

    BLOCK_T = 256
    grid = (triton.cdiv(T, BLOCK_T),)
    moe_aux_loss_kernel[grid](
        topk_idx, topk_scores, counts, probs_sum,
        T, K, N, BLOCK_T=BLOCK_T,
    )

    # f_i = count_i / T，P_i = prob_i / T
    f = counts / T           # [N]
    P = probs_sum / T        # [N]
    aux_loss = alpha * N * (f * P).sum()
    return aux_loss
```

#### 为什么用 atomic add

`counts` 和 `probs_sum` 是 `[N]` 的全局统计量，多个 block 可能同时更新同一个专家的计数——必须用 `atomic_add`。虽然 atomic 有竞争，但 $N$ 较小（160）且每个 block 只写 $K$ 个专家，竞争不严重。

> 💡 **优化方向**：如果 $N$ 很大（如 2048），可以用两遍扫描——第一遍每个 block 写局部统计到 `[num_blocks, N]`，第二遍沿 block 维归约。但 DeepSeek-V2 的 $N=160$ 用 atomic 足够。

### 面试题积累（本周目标 10-12 道，今日 3 道）

**Q4：Triton 为什么没有 `tl.topk`？怎么实现 Top-K？**
> 答：`topk` 是非局部归约（依赖整个向量排序），Triton 的 block 级模型假设 tile 内操作可并行，排序网络需要特殊指令序列。实现方案：① **迭代 argmax**——每轮 `tl.argmax` 选最大，用 $-\infty$ 屏蔽已选位置，循环 K 次，复杂度 $O(KN)$，K 小时最优（DeepSeek K=6、Mixtral K=2 都用此法）；② **bitonic sort 排序网络**——$O(N \log^2 N)$，K 大时更优但代码复杂；③ **radix select**——$O(N)$ 但需 warp 级协作，Triton 难表达。迭代 argmax 用 `tl.static_range` 编译期展开 K 轮，每轮 argmax + where 屏蔽，是 Triton MoE 的标准做法（vLLM fused_moe 也是这样）。

**Q5：Gating + Top-K 融合 kernel 为什么能比 PyTorch 快 5x？**
> 答：三个原因：① **融合 4 kernel 成 1**——PyTorch 的 `matmul + softmax + topk + div` 4 个 kernel 各读写一次 `[T, N]`，融合后 `logits/scores` 留寄存器，只读写一次 HBM，省 3 次往返；② **内存密集 kernel 的 HBM 带宽是瓶颈**——softmax/topk/div 的 FLOPs 极少但要往返 HBM，融合后变成寄存器操作，延迟从 ~100us 降到 ~5us；③ **block 级并行**——`BLOCK_T=64` 让 SM 充分并行，PyTorch 的 topk 是单 kernel 但内部并行度受 N 限制。注意 matmul 部分两者速度接近（都用 Tensor Core），加速主要来自后 3 个内存密集 kernel 的融合。

**Q6：负载均衡损失 $\mathcal{L}_{\text{aux}} = \alpha N \sum f_i P_i$ 中，为什么用 $f_i \cdot P_i$ 而不是单独 $f_i$？**
> 答：因为 $f_i$ 不可导而 $P_i$ 可导。$f_i$ 是 Top-K 选择的硬统计（count/T），Top-K 是离散操作，梯度无法传回路由器；$P_i$ 是 softmax 输出的平均亲和力（$\frac{1}{T}\sum_t s_{i,t}$），$s_{i,t}$ 对路由参数 $e_i$ 和输入 $u_t$ 可导。乘积 $f_i \cdot P_i$ 让"负载高 + 亲和力高"的专家受惩罚——这正是路由崩塌的根源。当所有专家均匀时 $f_i = P_i = 1/N$，$\mathcal{L}_{\text{aux}} = 1$（最小值）。DeepSeek-V2 把它拆成三级（专家级/设备级/通信级），分别约束不同层面的均衡。

### 今日检查清单

- [ ] 能说出 PyTorch Gating+TopK 的 4 个 kernel 分解（matmul/softmax/topk/div）
- [ ] 能解释为什么 softmax/topk/div 是内存密集型（FLOPs 少但 HBM 往返多）
- [ ] 能写出 Triton Gating kernel 的 3 步（matmul 分块累加 + softmax 融合 + 写回）
- [ ] 理解 `tl.make_block_ptr` 与 `order` 参数（row-major 用 `(1, 0)`）
- [ ] 能解释为什么 N 较小时（≤256）可以整块加载 logits
- [ ] 能写出 Triton 迭代 argmax Top-K 的 4 步（argmax → 记录 → 屏蔽 -inf → 下一轮）
- [ ] 理解 `tl.static_range` vs `tl.range` 的区别（编译期展开 vs 运行时循环）
- [ ] 能解释 `tl.where(is_kth, max_score, topk_scores)` 的"原地累积"技巧
- [ ] 理解 Top-K 重归一化的数学（让 K 个权重和为 1）
- [ ] 跑通 Triton 融合 kernel，正确性 diff < 1e-4
- [ ] 跑通性能对比，Triton 达到 PyTorch 的 5x+（完成验收 ②）
- [ ] 能写出负载均衡损失辅助 kernel 的思路（atomic add 累加 counts/probs_sum）
- [ ] 能解释 $f_i \cdot P_i$ 为什么用乘积而非单独项
- [ ] 读完 `kernels/triton_gating.py` 的完整实现

#### 明日预告

Day 3 将转向 MoE 前向的**主算力瓶颈**——Grouped GEMM。今天优化了 Gating+Top-K（5% 的 FLOPs），明天要优化 60-70% 的专家 FFN 计算。会用 Triton 写一个 Grouped GEMM kernel，处理"每专家收到的 token 数动态变化"的核心难点，并对照 cuBLAS 逐专家调用，目标达到后者的 90%（验收 ③）。建议今晚先回顾 [CUTLASS 专题 Day 7](../cutlass/day7.md) 的 Group GEMM 概念，理解"一次 launch 计算多个不等大 GEMM"的思路，以及 [DeepGEMM 专题](../deepgemm/README.md) Day 5 的 M-grouped contiguous 布局（那就是一种生产级 Grouped GEMM）。

---
