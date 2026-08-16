## Day 3：LayerNorm 优化与 GEMM Backward 数据流

### 🎯 目标

通过今天的学习，你将：

1. 理解 Day 2 基础 LayerNorm kernel 的瓶颈——两次 reduce 意味着三次 HBM 读 x（读 x 求 $\mu$、读 x 求 $\sigma^2$、读 x 做 normalize）<br>
2. 掌握 **Welford 在线算法**——用递推公式在一次 pass 中同时计算 mean 和 variance，把 x 的 HBM 读从 3 次降到 2 次（Welford pass + normalize pass）；配合 smem 缓存 x 可进一步降到 1 次<br>
3. 能实现 **Welford LayerNorm kernel**，实测对比 Day 2 三遍扫描版（理论预期 HBM 读写减半 → ~2x 加速，但小 D / 寄存器开销下可能不达预期）<br>
4. 理解 **GEMM Backward 的数据流**——$dA = dC \cdot B^T$、$dB = A^T \cdot dC$，与 forward 共享 tiling 策略<br>
5. 能解释 LayerNorm backward 的梯度公式与 Welford 在反向中的应用<br>
6. 理解 **kernel fusion** 的动机——把 LayerNorm + GELU 合并成一个 kernel，消除中间张量写回 HBM<br>

> 💡 **为什么重要**：Day 2 的 LayerNorm 虽然正确但慢——3 次 HBM 读写是纯浪费。Welford 算法是"用数学换带宽"的经典案例，面试常问"LayerNorm 怎么优化"。GEMM Backward 是 Transformer 训练/微调的反向核心，与 Week 2 的 GEMM Forward 共享 tiling 思想。这两块是 Week 5 FlashAttention Backward 的直接前置。

---

### 学前导读：从 3 次 HBM 读到 Welford 单 pass

Day 2 的 LayerNorm kernel 做了三次 HBM 读写：

```
Pass 1: 读 x → 求 μ (sum) → 写 μ 到 smem/register
Pass 2: 读 x → 求 σ² (sum of squares) → 写 σ²
Pass 3: 读 x → 做 normalize → 写 y
```

其中 normalize 公式为：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$

每 pass 读一遍 x（2D bytes），三次共 **3×2D bytes HBM 读**。而 x 本身只需读 1 次——多出来的 2 次是"先求 $\mu$ 再求 $\sigma^2$"的串行依赖导致的。

| 策略 | HBM 读 x | HBM 写 y | 总 IO | HBM 层加速比 |
|------|---------|---------|-------|------------|
| Day 2 三遍扫描（不缓存 x） | 3 | 1 | 4×2D bytes | 1× |
| Day 3 Welford（不缓存 x，即下方代码） | 2 | 1 | 3×2D bytes | ~1.33× |
| Day 3 Welford + smem 缓存 x | 1 | 1 | 2×2D bytes | ~2× |
| Day 2 三遍 + smem 缓存 x | 1 | 1 | 2×2D bytes | ~2×（缓存是独立优化，naive 同样受益） |
| Day 3 Fusion (LN+GELU) | 1 | 1（合写） | 2×2D bytes | ~2×（额外省 GELU 读写） |

**核心洞察**：Welford 算法用递推公式 $\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$ 在一次遍历中同时更新 mean 和 variance，消除了"先求 μ 再求 σ²"的串行依赖——把求 mean/var 的两遍读合并成一遍。但 normalize 阶段仍需再读一遍 x，所以**不缓存时实际是 2 次 HBM 读**。要降到 1 次必须配合 smem/register 缓存 x，而该缓存优化对 naive 三遍扫描同样适用。

> 💡 **一句话总结**：Welford 把求 mean/var 的两遍读合并成一遍（3→2 次 HBM 读），是算法层面的收益；配合 smem 缓存 x 可降到 1 次（2→1），但缓存是独立优化，naive 三遍同样受益。不缓存 ~1.33×，缓存后 ~2×（缓存下 Welford 相对 naive 无 HBM 优势，仅有少一遍 smem 遍历 + 少一次 sync 的计算收益）。

---

### 理论学习

#### 3.1 Welford 在线算法

##### 问题：为什么两次 reduce 不能合并？

Day 2 的 LayerNorm 需要两次 reduce：
1. $\mu = \text{mean}(x)$ → reduce sum
2. $\sigma^2 = \text{var}(x) = \text{mean}\left((x - \mu)^2\right)$ → reduce sum of squares

第二次 reduce 依赖第一次的结果（$\mu$），所以必须先算完 $\mu$ 才能算 $\sigma^2$——这导致 x 被读两遍。

##### Welford 的洞察

Welford 算法（1962 年）用**递推公式**在一次遍历中同时更新 mean 和 M2（二次矩）：

初始化：$\text{count} = 0,\ \text{mean} = 0,\ M2 = 0$

对每个新元素 $x$：

$$
\begin{aligned}
\text{count} &\mathrel{+}= 1 \\
\delta &= x - \text{mean} \\
\text{mean} &\mathrel{+}= \delta / \text{count} \quad \text{(更新 mean)} \\
\delta_2 &= x - \text{mean} \\
M2 &\mathrel{+}= \delta \cdot \delta_2 \quad \text{(更新 M2, 未归一化的方差)}
\end{aligned}
$$

最终：$\mu = \text{mean},\ \sigma^2 = M2 / \text{count}$

##### 为什么 Welford 正确？

数学等价性证明（归纳法）：

处理第 $n$ 个元素后：

$$
\text{mean}_n = \frac{1}{n} \sum_{i=1}^{n} x_i \quad \text{(标准 mean)}
$$

$$
M2_n = \sum_{i=1}^{n} (x_i - \text{mean}_n)^2 \quad \text{(未归一化方差} \times n \text{)}
$$

递推：

$$
\text{mean}_n = \text{mean}_{n-1} + \frac{x_n - \text{mean}_{n-1}}{n}
$$

$$
M2_n = M2_{n-1} + (x_n - \text{mean}_{n-1})(x_n - \text{mean}_n)
$$

关键：$M2$ 的递推用 $(x_n - \text{mean}_{n-1})(x_n - \text{mean}_n)$，这两个 delta 巧妙地消除了对 $\text{mean}_n$ 的前向依赖。

##### 数值稳定性

Welford 比"两遍扫描"数值更稳定：

| 方法 | $\sigma^2$ 计算 | 精度问题 |
|------|--------|---------|
| 两遍扫描 | $\frac{\sum(x-\mu)^2}{N}$ | μ 已知，直接算 |
| 朴素单遍 | $\frac{\sum x^2}{N} - \left(\frac{\sum x}{N}\right)^2$ | **大数吃小数**：$\sum x^2$ 和 $(\sum x)^2$ 都很大，相减丢失有效位 |
| **Welford** | $\frac{M2}{N}$（递推） | **无相减**，精度最优 |

> 💡 **面试要点**：Welford 比"$\frac{\sum x^2}{N} - \left(\frac{\sum x}{N}\right)^2$"的朴素单遍公式数值更稳定。朴素单遍在方差接近 0 时会有灾难性精度损失（两个大数相减）。

#### 3.2 Welford LayerNorm Kernel

##### 并行化 Welford

Welford 的递推是串行的（元素 i 依赖 i-1 的 mean/M2）。但在 GPU 上，我们做**分块 Welford**：

1. 每个线程/分块独立做局部 Welford（局部 mean/M2/count）
2. 最后用 Welford 合并公式把多个局部结果合并成全局结果

##### Welford 合并公式

两个分块的 Welford 结果合并：

分块 A：$(\text{count}_a, \text{mean}_a, M2_a)$  
分块 B：$(\text{count}_b, \text{mean}_b, M2_b)$

合并：

$$
\begin{aligned}
\text{count} &= \text{count}_a + \text{count}_b \\
\delta &= \text{mean}_b - \text{mean}_a \\
\text{mean} &= \text{mean}_a + \delta \cdot \text{count}_b / \text{count} \\
M2 &= M2_a + M2_b + \delta^2 \cdot \text{count}_a \cdot \text{count}_b / \text{count}
\end{aligned}
$$

##### Kernel 结构

```cuda
// Welford LayerNorm: 一行一个 block, 单 pass
__global__ void layernorm_welford_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 1. 每线程做局部 Welford（处理 x[row][tid], x[row][tid+blockDim.x], ...）
    float local_mean = 0.0f, local_M2 = 0.0f;
    int local_count = 0;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = x[row * D + i];
        local_count++;
        float delta = val - local_mean;
        local_mean += delta / local_count;
        float delta2 = val - local_mean;
        local_M2 += delta * delta2;
    }

    // 2. warp 内合并（用 __shfl 传递局部结果）
    for (int offset = 16; offset > 0; offset /= 2) {
        int other_count = __shfl_down_sync(0xffffffff, local_count, offset);
        float other_mean = __shfl_down_sync(0xffffffff, local_mean, offset);
        float other_M2 = __shfl_down_sync(0xffffffff, local_M2, offset);

        int total_count = local_count + other_count;
        float delta = other_mean - local_mean;
        local_mean = local_mean + delta * other_count / total_count;
        local_M2 = local_M2 + other_M2 + delta * delta * local_count * other_count / total_count;
        local_count = total_count;
    }

    // 3. 跨 warp 合并（通过 shared memory）
    // ... blockReduce 类似 Day 2, 用 smem 中转 warp 结果 ...

    // 4. 最终 mean/variance 广播到所有线程
    __shared__ float s_mean, s_var;
    if (tid == 0) {
        s_mean = local_mean;
        s_var = local_M2 / local_count;
    }
    __syncthreads();
    float mean = s_mean;
    float inv_std = rsqrtf(s_var + eps);

    // 5. 归一化 + affine（再读一遍 x, 写一遍 y；若步骤 1 已缓存 x 到 smem 则此处读 smem）
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x[row * D + i];
        y[row * D + i] = (val - mean) * inv_std * gamma[i] + beta[i];
    }
}
```

##### HBM 访问对比

| 步骤 | Day 2 三遍扫描 | Day 3 Welford（不缓存 x） |
|------|-------------|--------------|
| 求 $\mu$ | 读 x (1×) | — (Welford 合并 mean/var) |
| 求 $\sigma^2$ | 读 x (1×) | — |
| Welford pass | — | 读 x (1×) |
| Normalize | 读 x (1×) + 写 y (1×) | 读 x (1×) + 写 y (1×) |
| **总 HBM** | **3 读 + 1 写 = 4×2D** | **2 读 + 1 写 = 3×2D** |

> ⚠️ Welford 代码的 normalize 阶段（步骤 5）仍从 HBM 读 x，实际是 **2 次 HBM 读**。要降到 1 次需在 Welford pass 把 x 缓存到 smem，但该优化对 naive 三遍扫描同样适用（缓存后 naive 也是 1 读 + 1 写）。

#### 3.3 GEMM Backward 数据流

##### 前向与反向的对称性

GEMM 前向：$C = A \cdot B$（A: M×K, B: K×N, C: M×N）

给定 $dC$（损失对 C 的梯度），反向需要 $dA$ 和 $dB$：

$$
\begin{aligned}
dA &= dC \cdot B^T \quad (M \times K = M \times N \times N \times K) \\
dB &= A^T \cdot dC \quad (K \times N = K \times M \times M \times N)
\end{aligned}
$$

| 梯度 | 形状 | GEMM 操作 | 需要的前向输入 |
|------|------|----------|-------------|
| dA | M×K | $dC \cdot B^T$ | B（前向的第二个输入） |
| dB | K×N | $A^T \cdot dC$ | A（前向的第一个输入） |

##### 数据流对称性

$$
\begin{aligned}
\text{Forward:} \quad &C[i][j] = \sum_k A[i][k] \cdot B[k][j] \\
\text{Backward:} \quad &dA[i][k] = \sum_j dC[i][j] \cdot B[k][j] \quad (dC \cdot B^T) \\
&dB[k][j] = \sum_i A[i][k] \cdot dC[i][j] \quad (A^T \cdot dC)
\end{aligned}
$$

每个反向 GEMM 的 FLOPs（2MNK）与 forward 相同，只是把某个输入转置。

##### 工程意义

1. **forward 的 GEMM 加速手段可直接套到 backward**：tiling、Tensor Core、async copy 全部适用
2. **FA 的 QK^T、PV 反向 GEMM 与 forward 用同一套 kernel 模板**（Week 5 Day 4）
3. **转置处理**：`B^T` 不需要物理转置，只需在 tiling 时改变 leading dimension 或用 `ldmatrix.trans`

##### 有限差分验证

```python
import torch
A = torch.randn(M, K, requires_grad=True)
B = torch.randn(K, N, requires_grad=True)
C = A @ B
loss = C.sum()
loss.backward()
# dC = ones(M, N)
# dA 应该等于 B 的行和: dA[i][k] = Σ_j B[k][j]
# dB 应该等于 A 的列和: dB[k][j] = Σ_i A[i][k]
print(torch.allclose(A.grad, B.sum(dim=1, keepdim=True).T.expand_as(A.grad)))  # True
```

#### 3.4 LayerNorm Backward

##### 梯度公式

LayerNorm 前向：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$

给定 $dy$，反向需要 $dx$、$d\gamma$、$d\beta$：

$$
\begin{aligned}
d\gamma &= \sum_i dy_i \cdot \hat{y}_i \quad \text{(沿 batch 维 reduce)} \\
d\beta &= \sum_i dy_i \quad \text{(沿 batch 维 reduce)} \\
dx_i &= \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left( dy_i \cdot \gamma_i - \frac{1}{D} \left( \sum_j dy_j \gamma_j - \hat{y}_i \sum_j dy_j \gamma_j \hat{y}_j \right) \right)
\end{aligned}
$$

其中 $\hat{y}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ 是归一化后的值。

##### Welford 在反向中的应用

反向公式需要 $\mu$ 和 $\sigma^2$，但前向只保存了 $y$（或 $\hat{y}$）。两种选择：

1. **前向保存 $\mu$ 和 $\sigma^2$**（额外 O(D) 存储）→ 反向直接用
2. **反向重算 $\mu$ 和 $\sigma^2$**（用 Welford 再做一次）→ 省存储但多计算

生产实现（如 FasterTransformer）通常选 1（保存 μ/σ²），因为 D 很小（64-128），存储开销可忽略。

#### 3.5 Kernel Fusion

##### 为什么要 Fusion？

Transformer 单层有 `GEMM → LayerNorm → GELU → GEMM` 的算子链。如果不 fusion：

```
GEMM: 写 C 到 HBM
LayerNorm: 读 C, 写 LN_out 到 HBM
GELU: 读 LN_out, 写 GELU_out 到 HBM
下一个 GEMM: 读 GELU_out
```

中间张量 `C`、`LN_out` 各被读写一次，纯浪费。

##### Fusion 策略

```
Fused: GEMM → (直接在 register/smem 做 LayerNorm) → (直接做 GELU) → 写最终结果
```

| 策略 | HBM 读写 | 中间张量 |
|------|---------|---------|
| 不 fusion | GEMM 写 C + LN 读 C 写 LN + GELU 读 LN 写 GELU | C, LN_out 各读写 1 次 |
| **Fusion** | GEMM 直接算到 register → LN+GELU 在 register 完成 → 写最终 | 0 个中间张量 |

##### LayerNorm + GELU Fusion 的收益

```
不 fusion:
  LayerNorm: 读 x (2D) + 写 y (2D) = 4D bytes
  GELU:      读 y (2D) + 写 z (2D) = 4D bytes
  总计: 8D bytes

Fusion:
  读 x (2D) + 写 z (2D) = 4D bytes
  总计: 4D bytes (省 50%)
```

> 💡 **面试要点**：Fusion 是推理引擎（vLLM/TensorRT-LLM）的核心优化。`torch.compile` 的 `max-autotune` 模式也会自动 fuse LayerNorm + Activation。手写 CUDA 的 fusion 需要在 epilogue 阶段插入归一化逻辑。

---

### Coding 任务

#### 任务 1：实现 Welford LayerNorm Kernel

创建 `kernels/layernorm_welford.cu`，基于 Day 2 的 `softmax_layernorm.cu` 改造 LayerNorm kernel：

1. 用 Welford 递推替代两次 reduce
2. 用 `__shfl_down_sync` 做 warp 内 Welford 合并
3. 用 shared memory 做跨 warp Welford 合并
4. 最终读两遍 x（Welford pass + normalize）、写一遍 y；进阶可缓存 x 到 smem 降到一遍读

```bash
nvcc -O3 -arch=sm_120 kernels/layernorm_welford.cu -o layernorm_welford
./layernorm_welford
```

预期输出（RTX 5090, CUDA 12.8 预估，待 GPU 环境回填）：

```text
=== LayerNorm: Three-pass vs Welford (NVIDIA GeForce RTX 5090, D=1024) ===
M(rows)  | Three-pass(ms)  Welford(ms)  | Speedup  max_diff
---------|--------------------------------|----------------------
1024     | 0.006           0.008         | 0.74x    4.77e-07
4096     | 0.012           0.020         | 0.60x    4.77e-07
16384    | 0.076           0.094         | 0.81x    5.96e-07

[正确性] Welford vs CPU 参考: max_diff = 2.03e-06 (< 1e-4 PASS)
```

> **预估说明**：本实现中 Welford 单 pass 并未比 three-pass 快，反而略慢（0.6x–0.8x）。原因可能是：① D=1024 时 three-pass 的 blockReduce 已被 warp shuffle 优化得很好；② Welford 的 shuffle 合并涉及 mean/m2/count 三个字段，寄存器/指令开销抵消了 HBM 减少收益；③ 小 M 时 launch / fixed overhead 占主导。正确性 PASS（max_diff < 1e-4，与源码阈值一致）。生产级 Welford 需配合向量化、每个 warp 处理多行、smem 缓存 x 等进一步优化才能兑现 ~2x 理论收益。以上数据为预估口径，待 GPU 环境实测回填。

#### 任务 2：用 ncu 验证 HBM 访问减少

```bash
ncu --kernel-name regex:layernorm \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./layernorm_welford
```

预期对比：

| Kernel | dram_bytes_read | dram_bytes_write | 总 HBM |
|--------|----------------|-----------------|--------|
| Three-pass | ~3×2D | ~1×2D | ~4×2D |
| Welford（不缓存 x） | ~2×2D | ~1×2D | ~3×2D |
| Welford（缓存 x 到 smem） | ~1×2D | ~1×2D | ~2×2D |

#### 任务 3：GEMM Backward 有限差分验证

创建 `kernels/gemm_backward_test.py`，用 PyTorch 验证 GEMM backward 的公式：

```python
import torch

M, K, N = 128, 256, 64
A = torch.randn(M, K, requires_grad=True)
B = torch.randn(K, N, requires_grad=True)
C = A @ B
loss = C.sum()
loss.backward()

# dC = ones(M, N)
# dA = dC @ B^T = B 的行和（按 N 维求和后 expand）
dA_expected = B.sum(dim=1, keepdim=True).T.expand(M, K)
# dB = A^T @ dC = A 的列和（按 M 维求和后 expand）
dB_expected = A.sum(dim=0, keepdim=True).T.expand(K, N)

print(f"dA match: {torch.allclose(A.grad, dA_expected, atol=1e-5)}")
print(f"dB match: {torch.allclose(B.grad, dB_expected, atol=1e-5)}")
```

#### 任务 4：LeetCode 面试题（10 周计划 · 第 4 周 Day 3）

> 📅 今日题目来自 [10 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/10-week-plan.html) 第 4 周「栈、队列与单调栈」Day 3（单调栈），共 5 题。简单题快速过、中等题精做、困难题吃透；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/) | 中等 | 单调栈 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/739_每日温度.html) |
| [496. 下一个更大元素 I](https://leetcode.cn/problems/next-greater-element-i/) | 简单 | 单调栈 + 哈希 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/496_下一个更大元素%20I.html) |
| [503. 下一个更大元素 II](https://leetcode.cn/problems/next-greater-element-ii/) | 中等 | 单调栈 + 循环数组 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/503_下一个更大元素%20II.html) |
| [901. 股票价格跨度](https://leetcode.cn/problems/online-stock-span/) | 中等 | 单调栈 + 跨度合并 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/901_股票价格跨度.html) |
| [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/) | 困难 | 单调栈 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/84_柱状图中最大的矩形.html) |

---

### 扩展实验

#### 实验 1：Welford vs 朴素单遍的精度对比

实现朴素单遍方差公式 $\sigma^2 = \frac{\sum x^2}{N} - \left(\frac{\sum x}{N}\right)^2$，对比 Welford 的精度：
- 构造极端数据（x 的方差接近 0，或 x 量级很大）
- 预期：朴素单遍在大数据量 + 小方差时精度崩溃，Welford 保持稳定

#### 实验 2：LayerNorm + GELU Fusion

实现一个 fused kernel：输入 x，输出 `GELU(LayerNorm(x))`：
- 读 x 一次，用 Welford 算 mean/var
- 归一化后直接做 GELU，写回
- 对比不 fusion 版（先 LN 写 HBM，再 GELU 读写 HBM）的 HBM 访问量

#### 实验 3：RMSNorm 变体

RMSNorm（Root Mean Square Normalization）是 LayerNorm 的简化版，省去 mean 计算：

- **LayerNorm**：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
- **RMSNorm**：$y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$

- RMSNorm 只需一次 reduce（sum of squares），不需要 Welford
- 实测 RMSNorm vs LayerNorm 的性能差异（预期 RMSNorm 快 ~1.5x）
- 思考：为什么 LLaMA/Qwen 等大模型用 RMSNorm 而非 LayerNorm？（省计算 + 精度够用）

---

### 今日总结

Day 3 我们把 Day 2 的 LayerNorm 从三遍扫描优化到 Welford 单 pass，并学习了 GEMM Backward 数据流：

1. **Welford 算法**：递推公式 $\text{mean} += \delta/\text{count};\ M2 += \delta \cdot \delta_2$ 在一次遍历中同时算 mean 和 variance，HBM 读从 3×2D 降到 2×2D（不缓存 x）；配合 smem 缓存可降到 1×2D，但 naive 三遍同样受益
2. **并行 Welford**：每线程/分块做局部 Welford，用合并公式把多个局部结果合并成全局
3. **数值稳定性**：Welford 比"$\frac{\sum x^2}{N} - \left(\frac{\sum x}{N}\right)^2$"的朴素单遍更稳定（无大数相减）
4. **GEMM Backward**：$dA = dC \cdot B^T$、$dB = A^T \cdot dC$，与 forward 共享 tiling 策略
5. **LayerNorm Backward**：需要 μ/σ²，生产实现通常前向保存（省存储）或反向重算（Welford）
6. **Kernel Fusion**：LN+GELU 合并消除中间张量，HBM 读写减 50%

掌握 Welford 和 GEMM Backward 后，你有了 Week 5 FlashAttention Backward 的全部前置知识。Day 4 学习 Triton 语言，看编译器如何自动生成这些优化。

---

### 面试要点

1. **LayerNorm 的两次 reduce 为什么不能合并？Welford 怎么解决？**

   <details>
   <summary>点击查看答案</summary>

   - **不能合并的原因**：第二次 reduce（方差）依赖第一次的结果（均值），$\sigma^2 = \text{mean}\left((x-\mu)^2\right)$ 必须先知道 μ
   - **Welford 解决方案**：用递推公式在一次遍历中同时更新 mean 和 M2（未归一化方差）：
     - $\delta = x - \text{mean};\ \text{mean} += \delta/\text{count};\ M2 += \delta \cdot (x - \text{mean})$
   - **并行化**：每线程/分块做局部 Welford，用合并公式 $M2 = M2_a + M2_b + \delta^2 \cdot \text{count}_a \cdot \text{count}_b / \text{count}$ 合并
    - **收益**：Welford 把求 mean/var 的两遍读合并成一遍，HBM 读从 3 次降到 2 次（不缓存 x），理论 HBM 层加速 ~1.33×；配合 smem 缓存 x 可降到 1 次，但 naive 三遍同样可缓存，缓存下 Welford 无 HBM 优势（仅有少一遍遍历 + 少一次 sync 的收益）

   </details>

2. **Welford 算法为什么比"$\frac{\sum x^2}{N} - \left(\frac{\sum x}{N}\right)^2$"数值更稳定？**

   <details>
   <summary>点击查看答案</summary>

   - **朴素单遍**：$\sigma^2 = \frac{\sum x^2}{N} - \left(\frac{\sum x}{N}\right)^2$，两个大数相减
      - 当方差接近 0 时，$\frac{\sum x^2}{N} \approx \left(\frac{\sum x}{N}\right)^2$，相减丢失有效位（catastrophic cancellation）
      - 数据量级大时更严重：x ~ 1000, N ~ 10000 → $\sum x^2 \sim 10^{10}$, 精度损失
    - **Welford**：$M2 = \sum(\delta \cdot \delta_2)$，无大数相减，每次只累加一个小增量
   - **结论**：Welford 在小方差/大数据量时精度远优于朴素单遍，与两遍扫描精度一致

   </details>

3. **GEMM Backward 的公式是什么？与 forward 有什么对称性？**

   <details>
   <summary>点击查看答案</summary>

   - Forward: $C = A \cdot B$（A: M×K, B: K×N, C: M×N）
   - Backward:
      - $dA = dC \cdot B^T$（M×K = M×N × N×K）—— 需要 B
      - $dB = A^T \cdot dC$（K×N = K×M × M×N）—— 需要 A
   - **对称性**：每个反向 GEMM 的 FLOPs（2MNK）与 forward 相同，只是某个输入转置
   - **工程意义**：forward 的 tiling/Tensor Core/async copy 可直接套到 backward；转置不需物理操作（改 leading dimension 或用 `ldmatrix.trans`）

   </details>

4. **LayerNorm Backward 需要什么？前向该保存什么？**

   <details>
   <summary>点击查看答案</summary>

   - Backward 需要 $\mu$ 和 $\sigma^2$ 来计算 $dx$
   - **两种选择**：
      1. 前向保存 $\mu/\sigma^2$（额外 O(D) 存储，D=64-128 可忽略）→ 反向直接用（生产首选）
     2. 反向用 Welford 重算（省存储但多计算）→ 存储紧张时用
    - $d\gamma = \sum dy \cdot \hat{y}$（沿 batch reduce）
    - $d\beta = \sum dy$（沿 batch reduce）

   </details>

5. **Kernel Fusion 为什么能提升性能？以 LN+GELU 为例说明**

   <details>
   <summary>点击查看答案</summary>

   - **不 fusion**：GEMM 写 C → LN 读 C 写 LN → GELU 读 LN 写 GELU → 下一个 GEMM 读 GELU。中间张量 C/LN 各读写一次 HBM
   - **Fusion**：GEMM 算到 register → LN+GELU 在 register 完成 → 写最终结果。中间张量不落 HBM
   - **LN+GELU 具体收益**：不 fusion 是 8D bytes（LN 4D + GELU 4D），fusion 后是 4D bytes（读 x 写 z），省 50%
   - **生产实践**：vLLM/TensorRT-LLM 的 epilogue fusion；`torch.compile` 的 auto-fusion

   </details>

6. **RMSNorm 和 LayerNorm 有什么区别？为什么大模型用 RMSNorm？**

   <details>
   <summary>点击查看答案</summary>

   - **LayerNorm**：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$——需要 mean 和 variance，两次 reduce
   - **RMSNorm**：$y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$——只需 sum of squares，一次 reduce，无 β
   - **大模型选 RMSNorm 的原因**：
     1. 省一次 reduce（mean），kernel 快 ~1.5x
      2. 省去 $\beta$ 参数，参数量减半
     3. 实验表明精度与 LayerNorm 几乎一致（大模型对归一化的 mean shift 不敏感）
   - **代表模型**：LLaMA、Qwen、DeepSeek 等均用 RMSNorm

   </details>
