## Day 2：FP8 量化深入 —— E4M3/E5M2 kernel 与 GPTQ vs AWQ 对比

### 🎯 目标

通过今天的学习，你将：

1. 深化 Day 1 的 FP8 入门——掌握 **E4M3 与 E5M2** 两种格式的精度/范围权衡与适用场景<br>
2. 理解 **FP8 GEMM 的混合精度策略**——输入 FP8、累加 FP32、输出 FP16/FP32，以及 scaling factor 的作用<br>
3. 能对比 **GPTQ vs AWQ vs SmoothQuant** 三条量化路线——算法原理、校准方式、部署友好度<br>
4. 了解 **FP4 量化**趋势——Blackwell 的新精度，算力再翻倍<br>
5. 能手写 **FP8 dequant kernel**，验证 E4M3/E5M2 的精度范围<br>

> 💡 **为什么重要**：FP8 是 2025-2026 推理加速的核心方向，Hopper/Blackwell 原生支持。GPTQ vs AWQ 是面试"量化算法"的高频对比题。Day 1 给了量化概览，今天深化 FP8 和 W4A16 算法对比。

---

### 学前导读：从 INT8 到 FP8

Day 1 的 W8A16 是"权重 INT8 + 激活 FP16"——INT8 是整数，动态范围有限。FP8 是浮点数，用指数+尾数表示，动态范围更大，精度更均匀：

| 格式 | 总位 | 范围 | 精度 | 算力(FP16基准) |
|------|------|------|------|-------------|
| INT8 | 8 | ±127 | 均匀 | — |
| FP8 E4M3 | 8 | ±448 | 浮点 | 2x |
| FP8 E5M2 | 8 | ±57344 | 浮点 | 2x |
| FP4 E2M1 | 4 | ±6 | 浮点 | 4x |

> 💡 **一句话总结**：FP8 = INT8 的带宽 + FP16 的动态范围。Hopper/Blackwell 原生 Tensor Core 支持，算力比 FP16 翻倍。

---

### 理论学习

#### 2.1 FP8 格式详解

##### E4M3 vs E5M2

| 格式 | 符号 | 指数 | 尾数 | 偏置 | 最大值 | 最小正数 | 精度 |
|------|------|------|------|------|--------|---------|------|
| E4M3 | 1 | 4 | 3 | 7 | 448 | 2^-6 | ~1.5% |
| E5M2 | 1 | 5 | 2 | 15 | 57344 | 2^-14 | ~6% |

##### 为什么有两种格式？

- **E4M3**：尾数多（3 位），精度好 → 用于**前向**（权重/激活，精度敏感）
- **E5M2**：指数多（5 位），范围大 → 用于**反向**（梯度，范围大防溢出）
- 混合使用：前向 E4M3 + 反向 E5M2，兼顾精度和范围

##### FP8 vs INT8 的优势

| 维度 | INT8 | FP8 |
|------|------|-----|
| 动态范围 | ±127（固定） | ±448/±57344（浮点） |
| outlier 处理 | 需要 per-channel scale | 浮点自然容纳 outlier |
| 精度分布 | 均匀（线性量化） | 非均匀（小值精度高，大值精度低） |
| Tensor Core | 支持 | 原生支持（Hopper+） |
| 算力 | 同 FP16 | 2x FP16 |

> 💡 **面试要点**：FP8 的浮点格式让它比 INT8 更自然地容纳 outlier（大值用指数表示，不溢出）。这也是为什么 FP8 量化通常不需要复杂的 outlier 处理（如 SmoothQuant），而 INT8 需要。

#### 2.2 FP8 GEMM 与 Scaling Factor

##### 混合精度策略

```
输入: A (FP8 E4M3), B (FP8 E4M3)
累加: FP32 (Tensor Core 原生 FP32 累加)
输出: D (FP16 或 FP32)
```

##### Scaling Factor 的作用

FP8 的动态范围有限（E4M3 ±448），需要 scaling factor 把 FP16 数据映射到 FP8 范围：

```
A_fp8 = A_fp16 / scale_A    (量化)
A_fp16 = A_fp8 * scale_A    (反量化)
```

- per-tensor scale：整个张量一个 scale，简单但精度低
- per-channel scale：每行/列一个 scale，精度好（GEMM 中 scale 可提到点积外）
- per-token scale：每个 token 一个 scale（激活量化常用）

##### FP8 GEMM 伪代码

```cuda
// FP8 GEMM: D = A @ B, A/B 是 FP8, D 是 FP16
// scale_A, scale_B: per-channel scaling factors
__global__ void fp8_gemm_kernel(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    float scale_A, float scale_B,
    __half* D, int M, int N, int K)
{
    // 1. 从 global memory 加载 FP8 tile
    // 2. Tensor Core mma.sync (FP8 输入, FP32 累加)
    //    acc_fp32 += A_fp8 @ B_fp8  (硬件自动转 FP8->FP32 乘加)
    // 3. 反量化: D_fp16 = acc_fp32 * scale_A * scale_B
    // 4. 写回 D
}
```

> ⚠️ Hopper+ 的 Tensor Core 原生支持 FP8 mma.sync，不需要手动反量化到 FP16 再算。Scale 在 epilogue 阶段乘上去。

#### 2.3 GPTQ vs AWQ vs SmoothQuant

##### 三条 W4A16 量化路线

| 路线 | 算法 | 校准数据 | 求解方式 | 部署 |
|------|------|---------|---------|------|
| **GPTQ** | Hessian-based 逐列量化 | 需要（128 样本） | 二阶信息逐列求解 | exllama/AutoGPTQ |
| **AWQ** | Activation-aware 权重裁剪 | 需要（少量） | 搜索保护比例 + scale | AWQ/vLLM |
| **SmoothQuant** | 激活迁移到权重 | 需要 | per-channel scale 迁移 | 部署友好 |

##### GPTQ：Hessian-based 量化

原理：利用 Hessian 矩阵的逆，逐列量化权重，最小化量化误差对输出的影响。

```
1. 计算 H = X^T X (校准数据的二阶信息)
2. 逐列量化: w_q = round(w / s), 误差用 Hessian 逆补偿到剩余列
3. 输出: 量化后的 INT4 权重 + scale
```

- 优点：精度高（二阶信息捕获误差传播）
- 缺点：校准慢（Hessian 计算）、逐列求解串行

##### AWQ：Activation-Aware 量化

原理：不是所有权重都重要——激活大的通道对应的权重量化误差影响更大，应该保护。

```
1. 找到激活大的通道（"salient" channels）
2. 对这些通道的权重做 per-channel scale（放大后量化精度更高）
3. 对应激活做反向 scale（保持数学等价）
```

- 优点：校准快、部署友好（不改模型结构）
- 缺点：精度略低于 GPTQ

##### SmoothQuant：激活迁移

原理：激活有 outlier（大值），权重没有。把激活的 outlier "迁移"到权重，让两者都平滑。

```
1. 找到激活的 outlier 通道
2. 对这些通道: w' = w * s, x' = x / s  (s > 1)
3. 现在激活平滑了，权重稍大但仍可量化
4. 两者都用 INT8 量化
```

- 优点：W8A8（权重和激活都 INT8）、部署极友好
- 缺点：只做到 INT8，不如 W4A16 省显存

##### 对比总结

| 维度 | GPTQ | AWQ | SmoothQuant |
|------|------|-----|------------|
| 精度 (W4A16) | 最高 | 高 | — (W8A8) |
| 校准速度 | 慢 | 快 | 快 |
| 部署难度 | 中 | 低 | 最低 |
| 显存节省 | 4x (W4) | 4x (W4) | 2x (W8) |
| 推理加速 | W4A16 dequant GEMM | W4A16 dequant GEMM | W8A8 Tensor Core |

> 💡 **面试一句话**：GPTQ 精度最高但慢，AWQ 平衡（vLLM 默认），SmoothQuant 部署最友好（W8A8）。FP8 是新趋势，比 INT8 精度好 + 算力 2x。

#### 2.4 FP4 量化趋势

##### Blackwell 的 FP4

Blackwell（sm_120）支持 FP4（E2M1）：
- 4 位浮点：1 符号 + 2 指数 + 1 尾数
- 算力：FP16 的 4x（RTX 5090 FP4 理论 ~836 TFLOPS）
- 精度：极低（动态范围 ±6，与 Day 1 §1.4 NVFP4 表一致），需要精细的 scaling factor

##### FP4 的挑战

1. **精度损失大**：只有 1 位尾数，需要 per-block scaling + micro-scaling
2. **校准复杂**：需要更精细的 scaling factor 设计
3. **适用场景**：推理（容忍精度损失），训练需谨慎

---

### Coding 任务

#### 任务 1：FP8 dequant kernel

```cuda
#include <cuda_fp8.h>

__global__ void fp8_dequant_kernel(
    const __nv_fp8_e4m3* input,
    float* output,
    float scale,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = (float)input[idx] * scale;
    }
}

// 对比 E4M3 vs E5M2 的精度和范围
void test_fp8_formats() {
    // 构造 [-1000, 1000] 的数据, 分别量化为 E4M3 和 E5M2
    // 对比 max_diff 和有效位数
}
```

#### 任务 2：GPTQ vs AWQ 精度对比（PyTorch 模拟）

```python
def simulate_gptq(weight, calibration_data):
    """简化版 GPTQ: Hessian-based 逐列量化"""
    H = calibration_data.T @ calibration_data  # Hessian
    H_inv = torch.inverse(H + 1e-5 * torch.eye(H.shape[0]))
    w_q = torch.zeros_like(weight)
    for col in range(weight.shape[1]):
        w_q[:, col] = quantize_int4(weight[:, col])
        # 用 Hessian 逆补偿误差到剩余列
        error = weight[:, col] - dequantize(w_q[:, col])
        weight[:, col+1:] -= error.unsqueeze(1) * H_inv[col, col+1:]
    return w_q

def simulate_awq(weight, activation):
    """简化版 AWQ: activation-aware scale"""
    # 找激活大的通道
    act_scale = activation.abs().mean(dim=0)
    salient = act_scale > act_scale.median()
    # 对 salient 通道放大后量化
    scale = torch.ones(weight.shape[0])
    scale[salient] = 2.0  # 放大
    w_scaled = weight * scale.unsqueeze(1)
    w_q = quantize_int4(w_scaled)
    return w_q, scale

# 对比精度
for model_size in ['7B', '13B']:
    w_gptq = simulate_gptq(weight, calib)
    w_awq, scale_awq = simulate_awq(weight, act)
    diff_gptq = (dequantize(w_gptq) - weight).abs().mean()
    diff_awq = (dequantize(w_awq) * scale_awq.unsqueeze(1) - weight).abs().mean()
    print(f"{model_size}: GPTQ error={diff_gptq:.4f}, AWQ error={diff_awq:.4f}")
```

#### 任务 2b：FP8 GEMM 实测 benchmark（`torch._scaled_mm`）

创建 [kernels/fp8_gemm_benchmark.py](https://github.com/hzchenxiaobin/ai-infra-notes/blob/main/aiinfra/daily/week8/day2/kernels/fp8_gemm_benchmark.py)，用 `torch._scaled_mm` 做 FP8 E4M3 GEMM，对比 FP16/BF16 的性能与精度：

```bash
python kernels/fp8_gemm_benchmark.py
```

**实测输出**（RTX 5090, CUDA 12.8, PyTorch 2.9.1+cu128, 2026-08-06）：

| M×N×K | FP16 (ms) | FP16 (TF) | FP8 (ms) | FP8 (TF) | FP8/FP16 加速 | 最大误差 |
|--------|----------|----------|---------|---------|-------------|---------|
| 1024³ | 0.015 | 144.5 | 0.014 | 149.0 | 1.03x | 247.19 |
| 2048³ | 0.097 | 177.0 | 0.051 | 339.7 | 1.92x | 345.12 |
| 4096³ | 0.627 | 219.1 | 0.332 | 413.5 | 1.89x | 523.50 |
| 8192³ | 4.918 | 223.6 | 2.447 | 449.4 | 2.01x | 746.00 |

> 实测说明：上表由 `fp8_gemm_benchmark.py` 在 RTX 5090 上实跑得到。小矩阵（1024³）受 launch overhead 主导，加速比仅 1.03x；大矩阵（8192³）接近 2x 理论值。FP8 量化误差随矩阵规模增大而增大（累加次数增多）。

**代码要点**：
- `torch._scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype=torch.float16)` 调用 FP8 Tensor Core
- FP8 E4M3 → FP32 累加 → FP16 输出（混合精度策略）
- `scale_a = scale_b = 1.0` 是 per-tensor 最简方案；生产用 per-block（MXFP8）

#### 任务 3：LeetCode 面试题（10 周计划 · 第 8 周 Day 2）

> 📅 今日题目来自 [10 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/10-week-plan.html) 第 8 周「二分查找与动态规划基础」Day 2（旋转数组与峰值），共 5 题。简单题快速过、中等题精做、困难题吃透；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/) | 中等 | 旋转数组二分（比右端点） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/153_寻找旋转排序数组中的最小值.html) |
| [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/) | 中等 | 旋转数组二分（判断哪半有序） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/33_搜索旋转排序数组.html) |
| [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/) | 中等 | 两次二分找左/右边界 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/34_在排序数组中查找元素的第一个和最后一个位置.html) |
| [162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/) | 中等 | 非有序二分（爬坡法，顺梯度走） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/162_寻找峰值.html) |
| [540. 有序数组中的单一元素](https://leetcode.cn/problems/single-element-in-a-sorted-array/) | 中等 | 二分（奇偶下标配对） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/540_有序数组中的单一元素.html) |

---

### 扩展实验

#### 实验 1：FP8 vs INT8 精度对比

构造含 outlier 的数据，分别用 FP8 E4M3 和 INT8 (per-channel scale) 量化，对比：
- FP8 是否更自然地容纳 outlier？
- 精度差异如何？

#### 实验 2：Scaling Factor 粒度对比

对比 per-tensor / per-channel / per-token scale 的精度：
- per-tensor 最差（一个 scale 覆盖所有 outlier）
- per-channel 最好（GEMM 中 scale 可提外）

---

### 今日总结

1. **E4M3 vs E5M2**：E4M3 精度好用于前向，E5M2 范围大用于反向
2. **FP8 vs INT8**：FP8 浮点格式自然容纳 outlier，不需要复杂 outlier 处理，算力 2x FP16
3. **GPTQ**：Hessian-based，精度最高，校准慢
4. **AWQ**：Activation-aware，平衡精度与速度，vLLM 默认
5. **SmoothQuant**：激活迁移，W8A8，部署最友好
6. **FP4**：Blackwell 新精度，算力 4x FP16，需精细 scaling

---

### 面试要点

1. **FP8 的 E4M3 和 E5M2 分别用于什么？为什么有两种格式？**

   <details>
   <summary>答案</summary>

   - E4M3（4 指数 + 3 尾数）：精度好，范围 ±448，用于**前向**（权重/激活精度敏感）
   - E5M2（5 指数 + 2 尾数）：范围大 ±57344，用于**反向**（梯度范围大防溢出）
   - 混合使用：前向 E4M3 + 反向 E5M2

   </details>

2. **FP8 比 INT8 有什么优势？**

   <details>
   <summary>答案</summary>

   - 浮点格式自然容纳 outlier（大值用指数表示）
   - 不需要复杂 outlier 处理（如 SmoothQuant）
   - Hopper+ Tensor Core 原生支持，算力 2x FP16
   - 精度分布非均匀（小值精度高），更适合神经网络数据的分布

   </details>

3. **GPTQ 和 AWQ 的区别是什么？**

   <details>
   <summary>答案</summary>

   - GPTQ：Hessian-based 逐列量化，二阶信息补偿误差，精度最高，校准慢
   - AWQ：activation-aware，保护激活大的通道，校准快，部署友好
   - vLLM 默认用 AWQ（平衡），追求精度用 GPTQ

   </details>

4. **SmoothQuant 的原理是什么？**

   <details>
   <summary>答案</summary>

   - 激活有 outlier，权重没有。把激活的 outlier "迁移"到权重
   - `w' = w * s, x' = x / s`（s > 1 对 outlier 通道）
   - 迁移后激活平滑了，权重稍大但仍可 INT8 量化
   - 实现 W8A8（权重和激活都 INT8），部署极友好

   </details>

5. **FP8 GEMM 的 scaling factor 作用是什么？粒度怎么选？**

   <details>
   <summary>答案</summary>

   - 作用：FP8 范围有限（E4M3 ±448），需 scale 把 FP16 数据映射到 FP8
   - `A_fp8 = A_fp16 / scale; D_fp16 = (A_fp8 @ B_fp8) * scale_A * scale_B`
   - 粒度：per-tensor（简单精度低）/ per-channel（精度好，GEMM 中 scale 可提外）/ per-token（激活常用）
   - Scale 在 epilogue 阶段乘，不影响 Tensor Core 的 FP8 mma

   </details>
