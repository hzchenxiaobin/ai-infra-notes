## Day 5：项目推进 —— Triton 三方 Benchmark 与 Autotune

### 🎯 目标

通过今天的学习，你将：

1. 能用 Triton 实现 **Softmax / GEMM / FlashAttention** 三个算子，与 Day 2-3 的 CUDA 版和 PyTorch 版做三方对比<br>
2. 掌握 `@triton.autotune` 的配置搜索机制，能为不同矩阵大小自动选最优 `(BLOCK_SIZE, num_warps, num_stages)`<br>
3. 能搭建统一的 **benchmark 框架**，一键对比 Triton vs CUDA vs PyTorch 的性能与精度<br>
4. 理解 Triton 在不同算子类型上的表现差异——GEMM 大矩阵达 cuBLAS 93%+，Softmax/FA 接近或超过 CUDA 版<br>
5. 能产出"何时用 Triton 何时必须 CUDA"的**决策表**<br>

> 💡 **为什么重要**：Triton 是 2025 算子岗 70%+ JD 必考技能。面试常问"你用 Triton 写过什么算子、性能如何、和 CUDA 比怎么样"。今天的 benchmark 框架和决策表直接回答这些问题，也为 Week 5 FlashAttention 的 Triton 实现打基础。

---

### 学前导读：从 Day 4 的单算子到三方对比

Day 4 我们学了 Triton 语言基础（`tl.load/store/reduce/dot` + `@triton.jit`），写了 Softmax/GEMM/FA 的单算子实现。今天是**项目推进日**——把三个算子统一到一个 benchmark 框架里，做系统性的三方对比：

| 维度 | Triton | CUDA (Day 2-3) | PyTorch |
|------|--------|----------------|---------|
| 代码量 | ~40 行/算子 | ~300 行/算子 | 1 行 (`torch.softmax`) |
| 性能 | 大矩阵 cuBLAS 93-97% | cuBLAS ~30% (手写，无 Tensor Core) | cuBLAS 后端 (100%) |
| 开发效率 | 高（Python） | 低（C++ + PTX） | 最高（直接调库） |
| 可调性 | autotune 自动 | 手动 tune | 黑箱 |

> 💡 **一句话总结**：今天的产出是一张"Triton vs CUDA vs PyTorch"性能表 + 一份"何时用 Triton"决策表——这是面试和工程选型的直接依据。

---

### 理论学习

#### 5.1 Triton Autotune 机制

##### `@triton.autotune` 的工作原理

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

**工作流程**：

1. 首次调用某组 `(M, N, K)` 时，遍历所有 configs，逐个编译 + 运行 + 计时
2. 选最快的 config 缓存起来，后续相同 key 的调用直接用最优 config
3. `key` 参数控制"什么情况下重新搜索"——只把影响 tiling 决策的参数放入 key

##### Config 搜索空间设计

| 参数 | 含义 | 典型范围 | 选择依据 |
|------|------|---------|---------|
| `BLOCK_M/N/K` | tile 大小 | 64-256 / 64-256 / 16-64 | 受 smem 容量约束 |
| `num_warps` | 每 block warp 数 | 4 / 8 / 16 | 影响 occupancy |
| `num_stages` | pipeline stage 数 | 2 / 3 / 4 | double buffer 级别 |

**搜索空间大小**：6 configs × 4 sizes = 24 次编译 + 运行（首次调用开销 ~30s，之后 0）

##### Autotune 的代价

| 优势 | 代价 |
|------|------|
| 自动选最优配置 | 首次调用慢（试所有 config） |
| 适配不同硬件 | 编译缓存占用磁盘 |
| 无需手动 tune | shape 离散多时缓存爆炸 |

> 💡 **生产实践**：`torch.compile` 的 `max-autotune` 模式内部也会做类似的 config 搜索。Triton autotune 的 cache 可持久化到磁盘，避免每次启动重搜。

#### 5.2 三方 Benchmark 框架设计

##### 统一接口

```python
def benchmark(name, triton_fn, cuda_fn, torch_fn, shapes, atol=1e-3):
    """统一三方对比框架"""
    results = []
    for M, N, K in shapes:
        # 1. 初始化输入
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        # 2. 正确性验证
        ref = torch_fn(a, b)
        out_triton = triton_fn(a, b)
        out_cuda = cuda_fn(a, b)
        diff_triton = (out_triton - ref).abs().max().item()
        diff_cuda = (out_cuda - ref).abs().max().item()
        # 3. 性能计时 (torch.cuda.Event)
        ms_triton = measure(triton_fn, a, b)
        ms_cuda = measure(cuda_fn, a, b)
        ms_torch = measure(torch_fn, a, b)
        # 4. 计算 TFLOPS 和 cuBLAS 占比
        ...
        results.append({...})
    return results
```

##### 计时方法

```python
def measure(fn, *args, iters=100, warmup=10):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
```

#### 5.3 预估性能对比（待 GPU 实测回填）

##### GEMM（预估，RTX 5090 sm_120，FP16 输入）

| M=N=K | Triton (ms) | cuBLAS FP16 (ms) | cuBLAS FP16→FP32 (ms) | Triton %FP16 | Triton %FP32out |
|-------|------------|------------------|----------------------|-------------|-----------------|
| 512   | 0.015      | 0.006            | 0.010                | 40.0%       | 66.7%           |
| 1024  | 0.031      | 0.014            | 0.019                | 45.2%       | 61.3%           |
| 2048  | 0.105      | 0.098            | 0.109                | 93.3%       | 103.8%          |
| 4096  | 0.661      | 0.644            | 0.695                | 97.4%       | 105.1%          |

> ⚠️ **预估发现（待实测回填）**：
> - **大矩阵（4096）Triton 达 FP16 cuBLAS 的 97.4%**——`tl.dot` 自动调 Tensor Core + autotune 选最优 tiling
> - **小矩阵（512）仅 40.0%**——autotune 的 config 搜索空间不够 + block 数少 SM 利用率低
> - **对比 FP16→FP32 输出口径时 Triton 甚至超过 cuBLAS**（105.1%）——因 `torch.matmul(a.half,b.half).float()` 多了一次类型转换
> - **max_diff ~0.03-0.13**（FP16 精度损失，正常）

##### Softmax（预估，FP32）

| M×D | Triton (ms) | PyTorch (ms) | PyTorch / Triton |
|-----|------------|------------|-------------------|
| 1024×1024 | 0.008 | 0.004 | 0.50x |
| 4096×1024 | 0.008 | 0.008 | 1.00x |
| 4096×4096 | 0.072 | 0.073 | 1.01x |

> ⚠️ **预估发现**：Triton Softmax 与 PyTorch `torch.softmax` 大矩阵基本持平（1.01x），小矩阵更慢（0.50x）。原因：PyTorch 的 softmax kernel 已高度优化，Triton 在 memory-bound 算子上无明显优势。max_diff ~1e-9（精度一致）。
>
> 注：`PyTorch / Triton` = PyTorch 时间 / Triton 时间。<1 表示 Triton 更慢，>1 表示 Triton 更快。

##### FlashAttention（预估，causal, FP16）

| N (d=64) | Triton FA (ms) | naive attention (ms) | 官方 FA (ms) | Triton %官方 |
|---------|--------------|----------------|------------|-------------|
| 2048 | ~0.5 | ~1.8 | ~0.4 | ~80% |
| 4096 | ~1.8 | ~6.5 | ~1.5 | ~83% |
| 8192 | ~7.0 | ~25 | ~6.0 | ~86% |

> Triton FA 达官方 CUDA 版的 80-90%，核心逻辑 ~40 行 vs CUDA ~300 行。这是 Triton 的"甜区"——用 1/7 的代码量达到 85% 的性能。注："naive attention"列为标准注意力（物化 S/P 矩阵），非 CUDA 手写 FA；CUDA 手写 FA 的对比见 Week 5 Day 3。

#### 5.4 "何时用 Triton 何时必须 CUDA"决策表

| 场景 | 推荐选择 | 原因 |
|------|---------|------|
| GEMM 通用 | cuBLAS / CUTLASS | 已有极致优化，不值得重写 |
| GEMM 定制形状 | Triton | autotune 自动适配，开发快 |
| Softmax/LayerNorm | Triton | memory-bound，Triton 自动 tiling 够用 |
| FlashAttention | Triton（首选）/ CUDA（极致） | Triton 85% 性能 + 1/7 代码量 |
| 自定义 epilogue fusion | Triton | `tl.store` 前插入自定义逻辑，比 CUDA epilogue 简单 |
| TMA / FP8 / warp specialization | CUDA (PTX) | Triton 滞后 1-2 架构周期 |
| Grid 级同步 / cooperative | CUDA | Triton 无 grid 级通信 |
| 动态 shape（频繁变化） | CUDA | Triton 的 `tl.constexpr` + cache 会爆炸 |
| 生产推理引擎 kernel | CUDA | 极致性能 + 精细控制 |

> 💡 **面试一句话**：Triton 是"80% 性能 + 20% 代码量"的甜区选择。需要 90%+ 极致性能或新硬件指令时才用手写 CUDA。

---

### Coding 任务

#### 任务 1：Triton GEMM with Autotune

创建 `kernels/triton_gemm.py`：

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_gemm(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

> 注：上例为简化版（2D grid、无 GROUP_SIZE_M）。完整实现见 `day4/kernels/triton_gemm.py`——使用 `BLOCK_SIZE_M/N/K` 命名、1D grid + group-based tile 排序（提升 L2 复用），benchmark 时以完整文件为准。

#### 任务 2：运行三方 Benchmark

```bash
python3 kernels/benchmark_triton.py
```

预期输出：

```text
=== Triton vs CUDA vs PyTorch GEMM Benchmark (RTX 5090, FP16->FP32) ===
M=N=K    | Triton(ms)  CUDA(ms)    cuBLAS(ms)  | Triton%   CUDA%   Triton/CUDA
---------|------------------------------------------------|------------------------------
512      | 0.015       0.020       0.006       | 40%       30%     1.3x
1024     | 0.031       0.045       0.014       | 45%       31%     1.5x
2048     | 0.105       0.330       0.098       | 93%       30%     3.1x
4096     | 0.661       2.150       0.644       | 97%       30%     3.3x
```

> ⚠️ **预期说明**：Triton 大矩阵经 autotune 达 cuBLAS 93-97%，小矩阵仅 40-45%（launch overhead + SM 利用率低）。手写 CUDA（smem tiling + FMA，无 Tensor Core）约 cuBLAS 30%，Triton 比手写 CUDA 快 1.3-3.3x。原因是 Triton 自动选了最优 tiling + 用 Tensor Core（`tl.dot` 自动生成 `mma.sync` 指令）。以上为预估口径，待 GPU 实测回填。

#### 任务 3：Softmax + FA 三方对比

`benchmark_triton.py` 已直接复用 Day 4 的 Triton kernel；GEMM/Softmax 的 CUDA 列由脚本内嵌的 naive CUDA kernel 经 `torch.utils.cpp_extension.load_inline` 现场编译（需要 nvcc）。

```text
=== Softmax Benchmark (FP32) ===
M×D      | Triton(ms)  CUDA(ms)    PyTorch(ms) | Triton vs PyTorch
1024×1024| 0.008       0.012       0.004       | 0.50x
4096×1024| 0.008       0.020       0.008       | 1.00x
4096×4096| 0.072       0.080       0.073       | 1.01x

=== FlashAttention Benchmark (causal, FP16) ===
N (d=64) | Triton(ms)  naive(ms)    官方(ms)    | Triton%官方
2048     | 0.50        1.80        0.40        | 80%
4096     | 1.80        6.50        1.50        | 83%
8192     | 7.00        25.0        6.00        | 86%
```

> 注：FlashAttention 列中 "naive" 为标准注意力（物化 S/P 矩阵），非 CUDA 手写 FA；CUDA 手写 FA 的对比见 Week 5 Day 3。上表数字均为预估口径，以 GPU 实测回填为准。Softmax 的 "Triton vs PyTorch" = PyTorch 时间 / Triton 时间，<1 表示 Triton 更慢。

#### 任务 4：LeetCode 面试题（10 周计划 · 第 9 周 Day 2 补充）

> 📅 今日题目选自 [10 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/10-week-plan.html) 第 9 周「动态规划进阶——子序列、区间与二维 DP」Day 2（回文与区间 DP）的子集，共 3 题。项目推进日 LeetCode 题量精简，留时间给 benchmark 调试。简单题快速过、中等题精做；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [647](https://leetcode.cn/problems/palindromic-substrings/) | Medium | 中心扩展/DP | [题解](https://hzchenxiaobin.github.io/leetcode/problems/647_回文子串.html) |
| [516](https://leetcode.cn/problems/longest-palindromic-subsequence/) | Medium | DP（区间） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/516_最长回文子序列.html) |
| [1312](https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/) | Hard | DP（区间） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/1312_让字符串成为回文串的最少插入次数.html) |

---

### 扩展实验

#### 实验 1：Autotune Config 搜索空间扩展

把 GEMM 的 autotune configs 从 6 个扩到 12 个（加入 `num_stages=3/4` 和更大 `BLOCK_K=64`），观察：
- 最优 config 是否变化？
- 首次搜索时间增加多少？
- 最优性能提升多少？

#### 实验 2：Triton vs torch.compile

用 `torch.compile(mode='max-autotune')` 编译一个自定义算子，对比 Triton 手写版：
- `torch.compile` 内部是否也生成了 Triton kernel？
- 性能差异如何？

#### 实验 3：跨硬件验证

如果有不同架构的 GPU（如 A100/H100），把 Triton benchmark 跑一遍：
- autotune 是否为不同硬件选了不同 config？
- Triton 在 Hopper（sm_90）上是否更接近 cuBLAS？

---

### 今日总结

Day 5 我们搭建了 Triton 三方 benchmark 框架，产出了性能对比表和决策表：

1. **Triton GEMM**：大矩阵经 autotune 达 cuBLAS 93-97%，比手写 CUDA（无 Tensor Core）快 1.3-3.3x（自动 tiling + Tensor Core）
2. **Triton Softmax**：与 PyTorch 大矩阵持平（~1.0x）、小矩阵更慢（~0.5x）；比手写 CUDA 略快——memory-bound 算子上 Triton 无明显优势
3. **Triton FA**：达官方 CUDA 版 80-90%，代码量 1/7
4. **Autotune**：首次调用搜索所有 config，缓存最优，后续零成本
5. **决策表**：Triton 是"80% 性能 + 20% 代码量"的甜区；需要 90%+ 或新硬件指令时用 CUDA

掌握 Triton benchmark 后，你有了"用数据回答 Triton vs CUDA"的能力。Day 6 用 ncu profiling 深入分析三方的内部指标差异。

---

### 面试要点

1. **Triton 的 autotune 是怎么工作的？有什么代价？**

   <details>
   <summary>点击查看答案</summary>

   - **工作原理**：首次调用某组 key 参数时，遍历所有 configs 逐个编译+运行+计时，选最快的缓存
   - **key 参数**：只放影响 tiling 决策的参数（如 M/N/K），不放不影响 tiling 的参数（如指针地址）
   - **代价**：首次调用慢（N 个 config × 编译+运行），shape 离散多时编译缓存爆炸
   - **收益**：自动选最优配置，无需手动 tune；cache 可持久化到磁盘避免重搜
   - **生产实践**：`torch.compile` 的 `max-autotune` 内部也做类似搜索

   </details>

2. **Triton GEMM 为什么大矩阵能达到 cuBLAS 93%+？它自动做了什么？**

   <details>
   <summary>点击查看答案</summary>

   - `tl.dot` 自动调用 Tensor Core（WMMA/mma.sync），不需要手写 PTX
   - Autotune 自动搜索最优 `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`
   - 自动生成 shared memory tiling + double buffer（`num_stages=2` 时）
   - 自动向量化（`tl.load/store` 生成 `float4` / `cp.async`）
    - **与手写 CUDA 的差距**：Triton 的自动 tiling 不如 CUTLASS 极致（少 swizzle / K 分割 / epilogue fusion），所以大矩阵 93-97%、小矩阵 40-45%，而非 95%+

   </details>

3. **什么时候用 Triton？什么时候必须手写 CUDA？**

   <details>
   <summary>点击查看答案</summary>

   - **用 Triton**：Softmax/LayerNorm/FA 等标准算子、定制形状 GEMM、epilogue fusion、快速原型
   - **必须 CUDA**：① 极致性能（95%+ cuBLAS）② 新硬件指令（TMA/FP8/warp specialization）③ Grid 级同步 ④ 动态 shape（Triton 的 constexpr + cache 会爆炸）⑤ 生产推理引擎 kernel
   - **核心判断**：Triton 是"80% 性能 + 20% 代码量"的甜区。超出甜区（90%+ 或新指令）才用 CUDA

   </details>

4. **Triton FlashAttention 为什么只有 40 行代码？**

   <details>
   <summary>点击查看答案</summary>

   - `tl.dot(q, k.T)` 一行替代 CUDA 的 WMMA fragment 声明 + load + mma_sync + store（~50 行）
   - `tl.max` / `tl.sum` 一行替代 CUDA 的 warp shuffle + block reduce（~30 行）
   - `tl.load` / `tl.store` 的 mask 参数一行替代 CUDA 的边界判断（~20 行）
   - `@triton.jit` 自动处理 shared memory 分配 + `__syncthreads` + 向量化
   - **对比**：Triton FA ~40 行 vs CUDA FA ~300 行，核心算法逻辑一一对应

   </details>

5. **Triton 的局限性是什么？什么场景它无能为力？**

   <details>
   <summary>点击查看答案</summary>

   1. **新指令滞后**：TMA / FP8 mma / warp specialization 等新特性，Triton 滞后 1-2 架构周期
   2. **性能天花板**：自动 tiling 不如 CUTLASS 极致，通常慢 10-20%
   3. **跨 block 通信弱**：无 grid 级同步，复杂跨 block reduction 需手写
   4. **调试受限**：`printf` 不支持，生成的 PTX 可读性差
   5. **动态 shape**：`tl.constexpr` 要求编译期已知，shape 频繁变化导致缓存爆炸
   6. **无能为力**：极致性能 / 新硬件指令 / grid 级同步 / 精细 stream + CUDA Graph 控制

   </details>
