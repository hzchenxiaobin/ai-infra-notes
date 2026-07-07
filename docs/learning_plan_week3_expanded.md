# 第3周深度展开：Transformer 执行本质与算子手写（7天）

> **适用对象**：陈斌斌（已完成第2周学习，掌握 Warp Shuffle、Register Blocking GEMM cuBLAS 70%+、Multi-Stream、Nsight 性能分析、FlashAttention 简化版 Forward Kernel）
> **本周目标**：从 GPU 视角理解 Transformer 推理执行流程，手写 Softmax/LayerNorm/标准 Attention Kernel，完成算子 IO 分析与端到端 Profiling
> **时间投入**：工作日每天2.5h（早间1.5h + 晚间1h），周末每天6h，周计24.5h
> **周日里程碑**：手写 Softmax + LayerNorm Kernel 误差 < 1e-5，完成标准 Attention IO 量化分析，产出 Transformer 算子计算强度分类表

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 理解 Transformer 推理的 Prefill/Decode 两阶段执行特征，手写 Softmax/LayerNorm/标准 Attention 三个 memory-bound 算子，用 arithmetic intensity 对算子分类 |
| **核心产出** | ① Transformer forward 的 torch.profiler 时间线 ② Softmax Kernel（warp shuffle reduce）③ LayerNorm Kernel（两级 reduce）④ 标准 Attention Forward Kernel（含 IO 量化）⑤ 端到端 Profiling 报告 ⑥ Transformer 算子分类表 |
| **验收标准** | ① Softmax/LayerNorm 与 PyTorch 误差 < 1e-5 ② 手动计算的标准 Attention HBM 读写量与 ncu 实测一致（误差 < 15%）③ 能画出 Transformer forward 算子时间线 ④ 能用 arithmetic intensity 判断每个算子是 compute-bound 还是 memory-bound |
| **面试准备** | 积累10-12道进阶面试题，覆盖 Prefill/Decode、Softmax 数值稳定性、LayerNorm 并行化、Attention IO 复杂度、算子分类五大主题 |

### 本周知识图谱

```
Day 15: Transformer 推理流程 → Prefill vs Decode + torch.profiler 时间线
  ↓
Day 16: Softmax + LayerNorm Kernel → safe softmax + 两级 reduce + warp shuffle
  ↓
Day 17: 源码分析 → PyTorch ATen / FasterTransformer 的优化手法
  ↓
Day 18: Attention IO 分析 → 标准 Attention HBM 读写量 + O(N²) 量化
  ↓
Day 19: 项目推进 → 算子接入 Mini 引擎 + 端到端正确性
  ↓
Day 20: 端到端 Profiling → 定位 memory-bound 算子 + fusion 机会
  ↓
Day 21: 算子分类 → arithmetic intensity 分类表 + 优化方向总结
```

### 前置准备清单

#### 硬件/软件验证
- [ ] 已完成第2周所有 Coding 任务（GEMM cuBLAS 70%+、FlashAttention 简化版 PASS）
- [ ] `ncu --version` 正常（Week 2 Day 11 已验证）
- [ ] PyTorch 可用：`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [ ] `torch.__version__ >= 2.0`（torch.profiler API 稳定）
- [ ] nvcc 可编译 Week 2 的 warp_reduce.cu（确认 warp shuffle 基础代码可用）

#### 验证命令
```bash
# 验证 PyTorch + CUDA
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
# 预期输出：torch 2.x.x cuda 12.x available True

# 验证 nsys 可用（端到端 profiling 需要）
nsys --version
# 预期输出：NVIDIA Nsight Systems version 202x.x.x

# 验证第2周代码可编译（复用 warp reduce 基础）
nvcc -o /tmp/warp_reduce week2/day1/kernels/warp_reduce.cu -O3 -arch=sm_80 2>&1 | head
# 预期：无报错（若路径不同请按实际调整）
```

---

## Day 15（周一）：Trace Transformer 推理流程

> **今日目标**：理解 Transformer 推理的 Prefill/Decode 两阶段执行特征，用 torch.profiler 画出一次 forward 的算子时间线，识别耗时最长的算子。
> **时间分配**：早间1.5h（理论学习1h + 昇腾对照30min）+ 晚间1h（编程实践）
> **面试考察度**：⭐⭐⭐ 高频，"Transformer 推理的两个阶段有什么区别"是推理系统入门必考题

---

### 学习任务1：Transformer 推理的两阶段（45分钟）

#### 阅读内容
- **论文/博客**：vLLM 博客 "How vLLM serves LLM" 中 Prefill vs Decode 章节
- **补充阅读**：CUDA C Programming Guide 不需要，重点是模型执行流程而非 CUDA 细节
- **具体阅读重点**：
  - Prefill 阶段：并行处理整条 prompt，计算密集（大量 GEMM）
  - Decode 阶段：自回归逐 token 生成，访存密集（小 GEMM + KV Cache 读取）
  - 两阶段的算子形状差异如何导致性能特征完全不同

#### 核心概念笔记

**1. Prefill vs Decode 执行特征对比**

| 维度 | Prefill 阶段 | Decode 阶段 |
|------|-------------|-------------|
| **输入形状** | `(B, N_prompt, d)`，N_prompt 可达数千 | `(B, 1, d)`，每次只处理 1 个 token |
| **Attention 矩阵** | N×N 完整矩阵 | 1×N（单 query 对所有历史 key） |
| **计算量** | 大（GEMM 是 M×N×K 的大矩阵乘） | 小（GEMM 退化为向量×矩阵） |
| **瓶颈类型** | 通常是 **Compute-bound**（GEMM 主导） | 通常是 **Memory-bound**（KV Cache 读取主导） |
| **GPU 利用率** | 高（SM 充分利用） | 低（大量 SM 空闲，等显存） |
| **典型优化** | Tensor Core、FlashAttention | KV Cache、PagedAttention、CUDA Graph |

**2. Transformer 单层的数据流**

```
Input x (B, N, d)
   │
   ├─► LayerNorm1 ──► QKV Linear (GEMM) ──► Q, K, V (B, N, d)
   │                                      │
   │                                      ├─► Attention: S=QK^T → softmax → PV
   │                                      │    （Prefill: N×N 矩阵；Decode: 1×N）
   │                                      └─► KV Cache append（Decode 阶段）
   │
   ├─► Output Linear (GEMM) ──► residual add ──► LayerNorm2
   │
   └─► FFN: Linear(GELU(Linear(x))) （两个大 GEMM）──► residual add
```

**关键观察**：Transformer 单层包含 6 个主要算子类型：
1. **LayerNorm**（2 次）：element-wise + reduction，memory-bound
2. **QKV/Output/FFN Linear**（4 个 GEMM）：compute-bound（Prefill）或 memory-bound（Decode）
3. **Attention**：S=QK^T（GEMM）+ softmax（memory-bound）+ PV（GEMM）

**3. 算子执行顺序与依赖**

```
LayerNorm1 → QKV GEMM → Attention(QK^T → Softmax → PV) → Out GEMM → Residual →
LayerNorm2 → FFN GEMM1 → GELU → FFN GEMM2 → Residual → 下一层
```

> 💡 **为什么重要**：理解算子顺序是后续 kernel fusion 的基础。例如 LayerNorm + QKV GEMM 可以融合成单个 kernel，省去中间结果写回 HBM。

#### 昇腾对照

| CUDA/PyTorch 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| Prefill 阶段（大 GEMM） | Prefill 阶段（Cube Core 大矩阵乘） | 两阶段划分是模型层面的，与硬件无关，昇腾推理框架同样区分 |
| Decode 阶段（小 GEMM + KV Cache） | Decode 阶段（Vector + Cache 读取） | 昇腾 Decode 阶段 Vector Unit 利用率低，与 CUDA SM 空闲同理 |
| torch.profiler 时间线 | msprof timeline | 两者都提供算子级时间线，昇腾用 msprof 采集 |
| `F.softmax` / `F.layer_norm` | Ascend C 内置 Softmax/LayerNorm 算子 | PyTorch 调 CUDA kernel；CANN 调 NPU 算子，语义一致 |
| Attention（QK^T + softmax + PV） | FlashAttention 算子（CANN 内置） | 昇腾 CANN 已内置 FlashAttention，开发者无需手写 |

---

### 学习任务2：torch.profiler 使用方法（30分钟）

#### 核心 API

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,   # 采集 CPU 端调度
        torch.profiler.ProfilerActivity.CUDA,  # 采集 GPU 端 kernel
    ],
) as prof:
    for _ in range(5):
        out = model(x)

# 按 CUDA 时间排序，输出 top 算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# 导出 Chrome trace（可用 chrome://tracing 打开）
prof.export_chrome_trace("transformer_trace.json")
```

#### 关键指标解读

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `Self CUDA` | 该算子自身的 GPU 执行时间（不含子算子） | 排序依据，找 top3 |
| `Self CPU` | 该算子的 CPU 调度时间 | 判断 launch overhead |
| `CPU Mem` | CPU 端内存分配 | 判断是否有频繁分配 |
| `# Calls` | 调用次数 | 判断是否过度 launch |

---

### 学习任务3：昇腾 Profiler 对照（15分钟）

| 维度 | PyTorch profiler | 昇腾 msprof |
|------|-----------------|------------|
| 采集命令 | `torch.profiler.profile` | `msprof --application="python train.py" --output=./prof_data` |
| 输出格式 | Chrome trace JSON | .npu 离线文件 + timeline |
| 算子级视图 | `key_averages().table()` | Operator 详情面板 |
| GPU 利用率 | 需配合 nsys 看 SM 占用 | msprof 直接给出 AI Core 利用率 |
| 适用场景 | PyTorch 模型快速 profile | NPU 模型部署后 profile |

**关键发现**：PyTorch profiler 关注算子级时间分解；若要看 SM/Memory 利用率，仍需配合 nsys/ncu。昇腾 msprof 把两层信息合并到一个工具中。

---

### 晚间编程任务：Trace Transformer Forward（1小时）

#### 完整代码

```python
# trace_transformer.py —— 最小 Transformer Block + Prefill/Decode profiling
# 运行命令: python trace_transformer.py
# 依赖: pip install torch

import torch
import torch.nn as nn
import math


class MiniAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x)                                    # GEMM: B*N*d x d*3d
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                     # 3, B, n_heads, N, d_head
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # GEMM: Q x K^T -> N x N
        attn = torch.softmax(attn, dim=-1)                   # softmax（memory-bound）
        out = torch.matmul(attn, v)                          # GEMM: attn x V -> N x d_head
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)                                 # GEMM: Output Linear


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MiniAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # Attention + residual
        x = x + self.ffn(self.norm2(x))     # FFN + residual
        return x


def profile_phase(model, x, name, n_iter=5):
    """对一个阶段做 profiling 并输出 top 算子"""
    # warmup
    for _ in range(2):
        _ = model(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(n_iter):
            _ = model(x)
        torch.cuda.synchronize()

    print(f"\n===== {name} Phase (shape={tuple(x.shape)}) =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    prof.export_chrome_trace(f"trace_{name}.json")


def main():
    torch.manual_seed(42)
    d_model, n_heads = 512, 8
    model = TransformerBlock(d_model, n_heads).cuda().half()

    # Prefill: 处理长 prompt（N=1024）
    x_prefill = torch.randn(1, 1024, d_model, device="cuda", dtype=torch.float16)
    profile_phase(model, x_prefill, "prefill", n_iter=5)

    # Decode: 逐 token 生成（N=1）
    x_decode = torch.randn(1, 1, d_model, device="cuda", dtype=torch.float16)
    profile_phase(model, x_decode, "decode", n_iter=10)

    print("\n===== 观察要点 =====")
    print("1. Prefill 阶段：gemm 类算子 CUDA 时间占比最高（compute-bound）")
    print("2. Decode 阶段：总时间远小于 prefill，但单 token 时间占比不合理地高（memory-bound）")
    print("3. 对比 softmax/layernorm 在两阶段的绝对时间——decode 下它们可能占更大比例")
    print("4. 打开 trace_prefill.json（chrome://tracing）观察 kernel 顺序与间隙")


if __name__ == "__main__":
    main()
```

#### 运行步骤

```bash
# 运行（需 CUDA GPU）
python trace_transformer.py

# 打开 Chrome trace 可视化
# 1. 浏览器访问 chrome://tracing
# 2. Load trace_prefill.json 和 trace_decode.json
# 3. 观察 GPU kernel 的时间线排列
```

#### 预期输出与分析任务

```
===== Prefill Phase (shape=(1, 1024, 512)) =====
--------------------------------- ... ---------------------------------
Name            Self CUDA     Calls    ...
aten::_scaled_dot_product...    xxx us    5
aten::mm                        xxx us    20   ← QKV/Out/FFN GEMM
aten::layer_norm                xxx us    10
aten::softmax                   xxx us    5
...

===== Decode Phase (shape=(1, 1, 512)) =====
--------------------------------- ... ---------------------------------
Name            Self CUDA     Calls    ...
aten::mm                        xxx us    20   ← GEMM 但矩阵极小
aten::layer_norm                xxx us    10
aten::softmax                   xxx us    5
...
```

**分析任务清单**：
1. 找出 Prefill 阶段 CUDA 时间 top3 算子（预期是 mm/linear 类 GEMM）
2. 找出 Decode 阶段 CUDA 时间 top3 算子（预期 GEMM 占比下降，layernorm/softmax 占比上升）
3. 计算 Prefill 单 token 时间 vs Decode 单 token 时间（Prefill 快得多，因为并行度高）
4. 在 Chrome trace 中观察 kernel 之间的间隙（gap = launch overhead）

#### 练习题

**练习1（基础）**：修改 `d_model=1024, n_heads=16`，重新 profile，观察 GEMM 时间变化。
> 提示：GEMM 计算量与 d_model 平方相关，layernorm/softmax 与 d_model 线性相关。

**练习2（进阶）**：用 `nsys profile -o transformer_trace python trace_transformer.py` 采集系统级时间线，在 Nsight Systems GUI 中对比 Prefill 和 Decode 的 SM 利用率。
> 提示：Decode 阶段 SM 利用率会很低（绿色 bar 很短），这就是 memory-bound 的直观表现。

**练习3（综合）**：在 TransformerBlock 中加一个 `forward_with_fusion` 方法，手动把 LayerNorm + QKV Linear 融合为一个操作（用 `torch.compile` 或手写），对比 profile 结果。
> 提示：`torch.compile(model, mode="reduce-overhead")` 会自动做 kernel fusion，对比 fused vs unfused 的 kernel 数量。

---

### 今日面试题

**面试题1**：Transformer 推理的 Prefill 和 Decode 阶段有什么区别？为什么 Decode 通常是 memory-bound？（⭐⭐⭐ 高频）

**参考答案要点**：
- **Prefill**：输入是 `(B, N_prompt, d)`，N_prompt 可达数千。所有 GEMM 是大矩阵乘，计算量大，GPU SM 充分利用 → **Compute-bound**
- **Decode**：输入是 `(B, 1, d)`，每次只生成 1 个 token。GEMM 退化为向量×矩阵（M=1），计算量极小，但每次都要读取整个 KV Cache（N 个历史 token） → **Memory-bound**
- **根本原因**：Decode 阶段计算强度（FLOP/Byte）极低。M=1 的 GEMM 每读 1 行 K/V 只做 d 次乘加，arithmetic intensity ≈ d/(2d×4) ≈ 0.125 FLOP/Byte，远低于 Ridge Point
- **优化方向**：KV Cache（避免重算 K/V）、PagedAttention（减少 KV 显存碎片）、CUDA Graph（减少 launch overhead）、Continuous Batching（合并多个 decode 请求提高 M）

**面试题2**：Transformer 单层包含哪些算子？哪些是 compute-bound，哪些是 memory-bound？（⭐⭐⭐ 高频）

**参考答案要点**：
| 算子 | 类型（Prefill） | 类型（Decode） | 原因 |
|------|----------------|----------------|------|
| QKV/Out/FFN Linear (GEMM) | Compute-bound | Memory-bound | Prefill M 大；Decode M=1 |
| Attention QK^T (GEMM) | Compute-bound | Memory-bound | 同上 |
| Attention Softmax | Memory-bound | Memory-bound | element-wise + reduction |
| Attention PV (GEMM) | Compute-bound | Memory-bound | 同上 |
| LayerNorm | Memory-bound | Memory-bound | element-wise + reduction |
| GELU | Memory-bound | Memory-bound | element-wise |

---

### 今日自测清单

- [ ] 能解释 Prefill 和 Decode 的输入形状差异及其对性能的影响
- [ ] 能列出 Transformer 单层的 6 类算子及其执行顺序
- [ ] torch.profiler 代码运行成功，输出 Prefill/Decode 的算子时间表
- [ ] 找出 Prefill 阶段 CUDA 时间 top3 算子
- [ ] 能解释为什么 Decode 阶段 GEMM 变成 memory-bound（M=1 导致计算强度低）
- [ ] 能用 chrome://tracing 打开 trace 文件并观察 kernel 间隙
- [ ] 能对照昇腾 msprof 解释 PyTorch profiler 的对应关系

---

## Day 16（周二）：手写 Softmax 与 LayerNorm Kernel

> **今日目标**：复用 Week 2 的 Warp Shuffle 归约技术，手写 row-wise Softmax 和 LayerNorm Kernel，理解 safe softmax 的数值稳定性，掌握两级 block reduce 的工程写法。
> **时间分配**：早间1.5h（理论学习1h + 昇腾对照30min）+ 晚间1.5h（编程+调试）
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，Softmax/LayerNorm 是 Transformer 里最典型的 memory-bound 算子，手写 reduce 是标配

---

### 学习任务1：Softmax 数值稳定性与并行化（45分钟）

#### 阅读内容
- **资源**：PyTorch 官方文档 `torch.softmax` 的 numerical stability 说明
- **补充阅读**：Week 2 Day 8 的 Warp Shuffle 笔记（复用 `warpReduceSum` / `warpReduceMax`）
- **具体阅读重点**：
  - 朴素 softmax 的数值溢出问题（exp 大数爆炸）
  - safe softmax：先减 max 再 exp
  - row-wise 并行：一行一个 block，block 内做 reduce

#### 核心概念笔记

**1. 朴素 Softmax vs Safe Softmax**

```
朴素 Softmax（会溢出）：
  yi = exp(xi) / Σ exp(xj)
  问题：当 xi = 1000 时，exp(1000) = Inf，结果全 NaN

Safe Softmax（减去 max）：
  m = max(xj)
  yi = exp(xi - m) / Σ exp(xj - m)
  原理：exp(xi - m) ≤ exp(0) = 1，不会溢出
```

**2. Safe Softmax 的三遍扫描 vs 两遍扫描**

| 方法 | 扫描次数 | 操作 | 适用场景 |
|------|---------|------|---------|
| 三遍扫描 | 3 | ① 求 max ② 求 sum(exp(x-m)) ③ 归一化 | 教学版，清晰易读 |
| 两遍扫描（online） | 2 | ① 同时求 max 和 sum ② 归一化 | 生产版，减少一次全局读 |
| FlashAttention 版 | 1.x | 分块 online，边算边更新 | 极致优化，Week 4 主题 |

本日实现**三遍扫描版**（教学清晰），两遍/分块版留到 Week 4 FlashAttention。

**3. Row-wise 并行策略**

```
矩阵 shape: (M, D)，M 行，每行 D 个元素
并行映射: 一个 block 处理一行
  blockIdx.x = row index（0 ~ M-1）
  blockDim.x = 线程数（通常 256 或 512，需 ≥ D 的处理能力）

每个 block 内：
  Step 1: 所有线程协作求本行 max（block reduce max）
  Step 2: 所有线程协作求本行 sum(exp(x - max))（block reduce sum）
  Step 3: 所有线程协作做归一化写出
```

**4. Block Reduce 的两级结构（复用 Week 2 Day 8）**

```
Warp 级 reduce（32 线程）→ Shared Memory（32 个 warp 部分和）→ Warp 0 最终 reduce
```

- 第一级：每个 warp 用 `__shfl_down_sync` 折半累加/取 max，结果存在 lane 0
- 第二级：lane 0 写入 shared memory，warp 0 读取再做一次 warp reduce

#### 昇腾对照

| CUDA 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| `__shfl_down_sync` warp reduce | `__reduce_add` / `__reduce_max`（Ascend C 内置） | CUDA 需手写 butterfly 循环；昇腾提供高级 reduce API |
| `__shared__ float smem[32]` | L0 Buffer / UB 临时存储 | 都用于存储 warp 间中间结果 |
| row-wise softmax（一行一 block） | Ascend C Softmax 算子 | 昇腾 Softmax 算子内部同样按行 reduce，开发者无需手写 |
| safe softmax（减 max） | safe softmax（减 max） | 数值稳定性策略完全一致，跨平台通用 |
| 三遍扫描 | 两遍/向量化扫描 | 昇腾算子库已优化为向量化 reduce，CUDA 教学版用三遍 |

**关键差异**：CUDA 教学版需要开发者手写 reduce 细节；昇腾 Ascend C 提供 `__reduce_add_sync` 等内置函数，调用更简洁，但底层逻辑一致。

---

### 学习任务2：LayerNorm 公式与并行化（30分钟）

#### 核心概念笔记

**1. LayerNorm 公式**

```
输入: x ∈ R^D（一行 D 个元素）
参数: γ (gamma), β (beta) ∈ R^D

计算:
  μ = (1/D) Σ xi                      （均值）
  σ² = (1/D) Σ (xi - μ)²              （方差）
  yi = γi · (xi - μ) / sqrt(σ² + ε) + βi   （归一化 + affine）
```

**2. LayerNorm vs BatchNorm**

| 特性 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| 归一化维度 | 沿 feature 维（一行） | 沿 batch 维（一列） |
| 依赖 batch | 否（每样本独立） | 是（需 batch 统计） |
| 推理行为 | 训练/推理一致 | 推理用 running mean/var |
| 适用场景 | Transformer、RNN | CNN |

**3. LayerNorm 的 reduce 需求**

LayerNorm 需要**两次 reduce**：
1. 第一次：求 `μ = mean(x)`（reduce sum，然后除以 D）
2. 第二次：求 `σ² = mean((x - μ)²)`（reduce sum of squares，然后除以 D）

```
Step 1: 所有线程协作求 sum(x) → μ = sum / D
Step 2: 所有线程协作求 sum((x - μ)²) → σ² = sumSq / D
Step 3: 所有线程协作做归一化: y = (x - μ) / sqrt(σ² + ε) * γ + β
```

> 💡 **为什么 LayerNorm 是 memory-bound？** 每个元素读 1 次（x）、写 1 次（y），但只做 ~5 次浮点运算。Arithmetic intensity ≈ 5 / 8 ≈ 0.625 FLOP/Byte，远低于 Ridge Point（~12.6），纯 memory-bound。

#### 昇腾对照

| CUDA 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| 两次 block reduce（mean, variance） | LayerNorm 算子内部两次 reduce | 算法逻辑完全一致 |
| `rsqrtf(var + eps)` | `1/sqrt(var + eps)` | 数学等价，昇腾用 Vector Unit 指令 |
| `gamma[i], beta[i]` affine | affine 参数 | 参数语义一致，都存在 HBM |
| element-wise 归一化 | Vector Unit element-wise | 昇腾 Vector Unit 天然适合 element-wise |

---

### 晚间编程任务：Softmax + LayerNorm Kernel（1.5小时）

#### 完整代码

```cpp
// softmax_layernorm.cu —— Softmax + LayerNorm 完整实现
// 编译命令: nvcc -o softmax_layernorm softmax_layernorm.cu -O3 -arch=sm_80
// 运行命令: ./softmax_layernorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// 复用 Week 2 Day 8 的 Warp Shuffle 原语
// ============================================================
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ============================================================
// Block 级 reduce：warp 级 → shared memory → warp 0 最终 reduce
// smem 为外部传入的 shared memory 缓冲区（至少 32 个 float）
// 注意：返回后只有 warp 0 的线程持有正确结果，调用方需用 shared 变量广播
// ============================================================
__inline__ __device__ float blockReduceSum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__inline__ __device__ float blockReduceMax(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// ============================================================
// Softmax Kernel：一行一个 block，三遍扫描 safe softmax
// 输入: input[M][D]，输出: output[M][D]
// ============================================================
__global__ void softmax_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int M, int D) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = input + row * D;
    float* out_row = output + row * D;

    __shared__ float smem[32];   // warp 间 reduce 缓冲区
    __shared__ float row_max;
    __shared__ float row_sum;

    int tid = threadIdx.x;

    // Step 1: 求 max（数值稳定性）
    float local_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    local_max = blockReduceMax(local_max, smem);
    if (tid == 0) row_max = local_max;
    __syncthreads();

    // Step 2: 求 sum(exp(x - max))
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_sum += expf(in_row[i] - row_max);
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_sum = local_sum;
    __syncthreads();

    // Step 3: 归一化写出
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < D; i += blockDim.x) {
        out_row[i] = expf(in_row[i] - row_max) * inv_sum;
    }
}

// ============================================================
// LayerNorm Kernel：一行一个 block，两次 reduce
// 输入: input[M][N]，参数: gamma[N], beta[N]，输出: output[M][N]
// ============================================================
__global__ void layernorm_kernel(const float* __restrict__ input,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ output,
                                  int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;
    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    __shared__ float smem[32];
    __shared__ float row_mean;
    __shared__ float row_rstd;

    int tid = threadIdx.x;

    // Step 1: 求 mean = sum(x) / N
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += in_row[i];
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_mean = local_sum / N;
    __syncthreads();

    // Step 2: 求 variance = sum((x - mean)^2) / N，rstd = 1/sqrt(var + eps)
    float local_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = in_row[i] - row_mean;
        local_sq += diff * diff;
    }
    local_sq = blockReduceSum(local_sq, smem);
    if (tid == 0) row_rstd = rsqrtf(local_sq / N + eps);
    __syncthreads();

    // Step 3: 归一化 + affine: y = (x - mean) * rstd * gamma + beta
    for (int i = tid; i < N; i += blockDim.x) {
        out_row[i] = (in_row[i] - row_mean) * row_rstd * gamma[i] + beta[i];
    }
}

// ============================================================
// Host 辅助函数与验证
// ============================================================
void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
    }
}

void cpuSoftmax(const float* in, float* out, int M, int D) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * D;
        float* orow = out + r * D;
        float mx = ir[0];
        for (int i = 1; i < D; i++) mx = fmaxf(mx, ir[i]);
        float s = 0.0f;
        for (int i = 0; i < D; i++) { orow[i] = expf(ir[i] - mx); s += orow[i]; }
        for (int i = 0; i < D; i++) orow[i] /= s;
    }
}

void cpuLayerNorm(const float* in, const float* gamma, const float* beta,
                  float* out, int M, int N, float eps) {
    for (int r = 0; r < M; r++) {
        const float* ir = in + r * N;
        float* orow = out + r * N;
        float mean = 0.0f;
        for (int i = 0; i < N; i++) mean += ir[i];
        mean /= N;
        float var = 0.0f;
        for (int i = 0; i < N; i++) { float d = ir[i] - mean; var += d * d; }
        var /= N;
        float rstd = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < N; i++)
            orow[i] = (ir[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

bool checkResult(const float* a, const float* b, int n, float eps, const char* name) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    bool ok = maxDiff < eps;
    printf("%s: maxDiff = %.2e (%s)\n", name, maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    // 测试配置
    const int M = 128;       // 行数（batch * seq_len）
    const int D = 1024;      // 特征维（feature dim）
    const float eps = 1e-5f;
    const int threads = 256;

    printf("=== Softmax + LayerNorm Kernel Test ===\n");
    printf("Config: M=%d, D=%d, threads=%d\n\n", M, D, threads);

    size_t bytes = (size_t)M * D * sizeof(float);

    // Host 内存
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(D * sizeof(float));
    float *h_beta = (float*)malloc(D * sizeof(float));
    initData(h_in, M * D);
    for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    // Device 内存
    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_gamma, D * sizeof(float));
    cudaMalloc(&d_beta, D * sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Softmax 测试 ----
    printf("[Softmax]\n");
    cudaEventRecord(start);
    softmax_kernel<<<M, threads>>>(d_in, d_out, M, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float smMs;
    cudaEventElapsedTime(&smMs, start, stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuSoftmax(h_in, h_ref, M, D);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  Softmax vs CPU");
    printf("  Time: %.3f ms\n", smMs);

    // ---- LayerNorm 测试 ----
    printf("[LayerNorm]\n");
    cudaEventRecord(start);
    layernorm_kernel<<<M, threads>>>(d_in, d_gamma, d_beta, d_out, M, D, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float lnMs;
    cudaEventElapsedTime(&lnMs, start, stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cpuLayerNorm(h_in, h_gamma, h_beta, h_ref, M, D, eps);
    checkResult(h_out, h_ref, M * D, 1e-5f, "  LayerNorm vs CPU");
    printf("  Time: %.3f ms\n", lnMs);

    // 释放
    free(h_in); free(h_out); free(h_ref); free(h_gamma); free(h_beta);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o softmax_layernorm softmax_layernorm.cu -O3 -arch=sm_80

# 运行
./softmax_layernorm

# 预期输出
# === Softmax + LayerNorm Kernel Test ===
# Config: M=128, D=1024, threads=256
#
# [Softmax]
#   Softmax vs CPU: maxDiff = x.xx e-07 (PASS)
#   Time: x.xxx ms
# [LayerNorm]
#   LayerNorm vs CPU: maxDiff = x.xx e-06 (PASS)
#   Time: x.xxx ms
```

#### 练习题

**练习1（基础）**：修改 `D` 为 768、4096，重新运行，观察性能变化。
> 提示：D 增大时，每个 block 的工作量增加，但 reduce 次数不变（仍是 warp shuffle）。性能主要由 D 决定的内存读写量决定。

**练习2（进阶）**：实现一个**在线 Softmax**（两遍扫描）——第一遍同时求 max 和 sum，第二遍归一化。对比三遍扫描的性能差异。
> 提示：online softmax 的核心是 `l_new = l * exp(m_old - m_new) + sum_new`，在一次遍历中同时更新 m 和 l。这正是 FlashAttention 的基础。

**练习3（综合）**：用 `ncu` 分析 Softmax kernel，检查它是否真的是 memory-bound。
> 提示：`ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed ./softmax_layernorm`。预期 DRAM Throughput >> SM Throughput。

---

### 今日面试题

**面试题1**：Softmax 为什么要减去 max？不减会怎样？（⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **数值稳定性**：`exp(1000) = Inf`，直接算 `exp(xi)/Σexp(xj)` 会溢出。减去 max 后 `exp(xi - m) ≤ 1`，不会溢出
- **数学等价性**：`exp(xi - m) / Σexp(xj - m) = exp(xi)·exp(-m) / (Σexp(xj))·exp(-m) = exp(xi)/Σexp(xj)`，结果完全一致
- **不减的后果**：当输入有较大值（如 logits 未归一化时），exp 立即溢出为 Inf/NaN
- **实际场景**：FP16 下更容易溢出（FP16 max ≈ 65504），所以混合精度训练中 softmax 必须用 FP32 做 reduce

**面试题2**：LayerNorm 需要几次 reduce？每次 reduce 什么？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **两次 reduce**：
  1. 第一次：`μ = mean(x)` → reduce sum，然后除以 D
  2. 第二次：`σ² = mean((x - μ)²)` → reduce sum of squares，然后除以 D
- **为什么不能合并**：第二次 reduce 依赖第一次的结果（μ），必须先算完均值才能算方差
- **并行策略**：一行一个 block，block 内用 warp shuffle + shared memory 做两级 reduce
- **与 BatchNorm 区别**：LayerNorm 沿 feature 维归一化（每样本独立），不需要 batch 统计，推理时行为一致

**面试题3**：为什么 Softmax/LayerNorm 是 memory-bound？如何优化？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **Arithmetic intensity 低**：Softmax 每元素读 1 次写 1 次（8 bytes），做 ~3 次运算（exp、加、除），AI ≈ 0.375 FLOP/Byte，远低于 Ridge Point（~12.6）
- **优化方向**：
  1. **Kernel Fusion**：把 Softmax/LayerNorm 与相邻算子融合，避免中间结果写回 HBM（最重要）
  2. **向量化加载**：用 `float4` 做 128-bit 加载，提升带宽利用率
  3. **减少 reduce 次数**：online softmax 把三遍减为两遍
  4. **FP16/BF16 存储**：减少 HBM 读写量（但 reduce 用 FP32 保精度）

---

### 今日自测清单

- [ ] 能解释 safe softmax 为什么要减 max（数值稳定性 + 数学等价性）
- [ ] 能画出 Softmax 三遍扫描的流程（求 max → 求 sum → 归一化）
- [ ] 能复用 Week 2 的 warpReduceSum/warpReduceMax 实现 blockReduceSum/blockReduceMax
- [ ] Softmax Kernel 编译运行正确，与 CPU 对比误差 < 1e-5
- [ ] LayerNorm Kernel 编译运行正确，与 CPU 对比误差 < 1e-5
- [ ] 能解释 LayerNorm 为什么需要两次 reduce（方差依赖均值）
- [ ] 能用 ncu 验证 Softmax 是 memory-bound（DRAM Throughput >> SM Throughput）
- [ ] 能对照昇腾解释 `__reduce_add_sync` 与 `__shfl_down_sync` 的对应关系

---

## Day 17（周三）：源码分析 —— PyTorch / FasterTransformer

> **今日目标**：阅读 PyTorch ATen 和 FasterTransformer 中 Softmax/LayerNorm 的官方 CUDA 实现，理解工业级优化手法（向量化加载、warp vs block reduce 选择、register 缓存）。
> **时间分配**：早间1.5h（PyTorch 源码45min + FasterTransformer 45min）+ 晚间1h（对比分析）
> **面试考察度**：⭐⭐⭐ 中频，"你看过哪些开源 kernel 实现"是加分题

---

### 学习任务1：PyTorch ATen Softmax 源码（45分钟）

#### 阅读内容
- **源码地址**：https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu
- **辅助阅读**：PyTorch 的 ` Reduction.cuh` 中的 reduce 框架
- **具体阅读重点**：
  - `softmax_warp_forward` 函数：warp 级实现（D ≤ 1024 时一个 warp 处理一行）
  - `softmax_block_forward` 函数：block 级实现（D > 1024 时一个 block 处理一行）
  - 向量化加载：使用 `load` 模板按 4/8 元素批量加载

#### 核心概念笔记

**1. PyTorch Softmax 的两种实现路径**

| 路径 | 触发条件 | 并行粒度 | 优势 |
|------|---------|---------|------|
| `softmax_warp_forward` | D ≤ 1024（可被一个 warp 处理） | 一个 warp 一行 | 无需 `__syncthreads`，延迟最低 |
| `softmax_block_forward` | D > 1024 | 一个 block 一行 | 更多线程协作，处理大 D |

**关键洞察**：当 D 较小（如 768、1024）时，PyTorch 用**一个 warp 处理一行**，32 个线程用 `__shfl` 直接 reduce，完全不用 shared memory，避免同步开销。这与我们 Day 16 的"一个 block 一行"不同——我们的是教学通用版，PyTorch 是针对常见 D 优化的特化版。

**2. 向量化加载（Vectorized Load）**

```cpp
// PyTorch 用模板参数 ILP（Instruction-Level Parallelism）控制每次加载元素数
// ILP=4 时，一次加载 4 个 float（128-bit），减少指令数
template <int ILP>
__device__ void load(float* dst, const float* src, int d) {
    // 按 ILP 个元素一组加载
}
```

对比我们的 Day 16 实现（逐元素加载）：
- 我们：`for (int i = tid; i < D; i += blockDim.x) val = in_row[i];` —— 每次加载 1 个 float
- PyTorch：每次加载 4 个 float（float4），减少 4x 加载指令

**3. 数值精度的处理**

PyTorch 的 softmax 在 FP16 输入时，**reduce 用 FP32** 累加：
```cpp
// 伪代码
acc = static_cast<float>(input[i]);  // FP16 → FP32
sum += acc;                          // FP32 累加
output[i] = static_cast<half>(acc / sum);  // FP32 → FP16 写回
```

> 💡 **为什么 reduce 要用 FP32？** FP16 的最大值约 65504，累加多个 exp 值容易溢出。用 FP32 做 reduce 保证数值稳定，这是混合精度训练的标准做法。

#### 昇腾对照

| PyTorch 优化 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| warp 级 softmax（D≤1024） | Ascend C Softmax 算子向量化 | 昇腾 Vector Unit 天然处理向量，无需区分 warp/block 路径 |
| float4 向量化加载 | Vector Unit 一次处理多元素 | 昇腾 Vector 指令天然是向量化的，无需手动 float4 |
| FP32 reduce 保精度 | FP32 reduce 保精度 | 混合精度策略跨平台一致 |
| ILP 模板参数 | Vector 指令的 repeat 参数 | 两者都通过"一次处理多元素"提升吞吐 |

---

### 学习任务2：FasterTransformer LayerNorm 源码（45分钟）

#### 阅读内容
- **源码地址**：https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/layernorm_kernels.cu
- **具体阅读重点**：
  - `generalLayerNorm` 函数：支持 FP32/FP16/BF16 的模板实现
  - 一次 reduce 优化版（同时求 mean 和 variance）
  - `__half2` 双精度加载（FP16 场景下一次处理 2 个 half）

#### 核心概念笔记

**1. FasterTransformer 的一次 reduce 优化**

我们的 Day 16 实现是两次 reduce（先 mean 后 variance）。FasterTransformer 优化为**一次遍历同时求 mean 和 variance**：

```
Welford 算法（在线均值/方差）：
  遍历每个元素 xi：
    count++
    delta = xi - mean
    mean += delta / count
    M2 += delta * (xi - mean)      // M2 累积平方差
  最终：variance = M2 / count
```

| 方法 | 遍历次数 | reduce 次数 | 优势 |
|------|---------|------------|------|
| 我们的 Day 16（两次 reduce） | 2 | 2 | 清晰易读，教学版 |
| FasterTransformer（Welford） | 1 | 1 | 减少一次 HBM 读，但数值略有差异 |
| 两遍扫描（先 mean 后 var） | 2 | 2 | 标准 PyTorch 做法 |

**2. `__half2` 向量化（FP16 场景）**

```cpp
// 一次加载 2 个 half（32-bit），用 __half2 类型
__half2 val = *reinterpret_cast<const __half2*>(&input[i]);
// 用 __hadd2 做成对加法
```

对比 FP32 的 float4（128-bit = 4 个 float）：FP16 的 half2（32-bit = 2 个 half）等效带宽翻倍（同样 128-bit 可装 8 个 half）。

**3. Register 缓存 gamma/beta**

当 D 较小（如 768）时，FasterTransformer 把 gamma/beta 加载到 register 一次性使用：
```cpp
float g = gamma[i];  // 加载到 register
float b = beta[i];   // 加载到 register
// 后续归一化直接用 register 中的 g, b，不重复读 HBM
```

#### 昇腾对照

| FasterTransformer 优化 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| Welford 一次 reduce | LayerNorm 算子一次遍历 | 昇腾算子库同样用 Welford 或类似在线算法 |
| `__half2` 向量化 | Vector Unit half 向量 | 昇腾 Vector 指令支持 FP16 向量，天然高效 |
| gamma/beta register 缓存 | L0 Buffer 缓存参数 | 昇腾把热点参数预加载到 L0 Buffer |
| 模板支持 FP32/FP16/BF16 | 算子库多精度支持 | 两者都通过模板/重载支持多精度 |

**关键发现**：FasterTransformer 是 NVIDIA 官方的推理优化库，其 LayerNorm 实现比 PyTorch ATen 更激进（一次 reduce + half2 + register 缓存），是手写 kernel 的最佳参考。

---

### 学习任务3：对比手写版与官方实现（30分钟）

#### 对比分析表

| 维度 | 我们 Day 16 版本 | PyTorch ATen | FasterTransformer |
|------|----------------|-------------|-------------------|
| **加载方式** | 逐元素 float | float4 向量化 | half2/float4 向量化 |
| **Softmax 路径** | 一个 block 一行 | warp(D≤1024)/block(D>1024) | block + 向量化 |
| **LayerNorm reduce** | 两次（mean, var） | 两次 | 一次（Welford） |
| **精度处理** | FP32 全程 | FP16 输入→FP32 reduce→FP16 输出 | 模板多精度 |
| **gamma/beta** | 每次从 HBM 读 | 从 HBM 读 | register 缓存（小 D） |
| **性能（相对）** | 1x（基准） | ~1.5-2x | ~2-3x |

#### 关键差距分析

1. **向量化加载缺失**：我们逐元素加载，官方用 float4/half2。这是最直接的 2-4x 提升点
2. **没有精度混合**：我们全程 FP32，官方在 FP16 输入时用 FP32 reduce。我们的版本更精确但更慢
3. **reduce 次数**：LayerNorm 我们用两次 reduce，FT 用一次 Welford。减少一次 HBM 读

---

### 晚间编程任务：优化对比实验（1小时）

#### 任务清单

1. **将 Day 16 的 Softmax 改为 warp 级实现**（D=1024 时一个 warp 一行）
   - 提示：`blockDim.x = 32`（一个 warp），`gridDim.x = M`，用 `__shfl` 直接 reduce，不用 shared memory
2. **在 Day 16 的 LayerNorm 中加入 float4 向量化加载**
   - 提示：将 `for (int i = tid; i < N; i += blockDim.x)` 改为 `for (int i = tid*4; i < N; i += blockDim.x*4)`，用 `reinterpret_cast<const float4*>` 加载
3. **用 ncu 对比优化前后**的 DRAM Throughput 和 SM Throughput

#### ncu 对比命令

```bash
# 编译优化版
nvcc -o softmax_layernorm_opt softmax_layernorm_opt.cu -O3 -arch=sm_80 -lineinfo

# profile
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__time_duration.sum \
  --kernel-name regex:softmax_kernel \
  ./softmax_layernorm_opt
```

#### 练习题

**练习1（基础）**：在 PyTorch 中找到 `softmax_warp_forward` 的调用条件，确认 D=1024 时走的是 warp 路径还是 block 路径。
> 提示：看 `at::native::softmax_cuda` 中的 dispatch 逻辑，通常以 1024 为分界。

**练习2（进阶）**：将 Day 16 的 LayerNorm 改为 Welford 一次 reduce 版本，对比与两次 reduce 的性能差异。
> 提示：Welford 的核心是 `mean += delta / count; M2 += delta * (xi - mean);`，注意并行 Welford 需要合并多个线程的统计量（较复杂，可参考论文 "Welford's Algorithm for Parallel Variance"）。

**练习3（综合）**：阅读 FasterTransformer 的 `generalLayerNorm`，列出它比 PyTorch ATen 多了哪些优化，并评估每个优化的收益。
> 提示：从加载方式、reduce 次数、精度处理、register 使用四个维度对比。

---

### 今日面试题

**面试题1**：PyTorch 的 Softmax 在 D 较小时为什么用 warp 级实现而不是 block 级？（⭐⭐⭐ 中频）

**参考答案要点**：
- **避免 `__syncthreads`**：warp 级 reduce 用 `__shfl` 直接在寄存器间传递，不需要 shared memory 和同步屏障；block 级需要 `__syncthreads`，有同步开销
- **更低延迟**：warp 内 shuffle 延迟 ~1-2 cycles，shared memory ~20-30 cycles
- **足够并行度**：D=1024 时，32 个线程每个处理 32 个元素，并行度足够
- **适用条件**：D ≤ 1024（一个 warp 能处理），且 M 足够大（每个 warp 一行，warp 数 = M）

**面试题2**：FP16 训练时 Softmax/LayerNorm 的 reduce 为什么要用 FP32？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **FP16 溢出风险**：FP16 max ≈ 65504，exp(x) 在 x > 11 时就接近溢出（exp(11) ≈ 60000）
- **累加精度**：FP16 的尾数只有 10 位（约 3 位有效十进制），多次累加会丢失精度
- **标准做法**：输入 FP16 → cast 到 FP32 做 reduce（max/sum/mean/variance）→ cast 回 FP16 输出
- **昇腾对照**：昇腾混合精度算子同样用 FP32 做 reduce，跨平台一致

---

### 今日自测清单

- [ ] 能找到并阅读 PyTorch `SoftMax.cu` 的 `softmax_warp_forward` 函数
- [ ] 能解释 warp 级 softmax 与 block 级 softmax 的触发条件（D ≤ 1024）
- [ ] 能阅读 FasterTransformer `layernorm_kernels.cu` 的 `generalLayerNorm`
- [ ] 能列出 FasterTransformer LayerNorm 比手写版多的 3 个优化（Welford/half2/register 缓存）
- [ ] 理解 float4 向量化加载为什么能提升 2-4x 性能
- [ ] 能解释 FP16 reduce 为什么要 cast 到 FP32
- [ ] 完成 Day 16 版本的至少一项优化（warp 级或 float4），并用 ncu 验证提升

---

## Day 18（周四）：Attention IO 分析

> **今日目标**：实现标准 Attention forward，手动计算各阶段 HBM 读写量，用 ncu 验证实测值，理解标准 Attention 为什么是 O(N²) IO —— 为 Week 4 FlashAttention 做铺垫。
> **时间分配**：早间1.5h（IO 量化推导1h + 昇腾对照30min）+ 晚间1.5h（编程+ncu 验证）
> **面试考察度**：⭐⭐⭐⭐⭐ 必考，"标准 Attention 的 IO 复杂度"是 FlashAttention 的核心前置知识

---

### 学习任务1：标准 Attention 的 HBM 读写量推导（45分钟）

#### 阅读内容
- **论文**：FlashAttention 论文 Section 2.3（标准 Attention 的 IO 复杂度分析）
- **补充阅读**：Week 2 Day 12 的 FlashAttention 笔记（对比 O(N²) vs O(Nd)）
- **具体阅读重点**：
  - 标准 Attention 把 S=QK^T 和 P=softmax(S) 物化到 HBM
  - 每个中间矩阵的读写量计算
  - O(N²) 项的来源

#### 核心概念笔记

**1. 标准 Attention 的计算流程**

```
输入: Q ∈ R^{N×d},  K ∈ R^{N×d},  V ∈ R^{N×d}

Step 1: S = Q × K^T / sqrt(d)     →  S ∈ R^{N×N}  （写 HBM）
Step 2: P = softmax(S, dim=-1)    →  P ∈ R^{N×N}  （读 S，写 P）
Step 3: O = P × V                 →  O ∈ R^{N×d}  （读 P，写 O）
```

**2. 各阶段 HBM 读写量（以 N=4096, d=64, FP32 为例）**

| 阶段 | 操作 | 读 HBM | 写 HBM | 小计 |
|------|------|--------|--------|------|
| Step 1: S=QK^T | 读 Q,K；写 S | N·d + N·d = 2Nd | N² | 2Nd + N² |
| Step 2: P=softmax(S) | 读 S；写 P | N² | N² | 2N² |
| Step 3: O=PV | 读 P,V；写 O | N² + N·d | N·d | N² + 2Nd |
| **总计** | | **2Nd + 3N² + Nd** | **N² + N² + Nd** | **3N² + 4Nd** |

代入 N=4096, d=64：
- `3N² = 3 × 4096² = 50,331,648` 元素
- `4Nd = 4 × 4096 × 64 = 1,048,576` 元素
- **总计 ≈ 51.4M 元素 × 4 bytes = ~206 MB**

**关键洞察**：当 N >> d 时（长序列），`3N²` 项主导，HBM 读写量是 **O(N²)**。这正是 FlashAttention 要解决的——通过分块 + online softmax，避免物化 S 和 P，把 IO 降到 O(Nd)。

**3. IO 复杂度对比**

| 实现 | HBM 读写量 | 当 N=4096, d=64 | 当 N=8192, d=64 |
|------|-----------|----------------|----------------|
| 标准 Attention | O(N² + Nd) | ~206 MB | ~805 MB（4x） |
| FlashAttention | O(Nd) | ~2 MB | ~4 MB（2x） |
| **加速比** | | **~100x** | **~200x** |

> 💡 **为什么长序列加速更明显？** 标准 Attention 的 IO 随 N² 增长，FlashAttention 随 N 线性增长。N 翻倍时，标准 Attention IO 变 4x，FlashAttention 只变 2x。

**4. Memory-bound 判定**

标准 Attention 的 arithmetic intensity：
```
FLOPs = 2·N·d (QK^T) + 3·N (softmax) + 2·N·d·N... 
简化：主要计算是两个 GEMM，共 2·N²·d + 2·N²·d = 4·N²·d FLOPs
Bytes = 3N² + 4Nd ≈ 3N²（当 N >> d）

AI = 4·N²·d / 3N² = (4/3)·d ≈ 85 FLOP/Byte（d=64）
```

对比 A100 Ridge Point（~12.6 FLOP/Byte）：AI=85 > 12.6 → **标准 Attention 的 GEMM 部分是 compute-bound**。

但 **softmax 部分**（读 S N²，写 P N²）是纯 memory-bound：
```
softmax FLOPs ≈ 3N（每元素 exp + add + div）
softmax Bytes = 2N²
AI_softmax = 3N / 2N² = 1.5/N ≈ 0.0004 FLOP/Byte（N=4096）
```

**结论**：标准 Attention 是 GEMM(compute) + softmax(memory) + GEMM(compute) 的混合，其中 softmax 的 O(N²) 读写是瓶颈，FlashAttention 正是消除这一项。

#### 昇腾对照

| CUDA/FlashAttention 概念 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| HBM（Global Memory） | DDR/HBM | 两者都面临 HBM 带宽瓶颈 |
| Shared Memory（片上） | L0 Buffer / UB | FlashAttention 的 tile 驻留 ≈ 昇腾 L0 Buffer 预加载 |
| O(N²) 物化 S/P | CANN 已内置 FlashAttention | 昇腾算子库直接提供 FlashAttention，无需手写标准版 |
| softmax 的 O(N²) 读写 | softmax 算子分块 | 昇腾 softmax 算子同样用分块避免 O(N²) 中间结果 |

---

### 学习任务2：理解 O(N²) 的危害（30分钟）

#### 为什么 O(N²) 是问题？

```
N = 1024:   S/P 矩阵 = 1024² × 4 bytes = 4 MB     （L2 cache 可容纳）
N = 4096:   S/P 矩阵 = 4096² × 4 bytes = 64 MB    （超出 L2，频繁 HBM 读写）
N = 16384:  S/P 矩阵 = 16384² × 4 bytes = 1 GB    （HBM 都吃紧，OOM 风险）
N = 65536:  S/P 矩阵 = 65536² × 4 bytes = 16 GB   （直接 OOM）
```

| N | S/P 显存 | 能否放入 L2(~40MB) | 能否放入 HBM(40GB) |
|---|---------|------------------|------------------|
| 1024 | 4 MB | ✅ | ✅ |
| 4096 | 64 MB | ❌ | ✅ |
| 16384 | 1 GB | ❌ | ✅（紧张） |
| 65536 | 16 GB | ❌ | ❌（OOM） |

**结论**：长序列下，物化 S/P 不仅导致 HBM 读写量大，还可能直接 OOM。FlashAttention 不物化 S/P，显存始终是 O(Nd)。

---

### 晚间编程任务：标准 Attention Forward + IO 验证（1.5小时）

#### 完整代码

```cpp
// attention_naive.cu —— 标准 Attention Forward（物化 S 和 P，用于 IO 分析）
// 编译命令: nvcc -o attention_naive attention_naive.cu -O3 -arch=sm_80
// 运行命令: ./attention_naive

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 复用 warp reduce 原语
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}
__inline__ __device__ float blockReduceSum(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
__inline__ __device__ float blockReduceMax(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (lane < numWarps) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// ============================================================
// 标准 Attention Forward Kernel（物化 S 和 P 到 HBM）
// 一个 block 处理一行 query
// ============================================================
__global__ void attention_naive_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ S,
                                        float* __restrict__ P,
                                        float* __restrict__ O,
                                        int N, int d) {
    int qrow = blockIdx.x;
    if (qrow >= N) return;

    __shared__ float smem[32];
    __shared__ float row_max;
    __shared__ float row_sum;

    int tid = threadIdx.x;
    float scale = 1.0f / sqrtf((float)d);

    // Step 1: S[qrow][j] = sum_d Q[qrow][d] * K[j][d] * scale
    //         物化 S 到 HBM（这就是 O(N²) 写入的来源）
    for (int j = tid; j < N; j += blockDim.x) {
        float s_val = 0.0f;
        for (int dd = 0; dd < d; dd++) {
            s_val += Q[qrow * d + dd] * K[j * d + dd];
        }
        S[qrow * N + j] = s_val * scale;
    }
    __syncthreads();

    // Step 2: P[qrow][j] = softmax(S[qrow][:])
    //         读 S（O(N²) 读），写 P（O(N²) 写）
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, S[qrow * N + j]);
    }
    local_max = blockReduceMax(local_max, smem);
    if (tid == 0) row_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float p_val = expf(S[qrow * N + j] - row_max);
        P[qrow * N + j] = p_val;
        local_sum += p_val;
    }
    local_sum = blockReduceSum(local_sum, smem);
    if (tid == 0) row_sum = local_sum;
    __syncthreads();

    // Step 3: O[qrow][dd] = sum_j P[qrow][j] * V[j][dd]
    //         读 P（O(N²) 读），读 V，写 O
    float inv_sum = 1.0f / row_sum;
    for (int dd = tid; dd < d; dd += blockDim.x) {
        float o_val = 0.0f;
        for (int j = 0; j < N; j++) {
            o_val += (P[qrow * N + j] * inv_sum) * V[j * d + dd];
        }
        O[qrow * d + dd] = o_val;
    }
}

// ============================================================
// CPU 参考（用于验证）
// ============================================================
void cpuAttention(const float* Q, const float* K, const float* V,
                  float* O, int N, int d) {
    float* S = (float*)malloc(N * N * sizeof(float));
    float* P = (float*)malloc(N * N * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int dd = 0; dd < d; dd++) s += Q[i*d+dd] * K[j*d+dd];
            S[i*N+j] = s * scale;
        }
        float mx = S[i*N];
        for (int j = 1; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
        float sm = 0.0f;
        for (int j = 0; j < N; j++) { P[i*N+j] = expf(S[i*N+j]-mx); sm += P[i*N+j]; }
        for (int j = 0; j < N; j++) P[i*N+j] /= sm;
        for (int dd = 0; dd < d; dd++) {
            float o = 0.0f;
            for (int j = 0; j < N; j++) o += P[i*N+j] * V[j*d+dd];
            O[i*d+dd] = o;
        }
    }
    free(S); free(P);
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    bool ok = maxDiff < eps;
    printf("  maxDiff = %.2e (%s)\n", maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    // 测试多个 seq_len，观察 IO 随 N² 增长
    int seqLens[] = {256, 512, 1024, 2048};
    int d = 64;
    int threads = 256;

    printf("=== Standard Attention Forward (naive, materialize S/P) ===\n");
    printf("%-8s %-12s %-14s %-12s %-10s\n",
           "N", "S/P size(MB)", "HBM IO(MB)", "Time(ms)", "Check");
    printf("--------------------------------------------------------\n");

    for (int si = 0; si < 4; si++) {
        int N = seqLens[si];
        size_t bytesQKV = N * d * sizeof(float);
        size_t bytesSP = (size_t)N * N * sizeof(float);

        float *h_Q = (float*)malloc(bytesQKV), *h_K = (float*)malloc(bytesQKV);
        float *h_V = (float*)malloc(bytesQKV), *h_O = (float*)malloc(bytesQKV);
        float *h_O_cpu = (float*)malloc(bytesQKV);
        initData(h_Q, N*d); initData(h_K, N*d); initData(h_V, N*d);

        float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
        cudaMalloc(&d_Q, bytesQKV); cudaMalloc(&d_K, bytesQKV);
        cudaMalloc(&d_V, bytesQKV); cudaMalloc(&d_O, bytesQKV);
        cudaMalloc(&d_S, bytesSP);  cudaMalloc(&d_P, bytesSP);
        cudaMemcpy(d_Q, h_Q, bytesQKV, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, bytesQKV, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, bytesQKV, cudaMemcpyHostToDevice);

        // 理论 HBM IO：读 Q,K,V（3Nd）+ 读/写 S,P（各 2N²）+ 写 O（Nd）
        // = 4Nd + 4N²（简化，每元素 4 bytes）
        double hbmIO = (4.0 * N * d + 4.0 * N * N) * sizeof(float) / (1024.0*1024.0);
        double spSize = (double)bytesSP / (1024.0*1024.0);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        attention_naive_kernel<<<N, threads>>>(d_Q, d_K, d_V, d_S, d_P, d_O, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_O, d_O, bytesQKV, cudaMemcpyDeviceToHost);
        cpuAttention(h_Q, h_K, h_V, h_O_cpu, N, d);
        bool ok = checkResult(h_O, h_O_cpu, N*d, 1e-3f);

        printf("%-8d %-12.2f %-14.2f %-12.3f %-10s\n",
               N, spSize, hbmIO, ms, ok ? "PASS" : "FAIL");

        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_O_cpu);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S);
        cudaFree(d_P); cudaFree(d_O);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n观察要点：\n");
    printf("1. S/P size 随 N² 增长（N 翻倍 → size 4x）\n");
    printf("2. HBM IO 随 N² 增长（N 翻倍 → IO 4x）\n");
    printf("3. Time 近似随 N² 增长（长序列下 O(N²) IO 主导）\n");
    printf("4. 用 ncu 验证 dram__bytes_read.sum + dram__bytes_write.sum ≈ 理论 HBM IO\n");
    return 0;
}
```

#### 编译运行步骤

```bash
# 编译
nvcc -o attention_naive attention_naive.cu -O3 -arch=sm_80 -g -lineinfo

# 运行
./attention_naive

# 预期输出
# === Standard Attention Forward (naive, materialize S/P) ===
# N        S/P size(MB) HBM IO(MB)    Time(ms)     Check
# --------------------------------------------------------
# 256      0.25         1.00          0.xxx        PASS
# 512      1.00         4.00          x.xxx        PASS
# 1024     4.00         16.00         x.xxx        PASS
# 2048     16.00        64.00         xx.xxx       PASS
```

#### 用 ncu 验证 HBM 读写量

```bash
# profile N=1024 的 HBM 读写量
ncu --metrics \
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:attention_naive \
  ./attention_naive

# 预期：dram__bytes_read + dram__bytes_write ≈ 16 MB（理论值）
# 注意：实测值会略大于理论值（cache miss、额外访问等），误差 < 30% 属正常
```

#### 练习题

**练习1（基础）**：手动计算 N=512, d=64 时的理论 HBM IO 量，与程序输出对比。
> 提示：`4·N·d + 4·N² = 4·512·64 + 4·512² = 131072 + 1048576 = 1179648 元素 × 4 bytes = 4.5 MB`。

**练习2（进阶）**：修改代码，只保留 S=QK^T 一步（不物化 P，直接在 register 里做 softmax），对比 HBM 读写量。
> 提示：这就是 FlashAttention 的雏形——减少一个中间矩阵的读写。

**练习3（综合）**：用 ncu 测量 N=1024 和 N=2048 的实际 HBM 读写量，绘制 IO 随 N 增长的曲线，验证 O(N²)。
> 提示：N 翻倍时，实测 HBM IO 应该接近 4x。

---

### 今日面试题

**面试题1**：标准 Attention 的 HBM 读写复杂度是多少？为什么是 O(N²)？（⭐⭐⭐⭐⭐ 必考）

**参考答案要点**：
- **复杂度**：O(N² + Nd)，当 N >> d 时简化为 O(N²)
- **O(N²) 来源**：标准 Attention 物化两个 N×N 中间矩阵：
  - S = QK^T（N×N）：Step 1 写入 HBM，Step 2 读出 → 2N²
  - P = softmax(S)（N×N）：Step 2 写入 HBM，Step 3 读出 → 2N²
  - 合计 4N² 的 N² 项读写
- **O(Nd) 来源**：Q/K/V 的读写（3Nd）和 O 的读写（2Nd），线性于 N
- **危害**：N=4096 时 S/P 各 64MB，N=16384 时各 1GB，长序列下显存和带宽都吃紧
- **FlashAttention 解决**：不物化 S/P，在 SRAM 中完成 softmax，IO 降到 O(Nd)

**面试题2**：标准 Attention 中 softmax 部分的 arithmetic intensity 是多少？是 compute-bound 还是 memory-bound？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **softmax 的 FLOPs**：每元素约 3 次运算（exp + add + div），共 3N² FLOPs
- **softmax 的 Bytes**：读 S（N²）+ 写 P（N²）= 2N² × 4 bytes
- **AI = 3N² / (2N² × 4) = 3/8 ≈ 0.375 FLOP/Byte**
- **判定**：AI=0.375 远低于 Ridge Point（~12.6）→ **纯 memory-bound**
- **优化方向**：FlashAttention 把 softmax 从 HBM 搬到 SRAM，消除 O(N²) 读写

---

### 今日自测清单

- [ ] 能手推标准 Attention 三阶段的 HBM 读写量
- [ ] 能解释 O(N²) 项的来源（物化 S 和 P 两个 N×N 矩阵）
- [ ] 标准 Attention kernel 编译运行正确，与 CPU 对比误差 < 1e-3
- [ ] 能用 ncu 验证实测 HBM 读写量与理论值一致（误差 < 30%）
- [ ] 能计算 softmax 部分的 arithmetic intensity 并判定为 memory-bound
- [ ] 能解释 N 翻倍时 HBM IO 变 4x 的原因
- [ ] 能对照昇腾解释为什么 CANN 内置 FlashAttention（避免手写标准版）

---

## Day 19（周五）：项目推进 —— 算子接入 Mini 引擎

> **今日目标**：将 Day 16 手写的 Softmax/LayerNorm kernel 封装为 C++ 接口，接入一个最小化的 Transformer 推理引擎，替换 PyTorch 对应算子，验证端到端正确性并记录 latency 变化。
> **时间分配**：早间1.5h（架构设计1h + 昇腾对照30min）+ 晚间1.5h（编码+调试）
> **面试考察度**：⭐⭐⭐ 中频，"如何把自定义算子集成到推理框架"是工程能力体现

---

### 学习任务1：Mini 引擎架构设计（45分钟）

#### 设计目标

构建一个最小化的 Transformer 单层推理引擎，支持：
1. 用 PyTorch 做张量管理（malloc/autograd 不需要）
2. 用自定义 CUDA kernel 替换 Softmax/LayerNorm
3. GEMM 仍用 PyTorch 的 `torch.mm`（cuBLAS），本周不优化 GEMM
4. 对比"全 PyTorch" vs "自定义算子"的 latency 和正确性

#### 架构图

```
┌──────────────────────────────────────────────────┐
│              Mini Transformer Engine             │
│                                                  │
│  Input x (B, N, d)                               │
│    │                                             │
│    ├─► [LayerNorm1] ──► [QKV GEMM (cuBLAS)]      │
│    │                       │                     │
│    │                       ├─► [QK^T GEMM]       │
│    │                       ├─► [Softmax★] ← 自定义│
│    │                       ├─► [PV GEMM]         │
│    │                       └─► [Out GEMM]        │
│    │                                             │
│    ├─► [LayerNorm2] ★ ← 自定义                   │
│    │                                             │
│    └─► [FFN GEMM1] → [GELU] → [FFN GEMM2]        │
│                                                  │
│  ★ = 自定义 CUDA kernel，其余用 PyTorch          │
└──────────────────────────────────────────────────┘
```

#### 接口设计原则

自定义算子通过 PyTorch 的 C++ Extension 机制接入，或用更简单的 `torch.utils.cpp_extension.load_inline` 动态编译：

```python
from torch.utils.cpp_extension import load_inline

# 把 Day 16 的 CUDA 代码作为 inline 编译
cuda_source = open("softmax_layernorm.cu").read()
my_ops = load_inline(
    name="my_ops",
    cpp_sources="...",  # C++ 接口声明
    cuda_sources=cuda_source,
    functions=["softmax_forward", "layernorm_forward"],
    verbose=True,
)
```

#### 昇腾对照

| CUDA 集成方式 | 昇腾 CANN 对应 | 对照说明 |
|---------|------------|---------|
| PyTorch C++ Extension | Ascend C 自定义算子 + PyTorch NPU | 两者都需要把自定义 kernel 封装为框架可调用的接口 |
| `load_inline` 动态编译 | `aclOpCompile` + 动态加载 | 概念类似，都是运行时编译/加载算子 |
| `torch.mm`（cuBLAS） | `aclnnMatmul`（ACL MatMul） | GEMM 都调用官方库，不自写 |
| 自定义 Softmax/LayerNorm | CANN 内置算子 | 昇腾已内置，无需自定义；CUDA 手写是学习目的 |

**关键差异**：CUDA 手写算子集成是学习性质（理解算子内部）；昇腾 CANN 已内置优化算子，生产环境直接调用。

---

### 学习任务2：封装自定义算子（45分钟）

#### C++ Wrapper 代码

```cpp
// my_ops.cpp —— PyTorch C++ Extension 接口
// 把 Day 16 的 kernel 封装为 torch 可调用函数
#include <torch/extension.h>
#include <cuda_runtime.h>

// 声明 Day 16 的 kernel（实现在 .cu 文件中）
void launch_softmax(const float* input, float* output, int M, int D, cudaStream_t stream);
void launch_layernorm(const float* input, const float* gamma, const float* beta,
                      float* output, int M, int N, float eps, cudaStream_t stream);

at::Tensor softmax_forward(at::Tensor input) {
    // input: (M, D)
    int M = input.size(0), D = input.size(1);
    auto output = at::empty_like(input);
    launch_softmax(input.data_ptr<float>(), output.data_ptr<float>(),
                   M, D, at::cuda::getCurrentCUDAStream());
    return output;
}

at::Tensor layernorm_forward(at::Tensor input, at::Tensor gamma, at::Tensor beta, double eps) {
    int M = input.size(0), N = input.size(1);
    auto output = at::empty_like(input);
    launch_layernorm(input.data_ptr<float>(), gamma.data_ptr<float>(),
                     beta.data_ptr<float>(), output.data_ptr<float>(),
                     M, N, (float)eps, at::cuda::getCurrentCUDAStream());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_forward", &softmax_forward, "Softmax forward (CUDA)");
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
}
```

#### Python 集成代码

```python
# mini_engine.py —— Mini Transformer 引擎（自定义算子版）
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 加载自定义算子（需先准备 my_ops.cpp 和 softmax_layernorm.cu）
# 简化版：用 load_inline 动态编译
import os
cuda_src = open("softmax_layernorm.cu").read()
cpp_src = """
#include <torch/extension.h>
at::Tensor softmax_forward(at::Tensor input);
at::Tensor layernorm_forward(at::Tensor input, at::Tensor gamma, at::Tensor beta, double eps);
"""
my_ops = load_inline(
    name="my_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["softmax_forward", "layernorm_forward"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"],
)


class MiniAttention(nn.Module):
    """用自定义 Softmax 的 Attention"""
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # QK^T (cuBLAS)
        # ★ 自定义 Softmax 替换 torch.softmax
        # attn shape: (B, n_heads, N, N) → 展平为 (B*n_heads*N, N) 调用
        B_, H, N_, _ = attn.shape
        attn_flat = attn.reshape(B_*H*N_, N_)
        attn_flat = my_ops.softmax_forward(attn_flat)
        attn = attn_flat.reshape(B_, H, N_, N_)
        out = torch.matmul(attn, v)  # PV (cuBLAS)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, use_custom_ln=True):
        super().__init__()
        self.attn = MiniAttention(d_model, n_heads)
        # LayerNorm 参数
        self.norm1_weight = nn.Parameter(torch.ones(d_model))
        self.norm1_bias = nn.Parameter(torch.zeros(d_model))
        self.norm2_weight = nn.Parameter(torch.ones(d_model))
        self.norm2_bias = nn.Parameter(torch.zeros(d_model))
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.use_custom_ln = use_custom_ln
        self.eps = 1e-5

    def forward(self, x):
        B, N, D = x.shape
        # ★ 自定义 LayerNorm 替换 F.layer_norm
        if self.use_custom_ln:
            x_flat = x.reshape(B*N, D)
            x_norm = my_ops.layernorm_forward(x_flat, self.norm1_weight, self.norm1_bias, self.eps)
            x_norm = x_norm.reshape(B, N, D)
        else:
            x_norm = torch.nn.functional.layer_norm(x, (D,), self.norm1_weight, self.norm1_bias, self.eps)
        x = x + self.attn(x_norm)

        if self.use_custom_ln:
            x_flat = x.reshape(B*N, D)
            x_norm = my_ops.layernorm_forward(x_flat, self.norm2_weight, self.norm2_bias, self.eps)
            x_norm = x_norm.reshape(B, N, D)
        else:
            x_norm = torch.nn.functional.layer_norm(x, (D,), self.norm2_weight, self.norm2_bias, self.eps)
        x = x + self.ffn(x_norm)
        return x


def benchmark(model, x, name, n_iter=20):
    """对比 latency"""
    for _ in range(3):  # warmup
        _ = model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = model(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iter
    print(f"{name}: {ms:.3f} ms / forward")
    return ms


def main():
    torch.manual_seed(42)
    d_model, n_heads = 512, 8
    x = torch.randn(1, 1024, d_model, device="cuda", dtype=torch.float32)

    # 全 PyTorch 版
    model_pytorch = TransformerBlock(d_model, n_heads, use_custom_ln=False).cuda()
    # 自定义算子版
    model_custom = TransformerBlock(d_model, n_heads, use_custom_ln=True).cuda()
    model_custom.load_state_dict(model_pytorch.state_dict())

    # 正确性验证
    with torch.no_grad():
        out_pytorch = model_pytorch(x)
        out_custom = model_custom(x)
    max_diff = (out_pytorch - out_custom).abs().max().item()
    print(f"Max diff (PyTorch vs Custom): {max_diff:.2e}")
    assert max_diff < 1e-4, "Correctness check failed"

    # latency 对比
    print("\n=== Latency Comparison (Prefill, N=1024) ===")
    with torch.no_grad():
        ms_pt = benchmark(model_pytorch, x, "PyTorch (F.softmax + F.layer_norm)")
        ms_my = benchmark(model_custom, x, "Custom (my_ops.softmax + my_ops.layernorm)")
    print(f"Speedup: {ms_pt/ms_my:.2f}x")


if __name__ == "__main__":
    main()
```

#### 运行步骤

```bash
# 需要准备 softmax_layernorm.cu（Day 16 代码）+ 添加 launch 函数
# 运行
python mini_engine.py

# 预期输出
# Max diff (PyTorch vs Custom): x.xx e-06
#
# === Latency Comparison (Prefill, N=1024) ===
# PyTorch (F.softmax + F.layer_norm): x.xxx ms / forward
# Custom (my_ops.softmax + my_ops.layernorm): x.xxx ms / forward
# Speedup: 0.8x ~ 1.2x
```

> ⚠️ **预期结果**：自定义算子可能比 PyTorch **慢**（0.8x），因为 PyTorch 的 softmax/layernorm 已经过高度优化（向量化、warp 级、融合）。这正常——本周的目标是**理解算子**，不是超越官方实现。Week 4 的 FlashAttention 才是真正能超越的场景。

#### 练习题

**练习1（基础）**：在自定义 LayerNorm 中加入 float4 向量化加载（参考 Day 17），对比 latency 变化。
> 提示：向量化加载后，自定义算子可能接近 PyTorch 性能。

**练习2（进阶）**：用 `torch.compile(model_custom, mode="reduce-overhead")` 编译自定义版引擎，对比编译前后的 latency。
> 提示：`torch.compile` 会融合部分算子并减少 launch overhead，对 Decode 阶段（kernel 小而多）提升明显。

**练习3（综合）**：在 Mini Attention 中也用自定义 Softmax 替换 decode 阶段（N=1）的 softmax，对比 prefill 和 decode 下的 speedup 差异。
> 提示：Decode 下自定义算子的相对劣势可能更大（PyTorch 对小张量也有特化路径）。

---

### 今日面试题

**面试题1**：如何把自定义 CUDA 算子集成到 PyTorch 中？有几种方式？（⭐⭐⭐ 中频）

**参考答案要点**：
- **方式1：C++ Extension（推荐）**：写 `.cpp`（接口）+ `.cu`（kernel），用 `torch.utils.cpp_extension.load_inline` 动态编译或 `setup.py` 静态编译
- **方式2：TorchScript/Custom Operator**：用 `torch.ops.register` 注册自定义 op
- **方式3：Triton**：用 Python 写 kernel，`torch.compile` 自动集成（无需 C++）
- **集成要点**：① 用 `at::Tensor` 接收张量 ② 用 `data_ptr<float>()` 获取裸指针 ③ 用 `at::cuda::getCurrentCUDAStream()` 获取当前 stream ④ 用 `auto out = at::empty_like(input)` 分配输出
- **昇腾对照**：昇腾用 `aclOpCompile` + 自定义算子注册，机制类似

**面试题2**：为什么自定义 Softmax/LayerNorm 通常比 PyTorch 官方实现慢？（⭐⭐⭐ 中频）

**参考答案要点**：
- **PyTorch 已高度优化**：warp 级特化路径、float4/half2 向量化、Welford 一次 reduce、FP32 混合精度
- **教学版缺失优化**：逐元素加载、两次 reduce、全程 FP32
- **JIT/编译优化**：PyTorch 2.0 的 `torch.compile` 会做 kernel fusion，进一步拉开差距
- **超越场景**：只有当官方实现**没有覆盖**你的场景时（如 FlashAttention 的分块 softmax），自定义才有优势

---

### 今日自测清单

- [ ] 能用 `load_inline` 把 Day 16 kernel 集成到 PyTorch
- [ ] 自定义算子版 Transformer 编译运行成功
- [ ] 自定义版与 PyTorch 版输出误差 < 1e-4
- [ ] 记录了 prefill 阶段自定义版 vs PyTorch 版的 latency
- [ ] 能解释为什么自定义版通常比 PyTorch 慢（缺失向量化/warp 级优化）
- [ ] 能对照昇腾解释自定义算子集成的方式（C++ Extension vs aclOpCompile）

---

## Day 20（周六）：端到端 Profiling

> **今日目标**：对 Mini 引擎做端到端 profiling，用 Nsight Systems 看时间线、Nsight Compute 分析关键 kernel，定位 memory-bound 算子，列出 kernel fusion 机会。
> **时间分配**：6小时全天投入（nsys 采集2h + ncu 分析2h + fusion 分析2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"如何做端到端 profiling 定位瓶颈"是系统优化的标准流程

---

### 任务1：Nsight Systems 采集时间线（2小时）

#### 采集命令

```bash
# 用 nsys 采集 Mini 引擎的时间线
nsys profile -o mini_engine_timeline \
  --trace=cuda,nvtx \
  python mini_engine.py

# 生成 .nsys-rep 文件，用 Nsight Systems GUI 打开
# 或命令行导出关键信息
nsys stats -t cuda_gpu_kern_sum mini_engine_timeline.nsys-rep
```

#### 观察重点

1. **kernel 时间线排列**：观察 LayerNorm → QKV GEMM → Softmax → PV GEMM → ... 的顺序
2. **kernel 间隙（gap）**：相邻 kernel 之间的空白 = launch overhead
3. **GEMM vs element-wise 的时间占比**：GEMM（mm）应该占大头（Prefill 阶段）
4. **CUDA Stream 利用率**：是否所有 kernel 都在 default stream（串行）还是多 stream（并行）

#### 时间线分析表

| 观察项 | 预期（Prefill N=1024） | 含义 |
|--------|----------------------|------|
| GEMM kernel 总时间占比 | > 60% | compute-bound，GEMM 主导 |
| Softmax/LayerNorm 占比 | 10-20% | memory-bound，占比不应过高 |
| kernel 间隙占比 | < 10% | launch overhead 可接受 |
| 最大单 kernel 时间 | cuBLAS GEMM | GEMM 是最重算子 |

#### nsys 命令行输出解读

```bash
# 统计 GPU kernel 总时间
nsys stats -t cuda_gpu_kern_sum mini_engine_timeline.nsys-rep

# 预期输出（按时间降序）
# Time(%)  Total Time   Instances  Avg         Module      Kernel
# -------- -----------  ---------  --------    ----------  ------
#   45.2   1.234 ms     20         61.7 us     libcublas   ...gemm...
#   12.1   0.331 ms     10         33.1 us     my_ops      layernorm_kernel
#    8.5   0.232 ms     5          46.4 us     my_ops      softmax_kernel
#    ...
```

---

### 任务2：Nsight Compute 分析关键 kernel（2小时）

#### 分析目标

对 Mini 引擎中的 3 类 kernel 做 ncu 分析：
1. **Softmax kernel**（自定义）—— 预期 memory-bound
2. **LayerNorm kernel**（自定义）—— 预期 memory-bound
3. **cuBLAS GEMM**（官方）—— 预期 compute-bound（Prefill）

#### ncu 命令

```bash
# 编译带 lineinfo 的版本（ncu Source View 需要）
nvcc -o ... -lineinfo

# profile 自定义 kernel
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  smsp__average_warps_issue_stalled_long_scoreboard.pct,\
  gpu__time_duration.sum \
  --kernel-name regex:"softmax_kernel|layernorm_kernel" \
  python mini_engine.py

# profile cuBLAS GEMM（需允许 profile 第三方库）
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:"gemm" \
  python mini_engine.py
```

#### 预期结果与分析

| Kernel | SM Throughput | DRAM Throughput | 瓶颈类型 | 主要 Stall |
|--------|--------------|-----------------|---------|-----------|
| Softmax（自定义） | ~15-25% | ~50-70% | **Memory-bound** | Long Scoreboard |
| LayerNorm（自定义） | ~15-25% | ~50-70% | **Memory-bound** | Long Scoreboard |
| cuBLAS GEMM | ~70-85% | ~30-50% | **Compute-bound** | Math Pipe Throttle |

**分析结论**：
- Softmax/LayerNorm 的 DRAM Throughput >> SM Throughput → memory-bound，符合预期
- cuBLAS GEMM 的 SM Throughput >> DRAM Throughput → compute-bound，符合预期
- 自定义 kernel 的 DRAM Throughput 未达 80%+ → 带宽未充分利用，向量化加载可提升

#### 昇腾对照

| ncu 指标 | msprof 对应指标 | 含义 |
|---------|------------|------|
| SM Throughput | AI Core Utilization | 计算单元利用率 |
| DRAM Throughput | Memory Bandwidth Utilization | 显存带宽利用率 |
| Long Scoreboard Stall | Memory Access Stall | 内存访问等待 |
| Math Pipe Throttle | Compute Pipe Stall | 计算单元饱和 |

**关键差异**：ncu 把指标分散到多个视图；msprof 在一个 timeline 中集成 AI Core 利用率和 Memory 带宽。两者分析思路一致。

---

### 任务3：Kernel Fusion 机会分析（2小时）

#### Fusion 候选清单

基于 nsys 时间线，找出可融合的相邻算子：

| Fusion 候选 | 当前开销 | 融合后收益 | 实现难度 |
|------------|---------|-----------|---------|
| **LayerNorm + QKV GEMM** | LayerNorm 写 (B,N,d) 到 HBM，GEMM 再读 | 省去 (B,N,d) 一次读写 | 高（需融合 LN+GEMM） |
| **Softmax + Dropout** | Softmax 写 P，Dropout 读 P 再写 | 省去 P 一次读写 | 低（element-wise 融合） |
| **Bias + Activation** | GEMM 写结果，加 bias，过 GELU | 省去中间结果 | 中（epilogue fusion） |
| **Residual Add + LayerNorm** | Add 写结果，LN 读结果 | 省去一次读写 | 中 |

#### 优先级排序

1. **最高优先级**：LayerNorm + QKV GEMM（Prefill 阶段 GEMM 前的 LN 是 memory-bound，融合后省 HBM 读写）
2. **次高优先级**：Softmax + Dropout（如果有 dropout，融合简单收益大）
3. **中优先级**：GEMM + Bias + GELU（cuBLAS 的 epilogue fusion 或 CUTLASS 支持）

#### Fusion 收益估算

以 LayerNorm + QKV GEMM 为例（B=1, N=1024, d=512, FP32）：
```
未融合：
  LayerNorm: 读 x(2MB) + 写 y(2MB) = 4MB HBM IO
  QKV GEMM: 读 y(2MB) + 读 W(3MB) + 写 QKV(6MB) = 11MB HBM IO
  合计: 15MB

融合后：
  Fused LN+GEMM: 读 x(2MB) + 读 W(3MB) + 写 QKV(6MB) = 11MB
  节省: 4MB（LayerNorm 中间结果 y 的读写）
```

> 💡 **PyTorch 2.0 的 `torch.compile` 会自动做这些 fusion**。用 `torch.compile(model)` 后，nsys 时间线会显示 kernel 数量减少。

#### 验证 Fusion 效果

```python
# 用 torch.compile 自动融合
compiled_model = torch.compile(model_pytorch, mode="reduce-overhead")

# 重新 profile
nsys profile -o mini_engine_compiled python mini_engine_compiled.py

# 对比 kernel 数量：compiled 版应显著减少
nsys stats -t cuda_gpu_kern_sum mini_engine_compiled.nsys-rep
```

#### 练习题

**练习1（基础）**：用 nsys 采集未编译和 `torch.compile` 编译后的时间线，对比 kernel 总数。
> 提示：`torch.compile` 通常能把 LayerNorm+GEMM 等融合，kernel 数减少 30-50%。

**练习2（进阶）**：用 ncu 分析自定义 Softmax kernel 的 Long Scoreboard Stall 占比，解释为什么它高。
> 提示：Softmax 三遍扫描中，每遍都要从 HBM 读数据，Long Scoreboard = 等待 HBM 加载。

**练习3（综合）**：列出你的 Mini 引擎的 top3 瓶颈算子，并给出每个的优化方向。
> 提示：从 nsys 时间排序找 top3，再根据 ncu 判断是 memory-bound 还是 compute-bound。

---

### 今日面试题

**面试题1**：如何做端到端 profiling 定位 Transformer 推理的瓶颈？完整流程是什么？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **第一步（nsys 系统级）**：用 Nsight Systems 采集完整时间线，按 CUDA 时间排序找 top3 算子
2. **第二步（ncu kernel 级）**：对 top3 算子用 Nsight Compute 分析 SM Throughput 和 DRAM Throughput
3. **第三步（瓶颈判定）**：
   - DRAM Throughput >> SM Throughput → memory-bound → 优化方向：kernel fusion、向量化加载、减少 HBM 读写
   - SM Throughput >> DRAM Throughput → compute-bound → 优化方向：Tensor Core、增加 ILP、auto-tuning
4. **第四步（Stall 分析）**：看 Warp Stall Reasons 定位具体阻塞原因（Long Scoreboard = 等内存，Math Pipe = 计算饱和）
5. **第五步（Fusion 机会）**：从时间线找相邻 memory-bound 算子，评估融合收益
- **关键术语**：系统级 vs kernel 级、Roofline、Warp Stall、kernel fusion

**面试题2**：什么是 kernel fusion？为什么能提升性能？举一个 Transformer 中的例子。（⭐⭐⭐⭐ 高频）

**参考答案要点**：
- **定义**：把多个相邻算子合并成一个 kernel，避免中间结果写回 HBM
- **收益来源**：减少 HBM 读写次数。例如 A→B→C 三个算子，未融合要写 B 到 HBM 再读；融合后在 register/SRAM 中直接传递
- **Transformer 例子**：LayerNorm + QKV GEMM。未融合时 LayerNorm 输出写 HBM，GEMM 再读；融合后在 GEMM kernel 内部直接做归一化
- **限制**：① 只有相邻且数据依赖的算子能融合 ② 融合 kernel 可能增加 register/shared memory 压力，降低 occupancy ③ 复杂融合需要 CUTLASS/Triton 等工具
- **昇腾对照**：CANN 算子库已内置大量融合算子（如 LayerNorm+Matmul 融合算子）

---

### 今日自测清单

- [ ] 能用 nsys 采集 Mini 引擎时间线并导出 kernel 统计
- [ ] 能用 ncu 分析自定义 Softmax/LayerNorm 的 SM/DRAM Throughput
- [ ] 能判定 Softmax/LayerNorm 是 memory-bound（DRAM >> SM）
- [ ] 能判定 cuBLAS GEMM 是 compute-bound（SM >> DRAM）
- [ ] 能列出至少 3 个 kernel fusion 候选及其收益估算
- [ ] 能用 `torch.compile` 减少 kernel 数量并验证
- [ ] 能对照昇腾 msprof 解释 ncu 指标的映射关系
- [ ] 生成端到端 profiling 报告（含 top3 瓶颈算子 + 优化方向）

---

## Day 21（周日）：总结 Transformer 算子分类

> **今日目标**：将 Transformer 各组件按 arithmetic intensity 分类，整理本周所有产出，回顾 Week 3 知识图谱，为 Week 4 FlashAttention 深挖做铺垫。
> **时间分配**：6小时全天投入（算子分类2h + 产出整理2h + 面试复盘2h）
> **面试考察度**：⭐⭐⭐⭐ 高频，"Transformer 各算子是 compute-bound 还是 memory-bound"是系统优化的总结性考点

---

### 任务1：Transformer 算子分类表（2小时）

#### 分类标准

**Arithmetic Intensity (AI) = FLOPs / Bytes**

以 A100 为例（Ridge Point = Peak FLOP/s / Peak Bandwidth = 19.5T / 1.55T ≈ 12.6 FLOP/Byte）：
- AI < 12.6 → **Memory-bound**（数据喂不饱计算单元）
- AI > 12.6 → **Compute-bound**（算力是瓶颈）

#### Prefill 阶段算子分类

| 算子 | FLOPs | Bytes | AI (FLOP/Byte) | 瓶颈类型 | 原因 |
|------|-------|-------|----------------|---------|------|
| QKV GEMM (B,N,d)×(d,3d) | 2·B·N·d·3d | B·N·d + 3d² + 3B·N·d | ~6d/8 ≈ 384 (d=512) | **Compute** | 大矩阵乘，高 AI |
| QK^T GEMM (B,N,d)×(d,N) | 2·B·N²·d | 2B·N·d + B·N² | ~2d/(1+N/d) | **Compute**（N 大时） | N² 计算量大 |
| Attention Softmax | ~3·B·N² | 2·B·N² | ~1.5 | **Memory** | element-wise，低 AI |
| PV GEMM (B,N,N)×(N,d) | 2·B·N²·d | B·N² + B·N·d | ~2d/(1+d/N) | **Compute**（N 大时） | 大矩阵乘 |
| Output GEMM (B,N,d)×(d,d) | 2·B·N·d² | 2B·N·d + d² | ~d/2 ≈ 256 | **Compute** | 大矩阵乘 |
| LayerNorm | ~5·B·N·d | 2·B·N·d | ~2.5 | **Memory** | element-wise + reduce |
| FFN GEMM1 (B,N,d)×(d,4d) | 2·B·N·d·4d | B·N·d + 4d² + 4B·N·d | ~8d/10 ≈ 410 | **Compute** | 大矩阵乘 |
| GELU | ~4·B·N·4d | 2·B·N·4d | ~2 | **Memory** | element-wise |
| FFN GEMM2 (B,N,4d)×(4d,d) | 2·B·N·4d·d | 4B·N·d + 4d² + B·N·d | ~8d/10 ≈ 410 | **Compute** | 大矩阵乘 |

> 💡 **关键洞察**：Prefill 阶段 **GEMM 是 compute-bound，softmax/layernorm/gelu 是 memory-bound**。整体性能由 GEMM 主导（时间占比 60%+），所以 Prefill 是 compute-bound。

#### Decode 阶段算子分类（M=1，关键差异）

| 算子 | FLOPs | Bytes | AI (FLOP/Byte) | 瓶颈类型 | 与 Prefill 差异 |
|------|-------|-------|----------------|---------|---------------|
| QKV GEMM (B,1,d)×(d,3d) | 2·B·d·3d | B·d + 3d² + 3B·d | ~6d/(4+3d/B) | **Memory**（M=1） | GEMM 退化为向量×矩阵 |
| QK^T (B,1,d)×(d,N) | 2·B·N·d | B·d + B·N·d | ~2d/(1+d/N) | **Memory** | 计算量小但读 KV Cache 大 |
| Attention Softmax | ~3·B·N | 2·B·N | ~1.5 | **Memory** | 不变 |
| PV GEMM (B,1,N)×(N,d) | 2·B·N·d | B·N + B·N·d | ~2d/(1+d) | **Memory**（M=1） | 同上 |
| LayerNorm | ~5·B·d | 2·B·d | ~2.5 | **Memory** | 不变 |

> 💡 **关键洞察**：Decode 阶段 **几乎所有算子都是 memory-bound**（M=1 导致 GEMM 的 AI 骤降）。这就是为什么 Decode 是 memory-bound，优化重点是减少 HBM 读写（KV Cache）和 launch overhead（CUDA Graph）。

#### Prefill vs Decode 总览

| 维度 | Prefill | Decode |
|------|---------|--------|
| 主导算子类型 | GEMM（compute-bound） | KV Cache 读取（memory-bound） |
| 优化重点 | Tensor Core、FlashAttention | KV Cache、CUDA Graph、Continuous Batching |
| SM 利用率 | 高（60-85%） | 低（10-30%） |
| 单 token 延迟 | 低（并行处理 N 个 token） | 高（串行生成，每次 1 token） |
| 吞吐量瓶颈 | 算力 | 显存带宽 |

---

### 任务2：本周产出整理（2小时）

#### 产出清单

| 产出物 | 文件 | 验收标准 |
|--------|------|---------|
| Transformer profiler trace | trace_transformer.py + trace_*.json | Prefill/Decode 算子时间表 |
| Softmax + LayerNorm Kernel | softmax_layernorm.cu | 与 CPU 误差 < 1e-5 |
| 标准 Attention Kernel | attention_naive.cu | 与 CPU 误差 < 1e-3 |
| Mini 引擎 | mini_engine.py | 自定义算子版端到端 PASS |
| Profiling 报告 | mini_engine_timeline.nsys-rep + ncu 报告 | 含 top3 瓶颈分析 |
| 算子分类表 | （本文件） | Prefill/Decode 分类完整 |

#### GitHub 仓库结构建议

```
week3-transformer/
├── day15-trace/
│   ├── trace_transformer.py
│   ├── trace_prefill.json
│   └── trace_decode.json
├── day16-kernels/
│   ├── softmax_layernorm.cu
│   └── README.md
├── day17-source-analysis/
│   └── notes.md          # PyTorch/FT 源码分析笔记
├── day18-attention-io/
│   ├── attention_naive.cu
│   └── io_analysis.md     # HBM IO 量化表
├── day19-mini-engine/
│   ├── mini_engine.py
│   └── my_ops.cpp
├── day20-profiling/
│   ├── mini_engine_timeline.nsys-rep
│   └── profiling_report.md
└── day21-summary/
    └── operator_classification.md
```

#### 性能对比报告模板（profiling_report.md）

```markdown
# Week 3 Profiling 报告

## 测试环境
- GPU: [你的 GPU 型号]
- CUDA: 12.x
- PyTorch: 2.x

## Prefill 阶段（N=1024, d=512）

### Top3 算子（按 CUDA 时间）
| 排名 | 算子 | 时间占比 | 瓶颈类型 |
|------|------|---------|---------|
| 1 | cuBLAS GEMM (QKV) | xx% | Compute |
| 2 | cuBLAS GEMM (FFN) | xx% | Compute |
| 3 | Softmax | xx% | Memory |

### Memory-bound 算子优化方向
- Softmax: 向量化加载、kernel fusion
- LayerNorm: Welford 一次 reduce、float4

## Decode 阶段（N=1）
- 总 SM 利用率: xx%（远低于 Prefill）
- 瓶颈: KV Cache 读取 + launch overhead

## Kernel Fusion 机会
1. LayerNorm + QKV GEMM（节省 xx MB HBM IO）
2. ...
```

---

### 任务3：Week 4 预热 + 面试复盘（2小时）

#### Week 4 FlashAttention 预热

本周我们分析了标准 Attention 的 O(N²) IO 问题。Week 4 将深入：
1. **FlashAttention 算法**：Tiling + Online Softmax（Week 2 Day 12 已学简化版，Week 4 学完整版）
2. **FlashAttention-2 改进**：减少非 matmul FLOPs、更好的 work partitioning
3. **手写完整 FlashAttention kernel**：支持 batch、multi-head、不同 seq_len
4. **性能对比**：标准 Attention vs 手写 FlashAttention vs 官方 FlashAttention

**本周铺垫的关键概念**：
- ✅ 标准 Attention 的 O(N²) IO（Day 18）→ Week 4 用 FlashAttention 解决
- ✅ Online Softmax 三公式（Week 2 Day 12）→ Week 4 完整实现
- ✅ Softmax 的 memory-bound 本质（Day 16）→ Week 4 在 SRAM 中做 softmax
- ✅ Warp Shuffle reduce（Week 2 Day 8）→ Week 4 用于 online softmax 的分块 reduce

#### 面试复盘

回顾本周 10 道面试题，自问自答：

1. Prefill vs Decode 的区别？为什么 Decode 是 memory-bound？
2. Transformer 单层算子分类（compute vs memory）？
3. Softmax 为什么要减 max？
4. LayerNorm 需要几次 reduce？
5. 为什么 Softmax/LayerNorm 是 memory-bound？
6. PyTorch Softmax 为什么 D 小时用 warp 级？
7. FP16 reduce 为什么要用 FP32？
8. 标准 Attention 的 IO 复杂度？O(N²) 来源？
9. 如何做端到端 profiling 定位瓶颈？
10. 什么是 kernel fusion？举例。

---

### 今日面试题

**面试题1**：Transformer 的 Prefill 和 Decode 阶段，分别是什么 bound？为什么？（⭐⭐⭐⭐⭐ 必考，总结性考点）

**参考答案要点**：
- **Prefill 是 compute-bound**：输入 N 个 token，所有 GEMM 是大矩阵乘，arithmetic intensity 高（如 QKV GEMM 的 AI ≈ 384 >> Ridge Point 12.6）。SM 利用率高（60-85%），优化重点是 Tensor Core 和 FlashAttention
- **Decode 是 memory-bound**：每次只生成 1 个 token（M=1），GEMM 退化为向量×矩阵，arithmetic intensity 骤降（如 QKV GEMM 的 AI 从 384 降到 ~1）。SM 利用率低（10-30%），大部分时间在等 HBM 读写 KV Cache
- **根本原因**：M=1 导致 GEMM 的计算量（与 M 成正比）远小于数据读取量（与 N·d 成正比），AI = FLOPs/Bytes 极低
- **优化方向**：Prefill 优化算力（Tensor Core），Decode 优化访存（KV Cache、PagedAttention）和 launch overhead（CUDA Graph、Continuous Batching）

**面试题2**：给一个未知算子，如何判断它是 compute-bound 还是 memory-bound？（⭐⭐⭐⭐ 高频）

**参考答案要点**：
1. **理论计算**：算 FLOPs 和 Bytes，AI = FLOPs/Bytes，与 Ridge Point 比较
2. **工具验证**：用 ncu 看 SM Throughput 和 DRAM Throughput
   - DRAM Throughput >> SM Throughput → memory-bound
   - SM Throughput >> DRAM Throughput → compute-bound
3. **Roofline 定位**：在 Roofline 图上标出算子位置
4. **经验法则**：
   - element-wise（relu、layernorm、softmax）→ 几乎总是 memory-bound
   - 大 GEMM（M,N,K 都大）→ 通常 compute-bound
   - 小 GEMM（M=1 或某维很小）→ 通常 memory-bound
   - reduction（sum、max）→ memory-bound

---

### 今日自测清单

- [ ] 能列出 Prefill 阶段算子分类表（GEMM=compute，softmax/LN=memory）
- [ ] 能列出 Decode 阶段算子分类表（几乎全是 memory-bound）
- [ ] 能解释为什么 Decode 阶段 GEMM 变成 memory-bound（M=1 导致 AI 骤降）
- [ ] 能计算给定算子的 arithmetic intensity 并判定 bound 类型
- [ ] 能用 ncu 的 SM/DRAM Throughput 验证理论判定
- [ ] 整理本周所有代码到 GitHub 仓库
- [ ] 生成 profiling 报告（含 top3 瓶颈 + 优化方向）
- [ ] 能口述本周 10 道面试题的答案要点
- [ ] 理解 Week 4 FlashAttention 如何解决标准 Attention 的 O(N²) 问题

---

## 附录A：第3周面试题汇总

| 题号 | 题目 | 考察频率 | 相关天数 | 难度 |
|------|------|---------|---------|------|
| 1 | Prefill 和 Decode 的区别？为什么 Decode 是 memory-bound？ | ⭐⭐⭐⭐⭐ | Day 15, 21 | 中 |
| 2 | Transformer 单层包含哪些算子？compute 还是 memory bound？ | ⭐⭐⭐ | Day 15, 21 | 中 |
| 3 | Softmax 为什么要减 max？不减会怎样？ | ⭐⭐⭐⭐ | Day 16 | 易 |
| 4 | LayerNorm 需要几次 reduce？每次 reduce 什么？ | ⭐⭐⭐⭐ | Day 16 | 中 |
| 5 | 为什么 Softmax/LayerNorm 是 memory-bound？如何优化？ | ⭐⭐⭐⭐ | Day 16 | 中 |
| 6 | PyTorch Softmax 为什么 D 小时用 warp 级实现？ | ⭐⭐⭐ | Day 17 | 中 |
| 7 | FP16 训练时 Softmax/LayerNorm 的 reduce 为什么要用 FP32？ | ⭐⭐⭐⭐ | Day 17 | 中 |
| 8 | 标准 Attention 的 IO 复杂度？O(N²) 来源？ | ⭐⭐⭐⭐⭐ | Day 18 | 高 |
| 9 | 标准 Attention softmax 部分的 AI 是多少？什么 bound？ | ⭐⭐⭐⭐ | Day 18 | 中 |
| 10 | 如何做端到端 profiling 定位 Transformer 瓶颈？ | ⭐⭐⭐⭐ | Day 20 | 中 |
| 11 | 什么是 kernel fusion？举例 Transformer 中的 fusion | ⭐⭐⭐⭐ | Day 20 | 中 |
| 12 | 如何把自定义 CUDA 算子集成到 PyTorch？ | ⭐⭐⭐ | Day 19 | 中 |
| 13 | 给未知算子，如何判断 compute-bound 还是 memory-bound？ | ⭐⭐⭐⭐ | Day 21 | 中 |
| 14 | Prefill 和 Decode 分别是什么 bound？为什么？ | ⭐⭐⭐⭐⭐ | Day 21 | 中 |

---

## 附录B：Transformer 算子计算强度速查表

### Prefill 阶段（B=1, N=1024, d=512, FP32）

| 算子 | FLOPs | Bytes | AI (FLOP/Byte) | Bound | 优化方向 |
|------|-------|-------|----------------|-------|---------|
| QKV GEMM | 2·N·d·3d = 1.6G | 8·N·d = 4MB | ~384 | Compute | Tensor Core |
| QK^T GEMM | 2·N²·d = 1.1G | N²·4 = 4MB | ~256 | Compute | FlashAttention |
| Softmax | 3·N² = 3.1M | 2·N²·4 = 8MB | ~0.4 | **Memory** | FlashAttention (SRAM) |
| PV GEMM | 2·N²·d = 1.1G | N²·4 = 4MB | ~256 | Compute | FlashAttention |
| LayerNorm | 5·N·d = 2.6M | 2·N·d·4 = 8MB | ~0.3 | **Memory** | Fusion |
| GELU | 4·N·4d = 8.4M | 2·N·4d·4 = 32MB | ~0.25 | **Memory** | Epilogue fusion |

### Decode 阶段（B=1, N=1 生成, KV Cache 长度 L=1024, d=512）

| 算子 | FLOPs | Bytes | AI (FLOP/Byte) | Bound | 优化方向 |
|------|-------|-------|----------------|-------|---------|
| QKV GEMM (M=1) | 2·d·3d = 1.6M | 4·d·4 = 8KB | ~200 | Memory* | KV Cache |
| QK^T (1×L) | 2·L·d = 1M | L·d·4 = 2MB | ~0.5 | **Memory** | PagedAttention |
| Softmax (1×L) | 3·L = 3K | 2·L·4 = 8KB | ~0.4 | **Memory** | FlashAttention |
| PV GEMM (1×L) | 2·L·d = 1M | L·d·4 = 2MB | ~0.5 | **Memory** | PagedAttention |
| LayerNorm | 5·d = 2.6K | 2·d·4 = 4KB | ~0.6 | **Memory** | Fusion |

> *Decode 的 GEMM 虽然理论 AI 较高，但因矩阵极小（M=1），SM 无法充分利用，实际表现为 memory-bound。

---

## 附录C：第3周关键公式汇总

**1. Safe Softmax**
```
m = max(xj)
yi = exp(xi - m) / Σ exp(xj - m)
```

**2. LayerNorm**
```
μ = (1/D) Σ xi
σ² = (1/D) Σ (xi - μ)²
yi = γi · (xi - μ) / sqrt(σ² + ε) + βi
```

**3. 标准 Attention HBM 读写量**
```
Step 1 (S=QK^T): 2Nd + N²
Step 2 (P=softmax(S)): 2N²
Step 3 (O=PV): N² + 2Nd
总计: 3N² + 4Nd  →  O(N²) when N >> d
```

**4. Arithmetic Intensity**
```
AI = FLOPs / Bytes
Memory-bound: AI < Ridge Point (≈12.6 on A100)
Compute-bound: AI > Ridge Point
```

**5. Online Softmax 三公式（Week 4 预习）**
```
m_new = max(m, max(xj))
l_new = l * exp(m - m_new) + Σ exp(xj - m_new)
o_new = o * (l * exp(m - m_new) / l_new) + (exp(xj - m_new) / l_new) * vj
```

**6. Ridge Point（A100 示例）**
```
Ridge Point = Peak FLOP/s / Peak Bandwidth
            = 19.5 TFLOP/s / 1.55 TB/s
            ≈ 12.6 FLOP/Byte
```

---

## 附录D：昇腾→CUDA 第3周概念映射总表

| 维度 | CUDA 概念 | 昇腾 CANN 概念 | 差异说明 | 迁移难度 |
|------|---------|------------|---------|---------|
| **Prefill/Decode** | 两阶段划分（模型层） | 两阶段划分（模型层） | 与硬件无关，跨平台一致 | ★ |
| **torch.profiler** | torch.profiler + nsys | msprof | 两者都提供算子级时间线 | ★★ |
| **Warp Shuffle reduce** | `__shfl_down_sync` | `__reduce_add` (Ascend C) | CUDA 手写循环；昇腾高级 API | ★★★ |
| **Block reduce** | warp + smem + warp0 | Vector Unit + L0 Buffer | 两级结构一致 | ★★ |
| **Softmax 算子** | 手写 / PyTorch ATen | Ascend C 内置 Softmax | 昇腾已内置，无需手写 | ★★ |
| **LayerNorm 算子** | 手写 / FasterTransformer | Ascend C 内置 LayerNorm | 昇腾已内置 | ★★ |
| **向量化加载** | float4 / half2 | Vector Unit 向量指令 | 昇腾天然向量化 | ★★ |
| **FP32 reduce 保精度** | FP16→FP32→FP16 | 同策略 | 跨平台一致 | ★ |
| **HBM（Global Memory）** | HBM | DDR/HBM | 两者都面临带宽瓶颈 | ★ |
| **片上 SRAM** | Shared Memory | L0 Buffer / UB | FlashAttention tile 驻留 | ★ |
| **标准 Attention O(N²)** | 物化 S/P | CANN 已用 FlashAttention | 昇腾无需手写标准版 | ★★ |
| **Kernel Fusion** | 手写 / torch.compile | CANN 内置融合算子 | 昇腾算子库已融合 | ★★ |
| **SM Throughput** | ncu 指标 | AI Core Utilization | 含义一致 | ★ |
| **DRAM Throughput** | ncu 指标 | Memory Bandwidth Utilization | 含义一致 | ★ |
| **自定义算子集成** | C++ Extension | aclOpCompile + 注册 | 机制类似 | ★★★ |

---

## 附录E：性能诊断速查表（Week 3 专用）

| 现象 | 可能原因 | 检查方法 | 解决方案 |
|------|---------|---------|---------|
| 自定义 Softmax 比 PyTorch 慢 | 缺向量化加载 | ncu 看 DRAM Throughput | 加 float4 |
| Softmax 结果有 NaN | 未减 max | 检查代码有无 max subtraction | 用 safe softmax |
| LayerNorm 误差大 | reduce 精度不足 | 用 FP32 reduce | 避免 FP16 累加 |
| 标准 Attention OOM | N 太大，S/P 物化 | 看 N² 显存占用 | 用 FlashAttention |
| Decode 阶段很慢 | M=1，SM 空闲 | nsys 看 SM 利用率 | CUDA Graph + KV Cache |
| kernel 间隙大 | launch overhead | nsys 时间线看 gap | torch.compile |
| 自定义算子集成报错 | stream 未传 | 检查 `getCurrentCUDAStream` | 传 stream 给 kernel |
| ncu HBM 实测 >> 理论 | cache miss / 额外访问 | 对比理论值 | 误差 < 30% 属正常 |

---

> 💡 **Week 3 总结**：本周我们从"GPU 视角"理解了 Transformer——它不是黑盒，而是 GEMM（compute）+ Softmax/LayerNorm（memory）+ Attention（混合）的组合。Prefill 是 compute-bound（GEMM 主导），Decode 是 memory-bound（M=1 导致 AI 骤降）。标准 Attention 的 O(N²) IO 问题为 Week 4 FlashAttention 埋下伏笔。掌握算子的 arithmetic intensity 分类，就能在任何模型中快速判断瓶颈并给出优化方向。

