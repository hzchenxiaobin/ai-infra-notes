# Day 6（周六）：Profiling 与性能调优

> **今日目标**：用 ncu/nsys 分析 MoE 层的性能瓶颈，对比朴素实现 / Triton / vLLM 三版本，产出性能报告
> **面试考察度**：⭐⭐⭐⭐ 实践级，能解读 ncu 指标并定位 MoE 瓶颈

---

### 学习任务 1：nsys 抓 MoE 层时序（45 分钟）

```bash
# 用 nsys 抓 vLLM 或自实现 MoE 的 timeline
nsys profile -o moe_profile \
    --trace=cuda,nvtx,nccl \
    python3 benchmark/run_moe.py
nsys stats moe_profile.nsys-rep
```

#### 关键观察点

| 观察对象 | 期望 | 异常信号 |
|----------|------|----------|
| GEMM kernel 与 all-to-all 的交替 | 计算/通信 overlap | 长串连续 all-to-all（无 overlap） |
| 单个 Grouped GEMM 耗时 | 占 MoE 层 60%+ | 占比过低说明 dispatch/combine 开销大 |
| all-to-all 等待时间 | < GEMM 耗时 | 等待长 → 通信 bound |
| 专家负载分布 | 各专家 token 数接近均值 | 长尾专家 >> 均值 → 负载不均 |

### 学习任务 2：ncu 分析 Grouped GEMM kernel（45 分钟）

```bash
# profile Triton Grouped GEMM kernel
ncu --set full \
    --kernel-name "grouped_gemm" \
    --launch-skip 3 --launch-count 1 \
    -o grouped_gemm.ncu-rep \
    python3 benchmark/run_grouped_gemm.py
```

#### 关键指标

| 指标 | 含义 | MoE 目标 |
|------|------|----------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐 | > 60%（Grouped GEMM 难达 80%+） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽 | < 50%（应计算 bound） |
| `smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | Tensor Core 利用率 | > 50% |
| `launch__occupancy_limit_registers.avg.pct_of_peak_sustained_active` | 寄存器占用 | < 80% |
| `stall_long_sb` | 长延迟 stall | 排查 smem bank conflict |

### 学习任务 3：性能对比 Benchmark（45 分钟）

在 `benchmark/` 下创建对比脚本：

```python
# benchmark/compare_moe.py —— MoE 性能对比
import torch, time
from kernels.naive_moe import NaiveMoE
from kernels.triton_moe import TritonMoE

CONFIGS = [
    # (T, d_model, d_ff, num_experts, top_k)
    (1024, 4096, 14336, 8, 2),     # Mixtral 配置
    (4096, 5120, 8192, 64, 6),     # 类 DeepSeek
    (8192, 4096, 14336, 8, 2),
]

def bench(fn, *args, n=100, warmup=10):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000   # ms

for cfg in CONFIGS:
    naive = NaiveMoE(*cfg).cuda()
    triton = TritonMoE(*cfg).cuda()
    x = torch.randn(cfg[0], cfg[1], device='cuda')
    t_naive = bench(naive, x)
    t_triton = bench(triton, x)
    print(f"{cfg}: naive={t_naive:.2f}ms triton={t_triton:.2f}ms "
          f"speedup={t_naive/t_triton:.2f}x")
```

### 学习任务 4：调参实验（45 分钟）

| 实验 | 变量 | 预期 |
|------|------|------|
| 1 | top_k: 1 vs 2 vs 4 vs 6 | k 越大，Grouped GEMM 越大，dispatch 开销摊薄 |
| 2 | num_experts: 8 vs 64 vs 256 | 专家越多，每专家 token 越少，Grouped GEMM 越碎 |
| 3 | BLOCK_M: 16 vs 32 vs 64 vs 128 | 找 Triton kernel 最优 tile |
| 4 | 负载不均：均匀 vs Zipf 分布 | 长尾专家拖慢整体 |

### 今日检查清单

- [ ] 用 `nsys` 抓出 MoE 层的 GEMM/all-to-all 交替时序图
- [ ] 用 `ncu` 记录 Grouped GEMM 的 5 个关键指标
- [ ] `compare_moe.py` 跑出 3 种配置的 naive vs Triton 对比
- [ ] 完成 2 个调参实验并记录结论
- [ ] 性能报告写入 `benchmark/report.md`

---

