# Day 7（周日）：Profiling 调优与面试复盘

> **今日目标**：用 ncu/nsys 分析 DeepGEMM kernel 的性能瓶颈，对比 DeepGEMM / cuBLASLt / CUTLASS 三版本，完成面试复盘
> **面试考察度**：⭐⭐⭐⭐ 实践级，能解读 ncu 指标并定位瓶颈

---

### 学习任务 1：ncu 性能分析（45 分钟）

```bash
# 启用 lineinfo 便于 ncu 对应源码
DG_JIT_WITH_LINEINFO=1 python3 tests/test_fp8_fp4.py

# profile DeepGEMM kernel
ncu --set full \
    --kernel-name "sm90_fp8_gemm" \
    --launch-skip 5 --launch-count 1 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    python3 tests/test_fp8_fp4.py
```

#### 关键指标

| 指标 | 含义 | H800 FP8 目标 |
|------|------|---------------|
| `sm__throughput` | SM 吞吐占比 | > 85% |
| `dram__throughput` | DRAM 带宽占比 | < 30%（应计算 bound） |
| `...tensor_op_hmma` | Tensor Core 利用率 | > 85% |
| `...stalled_long_scoreboard` | 长延迟 stall（等 SMEM/TMA） | < 10% |
| `...bank_conflicts...shared` | SMEM bank conflict | 接近 0（swizzle 生效） |

### 学习任务 2：性能对比 Benchmark（45 分钟）

在 `benchmark/` 下创建对比脚本：

```python
# benchmark/compare_fp8_gemm.py
import torch, deep_gemm
from deep_gemm.testing import bench_kineto

SIZES = [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384)]

for M, N, K in SIZES:
    a = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
    # ... 准备 SFA/SFB ...

    d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    # DeepGEMM
    t_dg = bench_kineto(lambda: deep_gemm.fp8_gemm_nt((a, sfa), (b, sfb), d), 'gemm_')
    # cuBLASLt（DeepGEMM 自带 wrapper）
    t_cb = bench_kineto(lambda: deep_gemm.cublaslt_gemm_nt(a, b, d), 'nvjet')

    tflops_dg = 2*M*N*K/t_dg/1e12
    tflops_cb = 2*M*N*K/t_cb/1e12
    print(f"{M}x{N}x{K}: DeepGEMM={tflops_dg:.0f} TFLOPS ({t_dg*1e6:.0f}us) | "
          f"cuBLAS={tflops_cb:.0f} TFLOPS ({t_cb*1e6:.0f}us) | "
          f"speedup={t_cb/t_dg:.2f}x")
```

### 学习任务 3：nsys 抓 Mega MoE 时序（30 分钟）

```bash
# Mega MoE 的通信/计算 overlap 时序
nsys profile -o mega_moe_profile \
    --trace=cuda,nvtx \
    python3 tests/test_mega_moe.py
nsys stats mega_moe_profile.nsys-rep
```

观察点：
- NVLink 通信与 WGMMA 是否 overlap
- 单个 mega kernel 是否覆盖了传统 5 步的全部时间
- 对比传统分步 MoE 的 kernel 数量与总耗时

### 学习任务 4：面试题复盘（60 分钟）

#### 高频面试题

1. **DeepGEMM 为什么不直接用 CUTLASS？**
   - CUTLASS 模板太重，迭代慢；DeepGEMM 目标是"小而可读"
   - DeepGEMM 借鉴 CuTe 的 TMA/WGMMA wrapper，但不用 Layout 代数与 CollectiveBuilder
   - MoE 的 M-grouped + Mega MoE 融合在通用框架里难做精

2. **DeepGEMM 的 JIT 有什么优势？**
   - 针对每个具体 shape 生成最优 kernel（BLOCK_M/N/K 等编译期常量）
   - 安装时不编译，首次调用按需编译并缓存
   - NVRTC 模式比 NVCC 快 10x

3. **per-128-channel scaling 为什么比 per-tensor 精度高？**
   - per-tensor：1 个 scale 压全矩阵，outlier 拉大整体 scale
   - per-128-channel：每 128 列独立 scale，outlier 只影响局部
   - BLOCK_K=128 对齐 WGMMA K=32，4 次 WGMMA 后乘一次 scale，开销最小

4. **SM90 的 scale 是 FP32，SM100 为什么用 UE8M0？**
   - SM100 的 FP4 精度低，需要更细粒度 scale，scale 数组变大
   - UE8M0 把 scale 从 4 字节压到 1 字节，避免 scale 成为带宽瓶颈
   - UE8M0 只存指数（MX 格式），4 个 pack 成一个 int

5. **WGMMA 为什么比 Ampere mma.sync 强？**
   - 形状大（m64nNk32 vs m16n8k16）
   - 操作数直接从 SMEM 读（用 GmmaDescriptor），省 register load
   - 异步，可连续发射填满 pipeline

6. **warp specialization 的 TMA/Math warpgroup 怎么同步？**
   - 双 barrier：`full_barriers`（TMA→Math）+ `empty_barriers`（Math→TMA）
   - TMA `arrive_and_expect_tx` 通知字节数，Math `wait` 等数据满
   - Math `arrive` 通知 buffer 可覆盖，TMA `wait` 等释放
   - 多 stage 流水让 TMA 与 WGMMA 在不同 buffer 上同时进行

7. **`warpgroup_reg_dealloc/alloc` 解决什么问题？**
   - TMA warpgroup 几乎不用寄存器（24-40），Math warpgroup 需要大量累加器（232-240）
   - Hopper 允许运行时切寄存器上限，TMA 让出配额给 Math
   - Ampere 寄存器上限是编译期固定的，做不到

8. **DeepGEMM 的 Grouped GEMM 为什么只分 M 轴？**
   - MoE 的所有专家共享相同 [d_model, d_ff] shape，只有 token 数不同
   - 只分 M 轴让 N/K 固定，tile 调度统一，SM 利用率更高
   - CUTLASS 的通用 GemmGroup 允许每 group 任意 shape，但调度复杂

9. **Mega MoE 融合了什么，为什么能省时间？**
   - 融合 EP dispatch + Linear1 + SwiGLU + Linear2 + EP combine 成单 kernel
   - 把 NVLink 通信藏在 Tensor Core 计算里——传统 MoE 的 dispatch/combine SM 空闲
   - 需要多进程 + 对称内存 + PyTorch >= 2.9

10. **SM90 WGMMA 与 SM100 TCgen05 的区别？**
    - TCgen05 形状更大（m128 vs m64），支持 FP4
    - 都从 SMEM 读操作数，都异步
    - TCgen05 是 Blackwell 专属，WGMMA 是 Hopper 专属

11. **DeepGEMM 的持久化调度器解决什么问题？**
    - 一个 threadblock 串行处理多个 tile，减少 kernel launch 开销
    - 配合 Stream-K（K 维切分 + TMA_REDUCE_ADD）均衡小 M×N 的负载
    - 调度器支持 Normal / MGrouped / KGrouped / Masked 等多种 GemmType

12. **scale 乘法放在 WGMMA wait 之后、下一轮 TMA 之前，为什么？**
    - 这个位置把 scale 乘法藏进 TMA 搬运的影子（overlap）
    - scale 必须在 `warpgroup_arrive` 前从 SMEM 读完（避免下一 tile 污染）
    - 是 DeepGEMM 性能调优的关键细节

### 学习任务 5：总结与知识图谱（30 分钟）

#### 本周知识图谱

```
                    DeepGEMM（SM90+SM100 FP8/FP4/MoE）
                   /          |             \
              精度层          算子层          硬件层
              /  |  \         /  |  \        /  |  \
        FP8 e4m3 FP4 UE8M0  GEMM Grouped Mega TMA WGMMA TCgen05
        per-128  E2M1 MX    1D1D 1D2D  MoE  desc async  SM100
        channel  scale      cont mask k-grp      ↑        ↑
           |       |         |    |     |     mbarrier   FP4
           +-------+---------+----+-----+----+ reg_reconfig
                                |                |     |
                    warp specialization    persistent  swizzle
                    (TMA wg / Math wg)    + Stream-K  128B
                                |                |
                    ncu 调优（TC利用率/bank conflict/stall）
                                |
                DeepSeek-V3 MoE 训练/推理 / Mega MoE / SM100 FP4
```

#### 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)（v2.6.1） | 源码 | ⭐ 必读 |
| [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) | 论文 | ⭐ 必读 |
| [FlashAttention-3 论文精读](../../paper/flashattention3/README.md)（本仓库） | 论文 | ⭐ 必读 |
| [Hopper WGMMA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-group-matrix-multiply-instructions) | 文档 | ⭐ 必读 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | 源码 | 📌 推荐 |
| [CUTLASS 专题 Day 4](../cutlass/day4.md) 三层抽象 | 教程 | 📎 复习前置 |
| [CuTe 专题 Day 6](../cute/README.md) TMA + WGMMA | 教程 | 📎 复习前置 |
| [MoE 专题](../moe/README.md) Grouped GEMM | 教程 | 📎 关联 |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | 论文 | 📌 推荐 |

