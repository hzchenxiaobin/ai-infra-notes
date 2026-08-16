# GPU 执行审计报告

> 审计日期：2026-08-09
> 审计范围：`aiinfra/daily/`（10 周 70 天）、`aiinfra/topics/`（13 个专题）、`profiling/`（性能分析目录）
> 目标：识别所有需要 GPU 执行但尚未执行的教程/代码

## 1. 审计方法

### 1.1 判定"需要 GPU 执行"的标准

代码满足以下任一条件即视为需要 GPU：

- **CUDA `.cu` 文件**：需 `nvcc` 编译并在 GPU 上运行
- **Triton `@triton.jit` 脚本**：需 Triton JIT 编译器 + GPU
- **硬编码 `device="cuda"` 的 Python 脚本**：无 CPU 回退，必须 GPU
- **`assert torch.cuda.is_available()` 的脚本**：显式要求 GPU
- **使用 `torch.cuda.CUDAGraph` / `torch.cuda.Stream` 的脚本**：CUDA Graph/Stream 需 GPU
- **Markdown 中嵌入的 GPU 代码块**：教程预期在 GPU 上执行

**排除项**（不视为需要 GPU）：
- 纯标准库/纯 CPU 模拟脚本（如调度模拟器、摘要生成器）
- `device = "cuda" if torch.cuda.is_available() else "cpu"` 且教程重点不在 GPU 性能的脚本（已执行的标记但不算"未执行 GPU 代码"）
- C++ 专题的 `.cpp` 文件（纯 CPU C++ 面试题）

### 1.2 判定"已执行"的标准

按证据强度从高到低：

1. **`.ncu-rep` 文件**（最强）：Nsight Compute 二进制报告，只能由 `ncu --set full` 在 GPU 上实际运行 kernel 生成，无法手工伪造
2. **Chrome Trace JSON**：`torch.profiler.export_chrome_trace()` 导出的真实 GPU kernel 事件
3. **正确性验证表**：`week1/profiles/week1_profile_summary.md` 中的 17 kernel PASS/FAIL 表，含 `max_diff` 数值
4. **README 中粘贴的实测输出**：含具体时间戳、延迟数值、硬件型号
5. **`__pycache__` 存在**（弱证据）：文件被 import/运行过，但不区分 CPU/GPU 执行

### 1.3 已确认的执行环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| Compute Capability | 12.0 (sm_120) |
| CUDA 版本 | 12.8 |
| 执行日期 | 2026-08-04 ~ 2026-08-06 |

---

## 2. 已执行的 GPU 代码（供对照）

### 2.1 有 `.ncu-rep` 文件的 kernel（35 个）

| 位置 | Kernel | .ncu-rep 文件 |
|------|--------|--------------|
| profiling/week1/day1 | hello_gpu, vector_add | 2 |
| profiling/week1/day2 | occupancy_test | 1 |
| profiling/week1/day4 | transpose, bandwidth, transpose_tiles | 3 |
| profiling/week1/day5 | bank_conflict | 1 |
| profiling/week1/day6 | hello_gpu/occupancy/transpose/bank_conflict/matmul（复用） | 5 |
| profiling/week2/day1 | warp_reduce | 1 |
| profiling/week2/day4 | register_gemm, softmax_profile | 2 |
| profiling/week2/day5 | flash_attention | 1 |
| profiling/week2/day6 | integrated_gemm, histogram | 2 |
| profiling/week2/day7 | block_reduce, gemm_timed | 2 |
| profiling/week3/day2 | softmax_layernorm, sl_dscan | 2 |
| profiling/week3/day3 | softmax_layernorm_opt, warp_vs_block | 2 |
| profiling/leetgpu | 9 道 LeetGPU 题 | 9 |

### 2.2 正确性验证表中的 kernel（17 个，来自 `week1_profile_summary.md`）

| Kernel | 验证结果 | max_diff |
|--------|---------|---------|
| hello_gpu | PASS | N/A |
| transpose | PASS | < 1e-5 |
| bank_conflict | PASS | N/A |
| softmax_layernorm | PASS | 4.19e-09 |
| softmax_layernorm_opt | PASS | 1.12e-08 |
| attention_naive | PASS | < 1e-3 |
| warp_reduce | PASS | N/A |
| register_blocking_gemm | PASS | < 1e-2 |
| flash_attention (W2) | PASS | < 1e-3 |
| gemm_optimization_series | PASS | < 1e-2 |
| wmma_gemm | PASS | 1.00e-02 |
| flash_attention_v2 (W4) | PASS | 1.31e-04 |
| kv_cache | PASS | 9.96e-02 |
| paged_attention | PASS | 9.54e-07 |
| flash_decoding | **FAIL** | 3.43e-01 |
| triton_softmax | PASS | 7.45e-09 |
| triton_gemm | PASS | 0.00e+00 |

### 2.3 有实测输出粘贴的 README

| 位置 | 内容 |
|------|------|
| `daily/week1/profiles/week1_profile_summary.md` | RTX 5090 上 nsys 实测，GEMM 优化系列完整数据，WMMA 数据 |
| `daily/week2/day5/README.md` | 多流水线实测：Sequential 13.091ms → Multi-Stream 5.396ms，2.43x |
| `profiling/week2/day4/README.md` | register_gemm ncu 详细分析，SASS 级 stall 定位 |
| `profiling/week3/day1/README.md` | torch.profiler Chrome Trace，Prefill/Decode kernel 时间线 |
| `daily/week10/day2/README.md` | 真引擎模式 `--real` 实测输出（2026-08-06） |
| `daily/week10/day5/README.md` | 实测吞吐：~5694 tokens/s, TTFT ~2.9ms |

---

## 3. 需要 GPU 但未执行的代码

### 3.1 CUDA `.cu` kernel 文件（共 20 个）

#### Week 3 — Tensor Core WMMA/MMA 系列（6 个）

| 文件 | 路径 | 说明 |
|------|------|------|
| `wmma_gemm_tiled.cu` | `daily/week3/day2/kernels/` | WMMA Shared Memory Tiling GEMM |
| `mma_sync_gemm.cu` | `daily/week3/day3/kernels/` | mma.sync PTX GEMM |
| `cutlass_gemm_example.cu` | `daily/week3/day4/kernels/` | CUTLASS GEMM 示例 |
| `benchmark_all.cu` | `daily/week3/day5/kernels/` | WMMA 性能基准套件 |
| `wmma_gemm_dbuf.cu` | `daily/week3/day5/kernels/` | WMMA Double Buffer GEMM |
| `wmma_tiled_nopad.cu` | `daily/week3/day6/kernels/` | WMMA 去填充 Tiled GEMM |

> 注：`wmma_gemm.cu`（day1 基础版）已在正确性验证表中 PASS，但 day2-day6 的进阶版本均未执行。

#### Week 4（1 个）

| 文件 | 路径 | 说明 |
|------|------|------|
| `layernorm_welford.cu` | `daily/week4/day3/kernels/` | Welford 算法 LayerNorm |

> 注：`softmax_layernorm.cu`（day2）和 `softmax_layernorm_opt` 已执行，但 Welford 版本未在验证表中。

#### Week 5（1 个）

| 文件 | 路径 | 说明 |
|------|------|------|
| `gemm_backward.cu` | `daily/week5/day4/kernels/` | GEMM 反向传播 kernel |

#### Week 8 — 量化专题（3 个）

| 文件 | 路径 | 说明 |
|------|------|------|
| `fp8_dequant.cu` | `daily/week8/day1/kernels/` | FP8 反量化 kernel |
| `int8_kv_cache.cu` | `daily/week8/day1/kernels/` | INT8 KV Cache kernel |
| `w8a16_dequant.cu` | `daily/week8/day1/kernels/` | W8A16 权重反量化 kernel |

#### Week 10 补充（1 个）

| 文件 | 路径 | 说明 |
|------|------|------|
#### CUTLASS 专题（8 个）

| 文件 | 路径 | 说明 |
|------|------|------|
| `verify_env.cu` | `topics/cutlass/kernels/` | 环境验证 |
| `first_gemm.cu` | `topics/cutlass/kernels/` | 首个 CUTLASS GEMM |
| `cute_basics.cu` | `topics/cutlass/kernels/` | CuTe 基础 |
| `cute_copy.cu` | `topics/cutlass/kernels/` | CuTe Copy |
| `cutlass_gemm_bias_relu.cu` | `topics/cutlass/kernels/` | GEMM + Bias + ReLU 融合 |
| `cutlass_gemm_tiles.cu` | `topics/cutlass/kernels/` | TileShape 调参 |
| `cute_hierarchical.cu` | `topics/cutlass/kernels/` | CuTe 层级抽象 |
| `cutlass_gemm_3x.cu` | `topics/cutlass/kernels/` | CUTLASS 3.x GEMM |

> 注：`topics/cutlass/benchmark/report.md` 声称硬件为 H100，但实际环境为 RTX 5090，报告数据为模板/示意值，非实测。

#### CUDA 挑战题（1 个）

| 文件 | 路径 | 说明 |
|------|------|------|
| `starter.cu` | `topics/cuda/challenges/layer_normalization/starter/` | LayerNorm 挑战题 starter（5 行桩代码） |

---

### 3.2 Python GPU 脚本（共 13 个）

#### Week 4（1 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
| `triton_flash_attention.py` | `daily/week4/day4/kernels/` | `@triton.jit` + `device="cuda"` | Triton FlashAttention |

> 注：同目录的 `triton_softmax.py` 和 `triton_gemm.py` 已执行（正确性验证表 PASS），但 `triton_flash_attention.py` 未在验证表中。

#### Week 5（4 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
| `flash_attention_backward.py` | `daily/week5/day4/kernels/` | PyTorch（GPU 可选但教程预期 GPU） | FA 反向传播教学实现 |
| `benchmark_flash_attention.py` | `daily/week5/day5/kernels/` | `device = "cuda"`（硬编码） | FlashAttention 性能基准 |

#### Week 8（1 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
| `fp8_gemm_benchmark.py` | `daily/week8/day2/kernels/` | `device='cuda'`（硬编码，无回退） | FP8 GEMM 性能基准 |

#### Week 9（1 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
| `dist_allreduce_demo.py` | `daily/week9/day3/kernels/` | NCCL 需多卡 GPU（单卡自动切 gloo） | 分布式 AllReduce demo |

#### Week 10（1 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
| `benchmark_demo.py` | `daily/week10/day3/kernels/` | `assert torch.cuda.is_available()` | 性能基准 demo |

#### Week 10 补充（1 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
#### Triton 专题（7 个）

| 文件 | 路径 | GPU 依赖 | 说明 |
|------|------|---------|------|
| `vector_add.py` | `topics/triton/kernels/` | `@triton.jit` + `device='cuda'` | Triton 向量加法 |
| `matrix_transpose.py` | `topics/triton/kernels/` | `@triton.jit` + `device='cuda'` | Triton 矩阵转置 |
| `reduction.py` | `topics/triton/kernels/` | `@triton.jit` + `device='cuda'` | Triton 规约 |
| `softmax.py` | `topics/triton/kernels/` | `@triton.jit` + `device='cuda'` | Triton Softmax |
| `gemm.py` | `topics/triton/kernels/` | `@triton.jit` + `device='cuda'` | Triton GEMM |
| `flash_attention.py` | `topics/triton/kernels/` | `@triton.jit` + `device='cuda'` | Triton FlashAttention |
| `benchmark.py` | `topics/triton/benchmark/` | `torch.cuda.synchronize()` + `device='cuda'` | Triton 性能基准套件 |

> 注：`topics/triton/benchmark/report.md` 声称硬件为 H100，但实际环境为 RTX 5090，报告数据为模板/示意值，非实测。

---

### 3.3 Markdown 中嵌入的 GPU 代码（无独立文件，3 个专题）

| 专题 | 路径 | 说明 | 目标硬件 |
|------|------|------|---------|
| DeepGEMM | `topics/deepgemm/` | FP8 GEMM JIT 编译、warp-specialized kernel、Grouped GEMM、ncu 调优 | H100/H800（SM90a）或 B200（SM100a） |
| MoE | `topics/moe/` | Triton JIT MoE 路由 kernel、dispatch/compute overlap、all-to-all | GPU（Triton） |
| Transformer | `topics/transformer/` | 手写 MHA、可训练 mini-GPT、`torch.cuda` 训练 | GPU（PyTorch CUDA） |

> 这三个专题的 GPU 代码仅以 markdown 代码块形式存在，无独立 `.py`/`.cu` 文件，需手动提取后才能执行。

---

## 4. 统计汇总

### 4.1 按类别统计

| 类别 | 需要 GPU | 已执行 | 未执行 | 执行率 |
|------|---------|--------|--------|--------|
| CUDA `.cu` kernel（daily） | 25 | 9 | 15 + 1 unclear | ~36% |
| Python GPU 脚本（daily） | 15 | 8 | 7 | ~53% |
| CUDA `.cu`（topics/cutlass） | 8 | 0 | 8 | 0% |
| Python GPU 脚本（topics/triton） | 7 | 0 | 7 | 0% |
| CUDA 挑战题（topics/cuda） | 1 | 0 | 1 | 0% |
| Markdown 嵌入 GPU 代码（topics） | 3 专题 | 0 | 3 专题 | 0% |
| **合计** | **~59** | **~17** | **~42** | **~29%** |

### 4.2 按周/专题统计未执行数量

| 位置 | 未执行 .cu | 未执行 .py | 未执行专题 | 小计 |
|------|-----------|-----------|-----------|------|
| Week 3 | 6 | 0 | — | 6 |
| Week 4 | 1 | 1 | — | 2 |
| Week 5 | 1 | 4 | — | 5 |
| Week 8 | 3 | 1 | — | 4 |
| Week 9 | 0 | 1 | — | 1 |
| Week 10 | 1 | 2 | — | 3 |
| topics/cutlass | 8 | 1 | — | 9 |
| topics/triton | 0 | 7 | — | 7 |
| topics/cuda | 1 | 0 | — | 1 |
| topics/deepgemm | — | — | 1 | 1 |
| topics/moe | — | — | 1 | 1 |
| topics/transformer | — | — | 1 | 1 |
| **合计** | **21** | **17** | **3** | **41** |

### 4.3 按优先级排序的建议执行顺序

**P0 — 基础进度补齐（Week 3-5，已部分执行但关键进阶 kernel 缺失）：**

1. `week3/day2/wmma_gemm_tiled.cu` — WMMA Tiling 是 Week 3 核心进阶
2. `week3/day3/mma_sync_gemm.cu` — mma.sync PTX 是理解 Tensor Core 的关键
3. `week3/day5/wmma_gemm_dbuf.cu` — Double Buffer 是性能优化核心
4. `week4/day3/layernorm_welford.cu` — Welford 是生产级 LayerNorm 的标准算法
5. `week4/day4/triton_flash_attention.py` — Triton FA 是 Week 4 的 Triton 核心产出

**P1 — 量化与推理系统（Week 5, 8）：**

6. `week5/day4/gemm_backward.cu` — 反向传播是训练/微调的基础
7. `week5/day5/benchmark_flash_attention.py` — FA 性能基准
8. `week8/day1/fp8_dequant.cu` — FP8 量化是当前推理热点
9. `week8/day1/int8_kv_cache.cu` — INT8 KV Cache 是推理加速关键
10. `week8/day2/fp8_gemm_benchmark.py` — FP8 GEMM 性能对比
11. `week10/day3/benchmark_demo.py` — 性能基准 demo

**P2 — 专题代码执行（cutlass, triton）：**

12. `topics/cutlass/kernels/` 全部 8 个 `.cu` — CUTLASS 专题核心产出
13. `topics/triton/kernels/` 全部 6 个 `.py` — Triton 专题核心产出
14. `topics/triton/benchmark/benchmark.py` — Triton 性能基准

**P3 — 需特殊硬件/多卡（Week 9, DeepGEMM）：**

15. `week9/day3/dist_allreduce_demo.py` — 需多卡 NCCL
16. `topics/deepgemm/` — 需 H100/H800（SM90a）或 B200（SM100a）
17. `topics/moe/` — 需提取 markdown 代码并执行
18. `topics/transformer/` — 需提取 markdown 代码并执行

---

## 5. 补充说明

### 5.1 已执行但失败的 kernel

| Kernel | 结果 | 原因 |
|--------|------|------|
| `flash_decoding.cu` | FAIL (max_diff=3.43e-01) | 简化版合并步骤精度问题（已知限制） |

### 5.2 Benchmark 报告数据可信度

| 报告 | 声称硬件 | 实际硬件 | 判定 |
|------|---------|---------|------|
| `topics/cutlass/benchmark/report.md` | H100 80GB | RTX 5090 | 模板/示意值，非实测 |
| `topics/triton/benchmark/report.md` | H100 80GB | RTX 5090 | 模板/示意值，非实测 |
| `daily/week1/profiles/week1_profile_summary.md` | RTX 5090 | RTX 5090 | 真实实测 |
| `profiling/week2/day4/README.md` | RTX 5090 | RTX 5090 | 真实实测 |
| `profiling/week3/day1/README.md` | RTX 5090 | RTX 5090 | 真实实测 |

### 5.3 DeepGEMM 专题硬件限制

`topics/deepgemm/` 要求 SM90a（H100/H800）或 SM100a（B200）。RTX 5090 虽为 SM120（Blackwell 架构），但 DeepGEMM 的 JIT 编译器可能不完全支持消费级 Blackwell。需验证 `nvcc --gpu-architecture=sm_120` 是否兼容 DeepGEMM 的 `cuh` 模板。

### 5.4 纯 CPU 脚本（不纳入未执行统计）

以下脚本虽在 `kernels/` 目录下，但明确标注仅依赖标准库，无需 GPU：

| 文件 | 路径 | 说明 |
|------|------|------|
| `continuous_batcher.py` | `daily/week7/day1/kernels/` | Continuous Batching 模拟 |
| `vllm_scheduler_analyzer.py` | `daily/week7/day2/kernels/` | vLLM 调度器分析 |
| `chunked_prefill_simulator.py` | `daily/week7/day3/kernels/` | Chunked Prefill 模拟 |
| `prefix_cache_engine.py` | `daily/week7/day4/kernels/` | Prefix Cache 模拟 |
| `pd_disaggregated_simulator.py` | `daily/week7/day6/kernels/` | PD 分离模拟 |
| `week7_summary.py` | `daily/week7/day7/kernels/` | 周总结 |
| `advanced_features.py` | `daily/week8/day3/kernels/` | 高级特性模拟（标注"仅标准库"） |
| `ring_allreduce_sim.py` | `daily/week9/day3/kernels/` | Ring AllReduce 模拟 |
| `moe_routing_simulator.py` | `daily/week9/day5/kernels/` | MoE 路由模拟（标注"仅标准库+numpy"） |
| `mock_interview.py` | `daily/week10/day5/kernels/` | 模拟面试 |
| `week10_summary.py` | `daily/week10/day7/kernels/` | 周总结 |

### 5.5 `.gitignore` 对审计的影响

`profiling/.gitignore` 采用白名单策略（`*` 忽略全部，然后 `!*.cu !*.md !*.ncu-rep` 等），这意味着：

- 编译后的二进制文件（ELF 可执行文件）即使生成也不会被 git 跟踪
- `.nsys-rep` 文件不跟踪（但 `.ncu-rep` 跟踪）
- `.csv` 导出文件不跟踪
- stdout 日志文件不跟踪

因此，二进制文件的缺失不能作为"未执行"的证据。本报告以 `.ncu-rep` 文件、正确性验证表、README 实测输出作为主要证据。
