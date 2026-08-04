# AI Infra 8 周教程整改计划

> 基于文档：`tutorial_quality_evaluation.md`
> 制定日期：2026-08-04
> 目标：将教程从"概念良好、工程不足"提升到"面试硬通货"水平
> 总工期：约 10-13 周（4 个阶段，可与学习并行执行）

---

## 整改路线总览

```
Phase 1 (P0) 紧急修复        Phase 2 (P1) 补齐高频考点       Phase 3 (P2) 提升可信度      Phase 4 (P3) 打磨
┌──────────────────┐    ┌──────────────────────────┐    ┌────────────────────────┐    ┌──────────────┐
│ 1. 硬件参数统一   │    │ 5. Tensor Core/WMMA 专题  │    │ 11. Mini 引擎真整合     │    │ 18. 仓库卫生  │
│ 2. 事实错误修复   │    │ 6. CUTLASS 源码分析       │    │ 12. 量化专题            │    │ 19. 工具升级  │
│ 3. CUDA bug 修复  │───▶│ 7. Triton 语言专题        │───▶│ 13. CUDA Graph 实操     │───▶│ 20-25. 杂项   │
│ 4. 代码落盘       │    │ 8. 分布式推理专题         │    │ 14. 高级特性实操        │    │              │
│                  │    │ 9. 真实 Profiling 执行    │    │ 15. 多硬件对比          │    │              │
│ 工期：1-2 周     │    │ 10. FA3/FlashDecoding     │    │ 16. backward pass       │    │ 工期：1-2 周 │
│                  │    │                          │    │ 17. Ring Attention      │    │              │
│                  │    │ 工期：3-5 周              │    │ 工期：3-4 周            │    │              │
└──────────────────┘    └──────────────────────────┘    └────────────────────────┘    └──────────────┘
```

**验收原则**：每个任务完成后必须满足"验证标准"列的所有检查项，方可标记为完成。

---

## Phase 1：P0 紧急修复（工期 1-2 周）

> 目标：消除面试时会被当场打脸的硬伤。本阶段所有任务**必须在进入 Phase 2 前完成**。

### 任务 1.1：统一硬件参数为 RTX 5090 实测值

**目标**：以 `week1/day3/exercise/my_gpu_info.md` 的实测数据为唯一基准，消除三套矛盾数据。

**正确参数基准**（来源：`week1/day3/exercise/my_gpu_info.md`）：

| 参数 | 正确值 |
|------|--------|
| FP32 Peak | 104.75 TFLOPS |
| 带宽 | 1792 GB/s GDDR7（非 HBM） |
| Ridge Point | 58.45 FLOP/Byte |
| threads/SM | 1536 |
| warps/SM | 48 |
| blocks/SM | 24 |
| smem/SM | 100 KB |
| L2 Cache | 96 MB |
| SM 数量 | 170 |
| FP32 Cores/SM | 128 |
| Compute Capability | sm_120 |
| 显存 | 32 GB GDDR7 |

**需修改的文件与具体替换**：

| # | 文件 | 错误内容 | 正确内容 |
|---|------|---------|---------|
| 1 | `week1/day1/README.md` L108-111 | "108 SMs, 64 cores, 2048 threads, 64 warps, 32 blocks, 256KB regfile" | "170 SMs, 128 cores, 1536 threads, 48 warps, 24 blocks" |
| 2 | `week1/day1/README.md` L115-122 | 表头"RTX 5090"重复 3 列 + "80GB HBM2e/HBM3/HBM3e" + "989 TFLOPS" | 改为单列 RTX 5090 + "32GB GDDR7" + "104.75 TFLOPS" |
| 3 | `week1/day1/README.md` L165-169 | "132 SMs, 128 cores, 2048 threads, 64 warps, 32 blocks, 228KB smem" | "170 SMs, 128 cores, 1536 threads, 48 warps, 24 blocks, 100KB smem" |
| 4 | `week1/day1/README.md` L385 | "Shared Memory 最多 164 KB" | "Shared Memory 最多 100 KB" |
| 5 | `week1/day2/README.md` L126-131 | "64 warps/SM, 2048 threads, 32 blocks, 164 KB smem" | "48 warps/SM, 1536 threads, 24 blocks, 100 KB smem" |
| 6 | `week1/day3/README.md` L97-98 | "maxThreadsPerMultiProcessor 2048, maxBlocksPerMultiProcessor 32" | "1536, 24" |
| 7 | `week1/day3/README.md` L183-188 | "108 SMs, 64 cores, 19.49 TFLOP/s" | "170 SMs, 128 cores, 104.75 TFLOP/s" |
| 8 | `week1/day3/README.md` L198-203 | "memoryClockRate 1215 MHz, busWidth 5120 bits, 777.6 GB/s" | "14001 MHz, 512 bits, 1792 GB/s"（含 GDDR DDR ×2） |
| 9 | `week1/day3/README.md` L213 | "Ridge Point ≈ 25 FLOP/byte" | "Ridge Point ≈ 58.45 FLOP/byte" |
| 10 | `week1/day3/README.md` L296-311 | occupancy 手算用 2048/64 → 50% | 改用 1536/48 → 66.7%（与 `occupancy_problems.md` 对齐） |
| 11 | `week1/day4/README.md` L68-71 | "19.5 TFLOP/s, 1.55 TB/s HBM, Ridge 12.6" | "104.75 TFLOP/s, 1.792 TB/s GDDR7, Ridge 58.45" |
| 12 | `week1/day4/README.md` L86 | 同上重复 | 同上 |
| 13 | `week1/day4/README.md` L178 | "Blackwell 之后 L2 以 32 字节 sector" | "32 字节 sector 早于 Blackwell 多代" |
| 14 | `week1/day4/README.md` L285 | "48KB shared + 16KB L1, or 96KB shared + 0KB L1" | 删除或改为现代架构说明 |
| 15 | `week1/day6/README.md` L188-194 | "19.5 TFLOP/s, 1.55 TB/s HBM, ridge 12.6" | "104.75 TFLOP/s, 1.792 TB/s GDDR7, ridge 58.45" |
| 16 | `week1/day6/notes/day6_nsight_profiling.md` L79-82 | 同上 | 同上 |
| 17 | `week1/day7/README.md` L88 | "L2 Cache 40MB" | "L2 Cache 96MB" |
| 18 | `week1/day7/README.md` L96 | "192KB SRAM per SM" | "100KB smem+L1 per SM" |
| 19 | `week1/day7/README.md` L279 | "32 blocks 或 64 warps" | "24 blocks 或 48 warps" |
| 20 | `week1/day7/README.md` L361, L427 | "ridge ≈ 12.6, 19.5T / 1.55T" | "ridge ≈ 58.45, 104.75T / 1.792T" |
| 21 | `week1/tools/cuda_occupancy_calculator.py` L41-42 | sm_120 表项 `(2048, 64, 32, 65536, 228000, 49152, 4, 256)` | `(1536, 48, 24, 65536, 100000, 49152, ?, ?)`（需查 RTX 5090 实际分配粒度） |
| 22 | `week8` 全部文件 | "19.5 TFLOPS, 1.55 TB/s, ridge 12.6" | "104.75 TFLOPS, 1.792 TB/s, ridge 58.45" |

**验证标准**：
- [ ] `grep -rn "19\.5" week1/ week8/` 无结果（排除非硬件参数引用）
- [ ] `grep -rn "1\.55" week1/ week8/` 无结果
- [ ] `grep -rn "12\.6" week1/ week8/` 无结果
- [ ] `grep -rn "2048" week1/day1/README.md week1/day2/README.md week1/day3/README.md week1/day7/README.md` 无误用
- [ ] `grep -rn "989" week1/` 无结果
- [ ] `grep -rni "HBM" week1/day1/README.md week1/day4/README.md week1/day6/README.md` 无误用（HBM 改为 GDDR7）
- [ ] `python3 week1/tools/cuda_occupancy_calculator.py --cc 12.0 --registers 32 --block-size 256` 输出与 `occupancy_problems.md` 一致
- [ ] `week1/day3/README.md` occupancy 手算结果（66.7%）与 `week1/day3/exercise/occupancy_problems.md` 一致

**预估工时**：4-6 小时

---

### 任务 1.2：修复核心知识点事实错误

**目标**：消除 12 处事实性错误，确保面试白板推导不出错。

| # | 文件 | 错误 | 修复方式 | 验证 |
|---|------|------|---------|------|
| 1 | `week2/day1/README.md` L31, L84 | "Warp Shuffle 从 Blackwell(CC 12.0)起引入" | 改为"Warp Shuffle 自 **Kepler(sm_30, 2012)** 引入；`_sync` 后缀 + mask 必选始于 **Volta(sm_70, 2017)** 的 Independent Thread Scheduling" | `grep -rn "Blackwell.*Shuffle\|CC 12.0.*Shuffle\|12\.0.*Shuffle" week2/` 无结果 |
| 2 | `week1/day3/notes/cuda_programming_guide_performance.md` L172 | "From CC 12.0 起 CUDA 提供 warp shuffle" | 改为"Since CC 3.0 (Kepler)" | 同上 |
| 3 | `week4/day1/README.md` L173 | FA IO = "O(Nd)" | 改为"**Θ(N²d²/M)**（M 为 SRAM 大小），当 M=Θ(Nd) 时简化为 O(Nd)"；加注释引用 `paper/flashattention/README.md` Theorem 2 | `grep -rn "O(Nd)" week4/` 每处都带 Θ(N²d²/M) 标注 |
| 4 | `week4/day6/README.md` L73 | 同上 | 同上 | 同上 |
| 5 | `week4/day7/README.md` L290 | 同上 | 同上 | 同上 |
| 6 | `week4/day1/README.md` L49-54 | 标准 Attention IO "3N²+4Nd"，逐项求和为 5N²+4Nd | 修正为 **4N²+4Nd**（2N² S write + 2N² P write + Nd×4 read/write），逐项求和与总数一致 | 手算验证逐项求和 = 总数 |
| 7 | `week4/day6/README.md` L63-68 | 同上"3N²+4Nd" | 同上 | 同上 |
| 8 | `week4/day1/README.md` L172 | FA N=4096 IO "~2 MB" | 改为 **~4 MB**（4Nd·4B = 4×4096×64×4 = 4MB），与同日 Python 脚本一致 | `grep -rn "2 MB\|2MB" week4/day1/` 无误用 |
| 9 | `week4/day1/README.md` L173, L338, `week4/day6/README.md` L81 | IO 加速比三处不同（100x/51.6x/48.8x） | 统一为 **~50x**（标准 ~206MB / FA ~4MB），标注"理论值，实测 wall-clock 2-8x" | 三处数值一致 |
| 10 | `week8/day4/README.md` L124 | KV Cache "LLaMA2-7B 1 MB/token (FP16)" | 改为 **~524 KB/token**（2×32×32×128×2B = 524,288B），与 `week8/day6/README.md` L124 对齐 | `grep -rn "1 MB/token\|1MB/token" week8/` 无结果 |
| 11 | `week8/day6/README.md` L53, `day7/README.md` L45 | GEMM 优化"九层" | 统一为 **8 层**（Naive→Tiling→RegBlock→float4→Shuffle→DblBuf→TensorCore→Auto-tuning） | `grep -rn "九层\|9 层\|9层" week8/` 无结果 |
| 12 | `week8/day6/README.md` L53, `knowledge_selftest.py` L32 | Register Blocking "45%" | 统一为 **~30.8%**（Week2 Day6 实测值）或标注"理论 ~40%，实测 30.8%" | `grep -rn "45%" week8/` 无误用 |
| 13 | `week8/day3/README.md` L149, `interview_basics.py` L102 | `nvcc-- default - stream per - thread` | 改为 `nvcc --default-stream per-thread` | `grep -rn "nvcc--" week8/` 无结果 |
| 14 | `week1/day4/README.md` L178 | "Blackwell 之后 L2 以 32 字节 sector" | 改为"32 字节 sector 机制早于 Blackwell 多代" | — |
| 15 | `week1/day5/README.md` L81-90 | "模式 3：2-way Conflict" 技术错误 | `tile[threadIdx.x % 2]` 实际是两个 broadcast 到两个 bank = **无 conflict**，不是 2-way conflict；重写该模式或删除 | 代码验证：ncu bank conflict 计数为 0 |

**验证标准**：
- [ ] 上表 15 项全部修复
- [ ] `week4/day1/README.md` 中 IO 逐项求和与总数一致
- [ ] `week4` 中 FA IO 加速比三处统一
- [ ] `week8` 中 GEMM 层数、Register Blocking 占比各只有一种说法
- [ ] Week2 Day1 不再出现"Blackwell 引入 Shuffle"

**预估工时**：3-4 小时

---

### 任务 1.3：修复 CUDA 内核 bug

**目标**：消除 9 处真实 bug，确保所有 `.cu`/`.py` kernel 正确运行。

| # | 文件 | bug | 修复方式 | 验证 |
|---|------|-----|---------|------|
| 1 | `week5/day2/kernels/kv_cache.cu` L156 | `cache.append(0, d_k2, d_v2, 8)` 从 5 token 缓冲读 8 token | 将 `d_k2`/`d_v2` 分配从 5 扩大到 8，或改 `new_len=5` | 运行无越界，验证 Round 2 数据正确 |
| 2 | `week7/day4/kernels/custom_ops_module.py` L47 | `atomicMax((int*)&s_max, ...)` 对负 float 不保序 | 改用 shared memory + warp shuffle block reduce max（参考 `week3/day3/softmax_layernorm_opt.cu` 的 warp-level 实现） | softmax 输入含负数时 max 正确 |
| 3 | 同上 L152-163 | online softmax 更新混用两种归一化 | 改为标准 FA 公式：`out = out * (old_sum / new_sum) + v * exp(score - new_max) / new_sum` | 与 CPU reference max_diff < 1e-5 |
| 4 | 同上 L181/191/206 | kernel launch 未传 stream | 三处 `<<<grid, block>>>` 改为 `<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>` | 编译通过，多 stream 下无 ordering bug |
| 5 | 同上 L133 | `out_local[256]` 硬编码无检查 | 加 `TORCH_CHECK(D <= 256, "D must be <= 256")`；或改用动态 shared memory | D=257 时报错而非栈损坏 |
| 6 | 同上 L176-210 | 三个 wrapper 无形状/dtype/连续性检查 | 每个 wrapper 加 `TORCH_CHECK(input.is_contiguous())` + `TORCH_CHECK(input.dtype() == torch::kFloat32)` + 维度检查 | 非连续/非 float32 输入时报错 |
| 7 | `week3/day3/kernels/softmax_layernorm_opt.cu` L463, L492, L497, L535 | print 字符串 "Day16" | 改为 "Week3 Day3" 或 "day3" | `grep -rn "Day16" week3/` 无结果 |
| 8 | `week7/day2/kernels/full_scheduler.py` L192-193 | `else: break` 跳过后续 running 请求 | 改为 `else: continue` | 多请求场景下后续请求仍被调度 |
| 9 | `week3/day6/kernels/profiling_targets.cu` + `week3/day6/README.md` L335-339 | naive GEMM（无 tiling，AI≈0.25）被标为 "compute-bound" | 改为"memory-bound（未 tiling 的 GEMM AI≈0.25，远低于 ridge point）"；或换用 Week2 的 tiled GEMM 作为 compute-bound 示例 | 理论分析（AI=2K/8K=0.25 < ridge）与标注一致 |

**验证标准**：
- [ ] `nvcc -O3 -arch=sm_120 week5/day2/kernels/kv_cache.cu && ./a.out` 无越界，3 轮全过
- [ ] `python3 week7/day4/kernels/custom_ops_module.py` 在 CUDA 环境下运行，softmax 含负数时 max_diff < 1e-5
- [ ] `nvcc -O3 -arch=sm_120 week3/day3/kernels/softmax_layernorm_opt.cu && ./a.out` 无 "Day16" 输出
- [ ] `python3 week7/day2/kernels/full_scheduler.py` 多请求场景下后续 running 请求被正确调度
- [ ] `week3/day6` 的 GEMM bound-type 标注与理论分析一致

**预估工时**：4-6 小时

---

### 任务 1.4：把 markdown 代码落盘为真实文件

**目标**：将 Week2 和 Week4 中仅存在于 markdown 代码块的 kernel 代码提取为真实可编译文件。

**Week2 需落盘的文件**：

| # | 目标文件 | 源 markdown 位置 | 说明 |
|---|---------|-----------------|------|
| 1 | `week2/day1/kernels/warp_reduce.cu` | `week2/day1/README.md` L178-328（~145 行） | warpReduceSum + blockReduceSum + launchReduce + CPU 验证 |
| 2 | `week2/day2/kernels/register_blocking_gemm.cu` | `week2/day2/README.md` L119-332（~210 行） | BM=BN=128, BK=8, TM=TN=8 + cuBLAS 对比 + 正确性检查 |
| 3 | `week2/day5/kernels/flash_attention.cu` | `week2/day5/README.md` L383-610（~225 行） | Br=64, Bc=32, D=64 + online softmax + CPU 验证 |
| 4 | `week2/day6/kernels/integrated_gemm.cu` | `week2/day6/README.md` L278-557（~280 行） | Register Blocking + float4 load + float4 writeback |

**Week4 需落盘的文件**：

| # | 目标文件 | 源 markdown 位置 | 说明 |
|---|---------|-----------------|------|
| 5 | `week4/day2/kernels/flash_attention_v2.cu` | `week4/day2/README.md` L155-459（305 行） | 完整 FA forward kernel + warp 分区 + CPU 验证 |
| 6 | `week4/day5/kernels/mini_engine_fa.py` | `week4/day5/README.md` L145-272（~128 行） | PyTorch C++ Extension 集成 FA kernel |
| 7 | `week4/day6/kernels/benchmark_flash_attention.py` | `week4/day6/README.md` L129-234（~106 行） | benchmark 框架（标准 vs 手写 vs 官方） |

**Week4 额外需创建的文件**：

| # | 目标文件 | 说明 |
|---|---------|------|
| 8 | `week4/day4/kernels/flash_attention_fa2.cu` | 基于 Day2 kernel 修改 `WARPS_PER_BLOCK_FA2=16, ROWS_PER_WARP_FA2=4`（README L153-179 提到的修改） |

**操作步骤**（每个文件）：
1. 从 markdown 代码块提取完整代码
2. 创建 `kernels/` 目录（若不存在）
3. 写入 `.cu`/`.py` 文件
4. 编译/运行验证
5. 更新 README 中的 GitHub 链接，确保指向真实文件
6. 删除 Day7"本周目录结构"中与现实的矛盾描述

**验证标准**：
- [ ] `nvcc -O3 -arch=sm_120 week2/day1/kernels/warp_reduce.cu && ./a.out` 编译运行通过
- [ ] `nvcc -O3 -arch=sm_120 -lcublas week2/day2/kernels/register_blocking_gemm.cu && ./a.out` 编译运行通过
- [ ] `nvcc -O3 -arch=sm_120 week2/day5/kernels/flash_attention.cu && ./a.out` 编译运行通过，max_diff < 1e-3
- [ ] `nvcc -O3 -arch=sm_120 -lcublas week2/day6/kernels/integrated_gemm.cu && ./a.out` 编译运行通过
- [ ] `nvcc -O3 -arch=sm_120 week4/day2/kernels/flash_attention_v2.cu && ./a.out` 编译运行通过，max_diff < 1e-3
- [ ] `python3 week4/day5/kernels/mini_engine_fa.py` 运行通过（需 CUDA 环境 + PyTorch）
- [ ] `python3 week4/day6/kernels/benchmark_flash_attention.py` 运行通过
- [ ] `week2/day7/README.md` 的"本周目录结构"与磁盘实际一致
- [ ] `week4/day5/mini_engine_fa.py` 中 `open("flash_attention_v2.cu").read()` 路径正确指向已落盘文件

**预估工时**：3-4 小时

---

### Phase 1 整体验收清单

- [ ] 任务 1.1 ~ 1.4 全部完成
- [ ] `grep -rn "19\.5 TFLO\|1\.55 TB\|12\.6 FLOP" aiinfra/daily/` 无误用
- [ ] `grep -rn "Day16\|nvcc--" aiinfra/daily/` 无结果
- [ ] 所有 Week1-Week4 的 `.cu` 文件可独立编译运行
- [ ] `python3 build.py` 构建成功
- [ ] 提交 commit：`fix(daily): P0 紧急修复 - 统一硬件参数、修复事实错误与 CUDA bug、代码落盘`

---

## Phase 2：P1 补齐面试高频考点（工期 3-5 周）

> 目标：补齐 2025 年算子/Infra 岗位的高频考点，每个专题至少 1 天完整教程。

### 任务 2.1：新增 Tensor Core / WMMA 专项教程

**目标**：手写 `nvcuda::wmma` GEMM，让 GEMM 占比从 64% 提升到 85%+。

**落位建议**：`week2/day6b/`（在 Day6 与 Day7 之间插入）或扩展 Day6 为两天

**教程内容要求**：
- 理论：Tensor Core 架构（WMMA/MMA 指令、FP16 输入/FP32 累加、矩阵分片 m16n16k16）
- Coding：手写 `wmma_gemm.cu`（使用 `nvcuda::wmma::fragment`，对比 Week2 Day6 的 FMA GEMM）
- Profiling：`ncu` 对比 FMA vs WMMA 的 SM throughput、Tensor Core 利用率
- 性能目标：WMMA GEMM 达到 cuBLAS 85%+
- 面试题：≥5 道（WMMA fragment 生命周期、FP16 精度损失、Tensor Core 对齐要求、WMMA vs mma.sync、mixed precision 策略）

**需创建的文件**：
- `week2/day6b/README.md`（~550 行，遵循 8 段骨架）
- `week2/day6b/kernels/wmma_gemm.cu`（可编译，含 cuBLAS 对比）
- `week2/day6b/kernels/gemm_wmma_series.cu`（v7: WMMA, v8: WMMA+double buffer）

**验证标准**：
- [ ] `nvcc -O3 -arch=sm_120 -lcublas week2/day6b/kernels/wmma_gemm.cu && ./a.out` 编译运行通过
- [ ] WMMA GEMM 在 4096² 下达到 cuBLAS 85%+
- [ ] ncu 报告显示 Tensor Core 利用率 > 50%
- [ ] 面试题 ≥5 道
- [ ] 遵循 8 段骨架 + LeetGPU + LeetCode

**预估工时**：2-3 天

---

### 任务 2.2：新增 CUTLASS 源码分析教程

**目标**：真正读 CUTLASS 源码，理解三级 tiling 抽象，实例化调用。

**落位建议**：扩展 `week2/day3/` 和 `week2/day4/`（当前只提了名字）

**教程内容要求**：
- 理论：CUTLASS 分层抽象（Device → Kernel → Warp → Thread），`ThreadblockShape/WarpShape/InstructionShape` 三级 tiling
- 源码分析：读 `cutlass/gemm/device/gemm.h`，追踪一次 `cutlass::gemm::device::Gemm<...>::operator()` 的调用链
- Coding：实例化 `cutlass::gemm::device::Gemm` 调用，对比手写 WMMA GEMM
- 面试题：≥5 道（CUTLASS 三级 tiling、`Policy`/`Mma`/`Epilogue` 分离、`ContiguousK` 布局、CUTLASS 3.x vs 2.x、CuteLAYOUT）

**需创建的文件**：
- `week2/day3/kernels/cutlass_gemm_example.cu`（实例化 CUTLASS GEMM）
- 更新 `week2/day3/README.md` 和 `week2/day4/README.md`，加入真实源码引用与行号

**验证标准**：
- [ ] `nvcc -O3 -arch=sm_120 -I<path-to-cutlass> week2/day3/kernels/cutlass_gemm_example.cu && ./a.out` 编译运行通过
- [ ] README 中有 CUTLASS 源码的具体文件名与行号引用
- [ ] 面试题 ≥5 道

**预估工时**：2-3 天

---

### 任务 2.3：新增 Triton 语言专题

**目标**：用 Triton 重写核心算子，对比 CUDA 实现的行数与性能。

**落位建议**：新增 `week3/day3b/` 或扩展 `week3/day5/`

**教程内容要求**：
- 理论：Triton 编程模型（block-level programming、`tl.load`/`tl.store`、`tl.reduce`、自动 tiling）
- Coding 1：用 Triton 重写 Softmax（对比 Week3 Day2 的 CUDA 版本）
- Coding 2：用 Triton 重写 GEMM（对比 Week2 Day6 的 CUDA 版本）
- Coding 3：用 Triton 重写 FlashAttention forward（对比 Week4 Day2 的 CUDA 版本）
- Profiling：`ncu` 对比 Triton vs CUDA 的性能
- 面试题：≥5 道（Triton vs CUDA trade-off、`tl.reduce` 实现、Triton 的 autotune、Triton 在 FlashAttention 中的应用、Triton 的局限性）

**需创建的文件**：
- `week3/day3b/README.md`（~550 行）
- `week3/day3b/kernels/triton_softmax.py`
- `week3/day3b/kernels/triton_gemm.py`
- `week3/day3b/kernels/triton_flash_attention.py`

**验证标准**：
- [ ] 三个 Triton kernel 均可运行，正确性验证通过
- [ ] 有 Triton vs CUDA 的性能对比表
- [ ] 面试题 ≥5 道

**预估工时**：2-3 天

---

### 任务 2.4：新增分布式推理专题

**目标**：覆盖 TP/PP/DP + NCCL + 通信计算重叠。

**落位建议**：新增 `week7/day3b/` 或扩展 `week7/day3/`（当前 Day3 是 SGLang/LightLLM）

**教程内容要求**：
- 理论 1：Tensor Parallelism（column-parallel / row-parallel QKV 切分、all-reduce 通信、TP=2/4/8 的 trade-off）
- 理论 2：Pipeline Parallelism（GPipe / 1F1B / interleaved PP、micro-batching、bubble ratio）
- 理论 3：NCCL collectives（all-reduce / all-gather / reduce-scatter 的通信量、ring vs tree 拓扑）
- 理论 4：通信计算重叠（`torch.cuda.Stream` 双流、CUDA Graph + 通信重叠）
- Coding：用 `torch.distributed` + `torch.cuda.Stream` 实现一个简单的 TP 推理 demo（2 卡 TP，QKV column-parallel + all-reduce）
- Profiling：`nsys` 查看 comm 与 compute 的 overlap 时间线
- 面试题：≥5 道（TP 下 QKV 怎么切、all-reduce 通信量、1F1B bubble ratio、NCCL ring 拓扑、通信重叠实现）

**需创建的文件**：
- `week7/day3b/README.md`（~600 行）
- `week7/day3b/kernels/tp_inference_demo.py`（多卡 TP demo）
- `week7/day3b/kernels/comm_overlap_demo.py`（双流 overlap demo）

**验证标准**：
- [ ] TP demo 在多卡环境下运行通过（或单卡模拟）
- [ ] nsys 时间线显示 comm/compute overlap
- [ ] 面试题 ≥5 道
- [ ] 覆盖 TP/PP/NCCL/通信重叠四个子主题

**预估工时**：3-4 天

---

### 任务 2.5：真正执行 Profiling 并记录数据

**目标**：填补 profiling 数据表空白，让"Profiling 是核心能力"名副其实。

**执行对象**（至少选 2 个）：

| # | 对象 | 文件 | 填入位置 |
|---|------|------|---------|
| 1 | Week2 Day6 GEMM v4/v6 | `gemm_optimization_series.cu` | `week1/profiles/week1_profile_summary.md` + `week2/day4/notes/` |
| 2 | Week4 Day2 FlashAttention | `flash_attention_v2.cu` | `week4/day2/notes/` + `week4/day6/notes/` |
| 3 | Week3 Day2 Softmax/LayerNorm | `softmax_layernorm.cu` | `week3/day2/notes/` |
| 4 | Week5 Day4 PagedAttention | `paged_attention.cu` | `week5/day4/notes/` |

**每个对象需执行的操作**：
1. `nsys profile -o <name> ./<binary>` → 截图时间线，填入 kernel launch gap、总时间
2. `ncu --set full --kernel-name regex:<kernel> -o <name> ./<binary>` → 提取关键指标
3. 填入对应的 `notes/` 数据表：SM throughput、DRAM throughput、occupancy、registers/thread、L1 hit rate、bank conflicts、warp stall reasons
4. 绘制 Roofline 图（标出 kernel 位置 vs ridge point 58.45）
5. 记录 ncu 报告截图

**需填充的数据表**：

| 文件 | 当前状态 | 目标状态 |
|------|---------|---------|
| `week1/profiles/week1_profile_summary.md` | 全空白 | 4+ kernel 的完整 profiling 报告 |
| `week1/day1/notes/day1_hello_gpu.md` | 全空白 | hello_gpu 的 nsys 时间线 + ncu 基础指标 |
| `week1/day2/notes/day2_occupancy.md` | 全空白 | occupancy_test 的实测 vs 手算对比表 |
| `week1/day4/notes/day4_transpose.md` | 全空白 | naive vs tiled transpose 的带宽对比 |
| `week1/day5/notes/day5_bank_conflict.md` | 全空白 | bank conflict 计数实测 |
| `week1/day6/notes/day6_nsight_profiling.md` | 全空白 | 至少 3 个 kernel 的 ncu 完整报告 + Roofline |
| `week1/day7/notes/day7_summary.md` | 全空白 | Week1 综合 profiling 报告 |

**验证标准**：
- [ ] `week1/profiles/week1_profile_summary.md` 环境信息表已填写（GPU 型号、驱动、CUDA 版本）
- [ ] 至少 4 个 kernel 有完整的 ncu 指标表（SM%、DRAM%、occupancy、registers、stall reasons）
- [ ] 至少 1 张 nsys 时间线截图
- [ ] 至少 1 张 Roofline 图（kernel 位置标注）
- [ ] `week1/day6/README.md` L227-230 的"假设值"替换为真实测量值

**预估工时**：2-3 天（需 GPU 环境）

---

### 任务 2.6：补齐 FlashAttention-3 / FlashDecoding

**目标**：覆盖 FA3 和 FlashDecoding，链接仓库已有的 paper notes。

**落位建议**：
- FA3：扩展 `week4/day4/`（当前 FA2 之后加 FA3）
- FlashDecoding：新增 `week5/day4b/` 或扩展 `week5/day4/`

**FA3 教程内容**：
- 理论：FA3 的三大改进（async pipeline with `cp.async`/TMA、FP8 支持、warp specialization producer/consumer）
- 对比：FA1 → FA2 → FA3 的演进表（含 FLOPs、occupancy、warp 分配）
- 源码分析：链接仓库 `paper/flashattention3/README.md`
- Coding：概念级伪代码（FA3 的 producer/consumer warp group 分配）
- 面试题：≥3 道

**FlashDecoding 教程内容**：
- 理论：FlashDecoding 的核心思想（KV 跨 SM 切分，解决 decode 阶段单 query 并行度不足）
- 对比：标准 decode attention vs FlashDecoding（并行度、通信量）
- Coding：简化版 FlashDecoding kernel（单 query，KV 按 block 切分到不同 block，最后 all-reduce）
- 面试题：≥3 道

**需创建/修改的文件**：
- 更新 `week4/day4/README.md`（加 FA3 章节）
- 新增 `week5/day4b/README.md`（FlashDecoding）
- 新增 `week5/day4b/kernels/flash_decoding.cu`（简化版 kernel）

**验证标准**：
- [ ] `week4/day4/README.md` 有 FA3 完整章节（≥80 行）
- [ ] `week5/day4b/kernels/flash_decoding.cu` 可编译运行，正确性验证通过
- [ ] 两篇各 ≥3 道面试题
- [ ] FA3 章节链接 `paper/flashattention3/README.md`

**预估工时**：2-3 天

---

### Phase 2 整体验收清单

- [ ] 任务 2.1 ~ 2.6 全部完成
- [ ] 新增 5 个专题教程（Tensor Core/CUTLASS/Triton/分布式/FA3+FlashDecoding）
- [ ] profiling 数据表至少 4 个 kernel 有真实测量值
- [ ] 新增 ≥25 道面试题（5 专题 × ≥5 题）
- [ ] `python3 build.py` 构建成功
- [ ] 提交 commit：`feat(daily): P1 补齐 Tensor Core/CUTLASS/Triton/分布式/FA3/FlashDecoding 高频考点`

---

## Phase 3：P2 提升项目可信度与覆盖面（工期 3-4 周）

> 目标：让 Mini 引擎真正可用，补齐中高频考点。

### 任务 3.1：真正整合 Mini 引擎

**目标**：让 `custom_ops_module.py` 的 kernel 被 `mini_engine_v1` 调用，替换 `time.sleep` 模拟。

**具体子任务**：

| # | 子任务 | 文件 | 说明 |
|---|--------|------|------|
| 1 | custom kernel 接入 mini engine | `week6/day5/kernels/mini_engine_v1.py` | 让 `TransformerLayer` 支持 `use_custom_ops` 开关，调用 `custom_ops_module.py` 的 softmax/layernorm/attention |
| 2 | 真实 batched forward | `week6/day5/kernels/mini_engine_v1.py` | `_run_iteration` 实现 pad + merge 成单 batch tensor，一次 forward 处理多请求（当前是逐请求循环） |
| 3 | PagedAttention 接入 | `week6/day5/kernels/mini_engine_v1.py` | 用 `paged_attention.cu` 替换 `torch.cat` 式 KV Cache |
| 4 | 真实 timing | `week7/day5/kernels/stability_test.py` | 用 `torch.cuda.Event` 替换 `time.sleep(forward_delay)` |
| 5 | 真实 profiling | `week7/day6/kernels/full_chain_profile.py` | 用 `torch.profiler` + 真实 nsys/ncu 数据替换 `time.sleep` 模拟 |
| 6 | 修复假集成测试 | `week7/day5/kernels/stability_test.py` L341-368 | `test_custom_kernel_integration` 真正调用 `custom_ops_module.py` 而非用 0.8x 乘子 |

**验证标准**：
- [ ] `mini_engine_v1.py` 的 `TransformerLayer(use_custom_ops=True)` 能调用真实 CUDA kernel
- [ ] batched forward 下多请求性能 > 逐请求循环（wall-clock 对比）
- [ ] `stability_test.py` 的 custom kernel 测试真正调用 CUDA kernel
- [ ] `full_chain_profile.py` 产生真实的 `torch.profiler` 输出（非 `time.sleep`）
- [ ] 面试可清晰叙述"引擎各组件如何集成"

**预估工时**：4-5 天

---

### 任务 3.2：量化专题

**目标**：覆盖 W8A16/W4A16 weight-only 量化、INT8 KV Cache、FP8。

**落位建议**：新增 `week5/day6b/` 或 `week6/day6b/`

**教程内容**：
- 理论：量化基础（对称/非对称、per-channel/per-token、weight-only vs weight+activation）
- 算法：AWQ vs GPTQ 对比（activation-aware vs Hessian-based）
- Coding 1：W8A16 weight-only dequant kernel（INT8 weight → FP16 计算）
- Coding 2：INT8 KV Cache 量化 kernel（per-token scale）
- Coding 3：FP8 GEMM（Hopper/Blackwell `mma.sync` FP8）
- 面试题：≥5 道

**需创建的文件**：
- `week5/day6b/README.md`
- `week5/day6b/kernels/w8a16_dequant.cu`
- `week5/day6b/kernels/int8_kv_cache.cu`
- `week5/day6b/kernels/fp8_gemm.cu`

**验证标准**：
- [ ] 3 个量化 kernel 可编译运行
- [ ] 有量化前后精度对比表（perplexity 或 max_diff）
- [ ] 有量化前后性能对比（latency/throughput）
- [ ] 面试题 ≥5 道

**预估工时**：2-3 天

---

### 任务 3.3：CUDA Graph 实操

**目标**：实现 CUDA Graph 静态捕获 + 动态 shape 处理。

**落位建议**：新增 `week7/day6b/` 或扩展 `week7/day6/`

**教程内容**：
- 理论：CUDA Graph 原理（capture/replay、静态 shape 限制、dynamic shape 的 shape bucketing）
- Coding 1：`torch.cuda.CUDAGraph` 捕获 mini engine 的 decode 迭代
- Coding 2：shape bucketing 实现（对不同 batch size 预捕获多个 graph）
- Profiling：`nsys` 对比有/无 CUDA Graph 的 kernel launch gap
- 面试题：≥3 道

**需创建的文件**：
- `week7/day6b/README.md`
- `week7/day6b/kernels/cuda_graph_capture.py`
- `week7/day6b/kernels/shape_bucketing.py`

**验证标准**：
- [ ] CUDA Graph 捕获成功，replay 输出与 eager 一致
- [ ] nsys 显示 kernel launch gap 显著减少
- [ ] 面试题 ≥3 道

**预估工时**：1-2 天

---

### 任务 3.4：高级特性实操（Spec Decoding / Chunked Prefill / Prefix Caching）

**目标**：至少有一个特性真实集成进 mini engine（当前全是模拟）。

**子任务**（至少选 1 个）：

| # | 特性 | 当前状态 | 目标 |
|---|------|---------|------|
| 1 | Speculative Decoding | `week7/day3` 模拟 | 真实集成：draft model（小 Transformer）+ target model（mini engine）+ accept/reject |
| 2 | Chunked Prefill | `week6/day4` 模拟器 | 真实集成到 `mini_engine_v1` 的 scheduler |
| 3 | Prefix Caching | 仅提及 | 实现基于 block hash 的 prefix cache + LRU 淘汰 |

**建议优先级**：Chunked Prefill > Prefix Caching > Spec Decoding（前两者更易集成）

**验证标准**：
- [ ] 选定特性真实集成进 mini engine
- [ ] 有 with/without 特性的性能对比（latency/throughput）
- [ ] 面试题 ≥3 道

**预估工时**：2-3 天

---

### 任务 3.5：多硬件对比专题（Ascend NPU）

**目标**：对比 NVIDIA CUDA vs Ascend CANN 编程模型。

**落位建议**：新增 `week8/day3b/` 或独立专题

**教程内容**：
- 对比表：CUDA grid/block/thread vs Ascend grid/block/tiling
- 对比表：CUDA shared memory vs Ascend Unified Buffer
- 对比表：CUDA warp shuffle vs Ascend vector copy
- 对比表：Nsight Compute vs Ascend msprof
- 简单示例：用 Ascend C++ 写一个 vector add，对比 CUDA 版本
- 面试题：≥3 道

**验证标准**：
- [ ] 有完整的编程模型对比表（≥5 个维度）
- [ ] 有 Ascend 简单示例代码（或伪代码）
- [ ] 面试题 ≥3 道

**预估工时**：1-2 天

---

### 任务 3.6：补 backward pass / 反向传播

**目标**：覆盖 FlashAttention backward + GEMM backward。

**落位建议**：扩展 `week4/day2/` 或新增 `week4/day2b/`

**教程内容**：
- 理论：FA backward 的 recomputation 策略（`L_i = m_i + log ℓ_i` 重计算 trick）
- Coding：FA backward kernel（简化版）
- GEMM backward（`dC → dA/dB` 的数据流）
- 面试题：≥3 道

**验证标准**：
- [ ] FA backward kernel 可编译运行，梯度正确性验证（finite difference）
- [ ] `L_i = m_i + log ℓ_i` 在教程中有完整推导
- [ ] 面试题 ≥3 道

**预估工时**：2-3 天

---

### 任务 3.7：补 Ring Attention / 长上下文分布式注意力

**目标**：覆盖 Ring Attention 的原理与实现。

**落位建议**：扩展分布式推理专题（任务 2.4）或独立日

**教程内容**：
- 理论：Ring Attention 的核心思想（KV 跨 GPU 流式传输 + 本地 attention 计算 + 通信重叠）
- 与 FlashAttention 的关系
- Coding：简化版 Ring Attention 伪代码 / 单机模拟
- 面试题：≥3 道

**验证标准**：
- [ ] Ring Attention 原理图解清晰
- [ ] 有伪代码或单机模拟
- [ ] 面试题 ≥3 道

**预估工时**：1-2 天

---

### Phase 3 整体验收清单

- [ ] 任务 3.1 ~ 3.7 全部完成
- [ ] Mini 引擎真正集成 custom kernel + batched forward + PagedAttention
- [ ] 新增 ≥4 个专题教程（量化/CUDA Graph/高级特性/多硬件/backward/Ring Attention）
- [ ] 新增 ≥20 道面试题
- [ ] `python3 build.py` 构建成功
- [ ] 提交 commit：`feat(daily): P2 Mini 引擎真整合 + 量化/CUDA Graph/分布式注意力专题`

---

## Phase 4：P3 打磨（工期 1-2 周）

> 目标：仓库卫生、工具升级、文档修正。

### 任务 4.1：仓库卫生修复

| # | 修复项 | 文件 | 操作 |
|---|--------|------|------|
| 1 | 硬编码 macOS 路径 | `week1/day*/notes/*.md` | `/Users/chenbinbin/GitHub/aiinfra/week1` → 相对路径 |
| 2 | broken SVG 引用 | `week1/day1/README.md` L67/191/546, `day3/README.md` L20, `day4/README.md` L138, `day7/README.md` L44 | `../../images/` → `../images/`；创建缺失的 SVG 或删除引用 |
| 3 | LeetCode 表格重复 | `week1/day7/README.md` L509-522 & L526-539 | 删除重复段 |
| 4 | LeetCode 表格重复 | `week2/day7/README.md` L449-461 & L465-477 | 删除重复段 |
| 5 | LeetGPU 重复 | `week7/` Day1/2/3/4/5/7 六次 Matrix Transpose | 按 `SKILL.md` 表格改为不同题 |
| 6 | LeetGPU 重复 | `week4/day6` 与 `day2` 重复 Multi-Head Attention | `day6` 改为 `SKILL.md` 规划的题 |
| 7 | LeetCode 表格 copy-paste 错 | `week8/day3` 出现零钱兑换、`day4` 出现课程表、`day5` 出现最长有效括号、`day7` 出现排序链表 | 改为各天实际题目 |
| 8 | LeetCode 代码块放错 | `week8/day4/README.md` L286-303（LC 72 属 Day3）, `day5/README.md` L225-239（LC 32 属 Day2） | 移到正确的天 |
| 9 | 性能目标虚标 | `week2/README.md` L9 "70%+" / `week2/day7/README.md` L732 "65%+" | 统一为实测值 "64.3%" 或提升到 70%+ 后更新 |
| 10 | mock 面试时长 | `week8/day5/README.md` L45 "30 分钟" | 改为 "34 分钟"（2040 秒） |
| 11 | 面试题数量虚标 | `week8/README.md` L9, `day7/README.md` L73 "50+" | 改为 "43+" 或补齐到真实 50+ |
| 12 | `week4/day7/README.md` Q6 | `</details>` 前有 `---` 破坏嵌套 | 移动 `---` 到 `</details>` 之后 |
| 13 | `week5/day7/README.md` Q5 | `</details>` 位置错误 | 同上 |
| 14 | `week6/day7/README.md` Q5 | `</details>` 缺失 | 补全 |
| 15 | Day 3-6 Q5 尾巴 | `week5/day3-6` 每篇 Q5 末尾有孤立"跨平台通用"句子 | 删除 |

**验证标准**：
- [ ] `grep -rn "/Users/chenbinbin" aiinfra/daily/` 无结果
- [ ] `grep -rn "../../images/" aiinfra/daily/week1/day*/README.md` 无错误引用
- [ ] Week7 的 6 天 LeetGPU 题目各不相同
- [ ] `week8` 的 LeetCode 总结表格与各天题目一致
- [ ] `grep -rn "50+" aiinfra/daily/week8/` 改为 "43+" 或已补齐

**预估工时**：2-3 小时

---

### 任务 4.2：工具升级

| # | 工具 | 当前问题 | 升级方向 |
|---|------|---------|---------|
| 1 | `week8/day5/kernels/mock_interview.py` | 只是带 prompt 的计时器 | 升级为 LLM 驱动的交互式面试官（调用 API 生成追问） |
| 2 | `week8/day6/kernels/knowledge_selftest.py` | exact-match 评分脆弱 | 改为归一化匹配（strip + lower + 容差数值比较） |
| 3 | `week8/day3/kernels/interview_basics.py` + `interview_advanced.py` | 只展示参考答案，不评估 | 加 LLM 评分模式（可选）或至少加关键词匹配提示 |
| 4 | `week8/day7/kernels/week8_summary.py` | 只 print 字符串 | 加交互式自测模式（从 30 题随机抽 10 题） |

**验证标准**：
- [ ] mock_interview 能生成至少 1 个 follow-up 问题
- [ ] knowledge_selftest 的 formula 模式支持数值容差（如 "58.45" 和 "58.5" 都算对）
- [ ] interview_basics/advanced 有简单的答案质量提示

**预估工时**：1-2 天

---

### 任务 4.3：补 vLLM V1 架构

**目标**：反映 2025 年 vLLM 现状（当前描述的是 SOSP 2023 原版）。

**操作**：
- 在 `week5/day3/README.md` 和 `week6/day3/README.md` 中加"vLLM V1 演进"章节
- 内容：AsyncLLMEngine、V1 scheduler、prefix caching 原生支持、chunked prefill 默认开启
- 面试题：≥2 道

**验证标准**：
- [ ] 有 ≥1 个章节描述 vLLM V1 架构变化
- [ ] 面试题 ≥2 道

**预估工时**：0.5 天

---

### Phase 4 整体验收清单

- [ ] 任务 4.1 ~ 4.3 全部完成
- [ ] `grep -rn "/Users/chenbinbin" aiinfra/daily/` 无结果
- [ ] Week7 LeetGPU 无重复
- [ ] Week8 面试题数量声明与实际一致
- [ ] 工具升级完成
- [ ] 提交 commit：`chore(daily): P3 仓库卫生修复 + 工具升级 + vLLM V1 补充`

---

## 整体里程碑与时间线

| 里程碑 | 预计工期 | 累计 | 交付物 |
|--------|---------|------|--------|
| M1: Phase 1 完成 | 1-2 周 | 2 周 | 参数统一、bug 修复、代码落盘 |
| M2: Phase 2 完成 | 3-5 周 | 5-7 周 | Tensor Core/CUTLASS/Triton/分布式/真 profiling/FA3+FlashDecoding |
| M3: Phase 3 完成 | 3-4 周 | 8-11 周 | Mini 引擎真整合、量化/CUDA Graph/高级特性/多硬件/backward/Ring Attention |
| M4: Phase 4 完成 | 1-2 周 | 9-13 周 | 仓库卫生、工具升级、vLLM V1 |

**关键路径**：Phase 1 → Phase 2（任务 2.5 真 profiling 依赖 Phase 1 的代码落盘）→ Phase 3（任务 3.1 引擎整合依赖 Phase 2 的 custom kernel 修复）→ Phase 4

---

## 整改后预期效果对照

| 维度 | 整改前 | 整改后（目标） |
|------|--------|--------------|
| 事实准确性 | ★★☆☆☆ | ★★★★★ |
| 代码工程实战 | ★★★☆☆ | ★★★★☆ |
| Profiling 实操 | ★★☆☆☆ | ★★★★☆ |
| 市场对标覆盖 | ★★☆☆☆ | ★★★★☆ |
| 仓库卫生 | ★★★☆☆ | ★★★★★ |
| 面试题数量 | ~43 道 | ~80+ 道（新增 ~25 道 P1 + ~20 道 P2） |
| 面试高频考点覆盖 | 缺 Tensor Core/CUTLASS/Triton/分布式/量化/FA3 | 全部补齐 |
| Mini 引擎可信度 | 独立 demo，假集成 | 真整合，可叙述 |

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| GPU 环境不可用（影响 profiling 实操） | 中 | Phase 2 任务 2.5 无法完成 | 使用云 GPU（AutoDL 等）或模拟数据标注"待实测" |
| CUTLASS 源码过于庞大 | 中 | 任务 2.2 工时超预期 | 聚焦 `gemm.h` 核心接口，不深入 CuteLAYOUT |
| Triton 环境搭建困难 | 低 | 任务 2.3 延迟 | 使用 `pip install triton`，或 Google Colab |
| Mini 引擎整合改动面大 | 中 | 任务 3.1 引入新 bug | 分步替换（先 softmax → 再 attention → 再 KV Cache），每步验证 |
| 新增内容破坏 8 周节奏 | 低 | 教程膨胀 | 新增内容标为"附录/扩展"，不打乱原 8 周主线 |

---

## 附录：整改任务全量清单（按优先级排序）

| 优先级 | 任务编号 | 任务名称 | 预估工时 | 依赖 |
|--------|---------|---------|---------|------|
| P0 | 1.1 | 统一硬件参数 | 4-6h | — |
| P0 | 1.2 | 修复事实错误 | 3-4h | — |
| P0 | 1.3 | 修复 CUDA bug | 4-6h | — |
| P0 | 1.4 | 代码落盘 | 3-4h | — |
| P1 | 2.1 | Tensor Core/WMMA 专题 | 2-3d | 1.4 |
| P1 | 2.2 | CUTLASS 源码分析 | 2-3d | 1.4 |
| P1 | 2.3 | Triton 语言专题 | 2-3d | — |
| P1 | 2.4 | 分布式推理专题 | 3-4d | — |
| P1 | 2.5 | 真实 Profiling 执行 | 2-3d | 1.1, 1.4 |
| P1 | 2.6 | FA3/FlashDecoding | 2-3d | 1.2 |
| P2 | 3.1 | Mini 引擎真整合 | 4-5d | 1.3, 2.1 |
| P2 | 3.2 | 量化专题 | 2-3d | 2.1 |
| P2 | 3.3 | CUDA Graph 实操 | 1-2d | 3.1 |
| P2 | 3.4 | 高级特性实操 | 2-3d | 3.1 |
| P2 | 3.5 | 多硬件对比（Ascend） | 1-2d | — |
| P2 | 3.6 | backward pass | 2-3d | 2.6 |
| P2 | 3.7 | Ring Attention | 1-2d | 2.4 |
| P3 | 4.1 | 仓库卫生修复 | 2-3h | — |
| P3 | 4.2 | 工具升级 | 1-2d | — |
| P3 | 4.3 | vLLM V1 补充 | 0.5d | — |
