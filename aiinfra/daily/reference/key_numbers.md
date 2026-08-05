# 面试必背数字清单（唯一事实源）

> ⚠️ **硬性规定**：教程正文引用以下数字时，一律以本文件与 `reference/hardware_specs.md` 为准，禁止在正文中另写一套。每个数字都标注了来源；标注 [需核实] 的条目引用前必须确认。

## 1. RTX 5090 三个核心数（张口就来）

| 数字 | 值 | 来源 |
|---|---|---|
| 峰值 FP32 | **104.75 TFLOPS** | [实测] `week1/day3/exercise/my_gpu_info.md` |
| 显存带宽 | **1792 GB/s**（GDDR7） | [实测] 同上 |
| Ridge Point | **58.45 FLOPs/byte** | [实测] 由上面两数推出 |

**Ridge Point 推导（面试口述版）**：

```text
峰值算力 = CUDA Cores × Clock × 2(FMA) = 21760 × 2.407 GHz × 2 ≈ 104.75 TFLOPS
显存带宽 = 14001 MHz × 2(DDR) × 512 bit / 8 ≈ 1792 GB/s = 1.792 TB/s
Ridge Point = 104.75 / 1.792 ≈ 58.45 FLOPs/byte
```

含义：kernel 的算术强度（FLOPs/byte）> 58.45 才可能打满算力，否则 memory-bound。判断流程：算 AI → 和 58.45 比 → 定瓶颈 → 选优化方向（compute-bound 提 ILP/Tensor Core，memory-bound 减访存/融合）。

## 2. KV Cache 显存口算

**公式（MHA 基准，FP16）**：

```text
bytes/token = 2(K,V) × n_layer × n_kv_head × d_head × 2 bytes
```

**LLaMA-7B 示例**（n_layer=32，n_head=32，d_head=128，MHA）：

```text
2 × 32 × 32 × 128 × 2 B = 524,288 B ≈ 524 KB/token（= 512 KiB）
```

| 序列长度 | KV Cache（LLaMA-7B，FP16） |
|---|---|
| 4K tokens | ~2 GB |
| 32K tokens | ~16 GB |
| 1M tokens | **~524 GB**（2×32×32×128×2B×1M；注意不是 16 GB——16 GB 是漏乘 n_layer=32 的经典错误） |

**注意力变体（一般形式）**：把 `n_kv_head` 换成变体实际值即可——GQA-8（如 LLaMA-3-8B）降为 1/4，MQA 降为 1/n_head，MLA 压缩到低秩维度 d_c。变体对比表见 `week10/day7` §2.2。

**变体口算示例**（LLaMA-7B 级别：n_layer=32, n_head=32, d_head=128, fp16，MHA 基准 524 KB/token）：

```text
MHA  (n_kv_head=32):  524 KB/token
GQA-8(n_kv_head=8):   131 KB/token   (1/4)
MQA  (n_kv_head=1):    16 KB/token   (1/32)
MLA  (d_c=512):        65 KB/token   (存潜在向量，形态不同)
```

来源：[实测/手算] 与 `week10/day7`、`week10/day7` 教程口径一致；公式本身为教科书定义。

## 3. GEMM 优化增益（唯一口径：week10/day7 实测）

来源：[实测] `week10/day7/README.md`（RTX 5090，sm_120，CUDA 12.8，`kernels/gemm_optimization_series.cu`，FP32，cuBLAS 基线）。**面试统一背这一套**：

**cuBLAS 占比（Our TFLOPS / cuBLAS TFLOPS）**：

| M=N=K | v1 Naive | v2 SharedMem | v3 RegBlk | v4 +float4 | v5 +写回 | v6 DblBuf | cuBLAS |
|---|---|---|---|---|---|---|---|
| 1024 | 18.5% | 22.8% | 21.3% | 41.1% | **42.2%** | 42.1% | 37.0 TFLOPS |
| 2048 | 10.9% | 14.2% | 37.0% | 59.8% | **62.3%** | 60.1% | 63.0 TFLOPS |
| 4096 | 10.6% | 13.3% | 30.8% | 64.3% | **62.9%** | 63.8% | 68.2 TFLOPS |

**三个必背结论**（全部有上面实测数据支撑）：

1. **float4 向量化加载是最大单步收益**：4096 矩阵 30.8% → 64.3%，几乎翻倍（128-bit load 把 global→shared 加载指令数砍 3/4）。
2. **同步式 Double Buffering 基本无效**：v5→v6 在噪声内（62.9% → 63.8%），真双缓冲要 `cp.async`（Ampere+）/ TMA（Hopper+）。
3. **小矩阵天花板低**：1024 矩阵只有 ~42%，因为 block 数 (1024/128)² = 64 < 170 SM，wave 填不满。

**理论分层口径**（week10/day7 面试要点的叙事版，用于"每一层收益来源"的口述框架，**与实测数字并存使用、不得混为一谈**）：SharedMem 1%→15% → RegBlk →45% → float4 →55% → Warp Shuffle →60% → Double Buffering →70% → Auto-tune →80% → Tensor Core/指令级 →90%+。说这套时必须声明是"经验分层估计"，实测锚点用上面的表。

## 4. 标准 Attention vs FlashAttention IO（统一口径）

**统一公式口径（FP32，读/写分开统计，S/P 各写 1 次读 1 次）**：

```text
标准 Attention HBM IO = 4N² + 4Nd（元素数）
  = 读 Q,K(2Nd) + 写读 S(2N²) + 写读 P(2N²) + 读 V 写 O(2Nd)
FlashAttention HBM IO = O(Nd)（只读 Q/K/V、只写 O，S/P 不落 HBM）
```

**N=4096, d=64, FP32 代值**（来源：`week10/day3` 推导 + `week10/day7` 实跑脚本理论值输出）：

| 实现 | HBM IO（4N²+4Nd 口径） | 备注口径（3N²+4Nd，读写合并统计） |
|---|---|---|
| 标准 Attention | ~260 MB | ~206 MB |
| FlashAttention | **4 MB**（4Nd × 4B） | 4 MB |
| **IO 加速比** | ~65x | **~50x** |

> 📌 **面试统一表述**："标准 Attention IO 约 206 MB、FA 约 4 MB，**IO 加速比 ~50x 量级**；但这是理论上限，wall-clock 实测只有 **2–8x**（GEMM 的 FLOPs 没减少，且标准版的 GEMM 部分本来就 compute-bound）。" 两种系数口径（4N² vs 3N²）说清即可，结论不变：N≫d 时 N² 项主导。
>
> ⚠️ week10/day3 现有 "~100x / ~206MB vs 2MB" 表格与本口径不一致，属待收敛项（A2 任务处理），正文写作以本节为准。
