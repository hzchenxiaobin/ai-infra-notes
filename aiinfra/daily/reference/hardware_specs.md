# 硬件参数唯一事实源（Single Source of Truth）

> ⚠️ **硬性规定**：教程正文（`weekN/dayM/README.md`）中**禁止新写硬件参数数字**。凡涉及 GPU 算力、带宽、SM 数、显存等参数，一律引用本文件（`reference/hardware_specs.md`）或 `reference/key_numbers.md`。本文件数字改动必须同步更新 SKILL.md 自检清单的 grep 检查项。
>
> 标注约定：**[实测]** = 来自 `week1/day3/exercise/my_gpu_info.md`（RTX 5090 deviceQuery 实跑留档）；**[官方]** = 官方规格书/白皮书；**[需核实]** = 来源不一致或未经验证，引用前必须确认。

## 1. RTX 5090 实测参数表（本教程基准卡）

数据源：`aiinfra/daily/week1/day3/exercise/my_gpu_info.md`（deviceQuery 原始输出 + 手算复核，CUDA Driver 13.0 / Runtime 12.8）。

| 参数 | 数值 | 来源 |
|---|---|---|
| GPU 型号 | NVIDIA GeForce RTX 5090 | [实测] |
| 架构 | Blackwell（消费级，GB202） | [官方] |
| Compute Capability | 12.0（`sm_120`） | [实测] |
| 峰值 FP32 算力 | **104.75 TFLOPS**（21760 cores × 2.407 GHz × 2 FMA） | [实测]（与官方标称 ~104.8 一致） |
| 显存类型 / 容量 | GDDR7 / **32 GB**（32109 MB） | [实测] |
| 显存带宽 | **1792 GB/s**（14001 MHz × 2 DDR × 512-bit / 8） | [实测]（与官方标称一致） |
| **Ridge Point** | **58.45 FLOPs/byte**（104.75 TFLOPS ÷ 1.792 TB/s） | [实测]（推导见 `reference/key_numbers.md`） |
| SM 数量 | **170** | [实测] |
| FP32 CUDA Cores / SM | **128**（共 21760） | [实测] |
| 最大 threads / SM | **1536** | [实测] |
| 最大 warps / SM | **48**（1536 ÷ 32） | [实测]（由 threads/SM 推出） |
| 最大 blocks / SM | **24** | [官方]（CUDA Occupancy Calculator 口径，deviceQuery 不报告，[需核实]） |
| 最大 threads / block | 1024 | [实测] |
| Shared Memory / SM | **100 KB**（102400 B；每 block 上限 48 KB 静态 / 需动态申请更多） | [实测] |
| L2 Cache | **96 MB**（100663296 B） | [实测] |
| 寄存器 / SM | 65536 × 4 B = 256 KB | [实测]（deviceQuery 报 per-block 65536） |
| Warp size | 32 | [实测] |
| GPU Max Clock | 2407 MHz | [实测] |

## 2. 数据中心卡对比（A100 / H100 / B200）

用途：面试中"换一张卡结论怎么变"类问题。除标注 [需核实] 的条目外，均来自官方规格书。

| 参数 | RTX 5090 | A100（SXM） | H100（SXM） | B200 |
|---|---|---|---|---|
| 架构代际 | Blackwell 消费级（2025） | **Ampere**（2020） | **Hopper**（2022） | **Blackwell 数据中心**（2024，双 die） |
| Compute Capability | sm_120 | sm_80 | sm_90 | sm_100 |
| SM 数量 | 170 [实测] | 108 [官方] | 132 [官方] | [需核实]（公开资料 148 / 208 两种说法并存） |
| FP32 cores / SM | 128 | 64 | 128 | 128 [需核实] |
| 峰值 FP32 | **104.75 TFLOPS** | 19.5 TFLOPS [官方] | 67 TFLOPS [官方] | ~80 TFLOPS [需核实] |
| Tensor Core 代际 | 第 5 代 | 第 3 代 | 第 4 代 | 第 5 代（新增 FP4） |
| FP16/BF16 Tensor（dense） | ~209 TFLOPS [需核实] | 312 TFLOPS [官方] | 989.5 TFLOPS [官方] | ~2.25 PFLOPS [官方] |
| FP8 Tensor（dense） | 支持 [需核实] | 不支持 | 1979 TFLOPS [官方] | ~4.5 PFLOPS [官方] |
| 显存 | 32 GB GDDR7 | 40/80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e [官方]（部分渠道标 180 GB [需核实]） |
| 显存带宽 | **1792 GB/s** | 1555 GB/s（2039 GB/s @80GB）[官方] | 3350 GB/s（3.35 TB/s）[官方] | 8000 GB/s（8 TB/s）[官方] |
| L2 Cache | 96 MB | 40 MB [官方] | 50 MB [官方] | [需核实]（公开资料 96–128 MB 不一） |
| Shared Mem / SM（上限） | 100 KB [实测] | 164 KB [官方] | 228 KB [官方] | 228 KB [需核实] |
| 最大 threads / SM | 1536 [实测] | 2048 [官方] | 2048 [官方] | 2048 [需核实] |
| Ridge Point（FP32） | **58.45** | ~12.5（19.5 ÷ 1.555） | ~20（67 ÷ 3.35） | ~10 [需核实] |

> 💡 **教学要点**：消费卡（RTX 5090）算力/带宽比远高于数据中心卡（58.45 vs ~10–20），意味着同一个 kernel 在 5090 上更容易落入 memory-bound 区；A100 的 19.5 TFLOPS / 1555 GB/s（即 Ridge Point ~12.5）正是教程历史版本中 "12.6" 数字的来源——那是 **A100 的参数**，不是 RTX 5090 的，引用时注意区分。

## 3. 历史错误数字对照（防复发）

| 错误数字 | 真实身份 | 正确值（RTX 5090） |
|---|---|---|
| 19.5 TFLOPS | A100 FP32 峰值 | 104.75 TFLOPS |
| 1.55 TB/s（1555 GB/s） | A100 40GB 显存带宽 | 1792 GB/s（GDDR7，非 HBM） |
| Ridge Point 12.6 | A100 的 Ridge Point | 58.45 |
| "108 个 SM" | A100 的 SM 数 | 170 |

以上数字只允许出现在"A100 对比 / 错误示范"的显式标注语境（SKILL.md 自检清单含对应 grep 检查）。
