# MoE 一周学习计划

> **适用对象**：已完成 [CUTLASS 专题](../cutlass/README.md) Day 3（3.x GEMM）与 Day 7（Group GEMM），掌握 GEMM Tiling、Tensor Core、CollectiveBuilder；建议读过 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) 对 DeepSeekMoE 有基本认知
> **本周目标**：理解 MoE（Mixture-of-Experts）的算法原理与系统瓶颈，能用 Triton/CUTLASS 写出 Gating、Top-K、Grouped GEMM 等核心算子，掌握 Expert Parallelism 的 all-to-all 通信模式，最终用 vLLM/Triton 组装一个可运行的 MoE 层并完成性能调优
> **时间投入**：工作日每天 2.5h（早间 1.5h + 晚间 1h），周末每天 5h，周计 22.5h
> **周日里程碑**：用 Triton 实现一个 Top-2 路由的 MoE FFN 层（含 Gating + Grouped GEMM + Combine），性能达到 Megatron-LM 参考实现 70%+，产出 ncu 性能报告

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 MoE 算法（稀疏路由、负载均衡）、核心算子（Gating/Top-K/Grouped GEMM/Scatter-Gather）、EP 通信（all-to-all）、推理优化（动态路由、专家预取、共享专家） |
| **核心产出** | ① Triton Gating + Top-K 融合 kernel ② CUTLASS/Triton Grouped GEMM ③ All-to-all token dispatch demo ④ Triton MoE FFN 层 ⑤ ncu 性能报告 ⑥ 源码精读笔记（vLLM `fused_moe`、Megatron `MoE`） |
| **验收标准** | ① 能画出 MoE 前向数据流（输入 → 门控 → 分派 → 专家 → combine） ② Gating+Top-K kernel 在 4096×8 专家上达到纯 PyTorch 5x+ ③ Grouped GEMM 达到逐专家调用 cuBLAS 90%+ ④ 能解释 EP 的 all-to-all token dispatch 时序 ⑤ 能用 ncu 定位 MoE 层的通信/计算占比 |
| **面试准备** | 积累 10-12 道 MoE 面试题，覆盖路由算法、负载均衡、Grouped GEMM、EP 通信、DeepSeek-MoE 细粒度专家、推理优化 |

### 本专题与 [CUTLASS 专题](../cutlass/README.md) 的边界

| 维度 | CUTLASS 专题（Day 7 Group GEMM） | 本 MoE 专题 |
|------|----------------------------------|-------------|
| **视角** | 算子层——Group GEMM 作为一种 GEMM 变体 | 系统层——Group GEMM 只是 MoE 的一环 |
| **范围** | 固定/变长 problem size 的批量 GEMM | 完整 MoE 层：Gating → Dispatch → Grouped GEMM → Combine → 负载均衡 |
| **通信** | 不涉及 | Expert Parallelism 的 all-to-all 是核心 |
| **算法** | 不涉及 | Top-K 路由、auxiliary loss、容量因子、共享专家 |
| **产出** | 调 `cutlass::gemm::device::GemmGroup` | 用 Triton 拼出完整 MoE FFN 层 |

> 💡 **一句话总结**：CUTLASS 专题教你"算" MoE 里的 Grouped GEMM，本专题教你"组装"整个 MoE 层——前者是后者的子问题。掌握本专题后，再读 vLLM 的 `fused_moe.py` 或 Megatron 的 `transformer/moe.py` 会如读散文。

### 本周知识图谱

![MoE 一周学习路径：Day 1-7 渐进式 Pipeline](../images/moe_learning_pipeline.svg)

### 前置准备清单

#### 硬件/软件验证
- [ ] GPU Compute Capability >= 8.0（Ampere 及以上，需 Tensor Core）
- [ ] CUDA Toolkit >= 12.0，PyTorch >= 2.1
- [ ] Triton >= 2.1（`python3 -c "import triton; print(triton.__version__)"`）
- [ ] CUTLASS >= 3.5（Day 3 的 Grouped GEMM 用到）
- [ ] NCCL >= 2.18（Day 4 的 all-to-all 用到），单机多卡即可
- [ ] Nsight Compute / Nsight Systems 可用

#### 验证命令
```bash
# 验证 GPU 与多卡（Day 4 all-to-all 至少 2 卡）
nvidia-smi --query-gpu=compute_cap,name --format=csv
nvidia-smi --query-gpu=index --format=csv | wc -l   # >= 2

# 验证 Triton
python3 -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)"

# 验证 NCCL all-to-all 可用
python3 -c "import torch.distributed as dist; print('nccl', dist.is_nccl_available())"
```

#### 必读资源（本周会反复用到）
- ⭐ [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) — DeepSeekMoE 细粒度专家 + 共享专家 + 设备受限路由
- ⭐ [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Shazeer 2017，MoE 门控奠基
- ⭐ [GShard](https://arxiv.org/abs/2006.16668) — 专家并行 + top-2 路由 + 容量因子
- 📌 [Switch Transformer](https://arxiv.org/abs/2101.03961) — top-1 路由 + 负载均衡损失
- 📌 [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) — 开源 MoE 标杆
- 📌 [vLLM `fused_moe.py`](https://github.com/vllm-project/vllm) — 工程级 MoE 推理实现

---


---

## 目录结构

```
aiinfra/topics/moe/
├── README.md                    # 本文件（一周学习计划）
├── kernels/                     # 可编译代码示例
│   ├── naive_moe.py             # Day 1: 朴素 PyTorch MoE
│   ├── triton_gating.py         # Day 2: Triton Gating + Top-K 融合
│   ├── triton_grouped_gemm.py   # Day 3: Triton Grouped GEMM
│   ├── ep_demo.py               # Day 4: 2 卡 Expert Parallelism demo
│   └── triton_moe.py            # Day 6: 完整 Triton MoE FFN
├── notes/                       # 源码精读笔记
│   ├── vllm_fused_moe.md        # Day 5: vLLM fused_moe 精读
│   ├── megatron_moe.md          # Day 4: Megatron MoE 通信
│   └── deepseek_moe.md          # Day 1/7: DeepSeekMoE 算法笔记
└── benchmark/                   # 性能对比
    ├── compare_moe.py           # Day 6: MoE 性能对比脚本
    └── report.md                # Day 6: 性能报告
```

> 💡 **后续延伸**：完成本专题后，建议结合 [CUTLASS 专题](../cutlass/README.md) Day 7 的 Group GEMM 与本专题 Day 4 的 EP 通信，进一步研究 DeepSeek-V3 的训练系统（FP8 + EP + PP 三维并行）。MoE 是当前大模型规模化的核心架构，掌握本周内容后再读 V3/R1 的工程报告会有"豁然开朗"的感觉。
