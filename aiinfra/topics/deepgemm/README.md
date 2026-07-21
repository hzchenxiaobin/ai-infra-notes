# DeepGEMM 一周学习计划

> **适用对象**：已完成 [CUTLASS 专题](../cutlass/README.md) Day 4（三层抽象）与 [CuTe 专题](../cute/README.md) Day 6（TMA + WGMMA），掌握 Hopper 异步执行模型；建议读过 [FlashAttention-3 论文精读](../../paper/flashattention3/README.md) 理解 warp specialization 与 FP8 布局工程
> **本周目标**：理解 DeepSeek 开源的 DeepGEMM 库（v2.6.1，SM90 + SM100）的设计哲学与核心实现，能读懂其 warp-specialized FP8 GEMM kernel 源码（TMA + WGMMA + 持久化调度 + per-128-channel scaling），掌握 Grouped GEMM for MoE 与 Mega MoE 融合 kernel，最终用 DeepGEMM 跑出接近 H100/H800 FP8 峰值（~1550 TFLOPS）并完成 ncu 调优报告
> **时间投入**：工作日每天 2.5h（早间 1.5h + 晚间 1h），周末每天 5h，周计 22.5h
> **周日里程碑**：用 DeepGEMM 跑通 FP8 GEMM + Grouped GEMM + Mega MoE 三类核心 kernel，在 Hopper 上达到 85%+ FP8 峰值利用率，产出源码精读笔记与性能对比报告（DeepGEMM vs cuBLASLt vs CUTLASS 3.x）

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 DeepGEMM 的设计哲学（轻量 JIT + 借鉴 CuTe 但不依赖模板代数）、FP8/FP4 数据类型与 per-128-channel scaling、SM90 TMA + WGMMA + 持久化调度、warp specialization 与寄存器重配、Grouped GEMM（contiguous/masked/k-grouped）、Mega MoE 融合 kernel、SM100 TCgen05 |
| **核心产出** | ① DeepGEMM JIT 编译运行环境 ② FP8 GEMM 性能数据（vs cuBLASLt/CUTLASS）③ Grouped GEMM for MoE benchmark ④ `sm90_fp8_gemm_1d1d.cuh` 源码精读笔记 ⑤ Mega MoE 时序图 ⑥ ncu 瓶颈分析报告 |
| **验收标准** | ① 能画出 DeepGEMM 的 TMA warpgroup + Math warpgroup 时序图并标注 barrier 握手点 ② 能解释 per-128-channel scaling 的数学形式与 SM100 UE8M0 的差异 ③ FP8 GEMM 在 8192×8192 达到 H800 FP8 峰值 85%+ ④ 能说出 DeepGEMM 的 Grouped GEMM 只分组 M 轴的设计原因 ⑤ 能用 ncu 定位 stall reasons 并解读 Tensor Core 利用率 |
| **面试准备** | 积累 10-12 道面试题，覆盖 FP8 精度、warp specialization、TMA/WGMMA 异步、持久化调度、per-128-channel scaling、Mega MoE、与 CUTLASS/CuTe 对比 |

### 本专题与 [CUTLASS 专题](../cutlass/README.md) / [CuTe 专题](../cute/README.md) 的边界

| 维度 | CUTLASS 专题 | CuTe 专题 | 本 DeepGEMM 专题 |
|------|--------------|-----------|-------------------|
| **视角** | 库——用模板拼 GEMM | 原语——Layout/Tensor/Copy | 单点——FP8/FP4 GEMM 与 MoE 融合 kernel |
| **范围** | 全精度 GEMM/Conv/Epilogue | 通用 kernel 组装框架 | FP8/FP4/BF16 GEMM + Grouped + MoE + MQA + HC |
| **抽象层** | CollectiveBuilder + CuTe | Layout 代数 + Tensor engine | 裸 PTX + 轻量 CuTe（仅用 TMA desc / WGMMA wrapper） |
| **架构** | 全架构通用 | 全架构通用 | SM90（Hopper）+ SM100（Blackwell） |
| **JIT** | 编译期模板 | 编译期模板 | **运行时 JIT**（NVCC / NVRTC），安装时不编译 |
| **独有** | Epilogue 融合树 EVT | Layout 代数 | Mega MoE（单 kernel 融合 EP + 2×GEMM + SwiGLU） |

> 💡 **一句话总结**：DeepGEMM 的设计哲学是"借鉴 CuTe 的 TMA/WGMMA wrapper，但不用它的模板代数；用裸 PTX 写到极致可读，靠 JIT 在运行时针对具体 shape 生成最优 kernel"。CUTLASS/CuTe 专题教你"组装" GEMM 的通用框架，本专题教你"拆开"一个生产级 FP8 kernel 看里面每一行 PTX 怎么写——前者是广度，后者是深度。

### 本周知识图谱

![DeepGEMM 一周学习路径：Day 1-7 渐进式 Pipeline](../images/deepgemm_learning_pipeline.svg)

### 前置准备清单

#### 硬件/软件验证
- [ ] GPU Compute Capability == 9.0a（Hopper H100/H800）或 10.0a（Blackwell B200）
- [ ] CUDA Toolkit >= 12.3（SM90，**推荐 12.9+**）；>= 12.9（SM100）
- [ ] Python >= 3.8，PyTorch >= 2.1（Mega MoE 需 >= 2.9 的对称内存 API）
- [ ] C++20 编译器（NVCC 12.3+ 支持）
- [ ] CUTLASS 4.0+（Git submodule，DeepGEMM include 它的 `cute/` 与 `cutlass/arch/` 头文件）
- [ ] `{fmt}` 库（Git submodule）
- [ ] Nsight Compute >= 2024.1（能解读 sm_90a 的 WGMMA 指标）

#### 验证命令
```bash
# 验证 GPU 架构（需 9.0 或 10.0+）
nvidia-smi --query-gpu=compute_cap,name --format=csv
# 预期输出：9.0 / H100 / H800  或  10.0 / B200

# 验证 CUDA Toolkit（SM90 需 12.3+，SM100 需 12.9+）
nvcc --version

# 验证 PyTorch
python3 -c "import torch; print(torch.__version__)"
```

#### 克隆 DeepGEMM
```bash
# 必须带 --recursive（CUTLASS 与 fmt 是 submodule）
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && git describe --tags
# 确认版本 >= v2.6.0

# 开发模式：链接 include + 构建 CPP JIT 模块
cat develop.sh
./develop.sh

# 或安装模式
cat install.sh
./install.sh
```

#### 必读资源（本周会反复用到）
- ⭐ [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM) — 官方接口文档与 News（含 Mega MoE、MQA、FP4）
- ⭐ [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) — FP8 训练动机与精度策略
- ⭐ [FlashAttention-3 论文精读](../../paper/flashattention3/README.md) §3-4 — warp specialization + FP8 布局工程（与本专题共享底层机制）
- 📌 [Hopper WGMMA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-group-matrix-multiply-instructions) — 指令格式与布局约束
- 📌 [DeepEP](https://github.com/deepseek-ai/DeepEP) — DeepGEMM Mega MoE 的对称内存与 EP 通信搭档
- 📌 [CUTLASS 专题 Day 7](../cutlass/day7.md) Group GEMM — 对照 DeepGEMM 的 M-grouped 设计

---


---

## 目录结构

```
aiinfra/topics/deepgemm/
├── README.md                    # 本文件（一周学习计划）
├── kernels/                     # 可编译代码示例
│   ├── fp8_scaling_demo.py      # Day 2: per-128-channel scaling 精度对比
│   ├── grouped_gemm_demo.py     # Day 5: MoE Grouped GEMM demo
│   └── mega_moe_demo.py         # Day 6: Mega MoE 简化 demo
├── notes/                       # 源码精读笔记
│   ├── sm90_fp8_gemm_1d1d.md    # Day 3-4: 主 kernel 精读
│   ├── scheduler_gemm.md        # Day 4: 持久化调度器
│   ├── grouped_gemm.md          # Day 5: Grouped GEMM 三变体
│   └── mega_moe.md              # Day 6: Mega MoE 时序
└── benchmark/                   # 性能对比
    ├── compare_fp8_gemm.py      # Day 7: 三版本对比脚本
    └── report.md                # Day 7: 性能报告
```

> 💡 **后续延伸**：完成本专题后，建议回到 [CUTLASS 专题](../cutlass/README.md) Day 3-4 重读 `collective/sm90_mainloop.hpp`——你会发现 DeepGEMM 的手写 PTX 与 CUTLASS 的 CuTe 封装在做同一件事，只是抽象层不同。再读 [MoE 专题](../moe/README.md) Day 5 的 vLLM `fused_moe` 时，也能看清"DeepGEMM 的 Grouped GEMM / Mega MoE 是如何被 MoE 框架调用的"。配合 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的低延迟 EP kernel，你能拼出 DeepSeek-V3.2 推理的完整算子栈。DeepGEMM 是连接硬件特性与上层框架的关键一环，掌握它后再读任何 Hopper/Blackwell kernel 都会有"豁然开朗"的感觉。
