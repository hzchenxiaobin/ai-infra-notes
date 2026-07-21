# CuTe 一周学习计划

> **适用对象**：已完成 [CUTLASS 专题](../cutlass/README.md) Day 2（CuTe 编程模型入门），掌握 Layout/Tensor/copy 基本用法；或已完成 week2 GEMM 手写实战、对 CUTLASS 3.x 有基本认知
> **本周目标**：从"会调用 CuTe API"升级到"能读懂 CuTe 源码并用 CuTe 原语组装高性能 kernel"，深入 Layout 代数、Tensor 引擎、Copy 原语、Swizzle 与 TMA，最终用 CuTe 原语写出一个达到 cuBLAS 70%+ 的 GEMM
> **时间投入**：工作日每天 2.5h（早间 1.5h + 晚间 1h），周末每天 5h，周计 22.5h
> **周日里程碑**：用 CuTe 原语（Layout + Tensor + copy + MMA）实现一个 FP16 GEMM，达到 cuBLAS 70%+，产出源码精读笔记与性能报告

---

## 本周总览

| 维度 | 内容 |
|------|------|
| **整体目标** | 掌握 CuTe 的 Layout 代数、Tensor 引擎分层、Copy 原语体系、Swizzle 作为 Layout、TMA descriptor 与 warp-specialized 流水线 |
| **核心产出** | ① CuTe Layout 代数实验集 ② 自定义 Swizzle 的 shared memory 加载 ③ TMA-based GEMM ④ CuTe 源码精读笔记（`cute/tensor.hpp`、`cute/swizzle_layout.hpp`、`cute/copy.hpp`）⑤ 性能对比报告 |
| **验收标准** | ① 能手算嵌套 Layout 的偏移并写出 `coalesce`/`zipped` 后的等价 Layout ② 能解释 Swizzle 作为 Layout 的数学含义并画出 XOR 映射图 ③ 能用 `cute::copy` + `cp.async`/TMA 写出 3-stage 流水线 ④ CuTe GEMM 达到 cuBLAS 70%+（4096×4096 FP16）⑤ 能用 `ncu` 分析 CuTe kernel 的 stall reasons 并定位瓶颈 |
| **面试准备** | 积累 10-12 道 CuTe 面试题，覆盖 Layout 代数、Tensor 抽象、Copy 选择策略、Swizzle 原理、TMA 与 cp.async 对比、CuTe 与 CUTLASS 2.x 索引方式对比 |

### 本专题与 [CUTLASS 专题](../cutlass/README.md) 的边界

| 维度 | CUTLASS 专题（Day 2） | 本 CuTe 专题 |
|------|----------------------|--------------|
| **深度** | API 使用层——会调 `make_layout`/`make_tensor`/`copy` | 源码层——理解 Layout 代数、Tensor engine 分层、Swizzle 作为 Layout |
| **范围** | CuTe 作为 CUTLASS 3.x 的子模块简介 | CuTe 作为独立的 kernel 组装框架，脱离 GEMM 模板也能用 |
| **TMA** | 提一句"CuTe 自动用 TMA" | 深入 TMA descriptor 构建、`cute::TmaCopy`、warp specialization |
| **Swizzle** | "copy 自动加 swizzle" | 手写 Swizzle、XOR 映射推演、MMA swizzle pattern |
| **产出** | 调 `CollectiveBuilder` 跑通 GEMM | 用 CuTe 原语**手写** GEMM（不调 CollectiveBuilder） |

> 💡 **一句话总结**：CUTLASS 专题教你"用" CuTe，本专题教你"懂" CuTe——前者调 `CollectiveBuilder` 自动出 kernel，后者拆开 `CollectiveBuilder` 看里面是怎么用 CuTe 原语拼出来的。掌握本专题后，再读 CUTLASS 3.x 的 `collective/mainloop.hpp` 会如读散文。

### 本周知识图谱

![CuTe 一周学习路径：Day 1-7 渐进式 Pipeline](../images/cute_learning_pipeline.svg)

### 前置准备清单

#### 硬件/软件验证
- [ ] GPU Compute Capability >= 8.0（Ampere 及以上；Day 5/6 的 TMA 内容需 >= 9.0a 即 Hopper）
- [ ] CUDA Toolkit >= 12.0（CuTe 独立头文件需 12.x 的 `cuda/barrier`）
- [ ] CUTLASS >= 3.5（含稳定 CuTe，`git clone https://github.com/NVIDIA/cutlass.git`）
- [ ] CMake >= 3.18，Nsight Compute 可用

#### 验证命令
```bash
# 验证 GPU 架构（Day 5/6 需要 9.0+ 才能跑 TMA）
nvidia-smi --query-gpu=compute_cap,name --format=csv
# 预期输出：8.0 / 8.6 / 8.9 / 9.0 / 10.0 / 12.0

# 验证 CuTe 头文件可独立 include（不依赖 cutlass/gemm）
nvcc -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17 -x cu -E - < /dev/null \
  -include cute/tensor.hpp 2>&1 | tail -5
# 预期：无报错，说明 CuTe 头文件可独立编译
```

#### 克隆 CUTLASS（含 CuTe）
```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git describe --tags
# 确认版本 >= v3.5.0
```

#### 必读 CuTe 资源（本周会反复用到）
- ⭐ [CuTe 源码目录](https://github.com/NVIDIA/cutlass/tree/main/include/cute) — `include/cute/`
- ⭐ [CuTe Tutorial 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) — `examples/cute/`（本周 Day 7 的对标目标）
- 📌 [CuTe GTC 2023 演讲](https://developer.nvidia.com/gtc/2023/video/s40095) — CuTe 设计哲学官方讲解

---


---

## 目录结构

```
aiinfra/topics/cute/
├── README.md                  # 本文件（一周学习计划）
├── kernels/                   # 可编译代码示例
│   ├── cute_layout_algebra.cu    # Day 2: Layout 代数实验
│   ├── cute_tensor_engines.cu    # Day 3: Tensor 引擎分层
│   ├── cute_copy_pipeline.cu     # Day 4: copy 原语 + 3-stage 流水线
│   ├── cute_swizzle_demo.cu      # Day 5: Swizzle 作为 Layout
│   └── cute_gemm.cu              # Day 7: CuTe 原语组装 GEMM
├── notes/                     # 源码精读笔记
│   ├── layout_algebra.md         # Day 2: layout.hpp 代数运算
│   ├── tensor_engine.md          # Day 3: tensor.hpp engine 分层
│   ├── copy_dispatch.md          # Day 4: copy.hpp 调度机制
│   └── swizzle_layout.md         # Day 5: swizzle_layout.hpp 原理
└── benchmark/                 # 性能对比
    ├── compare_copy.py           # Day 4: copy 策略对比
    └── gemm_report.md            # Day 7: CuTe GEMM 性能报告
```

> 💡 **后续延伸**：完成本专题后，建议回到 [CUTLASS 专题](../cutlass/README.md) 重读 Day 3-4 的 `collective/mainloop.hpp` 源码——你会发现自己能看懂里面每个 `compose`/`local_partition`/`copy(tma_load, ...)` 的含义。CuTe 是 CUTLASS 3.x 的"底层语言"，掌握它后再读 CUTLASS 会有"豁然开朗"的感觉。
