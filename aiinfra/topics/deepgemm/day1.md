# Day 1（周一）：DeepGEMM 总览与 JIT 环境

> **今日目标**：理解 DeepGEMM 的定位、设计哲学（"借鉴 CuTe 但不依赖模板代数 + 运行时 JIT"），搭建环境跑通第一个 FP8 GEMM，建立源码地图
> **面试考察度**：⭐⭐⭐ 了解级，能说清 DeepGEMM 是什么、为什么 DeepSeek 要自研

---

### 学习任务 1：DeepGEMM 是什么（45 分钟）

#### 阅读内容
- **官方 README**：[DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)（注意 News 区，从 2025.02 到 2026.04 的演进）
- **背景**：[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) §4.1 FP8 训练
- **对比阅读**：回顾 [CUTLASS 专题 Day 7](../cutlass/day7.md) 的 Group GEMM

#### 核心要点

DeepGEMM 是 DeepSeek 开源的高性能 tensor core kernel 库，**所有 kernel 在运行时通过轻量 JIT 编译**，安装时无需 CUDA 编译。它把现代 LLM 的核心计算原语——FP8/FP4/BF16 GEMM、融合 MoE（Mega MoE）、MQA scoring、HyperConnection——统一到一个精简 CUDA 代码库里。

| 维度 | cuBLASLt FP8 | CUTLASS 4.x | Triton | DeepGEMM |
|------|--------------|-------------|--------|----------|
| FP8 性能（H800） | 基准 | ~95% cuBLAS | ~80% | ~1550 TFLOPS（~1.0x cuBLAS） |
| 代码可读 | 闭源 | 数十万行模板 | 中等 | 核心 ~3-5 千行，"clean and accessible" |
| JIT | 无 | 编译期模板 | 运行时 | 运行时（NVCC / NVRTC，10x 加速可选） |
| Block scaling | 不支持 | 支持 | 需手写 | 原生 per-128-channel |
| Grouped GEMM | 无 | GemmGroup | 需手写 | M-grouped（contiguous/masked/k-grouped） |
| MoE 融合 | 无 | 无 | 无 | **Mega MoE**（单 kernel 融合 EP+2×GEMM+SwiGLU） |
| 架构支持 | 全架构 | 全架构 | 全架构 | SM90 + SM100 |

> 💡 **一句话总结**：DeepGEMM 的设计哲学——借鉴 CuTe 的 TMA descriptor 与 WGMMA wrapper，但**不用** CuTe 的 Layout 代数与 CUTLASS 的 CollectiveBuilder；用裸 PTX + 轻量封装写到极致可读，靠 JIT 在运行时针对具体 shape 生成最优 kernel。README 原话："avoids heavy reliance on their templates or algebras... clean and accessible resource for learning NVIDIA GPU kernel optimization techniques."

#### 为什么 DeepSeek 要自己写

1. **236B/671B MoE 训练算力成本高**，FP8 是关键省算力手段（Hopper FP8 = 2× FP16 算力）
2. **标准 per-tensor FP8 在 MoE 上精度不够**，需要 per-128-channel scaling，cuBLASLt 不支持
3. **CUTLASS 模板太重**，迭代慢；Triton 在 Hopper 上达不到峰值（无法精细控制 warp specialization / 寄存器重配 / TMA multicast）
4. **MoE 场景需要 Grouped GEMM + EP 通信融合**，通用库无法做到单 kernel 融合（Mega MoE）
5. 于是自研一个"刚好够用、可读、可改、JIT"的库

#### 版本演进（News 时间线）

| 时间 | 里程碑 | 关键 PR |
|------|--------|---------|
| 2025.02 | 初始发布，FP8 GEMM + per-tensor scaling | — |
| 2025.04 | H800 达 **1550 TFLOPS** | [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74)/[#78](https://github.com/deepseek-ai/DeepGEMM/pull/78)/[#81](https://github.com/deepseek-ai/DeepGEMM/pull/81)/[#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) |
| 2025.05 | NVRTC JIT（10x 编译加速）+ 权重梯度 kernel | [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94)/[#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) |
| 2025.07 | **SM90 + SM100 双架构重构**，低 CPU 开销 JIT CPP 模块 | [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112) |
| 2025.09 | V3.2 MQA scoring kernel（lightning indexer） | [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200) |
| 2026.04 | **Mega MoE** + FP8xFP4 GEMM + FP4 indexer + PDL + 更快 JIT | [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) |

> ⚠️ **注意**：本专题以 v2.6.1（2026.04+）为基准，覆盖 SM90 + SM100、FP8/FP4、Grouped GEMM、Mega MoE 全部特性。早期版本的 per-tensor scaling 已被 per-128-channel scaling 取代，请以最新代码为准。

### 学习任务 2：JIT 编译机制（45 分钟）

DeepGEMM 最独特的设计是**运行时 JIT**——安装时不编译任何 CUDA，首次调用时按需编译并缓存。

#### JIT 流程

```
Python 调用 deep_gemm.fp8_gemm_nt(...)
    ↓
csrc/apis/gemm.hpp：根据 shape/dtype 选择模板参数
    ↓
csrc/jit_kernels/：渲染 .cu 源码（填入 BLOCK_M/N/K、stages 等编译期常量）
    ↓
调用 NVCC 或 NVRTC 编译（DG_JIT_USE_NVRTC=1 切换）
    ↓
缓存到 ~/.deep_gemm/（DG_JIT_CACHE_DIR 可配）
    ↓
后续调用直接加载缓存，零编译开销
```

#### 关键环境变量

| 变量 | 作用 | 默认 |
|------|------|------|
| `DG_JIT_USE_NVRTC` | 用 NVRTC 代替 NVCC（10x 快，个别 case 性能略低） | 0 |
| `DG_JIT_CACHE_DIR` | 编译缓存目录 | `~/.deep_gemm` |
| `DG_JIT_DEBUG` | 打印 JIT 调试信息 | 0 |
| `DG_PRINT_CONFIGS` | 打印每个 shape 选中的 config | 0 |
| `DG_JIT_DUMP_PTX` / `DG_JIT_DUMP_SASS` | dump PTX/SASS（调试用） | 0 |
| `DG_JIT_WITH_LINEINFO` | 嵌入源码行号（ncu profiling 用） | 0 |
| `DG_JIT_PTXAS_CHECK` | 断言无 local memory 使用 | 0 |

> 💡 **关键洞察**：JIT 让 DeepGEMM 能针对**每个具体 shape** 生成最优 kernel——`BLOCK_M/N/K`、`kNumStages`、`kNumTMAMulticast` 等都作为编译期常量嵌入，编译器可充分优化。这是它用"小代码量"达到"大库性能"的核心手段。代价是首次调用有编译延迟（NVRTC 可缓解）。

#### 运行时调参 API

```python
import deep_gemm

# 限制使用的 SM 数（留 SM 给其他 workload）
deep_gemm.set_num_sms(80)

# 设置近似 Tensor Core 利用率（影响 tile 调度策略）
deep_gemm.set_tc_util(0.9)

# 启用 PDL（Programmatic Dependent Launch，Hopper+ 的 kernel 间重叠）
deep_gemm.set_pdl(True)

# Grouped GEMM 的 M/K 对齐
deep_gemm.set_mk_alignment_for_contiguous_layout(128)
```

### 学习任务 3：环境搭建与第一个 GEMM（30 分钟）

```bash
cd DeepGEMM
./develop.sh    # 链接 include + 构建 CPP JIT 模块

# 跑 FP8 GEMM 正确性 + 性能测试
python3 tests/test_fp8_fp4.py
```

```text
# 预期输出（H800，截取）
Testing GEMM:
 > Perf (m=  8192, n=  8192, k=  8192, 1D1D, layout=NT, BF16, acc=0): 820.0 us | 1550 TFLOPS | ... GB/s | 1.02x cuBLAS
 > Perf (m= 16384, n= 16384, k= 16384, 1D1D, layout=NT, BF16, acc=0): 6.50 ms | 1568 TFLOPS | ... GB/s | 1.01x cuBLAS
Average FP8xFP8 GEMM speedup over cuBLASLt: 1.012x
```

### 学习任务 4：建立源码地图（30 分钟）

```
DeepGEMM/
├── deep_gemm/
│   ├── __init__.py             # Python 入口，导出所有 API
│   ├── include/deep_gemm/      # ★ 核心 C++/CUDA 头文件（header-only）
│   │   ├── impls/              #   ★ 各 kernel 实现
│   │   │   ├── sm90_fp8_gemm_1d1d.cuh       # Day 3-4 精读：SM90 FP8 GEMM
│   │   │   ├── sm90_fp8_gemm_1d2d.cuh       #   SM90 FP8 GEMM（2D scaling）
│   │   │   ├── sm90_bf16_gemm.cuh           #   SM90 BF16 GEMM
│   │   │   ├── sm90_fp8_mqa_logits.cuh      #   SM90 MQA logits（indexer）
│   │   │   ├── sm90_fp8_paged_mqa_logits.cuh
│   │   │   ├── sm90_tf32_hc_prenorm_gemm.cuh #   HyperConnection
│   │   │   ├── sm100_fp8_fp4_gemm_1d1d.cuh  # Day 6：Blackwell FP8/FP4
│   │   │   ├── sm100_bf16_gemm.cuh
│   │   │   ├── sm100_fp8_fp4_mega_moe.cuh   # Day 6：Mega MoE
│   │   │   ├── sm100_bf16_mega_moe.cuh
│   │   │   └── smxx_layout.cuh              #   布局转换
│   │   ├── mma/                #   ★ MMA 指令封装
│   │   │   ├── sm90.cuh        #     WGMMA（FP8MMA / BF16MMA / TF32MMA）
│   │   │   └── sm100.cuh       #     TCgen05（Blackwell）
│   │   ├── ptx/                #   ★ PTX 内联汇编
│   │   │   ├── wgmma.cuh       #     wgmma.fence / commit_group / wait_group
│   │   │   ├── tcgen05.cuh     #     Blackwell TCgen05 指令
│   │   │   ├── tma.cuh         #     TMA + tensormap.replace PTX
│   │   │   └── ld_st.cuh       #     ld_shared / st_shared
│   │   ├── scheduler/          #   ★ Block 调度器
│   │   │   ├── gemm.cuh        #     持久化调度 + Stream-K
│   │   │   ├── mega_moe.cuh    #     Mega MoE 调度
│   │   │   └── sm90/sm100_mqa_logits.cuh
│   │   ├── comm/barrier.cuh    #   mbarrier / grid_sync / nvlink_barrier
│   │   ├── common/             #   utils / math / types / tma_copy / compile
│   │   ├── epilogue/           #   store_cd / transform
│   │   └── layout/             #   sym_buffer / mega_moe / mqa_logits
│   ├── legacy/                 # A100 Triton kernel（SM80 回退）
│   ├── mega/                   # Mega MoE Python API（SymmBuffer 等）
│   ├── testing/                # bench_kineto / calc_diff
│   └── utils/                  # dist / layout / math
├── csrc/
│   ├── apis/                   # C++ API 层（gemm.hpp / attention.hpp / mega.hpp）
│   ├── jit_kernels/            # JIT 编译基础设施（heuristics + impls）
│   └── python_api.cpp          # pybind11 绑定
└── tests/                      # 测试与 benchmark
    ├── test_fp8_fp4.py         #   FP8/FP4 GEMM + Grouped
    ├── test_bf16.py            #   BF16 GEMM
    ├── test_mega_moe.py        #   Mega MoE
    ├── test_attention.py       #   MQA logits
    └── generators.py           #   测试数据生成器
```

#### 必读源码列表

| 文件 | 内容 | 优先级 | 对应 Day |
|------|------|--------|----------|
| `impls/sm90_fp8_gemm_1d1d.cuh` | SM90 FP8 GEMM 主 kernel | ⭐ 必读 | Day 3-4 |
| `mma/sm90.cuh` | WGMMA 指令封装 + smem desc 构造 | ⭐ 必读 | Day 3 |
| `scheduler/gemm.cuh` | 持久化 + Stream-K 调度器 | ⭐ 必读 | Day 4 |
| `ptx/wgmma.cuh` + `ptx/tma.cuh` | PTX 内联汇编 | ⭐ 必读 | Day 3 |
| `common/types.cuh` | GemmType / KernelType 枚举 | 📌 推荐 | Day 1 |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | Mega MoE | 📌 推荐 | Day 6 |
| `comm/barrier.cuh` | mbarrier / grid_sync / nvlink_barrier | 📌 推荐 | Day 4 |

### 今日检查清单

- [ ] 能说出 DeepGEMM 与 cuBLASLt/CUTLASS/Triton 的定位差异
- [ ] 能解释 DeepSeek 自研的 5 个原因（精度/MoE/JIT/可读/峰值）
- [ ] 能说出 JIT 编译流程与 `DG_JIT_USE_NVRTC` 的作用
- [ ] 成功跑通 `test_fp8_fp4.py`，记录 H800/H100 上的 TFLOPS
- [ ] 浏览了 `include/deep_gemm/` 目录，标记了 Day 3-4 精读文件

---

