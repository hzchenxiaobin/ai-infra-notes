# 昇腾950 NPU 架构白皮书精读

> 原文 PDF：[ascend950_npu_whitepaper.pdf](ascend950_npu_whitepaper.pdf)（华为，2026，共 40 页）
> 文档来源：[华为 OBS 公开下载](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)

---

## 1. Metadata

| 项目 | 内容 |
|---|---|
| Title | 昇腾950 NPU 架构白皮书 |
| 机构 | 华为技术有限公司 |
| Year | 2026 |
| 页数 | 40 |
| 覆盖型号 | 昇腾 950PR / 昇腾 950DT |
| 关键词 | DaVinciCore、灵衢 UB、HiF8、超节点、SIMD/SIMT、STARS2.0 |

> ⚠️ **注意**：白皮书全文只讲 **950PR 与 950DT** 两款，统一称"高速片上内存"（DRAM）与"超节点（Super Node）"；**未出现**"HiBL/HiZQ 自研 HBM"、"Atlas 950 SuperPoD/SuperCluster"等字样——这些名称见于华为其他发布材料，不属于本文档内容。

---

## 2. Summary

昇腾 950 系列是华为面向大模型时代的 NPU，采用"**共架构、双型号**"设计：同一套芯片架构通过搭配不同规格的片上内存，切分出面向推理 Prefill/推荐场景的 **950PR** 和面向训练全生命周期的 **950DT**。核心亮点：

- **第三代 DaVinciCore**：Cube + Vector 分离架构（AIC/AIV），业界首创 SIMD/SIMT 混合编程模型；
- **HiF8 自研 8bit 浮点**：动态范围接近 FP16，且无需 MXFP8 的 block-scale 因子；
- **灵衢 UB 2.0 互联**：片间双向带宽 2TB/s，同时提供 Load/Store 内存语义（UB Memory）与异步 RDMA 语义（URMA），并可通过 UBoE 跑在标准以太网上；
- **超节点规模**：从上代 384 卡提升至 **8192 卡**，整体集群超过 **128K 卡**，共享内存池最高 **128TB**。

---

## 3. 章节结构概览

| 章节 | 内容 |
|---|---|
| 第 1 章 | 关键术语（约 25 个定义） |
| 第 2 章 | 引言：AI 算力挑战（数据量 5 年从 64ZB → 近 500ZB；LLM 单次迭代通信量达数百 GB） |
| 第 3 章 | 架构概述：950PR/950DT 定位、主要特性、多 Die 合封、完整规格表 |
| 第 4 章 | 深度剖析（主体）：AI 子系统、AI CPU、内存子系统、STARS2.0 调度、DVPP、互连子系统、超节点能力 |
| 第 5 章 | 更多参考（《Ascend C 编程指南》《基于灵衢的超节点参考架构白皮书》） |

---

## 4. 双型号定位与差异

| | 昇腾 950PR | 昇腾 950DT |
|---|---|---|
| 定位 | 高性能推荐系统、大模型 **Prefill** 阶段、多模态推理 | 大模型全生命周期：预训练、后训练、推理（Decode+Prefill） |
| 片上内存 | 最高 128GB / 1.6TB/s（降配 112GB / 1.4TB/s） | 最高 144GB / 4TB/s（降配 96GB） |
| Cube Core 数 | 32 / 28 | 36 / 32 / 28 |
| Vector Core 数 | 64 / 56 | 72 / 64 / 56 |

两者通过冗余设计衍生多个降规版本（"/"分隔的数值即不同版本规格）。

---

## 5. 算力规格（原文表 3-1）

**Cube + Vector 总算力**（950PR / 950DT，满配在前）：

| 精度 | 950PR | 950DT |
|---|---|---|
| MXFP4 | 1784 / 1561 TFLOPS | 2007 / 1784 / 1561 TFLOPS |
| HiF8 / MXFP8 / FP8 | 919 / 804 TFLOPS | 1034 / 919 / 804 TFLOPS |
| INT8 | 919 / 804 TOPS | 1034 / 919 / 804 TOPS |
| BF16 / FP16 | 486 / 425 TFLOPS | 547 / 486 / 425 TFLOPS |
| TF32 | 243 / 212 TFLOPS | 273 / 243 / 212 TFLOPS |

**相对关系**（原文）：

- 相同频率下，HiF8/MXFP8/FP8 提供 2 倍 FP16 张量算力，MXFP4 提供 4 倍；
- 相比上一代 BF16，MXFP4 张量峰值算力提升 4 倍；
- Vector 单核 FP16/FP32 较上代提升 100%；FlashAttention 单核性能提升 1.5~2 倍。

**Vector 算力**（950DT 满配）：FP16/BF16 60 TFLOPS、FP32 30 TFLOPS、INT8 60 TOPS、INT32 15 TOPS、INT64 7 TOPS。

---

## 6. 芯片内部结构

多 Die 合封：**2 个 AI Die + 2 个 IO Die + 片上内存模块**（950PR 8 个 / 950DT 4 个），经 D2D Clink 和 Memory Interface 互连，构成 Chiplet UMA 整体（2 个 Die 间 L2 一致性由硬件维护）。

- **AI 子系统**：满配 36 个，每个含 1 Cube Core + 2 Vector Core（AIC/AIV 分离架构）；
- **AI CPU 子系统**：4 个 Cluster，每 Cluster 2 个自研 **Linx816**（ARMv8-A，物理双线程，支持 NEON）+ 4MB L3；满配 8C16T；
- **DVPP**：4 个子系统（4 VPC + 4 JPEGE + 8 JPEGD Core）；VPC 5760FPS@1080P，JPEGD 4096FPS@1080P（最大 32K×32K）；
- **STARS2.0 调度器**：HSCB 专用高速控制总线（ns 级调度开销）、2048 条 Host 下沉任务流、最多 16 个 AI CPU 任务 + 64 Host CPU 任务 + 64 UB jetty + 32 CCU 任务 + 32 SDMA 通道并发；算力切分（AIC/AIV/SDMA 最多 16 资源池）。

### 内存层次（原文表 4-2）

| 层级 | 容量 |
|---|---|
| L1 Buffer | 512KB / AI Core |
| L0A / L0B | 各 64KB |
| L0C | 256KB |
| Unified Buffer | 512KB / AI Core |
| CPU L1 / L2 | 64KB / 1MB per Core |
| CPU L3 | 4MB / Cluster |
| L2 Cache | 最高 128MB（512B Cache Line，4×128B Sector，支持 L2 Hint 与 CMO） |

内存 RAS 特性：Online ECC、巡检、预留行动态隔离。

---

## 7. 第三代 DaVinciCore 要点

- **随路量化**：Cube 核支持 L0C→UB 搬运时 FP32/INT32 → BF16/FP16/FP8/INT8 量化，及 NZ→ND/DN 排布转换；
- **Vector 核升级**：双发射 Register-Based SIMD，引入 RegFile；
- **SIMD/SIMT 新同构混合编程**（业界首创）：SIMD 为主、SIMT 为辅，基本块为 Vector Function；
- **CV 融合**：Cube→Vector 直通通道；
- **NDDMA**：支持最多 5 维数据重排；
- **BufferID 同步机制**：`get_buf` / `rel_buf` 类互斥锁语义。

### HiF8：华为自研 8bit 浮点

- 变长前缀码 Dot 域 + 原码阶码；
- **38 个指数表达**（FP8 E4M3 仅 18 个），综合阶码范围 $[-22, 15]$，接近 FP16 的 40 个；
- 不需要 MXFP8 的额外 8bit 缩放因子；
- 4 个特殊值编码（ZERO / NAN / ±INF）。

---

## 8. 互联与超节点（灵衢 UB 2.0）

**片间互联**：

| 接口 | 规格 |
|---|---|
| UB（灵衢总线） | 72 Lane HiLink SerDes（最高 112Gbps/lane），18 个 x4 Port，双向带宽 **2016GB/s（约 2TB/s）** |
| UBoE | 2×400Gbps（200GB/s 双向，与 UB 复用 SerDes 端口），支持 1x4 / 2x2 Port Bifurcation |
| PCIe | 5.0 x16，128GB/s 双向，兼容 GEN4/3/2/1，EP/RC 双模，与 UB 共用 4 个端口 |

**互联子系统组件**：

- **URMA**：异步内存拷贝语义（Jetty 队列 + Doorbell；RTP 可靠模式 4 Port / CTP 简易模式 9 Port）；
- **UB Memory**：同步 Load/Store/Atomic 访存语义，最高 **128TB** Host-Device / Device-Device 共享内存池；
- **CCU**：集合通信硬件加速（Broadcast / ReduceScatter / AllGather / AllReduce / All2All / All2Allv，内置 Reduce Unit + MemorySlice）；
- **UB On Chip Switch**：单 IO Die 内 9 个 x4 Port 转发，不占 DRAM 带宽、不进计算 Die；
- 拓扑：Clos、Full Mesh + Clos 混合、nD-Mesh；链路层重传 + 端到端可靠重传。

**规模**：超节点从上代 384 卡 → **8192 卡**；整体集群支持**超过 128K 卡**；超节点还支持超大内存池 / 超大存储池（UB 端口直连 CPU 侧内存与存储，免协议转换）。

---

## 9. 术语表

| 术语 | 含义 |
|---|---|
| **AIC / AIV** | AI Cube Core / AI Vector Core（AI Core 分离架构下的 Cube/Vector 核） |
| **AI CPU** | 片内自研 ARM 架构 CPU，昇腾 950 中指 **Linx816** |
| **CANN** | Compute Architecture for Neural Networks，昇腾异构计算软件栈 |
| **CCU** | 集合通信计算加速单元（Collective Communication Unit） |
| **CMO** | Cache Maintenance Operations，经 SDMA 的 L2 管理机制（Prefetch/Writeback/Invalid/Flush） |
| **CTP / RTP** | UB 的轻量级传输层模式 / 标准可靠传输层模式 |
| **DVPP** | DaVinci Vision Pre-Processing（JPEGD/JPEGE/VPC） |
| **HiF8** | 华为自研 8bit 锥形精度浮点格式 |
| **HSCB** | High Speed Control Bus，STARS 专用调度总线 |
| **MXFP4 / MXFP8** | OCP 微缩放（Microscaling）浮点格式 |
| **NCA** | Non-Cacheable Allocate |
| **NDDMA** | N 维 DMA 引擎，多维 Layout 搬运变换 |
| **SDMA** | 系统 DMA 引擎 |
| **STARS** | System Task and Resource Scheduler，片上调度和资源中心 |
| **UB（双义）** | Unified Buffer（AI Core 内 512KB 矢量缓存）/ Unified Bus（**灵衢总线**，华为统一互联协议） |
| **UBoE** | UB over Ethernet |
| **UB Memory** | UB 同步访存语义（Load/Store/Atomic） |
| **URMA** | UB Remote Memory Access，异步内存拷贝语义 |
| **UMA** | 统一内存访问架构 |
| **超节点（Super Node）** | 基于 UB 互连的大规模节点，8192 卡 |
| **Clos / nD-Mesh** | 组网拓扑 |
| **Sector Cache** | 128B 扇区缓存设计 |

---

## 10. 要点提炼（解读，非原文）

> 以下为面向 AI 基础设施从业者的分析性解读，不代表白皮书原文观点。

- **双型号分场景打法**对标 GPU 产品分层：950PR（低带宽大容量）≈ 面向推理 Prefill 和推荐的经济型 SKU；950DT（144GB / 4TB/s）规格介于 H100（80GB / 3.35TB/s）与 B200（192GB / 8TB/s）之间；BF16 满配 547 TFLOPS 显著低于 B200（约 2.25 PFLOPS dense），单卡算力差距靠**超节点规模**弥补——8192 卡超节点 + 2TB/s 片间互联，思路类似 NVL72 但规模放大一个量级。
- **灵衢 UB ≈ NVLink + NVSwitch + RDMA 的合体重构**：同时提供 Load/Store 内存语义（对标 NVLink C2C 内存共享，128TB 池）和 URMA 异步语义（对标 RDMA），UBoE 可直接跑标准以太交换机（对标 Ultra Ethernet 思路），降低对专用交换设备的依赖；UB Switch 承担类 NVSwitch 角色。
- **HiF8 是对 FP8 生态的差异化赌注**：在 OCP MXFP8 之外自研格式，主打更大动态范围（38 vs 18 指数）且免 block-scale 因子，面向训练易用性；生态接受度是关键风险点。
- **SIMD/SIMT 混合编程**明显向 CUDA 线程级编程模型靠拢（SIMT 处理 gather/scatter、分支等不规则场景），是对 CANN/Ascend C 生态短板的硬件级回应，降低 GPU 开发者迁移门槛。
- **推理侧系统设计**：KV Cache 压力靠"超节点 + CPU 超大内存池/存储池直连"解决，对应 GPU 世界的 KV Cache offloading / 分层存储方案；CCU 硬件卸载集合通信 ≈ NVLink SHARP / 在网计算的片内化。
- **片上内存表述刻意模糊**：全文只称"高速片上内存"，未出现 HBM 字样，结合"全栈自主可控的制造工艺"的表述，可解读为对供应链来源的淡化。

---

## 11. 推荐资源

- ⭐ [昇腾950 NPU 架构白皮书 PDF](ascend950_npu_whitepaper.pdf)（本文精读对象）
- 📌 《Ascend C 编程指南》（白皮书第 5 章引用）
- 📌 《基于灵衢的超节点参考架构白皮书》（白皮书第 5 章引用）
