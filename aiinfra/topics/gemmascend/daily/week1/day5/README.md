# 第 1 周 · 第 5 天：建立 Ascend 硬件与流水线概念

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 1 周"GEMM 与 Ascend 基础"的第 5 天教程，可独立阅读，不依赖前序内容。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 60 分钟，源码导读约 40 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 说出 **GM、L1、L0A、L0B、L0C、UB** 六层存储各自的职责与容量，并指出仓库中对应的地址空间修饰符（`__gm__`/`__cbuf__`/`__ca__`/`__cb__`/`__cc__`/`__ubuf__`）和类型别名（`gm_ptr`/`l1_ptr`/`l0a_ptr`/`l0b_ptr`/`l0c_ptr`/`ub_ptr`）；
2. 用一笔容量账解释"为什么大矩阵必须 Tiling + 分层搬运"：单核 L1 只有 512 KiB，而一个真实 GEMM 的输入有几百 MiB；
3. 解释 **AIC（Cube 核）与 AIV（Vector 核）的分工**：`__mix__(1, 2)` 混合核里谁负责搬运与 MAD、谁负责 epilogue 写回，以及 `kDirectStore` 时为什么可以不要 AIV；
4. 说出 **MTE2、MTE1、M、FIX、V、MTE3** 各条流水线在 GEMM 数据流中搬运/计算的那一段，并用 `asc_sync_notify`/`asc_sync_wait` 解释"满事件/空事件"握手；
5. 解释**多级 stage（多缓冲）** 如何让搬运引擎与计算单元重叠工作，以及 stage 数受哪些容量与事件数限制；
6. 跟着 `bf16_gemm_impl` 源码，完整走一遍 `GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM` 的数据流，独立画出当天产出的数据流图。

## 二、预备知识

- **第 1 天**：M/N/K 维度、计算量 `2MNK` 与数据量。今天用数据量公式算"输入为什么放不进片上内存"。
- **第 3 天**：`BLOCK_M/N/K` 与 Tiling。今天把软件视角的"Tile"落进硬件视角的"存储层级"——L1 里放的是 A/B 的 K 块，L0C 里放的是输出 Tile 的分片。
- **第 4 天**：FP32 累加器。今天会看到它的物理实体：L0C 就是那块 256 KiB、以 `float` 累加的片上缓冲（`common.hpp` 第 57 行注释 `// L0C (fp32 accumulator)`）。
- **不需要 Ascend 硬件**：今天全部内容是概念建立 + 静态源码阅读 + 纸笔演算；带硬件的可选实验放在实践步骤末尾。

背景一句话：前四天我们一直在"软件世界"里讨论 GEMM——布局、Tiling、数值类型。今天第一次进入"硬件世界"：仓库 `deep_gemm/include/deep_gemm/` 下的每个 device kernel，本质上都是围绕一套固定的存储层级和流水线在编排数据。看懂这套骨架，后面三周读 `bf16_gemm.hpp`、`fp8_gemm.hpp` 时就只是往骨架上挂细节。

## 三、详细讲解

### 3.1 为什么需要存储层次：一笔容量账

第 1 天的思考题是"为什么大矩阵不能一次性放进片上内存"。现在用仓库的真实常量把账算实。硬件常量全部定义在 `deep_gemm/include/deep_gemm/common.hpp` 第 50～61 行（注释标明 "Ascend 950DT core topology"）：

```cpp
inline constexpr uint32_t kNumAICores = 32;
inline constexpr uint32_t kNumAIVs = 2 * kNumAICores;   // 64

inline constexpr uint32_t L1SizeBytes = 512 * 1024;   // L1 per core
inline constexpr uint32_t L0ASizeBytes = 64 * 1024;   // L0A
inline constexpr uint32_t L0BSizeBytes = 64 * 1024;   // L0B
inline constexpr uint32_t L0CSizeBytes = 256 * 1024;  // L0C (fp32 accumulator)
inline constexpr uint32_t UBSizeBytes = 256 * 1024;   // UB
```

对照 README 性能表里的 shape `M=4096, N=7168, K=16384`（BF16）：

| 数据 | 大小 | 与片上缓冲的差距 |
| :--- | :--- | :--- |
| A `[4096, 16384]` | 128 MiB | 单核 L1（512 KiB）的 256 倍 |
| B `[7168, 16384]` | 224 MiB | 单核 L1 的 448 倍 |
| A + B 合计 | 352 MiB | 单核 L1 的 **704 倍** |
| D `[4096, 7168]` | 56 MiB | 单核 UB（256 KiB）的 224 倍 |

结论和第 1 天一致，但现在有了数字：**任何一层片上缓冲都装不下完整输入，必须靠第 3 天的 Tiling 切成小块，逐块搬进来、算完再搬出去**。反过来，片上缓冲也不能太小——L1 里要同时放下"正在算的 K 块"和"预取的 K 块"（多级 stage，见 3.5），L0C 里要放下输出 Tile 的分片。硬件层级就是在这两个约束之间设计的：

```text
容量大、带宽低                          容量小、带宽高、离计算单元近
┌────────────────────────────────────────────────────────────┐
│  GM (HBM, ~GiB 级)                                          │  存完整 A/B/D
│    │                                                        │
│    ▼  MTE2 搬运（经 L2 缓存，kernel 用 l2_ctrl 提示）           │
│  L1  512 KiB / 核 —— GM 与 L0 之间的中转站、多级 stage 缓冲       │
│    │                                                        │
│    ▼  MTE1 搬运                                              │
│  L0A 64 KiB ──┐                                             │
│  L0B 64 KiB ──┤ MAD（cube 矩阵乘加）                           │
│                ▼                                             │
│  L0C 256 KiB（FP32 累加器）                                    │
│    │                                                        │
│    ▼  FIX 搬运                                               │
│  UB 256 KiB —— AIV 的向量计算缓冲 / epilogue 中转               │
└────────────────────────────────────────────────────────────┘
```

（L2 是 GM 侧的缓存：`common.hpp` 第 60 行 `kL2SectorBytes = 128`，搬运函数都带 `asc_load_l2_cache_mode`/`asc_store_l2_cache_mode` 参数控制缓存提示，今天不展开。）

### 3.2 六层存储的职责：地址空间修饰符与类型别名

Ascend 的设备编译器（Bisheng）用**地址空间修饰符**区分不同的存储层级——同一段 C++ 代码里，`__cbuf__ T*` 与 `__ubuf__ T*` 指向的是两块完全不同的物理内存。仓库把这层信息封装在 `deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp` 里：

| 层级 | 修饰符 | 仓库类型别名（tile_ptr.hpp 第 93～97 行） | 容量 | 在 GEMM 中的角色 |
| :--- | :--- | :--- | :--- | :--- |
| GM（Global Memory） | `__gm__` | `gm_ptr<T, Major>`（定义在 `common.hpp` 第 137 行） | GiB 级（HBM） | 存完整的 A、B、D，host 传入的指针都在这里 |
| L1 | `__cbuf__` | `l1_ptr<T, M> = operand_ptr<__cbuf__ T*, T, M>` | 512 KiB/核 | GM→L0 的中转；A/B 的 K 块按多级 stage 轮换驻留 |
| L0A | `__ca__` | `l0a_ptr<T> = operand_ptr<__ca__ T*, T>` | 64 KiB | MAD 的 A 操作数专用缓冲 |
| L0B | `__cb__` | `l0b_ptr<T> = operand_ptr<__cb__ T*, T>` | 64 KiB | MAD 的 B 操作数专用缓冲 |
| L0C | `__cc__` | `l0c_ptr<T> = result_ptr<__cc__ T*, T>` | 256 KiB | MAD 的输出与 FP32 累加器 |
| UB（Unified Buffer） | `__ubuf__` | `ub_ptr<T> = result_ptr<__ubuf__ T*, T>` | 256 KiB | AIV 的向量计算与 epilogue 中转缓冲 |

三个观察：

1. **L0A/L0B/L0C 是三块独立的缓冲，不是一块"缓存"**。MAD 单元有两个独立的操作数端口（A 与 B）和一个累加输出口，`mad()`（`ascend/copy.hpp` 第 370～381 行）同时从 `l0a`/`l0b` 取数、把结果累加进 `l0c`。这就是为什么仓库要为它们分别声明指针并分别做容量检查。
2. **这些"指针"其实是携带形状的句柄**。`operand_ptr`（tile_ptr.hpp 第 18 行起）除了地址 `addr`，还带 `shape_mn`、`shape_k` 和 major 信息；`operator[]`（第 39～43 行）按下标返回"偏移了一个 stage 大小"的新句柄——这正是多级缓冲的地址计算方式：`size_per_stage() = shape_mn * shape_k * sizeof(T)`（第 37 行），`l1a[s]` 就是第 `s` 级 stage。
3. **容量是编译期断言强制的**。`bf16_gemm.hpp` 第 59～69 行把每一层的使用量都 `static_assert` 住：

```cpp
constexpr l1_ptr<bfloat16_t, kMajorA> l1a(BLOCK_M, ..., 0);          // 从地址 0 开始
constexpr l1_ptr<bfloat16_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1Stages));  // 紧跟 l1a 之后
constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);                     // 地址 0
constexpr l0b_ptr<bfloat16_t> l0b(MAD_N, MAD_K);
constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes,  "L1 overflow");
static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes, "L0A overflow");
static_assert(l0b.offset(kNumL0Stages) <= L0BSizeBytes, "L0B overflow");
static_assert(l0c.offset(num_mad_cd_stages) <= L0CSizeBytes, "L0C overflow");
```

注意 L1 的分配方式：`l1a` 和 `l1b` **共享同一块 512 KiB**，`l1b` 的起始地址是 `l1a.offset(kNumL1Stages)`（`l1a` 占完自己的多级 stage 之后）；而 L0A/L0B/L0C 是三块独立缓冲，各自从地址 0 开始。host 侧的启发式配置用同样的公式**预筛**候选：`csrc/jit_kernels/heuristics.hpp` 第 100～102 行直接用 `block_m * block_n * sizeof(float) > L0CSizeBytes` 否掉装不进 L0C 的 BLOCK_M/BLOCK_N 组合。

顺带认识仓库对设备原语的三层封装（`ascend.hpp` 第 6～12 行的注释写明了分层）：

```cpp
//   tile_ptr — operand_ptr / result_ptr (shape-carrying device pointers) + l*_ptr aliases
//   copy     — GM/L1/L0/UB data movement, scale-factor load, mad, quant-mode inference
//   sync     — intra-block / flag / ffts synchronization
```

今天按同样的顺序讲：地址（3.2）→ 搬运与计算（3.4、3.6）→ 同步（3.5）。

### 3.3 AIC 与 AIV：一个 block，两种执行体

看 kernel 的函数签名（`bf16_gemm.hpp` 第 27 行）：

```cpp
__global__ __mix__(1, kDirectStore ? 0 : 2) void bf16_gemm_impl(...)
```

`__mix__(cube 数, vector 数)` 声明一个**混合核**：每个 block 由 1 个 AIC（AI Cube 核）执行体和 2 个 AIV（AI Vector 核）执行体组成。仓库里的用法自成体系（grep `__mix__` 可见）：GEMM 类 kernel 是 `__mix__(1, 2)`（`fp8_gemm.hpp` 第 36 行、`fp8_dequant_gemm.hpp` 第 113 行、`mqa_logits.hpp` 第 623 行……），`kDirectStore` 时退化为 `__mix__(1, 0)`（不需要 AIV），纯调度类 kernel 用 `__mix__(0, 1)`（`sparse_mqa_logits_metadata.hpp` 第 305 行）或 `__global__ __vector__`（如 `transform_sf.hpp` 第 455 行的 SF 布局转换核，纯向量计算）。

launch 的 block 数等于使用的 AI 核数：`heuristics.hpp` 第 98 行 `c.launch_options.num_blocks = d.num_cores`，而 `num_cores` 来自 `runtime->get_num_sms()`（`csrc/apis/gemm.hpp` 第 176 行；README 的 `set_num_sms`/`get_num_sms` 接口就是它的开关）。默认等于 `kNumAICores = 32`（`csrc/apis/mega_moe.hpp` 第 159 行甚至断言了这一点），对应 32 个 AIC + 64 个 AIV——正好是 `common.hpp` 第 51～52 行的 `kNumAIVs = 2 * kNumAICores`。

**两种执行体跑同一份代码的不同分支**。kernel 内用 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏区分（`bf16_gemm.hpp` 第 81 行起）：

```cpp
if ASCEND_IS_AIC {
    // AIC（cube）：任务调度 + 全部数据搬运 + MAD
    set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>();   // 初始化 L1 stage 空闲事件
    set_flags<PIPE_M, PIPE_MTE1, kNumL0Stages>();      // 初始化 L0 stage 空闲事件
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        ...  // GM→L1→L0→MAD 的三重流水（3.6 详解）
    }
} else {
    // AIV（vector）：epilogue 写回
    if constexpr (kDirectStore) return;                // direct store 时没有 AIV
    epilogue.aiv_initialize();
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        ...  // 等待 AIC 通知 → UB 后处理 → 写回 GM
    }
}
```

分工一览（每条都在后续章节有源码细节）：

| 职责 | 执行体 | 源码位置 |
| :--- | :--- | :--- |
| 用 Scheduler 认领输出 Tile（M/N block） | 两者各自独立跑同一套调度 | `bf16_gemm.hpp` 第 72～74、89、180 行 |
| GM → L1 搬运（MTE2） | AIC | 第 117～118 行 `copy_gm_to_l1` |
| L1 → L0A/L0B 搬运（MTE1） | AIC | 第 142～145 行 `copy_l1_to_l0a/l0b` |
| `mad()` 矩阵乘加（M） | AIC（cube 单元） | 第 149 行 |
| L0C → UB 搬运（FIX） | AIC 发起 | `epilogue.hpp` 第 45 行 `copy_l0c_to_ub` |
| UB 上的后处理（bias/alpha/类型转换） | AIV（vector 单元） | `epilogue.hpp` 第 74 行 `op.process_mad_block` |
| UB → GM 写回（MTE3） | AIV | `epilogue.hpp` 第 81～82 行 `copy_ub_to_gm` |

AIC 与 AIV 之间怎么交接？两个机制：

1. **L0C → UB 的搬运自带"收件人"**。`epilogue.hpp` 第 42～47 行的 `aic_store`：

```cpp
__aicore__ void aic_store(L0C l0c_tile, uint32_t s_epi, uint32_t m_elems, uint32_t n_elems) const {
    const uint32_t aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs;   // 按 stage 轮转选出收件 AIV
    const uint32_t stage_idx = s_epi / kNumAIVs;
    copy_l0c_to_ub(ub[stage_idx], l0c_tile, m_elems, n_elems,
                   asc_dual_dst_mode::DUAL_DST_DISABLE, aiv_id);  // ← aiv_id 作为 dst_subblock_id
    set_intra_block(PIPE_FIX, aiv_id, stage_idx);                 // 通知该 AIV："你的数据好了"
}
```

`copy_l0c_to_ub` 的第 6 个参数 `dst_subblock_id` 指定写入哪个 AIV 的 UB——MAD 输出按 `s_epi % kNumAIVs` **round-robin 轮流分给两个 AIV**（`epilogue.hpp` 第 23 行注释 "Assign complete MAD tiles round-robin across available AIVs"）。

2. **AIC ↔ AIV 用"块内事件"握手**。`set_intra_block`/`wait_intra_block`（`ascend/sync.hpp` 第 29～40 行）是同一个混合 block 内 AIC 与 AIV 之间的通知：AIC 侧 `set_intra_block(PIPE_FIX, aiv_id, stage_idx)` 表示"FIX 管线已把第 stage_idx 级数据放进你（aiv_id）的 UB"；AIV 侧在 `aiv_drain` 里 `wait_intra_block(kPipe, aiv_id, stage_idx)`（`epilogue.hpp` 第 69 行）等这个通知，处理完再 `set_intra_block(PIPE_MTE3, aiv_id, stage_idx)`（第 84 行）告诉 AIC "这级 UB 空了，可以装下一块"。AIV 用 `asc_get_sub_block_id()`（第 51 行）识别自己是 0 号还是 1 号。

**`kDirectStore` 特例**：当不需要任何 epilogue 加工（输出算子是 `Identity`，见 `bf16_gemm.hpp` 第 56～57 行的 `static_assert`）时，L0C 的结果由 AIC 直接写回 GM——`copy_l0c_to_gm`（`bf16_gemm.hpp` 第 156～159 行），不需要 UB、不需要 AIV，所以核型是 `__mix__(1, 0)`；需要累加（`kWithAccumulation`）时改用 atomic add（第 37～38、82 行）。这是"epilogue 越简单、参与的执行体越少"的直接体现，第 26 天再展开。

### 3.4 流水线（Pipe）：MTE2、MTE1、M、FIX、V、MTE3

到目前为止我们把"搬运"和"计算"当成动词在用。硬件上它们各自是一条**独立的流水线（pipe）**，各有自己的执行队列：AIC 的代码只是往这些队列里**下发指令**，指令由对应引擎异步执行。`PIPE_*` 常量来自 CANN 的设备内建头 `c_api/asc_simd.h`（`ascend.hpp` 第 3 行引入），仓库不重定义它们——下表的"职责"一列是它们在本仓库中的用法归纳：

| Pipe | 职责（按仓库用法） | 仓库中的证据 |
| :--- | :--- | :--- |
| **MTE2** | GM → L1 的搬运（搬运引擎，Memory Transfer Engine） | `copy_gm_to_l1` → `asc_copy_gm2l1_nd2nz`（copy.hpp 第 101～131 行）；同步事件用 `PIPE_MTE2` |
| **MTE1** | L1 → L0A/L0B 的搬运 | `copy_l1_to_l0a/b` → `asc_copy_l12l0a/l0b`（copy.hpp 第 133～159 行） |
| **M** | cube 矩阵乘加：L0A × L0B 累加进 L0C | `mad()` → `asc_mmad`/`asc_mmad_mx`（copy.hpp 第 370～381 行） |
| **FIX** | L0C → UB 的输出搬运 | `aic_store` 中 `set_intra_block(PIPE_FIX, ...)`（epilogue.hpp 第 46 行） |
| **V** | AIV 上的向量计算（bias、alpha、类型转换等） | epilogue 算子默认 `kPipe = PIPE_MTE3`，非 MTE3 算子先在 V 上算再交 MTE3（epilogue.hpp 第 76～80 行） |
| **MTE3** | UB → GM 的搬运 | `copy_ub_to_gm` → `asc_copy_ub2gm_align`（copy.hpp 第 308～317 行） |

把它们叠到 3.1 的存储金字塔上，就得到本仓库 GEMM kernel 的通用骨架（`bf16_gemm.hpp`、`fp8_gemm.hpp`、`fp8_dequant_gemm.hpp`、`mqa_logits.hpp` 全都长这样）：

```text
GM ──MTE2──▶ L1 ──MTE1──▶ L0A/L0B ──M──▶ L0C ──FIX──▶ UB ──(V 加工)──▶ UB ──MTE3──▶ GM
     搬入          搬入           乘加          搬出          epilogue          写回
```

**流水线之间靠事件握手**。`asc_sync_notify(src, dst, event)` 表示"src 管线向 dst 管线发一个事件"，`asc_sync_wait(src, dst, event)` 表示"src 管线等 dst 管线的这个事件"。看 `bf16_gemm.hpp` AIC 主循环里 K 块的那段（第 112～121、133～152、167 行），事件天然分成两个方向：

```cpp
for (k_blk_idx...) {                                   // 每个BLOCK_K
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, s);            // ①空事件：MTE2 搬 L1[s] 前，
    copy_gm_to_l1(l1a[s], ...);                        //   先等 MTE1 把上一轮的 L1[s] 消费完
    copy_gm_to_l1(l1b[s], ...);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, s);          // ②满事件：告诉 MTE1，L1[s] 数据就绪

    for (k_mad_idx...) {                               // 每个MAD_K
        asc_sync_wait(PIPE_M, PIPE_MTE1, s_ab);        // ③空事件：MTE1 搬 L0[s_ab] 前等 M 释放
        copy_l1_to_l0a(l0a_tile, l1a[s], ...);         //   （L0A/L0B 的多级 stage）
        copy_l1_to_l0b(l0b_tile, l1b[s], ...);
        asc_sync_notify(PIPE_MTE1, PIPE_M, s_ab);      // ④满事件：L0[s_ab] 就绪

        asc_sync_wait(PIPE_MTE1, PIPE_M, s_ab);        // ⑤空事件：M 执行前等 MTE1 搬完
        mad(l0c_tile, l0a_tile, l0b_tile, ...,
            is_last ? ENABLE_UPDATE : ENABLE_KEEP, is_first);
        asc_sync_notify(PIPE_M, PIPE_MTE1, s_ab);      // ⑥满事件：M 接受指令后释放 L0[s_ab]
    }
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, s);          // ⑦空事件：MTE1 消费完 L1[s]，释放给 MTE2
}
```

读法：**奇数号是"等空"（消费者释放了缓冲才能写），偶数号是"报满"（生产者写完了才能被消费）**。这是一套标准的环形缓冲生产者-消费者协议；`s`/`s_ab` 就是缓冲下标。初始状态由第 83～84 行的 `set_flags` 预置：kernel 一启动就把所有 L1/L0 stage 的"空事件"全部发出，让 MTE2 可以立刻开搬。

事件数量是有限资源：`common.hpp` 第 61 行 `kNumPipeEvents = 8`。所以 stage 下标都是模轮转（`s = s == kNumL1Stages - 1 ? 0 : s + 1`，第 112 行），事件 `(src, dst, id)` 随 stage 复用。host 侧配置搜索甚至要给事件记账：`heuristics.hpp` 第 167～169 行 "Share L1 capacity and pipe events between operands and scale factors"——有 SF 的 kernel，L1 stage 数上限要减去 SF 占用的事件数。

### 3.5 多级 stage：用乒乓缓冲掩盖搬运延迟

如果 L1 只有一块缓冲（单缓冲），三条流水线只能串行：

```text
单缓冲：
MTE2   [K0→L1]          [等]   [K1→L1]          [等]  ...
MTE1            [K0→L0] [等]            [K1→L0] ...
M                       [K0 mad][闲]             [K1 mad] ...
       ↑ 每一轮：MTE2 搬完才能 MTE1 搬，MTE1 搬完才能 mad，全程三级互相干等
```

多级 stage 把"正在被消费的块"和"正在预取的块"错开。以 L1 双缓冲（`kNumL1Stages = 2`）为例：

```text
双缓冲：
MTE2   [K0→L1[0]] [K1→L1[1]] [K2→L1[0]] [K3→L1[1]] ...   ← 引擎几乎不停
MTE1   [  等  ] [K0:L1[0]→L0] [K1:L1[1]→L0] [K2:...] ...  ← 只在开头等一次
M      [  等  ] [    等    ] [K0: mad] [K1: mad] ...      ← 稳态后与搬运重叠
```

（示意图做了简化：MTE1/M 实际按 `MAD_K` 粒度交错，见第 22 天；今天只需要"稳态下三线并行"的直觉。）

这套机制在源码里的形态是三组并行的 stage 轮转下标，各自有独立的级数模板参数：

| 缓冲 | 下标 | 级数参数 | 物理含义 |
| :--- | :--- | :--- | :--- |
| L1 的 A/B | `s` | `kNumL1Stages` | GM→L1 的预取深度 |
| L0A/L0B | `s_ab` | `kNumL0Stages` | L1→L0 的预取深度 |
| L0C | `s_cd`（隐含在 `l0c[s_cd]`） | `num_mad_cd_stages = (BLOCK_M/MAD_M) × (BLOCK_N/MAD_N)` | 一个 BLOCK 输出切成几个 MAD tile |
| UB | `s_epi` | `kNumEpilogueStages × 2` | AIC 已写出、AIV 尚未写回的 MAD tile 数 |

各级数都不是随便选的，host 侧 `heuristics.hpp` 用容量公式逐个算出来：

- **L0 stage 恒为 2**（第 111～112 行注释 `// Select L0 stages: always 2 stage`）：L0A/L0B 总共各 64 KiB，塞不下更深的双份以上缓冲；第 117～118 行的注释说明了设计意图——"at least 2 stage in L0C, to overlap mad and store / fit l0a and l0b, to overlap load and mad"。
- **MAD tile 至少能在 L0C 里放两份**：第 127 行 `mad_m * mad_n * sizeof(float) > L0CSizeBytes / 2` 直接否掉——MAD 和 FIX 搬出要能重叠。
- **`MAD_K` 由 L0A/L0B 容量反推**：第 133～135 行 `l0a_k = L0ASizeBytes * 8 / (num_l0_stages * mad_m * elem_bits)`（L0B 同理），取两者与对齐约束的最小值。
- **L1 stage 数用剩余容量除以单块大小**：第 170～174 行 `num_l1_stages = min(事件数上限, (L1SizeBytes - sf_bytes) / ab_bytes)`，且最少 2 级。
- **epilogue stage 数受 UB 容量与块内事件数双限制**：第 190 行 `min(hw.num_intra_block_events, UBSizeBytes / epilogue_bytes)`，还要保证 `num_epilogue_stages × num_aivs ≥ num_cd_stages`（第 193～194 行）——每个 MAD tile 都得有地方承接。

同一个逻辑在 kernel 侧就是 3.2 节那组 `static_assert`。今天先用一组假设配置手工算一遍（实践步骤 1），第 16 天再正式做"片上内存规划"专题。

### 3.6 完整数据流：一个输出 Tile 的一生

现在把所有零件装起来。设一个具体的小场景：`M=128, N=256, K=128`（BF16，Normal GEMM），假设配置 `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, MAD_M=128, MAD_N=64, MAD_K=64, kNumL1Stages=3, kNumL0Stages=2`。跟随输出 Tile `(m_block=0, n_block=0)` 走一遍 `bf16_gemm_impl`：

1. **认领任务**：AIC 和 AIV 各自构造 `Scheduler`（`bf16_gemm.hpp` 第 72～74 行，用 `block_idx` 区分 32 个核），`get_next_block()` 发放属于本核的 `(m_block_idx, n_block_idx)`（第 89 行；持久调度细节第 17 天讲）。两边拿到同一套 tile 序列，才能各自知道"我在跟谁配合"。
2. **算 K 循环规模**：`num_k_blocks = ceil(128 / 64) = 2`（第 96 行）。
3. **第 1 个 K 块（`s = 0`）**：
   - MTE2 把 `A[0:128, 0:64]` 搬进 `l1a[0]`（128×64×2 B = 16 KiB）、`B[0:256, 0:64]` 搬进 `l1b[0]`（32 KiB）（第 117～118 行）；
   - MTE1 把 `l1a[0]`/`l1b[0]` 分块搬进 `l0a[s_ab]`/`l0b[s_ab]`（第 142～145 行，MAD_K 粒度轮转）；
   - M 执行 `mad(l0c_tile, l0a_tile, l0b_tile, ..., ENABLE_KEEP, is_first=true)`（第 149～151 行）——`is_first=true` 对应 `clear_C`，首个 K 块先清零 L0C；此后 `ENABLE_KEEP` 表示继续累加进同一块 L0C。
4. **第 2 个 K 块（`s = 1`）**：同上，MTE2 已经在 MTE1/M 消费 `l1a[0]` 时并行预取；MAD 以 `ENABLE_KEEP` 继续累加。K 走完时，`l0c[s_cd]` 里是该 tile 的 FP32 完整累加和。
5. **输出**（第 155～163 行）：本例 BLOCK_N=256 > MAD_N=64，输出被切成 `num_mad_cd_stages = (128/128) × (256/64) = 4` 个 MAD tile，每个 tile 走一次：
   - AIC：`epilogue.aic_store(l0c_tile, s_epi, ...)` → FIX 管线把 `l0c_tile` 搬进轮到的 AIV 的 `ub[stage_idx]`，并发块内事件（3.3 节）；
   - AIV：`aiv_drain` 等到事件 → `op.process_mad_block` 做 epilogue 加工（`Identity` 则什么都不做）→ `copy_ub_to_gm` 把 `D[0:128, 0:64]`（此 tile 对应的 128×64 分片）写回 GM → 发事件释放该 UB stage。
6. **循环**：AIC 继续认领下一个 tile，MTE2/MTE1/M 早已在为它工作——这就是 3.5 节的稳态。

把整个过程画成一张图（**这张图就是当天产出**，第 5～6 节给出参考版和自查清单）：

```text
                    ┌────────────── 每个 AI Core（block_idx ∈ [0, 32)）──────────────┐
                    │                                                              │
   GM (HBM)         │   AIC（cube 执行体）                     AIV ×2（vector 执行体）│
 ┌──────────┐       │  ┌─────────────────────────┐            ┌────────────────────┐ │
 │ A[M,K]   │       │  │ Scheduler 认领 (m,n) tile │            │ 同一套 Scheduler     │ │
 │ B[K,N]   │ MTE2  │  └───────┬─────────────────┘            │ （各自独立遍历）      │ │
 │ D[M,N]   ├───────┼──────────▼──────────────                └─────────┬──────────┘ │
 └──────────┘       │   L1 (512 KiB)  l1a[s], l1b[s]                     │            │
        ▲           │        │ MTE1                                   │            │
        │           │        ▼                                        │            │
        │           │   L0A (64 KiB) l0a[s_ab]   L0B (64 KiB) l0b[s_ab] │            │
        │           │        └────────── M：mad ──────────┐             │            │
        │           │                       (asc_mmad,    │             │            │
        │           │                        FP32 累加)   ▼             │            │
        │           │                       L0C (256 KiB) l0c[s_cd]     │            │
        │           │                            │ FIX（dst_subblock_id 指定收件 AIV） │
        │           │                            ▼             ▼         │            │
        │           │                       UB (256 KiB) ub[stage] ──▶ AIV 的 UB       │
        │           │                                    V：epilogue 加工             │
        │           └────────────────────────────────────MTE3：写回 D ◀──┘            │
        └──────────────────────────────── D[ m_idx:m_idx+128, n_idx:n_idx+256 ] ──────┘
```

一个容易忽略的细节：**搬运不只是复制，还负责布局转换**。`copy_gm_to_l1` 底层是 `asc_copy_gm2l1_nd2nz`（copy.hpp 第 122 行）——ND（普通行/列主序）到 NZ（分形布局，`FRAC_MN = 16`，common.hpp 第 63 行注释 "NZ fractal M/N rows"）的转换在搬运中完成；反过来 L0C 里是 NZ 布局的累加结果，写回 GM 前用 `asc_set_l0c2gm_nz2nd`（copy.hpp 第 9～12 行的 `set_l0c_nz2dn`）转回 ND。今天记住"GM 是 ND、L0 是 NZ、转换藏在搬运指令里"即可，布局细节第 18～19 天专题展开。

### 3.7 与前四天概念的对应

| 前几天的软件概念 | 今天的硬件落点 |
| :--- | :--- |
| 第 1 天：数据量 `2MNK` 与"放不下" | 3.1 的容量账：A+B 是单核 L1 的 704 倍 |
| 第 2 天：stride / major | `operand_ptr` 携带 major；MN-major 的 A 搬 L0 要走 `asc_copy_l12l0a_transpose`（copy.hpp 第 142 行） |
| 第 3 天：`BLOCK_M/N/K`、K 维循环 | L1 容量决定 BLOCK_K 与 `kNumL1Stages`；`num_k_blocks` 循环就是第 112 行的 for |
| 第 3 天：数据复用减少 GM 访问 | 多级 stage + L2 cache hint（`l2_ctrl`）都是复用手段；`kResidentA/B` 让 A/B 常驻 L1（`bf16_gemm.hpp` 第 109～110 行） |
| 第 4 天：FP32 累加器 | L0C，`l0c_ptr<float>`（第 64 行），`ENABLE_KEEP`/`is_first` 控制累加语义 |

## 四、源码导读

今天的导读主题是"**硬件抽象层的完整拼图**"，全程静态阅读、无需硬件。建议对照仓库依次打开以下文件（路径相对 `DeepGEMM-Ascend/`）。

### 站点 1：硬件常量与 GM 指针 —— `deep_gemm/include/deep_gemm/common.hpp`

- 第 20～26 行的注释解释了 `__aicore__`/`__gm__` 关键字与设备编译器（Bisheng 定义 `__CCE__`）的关系——理解"这些修饰符是编译器强制的地址空间"。
- 第 50～63 行：全部容量/拓扑常量（3.1 节已列）。注意注释 `// Ascend 950DT core topology`、`// L0C (fp32 accumulator)`、`// NZ fractal M/N rows`——三句话分别锚定今天、昨天、明天的主题。
- 第 137～186 行：`gm_ptr` ——唯一能被 host 构造并作为 launch 参数传入的指针类型（tile_ptr.hpp 第 99～100 行注释点明了这一点）；`index()/group()/batch()` 的地址算术今天浏览即可。

### 站点 2：地址空间与指针别名 —— `deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp`

- 第 8～16 行：`cvt_pointee` 对六种修饰符的特化——一张表看清"哪种指针能转换到哪种地址空间"。
- 第 18～56 行：`operand_ptr`。重点看三个成员：`size_per_stage()`（第 37 行，stage 字节数）、`offset(n)`（第 38 行）、`operator[](i)`（第 39～43 行，"前进 i 个 stage"）——多级缓冲的地址计算全部靠它们。
- 第 58～91 行：`result_ptr`，同样结构，服务 L0C/UB。
- 第 93～97 行：五个别名一行一个，背下来。

### 站点 3：搬运与计算原语 —— `deep_gemm/include/deep_gemm/ascend/copy.hpp`

按数据流顺序找齐六类原语（都是 `asc_*` 内建指令的薄封装）：

| 原语 | 行号 | 对应 Pipe | 说明 |
| :--- | :--- | :--- | :--- |
| `copy_gm_to_l1` | 101 | MTE2 | 含 `nd2nz` 分形转换与 NZ 参数设置（`set_mte2_nz`，第 14 行） |
| `copy_l1_to_l0a/l0b` | 133 / 147 | MTE1 | MN-major 时走 `_transpose` 变体（第 142、156 行） |
| `mad` | 370 | M | `asc_mmad`（BF16）/`asc_mmad_mx`（8bit 操作数，融合 UE8M0 缩放，第 4 天已见） |
| `copy_l0c_to_ub` | 266 | FIX | 自带 `quant_mode`（如 `F322BF16`，第 232 行 `infer_quant_mode`） |
| `copy_gm_to_ub` | 39 | MTE2 | AIV 直接从 GM 取小数据（如 bias）时用 |
| `copy_ub_to_gm` | 308 | MTE3 | 写回 D |
| `copy_l0c_to_gm` | 287 | — | `kDirectStore` 专用直通路径，含 `nz2nd`（`set_l0c_nz2dn`，第 9 行） |

另外注意 `fill_l1`（第 319 行）：边界 K 块不足时用它把 L1 尾部填零（`fill_tail_block`，第 333 行），这是"非对齐 shape 靠搬运侧补零"的证据，第 19 天细读。

### 站点 4：流水线同步 —— `deep_gemm/include/deep_gemm/ascend/sync.hpp`

- 第 15～27 行：`notify`/`wait` 对 `asc_sync_notify`/`asc_sync_wait` 的封装——注意参数顺序 `(kSrc, kDst, event)`：**src 向 dst 发 / src 等 dst 发**，方向读反会导致后面全看不懂。
- 第 29～40 行：`set_intra_block`/`wait_intra_block`——AIC↔AIV 的块内事件；第 35 行注释 "AIV ignores subblock_id for intra-block waiting" 提醒两边等待语义的细微差别。
- 第 63～73 行：`set_flags`/`wait_flags`——批量发事件的初始化辅助（`bf16_gemm.hpp` 第 83～84 行用它预置全部空事件）。

### 站点 5：AIC 主循环 —— `deep_gemm/include/deep_gemm/bf16_gemm.hpp`

- 第 27 行：`__mix__(1, kDirectStore ? 0 : 2)` 核型声明。
- 第 51～64 行：`num_mad_cd_stages` 的定义与五组指针声明——L1 共享、L0 三块独立。
- 第 66～69 行：四条容量 `static_assert`——把 3.2 节的账固化进编译期。
- 第 81～84 行：`if ASCEND_IS_AIC` 分支入口与两行 `set_flags` 初始化。
- 第 112～121 行：MTE2 的空/满事件对（`copy_gm_to_l1` 夹在中间）。
- 第 133～152 行：MTE1 与 M 的两对事件（`copy_l1_to_l0a/b` 与 `mad` 各夹一对）——对照 3.4 节的七步注释逐行核对。
- 第 155～163 行：K 完成后的分流：`kDirectStore` → `copy_l0c_to_gm`；否则 `epilogue.aic_store`。
- 第 173～192 行：AIV 分支——注意它也跑 `scheduler.get_next_block`（第 180 行），与 AIC 的 tile 序列保持一致。

### 站点 6：AIC/AIV 协作 —— `deep_gemm/include/deep_gemm/epilogue.hpp`

- 第 23～26 行：`kNumStages = kNumStagesPerAIV * kNumAIVs` 与 UB 布局——round-robin 的舞台。
- 第 37～39 行：`aic_wait_free`——AIC 写 UB 前要等 AIV 释放（`wait_intra_block(PIPE_FIX, ...)`），与 `aiv_drain` 末尾的 `set_intra_block(PIPE_MTE3, ...)`（第 84 行）配对。
- 第 41～47 行：`aic_store`——`dst_subblock_id = aiv_id` 指定收件 AIV（3.3 节）。
- 第 49～87 行：`aiv_drain`——第 57 行 `if (aiv_id != ...) continue` 表明每个 AIV 只处理分给自己的 MAD tile；第 76～80 行是 V→MTE3 的交接（`Identity` 的 `kPipe` 恰为 `PIPE_MTE3`，见 `epilogue/operators.hpp` 第 100 行，所以跳过）。
- 第 89～97 行：`aiv_initialize`——累加模式下设置 atomic add 指令。

**阅读检查点**（能回答再进入实践）：

1. `l1a` 和 `l1b` 为什么必须共享 512 KiB？它们的地址是怎么排的？
2. `__mix__(1, 2)` 的两个数字分别控制什么？`kDirectStore` 时各是多少？
3. `asc_sync_wait(PIPE_MTE1, PIPE_MTE2, s)` 这一行里，谁在等谁？等的是什么性质的事件？
4. `mad()` 的第 7、8 个参数（`unit_flag`、`clear_C`）在 K 循环里怎么随 `is_first`/`is_last` 变化？
5. AIC 怎么知道"某个 UB stage 可以再写了"？AIV 怎么知道"数据到了"？
6. `heuristics.hpp` 里 L1 stage 数、L0 stage 数、epilogue stage 数分别由什么上界决定？

## 五、实践步骤

以下步骤按顺序执行，**全部不需要 Ascend 硬件**；总时长约 2 小时。

### 步骤 1：手工演算片上内存占用（约 25 分钟）

不看 3.5 节的结论，用 `static_assert` 的公式独立完成。假设配置（本例为教学虚构，真实配置由 `select_gemm_config()` 搜索，第 12 天研究）：

```text
BLOCK_M=128, BLOCK_N=256, BLOCK_K=64（BF16，2 字节）
MAD_M=128, MAD_N=64, MAD_K=64
kNumL1Stages=3, kNumL0Stages=2, kNumEpilogueStages=2（每 AIV），kNumAIVs=2
```

按 `size_per_stage = shape_mn × shape_k × sizeof(T)` 计算：

1. `l1a` 单级、`l1b` 单级、L1 总占用 `l1b.offset(3)`；与 `L1SizeBytes` 比较（答案：16 KiB、32 KiB、144 KiB ≤ 512 KiB）；
2. `l0a`/`l0b` 单级与总占用（答案：16 KiB、8 KiB；两级各 32 KiB、16 KiB ≤ 64 KiB）；
3. `num_mad_cd_stages` 与 `l0c` 总占用（答案：`(128/128)×(256/64)=4`；单级 `128×64×4 B = 32 KiB`，共 128 KiB ≤ 256 KiB）；
4. epilogue 的 UB 占用：`kNumStages × MAD_M × MAD_N × sizeof(BF16)`（答案：`2×2×128×64×2 = 64 KiB ≤ 256 KiB`）；
5. 变式一：把 `BLOCK_N` 提到 512，哪些数字变化？L1 还装得下吗？L0C 呢？（答案：L1 = 48+192 = 240 KiB 仍可行；`num_mad_cd_stages = 8`，L0C 恰好 256 KiB 触顶。）
6. 变式二：再把 `BLOCK_N` 提到 1024 会怎样？（答案：L0C 需要 512 KiB > 256 KiB，编译期直接失败——这解释了 `heuristics.hpp` 第 100～102 行为何先用 L0C 容量过滤 BLOCK_M/BLOCK_N。）

### 步骤 2：画双缓冲时序图（约 15 分钟）

纸笔画两张图（对照 3.5 节修正）：

1. `kNumL1Stages=1`（单缓冲）时 MTE2/MTE1/M 三条线的时序，标出每条线的"等"区间；
2. `kNumL1Stages=2` 时的时序，标出稳态下三条线重叠的部分；
3. 在图上标出 3.4 节编号 ①～⑦ 的七个事件各发生在哪个时刻。

### 步骤 3：源码导读打卡（约 45 分钟）

按第四节顺序依次打开 6 个站点文件，找到列出的行号并通读相应片段。重点验证五处"硬件概念"的源码实体：

- `common.hpp` 第 54～58 行：五条容量常量；
- `tile_ptr.hpp` 第 93～97 行：五个地址空间别名；
- `bf16_gemm.hpp` 第 27 行的 `__mix__`、第 81 行的 `ASCEND_IS_AIC`、第 83～84 行的 `set_flags`；
- `bf16_gemm.hpp` 第 116～152 行：七步事件序列与被它们夹住的 `copy_gm_to_l1`/`copy_l1_to_l0a/b`/`mad`；
- `epilogue.hpp` 第 42～47 行与第 69～84 行：`dst_subblock_id` 指定收件 AIV、两个方向的 intra-block 事件。

无硬件时的替代任务就是本步骤本身——纯静态阅读。

（可选，有 Ascend 环境时）在仓库根目录执行：

```bash
./develop.sh
DG_PRINT_CONFIGS=1 python tests/test_bf16_gemm.py normal --skip-prof
```

`DG_PRINT_CONFIGS=1`（README "Environment Variables" 一节）会为每个 shape 打印所选配置，其中 `l1_stages`、`l0_stages`、`epi_stages`、`block_m/n/k` 就是今天演算的这些量——拿打印值回代步骤 1 的公式验证。

## 六、当天产出

学习计划要求今天的产出是**一张 Ascend GEMM 数据流图**。建议先自己画（不看 3.6 节），再对照下面的参考版修订。自查要素：

1. 是否画全了六层存储（GM、L1、L0A、L0B、L0C、UB）并标注容量？
2. 是否标出了六条 pipe（MTE2、MTE1、M、FIX、V、MTE3）各负责哪一段？
3. 是否区分了 AIC 与 AIV 两条执行体，标明谁发起哪段搬运、谁做计算？
4. 是否体现了多级 stage（`s`、`s_ab`、`s_cd`、`s_epi` 的轮转）？
5. 是否标出了 FP32 累加发生在 L0C、类型转换发生在 FIX（`copy_l0c_to_ub` 的 `quant_mode`）或 V（epilogue 算子）？
6. 是否画出了 `kDirectStore` 的旁路（L0C → GM 直达）？

参考版（完整版见 3.6 节，此处为最小骨架）：

```text
GM ──MTE2──▶ L1[l1a[s],l1b[s]] ──MTE1──▶ L0A[l0a[s_ab]] ──M──▶ L0C[l0c[s_cd]]
                                        └▶ L0B[l0b[s_ab]] ──┘     (FP32累加)
                                                                  │FIX
                                                                  ▼
                                              UB[ub[stage]] ──V──▶ (epilogue加工)
                                                                  │MTE3
GM ◀────────────────────────────── D ◀────────────────────────────┘
（执行体：MTE2/MTE1/M/FIX 由 AIC 发起；V/MTE3 由 AIV 执行）
```

配套自查清单：

- 图上每一层存储的容量常量是否写对了（512/64/64/256/256 KiB）？
- 每条箭头能否对应到 `copy.hpp` 里的一个具体函数？
- 能否指着图讲出"七步事件"中的任意一步在防什么竞争？

## 七、自测题

先独立作答，再对照本节末尾的参考答案。

1. 按数据流顺序说出六层存储的名称与容量，并各配一个仓库类型别名。
2. `__cbuf__`、`__ca__`、`__cb__`、`__cc__`、`__ubuf__` 分别修饰哪层存储？`l0a_ptr<T>` 的完整模板展开是什么？
3. `__mix__(1, 2)` 的两个参数是什么含义？`bf16_gemm_impl` 在 `kDirectStore` 下用几个 AIV？为什么此时不需要 AIV？
4. 在 `bf16_gemm_impl` 中，AIC 分支和 AIV 分支各自执行哪些工作？两边的 `scheduler.get_next_block()` 为什么要跑同一套序列？
5. GM→L1、L1→L0A/L0B、矩阵乘加、L0C→UB、UB→GM 各由哪条 pipe 负责？`copy.hpp` 中对应的封装函数分别叫什么？
6. `asc_sync_wait(PIPE_M, PIPE_MTE1, s_ab)` 是谁等谁？等的是"满事件"还是"空事件"？kernel 启动时这些事件如何被预置？
7. 手工演算：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, MAD_M=128, MAD_N=128, MAD_K=16`，BF16，`kNumL1Stages=4, kNumL0Stages=2`。L1、L0A、L0B、L0C 的总占用各是多少？四条 `static_assert` 都能过吗？
8. （思考）为什么 `heuristics.hpp` 强制 `num_l0_stages = 2`，而 L1 的 stage 数却按容量动态计算（最多到事件数上限）？

**参考答案：**

1. GM（HBM，GiB 级，`gm_ptr`）→ L1（512 KiB/核，`l1_ptr`）→ L0A（64 KiB，`l0a_ptr`）与 L0B（64 KiB，`l0b_ptr`）→ L0C（256 KiB，FP32 累加器，`l0c_ptr`）→ UB（256 KiB，`ub_ptr`）。容量见 `common.hpp` 第 54～58 行。
2. L1、L0A、L0B、L0C、UB。`l0a_ptr<T> = operand_ptr<__ca__ T*, T>`（tile_ptr.hpp 第 94 行）。修饰符由设备编译器 Bisheng 识别（`common.hpp` 第 20～26 行注释）。
3. 混合核的（cube 执行体数，vector 执行体数）。`kDirectStore` 时 `__mix__(1, 0)`：输出算子必须是 `Identity`（`bf16_gemm.hpp` 第 56～57 行断言），L0C 经 `copy_l0c_to_gm` 直接写回 GM，无需 UB 加工，AIV 分支直接 `return`（第 174 行）。
4. AIC：调度、MTE2（`copy_gm_to_l1`）、MTE1（`copy_l1_to_l0a/b`）、M（`mad`）、FIX（`epilogue.aic_store` 发起 L0C→UB）；AIV：`aiv_drain` 等 UB 就绪、V 上做 epilogue 加工、MTE3 写回 GM。两边必须对同一 tile 序列达成一致，AIC 才知道往哪个 AIV 的 UB 送数据（round-robin 由 `s_epi` 决定），AIV 才知道该等哪些 stage（`epilogue.hpp` 第 57 行的 `aiv_id` 过滤）。
5. MTE2（`copy_gm_to_l1`）、MTE1（`copy_l1_to_l0a`/`copy_l1_to_l0b`）、M（`mad` → `asc_mmad`）、FIX（`copy_l0c_to_ub`）、MTE3（`copy_ub_to_gm`）。另有 GM↔UB 直达的 `copy_gm_to_ub`/`copy_ub_to_gm` 也走 MTE2/MTE3。
6. MTE1 等 M 发的事件——MAD 消费完 `l0a[s_ab]/l0b[s_ab]` 后发出的"空事件"（释放 L0 stage）。注意方向参数是 `(PIPE_M, PIPE_MTE1, ...)`：第一个参数是等待方视角。预置靠 `set_flags<PIPE_M, PIPE_MTE1, kNumL0Stages>()`（`bf16_gemm.hpp` 第 84 行），把所有 L0 stage 一次性标成空闲，让 MTE1 立刻可搬。
7. `l1a` 单级 `128×128×2 = 32 KiB`、`l1b` 同为 32 KiB，L1 总占用 `l1b.offset(4) = 256 KiB ≤ 512 KiB` ✓；`l0a` 单级 `128×16×2 = 4 KiB`、两级 8 KiB ≤ 64 KiB ✓，`l0b` 同 ✓；`num_mad_cd_stages = 1`，`l0c = 128×128×4 = 64 KiB ≤ 256 KiB` ✓。四条断言全过。
8. L0A/L0B 各只有 64 KiB 且直接喂 MAD 单元，两份（load 与 mad 重叠）已是容量与收益的平衡点，更多级只会挤掉操作数空间；而 L1 有 512 KiB，stage 数上限实际由两个更松的约束决定——事件数（`kNumPipeEvents = 8`，见 `heuristics.hpp` 第 167～168 行的注释 "Share L1 capacity and pipe events"）和 `(L1SizeBytes - sf_bytes) / ab_bytes` 的整除结果（第 170～174 行），所以值得按 shape 动态搜索。

## 附录 A：本日概念速查

| 概念 | 一句话定义 | 仓库落点 |
| :--- | :--- | :--- |
| GM | HBM 主存，存完整 A/B/D | `__gm__`、`gm_ptr`（common.hpp:137） |
| L1 | 每核 512 KiB 的 GM→L0 中转，A/B 多级 stage 驻留 | `__cbuf__`、`l1_ptr`（tile_ptr.hpp:93） |
| L0A / L0B | 各 64 KiB，MAD 的 A/B 操作数专用缓冲 | `__ca__`/`__cb__`、`l0a_ptr`/`l0b_ptr`（tile_ptr.hpp:94-95） |
| L0C | 256 KiB，MAD 输出与 FP32 累加器 | `__cc__`、`l0c_ptr`（tile_ptr.hpp:96） |
| UB | 256 KiB，AIV 的向量计算/epilogue 中转缓冲 | `__ubuf__`、`ub_ptr`（tile_ptr.hpp:97） |
| AIC / AIV | cube（搬运发起 + MAD）与 vector（epilogue 写回）执行体 | `__mix__(1, 2)`、`ASCEND_IS_AIC`（bf16_gemm.hpp:27,81） |
| 混合核 | 一个 block 内 1 AIC + 2 AIV 的核型 | `__mix__`；纯向量核 `__global__ __vector__`（transform_sf.hpp:455） |
| MTE2 / MTE1 / MTE3 | GM→L1 / L1→L0 / UB→GM 搬运流水线 | `copy_gm_to_l1` / `copy_l1_to_l0a` / `copy_ub_to_gm` |
| M / FIX / V | 乘加 / L0C→UB 输出搬运 / 向量计算流水线 | `mad`（copy.hpp:370）/ `aic_store`（epilogue.hpp:41）/ `process_mad_block` |
| 事件握手 | `asc_sync_notify/wait(src, dst, e)` 的满/空事件 | `ascend/sync.hpp:15-27`；七步序列 bf16_gemm.hpp:116-167 |
| 多级 stage | 同一缓冲按 `offset()` 切成环形轮转的乒乓份 | `l1a[s]`、`l0a[s_ab]`、`l0c[s_cd]`、`ub[s_epi]` |
| kNumPipeEvents | 每对 pipe 可用事件数（8），限制 stage 深度 | common.hpp:61、heuristics.hpp:167-169 |
| NZ 分形 | L0 要求的 16 行分形布局，搬运中做 ND↔NZ 转换 | `FRAC_MN=16`、`asc_copy_gm2l1_nd2nz`、`set_l0c_nz2dn` |
| dst_subblock_id | L0C→UB 搬运指定写入哪个 AIV 的 UB | `aic_store`（epilogue.hpp:45） |

## 附录 B：遗留问题记录模板

```text
# 第 5 天遗留问题
1. （例如）asc_mmad 内部的乘加宽度、分形粒度如何决定 MAD_M/N/K 的合法取值 —— 计划第 24 天 MAD 专题
2. （例如）NZ 分形布局的物理排布细节与 FRAC_K=32/sizeof(T) 的由来 —— 计划第 18～19 天地址与搬运专题
3. （例如）kResidentA/B 常驻模式下 stage 与事件如何退化 —— 计划第 26 天特殊路径专题
4. ……
```
