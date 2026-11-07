# 第 7 周 · 第 3 天（总第 45 天）：理解 MegaMoE 通信与融合

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 7 周"MegaMoE、HC Prenorm 与扩展算子"的第 3 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 65 分钟，源码导读约 30 分钟，实践步骤约 50 分钟，自测题约 15 分钟。

昨天（第 44 天）我们走完了 `mega_moe_impl` 的计算主线，但把两件事刻意留在了黑盒里：一是"经 SymBuffer 读写远端"到底是怎么发生的，二是 `kOverlapReduce` 背后那套"边算边归约"的时间线。今天打开这两个黑盒，只读两个新文件——Host 侧的 `csrc/backends/hccl.hpp`（211 行，建立跨 rank 直访底座）和 Device 侧的 `deep_gemm/include/deep_gemm/comm/mega_moe.hpp`（277 行，同步与元数据协议），再把 `mega_moe.hpp` 里的通信站点串成完整时间线。今天的四个核心问题：**通信底座是什么（UBMEM 直访）；EP Dispatch 与 Combine 各自通信了什么、谁发给谁；计算和通信为什么、怎样融合；不同 token 数下瓶颈在计算还是通信**。当天产出是**MegaMoE 计算/通信时序图**（参考版本见 3.8 节）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 讲清 `HCCLSymmetricBuffer`（`csrc/backends/hccl.hpp:29-209`）的建立流程：`aclrtMalloc` → `HcclCommMemReg`（带全局唯一 tag）→ 取网络 layer 1 → 对每个 peer 用 `COMM_PROTOCOL_UB_MEM` 链路建 `COMM_ENGINE_AIV` 通道 → `HcclChannelGetRemoteMems` 把每个远端的虚拟地址填进 `peer_ptrs`，并解释为什么注册段进程生命周期持有、池化复用（`:166-171` 注释）；
2. 用六类通信对象表描述 EP Dispatch 与 Combine 的全部跨 rank 数据流（方向、内容、数量级、代码位置），并区分哪些是元数据写、哪些是数据拉取、哪些是数据推回；
3. 解释 `SymBuffer::map`（`layout/sym_buffer.hpp:36-46`）的"一层地址代数"：远端访问 = 本地指针加 `(远端基址 − 本地基址)`，Device Kernel 里远端读写与普通 GM 访问是同一条 MTE2/MTE3 通路；
4. 分析 Device 侧三个同步原语的分工与实现：`wait_until`（`:12-23`，60 秒超时兜底）、`grid_sync`（`:35-63`，SIMT worker 相位计数）、`rank_barrier`（`:65-98`，epoch 双槽增减交替，永不清零），并说明它们各保护什么不变量；
5. 追踪 Dispatch 元数据协议 `build_dispatch_metadata`（`comm/mega_moe.hpp:117-240`）的四拍流程与三次 `grid_sync`，以及计数如何以 `kCountArrivalBit | count` 打包发布、`wait_dispatch_metadata`（`:242-275`）如何凑证书；
6. 追踪 Combine 的两段协议：`advance_combine_push`（`mega_moe.hpp:999-1053`）如何按 `src_metadata` 把 BF16 行推回源 rank，`publish_combine_prefix`（`:1062-1077`）与 `advance_combine_reduce` 的 peer 检查（`:175-197`）如何让源 rank 在生产完成前缀内提前归约（early reduce），收尾为什么必须等 peer 的 `~0`（`:286-294`）；
7. 用 baseline 的七段 kernel 列表（`tests/test_mega_moe.py:207-209`）与融合 Kernel 的块级进度、环形接收池、`prefer_push` 轮转、`kOverlapReduce` 窗口，论证计算与通信为什么需要融合、融合后重叠发生在哪几处；
8. 用 benchmark 的口径公式（`test_mega_moe.py:221-231`）加均匀路由假设，手算 EP8 各配置的通信字节与带宽并复现 README 数据表（`README.md:164-173`），据此初步判断小 token 数（时延/同步受限）与大 token 数（带宽/计算受限）的不同瓶颈形态。

## 二、预备知识

- **第 5、23 天**：Ascend 存储层级与多级流水——今天远端访问就发生在 AIV 的 MTE2（读）/MTE3（写）管道上，`DSB_DDR` 内存屏障与第 23 天的同步语义一致。
- **第 25 天**：AIC/AIV 分工与 `__mix__(1, 2)`——今天的 64 个 AIV 里，AIV0 负责元数据与权重，AIV1 负责全部 pull/push 通信。
- **第 33 天**：SF 布局——`MX_SF_DIVISOR = 64`（每 64 个 K 元素一对 UE8M0 指数打包进 int16），决定 pull 一个 token 要额外搬运 `H/64` 个 int16。
- **第 36 天**：M-Grouped GEMM——expert 分段信息（`expert_recv_count_sum`）今天由 Kernel 内的 Dispatch 元数据现场生成。
- **第 43 天**：MegaMoE 对外接口——`SymmBuffer` 的 Python 包装（`deep_gemm/mega/__init__.py:20-60`）、`destroy()` 不释放 HBM、13 个视图、`Workspace` 容量公式。今天沿用贯穿配置 `R=2, E=8, S=0, T=128, k=2, H=I=512`。
- **第 44 天**：MegaMoE 计算主线——五段流水、三类角色、`ProgressStage` 四阶段进度表、环形接收池的三条复用等待。今天把这些机制从"本 rank 视角"扩展到"跨 rank 视角"。
- **分布式基础**：知道 rank、EP（Expert Parallelism）、进程组即可；不需要 HCCL 使用经验——今天正好从源码学。

硬件常量（`deep_gemm/include/deep_gemm/common.hpp:51-65`）：`kNumAICores = 32`、`kNumAIVs = 64`；`kL2SectorBytes = 128`。源码注释中出现的时钟频率（Scalar 1 GHz / Vector 1.65 GHz，`comm/mega_moe.hpp:14, 40`，Ascend 950DT）只用于推导 60 秒超时的周期数。

## 三、详细讲解

### 3.1 通信底座：HCCL 对称缓冲区与 UBMEM 直访

MegaMoE 的多卡通信**不走任何集合通信算子**（没有 AllReduce/AllGather/Send-Recv 调用）。它的底座是 `HCCLSymmetricBuffer`——把每个 rank 的一段相同大小的 HBM 注册为 **UBMEM 直访区**，之后 Device Kernel 凭远端虚拟地址直接 load/store。类头注释一句话概括（`csrc/backends/hccl.hpp:28`）：

> Leases one process-lifetime HCCL-registered segment and resolves every peer's UBMEM VA.

建立链路分三层，从外到内：

```text
Python SymmBuffer (mega/__init__.py:40-49)
  └─ _C.SymmBuffer (csrc/apis/mega_moe.hpp:45-62, torch::from_blob 包装)
       └─ HCCLSymmetricBuffer::acquire (csrc/backends/hccl.hpp:49-83)   ← 池化租借
            └─ 构造函数 (hccl.hpp:89-164)                                ← 真正注册
```

构造函数的五步（`hccl.hpp:89-164`）值得逐条读：

1. **分配并清零**：`aclrtMalloc(&buffer, num_bytes, ACL_MEM_MALLOC_HUGE_FIRST)` + `aclrtMemset`（`:101-102`）。`num_bytes` 由 `Workspace::get_num_bytes()` 给出，按 2 MiB 对齐（`layout/mega_moe.hpp:246-249`，注释 "HCCL registration uses 2 MiB alignment"）——**每个 rank 的注册段大小完全相同**，这是"对称"的前提。
2. **注册**：构造 `CommMem{type=DEVICE, addr, size}`，以全局唯一 tag 调 `HcclCommMemReg`（`:104-112`）。tag 由静态原子计数器生成（`deep_gemm_mega_moe_<n>`，`:110-111`）；`:109` 的注释是硬约束——**"Ranks must match process-wide registration order, including subgroup allocations"**：各 rank 的注册顺序必须一致，否则远端按 tag 匹配会失败。
3. **选网络层**：`HcclRankGraphGetLayers` 取 layer 列表，断言 `num_layers > 1` 后取 `layers[1]`——注释与断言写明 **"UBMEM requires network layer 1"**（`:120-121`）。
4. **建 peer 通道**：对每个其他 rank 调 `make_peer_descriptor`（`:173-208`）：在 `HcclRankGraphGetLinks` 返回的链路里找 `linkProtocol == COMM_PROTOCOL_UB_MEM` 的那条，把本地/远端 endpoint 的协议、地址、loc 拷进描述符（`notifyNum = 3`，`:199`）；找不到就断言失败（`:205-206`）。随后 `HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, ...)` 一次性建立到所有 peer 的通道（`:133-139`）——**通道引擎是 AIV**，与后文"通信在 AIV 管道上执行"呼应。
5. **解析远端地址**：对每条通道调 `HcclChannelGetRemoteMems`，按 tag 匹配出远端注册段的虚拟地址，填进 `peer_ptrs[peer_rank]`（`:141-163`）。单 rank 时直接返回（`:115-116`）。

`acquire`（`:49-83`）在构造之上加了一层**进程级池**：先按 `(group_name, comm, rank_idx, num_ranks, num_bytes ≥ 需求)` 在池里找最小可复用段（`HcclCommGetHandleWithName` 从 Python 进程组的 HCCL comm 名取句柄，`:58`）；复用前 `aclrtMemset` 清零（`:74`）；租借标志 `leased` 在 `shared_ptr` 析构时归还（`:77-82`）。为什么只租不还？`get_pool` 上方的注释（`:166-167`）说明：**"HcclCommMemReg has no matching HCCL unregister API. Keep registered allocations alive for the process lifetime and reuse them; freeing one leaves stale remote mappings."**——注册了就无法注销，提前释放会让远端留下悬空映射，所以干脆进程持有、反复复用。这也解释了第 43 天 Python 侧 `destroy()` 只是放弃引用。

最后注意 Python 侧的一个细节（`mega/__init__.py:48-49`）：构造完立即 `torch.npu.synchronize()` + `dist.barrier(group)`——注册是各 rank 独立做的 Host 操作，必须全体到齐、Device 侧才不会读到未注册完成的远端地址。

### 3.2 一层地址代数：`SymBuffer::map` 与"远端即普通 GM"

注册完成后，"通信"在 Kernel 里长什么样？答案出乎意料地简单。`layout::SymBuffer`（`layout/sym_buffer.hpp:12-48`）只保存三样东西：本 rank 基址 `base`、`rank_idx`、以及**每个 rank 的基址相对本 rank 的偏移** `offsets[i] = peer_base_i − my_base`（`:23-26`）。远端访问就是一次指针平移：

```cpp
// sym_buffer.hpp:36-41
static constexpr __aicore__ ptr_t map(ptr_t ptr, uint32_t dst_rank_idx, offset_ptr_t offsets) {
    if constexpr (kNumRanks == 1)
        return ptr;                                    // 单 rank 编译期退化
    const uint64_t address = __builtin_bit_cast(uint64_t, ptr) + offsets[dst_rank_idx];
    return __builtin_bit_cast(ptr_t, address);
}
```

也就是说：`workspace` 的每个 `Buffer` 区域在每个 rank 上占据**相同的逻辑偏移**（`offsets` 数组在 Host 构造 `SymBuffer<>` 时随 launch data 传入 Kernel，`csrc/jit_kernels/mega_moe.hpp:50`；`sym_buffer.hpp:16-17` 注释 "Keep rank and the active offsets near base in the launch data"）。于是 `SymBuffer::map(某区域指针, 对端 rank)` 得到的地址，就是对端 rank 上同一逻辑区域。Kernel 里所有跨 rank 操作——`build_dispatch_metadata` 写远端 `src_token_metadata`、pull 读远端 `x`、push 写远端 `combine_input`、prefix 写远端 `combine_ready_prefix`——全部走这个 `map`，**对 Device 代码而言远端读写与本地 GM 读写是同一套指令**（`copy_gm_to_ub` / `copy_ub_to_gm` / `asc_load_dev` / `asc_store_dev` / `asc_atomic_add`），只是 L2 策略不同（见 3.4/3.5）。

这就是"对称缓冲区"的含义：不是共享内存，而是**每 rank 一份同构副本 + 一张地址换算表**。

### 3.3 Device 侧同步协议：`wait_until`、`grid_sync`、`rank_barrier`

`comm/mega_moe.hpp` 除元数据外提供三个同步原语，自底向上：

**（1）`wait_until`（`:12-23`）——带超时的谓词轮询。** 所有跨 rank 等待的最底层：忙等 `pred()` 为真，超过 60 秒（按源码注释，Scalar 侧 1 GHz、SIMT 侧 1.65 GHz 的周期数，`:14-15, 40-41`）就 `print_timeout()` 后 `trap()`。三个打印函数（`:25-33`）输出 rank、core/aiv、peer、prefix 等现场信息——死锁定位的关键。

**（2）`grid_sync`（`:35-63`）——SIMT worker 之间的栅栏。** `build_dispatch_metadata` 以最多 32 个 AIV0 worker、每个 512 线程并行执行（`kNumDispatchThreads = 512`，`layout/mega_moe.hpp:20`），worker 之间要三次对齐（见 3.4）。实现是"相位计数"：`kFinishSumTag = 0x80000000`（`:38-39` 注释 "Worker deltas sum to the phase bit; low bits return to zero"）——worker 0 加 `tag − (num_workers − 1)`，其余 worker 各加 1，**总和恰好让低 31 位回零、第 31 位翻转**；每个 worker 比较 `(counter & tag) == expected_tag`，`expected_tag` 由自己原子加返回的旧值异或得出（`:50`），于是每轮自动换相位，永不清零计数器。第 5 节 B 组练习会让你手算两轮。

**（3）`rank_barrier`（`:65-98`）——跨 rank 全体栅栏。** 它要回答的问题是："本 rank 的全部工作已完成，且**全部其他 rank** 也完成了"。分两层：

- **rank 内先聚齐**：`asc_sync_pipe(PIPE_ALL)` + `asc_sync_inter_arrive/wait(PIPE_MTE3 / PIPE_S, kAllAIVSyncID=14)`（`:70-73`，进入前与 `:95-97` 的退出后各一次）——本 Block 两条 AIV 管道的存量工作排干后才代表本 rank 到达。
- **跨 rank 用 epoch 双槽**：每个 rank 的 `RankBarrier`（`layout/mega_moe.hpp:103-107`）含一个 `epoch` 和两个 `signals`。到达的 rank（`vec_core_idx == 0` 的那个 AIV 执行，`:76`）读 `epoch`，算出方向 `delta = (epoch & 2) ? -1 : +1` 与目标 `target = (delta < 0) ? 0 : num_ranks`（`:78-80`），然后**对包括自己在内的每个 rank** 的 `signals[epoch & 1]` 原子加 `delta`（`:82-87`），加完 `DSB_DDR`（`:88`）再等自己的那个 signal 等于 `target`（`:90-93`）。

epoch 的四拍轮换（`epoch` 每次屏障 `+1`，`:81`）：

| epoch | 槽 | delta | target | 槽内变化（R 个 rank 各加一次） |
|---|---|---|---|---|
| 0 | signals[0] | +1 | R | 0 → R |
| 1 | signals[1] | +1 | R | 0 → R |
| 2 | signals[0] | −1 | 0 | R → 0 |
| 3 | signals[1] | −1 | 0 | R → 0 |

**双槽 + 增减交替，使每次屏障结束时信号恰好归零，下一次直接复用，从不需要清零**——和 `grid_sync` 的相位位是同一个思想。`num_ranks == 1` 时编译期直接返回（`:74-75`）。

`rank_barrier` 在 Kernel 里出现两次：launch 开始清零元数据计数之后（`mega_moe.hpp:1179-1185`，`kBarrierIdx = 0`）——保证所有 rank 的清零与上一轮残留写全球可见后才开始累加；非重叠模式 combine_reduce 之前（`:262`，`kBarrierIdx = 1`）——保证所有 rank 的 push 都到齐后才读 `combine_input`。

配套的还有 `is_block_ready` / `wait_for_block`（`comm/mega_moe.hpp:100-114`）：读进度计数低字节（注释 "Counts fit in the low byte of the atomic uint32 counter"，`:103-104`）比较期望完成数。注意它们服务的是**本 rank 内**的块级进度（`ProgressStage` 四阶段，第 44 天已讲），跨 rank 的就绪信息另有专门通道（3.5 的 prefix）。

### 3.4 EP Dispatch 的通信对象：元数据先行、拉取式数据搬运

现在回答"Dispatch 到底通信了什么"。以"谁 → 谁、传什么、多大"整理成六类对象（前三类属于 Dispatch，后三类属于 Combine；`H`=hidden，`k`=top-k，`T`=本 rank token 数，`R`=rank 数，均匀路由下每 rank 接收 `T·k` 条路由）：

| # | 方向 | 数据 | 规模（每 rank） | 机制与代码位置 |
|---|---|---|---|---|
| 1 | 所有 rank → expert 持有者 | 每 (源 rank, 本地 expert) 的路由计数，打包 `kCountArrivalBit \| count`（到达数 [31:24] + 计数 [23:0]，`layout/mega_moe.hpp:22-24`） | `E/R` 个 uint32 × R 次到达 | 远端原子加 `expert_recv_count_sum`，`comm/mega_moe.hpp:230-236` |
| 2 | 所有 rank → expert 持有者 | 每条路由的 `(weight_bits << 32) \| token_topk_idx`（64 位：fp32 权重位模式 + `token_idx·k + slot`） | `T·k` × 8 B | 远端写 `src_token_metadata`，`comm/mega_moe.hpp:209-213` |
| 3 | expert 持有者 → 源 rank（读） | token 的 FP8 数据行 `H` 字节 + SF `H/64` 个 int16 | `T·k·(R−1)/R` 行 | 拉取式远端读 `x`/`x_sf`，`mega_moe.hpp:828-844` |
| 4 | expert 持有者 → 源 rank（写） | Linear2 输出的 BF16 行 `2H` 字节 | 同上 | 推回式远端写 `combine_input`，`mega_moe.hpp:1024-1036` |
| 5 | expert 持有者 → 源 rank（写） | 该路由的产出块号（`route_m_blocks`，仅重叠模式） | 每 active 路由 1 个 uint32 | 拉取时顺路写远端，`mega_moe.hpp:861-862` |
| 6 | expert 持有者 → 所有源 rank（写） | 已完成生产块前缀 `combine_ready_prefix`（uint64，按 128 B 扇区跨度，`layout/mega_moe.hpp:146`）；全部完成时写 `~0` | 每 rank 1 个 uint64 × R 个对端 | 远端写，`mega_moe.hpp:1062-1077` |

**Dispatch 元数据（对象 1、2）在 `build_dispatch_metadata`（`comm/mega_moe.hpp:117-240`）里四拍完成**（第 44 天已讲流程，今天看协议细节）：

1. **计数**（`:139-158`）：每个 worker 在 UB 里为全部 E 个 expert 计数（`expert_idx < num_experts` 过滤 `-1` 无效路由）；重叠模式下先把 `route_m_blocks` 写成 `~0`（`:152-153`，注释 "Clear route mappings before counts allow any peer to dispatch these inputs"——防止源 rank 读到上一轮的旧块号）；
2. **前缀拼接**（`:160-173`）：`grid_sync<0>` 后跨 worker 做前缀和，得到全局 `total_expert_count` 与各 worker 写偏移；
3. **暂存路由**（`:175-189`）：每条有效路由按 expert 分桶写入 `staged_token_topk_idx`，`grid_sync<1>`；
4. **发给持有者**（`:192-239`）：按 warp 分工，把 `(weight_bits << 32) | token_topk_idx` 经 `SymBuffer::map` 写进持有该 expert 的 rank 的 `src_token_metadata`（`:209-213`，注释 "Ship route weights with indices to avoid per-route remote Scalar reads"——把权重随元数据一起发，避免 pull 时逐路由远端标量读）；`grid_sync<2>` 后由 worker 0 发布计数：先写 `expert_recv_count`（按 `(src_rank, local_expert)` 寻址），`asc_threadfence()` 后再原子加 `expert_recv_count_sum` 的 `kCountArrivalBit | count`（`:230-236`，注释 "Release the count store before publishing its arrival"——先保证计数值可见，再让"到达"可见）。

三次 `grid_sync` 各保护一个不变量：计数齐了才能算前缀；暂存齐了才能按 warp 读取；元数据发完了才能发布计数——消费者 `wait_dispatch_metadata` 只看计数，元数据必须在计数到达前落盘。

`wait_dispatch_metadata`（`:242-275`）在接收侧收尾：对每个本地 expert 等到达位凑满 R（`:253-259`），把计数累加进可选的 `cumulative_local_expert_recv_stats`（`:261-262`，第 43 天的跨 launch 统计），再按 `ceil_div(count, BLOCK_M)` 折算 M-block 求和进 `num_m_blocks_certificate`（`:263-274`）——这个"证书"就是 routed 池的规模，AIC 与 AIV1 都靠它规划任务总量。注意 `csrc/apis/mega_moe.hpp:38-39` 的断言 `R·T_max·k < kCountArrivalBit`：全局路由总数必须小于 2²⁴，因为最坏情况全部路由涌向同一个本地 expert，计数值要塞进低 24 位。

**Dispatch 数据（对象 3）是"拉"不是"推"**（第 44 天已讲主线，今天看通信细节）：`advance_dispatch_pull`（`mega_moe.hpp:763-873`）由 expert 持有者按需远端读。三个通信侧细节：

- **源行置换**：用 `next_coprime` 选互质步长在拼接后的源行序列上跳跃（`:801-802`，注释 "Permute the concatenated source rows without changing their counts"）——让连续拉取的行均匀散布在不同源 rank 上，避免远端读热点；
- **L2 策略**：token 用 `NORMAL_LAST_VICTIM`、SF 用 `NORMAL_FIRST_VICTIM`（`:841-844`）——拉来的数据马上要写进本地 `l1_acts`（`kWriteThroughShare`，`:850-855`），远端读不值得在本地 L2 留驻；
- **顺路写回执**：每行解析 `src_token_metadata` 得 `(src_token_idx, src_topk_idx)` 后，把 `src_metadata[pool_m_idx] = (src_rank << 24) | combine_m_idx` 存在**本地**（`:864-865`，push 要用），重叠模式再把 `route_m_blocks` 写到**源 rank**（对象 5，`:861-862`）。

对照第 44 天的进度表：pull 完成一个 M-block 后发布本 rank 内的 `kDispatched`（`:772`），AIC 块级等待（`:535-540`）——**Dispatch 数据搬运与 GEMM1 计算按 M-block 流水重叠，这是融合的第一处 overlap**。

### 3.5 EP Combine 的通信对象：push、完成前缀与 early reduce

Combine 把每个 token 在各 expert 算出的 k(+1) 份结果送回源 rank 求和。两段协议：

**Push（对象 4）**：`advance_combine_push`（`mega_moe.hpp:999-1053`）对每个池内行：读本地 `src_metadata` 解出 `(dst_rank_idx = metadata >> 24, combine_m_idx = metadata & 0xffffff)`（`:1019-1020`），从 `routed.output` 的 ring 行拉一行 BF16 进 UB（4 个 push stage，`:1024-1027`），再 `copy_ub_to_gm` 写进**目标 rank** 的 `combine_input[combine_m_idx]`（`sym_buffer.map` 平移，`:1030-1036`）。通信侧三个细节：写远端用 `asc_store_l2_cache_mode::NOTALLOC_CLEAN`（`:1035`）——远端写不分配本地 L2；每步至多 2 行就返回主循环让位（`:1015` 注释 "Send at most two rows before yielding"）；每 `kNumMBlocksPerWait = 32·32/256 = 4` 个 M-block 批量等待并发布 `kCombineSent`（`:1003, :1044-1050`），让 AIC 能复用 ring 行。

**完成前缀（对象 6）与 early reduce**：非重叠模式下这很简单——`rank_barrier<1>` 等全体到齐再顺序归约。重叠模式（`kOverlapReduce`，判据 `T_max·(k+(S>0))·H·2B > 32 MiB`，`:1138-1140`）下，源 rank 不能等"全部 push 完成"，要看**每个 token 的各条路由分别好了没有**，这是一套生产者-消费者协议：

- **生产者发布**：`publish_combine_prefix`（`:1062-1077`）把"已完成且 `kCombineSent` 凑满的块前缀"写给**每一个**源 rank 的 `combine_ready_prefix[my_rank]`（`asc_sync_data_barrier(DSB_DDR)` 后 `asc_store_dev`，`:1073-1076`）；全部完成时写 `~uint64{0}`（`:1072`）；
- **消费者检查**：`advance_combine_reduce`（`:139-249`）处理每个 token 前，逐 top-k 槽读**自己的** `route_m_blocks[token_topk_idx]` 得知该路由的产出在哪个生产块，与缓存的 peer 前缀比较；缓存不足才刷新远端 `combine_ready_prefix`，仍不足则记下 `blocking_rank_idx`、跳到下一个 token（`:175-197`，注释 "Completed prefixes only advance; refresh a peer when its cached value is insufficient"）。超时打印 `print_peer_timeout` 会带上阻塞的 peer 与其 prefix（`mega_moe.hpp:282-284`）；
- **收尾排干**：主循环结束后 `combine_reduce`（`:251-295`）排干剩余 token；重叠模式还要等每个 peer 的 prefix 变成 `~0`（`:286-294`，注释 "Empty ranks also wait for peer writes before the next invocation reuses scratch"——没有本地 token 的 rank 也必须等，因为下一次 launch 会复用 `combine_input`，而远端的 push 可能还在途）。

加上对象 5（`route_m_blocks` 回执），Combine 一共三类跨 rank 写 + 一类本地查询。**"计算/通信时序图"（当天产出）要画的就是这些对象在三类角色时间线上的位置。**

### 3.6 计算和通信为什么需要融合

先看**不融合的代价**。baseline（`tests/test_mega_moe.py:130-156`，`run_baseline`）用现成组件逐段拼装：`ep_buffer.dispatch` → `m_grouped_fp8_fp4_gemm_nt_contiguous`（Linear1）→ TileLang SwiGLU → 同一 GEMM（Linear2）→ `ep_buffer.combine`，行首注释直言 **"Dispatch -> Linear1 -> SwiGLU -> Linear2 -> combine, without communication overlap"**（`:132`）。benchmark 里 baseline 的 kernel 列表（`:207-209`）暴露了段结构：`dispatch_impl`、`dispatch_copy_epilogue_impl`、`fp8_dequant_gemm_impl`、`fp8_gemm_impl`、`swiglu_forward_kernel`、`combine_impl`、`combine_reduce_epilogue_impl`——**七种以上独立 kernel 启动**。段与段之间：中间张量（`l1_y` [recv, 2I] bf16、`l2_x`、`l2_y`，`:139-145`）全部经 HBM 落盘再读回；每段结束隐含全局同步，通信 kernel 与计算 kernel 串行占用设备。

融合 Kernel（`mega_moe_impl`）把五段装进一次启动，收益按机制列：

1. **中间结果不重复搬运**：GEMM1→SwiGLU 走 Block 内 UB 交接（第 44 天）；Dispatch 拉来的行直接写进 `l1_acts`——它同时就是 GEMM1 的 A 矩阵；push 直接从 `routed.output` 读（写透 L2 共享，`kWriteThroughShare`，mode 12，`mega_moe.hpp:107`），无专门"发送缓冲"。
2. **通信内嵌在计算流水里，无独立通信 kernel**：pull/push/prefix 都是 AIV1 主循环 `wait_until` 里与其他工作轮转的普通 DMA（`:1080-1120`），`prefer_push` 翻转保证 push 与 pull 都不被饿死（`:1108-1118`）；pull 按 M-block 发布 `kDispatched`、AIC 块级消费——**Dispatch 与 GEMM1 重叠**；push 与 SwiGLU/GEMM2 的 epilogue 穿插——**Combine 与 GEMM2 重叠**。
3. **环形接收池控制重叠深度**：三块 routed 缓冲只开 `min(num_max_m_blocks, 窗口·32)` 行（`layout/mega_moe.hpp:221-226`，注释 "Measured overlap window; the deadlock-safe minimum can still lose throughput"），池容量小意味着边消费边复用，但回绕写入要等旧数据生命周期结束（第 44 天的三条等待）。
4. **reduce 尾部也参与重叠**（`kOverlapReduce`）：大 T 时 combine 数据量大（这正是 32 MiB 阈值的含义），reduce 若完全留到尾部会显著拉长时延；于是 AIV1 在主循环的 reduce 窗口里每步推进 2 个 chunk（`:1085-1089`），AIV0 在权重流水的每个 K 块间隙推进 1 个 chunk（`:389-391`），收尾才排干（`:251-295`）。3.5 的 prefix 协议就是为这个"提前开始"服务的。
5. **拉取式 Dispatch 天然摊平通信**：expert 持有者按计算节奏拉数据，不需要源 rank 集中发送、也不需要额外的接收缓冲管理。

一句话总结：**融合的本质是把"通信"从独立的集合调用变成 AIV 管道上的普通数据流，让块级进度（`ProgressStage`）与跨 rank 前缀（`combine_ready_prefix`）两套协议把它与计算流水对齐**。

### 3.7 不同 token 数下的瓶颈初判

现在用可复现的公式做定量分析。benchmark（`tests/test_mega_moe.py:203-243`）的三个口径：

```text
flops       = 6 · H · I · (recv_tokens + T · shared)          (:221)
hbm_bytes   ≈ 权重一次 + 激活读写 + BF16 输出                  (:222-228，第 50 天展开)
comm_bytes  = remote_tokens · H · 3                            (:229-231，注释：
               "Remote FP8 pulls + BF16 pushes; exclude local routes, SFs, metadata")
remote_tokens（均匀路由）= T · k · (R − 1) / R
```

`H × 3` 的含义：每条远端路由一次 FP8 拉取（1 B/element）加一次 BF16 推回（2 B/element）。带宽 = 平均每 rank 的 comm_bytes / 全部 rank 的最大时延（`:239` 注释 "Latency is the slowest rank"）。

**用 README 的 EP8 数据表（`README.md:164-173`，top-k=6、1 shared expert、8 rank）逐行验证**（均匀路由假设下误差 <1%）：

| 配置 E/H/I/T | remote/rank | comm bytes | 表中时延 | 手算带宽 | 表中带宽 |
|---|---|---|---|---|---|
| 96 / 5120 / 2304 / 64 | 336 | 5.16 MB | 138.7 µs | 37.2 GB/s | 37.5 |
| 96 / 7168 / 3072 / 64 | 336 | 7.22 MB | 222.2 µs | 32.5 GB/s | 32.8 |
| 96 / 5120 / 2304 / 256 | 1344 | 20.6 MB | 225.5 µs | 91.5 GB/s | 91.6 |
| 96 / 7168 / 3072 / 256 | 1344 | 28.9 MB | 359.2 µs | 80.5 GB/s | 80.5 |
| 384 / 5120 / 2304 / 4096 | 21504 | 330.3 MB | 2567.7 µs | 128.7 GB/s | 128.8 |
| 384 / 7168 / 3072 / 4096 | 21504 | 462.4 MB | 4600.7 µs | 100.5 GB/s | 100.6 |
| 384 / 5120 / 2304 / 16384 | 86016 | 1321.2 MB | 9768.6 µs | 135.3 GB/s | 135.2 |
| 384 / 7168 / 3072 / 16384 | 86016 | 1849.7 MB | 17904.2 µs | 103.3 GB/s | 103.3 |

（手算例：`remote = 4096·6·7/8 = 21504`，`bytes = 21504·7168·3 ≈ 462.4 MB`，`462.4 MB / 4600.7 µs ≈ 100.5 GB/s`。）由此得出两个瓶颈形态：

**小 T（64/256，decode 场景）——时延与同步受限。** 通信带宽只有 33～92 GB/s，远低于大 T 的 100～135：链路远未跑满，时间花在固定成本上——两次 `rank_barrier`、元数据三拍握手、证书等待、M-block 级进度等待、以及每行 push/pull 的时延。计算侧同样稀薄：均匀路由下每个 expert 只分到 `R·T·k/E` 个 token（E=96、T=64 时仅 32 个 → `valid_m=32`，MAD 档 M=128 的利用率只有 1/4；E=384、T=64 时每 expert 8 个 token，利用率 8/128），MAD 大量空转。**小 T 时谈"带宽"意义不大，时延才是关键，融合省掉的每次 kernel 启动与全局同步在这里占比最高。**

**大 T（4096/16384，prefill 场景）——带宽与计算双双逼近平台。** 通信带宽在 ~100–135 GB/s 平台化，TFLOPS/rank 也在 790–846 平台化（flops 口径可复现：`6·7168·3072·(98304+16384) / 17904.2 µs ≈ 846.5`，对照表中 846.3）；此时每 expert 有 512～2048 个 token，M-block 基本满块（`valid_m` 为 256 或 128 的整块）。**两者同时接近上限，恰恰说明 overlap 在起作用**——若串行，通信时间与计算时间相加会同时压低两个口径。另一个佐证是 HBM 带宽列随 T 增大反而下降（1850→325 GB/s）：口径是 bytes/整 kernel 时长，T 大时计算占主导，HBM 通道大量时间闲置。

**用源码规则判断走哪条路径**：`kOverlapReduce = T_max·(k+(S>0))·H·2B > 32 MiB`（`:1138-1140`）。EP8 默认配置（k=6、S=1，输入份数 k+1=7）：T=64 → 约 4.4/6.1 MiB、T=256 → 约 17.5/24.5 MiB（H=5120/7168），均关闭——小 T 时 reduce 尾部占比小，走"barrier 后一次性归约"的简单路径；T=4096 → 约 280/392 MiB，开启——reduce 尾部必须与主循环重叠。

注意：以上是**从公开数据与口径公式出发的初步判断**，精确的瓶颈归因（UBMEM 链路能力、AIV 通道利用率）需要 profiling，那是第 46 天（`scripts/profiling/msprof_mega_moe.py` 的 PIPE 级 trace）与第 51 天的主题。

### 3.8 当天产出·参考：MegaMoE 计算/通信时序图

参考版本给出两张：**非重叠模式**（小 T，如贯穿配置或 EP8 T=64）与**重叠模式**（大 T，如 EP8 T=4096）。图中 `rb` = `rank_barrier`，`gs` = `grid_sync`，加粗箭头为跨 rank 通信。

**图 1：非重叠模式（`kOverlapReduce = false`）**

```text
t ──────────────────────────────────────────────────────────────────────────────►
      ①元数据阶段                ②pull + GEMM1/SwiGLU 块级流水        ③push      ④归约
AIV0  [清零][rb0]──[SIMT元数据: 计数|gs0|前缀|暂存|gs1|发送|gs2]──[证书]
      [FP4→FP8 NZ 权重流水（随 AIC 的 routed 任务）─────────────────────][reduce 区间──────]
AIV1  [清零][rb0]─────────────────────────────────[pull b0][pull b1]··[SwiGLU b0]··[push]──[rb1][reduce 区间]
AIC   [shared L1][shared L2]──[等kDispatched(b0)]──[GEMM1 b0]··[GEMM2 b0]──────────[等kCombineDoneID]

通信标记（谁→谁）：
  ① 元数据：计数+路由打包   所有rank ──写──► expert持有者（对象1/2，每路由 8B + 每(src,expert) 4B）
  ② pull：FP8 行 + SF      expert持有者 ──读──► 源rank 的 x（对象3，H + H/32 B/行）
  ③ push：BF16 行          expert持有者 ──写──► 源rank 的 combine_input（对象4，2H B/行）
  rb0/rb1：all↔all 的 epoch 双槽信号（3.3）
关键串行点：rb0 → 证书（全部计数到达）→ 第一块 kDispatched → rb1 → reduce
```

**图 2：重叠模式（`kOverlapReduce = true`）——AIV1 主循环视角**

```text
t ──────────────────────────────────────────────────────────────────────────────►
AIV1  [清零][rb0][元数据轮询→证书]
      ┌──────────────────── wait_until 主循环（每步只做一件事就返回）────────────────────┐
      │ reduce 窗口开？ → publish_combine_prefix() ＋ advance_combine_reduce(2步) ←读peer前缀│
      │ 取任务(epi)    → advance_epilogue()（SwiGLU / 归还信用 / 发布 kLinear1·2Ready）     │
      │ 通信轮转      → prefer_push ? advance_combine_push(≤2 行远端写)                     │
      │                  : advance_dispatch_pull（远端读 ＋ 写 route_m_blocks 回执）        │
      │ （prefer_push 每步翻转，防止 push 或 pull 饿死）                                    │
      └── pull/push/epilogue 全部完成 且 ready_prefix.finished ────────────────────────────┘
      combine_reduce：排干剩余 token（按 token 查 route_m_blocks × peer 前缀，阻塞则跳过）
                      → 等每个 peer 的 combine_ready_prefix == ~0（防下轮复用踩踏）
AIV0  [清零][rb0][SIMT元数据][证书]──[权重流水 {每个 K 块末尾空隙: advance_combine_reduce(1步)}]──[排干]
AIC   [shared L1/L2]──[GEMM1/GEMM2 块级流水：kDispatched→A、kLinear1Ready→A、kCombineSent→ring 复用]
```

画自己的版本时的自查点：rb0 之前清零了什么、三次 `grid_sync` 保护什么、pull 与 push 各自的远端方向与字节数、`route_m_blocks` 与 `combine_ready_prefix` 谁写谁读、`~0` 的语义、AIC 在哪些点等待、为什么空 rank 也要等 `~0`。**瓶颈标注**（3.7 的结论）也应落图：图 1 上标出 rb0→证书的串行前缀（小 T 的固定成本）；图 2 上标出 push 与 GEMM2 epilogue 的并行段、reduce 窗口与主循环的交叠（大 T 的 overlap 收益）。

## 四、源码导读

今天的主线文件与阅读站点（建议按顺序读，前两个建议逐行）：

| # | 文件 | 关键函数/对象 | 看什么 |
|---|---|---|---|
| 1 | `csrc/backends/hccl.hpp` | `HCCLSymmetricBuffer::acquire` `:49-83`、构造函数 `:89-164`、`make_peer_descriptor` `:173-208`、`get_pool` `:166-171` | 注册五步、UBMEM 链路筛选（`:186`）、`COMM_ENGINE_AIV`（`:136`）、tag 匹配（`:153-162`）、"无注销 API"注释 |
| 2 | `deep_gemm/include/deep_gemm/comm/mega_moe.hpp` | `wait_until` `:12-23`、`grid_sync` `:35-63`、`rank_barrier` `:65-98`、`is_block_ready`/`wait_for_block` `:100-114`、`build_dispatch_metadata` `:117-240`、`wait_dispatch_metadata` `:242-275` | 三个同步原语（相位/epoch 注释）、元数据四拍与三次 grid_sync、打包发布顺序（`:230-236`） |
| 3 | `deep_gemm/include/deep_gemm/layout/sym_buffer.hpp` | `SymBuffer::map` `:36-46`、构造 `:23-26` | 一层地址代数；`kNumMaxRanks = 256`（`:10`，"Communication kernels encode rank indices in 8 bits"，与 `src_metadata` 的 rank[31:24] 呼应） |
| 4 | `deep_gemm/include/deep_gemm/mega_moe.hpp`（通信站点） | `kOverlapReduce` `:1138-1140`、入口清零+`rank_barrier` `:1179-1185`、`advance_dispatch_pull` `:763-873`、`advance_combine_push` `:999-1053`、`publish_combine_prefix` `:1062-1077`、主循环 `:1080-1120`、`advance_combine_reduce` 的 peer 检查 `:175-197`、`combine_reduce` 收尾 `:251-295`、AIV0 空隙 reduce `:389-391` | 每个站点标注方向（读/写）、远端目标、L2 模式 |
| 5 | `deep_gemm/include/deep_gemm/layout/mega_moe.hpp` | `kCountArrivalBit` `:22-24`、`RankBarrier` `:103-107`、metadata 区 `:142-163`、`combine_input` `:206-208`、`route_m_blocks` `:213-215`、`get_num_bytes` `:246-249` | 通信相关区域的容量与对齐（`combine_ready_prefix` 用 `kL2SectorBytes` 跨度防伪共享） |
| 6 | `csrc/apis/mega_moe.hpp` | `get_workspace` `:27-42`（`:38-39` 的 `R·T·k < kCountArrivalBit` 断言）、`SymmBuffer` `:45-62` | 视图借用注册内存；`from_blob` 的 deleter 里 `aclrtSynchronizeDevice`（`:55-59`，"Wait for device work before the last owner returns the lease"） |
| 7 | `csrc/jit_kernels/mega_moe.hpp` | `launch_mega_moe` `:24-51` | launch 前 `aclrtMemsetAsync` 清零 progress 区（`:44-46`）；`SymBuffer<>` 作为 kernel 参数（`:50`） |
| 8 | `deep_gemm/mega/__init__.py` | `SymmBuffer.__init__` `:40-49`、`destroy` `:62-73` | `synchronize + dist.barrier` 的必要性；destroy 只弃引用 |
| 9 | `tests/test_mega_moe.py` | `run_baseline` `:130-156`（`:132` 注释）、`benchmark` `:203-243`（kernel 列表 `:207-209`、三口径 `:221-231`）、最大容量注册与复用 `:249-258` | baseline 的"无重叠"证据；comm 字节的排除项注释 |
| 10 | `README.md` | MegaMoE 节 `:160-173` | EP8 数据表（3.7 的验证对象） |

**两个主文件的分块导读**：

- **`hccl.hpp` 按调用顺序**：`DG_HCCL_CHECK` 封装（`:19-24`，错误串经 `HcclGetErrorString`）→ `acquire` 的池查找条件（group/comm/rank/大小匹配 + 最小适配，`:62-69`）→ 构造五步（3.1）→ `make_peer_descriptor` 里只认 `COMM_PROTOCOL_UB_MEM`（`:184-186`），`notifyNum = 3`（`:199`）。
- **`comm/mega_moe.hpp` 按依赖顺序**：`wait_until`（超时兜底）→ `grid_sync`（注意 SIMT 侧不能复用 Scalar lambda，注释 `:52`，改用 `clock64` 与 `__asc_simt_vf::printf`）→ `rank_barrier`（先块内 `kAllAIVSyncID=14` 再跨 rank）→ `build_dispatch_metadata`（四拍；`static_assert` 限制 UB 用量 `:133`）→ `wait_dispatch_metadata`（到达位 → 证书；`DSB_DDR` 在 `:265, :273` 两处）。

## 五、实践步骤

### A 组：有 Ascend 环境（8 卡可对照 README 表；2 卡可跑通但数值不可对照）

1. **复现 3.7 的两个极端 case**：

   ```bash
   python tests/test_mega_moe.py --num-processes 8 --num-experts 96 --hidden 5120 \
       --intermediate-hidden 2304 --num-topk 6 --num-tokens 64 --num-shared-experts 1 \
       --num-correctness-tests 0 --skip-baseline
   python tests/test_mega_moe.py --num-processes 8 --num-experts 384 --hidden 7168 \
       --intermediate-hidden 3072 --num-topk 6 --num-tokens 4096 --num-shared-experts 1 \
       --num-correctness-tests 0 --skip-baseline
   ```

   （`--skip-baseline` 会连带跳过正确性断言，`test_mega_moe.py:309`，也免去第三方依赖。）记录两条输出里的 Comm GB/s 与 TFLOPS，对照 README 表与 3.7 手算值；用 `buffer.expert_recv_count.cpu()` 验证均匀路由假设的误差（`recv_counts.sum(dim=0)` 即每 expert 接收数）。

2. **空输入路径**：`--num-tokens 0` 重跑第一条命令。对照源码推演：T=0 时任务循环为空，但 `rank_barrier`（两次）、元数据握手、`combine_reduce` 里"空 rank 也等 peer 的 `~0`"（`mega_moe.hpp:286-294`）必须完整走完——这就是 3.7 说的"小 T 固定成本"的极端形态。

3. **（进阶）注册顺序与池复用**：阅读 `test_mega_moe.py:249-258`——测试先按 `buffer_size` 选出最大配置注册一次 `base`，后续 case 全部 `base=base` 复用。对照 `hccl.hpp:62-75` 回答：为什么所有 case 的 `num_bytes` 必须由同一个 `get_num_bytes()` 逻辑给出？如果两个 case 的 metadata 区域偏移不同，复用是否安全？（提示：`slice_input_buffers` 按 `workspace` 偏移切视图。）

### B 组：无 Ascend 环境（静态任务）

1. **带宽手算（当天核心练习）**：用 3.7 的公式重算 README 表的 8 行（3.7 已给全部答案，先遮住再算）。再回答：把 k 从 6 改成 8、其余不变，T=4096/H=7168 的 comm bytes 变为多少？（预期 `4096·8·7/8·7168·3 ≈ 617 MB`。）如果只有 2 卡（R=2），EP 系数 `(R−1)/R` 变为多少，对小 T 的 comm 占比有何影响？
2. **`rank_barrier` epoch 演算**：R=3，初始 `epoch=0`、`signals=[0,0]`。写出连续四次 barrier 的 (epoch, 槽, delta, target, 槽终值) 表（预期：0/s0/+1/3/3；1/s1/+1/3/3；2/s0/−1/0/0；3/s1/−1/0/0）。然后回答：如果只有单槽会发生什么？（第二次 barrier 的等待条件会与第一次的残留值混淆，除非显式清零——而清零本身有跨 rank 竞态。）
3. **`grid_sync` 相位演算**：`num_workers=3`，`kFinishSumTag=0x80000000`，counter 初值 0。写出两轮同步中每个 worker 的原子加值与 counter 终值（预期第一轮终值 `0x80000000`，第二轮经 `0xFFFFFFFE → 0xFFFFFFFF → 0x0` 回绕归零），并解释 `expected_tag = (old_value ^ tag) & tag`（`comm/mega_moe.hpp:50`）为什么能区分相邻两轮。
4. **`kOverlapReduce` 判据**：对 EP8 默认（k=6、S=1）的 T=64/256/4096/16384 与 H=5120/7168 的全部组合判断开关（预期：T=64/256 全关；T=4096/16384 全开；边界 `T_max > 32 MiB / ((k+1)·H·2)`，H=7168 时约 T=335）。再对贯穿配置 R=2, k=2, S=0, T=128, H=512 判断（预期关，`128·2·512·2 = 256 KiB`）。
5. **通信对象标注**：合上教程，默写 3.4 开头的六类对象表（方向、数据、规模、代码位置），特别标注哪几类只在 `kOverlapReduce` 开启时出现（预期：对象 5 `route_m_blocks` 只在重叠模式写远端；关闭模式下对象 6 被 `rank_barrier<1>` 整体替代）。
6. **画时序图（当天产出）**：按 3.8 的自查点独立画两张图，再对照修正。

## 六、当天产出

**MegaMoE 计算/通信时序图**——以 3.8 节两张参考图为答案，合上教程独立完成，至少包含：AIC/AIV0/AIV1 三条时间线；两次 `rank_barrier` 与元数据阶段的相对位置；六类通信对象的标注（方向、数据类型、每行字节数）；`kDispatched`/`kLinear1Ready`/`kLinear2Ready`/`kCombineSent` 四个进度点在时间线上的发布与消费位置；重叠模式下 reduce 窗口、`prefer_push` 轮转与 `~0` 收尾；最后把 3.7 的瓶颈结论标到图上（小 T 的串行前缀、大 T 的并行段）。有环境的同学附上 A 组第 1 步两个 case 的 Comm/TFLOPS 记录。写完后把画不出来的通信对象标出，回到第四节对应站点重读。

## 七、自测题

1. `HCCLSymmetricBuffer` 建立的是什么？为什么 MegaMoE 不需要调用任何集合通信算子？
   > 它把每个 rank 一段相同大小的 HBM 经 `HcclCommMemReg` 注册为 UBMEM 直访区，用 `COMM_PROTOCOL_UB_MEM` 链路建 `COMM_ENGINE_AIV` 通道，并把每个远端的 VA 填进 `peer_ptrs`（`hccl.hpp:89-164`）。此后 Kernel 里远端读写只是 `SymBuffer::map` 平移后的普通 GM 访问，通信内嵌在 AIV 的 MTE2/MTE3 管道上，无需集合调用。
2. `HcclCommMemReg` 的内存 tag 为什么要跨 rank 顺序一致？注册段为什么进程持有不释放？
   > 远端通过 `HcclChannelGetRemoteMems` 按 tag 匹配同一段内存（`:141-163`），各 rank 注册顺序不一致则匹配失败（`:109` 注释）；HCCL 没有对应的注销 API，提前释放会给远端留下悬空映射，所以池化复用到进程退出（`:166-171`）。
3. `SymBuffer::map` 做了什么运算？单 rank 时如何退化？
   > `ptr + (远端基址 − 本地基址)`（`sym_buffer.hpp:36-41`）；`kNumRanks == 1` 时编译期直接返回原指针。
4. `rank_barrier` 为什么用双槽 epoch 而不是每次清零信号？
   > epoch 四拍轮换：+1 凑 R、+1 凑 R、−1 归 0、−1 归 0（`:78-87`），每次屏障结束信号恰好归零、双槽交替复用；显式清零需要额外的跨 rank 同步，本身有竞态。
5. `build_dispatch_metadata` 的三次 `grid_sync` 分别保护什么？
   > gs0：计数齐了才能算跨 worker 前缀和（`:163`）；gs1：路由暂存齐了才能按 warp 分桶读取（`:190`）；gs2：元数据写远端完成后才能发布计数到达位（`:215`），保证消费者只见计数时数据必已落盘。
6. Dispatch 为什么设计成拉取式？`next_coprime` 置换解决什么问题？
   > expert 持有者按 GEMM1 的块级节奏拉数据，天然与计算重叠、无需集中发送（`advance_dispatch_pull`）；置换让连续拉取的源行均匀散布在不同源 rank 上，避免远端读热点（`mega_moe.hpp:801-802` 注释）。
7. push 写远端用哪种 L2 模式？为什么不留在本地 L2？
   > `asc_store_l2_cache_mode::NOTALLOC_CLEAN`（`:1035`）——远端写的数据本地不再读，不占用本地 L2 容量；本地生产侧的 `routed.output` 才用 `kWriteThroughShare`（mode 12）让 push 读取时命中。
8. `combine_ready_prefix` 协议中谁写、谁读、`~0` 的语义是什么？
   > 生产 rank 的 AIV1 把已完成块前缀写给每一个源 rank（`publish_combine_prefix`，`:1062-1077`）；源 rank 的 reduce 结合自己记录的 `route_m_blocks` 判断某条路由的产出块是否已完成（`:175-197`）；`~uint64{0}` 表示"全部完成"，收尾时每个 AIV 都要等自己负责的 peer 变成 `~0`（`:286-294`），防止下一次 launch 复用 `combine_input` 时远端 push 还在途。
9. `kOverlapReduce` 的判据是什么？EP8 默认配置（k=6、S=1、H=7168）下 T=256 与 T=4096 各走哪条路径？
   > `T_max·(k+(S>0))·H·2B > 32 MiB`（`:1138-1140`，实测阈值）。T=256：`256·7·7168·2 ≈ 24.5 MiB`，关闭——`rank_barrier<1>` 后一次性归约；T=4096 约 392 MiB，开启——reduce 窗口穿插进 AIV1 主循环与 AIV0 权重流水空隙。
10. 为什么小 T 时通信带宽（如 37.5 GB/s）远低于大 T（如 135 GB/s）？给出至少两个原因。
    > 一是固定成本占比高：两次 rank_barrier、元数据三拍握手与证书等待、块级进度等待都不随 T 缩小；二是 MAD 利用率低：每 expert 仅 `R·T·k/E` 个 token（E=96、T=64 时 32 个），`valid_m` 小于档位 M，计算稀薄拉长总时延，带宽口径（bytes/总时延）随之被压低。
11. baseline 为什么无法重叠计算与通信？代码证据在哪？
    > 五段是独立的库调用/kernel（`run_baseline`，`test_mega_moe.py:130-156`，行首注释 "without communication overlap"），段间中间张量经 HBM 往返（`l1_y`/`l2_x`/`l2_y`，`:139-145`）且隐含全局同步；benchmark 的 baseline kernel 列表（`:207-209`）列出七种以上 kernel 名，串行占用设备。

---

明天（第 46 天）转入 MegaMoE 的测试与性能入口：阅读 `scripts/profiling/msprof_mega_moe.py` 的 PIPE 级 trace 采集，看 `tests/test_mega_moe.py` 如何组织多 rank 测试与最大容量注册复用，并理解性能输出中计算带宽、内存带宽与通信带宽三个口径的来源——今天 3.7 的初步瓶颈判断，届时可以用 profiling 数据逐条验证。

