# 第 6 周 · 第 5 天（总第 40 天）：学习 Paged/Sparse MQA

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 6 周"Grouped GEMM 与 MQA Logits"的第 5 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 55 分钟，源码导读约 25 分钟，实践步骤约 60 分钟，自测题约 15 分钟。

昨天（第 39 天）我们读完了普通 MQA Logits：`logits[q, j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`，KV 连续存放、每行 query 有一个 `[ks, ke)` 窗口。今天沿着同一条公式走进两个生产变体：**Sparse MQA**（每个 query 只计算自己选中的 top-k 个 KV 块，Prefill 场景）和 **Paged Sparse MQA**（KV 分页存放在 paged cache 里、按请求组织，Decode 场景，并带 FIRST/REUSE 快照复用）。按计划，今天要读 `sparse_mqa_logits.hpp`、`paged_sparse_mqa_logits.hpp` 以及对应的 metadata 和 scheduler 文件，理解 **Paged KV 寻址和 Sparse block 选择**、**FIRST/REUSE 状态与 KV 快照复用**，并**梳理 Prefill 与 Decode 的主要差异**。当天产出是一张 **Paged/Sparse MQA 元数据与数据流图**（第六节给出参考版）。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 Sparse block 的选择语义：第 i 个块覆盖 token 区间 `[i·B + ks%B, i·B + ks%B + B)`（B = `sparse_block_kv` ∈ {8,16}），每个 query 取前 `min(capacity, ceil((ke−ks)/B))` 个块，输出是 `[Q, capacity×B]` 的 bf16 压缩矩阵（`csrc/apis/attention.hpp:56-58`、`deep_gemm/__init__.py:94-97`）；
2. 画出两套元数据张量的内部布局：非分页的 `SparseMQALayout` 四段（header / QBlockHeader / physical_kv_token_starts / kv_split_outputs + schedule，`csrc/jit_kernels/sparse_mqa_logits.hpp:13-39`）与分页的 `PagedSparseMQALayout` 四段（mapping / kv_splits / physical / compact，`csrc/jit_kernels/paged_sparse_mqa_logits.hpp:14-43`）；
3. 手算 Paged KV 寻址：逻辑块 → 逻辑页 → `block_table` 查物理页 → 物理 token 起点 `block_table[b/(page/B)]·page + (b%(page/B))·B`，以及 fused 页布局 `[page DATA | page SF]`（`paged_sparse_mqa_logits_metadata.hpp:87, :166`、`paged_sparse_mqa_logits.hpp:60-62`）；
4. 复述非分页元数据 kernel 的四步：`stage_indices`（块 ID → token 起点）→ `sparse_mqa_merge`（4 路 sorted union + query_mask）→ `sparse_mqa_kv_split_outputs`（64 块/split 的 masks 与 output_ranges）→ `sparse_mqa_logits_schedule_vf`（work 均分写 `ScheduleBoundary`）（`sparse_mqa_logits_metadata.hpp:116-143, :23-66, :68-114, :191-302`）；
5. 解释分页 prepare kernel 的流程与 `SplitKV::control` 位域：`plan_range` 代价模型 → `find_prepare_range` 负载均衡 → `bucket_q_block_interval` 按 query_mask 分 16 桶 → `emit_bucket_kv_splits` 生成 SplitKV 记录（`paged_sparse_mqa_logits_metadata.hpp:21-66, :360-394, :68-207, :209-301`）；
6. 讲清 FIRST/REUSE 状态机：FIRST 由 AIV 把分页 KV gather + NZ 转换后物化到 metadata 内的 compact 区并送 L1，REUSE 由 AIC 直接从 compact 区 GM→L1；状态由 Python 侧 `metadata._paged_sparse_mqa_state` 管理，改 KV 内容必须重建 metadata（`paged_sparse_mqa_logits.hpp:108-145, :209-226`、`deep_gemm/__init__.py:107-134`）；并用一张对比表说明 Prefill 与 Decode 路径的主要差异。

## 二、预备知识

- **第 39 天**：MQA Logits 公式、`__mix__(1,2)` 的 AIC/AIV 分工、FIX 管上的 ReLU、`copy_l0c_to_ub`/`mad`/MX scale 装载——今天的两个 Kernel 完全建立在这些设施上，只是 KV 的来源和输出的掩码方式变了。
- **第 22～24 天**：多级 stage 与 `asc_sync_wait/notify` 配对——sparse Kernel 里 AIV 与 AIC 通过 `set/wait_intra_block` 跨子块握手，读懂它们靠同一套事件思维。
- **第 32～33 天**：FP8/FP4 与 UE8M0 scale、`MX_SF_DIVISOR = 64`——sparse 输入仍是 MX 布局，每 token 的 SF 是 `kHeadDim/MX_SF_DIVISOR·sizeof(int16_t) = 2×2 = 4` 字节（`layout/sparse_mqa_logits.hpp:12`）。
- **第 16 天**：片上内存 offset 分配与容量 `static_assert`——今天用同样方法给 sparse Kernel 记 UB/L1 账（3.5 节）。
- **第 17 天**：调度与 work 划分——两个 metadata kernel 都在做"按工作量把 query 切给核"，与 `get_next_block()` 的思想同源，只是搬到了 device 上做。
- **第 5 天**：GM → L1 → L0 → MAD → L0C → Epilogue → GM 数据流——今天多了一条"GM →(AIV gather)→ UB →(NZ 转换)→ L1"的支线。

> 推荐先花 5 分钟默写第 39 天的 MQA Logits 流程图（调度 → Q/KV 搬运 → MAD+ReLU → reduce → 掩码存储）。另注：第五节 A 组任务需要 Ascend 环境（仓库 `README.md` 的 Requirements）；无环境时完成 B 组静态任务即可达到当天目标。今天的 Kernel 常量集中在 `deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.hpp`，建议把它打印在手边。

## 三、详细讲解

### 3.1 从"窗口"到"选块"：Sparse MQA 要算什么

普通 MQA 每行 query 计算连续窗口 `[ks, ke)` 内的**所有** KV token。Sparse MQA 把这个窗口按 `B = sparse_block_kv`（8 或 16）切成块，每个 query 只保留**自己选中的前 `min(capacity, ceil((ke−ks)/B))` 个块**（capacity = `sparse_kv_block_indices.size(1)`，即 `num_max_sparse_blocks`）。块 ID 列表就是 API 的核心输入 `sparse_kv_block_indices[Q, capacity]`：int32、每行有序、唯一、剩余位置是可重复的 padding（`csrc/apis/attention.hpp:29-37`）。数学写：

```text
第 i 个块覆盖 token：[i·B + ks%B, i·B + ks%B + B)     （ks%B 是窗口起点在块内的偏移）
logits[q, j]，j = 0..capacity·B−1：
    j 对应 q 选中的第 j/B 个块（块 ID = sparse_kv_block_indices[q, j/B]）的第 j%B 个 token
    值 = Σ_h relu(⟨q[q,h,:], kv[token(j), :]⟩) · w[q,h]
    超出有效块数或落在 [ks, ke) 之外的列是 padding，不参与比对
```

这与第 39 天的压缩输出一脉相承，但压缩的依据从"窗口前缀"换成了"选中块列表"。输出固定为 `logits[Q, capacity×B]` bf16、行距 512B 对齐（`get_logits_stride`，`csrc/apis/attention.hpp:23-27`）。有一个硬约束值得记住：**`capacity×B < 65536`**（"Empty KV splits also encode the end offset, so it must fit in 16 bits"，`csrc/jit_kernels/sparse_mqa_logits.hpp:23`）——所以 B=16 时 capacity 上限是 4095，B=8 时是 4096（测试 `test_sparse_mqa_capacity_boundaries` 专门压实这两组边界，`tests/test_attention.py:980-1005`）。

**手算例（今天反复用它）**：取 B=8，一个 QBlock 内 4 个 query（`kBlockQ = 4`，`layout/sparse_mqa_logits.hpp:10`），ks 都是 0：

```text
q0: ke=16 → 2 块，选中 {0, 1}
q1: ke=8  → 1 块，选中 {0}
q2: ke=24 → 3 块，选中 {0, 1, 2}   （为凑例子假设 capacity 足够）
q3: ke=32 → 4 块，选中 {0, 1, 2, 3}

则输出行宽 = capacity×8；q1 的第 0..7 列 = 块 0 的 8 个 token 的 logit；
q0 的第 0..7 列 = 块 0，第 8..15 列 = 块 1；以此类推。
```

为什么要"选块"？MoE/长上下文场景里每个 query 真正相关的 KV 只是稀疏子集（测试用 `overlap=0.8/0.9` 模拟相邻 query 保留 80%/90% 相同块，`tests/test_attention.py:633-664`）；只算选中的块，计算量从 `2·(ke−ks)·H·D` 降到 `2·selected·H·D`。而相邻 query 的选择高度重叠——这正是下一节"合并"要利用的。

### 3.2 两套 API 与元数据张量

今天有四个 Python 入口，分成"非分页（Prefill）"和"分页（Decode）"两套：

| 入口 | 位置 | 输入要点 | 返回 |
|---|---|---|---|
| `get_sparse_mqa_logits_metadata(ks, ke, num_kv_tokens, sparse_indices, dtype, B)` | `csrc/apis/attention.hpp:59-80` | ks/ke `[Q]` int32；KV token 总数；块 ID 表 | 元数据 int32 张量（1D） |
| `fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, capacity, B)` | 同上 `:180-226` | q `[Q,H,D]`（3D，H∈[4,32] 且 4\|H，D=128，FP4 打包为 64 字节）；kv `(kv[KV,D], sf[KV,2])` token-major 连续 | `logits[Q, capacity×B]` bf16 |
| `get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_indices, dtype, B)` | 同上 `:85-116`（Python 封装 `deep_gemm/__init__.py:87-112`） | context_lens `[Q]`；block_table `[Q, max_pages]`；indices `[Q]` 请求 ID；page_kv ∈ {64,128}；要求 `capacity % (page_kv/B) == 0` | 元数据 uint8 张量（1D） |
| `fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata, capacity, B)` | 同上 `:121-175`（Python 封装 `__init__.py:115-135`） | q `[Q,1,H,D]`（4D）；kv_cache `[num_pages, page_kv, 1, token_bytes]` uint8 | 同上 |

两组共同契约（`check_sparse_mqa_q_and_weights`，`attention.hpp:39-54`）：q/kv 同为 FP8(e4m3) 或 packed FP4；`sf_q` 是 `[Q·H, 2]` int16 连续（每 int16 打包 2 个 UE8M0，每 token 4 字节 SF）；weights `[Q,H]` bf16 且 `stride(1)==1`（行距可带 padding）。**元数据张量不只是调度信息，它本身就是 Kernel 的输入之一**：主 Kernel 启动时把它整个指针传进去（`launch_sparse_mqa_logits`，`csrc/jit_kernels/sparse_mqa_logits.hpp:71-99`），Kernel 内部再按 offset 拆出各段。

两套布局逐段对照（相对元数据张量基址的字节偏移）：

| 段 | `SparseMQALayout`（非分页，`:13-39`） | `PagedSparseMQALayout`（分页，`csrc/jit_kernels/paged_sparse_mqa_logits.hpp:14-43`） |
|---|---|---|
| ① 描述头 | `MetadataHeader`（32B：capacity + 三个 offset，`layout/sparse_mqa_logits.hpp:31-35`） | 无运行时头（布局由 Host 与 Kernel 的模板参数约定） |
| ② 每 query 信息 | `QBlockHeader[Q]`（`num_merged_kv_blocks`、`num_q_block_queries`，`:38-41`） | `mapping[Q] × (16384/B)` uint16：每 query 每个选中块在桶内的最终位置 |
| ③ 物理块表 | `physical_kv_token_starts[Q × capacity]`：并集块的 token 起点 | `physical_kv_token_starts[Q × (capacity+32) + 核数×128]`：桶内物理 token 起点，`0xffffffff` 为 padding |
| ④ 切分计划 | `QBlockSplitKVOutputInfo[Q × ceil(capacity/64)]` + `ScheduleBoundary[核数+1]` | `SplitKVRange[2×核数]` + `SplitKV` 记录数组（每核两段） |
| ⑤ 物化缓存 | 无 | `compact`：选中 KV 的紧凑副本（页式，每页 64 token），**只在分页版存在** |

第 ⑤ 段是理解 FIRST/REUSE 的钥匙：分页版把第一次读到的 KV **复制一份**存在 metadata 里，之后的调用不再碰原始 paged cache。容量上还有一个 Host 断言：分页版要求 `num_ai_cores ∈ [1,32]`（`paged_sparse_mqa_logits.hpp:26`），且 `capacity ≤ 16384/B`（`:27`）。

### 3.3 Paged KV 寻址与 fused 页布局

分页版的 KV cache 不是连续矩阵，而是 `[num_pages, page_kv, 1, token_bytes]` 的 uint8 张量：**每页先存满 `page_kv` 个 token 的 DATA，再存满 `page_kv` 个 token 的 SF**（"MQA pages store [page DATA | page SF]"，`attention.hpp:140`；测试的构造见 `make_random_paged_mqa_kv`，`tests/test_attention.py:667-683`）。`token_bytes = data_bytes + 4`（FP8 为 128+4，FP4 为 64+4）。页距 `page_stride` ≥ `page_kv × token_bytes`（Kernel 只要求 4 字节对齐，`paged_sparse_mqa_logits.hpp:59`；测试里取 512 对齐）。

**逻辑块 → 物理 token 起点**的换算只在一处出现，值得抄下来（`paged_sparse_mqa_logits_metadata.hpp:87` 与 `:166`）：

```text
每页含 page_kv/B 个稀疏块（page_kv=64,B=8 → 8 块/页；page_kv=128,B=8 → 16 块/页）
logical_page = block_idx / (page_kv / B)
in_page      = block_idx % (page_kv / B)
physical_page = block_table[query 行, logical_page]        // 每请求一张页表，行按 indices 展开
token_start   = physical_page * page_kv + in_page * B      // 物理空间里的 token 下标
```

注意 `block_table` 的行数等于 **query 数**（测试里先按请求建表再 `repeat_interleave` 到 query 行，`tests/test_attention.py:786-792`），同请求的 query 必须对引用页达成一致（`__init__.py:99-102` 的契约）。拿到物理 token 后，读它只需一个字节偏移（`paged_sparse_mqa_logits.hpp:60-62`）：

```text
byte_offset(token, token_bytes) = (token / page_kv) * page_stride + (token % page_kv) * token_bytes
```

gather 时 Kernel 尽量把**相邻两个块**合并成一次 `copy_gm_to_ub`（2 个 burst、步长 = 两块物理距离，`:76-90`）；距离超限或跨越 padding 才退化为两次单块搬运。FP4 + B=8 且页距 8 字节对齐时还有一个 SIMT 特殊路径 `paged_sparse_mqa_gather_sf`（`:12-40`），用 `__ldg` 逐行取 SF，避开 MTE2 的对齐限制。另外物理页顺序是乱的（测试用 `randperm` 打乱，还可选 reverse/pair_swap，`tests/test_attention.py:310-318`）——**寻址正确性完全不依赖页的物理顺序**，这也是 `bucket_q_block_interval` 里"对相邻条目排序"的原因（见 3.6 节）。

### 3.4 非分页元数据 Kernel：合并、切分与调度

`sparse_mqa_logits_metadata`（`__global__ __mix__(0, 1)`，纯 vector 核，`sparse_mqa_logits_metadata.hpp:304-371`）在 JIT 里按 `sparse_block_kv、num_metadata_cores、num_ai_cores、num_schedule_cores` 实例化（`csrc/jit_kernels/sparse_mqa_logits.hpp:52-60`），每个核负责一段 query（`:330-331`），每 `kBlockQ = 4` 个 query 一个 QBlock，四步流水：

**第一步：`sparse_mqa_logits_stage_indices`（`:116-143`）。** 把每行的块 ID 换算成 token 起点：`value = value·B + ks%B`（`:138-139`），块数截断到 `min(capacity, ceil((end−start)/B))`（`:127-128`），并记录 `token_end`。空窗口（`start ≥ end`）得到 0 块。

**第二步：`sparse_mqa_merge`（`:23-66`）。** 4 个 query 各有一张**有序**块列表，做 4 路归并得到**排序并集**。核心是 `utils/simt.hpp` 里的两个原语：`sorted_union_partition`（`:20-58`，把并集按线程均分，每个输入用倍增搜索定位起点）和 `sorted_union_advance`（`:61-81`，取各列表当前最小 key，返回命中该 key 的 `query_mask`）。每个并集块写两笔：`physical_kv_token_starts[...] = token_start` 和 `ub_query_masks[...] = query_mask`（`:50-51`）。**query_mask 是 4 位掩码，第 i 位 = query i 是否选中该块**——它是后续所有"按 query 过滤"的源头。归并结束后 `q_block_header->num_merged_kv_blocks` 记录并集大小。

用 3.1 节的手算例走一遍（ks=0，4 个 query 的块列表如前）：

```text
并集块：  0        1        2        3
query_mask: q0,q1,q2,q3 → 0b1111   q0,q2,q3 → 0b1101   q2,q3 → 0b1100   q3 → 0b1000
num_merged_kv_blocks = 4
```

**第三步：`sparse_mqa_kv_split_outputs`（`:68-114`）。** 把并集按 `kBlocksPerSplitKV = kSplitKV/B = 512/8 = 64` 块切成 KV split（非分页 `kSplitKV = get_split_kv(false) = 512`，`layout/sparse_mqa_logits.hpp:16-18, :28`）。每个 split 为每个 query 写两样东西（存进 `QBlockSplitKVOutputInfo`，`:43-50`）：

- `masks[q].lo/hi`：两个 uint32 位掩码，第 b 位 = 并集中第 `split·64 + b`（lo 为 b<32，hi 为 b≥32）个块**是否被 query q 选中**——注意是**每块一位**，不是每 token 一位，AIV 的 SIMD reduce 会把一位展开成 B 列（`simd/sparse_mqa_logits.hpp:182-189`）；
- `output_ranges[q]`：低 16 位 = 该 split 在 q 的压缩输出行里的**起始列**，高 16 位 = **有效列数**。公式：`output_begin = (累计选中块数 − 本 split 选中数)·B`，`num_tokens = 选中数·B − (最后一个 split 时 B − num_tail_tokens)`（`:106-108`）。

手算例（全部 4 块落进 split 0，`block_end = 各 query 的累计选中数`）：

| query | masks.lo（块位） | output_begin | num_tokens |
|---|---|---|---|
| q0 | `0b0011`（块 0,1） | (2−2)·8 = 0 | 16 |
| q1 | `0b0001`（块 0） | (1−1)·8 = 0 | 8 |
| q2 | `0b0111`（块 0,1,2） | (3−3)·8 = 0 | 24 |
| q3 | `0b1111`（块 0..3） | (4−4)·8 = 0 | 32 |

每个 query 的输出都从第 0 列开始——压缩是**按 query 各自进行**的，不同 query 选中的块在并集里的位置互不影响自己的列号。

**第四步：`sparse_mqa_logits_schedule_vf`（`:191-302`）。** 给每个 query 算工作量（`:213`）：

```text
work(q) = num_q_block_queries + ceil(num_merged_kv_blocks · B / 128) · (1 + ceil(num_q_block_queries / 2))
```

（第二项的 128 是 MAD 的 N 对齐粒度，`1 + ceil(q/2)` 是每个 split 的"一次 KV 装载 + 每 2 query 一拍 FIX"）。多线程用 warp 前缀和累计，再对每个核的目标 work 二分搜索，把 `ScheduleBoundary{query_begin, kv_split_begin}` 写进第 ④ 段（`:276-289`）。核与核的边界可以落在一个 QBlock 的**中间**（从某个 kv_split 开始），`SparseMQALogitsScheduler::peek` 用 `next_query_idx == end_query_idx` 特判收尾段（`scheduler/sparse_mqa_logits.hpp:23-32`）。手算例的 work = 4 + ceil(4·8/128)·(1+2) = 4+3 = **7**。

### 3.5 非分页主 Kernel：AIV gather + AIC MAD

`sparse_mqa_logits_impl<kIsFP4, kNumHeads, kSparseBlockKV>`（`__mix__(1 AIC + 2 AIV)`，`sparse_mqa_logits.hpp:97-414`）与第 39 天最大的结构差异是：**KV 不再由 AIC 直接从 GM 搬，而是 AIV 先"收集"进 L1，AIC 再从 L1 取**。两个分支用同一个 `SparseMQALogitsScheduler` 遍历同样的任务序列（`:157`），AIV 跑在前面发布，AIC 在后面消费。

**AIC 分支（`:160-239`）**：`asc_set_mmad_direction_n()` 后，把 Q（2 个 MMA，`kNumMMAsPerQBlock = kBlockQ/kNumQueriesPerMMA = 2`）连同 MX scale 装进 L1/L0A（`:179-202`，`kHeadPitch = aligned(H,16)`，H 不是 16 倍数时逐 query 补零）；然后按 kv_split 循环：`wait_intra_block(PIPE_MTE1, 0/1, 3)` 等 AIV 发布的 L1 stage → `copy_l1_to_l0b` + `copy_l1_to_l0b_mx` → `mad(l0c_tile, l0a[mma], l0b_tile, issued_m, issued_n, kHeadDim, ENABLE_UPDATE, true)` → `copy_l0c_to_ub(ub_score, ..., F322BF16, relu=true)`（`:213-231`，ReLU 仍在 FIX 管，同第 39 天）。`issued_n = aligned(split 块数·B, 128)`（`:207-208`），MAD 需要的 N 对齐产生的垃圾列稍后由 masks 屏蔽。

**AIV 分支（`:240-413`）**：三件事。

1. **gather（`sparse_mqa_logits_gather_kv_aiv`，`:51-95`）**：按 `physical_kv_token_starts` 从 GM 收集本 split 的 KV。两个 AIV 各负责半个 split（`gather_sub_idx = aiv_idx ^ ((query_begin/kBlockQ) & 1)`，`:275`，按 QBlock 奇偶翻转以均衡 UB bank）；每半 `kGatherRows = kSplitKV/2 = 256` 行。搬运尽量成对：相邻两块一次 `copy_gm_to_ub(2, block_bytes, next−first)`（`:78-86`）；不完整的尾块（"Sorted union blocks place all incomplete blocks at the end"，`:62-65`）只搬 `valid = min(B, num_kv_tokens − first)` 行（`:87-94`），padding 列留给掩码处理。
2. **NZ 转换 + 发布**：`vf_sparse_mqa_logits_convert_kv` 把 token-major 数据转成 L0B 需要的 NZ 分形布局（"data_dst[f, t, r] = data_src[t, f, r]"，`simd/sparse_mqa_logits.hpp:217-220`），然后 `copy_ubuf_to_cbuf` 把 DATA 和 SF 写进 `l1kv/l1_sf_kv` 的对应半区（`:330-335`），`set_intra_block(PIPE_MTE3, aiv_idx, 3)` 通知 AIC。
3. **reduce + 存储（`sparse_mqa_logits_reduce_select`，`:24-48`）**：等 AIC 交付的 `ub_score`，用 `masks.lo/hi` 逐块选列、乘 weights 跨头求和，按 `output_ranges` 把有效列**紧凑地**写进 `ub_selected_logits`，最后 `copy_ub_to_gm(logits.index(q, output_begin), ..., num_output_tokens)`（`:399-401`）。FP4 且有第二个 gather stage 时走融合变体（`kConvert=true`，`:371-382`）：reduce 与下一批 KV 的 NZ 转换重叠。

**UB 账本（H=8, B=8, FP8 实例）**：`kHeadPitch=16`、`kMMAHeads=32`、`kNumVecElems=128`（256B 向量 / 2B）。`ub_score` 每 stage 32×512×2 = 32KB；`kGatherStageBytes = 256×(128+4) + 257×128 = 66688B`；`kFixedUBBytes = (2×16 + 3×2×512)×2 + 256 = 6464B`。于是（`:131-133`）：2 个 score stage + 固定开销 + 2 个 gather stage = 65536+6464+133376 = 205376 ≤ 216KB → `kNumGatherStages = 2`；3 个 score stage 则 238144 > 221184 → `kNumScoreStages = 2`。L1 侧 `l1kv` 4 stage × 512×128 = 256KB，加 Q 与 SF 约 279KB ≤ 512KB（`static_assert`，`:152`）。这张账解释了为什么 gather 优先拿到双缓冲、score 只能两级——gather 是 AIV 关键路径。

### 3.6 分页 prepare Kernel：代价模型、分桶与 SplitKV 计划

`paged_sparse_mqa_prepare_impl`（`__mix__(0, 1)`，`paged_sparse_mqa_logits_metadata.hpp:522-547`）一次 launch 启动 **2×核数** 个 AIV block（`num_ranges = 2 * num_ai_cores`，`csrc/jit_kernels/paged_sparse_mqa_logits.hpp:51`），每个核的两个 AIV 最终各写一段 `SplitKVRange`。流程五步：

**① `plan_range`（`:21-66`）**：粗估每个 QBlock 的代价 `C = (max_blocks + Σblocks)·B + 4096·num_q_block_queries`（`:57`——"并集估计 + 选中总数 + 启动开销"），顺带把 `QBlockStats{num_q_block_queries, cost, packed_block_counts}` 写进 scratch，每个 prepare 段累计 `range_costs`。

**② `find_prepare_range`（`:360-394`）**：对 2×核数 个段代价做前缀和，每个核二分定位自己的目标代价区间；`align_q_block_begin`（`:396-406`）把切割点对齐到"同请求、QBlock 边界"，保证一个请求的页表契约不被拆散。

**③ `build_aiv_interval`（`:409-520`）**：核内两个 AIV 按**选中块数**再平衡（`:467-471`）；跨界 QBlock 用 `sorted_union_partition` 按代价比例切割（`:324-330`）。每个归属 QBlock 先把 4 行块 ID 装进 UB（`:481-485`），再进 ④⑤。

**④ `bucket_q_block_interval`（`:68-207`）**：今天最精巧的一段。4 个 query 的并集块按 `query_mask` 分进 **16 个桶**（`kNumBuckets = 1<<kBlockQ`，`layout/sparse_mqa_logits.hpp:70`）。每个桶对齐到 `kBlocksPerPage = kCompactPageTokens/B`（64/8 = 8）块，不足补 `0xffffffff`（`:144-154, :171-173`）。每块换算物理 token 起点（3.3 节公式，`:166`）后写进 `physical_kv_token_starts` 的桶基址处；同时生成 `mapping`：**每 query 每个选中块在桶内的最终位置**（uint16，`:176-195`）。桶内还会对相邻条目按物理地址排序（`:186-187` 的 `swap_pair`），让 gather 的成对搬运拿到连续地址。单 query 的 QBlock 走简化分支（`:82-96`）。

**⑤ `emit_bucket_kv_splits`（`:209-301`）**：每桶按 64 块切 KV split，生成 `SplitKV` 记录（`layout/sparse_mqa_logits.hpp:100-113`）。`control` 是一个 32 位打包：

| 位段 | 字段 | 含义 |
|---|---|---|
| [3:0] | `bucket_idx` | 本 split 属于哪个 query_mask 桶 |
| [10:4] | `num_blocks` | 本 split 的有效块数（7 位） |
| [13:11] | `num_q_block_queries` | QBlock 内实际 query 数 |
| [14] | `kQBlockBegin` | QBlock 的第一个 split（触发 Q 装载/输出起点） |
| [15] | `kQBlockEnd` | QBlock 的最后一个 split（触发输出写回） |

其余字段：`compact_page_idx`（本 split 在 compact 区的起始页）、`query_begin`、`num_q_block_kv_splits`、`map_begins[2]`（两个 query pair 各自的 mapping 行基址）、`output_ranges[4]`（低 16 位 token 起点、高 16 位数量，`:284-286`）。**排序规则**：被两个 AIV 共享的桶（query_mask 覆盖两个 pair）排在最前，之后两个 AIV 各自专属的桶交错（"Shared KV splits first, then interleave AIV0-only and AIV1-only KV splits"，`:275-277`）——共享数据只算一次，专属数据让两个 AIV 都有活干。

### 3.7 分页主 Kernel 与 FIRST/REUSE 快照

`paged_sparse_mqa_logits_impl<kFirst, kIsFP4, kNumHeads, kSparseBlockKV, kPageStride, kPageKV>`（`__mix__(1, 2)`，`paged_sparse_mqa_logits.hpp:147-507`）启动时先读本核的两个 `SplitKVRange` 拼成一条 kv_split 序列（`:190-197`），然后 AIC/AIV 以 pair 为粒度协同：每个 SplitKV 至多覆盖两个 query pair（`query_mask(control, pair_idx)` 取 2 位），`load_kv` 在 pair 0 或 pair 0 为空时装载、`release_kv` 在 pair 1 或 pair 1 为空时释放（`:268-269, :434-435`），`issued_n` 对齐到 128（`:270`）。**FIRST 与 REUSE 的分水岭就在 KV 从哪来**：

- **FIRST（kFirst=true）**：AIV 对每个 split 执行 `paged_sparse_mqa_prefetch_kv`（`:43-105`）——按 3.3 节的寻址从**原始分页 GM** gather DATA/SF 进 UB 的 RAW 缓冲（两个 128/256 行的 chunk 轮转）；再执行 `paged_sparse_mqa_publish_kv`（`:108-145`）：`vf_sparse_mqa_logits_convert_kv` 转 NZ 后，把结果**双写**——一份进本核 L1 的 `l1kv/l1_sf_kv` stage 供 AIC 立即使用，一份写进 metadata 的 compact 区（`asc_copy_ub2gm_align`，`:133-138`）。AIC 的 `prefetch_data` 分支为空（`if constexpr (!kFirst)`，`:210`），Q 在 `kQBlockBegin` 装载（`:275-283`）。
- **REUSE（kFirst=false）**：AIV 不再碰原始 KV；AIC 的 `prefetch_data` 用 `asc_copy_gm2l1_align` **直接把 compact 页 GM→L1**（`:221-223`，L2 模式 `NOTALLOC_KEEP`），Q 双缓冲预取（`next_l1q`，`:286`）。JIT 实例化时页参数直接换成 compact 的：`page_stride = kCompactPageTokens × token_bytes`、`page_kv = kCompactPageTokens(64)`（`csrc/jit_kernels/paged_sparse_mqa_logits.hpp:83-85`）——对 Kernel 来说，compact 区就是一张"页距规整、无碎片"的新页表。

AIC 的计算主干两条路相同：L0A 用 `pair/single` 两套 tile 适配 `query_mask`（2 query 用 pair、单 query 用 single，`:290-299, :337-339`），REUSE 走 `asc_copy_l12l0b` 只搬有效行（"masked MAD tail rows need no KV values"，`:329`），`mad` 后 `copy_l0c_to_ub(F322BF16, relu)` 并用 `set_intra_block(PIPE_FIX, pair_idx, stage)` 把分数 tile 移交**该 pair 归属的 AIV**（`:343-348`）。AIV 侧 `vf_sparse_mqa_logits_reduce_scatter` 用 `mapping` 把选中块的 logit **散射回该 query 自己的输出位置**（`simd/sparse_mqa_logits.hpp:206-215`），到 `kQBlockEnd` 时按 `output_ranges` 的 begin/count 用 `copy_ub_to_gm` 写回（`:480-495`）。

**状态机在 Python 侧**（`deep_gemm/__init__.py:87-135`）：prepare 返回时挂上 `metadata._paged_sparse_mqa_state = (config, True)`；`fp8_fp4_paged_sparse_mqa_logits` 每次校验 config（Q 数、capacity、B、page_kv、核数、dtype 任一不匹配即 `RuntimeError`），首次真实调用后翻转为 `(config, False)`。Host 注释说得很直白："FIRST caches KV in metadata; REUSE reads this snapshot, not the current raw KV. Regenerate metadata after changing KV contents, even if the sparse indices are unchanged"（`attention.hpp:119-120`）。测试把它压实成三条断言（`tests/test_attention.py:953-966`）：

```text
① FIRST 后把原始 KV 清零、weights 翻倍 → REUSE 结果 = 2×FIRST 结果（KV 来自快照，weights 现读）
② 重建 metadata → 结果 = 0（新快照读到了清零后的 KV）
③ 改 Q/核数等 config → RuntimeError
```

### 3.8 Prefill 与 Decode 的主要差异

把第 39 天和今天的内容放在一起，Prefill 与 Decode 两条生产路径的差异如下：

| 维度 | Prefill（非分页 sparse） | Decode（分页 sparse） |
|---|---|---|
| 有效区间 | 每行 `[ks, ke)` 窗口（`cu_seq_len_k_start/end`） | 每请求 `context_lens`，同请求 query 上下文递增（测试 `:716-718`） |
| 组织单位 | QBlock（4 query）独立 | 请求（`indices` 标识），QBlock 不跨请求切割（`align_q_block_begin`） |
| KV 存储 | token-major 连续 `(kv[KV,D], sf[KV,2])` | paged cache：fused 页 `[DATA\|SF]` + `block_table` 间接寻址 |
| Q 形状 | `[Q, H, D]` | `[Q, 1, H, D]` |
| `kSplitKV` | 512（`get_split_kv(false)`） | FP8 256 / FP4 512（`get_split_kv(true, is_fp4)`，`layout/sparse_mqa_logits.hpp:16-18`） |
| KV 进 L1 的路径 | AIV 从 GM gather → NZ 转换 → UB→L1 | FIRST：AIV 经页表 gather → NZ → compact(GM)+L1；REUSE：AIC 直接 compact GM→L1 |
| 调度信息 | `ScheduleBoundary`（query/kv_split 起点）+ 运行时 `SparseMQALogitsScheduler` | `SplitKVRange` × 2/核 + 静态 `SplitKV` 记录（无运行时调度器） |
| 跨调用复用 | 无（每次重算） | FIRST/REUSE 快照；多层解码 = prepare + FIRST + (L−1)×REUSE |

生产用例的参数也印证这个分工：Prefill 用例是 8192 个 query、两个请求（3157/5035 token）、KV 64K～512K、FP4、overlap 0.8/0.9（`test_sparse_mqa_logits_perf`，`:892-900`）；Decode 用例是 256 个请求、每请求 3~7 个 query、KV 64K、capacity 2048、B=8、page_kv=128、**4 层**（`run_paged_sparse_mqa_logits_perf_case`，`:451-587`）——R4 链测的就是"一次 prepare + 一次 FIRST + 三次 REUSE"的总时延（`:562-576`）。

### 3.9 测试如何验证

`tests/test_attention.py` 对今天的覆盖有五层，验证思路各不相同：

1. **与全量 Kernel 逐位对账**：同一输入同时跑 `fp8_fp4_sparse_mqa_logits`（或 paged 版）和第 39 天的全量 `fp8_fp4_mqa_logits`（或 `fp8_fp4_paged_mqa_logits`），把 sparse 输出按 `token_indices = sparse_indices·B + ks%B + offset` 散射回去，断言 `torch.equal`（分页或 FP8 时**逐位相等**，":860-861"；非分页 FP4 因 reduce 顺序差异放宽到 `diff < 0.001`，":862-864"）。这比对比参考实现更强：sparse 路径必须复现全量路径的每一位。
2. **快照语义**：3.7 节的三条断言 + `run_paged_sparse_mqa_logits_perf_case` 里的 `stale_logits/fresh_logits` 检查（`:537-545`），以及 `check_snapshot` 模式——把输入全部填 -1 后 `dry_run` 一遍再真实运行，验证 FIRST 不读已失效的调度输入（`:828-842`）。
3. **元数据契约**：`test_paged_sparse_mqa_metadata_contract`（`:903-977`）用 3 个 query 的小例子检查错误路径（容量/B/页数/Q 行数不匹配、核数改变都应报错）、小容量不再保留全尺寸 KV 存储，以及 **4GiB 以上的分配算术**（用 meta 张量 dry-run，`:968-977`）。
4. **边界**：`test_sparse_mqa_unaligned_ks`（ks 不对齐 B，`:1008-1045`）、`test_sparse_mqa_capacity_boundaries`（B=8/capacity=4096 与 B=16/capacity=4095，`:980-1005`）、`test_sparse_mqa_logits_boundaries`（小 Q、空上下文、split 尾巴，`:1145-1174`）。
5. **性能口径**：sparse 的 flops 只数选中 token（`2·Σ选中块·B·H·D`，`:878-886`）；paged sparse 另打印 `prepare / first-main / reuse-main / R4` 四个时延（`:578-587`）——分别对应元数据构建、快照建立、快照复用和四层链路。

## 四、源码导读

今天读码的目标是"把 3.4～3.7 节的流程钉到具体行号上"。建议按表顺序读：

| # | 文件 | 关键函数/对象 | 核对什么 |
|---|---|---|---|
| 1 | `deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.hpp` | 公共常量 `:8-18`（`kBlockQ=4`、`kSparseUBBytes=216KB`、`get_split_kv`）；sparse 结构 `:25-63`（`MetadataHeader`/`QBlockHeader`/`QBlockSplitKVOutputInfo`/`ScheduleBoundary`）；paged 结构 `:66-117`（`kMaxOutputTokens=16384`、`kNumBuckets=16`、`kCompactPageTokens=64`、`SplitKV` 位域 `:100-113`） | 3.2 节两张布局表 |
| 2 | `csrc/apis/attention.hpp` | `validate_sparse_indices` `:29-37`、`check_sparse_mqa_q_and_weights` `:39-54`、`get_sparse_mqa_logits_metadata` `:59-80`、`get_paged_sparse_mqa_logits_metadata` `:85-116`、`fp8_fp4_paged_sparse_mqa_logits` `:121-175`、`fp8_fp4_sparse_mqa_logits` `:180-226`、绑定 `:414-436` | 3.2 节契约表与 FIRST/REUSE 注释 |
| 3 | `deep_gemm/__init__.py` | `get_paged_sparse_mqa_logits_metadata` `:87-112`、`fp8_fp4_paged_sparse_mqa_logits` `:115-135` | 3.7 节状态机 |
| 4 | `csrc/jit_kernels/sparse_mqa_logits.hpp` | `SparseMQALayout` `:13-39`（16 位断言 `:23`）、`launch_sparse_mqa_logits_metadata` `:41-69`（核数选择 `:49-51`）、`launch_sparse_mqa_logits` `:71-99` | 元数据四段与 JIT 实参 |
| 5 | `csrc/jit_kernels/paged_sparse_mqa_logits.hpp` | `PagedSparseMQALayout` `:14-43`、`launch_paged_sparse_mqa_metadata` `:45-69`（`num_blocks=2×核数` `:65`）、`launch_paged_sparse_mqa_logits` `:71-94`（FIRST/REUSE 页参数 `:83-85`） | 3.2/3.7 节 |
| 6 | `deep_gemm/include/deep_gemm/utils/simt.hpp` | `sorted_union_partition` `:20-58`、`sorted_union_advance` `:61-81` | 多路有序归并原语 |
| 7 | `deep_gemm/include/deep_gemm/sparse_mqa_logits_metadata.hpp` | `sparse_mqa_logits_stage_indices` `:116-143`、`sparse_mqa_merge` `:23-66`、`sparse_mqa_kv_split_outputs` `:68-114`、`sparse_mqa_logits_schedule_vf` `:191-302`、入口 `:304-371` | 3.4 节四步与手算例 |
| 8 | `deep_gemm/include/deep_gemm/scheduler/sparse_mqa_logits.hpp` | `SparseMQALogitsTask` `:8-11`、`SparseMQALogitsScheduler` `:13-45`（`peek` `:23-32`） | 边界落在 QBlock 中间时的收尾 |
| 9 | `deep_gemm/include/deep_gemm/sparse_mqa_logits.hpp` | `get_num_kv_split_blocks` `:16-22`、`sparse_mqa_logits_gather_kv_aiv` `:51-95`、`sparse_mqa_logits_reduce_select` `:24-48`、`impl` `:97-414`（UB 推导 `:128-133`、AIC `:160-239`、AIV `:240-413`） | 3.5 节账本与数据流 |
| 10 | `deep_gemm/include/deep_gemm/paged_sparse_mqa_logits_metadata.hpp` | `plan_range` `:21-66`、`find_prepare_range` `:360-394`、`build_aiv_interval` `:409-520`、`bucket_q_block_interval` `:68-207`、`emit_bucket_kv_splits` `:209-301`、入口 `:522-547` | 3.6 节五步 |
| 11 | `deep_gemm/include/deep_gemm/paged_sparse_mqa_logits.hpp` | `paged_sparse_mqa_gather_sf` `:12-40`、`paged_sparse_mqa_prefetch_kv` `:43-105`、`paged_sparse_mqa_publish_kv` `:108-145`、`impl` `:147-507`（ranges `:190-197`、AIC `:198-361`、AIV `:362-506`） | 3.3/3.7 节 |
| 12 | `deep_gemm/include/deep_gemm/simd/sparse_mqa_logits.hpp` | `vf_sparse_mqa_logits_reduce_select` `:196-204`、`vf_sparse_mqa_logits_reduce_scatter` `:206-215`、`vf_sparse_mqa_logits_convert_kv` `:217-240`（NZ 转置注释 `:217-220`） | 选块/散射与 NZ 转换 |
| 13 | `deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.hpp`、`scheduler/paged_mqa_logits_metadata.hpp` | `PagedMQALogitsScheduler` `:33-138`（`get_physical_page` `:119-123`）、`paged_mqa_request_cost` `:20-31`、`paged_mqa_logits_metadata_simt_impl` `:155-177` | 属第 39 天的非稀疏 paged MQA，作 Decode 寻址参照 |
| 14 | `tests/test_attention.py` | `make_sparse_kv_block_indices` `:633-664`、`run_sparse_mqa_logits_case` `:686-889`、`run_paged_sparse_mqa_logits_perf_case` `:451-587`、`test_paged_sparse_mqa_metadata_contract` `:903-977`、main `:1218-1301` | 3.9 节五层验证 |

**逐段导读一：`sparse_mqa_logits_gather_kv_aiv` 的成对搬运**（`sparse_mqa_logits.hpp:76-94`）。`num_full_blocks` 先从尾部剥掉不完整块（`:62-65`——并集排序保证不完整的只会在末尾）；主循环 `block_idx += 2` 每次取相邻两块的物理起点 `first/next`，`copy_gm_to_ub(dst, src, 2, kBlockDataBytes, (next−first)·kDataBytes)` 把两块拼成一次双 burst 的 MTE2 搬运，SF 同理（FP8 时 DATA/SF 分开两笔，FP4 时 SF 已提前搬完）。注意 stride 是 `next−first` 个 **token** 的字节数，说明两块在 GM 里天然连续时 stride 恰好等于块大小——这正是 prepare 阶段"桶内按物理地址排序"想制造的局面。尾循环逐块搬 `valid = min(B, num_kv_tokens − first)` 行，行数不足的列不初始化，靠 masks 屏蔽。

**逐段导读二：`emit_bucket_kv_splits` 的排序与 control 组装**（`paged_sparse_mqa_logits_metadata.hpp:246-301`）。三个 warp 前缀和分别累计"双 pair 桶""仅 AIV0 桶""仅 AIV1 桶"的 split 数（`:246-248`），由此算出每个桶的 `bucket_split_begin`；`q_block_kv_split_idx` 的表达式（`:277`）实现"共享在前、专属交错"的全局排序。control 组装（`:280-281`）就是 3.6 节位域表的直接翻译；`map_begins` 的两个 16 位半区分别给 pair 内第一/第二个 query 的 mapping 行基址（`:287-299`），`query_mask==2`（只有第二个 query）时第一基址指向它、第二基址清零。

**逐段导读三：FIRST 的 publish 与 REUSE 的 prefetch**（`paged_sparse_mqa_logits.hpp:108-145` 与 `:209-226`）。`publish_kv` 对每个 chunk：等 RAW 数据落 UB（MTE2→V 事件）→ `vf_sparse_mqa_logits_convert_kv` 原地把 token-major 转 NZ → `asc_copy_ub2gm_align` 把 DATA 按 `n×32B` 行距、SF 按 `rows×4B` 写进 compact 页（`:133-138`，stride 参数保证多核写不同 QBlock 时互不越界）→ `asc_copy_ub2l1` 把同一份结果送进本核 L1 stage（`:139-142`，行数对齐到 128）。REUSE 的 `prefetch_data` 则只有四行：设 loop size/stride 全零，`asc_copy_gm2l1_align(l1kv + stage·kKVStageBytes, compact + page·kPageBytes, 1, n·kTokenBytes, ...)` 一笔整页搬进 L1（`:218-223`）。对比这两段就能体会"快照"的本质：**FIRST 花力气把碎片化的页表访问整理成规整页，REUSE 只为整页线性搬运付费**。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通 Prefill sparse 生产用例**：

   ```bash
   cd DeepGEMM-Ascend/tests
   python test_attention.py sparse --skip-prof
   ```

   预期输出 `Testing sparse MQA logits production cases:` 后 4 行 OK（overlap {0.8,0.9} × KV {64K,512K}）。核对 `test_sparse_mqa_logits_perf`（`:892-900`）的参数：H=32、FP4、B=8、capacity=2048、请求 (3157,5035)——对照 3.8 节的 Prefill 列。

2. **跑通 Paged sparse 生产用例并观察快照指标**：

   ```bash
   python test_attention.py paged_sparse --skip-prof     # 先验证正确性
   python test_attention.py paged_sparse                  # 去掉 --skip-prof
   ```

   无 `--skip-prof` 时会打印 `prepare / first-main / reuse-main / R4` 四个时延与 TFLOPS（`:578-587`）。记录四个数并回答：reuse-main 与 first-main 的差值来自哪些被省掉的工作？R4 为什么不等于 4×reuse-main？

3. **跑边界与契约测试**：

   ```bash
   python test_attention.py sparse_boundary
   python test_attention.py paged_sparse_boundary
   ```

   先读 `test_sparse_mqa_logits_boundaries` 的用例表（`:1149-1160`，元组为 `(heads, fp4, paged, nq, kv, block, capacity, unaligned)`），预测每个考察点压实哪条路径（如 `(12, True, False, 9, 640, 8, 128, True)` = 非 16 倍头数 + 非对齐 ks + 跨 split 容量），再与通过结果对账；`paged_sparse_boundary` 还会执行 3.7 节的三条快照断言（`:953-966`）。

4. **CUDA 对齐子集**（可选，用例很多时用它抽样）：

   ```bash
   DG_MQA_NUM_CASES=8 python test_attention.py sparse_cuda --skip-prof
   DG_MQA_NUM_CASES=8 python test_attention.py paged_sparse_cuda --skip-prof --formats fp8
   ```

   对照 `enumerate_cuda_sparse_mqa_logits`（`:598-630`）里 B16 capacity=4095 的注释（"capped at 4095 by Ascend's 16-bit output offsets"），验证 3.1 节的上限推导。

5. **复现手算例的可执行版**（选做）：仿照 `test_sparse_mqa_metadata_tensor`（`:1048-1070`）写一个 4 query、B=8 的小脚本，用 `torch.equal` 验证你手算的输出列语义。

### B 组：无 Ascend 环境（静态任务）

1. **调用链填表**：不看正文，把下表补全再对照 3.2/3.7 节：

   | 层 | 位置 | 输入 → 输出 |
   |---|---|---|
   | Python | `deep_gemm/__init__.py:87` | context_lens/block_table/indices/块表 → `_C._get_paged_sparse_mqa_logits_metadata` + 状态挂载 |
   | binding | `csrc/apis/attention.hpp:431-436` | 同名参数 → C++ 内部函数（下划线绑定） |
   | 检查/布局 | `csrc/apis/attention.hpp:85-116` | 张量 → `PagedSparseMQALayout`（四段 offset）+ scratch |
   | JIT prepare | `csrc/jit_kernels/paged_sparse_mqa_logits.hpp:45-69` | 模板实参 → `paged_sparse_mqa_prepare_impl` 实例 |
   | Python 主调用 | `deep_gemm/__init__.py:115` | 校验 config/状态 → `_C._fp8_fp4_paged_sparse_mqa_logits(..., is_first, ...)` |
   | JIT 主 kernel | 同文件 `:71-94` | `is_first` 决定页参数 → `paged_sparse_mqa_logits_impl<kFirst,...>` |

2. **完成 3.4 节手算例的调度部分**：设 3.1 节的 QBlock 之后还有第二个 QBlock（4 个 query、8 个并集块、`num_q_block_queries=4`）。算两个 QBlock 的 work、`kv_split_work`，若只有 2 个核，写出每个核分到的 `(query_begin, kv_split_begin)` 与 `ScheduleBoundary[3]`。

3. **Paged 寻址与 compact 页手算**：设 `page_kv=128, B=8`（16 块/页）、`block_table[0] = [7, 3, 5, ...]`。求 query 0 逻辑块 20 的物理 token 起点；再算 FP8 的 compact 单页字节数（64 token × (128+4)）。最后回答：`capacity` 必须满足什么整除关系，为什么？（提示：`capacity % (page_kv/B) == 0`，`attention.hpp:94`——桶按整页对齐。）

4. **`SplitKV::control` 编码练习**：把 `bucket_idx=5, num_blocks=32, num_q_block_queries=3, 是 QBlock 的第一个 split` 编码成 uint32（应得 `0x5A05`），并反过来解码 `control = 0x00009C0A` 的各字段（对照 `layout/sparse_mqa_logits.hpp:105-112` 的移位与掩码验算）。

5. **FIRST/REUSE 状态走查**：对照 `tests/test_attention.py:953-966` 的三条断言，画一个时序表（行：调用序号；列：metadata 状态、KV 来源、weights 来源、期望输出），并把 `dry_run` 分支（`__init__.py:130-134`、`attention.hpp:172-173`）在表里单独标一列，解释为什么 dry-run 不翻转状态。

6. **UB 账本换参数**：仿照 3.5 节，对 `H=16, B=16, FP8` 重算 `kBlocksPerSplitKV = 512/16 = 32`、`kGatherRows = 256`、`kGatherStageBytes = 256×132 + 257×128`，判断 `kNumGatherStages/kNumScoreStages` 是否变化，并指出 B=16 时 `masks` 每 split 只需覆盖 32 块——lo/hi 哪个还有用？

## 六、当天产出：Paged/Sparse MQA 元数据与数据流图

按计划，今天的产出是一张 **Paged/Sparse MQA 元数据与数据流图**。建议自己先画（两张子图：非分页 Prefill 一张、分页 Decode+快照一张，标出元数据各段与 AIC/AIV 的读写关系），再与下面的参考版对照补漏：

```text
┌─ 子图 A：非分页 Sparse（Prefill）──────────────────────────────────────────┐
│ Python: get_sparse_mqa_logits_metadata(ks, ke, num_kv, sparse_indices, …)  │
│   → SparseMQALayout: [MetadataHeader | QBlockHeader[Q] |                    │
│                       physical_kv_token_starts[Q×C] |                       │
│                       QBlockSplitKVOutputInfo | ScheduleBoundary[核+1]]     │
│ Device: sparse_mqa_logits_metadata   __mix__(0,1)，每核一段 query          │
│   每 QBlock: stage_indices(块ID→token起点)                                  │
│            → merge(4路 sorted union + query_mask)                           │
│            → kv_split_outputs(64块/split: masks.lo/hi + output_ranges)      │
│   schedule_vf: work = q + ceil(blocks·B/128)·(1+ceil(q/2)) → 边界           │
│                                                                              │
│ Python: fp8_fp4_sparse_mqa_logits((q,sf),(kv,sf),w,metadata,C,B)            │
│ Device: sparse_mqa_logits_impl   __mix__(1 AIC + 2 AIV)，共用 Scheduler     │
│   AIV(超前): gather KV(成对 copy_gm_to_ub, 尾块只搬 valid 行)                │
│             → convert NZ → copy_ubuf_to_cbuf → L1(l1kv 4级) ──┐             │
│   AIC(滞后): Q→L1→L0A(+MX sf)；每 split: 等 L1 → L0B(+MX sf) ←─┘             │
│             → mad(issued_m×issued_n×128) → copy_l0c_to_ub(ReLU) → ub_score  │
│   AIV: reduce_select(masks 选块·weights 跨头求和                             │
│         → output_ranges 压缩列) → copy_ub_to_gm → logits[Q, C·B]            │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 子图 B：分页 Sparse（Decode + FIRST/REUSE）───────────────────────────────┐
│ Python: get_paged_sparse_mqa_logits_metadata(context_lens, block_table,     │
│         indices, page_kv, sparse_indices, …)  → 挂 _paged_sparse_mqa_state  │
│   → PagedSparseMQALayout: [mapping(u16) | SplitKVRange×2核+SplitKV[] |      │
│                            physical_kv_token_starts | compact(物化KV)]      │
│ Device: paged_sparse_mqa_prepare_impl  __mix__(0,1)，2×核数 个 AIV          │
│   plan_range(代价) → find_prepare_range(按代价定位) → build_aiv_interval     │
│   → bucket_q_block_interval(16 桶·页对齐·block_table 寻址·mapping)           │
│   → emit_bucket_kv_splits(SplitKV: control 位域/compact 页/map_begins)       │
│                                                                              │
│ Python: fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, w, metadata, C, B)     │
│   首次 → FIRST(kFirst=true)，此后状态翻转 → REUSE(kFirst=false)              │
│ Device: paged_sparse_mqa_logits_impl  __mix__(1 AIC + 2 AIV)                │
│   FIRST-AIV: prefetch_kv(页表寻址 gather DATA/SF) → convert NZ              │
│              → publish: 写 compact(GM) + 拷 L1 stage                        │
│   REUSE-AIC: prefetch_data: compact GM → L1（页距=64·token_bytes）           │
│   AIC: Q 双缓冲；pair 循环(query_mask/load_kv/release_kv) → L0A pair/single │
│        → L0B → mad → copy_l0c_to_ub(ReLU) → FIX 移交 pair 归属 AIV          │
│   AIV: reduce_scatter(mapping 散射回原位)；kQBlockEnd 时                      │
│        按 output_ranges copy_ub_to_gm → logits[Q, C·B]                      │
│   注意: 改 KV 内容必须重建 metadata（REUSE 只认快照）                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

检查清单（自查你的图是否包含）：① 两套元数据各自的段名与所属（谁写、谁读）；② `query_mask` 从 merge 一路到 `masks.lo/hi`、`SplitKV::control` 的传递链；③ 逻辑块 → 物理 token 的寻址公式出现在哪一步；④ FIRST 与 REUSE 的 KV 来源分叉（AIV gather+publish vs AIC 整页 GM→L1）；⑤ compact 区在 metadata 内的位置与页参数；⑥ 输出 `[Q, C·B]` 的压缩列语义与 512B 行距；⑦ Prefill/Decode 差异表中的至少 5 行。

## 七、自测题

1. 写出第 q 个 query 第 i 个选中块覆盖的 token 区间。给定 `ks=5, B=8`、块 ID 为 3，token 起点是多少？这个换算发生在非分页元数据的哪一步？
2. sparse 输出为什么是 `[Q, capacity×B]` 而不是 `[Q, KV]`？`capacity×B < 65536` 的限制从哪来？B=16 时 capacity 上限是多少？
3. `query_mask` 是什么？它先后被编码进哪些数据结构，最后被谁消费？
4. `masks.lo/hi` 的第 b 位代表什么？`output_ranges` 的高低 16 位各是什么？为什么每个 query 的每个 split 的 `output_begin` 都从"该 query 自己的累计选中数"算起？
5. 分页寻址中 `token_start = block_table[b/(page_kv/B)]·page_kv + (b%(page_kv/B))·B`，其中 `block_table` 的行数为什么等于 query 数而不是请求数？fused 页为什么把 SF 放在 DATA 之后？
6. `bucket_q_block_interval` 为什么要把并集块按 query_mask 分桶，还要按页对齐补 `0xffffffff`？桶内为什么按物理地址排序？
7. `SplitKV::control` 的位域怎么划分？`kQBlockBegin/kQBlockEnd` 分别触发主 Kernel 里的什么动作？共享桶的 split 为什么排在两个 AIV 专属桶之前？
8. FIRST 和 REUSE 各自从哪里读 KV？谁在管理 FIRST/REUSE 状态，什么操作之后必须重建 metadata？dry-run 为什么不翻转状态？
9. 列出 Prefill 与 Decode 路径的至少 5 个差异，并各举一个源码证据。
10. R4 链由哪几段组成？为什么 reuse-main 的 TFLOPS 通常高于 first-main？prepare 的耗时为什么几乎与 KV 总长无关、只与 Q 数和选中块数有关？

<details>
<summary>参考答案（先自己做再展开）</summary>

1. `[i·B + ks%B, i·B + ks%B + B)`；token 起点 = 3·8 + 5 = 29；换算在 `sparse_mqa_logits_stage_indices`（`sparse_mqa_logits_metadata.hpp:138-139` 的 `×B + start%B`）。
2. 每行只存自己选中的块，列号 = 选中列表内的位置，避免为未选中块写 `-inf`；限制来自"空 split 也要编码结束偏移"的 16 位输出偏移（`csrc/jit_kernels/sparse_mqa_logits.hpp:23`）；B=16 时 4096·16 = 65536 不满足 `<`，上限 4095（测试 `:984, :1000-1004`）。
3. 4 位掩码，第 i 位 = QBlock 内 query i 是否选中当前并集块；由 `sparse_mqa_merge` 产生（`sparse_mqa_logits_metadata.hpp:46-51`），先存 `ub_query_masks`，再变成 `QBlockSplitKVOutputInfo::masks`（非分页）或 `SplitKV::control` 的 bucket_idx（分页，桶号就是 mask），最终被 AIV 的 `vf_sparse_mqa_logits_reduce_select/reduce_scatter` 消费。
4. 第 b 位 = 本 split 第 b 个**并集块**是否被该 query 选中（每块一位，SIMD 里展开成 B 列）；`output_ranges` 低 16 位 = 本 split 输出起始列、高 16 位 = 有效列数；因为压缩按 query 各自进行，起点只取决于该 query 在此之前累计选中了多少块（`:106-108`）。
5. 因为契约要求同请求的 query 对引用页达成一致，Host 直接按 query 行展开页表（测试 `:786-792` 的 `repeat_interleave`；`attention.hpp:100` 只查 `size(0)==Q`）；SF 放页尾使 DATA 段保持 `token_bytes` 等距、`byte_offset` 公式线性（`:60-62`），也匹配 `use_ascend_kv_layout=False` 的页格式（`attention.hpp:140`）。
6. 分桶让"同一组 query 共享的块"只装载一次并明确归属（桶号 = query_mask），供 `emit_bucket_kv_splits` 决定共享/专属排序；页对齐是因为 compact 存储按 `kCompactPageTokens=64` 的页组织（`kBlocksPerPage = 64/B`），桶基址必须整页起；物理地址排序让 gather 的成对 `copy_gm_to_ub` 拿到连续 burst（`:186-187` 与 `sparse_mqa_logits.hpp:78-86`）。
7. `[3:0]` bucket_idx、`[10:4]` num_blocks、`[13:11]` num_q_block_queries、`[14]` kQBlockBegin、`[15]` kQBlockEnd（`layout/sparse_mqa_logits.hpp:105-108`）；Begin 触发 Q 装载与输出起点记录，End 触发输出写回（AIC `:271-306`、AIV `:436-495`）；共享桶两个 AIV 都要消费，排前面保证两个 AIV 尽早同步拿到公共数据，专属桶交错填充两边的负载（`:275-277`）。
8. FIRST 从原始分页 GM（经页表寻址 gather）读，REUSE 从 metadata 内的 compact 区读；状态由 Python 侧 `metadata._paged_sparse_mqa_state` 管理（`__init__.py:107-134`）；改 KV 内容后必须重建（`attention.hpp:119-120`）；dry-run 不分配真实存储也不产生快照，翻转状态会让后续真实调用跳过 FIRST（`__init__.py:133-134` 的 `not dry_run` 条件）。
9. 参照 3.8 节对比表，任选 5 行并给出位置，如 kSplitKV（`layout/sparse_mqa_logits.hpp:16-18`）、KV 路径（`sparse_mqa_logits.hpp:330-335` vs `paged_sparse_mqa_logits.hpp:221-223`）、Q 形状（`attention.hpp:127` 的 `dim()==4` 与 `:187` 的 `dim()==3`）等。
10. 一次 prepare + 一次 FIRST + 3 次 REUSE（`tests/test_attention.py:568-571`）；REUSE 省掉页表寻址、NZ 转换和 compact 写回，KV 变成整页线性 GM→L1，单位时间完成的有效乘加更多；prepare 只读 context_lens/indices/块 ID/页表并做归并与桶划分，不读 KV 数据本身（`paged_sparse_mqa_prepare_impl` 的输入列表，`:524-526`）。
</details>
