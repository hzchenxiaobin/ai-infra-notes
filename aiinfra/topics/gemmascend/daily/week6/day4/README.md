# 第 6 周 · 第 4 天（总第 39 天）：学习普通 MQA Logits

> 本文是 [DeepGEMM-Ascend 逐日学习计划](../../../README.md)第 6 周"Grouped GEMM 与 MQA Logits"的第 4 天教程，可独立阅读。
> 文中仓库源码路径均相对 `DeepGEMM-Ascend/`。
> 建议时间分配（共 2～3 小时）：详细讲解约 50 分钟，源码导读约 25 分钟，实践步骤约 65 分钟，自测题约 15 分钟。

前三周我们把 BF16 GEMM 主线（调度 → 搬运 → MAD → 流水线 → Epilogue）和 Grouped GEMM 读完了。今天第一次离开"GEMM 家族"，进入 **MQA Logits**：它仍然用 MMAD 做矩阵乘，但计算对象、调度方式、Epilogue 形态都和普通 GEMM 不同。按计划，今天要读四个文件——`tests/test_attention.py`、`csrc/apis/attention.hpp`、`csrc/jit_kernels/mqa_logits.hpp`、`deep_gemm/include/deep_gemm/mqa_logits.hpp`——理解 **Q、KV、weights 的形状和 logits 计算过程**，并**对比 MQA Logits 与普通 GEMM 的调度及 Epilogue**。当天产出是一张 **MQA Logits 计算流程图**（第六节给出参考版）。Paged/Sparse 变体（`paged_sparse_mqa_logits.hpp` 等）留到第 40 天，今天只在对比处顺带指路。

## 一、学习目标

完成今天的学习后，你应当能够：

1. 写出 MQA Logits 一行的完整数学定义：`logits[q, j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩) · w[q,h]`，只在 `j ∈ [ks_q, ke_q)` 有效；并解释 ReLU 为什么发生在 FIX 管的 `L0C→UB` 转换里（`deep_gemm/include/deep_gemm/mqa_logits.hpp:400-403`，`ascend/copy.hpp:266-285`）；
2. 列出 Host 侧全部输入契约：Q `[Q, H, D]`、KV `[KV, D]`（同型 FP8/FP4）、SF `[Q·H, D/64]` / `[KV, D/64]` int16 列主序、weights `[Q, H]` bf16、`cu_seq_len_k_start/end` int32，以及 `H∈[4,64] 且 4|H`、`D∈{64,128}` 的约束（`csrc/apis/attention.hpp:241-299`）；
3. 说明"压缩输出"契约：输出 `logits[Q, max_seqlen_k]` bf16、行距 512B 对齐，第 q 行只写 `[ks, ke)` 对应的前缀（列号 = 原位置 − ks），以及测试如何把它还原成稠密矩阵（`tests/test_attention.py:166-172`）；
4. 手算一个 q_block 的调度量：`kNumQTokensPerBlock = 8·MAD_M/H`、窗口 `[start, end)`、`kv_token_base = start/8*8`、`num_kv_splits = ceil((end−base)/512)`、每个 split 的 `(kv_idx, actual_kv, issued_n)`（`scheduler/mqa_logits.hpp:31-70`）；
5. 描述 AIC/AIV 分工：AIC 做 Q/KV 搬运、MX scale 装载、`mad()`、带 ReLU 的 `copy_l0c_to_ub`；两个 AIV 按 tile 奇偶分摊 `vf_mqa_logits_reduce`（head 维加权求和）与 `ks/ke` 掩码存储（`mqa_logits.hpp:269-612`）；
6. 回答对比题：MQA Logits 与普通 GEMM 在调度（一维条带 vs 二维网格）、K 维（单次 MAD vs 两层循环累加）、Epilogue（FIX ReLU + head 缩减 + 掩码存储 vs bias/alpha/drain）上的核心差异。

## 二、预备知识

- **第 5 天**：GM → L1 → L0A/L0B → MAD → L0C → Epilogue → GM 数据流；AIC 与 AIV 的分工——今天的 Kernel 是这条流水线的又一个实例。
- **第 16 天**：片上内存 offset 分配与容量 `static_assert`——今天用同样方法给 MQA Kernel 记内存账。
- **第 18、19 天**：`gm_ptr` 的 `index()`/stride、`copy_gm_to_l1()`/`copy_l1_to_l0a/b()`——MQA 的 Q/KV 搬运复用同一套设施，只是多了 `_mx` 后缀的 scale 装载版本（`ascend/copy.hpp:161-229`）。
- **第 22～24 天**：K 维两层循环、`mad()` 与 `ENABLE_UPDATE`、L0C 累加——今天对比"MQA 没有 K 循环"时全靠这些概念做参照。
- **第 32～33 天**：FP8/FP4 与 UE8M0 scale、`MX_SF_DIVISOR = 64`（每对 scale 管 64 个 K 元素，即每 32 元素一个 E8M0）——MQA 输入就是这种 MX 布局。
- **第 36 天**：M-Grouped 的"行方向分组"——MQA 把 query×head 打包成 QH 行，思想同源：都是把不规整的逻辑维度拼成规整的 M 维。

> 推荐先花 5 分钟默写第 27 天的"BF16 GEMM 全流程"（任务调度 → 搬运 → MAD 累加 → AIC/AIV Epilogue → 写回），今天每讲一处 MQA 的差异都拿它对照。另注：第五节 A 组任务需要 Ascend 环境（Ascend 950 系列 + CANN 9.20，见仓库 `README.md` 的 Requirements）；无环境时完成 B 组静态任务即可达到当天目标。

## 三、详细讲解

### 3.1 MQA Logits 要算什么：一行公式与一个手算例

MQA（Multi-Query Attention）Logits 是 DeepSeek Lightning Indexer 的打分算子（仓库 `README.md:149-151`）。它与标准 attention score 的差别在于：**KV 只有一份共享头**（`kv: [KV, D]`，没有 head 维），而 Q 有 H 个头；每个头先算自己的点积分数，过 ReLU 后按每 query 每头的权重 `w` 加权求和，坍缩成一个 logit。用数学写：

```text
score[q, h, j] = ⟨q[q, h, :], kv[j, :]⟩            (点积，长度 D = head_dim)
logits[q, j]    = Σ_{h=0}^{H-1} relu(score[q, h, j]) · w[q, h]
                  仅在 ks[q] ≤ j < ke[q] 时有效，其余位置为 -inf
```

测试里的参考实现就是这句话的直接翻译（`tests/test_attention.py:94-112`）：

```python
# tests/test_attention.py:107-111
score = torch.einsum('qhd,kd->qhk', q_f32, kv_f32[kv_base:kv_end])
out = (score.relu() * weights_f32[:, :, None]).sum(dim=1)
mask = ((kv_positions[kv_base:kv_end][None, :] >= cu_seq_len_k_start[:, None]) &
        (kv_positions[kv_base:kv_end][None, :] < cu_seq_len_k_end[:, None]))
logits[:, kv_base:kv_end] = out.masked_fill(~mask, float('-inf'))
```

**手算例（今天反复用它）**：取 `Q=1, H=2, D=2`（真实 Kernel 要求 `H≥4`、`D∈{64,128}`，这里只为看清公式）：

```text
q[0] = [[1, -1],      # head 0
        [ 2,  0]]     # head 1
kv   = [[2, 0], [0, 2], [1, 1]]
w[0] = [0.5, 2.0]

score[0, 0, :] = [1·2+(-1)·0, 1·0+(-1)·2, 1·1+(-1)·1] = [ 2, -2,  0]
score[0, 1, :] = [2·2+ 0·0,     2·0+ 0·2,     2·1+ 0·1] = [ 4,  0,  2]
relu 后         = [2, 0, 0] / [4, 0, 2]
logits[0, j]    = 0.5·relu_h0[j] + 2.0·relu_h1[j]     = [9, 0, 4]

若 ks=1, ke=3：只有 j∈{1,2} 有效 → 压缩存储 logits[0] = [0, 4]
```

记住三个关键词：**ReLU 逐头施加**、**权重求和跨头坍缩**、**有效区间由 `[ks, ke)` 决定**。下面 3.2～3.9 节看仓库如何高效实现这三件事。

### 3.2 输入输出契约：Q、KV、weights 与压缩输出

Host 入口 `fp8_fp4_mqa_logits`（`csrc/apis/attention.hpp:228-309`）的参数检查就是一份完整契约表：

| 输入 | 形状 / dtype | 约束（源码行） | 语义 |
|---|---|---|---|
| `q.first` | `[Q, H, D]`，FP8(e4m3) 或 packed FP4 | contiguous；`H∈[4,64]` 且 `4|H`（`:254`）；`D∈{64,128}`（`:255-256`）；与 kv 同型（`:280-281`） | query，按 QH 行展平后每行 `D` 字节（FP4 为 `D/2`） |
| `q.second` (sf_q) | `[Q·H, D/64]` int16 | 必须提供（`:282`）；`size(-1)==D/64`（`:292`） | 每 int16 打包 2 个 UE8M0 scale，每个管 32 个 K 元素（`MX_SF_DIVISOR=64`，`common.hpp:65`） |
| `kv.first` | `[KV, D]`，与 q 同型 | contiguous（`:277-278`） | 共享 KV，每行一个 token |
| `kv.second` (sf_kv) | `[KV, D/64]` int16 | `:293` | 同 sf_q，按 KV 行组织 |
| `weights` | `[Q, H]` bf16 | `stride(1)==1`，行距可为 padding（`:275, :304`） | head 求和的逐头权重 |
| `cu_seq_len_k_start / k_end` | `[Q]` int32 | contiguous（`:244-245`） | 每行 query 的有效 KV 窗口 `[ks, ke)` |
| `max_seqlen_k` | 标量 | `> 0`（`:257`） | 所有行 `ke−ks` 的最大值 |

两点值得专门记下：

**（1）SF 的列主序要求。** `:294-299` 会把 sf_q/sf_kv 转置成 `stride(0)==1` 的"列主序"（第 18 天的 MN-major），因为 Kernel 里用 `gm_ptr<int16_t, Major::MN>` 访问（`mqa_logits.hpp:265, :364`；stride 分别是 `Q·H` 和 `KV`，见 `scheduler/mqa_logits.hpp:61-73`）。这与第 33 天 `transform_sf_into_required_layout` 的动机相同：NZ 分形搬运需要 scale 沿"行"连续。

**（2）输出的压缩存储。** `:301-302` 分配 `logits[Q, max_seqlen_k]` bf16，行距按 512 字节对齐（`get_logits_stride`，`:23-27`）。第 q 行**只写有效前缀**：KV 位置 `j` 存到第 `j − ks` 列。测试把它还原成稠密矩阵的方式一目了然（`tests/test_attention.py:168-172`）：

```python
out = torch.full((seq_len, seq_len_kv), float('-inf'), ...)
for i, (start, end) in enumerate(zip(starts, ends)):
    out[i, start:end] = kernel_out[i, :end - start]     # 前缀 → 原位置
```

好处：causal/random 窗口下输出不再有整段 `-inf` 列浪费带宽，每行只搬有效部分；代价是调用方要知道 `ks/ke` 才能解读列号。

### 3.3 Host 链路：检查、配置与启动

`fp8_fp4_mqa_logits` 收集完信息后组装 `MQALogitsDesc` 与 `MQALogitsConfig`（`:258-274`）：

```cpp
// csrc/apis/attention.hpp:258-274（摘录）
const MQALogitsDesc desc = { .is_fp4, .num_q_tokens, .num_kv_tokens, .num_heads, .head_dim };
const MQALogitsConfig config = {
    .split_kv = 512u,
    .mad_m = 64u / num_heads * num_heads,   // 不超过 64 的最大 H 倍数
    .num_cd_stages = 2u, .num_q_stages = 2u, .num_kv_stages = 4u,
    .num_accum_stages = 2u, .num_logits_stages = 3u,
    .launch_options = {.num_blocks = runtime->get_num_sms()},
};
```

与 GEMM 的 `select_gemm_config()`（启发式搜索，第 12 天）不同，**MQA 没有启发式**：`split_kv` 钉死 512，各级 stage 数固定，唯一随 `H` 变的是 `mad_m`。原因在 3.6 节的内存账本里——这些参数是被 L0A/L0C 容量"顶死"的（`mqa_logits.hpp:218-219, :229` 的两条注释与 static_assert）。

最后调用 `deep_gemm::launch_mqa_logits(...)`（`:304-307`），把 `weights_stride`、`logits_stride` 和九个数据指针传下去。Python 侧的入口链是：`deep_gemm.fp8_fp4_mqa_logits`（`deep_gemm/__init__.py:73` 从 `_C` 导出）→ `csrc/python_api.cpp:36` 注册的 `attention_api::register_apis` → `csrc/apis/attention.hpp:440-443` 的 pybind 绑定。

### 3.4 JIT 实例化：`launch_mqa_logits` 生成了什么

`csrc/jit_kernels/mqa_logits.hpp:30-78` 与第 12 天读过的 `launch_bf16_gemm` 同构：用 `std::format` 拼一段模板实例化代码，交给 `jit->compile("mqa_logits", code)`（缓存命中则免编译），再 `jit->launch`：

```cpp
// csrc/jit_kernels/mqa_logits.hpp:38-50（模板实参依次为）
// mqa_logits_impl<is_fp4, H, D, SPLIT_KV=512, MAD_M, CD=2, Q=2, KV=4, ACCUM=2, LOGITS=3, num_blocks>
```

进模板的是"编译期形状"（H、D、各级参数、核数），`Q/KV` 保持运行时参数（`:70-77` 的 launch 实参）。与 GEMM 一致的套路，区别只是配置来源从启发式变成了常量。数据指针侧：`q/kv/weights/logits` 包成 `gm_ptr`（带上各自的 stride），`sf_q/sf_kv/ks/ke` 以裸指针传递（`:70-77`）。

### 3.5 调度：一维条带轮转 + KV 窗口

设备端入口 `mqa_logits_impl`（`mqa_logits.hpp:623-639`）先用 `8·MAD_M/H` 算出每个 q_block 的 token 数，再构造 `MQALogitsScheduler`（`scheduler/mqa_logits.hpp:7-74`），最后进入公共实现 `mqa_logits_core_impl`。调度器只有三十行，行为如下：

```cpp
// scheduler/mqa_logits.hpp:18-52（行为摘要）
构造:  current_q_block_idx = core_id;                    // 每核从自己的编号起步
       num_q_blocks = ceil_div(num_q_tokens, kNumQTokensPerBlock);

next_q_block():
       q_block_idx = current_q_block_idx;                 // 条带轮转：
       current_q_block_idx += kNumCores;                  //   核 c 领 c, c+C, c+2C, ...
       // 求块内所有行 [ks, ke) 的并集窗口（逐行 clamp 到 num_kv_tokens）
       start = min over rows(ks);  end = max over rows(ke);
       block_kv_end = end;
       kv_token_base = start / 8 * 8;                     // 起点向下 8 对齐
       num_kv_splits = ceil_div(end - kv_token_base, kSplitKV);   // 512 一段
```

对比第 17 天的 Normal GEMM 调度（二维 M/N block 网格 + swizzle + `block_idx = iter×kNumCores + core_id`）：**MQA 只有一维的 Q block**，轮转公式退化为最简单的 `core_id + k·kNumCores`。N 维（KV）不参与调度——它由每个 q_block 自己的"窗口"导出：块内各行窗口的并集 `[start, end)` 决定要算哪些 KV token，再按 `SPLIT_KV=512` 切成若干 split。第 36 天 M-Grouped 的"组内 block 平移"在这里不存在，因为 q_block 本身就是均匀条带，尾部块用 `get_num_q_tokens_in_block` 截短（`:57-59`）。

每个 split 的落地量由两个查询函数给出（`:64-70`）：

```text
kv_idx    = kv_token_base + split_idx × 512            (get_kv_token_offset)
actual_kv = min(512, block_kv_end − kv_idx)，若 kv_idx ≥ block_kv_end 则 0
issued_n  = aligned(actual_kv, kNumVecElems)           (kernel 内，mqa_logits.hpp:346/544)
```

`issued_n` 把实际 token 数向上对齐到整数个向量寄存器宽度（`kNumVecElems = asc_get_vf_len()/2`），是后面搬运与 MAD 的 N 维长度；多出来的列由补零路径处理（3.6 节）。

### 3.6 Kernel 数据流：AIC 算分数，AIV 缩减并存储

`mqa_logits_core_impl`（`mqa_logits.hpp:195-613`）是今天的主菜。先看派生常量（`:213-229`）：

```text
MAD_M        = mad_m（Host 传入：≤64 的最大 H 倍数）
MAD_M_PADDED = aligned(MAD_M, 16)          // FRAC_MN = 16
MAD_N        = kSplitKV = 512
MAD_K        = kHeadDim                    // 64 或 128
kNumQHTilesPerBlock  = 8                   // 固定 8 个 QH tile
kNumQHRowsPerBlock   = 8 × MAD_M           // 一个 block 的 QH 行数
kNumQTokensPerBlock  = 8 × MAD_M / H       // 一个 block 的 query 数
kNumQTokensPerTile   = MAD_M / H           // 一个 tile 的 query 数
```

**关键视角**：MAD 的 M 维不是 query，而是 **QH 行**（query×head 展平）。`H=8, D=128` 时一个 block 有 512 个 QH 行（64 个 query），切成 8 个 `64×128` 的 L0A tile；`H=32` 时同样是 512 行，但只装 16 个 query——**片上布局对 H 不变，变的只是每块装多少 query**。这就是 `mad_m = 64/H*H` 的用意。默认测试头数下的派生值：

| H | MAD_M | MAD_M_PADDED | tokens/block = 8·MAD_M/H | tokens/tile = MAD_M/H |
|---|---|---|---|---|
| 8 | 64 | 64 | 64 | 8 |
| 12 | 60 | 64 | 40 | 5 |
| 16 | 64 | 64 | 32 | 4 |
| 20 | 60 | 64 | 24 | 3 |
| 32 | 64 | 64 | 16 | 2 |
| 64 | 64 | 64 | 8 | 1 |

#### AIC 分支（`mqa_logits.hpp:269-414`）

```text
对每个 q_block（scheduler.next_q_block）：
  ① Q 搬运：fill_mn_tail_block 补零尾行 → copy_gm_to_l1(Q 的 QH 行 × D)
             copy_gm_to_l1_mx(sf_q)                     (:291-313)
  ② L1→L0A：copy_l1_to_l0a(8 个 tile) + copy_l1_to_l0a_mx(scale)  (:315-325)
  ③ 对每个 kv_split：
       fill_mn_tail_block(l1kv, actual_kv → issued_n)   (:348)
       copy_gm_to_l1(kv) + copy_gm_to_l1_mx(sf_kv)      (:363-369)
       L1→L0B + copy_l1_to_l0b_mx(scale)                (:373-382)
       ④ 对每个 QH tile（is_tile_active 才做）：
            mad(l0c[cd_stage], l0a[tile], l0b,          (:396-397)
                M=QH 行, N=issued_n, K=D)
            copy_l0c_to_ub(ub_accum[accum_stage], l0c,  (:400-403)
                ..., F322BF16, relu=true, ...)
            set_intra_block(FIX → 归属 AIV)             (:404)
```

三个要点：

- **没有 K 循环。** `MAD_K = head_dim ≤ 128`，一次 `mad()` 吸收整个收缩维（`ascend/copy.hpp:370-381`，FP8/FP4 走 `asc_mmad_mx`）；跨 split 也**不累加**——每个 split 算的是不同 KV 列上的独立分数，各自走一遍 L0C→UB。这与 GEMM 的"`BLOCK_K` 外循环 + `MAD_K` 内循环 + L0C 跨 K 累加"（第 22 天）形成根本对照。
- **ReLU 在 FIX 管上。** `copy_l0c_to_ub` 的第 8 个参数是 `relu`（`ascend/copy.hpp:273`），这里传 `true`（`mqa_logits.hpp:402`）——FIX 管做 fp32→bf16 量化的同时施加 `asc_relu_pre_mode::NORMAL`。省掉一次专门的向量 ReLU pass；仓库 README 也点明该 Kernel 是 **FIX-pipe bound**（`README.md:151`："saturating the FIX pipe at 99% utilization"），所以把 ReLU 融进 FIX 是顺手的事。
- **`is_tile_active` 跳过空 tile**（`:328-340`）：tile 内任一行满足 `kv_idx < ke[row] && ks[row] < kv_end` 才做 MAD。causal/短窗口下靠它省掉大量"整 tile 都在窗口外"的乘加。

#### AIV 分支（`mqa_logits.hpp:415-612`）

`__global__ __mix__(1, 2)`（`:623`）表示每个 block 配 1 个 AIC + 2 个 AIV；AIV 用 `asc_get_sub_block_id()` 区分（`:416`）。两个 AIV 的活完全对称：

```text
对每个 q_block：
  ① ks/ke、weights 搬到 UB（MTE2）                     (:501-536)
  ② 对每个 kv_split：
       对归属自己的 QH tile（qh_tile_idx = sub_id ^ owner_offset，步进 2）(:556)
         wait_intra_block(accum_stage)                  (:563)
         ③ reduce：mqa_logits_reduce_queries            (:568-574)
              对 tile 内每个 query：
                sum = 0
                for h in 0..H-1:  sum += score[QH 行] × w[q, h]   (asc_mula)
                写入 ub_logits 该 query 的 512 宽行
         ④ 掩码存储：对每行 query                       (:585-601)
              store_begin = max(kv_idx, ks);  store_end = min(kv_idx+actual_kv, ke)
              若有重叠：dst_col = store_begin − ks
              mqa_logits_store(GM logits[row, dst_col], 行内偏移 line+store_begin−kv_idx)
```

**（a）head 缩减。** `vf_mqa_logits_reduce`（`:101-151`）是 Epilogue 的核心：score 的 QH 行在 UB 里按"每头一行、行长 issued_n"排列，循环 H 次做 `asc_mula(sum, value, weight)`（乘加进 bf16 向量寄存器），权重用 `asc_loadalign_brc_postupdate` 逐个广播（`:129-143`）。`mqa_logits_reduce_queries`（`:153-183`）还会在条件允许时**一次处理相邻两个 query**（`kNumQueries==1||2`，`:111`），摊薄权重加载。注意 score 进 reduce 前已被 FIX 管做过 ReLU，所以这里直接乘加——`simd/sparse_mqa_logits.hpp:64` 的注释"s is already ReLU'ed by FIX"说的就是这条约定。

**（b）owner 轮换。** `owner_offset = num_qh_tiles_in_block & kv_split_idx & 1`（`:385, :555`）：tile 数为偶数时它恒为 0，AIV0 拿偶数 tile、AIV1 拿奇数 tile；tile 数为奇数（小块）时它随 split 翻转，让两个 AIV 轮流处理那个"落单"的 tile。`accum_owner = (qh_tile_idx + owner_offset) & 1`（`:394`）保证 L0C→UB 的结果正好写给将要消费它的那个 AIV（`wait/set_intra_block` 在 PIPE_FIX 与 PIPE_V 之间配对，`:398-404, :563, :578`）。

**（c）32B 对齐的存储。** Ascend 950 的 MTE3 要求 UB 源地址 32B 对齐（`:33-34` 注释），而行内偏移 `store_begin − kv_idx` 未必对齐。`mqa_logits_store`（`:45-68`）把不齐的行头先经 `ub_logits_head` 暂存（`vf_mqa_logits_stage_head`，`:37-43`），再分两段 `asc_copy_ub2gm_align` 写出；GM 目标地址与长度不受限。补零行/补零列永远不会被存出（store 区间被 `actual_kv`、`ke` 夹住）。

#### 片上内存账本（H=8, D=128, FP8 实例）

按第 16 天的方法给 `:231-252` 的声明记账（容量常量见 `common.hpp:54-58`：L1=512KB，L0A/L0B=64KB，L0C=UB=256KB）：

| 缓冲 | 形状 | 级数 | 单级大小 | 合计 |
|---|---|---|---|---|
| `l1q` | 512 QH 行 × 128B | 2 | 64KB | 128KB |
| `l1kv` | 512 × 128B | 4 | 64KB | 256KB |
| `l1_sf_q` / `l1_sf_kv` | 512×2 / 512×2 int16 | 2 / 4 | 2KB | 4KB + 8KB |
| `l0a` | 8 tile × 64×128B | — | 8KB | **64KB（=L0A 容量）** |
| `l0b` | 512×128B | 1 | 64KB | 64KB |
| `l0c` | 64×512 fp32 | 2 | 128KB | **256KB（=L0C 容量）** |
| `ub_accum` | 64×512 bf16 | 2 | 64KB | 128KB |
| `ub_weights` / `ub_logits` | 64×8 / 8×512 bf16 | 2 / 3 | 1KB / 8KB | 2KB + 24KB |

L1 合计 396KB ≤ 512KB；UB 合计约 155KB ≤ 256KB。两个"恰好顶满"不是巧合：`kNumQHTilesPerBlock = 8` 来自 L0A 上限 `8 × MAD_M_PADDED × D × 1B ≤ 64KB`（`:218-219` 注释），`kNumCDStages = 2` 来自 `SPLIT_KV=512` 时 L0C 恰好放下两级（`:229` 的 static_assert）。这也解释了 3.3 节"配置为什么是常量"。

### 3.7 手算例：完整推演 SQ=200、SK=1200、H=8、D=128、Causal

下面把 3.5/3.6 的机制串成一遍可核对的演算。设定：`Q=200`，`KV=1200`，`H=8`，`D=128`，Causal（`ke[i] = i + (SK−SQ) = i+1000`，`ks=0`，见 `tests/test_attention.py:124-125`）；假设参与 4 个核（真实 950DT 为 32 核、`launch_options.num_blocks = get_num_sms()`，方法完全相同）。

**第一步：块划分。** `MAD_M = 64/8×8 = 64`，`kNumQTokensPerBlock = 8×64/8 = 64`，`num_q_blocks = ceil(200/64) = 4`。条带轮转：核 0→块 0，核 1→块 1，核 2→块 2，核 3→块 3（每核一块，正好分完）。

**第二步：每块的 KV 窗口与 split。** 用 `next_q_block` 的公式（`scheduler/mqa_logits.hpp:38-50`）：

| 块 | 行区间 | 窗口 end=max(ke) | kv_token_base | num_kv_splits | 各 split 的 (kv_idx, actual_kv) |
|---|---|---|---|---|---|
| 0 | 0–63 | 1063 | 0 | 3 | (0,512), (512,512), (1024,**39**) |
| 1 | 64–127 | 1127 | 0 | 3 | (0,512), (512,512), (1024,**103**) |
| 2 | 128–191 | 1191 | 0 | 3 | (0,512), (512,512), (1024,**167**) |
| 3 | 192–199（8 行） | 1199 | 0 | 3 | (0,512), (512,512), (1024,**175**) |

`issued_n = aligned(actual_kv, kNumVecElems)`：前两个 split 为 512，第三个 split 对齐后为一个向量宽度的整数倍（具体数值取决于芯片的 `asc_get_vf_len()`，方法一致）。

**第三步：QH tile 与 AIV 归属。** 块 0–2 各有 `64 token × 8 头 = 512` QH 行 = 8 个 tile；tile 数为偶数 → `owner_offset` 恒 0 → AIV0 处理 tile 0/2/4/6，AIV1 处理 tile 1/3/5/7。块 3 只有 `8×8 = 64` 行 = 1 个 tile：`owner_offset = 1 & kv_split_idx & 1`，split 0/2 归 AIV0、split 1 归 AIV1（`:385` 的轮换）。

**第四步：is_tile_active 省了多少。** 块 0 的 split 2 覆盖 `[1024, 1063)`：行 `ke = 1000..1063`，`ke > 1024` 的行是 25–63。tile 3（query 24–31，因 query 25 起 `ke=1025>1024`）到 tile 7 共 **5 个 tile 活跃，tile 0–2 被跳过**——AIC 不做 MAD，AIV 也不做 reduce/store（`:559-561` 注释"Mirror the AIC's skip"）。块 1/2 的 split 2 覆盖到 1127/1191，所有行 `ke ≥ 1064 > 1024`，8 个 tile 全活跃。全例合计 MAD 次数 = 块 0：8+8+5，块 1/2：各 8×3，块 3：1×3 → **21+24+24+3 = 72 次**，每次形状 `64 × issued_n × 128`。

**第五步：谁写哪些列。** 行 63（`ke=1063`）：split 0/1/2 分别写 `[0,512) [512,1024) [1024,1063)`，合成 `[0,1063)`；行 0（`ke=1000`）：split 2 的 `store_end = min(1063, 1000) = 1000 ≤ store_begin=1024`，跳过，只写 `[0,1000)`；行 150 属于块 2，写 `[0,1150)`。输出形状 `(200, max_seqlen_k=1199)`；causal 下 `ks=0`，压缩列号等于原列号。

**第六步：计算量口径。** 测试用 `flops = 2 × valid_sum × H × D`（`tests/test_attention.py:235-236`），`valid_sum = Σ(ke−ks)`。本例 `valid_sum = Σ_{i=0}^{199}(1000+i) = 219900`，`flops = 2×219900×8×128 ≈ 0.45 GFLOP`。注意这个口径**只数有效窗口内的乘加**——被 `is_tile_active` 跳过的 tile 天然不计入。

### 3.8 与普通 GEMM 的调度及 Epilogue 对比

| 维度 | 普通 BF16 GEMM（第 3～4 周） | MQA Logits（今天） |
|---|---|---|
| 计算对象 | `D = A×B`，输出 tile M×N | `logits[q,j] = Σ_h relu(q·k)·w`，输出行 q 的 KV 窗口 |
| M 维含义 | 输出行 | **QH 行**（query×head 展平，`MAD_M` 行一个 tile） |
| N 维含义 | 输出列 | KV token（`SPLIT_KV=512` 一段） |
| K 维 | `BLOCK_K`×`MAD_K` 两层循环，L0C 跨 K 累加 | **单次 `mad()`，K=head_dim；跨 split 不累加**（各写各的列） |
| 调度 | 二维 M/N 网格 + swizzle（`scheduler.hpp:144-177`） | **一维 Q block 条带**（`core_id + k·kNumCores`）；KV 范围由 `[ks,ke)` 窗口导出 |
| 边界处理 | `actual_m/n/k` 截断搬运 | `fill_mn_tail_block` 补零 + `is_tile_active` 跳空 tile |
| 输入精度 | bf16 | FP8/FP4 + UE8M0 MX scale（`_mx` 系列搬运，`copy.hpp:161-229`），weights bf16 |
| Epilogue | bias/alpha/accumulate、随机舍入、AIV drain（`epilogue.hpp`） | **FIX 管带 ReLU 的量化**（`:400-403`）+ AIV head 缩减（`asc_mula`）+ `ks/ke` 掩码存储 |
| 输出布局 | `[M, N]` 常规、支持 fp32/bf16/累加 | `[Q, max_seqlen_k]` **压缩前缀**，仅 bf16，512B 行对齐 |
| AIC/AIV | AIC 算 MAD，AIV 做 Epilogue | `__mix__(1,2)`：1 AIC（MAD+FIX）+ 2 AIV 按 tile 奇偶分摊 |
| 配置来源 | `select_gemm_config()` 启发式 | **常量配置**（容量顶死，`attention.hpp:265-274`） |

一句话总结：MQA Logits 复用了 GEMM 的搬运/MAD/流水线设施，但把"K 收缩"换成了"N 分段"、把"Epilogue 的代数运算"换成了"跨头坍缩 + 窗口掩码"。真正的新问题是**变长窗口调度**（`[ks, ke)`）与**压缩输出**——这两点在第 40 天的 Paged/Sparse 变体里会进一步演化成页表寻址和稀疏块选择。

### 3.9 测试如何验证

`tests/test_attention.py` 的普通 MQA 部分（Paged/Sparse 留到第 40 天）：

- **量化与参考**（`:79-91`）：`quantize_mqa_tensor` 按 32 元素一组取 amax，除以 448（FP8）或 6（FP4）后转 UE8M0 指数，两个字节打包进一个 int16——正是 `MX_SF_DIVISOR=64` 的布局。`run_mqa_logits_case` 同时构造两套参考：原始 bf16 输入的 `ref_bf16` 和量化模拟输入的 `ref`（`:162-163`）。
- **还原与掩码**（`:166-179`）：先断言输出形状/stride（512B 行对齐、`stride(1)==1`）、dtype bf16，散射回稠密矩阵后，先**逐位比对 `-inf` 掩码位置**，再把两边掩码处置 0 只比数值。
- **误差口径**（`:195-231`）：`calc_diff`（余弦相似度差，`tests/utils.py:36-43`）双阈值——对量化模拟 `tol=5e-4`，对原始 bf16 `orig_tol=1e-3`（FP8）/`2e-2`（FP4）。另有 `DG_MQA_TEST_PATTERN=ones` 的全 1 满规约精确检查（`:181-193`，`max_err < 1e-5`）。
- **ks/ke 生成**（`:115-135`）：四种 `Mode`——`FullRange`（整段）、`Causal`（`ke=i+offset`）、`Cp`（context-parallel 分块 causal：前半行窗口从 `cp_id·chunk` 起，后半行从 `(2·cp_size−1−cp_id)·chunk` 起）、`Random`（每行随机区间）。默认跑 `Cp+Causal`（`:27`）。返回值 `max(ke−ks)` 就是传给 kernel 的 `max_seqlen_k`（`:135`）。
- **用例组织**（`:254-270`）：`DG_MQA_SQ/DG_MQA_SK` 控制规模（默认 4096/8192），默认 `2 模式 × 2 格式 × 6 头数 × 1 head_dim = 24` 例；先 `compile_mqa_kernels` 两遍法预编译（`:245-251`，与第 30 天 `par_compile` 同套路）再逐例执行。参考实现按 `DG_MQA_SPLIT_KV`（默认 256）分段只为控制 einsum 的峰值内存（`:104`），Kernel 的 `SPLIT_KV` 恒为 512。
- **边界用例**（`:273-294`）：`SK=511/513` 强制出现"最后一个 split 只有 1 个 token"（`:278-279` 注释）；`SQ=17`（`512//32+1`）让 H=32 时出现 2 个块且尾块只有 1 个 query；H=12/64 覆盖非整除与满 tile 头数；Random+FP4 组合覆盖窗口错位。
- **性能口径**（`:233-240`）：`flops = 2·valid_sum·H·D`（只数有效窗口）、`bytes_moved = count_bytes(全部输入输出)`（`tests/utils.py:95-103`），并打印 AIC/AIV 各管道占比（`format_pipe_profile`，`:35-43`）——README 的 MQA 表（`README.md:153-158`，如 Prefill FP8 4096/8192/32/128：302.6us、681 TFLOPS、283 GB/s）就是这套口径的产出。

## 四、源码导读

今天读码的目标是"把 3.6 节的数据流图钉到具体行号上"。建议按表顺序读：

| # | 文件 | 关键函数/对象 | 核对什么 |
|---|---|---|---|
| 1 | `tests/test_attention.py` | `Mode` `:17-27`、`ref_mqa_logits` `:94-112`、`generate_ks_ke` `:115-135` | 公式、四种窗口模式与 3.1/3.9 节一致 |
| 2 | `tests/test_attention.py` | `quantize_mqa_tensor` `:79-91`、`run_mqa_logits_case` `:138-242`（散射 `:168-172`、双阈值 `:195-231`、perf `:233-240`） | 压缩输出契约与误差口径 |
| 3 | `tests/test_attention.py` | `test_mqa_logits` `:254-270`、`test_mqa_logits_boundaries` `:273-294`、main `:1269-1275` | 24 例的组成与边界用例意图 |
| 4 | `csrc/apis/attention.hpp` | `fp8_fp4_mqa_logits` `:228-309`（契约 `:241-299`、Desc/Config `:258-274`、SF 转置 `:294-299`、logits 分配 `:301-302`）、`get_logits_stride` `:23-27`、绑定 `:440-443` | 3.2/3.3 节的契约表与常量配置 |
| 5 | `csrc/jit_kernels/mqa_logits.hpp` | `MQALogitsDesc/Config` `:11-28`、`launch_mqa_logits` `:30-78`（模板 `:38-62`、launch 实参 `:70-77`） | 11 个模板实参与 3.4 节一致 |
| 6 | `deep_gemm/include/deep_gemm/scheduler/mqa_logits.hpp` | 构造 `:18-29`、`next_q_block` `:31-52`、`get_num_kv_block_tokens` `:64-70`、SF stride `:61-73` | 3.7 节手算表的每个数 |
| 7 | `deep_gemm/include/deep_gemm/mqa_logits.hpp` | 派生常量 `:213-229`、片上声明 `:231-252`、static_assert `:254-262` | 3.6 节内存账本 |
| 8 | 同上 | AIC：`asc_set_mmad_direction_n` `:270`、Q 搬运 `:291-325`、KV/split 循环 `:343-410`（`is_tile_active` `:328-340`、`mad` `:396-397`、`copy_l0c_to_ub` relu `:400-403`） | "无 K 循环 + FIX ReLU" |
| 9 | 同上 | AIV：`sub_id` `:416`、weights/seq 搬运 `:501-536`、tile 归属 `:556`、reduce `:568-574`、掩码存储 `:585-601`；`vf_mqa_logits_reduce` `:101-151`、`mqa_logits_reduce_queries` `:153-183`、`mqa_logits_store` `:45-68` | head 缩减与 32B 对齐存储 |
| 10 | 同上 | 入口 `mqa_logits_impl` `:615-639`（`__mix__(1,2)` `:623`、`kPageSize=0` `:638`）；paged 入口 `paged_mqa_logits_impl` `:641-675` | 普通/分页共用 core_impl，差异在调度器与 kPageSize |
| 11 | `deep_gemm/include/deep_gemm/ascend/copy.hpp` | `copy_gm_to_l1_mx` `:161-197`、`copy_l1_to_l0a/b_mx` `:199-229`、`copy_l0c_to_ub`（relu 参数）`:266-285`、`mad` `:370-381` | scale 装载与 `asc_mmad_mx` |
| 12 | `deep_gemm/include/deep_gemm/common.hpp` | 容量 `:54-58`、`FRAC_MN` `:63`、`MX_SF_DIVISOR` `:65`、`gm_ptr::index` `:156-165` | 账本常量与 QH 行寻址 |

**逐段导读一：`next_q_block` 的窗口推导**（`scheduler/mqa_logits.hpp:38-50`）。逐行扫描块内的 `kNumQTokensPerBlock` 行（`row_idx = min(q_base+q_idx, num_q_tokens-1)` 夹住尾块越界），对 `ks/ke` 先各自 clamp 到 `num_kv_tokens` 再取 min/max——所以窗口永远是合法 KV 区间的并集；`kv_token_base = start/8*8` 把起点向下对齐到 8 的倍数，`num_kv_splits = ceil_div(end−base, 512)`。空窗口（`end ≤ start`，如 Random 模式下 `ks==ke` 的行）自然得到 0 个 split，存储段 `store_begin ≥ store_end` 时整体跳过。

**逐段导读二：AIC 的 split 主循环**（`mqa_logits.hpp:343-410`）。循环体严格四拍：MTE1 等 KV stage → `fill_mn_tail_block` 把 `actual_kv` 之后的行补零到 `issued_n` → `copy_gm_to_l1` + `copy_gm_to_l1_mx`（普通路径 `:363-369`）→ MTE2 → `copy_l1_to_l0b` + `_mx` → M。内层 tile 循环先过 `is_tile_active`（`:390`），然后 `mad(l0c[cd_stage], l0a[tile], l0b, num_m_rows, issued_n, MAD_K, ENABLE_UPDATE)`。注意每次 MAD 都是**全 K 单拍**且 `cd_stage_idx` 在两级 L0C 间轮转（`:405`），没有 GEMM 里"首次 clear、后续 update"的 K 语义。`copy_l0c_to_ub` 之后 `set_intra_block(PIPE_FIX, accum_owner, accum_stage_idx)`（`:404`）把该 stage 的所有权交给对应 AIV。

**逐段导读三：AIV 的 reduce 与存储**（`mqa_logits.hpp:556-607`）。`for (qh_tile_idx = sub_id ^ owner_offset; ...; += 2)` 是两个 AIV 的分摊公式；`wait_intra_block(PIPE_V, sub_id, accum_stage)` 等到 AIC 交付的分数 tile 后，`mqa_logits_reduce_queries` 逐 query 调 `vf_mqa_logits_reduce`——`#pragma unroll 1` 的头循环（`:129`）逐头加载分数行并 `asc_mula` 乘加，权重经 `asc_loadalign_brc_postupdate` 广播（`:133`）；`issued_n/kNumVecElems` 的 switch（`:175-179`）按 1～4 个向量寄存器宽度选择模板。存储段（`:585-601`）逐行求 `[max(kv_idx,ks), min(kv_end,ke))`，`dst_col = store_begin − ks`（`:596`）实现压缩列号，`mqa_logits_store` 处理 32B 对齐（`:598-600`）。收尾 `set_intra_block(PIPE_V, ...)` 归还 accum stage（`:578`），logits stage 在 3 级间轮转（`:603`）。

## 五、实践步骤

### A 组：有 Ascend 环境

1. **跑通最小子集**（避免默认 `all` 选中 6 组测试的全量执行）：

   ```bash
   cd DeepGEMM-Ascend/tests
   python test_attention.py prefill --skip-prof --mode Causal --formats fp8 --num_heads 8 --head_dims 128
   ```

   预期打印 `Testing MQA logits (1 cases)` 后输出一行 ` > FP8 (SQ=4096, SK=8192, H= 8, D=128, Causal  ): OK`。核对 `--mode/--formats/--num_heads/--head_dims` 如何进入 `itertools.product`（`test_attention.py:1242-1259, :257-260`）。

2. **复算手算例（3.7 节的可执行版）**：

   ```bash
   DG_MQA_SQ=200 DG_MQA_SK=1200 python test_attention.py prefill --skip-prof --mode Causal --formats fp8 --num_heads 8 --head_dims 128
   ```

   对照：case tag 应显示 `SQ= 200, SK= 1200`；用 `max(ke−ks) = 1199`、`valid_sum = 219900` 核对你的手算（可另写两行 Python 在 CPU 上算这两个量再比对）。

3. **跑边界用例**：

   ```bash
   python test_attention.py boundary
   ```

   读 `test_mqa_logits_boundaries`（`:273-294`）的用例表，先在纸上预测每个考察点（511/513 → 尾 split 1 个 token；17 query + H=32 → 尾块 1 个 query；H=12 → MAD_M=60 非整除；Random+FP4 → 窗口错位），再与通过结果对账。

4. **观察 FIX 管瓶颈**（对应 README 的 99% 论断）：

   ```bash
   DG_MQA_SQ=4096 DG_MQA_SK=8192 python test_attention.py prefill --mode Causal --formats fp8 --num_heads 32 --head_dims 128
   ```

   去掉 `--skip-prof`，从 `AIC[... fix=xx.x%]` 读出 FIX 管占比；对照 `README.md:153-158` 的 Prefill FP8 行（4096/8192/32/128：302.6us、681 TFLOPS、283 GB/s）量级是否吻合。

5. **全 1 模式精确校验**：

   ```bash
   DG_MQA_TEST_PATTERN=ones DG_MQA_SQ=512 DG_MQA_SK=1024 python test_attention.py prefill --skip-prof --mode FullRange --formats fp8 --num_heads 32 --head_dims 128
   ```

   解释为什么全 1 输入可以近似逐位核对（满规约、无非线性截断，`max_err < 1e-5`，`:181-193`）。

### B 组：无 Ascend 环境（静态任务）

1. **调用链填表**：不看正文，把下表补全再对照 3.3/3.4 节：

   | 层 | 位置 | 输入 → 输出 |
   |---|---|---|
   | Python | `deep_gemm/__init__.py:73` | `fp8_fp4_mqa_logits((q,q_sf),(kv,kv_sf),w,ks,ke,L)` → `_C` |
   | binding | `csrc/apis/attention.hpp:440-443` | 同名参数 → C++ 函数 |
   | 检查/组装 | `csrc/apis/attention.hpp:241-302` | 张量 → `MQALogitsDesc` + `MQALogitsConfig` + 转置 SF + logits 分配 |
   | JIT | `csrc/jit_kernels/mqa_logits.hpp:38-67` | 模板实参 → `mqa_logits_impl<...>` 实例化代码 |
   | launch | 同上 `:70-77` | 指针/stride → kernel 运行时实参 |

2. **完成 3.7 节手算表的"块 2"一列**：写出块 2 的窗口、splits、各 split 活跃 tile 数（split 2 应为 8/8 全活跃），并算出核 2 处理的 query 行区间与行 150 的存储列区间。

3. **CPU 参考实现**（纯 PyTorch，无 NPU）：实现 3.1 节手算例的压缩输出并自查：

   ```python
   import torch

   def mqa_logits_ref(q, kv, w, ks, ke):
       # q: (Q, H, D); kv: (KV, D); w: (Q, H); ks/ke: 长度 Q 的 list
       score = torch.einsum('qhd,kd->qhk', q, kv)
       out = (score.relu() * w[:, :, None]).sum(dim=1)   # (Q, KV)
       L = max(e - s for s, e in zip(ks, ke))
       logits = torch.full((q.shape[0], L), float('-inf'))
       for i, (s, e) in enumerate(zip(ks, ke)):
           logits[i, :e - s] = out[i, s:e]               # 压缩前缀
       return logits

   q = torch.tensor([[[1., -1.], [2., 0.]]])   # Q=1, H=2, D=2
   kv = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])
   w = torch.tensor([[0.5, 2.0]])
   print(mqa_logits_ref(q, kv, w, [1], [3]))   # 期望 [[0., 4.]]（对照 3.1 手算）
   ```

   再把 `ks/ke` 换成 `[0], [3]`、`[2], [2]`（空窗口）各跑一次，解释空窗口行为（对应调度器 `end ≤ start` 的路径与存储段跳过）。

4. **换头数记账**：仿照 3.6 节账本，对 `H=12, D=128, FP8` 重算 `MAD_M=60、MAD_M_PADDED=64、tokens/block=40、tokens/tile=5`，并回答：L1/L0A/L0C 的占用与 H=8 有何不同？（提示：`kNumPaddedQHRowsPerBlock` 只依赖 `MAD_M_PADDED` 与 tile 数 8；每个 tile 内 60→64 的补零走 `:292-312` 的 `MAD_M != MAD_M_PADDED` 分支。）

5. **边界用例静态推演**：`test_attention.py:277-281` 用 `delta=±1` 构造 `SK=511/513` 的用例（元组为 `(seq_len, seq_len_kv, heads, head_dim, skip_prof, use_fp4, mode)`，513 的那个是 FP4 + FullRange）。对 `SK=513, H=32, D=128, FullRange` 算出 `num_kv_splits = 2` 与第二个 split 的 `(kv_idx, actual_kv) = (512, 1)`，并指出该用例专门压实的是哪条代码路径（尾 split 的补零 + 单 token 掩码存储）。

## 六、当天产出：MQA Logits 计算流程图

按计划，今天的产出是一张 **MQA Logits 计算流程图**。建议自己先画（从 Python 调用画到 GM 写回，标出 AIC/AIV 与各级流水），再与下面的参考版对照补漏：

```text
Python: deep_gemm.fp8_fp4_mqa_logits((q, q_sf), (kv, kv_sf), weights, ks, ke, L)
  │
  │ csrc/apis/attention.hpp:228  fp8_fp4_mqa_logits
  │   ├ 契约检查（H∈[4,64] 且 4|H；D∈{64,128}；q/kv 同型 FP8/FP4；SF 形状）
  │   ├ MQALogitsDesc{is_fp4, Q, KV, H, D}
  │   ├ MQALogitsConfig{split_kv=512, mad_m=64/H*H, cd=2, q=2, kv=4, accum=2, logits=3}
  │   ├ SF 转置列主序；logits[Q, L] bf16，行距 512B 对齐
  │   └ launch_mqa_logits(...)
  │        csrc/jit_kernels/mqa_logits.hpp:30
  │        ├ std::format → mqa_logits_impl<is_fp4,H,D,512,MAD_M,2,2,4,2,3,cores>
  │        ├ jit->compile("mqa_logits", code)     ← 缓存命中免编译
  │        └ jit->launch(kernel, num_blocks=核数, Q, KV, q, kv, sf_q, sf_kv, w, logits, ks, ke)
  ▼
Device: mqa_logits_impl   __mix__(1 AIC + 2 AIV)      mqa_logits.hpp:623
  │
  ├ 调度 MQALogitsScheduler(core_id, ...)             scheduler/mqa_logits.hpp:7
  │    q_block 条带轮转：核 c ← 块 c, c+C, c+2C, ...
  │    块窗口 = 块内各行 [ks,ke) 的并集 → base(8 对齐) → splits = ceil((end−base)/512)
  │
  ├ AIC（每个 q_block）                                mqa_logits.hpp:269-414
  │    Q : GM→L1（QH 行×D + sf_q，尾行补零）→ L0A（8 tile + MX scale）
  │    KV: 每个 split：GM→L1（actual_kv 行 + sf_kv，补零到 issued_n）→ L0B(+MX scale)
  │    每个活跃 QH tile：
  │        mad(L0C, L0A[t], L0B; M=QH 行, N=issued_n, K=D)   ← 无 K 循环
  │        copy_l0c_to_ub(ub_accum, …, F322BF16, relu=true)  ← FIX 管施加 ReLU
  │        set_intra_block(FIX → 归属 AIV)
  │
  └ AIV×2（tile 奇偶分摊，奇数 tile 时按 split 轮换）   mqa_logits.hpp:415-612
       weights/ks/ke : GM→UB
       每个归属 tile：
         wait accum → reduce: sum = Σ_h relu(score)·w[q,h]   (asc_mula, bf16)
         掩码存储: [max(kv_idx,ks), min(kv_end,ke))
                   → 压缩列 dst_col = store_begin − ks
                   → 行头 32B 对齐暂存 → MTE3 → GM logits[row, dst_col]
  ▼
输出: logits[Q, max_seqlen_k] bf16 — 每行只含 [ks, ke) 有效前缀
```

检查清单（自查你的图是否包含）：① `mad_m = 64/H*H` 与 QH 行视角；② `is_tile_active` 跳过；③ FIX 管上的 ReLU；④ 两个 AIV 的分摊与 accum stage 归属；⑤ 压缩列号与 512B 行对齐；⑥ 与普通 GEMM 的三处关键差异（无 K 循环、一维调度、掩码 Epilogue）。

## 七、自测题

1. 写出 MQA Logits 第 `(q, j)` 个输出的完整计算式，并指出 ReLU 施加在哪个阶段。
2. 为什么 MQA 的 Kernel 没有 `BLOCK_K/MAD_K` 两层循环，而 GEMM 必须有？跨 KV split 时 L0C 为什么不需要累加语义？
3. 输出为什么是 `[Q, max_seqlen_k]` 而不是 `[Q, KV]`？第 q 行的第 0 列对应 KV 的哪个位置？为什么行距要按 512B 对齐？
4. `MQALogitsScheduler::next_q_block` 如何决定一个块要处理的 KV 范围？`kv_token_base` 与 `num_kv_splits` 各怎么算？
5. `is_tile_active` 的判断条件是什么？它替 AIC 和 AIV 各省掉了什么？
6. `H=12` 时 `MAD_M`、`MAD_M_PADDED`、每块 query 数、每 tile query 数各是多少？片上内存占用与 `H=8` 相比有什么变化？
7. 两个 AIV 如何分摊一个块的 8 个 QH tile？只有 1 个 tile 的尾块靠什么机制避免一个 AIV 空转？
8. 测试如何验证"输出只在有效前缀有定义"？两套参考（`ref_bf16` 与 `ref`）和两档阈值分别对应什么口径？
9. 对照 3.8 节对比表：普通 GEMM 的调度是二维网格 + swizzle，MQA 为什么可以退化成一维条带？调度信息的"另一半"（KV 范围）从哪里来？
10. MQA Logits 的 flops 口径 `2·valid_sum·H·D` 与 GEMM 的 `2·M·N·K` 有什么本质区别？

<details>
<summary>参考答案（先自己做再展开）</summary>

1. `logits[q,j] = Σ_h relu(⟨q[q,h,:], kv[j,:]⟩)·w[q,h]`，仅 `j∈[ks_q, ke_q)`；ReLU 在 AIC 的 `copy_l0c_to_ub`（FIX 管做 fp32→bf16 量化）时施加（`mqa_logits.hpp:400-403`）。
2. 收缩维就是 `head_dim ≤ 128`，单次 `mad()` 可整体吸收；每个 KV split 计算的是不同输出列上的独立分数，各自独立写回（`store_end` 由 `min(kv_idx+actual_kv, ke)` 夹住），无需跨 split 累加。
3. 压缩存储避免整段 `-inf` 列浪费带宽；第 0 列对应 KV 位置 `ks`（`dst_col = store_begin − ks`）；512B 对齐是 `get_logits_stride` 的 `kStrideAlignBytes=512` 要求（`attention.hpp:23-27`），也与 MTE3 的 32B 源对齐及测试断言（`test_attention.py:166`）配套。
4. 逐行扫块内行的 `[ks, ke)`（clamp 到 `num_kv_tokens`）取 min/max 得窗口 `[start, end)`；`kv_token_base = start/8*8`，`num_kv_splits = ceil((end−base)/512)`（`scheduler/mqa_logits.hpp:38-50`）。
5. 条件：tile 内存在行满足 `kv_idx < ke[row] && ks[row] < kv_end`；AIC 跳过该 tile 的 `mad()` 与 L0C→UB，AIV 镜像跳过 reduce/store（`:390, :559-561`）。
6. `MAD_M=60`、`MAD_M_PADDED=64`、40 query/块、5 query/tile；L1/L0A/L0C 占用与 H=8 相同（由 `MAD_M_PADDED` 与 tile 数 8 决定），只是每个 tile 内有 4 行补零（`:292-312` 分支）。
7. `qh_tile_idx = sub_id ^ owner_offset`，步进 2；偶数 tile 时 `owner_offset=0` 固定奇偶分摊，奇数 tile 时 `owner_offset = num_qh_tiles_in_block & kv_split_idx & 1` 随 split 翻转轮换（`:385, :555`）。
8. 先散射回稠密矩阵、逐位比对 `-inf` 掩码，再掩码置 0 比数值；`ref_bf16`（原始 bf16 输入，`orig_tol=1e-3/2e-2`）考"对未量化问题的可用性"，`ref`（量化模拟输入，`tol=5e-4`）考"kernel 与数值模型的逐位一致性"。
9. 因为 N 维（KV）不构成输出 tile 网格——每行 query 的输出是一段变长窗口，由 per-row 的 `[ks, ke)` 在 kernel 内导出（scheduler 窗口），Host 无需静态切分 N。
10. GEMM 的 M×N×K 是全部实算；MQA 只数有效窗口内的乘加（`valid_sum = Σ(ke−ks)`），被 `is_tile_active` 跳过的补零乘加不计入——所以它衡量的是"有用计算"的密度。
</details>
