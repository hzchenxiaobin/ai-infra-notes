# Day 5（周五）：vLLM `fused_moe` 源码精读

> **本周定位**：本专题是 [CUTLASS 专题](../cutlass/README.md)（算子视角，Day 7 Group GEMM）之后的**系统视角**——把 Grouped GEMM、Top-K 路由、all-to-all 通信、负载均衡组装成一个完整的 MoE 层。本周目标是用 Triton 拼出一个 Top-2 路由的 MoE FFN 层,性能达到 Megatron-LM 参考实现 70%+,产出 ncu 性能报告。
> **前置要求**：已完成 Day 1-4（MoE 算法 + Gating + Grouped GEMM + EP 通信），理解单卡 MoE 前向与多卡 all-to-all
> **今日目标**：精读 vLLM `fused_moe.py` 的生产级 MoE 推理实现——三段式架构（dispatch + GEMM + combine）、Triton MoE GEMM kernel 的 tile 分配、dispatch/combine 的 Triton 实现、w1/w2 两层 GEMM + SiLU 融合，对比 Day 2-3 自写 kernel 与 DeepGEMM Mega MoE 的设计差异
> **时间投入**：2.5h（早间 1.5h 精读源码 + 晚间 1h 对照与总结）
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"vLLM fused_moe 怎么融合"与"单卡融合 vs 跨卡融合"必问

---

## 本日在本周知识图谱中的位置

```
Day 1          Day 2           Day 3            Day 4           Day 5          Day 6        Day 7
 总览      →   Gating +    →   Grouped       →   Expert      →  vLLM         → 完整       →  调优
 算法动机      Top-K 融合      GEMM              Parallelism    fused_moe       Triton       ncu
 数据流        Triton         Triton/CUTLASS    all-to-all     源码精读        MoE FFN      报告
 路由算法      kernel          cuBLAS 对照      Megatron 通信
 朴素实现                                                       ↑
                                                            你在这里（生产级 MoE 推理：vLLM fused_moe 源码精读）
```

| 本日产出 | 对应本周验收标准 |
|----------|-----------------|
| vLLM fused_moe 三段式架构精读笔记 | ⑤ ncu 定位 MoE 通信/计算占比（理解生产级实现是调优前提） |
| Triton MoE GEMM kernel 的 tile 分配 | ③ Grouped GEMM 90%+ cuBLAS（对照生产级 tile 策略） |
| dispatch/combine Triton kernel 精读 | ① MoE 前向数据流（生产级实现验证 Day 1 数据流） |
| 单卡融合 vs 跨卡融合（Mega MoE）对比 | 面试准备（两条融合路线的 tradeoff） |
| 与 Day 2-3 自写 kernel 的差距分析 | Day 6 组装完整 MoE 的改进方向 |

> ⚠️ **Day 5 的定位**：Day 2-3 自己写了 Gating 与 Grouped GEMM 的简化版，Day 4 理解了多卡 EP 通信。今天读 vLLM 的生产级实现，看它如何用**少数几个 Triton kernel** 把整个 MoE FFN 融合——这是 Day 6 组装完整 MoE 的"参考答案"。vLLM 是单卡融合（无 EP），与 DeepGEMM Mega MoE 的跨卡融合形成对照。

---

### 学习任务 1：vLLM fused_moe 的设计动机与目录结构（30 分钟）

#### 为什么 vLLM 要自己写 fused_moe

vLLM 是 LLM 推理框架，MoE 推理（如 Mixtral 8x7B、DeepSeek-V2）是核心场景。朴素 MoE（Day 1）的问题在推理时更严重：

| 问题 | 训练 | 推理 |
|------|------|------|
| **小 batch** | batch 大（4096+） | decode 时 batch 可能只有 1-128 |
| **小 GEMM 利用率** | 每 expert 收到 ~256 token | 每 expert 可能只收 1-16 token |
| **launch 开销** | 占比小 | 占比大（8 个小 GEMM × 5us = 40us） |
| **EP 通信** | 必需（专家在多卡） | 单卡可放下（Mixtral 8x7B ~47GB） |

vLLM 的 `fused_moe` 针对**单卡推理**优化——把 Gating + Dispatch + Grouped GEMM + Combine 融合成少数几个 Triton kernel，消除 launch 开销与小 GEMM 利用率问题。

#### 目录结构

vLLM 的 fused_moe 代码在 `vllm/model_executor/layers/fused_moe/`：

```
vllm/model_executor/layers/fused_moe/
├── __init__.py              # FusedMoE 模块定义
├── fused_moe.py             # 核心入口：fused_experts 函数
├── moe_align_block_size.py  # dispatch 前的对齐 kernel
├── topk.py                  # Top-K 重排序
└── fused_moe.py             # Triton kernels（dispatch / gemm / combine）
```

#### `FusedMoE` 模块的接口

```python
# vllm/model_executor/layers/fused_moe/__init__.py（简化）
class FusedMoE(torch.nn.Module):
    def forward(self, x):
        # x: [num_tokens, d]
        # 1. Gating + Top-K
        router_logits = self.gate(x)                    # [T, N]
        topk_weights, topk_indices = self.topk(router_logits)  # [T, K]
        
        # 2. 调用 fused_experts（今天的主角）
        return fused_experts(
            x, w1, w2, topk_weights, topk_indices,
            inplace=True,  # 推理时 inplace 省显存
        )
```

> 💡 **vLLM 的分层**：`FusedMoE` 模块负责 Gating + Top-K（可以用 Day 2 的 Triton kernel），`fused_experts` 负责剩下的 Dispatch + GEMM + Combine。今天精读 `fused_experts`。

### 学习任务 2：`fused_experts` 的三段式架构（45 分钟）

这是 Day 5 的**核心精读**内容——理解三段式架构才能读懂后续 kernel。

#### 三段式架构总览

读 `vllm/model_executor/layers/fused_moe/fused_moe.py`，`fused_experts` 把 MoE FFN 拆成三段：

```python
def fused_experts(x, w1, w2, topk_weights, topk_indices, inplace=False):
    """
    x: [T, d]
    w1: [N, K_ffn, d]   (第一层权重，gate + up 拼接)
    w2: [N, d, K_ffn]   (第二层权重)
    topk_weights: [T, K]
    topk_indices: [T, K]
    返回: [T, d]
    """
    # ---- 阶段 1: Dispatch（把 token 散到 expert 分组）----
    sorted_token_ids, expert_indices, num_tokens_post_padded = moe_align_block_size(
        topk_indices, num_experts, top_k, BLOCK_M
    )
    # sorted_token_ids: [max_num_tokens_per_expert * N] —— 按 expert 排序后的 token 索引
    # num_tokens_post_padded: 每 expert padding 到 BLOCK_M 倍数后的 token 数
    
    # ---- 阶段 2: GEMM（两层 FFN + SiLU 融合）----
    # 第一层 GEMM: x @ w1.T → [max_m * N, 2 * K_ffn]
    intermediate_cache = fused_experts_gemm1(x, w1, sorted_token_ids, ...)
    # SiLU 激活
    intermediate_cache = silu_and_mul(intermediate_cache)
    # 第二层 GEMM: h @ w2.T → [max_m * N, d]
    intermediate_cache = fused_experts_gemm2(intermediate_cache, w2, sorted_token_ids, ...)
    
    # ---- 阶段 3: Combine（把专家输出散回原位置 + 加权）----
    output = fused_experts_combine(intermediate_cache, topk_weights, sorted_token_indices, ...)
    return output
```

#### 三段的职责

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|---------|
| **Dispatch** | `topk_indices [T, K]` | `sorted_token_ids [max_m * N]` | 按 expert 排序 token + padding 到 BLOCK_M 倍数 |
| **GEMM 1** | `x [T, d]`, `w1 [N, 2K_ffn, d]` | `intermediate [max_m * N, 2K_ffn]` | Grouped GEMM（每 expert 一个 block） |
| **SiLU** | `intermediate [max_m * N, 2K_ffn]` | `intermediate [max_m * N, K_ffn]` | SiLU(gate) * up |
| **GEMM 2** | `intermediate [max_m * N, K_ffn]`, `w2 [N, d, K_ffn]` | `intermediate [max_m * N, d]` | Grouped GEMM |
| **Combine** | `intermediate [max_m * N, d]` | `output [T, d]` | 散回原位置 + 加权 |

> 💡 **关键洞察**：vLLM 用的是 **padded 布局**（[Day 3](../moe/day3.md) 学习任务 2 讲过）——每 expert padding 到 `BLOCK_M` 倍数，而不是 Day 3 的 contiguous 布局。这让 GEMM kernel 更简单（固定 BLOCK_M tile），但浪费一些显存与计算（padding 部分算 0）。推理场景下这个浪费可接受（batch 小），训练则用 contiguous。

#### `moe_align_block_size`：dispatch 的核心

读 `moe_align_block_size.py`，这个 kernel 把 `topk_indices [T, K]` 转成按 expert 排序的 `sorted_token_ids`：

```python
# moe_align_block_size.py（简化）
@triton.jit
def moe_align_block_size_kernel(
    topk_indices_ptr, sorted_token_ids_ptr, expert_indices_ptr,
    num_tokens_post_padded_ptr,
    num_experts, top_k, BLOCK_M: tl.constexpr,
):
    """按 expert 排序 token，每 expert padding 到 BLOCK_M 倍数。"""
    # 1. 统计每 expert 的 token 数
    # 2. 计算每 expert padding 后的 token 数（ceil_div(count, BLOCK_M) * BLOCK_M）
    # 3. 按 expert 顺序写入 sorted_token_ids
    #    - expert 0 的 token（padding 到 BLOCK_M 倍数）
    #    - expert 1 的 token（padding 到 BLOCK_M 倍数）
    #    - ...
    # 4. 记录 expert_indices（sorted_token_ids 每个位置属于哪个 expert）
```

输出布局：

```
sorted_token_ids: [max_num_tokens_per_expert * N]
┌──────────────────────────────┬──────────────────────────────┬─────┐
│ expert 0                     │ expert 1                     │ ... │
│ [token_id_0, token_id_1,     │ [token_id_5, token_id_8,     │     │
│  ..., padding, padding]      │  ..., padding, padding]      │     │
│  (padding 到 BLOCK_M 倍数)    │  (padding 到 BLOCK_M 倍数)    │     │
└──────────────────────────────┴──────────────────────────────┴─────┘
```

> ⚠️ **与 Day 3 的区别**：Day 3 用 contiguous 布局（无 padding，`group_offsets` 标记边界），vLLM 用 padded 布局（每 expert padding 到 BLOCK_M 倍数，`sorted_token_ids` 是扁平数组）。padded 让 GEMM tile 分配更简单（每 expert 的 tile 数是 `count // BLOCK_M` 的整数倍），但浪费 padding 计算。

### 学习任务 3：Triton MoE GEMM Kernel 精读（45 分钟）

这是 vLLM fused_moe 的**核心 kernel**——处理 padded 布局的 Grouped GEMM。

#### kernel 设计

读 `fused_moe.py` 的 `fused_experts_gemm1`（第一层 GEMM）：

```python
@triton.jit
def fused_experts_gemm1_kernel(
    x_ptr, w1_ptr, intermediate_ptr,
    sorted_token_ids_ptr, expert_indices_ptr,
    d: tl.constexpr, K_ffn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """第一层 GEMM: intermediate = x @ w1.T
    每个 program 处理一个 [BLOCK_M, BLOCK_N] tile。
    """
    pid_m = tl.program_id(0)   # 沿 M（token）的 block 编号
    pid_n = tl.program_id(1)   # 沿 N（K_ffn）的 block 编号
    
    # ---- 关键：通过 sorted_token_ids 找到当前 tile 属于哪个 expert ----
    # sorted_token_ids 每 BLOCK_M 个 token 一组，对应一个 expert
    expert_id = tl.load(expert_indices_ptr + pid_m * BLOCK_M // BLOCK_M)  # 简化
    
    # ---- 加载 token indices ----
    token_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + token_offsets, mask=token_offsets < max_tokens)
    
    # ---- 加载 x_tile: [BLOCK_M, d] ----
    x_offsets = token_ids[:, None] * d + tl.arange(0, d)[None, :]
    x_tile = tl.load(x_ptr + x_offsets, mask=token_ids[:, None] < num_tokens, other=0.0)
    
    # ---- 加载 w1_tile: [BLOCK_N, d]（从 expert_id 对应的权重）----
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w1_offsets = expert_id * K_ffn * d + n_offsets[:, None] * d + tl.arange(0, d)[None, :]
    w1_tile = tl.load(w1_ptr + w1_offsets)
    
    # ---- K 维分块累加（d 通常很大，分块）----
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, d, BLOCK_K):
        x_block = tl.load(...)   # [BLOCK_M, BLOCK_K]
        w_block = tl.load(...)   # [BLOCK_N, BLOCK_K]
        accum += tl.dot(x_block, w_block.T)
    
    # ---- 写回 ----
    out_offsets = pid_m * BLOCK_M * K_ffn + n_offsets  # 简化
    tl.store(intermediate_ptr + out_offsets, accum.to(intermediate_ptr.dtype.element_ty))
```

#### 关键设计：`sorted_token_ids` 间接寻址

vLLM 的核心技巧是**通过 `sorted_token_ids` 间接寻址**——不是直接 `x[m_offset:m_offset+BLOCK_M]`，而是 `x[sorted_token_ids[pid_m * BLOCK_M : ...]]`：

```python
token_ids = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
x_tile = tl.load(x_ptr + token_ids[:, None] * d + ...)
```

- `token_ids` 是 `[BLOCK_M]` 的索引数组，指向 `x` 的原始行
- padding 位置的 `token_id` 设为 `num_tokens`（超出范围），`mask=token_ids < num_tokens` 把它们填 0
- 这样 GEMM tile 可以跨 expert 边界（只要在同一 expert 的 padded 块内），但不会跨 expert

#### 与 Day 3 的对比

| 维度 | Day 3 Triton | vLLM fused_moe |
|------|-------------|----------------|
| **数据布局** | contiguous（无 padding） | padded（每 expert padding 到 BLOCK_M 倍数） |
| **tile 分配** | host 预计算 3 个表 | device 端用 `sorted_token_ids` 间接寻址 |
| **expert 归属** | `tile_expert[tile_id]` 查表 | `expert_indices[pid_m]` 查表 |
| **x 访问** | `x[m_offset:m_offset+BLOCK_M]`（连续） | `x[sorted_token_ids[...]]`（间接） |
| **边界处理** | mask 处理最后一个 tile | padding 位置填 0 |
| **显存浪费** | 无 | padding 部分 |

> 💡 **为什么 vLLM 用 padded**：① 推理 batch 小，padding 浪费可接受；② `sorted_token_ids` 间接寻址让 GEMM kernel 不需要处理变长（每 expert 都是 BLOCK_M 倍数），tile 分配更简单；③ `sorted_token_ids` 可以在 device 端用单个 Triton kernel（`moe_align_block_size`）生成，无需 host 预计算。

#### tile 分配的 swizzle

vLLM 的 GEMM kernel 用 `pid_m` 和 `pid_n` 两维并行，默认顺序可能导致 L2 不友好。vLLM 用类似的 swizzle 技巧：

```python
# 简化的 swizzle
num_pid_m = tl.cdiv(total_tokens, BLOCK_M)
num_pid_n = tl.cdiv(K_ffn, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

这与 [Day 3 学习任务 4](../moe/day3.md) 的 swizzle 同理——组内沿 N 排开复用 A。

### 学习任务 4：dispatch/combine Triton Kernel 精读（30 分钟）

vLLM 的 dispatch 与 combine 不是用 `all_to_all`（单卡推理无 EP），而是用 **Triton kernel 做 scatter-gather**。

#### Dispatch kernel（`moe_align_block_size`）

dispatch 的核心是 `moe_align_block_size` kernel（学习任务 2 已讲），它生成 `sorted_token_ids`。但实际的 token 数据搬运（`x` 按 `sorted_token_ids` 重组）融合进了 GEMM kernel——GEMM kernel 直接用 `sorted_token_ids` 间接寻址 `x`，**不显式搬运 token**。

> 💡 **关键设计**：vLLM 的 dispatch **不显式搬数据**——只生成 `sorted_token_ids` 索引数组，GEMM kernel 用间接寻址读 `x`。这比 Day 3 的"先 sort 再 GEMM"少一次显存往返，但 GEMM kernel 的访存模式更复杂（非连续 `x[sorted_token_ids[...]]`）。

#### Combine kernel

```python
@triton.jit
def fused_experts_combine_kernel(
    intermediate_ptr, output_ptr,
    topk_weights_ptr, sorted_token_ids_ptr,
    num_tokens, d, K,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """把专家输出散回原 token 位置，按 topk_weight 加权累加。"""
    pid_m = tl.program_id(0)   # 沿 token 的 block
    
    # 加载当前 block 的 token indices
    token_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = token_offsets < num_tokens
    
    # 对每个 token，累加 K 个专家的输出
    accum = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for k in range(K):
        # 找到该 token 第 k 个专家在 intermediate 中的位置
        expert_offset = ...  # 通过 sorted_token_ids 反查
        expert_output = tl.load(intermediate_ptr + expert_offset + ...)
        
        # 加权累加
        weight = tl.load(topk_weights_ptr + token_offsets * K + k)
        accum += weight[:, None] * expert_output
    
    tl.store(output_ptr + token_offsets[:, None] * d + tl.arange(0, BLOCK_D)[None, :], 
             accum, mask=mask[:, None])
```

- combine 用 `topk_weights` 加权累加 K 个专家的输出
- `sorted_token_ids` 的反向查找——从 `token_id` 找到它在 `intermediate` 中的位置

#### 与 Day 3 的 combine 对比

| 维度 | Day 3 | vLLM |
|------|-------|------|
| **combine 方式** | `output[token_src_idx] += score * sorted_output` | Triton kernel 加权累加 |
| **显存往返** | 先 gather 再加权 | 融合成单 kernel |
| **kernel 数** | 1 个 PyTorch 操作 | 1 个 Triton kernel |

### 学习任务 5：w1/w2 两层 GEMM + SiLU 融合（30 分钟）

MoE FFN 是两层 GEMM + SiLU 激活：

$$\text{FFN}(x) = (W_2) \cdot \text{SiLU}(x \cdot W_1^{\text{gate}}) \odot (x \cdot W_1^{\text{up}})$$

vLLM 把 gate 和 up 权重拼接成 `w1: [N, 2*K_ffn, d]`，第一层 GEMM 输出 `[max_m * N, 2*K_ffn]`，再用 `silu_and_mul` 融合激活。

#### `silu_and_mul` kernel

```python
@triton.jit
def silu_and_mul_kernel(x_ptr, out_ptr, K_ffn, BLOCK_SIZE: tl.constexpr):
    """SiLU(gate) * up，gate 和 up 在 x 的前半和后半。"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # gate 在前 K_ffn，up 在后 K_ffn
    gate = tl.load(x_ptr + offsets)                    # [BLOCK_SIZE]
    up = tl.load(x_ptr + offsets + K_ffn)              # [BLOCK_SIZE]
    
    # SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
    silu_gate = gate * tl.sigmoid(gate)
    
    out = silu_gate * up
    tl.store(out_ptr + offsets, out)
```

#### 为什么不融合进 GEMM

vLLM 把 SiLU 作为独立 kernel 而非融合进 GEMM epilogue，原因是：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **独立 SiLU kernel**（vLLM） | GEMM kernel 简单，可复用 | 多一次显存往返（写 intermediate → 读 → 写） |
| **融合进 GEMM epilogue**（DeepGEMM Mega MoE） | 省一次显存往返 | GEMM kernel 复杂，需定制 epilogue |

> 💡 **DeepGEMM Mega MoE 的做法**：[Day 6 学习任务 7](../deepgemm/day6.md) 讲过，Mega MoE 把 SwiGLU 融合进 L1 GEMM 的 epilogue——TMEM load → SwiGLU → FP8 cast → 写回，单 kernel 完成。vLLM 没融合是因为 Triton 的 epilogue 定制能力弱于手写 PTX，且推理场景显存往返开销可接受。

### 学习任务 6：与 Day 2-3 / Mega MoE / Megatron 对比（30 分钟）

#### 四种 MoE 实现的横向对比

| 维度 | Day 2-3 自写 | vLLM fused_moe | DeepGEMM Mega MoE | Megatron |
|------|-------------|----------------|---------------------|----------|
| **场景** | 学习 | 单卡推理 | 多卡推理（EP） | 训练（EP） |
| **数据布局** | contiguous | padded | contiguous + ring buffer | contiguous |
| **Gating** | 独立 Triton kernel | 独立（`topk.py`） | 融合进 dispatch warp | 独立 |
| **Dispatch** | host 预排序 | `moe_align_block_size` | warp specialization + NVLink | NCCL all-to-all |
| **GEMM** | Triton `tl.dot` | Triton `tl.dot` | `tcgen05.mma` + TMEM | cuBLAS 逐专家 |
| **SiLU** | 独立 | 独立 `silu_and_mul` | 融合进 epilogue | 独立 |
| **Combine** | PyTorch gather | Triton kernel | NVLink 写远端 + reduce | NCCL all-to-all |
| **EP 通信** | 无 | 无 | symmetric memory + NVLink | NCCL |
| **融合程度** | 低（多 kernel） | 中（3-4 kernel） | 高（单 kernel） | 低（多 kernel + 多 stream） |
| **性能** | 学习级 | 生产级（推理） | 极致（推理） | 生产级（训练） |

#### vLLM 与 Day 2-3 的差距

| 优化点 | Day 2-3 | vLLM | 差距 |
|--------|---------|------|------|
| **Gating 融合** | ✓（Day 2 单 kernel） | ✓（`topk.py`） | 相当 |
| **Grouped GEMM tile 分配** | host 预计算表 | device 端 `sorted_token_ids` | vLLM 更灵活 |
| **dispatch 显存往返** | 先 sort 再 GEMM（2 次） | 间接寻址（1 次） | vLLM 省 1 次 |
| **SiLU 融合** | ✗ | ✗ | 都没融合 |
| **combine 融合** | PyTorch gather | Triton kernel | vLLM 更快 |
| **FP8 支持** | ✗ | ✓（`fp8_w8a8`） | vLLM 支持量化 |

> 💡 **Day 6 的改进方向**：基于 vLLM 的设计，Day 6 组装完整 MoE 时可以：① 用 `sorted_token_ids` 间接寻址替代 Day 3 的 host 预计算；② 把 combine 融合成 Triton kernel；③ 考虑 FP8 量化支持。但 SiLU 融合进 GEMM epilogue 在 Triton 里较难，保持独立 kernel。

#### 单卡融合 vs 跨卡融合

| 维度 | vLLM（单卡融合） | DeepGEMM Mega MoE（跨卡融合） |
|------|----------------|------------------------------|
| **适用** | 单卡放得下全部专家 | 专家分布在多卡（EP） |
| **通信** | 无 | NVLink + symmetric memory |
| **融合粒度** | 3-4 个 Triton kernel | 单 kernel + warp specialization |
| **SiLU** | 独立 kernel | 融合进 epilogue |
| **dispatch** | `sorted_token_ids` 间接寻址 | NVLink 拉远端 token |
| **combine** | Triton kernel 加权 | NVLink 写远端 + reduce |
| **延迟** | ~200us（单卡） | ~280us（8 卡，但算力 8x） |
| **代表** | Mixtral 8x7B 推理 | DeepSeek-V3 推理 |

### 面试题积累（本周目标 10-12 道，今日 3 道）

**Q13：vLLM `fused_moe` 的三段式架构是什么？为什么用 padded 布局而非 contiguous？**
> 答：三段式：① Dispatch——`moe_align_block_size` kernel 把 `topk_indices` 转成按 expert 排序的 `sorted_token_ids`（每 expert padding 到 BLOCK_M 倍数）；② GEMM——两层 Triton Grouped GEMM（w1 + SiLU + w2），用 `sorted_token_ids` 间接寻址；③ Combine——Triton kernel 把专家输出散回原位置并按 `topk_weights` 加权。用 padded 布局的原因：① 推理 batch 小，padding 浪费可接受；② padded 让每 expert 的 tile 数是 BLOCK_M 倍数，GEMM tile 分配更简单；③ `sorted_token_ids` 在 device 端用单 kernel 生成，无需 host 预计算。训练场景 batch 大用 contiguous（省 padding 浪费）。

**Q14：vLLM 的 GEMM kernel 怎么通过 `sorted_token_ids` 间接寻址？为什么比 Day 3 的"先 sort 再 GEMM"省显存？**
> 答：vLLM 不显式搬运 token 数据，只生成 `sorted_token_ids` 索引数组。GEMM kernel 内 `token_ids = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_M + ...)`，然后 `x_tile = tl.load(x_ptr + token_ids[:, None] * d + ...)`——用索引间接寻址原始 `x`。padding 位置的 `token_id` 超出范围，`mask` 填 0。比 Day 3 省 1 次显存往返：Day 3 先 `sorted_x = x[sorted_token_idx]`（1 次写），再 GEMM 读 `sorted_x`（1 次读）= 2 次；vLLM 直接 GEMM 读 `x[sorted_token_ids]`（1 次间接读）= 1 次。代价是 GEMM 的访存模式非连续（`x[token_ids[...]]`），L2 命中率可能降低。

**Q15：vLLM 的单卡融合与 DeepGEMM Mega MoE 的跨卡融合有什么区别？**
> 答：五个维度：① **适用**——vLLM 单卡放得下全部专家（Mixtral 8x7B），Mega MoE 专家分布在多卡（DeepSeek-V3 671B）；② **通信**——vLLM 无通信，Mega MoE 用 NVLink + symmetric memory；③ **融合粒度**——vLLM 3-4 个 Triton kernel，Mega MoE 单 kernel + warp specialization；④ **SiLU**——vLLM 独立 kernel，Mega MoE 融合进 GEMM epilogue（TMEM load → SwiGLU → FP8 cast）；⑤ **dispatch/combine**——vLLM 用 `sorted_token_ids` 间接寻址（无数据搬运），Mega MoE 用 NVLink 拉远端 token + 写远端 combine buffer。本质上 vLLM 是"算子级融合"，Mega MoE 是"warp 级融合 + 通信/计算 overlap"——后者更极致但实现复杂度也更高。

### 今日检查清单

- [ ] 能说出 vLLM fused_moe 的三段式架构（dispatch + GEMM + combine）
- [ ] 能解释为什么 vLLM 用 padded 布局（推理 batch 小 + tile 分配简单 + device 端生成）
- [ ] 理解 `moe_align_block_size` kernel 的作用（生成 `sorted_token_ids`）
- [ ] 能画出 `sorted_token_ids` 的内存布局（按 expert 排序 + padding 到 BLOCK_M 倍数）
- [ ] 能写出 vLLM GEMM kernel 的间接寻址（`x[sorted_token_ids[...]]`）
- [ ] 理解 vLLM dispatch 不显式搬数据（只生成索引，GEMM 间接寻址）
- [ ] 能解释 vLLM 比Day 3 省 1 次显存往返的原因
- [ ] 能写出 `silu_and_mul` kernel 的逻辑（gate/up 拼接 + SiLU + 乘法）
- [ ] 理解 vLLM 不融合 SiLU 进 GEMM 的原因（Triton epilogue 定制能力弱）
- [ ] 能对比 vLLM / Day 2-3 / Mega MoE / Megatron 的 5 个维度（场景/布局/融合/通信/性能）
- [ ] 能说出单卡融合 vs 跨卡融合的 5 个区别
- [ ] 浏览了 vLLM `vllm/model_executor/layers/fused_moe/` 目录结构
- [ ] 读懂 `fused_moe.py` 的 `fused_experts` 函数签名与三段调用

#### 明日预告

Day 6 将**组装完整 Triton MoE FFN 层**——把 Day 2 的 Gating+Top-K、Day 3 的 Grouped GEMM、Day 5 学到的 vLLM 设计（`sorted_token_ids` 间接寻址 + combine 融合）拼成一个端到端的 MoE FFN。今天读完了 vLLM 的"参考答案"，明天要自己写一份。目标性能达到 Megatron-LM 参考实现的 70%+（验收 ⑥），并产出性能对比脚本。建议今晚回顾 Day 2-3 的 kernel 代码，思考如何用 vLLM 的间接寻址改进 Day 3 的 host 预计算方案。

---
