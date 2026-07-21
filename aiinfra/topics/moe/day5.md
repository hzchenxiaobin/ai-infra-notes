# Day 5（周五）：MoE 推理优化与 vLLM `fused_moe` 精读

> **今日目标**：理解 MoE 推理的特殊性（动态路由 + KV cache + 专家预取），精读 vLLM 的 `fused_moe.py` 工程实现
> **面试考察度**：⭐⭐⭐⭐ 实践级，能说清推理与训练的 MoE 差异

---

### 学习任务 1：MoE 推理的特殊性（30 分钟）

| 维度 | 训练 | 推理 |
|------|------|------|
| batch size | 大（千 token） | 小（1）/ 动态 |
| 容量因子 | 1.0-1.5（丢 token） | 不丢（必须服务所有请求） |
| 路由 | 每 token 独立 top-k | 同上，但需考虑 prefill/decode |
| 通信 | EP all-to-all | EP all-to-all，但 decode 阶段 batch 小 |
| KV cache | 不涉及 | 与 MoE 层交织 |

#### Prefill vs Decode 的 MoE 差异

- **Prefill**：大 batch（万 token），Grouped GEMM 高效，all-to-all 通信占比低
- **Decode**：batch 小（每步 1 token/请求），专家计算变成小 GEMM，通信占比飙升

> 💡 **关键洞察**：MoE 推理的 decode 阶段是性能杀手——每步只算几个 token，专家 GEMM 退化为 `d_model × d_ff` 的瘦 GEMM，SM 利用率极低。vLLM 的做法是跨请求合并 batch + 专家预取。

### 学习任务 2：vLLM `fused_moe` 精读（60 分钟）

阅读 [`vllm/model_executor/layers/fused_moe/fused_moe.py`](https://github.com/vllm-project/vllm) 的核心函数 `fused_experts`：

```python
# vLLM fused_experts 核心结构（简化）
def fused_experts(x, w1, w2, topk_idx, topk_val, ...):
    # 1. 按 expert 重新排列 token（dispatch）
    x_dispatched, group_offsets = moe_align_block_size(topk_idx, ...)
    # 2. 第一层 GEMM（Grouped GEMM: x @ w1.T）
    h = grouped_gemm(x_dispatched, w1, group_offsets, ...)
    # 3. 激活
    h = act_fn(h)
    # 4. 第二层 GEMM（Grouped GEMM: h @ w2.T）
    out = grouped_gemm(h, w2, group_offsets, ...)
    # 5. 加权 combine 回原 token 顺序
    return moe_combine(out, topk_idx, topk_val, ...)
```

#### 关键优化点

| 优化 | 位置 | 收益 |
|------|------|------|
| `moe_align_block_size` | dispatch | 用 Triton kernel 把 token 按 expert 排序 + padding 到 BLOCK 对齐 |
| Grouped GEMM（Triton） | 两个 GEMM | 1 次 launch 处理所有专家 |
| 激活融合 | GEMM 之间 | SiLU/GELU 与第二层 GEMM 的 epilogue 融合 |
| `moe_combine` | combine | 加权求和与 scatter 回原顺序融合 |

### 学习任务 3：DeepSeek-MoE 的推理优化（45 分钟）

复习 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) 中推理相关内容，重点关注：

1. **共享专家不参与路由**：2 个共享专家对所有 token 必过，可以与路由专家的 Grouped GEMM 并行做（一个大 GEMM + 一个 Grouped GEMM）
2. **细粒度专家的负载更均衡**：256 个小专家天然分散，decode 阶段即使 1 token 也能激活 8 个专家，SM 利用率比 8 大专家高
3. **MLA + MoE 的交织**：KV Cache 压缩后，MoE 层的显存占比相对升高，量化（FP8/INT4）更激进

### 学习任务 4：专家预取与计算-通信 overlap（30 分钟）

EP 推理时，next layer 的门控可以与当前 layer 的专家计算 overlap：

```
时间 →
Layer N:   [门控 N+1] --dispatch--> [专家 N] --combine-->
Layer N+1:               [门控 N+2] --dispatch--> [专家 N+1] ...
                         ↑ overlap 区间
```

Megatron-LM 的 `transformer/moe/moe_layer.py` 实现了这种 overlap——用 CUDA stream 把门控计算与上一层 combine 放在不同 stream。

### 今日检查清单

- [ ] 能说出 MoE 推理在 prefill / decode 阶段的不同瓶颈
- [ ] 读完 vLLM `fused_experts` 的主流程，能画出 5 个步骤
- [ ] 能解释 DeepSeek 共享专家为何可与路由专家并行
- [ ] 能说出专家预 fetch 的 overlap 原理
- [ ] 在本地跑一个 vLLM MoE 推理 demo（Mixtral 或 DeepSeek-V3）

---

