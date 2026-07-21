# Day 7（周日）：进阶专题与面试复盘

> **今日目标**：了解 MoE 前沿方向（auxiliary-loss-free 均衡、Llama-4 / DeepSeek-V3 架构、多 token 预测），完成面试复盘
> **面试考察度**：⭐⭐⭐ 了解级，能说出进阶方向

---

### 学习任务 1：进阶专题速览（60 分钟）

#### Auxiliary-Loss-Free 负载均衡（DeepSeek-V3）

DeepSeek-V3 抛弃了 V2 的三级 aux loss，改用**偏置项调整**：给门控 logits 加一个可学习的 bias $b_i$，bias 不参与前向计算只影响 argmax 路由，根据专家负载动态增减。

| 维度 | V2 三级 aux loss | V3 bias 调整 |
|------|------------------|--------------|
| 梯度通路 | 通过 $P_i$ 反传 | 无梯度，纯控制信号 |
| 超参 | 多个 loss 系数 | 单一 bias 更新率 |
| 性能 | aux loss 占总 loss 一定比例 | 零额外计算 |

#### DeepSeek-V3 的其他创新

- **FP8 训练**：MoE 专家权重的 FP8 GEMM，配合细粒度量化
- **多 Token 预测（MTP）**：一次预测多个 token，与 MoE 路由结合做投机解码
- **671B 总参 / 37B 激活**：256 路由专家 + 1 共享专家，Top-8

#### Llama-4 / Qwen-MoE 架构要点

| 模型 | 总参 | 激活 | 专家数 | 路由 | 特色 |
|------|------|------|--------|------|------|
| Mixtral 8x7B | 47B | 13B | 8 | Top-2 | 开源标杆 |
| DeepSeek-V3 | 671B | 37B | 256+1 | Top-8 | 细粒度 + 共享 + FP8 |
| Qwen-MoE | 14B | 2.7B | 60 | Top-4 | 共享专家 + 细粒度 |

### 学习任务 2：面试题复盘（60 分钟）

#### 高频面试题

1. **MoE 为什么能省算力？代价是什么？**
   - 每 token 只激活 K/N 个专家，计算量 $\propto$ 激活参数而非总参数
   - 代价：显存仍随总参数线性增长；专家并行引入 all-to-all 通信；负载不均需额外机制

2. **Top-1 和 Top-2 路由的优劣？**
   - Top-1（Switch）：算力最省，但路由抖动大，训练不稳
   - Top-2（GShard/Mixtral）：算力稍多，但更稳，是当前主流

3. **负载均衡损失为什么用 $f_i \cdot P_i$？**
   - $f_i$ 是硬统计不可导，$P_i$ 是软概率可导
   - 乘积让不可导的负载信号通过可导的门控概率反传梯度

4. **Grouped GEMM 比逐专家 GEMM 快在哪？**
   - 减少 kernel launch（$O(N)$ → 1）
   - SM 跨专家调度，负载均衡
   - 适合专家数多、每专家 token 中等的场景

5. **MoE 的 EP 通信为什么是 all-to-all 而不是 all-reduce？**
   - EP 把专家分散到不同卡，token 需发往目标专家卡 → 通信模式是"每对卡都有不同数据" → all-to-all
   - TP 是把权重切分，每卡算部分结果后合并 → all-reduce

6. **DeepSeek 的设备受限路由解决什么问题？**
   - 朴素 EP 下每 token 可能发往 $K_r$ 台设备，通信量 $O(K_r \cdot D)$
   - 限制到 $M$ 台后通信量 $O(M)$，代价是丢弃超出 $M$ 的专家选择

7. **DeepSeek-V3 为什么去掉 auxiliary loss？**
   - aux loss 占总 loss 一部分，可能干扰主任务梯度
   - bias 调整无梯度，纯控制信号，不污染主损失
   - 实验显示负载均衡效果相当甚至更好

8. **MoE 推理的 decode 阶段为什么是性能杀手？**
   - decode 每步 batch 小（1 token/请求），专家 GEMM 退化为瘦 GEMM
   - SM 利用率低，通信占比飙升
   - 解决：跨请求合并 batch、专家预取、共享专家并行

9. **MoE 的共享专家有什么用？**
   - 所有 token 必过，装通用知识
   - 缓解路由专家间的知识冗余（不同专家不必重复学通用模式）
   - 可与路由专家的 Grouped GEMM 并行做

10. **Triton 实现 Grouped GEMM 的关键点？**
    - 用 `pid` 反查 expert + tile id（预计算 tile_to_expert 表）
    - 各专家 tile 数不同，需要变长调度
    - BLOCK_M 选择需平衡小专家（tile 少）与大专家（tile 多）

11. **CUTLASS Group GEMM 与 Triton Grouped GEMM 的取舍？**
    - CUTLASS：C++ 模板，编译期优化，适合专家少、每专家 token 多
    - Triton：Python DSL，灵活，适合专家多、每专家 token 少（DeepSeek 细粒度场景）
    - vLLM 选 Triton 主因是 DeepSeek 系的 256 专家场景

12. **MoE 与投机解码（speculative decoding）如何结合？**
    - 草稿模型可以是共享专家子集（小 GEMM）
    - 验证阶段用完整 MoE，利用多 token 预测（MTP）加速
    - DeepSeek-V3 的 MTP 本质是把投机解码训练化

### 学习任务 3：总结与知识图谱（30 分钟）

#### 本周知识图谱

```
                        MoE（Mixture-of-Experts）
                       /          |             \
                 算法              算子              系统
                /  |  \          /  |  \          /  |  \
        稀疏门控  负载均衡  容量   Gating  Grouped  Dispatch  EP   推理    通信
          |       |        |    TopK    GEMM     Scatter   all-to-all
        Top-K    aux loss  cap   |       |        |          |
        Token   f*P 乘积  1.25   Triton  CUTLASS  argsort   NCCL
        Choice   |              融合     GemmGroup + gather  |
        Expert   bias(V3)       kernel           |          设备受限
        Shared                  |               group_offsets 路由(M)
          |                     |                 |          |
          +---------------------+-----------------+----------+
                                |
                        vLLM fused_moe / Megatron MoE
                                |
                        ncu/nsys 调优
                                |
                DeepSeek-V3 / Llama-4 / Mixtral
```

#### 推荐资源

| 资源 | 类型 | 优先级 |
|------|------|--------|
| [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md)（本仓库） | 论文 | ⭐ 必读 |
| [Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) Shazeer 2017 | 论文 | ⭐ 必读 |
| [GShard](https://arxiv.org/abs/2006.16668) | 论文 | ⭐ 必读 |
| [Switch Transformer](https://arxiv.org/abs/2101.03961) | 论文 | 📌 推荐 |
| [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | 论文 | 📌 推荐 |
| [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) | 论文 | 📌 推荐 |
| [vLLM `fused_moe.py`](https://github.com/vllm-project/vllm) | 源码 | ⭐ 必读 |
| [Megatron-LM MoE](https://github.com/NVIDIA/Megatron-LM) `megatron/core/transformer/moe/` | 源码 | 📌 推荐 |
| [CUTLASS 专题 Day 7](../cutlass/day7.md) Group GEMM | 教程 | 📎 复习前置 |
| [Week6 Day1 LeetGPU MoE Top-K Gating 题解](../../daily/week6/day1/README.md) | 练习 | 📎 参考 |

