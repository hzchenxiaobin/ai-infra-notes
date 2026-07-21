# Day 4（周四）：Expert Parallelism 与 All-to-All 通信

> **今日目标**：理解 Expert Parallelism（EP）的通信模式，掌握 all-to-all token dispatch 的时序，能解释 EP / TP / DP 的组合策略
> **面试考察度**：⭐⭐⭐⭐⭐ 核心考点，"MoE 怎么做多卡并行"必问

---

### 学习任务 1：为什么需要 Expert Parallelism（30 分钟）

#### MoE 的并行选择

| 并行策略 | 切分对象 | 通信 | 适用 |
|----------|----------|------|------|
| DP（数据并行） | 输入 batch | 梯度 all-reduce | 小模型 |
| TP（张量并行） | 权重矩阵 | 每层 all-reduce | 稠密大模型 |
| **EP（专家并行）** | 专家 | **all-to-all** | MoE 专属 |

#### EP 的核心思想

把 $N$ 个专家分散到 $D$ 张卡上，每卡 $\frac{N}{D}$ 个专家。每个 token 算完门控后，根据 top-k 结果 **发送到目标专家所在的卡**，算完再发回来。

```
GPU 0: 专家 0, 1, 2, 3     GPU 1: 专家 4, 5, 6, 7
         ↑ 收 token 0,2,5            ↑ 收 token 1,3,4,7
         | 算 FFN                     | 算 FFN
         ↓ 发回原卡                    ↓ 发回原卡
```

> 💡 **一句话总结**：EP 的通信是两次 all-to-all——dispatch 时把 token 按路由发到专家卡，combine 时把结果发回原卡。这两次 all-to-all 是 MoE 训练/推理的核心通信开销，也是 DeepSeek 提出"设备受限路由"压制的目标。

### 学习任务 2：All-to-All 通信时序（45 分钟）

![MoE Expert Parallelism 的双 all-to-all 时序](../images/moe_ep_alltoall_timeline.svg)

#### EP 前向完整时序

```
时间 →
GPU 0:  [门控] --dispatch all-to-all--> [专家 FFN] --combine all-to-all--> [加权求和]
GPU 1:  [门控] --dispatch all-to-all--> [专家 FFN] --combine all-to-all--> [加权求和]
                                         ↑↑↑
                              通信与计算可 overlap（pipeline）
```

#### NCCL all-to-all API

```python
import torch.distributed as dist

# 每张卡有 send/recv 计数表：send_counts[i] = 要发给卡 i 的 token 数
# dispatch 阶段
dist.all_to_all_single(
    output_tensor=recv_buffer,    # [recv_total, d_model]
    input_tensor=send_buffer,     # [send_total, d_model]
    output_split_sizes=recv_counts,   # 每张卡发给本卡多少
    input_split_sizes=send_counts,    # 本卡发给每张卡多少
    group=ep_group,
)
```

> ⚠️ **注意**：`all_to_all_single` 要求 `send_counts` 和 `recv_counts` 在所有卡上**预先通过 allgather 同步**——因为 NCCL 需要知道每对 (src, dst) 的数据量才能调度。这步额外的 allgather 在 token 数动态变化时是隐式开销。

### 学习任务 3：设备受限路由（DeepSeek-V2）（45 分钟）

复习 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) §5.4 的设备受限路由。

#### 问题：EP 的 all-to-all 规模

设 $D$ 张卡，每 token 激活 $K_r$ 个专家。朴素 EP 下，每个 token 可能发往 $K_r$ 个不同设备——all-to-all 通信量为 $O(K_r \cdot D)$ 的 scatter pattern。

#### DeepSeek 的限制：每 token 最多发往 $M$ 台设备

$$M < K_r \cdot \frac{N}{D}$$

DeepSeek-V2 取 $M=3$，$K_r=6$，$N=160$，$D=8$：原本每 token 最坏发 6 台设备，限制到 3 台。代价是部分专家被"截断"——超出 $M$ 台的专家选择被丢弃。

| 策略 | 通信量 / token | 负载均衡 | 丢弃率 |
|------|----------------|----------|--------|
| 朴素 EP | $O(K_r)$ | 依赖 aux loss | 0 |
| 设备受限路由（M） | $O(M)$ | 设备级均衡损失 | >0 |

#### 三级负载均衡损失

DeepSeek-V2 用三层 aux loss 同时约束：专家级（每专家 token 数）、设备级（每设备 token 数）、通信级（每对设备 token 数）。详见 [DeepSeek-V2 论文精读](../../paper/deepseek_v2/README.md) §5.4。

### 学习任务 4：动手实现 2 卡 EP demo（60 分钟）

```python
# ep_demo.py —— 2 卡 Expert Parallelism 最小 demo
# 运行: torchrun --nproc_per_node=2 ep_demo.py
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

def main():
    dist.init_process_group('nccl', rank=int(os.environ['RANK']),
                            world_size=int(os.environ['WORLD_SIZE']))
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    T, D, N, K = 16, 64, 4, 2      # 2 卡各持 2 专家
    experts_per_gpu = N // 2
    local_experts = torch.randn(experts_per_gpu, D, D, device='cuda')

    x = torch.randn(T, D, device='cuda')
    gate = torch.randn(D, N, device='cuda')

    # 1. 门控（每卡独立算）
    scores = F.softmax(x @ gate, dim=-1)
    topk_val, topk_idx = scores.topk(K, dim=-1)

    # 2. 计算 send_counts: 本卡要发给每张卡多少 (token, expert) 对
    #    expert e 在卡 e // experts_per_gpu 上
    target_gpu = topk_idx // experts_per_gpu           # [T, K]
    send_counts = torch.bincount(target_gpu.reshape(-1), minlength=2)

    # 3. allgather recv_counts
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)

    # 4. dispatch all-to-all（省略 pack/unpack 细节）
    send_buf = pack_tokens(x, topk_idx, topk_val, target_gpu)
    recv_buf = torch.empty(recv_counts.sum().item(), D, device='cuda')
    dist.all_to_all_single(recv_buf, send_buf,
                           output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())

    # 5. 本地专家计算
    out_local = recv_buf @ local_experts[local_expert_ids].T

    # 6. combine all-to-all（反向）
    # ... 类似 dispatch，把结果发回原卡 ...

if __name__ == '__main__':
    main()
```

### 今日检查清单

- [ ] 能说出 EP / TP / DP 三种并行切分的对象与通信模式
- [ ] 能画出 EP 前向的两次 all-to-all 时序图
- [ ] 能解释 DeepSeek 设备受限路由为什么把通信从 $O(K_r)$ 降到 $O(M)$
- [ ] `ep_demo.py` 在 2 卡上跑通，结果与单卡一致
- [ ] 用 `nsys` 抓出 all-to-all 的通信耗时占比

---

