## Day 6：Prefill/Decode 分离推理（PD Disaggregated）

### 🎯 目标

通过今天的学习，你将：

1. 理解 **PD 分离的核心动机**——prefill（compute-bound）与 decode（memory-bound）的资源错配，TTFT/TPOT SLO 矛盾
2. 掌握 **Mooncake / DistServe / vLLM disaggregated** 的架构——KV 传输层、调度层、双池分离
3. 量化 **KV Cache 跨节点传输开销**——KV bytes/token × 序列长度 / RDMA 带宽
4. 通过模拟器对比 **colocated vs disaggregated** 的 TTFT/TPOT 改善，理解何时划算、何时不划算

> 💡 **为什么重要**：2024-2026 推理服务面试中，"PD 分离"几乎必问。它是 vLLM V1、Mooncake、DistServe 的核心架构，解决了长 prompt 与高并发 decode 的根本矛盾。理解 PD 分离 = 理解推理系统的资源调度本质。

---

### 1. 为什么分离？资源错配与 SLO 矛盾

#### 1.1 prefill vs decode 的资源画像

回顾 Day 1 与 Week3/Day1 的 Roofline 分析：

| 阶段 | 瓶颈 | 资源画像 | GPU 利用特征 |
|------|------|---------|------------|
| **Prefill** | compute-bound（大 GEMM，AI ≫ Ridge Point） | 吃算力，不吃带宽 | SM 利用率 60-85%，单次 forward 大块占用 GPU |
| **Decode** | memory-bound（M=1 GEMM，AI ≪ Ridge Point） | 吃带宽，不吃算力 | SM 利用率 10-30%，KV Cache 读取主导 |

**错配**：prefill 是"大块计算任务"，decode 是"小块访存任务"。两者混跑在同一 GPU 上：
- prefill 占用 SM 时，decode 的 slot 空等（compute-bound 任务不释放 SM）
- decode 请求多时，prefill 排队（memory-bound 任务占着 GPU 但 SM 闲置）

#### 1.2 TTFT / TPOT SLO 矛盾

| 指标 | 含义 | 受谁影响 |
|------|------|---------|
| **TTFT**（Time To First Token） | 首 token 延迟 | prefill 阶段（compute-bound） |
| **TPOT**（Time Per Output Token） | 每 token 延迟 | decode 阶段（memory-bound） |

**SLO 矛盾**：用户要 TTFT < 200ms（prefill 要快）+ TPOT < 50ms（decode 要稳）。但 colocated 下：
- 长 prompt 的 prefill 会阻塞 decode slot → TPOT 飙升
- 高并发 decode 会挤占 prefill 资源 → TTFT 排队

> 💡 **一句话总结**：PD 分离不是"为了快"，而是"为了 SLO 解耦"——让 prefill 和 decode 各自追求自己的最优，互不干扰。

---

### 2. PD 分离架构：Mooncake / DistServe / vLLM V1

#### 2.1 核心架构：双池 + KV 传输层

```
         ┌─────────────────────────────┐
请求 ──▶ │  Router / Scheduler         │
         │  （按阶段分发）              │
         └────────┬────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  ┌───────────┐       ┌───────────┐
  │ Prefill   │──────▶│ Decode    │
  │ Pool      │ KV    │ Pool      │
  │ (compute) │传输层  │ (memory)  │
  └───────────┘       └───────────┘
        │                   │
   prefill 完成后      接收 KV Cache
   发送 KV Cache       继续 decode
```

| 组件 | 职责 | 代表实现 |
|------|------|---------|
| **Router/Scheduler** | 按阶段分发请求到对应池 | Mooncake Scheduler、vLLM V1 disaggregated scheduler |
| **Prefill Pool** | 专用处理 prefill（compute-bound） | 少量高算力 GPU（如 H100） |
| **Decode Pool** | 专用处理 decode（memory-bound） | 多量大显存 GPU（KV Cache 占大头） |
| **KV Transfer Layer** | prefill→decode 的 KV Cache 跨节点传输 | RDMA（RoCE）、NVLink（单机）、TCP（回退） |

#### 2.2 KV Cache 跨节点传输量计算

**公式**：

```
KV 传输 bytes = seq_len × kv_bytes_per_token
             = seq_len × 2 × n_layer × n_kv_head × d_head × dtype_bytes

传输时间 = KV bytes / RDMA 带宽
```

**LLaMA-7B 示例**（MHA, 524 KB/token, RDMA 100 GB/s）：

| 序列长度 | KV 传输量 | 传输时间（100 GB/s RDMA） |
|---------|----------|------------------------|
| 512 tokens | 256 MB | 2.6 ms |
| 4K tokens | 2 GB | 20 ms |
| 32K tokens | 16 GB | 160 ms |

> ⚠️ **长 prompt 时 KV 传输不可忽略**：32K prompt 的 KV 传输 160ms，可能抵消 PD 分离的 TTFT 改善。这就是为什么 Mooncake 强调"KV 传输层要用 RDMA/NVLink，不能用 TCP"。

#### 2.3 代表系统对比

| 系统 | KV 传输层 | 调度策略 | 特点 |
|------|---------|---------|------|
| **Mooncake**（Moonshot） | RDMA + 分层缓存 | prefill/decode 物理分离 + KV pool | 生产级，Kimi 一线实践 |
| **DistServe**（论文） | RDMA | 微批 prefill + decode 独立调度 | 学术原型，量化收益 |
| **vLLM V1 disaggregated** | NCCL / RDMA | 可选模式，默认仍 colocated | 开源实现，2024+ 主线 |

---

### 3. Coding：PD 分离模拟器

#### 任务 1：运行模拟器对比 colocated vs disaggregated

```bash
python kernels/pd_disaggregated_simulator.py
```

**预期输出**（RTX 5090 模拟参数,100 请求,avg prompt=512,avg decode=64）：

```text
===== Colocated（prefill+decode 共享 GPU） =====
  avg TTFT : 2373.1 ms  (p99 4722.2 ms)
  avg TPOT : 8.96 ms
  avg E2E  : 2964.2 ms

===== Disaggregated（prefill/decode 池分离） =====
  avg TTFT : 593.0 ms  (p99 901.1 ms)
  avg TPOT : 3.44 ms
  avg E2E  : 822.8 ms
  avg KV transfer : 2.8 ms

===== 对比 =====
  TTFT 改善: 2373.1 → 593.0 ms (75% 降低)
  TPOT 改善: 8.96 → 3.44 ms (62% 降低)
  E2E  改善: 2964.2 → 822.8 ms (72% 降低)
  KV 传输代价: 2.8 ms
```

##### 观察重点

1. **TTFT 大幅降低（75%）**：prefill 专用池不被 decode 拖慢，compute-bound 任务独占资源
2. **TPOT 大幅降低（62%）**：decode 专用池不被长 prefill 阻塞，memory-bound 任务稳定
3. **KV 传输代价（2.8 ms）**：avg prompt=512 的 KV 传输仅 2.8ms，远小于 TTFT 改善（2373→593=1780ms），**划算**
4. **p99 改善更显著**：colocated p99 TTFT 4722ms → disaggregated 901ms，尾延迟改善来自调度确定性

> ⚠️ **本模拟器为教学模型**：参数（prefill_tput、decode_tput、退化系数）为示意值，真实系统需用 ncu/nsys 实测。模拟器的价值在于**验证方向性结论**（PD 分离改善 TTFT/TPOT），而非绝对数字。

#### 任务 2：扫描 prompt 长度，找 PD 分离的"不划算"临界点

修改 `PDConfig.avg_prompt_len`，扫描 128/512/4K/32K，观察 KV 传输时间增长何时抵消 TTFT 改善。

> 思考：什么流量特征下 PD 分离不划算？（提示：短 prompt + 低 QPS 时，KV 传输开销占比大而 TTFT 改善小；长 prompt + 高 QPS 时反之。）

---

#### 任务 3：LeetCode 面试题（10 周计划 · 第 7 周 Day 6）

> 📅 今日题目来自 [10 周算法面试刷题计划](https://hzchenxiaobin.github.io/leetcode/problems/10-week-plan.html) 第 7 周「二叉树（下）+ 回溯 + 网格搜索」Day 6（回溯进阶），共 6 题。简单题快速过、中等题精做、困难题吃透；卡壳 20 分钟就看题解，看懂后自己默写一遍。

| 题目 | 难度 | 核心套路 | 题解 |
|------|------|---------|------|
| [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/) | 中等 | 回溯剪枝 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/22_括号生成.html) |
| [79. 单词搜索](https://leetcode.cn/problems/word-search/) | 中等 | DFS 回溯 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/79_单词搜索.html) |
| [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/) | 中等 | 回溯 + 判断 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/131_分割回文串.html) |
| [51. N 皇后](https://leetcode.cn/problems/n-queens/) | 困难 | 回溯 + 位运算 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/51_N皇后.html) |
| [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/) | 中等 | 回溯 + 分段合法性剪枝 | [题解](https://hzchenxiaobin.github.io/leetcode/problems/93_复原IP地址.html) |
| [89. 格雷编码](https://leetcode.cn/problems/gray-code/) | 中等 | 回溯 / 公式法（前缀补 1） | [题解](https://hzchenxiaobin.github.io/leetcode/problems/89_格雷编码.html) |

---
### 4. PD 分离 vs Chunked Prefill 的关系

| 维度 | Chunked Prefill | PD 分离 |
|------|----------------|--------|
| **解决的问题** | 长 prefill 阻塞 decode（同一 GPU 内） | prefill/decode 资源错配（跨 GPU/节点） |
| **粒度** | 把 prefill 切成 chunk，与 decode 交替跑 | prefill/decode 物理分池 |
| **KV 传输** | 无（同 GPU 内） | 有（跨节点） |
| **适用场景** | 中等 QPS、单机 | 高 QPS、多机集群 |
| **关系** | 互补——PD 分离的 prefill 池内部仍可用 chunked prefill | 互补——PD 分离的 prefill 池内部仍可用 chunked prefill |

> 💡 **面试要点**：Chunked Prefill 是"单机内调度优化"，PD 分离是"跨机架构优化"。两者不互斥——Mooncake 的 prefill 池内部就用 chunked prefill。

---

### 5. 面试要点

1. **PD 分离的动机是什么？**（⭐⭐⭐⭐⭐ 必考）
   - prefill compute-bound vs decode memory-bound 的资源错配
   - TTFT/TPOT SLO 矛盾：colocated 下长 prefill 阻塞 decode → TPOT 飙升
   - 分离后 prefill/decode 各自追求最优，互不干扰

2. **KV Cache 跨节点传输开销怎么算？**（⭐⭐⭐⭐ 高频）
   - `传输量 = seq_len × 2 × n_layer × n_kv_head × d_head × dtype_bytes`
   - `传输时间 = 传输量 / RDMA 带宽`
   - LLaMA-7B 32K prompt：16GB / 100GB/s RDMA = 160ms，不可忽略

3. **什么流量特征下 PD 分离不划算？**（⭐⭐⭐ 中频）
   - 短 prompt：KV 传输量小但 TTFT 改善也小，传输开销占比大
   - 低 QPS：建池开销（专用 GPU）摊不薄
   - 单机够用：GPU 数不足以支撑双池分离

4. **PD 分离和 chunked prefill 的关系？**（⭐⭐⭐ 中频）
   - 互补：chunked 是单机内调度，PD 分离是跨机架构
   - Mooncake 的 prefill 池内部用 chunked prefill
   - chunked 解决"长 prefill 阻塞 decode"，PD 分离解决"prefill/decode 资源错配"

5. **Mooncake / DistServe / vLLM V1 的区别？**（⭐⭐⭐ 中频）
   - Mooncake：生产级（Kimi），RDMA + 分层 KV 缓存
   - DistServe：学术原型，量化 PD 分离收益
   - vLLM V1：开源，disaggregated 为可选模式，默认仍 colocated

---

### 今日总结

1. **PD 分离动机**：prefill/decode 资源错配 + TTFT/TPOT SLO 矛盾
2. **架构**：双池（prefill compute + decode memory）+ KV 传输层（RDMA）
3. **KV 传输**：seq_len × kv_bytes_per_token / RDMA 带宽，长 prompt 时不可忽略
4. **收益**：TTFT/TPOT 大幅改善（模拟器 75%/62%），p99 改善更显著
5. **代价**：KV 传输开销 + 双池管理复杂度；短 prompt/低 QPS 时不划算
6. **与 chunked prefill 互补**：单机调度 + 跨机架构，Mooncake 两者都用

> 📖 延伸阅读：Mooncake 论文、DistServe 论文、vLLM V1 disaggregated RFC
