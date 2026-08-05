# interview_advanced.py —— 进阶篇面试题自测系统
# 运行命令: python interview_advanced.py
# 依赖: 仅标准库

import random
import time

QUESTIONS = [
    {
        "id": 1,
        "topic": "Attention 优化",
        "question": "FlashAttention 为什么比标准 Attention 快？",
        "answer": (
            "标准 Attention 物化 S=QK^T 和 P=softmax(S) 两个 N×N 矩阵到 HBM，IO 是 O(N²)\n"
            "FlashAttention 通过 tiling + online softmax 在 SRAM 中完成计算，IO 是 O(Nd)\n"
            "速度来自减少数据移动，不是减少 FLOPs；长序列、小 head dim 时收益最大"
        ),
        "freq": 5,
    },
    {
        "id": 2,
        "topic": "Attention 优化",
        "question": "推导 online softmax 的三个更新公式。",
        "answer": (
            "m_new = max(m_old, max(x_j))\n"
            "l_new = l_old × exp(m_old - m_new) + Σ exp(x_j - m_new)\n"
            "o_new = o_old × (l_old × exp(m_old - m_new) / l_new) + Σ (exp(x_j - m_new) / l_new) × v_j\n"
            "核心：exp(m_old - m_new) 把旧参考点统一到新参考点"
        ),
        "freq": 5,
    },
    {
        "id": 3,
        "topic": "Attention 优化",
        "question": "FlashAttention-1 和 FlashAttention-2 的主要区别是什么？",
        "answer": (
            "FA2 减少了 non-matmul FLOPs\n"
            "FA2 有更好的 work partitioning（warp groups 切分 Q/K/V 工作）\n"
            "FA2 减少了 warp 同步点\n"
            "FA2 提高了 occupancy 和 GEMM 占比"
        ),
        "freq": 4,
    },
    {
        "id": 4,
        "topic": "推理系统",
        "question": "Prefill 和 Decode 阶段有什么区别？各自优化目标是什么？",
        "answer": (
            "Prefill：输入完整 prompt，并行计算，GEMM 较大，compute-bound，关注 TTFT\n"
            "Decode：自回归逐 token 生成，M=1，GEMM 退化，memory-bound，关注 TBT\n"
            "优化：Prefill 用 FlashAttention；Decode 用 KV Cache、Continuous Batching"
        ),
        "freq": 5,
    },
    {
        "id": 5,
        "topic": "推理系统",
        "question": "KV Cache 的核心思想和显存占用公式是什么？",
        "answer": (
            "核心：避免 decode 阶段重复计算历史 K/V\n"
            "每 token 占用 ≈ 2 × layers × num_heads × d_head × bytes_per_val\n"
            "长文本 / 大 batch 时容易 OOM，是推理系统的显存瓶颈"
        ),
        "freq": 5,
    },
    {
        "id": 6,
        "topic": "推理系统",
        "question": "PagedAttention 解决了什么问题？核心设计是什么？",
        "answer": (
            "解决 KV Cache 静态分配造成的显存碎片和浪费\n"
            "把 KV Cache 分成固定大小的 block；逻辑连续、物理可不连续\n"
            "支持 copy-on-write，方便共享 prefix 和做 scheduling"
        ),
        "freq": 5,
    },
    {
        "id": 7,
        "topic": "vLLM 与调度",
        "question": "vLLM 的整体架构和请求生命周期是怎样的？",
        "answer": (
            "架构：LLMEngine → Scheduler → Worker → Model Runner\n"
            "请求生命周期：WAITING → RUNNING → FINISHED / SWAPPED\n"
            "Scheduler 每轮 iteration 选择可运行请求，构建当前 batch"
        ),
        "freq": 5,
    },
    {
        "id": 8,
        "topic": "vLLM 与调度",
        "question": "Continuous Batching 和 Dynamic Batching 的区别？为什么 LLM 更适合 Continuous？",
        "answer": (
            "Dynamic Batching：request-level，一起开始一起结束\n"
            "Continuous Batching：iteration-level，请求动态加入/退出\n"
            "LLM 生成长度差异大，Dynamic 会造成尾部请求等待，GPU 空转；Continuous 提高吞吐"
        ),
        "freq": 5,
    },
    {
        "id": 9,
        "topic": "vLLM 与调度",
        "question": "调度器中的抢占策略有哪些？默认用哪种？",
        "answer": (
            "Recompute：丢弃 KV Cache，之后重算 prompt\n"
            "Swap：把 KV Cache 换出到 CPU 内存\n"
            "默认 Recompute，因为通常比 CPU swap 更快"
        ),
        "freq": 4,
    },
    {
        "id": 10,
        "topic": "场景题",
        "question": "如何优化长文本 LLM 推理？",
        "answer": (
            "1. FlashAttention 降低 attention IO\n"
            "2. PagedAttention 管理 KV Cache 显存\n"
            "3. KV Cache 量化（INT8/INT4）减少显存\n"
            "4. 滑动窗口 / 稀疏 attention\n"
            "5. Chunked prefill 平滑 latency"
        ),
        "freq": 5,
    },
    {
        "id": 11,
        "topic": "场景题",
        "question": "设计一个高吞吐 LLM 推理服务，列出核心模块。",
        "answer": (
            "1. 模型加载与权重管理\n"
            "2. KV Cache 管理（PagedAttention）\n"
            "3. Continuous Batching Scheduler\n"
            "4. 多请求并发与异步返回\n"
            "5. 性能优化（FlashAttention、CUDA Graph、量化）\n"
            "6. 监控、扩缩容与容错"
        ),
        "freq": 5,
    },
    {
        "id": 12,
        "topic": "进阶对比",
        "question": "GQA / MQA / MHA 的区别和 trade-off？",
        "answer": (
            "MHA：每个 head 都有独立 K/V（n_kv_head=n_head），显存最大\n"
            "MQA：所有 head 共享一组 K/V（n_kv_head=1），显存最小但可能损失质量\n"
            "GQA：head 分组共享 K/V（n_kv_head=n_head/g），平衡显存和质量\n"
            "trade-off：显存占用 MHA > GQA > MQA；生成质量通常 MHA ≥ GQA > MQA\n"
            "2024+ 主流选 GQA-8（LLaMA-3、Qwen-2），KV Cache 降到 1/4 而精度几乎不掉"
        ),
        "freq": 4,
    },
    {
        "id": 13,
        "topic": "Attention 优化",
        "question": "手算：LLaMA-7B（32层, 32头, d=128, fp16）的每 token KV Cache？GQA-8 和 MQA 各降到多少？",
        "answer": (
            "公式：bytes/token = 2(K+V) × n_layer × n_kv_head × d_head × 2B\n"
            "MHA:  2×32×32×128×2B = 524 KB/token\n"
            "GQA-8 (n_kv_head=8):  524/4 = 131 KB/token\n"
            "MQA   (n_kv_head=1):  524/32 = 16.4 KB/token\n"
            "1M token 时 MHA 要 524 GB（漏乘 n_layer=32 是经典错误，会算成 16 GB）"
        ),
        "freq": 5,
    },
    {
        "id": 14,
        "topic": "Attention 优化",
        "question": "MLA（Multi-head Latent Attention）为什么 KV Cache 这么小？",
        "answer": (
            "MLA 不存完整 K/V，存一个低秩'潜在向量'（d_c 维，DeepSeek-V3 d_c≈512）\n"
            "每 token KV bytes ≈ 2×n_layer×d_c×2B，远小于 MHA 的 2×n_layer×n_head×d_head×2B\n"
            "attention 时用上投影矩阵现场解压 K/V，代价是额外解压 GEMM\n"
            "DeepSeek-V3：MLA 比 MHA 小 ~10x 且精度持平，是 2024-2026 推理优化的热门方向"
        ),
        "freq": 4,
    },
    {
        "id": 15,
        "topic": "Attention 优化",
        "question": "为什么 GQA 比 MQA 更受青睐？分组数 g 怎么选？",
        "answer": (
            "MQA（n_kv_head=1）显存最小但 perplexity 上升明显，质量损失不可接受\n"
            "GQA 通过分组（g=4~8）在显存与精度间取折中，LLaMA-3 用 GQA-8 实测精度接近 MHA\n"
            "g 的选择：g 太小接近 MQA（质量差），g 太大接近 MHA（显存大）；通常 g=4 或 8\n"
            "经验：d_head=128 时 GQA-8 的 KV Cache 降到 1/4，精度损失 <0.5%，性价比最高"
        ),
        "freq": 3,
    },
    {
        "id": 16,
        "topic": "系统设计",
        "question": "设计百万 QPS LLM serving 系统（澄清→估算→架构→权衡→延伸）",
        "answer": (
            "澄清: QPS=100w/s, 单请求 ~100 token, TTFT<500ms, TPOT<50ms\n"
            "估算: 100w×100=1亿 token/s, 单卡 ~1000 token/s → 需 ~10万卡\n"
            "架构: 异构调度(PD分离) + EP大规模并行 + KV传输层(RDMA) + 多级缓存\n"
            "  prefill池: compute-bound, 少GPU高算力; decode池: memory-bound, 多GPU大显存\n"
            "  EP: 专家切多节点, decode用EP(all-to-all小), prefill用TP(all-reduce摊薄)\n"
            "权衡: PD分离省TTFT/TPOT但增KV传输; EP省显存但增all-to-all; 量化省显存但损精度\n"
            "延伸: 成本=$/1M tokens = GPU时价×利用率/吞吐; 监控TTFT/TPOT/GPU利用率/显存"
        ),
        "freq": 5,
    },
    {
        "id": 17,
        "topic": "系统设计",
        "question": "给定 SLO(TTFT<200ms, TPOT<50ms)，反推集群配置与并行策略",
        "answer": (
            "反推: TTFT<200ms → prefill时间<200ms → 单卡prefill吞吐够或用PD分离\n"
            "  TPOT<50ms → decode单步<50ms → KV Cache读取+attention<50ms\n"
            "集群: 若单卡TPOT=80ms, 用TP=2降到40ms(通信开销~10ms)\n"
            "  或用PD分离: decode专用池无prefill干扰, TPOT稳定<50ms\n"
            "KV: 长序列用GQA-8/MLA降显存, INT8/FP8量化降带宽\n"
            "batch: 连续batching+chunked prefill, max_batch_size由显存反推\n"
            "验证: ncu测单kernel, nsys测端到端, 压测验SLO"
        ),
        "freq": 4,
    },
    {
        "id": 18,
        "topic": "系统设计",
        "question": "推理成本 $/1M tokens 怎么算？如何优化？",
        "answer": (
            "公式: $/1M tokens = (GPU时价$/h × 3600s) / (吞吐token/s × 1e6)\n"
            "  例: A100 $2/h, 吞吐 5000 token/s → $/1M = 2×3600/(5000×1e6)=$1.44\n"
            "优化: ①提升吞吐(量化/ batching/FA) ②降低GPU单价(spot实例/异构集群)\n"
            "  ③提升利用率(PD分离削峰填谷, 避免资源浪费)\n"
            "  ④KV Cache复用(prefix caching省重复计算)\n"
            "权衡: 量化降本但损精度; 大batch提吞吐但增延迟; PD分离省资源但增KV传输"
        ),
        "freq": 4,
    },
    {
        "id": 19,
        "topic": "系统设计",
        "question": "万卡训练系统设计（选学，训练岗）",
        "answer": (
            "并行: DP+TP+PP+EP四维并行, 万卡=如 8节点×8卡×125(EP)\n"
            "通信: TP用NVLink(节点内), PP用流水线micro-batch, EP用all-to-all\n"
            "  DP用all-reduce(梯度同步), ZeRO切优化器状态/梯度/参数\n"
            "容错: checkpoint频率(如每1000步), 故障检测(心跳+nccl超时), 热替换\n"
            "  显存: 激活重计算(省显存换计算), ZeRO分片, 梯度累积\n"
            "调度: Megatron/DeepSpeed框架, 资源管理(K8s/Slurm), 监控(GPU利用率/通信开销)"
        ),
        "freq": 3,
    },
    {
        "id": 20,
        "topic": "项目话术",
        "question": "STAR 法讲 GEMM 优化项目（含可追问细节与诚实局限）",
        "answer": (
            "Situation: 8周课程中手写GEMM从naive到优化, 目标cuBLAS 90%+\n"
            "Task: 在RTX 5090上实现GEMM优化系列, 量化每层收益\n"
            "Action: naive→tiling→reg blocking→float4→shuffle→dblbuf, 7个版本递进\n"
            "  关键决策: float4向量化加载(128-bit load)是最大单步收益(30.8%→64.3%)\n"
            "  同步dblbuf基本无效(62.9%→63.8%), 因未用cp.async真双缓冲\n"
            "Result: 4096³达cuBLAS 62.9%(FMA路线), 诚实声明未达90%(需Tensor Core/CUTLASS)\n"
            "可追问: ①为何float4收益最大(砍3/4指令) ②dblbuf为何无效(同步式) ③如何到90%(WMMA+smem tiling)\n"
            "诚实局限: 未实现Tensor Core版, 未做autotune, 小矩阵(1024)仅42%(wave填不满)"
        ),
        "freq": 4,
    },
    {
        "id": 21,
        "topic": "项目话术",
        "question": "STAR 法讲 Mini 推理引擎项目（含可追问细节与诚实局限）",
        "answer": (
            "Situation: 8周课程整合, 手写Mini推理引擎(从v0单请求到v1多请求并发)\n"
            "Task: 实现Continuous Batching+Scheduler+KV Cache+自定义kernel集成\n"
            "Action: v0(单请求同步)→v1(多请求异步Future+MiniScheduler)→v2(优先级/超时/抢占)\n"
            "  关键决策: token budget+num_seqs双约束, 锁内队列操作/锁外forward\n"
            "  稳定性: 500请求连续测试不崩溃(sleep模拟forward, 非真实GPU)\n"
            "Result: 端到端可跑, 调度逻辑正确(含RECOMPUTE/SWAP抢占), 但性能非真实(模拟)\n"
            "可追问: ①为何500非1000(模拟引擎) ②livelock怎么防(快照迭代) ③真实性能(需Mini引擎真整合)\n"
            "诚实局限: sleep模拟非真实forward, 无真实GPU推理性能, 无生产级容错(如OOM恢复)"
        ),
        "freq": 4,
    },
]


def self_test(num=5):
    """随机抽题，限时口述自测。"""
    print(f"=== 进阶篇面试自测（随机 {num} 题）===\n")
    sample = random.sample(QUESTIONS, min(num, len(QUESTIONS)))
    for i, q in enumerate(sample, 1):
        stars = "⭐" * q["freq"]
        print(f"[{i}/{num}] {stars} [{q['topic']}]")
        print(f"Q: {q['question']}")
        input("口述答案后按回车查看参考...")
        print(f"A: {q['answer']}")
        print()


def list_all():
    """列出所有题目。"""
    for q in QUESTIONS:
        stars = "⭐" * q["freq"]
        print(f"#{q['id']:>2} {stars} [{q['topic']}] {q['question']}")


if __name__ == "__main__":
    print("=== AI Infra 面试进阶篇自测系统 ===")
    print(f"共 {len(QUESTIONS)} 道题\n")
    print("命令：")
    print("  list  — 列出所有题目")
    print("  test  — 随机抽 5 题自测（默认）")
    print("  test N — 随机抽 N 题自测")
    cmd = input("\n输入命令: ").strip()
    if cmd == "list":
        list_all()
    elif cmd.startswith("test"):
        n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 5
        self_test(n)
    else:
        list_all()
