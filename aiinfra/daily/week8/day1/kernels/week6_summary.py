# week6_summary.py —— Week 6 调度策略总结与面试题自测（总结日脚本）
# 运行命令: python week6_summary.py
# 依赖: 仅标准库
#
# 工具：打印 7 种调度策略对比表 + 策略选择决策树 + 本周 15 道面试题自测卡片，
#       供 Day 7 总结日复盘使用。

import random

# ============================================================
# 7 种调度策略对比
# ============================================================

STRATEGIES = [
    {
        "name": "Static Batching",
        "principle": "固定 batch size，凑齐才开始，一起结束",
        "scene": "简单 demo / 请求等长",
        "pros": "实现最简单",
        "cons": "吞吐低、长请求阻塞、GPU 空等",
        "day": "Day1",
    },
    {
        "name": "Dynamic Batching",
        "principle": "请求级聚合 + 超时等待 + max_batch_size",
        "scene": "吞吐优先、非 LLM 自回归",
        "pros": "提高 GPU 利用率、凑批灵活",
        "cons": "request-level 阻塞、padding 浪费、长请求拖整批",
        "day": "Day1",
    },
    {
        "name": "Continuous Batching",
        "principle": "iteration-level 调度，每轮重建 batch，完成即走",
        "scene": "LLM 自回归推理（生成长度差异大）",
        "pros": "吞吐+延迟兼顾、短请求不等长请求",
        "cons": "实现复杂、需 PagedAttention 配合",
        "day": "Day2",
    },
    {
        "name": "Priority Scheduling",
        "principle": "高优先级请求优先 prefill/decode",
        "scene": "多租户、不同 SLA",
        "pros": "保障关键请求延迟、可配 SLA",
        "cons": "低优先级饥饿、优先级反转",
        "day": "Day5",
    },
    {
        "name": "Preemption (Recompute/Swap)",
        "principle": "显存不足时抢占 running 请求，RECOMPUTE 重算 / SWAP 换出",
        "scene": "显存压力下保高优先级",
        "pros": "过载优雅降级、不 OOM 崩溃",
        "cons": "额外重算/PCIe 开销",
        "day": "Day3",
    },
    {
        "name": "Chunked Prefill",
        "principle": "长 prompt 拆小 chunk 与 decode 交错",
        "scene": "长短请求混合、TPOT 敏感",
        "pros": "平滑 decode latency、TPOT 稳定",
        "cons": "调度复杂、TTFT 略增",
        "day": "Day4",
    },
    {
        "name": "Speculative Decoding",
        "principle": "小模型预测多 token + 大模型验证",
        "scene": "低延迟、有 draft model",
        "pros": "降低 TBT、提升有效吞吐",
        "cons": "需 draft model、验证开销",
        "day": "进阶",
    },
]

# ============================================================
# 策略选择决策树
# ============================================================

DECISION_TREE = [
    ("是否要求最低延迟（交互式）？", "是", "Static / 小 batch + Priority", "否"),
    ("请求到达是否连续？", "是", "Dynamic Batching", "否 → Static"),
    ("是否 LLM 自回归生成？", "是", "Continuous Batching", "否 → Dynamic"),
    ("是否多租户/多 SLA？", "是", "+ Priority Scheduling", "否"),
    ("是否有长 prompt？", "是", "+ Chunked Prefill", "否"),
    ("显存是否紧张？", "是", "+ Preemption (Recompute/Swap)", "否"),
    ("是否有 draft model？", "是", "+ Speculative Decoding", "否"),
]

# ============================================================
# 本周 15 道核心面试题（按主题分组）
# ============================================================

INTERVIEW_QUESTIONS = [
    # Dynamic Batching（Day1）
    {"topic": "Dynamic Batching", "day": "Day1",
     "q": "Dynamic Batching 的原理和优缺点？",
     "a": "原理：请求入队→等满/超时→聚合整批 forward→整批完成。优点：提高 GPU 利用率。缺点：request-level 阻塞（长请求拖整批）、padding 浪费。"},
    {"topic": "Dynamic Batching", "day": "Day1",
     "q": "Padding 有什么问题？如何优化？",
     "a": "不同长度请求 pad 到同一长度浪费计算。优化：长度分组（相似长度凑批）、attention mask（只算有效位置）、padding-free（用 position_ids 去重）。"},
    # Continuous Batching（Day2）
    {"topic": "Continuous Batching", "day": "Day2",
     "q": "Continuous Batching 和 Dynamic Batching 的区别？",
     "a": "Dynamic 是 request-level（整批一起开始结束）；Continuous 是 iteration-level（每轮重建 batch，完成即走、新请求随时插入）。后者吞吐 2-8x。"},
    {"topic": "Continuous Batching", "day": "Day2",
     "q": "Continuous Batching 为什么适合 LLM 推理？",
     "a": "LLM 生成长度差异大且事先未知。Dynamic 下短请求等长请求空等；Continuous 让 GPU 始终满载，短请求完成即退出。配合 PagedAttention 无碎片回收 slot。"},
    {"topic": "Continuous Batching", "day": "Day2",
     "q": "Prefill + Decode 混合调度的挑战？",
     "a": "①Token budget 分配（prefill 吃大块算力）②显存需求（prefill 需临时 KV 空间）③Latency 抖动（prefill 打断 decode 节奏）。解决：chunked prefill、限制每轮 prefill token 数。"},
    # vLLM Scheduler（Day3）
    {"topic": "vLLM Scheduler", "day": "Day3",
     "q": "vLLM Scheduler 的 schedule() 流程？",
     "a": "5 步：①free_finished 释放 block ②schedule_running 继续 decode（不够则 preempt）③schedule_swapped 换回 GPU ④schedule_waiting 加入新请求（swapped 非空时跳过）⑤构建 SchedulerOutputs。"},
    {"topic": "vLLM Scheduler", "day": "Day3",
     "q": "SchedulingBudget 的两个核心参数？",
     "a": "token_budget（每轮 token 上限，防 prefill 霸占算力）+ max_num_seqs（并发上限，控 batch 大小）。can_schedule() 两约束都满足才调度。"},
    {"topic": "vLLM Scheduler", "day": "Day3",
     "q": "Preemption 的两种模式？默认哪个？为什么？",
     "a": "RECOMPUTE（丢弃 KV 重 prefill，默认）/ SWAP（KV 换出 CPU）。默认 RECOMPUTE：不需 CPU 内存、通常重算比 PCIe 换入快。SWAP 适合 prompt 极长、抢占久的场景。"},
    # 框架对比（Day4）
    {"topic": "框架对比", "day": "Day4",
     "q": "Inflight Batching 和 Continuous Batching 区别？",
     "a": "本质相同（都是 iteration-level）。差异：vLLM 用 Python 调度（灵活）、TensorRT-LLM 用 C++ 调度（性能高但灵活性低，需重编译 engine）。"},
    {"topic": "框架对比", "day": "Day4",
     "q": "Chunked Prefill 是什么？解决什么问题？",
     "a": "长 prompt 拆成小 chunk（如 512-2048 token），每轮只 prefill 一块，剩余预算给 decode。解决：长 prefill 占满整轮算力导致 decode TPOT 突增。实测延迟尖峰降 40%。"},
    # Mini 引擎 v1（Day5）
    {"topic": "Mini 引擎 v1", "day": "Day5",
     "q": "多请求并发推理引擎需要解决哪些问题？",
     "a": "①请求队列（线程安全）②调度器（Continuous Batching）③KV Cache 管理（每请求独立）④异步返回（Future）⑤资源隔离（token budget）⑥生命周期（waiting→running→finished）。"},
    {"topic": "Mini 引擎 v1", "day": "Day5",
     "q": "优先级调度的优缺点？",
     "a": "优点：高优先级（付费用户）快响应、可配 SLA。缺点：低优先级饥饿、优先级反转。缓解：aging 老化、最大等待时间、资源预留。"},
    # Benchmark（Day6）
    {"topic": "Benchmark", "day": "Day6",
     "q": "如何做 throughput-latency benchmark？",
     "a": "两种方法：①固定并发扫描（同时提 N 请求，扫 N 找饱和点）②固定 QPS 测试（恒定速率发，看 P50/P99）。指标：Throughput、Avg/P50/P99、TTFT、TPOT、GPU util。"},
    {"topic": "Benchmark", "day": "Day6",
     "q": "如何识别饱和点？",
     "a": "throughput 增长率<5% + latency 开始飙升 + GPU util≈100% + 队列堆积。超过后 throughput 封顶、latency 因排队线性涨（conc 翻倍→latency 翻倍）。"},
    # 总结（Day7）
    {"topic": "总结", "day": "Day7",
     "q": "调度策略如何选择？",
     "a": "决策树：最低延迟→小batch+优先级；连续到达→Dynamic；LLM自回归→Continuous；多租户→+Priority；长prompt→+Chunked Prefill；显存紧→+Preemption。LLM 推理标配：Continuous + PagedAttention + Chunked Prefill。"},
]


def print_strategy_comparison():
    print("=" * 90)
    print("Week 6 调度策略对比表（7 种策略）")
    print("=" * 90)
    print(f"{'策略':<28} {'原理':<22} {'适用场景':<18} {'优点':<16} {'缺点':<14} {'Day'}")
    print("-" * 90)
    for s in STRATEGIES:
        print(f"{s['name']:<28} {s['principle']:<22} {s['scene']:<18} "
              f"{s['pros']:<16} {s['cons']:<14} {s['day']}")


def print_decision_tree():
    print("\n" + "=" * 90)
    print("调度策略选择决策树")
    print("=" * 90)
    for i, (q, yes, ans, no) in enumerate(DECISION_TREE, 1):
        print(f"\n  Step {i}: {q}")
        print(f"    ├─ {yes} → {ans}")
        print(f"    └─ {no}")
    print("\n  LLM 推理服务标配：Continuous Batching + PagedAttention + Chunked Prefill")
    print("  +（按需）Priority / Preemption / Speculative Decoding")


def print_interview_overview():
    print("\n" + "=" * 90)
    print("本周 15 道核心面试题（按主题分组）")
    print("=" * 90)
    topics = {}
    for q in INTERVIEW_QUESTIONS:
        topics.setdefault(q["topic"], []).append(q)
    for topic, qs in topics.items():
        print(f"\n  【{topic}】（{qs[0]['day']}）")
        for i, q in enumerate(qs, 1):
            print(f"    {i}. {q['q']}")


def self_test(num=5):
    print("\n" + "=" * 90)
    print(f"自测模式：随机抽 {num} 题（先看问题，按回车看答案）")
    print("=" * 90)
    sample = random.sample(INTERVIEW_QUESTIONS, min(num, len(INTERVIEW_QUESTIONS)))
    score = 0
    for i, q in enumerate(sample, 1):
        print(f"\n  [{i}/{len(sample)}] ({q['topic']}, {q['day']})")
        print(f"  Q: {q['q']}")
        try:
            input("  按回车看参考答案...")
        except EOFError:
            pass
        print(f"  A: {q['a']}")
        try:
            ans = input("  你答对了吗？(y/n): ").strip().lower()
        except EOFError:
            ans = ""
        if ans == "y":
            score += 1
    print(f"\n  自测得分：{score}/{len(sample)}")
    print(f"  {'优秀！Week 6 调度专题已掌握' if score >= len(sample)*0.8 else '继续复盘，重跑自测'}")


def main():
    print("Week 6 调度优化策略总结 —— 总结日复盘工具")
    print("对应 Day 42：策略对比 + 面试复盘 + GitHub 整理\n")

    print_strategy_comparison()
    print_decision_tree()
    print_interview_overview()

    print("\n" + "=" * 90)
    print("✅ Week 6 核心收获：")
    print("  1. Dynamic→Continuous：request-level → iteration-level 调度")
    print("  2. vLLM Scheduler：5 步 schedule() + SchedulingBudget + Preemption")
    print("  3. 框架对比：vLLM(Python,灵活) vs TRT-LLM(C++,快) vs LightLLM(Token Attn)")
    print("  4. Chunked Prefill：长 prompt 拆块与 decode 交错，平滑 TPOT")
    print("  5. Mini 引擎 v1：多请求 + Continuous Batching + 优先级 + Future")
    print("  6. Benchmark：throughput-latency 曲线找饱和点，P99 看尾延迟")
    print("=" * 90)

    try:
        run = input("\n是否开始自测？(y/n): ").strip().lower()
        if run == "y":
            self_test()
    except EOFError:
        pass

    print("\n✅ Week 6 总结复盘完成。")


if __name__ == "__main__":
    main()
