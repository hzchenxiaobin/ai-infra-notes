# week5_summary.py —— Week 5 推理系统核心问题与面试题自测（总结日脚本）
# 运行命令: python week5_summary.py
# 依赖: 仅标准库
#
# 工具：打印推理系统四大核心问题清单 + 本周 17 道面试题自测卡片，
#       供 Day 7 总结日复盘使用。

import random

CORE_PROBLEMS = [
    {
        "name": "① 内存管理",
        "problems": [
            "KV Cache 显存占用大（2×L×layers×heads×d_head×bytes）",
            "每请求生成长度不确定 → 动态分配碎片",
            "长文本 OOM（序列超显存）",
            "多轮对话历史上下文累积",
        ],
        "solutions": [
            "KV Cache 量化（INT8/FP8）", "GQA/MQA（减 KV 头数）",
            "PagedAttention（分页+block table，无碎片）",
            "滑动窗口/稀疏 attention", "Cache 复用（多轮 prefix caching）",
        ],
        "related_day": "Day2/Day4",
    },
    {
        "name": "② Batch 策略",
        "problems": [
            "Static Batching：凑齐才开始，长请求阻塞 batch",
            "Dynamic Batching：请求级聚合，引入等待延迟",
            "请求生成长度方差大 → 资源利用率低",
        ],
        "solutions": [
            "Continuous Batching：每轮 iteration 重建 batch",
            "完成的请求立即让位，新请求随时插入",
            "Chunked Prefill：大 prefill 拆小块与 decode 交错",
        ],
        "related_day": "Day3",
    },
    {
        "name": "③ Latency 隐藏",
        "problems": [
            "Decode 的 launch overhead 主导（kernel 太小太多）",
            "compute 与 communication 串行",
            "Prefill 大请求阻塞 decode",
        ],
        "solutions": [
            "CUDA Graph（消除 per-step launch）", "torch.compile（自动 fusion）",
            "Async Copy（overlap 传输与计算）", "Speculative Decoding（小模型预测+大模型验证）",
            "Chunked Prefill（prefill 与 decode 交错）",
        ],
        "related_day": "Day6",
    },
    {
        "name": "④ 调度开销",
        "problems": [
            "Python GIL / 多线程竞争",
            "每 iteration 重建 input tensors",
            "内存分配/释放开销",
            "CPU-GPU 同步点（cudaSynchronize）",
        ],
        "solutions": [
            "核心逻辑用 C++（减少 Python 层）", "预分配 tensor buffer",
            "异步采样", "减少 cudaSynchronize 调用",
        ],
        "related_day": "Day3/Day6",
    },
]

INTERVIEW_QUESTIONS = [
    ("Prefill vs Decode 的区别和瓶颈？", "Day1", "Prefill compute-bound(N×N)，Decode memory-bound(M=1)"),
    ("TTFT 和 TBT 是什么？如何优化？", "Day1/6", "TTFT=Prefill延迟→FlashAttention；TBT=Decode→KV Cache/CB"),
    ("KV Cache 核心思想和收益？", "Day2", "存历史K/V避免重算，FLOPs O(L·d²)→O(d²)"),
    ("KV Cache 内存占用如何计算？", "Day2", "2×n_layers×n_heads×d_head×bytes/token"),
    ("静态 vs 动态 KV Cache 分配？", "Day2", "静态浪费，动态碎片，PagedAttention 解决"),
    ("vLLM 整体架构？", "Day3", "LLMEngine→Scheduler→Worker 三层"),
    ("SequenceGroup 是什么？", "Day3", "一请求含多候选序列(beam/n>1)，共享prompt cache"),
    ("Scheduler 依据什么决策？", "Day3", "SchedulingBudget: token/num_seqs/显存三重预算"),
    ("Preemption 两种策略？", "Day3", "Recomputation(默认,短prompt) vs Swapping(长prompt)"),
    ("PagedAttention 解决什么问题？", "Day4", "分页+block table，无静态浪费+无动态碎片"),
    ("Copy-on-Write 应用场景？", "Day4", "beam/并行采样共享prompt block，写入时复制"),
    ("如何构建最简单的推理引擎？", "Day5", "Tokenizer+模型后端+KVCache+采样器+循环 5组件"),
    ("Prefill/Decode 各存什么到KV Cache？", "Day5", "Prefill存N个token的K/V，Decode每步追加1个"),
    ("如何做端到端 profiling？", "Day6", "nsys→cuda.Event→ncu 三层方法论"),
    ("TBT 为什么随序列长度增长？", "Day6", "读KV Cache随L增长→memory-bound；优化:量化/GQA/滑窗"),
    ("推理系统四大核心问题？", "Day7", "内存管理/Batch策略/Latency隐藏/调度开销"),
    ("Continuous vs Dynamic Batching？", "Day7", "Dynamic请求级聚合,Continuous每轮重建batch"),
]


def print_core_problems():
    print("=" * 70)
    print("推理系统四大核心问题")
    print("=" * 70)
    for cp in CORE_PROBLEMS:
        print(f"\n{cp['name']}  (相关: {cp['related_day']})")
        print(f"  问题:")
        for p in cp["problems"]:
            print(f"    • {p}")
        print(f"  解决方案:")
        for s in cp["solutions"]:
            print(f"    ✓ {s}")


def self_test(num=5):
    print("\n" + "=" * 70)
    print(f"面试题自测（随机 {num} 题，先看问题，想答案再看提示）")
    print("=" * 70)
    qs = random.sample(INTERVIEW_QUESTIONS, min(num, len(INTERVIEW_QUESTIONS)))
    for i, (q, day, hint) in enumerate(qs, 1):
        print(f"\n[{i}] {q}  (Day {day})")
        try:
            input("  按回车看提示...")
        except EOFError:
            pass
        print(f"  💡 {hint}")


def print_all_questions():
    print("\n" + "=" * 70)
    print(f"本周全部 {len(INTERVIEW_QUESTIONS)} 道面试题清单")
    print("=" * 70)
    for i, (q, day, hint) in enumerate(INTERVIEW_QUESTIONS, 1):
        print(f"  {i:2d}. [{day:8s}] {q}")


def print_cheatsheet():
    print("\n" + "=" * 70)
    print("推理系统优化速查表（现象→检查→解决）")
    print("=" * 70)
    cheatsheet = [
        ("TTFT 过高", "profile prefill", "FlashAttention、Tensor Core、并行 prefill"),
        ("TBT 过高", "profile decode", "KV Cache、PagedAttention、量化"),
        ("TBT 随 L 增长", "扫描不同 L", "GQA/MQA、滑动窗口、稀疏 attention"),
        ("显存 OOM", "监控显存", "PagedAttention、INT8 KV、减 batch"),
        ("Kernel 间隙大", "nsys timeline", "CUDA Graph、torch.compile、kernel fusion"),
        ("长请求阻塞 batch", "观察完成时间", "Continuous Batching"),
        ("多轮对话 TTFT 高", "检查 cache 复用", "session KV Cache / prefix caching"),
        ("显存碎片", "block allocator", "PagedAttention"),
        ("Throughput 低", "nsys SM util", "增大 batch、continuous batching"),
    ]
    print(f"{'现象':<18}{'检查方法':<18}{'解决方案'}")
    print("-" * 70)
    for sym, chk, sol in cheatsheet:
        print(f"{sym:<18}{chk:<18}{sol}")


def main():
    print("Week 5 Day 7 —— 推理系统核心问题总结与面试复盘")
    print()
    print_core_problems()
    print_cheatsheet()
    print_all_questions()
    self_test(num=5)
    print("\n✅ Week 5 总结完成。进入 Week 6：Batching & 调度优化。")


if __name__ == "__main__":
    main()
