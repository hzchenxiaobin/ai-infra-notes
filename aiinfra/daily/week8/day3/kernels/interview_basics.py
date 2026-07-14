# interview_basics.py —— 基础篇面试题自测系统
# 运行命令: python interview_basics.py
# 依赖: 仅标准库

import random
import time

QUESTIONS = [
    {
        "id": 1,
        "topic": "GPU 基础",
        "question": "什么是 SM、Warp、Thread？它们之间的关系是什么？",
        "answer": (
            "SM（Streaming Multiprocessor）：GPU 基本计算单元，含多核心+寄存器+shared memory\n"
            "Warp：32 thread 组成，是 GPU 调度基本单位（SIMT）\n"
            "Thread：最细粒度执行单元\n"
            "关系：Grid > Block > Warp > Thread；一个 SM 可同时运行多个 warp"
        ),
        "freq": 4,
    },
    {
        "id": 2,
        "topic": "GPU 基础",
        "question": "什么是 Occupancy？什么情况下会降低？",
        "answer": (
            "Occupancy = active_warp / max_warp_per_SM\n"
            "降低原因：① 寄存器过多 ② shared mem 过多 ③ block 非对齐 ④ grid 不足\n"
            "影响：低 occupancy → 无法隐藏延迟 → 性能下降"
        ),
        "freq": 4,
    },
    {
        "id": 3,
        "topic": "GPU 基础",
        "question": "解释 GPU 的 memory hierarchy。",
        "answer": (
            "Register(~0c) < Shared Mem/L1(~20c) < L2(~200c) < HBM(~500c)\n"
            "优化目标：热点数据驻留 register/shared mem，合并访问 global memory"
        ),
        "freq": 4,
    },
    {
        "id": 4,
        "topic": "GPU 基础",
        "question": "什么是 bank conflict？如何避免？",
        "answer": (
            "Shared memory 分 32 bank；同 warp 多 thread 访问同一 bank → 串行化\n"
            "避免：padding（s_A[BM][BK+1]）、向量化、连续 thread 访问不同 bank"
        ),
        "freq": 4,
    },
    {
        "id": 5,
        "topic": "Kernel 优化",
        "question": "如何把 GEMM 优化到 cuBLAS 80%？",
        "answer": (
            "Naive(1%) → Tiling(15%) → Register Blocking(40%) → float4(55%)\n"
            "→ Warp Shuffle(60%) → Double Buffer(70%) → Tensor Core(80%+)\n"
            "→ Auto-tuning(90%+)"
        ),
        "freq": 5,
    },
    {
        "id": 6,
        "topic": "Kernel 优化",
        "question": "float4 向量化加载为什么能提升性能？",
        "answer": (
            "4 个连续 float=16B=一条 128-bit load 指令\n"
            "减少指令数、提升带宽利用率；需地址对齐 + coalesced"
        ),
        "freq": 4,
    },
    {
        "id": 7,
        "topic": "Kernel 优化",
        "question": "Warp Shuffle 比 Shared Memory 快多少？为什么？",
        "answer": (
            "Shuffle ~1-2 cycles，SMem ~20-30 cycles\n"
            "原因：warp 内专用交换网络直接读寄存器，不经 SMem 读写路径\n"
            "局限：只限 warp 内（32 thread）"
        ),
        "freq": 4,
    },
    {
        "id": 8,
        "topic": "CUDA 编程",
        "question": "__syncthreads() 和 warp shuffle 的同步区别？",
        "answer": (
            "__syncthreads()：block 级同步，所有 thread 必须到达，有开销\n"
            "Warp shuffle：warp 内隐式同步，硬件自动完成，开销极小\n"
            "Shuffle 只限 warp 内，block 级仍需 __syncthreads()"
        ),
        "freq": 3,
    },
    {
        "id": 9,
        "topic": "CUDA 编程",
        "question": "Default Stream 有什么坑？",
        "answer": (
            "Stream 0 隐式同步所有 explicit stream → 并发被打断\n"
            "解决：cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)\n"
            "或 nvcc --default-stream per-thread"
        ),
        "freq": 3,
    },
    {
        "id": 10,
        "topic": "CUDA 编程",
        "question": "cudaMemcpyAsync 和 cudaMemcpy 的区别？",
        "answer": (
            "cudaMemcpy：同步，阻塞 host\n"
            "cudaMemcpyAsync：异步，需 pinned memory，可与其他 kernel overlap"
        ),
        "freq": 3,
    },
    {
        "id": 11,
        "topic": "Profiling",
        "question": "如何分析一个 CUDA kernel 的瓶颈？",
        "answer": (
            "1. ncu 看 SM/Memory Throughput + Achieved Occupancy\n"
            "2. Roofline 判 memory-bound / compute-bound\n"
            "3. Warp Stall Reasons（Long Scoreboard = mem 延迟）"
        ),
        "freq": 4,
    },
    {
        "id": 12,
        "topic": "Profiling",
        "question": "什么是 Roofline Model？RTX 5090 的 ridge point 是多少？",
        "answer": (
            "横轴=算术强度(FLOP/Byte)，纵轴=性能(FLOP/s)\n"
            "斜线=带宽限制，水平线=算力限制，交点=ridge point\n"
            "RTX 5090: 19.5 TFLOPS / 1.55 TB/s ≈ 12.6 FLOP/Byte"
        ),
        "freq": 4,
    },
]


def self_test(num=5):
    """随机抽题，限时口述自测。"""
    print(f"=== 基础篇面试自测（随机 {num} 题）===\n")
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
    print("=== AI Infra 面试基础篇自测系统 ===")
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
