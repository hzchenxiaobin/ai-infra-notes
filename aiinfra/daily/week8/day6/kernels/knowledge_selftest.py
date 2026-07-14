#!/usr/bin/env python3
# knowledge_selftest.py —— AI Infra 知识点自测系统
# 运行命令: python knowledge_selftest.py
# 依赖: 仅标准库
#
# 覆盖六大薄弱点：Online Softmax / GEMM 层次 / vLLM Scheduler /
#                  KV Cache 内存 / Roofline / Prefill-Decode
# 三种模式：
#   quiz    —— 随机抽题，限时口述后看答案
#   formula —— 关键公式默写（填空）
#   param   —— RTX 5090 关键参数快问快答

import random
import time

QUIZ_BANK = [
    {
        "topic": "Online Softmax",
        "q": "写出 online softmax 的三个更新公式（m / l / o）。",
        "a": (
            "m_new = max(m, max(xj))\n"
            "l_new = l * exp(m - m_new) + Σ exp(xj - m_new)\n"
            "o_new = o * (l * exp(m - m_new) / l_new) + Σ (exp(xj - m_new) / l_new) * vj\n"
            "\n要点：exp(m - m_new) 是统一参考点的缩放因子；"
            "避免物化 N×N 矩阵，IO 从 O(N²) 降到 O(Nd)。"
        ),
    },
    {
        "topic": "GEMM 优化层次",
        "q": "从 Naive 到 cuBLAS 80%+，按层次说出每层优化及增益。",
        "a": (
            "Naive (~1%) → Shared Memory Tiling (~15%) → Register Blocking (~45%)\n"
            "→ float4 向量化 (~55%) → Warp Shuffle (~60%)\n"
            "→ Double Buffering (~70%) → Tensor Core (~80%+) → Auto-tuning (~90%+)\n"
            "\n每层收益：Tiling 减少全局重复读；RegBlock 数据驻留寄存器；"
            "float4 128-bit load 提带宽；Shuffle 优化写回；DblBuf 掩盖传输。"
        ),
    },
    {
        "topic": "vLLM Scheduler",
        "q": "vLLM Scheduler 的请求状态机是怎样的？抢占策略有哪些？",
        "a": (
            "状态：WAITING → RUNNING → FINISHED / SWAPPED\n"
            "每轮 iteration 重新组 batch（Continuous Batching）：\n"
            "  1. 检查 RUNNING 中已完成的，移到 FINISHED\n"
            "  2. 显存不足时抢占 RUNNING（SWAPPED 或 RECOMPUTE）\n"
            "  3. 从 WAITING 补充新请求到 RUNNING\n"
            "抢占策略：Recompute（丢 KV 重算，通常更快）/ Swap（换出 CPU，省显存）。"
        ),
    },
    {
        "topic": "KV Cache 内存",
        "q": "LLaMA-7B（32 层 / 32 头 / d=128 / fp16）每 token KV Cache 占多少？4096 token 呢？",
        "a": (
            "bytes/token = 2(K+V) × L × H × d × bytes_per_elem\n"
            "            = 2 × 32 × 32 × 128 × 2 = 524288 B ≈ 524 KB/token\n"
            "4096 token: 4096 × 524 KB ≈ 2 GB\n"
            "\n注意：H 这里是 head 数，d 是 head_dim；"
            "多头时 H × d = hidden_dim。"
        ),
    },
    {
        "topic": "Roofline Model",
        "q": "RTX 5090 的 Ridge Point 是多少？如何计算？含义是什么？",
        "a": (
            "Ridge Point = Peak FLOP/s / Peak Bandwidth\n"
            "            = 19.5 TFLOPS / 1.55 TB/s ≈ 12.6 FLOP/Byte\n"
            "含义：算术强度 AI < 12.6 → memory-bound；AI > 12.6 → compute-bound。\n"
            "横轴 AI(FLOP/Byte)，纵轴 Performance(FLOP/s)；"
            "斜线段 = 带宽限制，水平段 = 算力限制。"
        ),
    },
    {
        "topic": "Prefill / Decode",
        "q": "Prefill 和 Decode 的输入形状、瓶颈类型、核心指标分别是什么？",
        "a": (
            "Prefill：输入 (B, N_prompt, d)，并行处理 prompt，compute-bound，指标 TTFT\n"
            "Decode ：输入 (B, 1, d)，自回归逐 token，memory-bound，指标 TBT\n"
            "\nDecode 阶段 M=1，GEMM 退化，arithmetic intensity 极低 → memory-bound。\n"
            "优化 Prefill 用 FlashAttention / chunked prefill；"
            "优化 Decode 用 KV Cache / PagedAttention / CUDA Graph。"
        ),
    },
    {
        "topic": "FlashAttention",
        "q": "FlashAttention 为什么快？FA1 和 FA2 的区别？",
        "a": (
            "快的原因：tiling + online softmax 在 SRAM 完成计算，"
            "不物化 N×N 矩阵，HBM IO 从 O(N²+Nd) 降到 O(Nd)。"
            "速度来自减少数据移动，不是减少 FLOPs。\n"
            "FA2 改进：① 减少非 matmul FLOPs ② 更好的 warp group 划分 "
            "③ 减少 warp 同步点 ④ 提高 occupancy。"
        ),
    },
    {
        "topic": "PagedAttention",
        "q": "PagedAttention 解决什么问题？核心设计是什么？",
        "a": (
            "问题：KV Cache 静态分配造成的显存碎片和浪费（reserved 但不用）。\n"
            "核心设计：① KV Cache 分成固定大小 block ② block table："
            "逻辑 block 连续，物理 block 可不连续 ③ copy-on-write 支持 prefix 共享。\n"
            "收益：提高显存利用率、支持动态长度、方便调度换出换入。"
        ),
    },
]

FORMULA_BLANKS = [
    {
        "prompt": "Online Softmax 的 m_new = ",
        "answer": "max(m, max(xj))",
    },
    {
        "prompt": "Online Softmax 的 l_new = l * ___ + Σ exp(xj - m_new)",
        "answer": "exp(m - m_new)",
    },
    {
        "prompt": "KV Cache bytes/token = 2 × L × H × ___ × bytes_per_elem",
        "answer": "d",
    },
    {
        "prompt": "GEMM FLOPs = 2 × M × ___ × K",
        "answer": "N",
    },
    {
        "prompt": "Roofline Ridge Point = Peak ___ / Peak Bandwidth",
        "answer": "FLOP/s",
    },
    {
        "prompt": "Arithmetic Intensity AI = ___ / Bytes",
        "answer": "FLOPs",
    },
    {
        "prompt": "Standard Attention HBM IO = O(N² + ___)",
        "answer": "Nd",
    },
]

PARAM_QUIZ = [
    {"q": "RTX 5090 FP32 Peak (TFLOPS)?", "a": "19.5"},
    {"q": "RTX 5090 Tensor Core FP16 (TFLOPS)?", "a": "312"},
    {"q": "RTX 5090 Memory Bandwidth (TB/s)?", "a": "1.55"},
    {"q": "RTX 5090 Ridge Point (FLOP/Byte)?", "a": "12.6"},
    {"q": "RTX 5090 Shared Memory per SM (KB)?", "a": "164"},
    {"q": "RTX 5090 Max Threads per SM?", "a": "2048"},
    {"q": "Warp Size?", "a": "32"},
    {"q": "Max Registers per Thread?", "a": "255"},
    {"q": "RTX 5090 Compute Capability (sm_?)?", "a": "120"},
    {"q": "LLaMA-7B KV Cache per token (KB, fp16)?", "a": "524"},
]


def mode_quiz():
    print("\n=== 模式：随机抽题口述 ===")
    print("每题限时 3 分钟口述，回车看答案，再回车下一题。输入 q 退出。\n")
    bank = QUIZ_BANK[:]
    random.shuffle(bank)
    for i, item in enumerate(bank, 1):
        print(f"【第 {i} 题 / {item['topic']}】")
        print(item["q"])
        print("-" * 50)
        start = time.time()
        cmd = input("回车看答案（q 退出）: ").strip()
        elapsed = time.time() - start
        if cmd == "q":
            break
        print(f"\n参考答案（用时 {elapsed:.0f}s）：")
        print(item["a"])
        print("\n" + "=" * 60 + "\n")


def mode_formula():
    print("\n=== 模式：关键公式默写 ===")
    print("填空，回车提交。输入 q 退出。\n")
    correct = 0
    blanks = FORMULA_BLANKS[:]
    random.shuffle(blanks)
    for i, item in enumerate(blanks, 1):
        print(f"【填空 {i}】{item['prompt']}")
        ans = input("你的答案: ").strip()
        if ans == "q":
            break
        if ans == item["answer"]:
            print("  ✓ 正确\n")
            correct += 1
        else:
            print(f"  ✗ 错误。正确答案：{item['answer']}\n")
    print(f"公式默写得分：{correct}/{len(blanks)}\n")


def mode_param():
    print("\n=== 模式：RTX 5090 参数快问快答 ===")
    print("输入数字/参数，回车提交。输入 q 退出。\n")
    correct = 0
    params = PARAM_QUIZ[:]
    random.shuffle(params)
    for i, item in enumerate(params, 1):
        print(f"【参数 {i}】{item['q']}")
        ans = input("你的答案: ").strip()
        if ans == "q":
            break
        if ans == item["a"]:
            print("  ✓ 正确\n")
            correct += 1
        else:
            print(f"  ✗ 错误。正确答案：{item['a']}\n")
    print(f"参数得分：{correct}/{len(params)}\n")


def main():
    print("=" * 60)
    print("       AI Infra 知识点自测系统（查漏补缺）")
    print("=" * 60)
    print("覆盖六大薄弱点 + 关键公式 + RTX 5090 参数")
    print("\n命令：")
    print("  quiz    —— 随机抽题，限时口述后看答案")
    print("  formula —— 关键公式默写（填空）")
    print("  param   —— RTX 5090 关键参数快问快答")
    print("  all     —— 依次执行三种模式")
    print("  q       —— 退出\n")
    while True:
        cmd = input("输入命令: ").strip()
        if cmd == "q":
            print("再见！查漏补缺，面试必过。")
            break
        elif cmd == "quiz":
            mode_quiz()
        elif cmd == "formula":
            mode_formula()
        elif cmd == "param":
            mode_param()
        elif cmd == "all":
            mode_quiz()
            mode_formula()
            mode_param()
        else:
            print("未知命令，可选：quiz / formula / param / all / q")


if __name__ == "__main__":
    main()
