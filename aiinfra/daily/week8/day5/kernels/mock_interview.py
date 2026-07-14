# mock_interview.py —— Mock 面试计时与提纲系统
# 运行命令: python mock_interview.py
# 依赖: 仅标准库

import time

SECTIONS = [
    {
        "name": "自我介绍",
        "duration": 120,
        "prompt": (
            "1. 姓名、背景\n"
            "2. 目前方向\n"
            "3. 项目核心亮点（一句话）\n"
            "4. 希望应聘的岗位"
        ),
    },
    {
        "name": "项目介绍",
        "duration": 300,
        "prompt": (
            "用 STAR 法介绍 Mini AI Infra 项目：\n"
            "Situation：为什么要做这个项目？\n"
            "Task：你的目标是什么？\n"
            "Action：你做了什么？（CUDA kernel、调度器、KV Cache）\n"
            "Result：量化成果（cuBLAS 70%、吞吐提升 X 倍）"
        ),
    },
    {
        "name": "技术难点 1：FlashAttention / Online Softmax",
        "duration": 360,
        "prompt": (
            "讲清楚：\n"
            "1. 标准 Attention 的 IO 瓶颈在哪里\n"
            "2. FlashAttention 怎么用 tiling + online softmax 解决\n"
            "3. 手写 online softmax 三公式并解释缩放因子\n"
            "4. 你在实现中遇到的最大挑战"
        ),
    },
    {
        "name": "技术难点 2：Continuous Batching / vLLM 调度",
        "duration": 360,
        "prompt": (
            "讲清楚：\n"
            "1. Dynamic Batching vs Continuous Batching 区别\n"
            "2. vLLM 的 LLMEngine → Scheduler → Worker 数据流\n"
            "3. PagedAttention 如何解决 KV Cache 碎片\n"
            "4. 抢占策略 Recompute vs Swap 的 trade-off"
        ),
    },
    {
        "name": "优化思路：给一个 kernel 怎么优化到 80% cuBLAS",
        "duration": 360,
        "prompt": (
            "从以下八层路径展开：\n"
            "Naive → Tiling → Register Blocking → float4 → Warp Shuffle → Double Buffer → Tensor Core → Auto-tuning\n"
            "每层说明：做了什么、为什么快、达到多少比例"
        ),
    },
    {
        "name": "场景题：设计一个 LLM 推理服务",
        "duration": 420,
        "prompt": (
            "从请求接入到 GPU 调度完整展开：\n"
            "1. 请求接入与 tokenize\n"
            "2. KV Cache 管理（PagedAttention）\n"
            "3. Continuous Batching Scheduler\n"
            "4. 模型执行与算子优化\n"
            "5. 异步返回与流式输出\n"
            "6. 监控、扩缩容与容错"
        ),
    },
    {
        "name": "反问环节",
        "duration": 120,
        "prompt": (
            "准备 2-3 个问题问面试官，例如：\n"
            "1. 团队目前在推理优化的重点方向是什么？\n"
            "2. 这个岗位日常更多做 kernel 还是做系统架构？\n"
            "3. 团队对新人的培养机制是怎样的？"
        ),
    },
]


def countdown(seconds):
    """倒计时，按回车可提前结束。"""
    end = time.time() + seconds
    while time.time() < end:
        remaining = int(end - time.time())
        if remaining <= 0:
            break
        mins, secs = divmod(remaining, 60)
        print(f"\r剩余时间: {mins:02d}:{secs:02d}  (按回车提前结束)", end="", flush=True)
        # 非阻塞等待 1 秒
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    print()


def run_section(section):
    print("=" * 60)
    print(f"【{section['name']}】限时 {section['duration']} 秒")
    print("-" * 60)
    print(section["prompt"])
    print("-" * 60)
    input("准备就绪后按回车开始...")
    start = time.time()
    countdown(section["duration"])
    elapsed = int(time.time() - start)
    print(f"本环节用时 {elapsed} 秒\n")
    return elapsed


def run_full_mock():
    print("=== AI Infra Mock 面试 ===")
    print("共 7 个环节，全程约 30 分钟\n")
    times = []
    for sec in SECTIONS:
        times.append(run_section(sec))
    print("=" * 60)
    print("Mock 面试结束！")
    print(f"总用时: {sum(times)} 秒")
    print("\n复盘清单：")
    print("  [ ] 是否超时？哪个环节需要压缩？")
    print("  [ ] 是否有口头禅（然后、那个、就是）？")
    print("  [ ] 技术点是否讲清楚了 why 而不仅是 what？")
    print("  [ ] 是否准备了 2-3 个反问问题？")


def show_outline():
    print("=== Mock 面试提纲 ===\n")
    for sec in SECTIONS:
        print(f"【{sec['name']}】{sec['duration']} 秒")
        print(sec["prompt"])
        print()


if __name__ == "__main__":
    print("命令：")
    print("  start  — 开始完整 Mock 面试")
    print("  outline — 查看面试提纲")
    cmd = input("\n输入命令: ").strip()
    if cmd == "start":
        run_full_mock()
    elif cmd == "outline":
        show_outline()
    else:
        run_full_mock()
