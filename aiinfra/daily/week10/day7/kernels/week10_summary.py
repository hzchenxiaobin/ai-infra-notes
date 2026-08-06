#!/usr/bin/env python3
# week10_summary.py —— Week 10 + 10 周总复盘自测系统
# 运行命令: python week10_summary.py
# 依赖: 仅标准库
#
# 五个模块：
#   1. Week 10 知识地图（7 天回顾）
#   2. 10 周能力地图 Checklist（自评 强项/待提升）
#   3. 30+ 面试题速查（按主题分组）
#   4. 关键公式 / RTX 5090 参数速答
#   5. 后续 6 个月规划填写

WEEK10_DAYS = [
    ("Day 1", "整合全部自定义 Kernel", "custom_ops_module.py（LayerNorm/Softmax/FlashAttention C++ Extension）"),
    ("Day 2", "系统联调（六步分层验证）", "mini_engine_v2.py + stability_test.py（500+ 请求稳定）"),
    ("Day 3", "项目文档完善", "README、Quick Start、依赖安装、Benchmark 结果"),
    ("Day 4", "高频面试题基础篇", "GPU 基础、Kernel 优化、CUDA 编程、Profiling"),
    ("Day 5", "Mock 面试", "STAR 法、技术难点深挖、Follow-up、录音复盘"),
    ("Day 6", "诊断流程实战剧本 + 手撕清单", "ncu/py-spy 诊断流程、手撕限时清单"),
    ("Day 7", "最终复盘", "10 周能力地图、后续路线、最终报告"),
]

CAPABILITY = [
    ("Kernel 优化", [
        ("GPU 执行模型（SM/Warp/Occupancy）", True),
        ("Shared Memory Tiling + Bank Conflict", True),
        ("Register Blocking + float4 + Warp Shuffle", True),
        ("ncu 瓶颈分析（Roofline / Stall）", True),
        ("手写 FlashAttention Forward Kernel", True),
        ("Tensor Core / WMMA / mma.sync（Week 3）", True),
        ("Double Buffering 完整实现（cp.async 已学，待 TMA）", False),
        ("CUTLASS 源码深度阅读", False),
    ]),
    ("推理系统", [
        ("Prefill / Decode 区别与瓶颈", True),
        ("KV Cache 设计与内存计算（含 GQA/MQA/MLA）", True),
        ("PagedAttention 概念与 block table", True),
        ("Continuous Batching 实现", True),
        ("Scheduler（双预算 + 抢占 + aging）", True),
        ("Mini 推理引擎 v2（500+ 请求稳定，真整合）", True),
        ("Chunked Prefill / Prefix Caching / PD 分离", True),
        ("PagedAttention 完整 CUDA 实现", False),
        ("C++ Scheduler（降 CPU overhead）", False),
    ]),
    ("分布式与量化（Week 8–9）", [
        ("TP/PP/DP 三维并行与通信量（概念）", True),
        ("1F1B 调度与 bubble ratio 公式", True),
        ("NCCL Collectives（ring all-reduce）", True),
        ("通信计算重叠（双 Stream + CUDA Graph）", True),
        ("MoE + EP 并行", True),
        ("量化（W8A16/INT8/FP8/FP4）对推理的影响", True),
        ("CUDA Graph 消除 Launch Overhead", True),
        ("分布式实操（torchrun + NCCL 实测）", False),
        ("FP8 GEMM 实操（torch._scaled_mm）", False),
        ("真实模型 vLLM 部署压测", False),
    ]),
    ("Profiling", [
        ("nsys 系统级时间线", True),
        ("ncu kernel 级指标 + Roofline 判 bound", True),
        ("端到端阶段计时 + vLLM 对比", True),
        ("诊断流程（ncu/py-spy 三层定位）", True),
    ]),
    ("系统设计", [
        ("设计 LLM 推理服务（6 要素）", True),
        ("vLLM 架构讲解 + trade-off", True),
        ("并行策略选型（TP vs PP vs DP vs EP）", True),
    ]),
    ("工程表达", [
        ("README + 架构图 + Benchmark 报告", True),
        ("面试题自问自答（基础 + 进阶 + 场景）", True),
        ("Mock 面试 + 录音复盘", True),
        ("STAR 法项目介绍 + 技术难点深挖", True),
        ("关键公式 / 参数秒答", True),
        ("核心流程图默画", True),
    ]),
]

INTERVIEW_QUESTIONS = [
    ("基础", "介绍一下你的 Mini AI Infra 项目", 5),
    ("基础", "GEMM 优化到 FP32 cuBLAS ~63%，每层收益？", 5),
    ("基础", "float4 向量化加载为什么快？条件？", 4),
    ("基础", "Warp Shuffle 比 Shared Memory 快多少？", 4),
    ("基础", "什么是 Occupancy？什么情况会降低？", 4),
    ("基础", "GPU memory hierarchy + 延迟", 4),
    ("基础", "Bank conflict 是什么？如何避免？", 4),
    ("基础", "如何分析一个 kernel 的瓶颈？", 4),
    ("基础", "Roofline Model + Ridge Point 计算", 4),
    ("进阶", "FlashAttention 为什么快？", 5),
    ("进阶", "推导 online softmax 三公式", 5),
    ("进阶", "FA1 和 FA2 的区别？", 4),
    ("进阶", "Prefill 和 Decode 的区别？", 5),
    ("进阶", "KV Cache 核心思想 + 内存计算", 5),
    ("进阶", "PagedAttention 解决什么问题？", 5),
    ("进阶", "Continuous vs Dynamic Batching", 5),
    ("进阶", "TP/PP/DP 通信量怎么比较？bubble ratio 怎么算？", 4),
    ("进阶", "FP8 vs INT8 vs FP4 的取舍？NVFP4 规格？", 4),
    ("进阶", "CUDA Graph 消除 Launch Overhead 的原理？", 4),
    ("进阶", "MoE + EP 并行的通信量？", 4),
    ("进阶", "设计一个 LLM 推理服务", 5),
    ("项目", "项目最大技术难点？如何解决？", 5),
    ("项目", "如果继续优化会做什么？", 5),
    ("项目", "你的 kernel 和官方差距多少？", 5),
    ("项目", "如何把自定义 kernel 集成到 PyTorch？", 5),
    ("项目", "custom kernel 真整合遇到什么坑？", 5),
    ("项目", "Mini 系统和 vLLM 差距在哪？", 5),
    ("项目", "显存不够怎么办？", 4),
    ("成长", "10 周最大收获和挑战？", 4),
    ("成长", "未来 3-6 个月规划？", 4),
]

KEY_FORMULAS = [
    "Online Softmax: m_new=max(m,max(xj)); l_new=l·exp(m-m_new)+Σexp(xj-m_new)",
    "KV Cache: bytes/token = 2 × L × H × d × bytes_per_elem （LLaMA-7B ≈ 524 KB）",
    "GEMM FLOPs = 2·M·N·K ；AI = FLOPs / Bytes",
    "Ridge Point = Peak FLOP/s / Peak Bandwidth （RTX5090 ≈ 58.45）",
    "FlashAttention IO: Standard O(N²+Nd) → FA O(Nd)",
    "Bubble ratio: (P-1)/(M+P-1) ；Interleaved: (P-1)/(V·M+P-1)",
    "TP 通信量: 2 × tokens × hidden（每层 all-reduce）",
]

RTX5090_PARAMS = [
    ("FP32 Peak", "104.75 TFLOPS"),
    ("Tensor Core FP16", "~209 TFLOPS (dense)"),
    ("Memory Bandwidth", "1.792 TB/s (GDDR7)"),
    ("Ridge Point", "58.45 FLOP/Byte"),
    ("SM 数量", "170"),
    ("Shared Memory / SM", "100 KB"),
    ("Max Threads / SM", "1536"),
    ("Warp Size", "32"),
    ("Compute Capability", "sm_120"),
    ("LLaMA-7B KV / token (fp16)", "524 KB"),
]


def section1_knowledge_map():
    print("\n" + "=" * 60)
    print("📊 1. Week 10 知识地图（7 天回顾）")
    print("=" * 60)
    for day, topic, output in WEEK10_DAYS:
        mark = "★" if day == "Day 7" else " "
        print(f"  {mark} {day}: {topic}")
        print(f"       → {output}")


def section2_capability_map():
    print("\n" + "=" * 60)
    print("📊 2. 10 周能力地图 Checklist（✅ 强项 / ⚠️ 待提升）")
    print("=" * 60)
    strong = 0
    weak = 0
    for area, items in CAPABILITY:
        print(f"\n  【{area}】")
        for skill, mastered in items:
            mark = "✅" if mastered else "⚠️"
            print(f"    {mark} {skill}")
            if mastered:
                strong += 1
            else:
                weak += 1
    total = strong + weak
    print(f"\n  汇总：✅ 强项 {strong}/{total}　⚠️ 待提升 {weak}/{total}")
    print(f"  强项占比：{strong * 100 // total}%")
    print("  → 待提升项即为后续 3 个月规划的重点")


def section3_interview_quiz():
    print("\n" + "=" * 60)
    print(f"📊 3. 面试题速查（{len(INTERVIEW_QUESTIONS)} 道高频，按主题分组）")
    print("=" * 60)
    by_topic = {}
    for topic, q, freq in INTERVIEW_QUESTIONS:
        by_topic.setdefault(topic, []).append((q, freq))
    idx = 1
    for topic in ["基础", "进阶", "项目", "成长"]:
        print(f"\n  【{topic}篇】")
        for q, freq in by_topic.get(topic, []):
            stars = "⭐" * freq
            print(f"   {idx:2d}. {q}  {stars}")
            idx += 1
    print(f"\n  共 {len(INTERVIEW_QUESTIONS)} 题，频率 ⭐⭐⭐⭐⭐ 为必考")
    cmd = input("\n  随机抽 1 题口述练习？（回车继续，q 跳过）: ").strip()
    if cmd == "q":
        return
    import random
    pick = random.choice(INTERVIEW_QUESTIONS)
    print(f"\n  抽到：【{pick[0]}】{pick[1]}  {'⭐' * pick[2]}")
    print("  → 限时 3 分钟口述，回车看下一题")


def section4_formulas_params():
    print("\n" + "=" * 60)
    print("📊 4. 关键公式 + RTX 5090 参数速答")
    print("=" * 60)
    print("\n  【关键公式】（默写检查）")
    for f in KEY_FORMULAS:
        print(f"   • {f}")
    print("\n  【RTX 5090 参数】（秒答检查）")
    for k, v in RTX5090_PARAMS:
        print(f"   • {k}: {v}")
    print("\n  → 闭眼能背 = 过关；卡壳 = 回 Day 6 查漏补缺")


def section5_roadmap():
    print("\n" + "=" * 60)
    print("📊 5. 后续 6 个月规划")
    print("=" * 60)
    roadmap = [
        ("Month 1", "深化 Kernel", "PagedAttention CUDA + CUTLASS 深度 + TMA", "GEMM cuBLAS 90%+"),
        ("Month 2", "系统强化", "C++ Scheduler + CUDA Graph 实操 + Chunked Prefill", "调度延迟降 10x"),
        ("Month 3", "多卡实操", "torchrun + NCCL 实测 + 真实模型 TP 部署", "多 GPU 推理跑通"),
        ("Month 4", "量化实操", "FP8 GEMM（torch._scaled_mm）+ vLLM 部署压测", "FP8 推理跑通"),
        ("Month 5", "多模态与长文本", "Multimodal + 100K 上下文 + MoE", "支持长上下文/MoE"),
        ("Month 6", "面试与影响力", "面试反馈 + 博客 + 开源", "建立个人影响力"),
    ]
    for m, theme, tasks, goal in roadmap:
        print(f"\n  【{m}】{theme}")
        print(f"    任务：{tasks}")
        print(f"    目标：{goal}")
    print("\n  原则：每月一个主线 + 一个量化目标 + 一个可展示产出")


def main():
    print("=" * 60)
    print("     Week 10 Day 7：10 周最终复盘")
    print("=" * 60)
    print("五个模块：知识地图 / 能力地图 / 面试题 / 公式参数 / 路线规划")
    print("\n命令：")
    print("  all  —— 依次输出五个模块")
    print("  1-5  —— 单独输出某模块")
    print("  q    —— 退出\n")
    while True:
        cmd = input("输入命令: ").strip()
        if cmd == "q":
            print("\n10 周学习完成！这是终点，也是新起点。面试顺利！")
            break
        elif cmd == "all":
            section1_knowledge_map()
            section2_capability_map()
            section3_interview_quiz()
            section4_formulas_params()
            section5_roadmap()
        elif cmd == "1":
            section1_knowledge_map()
        elif cmd == "2":
            section2_capability_map()
        elif cmd == "3":
            section3_interview_quiz()
        elif cmd == "4":
            section4_formulas_params()
        elif cmd == "5":
            section5_roadmap()
        else:
            print("未知命令，可选：all / 1-5 / q")


if __name__ == "__main__":
    main()
