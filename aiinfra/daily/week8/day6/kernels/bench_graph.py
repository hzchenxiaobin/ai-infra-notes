"""bench_graph.py —— Mini 引擎 decode 路径 eager vs CUDA Graph TBT 对比基准

测量对象（WS-3.1 整改）：Day 5 真整合产出的 MiniEngineV1Graph
（week8/day5/kernels/mini_engine_v1_graph.py），而不是独立的合成模型。
- eager 模式：use_graph=False，decode 逐请求 eager forward
- graph 模式：use_graph=True，decode 走 BucketedGraphRunner replay

微观 launch-gap 演示（10 层 Linear+LayerNorm 合成模型）仍在 bench_eager.py，
本文件负责引擎级 TBT（token-by-token）延迟对比。

运行: python3 bench_graph.py
依赖: pip install torch（CUDA Graph 对比需 GPU；无 GPU 时只跑 eager 基线）
nsys 验证 launch gap 消除:
    nsys profile --trace cuda -o graph_profile python3 bench_graph.py
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
# 从 week8/day6/kernels 回退到 week8/，再进入 day5/kernels
sys.path.insert(0, os.path.join(_here, "..", "..", "day5", "kernels"))

try:
    import torch

    from mini_engine_v1_graph import MiniEngineV1Graph, MiniLLM, MiniTokenizer
except ImportError as e:
    sys.exit(
        f"[错误] 无法导入 Day 5 的 mini_engine_v1_graph（{e}）。\n"
        f"  本脚本依赖 week8/day5/kernels/mini_engine_v1_graph.py（WS-3.1 真整合产出），\n"
        f"  请先确认该文件存在；仅想看合成模型的微观 launch-gap 演示请运行 bench_eager.py。"
    )


PROMPTS = [
    "hello world",
    "this is a longer prompt for testing",
    "short",
    "another test prompt here now",
    "a medium length prompt for batching",
    "yet another prompt to vary the load",
    "tiny",
    "one more prompt to keep the batch busy",
]


def run_engine(model, tokenizer, device, use_graph, rounds=3, max_new_tokens=6):
    """提交 4 请求并发负载，收集引擎记录的 decode TBT 时间（ms/step）。"""
    decode_times = []
    for _ in range(rounds):
        engine = MiniEngineV1Graph(
            model, tokenizer, max_token_budget=64, max_num_seqs=4,
            device=device, use_graph=use_graph,
        )
        futures = []
        for i in range(4):
            p = PROMPTS[i % len(PROMPTS)]
            futures.append(engine.submit(f"{p} round{len(decode_times)}",
                                         max_new_tokens=max_new_tokens, priority=i % 2))
        for f in futures:
            f.result(timeout=30)
        decode_times.extend(engine.decode_times)
        engine.shutdown()
    return decode_times


def report(name, times):
    if not times:
        print(f"  {name}: 无采样（CUDA 事件未记录）")
        return None
    avg = sum(times) / len(times)
    print(f"  {name}: avg {avg:.3f} ms/step | "
          f"min {min(times):.3f} | max {max(times):.3f} | {len(times)} steps")
    return avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = MiniLLM(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
    tokenizer = MiniTokenizer(vocab_size=1000)

    print("\n=== 测量对象：MiniEngineV1Graph decode 路径（Day 5 真整合引擎）===")

    eager_times = run_engine(model, tokenizer, device, use_graph=False)
    graph_times = run_engine(model, tokenizer, device, use_graph=True)

    print("\n--- TBT（token-by-token decode）延迟 ---")
    avg_e = report("Eager decode", eager_times)
    avg_g = report("Graph decode", graph_times)

    if avg_e and avg_g and avg_g > 0:
        print(f"\n  加速比: {avg_e / avg_g:.2f}x | 延迟降低: {(1 - avg_g / avg_e) * 100:.1f}%")
    elif device != "cuda":
        print("\n  [无 GPU] Graph 模式未启用（引擎自动回退 eager），跳过对比。")
        print("  需在 CUDA 环境实跑回填 README §6.1 的引擎级留档模板。")
    else:
        print("\n  [Graph 未捕获] 请检查 MiniEngineV1Graph 的捕获日志（回退 eager 时会打印 warn）。")

    if device == "cuda":
        print("\n用 nsys 抓 launch gap（Graph 应把 decode 的多次 launch 压成一次 replay）:")
        print("  nsys profile --trace cuda -o graph_profile python3 bench_graph.py")
        print("  nsys stats graph_profile.nsys-rep --report cuda_gpu_kern")


if __name__ == "__main__":
    main()
