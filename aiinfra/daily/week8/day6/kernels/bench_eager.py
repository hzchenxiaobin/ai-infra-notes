"""bench_eager.py —— Eager 模式 decode-like forward 微观 launch-gap 演示

用 nsys 抓 kernel 间 launch gap（需 GPU + CUDA）：
    nsys profile --trace cuda -o eager_profile python3 bench_eager.py

模型：10 层 Linear+LayerNorm（decode-like, seq=32）合成模型，
用于在 kernel 粒度直观展示 eager 模式的 launch overhead。

注意（WS-3.1 整改后分工）：本文件是**微观演示**；引擎级 eager vs CUDA Graph
的 TBT 对比基准在 bench_graph.py，测量对象是 Day 5 真整合的 MiniEngineV1Graph。

运行: python3 bench_eager.py
依赖: pip install torch

无 GPU 说明：CPU 环境下用 time.perf_counter 做粗略计时并打印结果；
精确的 kernel-level launch gap 与 CUDA Event 计时需在 GPU 环境实测。
"""

import torch
import torch.nn as nn


class DecodeLikeModel(nn.Module):
    """10 层 Linear+LayerNorm，seq=32，模拟 decode forward。"""

    def __init__(self, d_model=128, n_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def bench_eager(model, input_ids, iters=100):
    if input_ids.is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = model(input_ids)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    # CPU / 无 CUDA 时用 time.time 做粗略计时，并提示精确 timing 需 GPU
    import time
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    return elapsed_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    d_model, seq_len, n_layers = 128, 32, 10
    model = DecodeLikeModel(d_model, n_layers).to(device).eval()
    x = torch.randn(1, seq_len, d_model, device=device)

    if device == "cuda":
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

    avg_ms = bench_eager(model, x)
    print(f"=== Eager decode-like (10 layers, seq=32) ===")
    print(f"Eager: {avg_ms:.3f} ms/iter")

    if device == "cuda":
        print(f"\n用 nsys 抓 launch gap:")
        print(f"  nsys profile --trace cuda -o eager_profile python3 bench_eager.py")
        print(f"  nsys stats eager_profile.nsys-rep --report cuda_gpu_kern")


if __name__ == "__main__":
    main()
