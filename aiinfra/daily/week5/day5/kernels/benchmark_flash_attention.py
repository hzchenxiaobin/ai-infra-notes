# benchmark_flash_attention.py —— FlashAttention 性能对比框架
# 运行命令: python benchmark_flash_attention.py

import torch
import torch.nn.functional as F
import math
import json

try:
    from flash_attn import flash_attn_func
    HAS_OFFICIAL = True
except ImportError:
    HAS_OFFICIAL = False
    print("Warning: official flash_attn not installed, skipping official benchmark")

def standard_attention(Q, K, V):
    d = Q.size(-1)
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    P = F.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O

def benchmark(func, Q, K, V, n_iter=10):
    for _ in range(3):
        _ = func(Q, K, V)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        out = func(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iter
    return ms

def theoretical_io(N, d, dtype_size=4):
    std_io = (3 * N * N + 4 * N * d) * dtype_size / (1024 * 1024)
    fa_io = (4 * N * d) * dtype_size / (1024 * 1024)
    return std_io, fa_io

def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    configs = [
        {"B": 1, "H": 8, "N": 512, "d": 64},
        {"B": 1, "H": 8, "N": 1024, "d": 64},
        {"B": 1, "H": 8, "N": 2048, "d": 64},
        {"B": 1, "H": 8, "N": 4096, "d": 64},
        {"B": 1, "H": 8, "N": 8192, "d": 64},
        {"B": 4, "H": 8, "N": 2048, "d": 64},
        {"B": 1, "H": 16, "N": 2048, "d": 128},
    ]

    results = []

    print("=== FlashAttention Performance Benchmark ===")
    print(f"{'B':>3} {'H':>3} {'N':>5} {'d':>4} | {'Std(ms)':>10} {'Hand(ms)':>10} {'Off(ms)':>10} | {'Hand-Spd':>10} {'Off-Spd':>10} | {'StdIO(MB)':>10} {'FAIO(MB)':>10}")
    print("-" * 110)

    for cfg in configs:
        B, H, N, d = cfg["B"], cfg["H"], cfg["N"], cfg["d"]

        Q = torch.randn(B, H, N, d, device=device, dtype=dtype)
        K = torch.randn(B, H, N, d, device=device, dtype=dtype)
        V = torch.randn(B, H, N, d, device=device, dtype=dtype)

        ms_std = benchmark(standard_attention, Q, K, V)

        try:
            from mini_engine_fa import fa_ops
            ms_hand = benchmark(fa_ops.flash_attention_forward, Q, K, V)
            hand_speedup = ms_std / ms_hand
        except Exception:
            ms_hand = float('nan')
            hand_speedup = float('nan')

        if HAS_OFFICIAL:
            ms_off = benchmark(flash_attn_func, Q, K, V)
            off_speedup = ms_std / ms_off
        else:
            ms_off = float('nan')
            off_speedup = float('nan')

        std_io, fa_io = theoretical_io(N, d)

        print(f"{B:>3} {H:>3} {N:>5} {d:>4} | {ms_std:>10.3f} {ms_hand:>10.3f} {ms_off:>10.3f} | {hand_speedup:>10.2f}x {off_speedup:>10.2f}x | {std_io:>10.2f} {fa_io:>10.2f}")

        results.append({
            "B": B, "H": H, "N": N, "d": d,
            "std_ms": ms_std, "hand_ms": ms_hand, "off_ms": ms_off,
            "hand_speedup": hand_speedup, "off_speedup": off_speedup,
            "std_io_mb": std_io, "fa_io_mb": fa_io,
        })

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()
