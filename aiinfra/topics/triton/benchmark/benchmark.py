# benchmark.py —— Triton vs PyTorch vs CUDA 性能对比
# 运行: python3 benchmark/benchmark.py

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kernels'))
from gemm import matmul as triton_matmul


def benchmark_fn(fn, *args, n_iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / n_iters * 1000


def benchmark_gemm():
    print("=" * 70)
    print("GEMM Benchmark (FP16)")
    print("=" * 70)
    print(f"{'Size':<20} {'Triton (ms)':>12} {'PyTorch (ms)':>14} {'Ratio':>8}")
    print("-" * 56)

    sizes = [(512, 512, 512), (1024, 1024, 1024),
             (2048, 2048, 2048), (4096, 4096, 4096)]

    for M, N, K in sizes:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        triton_ms = benchmark_fn(triton_matmul, a, b)
        torch_ms = benchmark_fn(torch.matmul, a, b)

        ratio = torch_ms / triton_ms
        tflops = 2 * M * N * K / (triton_ms / 1000) / 1e12
        print(f"{M}x{N}x{K:<10} {triton_ms:>12.3f} {torch_ms:>14.3f} {ratio:>8.2f}x  ({tflops:.1f} TFLOPS)")


def benchmark_softmax():
    from softmax import softmax as triton_softmax

    print("\n" + "=" * 70)
    print("Softmax Benchmark (FP32)")
    print("=" * 70)
    print(f"{'Size':<20} {'Triton (ms)':>12} {'PyTorch (ms)':>14} {'Speedup':>8}")
    print("-" * 56)

    for M, N in [(128, 512), (1024, 4096), (4096, 4096)]:
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)

        triton_ms = benchmark_fn(triton_softmax, x)
        torch_ms = benchmark_fn(torch.softmax, x, dim=1)

        speedup = torch_ms / triton_ms
        print(f"{M}x{N:<14} {triton_ms:>12.3f} {torch_ms:>14.3f} {speedup:>8.2f}x")


def benchmark_flash_attention():
    from flash_attention import flash_attention, standard_attention

    print("\n" + "=" * 70)
    print("Attention Benchmark (FP16, D=64)")
    print("=" * 70)
    print(f"{'N':<10} {'Flash (ms)':>12} {'Standard (ms)':>16} {'Speedup':>8}")
    print("-" * 48)

    for N in [512, 1024, 2048, 4096]:
        q = torch.randn(2, N, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(2, N, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(2, N, 64, device='cuda', dtype=torch.float16)

        flash_ms = benchmark_fn(flash_attention, q, k, v)
        std_ms = benchmark_fn(standard_attention, q, k, v)

        speedup = std_ms / flash_ms
        print(f"{N:<10} {flash_ms:>12.3f} {std_ms:>16.3f} {speedup:>8.2f}x")


if __name__ == "__main__":
    benchmark_gemm()
    benchmark_softmax()
    benchmark_flash_attention()
