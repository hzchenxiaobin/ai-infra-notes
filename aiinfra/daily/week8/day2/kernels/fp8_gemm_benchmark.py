"""fp8_gemm_benchmark.py —— FP8 vs FP16 GEMM 实测 benchmark

用 torch._scaled_mm 做 FP8 E4M3 GEMM，对比 FP16 GEMM 的性能与精度。
在 RTX 5090 (sm_120, Blackwell) 上实测 FP8 Tensor Core 的算力翻倍收益。

运行: python3 fp8_gemm_benchmark.py
依赖: pip install torch (需 CUDA 12.8+ / sm_120+ 支持 FP8)
"""

import torch
import time


def benchmark_fp16_gemm(M, N, K, warmup=5, iters=20):
    """FP16 GEMM: torch.matmul"""
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / iters
    tflops = 2 * M * N * K / elapsed / 1e12
    return elapsed * 1000, tflops, c


def benchmark_fp8_gemm(M, N, K, warmup=5, iters=20):
    """FP8 E4M3 GEMM: torch._scaled_mm"""
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8 = b.t().to(torch.float8_e4m3fn)

    scale = torch.tensor(1.0, device='cuda')

    for _ in range(warmup):
        torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale, out_dtype=torch.float16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        c = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale, scale_b=scale, out_dtype=torch.float16)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / iters
    tflops = 2 * M * N * K / elapsed / 1e12
    return elapsed * 1000, tflops, c


def benchmark_bf16_gemm(M, N, K, warmup=5, iters=20):
    """BF16 GEMM: torch.matmul (BF16 Tensor Core)"""
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / iters
    tflops = 2 * M * N * K / elapsed / 1e12
    return elapsed * 1000, tflops, c


def main():
    print("=" * 70)
    print("FP8 vs FP16 vs BF16 GEMM Benchmark")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    if not hasattr(torch, '_scaled_mm'):
        print("[ERROR] torch._scaled_mm 不可用，需 PyTorch 2.1+ / CUDA 12.0+")
        return
    if not hasattr(torch, 'float8_e4m3fn'):
        print("[ERROR] torch.float8_e4m3fn 不可用，需 PyTorch 2.1+ / sm_89+")
        return

    print(f"{'M×N×K':>16} {'FP16 (ms)':>10} {'FP16 (TF)':>10} "
          f"{'BF16 (ms)':>10} {'BF16 (TF)':>10} "
          f"{'FP8 (ms)':>10} {'FP8 (TF)':>10} {'FP8/FP16 加速':>14} {'最大误差':>10}")
    print("-" * 110)

    for M, N, K in [(1024, 1024, 1024),
                     (2048, 2048, 2048),
                     (4096, 4096, 4096),
                     (8192, 8192, 8192)]:
        try:
            ms_f16, tf_f16, c_f16 = benchmark_fp16_gemm(M, N, K)
            ms_bf16, tf_bf16, c_bf16 = benchmark_bf16_gemm(M, N, K)
            ms_f8, tf_f8, c_f8 = benchmark_fp8_gemm(M, N, K)

            speedup = ms_f16 / ms_f8
            max_err = (c_f16.float() - c_f8.float()).abs().max().item()

            print(f"{f'{M}x{N}x{K}':>16} "
                  f"{ms_f16:>9.3f} {tf_f16:>9.1f} "
                  f"{ms_bf16:>9.3f} {tf_bf16:>9.1f} "
                  f"{ms_f8:>9.3f} {tf_f8:>9.1f} "
                  f"{speedup:>13.2f}x {max_err:>9.2f}")
        except Exception as e:
            print(f"{f'{M}x{N}x{K}':>16} ERROR: {e}")

    print()
    print("结论:")
    print("  - FP8 E4M3 GEMM 通过 torch._scaled_mm 调用 FP8 Tensor Core")
    print("  - 理论上 FP8 算力是 FP16 的 2x（Blackwell sm_120）")
    print("  - 实测加速比取决于矩阵大小（小矩阵 launch overhead 主导，大矩阵带宽/算力主导）")
    print("  - 精度：FP8 量化引入误差，最大误差随矩阵增大而增大（累加次数多）")
    print("  - Scaling factor：per-tensor scale=1.0 是最简方案，生产用 per-block (MXFP8)")


if __name__ == "__main__":
    main()
