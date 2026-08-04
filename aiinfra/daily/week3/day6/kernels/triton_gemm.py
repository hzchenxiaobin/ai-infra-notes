import torch
import triton
import triton.language as tl


autotune_configs = [
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
]


@triton.autotune(configs=autotune_configs, key=["M", "N", "K"])
@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "shape mismatch for GEMM"
    M, K = a.shape
    K2, N = b.shape
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


@torch.no_grad()
def _benchmark(fn, a, b, n_warmup=10, n_iter=50) -> float:
    for _ in range(n_warmup):
        fn(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def main():
    torch.manual_seed(42)
    sizes = [512, 1024, 2048, 4096]
    print("=== Triton GEMM (autotune) vs torch.matmul (cuBLAS) ===")
    header = f"{'M=N=K':<10}{'cuBLAS(ms)':<14}{'triton(ms)':<14}{'max_diff':<14}{'speedup':<10}{'check':<6}"
    print(header)
    print("-" * len(header))
    for size in sizes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        y_torch = torch.matmul(a, b)
        y_triton = triton_gemm(a, b)
        max_diff = (y_torch - y_triton).abs().max().item()
        t_torch = _benchmark(torch.matmul, a, b)
        t_triton = _benchmark(triton_gemm, a, b)
        speedup = t_torch / t_triton
        ok = "PASS" if max_diff < 1e-2 else "FAIL"
        print(f"{size:<10}{t_torch:<14.4f}{t_triton:<14.4f}{max_diff:<14.2e}{speedup:<10.2f}{ok:<6}")


if __name__ == "__main__":
    main()
