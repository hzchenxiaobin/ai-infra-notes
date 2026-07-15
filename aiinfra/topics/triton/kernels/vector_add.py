# vector_add.py —— Triton 入门：向量加法
# 运行: python3 kernels/vector_add.py

import torch
import triton
import triton.language as tl
import time


@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, z_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z


if __name__ == "__main__":
    # 正确性测试
    n = 1_000_000
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')

    z_triton = vector_add(x, y)
    z_torch = x + y

    max_diff = (z_triton - z_torch).abs().max().item()
    print(f"Vector size: {n}")
    print(f"Max diff:    {max_diff}")
    print(f"Passed:      {torch.allclose(z_triton, z_torch)}")

    # 性能对比
    n = 100_000_000
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')

    for _ in range(10):
        vector_add(x, y)
        x + y
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        z_triton = vector_add(x, y)
    torch.cuda.synchronize()
    triton_ms = (time.time() - start) / 100 * 1000

    start = time.time()
    for _ in range(100):
        z_torch = x + y
    torch.cuda.synchronize()
    torch_ms = (time.time() - start) / 100 * 1000

    bandwidth = 3 * n * 4 / (triton_ms / 1000) / 1e9

    print(f"\nPerformance (n={n}):")
    print(f"  Triton:   {triton_ms:.3f} ms  ({bandwidth:.1f} GB/s)")
    print(f"  PyTorch:  {torch_ms:.3f} ms")
    print(f"  Ratio:    {torch_ms / triton_ms:.2f}x")
