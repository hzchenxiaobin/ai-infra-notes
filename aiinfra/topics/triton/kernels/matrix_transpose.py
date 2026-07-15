# matrix_transpose.py —— Triton 矩阵转置
# 运行: python3 kernels/matrix_transpose.py

import torch
import triton
import triton.language as tl
import time


@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    input_block = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    output_block = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, M),
        strides=(stride_on, stride_om),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )

    x = tl.load(input_block)
    x_t = tl.trans(x)
    tl.store(output_block, x_t)


def transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    y = torch.empty(N, M, device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        x, y, M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return y


if __name__ == "__main__":
    M, N = 512, 768
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)

    y_triton = transpose(x)
    y_torch = x.t().contiguous()

    max_diff = (y_triton - y_torch).abs().max().item()
    print(f"Matrix: {M} x {N}")
    print(f"Max diff: {max_diff}")
    print(f"Passed: {torch.allclose(y_triton, y_torch)}")

    for _ in range(10):
        transpose(x)
        x.t().contiguous()
    torch.cuda.synchronize()

    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        transpose(x)
    torch.cuda.synchronize()
    triton_ms = (time.time() - start) / n_iters * 1000

    start = time.time()
    for _ in range(n_iters):
        x.t().contiguous()
    torch.cuda.synchronize()
    torch_ms = (time.time() - start) / n_iters * 1000

    bandwidth = 2 * M * N * 4 / (triton_ms / 1000) / 1e9

    print(f"\nPerformance ({M}x{N}):")
    print(f"  Triton:   {triton_ms:.3f} ms  ({bandwidth:.1f} GB/s)")
    print(f"  PyTorch:  {torch_ms:.3f} ms")
    print(f"  Ratio:    {torch_ms / triton_ms:.2f}x")
