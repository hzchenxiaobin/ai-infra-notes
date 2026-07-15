# softmax.py —— Triton fused softmax
# 运行: python3 kernels/softmax.py

import torch
import triton
import triton.language as tl
import time


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(input_ptr + row * input_row_stride + cols,
                mask=mask, other=-float('inf'))

    row_max = tl.max(x, axis=0)
    x_centered = x - row_max
    numerator = tl.exp(x_centered)
    row_sum = tl.sum(numerator, axis=0)
    result = numerator / row_sum

    tl.store(output_ptr + row * output_row_stride + cols,
             result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](
        x, y, n_cols,
        x.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


if __name__ == "__main__":
    n_rows, n_cols = 128, 512
    x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)

    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=1)

    max_diff = (y_triton - y_torch).abs().max().item()
    print(f"Matrix: {n_rows} x {n_cols}")
    print(f"Max diff: {max_diff:.8f}")
    print(f"Passed: {torch.allclose(y_triton, y_torch, atol=1e-6)}")

    n_rows, n_cols = 4096, 4096
    x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)

    for _ in range(10):
        softmax(x)
        torch.softmax(x, dim=1)
    torch.cuda.synchronize()

    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        y_triton = softmax(x)
    torch.cuda.synchronize()
    triton_ms = (time.time() - start) / n_iters * 1000

    start = time.time()
    for _ in range(n_iters):
        y_torch = torch.softmax(x, dim=1)
    torch.cuda.synchronize()
    torch_ms = (time.time() - start) / n_iters * 1000

    print(f"\nPerformance ({n_rows}x{n_cols}):")
    print(f"  Triton:   {triton_ms:.3f} ms")
    print(f"  PyTorch:  {torch_ms:.3f} ms")
    print(f"  Speedup:  {torch_ms / triton_ms:.2f}x")
