# reduction.py —— Triton block 级 reduce
# 运行: python3 kernels/reduction.py

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    x_ptr, y_ptr,
    n_rows, n_cols,
    x_row_stride,
    y_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + pid * x_row_stride + cols, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(y_ptr + pid * y_row_stride, result)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    y = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    row_sum_kernel[(n_rows,)](
        x, y, n_rows, n_cols,
        x.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


if __name__ == "__main__":
    n_rows, n_cols = 128, 4096
    x = torch.randn(n_rows, n_cols, device='cuda')

    y_triton = row_sum(x)
    y_torch = x.sum(dim=1)

    max_diff = (y_triton - y_torch).abs().max().item()
    print(f"Matrix: {n_rows} x {n_cols}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Passed: {torch.allclose(y_triton, y_torch, atol=1e-4)}")
