import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    x_stride,
    y_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x_row_ptr = x_ptr + row * x_stride
    x = tl.load(x_row_ptr + cols, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x_max = tl.where(x_max == -float("inf"), 0.0, x_max)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_sum = tl.where(x_sum == 0.0, 1.0, x_sum)
    y = x_exp / x_sum
    tl.store(y_ptr + row * y_stride + cols, y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "triton_softmax expects a 2D tensor"
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    grid = (n_rows,)
    softmax_kernel[grid](
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


@torch.no_grad()
def _benchmark(fn, *args, n_warmup=10, n_iter=100) -> float:
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def main():
    torch.manual_seed(42)
    shapes = [(128, 256), (256, 1024), (1024, 1024), (4096, 4096)]
    print("=== Triton Softmax vs torch.softmax ===")
    header = f"{'shape':<18}{'torch(ms)':<14}{'triton(ms)':<14}{'max_diff':<14}{'speedup':<10}{'check':<6}"
    print(header)
    print("-" * len(header))
    for M, N in shapes:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        y_torch = torch.softmax(x, dim=-1)
        y_triton = triton_softmax(x)
        max_diff = (y_torch - y_triton).abs().max().item()
        t_torch = _benchmark(lambda inp: torch.softmax(inp, dim=-1), x)
        t_triton = _benchmark(triton_softmax, x)
        speedup = t_torch / t_triton
        ok = "PASS" if max_diff < 1e-5 else "FAIL"
        print(f"{str((M, N)):<18}{t_torch:<14.4f}{t_triton:<14.4f}{max_diff:<14.2e}{speedup:<10.2f}{ok:<6}")


if __name__ == "__main__":
    main()
