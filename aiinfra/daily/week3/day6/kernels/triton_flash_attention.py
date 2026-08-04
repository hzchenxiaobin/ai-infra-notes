import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_ctx,
    scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    # bh index → (batch, head) using strides
    q_base = q_ptr + off_bh * stride_qh
    k_base = k_ptr + off_bh * stride_kh
    v_base = v_ptr + off_bh * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :]
    v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :]

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_ctx, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    for start_n in range(0, N_ctx, BLOCK_N):
        n_mask = (start_n + offs_n) < N_ctx
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, k.T.to(q.dtype)) * scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)

        l_new = alpha * l_i + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + off_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < N_ctx)


def triton_flash_attention(q, k, v, scale=None):
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    BLOCK_M = 64 if N >= 64 else 32
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    flash_attn_kernel[grid](
        q,
        k,
        v,
        o,
        N,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        D_HEAD=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return o


def naive_attention(q, k, v, scale=None):
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones_like(s), diagonal=0)
    s = s.masked_fill(mask == 0, -float("inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


@torch.no_grad()
def _benchmark(fn, *args, n_warmup=10, n_iter=50) -> float:
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
    print("=== Triton FlashAttention (causal) vs naive attention ===")
    cases = [(2, 4, 512, 64), (1, 8, 1024, 64), (1, 8, 2048, 64)]
    header = f"{'(B,H,N,D)':<22}{'naive(ms)':<14}{'triton(ms)':<14}{'max_diff':<14}{'speedup':<10}{'check':<6}"
    print(header)
    print("-" * len(header))
    for B, H, N, D in cases:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        y_naive = naive_attention(q, k, v)
        y_triton = triton_flash_attention(q, k, v)
        max_diff = (y_naive - y_triton).abs().max().item()
        t_naive = _benchmark(naive_attention, q, k, v)
        t_triton = _benchmark(triton_flash_attention, q, k, v)
        speedup = t_naive / t_triton
        ok = "PASS" if max_diff < 1e-2 else "FAIL"
        print(f"{str((B,H,N,D)):<22}{t_naive:<14.4f}{t_triton:<14.4f}{max_diff:<14.2e}{speedup:<10.2f}{ok:<6}")


if __name__ == "__main__":
    main()
