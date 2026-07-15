# flash_attention.py —— Triton FlashAttention 简化版（forward）
# 运行: python3 kernels/flash_attention.py

import torch
import triton
import triton.language as tl
import time


@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    N_CTX,
    scale,
    stride_qb, stride_qd,
    stride_kb, stride_kd,
    stride_vb, stride_vd,
    stride_ob, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = q_ptr + offs_m[:, None] * stride_qb + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)
    q = q * scale

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k_ptrs = k_ptr + (start_n + offs_n)[:, None] * stride_kb + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + (start_n + offs_n)[:, None] * stride_vb + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        s = tl.dot(q, tl.trans(k))

        m_block = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    o = acc / l_i[:, None]
    o_ptrs = o_ptr + offs_m[:, None] * stride_ob + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    B, N, D = q.shape
    assert k.shape == (B, N, D) and v.shape == (B, N, D)
    assert D in [16, 32, 64, 128], f"D={D} not supported, use power of 2"

    o = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid = (triton.cdiv(N, BLOCK_M),)
    flash_attn_kernel[grid](
        q, k, v, o,
        N, scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o


def standard_attention(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return o


if __name__ == "__main__":
    B, N, D = 2, 512, 64
    q = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, N, D, device='cuda', dtype=torch.float16)

    o_flash = flash_attention(q, k, v)
    o_std = standard_attention(q, k, v)

    max_diff = (o_flash - o_std).abs().max().item()
    print(f"Shape: B={B}, N={N}, D={D}")
    print(f"Max diff: {max_diff:.4f}")
    print(f"Passed: {torch.allclose(o_flash, o_std, atol=0.05)}")

    for N in [512, 1024, 2048, 4096]:
        q = torch.randn(2, N, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(2, N, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(2, N, 64, device='cuda', dtype=torch.float16)

        for _ in range(5):
            flash_attention(q, k, v)
            standard_attention(q, k, v)
        torch.cuda.synchronize()

        n_iters = 50
        start = time.time()
        for _ in range(n_iters):
            flash_attention(q, k, v)
        torch.cuda.synchronize()
        flash_ms = (time.time() - start) / n_iters * 1000

        start = time.time()
        for _ in range(n_iters):
            standard_attention(q, k, v)
        torch.cuda.synchronize()
        std_ms = (time.time() - start) / n_iters * 1000

        print(f"  N={N:4d}: Flash={flash_ms:.3f}ms  Standard={std_ms:.3f}ms  Speedup={std_ms/flash_ms:.2f}x")
