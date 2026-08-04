# mini_engine_fa.py —— Mini Transformer 引擎（FlashAttention 版）
# 运行命令: python mini_engine_fa.py
# 依赖: 需要 flash_attention_v2.cu 和 flash_attention_ops.cpp

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

cuda_src = open(os.path.join(os.path.dirname(__file__), "..", "..", "day2", "kernels", "flash_attention_v2.cu")).read()
cpp_src = """
#include <torch/extension.h>
at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V);
"""

fa_ops = load_inline(
    name="fa_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["flash_attention_forward"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_120", "-DWITH_TORCH"],
)

class MiniAttentionFA(nn.Module):
    """用自定义 FlashAttention 替换标准 Attention"""
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = fa_ops.flash_attention_forward(q.contiguous(), k.contiguous(), v.contiguous())

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)

class MiniAttentionStd(nn.Module):
    """标准 Attention（PyTorch 实现）"""
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, use_fa=True):
        super().__init__()
        attn_cls = MiniAttentionFA if use_fa else MiniAttentionStd
        self.attn = attn_cls(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def benchmark(model, x, name, n_iter=20):
    for _ in range(3):
        _ = model(x)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = model(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iter
    print(f"{name}: {ms:.3f} ms / forward")
    return ms

def main():
    torch.manual_seed(42)
    d_model, n_heads = 512, 8

    for N in [512, 1024, 2048]:
        print(f"\n===== N={N} =====")
        x = torch.randn(1, N, d_model, device="cuda", dtype=torch.float32)

        model_std = TransformerBlock(d_model, n_heads, use_fa=False).cuda()
        model_fa = TransformerBlock(d_model, n_heads, use_fa=True).cuda()
        model_fa.load_state_dict(model_std.state_dict())

        with torch.no_grad():
            out_std = model_std(x)
            out_fa = model_fa(x)
            max_diff = (out_std - out_fa).abs().max().item()
            print(f"Max diff (Std vs FlashAttention): {max_diff:.2e}")

        with torch.no_grad():
            ms_std = benchmark(model_std, x, f"Standard Attention (N={N})")
            ms_fa = benchmark(model_fa, x, f"FlashAttention (N={N})")
            print(f"Speedup: {ms_std / ms_fa:.2f}x")

if __name__ == "__main__":
    main()
