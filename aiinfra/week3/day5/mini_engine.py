# mini_engine.py —— Mini Transformer 引擎（自定义算子版 vs 全 PyTorch 版）
# 运行命令: python mini_engine.py
# 依赖: pip install torch
# 前置: kernels/softmax_layernorm_ext.cu 已存在

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

KERNEL_PATH = os.path.join(os.path.dirname(__file__), "kernels", "softmax_layernorm_ext.cu")
with open(KERNEL_PATH, "r") as f:
    cuda_src = f.read()

cpp_src = """
#include <torch/extension.h>
at::Tensor softmax_forward(at::Tensor input);
at::Tensor layernorm_forward(at::Tensor input, at::Tensor gamma, at::Tensor beta, double eps);
"""

my_ops = load_inline(
    name="my_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["softmax_forward", "layernorm_forward"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80", "-DWITH_TORCH"],
)


class MiniAttention(nn.Module):
    """用自定义 Softmax 的 Attention（GEMM 仍用 cuBLAS）"""

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
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # QK^T (cuBLAS)
        # ★ 自定义 Softmax 替换 torch.softmax
        B_, H, N_, _ = attn.shape
        attn_flat = attn.reshape(B_ * H * N_, N_).contiguous()
        attn_flat = my_ops.softmax_forward(attn_flat)
        attn = attn_flat.reshape(B_, H, N_, N_)
        out = torch.matmul(attn, v)  # PV (cuBLAS)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, use_custom_ln=True):
        super().__init__()
        self.attn = MiniAttention(d_model, n_heads)
        self.norm1_weight = nn.Parameter(torch.ones(d_model))
        self.norm1_bias = nn.Parameter(torch.zeros(d_model))
        self.norm2_weight = nn.Parameter(torch.ones(d_model))
        self.norm2_bias = nn.Parameter(torch.zeros(d_model))
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.use_custom_ln = use_custom_ln
        self.eps = 1e-5

    def _layernorm(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D).contiguous()
        if self.use_custom_ln:
            x_norm = my_ops.layernorm_forward(x_flat, self.norm1_weight, self.norm1_bias, self.eps)
        else:
            x_norm = torch.nn.functional.layer_norm(x_flat, (D,), self.norm1_weight, self.norm1_bias, self.eps)
        return x_norm.reshape(B, N, D)

    def forward(self, x):
        B, N, D = x.shape
        # ★ 自定义 LayerNorm 替换 F.layer_norm
        if self.use_custom_ln:
            x_flat = x.reshape(B * N, D).contiguous()
            x_norm = my_ops.layernorm_forward(x_flat, self.norm1_weight, self.norm1_bias, self.eps)
            x_norm = x_norm.reshape(B, N, D)
        else:
            x_norm = torch.nn.functional.layer_norm(x, (D,), self.norm1_weight, self.norm1_bias, self.eps)
        x = x + self.attn(x_norm)

        if self.use_custom_ln:
            x_flat = x.reshape(B * N, D).contiguous()
            x_norm = my_ops.layernorm_forward(x_flat, self.norm2_weight, self.norm2_bias, self.eps)
            x_norm = x_norm.reshape(B, N, D)
        else:
            x_norm = torch.nn.functional.layer_norm(x, (D,), self.norm2_weight, self.norm2_bias, self.eps)
        x = x + self.ffn(x_norm)
        return x


def benchmark(model, x, name, n_iter=20):
    """对比 latency"""
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
    x = torch.randn(1, 1024, d_model, device="cuda", dtype=torch.float32)

    model_pytorch = TransformerBlock(d_model, n_heads, use_custom_ln=False).cuda()
    model_custom = TransformerBlock(d_model, n_heads, use_custom_ln=True).cuda()
    model_custom.load_state_dict(model_pytorch.state_dict())

    with torch.no_grad():
        out_pytorch = model_pytorch(x)
        out_custom = model_custom(x)
    max_diff = (out_pytorch - out_custom).abs().max().item()
    print(f"Max diff (PyTorch vs Custom): {max_diff:.2e}")
    assert max_diff < 1e-4, "Correctness check failed"

    print("\n=== Latency Comparison (Prefill, N=1024) ===")
    with torch.no_grad():
        ms_pt = benchmark(model_pytorch, x, "PyTorch (F.softmax + F.layer_norm)")
        ms_my = benchmark(model_custom, x, "Custom (my_ops.softmax + my_ops.layernorm)")
    print(f"Speedup: {ms_pt / ms_my:.2f}x")

    print("\n=== 观察要点 ===")
    print("1. 自定义算子可能比 PyTorch 慢（0.8x），因为缺失向量化/warp 级优化")
    print("2. 这是正常的——本周目标是理解算子，不是超越官方实现")
    print("3. 用 nsys profile 看 kernel 数量：自定义版应多出 my_ops 的 kernel")
    print("4. torch.compile(model) 后再对比，看 fusion 能否弥补差距")


if __name__ == "__main__":
    main()
