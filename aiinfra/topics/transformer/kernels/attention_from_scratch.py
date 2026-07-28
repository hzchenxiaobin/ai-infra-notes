# attention_from_scratch.py —— 从零手写 Scaled Dot-Product Attention（Day 2 最小示例）
# 运行：python3 attention_from_scratch.py
# 仅依赖 PyTorch，CPU 即可运行
import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, causal=False):
    """手写缩放点积注意力。

    q, k, v: (B, H, T, d) 四维张量
    causal : 是否加因果掩码（decoder 自回归用）
    返回:    (B, H, T, d) 注意力输出
    """
    d = q.size(-1)
    # 1) QK^T 得到原始相关性分数，形状 (B, H, T, T)
    scores = q @ k.transpose(-2, -1)
    # 2) 除以 sqrt(d)，防止 d 较大时点积数值过大、softmax 饱和导致梯度消失
    scores = scores / math.sqrt(d)
    # 3) 因果掩码：位置 i 不允许看 j > i 的未来信息，置为 -inf 使 softmax 后权重为 0
    if causal:
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    # 4) softmax 归一化为注意力权重，再加权求和 V
    attn = F.softmax(scores, dim=-1)
    return attn @ v


class MultiHeadAttention(torch.nn.Module):
    """多头注意力：把 d_model 切成 H 个头，各自做注意力后拼接投影。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.w_out = torch.nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        B, T, C = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        # (B, T, C) -> (B, H, T, d)
        split = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = scaled_dot_product_attention(split(q), split(k), split(v), causal)
        # (B, H, T, d) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_out(out)


if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, T, d = 2, 4, 8, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    # 与 PyTorch 官方实现对齐：误差应在 1e-6 量级
    mine = scaled_dot_product_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"max abs diff vs F.scaled_dot_product_attention: {(mine - ref).abs().max():.2e}")

    # 因果性检查：改动未来位置的 K/V 不应影响当前输出
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 5:], v2[:, :, 5:] = 100.0, -100.0
    out2 = scaled_dot_product_attention(q, k2, v2, causal=True)
    print(f"causal check (first 5 positions unchanged): {torch.allclose(mine[:, :, :5], out2[:, :, :5])}")

    # 多头注意力模块跑通
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    print(f"MHA output shape: {mha(x).shape}")
