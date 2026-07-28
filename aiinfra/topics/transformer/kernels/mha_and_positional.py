# mha_and_positional.py —— Day 3: Multi-Head Attention + 位置编码（Sinusoidal / RoPE）
# 运行：python3 mha_and_positional.py
# 仅依赖 PyTorch，CPU 即可运行
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_from_scratch import scaled_dot_product_attention


# ============================================================
# Multi-Head Attention：d_model 切成 H 个头，各自注意力后拼接投影
# ============================================================
class MultiHeadAttention(nn.Module):
    """多头注意力（带可选位置编码注入）。

    参数：
        d_model: 模型维度
        n_heads: 头数（d_model 必须能被 n_heads 整除）
    形状：
        输入  x: (B, T, d_model)
        输出    : (B, T, d_model)
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} 不能被 n_heads={n_heads} 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Q/K/V 合并成一个投影（3*d_model），输出投影 d_model -> d_model
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rope=None, causal=True):
        B, T, C = x.shape
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)            # 各 (B, T, C)
        # (B, T, C) -> (B, H, T, head_dim)
        split = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        if rope is not None:                                # 在 head_dim 上应用 RoPE
            q, k = rope(q), rope(k)
        out = scaled_dot_product_attention(q, k, v, causal=causal)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C)
        return self.w_out(out)


# ============================================================
# Sinusoidal 位置编码（原论文，固定、零参数、可外推）
# ============================================================
def sinusoidal_positional_encoding(max_len, d_model):
    """返回 (max_len, d_model) 的 sinusoidal 位置编码。"""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)                      # 偶数维 sin
    pe[:, 1::2] = torch.cos(pos * div)                      # 奇数维 cos
    return pe


# ============================================================
# RoPE：按位置旋转 Q/K 的二维子空间，内积只依赖相对距离 m-n
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码。对 (B, H, T, head_dim) 的 Q/K 旋转前 head_dim 个维度。"""

    def __init__(self, head_dim, max_len=2048, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)                    # (max_len, head_dim/2)
        cos = freqs.cos()[None, None, :, :]                 # (1,1,max_len,head_dim/2)
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def _rotate_half(self, x):
        """把 x 的后半维取负、前半交换，用于旋转。x: (..., d) -> (..., d)"""
        d = x.size(-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """x: (B, H, T, head_dim) -> 旋转后的 (B, H, T, head_dim)。"""
        T = x.size(-2)
        cos = self.cos_cached[:, :, :T, :]                  # (1,1,T,head_dim/2)
        sin = self.sin_cached[:, :, :T, :]
        # cos/sin 在 head_dim 维扩展成完整维度（每个 (i, i+d/2) 对共享一个频率）
        cos_full = torch.cat([cos, cos], dim=-1)             # (1,1,T,head_dim)
        sin_full = torch.cat([sin, sin], dim=-1)
        return x * cos_full + self._rotate_half(x) * sin_full


# ============================================================
# 实验
# ============================================================
def demo_mha_params_and_alignment():
    """实验 1：MHA 参数量核对 + 与逐头循环实现对齐。"""
    print("=" * 64)
    print("实验 1：MHA 参数量核对 + 切分/拼接等价性")
    print("=" * 64)
    d_model, n_heads = 64, 4
    mha = MultiHeadAttention(d_model, n_heads)
    head_dim = d_model // n_heads

    # 参数量：w_qkv (d * 3d) + w_out (d * d) = 4d²
    n_params = sum(p.numel() for p in mha.parameters())
    print(f"配置: d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")
    print(f"参数量: {n_params} = 4*d² = 4*{d_model}² = {4 * d_model ** 2}")

    # 切分/拼接 vs 逐头循环（验证等价）
    torch.manual_seed(0)
    B, T = 2, 8
    x = torch.randn(B, T, d_model)
    out_fast = mha(x, causal=True)                          # 切分/拼接版

    # 逐头循环版：手动取每个头的 q/k/v
    qkv = mha.w_qkv(x).chunk(3, dim=-1)
    q_all = qkv[0].view(B, T, n_heads, head_dim).transpose(1, 2)
    k_all = qkv[1].view(B, T, n_heads, head_dim).transpose(1, 2)
    v_all = qkv[2].view(B, T, n_heads, head_dim).transpose(1, 2)
    outs = []
    for h in range(n_heads):
        o = scaled_dot_product_attention(
            q_all[:, h:h+1], k_all[:, h:h+1], v_all[:, h:h+1], causal=True
        )
        outs.append(o)
    out_slow = torch.cat(outs, dim=1).transpose(1, 2).contiguous().view(B, T, d_model)
    out_slow = mha.w_out(out_slow)
    print(f"切分/拼接 vs 逐头循环 max diff: {(out_fast - out_slow).abs().max():.2e}  (应≈0)")
    print(f"MHA 输出形状: {tuple(out_fast.shape)}")


def demo_permutation_equivariance():
    """实验 2：无位置编码时注意力是置换等变的——打乱输入，输出同样打乱。"""
    print("\n" + "=" * 64)
    print("实验 2：置换等变性（证明注意力对顺序无感知 → 需要位置编码）")
    print("=" * 64)
    torch.manual_seed(0)
    d_model, n_heads = 32, 4
    mha = MultiHeadAttention(d_model, n_heads)
    mha.eval()
    B, T = 1, 6
    x = torch.randn(B, T, d_model)

    with torch.no_grad():
        out_orig = mha(x, causal=False)                     # 双向（不掩码）才能看清

        perm = torch.randperm(T)
        x_perm = x[:, perm, :]                              # 打乱输入顺序
        out_perm = mha(x_perm, causal=False)

    # 置换等变：out_perm 应等于 out_orig 按同样 perm 打乱
    diff = (out_perm - out_orig[:, perm, :]).abs().max().item()
    print(f"原始输出 (前3维, 6个位置):\n{out_orig[0, :, :3].round(decimals=3)}")
    print(f"打乱输入的输出 (前3维):\n{out_perm[0, :, :3].round(decimals=3)}")
    print(f"打乱输入输出 == 原输出按同序打乱? max diff = {diff:.2e}")
    print("→ 注意力是置换等变的：它本身完全不知道 token 的先后顺序")


def demo_sinusoidal():
    """实验 3：sinusoidal 位置编码——多频率波形 + 相对位置线性性。"""
    print("\n" + "=" * 64)
    print("实验 3：Sinusoidal 位置编码（多频率波形 + 相对位置可线性表示）")
    print("=" * 64)
    d_model, max_len = 16, 40
    pe = sinusoidal_positional_encoding(max_len, d_model)

    # 不同维度对（频率）的波形：低维高频、高维低频
    print("各维度对的正弦频率（θ_i = 1/10000^(2i/d)）：")
    for i in [0, 1, 2, 3, 5, 7]:
        theta = 1.0 / (10000 ** (2 * i / d_model))
        period = 2 * math.pi / theta
        print(f"  维度对 {2*i}/{2*i+1}: θ={theta:.4e}, 周期≈{period:.1f} 步")

    # 前 6 个位置、4 个维度
    print(f"\n前 6 个位置的 PE（取前 4 维，sin/cos 交替）：")
    print("       sin(2i=0)  cos(2i=1)  sin(2i=2)  cos(2i=3)")
    for pos in range(6):
        vals = [f"{pe[pos, j]:>9.4f}" for j in range(4)]
        print(f"pos={pos}: " + "  ".join(vals))

    # 相对位置线性性：PE[pos+k] 是 PE[pos] 的固定线性变换
    # 对每个 (m, n) 对，验证 PE[m]·PE[n] 只依赖 m-n（内积视角的相对性）
    print("\n内积 PE[m]·PE[n] 只依赖相对距离 m-n（行=相对距离 k=m-n）：")
    for k in [0, 1, 3, 5]:
        dots = []
        for m in range(k, min(k + 4, max_len)):
            dots.append((pe[m] @ pe[m - k]).item())
        print(f"  k={k}: " + ", ".join(f"{d:.3f}" for d in dots) +
              f"  → 同一 k 下内积近似恒定")


def demo_rope_relative_position():
    """实验 4：RoPE 使 Q·K 只依赖相对距离 m-n，与绝对位置无关。"""
    print("\n" + "=" * 64)
    print("实验 4：RoPE——Q·K 点积只依赖相对距离（m-n）")
    print("=" * 64)
    head_dim, max_len = 16, 20
    rope = RotaryPositionalEmbedding(head_dim, max_len)
    torch.manual_seed(0)
    # 固定一对 Q/K 向量内容，只变位置
    q0 = torch.randn(1, 1, 1, head_dim)                     # (B,H,1,d) 一个 q
    k0 = torch.randn(1, 1, 1, head_dim)                     # 一个 k

    # 计算 <RoPE(q, m), RoPE(k, n)>，看是否只依赖 m-n
    print("固定 q/k 内容，变化绝对位置 m,n，看点积 <RoPE(q,m)·RoPE(k,n)>：")
    print(f"{'m':>4} {'n':>4} {'m-n':>6} {'点积':>10}")
    results = {}
    for m in [0, 5, 10, 15]:
        for n in [0, 2, 5, 10]:
            if n > m:
                continue
            # 先对长度 m+1 / n+1 的序列应用 RoPE，再取末尾位置 m / n
            qm = rope(q0.expand(1, 1, m + 1, head_dim))[:, :, -1:, :]
            kn = rope(k0.expand(1, 1, n + 1, head_dim))[:, :, -1:, :]
            dot = (qm * kn).sum().item()
            results.setdefault(m - n, []).append(dot)
            print(f"{m:>4} {n:>4} {m-n:>6} {dot:>10.4f}")

    print("\n同一相对距离 k=m-n 的点积是否恒定：")
    for k, dots in sorted(results.items()):
        spread = max(dots) - min(dots)
        flag = "✓ 恒定" if spread < 1e-5 else f"✗ 差异 {spread:.2e}"
        print(f"  k={k}: 点积={[round(d,4) for d in dots]} {flag}")
    print("→ RoPE 让点积只依赖相对距离，这就是它'长度外推'好的根源")


if __name__ == "__main__":
    demo_mha_params_and_alignment()
    demo_permutation_equivariance()
    demo_sinusoidal()
    demo_rope_relative_position()
