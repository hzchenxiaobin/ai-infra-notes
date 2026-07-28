# gpt_model.py —— Day 4: Transformer Block + 完整 decoder-only GPT
# 运行：python3 gpt_model.py
# 仅依赖 PyTorch + Day 2/3 的实现，CPU 即可运行
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mha_and_positional import MultiHeadAttention


# ============================================================
# FFN：升维 4 倍再降回，逐位置非线性变换（参数大头 8d²）
# ============================================================
class FFN(nn.Module):
    """前馈网络：d -> 4d -> d，GELU 激活。参数量 8d²（attention 的 2 倍）。"""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.proj(self.gelu(self.fc(x)))


# ============================================================
# Block：Pre-LN 风格（现代主流），MHA + FFN 各带残差
# ============================================================
class Block(nn.Module):
    """Pre-LN Transformer Block。
    x = x + MHA(LN(x))   # 残差 1：注意力分支
    x = x + FFN(LN(x))   # 残差 2：FFN 分支
    """

    def __init__(self, d_model, n_heads, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, ffn_ratio * d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))           # Pre-LN 残差 1
        x = x + self.drop(self.ffn(self.ln2(x)))            # Pre-LN 残差 2
        return x


class PostLNBlock(nn.Module):
    """Post-LN（原论文）：LN 在残差之后，用于 Day 4 对比实验。"""

    def __init__(self, d_model, n_heads, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, ffn_ratio * d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x + self.drop(self.attn(x)))           # Post-LN：LN 在残差外
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x


# ============================================================
# GPT：完整 decoder-only 模型（token emb + pos emb + N×Block + LN + LM head）
# ============================================================
class GPT(nn.Module):
    """字符级 / 子词级 decoder-only GPT，Pre-LN + learned 位置编码 + 权重共享。"""

    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)         # token embedding
        self.wpe = nn.Embedding(block_size, d_model)         # learned 位置 embedding
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)                    # 最终 LN（Pre-LN 必备）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享：LM head 和 token embedding 用同一张权重表
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))         # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)                               # (B, T, V)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_params_non_embedding(self):
        """只统计 Transformer 层（不含 wte/wpe），用于对比。"""
        return sum(p.numel() for n, p in self.named_parameters()
                   if "wte" not in n and "wpe" not in n)


# ============================================================
# 实验
# ============================================================
def demo_ffn():
    """实验 1：FFN 升维 4 倍——参数量 8d² 与前向形状。"""
    print("=" * 64)
    print("实验 1：FFN 升维 4 倍（参数量 8d²，是 attention 的 2 倍）")
    print("=" * 64)
    d_model = 64
    ffn = FFN(d_model, d_ff=4 * d_model)
    n_params = sum(p.numel() for p in ffn.parameters())
    # fc: d×4d + 4d, proj: 4d×d + d → 忽略 bias 后 ≈ 8d²
    print(f"配置: d_model={d_model}, d_ff={4*d_model} (4倍升维)")
    print(f"FFN 参数量: {n_params} (含 bias)")
    print(f"  手算 8d² = {8 * d_model**2} (忽略 bias)")
    print(f"  attention 是 4d² = {4 * d_model**2} → FFN 是 attention 的 2 倍 (参数大头)")

    x = torch.randn(2, 8, d_model)
    out = ffn(x)
    print(f"前向: {tuple(x.shape)} → {tuple(out.shape)}  (形状不变，逐位置变换)")


def demo_preln_vs_postln():
    """实验 2：Pre-LN vs Post-LN——残差路径的恒等性。"""
    print("\n" + "=" * 64)
    print("实验 2：Pre-LN vs Post-LN（残差路径恒等性 → 深层训练稳定性）")
    print("=" * 64)
    d_model = 64
    torch.manual_seed(0)
    x = torch.randn(2, 8, d_model)
    ln = nn.LayerNorm(d_model)
    zero_sub = lambda t: torch.zeros_like(t)                # 零子层（模拟训练初期）

    # Pre-LN: x + Sublayer(LN(x))，Sublayer=0 → output = x
    out_pre = x + zero_sub(ln(x))
    # Post-LN: LN(x + Sublayer(x))，Sublayer=0 → output = LN(x) ≠ x
    out_post = ln(x + zero_sub(x))

    print("数据流对比（Sublayer=0，模拟训练初期子层输出极小）：")
    print(f"  Pre-LN : x + Sublayer(LN(x)) = x + 0     = x       (恒等)")
    print(f"  Post-LN: LN(x + Sublayer(x))  = LN(x+0)  = LN(x)   (非恒等)")
    print(f"\nPre-LN  output == input? {torch.allclose(out_pre, x)}")
    print(f"Post-LN output == input? {torch.allclose(out_post, x)}")
    print(f"  input  mean={x.mean():.4f} std={x.std():.4f}")
    print(f"  Pre-LN mean={out_pre.mean():.4f} std={out_pre.std():.4f}  (与 input 完全一致)")
    print(f"  Post-LN mean={out_post.mean():.4f} std={out_post.std():.4f}  (被 LN 归一化，偏离 input)")
    print("→ Pre-LN 残差路径是恒等映射，梯度直通浅层；Post-LN 的 LN 在残差路径上，"
          "深层梯度不稳定")


def demo_block_assembly():
    """实验 3：完整 Block 组装——形状保持 + 可堆叠。"""
    print("\n" + "=" * 64)
    print("实验 3：完整 Block 组装（MHA + FFN + 残差 + Pre-LN）")
    print("=" * 64)
    d_model, n_heads = 64, 4
    block = Block(d_model, n_heads)
    n_params = sum(p.numel() for p in block.parameters())
    # attn 4d² + ffn 8d² + 2×LN(2d) ≈ 12d²
    print(f"配置: d_model={d_model}, n_heads={n_heads}")
    print(f"Block 参数量: {n_params} (手算 12d² = {12 * d_model**2}, 含 bias/LN 略多)")

    x = torch.randn(2, 8, d_model)
    out = block(x)
    print(f"前向: {tuple(x.shape)} → {tuple(out.shape)}  (残差保证形状不变)")

    # 堆叠 4 层
    blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(4)])
    h = x
    for i, blk in enumerate(blocks):
        h = blk(h)
        print(f"  经过 layer {i}: shape={tuple(h.shape)}")
    print("→ Block 可任意堆叠，形状始终 (B, T, d_model)")


def demo_gpt_params_and_forward():
    """实验 4：完整 GPT——参数量核对（GPT-2 small 124M）+ 前向冒烟 + 权重共享。"""
    print("\n" + "=" * 64)
    print("实验 4：完整 GPT 模型（参数量核对 + 前向冒烟 + 权重共享）")
    print("=" * 64)

    # --- GPT-2 small 配置 ---
    cfg = dict(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, d_model=768)
    model = GPT(**cfg)
    total = model.num_params()
    non_emb = model.num_params_non_embedding()

    # 分项统计
    wte = model.wte.weight.numel()
    wpe = model.wpe.weight.numel()
    per_block = sum(p.numel() for p in model.blocks[0].parameters())
    blocks_total = per_block * cfg["n_layer"]
    ln_f = sum(p.numel() for p in model.ln_f.parameters())

    print("GPT-2 small 配置: L=12, d=768, H=12, V=50257, block_size=1024")
    print(f"\n参数量分项：")
    print(f"  token embedding (wte, V×d)     : {wte:>12,}  ({wte/1e6:.1f}M)")
    print(f"  position embedding (wpe, T×d)  : {wpe:>12,}  ({wpe/1e6:.1f}M)")
    print(f"  每层 Block (12d²+bias+LN)      : {per_block:>12,}  ({per_block/1e6:.2f}M)")
    print(f"  {cfg['n_layer']} 层 Block 合计             : {blocks_total:>12,}  ({blocks_total/1e6:.1f}M)")
    print(f"  最终 LN                        : {ln_f:>12,}")
    print(f"  LM head (权重共享, 0 额外)     : {0:>12,}")
    print(f"  ────────────────────────────────")
    print(f"  总计                           : {total:>12,}  ({total/1e6:.1f}M)")
    print(f"\n手算验证：12层×12d² + V×d + T×d ≈ {12*12*768**2 + 50257*768 + 1024*768:,} ≈ 124M ✓")
    print(f"  (Attention 每层 4d²={4*768**2:,}, FFN 每层 8d²={8*768**2:,}, FFN 是参数大头)")

    # 权重共享验证
    print(f"\n权重共享: wte.weight is lm_head.weight? "
          f"{model.wte.weight is model.lm_head.weight}")
    print(f"  → 省下 V×d = {wte:,} ({wte/1e6:.1f}M) 参数 (~{wte/total*100:.0f}%)")

    # 前向冒烟测试
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, cfg["vocab_size"], (2, 16))
        logits = model(idx)
    print(f"\n前向冒烟: idx {tuple(idx.shape)} → logits {tuple(logits.shape)}  "
          f"(B, T, V) ✓")

    # --- 本周 Day 5 训练配置（~1M 参数）---
    print("\n" + "-" * 64)
    cfg_small = dict(vocab_size=65, block_size=128, n_layer=4, n_head=4, d_model=128)
    model_small = GPT(**cfg_small)
    print(f"本周 Day 5 训练配置: {cfg_small}")
    print(f"  参数量: {model_small.num_params():,} ({model_small.num_params()/1e6:.2f}M)")
    print(f"  → Tiny Shakespeare 字符级，CPU ~1h 可训")


if __name__ == "__main__":
    demo_ffn()
    demo_preln_vs_postln()
    demo_block_assembly()
    demo_gpt_params_and_forward()
