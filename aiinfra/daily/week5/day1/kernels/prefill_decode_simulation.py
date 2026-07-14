# prefill_decode_simulation.py —— 模拟 Transformer 推理的 Prefill/Decode 两阶段
# 运行命令: python prefill_decode_simulation.py
# 依赖: pip install torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class MiniTransformer(nn.Module):
    """最小 Transformer Block，用于演示 Prefill/Decode"""

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, use_cache=False, k_cache=None, v_cache=None):
        """
        x: (B, N, d_model)
        use_cache: 是否使用 KV Cache
        k_cache/v_cache: 历史 KV，shape (B, H, L, d_head)
        返回: output, (new_k_cache, new_v_cache)
        """
        B, N, _ = x.shape

        # LayerNorm + QKV
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.d_head ** -0.5
        if use_cache and k_cache is not None:
            # Decode: 把新 K/V 拼到历史 cache 后面
            k = torch.cat([k_cache, k], dim=2)  # (B, H, L+1, d)
            v = torch.cat([v_cache, v], dim=2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        x = x + self.out(out)

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x, (k, v)


def simulate_inference(model, prompt, max_new_tokens=20):
    """模拟完整推理流程：Prefill + Decode"""
    device = next(model.parameters()).device
    B, N = prompt.size(0), prompt.size(1)

    # ========== Prefill 阶段 ==========
    torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        logits, (k_cache, v_cache) = model(prompt, use_cache=False)
        first_token_logits = logits[:, -1, :]  # 取最后一个位置的 logits

    torch.cuda.synchronize()
    ttft = (time.time() - t_start) * 1000  # ms

    print(f"=== Prefill Phase ===")
    print(f"  Input shape: {tuple(prompt.shape)}")
    print(f"  TTFT: {ttft:.3f} ms")
    print(f"  KV Cache shape: {tuple(k_cache.shape)}")

    # ========== Decode 阶段 ==========
    generated = []
    decode_times = []

    # 简化：用 argmax 采样；decode 的输入用随机向量模拟新生成 token 的 embedding
    next_token = first_token_logits.argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())

    for step in range(max_new_tokens - 1):
        next_token_emb = model.qkv.weight.new_zeros(B, 1, model.d_model).normal_(0, 0.02)

        torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            logits, (k_cache, v_cache) = model(
                next_token_emb, use_cache=True, k_cache=k_cache, v_cache=v_cache
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        torch.cuda.synchronize()
        decode_times.append((time.time() - t_start) * 1000)
        generated.append(next_token.item())

    print(f"\n=== Decode Phase ===")
    print(f"  Generated {len(generated)} tokens")
    print(f"  Mean TBT: {sum(decode_times)/len(decode_times):.3f} ms")
    print(f"  Max TBT: {max(decode_times):.3f} ms")
    print(f"  Min TBT: {min(decode_times):.3f} ms")
    print(f"  Generated token IDs: {generated}")

    return ttft, decode_times


def profile_phase(model, x, name, n_iter=10):
    """Profile 一个阶段"""
    for _ in range(3):
        _ = model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = model(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iter
    print(f"{name}: {ms:.3f} ms")
    return ms


def main():
    torch.manual_seed(42)
    device = "cuda"
    d_model, n_heads = 512, 8
    model = MiniTransformer(d_model, n_heads).to(device).eval().half()

    # Prefill: 处理长 prompt
    N = 1024
    prompt = torch.randn(1, N, d_model, device=device, dtype=torch.float16)

    print(f"Model: d_model={d_model}, n_heads={n_heads}")
    print(f"Prompt length: {N}\n")

    simulate_inference(model, prompt, max_new_tokens=10)

    # 单独 profile prefill vs decode
    print("\n=== Standalone Profiling ===")
    profile_phase(model, prompt, f"Prefill (N={N})")

    decode_input = torch.randn(1, 1, d_model, device=device, dtype=torch.float16)
    profile_phase(model, decode_input, f"Decode single token")


if __name__ == "__main__":
    main()
