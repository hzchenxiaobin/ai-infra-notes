# mini_engine_v0.py —— Mini 推理引擎 v0（单请求 + KV Cache + Prefill/Decode 循环）
# 运行命令: python mini_engine_v0.py
# 依赖: pip install torch
#
# 整合 Week5 Day1-4 所学：Prefill/Decode 两阶段、KV Cache、Transformer 前向、
# argmax 采样，构成一个最小可运行的 LLM 推理引擎。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ============================================================
# 模型定义（对应 vLLM 的 ModelRunner 执行的 transformer）
# ============================================================

class MiniTransformerLayer(nn.Module):
    """单层 Transformer Block：Pre-LN + Self-Attention + FFN，支持 KV Cache"""

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

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        x: (B, N, d_model)
        kv_cache: (k_cache, v_cache) 各 shape (B, H, L, d_head)，或 None
        返回: x, (new_k, new_v)
        """
        B, N, _ = x.shape

        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, N, d_head)

        if use_cache and kv_cache is not None:
            # Decode: 把新 K/V 拼到历史 cache 后面
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)   # (B, H, L+N, d_head)
            v = torch.cat([v_cache, v], dim=2)

        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, N, L+N)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)   # (B, H, N, d_head)

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        x = x + self.out(out)
        x = x + self.ffn(self.norm2(x))

        return x, (k, v)


class MiniLLM(nn.Module):
    """最小 LLM：embedding + n_layers 层 transformer + lm_head"""

    def __init__(self, vocab_size=1000, d_model=512, n_heads=8, d_ff=2048, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MiniTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, kv_cache=None, use_cache=False):
        """
        input_ids: (B, N)
        kv_cache: list of (k,v) per layer, 或 None
        返回: logits (B, N, vocab), new_kv_cache
        """
        x = self.embedding(input_ids)   # (B, N, d)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = layer(x, layer_cache, use_cache)
            new_kv_cache.append(layer_new_cache)

        logits = self.lm_head(x)   # (B, N, vocab)
        return logits, new_kv_cache


# ============================================================
# Tokenizer（最简：空格分词 + 动态 vocab）
# ============================================================

class MiniTokenizer:
    """最简 tokenizer：按空格切词，动态分配 token id"""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 1   # 0 留给 <unk>

    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.lower().split():
            if word not in self.word_to_id:
                if self.next_id >= self.vocab_size:
                    break
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
            tokens.append(self.word_to_id[word])
        return tokens

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_word.get(i, f"<unk_{i}>") for i in ids)


# ============================================================
# Mini 推理引擎 v0（整合 Prefill/Decode + KV Cache）
# ============================================================

class MiniEngineV0:
    """Mini 推理引擎 v0：单请求 + KV Cache + Prefill/Decode 循环"""

    def __init__(self, model: MiniLLM, tokenizer: MiniTokenizer, device="cuda"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """端到端生成：encode → prefill → decode loop → decode"""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)

        # ========== Prefill：一次性处理整段 prompt ==========
        # use_cache=True 让每层把 prompt 的 K/V 存入 kv_cache
        logits, kv_cache = self.model(input_ids, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B,1)
        generated_ids = [next_token.item()]

        # ========== Decode Loop：每步只输入 1 个 token，复用 KV Cache ==========
        for _ in range(max_new_tokens - 1):
            logits, kv_cache = self.model(next_token, kv_cache=kv_cache, use_cache=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def generate_no_cache(self, prompt: str, max_new_tokens: int = 20) -> List[int]:
        """对照版：不用 KV Cache，每步重算完整历史"""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        current_ids = input_ids.clone()
        generated = []
        for _ in range(max_new_tokens):
            logits, _ = self.model(current_ids, use_cache=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=1)  # 重新拼历史
        return generated


# ============================================================
# 主流程
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(42)
    vocab_size, d_model, n_heads, n_layers = 1000, 512, 8, 4
    model = MiniLLM(vocab_size, d_model, n_heads, n_layers=n_layers)
    tokenizer = MiniTokenizer(vocab_size)
    engine = MiniEngineV0(model, tokenizer, device)

    prompt = "hello world this is a test"
    print(f"Prompt: {prompt}")

    # ① 端到端生成（带 KV Cache）
    generated = engine.generate(prompt, max_new_tokens=10)
    print(f"Generated (with cache): {generated}")

    # ② 正确性验证：with cache vs without cache 输出一致
    print("\n=== KV Cache Correctness Check ===")
    gen_cache = engine.generate(prompt, max_new_tokens=5)
    gen_no_cache = engine.generate_no_cache(prompt, max_new_tokens=5)
    # 比较首 token（prefill 输出应一致）
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    with torch.no_grad():
        logits, _ = model(input_ids, use_cache=False)
    first_with_cache = engine.generate(prompt, max_new_tokens=1)
    print(f"  with cache first token:    {tokenizer.encode(first_with_cache)}")
    print(f"  without cache tokens:       {gen_no_cache}")

    # ③ 多轮对话 KV Cache 复用演示
    print("\n=== Multi-turn Cache Reuse Demo ===")
    engine2 = MiniEngineV0(MiniLLM(vocab_size, d_model, n_heads, n_layers=n_layers), MiniTokenizer(vocab_size), device)
    torch.manual_seed(42)
    # round 1
    r1 = engine2.generate("hello world", max_new_tokens=5)
    print(f"  Round 1: '{r1}'")
    # round 2（新 engine 复用模型权重，cache 是 per-generate 的——演示概念）
    r2 = engine2.generate("please explain flashattention", max_new_tokens=5)
    print(f"  Round 2: '{r2}'  (新 prompt，独立 cache)")

    # ④ KV Cache 内存占用
    print("\n=== KV Cache Memory ===")
    bytes_per_token = 2 * n_layers * n_heads * (d_model // n_heads) * 4  # fp32
    print(f"  config: layers={n_layers}, heads={n_heads}, d_head={d_model//n_heads}, fp32")
    print(f"  bytes per token: {bytes_per_token} ({bytes_per_token/1024:.1f} KB)")
    for L in [256, 1024, 4096]:
        print(f"  seq_len={L}: {bytes_per_token * L / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
