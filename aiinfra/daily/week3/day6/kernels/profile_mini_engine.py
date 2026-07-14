# profile_mini_engine.py —— Mini Transformer Engine 端到端 Profiling
# 运行命令: python profile_mini_engine.py
# nsys 采集: nsys profile -o mini_engine_timeline --trace=cuda,nvtx python profile_mini_engine.py
# 依赖: pip install torch

import torch
import torch.nn as nn
import math


class MiniAttention(nn.Module):
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
        attn = torch.softmax(attn, dim=-1)                   # softmax (memory-bound)
        out = torch.matmul(attn, v)                          # PV (cuBLAS)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)                                 # Out GEMM


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MiniAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # Attention + residual
        x = x + self.ffn(self.norm2(x))     # FFN + residual
        return x


def profile_phase(model, x, name, n_iter=5):
    for _ in range(2):  # warmup
        _ = model(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(n_iter):
            _ = model(x)
        torch.cuda.synchronize()

    print(f"\n===== {name} Phase (shape={tuple(x.shape)}) =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    prof.export_chrome_trace(f"trace_{name}.json")


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
    model = TransformerBlock(d_model, n_heads).cuda().half()

    # Prefill: N=1024
    x_prefill = torch.randn(1, 1024, d_model, device="cuda", dtype=torch.float16)
    profile_phase(model, x_prefill, "prefill", n_iter=5)

    # Decode: N=1
    x_decode = torch.randn(1, 1, d_model, device="cuda", dtype=torch.float16)
    profile_phase(model, x_decode, "decode", n_iter=10)

    print("\n===== Latency =====")
    with torch.no_grad():
        ms_pre = benchmark(model, x_prefill, "Prefill (N=1024)")
        ms_dec = benchmark(model, x_decode, "Decode  (N=1)")
    print(f"Per-token: Prefill={ms_pre/1024*1e3:.1f} us/token, Decode={ms_dec*1e3:.1f} us/token")

    print("\n===== 观察要点 =====")
    print("1. Prefill: aten::mm 占 CUDA 时间大头（compute-bound GEMM）")
    print("2. Decode:  aten::mm/layernorm/softmax 占比改变，总 SM 利用率低")
    print("3. 用 nsys profile 采集后，看 GPU kernel 时间线的 gap（launch overhead）")
    print("4. 用 ncu 分析 softmax_kernel / layernorm 的 DRAM%% vs SM%%（memory-bound）")


if __name__ == "__main__":
    main()
