# trace_transformer.py —— 最小 Transformer Block + Prefill/Decode profiling
# 运行命令: python trace_transformer.py
# 依赖: pip install torch
#
# 对应 Week 3 Day 1 晚间编程任务 + 练习题 2（nsys）/ 练习题 3（torch.compile）

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
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MiniAttention(d_model, n_heads)
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


def profile_phase(model, x, name, n_iter=5):
    """对一个阶段做 profiling 并输出 top 算子"""
    for _ in range(2):
        _ = model(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(n_iter):
            _ = model(x)
        torch.cuda.synchronize()

    print(f"\n===== {name} Phase (shape={tuple(x.shape)}) =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    prof.export_chrome_trace(f"trace_{name}.json")


def benchmark(model, x, name, n_iter=20):
    """纯计时对比（不采集 profiler，更轻量）"""
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

    # ===== 1. torch.profiler profiling =====
    print("=" * 60)
    print("1. torch.profiler 分析")
    print("=" * 60)

    x_prefill = torch.randn(1, 1024, d_model, device="cuda", dtype=torch.float16)
    profile_phase(model, x_prefill, "prefill", n_iter=5)

    x_decode = torch.randn(1, 1, d_model, device="cuda", dtype=torch.float16)
    profile_phase(model, x_decode, "decode", n_iter=10)

    # ===== 2. 纯计时对比 Prefill vs Decode =====
    print("\n" + "=" * 60)
    print("2. Latency 对比（Prefill vs Decode）")
    print("=" * 60)
    with torch.no_grad():
        ms_pre = benchmark(model, x_prefill, "Prefill (N=1024)")
        ms_dec = benchmark(model, x_decode, "Decode  (N=1)")
    print(f"\nPrefill 单 token: {ms_pre / 1024:.4f} ms")
    print(f"Decode  单 token: {ms_dec:.4f} ms")
    print(f"Decode/Prefill per-token ratio: {ms_dec / (ms_pre / 1024):.1f}x")

    # ===== 3. torch.compile 对比（练习 3）=====
    print("\n" + "=" * 60)
    print("3. torch.compile 对比（kernel fusion）")
    print("=" * 60)
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            ms_compiled = benchmark(compiled_model, x_prefill, "Prefill compiled")
        print(f"Speedup: {ms_pre / ms_compiled:.2f}x")
    except Exception as e:
        print(f"torch.compile 不可用或失败: {e}")

    # ===== 4. nsys/ncu 提示 =====
    print("\n" + "=" * 60)
    print("4. nsys / ncu 分析命令")
    print("=" * 60)
    print("""
# nsys 采集系统级时间线（练习 2）
nsys profile -o transformer_trace python trace_transformer.py

# nsys 查看 kernel 统计
nsys stats -t cuda_gpu_kern_sum transformer_trace.nsys-rep

# ncu 分析特定 kernel（需先 nsys 找到 kernel 名）
ncu --kernel-name regex:"gemm|softmax|layer_norm|gelu" \\
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
sm__occupancy.avg.pct_of_peak_sustained_elapsed \\
    python trace_transformer.py

# 观察要点：
# 1. Prefill: GEMM kernel 的 sm__throughput 高 → compute-bound
# 2. Decode:  GEMM kernel 的 dram__throughput 高, sm__throughput 低 → memory-bound
# 3. 对比 softmax/layernorm 在两阶段的绝对时间
# 4. nsys 时间线中 Decode 的 kernel 间隙更大（launch overhead）
""")


if __name__ == "__main__":
    main()
