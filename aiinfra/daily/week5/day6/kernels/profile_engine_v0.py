# profile_engine_v0.py —— Mini 推理引擎 v0 端到端 Profiling
# 运行命令: python profile_engine_v0.py
# 依赖: pip install torch
#
# 测量三大指标：TTFT（Prefill）、TBT（Decode）、阶段 latency breakdown，
# 并扫描 prompt 长度观察 TTFT 的增长规律。

import sys
import os
import time
import statistics
import torch

# mini_engine_v0 在 day5/kernels，把它加到 path
_DAY5_KERNELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "day5", "kernels")
sys.path.insert(0, _DAY5_KERNELS)
from mini_engine_v0 import MiniLLM, MiniTokenizer, MiniEngineV0

_HAS_CUDA = torch.cuda.is_available()

def sync():
    """cuda.synchronize 的 CPU 安全包装"""
    if _HAS_CUDA:
        torch.cuda.synchronize()


def profile_engine(engine, prompt, max_new_tokens=20):
    """端到端 profiling：TTFT + TBT（含 forward/sampling/sync breakdown）"""
    input_ids = torch.tensor([engine.tokenizer.encode(prompt)], device=engine.device)

    # 预热（消除首次 launch / JIT 开销）
    for _ in range(3):
        _ = engine.model(input_ids, use_cache=False)
    sync()

    # ========== Prefill Profiling ==========
    sync()
    t_start = time.perf_counter()
    with torch.no_grad():
        logits, kv_cache = engine.model(input_ids, use_cache=True)
        first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    sync()
    ttft = (time.perf_counter() - t_start) * 1000   # ms

    print(f"=== Prefill Phase ===")
    print(f"  Prompt length: {input_ids.size(1)} tokens")
    print(f"  TTFT: {ttft:.3f} ms")
    print(f"  KV Cache shape per layer: {tuple(kv_cache[0][0].shape)}")

    # ========== Decode Profiling ==========
    decode_times = []
    breakdown = {"forward": [], "sampling": [], "sync": []}

    next_token = first_token
    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, kv_cache = engine.model(next_token, kv_cache=kv_cache, use_cache=True)
        t1 = time.perf_counter()
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        t2 = time.perf_counter()
        sync()
        t3 = time.perf_counter()

        decode_times.append((t3 - t0) * 1000)
        breakdown["forward"].append((t1 - t0) * 1000)
        breakdown["sampling"].append((t2 - t1) * 1000)
        breakdown["sync"].append((t3 - t2) * 1000)

    mean_tbt = statistics.mean(decode_times)
    p50 = statistics.median(decode_times)
    p99 = sorted(decode_times)[int(len(decode_times) * 0.99)] if len(decode_times) > 1 else decode_times[0]

    print(f"\n=== Decode Phase ({max_new_tokens} tokens) ===")
    print(f"  Mean TBT: {mean_tbt:.3f} ms")
    print(f"  P50 TBT:  {p50:.3f} ms")
    print(f"  P99 TBT:  {p99:.3f} ms")
    print(f"  Max TBT:  {max(decode_times):.3f} ms")
    print(f"  Min TBT:  {min(decode_times):.3f} ms")

    print(f"\n=== Decode Breakdown (mean ms, % of TBT) ===")
    total = mean_tbt
    for key, times in breakdown.items():
        m = statistics.mean(times)
        print(f"  {key:12s}: {m:.3f} ms ({m/total*100:.1f}%)")

    print(f"\n=== Throughput ===")
    total_time = ttft + sum(decode_times)
    print(f"  Total time: {total_time:.3f} ms")
    print(f"  Output throughput: {max_new_tokens / (sum(decode_times)/1000):.2f} tokens/s")
    print(f"  TTFT/TBT ratio: {ttft/mean_tbt:.1f}x  (Prefill 比 Decode 单步重多少倍)")

    return {"ttft": ttft, "decode_times": decode_times, "breakdown": breakdown}


def scan_prompt_lengths(engine, lengths=None):
    """扫描 prompt 长度，观察 TTFT 随 N 的增长规律"""
    if lengths is None:
        lengths = [32, 64, 128, 256, 512]
    print("\n=== TTFT vs Prompt Length ===")
    print(f"{'N (tokens)':>12} {'TTFT (ms)':>12} {'Per-token (ms)':>16} {'TTFT/N²':>12}")
    print("-" * 56)

    for L in lengths:
        ids = torch.randint(1, 1000, (1, L), device=engine.device)
        # 预热
        for _ in range(2):
            _ = engine.model(ids, use_cache=False)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = engine.model(ids, use_cache=True)
        sync()
        ttft = (time.perf_counter() - t0) * 1000
        print(f"{L:>12} {ttft:>12.3f} {ttft/L:>16.3f} {ttft/(L*L):>12.6f}")


def scan_decode_length(engine, prompt_len=64, max_new_tokens_list=None):
    """扫描生成长度，观察 TBT 是否随序列长度增长"""
    if max_new_tokens_list is None:
        max_new_tokens_list = [10, 30, 50, 100]
    print("\n=== TBT vs Generated Length (prompt_len fixed) ===")
    print(f"{'Gen tokens':>12} {'Mean TBT (ms)':>16} {'Last TBT (ms)':>16}")
    print("-" * 48)

    for K in max_new_tokens_list:
        ids = torch.randint(1, 1000, (1, prompt_len), device=engine.device)
        with torch.no_grad():
            logits, kv_cache = engine.model(ids, use_cache=True)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        sync()
        decode_times = []
        for _ in range(K):
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, kv_cache = engine.model(next_tok, kv_cache=kv_cache, use_cache=True)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            sync()
            decode_times.append((time.perf_counter() - t0) * 1000)
        print(f"{K:>12} {statistics.mean(decode_times):>16.3f} {decode_times[-1]:>16.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    torch.manual_seed(42)
    model = MiniLLM(vocab_size=1000, d_model=512, n_heads=8, n_layers=4)
    tokenizer = MiniTokenizer(vocab_size=1000)
    engine = MiniEngineV0(model, tokenizer, device)

    prompt = "hello world this is a test prompt for profiling the engine"
    profile_engine(engine, prompt, max_new_tokens=20)

    scan_prompt_lengths(engine, lengths=[32, 64, 128, 256, 512])

    scan_decode_length(engine, prompt_len=64, max_new_tokens_list=[10, 30, 50, 100])


if __name__ == "__main__":
    main()
