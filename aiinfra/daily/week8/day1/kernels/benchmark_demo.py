# benchmark_demo.py —— Kernel Benchmark 框架（SiLU 示例）
# 运行命令: python benchmark_demo.py --sizes 1m,4m,16m
# 依赖: torch (CUDA)
# 作用: 演示正确的 cudaEvent 计时流程，输出可直接贴进 README 的 Markdown 表格

import argparse


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit("ERROR: 需要 torch (CUDA)，请 pip install torch")


def benchmark_torch_silu(n, warmup=5, iters=100, torch=None):
    """用 torch SiLU 演示正确计时流程：warmup → cudaEvent → N 次平均 → 算带宽。"""
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    for _ in range(warmup):
        torch.nn.functional.silu(x, out=y)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.nn.functional.silu(x, out=y)
    stop.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(stop)
    t = ms / iters
    bytes_rw = 2 * n * 4
    bw_gbs = bytes_rw / (t / 1000) / 1e9
    return {"n": n, "time_ms": t, "bandwidth_gbs": bw_gbs}


def fmt_markdown_table(rows, peak_bw=1555):
    cols = ["规模 N", "单次耗时(ms)", "带宽(GB/s)", "带宽利用率"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        util = f"{r['bandwidth_gbs'] / peak_bw * 100:.1f}%"
        lines.append(f"| {r['n']:,} | {r['time_ms']:.4f} | "
                     f"{r['bandwidth_gbs']:.1f} | {util} |")
    return "\n".join(lines)


def parse_size(s):
    s = s.strip().lower()
    mult = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    return int(s[:-1]) * mult[s[-1]] if s[-1] in mult else int(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kernel benchmark (SiLU demo)")
    ap.add_argument("--sizes", type=str, default="1m,4m,16m",
                    help="规模，逗号分隔，支持 k/m/g 后缀")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    torch = _require_torch()
    assert torch.cuda.is_available(), "需要 CUDA 环境"

    sizes = [parse_size(s) for s in args.sizes.split(",")]
    results = []
    for n in sizes:
        r = benchmark_torch_silu(n, warmup=args.warmup,
                                 iters=args.iters, torch=torch)
        results.append(r)
        print(f"N={n:>12,}  time={r['time_ms']:.4f} ms  "
              f"BW={r['bandwidth_gbs']:.1f} GB/s")

    print("\n## README Benchmark 表（可直接粘贴）\n")
    print(fmt_markdown_table(results))
