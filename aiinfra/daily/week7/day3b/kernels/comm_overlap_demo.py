# comm_overlap_demo.py —— 通信-计算重叠 Demo（双 CUDA Stream）
# 运行命令: python comm_overlap_demo.py
# 依赖: torch + CUDA（单 GPU 即可）
# Profiling: nsys profile -o comm_overlap --trace=cuda,nvtx python comm_overlap_demo.py
"""
演示通信-计算重叠（comm-compute overlap）：
  - compute_stream: 执行 GEMM（模拟前向计算）
  - comm_stream   : 执行 all-reduce（模拟分布式通信）
  - 串行（不重叠）: total ≈ compute + comm
  - 重叠          : total ≈ max(compute, comm)
注意：单 GPU 上用 clone+sum 模拟 all-reduce；真实多卡用 torch.distributed.all_reduce
      重叠效果取决于 GPU 空闲 SM 是否足够容纳两路 kernel 并发
"""

import time
import torch


def dummy_gemm(stream, size=2048):
    """在指定 stream 上执行一次 GEMM，模拟前向计算"""
    with torch.cuda.stream(stream):
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        return torch.matmul(a, b)


def dummy_comm(stream, size=2048):
    """在指定 stream 上模拟 all-reduce：多次 add 近似 ring all-reduce 的多步通信"""
    with torch.cuda.stream(stream):
        t = torch.randn(size, size, device="cuda", dtype=torch.float16)
        for _ in range(4):
            t = t + t.clone()
        return t


def measure(fn, n_iters=10, warmup=3, wait_streams=()):
    """用 torch.cuda.Event 计时，返回平均毫秒"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    cur = torch.cuda.current_stream()
    for s in wait_streams:
        cur.wait_stream(s)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA 环境（单 GPU 即可）")
        return
    torch.cuda.init()
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()
    N, SIZE = 10, 2048
    print("=" * 64)
    print("  通信-计算重叠 Demo（双 CUDA Stream）")
    print("=" * 64)
    print(f"  iters={N}, size={SIZE}x{SIZE} FP16")

    compute_ms = measure(lambda: dummy_gemm(compute_stream, SIZE),
                         N, wait_streams=(compute_stream,))
    comm_ms = measure(lambda: dummy_comm(comm_stream, SIZE),
                      N, wait_streams=(comm_stream,))

    def serial_step():
        # 串行：comm 等 compute 完成（存在数据依赖）
        dummy_gemm(compute_stream, SIZE)
        comm_stream.wait_stream(compute_stream)
        dummy_comm(comm_stream, SIZE)

    def overlap_step():
        # 重叠：compute 与 comm 无数据依赖，分别 launch 到不同 stream 并发
        dummy_gemm(compute_stream, SIZE)
        dummy_comm(comm_stream, SIZE)

    serial_ms = measure(serial_step, N, wait_streams=(compute_stream, comm_stream))
    overlap_ms = measure(overlap_step, N, wait_streams=(compute_stream, comm_stream))

    print(f"\n  compute alone : {compute_ms:.2f} ms")
    print(f"  comm alone    : {comm_ms:.2f} ms")
    print(f"\n  [串行] total  : {serial_ms:.2f} ms  (compute + comm = {compute_ms + comm_ms:.2f})")
    print(f"  [重叠] total  : {overlap_ms:.2f} ms  (max = {max(compute_ms, comm_ms):.2f})")
    speedup = serial_ms / overlap_ms if overlap_ms > 0 else 0
    print(f"\n  加速比        : {speedup:.2f}x")
    print(f"  理论上限      : {serial_ms / max(compute_ms, comm_ms):.2f}x (完全重叠)")
    print("\n  nsys 可视化:")
    print("    nsys profile -o comm_overlap --trace=cuda python comm_overlap_demo.py")
    print("    nsys-ui comm_overlap.nsys-rep")
    print("    # timeline 中可见 compute_stream / comm_stream 的 kernel 交错或重叠")


if __name__ == "__main__":
    main()
