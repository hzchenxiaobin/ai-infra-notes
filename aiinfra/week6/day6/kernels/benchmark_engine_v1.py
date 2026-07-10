# benchmark_engine_v1.py —— Mini 引擎 v1 Latency/Throughput Benchmark
# 运行命令: python benchmark_engine_v1.py
# 依赖: 仅标准库（无 GPU/torch，用模拟引擎复刻 v1 的 Continuous Batching 调度行为）
#
# 本文件对 Week6 Day5 的 MiniEngineV1 做性能测试，核心三件事：
#   1. 固定并发数扫描（concurrency=1,2,4,8,16,32,64）→ 记录 throughput / avg / p99 latency
#   2. 固定 QPS 测试（恒定速率发请求）→ 观察 P50/P99 随负载变化
#   3. 绘制 throughput-latency 曲线，识别饱和点
#
# 为保证无 GPU 也能跑出有意义的曲线，用 SimulatedEngine 复刻 v1 的调度行为：
#   - forward 时间 = base + per_seq × batch × amort^(batch-1)（batch 越大 per-token 越省）
#   - max_num_seqs 限制每轮 batch 上限 → 超过后请求排队，throughput 封顶、latency 线性涨
#   - 该模型产生经典 throughput-latency 曲线：线性增长 → 饱和拐点 → 排队区

import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Dict, List


# ============================================================
# 模拟 Mini 引擎 v1（复刻 Continuous Batching 调度，无 torch 依赖）
# ============================================================

class SimRequest:
    """模拟 Request：记录 submit_time / finish_time 用于算 latency。"""
    _next_id = 0

    def __init__(self, prompt_len: int, max_new_tokens: int, submit_time: float):
        SimRequest._next_id += 1
        self.req_id = SimRequest._next_id
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.generated = 0
        self.status = "waiting"
        self.submit_time = submit_time
        self.start_time = -1.0
        self.finish_time = -1.0
        self.future = Future()

    @property
    def is_finished(self):
        return self.generated >= self.max_new_tokens


class SimulatedEngine:
    """模拟 MiniEngineV1：Continuous Batching + max_num_seqs + 摊销算力模型。

    forward 时间模型（模拟真实 GPU batch 摊销行为）：
      iter_time = base + per_seq × batch × amort^(batch-1)
    - base: 固定开销（kernel launch、调度），如 5ms
    - per_seq: 每请求边际成本，如 2ms
    - amort: batch 摊销系数（0-1），batch 越大每请求越省（接近算力上限）
    - max_num_seqs: 每轮 batch 上限，超过此值的并发请求排队等下一波

    这个模型产生经典的 throughput-latency 曲线：
      concurrency ≤ max_num_seqs：throughput 随并发线性增长，latency 平稳
      concurrency > max_num_seqs：throughput 封顶（=max_num_seqs/iter_time），latency 因排队线性涨
    """

    def __init__(self, max_num_seqs: int = 8, max_token_budget: int = 256,
                 base_iter_ms: float = 5.0, per_seq_ms: float = 2.0,
                 amort: float = 0.85, max_new_tokens_default: int = 8):
        self.max_num_seqs = max_num_seqs
        self.max_token_budget = max_token_budget
        self.base_iter = base_iter_ms / 1000.0
        self.per_seq = per_seq_ms / 1000.0
        self.amort = amort
        self.max_new_tokens_default = max_new_tokens_default

        self.waiting: deque = deque()
        self.running: Dict[int, SimRequest] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.iter_count = 0

    def submit(self, prompt_len: int = 8, max_new_tokens: int = None) -> Future:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens_default
        req = SimRequest(prompt_len, max_new_tokens, time.time())
        with self.lock:
            self.waiting.append(req)
        return req.future

    def _schedule(self) -> List[SimRequest]:
        batch = []
        with self.lock:
            # 1. 移除已完成，set_result
            finished = [r for r in self.running.values() if r.is_finished]
            for r in finished:
                r.finish_time = time.time()
                r.future.set_result(r.req_id)
                self.running.pop(r.req_id, None)
            # 2. 保留 running 的 decode
            for r in list(self.running.values()):
                if len(batch) < self.max_num_seqs:
                    batch.append(r)
            # 3. 从 waiting 补入新请求（token budget 约束）
            still_waiting = deque()
            for r in self.waiting:
                if len(batch) < self.max_num_seqs:
                    r.status = "running"
                    if r.start_time < 0:
                        r.start_time = time.time()
                    self.running[r.req_id] = r
                    batch.append(r)
                else:
                    still_waiting.append(r)
            self.waiting = still_waiting
        return batch

    def _run_iteration(self, batch: List[SimRequest]):
        # 摊销模型：batch 越大 per-iter 越久，但 per-token 越省
        bs = len(batch)
        iter_time = self.base_iter + self.per_seq * bs * (self.amort ** (bs - 1))
        time.sleep(iter_time)
        for r in batch:
            if r.status == "running":
                r.generated += 1
        self.iter_count += 1

    def _worker_loop(self):
        while not self.stop_event.is_set():
            batch = self._schedule()
            if batch:
                self._run_iteration(batch)
            else:
                time.sleep(0.0005)

    def shutdown(self):
        self.stop_event.set()
        self.worker.join(timeout=5)


# ============================================================
# Benchmark 框架
# ============================================================

def percentile(sorted_list, p):
    if not sorted_list:
        return 0.0
    idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
    return sorted_list[idx]


def run_fixed_concurrency(engine: SimulatedEngine, concurrency: int,
                          prompt_len: int = 8, max_new_tokens: int = 8) -> dict:
    """固定并发数测试：同时提交 concurrency 个请求，等全部完成，记录各请求 latency。"""
    reqs = []
    submit_start = time.time()
    for i in range(concurrency):
        req = SimRequest(prompt_len, max_new_tokens, time.time())
        reqs.append(req)
        with engine.lock:
            engine.waiting.append(req)
    for r in reqs:
        r.future.result()
    total_time = time.time() - submit_start

    # 各请求的真实 latency = finish_time - submit_time
    latencies = sorted([r.finish_time - r.submit_time for r in reqs])
    total_tokens = concurrency * max_new_tokens
    throughput = total_tokens / total_time if total_time > 0 else 0
    return {
        "concurrency": concurrency,
        "total_time": total_time,
        "throughput": throughput,
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": percentile(latencies, 50),
        "p99_latency": percentile(latencies, 99),
        "total_tokens": total_tokens,
    }


def run_qps_test(engine: SimulatedEngine, qps: float, duration: float = 3.0,
                 prompt_len: int = 8, max_new_tokens: int = 8) -> dict:
    """固定 QPS 测试：以恒定速率发请求，持续 duration 秒。"""
    reqs = []
    start = time.time()
    next_send = start
    while time.time() - start < duration:
        if time.time() >= next_send:
            req = SimRequest(prompt_len, max_new_tokens, time.time())
            reqs.append(req)
            with engine.lock:
                engine.waiting.append(req)
            next_send += 1.0 / qps
        else:
            time.sleep(0.0005)
    latencies = []
    for r in reqs:
        r.future.result()
        latencies.append(r.finish_time - r.submit_time)
    latencies.sort()
    total_tokens = len(reqs) * max_new_tokens
    actual_time = time.time() - start
    return {
        "qps": qps,
        "total_requests": len(reqs),
        "throughput": total_tokens / actual_time,
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": percentile(latencies, 50),
        "p99_latency": percentile(latencies, 99),
    }


def scan_concurrency(engine: SimulatedEngine, levels: List[int]) -> List[dict]:
    results = []
    for c in levels:
        r = run_fixed_concurrency(engine, c)
        results.append(r)
        print(f"  concurrency={c:>3} | throughput={r['throughput']:>6.1f} tok/s | "
              f"avg_lat={r['avg_latency']*1000:>7.1f}ms | "
              f"p99_lat={r['p99_latency']*1000:>7.1f}ms")
    return results


def find_saturation_point(results: List[dict]) -> dict:
    """识别饱和点：throughput 增长率 < 5% 的拐点（throughput 不再显著增长）。"""
    for i in range(1, len(results)):
        prev_tp = results[i - 1]["throughput"]
        curr_tp = results[i]["throughput"]
        growth = (curr_tp - prev_tp) / prev_tp if prev_tp > 0 else 0
        if growth < 0.05:
            return {
                "concurrency": results[i]["concurrency"],
                "throughput": curr_tp,
                "latency": results[i]["avg_latency"],
                "index": i,
            }
    return {"concurrency": results[-1]["concurrency"],
            "throughput": results[-1]["throughput"],
            "latency": results[-1]["avg_latency"],
            "index": len(results) - 1}


def print_throughput_latency_table(results: List[dict], sat: dict):
    print("\n" + "=" * 80)
    print("Throughput-Latency 曲线（固定并发扫描）")
    print("=" * 80)
    print(f"{'conc':>5} | {'throughput':>10} | {'avg_lat':>9} | {'p50_lat':>9} | "
          f"{'p99_lat':>9} | {'区域':<16}")
    print("-" * 80)
    for i, r in enumerate(results):
        if i < sat["index"]:
            note = "线性增长区"
        elif i == sat["index"]:
            note = "← 饱和点"
        else:
            note = "饱和后(排队)"
        print(f"{r['concurrency']:>5} | {r['throughput']:>8.1f}   | "
              f"{r['avg_latency']*1000:>7.1f}ms | {r['p50_latency']*1000:>7.1f}ms | "
              f"{r['p99_latency']*1000:>7.1f}ms | {note:<16}")
    print(f"\n  饱和点：concurrency={sat['concurrency']}, "
          f"throughput≈{sat['throughput']:.1f} tok/s, "
          f"latency={sat['latency']*1000:.1f}ms")
    print(f"  超过饱和点后：throughput 不再增长，latency 因排队急剧上升")


def main():
    print("Mini 引擎 v1 Latency/Throughput Benchmark")
    print("（用 SimulatedEngine 复刻 v1 的 Continuous Batching 调度行为）\n")

    # max_num_seqs=8：饱和拐点出现在 concurrency=8 附近
    engine = SimulatedEngine(
        max_num_seqs=8, max_token_budget=256,
        base_iter_ms=5.0, per_seq_ms=2.0,
        amort=0.85, max_new_tokens_default=8,
    )

    print("=" * 80)
    print("① 固定并发数扫描（concurrency = 1,2,4,8,16,32,64）")
    print("=" * 80)
    levels = [1, 2, 4, 8, 16, 32, 64]
    results = scan_concurrency(engine, levels)

    sat = find_saturation_point(results)
    print_throughput_latency_table(results, sat)

    # ② 固定 QPS 测试
    print("\n" + "=" * 80)
    print("② 固定 QPS 测试（恒定速率发请求，观察 P50/P99）")
    print("=" * 80)
    for qps in [5, 20, 50, 100, 200]:
        r = run_qps_test(engine, qps, duration=3.0)
        print(f"  qps={qps:>4} | requests={r['total_requests']:>3} | "
              f"throughput={r['throughput']:>6.1f} tok/s | "
              f"avg={r['avg_latency']*1000:>7.1f}ms | "
              f"p50={r['p50_latency']*1000:>7.1f}ms | "
              f"p99={r['p99_latency']*1000:>7.1f}ms")

    # ③ 瓶颈分析
    print("\n" + "=" * 80)
    print("③ 瓶颈分析与优化方向")
    print("=" * 80)
    print(f"  饱和点 throughput ≈ {sat['throughput']:.1f} tok/s（concurrency={sat['concurrency']}）")
    print(f"  饱和后 p99 latency 急剧上升（排队效应，concurrency 翻倍 → latency 翻倍）")
    print(f"\n  优化方向：")
    print(f"    • Compute-bound（throughput 不涨、SM util 高）→ 量化、更大 batch、模型优化")
    print(f"    • Memory-bound（latency 随并发线性涨）→ KV Cache 量化、PagedAttention")
    print(f"    • Launch overhead（kernel 间隙大）→ CUDA Graph、kernel fusion")
    print(f"    • Scheduling overhead（scheduler 成瓶颈）→ C++ scheduler、预分配 buffer")

    engine.shutdown()
    print("\n✅ Benchmark done.")


if __name__ == "__main__":
    main()
