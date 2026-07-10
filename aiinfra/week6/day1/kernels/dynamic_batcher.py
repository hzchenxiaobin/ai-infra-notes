# dynamic_batcher.py —— Dynamic Batching 实现（请求队列 + 超时等待 + 最大 batch size）
# 运行命令: python dynamic_batcher.py
# 依赖: 仅标准库

import time
import threading
from collections import deque
from typing import List, Optional


class Request:
    """一个推理请求"""
    def __init__(self, request_id: int, prompt_len: int, max_new_tokens: int = 10):
        self.request_id = request_id
        self.prompt_len = prompt_len        # prompt 长度（用于 padding 代价计算）
        self.max_new_tokens = max_new_tokens
        self.arrival_time = time.time()
        self.batch_id = -1                   # 被分到哪个 batch
        self.batch_size = 0                  # 所在 batch 的大小
        self.result = None
        self.done_event = threading.Event()

    @property
    def wait_time(self) -> float:
        """在队列中等待的时间（ms）"""
        return (time.time() - self.arrival_time) * 1000


class DynamicBatcher:
    """
    Dynamic Batcher：请求队列 + 超时等待 + 最大 batch size

    参数：
      max_batch_size:  每个 batch 最多聚合多少请求
      max_wait_time:   凑不满 batch 时最多等多久（秒）
    """

    def __init__(self, max_batch_size: int = 4, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue: deque[Request] = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.batch_count = 0

    def submit(self, request: Request):
        """提交请求到队列"""
        with self.lock:
            self.queue.append(request)

    def _collect_batch(self) -> List[Request]:
        """收集一个 batch：等待 timeout 或凑够 max_batch_size"""
        batch = []

        # 先等第一个请求到达
        while not batch and not self.stop_event.is_set():
            with self.lock:
                if self.queue:
                    batch.append(self.queue.popleft())
            if not batch:
                time.sleep(0.0005)

        if not batch:
            return []

        # 第一个请求到了，启动 timer 等更多
        deadline = time.time() + self.max_wait_time
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break   # 超时，发出去
            with self.lock:
                if self.queue:
                    batch.append(self.queue.popleft())
                else:
                    # 队列暂时空，短暂等待
                    time.sleep(min(0.001, remaining))

        return batch

    def _process_batch(self, batch: List[Request]):
        """处理一个 batch（用 sleep 模拟模型 forward）

        关键特性：batch 越大，per-request 时间越少（GEMM 的 M 增大，吞吐提升）
        """
        self.batch_count += 1
        batch_size = len(batch)

        # 计算 padding 代价（pad 到 batch 内最长 prompt）
        max_len = max(r.prompt_len for r in batch)
        total_padded = batch_size * max_len
        total_actual = sum(r.prompt_len for r in batch)
        padding_waste = total_padded - total_actual

        # 模拟 forward：base time + per-request time（batch 越大 per-request 越省）
        forward_time = 0.001 + 0.0005 * batch_size   # 秒
        time.sleep(forward_time)

        for req in batch:
            req.batch_id = self.batch_count
            req.batch_size = batch_size
            req.result = f"ok(batch={self.batch_count},bs={batch_size})"
            req.done_event.set()

        if batch_size > 1:
            print(f"  Batch {self.batch_count}: size={batch_size}, "
                  f"forward={forward_time*1000:.1f}ms, "
                  f"padding_waste={padding_waste} tokens "
                  f"({100*padding_waste/total_padded:.0f}%)")

    def _worker_loop(self):
        while not self.stop_event.is_set():
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

    def shutdown(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=2)


def main():
    print("=" * 64)
    print("Dynamic Batcher Demo")
    print("=" * 64)

    # 场景：10 个请求，间隔 20ms 到达，max_batch=4, wait=50ms
    batcher = DynamicBatcher(max_batch_size=4, max_wait_time=0.02)

    requests = []
    print("\nSubmitting 10 requests (interval=5ms, max_batch=4, wait=20ms)...\n")
    for i in range(10):
        prompt_len = 3 + (i % 5)   # prompt 长度 3-7，模拟不等长
        req = Request(request_id=i, prompt_len=prompt_len, max_new_tokens=5)
        batcher.submit(req)
        requests.append(req)
        time.sleep(0.005)   # 5ms 间隔（快到达，凑得出 batch）

    # 等待全部完成
    for req in requests:
        req.done_event.wait()
    total_time = max(r.arrival_time for r in requests) - min(r.arrival_time for r in requests)

    print(f"\n{'ID':>3} {'Batch':>6} {'BS':>3} {'Wait(ms)':>10} {'PromptLen':>10} {'Result':>30}")
    print("-" * 70)
    for req in requests:
        print(f"{req.request_id:>3} {req.batch_id:>6} {req.batch_size:>3} "
              f"{req.wait_time:>10.1f} {req.prompt_len:>10} {req.result:>30}")

    print(f"\nTotal batches: {batcher.batch_count}")
    print(f"Avg wait time: {sum(r.wait_time for r in requests)/len(requests):.1f} ms")

    # 对比：单请求 vs batch 的理论吞吐
    print("\n=== Throughput Comparison ===")
    single_time = 0.001 + 0.0005 * 1    # batch_size=1
    batch4_time = 0.001 + 0.0005 * 4    # batch_size=4
    single_throughput = 1 / single_time
    batch4_throughput = 4 / batch4_time
    print(f"  Single (B=1): {single_time*1000:.1f}ms/req, throughput={single_throughput:.0f} req/s")
    print(f"  Batch  (B=4): {batch4_time*1000:.1f}ms/req, throughput={batch4_throughput:.0f} req/s")
    print(f"  Speedup: {batch4_throughput/single_throughput:.2f}x")

    batcher.shutdown()
    print("\n✅ Dynamic Batcher demo done.")


if __name__ == "__main__":
    main()
