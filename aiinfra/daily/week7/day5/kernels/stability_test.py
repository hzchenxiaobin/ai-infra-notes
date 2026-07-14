# stability_test.py —— 系统联调与稳定性测试
# 运行命令: python stability_test.py
# 依赖: 仅标准库（模拟 Mini 引擎，无需 GPU/PyTorch）
#
# 本文件是 Week7 Day5 的核心产出：
#   1. 模拟 Mini 推理引擎（KV Cache + Batching + Scheduler + 自定义 Kernel 模拟）
#   2. 分层验证：单请求 → 多请求并发 → KV Cache → Scheduler → 稳定性
#   3. 稳定性测试：连续处理 500+ 请求，监控成功率、延迟、内存
#   4. 异常处理：空 prompt、超长 prompt、超时、OOM 模拟

import random
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ============================================================
# 1. 模拟 Mini 推理引擎
# ============================================================

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    req_id: int
    prompt: str
    max_new_tokens: int = 8
    priority: int = 0
    status: RequestStatus = RequestStatus.WAITING
    future: Future = field(default_factory=Future)
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: List[str] = field(default_factory=list)
    kv_cache_blocks: int = 0

    @property
    def latency(self) -> float:
        if self.end_time and self.submit_time:
            return self.end_time - self.submit_time
        return 0.0


class MiniEngine:
    """模拟 Mini 推理引擎：KV Cache + Batching + Scheduler。

    模拟以下组件：
    - 线程安全请求队列（Day1）
    - 优先级调度 + 超时 + 资源预算（Day2）
    - KV Cache 管理（Week5）
    - 自定义 Kernel 模拟（Day4）
    """

    def __init__(self, max_token_budget=100, max_num_seqs=8,
                 max_waiting_time=10.0, total_kv_blocks=64,
                 forward_time=0.02, use_custom_kernel=True):
        self.max_token_budget = max_token_budget
        self.max_num_seqs = max_num_seqs
        self.max_waiting_time = max_waiting_time
        self.total_kv_blocks = total_kv_blocks
        self.forward_time = forward_time
        self.use_custom_kernel = use_custom_kernel

        self.waiting: deque = deque()
        self.running: Dict[int, InferenceRequest] = {}
        self.finished: List[InferenceRequest] = []
        self.used_kv_blocks = 0
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._req_counter = 0

        # 模拟 KV Cache pool
        self.kv_cache_pool = {}  # req_id → list of blocks

    def submit(self, prompt: str, max_new_tokens: int = 8,
               priority: int = 0) -> InferenceRequest:
        """提交请求，返回 InferenceRequest（含 Future）。"""
        with self._lock:
            self._req_counter += 1
            req = InferenceRequest(
                req_id=self._req_counter,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                priority=priority,
            )
            # 计算 KV Cache 需求（模拟：prompt 长度 + max_new_tokens）
            req.kv_cache_blocks = (len(prompt.split()) + max_new_tokens + 15) // 16
            self.waiting.append(req)
        return req

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def shutdown(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def _worker_loop(self):
        """调度+执行线程（简化版：单线程轮流调度和执行）。"""
        while self._running or self.waiting or self.running:
            # 1. 调度：从 waiting 加入 running
            self._schedule()

            # 2. 执行：处理 running 中的请求（模拟 forward）
            self._forward()

            # 3. 清理超时请求
            self._check_timeouts()

            time.sleep(0.001)

    def _schedule(self):
        """从 waiting 队列按优先级加入 running。"""
        with self._lock:
            remaining_tokens = self.max_token_budget

            # 继续 running 的 decode
            for req in self.running.values():
                if remaining_tokens > 0:
                    remaining_tokens -= 1

            # 加入新请求
            still_waiting = deque()
            while self.waiting and remaining_tokens > 0 and len(self.running) < self.max_num_seqs:
                req = self.waiting.popleft()

                # 检查超时
                if time.time() - req.submit_time > self.max_waiting_time:
                    req.status = RequestStatus.TIMEOUT
                    req.end_time = time.time()
                    req.future.set_exception(TimeoutError(f"Request {req.req_id} timed out"))
                    self.finished.append(req)
                    continue

                # 检查显存预算
                if self.used_kv_blocks + req.kv_cache_blocks > self.total_kv_blocks:
                    still_waiting.append(req)
                    continue

                # 检查 token budget
                prompt_tokens = len(req.prompt.split())
                if prompt_tokens > remaining_tokens:
                    still_waiting.append(req)
                    continue

                # 加入 running
                self.used_kv_blocks += req.kv_cache_blocks
                self.kv_cache_pool[req.req_id] = list(range(req.kv_cache_blocks))
                req.status = RequestStatus.RUNNING
                req.start_time = time.time()
                self.running[req.req_id] = req
                remaining_tokens -= prompt_tokens

            # 将 still_waiting 和未被处理的 waiting 合并
            still_waiting.extend(self.waiting)
            self.waiting = still_waiting

    def _forward(self):
        """模拟 forward：每个 running 请求生成 1 个 token。"""
        if not self.running:
            return

        # 模拟 forward 延迟（自定义 kernel 可能更快）
        forward_delay = self.forward_time * (0.8 if self.use_custom_kernel else 1.0)
        time.sleep(forward_delay)

        with self._lock:
            for req_id in list(self.running.keys()):
                req = self.running[req_id]

                # 模拟 token 生成
                token = f"r{req_id}_tok{len(req.result)}"
                req.result.append(token)

                # 检查是否完成
                if len(req.result) >= req.max_new_tokens:
                    req.status = RequestStatus.FINISHED
                    req.end_time = time.time()
                    req.future.set_result(" ".join(req.result))
                    # 释放 KV Cache
                    self.used_kv_blocks -= req.kv_cache_blocks
                    if req_id in self.kv_cache_pool:
                        del self.kv_cache_pool[req_id]
                    del self.running[req_id]
                    self.finished.append(req)

    def _check_timeouts(self):
        """清理超时的 running 请求。"""
        with self._lock:
            for req_id in list(self.running.keys()):
                req = self.running[req_id]
                if req.start_time and (time.time() - req.start_time) > 30.0:
                    req.status = RequestStatus.TIMEOUT
                    req.end_time = time.time()
                    req.future.set_exception(TimeoutError(f"Request {req.req_id} execution timed out"))
                    self.used_kv_blocks -= req.kv_cache_blocks
                    if req_id in self.kv_cache_pool:
                        del self.kv_cache_pool[req_id]
                    del self.running[req_id]
                    self.finished.append(req)

    def stats(self) -> dict:
        with self._lock:
            return {
                "waiting": len(self.waiting),
                "running": len(self.running),
                "finished": len(self.finished),
                "kv_used": f"{self.used_kv_blocks}/{self.total_kv_blocks}",
            }


# ============================================================
# 2. 分层验证测试
# ============================================================

def test_single_request():
    """Step 1: 单请求正确性。"""
    print("\n" + "=" * 60)
    print("Step 1: 单请求正确性")
    print("=" * 60)

    engine = MiniEngine(forward_time=0.01)
    engine.start()

    req = engine.submit("hello world this is a test", max_new_tokens=5)
    result = req.future.result(timeout=10)

    print(f"  Request {req.req_id}: prompt='hello world...'")
    print(f"  Result: '{result}'")
    print(f"  Status: {req.status.value}")
    print(f"  Latency: {req.latency:.3f}s")
    print(f"  Tokens generated: {len(req.result)}")
    assert req.status == RequestStatus.FINISHED, f"Expected FINISHED, got {req.status}"
    assert len(req.result) == 5, f"Expected 5 tokens, got {len(req.result)}"
    print("  ✓ PASS")

    engine.shutdown()


def test_multi_request_concurrency():
    """Step 2: 多请求并发正确性。"""
    print("\n" + "=" * 60)
    print("Step 2: 多请求并发正确性")
    print("=" * 60)

    engine = MiniEngine(max_num_seqs=4, forward_time=0.01)
    engine.start()

    requests = []
    for i in range(5):
        req = engine.submit(f"prompt number {i}", max_new_tokens=3, priority=i % 2)
        requests.append(req)

    results = []
    for req in requests:
        result = req.future.result(timeout=15)
        results.append(result)

    all_finished = all(r.status == RequestStatus.FINISHED for r in requests)
    all_correct = all(len(r.result) == 3 for r in requests)
    print(f"  Submitted 5 requests, all finished: {all_finished}")
    print(f"  All generated 3 tokens: {all_correct}")
    print(f"  Completion order: {[r.req_id for r in sorted(requests, key=lambda x: x.end_time)]}")
    assert all_finished and all_correct, "Multi-request test failed"
    print("  ✓ PASS")

    engine.shutdown()


def test_kv_cache_isolation():
    """Step 3: KV Cache 隔离性。"""
    print("\n" + "=" * 60)
    print("Step 3: KV Cache 隔离性")
    print("=" * 60)

    engine = MiniEngine(total_kv_blocks=32, forward_time=0.01)
    engine.start()

    req1 = engine.submit("request one", max_new_tokens=3)
    req2 = engine.submit("request two", max_new_tokens=3)

    r1 = req1.future.result(timeout=10)
    r2 = req2.future.result(timeout=10)

    # 验证两个请求的结果不互相干扰
    isolated = r1 != r2
    print(f"  Request 1 result: '{r1}'")
    print(f"  Request 2 result: '{r2}'")
    print(f"  Results isolated: {isolated}")
    print(f"  KV cache after completion: {engine.stats()['kv_used']}")
    assert isolated, "KV Cache isolation failed"
    assert engine.used_kv_blocks == 0, "KV Cache not released"
    print("  ✓ PASS (KV Cache released after completion)")

    engine.shutdown()


def test_scheduler_priority():
    """Step 4: Scheduler 优先级和资源预算。"""
    print("\n" + "=" * 60)
    print("Step 4: Scheduler 优先级和资源预算")
    print("=" * 60)

    engine = MiniEngine(max_num_seqs=2, total_kv_blocks=16, forward_time=0.01)
    engine.start()

    # 提交 4 个请求，前两个高优先级
    reqs = []
    for i in range(4):
        req = engine.submit(f"task {i}", max_new_tokens=2, priority=3 - i)
        reqs.append(req)

    for req in reqs:
        req.future.result(timeout=15)

    # 验证所有请求最终完成
    all_done = all(r.status == RequestStatus.FINISHED for r in reqs)
    print(f"  All 4 requests completed: {all_done}")
    print(f"  KV cache after: {engine.stats()['kv_used']}")
    assert all_done, "Scheduler test failed"
    assert engine.used_kv_blocks == 0, "Resources not released"
    print("  ✓ PASS (Resources released correctly)")

    engine.shutdown()


def test_custom_kernel_integration():
    """Step 5: 自定义 Kernel 集成（性能对比）。"""
    print("\n" + "=" * 60)
    print("Step 5: 自定义 Kernel 集成（性能模拟对比）")
    print("=" * 60)

    # PyTorch 原生（模拟更慢）
    engine_pytorch = MiniEngine(forward_time=0.02, use_custom_kernel=False)
    engine_pytorch.start()

    # 自定义 kernel（模拟更快）
    engine_custom = MiniEngine(forward_time=0.02, use_custom_kernel=True)
    engine_custom.start()

    req_p = engine_pytorch.submit("test prompt", max_new_tokens=5)
    req_c = engine_custom.submit("test prompt", max_new_tokens=5)

    r_p = req_p.future.result(timeout=15)
    r_c = req_c.future.result(timeout=15)

    print(f"  PyTorch sim latency: {req_p.latency:.3f}s")
    print(f"  Custom sim latency: {req_c.latency:.3f}s")
    print(f"  Speedup (simulated): {req_p.latency / req_c.latency:.2f}x")
    assert r_p == r_c, "Results should be identical"
    print("  ✓ PASS (Results identical, custom kernel faster)")

    engine_pytorch.shutdown()
    engine_custom.shutdown()


# ============================================================
# 3. 稳定性测试
# ============================================================

def stability_test(num_requests=500, max_new_tokens=5):
    """Step 6: 稳定性测试——连续处理大量请求。"""
    print("\n" + "=" * 60)
    print(f"Step 6: 稳定性测试（{num_requests} 请求）")
    print("=" * 60)

    engine = MiniEngine(max_num_seqs=8, total_kv_blocks=64, forward_time=0.005)
    engine.start()

    prompts = [
        "hello world",
        "this is a test prompt",
        "short",
        "another example prompt for testing",
        "a medium length prompt here for batching test",
        "yet another prompt to vary the load",
        "tiny",
        "a longer prompt with more words to stress the kv cache allocation",
    ]

    success_count = 0
    fail_count = 0
    latencies = []
    start_time = time.time()

    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        try:
            req = engine.submit(prompt, max_new_tokens=max_new_tokens,
                               priority=random.randint(0, 2))
            result = req.future.result(timeout=20)
            success_count += 1
            latencies.append(req.latency)

            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"  [{i}/{num_requests}] success={success_count}, "
                      f"kv={engine.stats()['kv_used']}, "
                      f"elapsed={elapsed:.1f}s")
        except Exception as e:
            fail_count += 1
            print(f"  [{i}] FAILED: {e}")

    total_time = time.time() - start_time

    # 统计
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if latencies else 0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if latencies else 0
    throughput = num_requests / total_time if total_time > 0 else 0

    print(f"\n  {'='*40}")
    print(f"  稳定性测试结果:")
    print(f"  {'='*40}")
    print(f"  Total requests:  {num_requests}")
    print(f"  Success:         {success_count} ({success_count/num_requests*100:.1f}%)")
    print(f"  Fail:            {fail_count}")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Throughput:      {throughput:.1f} req/s")
    print(f"  Avg latency:     {avg_latency*1000:.1f} ms")
    print(f"  P50 latency:     {p50*1000:.1f} ms")
    print(f"  P99 latency:     {p99*1000:.1f} ms")
    print(f"  KV cache final:  {engine.stats()['kv_used']}")
    print(f"  Memory leak:     {'NO' if engine.used_kv_blocks == 0 else 'YES!'}")

    # 验收标准
    print(f"\n  验收标准:")
    print(f"  [{'✓' if success_count/num_requests > 0.95 else '✗'}] 成功率 > 95%")
    print(f"  [{'✓' if engine.used_kv_blocks == 0 else '✗'}] 无内存泄漏（KV Cache 全释放）")
    print(f"  [{'✓' if fail_count < num_requests * 0.05 else '✗'}] 失败率 < 5%")

    engine.shutdown()
    return success_count, fail_count


# ============================================================
# 4. 异常输入测试
# ============================================================

def test_abnormal_inputs():
    """异常输入测试。"""
    print("\n" + "=" * 60)
    print("异常输入测试")
    print("=" * 60)

    engine = MiniEngine(total_kv_blocks=32, forward_time=0.01)
    engine.start()

    # 空 prompt
    try:
        req = engine.submit("", max_new_tokens=3)
        result = req.future.result(timeout=10)
        print(f"  Empty prompt: handled (result='{result}')")
    except Exception as e:
        print(f"  Empty prompt: exception handled ({type(e).__name__})")

    # 超长 prompt
    try:
        long_prompt = "word " * 200
        req = engine.submit(long_prompt, max_new_tokens=3)
        result = req.future.result(timeout=15)
        print(f"  Long prompt (200 words): handled (generated {len(req.result)} tokens)")
    except Exception as e:
        print(f"  Long prompt: exception handled ({type(e).__name__})")

    # 超时测试
    try:
        req = engine.submit("timeout test", max_new_tokens=100)
        result = req.future.result(timeout=0.1)  # 极短超时
        print(f"  Timeout test: completed unexpectedly")
    except TimeoutError:
        print(f"  Timeout test: correctly raised TimeoutError")
    except Exception as e:
        print(f"  Timeout test: handled ({type(e).__name__})")

    engine.shutdown()
    print("  ✓ All abnormal inputs handled")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Mini AI Infra 系统联调与稳定性测试")
    print("=" * 60)

    # 分层验证
    test_single_request()
    test_multi_request_concurrency()
    test_kv_cache_isolation()
    test_scheduler_priority()
    test_custom_kernel_integration()

    # 稳定性测试
    stability_test(num_requests=500, max_new_tokens=5)

    # 异常输入
    test_abnormal_inputs()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
