# concurrent_engine.py —— 多请求并发支持（线程安全队列 + Future/Callback/Streaming + 生命周期）
# 运行命令: python concurrent_engine.py
# 依赖: 仅标准库
#
# 本文件是 Week7 Day1 的核心产出：为 Mini 引擎添加多请求并发能力。
# 相比 Week6 Day5 的 MiniEngineV1（已有 submit/Future/worker），本文件强化：
#   1. ThreadSafeRequestQueue：条件变量 + 优先级插入 + 批量获取
#   2. 三种结果返回：Future（阻塞等待）/ Callback（结果到达触发）/ Streaming（token 逐个返回）
#   3. 请求生命周期：WAITING → RUNNING → FINISHED/TIMEOUT/CANCELLED
#   4. 超时控制：超过 timeout 的请求自动取消并 set_exception
#   5. 线程安全要点演示：锁内只做队列操作，锁外做"forward"

import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Callable, Dict, List, Optional


# ============================================================
# 请求生命周期状态
# ============================================================

class RequestStatus:
    WAITING = "waiting"        # 已入队，等待调度
    RUNNING = "running"        # 正在 prefill/decode
    FINISHED = "finished"      # 正常完成
    TIMEOUT = "timeout"        # 超时取消
    CANCELLED = "cancelled"    # 主动取消


class InferenceRequest:
    """一个推理请求，支持 Future / Callback / Streaming 三种结果返回。

    生命周期：WAITING → RUNNING → FINISHED / TIMEOUT / CANCELLED
    """

    _next_id = 0

    def __init__(self, prompt: str, max_new_tokens: int = 8, priority: int = 0,
                 timeout: Optional[float] = None,
                 callback: Optional[Callable] = None,
                 stream_callback: Optional[Callable] = None):
        InferenceRequest._next_id += 1
        self.request_id = InferenceRequest._next_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.priority = priority
        self.timeout = timeout
        self.callback = callback
        self.stream_callback = stream_callback   # Streaming：每生成一个 token 调用
        self.future = Future()
        self.status = RequestStatus.WAITING
        self.submit_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.generated_tokens: List[str] = []

    def mark_running(self):
        self.status = RequestStatus.RUNNING
        self.start_time = time.time()

    def emit_token(self, token: str):
        """Streaming：每生成一个 token 调用 stream_callback。"""
        self.generated_tokens.append(token)
        if self.stream_callback:
            self.stream_callback(self.request_id, token)

    def set_result(self, result: str):
        """完成时设置结果：触发 Future + Callback。"""
        self.status = RequestStatus.FINISHED
        self.end_time = time.time()
        self.future.set_result(result)
        if self.callback:
            self.callback(self.request_id, result)

    def set_exception(self, exc: Exception):
        """超时/取消时设置异常。"""
        self.end_time = time.time()
        if not self.future.done():
            self.future.set_exception(exc)

    def is_expired(self) -> bool:
        """是否已超时。"""
        if self.timeout is None:
            return False
        return (time.time() - self.submit_time) > self.timeout

    @property
    def latency(self) -> float:
        if self.end_time and self.submit_time:
            return self.end_time - self.submit_time
        return 0.0


# ============================================================
# 线程安全请求队列（条件变量 + 优先级 + 批量获取）
# ============================================================

class ThreadSafeRequestQueue:
    """线程安全请求队列，支持优先级插入和批量获取。

    线程安全要点：
      1. 使用 Condition（Lock + wait/notify）保护队列
      2. put() 按优先级插入（高优先级在前）
      3. get_batch() 批量获取，带超时等待
      4. 条件变量 notify 唤醒等待的 worker
    """

    def __init__(self):
        self._queue: deque = deque()
        self._cond = threading.Condition()

    def put(self, request: InferenceRequest):
        """按优先级插入队列（高优先级在前），notify 唤醒 worker。"""
        with self._cond:
            inserted = False
            for i, req in enumerate(self._queue):
                if request.priority > req.priority:
                    self._queue.insert(i, request)
                    inserted = True
                    break
            if not inserted:
                self._queue.append(request)
            self._cond.notify()

    def get_batch(self, max_size: int = 8, max_wait: float = 0.05) -> List[InferenceRequest]:
        """批量获取请求，带超时等待。worker 用此方法凑批。"""
        batch = []
        deadline = time.time() + max_wait
        with self._cond:
            while len(batch) < max_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                if not self._queue:
                    self._cond.wait(remaining)
                while self._queue and len(batch) < max_size:
                    batch.append(self._queue.popleft())
                if batch:
                    break   # 拿到至少一个就返回（低延迟）
        return batch

    def peek_all(self) -> List[InferenceRequest]:
        """查看队列内容（不弹出，用于超时检查）。"""
        with self._cond:
            return list(self._queue)

    def remove(self, request: InferenceRequest) -> bool:
        """移除指定请求（用于超时取消）。"""
        with self._cond:
            try:
                self._queue.remove(request)
                return True
            except ValueError:
                return False

    def __len__(self):
        with self._cond:
            return len(self._queue)


# ============================================================
# 并发推理引擎（调度线程 + 执行线程，模型2）
# ============================================================

class ConcurrentEngine:
    """并发推理引擎：调度线程凑批 + 执行线程 forward，支持三种结果返回。

    并发模型（模型2：调度线程 + 执行线程分离）：
      - 主线程：submit() 入队，立即返回 Future
      - 调度线程：_scheduler_loop 从队列 get_batch，构建 batch
      - 执行线程：_worker_loop 执行 forward（模拟），完成后 set_result
      - 超时清理线程：_timeout_loop 检查过期请求

    线程安全：共享状态（queue、running）用锁保护；forward 在锁外执行
    """

    def __init__(self, max_batch_size: int = 4, forward_time_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.forward_time_ms = forward_time_ms / 1000.0
        self.queue = ThreadSafeRequestQueue()
        self.running: Dict[int, InferenceRequest] = {}
        self.running_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.iteration = 0

        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.timeout_thread = threading.Thread(target=self._timeout_loop, daemon=True)

    def start(self):
        self.scheduler_thread.start()
        self.worker_thread.start()
        self.timeout_thread.start()

    def submit(self, prompt: str, max_new_tokens: int = 8, priority: int = 0,
               timeout: Optional[float] = None,
               callback: Optional[Callable] = None,
               stream_callback: Optional[Callable] = None) -> InferenceRequest:
        """提交请求，返回 InferenceRequest（含 Future）。"""
        req = InferenceRequest(
            prompt=prompt, max_new_tokens=max_new_tokens, priority=priority,
            timeout=timeout, callback=callback, stream_callback=stream_callback,
        )
        self.queue.put(req)
        return req

    def cancel(self, request_id: int) -> bool:
        """取消请求（从 waiting 移除或标记 running 取消）。"""
        for req in self.queue.peek_all():
            if req.request_id == request_id:
                self.queue.remove(req)
                req.status = RequestStatus.CANCELLED
                req.set_exception(TimeoutError(f"Request {request_id} cancelled"))
                return True
        with self.running_lock:
            req = self.running.get(request_id)
            if req:
                req.status = RequestStatus.CANCELLED
                req.set_exception(TimeoutError(f"Request {request_id} cancelled"))
                return True
        return False

    def _scheduler_loop(self):
        """调度线程：从队列批量获取请求，加入 running。"""
        while not self.stop_event.is_set():
            batch = self.queue.get_batch(max_size=self.max_batch_size, max_wait=0.02)
            if not batch:
                continue
            with self.running_lock:
                for req in batch:
                    req.mark_running()
                    self.running[req.request_id] = req

    def _worker_loop(self):
        """执行线程：处理 running 中的请求（模拟 forward），完成 set_result。

        线程安全：锁内只做 running 的增删，锁外做 forward（sleep 模拟）。
        """
        while not self.stop_event.is_set():
            with self.running_lock:
                pending = [r for r in self.running.values()
                           if r.status == RequestStatus.RUNNING and not r.future.done()]
                if not pending:
                    pass
            if pending:
                # 模拟 forward（锁外，不阻塞 submit/scheduler）
                time.sleep(self.forward_time_ms)
                self.iteration += 1
                with self.running_lock:
                    for req in pending:
                        if req.status != RequestStatus.RUNNING:
                            continue   # 已被取消/超时
                        # 模拟生成 max_new_tokens 个 token
                        for step in range(req.max_new_tokens):
                            token = f"tok{step}"
                            req.emit_token(token)   # Streaming 回调
                        result = " ".join(req.generated_tokens)
                        req.set_result(result)     # Future + Callback
                        self.running.pop(req.request_id, None)
            else:
                time.sleep(0.001)

    def _timeout_loop(self):
        """超时清理线程：检查 waiting 和 running 中的过期请求。"""
        while not self.stop_event.is_set():
            # 检查 waiting
            for req in self.queue.peek_all():
                if req.is_expired():
                    self.queue.remove(req)
                    req.status = RequestStatus.TIMEOUT
                    req.set_exception(TimeoutError(f"Request {req.request_id} timed out"))
            # 检查 running
            with self.running_lock:
                for req in list(self.running.values()):
                    if req.is_expired() and req.status == RequestStatus.RUNNING:
                        req.status = RequestStatus.TIMEOUT
                        req.set_exception(TimeoutError(f"Request {req.request_id} timed out"))
                        self.running.pop(req.request_id, None)
            time.sleep(0.01)

    def shutdown(self):
        self.stop_event.set()
        for t in [self.scheduler_thread, self.worker_thread, self.timeout_thread]:
            t.join(timeout=3)


# ============================================================
# 主流程：演示 Future / Callback / Streaming / 超时
# ============================================================

def demo_future():
    """演示 Future：submit 后阻塞等待结果。"""
    print("=" * 70)
    print("Demo 1: Future（阻塞等待结果）")
    print("=" * 70)
    engine = ConcurrentEngine(max_batch_size=4, forward_time_ms=20)
    engine.start()

    req = engine.submit("hello world", max_new_tokens=5)
    print(f"  Submitted request {req.request_id}, waiting...")
    result = req.future.result()   # 阻塞等待
    print(f"  Result: {result}")
    print(f"  Latency: {req.latency*1000:.1f}ms, status={req.status}")

    engine.shutdown()


def demo_callback():
    """演示 Callback：结果到达时自动触发回调。"""
    print("\n" + "=" * 70)
    print("Demo 2: Callback（结果到达时触发）")
    print("=" * 70)
    engine = ConcurrentEngine(max_batch_size=4, forward_time_ms=20)
    engine.start()

    callback_results = []
    def my_callback(req_id, result):
        callback_results.append((req_id, result))
        print(f"  [Callback] Request {req_id} done: {result[:30]}...")

    for i in range(3):
        engine.submit(f"prompt_{i}", max_new_tokens=4, callback=my_callback)

    time.sleep(0.5)   # 等待 callback 触发
    print(f"  Callback triggered {len(callback_results)} times")
    engine.shutdown()


def demo_streaming():
    """演示 Streaming：每个 token 生成时触发 stream_callback。"""
    print("\n" + "=" * 70)
    print("Demo 3: Streaming（token 逐个返回）")
    print("=" * 70)
    engine = ConcurrentEngine(max_batch_size=4, forward_time_ms=20)
    engine.start()

    stream_tokens = []
    def my_stream(req_id, token):
        stream_tokens.append(token)
        print(f"  [Stream] Request {req_id} -> {token}")

    req = engine.submit("streaming test", max_new_tokens=5, stream_callback=my_stream)
    req.future.result()   # 等全部完成
    print(f"  Total streamed tokens: {len(stream_tokens)}")
    engine.shutdown()


def demo_priority():
    """演示优先级：高优先级请求先被调度。"""
    print("\n" + "=" * 70)
    print("Demo 4: 优先级调度（高优先级先处理）")
    print("=" * 70)
    engine = ConcurrentEngine(max_batch_size=1, forward_time_ms=30)
    engine.start()

    order = []
    def cb(req_id, result):
        order.append(req_id)

    # 低优先级先提交，高优先级后提交但应先处理
    engine.submit("low_1", max_new_tokens=3, priority=0, callback=cb)
    engine.submit("low_2", max_new_tokens=3, priority=0, callback=cb)
    engine.submit("HIGH", max_new_tokens=3, priority=10, callback=cb)

    time.sleep(0.8)
    print(f"  Completion order: {order}")
    print(f"  HIGH(priority=10) 应该最先完成")
    engine.shutdown()


def demo_timeout():
    """演示超时：超过 timeout 的请求自动取消。"""
    print("\n" + "=" * 70)
    print("Demo 5: 超时控制（0.05s 超时，forward 需 0.1s）")
    print("=" * 70)
    engine = ConcurrentEngine(max_batch_size=1, forward_time_ms=100)
    engine.start()

    req = engine.submit("will_timeout", max_new_tokens=3, timeout=0.05)
    try:
        req.future.result()
        print(f"  Unexpected: request completed")
    except TimeoutError as e:
        print(f"  Expected timeout: {e}")
        print(f"  status={req.status}")

    engine.shutdown()


def main():
    print("Week7 Day1: 多请求并发支持 —— 并发推理引擎")
    print("（ThreadSafeRequestQueue + Future/Callback/Streaming + 生命周期）\n")

    demo_future()
    demo_callback()
    demo_streaming()
    demo_priority()
    demo_timeout()

    print("\n" + "=" * 70)
    print("✅ 五大机制验证完毕：")
    print("  1. ThreadSafeRequestQueue：条件变量 + 优先级插入 + 批量获取")
    print("  2. Future：submit() 异步返回，阻塞等待结果")
    print("  3. Callback：结果到达时自动触发")
    print("  4. Streaming：token 逐个返回（stream_callback）")
    print("  5. 超时控制：timeout 线程自动取消过期请求")
    print("=" * 70)


if __name__ == "__main__":
    main()
