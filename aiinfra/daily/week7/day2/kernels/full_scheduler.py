# full_scheduler.py —— 完整调度器（优先级 + 超时 + 资源预算 + 抢占）
# 运行命令: python full_scheduler.py
# 依赖: 仅标准库
#
# 本文件是 Week7 Day2 的核心产出：在 Day1 的 ConcurrentEngine 基础上，
# 实现生产级调度器，支持：
#   1. 优先级调度：高优先级请求优先获得资源
#   2. 超时控制：等待超时 + 执行超时
#   3. 资源预算：token budget（计算预算）+ memory budget（显存预算）
#   4. 抢占：显存不足时抢占低优先级请求（recompute / swap 两种策略）
#   5. 公平性：防止低优先级请求无限饥饿（aging 机制）
#   6. Continuous Batching：running 请求继续 decode，新请求按预算加入

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ============================================================
# 请求状态与数据结构
# ============================================================

class RequestStatus(Enum):
    WAITING = "waiting"        # 已入队，等待调度
    RUNNING = "running"        # 正在 prefill/decode
    SWAPPED = "swapped"        # 被抢占并换出到 CPU
    FINISHED = "finished"      # 正常完成
    TIMEOUT = "timeout"        # 超时取消


@dataclass
class ScheduledRequest:
    """调度器中的请求表示。"""
    req_id: int
    prompt_len: int                              # prompt 长度（prefill 消耗的 token 数）
    max_new_tokens: int                          # 最大生成 token 数
    priority: int = 0                            # 优先级（越大越高）
    status: RequestStatus = RequestStatus.WAITING
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None           # 首次进入 RUNNING 的时间
    required_kv_blocks: int = 0                  # 该请求需要的 KV Cache 块数
    generated_tokens: int = 0                    # 已生成的 token 数
    original_priority: int = 0                   # 原始优先级（aging 用）
    swapped_data: Optional[object] = None        # swap 策略：换出的 KV Cache 数据

    def __lt__(self, other):
        return self.req_id < other.req_id


# ============================================================
# 显存预算管理
# ============================================================

class MemoryBudget:
    """模拟 GPU 显存的块分配（类比 vLLM PagedAttention 的 block allocator）。

    显存被划分为固定大小的 block（如每 block 存 16 个 token 的 KV Cache）。
    调度器在加入新请求时检查是否有足够空闲 block。
    """

    def __init__(self, total_blocks: int = 64):
        self.total_blocks = total_blocks
        self.used_blocks = 0

    def can_allocate(self, blocks: int) -> bool:
        return self.used_blocks + blocks <= self.total_blocks

    def allocate(self, blocks: int):
        if not self.can_allocate(blocks):
            raise RuntimeError(f"Out of memory: need {blocks}, have {self.total_blocks - self.used_blocks}")
        self.used_blocks += blocks

    def free(self, blocks: int):
        self.used_blocks = max(0, self.used_blocks - blocks)

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.used_blocks

    def __repr__(self):
        return f"MemoryBudget({self.used_blocks}/{self.total_blocks})"


# ============================================================
# 完整调度器
# ============================================================

class FullScheduler:
    """生产级调度器，支持六大功能。

    六大功能：
      1. 优先级调度    — heapq 按 (-priority, submit_time) 排序
      2. 超时控制      — waiting 超时丢弃，running 超时强制取消
      3. 资源预算      — token_budget（每轮 iteration 的计算预算）+ memory_budget（显存块）
      4. 抢占          — 显存不足时抢占最低优先级 running 请求（recompute 或 swap）
      5. 公平性        — aging 机制：等待时间过长自动提升优先级
      6. Continuous Batching — running 请求每轮 decode 消耗 1 token budget
    """

    def __init__(self,
                 token_budget: int = 100,
                 max_num_seqs: int = 8,
                 max_waiting_time: float = 10.0,
                 max_execution_time: float = 60.0,
                 enable_preemption: bool = True,
                 preempt_strategy: str = "recompute",    # "recompute" or "swap"
                 reserved_blocks: int = 8,
                 aging_threshold: float = 5.0,
                 total_memory_blocks: int = 64):
        self.token_budget = token_budget
        self.max_num_seqs = max_num_seqs
        self.max_waiting_time = max_waiting_time
        self.max_execution_time = max_execution_time
        self.enable_preemption = enable_preemption
        self.preempt_strategy = preempt_strategy
        self.reserved_blocks = reserved_blocks
        self.aging_threshold = aging_threshold

        self.waiting: List[tuple] = []                      # heapq: (-priority, submit_time, req)
        self.running: Dict[int, ScheduledRequest] = {}      # req_id → req
        self.swapped: List[ScheduledRequest] = []           # 被抢占的请求
        self.memory = MemoryBudget(total_blocks=total_memory_blocks)
        self.time = 0.0
        self._tick = 0

    # ---- 请求提交 ----

    def submit(self, req: ScheduledRequest):
        req.original_priority = req.priority
        req.submit_time = self.time
        heapq.heappush(self.waiting, (-req.priority, req.submit_time, req))

    # ---- 调度核心 ----

    def schedule(self) -> List[ScheduledRequest]:
        """每轮 iteration 调用一次，返回本轮要执行的 batch。"""
        self._tick += 1
        self.time += 1.0
        batch: List[ScheduledRequest] = []
        remaining_tokens = self.token_budget

        # ① 处理 swapped 请求：优先恢复（如果有空间）
        self._restore_swapped(batch, remaining_tokens)

        # ② 继续 running 请求的 decode（每请求消耗 1 token budget）
        remaining_tokens = self._continue_running(batch, remaining_tokens)

        # ③ 从 waiting 加入新请求（prefill，消耗 prompt_len token budget）
        self._admit_new_requests(batch, remaining_tokens)

        # ④ 应用 aging（公平性）
        self._apply_aging()

        # ⑤ 检查超时
        self._check_timeouts()

        return batch

    def _restore_swapped(self, batch: List[ScheduledRequest], remaining_tokens: int) -> int:
        """尝试恢复被 swap 的请求。"""
        still_swapped = []
        for req in self.swapped:
            if (self.memory.can_allocate(req.required_kv_blocks)
                    and len(batch) < self.max_num_seqs
                    and remaining_tokens > 0):
                self.memory.allocate(req.required_kv_blocks)
                req.status = RequestStatus.RUNNING
                self.running[req.req_id] = req
                batch.append(req)
            else:
                still_swapped.append(req)
        self.swapped = still_swapped
        return remaining_tokens

    def _continue_running(self, batch: List[ScheduledRequest], remaining_tokens: int) -> int:
        """让 running 请求继续 decode（每请求消耗 1 token budget）。"""
        for req_id in list(self.running.keys()):
            req = self.running[req_id]

            if req.generated_tokens >= req.max_new_tokens:
                req.status = RequestStatus.FINISHED
                self.memory.free(req.required_kv_blocks)
                del self.running[req_id]
                continue

            if remaining_tokens >= 1 and len(batch) < self.max_num_seqs:
                req.generated_tokens += 1
                batch.append(req)
                remaining_tokens -= 1
            else:
                break

        return remaining_tokens

    def _admit_new_requests(self, batch: List[ScheduledRequest], remaining_tokens: int):
        """从 waiting 队列中按优先级加入新请求。"""
        still_waiting = []

        while self.waiting and remaining_tokens > 0 and len(batch) < self.max_num_seqs:
            neg_priority, submit_time, req = heapq.heappop(self.waiting)

            if self.time - submit_time > self.max_waiting_time:
                req.status = RequestStatus.TIMEOUT
                print(f"  [Timeout] Request {req.req_id} waited too long")
                continue

            if req.prompt_len > remaining_tokens:
                still_waiting.append((neg_priority, submit_time, req))
                continue

            available_blocks = self.memory.free_blocks
            if req.priority > 0:
                available_blocks += self.reserved_blocks

            if req.required_kv_blocks <= available_blocks:
                self._admit_request(req, batch)
                remaining_tokens -= req.prompt_len
            elif self.enable_preemption:
                victim = self._select_victim(req)
                if victim:
                    self._preempt(victim)
                    self._admit_request(req, batch)
                    remaining_tokens -= req.prompt_len
                else:
                    still_waiting.append((neg_priority, submit_time, req))
            else:
                still_waiting.append((neg_priority, submit_time, req))

        for item in still_waiting:
            heapq.heappush(self.waiting, item)

    def _admit_request(self, req: ScheduledRequest, batch: List[ScheduledRequest]):
        """将请求加入 running 并分配资源。"""
        self.memory.allocate(req.required_kv_blocks)
        req.status = RequestStatus.RUNNING
        req.start_time = self.time
        self.running[req.req_id] = req
        batch.append(req)

    # ---- 抢占 ----

    def _select_victim(self, new_req: ScheduledRequest) -> Optional[ScheduledRequest]:
        """选择被抢占的请求：最低优先级 → 最少剩余 token → 最晚提交。"""
        candidates = [r for r in self.running.values() if r.priority < new_req.priority]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (r.priority, r.max_new_tokens - r.generated_tokens, -r.start_time or 0))

    def _preempt(self, victim: ScheduledRequest):
        """抢占请求：recompute（丢弃 KV Cache）或 swap（换出到 CPU）。"""
        print(f"  [Preempt] Request {victim.req_id} (p={victim.priority}) preempted by strategy={self.preempt_strategy}")

        self.memory.free(victim.required_kv_blocks)
        del self.running[victim.req_id]

        if self.preempt_strategy == "swap":
            victim.swapped_data = f"kv_cache_of_req_{victim.req_id}"
            victim.status = RequestStatus.SWAPPED
            self.swapped.append(victim)
        else:
            victim.status = RequestStatus.WAITING
            victim.required_kv_blocks = 0
            victim.start_time = None
            heapq.heappush(self.waiting, (-victim.priority, victim.submit_time, victim))

    # ---- 公平性 ----

    def _apply_aging(self):
        """等待时间超过阈值的请求自动提升优先级，防止饥饿。"""
        updated = []
        while self.waiting:
            neg_priority, submit_time, req = heapq.heappop(self.waiting)
            wait_time = self.time - submit_time
            if wait_time > self.aging_threshold and req.priority < req.original_priority + 5:
                req.priority = req.original_priority + (int(wait_time // self.aging_threshold))
                print(f"  [Aging] Request {req.req_id} priority boosted to {req.priority}")
            updated.append((-req.priority, submit_time, req))
        for item in updated:
            heapq.heappush(self.waiting, item)

    # ---- 超时 ----

    def _check_timeouts(self):
        """检查 running 请求的执行超时。"""
        for req_id in list(self.running.keys()):
            req = self.running[req_id]
            if req.start_time and (self.time - req.start_time) > self.max_execution_time:
                print(f"  [ExecTimeout] Request {req.req_id} execution timed out")
                req.status = RequestStatus.TIMEOUT
                self.memory.free(req.required_kv_blocks)
                del self.running[req_id]

    # ---- 辅助 ----

    def finish_request(self, req_id: int):
        """外部标记请求完成。"""
        if req_id in self.running:
            self.running[req_id].status = RequestStatus.FINISHED
            self.running[req_id].generated_tokens = self.running[req_id].max_new_tokens

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "memory": str(self.memory),
            "tick": self._tick,
        }


# ============================================================
# Demo 主程序
# ============================================================

def main():
    scheduler = FullScheduler(
        token_budget=50,
        max_num_seqs=4,
        max_waiting_time=10.0,
        max_execution_time=60.0,
        enable_preemption=True,
        preempt_strategy="recompute",
        reserved_blocks=4,
        aging_threshold=4.0,
        total_memory_blocks=32,
    )

    print("=" * 60)
    print("FullScheduler Demo: 优先级 + 超时 + 资源预算 + 抢占")
    print("=" * 60)

    # 提交 8 个请求，混合优先级
    for i in range(8):
        req = ScheduledRequest(
            req_id=i,
            prompt_len=8 + i * 2,
            max_new_tokens=4 + i % 3,
            priority=i % 3,           # 0, 1, 2, 0, 1, 2, 0, 1
            required_kv_blocks=3 + i % 4,
        )
        scheduler.submit(req)
        print(f"Submitted R{i} (p={req.priority}, prompt={req.prompt_len}, kv_blocks={req.required_kv_blocks})")

    print()

    # 模拟 8 轮调度
    for tick in range(8):
        batch = scheduler.schedule()
        print(f"\n--- Tick {tick} | {scheduler.stats()} ---")
        print(f"  Batch: [{', '.join(f'R{r.req_id}(p={r.priority},gen={r.generated_tokens}/{r.max_new_tokens})' for r in batch)}]")

        # 模拟完成部分请求
        if tick == 2:
            scheduler.finish_request(0)
        if tick == 4:
            scheduler.finish_request(1)
            scheduler.finish_request(2)

    print("\n" + "=" * 60)
    print("Demo complete.")
    print(f"Final stats: {scheduler.stats()}")


if __name__ == "__main__":
    main()
