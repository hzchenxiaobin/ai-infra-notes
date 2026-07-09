# mini_vllm_scheduler.py —— vLLM 核心架构的最小化模拟（LLMEngine + Scheduler + Worker）
# 运行命令: python mini_vllm_scheduler.py
# 依赖: 仅标准库（无需 torch / vllm）
#
# 演示三大核心机制：
#   1. 请求生命周期：WAITING → RUNNING → FINISHED（含 SWAPPED 抢占）
#   2. Continuous Batching：每轮 iteration 重新构建 batch，新请求随时加入
#   3. SchedulingBudget：token / num_seqs / 显存 三重预算约束

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ============================================================
# 数据模型（对应 vllm/sequence.py）
# ============================================================

class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"      # 被抢占，KV cache 换出到 CPU
    FINISHED = "FINISHED"


@dataclass
class Sequence:
    """单个序列（对应 vllm.Sequence）"""
    seq_id: int
    prompt_len: int          # prefill 的 token 数
    max_output_len: int      # 最多生成多少 token
    output_len: int = 0      # 已生成的 token 数
    status: SequenceStatus = SequenceStatus.WAITING
    kv_blocks: int = 0       # 当前占用的 KV cache block 数

    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED


@dataclass
class SequenceGroup:
    """一个请求对应一个 group（对应 vllm.SequenceGroup）
    实际 vLLM 中一个 group 可含多个采样序列（beam search / n>1），
    这里简化为单序列。"""
    request_id: int
    seq: Sequence
    arrival_iter: int        # 在第几个 iteration 到达


# ============================================================
# Scheduler（对应 vllm/engine/scheduler.py）
# ============================================================

@dataclass
class SchedulingBudget:
    """调度预算（对应 vllm.core.scheduling_budget.SchedulingBudget）"""
    max_num_seqs: int        # 本轮最多并行多少 sequence
    max_tokens: int          # 本轮最多处理多少 token（prefill+decode）
    max_blocks: int          # KV cache 剩余 block 数

    def can_add(self, seq: Sequence, block_size: int) -> bool:
        # 新 sequence 进 running 需要的 block 数（向上取整）
        need_blocks = (seq.total_len() + block_size - 1) // block_size
        return (self.num_seqs < self.max_num_seqs
                and self.tokens + seq.total_len() <= self.max_tokens
                and self.blocks + need_blocks <= self.max_blocks)

    def add(self, seq: Sequence, block_size: int):
        need_blocks = (seq.total_len() + block_size - 1) // block_size
        self.num_seqs += 1
        self.tokens += seq.total_len()
        self.blocks += need_blocks

    # 三个当前已用计数
    num_seqs: int = 0
    tokens: int = 0
    blocks: int = 0


@dataclass
class SchedulerOutputs:
    """scheduler 一轮的输出：本轮要运行哪些 sequence"""
    running_seqs: List[Sequence] = field(default_factory=list)
    preempted_seqs: List[Sequence] = field(default_factory=list)
    num_batched_tokens: int = 0


class Scheduler:
    """vLLM Scheduler 的核心逻辑（简化版）"""

    def __init__(self, block_size: int = 16, max_num_seqs: int = 4,
                 max_blocks: int = 64):
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_blocks = max_blocks        # 总 KV cache block 池
        self.used_blocks = 0                # 已分配 block 数

        self.waiting: List[SequenceGroup] = []     # WAITING 队列
        self.running: List[SequenceGroup] = []     # RUNNING 队列
        self.swapped: List[SequenceGroup] = []     # SWAPPED 队列（换出到 CPU）

    def add_request(self, sg: SequenceGroup):
        sg.seq.status = SequenceStatus.WAITING
        self.waiting.append(sg)

    def _alloc_blocks(self, seq: Sequence) -> int:
        """计算 seq 当前需要的 block 数"""
        return (seq.total_len() + self.block_size - 1) // self.block_size

    def _try_preempt(self) -> bool:
        """显存不足时，抢占最后加入的 running sequence（Recomputation 策略）"""
        if not self.running:
            return False
        # LIFO 抢占：弹出最后加入的
        victim = self.running.pop()
        victim.seq.status = SequenceStatus.WAITING
        victim.seq.output_len = 0          # recomputation：丢弃 KV cache
        self.used_blocks -= self._alloc_blocks(victim.seq)
        victim.seq.kv_blocks = 0
        self.waiting.insert(0, victim)     # 放回 waiting 队首，下次重新 prefill
        print(f"    ⚡ PREEMPT request {victim.request_id} "
              f"(recomputation, 释放 {self._alloc_blocks(victim.seq) if False else '?'} blocks)")
        return True

    def schedule(self) -> SchedulerOutputs:
        """一轮调度：决定本轮运行哪些 sequence（Continuous Batching 核心）"""
        out = SchedulerOutputs()

        # ---- Step 1: 保留所有 running（continuous batching 的基础）----
        running_seqs = [sg.seq for sg in self.running]
        out.running_seqs = list(running_seqs)

        # ---- Step 2: 从 waiting 中尽可能加入新请求 ----
        budget = SchedulingBudget(
            max_num_seqs=self.max_num_seqs,
            max_tokens=999999,                 # 简化：不限制 token 总数
            max_blocks=self.max_blocks - self.used_blocks,
        )
        for sg in running_seqs:
            budget.add(sg, self.block_size)

        still_waiting = []
        for sg in self.waiting:
            if budget.can_add(sg.seq, self.block_size):
                # 加入 running
                sg.seq.status = SequenceStatus.RUNNING
                need = self._alloc_blocks(sg.seq)
                self.used_blocks += need
                sg.seq.kv_blocks = need
                self.running.append(sg)
                out.running_seqs.append(sg.seq)
                budget.add(sg.seq, self.block_size)
                print(f"    + ADMIT  request {sg.request_id} "
                      f"(prefill {sg.seq.prompt_len} tok, alloc {need} blocks)")
            else:
                # 预算不足：尝试抢占
                if self._try_preempt():
                    # 抢占后重试
                    if budget.can_add(sg.seq, self.block_size):
                        sg.seq.status = SequenceStatus.RUNNING
                        need = self._alloc_blocks(sg.seq)
                        self.used_blocks += need
                        sg.seq.kv_blocks = need
                        self.running.append(sg)
                        out.running_seqs.append(sg.seq)
                        budget.add(sg.seq, self.block_size)
                        print(f"    + ADMIT  request {sg.request_id} "
                              f"(after preempt, alloc {need} blocks)")
                    else:
                        still_waiting.append(sg)
                else:
                    still_waiting.append(sg)
        self.waiting = still_waiting

        out.num_batched_tokens = sum(s.total_len() for s in out.running_seqs)
        return out


# ============================================================
# Worker（对应 vllm/worker/worker.py）
# ============================================================

class Worker:
    """执行模型前向（这里只模拟，不跑真模型）"""

    def execute_model(self, running_seqs: List[Sequence]) -> List[int]:
        """对每个 running sequence 执行一步：生成 1 个 token（decode）
        或完成 prefill。返回每个 seq 的新 token id。"""
        new_tokens = []
        for seq in running_seqs:
            # prefill 后第一个 token 由 prompt 末尾产出
            seq.output_len += 1
            tok = random.randint(0, 999)     # 假 token id
            new_tokens.append(tok)
        return new_tokens


# ============================================================
# LLMEngine（对应 vllm/engine/llm_engine.py）
# ============================================================

class LLMEngine:
    """vLLM 对外接口：管理整个推理生命周期"""

    def __init__(self, block_size: int = 16, max_num_seqs: int = 4,
                 max_blocks: int = 64):
        self.scheduler = Scheduler(block_size, max_num_seqs, max_blocks)
        self.worker = Worker()
        self.iteration = 0
        self.finished: List[SequenceGroup] = []

    def add_request(self, request_id: int, prompt_len: int,
                    max_output_len: int):
        sg = SequenceGroup(
            request_id=request_id,
            seq=Sequence(seq_id=request_id, prompt_len=prompt_len,
                         max_output_len=max_output_len),
            arrival_iter=self.iteration,
        )
        self.scheduler.add_request(sg)
        print(f"[iter {self.iteration}] ➕ add_request {request_id} "
              f"(prompt={prompt_len}, max_out={max_output_len})")

    def step(self) -> List[int]:
        """一轮推理：schedule → execute → 更新状态（对应 LLMEngine.step）"""
        self.iteration += 1
        print(f"\n[iter {self.iteration}] === step ===")

        # 1. Scheduler 决定本轮运行哪些 sequence
        sched_out = self.scheduler.schedule()
        if not sched_out.running_seqs:
            print("    (no running seqs)")
            return []

        print(f"    batch: {len(sched_out.running_seqs)} seqs, "
              f"{sched_out.num_batched_tokens} tokens, "
              f"used_blocks={self.scheduler.used_blocks}/{self.scheduler.max_blocks}")

        # 2. Worker 执行模型前向（每 seq 生成 1 个 token）
        new_tokens = self.worker.execute_model(sched_out.running_seqs)

        # 3. 更新 sequence 状态
        finished_this_step = []
        for sg in self.scheduler.running[:]:
            seq = sg.seq
            # 检查是否完成
            if seq.output_len >= seq.max_output_len:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.used_blocks -= seq.kv_blocks
                seq.kv_blocks = 0
                self.scheduler.running.remove(sg)
                self.finished.append(sg)
                finished_this_step.append(sg.request_id)
                print(f"    ✔ FINISH request {sg.request_id} "
                      f"(generated {seq.output_len} tokens, free {seq.total_len()//self.scheduler.block_size + 1} blocks)")
        return finished_this_step

    def has_unfinished(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)


# ============================================================
# 主流程：模拟 3 个请求交错到达，演示 Continuous Batching
# ============================================================

def main():
    random.seed(42)
    # block_size=16, 最多 4 并发, 总 64 blocks
    # （减小 max_blocks 可触发抢占，见扩展实验）
    engine = LLMEngine(block_size=16, max_num_seqs=4, max_blocks=64)

    print("=" * 60)
    print("Mini vLLM Scheduler Simulation")
    print("=" * 60)

    # iter 0: 请求 0 到达（长 prompt + 长输出）
    engine.add_request(0, prompt_len=32, max_output_len=8)

    # 跑几步
    for _ in range(2):
        engine.step()

    # iter 2: 请求 1、2 到达（制造 interleaving）
    engine.add_request(1, prompt_len=16, max_output_len=5)
    engine.add_request(2, prompt_len=48, max_output_len=6)

    # 继续跑到全部完成
    while engine.has_unfinished():
        engine.step()

    print("\n" + "=" * 60)
    print(f"All requests finished. Total iterations: {engine.iteration}")
    print(f"Finished order: {[sg.request_id for sg in engine.finished]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
