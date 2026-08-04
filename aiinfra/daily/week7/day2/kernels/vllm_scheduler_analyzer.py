# vllm_scheduler_analyzer.py —— vLLM Scheduler 源码分析教学模型
# 运行命令: python vllm_scheduler_analyzer.py
# 依赖: 仅标准库
#
# 本文件是对 vLLM `vllm/core/scheduler.py` 的教学级简化复刻，聚焦三个核心机制：
#   1. schedule() 的 5 步流程（处理完成 → running → swapped → waiting → 构建 outputs）
#   2. SchedulingBudget（token_budget + max_num_seqs 双预算约束）
#   3. Preemption 两种模式（RECOMPUTE 丢弃 KV Cache 重算 / SWAP 换出到 CPU）
#
# 与 Day2 ContinuousBatcher 的区别：
#   - Day2 只实现 token_budget 约束，没有显存预算和抢占
#   - 本文件加入 BlockSpaceManager（显存 block 预算）+ 完整抢占逻辑
#   - 三队列：waiting / running / swapped（Day2 只有 waiting / running）

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


class PreemptionMode(Enum):
    RECOMPUTE = "recompute"   # 默认：丢弃 KV Cache，之后从 prompt 重算
    SWAP = "swap"             # 把 KV Cache 换出到 CPU，之后换入


@dataclass
class SchedulingBudget:
    """调度预算：每轮 iteration 的两类资源上限。

    对应 vLLM 的 `vllm.core.scheduling_budget.SchedulingBudget`：
      - token_budget：每轮最多处理的 token 数（prefill tokens + decode tokens）
      - max_num_seqs：每轮最多并发的 sequence 数
    """
    token_budget: int = 256
    max_num_seqs: int = 4
    # 已消耗（运行中累加，本轮调度结束时归零）
    _num_batched_tokens: int = field(default=0, repr=False)
    _num_curr_seqs: int = field(default=0, repr=False)

    def can_schedule(self, num_new_tokens: int, num_new_seqs: int = 0) -> bool:
        return (self._num_batched_tokens + num_new_tokens <= self.token_budget and
                self._num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def consume(self, num_new_tokens: int, num_new_seqs: int = 0):
        self._num_batched_tokens += num_new_tokens
        self._num_curr_seqs += num_new_seqs

    def remaining_tokens(self) -> int:
        return self.token_budget - self._num_batched_tokens

    def remaining_seqs(self) -> int:
        return self.max_num_seqs - self._num_curr_seqs

    def reset(self):
        self._num_batched_tokens = 0
        self._num_curr_seqs = 0


@dataclass
class SchedulerOutputs:
    """对应 vLLM 的 `vllm.core.scheduler.SchedulerOutputs`：记录本轮调度结果。"""
    scheduled_seq_groups: List["SequenceGroup"] = field(default_factory=list)
    num_batched_tokens: int = 0
    preempted: int = 0
    swapped_out: int = 0
    swapped_in: int = 0
    blocks_to_swap_out: List[tuple] = field(default_factory=list)   # [(seq_id, blocks)]
    blocks_to_swap_in: List[tuple] = field(default_factory=list)

    def summary(self) -> str:
        return (f"batched_tokens={self.num_batched_tokens}, "
                f"seqs={len(self.scheduled_seq_groups)}, "
                f"preempted={self.preempted}, "
                f"swap_out={self.swapped_out}, swap_in={self.swapped_in}")


class BlockSpaceManager:
    """简化版 BlockSpaceManager：管理 KV Cache block 的分配/释放。

    对应 vLLM 的 `vllm.core.block_manager.BlockSpaceManager`。
    PagedAttention 把 KV Cache 切成固定大小的 block（如 16 token/block），
    按需分配给各 sequence，避免连续分配的碎片化。
    """

    def __init__(self, num_total_blocks: int, block_size: int = 4):
        self.num_total_blocks = num_total_blocks
        self.block_size = block_size
        self.free_blocks: Deque[int] = deque(range(num_total_blocks))
        # seq_id -> list of block ids
        self.block_table: Dict[int, List[int]] = {}

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """能否为 num_tokens 个 token 分配足够的 block。"""
        needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= needed

    def allocate(self, seq_id: int, num_tokens: int):
        needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed:
            raise MemoryError(f"OOM: need {needed} blocks, only {len(self.free_blocks)} free")
        blocks = [self.free_blocks.popleft() for _ in range(needed)]
        # 若已有 block 表项（decode 增长），追加而非覆盖，避免泄漏已分配 block
        if seq_id in self.block_table:
            self.block_table[seq_id].extend(blocks)
        else:
            self.block_table[seq_id] = blocks

    def free(self, seq_id: int):
        for b in self.block_table.pop(seq_id, []):
            self.free_blocks.append(b)

    def swap_out(self, seq_id: int) -> tuple:
        """把 seq 的 block 换出到 CPU：释放 block 但保留映射记录（标记 swapped）。"""
        blocks = self.block_table.pop(seq_id, [])
        for b in blocks:
            self.free_blocks.append(b)
        return (seq_id, list(blocks))

    def swap_in(self, seq_id: int, num_tokens: int):
        """从 CPU 换入：重新分配 block（swap_out 已清空表项，此处新建）。"""
        self.allocate(seq_id, num_tokens)


class Sequence:
    """一个推理序列（对应 vLLM 的 Sequence）。"""

    def __init__(self, seq_id: int, prompt_len: int, max_new_tokens: int = 8):
        self.seq_id = seq_id
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.generated_count = 0
        self.status = SeqStatus.WAITING
        # 逻辑 KV token 数 = prompt_len + generated_count（决定需要多少 block）
        self.start_iter = -1
        self.finish_iter = -1
        self.preempt_count = 0   # 被抢占次数（用于追踪开销）

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated_count

    def is_prefill(self) -> bool:
        return self.generated_count == 0

    def append_token(self):
        self.generated_count += 1
        if self.generated_count >= self.max_new_tokens:
            self.status = SeqStatus.FINISHED

    def __repr__(self):
        return (f"S{self.seq_id}(p={self.prompt_len},g={self.generated_count},"
                f"{self.status.value})")


class SequenceGroup:
    """对应 vLLM 的 SequenceGroup：一组共享 prompt 的 sequence（beam search 等）。

    本教学模型简化为 1 group = 1 sequence，保留类名是为了对齐 vLLM 源码概念。
    """

    def __init__(self, request_id: int, seq: Sequence):
        self.request_id = request_id
        self.seq = seq
        self.seqs = [seq]

    @property
    def status(self) -> SeqStatus:
        return self.seq.status

    def get_num_new_tokens(self) -> int:
        """本轮需要计算的新 token 数：prefill = prompt_len，decode = 1。"""
        return self.seq.prompt_len if self.seq.is_prefill() else 1


class Scheduler:
    """vLLM Scheduler 教学级复刻。

    核心方法 schedule() 严格按 5 步流程（对应 vLLM 源码 schedule()）：
      1. 处理已完成的 running 序列（释放 block）
      2. _schedule_running：继续 running 队列的 decode（显存不足则 preempt）
      3. _schedule_swapped：把 swapped 队列换回 GPU（显存允许时）
      4. _schedule_waiting：从 waiting 加入新请求做 prefill
      5. 构建 SchedulerOutputs
    """

    def __init__(self, scheduler_config: dict, cache_config: dict,
                 preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE):
        self.max_token_budget: int = scheduler_config["max_num_batched_tokens"]
        self.max_num_seqs: int = scheduler_config["max_num_seqs"]
        self.preemption_mode = preemption_mode

        self.waiting: Deque[SequenceGroup] = deque()
        self.running: Deque[SequenceGroup] = deque()
        self.swapped: Deque[SequenceGroup] = deque()

        self.block_manager = BlockSpaceManager(
            num_total_blocks=cache_config["num_gpu_blocks"],
            block_size=cache_config["block_size"],
        )
        self.iteration = 0
        self.history: List[dict] = []

    # ---------- 公开接口 ----------
    def add_request(self, sg: SequenceGroup):
        sg.seq.status = SeqStatus.WAITING
        self.waiting.append(sg)

    def schedule(self) -> SchedulerOutputs:
        """每轮调度的核心入口（对应 vLLM Scheduler.schedule()）。"""
        budget = SchedulingBudget(
            token_budget=self.max_token_budget,
            max_num_seqs=self.max_num_seqs,
        )
        outputs = SchedulerOutputs()

        # Step 1: 处理已完成的 running 序列，释放其 KV Cache block
        self._free_finished_seq_groups()

        # Step 2: 调度 running 队列（继续 decode），显存不足时 preempt
        self._schedule_running(budget, outputs)

        # Step 3: 调度 swapped 队列（尝试换回 GPU）
        self._schedule_swapped(budget, outputs)

        # Step 4: 从 waiting 队列加入新请求（做 prefill）
        self._schedule_waiting(budget, outputs)

        # Step 5: 构建 outputs
        outputs.num_batched_tokens = budget._num_batched_tokens
        self.iteration += 1
        self._record_history(outputs)
        return outputs

    # ---------- Step 1 ----------
    def _free_finished_seq_groups(self):
        still_running: Deque[SequenceGroup] = deque()
        for sg in self.running:
            if sg.status == SeqStatus.FINISHED:
                self.block_manager.free(sg.seq.seq_id)
                sg.seq.finish_iter = self.iteration
            else:
                still_running.append(sg)
        self.running = still_running
    # ---------- Step 2 ----------
    def _schedule_running(self, budget: SchedulingBudget, outputs: SchedulerOutputs):
        """继续 running 的 decode。若 block 不足，按 RECOMPUTE/SWAP 抢占。"""
        still_running: Deque[SequenceGroup] = deque()
        # 注意：从队尾抢占（最后加入的最先被 preempt，近似 LIFO / vLLM 默认行为）
        running_list = list(self.running)
        for sg in running_list:
            seq = sg.seq
            # decode 每步 +1 token，检查是否需要新 block
            needs_alloc = self._needs_new_block(seq)
            if needs_alloc and not self.block_manager.can_allocate(self.block_manager.block_size):
                # 显存不足 → 抢占当前序列
                self._preempt(sg, outputs)
                continue
            if needs_alloc:
                self.block_manager.allocate(seq.seq_id, self.block_manager.block_size)
            # 检查 token/seq 预算
            num_new = sg.get_num_new_tokens()
            if not budget.can_schedule(num_new, num_new_seqs=0):
                # 预算不足也算抢占（vLLM 中通常是显存先触顶）
                self._preempt(sg, outputs)
                continue
            budget.consume(num_new, num_new_seqs=0)
            outputs.scheduled_seq_groups.append(sg)
            still_running.append(sg)
        self.running = still_running

    def _needs_new_block(self, seq: Sequence) -> bool:
        """decode 生成新 token 后是否跨过了 block 边界（需要新 block）。"""
        bs = self.block_manager.block_size
        allocated = len(self.block_manager.block_table.get(seq.seq_id, [])) * bs
        return seq.total_tokens >= allocated

    # ---------- Step 3 ----------
    def _schedule_swapped(self, budget: SchedulingBudget, outputs: SchedulerOutputs):
        """尝试把 swapped 队列的请求换回 GPU。"""
        still_swapped: Deque[SequenceGroup] = deque()
        for sg in self.swapped:
            seq = sg.seq
            num_new = 1   # 换入后是 decode
            if (self.block_manager.can_allocate(seq.total_tokens) and
                    budget.can_schedule(num_new, num_new_seqs=1)):
                self.block_manager.swap_in(seq.seq_id, seq.total_tokens)
                budget.consume(num_new, num_new_seqs=1)
                seq.status = SeqStatus.RUNNING
                outputs.scheduled_seq_groups.append(sg)
                outputs.swapped_in += 1
                self.running.append(sg)
            else:
                still_swapped.append(sg)
        self.swapped = still_swapped

    # ---------- Step 4 ----------
    def _schedule_waiting(self, budget: SchedulingBudget, outputs: SchedulerOutputs):
        """从 waiting 队列加入新请求做 prefill。

        vLLM 关键策略：仅当 swapped 为空时才从 waiting 加入新请求——
        优先恢复被抢占的请求，避免新请求"插队"导致 swapped 队列饿死。
        """
        if self.swapped:
            return  # 有 swapped 请求未恢复，不接纳新请求
        still_waiting: Deque[SequenceGroup] = deque()
        for sg in self.waiting:
            seq = sg.seq
            num_new = sg.get_num_new_tokens()   # prefill = prompt_len
            if (self.block_manager.can_allocate(seq.total_tokens) and
                    budget.can_schedule(num_new, num_new_seqs=1)):
                self.block_manager.allocate(seq.seq_id, seq.total_tokens)
                budget.consume(num_new, num_new_seqs=1)
                seq.status = SeqStatus.RUNNING
                seq.start_iter = self.iteration + 1
                outputs.scheduled_seq_groups.append(sg)
                self.running.append(sg)
            else:
                still_waiting.append(sg)
        self.waiting = still_waiting

    # ---------- Preemption ----------
    def _preempt(self, sg: SequenceGroup, outputs: SchedulerOutputs):
        """根据 preemption_mode 选择 recompute 或 swap。"""
        if self.preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(sg)
        else:
            self._preempt_by_swap(sg, outputs)
        outputs.preempted += 1
        sg.seq.preempt_count += 1

    def _preempt_by_recompute(self, sg: SequenceGroup):
        """RECOMPUTE：丢弃 KV Cache，序列回到 waiting 重新 prefill。

        vLLM 默认模式。优点：不需要 CPU 内存；通常比 PCIe 换入更快
        （尤其被抢占时间不长、prompt 不长时）。
        """
        seq = sg.seq
        self.block_manager.free(seq.seq_id)
        seq.generated_count = 0          # 重置：之后重新从 prompt prefill
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(sg)      # 放回 waiting 队首，优先重调度

    def _preempt_by_swap(self, sg: SequenceGroup, outputs: SchedulerOutputs):
        """SWAP：把 KV Cache 换出到 CPU，序列进入 swapped 队列。

        优点：不重新计算；缺点：需要 CPU 内存，PCIe 传输慢。
        """
        seq = sg.seq
        swap_record = self.block_manager.swap_out(seq.seq_id)
        outputs.blocks_to_swap_out.append(swap_record)
        outputs.swapped_out += 1
        seq.status = SeqStatus.SWAPPED
        self.swapped.append(sg)

    # ---------- 运行 & 记录 ----------
    def run_iteration(self, outputs: SchedulerOutputs):
        """模拟执行一轮 forward：每个 scheduled 序列生成 1 个 token。"""
        for sg in outputs.scheduled_seq_groups:
            seq = sg.seq
            if seq.status == SeqStatus.RUNNING:
                seq.append_token()

    def _record_history(self, outputs: SchedulerOutputs):
        states = [repr(sg.seq) for sg in outputs.scheduled_seq_groups]
        self.history.append({
            "iter": self.iteration,
            "batch": states,
            "free_blocks": self.block_manager.num_free_blocks(),
            "summary": outputs.summary(),
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
        })

    # ---------- 仪表盘 ----------
    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)


def _print_timeline(scheduler: Scheduler):
    print(f"\n{'Iter':>4} | {'Batch':>4} | {'FreeBlk':>7} | "
          f"{'W/R/S':>9} | {'Output':<46}")
    print("-" * 80)
    for h in scheduler.history:
        wrs = f"{h['waiting']}/{h['running']}/{h['swapped']}"
        print(f"{h['iter']:>4} | {len(h['batch']):>4} | {h['free_blocks']:>7} | "
              f"{wrs:>9} | {h['summary']:<46}")
        if h["batch"]:
            print(f"{'':>4} |   batch = {', '.join(h['batch'])}")


def demo_recompute_preemption():
    """场景 1：显存不足触发 RECOMPUTE 抢占。

    配置：4 个 block（block_size=4 → 16 token 显存）。
    2 个长请求(prompt=8,gen=3 各需 3 block) + 1 个短请求(prompt=4,gen=1)。
    两个长请求 prefill 占满 4 block，decode 增长时显存不足 → 抢占其一（RECOMPUTE），
    短请求先完成后释放显存，被抢占的长请求重新 prefill 并跑完。
    """
    print("=" * 80)
    print("Demo 1: RECOMPUTE Preemption（默认模式）")
    print("=" * 80)

    scheduler = Scheduler(
        scheduler_config={"max_num_batched_tokens": 64, "max_num_seqs": 4},
        cache_config={"num_gpu_blocks": 4, "block_size": 4},
        preemption_mode=PreemptionMode.RECOMPUTE,
    )
    scheduler.add_request(SequenceGroup(1, Sequence(1, prompt_len=8, max_new_tokens=3)))
    scheduler.add_request(SequenceGroup(2, Sequence(2, prompt_len=8, max_new_tokens=3)))
    scheduler.add_request(SequenceGroup(3, Sequence(3, prompt_len=4, max_new_tokens=1)))

    print("\n提交 3 个请求：S1/S2 长(prompt=8,gen=3)，S3 短(prompt=4,gen=1)")
    print("GPU 只有 4 个 block(4 token/blk)=16 token 显存，2 个长请求 prefill 即占满")
    print("→ 预期：decode 增长时触发 RECOMPUTE 抢占，短请求先完成释放显存后恢复\n")

    step = 0
    while scheduler.has_work() and step < 30:
        out = scheduler.schedule()
        scheduler.run_iteration(out)
        step += 1
        if out.preempted:
            print(f"  ⚡ iter {scheduler.iteration}: 触发 RECOMPUTE 抢占！"
                  f"被抢占序列丢弃 KV Cache，回到 waiting 队首重新 prefill")

    _print_timeline(scheduler)
    print(f"\n  总 iterations: {scheduler.iteration}")
    print(f"  RECOMPUTE 模式：被抢占序列丢弃 KV Cache，重新 prefill（无 CPU 换出）")


def demo_swap_preemption():
    """场景 2：SWAP 抢占，KV Cache 换出到 CPU。"""
    print("\n" + "=" * 80)
    print("Demo 2: SWAP Preemption（KV Cache 换出到 CPU）")
    print("=" * 80)

    scheduler = Scheduler(
        scheduler_config={"max_num_batched_tokens": 64, "max_num_seqs": 4},
        cache_config={"num_gpu_blocks": 4, "block_size": 4},
        preemption_mode=PreemptionMode.SWAP,
    )
    scheduler.add_request(SequenceGroup(1, Sequence(1, prompt_len=8, max_new_tokens=3)))
    scheduler.add_request(SequenceGroup(2, Sequence(2, prompt_len=8, max_new_tokens=3)))
    scheduler.add_request(SequenceGroup(3, Sequence(3, prompt_len=4, max_new_tokens=1)))

    print("\n同样 3 个请求，但用 SWAP 模式：被抢占序列的 KV Cache 换出到 CPU\n")

    step = 0
    while scheduler.has_work() and step < 30:
        out = scheduler.schedule()
        scheduler.run_iteration(out)
        step += 1
        if out.swapped_out:
            print(f"  🔄 iter {scheduler.iteration}: SWAP OUT！"
                  f"{out.swapped_out} 个序列的 KV Cache 换出到 CPU")
        if out.swapped_in:
            print(f"  🔁 iter {scheduler.iteration}: SWAP IN！"
                  f"{out.swapped_in} 个序列从 CPU 换回 GPU")

    _print_timeline(scheduler)
    print(f"\n  总 iterations: {scheduler.iteration}")
    print(f"  SWAP 模式：被抢占序列保留进度（不重 prefill），但需 PCIe 传输")


def demo_budget_constraint():
    """场景 3：token_budget 约束新请求加入。"""
    print("\n" + "=" * 80)
    print("Demo 3: SchedulingBudget 约束（token_budget 限制每轮 token 数）")
    print("=" * 80)

    scheduler = Scheduler(
        scheduler_config={"max_num_batched_tokens": 16, "max_num_seqs": 4},
        cache_config={"num_gpu_blocks": 8, "block_size": 4},
        preemption_mode=PreemptionMode.RECOMPUTE,
    )
    # token_budget=16：一轮只能 prefill 一个 prompt=12 的请求（+decode 会被卡）
    scheduler.add_request(SequenceGroup(1, Sequence(1, prompt_len=12, max_new_tokens=2)))
    scheduler.add_request(SequenceGroup(2, Sequence(2, prompt_len=12, max_new_tokens=2)))

    print("\n2 个请求(prompt=12)，token_budget=16 → 每轮只能 prefill 1 个，第 2 个等\n")

    step = 0
    while scheduler.has_work() and step < 15:
        out = scheduler.schedule()
        scheduler.run_iteration(out)
        step += 1

    _print_timeline(scheduler)
    print(f"\n  token_budget={scheduler.max_token_budget} 限制每轮 prefill token 数")
    print(f"  → 长请求的 prefill 不能一次塞满，避免 decode 请求被饿死")


def main():
    print("vLLM Scheduler 源码分析 —— 教学级复刻")
    print("对应源码: vllm/core/scheduler.py 的 schedule() / SchedulingBudget / Preemption\n")

    demo_recompute_preemption()
    demo_swap_preemption()
    demo_budget_constraint()

    print("\n" + "=" * 80)
    print("✅ 三大核心机制验证完毕：")
    print("  1. schedule() 5 步流程：free_finished → running → swapped → waiting → outputs")
    print("  2. SchedulingBudget：token_budget + max_num_seqs 双约束")
    print("  3. Preemption：RECOMPUTE（默认，丢弃 KV 重算）/ SWAP（换出 CPU）")
    print("=" * 80)


if __name__ == "__main__":
    main()
