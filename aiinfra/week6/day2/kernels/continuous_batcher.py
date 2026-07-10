# continuous_batcher.py —— Continuous Batching 实现（iteration-level 调度，请求动态加入/退出）
# 运行命令: python continuous_batcher.py
# 依赖: 仅标准库
#
# 核心区别（vs Day1 Dynamic Batching）：
#   Dynamic Batching = request-level：一个 batch 一起开始一起结束
#   Continuous Batching = iteration-level：每轮重建 batch，完成即走、新请求随时插入

import time
import threading
from collections import deque
from enum import Enum
from typing import List, Dict


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    """一个推理序列（对应 vLLM 的 Sequence）"""
    def __init__(self, seq_id: int, prompt_len: int, max_new_tokens: int = 10):
        self.seq_id = seq_id
        self.prompt_len = prompt_len          # prefill 消耗的 token budget
        self.max_new_tokens = max_new_tokens
        self.generated_count = 0
        self.status = SeqStatus.WAITING
        self.arrival_time = time.time()
        self.start_iter = -1                   # 第几个 iteration 开始运行
        self.finish_iter = -1                  # 第几个 iteration 完成
        self.done_event = threading.Event()

    def is_prefill(self) -> bool:
        """是否需要 prefill（第一次运行）"""
        return self.generated_count == 0 and self.status == SeqStatus.RUNNING

    def append_token(self):
        """生成一个新 token"""
        self.generated_count += 1
        if self.generated_count >= self.max_new_tokens:
            self.status = SeqStatus.FINISHED
            self.done_event.set()

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated_count


class ContinuousBatcher:
    """
    Continuous Batcher：每轮 iteration 重新构建 batch

    与 Dynamic Batcher 的关键区别：
      - Dynamic: 凑满/超时 → 整批 forward → 整批完成 → 下一批
      - Continuous: 每轮 iteration → 保留 running + 从 waiting 补入 → forward 1 步 → 完成的退出
    """

    def __init__(self, max_token_budget: int = 50, max_num_seqs: int = 8):
        self.max_token_budget = max_token_budget
        self.max_num_seqs = max_num_seqs
        self.waiting_queue: deque[Sequence] = deque()
        self.running: Dict[int, Sequence] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.iteration = 0
        self.history = []   # 记录每轮 batch 状态，用于打印时间线

    def submit(self, seq: Sequence):
        """提交新请求到 waiting 队列"""
        with self.lock:
            self.waiting_queue.append(seq)

    def _schedule(self) -> List[Sequence]:
        """每轮调度：保留 running + 从 waiting 补入（token budget 约束）"""
        batch = []
        token_budget = self.max_token_budget

        with self.lock:
            # 1. 移除已完成的 running 序列
            finished_ids = [sid for sid, s in self.running.items() if s.status == SeqStatus.FINISHED]
            for sid in finished_ids:
                self.running.pop(sid)

            # 2. 保留正在运行的序列（decode 每步消耗 1 token budget）
            for seq in self.running.values():
                if token_budget >= 1 and len(batch) < self.max_num_seqs:
                    batch.append(seq)
                    token_budget -= 1

            # 3. 从 waiting 队列补入新请求（prefill 消耗 prompt_len token budget）
            still_waiting = deque()
            for seq in self.waiting_queue:
                cost = seq.prompt_len  # prefill cost
                if token_budget >= cost and len(batch) < self.max_num_seqs:
                    seq.status = SeqStatus.RUNNING
                    seq.start_iter = self.iteration + 1
                    self.running[seq.seq_id] = seq
                    batch.append(seq)
                    token_budget -= cost
                else:
                    still_waiting.append(seq)
            self.waiting_queue = still_waiting

        return batch

    def _run_iteration(self, batch: List[Sequence]):
        """运行一个 iteration：每个序列生成 1 个 token"""
        # 模拟 forward：batch 越大 per-token 越省
        forward_time = 0.002 + 0.0005 * len(batch)
        time.sleep(forward_time)

        for seq in batch:
            if seq.status == SeqStatus.RUNNING:
                seq.append_token()
                if seq.status == SeqStatus.FINISHED:
                    seq.finish_iter = self.iteration + 1

    def _worker_loop(self):
        while not self.stop_event.is_set():
            batch = self._schedule()
            if batch:
                self.iteration += 1
                self._run_iteration(batch)

                # 记录历史（用于打印时间线）
                states = []
                for seq in batch:
                    phase = "prefill" if seq.is_prefill() and seq.generated_count == 0 else "decode"
                    if seq.status == SeqStatus.FINISHED:
                        phase = "done"
                    states.append(f"S{seq.seq_id}({phase})")
                self.history.append({
                    "iter": self.iteration,
                    "batch_size": len(batch),
                    "states": states,
                    "finished": sum(1 for s in batch if s.status == SeqStatus.FINISHED),
                })
            else:
                time.sleep(0.001)

    def shutdown(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=3)


def main():
    print("=" * 70)
    print("Continuous Batcher Demo (iteration-level scheduling)")
    print("=" * 70)

    batcher = ContinuousBatcher(max_token_budget=20, max_num_seqs=4)

    sequences = []
    print("\nSubmitting 3 sequences with staggered arrival...\n")

    # S1: prompt=3, gen 4 tokens
    s1 = Sequence(seq_id=1, prompt_len=3, max_new_tokens=4)
    batcher.submit(s1)
    sequences.append(s1)
    print("  S1 submitted: prompt=3, gen=4")

    # S2: prompt=5, gen 8 tokens（更长，模拟长请求）
    s2 = Sequence(seq_id=2, prompt_len=5, max_new_tokens=8)
    batcher.submit(s2)
    sequences.append(s2)
    print("  S2 submitted: prompt=5, gen=8 (long request)")

    time.sleep(0.005)   # 等 S1/S2 开始 decode 后再提交 S3

    # S3: prompt=2, gen 3 tokens（短请求，应先于 S2 完成）
    s3 = Sequence(seq_id=3, prompt_len=2, max_new_tokens=3)
    batcher.submit(s3)
    sequences.append(s3)
    print("  S3 submitted: prompt=2, gen=3  (arrives during S1/S2 decode, short)")

    # 等待全部完成
    for s in sequences:
        s.done_event.wait()

    print(f"\n{'Iter':>5} {'BatchSize':>10} {'States':>40} {'Finished':>10}")
    print("-" * 70)
    for h in batcher.history:
        states_str = ", ".join(h["states"])
        print(f"{h['iter']:>5} {h['batch_size']:>10} {states_str:>40} {h['finished']:>10}")

    print(f"\nTotal iterations: {batcher.iteration}")
    print(f"\nCompletion summary:")
    for s in sequences:
        print(f"  S{s.seq_id}: start_iter={s.start_iter}, finish_iter={s.finish_iter}, "
              f"total_iters={s.finish_iter - s.start_iter + 1}, "
              f"prompt={s.prompt_len}, gen={s.generated_count}")

    # 对比 Dynamic Batching 的理论表现
    print(f"\n=== Continuous vs Dynamic Batching Comparison ===")
    print(f"  Continuous: S3 (短请求) 在 iter {s3.finish_iter} 完成，立即退出，不等待 S2")
    print(f"  Dynamic:    S1,S2,S3 同 batch 开始，S3 完成后要等 S2（到 iter {s2.finish_iter}）")
    s3_dynamic_finish = max(s1.finish_iter, s2.finish_iter, s3.finish_iter)
    print(f"  S3 等待节省: {s3_dynamic_finish - s3.finish_iter} iterations")

    batcher.shutdown()
    print("\n✅ Continuous Batcher demo done.")


if __name__ == "__main__":
    main()
