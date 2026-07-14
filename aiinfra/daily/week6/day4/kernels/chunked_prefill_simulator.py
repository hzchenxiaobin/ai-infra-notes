# chunked_prefill_simulator.py —— Chunked Prefill vs Naive Prefill 延迟对比模拟
# 运行命令: python chunked_prefill_simulator.py
# 依赖: 仅标准库
#
# 本文件模拟两种 prefill 调度策略对 decode 延迟的影响：
#   1. Naive Prefill：长 prompt 一次性 prefill，整轮算力被 prefill 占满 → decode 请求被阻塞、latency 突增
#   2. Chunked Prefill：长 prompt 拆成多个小 chunk，每轮只 prefill 一个 chunk，剩余预算给 decode → latency 平滑
#
# 对应 TensorRT-LLM / vLLM(0.5+) 的 Chunked Prefill 机制：
#   - vLLM 默认把长 prompt 一次性 prefill（naive）
#   - vLLM 0.5+ 与 TensorRT-LLM 原生支持 chunked prefill（长 prompt 分块与 decode 交错）
#
# 与 Day2 ContinuousBatcher / Day3 Scheduler 的关系：
#   - Day2/Day3 的 _schedule_waiting 一次性 prefill 整个 prompt（消耗 prompt_len token budget）
#   - 本文件把 prefill 改成"每轮只消耗 chunk_size token"，多轮完成一个长 prefill

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


class SeqStatus:
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    """一个推理序列。prefill 进度用 prefilled_tokens 追踪（chunked 模式下分块累加）。"""
    seq_id: int
    prompt_len: int
    max_new_tokens: int = 6
    status: str = SeqStatus.WAITING
    prefilled_tokens: int = 0       # 已 prefill 的 prompt token 数（=prompt_len 时 prefill 完成）
    generated_count: int = 0
    start_iter: int = -1
    finish_iter: int = -1

    @property
    def is_prefill_done(self) -> bool:
        return self.prefilled_tokens >= self.prompt_len

    def prefill_chunk(self, chunk_size: int) -> int:
        """prefill 一块，返回实际处理的 token 数。"""
        remaining = self.prompt_len - self.prefilled_tokens
        n = min(chunk_size, remaining)
        self.prefilled_tokens += n
        return n

    def decode_step(self):
        """生成 1 个 token。"""
        self.generated_count += 1
        if self.generated_count >= self.max_new_tokens:
            self.status = SeqStatus.FINISHED


@dataclass
class IterRecord:
    """一轮 iteration 的记录，用于画延迟曲线。"""
    iter: int
    prefill_tokens: int          # 本轮 prefill 的 token 数（大 → decode 被挤压）
    decode_seqs: int             # 本轮 decode 的序列数
    decode_latencies: List[float] = field(default_factory=list)  # 本轮各 decode 序列的感知延迟
    events: List[str] = field(default_factory=list)              # 本轮发生的事件


class PrefillScheduler:
    """支持 naive / chunked 两种 prefill 策略的调度器。

    token_budget：每轮 iteration 最多处理的 token 数（prefill + decode 共享）。
    - naive 模式：新请求一次性 prefill 整个 prompt（消耗 prompt_len）
    - chunked 模式：新请求每轮只 prefill chunk_size 个 token，剩余预算给 decode
    """

    def __init__(self, max_token_budget: int = 32, max_num_seqs: int = 8,
                 chunk_size: int = 8, use_chunked: bool = False):
        self.max_token_budget = max_token_budget
        self.max_num_seqs = max_num_seqs
        self.chunk_size = chunk_size
        self.use_chunked = use_chunked
        self.waiting: Deque[Sequence] = deque()
        self.running: Dict[int, Sequence] = {}
        self.iteration = 0
        self.history: List[IterRecord] = []

    def submit(self, seq: Sequence):
        self.waiting.append(seq)

    def _decode_cost(self, num_decode: int) -> float:
        """模拟 decode 延迟：decode 越多每 token 越省（batch 摊销），但有下限。"""
        if num_decode == 0:
            return 0.0
        return 0.5 + 0.3 / num_decode   # 纯 decode 时 ~0.8ms/token；decode 越多越接近 0.5

    def _prefill_cost_per_token(self, prefill_tokens: int, num_decode: int) -> float:
        """模拟 prefill 对 decode 延迟的挤压：prefill tokens 越多，本轮 decode 越慢。

        naive 模式下 prefill_tokens 可能很大（=prompt_len）→ decode 延迟飙升
        chunked 模式下 prefill_tokens 被 chunk_size 封顶 → decode 延迟可控
        """
        base = 0.8
        # prefill 是 compute-bound，会抢占 SM 资源，decode 被拖慢
        pressure = 0.05 * prefill_tokens
        return base + pressure

    def schedule_iteration(self) -> IterRecord:
        budget = self.max_token_budget
        rec = IterRecord(iter=self.iteration + 1, prefill_tokens=0, decode_seqs=0)

        # 1. 移除已完成
        finished = [sid for sid, s in self.running.items() if s.status == SeqStatus.FINISHED]
        for sid in finished:
            self.running.pop(sid)

        # 2. 保留 running 的 decode（若 prefill 已完成）或继续 prefill chunk
        decode_batch: List[Sequence] = []
        for seq in self.running.values():
            if not seq.is_prefill_done:
                # chunked 模式下，未完成 prefill 的序列本轮继续 prefill 一个 chunk
                if self.use_chunked and budget >= self.chunk_size:
                    n = seq.prefill_chunk(self.chunk_size)
                    budget -= n
                    rec.prefill_tokens += n
                    rec.events.append(f"S{seq.seq_id} prefill +{n}({seq.prefilled_tokens}/{seq.prompt_len})")
                elif not self.use_chunked:
                    # naive 模式：prefill 应该早已一次性完成，不应到这里
                    pass
                # naive 模式且未完成：什么都不做（等一次性 prefill，见 step 3）
            else:
                # prefill 完成，本轮 decode 1 步
                if budget >= 1 and len(decode_batch) < self.max_num_seqs:
                    decode_batch.append(seq)
                    budget -= 1

        # 3. 从 waiting 加入新请求
        still_waiting: Deque[Sequence] = deque()
        for seq in self.waiting:
            if len(self.running) >= self.max_num_seqs:
                still_waiting.append(seq)
                continue
            if self.use_chunked:
                # chunked：本轮只 prefill 一个 chunk
                if budget >= self.chunk_size:
                    seq.status = SeqStatus.RUNNING
                    if seq.start_iter < 0:
                        seq.start_iter = self.iteration + 1
                    n = seq.prefill_chunk(self.chunk_size)
                    budget -= n
                    rec.prefill_tokens += n
                    self.running[seq.seq_id] = seq
                    rec.events.append(f"S{seq.seq_id} prefill +{n}({seq.prefilled_tokens}/{seq.prompt_len}) [new]")
                else:
                    still_waiting.append(seq)
            else:
                # naive：一次性 prefill 整个 prompt
                if budget >= seq.prompt_len:
                    seq.status = SeqStatus.RUNNING
                    seq.start_iter = self.iteration + 1
                    n = seq.prefill_chunk(seq.prompt_len)   # 一次到位
                    budget -= n
                    rec.prefill_tokens += n
                    self.running[seq.seq_id] = seq
                    rec.events.append(f"S{seq.seq_id} prefill ALL {n} [new]")
                else:
                    still_waiting.append(seq)
        self.waiting = still_waiting

        # 4. 执行 decode
        rec.decode_seqs = len(decode_batch)
        # 本轮每个 decode 序列的感知延迟 = prefill 挤压后的 per-token 成本
        per_token = self._prefill_cost_per_token(rec.prefill_tokens, len(decode_batch))
        decode_per_token = self._decode_cost(len(decode_batch))
        # 实际延迟：prefill 挤压 + decode 本身
        actual = per_token if rec.prefill_tokens > 0 else decode_per_token
        for seq in decode_batch:
            seq.decode_step()
            if seq.status == SeqStatus.FINISHED:
                seq.finish_iter = self.iteration + 1
            rec.decode_latencies.append(actual)

        self.iteration += 1
        self.history.append(rec)
        return rec

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)


def run_scenario(use_chunked: bool, label: str) -> List[IterRecord]:
    """跑一个场景：2 个短 decode 请求 + 1 个长 prompt 请求同时到达。

    短请求 S1/S2 正在 decode，此时 S3（prompt=24）到达。
    - naive：S3 一次性 prefill 24 token，整轮算力被占 → S1/S2 decode 延迟飙升
    - chunked：S3 分 3 轮 prefill（chunk=8），每轮 S1/S2 仍能 decode → 延迟平滑
    """
    print("=" * 78)
    print(f"场景：{label}")
    print("=" * 78)

    sched = PrefillScheduler(
        max_token_budget=32, max_num_seqs=8,
        chunk_size=8, use_chunked=use_chunked,
    )

    # S1/S2：短请求，已 prefill 完成，正在 decode
    s1 = Sequence(seq_id=1, prompt_len=4, max_new_tokens=6)
    s1.prefilled_tokens = 4   # 已 prefill
    s1.status = SeqStatus.RUNNING
    s1.start_iter = 0
    sched.running[1] = s1

    s2 = Sequence(seq_id=2, prompt_len=4, max_new_tokens=6)
    s2.prefilled_tokens = 4
    s2.status = SeqStatus.RUNNING
    s2.start_iter = 0
    sched.running[2] = s2

    # S3：长 prompt 请求，到达后开始 prefill
    s3 = Sequence(seq_id=3, prompt_len=24, max_new_tokens=4)
    sched.submit(s3)

    print("\n初始：S1/S2 正在 decode(gen=0/6)，S3 等待 prefill(prompt=24)\n")

    while sched.has_work() and sched.iteration < 25:
        rec = sched.schedule_iteration()
        lat_str = ", ".join(f"{l:.1f}" for l in rec.decode_latencies) or "-"
        ev_str = "; ".join(rec.events)
        print(f"  iter{rec.iter:>2} | prefill_tk={rec.prefill_tokens:>2} | "
              f"decode={rec.decode_seqs} | lat=[{lat_str}] | {ev_str}")

    print(f"\n  总 iterations: {sched.iteration}")
    print(f"  S1 finish_iter={s1.finish_iter}, S2 finish_iter={s2.finish_iter}, "
          f"S3 finish_iter={s3.finish_iter}")
    return sched.history


def print_latency_comparison(naive_hist: List[IterRecord], chunked_hist: List[IterRecord]):
    """对比两种策略下 decode 序列的延迟曲线。"""
    print("\n" + "=" * 78)
    print("Decode 延迟对比（S1/S2 的每轮感知延迟 ms/token）")
    print("=" * 78)

    print(f"\n{'iter':>4} | {'naive 延迟':>22} | {'chunked 延迟':>22} | {'差异':>8}")
    print("-" * 78)
    max_iter = max(len(naive_hist), len(chunked_hist))
    naive_max_spike = 0.0
    chunked_max_spike = 0.0
    for i in range(max_iter):
        n_lats = naive_hist[i].decode_latencies if i < len(naive_hist) else []
        c_lats = chunked_hist[i].decode_latencies if i < len(chunked_hist) else []
        n_str = ", ".join(f"{l:.1f}" for l in n_lats) or "(无decode)"
        c_str = ", ".join(f"{l:.1f}" for l in c_lats) or "(无decode)"
        n_max = max(n_lats) if n_lats else 0
        c_max = max(c_lats) if c_lats else 0
        naive_max_spike = max(naive_max_spike, n_max)
        chunked_max_spike = max(chunked_max_spike, c_max)
        diff = n_max - c_max if (n_lats and c_lats) else 0
        print(f"{i+1:>4} | {n_str:>22} | {c_str:>22} | {diff:>+7.1f}")

    print(f"\n  Naive   最大延迟尖峰: {naive_max_spike:.1f} ms/token")
    print(f"  Chunked 最大延迟尖峰: {chunked_max_spike:.1f} ms/token")
    print(f"  延迟尖峰降低: {naive_max_spike - chunked_max_spike:.1f} ms/token "
          f"({(1 - chunked_max_spike/naive_max_spike)*100:.0f}%)" if naive_max_spike > 0 else "")

    print(f"\n  Naive   总 iterations: {len(naive_hist)}")
    print(f"  Chunked 总 iterations: {len(chunked_hist)}")
    print("\n  结论：chunked prefill 把长 prompt 的 prefill 拆成小块与 decode 交错，")
    print("        decode 延迟尖峰大幅降低，TPOT（time-per-output-token）更稳定。")


def main():
    print("Chunked Prefill vs Naive Prefill —— 延迟对比模拟")
    print("对应：TensorRT-LLM Inflight Batching + Chunked Prefill / vLLM 0.5+ chunked prefill\n")

    naive_hist = run_scenario(use_chunked=False, label="Naive Prefill（一次性 prefill 整个 prompt）")
    chunked_hist = run_scenario(use_chunked=True, label="Chunked Prefill（prompt 分块与 decode 交错）")

    print_latency_comparison(naive_hist, chunked_hist)

    print("\n" + "=" * 78)
    print("✅ 核心机制验证完毕：")
    print("  1. Inflight/Continuous Batching：请求动态加入/退出，每轮重建 batch")
    print("  2. Chunked Prefill：长 prompt 拆 chunk 与 decode 交错，平滑 TPOT")
    print("  3. TensorRT-LLM C++ scheduler 原生支持；vLLM 0.5+ 也已支持")
    print("=" * 78)


if __name__ == "__main__":
    main()
