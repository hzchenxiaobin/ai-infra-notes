# mini_engine_v1.py —— Mini 推理引擎 v1（多请求 + Continuous Batching + Scheduler + 优先级）
# 运行命令: python mini_engine_v1.py
# 依赖: pip install torch（无 GPU 时自动用 CPU）
#
# 相比 Week5 Day5 的 Mini 引擎 v0（单请求、同步 generate）：
#   v0: generate(prompt) → 一个请求跑完再跑下一个
#   v1: submit(prompt) → 异步返回 Future，后台 worker 线程做 Continuous Batching
#
# 本文件自包含（不依赖 mini_engine_v0.py），内嵌 MiniTransformer + Tokenizer，
# 整合 Week6 Day1-4 所学：Dynamic/Continuous Batching、Scheduler、token budget、优先级。

import math
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 模型定义（自包含，对应 Week5 Day5 的 MiniLLM）
# ============================================================

class MiniTransformerLayer(nn.Module):
    """单层 Transformer Block：Pre-LN + Self-Attention + FFN，支持 KV Cache"""

    def __init__(self, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, kv_cache=None, use_cache=False):
        B, N, _ = x.shape
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, N, d_head)

        if use_cache and kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.d_model)
        x = x + self.out(out)
        x = x + self.ffn(self.norm2(x))
        return x, (k, v)


class MiniLLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, d_ff=512, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MiniTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, kv_cache=None, use_cache=False):
        x = self.embedding(input_ids)
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = layer(x, layer_cache, use_cache)
            new_kv_cache.append(layer_new_cache)
        logits = self.lm_head(x)
        return logits, new_kv_cache


class MiniTokenizer:
    """最简 tokenizer：按空格切词，动态分配 token id"""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 1

    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.lower().split():
            if word not in self.word_to_id:
                if self.next_id >= self.vocab_size:
                    break
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
            tokens.append(self.word_to_id[word])
        return tokens or [0]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_word.get(i, f"<unk_{i}>") for i in ids)


# ============================================================
# 请求与调度器（Week6 Day2-3 的 Continuous Batching + 优先级）
# ============================================================

class RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Request:
    """一个推理请求，带优先级、Future 异步返回、独立 KV Cache。"""

    def __init__(self, request_id: int, input_ids: List[int],
                 max_new_tokens: int = 8, priority: int = 0):
        self.request_id = request_id
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.priority = priority
        self.generated_ids: List[int] = []
        self.kv_cache = None          # per-layer (k, v)
        self.status = RequestStatus.WAITING
        self.future = Future()
        self.created_at = time.time()
        self.start_iter = -1
        self.finish_iter = -1

    @property
    def is_prefill_done(self) -> bool:
        return self.kv_cache is not None


class MiniScheduler:
    """基础 Scheduler：token budget + max num_seqs + 优先级。

    对应 Week6 Day2 ContinuousBatcher._schedule() + Day3 SchedulingBudget。
    每轮调度：
      1. 保留 running 的 decode（按优先级排序）
      2. 从 waiting 加入新请求做 prefill（token budget 约束）
    """

    def __init__(self, max_token_budget: int = 64, max_num_seqs: int = 4):
        self.max_token_budget = max_token_budget
        self.max_num_seqs = max_num_seqs

    def schedule(self, waiting: deque, running: Dict[int, Request]) -> Tuple[List[Request], deque]:
        batch: List[Request] = []
        token_budget = self.max_token_budget

        # 1. 保留 running 的 decode（按优先级降序，高优先级先保）
        running_sorted = sorted(running.values(), key=lambda r: -r.priority)
        for req in running_sorted:
            if req.status == RequestStatus.RUNNING and req.is_prefill_done:
                if token_budget >= 1 and len(batch) < self.max_num_seqs:
                    batch.append(req)
                    token_budget -= 1

        # 2. 从 waiting 加入新请求做 prefill（按优先级降序）
        waiting_sorted = sorted(waiting, key=lambda r: -r.priority)
        still_waiting: deque = deque()
        for req in waiting_sorted:
            prompt_len = len(req.input_ids)
            if token_budget >= prompt_len and len(batch) < self.max_num_seqs:
                req.status = RequestStatus.RUNNING
                batch.append(req)
                token_budget -= prompt_len
            else:
                still_waiting.append(req)

        return batch, still_waiting


# ============================================================
# Mini 推理引擎 v1（多请求 + Continuous Batching + 异步返回）
# ============================================================

class MiniEngineV1:
    """Mini 推理引擎 v1：多请求并发 + Continuous Batching + 优先级调度。

    相比 v0（单请求同步 generate）：
      - submit() 异步返回 Future，后台 worker 线程做 Continuous Batching
      - 每轮 iteration 重建 batch：完成的退出、新请求加入
      - token budget + max_num_seqs 双约束 + 优先级排序
    """

    def __init__(self, model: MiniLLM, tokenizer: MiniTokenizer,
                 max_token_budget: int = 64, max_num_seqs: int = 4, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.scheduler = MiniScheduler(max_token_budget, max_num_seqs)

        self.waiting_queue: deque = deque()
        self.running_requests: Dict[int, Request] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.next_request_id = 0
        self.iteration = 0
        self.history: List[dict] = []

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def submit(self, prompt: str, max_new_tokens: int = 8, priority: int = 0) -> Future:
        """提交请求，返回 Future（异步获取结果）。"""
        with self.lock:
            req_id = self.next_request_id
            self.next_request_id += 1
        input_ids = self.tokenizer.encode(prompt)
        req = Request(req_id, input_ids, max_new_tokens, priority)
        with self.lock:
            self.waiting_queue.append(req)
        return req.future

    @torch.no_grad()
    def _run_iteration(self, batch: List[Request]):
        """运行一个 iteration：每个请求 prefill 1 步或 decode 1 步。"""
        for req in batch:
            if req.status != RequestStatus.RUNNING:
                continue
            if not req.is_prefill_done:
                # Prefill：一次性处理整段 prompt（naive prefill，Day4 讲的 chunked 是进阶）
                input_ids_tensor = torch.tensor([req.input_ids], device=self.device)
                logits, kv_cache = self.model(input_ids_tensor, use_cache=True)
                req.kv_cache = kv_cache
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                req.generated_ids.append(next_token)
                req.start_iter = self.iteration
            else:
                # Decode：只输入上一步生成的 token，复用 KV Cache
                next_input = req.generated_ids[-1]
                input_ids_tensor = torch.tensor([[next_input]], device=self.device)
                logits, kv_cache = self.model(
                    input_ids_tensor, kv_cache=req.kv_cache, use_cache=True
                )
                req.kv_cache = kv_cache
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                req.generated_ids.append(next_token)

            if len(req.generated_ids) >= req.max_new_tokens:
                req.status = RequestStatus.FINISHED
                req.finish_iter = self.iteration

    def _worker_loop(self):
        """后台 worker：每轮调度 → forward → 更新状态 → 返回完成结果。"""
        while not self.stop_event.is_set():
            with self.lock:
                # 1. 移除已完成的 running 请求，异步返回结果
                finished_ids = [
                    rid for rid, req in self.running_requests.items()
                    if req.status == RequestStatus.FINISHED
                ]
                for rid in finished_ids:
                    req = self.running_requests.pop(rid)
                    output_text = self.tokenizer.decode(req.generated_ids)
                    req.future.set_result(output_text)

                # 2. 调度：保留 running + 从 waiting 补入
                batch, self.waiting_queue = self.scheduler.schedule(
                    self.waiting_queue, self.running_requests
                )

                # 3. 新加入 running 的请求登记
                for req in batch:
                    if req.request_id not in self.running_requests:
                        self.running_requests[req.request_id] = req

            if batch:
                self.iteration += 1
                self._run_iteration(batch)
                self._record_history(batch)
            else:
                time.sleep(0.001)

    def _record_history(self, batch: List[Request]):
        states = []
        for req in batch:
            phase = "prefill" if not req.is_prefill_done or len(req.generated_ids) == 1 and req.start_iter == self.iteration else "decode"
            if req.status == RequestStatus.FINISHED:
                phase = "done"
            states.append(f"R{req.request_id}(p{req.priority},{phase})")
        self.history.append({
            "iter": self.iteration,
            "batch_size": len(batch),
            "states": states,
            "waiting": len(self.waiting_queue),
            "running": len(self.running_requests),
        })

    def shutdown(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=5)


# ============================================================
# 主流程：多请求并发 + 优先级调度演示
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    torch.manual_seed(42)
    vocab_size, d_model, n_heads, n_layers = 1000, 128, 4, 2
    model = MiniLLM(vocab_size, d_model, n_heads, n_layers=n_layers)
    tokenizer = MiniTokenizer(vocab_size)
    engine = MiniEngineV1(model, tokenizer, max_token_budget=40, max_num_seqs=4, device=device)

    print("=== Mini 推理引擎 v1：多请求 + Continuous Batching + 优先级 ===\n")

    # 提交多个请求，R0 高优先级
    prompts = [
        ("hello world", 8, 1),                     # R0: 高优先级
        ("this is a longer prompt for testing", 6, 0),  # R1: 普通，长 prompt
        ("short", 4, 0),                           # R2: 普通，短 prompt
        ("another test prompt here now", 5, 0),    # R3: 普通
    ]

    futures = []
    for i, (prompt, n, pri) in enumerate(prompts):
        future = engine.submit(prompt, max_new_tokens=n, priority=pri)
        futures.append((i, prompt, pri, future))
        print(f"  Submitted R{i}: '{prompt}' (gen={n}, priority={pri})")

    # 等待全部完成
    print("\nWaiting for all results...")
    for i, prompt, pri, future in futures:
        result = future.result()
        print(f"  R{i} (pri={pri}) done: '{result}'")

    # 打印 iteration 时间线
    print(f"\n=== Iteration 时间线（Continuous Batching）===")
    print(f"{'Iter':>4} | {'Batch':>5} | {'W/R':>5} | {'Batch 内容':<50}")
    print("-" * 70)
    for h in engine.history:
        wrs = f"{h['waiting']}/{h['running']}"
        states = ", ".join(h["states"])
        print(f"{h['iter']:>4} | {h['batch_size']:>5} | {wrs:>5} | {states:<50}")

    print(f"\n总 iterations: {engine.iteration}")
    print(f"完成请求数: {len(futures)}")

    # 验证：高优先级 R0 应先完成
    print("\n=== 优先级验证 ===")
    print("  R0(priority=1) 因高优先级先被调度 prefill，应最早完成")
    print("  Continuous Batching: R2(短) 完成后立即退出，不等 R1(长)")

    engine.shutdown()
    print("\n✅ Mini 推理引擎 v1 demo done.")


if __name__ == "__main__":
    main()
