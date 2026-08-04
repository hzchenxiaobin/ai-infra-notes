"""prefix_cache_engine.py —— 带 Prefix Caching 的简化推理引擎

实现：
- BlockPool: 管理 KV cache blocks (block_size=16)
- PrefixCache: 基于 OrderedDict 的 LRU 缓存，block hash -> physical block
- PrefixCacheEngine: 调度器，每个请求先做 prefix 匹配，剩余部分 prefill

运行: python3 prefix_cache_engine.py
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

BLOCK_SIZE = 16
PREFILL_LATENCY_PER_TOKEN = 0.1  # ms (模拟)


@dataclass
class KVBlock:
    block_id: int
    token_ids: List[int] = field(default_factory=list)
    ref_count: int = 0


class BlockPool:
    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        self.blocks: List[KVBlock] = [KVBlock(i) for i in range(max_blocks)]
        self.free_list: List[int] = list(range(max_blocks))

    def allocate(self) -> int:
        if not self.free_list:
            raise RuntimeError("BlockPool exhausted")
        bid = self.free_list.pop(0)
        self.blocks[bid].token_ids = []
        self.blocks[bid].ref_count = 1
        return bid

    def free(self, bid: int):
        self.blocks[bid].ref_count -= 1
        if self.blocks[bid].ref_count <= 0:
            self.blocks[bid].ref_count = 0
            self.blocks[bid].token_ids = []
            self.free_list.append(bid)

    def incref(self, bid: int):
        self.blocks[bid].ref_count += 1


class PrefixCache:
    def __init__(self, pool: BlockPool, max_cached: int):
        self.pool = pool
        self.max_cached = max_cached
        self.cache: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _hash(tokens: List[int]) -> str:
        return hashlib.md5(bytes(tokens)).hexdigest()

    def get(self, tokens: List[int]) -> Optional[int]:
        h = self._hash(tokens)
        if h in self.cache:
            self.cache.move_to_end(h)
            bid = self.cache[h]
            self.pool.incref(bid)
            return bid
        return None

    def put(self, tokens: List[int], bid: int):
        h = self._hash(tokens)
        if h not in self.cache:
            self.cache[h] = bid
            self.pool.incref(bid)
            if len(self.cache) > self.max_cached:
                old_h, old_bid = self.cache.popitem(last=False)
                self.pool.free(old_bid)
        else:
            self.cache.move_to_end(h)


@dataclass
class Request:
    req_id: int
    token_ids: List[int]
    matched_blocks: List[int] = field(default_factory=list)
    prefill_done: int = 0  # 已 prefill 的 token 数

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def remaining_prefill(self) -> int:
        return self.total_tokens - self.prefill_done


class PrefixCacheEngine:
    def __init__(self, max_blocks: int = 256, max_cached: int = 128):
        self.pool = BlockPool(max_blocks)
        self.cache = PrefixCache(self.pool, max_cached)
        self.next_req_id = 0

    def submit(self, token_ids: List[int]) -> Request:
        req = Request(req_id=self.next_req_id, token_ids=list(token_ids))
        self.next_req_id += 1
        self._match_prefix(req)
        return req

    def _match_prefix(self, req: Request):
        tokens = req.token_ids
        n_blocks = len(tokens) // BLOCK_SIZE
        matched = 0
        for i in range(n_blocks):
            block_tokens = tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            bid = self.cache.get(block_tokens)
            if bid is not None:
                req.matched_blocks.append(bid)
                matched += BLOCK_SIZE
            else:
                break
        req.prefill_done = matched

    def prefill(self, req: Request) -> float:
        remaining = req.remaining_prefill
        if remaining <= 0:
            return 0.0
        latency = remaining * PREFILL_LATENCY_PER_TOKEN
        new_blocks_start = req.prefill_done
        for i in range(new_blocks_start, len(req.token_ids), BLOCK_SIZE):
            block_tokens = req.token_ids[i : i + BLOCK_SIZE]
            if len(block_tokens) < BLOCK_SIZE:
                req.prefill_done = len(req.token_ids)
                break
            bid = self.pool.allocate()
            self.pool.blocks[bid].token_ids = list(block_tokens)
            req.matched_blocks.append(bid)
            self.cache.put(block_tokens, bid)
            req.prefill_done = i + BLOCK_SIZE
        if req.prefill_done < len(req.token_ids):
            req.prefill_done = len(req.token_ids)
        return latency

    def release(self, req: Request):
        for bid in req.matched_blocks:
            self.pool.free(bid)
        req.matched_blocks.clear()


def run_scenario_1_no_cache():
    print("\n--- Scenario 1: No prefix caching ---")
    engine = PrefixCacheEngine(max_blocks=256, max_cached=0)
    system_prompt = list(range(64))
    user_input = list(range(64, 192))
    full_tokens = system_prompt + user_input
    total = 0.0
    for i in range(3):
        req = engine.submit(full_tokens)
        latency = engine.prefill(req)
        print(f"  Request {i+1}: prefill {req.remaining_prefill + req.prefill_done} tokens, "
              f"latency = {latency:.1f} ms")
        total += latency
        engine.release(req)
    print(f"  Total: {total:.1f} ms")
    return total


def run_scenario_2_with_cache():
    print("\n--- Scenario 2: With prefix caching ---")
    engine = PrefixCacheEngine(max_blocks=256, max_cached=128)
    system_prompt = list(range(64))
    user_input = list(range(64, 192))
    full_tokens = system_prompt + user_input
    total = 0.0
    for i in range(3):
        req = engine.submit(full_tokens)
        prefill_before = req.prefill_done
        latency = engine.prefill(req)
        remaining = req.remaining_prefill + (req.total_tokens - prefill_before - req.remaining_prefill)
        status = f"{req.total_tokens - prefill_before} tokens ({prefill_before} prefix hit)" if prefill_before > 0 else f"{req.total_tokens} tokens (cache miss)"
        print(f"  Request {i+1}: prefill {status}, latency = {latency:.1f} ms")
        total += latency
        engine.release(req)
    print(f"  Total: {total:.1f} ms")
    return total


def run_scenario_3_multi_turn():
    print("\n--- Scenario 3: Multi-turn dialogue ---")
    engine = PrefixCacheEngine(max_blocks=256, max_cached=128)
    system_prompt = list(range(32))
    turns = [
        list(range(32, 40)),
        list(range(40, 48)),
        list(range(48, 56)),
    ]
    total_cached = 0.0
    total_nocache = 0.0
    accumulated = list(system_prompt)
    for i, turn in enumerate(turns):
        accumulated += turn
        req = engine.submit(accumulated)
        prefill_before = req.prefill_done
        latency = engine.prefill(req)
        total_cached += latency
        total_nocache += len(accumulated) * PREFILL_LATENCY_PER_TOKEN
        hit_info = f"{prefill_before} prefix hit" if prefill_before > 0 else "cache miss"
        print(f"  Turn {i+1}: prefill {len(accumulated) - prefill_before} tokens ({hit_info}), "
              f"latency = {latency:.1f} ms")
        engine.release(req)
    print(f"  Total with cache: {total_cached:.1f} ms")
    print(f"  Total without cache: {total_nocache:.1f} ms")
    print(f"  Speedup: {total_nocache / total_cached:.2f}x")
    return total_cached


def main():
    print("=== Prefix Caching Engine Demo ===")
    print(f"System prompt: 64 tokens, User input: 128 tokens, BLOCK_SIZE={BLOCK_SIZE}")

    t1 = run_scenario_1_no_cache()
    t2 = run_scenario_2_with_cache()
    print(f"\n  Speedup (scenario 2 vs 1): {t1 / t2:.2f}x")

    run_scenario_3_multi_turn()


if __name__ == "__main__":
    main()
