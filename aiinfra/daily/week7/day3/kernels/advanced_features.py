# advanced_features.py —— 高级特性模拟（Speculative Decoding + Chunked Prefill + Prefix Caching）
# 运行命令: python advanced_features.py
# 依赖: 仅标准库
#
# 本文件是 Week7 Day3 的核心产出：模拟三大高级推理特性，评估收益。
#   1. Speculative Decoding：小模型 draft + 大模型 verify，测量加速比
#   2. Chunked Prefill：长 prompt 分块，与 decode 交错，平滑延迟
#   3. Prefix Caching：缓存公共前缀的 KV Cache，降低 TTFT

import hashlib
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. Speculative Decoding 模拟
# ============================================================

@dataclass
class SpecDecodeResult:
    draft_tokens: List[int]
    accepted: int
    rejected: int
    time_draft: float
    time_verify: float
    time_traditional: float


def simulate_speculative_decoding(
    num_tokens: int = 100,
    draft_k: int = 4,
    accept_rate: float = 0.7,
    time_target_forward: float = 0.03,
    time_draft_forward: float = 0.005,
) -> SpecDecodeResult:
    """模拟 Speculative Decoding 过程。

    传统 decode：每步 1 次 target forward → 1 token
    Speculative decode：每步 k 次 draft forward + 1 次 target forward → 平均 k*α+1 tokens
    """
    random.seed(42)
    generated = 0
    total_draft = 0
    total_accepted = 0
    total_rejected = 0
    time_draft_total = 0.0
    time_verify_total = 0.0

    while generated < num_tokens:
        draft_tokens = list(range(draft_k))
        total_draft += draft_k

        accepted = 0
        for token in draft_tokens:
            if random.random() < accept_rate:
                accepted += 1
            else:
                break

        total_accepted += accepted
        total_rejected += (draft_k - accepted)

        time_draft_total += draft_k * time_draft_forward
        time_verify_total += time_target_forward

        generated += accepted + 1

    time_traditional = num_tokens * time_target_forward

    return SpecDecodeResult(
        draft_tokens=list(range(draft_k)),
        accepted=total_accepted,
        rejected=total_rejected,
        time_draft=time_draft_total,
        time_verify=time_verify_total,
        time_traditional=time_traditional,
    )


# ============================================================
# 2. Chunked Prefill 模拟
# ============================================================

@dataclass
class ChunkedPrefillResult:
    prompt_len: int
    chunk_size: int
    num_chunks: int
    decode_tokens_per_chunk: int
    total_decode_time_traditional: float
    total_decode_time_chunked: float
    max_latency_traditional: float
    max_latency_chunked: float


def simulate_chunked_prefill(
    prompt_len: int = 2048,
    chunk_size: int = 512,
    num_decode_requests: int = 3,
    decode_tokens: int = 10,
    time_per_token: float = 0.03,
) -> ChunkedPrefillResult:
    """模拟 Chunked Prefill 与传统 Prefill 的延迟对比。

    传统：长 prompt 一次性 prefill → 阻塞所有 decode 请求
    Chunked：prompt 分块，每块与 decode 交错 → 平滑 decode 延迟
    """
    num_chunks = (prompt_len + chunk_size - 1) // chunk_size

    # 传统：prefill 完整 prompt 阻塞 decode
    prefill_time_traditional = prompt_len * time_per_token
    decode_time_traditional = decode_tokens * time_per_token
    max_latency_traditional = prefill_time_traditional + decode_time_traditional

    # Chunked：每个 chunk 的 prefill 时间
    chunk_prefill_time = chunk_size * time_per_token
    # 每个 chunk 后可以服务 decode 请求
    decode_per_chunk = decode_tokens // num_chunks
    chunk_decode_time = decode_per_chunk * time_per_token

    # Chunked 模式下 decode 请求的最大等待时间
    max_latency_chunked = chunk_prefill_time + chunk_decode_time

    total_decode_traditional = prefill_time_traditional + num_decode_requests * decode_time_traditional
    total_decode_chunked = prompt_len * time_per_token + num_decode_requests * decode_time_traditional

    return ChunkedPrefillResult(
        prompt_len=prompt_len,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        decode_tokens_per_chunk=decode_per_chunk,
        total_decode_time_traditional=total_decode_traditional,
        total_decode_time_chunked=total_decode_chunked,
        max_latency_traditional=max_latency_traditional,
        max_latency_chunked=max_latency_chunked,
    )


# ============================================================
# 3. Prefix Caching 模拟
# ============================================================

class PrefixCache:
    """LRU 前缀缓存，模拟 vLLM/SGLang 的 Prefix Caching。

    缓存 key = prefix 的 hash，value = KV Cache 的模拟数据。
    命中时直接复用，跳过 prefill。
    """

    def __init__(self, max_entries: int = 128):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def _hash_prefix(self, tokens: List[int]) -> str:
        return hashlib.md5(str(tokens).encode()).hexdigest()

    def get(self, prefix_tokens: List[int]) -> Optional[dict]:
        key = self._hash_prefix(prefix_tokens)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, prefix_tokens: List[int], kv_cache: dict):
        key = self._hash_prefix(prefix_tokens)
        self._cache[key] = kv_cache
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def simulate_prefix_caching(
    num_requests: int = 100,
    vocab_size: int = 1000,
    system_prompt_len: int = 50,
    user_prompt_len: int = 20,
    cache_size: int = 64,
) -> dict:
    """模拟多轮对话场景下的 Prefix Caching 效果。

    场景：系统提示固定（50 token），用户输入变化（20 token）。
    有缓存时：系统提示的 KV Cache 直接复用，只 prefill 用户输入。
    无缓存时：每次都从头 prefill 整个 prompt。
    """
    random.seed(42)
    cache = PrefixCache(max_entries=cache_size)

    system_prompt = [random.randint(0, vocab_size) for _ in range(system_prompt_len)]

    time_per_token = 0.01
    total_time_no_cache = 0.0
    total_time_with_cache = 0.0

    for i in range(num_requests):
        user_prompt = [random.randint(0, vocab_size) for _ in range(user_prompt_len)]
        full_prompt = system_prompt + user_prompt

        # 无缓存：全量 prefill
        total_time_no_cache += len(full_prompt) * time_per_token

        # 有缓存：检查系统提示是否命中
        cached = cache.get(system_prompt)
        if cached:
            # 命中：只 prefill 用户输入部分
            total_time_with_cache += user_prompt_len * time_per_token
        else:
            # 未命中：全量 prefill + 缓存系统提示
            total_time_with_cache += len(full_prompt) * time_per_token
            cache.put(system_prompt, {"kv_cache": "simulated"})

    return {
        "num_requests": num_requests,
        "system_prompt_len": system_prompt_len,
        "user_prompt_len": user_prompt_len,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "hit_rate": cache.hit_rate,
        "total_time_no_cache": total_time_no_cache,
        "total_time_with_cache": total_time_with_cache,
        "speedup": total_time_no_cache / total_time_with_cache if total_time_with_cache > 0 else 0,
    }


# ============================================================
# 4. 特性收益评估
# ============================================================

def evaluate_features():
    """运行三大特性模拟，输出收益评估报告。"""
    print("=" * 70)
    print("高级特性收益评估报告")
    print("=" * 70)

    # --- Speculative Decoding ---
    print("\n📊 1. Speculative Decoding")
    print("-" * 50)
    for k in [2, 4, 8]:
        for alpha in [0.5, 0.7, 0.9]:
            result = simulate_speculative_decoding(
                num_tokens=100, draft_k=k, accept_rate=alpha,
                time_target_forward=0.03, time_draft_forward=0.005,
            )
            spec_time = result.time_draft + result.time_verify
            speedup = result.time_traditional / spec_time
            print(f"  k={k}, α={alpha:.1f}: "
                  f"traditional={result.time_traditional:.2f}s, "
                  f"spec={spec_time:.2f}s, "
                  f"speedup={speedup:.2f}x, "
                  f"accepted={result.accepted}, rejected={result.rejected}")

    # --- Chunked Prefill ---
    print("\n📊 2. Chunked Prefill")
    print("-" * 50)
    for prompt_len in [512, 2048, 8192]:
        for chunk_size in [256, 512, 1024]:
            result = simulate_chunked_prefill(
                prompt_len=prompt_len, chunk_size=chunk_size,
                num_decode_requests=3, decode_tokens=10,
                time_per_token=0.01,
            )
            latency_reduction = (
                (result.max_latency_traditional - result.max_latency_chunked)
                / result.max_latency_traditional * 100
            )
            print(f"  prompt={prompt_len}, chunk={chunk_size}: "
                  f"chunks={result.num_chunks}, "
                  f"max_latency: {result.max_latency_traditional:.2f}s → "
                  f"{result.max_latency_chunked:.2f}s "
                  f"(-{latency_reduction:.0f}%)")

    # --- Prefix Caching ---
    print("\n📊 3. Prefix Caching")
    print("-" * 50)
    for cache_size in [16, 64, 256]:
        result = simulate_prefix_caching(
            num_requests=100, vocab_size=1000,
            system_prompt_len=50, user_prompt_len=20,
            cache_size=cache_size,
        )
        print(f"  cache_size={cache_size}: "
              f"hits={result['cache_hits']}, misses={result['cache_misses']}, "
              f"hit_rate={result['hit_rate']:.1%}, "
              f"time: {result['total_time_no_cache']:.2f}s → "
              f"{result['total_time_with_cache']:.2f}s, "
              f"speedup={result['speedup']:.2f}x")

    # --- 集成优先级 ---
    print("\n📋 集成优先级建议")
    print("-" * 50)
    print("  1. Prefix Caching   — 收益高、复杂度中 → Phase 1 优先")
    print("  2. Chunked Prefill  — 平滑延迟、复杂度中 → Phase 1")
    print("  3. CUDA Graph       — 降 launch 开销、复杂度中 → Phase 2")
    print("  4. Speculative Decoding — 降 TBT、复杂度高 → Phase 2 可选")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate_features()
