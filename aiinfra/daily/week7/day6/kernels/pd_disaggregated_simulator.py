# pd_disaggregated_simulator.py —— Prefill/Decode 分离推理模拟器
# 运行命令: python pd_disaggregated_simulator.py
# 依赖: 仅标准库（模拟调度，无需 GPU/PyTorch）
#
# 对比两种部署模式：
#   colocated: prefill + decode 共享同一 GPU 池（vLLM 默认）
#   disaggregated: prefill 池 + decode 池物理分离，KV Cache 跨节点传输
#
# 量化 TTFT / TPOT 改善，验证 PD 分离的核心收益：
#   - prefill 不被 decode 拖慢（compute-bound vs memory-bound 资源错配消除）
#   - decode 不被 prefill 阻塞（长 prompt 不再挤占 decode slot）
# 代价：KV Cache 跨节点传输开销（RDMA/NVLink）

import random
from dataclasses import dataclass, field
from typing import List


# ============================================================
# 参数（可调，用于扫描）
# ============================================================
@dataclass
class PDConfig:
    prefill_gpus: int = 3          # prefill 池 GPU 数
    decode_gpus: int = 5           # decode 池 GPU 数
    colocated_gpus: int = 8        # colocated 模式总 GPU 数（= prefill+decode）
    prefill_tput: float = 300.0    # tokens/s per GPU（prefill, compute-bound）
    decode_tput: float = 400.0     # tokens/s per GPU（decode, memory-bound）
    colocated_decode_penalty: float = 2.5  # colocated 下 decode 被 prefill 干扰的 TPOT 退化系数
    kv_bytes_per_token: int = 524_288  # LLaMA-7B MHA, 524 KB/token
    rdma_bw_gbs: float = 100.0     # RDMA 跨节点带宽（GB/s）
    avg_prompt_len: int = 512
    avg_decode_len: int = 64


@dataclass
class Request:
    req_id: int
    prompt_len: int
    decode_len: int
    arrival_time: float


@dataclass
class Metrics:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    kv_transfer_ms: float = 0.0


def gen_requests(n: int, cfg: PDConfig, seed: int = 42) -> List[Request]:
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        pl = int(rng.gauss(cfg.avg_prompt_len, cfg.avg_prompt_len * 0.3))
        dl = int(rng.gauss(cfg.avg_decode_len, cfg.avg_decode_len * 0.3))
        reqs.append(Request(i, max(16, pl), max(8, dl), i * 0.1))
    return reqs


# ============================================================
# Colocated 模式：prefill + decode 共享 GPU
# ============================================================
def simulate_colocated(reqs: List[Request], cfg: PDConfig) -> List[Metrics]:
    """colocated: prefill 与 decode 混跑，互相干扰。

    - prefill 大块占用 GPU → decode slot 空等 → TPOT 退化
    - decode 请求多 → prefill 排队 → TTFT 退化
    有效吞吐 = 总吞吐 × 0.6（资源争用效率损失 40%）
    """
    prefill_total = cfg.prefill_tput * cfg.colocated_gpus * 0.6
    decode_total = cfg.decode_tput * cfg.colocated_gpus * 0.6
    results = []
    queue_penalty = 0.0
    for i, r in enumerate(reqs):
        # 排队累积：前 10 个请求后开始有排队
        queue = max(0, (i - 10) * 0.05)
        prefill_s = r.prompt_len / prefill_total + queue
        active_seqs = max(1, min(8, i // 4 + 1))
        tpot_s = 1.0 / (decode_total / active_seqs) * cfg.colocated_decode_penalty
        decode_s = r.decode_len * tpot_s
        ttft_ms = prefill_s * 1000
        tpot_ms = tpot_s * 1000
        results.append(Metrics(
            ttft_ms=ttft_ms, tpot_ms=tpot_ms,
            e2e_ms=ttft_ms + decode_s * 1000,
            kv_transfer_ms=0.0,
        ))
    return results


# ============================================================
# Disaggregated 模式：prefill 池 + decode 池分离
# ============================================================
def simulate_disaggregated(reqs: List[Request], cfg: PDConfig) -> List[Metrics]:
    """prefill 池专用 prefill_tput，decode 池专用 decode_tput。
    KV Cache 跨节点传输：kv_bytes / rdam_bw。
    无 prefill 干扰，decode 池稳定吞吐。
    """
    prefill_total = cfg.prefill_tput * cfg.prefill_gpus
    decode_total = cfg.decode_tput * cfg.decode_gpus
    results = []
    for i, r in enumerate(reqs):
        prefill_s = r.prompt_len / prefill_total
        kv_bytes = r.prompt_len * cfg.kv_bytes_per_token
        kv_transfer_s = kv_bytes / (cfg.rdma_bw_gbs * 1e9)
        active_seqs = max(1, min(8, i // 4 + 1))  # 与 colocated 相同逻辑
        tpot_s = 1.0 / (decode_total / active_seqs)  # 无干扰，无退化系数
        decode_s = r.decode_len * tpot_s
        ttft_ms = prefill_s * 1000
        tpot_ms = tpot_s * 1000
        results.append(Metrics(
            ttft_ms=ttft_ms, tpot_ms=tpot_ms,
            e2e_ms=ttft_ms + kv_transfer_s * 1000 + decode_s * 1000,
            kv_transfer_ms=kv_transfer_s * 1000,
        ))
    return results


def summarize(name: str, results: List[Metrics]) -> dict:
    n = len(results)
    avg_ttft = sum(r.ttft_ms for r in results) / n
    avg_tpot = sum(r.tpot_ms for r in results) / n
    avg_e2e = sum(r.e2e_ms for r in results) / n
    avg_kv = sum(r.kv_transfer_ms for r in results) / n
    p99_ttft = sorted(r.ttft_ms for r in results)[int(n * 0.99) - 1]
    print(f"\n===== {name} =====")
    print(f"  avg TTFT : {avg_ttft:.1f} ms  (p99 {p99_ttft:.1f} ms)")
    print(f"  avg TPOT : {avg_tpot:.2f} ms")
    print(f"  avg E2E  : {avg_e2e:.1f} ms")
    if avg_kv > 0:
        print(f"  avg KV transfer : {avg_kv:.1f} ms")
    return {"ttft": avg_ttft, "tpot": avg_tpot, "e2e": avg_e2e, "kv": avg_kv}


def main():
    cfg = PDConfig()
    reqs = gen_requests(100, cfg)
    print("=" * 60)
    print("PD 分离推理模拟器（Prefill/Decode Disaggregated Serving）")
    print("=" * 60)
    print(f"配置: colocated={cfg.colocated_gpus} GPU | "
          f"disaggregated prefill={cfg.prefill_gpus} + decode={cfg.decode_gpus}")
    print(f"请求: {len(reqs)} 个, avg prompt={cfg.avg_prompt_len}, avg decode={cfg.avg_decode_len}")
    print(f"KV: {cfg.kv_bytes_per_token//1024} KB/token, RDMA={cfg.rdma_bw_gbs} GB/s")

    colo = simulate_colocated(reqs, cfg)
    dis = simulate_disaggregated(reqs, cfg)

    c = summarize("Colocated（prefill+decode 共享 GPU）", colo)
    d = summarize("Disaggregated（prefill/decode 池分离）", dis)

    print("\n===== 对比 =====")
    print(f"  TTFT 改善: {c['ttft']:.1f} → {d['ttft']:.1f} ms "
          f"({(1 - d['ttft']/c['ttft'])*100:.0f}% 降低)")
    print(f"  TPOT 改善: {c['tpot']:.2f} → {d['tpot']:.2f} ms "
          f"({(1 - d['tpot']/c['tpot'])*100:.0f}% 降低)")
    print(f"  E2E  改善: {c['e2e']:.1f} → {d['e2e']:.1f} ms "
          f"({(1 - d['e2e']/c['e2e'])*100:.0f}% 降低)")
    print(f"  KV 传输代价: {d['kv']:.1f} ms")

    print("\n===== 观察要点 =====")
    print("1. TTFT 降低: prefill 专用池不被 decode 拖慢（compute-bound 任务隔离）")
    print("2. TPOT 降低: decode 专用池不被长 prefill 阻塞（memory-bound 任务隔离）")
    print("3. KV 传输代价: disaggregated 多了跨节点 KV 搬运，长 prompt 时不可忽略")
    print("4. 何时划算: TTFT/TPOT 改善 > KV 传输开销；短 prompt 或低 QPS 时不划算")


if __name__ == "__main__":
    main()
