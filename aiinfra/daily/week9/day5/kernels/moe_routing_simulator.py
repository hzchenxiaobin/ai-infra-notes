# moe_routing_simulator.py —— MoE Top-K 路由 + EP all-to-all 通信量模拟
# 运行命令: python moe_routing_simulator.py
# 依赖: 仅标准库 + numpy（pip install numpy）
#
# 演示 MoE 推理两大核心机制：
#   1. Top-K 路由：softmax(gate(x)) → top-k → 分发到 k 个专家
#   2. EP all-to-all 通信量：推导 dispatch/combine 阶段的跨节点流量
#
# 不模拟真实前向，只验证路由正确性 + 通信量公式

import random
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("Warning: numpy not installed, using pure Python fallback")


# ============================================================
# MoE 配置
# ============================================================
@dataclass
class MoEConfig:
    num_experts: int = 8          # 总专家数
    top_k: int = 2                # 每个 token 选 k 个专家
    hidden_dim: int = 512         # token hidden dim
    expert_hidden: int = 512      # 专家中间层（FFN）
    num_tokens: int = 1024        # 输入 token 数
    ep_size: int = 4              # EP 并行度（节点数）
    dtype_bytes: int = 2          # fp16


# ============================================================
# Top-K 路由 kernel（纯 Python 模拟，验证逻辑）
# ============================================================
def gate_softmax_topk(tokens, num_experts, top_k, seed=42):
    """模拟 gate network: x @ W_gate → softmax → top-k。

    真实实现是 GEMM + softmax + top-k 融合 kernel（Triton/CUDA）。
    这里用 numpy 验证路由逻辑与负载均衡。
    """
    rng = random.Random(seed)
    n = len(tokens)
    # 模拟 gate logits（随机）
    logits = np.array([[rng.gauss(0, 1) for _ in range(num_experts)] for _ in range(n)]) if HAS_NP else \
             [[rng.gauss(0, 1) for _ in range(num_experts)] for _ in range(n)]
    # softmax
    if HAS_NP:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        # top-k
        topk_idx = np.argsort(-probs, axis=-1)[:, :top_k]
        topk_val = np.take_along_axis(probs, topk_idx, axis=-1)
    else:
        probs = []
        for row in logits:
            mx = max(row)
            exp = [pow(2.718, x - mx) for x in row]
            s = sum(exp)
            probs.append([e / s for e in exp])
        topk_idx = [sorted(range(len(p)), key=lambda i: -p[i])[:top_k] for p in probs]
        topk_val = [[p[i] for i in idx] for p, idx in zip(probs, topk_idx)]
    return topk_idx, topk_val


def analyze_load_balance(topk_idx, num_experts, num_tokens):
    """统计每个专家被选中的次数，分析负载均衡。"""
    counts = [0] * num_experts
    for idx_row in topk_idx:
        for e in idx_row:
            counts[e] += 1
    k_per_token = len(topk_idx[0]) if len(topk_idx) > 0 else 0
    total_assignments = num_tokens * k_per_token
    expected = total_assignments / num_experts
    max_dev = max(abs(c - expected) / expected for c in counts) if expected > 0 else 0
    print(f"\n  专家负载（期望/token: {expected:.1f}）:")
    for e, c in enumerate(counts):
        bar = "█" * int(c / max(counts) * 30) if max(counts) > 0 else ""
        print(f"    Expert {e}: {c:4d}  {bar}")
    print(f"  最大偏差: {max_dev*100:.1f}%  ({'均衡' if max_dev < 0.2 else '不均衡，需 aux-loss'})")
    return counts, max_dev


# ============================================================
# EP all-to-all 通信量推导
# ============================================================
def compute_ep_communication(cfg: MoEConfig):
    """推导 EP dispatch + combine 的 all-to-all 通信量。

    EP 并行：num_experts 个专家分布在 ep_size 个节点上。
    每个 token 选 top_k 个专家，这些专家可能跨节点。

    Dispatch 阶段（输入分发）:
      每个 token 发给 top_k 个专家，每个专家收到 token 的 hidden_dim 向量
      跨节点发送量 = num_tokens × top_k × hidden_dim × dtype_bytes
      （若所有 top_k 专家都在本地，则无跨节点流量；最坏情况全跨节点）

    Combine 阶段（输出回收）:
      每个专家输出 expert_hidden 维向量，发回原节点
      跨节点发送量 = num_tokens × top_k × expert_hidden × dtype_bytes
    """
    n, k, h, eh, ep, b = cfg.num_tokens, cfg.top_k, cfg.hidden_dim, cfg.expert_hidden, cfg.ep_size, cfg.dtype_bytes

    # 假设专家均匀分布在 ep 个节点，每个节点 n/ep 个 token
    tokens_per_node = n // ep
    experts_per_node = cfg.num_experts // ep

    # 最坏情况：每个 token 的 top_k 专家都不在本节点
    # Dispatch: 每个 token 发 k × h 向量给远程
    dispatch_bytes_per_token = k * h * b
    dispatch_total = n * dispatch_bytes_per_token

    # Combine: 每个专家返回 eh 维向量
    combine_bytes_per_token = k * eh * b
    combine_total = n * combine_bytes_per_token

    # 实际平均情况：top_k 中有 k × (1 - 1/ep) 个专家在远程（均匀分布假设）
    remote_ratio = 1 - 1 / ep
    dispatch_remote = dispatch_total * remote_ratio
    combine_remote = combine_total * remote_ratio

    return {
        "dispatch_per_token": dispatch_bytes_per_token,
        "combine_per_token": combine_bytes_per_token,
        "dispatch_total": dispatch_total,
        "combine_total": combine_total,
        "dispatch_remote": dispatch_remote,
        "combine_remote": combine_remote,
        "remote_ratio": remote_ratio,
    }


def main():
    cfg = MoEConfig()
    print("=" * 60)
    print("MoE Top-K 路由 + EP all-to-all 通信量模拟器")
    print("=" * 60)
    print(f"配置: {cfg.num_experts} 专家, top_k={cfg.top_k}, "
          f"{cfg.num_tokens} tokens, hidden={cfg.hidden_dim}, EP={cfg.ep_size}")

    # 1. Top-K 路由
    print("\n===== 1. Top-K 路由（gate softmax + top-k）=====")
    tokens = list(range(cfg.num_tokens))
    topk_idx, topk_val = gate_softmax_topk(tokens, cfg.num_experts, cfg.top_k)
    print(f"  前 5 个 token 的路由:")
    for i in range(5):
        idxs = topk_idx[i][:cfg.top_k] if HAS_NP else topk_idx[i]
        vals = topk_val[i][:cfg.top_k] if HAS_NP else topk_val[i]
        print(f"    token {i}: experts={idxs}, weights={[f'{v:.3f}' for v in vals]}")

    # 2. 负载均衡分析
    print("\n===== 2. 负载均衡分析 =====")
    counts, max_dev = analyze_load_balance(topk_idx, cfg.num_experts, cfg.num_tokens)

    # 3. EP 通信量
    print("\n===== 3. EP all-to-all 通信量推导 =====")
    comm = compute_ep_communication(cfg)
    print(f"  Dispatch（输入分发）:")
    print(f"    每 token: {comm['dispatch_per_token']//1024} KB (top_k × hidden × {cfg.dtype_bytes}B)")
    print(f"    总量(最坏): {comm['dispatch_total']/1e6:.1f} MB")
    print(f"    总量(远程,比例{comm['remote_ratio']:.2f}): {comm['dispatch_remote']/1e6:.1f} MB")
    print(f"  Combine（输出回收）:")
    print(f"    每 token: {comm['combine_per_token']//1024} KB (top_k × expert_hidden × {cfg.dtype_bytes}B)")
    print(f"    总量(最坏): {comm['combine_total']/1e6:.1f} MB")
    print(f"    总量(远程): {comm['combine_remote']/1e6:.1f} MB")
    print(f"  all-to-all 总跨节点流量: {(comm['dispatch_remote']+comm['combine_remote'])/1e6:.1f} MB")

    print("\n===== 4. EP vs TP 选择 =====")
    print(f"  EP{cfg.ep_size}: 专家切到 {cfg.ep_size} 节点, all-to-all 流量 "
          f"{(comm['dispatch_remote']+comm['combine_remote'])/1e6:.1f} MB")
    print(f"  TP{cfg.ep_size}: 权重切到 {cfg.ep_size} 节点, 每层 2 次 all-reduce")
    print(f"  → Decode 阶段 EP 优于 TP: decode batch 小, all-reduce 开销占比大")
    print(f"  → Prefill 阶段 TP 可能更优: batch 大, all-reduce 摊薄, 且无 all-to-all")

    print("\n===== 观察要点 =====")
    print("1. Top-K 路由: gate(x) → softmax → top-k, 每 token 选 k 个专家")
    print("2. 负载均衡: 无 aux-loss 时偏差可能 >20%, DeepSeek 用 aux-loss-free 策略")
    print(f"3. EP 通信量: ∝ num_tokens × top_k × hidden × (1 - 1/EP), EP 越大远程比例越高")
    print("4. EP vs TP: decode 选 EP(all-to-all 小), prefill 可能选 TP(all-reduce 摊薄)")


if __name__ == "__main__":
    main()
