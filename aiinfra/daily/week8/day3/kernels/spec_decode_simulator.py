# spec_decode_simulator.py —— 投机解码模拟器（draft + verify Monte Carlo + 接受率理论扫描）
# 运行命令: python spec_decode_simulator.py
# 依赖: 仅标准库
#
# 本文件是 Week8 Day3 的核心产出：
#   1. Monte Carlo 模拟：draft k 个 token，大模型顺序验证（第一个拒绝即停），实测加速比
#   2. 理论扫描：每步产出 token 的精确期望 (1-α^(k+1))/(1-α) vs 近似上界 k·α+1，
#      k×α 网格扫描（周里程碑"投机解码接受率扫描数据留档"）
#   3. 失效边界：α 低 + k 大时 draft 开销超过收益，实测变慢
#
# Chunked Prefill / Prefix Caching 的模拟器见 Week 7：
#   ../../week7/day3/kernels/chunked_prefill_simulator.py
#   ../../week7/day4/kernels/prefix_cache_engine.py

import random
from dataclasses import dataclass


# ============================================================
# 1. Monte Carlo 模拟
# ============================================================

@dataclass
class SpecDecodeResult:
    num_steps: int
    accepted: int
    rejected: int
    time_draft: float
    time_verify: float
    time_traditional: float

    @property
    def time_spec(self) -> float:
        return self.time_draft + self.time_verify

    @property
    def speedup(self) -> float:
        return self.time_traditional / self.time_spec if self.time_spec > 0 else 0.0

    @property
    def tokens_per_step(self) -> float:
        return (self.accepted + self.num_steps) / self.num_steps if self.num_steps else 0.0


def simulate_speculative_decoding(
    num_tokens: int = 100,
    draft_k: int = 4,
    accept_rate: float = 0.7,
    time_target_forward: float = 0.03,
    time_draft_forward: float = 0.005,
    seed: int = 42,
) -> SpecDecodeResult:
    """模拟 Speculative Decoding 过程。

    传统 decode：每步 1 次 target forward → 1 token
    Speculative decode：每步 k 次 draft forward + 1 次 target verify
      → 每步产出 accepted + 1 个 token（第一个拒绝即停止，verify 补 1 个修正 token）

    验证规则与真实一致：draft token 顺序验证，遇到第一个拒绝就停止本步
    （这正是 k·α+1 是"近似上界"、精确期望要用等比级数求和的原因）。
    """
    rng = random.Random(seed)
    generated = 0
    num_steps = 0
    total_accepted = 0
    total_rejected = 0
    time_draft_total = 0.0
    time_verify_total = 0.0

    while generated < num_tokens:
        num_steps += 1

        accepted = 0
        for _ in range(draft_k):
            if rng.random() < accept_rate:
                accepted += 1
            else:
                break

        total_accepted += accepted
        total_rejected += draft_k - accepted
        time_draft_total += draft_k * time_draft_forward
        time_verify_total += time_target_forward
        generated += accepted + 1

    time_traditional = num_tokens * time_target_forward

    return SpecDecodeResult(
        num_steps=num_steps,
        accepted=total_accepted,
        rejected=total_rejected,
        time_draft=time_draft_total,
        time_verify=time_verify_total,
        time_traditional=time_traditional,
    )


# ============================================================
# 2. 理论分析：精确期望 vs 近似上界
# ============================================================

def expected_tokens_per_step(draft_k: int, accept_rate: float) -> float:
    """每步产出 token 数的精确期望（等比级数求和）。

    E[tokens/step] = 1 + α + α² + ... + α^k = (1 - α^(k+1)) / (1 - α)
    （前 k 项是 draft 被接受个数的期望，最后 1 是 verify 阶段补的修正 token）
    """
    if accept_rate >= 1.0:
        return float(draft_k + 1)
    return (1 - accept_rate ** (draft_k + 1)) / (1 - accept_rate)


def approx_tokens_per_step(draft_k: int, accept_rate: float) -> float:
    """近似上界：k·α+1。假设 k 个 draft token 各自独立以概率 α 被接受，
    忽略了顺序停止规则，因此恒 >= 精确期望。"""
    return draft_k * accept_rate + 1


def theoretical_speedup(
    draft_k: int,
    accept_rate: float,
    time_target_forward: float = 0.03,
    time_draft_forward: float = 0.005,
) -> float:
    """理论加速比 = E[tokens/step] × T_fwd / (k·t_d + T_fwd)。"""
    denom = draft_k * time_draft_forward + time_target_forward
    return expected_tokens_per_step(draft_k, accept_rate) * time_target_forward / denom


# ============================================================
# 3. 收益评估报告
# ============================================================

def evaluate_monte_carlo():
    print("\n📊 1. Monte Carlo 模拟（num_tokens=100, t_d=0.005s, T_fwd=0.03s）")
    print("-" * 60)
    print(f"  {'k':>2} {'α':>4} {'传统':>7} {'spec':>7} {'加速比':>7} {'接受':>5} {'拒绝':>5}")
    for k in [2, 4, 8]:
        for alpha in [0.5, 0.7, 0.9]:
            r = simulate_speculative_decoding(
                num_tokens=100, draft_k=k, accept_rate=alpha,
                time_target_forward=0.03, time_draft_forward=0.005,
            )
            print(f"  {k:>2} {alpha:>4.1f} {r.time_traditional:>6.2f}s {r.time_spec:>6.2f}s "
                  f"{r.speedup:>6.2f}x {r.accepted:>5} {r.rejected:>5}")


def evaluate_theory_vs_simulation():
    print("\n📊 2. 理论 vs 模拟：k·α+1 是上界，不是期望（k=4, α=0.7 展开）")
    print("-" * 60)
    for k, alpha in [(4, 0.7), (8, 0.5), (8, 0.9)]:
        exact = expected_tokens_per_step(k, alpha)
        approx = approx_tokens_per_step(k, alpha)
        speed_exact = theoretical_speedup(k, alpha)
        r = simulate_speculative_decoding(
            num_tokens=100, draft_k=k, accept_rate=alpha,
            time_target_forward=0.03, time_draft_forward=0.005,
        )
        print(f"  k={k}, α={alpha}: E[tokens/step] 精确={exact:.2f} vs 近似上界={approx:.2f}")
        print(f"    加速比：理论(精确期望)={speed_exact:.2f}x, 理论(近似上界)="
              f"{approx * 0.03 / (k * 0.005 + 0.03):.2f}x, MC 模拟={r.speedup:.2f}x")


def evaluate_acceptance_scan():
    print("\n📊 3. 接受率扫描：每步产出 token 的精确期望 E = (1-α^(k+1))/(1-α)")
    print("-" * 60)
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    ks = [1, 2, 4, 8]
    header = "  k\\α   " + "".join(f"{a:>7.1f}" for a in alphas)
    print(header)
    for k in ks:
        row = f"  {k:<4}" + "".join(
            f"{expected_tokens_per_step(k, a):>7.2f}" for a in alphas)
        print(row)
    print("\n  对应理论加速比（t_d=0.005s, T_fwd=0.03s）：")
    print(header)
    for k in ks:
        row = f"  {k:<4}" + "".join(
            f"{theoretical_speedup(k, a):>6.2f}x" for a in alphas)
        print(row)


def evaluate_failure_boundary():
    print("\n📊 4. 失效边界：α 低 + k 大 → draft 开销超过收益，变慢")
    print("-" * 60)
    for k, alpha in [(4, 0.3), (8, 0.3), (8, 0.5)]:
        r = simulate_speculative_decoding(
            num_tokens=100, draft_k=k, accept_rate=alpha,
            time_target_forward=0.03, time_draft_forward=0.005,
        )
        verdict = "变慢！" if r.speedup < 1.0 else "仍加速"
        print(f"  k={k}, α={alpha}: traditional={r.time_traditional:.2f}s, "
              f"spec={r.time_spec:.2f}s, speedup={r.speedup:.2f}x ({verdict})")


def evaluate():
    print("=" * 70)
    print("投机解码收益评估报告（Speculative Decoding）")
    print("=" * 70)
    evaluate_monte_carlo()
    evaluate_theory_vs_simulation()
    evaluate_acceptance_scan()
    evaluate_failure_boundary()
    print("\n📋 结论与集成优先级")
    print("-" * 60)
    print("  - α 是加速比上限，k 是杠杆：低 α 用小 k（2-4），高 α 用大 k（4-8）")
    print("  - draft 质量决定 α：Medusa ~0.5-0.6，EAGLE ~0.6-0.7，MTP ~0.7-0.8")
    print("  - 集成优先级：Prefix Caching / Chunked Prefill（Week 7 已实现）→")
    print("    CUDA Graph（Day 4）→ Speculative Decoding（收益高、复杂度高，可选）")


if __name__ == "__main__":
    evaluate()
