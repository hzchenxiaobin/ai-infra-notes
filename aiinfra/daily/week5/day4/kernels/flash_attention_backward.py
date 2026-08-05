# flash_attention_backward.py —— Simplified FlashAttention Backward (PyTorch, teaching)
# 运行命令: python3 flash_attention_backward.py
# 依赖: pip install torch
#
# 核心要点:
#   1. Forward 只存 Q/K/V/O + L (logsumexp)，内存 O(Nd)，不存 N×N 的 S/P
#   2. Backward 用 L 重算 softmax 权重 P = exp(S - L)，分块累加 dQ/dK/dV
#   3. 用 torch.autograd.gradcheck 做数值验证

import torch
import math


class FlashAttentionFunction(torch.autograd.Function):
    """FlashAttention 的简化 PyTorch 实现，重点演示 backward 的 recomputation 策略。"""

    @staticmethod
    def forward(ctx, Q, K, V, Br=64, Bc=64):
        # forward: O = softmax(QK^T * scale) @ V
        # 只保存 L = logsumexp(S) —— 每行一个标量，O(N) 总量
        N, d = Q.shape[-2], Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale       # (..., N, N) 仅临时
        L = torch.logsumexp(S, dim=-1)                          # (..., N) ← 关键：只存这个
        P = torch.exp(S - L.unsqueeze(-1))                      # softmax(S) 行归一
        O = torch.matmul(P, V)

        # save_for_backward 只存 O(Nd) 的张量 + O(N) 的 L，不存 P（O(N²)）
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.Br, ctx.Bc = Br, Bc
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        Br, Bc = ctx.Br, ctx.Bc
        N = Q.shape[-2]

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # FA backward 的关键技巧：D_i = rowsum(P_i * dP_i) = O_i · dO_i
        # 推导: dP_ij = dO_i·V_j, 故 sum_j P_ij dP_ij = dO_i · sum_j P_ij V_j = dO_i · O_i
        # 用保存的 O 与传入的 dO 一次算出，无需在 tile 循环里累加。
        Di = (O * dO).sum(dim=-1, keepdim=True)            # (..., N, 1)

        # 分块重算 S/P，逐块累加梯度。关键：P_ij = exp(S_ij - L_i)，
        # 只用 Q/K/L 即可恢复，无需存任何 N×N 矩阵。
        for q0 in range(0, N, Br):
            q1 = min(q0 + Br, N)
            Qi = Q[..., q0:q1, :]
            Li = L[..., q0:q1]                       # (..., Br)
            Di_q = Di[..., q0:q1, :]                 # (..., Br, 1)
            dOi = dO[..., q0:q1, :]
            dQi = torch.zeros_like(Qi)

            for kv0 in range(0, N, Bc):
                kv1 = min(kv0 + Bc, N)
                Kj = K[..., kv0:kv1, :]
                Vj = V[..., kv0:kv1, :]

                # === 重算 S_ij / P_ij（recomputation 的核心）===
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale
                Pij = torch.exp(Sij - Li.unsqueeze(-1))          # (..., Br, Bc)

                # === 标准 attention 反向公式 ===
                # dV_j += P_ij^T @ dO_i
                dV[..., kv0:kv1, :] += torch.matmul(Pij.transpose(-2, -1), dOi)
                # dP_ij = dO_i @ V_j^T
                dPij = torch.matmul(dOi, Vj.transpose(-2, -1))
                # dS_ij = P_ij * (dP_ij - D_i)，D_i 已用 O·dO 全局算好
                dSij = Pij * (dPij - Di_q)
                # dQ_i += dS_ij @ K_j * scale
                dQi += torch.matmul(dSij, Kj) * scale
                # dK_j += dS_ij^T @ Q_i * scale
                dK[..., kv0:kv1, :] += torch.matmul(dSij.transpose(-2, -1), Qi) * scale

            dQ[..., q0:q1, :] = dQi
        return dQ, dK, dV, None, None


def flash_attention(Q, K, V):
    return FlashAttentionFunction.apply(Q, K, V)


def standard_attention(Q, K, V):
    d = Q.size(-1)
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def memory_mb(N, d, elems_per_float=4):
    fa_saved = (4 * N * d + N) * elems_per_float           # Q,K,V,O + L
    std_saved = (4 * N * d + N * N) * elems_per_float      # 额外存 P
    return fa_saved / (1024 * 1024), std_saved / (1024 * 1024)


def main():
    torch.manual_seed(42)

    # 1) gradcheck：要求 float64 + 小尺寸
    print("=== torch.autograd.gradcheck ===")
    N, d = 8, 4
    Q = torch.randn(1, N, d, dtype=torch.float64, requires_grad=True)
    K = torch.randn(1, N, d, dtype=torch.float64, requires_grad=True)
    V = torch.randn(1, N, d, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(flash_attention, (Q, K, V), eps=1e-6, atol=1e-4)
    print(f"gradcheck: {'PASS' if ok else 'FAIL'}\n")

    # 2) 正确性：与 standard attention 对比（含反向梯度）
    print("=== Correctness vs standard attention (fwd + bwd) ===")
    N, d = 128, 32
    Q = torch.randn(2, N, d, dtype=torch.float64, requires_grad=True)
    K = torch.randn(2, N, d, dtype=torch.float64, requires_grad=True)
    V = torch.randn(2, N, d, dtype=torch.float64, requires_grad=True)
    O_fa = flash_attention(Q, K, V)
    O_std = standard_attention(Q, K, V)
    print(f"  fwd maxDiff = {(O_fa - O_std).abs().max().item():.2e}")

    g = torch.randn_like(O_fa)
    dQ_fa, dK_fa, dV_fa = torch.autograd.grad(O_fa, (Q, K, V), g)
    O_std.backward(g)
    print(f"  dQ maxDiff = {(dQ_fa - Q.grad).abs().max().item():.2e}")
    print(f"  dK maxDiff = {(dK_fa - K.grad).abs().max().item():.2e}")
    print(f"  dV maxDiff = {(dV_fa - V.grad).abs().max().item():.2e}\n")

    # 3) 内存：FA 存 O(Nd)，标准存 O(N²)
    print("=== Saved-tensor memory (forward) ===")
    print(f"{'N':>6} {'d':>4} {'FA(MB)':>10} {'Std(MB)':>10} {'ratio':>8}")
    for N, d in [(1024, 64), (4096, 64), (8192, 64)]:
        fa, std = memory_mb(N, d)
        print(f"{N:6d} {d:4d} {fa:10.3f} {std:10.3f} {std / fa:8.1f}x")
    print("\nFA 仅存 Q/K/V/O + L = O(Nd)；标准 autograd 额外物化 P = O(N²)。")


if __name__ == "__main__":
    main()
