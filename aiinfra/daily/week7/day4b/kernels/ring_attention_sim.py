# ring_attention_sim.py —— 单机模拟 Ring Attention（N GPU 环形 KV 传输 + online softmax）
# 运行命令: python ring_attention_sim.py
# 依赖: numpy（无需 GPU / torch，纯 CPU 模拟分布式逻辑）
"""
演示 Ring Attention 的核心机制：
  - N 个 GPU 各持 1/N 的 Q, K, V（按 sequence 维切分）
  - KV 在 GPU 间环形传递：每步 GPU i 把自己的 KV 发给 GPU (i+1)%N
  - 每步本地用 online softmax 增量更新 (m, l, O)，与通信重叠
  - N 步后每卡已"看过"全部 KV，O/l 即为完整 attention 输出
对比：
  - standard attention：所有 Q,K,V 在一张"卡"上一次性算（baseline）
  - ring attention：分布式 N 步累加（被验证与 baseline 数值一致）
"""

import numpy as np

N_GPUS = 4
SEQ = 16
D = 8
SCALE = D ** -0.5


def standard_attention(Q, K, V):
    S = (Q @ K.T) * SCALE
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def online_softmax_update(m_old, l_old, o_old, s_block, v_block):
    m_block = s_block.max(axis=-1)
    m_new = np.maximum(m_old, m_block)
    alpha = np.exp(m_old - m_new)[:, None]
    beta = np.exp(s_block - m_new[:, None])
    l_new = l_old * alpha[:, 0] + beta.sum(axis=-1)
    o_new = o_old * alpha + beta @ v_block
    return m_new, l_new, o_new


def ring_attention(Q, K, V, n=N_GPUS, log=False):
    q_shards = np.array_split(Q, n, axis=0)
    k_shards = np.array_split(K, n, axis=0)
    v_shards = np.array_split(V, n, axis=0)

    states = [
        (np.full((q.shape[0],), -np.inf), np.zeros(q.shape[0]),
         np.zeros((q.shape[0], D)))
        for q in q_shards
    ]
    cur_k, cur_v = list(k_shards), list(v_shards)

    for step in range(n):
        for i in range(n):
            q = q_shards[i]
            s = (q @ cur_k[i].T) * SCALE
            m, l, o = states[i]
            m, l, o = online_softmax_update(m, l, o, s, cur_v[i])
            states[i] = (m, l, o)
            if log and i == 0:
                kv_id = (i - step) % n
                print(f"    step {step} GPU{i}: 用 KV{kv_id} 本地算 attn → 更新 (m,l,O)")

        new_k = [cur_k[(i - 1) % n] for i in range(n)]
        new_v = [cur_v[(i - 1) % n] for i in range(n)]
        cur_k, cur_v = new_k, new_v
        if log:
            print(f"  step {step} 完成: 各卡把 KV 发给右邻 (i+1)%{n}")

    outs = [states[i][2] / states[i][1][:, None] for i in range(n)]
    return np.concatenate(outs, axis=0)


def main():
    np.random.seed(42)
    Q = np.random.randn(SEQ, D).astype(np.float32)
    K = np.random.randn(SEQ, D).astype(np.float32)
    V = np.random.randn(SEQ, D).astype(np.float32)

    print("=" * 64)
    print("  Ring Attention 单机模拟（N=4 GPU）")
    print("=" * 64)
    print(f"  seq={SEQ}, d={D}, scale={SCALE:.4f}, N_GPUS={N_GPUS}")

    ref = standard_attention(Q, K, V)
    print("\n[1] standard attention（全部在一张卡上）:")
    print(f"    output[0, :3] = {ref[0, :3]}")

    print("\n[2] ring attention（KV 环形传输 + online softmax 增量合并）:")
    out = ring_attention(Q, K, V, n=N_GPUS, log=True)
    print(f"    output[0, :3] = {out[0, :3]}")

    max_diff = np.abs(ref - out).max()
    print("\n[3] 正确性校验:")
    print(f"    max|ref - ring| = {max_diff:.3e}")
    print(f"    结果: {'PASS ✅' if max_diff < 1e-5 else 'FAIL ❌'}")

    print("\n[4] 显存与通信分析:")
    kv_total = SEQ * D * 4 * 2
    kv_block = kv_total / N_GPUS
    print(f"    全量 KV = {kv_total} B, 每块 KV = {kv_total}/{N_GPUS} = {kv_block:.0f} B")
    print(f"    Ring 每卡峰值 KV buffer = 1 块 = {kv_block:.0f} B（流式轮转，只留当前块）")
    print(f"    朴素(全量 gather)每卡 KV buffer = {kv_total} B → Ring 省 {(N_GPUS-1)/N_GPUS*100:.0f}%")
    print(f"    总通信量两者相同(≈ N×KV)，但 Ring 把通信切成分块与计算重叠")
    print(f"    → 核心收益: KV buffer 显存省 (N-1)/N + comm/compute 重叠, 而非总通信量减少")

    print("\n[5] compute / comm 重叠示意（双流）:")
    print("    compute: |attn KV0|attn KV1|attn KV2|attn KV3|")
    print("    comm:    ........|send KV0|send KV1|send KV2|..")
    print("    重叠后:  total ≈ max(T_compute, T_comm) × N 步")


if __name__ == "__main__":
    main()
