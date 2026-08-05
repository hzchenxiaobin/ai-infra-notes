"""pipeline_schedule_sim.py —— Pipeline Parallelism 调度模拟器

模拟 GPipe vs 1F1B vs Interleaved 1F1B 的调度时间线与 bubble ratio。
不依赖任何 GPU/CUDA，纯 Python 实现，用于理解 PP 调度策略。

运行: python3 pipeline_schedule_sim.py
"""


def simulate_gpipe(P, M):
    """模拟 GPipe 调度, 返回 (timeline, bubble, total_time).

    GPipe: 先做全部 M 个 forward, 再做全部 M 个 backward.
    每个 stage 的 timeline 是 list of (phase, micro_batch, start, end).
    时间单位: 1 个 micro-batch 在一个 stage 上的执行时间 = 1.
    """
    timeline = [[] for _ in range(P)]
    # Forward phase
    for m in range(M):
        for s in range(P):
            prev_end = timeline[s][-1][3] if timeline[s] else 0
            dep_end = timeline[s-1][-1][3] if s > 0 and timeline[s-1] else 0
            start = max(prev_end, dep_end)
            timeline[s].append(('F', m, start, start + 1))
    # Backward phase (反向)
    for m in reversed(range(M)):
        for s in reversed(range(P)):
            prev_end = timeline[s][-1][3] if timeline[s] else 0
            dep_end = timeline[s+1][-1][3] if s < P-1 and timeline[s+1] else 0
            start = max(prev_end, dep_end)
            timeline[s].append(('B', m, start, start + 1))
    total = max(ts[-1][3] for ts in timeline if ts)
    useful = 2 * M * P
    bubble = (total * P - useful) / (total * P)
    return timeline, bubble, total


def simulate_1f1b(P, M):
    """模拟 1F1B 调度 (steady-state interleaved forward/backward).

    1F1B 的关键: warmup 阶段每个 stage i 先做 P-1-i 个 forward,
    然后进入稳态 1F1B (forward + backward 交替),
    最后 cooldown 把剩余的 backward 做完.
    显存优势: 每个 stage 稳态只有 P 份未 backward 的 activation (GPipe 是 M 份).
    bubble 与 GPipe 相同: (P-1)/(M+P-1).
    """
    timeline = [[] for _ in range(P)]
    fwd_done = [[-1] * M for _ in range(P)]
    bwd_done = [[-1] * M for _ in range(P)]

    total_ops = 2 * M * P
    done_ops = 0
    t = 0
    while done_ops < total_ops:
        scheduled_this_step = [False] * P
        for s in range(P):
            # 1F1B: 优先 backward (稳态时 forward 之后尽快 backward)
            bwd_m = -1
            for m in range(M):
                if bwd_done[s][m] < 0 and fwd_done[s][m] >= 0 and t >= fwd_done[s][m]:
                    if s == P - 1:
                        bwd_m = m
                        break
                    elif bwd_done[s+1][m] >= 0 and t >= bwd_done[s+1][m]:
                        bwd_m = m
                        break
            if bwd_m >= 0:
                prev_end = timeline[s][-1][3] if timeline[s] else 0
                start = max(t, prev_end)
                timeline[s].append(('B', bwd_m, start, start + 1))
                bwd_done[s][bwd_m] = start + 1
                scheduled_this_step[s] = True
                done_ops += 1
                continue
            # 否则 forward
            fwd_m = -1
            for m in range(M):
                if fwd_done[s][m] < 0:
                    if s == 0:
                        fwd_m = m
                        break
                    elif fwd_done[s-1][m] >= 0 and t >= fwd_done[s-1][m]:
                        fwd_m = m
                        break
            if fwd_m >= 0:
                prev_end = timeline[s][-1][3] if timeline[s] else 0
                start = max(t, prev_end)
                timeline[s].append(('F', fwd_m, start, start + 1))
                fwd_done[s][fwd_m] = start + 1
                scheduled_this_step[s] = True
                done_ops += 1
        t += 1
    total = max(ts[-1][3] for ts in timeline if ts)
    useful = 2 * M * P
    bubble = (total * P - useful) / (total * P)
    return timeline, bubble, total


def simulate_interleaved_1f1b(P, M, V):
    """模拟 Interleaved 1F1B (virtual pipeline).

    V 个 virtual stage, 每个 device 负责 V 段层.
    等效工作总量变为 V*M, 理论 bubble 降为 (P-1)/(V*M+P-1).
    通信次数增 V 倍.

    注意: 本模拟器用贪心调度 (每步优先 backward), 模拟 bubble 会略高于
    理论最优值 (Megatron-LM 的调度有更精细的 warmup/cooldown 策略).
    理论公式 bubble_ratio(P,M,V) 是面试口径, 模拟器用于直观理解调度形状.
    """
    timeline = [[] for _ in range(P)]
    # V 个 virtual stage, 每个 device 有 V 段
    # 简化: 把 V*M 个 micro-batch 分成 V 组, 每组 M 个, 轮流喂给 P 个 device
    total_fwd = V * M
    total_bwd = V * M
    fwd_done = [[[-1] * M for _ in range(V)] for _ in range(P)]
    bwd_done = [[[-1] * M for _ in range(V)] for _ in range(P)]

    total_ops = 2 * V * M * P
    done_ops = 0
    t = 0
    while done_ops < total_ops:
        for s in range(P):
            # 优先 backward
            found = False
            for v in range(V):
                for m in range(M):
                    if bwd_done[s][v][m] < 0 and fwd_done[s][v][m] >= 0 and t >= fwd_done[s][v][m]:
                        # backward 依赖: virtual stage (v, s) 依赖 (v, s+1) 或 (v+1, 0) 若 s==P-1
                        dep_ok = False
                        if s < P - 1:
                            if bwd_done[s+1][v][m] >= 0 and t >= bwd_done[s+1][v][m]:
                                dep_ok = True
                        else:
                            # 最后一个 device 的 virtual stage v 依赖 virtual stage v+1 的 device 0 backward
                            if v == V - 1:
                                dep_ok = True
                            elif bwd_done[0][v+1][m] >= 0 and t >= bwd_done[0][v+1][m]:
                                dep_ok = True
                        if dep_ok:
                            prev_end = timeline[s][-1][3] if timeline[s] else 0
                            start = max(t, prev_end)
                            timeline[s].append(('B', (v, m), start, start + 1))
                            bwd_done[s][v][m] = start + 1
                            done_ops += 1
                            found = True
                            break
                if found:
                    break
            if found:
                continue
            # forward
            for v in range(V):
                for m in range(M):
                    if fwd_done[s][v][m] < 0:
                        dep_ok = False
                        if s == 0:
                            if v == 0:
                                dep_ok = True
                            elif fwd_done[P-1][v-1][m] >= 0 and t >= fwd_done[P-1][v-1][m]:
                                dep_ok = True
                        else:
                            if fwd_done[s-1][v][m] >= 0 and t >= fwd_done[s-1][v][m]:
                                dep_ok = True
                        if dep_ok:
                            prev_end = timeline[s][-1][3] if timeline[s] else 0
                            start = max(t, prev_end)
                            timeline[s].append(('F', (v, m), start, start + 1))
                            fwd_done[s][v][m] = start + 1
                            done_ops += 1
                            found = True
                            break
                if found:
                    break
        t += 1
    total = max(ts[-1][3] for ts in timeline if ts)
    useful = 2 * V * M * P
    bubble = (total * P - useful) / (total * P)
    return timeline, bubble, total


def bubble_ratio(P, M, V=1):
    """理论 bubble ratio 公式."""
    if V == 1:
        return (P - 1) / (M + P - 1)
    else:
        return (P - 1) / (V * M + P - 1)


def print_timeline(timeline, P, max_steps=20):
    """打印调度时间线 (每个 stage 的前 max_steps 个操作)."""
    for s in range(P):
        ops = timeline[s][:max_steps]
        ops_str = ' '.join(f"{p}{m if isinstance(m, int) else m[0]}[{st}-{en}]"
                          for p, m, st, en in ops)
        suffix = ' ...' if len(timeline[s]) > max_steps else ''
        print(f"  Stage {s}: {ops_str}{suffix}")


def main():
    print("=" * 70)
    print("Pipeline Parallelism 调度模拟器")
    print("=" * 70)

    print("\n--- 1. GPipe vs 1F1B 对比 ---\n")
    print(f"{'P':>4} {'M':>4} {'GPipe bubble':>14} {'1F1B bubble':>12} "
          f"{'GPipe total':>12} {'1F1B total':>12} {'理论 (P-1)/(M+P-1)':>20}")
    for P, M in [(4, 4), (4, 8), (8, 8), (8, 16), (8, 32)]:
        tl_g, bub_g, total_g = simulate_gpipe(P, M)
        tl_1, bub_1, total_1 = simulate_1f1b(P, M)
        theory = bubble_ratio(P, M, V=1)
        print(f"{P:>4} {M:>4} {bub_g:>13.2%} {bub_1:>11.2%} "
              f"{total_g:>11} {total_1:>11} {theory:>19.2%}")

    print("\n--- 2. 1F1B 调度时间线 (P=4, M=4) ---\n")
    tl, bub, total = simulate_1f1b(4, 4)
    print_timeline(tl, 4)
    print(f"\n  bubble = {bub:.2%}, total time = {total}")

    print("\n--- 3. Interleaved 1F1B: V 扫描 (P=8, M=16) ---\n")
    print(f"{'V':>4} {'理论 bubble':>12} {'模拟 bubble':>12}")
    for V in [1, 2, 4]:
        theory = bubble_ratio(8, 16, V=V)
        tl_v, bub_v, total_v = simulate_interleaved_1f1b(8, 16, V)
        print(f"{V:>4} {theory:>11.2%} {bub_v:>11.2%}")

    print("\n--- 4. 结论 ---\n")
    print("  - GPipe 与 1F1B 的 bubble 相同: (P-1)/(M+P-1)")
    print("  - 1F1B 的优势是显存: 稳态 P 份 activation (vs GPipe 的 M 份)")
    print("  - Interleaved 1F1B (V 个 virtual stage): bubble 降为 (P-1)/(V*M+P-1)")
    print("  - Interleaved 代价: 通信次数增 V 倍")


if __name__ == "__main__":
    main()
