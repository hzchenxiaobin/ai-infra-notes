#!/usr/bin/env python3
"""Generate PNG diagrams for the 'min time max power' solution using matplotlib."""

import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_fig(filename: str) -> str:
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def draw_example_graph():
    """Draw Example 1 graph with node costs and edge times."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('示例 1：图结构与边权/点权', fontsize=14, fontweight='bold', pad=10)

    nodes = {
        0: (1.0, 2.0),
        1: (3.0, 3.0),
        2: (3.0, 1.0),
        3: (5.0, 1.0),
        4: (7.0, 2.0),
    }
    costs = {0: 2, 1: 3, 2: 1, 3: 1, 4: 1}
    edges = [
        (0, 1, 1),
        (1, 4, 1),
        (0, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
    ]

    for u, v, t in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=1.5, color='#555555',
                                connectionstyle="arc3,rad=0")
        ax.add_patch(arrow)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset_y = 0.15 if mid_y > 2 else -0.15
        ax.text(mid_x, mid_y + offset_y, f't={t}',
                fontsize=11, ha='center', va='center',
                color='#d35400', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    for u, (x, y) in nodes.items():
        color = '#3498db' if u == 0 else '#2ecc71' if u == 4 else '#ecf0f1'
        edgecolor = '#2980b9' if u == 0 else '#27ae60' if u == 4 else '#7f8c8d'
        circle = plt.Circle((x, y), 0.35, facecolor=color, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, str(u), fontsize=14, ha='center', va='center', fontweight='bold')
        ax.text(x, y - 0.65, f'cost={costs[u]}',
                fontsize=11, ha='center', va='center', color='#8e44ad', fontweight='bold')

    ax.text(0.1, 0.1, 'source=0, target=4, power=4',
            fontsize=10, ha='left', va='bottom', color='#555555', transform=ax.transAxes)

    return save_fig('min_time_max_power_example_graph.png')


def draw_counterexample_graph():
    """Draw the counterexample graph showing why one-state Dijkstra fails."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('反例：每个节点只保留一个状态会漏解', fontsize=14, fontweight='bold', pad=10)

    nodes = {
        0: (1.0, 2.0),
        3: (3.0, 3.0),
        1: (5.0, 2.0),
        2: (7.0, 2.0),
    }
    costs = {0: 1, 1: 1, 2: 1, 3: 10}
    edges = [
        (0, 3, 1),
        (3, 1, 1),
        (0, 1, 3),
        (1, 2, 1),
    ]

    for u, v, t in edges:
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=1.5, color='#555555',
                                connectionstyle="arc3,rad=0")
        ax.add_patch(arrow)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset_y = 0.15 if u == 0 and v == 3 else -0.15 if u == 0 and v == 1 else 0.12
        ax.text(mid_x, mid_y + offset_y, f't={t}',
                fontsize=11, ha='center', va='center',
                color='#d35400', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    for u, (x, y) in nodes.items():
        color = '#3498db' if u == 0 else '#2ecc71' if u == 2 else '#ecf0f1'
        edgecolor = '#2980b9' if u == 0 else '#27ae60' if u == 2 else '#7f8c8d'
        circle = plt.Circle((x, y), 0.35, facecolor=color, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, str(u), fontsize=14, ha='center', va='center', fontweight='bold')
        ax.text(x, y - 0.65, f'cost={costs[u]}',
                fontsize=11, ha='center', va='center', color='#8e44ad', fontweight='bold')

    ax.text(0.5, 0.05, 'power=11，路径 0→3→1 到 1 更快但耗电 11，无法继续；\n'
                        '路径 0→1 到 1 较慢但耗电 1，可以走 1→2。正确答案 [4, 9]。',
            fontsize=10, ha='center', va='bottom', color='#333333', transform=ax.transAxes)

    return save_fig('min_time_max_power_counterexample.png')


def draw_state_design():
    """Draw state design for DP approach."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('状态设计：按耗电拆点', fontsize=14, fontweight='bold', pad=10)

    # dp[u][p] box
    box1 = FancyBboxPatch((1, 2.8), 2.8, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#eaf2f8', edgecolor='#2980b9', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.4, 3.45, 'dp[u][p] = min_time', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(2.4, 3.1, '到达 u 且总耗电为 p', fontsize=10, ha='center', va='center', color='#2c3e50')
    ax.text(2.4, 2.85, '时的最小时间', fontsize=10, ha='center', va='center', color='#2c3e50')

    # Arrow down
    ax.annotate('', xy=(2.4, 2.6), xytext=(2.4, 2.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#7f8c8d'))

    # Relaxation box
    box2 = FancyBboxPatch((0.5, 0.8), 5.6, 1.6, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='#fdebd0', edgecolor='#e67e22', linewidth=2)
    ax.add_patch(box2)
    ax.text(3.3, 2.0, '边 u → v，边权时间 t，节点耗电 cost[u]', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(3.3, 1.5, '若 p + cost[u] <= power：', fontsize=11, ha='center', va='center')
    ax.text(3.3, 1.05, 'dp[v][p + cost[u]] = min(dp[v][p + cost[u]], dp[u][p] + t)',
            fontsize=10, ha='center', va='center', family='monospace')

    return save_fig('min_time_max_power_state_design.png')


def draw_dp_steps():
    """Draw DP execution table for the counterexample."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('DP 执行过程（反例：power=11）', fontsize=14, fontweight='bold', pad=10)

    # dp table values: (node, power) -> time
    # After algorithm converges:
    # node 0: p=0 -> 0
    # node 3: p=1 -> 1
    # node 1: p=1 -> 3 (direct), p=11 -> 2 (via 3)
    # node 2: p=2 -> 4
    table_data = [
        ['节点 0', 'dp[0][0] = 0', '其余 INF'],
        ['节点 3', 'dp[3][1] = 1', '其余 INF'],
        ['节点 1', 'dp[1][1] = 3', 'dp[1][11] = 2', '其余 INF'],
        ['节点 2', 'dp[2][2] = 4', '其余 INF'],
    ]

    y_start = 3.5
    row_h = 0.55
    for i, row in enumerate(table_data):
        y = y_start - i * row_h
        # Header cell
        rect = mpatches.Rectangle((0.3, y), 1.2, row_h,
                                  facecolor='#e8e8e8', edgecolor='#999999', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.9, y + row_h / 2, row[0], fontsize=11, ha='center', va='center', fontweight='bold')

        for j, cell in enumerate(row[1:]):
            x = 1.6 + j * 2.8
            rect = mpatches.Rectangle((x, y), 2.6, row_h,
                                      facecolor='#ffffff', edgecolor='#999999', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + 1.3, y + row_h / 2, cell,
                    fontsize=10, ha='center', va='center')

    # Highlight answer
    ax.text(0.3, 0.8, '最终：节点 2 的最短时间为 dp[2][2]=4，对应耗电 2，剩余电量 11-2=9',
            fontsize=11, ha='left', va='center', color='#27ae60', fontweight='bold')
    ax.text(0.3, 0.4, '若只保留“到达 1 的最快状态”(2,11)，会因电量不足而误判节点 1 为死路。',
            fontsize=10, ha='left', va='center', color='#c0392b')

    return save_fig('min_time_max_power_dijkstra_steps.png')


def main():
    paths = [
        draw_example_graph(),
        draw_counterexample_graph(),
        draw_state_design(),
        draw_dp_steps(),
    ]
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
