#!/usr/bin/env python3
"""Generate SVG diagrams for the 'Trapping Rain Water' solution."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_PROP = FontProperties(fname=FONT_PATH)


def sketch_line(ax, x, y, color='#333333', linewidth=2, alpha=0.9):
    """Draw a slightly wiggly line to mimic hand-drawn style."""
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha,
            solid_capstyle='round', solid_joinstyle='round')


def save_svg(filename: str) -> str:
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def draw_trap_water_column():
    """Draw a single column showing how water is trapped."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('位置 i 的接水量由左右最高柱子的短板决定',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    heights = [2, 0, 3]
    labels = ['left\nmax=3', 'position i\nheight=1', 'right\nmax=4']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    x_positions = [1.5, 3.5, 5.5]
    water_level = 3

    # Ground line
    sketch_line(ax, [0.5, 7.0], [0, 0], color='#7f8c8d', linewidth=3)

    for i, (h, label, color, x) in enumerate(zip(heights, labels, colors, x_positions)):
        # Column
        rect = mpatches.Rectangle((x - 0.4, 0), 0.8, h,
                                  facecolor=color, edgecolor='#2c3e50',
                                  linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, h + 0.2, f'h={h}' if i == 1 else label,
                fontsize=11, ha='center', va='bottom', fontweight='bold',
                color='#2c3e50', fontproperties=FONT_PROP)

    # Water between columns
    water = mpatches.Rectangle((x_positions[0] + 0.4, 0), 2.2, water_level,
                               facecolor='#5dade2', edgecolor='#2874a6',
                               linewidth=2, alpha=0.4)
    ax.add_patch(water)

    # Water level dashed line
    sketch_line(ax, [x_positions[0] + 0.4, x_positions[2] - 0.4],
                [water_level, water_level], color='#2874a6', linewidth=1.5)
    ax.text(3.5, water_level + 0.15, f'water level = min(3, 4) = {water_level}',
            fontsize=11, ha='center', va='bottom', color='#2874a6',
            fontweight='bold', fontproperties=FONT_PROP)

    # Formula box
    box = FancyBboxPatch((1.5, 3.8), 4.5, 0.6,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(box)
    ax.text(3.75, 4.1, 'water[i] = min(left_max, right_max) - height[i]',
            fontsize=12, ha='center', va='center', fontweight='bold',
            family='monospace')

    return save_svg('trap_water_column.svg')


def draw_two_pointers():
    """Illustrate why we move the shorter side pointer."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('双指针：移动高度较小的一侧',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    heights = [1, 0, 2, 0, 1, 0, 3]
    n = len(heights)
    width = 0.7
    start_x = 0.5

    # Ground
    sketch_line(ax, [0.3, 7.3], [0, 0], color='#7f8c8d', linewidth=3)

    for i, h in enumerate(heights):
        x = start_x + i * 0.95
        color = '#3498db' if i in [0, 6] else '#ecf0f1'
        edge = '#2980b9' if i in [0, 6] else '#7f8c8d'
        rect = mpatches.Rectangle((x, 0), width, h,
                                  facecolor=color, edgecolor=edge, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width / 2, -0.25, str(i), fontsize=9, ha='center',
                va='top', color='#2c3e50', fontproperties=FONT_PROP)

    # Pointers
    left_x = start_x + width / 2
    right_x = start_x + (n - 1) * 0.95 + width / 2
    ax.annotate('left\nleft_max=1', xy=(left_x, heights[0]), xytext=(left_x, 3.2),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
                fontsize=11, ha='center', va='bottom', color='#e74c3c',
                fontweight='bold', fontproperties=FONT_PROP)
    ax.annotate('right\nright_max=3', xy=(right_x, heights[-1]), xytext=(right_x, 3.2),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                fontsize=11, ha='center', va='bottom', color='#2ecc71',
                fontweight='bold', fontproperties=FONT_PROP)

    # Highlight smaller side
    ax.text(1.0, 3.8, 'height[left] < height[right] → 移动 left',
            fontsize=12, ha='left', va='center', color='#e74c3c',
            fontweight='bold', fontproperties=FONT_PROP)

    # Explanation box
    box = FancyBboxPatch((0.5, 4.0), 6.8, 0.4,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor='#eafaf1', edgecolor='#2ecc71', linewidth=2)
    ax.add_patch(box)
    ax.text(3.9, 4.2, '较小侧的 max 已经确定，不会受另一侧未知情况影响',
            fontsize=11, ha='center', va='center', fontweight='bold',
            color='#27ae60', fontproperties=FONT_PROP)

    return save_svg('trap_two_pointers.svg')


def draw_algorithm_flow():
    """Draw Kadane-like algorithm flow for trapping rain water."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('双指针算法流程',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    boxes = [
        ('初始化\nleft=0, right=n-1\nleft_max=0, right_max=0\nans=0', 4.0, 5.2, '#eaf2f8', '#2980b9'),
        ('left < right ?', 4.0, 4.1, '#fdebd0', '#e67e22'),
        ('height[left] < height[right]', 4.0, 3.0, '#fdebd0', '#e67e22'),
        ('更新 left_max\n若 height[left] >= left_max\n否则 ans += left_max - height[left]\nleft++', 2.2, 1.7, '#eafaf1', '#27ae60'),
        ('更新 right_max\n若 height[right] >= right_max\n否则 ans += right_max - height[right]\nright--', 5.8, 1.7, '#eafaf1', '#27ae60'),
        ('返回 ans', 4.0, 0.5, '#eaf2f8', '#2980b9'),
    ]

    for text, x, y, facecolor, edgecolor in boxes:
        box = FancyBboxPatch((x - 1.5, y - 0.35), 3.0, 0.7,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=10, ha='center', va='center',
                fontweight='bold', color='#2c3e50', fontproperties=FONT_PROP)

    # Arrows
    arrows = [
        ((4.0, 4.85), (4.0, 4.45)),
        ((4.0, 3.75), (4.0, 3.35)),
        ((2.5, 3.0), (2.2, 2.05)),
        ((5.5, 3.0), (5.8, 2.05)),
        ((2.2, 1.35), (4.0, 0.85)),
        ((5.8, 1.35), (4.0, 0.85)),
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    ax.text(2.0, 3.5, 'Yes', fontsize=10, ha='center', color='#e74c3c',
            fontweight='bold', fontproperties=FONT_PROP)
    ax.text(6.0, 3.5, 'No', fontsize=10, ha='center', color='#2ecc71',
            fontweight='bold', fontproperties=FONT_PROP)

    return save_svg('trap_algorithm_flow.svg')


def main():
    paths = [
        draw_trap_water_column(),
        draw_two_pointers(),
        draw_algorithm_flow(),
    ]
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
