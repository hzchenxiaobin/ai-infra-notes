#!/usr/bin/env python3
"""Generate SVG diagrams for the 'Maximum Subarray' solution."""

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


def save_svg(filename: str) -> str:
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def draw_prefix_sum():
    """Show how subarray sum is represented by prefix sums."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('前缀和：子数组 [i, j] 的和等于 prefix[j+1] - prefix[i]',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    prefix = [0]
    for v in nums:
        prefix.append(prefix[-1] + v)

    n = len(nums)
    cell_w = 0.85
    start_x = 0.5
    y_nums = 2.6
    y_prefix = 1.1

    # Draw nums row
    for i, v in enumerate(nums):
        x = start_x + i * cell_w
        rect = mpatches.Rectangle((x, y_nums), cell_w, 0.7,
                                  facecolor='#eaf2f8', edgecolor='#2980b9',
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y_nums + 0.35, str(v),
                fontsize=11, ha='center', va='center', color='#2c3e50',
                fontweight='bold', fontproperties=FONT_PROP)
        ax.text(x + cell_w / 2, y_nums - 0.25, f'i={i}',
                fontsize=9, ha='center', va='top', color='#7f8c8d',
                fontproperties=FONT_PROP)

    # Draw prefix row
    for i, v in enumerate(prefix):
        x = start_x + i * cell_w
        rect = mpatches.Rectangle((x, y_prefix), cell_w, 0.7,
                                  facecolor='#eafaf1', edgecolor='#27ae60',
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y_prefix + 0.35, str(v),
                fontsize=10, ha='center', va='center', color='#2c3e50',
                fontweight='bold', fontproperties=FONT_PROP)
        ax.text(x + cell_w / 2, y_prefix - 0.25, f'p={i}',
                fontsize=9, ha='center', va='top', color='#7f8c8d',
                fontproperties=FONT_PROP)

    # Highlight subarray [3, 6]
    i, j = 3, 6
    ax.plot([start_x + i * cell_w, start_x + (j + 1) * cell_w + cell_w],
            [y_nums + 0.85, y_nums + 0.85], color='#e74c3c', linewidth=3)
    ax.text(start_x + (i + (j + 1 - i) / 2) * cell_w + cell_w / 2, y_nums + 1.05,
            '子数组 [3, 6] = [4, -1, 2, 1]', fontsize=11, ha='center',
            va='bottom', color='#e74c3c', fontweight='bold',
            fontproperties=FONT_PROP)

    # Formula box
    box = FancyBboxPatch((1.2, 3.6), 6.6, 0.6,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(box)
    ax.text(4.5, 3.9, 'sum(i, j) = prefix[j+1] - prefix[i]',
            fontsize=12, ha='center', va='center', fontweight='bold',
            family='monospace')

    return save_svg('maxsubarray_prefix_sum.svg')


def draw_kadane():
    """State transition for Kadane's algorithm."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('Kadane：dp[i] 选择「继承」或「重新开始」',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    nums = [-2, 1, -3, 4, -1, 2]
    dp = []
    cur = 0
    for v in nums:
        if cur <= 0:
            cur = v
        else:
            cur += v
        dp.append(cur)

    cell_w = 0.95
    start_x = 0.5
    y = 2.0
    for i, (v, d) in enumerate(zip(nums, dp)):
        x = start_x + i * cell_w
        rect = mpatches.Rectangle((x, y), cell_w, 0.8,
                                  facecolor='#eaf2f8', edgecolor='#2980b9',
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y + 0.55, f'nums[{i}]={v}',
                fontsize=10, ha='center', va='center', color='#2c3e50',
                fontweight='bold', fontproperties=FONT_PROP)
        ax.text(x + cell_w / 2, y + 0.15, f'dp={d}',
                fontsize=10, ha='center', va='center', color='#27ae60',
                fontweight='bold', fontproperties=FONT_PROP)

    # Arrows showing transitions
    for i in range(1, len(nums)):
        x_from = start_x + (i - 1) * cell_w + cell_w / 2
        x_to = start_x + i * cell_w + cell_w / 2
        ax.annotate('', xy=(x_to, y + 0.85), xytext=(x_from, y + 0.85),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # Decision branches
    branch_y = 3.4
    ax.annotate('', xy=(3.5, branch_y - 0.15), xytext=(2.0, branch_y - 0.15),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.text(1.0, branch_y, 'dp[i-1] \u003c= 0\n从 nums[i] 重新开始',
            fontsize=10, ha='center', va='bottom', color='#e74c3c',
            fontweight='bold', fontproperties=FONT_PROP)

    ax.annotate('', xy=(6.0, branch_y - 0.15), xytext=(4.5, branch_y - 0.15),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    ax.text(7.0, branch_y, 'dp[i-1] > 0\n接在后面',
            fontsize=10, ha='center', va='bottom', color='#2ecc71',
            fontweight='bold', fontproperties=FONT_PROP)

    # Formula box
    box = FancyBboxPatch((1.5, 0.4), 5.0, 0.6,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(box)
    ax.text(4.0, 0.7, 'dp[i] = max(nums[i], dp[i-1] + nums[i])',
            fontsize=12, ha='center', va='center', fontweight='bold',
            family='monospace')

    return save_svg('maxsubarray_kadane.svg')


def draw_algorithm_flow():
    """Flowchart for Kadane's algorithm."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Kadane 算法流程',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    boxes = [
        ('初始化\ncur_sum = nums[0]\nmax_sum = nums[0]\ni = 1', 4.0, 5.2, '#eaf2f8', '#2980b9'),
        ('i < n ?', 4.0, 4.1, '#fdebd0', '#e67e22'),
        ('cur_sum > 0 ?', 4.0, 3.0, '#fdebd0', '#e67e22'),
        ('cur_sum += nums[i]\n接在后面', 2.2, 1.9, '#eafaf1', '#27ae60'),
        ('cur_sum = nums[i]\n重新开始', 5.8, 1.9, '#eafaf1', '#27ae60'),
        ('max_sum = max(max_sum, cur_sum)\ni++', 4.0, 0.8, '#eaf2f8', '#2980b9'),
    ]

    for text, x, y, facecolor, edgecolor in boxes:
        box = FancyBboxPatch((x - 1.5, y - 0.35), 3.0, 0.7,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=10, ha='center', va='center',
                fontweight='bold', color='#2c3e50', fontproperties=FONT_PROP)

    arrows = [
        ((4.0, 4.85), (4.0, 4.45)),
        ((4.0, 3.75), (4.0, 3.35)),
        ((2.5, 3.0), (2.2, 2.25)),
        ((5.5, 3.0), (5.8, 2.25)),
        ((2.2, 1.55), (4.0, 1.15)),
        ((5.8, 1.55), (4.0, 1.15)),
        ((4.0, 0.45), (4.0, -0.05)),  # loop back hidden, not used
    ]
    for start, end in arrows[:-1]:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # Loop back arrow to i < n
    ax.annotate('', xy=(5.5, 4.1), xytext=(5.5, 0.8),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5,
                               connectionstyle="arc3,rad=0.3"))

    ax.text(2.0, 3.5, 'Yes', fontsize=10, ha='center', color='#2ecc71',
            fontweight='bold', fontproperties=FONT_PROP)
    ax.text(6.0, 3.5, 'No', fontsize=10, ha='center', color='#e74c3c',
            fontweight='bold', fontproperties=FONT_PROP)
    ax.text(6.3, 2.5, 'Yes', fontsize=10, ha='center', color='#2ecc71',
            fontweight='bold', fontproperties=FONT_PROP)
    ax.text(1.7, 2.5, 'No', fontsize=10, ha='center', color='#e74c3c',
            fontweight='bold', fontproperties=FONT_PROP)

    return save_svg('maxsubarray_algorithm_flow.svg')


def draw_divide_conquer():
    """Divide and conquer view of maximum subarray."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    ax.set_title('分治法：最大子数组和只可能来自三个位置',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50',
                 fontproperties=FONT_PROP)

    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    n = len(nums)
    cell_w = 0.8
    start_x = 0.4
    y = 2.0
    mid = n // 2

    for i, v in enumerate(nums):
        x = start_x + i * cell_w
        color = '#fdebd0' if i == mid or i == mid + 1 else '#eaf2f8'
        rect = mpatches.Rectangle((x, y), cell_w, 0.7,
                                  facecolor=color, edgecolor='#2980b9',
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y + 0.35, str(v),
                fontsize=10, ha='center', va='center', color='#2c3e50',
                fontweight='bold', fontproperties=FONT_PROP)

    # Mid divider
    ax.plot([start_x + (mid + 1) * cell_w, start_x + (mid + 1) * cell_w],
            [y - 0.2, y + 0.9], color='#e67e22', linewidth=2, linestyle='--')
    ax.text(start_x + (mid + 0.5) * cell_w, y + 1.1, 'mid',
            fontsize=10, ha='center', va='bottom', color='#e67e22',
            fontweight='bold', fontproperties=FONT_PROP)

    # Branches
    branch_y = 3.3
    # Left
    ax.annotate('left_max', xy=(start_x + mid * cell_w / 2, y + 0.75),
                xytext=(start_x + mid * cell_w / 2, branch_y),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2),
                fontsize=11, ha='center', va='bottom', color='#3498db',
                fontweight='bold', fontproperties=FONT_PROP)
    # Right
    ax.annotate('right_max', xy=(start_x + (mid + 1 + (n - mid - 1) / 2) * cell_w, y + 0.75),
                xytext=(start_x + (mid + 1 + (n - mid - 1) / 2) * cell_w, branch_y),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2),
                fontsize=11, ha='center', va='bottom', color='#3498db',
                fontweight='bold', fontproperties=FONT_PROP)
    # Cross
    ax.annotate('cross_max', xy=(start_x + (mid + 0.5) * cell_w, y + 0.75),
                xytext=(start_x + (mid + 0.5) * cell_w, branch_y + 0.4),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2),
                fontsize=11, ha='center', va='bottom', color='#e74c3c',
                fontweight='bold', fontproperties=FONT_PROP)

    # Result box
    box = FancyBboxPatch((1.5, 0.4), 5.0, 0.6,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(box)
    ax.text(4.0, 0.7, 'max(left_max, right_max, cross_max)',
            fontsize=12, ha='center', va='center', fontweight='bold',
            family='monospace')

    return save_svg('maxsubarray_divide_conquer.svg')


def main():
    paths = [
        draw_prefix_sum(),
        draw_kadane(),
        draw_algorithm_flow(),
        draw_divide_conquer(),
    ]
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
