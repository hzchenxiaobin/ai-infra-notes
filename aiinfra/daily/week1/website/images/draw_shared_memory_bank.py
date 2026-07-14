#!/usr/bin/env python3
"""
Draw a sketch-style Shared Memory Bank structure diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Enable xkcd hand-drawn style for lines/boxes
plt.xkcd()

# Use a Chinese-compatible font while keeping sketch style
plt.rcParams['font.family'] = ['Hiragino Sans GB', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.6, 'Shared Memory Bank 结构', fontsize=26, ha='center', va='top', weight='bold')
ax.text(7, 9.15, '32 个 bank，每个 bank 4 bytes（float 类型）', fontsize=14, ha='center', va='top')

# Draw 32 bank boxes at the top
bank_box_w = 0.35
bank_box_h = 0.6
start_x = 1.0
start_y = 7.7
num_banks = 32
gap = 0.05

for i in range(num_banks):
    x = start_x + i * (bank_box_w + gap)
    rect = FancyBboxPatch((x, start_y), bank_box_w, bank_box_h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor='#E3F2FD' if i % 2 == 0 else '#BBDEFB',
                          edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    if i % 4 == 0 or i == 31:
        ax.text(x + bank_box_w / 2, start_y + bank_box_h / 2, f'B{i}',
                fontsize=9, ha='center', va='center', weight='bold')

# Labels for first and last
ax.text(start_x + bank_box_w / 2, start_y - 0.25, 'Bank 0', fontsize=10, ha='center')
ax.text(start_x + (num_banks - 1) * (bank_box_w + gap) + bank_box_w / 2, start_y - 0.25, 'Bank 31', fontsize=10, ha='center')
ax.text(7, start_y + bank_box_h + 0.45, '一个 warp 的 32 个线程可以并行访问 32 个不同 bank', fontsize=12, ha='center', style='italic')

# Formula box
formula_y = 6.5
rect = FancyBboxPatch((2.5, formula_y - 0.5), 9, 1.0,
                      boxstyle="round,pad=0.1,rounding_size=0.2",
                      facecolor='#FFF9E6', edgecolor='black', linewidth=2)
ax.add_patch(rect)
ax.text(7, formula_y, '地址到 bank 的映射：bank = (address / 4) % 32', fontsize=16, ha='center', va='center', weight='bold')

# Examples box
examples_y = 4.9
rect = FancyBboxPatch((1.0, examples_y - 1.05), 12, 1.9,
                      boxstyle="round,pad=0.1,rounding_size=0.2",
                      facecolor='#F0FFF0', edgecolor='black', linewidth=2)
ax.add_patch(rect)
ax.text(1.5, examples_y + 0.6, '举例：', fontsize=14, weight='bold')
ax.text(1.7, examples_y + 0.18, '地址 0, 128, 256 ...  →  Bank 0    （因为 (0/4)%32=0, (128/4)%32=0）', fontsize=12)
ax.text(1.7, examples_y - 0.24, '地址 4, 132, 260 ...  →  Bank 1    （因为 (4/4)%32=1, (132/4)%32=1）', fontsize=12)
ax.text(1.7, examples_y - 0.66, '地址 i × 4 且 i % 32 == k  →  Bank k', fontsize=12)

# Conflict patterns
pattern_y = 2.5
ax.text(7, pattern_y + 0.85, '三种访问模式', fontsize=16, ha='center', weight='bold')

patterns = [
    ('不同 bank', '#C8E6C9', '无 Conflict', '每个线程访问不同 bank'),
    ('同一地址', '#A5D6A7', 'Broadcast 无 Conflict', '硬件广播，1 个 cycle'),
    ('同 bank 不同地址', '#FFCDD2', 'Bank Conflict', '需要串行访问，性能下降'),
]

box_w = 3.8
box_h = 1.45
x_positions = [0.8, 5.1, 9.4]

for x, (title, color, result, desc) in zip(x_positions, patterns):
    rect = FancyBboxPatch((x, pattern_y - box_h + 0.2), box_w, box_h,
                          boxstyle="round,pad=0.08,rounding_size=0.15",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + box_w / 2, pattern_y - 0.05, title, fontsize=13, ha='center', weight='bold')
    ax.text(x + box_w / 2, pattern_y - 0.42, result, fontsize=12, ha='center')
    ax.text(x + box_w / 2, pattern_y - 0.78, desc, fontsize=10, ha='center', color='#444444')

# Bottom note
ax.text(7, 0.35, 'Bank Conflict 定义：同一个 warp 内多个线程同时访问同一个 bank 的不同地址', fontsize=12, ha='center', style='italic')

plt.tight_layout()
plt.savefig('/Users/chenbinbin/GitHub/aiinfra/week1/website/images/shared_memory_bank_structure.svg', format='svg', bbox_inches='tight', dpi=150)
plt.savefig('/Users/chenbinbin/GitHub/aiinfra/week1/website/images/shared_memory_bank_structure.png', format='png', bbox_inches='tight', dpi=150)
print('Saved shared_memory_bank_structure.svg and .png')
