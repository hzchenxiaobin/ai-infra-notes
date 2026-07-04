#!/usr/bin/env python3
"""Generate figures for the LeetGPU matrix multiplication solution note."""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent


def draw_block(color, x, y, width, height, ax, label="", text_color="white", fontsize=9, alpha=1.0, edgecolor="black", lw=1.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color, edgecolor=edgecolor, linewidth=lw, alpha=alpha
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=fontsize, color=text_color, weight="bold")


def draw_arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_matrix_outline(ax, x, y, n_rows, n_cols, cell_size, label="", color="#E5E7EB", lw=1.5):
    """Draw a matrix grid outline with cells."""
    width = n_cols * cell_size
    height = n_rows * cell_size
    rect = mpatches.Rectangle((x, y - height), width, height, fill=False, edgecolor="#374151", linewidth=lw)
    ax.add_patch(rect)
    # grid lines
    for i in range(1, n_rows):
        ax.plot([x, x + width], [y - i * cell_size, y - i * cell_size], color=color, lw=0.8)
    for j in range(1, n_cols):
        ax.plot([x + j * cell_size, x + j * cell_size], [y, y - height], color=color, lw=0.8)
    if label:
        ax.text(x + width / 2, y + 0.25, label, ha="center", va="bottom", fontsize=11, weight="bold")


def highlight_cells(ax, x, y, n_rows, n_cols, cell_size, indices, color, alpha=0.7, label=None):
    """Highlight specific cells in a matrix grid. indices: list of (row, col)."""
    for r, c in indices:
        rect = mpatches.Rectangle(
            (x + c * cell_size, y - (r + 1) * cell_size),
            cell_size, cell_size,
            facecolor=color, edgecolor="#374151", linewidth=1, alpha=alpha
        )
        ax.add_patch(rect)
    if label:
        # place label near first cell
        r, c = indices[0]
        ax.text(x + (c + 0.5) * cell_size, y - (r + 0.5) * cell_size, label,
                ha="center", va="center", fontsize=8, color="white", weight="bold")


def generate_naive_flow():
    """Naive GEMM: each thread computes one C element by reading row of A and column of B."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(6, 4.7, "Naive GEMM：每个线程计算 C 的一个元素", ha="center", va="center", fontsize=15, weight="bold")

    # Matrix A (M x K), show one row highlighted
    cell = 0.35
    ax_A = 0.8
    ay_A = 3.6
    draw_matrix_outline(ax, ax_A, ay_A, 5, 6, cell, label="A (M×K)")
    highlight_cells(ax, ax_A, ay_A, 5, 6, cell, [(1, c) for c in range(6)], "#4C78A8", label="行 i")

    # Matrix B (K x N), show one column highlighted
    ax_B = 4.8
    ay_B = 3.6
    draw_matrix_outline(ax, ax_B, ay_B, 6, 5, cell, label="B (K×N)")
    highlight_cells(ax, ax_B, ay_B, 6, 5, cell, [(r, 2) for r in range(6)], "#F58518", label="列 j")

    # Matrix C (M x N), show one element highlighted
    ax_C = 8.8
    ay_C = 3.6
    draw_matrix_outline(ax, ax_C, ay_C, 5, 5, cell, label="C (M×N)")
    highlight_cells(ax, ax_C, ay_C, 5, 5, cell, [(1, 2)], "#59A14F", label="C[i][j]")

    # Arrows
    draw_arrow(ax, ax_A + 6 * cell + 0.15, ay_A - (1 + 0.5) * cell, ax_C + 2 * cell - 0.15, ay_C - 0.5 * cell, color="#4C78A8", lw=2)
    draw_arrow(ax, ax_B + (2 + 0.5) * cell, ay_B - 6 * cell - 0.15, ax_C + 2.5 * cell, ay_C - (1 + 1) * cell + 0.15, color="#F58518", lw=2)

    # Formula
    ax.text(6, 1.5,
            r"$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \times B[k][j]$" + "\n"
            "每个线程：读 A 的一整行 + B 的一整列 → 写 C 的一个元素",
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    # Bottleneck note
    ax.text(6, 0.6,
            "瓶颈：A 的同一行 / B 的同一列被多个线程重复读取，全局内存访问次数多",
            ha="center", va="center", fontsize=10, color="#991B1B",
            bbox=dict(boxstyle="round", facecolor="#FEE2E2", edgecolor="#EF4444"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matmul_naive.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_tiled_flow():
    """Shared memory tiling: blocks load tiles of A/B and accumulate over K phases."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7, 5.7, "Shared Memory Tiling：分块加载 + K 维复用", ha="center", va="center", fontsize=15, weight="bold")

    cell = 0.35

    # Matrix A with a tile highlighted
    ax_A = 0.8
    ay_A = 4.4
    draw_matrix_outline(ax, ax_A, ay_A, 6, 8, cell, label="A (M×K)")
    # Highlight one TILE x TILE block: rows 1-3, cols 2-4
    tile_a_indices = [(r, c) for r in range(1, 4) for c in range(2, 5)]
    highlight_cells(ax, ax_A, ay_A, 6, 8, cell, tile_a_indices, "#4C78A8", label="A tile")

    # Matrix B with a tile highlighted
    ax_B = 4.8
    ay_B = 4.4
    draw_matrix_outline(ax, ax_B, ay_B, 8, 6, cell, label="B (K×N)")
    tile_b_indices = [(r, c) for r in range(2, 5) for c in range(1, 4)]
    highlight_cells(ax, ax_B, ay_B, 8, 6, cell, tile_b_indices, "#F58518", label="B tile")

    # Matrix C with one output tile highlighted
    ax_C = 9.6
    ay_C = 4.4
    draw_matrix_outline(ax, ax_C, ay_C, 6, 6, cell, label="C (M×N)")
    tile_c_indices = [(r, c) for r in range(1, 4) for c in range(1, 4)]
    highlight_cells(ax, ax_C, ay_C, 6, 6, cell, tile_c_indices, "#59A14F", label="C tile")

    # Shared memory boxes in the middle
    sx = 4.3
    sy = 1.8
    # A tile in SMEM
    draw_block("#4C78A8", sx, sy, 2.2, 1.2, ax, label="s_A[TILE][TILE]\n(Shared Memory)", fontsize=9)
    # B tile in SMEM
    draw_block("#F58518", sx + 2.8, sy, 2.2, 1.2, ax, label="s_B[TILE][TILE]\n(Shared Memory)", fontsize=9)

    # Arrows from global tiles to shared memory
    draw_arrow(ax, ax_A + (2 + 1.5) * cell, ay_A - (1 + 3) * cell - 0.15, sx + 1.1, sy + 1.2 + 0.1, color="#4C78A8", lw=2)
    draw_arrow(ax, ax_B + (1 + 1.5) * cell, ay_B - (2 + 3) * cell - 0.15, sx + 2.8 + 1.1, sy + 1.2 + 0.1, color="#F58518", lw=2)

    # Arrows from SMEM to C tile
    draw_arrow(ax, sx + 2.2 + 0.1, sy + 0.6, ax_C + (1 + 0.5) * cell - 0.2, ay_C - (1 + 3) * cell - 0.15, color="#59A14F", lw=2)

    # Phase annotation
    ax.text(7, 0.7,
            "① Block 协作把 A/B 的 TILE 加载到 Shared Memory\n"
            "② 每个线程用 s_A 的一行 × s_B 的一列累加部分和\n"
            "③ 沿 K 维滑动 tile，重复加载 → 累加，直到 K 遍历完成",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    # Phase labels near K dimension
    ax.text(2.5, 3.2, "K 维度分块", ha="center", fontsize=10, color="#4C78A8", weight="bold")
    ax.text(5.8, 3.2, "K 维度分块", ha="center", fontsize=10, color="#F58518", weight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matmul_tiled.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_thread_block_mapping():
    """Show how thread blocks and threads map to C tile."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(6, 5.2, "Thread / Block 映射：一个 Block 负责 C 的一个 TILE", ha="center", va="center", fontsize=15, weight="bold")

    cell = 0.55
    # C matrix with 4x4 blocks (tiled)
    n_tiles = 4
    tile_size = 3
    mx = 2.0
    my = 4.0
    width = n_tiles * tile_size * cell
    height = n_tiles * tile_size * cell
    rect = mpatches.Rectangle((mx, my - height), width, height, fill=False, edgecolor="#374151", linewidth=2)
    ax.add_patch(rect)

    # draw tile boundaries
    for i in range(1, n_tiles):
        ax.plot([mx, mx + width], [my - i * tile_size * cell, my - i * tile_size * cell], color="#9CA3AF", lw=1.5)
        ax.plot([mx + i * tile_size * cell, mx + i * tile_size * cell], [my, my - height], color="#9CA3AF", lw=1.5)

    # Label grid dims
    for bi in range(n_tiles):
        for bj in range(n_tiles):
            x = mx + bj * tile_size * cell
            y = my - bi * tile_size * cell
            rect = mpatches.Rectangle((x, y - tile_size * cell), tile_size * cell, tile_size * cell,
                                       facecolor="#E5E7EB", edgecolor="#6B7280", linewidth=1, alpha=0.6)
            ax.add_patch(rect)
            ax.text(x + tile_size * cell / 2, y - tile_size * cell / 2,
                    f"Block\n({bj},{bi})", ha="center", va="center", fontsize=8, color="#374151")

    ax.text(mx + width / 2, my + 0.25, "C 矩阵 (M×N)", ha="center", fontsize=11, weight="bold")

    # Zoom into Block (1,1): second column, second row
    target_bj, target_bi = 1, 1
    zx = 8.6
    zy = 4.0
    zoom_size = tile_size * cell
    rect = mpatches.Rectangle((zx, zy - zoom_size), zoom_size, zoom_size, fill=False, edgecolor="#374151", linewidth=2)
    ax.add_patch(rect)

    # Internal threads
    for ti in range(tile_size):
        for tj in range(tile_size):
            x = zx + tj * cell
            y = zy - ti * cell
            color = "#4C78A8" if (ti, tj) == (1, 1) else "#F58518"
            rect = mpatches.Rectangle((x, y - cell), cell, cell,
                                       facecolor=color, edgecolor="white", linewidth=1, alpha=0.85)
            ax.add_patch(rect)
            ax.text(x + cell / 2, y - cell / 2, f"T\n({tj},{ti})", ha="center", va="center", fontsize=8, color="white", weight="bold")

    ax.text(zx + zoom_size / 2, zy + 0.25, f"Block ({target_bj},{target_bi}) 内部：TILE_SIZE×TILE_SIZE 线程", ha="center", fontsize=11, weight="bold")

    # Arrow from Block (1,1) center to zoom box
    src_x = mx + (target_bj + 0.5) * tile_size * cell
    src_y = my - (target_bi + 0.5) * tile_size * cell
    dst_x = zx - 0.1
    dst_y = zy - zoom_size / 2
    draw_arrow(ax, src_x, src_y, dst_x, dst_y, color="#374151", lw=2)

    # Mapping formula (placed below arrow to avoid overlap)
    ax.text(6, 0.65,
            "row = blockIdx.y × TILE_SIZE + threadIdx.y\n"
            "col = blockIdx.x × TILE_SIZE + threadIdx.x\n"
            "每个线程负责 C[row][col] 的最终累加结果",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#FEF3C7", edgecolor="#F59E0B"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matmul_thread_block_mapping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_naive_flow()
    generate_tiled_flow()
    generate_thread_block_mapping()
    print(f"Generated matrix multiplication figures in {OUT_DIR}")
