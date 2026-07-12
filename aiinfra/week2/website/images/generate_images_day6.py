#!/usr/bin/env python3
"""Generate SVG diagrams for Week 2 Day 6 (Integrated GEMM).

All diagrams follow the repo's hand-drawn sketch style (Excalidraw-like):
white background, rough turbulence filter on shapes, 3-4 soft accent colors,
hand-writing font family, crisp (un-filtered) text.
"""

from pathlib import Path

FONT = "'Comic Sans MS', 'Segoe UI', 'Kaiti SC', 楷体, cursive"

DEFS = """  <defs>
    <filter id="rough2">
      <feTurbulence type="fractalNoise" baseFrequency="0.025" numOctaves="2" seed="7"/>
      <feDisplacementMap in="SourceGraphic" scale="1.5"/>
    </filter>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/>
    </marker>
  </defs>"""


def save_svg(filename: str, content: str) -> None:
    path = Path(__file__).parent / filename
    path.write_text(content, encoding="utf-8")
    print(f"Generated: {path}")


def gemm_optimization_layers() -> str:
    """优化层次阶梯：cuBLAS(100%) 在顶端，Naive(1%) 在底端，左侧手绘上箭头表"性能↑"。"""
    # 每层: (name, desc, pct, accent, tint, dashed)
    # 从顶到底：cuBLAS -> Naive
    layers = [
        ("cuBLAS（NVIDIA 官方优化）", "PTX 内联 + 完整流水线 + Tensor Core + auto-tuning", "100%", "#888", "#f6f6f6", False),
        ("+ Tensor Core / CUTLASS (进阶)", "WMMA 指令，矩阵乘加硬件加速", "~90%+", "#4a7a3a", "#e6f4ea", True),
        ("+ 参数 Auto-tuning (进阶)", "针对不同尺寸选最优 BM/BN/BK/TM/TN", "~80%+", "#4a7a3a", "#e6f4ea", True),
        ("+ Double Buffering", "软件流水线，计算掩盖传输延迟", "~70%", "#446688", "#e8f0fe", False),
        ("+ Warp Shuffle 写回优化 (Day 6)", "Warp 内协作，减少非合并访问", "~60%", "#446688", "#e8f0fe", False),
        ("+ float4 向量化加载 (Day 6)", "128-bit load，提升带宽利用率", "~55%", "#446688", "#e8f0fe", False),
        ("+ Register Blocking (Day 2)", "TM×TN thread tile，累加器驻留寄存器", "~45%", "#d6a040", "#fff8e1", False),
        ("+ Shared Memory Tiling", "A/B tile 预取到 Shared Memory，K 维复用", "~15%", "#d6a040", "#fff8e1", False),
        ("Naive GEMM", "每线程算 1 元素，直接访问 Global Memory", "~1%", "#b85450", "#fce4ec", False),
    ]
    bar_w, bar_h, gap = 600, 46, 8
    step = bar_h + gap
    x0 = 90
    top_y = 72  # 第一条（cuBLAS）的 y
    rows = []
    for i, (name, desc, pct, accent, tint, dashed) in enumerate(layers):
        y = top_y + i * step
        dash = ' stroke-dasharray="5,3"' if dashed else ""
        sw = "2" if i == 0 else "1.5"
        rows.append(f"""    <g transform="translate({x0}, {y})">
      <rect x="0" y="0" width="{bar_w}" height="{bar_h}" rx="8" fill="{tint}" stroke="{accent}" stroke-width="{sw}"{dash} filter="url(#rough2)"/>
      <rect x="0" y="0" width="6" height="{bar_h}" fill="{accent}" opacity="0.6" rx="3" filter="url(#rough2)"/>
      <text x="18" y="29" font-size="13" fill="#444" font-weight="bold">{name}</text>
      <text x="210" y="29" font-size="11" fill="#777">{desc}</text>
      <text x="572" y="30" text-anchor="middle" font-size="16" fill="{accent}" font-weight="bold">{pct}</text>
    </g>""")

    # Day 2 marker -> Register Blocking (index 6 from top) center
    day2_y = top_y + 6 * step + bar_h / 2
    # Day 6 marker -> Double Buffering (index 3 from top) center
    day6_y = top_y + 3 * step + bar_h / 2
    body = "\n".join(rows)
    last_y = top_y + 8 * step + bar_h
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 580" font-family="{FONT}">
{DEFS}

  <rect width="780" height="580" fill="#fafafa"/>

  <text x="390" y="34" text-anchor="middle" font-size="18" fill="#444" font-weight="bold">GEMM 优化层次：从 1% 到 70%+</text>
  <text x="390" y="52" text-anchor="middle" font-size="11" fill="#888">每叠加一层优化，cuBLAS 占比提升一档</text>

  <!-- 左侧手绘上箭头：性能方向（底=Naive 1%，顶=cuBLAS 100%）-->
  <line x1="52" y1="{last_y - 6}" x2="52" y2="{top_y + 6}" stroke="#888" stroke-width="1.6" marker-end="url(#arr)" filter="url(#rough2)"/>
  <text x="40" y="{(top_y + last_y) // 2}" font-size="12" fill="#888" font-weight="bold" transform="rotate(-90, 40, {(top_y + last_y) // 2})" text-anchor="middle">性能 ↑</text>

{body}

  <!-- Day 2 / Day 6 右侧标记 -->
  <line x1="700" y1="{day2_y}" x2="742" y2="{day2_y}" stroke="#d6a040" stroke-width="1.6" filter="url(#rough2)"/>
  <text x="752" y="{day2_y + 4}" font-size="10" fill="#d6a040" font-weight="bold" transform="rotate(90, 752, {day2_y + 4})" text-anchor="middle">Day 2</text>

  <line x1="700" y1="{day6_y}" x2="742" y2="{day6_y}" stroke="#446688" stroke-width="1.6" filter="url(#rough2)"/>
  <text x="752" y="{day6_y + 4}" font-size="10" fill="#446688" font-weight="bold" transform="rotate(90, 752, {day6_y + 4})" text-anchor="middle">Day 6</text>

  <text x="390" y="562" text-anchor="middle" font-size="12" fill="#888">Day 6 目标：从 ~45%（Day 2）跨越到 70%+</text>
</svg>"""


def float4_vectorized_load() -> str:
    """左：逐元素 4×32-bit（红）；右：float4 1×128-bit（绿）；下方三条件；底栏收益。"""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 460" font-family="{FONT}">
{DEFS}

  <rect width="760" height="460" fill="#fafafa"/>

  <text x="380" y="32" text-anchor="middle" font-size="18" fill="#444" font-weight="bold">float4 向量化加载 vs 逐元素加载</text>

  <!-- 左：逐元素加载 -->
  <g transform="translate(40, 60)">
    <text x="0" y="0" font-size="13" fill="#b85450" font-weight="bold">逐元素加载：4 条 32-bit load</text>
    <rect x="0" y="10" width="320" height="150" rx="8" fill="#fce4ec" stroke="#b85450" stroke-width="1.5" filter="url(#rough2)"/>
    <text x="14" y="38" font-family="monospace" font-size="12" fill="#444">float a0 = ptr[0];  // 32-bit</text>
    <text x="14" y="58" font-family="monospace" font-size="12" fill="#444">float a1 = ptr[1];  // 32-bit</text>
    <text x="14" y="78" font-family="monospace" font-size="12" fill="#444">float a2 = ptr[2];  // 32-bit</text>
    <text x="14" y="98" font-family="monospace" font-size="12" fill="#444">float a3 = ptr[3];  // 32-bit</text>
    <rect x="14" y="118" width="130" height="28" rx="6" fill="#b85450" opacity="0.2" stroke="#b85450" stroke-width="1.2" filter="url(#rough2)"/>
    <text x="79" y="137" text-anchor="middle" font-size="12" fill="#b85450" font-weight="bold">4 条指令</text>
  </g>

  <!-- 右：float4 加载 -->
  <g transform="translate(400, 60)">
    <text x="0" y="0" font-size="13" fill="#4a7a3a" font-weight="bold">float4 加载：1 条 128-bit load</text>
    <rect x="0" y="10" width="320" height="150" rx="8" fill="#e6f4ea" stroke="#4a7a3a" stroke-width="1.5" filter="url(#rough2)"/>
    <text x="14" y="38" font-family="monospace" font-size="12" fill="#444">float4 val =</text>
    <text x="14" y="58" font-family="monospace" font-size="12" fill="#444">  reinterpret_cast&lt;const float4*&gt;(ptr)[0];</text>
    <text x="14" y="84" font-size="11" fill="#888">// val.x=ptr[0]  val.y=ptr[1]</text>
    <text x="14" y="102" font-size="11" fill="#888">// val.z=ptr[2]  val.w=ptr[3]</text>
    <rect x="176" y="118" width="130" height="28" rx="6" fill="#4a7a3a" opacity="0.2" stroke="#4a7a3a" stroke-width="1.2" filter="url(#rough2)"/>
    <text x="241" y="137" text-anchor="middle" font-size="12" fill="#4a7a3a" font-weight="bold">1 条指令</text>
  </g>

  <!-- 中间对比箭头 -->
  <text x="380" y="135" text-anchor="middle" font-size="22" fill="#888">→</text>

  <!-- 使用条件 -->
  <g transform="translate(40, 240)">
    <rect x="0" y="0" width="680" height="120" rx="8" fill="#f6f6f6" stroke="#888" stroke-width="1.5" filter="url(#rough2)"/>
    <rect x="0" y="0" width="680" height="26" fill="#446688" opacity="0.12" rx="8" filter="url(#rough2)"/>
    <text x="340" y="18" text-anchor="middle" font-size="13" fill="#446688" font-weight="bold">使用条件</text>
    <text x="20" y="52" font-size="12" fill="#4a7a3a" font-weight="bold">✓ 地址 16 字节对齐</text>
    <text x="210" y="52" font-size="11" fill="#777">cudaMalloc 分配的内存天然对齐</text>
    <text x="20" y="78" font-size="12" fill="#4a7a3a" font-weight="bold">✓ Coalesced 访问模式</text>
    <text x="210" y="78" font-size="11" fill="#777">连续线程访问连续地址，合并为最少 cache line</text>
    <text x="20" y="104" font-size="12" fill="#4a7a3a" font-weight="bold">✓ 数据布局支持</text>
    <text x="210" y="104" font-size="11" fill="#777">行优先矩阵的连续行元素天然连续</text>
  </g>

  <!-- 收益底栏 -->
  <rect x="150" y="385" width="460" height="42" rx="8" fill="#e8f0fe" stroke="#446688" stroke-width="1.6" filter="url(#rough2)"/>
  <text x="380" y="412" text-anchor="middle" font-size="13" fill="#446688" font-weight="bold">收益：Global Memory 带宽利用率提升 10-15%</text>
</svg>"""


def parameter_tuning_table() -> str:
    """手绘扫描表：TM×TN 区（8×8 基准 / 8×16 / 16×8 / 16×16 SPILL）+ BK 区。"""
    # 行: (label, c1, c2, c3, reg, accent, tint, bold)
    tm_rows = [
        ("8×8",   "基准",  "基准",  "基准",  "~88",  "#446688", "#e8f0fe", True),
        ("8×16",  "+5%",  "+8%",  "+10%", "~152", "#4a7a3a", "#f6f6f6", False),
        ("16×8",  "+3%",  "+5%",  "+8%",  "~152", "#4a7a3a", "#f6f6f6", False),
        ("16×16", "SPILL!","SPILL!","SPILL!","~256","#b85450", "#fce4ec", True),
    ]
    bk_rows = [
        ("BK=4",  "-2%", "+3%", "+5%",  "smem↓", "#4a7a3a", "#f6f6f6", False),
        ("BK=8",  "基准", "基准", "基准", "~88",   "#446688", "#e8f0fe", True),
        ("BK=16", "+1%", "+2%", "+3%",  "smem↑", "#d6a040", "#fff8e1", False),
    ]

    x0, y0 = 40, 64
    tw = 680
    rh = 34
    col_x = [80, 200, 340, 480, 620]  # 列中心（相对表内）

    def row_svg(y, label, c1, c2, c3, reg, accent, tint, bold, label_accent=None):
        if label_accent is None:
            label_accent = accent
        fw = ' font-weight="bold"' if bold else ""
        return f"""    <g transform="translate({x0}, {y})">
      <rect x="0" y="0" width="{tw}" height="{rh}" fill="{tint}" stroke="#ddd" stroke-width="1" filter="url(#rough2)"/>
      <text x="{col_x[0]}" y="22" text-anchor="middle" font-size="12" fill="{label_accent}"{fw}>{label}</text>
      <text x="{col_x[1]}" y="22" text-anchor="middle" font-size="12" fill="{'#b85450' if 'SPILL' in c1 else accent}"{fw}>{c1}</text>
      <text x="{col_x[2]}" y="22" text-anchor="middle" font-size="12" fill="{'#b85450' if 'SPILL' in c2 else accent}"{fw}>{c2}</text>
      <text x="{col_x[3]}" y="22" text-anchor="middle" font-size="12" fill="{'#b85450' if 'SPILL' in c3 else accent}"{fw}>{c3}</text>
      <text x="{col_x[4]}" y="22" text-anchor="middle" font-size="12" fill="{'#b85450' if '256' in reg else '#d6a040' if '152' in reg else accent}"{fw}>{reg}</text>
    </g>"""

    # 表头
    header = f"""    <g transform="translate({x0}, {y0})">
      <rect x="0" y="0" width="{tw}" height="{rh}" fill="#eee" stroke="#bbb" stroke-width="1.3" filter="url(#rough2)"/>
      <text x="{col_x[0]}" y="22" text-anchor="middle" font-size="12" fill="#444" font-weight="bold">TM×TN</text>
      <text x="{col_x[1]}" y="22" text-anchor="middle" font-size="12" fill="#444" font-weight="bold">1024 矩阵</text>
      <text x="{col_x[2]}" y="22" text-anchor="middle" font-size="12" fill="#444" font-weight="bold">2048 矩阵</text>
      <text x="{col_x[3]}" y="22" text-anchor="middle" font-size="12" fill="#444" font-weight="bold">4096 矩阵</text>
      <text x="{col_x[4]}" y="22" text-anchor="middle" font-size="12" fill="#444" font-weight="bold">Register</text>
    </g>"""

    tm_y = y0 + rh
    tm_blocks = []
    for i, r in enumerate(tm_rows):
        tm_blocks.append(row_svg(tm_y + i * rh, *r))

    # 分隔标题行
    sep_y = tm_y + len(tm_rows) * rh
    sep = f"""    <g transform="translate({x0}, {sep_y})">
      <rect x="0" y="0" width="{tw}" height="{rh}" fill="#eee" stroke="#bbb" stroke-width="1.3" filter="url(#rough2)"/>
      <text x="{tw // 2}" y="22" text-anchor="middle" font-size="12" fill="#444" font-weight="bold">BK 扫描（固定 TM=TN=8）</text>
    </g>"""

    bk_y = sep_y + rh
    bk_blocks = []
    for i, r in enumerate(bk_rows):
        bk_blocks.append(row_svg(bk_y + i * rh, *r))

    last_y = bk_y + len(bk_rows) * rh
    total_h = last_y + 50
    tm_txt = "\n".join(tm_blocks)
    bk_txt = "\n".join(bk_blocks)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {total_h}" font-family="{FONT}">
{DEFS}

  <rect width="760" height="{total_h}" fill="#fafafa"/>

  <text x="380" y="34" text-anchor="middle" font-size="18" fill="#444" font-weight="bold">参数精调扫描表</text>
  <text x="380" y="52" text-anchor="middle" font-size="11" fill="#888">绿=正向收益，红=溢出/负向，橙=寄存器偏高</text>

{header}

{tm_txt}

{sep}

{bk_txt}

  <text x="380" y="{last_y + 28}" text-anchor="middle" font-size="12" fill="#888">精调步骤：先扫 TM×TN → 再扫 BK → 最后扫 BM/BN</text>
</svg>"""


def main() -> None:
    diagrams = {
        "gemm_optimization_layers.svg": gemm_optimization_layers(),
        "float4_vectorized_load.svg": float4_vectorized_load(),
        "parameter_tuning_table.svg": parameter_tuning_table(),
    }
    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
