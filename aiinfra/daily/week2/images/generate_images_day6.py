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


def cache_line_sector() -> str:
    """Cache Line (128B) 与 Sector (32B)：结构关系 + 合并访问 vs 散乱访问对比。"""
    # ---- Part A: 1 cache line = 4 sectors ----
    cl_x, cl_y, cl_w, cl_h = 90, 90, 600, 64
    gap = 4
    sw = (cl_w - 5 * gap) / 4  # 145
    sec_a = []
    for i in range(4):
        sx = cl_x + gap + i * (sw + gap)
        cx = sx + sw / 2
        sec_a.append(
            f'    <rect x="{sx:.1f}" y="{cl_y + 10}" width="{sw:.1f}" height="{cl_h - 20}" rx="4" '
            f'fill="#d4e6f7" stroke="#446688" stroke-width="1.3" filter="url(#rough2)"/>\n'
            f'    <text x="{cx:.1f}" y="{cl_y + 30}" text-anchor="middle" font-size="11" '
            f'fill="#446688" font-weight="bold">Sector {i}</text>\n'
            f'    <text x="{cx:.1f}" y="{cl_y + 49}" text-anchor="middle" font-size="11" fill="#446688">32 B</text>'
        )
    sec_a_svg = "\n".join(sec_a)

    # ---- Part B: 32 thread cells (coalesced) ----
    band_x, band_y, band_w = 90, 224, 600
    cw = band_w / 32
    cells_b = []
    for i in range(32):
        cx = band_x + i * cw
        cells_b.append(
            f'<rect x="{cx:.2f}" y="{band_y}" width="{cw:.2f}" height="22" '
            f'fill="#cde9d4" stroke="#4a7a3a" stroke-width="0.5"/>'
        )
    cells_b_svg = "\n".join(cells_b)
    brackets_b = []
    for i in range(4):
        bx = band_x + i * (band_w / 4)
        bw = band_w / 4
        brackets_b.append(
            f'<rect x="{bx:.1f}" y="{band_y + 28}" width="{bw:.1f}" height="20" rx="3" '
            f'fill="#4a7a3a" opacity="0.15" stroke="#4a7a3a" stroke-width="1.1" filter="url(#rough2)"/>'
        )
    brackets_b_svg = "\n".join(brackets_b)

    # ---- Part C: 8 cache lines × 4 sectors grid (scattered) ----
    gx0, gy0 = 90, 388
    gsw, gsh = 147, 18
    hgap, vgap = 4, 3
    grid_c = []
    for r in range(8):
        for c in range(4):
            sx = gx0 + c * (gsw + hgap)
            sy = gy0 + r * (gsh + vgap)
            grid_c.append(
                f'<rect x="{sx}" y="{sy}" width="{gsw}" height="{gsh}" rx="3" '
                f'fill="#fce4ec" stroke="#b85450" stroke-width="0.8" filter="url(#rough2)"/>'
            )
            grid_c.append(
                f'<circle cx="{sx + gsw/2}" cy="{sy + gsh/2}" r="3.2" fill="#b85450"/>'
            )
    grid_c_svg = "\n".join(grid_c)
    grid_top = gy0
    grid_bot = gy0 + 7 * (gsh + vgap) + gsh
    grid_mid = (grid_top + grid_bot) // 2

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 610" font-family="{FONT}">
{DEFS}

  <rect width="780" height="610" fill="#fafafa"/>

  <text x="390" y="32" text-anchor="middle" font-size="18" fill="#444" font-weight="bold">Cache Line 与 Sector：GPU 访存的两级粒度</text>
  <text x="390" y="52" text-anchor="middle" font-size="11" fill="#888">cache line（128B）管存储组织 · sector（32B）管数据传输</text>

  <!-- Part A: 结构关系 -->
  <text x="390" y="80" text-anchor="middle" font-size="13" fill="#446688" font-weight="bold">1 Cache Line（128B）= 4 Sectors（32B × 4）</text>
  <rect x="{cl_x}" y="{cl_y}" width="{cl_w}" height="{cl_h}" rx="8" fill="#e8f0fe" stroke="#446688" stroke-width="1.8" filter="url(#rough2)"/>
{sec_a_svg}
  <text x="390" y="{cl_y + cl_h + 18}" text-anchor="middle" font-size="11" fill="#888">按 sector 粒度填充：只触达 1 个 sector 就只搬 1 个，不必整行搬运</text>

  <!-- Part B: 合并访问 -->
  <text x="40" y="204" font-size="13" fill="#4a7a3a" font-weight="bold">✓ 合并访问 Coalesced</text>
  <text x="40" y="220" font-size="11" fill="#888">warp 32 线程读连续 float：32 × 4B = 128B</text>
  <text x="{band_x + band_w + 6}" y="{band_y + 15}" font-size="10" fill="#4a7a3a">32 线程</text>
{cells_b_svg}
{brackets_b_svg}
  <text x="390" y="{band_y + 62}" text-anchor="middle" font-size="10" fill="#4a7a3a">恰好 1 条 cache line = 4 sector</text>
  <rect x="230" y="296" width="320" height="32" rx="16" fill="#e6f4ea" stroke="#4a7a3a" stroke-width="1.4" filter="url(#rough2)"/>
  <text x="390" y="317" text-anchor="middle" font-size="12" fill="#4a7a3a" font-weight="bold">1 次事务 · 传 128B · 利用率 100%</text>

  <!-- Part C: 散乱访问 -->
  <text x="40" y="366" font-size="13" fill="#b85450" font-weight="bold">✗ 散乱访问 Strided/Scattered</text>
  <text x="40" y="382" font-size="11" fill="#888">32 线程地址各落一个 sector：8 cache line × 4 sector = 32 sector</text>
  <line x1="78" y1="{grid_top}" x2="78" y2="{grid_bot}" stroke="#b85450" stroke-width="1.4" filter="url(#rough2)"/>
  <text x="70" y="{grid_mid}" text-anchor="middle" font-size="10" fill="#b85450" font-weight="bold" transform="rotate(-90, 70, {grid_mid})">32 sector · 1024B</text>
{grid_c_svg}
  <rect x="190" y="568" width="400" height="32" rx="16" fill="#fce4ec" stroke="#b85450" stroke-width="1.4" filter="url(#rough2)"/>
  <text x="390" y="589" text-anchor="middle" font-size="12" fill="#b85450" font-weight="bold">32 次事务 · 传 1024B · 有效 128B · 利用率 12.5%</text>
</svg>"""


def l2_cache_line_management() -> str:
    """「L2 以 128B cache line 管理」的含义：tag/命中/分配按行（128B），填充/传输按 sector（32B），
    每 sector 独立 valid bit；含地址拆解与两次访问示例。"""

    def line(x, y, w, h, filled, accent, tint):
        """一条 cache line：Tag 单元 + 4 个带 valid bit 的 sector。"""
        parts = [
            f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{tint}" '
            f'stroke="{accent}" stroke-width="1.5" filter="url(#rough2)"/>'
        ]
        gap = 8
        tag_w = 110
        tx, ty, th = x + 10, y + 10, h - 20
        parts.append(
            f'    <rect x="{tx}" y="{ty}" width="{tag_w}" height="{th}" rx="4" '
            f'fill="{accent}" opacity="0.35" stroke="{accent}" stroke-width="1.2" filter="url(#rough2)"/>'
        )
        parts.append(f'    <text x="{tx + tag_w/2}" y="{y + h/2 - 2}" text-anchor="middle" font-size="11" fill="{accent}" font-weight="bold">Tag</text>')
        parts.append(f'    <text x="{tx + tag_w/2}" y="{y + h/2 + 14}" text-anchor="middle" font-size="9" fill="{accent}">地址高位</text>')
        sx0 = tx + tag_w + gap
        right = x + w - 10
        sw = (right - sx0 - 3 * gap) / 4
        for i in range(4):
            sx = sx0 + i * (sw + gap)
            cx = sx + sw / 2
            if filled[i]:
                parts.append(
                    f'    <rect x="{sx:.1f}" y="{ty}" width="{sw:.1f}" height="{th}" rx="4" '
                    f'fill="{accent}" opacity="0.30" stroke="{accent}" stroke-width="1.2" filter="url(#rough2)"/>'
                )
                vtxt, vfill, vextra = "V=1", accent, ""
            else:
                parts.append(
                    f'    <rect x="{sx:.1f}" y="{ty}" width="{sw:.1f}" height="{th}" rx="4" '
                    f'fill="#ffffff" stroke="{accent}" stroke-width="0.8" stroke-dasharray="3,2" opacity="0.6"/>'
                )
                vtxt, vfill, vextra = "V=0", "#bbb", "（空）"
            sfill = accent if filled[i] else "#bbb"
            parts.append(f'    <text x="{cx:.1f}" y="{y + h/2 - 2}" text-anchor="middle" font-size="10" fill="{sfill}" font-weight="bold">S{i} · 32B</text>')
            parts.append(f'    <text x="{cx:.1f}" y="{y + h/2 + 14}" text-anchor="middle" font-size="9" fill="{vfill}">{vtxt}{vextra}</text>')
        return "\n".join(parts)

    # Part A：cache line 结构（完整行）
    line_a = line(90, 96, 600, 60, [True, True, True, True], "#446688", "#e8f0fe")

    # Part B：地址拆解 bar
    bar_y, bar_h = 236, 44
    cells = [
        ("Tag", 220, "#446688", "#e8f0fe", "→ 命中判断"),
        ("Index", 160, "#446688", "#e8f0fe", "→ 定位到哪一行"),
        ("Sector（2 bit）", 110, "#4a7a3a", "#e6f4ea", "→ 行内哪个 sector"),
        ("Byte（5 bit）", 110, "#888", "#f0f0f0", "→ sector 内字节"),
    ]
    bar_parts = []
    bx = 90
    for name, w, accent, tint, ann in cells:
        bar_parts.append(
            f'    <rect x="{bx}" y="{bar_y}" width="{w}" height="{bar_h}" rx="5" fill="{tint}" '
            f'stroke="{accent}" stroke-width="1.3" filter="url(#rough2)"/>'
        )
        bar_parts.append(f'    <text x="{bx + w/2}" y="{bar_y + bar_h/2 + 4}" text-anchor="middle" font-size="11" fill="{accent}" font-weight="bold">{name}</text>')
        bar_parts.append(f'    <text x="{bx + w/2}" y="{bar_y + bar_h + 18}" text-anchor="middle" font-size="10" fill="{accent}">{ann}</text>')
        bx += w
    bar_svg = "\n".join(bar_parts)

    # Part C：两次访问
    line_c1 = line(90, 372, 600, 56, [False, False, True, False], "#4a7a3a", "#e6f4ea")
    line_c2 = line(90, 488, 600, 56, [True, False, True, False], "#4a7a3a", "#e6f4ea")

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 680" font-family="{FONT}">
{DEFS}

  <rect width="780" height="680" fill="#fafafa"/>

  <text x="390" y="32" text-anchor="middle" font-size="18" fill="#444" font-weight="bold">「L2 以 128B cache line 管理」是什么意思</text>
  <text x="390" y="52" text-anchor="middle" font-size="11" fill="#888">管理动作（tag / 命中 / 分配）按 128B 行 · 填充与传输按 32B sector</text>

  <!-- Part A: cache line 结构 -->
  <text x="390" y="84" text-anchor="middle" font-size="13" fill="#446688" font-weight="bold">① L2 的一行长什么样：Tag + 4 个 sector（各带 valid bit）</text>
{line_a}
  <text x="165" y="180" text-anchor="middle" font-size="10" fill="#446688">命中判断：Tag 匹配</text>
  <text x="165" y="194" text-anchor="middle" font-size="10" fill="#446688">以 128B 行为单位</text>
  <text x="470" y="180" text-anchor="middle" font-size="10" fill="#888">每 sector 独立 valid bit</text>
  <text x="470" y="194" text-anchor="middle" font-size="10" fill="#888">填充 / 传输以 32B 为单位</text>

  <!-- Part B: 地址拆解 -->
  <text x="390" y="226" text-anchor="middle" font-size="13" fill="#446688" font-weight="bold">② 地址拆解：命中看 Tag + Index，搬运看 Sector 位</text>
{bar_svg}

  <!-- Part C: 两次访问 -->
  <text x="390" y="330" text-anchor="middle" font-size="13" fill="#4a7a3a" font-weight="bold">③ 两次访问看「管理」与「传输」的分工</text>

  <text x="90" y="364" font-size="12" fill="#444" font-weight="bold">第 1 次：读 4B（落在 S2）→ Tag miss，分配一行、写入 Tag</text>
{line_c1}
  <text x="390" y="450" text-anchor="middle" font-size="10" fill="#4a7a3a">→ 只从 DRAM 搬触达的 S2（32B），其余 sector 留空不搬</text>

  <text x="90" y="480" font-size="12" fill="#444" font-weight="bold">第 2 次：读同一行的 S0 → Tag hit（行已存在），但 V0=0</text>
{line_c2}
  <text x="390" y="566" text-anchor="middle" font-size="10" fill="#4a7a3a">→ 行级命中、sector 级缺失：只补搬 S0（32B），无需整行重取</text>

  <!-- 底部总结 -->
  <g transform="translate(90, 588)">
    <rect x="0" y="0" width="600" height="76" rx="8" fill="#f6f6f6" stroke="#888" stroke-width="1.4" filter="url(#rough2)"/>
    <rect x="0" y="0" width="600" height="20" fill="#446688" opacity="0.12" rx="8" filter="url(#rough2)"/>
    <text x="300" y="15" text-anchor="middle" font-size="12" fill="#446688" font-weight="bold">为什么「管理粗、传输细」</text>
    <text x="20" y="40" font-size="11" fill="#444">tag 按行存：表项数 = 容量 ÷ 128B；若按 32B sector 存 tag，表项 ×4，硬件开销大</text>
    <text x="20" y="58" font-size="11" fill="#444">valid bit 按 sector：支持按需填充，不规则访问不浪费 DRAM 带宽</text>
  </g>
</svg>"""


def memory_hierarchy_transfer() -> str:
    """GPU 访存层次：DRAM → L2 → L1 → Register，标注 cache line（128B 存储）与 sector（32B 传输）粒度。"""

    def cache_line(x, y, w, h, filled, accent, tint, label, ann):
        gap = 4
        sw = (w - 5 * gap) / 4
        parts = [
            f'      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
            f'fill="{tint}" stroke="{accent}" stroke-width="1.5" filter="url(#rough2)"/>'
        ]
        for i in range(4):
            sx = x + gap + i * (sw + gap)
            cx = sx + sw / 2
            cy = y + h / 2 + 3
            if filled[i]:
                parts.append(
                    f'      <rect x="{sx:.1f}" y="{y + 5}" width="{sw:.1f}" height="{h - 10}" rx="3" '
                    f'fill="{accent}" opacity="0.32" stroke="{accent}" stroke-width="1" filter="url(#rough2)"/>'
                )
                parts.append(
                    f'      <text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="10" fill="{accent}">S{i}</text>'
                )
            else:
                parts.append(
                    f'      <rect x="{sx:.1f}" y="{y + 5}" width="{sw:.1f}" height="{h - 10}" rx="3" '
                    f'fill="#ffffff" stroke="{accent}" stroke-width="0.8" stroke-dasharray="3,2" opacity="0.55"/>'
                )
                parts.append(
                    f'      <text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="9" fill="#bbb">空</text>'
                )
        if label:
            parts.append(
                f'      <text x="{x - 6}" y="{y + h/2 + 3}" text-anchor="end" font-size="10" fill="#888">{label}</text>'
            )
        if ann:
            parts.append(
                f'      <text x="{x + w + 8}" y="{y + h/2 + 3}" font-size="10" fill="{accent}">{ann}</text>'
            )
        return "\n".join(parts)

    # Register cells (8 registers)
    nreg, rcw, rcgap = 8, 52, 4
    reg_total = nreg * rcw + (nreg - 1) * rcgap
    reg_x0 = (540 - reg_total) / 2
    reg_parts = []
    for i in range(nreg):
        cx = reg_x0 + i * (rcw + rcgap)
        reg_parts.append(
            f'      <rect x="{cx:.1f}" y="22" width="{rcw}" height="26" rx="4" '
            f'fill="#d6a040" opacity="0.25" stroke="#d6a040" stroke-width="1.1" filter="url(#rough2)"/>'
        )
        reg_parts.append(
            f'      <text x="{cx + rcw/2:.1f}" y="39" text-anchor="middle" font-size="10" fill="#d6a040">r{i}</text>'
        )
    reg_svg = "\n".join(reg_parts)

    # L1 cache lines: CL0 完整，CL1 仅 S2 驻留（演示按 sector 填充）
    cl_w, cl_h, cl_x = 340, 40, 40
    l1_cl0 = cache_line(cl_x, 26, cl_w, cl_h, [True, True, True, True], "#446688", "#e8f0fe", "CL0", "✓ 完整 128B")
    l1_cl1 = cache_line(cl_x, 80, cl_w, cl_h, [False, False, True, False], "#446688", "#e8f0fe", "CL1", "只 1 sector 驻留")

    # L2 cache lines：CL0 完整，CL1 有 3 sector（L1 的 S2 即来自此处）
    l2_cl0 = cache_line(cl_x, 26, cl_w, cl_h, [True, True, True, True], "#4a7a3a", "#e6f4ea", "CL0", "128B")
    l2_cl1 = cache_line(cl_x, 80, cl_w, cl_h, [True, True, True, False], "#4a7a3a", "#e6f4ea", "CL1", "3 sector 已驻留")

    # DRAM grid：1 sector（绿）= 本次搬运，4 sector（蓝）= 其所在 cache line
    cols, rows = 24, 3
    dcw, dch, dgap = 18, 16, 2
    dram_w = cols * dcw + (cols - 1) * dgap
    dram_x0 = (540 - dram_w) / 2
    dram_y0 = 22
    cl_set = {(1, 10), (1, 11), (1, 12), (1, 13)}
    fetch = (1, 11)
    dram_parts = []
    for r in range(rows):
        for c in range(cols):
            cx = dram_x0 + c * (dcw + dgap)
            cy = dram_y0 + r * (dch + dgap)
            if (r, c) == fetch:
                fill, op, stroke = "#4a7a3a", "0.55", "#4a7a3a"
            elif (r, c) in cl_set:
                fill, op, stroke = "#446688", "0.28", "#446688"
            else:
                fill, op, stroke = "#ddd", "1", "#aaa"
            dram_parts.append(
                f'      <rect x="{cx:.1f}" y="{cy}" width="{dcw}" height="{dch}" rx="2" '
                f'fill="{fill}" opacity="{op}" stroke="{stroke}" stroke-width="0.6"/>'
            )
    dram_svg = "\n".join(dram_parts)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 740" font-family="{FONT}">
{DEFS}

  <rect width="780" height="740" fill="#fafafa"/>

  <text x="390" y="32" text-anchor="middle" font-size="18" fill="#444" font-weight="bold">GPU 访存层次：Cache Line、Sector 与搬运单位</text>
  <text x="390" y="52" text-anchor="middle" font-size="11" fill="#888">DRAM → L2 → L1 → Register：存储按 cache line（128B），传输按 sector（32B）</text>

  <!-- Register layer -->
  <g transform="translate(150, 64)">
    <text x="0" y="0" font-size="13" fill="#d6a040" font-weight="bold">Register File · 每线程私有</text>
    <rect x="0" y="8" width="540" height="56" rx="8" fill="#fff8e1" stroke="#d6a040" stroke-width="1.5" filter="url(#rough2)"/>
{reg_svg}
  </g>

  <!-- Arrow 1: L1 -> Register -->
  <line x1="420" y1="182" x2="420" y2="132" stroke="#888" stroke-width="1.6" marker-end="url(#arr)" filter="url(#rough2)"/>
  <text x="432" y="150" font-size="11" fill="#d6a040" font-weight="bold">LDG.128 = 16B（float4）</text>
  <text x="432" y="164" font-size="10" fill="#888">LDG.32 = 4B｜指令宽度由代码决定</text>

  <!-- L1 Cache layer -->
  <g transform="translate(150, 176)">
    <text x="0" y="0" font-size="13" fill="#446688" font-weight="bold">L1 Cache · 每 SM 私有 · 按 cache line（128B = 4 sector）组织</text>
    <rect x="0" y="8" width="540" height="140" rx="8" fill="#e8f0fe" stroke="#446688" stroke-width="1.5" filter="url(#rough2)"/>
{l1_cl0}
{l1_cl1}
    <text x="40" y="140" font-size="10" fill="#888">虚线扇区 = 该 sector 未被触达，位置留空（按 sector 填充，不必整行搬运）</text>
  </g>

  <!-- Arrow 2: L2 -> L1 -->
  <line x1="420" y1="378" x2="420" y2="328" stroke="#888" stroke-width="1.6" marker-end="url(#arr)" filter="url(#rough2)"/>
  <text x="432" y="346" font-size="11" fill="#446688" font-weight="bold">sector 32B</text>
  <text x="432" y="360" font-size="10" fill="#888">传输原子单位，不可再分</text>

  <!-- L2 Cache layer -->
  <g transform="translate(150, 372)">
    <text x="0" y="0" font-size="13" fill="#4a7a3a" font-weight="bold">L2 Cache · 所有 SM 共享 · 按 cache line（128B = 4 sector）组织</text>
    <rect x="0" y="8" width="540" height="140" rx="8" fill="#e6f4ea" stroke="#4a7a3a" stroke-width="1.5" filter="url(#rough2)"/>
{l2_cl0}
{l2_cl1}
    <text x="40" y="140" font-size="10" fill="#888">L1 miss 后向 L2 取，L2 同样按 sector 粒度向 L1 供给</text>
  </g>

  <!-- Arrow 3: DRAM -> L2 -->
  <line x1="420" y1="574" x2="420" y2="524" stroke="#888" stroke-width="1.6" marker-end="url(#arr)" filter="url(#rough2)"/>
  <text x="432" y="542" font-size="11" fill="#4a7a3a" font-weight="bold">sector 32B</text>
  <text x="432" y="556" font-size="10" fill="#888">整 sector 搬运，哪怕线程只用 4B</text>

  <!-- DRAM layer -->
  <g transform="translate(150, 568)">
    <text x="0" y="0" font-size="13" fill="#666" font-weight="bold">DRAM / HBM · Global Memory · 海量数据</text>
    <rect x="0" y="8" width="540" height="88" rx="8" fill="#f0f0f0" stroke="#888" stroke-width="1.5" filter="url(#rough2)"/>
{dram_svg}
    <text x="20" y="86" font-size="10" fill="#4a7a3a">■ 绿 = 本次搬运的 1 sector（32B）</text>
    <text x="250" y="86" font-size="10" fill="#446688">■ 蓝 = 其所在 cache line（128B）区域</text>
  </g>

  <!-- Summary box -->
  <g transform="translate(90, 672)">
    <rect x="0" y="0" width="600" height="60" rx="8" fill="#f6f6f6" stroke="#888" stroke-width="1.4" filter="url(#rough2)"/>
    <rect x="0" y="0" width="600" height="20" fill="#446688" opacity="0.12" rx="8" filter="url(#rough2)"/>
    <text x="300" y="15" text-anchor="middle" font-size="12" fill="#446688" font-weight="bold">两个粒度的分工</text>
    <text x="20" y="38" font-size="11" fill="#444">存储组织：L1/L2 按 cache line（128B = 4 sector）存 tag、做命中判断</text>
    <text x="20" y="54" font-size="11" fill="#444">数据搬运：DRAM→L2、L2→L1 均按 sector（32B）传输，按 sector 填充 cache line</text>
  </g>
</svg>"""


def main() -> None:
    diagrams = {
        "gemm_optimization_layers.svg": gemm_optimization_layers(),
        "float4_vectorized_load.svg": float4_vectorized_load(),
        "parameter_tuning_table.svg": parameter_tuning_table(),
        "cache_line_sector.svg": cache_line_sector(),
        "l2_cache_line_management.svg": l2_cache_line_management(),
        "memory_hierarchy_transfer.svg": memory_hierarchy_transfer(),
    }
    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
