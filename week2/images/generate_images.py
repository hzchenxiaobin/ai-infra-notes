#!/usr/bin/env python3
"""Generate SVG diagrams for Week 2 website."""

from pathlib import Path


def save_svg(filename: str, content: str) -> None:
    path = Path(__file__).parent / filename
    path.write_text(content, encoding="utf-8")
    print(f"Generated: {path}")


# ============================================================
# Day 1: Warp Shuffle
# ============================================================

def warp_shuffle_primitives() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="560" viewBox="0 0 760 560">
  <rect width="760" height="560" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Warp Shuffle 四大家族数据流向</text>

  <!-- __shfl_sync (broadcast) -->
  <text x="380" y="72" text-anchor="middle" font-size="15" font-weight="bold" fill="#58a6ff">__shfl_sync(mask, val, srcLane=4)</text>
  <g transform="translate(140, 85)">
    <text x="0" y="14" font-size="11" fill="#8b949e">Lane</text>
    <rect x="0" y="20" width="40" height="24" fill="#30363d" stroke="#484f58"/><text x="20" y="36" text-anchor="middle" font-size="10" fill="#c9d1d9">0</text>
    <rect x="0" y="44" width="40" height="24" fill="#30363d" stroke="#484f58"/><text x="20" y="60" text-anchor="middle" font-size="10" fill="#c9d1d9">1</text>
    <rect x="0" y="68" width="40" height="24" fill="#30363d" stroke="#484f58"/><text x="20" y="84" text-anchor="middle" font-size="10" fill="#c9d1d9">2</text>
    <rect x="0" y="92" width="40" height="24" fill="#30363d" stroke="#484f58"/><text x="20" y="108" text-anchor="middle" font-size="10" fill="#c9d1d9">3</text>
    <rect x="0" y="116" width="40" height="24" fill="#238636" stroke="#3fb950" stroke-width="2"/><text x="20" y="132" text-anchor="middle" font-size="10" fill="#c9d1d9" font-weight="bold">4</text>
    <rect x="0" y="140" width="40" height="24" fill="#30363d" stroke="#484f58"/><text x="20" y="156" text-anchor="middle" font-size="10" fill="#c9d1d9">5</text>
    <rect x="0" y="164" width="40" height="24" fill="#30363d" stroke="#484f58"/><text x="20" y="180" text-anchor="middle" font-size="10" fill="#c9d1d9">...</text>

    <!-- arrows -->
    <line x1="40" y1="32" x2="100" y2="32" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr56)"/>
    <line x1="40" y1="56" x2="100" y2="56" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr56)"/>
    <line x1="40" y1="80" x2="100" y2="80" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr56)"/>
    <line x1="40" y1="104" x2="100" y2="104" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr56)"/>
    <line x1="40" y1="128" x2="100" y2="128" stroke="#3fb950" stroke-width="2" marker-end="url(#arr56)"/>
    <line x1="40" y1="152" x2="100" y2="152" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr56)"/>
    <line x1="40" y1="176" x2="100" y2="176" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arr56)"/>

    <!-- result -->
    <rect x="100" y="20" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="36" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>
    <rect x="100" y="44" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="60" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>
    <rect x="100" y="68" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="84" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>
    <rect x="100" y="92" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="108" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>
    <rect x="100" y="116" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="132" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>
    <rect x="100" y="140" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="156" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>
    <rect x="100" y="164" width="40" height="24" fill="#1f6feb" opacity="0.4" stroke="#58a6ff"/><text x="120" y="180" text-anchor="middle" font-size="9" fill="#c9d1d9">V4</text>

    <text x="160" y="104" font-size="11" fill="#3fb950">广播：所有 lane 得到 lane 4 的值</text>
  </g>

  <line x1="40" y1="295" x2="720" y2="295" stroke="#30363d" stroke-width="1"/>

  <!-- __shfl_down_sync -->
  <text x="190" y="325" text-anchor="middle" font-size="14" font-weight="bold" fill="#d29922">__shfl_down_sync(delta=2)</text>
  <g transform="translate(60, 340)">
    <text x="0" y="14" font-size="10" fill="#8b949e">Lane</text>
    <rect x="0" y="18" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="32" text-anchor="middle" font-size="9" fill="#c9d1d9">0</text>
    <rect x="0" y="38" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="52" text-anchor="middle" font-size="9" fill="#c9d1d9">1</text>
    <rect x="0" y="58" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="72" text-anchor="middle" font-size="9" fill="#c9d1d9">2</text>
    <rect x="0" y="78" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="92" text-anchor="middle" font-size="9" fill="#c9d1d9">3</text>
    <rect x="0" y="98" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="112" text-anchor="middle" font-size="9" fill="#c9d1d9">4</text>
    <rect x="0" y="118" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="132" text-anchor="middle" font-size="9" fill="#c9d1d9">5</text>

    <text x="42" y="32" font-size="10" fill="#c9d1d9">V0</text>
    <text x="42" y="52" font-size="10" fill="#c9d1d9">V1</text>
    <text x="42" y="72" font-size="10" fill="#c9d1d9">V2</text>
    <text x="42" y="92" font-size="10" fill="#c9d1d9">V3</text>
    <text x="42" y="112" font-size="10" fill="#c9d1d9">V4</text>
    <text x="42" y="132" font-size="10" fill="#c9d1d9">V5</text>

    <line x1="55" y1="28" x2="85" y2="68" stroke="#d29922" stroke-width="1.5"/>
    <line x1="55" y1="48" x2="85" y2="88" stroke="#d29922" stroke-width="1.5"/>
    <line x1="55" y1="68" x2="85" y2="108" stroke="#d29922" stroke-width="1.5"/>
    <line x1="55" y1="88" x2="85" y2="128" stroke="#d29922" stroke-width="1.5"/>

    <text x="95" y="32" font-size="10" fill="#e3b341" font-weight="bold">V2</text>
    <text x="95" y="52" font-size="10" fill="#e3b341" font-weight="bold">V3</text>
    <text x="95" y="72" font-size="10" fill="#e3b341" font-weight="bold">V4</text>
    <text x="95" y="92" font-size="10" fill="#e3b341" font-weight="bold">V5</text>
    <text x="95" y="112" font-size="10" fill="#8b949e">?</text>
    <text x="95" y="132" font-size="10" fill="#8b949e">?</text>

    <text x="0" y="165" font-size="10" fill="#8b949e">向下偏移：lane i 读 lane (i+2)</text>
  </g>

  <!-- __shfl_xor_sync -->
  <text x="520" y="325" text-anchor="middle" font-size="14" font-weight="bold" fill="#a371f7">__shfl_xor_sync(laneMask=2)</text>
  <g transform="translate(400, 340)">
    <text x="0" y="14" font-size="10" fill="#8b949e">Lane</text>
    <rect x="0" y="18" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="32" text-anchor="middle" font-size="9" fill="#c9d1d9">0</text>
    <rect x="0" y="38" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="52" text-anchor="middle" font-size="9" fill="#c9d1d9">1</text>
    <rect x="0" y="58" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="72" text-anchor="middle" font-size="9" fill="#c9d1d9">2</text>
    <rect x="0" y="78" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="92" text-anchor="middle" font-size="9" fill="#c9d1d9">3</text>
    <rect x="0" y="98" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="112" text-anchor="middle" font-size="9" fill="#c9d1d9">4</text>
    <rect x="0" y="118" width="32" height="20" fill="#30363d" stroke="#484f58"/><text x="16" y="132" text-anchor="middle" font-size="9" fill="#c9d1d9">5</text>

    <text x="42" y="32" font-size="10" fill="#c9d1d9">V0</text>
    <text x="42" y="52" font-size="10" fill="#c9d1d9">V1</text>
    <text x="42" y="72" font-size="10" fill="#c9d1d9">V2</text>
    <text x="42" y="92" font-size="10" fill="#c9d1d9">V3</text>
    <text x="42" y="112" font-size="10" fill="#c9d1d9">V4</text>
    <text x="42" y="132" font-size="10" fill="#c9d1d9">V5</text>

    <line x1="55" y1="28" x2="85" y2="68" stroke="#a371f7" stroke-width="1.5"/>
    <line x1="55" y1="48" x2="85" y2="88" stroke="#a371f7" stroke-width="1.5"/>
    <line x1="55" y1="68" x2="85" y2="28" stroke="#a371f7" stroke-width="1.5"/>
    <line x1="55" y1="88" x2="85" y2="48" stroke="#a371f7" stroke-width="1.5"/>
    <line x1="55" y1="108" x2="85" y2="128" stroke="#a371f7" stroke-width="1.5"/>
    <line x1="55" y1="128" x2="85" y2="108" stroke="#a371f7" stroke-width="1.5"/>

    <text x="95" y="32" font-size="10" fill="#a371f7" font-weight="bold">V2</text>
    <text x="95" y="52" font-size="10" fill="#a371f7" font-weight="bold">V3</text>
    <text x="95" y="72" font-size="10" fill="#a371f7" font-weight="bold">V0</text>
    <text x="95" y="92" font-size="10" fill="#a371f7" font-weight="bold">V1</text>
    <text x="95" y="112" font-size="10" fill="#a371f7" font-weight="bold">V6</text>
    <text x="95" y="132" font-size="10" fill="#a371f7" font-weight="bold">V4</text>

    <text x="0" y="165" font-size="10" fill="#8b949e">XOR 交换：lane i 与 lane (i^2) 交换</text>
  </g>

  <!-- latency comparison -->
  <rect x="200" y="520" width="360" height="30" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="380" y="540" text-anchor="middle" font-size="13" fill="#c9d1d9">Shuffle ~1-2 cycles ≪ Shared Memory ~20-30 cycles</text>
</svg>'''


def butterfly_reduction() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="480" viewBox="0 0 760 480">
  <rect width="760" height="480" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Warp Reduce Butterfly 模式（5 步归约）</text>

  <!-- 32 lanes represented as 8 for clarity -->
  <g transform="translate(80, 60)">
    <!-- Lane labels -->
    <text x="-10" y="24" text-anchor="end" font-size="11" fill="#8b949e">Lane 0</text>
    <text x="-10" y="64" text-anchor="end" font-size="11" fill="#8b949e">Lane 1</text>
    <text x="-10" y="104" text-anchor="end" font-size="11" fill="#8b949e">...</text>
    <text x="-10" y="144" text-anchor="end" font-size="11" fill="#8b949e">Lane 15</text>
    <text x="-10" y="184" text-anchor="end" font-size="11" fill="#8b949e">Lane 16</text>
    <text x="-10" y="224" text-anchor="end" font-size="11" fill="#8b949e">...</text>
    <text x="-10" y="264" text-anchor="end" font-size="11" fill="#8b949e">Lane 30</text>
    <text x="-10" y="304" text-anchor="end" font-size="11" fill="#8b949e">Lane 31</text>

    <!-- Step 0: initial values -->
    <text x="50" y="0" text-anchor="middle" font-size="11" fill="#58a6ff" font-weight="bold">初始</text>
    <circle cx="50" cy="20" r="10" fill="#30363d" stroke="#58a6ff"/><text x="50" y="24" text-anchor="middle" font-size="8" fill="#c9d1d9">v0</text>
    <circle cx="50" cy="60" r="10" fill="#30363d" stroke="#58a6ff"/><text x="50" y="64" text-anchor="middle" font-size="8" fill="#c9d1d9">v1</text>
    <circle cx="50" cy="100" r="10" fill="#30363d" stroke="#58a6ff"/>
    <circle cx="50" cy="140" r="10" fill="#30363d" stroke="#58a6ff"/><text x="50" y="144" text-anchor="middle" font-size="8" fill="#c9d1d9">v15</text>
    <circle cx="50" cy="180" r="10" fill="#30363d" stroke="#58a6ff"/><text x="50" y="184" text-anchor="middle" font-size="8" fill="#c9d1d9">v16</text>
    <circle cx="50" cy="220" r="10" fill="#30363d" stroke="#58a6ff"/>
    <circle cx="50" cy="260" r="10" fill="#30363d" stroke="#58a6ff"/><text x="50" y="264" text-anchor="middle" font-size="8" fill="#c9d1d9">v30</text>
    <circle cx="50" cy="300" r="10" fill="#30363d" stroke="#58a6ff"/><text x="50" y="304" text-anchor="middle" font-size="8" fill="#c9d1d9">v31</text>

    <!-- Step 1: offset=16 -->
    <text x="190" y="0" text-anchor="middle" font-size="11" fill="#d29922" font-weight="bold">Step 1: offset=16</text>
    <line x1="60" y1="20" x2="180" y2="180" stroke="#d29922" stroke-width="1.5"/>
    <line x1="60" y1="180" x2="180" y2="20" stroke="#d29922" stroke-width="1.5"/>
    <circle cx="190" cy="20" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/><text x="190" y="24" text-anchor="middle" font-size="7" fill="#c9d1d9">v0+16</text>
    <circle cx="190" cy="60" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/>
    <circle cx="190" cy="100" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/>
    <circle cx="190" cy="140" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/>
    <circle cx="190" cy="180" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/>
    <circle cx="190" cy="220" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/>
    <circle cx="190" cy="260" r="10" fill="#1f6feb" opacity="0.5" stroke="#d29922"/>
    <circle cx="190" cy="300" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>

    <!-- Step 2: offset=8 -->
    <text x="330" y="0" text-anchor="middle" font-size="11" fill="#d29922" font-weight="bold">Step 2: offset=8</text>
    <line x1="200" y1="20" x2="320" y2="100" stroke="#d29922" stroke-width="1.5"/>
    <line x1="200" y1="100" x2="320" y2="20" stroke="#d29922" stroke-width="1.5"/>
    <circle cx="330" cy="20" r="10" fill="#1f6feb" opacity="0.6" stroke="#d29922"/><text x="330" y="24" text-anchor="middle" font-size="7" fill="#c9d1d9">+v8</text>
    <circle cx="330" cy="60" r="10" fill="#1f6feb" opacity="0.6" stroke="#d29922"/>
    <circle cx="330" cy="100" r="10" fill="#1f6feb" opacity="0.6" stroke="#d29922"/>
    <circle cx="330" cy="140" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="330" cy="180" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="330" cy="220" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="330" cy="260" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="330" cy="300" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>

    <!-- Step 3: offset=4 -->
    <text x="470" y="0" text-anchor="middle" font-size="11" fill="#d29922" font-weight="bold">Step 3: offset=4</text>
    <line x1="340" y1="20" x2="460" y2="60" stroke="#d29922" stroke-width="1.5"/>
    <line x1="340" y1="60" x2="460" y2="20" stroke="#d29922" stroke-width="1.5"/>
    <circle cx="470" cy="20" r="10" fill="#1f6feb" opacity="0.7" stroke="#d29922"/><text x="470" y="24" text-anchor="middle" font-size="7" fill="#c9d1d9">+v4</text>
    <circle cx="470" cy="60" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="470" cy="100" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="470" cy="140" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="470" cy="180" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="470" cy="220" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="470" cy="260" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <circle cx="470" cy="300" r="10" fill="#8b949e" opacity="0.3" stroke="#484f58"/>

    <!-- Step 4: offset=2 -->
    <text x="570" y="0" text-anchor="middle" font-size="11" fill="#d29922" font-weight="bold">offset=2</text>
    <line x1="480" y1="20" x2="560" y2="40" stroke="#d29922" stroke-width="1.5"/>
    <circle cx="570" cy="20" r="10" fill="#1f6feb" opacity="0.8" stroke="#d29922"/><text x="570" y="24" text-anchor="middle" font-size="7" fill="#c9d1d9">+v2</text>

    <!-- Step 5: offset=1 -->
    <text x="660" y="0" text-anchor="middle" font-size="11" fill="#3fb950" font-weight="bold">offset=1</text>
    <line x1="580" y1="20" x2="650" y2="30" stroke="#3fb950" stroke-width="1.5"/>
    <circle cx="660" cy="20" r="12" fill="#238636" stroke="#3fb950" stroke-width="2"/><text x="660" y="24" text-anchor="middle" font-size="8" fill="#fff" font-weight="bold">SUM</text>
  </g>

  <!-- Result annotation -->
  <rect x="200" y="400" width="360" height="30" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="380" y="420" text-anchor="middle" font-size="13" fill="#3fb950" font-weight="bold">5 步后 lane 0 持有 32 线程的累加和</text>

  <text x="380" y="450" text-anchor="middle" font-size="12" fill="#8b949e">每步: val += __shfl_down_sync(0xFFFFFFFF, val, offset)</text>
</svg>'''


def two_level_reduction() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="520" viewBox="0 0 760 520">
  <rect width="760" height="520" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">两级归约：Warp Reduce + Block Reduce</text>

  <!-- Step 1: per-thread accumulation -->
  <text x="380" y="72" text-anchor="middle" font-size="14" font-weight="bold" fill="#58a6ff">Step 1: 每个线程读取 Global Memory 并做 per-thread 累加（grid-stride loop）</text>
  <g transform="translate(80, 85)">
    <rect x="0" y="0" width="600" height="40" rx="6" fill="#161b22" stroke="#30363d"/>
    <text x="20" y="25" font-size="12" fill="#c9d1d9">for (i = tid; i &lt; n; i += blockDim.x * gridDim.x)  sum += input[i];</text>
  </g>

  <!-- Step 2: warp reduce -->
  <text x="380" y="155" text-anchor="middle" font-size="14" font-weight="bold" fill="#d29922">Step 2: Warp 级归约（__shfl_down_sync butterfly，每个 warp 的 32 线程累加到 lane 0）</text>
  <g transform="translate(120, 170)">
    <!-- Warp 0 -->
    <rect x="0" y="0" width="100" height="60" rx="6" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
    <text x="50" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">Warp 0</text>
    <text x="50" y="42" text-anchor="middle" font-size="10" fill="#8b949e">32 threads → lane 0</text>
    <circle cx="85" cy="30" r="8" fill="#d29922"/><text x="85" y="34" text-anchor="middle" font-size="8" fill="#0d1117" font-weight="bold">S0</text>

    <rect x="130" y="0" width="100" height="60" rx="6" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
    <text x="180" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">Warp 1</text>
    <text x="180" y="42" text-anchor="middle" font-size="10" fill="#8b949e">32 threads → lane 0</text>
    <circle cx="215" cy="30" r="8" fill="#d29922"/><text x="215" y="34" text-anchor="middle" font-size="8" fill="#0d1117" font-weight="bold">S1</text>

    <rect x="260" y="0" width="100" height="60" rx="6" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
    <text x="310" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">Warp 2</text>
    <text x="310" y="42" text-anchor="middle" font-size="10" fill="#8b949e">32 threads → lane 0</text>
    <circle cx="345" cy="30" r="8" fill="#d29922"/><text x="345" y="34" text-anchor="middle" font-size="8" fill="#0d1117" font-weight="bold">S2</text>

    <text x="400" y="34" font-size="16" fill="#8b949e">...</text>

    <rect x="430" y="0" width="100" height="60" rx="6" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
    <text x="480" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">Warp 31</text>
    <text x="480" y="42" text-anchor="middle" font-size="10" fill="#8b949e">32 threads → lane 0</text>
    <circle cx="515" cy="30" r="8" fill="#d29922"/><text x="515" y="34" text-anchor="middle" font-size="8" fill="#0d1117" font-weight="bold">S31</text>
  </g>

  <!-- Step 3: shared memory -->
  <text x="380" y="270" text-anchor="middle" font-size="14" font-weight="bold" fill="#a371f7">Step 3: lane 0 写入 Shared Memory → __syncthreads()</text>
  <g transform="translate(200, 285)">
    <rect x="0" y="0" width="360" height="50" rx="6" fill="#8957e5" opacity="0.15" stroke="#a371f7" stroke-width="2"/>
    <text x="180" y="20" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">__shared__ float warpSums[32]</text>
    <text x="180" y="38" text-anchor="middle" font-size="11" fill="#8b949e">warpSums[0]=S0, warpSums[1]=S1, ..., warpSums[31]=S31</text>
  </g>

  <!-- Step 4: warp 0 final reduce -->
  <text x="380" y="370" text-anchor="middle" font-size="14" font-weight="bold" fill="#3fb950">Step 4: Warp 0 读取 warpSums[0..31]，再执行一次 warpReduceSum</text>
  <g transform="translate(250, 385)">
    <rect x="0" y="0" width="260" height="60" rx="6" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
    <text x="130" y="22" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Warp 0 (32 lanes)</text>
    <text x="130" y="42" text-anchor="middle" font-size="11" fill="#8b949e">lane i 读 warpSums[i] → warpReduceSum → lane 0</text>
  </g>

  <!-- Final result -->
  <line x1="380" y1="450" x2="380" y2="475" stroke="#3fb950" stroke-width="2"/>
  <rect x="280" y="475" width="200" height="35" rx="8" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="380" y="498" text-anchor="middle" font-size="14" fill="#fff" font-weight="bold">output[blockIdx.x] = sum</text>
</svg>'''


# ============================================================
# Day 2: Register Blocking
# ============================================================

def register_blocking_dataflow() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500" viewBox="0 0 760 500">
  <rect width="760" height="500" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Register Blocking 三级数据复用</text>

  <!-- Global Memory -->
  <rect x="40" y="70" width="160" height="80" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="120" y="100" text-anchor="middle" font-size="15" fill="#c9d1d9" font-weight="bold">Global Memory</text>
  <text x="120" y="122" text-anchor="middle" font-size="12" fill="#8b949e">A[M][K], B[K][N]</text>
  <text x="120" y="140" text-anchor="middle" font-size="11" fill="#8b949e">~400-800 cycles</text>

  <!-- Arrow down -->
  <line x1="120" y1="150" x2="120" y2="185" stroke="#58a6ff" stroke-width="2" marker-end="url(#arr5)"/>
  <text x="135" y="170" font-size="11" fill="#58a6ff">协作加载</text>

  <!-- Shared Memory -->
  <rect x="40" y="190" width="160" height="80" rx="8" fill="#a371f7" opacity="0.2" stroke="#a371f7" stroke-width="2"/>
  <text x="120" y="220" text-anchor="middle" font-size="15" fill="#c9d1d9" font-weight="bold">Shared Memory</text>
  <text x="120" y="242" text-anchor="middle" font-size="12" fill="#8b949e">s_A[BM][BK], s_B[BK][BN]</text>
  <text x="120" y="260" text-anchor="middle" font-size="11" fill="#8b949e">~20-30 cycles</text>

  <!-- Arrows to registers -->
  <line x1="120" y1="270" x2="120" y2="305" stroke="#58a6ff" stroke-width="2" marker-end="url(#arr5)"/>
  <text x="135" y="290" font-size="11" fill="#58a6ff">加载到寄存器</text>

  <!-- Registers -->
  <rect x="20" y="310" width="80" height="60" rx="6" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="60" y="335" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">r_A[TM]</text>
  <text x="60" y="355" text-anchor="middle" font-size="10" fill="#8b949e">A 子行</text>

  <rect x="120" y="310" width="80" height="60" rx="6" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="160" y="335" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">r_B[TN]</text>
  <text x="160" y="355" text-anchor="middle" font-size="10" fill="#8b949e">B 子列</text>

  <!-- FMA -->
  <line x1="60" y1="370" x2="100" y2="410" stroke="#3fb950" stroke-width="2"/>
  <line x1="160" y1="370" x2="120" y2="410" stroke="#3fb950" stroke-width="2"/>

  <rect x="40" y="410" width="140" height="60" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="110" y="435" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">FMA 累加</text>
  <text x="110" y="455" text-anchor="middle" font-size="13" fill="#58a6ff" font-weight="bold">acc[TM][TN]</text>

  <!-- Right side: optimization levels -->
  <g transform="translate(320, 80)">
    <text x="180" y="0" text-anchor="middle" font-size="16" fill="#c9d1d9" font-weight="bold">优化层次与性能目标</text>

    <rect x="0" y="20" width="360" height="35" rx="6" fill="#f85149" opacity="0.15" stroke="#f85149"/>
    <text x="15" y="42" font-size="12" fill="#c9d1d9">Naive GEMM</text>
    <text x="200" y="42" font-size="12" fill="#f85149" font-weight="bold">~1-3% peak</text>

    <rect x="0" y="65" width="360" height="35" rx="6" fill="#d29922" opacity="0.15" stroke="#d29922"/>
    <text x="15" y="87" font-size="12" fill="#c9d1d9">Shared Memory Tiling</text>
    <text x="200" y="87" font-size="12" fill="#d29922" font-weight="bold">~15-25% peak</text>

    <rect x="0" y="110" width="360" height="35" rx="6" fill="#a371f7" opacity="0.15" stroke="#a371f7"/>
    <text x="15" y="132" font-size="12" fill="#c9d1d9">Register Blocking</text>
    <text x="200" y="132" font-size="12" fill="#a371f7" font-weight="bold">~40-60% peak</text>

    <rect x="0" y="155" width="360" height="35" rx="6" fill="#58a6ff" opacity="0.15" stroke="#58a6ff"/>
    <text x="15" y="177" font-size="12" fill="#c9d1d9">Warp-level + Shuffle</text>
    <text x="200" y="177" font-size="12" fill="#58a6ff" font-weight="bold">~60-80% peak</text>

    <rect x="0" y="200" width="360" height="35" rx="6" fill="#3fb950" opacity="0.15" stroke="#3fb950"/>
    <text x="15" y="222" font-size="12" fill="#c9d1d9">软件流水线 (Double Buffer)</text>
    <text x="200" y="222" font-size="12" fill="#3fb950" font-weight="bold">~80-95% peak</text>

    <!-- Register usage -->
    <text x="180" y="270" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Register 使用量（TM=TN=8）</text>
    <rect x="20" y="285" width="320" height="140" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="40" y="310" font-size="12" fill="#3fb950">acc[8][8] = 64 个 float</text>
    <text x="40" y="332" font-size="12" fill="#58a6ff">r_A[8] = 8 个 float</text>
    <text x="40" y="354" font-size="12" fill="#58a6ff">r_B[8] = 8 个 float</text>
    <text x="40" y="376" font-size="12" fill="#8b949e">索引/临时变量 ≈ 8 个</text>
    <line x1="40" y1="388" x2="320" y2="388" stroke="#30363d"/>
    <text x="40" y="412" font-size="14" fill="#e3b341" font-weight="bold">总计 ≈ 88 registers（上限 255）</text>
  </g>
</svg>'''


def thread_tile_mapping() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="480" viewBox="0 0 760 480">
  <rect width="760" height="480" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Thread Tile 二维映射（BM=128, BN=128, TM=8, TN=8）</text>

  <!-- Block tile -->
  <g transform="translate(80, 70)">
    <text x="0" y="-5" font-size="14" fill="#58a6ff" font-weight="bold">Block Tile (BM×BN = 128×128)</text>
    <rect x="0" y="0" width="320" height="320" fill="none" stroke="#58a6ff" stroke-width="2" rx="4"/>

    <!-- Thread tiles grid (16x16 = 256 threads) -->
    <g opacity="0.6">
      <!-- Row 0 -->
      <rect x="0" y="0" width="20" height="20" fill="#1f6feb" opacity="0.3" stroke="#30363d" stroke-width="0.5"/>
      <rect x="20" y="0" width="20" height="20" fill="#1f6feb" opacity="0.2" stroke="#30363d" stroke-width="0.5"/>
      <rect x="40" y="0" width="20" height="20" fill="#1f6feb" opacity="0.3" stroke="#30363d" stroke-width="0.5"/>
      <text x="280" y="-2" font-size="9" fill="#8b949e">... 16 cols</text>
    </g>

    <!-- Highlight one thread tile -->
    <rect x="40" y="40" width="20" height="20" fill="#238636" opacity="0.5" stroke="#3fb950" stroke-width="2"/>
    <text x="50" y="54" text-anchor="middle" font-size="8" fill="#fff" font-weight="bold">T</text>

    <!-- Thread tile annotation -->
    <line x1="60" y1="50" x2="200" y2="50" stroke="#3fb950" stroke-width="1" stroke-dasharray="4"/>
    <rect x="200" y="35" width="120" height="30" rx="4" fill="#161b22" stroke="#3fb950"/>
    <text x="260" y="54" text-anchor="middle" font-size="11" fill="#3fb950">TM×TN = 8×8</text>

    <!-- Grid lines -->
    <line x1="0" y1="20" x2="320" y2="20" stroke="#30363d" stroke-width="0.5"/>
    <line x1="0" y1="40" x2="320" y2="40" stroke="#30363d" stroke-width="0.5"/>
    <line x1="0" y1="60" x2="320" y2="60" stroke="#30363d" stroke-width="0.5"/>
    <line x1="20" y1="0" x2="20" y2="320" stroke="#30363d" stroke-width="0.5"/>
    <line x1="40" y1="0" x2="40" y2="320" stroke="#30363d" stroke-width="0.5"/>
    <line x1="60" y1="0" x2="60" y2="320" stroke="#30363d" stroke-width="0.5"/>

    <text x="160" y="340" text-anchor="middle" font-size="11" fill="#8b949e">16×16 = 256 个 Thread Tile，每个线程负责 8×8 输出</text>
  </g>

  <!-- Right side: formulas -->
  <g transform="translate(460, 80)">
    <rect x="0" y="0" width="280" height="280" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="140" y="28" text-anchor="middle" font-size="14" fill="#58a6ff" font-weight="bold">线程映射公式</text>

    <text x="20" y="60" font-size="12" fill="#c9d1d9">threadIdx.x 范围: 0 ~ 255</text>
    <text x="20" y="85" font-size="12" fill="#c9d1d9">BN / TN = 128 / 8 = 16</text>

    <line x1="20" y1="100" x2="260" y2="100" stroke="#30363d"/>

    <text x="20" y="125" font-size="13" fill="#3fb950" font-family="monospace">threadRow = threadIdx.x / 16</text>
    <text x="20" y="145" font-size="13" fill="#3fb950" font-family="monospace">threadCol = threadIdx.x % 16</text>

    <line x1="20" y1="160" x2="260" y2="160" stroke="#30363d"/>

    <text x="20" y="185" font-size="12" fill="#c9d1d9">线程 (threadRow, threadCol)</text>
    <text x="20" y="205" font-size="12" fill="#c9d1d9">负责输出行:</text>
    <text x="30" y="225" font-size="11" fill="#d29922" font-family="monospace">[cRow + threadRow*TM,</text>
    <text x="30" y="242" font-size="11" fill="#d29922" font-family="monospace"> cRow + (threadRow+1)*TM)</text>

    <text x="20" y="265" font-size="12" fill="#c9d1d9">每 Block 线程数 = 16×16 = 256</text>
  </g>

  <!-- Bottom: key params table -->
  <g transform="translate(80, 410)">
    <rect x="0" y="0" width="600" height="60" rx="6" fill="#161b22" stroke="#30363d"/>
    <text x="15" y="22" font-size="12" fill="#58a6ff">BM=128</text>
    <text x="100" y="22" font-size="12" fill="#58a6ff">BN=128</text>
    <text x="185" y="22" font-size="12" fill="#58a6ff">BK=8</text>
    <text x="250" y="22" font-size="12" fill="#3fb950">TM=8</text>
    <text x="315" y="22" font-size="12" fill="#3fb950">TN=8</text>
    <text x="380" y="22" font-size="12" fill="#d29922">Threads=256</text>
    <text x="490" y="22" font-size="12" fill="#d29922">Regs≈88</text>
    <text x="300" y="45" text-anchor="middle" font-size="11" fill="#8b949e">Shared Mem: s_A[128][8] + s_B[8][128] = 2×4KB = 8KB</text>
  </g>
</svg>'''


def double_buffering() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">
  <rect width="760" height="420" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Double Buffering（软件流水线）</text>

  <!-- Single buffer (top) -->
  <text x="380" y="72" text-anchor="middle" font-size="15" fill="#f85149" font-weight="bold">单缓冲：Load 和 Compute 串行，存在空闲等待</text>
  <g transform="translate(80, 85)">
    <rect x="0" y="0" width="120" height="35" rx="4" fill="#d29922" opacity="0.3" stroke="#d29922"/>
    <text x="60" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Load Tile 0</text>
    <rect x="120" y="0" width="120" height="35" rx="4" fill="#58a6ff" opacity="0.3" stroke="#58a6ff"/>
    <text x="180" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Compute 0</text>
    <rect x="240" y="0" width="120" height="35" rx="4" fill="#d29922" opacity="0.3" stroke="#d29922"/>
    <text x="300" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Load Tile 1</text>
    <rect x="360" y="0" width="120" height="35" rx="4" fill="#58a6ff" opacity="0.3" stroke="#58a6ff"/>
    <text x="420" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Compute 1</text>

    <!-- idle gap -->
    <rect x="120" y="40" width="120" height="15" fill="none" stroke="#f85149" stroke-dasharray="3"/>
    <text x="180" y="68" text-anchor="middle" font-size="10" fill="#f85149">↑ 空闲等待</text>
  </g>

  <line x1="40" y1="180" x2="720" y2="180" stroke="#30363d" stroke-width="1"/>

  <!-- Double buffer (bottom) -->
  <text x="380" y="215" text-anchor="middle" font-size="15" fill="#3fb950" font-weight="bold">双缓冲：Compute 和 Load 并行，掩盖传输延迟</text>
  <g transform="translate(80, 230)">
    <!-- Lane 1 (Buf 0) -->
    <text x="-10" y="22" text-anchor="end" font-size="11" fill="#8b949e">Buf 0</text>
    <rect x="0" y="0" width="120" height="35" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="60" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Load 0</text>
    <rect x="240" y="0" width="120" height="35" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="300" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Load 2</text>
    <rect x="480" y="0" width="120" height="35" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="540" y="22" text-anchor="middle" font-size="12" fill="#c9d1d9">Load 4</text>

    <!-- Lane 2 (Buf 1) -->
    <text x="-10" y="62" text-anchor="end" font-size="11" fill="#8b949e">Buf 1</text>
    <rect x="120" y="40" width="120" height="35" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="180" y="62" text-anchor="middle" font-size="12" fill="#c9d1d9">Load 1</text>
    <rect x="360" y="40" width="120" height="35" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="420" y="62" text-anchor="middle" font-size="12" fill="#c9d1d9">Load 3</text>

    <!-- Compute lane -->
    <text x="-10" y="105" text-anchor="end" font-size="11" fill="#8b949e">Compute</text>
    <rect x="120" y="85" width="120" height="35" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="180" y="107" text-anchor="middle" font-size="12" fill="#c9d1d9">Compute 0</text>
    <rect x="240" y="85" width="120" height="35" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="300" y="107" text-anchor="middle" font-size="12" fill="#c9d1d9">Compute 1</text>
    <rect x="360" y="85" width="120" height="35" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="420" y="107" text-anchor="middle" font-size="12" fill="#c9d1d9">Compute 2</text>
    <rect x="480" y="85" width="120" height="35" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="540" y="107" text-anchor="middle" font-size="12" fill="#c9d1d9">Compute 3</text>

    <!-- overlap annotation -->
    <rect x="120" y="40" width="120" height="80" fill="none" stroke="#3fb950" stroke-width="2" stroke-dasharray="4" rx="4"/>
    <text x="180" y="140" text-anchor="middle" font-size="10" fill="#3fb950">↑ Load 与 Compute 重叠</text>
  </g>

  <rect x="160" y="385" width="440" height="30" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="380" y="405" text-anchor="middle" font-size="13" fill="#c9d1d9">声明两份 shared memory buffer，奇偶 tile 交替使用</text>
</svg>'''


# ============================================================
# Day 3: CUDA Streams
# ============================================================

def multi_stream_overlap() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500" viewBox="0 0 760 500">
  <rect width="760" height="500" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Multi-Stream H2D/Compute/D2H 重叠流水线</text>

  <!-- Sequential (top) -->
  <text x="380" y="72" text-anchor="middle" font-size="15" fill="#f85149" font-weight="bold">顺序执行（无 Stream）：总计 = H2D + Compute + D2H</text>
  <g transform="translate(60, 85)">
    <text x="0" y="22" font-size="12" fill="#8b949e">Default</text>
    <rect x="60" y="5" width="100" height="30" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="110" y="25" text-anchor="middle" font-size="10" fill="#c9d1d9">H2D</text>
    <rect x="160" y="5" width="100" height="30" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="210" y="25" text-anchor="middle" font-size="10" fill="#c9d1d9">Compute</text>
    <rect x="260" y="5" width="100" height="30" rx="4" fill="#a371f7" opacity="0.4" stroke="#a371f7"/>
    <text x="310" y="25" text-anchor="middle" font-size="10" fill="#c9d1d9">D2H</text>
    <rect x="360" y="5" width="100" height="30" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="410" y="25" text-anchor="middle" font-size="10" fill="#c9d1d9">H2D</text>
    <rect x="460" y="5" width="100" height="30" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="510" y="25" text-anchor="middle" font-size="10" fill="#c9d1d9">Compute</text>
    <rect x="560" y="5" width="100" height="30" rx="4" fill="#a371f7" opacity="0.4" stroke="#a371f7"/>
    <text x="610" y="25" text-anchor="middle" font-size="10" fill="#c9d1d9">D2H</text>
  </g>

  <line x1="40" y1="140" x2="720" y2="140" stroke="#30363d" stroke-width="1"/>

  <!-- Multi-Stream (bottom) -->
  <text x="380" y="175" text-anchor="middle" font-size="15" fill="#3fb950" font-weight="bold">Multi-Stream 重叠：总计 ≈ max(H2D, Compute) + 流水线填充</text>

  <g transform="translate(60, 190)">
    <!-- Stream 1 -->
    <text x="0" y="22" font-size="12" fill="#8b949e">Stream 1</text>
    <rect x="60" y="5" width="80" height="30" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="100" y="25" text-anchor="middle" font-size="9" fill="#c9d1d9">H2D</text>
    <rect x="140" y="5" width="80" height="30" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="180" y="25" text-anchor="middle" font-size="9" fill="#c9d1d9">Comp</text>
    <rect x="220" y="5" width="80" height="30" rx="4" fill="#a371f7" opacity="0.4" stroke="#a371f7"/>
    <text x="260" y="25" text-anchor="middle" font-size="9" fill="#c9d1d9">D2H</text>

    <!-- Stream 2 -->
    <text x="0" y="57" font-size="12" fill="#8b949e">Stream 2</text>
    <rect x="100" y="40" width="80" height="30" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="140" y="60" text-anchor="middle" font-size="9" fill="#c9d1d9">H2D</text>
    <rect x="180" y="40" width="80" height="30" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="220" y="60" text-anchor="middle" font-size="9" fill="#c9d1d9">Comp</text>
    <rect x="260" y="40" width="80" height="30" rx="4" fill="#a371f7" opacity="0.4" stroke="#a371f7"/>
    <text x="300" y="60" text-anchor="middle" font-size="9" fill="#c9d1d9">D2H</text>

    <!-- Stream 3 -->
    <text x="0" y="92" font-size="12" fill="#8b949e">Stream 3</text>
    <rect x="140" y="75" width="80" height="30" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="180" y="95" text-anchor="middle" font-size="9" fill="#c9d1d9">H2D</text>
    <rect x="220" y="75" width="80" height="30" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="260" y="95" text-anchor="middle" font-size="9" fill="#c9d1d9">Comp</text>
    <rect x="300" y="75" width="80" height="30" rx="4" fill="#a371f7" opacity="0.4" stroke="#a371f7"/>
    <text x="340" y="95" text-anchor="middle" font-size="9" fill="#c9d1d9">D2H</text>

    <!-- Stream 4 -->
    <text x="0" y="127" font-size="12" fill="#8b949e">Stream 4</text>
    <rect x="180" y="110" width="80" height="30" rx="4" fill="#d29922" opacity="0.4" stroke="#d29922"/>
    <text x="220" y="130" text-anchor="middle" font-size="9" fill="#c9d1d9">H2D</text>
    <rect x="260" y="110" width="80" height="30" rx="4" fill="#58a6ff" opacity="0.4" stroke="#58a6ff"/>
    <text x="300" y="130" text-anchor="middle" font-size="9" fill="#c9d1d9">Comp</text>
    <rect x="340" y="110" width="80" height="30" rx="4" fill="#a371f7" opacity="0.4" stroke="#a371f7"/>
    <text x="380" y="130" text-anchor="middle" font-size="9" fill="#c9d1d9">D2H</text>

    <!-- overlap highlight -->
    <rect x="140" y="5" width="160" height="135" fill="none" stroke="#3fb950" stroke-width="2" stroke-dasharray="4" rx="4"/>
    <text x="220" y="160" text-anchor="middle" font-size="11" fill="#3fb950">↑ H2D 与 Compute 重叠</text>

    <!-- Hardware engines -->
    <line x1="0" y1="180" x2="560" y2="180" stroke="#30363d"/>
    <text x="0" y="200" font-size="13" fill="#c9d1d9" font-weight="bold">硬件引擎独立并行：</text>
    <rect x="200" y="185" width="120" height="25" rx="4" fill="#d29922" opacity="0.2" stroke="#d29922"/>
    <text x="260" y="202" text-anchor="middle" font-size="11" fill="#d29922">Copy Engine (DMA)</text>
    <rect x="340" y="185" width="140" height="25" rx="4" fill="#58a6ff" opacity="0.2" stroke="#58a6ff"/>
    <text x="410" y="202" text-anchor="middle" font-size="11" fill="#58a6ff">Compute Engine (SM)</text>

    <!-- Speedup -->
    <rect x="120" y="230" width="360" height="35" rx="6" fill="#161b22" stroke="#3fb950"/>
    <text x="300" y="252" text-anchor="middle" font-size="13" fill="#3fb950" font-weight="bold">加速比通常 1.2x ~ 1.8x（取决于 Copy Engine 数量）</text>
  </g>
</svg>'''


def default_stream_sync() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">
  <rect width="760" height="460" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Default Stream 隐式同步陷阱</text>

  <!-- Scenario: multi-stream without default stream -->
  <text x="380" y="72" text-anchor="middle" font-size="15" fill="#3fb950" font-weight="bold">✅ 正确：全部使用 Explicit Stream，并发正常</text>
  <g transform="translate(80, 85)">
    <text x="0" y="22" font-size="12" fill="#8b949e">Stream A</text>
    <rect x="70" y="5" width="120" height="30" rx="4" fill="#58a6ff" opacity="0.3" stroke="#58a6ff"/>
    <text x="130" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">Kernel A</text>
    <rect x="190" y="5" width="120" height="30" rx="4" fill="#a371f7" opacity="0.3" stroke="#a371f7"/>
    <text x="250" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">D2H A</text>

    <text x="0" y="57" font-size="12" fill="#8b949e">Stream B</text>
    <rect x="110" y="40" width="120" height="30" rx="4" fill="#3fb950" opacity="0.3" stroke="#3fb950"/>
    <text x="170" y="60" text-anchor="middle" font-size="11" fill="#c9d1d9">Kernel B</text>
    <rect x="230" y="40" width="120" height="30" rx="4" fill="#d29922" opacity="0.3" stroke="#d29922"/>
    <text x="290" y="60" text-anchor="middle" font-size="11" fill="#c9d1d9">D2H B</text>

    <rect x="110" y="5" width="200" height="65" fill="none" stroke="#3fb950" stroke-width="2" stroke-dasharray="4" rx="4"/>
    <text x="210" y="88" text-anchor="middle" font-size="11" fill="#3fb950">↑ A 和 B 并发执行</text>
  </g>

  <line x1="40" y1="200" x2="720" y2="200" stroke="#30363d" stroke-width="1"/>

  <!-- Scenario: default stream breaks concurrency -->
  <text x="380" y="235" text-anchor="middle" font-size="15" fill="#f85149" font-weight="bold">❌ 错误：某处调用 cudaMemcpy（Default Stream），打断所有并发</text>
  <g transform="translate(80, 250)">
    <text x="0" y="22" font-size="12" fill="#8b949e">Stream A</text>
    <rect x="70" y="5" width="100" height="30" rx="4" fill="#58a6ff" opacity="0.3" stroke="#58a6ff"/>
    <text x="120" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">Kernel A</text>

    <text x="0" y="57" font-size="12" fill="#8b949e">Stream B</text>
    <rect x="110" y="40" width="100" height="30" rx="4" fill="#3fb950" opacity="0.3" stroke="#3fb950"/>
    <text x="160" y="60" text-anchor="middle" font-size="11" fill="#c9d1d9">Kernel B</text>

    <!-- Default stream blocks everything -->
    <text x="0" y="92" font-size="12" fill="#f85149" font-weight="bold">Default</text>
    <rect x="70" y="75" width="200" height="30" rx="4" fill="#f85149" opacity="0.3" stroke="#f85149" stroke-width="2"/>
    <text x="170" y="95" text-anchor="middle" font-size="11" fill="#c9d1d9">cudaMemcpy（同步阻塞）</text>

    <!-- blocked annotation -->
    <line x1="180" y1="10" x2="180" y2="75" stroke="#f85149" stroke-width="1" stroke-dasharray="3"/>
    <line x1="220" y1="45" x2="220" y2="75" stroke="#f85149" stroke-width="1" stroke-dasharray="3"/>
    <text x="350" y="50" font-size="11" fill="#f85149">← 所有 Stream 被阻塞</text>
    <text x="350" y="68" font-size="11" fill="#f85149">   等待 Default Stream 完成</text>

    <!-- After default stream -->
    <text x="0" y="127" font-size="12" fill="#8b949e">Stream A</text>
    <rect x="270" y="110" width="100" height="30" rx="4" fill="#58a6ff" opacity="0.3" stroke="#58a6ff"/>
    <text x="320" y="130" text-anchor="middle" font-size="11" fill="#c9d1d9">续 A</text>

    <text x="0" y="162" font-size="12" fill="#8b949e">Stream B</text>
    <rect x="270" y="145" width="100" height="30" rx="4" fill="#3fb950" opacity="0.3" stroke="#3fb950"/>
    <text x="320" y="165" text-anchor="middle" font-size="11" fill="#c9d1d9">续 B</text>
  </g>

  <rect x="100" y="425" width="560" height="30" rx="6" fill="#161b22" stroke="#30363d"/>
  <text x="380" y="445" text-anchor="middle" font-size="13" fill="#c9d1d9">解决：cudaStreamCreateWithFlags(&amp;stream, cudaStreamNonBlocking) 或 --default-stream per-thread</text>
</svg>'''


def stream_event_dependency() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400">
  <rect width="760" height="400" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">cudaEvent 跨 Stream 依赖管理</text>

  <!-- Stream A -->
  <text x="380" y="72" text-anchor="middle" font-size="14" fill="#8b949e">场景：Stream C 必须等待 Stream A 和 Stream B 的 D2H 都完成</text>
  <g transform="translate(80, 90)">
    <text x="0" y="22" font-size="13" fill="#58a6ff" font-weight="bold">Stream A</text>
    <rect x="80" y="5" width="100" height="30" rx="4" fill="#58a6ff" opacity="0.3" stroke="#58a6ff"/>
    <text x="130" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">Kernel A</text>
    <rect x="180" y="5" width="100" height="30" rx="4" fill="#a371f7" opacity="0.3" stroke="#a371f7"/>
    <text x="230" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">D2H A</text>
    <circle cx="290" cy="20" r="8" fill="#d29922"/><text x="290" y="24" text-anchor="middle" font-size="8" fill="#0d1117" font-weight="bold">E1</text>
    <text x="310" y="24" font-size="10" fill="#d29922">cudaEventRecord(event1, streamA)</text>
  </g>

  <g transform="translate(80, 140)">
    <text x="0" y="22" font-size="13" fill="#3fb950" font-weight="bold">Stream B</text>
    <rect x="80" y="5" width="100" height="30" rx="4" fill="#3fb950" opacity="0.3" stroke="#3fb950"/>
    <text x="130" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">Kernel B</text>
    <rect x="180" y="5" width="100" height="30" rx="4" fill="#a371f7" opacity="0.3" stroke="#a371f7"/>
    <text x="230" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">D2H B</text>
    <circle cx="290" cy="20" r="8" fill="#d29922"/><text x="290" y="24" text-anchor="middle" font-size="8" fill="#0d1117" font-weight="bold">E2</text>
    <text x="310" y="24" font-size="10" fill="#d29922">cudaEventRecord(event2, streamB)</text>
  </g>

  <!-- Arrows to Stream C -->
  <line x1="370" y1="110" x2="370" y2="195" stroke="#d29922" stroke-width="2" stroke-dasharray="4" marker-end="url(#arrY)"/>
  <line x1="370" y1="160" x2="370" y2="195" stroke="#d29922" stroke-width="2" stroke-dasharray="4" marker-end="url(#arrY)"/>
  <text x="390" y="180" font-size="10" fill="#d29922">cudaStreamWaitEvent</text>

  <g transform="translate(80, 200)">
    <text x="0" y="22" font-size="13" fill="#e3b341" font-weight="bold">Stream C</text>
    <rect x="290" y="5" width="120" height="30" rx="4" fill="#d29922" opacity="0.3" stroke="#e3b341"/>
    <text x="350" y="25" text-anchor="middle" font-size="11" fill="#c9d1d9">Post-process</text>
    <text x="420" y="24" font-size="10" fill="#3fb950">← 仅在 E1 和 E2 都完成后才开始</text>
  </g>

  <!-- Code snippet -->
  <g transform="translate(80, 260)">
    <rect x="0" y="0" width="600" height="120" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="15" y="25" font-size="12" fill="#8b949e" font-family="monospace">// 1. 在 Stream A 和 B 中记录事件</text>
    <text x="15" y="45" font-size="12" fill="#58a6ff" font-family="monospace">cudaEventRecord(event1, streamA);</text>
    <text x="15" y="65" font-size="12" fill="#3fb950" font-family="monospace">cudaEventRecord(event2, streamB);</text>
    <text x="15" y="85" font-size="12" fill="#8b949e" font-family="monospace">// 2. Stream C 等待两个事件</text>
    <text x="15" y="105" font-size="12" fill="#d29922" font-family="monospace">cudaStreamWaitEvent(streamC, event1, 0);</text>
    <text x="300" y="105" font-size="12" fill="#d29922" font-family="monospace">cudaStreamWaitEvent(streamC, event2, 0);</text>
  </g>
</svg>'''


def main() -> None:
    diagrams = {
        # Day 1: Warp Shuffle
        "warp_shuffle_primitives.svg": warp_shuffle_primitives(),
        "butterfly_reduction.svg": butterfly_reduction(),
        "two_level_reduction.svg": two_level_reduction(),
        # Day 2: Register Blocking
        "register_blocking_dataflow.svg": register_blocking_dataflow(),
        "thread_tile_mapping.svg": thread_tile_mapping(),
        "double_buffering.svg": double_buffering(),
        # Day 3: CUDA Streams
        "multi_stream_overlap.svg": multi_stream_overlap(),
        "default_stream_sync.svg": default_stream_sync(),
        "stream_event_dependency.svg": stream_event_dependency(),
    }

    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
