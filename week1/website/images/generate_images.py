#!/usr/bin/env python3
"""Generate SVG diagrams for Week 1 website."""

from pathlib import Path


def save_svg(filename: str, content: str) -> None:
    path = Path(__file__).parent / filename
    path.write_text(content, encoding="utf-8")
    print(f"Generated: {path}")


def gpu_memory_hierarchy() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <defs>
    <linearGradient id="gradFast" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#58a6ff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1f6feb;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="gradSlow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#d29922;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#9e6a03;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="720" height="420" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">GPU 内存层次结构</text>

  <!-- Pyramid levels -->
  <polygon points="360,70 520,130 200,130" fill="url(#gradFast)"/>
  <polygon points="200,130 520,130 540,190 180,190" fill="#4c8dff"/>
  <polygon points="180,190 540,190 560,250 160,250" fill="#6c9fff"/>
  <polygon points="160,250 560,250 580,310 140,310" fill="#8bb3ff"/>
  <polygon points="140,310 580,310 600,370 120,370" fill="url(#gradSlow)"/>

  <!-- Labels -->
  <text x="360" y="112" text-anchor="middle" font-size="15" font-weight="bold" fill="#0d1117">Register (~1 cycle)</text>
  <text x="360" y="167" text-anchor="middle" font-size="15" font-weight="bold" fill="#0d1117">Shared Memory / L1 (~20-30 cycles)</text>
  <text x="360" y="227" text-anchor="middle" font-size="15" font-weight="bold" fill="#0d1117">L2 Cache (~200 cycles)</text>
  <text x="360" y="287" text-anchor="middle" font-size="15" font-weight="bold" fill="#0d1117">Global Memory (HBM/GDDR)</text>
  <text x="360" y="348" text-anchor="middle" font-size="15" font-weight="bold" fill="#0d1117">~400-800 cycles</text>

  <!-- Arrow -->
  <text x="620" y="220" text-anchor="start" font-size="14" fill="#8b949e">慢</text>
  <line x1="610" y1="200" x2="610" y2="360" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowDown)"/>
</svg>'''


def sm_architecture() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440">
  <rect width="720" height="440" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">SM (Streaming Multiprocessor) 架构简图</text>

  <!-- SM box -->
  <rect x="60" y="60" width="600" height="360" rx="12" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="88" text-anchor="middle" font-size="16" font-weight="bold" fill="#58a6ff">一个 Streaming Multiprocessor</text>

  <!-- Components -->
  <rect x="90" y="110" width="160" height="80" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="170" y="145" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">CUDA Cores</text>
  <text x="170" y="168" text-anchor="middle" font-size="12" fill="#8b949e">整数 / FP32 / FP64</text>

  <rect x="280" y="110" width="160" height="80" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="145" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Tensor Cores</text>
  <text x="360" y="168" text-anchor="middle" font-size="12" fill="#8b949e">矩阵计算加速</text>

  <rect x="470" y="110" width="160" height="80" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="550" y="145" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Warp Scheduler</text>
  <text x="550" y="168" text-anchor="middle" font-size="12" fill="#8b949e">调度 32-thread warp</text>

  <rect x="90" y="220" width="220" height="80" rx="8" fill="#238636" opacity="0.3" stroke="#3fb950" stroke-width="2"/>
  <text x="200" y="255" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Register File</text>
  <text x="200" y="278" text-anchor="middle" font-size="12" fill="#8b949e">~256 KB / SM</text>

  <rect x="350" y="220" width="280" height="80" rx="8" fill="#d29922" opacity="0.3" stroke="#e3b341" stroke-width="2"/>
  <text x="490" y="255" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Shared Memory / L1 Cache</text>
  <text x="490" y="278" text-anchor="middle" font-size="12" fill="#8b949e">~100-164 KB / SM</text>

  <rect x="90" y="330" width="540" height="60" rx="8" fill="#8957e5" opacity="0.3" stroke="#a371f7" stroke-width="2"/>
  <text x="360" y="365" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Load / Store 单元</text>
</svg>'''


def coalesced_access() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="320" viewBox="0 0 720 320">
  <rect width="720" height="320" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Coalesced Global Memory Access</text>

  <!-- Threads -->
  <text x="80" y="80" text-anchor="middle" font-size="14" fill="#8b949e">Warp Threads</text>
  <circle cx="80" cy="110" r="14" fill="#58a6ff"/>
  <text x="80" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T0</text>
  <circle cx="80" cy="150" r="14" fill="#58a6ff"/>
  <text x="80" y="155" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T1</text>
  <circle cx="80" cy="190" r="14" fill="#58a6ff"/>
  <text x="80" y="195" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T2</text>
  <circle cx="80" cy="230" r="14" fill="#58a6ff"/>
  <text x="80" y="235" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T3</text>

  <!-- Arrows -->
  <line x1="94" y1="110" x2="250" y2="110" stroke="#58a6ff" stroke-width="2"/>
  <line x1="94" y1="150" x2="250" y2="150" stroke="#58a6ff" stroke-width="2"/>
  <line x1="94" y1="190" x2="250" y2="190" stroke="#58a6ff" stroke-width="2"/>
  <line x1="94" y1="230" x2="250" y2="230" stroke="#58a6ff" stroke-width="2"/>

  <!-- Memory blocks -->
  <rect x="250" y="90" width="60" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="280" y="115" text-anchor="middle" font-size="12" fill="#c9d1d9">addr+0</text>
  <rect x="310" y="90" width="60" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="340" y="115" text-anchor="middle" font-size="12" fill="#c9d1d9">addr+4</text>
  <rect x="370" y="90" width="60" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="400" y="115" text-anchor="middle" font-size="12" fill="#c9d1d9">addr+8</text>
  <rect x="430" y="90" width="60" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="460" y="115" text-anchor="middle" font-size="12" fill="#c9d1d9">addr+12</text>

  <text x="250" y="75" font-size="13" fill="#8b949e">Global Memory（连续地址）</text>

  <!-- Result -->
  <rect x="560" y="150" width="120" height="50" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="620" y="172" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">合并为</text>
  <text x="620" y="190" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">1 次事务</text>
</svg>'''


def bank_conflict() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Shared Memory Bank Conflict</text>

  <!-- Banks -->
  <text x="360" y="70" text-anchor="middle" font-size="15" fill="#8b949e">Shared Memory：32 banks，每 bank 4 bytes</text>

  <!-- Bank columns -->
  <g transform="translate(120, 100)">
    <!-- Headers -->
    <rect x="0" y="0" width="40" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="20" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">B0</text>
    <rect x="40" y="0" width="40" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="60" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">B1</text>
    <rect x="80" y="0" width="40" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="100" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">B2</text>
    <rect x="120" y="0" width="40" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="140" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">...</text>
    <rect x="160" y="0" width="40" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="180" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">B31</text>

    <!-- Data row -->
    <rect x="0" y="30" width="40" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
    <rect x="40" y="30" width="40" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
    <rect x="80" y="30" width="40" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
    <rect x="120" y="30" width="40" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
    <rect x="160" y="30" width="40" height="40" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  </g>

  <!-- Conflict case -->
  <text x="360" y="210" text-anchor="middle" font-size="16" font-weight="bold" fill="#f85149">❌ Bank Conflict：同一 warp 的多个线程同时访问同一个 bank</text>
  <circle cx="200" cy="260" r="16" fill="#f85149"/>
  <text x="200" y="265" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">T0</text>
  <circle cx="240" cy="260" r="16" fill="#f85149"/>
  <text x="240" y="265" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">T1</text>
  <circle cx="280" cy="260" r="16" fill="#f85149"/>
  <text x="280" y="265" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">T2</text>

  <line x1="200" y1="276" x2="200" y2="330" stroke="#f85149" stroke-width="2"/>
  <line x1="240" y1="276" x2="240" y2="330" stroke="#f85149" stroke-width="2"/>
  <line x1="280" y1="276" x2="280" y2="330" stroke="#f85149" stroke-width="2"/>

  <text x="240" y="350" text-anchor="middle" font-size="13" fill="#f85149">T0/T1/T2 都访问 Bank 0 → 串行执行</text>

  <!-- Padding fix -->
  <text x="360" y="390" text-anchor="middle" font-size="14" fill="#3fb950">✅ 解决：tile[TILE_DIM][TILE_DIM + 1] padding 让数据错开 bank</text>
</svg>'''


def roofline_model() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440">
  <rect width="720" height="440" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Roofline 模型</text>

  <!-- Axes -->
  <line x1="80" y1="380" x2="660" y2="380" stroke="#8b949e" stroke-width="2"/>
  <line x1="80" y1="380" x2="80" y2="60" stroke="#8b949e" stroke-width="2"/>

  <text x="360" y="410" text-anchor="middle" font-size="14" fill="#c9d1d9">Arithmetic Intensity (FLOPs / byte) →</text>
  <text x="30" y="220" text-anchor="middle" font-size="14" fill="#c9d1d9" transform="rotate(-90, 30, 220)">Attainable FLOP/s →</text>

  <!-- Roofline -->
  <polyline points="80,340 280,180 660,180" fill="none" stroke="#58a6ff" stroke-width="3"/>
  <text x="600" y="165" font-size="13" fill="#58a6ff" font-weight="bold">Peak Compute</text>
  <text x="160" y="320" font-size="13" fill="#d29922" font-weight="bold">Memory-bound slope</text>

  <!-- Regions -->
  <text x="140" y="260" font-size="15" fill="#f85149" font-weight="bold">Memory-Bound</text>
  <text x="480" y="240" font-size="15" fill="#3fb950" font-weight="bold">Compute-Bound</text>

  <!-- Sample points -->
  <circle cx="120" cy="310" r="8" fill="#f85149"/>
  <text x="120" y="300" text-anchor="middle" font-size="12" fill="#c9d1d9">transpose_naive</text>

  <circle cx="200" cy="260" r="8" fill="#f85149"/>
  <text x="200" y="250" text-anchor="middle" font-size="12" fill="#c9d1d9">transpose_optimized</text>

  <circle cx="500" cy="200" r="8" fill="#3fb950"/>
  <text x="500" y="230" text-anchor="middle" font-size="12" fill="#c9d1d9">compute_intensive</text>
</svg>'''


def week1_roadmap() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="500" viewBox="0 0 720 500">
  <rect width="720" height="500" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Week 1 学习地图</text>

  <!-- Nodes -->
  <rect x="260" y="70" width="200" height="50" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="101" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Day 1：GPU 执行模型</text>

  <rect x="260" y="150" width="200" height="50" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="181" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Day 2：Occupancy</text>

  <rect x="260" y="230" width="200" height="50" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="261" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Day 3：CUDA Samples</text>

  <rect x="260" y="310" width="200" height="50" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="341" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Day 4：Memory Hierarchy</text>

  <rect x="260" y="390" width="200" height="50" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="421" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Day 5：Bank Conflict</text>

  <!-- Arrows -->
  <line x1="360" y1="120" x2="360" y2="150" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowDown)"/>
  <line x1="360" y1="200" x2="360" y2="230" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowDown)"/>
  <line x1="360" y1="280" x2="360" y2="310" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowDown)"/>
  <line x1="360" y1="360" x2="360" y2="390" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowDown)"/>

  <!-- Side notes -->
  <text x="490" y="175" font-size="13" fill="#8b949e">资源约束</text>
  <text x="490" y="335" font-size="13" fill="#8b949e">内存优化</text>
</svg>'''



def grid_block_thread() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" viewBox="0 0 720 480">
  <rect width="720" height="480" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Grid / Block / Thread 层次结构</text>

  <!-- Grid box -->
  <rect x="80" y="70" width="560" height="380" rx="12" fill="none" stroke="#58a6ff" stroke-width="3" stroke-dasharray="8,4"/>
  <text x="360" y="100" text-anchor="middle" font-size="18" font-weight="bold" fill="#58a6ff">Grid (gridDim = 2 × 2)</text>

  <!-- Blocks -->
  <rect x="110" y="130" width="240" height="140" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="230" y="155" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Block (0, 0)</text>

  <rect x="370" y="130" width="240" height="140" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="490" y="155" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Block (1, 0)</text>

  <rect x="110" y="290" width="240" height="140" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="230" y="315" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Block (0, 1)</text>

  <rect x="370" y="290" width="240" height="140" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="490" y="315" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Block (1, 1)</text>

  <!-- Threads in first block -->
  <circle cx="150" cy="195" r="14" fill="#3fb950"/>
  <text x="150" y="200" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T0</text>
  <circle cx="190" cy="195" r="14" fill="#3fb950"/>
  <text x="190" y="200" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T1</text>
  <circle cx="230" cy="195" r="14" fill="#3fb950"/>
  <text x="230" y="200" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T2</text>
  <circle cx="270" cy="195" r="14" fill="#3fb950"/>
  <text x="270" y="200" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T3</text>

  <circle cx="150" cy="240" r="14" fill="#3fb950"/>
  <text x="150" y="245" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T4</text>
  <circle cx="190" cy="240" r="14" fill="#3fb950"/>
  <text x="190" y="245" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T5</text>
  <circle cx="230" cy="240" r="14" fill="#3fb950"/>
  <text x="230" y="245" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T6</text>
  <circle cx="270" cy="240" r="14" fill="#3fb950"/>
  <text x="270" y="245" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T7</text>

  <text x="230" y="270" text-anchor="middle" font-size="12" fill="#8b949e">blockDim = 4 × 2 = 8 threads</text>

  <!-- Legend -->
  <text x="360" y="450" text-anchor="middle" font-size="13" fill="#8b949e">总线程数 = gridDim.x × gridDim.y × blockDim.x × blockDim.y = 2 × 2 × 8 = 32</text>
</svg>'''


def warp_divergence() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Warp Divergence（分支发散）</text>

  <!-- Warp threads -->
  <text x="80" y="80" text-anchor="start" font-size="15" fill="#8b949e">一个 Warp 中的 8 个线程（实际为 32 个）</text>
  <circle cx="100" cy="110" r="16" fill="#58a6ff"/>
  <text x="100" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T0</text>
  <circle cx="150" cy="110" r="16" fill="#58a6ff"/>
  <text x="150" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T1</text>
  <circle cx="200" cy="110" r="16" fill="#58a6ff"/>
  <text x="200" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T2</text>
  <circle cx="250" cy="110" r="16" fill="#58a6ff"/>
  <text x="250" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T3</text>
  <circle cx="300" cy="110" r="16" fill="#f85149"/>
  <text x="300" y="115" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">T4</text>
  <circle cx="350" cy="110" r="16" fill="#f85149"/>
  <text x="350" y="115" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">T5</text>
  <circle cx="400" cy="110" r="16" fill="#58a6ff"/>
  <text x="400" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T6</text>
  <circle cx="450" cy="110" r="16" fill="#58a6ff"/>
  <text x="450" y="115" text-anchor="middle" font-size="10" fill="#0d1117" font-weight="bold">T7</text>

  <!-- Code -->
  <rect x="80" y="150" width="560" height="60" rx="8" fill="#1f2937" stroke="#30363d"/>
  <text x="100" y="180" font-family="monospace" font-size="14" fill="#c9d1d9">if (threadIdx.x % 2 == 0) { /* 蓝色线程执行 */ }</text>
  <text x="100" y="200" font-family="monospace" font-size="14" fill="#c9d1d9">else { /* 红色线程执行 */ }</text>

  <!-- Divergence visualization -->
  <rect x="80" y="240" width="300" height="100" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="230" y="275" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">路径 A：蓝色线程</text>
  <text x="230" y="300" text-anchor="middle" font-size="13" fill="#8b949e">执行的线程：T0, T1, T2, T3, T6, T7</text>
  <text x="230" y="322" text-anchor="middle" font-size="13" fill="#8b949e">红色线程被 mask 掉</text>

  <rect x="400" y="240" width="300" height="100" rx="8" fill="#f85149" opacity="0.3" stroke="#f85149" stroke-width="2"/>
  <text x="550" y="275" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">路径 B：红色线程</text>
  <text x="550" y="300" text-anchor="middle" font-size="13" fill="#8b949e">执行的线程：T4, T5</text>
  <text x="550" y="322" text-anchor="middle" font-size="13" fill="#8b949e">蓝色线程被 mask 掉</text>

  <text x="360" y="370" text-anchor="middle" font-size="14" fill="#f85149" font-weight="bold">两条路径串行执行 → 性能下降</text>
</svg>'''


def simt_vs_simd() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">
  <rect width="720" height="360" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">SIMT vs SIMD</text>

  <!-- SIMD -->
  <text x="180" y="80" text-anchor="middle" font-size="18" font-weight="bold" fill="#c9d1d9">SIMD</text>
  <text x="180" y="105" text-anchor="middle" font-size="13" fill="#8b949e">Single Instruction Multiple Data</text>

  <rect x="80" y="130" width="200" height="140" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="180" y="165" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">一条指令</text>
  <text x="180" y="190" text-anchor="middle" font-size="13" fill="#8b949e">同时处理多组数据</text>
  <text x="180" y="215" text-anchor="middle" font-size="13" fill="#8b949e">数据向量宽度固定</text>
  <text x="180" y="240" text-anchor="middle" font-size="13" fill="#8b949e">如：AVX-512</text>

  <!-- SIMT -->
  <text x="540" y="80" text-anchor="middle" font-size="18" font-weight="bold" fill="#c9d1d9">SIMT</text>
  <text x="540" y="105" text-anchor="middle" font-size="13" fill="#8b949e">Single Instruction Multiple Threads</text>

  <rect x="440" y="130" width="200" height="140" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="540" y="165" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">一条指令</text>
  <text x="540" y="190" text-anchor="middle" font-size="13" fill="#8b949e">多个线程各自执行</text>
  <text x="540" y="215" text-anchor="middle" font-size="13" fill="#8b949e">每个线程有独立 PC</text>
  <text x="540" y="240" text-anchor="middle" font-size="13" fill="#8b949e">如：NVIDIA GPU Warp</text>

  <!-- Comparison -->
  <text x="360" y="310" text-anchor="middle" font-size="15" fill="#c9d1d9" font-weight="bold">SIMT 可以模拟 SIMD，但 SIMD 无法模拟 SIMT 的分支行为</text>
</svg>'''


def occupancy_concept() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">
  <rect width="720" height="360" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Occupancy 概念</text>

  <!-- SM slot -->
  <rect x="80" y="80" width="560" height="120" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="110" text-anchor="middle" font-size="16" font-weight="bold" fill="#c9d1d9">一个 SM 的 Warp 槽位（例如最大 32 个 warp）</text>

  <!-- Active warps -->
  <rect x="100" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="140" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="180" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="220" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="260" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="300" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="340" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <rect x="380" y="130" width="30" height="50" fill="#238636" stroke="#3fb950" stroke-width="2"/>

  <!-- Empty slots -->
  <rect x="420" y="130" width="30" height="50" fill="#30363d" stroke="#484f58" stroke-width="2"/>
  <rect x="460" y="130" width="30" height="50" fill="#30363d" stroke="#484f58" stroke-width="2"/>
  <rect x="500" y="130" width="30" height="50" fill="#30363d" stroke="#484f58" stroke-width="2"/>
  <rect x="540" y="130" width="30" height="50" fill="#30363d" stroke="#484f58" stroke-width="2"/>

  <text x="240" y="205" text-anchor="middle" font-size="13" fill="#3fb950" font-weight="bold">Active Warps = 8</text>
  <text x="500" y="205" text-anchor="middle" font-size="13" fill="#8b949e">Empty Slots = 4</text>

  <!-- Formula -->
  <rect x="80" y="240" width="560" height="80" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="275" text-anchor="middle" font-size="18" font-weight="bold" fill="#c9d1d9">Occupancy = 8 / 12 = 67%</text>
  <text x="360" y="305" text-anchor="middle" font-size="14" fill="#8b949e">实际中分母是 SM 支持的最大 warp 数</text>
</svg>'''


def resource_constraints() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">影响 Occupancy 的三大资源约束</text>

  <!-- Registers -->
  <rect x="80" y="80" width="180" height="120" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="170" y="115" text-anchor="middle" font-size="15" font-weight="bold" fill="#c9d1d9">寄存器 Registers</text>
  <text x="170" y="145" text-anchor="middle" font-size="12" fill="#8b949e">每个 SM 有限</text>
  <text x="170" y="165" text-anchor="middle" font-size="12" fill="#8b949e">线程用得越多</text>
  <text x="170" y="185" text-anchor="middle" font-size="12" fill="#8b949e">同时驻留 warp 越少</text>

  <!-- Shared Memory -->
  <rect x="270" y="80" width="180" height="120" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="360" y="115" text-anchor="middle" font-size="15" font-weight="bold" fill="#c9d1d9">共享内存 Shared Mem</text>
  <text x="360" y="145" text-anchor="middle" font-size="12" fill="#8b949e">每个 SM 有限</text>
  <text x="360" y="165" text-anchor="middle" font-size="12" fill="#8b949e">block 用得越多</text>
  <text x="360" y="185" text-anchor="middle" font-size="12" fill="#8b949e">同时驻留 block 越少</text>

  <!-- Block Size -->
  <rect x="460" y="80" width="180" height="120" rx="8" fill="#8957e5" opacity="0.2" stroke="#a371f7" stroke-width="2"/>
  <text x="550" y="115" text-anchor="middle" font-size="15" font-weight="bold" fill="#c9d1d9">Block 大小与数量</text>
  <text x="550" y="145" text-anchor="middle" font-size="12" fill="#8b949e">最大 thread/block</text>
  <text x="550" y="165" text-anchor="middle" font-size="12" fill="#8b949e">最大 block/SM</text>
  <text x="550" y="185" text-anchor="middle" font-size="12" fill="#8b949e">最大 warp/SM</text>

  <!-- SM box -->
  <rect x="80" y="240" width="560" height="120" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="275" text-anchor="middle" font-size="16" font-weight="bold" fill="#c9d1d9">SM 资源池</text>
  <text x="360" y="305" text-anchor="middle" font-size="13" fill="#8b949e">Register File + Shared Memory + Warp Slots</text>
  <text x="360" y="335" text-anchor="middle" font-size="13" fill="#f85149" font-weight="bold">任一资源耗尽都会限制 Occupancy</text>
</svg>'''


def register_spilling() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="380" viewBox="0 0 720 380">
  <rect width="720" height="380" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Register Spilling（寄存器溢出）</text>

  <!-- Thread -->
  <rect x="80" y="80" width="200" height="240" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="180" y="110" text-anchor="middle" font-size="15" font-weight="bold" fill="#c9d1d9">一个 Thread</text>

  <!-- Registers -->
  <rect x="100" y="140" width="160" height="50" rx="4" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="180" y="170" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">寄存器（快）</text>

  <text x="180" y="215" text-anchor="middle" font-size="12" fill="#f85149" font-weight="bold">变量太多，寄存器不够</text>
  <line x1="180" y1="225" x2="180" y2="250" stroke="#f85149" stroke-width="2" marker-end="url(#arrowDown)"/>

  <!-- Local memory -->
  <rect x="100" y="260" width="160" height="50" rx="4" fill="#d29922" stroke="#e3b341" stroke-width="2"/>
  <text x="180" y="290" text-anchor="middle" font-size="13" fill="#0d1117" font-weight="bold">Local Memory（慢）</text>

  <!-- Note -->
  <rect x="320" y="120" width="340" height="160" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="490" y="155" text-anchor="middle" font-size="14" font-weight="bold" fill="#c9d1d9">Local Memory 真相</text>
  <text x="490" y="185" text-anchor="middle" font-size="13" fill="#8b949e">实际存储在 Global Memory 中</text>
  <text x="490" y="210" text-anchor="middle" font-size="13" fill="#8b949e">访问延迟 ~400-800 cycles</text>
  <text x="490" y="235" text-anchor="middle" font-size="13" fill="#8b949e">性能急剧下降</text>
  <text x="490" y="265" text-anchor="middle" font-size="13" fill="#f85149" font-weight="bold">应尽量避免 spilling</text>
</svg>'''


def occupancy_curve() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Occupancy 与性能关系</text>

  <!-- Axes -->
  <line x1="80" y1="340" x2="640" y2="340" stroke="#8b949e" stroke-width="2"/>
  <line x1="80" y1="340" x2="80" y2="60" stroke="#8b949e" stroke-width="2"/>

  <text x="360" y="375" text-anchor="middle" font-size="14" fill="#c9d1d9">Occupancy →</text>
  <text x="40" y="200" text-anchor="middle" font-size="14" fill="#c9d1d9" transform="rotate(-90, 40, 200)">Performance →</text>

  <!-- Curve -->
  <path d="M 80,320 Q 160,300 240,220 T 400,140 T 640,120" fill="none" stroke="#58a6ff" stroke-width="3"/>

  <!-- Regions -->
  <text x="140" y="290" font-size="13" fill="#f85149" font-weight="bold">低 Occupancy</text>
  <text x="140" y="310" font-size="12" fill="#8b949e">无法隐藏延迟</text>

  <text x="380" y="170" font-size="13" fill="#3fb950" font-weight="bold">足够隐藏延迟</text>
  <text x="380" y="150" font-size="12" fill="#8b949e">再提升收益有限</text>

  <!-- Dashed line -->
  <line x1="80" y1="120" x2="640" y2="120" stroke="#8b949e" stroke-width="1" stroke-dasharray="5,5"/>
  <text x="650" y="125" font-size="12" fill="#8b949e">理论峰值</text>

  <text x="360" y="250" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">目标：occupancy 足够高即可，不必追求 100%</text>
</svg>'''


def device_query_output() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440">
  <rect width="720" height="440" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">deviceQuery 输出示例结构</text>

  <!-- Terminal window -->
  <rect x="60" y="60" width="600" height="340" rx="10" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <rect x="60" y="60" width="600" height="30" rx="10" fill="#30363d"/>
  <circle cx="80" cy="75" r="6" fill="#f85149"/>
  <circle cx="100" cy="75" r="6" fill="#d29922"/>
  <circle cx="120" cy="75" r="6" fill="#3fb950"/>

  <!-- Output lines -->
  <text x="80" y="120" font-family="monospace" font-size="13" fill="#58a6ff">Detected 1 CUDA Capable device(s)</text>
  <text x="80" y="150" font-family="monospace" font-size="13" fill="#c9d1d9">Device 0: "NVIDIA A100-PCIE-40GB"</text>
  <text x="80" y="180" font-family="monospace" font-size="13" fill="#8b949e">  CUDA Capability Major/Minor version number:    8.0</text>
  <text x="80" y="205" font-family="monospace" font-size="13" fill="#8b949e">  Total amount of global memory:                 40536 MBytes</text>
  <text x="80" y="230" font-family="monospace" font-size="13" fill="#8b949e">  (108) Multiprocessors, (64) CUDA Cores/MP:     6912 CUDA Cores</text>
  <text x="80" y="255" font-family="monospace" font-size="13" fill="#8b949e">  GPU Max Clock rate:                            1410 MHz</text>
  <text x="80" y="280" font-family="monospace" font-size="13" fill="#8b949e">  Memory Clock rate:                             1215 Mhz</text>
  <text x="80" y="305" font-family="monospace" font-size="13" fill="#8b949e">  Memory Bus Width:                              5120-bit</text>
  <text x="80" y="330" font-family="monospace" font-size="13" fill="#8b949e">  Maximum number of threads per multiprocessor:  2048</text>
  <text x="80" y="355" font-family="monospace" font-size="13" fill="#8b949e">  Maximum number of threads per block:           1024</text>
  <text x="80" y="380" font-family="monospace" font-size="13" fill="#3fb950">Result = PASS</text>
</svg>'''


def occupancy_calculator_workflow() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">CUDA Occupancy Calculator 使用流程</text>

  <!-- Steps -->
  <rect x="60" y="80" width="140" height="80" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="130" y="115" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">1. 输入 GPU</text>
  <text x="130" y="140" text-anchor="middle" font-size="12" fill="#8b949e">Compute Capability</text>

  <line x1="200" y1="120" x2="240" y2="120" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowRight)"/>

  <rect x="250" y="80" width="140" height="80" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="320" y="115" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">2. 输入 Kernel</text>
  <text x="320" y="140" text-anchor="middle" font-size="12" fill="#8b949e">Block / Regs / Shared</text>

  <line x1="390" y1="120" x2="430" y2="120" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowRight)"/>

  <rect x="440" y="80" width="140" height="80" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="510" y="115" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">3. 获取结果</text>
  <text x="510" y="140" text-anchor="middle" font-size="12" fill="#8b949e">理论 Occupancy</text>

  <!-- Input details -->
  <rect x="60" y="200" width="280" height="160" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="200" y="230" text-anchor="middle" font-size="14" font-weight="bold" fill="#c9d1d9">输入参数</text>
  <text x="80" y="260" font-size="12" fill="#8b949e">• Threads per block</text>
  <text x="80" y="285" font-size="12" fill="#8b949e">• Registers per thread</text>
  <text x="80" y="310" font-size="12" fill="#8b949e">• Shared memory per block</text>
  <text x="80" y="335" font-size="12" fill="#8b949e">• GPU Compute Capability</text>

  <!-- Output details -->
  <rect x="380" y="200" width="280" height="160" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="520" y="230" text-anchor="middle" font-size="14" font-weight="bold" fill="#c9d1d9">输出结果</text>
  <text x="400" y="260" font-size="12" fill="#8b949e">• Active Warps per SM</text>
  <text x="400" y="285" font-size="12" fill="#8b949e">• Occupancy (%)</text>
  <text x="400" y="310" font-size="12" fill="#8b949e">• Active Blocks per SM</text>
  <text x="400" y="335" font-size="12" fill="#8b949e">• 哪个资源是瓶颈</text>
</svg>'''


def cuda_guide_ch5() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">CUDA C Programming Guide 第 5 章核心要点</text>

  <!-- Central node -->
  <rect x="280" y="90" width="160" height="60" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="125" text-anchor="middle" font-size="15" fill="#c9d1d9" font-weight="bold">Performance Guidelines</text>

  <!-- Branches -->
  <rect x="60" y="200" width="150" height="80" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="135" y="235" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Memory Coalescing</text>
  <text x="135" y="260" text-anchor="middle" font-size="11" fill="#8b949e">合并全局内存访问</text>

  <rect x="220" y="200" width="150" height="80" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="295" y="235" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Shared Memory</text>
  <text x="295" y="260" text-anchor="middle" font-size="11" fill="#8b949e">Bank conflict / Tiling</text>

  <rect x="380" y="200" width="150" height="80" rx="8" fill="#8957e5" opacity="0.2" stroke="#a371f7" stroke-width="2"/>
  <text x="455" y="235" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Occupancy</text>
  <text x="455" y="260" text-anchor="middle" font-size="11" fill="#8b949e">隐藏延迟能力</text>

  <rect x="540" y="200" width="150" height="80" rx="8" fill="#f85149" opacity="0.2" stroke="#f85149" stroke-width="2"/>
  <text x="615" y="235" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Instruction Throughput</text>
  <text x="615" y="260" text-anchor="middle" font-size="11" fill="#8b949e">指令吞吐优化</text>

  <!-- Lines -->
  <line x1="320" y1="150" x2="135" y2="200" stroke="#8b949e" stroke-width="2"/>
  <line x1="360" y1="150" x2="295" y2="200" stroke="#8b949e" stroke-width="2"/>
  <line x1="400" y1="150" x2="455" y2="200" stroke="#8b949e" stroke-width="2"/>
  <line x1="440" y1="150" x2="615" y2="200" stroke="#8b949e" stroke-width="2"/>

  <!-- Bottom note -->
  <text x="360" y="330" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">这一章是 CUDA 性能优化的官方圣经</text>
  <text x="360" y="355" text-anchor="middle" font-size="13" fill="#8b949e">建议通读一遍，后续 Day 4-6 会反复用到这些概念</text>
</svg>'''


def stride_access() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">
  <rect width="720" height="360" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Coalesced vs Stride Access</text>

  <!-- Coalesced -->
  <text x="180" y="80" text-anchor="middle" font-size="16" font-weight="bold" fill="#3fb950">✅ Coalesced</text>

  <circle cx="80" cy="120" r="14" fill="#58a6ff"/>
  <text x="80" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T0</text>
  <circle cx="120" cy="120" r="14" fill="#58a6ff"/>
  <text x="120" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T1</text>
  <circle cx="160" cy="120" r="14" fill="#58a6ff"/>
  <text x="160" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T2</text>
  <circle cx="200" cy="120" r="14" fill="#58a6ff"/>
  <text x="200" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T3</text>

  <line x1="80" y1="134" x2="80" y2="170" stroke="#58a6ff" stroke-width="2"/>
  <line x1="120" y1="134" x2="120" y2="170" stroke="#58a6ff" stroke-width="2"/>
  <line x1="160" y1="134" x2="160" y2="170" stroke="#58a6ff" stroke-width="2"/>
  <line x1="200" y1="134" x2="200" y2="170" stroke="#58a6ff" stroke-width="2"/>

  <rect x="70" y="170" width="150" height="40" rx="4" fill="#238636" stroke="#3fb950" stroke-width="2"/>
  <text x="145" y="195" text-anchor="middle" font-size="12" fill="#0d1117" font-weight="bold">1 次内存事务</text>

  <!-- Stride -->
  <text x="540" y="80" text-anchor="middle" font-size="16" font-weight="bold" fill="#f85149">❌ Stride</text>

  <circle cx="440" cy="120" r="14" fill="#58a6ff"/>
  <text x="440" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T0</text>
  <circle cx="480" cy="120" r="14" fill="#58a6ff"/>
  <text x="480" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T1</text>
  <circle cx="520" cy="120" r="14" fill="#58a6ff"/>
  <text x="520" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T2</text>
  <circle cx="560" cy="120" r="14" fill="#58a6ff"/>
  <text x="560" y="125" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T3</text>

  <line x1="440" y1="134" x2="440" y2="170" stroke="#58a6ff" stroke-width="2"/>
  <line x1="480" y1="134" x2="520" y2="170" stroke="#58a6ff" stroke-width="2"/>
  <line x1="520" y1="134" x2="600" y2="170" stroke="#58a6ff" stroke-width="2"/>
  <line x1="560" y1="134" x2="680" y2="170" stroke="#58a6ff" stroke-width="2"/>

  <rect x="430" y="170" width="60" height="40" rx="4" fill="#d29922" stroke="#e3b341" stroke-width="2"/>
  <rect x="500" y="170" width="60" height="40" rx="4" fill="#d29922" stroke="#e3b341" stroke-width="2"/>
  <rect x="570" y="170" width="60" height="40" rx="4" fill="#d29922" stroke="#e3b341" stroke-width="2"/>
  <rect x="640" y="170" width="60" height="40" rx="4" fill="#d29922" stroke="#e3b341" stroke-width="2"/>
  <text x="565" y="235" text-anchor="middle" font-size="12" fill="#f85149" font-weight="bold">4 次内存事务</text>

  <!-- Code examples -->
  <rect x="60" y="270" width="280" height="70" rx="8" fill="#1f2937" stroke="#30363d"/>
  <text x="70" y="295" font-family="monospace" font-size="12" fill="#c9d1d9">// Coalesced</text>
  <text x="70" y="315" font-family="monospace" font-size="12" fill="#c9d1d9">x[idx]</text>

  <rect x="380" y="270" width="320" height="70" rx="8" fill="#1f2937" stroke="#30363d"/>
  <text x="390" y="295" font-family="monospace" font-size="12" fill="#c9d1d9">// Stride</text>
  <text x="390" y="315" font-family="monospace" font-size="12" fill="#c9d1d9">x[idx * 32]</text>
</svg>'''


def shared_memory_tiling() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Shared Memory Tiling 原理</text>

  <!-- Global memory matrix -->
  <rect x="60" y="80" width="240" height="240" fill="none" stroke="#58a6ff" stroke-width="2"/>
  <text x="180" y="70" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Global Memory（大矩阵）</text>

  <!-- Tiles in global -->
  <rect x="60" y="80" width="80" height="80" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="100" y="125" text-anchor="middle" font-size="11" fill="#c9d1d9">Tile 0</text>
  <rect x="140" y="80" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="220" y="80" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="60" y="160" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="140" y="160" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="220" y="160" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="60" y="240" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="140" y="240" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>
  <rect x="220" y="240" width="80" height="80" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="1"/>

  <!-- Shared memory tile -->
  <rect x="420" y="120" width="120" height="120" rx="4" fill="#238636" opacity="0.3" stroke="#3fb950" stroke-width="2"/>
  <text x="480" y="115" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Shared Memory（Tile）</text>

  <line x1="320" y1="160" x2="420" y2="180" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowRight)"/>
  <text x="370" y="160" text-anchor="middle" font-size="12" fill="#8b949e">加载</text>

  <line x1="480" y1="240" x2="480" y2="300" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowDown)"/>
  <text x="520" y="280" text-anchor="start" font-size="12" fill="#8b949e">在 SM 内快速复用</text>

  <!-- Steps -->
  <rect x="60" y="350" width="600" height="50" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="380" text-anchor="middle" font-size="13" fill="#c9d1d9">1. 将大矩阵分块  →  2. 加载一个 tile 到 Shared Memory  →  3. 在线程间复用数据</text>
</svg>'''


def matrix_transpose() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">矩阵转置的内存访问模式</text>

  <!-- Input matrix -->
  <rect x="80" y="80" width="200" height="200" fill="none" stroke="#58a6ff" stroke-width="2"/>
  <text x="180" y="70" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Input A（行优先）</text>

  <!-- Rows highlighted -->
  <rect x="80" y="110" width="200" height="30" fill="#1f6feb" opacity="0.3"/>
  <rect x="80" y="150" width="200" height="30" fill="#1f6feb" opacity="0.2"/>
  <rect x="80" y="190" width="200" height="30" fill="#1f6feb" opacity="0.1"/>

  <text x="100" y="130" font-size="11" fill="#c9d1d9">Row 0</text>
  <text x="100" y="170" font-size="11" fill="#c9d1d9">Row 1</text>
  <text x="100" y="210" font-size="11" fill="#c9d1d9">Row 2</text>

  <!-- Arrow -->
  <text x="360" y="180" text-anchor="middle" font-size="20" fill="#c9d1d9">→</text>
  <text x="360" y="210" text-anchor="middle" font-size="13" fill="#8b949e">转置</text>

  <!-- Output matrix -->
  <rect x="440" y="80" width="200" height="200" fill="none" stroke="#3fb950" stroke-width="2"/>
  <text x="540" y="70" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Output A^T</text>

  <!-- Columns highlighted (were rows) -->
  <rect x="470" y="80" width="30" height="200" fill="#238636" opacity="0.3"/>
  <rect x="510" y="80" width="30" height="200" fill="#238636" opacity="0.2"/>
  <rect x="550" y="80" width="30" height="200" fill="#238636" opacity="0.1"/>

  <text x="475" y="100" font-size="11" fill="#c9d1d9" transform="rotate(90, 475, 100)">Col 0</text>
  <text x="515" y="100" font-size="11" fill="#c9d1d9" transform="rotate(90, 515, 100)">Col 1</text>
  <text x="555" y="100" font-size="11" fill="#c9d1d9" transform="rotate(90, 555, 100)">Col 2</text>

  <!-- Problem note -->
  <rect x="80" y="310" width="560" height="70" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="340" text-anchor="middle" font-size="13" fill="#f85149" font-weight="bold">问题：读是 coalesced（连续行），写变成 stride access（连续列）</text>
  <text x="360" y="365" text-anchor="middle" font-size="13" fill="#8b949e">解决：用 Shared Memory 做中间缓冲，调整读写模式</text>
</svg>'''


def shared_memory_bank_structure() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Shared Memory Bank 结构</text>

  <text x="360" y="65" text-anchor="middle" font-size="14" fill="#8b949e">32 个 bank，每个 bank 4 bytes（对于 float 类型，每行对应一个 warp 的访问）</text>

  <!-- Bank headers -->
  <g transform="translate(60, 90)">
    <rect x="0" y="0" width="50" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="25" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">Bank 0</text>
    <rect x="50" y="0" width="50" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="75" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">Bank 1</text>
    <rect x="100" y="0" width="50" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="125" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">Bank 2</text>
    <rect x="150" y="0" width="50" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="175" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">...</text>
    <rect x="200" y="0" width="50" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="225" y="20" text-anchor="middle" font-size="11" fill="#c9d1d9">Bank 31</text>
  </g>

  <!-- Address mapping -->
  <text x="60" y="155" font-size="13" fill="#c9d1d9">地址到 bank 的映射：</text>
  <text x="60" y="180" font-family="monospace" font-size="13" fill="#58a6ff">bank = (address / 4) % 32</text>

  <!-- Examples -->
  <rect x="60" y="210" width="600" height="70" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="70" y="235" font-size="12" fill="#c9d1d9">地址 0, 128, 256 ... → Bank 0</text>
  <text x="70" y="255" font-size="12" fill="#c9d1d9">地址 4, 132, 260 ... → Bank 1</text>
  <text x="70" y="275" font-size="12" fill="#c9d1d9">地址 i * 4 且 i % 32 == k → Bank k</text>

  <!-- Note -->
  <text x="360" y="320" text-anchor="middle" font-size="13" fill="#f85149" font-weight="bold">一个 warp 内多个线程访问同一 bank 的不同地址 → Bank Conflict</text>
  <text x="360" y="345" text-anchor="middle" font-size="13" fill="#3fb950">一个 warp 内多个线程访问同一地址 → Broadcast，无 Conflict</text>
  <text x="360" y="370" text-anchor="middle" font-size="13" fill="#8b949e">一个 warp 内线程访问不同 bank → 无 Conflict</text>
</svg>'''


def padding_solution() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Padding 解决 Bank Conflict</text>

  <!-- Without padding -->
  <text x="180" y="70" text-anchor="middle" font-size="16" font-weight="bold" fill="#f85149">❌ 无 Padding</text>
  <text x="180" y="95" text-anchor="middle" font-size="12" fill="#8b949e">float tile[32][32]</text>

  <g transform="translate(60, 110)">
    <rect x="0" y="0" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="15" y="20" text-anchor="middle" font-size="9" fill="#fff">B0</text>
    <rect x="30" y="0" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="45" y="20" text-anchor="middle" font-size="9" fill="#fff">B1</text>
    <rect x="60" y="0" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="75" y="20" text-anchor="middle" font-size="9" fill="#fff">B2</text>
    <rect x="90" y="0" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="105" y="20" text-anchor="middle" font-size="9" fill="#fff">...</text>
    <rect x="120" y="0" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="135" y="20" text-anchor="middle" font-size="9" fill="#fff">B31</text>

    <rect x="0" y="30" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="15" y="50" text-anchor="middle" font-size="9" fill="#fff">B0</text>
    <rect x="30" y="30" width="30" height="30" fill="#f85149" stroke="#ff7b72" stroke-width="2"/>
    <text x="45" y="50" text-anchor="middle" font-size="9" fill="#fff">B1</text>

    <text x="75" y="90" text-anchor="middle" font-size="11" fill="#f85149">同一列的数据都在同一个 bank</text>
    <text x="75" y="110" text-anchor="middle" font-size="11" fill="#f85149">按列读 → 32-way conflict</text>
  </g>

  <!-- With padding -->
  <text x="540" y="70" text-anchor="middle" font-size="16" font-weight="bold" fill="#3fb950">✅ 有 Padding</text>
  <text x="540" y="95" text-anchor="middle" font-size="12" fill="#8b949e">float tile[32][33]</text>

  <g transform="translate(420, 110)">
    <rect x="0" y="0" width="30" height="30" fill="#3fb950" stroke="#56d364" stroke-width="2"/>
    <text x="15" y="20" text-anchor="middle" font-size="9" fill="#0d1117">B0</text>
    <rect x="30" y="0" width="30" height="30" fill="#3fb950" stroke="#56d364" stroke-width="2"/>
    <text x="45" y="20" text-anchor="middle" font-size="9" fill="#0d1117">B1</text>
    <rect x="60" y="0" width="30" height="30" fill="#3fb950" stroke="#56d364" stroke-width="2"/>
    <text x="75" y="20" text-anchor="middle" font-size="9" fill="#0d1117">B2</text>
    <rect x="90" y="0" width="30" height="30" fill="#3fb950" stroke="#56d364" stroke-width="2"/>
    <text x="105" y="20" text-anchor="middle" font-size="9" fill="#0d1117">...</text>
    <rect x="120" y="0" width="30" height="30" fill="#3fb950" stroke="#56d364" stroke-width="2"/>
    <text x="135" y="20" text-anchor="middle" font-size="9" fill="#0d1117">B31</text>
    <rect x="150" y="0" width="20" height="30" fill="#30363d" stroke="#484f58"/>
    <text x="160" y="20" text-anchor="middle" font-size="8" fill="#8b949e">pad</text>

    <rect x="0" y="30" width="30" height="30" fill="#238636" stroke="#3fb950" stroke-width="2"/>
    <text x="15" y="50" text-anchor="middle" font-size="9" fill="#fff">B1</text>
    <rect x="30" y="30" width="30" height="30" fill="#238636" stroke="#3fb950" stroke-width="2"/>
    <text x="45" y="50" text-anchor="middle" font-size="9" fill="#fff">B2</text>

    <text x="85" y="90" text-anchor="middle" font-size="11" fill="#3fb950">每行多一个 padding 单元</text>
    <text x="85" y="110" text-anchor="middle" font-size="11" fill="#3fb950">同一列的数据错开 bank</text>
  </g>

  <!-- Code -->
  <rect x="60" y="260" width="600" height="120" rx="8" fill="#1f2937" stroke="#30363d" stroke-width="2"/>
  <text x="70" y="290" font-family="monospace" font-size="13" fill="#c9d1d9">// 有 conflict</text>
  <text x="70" y="310" font-family="monospace" font-size="13" fill="#c9d1d9">__shared__ float tile[TILE_DIM][TILE_DIM];</text>
  <text x="70" y="340" font-family="monospace" font-size="13" fill="#c9d1d9">// 无 conflict</text>
  <text x="70" y="360" font-family="monospace" font-size="13" fill="#c9d1d9">__shared__ float tile[TILE_DIM][TILE_DIM + 1];</text>
</svg>'''


def bank_access_patterns() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" viewBox="0 0 720 480">
  <rect width="720" height="480" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Shared Memory 访问模式总结</text>

  <!-- Pattern 1: No conflict -->
  <rect x="60" y="70" width="600" height="100" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="80" y="100" font-size="15" font-weight="bold" fill="#3fb950">✅ 模式 1：每个线程访问不同 bank</text>
  <text x="80" y="125" font-family="monospace" font-size="12" fill="#c9d1d9">tile[threadIdx.x]  // 线程 i 访问 bank i</text>
  <text x="80" y="150" font-size="12" fill="#8b949e">结果：1 个 cycle 完成，最快</text>

  <!-- Pattern 2: Broadcast -->
  <rect x="60" y="190" width="600" height="100" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="80" y="220" font-size="15" font-weight="bold" fill="#58a6ff">✅ 模式 2：所有线程访问同一地址（Broadcast）</text>
  <text x="80" y="245" font-family="monospace" font-size="12" fill="#c9d1d9">tile[0]  // 所有线程读同一个地址</text>
  <text x="80" y="270" font-size="12" fill="#8b949e">结果：1 个 cycle 完成，有专门广播机制</text>

  <!-- Pattern 3: 2-way conflict -->
  <rect x="60" y="310" width="600" height="70" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="80" y="340" font-size="15" font-weight="bold" fill="#e3b341">⚠️ 模式 3：2-way bank conflict</text>
  <text x="80" y="365" font-family="monospace" font-size="12" fill="#c9d1d9">tile[threadIdx.x % 2]  // 线程分成两组访问两个 bank</text>

  <!-- Pattern 4: 32-way conflict -->
  <rect x="60" y="400" width="600" height="60" rx="8" fill="#f85149" opacity="0.2" stroke="#f85149" stroke-width="2"/>
  <text x="80" y="425" font-size="15" font-weight="bold" fill="#f85149">❌ 模式 4：32-way bank conflict（最坏情况）</text>
  <text x="80" y="450" font-family="monospace" font-size="12" fill="#c9d1d9">tile[threadIdx.x * 32]  // 所有线程访问同一个 bank</text>
</svg>'''


def nsight_tools_comparison() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Nsight Compute vs Nsight Systems</text>

  <!-- Nsight Compute -->
  <rect x="60" y="80" width="280" height="260" rx="12" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="200" y="115" text-anchor="middle" font-size="18" font-weight="bold" fill="#58a6ff">Nsight Compute</text>
  <text x="200" y="140" text-anchor="middle" font-size="13" fill="#c9d1d9">(ncu)</text>

  <text x="80" y="175" font-size="13" fill="#c9d1d9">• Kernel 级分析</text>
  <text x="80" y="200" font-size="13" fill="#c9d1d9">• 详细硬件指标</text>
  <text x="80" y="225" font-size="13" fill="#c9d1d9">• Occupancy / Throughput</text>
  <text x="80" y="250" font-size="13" fill="#c9d1d9">• Memory / Compute bound</text>
  <text x="80" y="275" font-size="13" fill="#c9d1d9">• Roofline 分析</text>
  <text x="80" y="300" font-size="13" fill="#8b949e">适合：优化单个 kernel</text>

  <!-- Nsight Systems -->
  <rect x="380" y="80" width="280" height="260" rx="12" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="520" y="115" text-anchor="middle" font-size="18" font-weight="bold" fill="#3fb950">Nsight Systems</text>
  <text x="520" y="140" text-anchor="middle" font-size="13" fill="#c9d1d9">(nsys)</text>

  <text x="400" y="175" font-size="13" fill="#c9d1d9">• 应用级时间线</text>
  <text x="400" y="200" font-size="13" fill="#c9d1d9">• CPU / GPU 交互</text>
  <text x="400" y="225" font-size="13" fill="#c9d1d9">• Kernel launch overhead</text>
  <text x="400" y="250" font-size="13" fill="#c9d1d9">• 多流并行分析</text>
  <text x="400" y="275" font-size="13" fill="#c9d1d9">• 端到端 latency</text>
  <text x="400" y="300" font-size="13" fill="#8b949e">适合：系统级性能分析</text>

  <!-- Bottom note -->
  <text x="360" y="370" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">先用 nsys 找瓶颈 kernel，再用 ncu 深入分析该 kernel</text>
</svg>'''


def profiling_workflow() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="320" viewBox="0 0 720 320">
  <rect width="720" height="320" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">GPU Profiling 工作流程</text>

  <!-- Steps -->
  <rect x="60" y="90" width="120" height="80" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="120" y="125" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">1. 运行程序</text>
  <text x="120" y="150" text-anchor="middle" font-size="12" fill="#8b949e">确认功能正确</text>

  <line x1="180" y1="130" x2="220" y2="130" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowRight)"/>

  <rect x="220" y="90" width="120" height="80" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="280" y="125" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">2. nsys</text>
  <text x="280" y="150" text-anchor="middle" font-size="12" fill="#8b949e">找耗时 kernel</text>

  <line x1="340" y1="130" x2="380" y2="130" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowRight)"/>

  <rect x="380" y="90" width="120" height="80" rx="8" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
  <text x="440" y="125" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">3. ncu</text>
  <text x="440" y="150" text-anchor="middle" font-size="12" fill="#8b949e">分析瓶颈 kernel</text>

  <line x1="500" y1="130" x2="540" y2="130" stroke="#8b949e" stroke-width="2" marker-end="url(#arrowRight)"/>

  <rect x="540" y="90" width="120" height="80" rx="8" fill="#238636" opacity="0.3" stroke="#3fb950" stroke-width="2"/>
  <text x="600" y="125" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">4. 优化</text>
  <text x="600" y="150" text-anchor="middle" font-size="12" fill="#8b949e">针对性改进</text>

  <!-- Questions -->
  <rect x="60" y="210" width="600" height="80" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="240" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">每次 profiling 要回答的问题</text>
  <text x="360" y="270" text-anchor="middle" font-size="13" fill="#8b949e">哪个 kernel 最耗时？→ 它是 memory-bound 还是 compute-bound？→ 具体瓶颈是什么？→ 如何优化？</text>
</svg>'''


def ncu_metrics_overview() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440">
  <rect width="720" height="440" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">ncu 常用指标分类</text>

  <!-- Occupancy -->
  <rect x="60" y="80" width="190" height="120" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="155" y="110" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">并行度</text>
  <text x="75" y="140" font-size="12" fill="#8b949e">sm__occupancy.avg</text>
  <text x="75" y="160" font-size="12" fill="#8b949e">sm__warps_active.avg</text>
  <text x="75" y="180" font-size="12" fill="#8b949e">launch__registers_per_thread</text>

  <!-- Memory -->
  <rect x="265" y="80" width="190" height="120" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="360" y="110" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">内存</text>
  <text x="280" y="140" font-size="12" fill="#8b949e">dram__throughput.avg</text>
  <text x="280" y="160" font-size="12" fill="#8b949e">l1tex__t_bytes_pipe_lsu...</text>
  <text x="280" y="180" font-size="12" fill="#8b949e">l1tex__data_bank_conflicts</text>

  <!-- Compute -->
  <rect x="470" y="80" width="190" height="120" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="565" y="110" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">计算</text>
  <text x="485" y="140" font-size="12" fill="#8b949e">sm__throughput.avg</text>
  <text x="485" y="160" font-size="12" fill="#8b949e">sm__cycles_elapsed.avg</text>
  <text x="485" y="180" font-size="12" fill="#8b949e">smsp__sass_thread_inst_executed</text>

  <!-- Latency -->
  <rect x="60" y="230" width="190" height="120" rx="8" fill="#8957e5" opacity="0.2" stroke="#a371f7" stroke-width="2"/>
  <text x="155" y="260" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">延迟</text>
  <text x="75" y="290" font-size="12" fill="#8b949e">sm__cycles_elapsed.avg</text>
  <text x="75" y="310" font-size="12" fill="#8b949e">launch__duration</text>
  <text x="75" y="330" font-size="12" fill="#8b949e">gpu__time_duration.avg</text>

  <!-- Bottleneck -->
  <rect x="265" y="230" width="395" height="120" rx="8" fill="#f85149" opacity="0.2" stroke="#f85149" stroke-width="2"/>
  <text x="462" y="260" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">瓶颈判断</text>
  <text x="285" y="290" font-size="12" fill="#8b949e">Memory Throughput 高 + Compute Throughput 低 → Memory-bound</text>
  <text x="285" y="315" font-size="12" fill="#8b949e">Compute Throughput 高 + Memory Throughput 低 → Compute-bound</text>
  <text x="285" y="340" font-size="12" fill="#8b949e">两者都低 → Latency / Occupancy 问题</text>
</svg>'''


def memory_compute_bound() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">
  <rect width="720" height="360" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">如何判断 Kernel 瓶颈类型</text>

  <!-- Memory bound -->
  <rect x="60" y="80" width="280" height="220" rx="12" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="200" y="115" text-anchor="middle" font-size="17" font-weight="bold" fill="#e3b341">Memory-Bound</text>

  <text x="80" y="155" font-size="13" fill="#c9d1d9">特征：</text>
  <text x="80" y="180" font-size="12" fill="#8b949e">• dram__throughput 接近峰值</text>
  <text x="80" y="205" font-size="12" fill="#8b949e">• sm__throughput 较低</text>
  <text x="80" y="230" font-size="12" fill="#8b949e">• Arithmetic Intensity 低</text>

  <text x="80" y="270" font-size="13" fill="#c9d1d9">优化方向：</text>
  <text x="80" y="290" font-size="12" fill="#8b949e">• 合并内存访问</text>
  <text x="80" y="310" font-size="12" fill="#8b949e">• 用 shared memory / cache</text>
  <text x="80" y="330" font-size="12" fill="#8b949e">• 减少数据读写</text>

  <!-- Compute bound -->
  <rect x="380" y="80" width="280" height="220" rx="12" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="520" y="115" text-anchor="middle" font-size="17" font-weight="bold" fill="#3fb950">Compute-Bound</text>

  <text x="400" y="155" font-size="13" fill="#c9d1d9">特征：</text>
  <text x="400" y="180" font-size="12" fill="#8b949e">• sm__throughput 接近峰值</text>
  <text x="400" y="205" font-size="12" fill="#8b949e">• dram__throughput 较低</text>
  <text x="400" y="230" font-size="12" fill="#8b949e">• Arithmetic Intensity 高</text>

  <text x="400" y="270" font-size="13" fill="#c9d1d9">优化方向：</text>
  <text x="400" y="290" font-size="12" fill="#8b949e">• 使用 Tensor Core</text>
  <text x="400" y="310" font-size="12" fill="#8b949e">• 指令级优化</text>
  <text x="400" y="330" font-size="12" fill="#8b949e">• 提高 occupancy</text>
</svg>'''


def week1_knowledge_map() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="520" viewBox="0 0 720 520">
  <rect width="720" height="520" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Week 1 知识地图</text>

  <!-- Center -->
  <circle cx="360" cy="160" r="50" fill="#1f6feb" stroke="#58a6ff" stroke-width="3"/>
  <text x="360" y="155" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">GPU 性能</text>
  <text x="360" y="175" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">= Memory + 并行度</text>

  <!-- Branches -->
  <line x1="320" y1="200" x2="180" y2="280" stroke="#58a6ff" stroke-width="2"/>
  <line x1="400" y1="200" x2="540" y2="280" stroke="#58a6ff" stroke-width="2"/>
  <line x1="360" y1="210" x2="360" y2="300" stroke="#58a6ff" stroke-width="2"/>

  <!-- Day 1-2 -->
  <rect x="60" y="280" width="180" height="180" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="150" y="310" text-anchor="middle" font-size="14" font-weight="bold" fill="#58a6ff">执行模型</text>
  <text x="75" y="340" font-size="12" fill="#c9d1d9">• SM / Warp / SIMT</text>
  <text x="75" y="365" font-size="12" fill="#c9d1d9">• Grid / Block / Thread</text>
  <text x="75" y="390" font-size="12" fill="#c9d1d9">• Occupancy</text>
  <text x="75" y="415" font-size="12" fill="#8b949e">Day 1-2</text>

  <!-- Day 3 -->
  <rect x="270" y="300" width="180" height="160" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="360" y="330" text-anchor="middle" font-size="14" font-weight="bold" fill="#3fb950">硬件认知</text>
  <text x="285" y="360" font-size="12" fill="#c9d1d9">• deviceQuery</text>
  <text x="285" y="385" font-size="12" fill="#c9d1d9">• GPU 峰值算力</text>
  <text x="285" y="410" font-size="12" fill="#c9d1d9">• 显存带宽</text>
  <text x="285" y="435" font-size="12" fill="#8b949e">Day 3</text>

  <!-- Day 4-6 -->
  <rect x="480" y="280" width="180" height="180" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="570" y="310" text-anchor="middle" font-size="14" font-weight="bold" fill="#e3b341">内存优化</text>
  <text x="495" y="340" font-size="12" fill="#c9d1d9">• Coalescing</text>
  <text x="495" y="365" font-size="12" fill="#c9d1d9">• Shared Memory Tiling</text>
  <text x="495" y="390" font-size="12" fill="#c9d1d9">• Bank Conflict</text>
  <text x="495" y="415" font-size="12" fill="#c9d1d9">• Nsight Profiling</text>
  <text x="495" y="440" font-size="12" fill="#8b949e">Day 4-6</text>

  <!-- Bottom note -->
  <text x="360" y="500" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">Week 1 核心：建立 GPU 性能直觉</text>
</svg>'''


def optimization_decision_tree() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" viewBox="0 0 720 480">
  <rect width="720" height="480" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">GPU 性能优化决策树</text>

  <!-- Start -->
  <rect x="280" y="70" width="160" height="50" rx="8" fill="#1f6feb" stroke="#58a6ff" stroke-width="2"/>
  <text x="360" y="100" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Profiling 找到瓶颈</text>

  <!-- Question 1 -->
  <rect x="280" y="150" width="160" height="50" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="175" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">Occupancy 低？</text>

  <line x1="360" y1="120" x2="360" y2="150" stroke="#8b949e" stroke-width="2"/>

  <!-- Yes branch -->
  <line x1="280" y1="175" x2="160" y2="230" stroke="#8b949e" stroke-width="2"/>
  <text x="200" y="195" font-size="11" fill="#3fb950">是</text>

  <rect x="60" y="230" width="180" height="80" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="150" y="255" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">优化 Occupancy</text>
  <text x="70" y="280" font-size="11" fill="#8b949e">• 减少寄存器使用</text>
  <text x="70" y="300" font-size="11" fill="#8b949e">• 调整 block 大小</text>

  <!-- No branch -->
  <line x1="440" y1="175" x2="560" y2="230" stroke="#8b949e" stroke-width="2"/>
  <text x="500" y="195" font-size="11" fill="#f85149">否</text>

  <rect x="480" y="230" width="180" height="80" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="570" y="255" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">判断瓶颈类型</text>
  <text x="490" y="280" font-size="11" fill="#8b949e">• Memory-bound?</text>
  <text x="490" y="300" font-size="11" fill="#8b949e">• Compute-bound?</text>

  <!-- Memory bound -->
  <line x1="520" y1="310" x2="440" y2="360" stroke="#8b949e" stroke-width="2"/>
  <text x="470" y="335" font-size="11" fill="#e3b341">Memory</text>

  <rect x="320" y="360" width="180" height="90" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="410" y="385" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">内存优化</text>
  <text x="330" y="410" font-size="11" fill="#8b949e">• Coalesced access</text>
  <text x="330" y="430" font-size="11" fill="#8b949e">• Shared memory tiling</text>
  <text x="330" y="450" font-size="11" fill="#8b949e">• 减少数据读写</text>

  <!-- Compute bound -->
  <line x1="620" y1="310" x2="620" y2="360" stroke="#8b949e" stroke-width="2"/>
  <text x="625" y="340" font-size="11" fill="#3fb950">Compute</text>

  <rect x="530" y="360" width="180" height="90" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="620" y="385" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">计算优化</text>
  <text x="540" y="410" font-size="11" fill="#8b949e">• Tensor Core</text>
  <text x="540" y="430" font-size="11" fill="#8b949e">• 指令级优化</text>
  <text x="540" y="450" font-size="11" fill="#8b949e">• 提高指令吞吐</text>
</svg>'''


def week1_interview_prep() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400">
  <rect width="720" height="400" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Week 1 面试准备框架</text>

  <!-- Three pillars -->
  <rect x="60" y="90" width="190" height="220" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="155" y="125" text-anchor="middle" font-size="15" font-weight="bold" fill="#58a6ff">概念</text>
  <text x="80" y="160" font-size="12" fill="#c9d1d9">• SM / Warp / SIMT</text>
  <text x="80" y="185" font-size="12" fill="#c9d1d9">• Occupancy</text>
  <text x="80" y="210" font-size="12" fill="#c9d1d9">• Coalescing</text>
  <text x="80" y="235" font-size="12" fill="#c9d1d9">• Bank Conflict</text>
  <text x="80" y="260" font-size="12" fill="#c9d1d9">• Roofline</text>

  <rect x="265" y="90" width="190" height="220" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
  <text x="360" y="125" text-anchor="middle" font-size="15" font-weight="bold" fill="#3fb950">代码</text>
  <text x="285" y="160" font-size="12" fill="#c9d1d9">• 写第一个 CUDA</text>
  <text x="285" y="185" font-size="12" fill="#c9d1d9">• 查寄存器用量</text>
  <text x="285" y="210" font-size="12" fill="#c9d1d9">• 矩阵转置优化</text>
  <text x="285" y="235" font-size="12" fill="#c9d1d9">• Bank conflict 实验</text>
  <text x="285" y="260" font-size="12" fill="#c9d1d9">• ncu / nsys 使用</text>

  <rect x="470" y="90" width="190" height="220" rx="8" fill="#d29922" opacity="0.2" stroke="#e3b341" stroke-width="2"/>
  <text x="565" y="125" text-anchor="middle" font-size="15" font-weight="bold" fill="#e3b341">表达</text>
  <text x="490" y="160" font-size="12" fill="#c9d1d9">• 用自己的话解释</text>
  <text x="490" y="185" font-size="12" fill="#c9d1d9">• 画图说明</text>
  <text x="490" y="210" font-size="12" fill="#c9d1d9">• 举实际例子</text>
  <text x="490" y="235" font-size="12" fill="#c9d1d9">• 说出优化思路</text>
  <text x="490" y="260" font-size="12" fill="#c9d1d9">• 承认不懂的地方</text>

  <!-- Bottom -->
  <rect x="60" y="340" width="600" height="40" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="365" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">面试不是背诵，而是展示你理解概念、能动手、会思考</text>
</svg>'''

def thread_id_calculation() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="540" viewBox="0 0 720 540">
  <rect width="720" height="540" fill="#0d1117"/>
  <text x="360" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">CUDA 线程 ID 计算</text>

  <!-- 1D grid + 1D block -->
  <text x="60" y="72" font-size="16" font-weight="bold" fill="#58a6ff">1D grid + 1D block</text>

  <!-- Grid label -->
  <text x="60" y="100" font-size="13" fill="#8b949e">Grid (gridDim.x = 3)</text>

  <!-- Blocks -->
  <rect x="60" y="115" width="180" height="70" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="150" y="138" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Block 0</text>
  <text x="150" y="158" text-anchor="middle" font-size="11" fill="#8b949e">blockIdx.x = 0</text>

  <rect x="250" y="115" width="180" height="70" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="340" y="138" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Block 1</text>
  <text x="340" y="158" text-anchor="middle" font-size="11" fill="#8b949e">blockIdx.x = 1</text>

  <rect x="440" y="115" width="180" height="70" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="530" y="138" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Block 2</text>
  <text x="530" y="158" text-anchor="middle" font-size="11" fill="#8b949e">blockIdx.x = 2</text>

  <!-- Threads in Block 1 (zoom) -->
  <text x="250" y="215" font-size="13" fill="#8b949e">Block 1 内部：blockDim.x = 4</text>
  <circle cx="265" cy="245" r="14" fill="#3fb950"/>
  <text x="265" y="250" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T0</text>
  <circle cx="305" cy="245" r="14" fill="#3fb950"/>
  <text x="305" y="250" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T1</text>
  <circle cx="345" cy="245" r="14" fill="#3fb950"/>
  <text x="345" y="250" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T2</text>
  <circle cx="385" cy="245" r="14" fill="#3fb950"/>
  <text x="385" y="250" text-anchor="middle" font-size="9" fill="#0d1117" font-weight="bold">T3</text>

  <!-- Thread labels -->
  <text x="265" y="275" text-anchor="middle" font-size="10" fill="#8b949e">threadIdx.x=0</text>
  <text x="385" y="275" text-anchor="middle" font-size="10" fill="#8b949e">threadIdx.x=3</text>

  <!-- Formula -->
  <rect x="60" y="305" width="600" height="55" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="328" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">global_tid = blockIdx.x × blockDim.x + threadIdx.x</text>
  <text x="360" y="350" text-anchor="middle" font-size="13" fill="#e3b341">例：Block 1 的 T3 → 1 × 4 + 3 = 7</text>

  <!-- 2D grid + 2D block -->
  <text x="60" y="395" font-size="16" font-weight="bold" fill="#58a6ff">2D grid + 2D block（图像处理常用）</text>

  <!-- 2D grid -->
  <rect x="60" y="415" width="120" height="80" rx="6" fill="#1f6feb" opacity="0.15" stroke="#58a6ff" stroke-width="1.5"/>
  <rect x="180" y="415" width="120" height="80" rx="6" fill="#1f6feb" opacity="0.15" stroke="#58a6ff" stroke-width="1.5"/>
  <rect x="60" y="495" width="120" height="20" fill="none"/>
  <text x="130" y="440" text-anchor="middle" font-size="12" fill="#c9d1d9">Block (0,0)</text>
  <text x="250" y="440" text-anchor="middle" font-size="12" fill="#c9d1d9">Block (1,0)</text>

  <!-- Highlight a block -->
  <rect x="180" y="415" width="120" height="80" rx="6" fill="none" stroke="#e3b341" stroke-width="2"/>
  <text x="240" y="520" text-anchor="middle" font-size="11" fill="#e3b341">blockIdx = (1, 0)</text>

  <!-- Formula -->
  <rect x="330" y="415" width="330" height="90" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
  <text x="360" y="440" font-size="13" fill="#c9d1d9" font-weight="bold">row = blockIdx.y × blockDim.y + threadIdx.y</text>
  <text x="360" y="465" font-size="13" fill="#c9d1d9" font-weight="bold">col = blockIdx.x × blockDim.x + threadIdx.x</text>
  <text x="360" y="490" font-size="12" fill="#8b949e">global_tid = row × (gridDim.x × blockDim.x) + col</text>
</svg>'''


def main() -> None:
    diagrams = {
        "gpu_memory_hierarchy.svg": gpu_memory_hierarchy(),
        "thread_id_calculation.svg": thread_id_calculation(),
        "sm_architecture.svg": sm_architecture(),
        "coalesced_access.svg": coalesced_access(),
        "bank_conflict.svg": bank_conflict(),
        "roofline_model.svg": roofline_model(),
        "week1_roadmap.svg": week1_roadmap(),
        "grid_block_thread.svg": grid_block_thread(),
        "warp_divergence.svg": warp_divergence(),
        "simt_vs_simd.svg": simt_vs_simd(),
        "occupancy_concept.svg": occupancy_concept(),
        "resource_constraints.svg": resource_constraints(),
        "register_spilling.svg": register_spilling(),
        "occupancy_curve.svg": occupancy_curve(),
        "device_query_output.svg": device_query_output(),
        "occupancy_calculator_workflow.svg": occupancy_calculator_workflow(),
        "cuda_guide_ch5.svg": cuda_guide_ch5(),
        "stride_access.svg": stride_access(),
        "shared_memory_tiling.svg": shared_memory_tiling(),
        "matrix_transpose.svg": matrix_transpose(),
        "shared_memory_bank_structure.svg": shared_memory_bank_structure(),
        "padding_solution.svg": padding_solution(),
        "bank_access_patterns.svg": bank_access_patterns(),
        "nsight_tools_comparison.svg": nsight_tools_comparison(),
        "profiling_workflow.svg": profiling_workflow(),
        "ncu_metrics_overview.svg": ncu_metrics_overview(),
        "memory_compute_bound.svg": memory_compute_bound(),
        "week1_knowledge_map.svg": week1_knowledge_map(),
        "optimization_decision_tree.svg": optimization_decision_tree(),
        "week1_interview_prep.svg": week1_interview_prep(),
    }

    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
