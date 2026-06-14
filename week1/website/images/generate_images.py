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

def main() -> None:
    diagrams = {
        "gpu_memory_hierarchy.svg": gpu_memory_hierarchy(),
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
    }

    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
