#!/usr/bin/env python3
"""Generate SVG diagrams for Week 2 Day 6 (Integrated GEMM)."""

from pathlib import Path


def save_svg(filename: str, content: str) -> None:
    path = Path(__file__).parent / filename
    path.write_text(content, encoding="utf-8")
    print(f"Generated: {path}")


def gemm_optimization_layers() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="560" viewBox="0 0 760 560">
  <rect width="760" height="560" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">GEMM 优化层次：从 1% 到 70%+</text>

  <!-- Level 0: Naive -->
  <g transform="translate(80, 60)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#f85149" opacity="0.15" stroke="#f85149"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">Naive GEMM</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">每线程算 1 元素，直接访问 Global Memory</text>
    <text x="560" y="27" font-size="14" fill="#f85149" font-weight="bold">~1%</text>
  </g>

  <!-- Level 1: Shared Memory Tiling -->
  <g transform="translate(80, 112)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#d29922" opacity="0.15" stroke="#d29922"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">+ Shared Memory Tiling</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">A/B tile 预取到 Shared Memory，K 维度复用</text>
    <text x="560" y="27" font-size="14" fill="#d29922" font-weight="bold">~15%</text>
  </g>

  <!-- Level 2: Register Blocking -->
  <g transform="translate(80, 164)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#a371f7" opacity="0.15" stroke="#a371f7"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">+ Register Blocking (Day 2)</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">TM×TN thread tile，累加器驻留寄存器</text>
    <text x="560" y="27" font-size="14" fill="#a371f7" font-weight="bold">~45%</text>
  </g>

  <!-- Level 3: float4 -->
  <g transform="translate(80, 216)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#58a6ff" opacity="0.15" stroke="#58a6ff"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">+ float4 向量化加载 (Day 6)</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">128-bit load，提升带宽利用率</text>
    <text x="560" y="27" font-size="14" fill="#58a6ff" font-weight="bold">~55%</text>
  </g>

  <!-- Level 4: Warp Shuffle -->
  <g transform="translate(80, 268)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#3fb950" opacity="0.15" stroke="#3fb950"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">+ Warp Shuffle 写回优化 (Day 6)</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">Warp 内协作，减少非合并访问</text>
    <text x="560" y="27" font-size="14" fill="#3fb950" font-weight="bold">~60%</text>
  </g>

  <!-- Level 5: Double Buffer -->
  <g transform="translate(80, 320)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#238636" opacity="0.15" stroke="#238636"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">+ Double Buffering</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">软件流水线，计算掩盖传输延迟</text>
    <text x="560" y="27" font-size="14" fill="#3fb950" font-weight="bold">~70%</text>
  </g>

  <!-- Level 6: Auto-tuning -->
  <g transform="translate(80, 372)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-dasharray="4"/>
    <text x="15" y="27" font-size="13" fill="#8b949e">+ 参数 Auto-tuning (进阶)</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">针对不同尺寸选择最优 BM/BN/BK/TM/TN</text>
    <text x="560" y="27" font-size="14" fill="#58a6ff">~80%+</text>
  </g>

  <!-- Level 7: Tensor Core -->
  <g transform="translate(80, 424)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-dasharray="4"/>
    <text x="15" y="27" font-size="13" fill="#8b949e">+ Tensor Core / CUTLASS (进阶)</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">WMMA 指令，矩阵乘加硬件加速</text>
    <text x="560" y="27" font-size="14" fill="#58a6ff">~90%+</text>
  </g>

  <!-- cuBLAS -->
  <g transform="translate(80, 476)">
    <rect x="0" y="0" width="600" height="42" rx="6" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="15" y="27" font-size="13" fill="#c9d1d9">cuBLAS（NVIDIA 官方优化）</text>
    <text x="200" y="27" font-size="12" fill="#8b949e">PTX 内联 + 完整流水线 + Tensor Core + auto-tuning</text>
    <text x="560" y="27" font-size="14" fill="#c9d1d9" font-weight="bold">100%</text>
  </g>

  <!-- Day 2 vs Day 6 marker -->
  <line x1="690" y1="185" x2="730" y2="185" stroke="#a371f7" stroke-width="2"/>
  <text x="735" y="180" font-size="10" fill="#a371f7" font-weight="bold" transform="rotate(90, 735, 180)">Day 2</text>

  <line x1="690" y1="341" x2="730" y2="341" stroke="#3fb950" stroke-width="2"/>
  <text x="735" y="336" font-size="10" fill="#3fb950" font-weight="bold" transform="rotate(90, 735, 336)">Day 6</text>

  <text x="380" y="545" text-anchor="middle" font-size="13" fill="#c9d1d9">Day 6 目标：从 ~45% 跨越到 70%+</text>
</svg>'''


def float4_vectorized_load() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" viewBox="0 0 760 440">
  <rect width="760" height="440" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">float4 向量化加载 vs 逐元素加载</text>

  <!-- 逐元素加载 -->
  <g transform="translate(40, 65)">
    <text x="0" y="0" font-size="15" fill="#f85149" font-weight="bold">逐元素加载：4 条 32-bit load 指令</text>
    <rect x="0" y="15" width="320" height="140" rx="8" fill="#f85149" opacity="0.08" stroke="#f85149" stroke-width="2"/>

    <text x="15" y="40" font-size="12" fill="#c9d1d9" font-family="monospace">float a0 = ptr[0];  // load 32-bit</text>
    <text x="15" y="60" font-size="12" fill="#c9d1d9" font-family="monospace">float a1 = ptr[1];  // load 32-bit</text>
    <text x="15" y="80" font-size="12" fill="#c9d1d9" font-family="monospace">float a2 = ptr[2];  // load 32-bit</text>
    <text x="15" y="100" font-size="12" fill="#c9d1d9" font-family="monospace">float a3 = ptr[3];  // load 32-bit</text>

    <rect x="15" y="115" width="120" height="28" rx="4" fill="#f85149" opacity="0.2" stroke="#f85149"/>
    <text x="75" y="133" text-anchor="middle" font-size="12" fill="#f85149" font-weight="bold">4 条指令</text>
  </g>

  <!-- float4 加载 -->
  <g transform="translate(400, 65)">
    <text x="0" y="0" font-size="15" fill="#3fb950" font-weight="bold">float4 加载：1 条 128-bit load 指令</text>
    <rect x="0" y="15" width="320" height="140" rx="8" fill="#3fb950" opacity="0.08" stroke="#3fb950" stroke-width="2"/>

    <text x="15" y="40" font-size="12" fill="#c9d1d9" font-family="monospace">float4 val =</text>
    <text x="15" y="60" font-size="12" fill="#c9d1d9" font-family="monospace">  reinterpret_cast&lt;const float4*&gt;(ptr)[0];</text>
    <text x="15" y="85" font-size="12" fill="#8b949e">// val.x = ptr[0]</text>
    <text x="15" y="105" font-size="12" fill="#8b949e">// val.y = ptr[1]</text>
    <text x="15" y="125" font-size="12" fill="#8b949e">// val.z = ptr[2], val.w = ptr[3]</text>

    <rect x="180" y="115" width="120" height="28" rx="4" fill="#3fb950" opacity="0.2" stroke="#3fb950"/>
    <text x="240" y="133" text-anchor="middle" font-size="12" fill="#3fb950" font-weight="bold">1 条指令</text>
  </g>

  <!-- Conditions -->
  <g transform="translate(40, 235)">
    <rect x="0" y="0" width="680" height="115" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="340" y="25" text-anchor="middle" font-size="15" fill="#58a6ff" font-weight="bold">使用条件</text>

    <text x="20" y="55" font-size="13" fill="#3fb950">✅ 地址 16 字节对齐</text>
    <text x="200" y="55" font-size="11" fill="#8b949e">cudaMalloc 分配的内存天然对齐</text>

    <text x="20" y="80" font-size="13" fill="#3fb950">✅ Coalesced 访问模式</text>
    <text x="200" y="80" font-size="11" fill="#8b949e">连续线程访问连续地址，合并为最少 cache line</text>

    <text x="20" y="105" font-size="13" fill="#3fb950">✅ 数据布局支持</text>
    <text x="200" y="105" font-size="11" fill="#8b949e">行优先矩阵的连续行元素天然连续</text>
  </g>

  <!-- Performance -->
  <rect x="160" y="370" width="440" height="40" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="380" y="395" text-anchor="middle" font-size="14" fill="#58a6ff" font-weight="bold">收益：Global Memory 带宽利用率提升 10-15%</text>
</svg>'''


def parameter_tuning_table() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" viewBox="0 0 760 440">
  <rect width="760" height="440" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">参数精调扫描表</text>

  <!-- Table header -->
  <g transform="translate(40, 60)">
    <rect x="0" y="0" width="680" height="35" fill="#30363d" stroke="#484f58"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">TM×TN</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">1024 矩阵</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">2048 矩阵</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">4096 矩阵</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">Register</text>
  </g>

  <!-- Row 1: 8x8 baseline -->
  <g transform="translate(40, 95)">
    <rect x="0" y="0" width="680" height="35" fill="#a371f7" opacity="0.1" stroke="#30363d"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#a371f7" font-weight="bold">8×8</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#8b949e">基准</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#8b949e">基准</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#8b949e">基准</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#3fb950">~88</text>
  </g>

  <!-- Row 2: 8x16 -->
  <g transform="translate(40, 130)">
    <rect x="0" y="0" width="680" height="35" fill="#161b22" stroke="#30363d"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9">8×16</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+5%</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+8%</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+10%</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#d29922">~152</text>
  </g>

  <!-- Row 3: 16x8 -->
  <g transform="translate(40, 165)">
    <rect x="0" y="0" width="680" height="35" fill="#161b22" stroke="#30363d"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9">16×8</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+3%</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+5%</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+8%</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#d29922">~152</text>
  </g>

  <!-- Row 4: 16x16 DANGER -->
  <g transform="translate(40, 200)">
    <rect x="0" y="0" width="680" height="35" fill="#f85149" opacity="0.1" stroke="#f85149"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#f85149" font-weight="bold">16×16</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#f85149">SPILL!</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#f85149">SPILL!</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#f85149">SPILL!</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#f85149" font-weight="bold">~256</text>
  </g>

  <!-- BK scan -->
  <g transform="translate(40, 255)">
    <rect x="0" y="0" width="680" height="35" fill="#30363d" stroke="#484f58"/>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9" font-weight="bold">BK 扫描（固定 TM=TN=8）</text>
  </g>

  <g transform="translate(40, 290)">
    <rect x="0" y="0" width="680" height="35" fill="#161b22" stroke="#30363d"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9">BK=4</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#f85149">-2%</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#8b949e">+3%</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+5%</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#8b949e">smem↓</text>
  </g>

  <g transform="translate(40, 325)">
    <rect x="0" y="0" width="680" height="35" fill="#a371f7" opacity="0.1" stroke="#30363d"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#a371f7" font-weight="bold">BK=8</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#8b949e">基准</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#8b949e">基准</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#8b949e">基准</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#3fb950">~88</text>
  </g>

  <g transform="translate(40, 360)">
    <rect x="0" y="0" width="680" height="35" fill="#161b22" stroke="#30363d"/>
    <text x="80" y="23" text-anchor="middle" font-size="12" fill="#c9d1d9">BK=16</text>
    <text x="200" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+1%</text>
    <text x="340" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+2%</text>
    <text x="480" y="23" text-anchor="middle" font-size="12" fill="#3fb950">+3%</text>
    <text x="620" y="23" text-anchor="middle" font-size="12" fill="#d29922">smem↑</text>
  </g>

  <text x="380" y="425" text-anchor="middle" font-size="12" fill="#8b949e">精调步骤：先扫 TM×TN → 再扫 BK → 最后扫 BM/BN</text>
</svg>'''


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
