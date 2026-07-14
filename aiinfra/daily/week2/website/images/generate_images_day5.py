#!/usr/bin/env python3
"""Generate SVG diagrams for Week 2 Day 5 (FlashAttention)."""

from pathlib import Path


def save_svg(filename: str, content: str) -> None:
    path = Path(__file__).parent / filename
    path.write_text(content, encoding="utf-8")
    print(f"Generated: {path}")


def flash_attention_tiling() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">
  <rect width="760" height="460" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">FlashAttention 分块策略（Tiling）</text>

  <!-- Output O matrix -->
  <g transform="translate(80, 65)">
    <text x="0" y="-5" font-size="14" fill="#58a6ff" font-weight="bold">Attention Output O (N×d)</text>
    <rect x="0" y="0" width="560" height="280" fill="none" stroke="#58a6ff" stroke-width="2" rx="4"/>

    <!-- Q tiles -->
    <rect x="10" y="10" width="170" height="50" rx="4" fill="#1f6feb" opacity="0.3" stroke="#58a6ff" stroke-width="2"/>
    <text x="95" y="40" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Q Tile (Br×d)</text>
    <text x="95" y="55" text-anchor="middle" font-size="10" fill="#8b949e">驻留 SRAM</text>

    <rect x="190" y="10" width="170" height="50" rx="4" fill="#1f6feb" opacity="0.15" stroke="#58a6ff" stroke-width="1" stroke-dasharray="3"/>
    <text x="275" y="40" text-anchor="middle" font-size="12" fill="#8b949e">Q Tile</text>

    <rect x="370" y="10" width="170" height="50" rx="4" fill="#1f6feb" opacity="0.15" stroke="#58a6ff" stroke-width="1" stroke-dasharray="3"/>
    <text x="455" y="40" text-anchor="middle" font-size="12" fill="#8b949e">Q Tile ...</text>

    <!-- Inner loop: KV tiles -->
    <text x="10" y="90" font-size="13" fill="#d29922" font-weight="bold">内循环：遍历 K/V tile</text>

    <rect x="10" y="100" width="100" height="40" rx="4" fill="#d29922" opacity="0.3" stroke="#d29922" stroke-width="2"/>
    <text x="60" y="125" text-anchor="middle" font-size="12" fill="#c9d1d9">KV Tile 0</text>

    <line x1="110" y1="120" x2="130" y2="120" stroke="#d29922" stroke-width="2" marker-end="url(#arrKV)"/>

    <rect x="135" y="100" width="100" height="40" rx="4" fill="#d29922" opacity="0.2" stroke="#d29922"/>
    <text x="185" y="125" text-anchor="middle" font-size="12" fill="#c9d1d9">KV Tile 1</text>

    <line x1="235" y1="120" x2="255" y2="120" stroke="#d29922" stroke-width="2" marker-end="url(#arrKV)"/>

    <rect x="260" y="100" width="100" height="40" rx="4" fill="#d29922" opacity="0.15" stroke="#d29922"/>
    <text x="310" y="125" text-anchor="middle" font-size="12" fill="#c9d1d9">KV Tile 2</text>

    <text x="380" y="125" font-size="14" fill="#8b949e">...</text>

    <!-- Compute flow -->
    <rect x="10" y="165" width="540" height="100" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="280" y="190" text-anchor="middle" font-size="13" fill="#c9d1d9">每步计算：</text>
    <text x="280" y="212" text-anchor="middle" font-size="12" fill="#a371f7" font-family="monospace">S_tile = Q_tile × KV_tile^T  (Br×Bc)</text>
    <text x="280" y="232" text-anchor="middle" font-size="12" fill="#3fb950" font-family="monospace">→ Online Softmax 更新 (m, l, o)</text>
    <text x="280" y="252" text-anchor="middle" font-size="12" fill="#58a6ff" font-family="monospace">→ 累加输出 acc[d] += p_norm × V_tile[d]</text>
  </g>

  <!-- HBM comparison -->
  <g transform="translate(80, 365)">
    <rect x="0" y="0" width="260" height="80" rx="8" fill="#f85149" opacity="0.1" stroke="#f85149" stroke-width="2"/>
    <text x="130" y="25" text-anchor="middle" font-size="14" fill="#f85149" font-weight="bold">标准 Attention</text>
    <text x="130" y="48" text-anchor="middle" font-size="13" fill="#c9d1d9">HBM 访问: O(N²)</text>
    <text x="130" y="68" text-anchor="middle" font-size="11" fill="#8b949e">需存储 S, P 两个 N×N 矩阵</text>

    <rect x="290" y="0" width="260" height="80" rx="8" fill="#3fb950" opacity="0.1" stroke="#3fb950" stroke-width="2"/>
    <text x="420" y="25" text-anchor="middle" font-size="14" fill="#3fb950" font-weight="bold">FlashAttention</text>
    <text x="420" y="48" text-anchor="middle" font-size="13" fill="#c9d1d9">HBM 访问: O(Nd)</text>
    <text x="420" y="68" text-anchor="middle" font-size="11" fill="#8b949e">中间计算在 SRAM 完成</text>
  </g>
</svg>'''


def online_softmax_formula() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500" viewBox="0 0 760 500">
  <rect width="760" height="500" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Online Softmax 三个更新公式</text>

  <!-- State variables -->
  <g transform="translate(40, 60)">
    <rect x="0" y="0" width="680" height="60" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="20" y="25" font-size="13" fill="#58a6ff" font-weight="bold">Running 状态</text>
    <text x="20" y="45" font-size="12" fill="#c9d1d9">m = running max | l = running sum | o = running output（初始：m=-∞, l=0, o=0）</text>
  </g>

  <!-- Formula 1 -->
  <g transform="translate(40, 140)">
    <rect x="0" y="0" width="680" height="85" rx="8" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="2"/>
    <text x="20" y="25" font-size="14" fill="#58a6ff" font-weight="bold">公式 1 — Max 更新</text>
    <rect x="20" y="35" width="640" height="40" rx="4" fill="#0d1117" stroke="#30363d"/>
    <text x="340" y="60" text-anchor="middle" font-size="16" fill="#c9d1d9" font-family="monospace">m_new = max(m, max(x_j))</text>
  </g>

  <!-- Formula 2 -->
  <g transform="translate(40, 240)">
    <rect x="0" y="0" width="680" height="105" rx="8" fill="#d29922" opacity="0.1" stroke="#d29922" stroke-width="2"/>
    <text x="20" y="25" font-size="14" fill="#d29922" font-weight="bold">公式 2 — Sum 更新</text>
    <rect x="20" y="35" width="640" height="40" rx="4" fill="#0d1117" stroke="#30363d"/>
    <text x="340" y="60" text-anchor="middle" font-size="14" fill="#c9d1d9" font-family="monospace">l_new = l × exp(m - m_new) + Σ exp(x_j - m_new)</text>
    <text x="20" y="92" font-size="11" fill="#8b949e">l × exp(m - m_new)：旧 sum 从旧参考点 m 缩放到新参考点 m_new</text>
  </g>

  <!-- Formula 3 -->
  <g transform="translate(40, 360)">
    <rect x="0" y="0" width="680" height="120" rx="8" fill="#3fb950" opacity="0.1" stroke="#3fb950" stroke-width="2"/>
    <text x="20" y="25" font-size="14" fill="#3fb950" font-weight="bold">公式 3 — Output 更新</text>
    <rect x="20" y="35" width="640" height="50" rx="4" fill="#0d1117" stroke="#30363d"/>
    <text x="340" y="58" text-anchor="middle" font-size="12" fill="#c9d1d9" font-family="monospace">o_new = o × (l × exp(m - m_new) / l_new)</text>
    <text x="340" y="76" text-anchor="middle" font-size="12" fill="#c9d1d9" font-family="monospace">     + (exp(x_j - m_new) / l_new) × v_j</text>
    <text x="20" y="105" font-size="11" fill="#8b949e">前半：旧输出按新概率重新归一化 | 后半：新块贡献以新权重加权 V</text>
  </g>

  <!-- Key insight -->
  <text x="380" y="490" text-anchor="middle" font-size="13" fill="#a371f7" font-weight="bold">关键缩放因子：exp(m - m_new) 保证全局参考点一致</text>
</svg>'''


def hbm_comparison() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">
  <rect width="760" height="420" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">FlashAttention HBM 访问对比（N=4096, d=64）</text>

  <!-- Standard Attention -->
  <g transform="translate(40, 65)">
    <text x="0" y="0" font-size="16" fill="#f85149" font-weight="bold">标准 Attention</text>
    <rect x="0" y="10" width="680" height="140" rx="8" fill="#f85149" opacity="0.08" stroke="#f85149" stroke-width="2"/>

    <text x="20" y="35" font-size="12" fill="#c9d1d9" font-family="monospace">读 Q:  N×d = 262K</text>
    <text x="20" y="55" font-size="12" fill="#c9d1d9" font-family="monospace">读 K:  N×d = 262K</text>
    <text x="20" y="75" font-size="12" fill="#f85149" font-family="monospace">写 S:  N×N = 16M  ← O(N²) 瓶颈</text>
    <text x="20" y="95" font-size="12" fill="#f85149" font-family="monospace">读 S:  N×N = 16M</text>
    <text x="20" y="115" font-size="12" fill="#f85149" font-family="monospace">写 P:  N×N = 16M  ← O(N²) 瓶颈</text>
    <text x="350" y="35" font-size="12" fill="#c9d1d9" font-family="monospace">读 P:  N×N = 16M</text>
    <text x="350" y="55" font-size="12" fill="#c9d1d9" font-family="monospace">读 V:  N×d = 262K</text>
    <text x="350" y="75" font-size="12" fill="#c9d1d9" font-family="monospace">写 O:  N×d = 262K</text>

    <rect x="350" y="90" width="300" height="40" rx="4" fill="#f85149" opacity="0.2" stroke="#f85149"/>
    <text x="500" y="115" text-anchor="middle" font-size="14" fill="#f85149" font-weight="bold">总计 ≈ 48M elements ≈ 192MB</text>
  </g>

  <!-- FlashAttention -->
  <g transform="translate(40, 225)">
    <text x="0" y="0" font-size="16" fill="#3fb950" font-weight="bold">FlashAttention</text>
    <rect x="0" y="10" width="680" height="120" rx="8" fill="#3fb950" opacity="0.08" stroke="#3fb950" stroke-width="2"/>

    <text x="20" y="35" font-size="12" fill="#3fb950" font-family="monospace">读 Q:  N×d = 262K  （Q tile 驻留 SRAM）</text>
    <text x="20" y="55" font-size="12" fill="#c9d1d9" font-family="monospace">读 K:  N×d = 262K  （K tile 逐块滑入）</text>
    <text x="20" y="75" font-size="12" fill="#c9d1d9" font-family="monospace">读 V:  N×d = 262K  （V tile 逐块滑入）</text>
    <text x="20" y="95" font-size="12" fill="#c9d1d9" font-family="monospace">写 O:  N×d = 262K</text>
    <text x="350" y="55" font-size="12" fill="#3fb950" font-family="monospace">S, P 在 SRAM 中计算</text>
    <text x="350" y="75" font-size="12" fill="#3fb950" font-family="monospace">不写入 HBM</text>

    <rect x="350" y="90" width="300" height="35" rx="4" fill="#3fb950" opacity="0.2" stroke="#3fb950"/>
    <text x="500" y="113" text-anchor="middle" font-size="14" fill="#3fb950" font-weight="bold">总计 ≈ 1M elements ≈ 4MB</text>
  </g>

  <!-- Speedup -->
  <rect x="160" y="370" width="440" height="35" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
  <text x="380" y="392" text-anchor="middle" font-size="15" fill="#58a6ff" font-weight="bold">HBM 访问减少 ~48x → 长序列加速 2-4x</text>
</svg>'''


def main() -> None:
    diagrams = {
        "flash_attention_tiling.svg": flash_attention_tiling(),
        "online_softmax_formula.svg": online_softmax_formula(),
        "hbm_comparison.svg": hbm_comparison(),
    }
    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
