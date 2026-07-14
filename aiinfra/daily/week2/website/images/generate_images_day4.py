#!/usr/bin/env python3
"""Generate SVG diagrams for Week 2 Day 4 (Nsight Compute)."""

from pathlib import Path


def save_svg(filename: str, content: str) -> None:
    path = Path(__file__).parent / filename
    path.write_text(content, encoding="utf-8")
    print(f"Generated: {path}")


def ncu_metrics_overview() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500" viewBox="0 0 760 500">
  <rect width="760" height="500" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">ncu 关键性能指标分类</text>

  <!-- Parallelism -->
  <g transform="translate(40, 65)">
    <rect x="0" y="0" width="220" height="130" rx="8" fill="#1f6feb" opacity="0.1" stroke="#58a6ff" stroke-width="2"/>
    <text x="110" y="25" text-anchor="middle" font-size="14" fill="#58a6ff" font-weight="bold">并行度</text>
    <text x="15" y="50" font-size="11" fill="#c9d1d9">sm__occupancy.avg</text>
    <text x="15" y="70" font-size="10" fill="#8b949e">Occupancy 百分比</text>
    <text x="15" y="90" font-size="11" fill="#c9d1d9">launch__registers_per_thread</text>
    <text x="15" y="110" font-size="10" fill="#8b949e">每线程寄存器数</text>
  </g>

  <!-- Memory -->
  <g transform="translate(270, 65)">
    <rect x="0" y="0" width="220" height="130" rx="8" fill="#a371f7" opacity="0.1" stroke="#a371f7" stroke-width="2"/>
    <text x="110" y="25" text-anchor="middle" font-size="14" fill="#a371f7" font-weight="bold">内存</text>
    <text x="15" y="50" font-size="11" fill="#c9d1d9">dram__throughput.avg</text>
    <text x="15" y="70" font-size="10" fill="#8b949e">显存带宽利用率</text>
    <text x="15" y="90" font-size="11" fill="#c9d1d9">l1tex__data_bank_conflicts</text>
    <text x="15" y="110" font-size="10" fill="#8b949e">Shared mem bank conflict</text>
  </g>

  <!-- Compute -->
  <g transform="translate(500, 65)">
    <rect x="0" y="0" width="220" height="130" rx="8" fill="#3fb950" opacity="0.1" stroke="#3fb950" stroke-width="2"/>
    <text x="110" y="25" text-anchor="middle" font-size="14" fill="#3fb950" font-weight="bold">计算</text>
    <text x="15" y="50" font-size="11" fill="#c9d1d9">sm__throughput.avg</text>
    <text x="15" y="70" font-size="10" fill="#8b949e">SM 计算利用率</text>
    <text x="15" y="90" font-size="11" fill="#c9d1d9">sm__cycles_elapsed.avg</text>
    <text x="15" y="110" font-size="10" fill="#8b949e">执行周期数</text>
  </g>

  <!-- Stall Reasons -->
  <g transform="translate(40, 215)">
    <rect x="0" y="0" width="680" height="120" rx="8" fill="#f85149" opacity="0.1" stroke="#f85149" stroke-width="2"/>
    <text x="340" y="25" text-anchor="middle" font-size="14" fill="#f85149" font-weight="bold">Warp Stall Reasons（阻塞原因分布）</text>
    <text x="15" y="50" font-size="11" fill="#c9d1d9">Long Scoreboard</text>
    <text x="170" y="50" font-size="10" fill="#8b949e">全局内存加载等待 → 增加 tiling / double buffer</text>
    <text x="15" y="72" font-size="11" fill="#c9d1d9">Math Pipe Throttle</text>
    <text x="170" y="72" font-size="10" fill="#8b949e">FMA 过载 → 增加独立指令 / ILP</text>
    <text x="15" y="94" font-size="11" fill="#c9d1d9">MIO Throttle</text>
    <text x="170" y="94" font-size="10" fill="#8b949e">内存指令发射瓶颈 → 减少 smem 访问</text>
    <text x="15" y="116" font-size="11" fill="#c9d1d9">Wait / Barrier</text>
    <text x="170" y="116" font-size="10" fill="#8b949e">同步等待 → 减少同步点</text>
  </g>

  <!-- Normal ranges -->
  <g transform="translate(40, 355)">
    <rect x="0" y="0" width="680" height="120" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="340" y="25" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">正常范围参考</text>
    <text x="20" y="50" font-size="12" fill="#3fb950">SM Throughput &gt; 60%</text>
    <text x="200" y="50" font-size="12" fill="#3fb950">Memory Throughput &gt; 60%</text>
    <text x="420" y="50" font-size="12" fill="#3fb950">Occupancy &gt; 理论值 70%</text>
    <text x="20" y="75" font-size="12" fill="#3fb950">L1 Hit Rate &gt; 80%</text>
    <text x="200" y="75" font-size="12" fill="#d29922">每项 Stall &lt; 20%</text>
    <text x="420" y="75" font-size="12" fill="#3fb950">Register &lt; 80%</text>
    <text x="340" y="105" text-anchor="middle" font-size="12" fill="#58a6ff">指标不达标 → 针对性优化 → 重新 profile 验证</text>
  </g>
</svg>'''


def profile_optimize_loop() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">
  <rect width="760" height="420" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Profile → 优化 → 验证 完整闭环</text>

  <!-- Step 1: Baseline -->
  <g transform="translate(40, 70)">
    <rect x="0" y="0" width="160" height="80" rx="8" fill="#1f6feb" opacity="0.2" stroke="#58a6ff" stroke-width="2"/>
    <text x="80" y="30" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">1. Baseline</text>
    <text x="80" y="52" text-anchor="middle" font-size="11" fill="#8b949e">ncu 采集指标</text>
    <text x="80" y="70" text-anchor="middle" font-size="11" fill="#8b949e">SM/Mem Throughput</text>
  </g>

  <line x1="205" y1="110" x2="245" y2="110" stroke="#58a6ff" stroke-width="2" marker-end="url(#arrR4)"/>

  <!-- Step 2: Bottleneck -->
  <g transform="translate(250, 70)">
    <rect x="0" y="0" width="160" height="80" rx="8" fill="#d29922" opacity="0.2" stroke="#d29922" stroke-width="2"/>
    <text x="80" y="30" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">2. 定位瓶颈</text>
    <text x="80" y="52" text-anchor="middle" font-size="11" fill="#8b949e">Roofline + Stall</text>
    <text x="80" y="70" text-anchor="middle" font-size="11" fill="#8b949e">判断 bound 类型</text>
  </g>

  <line x1="415" y1="110" x2="455" y2="110" stroke="#d29922" stroke-width="2" marker-end="url(#arrR4)"/>

  <!-- Step 3: Optimize -->
  <g transform="translate(460, 70)">
    <rect x="0" y="0" width="160" height="80" rx="8" fill="#a371f7" opacity="0.2" stroke="#a371f7" stroke-width="2"/>
    <text x="80" y="30" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">3. 针对优化</text>
    <text x="80" y="52" text-anchor="middle" font-size="11" fill="#8b949e">只改确认有收益的</text>
    <text x="80" y="70" text-anchor="middle" font-size="11" fill="#8b949e">优化点</text>
  </g>

  <line x1="625" y1="110" x2="665" y2="110" stroke="#a371f7" stroke-width="2" marker-end="url(#arrR4)"/>

  <!-- Step 4: Validate -->
  <g transform="translate(270, 190)">
    <rect x="0" y="0" width="220" height="80" rx="8" fill="#238636" opacity="0.2" stroke="#3fb950" stroke-width="2"/>
    <text x="110" y="30" text-anchor="middle" font-size="14" fill="#c9d1d9" font-weight="bold">4. 重新 Profile 验证</text>
    <text x="110" y="52" text-anchor="middle" font-size="11" fill="#8b949e">对比前后指标变化</text>
    <text x="110" y="70" text-anchor="middle" font-size="11" fill="#8b949e">确认性能提升</text>
  </g>

  <!-- Loop arrow back to step 1 -->
  <path d="M 380 270 Q 380 310 200 310 Q 80 310 80 150" fill="none" stroke="#3fb950" stroke-width="2" stroke-dasharray="4" marker-end="url(#arrU4)"/>
  <text x="80" y="330" font-size="11" fill="#3fb950">循环迭代</text>

  <!-- Decision branches -->
  <g transform="translate(40, 190)">
    <rect x="0" y="0" width="200" height="130" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="100" y="25" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Memory Bound 分支</text>
    <text x="15" y="50" font-size="11" fill="#a371f7">→ Coalesced access 检查</text>
    <text x="15" y="70" font-size="11" fill="#a371f7">→ Bank conflict 检查</text>
    <text x="15" y="90" font-size="11" fill="#a371f7">→ float4 向量化加载</text>
    <text x="15" y="110" font-size="11" fill="#a371f7">→ Double Buffering</text>
  </g>

  <g transform="translate(520, 190)">
    <rect x="0" y="0" width="200" height="130" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="100" y="25" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">Compute Bound 分支</text>
    <text x="15" y="50" font-size="11" fill="#3fb950">→ 增大 TM×TN 计算强度</text>
    <text x="15" y="70" font-size="11" fill="#3fb950">→ 减少 Warp Stall</text>
    <text x="15" y="90" font-size="11" fill="#3fb950">→ 增加 ILP</text>
    <text x="15" y="110" font-size="11" fill="#3fb950">→ 检查 FMA 利用率</text>
  </g>

  <line x1="330" y1="110" x2="140" y2="190" stroke="#d29922" stroke-width="1.5" stroke-dasharray="3"/>
  <line x1="540" y1="110" x2="620" y2="190" stroke="#d29922" stroke-width="1.5" stroke-dasharray="3"/>
</svg>'''


def stall_reason_bar() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400">
  <rect width="760" height="400" fill="#0d1117"/>
  <text x="380" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#c9d1d9">Warp Stall Reasons 示例（Register Blocking GEMM）</text>

  <!-- Bar chart -->
  <g transform="translate(80, 70)">
    <!-- Axes -->
    <line x1="0" y1="280" x2="500" y2="280" stroke="#8b949e" stroke-width="2"/>
    <line x1="0" y1="0" x2="0" y2="280" stroke="#8b949e" stroke-width="2"/>
    <text x="-10" y="145" text-anchor="middle" font-size="13" fill="#c9d1d9" transform="rotate(-90, -10, 145)">Stall 占比 (%)</text>

    <!-- Grid lines -->
    <line x1="0" y1="56" x2="500" y2="56" stroke="#30363d" stroke-width="0.5"/>
    <text x="-5" y="60" text-anchor="end" font-size="10" fill="#8b949e">40%</text>
    <line x1="0" y1="112" x2="500" y2="112" stroke="#30363d" stroke-width="0.5"/>
    <text x="-5" y="116" text-anchor="end" font-size="10" fill="#8b949e">30%</text>
    <line x1="0" y1="168" x2="500" y2="168" stroke="#30363d" stroke-width="0.5"/>
    <text x="-5" y="172" text-anchor="end" font-size="10" fill="#8b949e">20%</text>
    <line x1="0" y1="224" x2="500" y2="224" stroke="#30363d" stroke-width="0.5" stroke-dasharray="3"/>
    <text x="-5" y="228" text-anchor="end" font-size="10" fill="#f85149">10%</text>

    <!-- Long Scoreboard bar (35.2%) -->
    <rect x="30" y="84" width="60" height="196" fill="#f85149" opacity="0.7" stroke="#f85149"/>
    <text x="60" y="78" text-anchor="middle" font-size="12" fill="#f85149" font-weight="bold">35.2%</text>
    <text x="60" y="300" text-anchor="middle" font-size="10" fill="#c9d1d9">Long</text>
    <text x="60" y="314" text-anchor="middle" font-size="10" fill="#c9d1d9">Scoreboard</text>

    <!-- Math Pipe Throttle (12.1%) -->
    <rect x="120" y="212" width="60" height="68" fill="#d29922" opacity="0.7" stroke="#d29922"/>
    <text x="150" y="206" text-anchor="middle" font-size="12" fill="#d29922" font-weight="bold">12.1%</text>
    <text x="150" y="300" text-anchor="middle" font-size="10" fill="#c9d1d9">Math Pipe</text>
    <text x="150" y="314" text-anchor="middle" font-size="10" fill="#c9d1d9">Throttle</text>

    <!-- MIO Throttle (8.5%) -->
    <rect x="210" y="232" width="60" height="48" fill="#8b949e" opacity="0.7" stroke="#8b949e"/>
    <text x="240" y="226" text-anchor="middle" font-size="11" fill="#8b949e">8.5%</text>
    <text x="240" y="300" text-anchor="middle" font-size="10" fill="#c9d1d9">MIO</text>
    <text x="240" y="314" text-anchor="middle" font-size="10" fill="#c9d1d9">Throttle</text>

    <!-- Wait (6.3%) -->
    <rect x="300" y="245" width="60" height="35" fill="#8b949e" opacity="0.5" stroke="#8b949e"/>
    <text x="330" y="239" text-anchor="middle" font-size="11" fill="#8b949e">6.3%</text>
    <text x="330" y="300" text-anchor="middle" font-size="10" fill="#c9d1d9">Wait</text>

    <!-- Others (5.2%) -->
    <rect x="390" y="251" width="60" height="29" fill="#8b949e" opacity="0.3" stroke="#484f58"/>
    <text x="420" y="245" text-anchor="middle" font-size="11" fill="#8b949e">5.2%</text>
    <text x="420" y="300" text-anchor="middle" font-size="10" fill="#c9d1d9">Others</text>
  </g>

  <!-- Analysis box -->
  <g transform="translate(540, 70)">
    <rect x="0" y="0" width="180" height="280" rx="8" fill="#161b22" stroke="#30363d" stroke-width="2"/>
    <text x="90" y="25" text-anchor="middle" font-size="13" fill="#c9d1d9" font-weight="bold">分析结论</text>
    <text x="15" y="55" font-size="11" fill="#f85149">Long Scoreboard</text>
    <text x="15" y="72" font-size="10" fill="#8b949e">35.2% 占比最高</text>
    <text x="15" y="90" font-size="10" fill="#8b949e">→ 全局内存延迟</text>
    <text x="15" y="108" font-size="10" fill="#8b949e">  是主要瓶颈</text>
    <line x1="15" y1="120" x2="165" y2="120" stroke="#30363d"/>
    <text x="15" y="145" font-size="11" fill="#3fb950">优化建议：</text>
    <text x="15" y="165" font-size="10" fill="#c9d1d9">1. float4 向量化加载</text>
    <text x="15" y="185" font-size="10" fill="#c9d1d9">2. Double Buffering</text>
    <text x="15" y="205" font-size="10" fill="#c9d1d9">3. 增大 TM×TN</text>
    <text x="15" y="225" font-size="10" fill="#c9d1d9">   提升计算强度</text>
    <line x1="15" y1="240" x2="165" y2="240" stroke="#30363d"/>
    <text x="90" y="262" text-anchor="middle" font-size="11" fill="#58a6ff">→ 优化后重新</text>
    <text x="90" y="276" text-anchor="middle" font-size="11" fill="#58a6ff">  profile 验证</text>
  </g>
</svg>'''


def main() -> None:
    diagrams = {
        "ncu_metrics_overview.svg": ncu_metrics_overview(),
        "profile_optimize_loop.svg": profile_optimize_loop(),
        "stall_reason_bar.svg": stall_reason_bar(),
    }
    for filename, content in diagrams.items():
        save_svg(filename, content)


if __name__ == "__main__":
    main()
