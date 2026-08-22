"""Landing page (public/index.html) builder — a designed, sidebar-less home page."""

import html
import re
from pathlib import Path

from .common import COURSE_OVERVIEW_SOURCE, GITHUB_REPO_URL
LEETCODE_URL = "https://hzchenxiaobin.github.io/leetcode/"
LEETGPU_URL = "https://github.com/hzchenxiaobin/leetgpu"

# Phase grouping for the 10-week roadmap
PHASES = [
    ("阶段一", "基础内功", "GPU 执行本质 → Kernel 优化 → Tensor Core → Transformer 算子", range(1, 5)),
    ("阶段二", "推理系统", "FlashAttention → KV Cache → Batching 调度 → 推理加速", range(5, 9)),
    ("阶段三", "分布式与冲刺", "分布式并行 → 项目整合 → 面试冲刺", range(9, 11)),
]

WEEKLY_RHYTHM = [
    ("Day 1-2", "🔬", "理论 + 基础 Kernel", "概念建模 + 最简实现"),
    ("Day 3-4", "📖", "进阶实现 / 源码", "进阶优化或开源源码导读"),
    ("Day 5", "🛠", "项目推进", "接入 Mini 引擎或 benchmark"),
    ("Day 6", "📊", "Profiling", "ncu / nsys 实测 + Roofline"),
    ("Day 7", "🧘", "复盘 + 面试", "知识地图 / 手撕清单 / 面试 Q&A"),
]

_RESOURCES = [
    ("📄", "论文精读", "AI Infra 经典论文逐篇精读笔记", "paper/index.html"),
    ("🧩", "LeetCode 题解", "面试高频算法题解（独立站点）", LEETCODE_URL),
    ("🎮", "LeetGPU 题解", "CUDA 在线刷题与题解（独立仓库）", LEETGPU_URL),
    ("💻", "GitHub 仓库", "本站的全部源码与 Markdown 原文", GITHUB_REPO_URL),
]

_WEEK_ROW_PATTERN = re.compile(
    r"^\|\s*\*\*Week\s+(\d+)\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", re.MULTILINE
)


def _strip_md(text: str) -> str:
    """Remove markdown emphasis markers from a table cell."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def parse_week_cards(fallback_titles: dict) -> list:
    """Parse the 10-week roadmap table from the course overview README.

    Returns [{'num': int, 'title': str, 'goal': str}, ...] sorted by week num.
    Falls back to fallback_titles (num -> title) when parsing yields nothing.
    """
    weeks = []
    if COURSE_OVERVIEW_SOURCE.exists():
        text = COURSE_OVERVIEW_SOURCE.read_text(encoding="utf-8")
        for match in _WEEK_ROW_PATTERN.finditer(text):
            weeks.append({
                "num": int(match.group(1)),
                "title": _strip_md(match.group(2)),
                "goal": _strip_md(match.group(3)),
            })
        weeks.sort(key=lambda w: w["num"])
    if not weeks:
        weeks = [
            {"num": num, "title": title, "goal": ""}
            for num, title in sorted(fallback_titles.items())
        ]
    return weeks


def _split_display(display: str) -> tuple:
    """Split a topic display name like '🧩 MoE' into (icon, name)."""
    parts = display.split(" ", 1)
    if len(parts) == 2 and not parts[0].isascii():
        return parts[0], parts[1]
    return "📁", display


def _build_week_cards_html(weeks: list) -> str:
    by_num = {w["num"]: w for w in weeks}
    out = []
    for phase_no, phase_name, phase_desc, nums in PHASES:
        cards = []
        for num in nums:
            week = by_num.get(num)
            if week is None:
                continue
            title = html.escape(week["title"])
            goal = html.escape(week["goal"])
            cards.append(f'''        <a class="week-card" href="week{num}/index.html">
          <div class="week-card-top">
            <span class="week-card-badge">W{num}</span>
            <span class="week-card-arrow">→</span>
          </div>
          <div class="week-card-title">{title}</div>
          <div class="week-card-goal">{goal}</div>
        </a>''')
        if not cards:
            continue
        cards_html = "\n".join(cards)
        out.append(f'''      <div class="phase-group">
        <div class="phase-header">
          <span class="phase-no">{phase_no}</span>
          <span class="phase-name">{phase_name}</span>
          <span class="phase-desc">{phase_desc}</span>
        </div>
        <div class="week-grid">
{cards_html}
        </div>
      </div>''')
    return "\n".join(out)


def _build_rhythm_html() -> str:
    cards = []
    for days, icon, title, desc in WEEKLY_RHYTHM:
        cards.append(f'''        <div class="rhythm-card">
          <div class="rhythm-days">{days}</div>
          <div class="rhythm-title">{icon} {title}</div>
          <div class="rhythm-desc">{desc}</div>
        </div>''')
    return "\n".join(cards)


def _build_topic_cards_html(topics: list, topic_display) -> str:
    cards = []
    for slug in sorted(topics):
        icon, name = _split_display(topic_display(slug))
        cards.append(f'''        <a class="topic-card" href="{slug}/index.html">
          <span class="topic-card-icon">{icon}</span>
          <span class="topic-card-name">{html.escape(name)}</span>
          <span class="topic-card-arrow">→</span>
        </a>''')
    return "\n".join(cards)


def _build_resource_cards_html() -> str:
    cards = []
    for icon, name, desc, url in _RESOURCES:
        cards.append(f'''        <a class="resource-card" href="{url}">
          <span class="resource-card-icon">{icon}</span>
          <span class="resource-card-body">
            <span class="resource-card-name">{name}</span>
            <span class="resource-card-desc">{desc}</span>
          </span>
        </a>''')
    return "\n".join(cards)


def build_home(public_dir: Path, fallback_week_titles: dict, topics: list, topic_display) -> None:
    """Generate the landing page at public_dir/index.html."""
    weeks = parse_week_cards(fallback_week_titles)
    topic_count = len(topics)

    week_cards_html = _build_week_cards_html(weeks)
    rhythm_html = _build_rhythm_html()
    topic_cards_html = _build_topic_cards_html(topics, topic_display)
    resource_cards_html = _build_resource_cards_html()

    page = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Infra 10 周学习计划</title>
    <meta name="description" content="AI Infra 工程实战学习路线：CUDA Kernel 优化、推理系统、分布式并行，10 周从会写 kernel 进阶到能做系统优化。">
    <link rel="stylesheet" href="css/style.css?v=8">
</head>
<body class="landing">
    <header class="landing-nav">
        <a class="landing-nav-brand" href="index.html">AI Infra <span>Notes</span></a>
        <nav class="landing-nav-links">
            <a href="plan.html">10 周计划</a>
            <a href="#topics">专题笔记</a>
            <a href="paper/index.html">论文精读</a>
            <a class="landing-nav-github" href="{GITHUB_REPO_URL}">GitHub ↗</a>
        </nav>
    </header>

    <section class="hero">
        <div class="hero-inner">
            <div class="hero-eyebrow">工程实战 · 10 周递进式路线</div>
            <h1 class="hero-title">AI Infra <span class="hero-title-accent">10 周学习计划</span></h1>
            <p class="hero-subtitle">从「会写 Kernel」进阶到「能做系统优化」</p>
            <p class="hero-meta">适合具备 CUDA / 算子优化基础，希望转向 AI Infra（推理系统 / 分布式 / 内核优化）的工程师 · 每日 3～5 小时</p>
            <div class="hero-actions">
                <a class="btn btn-primary" href="week1/index.html">🚀 开始 Week 1</a>
                <a class="btn btn-secondary" href="plan.html">📋 查看完整计划</a>
            </div>
        </div>
    </section>

    <section class="stats-strip">
        <div class="stat-item"><span class="stat-value">10</span><span class="stat-label">周学习路线</span></div>
        <div class="stat-item"><span class="stat-value">70+</span><span class="stat-label">每日实战任务</span></div>
        <div class="stat-item"><span class="stat-value">{topic_count}</span><span class="stat-label">专题笔记</span></div>
        <div class="stat-item"><span class="stat-value">∞</span><span class="stat-label">持续更新</span></div>
    </section>

    <main class="landing-main">
        <section class="landing-section" id="roadmap">
            <h2 class="section-title">学习路线</h2>
            <p class="section-subtitle">三个阶段、十个主题，从 GPU 执行本质一路走到分布式推理与面试冲刺。</p>
{week_cards_html}
        </section>

        <section class="landing-section">
            <h2 class="section-title">每周节奏</h2>
            <p class="section-subtitle">每周 7 天固定循环：理论 → 进阶 → 项目 → Profiling → 复盘。</p>
            <div class="rhythm-grid">
{rhythm_html}
            </div>
        </section>

        <section class="landing-section" id="topics">
            <h2 class="section-title">专题笔记</h2>
            <p class="section-subtitle">围绕路线沉淀的专题深挖，可随时按主题查阅。</p>
            <div class="topic-grid">
{topic_cards_html}
            </div>
        </section>

        <section class="landing-section">
            <h2 class="section-title">更多资源</h2>
            <div class="resource-grid">
{resource_cards_html}
            </div>
        </section>
    </main>

    <footer class="landing-footer">
        <span>AI Infra Notes · 由 <a href="{GITHUB_REPO_URL}">GitHub</a> 驱动 · Deployed on GitHub Pages</span>
    </footer>
</body>
</html>
'''
    (public_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Generated: {public_dir / 'index.html'}")
