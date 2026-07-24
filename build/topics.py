"""Builder for all topic websites under aiinfra/topics/."""

import re
import shutil
from pathlib import Path
from typing import Optional

from .common import HEADING_RENDERER_TOPICS, REPO_ROOT, page_template

TOPICS_DIR = REPO_ROOT / "aiinfra" / "topics"
IMAGES_SRC = TOPICS_DIR / "images"

TOPIC_DISPLAY_NAMES = {
    "cpp": "🖥️ C++",
    "cuda": "🟢 CUDA",
    "cutlass": "⚡ CUTLASS",
    "triton": "🐍 Triton",
    "cute": "🔷 CuTe",
    "deepgemm": "🔶 DeepGEMM",
    "moe": "🧩 MoE",
}


def topic_display(slug: str) -> str:
    """Return a human-readable display name for a topic slug."""
    return TOPIC_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


def discover_topics() -> list:
    """Return sorted list of topic slugs that have README.md."""
    topics = []
    skip = {"images", "website", "__pycache__"}
    for subdir in sorted(TOPICS_DIR.iterdir()):
        if subdir.is_dir() and subdir.name not in skip and (subdir / "README.md").exists():
            topics.append(subdir.name)
    return topics


def _rewrite_local_paths(markdown_text: str) -> str:
    """Rewrite local asset/cross-links so they work after deployment."""
    markdown_text = re.sub(r"\]\(\.\./images/", "](images/", markdown_text)
    markdown_text = re.sub(
        r"\]\(\.\./([a-zA-Z0-9_-]+)/README\.md\)",
        r"](../\1/index.html)",
        markdown_text,
    )
    markdown_text = re.sub(
        r"\]\(\.\./([a-zA-Z0-9_-]+)/day(\d+)\.md\)",
        r"](../\1/day\2.html)",
        markdown_text,
    )
    markdown_text = re.sub(
        r"\]\(\.\./\.\./paper/([a-zA-Z0-9_-]+)/README\.md\)",
        r"](../paper/\1/index.html)",
        markdown_text,
    )
    markdown_text = re.sub(
        r"\]\(\.\./\.\./daily/([^)]+)/README\.md\)",
        r"](../daily/\1.html)",
        markdown_text,
    )
    return markdown_text


def _extract_day_headings_from_readme(readme_text: str) -> list:
    """Parse ## Day N headings from README to generate overview cards/anchors."""
    heading_pattern = re.compile(r"^## Day (\d+)[（(][^)）]*[）)]*[：:]\s*(.+)$", re.MULTILINE)
    days = []
    for match in heading_pattern.finditer(readme_text):
        days.append({
            "num": int(match.group(1)),
            "title": match.group(2).strip(),
        })
    days.sort(key=lambda d: d["num"])
    return days


def _extract_day_files(topic_dir: Path) -> list:
    """Extract day info from <topic>/dayN.md files if they exist."""
    day_title_pattern = re.compile(r"^# Day (\d+)(?:[（(][^)）]*[）)])*[：:]\s*(.+)$")
    days = []
    for md_path in sorted(topic_dir.glob("day*.md")):
        text = md_path.read_text(encoding="utf-8")
        text = _rewrite_local_paths(text)
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = day_title_pattern.match(first_line)
        if not match:
            print(f"Warning: skipping {md_path}, cannot parse Day title")
            continue
        days.append({
            "num": int(match.group(1)),
            "title": match.group(2).strip(),
            "markdown": "\n".join(text.strip().splitlines()[1:]),
        })
    days.sort(key=lambda d: d["num"])
    return days


def _build_nav(topic_slug: str, topic_display_name: str,
               current_day: Optional[int] = None, nav_days: list = None,
               day_files: list = None) -> str:
    """Build sidebar navigation for a topic site."""
    if nav_days is None:
        nav_days = []
    if day_files is None:
        day_files = []
    has_day_files = bool(day_files)

    lines = []
    lines.append('<div class="nav-section-title">返回主站</div>')
    lines.append('<a class="nav-link" href="../index.html">← AI Infra 主页</a>')
    lines.append('<a class="nav-link" href="../plan.html">📋 8 周计划</a>')

    lines.append(f'<div class="nav-section-title" style="margin-top:1rem;">{topic_display_name} 专题</div>')

    overview_active = " active" if current_day is None else ""
    lines.append(f'<a class="nav-link{overview_active}" href="index.html">📌 专题概览</a>')

    for day in nav_days:
        day_active = " active" if current_day == day["num"] else ""
        if has_day_files:
            href = f'day{day["num"]}.html'
        else:
            href = f'index.html#day-{day["num"]}'
        lines.append(
            f'<a class="nav-link day-link{day_active}" href="{href}">'
            f'Day {day["num"]}：{day["title"]}'
            f'</a>'
        )

    return "\n".join(lines)


def _copy_topic_images(topic_slug: str, output_dir: Path) -> int:
    """Copy topic-related SVG images from aiinfra/topics/images/ to output/images/."""
    dst = output_dir / "images"
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    prefix = f"{topic_slug}_"
    if IMAGES_SRC.exists():
        for item in IMAGES_SRC.iterdir():
            if item.is_file() and item.suffix.lower() == ".svg" and item.name.startswith(prefix):
                shutil.copy2(item, dst / item.name)
                copied += 1
    print(f"Copied {copied} {topic_slug} SVG images")
    return copied


def _copy_local_dirs(topic_dir: Path, output_dir: Path) -> None:
    """Copy local asset dirs (kernels, notes, benchmark, images) if they exist."""
    for name in ("kernels", "notes", "benchmark", "images"):
        src = topic_dir / name
        if src.exists() and any(src.iterdir()):
            dst = output_dir / name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied: {src} -> {dst}")


def _build_topic(topic_dir: Path, output_dir: Path) -> None:
    """Generate a single topic website."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = topic_dir.name
    display = topic_display(slug)

    readme_path = topic_dir / "README.md"
    overview = _rewrite_local_paths(readme_path.read_text(encoding="utf-8"))
    readme_days = _extract_day_headings_from_readme(overview)
    day_files = _extract_day_files(topic_dir)

    if day_files:
        day_cards_html = '<div class="day-cards">\n'
        for day in day_files:
            day_cards_html += (
                f'<a class="day-card" href="day{day["num"]}.html">\n'
                f'  <div class="day-card-number">Day {day["num"]}</div>\n'
                f'  <div class="day-card-title">{day["title"]}</div>\n'
                f'</a>\n'
            )
        day_cards_html += '</div>\n'
        overview_with_cards = overview + '\n\n## 🚀 进入每日学习\n\n' + day_cards_html
    else:
        if readme_days:
            day_cards_html = '<div class="day-cards">\n'
            for day in readme_days:
                day_cards_html += (
                    f'<a class="day-card" href="#day-{day["num"]}">\n'
                    f'  <div class="day-card-number">Day {day["num"]}</div>\n'
                    f'  <div class="day-card-title">{day["title"]}</div>\n'
                    f'</a>\n'
                )
            day_cards_html += '</div>\n'
            overview_with_cards = overview + '\n\n## 🚀 进入每日学习\n\n' + day_cards_html
        else:
            overview_with_cards = overview

    nav_days = day_files if day_files else readme_days
    root_prefix = "../"
    overview_html = page_template(
        title=f"{display} 专题",
        nav_html=_build_nav(slug, display, current_day=None, nav_days=nav_days, day_files=day_files),
        markdown=overview_with_cards,
        is_overview=True,
        root_prefix=root_prefix,
        page_title=f"{display} 专题 - AI Infra 学习笔记",
        heading_renderer_js=HEADING_RENDERER_TOPICS,
        back_link_href="index.html",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for day in day_files:
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=_build_nav(slug, display, current_day=day["num"], nav_days=nav_days, day_files=day_files),
            markdown=day["markdown"],
            is_overview=False,
            root_prefix=root_prefix,
            page_title=f"{display} Day {day['num']} - {day['title']}",
            heading_renderer_js=HEADING_RENDERER_TOPICS,
            back_link_href="index.html",
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")

    _copy_topic_images(slug, output_dir)
    _copy_local_dirs(topic_dir, output_dir)


def build(public_dir: Path) -> None:
    """Build all topic websites into public_dir/<topic>/."""
    topics = discover_topics()
    print(f"Discovered {len(topics)} topics: {topics}")
    for slug in topics:
        _build_topic(TOPICS_DIR / slug, public_dir / slug)
