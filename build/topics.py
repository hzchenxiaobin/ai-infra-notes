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
    "interview": "💼 Interview",
    "transformer": "🤖 Transformer",
    "misc": "📒 杂七杂八",
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
    def daily_repl(match):
        rel = match.group(1)
        day_match = re.fullmatch(r"(week\d+)/day(\d+)", rel)
        if day_match:
            return f"](../{day_match.group(1)}/day{day_match.group(2)}.html)"
        return f"](../{rel}/index.html)"

    markdown_text = re.sub(
        r"\]\(\.\./\.\./daily/([^)]+)/README\.md\)",
        daily_repl,
        markdown_text,
    )
    markdown_text = re.sub(
        r"\]\((day\d+)\.md\)",
        r"](\1.html)",
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


SOLUTION_GROUP_LABELS = {
    "high": "高频题",
    "medium": "中频题",
    "low": "低频题",
}

SOLUTION_TYPE_LABELS = {
    "reduction": "归约",
    "gemm": "矩阵乘",
    "matrix-ops": "矩阵操作",
    "attention": "注意力",
    "scan": "前缀和",
    "gemv": "矩阵向量",
    "selection": "选择",
    "elementwise": "逐元素",
    "convolution": "卷积",
}

SOLUTION_GROUP_ORDER = {"high": 0, "medium": 1, "low": 2}

SOLUTION_SKIP_DIRS = {"challenges", "kernels", "notes", "benchmark", "images", "__pycache__"}


def _rewrite_solution_paths(markdown_text: str, depth: int) -> str:
    """Rewrite asset/cross-links in a solution file located `depth` dirs below
    the topic dir so they work on the generated page (which mirrors the source
    tree). Images live in <output>/images/; relative .md links become .html."""
    markdown_text = re.sub(
        r"\]\((?:\.\./)+images/",
        "](" + "../" * depth + "images/",
        markdown_text,
    )

    def repl(match):
        url = match.group(1)
        if url.endswith("README.md"):
            url = url[: -len("README.md")] + "index.html"
        else:
            url = url[:-3] + ".html"
        return "](" + url + ")"

    return re.sub(r"\]\((?!https?://|#)([^)]+\.md)\)", repl, markdown_text)


def _extract_solution_files(topic_dir: Path) -> list:
    """Extract standalone solution pages (<slug>.md, excluding README/SKILL/dayN),
    recursing into frequency/type subdirectories (e.g. high/reduction/)."""
    solution_title_pattern = re.compile(r"^#\s+LeetGPU\s+(.+?)\s+题解\s*$")
    solutions = []
    for md_path in sorted(topic_dir.rglob("*.md")):
        rel = md_path.relative_to(topic_dir)
        if any(part in SOLUTION_SKIP_DIRS for part in rel.parts[:-1]):
            continue
        name = md_path.name
        if name in ("README.md", "SKILL.md") or re.match(r"day\d+\.md$", name):
            continue
        depth = len(rel.parts) - 1
        text = md_path.read_text(encoding="utf-8")
        text = _rewrite_solution_paths(text, depth)
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = solution_title_pattern.match(first_line)
        if match:
            title = match.group(1).strip()
        elif first_line.startswith("#"):
            title = first_line.lstrip("# ").strip()
        else:
            print(f"Warning: skipping {md_path}, cannot parse solution title")
            continue
        solutions.append({
            "slug": rel.with_suffix("").as_posix(),
            "depth": depth,
            "group": rel.parts[0] if depth >= 1 else "",
            "type": rel.parts[1] if depth >= 2 else "",
            "title": title,
            "markdown": "\n".join(text.strip().splitlines()[1:]),
        })
    solutions.sort(key=lambda s: (SOLUTION_GROUP_ORDER.get(s["group"], 99), s["slug"]))
    return solutions


def _build_nav(topic_slug: str, topic_display_name: str,
               current_day: Optional[int] = None, nav_days: list = None,
               day_files: list = None, nav_solutions: list = None,
               current_slug: Optional[str] = None, href_prefix: str = "") -> str:
    """Build sidebar navigation for a topic site.

    `href_prefix` is prepended to all hrefs so pages in subdirectories
    (e.g. high/reduction/) can still reach topic-root and main-site pages.
    """
    if nav_days is None:
        nav_days = []
    if day_files is None:
        day_files = []
    if nav_solutions is None:
        nav_solutions = []
    has_day_files = bool(day_files)

    lines = []
    lines.append('<div class="nav-section-title">返回主站</div>')
    lines.append(f'<a class="nav-link" href="{href_prefix}../index.html">← AI Infra 主页</a>')
    lines.append(f'<a class="nav-link" href="{href_prefix}../plan.html">📋 10 周计划</a>')

    lines.append(f'<div class="nav-section-title" style="margin-top:1rem;">{topic_display_name} 专题</div>')

    overview_active = " active" if current_day is None and current_slug is None else ""
    lines.append(f'<a class="nav-link{overview_active}" href="{href_prefix}index.html">📌 专题概览</a>')

    for day in nav_days:
        day_active = " active" if current_day == day["num"] else ""
        if has_day_files:
            href = f'{href_prefix}day{day["num"]}.html'
        else:
            href = f'{href_prefix}index.html#day-{day["num"]}'
        lines.append(
            f'<a class="nav-link day-link{day_active}" href="{href}">'
            f'Day {day["num"]}：{day["title"]}'
            f'</a>'
        )

    if nav_solutions:
        lines.append('<div class="nav-section-title" style="margin-top:1rem;">📝 LeetGPU 题解</div>')
        for sol in nav_solutions:
            sol_active = " active" if current_slug == sol["slug"] else ""
            label = solution_label(sol)
            lines.append(
                f'<a class="nav-link day-link{sol_active}" href="{href_prefix}{sol["slug"]}.html">'
                f'{label}{sol["title"]}'
                f'</a>'
            )

    return "\n".join(lines)


def solution_label(sol: dict) -> str:
    """Return a short '[高频题 · 归约] '-style prefix for a solution, or ''."""
    group = SOLUTION_GROUP_LABELS.get(sol.get("group", ""), "")
    stype = SOLUTION_TYPE_LABELS.get(sol.get("type", ""), "")
    if group and stype:
        return f"[{group} · {stype}] "
    if group:
        return f"[{group}] "
    return ""


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
    solution_files = _extract_solution_files(topic_dir)

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

    if solution_files:
        sol_cards_html = '<div class="day-cards">\n'
        for sol in solution_files:
            label = solution_label(sol)
            sol_cards_html += (
                f'<a class="day-card" href="{sol["slug"]}.html">\n'
                f'  <div class="day-card-number">{label or "题解"}</div>\n'
                f'  <div class="day-card-title">{sol["title"]}</div>\n'
                f'</a>\n'
            )
        sol_cards_html += '</div>\n'
        overview_with_cards += '\n\n## 📝 LeetGPU 题解\n\n' + sol_cards_html

    nav_days = day_files if day_files else readme_days
    root_prefix = "../"
    overview_html = page_template(
        title=f"{display} 专题",
        nav_html=_build_nav(slug, display, current_day=None, nav_days=nav_days,
                            day_files=day_files, nav_solutions=solution_files),
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
            nav_html=_build_nav(slug, display, current_day=day["num"], nav_days=nav_days,
                                day_files=day_files, nav_solutions=solution_files),
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

    for sol in solution_files:
        href_prefix = "../" * sol["depth"]
        html = page_template(
            title=f"LeetGPU {sol['title']} 题解",
            nav_html=_build_nav(slug, display, nav_days=nav_days, day_files=day_files,
                                nav_solutions=solution_files, current_slug=sol["slug"],
                                href_prefix=href_prefix),
            markdown=sol["markdown"],
            is_overview=False,
            root_prefix="../" * (sol["depth"] + 1),
            page_title=f"{display} 题解 - {sol['title']}",
            heading_renderer_js=HEADING_RENDERER_TOPICS,
            back_link_href=href_prefix + "index.html",
        )
        out_path = output_dir / f"{sol['slug']}.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"Generated: {out_path}")

    _copy_topic_images(slug, output_dir)
    _copy_local_dirs(topic_dir, output_dir)


def build(public_dir: Path) -> None:
    """Build all topic websites into public_dir/<topic>/."""
    topics = discover_topics()
    print(f"Discovered {len(topics)} topics: {topics}")
    for slug in topics:
        _build_topic(TOPICS_DIR / slug, public_dir / slug)
