#!/usr/bin/env python3
"""
Build the Week 1 website from README.md.
Generates:
  - index.html: overview page
  - day1.html ~ day7.html: one page per day
  - Extra markdown pages as HTML (e.g., exercise/day3/occupancy_problems.html)
  - Copies referenced source directories (kernels/, exercise/, notes/) for download links
"""

import re
import shutil
from pathlib import Path
from typing import Optional


OCCUPANCY_CALCULATOR_MARKER = '<div id="occ-calc-placeholder"></div>'

# Markdown documents that should also be deployed as standalone HTML pages.
# Paths are relative to week1/ (the parent directory of week1/website/).
EXTRA_MARKDOWN_PAGES = [
    {
        "source": "exercise/day3/occupancy_problems.md",
        "output": "exercise/day3/occupancy_problems.html",
        "title": "Occupancy 手算练习题",
    },
    {
        "source": "notes/cuda_programming_guide_performance.md",
        "output": "notes/cuda_programming_guide_performance.html",
        "title": "CUDA Programming Guide 性能优化笔记",
    },
    {
        "source": "notes/week1_notes.md",
        "output": "notes/week1_notes.html",
        "title": "Week 1 学习笔记模板",
    },
]

# Source directories to copy into the website output so that links to .cu,
# .md (download), and other files keep working on GitHub Pages.
EXTRA_SOURCE_DIRECTORIES = [
    "kernels",
    "exercise",
    "notes",
]


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    return text


def rewrite_md_links_to_html(markdown_text: str) -> str:
    """Rewrite local .md links to .html for GitHub Pages deployment.

    README.md source uses .md links so they work on GitHub's markdown viewer.
    When deployed to GitHub Pages, the markdown pages are rendered as .html,
    so the links need to point to .html files.
    """
    def replace_link(match):
        url = match.group(1)
        # Only rewrite local links ending with .md (keep external URLs and anchors as-is)
        if url.endswith(".md"):
            new_url = url[:-3] + ".html"
            # README.md is rendered as the overview page index.html
            if new_url.endswith("README.html"):
                new_url = new_url[: -len("README.html")] + "index.html"
            return f"]({new_url})"
        return match.group(0)

    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", replace_link, markdown_text)


def split_by_days(markdown_text: str):
    """Split README into overview + 7 daily sections."""
    # Pattern: ## Day N：...
    day_pattern = re.compile(r"^(## Day (\d+)[：:].*)$", re.MULTILINE)

    matches = list(day_pattern.finditer(markdown_text))
    if not matches:
        raise ValueError("No Day sections found in README.md")

    # Content before first Day heading -> overview
    overview = markdown_text[:matches[0].start()].strip()

    days = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section = markdown_text[start:end].strip()

        day_num = int(match.group(2))
        title_match = re.match(r"^## Day \d+[：:]\s*(.+)$", match.group(1))
        title = title_match.group(1) if title_match else f"Day {day_num}"

        days.append({
            "num": day_num,
            "title": title,
            "markdown": section,
        })

    return overview, days


def build_nav(current_day: Optional[int]) -> str:
    """Build sidebar navigation. current_day=None means overview page."""
    lines = []

    overview_class = "nav-link active" if current_day is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="index.html">📌 课程概览</a>')

    lines.append('<div class="nav-section-title">每日任务</div>')
    for day in range(1, 8):
        cls = "nav-link day-link active" if current_day == day else "nav-link day-link"
        lines.append(f'<a class="{cls}" href="day{day}.html">Day {day}</a>')

    return "\n".join(lines)


def page_template(title: str, nav_html: str, markdown: str, is_overview: bool = False, extra_scripts: str = "") -> str:
    escaped_markdown = escape_for_template_string(markdown)
    page_title = f"Week 1 - {title}"
    back_link = '<a class="back-link" href="index.html">← 返回概览</a>' if not is_overview else ''
    bottom_nav = '<div class="day-nav-bottom"><a class="back-link" href="index.html">← 返回概览</a></div>' if not is_overview else ''

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="css/style.css">
    <!-- Marked.js for Markdown rendering (local v4.3.0) -->
    <script src="js/marked.min.js"></script>
    <!-- Prism.js for syntax highlighting (local) -->
    <link href="css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="js/prism.min.js"></script>
    <script src="js/prism-c.min.js"></script>
    <script src="js/prism-bash.min.js"></script>
    <script src="js/prism-python.min.js"></script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="index.html" style="text-decoration: none;">
                    <h1 class="sidebar-title">AI Infra 8 周计划</h1>
                    <p class="sidebar-subtitle">Week 1 学习指南</p>
                </a>
            </div>
            <nav class="sidebar-nav">
{nav_html}
            </nav>
        </aside>

        <main class="main-content">
            <div class="page-header">
                <h1 class="page-title">{title}</h1>
                {back_link}
            </div>
            <article class="content" id="content"></article>
            {bottom_nav}
        </main>
    </div>

    <button class="back-to-top" aria-label="Back to top">↑</button>

    <script>
        const markdown = `{escaped_markdown}`;

        const renderer = new marked.Renderer();
        renderer.heading = function(text, level, raw) {{
            let anchor = raw.toLowerCase()
                .replace(/[^\\w\\s-]/g, '')
                .replace(/\\s+/g, '-')
                .replace(/-+/g, '-')
                .replace(/^-|-$/g, '');

            const dayMatch = raw.match(/^Day (\\d+)[:：]\\s*(.+)$/);
            if (dayMatch) {{
                anchor = 'day-' + dayMatch[1];
            }}

            if (level === 2 && anchor) {{
                return '<h' + level + ' id="' + anchor + '">' + text + '</h' + level + '>';
            }}
            return '<h' + level + '>' + text + '</h' + level + '>';
        }};

        marked.setOptions({{
            renderer: renderer,
            headerIds: false,
            gfm: true,
            breaks: false,
            sanitize: false
        }});

        try {{
            if (typeof marked === 'undefined') {{
                throw new Error('marked.js failed to load. Please check js/marked.min.js exists.');
            }}
            document.getElementById('content').innerHTML = marked.parse(markdown);

            if (window.Prism) {{
                Prism.highlightAll();
            }}
        }} catch (err) {{
            document.getElementById('content').innerHTML = '<div style="padding: 20px; color: #ff7b72; background: #2d1515; border-radius: 8px;">' +
                '<h2>⚠️ 页面渲染失败</h2>' +
                '<p>' + err.message + '</p>' +
                '<p>请打开浏览器控制台（Cmd + Option + J）查看详细错误。</p>' +
                '</div>';
            console.error('Markdown render error:', err);
        }}
    </script>
    {extra_scripts}
    <script src="js/main.js"></script>
</body>
</html>
"""


def copy_extra_directories(base_dir: Path, output_dir: Path) -> None:
    """Copy source directories into website output for download/source-code links."""
    for rel_dir in EXTRA_SOURCE_DIRECTORIES:
        src = (base_dir / rel_dir).resolve()
        dst_name = Path(rel_dir).name
        dst = output_dir / dst_name
        if not src.exists():
            print(f"Warning: source directory not found: {src}")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied: {src} -> {dst}")


def build_extra_pages(base_dir: Path, output_dir: Path, nav_html: str) -> None:
    """Build standalone HTML pages from extra markdown documents."""
    for page in EXTRA_MARKDOWN_PAGES:
        source_path = (base_dir / page["source"]).resolve()
        output_path = output_dir / page["output"]
        if not source_path.exists():
            print(f"Warning: extra page source not found: {source_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_text = source_path.read_text(encoding="utf-8")
        # README references images as "website/images/xxx.svg"; adjust for website root.
        markdown_text = markdown_text.replace("](website/images/", "](images/")
        # Rewrite .md links to .html so they work on GitHub Pages.
        markdown_text = rewrite_md_links_to_html(markdown_text)

        html = page_template(
            title=page["title"],
            nav_html=nav_html,
            markdown=markdown_text,
            is_overview=False,
        )
        output_path.write_text(html, encoding="utf-8")
        print(f"Generated: {output_path}")


def build_website(readme_path: Path, output_dir: Path) -> None:
    markdown_text = readme_path.read_text(encoding="utf-8")
    # README references images as "website/images/xxx.svg" (for GitHub viewing),
    # but website HTML is in website/, so we need to reference them as "images/xxx.svg"
    markdown_text = markdown_text.replace("](website/images/", "](images/")
    overview, days = split_by_days(markdown_text)

    # Rewrite .md links to .html for GitHub Pages deployment.
    overview = rewrite_md_links_to_html(overview)
    for day in days:
        day["markdown"] = rewrite_md_links_to_html(day["markdown"])

    # Build day cards HTML for overview page
    day_cards_html = '<div class="day-cards">\n'
    for day in days:
        day_cards_html += (
            f'<a class="day-card" href="day{day["num"]}.html">\n'
            f'  <div class="day-card-number">Day {day["num"]}</div>\n'
            f'  <div class="day-card-title">{day["title"]}</div>\n'
            f'</a>\n'
        )
    day_cards_html += '</div>\n'

    overview_with_cards = overview + '\n\n## 🚀 进入每日学习\n\n' + day_cards_html

    # Generate overview page
    overview_html = page_template(
        title="课程概览",
        nav_html=build_nav(current_day=None),
        markdown=overview_with_cards,
        is_overview=True,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    # Generate day pages
    for day in days:
        has_calc = OCCUPANCY_CALCULATOR_MARKER in day["markdown"]
        extra_scripts = (
            '<script src="js/occupancy-calculator.js"></script>'
            if has_calc else ""
        )
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=build_nav(current_day=day["num"]),
            markdown=day["markdown"],
            is_overview=False,
            extra_scripts=extra_scripts,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")

    # Build extra markdown pages and copy source directories for GitHub Pages links.
    copy_extra_directories(output_dir.parent, output_dir)
    build_extra_pages(output_dir.parent, output_dir, build_nav(current_day=None))


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    readme_path = base_dir.parent / "README.md"
    output_dir = base_dir
    build_website(readme_path, output_dir)
