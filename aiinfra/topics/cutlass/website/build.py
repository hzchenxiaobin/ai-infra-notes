#!/usr/bin/env python3
"""
Build the CUTLASS topic website from README.md (overview) and dayN.md (per-day).

Generates:
  - index.html: overview page from README.md
  - day1.html ~ dayN.html: one page per day
  - images/: copies from aiinfra/topics/images/ (cutlass SVGs)
  - kernels/: copies from aiinfra/topics/cutlass/kernels/

This builder is designed to be copied into public/cutlass/ by the root build.py,
so all asset links use root_prefix="../" to reach public/css/, public/js/, etc.
"""

import re
import shutil
from pathlib import Path
from typing import Optional


CUTLASS_DIR = Path(__file__).parent.parent
TOPICS_DIR = CUTLASS_DIR.parent
WEBSITE_DIR = Path(__file__).parent


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("</script>", "\\x3c/script>")
    return text


def rewrite_local_paths(markdown_text: str) -> str:
    """Rewrite local asset paths so they work after deployment to public/cutlass/."""
    # Images referenced as ../images/xxx.svg from aiinfra/topics/cutlass/dayN.md
    # should become images/xxx.svg relative to public/cutlass/dayN.html.
    markdown_text = re.sub(r"\]\(\.\./images/", "](images/", markdown_text)
    return markdown_text


def extract_days() -> list:
    """Extract day info from cutlass/dayN.md files.

    Returns list of {"num": int, "title": str, "markdown": str} sorted by day number.
    """
    day_title_pattern = re.compile(r"^# Day (\d+)[：:]\s*(.+)$")
    days = []
    for md_path in sorted(CUTLASS_DIR.glob("day*.md")):
        text = md_path.read_text(encoding="utf-8")
        text = rewrite_local_paths(text)
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


def build_nav(current_day: Optional[int] = None) -> str:
    """Build sidebar navigation for the CUTLASS topic site."""
    lines = []
    lines.append('<div class="nav-section-title">返回主站</div>')
    lines.append('<a class="nav-link" href="../index.html">← AI Infra 主页</a>')
    lines.append('<a class="nav-link" href="../plan.html">📋 8 周计划</a>')

    lines.append('<div class="nav-section-title" style="margin-top:1rem;">CUTLASS 专题</div>')

    overview_active = " active" if current_day is None else ""
    lines.append(f'<a class="nav-link{overview_active}" href="index.html">📌 专题概览</a>')

    for day in extract_days():
        day_active = " active" if current_day == day["num"] else ""
        lines.append(
            f'<a class="nav-link day-link{day_active}" href="day{day["num"]}.html">'
            f'Day {day["num"]}：{day["title"]}'
            f'</a>'
        )

    return "\n".join(lines)


def page_template(
    title: str,
    nav_html: str,
    markdown: str,
    is_overview: bool = False,
    page_title: Optional[str] = None,
) -> str:
    escaped_markdown = escape_for_template_string(markdown)
    page_title = page_title if page_title is not None else f"CUTLASS - {title}"
    back_link = '<a class="back-link" href="index.html">← 返回概览</a>' if not is_overview else ""
    bottom_nav = '<div class="day-nav-bottom"><a class="back-link" href="index.html">← 返回概览</a></div>' if not is_overview else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="../css/style.css?v=4">
    <!-- Marked.js for Markdown rendering (local v4.3.0) -->
    <script src="../js/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="../js/markdown-math.js"></script>
    <!-- Prism.js for syntax highlighting (local) -->
    <link href="../css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="../js/prism.min.js"></script>
    <script src="../js/prism-c.min.js"></script>
    <script src="../js/prism-cpp.min.js"></script>
    <script>Prism.languages.cuda=Prism.languages.extend("c",{{builtin:/\\b(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__syncthreads|__threadfence|__threadfence_block|blockIdx|threadIdx|blockDim|gridDim|warpSize)\\b/}});</script>
    <script src="../js/prism-bash.min.js"></script>
    <script src="../js/prism-python.min.js"></script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="../index.html" style="text-decoration: none;">
                    <h1 class="sidebar-title">AI Infra 8 周计划</h1>
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
    <script src="../js/main.js?v=5"></script>
</body>
</html>
"""


def copy_images(output_dir: Path) -> None:
    """Copy cutlass SVG images from aiinfra/topics/images/ to output/images/."""
    src = TOPICS_DIR / "images"
    dst = output_dir / "images"
    if not src.exists():
        print(f"Warning: images source not found: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src.iterdir():
        if item.is_file() and item.suffix.lower() == ".svg":
            shutil.copy2(item, dst / item.name)
            copied += 1
    print(f"Copied {copied} SVG images: {src} -> {dst}")


def copy_kernels(output_dir: Path) -> None:
    """Copy cutlass kernel sources to output/kernels/."""
    src = CUTLASS_DIR / "kernels"
    dst = output_dir / "kernels"
    if not src.exists():
        print(f"Warning: kernels source not found: {src}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Copied: {src} -> {dst}")


def build_website(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    days = extract_days()
    if not days:
        raise ValueError(f"No day*.md files found in {CUTLASS_DIR}")

    # Overview page from README.md
    readme_path = CUTLASS_DIR / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"CUTLASS README not found: {readme_path}")
    overview = rewrite_local_paths(readme_path.read_text(encoding="utf-8"))

    # Append day cards to overview
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

    overview_html = page_template(
        title="CUTLASS 专题",
        nav_html=build_nav(current_day=None),
        markdown=overview_with_cards,
        is_overview=True,
        page_title="CUTLASS 专题 - AI Infra 学习笔记",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    # Per-day pages
    for day in days:
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=build_nav(current_day=day["num"]),
            markdown=day["markdown"],
            is_overview=False,
            page_title=f"CUTLASS Day {day['num']} - {day['title']}",
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")

    copy_images(output_dir)
    copy_kernels(output_dir)


if __name__ == "__main__":
    build_website(WEBSITE_DIR)
