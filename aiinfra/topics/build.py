#!/usr/bin/env python3
"""
Build all topic websites under aiinfra/topics/.

Discovers every subdirectory that contains README.md, generates:
  - website/<topic>/index.html      (overview from README.md)
  - website/<topic>/day1.html ~ dayN.html (when dayN.md files exist)
  - website/<topic>/images/         (topic-related SVGs)
  - website/<topic>/kernels/ etc.   (copied local assets if present)

The root build.py copies website/ into public/, so assets use root_prefix="../".
"""

import re
import shutil
from pathlib import Path
from typing import Optional


TOPICS_DIR = Path(__file__).parent
WEBSITE_DIR = TOPICS_DIR / "website"
IMAGES_SRC = TOPICS_DIR / "images"

# Display names for known topics; new topics fall back to title-cased slug.
TOPIC_DISPLAY_NAMES = {
    "cutlass": "⚡ CUTLASS",
    "triton": "🐍 Triton",
    "cute": "🔷 CuTe",
    "deepgemm": "🔶 DeepGEMM",
    "moe": "🧩 MoE",
}


def topic_display(slug: str) -> str:
    """Return a human-readable display name for a topic slug."""
    return TOPIC_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("</script>", "\\x3c/script>")
    return text


def rewrite_local_paths(markdown_text: str) -> str:
    """Rewrite local asset/cross-links so they work after deployment to public/<topic>/."""
    # Images referenced as ../images/xxx.svg
    markdown_text = re.sub(r"\]\(\.\./images/", "](images/", markdown_text)
    # Cross-topic README links: ../cutlass/README.md -> ../cutlass/index.html
    markdown_text = re.sub(
        r"\]\(\.\./([a-zA-Z0-9_-]+)/README\.md\)",
        r"](../\1/index.html)",
        markdown_text,
    )
    # Cross-topic day links: ../cutlass/day7.md -> ../cutlass/day7.html
    markdown_text = re.sub(
        r"\]\(\.\./([a-zA-Z0-9_-]+)/day(\d+)\.md\)",
        r"](../\1/day\2.html)",
        markdown_text,
    )
    # Paper links: ../../paper/flashattention3/README.md -> ../paper/flashattention3/index.html
    markdown_text = re.sub(
        r"\]\(\.\./\.\./paper/([a-zA-Z0-9_-]+)/README\.md\)",
        r"](../paper/\1/index.html)",
        markdown_text,
    )
    # Daily links: ../../daily/week6/day1/README.md -> ../daily/week6/day1.html
    markdown_text = re.sub(
        r"\]\(\.\./\.\./daily/([^)]+)/README\.md\)",
        r"](../daily/\1.html)",
        markdown_text,
    )
    return markdown_text


def extract_day_headings_from_readme(readme_text: str) -> list:
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


def extract_day_files(topic_dir: Path) -> list:
    """Extract day info from <topic>/dayN.md files if they exist."""
    day_title_pattern = re.compile(r"^# Day (\d+)(?:[（(][^)）]*[）)])*[：:]\s*(.+)$")
    days = []
    for md_path in sorted(topic_dir.glob("day*.md")):
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


def build_nav(topic_slug: str, topic_display_name: str,
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


def page_template(
    title: str,
    nav_html: str,
    markdown: str,
    is_overview: bool = False,
    page_title: Optional[str] = None,
) -> str:
    escaped_markdown = escape_for_template_string(markdown)
    page_title = page_title if page_title is not None else title
    back_link = '<a class="back-link" href="index.html">← 返回概览</a>' if not is_overview else ""
    bottom_nav = '<div class="day-nav-bottom"><a class="back-link" href="index.html">← 返回概览</a></div>' if not is_overview else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="../css/style.css?v=4">
    <script src="../js/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="../js/markdown-math.js"></script>
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

            const dayMatch = raw.match(/^Day (\\d+)(?:[（(][^)）]*[）)])*[:：]\\s*(.+)$/);
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


def copy_topic_images(topic_slug: str, output_dir: Path) -> int:
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


def copy_local_dirs(topic_dir: Path, output_dir: Path) -> None:
    """Copy local asset dirs (kernels, notes, benchmark, images) if they exist."""
    for name in ("kernels", "notes", "benchmark", "images"):
        src = topic_dir / name
        if src.exists() and any(src.iterdir()):
            dst = output_dir / name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied: {src} -> {dst}")


def build_topic(topic_dir: Path, output_dir: Path) -> None:
    """Generate a single topic website."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = topic_dir.name
    display = topic_display(slug)

    readme_path = topic_dir / "README.md"
    overview = rewrite_local_paths(readme_path.read_text(encoding="utf-8"))
    readme_days = extract_day_headings_from_readme(overview)
    day_files = extract_day_files(topic_dir)

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
    overview_html = page_template(
        title=f"{display} 专题",
        nav_html=build_nav(slug, display, current_day=None, nav_days=nav_days, day_files=day_files),
        markdown=overview_with_cards,
        is_overview=True,
        page_title=f"{display} 专题 - AI Infra 学习笔记",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for day in day_files:
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=build_nav(slug, display, current_day=day["num"], nav_days=nav_days, day_files=day_files),
            markdown=day["markdown"],
            is_overview=False,
            page_title=f"{display} Day {day['num']} - {day['title']}",
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")

    copy_topic_images(slug, output_dir)
    copy_local_dirs(topic_dir, output_dir)


def discover_topics() -> list:
    """Return sorted list of topic slugs that have README.md."""
    topics = []
    skip = {"images", "website", "__pycache__"}
    for subdir in sorted(TOPICS_DIR.iterdir()):
        if subdir.is_dir() and subdir.name not in skip and (subdir / "README.md").exists():
            topics.append(subdir.name)
    return topics


def main() -> None:
    if WEBSITE_DIR.exists():
        shutil.rmtree(WEBSITE_DIR)
    WEBSITE_DIR.mkdir(parents=True, exist_ok=True)

    topics = discover_topics()
    print(f"Discovered {len(topics)} topics: {topics}")
    for slug in topics:
        build_topic(TOPICS_DIR / slug, WEBSITE_DIR / slug)


if __name__ == "__main__":
    main()
