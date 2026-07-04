#!/usr/bin/env python3
"""
Build the LeetCode solution website from markdown files in leetcode/.
Generates:
  - index.html: problem list page
  - problems/<slug>.html: individual solution pages
Uses relative paths so the site works when deployed under a repository
path prefix (e.g. https://user.github.io/repo-name/leetcode/).
"""

import re
from pathlib import Path
from typing import List, Dict, Optional


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    return text


def parse_title(markdown_text: str) -> str:
    """Extract title from the first level-1 heading."""
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    return match.group(1).strip() if match else "题解"


def build_nav(current_slug: Optional[str], problems: List[Dict], root_prefix: str) -> str:
    """Build sidebar navigation. current_slug=None means overview page."""
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="{root_prefix}leetcode/index.html">📌 题解列表</a>')

    lines.append('<div class="nav-section-title">题目</div>')
    for p in problems:
        cls = "nav-link active" if current_slug == p["slug"] else "nav-link"
        lines.append(f'<a class="{cls}" href="{root_prefix}leetcode/problems/{p["slug"]}.html">{p["title"]}</a>')

    return "\n".join(lines)


def page_template(title: str, nav_html: str, markdown: str, root_prefix: str) -> str:
    escaped_markdown = escape_for_template_string(markdown)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="{root_prefix}css/style.css?v=2">
    <!-- Marked.js for Markdown rendering -->
    <script src="{root_prefix}js/marked.min.js"></script>
    <!-- Prism.js for syntax highlighting -->
    <link href="{root_prefix}css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="{root_prefix}js/prism.min.js"></script>
    <script src="{root_prefix}js/prism-c.min.js"></script>
    <script src="{root_prefix}js/prism-bash.min.js"></script>
    <script src="{root_prefix}js/prism-python.min.js"></script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="{root_prefix}index.html" style="text-decoration: none;">
                    <h1 class="sidebar-title">AI Infra 学习笔记</h1>
                    <p class="sidebar-subtitle">LeetCode 题解</p>
                </a>
            </div>
            <nav class="sidebar-nav">
{nav_html}
            </nav>
        </aside>

        <main class="main-content">
            <div class="page-header">
                <h1 class="page-title">{title}</h1>
                <a class="back-link" href="{root_prefix}leetcode/index.html">← 返回题解列表</a>
            </div>
            <article class="content" id="content"></article>
            <div class="day-nav-bottom"><a class="back-link" href="{root_prefix}leetcode/index.html">← 返回题解列表</a></div>
        </main>
    </div>

    <button class="back-to-top" aria-label="Back to top">↑</button>

    <script>
        const markdown = `{escaped_markdown}`;

        const renderer = new marked.Renderer();
        marked.setOptions({{
            renderer: renderer,
            headerIds: false,
            gfm: true,
            breaks: false,
            sanitize: false
        }});

        try {{
            if (typeof marked === 'undefined') {{
                throw new Error('marked.js failed to load.');
            }}
            document.getElementById('content').innerHTML = marked.parse(markdown);

            if (window.Prism) {{
                Prism.highlightAll();
            }}
        }} catch (err) {{
            document.getElementById('content').innerHTML = '<div style="padding: 20px; color: #ff7b72; background: #2d1515; border-radius: 8px;">' +
                '<h2>⚠️ 页面渲染失败</h2>' +
                '<p>' + err.message + '</p>' +
                '</div>';
            console.error('Markdown render error:', err);
        }}
    </script>
    <script src="{root_prefix}js/main.js?v=3"></script>
</body>
</html>"""


def build_website(leetcode_dir: Path, output_dir: Path) -> None:
    problems_dir = output_dir / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)

    # Find all markdown files in leetcode/ (excluding website/ and images/)
    md_files = sorted([
        f for f in leetcode_dir.iterdir()
        if f.is_file() and f.suffix == ".md" and f.parent.name == "leetcode"
    ])

    problems = []
    for md_file in md_files:
        markdown_text = md_file.read_text(encoding="utf-8")

        title = parse_title(markdown_text)
        slug = md_file.stem
        problems.append({
            "slug": slug,
            "title": title,
            "markdown": markdown_text,
        })

    # Build overview page (at leetcode/index.html -> root_prefix="../")
    cards_html = '<div class="day-cards">\n'
    for p in problems:
        cards_html += (
            f'<a class="day-card" href="./problems/{p["slug"]}.html">\n'
            f'  <div class="day-card-number">LeetCode</div>\n'
            f'  <div class="day-card-title">{p["title"]}</div>\n'
            f'</a>\n'
        )
    cards_html += '</div>\n'

    overview_markdown = "# LeetCode 题解\n\n> 算法题解题笔记与思路整理。\n\n## 题目列表\n\n" + cards_html
    # Markdown references images as "images/xxx.svg"; overview is at leetcode/index.html
    overview_markdown = overview_markdown.replace("](images/", "](./images/")

    overview_html = page_template(
        title="LeetCode 题解",
        nav_html=build_nav(current_slug=None, problems=problems, root_prefix="../"),
        markdown=overview_markdown,
        root_prefix="../",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    # Build problem pages (at leetcode/problems/<slug>.html -> root_prefix="../../")
    for p in problems:
        # Problem pages are one level deeper, so images are at ../images/
        problem_markdown = p["markdown"].replace("](images/", "](../images/")
        html = page_template(
            title=p["title"],
            nav_html=build_nav(current_slug=p["slug"], problems=problems, root_prefix="../../"),
            markdown=problem_markdown,
            root_prefix="../../",
        )
        slug_html = f"{p['slug']}.html"
        (problems_dir / slug_html).write_text(html, encoding="utf-8")
        print(f"Generated: {problems_dir / slug_html}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    leetcode_dir = base_dir.parent
    output_dir = base_dir
    build_website(leetcode_dir, output_dir)
