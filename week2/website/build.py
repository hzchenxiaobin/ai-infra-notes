#!/usr/bin/env python3
"""
Build the Week 2 website from README.md.
Generates:
  - index.html: overview page
  - day1.html ~ dayN.html: one page per day
Uses relative paths (../css/..., ../js/...) for shared resources,
since week2/ is one level below the deployment root on GitHub Pages.
"""

import re
from pathlib import Path
from typing import Optional

PLAN_SOURCE = Path(__file__).parent.parent.parent / "docs" / "AI_Infra_8_week_plan_detailed.md"


def extract_plan_weeks(plan_path: Path) -> list:
    if not plan_path.exists():
        return []
    text = plan_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s*[^\s]*\s*Week\s*(\d+)[:：]\s*(.+)$", re.MULTILINE)
    weeks = []
    for match in pattern.finditer(text):
        weeks.append({"num": int(match.group(1)), "title": match.group(2).strip()})
    return weeks


def escape_for_template_string(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    return text


def split_by_days(markdown_text: str):
    day_pattern = re.compile(r"^(## Day (\d+)[：:].*)$", re.MULTILINE)
    matches = list(day_pattern.finditer(markdown_text))
    if not matches:
        raise ValueError("No Day sections found in README.md")
    overview = markdown_text[:matches[0].start()].strip()
    days = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section = markdown_text[start:end].strip()
        day_num = int(match.group(2))
        title_match = re.match(r"^## Day \d+[：:]\s*(.+)$", match.group(1))
        title = title_match.group(1) if title_match else f"Day {day_num}"
        days.append({"num": day_num, "title": title, "markdown": section})
    return overview, days


def build_nav(current_day: Optional[int] = None, weeks: Optional[list] = None,
              days: Optional[list] = None) -> str:
    if weeks is None:
        weeks = []
    if days is None:
        days = []
    existing_days = [d["num"] for d in days]

    lines = []

    overview_active = current_day is None
    overview_class = "nav-link active" if overview_active else "nav-link"
    lines.append(f'<a class="{overview_class}" href="index.html">📌 Week 2 概览</a>')

    lines.append('<div class="nav-section-title">8 周学习路线</div>')

    # Week 1 link (not expanded on week2 pages)
    lines.append('<div class="nav-accordion-item">')
    lines.append('  <div class="nav-accordion-header">')
    lines.append('    <a class="nav-link week-link" href="../index.html">Week 1：GPU 执行本质 + Profiling</a>')
    lines.append('  </div>')
    lines.append('</div>')

    # Week 2 (expanded with days)
    lines.append('<div class="nav-accordion-item is-expanded">')
    lines.append('  <div class="nav-accordion-header">')
    week2_cls = "nav-link week-link active" if current_day is not None else "nav-link week-link"
    lines.append(f'    <a class="{week2_cls}" href="index.html">Week 2：CUDA 进阶优化与性能分析</a>')
    lines.append('    <button class="nav-accordion-toggle" aria-label="收起/展开 Week 2" aria-expanded="true">▼</button>')
    lines.append('  </div>')
    lines.append('  <div class="nav-accordion-content">')
    lines.append('    <div class="nav-section">')
    for day_num in existing_days:
        cls = "nav-link day-link active" if current_day == day_num else "nav-link day-link"
        lines.append(f'<a class="{cls}" href="day{day_num}.html">Day {day_num}</a>')
    lines.append('    </div>')
    lines.append('  </div>')
    lines.append('</div>')

    # Weeks 3-8
    for week in weeks:
        if week["num"] <= 2:
            continue
        lines.append('<div class="nav-accordion-item">')
        lines.append('  <div class="nav-accordion-header">')
        lines.append(
            f'    <a class="nav-link week-link" href="../plan.html#week-{week["num"]}">'
            f'Week {week["num"]}：{week["title"]}'
            f'</a>'
        )
        lines.append('  </div>')
        lines.append('</div>')

    # More section
    lines.append('<div class="nav-section-title">更多</div>')
    lines.append('<a class="nav-link" href="../leetcode/index.html">🧩 LeetCode 题解</a>')

    return "\n".join(lines)


def page_template(title: str, nav_html: str, markdown: str,
                  is_overview: bool = False, page_title: Optional[str] = None) -> str:
    escaped_markdown = escape_for_template_string(markdown)
    page_title = page_title if page_title is not None else f"Week 2 - {title}"
    back_link = '' if is_overview else '<a class="back-link" href="index.html">← 返回概览</a>'
    bottom_nav = '' if is_overview else '<div class="day-nav-bottom"><a class="back-link" href="index.html">← 返回概览</a></div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="../css/style.css">
    <script src="../js/marked.min.js"></script>
    <link href="../css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="../js/prism.min.js"></script>
    <script src="../js/prism-c.min.js"></script>
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
    <script src="../js/main.js"></script>
</body>
</html>
"""


def build_website(readme_path: Path, output_dir: Path) -> None:
    markdown_text = readme_path.read_text(encoding="utf-8")
    overview, days = split_by_days(markdown_text)

    plan_weeks = extract_plan_weeks(PLAN_SOURCE)

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
        title="Week 2 概览",
        nav_html=build_nav(current_day=None, weeks=plan_weeks, days=days),
        markdown=overview_with_cards,
        is_overview=True,
        page_title="Week 2 - CUDA 进阶优化",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    # Generate day pages
    for day in days:
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=build_nav(current_day=day["num"], weeks=plan_weeks, days=days),
            markdown=day["markdown"],
            is_overview=False,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    readme_path = base_dir.parent / "README.md"
    output_dir = base_dir
    build_website(readme_path, output_dir)
