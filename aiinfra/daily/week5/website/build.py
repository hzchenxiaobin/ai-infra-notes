#!/usr/bin/env python3
"""
Build the Week 5 website from README.md (overview) and dayN/README.md (per-day).
Generates:
  - index.html: overview page (from week5/README.md)
  - dayN.html: one page per day (from week5/dayN/README.md)
Uses relative paths (../css/..., ../js/...) for shared resources,
since week5/ is one level below the deployment root on GitHub Pages.
"""

import re
from pathlib import Path
from typing import Optional

PLAN_SOURCE = Path(__file__).parent.parent.parent.parent.parent / "aiinfra" / "daily" / "plan" / "AI_Infra_8_week_plan_detailed.md"
WEEK5_DIR = Path(__file__).parent.parent


def rewrite_md_links_to_html(markdown_text: str, root_prefix: str = "") -> str:
    """Rewrite local .md links to .html for GitHub Pages deployment."""

    def replace_link(match):
        url = match.group(1)
        if not url.endswith(".md"):
            return match.group(0)
        new_url = url[:-3] + ".html"
        if new_url.endswith("README.html"):
            new_url = new_url[: -len("README.html")] + "index.html"
        # ../../../../leetgpu/x.md -> <root_prefix>leetgpu/x.html
        # ../../../../leetcode/daily/weekN/dayM/x.md -> <root_prefix>leetcode/problems/x.html
        if new_url.startswith("../../../../"):
            inner = new_url[len("../../../../"):]
            # LeetGPU solution pages are emitted flat by leetgpu/website/build.py:
            # leetgpu/weekN/dayM/leetgpu-xxx-solution.md -> leetgpu/leetgpu-xxx-solution.html
            inner = re.sub(
                r"^leetgpu/week\d+/day\d+/(leetgpu-.*-solution\.html)$",
                r"leetgpu/\1",
                inner,
            )
            # LeetCode solution pages are emitted flat by leetcode/website/build.py:
            # leetcode/daily/weekN/dayM/xxx.md -> leetcode/problems/xxx.html
            inner = re.sub(
                r"^leetcode/daily/week\d+/day\d+/([^/]+\.html)$",
                r"leetcode/problems/\1",
                inner,
            )
            new_url = root_prefix + inner
        return f"]({new_url})"

    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", replace_link, markdown_text)


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
    # Escape closing </script> so that any inline <script> tags inside the
    # markdown do not prematurely close the outer <script> element that holds
    # the markdown template literal.
    text = text.replace("</script>", "\\x3c/script>")
    return text


def load_overview_and_days():
    """Load overview from week5/README.md and per-day markdown from week5/dayN/README.md."""

    readme_path = WEEK5_DIR / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Week 5 README not found: {readme_path}")
    overview = readme_path.read_text(encoding="utf-8").replace("](website/images/", "](images/")

    day_title_pattern = re.compile(r"^## Day (\d+)[：:]\s*(.+)$")
    days = []
    for day_dir in sorted(WEEK5_DIR.glob("day*")):
        readme = day_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        text = re.sub(r"\]\((?:\.\./)?(?:website/)?images/", "](images/", text)
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = day_title_pattern.match(first_line)
        if not match:
            raise ValueError(f"Cannot parse Day title from first line of {readme}: {first_line!r}")
        days.append({
            "num": int(match.group(1)),
            "title": match.group(2).strip(),
            "markdown": "\n".join(text.strip().splitlines()[1:]),
        })

    if not days:
        raise ValueError(f"No day*/README.md files found in {WEEK5_DIR}")
    days.sort(key=lambda d: d["num"])
    return overview, days


def get_day_info(week_dir: Path) -> list:
    """Return sorted day info [{'num': int, 'title': str}, ...] by parsing README titles."""
    day_title_pattern = re.compile(r"^## Day (\d+)[：:]\s*(.+)$")
    info = []
    for day_dir in sorted(week_dir.glob("day*")):
        readme = day_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = day_title_pattern.match(first_line)
        if match:
            info.append({"num": int(match.group(1)), "title": match.group(2).strip()})
    return sorted(info, key=lambda d: d["num"])


def build_nav(current_day: Optional[int] = None, weeks: Optional[list] = None,
              days: Optional[list] = None, current_is_overview: bool = False) -> str:
    if weeks is None:
        weeks = []
    if days is None:
        days = []
    existing_days = days

    lines = []

    lines.append('<a class="nav-link" href="../index.html">📌 课程概览</a>')

    lines.append('<div class="nav-section-title">8 周学习路线</div>')

    # Titles for all weeks; week 5 title is hardcoded, others come from the plan.
    week_titles = {5: "推理系统与 KV Cache"}
    for week in weeks:
        week_titles[week["num"]] = week["title"]

    repo_root = WEEK5_DIR.parent
    existing_days = get_day_info(WEEK5_DIR)
    week_data = [
        {
            "num": 1,
            "href": "../week1/index.html",
            "day_prefix": "../week1/",
            "days": get_day_info(repo_root / "week1"),
        },
        {
            "num": 2,
            "href": "../week2/index.html",
            "day_prefix": "../week2/",
            "days": get_day_info(repo_root / "week2"),
        },
        {
            "num": 3,
            "href": "../week3/index.html",
            "day_prefix": "../week3/",
            "days": get_day_info(repo_root / "week3"),
        },
        {
            "num": 4,
            "href": "../week4/index.html",
            "day_prefix": "../week4/",
            "days": get_day_info(repo_root / "week4"),
        },
        {
            "num": 5,
            "href": "index.html",
            "day_prefix": "",
            "days": existing_days,
        },
        {
            "num": 6,
            "href": "../week6/index.html",
            "day_prefix": "../week6/",
            "days": get_day_info(repo_root / "week6"),
        },
        {
            "num": 7,
            "href": "../week7/index.html",
            "day_prefix": "../week7/",
            "days": get_day_info(repo_root / "week7"),
        },
        {
            "num": 8,
            "href": "../week8/index.html",
            "day_prefix": "../week8/",
            "days": get_day_info(repo_root / "week8"),
        },
    ]
    for week in weeks:
        if week["num"] <= 8:
            continue
        week_data.append({
            "num": week["num"],
            "href": f"../plan.html#week-{week['num']}",
            "day_prefix": "",
            "days": [],
        })

    for info in week_data:
        is_current = info["num"] == 5
        expanded_cls = " is-expanded" if is_current else ""
        week_active_cls = " active" if (is_current and not current_is_overview) else ""
        overview_active_cls = " active" if (is_current and current_is_overview) else ""
        aria_expanded = "true" if is_current else "false"
        toggle_icon = "▼" if is_current else "▶"

        lines.append(f'<div class="nav-accordion-item{expanded_cls}">')
        lines.append('  <div class="nav-accordion-header">')
        lines.append(
            f'    <a class="nav-link week-link{week_active_cls}" href="{info["href"]}">'
            f'Week {info["num"]}：{week_titles.get(info["num"], "")}'
            f'</a>'
            f'<button class="nav-accordion-toggle" aria-label="收起/展开 Week {info["num"]}" aria-expanded="{aria_expanded}">{toggle_icon}</button>'
        )
        lines.append('  </div>')
        lines.append('  <div class="nav-accordion-content">')
        lines.append('    <div class="nav-section">')
        lines.append(f'<a class="nav-link overview-link{overview_active_cls}" href="{info["href"]}">📌 Week {info["num"]} 概览</a>')
        for day in info["days"]:
            day_num = day["num"]
            day_active = " active" if (is_current and current_day == day_num) else ""
            day_title = day["title"]
            lines.append(
                f'<a class="nav-link day-link{day_active}" href="{info["day_prefix"]}day{day_num}.html">Day {day_num}：{day_title}</a>'
            )
        lines.append('    </div>')
        lines.append('  </div>')
        lines.append('</div>')

    return "\n".join(lines)


def page_template(title: str, nav_html: str, markdown: str,
                  is_overview: bool = False, page_title: Optional[str] = None) -> str:
    escaped_markdown = escape_for_template_string(markdown)
    page_title = page_title if page_title is not None else f"Week 5 - {title}"
    back_link = '' if is_overview else '<a class="back-link" href="index.html">← 返回概览</a>'
    bottom_nav = '' if is_overview else '<div class="day-nav-bottom"><a class="back-link" href="index.html">← 返回概览</a></div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="../css/style.css?v=3">
    <script src="../js/marked.min.js"></script>
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
    <script src="../js/main.js?v=4"></script>
</body>
</html>
"""


def build_website(output_dir: Path) -> None:
    overview, days = load_overview_and_days()

    overview = rewrite_md_links_to_html(overview, root_prefix="../")
    for day in days:
        day["markdown"] = rewrite_md_links_to_html(day["markdown"], root_prefix="../")

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

    overview_with_cards = overview + '\n\n' + day_cards_html

    # Generate overview page
    overview_html = page_template(
        title="Week 5 概览",
        nav_html=build_nav(current_day=None, weeks=plan_weeks, days=days, current_is_overview=True),
        markdown=overview_with_cards,
        is_overview=True,
        page_title="Week 5 - 推理系统与 KV Cache",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    # Generate day pages
    for day in days:
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=build_nav(current_day=day["num"], weeks=plan_weeks, days=days, current_is_overview=False),
            markdown=day["markdown"],
            is_overview=False,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    output_dir = base_dir
    build_website(output_dir)
