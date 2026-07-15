#!/usr/bin/env python3
"""
Build the Week 1 website from README.md (overview) and dayN/README.md (per-day).
Generates:
  - index.html: overview page
  - day1.html ~ day7.html: one page per day (from week1/dayN/README.md)
  - Extra markdown pages as HTML (e.g., exercise/occupancy_problems.html)
  - Copies referenced source directories (kernels/, exercise/, notes/) for download links
"""

import re
import shutil
from pathlib import Path
from typing import Optional


OCCUPANCY_CALCULATOR_MARKER = '<div id="occ-calc-placeholder"></div>'

# Source markdown for the full 8-week plan overview page.
PLAN_SOURCE = Path(__file__).parent.parent.parent / "plan" / "AI_Infra_8_week_plan_detailed.md"
WEEK1_DIR = Path(__file__).parent.parent

# Markdown documents that should also be deployed as standalone HTML pages.
# Paths are relative to week1/ (the parent directory of week1/website/).
EXTRA_MARKDOWN_PAGES = [
    {
        "source": "day3/exercise/occupancy_problems.md",
        "output": "exercise/occupancy_problems.html",
        "title": "Occupancy 手算练习题",
    },
    {
        "source": "day3/notes/cuda_programming_guide_performance.md",
        "output": "notes/cuda_programming_guide_performance.html",
        "title": "CUDA Programming Guide 性能优化笔记",
    },
    {
        "source": "notes/week1_notes.md",
        "output": "notes/week1_notes.html",
        "title": "Week 1 学习笔记模板",
    },
    {
        "source": "profiles/week1_profile_summary.md",
        "output": "profiles/week1_profile_summary.html",
        "title": "Week 1 Profiling 报告汇总",
    },
]

# Subdirectories under each dayN/ to copy (merged flat) into the website output.
DAY_SOURCE_SUBDIRS = ["kernels", "exercise", "notes"]


def extract_plan_weeks(plan_path: Path) -> list:
    """Extract week numbers and titles from the 8-week plan markdown."""
    if not plan_path.exists():
        return []

    text = plan_path.read_text(encoding="utf-8")
    # Match headings like "## 🚀 Week 1：Title" or "## Week 1: Title"
    pattern = re.compile(r"^##\s*[^\s]*\s*Week\s*(\d+)[:：]\s*(.+)$", re.MULTILINE)
    weeks = []
    for match in pattern.finditer(text):
        weeks.append({
            "num": int(match.group(1)),
            "title": match.group(2).strip(),
        })
    return weeks


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    # Escape closing </script> so that any inline <script> tags inside the
    # markdown do not prematurely close the outer <script> element that holds
    # the markdown template literal.
    text = text.replace("</script>", "\\x3c/script>")
    return text


def compute_root_prefix(output_path: Path, output_dir: Path) -> str:
    """Return the relative prefix (e.g. '../') from output_path's directory to output_dir."""
    try:
        rel_parent = output_path.parent.relative_to(output_dir)
    except ValueError:
        return ""
    depth = len(rel_parent.parts)
    return "../" * depth if depth > 0 else ""


def rewrite_md_links_to_html(markdown_text: str, root_prefix: str = "") -> str:
    """Rewrite local .md links to .html for GitHub Pages deployment.

    README.md source uses .md links so they work on GitHub's markdown viewer.
    When deployed to GitHub Pages, the markdown pages are rendered as .html,
    so the links need to point to .html files.

    Links that escape the week directory (starting with ../../) are rewritten
    to relative paths from the generated page's location (e.g. leetgpu/x.html
    for root-level day pages, ../leetgpu/x.html for subdir pages).
    """
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


def load_overview_and_days():
    """Load overview from week1/README.md and per-day markdown from week1/dayN/README.md.

    Returns (overview_text, days) where days is a list of
    {"num": int, "title": str, "markdown": str} sorted by day number.
    Image paths are rewritten from "../website/images/" to "images/" so they
    resolve correctly in the website output directory.
    """
    readme_path = WEEK1_DIR / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Week 1 README not found: {readme_path}")
    overview = readme_path.read_text(encoding="utf-8")
    overview = re.sub(r"\]\((?:website/)?images/", "](images/", overview)

    day_title_pattern = re.compile(r"^## Day (\d+)[：:]\s*(.+)$")
    days = []
    for day_dir in sorted(WEEK1_DIR.glob("day*")):
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
        raise ValueError(f"No day*/README.md files found in {WEEK1_DIR}")
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


def build_nav(
    current_day: Optional[int] = None,
    current_page: str = "week1",
    root_prefix: str = "",
    weeks: Optional[list] = None,
) -> str:
    """Build sidebar navigation.

    current_page: "week1" for week1 pages, "plan" for the 8-week plan page.
    current_day: active day number for week1 day pages.
    weeks: list of {"num": int, "title": str} for Week 2~8 from the detailed plan.
    """
    if weeks is None:
        weeks = []

    lines = []

    overview_active = current_page == "week1" and current_day is None
    overview_class = "nav-link active" if overview_active else "nav-link"
    lines.append(f'<a class="{overview_class}" href="{root_prefix}index.html">📌 课程概览</a>')

    lines.append('<div class="nav-section-title">8 周学习路线</div>')

    # Titles for all 8 weeks; week 1 title is hardcoded, others come from the plan.
    week_titles = {1: "GPU 执行本质 + Profiling"}
    for week in weeks:
        week_titles[week["num"]] = week["title"]

    # Week metadata: href, day-link prefix, and available day numbers.
    repo_root = WEEK1_DIR.parent
    week_data = [
        {
            "num": 1,
            "href": f"{root_prefix}index.html",
            "day_prefix": root_prefix,
            "days": get_day_info(WEEK1_DIR),
        },
        {
            "num": 2,
            "href": f"{root_prefix}week2/index.html",
            "day_prefix": f"{root_prefix}week2/",
            "days": get_day_info(repo_root / "week2"),
        },
        {
            "num": 3,
            "href": f"{root_prefix}week3/index.html",
            "day_prefix": f"{root_prefix}week3/",
            "days": get_day_info(repo_root / "week3"),
        },
        {
            "num": 4,
            "href": f"{root_prefix}week4/index.html",
            "day_prefix": f"{root_prefix}week4/",
            "days": get_day_info(repo_root / "week4"),
        },
        {
            "num": 5,
            "href": f"{root_prefix}week5/index.html",
            "day_prefix": f"{root_prefix}week5/",
            "days": get_day_info(repo_root / "week5"),
        },
        {
            "num": 6,
            "href": f"{root_prefix}week6/index.html",
            "day_prefix": f"{root_prefix}week6/",
            "days": get_day_info(repo_root / "week6"),
        },
        {
            "num": 7,
            "href": f"{root_prefix}week7/index.html",
            "day_prefix": f"{root_prefix}week7/",
            "days": get_day_info(repo_root / "week7"),
        },
        {
            "num": 8,
            "href": f"{root_prefix}week8/index.html",
            "day_prefix": f"{root_prefix}week8/",
            "days": get_day_info(repo_root / "week8"),
        },
    ]
    for week in weeks:
        if week["num"] <= 8:
            continue
        week_data.append({
            "num": week["num"],
            "href": f"{root_prefix}plan.html#week-{week['num']}",
            "day_prefix": "",
            "days": [],
        })

    for info in week_data:
        is_current = current_page == "week1" and info["num"] == 1
        expanded_cls = " is-expanded" if is_current else ""
        active_cls = " active" if is_current else ""
        aria_expanded = "true" if is_current else "false"
        toggle_icon = "▼" if is_current else "▶"

        lines.append(f'<div class="nav-accordion-item{expanded_cls}">')
        lines.append('  <div class="nav-accordion-header">')
        lines.append(
            f'    <a class="nav-link week-link{active_cls}" href="{info["href"]}">'
            f'Week {info["num"]}：{week_titles.get(info["num"], "")}'
            f'</a>'
            f'<button class="nav-accordion-toggle" aria-label="收起/展开 Week {info["num"]}" aria-expanded="{aria_expanded}">{toggle_icon}</button>'
        )
        lines.append('  </div>')
        lines.append('  <div class="nav-accordion-content">')
        lines.append('    <div class="nav-section">')
        lines.append(f'<a class="nav-link overview-link" href="{info["href"]}">📌 Week {info["num"]} 概览</a>')
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


def page_template(
    title: str,
    nav_html: str,
    markdown: str,
    is_overview: bool = False,
    extra_scripts: str = "",
    root_prefix: str = "",
    page_title: Optional[str] = None,
) -> str:
    escaped_markdown = escape_for_template_string(markdown)
    page_title = page_title if page_title is not None else f"Week 1 - {title}"
    back_link = f'<a class="back-link" href="{root_prefix}index.html">← 返回概览</a>' if not is_overview else ''
    bottom_nav = f'<div class="day-nav-bottom"><a class="back-link" href="{root_prefix}index.html">← 返回概览</a></div>' if not is_overview else ''

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="{root_prefix}css/style.css?v=4">
    <!-- Marked.js for Markdown rendering (local v4.3.0) -->
    <script src="{root_prefix}js/marked.min.js"></script>
    <!-- Prism.js for syntax highlighting (local) -->
    <link href="{root_prefix}css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="{root_prefix}js/prism.min.js"></script>
    <script src="{root_prefix}js/prism-c.min.js"></script>
    <script>Prism.languages.cuda=Prism.languages.extend("c",{{builtin:/\\b(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__syncthreads|__threadfence|__threadfence_block|blockIdx|threadIdx|blockDim|gridDim|warpSize)\\b/}});</script>
    <script src="{root_prefix}js/prism-bash.min.js"></script>
    <script src="{root_prefix}js/prism-python.min.js"></script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="{root_prefix}index.html" style="text-decoration: none;">
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
    {extra_scripts}
    <script src="{root_prefix}js/main.js?v=5"></script>
</body>
</html>
"""


def copy_extra_directories(base_dir: Path, output_dir: Path) -> None:
    """Copy source directories from dayN/{kernels,exercise,notes} into website output.

    All day subdirectories are merged flat (e.g. day1/kernels/*.cu and
    day2/kernels/*.cu both go to website/kernels/).  Week-level notes/ and
    tools/ are also copied so their links keep working.
    """
    for subdir in DAY_SOURCE_SUBDIRS:
        dst = output_dir / subdir
        dst.mkdir(parents=True, exist_ok=True)
        # Merge from each dayN/
        for day_dir in sorted(base_dir.glob("day*")):
            src = day_dir / subdir
            if not src.exists():
                continue
            for item in src.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst / item.name)
                else:
                    shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            print(f"Copied: {src} -> {dst}")
        # Also copy week-level notes/ (e.g. week1_notes.md)
        if subdir == "notes":
            week_notes = base_dir / "notes"
            if week_notes.exists():
                for item in week_notes.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst / item.name)
                    else:
                        shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                print(f"Copied: {week_notes} -> {dst}")
        # Also copy week-level tools/
        if subdir == "exercise":
            week_tools = base_dir / "tools"
            if week_tools.exists():
                tools_dst = output_dir / "tools"
                tools_dst.mkdir(parents=True, exist_ok=True)
                for item in week_tools.iterdir():
                    if item.is_file():
                        shutil.copy2(item, tools_dst / item.name)
                print(f"Copied: {week_tools} -> {tools_dst}")

    # Copy week-level profiles/ directory
    week_profiles = base_dir / "profiles"
    if week_profiles.exists():
        profiles_dst = output_dir / "profiles"
        profiles_dst.mkdir(parents=True, exist_ok=True)
        for item in week_profiles.iterdir():
            if item.is_file():
                shutil.copy2(item, profiles_dst / item.name)
        print(f"Copied: {week_profiles} -> {profiles_dst}")


def build_plan_page(output_dir: Path, weeks: list) -> None:
    """Build the full 8-week plan overview page from aiinfra/daily/plan/AI_Infra_8_week_plan_detailed.md."""
    if not PLAN_SOURCE.exists():
        print(f"Warning: 8-week plan source not found: {PLAN_SOURCE}")
        return

    markdown_text = PLAN_SOURCE.read_text(encoding="utf-8")

    # Inject explicit anchors before each Week heading so sidebar links reliably
    # jump to the right section, regardless of the Markdown renderer's id logic.
    def add_week_anchor(match: re.Match) -> str:
        return f'<a id="week-{match.group(2)}"></a>\n{match.group(0)}'

    week_heading_pattern = re.compile(r"^(##\s*[^\s]*\s*Week\s*(\d+)[:：].*)$", re.MULTILINE)
    markdown_text = week_heading_pattern.sub(add_week_anchor, markdown_text)

    nav_html = build_nav(
        current_day=None,
        current_page="plan",
        root_prefix="",
        weeks=weeks,
    )

    html = page_template(
        title="8 周学习计划",
        nav_html=nav_html,
        markdown=markdown_text,
        is_overview=True,
        page_title="AI Infra 8 周计划",
    )
    (output_dir / "plan.html").write_text(html, encoding="utf-8")
    print(f"Generated: {output_dir / 'plan.html'}")


def build_extra_pages(base_dir: Path, output_dir: Path, weeks: Optional[list] = None) -> None:
    """Build standalone HTML pages from extra markdown documents."""
    for page in EXTRA_MARKDOWN_PAGES:
        source_path = (base_dir / page["source"]).resolve()
        output_path = output_dir / page["output"]
        if not source_path.exists():
            print(f"Warning: extra page source not found: {source_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        root_prefix = compute_root_prefix(output_path, output_dir)
        page_nav_html = build_nav(
            current_day=None,
            current_page="week1",
            root_prefix=root_prefix,
            weeks=weeks,
        )
        markdown_text = source_path.read_text(encoding="utf-8")
        # Rewrite image paths (both "website/images/" and "../website/images/")
        markdown_text = re.sub(r"\]\((?:\.\./)?(?:website/)?images/", "](images/", markdown_text)
        # Rewrite .md links to .html so they work on GitHub Pages.
        markdown_text = rewrite_md_links_to_html(markdown_text, root_prefix=root_prefix)

        html = page_template(
            title=page["title"],
            nav_html=page_nav_html,
            markdown=markdown_text,
            is_overview=False,
            root_prefix=root_prefix,
        )
        output_path.write_text(html, encoding="utf-8")
        print(f"Generated: {output_path}")


def build_website(output_dir: Path) -> None:
    overview, days = load_overview_and_days()

    # Rewrite .md links to .html for GitHub Pages deployment.
    # Overview and day pages are generated at the site root.
    overview = rewrite_md_links_to_html(overview, root_prefix="")
    for day in days:
        day["markdown"] = rewrite_md_links_to_html(day["markdown"], root_prefix="")

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

    # Load week titles from the detailed 8-week plan for the sidebar.
    plan_weeks = extract_plan_weeks(PLAN_SOURCE)

    # Generate overview page
    overview_html = page_template(
        title="课程概览",
        nav_html=build_nav(current_day=None, current_page="week1", weeks=plan_weeks),
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
            nav_html=build_nav(current_day=day["num"], current_page="week1", weeks=plan_weeks),
            markdown=day["markdown"],
            is_overview=False,
            extra_scripts=extra_scripts,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")

    # Build the full 8-week plan overview page.
    build_plan_page(output_dir, plan_weeks)

    # Build extra markdown pages and copy source directories for GitHub Pages links.
    copy_extra_directories(output_dir.parent, output_dir)
    build_extra_pages(output_dir.parent, output_dir, plan_weeks)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    output_dir = base_dir
    build_website(output_dir)
