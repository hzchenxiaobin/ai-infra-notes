"""Unified builder for Week 1-8 websites."""

import re
import shutil
from pathlib import Path
from typing import Optional

from .common import (
    COURSE_OVERVIEW_SOURCE,
    DAILY_DIR,
    DAY_TITLE_PATTERN,
    HEADING_RENDERER_WEEKS,
    PLAN_SOURCE,
    build_day_cards_html,
    compute_root_prefix,
    escape_for_template_string,
    extract_plan_weeks,
    get_day_info,
    load_overview_and_days,
    page_template,
    rewrite_md_links_to_html_weeks,
)

OCCUPANCY_CALCULATOR_MARKER = '<div id="occ-calc-placeholder"></div>'

WEEK_TITLES = {
    1: "GPU 执行本质 + Profiling",
    2: "GEMM & Kernel 优化",
    3: "Transformer 执行本质",
    4: "FlashAttention 深挖",
    5: "推理系统与 KV Cache",
    6: "Batching & 调度",
    7: "系统整合",
    8: "项目打磨 + 面试准备",
}

WEEK_OVERVIEW_PAGE_TITLES = {
    1: "Week 1 概览",
    2: "Week 2 - CUDA 进阶优化",
    3: "Week 3 - Transformer 执行本质与算子手写",
    4: "Week 4 - Transformer 执行本质与算子手写",
    5: "Week 5 - 推理系统与 KV Cache",
    6: "Week 6 - Batching & 调度",
    7: "Week 7 - 系统整合",
    8: "Week 8 - 项目打磨 + 面试准备",
}

WEEKS_WITH_CARDS_HEADING = {1, 2}

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

DAY_SOURCE_SUBDIRS = ["kernels", "exercise", "notes"]


def _week_dir(week_num: int) -> Path:
    return DAILY_DIR / f"week{week_num}"


def rewrite_week1_resource_links(markdown_text: str, root_prefix: str = "") -> str:
    """Rewrite relative resource links so they resolve from pages under week1/."""
    text = re.sub(r"\]\((?:\.\./)?images/", f"]({root_prefix}images/", markdown_text)
    text = text.replace("](../tools/", "](tools/")
    text = text.replace("](../notes/", "](notes/")
    text = re.sub(r"\]\(\.\./day\d+/notes/", "](notes/", text)
    return text


def build_week_nav(
    current_week: Optional[int],
    current_day: Optional[int] = None,
    current_page: str = "week",
    current_is_overview: bool = False,
    root_prefix: str = "",
    weeks: Optional[list] = None,
    current_week_days: Optional[list] = None,
    relative_current_week: bool = True,
) -> str:
    """Build sidebar navigation for any week page (or course overview / plan page).

    current_page: "week" for week pages, "overview" for course overview, "plan" for plan page.
    current_week: which week number is current (None for overview/plan pages).
    relative_current_week: when True, current week links use bare "index.html"/"dayN.html"
        (weeks 2-8). When False, all week links use "{root_prefix}week{N}/..." (week 1).
    """
    if weeks is None:
        weeks = []
    if current_week_days is None:
        current_week_days = []

    lines = []

    overview_active = current_page == "overview"
    overview_class = "nav-link active" if overview_active else "nav-link"
    lines.append(f'<a class="{overview_class}" href="{root_prefix}index.html">📌 课程概览</a>')

    lines.append('<div class="nav-section-title">8 周学习路线</div>')

    week_titles = dict(WEEK_TITLES)
    for week in weeks:
        week_titles[week["num"]] = week["title"]

    week_data = []
    for num in range(1, 9):
        if current_week == num and relative_current_week:
            week_data.append({
                "num": num,
                "href": "index.html",
                "day_prefix": "",
                "days": current_week_days,
            })
        else:
            week_data.append({
                "num": num,
                "href": f"{root_prefix}week{num}/index.html",
                "day_prefix": f"{root_prefix}week{num}/",
                "days": get_day_info(_week_dir(num)) if current_week != num else current_week_days,
            })

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
        is_current = current_page == "week" and info["num"] == current_week
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


def build_week(week_num: int, public_dir: Path, plan_weeks: list) -> None:
    """Build a single week's website (weeks 2-8). Week 1 is handled by build_week1."""
    week_dir = _week_dir(week_num)
    output_dir = public_dir / f"week{week_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    overview, days = load_overview_and_days(week_dir)

    root_prefix = "../"
    overview = rewrite_md_links_to_html_weeks(overview, root_prefix=root_prefix)
    for day in days:
        day["markdown"] = rewrite_md_links_to_html_weeks(day["markdown"], root_prefix=root_prefix)

    cards = build_day_cards_html(days, root_prefix="")
    if week_num in WEEKS_WITH_CARDS_HEADING:
        overview_with_cards = overview + '\n\n## 🚀 进入每日学习\n\n' + cards
    else:
        overview_with_cards = overview + '\n\n' + cards

    overview_html = page_template(
        title=f"Week {week_num} 概览",
        nav_html=build_week_nav(
            current_week=week_num,
            current_day=None,
            current_is_overview=True,
            root_prefix=root_prefix,
            weeks=plan_weeks,
            current_week_days=days,
        ),
        markdown=overview_with_cards,
        is_overview=True,
        root_prefix=root_prefix,
        page_title=WEEK_OVERVIEW_PAGE_TITLES.get(week_num, f"Week {week_num} 概览"),
        heading_renderer_js=HEADING_RENDERER_WEEKS,
        back_link_href="index.html",
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for day in days:
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            nav_html=build_week_nav(
                current_week=week_num,
                current_day=day["num"],
                root_prefix=root_prefix,
                weeks=plan_weeks,
                current_week_days=days,
            ),
            markdown=day["markdown"],
            is_overview=False,
            root_prefix=root_prefix,
            back_link_href="index.html",
            heading_renderer_js=HEADING_RENDERER_WEEKS,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")


def build_plan_page(public_dir: Path, plan_weeks: list) -> None:
    """Build the full 8-week plan overview page."""
    if not PLAN_SOURCE.exists():
        print(f"Warning: 8-week plan source not found: {PLAN_SOURCE}")
        return

    markdown_text = PLAN_SOURCE.read_text(encoding="utf-8")

    def add_week_anchor(match: re.Match) -> str:
        return f'<a id="week-{match.group(2)}"></a>\n{match.group(0)}'

    week_heading_pattern = re.compile(r"^(##\s*[^\s]*\s*Week\s*(\d+)[:：].*)$", re.MULTILINE)
    markdown_text = week_heading_pattern.sub(add_week_anchor, markdown_text)

    nav_html = build_week_nav(
        current_week=None,
        current_page="plan",
        root_prefix="",
        weeks=plan_weeks,
    )

    html = page_template(
        title="8 周学习计划",
        nav_html=nav_html,
        markdown=markdown_text,
        is_overview=True,
        page_title="AI Infra 8 周计划",
        heading_renderer_js=HEADING_RENDERER_WEEKS,
    )
    (public_dir / "plan.html").write_text(html, encoding="utf-8")
    print(f"Generated: {public_dir / 'plan.html'}")


def _copy_extra_directories(week1_dir: Path, output_dir: Path) -> None:
    """Copy source directories from dayN/{kernels,exercise,notes} into website output."""
    for subdir in DAY_SOURCE_SUBDIRS:
        dst = output_dir / subdir
        dst.mkdir(parents=True, exist_ok=True)
        for day_dir in sorted(week1_dir.glob("day*")):
            src = day_dir / subdir
            if not src.exists():
                continue
            for item in src.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst / item.name)
                else:
                    shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            print(f"Copied: {src} -> {dst}")
        if subdir == "notes":
            week_notes = week1_dir / "notes"
            if week_notes.exists():
                for item in week_notes.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst / item.name)
                    else:
                        shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                print(f"Copied: {week_notes} -> {dst}")
        if subdir == "exercise":
            week_tools = week1_dir / "tools"
            if week_tools.exists():
                tools_dst = output_dir / "tools"
                tools_dst.mkdir(parents=True, exist_ok=True)
                for item in week_tools.iterdir():
                    if item.is_file():
                        shutil.copy2(item, tools_dst / item.name)
                print(f"Copied: {week_tools} -> {tools_dst}")

    week_profiles = week1_dir / "profiles"
    if week_profiles.exists():
        profiles_dst = output_dir / "profiles"
        profiles_dst.mkdir(parents=True, exist_ok=True)
        for item in week_profiles.iterdir():
            if item.is_file():
                shutil.copy2(item, profiles_dst / item.name)
        print(f"Copied: {week_profiles} -> {profiles_dst}")


def _build_extra_pages(week1_dir: Path, output_dir: Path, public_dir: Path, plan_weeks: list) -> None:
    """Build standalone HTML pages from extra markdown documents."""
    for page in EXTRA_MARKDOWN_PAGES:
        source_path = (week1_dir / page["source"]).resolve()
        output_path = output_dir / page["output"]
        if not source_path.exists():
            print(f"Warning: extra page source not found: {source_path}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        root_prefix = compute_root_prefix(output_path, public_dir)
        page_nav_html = build_week_nav(
            current_week=1,
            current_day=None,
            current_is_overview=False,
            root_prefix=root_prefix,
            weeks=plan_weeks,
            current_week_days=get_day_info(week1_dir),
            relative_current_week=False,
        )
        markdown_text = source_path.read_text(encoding="utf-8")
        markdown_text = rewrite_md_links_to_html_weeks(markdown_text, root_prefix=root_prefix)
        markdown_text = rewrite_week1_resource_links(markdown_text, root_prefix=root_prefix)

        html = page_template(
            title=page["title"],
            page_title=f"Week 1 - {page['title']}",
            nav_html=page_nav_html,
            markdown=markdown_text,
            is_overview=False,
            root_prefix=root_prefix,
            heading_renderer_js=HEADING_RENDERER_WEEKS,
        )
        output_path.write_text(html, encoding="utf-8")
        print(f"Generated: {output_path}")


def build_week1(public_dir: Path, plan_weeks: list) -> None:
    """Build Week 1 website: course overview, plan page, week1 pages, and extras."""
    week1_dir = _week_dir(1)
    week1_output_dir = public_dir / "week1"
    week1_output_dir.mkdir(parents=True, exist_ok=True)

    overview, days = load_overview_and_days(week1_dir)

    for day in days:
        day["markdown"] = rewrite_md_links_to_html_weeks(day["markdown"], root_prefix="")

    week1_root_prefix = "../"

    # --- 1. Course overview landing page (public/index.html) ---
    if not COURSE_OVERVIEW_SOURCE.exists():
        raise FileNotFoundError(f"Course overview source not found: {COURSE_OVERVIEW_SOURCE}")
    course_overview = COURSE_OVERVIEW_SOURCE.read_text(encoding="utf-8")
    course_overview = rewrite_md_links_to_html_weeks(course_overview, root_prefix="")
    course_overview = re.sub(r"\]\((?:\.\./)*images/", "](images/", course_overview)

    course_overview_html = page_template(
        title="课程概览",
        page_title="AI Infra 8 周学习计划",
        nav_html=build_week_nav(current_week=None, current_page="overview", weeks=plan_weeks),
        markdown=course_overview,
        is_overview=True,
        heading_renderer_js=HEADING_RENDERER_WEEKS,
    )
    (public_dir / "index.html").write_text(course_overview_html, encoding="utf-8")
    print(f"Generated: {public_dir / 'index.html'}")

    # --- 2. Week 1 overview page (public/week1/index.html) ---
    week1_overview_html_src = rewrite_md_links_to_html_weeks(overview, root_prefix=week1_root_prefix)
    week1_overview_html_src = rewrite_week1_resource_links(week1_overview_html_src, root_prefix=week1_root_prefix)
    week1_overview_with_cards = (
        week1_overview_html_src + '\n\n## 🚀 进入每日学习\n\n' +
        build_day_cards_html(days, root_prefix="")
    )

    week1_overview_html = page_template(
        title="Week 1 概览",
        page_title="Week 1 - Week 1 概览",
        nav_html=build_week_nav(
            current_week=1,
            current_day=None,
            current_is_overview=True,
            root_prefix=week1_root_prefix,
            weeks=plan_weeks,
            current_week_days=days,
            relative_current_week=False,
        ),
        markdown=week1_overview_with_cards,
        is_overview=True,
        root_prefix=week1_root_prefix,
        heading_renderer_js=HEADING_RENDERER_WEEKS,
    )
    (week1_output_dir / "index.html").write_text(week1_overview_html, encoding="utf-8")
    print(f"Generated: {week1_output_dir / 'index.html'}")

    # --- 3. Week 1 day pages (public/week1/dayN.html) ---
    for day in days:
        day["markdown"] = rewrite_md_links_to_html_weeks(day["markdown"], root_prefix=week1_root_prefix)
        day["markdown"] = rewrite_week1_resource_links(day["markdown"], root_prefix=week1_root_prefix)

        has_calc = OCCUPANCY_CALCULATOR_MARKER in day["markdown"]
        extra_scripts = (
            '<script src="../js/occupancy-calculator.js"></script>'
            if has_calc else ""
        )
        html = page_template(
            title=f"Day {day['num']}：{day['title']}",
            page_title=f"Week 1 - Day {day['num']}：{day['title']}",
            nav_html=build_week_nav(
                current_week=1,
                current_day=day["num"],
                root_prefix=week1_root_prefix,
                weeks=plan_weeks,
                current_week_days=days,
                relative_current_week=False,
            ),
            markdown=day["markdown"],
            is_overview=False,
            root_prefix=week1_root_prefix,
            extra_scripts=extra_scripts,
            heading_renderer_js=HEADING_RENDERER_WEEKS,
        )
        filename = f"day{day['num']}.html"
        (week1_output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {week1_output_dir / filename}")

    # --- 4. Plan page (public/plan.html) ---
    build_plan_page(public_dir, plan_weeks)

    # --- 5. Extra markdown pages + source directories ---
    _copy_extra_directories(week1_dir, week1_output_dir)
    _build_extra_pages(week1_dir, week1_output_dir, public_dir, plan_weeks)
