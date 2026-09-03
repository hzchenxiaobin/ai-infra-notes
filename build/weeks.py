"""Unified builder for Week 1-10 websites."""

import re
import shutil
from pathlib import Path
from typing import Optional

from .common import (
    COURSE_OVERVIEW_SOURCE,
    DAILY_DIR,
    PLAN_SOURCE,
    build_day_cards_html,
    compute_root_prefix,
    get_day_info,
    load_overview_and_days,
    rewrite_md_links_to_html_weeks,
    solution_page_template,
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
    9: "分布式并行与多硬件",
    10: "项目整合与面试冲刺",
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
    9: "Week 9 - 分布式并行与多硬件",
    10: "Week 10 - 项目整合与面试冲刺",
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


def _merged_week_titles(weeks: Optional[list]) -> dict:
    """Week number -> title, with plan titles overriding the built-in defaults."""
    week_titles = dict(WEEK_TITLES)
    for week in weeks or []:
        week_titles[week["num"]] = week["title"]
    return week_titles


def _split_week_h1(overview: str, week_num: int, week_titles: dict) -> tuple:
    """Split the leading '# Week N：Title' H1 into (eyebrow, title, body).

    The page template renders eyebrow/title, so the H1 is removed
    from the markdown body to avoid a duplicate heading."""
    match = re.match(r"\s*#\s*(Week\s*\d+)[：:]\s*(.+)", overview)
    if match:
        eyebrow = match.group(1)
        title = match.group(2).strip()
        body = overview[match.end():].lstrip("\n")
        return eyebrow, title, body
    return f"Week {week_num}", week_titles.get(week_num, f"Week {week_num}"), overview


def _day_pills(days: list, current_day=None, overview_active: bool = False, prefix: str = "") -> list:
    """Pill strip items for the week pages: 概览 + one pill per day."""
    pills = [{"label": "📌 概览", "href": f"{prefix}index.html", "active": overview_active}]
    for day in days:
        pills.append({
            "label": f"Day {day['num']}",
            "href": f"{prefix}day{day['num']}.html",
            "active": current_day == day["num"],
        })
    return pills


def _day_prev_next(week_num: int, days: list, index: int, week_titles: dict) -> tuple:
    """(prev_link, next_link) for a day page; each link is (href, label) or None."""
    if index > 0:
        prev_day = days[index - 1]
        prev_link = (f"day{prev_day['num']}.html", f"Day {prev_day['num']}：{prev_day['title']}")
    else:
        prev_link = ("index.html", "本周概览")

    if index + 1 < len(days):
        next_day = days[index + 1]
        next_link = (f"day{next_day['num']}.html", f"Day {next_day['num']}：{next_day['title']}")
    elif _week_dir(week_num + 1).exists():
        next_num = week_num + 1
        next_link = (f"../week{next_num}/index.html", f"Week {next_num}：{week_titles.get(next_num, '')}")
    else:
        next_link = None
    return prev_link, next_link


def rewrite_week1_resource_links(markdown_text: str, root_prefix: str = "") -> str:
    """Rewrite relative resource links so they resolve from pages under week1/."""
    text = re.sub(r"\]\((?:\.\./)?images/", f"]({root_prefix}images/", markdown_text)
    text = text.replace("](../tools/", "](tools/")
    text = text.replace("](../notes/", "](notes/")
    text = re.sub(r"\]\(\.\./day\d+/notes/", "](notes/", text)
    return text


def build_week(week_num: int, public_dir: Path, plan_weeks: list) -> None:
    """Build a single week's website (weeks 2-10). Week 1 is handled by build_week1."""
    week_dir = _week_dir(week_num)
    output_dir = public_dir / f"week{week_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    overview, days = load_overview_and_days(week_dir)

    root_prefix = "../"
    overview = rewrite_md_links_to_html_weeks(overview, root_prefix=root_prefix)
    for day in days:
        day["markdown"] = rewrite_md_links_to_html_weeks(day["markdown"], root_prefix=root_prefix)

    cards = build_day_cards_html(days, root_prefix="")
    week_titles = _merged_week_titles(plan_weeks)
    eyebrow, week_title, overview_body = _split_week_h1(overview, week_num, week_titles)
    if week_num in WEEKS_WITH_CARDS_HEADING:
        overview_with_cards = overview_body + '\n\n## 🚀 进入每日学习\n\n' + cards
    else:
        overview_with_cards = overview_body + '\n\n' + cards

    overview_html = solution_page_template(
        title=week_title,
        eyebrow=eyebrow,
        markdown=overview_with_cards,
        back_link=(f"{root_prefix}index.html", "返回首页"),
        root_prefix=root_prefix,
        page_title=WEEK_OVERVIEW_PAGE_TITLES.get(week_num, f"Week {week_num} 概览"),
        day_pills=_day_pills(days, overview_active=True),
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for index, day in enumerate(days):
        prev_link, next_link = _day_prev_next(week_num, days, index, week_titles)
        html = solution_page_template(
            title=day["title"],
            eyebrow=f"Week {week_num} · Day {day['num']}",
            markdown=day["markdown"],
            back_link=("index.html", f"返回 Week {week_num} 概览"),
            root_prefix=root_prefix,
            page_title=f"Week {week_num} - Day {day['num']}：{day['title']}",
            day_pills=_day_pills(days, current_day=day["num"]),
            prev_link=prev_link,
            next_link=next_link,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")


def build_plan_page(public_dir: Path, plan_weeks: list) -> None:
    """Build the full 10-week plan overview page."""
    if not PLAN_SOURCE.exists():
        print(f"Warning: 10-week plan source not found: {PLAN_SOURCE}")
        return

    markdown_text = PLAN_SOURCE.read_text(encoding="utf-8")
    # Strip the plan's own H1 — the hero header shows the title instead.
    markdown_text = re.sub(r"^\s*#\s+.+\n", "", markdown_text, count=1)

    # Prepend the course overview (aiinfra/daily/README.md): its content moved
    # here when public/index.html became a designed landing page.
    if COURSE_OVERVIEW_SOURCE.exists():
        overview_text = COURSE_OVERVIEW_SOURCE.read_text(encoding="utf-8")
        overview_text = rewrite_md_links_to_html_weeks(overview_text, root_prefix="")
        overview_text = re.sub(r"\]\((?:\.\./)*images/", "](images/", overview_text)
        overview_lines = overview_text.strip().splitlines()
        if overview_lines and overview_lines[0].startswith("# "):
            overview_lines = overview_lines[1:]
        overview_text = "\n".join(overview_lines).strip()
        markdown_text = overview_text + "\n\n---\n\n" + markdown_text

    def add_week_anchor(match: re.Match) -> str:
        return f'<a id="week-{match.group(2)}"></a>\n{match.group(0)}'

    week_heading_pattern = re.compile(r"^(##\s*[^\s]*\s*Week\s*(\d+)[:：].*)$", re.MULTILINE)
    markdown_text = week_heading_pattern.sub(add_week_anchor, markdown_text)

    week_pills = [
        {"label": f"W{num}", "href": f"week{num}/index.html"}
        for num in range(1, 11)
        if _week_dir(num).exists()
    ]

    html = solution_page_template(
        title="AI Infra 10 周学习计划",
        eyebrow="📋 完整学习计划",
        markdown=markdown_text,
        back_link=("index.html", "返回首页"),
        page_title="AI Infra 10 周计划",
        day_pills=week_pills,
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
        markdown_text = source_path.read_text(encoding="utf-8")
        markdown_text = rewrite_md_links_to_html_weeks(markdown_text, root_prefix=root_prefix)
        markdown_text = rewrite_week1_resource_links(markdown_text, root_prefix=root_prefix)

        html = solution_page_template(
            title=page["title"],
            eyebrow="Week 1",
            page_title=f"Week 1 - {page['title']}",
            markdown=markdown_text,
            back_link=("../index.html", "返回 Week 1 概览"),
            root_prefix=root_prefix,
            day_pills=_day_pills(get_day_info(week1_dir), prefix="../"),
        )
        output_path.write_text(html, encoding="utf-8")
        print(f"Generated: {output_path}")


def build_week1(public_dir: Path, plan_weeks: list) -> None:
    """Build Week 1 website: plan page, week1 pages, and extras."""
    week1_dir = _week_dir(1)
    week1_output_dir = public_dir / "week1"
    week1_output_dir.mkdir(parents=True, exist_ok=True)

    overview, days = load_overview_and_days(week1_dir)

    for day in days:
        day["markdown"] = rewrite_md_links_to_html_weeks(day["markdown"], root_prefix="../")

    week1_root_prefix = "../"

    # --- 1. Week 1 overview page (public/week1/index.html) ---
    # (The site landing page public/index.html is built separately by build.home.)
    week1_overview_html_src = rewrite_md_links_to_html_weeks(overview, root_prefix=week1_root_prefix)
    week1_overview_html_src = rewrite_week1_resource_links(week1_overview_html_src, root_prefix=week1_root_prefix)
    week_titles = _merged_week_titles(plan_weeks)
    eyebrow, week_title, overview_body = _split_week_h1(week1_overview_html_src, 1, week_titles)
    week1_overview_with_cards = (
        overview_body + '\n\n## 🚀 进入每日学习\n\n' +
        build_day_cards_html(days, root_prefix="")
    )

    week1_overview_html = solution_page_template(
        title=week_title,
        eyebrow=eyebrow,
        page_title="Week 1 - Week 1 概览",
        markdown=week1_overview_with_cards,
        back_link=(f"{week1_root_prefix}index.html", "返回首页"),
        root_prefix=week1_root_prefix,
        day_pills=_day_pills(days, overview_active=True),
    )
    (week1_output_dir / "index.html").write_text(week1_overview_html, encoding="utf-8")
    print(f"Generated: {week1_output_dir / 'index.html'}")

    # --- 2. Week 1 day pages (public/week1/dayN.html) ---
    for index, day in enumerate(days):
        day["markdown"] = rewrite_md_links_to_html_weeks(day["markdown"], root_prefix=week1_root_prefix)
        day["markdown"] = rewrite_week1_resource_links(day["markdown"], root_prefix=week1_root_prefix)

        has_calc = OCCUPANCY_CALCULATOR_MARKER in day["markdown"]
        extra_scripts = (
            '<script src="../js/occupancy-calculator.js"></script>'
            if has_calc else ""
        )
        prev_link, next_link = _day_prev_next(1, days, index, week_titles)
        html = solution_page_template(
            title=day["title"],
            eyebrow=f"Week 1 · Day {day['num']}",
            page_title=f"Week 1 - Day {day['num']}：{day['title']}",
            markdown=day["markdown"],
            back_link=("index.html", "返回 Week 1 概览"),
            root_prefix=week1_root_prefix,
            day_pills=_day_pills(days, current_day=day["num"]),
            prev_link=prev_link,
            next_link=next_link,
            extra_scripts=extra_scripts,
        )
        filename = f"day{day['num']}.html"
        (week1_output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {week1_output_dir / filename}")

    # --- 3. Plan page (public/plan.html) ---
    build_plan_page(public_dir, plan_weeks)

    # --- 4. Extra markdown pages + source directories ---
    _copy_extra_directories(week1_dir, week1_output_dir)
    _build_extra_pages(week1_dir, week1_output_dir, public_dir, plan_weeks)
