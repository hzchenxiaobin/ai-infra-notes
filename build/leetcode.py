"""Builder for the LeetCode solution website."""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .common import REPO_ROOT, escape_for_template_string, page_template

LEETCODE_DIR = REPO_ROOT / "leetcode"


def _rewrite_md_links_to_html(markdown_text: str) -> str:
    """Rewrite local .md links to .html for GitHub Pages deployment.

    Solution pages are emitted flat in the problems/ output directory.
    """

    def replace_link(match):
        url = match.group(1)
        if not url.endswith(".md"):
            return match.group(0)
        new_url = url[:-3] + ".html"
        if new_url.endswith("README.html"):
            new_url = new_url[: -len("README.html")] + "index.html"
        new_url = re.sub(r"^(?:\.\./)+(?:daily|contest)/[^/]+/(?:[^/]+/)?", "./", new_url)
        return f"]({new_url})"

    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", replace_link, markdown_text)


def _parse_title(markdown_text: str, filename: str = "") -> str:
    """Extract title from an explicit HTML title comment, then first H1, then filename."""
    match = re.search(r"<!--\s*title:\s*(.+?)\s*-->", markdown_text)
    if match:
        title = match.group(1).strip()
    else:
        match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        title = match.group(1).strip() if match else "题解"

    q_match = re.match(r"^(Q\d+)\.", filename)
    if q_match:
        prefix = q_match.group(1)
        if not re.match(rf"^{prefix}\b", title):
            title = f"{prefix}. {title}"

    return title


def _extract_leetcode_url(markdown_text: str) -> Optional[str]:
    match = re.search(r"https://leetcode\.cn/problems/([^/\s)]+)/?", markdown_text)
    if match:
        return f"https://leetcode.cn/problems/{match.group(1)}/"
    return None


def _build_nav(current_slug: Optional[str], problems: List[Dict], root_prefix: str) -> str:
    """Build sidebar navigation as a three-level accordion."""
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="{root_prefix}leetcode/index.html">📌 题解列表</a>')
    lines.append('<div class="nav-section-title">题目</div>')

    current_path: List[str] = []
    if current_slug is not None:
        for p in problems:
            if p["slug"] == current_slug:
                if p["category"] == "contest":
                    current_path = ["contest", p["contest"]]
                elif p["category"] == "daily":
                    current_path = ["daily", p["week"], p["day"]]
                break

    tree: Dict[str, Dict] = {
        "contest": {"title": "周赛", "children": {}, "problems": []},
        "daily": {"title": "每日一题", "children": {}, "problems": []},
    }

    for p in problems:
        if p["category"] == "contest":
            contest = p["contest"]
            if contest not in tree["contest"]["children"]:
                tree["contest"]["children"][contest] = {
                    "title": contest, "children": {}, "problems": []
                }
            tree["contest"]["children"][contest]["problems"].append(p)
        elif p["category"] == "daily":
            week = p["week"]
            day = p["day"]
            if week not in tree["daily"]["children"]:
                tree["daily"]["children"][week] = {
                    "title": week, "children": {}, "problems": []
                }
            if day not in tree["daily"]["children"][week]["children"]:
                tree["daily"]["children"][week]["children"][day] = {
                    "title": day, "children": {}, "problems": []
                }
            tree["daily"]["children"][week]["children"][day]["problems"].append(p)

    def sort_key_numeric(name: str) -> int:
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else 0

    def render_accordion(node: Dict, path: List[str], level: int) -> List[str]:
        result: List[str] = []
        title = node["title"]

        if len(path) == 3 and path[0] == "daily":
            for p in node.get("problems", []):
                cls = "nav-link active" if current_slug == p["slug"] else "nav-link"
                result.append(
                    f'<a class="{cls}" href="{root_prefix}leetcode/problems/{p["slug"]}.html">'
                    f'<span class="nav-day-tag">{title}</span>'
                    f'{p["title"]}'
                    f'</a>'
                )
            return result

        is_expanded = bool(current_path and current_path[:len(path)] == path)
        expanded_cls = " is-expanded" if is_expanded else ""
        aria_expanded = "true" if is_expanded else "false"
        toggle_icon = "▼" if is_expanded else "▶"
        level_cls = f" level-{level}"

        result.append(f'<div class="nav-accordion-item{level_cls}{expanded_cls}">')
        result.append('  <div class="nav-accordion-header">')
        result.append(
            f'    <span class="nav-link week-link">{title}</span>'
            f'<button class="nav-accordion-toggle" aria-label="收起/展开 {title}" aria-expanded="{aria_expanded}">{toggle_icon}</button>'
        )
        result.append('  </div>')
        result.append('  <div class="nav-accordion-content">')
        result.append('    <div class="nav-section">')

        children = node.get("children", {})
        if children:
            child_items = list(children.items())
            if path == ["contest"]:
                child_items.sort(key=lambda x: sort_key_numeric(x[0]), reverse=True)
            elif path == ["daily"]:
                child_items.sort(key=lambda x: sort_key_numeric(x[0]))
            elif len(path) == 2 and path[0] == "daily":
                child_items.sort(key=lambda x: sort_key_numeric(x[0]))
            else:
                child_items.sort(key=lambda x: x[0])

            for key, child in child_items:
                child_path = path + [key]
                result.extend(render_accordion(child, child_path, level + 1))

        for p in node.get("problems", []):
            cls = "nav-link active" if current_slug == p["slug"] else "nav-link"
            result.append(
                f'<a class="{cls}" href="{root_prefix}leetcode/problems/{p["slug"]}.html">{p["title"]}</a>'
            )

        result.append('    </div>')
        result.append('  </div>')
        result.append('</div>')
        return result

    for key in ["contest", "daily"]:
        if tree[key]["children"] or tree[key]["problems"]:
            lines.extend(render_accordion(tree[key], [key], 1))

    lines.append('<div class="nav-section-title">更多</div>')
    lines.append(f'<a class="nav-link" href="https://hzchenxiaobin.github.io/ai-infra-notes/index.html">📚 AI Infra 学习笔记</a>')
    lines.append(f'<a class="nav-link" href="{root_prefix}leetgpu/index.html">🎮 LeetGPU 题解</a>')

    return "\n".join(lines)


def build(public_dir: Path) -> None:
    """Build the LeetCode website into public_dir/leetcode/."""
    output_dir = public_dir / "leetcode"
    problems_dir = output_dir / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    top_level_images = LEETCODE_DIR / "images"
    if top_level_images.exists():
        shutil.copytree(top_level_images, images_dir, dirs_exist_ok=True)

    md_files = sorted([
        f for f in LEETCODE_DIR.rglob("*.md")
        if f.is_file()
        and "website" not in f.parts
        and "images" not in f.parts
        and f.name != "SKILL.md"
    ])

    for md_file in md_files:
        local_images = md_file.parent / "images"
        if local_images.exists() and local_images.is_dir():
            shutil.copytree(local_images, images_dir, dirs_exist_ok=True)

    problems = []
    seen_slugs = {}
    for md_file in md_files:
        markdown_text = md_file.read_text(encoding="utf-8")
        markdown_text = _rewrite_md_links_to_html(markdown_text)

        title = _parse_title(markdown_text, filename=md_file.name)
        base_slug = md_file.stem
        rel_parts = md_file.relative_to(LEETCODE_DIR).parts
        if rel_parts[0] == "contest" and len(rel_parts) > 1:
            category = "contest"
            contest = rel_parts[1]
            week = None
            day = None
            folder = contest
        elif rel_parts[0] == "daily" and len(rel_parts) > 3:
            category = "daily"
            contest = None
            week = rel_parts[1]
            day = rel_parts[2]
            folder = day
        else:
            category = "other"
            contest = None
            week = None
            day = None
            folder = md_file.parent.name

        slug = base_slug
        if slug in seen_slugs:
            seen_slugs[slug] += 1
            if category == "contest":
                slug = f"{contest}-{base_slug}"
            elif category == "daily":
                slug = f"{week}-{day}-{base_slug}"
            else:
                slug = f"{folder}-{base_slug}"
        else:
            seen_slugs[slug] = 1

        problems.append({
            "slug": slug,
            "title": title,
            "leetcode_url": _extract_leetcode_url(markdown_text),
            "category": category,
            "contest": contest,
            "week": week,
            "day": day,
            "folder": folder,
            "markdown": markdown_text,
        })

    contest_groups: Dict[str, List[Dict]] = {}
    weekly_groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for p in problems:
        if p["category"] == "contest":
            contest_groups.setdefault(p["contest"], []).append(p)
        elif p["category"] == "daily":
            weekly_groups[p["week"]][p["day"]].append(p)

    def sort_key_numeric(name: str) -> int:
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else 0

    daily_markdown = ""
    for week in sorted(weekly_groups.keys(), key=sort_key_numeric, reverse=False):
        daily_markdown += f'<div class="leetcode-section">\n'
        daily_markdown += f'  <div class="leetcode-section-title">第 {sort_key_numeric(week)} 周</div>\n'
        daily_markdown += f'  <div class="leetcode-problem-list">\n'
        for day in sorted(weekly_groups[week].keys(), key=sort_key_numeric):
            for p in weekly_groups[week][day]:
                daily_markdown += (
                    f'    <a class="leetcode-problem-link" href="./problems/{p["slug"]}.html">'
                    f'<span class="leetcode-problem-day">{day}</span>'
                    f'<span class="leetcode-problem-title">{p["title"]}</span>'
                    f'</a>\n'
                )
        daily_markdown += '  </div>\n'
        daily_markdown += '</div>\n\n'

    contest_markdown = ""
    for contest in sorted(contest_groups.keys(), key=sort_key_numeric, reverse=True):
        contest_markdown += f'<div class="leetcode-section">\n'
        contest_markdown += f'  <div class="leetcode-section-title">周赛 {contest}</div>\n'
        contest_markdown += f'  <div class="leetcode-problem-list">\n'
        for p in contest_groups[contest]:
            contest_markdown += (
                f'    <a class="leetcode-problem-link" href="./problems/{p["slug"]}.html">'
                f'{p["title"]}'
                f'</a>\n'
            )
        contest_markdown += '  </div>\n'
        contest_markdown += '</div>\n\n'

    picker_problems = []
    seen_slugs = set()
    for p in problems:
        if p.get("leetcode_url") and p["slug"] not in seen_slugs:
            seen_slugs.add(p["slug"])
            picker_problems.append({"title": p["title"], "url": p["leetcode_url"]})
    problems_json = json.dumps(picker_problems, ensure_ascii=False)

    random_picker_html = f"""<div class="random-pick">
  <button id="random-pick-btn" class="random-btn" data-problems='{problems_json}'>🎲 随机选一道题练习</button>
</div>
<style>
.random-pick {{
  margin: 1rem 0 1.5rem;
  padding: 1rem;
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}}
.random-btn {{
  background: #2563eb;
  color: #fff;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 6px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;
}}
.random-btn:hover {{ background: #1d4ed8; }}
</style>

"""

    overview_markdown = (
        random_picker_html
        + '<div class="leetcode-overview-row">\n'
        '  <div class="leetcode-col leetcode-col-daily">\n'
        f'{daily_markdown}'
        '  </div>\n'
        '  <div class="leetcode-col leetcode-col-contest">\n'
        f'{contest_markdown}'
        '  </div>\n'
        '</div>\n'
    )

    overview_markdown = overview_markdown.replace("](images/", "](./images/")

    root_prefix = "../"
    overview_html = page_template(
        title="LeetCode 题解",
        nav_html=_build_nav(current_slug=None, problems=problems, root_prefix=root_prefix),
        markdown=overview_markdown,
        root_prefix=root_prefix,
        sidebar_title="LeetCode 题解",
        sidebar_title_style="font-size: 1.5rem; margin-bottom: 0;",
        show_back_link=False,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for p in problems:
        problem_markdown = p["markdown"].replace("](images/", "](../images/")
        root_prefix = "../../"
        html = page_template(
            title=p["title"],
            nav_html=_build_nav(current_slug=p["slug"], problems=problems, root_prefix=root_prefix),
            markdown=problem_markdown,
            root_prefix=root_prefix,
            sidebar_title="LeetCode 题解",
            sidebar_title_style="font-size: 1.5rem; margin-bottom: 0;",
            show_back_link=False,
        )
        slug_html = f"{p['slug']}.html"
        (problems_dir / slug_html).write_text(html, encoding="utf-8")
        print(f"Generated: {problems_dir / slug_html}")
