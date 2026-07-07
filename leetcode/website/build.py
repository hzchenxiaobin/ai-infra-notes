#!/usr/bin/env python3
"""
Build the LeetCode solution website from markdown files in leetcode/.
Generates:
  - index.html: problem list page
  - problems/<slug>.html: individual solution pages
Uses relative paths so the site works when deployed under a repository
path prefix (e.g. https://user.github.io/repo-name/leetcode/).

Problems are organized under two top-level directories:
  - leetcode/contest/<contest-number>/
  - leetcode/daily/week<week>/day<day>/
The sidebar and overview page display them grouped by category and folder.
"""

import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    return text


def parse_title(markdown_text: str, filename: str = "") -> str:
    """Extract title from the first level-1 heading.

    If the filename starts with a 'Qx.' prefix and the heading does not,
    prepend the prefix to the heading for sidebar/list display.
    """
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    title = match.group(1).strip() if match else "题解"

    q_match = re.match(r"^(Q\d+)\.", filename)
    if q_match:
        prefix = q_match.group(1)
        if not re.match(rf"^{prefix}\b", title):
            title = f"{prefix}. {title}"

    return title


def build_nav(current_slug: Optional[str], problems: List[Dict], root_prefix: str) -> str:
    """Build sidebar navigation as a three-level accordion.

    Top level: 周赛 / 每日一题
      周赛 -> contest numbers (descending) -> problems
      每日一题 -> weeks (descending) -> days (ascending) -> problems

    current_slug=None means overview page.
    """
    lines = []

    overview_class = "nav-link active" if current_slug is None else "nav-link"
    lines.append(f'<a class="{overview_class}" href="{root_prefix}leetcode/index.html">📌 题解列表</a>')
    lines.append('<div class="nav-section-title">题目</div>')

    # Determine the path to the container of the current problem
    current_path: List[str] = []
    if current_slug is not None:
        for p in problems:
            if p["slug"] == current_slug:
                if p["category"] == "contest":
                    current_path = ["contest", p["contest"]]
                elif p["category"] == "daily":
                    current_path = ["daily", p["week"], p["day"]]
                break

    # Build navigation tree
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
                # Contest numbers descending
                child_items.sort(key=lambda x: sort_key_numeric(x[0]), reverse=True)
            elif path == ["daily"]:
                # Weeks descending
                child_items.sort(key=lambda x: sort_key_numeric(x[0]), reverse=True)
            elif len(path) == 2 and path[0] == "daily":
                # Days ascending
                child_items.sort(key=lambda x: sort_key_numeric(x[0]))
            else:
                child_items.sort(key=lambda x: x[0])

            for key, child in child_items:
                child_path = path + [key]
                child_problems = child.get("problems", [])
                child_children = child.get("children", {})
                # Daily day nodes with a single problem become direct links
                # instead of nested accordions.
                if (
                    len(child_path) == 3
                    and child_path[0] == "daily"
                    and len(child_problems) == 1
                    and not child_children
                ):
                    p = child_problems[0]
                    cls = "nav-link active" if current_slug == p["slug"] else "nav-link"
                    result.append(
                        f'<a class="{cls}" href="{root_prefix}leetcode/problems/{p["slug"]}.html">{child["title"]}</a>'
                    )
                else:
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

    return "\n".join(lines)


def page_template(title: str, nav_html: str, markdown: str, root_prefix: str) -> str:
    escaped_markdown = escape_for_template_string(markdown)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="{root_prefix}css/style.css?v=3">
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
    <script src="{root_prefix}js/main.js?v=4"></script>
</body>
</html>"""


def build_website(leetcode_dir: Path, output_dir: Path) -> None:
    problems_dir = output_dir / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate image assets from leetcode/images/ (shared) and from each
    # markdown-adjacent images/ directory (per-problem) into the output.
    top_level_images = leetcode_dir / "images"
    if top_level_images.exists():
        shutil.copytree(top_level_images, images_dir, dirs_exist_ok=True)

    # Find all markdown files in leetcode/ and its subdirectories
    # (excluding website/, images/ and SKILL.md metadata files)
    md_files = sorted([
        f for f in leetcode_dir.rglob("*.md")
        if f.is_file()
        and "website" not in f.parts
        and "images" not in f.parts
        and f.name != "SKILL.md"
    ])

    # Copy per-problem local images (e.g. leetcode/daily/week1/day1/images/)
    # into the output images/ directory so relative "../images/" references work.
    for md_file in md_files:
        local_images = md_file.parent / "images"
        if local_images.exists() and local_images.is_dir():
            shutil.copytree(local_images, images_dir, dirs_exist_ok=True)

    problems = []
    for md_file in md_files:
        markdown_text = md_file.read_text(encoding="utf-8")

        title = parse_title(markdown_text, filename=md_file.name)
        slug = md_file.stem
        rel_parts = md_file.relative_to(leetcode_dir).parts
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
        problems.append({
            "slug": slug,
            "title": title,
            "category": category,
            "contest": contest,
            "week": week,
            "day": day,
            "folder": folder,
            "markdown": markdown_text,
        })

    # Group problems by category and folder for the overview page
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for p in problems:
        groups.setdefault((p["category"], p["folder"]), []).append(p)

    # Build overview page (at leetcode/index.html -> root_prefix="../")
    overview_markdown = "# LeetCode 题解\n\n> 算法题解题笔记与思路整理。\n\n## 题目列表\n\n"
    for group in sorted(groups.keys(), key=lambda g: (g[0] != "contest", g[0], g[1])):
        category, folder = group
        if category == "contest":
            section_heading = f"周赛 {folder}"
        elif category == "daily":
            section_heading = f"每日一题 {folder}"
        else:
            section_heading = folder
        overview_markdown += f"### {section_heading}\n\n"
        overview_markdown += '<div class="day-cards">\n'
        for p in groups[group]:
            overview_markdown += (
                f'<a class="day-card" href="./problems/{p["slug"]}.html">\n'
                f'  <div class="day-card-number">{folder}</div>\n'
                f'  <div class="day-card-title">{p["title"]}</div>\n'
                f'</a>\n'
            )
        overview_markdown += '</div>\n\n'

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
