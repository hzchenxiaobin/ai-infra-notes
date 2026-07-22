#!/usr/bin/env python3
"""
Build the combined website for GitHub Pages.
Generates:
  - public/ (deployment root)
    - Shared css/js (copied from static/)
    - Course overview + plan + week1 pages (built by build.weeks)
    - week2~week8 pages (built by build.weeks)
    - leetcode website (built by build.leetcode)
    - leetgpu website (built by build.leetgpu)
    - topic websites (built by build.topics)
    - paper reading website (built by build.paper)
"""

import shutil
from pathlib import Path

from build.common import copy_static_assets, extract_plan_weeks
from build.weeks import build_week, build_week1
from build.leetcode import build as build_leetcode
from build.leetgpu import build as build_leetgpu
from build.topics import build as build_topics, discover_topics, topic_display
from build.paper import build as build_paper


def compute_relative_path(from_file: Path, to_path: str) -> str:
    """Compute a relative path from from_file to to_path (relative to site root)."""
    from_dir = from_file.parent
    depth = len(from_dir.parts)
    if depth == 0:
        return to_path
    return "../" * depth + to_path


def insert_extra_nav(html_text: str, html_file: Path, public_dir: Path, topics: list) -> str:
    """Insert extra cross-site links into the sidebar navigation."""
    rel_leetcode = compute_relative_path(
        html_file.relative_to(public_dir), "leetcode/index.html"
    )
    rel_leetgpu = compute_relative_path(
        html_file.relative_to(public_dir), "leetgpu/index.html"
    )
    rel_paper = compute_relative_path(
        html_file.relative_to(public_dir), "paper/index.html"
    )
    lines = [
        '<div class="nav-section-title">更多</div>',
        f'<a class="nav-link" href="{rel_paper}">📄 论文精读</a>',
        f'<a class="nav-link" href="{rel_leetcode}">🧩 LeetCode 题解</a>',
        f'<a class="nav-link" href="{rel_leetgpu}">🎮 LeetGPU 题解</a>',
    ]
    for slug in sorted(topics):
        rel = compute_relative_path(html_file.relative_to(public_dir), f"{slug}/index.html")
        display = topic_display(slug)
        lines.append(f'<a class="nav-link" href="{rel}">{display} 专题</a>')
    extra_section = "\n".join(lines) + "\n"
    return html_text.replace(
        "            </nav>\n        </aside>",
        "            </nav>\n" + extra_section + "        </aside>",
    )


def copy_images(src: Path, dst: Path) -> None:
    """Copy all image files (svg, png) from src to dst."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file() and item.suffix.lower() in (".svg", ".png"):
            shutil.copy2(item, dst / item.name)
        elif item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)


def main() -> None:
    repo_root = Path(__file__).parent
    public_dir = repo_root / "public"

    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir()

    print("Copying static assets (css/js)...")
    copy_static_assets(public_dir)

    print("Copying course overview images to public/images/...")
    public_images = public_dir / "images"
    copy_images(repo_root / "aiinfra" / "daily" / "week1" / "images", public_images)
    for images_src in [repo_root / "images", repo_root / "aiinfra" / "daily" / "images"]:
        copy_images(images_src, public_images)

    plan_weeks = extract_plan_weeks()

    print("Building Week 1 website...")
    build_week1(public_dir, plan_weeks)

    for week_num in range(2, 9):
        print(f"Building Week {week_num} website...")
        build_week(week_num, public_dir, plan_weeks)
        copy_images(
            repo_root / "aiinfra" / "daily" / f"week{week_num}" / "images",
            public_dir / f"week{week_num}" / "images",
        )

    print("Building LeetCode website...")
    build_leetcode(public_dir)

    print("Building LeetGPU website...")
    build_leetgpu(public_dir)

    print("Building topic websites...")
    build_topics(public_dir)

    print("Building Paper Reading website...")
    build_paper(public_dir)

    print("Inserting extra navigation links...")
    topics = discover_topics()
    excluded_parts = {"leetcode", "leetgpu"} | set(topics)
    course_pages = [
        p for p in public_dir.rglob("*.html")
        if not any(part in excluded_parts for part in p.relative_to(public_dir).parts)
    ]
    for html_file in course_pages:
        if html_file.is_file():
            html_text = html_file.read_text(encoding="utf-8")
            html_text = insert_extra_nav(html_text, html_file, public_dir, topics)
            html_file.write_text(html_text, encoding="utf-8")
            print(f"Updated nav: {html_file}")

    print("Combined website built successfully in public/")


if __name__ == "__main__":
    main()
