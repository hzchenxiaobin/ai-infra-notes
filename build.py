#!/usr/bin/env python3
"""
Build the combined website for GitHub Pages.
Generates:
  - public/ (deployment root)
    - Shared css/js (copied from static/)
    - Landing page index.html (built by build.home)
    - Plan + week1 pages (built by build.weeks)
    - week2~week10 pages (built by build.weeks)
    - topic websites (built by build.topics)
    - paper reading website (built by build.paper)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from build.common import copy_static_assets, extract_plan_weeks
from build.home import build_home
from build.weeks import WEEK_TITLES, build_week, build_week1
from build.topics import build as build_topics, discover_topics, topic_display
from build.paper import build as build_paper


CHECKS = [
    ("build/check_course.py", "Course consistency (dup-title/stale/dangling)"),
    ("build/lint_md_code.py", "Markdown python code-block indentation"),
]

SKIP_CHECK_FLAG = "--ignore-checks"
SKIP_CHECK_ENV = "AIINFRA_IGNORE_CHECKS"


def checks_should_run() -> bool:
    if SKIP_CHECK_FLAG in sys.argv:
        return False
    return os.environ.get(SKIP_CHECK_ENV, "") not in ("1", "true", "yes")


def run_checks(repo_root: Path) -> None:
    """Run course consistency + md code lint checks. Fail build on findings."""
    if not checks_should_run():
        print("Skipping content checks (--ignore-checks / AIINFRA_IGNORE_CHECKS=1).")
        return

    failed_labels = []
    for rel_script, label in CHECKS:
        script = repo_root / rel_script
        print(f"Running {label}...")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ✗ {label} FAILED (exit {result.returncode})")
            failed_labels.append(label)
        else:
            print(f"  ✓ {label} passed")

    if failed_labels:
        print("\n" + "=" * 60)
        print("❌  构建已中止：以下内容检查未通过")
        for label in failed_labels:
            print(f"   - {label}")
        print("=" * 60)
        print(
            "\n如需本地强制跳过检查，可使用 --ignore-checks "
            "或设置环境变量 AIINFRA_IGNORE_CHECKS=1。"
        )
        sys.exit(1)


def compute_relative_path(from_file: Path, to_path: str) -> str:
    """Compute a relative path from from_file to to_path (relative to site root)."""
    from_dir = from_file.parent
    depth = len(from_dir.parts)
    if depth == 0:
        return to_path
    return "../" * depth + to_path


def insert_extra_nav(html_text: str, html_file: Path, public_dir: Path, topics: list) -> str:
    """Insert extra cross-site links into the sidebar navigation."""
    rel_paper = compute_relative_path(
        html_file.relative_to(public_dir), "paper/index.html"
    )
    lines = [
        '<div class="nav-section-title">更多</div>',
        f'<a class="nav-link" href="{rel_paper}">📄 论文精读</a>',
        '<a class="nav-link" href="https://hzchenxiaobin.github.io/leetcode/">🧩 LeetCode 题解</a>',
        '<a class="nav-link" href="https://github.com/hzchenxiaobin/leetgpu">🎮 LeetGPU 题解</a>',
    ]
    if topics:
        topic_links = []
        for slug in sorted(topics):
            rel = compute_relative_path(html_file.relative_to(public_dir), f"{slug}/index.html")
            display = topic_display(slug)
            topic_links.append(
                f'      <a class="nav-link day-link" href="{rel}">{display} 专题</a>'
            )
        lines.append('<div class="nav-accordion-item">')
        lines.append('  <div class="nav-accordion-header">')
        lines.append('    <span class="nav-link week-link">✨ 专题笔记</span>')
        lines.append('    <button class="nav-accordion-toggle" aria-label="收起/展开专题笔记" aria-expanded="false">▶</button>')
        lines.append('  </div>')
        lines.append('  <div class="nav-accordion-content">')
        lines.append('    <div class="nav-section">')
        lines.extend(topic_links)
        lines.append('    </div>')
        lines.append('  </div>')
        lines.append('</div>')
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

    # 防回归：构建前跑内容一致性检查（标题重复/旧口径/悬空链接 + md 代码缩进）
    run_checks(repo_root)

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
    topics = discover_topics()

    print("Building landing page (index.html)...")
    build_home(public_dir, WEEK_TITLES, topics, topic_display)

    print("Building Week 1 website...")
    build_week1(public_dir, plan_weeks)

    for week_num in range(2, 11):
        print(f"Building Week {week_num} website...")
        build_week(week_num, public_dir, plan_weeks)
        copy_images(
            repo_root / "aiinfra" / "daily" / f"week{week_num}" / "images",
            public_dir / f"week{week_num}" / "images",
        )

    print("Building topic websites...")
    build_topics(public_dir)

    print("Building Paper Reading website...")
    build_paper(public_dir)

    print("Inserting extra navigation links...")
    excluded_parts = set(topics)
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
