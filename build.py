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

    print("Combined website built successfully in public/")


if __name__ == "__main__":
    main()
