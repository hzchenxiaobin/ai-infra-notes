#!/usr/bin/env python3
"""
Build the combined website for GitHub Pages.
Generates:
  - public/ (deployment root)
    - week1 website files (copied from aiinfra/daily/week1/website)
    - week2 website files (copied from aiinfra/daily/week2/website)
    - week3 website files (copied from aiinfra/daily/week3/website)
    - week4 website files (copied from aiinfra/daily/week4/website)
    - week5 website files (copied from aiinfra/daily/week5/website)
    - week6 website files (copied from aiinfra/daily/week6/website)
    - week7 website files (copied from aiinfra/daily/week7/website)
    - week8 website files (copied from aiinfra/daily/week8/website)
    - leetcode website files (copied from leetcode/website)
    - leetgpu website files (copied from leetgpu/website)
    - topic website files (built by aiinfra/topics/build.py and copied from aiinfra/topics/website/)
    - paper reading website files (copied from aiinfra/paper/website)
"""

import shutil
import subprocess
from pathlib import Path

from aiinfra.topics.build import discover_topics, topic_display


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


def copy_directory_contents(src: Path, dst: Path, skip: set = None) -> None:
    """Copy all files and subdirectories from src to dst."""
    if skip is None:
        skip = set()
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)


def main() -> None:
    repo_root = Path(__file__).parent
    public_dir = repo_root / "public"

    # Clean public directory
    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir()

    # Build Week 1 website
    print("Building Week 1 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week1" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 1 website to public/
    print("Copying Week 1 website to public/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week1" / "website",
        public_dir,
        skip={"build.py", "README.md"},
    )

    # Copy course overview images (referenced by index.html) to public/images/
    print("Copying course overview images to public/images/...")
    public_images = public_dir / "images"
    for images_src in [repo_root / "images", repo_root / "aiinfra" / "daily" / "images"]:
        if images_src.exists():
            for item in images_src.iterdir():
                if item.is_file() and item.suffix == ".svg":
                    shutil.copy2(item, public_images / item.name)

    # Build Week 2 website
    print("Building Week 2 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week2" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 2 website to public/week2/
    print("Copying Week 2 website to public/week2/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week2" / "website",
        public_dir / "week2",
        skip={"build.py", "README.md"},
    )

    # Build Week 3 website
    print("Building Week 3 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week3" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 3 website to public/week3/
    print("Copying Week 3 website to public/week3/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week3" / "website",
        public_dir / "week3",
        skip={"build.py", "README.md"},
    )

    # Build Week 4 website
    print("Building Week 4 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week4" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 4 website to public/week4/
    print("Copying Week 4 website to public/week4/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week4" / "website",
        public_dir / "week4",
        skip={"build.py", "README.md"},
    )

    # Build Week 5 website
    print("Building Week 5 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week5" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 5 website to public/week5/
    print("Copying Week 5 website to public/week5/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week5" / "website",
        public_dir / "week5",
        skip={"build.py", "README.md"},
    )

    # Build Week 6 website
    print("Building Week 6 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week6" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 6 website to public/week6/
    print("Copying Week 6 website to public/week6/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week6" / "website",
        public_dir / "week6",
        skip={"build.py", "README.md"},
    )

    # Build Week 7 website
    print("Building Week 7 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week7" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 7 website to public/week7/
    print("Copying Week 7 website to public/week7/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week7" / "website",
        public_dir / "week7",
        skip={"build.py", "README.md"},
    )

    # Build Week 8 website
    print("Building Week 8 website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "daily" / "week8" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 8 website to public/week8/
    print("Copying Week 8 website to public/week8/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "daily" / "week8" / "website",
        public_dir / "week8",
        skip={"build.py", "README.md"},
    )

    # Build LeetCode website
    print("Building LeetCode website...")
    subprocess.run(
        ["python3", str(repo_root / "leetcode" / "website" / "build.py")],
        check=True,
    )

    # Copy LeetCode website to public/leetcode/
    print("Copying LeetCode website to public/leetcode/...")
    leetcode_dst = public_dir / "leetcode"
    copy_directory_contents(
        repo_root / "leetcode" / "website",
        leetcode_dst,
        skip={"build.py"},
    )

    # Copy LeetCode images to public/leetcode/images/
    leetcode_images_src = repo_root / "leetcode" / "images"
    leetcode_images_dst = leetcode_dst / "images"
    if leetcode_images_src.exists():
        copy_directory_contents(leetcode_images_src, leetcode_images_dst)

    # Build LeetGPU website
    print("Building LeetGPU website...")
    subprocess.run(
        ["python3", str(repo_root / "leetgpu" / "website" / "build.py")],
        check=True,
    )

    # Copy LeetGPU website to public/leetgpu/
    print("Copying LeetGPU website to public/leetgpu/...")
    leetgpu_dst = public_dir / "leetgpu"
    copy_directory_contents(
        repo_root / "leetgpu" / "website",
        leetgpu_dst,
        skip={"build.py"},
    )

    # Copy LeetGPU images to public/leetgpu/images/
    leetgpu_images_src = repo_root / "leetgpu" / "images"
    leetgpu_images_dst = leetgpu_dst / "images"
    if leetgpu_images_src.exists():
        copy_directory_contents(leetgpu_images_src, leetgpu_images_dst)

    # Build all topic websites (auto-discovers subdirectories with README.md)
    print("Building topic websites...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "topics" / "build.py")],
        check=True,
    )

    # Copy topic websites to public/<topic>/
    print("Copying topic websites to public/...")
    copy_directory_contents(
        repo_root / "aiinfra" / "topics" / "website",
        public_dir,
    )

    # Build Paper Reading website
    print("Building Paper Reading website...")
    subprocess.run(
        ["python3", str(repo_root / "aiinfra" / "paper" / "website" / "build.py")],
        check=True,
    )

    # Copy Paper Reading website to public/paper/
    print("Copying Paper Reading website to public/paper/...")
    paper_dst = public_dir / "paper"
    copy_directory_contents(
        repo_root / "aiinfra" / "paper" / "website",
        paper_dst,
        skip={"build.py", "README.md"},
    )

    # Copy Paper Reading images to public/paper/images/
    paper_images_src = repo_root / "aiinfra" / "paper" / "images"
    paper_images_dst = paper_dst / "images"
    if paper_images_src.exists():
        copy_directory_contents(paper_images_src, paper_images_dst)

    # Insert extra navigation links into all course pages (aiinfra/daily/week1~week8
    # and extra pages), but not into the leetcode, leetgpu or any topic subsites.
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
