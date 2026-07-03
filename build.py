#!/usr/bin/env python3
"""
Build the combined website for GitHub Pages.
Generates:
  - public/ (deployment root)
    - week1 website files (copied from week1/website)
    - week2 website files (copied from week2/website)
    - leetcode website files (copied from leetcode/website)
"""

import shutil
import subprocess
from pathlib import Path


def compute_relative_path(from_file: Path, to_path: str) -> str:
    """Compute a relative path from from_file to to_path (relative to site root)."""
    from_dir = from_file.parent
    depth = len(from_dir.parts)
    if depth == 0:
        return to_path
    return "../" * depth + to_path


def insert_leetcode_nav(html_text: str, html_file: Path, public_dir: Path) -> str:
    """Insert a LeetCode link into the week1/week2 sidebar navigation."""
    rel_path = compute_relative_path(html_file.relative_to(public_dir), "leetcode/index.html")
    leetcode_section = f'''<div class="nav-section-title">更多</div>
<a class="nav-link" href="{rel_path}">🧩 LeetCode 题解</a>
'''
    return html_text.replace(
        "            </nav>\n        </aside>",
        "            </nav>\n" + leetcode_section + "        </aside>",
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
        ["python3", str(repo_root / "week1" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 1 website to public/
    print("Copying Week 1 website to public/...")
    copy_directory_contents(
        repo_root / "week1" / "website",
        public_dir,
        skip={"build.py", "README.md"},
    )

    # Build Week 2 website
    print("Building Week 2 website...")
    subprocess.run(
        ["python3", str(repo_root / "week2" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 2 website to public/week2/
    print("Copying Week 2 website to public/week2/...")
    copy_directory_contents(
        repo_root / "week2" / "website",
        public_dir / "week2",
        skip={"build.py", "README.md"},
    )

    # Build Week 3 website
    print("Building Week 3 website...")
    subprocess.run(
        ["python3", str(repo_root / "week3" / "website" / "build.py")],
        check=True,
    )

    # Copy Week 3 website to public/week3/
    print("Copying Week 3 website to public/week3/...")
    copy_directory_contents(
        repo_root / "week3" / "website",
        public_dir / "week3",
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
        ["python3", str(repo_root / "LeetGPU" / "website" / "build.py")],
        check=True,
    )

    # Copy LeetGPU website to public/LeetGPU/
    print("Copying LeetGPU website to public/LeetGPU/...")
    leetgpu_dst = public_dir / "LeetGPU"
    copy_directory_contents(
        repo_root / "LeetGPU" / "website",
        leetgpu_dst,
        skip={"build.py"},
    )

    # Copy LeetGPU images to public/LeetGPU/images/
    leetgpu_images_src = repo_root / "LeetGPU" / "images"
    leetgpu_images_dst = leetgpu_dst / "images"
    if leetgpu_images_src.exists():
        copy_directory_contents(leetgpu_images_src, leetgpu_images_dst)

    # Insert LeetCode navigation link into Week 1 pages (root + subdirectories, excluding week2/leetcode/LeetGPU)
    week1_pages = [
        p for p in public_dir.rglob("*.html")
        if "leetcode" not in p.relative_to(public_dir).parts
        and "week2" not in p.relative_to(public_dir).parts
        and "LeetGPU" not in p.relative_to(public_dir).parts
    ]
    for html_file in week1_pages:
        if html_file.is_file():
            html_text = html_file.read_text(encoding="utf-8")
            html_text = insert_leetcode_nav(html_text, html_file, public_dir)
            html_file.write_text(html_text, encoding="utf-8")
            print(f"Updated nav: {html_file}")

    print("Combined website built successfully in public/")


if __name__ == "__main__":
    main()
