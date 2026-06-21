#!/usr/bin/env python3
"""
Build the combined website for GitHub Pages.
Generates:
  - public/ (deployment root)
    - week1 website files (copied from week1/website)
    - leetcode website files (copied from leetcode/website)
"""

import shutil
import subprocess
from pathlib import Path


def insert_leetcode_nav(html_text: str) -> str:
    """Insert a LeetCode link into the week1 sidebar navigation."""
    leetcode_section = '''<div class="nav-section-title">更多</div>
<a class="nav-link" href="/leetcode/index.html">🧩 LeetCode 题解</a>
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

    # Insert LeetCode navigation link into Week 1 pages
    week1_pages = [public_dir / "index.html"] + list(public_dir.glob("day*.html"))
    for html_file in week1_pages:
        if html_file.is_file():
            html_text = html_file.read_text(encoding="utf-8")
            html_text = insert_leetcode_nav(html_text)
            html_file.write_text(html_text, encoding="utf-8")
            print(f"Updated nav: {html_file}")

    print("Combined website built successfully in public/")


if __name__ == "__main__":
    main()
