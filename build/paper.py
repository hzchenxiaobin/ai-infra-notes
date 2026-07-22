"""Builder for the paper reading website."""

import re
import shutil
from pathlib import Path
from typing import Optional

from .common import REPO_ROOT, paper_page_template

PAPER_DIR = REPO_ROOT / "aiinfra" / "paper"
IMAGES_DIR = PAPER_DIR / "images"


def _rewrite_readme_paths(markdown_text: str) -> str:
    """Rewrite local paths in README.md so they work after deployment."""
    markdown_text = re.sub(
        r"\]\(\.\./SKILL\.md\)",
        "](../index.html)",
        markdown_text,
    )
    return markdown_text


def _extract_title(markdown_text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def _find_papers() -> list:
    """Discover paper subdirectories under aiinfra/paper/."""
    papers = []
    skip = {"images", "website", "__pycache__"}
    for subdir in sorted(PAPER_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in skip:
            continue
        readme = subdir / "README.md"
        pdfs = sorted(subdir.glob("*.pdf"))
        if not readme.exists() and not pdfs:
            continue
        title = subdir.name.replace("_", " ")
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            title = _extract_title(text, title)
            markdown = _rewrite_readme_paths(text)
        else:
            markdown = None
        papers.append({
            "slug": subdir.name,
            "title": title,
            "short_title": title.split("——")[0].strip(),
            "dir": subdir,
            "readme": readme if readme.exists() else None,
            "pdfs": pdfs,
            "markdown": markdown,
        })
    return papers


def _build_nav(papers: list, current_slug: Optional[str], root_prefix: str) -> str:
    """Build sidebar navigation for the paper site."""
    home_active = ' active' if current_slug is None else ''
    lines = [
        f'<a class="nav-link{home_active}" href="{root_prefix}paper/index.html">📌 论文列表</a>',
        '<div class="nav-section-title">论文笔记</div>',
    ]
    for paper in papers:
        active = ' active' if current_slug == paper["slug"] else ''
        if paper["readme"]:
            href = f"{root_prefix}paper/{paper['slug']}/index.html"
        elif paper["pdfs"]:
            href = f"{root_prefix}paper/{paper['slug']}/{paper['pdfs'][0].name}"
        else:
            continue
        lines.append(f'<a class="nav-link{active}" href="{href}">{paper["short_title"]}</a>')
    return "\n".join(lines)


def _build_index_content(papers: list, root_prefix: str) -> str:
    """Generate the markdown/HTML content for the paper list page."""
    lines = [
        "欢迎来到论文精读页面。这里收录了 AI Infra 相关的重要论文笔记与原文 PDF。",
        "",
        "## 论文列表",
        "",
        "| 论文 | 阅读笔记 | 原文 PDF |",
        "| --- | --- | --- |",
    ]
    for paper in papers:
        note_cell = "—"
        if paper["readme"]:
            note_cell = f'[阅读笔记]({root_prefix}paper/{paper["slug"]}/index.html)'
        pdf_cell = "<br>".join(
            f'[{pdf.name}]({root_prefix}paper/{paper["slug"]}/{pdf.name})'
            for pdf in paper["pdfs"]
        ) or "—"
        lines.append(f"| **{paper['title']}** | {note_cell} | {pdf_cell} |")
    return "\n".join(lines)


def build(public_dir: Path) -> None:
    """Build the paper reading website into public_dir/paper/."""
    output_dir = public_dir / "paper"

    papers = _find_papers()
    print(f"Found {len(papers)} papers: {[p['slug'] for p in papers]}")

    website_images = output_dir / "images"
    if IMAGES_DIR.exists():
        website_images.mkdir(parents=True, exist_ok=True)
        for img in IMAGES_DIR.iterdir():
            if img.is_file():
                shutil.copy2(img, website_images / img.name)
        print(f"Copied images to {website_images}")

    root_prefix = "../"
    overview_nav = _build_nav(papers, current_slug=None, root_prefix=root_prefix)
    overview_md = _build_index_content(papers, root_prefix=root_prefix)
    overview_html = paper_page_template(
        title="论文精读",
        page_title="论文精读",
        root_prefix=root_prefix,
        nav=overview_nav,
        markdown=overview_md,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for paper in papers:
        paper_web_dir = output_dir / paper["slug"]
        paper_web_dir.mkdir(parents=True, exist_ok=True)

        for pdf in paper["pdfs"]:
            shutil.copy2(pdf, paper_web_dir / pdf.name)

        if paper["readme"]:
            root_prefix = "../../"
            paper_nav = _build_nav(papers, current_slug=paper["slug"], root_prefix=root_prefix)
            paper_html = paper_page_template(
                title=f"{paper['title']} - 论文精读",
                page_title=paper["title"],
                root_prefix=root_prefix,
                nav=paper_nav,
                markdown=paper["markdown"],
            )
            (paper_web_dir / "index.html").write_text(paper_html, encoding="utf-8")
            print(f"Generated: {paper_web_dir / 'index.html'}")
        else:
            print(f"Skipped note page for {paper['slug']} (no README.md)")
