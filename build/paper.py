"""Builder for the paper reading website."""

import re
import shutil
from pathlib import Path

from .common import REPO_ROOT, week_page_template
from .topics import _split_topic_h1

PAPER_DIR = REPO_ROOT / "aiinfra" / "paper"
IMAGES_DIR = PAPER_DIR / "images"

HEADING_RENDERER_PAPER = """renderer.heading = function(text, level, raw) {
            let anchor = raw.toLowerCase()
                .replace(/[^\\w\\s-]/g, '')
                .replace(/\\s+/g, '-')
                .replace(/-+/g, '-')
                .replace(/^-|-$/g, '');
            if (anchor && level >= 2) {
                return '<h' + level + ' id="' + anchor + '">' + text + '</h' + level + '>';
            }
            return '<h' + level + '>' + text + '</h' + level + '>';
        };"""


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
    overview_md = _build_index_content(papers, root_prefix=root_prefix)
    overview_html = week_page_template(
        title="论文精读",
        eyebrow="📄 Paper Reading",
        markdown=overview_md,
        root_prefix=root_prefix,
        page_title="论文精读 - AI Infra 学习笔记",
        heading_renderer_js=HEADING_RENDERER_PAPER,
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    note_papers = [p for p in papers if p["readme"]]
    for index, paper in enumerate(note_papers):
        paper_web_dir = output_dir / paper["slug"]
        paper_web_dir.mkdir(parents=True, exist_ok=True)

        for pdf in paper["pdfs"]:
            shutil.copy2(pdf, paper_web_dir / pdf.name)

        if index > 0:
            prev_paper = note_papers[index - 1]
            prev_link = (f"../{prev_paper['slug']}/index.html", prev_paper["short_title"])
        else:
            prev_link = ("../index.html", "论文列表")
        if index + 1 < len(note_papers):
            next_paper = note_papers[index + 1]
            next_link = (f"../{next_paper['slug']}/index.html", next_paper["short_title"])
        else:
            next_link = None

        _, paper_body = _split_topic_h1(paper["markdown"], paper["title"])
        paper_html = week_page_template(
            title=paper["title"],
            eyebrow="📄 论文精读",
            markdown=paper_body,
            root_prefix="../../",
            page_title=f"{paper['title']} - 论文精读",
            day_pills=[{"label": "📌 论文列表", "href": "../index.html"}],
            prev_link=prev_link,
            next_link=next_link,
            heading_renderer_js=HEADING_RENDERER_PAPER,
        )
        (paper_web_dir / "index.html").write_text(paper_html, encoding="utf-8")
        print(f"Generated: {paper_web_dir / 'index.html'}")

    for paper in papers:
        if not paper["readme"] and paper["pdfs"]:
            paper_web_dir = output_dir / paper["slug"]
            paper_web_dir.mkdir(parents=True, exist_ok=True)
            for pdf in paper["pdfs"]:
                shutil.copy2(pdf, paper_web_dir / pdf.name)
            print(f"Skipped note page for {paper['slug']} (no README.md)")
